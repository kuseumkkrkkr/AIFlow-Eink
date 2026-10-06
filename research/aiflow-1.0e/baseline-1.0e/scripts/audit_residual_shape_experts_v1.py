#!/usr/bin/env python3
"""Audit external-only binary shape experts against residual formula errors."""

from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
import torch

from calibrate_project_punctuation_v1 import _collision_free_eval_indices, _encode
from character_tensor_v1 import ROOT
from evaluate_48hz_prefix_v1 import INPUT_MODE, _load_model, _sha256
from evaluate_joint_hwr_grouping_v1 import _candidate_embeddings, _probabilities
from train_project_owned_grouping_v1 import Sample


SCHEMA = "aiflow-residual-shape-expert-audit/v1"
SEED = 20260820
PAIRS = (
    ("circlearrowright_to_0", r"\circlearrowright", "0"),
    ("mathfrak_a_to_8", r"\mathfrak{A}", "8"),
    ("pi_to_5", r"\pi", "5"),
    ("latin_l_to_open_fence", "L", "("),
    ("q_to_y", "q", "y"),
    ("times_to_x_merged_probe", r"\times", "x"),
)
WIDTHS = (5, 10, 20, 32, 64)
RATIO_FLOORS = (0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2)
DEFAULT_CACHE = ROOT / "artifacts" / "unified_head_20260813" / "unified_math_8ep_full" / "cache"
DEFAULT_HWR = ROOT / "artifacts" / "unified_head_20260814" / "uniform_time_final_all_writers" / "project_symbol_head_checkpoint.pt"
DEFAULT_RUNTIME = Path(r"D:\AIFlow-Workspace\PrivateData\candidate-context-runtime-20260820-r43-layout-shadow-selected.json")
DEFAULT_TRUTH = Path(r"D:\AIFlow-Workspace\PrivateData\math-ink-data-collector\derived\public-candidate-20260819-r3\data\formulas_valid.jsonl")
DEFAULT_OUTPUT = ROOT / "artifacts" / "residual_shape_expert_audit_20260820_r2_shadow" / "residual_shape_expert_audit.json"


def _d_path(path: Path, label: str, *, file: bool = False) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.drive.upper() != "D:" or file and not resolved.is_file():
        raise ValueError(f"{label} must remain on D: {resolved}")
    return resolved


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _split(record_id: str) -> str:
    digest = hashlib.sha256(record_id.encode("utf-8")).digest()
    return "calibration" if digest[0] % 2 == 0 else "holdout"


def _counts(before: np.ndarray, changed: np.ndarray, truth: np.ndarray, target: int) -> dict[str, int]:
    after = before.copy()
    after[changed] = target
    return {
        "records": int(len(truth)),
        "changed": int(changed.sum()),
        "improved": int((changed & (before != truth) & (target == truth)).sum()),
        "regressed": int((changed & (before == truth) & (target != truth)).sum()),
        "non_target_changes": int((changed & (truth != target)).sum()),
        "baseline_hits": int((before == truth).sum()),
        "challenger_hits": int((after == truth).sum()),
    }


def _gate(
    baseline: np.ndarray, truth: np.ndarray, probability: np.ndarray,
    margins: np.ndarray, baseline_index: int, target_index: int,
    indices: np.ndarray, width: int, ratio_floor: float, margin_floor: float,
) -> tuple[np.ndarray, dict[str, int]]:
    target_probability = probability[:, target_index]
    baseline_probability = probability[:, baseline_index]
    rank = 1 + (probability > target_probability[:, None]).sum(axis=1)
    ratio = target_probability / np.maximum(baseline_probability, 1e-12)
    selected = np.zeros(len(truth), dtype=bool)
    selected[indices] = (
        (baseline[indices] == baseline_index)
        & (rank[indices] <= width)
        & (ratio[indices] >= ratio_floor)
        & (margins[indices] >= margin_floor)
    )
    return selected, _counts(
        baseline[indices], selected[indices], truth[indices], target_index,
    )


def _operating_point(
    baseline: np.ndarray, truth: np.ndarray, probability: np.ndarray,
    margins: np.ndarray, baseline_index: int, target_index: int,
    calibration_indices: np.ndarray,
) -> dict[str, Any]:
    target_probability = probability[:, target_index]
    baseline_probability = probability[:, baseline_index]
    rank = 1 + (probability > target_probability[:, None]).sum(axis=1)
    ratio = target_probability / np.maximum(baseline_probability, 1e-12)
    trials = []
    for width in WIDTHS:
        for ratio_floor in RATIO_FLOORS:
            eligible = (
                (baseline == baseline_index)
                & (rank <= width)
                & (ratio >= ratio_floor)
            )
            bad = calibration_indices[
                eligible[calibration_indices]
                & (truth[calibration_indices] != target_index)
            ]
            margin_floor = 0.0 if not len(bad) else max(0.0, float(
                np.nextafter(margins[bad].max(), np.inf)
            ))
            changed, counts = _gate(
                baseline, truth, probability, margins, baseline_index,
                target_index, calibration_indices, width, ratio_floor,
                margin_floor,
            )
            if counts["non_target_changes"]:
                raise AssertionError("calibration precision gate failed")
            trials.append({
                "width": width,
                "ratio_floor": ratio_floor,
                "margin_floor": margin_floor,
                "calibration": counts,
                "changed_indices": np.flatnonzero(changed).tolist(),
            })
    return max(
        trials,
        key=lambda row: (
            row["calibration"]["improved"],
            -row["width"], row["ratio_floor"], row["margin_floor"],
        ),
    )


def _current_groups(
    runtime: dict[str, Any], truth_rows: list[dict[str, Any]], model,
    device: torch.device,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    source = {str(row["sample_id"]): row for row in truth_rows}
    samples = []
    records = []
    for formula in runtime["formulas"]:
        formula_id = str(formula["formula_id"])
        raw = source[formula_id]
        groups = list(formula["groups"])
        symbols = list(formula["symbols"])
        if len(groups) != len(symbols):
            raise ValueError(f"runtime group/symbol mismatch: {formula_id}")
        targets = [str(cell["token"]) for cell in raw.get("target_cells") or []]
        aligned = len(targets) == len(symbols)
        for index, (group, symbol) in enumerate(zip(groups, symbols, strict=True)):
            sample_id = f"{formula_id}::{index}"
            candidate = {"source_indices": [int(value) for value in group["stroke_indices"]]}
            samples.append(Sample(
                sample_id, str(raw.get("writer_id", "")),
                sorted(raw["strokes"], key=lambda row: int(row["order"])),
                tuple(), [candidate], np.zeros((1, 1), dtype=np.float32),
            ))
            records.append({
                "sample_id": sample_id,
                "formula_id": formula_id,
                "index": index,
                "record_id": str(symbol["record_id"]),
                "final": str(symbol["finalized_top1"]),
                "truth": targets[index] if aligned else None,
            })
    embeddings, slices = _candidate_embeddings(samples, model, device)
    if any((value.stop or 0) - (value.start or 0) != 1 for value in slices.values()):
        raise AssertionError("current group embedding coverage mismatch")
    probability = _probabilities(model.math_head, embeddings, device)
    return records, embeddings.numpy(), probability


def _score_current(
    runtime: dict[str, Any], truth_rows: list[dict[str, Any]],
    records: list[dict[str, Any]], changed: np.ndarray, target: str,
) -> dict[str, Any]:
    truth = {
        str(row["sample_id"]): [str(cell["token"]) for cell in row.get("target_cells") or []]
        for row in truth_rows
    }
    before = {str(row["formula_id"]): list(row["finalized_tokens"]) for row in runtime["formulas"]}
    after = {formula_id: list(tokens) for formula_id, tokens in before.items()}
    changes = []
    for position in np.flatnonzero(changed):
        row = records[int(position)]
        after[row["formula_id"]][int(row["index"])] = target
        changes.append({
            "formula_id": row["formula_id"], "record_id": row["record_id"],
            "index": row["index"], "from": row["final"], "to": target,
        })
    ids = sorted(truth)
    improved = [formula_id for formula_id in ids if before[formula_id] != truth[formula_id] and after[formula_id] == truth[formula_id]]
    regressed = [formula_id for formula_id in ids if before[formula_id] == truth[formula_id] and after[formula_id] != truth[formula_id]]
    return {
        "baseline_exact": sum(before[value] == truth[value] for value in ids),
        "challenger_exact": sum(after[value] == truth[value] for value in ids),
        "formulas": len(ids),
        "changes": changes,
        "improved_formulas": improved,
        "regressed_formulas": regressed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--hwr", type=Path, default=DEFAULT_HWR)
    parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--truth", type=Path, default=DEFAULT_TRUTH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    cache = _d_path(args.cache, "cache")
    hwr = _d_path(args.hwr, "HWR checkpoint", file=True)
    runtime_path = _d_path(args.runtime, "runtime", file=True)
    truth_path = _d_path(args.truth, "truth", file=True)
    output = _d_path(args.output, "output")
    if output.exists() or args.batch_size < 1:
        parser.error("output must be new and batch size positive")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")

    manifest_path = cache / "cache_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    labels = list(manifest["config"]["math_labels"])
    label_index = {token: index for index, token in enumerate(labels)}
    if len(labels) != 372 or any(token not in label_index for _, left, right in PAIRS for token in (left, right)):
        raise ValueError("pair labels are outside the 372-class contract")
    train_info = manifest["sets"]["math_train"]
    eval_info = manifest["sets"]["math_eval"]
    train_features = np.load(cache / train_info["features"], mmap_mode="r")
    train_labels = np.load(cache / train_info["labels"], mmap_mode="r")
    eval_features = np.load(cache / eval_info["features"], mmap_mode="r")
    eval_labels = np.load(cache / eval_info["labels"], mmap_mode="r")
    eval_truth_rows = _rows(cache / eval_info["truth"])
    model, checkpoint_labels, _ = _load_model(hwr, device)
    if checkpoint_labels != labels:
        raise ValueError("cache and HWR labels differ")
    clean_indices, collision_audit = _collision_free_eval_indices(train_features, eval_features, INPUT_MODE)
    clean_numpy = clean_indices.numpy()
    clean_features = np.asarray(eval_features[clean_numpy])
    clean_truth = np.asarray(eval_labels[clean_numpy])
    clean_rows = [eval_truth_rows[index] for index in clean_numpy.tolist()]
    clean_embeddings = _encode(model, clean_features, device, INPUT_MODE, batch_size=args.batch_size)
    clean_probability = _probabilities(model.math_head, clean_embeddings, device)
    clean_baseline = clean_probability.argmax(axis=1)
    split = np.asarray([_split(str(row["record_id"])) for row in clean_rows])
    calibration_indices = np.flatnonzero(split == "calibration")
    holdout_indices = np.flatnonzero(split == "holdout")

    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    truth_rows = _rows(truth_path)
    current_records, current_embeddings, current_probability = _current_groups(
        runtime, truth_rows, model, device,
    )
    current_final = np.asarray([row["final"] for row in current_records], dtype=object)
    pair_reports = []
    models: dict[str, LogisticRegression] = {}
    for name, baseline_token, target_token in PAIRS:
        baseline_index = label_index[baseline_token]
        target_index = label_index[target_token]
        pair_train = np.flatnonzero(np.isin(train_labels, [baseline_index, target_index]))
        pair_features = np.asarray(train_features[pair_train])
        pair_target = (np.asarray(train_labels[pair_train]) == target_index).astype(np.int64)
        classifier = LogisticRegression(
            C=0.01, class_weight="balanced", solver="lbfgs", max_iter=2000,
            random_state=SEED,
        ).fit(
            _encode(model, pair_features, device, INPUT_MODE, batch_size=args.batch_size).numpy(),
            pair_target,
        )
        models[name] = classifier
        margins = classifier.decision_function(clean_embeddings.numpy())
        selected = _operating_point(
            clean_baseline, clean_truth, clean_probability, margins,
            baseline_index, target_index, calibration_indices,
        )
        _, holdout = _gate(
            clean_baseline, clean_truth, clean_probability, margins,
            baseline_index, target_index, holdout_indices,
            selected["width"], selected["ratio_floor"], selected["margin_floor"],
        )
        current_margins = classifier.decision_function(current_embeddings)
        target_probability = current_probability[:, target_index]
        baseline_probability = current_probability[:, baseline_index]
        target_rank = 1 + (current_probability > target_probability[:, None]).sum(axis=1)
        target_ratio = target_probability / np.maximum(baseline_probability, 1e-12)
        current_changed = (
            (current_final == baseline_token)
            & (target_rank <= selected["width"])
            & (target_ratio >= selected["ratio_floor"])
            & (current_margins >= selected["margin_floor"])
        )
        current = _score_current(runtime, truth_rows, current_records, current_changed, target_token)
        focus = [
            {
                **row,
                "target_rank": int(target_rank[index]),
                "target_probability_ratio": float(target_ratio[index]),
                "expert_margin": float(current_margins[index]),
                "gate_changed": bool(current_changed[index]),
            }
            for index, row in enumerate(current_records)
            if row["formula_id"] in {
                "aiflow_0023", "aiflow_0032", "aiflow_0036",
                "aiflow_0038", "aiflow_0046", "aiflow_0104",
            } and row["final"] == baseline_token
        ]
        pair_reports.append({
            "name": name,
            "baseline_token": baseline_token,
            "target_token": target_token,
            "training_support": {
                baseline_token: int((pair_target == 0).sum()),
                target_token: int((pair_target == 1).sum()),
            },
            "operating_point": {
                key: value for key, value in selected.items() if key != "changed_indices"
            },
            "untouched_external_holdout": holdout,
            "external_admissible": holdout["changed"] > 0 and holdout["non_target_changes"] == 0,
            "current159_diagnostic": current,
            "residual_focus": focus,
        })

    cross_source = next(row for row in truth_rows if row["sample_id"] == "aiflow_0038")
    cross_sample = Sample(
        "aiflow_0038::all_strokes", str(cross_source.get("writer_id", "")),
        sorted(cross_source["strokes"], key=lambda row: int(row["order"])), tuple(),
        [{"source_indices": list(range(len(cross_source["strokes"]))) }],
        np.zeros((1, 1), dtype=np.float32),
    )
    cross_embedding, _ = _candidate_embeddings([cross_sample], model, device)
    cross_probability = _probabilities(model.math_head, cross_embedding, device)[0]
    cross_classifier = models["times_to_x_merged_probe"]
    cross_target = label_index["x"]
    cross_baseline = label_index[r"\times"]
    cross_probe = {
        "formula_id": "aiflow_0038",
        "posthoc_merged_all_strokes_only": True,
        "hwr_top1": labels[int(cross_probability.argmax())],
        "x_rank": int(1 + (cross_probability > cross_probability[cross_target]).sum()),
        "times_rank": int(1 + (cross_probability > cross_probability[cross_baseline]).sum()),
        "x_over_times_probability_ratio": float(
            cross_probability[cross_target] / max(cross_probability[cross_baseline], 1e-12)
        ),
        "expert_margin_x_over_times": float(
            cross_classifier.decision_function(cross_embedding.numpy())[0]
        ),
        "grouping_change_applied": False,
    }

    payload = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "diagnostic_only_no_runtime_change",
        "protocol": {
            "expert_fit": "external math_train only",
            "gate_calibration": "collision-free external math_eval stable SHA half",
            "gate_acceptance": "untouched collision-free external math_eval complementary SHA half",
            "current159": "evaluation only; no threshold or pair selected from current truth",
        },
        "collision_audit": collision_audit,
        "external_split": {
            "calibration": int(len(calibration_indices)),
            "holdout": int(len(holdout_indices)),
        },
        "pairs": pair_reports,
        "merged_cross_probe": cross_probe,
        "contracts": {
            "external_train_only": True,
            "target_label_used_by_gate": False,
            "writer_identity_used_by_gate": False,
            "arithmetic_evaluation": False,
            "current_grouping_mutations": 0,
            "product_default_enabled": False,
        },
        "sources": {
            "cache_manifest": {"path": str(manifest_path), "sha256": _sha256(manifest_path)},
            "hwr": {"path": str(hwr), "sha256": _sha256(hwr)},
            "runtime": {"path": str(runtime_path), "sha256": _sha256(runtime_path)},
            "truth": {"path": str(truth_path), "sha256": _sha256(truth_path)},
        },
    }
    output.parent.mkdir(parents=True, exist_ok=False)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({
        "event": "residual_shape_expert_audit_complete",
        "output": str(output), "sha256": _sha256(output),
        "pair_decisions": {
            row["name"]: {
                "external_admissible": row["external_admissible"],
                "current_exact": row["current159_diagnostic"]["challenger_exact"],
                "regressions": len(row["current159_diagnostic"]["regressed_formulas"]),
            }
            for row in pair_reports
        },
        "merged_cross_probe": cross_probe,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
