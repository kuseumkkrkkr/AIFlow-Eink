"""Select a sparse calibration-only prototype policy on clean-room synthetic writers."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from build_cleanroom_writer_style_bank_v5 import sha256
from evaluate_writer_adaptation_v8 import (
    _aggregate,
    _candidate_masked,
    _frozen_outputs,
    _load_v7,
    _score,
    _split,
    _writer_regressions,
)
from hierarchical_writer_model_v7 import DEFAULT_CHECKPOINT


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BANK = ROOT / "artifacts/cleanroom_writer_style_v5_20260823_smoke32_r4_r2"
DEFAULT_V7 = ROOT / "artifacts/hierarchical_writer_model_v7_20260823_r1"
DEFAULT_OUTPUT = ROOT / "artifacts/writer_adaptation_v9_20260823_r1_shadow"
SUPPORT_SIZES = (4, 6, 10, 20, 22, 24)
SEED = 20260823


@dataclass(frozen=True)
class Policy:
    name: str
    support_threshold: int
    prototype_weight: float


POLICIES = tuple(
    [Policy("identity", 999, 0.0)]
    + [Policy(f"support{threshold}_prototype{str(weight).replace('.', '')}", threshold, weight)
       for threshold in (1, 2, 3) for weight in (0.25, 0.5, 1.0, 2.0)]
)


def _load_bank(path: Path):
    report = json.loads((path / "report.json").read_text(encoding="utf-8"))
    if report["status"] != "STYLE_BANK_SHADOW_CANDIDATE" or not all(report["gates"].values()):
        raise ValueError("clean-room style bank gate is not fully passed")
    with np.load(path / "synthetic_writer_style_bank_v5_r4.npz", allow_pickle=False) as payload:
        return ({key: np.asarray(payload[key]).copy() for key in payload.files}, report)


def _adapt(policy: Policy, cal_embeddings: np.ndarray, cal_logits: np.ndarray, cal_truth: np.ndarray,
           eval_embeddings: np.ndarray, eval_logits: np.ndarray) -> tuple[np.ndarray, dict]:
    baseline_top5 = np.argsort(eval_logits, axis=1)[:, -5:][:, ::-1]
    adapted = eval_logits.copy()
    counts = Counter(int(value) for value in cal_truth.tolist())
    supported = sorted(label for label, count in counts.items() if count >= policy.support_threshold)
    cal_norm = cal_embeddings / np.maximum(np.linalg.norm(cal_embeddings, axis=1, keepdims=True), 1.0e-8)
    self_improved = 0; self_regressed = 0
    per_class_loo = {}
    if policy.prototype_weight > 0 and supported:
        # Calibration-only paired leave-one-out. The class support threshold is
        # measured before holding out the scored calibration glyph.
        for label in supported:
            label_improved = 0; label_regressed = 0
            members = np.flatnonzero(cal_truth == label)
            for target in members.tolist():
                peers = members[members != target]
                if not len(peers):
                    continue
                centroid = cal_norm[peers].mean(axis=0)
                centroid /= max(float(np.linalg.norm(centroid)), 1.0e-8)
                baseline = cal_logits[target]
                candidate = baseline.copy()
                candidate[label] += policy.prototype_weight * float(cal_norm[target] @ centroid)
                top5 = np.argsort(baseline)[-5:][::-1]
                masked = np.full_like(candidate, -1.0e9); masked[top5] = candidate[top5]
                old_correct = int(np.argmax(baseline) == label); new_correct = int(np.argmax(masked) == label)
                label_improved += int(not old_correct and new_correct)
                label_regressed += int(old_correct and not new_correct)
            self_improved += label_improved; self_regressed += label_regressed
            per_class_loo[label] = {"improved": label_improved, "regressed": label_regressed}
    supported = [label for label in supported if per_class_loo.get(label, {}).get("improved", 0) > 0
                 and per_class_loo[label]["regressed"] == 0]
    self_validation_active = bool(supported)
    if policy.prototype_weight > 0 and supported:
        eval_norm = eval_embeddings / np.maximum(np.linalg.norm(eval_embeddings, axis=1, keepdims=True), 1.0e-8)
        for label in supported:
            centroid = cal_norm[cal_truth == label].mean(axis=0)
            centroid /= max(float(np.linalg.norm(centroid)), 1.0e-8)
            adapted[:, label] += policy.prototype_weight * (eval_norm @ centroid)
    return _candidate_masked(adapted, baseline_top5), {
        "calibration_glyphs": len(cal_truth), "class_support": dict(sorted(counts.items())),
        "supported_classes": supported, "identity_fallback": not supported,
        "calibration_loo_improved": self_improved, "calibration_loo_regressed": self_regressed,
        "calibration_loo_per_class": per_class_loo,
        "calibration_self_validation_active": self_validation_active,
    }


def _episode_indices(writer: int, size: int, labels: np.ndarray, writers: np.ndarray, splits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    calibration = np.flatnonzero((writers == writer) & (splits == 0))
    query = np.flatnonzero((writers == writer) & (splits == 1))
    by_label: dict[int, list[int]] = defaultdict(list)
    for index in calibration.tolist():
        by_label[int(labels[index])].append(index)
    ordered_labels = sorted(by_label, key=lambda label: hashlib.sha256(f"{SEED}:{writer}:{size}:{label}".encode()).hexdigest())
    pair_count = max(1, size // 4)
    pair_labels = ordered_labels[:pair_count]
    single_labels = ordered_labels[pair_count:pair_count + size - 2 * pair_count]
    selected = [index for label in pair_labels for index in by_label[label][:2]] + [by_label[label][0] for label in single_labels]
    if len(selected) != size:
        raise AssertionError("sparse calibration episode size mismatch")
    return np.asarray(selected, dtype=np.int64), query


def _array_score(truth: np.ndarray, baseline: np.ndarray, adapted: np.ndarray, formula_size: int = 8) -> dict:
    base_rank = np.argsort(baseline, axis=1)[:, -5:][:, ::-1]
    new_rank = np.argsort(adapted, axis=1)[:, -5:][:, ::-1]
    chunks = [np.arange(start, min(start + formula_size, len(truth))) for start in range(0, len(truth), formula_size)]
    base_correct = base_rank[:, 0] == truth; new_correct = new_rank[:, 0] == truth
    return {
        "glyphs": len(truth), "formulae": len(chunks),
        "baseline_top1_correct": int(base_correct.sum()), "adapted_top1_correct": int(new_correct.sum()),
        "baseline_formula_exact": int(sum(bool(np.all(base_correct[chunk])) for chunk in chunks)),
        "adapted_formula_exact": int(sum(bool(np.all(new_correct[chunk])) for chunk in chunks)),
        "candidate_violations": int(sum(set(base_rank[i].tolist()) != set(new_rank[i].tolist()) for i in range(len(truth)))),
        "paired_improved": int(np.sum(~base_correct & new_correct)), "paired_regressed": int(np.sum(base_correct & ~new_correct)),
    }


def _synthetic_trial(policy: Policy, writer: int, labels: np.ndarray, writers: np.ndarray, splits: np.ndarray,
                     embeddings: np.ndarray, logits: np.ndarray) -> dict:
    totals = Counter(); episode_states = []
    for size in SUPPORT_SIZES:
        cal, query = _episode_indices(writer, size, labels, writers, splits)
        adapted, state = _adapt(policy, embeddings[cal], logits[cal], labels[cal], embeddings[query], logits[query])
        totals.update(_array_score(labels[query], logits[query], adapted))
        episode_states.append({"support_size": size, **state})
    glyphs = totals["glyphs"]; formulae = totals["formulae"]
    return {
        "synthetic_writer_index": writer, "episodes": episode_states,
        "baseline": {"top1": totals["baseline_top1_correct"] / glyphs,
                     "formula_exact": totals["baseline_formula_exact"] / formulae},
        "adapted": {"top1": totals["adapted_top1_correct"] / glyphs,
                    "formula_exact": totals["adapted_formula_exact"] / formulae},
        "glyphs": glyphs, "formulae": formulae, "candidate_violations": totals["candidate_violations"],
        "paired_improved": totals["paired_improved"], "paired_regressed": totals["paired_regressed"],
    }


def _synthetic_aggregate(rows: list[dict]) -> dict:
    glyphs = sum(row["glyphs"] for row in rows); formulae = sum(row["formulae"] for row in rows)
    return {
        "writers": len(rows), "glyphs": glyphs, "formulae": formulae,
        "baseline": {"top1": sum(row["baseline"]["top1"] * row["glyphs"] for row in rows) / glyphs,
                     "formula_exact": sum(row["baseline"]["formula_exact"] * row["formulae"] for row in rows) / formulae},
        "adapted": {"top1": sum(row["adapted"]["top1"] * row["glyphs"] for row in rows) / glyphs,
                    "formula_exact": sum(row["adapted"]["formula_exact"] * row["formulae"] for row in rows) / formulae},
        "candidate_violations": sum(row["candidate_violations"] for row in rows),
        "paired_improved": sum(row["paired_improved"] for row in rows),
        "paired_regressed": sum(row["paired_regressed"] for row in rows),
    }


def _nonregressive(rows: list[dict]) -> bool:
    return all(row["candidate_violations"] == 0 and row["adapted"]["top1"] >= row["baseline"]["top1"]
               and row["adapted"]["formula_exact"] >= row["baseline"]["formula_exact"] for row in rows)


def _positive(summary: dict) -> bool:
    return summary["adapted"]["top1"] > summary["baseline"]["top1"] or summary["adapted"]["formula_exact"] > summary["baseline"]["formula_exact"]


def _legacy_trials(policy: Policy, v7: Path, checkpoint: Path):
    features, metadata = _load_v7(v7)
    labels, embeddings, logits = _frozen_outputs(features, checkpoint)
    split_manifest = {writer: _split(writer, [row["sample_id"] for row in metadata if row["writer_id"] == writer])
                      for writer in sorted({row["writer_id"] for row in metadata})}
    rows = []; states = {}
    for writer, split in split_manifest.items():
        cal_ids = set(split["calibration_formulae"]); eval_ids = set(split["evaluation_formulae"])
        cal = [i for i, row in enumerate(metadata) if row["writer_id"] == writer and row["sample_id"] in cal_ids and row["label_index"] >= 0]
        evaluation = [i for i, row in enumerate(metadata) if row["writer_id"] == writer and row["sample_id"] in eval_ids and row["label_index"] >= 0]
        truth_cal = np.asarray([metadata[i]["label_index"] for i in cal], dtype=np.int64)
        truth_eval = np.asarray([metadata[i]["label_index"] for i in evaluation], dtype=np.int64)
        adapted, state = _adapt(policy, embeddings[cal], logits[cal], truth_cal, embeddings[evaluation], logits[evaluation])
        result = _score(metadata, evaluation, truth_eval, logits[evaluation], adapted)
        base_rank = np.argsort(logits[evaluation], axis=1)[:, -5:][:, ::-1]
        new_rank = np.argsort(adapted, axis=1)[:, -5:][:, ::-1]
        base_correct = base_rank[:, 0] == truth_eval; new_correct = new_rank[:, 0] == truth_eval
        result.update({"writer_id": writer, "calibration_glyphs": len(cal), "evaluation_glyphs": len(evaluation),
                       "paired_improved": int(np.sum(~base_correct & new_correct)),
                       "paired_regressed": int(np.sum(base_correct & ~new_correct))})
        rows.append(result); states[writer] = state
    return labels, split_manifest, rows, states


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--v7", type=Path, default=DEFAULT_V7)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    for path in (args.bank.resolve(), args.v7.resolve(), args.checkpoint.resolve(), args.output.resolve()):
        if path.drive.upper() != "D:": parser.error("all paths must remain on D:")
    if args.output.exists(): parser.error(f"refusing to overwrite output: {args.output}")

    arrays, bank_report = _load_bank(args.bank)
    features = arrays["features"].astype(np.float32); truth = arrays["labels"].astype(np.int64)
    writers = arrays["writer_index"].astype(np.int16); splits = arrays["episode_split"].astype(np.int8)
    checkpoint_labels, embeddings, logits = _frozen_outputs(features, args.checkpoint)
    if max(truth) >= len(checkpoint_labels): raise ValueError("bank label outside frozen checkpoint")
    writer_ids = sorted(set(writers.tolist()))
    ordered_writers = sorted(writer_ids, key=lambda writer: hashlib.sha256(f"{SEED}:writer-split:{writer}".encode()).hexdigest())
    train_writers = ordered_writers[:24]; validation_writers = ordered_writers[24:]

    grid = {}
    for policy in POLICIES:
        rows = [_synthetic_trial(policy, writer, truth, writers, splits, embeddings, logits) for writer in writer_ids]
        train = [row for row in rows if row["synthetic_writer_index"] in train_writers]
        validation = [row for row in rows if row["synthetic_writer_index"] in validation_writers]
        grid[policy.name] = {"policy": asdict(policy), "train_per_writer": train,
                             "train": _synthetic_aggregate(train), "validation_per_writer": validation,
                             "validation": _synthetic_aggregate(validation)}
    eligible = [policy for policy in POLICIES if policy.name != "identity" and _nonregressive(grid[policy.name]["train_per_writer"])
                and _positive(grid[policy.name]["train"])]
    selected = max(eligible, key=lambda policy: (grid[policy.name]["train"]["adapted"]["formula_exact"] - grid[policy.name]["train"]["baseline"]["formula_exact"],
                                                 grid[policy.name]["train"]["adapted"]["top1"] - grid[policy.name]["train"]["baseline"]["top1"])) if eligible else POLICIES[0]
    selected_grid = grid[selected.name]
    synthetic_validation_pass = selected.name != "identity" and _nonregressive(selected_grid["validation_per_writer"]) and _positive(selected_grid["validation"])

    legacy_opened = synthetic_validation_pass
    legacy_rows = []; legacy_states = {}; legacy_splits = {}; legacy_aggregate = None; legacy_regressions = []
    if legacy_opened:
        _labels, legacy_splits, legacy_rows, legacy_states = _legacy_trials(selected, args.v7, args.checkpoint)
        legacy_aggregate = _aggregate(legacy_rows)
        legacy_regressions = _writer_regressions(legacy_rows)
    legacy_pass = bool(legacy_opened and not legacy_regressions and legacy_aggregate["candidate_violations"] == 0
                       and (legacy_aggregate["adapted"]["top1"] > legacy_aggregate["baseline"]["top1"]
                            or legacy_aggregate["adapted"]["formula_exact"] > legacy_aggregate["baseline"]["formula_exact"]))
    status = "WRITER_ADAPTER_V9_LEGACY_GATE_PASS" if legacy_pass else "PRE_ACCEPTANCE_FAIL_CLOSED"

    args.output.mkdir(parents=True)
    split_manifest = {"schema": "aiflow-writer-adaptation-split/v9", "seed": SEED,
                      "synthetic": {"train_writers": train_writers, "validation_writers": validation_writers,
                                    "support_sizes": SUPPORT_SIZES, "query": "all query rows; pseudo-formula chunks of 8"},
                      "legacy": legacy_splits if legacy_opened else "unopened",
                      "known_replay": "unopened", "crohme": "unopened", "mathwriting": "unopened"}
    split_path = args.output / "split_manifest.json"
    split_path.write_text(json.dumps(split_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    state_path = args.output / "adapter_policy.json"
    state_path.write_text(json.dumps({"selected_policy": asdict(selected), "legacy_calibration_states": legacy_states}, ensure_ascii=False, indent=2), encoding="utf-8")
    report = {
        "schema": "aiflow-writer-adaptation/v9", "generated_at": datetime.now(timezone.utc).isoformat(), "status": status,
        "selection": {"source": "clean-room synthetic writer bank only", "grid": grid, "selected": asdict(selected),
                      "synthetic_validation_pass": synthetic_validation_pass},
        "legacy": {"opened_after_synthetic_freeze": legacy_opened, "role": "known development validation; not promotion evidence",
                   "per_writer": legacy_rows, "aggregate": legacy_aggregate, "writer_regressions": legacy_regressions},
        "gates": {"synthetic_writer_disjoint": True, "synthetic_validation_nonregression_positive": synthetic_validation_pass,
                  "legacy_all_writer_top1_formula_nonregression": legacy_opened and not legacy_regressions,
                  "legacy_aggregate_positive": legacy_pass, "candidate_violations_zero": bool(legacy_pass),
                  "known_replay_opened": False, "crohme_opened": False, "mathwriting_opened": False,
                  "checkpoint_changed": False, "hwr_changed": False, "runtime_changed": False,
                  "product_promotion": False, "acceptance_opened": False},
        "inputs": {"bank_report_sha256": sha256(args.bank / "report.json"),
                   "bank_npz_sha256": sha256(args.bank / "synthetic_writer_style_bank_v5_r4.npz"),
                   "checkpoint_sha256": sha256(args.checkpoint), "v7_report_sha256": sha256(args.v7 / "report.json") if legacy_opened else None},
        "outputs": {"split_manifest_sha256": sha256(split_path), "adapter_policy_sha256": sha256(state_path)},
        "contracts": {"fit_inputs": "calibration embeddings and calibration labels only",
                      "evaluation_labels": "scoring only", "candidate_set": "frozen baseline Top-5 exact set",
                      "writer_id_rules": False, "sparse_support_sizes": SUPPORT_SIZES,
                      "promotion_evidence": False, "training_performed_on_hwr": False},
    }
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "status": status, "selected": asdict(selected),
                      "synthetic_validation_pass": synthetic_validation_pass, "legacy_aggregate": legacy_aggregate,
                      "legacy_regressions": legacy_regressions}, ensure_ascii=False))
    return 0 if legacy_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
