#!/usr/bin/env python3
"""Evaluate a fixed, candidate-preserving affine TTA shadow on owned HWR traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from build_normalized_ink_v1 import SourceSample, _canonicalize
from character_tensor_v1 import _json_lines
from evaluate_48hz_prefix_v1 import _load_model, _prefix_tensor, resample_direct_48hz
from train_character_classifier_v1 import apply_input_mode


SCHEMA = "aiflow-hwr-affine-tta-shadow/v1"
VIEWS = ("identity", "x_compress_0.95", "x_expand_1.05", "rotate_m2deg", "rotate_p2deg")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_affine(features: np.ndarray, view: str) -> np.ndarray:
    """Transform normalized x/y only, then restore the model's square-bbox contract."""
    result = np.asarray(features, dtype=np.float32).copy()
    if view == "identity":
        return result
    xy = result[:, :2].astype(np.float64, copy=True) - 0.5
    if view == "x_compress_0.95":
        matrix = np.asarray([[0.95, 0.0], [0.0, 1.0]], dtype=np.float64)
    elif view == "x_expand_1.05":
        matrix = np.asarray([[1.05, 0.0], [0.0, 1.0]], dtype=np.float64)
    elif view in {"rotate_m2deg", "rotate_p2deg"}:
        angle = math.radians(-2.0 if view == "rotate_m2deg" else 2.0)
        matrix = np.asarray(
            [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
            dtype=np.float64,
        )
    else:
        raise ValueError(f"unknown fixed TTA view: {view}")
    transformed = xy @ matrix.T + 0.5
    low, high = transformed.min(axis=0), transformed.max(axis=0)
    extent = max(float(high[0] - low[0]), float(high[1] - low[1]))
    if extent <= 0.0:
        return result
    padding = (1.0 - (high - low) / extent) / 2.0
    result[:, :2] = ((transformed - low) / extent + padding).astype(np.float32)
    if not np.isfinite(result).all() or (result[:, :2] < -1e-6).any() or (result[:, :2] > 1.0 + 1e-6).any():
        raise AssertionError(f"affine view escaped normalized coordinate bounds: {view}")
    return result


def _paired_writer_bootstrap(
    baseline_exact: dict[str, bool], challenger_exact: dict[str, bool], writer_by_id: dict[str, str]
) -> dict[str, Any]:
    by_writer: dict[str, list[float]] = defaultdict(list)
    for sample_id, baseline in baseline_exact.items():
        by_writer[writer_by_id[sample_id]].append(float(challenger_exact[sample_id]) - float(baseline))
    writers = sorted(by_writer)
    if not writers:
        raise AssertionError("no writer clusters for paired bootstrap")
    deltas = np.asarray([np.mean(by_writer[writer]) for writer in writers], dtype=np.float64)
    rng = np.random.default_rng(20261002)
    draws = np.empty(10000, dtype=np.float64)
    for index in range(len(draws)):
        draws[index] = float(rng.choice(deltas, size=len(deltas), replace=True).mean()) * 100.0
    return {
        "method": "paired writer-cluster bootstrap; writers sampled with replacement",
        "iterations": int(len(draws)),
        "seed": 20261002,
        "writer_clusters": len(writers),
        "delta_formula_exact_pp_95_interval": [
            float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))
        ],
        "scope": "consumed-development exploratory shadow only",
    }


def audit(summary_path: Path, trace_path: Path, device_name: str = "cuda") -> dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    traces = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if summary.get("crohme_training_or_tuning") is not False:
        raise ValueError("source run is not explicitly CROHME-excluded")
    inputs = summary["inputs"]
    dataset_root = Path(inputs["dataset_root"])
    formulas_path = dataset_root / "data" / "formulas_valid.jsonl"
    ownership_path = dataset_root / "data" / "ownership_train.jsonl"
    checkpoint_path = Path(inputs["checkpoint"])
    if _sha256(formulas_path) != inputs["formulas_valid_sha256"]:
        raise AssertionError("source formula data hash differs from frozen experiment")
    if _sha256(ownership_path) != inputs["ownership_train_sha256"]:
        raise AssertionError("source ownership data hash differs from frozen experiment")
    if _sha256(checkpoint_path) != inputs["checkpoint_sha256"]:
        raise AssertionError("HWR checkpoint hash differs from frozen experiment")

    trace_by_id = {str(row["sample_id"]): row for row in traces}
    if len(trace_by_id) != len(traces) or len(traces) != int(summary["formulas"]):
        raise AssertionError("trace formula count or ID uniqueness check failed")
    formula_sources = {str(row["sample_id"]): row for row in _json_lines(formulas_path)}
    accepted_annotations = [row for row in _json_lines(ownership_path) if row.get("accepted")]
    annotations = {str(row["sample_id"]): row for row in accepted_annotations}
    expected_ids = set(trace_by_id)
    if not expected_ids <= set(formula_sources) or not expected_ids <= set(annotations):
        raise AssertionError("source formulas/ownership do not cover the traced formula IDs")

    x_values: list[np.ndarray] = []
    targets: list[str] = []
    top5_rows: list[list[str]] = []
    formula_ids: list[str] = []
    writer_by_id: dict[str, str] = {}
    for sample_id in sorted(expected_ids):
        trace = trace_by_id[sample_id]
        annotation = annotations[sample_id]
        symbols = {
            tuple(int(value) for value in symbol["stroke_indices"]): symbol
            for symbol in trace["oracle_group_hwr"]["symbols"]
        }
        groups = annotation["groups"]
        labels = annotation["labels"]
        if len(groups) != len(labels) or len(groups) != len(symbols):
            raise AssertionError(f"group count mismatch for {sample_id}")
        writer_by_id[sample_id] = str(annotation["writer_id"])
        source = formula_sources[sample_id]
        strokes = sorted(source["strokes"], key=lambda stroke: int(stroke["order"]))
        for index, group in enumerate(groups):
            key = tuple(sorted({int(value) for value in groups[index]}))
            symbol = symbols.get(key)
            if symbol is None:
                raise AssertionError(f"ownership group not found in saved trace: {sample_id}:{key}")
            label = str(labels[index])
            if label != str(symbol["target_label"]):
                raise AssertionError(f"ownership/trace label mismatch: {sample_id}:{key}")
            raw_strokes = [
                [
                    (float(point["x"]), float(point["y"]), float(point.get("t_ms", 0.0)))
                    for point in strokes[stroke_index]["points"]
                ]
                for stroke_index in key
            ]
            record = _canonicalize(SourceSample(
                "project_owned_group_candidate",
                f"{sample_id}:{','.join(map(str, key))}",
                label,
                "grouping_candidate",
                "candidate_only",
                raw_strokes,
            ))
            resampled_strokes = resample_direct_48hz(record)
            features = apply_input_mode(_prefix_tensor(resampled_strokes), "uniform-time")
            if features.shape != (128, 5):
                raise AssertionError(f"unexpected HWR input shape: {sample_id}:{key}")
            x_values.append(features)
            targets.append(label)
            top5_rows.append([str(token) for token in symbol["prediction"]["top5"]])
            formula_ids.append(sample_id)

    device = torch.device(device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu")
    model, labels, _ = _load_model(checkpoint_path, device)
    label_index = {label: index for index, label in enumerate(labels)}
    if len(label_index) != 372:
        raise AssertionError("unexpected model vocabulary")
    all_features = np.stack(x_values).astype(np.float32, copy=False)
    start = time.perf_counter()
    logits_by_view: dict[str, np.ndarray] = {}
    with torch.inference_mode():
        for view in VIEWS:
            transformed = np.stack([_canonical_affine(features, view) for features in all_features])
            batch_logits = []
            for begin in range(0, len(transformed), 128):
                batch = torch.from_numpy(transformed[begin:begin + 128]).to(device)
                batch_logits.append(model.math_head(model.encode(batch)).float().cpu().numpy())
            logits_by_view[view] = np.concatenate(batch_logits, axis=0)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - start

    baseline_logits = logits_by_view["identity"]
    replay_top1 = [labels[int(index)] for index in np.argmax(baseline_logits, axis=1)]
    top1_mismatches = [
        index for index, (actual, saved) in enumerate(zip(replay_top1, top5_rows, strict=True))
        if actual != saved[0]
    ]
    if top1_mismatches:
        first = top1_mismatches[0]
        raise AssertionError(
            f"identity Top-1 replay differs from frozen trace at {formula_ids[first]} "
            f"group {index}: {replay_top1[first]} != {top5_rows[first][0]}"
        )
    baseline_top5_indices = np.asarray(
        [[label_index[token] for token in row] for row in top5_rows], dtype=np.int64
    )
    mean_probabilities = np.mean(
        [torch.softmax(torch.from_numpy(logits), dim=1).numpy() for logits in logits_by_view.values()],
        axis=0,
    )
    baseline_tokens = [row[0] for row in top5_rows]
    tta_tokens: list[str] = []
    target_ranks: list[int] = []
    view_top1: dict[str, list[str]] = {view: [] for view in VIEWS}
    for row_index, candidate_indices in enumerate(baseline_top5_indices):
        for view in VIEWS:
            winner = int(np.argmax(logits_by_view[view][row_index]))
            view_top1[view].append(labels[winner])
        winner_index = int(candidate_indices[np.argmax(mean_probabilities[row_index, candidate_indices])])
        tta_tokens.append(labels[winner_index])
    vote_tokens = []
    for row_index, original_top5 in enumerate(top5_rows):
        vote_counts = Counter(
            view_top1[view][row_index]
            for view in VIEWS
            if view_top1[view][row_index] in original_top5
        )
        vote_tokens.append(max(original_top5, key=lambda token: vote_counts[token]))

    baseline_correct_by_formula: dict[str, list[bool]] = defaultdict(list)
    tta_correct_by_formula: dict[str, list[bool]] = defaultdict(list)
    writer_by_formula = writer_by_id
    for sample_id, baseline, challenger, target in zip(formula_ids, baseline_tokens, tta_tokens, targets, strict=True):
        baseline_correct_by_formula[sample_id].append(baseline == target)
        tta_correct_by_formula[sample_id].append(challenger == target)
    baseline_exact = {key: all(values) for key, values in baseline_correct_by_formula.items()}
    tta_exact = {key: all(values) for key, values in tta_correct_by_formula.items()}
    vote_correct_by_formula: dict[str, list[bool]] = defaultdict(list)
    for sample_id, prediction, target in zip(formula_ids, vote_tokens, targets, strict=True):
        vote_correct_by_formula[sample_id].append(prediction == target)
    vote_exact = {key: all(values) for key, values in vote_correct_by_formula.items()}
    changed_tokens = [
        index for index, (before, after) in enumerate(zip(baseline_tokens, tta_tokens, strict=True))
        if before != after
    ]
    changed_formulas = []
    for sample_id in sorted(expected_ids):
        indices = [i for i, value in enumerate(formula_ids) if value == sample_id]
        if any(baseline_tokens[i] != tta_tokens[i] for i in indices):
            changed_formulas.append({
                "sample_id": sample_id,
                "writer_id": writer_by_id[sample_id],
                "target_tokens": [targets[i] for i in indices],
                "baseline_top1": [baseline_tokens[i] for i in indices],
                "tta_top1": [tta_tokens[i] for i in indices],
                "formula_exact_before": baseline_exact[sample_id],
                "formula_exact_after": tta_exact[sample_id],
                "view_top1": {view: [view_top1[view][i] for i in indices] for view in VIEWS},
            })

    token_hits = lambda predictions: sum(prediction == target for prediction, target in zip(predictions, targets, strict=True))
    per_view = {}
    for view, predictions in view_top1.items():
        view_exact = {
            sample_id: all(
                predictions[index] == targets[index]
                for index, formula_id in enumerate(formula_ids)
                if formula_id == sample_id
            )
            for sample_id in expected_ids
        }
        view_transitions = Counter(
            "both_exact" if baseline_exact[sample_id] and view_exact[sample_id]
            else "both_wrong" if not baseline_exact[sample_id] and not view_exact[sample_id]
            else "view_recovery" if view_exact[sample_id]
            else "view_regression"
            for sample_id in expected_ids
        )
        view_recovered = sorted(
            sample_id for sample_id in expected_ids
            if not baseline_exact[sample_id] and view_exact[sample_id]
        )
        view_regressed = sorted(
            sample_id for sample_id in expected_ids
            if baseline_exact[sample_id] and not view_exact[sample_id]
        )
        per_view[view] = {
            "token_hits": token_hits(predictions),
            "token_count": len(targets),
            "formula_exact": sum(view_exact.values()),
            "changed_tokens_vs_identity": sum(
                a != b for a, b in zip(predictions, baseline_tokens, strict=True)
            ),
            "formula_transition_vs_identity": dict(view_transitions),
            "recovered_formula_ids": view_recovered,
            "regressed_formula_ids": view_regressed,
            "paired_writer_cluster_bootstrap_vs_identity": _paired_writer_bootstrap(
                baseline_exact, view_exact, writer_by_id
            ),
        }

    expected_formula_exact = int(summary["oracle_group_top1_token_exact"])
    if expected_formula_exact != sum(baseline_exact.values()):
        raise AssertionError(
            f"identity replay formula exact differs from frozen summary: "
            f"{sum(baseline_exact.values())} != {expected_formula_exact}"
        )
    baseline_coverage = sum(
        targets[i] in [labels[int(index)] for index in baseline_top5_indices[i]]
        for i in range(len(targets))
    )
    baseline_top5_formula_exact = sum(
        all(
            targets[i] in [labels[int(index)] for index in baseline_top5_indices[i]]
            for i, formula_id in enumerate(formula_ids)
            if formula_id == sample_id
        )
        for sample_id in expected_ids
    )
    if baseline_top5_formula_exact != int(summary["oracle_group_top5_oracle"]):
        raise AssertionError(
            f"identity Top-5 formula coverage differs from frozen summary: "
            f"{baseline_top5_formula_exact} != {summary['oracle_group_top5_oracle']}"
        )
    exact_delta = sum(tta_exact.values()) - sum(baseline_exact.values())
    recovered = sorted(key for key in expected_ids if not baseline_exact[key] and tta_exact[key])
    regressed = sorted(key for key in expected_ids if baseline_exact[key] and not tta_exact[key])
    transitions = Counter(
        "both_exact" if baseline_exact[key] and tta_exact[key]
        else "both_wrong" if not baseline_exact[key] and not tta_exact[key]
        else "tta_recovery" if tta_exact[key]
        else "tta_regression"
        for key in expected_ids
    )
    vote_transitions = Counter(
        "both_exact" if baseline_exact[key] and vote_exact[key]
        else "both_wrong" if not baseline_exact[key] and not vote_exact[key]
        else "vote_recovery" if vote_exact[key]
        else "vote_regression"
        for key in expected_ids
    )
    vote_recovered = sorted(key for key in expected_ids if not baseline_exact[key] and vote_exact[key])
    vote_regressed = sorted(key for key in expected_ids if baseline_exact[key] and not vote_exact[key])
    return {
        "schema": SCHEMA,
        "status": "consumed_development_frozen_inference_shadow_only",
        "scope": "gold groups; no training, no thresholds, no CROHME, no product change",
        "inputs": {
            "summary_path": str(summary_path),
            "summary_sha256": _sha256(summary_path),
            "trace_path": str(trace_path),
            "trace_sha256": _sha256(trace_path),
            "dataset_root": str(dataset_root),
            "formulas_valid_sha256": _sha256(formulas_path),
            "ownership_train_sha256": _sha256(ownership_path),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "formula_count": len(expected_ids),
            "glyph_group_count": len(targets),
            "writer_count": len(set(writer_by_id.values())),
            "device": str(device),
        },
        "policy": {
            "fixed_views": list(VIEWS),
            "aggregators": [
                "mean softmax probabilities",
                "plurality of view Top-1 votes, ties resolved by original Top-5 order",
            ],
            "selection": "restrict each final winner to the original identity Top-5",
            "view_fit_or_threshold_search": False,
            "candidate_creation_or_deletion": False,
        },
        "metrics": {
            "identity_replay": {
                "formula_exact": sum(baseline_exact.values()),
                "token_hits": token_hits(baseline_tokens),
                "token_count": len(targets),
                "identity_top5_target_symbol_hits": baseline_coverage,
                "identity_top5_formula_exact": baseline_top5_formula_exact,
                "identity_top1_symbol_mismatches_vs_trace": len(top1_mismatches),
            },
            "fixed_affine_tta": {
                "formula_exact": sum(tta_exact.values()),
                "token_hits": token_hits(tta_tokens),
                "token_count": len(targets),
                "delta_formula_exact": exact_delta,
                "delta_token_hits": token_hits(tta_tokens) - token_hits(baseline_tokens),
                "transition": dict(transitions),
                "recovered_formula_ids": recovered,
                "regressed_formula_ids": regressed,
                "paired_writer_cluster_bootstrap": _paired_writer_bootstrap(
                    baseline_exact, tta_exact, writer_by_formula
                ),
            },
            "fixed_affine_vote": {
                "formula_exact": sum(vote_exact.values()),
                "token_hits": token_hits(vote_tokens),
                "token_count": len(targets),
                "delta_formula_exact": sum(vote_exact.values()) - sum(baseline_exact.values()),
                "delta_token_hits": token_hits(vote_tokens) - token_hits(baseline_tokens),
                "transition": dict(vote_transitions),
                "recovered_formula_ids": vote_recovered,
                "regressed_formula_ids": vote_regressed,
                "paired_writer_cluster_bootstrap": _paired_writer_bootstrap(
                    baseline_exact, vote_exact, writer_by_id
                ),
            },
            "per_view_top1": per_view,
            "changed_token_count": len(changed_tokens),
            "changed_formula_count": len(changed_formulas),
            "inference_seconds_all_views": elapsed_seconds,
            "inference_ms_per_glyph_all_views": elapsed_seconds * 1000.0 / len(targets),
            "timing_device_note": "local research hardware only; not Android latency evidence",
        },
        "changed_formulas": changed_formulas,
        "checks": {
            "source_data_hashes_match_frozen_run": True,
            "checkpoint_hash_matches_frozen_run": True,
            "all_149_formula_ids_replayed": len(expected_ids) == int(summary["formulas"]),
            "identity_replay_matches_frozen_formula_metric": expected_formula_exact == sum(baseline_exact.values()),
            "identity_top1_replay_matches_frozen_trace": len(top1_mismatches) == 0,
            "identity_top5_replay_matches_frozen_formula_metric": (
                baseline_top5_formula_exact == int(summary["oracle_group_top5_oracle"])
            ),
            "tta_output_preserves_identity_top5": True,
            "all_checks_pass": True,
        },
        "product_default_enabled": False,
        "crohme_training_or_tuning": False,
        "promotion_eligible": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    result = audit(args.summary, args.trace, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "identity_formula_exact": result["metrics"]["identity_replay"]["formula_exact"],
        "tta_formula_exact": result["metrics"]["fixed_affine_tta"]["formula_exact"],
        "delta_formula_exact": result["metrics"]["fixed_affine_tta"]["delta_formula_exact"],
        "all_checks_pass": result["checks"]["all_checks_pass"],
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
