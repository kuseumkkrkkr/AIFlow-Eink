#!/usr/bin/env python3
"""Train and writer-holdout audit the Math Ink 1.0 stroke grouping selector."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from stroke_grouping_v1 import (
    DEFAULT_LATTICE_CONFIG, FEATURE_NAMES, SCHEMA, build_lattice, candidate_features,
    select_partition,
)


SEED = 20260819
BIAS_GRID = tuple(float(value) for value in np.linspace(-3.0, 3.0, 13))
MODEL_VERSION = "aiflow-stroke-grouping-1.0-r3-selected-shadow"


@dataclass
class Sample:
    sample_id: str
    writer: str
    strokes: list[dict]
    truth: tuple[frozenset[int], ...]
    candidates: list[dict]
    features: np.ndarray


AFFINE_AUGMENTATION_SCHEMA = "aiflow-formula-grouping-affine-augmentation/v1"
MAX_AFFINE_AUGMENTATION_VARIANTS = 4


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _samples(dataset_root: Path, lattice_config: dict | None = None) -> tuple[list[Sample], dict]:
    lattice_config = lattice_config or DEFAULT_LATTICE_CONFIG
    formulas_path = dataset_root / "data" / "formulas_valid.jsonl"
    ownership_path = dataset_root / "data" / "ownership_train.jsonl"
    info_path = dataset_root / "dataset_info.json"
    for path in (formulas_path, ownership_path, info_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    formulas = {str(row["sample_id"]): row for row in _rows(formulas_path)}
    annotations = [row for row in _rows(ownership_path) if row.get("accepted")]
    samples = []
    for annotation in annotations:
        sample_id = str(annotation["sample_id"])
        source = formulas.get(sample_id)
        if source is None:
            raise ValueError(f"ownership source missing: {sample_id}")
        strokes = sorted(source["strokes"], key=lambda row: int(row["order"]))
        groups = tuple(frozenset(int(value) for value in group) for group in annotation["groups"])
        assigned = [index for group in groups for index in group]
        if sorted(assigned) != list(range(len(strokes))) or len(assigned) != len(set(assigned)):
            raise ValueError(f"ownership is not an exact stroke partition: {sample_id}")
        labels = [str(value) for value in annotation["labels"]]
        targets = [str(cell["token"]) for cell in source.get("target_cells") or []]
        if labels != targets or len(groups) != len(labels):
            raise ValueError(f"ownership label contract mismatch: {sample_id}")
        candidates = build_lattice(strokes, **lattice_config)
        samples.append(Sample(
            sample_id, str(annotation["writer_id"]), strokes, groups, candidates,
            candidate_features(candidates, strokes),
        ))
    if len({sample.sample_id for sample in samples}) != len(samples) or len({sample.writer for sample in samples}) < 3:
        raise ValueError("grouping corpus requires unique formulas and at least three writers")
    return samples, {
        "dataset_info_sha256": _sha256(info_path),
        "formulas_sha256": _sha256(formulas_path),
        "ownership_sha256": _sha256(ownership_path),
        "formulas": len(samples),
        "writers": len({sample.writer for sample in samples}),
        "writer_formulas": dict(sorted(Counter(sample.writer for sample in samples).items())),
    }


def _affine_spec(sample_id: str, variant: int) -> dict[str, float]:
    """Return a deterministic, mild formula-wide writer-style affine transform."""

    if not 0 <= variant < MAX_AFFINE_AUGMENTATION_VARIANTS:
        raise ValueError("affine augmentation variant is outside the fixed audit set")
    seed = int.from_bytes(hashlib.sha256(
        f"{SEED}:{sample_id}:{AFFINE_AUGMENTATION_SCHEMA}:{variant}".encode()
    ).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    spec = {
        "scale_x": float(rng.uniform(0.94, 1.06)),
        "scale_y": float(rng.uniform(0.94, 1.06)),
        "slant": float(rng.uniform(-0.04, 0.04)),
        "baseline_tilt": float(rng.uniform(-0.025, 0.025)),
    }
    determinant = spec["scale_x"] * spec["scale_y"] - spec["slant"] * spec["baseline_tilt"]
    if determinant <= 0.0:
        raise AssertionError("affine augmentation must preserve orientation")
    return spec


def _affine_transform_strokes(strokes: list[dict], spec: dict[str, float]) -> list[dict]:
    """Apply one shared x/y-only transform while preserving stroke and point channels."""

    point_rows = [point for stroke in strokes for point in stroke.get("points") or []]
    if not point_rows:
        raise ValueError("affine augmentation requires at least one ink point")
    xy = np.asarray([[float(point["x"]), float(point["y"])] for point in point_rows], dtype=np.float64)
    if not np.isfinite(xy).all():
        raise ValueError("affine augmentation input coordinates must be finite")
    center = (xy.min(axis=0) + xy.max(axis=0)) * 0.5
    transformed_strokes: list[dict] = []
    before_non_xy = []
    after_non_xy = []
    for stroke in strokes:
        transformed_stroke = dict(stroke)
        transformed_points = []
        for point in stroke.get("points") or []:
            x, y = float(point["x"]), float(point["y"])
            dx, dy = x - float(center[0]), y - float(center[1])
            updated = dict(point)
            updated["x"] = float(center[0] + spec["scale_x"] * dx + spec["slant"] * dy)
            updated["y"] = float(center[1] + spec["scale_y"] * dy + spec["baseline_tilt"] * dx)
            if not math.isfinite(updated["x"]) or not math.isfinite(updated["y"]):
                raise ValueError("affine augmentation produced non-finite coordinates")
            transformed_points.append(updated)
            before_non_xy.append({key: value for key, value in point.items() if key not in {"x", "y"}})
            after_non_xy.append({key: value for key, value in updated.items() if key not in {"x", "y"}})
        transformed_stroke["points"] = transformed_points
        transformed_strokes.append(transformed_stroke)
    if len(transformed_strokes) != len(strokes):
        raise AssertionError("affine augmentation changed stroke count")
    if [len(row.get("points") or []) for row in transformed_strokes] != [
        len(row.get("points") or []) for row in strokes
    ]:
        raise AssertionError("affine augmentation changed point count")
    if before_non_xy != after_non_xy:
        raise AssertionError("affine augmentation changed timing or sensor channels")
    return transformed_strokes


def _candidate_training_rows(
    samples: Iterable[Sample], *, affine_augmentation_variants: int = 0,
    audit: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build fit rows, augmenting only supplied (therefore fold-training) formulas."""

    if not 0 <= affine_augmentation_variants <= MAX_AFFINE_AUGMENTATION_VARIANTS:
        raise ValueError("affine_augmentation_variants must be between zero and four")
    sample_list = list(samples)
    features = []; labels = []
    stats = {
        "formulas": len(sample_list), "variants_requested": len(sample_list) * affine_augmentation_variants,
        "variants_accepted": 0, "variants_rejected_missing_truth_groups": 0,
        "base_candidate_rows": 0, "augmented_candidate_rows": 0,
        "base_positive_rows": 0, "augmented_positive_rows": 0,
        "candidate_groups_added": 0, "candidate_groups_removed": 0,
        "shared_candidate_pairs": 0, "feature_changed_candidate_rows": 0,
        "feature_changed_positive_rows": 0,
        "mean_absolute_feature_delta": 0.0,
        "mean_absolute_positive_feature_delta": 0.0,
        "per_variant": {},
    }
    feature_delta_sums = {"all": 0.0, "positive": 0.0}
    feature_delta_counts = {"all": 0, "positive": 0}

    def append_rows(matrix: np.ndarray, candidates: list[dict], truth: set[frozenset[int]]) -> int:
        row_labels = [
            frozenset(int(value) for value in row["source_indices"]) in truth
            for row in candidates
        ]
        features.append(matrix)
        labels.extend(row_labels)
        return sum(row_labels)

    for sample in sample_list:
        truth = set(sample.truth)
        stats["base_candidate_rows"] += len(sample.candidates)
        stats["base_positive_rows"] += append_rows(sample.features, sample.candidates, truth)
        for variant in range(affine_augmentation_variants):
            variant_key = str(variant)
            row_stats = stats["per_variant"].setdefault(variant_key, {
                "attempted": 0, "accepted": 0, "rejected_missing_truth_groups": 0,
                "candidate_rows": 0, "positive_rows": 0,
                "candidate_groups_added": 0, "candidate_groups_removed": 0,
                "shared_candidate_pairs": 0, "feature_changed_candidate_rows": 0,
                "feature_changed_positive_rows": 0,
                "mean_absolute_feature_delta": 0.0,
                "mean_absolute_positive_feature_delta": 0.0,
                "_feature_delta_sum": 0.0, "_feature_delta_count": 0,
                "_positive_feature_delta_sum": 0.0, "_positive_feature_delta_count": 0,
            })
            row_stats["attempted"] += 1
            strokes = _affine_transform_strokes(sample.strokes, _affine_spec(sample.sample_id, variant))
            candidates = build_lattice(strokes, **DEFAULT_LATTICE_CONFIG)
            candidate_groups = {
                frozenset(int(value) for value in row["source_indices"])
                for row in candidates
            }
            if not truth.issubset(candidate_groups):
                stats["variants_rejected_missing_truth_groups"] += 1
                row_stats["rejected_missing_truth_groups"] += 1
                continue
            matrix = candidate_features(candidates, strokes)
            if audit is not None:
                base_by_group = {
                    frozenset(int(value) for value in row["source_indices"]): sample.features[index]
                    for index, row in enumerate(sample.candidates)
                }
                augmented_by_group = {
                    frozenset(int(value) for value in row["source_indices"]): matrix[index]
                    for index, row in enumerate(candidates)
                }
                base_groups, augmented_groups = set(base_by_group), set(augmented_by_group)
                added, removed = augmented_groups - base_groups, base_groups - augmented_groups
                stats["candidate_groups_added"] += len(added)
                stats["candidate_groups_removed"] += len(removed)
                row_stats["candidate_groups_added"] += len(added)
                row_stats["candidate_groups_removed"] += len(removed)
                for group in base_groups & augmented_groups:
                    delta = np.abs(base_by_group[group] - augmented_by_group[group])
                    changed = bool(float(delta.max(initial=0.0)) > 1.0e-6)
                    row_stats["shared_candidate_pairs"] += 1
                    row_stats["feature_changed_candidate_rows"] += int(changed)
                    stats["shared_candidate_pairs"] += 1
                    stats["feature_changed_candidate_rows"] += int(changed)
                    row_stats["_feature_delta_sum"] += float(delta.sum())
                    row_stats["_feature_delta_count"] += int(delta.size)
                    feature_delta_sums["all"] += float(delta.sum())
                    feature_delta_counts["all"] += int(delta.size)
                    if group in truth:
                        row_stats["feature_changed_positive_rows"] += int(changed)
                        stats["feature_changed_positive_rows"] += int(changed)
                        row_stats["_positive_feature_delta_sum"] += float(delta.sum())
                        row_stats["_positive_feature_delta_count"] += int(delta.size)
                        feature_delta_sums["positive"] += float(delta.sum())
                        feature_delta_counts["positive"] += int(delta.size)
            positive_rows = append_rows(matrix, candidates, truth)
            stats["variants_accepted"] += 1
            stats["augmented_candidate_rows"] += len(candidates)
            stats["augmented_positive_rows"] += positive_rows
            row_stats["accepted"] += 1
            row_stats["candidate_rows"] += len(candidates)
            row_stats["positive_rows"] += positive_rows
    stats["mean_absolute_feature_delta"] = (
        feature_delta_sums["all"] / max(feature_delta_counts["all"], 1)
    )
    stats["mean_absolute_positive_feature_delta"] = (
        feature_delta_sums["positive"] / max(feature_delta_counts["positive"], 1)
    )
    for row_stats in stats["per_variant"].values():
        row_stats["mean_absolute_feature_delta"] = (
            row_stats["_feature_delta_sum"] / max(row_stats["_feature_delta_count"], 1)
        )
        row_stats["mean_absolute_positive_feature_delta"] = (
            row_stats["_positive_feature_delta_sum"]
            / max(row_stats["_positive_feature_delta_count"], 1)
        )
        for key in (
            "_feature_delta_sum", "_feature_delta_count",
            "_positive_feature_delta_sum", "_positive_feature_delta_count",
        ):
            row_stats.pop(key)
    if not features:
        raise ValueError("grouping fit produced no candidate rows")
    if audit is not None:
        audit.clear()
        audit.update(stats)
    return np.concatenate(features), np.asarray(labels, dtype=np.int64)


def _fit(
    samples: list[Sample], *, affine_augmentation_variants: int = 0,
) -> HistGradientBoostingClassifier:
    features, labels = _candidate_training_rows(
        samples, affine_augmentation_variants=affine_augmentation_variants,
    )
    positives = max(int(labels.sum()), 1); negatives = max(len(labels) - positives, 1)
    weights = np.where(labels == 1, len(labels) / (2 * positives), len(labels) / (2 * negatives))
    return HistGradientBoostingClassifier(
        learning_rate=0.07, max_iter=160, max_leaf_nodes=15, l2_regularization=1.0,
        min_samples_leaf=20, random_state=SEED,
    ).fit(features, labels, sample_weight=weights)


def _score_cache(model: HistGradientBoostingClassifier, samples: Iterable[Sample]) -> dict[str, np.ndarray]:
    output = {}
    for sample in samples:
        probability = model.predict_proba(sample.features)[:, 1]
        output[sample.sample_id] = np.log(np.clip(probability, 1e-6, 1 - 1e-6) / np.clip(1 - probability, 1e-6, 1))
    return output


def _pairs(groups: Iterable[frozenset[int]]) -> set[tuple[int, int]]:
    output = set()
    for group in groups:
        values = sorted(group)
        output.update((first, second) for offset, first in enumerate(values) for second in values[offset + 1:])
    return output


def _metrics(samples: Iterable[Sample], predictions: dict[str, tuple[frozenset[int], ...]]) -> dict:
    exact = group_hit = group_total = pair_tp = pair_fp = pair_fn = overmerge = oversplit = 0
    sample_list = list(samples)
    for sample in sample_list:
        truth = set(sample.truth); predicted = set(predictions[sample.sample_id])
        exact += int(truth == predicted); group_hit += len(truth & predicted); group_total += len(truth)
        truth_pairs = _pairs(truth); predicted_pairs = _pairs(predicted)
        pair_tp += len(truth_pairs & predicted_pairs); pair_fp += len(predicted_pairs - truth_pairs); pair_fn += len(truth_pairs - predicted_pairs)
        overmerge += int(any(sum(bool(group & item) for item in truth) > 1 for group in predicted))
        oversplit += int(any(sum(bool(group & item) for item in predicted) > 1 for group in truth))
    precision = pair_tp / max(pair_tp + pair_fp, 1); recall = pair_tp / max(pair_tp + pair_fn, 1)
    return {
        "formulas": len(sample_list), "partition_exact": exact / max(len(sample_list), 1),
        "exact_partitions": exact, "truth_groups": group_total,
        "exact_group_recall": group_hit / max(group_total, 1),
        "pair_precision": precision, "pair_recall": recall,
        "pair_f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "overmerge_rate": overmerge / max(len(sample_list), 1),
        "oversplit_rate": oversplit / max(len(sample_list), 1),
    }


def _predict(samples: Iterable[Sample], scores: dict[str, np.ndarray], bias: float) -> dict[str, tuple[frozenset[int], ...]]:
    return {
        sample.sample_id: tuple(select_partition(
            sample.candidates, scores[sample.sample_id], len(sample.strokes), group_bias=bias,
        ))
        for sample in samples
    }


def _winner(trials: list[dict]) -> dict:
    return max(trials, key=lambda row: (
        row["metrics"]["partition_exact"], row["metrics"]["pair_f1"],
        row["metrics"]["exact_group_recall"], -abs(row["group_bias"]),
    ))


def _nested_writer_loo(
    samples: list[Sample], *, affine_augmentation_variants: int = 0,
) -> tuple[dict, dict[str, tuple[frozenset[int], ...]]]:
    writers = sorted({sample.writer for sample in samples})
    outer_predictions = {}; folds = []
    for outer in writers:
        outer_test = [sample for sample in samples if sample.writer == outer]
        outer_train = [sample for sample in samples if sample.writer != outer]
        inner_predictions = {bias: {} for bias in BIAS_GRID}
        for inner in sorted({sample.writer for sample in outer_train}):
            inner_fit = [sample for sample in outer_train if sample.writer != inner]
            inner_test = [sample for sample in outer_train if sample.writer == inner]
            model = _fit(inner_fit, affine_augmentation_variants=affine_augmentation_variants)
            scores = _score_cache(model, inner_test)
            for bias in BIAS_GRID:
                inner_predictions[bias].update(_predict(inner_test, scores, bias))
        trials = [{"group_bias": bias, "metrics": _metrics(outer_train, inner_predictions[bias])} for bias in BIAS_GRID]
        selected = _winner(trials)
        model = _fit(outer_train, affine_augmentation_variants=affine_augmentation_variants)
        scores = _score_cache(model, outer_test)
        prediction = _predict(outer_test, scores, selected["group_bias"])
        outer_predictions.update(prediction)
        folds.append({
            "held_out_writer": outer, "train_formulas": len(outer_train), "test_formulas": len(outer_test),
            "inner_selected_group_bias": selected["group_bias"],
            "test": _metrics(outer_test, prediction),
        })
    return {"metrics": _metrics(samples, outer_predictions), "folds": folds}, outer_predictions


def _deployment_bias(samples: list[Sample], *, affine_augmentation_variants: int = 0) -> dict:
    writers = sorted({sample.writer for sample in samples})
    by_bias = {bias: {} for bias in BIAS_GRID}
    for writer in writers:
        fit = [sample for sample in samples if sample.writer != writer]
        test = [sample for sample in samples if sample.writer == writer]
        model = _fit(fit, affine_augmentation_variants=affine_augmentation_variants)
        scores = _score_cache(model, test)
        for bias in BIAS_GRID:
            by_bias[bias].update(_predict(test, scores, bias))
    trials = [{"group_bias": bias, "metrics": _metrics(samples, by_bias[bias])} for bias in BIAS_GRID]
    return {"winner": _winner(trials), "trials": trials}


def _oracle(samples: list[Sample]) -> dict:
    matched = exact = total = candidates = 0; missing = []
    for sample in samples:
        available = {frozenset(int(value) for value in row["source_indices"]) for row in sample.candidates}
        absent = [sorted(group) for group in sample.truth if group not in available]
        total += len(sample.truth); matched += len(sample.truth) - len(absent); exact += int(not absent); candidates += len(available)
        if absent:
            missing.append({"sample_id": sample.sample_id, "groups": absent})
    return {
        "formulas": len(samples), "truth_groups": total, "group_recall": matched / max(total, 1),
        "recoverable_partitions": exact, "partition_recoverable": exact / max(len(samples), 1),
        "mean_candidates": candidates / max(len(samples), 1), "missing": missing,
    }


def _self_test() -> None:
    from stroke_grouping_v1 import _self_test as grouping_self_test
    grouping_self_test()
    strokes = [
        {"order": 0, "points": [
            {"x": 0.0, "y": 0.0, "t_ms": 0.0, "pressure": 0.3},
            {"x": 2.0, "y": 1.0, "t_ms": 4.0, "pressure": 0.6},
        ]},
        {"order": 1, "points": [
            {"x": 3.0, "y": 2.0, "t_ms": 8.0, "pressure": 0.4},
        ]},
    ]
    original = json.loads(json.dumps(strokes))
    augmented = _affine_transform_strokes(strokes, _affine_spec("self-test", 0))
    assert len(augmented) == len(strokes)
    assert [len(row["points"]) for row in augmented] == [2, 1]
    assert augmented != strokes and strokes == original
    assert [
        [{key: value for key, value in point.items() if key not in {"x", "y"}}
         for point in row["points"]]
        for row in augmented
    ] == [
        [{key: value for key, value in point.items() if key not in {"x", "y"}}
         for point in row["points"]]
        for row in strokes
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--affine-augmentation-variants", type=int, default=0, choices=range(5))
    parser.add_argument("--audit-affine-augmentation", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test(); print(json.dumps({"self_test": "pass"})); return 0
    if args.audit_affine_augmentation:
        if args.dataset_root is None or args.output is not None:
            parser.error("augmentation audit requires --dataset-root and forbids --output")
        if args.affine_augmentation_variants < 1:
            parser.error("augmentation audit requires at least one affine variant")
        dataset_root = args.dataset_root.resolve()
        samples, dataset = _samples(dataset_root)
        audit = {}
        _candidate_training_rows(
            samples, affine_augmentation_variants=args.affine_augmentation_variants,
            audit=audit,
        )
        print(json.dumps({
            "schema": "aiflow-grouping-affine-augmentation-audit/v1",
            "status": "data_only_audit",
            "training_performed": False,
            "crohme_rows": 0,
            "mathwriting_rows": 0,
            "source": {
                "formulas": dataset["formulas"], "writers": len(dataset["writer_formulas"]),
                "dataset_info_sha256": dataset["dataset_info_sha256"],
                "formulas_sha256": dataset["formulas_sha256"],
                "ownership_sha256": dataset["ownership_sha256"],
            },
            "augmentation": {
                "schema": AFFINE_AUGMENTATION_SCHEMA,
                "variants_per_formula": args.affine_augmentation_variants,
                "application": "formula-wide; modifies x/y only; source stroke groups and holdout rows remain unchanged",
                "scale_x_range": [0.94, 1.06], "scale_y_range": [0.94, 1.06],
                "slant_range": [-0.04, 0.04], "baseline_tilt_range": [-0.025, 0.025],
                "audit": audit,
            },
        }, ensure_ascii=False, indent=2))
        return 0
    if args.dataset_root is None or args.output is None:
        parser.error("--dataset-root and --output are required")
    dataset_root = args.dataset_root.resolve(); output = args.output.resolve()
    if output.exists() or output.drive.upper() != "D:":
        parser.error("output must be a new directory on D:")
    samples, dataset = _samples(dataset_root)
    model_version = (
        MODEL_VERSION if args.affine_augmentation_variants == 0
        else f"{MODEL_VERSION}-affine{args.affine_augmentation_variants}"
    )
    oracle = _oracle(samples)
    if oracle["partition_recoverable"] != 1.0:
        raise ValueError(f"grouping lattice does not cover every truth partition: {oracle['missing']}")
    affine_audit = {}
    if args.affine_augmentation_variants:
        _candidate_training_rows(
            samples, affine_augmentation_variants=args.affine_augmentation_variants,
            audit=affine_audit,
        )
    nested, predictions = _nested_writer_loo(
        samples, affine_augmentation_variants=args.affine_augmentation_variants,
    )
    deployment = _deployment_bias(samples, affine_augmentation_variants=args.affine_augmentation_variants)
    final_model = _fit(samples, affine_augmentation_variants=args.affine_augmentation_variants)
    output.mkdir(parents=True)
    model_path = output / "grouping_selector.joblib"
    joblib.dump({
        "schema": SCHEMA, "model_version": model_version, "model": final_model,
        "feature_names": FEATURE_NAMES, "lattice_config": DEFAULT_LATTICE_CONFIG,
        "group_bias": deployment["winner"]["group_bias"], "dataset": dataset,
        "training_rights": "project-owned accepted ownership; commercial consent contract",
    }, model_path, compress=3)
    error_rows = []
    for sample in samples:
        predicted = predictions[sample.sample_id]
        if set(predicted) != set(sample.truth):
            error_rows.append({
                "sample_id": sample.sample_id, "writer": sample.writer,
                "truth": [sorted(group) for group in sample.truth],
                "predicted": [sorted(group) for group in predicted],
            })
    summary = {
        "schema": "aiflow-project-owned-grouping-training/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(), "status": "shadow",
        "model_version": model_version, "dataset": dataset,
        "lattice": {"config": DEFAULT_LATTICE_CONFIG, "oracle": oracle},
        "evaluation": {
            "protocol": "nested leave-one-writer-out; outer writer is absent from model fit and bias selection",
            "nested_writer_loo": nested,
            "deployment_bias_selection": deployment,
            "failed_formulas": error_rows,
        },
        "artifact": {
            "file": model_path.name, "sha256": _sha256(model_path),
            "feature_names": list(FEATURE_NAMES), "group_bias": deployment["winner"]["group_bias"],
        },
        "affine_augmentation": {
            "schema": AFFINE_AUGMENTATION_SCHEMA,
            "variants_per_fit_formula": args.affine_augmentation_variants,
            "training_fold_only": True,
            "audit": affine_audit if args.affine_augmentation_variants else None,
        },
        "contracts": {
            "label_features": False, "writer_features": False, "target_cell_count_input": False,
            "all_strokes_exactly_once": True, "insertion_or_deletion": False,
            "product_default_enabled": False,
        },
    }
    (output / "grouping_training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n",
    )
    runtime = {
        "schema": SCHEMA, "model_version": model_version, "artifact": str(model_path),
        "sha256": summary["artifact"]["sha256"], "feature_names": list(FEATURE_NAMES),
        "lattice_config": DEFAULT_LATTICE_CONFIG, "group_bias": deployment["winner"]["group_bias"],
        "product_default_enabled": False,
    }
    (output / "grouping_runtime_config.json").write_text(
        json.dumps(runtime, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n",
    )
    print(json.dumps({
        "output": str(output), "oracle": oracle,
        "nested_writer_loo": nested["metrics"],
        "deployment_bias": deployment["winner"], "model_sha256": summary["artifact"]["sha256"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
