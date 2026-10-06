#!/usr/bin/env python3
"""Train and writer-holdout audit the Math Ink 1.0 stroke grouping selector."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
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


def _candidate_training_rows(samples: Iterable[Sample]) -> tuple[np.ndarray, np.ndarray]:
    features = []; labels = []
    for sample in samples:
        truth = set(sample.truth)
        features.append(sample.features)
        labels.extend(frozenset(int(value) for value in row["source_indices"]) in truth for row in sample.candidates)
    return np.concatenate(features), np.asarray(labels, dtype=np.int64)


def _fit(samples: list[Sample]) -> HistGradientBoostingClassifier:
    features, labels = _candidate_training_rows(samples)
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


def _nested_writer_loo(samples: list[Sample]) -> tuple[dict, dict[str, tuple[frozenset[int], ...]]]:
    writers = sorted({sample.writer for sample in samples})
    outer_predictions = {}; folds = []
    for outer in writers:
        outer_test = [sample for sample in samples if sample.writer == outer]
        outer_train = [sample for sample in samples if sample.writer != outer]
        inner_predictions = {bias: {} for bias in BIAS_GRID}
        for inner in sorted({sample.writer for sample in outer_train}):
            inner_fit = [sample for sample in outer_train if sample.writer != inner]
            inner_test = [sample for sample in outer_train if sample.writer == inner]
            model = _fit(inner_fit); scores = _score_cache(model, inner_test)
            for bias in BIAS_GRID:
                inner_predictions[bias].update(_predict(inner_test, scores, bias))
        trials = [{"group_bias": bias, "metrics": _metrics(outer_train, inner_predictions[bias])} for bias in BIAS_GRID]
        selected = _winner(trials)
        model = _fit(outer_train); scores = _score_cache(model, outer_test)
        prediction = _predict(outer_test, scores, selected["group_bias"])
        outer_predictions.update(prediction)
        folds.append({
            "held_out_writer": outer, "train_formulas": len(outer_train), "test_formulas": len(outer_test),
            "inner_selected_group_bias": selected["group_bias"],
            "test": _metrics(outer_test, prediction),
        })
    return {"metrics": _metrics(samples, outer_predictions), "folds": folds}, outer_predictions


def _deployment_bias(samples: list[Sample]) -> dict:
    writers = sorted({sample.writer for sample in samples})
    by_bias = {bias: {} for bias in BIAS_GRID}
    for writer in writers:
        fit = [sample for sample in samples if sample.writer != writer]
        test = [sample for sample in samples if sample.writer == writer]
        model = _fit(fit); scores = _score_cache(model, test)
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test(); print(json.dumps({"self_test": "pass"})); return 0
    if args.dataset_root is None or args.output is None:
        parser.error("--dataset-root and --output are required")
    dataset_root = args.dataset_root.resolve(); output = args.output.resolve()
    if output.exists() or output.drive.upper() != "D:":
        parser.error("output must be a new directory on D:")
    samples, dataset = _samples(dataset_root)
    oracle = _oracle(samples)
    if oracle["partition_recoverable"] != 1.0:
        raise ValueError(f"grouping lattice does not cover every truth partition: {oracle['missing']}")
    nested, predictions = _nested_writer_loo(samples)
    deployment = _deployment_bias(samples)
    final_model = _fit(samples)
    output.mkdir(parents=True)
    model_path = output / "grouping_selector.joblib"
    joblib.dump({
        "schema": SCHEMA, "model_version": MODEL_VERSION, "model": final_model,
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
        "model_version": MODEL_VERSION, "dataset": dataset,
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
        "schema": SCHEMA, "model_version": MODEL_VERSION, "artifact": str(model_path),
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
