#!/usr/bin/env python3
"""Audit UCI duplicates, HWRT support, and BDSHWA character segmentation.

This script never trains a model and never mutates source archives. It writes
auditable reports and a before/after UCI plot for the character-classifier
admission decision.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import statistics
import zipfile
from collections import Counter
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from build_normalized_ink_v1 import DEFAULT_BDSHWA_ARCHIVE, _bdshwa_csv_archive


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANONICAL_ROOT = ROOT / "datasets" / "normalized" / "v1"
DEFAULT_OUTPUT = ROOT / "research" / "character_classifier_data_audit_20260812"


def _json_lines(path: Path) -> Iterable[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def deduplicate_exact_consecutive_xy(points: list[tuple[float, float]]) -> tuple[list[tuple[float, float]], int]:
    """Remove only adjacent exact XY duplicates; return-to-point loops remain."""
    output: list[tuple[float, float]] = []
    removed = 0
    for point in points:
        if output and point == output[-1]:
            removed += 1
        else:
            output.append(point)
    return output, removed


def _uci_raw_samples() -> list[tuple[int, str, list[tuple[float, float]], int]]:
    path = ROOT / "datasets" / "10_approved_external" / "uci_character_trajectories" / "raw" / "character+trajectories.official.zip"
    with zipfile.ZipFile(path) as archive:
        data = loadmat(io.BytesIO(archive.read("mixoutALL_shifted.mat")), squeeze_me=True, struct_as_record=False)
    labels = data["consts"].key
    class_ids = data["consts"].charlabels
    output = []
    for index, matrix in enumerate(data["mixout"]):
        points = [(float(matrix[0, point]), float(matrix[1, point])) for point in range(matrix.shape[1])]
        output.append((index, str(labels[int(class_ids[index]) - 1]), points, deduplicate_exact_consecutive_xy(points)[1]))
    return output


def _plot_uci(samples: list[tuple[int, str, list[tuple[float, float]], int]], output: Path) -> dict:
    selected = sorted((row for row in samples if row[3]), key=lambda row: (-row[3], row[0]))[:6]
    if not selected:
        raise AssertionError("UCI duplicate audit found no duplicate samples")
    figure, axes = plt.subplots(2, len(selected), figsize=(3.2 * len(selected), 6), constrained_layout=True)
    for column, (index, label, points, removed) in enumerate(selected):
        before = np.asarray(points, dtype=float)
        after = np.asarray(deduplicate_exact_consecutive_xy(points)[0], dtype=float)
        for axis, values, title in ((axes[0, column], before, "before"), (axes[1, column], after, "after")):
            axis.plot(values[:, 0], values[:, 1], color="#1f2937", linewidth=1.1)
            axis.scatter(values[:, 0], values[:, 1], s=6, color="#2563eb")
            axis.set_aspect("equal", adjustable="datalim")
            axis.invert_yaxis()
            axis.set_xticks([]); axis.set_yticks([])
            axis.set_title(f"{title}: #{index} '{label}'\nremoved={removed}", fontsize=9)
    figure.suptitle("UCI exact consecutive XY duplicates only", fontsize=13)
    path = output / "uci_exact_duplicate_before_after.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return {"file": path.name, "selected_samples": [row[0] for row in selected]}


def _canonical_uci_dedup(canonical_root: Path) -> dict:
    removed = remaining = records = 0
    for row in _json_lines(canonical_root / "uci.jsonl.gz"):
        records += 1
        removed += int(row["normalization"]["removed_consecutive_exact_xy_points"])
        for stroke in row["strokes"]:
            points = stroke["points"]
            remaining += sum(left[:2] == right[:2] for left, right in zip(points, points[1:]))
    return {"records": records, "canonical_removed": removed, "remaining_consecutive_exact_xy": remaining}


def _quantile(values: list[int], percentile: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(len(ordered) * percentile) - 1)]


def _support_band(count: int, mean: float) -> str:
    ratio = count / mean
    if ratio <= 0.25:
        return "at_or_below_0.25x_mean"
    if ratio <= 0.5:
        return "0.25x_to_0.5x_mean"
    if ratio <= 1.0:
        return "0.5x_to_1x_mean"
    if ratio <= 2.0:
        return "1x_to_2x_mean"
    if ratio <= 4.0:
        return "2x_to_4x_mean"
    return "above_4x_mean"


def _hwrt_support(canonical_root: Path, output: Path) -> dict:
    counts: Counter[str] = Counter()
    for row in _json_lines(canonical_root / "hwrt.jsonl.gz"):
        if row["point_count"] < 2:
            raise AssertionError("one-point HWRT row survived canonical filtering")
        counts[str(row["label"])] += 1
    values = list(counts.values())
    mean = statistics.fmean(values)
    bands = Counter(_support_band(value, mean) for value in values)
    with (output / "hwrt_class_support.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("label", "count", "mean_ratio", "delta_from_mean", "band"))
        writer.writeheader()
        for label, count in sorted(counts.items(), key=lambda pair: (pair[1], pair[0])):
            writer.writerow({"label": label, "count": count, "mean_ratio": f"{count / mean:.6f}", "delta_from_mean": f"{count - mean:.6f}", "band": _support_band(count, mean)})
    return {
        "classes": len(counts),
        "samples": sum(values),
        "mean": mean,
        "median": statistics.median(values),
        "stdev": statistics.pstdev(values),
        "min": min(values),
        "p05": _quantile(values, 0.05),
        "p25": _quantile(values, 0.25),
        "p75": _quantile(values, 0.75),
        "p95": _quantile(values, 0.95),
        "max": max(values),
        "bands": dict(sorted(bands.items())),
        "under_0_25x_mean": [label for label, count in sorted(counts.items()) if count <= mean * 0.25],
        "over_4x_mean": [label for label, count in sorted(counts.items()) if count > mean * 4.0],
        "decision": {
            "rebalance": "class_balanced_sampler_plus_capped_loss_weight",
            "do_not_use": "predictive_rebalancer",
            "atomic_label_policy": "do_not_split_HWRT_labels_without_box_level_annotation",
            "reason": "frequency correction cannot create missing handwriting variation; compositional LaTex labels are source annotation units and automatic splitting would corrupt stroke ownership",
        },
    }


def _support_summary(labels: list[str], counts: np.ndarray) -> dict:
    observed = [(label, int(count)) for label, count in zip(labels, counts, strict=True) if count]
    values = [count for _, count in observed]
    if not values:
        raise AssertionError("classifier input has no observed labels")
    mean = statistics.fmean(values)
    bands = Counter(_support_band(value, mean) for value in values)
    return {
        "output_labels": len(labels),
        "observed_labels": len(observed),
        "missing_labels": [label for label, count in zip(labels, counts, strict=True) if not count],
        "records": sum(values),
        "mean": mean,
        "median": statistics.median(values),
        "p05": _quantile(values, 0.05),
        "p95": _quantile(values, 0.95),
        "min": min(values),
        "max": max(values),
        "bands": dict(sorted(bands.items())),
        "at_or_below_0_25x_mean": sum(value <= mean * 0.25 for value in values),
        "above_4x_mean": sum(value > mean * 4.0 for value in values),
    }


def _feature_integrity(features: np.ndarray) -> dict:
    expected_shape = (128, 5)
    if features.ndim != 3 or tuple(features.shape[1:]) != expected_shape:
        return {"records": int(features.shape[0]) if features.ndim else 0, "valid_shape": False, "expected_shape": list(expected_shape), "observed_shape": list(features.shape)}
    result = Counter(records=int(len(features)), non_finite=0, xy_out_of_range=0, negative_delta_t=0, invalid_stroke_start=0, invalid_observed=0, no_stroke_start=0, mixed_observed=0, spatially_collapsed=0)
    for start in range(0, len(features), 4096):
        batch = np.asarray(features[start:start + 4096], dtype=np.float32)
        finite = np.isfinite(batch).all(axis=(1, 2))
        xy_ok = ((batch[:, :, :2] >= 0).all(axis=(1, 2)) & (batch[:, :, :2] <= 1).all(axis=(1, 2)))
        delta_ok = (batch[:, :, 2] >= 0).all(axis=1)
        stroke = batch[:, :, 3]
        observed = batch[:, :, 4]
        stroke_ok = ((stroke == 0) | (stroke == 1)).all(axis=1)
        observed_ok = ((observed == 0) | (observed == 1)).all(axis=1)
        starts = stroke.sum(axis=1)
        mixed_observed = ~((observed == 0).all(axis=1) | (observed == 1).all(axis=1))
        span = batch[:, :, :2].max(axis=1) - batch[:, :, :2].min(axis=1)
        collapsed = finite & (span == 0).all(axis=1)
        result["non_finite"] += int((~finite).sum())
        result["xy_out_of_range"] += int((~xy_ok).sum())
        result["negative_delta_t"] += int((~delta_ok).sum())
        result["invalid_stroke_start"] += int((~stroke_ok).sum())
        result["invalid_observed"] += int((~observed_ok).sum())
        result["no_stroke_start"] += int((starts < 1).sum())
        result["mixed_observed"] += int(mixed_observed.sum())
        result["spatially_collapsed"] += int(collapsed.sum())
    result["valid_shape"] = True
    result["expected_shape"] = list(expected_shape)
    return dict(result)


def _spatially_collapsed_label_counts(features: np.ndarray, labels: np.ndarray, vocabulary: list[str]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for start in range(0, len(features), 4096):
        batch = np.asarray(features[start:start + 4096], dtype=np.float32)
        finite = np.isfinite(batch).all(axis=(1, 2))
        span = batch[:, :, :2].max(axis=1) - batch[:, :, :2].min(axis=1)
        for index in np.flatnonzero(finite & (span == 0).all(axis=1)):
            counts[vocabulary[int(labels[start + int(index)])]] += 1
    return dict(sorted(counts.items()))


def _collision_summary(groups: list[list[tuple[str, str]]], records: int) -> dict:
    cross_split_same_label = cross_split_cross_label = within_split_cross_label = 0
    examples = []
    cross_label_examples = []
    for group in groups:
        by_split: dict[str, set[str]] = {}
        for split, label in group:
            by_split.setdefault(split, set()).add(label)
        labels = set().union(*by_split.values())
        cross_split = len(by_split) > 1
        same_label_cross_split = cross_split and bool(by_split.get("train", set()) & by_split.get("eval", set()))
        cross_label = len(labels) > 1
        cross_split_same_label += same_label_cross_split
        cross_split_cross_label += cross_split and cross_label
        within_split_cross_label += any(len(values) > 1 for values in by_split.values())
        if len(examples) < 8:
            examples.append({"records": len(group), "splits": {split: sorted(values) for split, values in sorted(by_split.items())}})
        if cross_label and len(cross_label_examples) < 8:
            cross_label_examples.append({"records": len(group), "splits": {split: sorted(values) for split, values in sorted(by_split.items())}})
    return {
        "records": records,
        "unique_tensors": records - sum(len(group) - 1 for group in groups),
        "duplicate_tensor_groups": len(groups),
        "duplicate_rows_beyond_first": sum(len(group) - 1 for group in groups),
        "cross_split_same_label_groups": cross_split_same_label,
        "cross_split_cross_label_groups": cross_split_cross_label,
        "within_split_cross_label_groups": within_split_cross_label,
        "examples": examples,
        "cross_label_examples": cross_label_examples,
    }


def _tensor_collisions(train_features: np.ndarray, train_labels: np.ndarray, eval_features: np.ndarray, eval_labels: np.ndarray, vocabulary: list[str]) -> dict:
    seen: dict[bytes, tuple[str, str] | list[tuple[str, str]]] = {}
    collisions: dict[bytes, list[tuple[str, str]]] = {}
    total = 0
    for split, features, labels in (("train", train_features, train_labels), ("eval", eval_features, eval_labels)):
        if len(features) != len(labels):
            raise AssertionError(f"{split} feature/label count mismatch")
        for start in range(0, len(features), 1024):
            batch = np.asarray(features[start:start + 1024], dtype=np.float32)
            for tensor, index in zip(batch, labels[start:start + len(batch)], strict=True):
                label = vocabulary[int(index)]
                digest = hashlib.sha256(np.ascontiguousarray(tensor).tobytes()).digest()
                current = (split, label)
                previous = seen.get(digest)
                if previous is None:
                    seen[digest] = current
                elif isinstance(previous, tuple):
                    group = [previous, current]
                    seen[digest] = group
                    collisions[digest] = group
                else:
                    previous.append(current)
                total += 1
    return _collision_summary(list(collisions.values()), total)


def _classifier_input_audit(cache_dir: Path) -> dict:
    manifest_path = cache_dir / "cache_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"classifier cache manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result: dict[str, dict] = {"cache_manifest": manifest_path.name, "heads": {}}
    for head, train_key, eval_key, vocabulary_key in (
        ("math", "math_train", "math_eval", "math_labels"),
        ("auxiliary", "auxiliary_train", "auxiliary_eval", "auxiliary_labels"),
    ):
        train = manifest["sets"][train_key]
        evaluation = manifest["sets"][eval_key]
        vocabulary = list(manifest["config"][vocabulary_key])
        train_features = np.load(cache_dir / train["features"], mmap_mode="r")
        train_labels = np.load(cache_dir / train["labels"], mmap_mode="r")
        eval_features = np.load(cache_dir / evaluation["features"], mmap_mode="r")
        eval_labels = np.load(cache_dir / evaluation["labels"], mmap_mode="r")
        if int(train_labels.max(initial=-1)) >= len(vocabulary) or int(eval_labels.max(initial=-1)) >= len(vocabulary):
            raise AssertionError(f"{head} cache label index is outside the vocabulary")
        result["heads"][head] = {
            "training_support": _support_summary(vocabulary, np.bincount(train_labels, minlength=len(vocabulary))),
            "evaluation_support": _support_summary(vocabulary, np.bincount(eval_labels, minlength=len(vocabulary))),
            "training_input_integrity": _feature_integrity(train_features),
            "evaluation_input_integrity": _feature_integrity(eval_features),
            "training_spatially_collapsed_labels": _spatially_collapsed_label_counts(train_features, train_labels, vocabulary),
            "evaluation_spatially_collapsed_labels": _spatially_collapsed_label_counts(eval_features, eval_labels, vocabulary),
            "exact_tensor_collisions": _tensor_collisions(train_features, train_labels, eval_features, eval_labels, vocabulary),
        }
    return result


def _prediction_diagnostics(cache_dir: Path, prediction_dir: Path) -> dict:
    prediction_path = prediction_dir / "external_holdout_predictions.jsonl"
    if not prediction_path.is_file():
        raise FileNotFoundError(f"fixed-holdout predictions are missing: {prediction_path}")
    predictions = {str(row["record_id"]): list(row["topk"]) for row in _json_lines(prediction_path)}
    manifest = json.loads((cache_dir / "cache_manifest.json").read_text(encoding="utf-8"))
    result: dict[str, dict] = {"status": "diagnostic_only", "fixed_holdout_used_for_selection": False, "heads": {}}
    all_truth_ids: set[str] = set()
    for head, key, train_key, vocabulary_key in (("math", "math_eval", "math_train", "math_labels"), ("auxiliary", "auxiliary_eval", "auxiliary_train", "auxiliary_labels")):
        truths = list(_json_lines(cache_dir / manifest["sets"][key]["truth"]))
        all_truth_ids.update(str(row["record_id"]) for row in truths)
        vocabulary = list(manifest["config"][vocabulary_key])
        train_labels = np.load(cache_dir / manifest["sets"][train_key]["labels"], mmap_mode="r")
        training_support = Counter(vocabulary[int(index)] for index in train_labels)
        mean_support = statistics.fmean(training_support.values())
        support: Counter[str] = Counter()
        hits: Counter[str] = Counter()
        confusions: Counter[tuple[str, str]] = Counter()
        missing = 0
        for truth in truths:
            label = str(truth["label"])
            support[label] += 1
            topk = predictions.get(str(truth["record_id"]))
            if not topk:
                missing += 1
                continue
            if topk[0] == label:
                hits[label] += 1
            else:
                confusions[(label, topk[0])] += 1
        top_confusions = [
            {"truth": truth, "predicted": predicted, "records": count, "rate_within_truth": count / support[truth]}
            for (truth, predicted), count in sorted(confusions.items(), key=lambda item: (-item[1], item[0]))[:20]
        ]
        weakest = [
            {"label": label, "records": count, "top1": hits[label] / count}
            for label, count in sorted(support.items(), key=lambda item: ((hits[item[0]] / item[1]), -item[1], item[0]))[:20]
        ]
        by_support: dict[str, Counter[str]] = {}
        for label, count in support.items():
            bucket = _support_band(training_support[label], mean_support)
            bucket_counts = by_support.setdefault(bucket, Counter())
            bucket_counts["records"] += count
            bucket_counts["top1_hits"] += hits[label]
            bucket_counts["labels"] += 1
        result["heads"][head] = {
            "records": len(truths),
            "missing_predictions": missing,
            "top_confusions": top_confusions,
            "lowest_label_top1": weakest,
            "top1_by_training_support_band": {
                bucket: {"records": values["records"], "labels": values["labels"], "top1": values["top1_hits"] / values["records"]}
                for bucket, values in sorted(by_support.items())
            },
        }
    result["unexpected_prediction_ids"] = len(set(predictions) - all_truth_ids)
    return result


def _correction_evidence(calibration_report: Path | None) -> dict:
    if calibration_report is None:
        return {"status": "not_supplied", "decision": "do_not_create_a_character_correction_from_fixed_holdout_diagnostics"}
    report = json.loads(calibration_report.read_text(encoding="utf-8"))
    pooled = report.get("pooled_direct", {})
    acceptance = report.get("acceptance", {})
    return {
        "status": "writer_disjoint_reported",
        "mode": report.get("mode"),
        "writer_groups": report.get("writer_groups"),
        "direct_top1_before": pooled.get("base", {}).get("top1"),
        "direct_top1_after": pooled.get("calibrated", {}).get("top1"),
        "punctuation_top1_before": pooled.get("base", {}).get("punctuation", {}).get("top1"),
        "punctuation_top1_after": pooled.get("calibrated", {}).get("punctuation", {}).get("top1"),
        "accepted_for_research_candidate": acceptance.get("accepted_for_research_candidate"),
        "decision": "retain_only_the_existing_writer_disjoint_output_row_calibration; do_not_edit_labels_or_expand_the_correction_set_from_fixed_holdout_diagnostics",
    }


def _bdshwa_segmentability(archive_path: Path) -> dict:
    category_counts: Counter[str] = Counter()
    with _bdshwa_csv_archive(archive_path) as archive:
        names = sorted(name for name in archive.namelist() if name.startswith("Raw_Data/") and name.endswith(".csv"))
        raw_header: list[str] = []
        for name in names:
            with archive.open(name) as stream:
                reader = csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8-sig", newline=""))
                raw_header = raw_header or list(reader.fieldnames or [])
                first_row = next(reader, None)
                if first_row is not None:
                    category_counts[str(first_row.get("category", ""))] += 1
    task_keys = {"category", "mode", "script", "task_index", "task_label", "task_prompt", "writing_condition"} & set(raw_header)
    boundary_fields = {"character", "char", "char_id", "character_index", "character_label", "segment", "segment_id", "char_start", "char_end"}
    return {
        "status": "rejected_for_character_classifier",
        "character_segmentable": False,
        "records": len(names),
        "category_counts": dict(sorted(category_counts.items())),
        "task_keys": sorted(task_keys),
        "raw_header": raw_header,
        "character_boundary_fields_present": sorted(boundary_fields & set(raw_header)),
        "reason": "Each file supplies only a whole-task prompt plus pen-down stroke IDs. It has no character boundary, character label, or character-to-stroke ownership field; freehand, shape, and wave tasks also have no text transcript suitable for character supervision.",
    }


def _write_markdown(output: Path, result: dict) -> None:
    support = result["hwrt_support"]
    lines = [
        "# Character classifier data audit",
        "",
        "## UCI duplicate decision",
        "",
        f"- Raw UCI samples: {result['uci_duplicates']['records']}",
        f"- Raw exact consecutive duplicate XY points: {result['uci_duplicates']['raw_points_removed']}",
        f"- Canonical duplicate removals: {result['uci_duplicates']['canonical_removed']} (including {result['uci_duplicates']['rounding_collisions_removed']} rounding collisions).",
        f"- Remaining canonical exact consecutive duplicates: {result['uci_duplicates']['remaining_consecutive_exact_xy']}",
        f"- Samples affected: {result['uci_duplicates']['affected_records']}",
        "- Decision: remove only adjacent exact XY duplicates; preserve return-to-point loops.",
        "",
        "## HWRT support distribution",
        "",
        f"- Usable samples: {support['samples']} across {support['classes']} classes.",
        f"- Mean {support['mean']:.2f}; median {support['median']:.1f}; p05 {support['p05']}; p95 {support['p95']}; range {support['min']}..{support['max']}.",
        f"- Distribution bands: `{json.dumps(support['bands'], sort_keys=True)}`.",
        "- Decision: retain class-balanced sampling only. The prior sampler-plus-loss double correction was rejected; do not use a predictive rebalancer or split atomic source labels automatically.",
        "",
        "## BDSHWA decision",
        "",
        "- Rejected from the character-classifier corpus: whole-task prompts and stroke IDs do not supply character boundaries or ownership.",
        "- Raw archive remains untouched as audit evidence; it is not a tensor, training, or evaluation input.",
        "",
    ]
    if classifier := result.get("classifier_input_audit"):
        lines += ["## Classifier-input audit", ""]
        for head, audit in classifier["heads"].items():
            support = audit["training_support"]
            integrity = audit["training_input_integrity"]
            collisions = audit["exact_tensor_collisions"]
            collapsed_labels = audit["training_spatially_collapsed_labels"]
            conflict_examples = collisions["cross_label_examples"]
            lines += [
                f"- {head}: {support['records']} training rows; {support['observed_labels']}/{support['output_labels']} output labels observed; missing `{', '.join(support['missing_labels']) or 'none'}`.",
                f"- {head}: support range {support['min']}..{support['max']}; {support['at_or_below_0_25x_mean']} labels at or below 0.25x observed-class mean and {support['above_4x_mean']} above 4x.",
                f"- {head}: invalid tensor rows {sum(integrity[key] for key in ('non_finite', 'xy_out_of_range', 'negative_delta_t', 'invalid_stroke_start', 'invalid_observed', 'no_stroke_start', 'mixed_observed'))}; spatially collapsed rows {integrity['spatially_collapsed']}.",
                f"- {head}: exact 128x5 duplicate groups {collisions['duplicate_tensor_groups']}; train/eval same-label groups {collisions['cross_split_same_label_groups']}; cross-label groups across splits {collisions['cross_split_cross_label_groups']}.",
            ]
            if collapsed_labels:
                lines.append(f"- {head}: spatially collapsed training labels `{json.dumps(collapsed_labels, ensure_ascii=False, sort_keys=True)}`.")
            if conflict_examples:
                lines.append(f"- {head}: cross-label exact-tensor evidence `{json.dumps(conflict_examples, ensure_ascii=False, sort_keys=True)}`.")
        lines += ["- Decision: exact-tensor collisions and fixed-holdout predictions are diagnostics, not a source of new labels, data deletion, or correction tuning.", ""]
    if diagnostics := result.get("fixed_holdout_diagnostics"):
        lines += ["## Fixed-holdout character confusions", ""]
        for head, diagnostic in diagnostics["heads"].items():
            pairs = diagnostic["top_confusions"][:5]
            rendered = ", ".join(f"`{item['truth']} -> {item['predicted']}` ({item['records']})" for item in pairs) or "none"
            bands = ", ".join(f"{band}: {values['top1']:.2%}" for band, values in diagnostic["top1_by_training_support_band"].items())
            lines.append(f"- {head}: top diagnostic Top-1 confusions: {rendered}.")
            lines.append(f"- {head}: diagnostic Top-1 by training-support band: {bands}.")
        lines += ["- The fixed external holdout remains excluded from epoch selection and correction-set expansion.", ""]
    if correction := result.get("character_correction"):
        lines += ["## Character-correction boundary", "", f"- Status: {correction['status']}; decision: `{correction['decision']}`."]
        if correction["status"] == "writer_disjoint_reported":
            lines.append(f"- Existing writer-LOO output-row calibration: {correction['direct_top1_before']:.4f} -> {correction['direct_top1_after']:.4f} direct Top-1; punctuation {correction['punctuation_top1_before']:.4f} -> {correction['punctuation_top1_after']:.4f}; accepted={correction['accepted_for_research_candidate']}.")
        lines.append("")
    (output / "CHARACTER_CLASSIFIER_DATA_AUDIT.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def audit(canonical_root: Path, output: Path, bdshwa_archive: Path, cache_dir: Path | None = None, prediction_dir: Path | None = None, calibration_report: Path | None = None) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    uci = _uci_raw_samples()
    duplicates = [row for row in uci if row[3]]
    canonical_uci = _canonical_uci_dedup(canonical_root)
    raw_removed = sum(row[3] for row in uci)
    if canonical_uci["canonical_removed"] < raw_removed or canonical_uci["remaining_consecutive_exact_xy"]:
        raise AssertionError("UCI canonical duplicate removal is incomplete")
    result = {
        "schema": "aiflow-character-classifier-data-audit/v2",
        "uci_duplicates": {
            "records": len(uci),
            "affected_records": len(duplicates),
            "raw_points_removed": raw_removed,
            "canonical_removed": canonical_uci["canonical_removed"],
            "rounding_collisions_removed": canonical_uci["canonical_removed"] - raw_removed,
            "remaining_consecutive_exact_xy": canonical_uci["remaining_consecutive_exact_xy"],
            "plot": _plot_uci(uci, output),
        },
        "hwrt_support": _hwrt_support(canonical_root, output),
        "bdshwa": _bdshwa_segmentability(bdshwa_archive),
    }
    if cache_dir is not None:
        result["classifier_input_audit"] = _classifier_input_audit(cache_dir)
    if prediction_dir is not None:
        if cache_dir is None:
            raise ValueError("--prediction-dir requires --cache-dir")
        result["fixed_holdout_diagnostics"] = _prediction_diagnostics(cache_dir, prediction_dir)
    result["character_correction"] = _correction_evidence(calibration_report)
    (output / "character_classifier_data_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    _write_markdown(output, result)
    return result


def _self_test() -> None:
    points = [(0.0, 0.0), (0.0, 0.0), (1.0, 0.0), (0.0, 0.0)]
    result, removed = deduplicate_exact_consecutive_xy(points)
    assert result == [(0.0, 0.0), (1.0, 0.0), (0.0, 0.0)] and removed == 1
    assert _support_band(1, 10) == "at_or_below_0.25x_mean"
    assert _support_band(50, 10) == "above_4x_mean"
    collisions = _collision_summary([[('train', 'a'), ('eval', 'a')], [('train', 'x'), ('eval', 'y')]], 4)
    assert collisions["cross_split_same_label_groups"] == 1 and collisions["cross_split_cross_label_groups"] == 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path, default=DEFAULT_CANONICAL_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bdshwa-archive", type=Path, default=DEFAULT_BDSHWA_ARCHIVE)
    parser.add_argument("--cache-dir", type=Path, help="existing local classifier cache to audit at the exact 128x5 input boundary")
    parser.add_argument("--prediction-dir", type=Path, help="existing external-only checkpoint output containing fixed-holdout predictions; diagnostics only")
    parser.add_argument("--calibration-report", type=Path, help="existing writer-disjoint project-symbol calibration report")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test(); print(json.dumps({"self_test": "pass"})); return 0
    print(json.dumps(audit(args.canonical_root, args.output, args.bdshwa_archive, args.cache_dir, args.prediction_dir, args.calibration_report), ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
