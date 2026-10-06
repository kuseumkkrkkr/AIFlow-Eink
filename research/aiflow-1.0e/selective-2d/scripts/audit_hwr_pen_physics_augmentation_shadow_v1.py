#!/usr/bin/env python3
"""One-shot label-preservation audit for the existing clean-room pen-physics augmenter.

This is a frozen-checkpoint inference shadow only. It does not train, tune the
augmenter, or change product behavior. Optional sample rendering writes only a
visual QA image from consumed development traces, never training data. A
positive result is not evidence of a training gain or generalization.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from build_normalized_ink_v1 import SourceSample, _canonicalize
from cleanroom_pen_physics_v3 import (
    GENERIC_CLASS_PROFILE,
    MOTOR_LIGHT_PHYSICS,
    simulate_pen_physics_v3,
)
from evaluate_48hz_prefix_v1 import _load_model, _prefix_tensor, resample_direct_48hz
from train_character_classifier_v1 import apply_input_mode


SCHEMA = "aiflow-hwr-pen-physics-augmentation-shadow/v1"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AFFINE_REPORT = (
    ROOT / "artifacts" / "hwr_failure_cause_microscope_20260928"
    / "group_mean_geometry_prior_shadow_20261001" / "affine_tta_shadow.json"
)
INFERENCE_BATCH_SIZE = 128
GENERATOR_SEED = 20261001
BOOTSTRAP_SEED = 20261002


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_lines(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _predict_logits(model, features: np.ndarray, device: torch.device) -> np.ndarray:
    output = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(features), INFERENCE_BATCH_SIZE):
            batch = torch.from_numpy(features[start:start + INFERENCE_BATCH_SIZE]).to(device)
            output.append(model.math_head(model.encode(batch)).float().cpu().numpy())
    logits = np.concatenate(output, axis=0)
    if logits.ndim != 2 or logits.shape != (len(features), 372):
        raise ValueError(f"unexpected frozen HWR logits shape: {logits.shape}")
    if not np.isfinite(logits).all():
        raise ValueError("frozen HWR returned non-finite logits")
    return logits


def _scores(
    logits: np.ndarray,
    targets: list[str],
    formula_ids: list[str],
    writer_by_formula: dict[str, str],
    label_to_index: dict[str, int],
) -> dict[str, Any]:
    top5_indices = np.argsort(-logits, axis=1)[:, :5]
    classes = np.asarray([int(np.argmax(row)) for row in logits], dtype=np.int64)
    class_ids = np.asarray(top5_indices, dtype=np.int64)
    formula_top1: dict[str, list[bool]] = defaultdict(list)
    formula_top5: dict[str, list[bool]] = defaultdict(list)
    formula_supported: dict[str, list[bool]] = defaultdict(list)
    target_indices = np.asarray([int(label_to_index[target]) for target in targets], dtype=np.int64)
    if len(target_indices) != len(formula_ids):
        raise ValueError("target labels and formula IDs do not align")
    for index, formula_id in enumerate(formula_ids):
        target_index = int(target_indices[index])
        supported = target_index >= 0
        formula_supported[formula_id].append(supported)
        formula_top1[formula_id].append(bool(supported and classes[index] == target_index))
        formula_top5[formula_id].append(bool(supported and target_index in class_ids[index]))
    top1_exact = {key: all(values) for key, values in formula_top1.items()}
    top5_exact = {key: all(values) for key, values in formula_top5.items()}
    supported_formula_ids = sorted(
        key for key, values in formula_supported.items() if all(values)
    )
    unsupported_formula_ids = sorted(set(formula_supported) - set(supported_formula_ids))
    scorable_tokens = target_indices >= 0
    token_top1_hits = classes == target_indices
    token_top5_hits = np.asarray([
        target_indices[index] in class_ids[index] if scorable_tokens[index] else False
        for index in range(len(targets))
    ], dtype=bool)
    return {
        "token_top1_hits": int(token_top1_hits.sum()),
        "token_top5_hits": int(token_top5_hits.sum()),
        "token_count": len(targets),
        "scorable_token_count": int(scorable_tokens.sum()),
        "unscorable_token_count": int((~scorable_tokens).sum()),
        "token_top1_hits_supported": int((token_top1_hits & scorable_tokens).sum()),
        "token_top5_hits_supported": int((token_top5_hits & scorable_tokens).sum()),
        "formula_top1_exact": sum(top1_exact.values()),
        "formula_top5_exact": sum(top5_exact.values()),
        "formula_count": len(top1_exact),
        "supported_formula_count": len(supported_formula_ids),
        "unsupported_formula_count": len(unsupported_formula_ids),
        "formula_top1_exact_supported": sum(top1_exact[key] for key in supported_formula_ids),
        "formula_top5_exact_supported": sum(top5_exact[key] for key in supported_formula_ids),
        "supported_formula_ids": supported_formula_ids,
        "unsupported_formula_ids": unsupported_formula_ids,
        "top1_exact_by_formula": top1_exact,
        "top5_exact_by_formula": top5_exact,
    }


def _paired_writer_bootstrap(
    before: dict[str, bool], after: dict[str, bool], writer_by_formula: dict[str, str],
) -> dict[str, Any]:
    by_writer: dict[str, list[float]] = defaultdict(list)
    for formula_id, was_exact in before.items():
        by_writer[writer_by_formula[formula_id]].append(
            float(after[formula_id]) - float(was_exact)
        )
    writer_deltas = np.asarray(
        [np.mean(values) for _writer, values in sorted(by_writer.items())],
        dtype=np.float64,
    )
    if len(writer_deltas) == 0:
        raise ValueError("writer-cluster bootstrap has no formula groups")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.empty(10_000, dtype=np.float64)
    for index in range(len(draws)):
        draws[index] = float(
            rng.choice(writer_deltas, size=len(writer_deltas), replace=True).mean()
        ) * 100.0
    return {
        "method": "paired writer-cluster bootstrap; writers sampled with replacement",
        "iterations": int(len(draws)),
        "seed": BOOTSTRAP_SEED,
        "writer_clusters": int(len(writer_deltas)),
        "delta_formula_top1_exact_pp_95_interval": [
            float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975)),
        ],
        "scope": "consumed-development exploratory shadow only",
    }


def _token_failure_breakdown(
    original_logits: np.ndarray,
    augmented_logits: np.ndarray,
    targets: list[str],
    features: np.ndarray,
    augmented_features: np.ndarray,
    label_to_index: dict[str, int],
) -> dict[str, Any]:
    if features.shape != augmented_features.shape or len(features) != len(targets):
        raise ValueError("original/augmented feature rows do not align with targets")
    original_top1 = np.argmax(original_logits, axis=1)
    augmented_top1 = np.argmax(augmented_logits, axis=1)
    original_top5 = np.argsort(-original_logits, axis=1)[:, :5]
    augmented_top5 = np.argsort(-augmented_logits, axis=1)[:, :5]
    by_label: dict[str, Counter] = defaultdict(Counter)
    by_stroke_bucket: dict[str, Counter] = defaultdict(Counter)
    by_label_and_stroke_bucket: dict[str, dict[str, Counter]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    multistroke_layout: dict[str, list[dict[str, float]]] = defaultdict(list)
    for index, target in enumerate(targets):
        target_index = label_to_index.get(target, -1)
        before_top1 = bool(target_index >= 0 and original_top1[index] == target_index)
        after_top1 = bool(target_index >= 0 and augmented_top1[index] == target_index)
        before_top5 = bool(target_index >= 0 and target_index in original_top5[index])
        after_top5 = bool(target_index >= 0 and target_index in augmented_top5[index])
        row = {
            "count": 1,
            "top1_before": int(before_top1),
            "top1_after": int(after_top1),
            "top1_recovered": int(not before_top1 and after_top1),
            "top1_regressed": int(before_top1 and not after_top1),
            "top5_before": int(before_top5),
            "top5_after": int(after_top5),
            "top5_recovered": int(not before_top5 and after_top5),
            "top5_regressed": int(before_top5 and not after_top5),
            "top1_prediction_changed": int(original_top1[index] != augmented_top1[index]),
        }
        by_label[target].update(row)
        stroke_count = int(np.count_nonzero(features[index, :, 3] > 0.5))
        bucket = "1" if stroke_count == 1 else "2-3" if stroke_count <= 3 else "4+"
        by_stroke_bucket[bucket].update(row)
        by_label_and_stroke_bucket[target][bucket].update(row)
        if stroke_count > 1:
            outcome = (
                "lost" if before_top5 and not after_top5 else
                "gained" if not before_top5 and after_top5 else
                "retained" if before_top5 and after_top5 else
                "still_missing"
            )
            layout = _inter_stroke_layout_delta(features[index], augmented_features[index])
            if layout is not None:
                multistroke_layout[outcome].append(layout)

    def materialize(table: dict[str, Counter]) -> dict[str, dict[str, int]]:
        return {
            key: {metric: int(value) for metric, value in sorted(counts.items())}
            for key, counts in sorted(table.items())
        }

    layout_fields = (
        "mean_relative_vector_displacement",
        "p95_relative_vector_displacement",
        "max_relative_vector_displacement",
        "mean_pairwise_spacing_change",
        "max_pairwise_spacing_change",
    )
    layout_summary = {
        outcome: {
            "group_count": len(rows),
            **{
                field: float(np.mean([row[field] for row in rows]))
                for field in layout_fields
            },
        }
        for outcome, rows in sorted(multistroke_layout.items())
        if rows
    }

    return {
        "per_target_label": materialize(by_label),
        "per_group_stroke_count_bucket": materialize(by_stroke_bucket),
        "per_target_label_and_stroke_count_bucket": {
            label: materialize(dict(buckets))
            for label, buckets in sorted(by_label_and_stroke_bucket.items())
        },
        "multistroke_layout_by_top5_transition": layout_summary,
    }


def _inter_stroke_layout_delta(
    original: np.ndarray, augmented: np.ndarray,
) -> dict[str, float] | None:
    """Measure within-symbol stroke-centroid relation changes in unit-square XY."""
    def centroids(row: np.ndarray) -> np.ndarray:
        starts = np.flatnonzero(row[:, 3] > 0.5).tolist()
        if not starts or starts[0] != 0:
            raise ValueError("symbol tensor does not begin with stroke_start")
        ends = starts[1:] + [len(row)]
        return np.stack([
            row[start:end, :2].mean(axis=0)
            for start, end in zip(starts, ends, strict=True)
        ])

    before, after = centroids(original), centroids(augmented)
    if len(before) < 2 or len(after) != len(before):
        return None
    pairs = [(left, right) for left in range(len(before)) for right in range(left + 1, len(before))]
    vector_delta = np.asarray([
        np.linalg.norm((after[right] - after[left]) - (before[right] - before[left]))
        for left, right in pairs
    ], dtype=np.float64)
    distance_delta = np.asarray([
        abs(
            np.linalg.norm(after[right] - after[left])
            - np.linalg.norm(before[right] - before[left])
        )
        for left, right in pairs
    ], dtype=np.float64)
    return {
        "mean_relative_vector_displacement": float(vector_delta.mean()),
        "p95_relative_vector_displacement": float(np.quantile(vector_delta, 0.95)),
        "max_relative_vector_displacement": float(vector_delta.max()),
        "mean_pairwise_spacing_change": float(distance_delta.mean()),
        "max_pairwise_spacing_change": float(distance_delta.max()),
    }


def _write_sample_sheet(
    path: Path,
    original: np.ndarray,
    augmented: np.ndarray,
    targets: list[str],
    class_labels: list[str],
    original_logits: np.ndarray,
    augmented_logits: np.ndarray,
) -> dict[str, Any]:
    """Render paired examples from the frozen development set for visual QA."""

    from PIL import Image, ImageDraw, ImageFont

    preferred = ("0", "1", "2", "5", "9", "x", "y", "+", "=", r"\times", r"\div", r"\sqrt{}")
    target_set = set(targets)
    available = [label for label in preferred if label in target_set]
    if not available:
        raise ValueError("no preferred target labels are present for image sampling")
    panel_size, margin, row_height = 184, 10, 205
    image = Image.new("RGB", (520, 46 + row_height * len(available)), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((margin, 10), "Frozen development ink only — original vs one physics view", fill="#14213d", font=font)

    def draw_group(feature: np.ndarray, x0: int, y0: int, color: str) -> None:
        starts = np.flatnonzero(feature[:, 3] > 0.5).tolist()
        if not starts or starts[0] != 0:
            raise ValueError("sample tensor has invalid stroke-start channel")
        ends = starts[1:] + [len(feature)]
        for start, end in zip(starts, ends, strict=True):
            points = [
                (
                    x0 + margin + float(feature[index, 0]) * (panel_size - 2 * margin),
                    y0 + margin + float(feature[index, 1]) * (panel_size - 2 * margin),
                )
                for index in range(start, end)
            ]
            if len(points) == 1:
                px, py = points[0]
                draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=color)
            elif len(points) > 1:
                draw.line(points, fill=color, width=3, joint="curve")

    for row, label in enumerate(available):
        candidates = [index for index, target in enumerate(targets) if target == label]
        displacements = np.asarray([
            np.sqrt(np.mean(np.square(augmented[index, :, :2] - original[index, :, :2])))
            for index in candidates
        ])
        selected = candidates[int(np.argmin(np.abs(displacements - np.median(displacements))))]
        y = 46 + row * row_height
        before_id, after_id = int(np.argmax(original_logits[selected])), int(np.argmax(augmented_logits[selected]))
        draw.text(
            (margin, y + 2),
            f"truth={label}  Top1={class_labels[before_id]} -> {class_labels[after_id]}",
            fill="#14213d", font=font,
        )
        for panel, (title, features, color) in enumerate((
            ("original", original, "#1d3557"),
            ("augmented", augmented, "#d1495b"),
        )):
            x = margin + panel * (panel_size + 52)
            panel_y = y + 18
            draw.text((x, panel_y), title, fill="#33415c", font=font)
            box_y = panel_y + 16
            draw.rectangle((x, box_y, x + panel_size, box_y + panel_size), outline="#d9e0ea", width=1)
            draw_group(features[selected], x, box_y, color)

    path = path.resolve()
    if path.exists():
        raise FileExistsError(f"sample image already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return {
        "path": str(path),
        "scope": "frozen_development_evaluation_only_not_training",
        "label_count": len(available),
        "labels": available,
        "selection": "one median-displacement example per available digit/Latin/operator target",
    }


def audit(
    affine_report_path: Path,
    device_name: str = "cuda",
    sample_image_path: Path | None = None,
) -> dict[str, Any]:
    report_path = affine_report_path.resolve()
    if any("crohme" in part.lower() for part in report_path.parts):
        raise ValueError("CROHME paths are forbidden")
    affine_report = json.loads(report_path.read_text(encoding="utf-8"))
    inputs = affine_report["inputs"]
    summary_path = Path(inputs["summary_path"]).resolve()
    trace_path = Path(inputs["trace_path"]).resolve()
    dataset_root = Path(inputs["dataset_root"]).resolve()
    checkpoint_path = Path(inputs["checkpoint_path"]).resolve()
    for path in (summary_path, trace_path, dataset_root, checkpoint_path):
        if any("crohme" in part.lower() for part in path.parts):
            raise ValueError("CROHME data paths are forbidden")
    for path_key, path in (
        ("summary_sha256", summary_path),
        ("trace_sha256", trace_path),
        ("formulas_valid_sha256", dataset_root / "data" / "formulas_valid.jsonl"),
        ("ownership_train_sha256", dataset_root / "data" / "ownership_train.jsonl"),
        ("checkpoint_sha256", checkpoint_path),
    ):
        if _sha256(path) != inputs[path_key]:
            raise AssertionError(f"frozen source hash mismatch: {path_key}")

    trace_rows = _json_lines(trace_path)
    trace_by_id = {str(row["sample_id"]): row for row in trace_rows}
    formula_rows = _json_lines(dataset_root / "data" / "formulas_valid.jsonl")
    annotation_rows = [
        row for row in _json_lines(dataset_root / "data" / "ownership_train.jsonl")
        if row.get("accepted")
    ]
    ownership_label_counts = Counter(
        str(label) for row in annotation_rows for label in row["labels"]
    )
    formulas = {str(row["sample_id"]): row for row in formula_rows}
    annotations = {str(row["sample_id"]): row for row in annotation_rows}
    expected_ids = set(trace_by_id)
    if len(trace_by_id) != len(trace_rows) or not expected_ids <= set(formulas) or not expected_ids <= set(annotations):
        raise AssertionError("frozen trace/formula/ownership ID coverage mismatch")

    feature_rows: list[np.ndarray] = []
    targets: list[str] = []
    formula_ids: list[str] = []
    writer_by_formula: dict[str, str] = {}
    for formula_id in sorted(expected_ids):
        trace = trace_by_id[formula_id]
        annotation = annotations[formula_id]
        symbols = {
            tuple(sorted(int(value) for value in symbol["stroke_indices"])): symbol
            for symbol in trace["oracle_group_hwr"]["symbols"]
        }
        source = formulas[formula_id]
        strokes = sorted(source["strokes"], key=lambda row: int(row["order"]))
        groups, labels = annotation["groups"], annotation["labels"]
        if len(groups) != len(labels) or len(groups) != len(symbols):
            raise AssertionError(f"ownership/trace group mismatch: {formula_id}")
        writer_by_formula[formula_id] = str(annotation["writer_id"])
        for group, label_value in zip(groups, labels, strict=True):
            stroke_indices = tuple(sorted({int(value) for value in group}))
            symbol = symbols.get(stroke_indices)
            label = str(label_value)
            if symbol is None or str(symbol["target_label"]) != label:
                raise AssertionError(f"ownership/trace label mismatch: {formula_id}:{stroke_indices}")
            raw_strokes = [
                [
                    (float(point["x"]), float(point["y"]), float(point.get("t_ms", 0.0)))
                    for point in strokes[stroke_index]["points"]
                ]
                for stroke_index in stroke_indices
            ]
            sample = SourceSample(
                "project_owned_group_candidate",
                f"{formula_id}:{','.join(map(str, stroke_indices))}",
                label,
                "grouping_candidate",
                "candidate_only",
                raw_strokes,
            )
            normalized = _canonicalize(sample)
            resampled = resample_direct_48hz(normalized)
            features = apply_input_mode(_prefix_tensor(resampled), "uniform-time")
            if features.shape != (128, 5) or not np.isfinite(features).all():
                raise AssertionError(f"invalid normalized group tensor: {formula_id}:{stroke_indices}")
            feature_rows.append(features.astype(np.float32, copy=False))
            targets.append(label)
            formula_ids.append(formula_id)

    values = np.stack(feature_rows).astype(np.float32, copy=False)
    before_channels = values[:, :, 2:].copy()
    started = time.perf_counter()
    augmented_tensor, physics_rows = simulate_pen_physics_v3(
        torch.from_numpy(values),
        MOTOR_LIGHT_PHYSICS,
        torch.Generator(device="cpu").manual_seed(GENERATOR_SEED),
        row_profiles=[GENERIC_CLASS_PROFILE] * len(values),
        return_diagnostics=True,
    )
    generation_seconds = time.perf_counter() - started
    augmented = augmented_tensor.cpu().numpy().astype(np.float32, copy=False)
    if not np.array_equal(augmented[:, :, 2:], before_channels):
        raise AssertionError("augmentation changed non-spatial channels")
    if not np.isfinite(augmented).all() or (augmented[:, :, :2] < 0.0).any() or (augmented[:, :, :2] > 1.0).any():
        raise AssertionError("augmentation produced invalid normalized coordinates")

    device = torch.device(device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu")
    model, labels, _checkpoint = _load_model(checkpoint_path, device)
    if len(labels) != 372 or len(set(labels)) != 372:
        raise AssertionError("frozen checkpoint vocabulary does not contain 372 unique labels")
    label_to_index = {label: index for index, label in enumerate(labels)}
    missing = sorted(set(targets) - set(label_to_index))
    score_label_to_index = {
        target: label_to_index.get(target, -1) for target in set(targets)
    }

    inference_started = time.perf_counter()
    original_logits = _predict_logits(model, values, device)
    augmented_logits = _predict_logits(model, augmented, device)
    inference_seconds = time.perf_counter() - inference_started
    sample_image = None
    if sample_image_path is not None:
        sample_image = _write_sample_sheet(
            sample_image_path, values, augmented, targets, labels,
            original_logits, augmented_logits,
        )
    original_metrics = _scores(
        original_logits, targets, formula_ids, writer_by_formula, score_label_to_index,
    )
    augmented_metrics = _scores(
        augmented_logits, targets, formula_ids, writer_by_formula, score_label_to_index,
    )

    before_exact = original_metrics.pop("top1_exact_by_formula")
    after_exact = augmented_metrics.pop("top1_exact_by_formula")
    supported_formula_ids = set(original_metrics.pop("supported_formula_ids"))
    augmented_supported_formula_ids = set(augmented_metrics.pop("supported_formula_ids"))
    if supported_formula_ids != augmented_supported_formula_ids:
        raise AssertionError("supported formula coverage changed between identity and augmentation")
    supported_before_exact = {
        formula_id: before_exact[formula_id] for formula_id in supported_formula_ids
    }
    supported_after_exact = {
        formula_id: after_exact[formula_id] for formula_id in supported_formula_ids
    }
    original_metrics.pop("top5_exact_by_formula")
    augmented_metrics.pop("top5_exact_by_formula")
    original_metrics.pop("unsupported_formula_ids")
    augmented_metrics.pop("unsupported_formula_ids")
    changed_top1 = int(np.sum(np.argmax(original_logits, axis=1) != np.argmax(augmented_logits, axis=1)))
    top1_recovered = int(augmented_metrics["token_top1_hits"] - original_metrics["token_top1_hits"])
    top5_delta = int(augmented_metrics["token_top5_hits"] - original_metrics["token_top5_hits"])
    diag_rms = np.asarray([row["rms_displacement"] for row in physics_rows], dtype=np.float64)
    diag_path = np.asarray([row["path_length_ratio"] for row in physics_rows], dtype=np.float64)
    exact_transition = {
        "both_wrong": 0,
        "both_exact": 0,
        "physics_recovery": 0,
        "physics_regression": 0,
    }
    for formula_id in before_exact:
        before, after = before_exact[formula_id], after_exact[formula_id]
        key = (
            "both_exact" if before and after else
            "both_wrong" if not before and not after else
            "physics_recovery" if after else
            "physics_regression"
        )
        exact_transition[key] += 1

    identity = affine_report["metrics"]["identity_replay"]
    if (
        original_metrics["formula_top1_exact"] != int(identity["formula_exact"])
        or original_metrics["token_top1_hits"] != int(identity["token_hits"])
        or original_metrics["token_top5_hits"] != int(identity["identity_top5_target_symbol_hits"])
        or original_metrics["token_count"] != int(identity["token_count"])
    ):
        raise AssertionError("reconstructed identity baseline differs from frozen affine audit")

    return {
        "schema": SCHEMA,
        "status": "consumed_development_augmentation_shadow_only",
        "training_performed": False,
        "training_data_written": False,
        "files_written": sample_image is not None,
        "product_default_changed": False,
        "crohme_rows": 0,
        "augmentation": {
            "implementation": "existing cleanroom_pen_physics_v3.simulate_pen_physics_v3",
            "config": "MOTOR_LIGHT_PHYSICS",
            "class_profile": "GENERIC_CLASS_PROFILE",
            "seed": GENERATOR_SEED,
            "fitted_or_tuned": False,
            "sample_count": len(values),
            "changed_xy_rows": int(np.count_nonzero(diag_rms > 0.0)),
            "topology_reverted_rows": int(sum(bool(row["topology_reverted"]) for row in physics_rows)),
            "path_length_reverted_rows": int(sum(bool(row["path_length_reverted"]) for row in physics_rows)),
            "rms_displacement": {
                "mean": float(diag_rms.mean()),
                "p50": float(np.quantile(diag_rms, 0.50)),
                "p95": float(np.quantile(diag_rms, 0.95)),
                "max": float(diag_rms.max()),
            },
            "path_length_ratio": {
                "min": float(diag_path.min()),
                "median": float(np.median(diag_path)),
                "max": float(diag_path.max()),
            },
            "non_spatial_channels_unchanged": True,
            "normalized_xy_finite_and_bounded": True,
            "generation_seconds": generation_seconds,
        },
        "frozen_hwr": {
            "checkpoint_sha256": inputs["checkpoint_sha256"],
            "source_formula_sha256": inputs["formulas_valid_sha256"],
            "source_ownership_sha256": inputs["ownership_train_sha256"],
            "formula_trace_sha256": inputs["trace_sha256"],
            "target_labels_absent_from_checkpoint_vocabulary": missing,
            "evaluation_gold_ownership_formula_count": len(annotation_rows),
            "evaluation_gold_ownership_symbol_count": int(sum(ownership_label_counts.values())),
            "evaluation_gold_ownership_unique_label_count": len(ownership_label_counts),
            "evaluation_gold_ownership_symbol_count_by_label": dict(sorted(ownership_label_counts.items())),
            "evaluation_gold_labels_absent_from_checkpoint_vocabulary": sorted(
                set(ownership_label_counts) - set(labels)
            ),
            "checkpoint_labels_without_evaluation_gold_examples": len(
                set(labels) - set(ownership_label_counts)
            ),
            "formula_count": original_metrics["formula_count"],
            "owned_group_count": original_metrics["token_count"],
            "device": str(device),
        },
        "identity_vs_physics": {
            "identity": original_metrics,
            "physics_augmented": augmented_metrics,
            "delta_token_top1_hits": top1_recovered,
            "delta_token_top5_hits": top5_delta,
            "changed_token_top1_predictions": changed_top1,
            "delta_formula_top1_exact": (
                augmented_metrics["formula_top1_exact"] - original_metrics["formula_top1_exact"]
            ),
            "delta_supported_formula_top1_exact": (
                augmented_metrics["formula_top1_exact_supported"]
                - original_metrics["formula_top1_exact_supported"]
            ),
            "formula_exact_transition": exact_transition,
            "paired_writer_cluster_bootstrap": _paired_writer_bootstrap(
                before_exact, after_exact, writer_by_formula,
            ),
            "supported_classes_paired_writer_cluster_bootstrap": _paired_writer_bootstrap(
                supported_before_exact, supported_after_exact, writer_by_formula,
            ),
            "token_failure_breakdown": _token_failure_breakdown(
                original_logits, augmented_logits, targets, values, augmented,
                label_to_index,
            ),
            "sample_image": sample_image,
            "inference_seconds_both_views": inference_seconds,
        },
        "interpretation_limits": [
            "this checks label preservation under one fixed synthetic view, not a benefit from training with augmentation",
            "the frozen model and consumed development writers may be dependent; do not treat this as independent accuracy",
            "any training experiment requires train-only augmentation and a newly frozen writer/formula-disjoint acceptance set",
            "external isolated-symbol corpora cannot supervise formula ownership, spatial relations, or strict formula decoding",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--affine-report", type=Path, default=DEFAULT_AFFINE_REPORT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--sample-image", type=Path, default=None)
    args = parser.parse_args()
    report = audit(args.affine_report, args.device, args.sample_image)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
