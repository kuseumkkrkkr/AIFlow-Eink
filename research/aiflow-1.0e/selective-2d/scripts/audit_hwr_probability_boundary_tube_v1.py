#!/usr/bin/env python3
"""Probe the frozen HWR teacher along an empirical, correlated class-2 shape tube.

This is a training-pool diagnostic only. The tube is fitted on HWRT train rows
and queried on UJI train rows; held-out splits and CROHME are never read. It
does not train, tune, or promote a model.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = Path(
    r"D:\AIFlow-Workspace\PrivateData\math-ink-data-collector\derived"
    r"\hwr-augmentation-20261001\affine-distill-v1"
)
DEFAULT_CHECKPOINT = Path(
    r"D:\AIFlow-Workspace\PrivateData\math-ink-data-collector\derived"
    r"\hwr-head-calibration-20260819-r1\final_all_writers_steps250_lr1e-3"
    r"\project_symbol_head_checkpoint.pt"
)
DEFAULT_REPORT_DIR = ROOT / "artifacts" / "hwr_probability_boundary_tube_20261004_interpolation"
DEFAULT_CROSS_CLASS_REPORT_DIR = ROOT / "artifacts" / "hwr_probability_boundary_tube_20261004_crossclass"
DEFAULT_CROSS_CLASS_REFINED_REPORT_DIR = ROOT / "artifacts" / "hwr_probability_boundary_tube_20261004_crossclass_refined"
SCALES = (0.0, 0.25, 0.50, 0.75, 1.0)
DISTILL_TEMPERATURE = 2.0
MAX_COMPONENTS = 12
MIN_FIT_ROWS = 24
PATH_RATIO_RANGE = (0.78, 1.22)
SEED = 20261004
MIN_COMMIT_HEADROOM_GIB = 1.5
np: Any = None


def _load_numpy() -> Any:
    """Load NumPy only after the low-commit safety gate has passed."""
    global np
    if np is None:
        import numpy as numpy_module
        np = numpy_module
    return np


def _available_commit_gib() -> float:
    """Return Windows commit headroom without loading torch/model state."""
    if os.name != "nt":
        raise RuntimeError("memory guard supports Windows only; refusing heavyweight HWR inference")

    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_uint32),
            ("dwMemoryLoad", ctypes.c_uint32),
            ("ullTotalPhys", ctypes.c_uint64),
            ("ullAvailPhys", ctypes.c_uint64),
            ("ullTotalPageFile", ctypes.c_uint64),
            ("ullAvailPageFile", ctypes.c_uint64),
            ("ullTotalVirtual", ctypes.c_uint64),
            ("ullAvailVirtual", ctypes.c_uint64),
            ("ullAvailExtendedVirtual", ctypes.c_uint64),
        ]

    status = MemoryStatusEx()
    status.dwLength = ctypes.sizeof(MemoryStatusEx)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("GlobalMemoryStatusEx could not read current commit headroom")
    return float(status.ullAvailPageFile) / (1024.0 ** 3)


def _guard_commit(stage: str) -> float | None:
    try:
        available = _available_commit_gib()
    except (OSError, RuntimeError) as exc:
        reason = str(exc)
        available = None
    else:
        reason = "commit headroom is below the safety floor" if available < MIN_COMMIT_HEADROOM_GIB else ""
    if reason:
        print(json.dumps({
            "event": "probability_boundary_tube_refused",
            "stage": stage,
            "reason": reason,
            "available_commit_headroom_gib": None if available is None else round(available, 3),
            "minimum_commit_headroom_gib": MIN_COMMIT_HEADROOM_GIB,
            "numpy_loaded": np is not None,
        }, ensure_ascii=False), flush=True)
        return None
    return available


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _single_stroke_rows(features: np.ndarray) -> np.ndarray:
    if features.ndim != 3 or features.shape[1:] != (128, 5):
        raise ValueError(f"expected [N,128,5] HWR features; got {features.shape}")
    starts = features[:, :, 3] > 0.5
    return starts[:, 0] & (starts.sum(axis=1) == 1)


def _fit_empirical_tube(features: np.ndarray, coverage: float = 0.68) -> dict[str, Any]:
    """Fit a low-rank metric and q68 pairwise radius from real class samples."""
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (128, 5):
        raise ValueError(f"expected [N,128,5] features; got {values.shape}")
    if not np.isfinite(values).all() or (values[:, :, :2] < 0.0).any() or (values[:, :, :2] > 1.0).any():
        raise ValueError("fit inputs must be finite unit-square trajectories")
    if len(values) < MIN_FIT_ROWS:
        raise ValueError(f"empirical tube needs at least {MIN_FIT_ROWS} rows; got {len(values)}")
    if not 0.5 < coverage < 0.95:
        raise ValueError("coverage must be between 0.5 and 0.95")
    if not _single_stroke_rows(values).all():
        raise ValueError("empirical class-2 pilot accepts single-stroke rows only")

    xy = values[:, :, :2].reshape(len(values), -1)
    center = np.median(xy, axis=0)
    centered = xy - center
    _u, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    variance = np.square(singular_values) / max(len(values) - 1, 1)
    positive = variance > max(float(variance[0]) * 1.0e-8, 1.0e-12)
    rank = int(positive.sum())
    if rank < 2:
        raise ValueError("training rows do not contain enough shape variation")
    cumulative = np.cumsum(variance[:rank]) / max(float(variance[:rank].sum()), 1.0e-12)
    selected_rank = min(
        rank,
        MAX_COMPONENTS,
        max(2, int(np.searchsorted(cumulative, 0.95, side="left") + 1)),
    )
    basis = vt[:selected_rank]
    eigenvalues = np.maximum(variance[:selected_rank], max(float(variance[0]) * 1.0e-6, 1.0e-10))
    coefficients = centered @ basis.T
    standardized = coefficients / np.sqrt(eigenvalues)[None, :]
    pairwise = np.linalg.norm(standardized[:, None, :] - standardized[None, :, :], axis=2)
    pairwise_radii = pairwise[np.triu_indices(len(values), k=1)]
    radius = float(np.quantile(pairwise_radii, coverage, method="linear"))
    if not math.isfinite(radius) or radius <= 1.0e-8:
        raise ValueError("empirical pairwise coverage radius is degenerate")
    explained = float(variance[:selected_rank].sum() / max(float(variance.sum()), 1.0e-12))
    return {
        "center": center,
        "basis": basis,
        "eigenvalues": eigenvalues,
        "standardized_coefficients": standardized,
        "pairwise_radii": pairwise_radii,
        "pairwise_q68_radius": radius,
        "coverage": coverage,
        "component_count": selected_rank,
        "explained_variance_ratio": explained,
        "fit_rows": int(len(values)),
        "control_points_per_stroke": 128,
        "topology": "one stroke; original 128-point arc-length samples",
    }


def _select_empirical_donors(
    fit_features: np.ndarray,
    query_features: np.ndarray,
    tube: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Pair each query with a real same-label trajectory near the q68 limit."""
    fit_xy = np.asarray(fit_features[:, :, :2], dtype=np.float64)
    query = np.asarray(query_features, dtype=np.float32)
    query_xy = np.asarray(query[:, :, :2], dtype=np.float64)
    query_flat = query_xy.reshape(len(query_xy), -1)
    query_coeff = ((query_flat - tube["center"]) @ tube["basis"].T) / np.sqrt(tube["eigenvalues"])[None, :]

    # Consider both recorded directions; do not morph opposite pen travel
    # directions into one another when a direction-compatible donor exists.
    variants = np.concatenate((fit_xy, fit_xy[:, ::-1, :]), axis=0)
    variant_ids = np.tile(np.arange(len(fit_xy)), 2)
    variant_reversed = np.concatenate((np.zeros(len(fit_xy), dtype=bool), np.ones(len(fit_xy), dtype=bool)))
    variant_flat = variants.reshape(len(variants), -1)
    variant_coeff = ((variant_flat - tube["center"]) @ tube["basis"].T) / np.sqrt(tube["eigenvalues"])[None, :]
    distances = np.linalg.norm(query_coeff[:, None, :] - variant_coeff[None, :, :], axis=2)

    fit_vectors = variants[:, -1, :] - variants[:, 0, :]
    query_vectors = query_xy[:, -1, :] - query_xy[:, 0, :]
    cosine = (query_vectors @ fit_vectors.T) / np.maximum(
        np.linalg.norm(query_vectors, axis=1)[:, None] * np.linalg.norm(fit_vectors, axis=1)[None, :],
        1.0e-8,
    )
    donor_variant_indices = np.empty(len(query), dtype=np.int64)
    orientation_fallback = np.zeros(len(query), dtype=bool)
    q68 = float(tube["pairwise_q68_radius"])
    for row in range(len(query)):
        compatible = np.flatnonzero(cosine[row] >= 0.0)
        if not len(compatible):
            compatible = np.arange(len(variants))
            orientation_fallback[row] = True
        within = compatible[distances[row, compatible] <= q68]
        candidates = within if len(within) else compatible
        if len(within):
            donor_variant_indices[row] = candidates[np.argmax(distances[row, candidates])]
        else:
            donor_variant_indices[row] = candidates[np.argmin(distances[row, candidates])]

    selected_distances = distances[np.arange(len(query)), donor_variant_indices]
    max_fraction = np.minimum(1.0, q68 / np.maximum(selected_distances, 1.0e-8))
    donors = query.copy()
    donors[:, :, :2] = variants[donor_variant_indices].astype(np.float32)
    return donors, {
        "donor_fit_row_indices": variant_ids[donor_variant_indices],
        "donor_reversed": variant_reversed[donor_variant_indices],
        "distance_mahalanobis": selected_distances,
        "max_interpolation_fraction": max_fraction,
        "endpoint_direction_cosine": cosine[np.arange(len(query)), donor_variant_indices],
        "orientation_filter_fallback": orientation_fallback,
        "q68_radius": q68,
    }


def _select_empirical_rival_donors(
    fit_features: np.ndarray,
    fit_labels: np.ndarray,
    query_features: np.ndarray,
    query_rival_ids: np.ndarray,
    tube: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Pair each query with a real HWRT sample from its frozen-teacher rival class."""
    fit = np.asarray(fit_features, dtype=np.float32)
    labels = np.asarray(fit_labels, dtype=np.int64)
    query = np.asarray(query_features, dtype=np.float32)
    rival_ids = np.asarray(query_rival_ids, dtype=np.int64)
    if fit.ndim != 3 or fit.shape[1:] != (128, 5) or labels.shape != (len(fit),):
        raise ValueError("rival fit arrays must be row-aligned [N,128,5] features and [N] labels")
    if query.ndim != 3 or query.shape[1:] != (128, 5) or rival_ids.shape != (len(query),):
        raise ValueError("rival query arrays must be row-aligned [N,128,5] features and [N] labels")
    if not _single_stroke_rows(fit).all() or not _single_stroke_rows(query).all():
        raise ValueError("cross-class boundary pilot accepts single-stroke rows only")

    donors = query.copy()
    max_fraction = np.zeros(len(query), dtype=np.float64)
    donor_available = np.zeros(len(query), dtype=bool)
    donor_row_indices = np.full(len(query), -1, dtype=np.int64)
    donor_reversed = np.zeros(len(query), dtype=bool)
    distances_out = np.full(len(query), np.nan, dtype=np.float64)
    direction_cosine_out = np.full(len(query), np.nan, dtype=np.float64)
    orientation_fallback = np.zeros(len(query), dtype=bool)
    class_support: dict[int, int] = {}
    center = np.asarray(tube["center"], dtype=np.float64)
    basis = np.asarray(tube["basis"], dtype=np.float64)
    eigenvalues = np.asarray(tube["eigenvalues"], dtype=np.float64)
    q68 = float(tube["pairwise_q68_radius"])
    query_xy = np.asarray(query[:, :, :2], dtype=np.float64)
    query_coeff = ((query_xy.reshape(len(query), -1) - center) @ basis.T) / np.sqrt(eigenvalues)[None, :]

    for rival_id in np.unique(rival_ids):
        query_indices = np.flatnonzero(rival_ids == rival_id)
        fit_indices = np.flatnonzero(labels == rival_id)
        class_support[int(rival_id)] = int(len(fit_indices))
        if not len(fit_indices):
            continue
        fit_xy = np.asarray(fit[fit_indices, :, :2], dtype=np.float64)
        variants = np.concatenate((fit_xy, fit_xy[:, ::-1, :]), axis=0)
        variant_ids = np.tile(fit_indices, 2)
        variant_reversed = np.concatenate((np.zeros(len(fit_indices), dtype=bool), np.ones(len(fit_indices), dtype=bool)))
        variant_coeff = ((variants.reshape(len(variants), -1) - center) @ basis.T) / np.sqrt(eigenvalues)[None, :]
        fit_vectors = variants[:, -1, :] - variants[:, 0, :]
        for query_index in query_indices:
            distances = np.linalg.norm(variant_coeff - query_coeff[query_index], axis=1)
            query_vector = query_xy[query_index, -1, :] - query_xy[query_index, 0, :]
            cosine = (fit_vectors @ query_vector) / np.maximum(
                np.linalg.norm(fit_vectors, axis=1) * np.linalg.norm(query_vector), 1.0e-8,
            )
            compatible = np.flatnonzero(cosine >= 0.0)
            if not len(compatible):
                compatible = np.arange(len(variants))
                orientation_fallback[query_index] = True
            within = compatible[distances[compatible] <= q68]
            candidates = within if len(within) else compatible
            selected = candidates[np.argmax(distances[candidates])] if len(within) else candidates[np.argmin(distances[candidates])]
            distance = float(distances[selected])
            donors[query_index, :, :2] = variants[selected].astype(np.float32)
            donor_available[query_index] = True
            donor_row_indices[query_index] = int(variant_ids[selected])
            donor_reversed[query_index] = bool(variant_reversed[selected])
            distances_out[query_index] = distance
            direction_cosine_out[query_index] = float(cosine[selected])
            max_fraction[query_index] = min(1.0, q68 / max(distance, 1.0e-8))

    return donors, {
        "donor_available": donor_available,
        "donor_fit_row_indices": donor_row_indices,
        "donor_reversed": donor_reversed,
        "distance_mahalanobis": distances_out,
        "max_interpolation_fraction": max_fraction,
        "endpoint_direction_cosine": direction_cosine_out,
        "orientation_filter_fallback": orientation_fallback,
        "fit_support_by_rival_id": class_support,
        "q68_radius": q68,
    }


def _apply_tube(
    features: np.ndarray,
    donors: np.ndarray,
    max_fraction: np.ndarray,
    scale: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Interpolate along one real-to-real, same-class trajectory path."""
    if not 0.0 <= scale <= 1.0:
        raise ValueError("tube scale must be in [0, 1]")
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (128, 5):
        raise ValueError("features must have shape [N,128,5]")
    if not np.isfinite(values).all() or (values[:, :, :2] < 0.0).any() or (values[:, :, :2] > 1.0).any():
        raise ValueError("tube inputs must be finite unit-square trajectories")
    donor_values = np.asarray(donors, dtype=np.float32)
    if donor_values.shape != values.shape or max_fraction.shape != (len(values),):
        raise ValueError("donor trajectories do not align with query rows")
    if not _single_stroke_rows(values).all():
        raise ValueError("empirical class-2 pilot accepts single-stroke rows only")

    if not _single_stroke_rows(donor_values).all():
        raise ValueError("empirical class-2 donors must preserve single-stroke topology")
    if not np.isfinite(donor_values).all() or (donor_values[:, :, :2] < 0.0).any() or (donor_values[:, :, :2] > 1.0).any():
        raise ValueError("donor inputs must be finite unit-square trajectories")
    delta = donor_values[:, :, :2] - values[:, :, :2]
    output = values.copy()
    output[:, :, :2] += (delta * (max_fraction * scale)[:, None, None]).astype(np.float32)

    original_steps = np.linalg.norm(np.diff(values[:, :, :2], axis=1), axis=2)
    changed_steps = np.linalg.norm(np.diff(output[:, :, :2], axis=1), axis=2)
    original_length = np.maximum(original_steps.sum(axis=1), 1.0e-8)
    path_ratio = changed_steps.sum(axis=1) / original_length
    xy_delta = output[:, :, :2] - values[:, :, :2]
    rms = np.sqrt(np.mean(np.square(xy_delta), axis=(1, 2)))
    max_point = np.linalg.norm(xy_delta, axis=2).max(axis=1)
    valid = (
        np.isfinite(output).all(axis=(1, 2))
        & (output[:, :, :2].min(axis=(1, 2)) >= 0.0)
        & (output[:, :, :2].max(axis=(1, 2)) <= 1.0)
        & (path_ratio >= PATH_RATIO_RANGE[0])
        & (path_ratio <= PATH_RATIO_RANGE[1])
    )
    non_xy_unchanged = np.all(output[:, :, 2:] == values[:, :, 2:], axis=(1, 2))
    start_mask_unchanged = np.all(
        (output[:, :, 3] > 0.5) == (values[:, :, 3] > 0.5), axis=1,
    )
    diagnostics = {
        "valid": valid,
        "rms": rms,
        "max_point_displacement": max_point,
        "path_ratio": path_ratio,
        "non_xy_unchanged": non_xy_unchanged,
        "stroke_starts_unchanged": start_mask_unchanged,
        "endpoint_displacement_p95": float(np.quantile(
            np.linalg.norm(delta[:, (0, -1), :], axis=2), 0.95,
        )),
        "adjacent_displacement_step_p95": float(np.quantile(
            np.linalg.norm(np.diff(delta, axis=1), axis=2), 0.95,
        )),
    }
    if not non_xy_unchanged.all() or not start_mask_unchanged.all():
        raise AssertionError("tube transform modified time, observed, or stroke-start channels")
    return output, diagnostics


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = np.asarray(logits, dtype=np.float64) / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    probabilities = np.exp(scaled)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def _boundary_margin(
    logits: np.ndarray,
    target_id: int,
    rival_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Use all competitors unless an explicitly requested fixed pair is supplied."""
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or not 0 <= target_id < values.shape[1]:
        raise ValueError("boundary logits or target class are malformed")
    if rival_ids is not None:
        rivals = np.asarray(rival_ids, dtype=np.int64)
        if rivals.shape != (len(values),) or np.any(rivals == target_id):
            raise ValueError("fixed rivals must align with query rows and differ from the target")
        return values[:, target_id] - values[np.arange(len(values)), rivals]
    competitors = values.copy()
    competitors[:, target_id] = -np.inf
    return values[:, target_id] - competitors.max(axis=1)


def _summarize_logits(logits: np.ndarray, labels: list[str], target_id: int) -> dict[str, Any]:
    probabilities = _softmax(logits, DISTILL_TEMPERATURE)
    strongest_rival = np.asarray(logits, dtype=np.float64).copy()
    target_logits = strongest_rival[:, target_id].copy()
    strongest_rival[:, target_id] = -np.inf
    rival_ids = np.argmax(strongest_rival, axis=1)
    margins = target_logits - strongest_rival[np.arange(len(logits)), rival_ids]
    rankings = 1 + (logits > logits[:, target_id, None]).sum(axis=1)
    top5 = np.argsort(-logits, axis=1)[:, :5]
    ranks = np.asarray(rankings, dtype=np.int64)
    return {
        "rows": int(len(logits)),
        "target_top1": int(np.sum(np.argmax(logits, axis=1) == target_id)),
        "target_top5": int(np.sum(np.any(top5 == target_id, axis=1))),
        "target_softmax_probability_mean_t2": float(probabilities[:, target_id].mean()),
        "target_softmax_probability_p10_t2": float(np.quantile(probabilities[:, target_id], 0.10)),
        "target_logit_margin_mean": float(margins.mean()),
        "target_logit_margin_p10": float(np.quantile(margins, 0.10)),
        "target_logit_margin_p50": float(np.quantile(margins, 0.50)),
        "target_margin_nonpositive": int(np.sum(margins <= 0.0)),
        "target_rank_median": float(np.median(ranks)),
        "strongest_rivals": [
            {"label": labels[int(index)], "count": int(count)}
            for index, count in sorted(
                zip(*np.unique(rival_ids, return_counts=True), strict=True),
                key=lambda item: (-int(item[1]), int(item[0])),
            )[:10]
        ],
    }


def _predict_logits(model, features: np.ndarray, torch, device, batch_size: int) -> np.ndarray:
    output = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            batch = torch.from_numpy(np.asarray(features[start:start + batch_size], dtype=np.float32)).to(device)
            logits = model.math_head(model.encode(batch)).float().cpu().numpy()
            if logits.ndim != 2 or logits.shape[1] != 372 or not np.isfinite(logits).all():
                raise ValueError("frozen teacher returned malformed or non-finite logits")
            output.append(logits)
    return np.concatenate(output, axis=0) if output else np.empty((0, 372), dtype=np.float32)


def _draw_sample_sheet(
    originals: np.ndarray,
    views: list[np.ndarray],
    labels: list[str],
    valid_by_scale: list[np.ndarray],
    path: Path,
    scale_names: tuple[str, ...] | None = None,
    cell_width: int = 230,
    repeat_row_label: bool = True,
    max_examples: int = 16,
) -> dict[str, Any]:
    from PIL import Image, ImageDraw, ImageFont

    if max_examples < 1:
        raise ValueError("sample-sheet max_examples must be positive")
    shown = min(max_examples, len(originals))
    cell_w, cell_h, padding = cell_width, 106, 8
    cols = 5
    image = Image.new("RGB", (cols * cell_w, (shown + cols - 1) // cols * cell_h), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    scale_names = scale_names or ("B", ".25", ".50", ".75", "q68")
    if len(scale_names) != len(views) + 1 or len(valid_by_scale) != len(views):
        raise ValueError("sample-sheet scale names and validity masks must align with the views")
    panel_width = cell_w // len(scale_names)
    for index in range(shown):
        x0 = (index % cols) * cell_w
        y0 = (index // cols) * cell_h
        for col, feature in enumerate([originals[index], *(view[index] for view in views)]):
            left = x0 + col * panel_width
            top = y0 + 20
            glyph_extent = min(panel_width - 4, 62)
            glyph_left = left + (panel_width - glyph_extent) / 2.0
            glyph_top = top + (62 - glyph_extent) / 2.0
            draw.text((left, y0 + 3), scale_names[col], fill="#34495e", font=font)
            color = "#111111" if col == 0 else ("#1f77b4" if valid_by_scale[col - 1][index] else "#c0392b")
            starts = np.flatnonzero(feature[:, 3] > 0.5).tolist()
            ends = starts[1:] + [len(feature)]
            for stroke_start, stroke_end in zip(starts, ends, strict=True):
                points = [
                    (glyph_left + float(feature[point, 0]) * glyph_extent, glyph_top + float(feature[point, 1]) * glyph_extent)
                    for point in range(stroke_start, stroke_end)
                ]
                if len(points) > 1:
                    draw.line(points, fill=color, width=2, joint="curve")
            if repeat_row_label:
                draw.text((left, y0 + 87), labels[index], fill="#111111", font=font)
        if not repeat_row_label:
            draw.text((x0, y0 + 87), labels[index], fill="#111111", font=font)
    if path.exists():
        raise FileExistsError(path)
    image.save(path)
    return {"path": str(path.resolve()), "rows_shown": shown, "red_means_geometry_gate_reject": True, "equal_xy_display_scale": True}


def _make_blind_review_packet(
    candidates: np.ndarray,
    crossing_rows: np.ndarray,
    output_dir: Path,
) -> dict[str, Any]:
    """Render shuffled isolated candidates without teacher labels, margins, or endpoints."""
    from PIL import Image, ImageDraw, ImageFont

    if candidates.shape != (len(crossing_rows), 2, 128, 5):
        raise ValueError("blind review candidates must be aligned [N,2,128,5] endpoints")
    features = candidates.reshape(-1, 128, 5)
    order = np.random.default_rng(SEED).permutation(len(features))
    columns, cell_width, cell_height, glyph_extent = 10, 110, 122, 90
    picture = Image.new("RGB", (columns * cell_width, math.ceil(len(features) / columns) * cell_height), "white")
    draw = ImageDraw.Draw(picture)
    font = ImageFont.load_default()
    mapping = []
    queue_path = output_dir / "blind_human_review_queue.csv"
    with queue_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("candidate_id", "human_label", "human_confidence", "notes"))
        writer.writeheader()
        for sheet_index, flat_index in enumerate(order):
            candidate_id = f"C{sheet_index + 1:03d}"
            x0 = (sheet_index % columns) * cell_width
            y0 = (sheet_index // columns) * cell_height
            draw.text((x0 + 10, y0 + 3), candidate_id, fill="#111111", font=font)
            feature = features[int(flat_index)]
            starts = np.flatnonzero(feature[:, 3] > 0.5).tolist()
            ends = starts[1:] + [len(feature)]
            for stroke_start, stroke_end in zip(starts, ends, strict=True):
                points = [
                    (x0 + 10 + float(point[0]) * glyph_extent, y0 + 20 + float(point[1]) * glyph_extent)
                    for point in feature[stroke_start:stroke_end]
                ]
                if len(points) > 1:
                    draw.line(points, fill="#111111", width=2, joint="curve")
            writer.writerow({"candidate_id": candidate_id, "human_label": "", "human_confidence": "", "notes": ""})
            mapping.append({
                "candidate_id": candidate_id,
                "cache_row": int(flat_index) // 2,
                "query_row_index": int(crossing_rows[int(flat_index) // 2]),
                "endpoint": "positive" if int(flat_index) % 2 == 0 else "nonpositive",
            })
    picture_path = output_dir / "blind_boundary_candidates.png"
    picture.save(picture_path)
    mapping_path = output_dir / "blind_review_mapping.json"
    mapping_path.write_text(json.dumps(mapping, indent=2) + "\n", encoding="utf-8")
    return {
        "image": str(picture_path.resolve()), "queue": str(queue_path.resolve()),
        "mapping": str(mapping_path.resolve()), "rows": int(len(features)),
        "equal_xy_display_scale": True, "teacher_information_shown": False,
        "human_label_cells_filled": 0, "shuffle_seed": SEED,
    }


def _apply_tube_per_row_scales(
    features: np.ndarray,
    donors: np.ndarray,
    max_fraction: np.ndarray,
    scales: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply individually selected path scales while reusing the audited geometry gates."""
    scale_values = np.asarray(scales, dtype=np.float64)
    values = np.asarray(features, dtype=np.float32)
    donor_values = np.asarray(donors, dtype=np.float32)
    fractions = np.asarray(max_fraction, dtype=np.float64)
    if scale_values.shape != (len(values),) or fractions.shape != (len(values),):
        raise ValueError("per-row path scales and q68 fractions must align with feature rows")
    if not np.isfinite(scale_values).all() or (scale_values < 0.0).any() or (scale_values > 1.0).any():
        raise ValueError("per-row path scales must be finite values in [0,1]")
    output = np.empty_like(values)
    valid = np.zeros(len(values), dtype=bool)
    for index, scale in enumerate(scale_values):
        output[index], diagnostic = _apply_tube(
            values[index:index + 1], donor_values[index:index + 1],
            fractions[index:index + 1], float(scale),
        )
        valid[index] = bool(diagnostic["valid"][0])
    return output, valid


def _refine_crossclass_boundaries(
    originals: np.ndarray,
    donors: np.ndarray,
    max_fraction: np.ndarray,
    donor_available: np.ndarray,
    rival_ids: np.ndarray,
    target_id: int,
    labels: list[str],
    first_crossing_scale: np.ndarray,
    margins_by_scale: list[np.ndarray],
    model: Any,
    torch: Any,
    device: Any,
    batch_size: int,
    output_dir: Path,
    rounds: int = 5,
    boundary_rule: str = "global-top1",
    query_training_rows: np.ndarray | None = None,
    donor_training_rows: np.ndarray | None = None,
) -> dict[str, Any]:
    """Bisect only geometry-valid, coarse teacher-boundary crossings."""
    crossing_rows = np.flatnonzero(np.isfinite(first_crossing_scale))
    if not len(crossing_rows):
        return {
            "crossing_rows": 0,
            "refined_rows": 0,
            "geometry_stopped_rows": 0,
            "rounds_requested": rounds,
            "records": [],
        }
    if query_training_rows is None or donor_training_rows is None:
        raise ValueError("candidate caches require actual source training-row indices")
    query_training_rows = np.asarray(query_training_rows, dtype=np.int64)
    donor_training_rows = np.asarray(donor_training_rows, dtype=np.int64)
    if query_training_rows.shape != (len(originals),) or donor_training_rows.shape != (len(originals),):
        raise ValueError("source training-row indices must align with query trajectories")
    if np.any(query_training_rows[crossing_rows] < 0) or np.any(donor_training_rows[crossing_rows] < 0):
        raise ValueError("crossing candidates cannot have missing source rows")

    lower = np.empty(len(crossing_rows), dtype=np.float64)
    upper = np.empty(len(crossing_rows), dtype=np.float64)
    lower_margin = np.empty(len(crossing_rows), dtype=np.float64)
    upper_margin = np.empty(len(crossing_rows), dtype=np.float64)
    initial_lower = np.empty(len(crossing_rows), dtype=np.float64)
    iterations = np.zeros(len(crossing_rows), dtype=np.int64)
    geometry_stopped = np.zeros(len(crossing_rows), dtype=bool)
    for local_index, row_index in enumerate(crossing_rows):
        upper_index = next(
            index for index, scale in enumerate(SCALES)
            if np.isclose(scale, first_crossing_scale[row_index])
        )
        if upper_index < 1:
            raise AssertionError("a crossing must have a positive tested scale")
        lower[local_index] = SCALES[upper_index - 1]
        upper[local_index] = SCALES[upper_index]
        initial_lower[local_index] = lower[local_index]
        lower_margin[local_index] = margins_by_scale[upper_index - 1][row_index]
        upper_margin[local_index] = margins_by_scale[upper_index][row_index]
        if lower_margin[local_index] <= 0.0 or upper_margin[local_index] > 0.0:
            raise AssertionError("coarse crossing did not provide a positive-to-nonpositive margin bracket")

    active = np.arange(len(crossing_rows), dtype=np.int64)
    for _round_index in range(rounds):
        if not len(active):
            break
        midpoint = (lower[active] + upper[active]) / 2.0
        query_rows = crossing_rows[active]
        candidates, geometry_valid = _apply_tube_per_row_scales(
            originals[query_rows], donors[query_rows], max_fraction[query_rows], midpoint,
        )
        geometry_valid &= donor_available[query_rows]
        rejected = active[~geometry_valid]
        geometry_stopped[rejected] = True
        valid_active = active[geometry_valid]
        if not len(valid_active):
            active = np.empty(0, dtype=np.int64)
            continue
        valid_rows = crossing_rows[valid_active]
        logits = _predict_logits(model, candidates[geometry_valid], torch, device, batch_size)
        fixed_rivals = rival_ids[valid_rows] if boundary_rule == "fixed-pair" else None
        candidate_margins = _boundary_margin(logits, target_id, fixed_rivals)
        positive = candidate_margins > 0.0
        lower[valid_active[positive]] = midpoint[geometry_valid][positive]
        lower_margin[valid_active[positive]] = candidate_margins[positive]
        upper[valid_active[~positive]] = midpoint[geometry_valid][~positive]
        upper_margin[valid_active[~positive]] = candidate_margins[~positive]
        iterations[valid_active] += 1
        active = valid_active

    lower_views, lower_valid = _apply_tube_per_row_scales(
        originals[crossing_rows], donors[crossing_rows], max_fraction[crossing_rows], lower,
    )
    upper_views, upper_valid = _apply_tube_per_row_scales(
        originals[crossing_rows], donors[crossing_rows], max_fraction[crossing_rows], upper,
    )
    lower_valid &= donor_available[crossing_rows]
    upper_valid &= donor_available[crossing_rows]

    lower_logits = np.full((len(crossing_rows), len(labels)), np.nan, dtype=np.float32)
    upper_logits = np.full_like(lower_logits, np.nan)
    if lower_valid.any():
        lower_logits[lower_valid] = _predict_logits(model, lower_views[lower_valid], torch, device, batch_size)
    if upper_valid.any():
        upper_logits[upper_valid] = _predict_logits(model, upper_views[upper_valid], torch, device, batch_size)
    rule_rivals = rival_ids[crossing_rows] if boundary_rule == "fixed-pair" else None
    verified_lower_margins = _boundary_margin(lower_logits, target_id, rule_rivals)
    verified_upper_margins = _boundary_margin(upper_logits, target_id, rule_rivals)
    if np.any(verified_lower_margins[lower_valid] <= 0.0) or np.any(verified_upper_margins[upper_valid] > 0.0):
        raise AssertionError("fresh endpoint inference violated the refined boundary bracket")

    def posterior_summary(side_logits: np.ndarray, valid: np.ndarray) -> dict[str, Any]:
        if not valid.any():
            return {"rows": 0}
        row_ids = crossing_rows[valid]
        logits = side_logits[valid]
        probabilities = _softmax(logits, DISTILL_TEMPERATURE)
        rivals = rival_ids[row_ids]
        competitors = logits.copy()
        competitors[:, target_id] = -np.inf
        active_rivals = np.argmax(competitors, axis=1)
        winner_ids = np.argmax(logits, axis=1)
        target_probability = probabilities[:, target_id]
        rival_probability = probabilities[np.arange(len(row_ids)), rivals]
        pair_mass = target_probability + rival_probability
        pair_target = target_probability / np.maximum(pair_mass, 1.0e-12)
        pair_rival = rival_probability / np.maximum(pair_mass, 1.0e-12)
        pair_entropy = -(
            pair_target * np.log(np.maximum(pair_target, 1.0e-12))
            + pair_rival * np.log(np.maximum(pair_rival, 1.0e-12))
        )
        by_temperature = {}
        for temperature in (1.0, 2.0, 4.0):
            current = _softmax(logits, temperature)
            sorted_probs = np.sort(current, axis=1)[:, ::-1]
            current_target = current[:, target_id]
            current_active = current[np.arange(len(row_ids)), active_rivals]
            by_temperature[str(temperature)] = {
                "target_probability_mean": float(current_target.mean()),
                "active_rival_probability_mean": float(current_active.mean()),
                "other_mass_mean": float(np.mean(1.0 - current_target - current_active)),
                "top2_mass_mean": float(sorted_probs[:, :2].sum(axis=1).mean()),
                "top5_mass_mean": float(sorted_probs[:, :5].sum(axis=1).mean()),
                "top10_mass_mean": float(sorted_probs[:, :10].sum(axis=1).mean()),
                "entropy_nats_mean": float(np.mean(-np.sum(current * np.log(np.maximum(current, 1.0e-12)), axis=1))),
            }
        return {
            "rows": int(len(row_ids)),
            "full_372_way_p2_mean": float(target_probability.mean()),
            "full_372_way_p2_p10": float(np.quantile(target_probability, 0.10)),
            "full_372_way_prival_mean": float(rival_probability.mean()),
            "full_372_way_prival_p10": float(np.quantile(rival_probability, 0.10)),
            "full_372_way_other_mass_mean": float(np.mean(1.0 - pair_mass)),
            "pair_renormalized_p2_mean": float(pair_target.mean()),
            "pair_renormalized_prival_mean": float(pair_rival.mean()),
            "pair_renormalized_entropy_mean": float(pair_entropy.mean()),
            "teacher_target_top1_rows": int(np.sum(winner_ids == target_id)),
            "teacher_target_strict_global_winner_rows": int(np.sum(_boundary_margin(logits, target_id) > 0.0)),
            "active_rival_differs_from_donor_rival_rows": int(np.sum(active_rivals != rivals)),
            "active_rival_counts": {
                labels[int(class_id)]: int(np.sum(active_rivals == class_id)) for class_id in np.unique(active_rivals)
            },
            "by_temperature": by_temperature,
        }

    display_local = sorted(
        range(len(crossing_rows)),
        key=lambda local_index: (int(rival_ids[crossing_rows[local_index]]), int(crossing_rows[local_index])),
    )
    display_rows = crossing_rows[np.asarray(display_local, dtype=np.int64)]
    records = []
    for local_index, row_index in enumerate(crossing_rows):
        records.append({
            "query_row_index": int(row_index),
            "query_training_row_index": int(query_training_rows[row_index]),
            "donor_training_row_index": int(donor_training_rows[row_index]),
            "rival_label": labels[int(rival_ids[row_index])],
            "first_coarse_crossing_scale": float(first_crossing_scale[row_index]),
            "initial_positive_scale": float(initial_lower[local_index]),
            "initial_nonpositive_scale": float(first_crossing_scale[row_index]),
            "final_positive_scale": float(lower[local_index]),
            "final_positive_margin": float(lower_margin[local_index]),
            "final_nonpositive_scale": float(upper[local_index]),
            "final_nonpositive_margin": float(upper_margin[local_index]),
            "final_bracket_width": float(upper[local_index] - lower[local_index]),
            "binary_search_steps": int(iterations[local_index]),
            "geometry_stopped_refinement": bool(geometry_stopped[local_index]),
            "final_bracket_valid": bool(lower_valid[local_index] and upper_valid[local_index]),
            "positive_side_global_margin": float(_boundary_margin(lower_logits[local_index:local_index + 1], target_id)[0]),
            "nonpositive_side_global_margin": float(_boundary_margin(upper_logits[local_index:local_index + 1], target_id)[0]),
        })
    coarse_views, coarse_valid = _apply_tube_per_row_scales(
        originals[display_rows], donors[display_rows], max_fraction[display_rows],
        initial_lower[np.asarray(display_local, dtype=np.int64)],
    )
    near_lower_views, near_lower_valid = _apply_tube_per_row_scales(
        originals[display_rows], donors[display_rows], max_fraction[display_rows],
        lower[np.asarray(display_local, dtype=np.int64)],
    )
    near_upper_views, near_upper_valid = _apply_tube_per_row_scales(
        originals[display_rows], donors[display_rows], max_fraction[display_rows],
        upper[np.asarray(display_local, dtype=np.int64)],
    )
    q68_views, q68_valid = _apply_tube_per_row_scales(
        originals[display_rows], donors[display_rows], max_fraction[display_rows],
        np.ones(len(display_rows), dtype=np.float64),
    )
    rival_names = [
        f"{sheet_index + 1:02d} 2>{labels[int(rival_ids[row_index])][:8]}"
        for sheet_index, row_index in enumerate(display_rows)
    ]
    queue_path = output_dir / "human_review_queue.csv"
    queue_fields = (
        "sheet_index", "query_row_index", "rival_label", "coarse_crossing_scale",
        "last_positive_scale", "last_positive_margin", "first_nonpositive_scale",
        "first_nonpositive_margin", "bracket_width", "human_label_positive_side",
        "human_label_nonpositive_side", "human_confidence", "review_notes",
    )
    record_by_query = {record["query_row_index"]: record for record in records}
    with queue_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=queue_fields)
        writer.writeheader()
        for sheet_index, row_index in enumerate(display_rows, start=1):
            record = record_by_query[int(row_index)]
            writer.writerow({
                "sheet_index": sheet_index,
                "query_row_index": int(row_index),
                "rival_label": record["rival_label"],
                "coarse_crossing_scale": record["first_coarse_crossing_scale"],
                "last_positive_scale": record["final_positive_scale"],
                "last_positive_margin": record["final_positive_margin"],
                "first_nonpositive_scale": record["final_nonpositive_scale"],
                "first_nonpositive_margin": record["final_nonpositive_margin"],
                "bracket_width": record["final_bracket_width"],
                "human_label_positive_side": "",
                "human_label_nonpositive_side": "",
                "human_confidence": "",
                "review_notes": "",
            })
    preview = _draw_sample_sheet(
        originals[display_rows],
        [coarse_views, near_lower_views, near_upper_views, q68_views],
        rival_names,
        [coarse_valid, near_lower_valid, near_upper_valid, q68_valid],
        output_dir / "class2_teacher_boundary_refinement.png",
        scale_names=("B", "c+", "n+", "n-", "q68"),
        cell_width=280,
        repeat_row_label=False,
        max_examples=len(display_rows),
    )

    initial_widths = [record["initial_nonpositive_scale"] - record["initial_positive_scale"] for record in records]
    final_widths = [record["final_bracket_width"] for record in records]
    numeric_artifacts = {}
    candidate_features = np.stack((lower_views, upper_views), axis=1)
    for name, values in (
        ("candidate_features", candidate_features),
        ("teacher_logits", np.stack((lower_logits, upper_logits), axis=1)),
        ("geometry_valid", np.stack((lower_valid, upper_valid), axis=1)),
        ("query_row_indices", crossing_rows),
        ("query_training_row_indices", query_training_rows[crossing_rows]),
        ("donor_training_row_indices", donor_training_rows[crossing_rows]),
    ):
        path = output_dir / f"{name}.npy"
        if path.exists():
            raise FileExistsError(path)
        np.save(path, values, allow_pickle=False)
        numeric_artifacts[name] = {
            "path": str(path.resolve()), "sha256": _sha256(path), "shape": list(values.shape),
            "dtype": str(values.dtype),
        }
    blind_review = (
        _make_blind_review_packet(candidate_features, crossing_rows, output_dir)
        if np.all(lower_valid & upper_valid) else None
    )
    return {
        "boundary_rule": boundary_rule,
        "boundary_definition": (
            "frozen teacher logit(2)-max(logits of all other classes) crossing along the q68-capped interpolation path"
            if boundary_rule == "global-top1" else
            "frozen teacher logit(2)-logit(baseline-selected rival) crossing along the q68-capped interpolation path"
        ),
        "training_or_human_label_claim": False,
        "rounds_requested": int(rounds),
        "crossing_rows": int(len(crossing_rows)),
        "refined_rows": int(np.sum(iterations > 0)),
        "geometry_stopped_rows": int(geometry_stopped.sum()),
        "final_bracket_valid_rows": int(np.sum(lower_valid & upper_valid)),
        "initial_bracket_width_p50": float(np.median(initial_widths)),
        "final_bracket_width_p50": float(np.median(final_widths)),
        "final_bracket_width_max": float(np.max(final_widths)),
        "full_372_way_posterior_at_positive_side": posterior_summary(lower_logits, lower_valid),
        "full_372_way_posterior_at_nonpositive_side": posterior_summary(upper_logits, upper_valid),
        "numeric_candidate_cache": {
            "artifacts": numeric_artifacts,
            "class_labels": labels,
            "endpoint_order": ["positive", "nonpositive"],
            "human_labels_available": False,
            "use": "reproducible frozen-teacher pseudo-target experiments; no assigned hard synthetic labels",
        },
        "rows": records,
        "sample_sheet": preview,
        "displayed_query_row_indices": [int(row_index) for row_index in display_rows],
        "blind_human_review": blind_review,
        "human_review_queue": {
            "path": str(queue_path.resolve()),
            "rows": int(len(display_rows)),
            "human_label_cells_filled": 0,
            "purpose": "teacher-labelled diagnostic sheet; use the separate blind queue for human boundary measurement",
        },
        "sample_sheet_scale_labels": {
            "B": "original UJI 2", "c+": "previous coarse positive margin",
            "n+": "refined positive side", "n-": "refined non-positive side",
            "q68": "maximum capped path scale",
        },
        "probability_note": "T=2 softmax summaries are frozen-teacher pseudo-target diagnostics, not calibrated human confidence.",
    }


def _run_audit(args) -> int:
    started = time.perf_counter()
    manifest_path = args.data_dir / "prepared_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "pass":
        raise ValueError("prepared data manifest did not pass")
    if manifest.get("input_policy", {}).get("crohme", "").lower().find("zero rows") < 0:
        raise ValueError("manifest does not prove CROHME exclusion")
    if manifest.get("current_checkpoint", {}).get("sha256") != _sha256(args.checkpoint):
        raise ValueError("prepared cache belongs to a different frozen checkpoint")
    if args.report_dir.exists() and any(args.report_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty report directory: {args.report_dir}")

    before_torch = _guard_commit("before_torch_import")
    if before_torch is None:
        return 78
    args.commit_headroom_gib_before_torch = before_torch

    # Load the heavyweight runtime only after the second commit-memory check.
    import torch
    from run_hwr_affine_distillation_experiment_v1 import SOURCE_IDS, _load_teacher

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    teacher, labels, _checkpoint = _load_teacher(args.checkpoint, device)
    if "2" not in labels:
        raise ValueError("frozen 372-class vocabulary has no exact '2' label")
    target_id = labels.index("2")

    features = np.load(args.data_dir / "train_features.npy", mmap_mode="r")
    targets = np.load(args.data_dir / "train_labels.npy", mmap_mode="r")
    sources = np.load(args.data_dir / "train_sources.npy", mmap_mode="r")
    if not (len(features) == len(targets) == len(sources)):
        raise ValueError("prepared training arrays are not row-aligned")
    if manifest.get("cache", {}).get("train", {}).get("rows") != len(features):
        raise ValueError("prepared training row count differs from the manifest")
    class_rows = np.flatnonzero(np.asarray(targets) == target_id)
    fit_rows = class_rows[np.asarray(sources[class_rows]) == SOURCE_IDS["hwrt"]]
    query_rows = class_rows[np.asarray(sources[class_rows]) == SOURCE_IDS["uji"]]
    raw_fit_rows, raw_query_rows = len(fit_rows), len(query_rows)
    fit_shapes = _single_stroke_rows(np.asarray(features[fit_rows]))
    query_shapes = _single_stroke_rows(np.asarray(features[query_rows]))
    fit_rows = fit_rows[fit_shapes]
    query_rows = query_rows[query_shapes]
    if len(fit_rows) < MIN_FIT_ROWS or not len(query_rows):
        raise ValueError(
            f"insufficient cross-source single-stroke support for '2': HWRT={len(fit_rows)}, UJI={len(query_rows)}"
        )
    if args.max_query_rows > 0:
        query_rows = query_rows[:args.max_query_rows]
    fit_features = np.asarray(features[fit_rows], dtype=np.float32).copy()
    originals = np.asarray(features[query_rows], dtype=np.float32).copy()
    tube = _fit_empirical_tube(fit_features, coverage=0.68)
    baseline_logits = _predict_logits(teacher, originals, torch, device, args.batch_size)
    rival_ids = None
    if args.path_mode == "cross-class":
        rival_logits = baseline_logits.copy()
        rival_logits[:, target_id] = -np.inf
        rival_ids = np.argmax(rival_logits, axis=1).astype(np.int64)
        rival_rows = np.flatnonzero(
            (np.asarray(sources) == SOURCE_IDS["hwrt"]) & (np.asarray(targets) != target_id)
        )
        rival_shapes = _single_stroke_rows(np.asarray(features[rival_rows]))
        rival_rows = rival_rows[rival_shapes]
        rival_features = np.asarray(features[rival_rows], dtype=np.float32).copy()
        donors, donor_selection = _select_empirical_rival_donors(
            rival_features, np.asarray(targets[rival_rows]), originals, rival_ids, tube,
        )
        donor_available = donor_selection["donor_available"]
        donor_training_rows = np.full(len(originals), -1, dtype=np.int64)
        donor_training_rows[donor_available] = rival_rows[
            donor_selection["donor_fit_row_indices"][donor_available]
        ]
        query_labels = [f"2>{labels[int(rival_id)]}" for rival_id in rival_ids]
    else:
        donors, donor_selection = _select_empirical_donors(fit_features, originals, tube)
        donor_available = np.ones(len(originals), dtype=bool)
        donor_training_rows = fit_rows[donor_selection["donor_fit_row_indices"]]
        query_labels = ["2"] * len(originals)
    views, transform_reports, valid_by_scale = [], [], []
    for scale in SCALES[1:]:
        view, diagnostic = _apply_tube(
            originals, donors, donor_selection["max_interpolation_fraction"], scale,
        )
        views.append(view)
        valid = diagnostic["valid"] & donor_available
        valid_by_scale.append(valid)
        transform_reports.append({
            "interpolation_fraction_of_query_q68_limit": scale,
            "generated_rows": len(view),
            "valid_rows": int(valid.sum()),
            "rejected_rows": int((~valid).sum()),
            "rejection_reasons": {
                "non_finite": int((~np.isfinite(view).all(axis=(1, 2))).sum()),
                "xy_out_of_range": int(((view[:, :, :2] < 0.0) | (view[:, :, :2] > 1.0)).any(axis=(1, 2)).sum()),
                "path_ratio_out_of_range": int((
                    (diagnostic["path_ratio"] < PATH_RATIO_RANGE[0])
                    | (diagnostic["path_ratio"] > PATH_RATIO_RANGE[1])
                ).sum()),
                "rival_donor_unavailable": int((~donor_available).sum()),
            },
            "rms_p50": float(np.quantile(diagnostic["rms"], 0.50)),
            "rms_p95": float(np.quantile(diagnostic["rms"], 0.95)),
            "max_point_displacement_p95": float(np.quantile(diagnostic["max_point_displacement"], 0.95)),
            "mean_path_ratio": float(diagnostic["path_ratio"].mean()),
            "all_non_xy_channels_unchanged": bool(diagnostic["non_xy_unchanged"].all()),
            "endpoint_displacement_p95": diagnostic["endpoint_displacement_p95"],
            "adjacent_displacement_step_p95": diagnostic["adjacent_displacement_step_p95"],
        })

    logits_by_scale = [baseline_logits]
    for view, valid in zip(views, valid_by_scale, strict=True):
        logits = np.full((len(view), len(labels)), np.nan, dtype=np.float32)
        if valid.any():
            logits[valid] = _predict_logits(teacher, view[valid], torch, device, args.batch_size)
        logits_by_scale.append(logits)
    complete_path = np.logical_and.reduce([np.ones(len(originals), dtype=bool), *valid_by_scale])
    summaries = []
    baseline_logits_matched = logits_by_scale[0][complete_path]
    baseline_top1_matched = np.argmax(baseline_logits_matched, axis=1) == target_id
    baseline_probability_matched = _softmax(baseline_logits_matched, DISTILL_TEMPERATURE)[:, target_id]
    for scale_index, (scale, logits) in enumerate(zip(SCALES, logits_by_scale, strict=True)):
        valid = np.ones(len(logits), dtype=bool) if scale_index == 0 else valid_by_scale[scale_index - 1]
        all_valid_logits = logits[valid]
        matched_logits = logits[complete_path]
        summaries.append({
            "interpolation_fraction_of_query_q68_limit": scale,
            "geometry_valid_rows": int(valid.sum()),
            "complete_matched_path_rows": int(complete_path.sum()),
            "teacher_all_valid": _summarize_logits(all_valid_logits, labels, target_id) if len(all_valid_logits) else {"rows": 0},
            "teacher_complete_matched_paths": _summarize_logits(matched_logits, labels, target_id) if len(matched_logits) else {"rows": 0},
        })
    fixed_rivals = rival_ids if args.boundary_rule == "fixed-pair" else None
    margins = [_boundary_margin(logits, target_id, fixed_rivals) for logits in logits_by_scale]
    paired_changes = []
    baseline_margin_matched = margins[0][complete_path]
    for scale, logits, margin in zip(SCALES, logits_by_scale, margins, strict=True):
        matched_logits = logits[complete_path]
        matched_top1 = np.argmax(matched_logits, axis=1) == target_id
        matched_probability = _softmax(matched_logits, DISTILL_TEMPERATURE)[:, target_id]
        paired_changes.append({
            "interpolation_fraction_of_query_q68_limit": scale,
            "paired_rows": int(complete_path.sum()),
            "top1_recovered_to_target": int(np.sum(~baseline_top1_matched & matched_top1)),
            "top1_regressed_from_target": int(np.sum(baseline_top1_matched & ~matched_top1)),
            "mean_target_probability_delta_t2": float((matched_probability - baseline_probability_matched).mean()),
            "mean_target_logit_margin_delta": float((margin[complete_path] - baseline_margin_matched).mean()),
        })
    baseline_correct = margins[0] > 0.0
    baseline_wrong = ~baseline_correct
    first_crossing_scale = np.full(len(originals), np.nan, dtype=np.float64)
    first_recovery_scale = np.full(len(originals), np.nan, dtype=np.float64)
    still_correct = baseline_correct.copy()
    still_wrong = baseline_wrong.copy()
    for scale, margin in zip(SCALES[1:], margins[1:], strict=True):
        newly_crossed = still_correct & (margin <= 0.0) & complete_path
        first_crossing_scale[newly_crossed] = scale
        still_correct &= margin > 0.0
        newly_recovered = still_wrong & (margin > 0.0) & complete_path
        first_recovery_scale[newly_recovered] = scale
        still_wrong &= margin <= 0.0
    crossed_from_correct = np.isfinite(first_crossing_scale)
    recovered_to_correct = np.isfinite(first_recovery_scale)
    crossings = {
        "query_rows": int(len(originals)),
        "complete_matched_paths": int(complete_path.sum()),
        "baseline_target_wrong_rows": int(baseline_wrong.sum()),
        "baseline_target_wrong_rows_on_complete_paths": int((baseline_wrong & complete_path).sum()),
        "baseline_correct_rows_on_complete_paths": int((baseline_correct & complete_path).sum()),
        "perturbation_induced_margin_crossings_from_correct": int(crossed_from_correct.sum()),
        "same_label_path_recoveries_from_wrong_to_correct": int(recovered_to_correct.sum()),
        "rows_by_crossing": [
            int(index) for index in np.flatnonzero(crossed_from_correct)
        ],
        "rows_by_recovery": [int(index) for index in np.flatnonzero(recovered_to_correct)],
    }

    args.report_dir.mkdir(parents=True, exist_ok=True)
    boundary_refinement = None
    if args.refine_boundaries:
        if args.path_mode != "cross-class" or rival_ids is None:
            raise ValueError("adaptive boundary refinement requires --path-mode cross-class")
        boundary_refinement = _refine_crossclass_boundaries(
            originals, donors, donor_selection["max_interpolation_fraction"], donor_available,
            rival_ids, target_id, labels, first_crossing_scale, margins,
            teacher, torch, device, args.batch_size, args.report_dir,
            boundary_rule=args.boundary_rule,
            query_training_rows=query_rows,
            donor_training_rows=donor_training_rows,
        )
    preview = _draw_sample_sheet(
        originals, views, query_labels, valid_by_scale,
        args.report_dir / "class2_empirical_tube_preview.png",
    )
    report = {
        "schema": "aiflow-hwr-probability-boundary-tube-audit/v2",
        "status": "training_pool_teacher_boundary_diagnostic_only",
        "scope": f"class '2'; single-stroke topology; HWRT train fit and UJI train query; path_mode={args.path_mode}",
        "method": {
            "shape_space": "median-centered low-rank PCA over aligned 128-point XY trajectories",
            "neighbor_correlation": (
                "convex interpolation between a real UJI class-2 query and a direction-compatible real HWRT sample from the query's frozen-teacher strongest rival class; no independent point noise"
                if args.path_mode == "cross-class"
                else "convex interpolation between a real UJI query trajectory and a direction-compatible real HWRT same-class trajectory; no independent point noise"
            ),
            "radius": "68th percentile of pairwise distances among low-rank standardized HWRT training shapes; this is a data-derived distance limit, not a Gaussian +/-1 SD region",
            "sampling": "sweep 25/50/75/100% of the capped query-to-donor interpolation path; intermediate points are synthetic and human plausibility is not established by convexity alone",
            "scales": list(SCALES),
            "softmax_temperature": DISTILL_TEMPERATURE,
            "boundary_rule": args.boundary_rule,
            "time_transform": "none; observed and ordinal timing channels are preserved byte-identically",
            "boundary": (
                "logit(2) minus the baseline-selected strongest rival-class logit <= 0; rival label is held fixed along each path"
                if args.boundary_rule == "fixed-pair"
                else "target logit minus maximum logit of all competing classes <= 0"
            ),
            "geometry_gates": {
                "path_length_ratio": list(PATH_RATIO_RANGE),
                "xy_bounds": "convex combinations of valid real inputs; no clipping",
                "path_length_ratio_is_only_morphology_gate": True,
            },
        },
        "provenance": {
            "prepared_manifest_sha256": _sha256(manifest_path),
            "audit_script_sha256": _sha256(Path(__file__)),
            "teacher_checkpoint_sha256": _sha256(args.checkpoint),
            "train_features_sha256": _sha256(args.data_dir / "train_features.npy"),
            "train_labels_sha256": _sha256(args.data_dir / "train_labels.npy"),
            "train_sources_sha256": _sha256(args.data_dir / "train_sources.npy"),
            "fit_source": "HWRT official train split only",
            "query_source": "UJI Pen v2 official writer-disjoint train split only",
            "fit_rows": int(len(fit_features)),
            "query_rows": int(len(originals)),
            "class2_single_stroke_rows_after_topology_filter": {
                "hwrt_fit": int(len(fit_features)),
                "uji_query": int(len(originals)),
            },
            "class2_rows_before_topology_filter": {
                "hwrt_train": int(raw_fit_rows),
                "uji_train": int(raw_query_rows),
            },
            "path_mode": args.path_mode,
            "boundary_rule": args.boundary_rule,
            "rival_donor_support_by_id": donor_selection.get("fit_support_by_rival_id", {}),
            "rival_donor_support_by_label": (
                {labels[int(key)]: int(value) for key, value in donor_selection["fit_support_by_rival_id"].items()}
                if args.path_mode == "cross-class" else {}
            ),
            "query_strongest_rival_counts": (
                {labels[int(key)]: int(np.sum(rival_ids == int(key))) for key in np.unique(rival_ids)}
                if rival_ids is not None else {}
            ),
            "rival_donor_available_rows": int(donor_available.sum()),
            "heldout_rows_read": 0,
            "crohme_rows": 0,
        },
        "empirical_tube_fit": {
            "pairwise_distance_quantile": tube["coverage"],
            "pairwise_q68_radius_mahalanobis": tube["pairwise_q68_radius"],
            "fit_rows": tube["fit_rows"],
            "component_count": tube["component_count"],
            "explained_variance_ratio": tube["explained_variance_ratio"],
            "pairwise_distance_p50": float(np.quantile(tube["pairwise_radii"], 0.50)),
            "pairwise_distance_p68": float(np.quantile(tube["pairwise_radii"], 0.68)),
            "pairwise_distance_p95": float(np.quantile(tube["pairwise_radii"], 0.95)),
        },
        "transform_by_scale": transform_reports,
        "teacher_probability_boundary_by_scale": summaries,
        "paired_teacher_change_on_complete_paths": paired_changes,
        "teacher_margin_boundary": {
            "definition": (
                "first tested interpolation fraction where logit(2) minus the query's baseline-selected rival logit is non-positive; rival is held fixed along path"
                if args.boundary_rule == "fixed-pair"
                else "first tested interpolation fraction where logit(2) minus strongest rival logit is non-positive, among baseline-top1-correct rows with a complete valid path"
            ),
            "baseline_top1_correct_rows": int(baseline_correct.sum()),
            "baseline_top1_wrong_rows": int(baseline_wrong.sum()),
            "complete_matched_paths": crossings["complete_matched_paths"],
            "baseline_correct_rows_on_complete_paths": crossings["baseline_correct_rows_on_complete_paths"],
            "baseline_wrong_rows_on_complete_paths": crossings["baseline_target_wrong_rows_on_complete_paths"],
            "correct_to_wrong_crossings": int(crossed_from_correct.sum()),
            "first_wrong_to_correct_recoveries": int(recovered_to_correct.sum()),
            "first_crossing_count_by_fraction": {
                str(scale): int(np.sum(first_crossing_scale == scale)) for scale in SCALES[1:]
            },
            "first_recovery_count_by_fraction": {
                str(scale): int(np.sum(first_recovery_scale == scale)) for scale in SCALES[1:]
            },
            "rows_by_crossing": [int(index) for index in np.flatnonzero(crossed_from_correct)],
            "rows_by_recovery": [int(index) for index in np.flatnonzero(recovered_to_correct)],
            "interpretation": (
                "cross-label real-to-real path; crossing estimates only the frozen teacher's boundary, not a human decision threshold or valid soft label"
                if args.path_mode == "cross-class"
                else "same-label real-to-real path; recovery shows one direction of the teacher boundary, not a human decision threshold"
            ),
            "training_pool_boundary_only": True,
        },
        "adaptive_teacher_boundary_refinement": boundary_refinement,
        "sample_sheet": preview,
        "sample_sheet_scale_labels": {"B": "original UJI query", ".25": "25% of capped q68 path", ".50": "50%", ".75": "75%", "q68": "maximum capped q68 path"},
        "interpretation_boundary": (
            "training-pool diagnostic only; no independent performance estimate, no student training, "
            "no model selection, and no product promotion"
        ),
        "runtime": {
            "device": str(device),
            "commit_headroom_gib_after_numpy": args.commit_headroom_gib_after_numpy,
            "commit_headroom_gib_before_model_load": args.commit_headroom_gib_before_torch,
            "commit_headroom_safety_floor_gib": MIN_COMMIT_HEADROOM_GIB,
        },
        "seconds": time.perf_counter() - started,
    }
    (args.report_dir / "probability_boundary_tube_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps({
        "event": "probability_boundary_tube_audit_complete",
        "report": str((args.report_dir / "probability_boundary_tube_audit.json").resolve()),
        "fit_rows": len(fit_features), "query_rows": len(originals),
        "path_mode": args.path_mode,
        "boundary_rule": args.boundary_rule,
        "pairwise_q68_radius_mahalanobis": tube["pairwise_q68_radius"],
        "correct_to_wrong_crossings": report["teacher_margin_boundary"]["correct_to_wrong_crossings"],
        "wrong_to_correct_recoveries": report["teacher_margin_boundary"]["first_wrong_to_correct_recoveries"],
        "refined_boundary_rows": (
            boundary_refinement["refined_rows"] if boundary_refinement is not None else 0
        ),
        "crohme_rows": 0,
    }, ensure_ascii=False), flush=True)
    return 0


def _self_test() -> int:
    diagnostic_logits = np.asarray([[5.0, 4.0, 6.0], [7.0, 4.0, 6.0]], dtype=np.float32)
    global_margin = _boundary_margin(diagnostic_logits, 0)
    pair_margin = _boundary_margin(diagnostic_logits, 0, np.asarray([1, 1]))
    if not np.array_equal(global_margin, [-1.0, 1.0]) or not np.array_equal(pair_margin, [1.0, 3.0]):
        raise AssertionError("global boundary must detect a third class overtaking the target")
    count = 40
    u = np.linspace(0.0, 1.0, 128, dtype=np.float32)
    base = np.zeros((count, 128, 5), dtype=np.float32)
    base[:, :, 0] = 0.18 + 0.64 * u[None, :]
    base[:, :, 1] = 0.2 + 0.52 * (1.0 - np.abs(2.0 * u[None, :] - 1.0))
    base[:, 0, 3] = 1.0
    base[:, :, 2] = np.linspace(0.0, 1.0, 128, dtype=np.float32)[None, :]
    base[:, :, 4] = 1.0
    smooth = np.sin(np.pi * u)
    for index in range(count):
        base[index, :, 0] += 0.008 * np.sin((1 + index % 3) * np.pi * u) * smooth
        base[index, :, 1] += 0.009 * np.cos((1 + index % 4) * np.pi * u) * smooth
    fit, originals = base[:24].copy(), base[24:32].copy()
    tube = _fit_empirical_tube(fit)
    donors, selection = _select_empirical_donors(fit, originals, tube)
    zero, zero_diag = _apply_tube(originals, donors, selection["max_interpolation_fraction"], 0.0)
    if not np.array_equal(zero, originals) or not zero_diag["valid"].all():
        raise AssertionError("scale-zero output is not an exact identity")
    first, first_diag = _apply_tube(originals, donors, selection["max_interpolation_fraction"], 0.5)
    second, second_diag = _apply_tube(originals, donors, selection["max_interpolation_fraction"], 0.5)
    if not np.array_equal(first, second) or not first_diag["valid"].all() or not second_diag["valid"].all():
        raise AssertionError("fixed directions must yield deterministic, valid views")
    if not np.array_equal(first[:, :, 2:], originals[:, :, 2:]):
        raise AssertionError("time, stroke-start, and observed channels changed")
    if float(np.mean(np.linalg.norm(np.diff(first[:, :, :2], axis=1), axis=2))) <= 0.0:
        raise AssertionError("spatial path collapsed")
    if not np.isfinite(first_diag["adjacent_displacement_step_p95"]):
        raise AssertionError("neighbor-correlated displacement diagnostic is not finite")
    if tube["pairwise_q68_radius"] <= 0.0 or tube["component_count"] < 2:
        raise AssertionError("empirical tube fit is degenerate")
    rival_fit = fit.copy()
    rival_fit[:, :, 1] += 0.018 * np.sin(2.0 * np.pi * u)[None, :]
    rival_donors, rival_selection = _select_empirical_rival_donors(
        rival_fit, np.full(len(rival_fit), 17, dtype=np.int64),
        originals, np.full(len(originals), 17, dtype=np.int64), tube,
    )
    if not rival_selection["donor_available"].all():
        raise AssertionError("class-conditioned donor selection lost a supported rival row")
    if not np.array_equal(rival_selection["donor_fit_row_indices"].shape, (len(originals),)):
        raise AssertionError("class-conditioned donor row indices are malformed")
    if np.array_equal(rival_donors[:, :, :2], originals[:, :, :2]):
        raise AssertionError("class-conditioned donor selection returned only the original queries")
    bad = base.copy(); bad[:, 20, 3] = 1.0
    try:
        _fit_empirical_tube(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("mixed stroke topology must be rejected")
    print(json.dumps({
        "event": "probability_boundary_tube_self_test_pass",
        "fit_rows": tube["fit_rows"],
        "components": tube["component_count"],
        "q68_radius": tube["pairwise_q68_radius"],
        "cross_class_donor_rows": int(rival_selection["donor_available"].sum()),
        "crohme_rows": 0,
    }, ensure_ascii=False), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("self-test", "audit"), required=True)
    parser.add_argument(
        "--path-mode", choices=("same-class", "cross-class"), default="same-class",
        help="probe same-label writer variation or interpolate toward each UJI query's frozen-teacher strongest rival class",
    )
    parser.add_argument(
        "--refine-boundaries", action="store_true",
        help="for cross-class paths, bisect geometry-valid frozen-teacher crossings and save near-boundary diagnostics",
    )
    parser.add_argument(
        "--boundary-rule", choices=("global-top1", "fixed-pair"), default="global-top1",
        help="global-top1 follows the best of all competing classes; fixed-pair reproduces the previous pairwise diagnostic",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-query-rows", type=int, default=0, help="0 audits all eligible UJI train '2' rows")
    args = parser.parse_args()
    if args.report_dir == DEFAULT_REPORT_DIR:
        if args.path_mode == "cross-class":
            args.report_dir = (
                DEFAULT_CROSS_CLASS_REFINED_REPORT_DIR if args.refine_boundaries
                else DEFAULT_CROSS_CLASS_REPORT_DIR
            )
            if args.boundary_rule == "global-top1":
                args.report_dir = args.report_dir.with_name(args.report_dir.name + "_global_top1")
    if args.refine_boundaries and args.path_mode != "cross-class":
        parser.error("--refine-boundaries requires --path-mode cross-class")
    if args.boundary_rule == "fixed-pair" and args.path_mode != "cross-class":
        parser.error("--boundary-rule fixed-pair requires --path-mode cross-class")
    if args.batch_size < 1 or args.max_query_rows < 0:
        parser.error("batch size must be positive and max query rows non-negative")
    before_numpy = _guard_commit("before_numpy_import")
    if before_numpy is None:
        return 78
    _load_numpy()
    after_numpy = _guard_commit("after_numpy_import")
    if after_numpy is None:
        return 78
    if args.mode == "self-test":
        return _self_test()
    if not (args.data_dir / "prepared_manifest.json").is_file():
        raise FileNotFoundError(args.data_dir / "prepared_manifest.json")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    args.commit_headroom_gib_after_numpy = after_numpy
    return _run_audit(args)


if __name__ == "__main__":
    raise SystemExit(main())
