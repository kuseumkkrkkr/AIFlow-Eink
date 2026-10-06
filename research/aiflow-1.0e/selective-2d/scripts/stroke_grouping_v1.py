#!/usr/bin/env python3
"""Commercial-safe raw-stroke grouping for AIFlow Math Ink 1.0.

The lattice is deliberately label-free.  It preserves every source stroke,
offers plausible multi-stroke glyph groups, and selects an exact-cover
partition from geometry scores only.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np


SCHEMA = "aiflow-stroke-grouping-selector/v1"
FEATURE_NAMES = (
    "stroke_count", "point_count_log", "width_ref", "height_ref", "aspect_log",
    "temporal_span", "temporal_contiguous", "singleton", "has_spatial_pair",
    "pair_gap_mean_ref", "pair_gap_max_ref", "x_overlap_mean", "y_overlap_mean",
    "stroke_width_mean_ref", "stroke_width_std_ref", "stroke_height_mean_ref",
    "stroke_height_std_ref", "path_length_ref", "endpoint_gap_mean_ref",
    "candidate_cx_formula", "candidate_cy_formula", "candidate_width_formula",
)
DEFAULT_LATTICE_CONFIG = {
    "temporal_window": 6,
    "spatial_neighbors": 4,
    "max_long_group_width_fraction": 1.0,
}


@dataclass(frozen=True)
class GroupingResultV1:
    groups: tuple[tuple[int, ...], ...]
    candidate_count: int
    model_version: str


def _xy(point: Any) -> tuple[float, float]:
    if isinstance(point, dict):
        return float(point["x"]), float(point["y"])
    return float(point[0]), float(point[1])


def _stroke_box(stroke: dict[str, Any]) -> tuple[float, float, float, float]:
    points = [_xy(point) for point in stroke.get("points") or []]
    if not points:
        raise ValueError("stroke grouping requires non-empty strokes")
    xs, ys = zip(*points, strict=True)
    if not all(math.isfinite(value) for value in (*xs, *ys)):
        raise ValueError("stroke grouping requires finite coordinates")
    return min(xs), min(ys), max(xs), max(ys)


def _box_gap(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    dx = max(first[0] - second[2], second[0] - first[2], 0.0)
    dy = max(first[1] - second[3], second[1] - first[3], 0.0)
    return math.hypot(dx, dy)


def _overlap(first_low: float, first_high: float, second_low: float, second_high: float) -> float:
    overlap = max(0.0, min(first_high, second_high) - max(first_low, second_low))
    denominator = max(min(first_high - first_low, second_high - second_low), 1e-6)
    return overlap / denominator


def _group_box(indices: frozenset[int], boxes: Sequence[tuple[float, float, float, float]]) -> dict[str, float]:
    return {
        "left": min(boxes[index][0] for index in indices),
        "top": min(boxes[index][1] for index in indices),
        "right": max(boxes[index][2] for index in indices),
        "bottom": max(boxes[index][3] for index in indices),
    }


def build_lattice(
    strokes: Sequence[dict[str, Any]], *, temporal_window: int = 6, spatial_neighbors: int = 4,
    max_long_group_width_fraction: float = 1.0,
) -> list[dict[str, Any]]:
    """Build singleton, bounded temporal, and nearest-spatial group candidates."""
    if temporal_window < 1 or spatial_neighbors < 0 or not 0.0 < max_long_group_width_fraction <= 1.0:
        raise ValueError("invalid grouping lattice configuration")
    ordered = sorted(strokes, key=lambda row: int(row.get("order", 0)))
    if not ordered or any(not row.get("points") for row in ordered):
        raise ValueError("stroke grouping requires a non-empty formula")
    orders = [int(row.get("order", 0)) for row in ordered]
    if len(set(orders)) != len(orders):
        raise ValueError("stroke orders must be unique")
    boxes = [_stroke_box(stroke) for stroke in ordered]
    formula_width = max(max(box[2] for box in boxes) - min(box[0] for box in boxes), 1e-6)
    evidence: dict[frozenset[int], set[str]] = {}

    def add(indices: frozenset[int], reason: str) -> None:
        evidence.setdefault(indices, set()).add(reason)

    for index in range(len(ordered)):
        add(frozenset({index}), "singleton")
    for start in range(len(ordered)):
        for length in range(2, min(temporal_window, len(ordered) - start) + 1):
            group = frozenset(range(start, start + length))
            width = _group_box(group, boxes)["right"] - _group_box(group, boxes)["left"]
            if length <= 3 or width / formula_width <= max_long_group_width_fraction:
                add(group, f"temporal:{length}")
    for index, box in enumerate(boxes):
        neighbors = sorted(
            (other for other in range(len(boxes)) if other != index),
            key=lambda other: (_box_gap(box, boxes[other]), abs(other - index), other),
        )[:spatial_neighbors]
        for other in neighbors:
            add(frozenset({index, other}), "spatial_pair")

    candidates = []
    for indices, reasons in evidence.items():
        box = _group_box(indices, boxes)
        candidates.append({
            "source_indices": sorted(indices),
            "box": box,
            "evidence": sorted(reasons),
            "candidate_id": "g:" + ",".join(str(index) for index in sorted(indices)),
        })
    candidates.sort(key=lambda row: (row["box"]["left"], len(row["source_indices"]), row["source_indices"]))
    return candidates


def candidate_features(candidates: Sequence[dict[str, Any]], strokes: Sequence[dict[str, Any]]) -> np.ndarray:
    ordered = sorted(strokes, key=lambda row: int(row.get("order", 0)))
    boxes = [_stroke_box(stroke) for stroke in ordered]
    extents = [max(box[2] - box[0], box[3] - box[1]) for box in boxes]
    reference = float(np.median([value for value in extents if value > 1e-6])) if any(value > 1e-6 for value in extents) else 1.0
    formula_left = min(box[0] for box in boxes); formula_top = min(box[1] for box in boxes)
    formula_right = max(box[2] for box in boxes); formula_bottom = max(box[3] for box in boxes)
    formula_width = max(formula_right - formula_left, 1e-6)
    formula_height = max(formula_bottom - formula_top, 1e-6)
    paths = []
    endpoints = []
    for stroke in ordered:
        points = [_xy(point) for point in stroke["points"]]
        paths.append(sum(math.dist(first, second) for first, second in zip(points, points[1:])))
        endpoints.append((points[0], points[-1]))
    rows = []
    for candidate in candidates:
        indices = [int(index) for index in candidate["source_indices"]]
        selected = [boxes[index] for index in indices]
        box = candidate["box"]
        width = float(box["right"] - box["left"]); height = float(box["bottom"] - box["top"])
        gaps = []; x_overlaps = []; y_overlaps = []; endpoint_gaps = []
        for offset, first in enumerate(indices):
            for second in indices[offset + 1:]:
                gaps.append(_box_gap(boxes[first], boxes[second]) / reference)
                x_overlaps.append(_overlap(boxes[first][0], boxes[first][2], boxes[second][0], boxes[second][2]))
                y_overlaps.append(_overlap(boxes[first][1], boxes[first][3], boxes[second][1], boxes[second][3]))
                endpoint_gaps.append(min(math.dist(a, b) for a in endpoints[first] for b in endpoints[second]) / reference)
        widths = [(value[2] - value[0]) / reference for value in selected]
        heights = [(value[3] - value[1]) / reference for value in selected]
        reasons = set(candidate.get("evidence") or [])
        rows.append([
            len(indices), math.log1p(sum(len(ordered[index]["points"]) for index in indices)),
            width / reference, height / reference, math.log(max(width, 1e-6) / max(height, 1e-6)),
            max(indices) - min(indices) + 1, float(max(indices) - min(indices) + 1 == len(indices)),
            float(len(indices) == 1), float("spatial_pair" in reasons),
            float(np.mean(gaps)) if gaps else 0.0, max(gaps, default=0.0),
            float(np.mean(x_overlaps)) if x_overlaps else 0.0,
            float(np.mean(y_overlaps)) if y_overlaps else 0.0,
            float(np.mean(widths)), float(np.std(widths)), float(np.mean(heights)), float(np.std(heights)),
            sum(paths[index] for index in indices) / reference,
            float(np.mean(endpoint_gaps)) if endpoint_gaps else 0.0,
            ((box["left"] + box["right"]) / 2.0 - formula_left) / formula_width,
            ((box["top"] + box["bottom"]) / 2.0 - formula_top) / formula_height,
            width / formula_width,
        ])
    values = np.asarray(rows, dtype=np.float32)
    if values.shape != (len(candidates), len(FEATURE_NAMES)) or not np.isfinite(values).all():
        raise AssertionError("invalid stroke-group feature matrix")
    return values


def mask_candidate_features(
    features: np.ndarray, feature_names: Iterable[str], *, value: float = 0.0,
) -> np.ndarray:
    """Return an inference-only feature mask; callers must opt in explicitly."""
    values = np.asarray(features, dtype=np.float32).copy()
    if values.ndim != 2 or values.shape[1] != len(FEATURE_NAMES):
        raise ValueError("candidate feature matrix does not match the grouping contract")
    if not math.isfinite(float(value)):
        raise ValueError("feature mask value must be finite")
    raw_names = (feature_names,) if isinstance(feature_names, str) else feature_names
    requested = tuple(dict.fromkeys(str(name) for name in raw_names))
    unknown = sorted(set(requested) - set(FEATURE_NAMES))
    if unknown:
        raise ValueError(f"unknown candidate features for masking: {unknown}")
    for name in requested:
        values[:, FEATURE_NAMES.index(name)] = float(value)
    return values


def group_shape_features(
    strokes: Sequence[dict[str, Any]], indices: Sequence[int],
) -> dict[str, float]:
    """Return dimensionless shape evidence used by conservative slot guards."""
    ordered = sorted(strokes, key=lambda row: int(row.get("order", 0)))
    selected = [ordered[int(index)] for index in indices]
    if not selected:
        raise ValueError("group shape requires at least one stroke")
    arrays = [np.asarray([_xy(point) for point in stroke["points"]], dtype=np.float64) for stroke in selected]
    points = np.concatenate(arrays, axis=0)
    left, top = points.min(axis=0); right, bottom = points.max(axis=0)
    width = float(right - left); height = float(bottom - top)
    diagonal = max(math.hypot(width, height), 1e-8)
    path = sum(float(np.linalg.norm(np.diff(stroke, axis=0), axis=1).sum()) for stroke in arrays)
    longest = max(arrays, key=len)
    direction = longest[-1] - longest[0]
    return {
        "left": float(left), "top": float(top), "right": float(right), "bottom": float(bottom),
        "width": width, "height": height,
        "cx": float((left + right) / 2.0), "cy": float((top + bottom) / 2.0),
        "aspect_log": float(math.log((width + 1e-6) / (height + 1e-6))),
        "path_over_diag": path / diagonal,
        "direction_x": float(direction[0] / diagonal),
        "direction_y": float(direction[1] / diagonal),
        "stroke_count": float(len(arrays)),
        "point_count_log": float(math.log1p(len(points))),
    }


def _normalized_group_paths(
    strokes: Sequence[dict[str, Any]], indices: Sequence[int], *, points: int,
) -> list[np.ndarray]:
    ordered = sorted(strokes, key=lambda row: int(row.get("order", 0)))
    arrays = [
        np.asarray([_xy(point) for point in ordered[int(index)]["points"]], dtype=np.float64)
        for index in indices
    ]
    if not arrays or points < 2:
        raise ValueError("normalized group paths require strokes and at least two samples")
    combined = np.concatenate(arrays, axis=0)
    low = combined.min(axis=0); high = combined.max(axis=0)
    center = (low + high) / 2.0
    scale = max(float((high - low).max()), 1e-8)
    output = []
    for array in arrays:
        normalized = (array - center) / scale
        if len(normalized) == 1:
            output.append(np.repeat(normalized, points, axis=0))
            continue
        distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(normalized, axis=0), axis=1))]
        if float(distance[-1]) <= 1e-10:
            output.append(np.repeat(normalized[:1], points, axis=0))
            continue
        target = np.linspace(0.0, float(distance[-1]), points)
        output.append(np.column_stack([
            np.interp(target, distance, normalized[:, axis]) for axis in range(2)
        ]))
    return output


def normalized_group_dtw_distance(
    strokes: Sequence[dict[str, Any]], first: Sequence[int], second: Sequence[int],
    *, points_per_stroke: int = 32, band: int = 8,
) -> float:
    """Compare two same-stroke-count glyphs without using timing or labels."""
    if len(first) != len(second) or not first:
        return math.inf
    if band < 0:
        raise ValueError("DTW band must be non-negative")
    left = _normalized_group_paths(strokes, first, points=points_per_stroke)
    right = _normalized_group_paths(strokes, second, points=points_per_stroke)
    distances = []
    for left_path, right_path in zip(left, right, strict=True):
        rows = len(left_path); columns = len(right_path)
        cost = np.full((rows + 1, columns + 1), math.inf, dtype=np.float64)
        steps = np.zeros((rows + 1, columns + 1), dtype=np.int32)
        cost[0, 0] = 0.0
        for row in range(1, rows + 1):
            for column in range(max(1, row - band), min(columns, row + band) + 1):
                choices = (
                    (cost[row - 1, column], steps[row - 1, column]),
                    (cost[row, column - 1], steps[row, column - 1]),
                    (cost[row - 1, column - 1], steps[row - 1, column - 1]),
                )
                previous_cost, previous_steps = min(choices, key=lambda value: value[0])
                cost[row, column] = previous_cost + float(
                    np.linalg.norm(left_path[row - 1] - right_path[column - 1])
                )
                steps[row, column] = previous_steps + 1
        if not math.isfinite(float(cost[rows, columns])):
            return math.inf
        distances.append(float(cost[rows, columns]) / max(int(steps[rows, columns]), 1))
    return float(np.mean(distances))


def select_partition(
    candidates: Sequence[dict[str, Any]], scores: Sequence[float], stroke_count: int,
    *, group_bias: float = 0.0, beam_width: int = 256, options_per_stroke: int = 64,
) -> list[frozenset[int]]:
    """Select a score-maximizing exact cover; no stroke may be lost or duplicated."""
    if len(candidates) != len(scores):
        raise ValueError("candidate and score counts differ")
    if stroke_count < 1:
        return []
    full_mask = (1 << stroke_count) - 1
    prepared = []
    by_stroke: dict[int, list[int]] = {index: [] for index in range(stroke_count)}
    for candidate, score in zip(candidates, scores, strict=True):
        group = frozenset(int(value) for value in candidate["source_indices"])
        if not group or min(group) < 0 or max(group) >= stroke_count:
            continue
        mask = sum(1 << index for index in group)
        prepared.append((mask, group, float(score) + float(group_bias)))
        position = len(prepared) - 1
        for index in group:
            by_stroke[index].append(position)
    for index in by_stroke:
        by_stroke[index].sort(key=lambda value: prepared[value][2], reverse=True)
        by_stroke[index] = by_stroke[index][:options_per_stroke]
    beams: dict[int, tuple[float, tuple[frozenset[int], ...]]] = {0: (0.0, ())}
    for _ in range(stroke_count):
        expanded: dict[int, tuple[float, tuple[frozenset[int], ...]]] = {}
        for used, (total, groups) in beams.items():
            if used == full_mask:
                expanded[used] = max(expanded.get(used, (-math.inf, ())), (total, groups), key=lambda row: row[0])
                continue
            first = next(index for index in range(stroke_count) if not used & (1 << index))
            for position in by_stroke[first]:
                mask, group, score = prepared[position]
                if used & mask:
                    continue
                proposal = (total + score, groups + (group,))
                new_mask = used | mask
                if new_mask not in expanded or proposal[0] > expanded[new_mask][0]:
                    expanded[new_mask] = proposal
        if not expanded:
            raise ValueError("no complete grouping partition is reachable")
        beams = dict(sorted(expanded.items(), key=lambda row: row[1][0], reverse=True)[:beam_width])
        if set(beams) == {full_mask}:
            break
    if full_mask not in beams:
        raise ValueError("grouping beam did not find an exact cover")
    return list(beams[full_mask][1])


def enumerate_partitions(
    candidates: Sequence[dict[str, Any]], scores: Sequence[float], stroke_count: int,
    *, top_n: int = 32, group_bias: float = 0.0, beam_width: int = 2048,
    options_per_stroke: int = 64,
) -> list[tuple[float, tuple[frozenset[int], ...]]]:
    """Return score-ranked exact covers without emitting permutation duplicates."""
    if top_n < 1 or beam_width < top_n or len(candidates) != len(scores):
        raise ValueError("invalid n-best grouping request")
    if stroke_count < 1:
        return [(0.0, ())]
    full_mask = (1 << stroke_count) - 1
    prepared = []
    by_stroke: dict[int, list[int]] = {index: [] for index in range(stroke_count)}
    for candidate, score in zip(candidates, scores, strict=True):
        group = frozenset(int(value) for value in candidate["source_indices"])
        if not group or min(group) < 0 or max(group) >= stroke_count:
            continue
        mask = sum(1 << index for index in group)
        prepared.append((mask, group, float(score) + float(group_bias)))
        position = len(prepared) - 1
        for index in group:
            by_stroke[index].append(position)
    for index in by_stroke:
        by_stroke[index].sort(key=lambda value: prepared[value][2], reverse=True)
        by_stroke[index] = by_stroke[index][:options_per_stroke]
    beams = [(0, 0.0, ())]
    completed: dict[tuple[tuple[int, ...], ...], float] = {}
    for _ in range(stroke_count):
        expanded = []
        for used, total, groups in beams:
            if used == full_mask:
                key = tuple(tuple(sorted(group)) for group in groups)
                completed[key] = max(completed.get(key, -math.inf), total)
                continue
            first = next(index for index in range(stroke_count) if not used & (1 << index))
            for position in by_stroke[first]:
                mask, group, score = prepared[position]
                if not used & mask:
                    expanded.append((used | mask, total + score, groups + (group,)))
        if not expanded:
            break
        expanded.sort(key=lambda row: row[1], reverse=True)
        beams = expanded[:beam_width]
        for used, total, groups in beams:
            if used == full_mask:
                key = tuple(tuple(sorted(group)) for group in groups)
                completed[key] = max(completed.get(key, -math.inf), total)
        if len(completed) >= top_n and all(used == full_mask for used, _total, _groups in beams):
            break
    ranked = sorted(completed.items(), key=lambda row: row[1], reverse=True)[:top_n]
    return [
        (score, tuple(frozenset(group) for group in groups))
        for groups, score in ranked
    ]


def enumerate_partitions_exact(
    candidates: Sequence[dict[str, Any]], scores: Sequence[float], stroke_count: int,
    *, top_n: int = 32, group_bias: float = 0.0,
    options_per_stroke: int = 64, max_strokes: int = 12,
) -> list[tuple[float, tuple[frozenset[int], ...]]]:
    """Return exact top-N additive-score covers within explicit search caps.

    Unlike ``enumerate_partitions``, this k-best dynamic program does not beam
    prune partial covers. It is intended for bounded research/shadow searches;
    the result is exact over the candidates retained by ``options_per_stroke``.
    """
    if (
        top_n < 1 or top_n > 32 or options_per_stroke < 1
        or max_strokes < 1 or stroke_count > max_strokes
        or len(candidates) != len(scores)
    ):
        raise ValueError("invalid exact n-best grouping request")
    if stroke_count < 1:
        return [(0.0, ())]

    full_mask = (1 << stroke_count) - 1
    unique: dict[tuple[int, ...], float] = {}
    for candidate, raw_score in zip(candidates, scores, strict=True):
        score = float(raw_score) + float(group_bias)
        if not math.isfinite(score):
            raise ValueError("exact partition scorer produced a non-finite score")
        raw_group = tuple(int(value) for value in candidate["source_indices"])
        if len(set(raw_group)) != len(raw_group):
            continue
        group = tuple(sorted(raw_group))
        if not group or group[0] < 0 or group[-1] >= stroke_count:
            continue
        unique[group] = max(unique.get(group, -math.inf), score)

    prepared = []
    by_stroke: dict[int, list[int]] = {index: [] for index in range(stroke_count)}
    for group, score in unique.items():
        mask = sum(1 << index for index in group)
        position = len(prepared)
        prepared.append((mask, frozenset(group), score))
        for index in group:
            by_stroke[index].append(position)
    for index in by_stroke:
        by_stroke[index].sort(
            key=lambda position: (
                -prepared[position][2], tuple(sorted(prepared[position][1])),
            ),
        )
        by_stroke[index] = by_stroke[index][:options_per_stroke]

    # Each transition adds the group containing the first uncovered stroke,
    # so every partition has one canonical path and every new mask is larger.
    states: list[list[tuple[float, tuple[frozenset[int], ...]]]] = [
        [] for _ in range(full_mask + 1)
    ]
    states[0] = [(0.0, ())]
    for used in range(full_mask):
        if not states[used]:
            continue
        first = next(index for index in range(stroke_count) if not used & (1 << index))
        for total, groups in states[used]:
            for position in by_stroke[first]:
                mask, group, score = prepared[position]
                if used & mask:
                    continue
                next_used = used | mask
                proposal = (total + score, groups + (group,))
                bucket = states[next_used]
                key = tuple(tuple(sorted(item)) for item in proposal[1])
                replaced = False
                for index, (old_score, old_groups) in enumerate(bucket):
                    old_key = tuple(tuple(sorted(item)) for item in old_groups)
                    if old_key == key:
                        if proposal[0] > old_score:
                            bucket[index] = proposal
                        replaced = True
                        break
                if not replaced:
                    bucket.append(proposal)
                bucket.sort(key=lambda row: (
                    -row[0], tuple(tuple(sorted(item)) for item in row[1]),
                ))
                del bucket[top_n:]

    if not states[full_mask]:
        raise ValueError("no exact-cover partition is reachable within exact-search caps")
    return states[full_mask][:top_n]


class StrokeGroupingSelectorV1:
    def __init__(self, model: Any, *, group_bias: float, model_version: str, lattice_config: dict[str, Any]) -> None:
        self.model = model
        self.group_bias = float(group_bias)
        self.model_version = str(model_version)
        required = set(DEFAULT_LATTICE_CONFIG)
        if set(lattice_config) != required:
            raise ValueError("stroke grouping lattice configuration mismatch")
        self.lattice_config = {
            "temporal_window": int(lattice_config["temporal_window"]),
            "spatial_neighbors": int(lattice_config["spatial_neighbors"]),
            "max_long_group_width_fraction": float(
                lattice_config["max_long_group_width_fraction"]
            ),
        }

    @classmethod
    def from_artifact(cls, path: Path) -> "StrokeGroupingSelectorV1":
        payload = joblib.load(path)
        if payload.get("schema") != SCHEMA or tuple(payload.get("feature_names") or ()) != FEATURE_NAMES:
            raise ValueError("stroke grouping artifact contract mismatch")
        return cls(
            payload["model"], group_bias=float(payload["group_bias"]),
            model_version=str(payload["model_version"]), lattice_config=dict(payload["lattice_config"]),
        )

    def group(
        self, strokes: Sequence[dict[str, Any]], *,
        neutralized_features: Iterable[str] = (),
    ) -> GroupingResultV1:
        candidates = build_lattice(strokes, **self.lattice_config)
        raw_names = (neutralized_features,) if isinstance(neutralized_features, str) else neutralized_features
        masked_features = tuple(dict.fromkeys(str(name) for name in raw_names))
        feature_matrix = candidate_features(candidates, strokes)
        if masked_features:
            feature_matrix = mask_candidate_features(feature_matrix, masked_features)
        probability = self.model.predict_proba(feature_matrix)[:, 1]
        logits = np.log(np.clip(probability, 1e-6, 1 - 1e-6) / np.clip(1 - probability, 1e-6, 1))
        selected = select_partition(candidates, logits, len(strokes), group_bias=self.group_bias)
        boxes = {frozenset(row["source_indices"]): row["box"] for row in candidates}
        ordered = sorted(selected, key=lambda group: (boxes[group]["left"], boxes[group]["top"], min(group)))
        version = self.model_version
        if masked_features:
            version = f"{version}+mask({','.join(masked_features)})"
        return GroupingResultV1(
            tuple(tuple(sorted(group)) for group in ordered), len(candidates), version,
        )


def _self_test() -> None:
    strokes = [
        {"order": 0, "points": [{"x": 0, "y": 0}, {"x": 10, "y": 10}]},
        {"order": 1, "points": [{"x": 10, "y": 0}, {"x": 0, "y": 10}]},
        {"order": 2, "points": [{"x": 40, "y": 0}, {"x": 40, "y": 10}]},
    ]
    candidates = build_lattice(strokes)
    features = candidate_features(candidates, strokes)
    assert features.shape == (len(candidates), len(FEATURE_NAMES))
    masked_features = mask_candidate_features(features, ("singleton",))
    assert np.array_equal(features, candidate_features(candidates, strokes))
    assert np.all(masked_features[:, FEATURE_NAMES.index("singleton")] == 0.0)
    assert np.array_equal(
        masked_features[:, FEATURE_NAMES.index("stroke_count")],
        features[:, FEATURE_NAMES.index("stroke_count")],
    )
    try:
        mask_candidate_features(features, ("not_a_group_feature",))
    except ValueError:
        pass
    else:
        raise AssertionError("feature mask accepted an unknown feature name")
    scores = [5.0 if set(row["source_indices"]) == {0, 1} else 1.0 if row["source_indices"] == [2] else -5.0 for row in candidates]
    assert set(select_partition(candidates, scores, 3)) == {frozenset({0, 1}), frozenset({2})}
    ranked = enumerate_partitions(candidates, scores, 3, top_n=2)
    assert set(ranked[0][1]) == {frozenset({0, 1}), frozenset({2})}
    exact_ranked = enumerate_partitions_exact(candidates, scores, 3, top_n=32)
    exhaustive = []
    by_group = {
        frozenset(int(index) for index in row["source_indices"]): float(score)
        for row, score in zip(candidates, scores, strict=True)
    }

    def visit(used: frozenset[int], total: float, groups: tuple[frozenset[int], ...]) -> None:
        if len(used) == len(strokes):
            exhaustive.append((total, groups))
            return
        first = next(index for index in range(len(strokes)) if index not in used)
        for group, score in by_group.items():
            if first in group and not (used & group):
                visit(used | group, total + score, groups + (group,))

    visit(frozenset(), 0.0, ())
    key = lambda row: (-row[0], tuple(tuple(sorted(group)) for group in row[1]))
    assert exact_ranked == sorted(exhaustive, key=key)[:32]

    class _ProbeModel:
        def predict_proba(self, matrix):
            logits = (
                0.25 * matrix[:, FEATURE_NAMES.index("stroke_count")]
                - 0.75 * matrix[:, FEATURE_NAMES.index("singleton")]
            )
            positive = 1.0 / (1.0 + np.exp(-logits))
            return np.column_stack((1.0 - positive, positive))

    selector = StrokeGroupingSelectorV1(
        _ProbeModel(), group_bias=0.0, model_version="test-ranker",
        lattice_config=DEFAULT_LATTICE_CONFIG,
    )
    selector_default = selector.group(strokes)
    selector_explicit_default = selector.group(strokes, neutralized_features=())
    assert selector_default == selector_explicit_default
    selector_shadow = selector.group(strokes, neutralized_features="singleton")
    assert selector_shadow.model_version == "test-ranker+mask(singleton)"
    assert sorted(index for group in selector_shadow.groups for index in group) == list(range(len(strokes)))

    beam_counterexample = [
        {"source_indices": [0]}, {"source_indices": [1]},
        {"source_indices": [0, 1]},
    ]
    beam_scores = [10.0, -100.0, 0.0]
    approximate = enumerate_partitions(
        beam_counterexample, beam_scores, 2, top_n=1, beam_width=1,
    )
    exact = enumerate_partitions_exact(
        beam_counterexample, beam_scores, 2, top_n=1,
    )
    assert approximate[0][0] == -90.0
    assert exact[0] == (0.0, (frozenset({0, 1}),))
    try:
        enumerate_partitions_exact(candidates, scores, 13, max_strokes=12)
    except ValueError:
        pass
    else:
        raise AssertionError("exact partition search exceeded its stroke cap")
    shape = group_shape_features(strokes, [0, 1])
    assert shape["stroke_count"] == 2.0 and shape["path_over_diag"] > 1.0
    assert normalized_group_dtw_distance(strokes, [0, 1], [0, 1]) == 0.0
    assert math.isinf(normalized_group_dtw_distance(strokes, [0], [0, 1]))


if __name__ == "__main__":
    _self_test()
    print('{"self_test":"pass"}')
