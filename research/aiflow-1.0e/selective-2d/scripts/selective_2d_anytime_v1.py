#!/usr/bin/env python3
"""Bounded, candidate-preserving selective 2D routing for online math ink.

This module deliberately has no training or CROHME dependency.  It turns the
existing grouping lattice into an anytime challenger: a cheap exact-cover is
the incumbent, and only a bounded spatially connected risk region receives
non-contiguous multi-stroke candidates and additional partition search.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
import math
from typing import Any, Callable, Iterable, Mapping, Sequence

from stroke_grouping_v1 import (
    build_lattice, enumerate_partitions, enumerate_partitions_exact,
)


SCHEMA = "aiflow-selective-2d-anytime/v1"
ROUTES = frozenset(("fast", "local_2d", "fallback"))


@dataclass(frozen=True)
class Selective2DConfigV1:
    temporal_window: int = 6
    spatial_neighbors: int = 4
    max_region_strokes: int = 12
    max_region_fraction: float = 0.50
    max_hyperedge_size: int = 4
    max_candidates_per_stroke: int = 64
    partition_schedule: tuple[int, ...] = (4, 8, 16, 32)
    acceptance_margin: float = 0.02
    partition_margin: float = 0.25
    symbol_margin: float = 0.15
    reject_local_group_count_increase: bool = False
    exact_partition_search: bool = False

    def __post_init__(self) -> None:
        if not 2 <= self.max_hyperedge_size <= 4:
            raise ValueError("max_hyperedge_size must be in [2, 4]")
        if not 1 <= self.max_region_strokes <= 12 or not 0.0 < self.max_region_fraction <= 0.50:
            raise ValueError("max_region_strokes must be in [1, 12]")
        if not self.partition_schedule or tuple(sorted(self.partition_schedule)) != self.partition_schedule:
            raise ValueError("partition_schedule must be non-empty and ascending")
        if self.partition_schedule[-1] > 32:
            raise ValueError("partition search is bounded at 32")
        if self.max_candidates_per_stroke < 1 or self.acceptance_margin < 0:
            raise ValueError("invalid selective 2D budget")


def _xy(point: Any) -> tuple[float, float]:
    if isinstance(point, dict):
        return float(point["x"]), float(point["y"])
    return float(point[0]), float(point[1])


def _box(stroke: Mapping[str, Any]) -> tuple[float, float, float, float]:
    points = [_xy(point) for point in stroke.get("points") or ()]
    if not points:
        raise ValueError("selective 2D requires non-empty strokes")
    xs, ys = zip(*points, strict=True)
    if not all(math.isfinite(value) for value in (*xs, *ys)):
        raise ValueError("selective 2D requires finite stroke coordinates")
    return min(xs), min(ys), max(xs), max(ys)


def _group_box(indices: Iterable[int], boxes: Sequence[tuple[float, float, float, float]]) -> dict[str, float]:
    selected = [boxes[int(index)] for index in indices]
    if not selected:
        raise ValueError("group must contain a stroke")
    return {
        "left": min(row[0] for row in selected), "top": min(row[1] for row in selected),
        "right": max(row[2] for row in selected), "bottom": max(row[3] for row in selected),
    }


def _gap(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    dx = max(left[0] - right[2], right[0] - left[2], 0.0)
    dy = max(left[1] - right[3], right[1] - left[3], 0.0)
    return math.hypot(dx, dy)


def _candidate_id(indices: Iterable[int]) -> str:
    return "g:" + ",".join(str(index) for index in sorted(indices))


def _validate_strokes(strokes: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    ordered = sorted(strokes, key=lambda row: int(row.get("order", -1)))
    if not ordered or [int(row.get("order", -1)) for row in ordered] != list(range(len(ordered))):
        raise ValueError("strokes must have contiguous zero-based order")
    for stroke in ordered:
        _box(stroke)
    return ordered


def _connected(indices: tuple[int, ...], adjacency: Mapping[int, set[int]]) -> bool:
    seen = {indices[0]}
    pending = [indices[0]]
    allowed = set(indices)
    while pending:
        current = pending.pop()
        for other in adjacency[current] & allowed:
            if other not in seen:
                seen.add(other)
                pending.append(other)
    return seen == allowed


def build_local_hypergraph(
    strokes: Sequence[Mapping[str, Any]], region_indices: Iterable[int], locked_groups: Iterable[Iterable[int]],
    config: Selective2DConfigV1 = Selective2DConfigV1(),
) -> list[dict[str, Any]]:
    """Return an exact-cover candidate set with only the selected region mutable."""
    ordered = _validate_strokes(strokes)
    region = frozenset(int(value) for value in region_indices)
    if not region or len(region) > config.max_region_strokes or min(region) < 0 or max(region) >= len(ordered):
        raise ValueError("invalid local region")
    locked = [frozenset(int(value) for value in group) for group in locked_groups]
    if any(not group or group & region for group in locked):
        raise ValueError("locked groups must be non-empty and outside the local region")
    if set().union(region, *locked) != set(range(len(ordered))):
        raise ValueError("local region and locked groups must exactly cover source strokes")
    if sum(len(group) for group in locked) != len(set().union(*locked)):
        raise ValueError("locked groups overlap")

    boxes = [_box(stroke) for stroke in ordered]
    baseline = build_lattice(
        ordered, temporal_window=config.temporal_window, spatial_neighbors=config.spatial_neighbors,
    )
    output: dict[frozenset[int], dict[str, Any]] = {}
    for row in baseline:
        group = frozenset(int(value) for value in row["source_indices"])
        if group <= region:
            output[group] = {**row, "evidence": sorted(set(row.get("evidence") or ()))}
    for group in locked:
        output[group] = {
            "source_indices": sorted(group), "box": _group_box(group, boxes),
            "evidence": ["locked_fast_group"], "candidate_id": _candidate_id(group),
        }

    adjacency: dict[int, set[int]] = {index: set() for index in region}
    for index in region:
        nearest = sorted(
            (other for other in region if other != index),
            key=lambda other: (_gap(boxes[index], boxes[other]), abs(index - other), other),
        )[:config.spatial_neighbors]
        for other in nearest:
            adjacency[index].add(other)
            adjacency[other].add(index)
    for size in range(3, config.max_hyperedge_size + 1):
        for group_tuple in combinations(sorted(region), size):
            if not _connected(group_tuple, adjacency):
                continue
            group = frozenset(group_tuple)
            current = output.get(group)
            if current is None:
                output[group] = {
                    "source_indices": list(group_tuple), "box": _group_box(group, boxes),
                    "evidence": [f"local_hyperedge:{size}"], "candidate_id": _candidate_id(group),
                }
            else:
                current["evidence"] = sorted(set(current["evidence"]) | {f"local_hyperedge:{size}"})
    rows = sorted(output.values(), key=lambda row: (row["box"]["left"], len(row["source_indices"]), row["source_indices"]))
    return rows


def default_group_score(candidate: Mapping[str, Any], _strokes: Sequence[Mapping[str, Any]]) -> float:
    """A deterministic geometry-only fallback score; production injects a frozen scorer."""
    size = len(candidate["source_indices"])
    evidence = set(candidate.get("evidence") or ())
    return (0.0 if size == 1 else -0.08 * (size - 1)) + (0.02 if "spatial_pair" in evidence else 0.0)


def _scores(
    candidates: Sequence[Mapping[str, Any]], strokes: Sequence[Mapping[str, Any]],
    scorer: Callable[[Mapping[str, Any], Sequence[Mapping[str, Any]]], float],
) -> list[float]:
    values = [float(scorer(row, strokes)) for row in candidates]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("candidate scorer produced a non-finite score")
    return values


def _risk_seeds(
    strokes: Sequence[Mapping[str, Any]], candidates: Sequence[Mapping[str, Any]],
    ranked: Sequence[tuple[float, tuple[frozenset[int], ...]]],
    symbol_margins: Mapping[frozenset[int], float], config: Selective2DConfigV1,
) -> tuple[set[int], list[str]]:
    if not ranked:
        raise ValueError("risk scout requires an incumbent partition")
    seeds: set[int] = set()
    reasons: list[str] = []
    if len(ranked) > 1 and ranked[0][0] - ranked[1][0] <= config.partition_margin:
        incumbent = {frozenset(group) for group in ranked[0][1]}
        alternative = {frozenset(group) for group in ranked[1][1]}
        disputed_groups = incumbent.symmetric_difference(alternative)
        if disputed_groups:
            seeds.update(index for group in disputed_groups for index in group)
            reasons.append("partition_margin")
    low_confidence_strokes: set[int] = set()
    for group, margin in symbol_margins.items():
        if margin <= config.symbol_margin:
            seeds.update(group)
            low_confidence_strokes.update(group)
            reasons.append("symbol_margin")
    # A confident per-glyph prediction can still be a wrong segmentation in a
    # superscript/fraction layout.  Inspect the Fast *groups* (not raw strokes)
    # so a multi-stroke base symbol is not mistaken for its own script.
    fast_boxes = [(_group_box(group, [_box(stroke) for stroke in strokes]), group) for group in ranked[0][1]]
    for parent_box, parent_group in fast_boxes:
        parent_width = max(parent_box["right"] - parent_box["left"], 1e-6)
        parent_height = max(parent_box["bottom"] - parent_box["top"], 1e-6)
        parent_cx = (parent_box["left"] + parent_box["right"]) / 2.0
        parent_cy = (parent_box["top"] + parent_box["bottom"]) / 2.0
        for child_box, child_group in fast_boxes:
            if child_group == parent_group:
                continue
            child_cx = (child_box["left"] + child_box["right"]) / 2.0
            child_cy = (child_box["top"] + child_box["bottom"]) / 2.0
            # A compact group above and to the right is a deliberately
            # conservative script cue.  It is independent of HWR confidence.
            if (
                child_cx >= parent_cx + 0.15 * parent_width
                and child_cx <= parent_box["right"] + 1.5 * parent_width
                and child_cy <= parent_cy - 0.25 * parent_height
            ):
                seeds.update(parent_group)
                seeds.update(child_group)
                reasons.append("structural_layout")
    boxes = [_box(stroke) for stroke in strokes]
    for row in candidates:
        evidence = set(row.get("evidence") or ())
        if "spatial_pair" in evidence and len(row["source_indices"]) == 2:
            left, right = (boxes[int(value)] for value in row["source_indices"])
            x_overlap = min(left[2], right[2]) - max(left[0], right[0])
            y_overlap = min(left[3], right[3]) - max(left[1], right[1])
            left_height = max(left[3] - left[1], 1e-6)
            right_height = max(right[3] - right[1], 1e-6)
            vertical_offset = abs((left[1] + left[3] - right[1] - right[3]) / 2.0)
            # A spatial pair exists in ordinary multi-stroke glyphs as well.
            # Geometry alone was empirically universal on the owned dev rows;
            # only let it expand an already uncertain Fast HWR group.
            if (
                low_confidence_strokes.intersection(int(value) for value in row["source_indices"])
                and x_overlap > 0.0
                and (y_overlap > 0.0 or vertical_offset >= 0.30 * min(left_height, right_height))
            ):
                seeds.update(int(value) for value in row["source_indices"])
                reasons.append("spatial_overlap_low_confidence")
    return seeds, sorted(set(reasons))


def _expand_region(
    strokes: Sequence[Mapping[str, Any]], seeds: set[int],
    incumbent_groups: Sequence[Iterable[int]], config: Selective2DConfigV1,
) -> set[int]:
    if not seeds:
        return set()
    boxes = [_box(stroke) for stroke in strokes]
    groups = [frozenset(int(value) for value in group) for group in incumbent_groups]

    def close_fast_groups(indices: set[int]) -> set[int]:
        closed = set(indices)
        for group in groups:
            if group & closed:
                closed.update(group)
        return closed

    region = close_fast_groups(set(seeds))
    for index in sorted(seeds):
        neighbors = sorted(
            (value for value in range(len(strokes)) if value not in region),
            key=lambda value: (_gap(boxes[index], boxes[value]), abs(index - value), value),
        )
        if not neighbors:
            continue
        candidate = close_fast_groups(region | {neighbors[0]})
        if len(candidate) > config.max_region_strokes:
            continue
        if len(candidate) / len(strokes) > config.max_region_fraction:
            continue
        region = candidate
    return region


def _partition_key(groups: Sequence[frozenset[int]]) -> tuple[tuple[int, ...], ...]:
    return tuple(sorted(tuple(sorted(group)) for group in groups))


def _group_count_increase_rejected(
    challenger: Sequence[Iterable[int]], incumbent: Sequence[Iterable[int]], enabled: bool,
) -> bool:
    return bool(enabled and len(challenger) > len(incumbent))


class Selective2DAnytimeSolverV1:
    """Run a deterministic Fast-vs-local-2D tournament with a bounded search."""

    def __init__(
        self, *, config: Selective2DConfigV1 = Selective2DConfigV1(),
        group_scorer: Callable[[Mapping[str, Any], Sequence[Mapping[str, Any]]], float] = default_group_score,
        group_score_batch: Callable[[Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]]], Sequence[float]] | None = None,
        joint_scorer: Callable[[Sequence[frozenset[int]], Sequence[Mapping[str, Any]], Sequence[float]], float] | None = None,
    ) -> None:
        self.config, self.group_scorer, self.group_score_batch, self.joint_scorer = config, group_scorer, group_score_batch, joint_scorer

    def _score_rows(self, candidates, strokes) -> list[float]:
        values = list(self.group_score_batch(candidates, strokes)) if self.group_score_batch else _scores(candidates, strokes, self.group_scorer)
        if len(values) != len(candidates) or not all(math.isfinite(float(value)) for value in values):
            raise ValueError("candidate batch scorer produced invalid scores")
        return [float(value) for value in values]

    def _rank_partitions(self, candidates, scores, stroke_count, *, top_n):
        if self.config.exact_partition_search and stroke_count <= 12:
            return enumerate_partitions_exact(
                candidates, scores, stroke_count, top_n=top_n,
                options_per_stroke=self.config.max_candidates_per_stroke,
                max_strokes=12,
            ), "exact_k_best_capped"
        mode = "beam_fallback_over_12_strokes" if self.config.exact_partition_search else "beam"
        return enumerate_partitions(
            candidates, scores, stroke_count, top_n=top_n,
            beam_width=max(32, top_n),
            options_per_stroke=self.config.max_candidates_per_stroke,
        ), mode

    def solve(
        self, strokes: Sequence[Mapping[str, Any]], *,
        symbol_margins: Mapping[frozenset[int], float] | None = None,
        allow_local: bool = True,
    ) -> dict[str, Any]:
        ordered = _validate_strokes(strokes)
        baseline = build_lattice(
            ordered, temporal_window=self.config.temporal_window,
            spatial_neighbors=self.config.spatial_neighbors,
        )
        baseline_scores = self._score_rows(baseline, ordered)
        ranked, fast_search_mode = self._rank_partitions(
            baseline, baseline_scores, len(ordered), top_n=2,
        )
        if not ranked:
            raise ValueError("fast route has no exact-cover partition")
        incumbent_groups = ranked[0][1]
        selected = {frozenset(row["source_indices"]): row for row in baseline}
        if not allow_local:
            return {
                "route": "fast", "groups": [sorted(group) for group in incumbent_groups],
                "candidate_rows": [selected[group] for group in incumbent_groups],
                "audit": {
                    "schema": SCHEMA, "route": "fast", "risk_reasons": ["router_off"],
                    "risk_strokes": [], "local_region_seed_strokes": [],
                    "local_region_strokes": [], "locked_fast_groups": [],
                    "candidate_groups_fast": len(baseline), "candidate_groups_local": 0,
                    "partition_schedule_completed": [], "all_strokes_exactly_once": True,
                    "local_processed_stroke_fraction": 0.0,
                    "local_requested_stroke_fraction": 0.0,
                    "fast_partition_search_mode": fast_search_mode,
                    "fast_incumbent_groups": [sorted(group) for group in incumbent_groups],
                    "config": asdict(self.config),
                },
            }
        margin_map = dict(symbol_margins or {})
        seeds, reasons = _risk_seeds(ordered, baseline, ranked, margin_map, self.config)
        region = _expand_region(
            ordered, seeds, incumbent_groups, self.config,
        )
        base_audit = {
            "schema": SCHEMA, "route": "fast", "risk_reasons": reasons,
            "risk_strokes": sorted(seeds),
            "local_region_seed_strokes": sorted(seeds),
            "local_region_strokes": sorted(region), "locked_fast_groups": [],
            "candidate_groups_fast": len(baseline), "candidate_groups_local": 0,
            "partition_schedule_completed": [], "all_strokes_exactly_once": True,
            "local_processed_stroke_fraction": 0.0,
            "local_requested_stroke_fraction": len(region) / len(ordered),
            "fast_partition_search_mode": fast_search_mode,
            "fast_incumbent_groups": [sorted(group) for group in incumbent_groups],
            "config": asdict(self.config),
        }
        if not region:
            return {
                "route": "fast", "groups": [sorted(group) for group in incumbent_groups],
                "candidate_rows": [selected[group] for group in incumbent_groups], "audit": base_audit,
            }

        locked = [group for group in incumbent_groups if not group & region]
        local_group_count = sum(group <= region for group in incumbent_groups)
        fraction_exception = (
            reasons == ["structural_layout"]
            and local_group_count == 2
            and len(region) <= self.config.max_region_strokes
        )
        fraction_exception_reason = (
            "structural_pair_minimum"
            if fraction_exception
            and len(region) / len(ordered) > self.config.max_region_fraction
            else None
        )
        base_audit = {
            **base_audit,
            "local_region_strokes": sorted(region),
            "locked_fast_groups": [sorted(group) for group in locked],
            "local_requested_stroke_fraction": len(region) / len(ordered),
            "region_fraction_exception": fraction_exception_reason,
        }
        region_budget_violations = []
        if len(region) > self.config.max_region_strokes:
            region_budget_violations.append("max_region_strokes")
        if (
            len(region) / len(ordered) > self.config.max_region_fraction
            and fraction_exception_reason is None
        ):
            region_budget_violations.append("max_region_fraction")
        if not region or region_budget_violations:
            selected = {frozenset(row["source_indices"]): row for row in baseline}
            return {
                "route": "fast", "groups": [sorted(group) for group in incumbent_groups],
                "candidate_rows": [selected[group] for group in incumbent_groups],
                "audit": {
                    **base_audit,
                    "skip_reason": "region_budget" if region_budget_violations else None,
                    "region_budget_violations": region_budget_violations,
                    "local_processed_stroke_fraction": 0.0,
                },
            }
        local = build_local_hypergraph(ordered, region, locked, self.config)
        local_scores = self._score_rows(local, ordered)
        by_group = {frozenset(row["source_indices"]): index for index, row in enumerate(local)}
        joint_score_stats = {
            "evaluations": 0,
            "finite": 0,
            "non_finite": 0,
            "finite_not_promoted": 0,
            "promoted": 0,
        }

        def score(groups: Sequence[frozenset[int]]) -> float:
            if self.joint_scorer is not None:
                value = float(self.joint_scorer(groups, local, local_scores))
                joint_score_stats["evaluations"] += 1
                if math.isfinite(value):
                    joint_score_stats["finite"] += 1
                    return value
                joint_score_stats["non_finite"] += 1
                return -math.inf
            return sum(local_scores[by_group[group]] for group in groups)

        incumbent_score = score(incumbent_groups)
        winner = incumbent_groups
        winner_score = incumbent_score
        group_count_increase_rejections = 0
        incumbent_key = _partition_key(incumbent_groups)
        completed: list[int] = []
        local_search_modes: list[str] = []
        seen: set[tuple[tuple[int, ...], ...]] = set()
        max_top_n = self.config.partition_schedule[-1]
        ranked_alternatives, search_mode = self._rank_partitions(
            local, local_scores, len(ordered), top_n=max_top_n,
        )
        for top_n in self.config.partition_schedule:
            # The schedules are prefixes of one deterministic ranking. Reuse
            # that ranking instead of rerunning the same bounded search at
            # 4, 8, 16, and 32; promotion is still evaluated stage by stage.
            alternatives = ranked_alternatives[:top_n]
            completed.append(top_n)
            local_search_modes.append(search_mode)
            for _geometry, groups in alternatives:
                key = _partition_key(groups)
                if key in seen:
                    continue
                seen.add(key)
                value = score(groups)
                if math.isfinite(value) and value >= winner_score + self.config.acceptance_margin:
                    if _group_count_increase_rejected(
                        groups, incumbent_groups,
                        self.config.reject_local_group_count_increase,
                    ):
                        group_count_increase_rejections += 1
                        if self.joint_scorer is not None:
                            joint_score_stats["finite_not_promoted"] += 1
                        continue
                    winner, winner_score = groups, value
                    if self.joint_scorer is not None:
                        joint_score_stats["promoted"] += 1
                elif math.isfinite(value) and self.joint_scorer is not None:
                    joint_score_stats["finite_not_promoted"] += 1
            # ``top_n`` is a prefix, so a current loser does not bound unseen
            # partitions.  Stop only when enumeration itself is exhausted.
            if len(alternatives) < top_n:
                break
        assigned = sorted(index for group in winner for index in group)
        if assigned != list(range(len(ordered))):
            raise AssertionError("selective 2D lost or duplicated a source stroke")
        route = "local_2d" if _partition_key(winner) != incumbent_key else "fast"
        if self.config.reject_local_group_count_increase and len(winner) > len(incumbent_groups):
            raise AssertionError("group-count guard allowed a larger local partition")
        score_delta = (
            winner_score - incumbent_score
            if math.isfinite(winner_score) and math.isfinite(incumbent_score) else None
        )
        return {
            "route": route, "groups": [sorted(group) for group in winner],
            "candidate_rows": [local[by_group[group]] for group in winner],
            "audit": {
                **base_audit, "route": route, "candidate_groups_local": len(local),
                "partition_schedule_completed": completed,
                "local_partition_search_modes": local_search_modes,
                "incumbent_score": incumbent_score if math.isfinite(incumbent_score) else None,
                "winner_score": winner_score if math.isfinite(winner_score) else None,
                "winner_score_delta": score_delta,
                "joint_score_improved": route == "local_2d",
                "local_group_count_increase_guard_enabled": self.config.reject_local_group_count_increase,
                "local_group_count_increase_rejections": group_count_increase_rejections,
                "local_processed_stroke_fraction": len(region) / len(ordered),
                "local_requested_stroke_fraction": len(region) / len(ordered),
                "joint_score_search": (
                    dict(joint_score_stats) if self.joint_scorer is not None else None
                ),
            },
        }


def _self_test() -> None:
    strokes = [
        {"order": 0, "points": [{"x": 0.0, "y": 0.0}, {"x": 0.0, "y": 2.0}]},
        {"order": 1, "points": [{"x": 100.0, "y": 0.0}, {"x": 100.0, "y": 2.0}]},
        {"order": 2, "points": [{"x": 10.0, "y": 0.0}, {"x": 10.0, "y": 2.0}]},
        {"order": 3, "points": [{"x": 1.0, "y": 1.0}, {"x": 9.0, "y": 1.0}]},
        {"order": 4, "points": [{"x": 200.0, "y": 0.0}, {"x": 200.0, "y": 2.0}]},
        {"order": 5, "points": [{"x": 220.0, "y": 0.0}, {"x": 220.0, "y": 2.0}]},
        {"order": 6, "points": [{"x": 240.0, "y": 0.0}, {"x": 240.0, "y": 2.0}]},
        {"order": 7, "points": [{"x": 260.0, "y": 0.0}, {"x": 260.0, "y": 2.0}]},
    ]
    config = Selective2DConfigV1(partition_margin=-1.0, acceptance_margin=0.0)
    local = build_local_hypergraph(strokes, {0, 1, 2, 3}, ({4}, {5}, {6}, {7}), config)
    assert any(row["source_indices"] == [0, 2, 3] for row in local)
    solver = Selective2DAnytimeSolverV1(
        config=config,
        group_scorer=lambda row, _strokes: (
            5.0 if row["source_indices"] == [0, 2, 3]
            else (-1.0 if len(row["source_indices"]) > 1 else 0.0)
        ),
    )
    result = solver.solve(strokes, symbol_margins={frozenset({0}): 0.0, frozenset({2}): 0.0})
    assert result["route"] == "local_2d"
    assert {tuple(group) for group in result["groups"]} == {(0, 2, 3), (1,), (4,), (5,), (6,), (7,)}
    assert result["audit"]["all_strokes_exactly_once"] is True
    exact_config = Selective2DConfigV1(
        partition_margin=-1.0, acceptance_margin=0.0,
        exact_partition_search=True,
    )
    exact_solver = Selective2DAnytimeSolverV1(
        config=exact_config,
        group_scorer=lambda row, _strokes: (
            5.0 if row["source_indices"] == [0, 2, 3]
            else (-1.0 if len(row["source_indices"]) > 1 else 0.0)
        ),
    )
    for prefix_solver in (solver, exact_solver):
        local_scores = prefix_solver._score_rows(local, strokes)
        full_ranking, _mode = prefix_solver._rank_partitions(
            local, local_scores, len(strokes), top_n=32,
        )
        for budget in config.partition_schedule:
            prefix, _mode = prefix_solver._rank_partitions(
                local, local_scores, len(strokes), top_n=budget,
            )
            assert prefix == full_ranking[:budget]
    exact_fast = exact_solver.solve(strokes, allow_local=False)
    assert exact_fast["audit"]["fast_partition_search_mode"] == "exact_k_best_capped"
    assert exact_fast["audit"]["all_strokes_exactly_once"] is True
    exact_search_requests: list[int] = []
    exact_rank_partitions = exact_solver._rank_partitions

    def counted_exact_rank(candidates, scores, stroke_count, *, top_n):
        exact_search_requests.append(top_n)
        return exact_rank_partitions(candidates, scores, stroke_count, top_n=top_n)

    exact_solver._rank_partitions = counted_exact_rank
    exact_local = exact_solver.solve(
        strokes, symbol_margins={frozenset({0}): 0.0, frozenset({2}): 0.0},
    )
    assert exact_search_requests == [2, exact_config.partition_schedule[-1]]
    exact_modes = exact_local["audit"]["local_partition_search_modes"]
    assert exact_modes and all(mode == "exact_k_best_capped" for mode in exact_modes)
    assert exact_local["audit"]["all_strokes_exactly_once"] is True
    exact_audit = exact_local["audit"]
    seed_region = set(exact_audit["local_region_seed_strokes"])
    local_region = set(exact_audit["local_region_strokes"])
    locked_groups = {frozenset(group) for group in exact_audit["locked_fast_groups"]}
    fast_groups = {frozenset(group) for group in exact_audit["fast_incumbent_groups"]}
    assert seed_region <= local_region
    assert all(
        group <= local_region for group in fast_groups if group & seed_region
    )
    assert locked_groups == {group for group in fast_groups if not group & local_region}
    assert set().union(local_region, *locked_groups) == set(range(len(strokes)))
    assert math.isclose(
        exact_audit["local_processed_stroke_fraction"], len(local_region) / len(strokes),
    )
    default_solver = Selective2DAnytimeSolverV1()
    default_fast = default_solver.solve(strokes, allow_local=False)
    baseline_rows = build_lattice(strokes, temporal_window=6, spatial_neighbors=4)
    baseline_scores = [default_group_score(row, strokes) for row in baseline_rows]
    legacy_ranked = enumerate_partitions(
        baseline_rows, baseline_scores, len(strokes), top_n=2,
        beam_width=32, options_per_stroke=64,
    )
    assert _partition_key(default_fast["groups"]) == _partition_key(legacy_ranked[0][1])
    assert default_fast["audit"]["fast_partition_search_mode"] == "beam"
    joint_solver = Selective2DAnytimeSolverV1(
        config=config,
        group_scorer=lambda row, _strokes: 1.0 if len(row["source_indices"]) == 1 else 0.0,
        joint_scorer=lambda groups, _rows, _scores: (
            5.0 if _partition_key(groups) == _partition_key(
                (frozenset({0, 2, 3}), frozenset({1}), frozenset({4}),
                 frozenset({5}), frozenset({6}), frozenset({7}))
            ) else -5.0
        ),
    )
    joint_search_requests: list[int] = []
    joint_rank_partitions = joint_solver._rank_partitions

    def counted_joint_rank(candidates, scores, stroke_count, *, top_n):
        joint_search_requests.append(top_n)
        return joint_rank_partitions(candidates, scores, stroke_count, top_n=top_n)

    joint_solver._rank_partitions = counted_joint_rank
    joint_result = joint_solver.solve(
        strokes, symbol_margins={frozenset({0}): 0.0, frozenset({2}): 0.0},
    )
    assert joint_search_requests == [2, config.partition_schedule[-1]]
    assert joint_result["route"] == "local_2d"
    assert {tuple(group) for group in joint_result["groups"]} == {(0, 2, 3), (1,), (4,), (5,), (6,), (7,)}
    invalid_joint_solver = Selective2DAnytimeSolverV1(
        config=Selective2DConfigV1(partition_margin=-1.0, acceptance_margin=0.0),
        joint_scorer=lambda _groups, _rows, _scores: -math.inf,
    )
    invalid_joint_result = invalid_joint_solver.solve(
        strokes, symbol_margins={frozenset({0}): 0.0, frozenset({2}): 0.0},
    )
    assert invalid_joint_result["route"] == "fast"
    assert invalid_joint_result["audit"]["winner_score_delta"] is None
    script_strokes = [
        {"order": 0, "points": [{"x": 0.0, "y": 10.0}, {"x": 4.0, "y": 20.0}]},
        {"order": 1, "points": [{"x": 5.0, "y": 10.0}, {"x": 9.0, "y": 20.0}]},
        {"order": 2, "points": [{"x": 10.0, "y": 0.0}, {"x": 13.0, "y": 5.0}]},
    ]
    seeds, reasons = _risk_seeds(
        script_strokes, (), [(0.0, (frozenset({0}), frozenset({1}), frozenset({2})))], {}, config,
    )
    assert reasons == ["structural_layout"] and seeds == {1, 2}
    horizontal_strokes = [{
        "order": index,
        "points": [
            {"x": float(index * 12), "y": 10.0},
            {"x": float(index * 12 + 4), "y": 14.0},
        ],
    } for index in range(4)]
    partition_seeds, partition_reasons = _risk_seeds(
        horizontal_strokes, (), [
            (1.0, (frozenset({0, 1}), frozenset({2}), frozenset({3}))),
            (0.9, (frozenset({0}), frozenset({1, 2}), frozenset({3}))),
        ], {}, Selective2DConfigV1(),
    )
    assert partition_reasons == ["partition_margin"]
    assert partition_seeds == {0, 1, 2}
    fraction_limited_solver = Selective2DAnytimeSolverV1(
        config=Selective2DConfigV1(max_region_fraction=0.50),
    )
    fraction_limited = fraction_limited_solver.solve(
        script_strokes,
        symbol_margins={
            frozenset({0}): 0.0, frozenset({1}): 0.0, frozenset({2}): 0.0,
        },
    )
    assert fraction_limited["route"] == "fast"
    assert fraction_limited["audit"]["skip_reason"] == "region_budget"
    assert fraction_limited["audit"]["region_budget_violations"] == ["max_region_fraction"]
    assert fraction_limited["audit"]["local_requested_stroke_fraction"] == 1.0
    assert fraction_limited["audit"]["local_processed_stroke_fraction"] == 0.0
    structural_pair_solver = Selective2DAnytimeSolverV1(
        config=Selective2DConfigV1(partition_margin=-1.0, max_region_fraction=0.50),
    )
    structural_pair = structural_pair_solver.solve(script_strokes)
    assert structural_pair["route"] == "fast"
    assert structural_pair["audit"]["risk_reasons"] == ["structural_layout"]
    assert structural_pair["audit"]["local_region_strokes"] == [1, 2]
    assert structural_pair["audit"]["region_fraction_exception"] == "structural_pair_minimum"
    assert structural_pair["audit"]["local_processed_stroke_fraction"] == 2 / 3
    assert structural_pair["audit"]["all_strokes_exactly_once"] is True


if __name__ == "__main__":
    _self_test()
    print('{"self_test":"pass"}')
