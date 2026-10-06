#!/usr/bin/env python3
"""Shadow-replay high-risk regions skipped only by the per-formula area cap."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

import joblib
import numpy as np

from selective_2d_anytime_v1 import (
    Selective2DAnytimeSolverV1,
    Selective2DConfigV1,
    build_local_hypergraph,
)
from stroke_grouping_v1 import FEATURE_NAMES, candidate_features


SCHEMA = "aiflow-hwr-region-budget-skip-replay/v1"


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _key(groups: Sequence[Sequence[int] | frozenset[int]]) -> tuple[tuple[int, ...], ...]:
    return tuple(sorted(tuple(sorted(int(index) for index in group)) for group in groups))


def _is_exact_cover(groups: Sequence[Sequence[int] | frozenset[int]], stroke_count: int) -> bool:
    flattened = [int(index) for group in groups for index in group]
    return len(flattened) == stroke_count and sorted(flattened) == list(range(stroke_count))


def _run_search_without_labels(
    strokes: Sequence[Mapping[str, Any]],
    region: set[int],
    incumbent: Sequence[frozenset[int]],
    locked: Sequence[frozenset[int]],
    config: Selective2DConfigV1,
    model: Any,
) -> dict[str, Any]:
    """Run the same bounded group-score search; receives no targets or labels."""
    local_rows = build_local_hypergraph(strokes, region, locked, config)
    if not local_rows:
        raise ValueError("budget-skipped region produced an empty local graph")
    features = candidate_features(local_rows, strokes)
    probabilities = np.clip(model.predict_proba(features)[:, 1], 1e-6, 1 - 1e-6)
    scores = np.log(probabilities / (1 - probabilities)).tolist()
    score_by_group = {
        frozenset(int(index) for index in row["source_indices"]): float(score)
        for row, score in zip(local_rows, scores, strict=True)
    }
    if any(group not in score_by_group for group in incumbent):
        raise AssertionError("local graph lost an incumbent Fast group")
    baseline_score = sum(score_by_group[group] for group in incumbent)
    winner = tuple(incumbent)
    winner_score = baseline_score
    solver = Selective2DAnytimeSolverV1(config=config)
    seen: set[tuple[tuple[int, ...], ...]] = set()
    ranked_last: list[tuple[float, tuple[frozenset[int], ...]]] = []
    evaluated = 0
    completed: list[int] = []
    for top_n in config.partition_schedule:
        ranked, _mode = solver._rank_partitions(local_rows, scores, len(strokes), top_n=top_n)
        ranked_last = ranked
        completed.append(top_n)
        for _geometry, groups in ranked:
            key = _key(groups)
            if key in seen:
                continue
            seen.add(key)
            evaluated += 1
            value = sum(score_by_group[group] for group in groups)
            if value >= winner_score + config.acceptance_margin:
                winner, winner_score = groups, value
        if len(ranked) < top_n:
            break
    return {
        "local_candidate_count": len(local_rows),
        "candidate_partition_count_evaluated": evaluated,
        "partition_schedule_completed": completed,
        "incumbent_score": baseline_score,
        "winner_score": winner_score,
        "winner_delta": winner_score - baseline_score,
        "winner_groups": [sorted(group) for group in winner],
        "winner_matches_incumbent": _key(winner) == _key(incumbent),
        "top_ranked_partition": [sorted(group) for group in ranked_last[0][1]] if ranked_last else [],
        "top32_partitions": [
            [sorted(group) for group in groups] for _score, groups in ranked_last
        ],
        "top32_partitions_are_exact_covers": all(
            _is_exact_cover(groups, len(strokes)) for _score, groups in ranked_last
        ),
    }


def audit(trace_path: Path, reachability_path: Path, ranker_path: Path) -> dict[str, Any]:
    traces = _jsonl(trace_path)
    reachability = _json(reachability_path)
    trace_sha = _sha256(trace_path)
    if reachability.get("inputs", {}).get("traces_sha256") != trace_sha:
        raise ValueError("reachability report does not match trace hash")
    formulas_path = Path(reachability["inputs"]["formulas"])
    if _sha256(formulas_path) != reachability["inputs"].get("formulas_sha256"):
        raise ValueError("raw formula source hash does not match reachability report")
    formulas = {str(row["sample_id"]): row for row in _jsonl(formulas_path)}
    reach_rows = {str(row["sample_id"]): row for row in reachability["formula_level"]}
    trace_rows = {str(row["sample_id"]): row for row in traces}
    if len(trace_rows) != len(traces) or set(trace_rows) != set(reach_rows):
        raise ValueError("trace/reachability formula IDs do not reconcile")
    if not set(trace_rows) <= set(formulas):
        raise ValueError("some trace formulas are absent from raw source")

    payload = joblib.load(ranker_path)
    model = payload["grouping_model"]
    if getattr(model, "n_features_in_", len(FEATURE_NAMES)) != len(FEATURE_NAMES):
        raise ValueError("partition-ranker feature count mismatch")
    config_values = dict(reachability["inputs"]["selective_2d_config"])
    config_values["partition_schedule"] = tuple(config_values["partition_schedule"])
    config = Selective2DConfigV1(**config_values)

    baseline_fractions: list[float] = []
    baseline_processed = 0
    total_strokes = 0
    skip_rows = [
        row for row in traces
        if row["grouping"].get("skip_reason") == "region_budget"
    ]
    results: list[dict[str, Any]] = []
    errors: list[str] = []

    for trace in traces:
        grouping = trace["grouping"]
        region = set(int(index) for index in grouping.get("local_region_strokes", []))
        formula_id = str(trace["sample_id"])
        strokes = sorted(formulas[formula_id]["strokes"], key=lambda row: int(row["order"]))
        stroke_count = len(strokes)
        total_strokes += stroke_count
        if grouping.get("partition_schedule_completed") and int(grouping.get("candidate_groups_local") or 0) > 0:
            baseline_processed += len(region)
            baseline_fractions.append(len(region) / max(stroke_count, 1))

    for trace in skip_rows:
        formula_id = str(trace["sample_id"])
        grouping = trace["grouping"]
        region = set(int(index) for index in grouping.get("local_region_strokes", []))
        strokes = sorted(formulas[formula_id]["strokes"], key=lambda row: int(row["order"]))
        stroke_count = len(strokes)
        incumbent = [
            frozenset(int(index) for index in group)
            for group in grouping.get("fast_incumbent_groups", grouping["fast_groups"])
        ]
        locked = [
            frozenset(int(index) for index in group)
            for group in grouping.get("locked_fast_groups", [])
        ]
        violations = list(grouping.get("region_budget_violations") or [])
        if not region or len(region) > config.max_region_strokes:
            errors.append(f"{formula_id}: skipped region is empty or exceeds 12 strokes")
            continue
        if violations != ["max_region_fraction"]:
            errors.append(f"{formula_id}: skip cause is not exclusively max_region_fraction")
            continue
        if len(strokes) != int(trace["source"]["stroke_count"]):
            errors.append(f"{formula_id}: raw stroke count mismatch")
            continue
        if set().union(region, *locked) != set(range(stroke_count)):
            errors.append(f"{formula_id}: region and locked groups do not exactly cover strokes")
            continue

        # This search call receives no target grouping or tokens. Labels join only below.
        search = _run_search_without_labels(strokes, region, incumbent, locked, config, model)
        target_groups = [
            frozenset(int(index) for index in group)
            for group in trace["source"]["target_grouping"]
        ]
        target_key = _key(target_groups)
        target_groups_crossing = [
            sorted(group) for group in target_groups
            if group.intersection(region) and not group <= region
        ]
        target_outside = {group for group in target_groups if not group.intersection(region)}
        locked_match = set(locked) == target_outside
        target_local = [group for group in target_groups if group <= region]
        local = build_local_hypergraph(strokes, region, locked, config)
        candidate_groups = {
            frozenset(int(index) for index in row["source_indices"]) for row in local
        }
        missing_local_target_groups = [sorted(group) for group in target_local if group not in candidate_groups]
        target_reachable = not target_groups_crossing and locked_match and not missing_local_target_groups
        target_rank = next(
            (
                rank for rank, groups in enumerate(search["top32_partitions"], start=1)
                if _key(groups) == target_key
            ),
            None,
        )
        target_is_incumbent = _key(incumbent) == target_key
        target_delta = (
            sum(_score_for_group(model, local, strokes, group) for group in target_groups)
            - search["incumbent_score"]
            if target_reachable else None
        )
        target_would_promote = bool(
            target_reachable
            and not target_is_incumbent
            and target_rank is not None
            and target_rank <= 32
            and target_delta is not None
            and target_delta >= config.acceptance_margin
        )
        target_is_winner = _key(search["winner_groups"]) == target_key
        results.append({
            "sample_id": formula_id,
            "formula_strokes": stroke_count,
            "region_strokes": len(region),
            "region_fraction": len(region) / max(stroke_count, 1),
            "budget_violations": violations,
            "local_search": search,
            "target_posthoc_audit": {
                "reachable_in_local_graph": target_reachable,
                "target_rank_in_bounded_top32": target_rank,
                "target_is_fast_incumbent": target_is_incumbent,
                "target_score_delta_vs_fast": target_delta,
                "would_be_promoted_by_current_margin": target_would_promote,
                "target_partition_is_search_winner": target_is_winner,
                "corrected_wrong_fast_partition": target_is_winner and not target_is_incumbent,
                "crossing_target_groups": target_groups_crossing,
                "locked_groups_match_target_outside": locked_match,
                "missing_local_target_groups": missing_local_target_groups,
            },
        })

    processed_after = baseline_processed + sum(row["region_strokes"] for row in results)
    baseline_median = median(baseline_fractions) if baseline_fractions else 0.0
    all_route_fractions = [
        len(trace["grouping"].get("local_region_strokes", []))
        / max(len(formulas[str(trace["sample_id"])]["strokes"]), 1)
        for trace in traces
        if trace["grouping"].get("partition_schedule_completed")
        and int(trace["grouping"].get("candidate_groups_local") or 0) > 0
    ] + [row["region_fraction"] for row in results]
    routed_median_after = median(all_route_fractions) if all_route_fractions else 0.0
    checks = {
        "trace_hash_matches_reachability": reachability["inputs"]["traces_sha256"] == trace_sha,
        "raw_formula_hash_matches_reachability": _sha256(formulas_path) == reachability["inputs"]["formulas_sha256"],
        "all_skips_are_fraction_only_and_within_stroke_cap": len(results) == len(skip_rows) and not errors,
        "label_free_partition_search_completed_for_every_skip": len(results) == len(skip_rows),
        "exact_cover_partitions_only": all(
            row["local_search"]["top32_partitions_are_exact_covers"]
            for row in results
        ),
    }
    return {
        "schema": SCHEMA,
        "scope": "frozen consumed-development replay; no fitting, CROHME, candidate-label search, or product promotion",
        "inputs": {
            "trace": {"path": str(trace_path), "sha256": trace_sha},
            "reachability": {"path": str(reachability_path), "sha256": _sha256(reachability_path)},
            "raw_formula_source": {"path": str(formulas_path), "sha256": _sha256(formulas_path)},
            "ranker": {"path": str(ranker_path), "sha256": _sha256(ranker_path)},
            "audit_script_sha256": _sha256(Path(__file__)),
        },
        "search_contract": {
            "region_cap": config.max_region_strokes,
            "existing_per_formula_fraction_cap": config.max_region_fraction,
            "counterfactual_policy": "bypass only max_region_fraction for existing risk regions; preserve 12-stroke cap, frozen ranker, candidate cap, partition schedule, and promotion margin",
            "target_labels_enter_after_search": True,
            "partition_schedule": list(config.partition_schedule),
        },
        "summary": {
            "formulas": len(traces),
            "region_budget_skips": len(skip_rows),
            "replayed_budget_skips": len(results),
            "target_partition_reachable": sum(row["target_posthoc_audit"]["reachable_in_local_graph"] for row in results),
            "target_partition_in_top32": sum(row["target_posthoc_audit"]["target_rank_in_bounded_top32"] is not None for row in results),
            "target_partition_would_be_promoted": sum(row["target_posthoc_audit"]["would_be_promoted_by_current_margin"] for row in results),
            "target_partition_is_search_winner": sum(row["target_posthoc_audit"]["target_partition_is_search_winner"] for row in results),
            "wrong_fast_partition_corrected": sum(row["target_posthoc_audit"]["corrected_wrong_fast_partition"] for row in results),
            "baseline_processed_strokes": baseline_processed,
            "additional_processed_strokes": sum(row["region_strokes"] for row in results),
            "processed_strokes_after": processed_after,
            "total_strokes": total_strokes,
            "processed_fraction_after": processed_after / max(total_strokes, 1),
            "total_2d_stroke_fraction_gate_pass": processed_after / max(total_strokes, 1) <= 0.35,
            "routed_region_median_before": baseline_median,
            "routed_region_median_after": routed_median_after,
            "routed_region_median_gate_pass": routed_median_after <= 0.50,
        },
        "formula_level": results,
        "invariant_errors": errors,
        "verification": {"checks": checks, "all_checks_pass": all(checks.values())},
    }


def _score_for_group(model: Any, local_rows: Sequence[Mapping[str, Any]], strokes: Sequence[Mapping[str, Any]], group: frozenset[int]) -> float:
    row_by_group = {
        frozenset(int(index) for index in row["source_indices"]): row for row in local_rows
    }
    row = row_by_group.get(group)
    if row is None:
        raise ValueError(f"group not in local candidate graph: {sorted(group)}")
    features = candidate_features([row], strokes)
    probability = float(np.clip(model.predict_proba(features)[0, 1], 1e-6, 1 - 1e-6))
    return float(np.log(probability / (1 - probability)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--reachability", type=Path, required=True)
    parser.add_argument("--ranker", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite output: {args.output}")
    report = audit(args.trace, args.reachability, args.ranker)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output), "summary": report["summary"],
        "verification": report["verification"], "invariant_errors": report["invariant_errors"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
