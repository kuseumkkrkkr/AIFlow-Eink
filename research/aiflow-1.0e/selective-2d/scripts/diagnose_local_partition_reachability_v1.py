#!/usr/bin/env python3
"""Forensic reachability audit of target partitions in Selective-2D local graphs."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from selective_2d_anytime_v1 import (
    Selective2DAnytimeSolverV1, Selective2DConfigV1, _risk_seeds,
    build_local_hypergraph,
)


SCHEMA = "aiflow-hwr-local-partition-reachability/v4"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _error_shape(selected: list[list[int]], target: list[list[int]]) -> str:
    if {frozenset(group) for group in selected} == {frozenset(group) for group in target}:
        return "exact"
    if len(selected) > len(target):
        return "oversegmentation"
    if len(selected) < len(target):
        return "undersegmentation"
    return "same_count_wrong_partition"


def _partition_key(groups: Any) -> tuple[tuple[int, ...], ...]:
    return tuple(sorted(tuple(sorted(int(index) for index in group)) for group in groups))


def diagnose(
    traces_path: Path, dataset_root: Path, partition_ranker_path: Path | None = None,
) -> dict[str, Any]:
    formulas_path = dataset_root / "data" / "formulas_valid.jsonl"
    traces = _read_jsonl(traces_path)
    formulas = {
        str(row["sample_id"]): row for row in _read_jsonl(formulas_path)
    }
    config = Selective2DConfigV1()
    records = []
    outcomes: Counter[str] = Counter()
    outcomes_by_shape: dict[str, Counter[str]] = {}
    rebuilt_local_graphs = 0
    search_outcomes: Counter[str] = Counter()
    replayed_searches = 0
    replayed_exact_covers = 0
    replayed_partitions = 0
    score_reconciliations = 0
    score_reconciliation_max_abs_delta = 0.0
    score_feature_comparisons = 0
    score_feature_counterfactual_checks = 0
    score_feature_counterfactual_max_abs_delta = 0.0
    score_feature_effects: dict[str, list[float]] = {}
    score_feature_flip_counts: Counter[str] = Counter()
    negative_score_feature_comparisons = 0
    replayed_router_cues = 0
    ranker_payload = None
    solver = None
    exact_solver = None
    grouping_model = None
    group_feature_names: tuple[str, ...] = ()
    if partition_ranker_path is not None:
        import joblib
        import numpy as np

        from stroke_grouping_v1 import FEATURE_NAMES as GROUP_FEATURE_NAMES
        from stroke_grouping_v1 import build_lattice, candidate_features

        ranker_payload = joblib.load(partition_ranker_path)
        grouping_model = ranker_payload["grouping_model"]
        group_feature_names = tuple(GROUP_FEATURE_NAMES)
        if getattr(grouping_model, "n_features_in_", len(group_feature_names)) != len(group_feature_names):
            raise AssertionError("partition ranker feature count does not match grouping contract")
        score_feature_effects = {name: [] for name in group_feature_names}

        def score_feature_matrix(features) -> Any:
            probability = grouping_model.predict_proba(features)[:, 1]
            probability = np.clip(probability, 1e-6, 1.0 - 1e-6)
            return np.log(probability / (1.0 - probability))

        def batch_score(rows: list[dict[str, Any]], strokes: list[dict[str, Any]]) -> list[float]:
            return score_feature_matrix(candidate_features(rows, strokes)).tolist()

        solver = Selective2DAnytimeSolverV1(
            config=Selective2DConfigV1(), group_score_batch=batch_score,
        )
        exact_solver = Selective2DAnytimeSolverV1(
            config=Selective2DConfigV1(exact_partition_search=True),
            group_score_batch=batch_score,
        )

    for trace in traces:
        sample_id = str(trace["sample_id"])
        if sample_id not in formulas:
            raise AssertionError(f"formula missing from source data: {sample_id}")
        source = formulas[sample_id]
        grouping = trace["grouping"]
        selected = [
            [int(index) for index in group]
            for group in grouping.get("fast_incumbent_groups", grouping["fast_groups"])
        ]
        target = [
            [int(index) for index in group]
            for group in trace["source"]["target_grouping"]
        ]
        target_sets = [frozenset(group) for group in target]
        if bool(grouping["fast_group_exact"]) != (_error_shape(selected, target) == "exact"):
            raise AssertionError(f"grouping exactness mismatch: {sample_id}")

        shape = _error_shape(selected, target)
        reported_region = set(
            int(index) for index in grouping.get("local_region_strokes", [])
        )
        seed_region = set(
            int(index) for index in grouping.get(
                "local_region_seed_strokes", reported_region,
            )
        )
        if "local_region_seed_strokes" in grouping:
            region = reported_region
        else:
            # Older traces stored router seeds under local_region_strokes.
            region = set()
            for group in selected:
                if set(group) & seed_region:
                    region.update(group)
        region = frozenset(region)
        target_outside = {group for group in target_sets if not group & region}
        if "locked_fast_groups" in grouping:
            locked = [
                frozenset(int(index) for index in group)
                for group in grouping["locked_fast_groups"]
            ]
        else:
            locked = [frozenset(group) for group in selected if not set(group) & region]
        schedule = [int(value) for value in grouping.get("partition_schedule_completed", [])]
        stored_local_count = int(grouping.get("candidate_groups_local") or 0)
        reconstructed_local_count = None
        target_local_count = 0
        target_local_groups_found = 0
        target_groups_crossing_region = []
        locked_groups_match_target = None
        target_partition_reachable = None
        search_replay = None
        router_cue_audit = None

        if shape == "exact":
            outcome = "fast_partition_exact"
            target_partition_reachable = True
        elif not region:
            outcome = "router_not_triggered"
        elif not schedule or stored_local_count == 0:
            outcome = "local_search_not_run"
        else:
            strokes = sorted(source["strokes"], key=lambda row: int(row["order"]))
            try:
                local_rows = build_local_hypergraph(strokes, region, locked, config)
            except ValueError as error:
                raise ValueError(f"local partition replay failed for {sample_id}: {error}") from error
            reconstructed_local_count = len(local_rows)
            if reconstructed_local_count != stored_local_count:
                raise AssertionError(
                    f"local graph count mismatch for {sample_id}: "
                    f"reconstructed={reconstructed_local_count} stored={stored_local_count}"
                )
            rebuilt_local_graphs += 1
            local_candidates = {
                frozenset(int(index) for index in row["source_indices"])
                for row in local_rows
            }
            target_groups_crossing_region = [
                sorted(group)
                for group in target_sets
                if group & region and not group <= region
            ]
            target_outside = {group for group in target_sets if not group & region}
            locked_groups_match_target = set(locked) == target_outside
            local_target_groups = [group for group in target_sets if group <= region]
            target_local_count = len(local_target_groups)
            target_local_groups_found = sum(group in local_candidates for group in local_target_groups)
            target_partition_reachable = (
                not target_groups_crossing_region
                and locked_groups_match_target
                and target_local_groups_found == target_local_count
            )
            if target_groups_crossing_region or not locked_groups_match_target:
                outcome = "gold_partition_blocked_by_locked_boundary"
            elif target_local_groups_found != target_local_count:
                outcome = "target_group_missing_from_local_hypergraph"
            else:
                outcome = "gold_partition_reachable_but_not_promoted"

            if target_partition_reachable and solver is not None:
                strokes = sorted(source["strokes"], key=lambda row: int(row["order"]))
                local_scores = solver._score_rows(local_rows, strokes)
                score_by_group = {
                    frozenset(int(index) for index in row["source_indices"]): float(score)
                    for row, score in zip(local_rows, local_scores, strict=True)
                }
                feature_matrix = candidate_features(local_rows, strokes)
                if feature_matrix.shape[1] != len(GROUP_FEATURE_NAMES):
                    raise AssertionError("local grouping feature contract mismatch")
                feature_by_group = {
                    frozenset(int(index) for index in row["source_indices"]): {
                        name: float(value)
                        for name, value in zip(GROUP_FEATURE_NAMES, feature_values, strict=True)
                    }
                    for row, feature_values in zip(local_rows, feature_matrix, strict=True)
                }
                target_partition = [frozenset(group) for group in local_target_groups]
                target_partition.extend(frozenset(group) for group in target_outside)
                incumbent = [
                    frozenset(int(index) for index in group)
                    for group in grouping.get("fast_incumbent_groups", grouping["fast_groups"])
                ]
                target_key = _partition_key(target_partition)
                incumbent_key = _partition_key(incumbent)
                target_score = sum(score_by_group[group] for group in target_partition)
                incumbent_score = sum(score_by_group[group] for group in incumbent)
                target_group_score_decomposition = []
                target_candidate_cap_exclusions = []
                local_row_index = {
                    frozenset(int(index) for index in row["source_indices"]): index
                    for index, row in enumerate(local_rows)
                }
                for target_group in local_target_groups:
                    target_key_group = frozenset(target_group)
                    overlapping_incumbent = [
                        group for group in incumbent if group & target_key_group
                    ]
                    target_row_index = local_row_index[target_key_group]
                    incident_ranks = {}
                    for stroke_index in sorted(target_key_group):
                        incident = [
                            index for index, row in enumerate(local_rows)
                            if stroke_index in row["source_indices"]
                        ]
                        incident.sort(key=lambda index: local_scores[index], reverse=True)
                        incident_ranks[str(stroke_index)] = incident.index(target_row_index) + 1
                    cap_excluded = any(rank > config.max_candidates_per_stroke for rank in incident_ranks.values())
                    candidate_cap_exclusion = {
                        "target_group": sorted(target_key_group),
                        "incident_candidate_rank_by_stroke": incident_ranks,
                        "excluded_by_per_stroke_cap": cap_excluded,
                    }
                    target_candidate_cap_exclusions.append(candidate_cap_exclusion)
                    comparison_groups = [target_key_group, *overlapping_incumbent]
                    if any(group not in feature_by_group for group in comparison_groups):
                        raise AssertionError(
                            f"score-feature row missing for target/fast comparison: {sample_id}"
                        )
                    comparison_matrix = np.asarray([
                        [feature_by_group[group][name] for name in group_feature_names]
                        for group in comparison_groups
                    ], dtype=np.float32)
                    comparison_logits = score_feature_matrix(comparison_matrix)
                    observed_score_delta = (
                        score_by_group[target_key_group]
                        - sum(score_by_group[group] for group in overlapping_incumbent)
                    )
                    reconstructed_score_delta = float(
                        comparison_logits[0] - comparison_logits[1:].sum()
                    )
                    score_feature_counterfactual_checks += 1
                    score_feature_counterfactual_max_abs_delta = max(
                        score_feature_counterfactual_max_abs_delta,
                        abs(reconstructed_score_delta - observed_score_delta),
                    )
                    if abs(reconstructed_score_delta - observed_score_delta) > 1e-5:
                        raise AssertionError(
                            f"feature-row score replay mismatch: {sample_id}"
                        )
                    neutralization_effects = {}
                    if observed_score_delta < -1e-6:
                        negative_score_feature_comparisons += 1
                    for feature_index, feature_name in enumerate(group_feature_names):
                        neutralized = comparison_matrix.copy()
                        neutralized[:, feature_index] = float(
                            comparison_matrix[:, feature_index].mean()
                        )
                        neutralized_logits = score_feature_matrix(neutralized)
                        neutralized_delta = float(
                            neutralized_logits[0] - neutralized_logits[1:].sum()
                        )
                        delta_change = neutralized_delta - reconstructed_score_delta
                        neutralization_effects[feature_name] = {
                            "counterfactual_target_minus_fast_logit": neutralized_delta,
                            "delta_change_after_equalizing_feature": delta_change,
                        }
                        if observed_score_delta < -1e-6:
                            score_feature_effects[feature_name].append(delta_change)
                            score_feature_flip_counts[feature_name] += int(
                                neutralized_delta >= -1e-6
                            )
                    target_group_score_decomposition.append({
                        **candidate_cap_exclusion,
                        "target_group_score": score_by_group[target_key_group],
                        "target_group_score_features": feature_by_group[target_key_group],
                        "overlapping_fast_groups": [sorted(group) for group in overlapping_incumbent],
                        "overlapping_fast_group_score_features": [
                            {
                                "group": sorted(group),
                                "score": score_by_group[group],
                                "features": feature_by_group[group],
                            }
                            for group in overlapping_incumbent
                        ],
                        "overlapping_fast_group_score_sum": sum(
                            score_by_group[group] for group in overlapping_incumbent
                        ),
                        "target_minus_fast_local_score": (
                            observed_score_delta
                        ),
                        "one_feature_equalization_counterfactuals": neutralization_effects,
                        "counterfactual_scope": (
                            "posthoc diagnostic only; equalizes one feature within this paired comparison; "
                            "not causal proof and not used for threshold selection"
                        ),
                    })
                    score_feature_comparisons += 1
                decomposition_delta = sum(
                    row["target_minus_fast_local_score"]
                    for row in target_group_score_decomposition
                )
                incumbent_overlap_counts = {
                    group: sum(bool(group & target_group) for target_group in local_target_groups)
                    for group in incumbent
                    if group & region
                }
                score_decomposition_is_additive = all(
                    count == 1 for count in incumbent_overlap_counts.values()
                )
                score_decomposition_reconciled = (
                    not score_decomposition_is_additive
                    or abs(decomposition_delta - (target_score - incumbent_score)) <= 1e-5
                )
                if not score_decomposition_reconciled:
                    raise AssertionError(
                        f"target-vs-fast score decomposition mismatch for {sample_id}: "
                        f"decomposition={decomposition_delta} total={target_score - incumbent_score}"
                    )
                trace_target_score = grouping.get("target_partition_score")
                target_score_trace_delta = None
                if trace_target_score is not None:
                    target_score_trace_delta = target_score - float(trace_target_score)
                    score_reconciliations += 1
                    score_reconciliation_max_abs_delta = max(
                        score_reconciliation_max_abs_delta,
                        abs(target_score_trace_delta),
                    )
                    if abs(target_score_trace_delta) > 1e-5:
                        raise AssertionError(
                            f"target grouping score replay mismatch for {sample_id}: "
                            f"local={target_score} trace={trace_target_score}"
                        )
                seen: set[tuple[tuple[int, ...], ...]] = set()
                target_rank = None
                target_ranked_score = None
                exact_target_rank = None
                exact_search_mode = None
                winner = incumbent
                winner_key = incumbent_key
                winner_score = incumbent_score
                schedule_modes = []
                schedule_completed = []
                exact_cover_checks = []
                for top_n in config.partition_schedule:
                    alternatives, search_mode = solver._rank_partitions(
                        local_rows, local_scores, len(strokes), top_n=top_n,
                    )
                    schedule_completed.append(top_n)
                    schedule_modes.append(search_mode)
                    for geometry_score, groups in alternatives:
                        key = _partition_key(groups)
                        if key in seen:
                            continue
                        seen.add(key)
                        replayed_partitions += 1
                        assigned = sorted(index for group in groups for index in group)
                        exact_cover_checks.append(assigned == list(range(len(strokes))))
                        if key == target_key:
                            target_rank = len(seen)
                            target_ranked_score = float(geometry_score)
                        if float(geometry_score) >= winner_score + config.acceptance_margin:
                            winner = list(groups)
                            winner_key = key
                            winner_score = float(geometry_score)
                    if len(alternatives) < top_n:
                        break
                if any(not exact for exact in exact_cover_checks):
                    raise AssertionError(f"local search replay emitted a non-exact cover: {sample_id}")
                if len(strokes) <= 12:
                    exact_alternatives, exact_search_mode = exact_solver._rank_partitions(
                        local_rows, local_scores, len(strokes), top_n=32,
                    )
                    exact_target_rank = next((
                        index + 1 for index, (_score, groups) in enumerate(exact_alternatives)
                        if _partition_key(groups) == target_key
                    ), None)
                replayed_searches += 1
                replayed_exact_covers += len(exact_cover_checks)
                selected_key = _partition_key(grouping.get("groups", []))
                if target_rank is None:
                    if any(row["excluded_by_per_stroke_cap"] for row in target_candidate_cap_exclusions):
                        replay_outcome = "target_partition_excluded_by_per_stroke_candidate_cap"
                    elif exact_target_rank is not None:
                        replay_outcome = "target_partition_pruned_by_beam_before_top32"
                    elif exact_search_mode is not None:
                        replay_outcome = "target_partition_ranked_below_exact_capped_top32"
                    else:
                        replay_outcome = "target_partition_not_in_beam_top32_full_formula_over_12_strokes"
                elif winner_key == target_key:
                    replay_outcome = "target_partition_geometry_winner"
                elif target_score < incumbent_score + config.acceptance_margin:
                    replay_outcome = "target_partition_below_acceptance_margin"
                else:
                    replay_outcome = "higher_scoring_challenger_won_geometry_search"
                search_outcomes[replay_outcome] += 1
                search_replay = {
                    "mode": schedule_modes,
                    "schedule_completed": schedule_completed,
                    "unique_partitions_enumerated": len(seen),
                    "target_partition_rank_in_local_top32": target_rank,
                    "target_partition_score": target_score,
                    "trace_fast_lattice_target_partition_score": trace_target_score,
                    "target_score_delta_vs_trace_fast_lattice": target_score_trace_delta,
                    "target_partition_score_delta_vs_fast_incumbent": target_score - incumbent_score,
                    "target_group_score_decomposition": target_group_score_decomposition,
                    "score_decomposition_is_additive": score_decomposition_is_additive,
                    "score_decomposition_reconciled": score_decomposition_reconciled,
                    "fast_groups_overlapping_multiple_target_groups": [
                        sorted(group) for group, count in incumbent_overlap_counts.items() if count > 1
                    ],
                    "target_partition_candidate_cap_audit": target_candidate_cap_exclusions,
                    "target_partition_any_edge_excluded_by_candidate_cap": any(
                        row["excluded_by_per_stroke_cap"] for row in target_candidate_cap_exclusions
                    ),
                    "exact_search_mode": exact_search_mode,
                    "exact_capped_target_partition_rank_in_top32": exact_target_rank,
                    "target_partition_was_in_geometry_shortlist": target_rank is not None,
                    "target_partition_score_seen_during_search": target_ranked_score,
                    "fast_incumbent_score": incumbent_score,
                    "geometry_winner_score": winner_score,
                    "geometry_winner_groups": [list(group) for group in winner],
                    "geometry_winner_matches_recorded_preview": winner_key == selected_key,
                    "geometry_winner_matches_target": winner_key == target_key,
                    "target_ranked_but_preview_not_target": target_rank is not None and selected_key != target_key,
                    "replay_outcome": replay_outcome,
                }

        if outcome == "router_not_triggered" and solver is not None:
            strokes = sorted(source["strokes"], key=lambda row: int(row["order"]))
            symbol_margins = {}
            for symbol in grouping.get("selected_symbols", []):
                prediction = symbol.get("hwr_prediction") or {}
                indices = frozenset(int(index) for index in symbol.get("stroke_indices", []))
                if indices and prediction.get("top1_top2_probability_margin") is not None:
                    symbol_margins[indices] = float(prediction["top1_top2_probability_margin"])
            baseline = build_lattice(
                strokes, temporal_window=config.temporal_window,
                spatial_neighbors=config.spatial_neighbors,
            )
            baseline_scores = solver._score_rows(baseline, strokes)
            fast_ranked, fast_mode = solver._rank_partitions(
                baseline, baseline_scores, len(strokes), top_n=2,
            )
            _replayed_seeds, replayed_reasons = _risk_seeds(
                strokes, baseline, fast_ranked, symbol_margins, config,
            )
            partition_gap = (
                float(fast_ranked[0][0] - fast_ranked[1][0])
                if len(fast_ranked) > 1 else None
            )
            minimum_symbol_margin = min(symbol_margins.values()) if symbol_margins else None
            router_cue_audit = {
                "fast_partition_search_mode": fast_mode,
                "fast_top1_top2_partition_score_gap": partition_gap,
                "partition_margin_threshold": config.partition_margin,
                "partition_margin_triggered": (
                    partition_gap is not None and partition_gap <= config.partition_margin
                ),
                "minimum_fast_group_hwr_margin": minimum_symbol_margin,
                "symbol_margin_threshold": config.symbol_margin,
                "symbol_margin_triggered": (
                    minimum_symbol_margin is not None
                    and minimum_symbol_margin <= config.symbol_margin
                ),
                "replayed_risk_reasons": replayed_reasons,
                "replay_matches_trace_risk_reasons": (
                    replayed_reasons == list(grouping.get("risk_reasons") or [])
                ),
            }
            replayed_router_cues += 1

        outcomes[outcome] += 1
        if shape != "exact":
            outcomes_by_shape.setdefault(shape, Counter())[outcome] += 1
        records.append({
            "sample_id": sample_id,
            "fast_group_exact": shape == "exact",
            "partition_error_shape": shape,
            "target_partition_rank_in_fast_top32": grouping.get("target_partition_rank_in_top32"),
            "risk_reasons": grouping.get("risk_reasons", []),
            "local_region_seed_stroke_count": len(seed_region),
            "local_region_stroke_count": len(region),
            "local_region_strokes": sorted(region),
            "local_candidate_count_stored": stored_local_count,
            "local_candidate_count_reconstructed": reconstructed_local_count,
            "partition_schedule_completed": schedule,
            "target_local_group_count": target_local_count,
            "target_local_groups_found": target_local_groups_found,
            "target_groups_crossing_region": target_groups_crossing_region,
            "target_groups_outside_region": sorted([sorted(group) for group in target_outside]) if shape != "exact" else [],
            "locked_groups_recorded": sorted([sorted(group) for group in locked]) if region else [],
            "locked_groups_missing_from_target_outside": (
                sorted([sorted(group) for group in set(locked) - target_outside])
                if region and shape != "exact" else []
            ),
            "target_outside_groups_missing_from_locks": (
                sorted([sorted(group) for group in target_outside - set(locked)])
                if region and shape != "exact" else []
            ),
            "locked_groups_match_target": locked_groups_match_target,
            "target_partition_reachable_in_local_graph": target_partition_reachable,
            "local_search_skip_reason": grouping.get("skip_reason"),
            "router_cue_audit": router_cue_audit,
            "local_partition_search_replay": search_replay,
            "winner_score_delta": grouping.get("winner_score_delta"),
            "outcome": outcome,
        })

    feature_comparison_rows = [
        comparison
        for record in records
        for comparison in (
            (record.get("local_partition_search_replay") or {}).get(
                "target_group_score_decomposition", []
            )
        )
    ]
    verification = {
        "all_trace_formulas_present_in_source": len(records) == len(traces),
        "all_reconstructed_local_graph_counts_match_runtime": rebuilt_local_graphs == sum(
            bool(row["local_candidate_count_reconstructed"] is not None)
            for row in records
        ),
        "all_grouping_error_shapes_classified": sum(
            sum(counts.values()) for counts in outcomes_by_shape.values()
        ) == sum(not row["fast_group_exact"] for row in records),
            "all_replayed_local_partitions_preserve_exact_cover": replayed_exact_covers == replayed_partitions,
        "all_local_target_and_fast_score_features_recorded": (
            partition_ranker_path is None
            or (
                score_feature_comparisons > 0
                and score_feature_comparisons == len(feature_comparison_rows)
                and all(
                    "target_group_score_features" in row
                    and len(row.get("overlapping_fast_group_score_features", []))
                    == len(row.get("overlapping_fast_groups", []))
                    for row in feature_comparison_rows
                )
            )
        ),
        "all_feature_row_scores_reconcile_with_saved_group_scores": (
            partition_ranker_path is None
            or (
                score_feature_counterfactual_checks == score_feature_comparisons
                and score_feature_counterfactual_max_abs_delta <= 1e-5
            )
        ),
            "all_replayed_fast_lattice_target_scores_reconcile": (
                score_reconciliation_max_abs_delta <= 1e-5
            ),
            "all_additive_target_group_score_decompositions_reconcile": all(
                row["local_partition_search_replay"]["score_decomposition_reconciled"]
                for row in records
                if (row.get("local_partition_search_replay") or {}).get("score_decomposition_is_additive")
            ),
            "all_router_cue_replays_match_trace": (
                solver is None or all(
                    row.get("router_cue_audit", {}).get("replay_matches_trace_risk_reasons") is True
                    for row in records if row["outcome"] == "router_not_triggered"
                )
            ),
            "all_reachable_local_targets_replayed_when_ranker_supplied": (
                partition_ranker_path is None
                or replayed_searches == sum(
                    bool(row["target_partition_reachable_in_local_graph"])
                    for row in records if row["partition_error_shape"] != "exact"
                )
            ),
    }
    return {
        "schema": SCHEMA,
        "scope": (
            "forensic local-hypergraph reachability only; frozen project-owned traces; "
            "no training, CROHME, threshold selection, or promotion"
        ),
        "inputs": {
            "traces": str(traces_path),
            "traces_sha256": _sha256(traces_path),
            "formulas": str(formulas_path),
            "formulas_sha256": _sha256(formulas_path),
            "selective_2d_config": config.__dict__,
            "partition_ranker": str(partition_ranker_path) if partition_ranker_path else None,
            "partition_ranker_sha256": _sha256(partition_ranker_path) if partition_ranker_path else None,
            "local_group_score_feature_names": (
                list(GROUP_FEATURE_NAMES) if partition_ranker_path is not None else None
            ),
        },
        "summary": {
            "formulas": len(records),
            "fast_group_errors": sum(not row["fast_group_exact"] for row in records),
            "outcomes": dict(sorted(outcomes.items())),
            "outcomes_by_partition_error_shape": {
                shape: dict(sorted(counts.items()))
                for shape, counts in sorted(outcomes_by_shape.items())
            },
            "local_graphs_reconstructed": rebuilt_local_graphs,
            "local_partition_search_replays": replayed_searches,
            "local_partition_partitions_enumerated": replayed_partitions,
            "local_partition_exact_cover_partitions": replayed_exact_covers,
            "local_target_vs_fast_score_feature_comparisons": score_feature_comparisons,
            "one_feature_equalization_counterfactual_checks": score_feature_counterfactual_checks,
            "negative_target_vs_fast_score_comparisons": negative_score_feature_comparisons,
            "feature_equalization_effects_on_negative_score_comparisons": {
                name: {
                    "comparisons": len(values),
                    "mean_delta_change": (sum(values) / len(values)) if values else None,
                    "counterfactual_flips_to_nonnegative": score_feature_flip_counts[name],
                }
                for name, values in sorted(
                    score_feature_effects.items(),
                    key=lambda item: (
                        -(sum(item[1]) / len(item[1])) if item[1] else float("inf"),
                        item[0],
                    ),
                )
            },
            "feature_equalization_counterfactual_max_abs_score_delta": (
                score_feature_counterfactual_max_abs_delta
            ),
            "fast_lattice_target_scores_reconciled": score_reconciliations,
            "fast_lattice_target_score_reconciliation_max_abs_delta": score_reconciliation_max_abs_delta,
            "router_cue_replays": replayed_router_cues,
            "local_targets_excluded_by_per_stroke_candidate_cap": sum(
                bool(row["local_partition_search_replay"].get("target_partition_any_edge_excluded_by_candidate_cap"))
                for row in records if row.get("local_partition_search_replay")
            ),
            "local_partition_search_replay_outcomes": dict(sorted(search_outcomes.items())),
        },
        "verification": {"checks": verification, "all_checks_pass": all(verification.values())},
        "formula_level": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-traces", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--partition-ranker", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = diagnose(args.input_traces, args.dataset_root, args.partition_ranker)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": report["summary"], "verification": report["verification"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
