#!/usr/bin/env python3
"""Run the local-only Selective-2D diagnostic loop.

No fitting, external collection, CROHME access, or product configuration is
performed.  The report separates oracle-group HWR headroom from Fast-vs-local
grouping behavior so one cannot hide the other.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
import subprocess
import sys
from time import perf_counter
from pathlib import Path

def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _partition(groups: list[list[int]]) -> set[frozenset[int]]:
    return {frozenset(int(index) for index in group) for group in groups}


def _partition_error_shape(groups: list[list[int]], target_groups: list[list[int]]) -> str:
    if _partition(groups) == _partition(target_groups):
        return "exact"
    if len(groups) > len(target_groups):
        return "oversegmentation"
    if len(groups) < len(target_groups):
        return "undersegmentation"
    return "same_count_wrong_partition"


SEMANTIC_GUARD_STAGES = (
    "decoder", "after_fence_guard", "after_infix_guard", "after_equation_guard",
    "after_unique_bar_equation_guard", "after_unique_exact_equation_candidates_guard",
    "after_boundary_bar_as_unit_guard",
    "after_unique_candidate_arithmetic_equation_guard", "after_terminal_rhs_bar_guard",
)
FINAL_SEMANTIC_STAGE = SEMANTIC_GUARD_STAGES[-1]


def _preview_metrics(preview: dict, target_groups: list[list[int]], target_tokens: list[str]) -> dict:
    groups = [[int(index) for index in group] for group in preview["groups"]]
    group_exact = _partition(groups) == _partition(target_groups)
    group_count_delta = len(groups) - len(target_groups)
    group_error_shape = _partition_error_shape(groups, target_groups)
    if (group_error_shape == "exact") != group_exact:
        raise AssertionError("A/B/C partition error classification is inconsistent")
    decoder = dict(preview.get("decoder") or {})
    audit = dict(preview.get("audit") or {})
    if preview.get("status") != "development_only_posthoc_shadow":
        raise AssertionError("A/B/C preview escaped its shadow-only runtime")
    if audit.get("all_strokes_exactly_once") is not True:
        raise AssertionError("A/B/C preview violated exact stroke cover")
    if audit.get("crohme_training_or_tuning") is not False:
        raise AssertionError("A/B/C preview violated the CROHME boundary")
    if audit.get("product_default_enabled") is not False:
        raise AssertionError("A/B/C preview enabled product defaults")
    guard_config = audit.get("config") or {}
    guard_enabled = bool(audit.get(
        "local_group_count_increase_guard_enabled",
        guard_config.get("reject_local_group_count_increase", False),
    ))
    fast_incumbent_groups = audit.get("fast_incumbent_groups") or []
    if guard_enabled and len(groups) > len(fast_incumbent_groups):
        raise AssertionError("group-count guard returned more groups than the Fast incumbent")
    symbols = list(preview.get("symbols") or [])
    if len(symbols) != int(audit.get("selected_candidate_count", -1)):
        raise AssertionError("A/B/C symbol and selected-group counts differ")
    joint_encoded = int(audit.get("joint_hwr_candidates_encoded", 0))
    joint_budget = int(audit.get("joint_hwr_candidate_budget", 0))
    if joint_encoded > joint_budget:
        raise AssertionError("A/B/C exceeded its local HWR candidate budget")
    if bool(audit.get("joint_hwr_scoring_enabled")) != (joint_budget > 0):
        raise AssertionError("A/B/C joint-HWR budget metadata is inconsistent")
    joint_diagnostics = audit.get("joint_hwr_partition_diagnostics")
    joint_score_search = audit.get("joint_score_search")
    if joint_budget > 0:
        if not isinstance(joint_diagnostics, dict):
            raise AssertionError(
                f"A/B/C joint-HWR partition diagnostics are missing: "
                f"formula={preview.get('formula_id')} route={preview.get('route')}"
            )
        evaluated = int(joint_diagnostics.get("partitions_evaluated", -1))
        diagnostic_outcomes = sum(int(joint_diagnostics.get(key, -1)) for key in (
            "candidate_row_missing", "candidate_budget_rejected", "decoder_rejected",
            "decoder_accepted",
        ))
        if evaluated != diagnostic_outcomes:
            raise AssertionError("A/B/C joint-HWR partition outcomes do not reconcile")
        if sum((joint_diagnostics.get("decoder_rejection_reasons") or {}).values()) != int(
            joint_diagnostics.get("decoder_rejected", -1)
        ):
            raise AssertionError("A/B/C joint-HWR decoder rejection reasons do not reconcile")
        if evaluated == 0:
            if joint_score_search is not None or audit.get("partition_schedule_completed"):
                raise AssertionError(
                    f"A/B/C zero-evaluation route has inconsistent search metadata: "
                    f"formula={preview.get('formula_id')} route={preview.get('route')}"
                )
        else:
            if not isinstance(joint_score_search, dict):
                raise AssertionError(
                    f"A/B/C joint-score search details are missing: "
                    f"formula={preview.get('formula_id')} route={preview.get('route')} "
                    f"evaluated={evaluated}"
                )
            if int(joint_score_search.get("evaluations", -1)) != evaluated:
                raise AssertionError("A/B/C joint scorer and decoder evaluation counts differ")
            if int(joint_score_search.get("finite", -1)) != int(
                joint_diagnostics.get("decoder_accepted", -1)
            ):
                raise AssertionError("A/B/C finite scores and accepted decodes differ")
            if int(joint_score_search.get("evaluations", -1)) != int(
                joint_score_search.get("finite", -1)
            ) + int(joint_score_search.get("non_finite", -1)):
                raise AssertionError("A/B/C joint score finite/non-finite counts do not reconcile")
    elif joint_diagnostics is not None or joint_score_search is not None:
        raise AssertionError("A/B/C router-off run contains joint-HWR diagnostics")
    if decoder.get("accepted") and decoder.get("top5_preserved") is not True:
        raise AssertionError("A/B/C decoder invented a token outside HWR Top-5")
    semantic_shadow = preview.get("semantic_guard_shadow")
    semantic_guard_metrics = None
    if semantic_shadow is not None:
        if audit.get("semantic_guard_shadow_enabled") is not True:
            raise AssertionError("semantic guard shadow is missing its opt-in audit flag")
        if semantic_shadow.get("product_default_enabled") is not False:
            raise AssertionError("semantic guard shadow enabled a product default")
        if semantic_shadow.get("status") == "applied_shadow_only":
            if semantic_shadow.get("candidate_preservation") is not True:
                raise AssertionError("semantic guard shadow violated candidate preservation")
            if any(int(semantic_shadow.get(key, -1)) != 0 for key in (
                "new_tokens", "deleted_glyphs", "grouping_mutations",
            )):
                raise AssertionError("semantic guard shadow changed token/group ownership")
            if semantic_shadow.get("base_decoder_tokens") != decoder.get("tokens"):
                raise AssertionError("semantic guard shadow mutated the base decoder tokens")
        elif decoder.get("accepted"):
            raise AssertionError("semantic guard shadow skipped an accepted decoder result")
    token_hits = None
    token_exact = None
    hwr_top1_hits = None
    hwr_top5_hits = None
    hwr_target_rank_histogram = None
    hwr_top5_complete = None
    if group_exact and decoder.get("accepted"):
        if len(target_groups) != len(target_tokens) or len(groups) != len(decoder.get("tokens") or []):
            raise AssertionError("group/token cardinality mismatch in HWR tournament")
    if group_exact:
        if len(target_groups) != len(target_tokens):
            raise AssertionError("target group/token cardinality mismatch in HWR tournament")
        truth_by_group = {
            frozenset(int(index) for index in group): str(token)
            for group, token in zip(target_groups, target_tokens, strict=True)
        }
        symbols_by_group = {
            frozenset(int(index) for index in symbol.get("stroke_indices") or []): symbol
            for symbol in symbols
        }
        if len(symbols_by_group) != len(groups) or any(
            frozenset(group) not in symbols_by_group for group in groups
        ):
            raise AssertionError("HWR symbol coverage mismatch in HWR tournament")
        hwr_top1_hits = 0
        hwr_top5_hits = 0
        hwr_target_rank_histogram = {}
        hwr_top5_complete = True
        for group in groups:
            symbol = symbols_by_group[frozenset(group)]
            target = truth_by_group[frozenset(group)]
            topk = [str(token) for token in symbol.get("hwr_topk") or []]
            if len(topk) != 5 or len(set(topk)) != 5:
                raise AssertionError("HWR tournament did not preserve Top-5")
            rank = topk.index(target) + 1 if target in topk else None
            rank_key = str(rank) if rank is not None else "missing_or_outside_top5"
            hwr_target_rank_histogram[rank_key] = hwr_target_rank_histogram.get(rank_key, 0) + 1
            hwr_top1_hits += int(bool(topk) and topk[0] == target)
            hwr_top5_hits += int(rank is not None)
            hwr_top5_complete = hwr_top5_complete and rank is not None
        if decoder.get("accepted"):
            token_hits = sum(
                str(predicted) == truth_by_group[frozenset(group)]
                for group, predicted in zip(groups, decoder["tokens"], strict=True)
            )
            token_exact = token_hits == len(target_tokens)
        if semantic_shadow is not None and semantic_shadow.get("status") == "applied_shadow_only":
            stage_maps = {}
            expected_group_keys = {frozenset(group) for group in groups}
            for stage in SEMANTIC_GUARD_STAGES:
                rows = list((semantic_shadow.get("stages") or {}).get(stage) or [])
                by_group = {
                    frozenset(int(index) for index in row.get("stroke_indices") or []): str(row["token"])
                    for row in rows
                }
                if len(by_group) != len(rows) or set(by_group) != expected_group_keys:
                    raise AssertionError(f"semantic guard stage {stage} changed group ownership")
                if any(
                    token not in symbols_by_group[group_key]["hwr_topk"]
                    for group_key, token in by_group.items()
                ):
                    raise AssertionError(f"semantic guard stage {stage} emitted outside HWR Top-5")
                stage_maps[stage] = by_group
            semantic_guard_metrics = {
                "token_count": len(target_tokens),
                "token_hits_by_stage": {},
                "formula_exact_by_stage": {},
                "formula_exact_transitions": {},
                "unique_unpaired_bar_equation": dict(
                    (semantic_shadow.get("audits") or {}).get("unique_unpaired_bar_equation") or {}
                ),
                "semantic_audits": dict(semantic_shadow.get("audits") or {}),
            }
            for stage in SEMANTIC_GUARD_STAGES:
                predictions = [stage_maps[stage][frozenset(group)] for group in target_groups]
                hits = sum(
                    predicted == truth
                    for predicted, truth in zip(predictions, target_tokens, strict=True)
                )
                semantic_guard_metrics["token_hits_by_stage"][stage] = hits
                semantic_guard_metrics["formula_exact_by_stage"][stage] = predictions == target_tokens
            for before, after in zip(SEMANTIC_GUARD_STAGES, SEMANTIC_GUARD_STAGES[1:]):
                semantic_guard_metrics["formula_exact_transitions"][f"{before}_to_{after}"] = (
                    "recovered" if (
                        not semantic_guard_metrics["formula_exact_by_stage"][before]
                        and semantic_guard_metrics["formula_exact_by_stage"][after]
                    ) else "regressed" if (
                        semantic_guard_metrics["formula_exact_by_stage"][before]
                        and not semantic_guard_metrics["formula_exact_by_stage"][after]
                    ) else "both_exact" if (
                        semantic_guard_metrics["formula_exact_by_stage"][before]
                        and semantic_guard_metrics["formula_exact_by_stage"][after]
                    ) else "both_wrong"
                )
    return {
        "route": str(preview.get("route", "unknown")),
        "groups": groups,
        "target_groups": [[int(index) for index in group] for group in target_groups],
        "target_tokens": [str(token) for token in target_tokens],
        "group_exact": group_exact,
        "group_count_delta": group_count_delta,
        "group_error_shape": group_error_shape,
        "decoder_accepted": bool(decoder.get("accepted")),
        "decoder_reason": decoder.get("reason"),
        "decoder_tokens": [str(token) for token in decoder.get("tokens") or []],
        "decoder_latex": decoder.get("latex"),
        "decoder_joint_score": decoder.get("joint_token_relation_score"),
        "selected_symbols": [
            {
                "stroke_indices": [int(index) for index in symbol.get("stroke_indices") or []],
                "hwr_topk": [str(token) for token in symbol.get("hwr_topk") or []],
                "hwr_topk_probabilities": [
                    float(value) for value in symbol.get("hwr_topk_probabilities") or []
                ],
            }
            for symbol in symbols
        ],
        "fast_incumbent_groups": audit.get("fast_incumbent_groups"),
        "local_group_count_increase_guard_enabled": guard_enabled,
        "local_group_count_increase_rejections": int(
            audit.get("local_group_count_increase_rejections", 0)
        ),
        "risk_reasons": audit.get("risk_reasons"),
        "risk_strokes": audit.get("risk_strokes"),
        "local_region_strokes": audit.get("local_region_strokes"),
        "locked_fast_groups": audit.get("locked_fast_groups"),
        "candidate_groups_local": audit.get("candidate_groups_local"),
        "partition_schedule_completed": audit.get("partition_schedule_completed"),
        "token_hits_if_groups_exact": token_hits,
        "token_exact_if_groups_exact": token_exact,
        "hwr_top1_hits_if_groups_exact": hwr_top1_hits,
        "hwr_top5_hits_if_groups_exact": hwr_top5_hits,
        "hwr_target_token_count_if_groups_exact": len(groups) if group_exact else None,
        "hwr_top5_complete_if_groups_exact": hwr_top5_complete,
        "hwr_target_rank_histogram_if_groups_exact": hwr_target_rank_histogram,
        "semantic_guard_shadow_if_groups_exact": semantic_guard_metrics,
        "hwr_candidates_encoded": int(audit.get("hwr_encoded_candidates", 0)),
        "joint_hwr_candidates_encoded": int(audit.get("joint_hwr_candidates_encoded", 0)),
        "joint_hwr_budget_exhausted": bool(audit.get("joint_hwr_budget_exhausted", False)),
        "joint_geometry_prior_weight": float(audit.get("joint_geometry_prior_weight", 0.0)),
        "joint_geometry_prior_normalization": str(
            audit.get("joint_geometry_prior_normalization", "stroke_count")
        ),
        "joint_selected_score_breakdown": audit.get("joint_selected_score_breakdown"),
        "joint_incumbent_score_breakdown": audit.get("joint_incumbent_score_breakdown"),
        "joint_best_challenger_vs_fast": audit.get("joint_best_challenger_vs_fast"),
        "joint_hwr_partition_diagnostics": joint_diagnostics,
        "joint_score_search": joint_score_search,
        "local_processed_stroke_fraction": audit.get("local_processed_stroke_fraction"),
        "winner_score_delta": audit.get("winner_score_delta"),
    }


def _paired_grouping_summary(records: list[dict]) -> dict:
    def compare(left_arm: str, right_arm: str) -> dict:
        counts = Counter()
        for record in records:
            arms = record["hwr_tournament"]
            left_exact = bool(arms[left_arm]["group_exact"])
            right_exact = bool(arms[right_arm]["group_exact"])
            if left_exact and right_exact:
                counts["both_exact"] += 1
            elif left_exact:
                counts["left_exact_right_regression"] += 1
            elif right_exact:
                counts["right_only_recovery"] += 1
            else:
                counts["both_wrong"] += 1
        if sum(counts.values()) != len(records):
            raise AssertionError(f"paired {left_arm}/{right_arm} transitions do not reconcile")
        left_exact = counts["both_exact"] + counts["left_exact_right_regression"]
        right_exact = counts["both_exact"] + counts["right_only_recovery"]
        return {
            "left_arm": left_arm,
            "right_arm": right_arm,
            "exact_transition_counts": dict(sorted(counts.items())),
            "left_group_exact": left_exact,
            "right_group_exact": right_exact,
            "right_minus_left_exact_formula_delta": right_exact - left_exact,
        }

    transitions = Counter()
    local_routes = []
    local_error_shapes = Counter()
    group_count_changes = Counter()
    evaluation_totals = Counter()
    joint_score_deltas = []
    geometry_partition_mismatches = 0
    for record in records:
        arms = record["hwr_tournament"]
        fast = arms["fast"]
        geometry = arms["selective_geometry"]
        joint = arms["selective_joint_hwr"]
        fast_exact = bool(record["fast_group_exact"])
        if bool(fast["group_exact"]) != fast_exact:
            raise AssertionError(f"Fast paired grouping metric mismatch: {record['sample_id']}")
        joint_exact = bool(joint["group_exact"])
        if fast_exact and joint_exact:
            transitions["both_exact"] += 1
        elif fast_exact:
            transitions["fast_exact_joint_regression"] += 1
        elif joint_exact:
            transitions["joint_recovery"] += 1
        else:
            transitions["both_wrong"] += 1
        if _partition(geometry["groups"]) != _partition(fast["groups"]):
            geometry_partition_mismatches += 1
        if joint["route"] == "local_2d":
            local_routes.append(joint)
            local_error_shapes[joint["group_error_shape"]] += 1
            group_count_changes[str(
                len(joint["groups"]) - len(joint["fast_incumbent_groups"] or [])
            )] += 1
            if joint.get("winner_score_delta") is not None:
                joint_score_deltas.append(float(joint["winner_score_delta"]))
        diagnostics = joint.get("joint_hwr_partition_diagnostics") or {}
        search = joint.get("joint_score_search") or {}
        for key in (
            "partitions_evaluated", "decoder_accepted", "decoder_rejected",
            "candidate_row_missing", "candidate_budget_rejected",
        ):
            evaluation_totals[key] += int(diagnostics.get(key, 0))
        evaluation_totals["joint_score_promoted_partition_events"] += int(search.get("promoted", 0))

    if sum(transitions.values()) != len(records):
        raise AssertionError("paired grouping transition counts do not reconcile")
    score_deltas = sorted(joint_score_deltas)
    score_summary = None
    if score_deltas:
        score_summary = {
            "min": score_deltas[0],
            "median": score_deltas[len(score_deltas) // 2],
            "max": score_deltas[-1],
        }
    fast_exact_total = transitions["both_exact"] + transitions["fast_exact_joint_regression"]
    joint_exact_total = transitions["both_exact"] + transitions["joint_recovery"]
    prior_arm_rows = [
        record["hwr_tournament"]["selective_joint_hwr_geometry_prior"]
        for record in records
    ]

    def component_summary(name: str) -> dict | None:
        values = sorted(
            float(row["joint_selected_score_breakdown"][name])
            for row in prior_arm_rows
            if row.get("joint_selected_score_breakdown") is not None
        )
        if not values:
            return None
        return {"count": len(values), "min": values[0], "median": values[len(values) // 2], "max": values[-1]}

    def challenger_delta_summary(name: str) -> dict | None:
        values = sorted(
            float(row["joint_best_challenger_vs_fast"][name])
            for row in prior_arm_rows
            if row.get("joint_best_challenger_vs_fast") is not None
        )
        if not values:
            return None
        return {"count": len(values), "min": values[0], "median": values[len(values) // 2], "max": values[-1]}

    paired = {
        "formulas_paired_by_sample_id": len(records),
        "exact_transition_counts": dict(sorted(transitions.items())),
        "fast_group_exact": fast_exact_total,
        "joint_hwr_group_exact": joint_exact_total,
        "joint_minus_fast_exact_formula_delta": joint_exact_total - fast_exact_total,
        "joint_local_routes": len(local_routes),
        "joint_local_route_exact": sum(bool(row["group_exact"]) for row in local_routes),
        "joint_local_route_error_shapes": dict(sorted(local_error_shapes.items())),
        "joint_local_route_group_count_change_vs_fast_histogram": dict(sorted(group_count_changes.items())),
        "joint_local_route_decoder_score_delta": score_summary,
        "joint_search_counts": dict(sorted(evaluation_totals.items())),
        "geometry_shadow_partition_mismatches_vs_fast": geometry_partition_mismatches,
        "fast_vs_joint_hwr": compare("fast", "selective_joint_hwr"),
        "fast_vs_joint_hwr_geometry_prior": compare("fast", "selective_joint_hwr_geometry_prior"),
        "joint_hwr_vs_geometry_prior": compare(
            "selective_joint_hwr", "selective_joint_hwr_geometry_prior",
        ),
        "geometry_prior_arm": {
            "formulas": len(prior_arm_rows),
            "local_routes": sum(row["route"] == "local_2d" for row in prior_arm_rows),
            "group_exact": sum(bool(row["group_exact"]) for row in prior_arm_rows),
            "local_route_error_shapes": dict(sorted(Counter(
                row["group_error_shape"] for row in prior_arm_rows
                if row["route"] == "local_2d"
            ).items())),
            "selected_score_components": {
                name: component_summary(name)
                for name in ("hwr_score", "geometry_prior_score", "total_score")
            },
            "best_challenger_delta_vs_fast": {
                name: challenger_delta_summary(name)
                for name in (
                    "hwr_score_delta_vs_fast", "geometry_prior_delta_vs_fast",
                    "total_score_delta_vs_fast",
                )
            },
        },
        "geometry_prior_weight": records[0]["hwr_tournament"][
            "selective_joint_hwr_geometry_prior"
        ]["joint_geometry_prior_weight"] if records else None,
        "joint_regression_free_gate_passed": transitions["fast_exact_joint_regression"] == 0,
        "promotion_eligible": False,
    }
    if records and "selective_joint_hwr_geometry_prior_group_count_guard" in records[0]["hwr_tournament"]:
        paired["fast_vs_group_count_guard"] = compare(
            "fast", "selective_joint_hwr_geometry_prior_group_count_guard",
        )
        paired["geometry_prior_vs_group_count_guard"] = compare(
            "selective_joint_hwr_geometry_prior",
            "selective_joint_hwr_geometry_prior_group_count_guard",
        )
    group_mean_arm = "selective_joint_hwr_geometry_prior_group_mean"
    if records and group_mean_arm in records[0]["hwr_tournament"]:
        paired["fast_vs_group_mean_geometry_prior"] = compare("fast", group_mean_arm)
        paired["stroke_vs_group_mean_geometry_prior"] = compare(
            "selective_joint_hwr_geometry_prior", group_mean_arm,
        )
    guarded_group_mean_arm = "selective_joint_hwr_geometry_prior_group_mean_group_count_guard"
    if records and guarded_group_mean_arm in records[0]["hwr_tournament"]:
        paired["fast_vs_guarded_group_mean_geometry_prior"] = compare(
            "fast", guarded_group_mean_arm,
        )
        paired["stroke_guard_vs_group_mean_guard"] = compare(
            "selective_joint_hwr_geometry_prior_group_count_guard",
            guarded_group_mean_arm,
        )
    return paired


def _group_count_guard_shadow_summary(records: list[dict], *, semantic_shadow_enabled: bool) -> dict:
    arm_name = "selective_joint_hwr_geometry_prior_group_count_guard"
    if not records or arm_name not in records[0]["hwr_tournament"]:
        return {"enabled": False, "product_default_enabled": False}
    formula_transitions = Counter()
    recovered = []
    regressed = []
    if semantic_shadow_enabled:
        for record in records:
            fast_metric = record["hwr_tournament"]["fast"]["semantic_guard_shadow_if_groups_exact"]
            guarded_metric = record["hwr_tournament"][arm_name]["semantic_guard_shadow_if_groups_exact"]
            fast_exact = bool(
                fast_metric and fast_metric["formula_exact_by_stage"][FINAL_SEMANTIC_STAGE]
            )
            guarded_exact = bool(
                guarded_metric and guarded_metric["formula_exact_by_stage"][FINAL_SEMANTIC_STAGE]
            )
            if fast_exact and guarded_exact:
                formula_transitions["both_exact"] += 1
            elif fast_exact:
                formula_transitions["fast_only_exact"] += 1
                regressed.append(str(record["sample_id"]))
            elif guarded_exact:
                formula_transitions["guard_only_recovery"] += 1
                recovered.append(str(record["sample_id"]))
            else:
                formula_transitions["both_not_exact"] += 1
    rows = [record["hwr_tournament"][arm_name] for record in records]
    return {
        "enabled": True,
        "product_default_enabled": False,
        "arm": arm_name,
        "group_exact": sum(bool(row["group_exact"]) for row in rows),
        "local_2d_routes": sum(row["route"] == "local_2d" for row in rows),
        "group_count_increase_rejections": sum(
            int(row["local_group_count_increase_rejections"]) for row in rows
        ),
        "semantic_shadow_enabled": semantic_shadow_enabled,
        "semantic_formula_exact_transitions_vs_fast": (
            dict(sorted(formula_transitions.items())) if semantic_shadow_enabled else None
        ),
        "semantic_formula_exact_recovered_ids": recovered if semantic_shadow_enabled else None,
        "semantic_formula_exact_regressed_ids": regressed if semantic_shadow_enabled else None,
        "promotion_eligible": False,
    }


def _semantic_guard_shadow_summary(records: list[dict]) -> dict:
    output = {}
    arms = [
        "fast", "selective_geometry", "selective_joint_hwr",
        "selective_joint_hwr_geometry_prior",
    ]
    if records and "selective_joint_hwr_geometry_prior_group_count_guard" in records[0]["hwr_tournament"]:
        arms.append("selective_joint_hwr_geometry_prior_group_count_guard")
    for arm in (
        "selective_joint_hwr_geometry_prior_group_mean",
        "selective_joint_hwr_geometry_prior_group_mean_group_count_guard",
    ):
        if records and arm in records[0]["hwr_tournament"]:
            arms.append(arm)
    for arm in arms:
        evaluated = [
            (str(row["sample_id"]), row["hwr_tournament"][arm]["semantic_guard_shadow_if_groups_exact"])
            for row in records
            if row["hwr_tournament"][arm]["semantic_guard_shadow_if_groups_exact"] is not None
        ]
        stage_exact = Counter()
        stage_hits = Counter()
        bar_status = Counter()
        bar_changed = []
        boundary_bar_status = Counter()
        boundary_bar_changed = []
        exact_equation_status = Counter()
        exact_equation_changed = []
        candidate_equation_status = Counter()
        candidate_equation_changed = []
        terminal_rhs_bar_status = Counter()
        terminal_rhs_bar_changed = []
        transitions = {f"{before}_to_{after}": Counter() for before, after in zip(
            SEMANTIC_GUARD_STAGES, SEMANTIC_GUARD_STAGES[1:], strict=False,
        )}
        token_count = 0
        for sample_id, metric in evaluated:
            token_count += int(metric["token_count"])
            stage_exact.update({
                stage: int(bool(metric["formula_exact_by_stage"][stage]))
                for stage in SEMANTIC_GUARD_STAGES
            })
            stage_hits.update({
                stage: int(metric["token_hits_by_stage"][stage])
                for stage in SEMANTIC_GUARD_STAGES
            })
            for key, outcome in metric["formula_exact_transitions"].items():
                transitions[key][str(outcome)] += 1
            bar_audit = dict(metric.get("unique_unpaired_bar_equation") or {})
            bar_status[str(bar_audit.get("status", "missing"))] += 1
            if bar_audit.get("status") == "changed_shadow_only":
                bar_changed.append({
                    "sample_id": sample_id,
                    "changes": list(bar_audit.get("changes") or []),
                })
            boundary_audit = dict(
                (metric.get("semantic_audits") or {}).get("boundary_bar_as_unit") or {}
            )
            boundary_bar_status[str(boundary_audit.get("status", "missing"))] += 1
            if boundary_audit.get("status") == "changed_shadow_only":
                boundary_bar_changed.append({
                    "sample_id": sample_id,
                    "changes": list(boundary_audit.get("changes") or []),
                })
            exact_equation_audit = dict(
                (metric.get("semantic_audits") or {}).get(
                    "unique_exact_equation_candidates"
                ) or {}
            )
            exact_equation_status[str(exact_equation_audit.get("status", "missing"))] += 1
            if exact_equation_audit.get("status") == "changed_shadow_only":
                exact_equation_changed.append({
                    "sample_id": sample_id,
                    "changes": list(exact_equation_audit.get("changes") or []),
                })
            candidate_equation_audit = dict(
                (metric.get("semantic_audits") or {}).get(
                    "unique_candidate_arithmetic_equation"
                ) or {}
            )
            candidate_equation_status[
                str(candidate_equation_audit.get("status", "missing"))
            ] += 1
            if candidate_equation_audit.get("status") == "changed_shadow_only":
                candidate_equation_changed.append({
                    "sample_id": sample_id,
                    "changes": list(candidate_equation_audit.get("changes") or []),
                })
            terminal_audit = dict(
                (metric.get("semantic_audits") or {}).get("terminal_rhs_bar") or {}
            )
            terminal_rhs_bar_status[str(terminal_audit.get("status", "missing"))] += 1
            if terminal_audit.get("status") == "changed_shadow_only":
                terminal_rhs_bar_changed.append({
                    "sample_id": sample_id,
                    "changes": list(terminal_audit.get("changes") or []),
                })
        if any(sum(counts.values()) != len(evaluated) for counts in transitions.values()):
            raise AssertionError(f"semantic guard formula transitions do not reconcile for {arm}")
        output[arm] = {
            "scope": "only formulas with exact selected groups; shadow-only; no promotion",
            "evaluated_group_exact_formulas": len(evaluated),
            "evaluated_tokens": token_count,
            "formula_exact_by_stage": dict(stage_exact),
            "token_hits_by_stage": dict(stage_hits),
            "formula_exact_transitions": {
                key: dict(counts) for key, counts in transitions.items()
            },
            "unique_bar_guard_status": dict(bar_status),
            "unique_bar_guard_changed_formulas": bar_changed,
            "unique_bar_guard_exact_transition": dict(
                transitions["after_equation_guard_to_after_unique_bar_equation_guard"]
            ),
            "boundary_bar_as_unit_status": dict(boundary_bar_status),
            "boundary_bar_as_unit_changed_formulas": boundary_bar_changed,
            "boundary_bar_as_unit_exact_transition": dict(
                transitions[
                    "after_unique_exact_equation_candidates_guard_to_after_boundary_bar_as_unit_guard"
                ]
            ),
            "unique_exact_equation_candidates_status": dict(exact_equation_status),
            "unique_exact_equation_candidates_changed_formulas": exact_equation_changed,
            "unique_exact_equation_candidates_exact_transition": dict(
                transitions[
                    "after_unique_bar_equation_guard_to_after_unique_exact_equation_candidates_guard"
                ]
            ),
            "unique_candidate_arithmetic_equation_status": dict(candidate_equation_status),
            "unique_candidate_arithmetic_equation_changed_formulas": candidate_equation_changed,
            "unique_candidate_arithmetic_equation_exact_transition": dict(
                transitions[
                    "after_boundary_bar_as_unit_guard_to_after_unique_candidate_arithmetic_equation_guard"
                ]
            ),
            "terminal_rhs_bar_status": dict(terminal_rhs_bar_status),
            "terminal_rhs_bar_changed_formulas": terminal_rhs_bar_changed,
            "terminal_rhs_bar_exact_transition": dict(
                transitions[
                    "after_unique_candidate_arithmetic_equation_guard_to_after_terminal_rhs_bar_guard"
                ]
            ),
        }
    return {
        "schema": "aiflow-selective-semantic-guard-shadow-summary/v5",
        "scope": (
            "A/B/C/D conditional on exact groups; deterministic candidate-preserving guard shadow; "
            "single-bar and unique-equation candidate rules are exploratory and shadow-only"
        ),
        "product_default_enabled": False,
        "crohme_training_or_tuning": False,
        "arms": output,
    }


def _self_test_semantic_guard_metrics() -> None:
    groups = [[0], [1], [2]]
    symbols = [
        {"stroke_indices": [0], "hwr_topk": ["5", "6", "7", "8", "9"]},
        {"stroke_indices": [1], "hwr_topk": ["x", r"\times", "X", r"\chi", "m"]},
        {"stroke_indices": [2], "hwr_topk": ["0", "1", "2", "3", "4"]},
    ]
    stages = {
        "decoder": [
            {"stroke_indices": [0], "token": "5"},
            {"stroke_indices": [1], "token": "x"},
            {"stroke_indices": [2], "token": "0"},
        ],
        "after_fence_guard": [
            {"stroke_indices": [0], "token": "5"},
            {"stroke_indices": [1], "token": "x"},
            {"stroke_indices": [2], "token": "0"},
        ],
        "after_infix_guard": [
            {"stroke_indices": [0], "token": "5"},
            {"stroke_indices": [1], "token": r"\times"},
            {"stroke_indices": [2], "token": "0"},
        ],
        "after_equation_guard": [
            {"stroke_indices": [0], "token": "5"},
            {"stroke_indices": [1], "token": r"\times"},
            {"stroke_indices": [2], "token": "0"},
        ],
        "after_unique_bar_equation_guard": [
            {"stroke_indices": [0], "token": "5"},
            {"stroke_indices": [1], "token": r"\times"},
            {"stroke_indices": [2], "token": "0"},
        ],
        "after_unique_exact_equation_candidates_guard": [
            {"stroke_indices": [0], "token": "5"},
            {"stroke_indices": [1], "token": r"\times"},
            {"stroke_indices": [2], "token": "0"},
        ],
        "after_boundary_bar_as_unit_guard": [
            {"stroke_indices": [0], "token": "5"},
            {"stroke_indices": [1], "token": r"\times"},
            {"stroke_indices": [2], "token": "0"},
        ],
        "after_unique_candidate_arithmetic_equation_guard": [
            {"stroke_indices": [0], "token": "5"},
            {"stroke_indices": [1], "token": r"\times"},
            {"stroke_indices": [2], "token": "0"},
        ],
        "after_terminal_rhs_bar_guard": [
            {"stroke_indices": [0], "token": "5"},
            {"stroke_indices": [1], "token": r"\times"},
            {"stroke_indices": [2], "token": "0"},
        ],
    }
    preview = {
        "status": "development_only_posthoc_shadow",
        "route": "fast",
        "groups": groups,
        "symbols": symbols,
        "decoder": {"accepted": True, "top5_preserved": True, "tokens": ["5", "x", "0"]},
        "audit": {
            "all_strokes_exactly_once": True,
            "crohme_training_or_tuning": False,
            "product_default_enabled": False,
            "selected_candidate_count": 3,
            "joint_hwr_candidates_encoded": 0,
            "joint_hwr_candidate_budget": 0,
            "joint_hwr_scoring_enabled": False,
            "semantic_guard_shadow_enabled": True,
        },
        "semantic_guard_shadow": {
            "status": "applied_shadow_only",
            "product_default_enabled": False,
            "candidate_preservation": True,
            "base_decoder_tokens": ["5", "x", "0"],
            "new_tokens": 0,
            "deleted_glyphs": 0,
            "grouping_mutations": 0,
            "stages": stages,
            "audits": {
                "unique_unpaired_bar_equation": {"status": "skipped", "changes": []},
                "unique_exact_equation_candidates": {"status": "skipped", "changes": []},
                "boundary_bar_as_unit": {"status": "skipped", "changes": []},
                "unique_candidate_arithmetic_equation": {"status": "skipped", "changes": []},
                "terminal_rhs_bar": {"status": "skipped", "changes": []},
            },
        },
    }
    metrics = _preview_metrics(preview, groups, ["5", r"\times", "0"])
    shadow = metrics["semantic_guard_shadow_if_groups_exact"]
    assert shadow["formula_exact_by_stage"]["decoder"] is False
    assert shadow["formula_exact_by_stage"]["after_infix_guard"] is True
    assert shadow["formula_exact_by_stage"]["after_unique_bar_equation_guard"] is True
    assert shadow["formula_exact_by_stage"][
        "after_unique_exact_equation_candidates_guard"
    ] is True
    assert shadow["formula_exact_by_stage"]["after_boundary_bar_as_unit_guard"] is True
    assert shadow["formula_exact_by_stage"][
        "after_unique_candidate_arithmetic_equation_guard"
    ] is True
    assert shadow["formula_exact_by_stage"]["after_terminal_rhs_bar_guard"] is True
    assert shadow["formula_exact_transitions"]["after_fence_guard_to_after_infix_guard"] == "recovered"
    arms = (
        "fast", "selective_geometry", "selective_joint_hwr",
        "selective_joint_hwr_geometry_prior",
    )
    summary = _semantic_guard_shadow_summary([{
        "sample_id": "semantic-guard-self-test",
        "hwr_tournament": {
            arm: {"semantic_guard_shadow_if_groups_exact": shadow}
            for arm in arms
        },
    }])
    for arm in arms:
        assert summary["arms"][arm]["formula_exact_by_stage"][
            "after_boundary_bar_as_unit_guard"
        ] == 1
        assert summary["arms"][arm]["boundary_bar_as_unit_status"] == {"skipped": 1}
        assert summary["arms"][arm]["unique_exact_equation_candidates_status"] == {"skipped": 1}
        assert summary["arms"][arm]["unique_candidate_arithmetic_equation_status"] == {"skipped": 1}
        assert summary["arms"][arm]["terminal_rhs_bar_status"] == {"skipped": 1}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--partition-ranker", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--oracle-microscope-summary", type=Path,
        help="reuse a frozen layer-microscope summary after validating its dataset/model hashes",
    )
    parser.add_argument(
        "--exact-partitions", action="store_true",
        help="forensic exact k-best search (<=12 strokes); never promotion evidence",
    )
    parser.add_argument(
        "--hwr-tournament", action="store_true",
        help="run Fast / geometry-local / joint-HWR / prior-fusion A-B-C-D shadows; no training or promotion",
    )
    parser.add_argument(
        "--max-joint-hwr-candidates", type=int, default=64,
        help="per-formula local HWR encoding cap for the joint challenger (1-64)",
    )
    parser.add_argument(
        "--joint-geometry-prior-weight", type=float, default=1.0,
        help="shadow-only weight for the normalized frozen-grouping prior arm; not tuned on this set",
    )
    parser.add_argument(
        "--semantic-guard-shadow", action="store_true",
        help=("report candidate-preserving fence/infix/equation and exploratory Top-5 equation "
              "guard deltas without changing A/B/C/D outputs"),
    )
    parser.add_argument(
        "--group-count-increase-guard-shadow", action="store_true",
        help="add an opt-in shadow arm rejecting Local partitions with more groups than Fast",
    )
    parser.add_argument(
        "--per-group-geometry-prior-shadow", action="store_true",
        help=("add a shadow-only joint-HWR arm normalizing the frozen grouping prior by selected groups "
              "instead of raw stroke count"),
    )
    args = parser.parse_args()
    if args.semantic_guard_shadow and not args.hwr_tournament:
        parser.error("--semantic-guard-shadow requires --hwr-tournament")
    if args.group_count_increase_guard_shadow and not args.hwr_tournament:
        parser.error("--group-count-increase-guard-shadow requires --hwr-tournament")
    if args.per_group_geometry_prior_shadow and not args.hwr_tournament:
        parser.error("--per-group-geometry-prior-shadow requires --hwr-tournament")

    # Keep ``--help`` usable in constrained Python environments without
    # importing NumPy/joblib/model code before argparse exits.
    import joblib
    import numpy as np
    from selective_2d_anytime_v1 import (
        Selective2DAnytimeSolverV1, Selective2DConfigV1,
    )
    from stroke_grouping_v1 import candidate_features
    if args.hwr_tournament:
        import torch
        from evaluate_48hz_prefix_v1 import _load_model
        from raw_formula_context_runtime_v1 import RawFormulaContextRuntimeV1

    formulas_path = args.dataset_root / "data" / "formulas_valid.jsonl"
    ownership_path = args.dataset_root / "data" / "ownership_train.jsonl"
    formulas = {str(row["sample_id"]): row for row in _rows(formulas_path)}
    annotations = [row for row in _rows(ownership_path) if row.get("accepted")]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ceiling_path = args.output_dir / "oracle_group_decoder_ceiling.json"
    oracle_source = {"kind": "fresh_subprocess"}
    if args.oracle_microscope_summary is not None:
        microscope_path = args.oracle_microscope_summary.resolve()
        microscope = json.loads(microscope_path.read_text(encoding="utf-8"))
        fingerprint = microscope.get("reproducibility", {}).get("inputs", {})
        source_files = fingerprint.get("dataset_files", {})
        source_artifacts = fingerprint.get("artifacts", {})
        expected_files = {
            "data/formulas_valid.jsonl": _sha256(formulas_path),
            "data/ownership_train.jsonl": _sha256(ownership_path),
        }
        if source_files != expected_files:
            raise ValueError("oracle microscope summary dataset hashes do not match this run")
        if source_artifacts.get("hwr_checkpoint_sha256") != _sha256(args.checkpoint):
            raise ValueError("oracle microscope summary HWR checkpoint hash does not match this run")
        if source_artifacts.get("partition_ranker_sha256") != _sha256(args.partition_ranker):
            raise ValueError("oracle microscope summary grouping ranker hash does not match this run")
        if int(microscope.get("dataset", {}).get("formula_count", -1)) != len(annotations):
            raise ValueError("oracle microscope summary accepted-formula count does not match this run")
        oracle = microscope.get("oracle_group_hwr", {})
        ceiling = {
            "top1_token_exact": int(oracle["formula_top1_exact"]),
            "top5_oracle": int(oracle["formula_top5_complete"]),
            "decoder_token_exact": int(oracle["strict_decoder_formula_exact"]),
        }
        ceiling_path.write_text(json.dumps({
            "schema": "aiflow-oracle-group-decoder-ceiling-reused/v1",
            "scope": "reused from hash-verified frozen layer microscope; no training or CROHME",
            **ceiling,
            "source_summary": str(microscope_path),
            "source_summary_sha256": _sha256(microscope_path),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        oracle_source = {
            "kind": "hash_verified_layer_microscope",
            "path": str(microscope_path),
            "sha256": _sha256(microscope_path),
        }
    else:
        subprocess.run([
            sys.executable, str(Path(__file__).with_name("evaluate_oracle_group_decoder_ceiling_v1.py")),
            "--dataset-root", str(args.dataset_root), "--checkpoint", str(args.checkpoint), "--output", str(ceiling_path),
        ], check=True)
        ceiling = json.loads(ceiling_path.read_text(encoding="utf-8"))
    ranker_payload = joblib.load(args.partition_ranker)
    grouping_model = ranker_payload["grouping_model"]

    def batch_score(rows: list[dict], strokes: list[dict]) -> list[float]:
        probability = grouping_model.predict_proba(candidate_features(rows, strokes))[:, 1]
        probability = np.clip(probability, 1e-6, 1.0 - 1e-6)
        return np.log(probability / (1.0 - probability)).tolist()

    solver_config = Selective2DConfigV1(exact_partition_search=args.exact_partitions)
    solver = Selective2DAnytimeSolverV1(
        config=solver_config,
        group_score_batch=batch_score,
    )
    joint_runtime = None
    hwr_model_report = None
    if args.hwr_tournament:
        if not 1 <= args.max_joint_hwr_candidates <= 64:
            raise ValueError("--max-joint-hwr-candidates must be in [1, 64]")
        torch.set_num_threads(1)
        device = torch.device("cpu")
        hwr_model, labels, hwr_model_report = _load_model(args.checkpoint, device)
        if len(labels) != 372:
            raise AssertionError(f"expected 372 HWR labels, received {len(labels)}")
        joint_runtime = object.__new__(RawFormulaContextRuntimeV1)
        for name, value in {
            "ranker_payload": ranker_payload, "hwr": hwr_model,
            "labels": labels, "device": device,
        }.items():
            object.__setattr__(joint_runtime, name, value)
    records = []
    for annotation in annotations:
        source = formulas[str(annotation["sample_id"])]
        strokes = sorted(source["strokes"], key=lambda row: int(row["order"]))
        fast = solver.solve(strokes, allow_local=False)
        # This pass deliberately has no HWR margins: it isolates the new
        # confidence-independent structural-layout scout from future HWR tuning.
        structural = solver.solve(strokes, symbol_margins={})
        truth = _partition(annotation["groups"])
        record = {
            "sample_id": str(annotation["sample_id"]),
            "structural_target": bool(source.get("target_relations")),
            "fast_group_exact": _partition(fast["groups"]) == truth,
            "structural_route": structural["route"],
            "structural_group_exact": _partition(structural["groups"]) == truth,
            "risk_reasons": structural["audit"]["risk_reasons"],
            "local_region_strokes": structural["audit"]["local_region_strokes"],
        }
        if joint_runtime is not None:
            request = {"formula_id": str(annotation["sample_id"]), "strokes": strokes}
            target_groups = [[int(index) for index in group] for group in annotation["groups"]]
            target_tokens = [str(token) for token in annotation["labels"]]
            arms = {}
            arm_specs = [
                ("fast", False, False, 0.0, solver_config, "stroke_count"),
                ("selective_geometry", True, False, 0.0, solver_config, "stroke_count"),
                ("selective_joint_hwr", True, True, 0.0, solver_config, "stroke_count"),
                ("selective_joint_hwr_geometry_prior", True, True,
                 args.joint_geometry_prior_weight, solver_config, "stroke_count"),
            ]
            if args.group_count_increase_guard_shadow:
                arm_specs.append((
                    "selective_joint_hwr_geometry_prior_group_count_guard",
                    True, True, args.joint_geometry_prior_weight,
                    replace(solver_config, reject_local_group_count_increase=True), "stroke_count",
                ))
            if args.per_group_geometry_prior_shadow:
                arm_specs.append((
                    "selective_joint_hwr_geometry_prior_group_mean",
                    True, True, args.joint_geometry_prior_weight, solver_config, "group_count",
                ))
                if args.group_count_increase_guard_shadow:
                    arm_specs.append((
                        "selective_joint_hwr_geometry_prior_group_mean_group_count_guard",
                        True, True, args.joint_geometry_prior_weight,
                        replace(solver_config, reject_local_group_count_increase=True), "group_count",
                    ))
            for arm, allow_local, joint_hwr, geometry_prior_weight, arm_config, prior_normalization in arm_specs:
                started = perf_counter()
                preview = joint_runtime.selective_grouping_preview(
                    request,
                    config=arm_config,
                    include_hwr=True,
                    joint_hwr=joint_hwr,
                    max_joint_hwr_candidates=args.max_joint_hwr_candidates,
                    joint_geometry_prior_weight=geometry_prior_weight,
                    joint_geometry_prior_normalization=prior_normalization,
                    allow_local=allow_local,
                    include_semantic_guard_shadow=args.semantic_guard_shadow,
                )
                metrics = _preview_metrics(preview, target_groups, target_tokens)
                metrics["local_group_count_increase_guard_enabled"] = bool(
                    arm_config.reject_local_group_count_increase
                )
                if (
                    arm_config.reject_local_group_count_increase
                    and len(metrics["groups"]) > len(metrics["fast_incumbent_groups"] or [])
                ):
                    raise AssertionError("guarded shadow arm exceeded its Fast group count")
                metrics["elapsed_ms"] = (perf_counter() - started) * 1000.0
                if arm == "fast":
                    metrics["matches_group_only_fast"] = (
                        _partition(preview["groups"]) == _partition(fast["groups"])
                    )
                arms[arm] = metrics
            record["hwr_tournament"] = arms
        records.append(record)
    structural_rows = [row for row in records if row["structural_target"]]
    routed = [row for row in structural_rows if "structural_layout" in row["risk_reasons"]]
    summary = {
        "schema": "aiflow-selective-2d-research-loop/v1",
        "scope": "local shadow diagnostic only; no fitting, CROHME, external collection, or promotion",
        "partition_search_mode": "exact_k_best_capped_opt_in" if args.exact_partitions else "beam_approx_default",
        "crohme_training_or_tuning": False,
        "product_default_enabled": False,
        "promotion_eligible": False,
        "formulas": len(records),
        "fast_group_exact": sum(row["fast_group_exact"] for row in records),
        "structural_scout_group_exact": sum(row["structural_group_exact"] for row in records),
        "structural_targets": len(structural_rows),
        "structural_scout_recall": len(routed) / len(structural_rows) if structural_rows else None,
        "local_2d_wins_without_hwr_margin": sum(row["structural_route"] == "local_2d" for row in records),
        "oracle_group_top1_token_exact": ceiling["top1_token_exact"],
        "oracle_group_top5_oracle": ceiling["top5_oracle"],
        "oracle_group_decoder_token_exact": ceiling["decoder_token_exact"],
        "inputs": {
            "dataset_root": str(args.dataset_root.resolve()),
            "formulas_valid_sha256": _sha256(formulas_path),
            "ownership_train_sha256": _sha256(ownership_path),
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": _sha256(args.checkpoint),
            "partition_ranker": str(args.partition_ranker.resolve()),
            "partition_ranker_sha256": _sha256(args.partition_ranker),
            "oracle_group_decoder_ceiling_sha256": _sha256(ceiling_path),
            "oracle_group_decoder_ceiling_source": oracle_source,
        },
        "run_config": {
            "formula_rows": len(formulas),
            "accepted_ownership_rows": len(annotations),
            "selective_2d_config": solver_config.__dict__,
            "partition_ranker_lattice_config": ranker_payload.get("lattice_config"),
            "partition_ranker_top_n": ranker_payload.get("top_n"),
            "hwr_tournament": bool(args.hwr_tournament),
            "max_joint_hwr_candidates": args.max_joint_hwr_candidates if args.hwr_tournament else 0,
            "joint_geometry_prior_weight": args.joint_geometry_prior_weight if args.hwr_tournament else 0.0,
            "semantic_guard_shadow": bool(args.semantic_guard_shadow),
            "group_count_increase_guard_shadow": bool(args.group_count_increase_guard_shadow),
            "per_group_geometry_prior_shadow": bool(args.per_group_geometry_prior_shadow),
            "hwr_model_input_contract": (hwr_model_report or {}).get("input_contract"),
        },
        "records": records,
    }
    if joint_runtime is not None:
        tournament_summary = {}
        tournament_arms = [
            "fast", "selective_geometry", "selective_joint_hwr",
            "selective_joint_hwr_geometry_prior",
        ]
        if args.group_count_increase_guard_shadow:
            tournament_arms.append("selective_joint_hwr_geometry_prior_group_count_guard")
        if args.per_group_geometry_prior_shadow:
            tournament_arms.append("selective_joint_hwr_geometry_prior_group_mean")
            if args.group_count_increase_guard_shadow:
                tournament_arms.append(
                    "selective_joint_hwr_geometry_prior_group_mean_group_count_guard"
                )
        for arm in tournament_arms:
            rows = [row["hwr_tournament"][arm] for row in records]
            token_rows = [row for row in rows if row["token_exact_if_groups_exact"] is not None]
            hwr_rows = [row for row in rows if row["hwr_target_token_count_if_groups_exact"] is not None]
            target_rank_histogram = {}
            for row in hwr_rows:
                for rank, count in row["hwr_target_rank_histogram_if_groups_exact"].items():
                    target_rank_histogram[rank] = target_rank_histogram.get(rank, 0) + count
            tournament_summary[arm] = {
                "formulas": len(rows),
                "joint_geometry_prior_normalization": rows[0][
                    "joint_geometry_prior_normalization"
                ] if rows else None,
                "group_exact": sum(row["group_exact"] for row in rows),
                "group_error_shapes": dict(sorted(Counter(
                    row["group_error_shape"] for row in rows
                ).items())),
                "group_count_delta_histogram": dict(sorted(Counter(
                    str(row["group_count_delta"]) for row in rows
                ).items())),
                "decoder_accepted": sum(row["decoder_accepted"] for row in rows),
                "token_exact_evaluable_on_exact_groups": len(token_rows),
                "token_exact_on_exact_groups": sum(row["token_exact_if_groups_exact"] for row in token_rows),
                "hwr_tokens_evaluable_on_exact_groups": sum(
                    row["hwr_target_token_count_if_groups_exact"] for row in hwr_rows
                ),
                "hwr_top1_token_hits_on_exact_groups": sum(
                    row["hwr_top1_hits_if_groups_exact"] for row in hwr_rows
                ),
                "hwr_top5_token_hits_on_exact_groups": sum(
                    row["hwr_top5_hits_if_groups_exact"] for row in hwr_rows
                ),
                "hwr_top5_complete_formulas_on_exact_groups": sum(
                    row["hwr_top5_complete_if_groups_exact"] for row in hwr_rows
                ),
                "hwr_target_rank_histogram_on_exact_groups": dict(sorted(target_rank_histogram.items())),
                "hwr_candidates_encoded": sum(row["hwr_candidates_encoded"] for row in rows),
                "joint_hwr_candidates_encoded": sum(row["joint_hwr_candidates_encoded"] for row in rows),
                "joint_hwr_budget_exhausted_formulas": sum(row["joint_hwr_budget_exhausted"] for row in rows),
                "local_2d_routes": sum(row["route"] == "local_2d" for row in rows),
                "local_group_count_increase_guard_enabled_formulas": sum(
                    row["local_group_count_increase_guard_enabled"] for row in rows
                ),
                "local_group_count_increase_rejections": sum(
                    row["local_group_count_increase_rejections"] for row in rows
                ),
                "mean_host_elapsed_ms": sum(row["elapsed_ms"] for row in rows) / max(len(rows), 1),
            }
        summary["hwr_tournament"] = {
            "scope": "shadow A/B/C/D plus optional guarded arm; existing HWR Top-5; no training, threshold selection, or promotion",
            "max_joint_hwr_candidates_per_formula": args.max_joint_hwr_candidates,
            "joint_geometry_prior_weight": args.joint_geometry_prior_weight,
            "joint_geometry_prior_normalization_default": "stroke_count",
            "per_group_geometry_prior_shadow": bool(args.per_group_geometry_prior_shadow),
            "arms": tournament_summary,
            "paired_grouping": _paired_grouping_summary(records),
        }
    if args.semantic_guard_shadow:
        summary["semantic_guard_shadow"] = _semantic_guard_shadow_summary(records)
    if args.group_count_increase_guard_shadow:
        summary["group_count_increase_guard_shadow"] = _group_count_guard_shadow_summary(
            records, semantic_shadow_enabled=bool(args.semantic_guard_shadow),
        )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
