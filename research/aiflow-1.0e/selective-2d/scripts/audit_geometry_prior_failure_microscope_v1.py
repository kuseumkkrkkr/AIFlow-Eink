#!/usr/bin/env python3
"""Microscope paired failure causes for frozen geometry-prior shadow arms."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
from typing import Any

from audit_decoder_failure_beam_v1 import _beam_audit
from selective_decoder_v1 import decode_selective_partition


FINAL_STAGE = "after_boundary_bar_as_unit_guard"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _group_key(group: list[int] | tuple[int, ...]) -> tuple[int, ...]:
    return tuple(sorted(int(index) for index in group))


def _is_exact(record: dict[str, Any], arm: str) -> bool:
    metrics = record["hwr_tournament"][arm]
    if not metrics["group_exact"]:
        return False
    semantic = metrics.get("semantic_guard_shadow_if_groups_exact") or {}
    return bool((semantic.get("formula_exact_by_stage") or {}).get(FINAL_STAGE, False))


def _stroke_box(stroke: dict[str, Any]) -> tuple[float, float, float, float]:
    points = stroke.get("points") or []
    if not points:
        raise ValueError("stroke has no raw points")
    xs = [float(point["x"]) for point in points]
    ys = [float(point["y"]) for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def _merge_gap_details(
    metrics: dict[str, Any], formula: dict[str, Any],
) -> tuple[float | None, list[dict[str, Any]]]:
    if metrics.get("route") != "local_2d":
        return None, []
    boxes = [_stroke_box(stroke) for stroke in formula["strokes"]]
    details = []
    for group in metrics["groups"]:
        for offset, left_index in enumerate(group):
            for right_index in group[offset + 1:]:
                left = boxes[int(left_index)]
                right = boxes[int(right_index)]
                horizontal_gap = max(0.0, max(left[0], right[0]) - min(left[2], right[2]))
                min_width = max(min(left[2] - left[0], right[2] - right[0]), 1e-9)
                details.append({
                    "stroke_pair": sorted((int(left_index), int(right_index))),
                    "horizontal_gap": horizontal_gap,
                    "min_stroke_width": min_width,
                    "normalized_horizontal_gap": horizontal_gap / min_width,
                })
    return max((row["normalized_horizontal_gap"] for row in details), default=0.0), details


def _merge_gap_router_counterfactual(
    records: list[dict[str, Any]], formula_by_id: dict[str, dict[str, Any]],
    fast_arm: str, challenger_arm: str,
) -> dict[str, Any]:
    thresholds = (0.1, 0.15, 0.2, 0.25, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0)
    local_rows = []
    risk_by_id: dict[str, float] = {}
    for record in records:
        sample_id = str(record["sample_id"])
        challenger = record["hwr_tournament"][challenger_arm]
        risk, pairs = _merge_gap_details(challenger, formula_by_id[sample_id])
        if risk is None:
            continue
        risk_by_id[sample_id] = risk
        local_rows.append({
            "sample_id": sample_id,
            "max_normalized_horizontal_gap": risk,
            "merged_pairs": pairs,
            "challenger_group_exact": bool(challenger["group_exact"]),
            "challenger_semantic_formula_exact": _is_exact(record, challenger_arm),
            "fast_group_exact": bool(record["hwr_tournament"][fast_arm]["group_exact"]),
            "fast_semantic_formula_exact": _is_exact(record, fast_arm),
            "challenger_group_error_shape": challenger.get("group_error_shape"),
        })

    threshold_rows = []
    for threshold in thresholds:
        veto_ids = {sample_id for sample_id, risk in risk_by_id.items() if risk > threshold}
        effective_exact = {
            str(record["sample_id"]): _is_exact(
                record, fast_arm if str(record["sample_id"]) in veto_ids else challenger_arm,
            )
            for record in records
        }
        fast_exact = {
            str(record["sample_id"]): _is_exact(record, fast_arm)
            for record in records
        }
        challenger_exact = {
            str(record["sample_id"]): _is_exact(record, challenger_arm)
            for record in records
        }
        effective_group_exact = {
            str(record["sample_id"]): bool(
                record["hwr_tournament"][fast_arm if str(record["sample_id"]) in veto_ids else challenger_arm]["group_exact"]
            )
            for record in records
        }
        fast_group_exact = {
            str(record["sample_id"]): bool(record["hwr_tournament"][fast_arm]["group_exact"])
            for record in records
        }
        threshold_rows.append({
            "threshold": threshold,
            "veto_count": len(veto_ids),
            "vetoed_ids": sorted(veto_ids),
            "group_exact_count": sum(effective_group_exact.values()),
            "group_recovered_vs_fast_ids": sorted(
                sample_id for sample_id, exact in effective_group_exact.items()
                if exact and not fast_group_exact[sample_id]
            ),
            "group_regressed_vs_fast_ids": sorted(
                sample_id for sample_id, exact in effective_group_exact.items()
                if fast_group_exact[sample_id] and not exact
            ),
            "semantic_formula_exact_count": sum(effective_exact.values()),
            "recovered_vs_fast_ids": sorted(
                sample_id for sample_id, exact in effective_exact.items()
                if exact and not fast_exact[sample_id]
            ),
            "regressed_vs_fast_ids": sorted(
                sample_id for sample_id, exact in effective_exact.items()
                if fast_exact[sample_id] and not exact
            ),
            "change_vs_unvetoed_challenger": sum(effective_exact.values())
            - sum(challenger_exact.values()),
        })
    return {
        "feature": "maximum pairwise horizontal bounding-box gap divided by the narrower stroke width, among strokes merged by the selected local-2D partition",
        "counterfactual": "if a routed local-2D partition exceeds the listed threshold, use the saved Fast output for that formula",
        "local_2d_formula_count": len(local_rows),
        "threshold_selection": "none; fixed descriptive sweep only on consumed development data",
        "local_route_details": local_rows,
        "sweep": threshold_rows,
        "interpretation_limit": "posthoc feature diagnostic; small routed sample and consumed cohort, not a validated router or promotion evidence",
    }


def _selected_group_decoder_microscope(
    records: list[dict[str, Any]], formula_by_id: dict[str, dict[str, Any]], arm: str,
) -> dict[str, Any]:
    beam_widths = (32, 128)
    rows_by_width: dict[int, list[dict[str, Any]]] = {width: [] for width in beam_widths}
    replay_match_counts = Counter()
    exact_group_records = [
        record for record in records
        if record["hwr_tournament"][arm]["group_exact"]
    ]

    for record in exact_group_records:
        sample_id = str(record["sample_id"])
        metrics = record["hwr_tournament"][arm]
        formula = formula_by_id[sample_id]
        groups = [[int(index) for index in group] for group in metrics["groups"]]
        target_by_group = {
            _group_key(group): str(token)
            for group, token in zip(metrics["target_groups"], metrics["target_tokens"], strict=True)
        }
        if set(target_by_group) != {_group_key(group) for group in groups}:
            raise AssertionError(f"exact group partition does not join to target: {sample_id}")
        targets = [target_by_group[_group_key(group)] for group in groups]
        symbols = []
        for symbol in metrics["selected_symbols"]:
            stroke_indices = [int(index) for index in symbol["stroke_indices"]]
            points = [
                point
                for index in stroke_indices
                for point in (formula["strokes"][index].get("points") or [])
            ]
            if not points:
                raise AssertionError(f"selected HWR group has no raw points: {sample_id}")
            xs = [float(point["x"] if isinstance(point, dict) else point[0]) for point in points]
            ys = [float(point["y"] if isinstance(point, dict) else point[1]) for point in points]
            symbols.append({
                "stroke_indices": stroke_indices,
                "hwr_topk": [str(token) for token in symbol["hwr_topk"]],
                "hwr_topk_probabilities": [float(value) for value in symbol["hwr_topk_probabilities"]],
                "geometry": {
                    "left": min(xs), "right": max(xs),
                    "top": min(ys), "bottom": max(ys),
                },
            })

        for width in beam_widths:
            saved_decoder = {
                "accepted": bool(metrics["decoder_accepted"]),
                "tokens": metrics["decoder_tokens"] if metrics["decoder_accepted"] else None,
                "latex": metrics["decoder_latex"],
            }
            if width == 128:
                # This wider replay is diagnostic only; the selected 32-beam result above
                # is independently checked against the saved run record.
                widened = decode_selective_partition(
                    sample_id, groups, symbols,
                    stroke_count=len(formula["strokes"]), token_beam=width,
                )
                saved_decoder = {
                    "accepted": bool(widened.get("accepted")),
                    "tokens": widened.get("tokens"),
                    "latex": widened.get("latex"),
                }
            trace = {
                "sample_id": sample_id,
                "source": {"target_tokens": targets, "stroke_count": len(formula["strokes"])},
                "oracle_group_hwr": {"decoder": saved_decoder},
            }
            audited = _beam_audit(trace, groups, symbols, width)
            matches_saved = (
                audited["selected_tokens"] == metrics["decoder_tokens"]
                and audited["selected_latex"] == metrics["decoder_latex"]
                and bool(audited["selected_beam_rank"] is not None) == bool(metrics["decoder_accepted"])
            ) if width == 32 else bool(audited["decoder_replay_matches_saved_trace"])
            replay_match_counts[width] += int(matches_saved)
            final_exact = _is_exact(record, arm)
            if final_exact:
                continue
            if not audited["candidate_complete"]:
                cause = "target_missing_from_top5"
            elif audited["target_beam_rank"] is None:
                cause = "target_pruned_by_score_only_beam"
            elif not audited["target_structurally_valid"]:
                cause = "target_reaches_beam_but_fails_strict_ast"
            elif audited["selected_tokens"] != targets:
                cause = "valid_target_survives_but_loses_score"
            else:
                cause = "decoder_selects_target_before_semantic_shadow"
            rows_by_width[width].append({
                "sample_id": sample_id,
                "cause": cause,
                "candidate_complete": audited["candidate_complete"],
                "target_beam_rank": audited["target_beam_rank"],
                "first_lost_position": audited["first_lost_position"],
                "first_lost_reason": audited["first_lost_reason"],
                "first_lost_rank_before_prune": audited["first_lost_rank_before_prune"],
                "first_lost_score_gap_vs_beam_cutoff": audited["first_lost_score_gap_vs_beam_cutoff"],
                "target_structurally_valid": audited["target_structurally_valid"],
                "target_rank_among_valid_beam_candidates": audited["target_rank_among_valid_beam_candidates"],
                "valid_beam_candidate_count": audited["valid_beam_candidate_count"],
                "selected_tokens": audited["selected_tokens"],
                "target_tokens": audited["target_tokens"],
                "target_to_selected_score_gap": audited["target_to_selected_score_gap"],
                "score_gap_decomposition": audited["score_gap_decomposition"],
                "token_level_selection": audited["token_level_selection"],
            })

    width_summaries = {}
    for width, residual_rows in rows_by_width.items():
        causes = Counter(row["cause"] for row in residual_rows)
        scored = [row for row in residual_rows if row["cause"] == "valid_target_survives_but_loses_score"]
        mismatches = [
            token for row in scored for token in row["token_level_selection"]
            if token["mismatch"]
        ]
        gaps = [float(row["target_to_selected_score_gap"]) for row in scored
                if row["target_to_selected_score_gap"] is not None]
        width_summaries[str(width)] = {
            "residual_formula_count": len(residual_rows),
            "disjoint_cause_counts": dict(sorted(causes.items())),
            "valid_surviving_target_mismatch_positions": len(mismatches),
            "mismatch_target_hwr_rank_histogram": dict(sorted(Counter(
                str(row["target_hwr_rank"]) for row in mismatches
            ).items())),
            "mismatch_selected_hwr_rank_histogram": dict(sorted(Counter(
                str(row["selected_hwr_rank"]) for row in mismatches
            ).items())),
            "mean_score_gap_components": {
                name: (
                    sum(float(row["score_gap_decomposition"][key] or 0.0) for row in scored) / len(scored)
                    if scored else None
                )
                for name, key in (
                    ("token_log_probability", "token_log_probability_component"),
                    ("relation", "relation_component"),
                    ("total", "reconstructed_total"),
                )
            },
            "valid_target_to_selected_score_gap": {
                "count": len(gaps), "mean": sum(gaps) / len(gaps) if gaps else None,
                "min": min(gaps) if gaps else None, "max": max(gaps) if gaps else None,
            },
            "replay_matches": replay_match_counts[width],
            "replay_rows": len(exact_group_records),
            "formula_details": residual_rows,
        }
    return {
        "beam32_authoritative_replay": "all exact-group formulas replayed against the saved research-run decoder output",
        "beam128_replay": "wider score/AST diagnostic replay only; not a product setting",
        "exact_group_formula_count": len(exact_group_records),
        "beam_widths": width_summaries,
        "interpretation_limit": "wider beam changes reachability but this is still a consumed-development shadow diagnostic",
    }


def _writer_bootstrap(
    records: list[dict[str, Any]], writers: dict[str, str], baseline: str, challenger: str,
    *, iterations: int = 10_000, seed: int = 20261001,
) -> dict[str, Any]:
    by_writer: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_writer.setdefault(writers[record["sample_id"]], []).append(record)
    writer_ids = sorted(by_writer)
    rng = random.Random(seed)
    results: dict[str, list[float]] = {}
    for metric in ("group_exact", "semantic_formula_exact"):
        deltas: list[float] = []
        for _ in range(iterations):
            sampled = [rng.choice(writer_ids) for _ in writer_ids]
            denominator = sum(len(by_writer[writer]) for writer in sampled)
            delta = 0
            for writer in sampled:
                for record in by_writer[writer]:
                    before = bool(record["hwr_tournament"][baseline]["group_exact"])
                    after = bool(record["hwr_tournament"][challenger]["group_exact"])
                    if metric == "semantic_formula_exact":
                        before = _is_exact(record, baseline)
                        after = _is_exact(record, challenger)
                    delta += int(after) - int(before)
            deltas.append(100.0 * delta / denominator)
        deltas.sort()
        results[metric] = [deltas[int(0.025 * (iterations - 1))], deltas[int(0.975 * (iterations - 1))]]
    return {
        "writer_count": len(writer_ids),
        "iterations": iterations,
        "seed": seed,
        "delta_rate_pp_95_interval": results,
        "interpretation": "descriptive paired bootstrap on consumed development data; not acceptance evidence",
    }


def _arm_failure_rows(
    records: list[dict[str, Any]], arm: str, trace_by_id: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    group_shapes: Counter[str] = Counter()
    cause_counts: Counter[str] = Counter()
    stage_counts: Counter[str] = Counter()
    target_rank_counts: Counter[str] = Counter()
    token_error_rank_counts: Counter[str] = Counter()
    token_error_choice_counts: Counter[str] = Counter()
    residual_token_error_rank_counts: Counter[str] = Counter()
    residual_token_error_choice_counts: Counter[str] = Counter()
    top5_misses: list[dict[str, Any]] = []
    token_residuals: list[dict[str, Any]] = []
    group_error_details: list[dict[str, Any]] = []
    exact_group_count = top5_complete_count = target_hits = target_count = 0

    for record in records:
        sample_id = record["sample_id"]
        metrics = record["hwr_tournament"][arm]
        if not metrics["group_exact"]:
            group_shapes[str(metrics.get("group_error_shape", "unknown"))] += 1
            cause_counts["group_partition_wrong"] += 1
            group_error_details.append({
                "sample_id": sample_id,
                "group_error_shape": metrics.get("group_error_shape"),
                "groups": metrics["groups"],
                "target_groups": metrics["target_groups"],
                "group_count_delta": metrics.get("group_count_delta"),
                "route": metrics.get("route"),
                "local_candidate_count": metrics.get("candidate_groups_local"),
                "partition_schedule_completed": metrics.get("partition_schedule_completed"),
                "score": metrics.get("joint_selected_score_breakdown"),
            })
            continue

        exact_group_count += 1
        if bool(metrics.get("hwr_top5_complete_if_groups_exact")):
            top5_complete_count += 1
        semantic = metrics.get("semantic_guard_shadow_if_groups_exact") or {}
        stage_exact = semantic.get("formula_exact_by_stage") or {}
        for stage, exact in stage_exact.items():
            if exact:
                stage_counts[stage] += 1

        target_by_group = {
            _group_key(group): str(token)
            for group, token in zip(metrics["target_groups"], metrics["target_tokens"], strict=True)
        }
        symbols_by_group = {
            _group_key(symbol["stroke_indices"]): symbol
            for symbol in metrics["selected_symbols"]
        }
        decoder_tokens = [str(token) for token in metrics["decoder_tokens"]]
        if len(decoder_tokens) != len(metrics["groups"]):
            raise AssertionError(f"group/token mismatch: {sample_id}:{arm}")
        missing_for_formula = []
        errors_for_formula = []
        for index, group in enumerate(metrics["groups"]):
            key = _group_key(group)
            target = target_by_group[key]
            symbol = symbols_by_group[key]
            top5 = [str(token) for token in symbol["hwr_topk"]]
            target_count += 1
            if target in top5:
                target_hits += 1
                rank = top5.index(target) + 1
                target_rank_counts[str(rank)] += 1
            else:
                rank = None
                target_rank_counts["missing_or_outside_top5"] += 1
                missing = {
                    "stroke_indices": list(key), "target": target, "top5": top5,
                    "trace": None,
                }
                trace = trace_by_id.get(sample_id)
                if trace:
                    traced_symbol = next((
                        item for item in trace["oracle_group_hwr"]["symbols"]
                        if _group_key(item["stroke_indices"]) == key
                    ), None)
                    if traced_symbol:
                        missing["trace"] = {
                            "target_in_vocab": traced_symbol["final_head_probe"]["target_in_vocab"],
                            "raw_point_count": traced_symbol["preprocessing"]["raw_point_count"],
                            "raw_points_per_stroke": traced_symbol["preprocessing"]["raw_points_per_stroke"],
                            "final_target_rank": traced_symbol["prediction"]["target_rank"],
                            "layer_trajectory": [
                                {
                                    "stage": layer["stage"],
                                    "target_rank": layer["target_rank"],
                                    "predicted_top1": layer["predicted_top1"],
                                    "target_minus_best_other_logit": layer["target_minus_best_other_logit"],
                                }
                                for layer in traced_symbol["layer_logit_lens"]
                            ],
                        }
                missing_for_formula.append(missing)
            predicted = decoder_tokens[index]
            if predicted != target:
                choice = "decoder_kept_hwr_top1" if predicted == top5[0] else "decoder_changed_hwr_top1"
                token_error_rank_counts[str(rank) if rank is not None else "missing"] += 1
                token_error_choice_counts[choice] += 1
                errors_for_formula.append({
                    "stroke_indices": list(key), "target": target, "target_rank": rank,
                    "hwr_top1": top5[0], "decoder_token": predicted, "choice_vs_hwr_top1": choice,
                })
        final_exact = bool(stage_exact.get(FINAL_STAGE, False))
        if final_exact:
            cause_counts["group_and_semantic_token_sequence_exact"] += 1
        elif missing_for_formula:
            cause_counts["group_exact_target_outside_top5"] += 1
            top5_misses.append({"sample_id": sample_id, "glyphs": missing_for_formula})
        else:
            cause_counts["group_exact_top5_present_but_sequence_wrong"] += 1
            token_residuals.append({"sample_id": sample_id, "glyphs": errors_for_formula})
            for error in errors_for_formula:
                residual_token_error_rank_counts[
                    str(error["target_rank"]) if error["target_rank"] is not None else "missing"
                ] += 1
                residual_token_error_choice_counts[str(error["choice_vs_hwr_top1"])] += 1

    cause_counts["group_partition_wrong"] = sum(group_shapes.values())
    summary = {
        "arm": arm,
        "group_exact_formulas": exact_group_count,
        "semantic_formula_exact_by_stage": dict(sorted(stage_counts.items())),
        "disjoint_formula_causes": dict(sorted(cause_counts.items())),
        "group_error_shapes": dict(sorted(group_shapes.items())),
        "top5_complete_exact_group_formulas": top5_complete_count,
        "target_symbol_hits_on_exact_groups": target_hits,
        "target_symbols_on_exact_groups": target_count,
        "target_rank_histogram": dict(sorted(target_rank_counts.items())),
        "wrong_token_target_rank_histogram": dict(sorted(token_error_rank_counts.items())),
        "wrong_token_decoder_vs_hwr_top1": dict(sorted(token_error_choice_counts.items())),
        "candidate_present_residual_wrong_token_count": sum(residual_token_error_rank_counts.values()),
        "candidate_present_residual_target_rank_histogram": dict(sorted(residual_token_error_rank_counts.items())),
        "candidate_present_residual_decoder_vs_hwr_top1": dict(sorted(residual_token_error_choice_counts.items())),
        "group_error_details": group_error_details,
    }
    details = [
        *({"cause": "target_outside_top5", **row} for row in top5_misses),
        *({"cause": "candidate_present_token_residual", **row} for row in token_residuals),
    ]
    if sum(cause_counts.values()) != len(records):
        raise AssertionError(f"disjoint cause counts do not reconcile for {arm}")
    if target_hits + target_rank_counts["missing_or_outside_top5"] != target_count:
        raise AssertionError(f"target rank counts do not reconcile for {arm}")
    return summary, details


def audit(
    summary_path: Path, baseline_arm: str, challenger_arm: str,
    traces_path: Path | None = None,
) -> dict[str, Any]:
    summary = _read_json(summary_path)
    if summary.get("schema") != "aiflow-selective-2d-research-loop/v1":
        raise ValueError("unexpected summary schema")
    if summary.get("product_default_enabled") is not False:
        raise AssertionError("product default must remain disabled")
    if summary.get("crohme_training_or_tuning") is not False:
        raise AssertionError("CROHME boundary was crossed")
    if summary.get("promotion_eligible") is not False:
        raise AssertionError("source run is not shadow-only")
    records = summary.get("records") or []
    ids = [str(row["sample_id"]) for row in records]
    if len(ids) != len(set(ids)) or not records:
        raise AssertionError("formula IDs must be nonempty and unique")
    for record in records:
        arms = record.get("hwr_tournament", {})
        if baseline_arm not in arms or challenger_arm not in arms or "fast" not in arms:
            raise ValueError(f"requested arms missing for {record['sample_id']}")

    dataset_root = Path(summary["inputs"]["dataset_root"])
    ownership_path = dataset_root / "data" / "ownership_train.jsonl"
    ownership_rows = [row for row in _read_jsonl(ownership_path) if row.get("accepted")]
    writer_by_id = {str(row["sample_id"]): str(row["writer_id"]) for row in ownership_rows}
    if set(writer_by_id) != set(ids):
        raise AssertionError("accepted ownership IDs do not match summary formulas")
    formula_path = dataset_root / "data" / "formulas_valid.jsonl"
    formulas = {str(row["sample_id"]): row for row in _read_jsonl(formula_path)}
    if not set(ids) <= set(formulas):
        raise AssertionError("summary formulas missing from frozen dataset")
    formulas_sha256 = _sha256(formula_path)
    ownership_sha256 = _sha256(ownership_path)

    trace_by_id: dict[str, dict[str, Any]] = {}
    if traces_path is not None:
        traces = _read_jsonl(traces_path)
        trace_by_id = {str(row["sample_id"]): row for row in traces}
        if len(trace_by_id) != len(traces) or not set(ids) <= set(trace_by_id):
            raise AssertionError("layer traces are duplicated or incomplete")

    base_summary, _base_details = _arm_failure_rows(records, baseline_arm, trace_by_id)
    challenger_summary, challenger_details = _arm_failure_rows(records, challenger_arm, trace_by_id)
    decoder_microscope = _selected_group_decoder_microscope(
        records, formulas, challenger_arm,
    )

    group_transition: Counter[str] = Counter()
    strict_transition: Counter[str] = Counter()
    group_recovered: list[str] = []
    group_regressed: list[str] = []
    strict_recovered: list[str] = []
    strict_regressed: list[str] = []
    paired_details = []
    for record in records:
        sample_id = str(record["sample_id"])
        before = record["hwr_tournament"][baseline_arm]
        after = record["hwr_tournament"][challenger_arm]
        bg, ag = bool(before["group_exact"]), bool(after["group_exact"])
        group_key = "both_exact" if bg and ag else "baseline_only_exact" if bg else "challenger_only_exact" if ag else "both_group_wrong"
        group_transition[group_key] += 1
        be, ae = _is_exact(record, baseline_arm), _is_exact(record, challenger_arm)
        exact_key = "both_exact" if be and ae else "baseline_only_exact" if be else "challenger_only_exact" if ae else "both_inexact"
        strict_transition[exact_key] += 1
        if ag and not bg:
            group_recovered.append(sample_id)
        if bg and not ag:
            group_regressed.append(sample_id)
        if ae and not be:
            strict_recovered.append(sample_id)
        if be and not ae:
            strict_regressed.append(sample_id)
        if ag != bg or ae != be:
            paired_details.append({
                "sample_id": sample_id,
                "baseline": {
                    "group_exact": bg, "group_error_shape": before.get("group_error_shape"),
                    "groups": before["groups"], "target_groups": before["target_groups"],
                    "decoder_tokens": before["decoder_tokens"], "target_tokens": before["target_tokens"],
                    "score": before.get("joint_selected_score_breakdown"),
                },
                "challenger": {
                    "group_exact": ag, "group_error_shape": after.get("group_error_shape"),
                    "groups": after["groups"], "target_groups": after["target_groups"],
                    "decoder_tokens": after["decoder_tokens"], "target_tokens": after["target_tokens"],
                    "score": after.get("joint_selected_score_breakdown"),
                },
            })

    cause_total = sum(challenger_summary["disjoint_formula_causes"].values())
    checks = {
        "source_shadow_only": summary.get("promotion_eligible") is False,
        "no_crohme_training_or_tuning": summary.get("crohme_training_or_tuning") is False,
        "product_default_disabled": summary.get("product_default_enabled") is False,
        "unique_formula_rows": len(ids) == len(set(ids)),
        "ownership_ids_match": set(writer_by_id) == set(ids),
        "formula_source_contains_all_ids": set(ids) <= set(formulas),
        "formula_hash_matches_summary": (
            summary.get("inputs", {}).get("formulas_valid_sha256") is None
            or summary["inputs"]["formulas_valid_sha256"] == formulas_sha256
        ),
        "ownership_hash_matches_summary": (
            summary.get("inputs", {}).get("ownership_train_sha256") is None
            or summary["inputs"]["ownership_train_sha256"] == ownership_sha256
        ),
        "layer_trace_ids_match_when_supplied": traces_path is None or set(ids) <= set(trace_by_id),
        "current_decoder_replay_matches_saved_for_all_exact_groups": (
            decoder_microscope["beam_widths"]["32"]["replay_matches"]
            == decoder_microscope["exact_group_formula_count"]
        ),
        "wider_decoder_replay_matches_decoder_function_for_all_exact_groups": (
            decoder_microscope["beam_widths"]["128"]["replay_matches"]
            == decoder_microscope["exact_group_formula_count"]
        ),
        "decoder_residual_cause_buckets_reconcile_at_both_widths": all(
            sum(decoder_microscope["beam_widths"][str(width)]["disjoint_cause_counts"].values())
            == decoder_microscope["beam_widths"][str(width)]["residual_formula_count"]
            for width in (32, 128)
        ),
        "challenger_failure_causes_reconcile": cause_total == len(records),
        "paired_group_transition_reconciles": sum(group_transition.values()) == len(records),
        "paired_exact_transition_reconciles": sum(strict_transition.values()) == len(records),
    }
    if not all(checks.values()):
        raise AssertionError(f"audit validation failed: {checks}")

    return {
        "schema": "aiflow-geometry-prior-failure-microscope/v1",
        "scope": "consumed 149-formula development cohort; paired forensic shadow only",
        "interpretation_limits": [
            "This is descriptive analysis of the consumed development cohort, not fresh acceptance or promotion evidence.",
            "Layer logit lenses locate score movement but do not prove a causal layer defect.",
            "Formula exact means exact target token sequence in the saved semantic shadow, not official CROHME Expression Rate.",
        ],
        "inputs": {
            "summary": {"path": str(summary_path.resolve()), "sha256": _sha256(summary_path)},
            "formulas_valid": {"path": str(formula_path.resolve()), "sha256": formulas_sha256},
            "ownership_train": {"path": str(ownership_path.resolve()), "sha256": ownership_sha256},
            "layer_traces": None if traces_path is None else {
                "path": str(traces_path.resolve()), "sha256": _sha256(traces_path),
            },
        },
        "formulas": len(records),
        "arms": {"baseline": baseline_arm, "challenger": challenger_arm},
        "paired_transitions": {
            "group_exact": {
                "counts": dict(sorted(group_transition.items())),
                "recovered_ids": group_recovered,
                "regressed_ids": group_regressed,
            },
            "semantic_formula_exact": {
                "counts": dict(sorted(strict_transition.items())),
                "recovered_ids": strict_recovered,
                "regressed_ids": strict_regressed,
            },
            "changed_formula_details": paired_details,
            "writer_cluster_bootstrap": _writer_bootstrap(
                records, writer_by_id, baseline_arm, challenger_arm,
            ),
        },
        "merge_gap_router_counterfactual": _merge_gap_router_counterfactual(
            records, formulas, "fast", challenger_arm,
        ),
        "selected_group_decoder_microscope": decoder_microscope,
        "failure_microscope": {
            "baseline": base_summary,
            "challenger": challenger_summary,
            "challenger_failure_details": challenger_details,
        },
        "validation": checks,
        "product_default_enabled": False,
        "crohme_training_or_tuning": False,
        "promotion_eligible": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--baseline-arm", required=True)
    parser.add_argument("--challenger-arm", required=True)
    parser.add_argument("--layer-traces", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.summary.resolve(), args.baseline_arm, args.challenger_arm,
                   args.layer_traces.resolve() if args.layer_traces else None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "formulas": result["formulas"],
        "group_transition": result["paired_transitions"]["group_exact"]["counts"],
        "semantic_formula_transition": result["paired_transitions"]["semantic_formula_exact"]["counts"],
        "challenger_failure_causes": result["failure_microscope"]["challenger"]["disjoint_formula_causes"],
        "writer_cluster_bootstrap": result["paired_transitions"]["writer_cluster_bootstrap"]["delta_rate_pp_95_interval"],
        "validation": result["validation"],
        "output": str(args.output.resolve()),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
