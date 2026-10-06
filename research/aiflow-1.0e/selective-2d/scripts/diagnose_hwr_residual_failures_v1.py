#!/usr/bin/env python3
"""Cause-by-cause microscope of the frozen 149-formula HWR residuals."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from audit_semantic_equation_guard_on_microscope_v1 import _rows
from selective_decoder_v1 import DEFAULT_TOKEN_BEAM
from semantic_equation_guard_v1 import apply_semantic_equation_guard


SCHEMA = "aiflow-hwr-residual-failure-microscope/v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _target_sequence_beam_reachability(
    glyphs: list[dict[str, Any]], target_tokens: list[str], beam_width: int,
) -> dict[str, Any]:
    """Audit whether the exact target prefix survives the decoder's score-only beam."""
    if len(glyphs) != len(target_tokens):
        raise ValueError("target/glyph count mismatch in beam audit")
    if beam_width < 1:
        raise ValueError("beam_width must be positive")

    beam: list[tuple[float, tuple[str, ...]]] = [(0.0, ())]
    for index, (glyph, target) in enumerate(zip(glyphs, target_tokens, strict=True), start=1):
        top5 = [str(value) for value in glyph["top5"]]
        probabilities = [float(value) for value in glyph["top5_probabilities"]]
        if len(top5) != 5 or len(probabilities) != 5:
            raise ValueError("beam audit requires exactly five token candidates")
        if target not in top5:
            return {
                "candidate_complete": False,
                "survives": False,
                "first_lost_position": index,
                "loss_reason": "target_missing_from_top5",
            }

        expanded = [
            (score + math.log(max(1e-8, probability)), prefix + (token,))
            for score, prefix in beam
            for token, probability in zip(top5, probabilities, strict=True)
        ]
        expanded.sort(key=lambda row: (-row[0], row[1]))
        beam = expanded[:beam_width]
        target_prefix = tuple(target_tokens[:index])
        if not any(prefix == target_prefix for _, prefix in beam):
            return {
                "candidate_complete": True,
                "survives": False,
                "first_lost_position": index,
                "loss_reason": "pruned_by_score_only_beam",
            }

    return {
        "candidate_complete": True,
        "survives": True,
        "first_lost_position": None,
        "loss_reason": None,
    }


def diagnose(input_path: Path) -> dict[str, Any]:
    traces = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    formula_rows = []
    all_glyphs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    failure_causes: Counter[str] = Counter()
    primary_failure_causes: Counter[str] = Counter()
    group_partition_error_shapes: Counter[str] = Counter()
    fast_group_count_deltas: Counter[str] = Counter()
    rank_histogram: Counter[str] = Counter()
    residual_rank_histogram: Counter[str] = Counter()
    confusions: Counter[tuple[str, str]] = Counter()
    group_cross_tab: Counter[tuple[str, str, str]] = Counter()
    stage_exact = Counter()
    stage_glyph_hits = Counter()
    layer_rows: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    complexity: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    relation_formulas = relation_exact = 0
    group_rank_histogram: Counter[str] = Counter()
    top5_complete_formulas = 0
    beam_reachability: Counter[str] = Counter()
    beam_first_lost_position: Counter[str] = Counter()
    residual_candidate_beam_split: Counter[str] = Counter()

    for trace in traces:
        sample_id = str(trace["sample_id"])
        target = [str(value) for value in trace["source"]["target_tokens"]]
        symbols = trace["oracle_group_hwr"]["symbols"]
        rows = _rows(trace)
        if len(target) != len(symbols) or len(rows) != len(target):
            raise AssertionError(f"target/symbol count mismatch: {sample_id}")

        guard_stages = trace["semantic_guard_shadow"]["stage_tokens"]
        top1 = [str(value) for value in guard_stages["hwr_top1"]]
        after_fence = [str(value) for value in guard_stages["after_fence_guard"]]
        after_infix = [str(value) for value in guard_stages["after_infix_guard"]]
        predictions = {
            row["record_id"]: token for row, token in zip(rows, after_infix, strict=True)
        }
        after_map, equation_audit = apply_semantic_equation_guard(rows, predictions)
        after_equation = [after_map[row["record_id"]] for row in rows]
        stages = {
            "hwr_top1": top1,
            "after_fence": after_fence,
            "after_infix": after_infix,
            "after_equation": after_equation,
        }
        stage_exact.update({
            key: int(tokens == target) for key, tokens in stages.items()
        })
        for stage, tokens in stages.items():
            stage_glyph_hits[stage] += sum(a == b for a, b in zip(tokens, target, strict=True))

        group = trace["grouping"]
        group_exact = bool(group["fast_group_exact"])
        selected_group_count = len(group["selected_symbols"])
        group_count_delta = selected_group_count - len(target)
        if group_exact:
            partition_error_shape = "exact"
        elif group_count_delta > 0:
            partition_error_shape = "oversegmentation"
        elif group_count_delta < 0:
            partition_error_shape = "undersegmentation"
        else:
            partition_error_shape = "same_count_wrong_partition"
        fast_group_count_deltas[str(group_count_delta)] += 1
        if not group_exact:
            group_partition_error_shapes[partition_error_shape] += 1
        top5_complete = all(bool(symbol["prediction"]["target_in_top5"]) for symbol in symbols)
        top5_complete_formulas += int(top5_complete)
        final_exact = after_equation == target
        post_guard_errors = []
        glyph_rows = []
        for index, (symbol, truth, final_token, row) in enumerate(
            zip(symbols, target, after_equation, rows, strict=True)
        ):
            prediction = symbol["prediction"]
            top5 = [str(value) for value in prediction["top5"]]
            rank = prediction.get("target_rank")
            rank_key = str(rank) if rank is not None else "missing"
            rank_histogram[rank_key] += 1
            if prediction["top1"] != truth:
                confusions[(truth, str(prediction["top1"]))] += 1
            if rank == 1:
                error_type = "hwr_top1_correct"
            elif rank is not None and rank <= 5:
                error_type = "hwr_top1_wrong_target_rank_2_to_5"
            else:
                error_type = "hwr_top1_wrong_target_below_5_or_missing"

            preprocessing = symbol["preprocessing"]
            point_count = int(preprocessing["raw_point_count"])
            stroke_count = int(preprocessing["stroke_count"])
            single_point_count = int(preprocessing["single_point_stroke_count"])
            confidence = {
                "top1_probability": float(prediction["top1_probability"]),
                "target_probability": prediction.get("target_probability"),
                "top1_top2_margin": float(prediction["top1_top2_probability_margin"]),
                "normalized_entropy": float(prediction["normalized_entropy"]),
            }
            glyph = {
                "position": index,
                "stroke_indices": symbol["stroke_indices"],
                "target": truth,
                "target_rank": rank,
                "error_type": error_type,
                "top1": str(prediction["top1"]),
                "after_equation_guard": final_token,
                "top5": top5,
                "top5_probabilities": [float(value) for value in prediction["top5_probabilities"]],
                **confidence,
                "raw_point_count": point_count,
                "stroke_count": stroke_count,
                "single_point_stroke_count": single_point_count,
                "pool_attention": symbol.get("pool_attention"),
                "network_layer_rms": {
                    name: float(summary["rms"])
                    for name, summary in symbol.get("network_layers", {}).items()
                    if summary.get("rms") is not None
                },
            }
            if final_token != truth:
                post_guard_errors.append(glyph)
                residual_rank_histogram[rank_key] += 1
            glyph_rows.append(glyph)
            all_glyphs[error_type].append(glyph)
            complexity[error_type]["raw_point_count"].append(float(point_count))
            complexity[error_type]["stroke_count"].append(float(stroke_count))
            complexity[error_type]["single_point_stroke_count"].append(float(single_point_count))
            for layer, rms in glyph["network_layer_rms"].items():
                layer_rows[error_type][layer].append(rms)

        target_beam = _target_sequence_beam_reachability(
            glyph_rows, target, beam_width=DEFAULT_TOKEN_BEAM,
        )
        beam_reachability[
            "candidate_complete_survived" if target_beam["survives"]
            else "candidate_complete_pruned" if target_beam["candidate_complete"]
            else "candidate_incomplete"
        ] += 1
        if target_beam["first_lost_position"] is not None:
            beam_first_lost_position[str(target_beam["first_lost_position"])] += 1
        if not final_exact and group_exact and top5_complete:
            residual_candidate_beam_split[
                "target_path_pruned_by_beam" if not target_beam["survives"]
                else "target_path_survives_but_not_selected"
            ] += 1

        if group["target_partition_rank_in_top32"] is None:
            target_partition_rank = ">32/not_enumerated"
        else:
            target_partition_rank = str(group["target_partition_rank_in_top32"])
        group_rank_histogram[target_partition_rank] += 1
        group_cross_tab[(
            "group_exact" if group_exact else "group_wrong",
            "top5_complete" if top5_complete else "candidate_missing",
            "semantic_exact" if final_exact else "semantic_wrong",
        )] += 1

        target_relations = trace["source"].get("target_relations") or []
        if target_relations:
            relation_formulas += 1
            relation_exact += int(bool(trace["oracle_group_hwr"]["decoder"].get("relations_exact")))

        selected_group_sequence_exact = bool(
            group_exact and trace["layout_and_decoder"]["selected_token_sequence_exact_if_groups_correct"]
        )
        if not final_exact:
            if not top5_complete:
                failure_causes["oracle_group_candidate_ceiling"] += 1
            else:
                failure_causes["oracle_group_candidate_ranking_or_context"] += 1
            if not group_exact:
                failure_causes["fast_group_partition_wrong_overlap"] += 1
                primary_failure_causes[f"fast_{partition_error_shape}"] += 1
            elif not top5_complete:
                primary_failure_causes["target_token_missing_from_top5"] += 1
            else:
                primary_failure_causes["candidate_ranking_or_decoder_context"] += 1
            if target_relations and not bool(trace["oracle_group_hwr"]["decoder"].get("relations_exact")):
                failure_causes["target_relation_not_recovered_overlap"] += 1

        formula_rows.append({
            "sample_id": sample_id,
            "target_tokens": target,
            "top1_tokens": top1,
            "after_fence_tokens": after_fence,
            "after_infix_tokens": after_infix,
            "after_equation_guard_tokens": after_equation,
            "hwr_top1_exact": top1 == target,
            "after_equation_guard_exact": final_exact,
            "top5_complete": top5_complete,
            "fast_group_exact": group_exact,
            "selected_group_count": selected_group_count,
            "target_group_count": len(target),
            "group_count_delta": group_count_delta,
            "partition_error_shape": partition_error_shape,
            "target_partition_rank_in_top32": target_partition_rank,
            "target_group_candidate_recall": group["target_group_candidate_recall"],
            "selected_token_sequence_exact_if_groups_correct": selected_group_sequence_exact,
            "route": group["route"],
            "risk_reasons": group["risk_reasons"],
            "target_relations": target_relations,
            "oracle_relation_exact": trace["oracle_group_hwr"]["decoder"].get("relations_exact"),
            "failed_glyphs_after_all_shadow_guards": post_guard_errors,
            "target_sequence_beam_reachability": target_beam,
            "all_glyph_diagnostics": glyph_rows,
            "equation_guard_changes": equation_audit.get("changes", []),
        })

    final_exact_count = stage_exact["after_equation"]
    formulas = len(traces)
    tokens = sum(len(trace["source"]["target_tokens"]) for trace in traces)
    error_types = {}
    for error_type, rows in all_glyphs.items():
        error_types[error_type] = {
            "glyph_count": len(rows),
            "mean_top1_probability": _mean([row["top1_probability"] for row in rows]),
            "mean_target_probability": _mean([
                float(row["target_probability"])
                for row in rows if row["target_probability"] is not None
            ]),
            "mean_top1_top2_margin": _mean([row["top1_top2_margin"] for row in rows]),
            "mean_normalized_entropy": _mean([row["normalized_entropy"] for row in rows]),
            "mean_raw_point_count": _mean(complexity[error_type]["raw_point_count"]),
            "mean_stroke_count": _mean(complexity[error_type]["stroke_count"]),
            "mean_single_point_stroke_count": _mean(
                complexity[error_type]["single_point_stroke_count"]
            ),
            "layer_mean_rms": {
                layer: _mean(values) for layer, values in sorted(layer_rows[error_type].items())
            },
        }
    exact_checks = {
        "formula_count_149": formulas == 149,
        "token_count_579": tokens == 579,
        "hwr_top1_tokens_470": stage_glyph_hits["hwr_top1"] == 470,
        "hwr_top1_exact_76": stage_exact["hwr_top1"] == 76,
        "after_infix_tokens_485": stage_glyph_hits["after_infix"] == 485,
        "after_infix_exact_84": stage_exact["after_infix"] == 84,
        "after_equation_tokens_488": stage_glyph_hits["after_equation"] == 488,
        "after_equation_exact_87": final_exact_count == 87,
        "fast_group_exact_125": sum(bool(trace["grouping"]["fast_group_exact"]) for trace in traces) == 125,
        "group_error_shapes_cover_all_group_misses": (
            sum(group_partition_error_shapes.values())
            == formulas - sum(bool(trace["grouping"]["fast_group_exact"]) for trace in traces)
        ),
        "group_count_deltas_cover_all_formulas": sum(fast_group_count_deltas.values()) == formulas,
        "primary_failure_buckets_cover_residual_formulas": (
            sum(primary_failure_causes.values()) == formulas - final_exact_count
        ),
        "beam_reachability_rows_cover_all_formulas": sum(beam_reachability.values()) == formulas,
        "beam_candidate_completeness_matches_top5_gate": (
            beam_reachability["candidate_complete_survived"]
            + beam_reachability["candidate_complete_pruned"]
            == top5_complete_formulas
        ),
        "beam_residual_candidate_split_covers_bucket": (
            sum(residual_candidate_beam_split.values())
            == primary_failure_causes["candidate_ranking_or_decoder_context"]
        ),
        "all_layer_rows_finite": all(
            summary["finite_fraction"] == 1.0
            for trace in traces
            for symbol in trace["oracle_group_hwr"]["symbols"]
            for summary in symbol["network_layers"].values()
        ),
    }
    if not all(exact_checks.values()):
        raise AssertionError(f"residual microscope checks failed: {exact_checks}")

    return {
        "schema": SCHEMA,
        "scope": "frozen 149-formula diagnostic only; oracle HWR groups for glyph causes; no training, no CROHME, no thresholds selected",
        "input": {"path": str(input_path), "sha256": _sha256(input_path)},
        "summary": {
            "formulas": formulas,
            "tokens": tokens,
            "stage_formula_exact": dict(stage_exact),
            "stage_token_hits": dict(stage_glyph_hits),
            "residual_formula_count_after_shadow": formulas - final_exact_count,
            "residual_glyph_error_count_after_shadow": sum(residual_rank_histogram.values()),
            "residual_glyph_error_target_rank_histogram_after_shadow": dict(
                sorted(residual_rank_histogram.items(), key=lambda item: (item[0] == "missing", int(item[0]) if item[0].isdigit() else 999))
            ),
            "fast_group_exact": sum(bool(trace["grouping"]["fast_group_exact"]) for trace in traces),
            "fast_group_wrong": formulas - sum(bool(trace["grouping"]["fast_group_exact"]) for trace in traces),
            "fast_group_count_delta_histogram": dict(sorted(fast_group_count_deltas.items())),
            "group_partition_error_shapes": dict(sorted(group_partition_error_shapes.items())),
            "target_partition_rank_histogram": dict(sorted(group_rank_histogram.items())),
            "target_group_candidate_recall": min(float(trace["grouping"]["target_group_candidate_recall"]) for trace in traces),
            "relation_formula_count": relation_formulas,
            "oracle_relation_exact_count": relation_exact,
            "failure_causes_nonexclusive": dict(sorted(failure_causes.items())),
            "failure_causes_primary_after_shadow": dict(sorted(primary_failure_causes.items())),
            "target_sequence_beam_reachability": {
                "token_beam": DEFAULT_TOKEN_BEAM,
                "formula_counts": dict(sorted(beam_reachability.items())),
                "first_lost_position_histogram": dict(sorted(
                    beam_first_lost_position.items(), key=lambda row: int(row[0])
                )),
                "residual_candidate_ranking_split": dict(sorted(residual_candidate_beam_split.items())),
            },
            "top1_target_rank_histogram": dict(sorted(rank_histogram.items(), key=lambda item: (item[0] == "missing", int(item[0]) if item[0].isdigit() else 999))),
            "top_confusions": [
                {"target": truth, "top1": pred, "count": count}
                for (truth, pred), count in confusions.most_common(20)
            ],
            "hwr_top1_candidate_status_summary": error_types,
            "grouping_x_candidate_x_semantic_crosstab": [
                {"grouping": key[0], "candidate_coverage": key[1], "semantic_result": key[2], "formulas": count}
                for key, count in sorted(group_cross_tab.items())
            ],
        },
        "verification": {"checks": exact_checks, "all_checks_pass": all(exact_checks.values())},
        "formula_level": formula_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-traces", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = diagnose(args.input_traces)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": report["summary"], "verification": report["verification"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
