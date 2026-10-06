#!/usr/bin/env python3
"""Cross-check layer, decoder, and grouping failure causes without fitting."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


SCHEMA = "aiflow-hwr-failure-cause-microscope/v1"
LAYER_STAGES = (
    "encoder.input",
    "encoder.block_0.output",
    "encoder.block_1.output",
    "encoder.block_2.output",
    "encoder.block_3.output",
    "encoder.output",
)


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


def _checkpoint_labels(path: Path) -> list[str]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    labels = list(checkpoint["math_labels"])
    if len(labels) != 372 or checkpoint.get("auxiliary_labels"):
        raise ValueError("expected the frozen unified 372-class HWR checkpoint")
    return labels


def audit(
    traces_path: Path,
    decoder_path: Path,
    grouping_path: Path,
    checkpoint_path: Path,
    wide_beam_path: Path | None = None,
) -> dict[str, Any]:
    traces = _jsonl(traces_path)
    decoder = _json(decoder_path)
    grouping = _json(grouping_path)
    labels = _checkpoint_labels(checkpoint_path)
    label_set = set(labels)
    trace_by_id = {row["sample_id"]: row for row in traces}
    if len(trace_by_id) != len(traces):
        raise AssertionError("duplicate sample_id in layer traces")

    oov_tokens: list[dict[str, Any]] = []
    trace_target_symbol_count = 0
    target_top5_misses: Counter[str] = Counter()
    trace_checks: Counter[str] = Counter()
    for trace in traces:
        for name, passed in trace.get("checks", {}).items():
            if not passed:
                trace_checks[name] += 1
        for symbol in trace["oracle_group_hwr"]["symbols"]:
            trace_target_symbol_count += 1
            if tuple(layer["stage"] for layer in symbol["layer_logit_lens"]) != LAYER_STAGES:
                trace_checks["layer_logit_lens_stage_contract"] += 1
            token = symbol["target_label"]
            in_vocab = token in label_set
            probe = symbol["final_head_probe"]
            if bool(probe["target_in_vocab"]) != in_vocab:
                raise AssertionError(f"checkpoint/trace vocabulary mismatch: {trace['sample_id']} {token}")
            prediction = symbol["prediction"]
            in_top5 = token in prediction["top5"]
            if not in_top5:
                target_top5_misses["in_vocab" if in_vocab else "oov"] += 1
            if not in_vocab:
                oov_tokens.append({
                    "sample_id": trace["sample_id"],
                    "target": token,
                    "stroke_indices": symbol["stroke_indices"],
                    "top1": prediction["top1"],
                    "top5": prediction["top5"],
                    "stage_ranks": [
                        {"stage": row["stage"], "rank": row["target_rank"]}
                        for row in symbol["layer_logit_lens"]
                    ],
                })

    arm_rows = decoder["formula_level"]
    residuals = [row for row in arm_rows if row["current_arm_shadow_residual"]]
    buckets: Counter[str] = Counter()
    residual_details: list[dict[str, Any]] = []
    score_checks = 0
    score_decomposition_max_abs_error = 0.0
    survivor_relation_gaps: list[float] = []
    residual_top5_misses: Counter[str] = Counter()
    invalid_survivor_count = 0
    for row in residuals:
        if not row["candidate_complete"]:
            cause = "top5_missing"
        elif row["target_beam_rank"] is None:
            cause = "beam_pruned"
        else:
            if not row["target_structurally_valid"]:
                invalid_survivor_count += 1
                raise AssertionError(f"surviving target fails strict AST: {row['sample_id']}")
            cause = "survives_but_loses"
        buckets[cause] += 1
        detail: dict[str, Any] = {
            "sample_id": row["sample_id"],
            "cause": cause,
            "target_tokens": row["target_tokens"],
            "selected_tokens": row["selected_tokens"],
            "target_beam_rank": row["target_beam_rank"],
            "first_lost_position": row["first_lost_position"],
            "first_lost_reason": row["first_lost_reason"],
        }
        if cause == "top5_missing":
            trace = trace_by_id[row["sample_id"]]
            glyphs = []
            for symbol in trace["oracle_group_hwr"]["symbols"]:
                prediction = symbol["prediction"]
                if symbol["target_label"] in prediction["top5"]:
                    continue
                residual_top5_misses[
                    "in_vocab" if symbol["target_label"] in label_set else "oov"
                ] += 1
                glyphs.append({
                    "target": symbol["target_label"],
                    "in_checkpoint_vocabulary": symbol["target_label"] in label_set,
                    "stroke_indices": symbol["stroke_indices"],
                    "raw_point_count": symbol["preprocessing"]["raw_point_count"],
                    "raw_points_per_stroke": symbol["preprocessing"]["raw_points_per_stroke"],
                    "target_rank": prediction["target_rank"],
                    "target_probability": prediction["target_probability"],
                    "top1": prediction["top1"],
                    "top1_probability": prediction["top1_probability"],
                    "top5": prediction["top5"],
                    "layer_trajectory": [
                        {
                            "stage": layer["stage"],
                            "top1": layer["predicted_top1"],
                            "target_rank": layer["target_rank"],
                            "target_margin_logit": layer["target_minus_best_other_logit"],
                        }
                        for layer in symbol["layer_logit_lens"]
                    ],
                })
            detail["missing_glyphs"] = glyphs
        elif cause == "survives_but_loses":
            decomp = row["score_gap_decomposition"]
            relation_gap = float(decomp["relation_component"])
            reconstructed = float(decomp["reconstructed_total"])
            total = float(row["target_to_selected_score_gap"])
            error = abs(reconstructed - total)
            score_checks += 1
            score_decomposition_max_abs_error = max(score_decomposition_max_abs_error, error)
            survivor_relation_gaps.append(relation_gap)
            detail["score_gap_decomposition"] = decomp
            detail["token_level_selection"] = row["token_level_selection"]
        residual_details.append(detail)

    grouping_summary = grouping["summary"]
    grouping_outcomes = grouping_summary["outcomes"]
    grouping_error_count = int(grouping_summary["fast_group_errors"])
    grouping_outcome_error_count = sum(
        int(count) for name, count in grouping_outcomes.items() if name != "fast_partition_exact"
    )
    expected_trace_sha = _sha256(traces_path)
    decoder_trace_sha = decoder["inputs"].get("traces_sha256")
    grouping_trace_sha = grouping["inputs"].get("traces_sha256")
    beam_comparison = None
    extra_checks: dict[str, bool] = {}
    if wide_beam_path is not None:
        wide = _json(wide_beam_path)
        wide_by_id = {row["sample_id"]: row for row in wide["formula_level"]}
        base_by_id = {row["sample_id"]: row for row in arm_rows}
        prior_beam_pruned = [
            row for row in residuals
            if row["candidate_complete"] and row["target_beam_rank"] is None
        ]
        reached_by_wide_beam = [
            row for row in prior_beam_pruned
            if wide_by_id[row["sample_id"]]["target_beam_rank"] is not None
        ]
        reached_not_selected = [
            row for row in reached_by_wide_beam
            if wide_by_id[row["sample_id"]]["selected_tokens"] != row["target_tokens"]
        ]
        wide_exact_selected = sum(
            wide_by_id[row["sample_id"]]["selected_tokens"] == row["target_tokens"]
            for row in residuals
        )
        beam_comparison = {
            "narrow_beam_width": decoder["beam_width"],
            "wide_beam_width": wide["beam_width"],
            "prior_pruned_formula_count": len(prior_beam_pruned),
            "prior_pruned_targets_reaching_wide_beam": len(reached_by_wide_beam),
            "prior_pruned_targets_still_pruned": len(prior_beam_pruned) - len(reached_by_wide_beam),
            "newly_reached_targets_selected_exactly": len(reached_by_wide_beam) - len(reached_not_selected),
            "newly_reached_but_not_selected": [
                {
                    "sample_id": row["sample_id"],
                    "wide_beam_target_rank": wide_by_id[row["sample_id"]]["target_beam_rank"],
                    "selected_tokens": wide_by_id[row["sample_id"]]["selected_tokens"],
                    "target_tokens": row["target_tokens"],
                    "score_gap_decomposition": wide_by_id[row["sample_id"]]["score_gap_decomposition"],
                }
                for row in reached_not_selected
            ],
            "all_prior_residuals_exactly_selected_at_wide_width": wide_exact_selected,
        }
        extra_checks = {
            "wide_beam_same_arm_and_inputs": (
                wide["arm"] == decoder["arm"]
                and wide["beam_width"] > decoder["beam_width"]
                and wide["inputs"].get("traces_sha256") == decoder_trace_sha == expected_trace_sha
                and wide["inputs"].get("shadow_sha256") == decoder["inputs"].get("shadow_sha256")
                and set(wide_by_id) == set(base_by_id)
            ),
            "wide_beam_does_not_promote_any_prior_residual": wide_exact_selected == 0,
        }

    checks = {
        "trace_ids_unique": len(trace_by_id) == len(traces),
        "oracle_symbol_count_matches_target_token_count": trace_target_symbol_count
        == sum(len(row["source"]["target_tokens"]) for row in traces),
        "decoder_trace_hash_matches": decoder_trace_sha == expected_trace_sha,
        "grouping_trace_hash_matches": grouping_trace_sha == expected_trace_sha,
        "decoder_replay_parity_all_formulas": decoder["verification"]["checks"]["decoder_replay_matches_saved_trace_for_every_formula"],
        "decoder_failure_buckets_reconcile": sum(buckets.values()) == len(residuals),
        "top5_missing_formula_rows_have_missing_glyph_evidence": all(
            row.get("missing_glyphs") for row in residual_details if row["cause"] == "top5_missing"
        ),
        "checkpoint_and_trace_vocab_agree": not oov_tokens or all(
            item["target"] not in label_set for item in oov_tokens
        ),
        "survivor_score_gaps_reconcile": score_checks == buckets["survives_but_loses"]
        and score_decomposition_max_abs_error <= 1e-9,
        "survivor_relation_component_is_zero": all(abs(gap) <= 1e-12 for gap in survivor_relation_gaps),
        "all_surviving_targets_are_strict_ast_valid": invalid_survivor_count == 0,
        "grouping_failure_outcomes_reconcile": grouping_outcome_error_count == grouping_error_count,
        "local_partition_enumeration_is_exact_cover_only": grouping_summary.get(
            "local_partition_partitions_enumerated"
        ) == grouping_summary.get("local_partition_exact_cover_partitions"),
        "grouping_replay_all_checks_pass": grouping["verification"]["all_checks_pass"],
        "critical_layer_trace_checks_all_pass": not trace_checks,
    }
    checks.update(extra_checks)
    return {
        "schema": SCHEMA,
        "scope": "forensic replay only; no fitting, threshold selection, CROHME access, or promotion",
        "inputs": {
            "traces": {"path": str(traces_path), "sha256": expected_trace_sha},
            "decoder_audit": {"path": str(decoder_path), "sha256": _sha256(decoder_path)},
            "grouping_audit": {"path": str(grouping_path), "sha256": _sha256(grouping_path)},
            "checkpoint": {"path": str(checkpoint_path), "sha256": _sha256(checkpoint_path)},
            "wide_beam_audit": (
                {"path": str(wide_beam_path), "sha256": _sha256(wide_beam_path)}
                if wide_beam_path is not None else None
            ),
        },
        "checkpoint_vocabulary": {"classes": len(labels), "oov_target_token_count": len(oov_tokens)},
        "summary": {
            "formulas": len(traces),
            "target_tokens": sum(len(row["source"]["target_tokens"]) for row in traces),
            "layer_trace_failed_checks": dict(trace_checks),
            "all_formula_residuals": len(residuals),
            "decoder_failure_causes": dict(sorted(buckets.items())),
            "all_trace_top5_missing_glyphs_by_vocabulary_status": dict(sorted(target_top5_misses.items())),
            "residual_top5_missing_glyphs_by_vocabulary_status": dict(sorted(residual_top5_misses.items())),
            "surviving_valid_targets": score_checks,
            "survivor_relation_component_nonzero_count": sum(abs(gap) > 1e-12 for gap in survivor_relation_gaps),
            "survivor_score_decomposition_max_abs_error": score_decomposition_max_abs_error,
            "grouping_fast_exact": grouping_outcomes.get("fast_partition_exact"),
            "grouping_fast_errors": grouping_error_count,
            "grouping_failure_outcomes": grouping_outcomes,
            "grouping_local_partitions_enumerated": grouping_summary.get("local_partition_partitions_enumerated"),
            "grouping_local_exact_cover_partitions": grouping_summary.get("local_partition_exact_cover_partitions"),
        },
        "oov_target_details": oov_tokens,
        "residual_details": residual_details,
        "beam_width_comparison": beam_comparison,
        "verification": {"checks": checks, "all_checks_pass": all(checks.values())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--decoder-audit", type=Path, required=True)
    parser.add_argument("--grouping-audit", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--wide-beam-audit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite output: {args.output}")
    report = audit(
        args.traces,
        args.decoder_audit,
        args.grouping_audit,
        args.checkpoint,
        args.wide_beam_audit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": report["summary"], "verification": report["verification"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
