#!/usr/bin/env python3
"""Partition guarded HWR replay failures into disjoint, auditable causes."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


GUARDED_ARM = "selective_joint_hwr_geometry_prior_group_count_guard"
FINAL_STAGE = "after_boundary_bar_as_unit_guard"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decompose(summary: dict, *, source_path: Path) -> dict:
    if summary.get("schema") != "aiflow-selective-2d-research-loop/v1":
        raise ValueError("unexpected source summary schema")
    if summary.get("product_default_enabled") is not False:
        raise AssertionError("source summary enabled product defaults")
    if summary.get("crohme_training_or_tuning") is not False:
        raise AssertionError("source summary crossed the CROHME boundary")
    if summary.get("promotion_eligible") is not False:
        raise AssertionError("source summary is not shadow-only")
    if not summary.get("run_config", {}).get("group_count_increase_guard_shadow"):
        raise ValueError("source summary lacks the opt-in guarded arm")

    rows = summary.get("records") or []
    if len(rows) != int(summary.get("formulas", -1)):
        raise AssertionError("record count does not match formula count")
    buckets: dict[str, list[str]] = {
        "group_partition_wrong": [],
        "target_outside_hwr_top5": [],
        "target_in_top5_but_token_sequence_wrong": [],
        "group_and_token_sequence_exact": [],
    }
    group_error_shapes = Counter()
    rejected_guard_ids = []
    candidate_present_token_errors = []
    error_rank_counts = Counter()
    error_choice_counts = Counter()
    for record in rows:
        sample_id = str(record["sample_id"])
        metrics = record["hwr_tournament"][GUARDED_ARM]
        if int(metrics.get("local_group_count_increase_rejections", 0)):
            rejected_guard_ids.append(sample_id)
        if not metrics["group_exact"]:
            buckets["group_partition_wrong"].append(sample_id)
            group_error_shapes[str(metrics["group_error_shape"])] += 1
            continue
        if metrics.get("hwr_top5_complete_if_groups_exact") is not True:
            buckets["target_outside_hwr_top5"].append(sample_id)
            continue
        semantic = metrics.get("semantic_guard_shadow_if_groups_exact")
        if not isinstance(semantic, dict):
            raise AssertionError(f"semantic shadow unavailable on exact groups: {sample_id}")
        final_exact = semantic["formula_exact_by_stage"].get(FINAL_STAGE)
        if final_exact is True:
            buckets["group_and_token_sequence_exact"].append(sample_id)
        else:
            buckets["target_in_top5_but_token_sequence_wrong"].append(sample_id)
            groups = [tuple(sorted(int(index) for index in group)) for group in metrics["groups"]]
            target_groups = [
                tuple(sorted(int(index) for index in group)) for group in metrics["target_groups"]
            ]
            target_tokens = [str(token) for token in metrics["target_tokens"]]
            decoder_tokens = [str(token) for token in metrics["decoder_tokens"]]
            if len(target_groups) != len(target_tokens) or len(groups) != len(decoder_tokens):
                raise AssertionError(f"group/token sequence length mismatch: {sample_id}")
            targets_by_group = dict(zip(target_groups, target_tokens, strict=True))
            predictions_by_group = dict(zip(groups, decoder_tokens, strict=True))
            symbols_by_group = {
                tuple(sorted(int(index) for index in symbol["stroke_indices"])): symbol
                for symbol in metrics["selected_symbols"]
            }
            token_errors = []
            for group in groups:
                target = targets_by_group[group]
                predicted = predictions_by_group[group]
                if predicted == target:
                    continue
                topk = [str(token) for token in symbols_by_group[group]["hwr_topk"]]
                rank = topk.index(target) + 1
                error_rank_counts[str(rank)] += 1
                hwr_top1 = topk[0]
                choice_kind = "decoder_kept_hwr_top1" if predicted == hwr_top1 else "decoder_changed_hwr_top1"
                error_choice_counts[choice_kind] += 1
                token_errors.append({
                    "stroke_indices": list(group),
                    "target": target,
                    "target_hwr_rank": rank,
                    "hwr_top1": hwr_top1,
                    "decoder_token": predicted,
                    "decoder_choice_vs_hwr_top1": choice_kind,
                })
            candidate_present_token_errors.append({
                "sample_id": sample_id,
                "base_decoder_token_errors": token_errors,
            })

    counts = {name: len(ids) for name, ids in buckets.items()}
    if sum(counts.values()) != len(rows):
        raise AssertionError("failure buckets overlap or omit formulas")
    arm_summary = summary["hwr_tournament"]["arms"][GUARDED_ARM]
    if counts["group_partition_wrong"] != len(rows) - int(arm_summary["group_exact"]):
        raise AssertionError("group-failure count does not reconcile with arm summary")
    if counts["group_and_token_sequence_exact"] != int(
        summary["semantic_guard_shadow"]["arms"][GUARDED_ARM]["formula_exact_by_stage"][FINAL_STAGE]
    ):
        raise AssertionError("group+token exact count does not reconcile with semantic summary")

    prior_arm = "selective_joint_hwr_geometry_prior"
    fast_arm = "fast"
    stage_counts = {}
    for arm in (fast_arm, prior_arm, GUARDED_ARM):
        stage_summary = summary["semantic_guard_shadow"]["arms"][arm]
        stage_counts[arm] = {
            "evaluated_group_exact_formulas": stage_summary.get("evaluated_group_exact_formulas"),
            "group_and_token_sequence_exact_by_stage": stage_summary["formula_exact_by_stage"],
            "token_hits_by_stage": stage_summary["token_hits_by_stage"],
        }

    return {
        "schema": "aiflow-hwr-guarded-failure-decomposition/v1",
        "status": "consumed_development_shadow_only",
        "source_summary": str(source_path.resolve()),
        "source_summary_sha256": _sha256(source_path),
        "formulas": len(rows),
        "arm": GUARDED_ARM,
        "disjoint_failure_partition": {
            "counts": counts,
            "ids": buckets,
            "group_error_shapes": dict(sorted(group_error_shapes.items())),
        },
        "candidate_present_residual_token_errors": {
            "scope": "base decoder choices inside formulas still inexact after the semantic shadow",
            "wrong_token_count_by_target_hwr_rank": dict(sorted(error_rank_counts.items())),
            "wrong_token_count_by_decoder_vs_hwr_top1": dict(sorted(error_choice_counts.items())),
            "formula_token_errors": candidate_present_token_errors,
        },
        "hwr_candidate_coverage_on_exact_groups": {
            "target_token_count": int(arm_summary["hwr_tokens_evaluable_on_exact_groups"]),
            "top5_hits": int(arm_summary["hwr_top5_token_hits_on_exact_groups"]),
            "target_rank_histogram": arm_summary["hwr_target_rank_histogram_on_exact_groups"],
        },
        "guard_rejected_group_count_increase_ids": rejected_guard_ids,
        "stagewise_group_and_token_sequence_metrics": stage_counts,
        "paired_guard_transition_vs_fast": summary["group_count_increase_guard_shadow"],
        "product_default_enabled": False,
        "crohme_training_or_tuning": False,
        "promotion_eligible": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.summary.resolve()
    result = decompose(json.loads(source.read_text(encoding="utf-8")), source_path=source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "formulas": result["formulas"],
        "disjoint_failure_partition": result["disjoint_failure_partition"]["counts"],
        "group_error_shapes": result["disjoint_failure_partition"]["group_error_shapes"],
        "output": str(args.output.resolve()),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
