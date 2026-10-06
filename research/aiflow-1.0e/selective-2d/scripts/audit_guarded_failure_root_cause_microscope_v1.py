#!/usr/bin/env python3
"""Join guarded formula failures to existing reachability and layer audits."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(
    summary_path: Path,
    decomposition_path: Path,
    partition_path: Path,
    layer_path: Path,
) -> dict[str, Any]:
    summary = _read(summary_path)
    decomposition = _read(decomposition_path)
    partition = _read(partition_path)
    layer = _read(layer_path)

    if summary.get("schema") != "aiflow-selective-2d-research-loop/v1":
        raise ValueError("unexpected guarded summary schema")
    if summary.get("crohme_training_or_tuning") is not False:
        raise AssertionError("CROHME boundary was crossed")
    if summary.get("product_default_enabled") is not False:
        raise AssertionError("product default is unexpectedly enabled")
    if decomposition.get("source_summary_sha256") != _sha256(summary_path):
        raise AssertionError("decomposition does not match guarded summary")

    groups = decomposition["disjoint_failure_partition"]["ids"]
    group_ids = list(groups["group_partition_wrong"])
    top5_ids = list(groups["target_outside_hwr_top5"])
    records = {str(row["sample_id"]): row for row in summary.get("records", [])}
    if len(records) != 149:
        raise AssertionError("guarded summary must contain 149 unique formula records")
    if not set(group_ids) <= set(records):
        raise AssertionError("guarded grouping failures do not join to formula records")
    partition_rows = {str(row["sample_id"]): row for row in partition["formula_level"]}
    layer_rows = {str(row["sample_id"]): row for row in layer["residual_details"]}
    if len(partition_rows) != 149 or len(layer_rows) != len(layer["residual_details"]):
        raise AssertionError("forensic source rows are incomplete or duplicated")
    if not set(group_ids) <= set(partition_rows):
        raise AssertionError("guarded grouping failures do not join to partition audit")

    group_causes: Counter[str] = Counter()
    grouped_details = []
    token_exact_group_failures = []
    feature_effects: dict[str, list[float]] = {}
    feature_flips: Counter[str] = Counter()
    feature_comparisons = 0
    feature_comparison_formulas: set[str] = set()
    for sample_id in group_ids:
        row = partition_rows[sample_id]
        outcome = str(row["outcome"])
        replay = row.get("local_partition_search_replay") or {}
        guarded_metrics = records[sample_id]["hwr_tournament"][decomposition["arm"]]
        target_tokens = [str(token) for token in guarded_metrics["target_tokens"]]
        predicted_tokens = [str(token) for token in guarded_metrics["decoder_tokens"]]
        token_sequence_exact = predicted_tokens == target_tokens
        if token_sequence_exact:
            token_exact_group_failures.append(sample_id)
        group_causes[outcome] += 1
        grouped_details.append({
            "sample_id": sample_id,
            "error_shape": row["partition_error_shape"],
            "cause": outcome,
            "token_sequence_exact_despite_group_error": token_sequence_exact,
            "local_region_seed_strokes": row["local_region_seed_stroke_count"],
            "local_region_strokes": row["local_region_stroke_count"],
            "local_candidate_count": row["local_candidate_count_stored"],
            "target_partition_reachable": row["target_partition_reachable_in_local_graph"],
            "locked_boundary_crossings": row["target_groups_crossing_region"],
            "locked_group_mismatches": {
                "target_outside_missing_from_locks": row["target_outside_groups_missing_from_locks"],
                "locked_groups_missing_from_target": row["locked_groups_missing_from_target_outside"],
            },
            "router_cues": row.get("router_cue_audit"),
            "search": {
                "outcome": replay.get("replay_outcome"),
                "target_rank": replay.get("target_partition_rank_in_local_top32"),
                "target_score_delta_vs_incumbent": replay.get(
                    "target_partition_score_delta_vs_fast_incumbent"
                ),
                "target_in_geometry_shortlist": replay.get(
                    "target_partition_was_in_geometry_shortlist"
                ),
            },
        })
        if outcome == "gold_partition_reachable_but_not_promoted":
            for comparison in replay.get("target_group_score_decomposition", []):
                if float(comparison["target_minus_fast_local_score"]) >= -1e-6:
                    continue
                feature_comparisons += 1
                feature_comparison_formulas.add(sample_id)
                for feature, effect in comparison[
                    "one_feature_equalization_counterfactuals"
                ].items():
                    feature_effects.setdefault(feature, []).append(float(
                        effect["delta_change_after_equalizing_feature"]
                    ))
                    if float(effect["counterfactual_target_minus_fast_logit"]) >= -1e-6:
                        feature_flips[feature] += 1

    feature_equalization_summary = [
        {
            "feature": feature,
            "negative_comparisons": len(values),
            "mean_delta_change": mean(values),
            "counterfactual_flips_to_nonnegative": feature_flips[feature],
        }
        for feature, values in sorted(
            feature_effects.items(),
            key=lambda item: (-mean(item[1]), item[0]),
        )
    ]

    top5_details = []
    for sample_id in top5_ids:
        row = layer_rows.get(sample_id)
        if row is None or row.get("cause") != "top5_missing":
            raise AssertionError(f"Top-5 miss lacks layer evidence: {sample_id}")
        top5_details.append({
            "sample_id": sample_id,
            "target_tokens": row["target_tokens"],
            "selected_tokens": row["selected_tokens"],
            "first_lost_position": row["first_lost_position"],
            "missing_glyphs": row["missing_glyphs"],
        })

    if set(top5_ids) != {
        sample_id for sample_id, row in layer_rows.items()
        if row.get("cause") == "top5_missing"
    }:
        raise AssertionError("guarded Top-5 misses disagree with layer-level audit")
    if sum(group_causes.values()) != len(group_ids):
        raise AssertionError("grouping cause buckets do not reconcile")

    return {
        "schema": "aiflow-hwr-guarded-failure-root-cause-microscope/v1",
        "scope": "consumed 149-formula development cohort; forensic replay only",
        "interpretation_limits": {
            "local_partition_replay": (
                "frozen geometry-ranker replay; candidate reachability is direct evidence, "
                "but replay scores are not direct attribution of the current joint-HWR score"
            ),
            "feature_equalization": (
                "posthoc correlated-feature counterfactual; not causal proof, threshold selection, "
                "or promotion evidence"
            ),
            "token_sequence_crosscheck": (
                "exact token-list comparison only; not normalized-LaTeX/relation-graph acceptance"
            ),
        },
        "inputs": {
            "guarded_summary": {"path": str(summary_path), "sha256": _sha256(summary_path)},
            "decomposition": {"path": str(decomposition_path), "sha256": _sha256(decomposition_path)},
            "partition_replay": {"path": str(partition_path), "sha256": _sha256(partition_path)},
            "layer_audit": {"path": str(layer_path), "sha256": _sha256(layer_path)},
        },
        "grouping_failure_causes": {
            "count": len(group_ids),
            "outcome_counts": dict(sorted(group_causes.items())),
            "token_sequence_exact_despite_group_error": {
                "count": len(token_exact_group_failures),
                "sample_ids": token_exact_group_failures,
                "metric_note": "token sequence only; not strict normalized-LaTeX accuracy",
            },
            "formulas": grouped_details,
        },
        "reachable_target_group_score_microscope": {
            "scope": "negative target-group versus overlapping Fast-group local-score comparisons in reachable-but-not-promoted formulas",
            "formulas": len(feature_comparison_formulas),
            "comparisons": feature_comparisons,
            "feature_equalization_effects": feature_equalization_summary,
            "interpretation": "posthoc one-feature equalization; correlation-sensitive diagnostic, not causal proof or promotion evidence",
        },
        "top5_candidate_misses": {"count": len(top5_ids), "formulas": top5_details},
        "candidate_present_decoder_residuals": decomposition[
            "candidate_present_residual_token_errors"
        ],
        "validation": {
            "guarded_summary_hash_matches_decomposition": True,
            "all_group_failures_joined": len(grouped_details) == len(group_ids),
            "all_group_failures_joined_to_formula_tokens": len(grouped_details) == len(group_ids),
            "token_exact_group_failures_are_subset": set(token_exact_group_failures) <= set(group_ids),
            "all_top5_misses_joined_to_layer_trace": len(top5_details) == len(top5_ids),
            "grouping_cause_counts_reconcile": sum(group_causes.values()) == len(group_ids),
            "crohme_training_or_tuning": False,
            "product_default_enabled": False,
            "promotion_eligible": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--decomposition", type=Path, required=True)
    parser.add_argument("--partition-replay", type=Path, required=True)
    parser.add_argument("--layer-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite output: {args.output}")
    result = audit(
        args.summary.resolve(), args.decomposition.resolve(),
        args.partition_replay.resolve(), args.layer_audit.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "grouping_failure_causes": result["grouping_failure_causes"]["outcome_counts"],
        "top5_candidate_misses": result["top5_candidate_misses"]["count"],
        "validation": result["validation"],
        "output": str(args.output.resolve()),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
