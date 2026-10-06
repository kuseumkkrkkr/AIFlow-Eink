#!/usr/bin/env python3
"""Run new candidate-preserving formula guards on raw CROHME strokes.

CROHME is used only as a repeated noncommercial diagnostic.  The evaluator
passes no truth, writer, or target glyph count into inference and reruns the
pre-merge baseline only for formulas whose selected partition changed.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path

from audit_replay_protocol_v1 import CROHME_ALIASES, DEFAULT_CROHME, _crohme_sample
from evaluate_48hz_prefix_v1 import _sha256
from formula_acceptance_guard_v1 import (
    AUTO_ACCEPTED, assess_formula, load_configuration,
)
from raw_formula_context_runtime_v1 import RawFormulaContextRuntimeV1
from replay_evaluate_hwr_v1 import load_crohme


SCHEMA = "aiflow-formula-context-crohme-diagnostic/v1"
TICK_MS = 1000.0 / 48.0


def _source(formula_id: str, sample: dict) -> dict:
    tick = 0
    strokes = []
    for order, stroke in enumerate(sample["strokes"]):
        points = []
        for x, y, _ in stroke:
            points.append({"x": float(x), "y": float(y), "t_ms": tick * TICK_MS})
            tick += 1
        strokes.append({"order": order, "points": points})
    return {"formula_id": formula_id, "strokes": strokes}


def _grouping_exact(result: dict, truth_groups: list[list[int]]) -> bool:
    selected = {frozenset(row["stroke_indices"]) for row in result["groups"]}
    truth = {frozenset(int(value) for value in group) for group in truth_groups}
    return selected == truth


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition-ranker", type=Path, required=True)
    parser.add_argument("--hwr-checkpoint", type=Path, required=True)
    parser.add_argument("--context-checkpoint", type=Path, required=True)
    parser.add_argument("--formula-sequence-config", type=Path, required=True)
    parser.add_argument("--formula-syntax-rescue-config", type=Path, required=True)
    parser.add_argument("--candidate-context-fusion-config", type=Path, required=True)
    parser.add_argument("--candidate-context-auxiliary-hwr-checkpoint", type=Path, required=True)
    parser.add_argument("--formula-placement-config", type=Path, required=True)
    parser.add_argument("--straight-equality-config", type=Path, required=True)
    parser.add_argument("--latin-auxiliary-checkpoint", type=Path, required=True)
    parser.add_argument("--pairwise-shape-expert", type=Path)
    parser.add_argument("--pairwise-shape-config", type=Path)
    parser.add_argument("--formula-acceptance-config", type=Path)
    parser.add_argument("--crohme", type=Path, default=DEFAULT_CROHME)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    crohme = args.crohme.expanduser().resolve()
    if output.exists() or output.drive.upper() != "D:":
        parser.error("output must be a new file on D:")
    if not crohme.is_dir():
        parser.error("CROHME root is missing")
    acceptance_configuration = None
    acceptance_configuration_sha256 = None
    if args.formula_acceptance_config is not None:
        acceptance_configuration, acceptance_configuration_sha256 = load_configuration(
            args.formula_acceptance_config,
        )
    runtime = RawFormulaContextRuntimeV1.from_artifacts(
        args.partition_ranker, args.hwr_checkpoint, args.context_checkpoint,
        device=args.device,
        formula_sequence_config=args.formula_sequence_config,
        formula_syntax_rescue_config=args.formula_syntax_rescue_config,
        candidate_context_fusion_config=args.candidate_context_fusion_config,
        candidate_context_auxiliary_hwr_checkpoint=(
            args.candidate_context_auxiliary_hwr_checkpoint
        ),
        formula_placement_config=args.formula_placement_config,
        straight_equality_config=args.straight_equality_config,
        latin_auxiliary_checkpoint=args.latin_auxiliary_checkpoint,
        pairwise_shape_expert=args.pairwise_shape_expert,
        pairwise_shape_config=args.pairwise_shape_config,
        allow_posthoc_shadow=True,
    )
    if runtime.repeat_merge_configuration is None:
        raise ValueError("repeat merge configuration is not enabled")
    if runtime.latin_configuration is None:
        raise ValueError("Latin t context configuration is not enabled")
    if not runtime.cross_merge_configuration.get(
        "auxiliary_nested_expression_enabled", False,
    ):
        raise ValueError("nested-expression cross merge is not enabled")
    if not runtime.placement_configuration["formula_role_typo"].get(
        "value_cross_plus_open_fence_enabled", False,
    ):
        raise ValueError("open-fence plus rescue is not enabled")
    if runtime.singleton_configuration["token_candidate_maximum_ranks"] != {
        "/": 1, r"\times": 2,
    }:
        raise ValueError("ranked singleton rescue is not enabled")
    pairwise_enabled = runtime.pairwise_shape_expert is not None
    if pairwise_enabled:
        baseline_runtime = replace(
            runtime,
            pairwise_shape_expert=None,
            pairwise_shape_configuration=None,
            pairwise_shape_expert_sha256=None,
            pairwise_shape_config_sha256=None,
        )
        baseline_mode = "same_runtime_without_pairwise_shape_rescue"
    else:
        baseline_singleton = deepcopy(runtime.singleton_configuration)
        baseline_singleton["token_confidence_thresholds"][r"\times"] = 0.45
        baseline_singleton["token_candidate_maximum_ranks"] = {
            "/": 1, r"\times": 1,
        }
        baseline_runtime = replace(
            runtime,
            singleton_configuration=baseline_singleton,
        )
        baseline_mode = "legacy_singleton_rank1_runtime"
    protocol, duplicates = load_crohme(crohme)
    labels = set(runtime.labels)
    eligible = []
    skipped = {
        "missing_truth_partition": 0,
        "incomplete_truth_partition": 0,
        "unsupported_truth_label": 0,
        "single_point_source_stroke": 0,
    }
    for formula in protocol:
        formula_id = str(formula["record_id"])
        sample = _crohme_sample(crohme / formula_id)
        if sample is None:
            skipped["missing_truth_partition"] += 1
            continue
        if not sample["partition_complete"]:
            skipped["incomplete_truth_partition"] += 1
            continue
        truth_tokens = [CROHME_ALIASES.get(str(value), str(value)) for value in sample["labels"]]
        if any(token not in labels for token in truth_tokens):
            skipped["unsupported_truth_label"] += 1
            continue
        if any(len(stroke) < 2 for stroke in sample["strokes"]):
            skipped["single_point_source_stroke"] += 1
            continue
        eligible.append((formula_id, sample, truth_tokens))

    candidate_results = {}
    changed_ids = []
    repeat_changed_ids = []
    single_equality_changed_ids = []
    latin_t_changed_ids = []
    nested_cross_changed_ids = []
    open_fence_plus_changed_ids = []
    singleton_audit_ids = []
    pairwise_shape_changed_ids = []
    for formula_id, sample, _truth_tokens in eligible:
        result = runtime.infer(_source(formula_id, sample))
        candidate_results[formula_id] = result
        fusion_audit = result["audit"]["candidate_context_fusion"]
        if fusion_audit["repeat_merge_rescue"].get("changed_formulas", 0):
            repeat_changed_ids.append(formula_id)
        equality_changes = fusion_audit["straight_equality_rescue"].get("changes", [])
        if any(
            change.get("rule") == "single_glyph_aligned_straight_component_pair"
            for change in equality_changes
        ):
            single_equality_changed_ids.append(formula_id)
        if fusion_audit["latin_t_context_rescue"].get("changed_formulas", 0):
            latin_t_changed_ids.append(formula_id)
        if any(
            change.get("admission_rule") == "auxiliary_nested_expression"
            for change in fusion_audit["cross_merge_rescue"].get("changes", [])
        ):
            nested_cross_changed_ids.append(formula_id)
        role_changes = fusion_audit["formula_placement_rescue"].get(
            "stage_audits", {},
        ).get("formula_role_typo", {}).get("changes", [])
        if any(
            change.get("rule") == "value_cross_plus_open_fence"
            for change in role_changes
        ):
            open_fence_plus_changed_ids.append(formula_id)
        if fusion_audit.get("changed", 0):
            singleton_audit_ids.append(formula_id)
        if result["audit"]["pairwise_shape_rescue"].get("changed", 0):
            pairwise_shape_changed_ids.append(formula_id)
    baseline_trigger_ids = set(
        pairwise_shape_changed_ids if pairwise_enabled else singleton_audit_ids
    )
    baseline_results = {
        formula_id: baseline_runtime.infer(_source(formula_id, sample))
        for formula_id, sample, _truth_tokens in eligible
        if formula_id in baseline_trigger_ids
    }
    ranked_singleton_changed_ids = (
        [] if pairwise_enabled else sorted(
            formula_id for formula_id, baseline in baseline_results.items()
            if baseline["finalized_tokens"]
            != candidate_results[formula_id]["finalized_tokens"]
        )
    )
    changed_ids = sorted(
        pairwise_shape_changed_ids if pairwise_enabled
        else ranked_singleton_changed_ids
    )
    grouping_before = grouping_after = formula_before = formula_after = 0
    grouping_improved = []; grouping_regressed = []
    formula_improved = []; formula_regressed = []
    changes = []
    for formula_id, sample, truth_tokens in eligible:
        candidate = candidate_results[formula_id]
        baseline = baseline_results.get(formula_id, candidate)
        before_group = _grouping_exact(baseline, sample["groups"])
        after_group = _grouping_exact(candidate, sample["groups"])
        before_formula = baseline["finalized_tokens"] == truth_tokens
        after_formula = candidate["finalized_tokens"] == truth_tokens
        grouping_before += before_group; grouping_after += after_group
        formula_before += before_formula; formula_after += after_formula
        if after_group and not before_group:
            grouping_improved.append(formula_id)
        if before_group and not after_group:
            grouping_regressed.append(formula_id)
        if after_formula and not before_formula:
            formula_improved.append(formula_id)
        if before_formula and not after_formula:
            formula_regressed.append(formula_id)
        if formula_id in baseline_results:
            changes.append({
                "formula_id": formula_id,
                "truth_tokens": truth_tokens,
                "before_tokens": baseline["finalized_tokens"],
                "after_tokens": candidate["finalized_tokens"],
                "before_groups": [row["stroke_indices"] for row in baseline["groups"]],
                "after_groups": [row["stroke_indices"] for row in candidate["groups"]],
                "repeat_merge_audit": candidate["audit"]["candidate_context_fusion"][
                    "repeat_merge_rescue"
                ],
                "straight_equality_audit": candidate["audit"]["candidate_context_fusion"][
                    "straight_equality_rescue"
                ],
                "latin_t_context_audit": candidate["audit"]["candidate_context_fusion"][
                    "latin_t_context_rescue"
                ],
                "cross_merge_audit": candidate["audit"]["candidate_context_fusion"][
                    "cross_merge_rescue"
                ],
                "formula_placement_audit": candidate["audit"][
                    "candidate_context_fusion"
                ]["formula_placement_rescue"],
                "singleton_shape_audit": {
                    "configuration": candidate["audit"][
                        "candidate_context_fusion"
                    ]["configuration"],
                    "changes": candidate["audit"][
                        "candidate_context_fusion"
                    ]["changes"],
                },
                "pairwise_shape_audit": candidate["audit"][
                    "pairwise_shape_rescue"
                ],
            })
    acceptance_report = {"enabled": False}
    if acceptance_configuration is not None:
        decisions = {
            formula_id: assess_formula(candidate_results[formula_id], acceptance_configuration)
            for formula_id, _sample, _truth_tokens in eligible
        }
        accepted_ids = [
            formula_id for formula_id, decision in decisions.items()
            if decision["decision_status"] == AUTO_ACCEPTED
        ]
        truth_by_id = {
            formula_id: truth_tokens for formula_id, _sample, truth_tokens in eligible
        }
        accepted_exact = sum(
            candidate_results[formula_id]["finalized_tokens"] == truth_by_id[formula_id]
            for formula_id in accepted_ids
        )
        reason_counts = Counter(
            reason for decision in decisions.values()
            for reason in decision["review_reasons"]
        )
        acceptance_report = {
            "schema": "aiflow-formula-acceptance-crohme-diagnostic/v1",
            "enabled": True,
            "configuration_sha256": acceptance_configuration_sha256,
            "formulas": len(decisions),
            "auto_accepted": len(accepted_ids),
            "auto_accept_coverage": len(accepted_ids) / max(len(decisions), 1),
            "auto_accepted_exact": accepted_exact,
            "auto_accepted_errors": len(accepted_ids) - accepted_exact,
            "auto_accepted_precision": accepted_exact / max(len(accepted_ids), 1),
            "review_required": len(decisions) - len(accepted_ids),
            "review_reasons": dict(sorted(reason_counts.items())),
            "candidate_preserving": True,
            "token_mutations": 0,
            "grouping_mutations": 0,
            "arithmetic_evaluation": False,
            "product_validation": False,
            "product_default_enabled": False,
        }
    report = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "repeated_noncommercial_diagnostic_only",
        "training_performed": False,
        "product_validation": False,
        "dataset": "CROHME CC BY-NC repeated raw-stroke diagnostic only",
        "coverage": {
            "protocol_formulas": len(protocol),
            "exact_formula_duplicates_removed": len(duplicates),
            "eligible_formulas": len(eligible),
            "skipped": skipped,
        },
        "scores": {
            "grouping_exact_before": grouping_before,
            "grouping_exact_after": grouping_after,
            "formula_exact_before": formula_before,
            "formula_exact_after": formula_after,
            "grouping_improved": grouping_improved,
            "grouping_regressed": grouping_regressed,
            "formula_improved": formula_improved,
            "formula_regressed": formula_regressed,
        },
        "repeat_merge": {
            "changed_formulas": len(repeat_changed_ids),
            "changed_formula_ids": sorted(repeat_changed_ids),
        },
        "single_glyph_equality": {
            "changed_formulas": len(single_equality_changed_ids),
            "changed_formula_ids": sorted(single_equality_changed_ids),
        },
        "latin_t_context": {
            "changed_formulas": len(latin_t_changed_ids),
            "changed_formula_ids": sorted(latin_t_changed_ids),
        },
        "nested_expression_cross_merge": {
            "changed_formulas": len(nested_cross_changed_ids),
            "changed_formula_ids": sorted(nested_cross_changed_ids),
        },
        "open_fence_plus_rescue": {
            "changed_formulas": len(open_fence_plus_changed_ids),
            "changed_formula_ids": sorted(open_fence_plus_changed_ids),
        },
        "ranked_singleton_rescue": {
            "changed_formulas": len(ranked_singleton_changed_ids),
            "changed_formula_ids": sorted(ranked_singleton_changed_ids),
        },
        "pairwise_shape_rescue": {
            "enabled": pairwise_enabled,
            "baseline_mode": baseline_mode,
            "changed_formulas": len(pairwise_shape_changed_ids),
            "changed_formula_ids": sorted(pairwise_shape_changed_ids),
        },
        "formula_acceptance_guard": acceptance_report,
        "current_loop": {
            "changed_formulas": len(changed_ids),
            "changed_formula_ids": sorted(changed_ids),
            "changes": changes,
        },
        "contracts": {
            "all_strokes_exactly_once": True,
            "target_label_or_glyph_count_input": False,
            "candidate_preserving": True,
            "arithmetic_evaluation": False,
            "product_default_enabled": False,
        },
        "inputs": {
            "partition_ranker_sha256": _sha256(args.partition_ranker.resolve()),
            "hwr_checkpoint_sha256": _sha256(args.hwr_checkpoint.resolve()),
            "context_checkpoint_sha256": _sha256(args.context_checkpoint.resolve()),
            "candidate_context_fusion_config_sha256": _sha256(
                args.candidate_context_fusion_config.resolve()
            ),
            "latin_auxiliary_checkpoint_sha256": _sha256(
                args.latin_auxiliary_checkpoint.resolve()
            ),
            "pairwise_shape_expert_sha256": (
                _sha256(args.pairwise_shape_expert.resolve())
                if args.pairwise_shape_expert is not None else None
            ),
            "pairwise_shape_config_sha256": (
                _sha256(args.pairwise_shape_config.resolve())
                if args.pairwise_shape_config is not None else None
            ),
            "formula_acceptance_config_sha256": acceptance_configuration_sha256,
        },
        "limits": [
            "CROHME is noncommercial repeated diagnostic evidence, not product validation",
            "formulas containing unsupported labels or single-point source strokes are excluded",
            "fresh commercial writer/formula-disjoint acceptance remains required",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    print(json.dumps({
        "output": str(output), "sha256": _sha256(output),
        "coverage": report["coverage"], "scores": report["scores"],
        "repeat_merge_changed": len(repeat_changed_ids),
        "single_glyph_equality_changed": len(single_equality_changed_ids),
        "latin_t_context_changed": len(latin_t_changed_ids),
        "nested_expression_cross_merge_changed": len(nested_cross_changed_ids),
        "open_fence_plus_rescue_changed": len(open_fence_plus_changed_ids),
        "ranked_singleton_rescue_changed": len(ranked_singleton_changed_ids),
        "pairwise_shape_rescue_changed": len(pairwise_shape_changed_ids),
        "formula_acceptance_guard": acceptance_report,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
