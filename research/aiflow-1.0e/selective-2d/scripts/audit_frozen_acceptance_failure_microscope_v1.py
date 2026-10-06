#!/usr/bin/env python3
"""Filter saved HWR/grouping/decoder diagnostics to the frozen 53-formula acceptance set."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import median
from typing import Any

import joblib

SCHEMA = "aiflow-hwr-frozen-acceptance-failure-microscope/v1"
BASELINE_STAGE = "fast_fence_infix_equation"
CONTEXT_STAGE = "context_fence_infix_equation"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def audit(provenance_path: Path, grouping_tournament_path: Path,
          partition_failure_path: Path, failure_matrix_path: Path,
          partition_ranker_path: Path) -> dict[str, Any]:
    provenance = _json(provenance_path)
    if not provenance["verification"].get("all_provenance_reconciliation_checks_pass"):
        raise ValueError("the input provenance audit did not pass reconciliation checks")
    if not provenance["verification"].get("frozen_acceptance_has_no_exact_or_normalized_ink_duplicates"):
        raise ValueError("frozen acceptance overlaps training ink under the saved signature audit")

    input_paths = provenance["inputs"]
    context_report_path = Path(input_paths["context_transfer_report"]["path"])
    formulas_path = Path(input_paths["formula_source"]["path"])
    trace_summary_path = Path(input_paths["trace_summary"]["path"])
    acceptance_path = Path(input_paths["frozen_acceptance_ownership"]["path"])
    traces_path = trace_summary_path.with_name("formula_traces.jsonl")
    context_report = _json(context_report_path)
    trace_summary = _json(trace_summary_path)
    acceptance_rows = _jsonl(acceptance_path)
    traces = _jsonl(traces_path)
    formulas = _jsonl(formulas_path)
    grouping_tournament = _json(grouping_tournament_path)
    partition_failure = _json(partition_failure_path)
    failure_matrix = _json(failure_matrix_path)
    partition_ranker = joblib.load(partition_ranker_path)

    accepted = {str(row["sample_id"]): row for row in acceptance_rows if row.get("accepted")}
    if len(accepted) != len(acceptance_rows):
        raise ValueError("frozen acceptance file contains rows not marked accepted")
    accepted_ids = set(accepted)
    formula_by_id = {str(row["sample_id"]): row for row in formulas}
    trace_by_id = {str(row["sample_id"]): row for row in traces}
    context_by_id = {str(row["sample_id"]): row for row in context_report["actual_fast_partition_formula_cases"]}
    tournament_by_id = {str(row["sample_id"]): row for row in grouping_tournament["records"]}
    partition_by_id = {str(row["sample_id"]): row for row in partition_failure["formula_level"]}
    if any(accepted_ids - set(mapping) for mapping in (
        formula_by_id, trace_by_id, context_by_id, tournament_by_id, partition_by_id
    )):
        raise ValueError("a frozen acceptance formula is missing from one or more diagnostic sources")

    trace_sha = _sha256(traces_path)
    expected_trace_hwr_sha = trace_summary["runtime"]["checkpoint_sha256"]
    decoder_trace_sha = failure_matrix["inputs"]["traces"]["sha256"]
    partition_trace_sha = partition_failure["inputs"]["traces_sha256"]
    if not (trace_sha == decoder_trace_sha == partition_trace_sha):
        raise ValueError("decoder, partition, and layer traces do not share the same source hash")
    if _sha256(formulas_path) != trace_summary["reproducibility"]["inputs"]["dataset_files"]["data/formulas_valid.jsonl"]:
        raise ValueError("frozen formula source hash differs from layer trace")
    if _sha256(formulas_path) != partition_failure["inputs"]["formulas_sha256"]:
        raise ValueError("frozen formula source hash differs from partition audit")
    if expected_trace_hwr_sha != context_report["checkpoints"]["current_hwr_sha256"]:
        raise ValueError("HWR checkpoint differs between context report and layer trace")
    if expected_trace_hwr_sha != grouping_tournament["inputs"]["checkpoint_sha256"]:
        raise ValueError("HWR checkpoint differs between grouping tournament and layer trace")
    ranker_sha = _sha256(partition_ranker_path)
    if ranker_sha != partition_failure["inputs"]["partition_ranker_sha256"]:
        raise ValueError("partition ranker hash differs from the per-formula partition audit")
    if ranker_sha != grouping_tournament["inputs"]["partition_ranker_sha256"]:
        raise ValueError("partition ranker hash differs from the grouping tournament")

    stage_rows = []
    token_total = top1_correct = top5_correct = 0
    top1_formula_exact = top5_formula_complete = 0
    oracle_group_complete_by_fast_group: dict[bool, list[bool]] = {True: [], False: []}
    decoder_residuals = {
        str(row["sample_id"]): row
        for row in failure_matrix["residual_details"]
        if str(row["sample_id"]) in accepted_ids
    }

    for sample_id in sorted(accepted_ids):
        owner = accepted[sample_id]
        formula = formula_by_id[sample_id]
        target_tokens = [str(cell["token"]) for cell in formula["target_cells"]]
        if target_tokens != [str(token) for token in owner["labels"]]:
            raise AssertionError(f"acceptance labels differ from formula source for {sample_id}")
        trace = trace_by_id[sample_id]
        if target_tokens != [str(token) for token in trace["source"]["target_tokens"]]:
            raise AssertionError(f"acceptance labels differ from layer trace for {sample_id}")
        symbols = trace["oracle_group_hwr"]["symbols"]
        target_tokens_from_trace = [str(symbol["target_label"]) for symbol in symbols]
        if target_tokens != target_tokens_from_trace:
            raise AssertionError(f"oracle-group symbol order differs from acceptance for {sample_id}")
        fast_group_exact = bool(context_by_id[sample_id]["fast_group_exact"])
        tournament_fast = tournament_by_id[sample_id]["hwr_tournament"]["fast"]
        if (
            fast_group_exact != bool(tournament_fast["group_exact"])
            or fast_group_exact != bool(partition_by_id[sample_id]["fast_group_exact"])
        ):
            raise AssertionError(f"group exactness differs between reports for {sample_id}")
        oracle_top1 = all(symbol["prediction"]["top1"] == symbol["target_label"] for symbol in symbols)
        oracle_top5 = all(symbol["target_label"] in symbol["prediction"]["top5"] for symbol in symbols)
        oracle_group_complete_by_fast_group[fast_group_exact].append(oracle_top5)
        token_total += len(symbols)
        top1_correct += sum(symbol["prediction"]["top1"] == symbol["target_label"] for symbol in symbols)
        top5_correct += sum(symbol["target_label"] in symbol["prediction"]["top5"] for symbol in symbols)
        top1_formula_exact += oracle_top1
        top5_formula_complete += oracle_top5

        case = context_by_id[sample_id]
        stages = case["stage_scores"]
        stage_rows.append({
            "sample_id": sample_id,
            "writer_id": str(owner["writer_id"]),
            "target_tokens": target_tokens,
            "fast_group_exact": fast_group_exact,
            "oracle_hwr_top5_complete": oracle_top5,
            "fast_selected_exact": bool(stages["fast_selected"]["formula_exact"]),
            "guarded_exact": bool(stages[BASELINE_STAGE]["formula_exact"]),
            "context_exact": bool(stages[CONTEXT_STAGE]["formula_exact"]),
            "context_changes": int(case["context_changes"]),
            "candidate_violations": int(stages[CONTEXT_STAGE]["candidate_violations"]),
            "grouping_outcome": str(partition_by_id[sample_id]["outcome"]),
            "partition_error_shape": str(partition_by_id[sample_id]["partition_error_shape"]),
            "decoder_residual_cause": (
                str(decoder_residuals[sample_id]["cause"]) if sample_id in decoder_residuals else None
            ),
        })

    group_errors = [row for row in stage_rows if not row["fast_group_exact"]]
    correct_group_rows = [row for row in stage_rows if row["fast_group_exact"]]
    final_errors = [row for row in stage_rows if not row["context_exact"]]
    group_outcomes = Counter(row["grouping_outcome"] for row in group_errors)
    group_error_shapes = Counter(row["partition_error_shape"] for row in group_errors)
    decoder_causes = Counter(row["decoder_residual_cause"] for row in stage_rows if row["decoder_residual_cause"])

    reachable_deltas = []
    reachable_rows = []
    for sample_id in sorted(accepted_ids):
        audit_row = partition_by_id[sample_id]
        if audit_row["outcome"] != "gold_partition_reachable_but_not_promoted":
            continue
        replay = audit_row.get("local_partition_search_replay") or {}
        delta = replay.get("target_partition_score_delta_vs_fast_incumbent")
        if delta is None:
            raise AssertionError(f"reachable target lacks score-gap replay for {sample_id}")
        reachable_deltas.append(float(delta))
        reachable_rows.append({
            "sample_id": sample_id,
            "target_rank_in_fast_top32": audit_row["target_partition_rank_in_fast_top32"],
            "target_score_delta_vs_fast": float(delta),
            "target_excluded_by_candidate_cap": bool(
                replay.get("target_partition_any_edge_excluded_by_candidate_cap", False)
            ),
        })

    baseline_exact = sum(row["guarded_exact"] for row in stage_rows)
    context_exact = sum(row["context_exact"] for row in stage_rows)
    group_correct_context_exact = sum(row["context_exact"] for row in correct_group_rows)
    group_wrong_context_exact = context_exact - group_correct_context_exact
    if len(final_errors) != len(group_errors) - (len(group_errors) - sum(not row["context_exact"] for row in group_errors)) + sum(
        not row["context_exact"] for row in correct_group_rows
    ):
        raise AssertionError("failure factorization did not reconcile")

    checks = {
        "acceptance_formula_count_matches_freeze": len(accepted_ids) == int(provenance["policy"]["frozen_acceptance_subset"]["formulas"]),
        "accepted_targets_match_raw_and_layer_trace": len(stage_rows) == len(accepted_ids) == 53,
        "all_diagnostics_share_exact_trace_hash": trace_sha == decoder_trace_sha == partition_trace_sha,
        "fast_group_outcomes_match_across_reports": len(stage_rows) == len(accepted_ids),
        "grouping_failure_buckets_reconcile": sum(group_outcomes.values()) == len(group_errors),
        "decoder_failure_buckets_reconcile": sum(decoder_causes.values()) == len(decoder_residuals),
        "context_candidate_violations_zero": sum(row["candidate_violations"] for row in stage_rows) == 0,
        "partition_cap_did_not_exclude_reachable_targets": all(not row["target_excluded_by_candidate_cap"] for row in reachable_rows),
        "all_reachable_target_deltas_below_zero": all(delta < 0 for delta in reachable_deltas),
        "final_formula_failure_factorization_reconciles": sum(not row["context_exact"] for row in correct_group_rows)
        + sum(not row["context_exact"] for row in group_errors)
        == len(final_errors),
        "partition_ranker_hash_matches_all_grouping_audits": True,
    }

    return {
        "schema": SCHEMA,
        "scope": "post-hoc failure microscope on the frozen acceptance subset; no fitting, CROHME use, threshold selection, or promotion",
        "inputs": {
            "provenance_audit": {"path": str(provenance_path), "sha256": _sha256(provenance_path)},
            "acceptance_manifest": provenance["inputs"]["frozen_acceptance_manifest"],
            "acceptance_ownership": provenance["inputs"]["frozen_acceptance_ownership"],
            "formula_source": provenance["inputs"]["formula_source"],
            "trace_summary": provenance["inputs"]["trace_summary"],
            "trace_file": {"path": str(traces_path), "sha256": trace_sha},
            "grouping_tournament": {"path": str(grouping_tournament_path), "sha256": _sha256(grouping_tournament_path)},
            "partition_failure_audit": {"path": str(partition_failure_path), "sha256": _sha256(partition_failure_path)},
            "decoder_failure_matrix": {"path": str(failure_matrix_path), "sha256": _sha256(failure_matrix_path)},
            "partition_ranker": {"path": str(partition_ranker_path), "sha256": ranker_sha},
            "hwr_checkpoint_sha256": expected_trace_hwr_sha,
        },
        "summary": {
            "formulas": len(stage_rows),
            "writers": len({row["writer_id"] for row in stage_rows}),
            "target_symbols": token_total,
            "oracle_group_hwr": {
                "top1_correct_symbols": top1_correct,
                "top1_symbol_accuracy": top1_correct / token_total,
                "top5_correct_symbols": top5_correct,
                "top5_symbol_recall": top5_correct / token_total,
                "top1_formula_exact": top1_formula_exact,
                "top5_formula_complete": top5_formula_complete,
                "top5_complete_by_fast_group_status": {
                    str(group_ok): {
                        "formulas": len(values),
                        "complete": sum(values),
                        "incomplete": len(values) - sum(values),
                    }
                    for group_ok, values in oracle_group_complete_by_fast_group.items()
                },
            },
            "strict_formula_exact": {
                "fast_selected": sum(row["fast_selected_exact"] for row in stage_rows),
                "after_guards": baseline_exact,
                "after_context": context_exact,
                "context_net_gain": context_exact - baseline_exact,
                "context_rescues": sum(not row["guarded_exact"] and row["context_exact"] for row in stage_rows),
                "context_regressions": sum(row["guarded_exact"] and not row["context_exact"] for row in stage_rows),
                "by_fast_group_status": {
                    "group_exact": {
                        "formulas": len(correct_group_rows),
                        "context_exact": group_correct_context_exact,
                        "context_errors": len(correct_group_rows) - group_correct_context_exact,
                    },
                    "group_wrong": {
                        "formulas": len(group_errors),
                        "context_exact": group_wrong_context_exact,
                        "context_errors": len(group_errors) - group_wrong_context_exact,
                    },
                },
            },
            "grouping_failures": {
                "fast_group_exact": len(stage_rows) - len(group_errors),
                "fast_group_errors": len(group_errors),
                "error_shapes": dict(group_error_shapes),
                "root_causes": dict(group_outcomes),
                "gold_reachable_score_gaps": {
                    "count": len(reachable_deltas),
                    "min": min(reachable_deltas) if reachable_deltas else None,
                    "median": median(reachable_deltas) if reachable_deltas else None,
                    "max": max(reachable_deltas) if reachable_deltas else None,
                    "candidate_cap_exclusions": sum(row["target_excluded_by_candidate_cap"] for row in reachable_rows),
                    "formula_details": reachable_rows,
                },
                "failed_formula_details": [row for row in stage_rows if not row["fast_group_exact"]],
            },
            "oracle_group_decoder_residuals": {
                "formulas": len(decoder_residuals),
                "causes": dict(decoder_causes),
                "interpretation": "residual causes use oracle target groups and the saved guarded decoder; they isolate decoder/HWR from grouping",
            },
            "partition_ranker_governance": {
                "model_version": partition_ranker.get("model_version"),
                "training_scope": partition_ranker.get("grouping_training_scope"),
                "posthoc_test_tuning": bool(partition_ranker.get("posthoc_test_tuning")),
                "evaluation_only_writer_loo": bool(partition_ranker.get("evaluation_only_writer_loo")),
                "held_writer_absent_from_candidate_head_fit": partition_ranker.get("candidate_policy", {}).get("held_writer_absent_from_candidate_head_fit"),
                "product_default_enabled": bool(partition_ranker.get("product_default_enabled")),
                "requires_explicit_shadow_opt_in": bool(partition_ranker.get("requires_explicit_shadow_opt_in")),
            },
            "selection_policy": {
                "original_freeze_excluded_acceptance_specific_training_and_pre_freeze_predictions": (
                    not bool(_json(Path(provenance["inputs"]["frozen_acceptance_manifest"]["path"]))["training_performed"])
                    and not bool(_json(Path(provenance["inputs"]["frozen_acceptance_manifest"]["path"]))["model_predictions_opened_before_freeze"])
                ),
                "post_freeze_development_reuse": True,
                "independent_acceptance_claim_permitted_now": False,
                "reason": "the frozen 53 cases have been included in the later 149-case development microscope; use this only to diagnose, not tune or promote",
            },
            "formula_level_details": stage_rows,
        },
        "verification": {**checks, "all_checks_pass": all(checks.values())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provenance-audit", type=Path, required=True)
    parser.add_argument("--grouping-tournament", type=Path, required=True)
    parser.add_argument("--partition-failure-audit", type=Path, required=True)
    parser.add_argument("--decoder-failure-matrix", type=Path, required=True)
    parser.add_argument("--partition-ranker", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite output: {args.output}")
    report = audit(
        args.provenance_audit,
        args.grouping_tournament,
        args.partition_failure_audit,
        args.decoder_failure_matrix,
        args.partition_ranker,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "summary": report["summary"], "verification": report["verification"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
