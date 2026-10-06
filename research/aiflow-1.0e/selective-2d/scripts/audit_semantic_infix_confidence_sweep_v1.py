#!/usr/bin/env python3
"""Microscope same-K sensitivity audit for semantic infix confidence gating."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from audit_widened_semantic_guard_shadow_v1 import _guard_formula, _jsonl


SCHEMA = "aiflow-hwr-semantic-infix-confidence-sweep/v1"
GRID = (0.0, 0.05, 0.10, 0.12, 0.20, 0.30, 0.45, 0.60)
DOMINANCE_GRID = (0.0001, 0.0003, 0.001, 0.002, 0.01, 0.1, 1.0)
INFIX_OPERATORS = {"+", "-", "/", r"\times", r"\div", r"\cdot"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _compare_rows(
    traces: list[dict[str, Any]],
    baseline: list[dict[str, Any]],
    challenger: list[dict[str, Any]],
    *,
    stage: str = "after_expression",
) -> dict[str, Any]:
    target_by_id = {
        str(trace["sample_id"]): [str(token) for token in trace["source"]["target_tokens"]]
        for trace in traces
    }
    baseline_by_id = {str(row["sample_id"]): row for row in baseline}
    challenger_by_id = {str(row["sample_id"]): row for row in challenger}
    if set(baseline_by_id) != set(challenger_by_id) or set(baseline_by_id) != set(target_by_id):
        raise AssertionError("formula ID coverage differs across confidence-sweep arms")

    token_hits = {"baseline": 0, "challenger": 0}
    exact_ids: dict[str, set[str]] = {"baseline": set(), "challenger": set()}
    recovered_tokens: list[dict[str, Any]] = []
    regressed_tokens: list[dict[str, Any]] = []
    wrong_to_wrong: list[dict[str, Any]] = []
    for sample_id, targets in target_by_id.items():
        before = baseline_by_id[sample_id]["pipeline_stages"][stage]["tokens"]
        after = challenger_by_id[sample_id]["pipeline_stages"][stage]["tokens"]
        if len(before) != len(targets) or len(after) != len(targets):
            raise AssertionError(f"token length mismatch: {sample_id}")
        token_hits["baseline"] += sum(a == b for a, b in zip(before, targets, strict=True))
        token_hits["challenger"] += sum(a == b for a, b in zip(after, targets, strict=True))
        if before == targets:
            exact_ids["baseline"].add(sample_id)
        if after == targets:
            exact_ids["challenger"].add(sample_id)
        for position, (old, new, target) in enumerate(zip(before, after, targets, strict=True)):
            if old == new:
                continue
            item = {
                "sample_id": sample_id,
                "position": position,
                "target": target,
                "baseline": old,
                "challenger": new,
            }
            if old != target and new == target:
                recovered_tokens.append(item)
            elif old == target and new != target:
                regressed_tokens.append(item)
            elif old != target and new != target:
                wrong_to_wrong.append(item)

    base_exact = exact_ids["baseline"]
    next_exact = exact_ids["challenger"]
    return {
        "stage": stage,
        "baseline": {
            "formula_exact": len(base_exact),
            "token_hits": token_hits["baseline"],
        },
        "challenger": {
            "formula_exact": len(next_exact),
            "token_hits": token_hits["challenger"],
        },
        "delta": {
            "formula_exact": len(next_exact) - len(base_exact),
            "recovered_formula_ids": sorted(next_exact - base_exact),
            "regressed_formula_ids": sorted(base_exact - next_exact),
            "token_hits": token_hits["challenger"] - token_hits["baseline"],
            "recovered_tokens": recovered_tokens,
            "regressed_tokens": regressed_tokens,
            "wrong_to_wrong_changes": wrong_to_wrong,
        },
    }


def _operator_candidates(trace: dict[str, Any], position: int) -> list[dict[str, Any]]:
    prediction = trace["oracle_group_hwr"]["symbols"][position]["prediction"]
    diagnostic = prediction["diagnostic_topk"]
    result = []
    for rank, (token, probability) in enumerate(
        zip(diagnostic["tokens"][:20], diagnostic["probabilities"][:20], strict=True),
        start=1,
    ):
        if str(token) in INFIX_OPERATORS:
            result.append({"rank": rank, "token": str(token), "probability": float(probability)})
    return result


def _collect_infix_changes(
    traces: list[dict[str, Any]], rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    trace_by_id = {str(trace["sample_id"]): trace for trace in traces}
    changes = []
    for row in rows:
        sample_id = str(row["sample_id"])
        for change in row["pipeline_guard_audits"]["infix"].get("changes", []):
            position = int(change["context_index"])
            trace = trace_by_id[sample_id]
            changes.append({
                "sample_id": sample_id,
                **change,
                "target": str(trace["source"]["target_tokens"][position]),
                "operator_candidates_top20": _operator_candidates(trace, position),
            })
    return changes


def _policy_sweep_row(
    traces: list[dict[str, Any]],
    baseline: list[dict[str, Any]],
    top5: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    *,
    threshold_name: str,
    threshold: float,
) -> dict[str, Any]:
    return {
        threshold_name: threshold,
        "vs_top20_unique_policy": _compare_rows(traces, baseline, rows),
        "infix_stage_vs_top20_unique_policy": _compare_rows(
            traces, baseline, rows, stage="after_infix",
        ),
        "vs_top5_unique_policy": _compare_rows(traces, top5, rows),
        "gate_audit_totals": {
            name: sum(
                int(row["pipeline_guard_audits"]["infix"].get(name, 0))
                for row in rows
            )
            for name in (
                "resolved_ambiguous_operator", "resolved_dominant_operator",
                "numeric_operator_role_contexts", "numeric_operator_role_changes",
                "skipped_probability_floor", "skipped_ambiguous_operator",
            )
        },
        "infix_changes": _collect_infix_changes(traces, rows),
    }


def _writer_stratified_delta(
    traces: list[dict[str, Any]],
    baseline: list[dict[str, Any]],
    challenger: list[dict[str, Any]],
    writer_by_id: dict[str, str],
) -> list[dict[str, Any]]:
    base_by_id = {str(row["sample_id"]): row for row in baseline}
    challenger_by_id = {str(row["sample_id"]): row for row in challenger}
    target_by_id = {
        str(trace["sample_id"]): [str(token) for token in trace["source"]["target_tokens"]]
        for trace in traces
    }
    writer_ids = sorted(set(writer_by_id.values()))
    output = []
    for writer_id in writer_ids:
        sample_ids = sorted(sample_id for sample_id, writer in writer_by_id.items() if writer == writer_id)
        base_exact_ids: set[str] = set()
        challenger_exact_ids: set[str] = set()
        base_hits = 0
        challenger_hits = 0
        for sample_id in sample_ids:
            target = target_by_id[sample_id]
            before = base_by_id[sample_id]["pipeline_stages"]["after_expression"]["tokens"]
            after = challenger_by_id[sample_id]["pipeline_stages"]["after_expression"]["tokens"]
            base_hits += sum(a == b for a, b in zip(before, target, strict=True))
            challenger_hits += sum(a == b for a, b in zip(after, target, strict=True))
            if before == target:
                base_exact_ids.add(sample_id)
            if after == target:
                challenger_exact_ids.add(sample_id)
        output.append({
            "writer_id": writer_id,
            "formulas": len(sample_ids),
            "baseline_exact": len(base_exact_ids),
            "challenger_exact": len(challenger_exact_ids),
            "exact_delta": len(challenger_exact_ids) - len(base_exact_ids),
            "recovered_formula_ids": sorted(challenger_exact_ids - base_exact_ids),
            "regressed_formula_ids": sorted(base_exact_ids - challenger_exact_ids),
            "baseline_token_hits": base_hits,
            "challenger_token_hits": challenger_hits,
            "token_hit_delta": challenger_hits - base_hits,
        })
    return output


def _writer_cluster_bootstrap(
    writer_rows: list[dict[str, Any]], *, iterations: int = 10_000, seed: int = 20261001,
) -> dict[str, Any]:
    rng = random.Random(seed)
    deltas = []
    for _ in range(iterations):
        sampled = [rng.choice(writer_rows) for _ in writer_rows]
        denominator = sum(int(row["formulas"]) for row in sampled)
        baseline_exact = sum(int(row["baseline_exact"]) for row in sampled)
        challenger_exact = sum(int(row["challenger_exact"]) for row in sampled)
        deltas.append(100.0 * (challenger_exact - baseline_exact) / denominator)
    ordered = sorted(deltas)
    low = ordered[int(0.025 * (iterations - 1))]
    high = ordered[int(0.975 * (iterations - 1))]
    return {
        "method": "paired writer-cluster bootstrap; writers sampled with replacement",
        "iterations": iterations,
        "seed": seed,
        "delta_exact_rate_pp_95_interval": [low, high],
        "interpretation": "descriptive only; this consumed cohort was already used to inspect the challenger",
    }


def audit(
    trace_path: Path,
    reference_path: Path,
    ownership_path: Path,
    writer_reference_path: Path,
) -> dict[str, Any]:
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    if reference.get("schema") != "aiflow-hwr-widened-semantic-guard-shadow/v1":
        raise ValueError("unexpected widened-shadow reference schema")
    writer_reference = json.loads(writer_reference_path.read_text(encoding="utf-8"))
    if writer_reference.get("schema") != "aiflow-semantic-infix-writer-stratified-shadow/v1":
        raise ValueError("unexpected writer-stratified reference schema")
    traces = _jsonl(trace_path)
    actual_trace_sha = _sha256(trace_path)
    expected_trace_sha = reference["inputs"]["top20_trace"]["sha256"]
    if actual_trace_sha != expected_trace_sha:
        raise AssertionError("Top-20 trace hash differs from the frozen shadow report")
    actual_ownership_sha = _sha256(ownership_path)
    expected_ownership_sha = writer_reference["inputs"]["ownership_train"]["sha256"]
    if actual_ownership_sha != expected_ownership_sha:
        raise AssertionError("ownership source hash differs from the writer-stratified reference")
    ownership_rows = _jsonl(ownership_path)
    writer_by_id = {
        str(row["sample_id"]): str(row["writer_id"])
        for row in ownership_rows
    }
    trace_ids = {str(trace["sample_id"]) for trace in traces}
    if set(writer_by_id) != trace_ids or len(writer_by_id) != 149:
        raise AssertionError("frozen ownership IDs do not exactly cover the trace formulas")

    k5 = [_guard_formula(trace, 5) for trace in traces]
    k20_unique = [_guard_formula(trace, 20) for trace in traces]
    arms: dict[str, list[dict[str, Any]]] = {}
    reports: list[dict[str, Any]] = []
    for threshold in GRID:
        arm_name = f"ratio_{threshold:.2f}"
        rows = [
            _guard_formula(
                trace, 20, infix_policy="operator_argmax",
                minimum_infix_probability_ratio=threshold,
            )
            for trace in traces
        ]
        arms[arm_name] = rows
        reports.append(_policy_sweep_row(
            traces, k20_unique, k5, rows,
            threshold_name="minimum_probability_ratio",
            threshold=threshold,
        ))

    dominance_reports = []
    dominance_arms: dict[str, list[dict[str, Any]]] = {}
    for threshold in DOMINANCE_GRID:
        arm_name = f"dominance_{threshold:g}"
        rows = [
            _guard_formula(
                trace, 20, infix_policy="operator_relative",
                maximum_operator_competitor_probability_ratio=threshold,
            )
            for trace in traces
        ]
        dominance_arms[arm_name] = rows
        sweep_row = _policy_sweep_row(
            traces, k20_unique, k5, rows,
            threshold_name="maximum_competitor_probability_ratio",
            threshold=threshold,
        )
        sweep_row["writer_stratified_vs_top20_unique"] = _writer_stratified_delta(
            traces, k20_unique, rows, writer_by_id,
        )
        sweep_row["writer_cluster_bootstrap_vs_top20_unique"] = _writer_cluster_bootstrap(
            sweep_row["writer_stratified_vs_top20_unique"],
        )
        sweep_row["writer_stratified_vs_top5_unique"] = _writer_stratified_delta(
            traces, k5, rows, writer_by_id,
        )
        sweep_row["writer_cluster_bootstrap_vs_top5_unique"] = _writer_cluster_bootstrap(
            sweep_row["writer_stratified_vs_top5_unique"],
        )
        dominance_reports.append(sweep_row)

    zero_arm = arms["ratio_0.00"]
    relative_dominance_002 = dominance_arms["dominance_0.002"]
    numeric_context_002 = [
        _guard_formula(
            trace, 20, infix_policy="operator_numeric_context",
            maximum_operator_competitor_probability_ratio=0.002,
        )
        for trace in traces
    ]
    numeric_context_report = _policy_sweep_row(
        traces, k20_unique, k5, numeric_context_002,
        threshold_name="maximum_competitor_probability_ratio",
        threshold=0.002,
    )
    numeric_context_report["vs_top20_relative_dominance_0.002"] = _compare_rows(
        traces, relative_dominance_002, numeric_context_002,
    )
    numeric_context_report["writer_stratified_vs_top20_unique"] = _writer_stratified_delta(
        traces, k20_unique, numeric_context_002, writer_by_id,
    )
    numeric_context_report["writer_cluster_bootstrap_vs_top20_unique"] = _writer_cluster_bootstrap(
        numeric_context_report["writer_stratified_vs_top20_unique"],
    )
    numeric_context_report["writer_stratified_vs_top20_relative_dominance_0.002"] = _writer_stratified_delta(
        traces, relative_dominance_002, numeric_context_002, writer_by_id,
    )
    numeric_context_report["writer_cluster_bootstrap_vs_top20_relative_dominance_0.002"] = _writer_cluster_bootstrap(
        numeric_context_report["writer_stratified_vs_top20_relative_dominance_0.002"],
    )
    reference_rows = reference["pipeline_formula_level_by_k"]["top20_operator_argmax_shadow"]
    reference_by_id = {str(row["sample_id"]): row for row in reference_rows}
    zero_matches_reference = all(
        row["pipeline_stages"]["after_expression"]["tokens"]
        == reference_by_id[str(row["sample_id"])]["stages"]["after_expression"]["tokens"]
        for row in zero_arm
    )
    if not zero_matches_reference:
        raise AssertionError("ratio=0 replay differs from the saved Top-20 argmax arm")
    if len(traces) != 149:
        raise AssertionError(f"expected the frozen 149-formula cohort, got {len(traces)}")

    report = {
        "schema": SCHEMA,
        "scope": "posthoc consumed-development sensitivity; Top-20 held fixed; no training, CROHME, or default-policy promotion",
        "inputs": {
            "top20_trace_sha256": actual_trace_sha,
            "reference_report_sha256": _sha256(reference_path),
            "writer_reference_report_sha256": _sha256(writer_reference_path),
            "ownership_sha256": actual_ownership_sha,
            "writer_count": len(set(writer_by_id.values())),
            "formula_count": len(traces),
            "threshold_grid": list(GRID),
            "operator_dominance_threshold_grid": list(DOMINANCE_GRID),
        },
        "interpretation": {
            "causal_control": "Top-20 unique-policy comparison isolates ambiguity-policy changes at fixed K=20",
            "promotion": "not eligible; cohort already consumed for development and threshold grid is exploratory",
        },
        "reference_comparisons": {
            "top20_unique_vs_top5_unique": _compare_rows(traces, k5, k20_unique),
            "top20_argmax_ratio_zero_vs_top20_unique": reports[0]["vs_top20_unique_policy"],
            "top20_argmax_ratio_zero_vs_top5_unique": reports[0]["vs_top5_unique_policy"],
            "top20_relative_dominance_sweeps": {
                str(row["maximum_competitor_probability_ratio"]): row["vs_top20_unique_policy"]
                for row in dominance_reports
            },
            "relative_dominance_0.002_vs_top20_argmax_0.0": _compare_rows(
                traces, zero_arm, relative_dominance_002,
            ),
            "numeric_context_dominance_0.002_vs_top20_relative_dominance_0.002": (
                numeric_context_report["vs_top20_relative_dominance_0.002"]
            ),
        },
        "sweep": reports,
        "relative_dominance_sweep": dominance_reports,
        "numeric_context_dominance_0.002": numeric_context_report,
        "verification": {
            "trace_hash_matches_reference": actual_trace_sha == expected_trace_sha,
            "ownership_hash_matches_writer_reference": actual_ownership_sha == expected_ownership_sha,
            "ownership_ids_exactly_match_trace": set(writer_by_id) == trace_ids,
            "nine_writer_groups_reconciled": len(set(writer_by_id.values())) == 9,
            "ratio_zero_replays_saved_top20_argmax_arm": zero_matches_reference,
            "relative_dominance_002_matches_top20_argmax_outputs": all(
                relative_dominance_002[index]["pipeline_stages"]["after_infix"]["tokens"]
                == zero_arm[index]["pipeline_stages"]["after_infix"]["tokens"]
                and relative_dominance_002[index]["pipeline_stages"]["after_expression"]["tokens"]
                == zero_arm[index]["pipeline_stages"]["after_expression"]["tokens"]
                for index in range(len(zero_arm))
            ),
            "frozen_formula_count_is_149": len(traces) == 149,
            "all_arms_preserve_candidates": all(
                all(row["candidate_preservation"] for row in arm)
                for arm in [*arms.values(), *dominance_arms.values(), numeric_context_002]
            ),
            "all_formula_lengths_reconcile": all(
                len(row["pipeline_stages"]["after_expression"]["tokens"])
                == len(trace["source"]["target_tokens"])
                for arm in [*arms.values(), numeric_context_002]
                for row in arm
                for trace in traces
                if str(row["sample_id"]) == str(trace["sample_id"])
            ) and all(
                len(row["pipeline_stages"]["after_expression"]["tokens"])
                == len(trace["source"]["target_tokens"])
                for arm in dominance_arms.values()
                for row in arm
                for trace in traces
                if str(row["sample_id"]) == str(trace["sample_id"])
            ),
        },
    }
    report["verification"]["all_checks_pass"] = all(report["verification"].values())
    if not report["verification"]["all_checks_pass"]:
        raise AssertionError("confidence-sweep verification failed")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--ownership", type=Path, required=True)
    parser.add_argument("--writer-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.trace, args.reference_report, args.ownership, args.writer_reference)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "all_checks_pass": report["verification"]["all_checks_pass"],
        "thresholds": len(report["sweep"]),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
