#!/usr/bin/env python3
"""Replay the current optional semantic-guard runtime on the provenance-filtered cohort."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from raw_formula_context_runtime_v1 import _candidate_preserving_semantic_guard_shadow


SCHEMA = "aiflow-hwr-filtered-runtime-semantic-guard-shadow/v1"
FINAL_STAGE = "after_terminal_rhs_bar_guard"


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


def audit(failure_report_path: Path, traces_path: Path) -> dict[str, Any]:
    failure_report = _read_json(failure_report_path)
    traces = _read_jsonl(traces_path)
    if failure_report.get("schema") != "aiflow-hwr-provenance-filtered-failure-microscope/v1":
        raise ValueError("unexpected provenance-filtered failure report schema")
    if failure_report.get("cohort", {}).get("retained_posthoc_formulas") != 49:
        raise AssertionError("expected the audited posthoc 49-formula cohort")
    expected_trace_hash = failure_report["inputs"]["layer_traces"]["sha256"]
    actual_trace_hash = _sha256(traces_path)
    if actual_trace_hash != expected_trace_hash:
        raise AssertionError("trace hash does not match provenance-filtered failure report")

    trace_by_id = {str(row["sample_id"]): row for row in traces}
    if len(trace_by_id) != len(traces) or len(trace_by_id) != 149:
        raise AssertionError("expected 149 unique source trace formulas")
    retained_ids = [str(value) for value in failure_report["cohort"]["retained_ids"]]
    if len(retained_ids) != 49 or len(set(retained_ids)) != 49 or not set(retained_ids) <= set(trace_by_id):
        raise AssertionError("retained provenance-filtered IDs do not reconcile")

    failure_by_id = {
        str(row["sample_id"]): row
        for row in failure_report.get("failure_rows", [])
    }
    stage_formula_exact: Counter[str] = Counter()
    stage_token_hits: Counter[str] = Counter()
    guard_status: Counter[str] = Counter()
    equation_status: Counter[str] = Counter()
    terminal_bar_status: Counter[str] = Counter()
    formula_rows: list[dict[str, Any]] = []
    candidate_checks = 0
    candidate_violations = 0
    grouping_mutations = 0
    default_mutations = 0

    for sample_id in sorted(retained_ids):
        trace = trace_by_id[sample_id]
        source = trace["source"]
        target_groups = [
            [int(index) for index in group]
            for group in source["target_grouping"]
        ]
        target_tokens = [str(token) for token in source["target_tokens"]]
        symbols_by_group = {
            frozenset(int(index) for index in symbol["stroke_indices"]): symbol
            for symbol in trace["oracle_group_hwr"]["symbols"]
        }
        if len(symbols_by_group) != len(trace["oracle_group_hwr"]["symbols"]):
            raise AssertionError(f"duplicate oracle group symbol: {sample_id}")
        if set(symbols_by_group) != {frozenset(group) for group in target_groups}:
            raise AssertionError(f"oracle groups do not exactly match target grouping: {sample_id}")
        if len(target_groups) != len(target_tokens):
            raise AssertionError(f"target group/token count mismatch: {sample_id}")

        symbols = []
        baseline_tokens = []
        for group in target_groups:
            symbol = symbols_by_group[frozenset(group)]
            prediction = symbol["prediction"]
            baseline_tokens.append(str(prediction["top1"]))
            symbols.append({
                "stroke_indices": [int(index) for index in symbol["stroke_indices"]],
                "hwr_topk": [str(token) for token in prediction["top5"]],
                "hwr_topk_probabilities": [
                    float(value) for value in prediction["top5_probabilities"]
                ],
                "geometry": dict(symbol["preprocessing"]["raw_bbox"]),
            })
        for token, symbol in zip(baseline_tokens, symbols, strict=True):
            if token not in symbol["hwr_topk"]:
                raise AssertionError(f"saved HWR Top-1 is not in its Top-5: {sample_id}")

        decoder_trace = trace["oracle_group_hwr"].get("decoder") or {}
        decoder = {
            "accepted": bool(decoder_trace.get("accepted")),
            "tokens": [str(token) for token in decoder_trace.get("tokens") or []],
        }
        if decoder["accepted"] and decoder["tokens"] != baseline_tokens:
            raise AssertionError(f"saved oracle decoder differs from HWR Top-1: {sample_id}")
        shadow = _candidate_preserving_semantic_guard_shadow(
            sample_id, target_groups, symbols, decoder,
        )
        guard_status[str(shadow.get("status"))] += 1
        metric: dict[str, Any] = {
            "sample_id": sample_id,
            "fast_grouping_outcome": (
                failure_by_id.get(sample_id, {}).get("grouping", {}).get("outcome")
            ),
            "oracle_decoder_failure_cause": (
                failure_by_id.get(sample_id, {}).get("decoder_with_oracle_groups", {}).get("outcome")
            ),
            "shadow_status": shadow.get("status"),
        }
        if shadow.get("status") != "applied_shadow_only":
            formula_rows.append(metric)
            continue

        stages = shadow["stages"]
        base_tokens = [str(row["token"]) for row in stages["decoder"]]
        final_tokens = [str(row["token"]) for row in stages[FINAL_STAGE]]
        if base_tokens != decoder["tokens"]:
            raise AssertionError(f"semantic shadow mutated the saved decoder: {sample_id}")
        candidate_checks += sum(len(stages[name]) for name in stages)
        candidate_violations += int(not bool(shadow.get("candidate_preservation")))
        grouping_mutations += int(int(shadow.get("grouping_mutations", 0)) != 0)
        default_mutations += int(bool(shadow.get("product_default_enabled")))

        stage_exact: dict[str, bool] = {}
        stage_hits: dict[str, int] = {}
        for stage, rows in stages.items():
            if [frozenset(row["stroke_indices"]) for row in rows] != [
                frozenset(group) for group in target_groups
            ]:
                raise AssertionError(f"group ownership changed at {stage}: {sample_id}")
            tokens = [str(row["token"]) for row in rows]
            if any(
                token not in symbols[index]["hwr_topk"]
                for index, token in enumerate(tokens)
            ):
                raise AssertionError(f"shadow emitted outside HWR Top-5: {sample_id}/{stage}")
            hits = sum(pred == truth for pred, truth in zip(tokens, target_tokens, strict=True))
            exact = tokens == target_tokens
            stage_token_hits[stage] += hits
            stage_formula_exact[stage] += int(exact)
            stage_hits[stage] = hits
            stage_exact[stage] = exact

        audits = shadow["audits"]
        equation_audit = audits["unique_candidate_arithmetic_equation"]
        terminal_audit = audits["terminal_rhs_bar"]
        equation_status[str(equation_audit.get("status"))] += 1
        terminal_bar_status[str(terminal_audit.get("status"))] += 1
        metric.update({
            "target_tokens": target_tokens,
            "baseline_decoder_tokens": base_tokens,
            "final_shadow_tokens": final_tokens,
            "formula_token_exact_by_stage": stage_exact,
            "token_hits_by_stage": stage_hits,
            "unique_candidate_arithmetic_equation": equation_audit,
            "terminal_rhs_bar": terminal_audit,
            "candidate_preservation": shadow["candidate_preservation"],
            "new_tokens": shadow["new_tokens"],
            "deleted_glyphs": shadow["deleted_glyphs"],
            "grouping_mutations": shadow["grouping_mutations"],
            "product_default_enabled": shadow["product_default_enabled"],
        })
        formula_rows.append(metric)

    eligible = [row for row in formula_rows if row.get("formula_token_exact_by_stage") is not None]
    baseline_exact = sum(row["formula_token_exact_by_stage"]["decoder"] for row in eligible)
    final_exact = sum(row["formula_token_exact_by_stage"][FINAL_STAGE] for row in eligible)
    recovered_ids = [
        row["sample_id"] for row in eligible
        if not row["formula_token_exact_by_stage"]["decoder"]
        and row["formula_token_exact_by_stage"][FINAL_STAGE]
    ]
    regressed_ids = [
        row["sample_id"] for row in eligible
        if row["formula_token_exact_by_stage"]["decoder"]
        and not row["formula_token_exact_by_stage"][FINAL_STAGE]
    ]
    oracle_decoder_residual_ids = {
        sample_id for sample_id in retained_ids
        if sample_id in failure_by_id
        and failure_by_id[sample_id]["decoder_with_oracle_groups"]["outcome"]
        != "decoder_arm_exact_with_oracle_groups"
    }
    shadow_recovered_residuals = sorted(oracle_decoder_residual_ids & set(recovered_ids))
    checks = {
        "trace_hash_matches_failure_microscope": actual_trace_hash == expected_trace_hash,
        "retained_ids_reconcile_at_49": len(retained_ids) == 49,
        "all_49_decoders_accepted": guard_status.get("applied_shadow_only", 0) == 49,
        "candidate_violations_zero": candidate_violations == 0,
        "grouping_mutations_zero": grouping_mutations == 0,
        "product_default_never_enabled": default_mutations == 0,
        "no_shadow_token_deletions_or_insertions": all(
            row.get("new_tokens", 0) == 0 and row.get("deleted_glyphs", 0) == 0
            for row in eligible
        ),
        "stage_metrics_cover_each_shadow_formula": all(
            set(row["formula_token_exact_by_stage"]) == set(row["token_hits_by_stage"])
            for row in eligible
        ),
    }
    return {
        "schema": SCHEMA,
        "scope": (
            "shadow-only oracle-group token-sequence replay on 49 previously viewed formulas; "
            "53 acceptance and 47 exact raw-ink duplicates excluded; no training, CROHME, "
            "threshold selection, product activation, or independent accuracy claim"
        ),
        "inputs": {
            "failure_report": {
                "path": str(failure_report_path.resolve()),
                "sha256": _sha256(failure_report_path),
            },
            "traces": {"path": str(traces_path.resolve()), "sha256": actual_trace_hash},
            "runtime_shadow_schema": "aiflow-selective-semantic-guard-shadow/v9",
        },
        "summary": {
            "source_formulas": len(trace_by_id),
            "posthoc_formulas": len(retained_ids),
            "shadow_applied_formulas": len(eligible),
            "baseline_oracle_group_token_exact": baseline_exact,
            "final_shadow_oracle_group_token_exact": final_exact,
            "recovered_formula_ids": sorted(recovered_ids),
            "regressed_formula_ids": sorted(regressed_ids),
            "oracle_decoder_residuals": len(oracle_decoder_residual_ids),
            "oracle_decoder_residuals_recovered_by_shadow": shadow_recovered_residuals,
            "final_stage": FINAL_STAGE,
            "token_exact_by_stage": dict(sorted(stage_formula_exact.items())),
            "token_hits_by_stage": dict(sorted(stage_token_hits.items())),
            "shadow_status": dict(sorted(guard_status.items())),
            "unique_candidate_arithmetic_equation_status": dict(sorted(equation_status.items())),
            "terminal_rhs_bar_status": dict(sorted(terminal_bar_status.items())),
            "candidate_checks": candidate_checks,
        },
        "formula_rows": formula_rows,
        "verification": {"checks": checks, "all_checks_pass": all(checks.values())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failure-report", type=Path, required=True)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite output: {args.output}")
    report = audit(args.failure_report, args.traces)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "summary": report["summary"],
        "verification": report["verification"],
        "output": str(args.output.resolve()),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
