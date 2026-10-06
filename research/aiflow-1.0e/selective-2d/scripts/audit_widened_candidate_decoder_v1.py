#!/usr/bin/env python3
"""Posthoc shadow audit of wider per-symbol HWR candidate lists."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from selective_decoder_v1 import DEFAULT_TOKEN_BEAM, _strict_latex


SCHEMA = "aiflow-hwr-widened-candidate-shadow-audit/v1"


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _decode(
    formula: dict[str, Any], candidate_k: int, beam_width: int,
) -> dict[str, Any]:
    target_tokens = [str(token) for token in formula["source"]["target_tokens"]]
    symbols = formula["oracle_group_hwr"]["symbols"]
    if len(symbols) != len(target_tokens):
        raise AssertionError(f"oracle symbol count mismatch: {formula['sample_id']}")

    rows: list[dict[str, Any]] = []
    options: list[list[tuple[float, str]]] = []
    target_ranks: list[int | None] = []
    for ordinal, symbol in enumerate(symbols):
        prediction = symbol["prediction"]
        widened = prediction.get("diagnostic_topk")
        if not widened or len(widened["tokens"]) < candidate_k:
            raise ValueError(f"missing diagnostic top-{candidate_k}: {formula['sample_id']}:{ordinal}")
        tokens = [str(token) for token in widened["tokens"][:candidate_k]]
        probabilities = [float(value) for value in widened["probabilities"][:candidate_k]]
        if len(set(tokens)) != candidate_k or len(probabilities) != candidate_k:
            raise AssertionError(f"invalid candidate rows: {formula['sample_id']}:{ordinal}")
        if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in probabilities):
            raise AssertionError(f"invalid probability: {formula['sample_id']}:{ordinal}")
        if any(a < b for a, b in zip(probabilities, probabilities[1:])):
            raise AssertionError(f"candidate probabilities are not rank-sorted: {formula['sample_id']}:{ordinal}")
        target_ranks.append(prediction.get("target_rank"))
        geometry = symbol["preprocessing"]["raw_bbox"]
        record_id = f"{formula['sample_id']}:{ordinal}"
        rows.append({
            "record_id": record_id,
            "formula_id": formula["sample_id"],
            "final_topk": tokens,
            "final_topk_probabilities": probabilities,
            "geometry": geometry,
        })
        options.append([
            (math.log(max(1e-8, probability)), token)
            for token, probability in zip(tokens, probabilities, strict=True)
        ])

    beams: list[tuple[float, tuple[str, ...]]] = [(0.0, ())]
    branch_evaluations = 0
    for choices in options:
        branch_evaluations += len(beams) * len(choices)
        beams = sorted(
            (
                (score + log_probability, tokens + (token,))
                for score, tokens in beams
                for log_probability, token in choices
            ),
            key=lambda row: (-row[0], row[1]),
        )[:beam_width]

    best: tuple[float, tuple[str, ...], str, list[dict[str, Any]], float] | None = None
    valid_candidates = 0
    for token_score, tokens in beams:
        predictions = {
            str(row["record_id"]): token
            for row, token in zip(rows, tokens, strict=True)
        }
        try:
            latex, relations, relation_score = _strict_latex(rows, predictions)
        except ValueError:
            continue
        valid_candidates += 1
        score = token_score / len(rows) + relation_score
        if best is None or score > best[0]:
            best = (score, tokens, latex, relations, relation_score)

    candidate_complete = all(rank is not None and rank <= candidate_k for rank in target_ranks)
    if best is None:
        return {
            "accepted": False,
            "tokens": None,
            "latex": None,
            "target_tokens": target_tokens,
            "candidate_complete": candidate_complete,
            "target_ranks": target_ranks,
            "valid_beam_candidates": 0,
            "branch_evaluations": branch_evaluations,
        }
    _score, selected, latex, relations, relation_score = best
    return {
        "accepted": True,
        "tokens": list(selected),
        "latex": latex,
        "target_tokens": target_tokens,
        "candidate_complete": candidate_complete,
        "target_ranks": target_ranks,
        "valid_beam_candidates": valid_candidates,
        "relation_count": len(relations),
        "relation_score": relation_score,
        "branch_evaluations": branch_evaluations,
    }


def audit(
    traces_path: Path,
    baseline_traces_path: Path,
    candidate_ks: list[int],
    beam_width: int,
) -> dict[str, Any]:
    traces = _jsonl(traces_path)
    baseline_traces = _jsonl(baseline_traces_path)
    trace_by_id = {row["sample_id"]: row for row in traces}
    baseline_by_id = {row["sample_id"]: row for row in baseline_traces}
    if set(trace_by_id) != set(baseline_by_id):
        raise AssertionError("widened and baseline traces contain different formula IDs")

    maximum_k = max(candidate_ks)
    baseline_candidate_parity = 0
    for sample_id, trace in trace_by_id.items():
        baseline = baseline_by_id[sample_id]
        if trace["source"]["target_tokens"] != baseline["source"]["target_tokens"]:
            raise AssertionError(f"target token mismatch: {sample_id}")
        symbols = trace["oracle_group_hwr"]["symbols"]
        baseline_symbols = baseline["oracle_group_hwr"]["symbols"]
        old_by_group = {tuple(row["stroke_indices"]): row for row in baseline_symbols}
        for symbol in symbols:
            old = old_by_group[tuple(symbol["stroke_indices"])]
            current_prediction = symbol["prediction"]
            old_prediction = old["prediction"]
            diagnostic = current_prediction.get("diagnostic_topk") or {}
            if len(diagnostic.get("tokens", [])) != maximum_k:
                raise AssertionError(f"diagnostic Top-K width mismatch: {sample_id}")
            if current_prediction["top5"] != old_prediction["top5"]:
                raise AssertionError(f"Top-5 replay changed: {sample_id}:{symbol['stroke_indices']}")
            if current_prediction["top5_probabilities"] != old_prediction["top5_probabilities"]:
                raise AssertionError(f"Top-5 probabilities changed: {sample_id}:{symbol['stroke_indices']}")
            baseline_candidate_parity += 1

    results_by_k: dict[str, dict[str, Any]] = {}
    details_by_k: dict[str, list[dict[str, Any]]] = {}
    for candidate_k in candidate_ks:
        rows = []
        for formula in traces:
            result = _decode(formula, candidate_k, beam_width)
            result["sample_id"] = formula["sample_id"]
            rows.append(result)
        results_by_k[str(candidate_k)] = rows
        details_by_k[str(candidate_k)] = rows

    baseline_rows = results_by_k[str(candidate_ks[0])]
    saved_baseline_matches = 0
    for row in baseline_rows:
        saved = trace_by_id[row["sample_id"]]["oracle_group_hwr"]["decoder"]
        if bool(saved["accepted"]) == bool(row["accepted"]):
            same = (
                (not row["accepted"])
                or (
                    saved["tokens"] == row["tokens"]
                    and saved["latex"] == row["latex"]
                )
            )
            saved_baseline_matches += int(same)

    baseline_exact_ids = {
        row["sample_id"] for row in baseline_rows
        if row["accepted"] and row["tokens"] == row["target_tokens"]
    }
    metrics: dict[str, Any] = {}
    for candidate_k in candidate_ks:
        rows = results_by_k[str(candidate_k)]
        exact_ids = {
            row["sample_id"] for row in rows
            if row["accepted"] and row["tokens"] == row["target_tokens"]
        }
        rank_histogram = Counter(
            str(rank) if rank is not None else "oov"
            for row in rows for rank in row["target_ranks"]
        )
        metrics[str(candidate_k)] = {
            "candidate_complete_formulas": sum(row["candidate_complete"] for row in rows),
            "strict_ast_accepted_formulas": sum(row["accepted"] for row in rows),
            "strict_formula_exact": len(exact_ids),
            "recovered_vs_top5": sorted(exact_ids - baseline_exact_ids),
            "regressed_vs_top5": sorted(baseline_exact_ids - exact_ids),
            "changed_selection_vs_top5": sum(
                row["tokens"] != baseline_rows[index]["tokens"]
                for index, row in enumerate(rows)
            ),
            "branch_evaluations_total": sum(row["branch_evaluations"] for row in rows),
            "mean_branch_evaluations_per_formula": (
                sum(row["branch_evaluations"] for row in rows) / len(rows) if rows else None
            ),
            "target_rank_histogram": dict(sorted(
                rank_histogram.items(),
                key=lambda item: (item[0] == "oov", int(item[0]) if item[0].isdigit() else 9999),
            )),
        }

    checks = {
        "formula_ID_sets_match": set(trace_by_id) == set(baseline_by_id),
        "source_targets_and_top5_logits_match_baseline": baseline_candidate_parity
        == sum(len(row["source"]["target_tokens"]) for row in traces),
        "k5_shadow_decoder_matches_saved_decoder": saved_baseline_matches == len(traces),
        "candidate_completeness_is_monotonic": all(
            metrics[str(left)]["candidate_complete_formulas"]
            <= metrics[str(right)]["candidate_complete_formulas"]
            for left, right in zip(candidate_ks, candidate_ks[1:])
        ),
        "recovery_and_regression_ids_reconcile": all(
            len(metrics[str(k)]["recovered_vs_top5"])
            == len(set(metrics[str(k)]["recovered_vs_top5"]))
            and len(metrics[str(k)]["regressed_vs_top5"])
            == len(set(metrics[str(k)]["regressed_vs_top5"]))
            for k in candidate_ks
        ),
        "oov_symbol_remains_unreachable": all(
            any(rank is None for row in details_by_k[str(k)] for rank in row["target_ranks"])
            for k in candidate_ks
        ),
    }
    return {
        "schema": SCHEMA,
        "scope": "posthoc consumed-development shadow only; oracle grouping; no training, CROHME, threshold selection, or promotion",
        "inputs": {
            "widened_topk_traces": {"path": str(traces_path), "sha256": _sha256(traces_path)},
            "top5_baseline_traces": {"path": str(baseline_traces_path), "sha256": _sha256(baseline_traces_path)},
        },
        "configuration": {
            "candidate_ks": candidate_ks,
            "beam_width": beam_width,
            "selection_score": "mean token log-probability + existing strict decoder relation score",
            "decoder_ast": "existing selective_decoder_v1._strict_latex",
        },
        "summary": {
            "formulas": len(traces),
            "target_tokens": sum(len(row["source"]["target_tokens"]) for row in traces),
            "baseline_exact_at_k5": metrics[str(candidate_ks[0])]["strict_formula_exact"],
            "candidate_expansion_metrics": metrics,
        },
        "formula_level_by_k": details_by_k,
        "verification": {"checks": checks, "all_checks_pass": all(checks.values())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--baseline-traces", type=Path, required=True)
    parser.add_argument("--candidate-ks", type=int, nargs="+", default=[5, 8, 10, 16, 20])
    parser.add_argument("--beam-width", type=int, default=DEFAULT_TOKEN_BEAM)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite output: {args.output}")
    if not args.candidate_ks or args.candidate_ks[0] != 5:
        parser.error("candidate K values must begin with 5 for parity")
    if any(k < 5 or k > 372 for k in args.candidate_ks):
        parser.error("candidate K values must be within [5, 372]")
    if sorted(set(args.candidate_ks)) != args.candidate_ks:
        parser.error("candidate K values must be unique and strictly increasing")
    if args.beam_width < 1:
        parser.error("beam width must be positive")
    report = audit(args.traces, args.baseline_traces, args.candidate_ks, args.beam_width)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": report["summary"], "verification": report["verification"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
