#!/usr/bin/env python3
"""Microscope Top-5 decoder headroom from a saved shadow tournament.

This posthoc diagnostic needs no model load and performs no fitting, tuning,
CROHME access, or product promotion.  It replays the decoder's current
score-only Top-32 token beam to separate per-symbol Top-5 coverage from
formula-level candidate-beam coverage.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any


TOKEN_BEAM = 32  # Must stay in sync with selective_decoder_v1.DEFAULT_TOKEN_BEAM.


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _top_token_beam(options: list[list[tuple[float, str]]]) -> list[tuple[float, tuple[str, ...]]]:
    beams: list[tuple[float, tuple[str, ...]]] = [(0.0, ())]
    for choices in options:
        beams = sorted(
            (
                (score + value, picked + (token,))
                for score, picked in beams
                for value, token in choices
            ),
            key=lambda row: (-row[0], row[1]),
        )[:TOKEN_BEAM]
    return beams


def _analyze_arm(records: list[dict[str, Any]], arm_name: str) -> dict[str, Any]:
    formula_counts = Counter()
    bucket_counts = Counter()
    token_counts = Counter()
    target_rank_histogram = Counter()
    wrong_top1_confusions = Counter()
    wrong_decoder_confusions = Counter()
    beam_rank_histogram = Counter()
    bucket_sample_ids: dict[str, list[str]] = {}
    group_exact_rows = [row for row in records if row["hwr_tournament"][arm_name]["group_exact"]]

    for record in group_exact_rows:
        row = record["hwr_tournament"][arm_name]
        sample_id = str(record["sample_id"])
        groups = [tuple(int(index) for index in group) for group in row["groups"]]
        target_groups = [tuple(int(index) for index in group) for group in row["target_groups"]]
        target_tokens = [str(token) for token in row["target_tokens"]]
        if len(target_groups) != len(target_tokens) or len(groups) != len(target_groups):
            raise AssertionError(f"group/token count mismatch: {sample_id} / {arm_name}")
        truth_by_group = {
            frozenset(group): token
            for group, token in zip(target_groups, target_tokens, strict=True)
        }
        symbols_by_group = {
            frozenset(int(index) for index in symbol["stroke_indices"]): symbol
            for symbol in row["selected_symbols"]
        }
        decoder_tokens = [str(token) for token in row.get("decoder_tokens") or []]
        if len(decoder_tokens) != len(groups):
            raise AssertionError(f"decoder token count mismatch: {sample_id} / {arm_name}")

        gold_sequence = tuple(truth_by_group[frozenset(group)] for group in groups)
        top1_sequence: list[str] = []
        decoder_sequence = tuple(decoder_tokens)
        options: list[list[tuple[float, str]]] = []
        formula_top5_complete = True
        for group, target in zip(groups, gold_sequence, strict=True):
            symbol = symbols_by_group.get(frozenset(group))
            if symbol is None:
                raise AssertionError(f"selected HWR symbol missing: {sample_id} / {group}")
            topk = [str(token) for token in symbol.get("hwr_topk") or []]
            probabilities = [float(value) for value in symbol.get("hwr_topk_probabilities") or []]
            if len(topk) != 5 or len(set(topk)) != 5 or len(probabilities) != 5:
                raise AssertionError(f"invalid Top-5 row: {sample_id} / {arm_name}")
            top1_sequence.append(topk[0])
            formula_top5_complete &= target in topk
            options.append([
                (math.log(max(1e-8, probability)), token)
                for token, probability in zip(topk, probabilities, strict=True)
            ])

        beam = _top_token_beam(options)
        beam_rank = next(
            (rank for rank, (_score, tokens) in enumerate(beam, start=1) if tokens == gold_sequence),
            None,
        )
        decoder_exact = decoder_sequence == gold_sequence
        top1_exact = tuple(top1_sequence) == gold_sequence
        if row.get("token_exact_if_groups_exact") is not None and bool(
            row["token_exact_if_groups_exact"]
        ) != decoder_exact:
            raise AssertionError(f"decoder exactness mismatch: {sample_id} / {arm_name}")
        if row.get("hwr_top5_complete_if_groups_exact") is not None and bool(
            row["hwr_top5_complete_if_groups_exact"]
        ) != formula_top5_complete:
            raise AssertionError(f"Top-5 coverage mismatch: {sample_id} / {arm_name}")
        if decoder_exact:
            bucket = "decoder_exact"
        elif not formula_top5_complete:
            bucket = "gold_missing_from_symbol_top5"
        elif beam_rank is None:
            bucket = "gold_tuple_pruned_before_structural_decoder"
        else:
            bucket = "gold_tuple_in_beam_but_not_selected"
        bucket_counts[bucket] += 1
        formula_counts["top1_formula_exact"] += int(top1_exact)
        formula_counts["top5_complete"] += int(formula_top5_complete)
        formula_counts["decoder_exact"] += int(decoder_exact)
        formula_counts["decoder_equals_independent_top1"] += int(decoder_sequence == tuple(top1_sequence))
        formula_counts["decoder_rescued_top1_formula"] += int(decoder_exact and not top1_exact)
        formula_counts["decoder_regressed_top1_formula"] += int(top1_exact and not decoder_exact)
        beam_rank_histogram[str(beam_rank) if beam_rank is not None else "outside_top32"] += 1
        if bucket != "decoder_exact":
            bucket_sample_ids.setdefault(bucket, [])
            if len(bucket_sample_ids[bucket]) < 25:
                bucket_sample_ids[bucket].append(sample_id)

        for group, target, top1, decoded in zip(
            groups, gold_sequence, top1_sequence, decoder_sequence, strict=True,
        ):
            symbol = symbols_by_group[frozenset(group)]
            topk = [str(token) for token in symbol["hwr_topk"]]
            target_rank = topk.index(target) + 1 if target in topk else None
            target_rank_histogram[str(target_rank) if target_rank is not None else "outside_top5"] += 1
            token_counts["total"] += 1
            token_counts["top1_hits"] += int(top1 == target)
            token_counts["decoder_hits"] += int(decoded == target)
            token_counts["decoder_top1_agreements"] += int(decoded == top1)
            token_counts["decoder_top1_disagreements"] += int(decoded != top1)
            if decoded not in topk:
                raise AssertionError(f"decoder emitted token outside HWR Top-5: {sample_id}")
            if top1 != target:
                wrong_top1_confusions[(target, top1)] += 1
            if decoded != target:
                wrong_decoder_confusions[(target, decoded)] += 1
                token_counts["decoder_wrong_gold_in_top5"] += int(target_rank is not None)
                token_counts["decoder_wrong_gold_outside_top5"] += int(target_rank is None)

    formula_total = len(group_exact_rows)
    bucket_total = sum(bucket_counts.values())
    if bucket_total != formula_total:
        raise AssertionError(f"decoder failure buckets do not reconcile: {arm_name}")
    return {
        "group_exact_formulas": formula_total,
        "formula_failure_buckets": dict(sorted(bucket_counts.items())),
        "top1_formula_exact": formula_counts["top1_formula_exact"],
        "top5_complete_formulas": formula_counts["top5_complete"],
        "decoder_exact_formulas": formula_counts["decoder_exact"],
        "decoder_equals_independent_top1_formulas": formula_counts["decoder_equals_independent_top1"],
        "decoder_rescued_top1_formula_count": formula_counts["decoder_rescued_top1_formula"],
        "decoder_regressed_top1_formula_count": formula_counts["decoder_regressed_top1_formula"],
        "top32_gold_sequence_rank_histogram": dict(sorted(beam_rank_histogram.items())),
        "target_symbol_rank_histogram": dict(sorted(target_rank_histogram.items())),
        "token_counts": dict(token_counts),
        "top1_error_confusions": [
            {"gold": gold, "top1": predicted, "count": count}
            for (gold, predicted), count in wrong_top1_confusions.most_common(20)
        ],
        "decoder_error_confusions": [
            {"gold": gold, "decoder": predicted, "count": count}
            for (gold, predicted), count in wrong_decoder_confusions.most_common(20)
        ],
        "sample_ids_by_failure_bucket": bucket_sample_ids,
    }


def _pair_counts(before: list[bool], after: list[bool]) -> dict[str, int]:
    if len(before) != len(after):
        raise AssertionError("paired stage formula counts differ")
    result = Counter()
    for left, right in zip(before, after, strict=True):
        if left and right:
            result["both_exact"] += 1
        elif left:
            result["regressed"] += 1
        elif right:
            result["recovered"] += 1
        else:
            result["both_wrong"] += 1
    return dict(sorted(result.items()))


def _semantic_guard_subset(
    records: list[dict[str, Any]], arm_name: str, traces: dict[str, dict[str, Any]],
    equation_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    selected_rows = [
        record for record in records
        if record["hwr_tournament"][arm_name]["group_exact"]
    ]
    stage_names = (
        "hwr_top1", "after_fence_guard", "after_infix_guard", "after_equation_guard",
    )
    stage_exact = {name: [] for name in stage_names}
    top5_label_mismatch = 0
    probability_mismatch = 0
    maximum_probability_delta = 0.0
    order_matches_oracle = 0
    parity_group_count = 0

    for record in selected_rows:
        sample_id = str(record["sample_id"])
        preview = record["hwr_tournament"][arm_name]
        trace = traces[sample_id]
        predicted_groups = [
            frozenset(int(index) for index in group) for group in preview["groups"]
        ]
        oracle_symbols = list(trace["oracle_group_hwr"]["symbols"])
        oracle_groups = [
            frozenset(int(index) for index in symbol["stroke_indices"])
            for symbol in oracle_symbols
        ]
        if predicted_groups != oracle_groups:
            raise AssertionError(
                f"group-exact formula changed formula order: {sample_id} / {arm_name}"
            )
        order_matches_oracle += 1
        predicted_symbols = {
            frozenset(int(index) for index in symbol["stroke_indices"]): symbol
            for symbol in preview["selected_symbols"]
        }
        if set(predicted_symbols) != set(oracle_groups):
            raise AssertionError(f"symbol coverage mismatch: {sample_id} / {arm_name}")
        target_by_group = {
            frozenset(int(index) for index in group): str(token)
            for group, token in zip(
                preview["target_groups"], preview["target_tokens"], strict=True,
            )
        }
        for group, oracle_symbol in zip(oracle_groups, oracle_symbols, strict=True):
            symbol = predicted_symbols[group]
            prediction = oracle_symbol["prediction"]
            parity_group_count += 1
            topk = [str(token) for token in symbol["hwr_topk"]]
            reference_topk = [str(token) for token in prediction["top5"]]
            top5_label_mismatch += int(topk != reference_topk)
            if target_by_group[group] != str(oracle_symbol["target_label"]):
                raise AssertionError(f"target label differs across traces: {sample_id}")
            deltas = [
                abs(float(left) - float(right))
                for left, right in zip(
                    symbol["hwr_topk_probabilities"],
                    prediction["top5_probabilities"], strict=True,
                )
            ]
            row_max_delta = max(deltas, default=0.0)
            maximum_probability_delta = max(maximum_probability_delta, row_max_delta)
            probability_mismatch += int(row_max_delta > 1e-6)

        guard_trace = trace["semantic_guard_shadow"]
        exact = guard_trace["exact_by_stage"]
        baseline = bool(exact["hwr_top1"])
        if preview.get("token_exact_if_groups_exact") is not None and bool(
            preview["token_exact_if_groups_exact"]
        ) != baseline:
            raise AssertionError(f"Fast decoder/trace exactness mismatch: {sample_id}")
        stage_exact["hwr_top1"].append(baseline)
        stage_exact["after_fence_guard"].append(bool(exact["after_fence_guard"]))
        stage_exact["after_infix_guard"].append(bool(exact["after_infix_guard"]))
        equation = equation_rows.get(sample_id)
        if equation is not None and bool(equation["exact"]["after_infix"]) != bool(
            exact["after_infix_guard"]
        ):
            raise AssertionError(f"equation replay input-stage mismatch: {sample_id}")
        stage_exact["after_equation_guard"].append(
            bool(equation["exact"]["after_equation_guard"])
            if equation is not None else bool(exact["after_infix_guard"])
        )

    if top5_label_mismatch or probability_mismatch:
        raise AssertionError(
            f"group-exact HWR Top-5 differs from oracle trace: {arm_name}, "
            f"labels={top5_label_mismatch}, probability_rows={probability_mismatch}"
        )

    counts = {name: sum(values) for name, values in stage_exact.items()}
    stage_transitions = {
        f"{left}_to_{right}": _pair_counts(stage_exact[left], stage_exact[right])
        for left, right in zip(stage_names, stage_names[1:])
    }
    return {
        "scope": (
            "candidate-preserving posthoc shadow, evaluated only where predicted groups exactly "
            "match ownership; semantic stages are reused from frozen oracle-order traces"
        ),
        "group_exact_formulas": len(selected_rows),
        "strict_formula_exact_counts_on_group_exact_subset": counts,
        "strict_formula_exact_rates_over_all_formulas": {
            name: count / max(len(records), 1) for name, count in counts.items()
        },
        "stage_transition_counts": stage_transitions,
        "reference_parity": {
            "selected_group_symbols_compared": parity_group_count,
            "top5_label_mismatches": top5_label_mismatch,
            "top5_probability_rows_over_1e-6": probability_mismatch,
            "maximum_top5_probability_abs_delta": maximum_probability_delta,
            "predicted_group_order_matches_oracle_formula_order": order_matches_oracle,
        },
    }


def analyze_summary(
    source: dict[str, Any], source_sha256: str, *,
    semantic_guard_traces: dict[str, dict[str, Any]] | None = None,
    equation_rows: dict[str, dict[str, Any]] | None = None,
    semantic_trace_sha256: str | None = None,
    equation_audit_sha256: str | None = None,
) -> dict[str, Any]:
    tournament = source.get("hwr_tournament")
    records = source.get("records")
    if not isinstance(tournament, dict) or not isinstance(records, list):
        raise ValueError("input is not a detailed HWR tournament summary")
    arms = tournament.get("arms")
    if not isinstance(arms, dict):
        raise ValueError("input summary has no tournament arms")
    diagnostics = {}
    for arm_name, arm_summary in arms.items():
        result = _analyze_arm(records, arm_name)
        for key, expected in (
            ("group_exact_formulas", "group_exact"),
            ("top5_complete_formulas", "hwr_top5_complete_formulas_on_exact_groups"),
            ("decoder_exact_formulas", "token_exact_on_exact_groups"),
        ):
            if result[key] != int(arm_summary.get(expected, -1)):
                raise AssertionError(f"saved summary aggregate mismatch: {arm_name} / {key}")
        diagnostics[arm_name] = result
        if semantic_guard_traces is not None and equation_rows is not None:
            diagnostics[arm_name]["semantic_guard_on_group_exact_subset"] = (
                _semantic_guard_subset(records, arm_name, semantic_guard_traces, equation_rows)
            )
    report = {
        "schema": "aiflow-decoder-candidate-headroom-microscope/v1",
        "scope": "posthoc shadow diagnostics only; no fitting, tuning, CROHME, or promotion",
        "input_summary_sha256": source_sha256,
        "token_beam": TOKEN_BEAM,
        "arms": diagnostics,
    }
    if semantic_guard_traces is not None:
        report["semantic_guard_replay_inputs"] = {
            "formula_trace_sha256": semantic_trace_sha256,
            "equation_guard_audit_sha256": equation_audit_sha256,
            "product_default_enabled": False,
            "promotion_eligible": False,
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--semantic-guard-traces", type=Path)
    parser.add_argument("--semantic-equation-audit", type=Path)
    args = parser.parse_args()
    input_path = args.input_summary.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not input_path.is_file():
        parser.error(f"input summary does not exist: {input_path}")
    if output_path.exists():
        parser.error(f"refusing to overwrite diagnostic output: {output_path}")
    source_sha256 = _sha256(input_path)
    source = json.loads(input_path.read_text(encoding="utf-8"))
    if bool(args.semantic_guard_traces) != bool(args.semantic_equation_audit):
        parser.error("--semantic-guard-traces and --semantic-equation-audit must be supplied together")
    traces_by_id = equation_rows = None
    trace_sha256 = equation_sha256 = None
    if args.semantic_guard_traces is not None:
        trace_path = args.semantic_guard_traces.expanduser().resolve()
        equation_path = args.semantic_equation_audit.expanduser().resolve()
        trace_sha256 = _sha256(trace_path)
        equation_sha256 = _sha256(equation_path)
        trace_summary_path = trace_path.parent / "summary.json"
        if not trace_summary_path.is_file():
            parser.error(f"semantic layer summary is missing beside formula traces: {trace_summary_path}")
        trace_summary = json.loads(trace_summary_path.read_text(encoding="utf-8"))
        equation_audit = json.loads(equation_path.read_text(encoding="utf-8"))
        if equation_audit.get("input", {}).get("sha256") != trace_sha256:
            parser.error("semantic equation audit does not match the supplied formula trace")
        tournament_inputs = source.get("inputs", {})
        if trace_summary.get("runtime", {}).get("checkpoint_sha256") != tournament_inputs.get(
            "checkpoint_sha256"
        ):
            parser.error("semantic guard traces use a different HWR checkpoint")
        if trace_summary.get("runtime", {}).get("partition_ranker_sha256") != tournament_inputs.get(
            "partition_ranker_sha256"
        ):
            parser.error("semantic guard traces use a different grouping ranker")
        if int(trace_summary.get("dataset", {}).get("formula_count", -1)) != len(
            source.get("records", [])
        ):
            parser.error("semantic guard formula count does not match the tournament")
        trace_rows = [
            json.loads(line)
            for line in trace_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        traces_by_id = {str(row["sample_id"]): row for row in trace_rows}
        equation_rows = {
            str(row["sample_id"]): row
            for row in equation_audit.get("formula_level_trace", [])
        }
        expected_ids = {str(row["sample_id"]) for row in source.get("records", [])}
        if len(traces_by_id) != len(trace_rows) or set(traces_by_id) != expected_ids:
            parser.error("semantic guard trace sample IDs do not match the tournament")
        if not set(equation_rows) <= expected_ids:
            parser.error("equation guard trace contains unknown sample IDs")
        global_top1 = sum(
            bool(row["semantic_guard_shadow"]["exact_by_stage"]["hwr_top1"])
            for row in trace_rows
        )
        global_fence = sum(
            bool(row["semantic_guard_shadow"]["exact_by_stage"]["after_fence_guard"])
            for row in trace_rows
        )
        global_infix = sum(
            bool(row["semantic_guard_shadow"]["exact_by_stage"]["after_infix_guard"])
            for row in trace_rows
        )
        global_equation = int(equation_audit["metrics"]["after_equation_guard"]["formula_exact"])
        global_equation_replay = sum(
            bool(
                equation_rows[row["sample_id"]]["exact"]["after_equation_guard"]
                if row["sample_id"] in equation_rows
                else row["semantic_guard_shadow"]["exact_by_stage"]["after_infix_guard"]
            )
            for row in trace_rows
        )
        if global_top1 != int(trace_summary["oracle_group_hwr"]["formula_top1_exact"]):
            parser.error("semantic trace baseline does not reconcile with its layer summary")
        if global_fence != int(
            trace_summary["semantic_guard_shadow"]["stages"]["after_fence_guard"]["formula_exact"]
        ):
            parser.error("semantic fence stage does not reconcile with its layer summary")
        if global_infix != int(
            trace_summary["semantic_guard_shadow"]["stages"]["after_infix_guard"]["formula_exact"]
        ):
            parser.error("semantic infix stage does not reconcile with its layer summary")
        if global_equation_replay != global_equation:
            parser.error("semantic equation replay does not reconcile with its audit summary")
        records_by_id = {str(row["sample_id"]): row for row in source["records"]}
        for sample_id, trace in traces_by_id.items():
            reference = records_by_id[sample_id]["hwr_tournament"]["fast"]
            expected_targets = {
                frozenset(int(index) for index in group): str(token)
                for group, token in zip(
                    reference["target_groups"], reference["target_tokens"], strict=True,
                )
            }
            reference_symbols = {
                frozenset(int(index) for index in symbol["stroke_indices"]): str(
                    symbol["target_label"]
                )
                for symbol in trace["oracle_group_hwr"]["symbols"]
            }
            if expected_targets != reference_symbols:
                parser.error(f"ownership labels differ between traces: {sample_id}")
    report = analyze_summary(
        source, source_sha256,
        semantic_guard_traces=traces_by_id,
        equation_rows=equation_rows,
        semantic_trace_sha256=trace_sha256,
        equation_audit_sha256=equation_sha256,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output_path),
        "input_summary_sha256": source_sha256,
        "arms": {
            key: {
                "group_exact_formulas": value["group_exact_formulas"],
                "formula_failure_buckets": value["formula_failure_buckets"],
            }
            for key, value in report["arms"].items()
        },
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
