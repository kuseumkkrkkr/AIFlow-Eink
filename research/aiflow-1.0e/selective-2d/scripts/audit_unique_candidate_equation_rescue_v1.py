#!/usr/bin/env python3
"""Shadow-test unique exact arithmetic equations recoverable from HWR candidates."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
import random
from pathlib import Path
from typing import Any

from raw_formula_context_runtime_v1 import _candidate_preserving_semantic_guard_shadow
from semantic_equation_guard_v1 import ARITHMETIC_TOKENS, _flat, is_exact_arithmetic_equation
from semantic_infix_guard_v1 import apply_semantic_infix_guard


SCHEMA = "aiflow-hwr-unique-candidate-equation-rescue/v1"
MAX_CHANGED_GLYPHS = 3


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _diagnostic_topk(trace: dict[str, Any]) -> tuple[list[list[str]], list[list[float]]]:
    candidates: list[list[str]] = []
    probabilities: list[list[float]] = []
    for symbol in trace["oracle_group_hwr"]["symbols"]:
        diagnostic = symbol["prediction"]["diagnostic_topk"]
        tokens = [str(token) for token in diagnostic["tokens"]]
        probs = [float(value) for value in diagnostic["probabilities"]]
        if len(tokens) != int(diagnostic["k"]) or len(tokens) != len(probs):
            raise AssertionError(f"invalid diagnostic Top-K: {trace['sample_id']}")
        if len(set(tokens)) != len(tokens):
            raise AssertionError(f"duplicate diagnostic candidate: {trace['sample_id']}")
        candidates.append(tokens)
        probabilities.append(probs)
    return candidates, probabilities


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows_and_candidates(trace: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str], list[list[str]], list[list[float]]]:
    targets = [str(token) for token in trace["source"]["target_tokens"]]
    symbols = trace["oracle_group_hwr"]["symbols"]
    if len(targets) != len(symbols):
        raise AssertionError(f"trace target/symbol mismatch: {trace['sample_id']}")
    rows = []
    baseline = []
    candidates = []
    probabilities = []
    for index, symbol in enumerate(symbols):
        prediction = symbol["prediction"]
        top5 = [str(token) for token in prediction["top5"]]
        probs = [float(value) for value in prediction["top5_probabilities"]]
        if len(top5) != 5 or len(probs) != 5 or len(set(top5)) != 5:
            raise AssertionError(f"invalid Top-5 candidates: {trace['sample_id']}:{index}")
        box = symbol["preprocessing"]["raw_bbox"]
        left, right = float(box["left"]), float(box["right"])
        top, bottom = float(box["top"]), float(box["bottom"])
        rows.append({
            "record_id": f"{trace['sample_id']}:{index}",
            "formula_id": str(trace["sample_id"]),
            "context": {"index": index, "length": len(symbols)},
            "final_topk": top5,
            "final_topk_probabilities": probs,
            "geometry": {
                "left": left, "right": right, "top": top, "bottom": bottom,
                "center_x": (left + right) / 2.0,
                "center_y": (top + bottom) / 2.0,
                "width_rel": max(right - left, 1e-6),
                "height_rel": max(bottom - top, 1e-6),
            },
        })
        baseline.append(str(prediction["top1"]))
        candidates.append(top5)
        probabilities.append(probs)
    return rows, baseline, candidates, probabilities


def _search_unique_equation(
    rows: list[dict[str, Any]], baseline: list[str], candidates: list[list[str]],
) -> tuple[list[str] | None, dict[str, Any]]:
    audit: dict[str, Any] = {
        "status": "skipped",
        "reason": "not_flat_single_top1_equality",
        "valid_candidate_count": 0,
        "candidate_sequences_checked": 0,
        "changes": [],
    }
    if not 3 <= len(rows) <= 12 or not _flat(rows) or baseline.count("=") != 1:
        return None, audit
    unknown_positions = [
        index for index, token in enumerate(baseline)
        if token not in ARITHMETIC_TOKENS
    ]
    if not unknown_positions:
        audit["reason"] = "baseline_already_in_arithmetic_vocabulary"
        return None, audit
    if len(unknown_positions) > MAX_CHANGED_GLYPHS:
        audit.update({
            "reason": "unknown_symbol_count_exceeds_edit_budget",
            "unknown_positions": unknown_positions,
        })
        return None, audit
    options: dict[int, list[str]] = {}
    for index in unknown_positions:
        options[index] = [
            token for token in candidates[index]
            if token in ARITHMETIC_TOKENS and token != baseline[index]
        ]
        if not options[index]:
            audit.update({
                "reason": "unsupported_symbol_has_no_arithmetic_top5_alternative",
                "blocked_position": index,
                "blocked_token": baseline[index],
            })
            return None, audit

    valid: dict[tuple[str, ...], float] = {}
    for replacements in itertools.product(*(options[index] for index in unknown_positions)):
        proposal = list(baseline)
        for index, token in zip(unknown_positions, replacements, strict=True):
            proposal[index] = token
        audit["candidate_sequences_checked"] += 1
        if is_exact_arithmetic_equation(tuple(proposal)):
            score_delta = sum(
                math.log(max(rows[index]["final_topk_probabilities"][
                    rows[index]["final_topk"].index(token)
                ], 1e-12))
                - math.log(max(rows[index]["final_topk_probabilities"][0], 1e-12))
                for index, token in zip(unknown_positions, replacements, strict=True)
            )
            valid[tuple(proposal)] = score_delta
            if len(valid) > 1:
                break
    audit["valid_candidate_count"] = len(valid)
    if len(valid) != 1:
        audit["reason"] = "no_exact_candidate" if not valid else "multiple_exact_candidates"
        return None, audit

    selected, score_delta = next(iter(valid.items()))
    changes = []
    for index, (before, after) in enumerate(zip(baseline, selected, strict=True)):
        if before == after:
            continue
        changes.append({
            "position": index,
            "record_id": str(rows[index]["record_id"]),
            "before": before,
            "after": after,
            "candidate_rank": candidates[index].index(after) + 1,
            "candidate_probability": rows[index]["final_topk_probabilities"][candidates[index].index(after)],
            "selected_minus_top1_log_probability": (
                math.log(max(rows[index]["final_topk_probabilities"][candidates[index].index(after)], 1e-12))
                - math.log(max(rows[index]["final_topk_probabilities"][0], 1e-12))
            ),
        })
    audit.update({
        "status": "changed_shadow_only",
        "reason": "unique_exact_candidate_equation",
        "score_delta_vs_hwr_top1_nats": score_delta,
        "changes": changes,
    })
    return list(selected), audit


def _terminal_rhs_bar_rescue(
    baseline: list[str], candidates: list[list[str]],
) -> tuple[list[str], dict[str, Any]]:
    audit: dict[str, Any] = {
        "status": "skipped",
        "reason": "not_single_terminal_rhs_bar_after_equality",
        "changes": [],
    }
    equality_positions = [index for index, token in enumerate(baseline) if token == "="]
    bar_positions = [index for index, token in enumerate(baseline) if token == "|"]
    if (
        len(equality_positions) != 1
        or equality_positions[0] != len(baseline) - 2
        or baseline[-1] != "|"
        or bar_positions != [len(baseline) - 1]
    ):
        return list(baseline), audit
    options = candidates[-1]
    if "1" not in options:
        audit.update({"reason": "unit_candidate_missing_from_top5", "bar_position": len(baseline) - 1})
        return list(baseline), audit
    selected = list(baseline)
    selected[-1] = "1"
    audit.update({
        "status": "changed_shadow_only",
        "reason": "single_terminal_rhs_bar_is_incomplete_fence",
        "changes": [{
            "position": len(baseline) - 1,
            "before": "|",
            "after": "1",
            "candidate_rank": options.index("1") + 1,
        }],
    })
    return selected, audit


def _writer_bootstrap(
    rows: list[dict[str, Any]], writer_by_id: dict[str, str], *, iterations: int = 10_000,
    seed: int = 20261001,
) -> dict[str, Any]:
    by_writer: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_writer.setdefault(writer_by_id[row["sample_id"]], []).append(row)
    writers = sorted(by_writer)
    rng = random.Random(seed)
    deltas = []
    for _ in range(iterations):
        sampled = [rng.choice(writers) for _ in writers]
        denominator = sum(len(by_writer[writer]) for writer in sampled)
        before = sum(int(row["baseline_exact"]) for writer in sampled for row in by_writer[writer])
        after = sum(int(row["challenger_exact"]) for writer in sampled for row in by_writer[writer])
        deltas.append(100.0 * (after - before) / denominator)
    deltas.sort()
    return {
        "method": "paired writer-cluster bootstrap; writers sampled with replacement",
        "iterations": iterations,
        "seed": seed,
        "delta_exact_rate_pp_95_interval": [
            deltas[int(0.025 * (iterations - 1))],
            deltas[int(0.975 * (iterations - 1))],
        ],
        "interpretation": "consumed-development exploratory shadow only",
    }


def _arm_comparison(
    rows: list[dict[str, Any]], tokens_key: str, *, baseline_tokens_key: str = "baseline_tokens",
    baseline_exact_key: str = "baseline_exact", baseline_hits_key: str = "baseline_hits",
) -> dict[str, Any]:
    baseline_exact = sum(int(row[baseline_exact_key]) for row in rows)
    challenger_exact = sum(row[tokens_key] == row["targets"] for row in rows)
    baseline_hits = sum(int(row[baseline_hits_key]) for row in rows)
    challenger_hits = sum(
        sum(a == b for a, b in zip(row[tokens_key], row["targets"], strict=True))
        for row in rows
    )
    changed_formula_ids = []
    recovered_formula_ids = []
    regressed_formula_ids = []
    recovered_tokens = []
    regressed_tokens = []
    wrong_to_wrong = []
    changed_token_count = 0
    for row in rows:
        before, after, targets = row[baseline_tokens_key], row[tokens_key], row["targets"]
        if before != after:
            changed_formula_ids.append(row["sample_id"])
        before_exact = before == targets
        after_exact = after == targets
        if not before_exact and after_exact:
            recovered_formula_ids.append(row["sample_id"])
        elif before_exact and not after_exact:
            regressed_formula_ids.append(row["sample_id"])
        for position, (old, new, target) in enumerate(zip(before, after, targets, strict=True)):
            if old == new:
                continue
            changed_token_count += 1
            item = {
                "sample_id": row["sample_id"], "position": position,
                "target": target, "before": old, "after": new,
            }
            if old != target and new == target:
                recovered_tokens.append(item)
            elif old == target and new != target:
                regressed_tokens.append(item)
            elif old != target and new != target:
                wrong_to_wrong.append(item)
    return {
        "baseline": {"formula_exact": baseline_exact, "token_hits": baseline_hits},
        "challenger": {"formula_exact": challenger_exact, "token_hits": challenger_hits},
        "delta": {
            "formula_exact": challenger_exact - baseline_exact,
            "token_hits": challenger_hits - baseline_hits,
            "changed_formula_ids": changed_formula_ids,
            "recovered_formula_ids": recovered_formula_ids,
            "regressed_formula_ids": regressed_formula_ids,
            "changed_token_count": changed_token_count,
            "recovered_tokens": recovered_tokens,
            "regressed_tokens": regressed_tokens,
            "wrong_to_wrong_changes": wrong_to_wrong,
        },
    }


def _self_test() -> None:
    tokens = [r"\lfloor", r"\mathscr{C}", r"\div", "3", "=", "6"]
    candidates = [
        [r"\lfloor", "1", "|", "I", r"\mid"],
        [r"\mathscr{C}", r"\mathscr{E}", "8", r"\mathcal{C}", r"\zeta"],
        [r"\div", r"\pm", r"\Psi", "+", r"\doteq"],
        ["3", r"\beta", r"\mathcal{G}", r"\}", "B"],
        ["=", r"\Xi", r"\simeq", r"\approx", r"\supseteq"],
        ["6", "b", "8", r"\flat", r"\delta"],
    ]
    rows = []
    for index, top5 in enumerate(candidates):
        rows.append({
            "record_id": f"eq:{index}",
            "formula_id": "eq",
            "context": {"index": index, "length": len(tokens)},
            "final_topk": top5,
            "final_topk_probabilities": [0.8, 0.1, 0.05, 0.03, 0.02],
            "geometry": {
                "center_x": float(index), "center_y": 0.0,
                "height_rel": 1.0, "width_rel": 0.5,
            },
        })
    selected, audit_row = _search_unique_equation(rows, tokens, candidates)
    assert selected == ["1", "8", r"\div", "3", "=", "6"]
    assert audit_row["valid_candidate_count"] == 1
    assert len(audit_row["changes"]) == 2
    assert all(
        token in row["final_topk"]
        for token, row in zip(selected, rows, strict=True)
    )

    ambiguous_rows = []
    ambiguous_tokens = ["x", "=", "x"]
    ambiguous_candidates = [["x", "1", "2", "3", "4"], ["=", "+", "-", "/", "0"], ["x", "1", "2", "3", "4"]]
    for index, top5 in enumerate(ambiguous_candidates):
        ambiguous_rows.append({
            "record_id": f"ambiguous:{index}",
            "formula_id": "ambiguous",
            "context": {"index": index, "length": 3},
            "final_topk": top5,
            "final_topk_probabilities": [0.8, 0.1, 0.05, 0.03, 0.02],
            "geometry": {
                "center_x": float(index), "center_y": 0.0,
                "height_rel": 1.0, "width_rel": 0.5,
            },
        })
    ambiguous, ambiguous_audit = _search_unique_equation(
        ambiguous_rows, ambiguous_tokens, ambiguous_candidates,
    )
    assert ambiguous is None and ambiguous_audit["reason"] == "multiple_exact_candidates"

    bar_candidates = [["f", "F", "h", "\\Gamma", "1"], ["(", "[", "1", "x", "f"],
                      ["y", "x", "1", "2", "z"], [")", "]", "1", "x", "f"],
                      ["=", "\\simeq", "+", "1", "x"], ["|", "1", "I", "\\mid", "\\rfloor"]]
    bar_selected, bar_audit = _terminal_rhs_bar_rescue(
        ["f", "(", "y", ")", "=", "|"], bar_candidates,
    )
    assert bar_selected == ["f", "(", "y", ")", "=", "1"]
    assert bar_audit["reason"] == "single_terminal_rhs_bar_is_incomplete_fence"
    unchanged_bar, unchanged_bar_audit = _terminal_rhs_bar_rescue(
        ["f", "(", "y", ")", "=", "|", "x", "|"],
        [*bar_candidates, ["x", "1", "2", "3", "4"], ["|", "1", "I", "\\mid", "\\rfloor"]],
    )
    assert unchanged_bar[-1] == "|"
    assert unchanged_bar_audit["status"] == "skipped"

    wide_tokens = ["5", r"\times", ")", "=", "5"]
    wide_candidates = [
        ["5", "E", "s", r"\varpi", "S"],
        [r"\times", "X", "x", r"\mathcal{X}", r"\chi"],
        [")", "]", r"\}", r"\rangle", r"\rfloor", r"\prime", r"\int", "I", "J", "|", r"\lambda", r"\rrbracket", r"\Lambda", "1", "l", ">", r"\rceil", r"\upharpoonright", r"\AE", r"\nu"],
        ["=", r"\geq", r"\gtrsim", "]", r"\succ"],
        ["5", "J", r"\mathcal{N}", "S", "s"],
    ]
    wide_rows = []
    for index, topk in enumerate(wide_candidates):
        probabilities = [0.995] + [0.005 / (len(topk) - 1)] * (len(topk) - 1)
        wide_rows.append({
            "record_id": f"wide:{index}",
            "formula_id": "wide",
            "context": {"index": index, "length": len(wide_tokens)},
            "final_topk": topk,
            "final_topk_probabilities": probabilities,
            "geometry": {
                "center_x": float(index), "center_y": 0.0,
                "height_rel": 1.0, "width_rel": 0.5,
            },
        })
    wide_selected, wide_audit = _search_unique_equation(
        wide_rows, wide_tokens, wide_candidates,
    )
    assert wide_selected == ["5", r"\times", "1", "=", "5"]
    assert wide_audit["changes"][0]["candidate_rank"] == 14


def audit(
    trace_path: Path, ownership_path: Path, cause_matrix_path: Path,
    grouping_audit_path: Path, writer_reference_path: Path,
    semantic_reference_path: Path, topk20_trace_path: Path | None = None,
) -> dict[str, Any]:
    traces = _jsonl(trace_path)
    ownership = _jsonl(ownership_path)
    cause = _json(cause_matrix_path)
    grouping_audit = _json(grouping_audit_path)
    writer_reference = _json(writer_reference_path)
    semantic_reference = _json(semantic_reference_path)
    topk20_traces = _jsonl(topk20_trace_path) if topk20_trace_path else []
    topk20_by_id = {str(row["sample_id"]): row for row in topk20_traces}
    trace_sha, ownership_sha = _sha256(trace_path), _sha256(ownership_path)
    grouping_sha = _sha256(grouping_audit_path)
    semantic_reference_sha = _sha256(semantic_reference_path)
    if cause.get("schema") != "aiflow-hwr-failure-cause-microscope/v1":
        raise ValueError("unexpected failure-cause matrix schema")
    if writer_reference.get("schema") != "aiflow-semantic-infix-writer-stratified-shadow/v1":
        raise ValueError("unexpected writer-reference schema")
    if cause["inputs"]["traces"]["sha256"] != trace_sha:
        raise AssertionError("trace hash differs from failure-cause matrix")
    if cause["inputs"]["grouping_audit"]["sha256"] != grouping_sha:
        raise AssertionError("grouping audit hash differs from failure-cause matrix")
    if writer_reference["inputs"]["ownership_train"]["sha256"] != ownership_sha:
        raise AssertionError("ownership hash differs from writer reference")
    if semantic_reference.get("schema") != "aiflow-semantic-guard-saved-tournament-audit/v10":
        raise ValueError("unexpected semantic-guard reference schema")
    ownership_hashes = [
        value for path, value in semantic_reference["inputs"]["dataset_file_sha256"].items()
        if str(path).lower().endswith("ownership_train.jsonl")
    ]
    if ownership_hashes != [ownership_sha]:
        raise AssertionError("semantic reference ownership hash differs")
    if semantic_reference["inputs"]["hwr_checkpoint_sha256"] != cause["inputs"]["checkpoint"]["sha256"]:
        raise AssertionError("semantic reference HWR checkpoint differs from failure matrix")
    trace_ids = {str(trace["sample_id"]) for trace in traces}
    if topk20_trace_path and set(topk20_by_id) != trace_ids:
        raise AssertionError("Top-K diagnostic IDs do not exactly match the frozen trace formulas")
    writer_by_id = {
        str(row["sample_id"]): str(row["writer_id"])
        for row in ownership if row.get("accepted")
    }
    if set(writer_by_id) != trace_ids:
        raise AssertionError("ownership IDs do not exactly match trace formulas")
    grouping_by_id = {
        str(row["sample_id"]): bool(row["fast_group_exact"])
        for row in grouping_audit["formula_level"]
    }
    if set(grouping_by_id) != trace_ids:
        raise AssertionError("grouping audit IDs do not exactly match trace formulas")

    formula_rows = []
    audits = []
    raw_top1_hits = stack_hits = numeric_hits = challenger_hits = combined_hits = terminal_bar_hits = wide_equation_hits = 0
    raw_top1_exact = stack_exact = numeric_exact = challenger_exact = combined_exact = terminal_bar_exact = wide_equation_exact = 0
    token_count = 0
    for trace in traces:
        sample_id = str(trace["sample_id"])
        targets = [str(token) for token in trace["source"]["target_tokens"]]
        rows, raw_top1, candidates, _ = _rows_and_candidates(trace)
        if len(targets) != len(raw_top1):
            raise AssertionError(f"target count mismatch: {sample_id}")
        groups = [
            [int(index) for index in symbol["stroke_indices"]]
            for symbol in trace["oracle_group_hwr"]["symbols"]
        ]
        symbols = []
        for symbol in trace["oracle_group_hwr"]["symbols"]:
            symbols.append({
                "stroke_indices": [int(index) for index in symbol["stroke_indices"]],
                "hwr_topk": [str(token) for token in symbol["prediction"]["top5"]],
                "hwr_topk_probabilities": [
                    float(value) for value in symbol["prediction"]["top5_probabilities"]
                ],
                "geometry": symbol["preprocessing"]["raw_bbox"],
            })
        existing_shadow = _candidate_preserving_semantic_guard_shadow(
            sample_id, groups, symbols, {"accepted": True, "tokens": raw_top1},
        )
        if existing_shadow.get("status") != "applied_shadow_only":
            raise AssertionError(f"existing shadow stack did not replay: {sample_id}")
        stack_tokens = [
            str(row["token"])
            for row in existing_shadow["stages"]["after_boundary_bar_as_unit_guard"]
        ]
        baseline_predictions = {
            str(row["record_id"]): token
            for row, token in zip(rows, stack_tokens, strict=True)
        }
        numeric_map, numeric_audit = apply_semantic_infix_guard(
            rows, baseline_predictions, minimum_probability_ratio=0.0,
            ambiguity_policy="numeric_context_dominance",
            maximum_competitor_probability_ratio=0.002,
        )
        numeric_tokens = [numeric_map[str(row["record_id"])] for row in rows]
        eq_output, audit_row = _search_unique_equation(rows, stack_tokens, candidates)
        combined_output, combined_audit = _search_unique_equation(
            rows, numeric_tokens, candidates,
        )
        chosen = eq_output or stack_tokens
        combined_tokens = combined_output or numeric_tokens
        terminal_bar_tokens, terminal_bar_audit = _terminal_rhs_bar_rescue(
            combined_tokens, candidates,
        )
        wide_equation_tokens = list(terminal_bar_tokens)
        wide_equation_audit: dict[str, Any] = {
            "status": "skipped", "reason": "topk20_trace_not_provided", "changes": [],
        }
        if topk20_trace_path:
            wide_trace = topk20_by_id[sample_id]
            if [str(token) for token in wide_trace["source"]["target_tokens"]] != targets:
                raise AssertionError(f"Top-K diagnostic target sequence mismatch: {sample_id}")
            wide_candidates, wide_probabilities = _diagnostic_topk(wide_trace)
            if len(wide_candidates) != len(candidates):
                raise AssertionError(f"Top-K diagnostic symbol count mismatch: {sample_id}")
            for index, (top5, probs5, widened, widened_probs) in enumerate(zip(
                candidates,
                [list(map(float, symbol["prediction"]["top5_probabilities"]))
                 for symbol in trace["oracle_group_hwr"]["symbols"]],
                wide_candidates,
                wide_probabilities,
                strict=True,
            )):
                if widened[:5] != top5 or any(
                    not math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-10)
                    for a, b in zip(widened_probs[:5], probs5, strict=True)
                ):
                    raise AssertionError(f"Top-K diagnostic prefix mismatch: {sample_id}:{index}")
            wide_rows = [dict(row) for row in rows]
            for index, row in enumerate(wide_rows):
                row["final_topk"] = wide_candidates[index]
                row["final_topk_probabilities"] = wide_probabilities[index]
            wide_output, wide_equation_audit = _search_unique_equation(
                wide_rows, terminal_bar_tokens, wide_candidates,
            )
            if wide_output is not None:
                wide_equation_tokens = wide_output
        for arm, tokens in (
            ("numeric-context", numeric_tokens),
            ("unique-equation", chosen),
            ("combined", combined_tokens),
            ("terminal-rhs-bar", terminal_bar_tokens),
        ):
            if any(token not in top5 for token, top5 in zip(tokens, candidates, strict=True)):
                raise AssertionError(f"{arm} shadow invented token: {sample_id}")
        for selected, selected_audit, arm in (
            (chosen, audit_row, "unique-equation"),
            (combined_tokens, combined_audit, "combined"),
        ):
            if selected_audit["status"] == "changed_shadow_only":
                if len(selected_audit["changes"]) > MAX_CHANGED_GLYPHS:
                    raise AssertionError(f"{arm} exceeded edit budget: {sample_id}")
                if not is_exact_arithmetic_equation(tuple(selected)):
                    raise AssertionError(f"{arm} output was not an exact equation: {sample_id}")
        raw_hits = sum(a == b for a, b in zip(raw_top1, targets, strict=True))
        base_hits = sum(a == b for a, b in zip(stack_tokens, targets, strict=True))
        next_hits = sum(a == b for a, b in zip(chosen, targets, strict=True))
        numeric_hit_count = sum(a == b for a, b in zip(numeric_tokens, targets, strict=True))
        combined_hit_count = sum(a == b for a, b in zip(combined_tokens, targets, strict=True))
        terminal_bar_hit_count = sum(a == b for a, b in zip(terminal_bar_tokens, targets, strict=True))
        wide_equation_hit_count = sum(a == b for a, b in zip(wide_equation_tokens, targets, strict=True))
        raw_exact = raw_top1 == targets
        base_exact = stack_tokens == targets
        next_exact = chosen == targets
        numeric_formula_exact = numeric_tokens == targets
        combined_formula_exact = combined_tokens == targets
        terminal_bar_formula_exact = terminal_bar_tokens == targets
        wide_equation_formula_exact = wide_equation_tokens == targets
        token_count += len(targets)
        raw_top1_hits += raw_hits
        stack_hits += base_hits
        numeric_hits += numeric_hit_count
        challenger_hits += next_hits
        combined_hits += combined_hit_count
        terminal_bar_hits += terminal_bar_hit_count
        wide_equation_hits += wide_equation_hit_count
        raw_top1_exact += int(raw_exact)
        stack_exact += int(base_exact)
        numeric_exact += int(numeric_formula_exact)
        challenger_exact += int(next_exact)
        combined_exact += int(combined_formula_exact)
        terminal_bar_exact += int(terminal_bar_formula_exact)
        wide_equation_exact += int(wide_equation_formula_exact)
        formula_rows.append({
            "sample_id": sample_id,
            "fast_group_exact": grouping_by_id[sample_id],
            "raw_top1_exact": raw_exact,
            "baseline_exact": base_exact,
            "numeric_exact": numeric_formula_exact,
            "challenger_exact": next_exact,
            "combined_exact": combined_formula_exact,
            "terminal_bar_exact": terminal_bar_formula_exact,
            "wide_equation_exact": wide_equation_formula_exact,
            "raw_top1_hits": raw_hits,
            "baseline_hits": base_hits,
            "numeric_hits": numeric_hit_count,
            "challenger_hits": next_hits,
            "combined_hits": combined_hit_count,
            "terminal_bar_hits": terminal_bar_hit_count,
            "wide_equation_hits": wide_equation_hit_count,
            "targets": targets,
            "raw_top1_tokens": raw_top1,
            "baseline_tokens": stack_tokens,
            "numeric_tokens": numeric_tokens,
            "challenger_tokens": chosen,
            "combined_tokens": combined_tokens,
            "terminal_bar_tokens": terminal_bar_tokens,
            "wide_equation_tokens": wide_equation_tokens,
            "wide_candidate_sets": wide_candidates if topk20_trace_path else candidates,
            "changed": stack_tokens != chosen,
            "numeric_changed": stack_tokens != numeric_tokens,
            "combined_changed": stack_tokens != combined_tokens,
            "terminal_bar_changed": stack_tokens != terminal_bar_tokens,
            "terminal_bar_only_changed": combined_tokens != terminal_bar_tokens,
            "audit": audit_row,
            "numeric_audit": numeric_audit,
            "combined_audit": combined_audit,
            "terminal_bar_audit": terminal_bar_audit,
            "wide_equation_audit": wide_equation_audit,
            "existing_shadow_audits": existing_shadow["audits"],
        })
        audits.append(audit_row)

    transitions = {
        "recovered_formula_ids": [],
        "regressed_formula_ids": [],
        "changed_formula_ids": [],
        "recovered_tokens": [],
        "regressed_tokens": [],
    }
    for row in formula_rows:
        sample_id = row["sample_id"]
        if row["changed"]:
            transitions["changed_formula_ids"].append(sample_id)
        if not row["baseline_exact"] and row["challenger_exact"]:
            transitions["recovered_formula_ids"].append(sample_id)
        if row["baseline_exact"] and not row["challenger_exact"]:
            transitions["regressed_formula_ids"].append(sample_id)
        for position, (before, after, target) in enumerate(zip(
            row["baseline_tokens"], row["challenger_tokens"], row["targets"], strict=True,
        )):
            if before == after:
                continue
            item = {"sample_id": sample_id, "position": position, "target": target,
                    "before": before, "after": after}
            if before != target and after == target:
                transitions["recovered_tokens"].append(item)
            elif before == target and after != target:
                transitions["regressed_tokens"].append(item)

    writer_rows = []
    for writer in sorted(set(writer_by_id.values())):
        rows = [row for row in formula_rows if writer_by_id[row["sample_id"]] == writer]
        writer_rows.append({
            "writer_id": writer,
            "formulas": len(rows),
            "baseline_exact": sum(int(row["baseline_exact"]) for row in rows),
            "challenger_exact": sum(int(row["challenger_exact"]) for row in rows),
            "exact_delta": sum(int(row["challenger_exact"]) - int(row["baseline_exact"]) for row in rows),
        })
    changes = [row for row in formula_rows if row["changed"]]
    exact_group_rows = [row for row in formula_rows if row["fast_group_exact"]]
    exact_group_tokens = sum(len(row["targets"]) for row in exact_group_rows)
    exact_group_stack_hits = sum(int(row["baseline_hits"]) for row in exact_group_rows)
    exact_group_challenger_hits = sum(int(row["challenger_hits"]) for row in exact_group_rows)
    exact_group_stack_exact = sum(int(row["baseline_exact"]) for row in exact_group_rows)
    exact_group_challenger_exact = sum(int(row["challenger_exact"]) for row in exact_group_rows)
    all_arm_comparisons = {
        "numeric_context_only": _arm_comparison(formula_rows, "numeric_tokens"),
        "unique_equation_only": _arm_comparison(formula_rows, "challenger_tokens"),
        "numeric_then_unique_equation": _arm_comparison(formula_rows, "combined_tokens"),
        "combined_then_terminal_rhs_bar": _arm_comparison(formula_rows, "terminal_bar_tokens"),
    }
    exact_group_comparisons = {
        "numeric_context_only": _arm_comparison(exact_group_rows, "numeric_tokens"),
        "unique_equation_only": _arm_comparison(exact_group_rows, "challenger_tokens"),
        "numeric_then_unique_equation": _arm_comparison(exact_group_rows, "combined_tokens"),
        "combined_then_terminal_rhs_bar": _arm_comparison(exact_group_rows, "terminal_bar_tokens"),
        "topk20_unique_equation_after_terminal_bar": _arm_comparison(
            exact_group_rows, "wide_equation_tokens",
            baseline_tokens_key="terminal_bar_tokens",
            baseline_exact_key="terminal_bar_exact",
            baseline_hits_key="terminal_bar_hits",
        ),
    }
    all_arm_comparisons["topk20_unique_equation_after_terminal_bar"] = _arm_comparison(
        formula_rows, "wide_equation_tokens",
        baseline_tokens_key="terminal_bar_tokens",
        baseline_exact_key="terminal_bar_exact",
        baseline_hits_key="terminal_bar_hits",
    )
    exact_group_writer_rows = [
        {
            "writer_id": writer,
            "formulas": sum(writer_by_id[row["sample_id"]] == writer for row in exact_group_rows),
            "baseline_exact": sum(
                int(row["baseline_exact"])
                for row in exact_group_rows if writer_by_id[row["sample_id"]] == writer
            ),
            "challenger_exact": sum(
                int(row["challenger_exact"])
                for row in exact_group_rows if writer_by_id[row["sample_id"]] == writer
            ),
            "numeric_context_exact": sum(
                int(row["numeric_exact"])
                for row in exact_group_rows if writer_by_id[row["sample_id"]] == writer
            ),
            "combined_exact": sum(
                int(row["combined_exact"])
                for row in exact_group_rows if writer_by_id[row["sample_id"]] == writer
            ),
        }
        for writer in sorted(set(writer_by_id.values()))
    ]
    exact_group_writer_rows = [row for row in exact_group_writer_rows if row["formulas"]]
    exact_group_bootstrap = {}
    for arm, tokens_key in (
        ("numeric_context_only", "numeric_tokens"),
        ("unique_equation_only", "challenger_tokens"),
        ("numeric_then_unique_equation", "combined_tokens"),
        ("combined_then_terminal_rhs_bar", "terminal_bar_tokens"),
        ("topk20_unique_equation_after_terminal_bar", "wide_equation_tokens"),
    ):
        bootstrap_rows = [
            {
                "sample_id": row["sample_id"],
                "formulas": 1,
                "baseline_exact": int(
                    row["terminal_bar_exact"]
                    if arm == "topk20_unique_equation_after_terminal_bar"
                    else row["baseline_exact"]
                ),
                "challenger_exact": int(row[tokens_key] == row["targets"]),
            }
            for row in exact_group_rows
        ]
        exact_group_bootstrap[arm] = _writer_bootstrap(bootstrap_rows, writer_by_id)
    expected_exact = semantic_reference["arms"]["fast"]["formula_exact_by_stage"][
        "after_boundary_bar_as_unit_guard"
    ]
    expected_hits = semantic_reference["arms"]["fast"]["token_hits_by_stage"][
        "after_boundary_bar_as_unit_guard"
    ]
    checks = {
        "formula_count_is_149": len(formula_rows) == 149,
        "token_count_is_579": token_count == 579,
        "trace_hash_matches_failure_matrix": cause["inputs"]["traces"]["sha256"] == trace_sha,
        "grouping_hash_matches_failure_matrix": cause["inputs"]["grouping_audit"]["sha256"] == grouping_sha,
        "ownership_hash_matches_writer_reference": writer_reference["inputs"]["ownership_train"]["sha256"] == ownership_sha,
        "ownership_ids_exactly_match_trace": set(writer_by_id) == trace_ids,
        "writer_count_is_9": len(set(writer_by_id.values())) == 9,
        "raw_top1_replays_76_formula_470_token_baseline": raw_top1_exact == 76 and raw_top1_hits == 470,
        "fast_exact_group_count_is_125": len(exact_group_rows) == 125,
        "fast_exact_group_token_count_matches_reference": exact_group_tokens == int(
            semantic_reference["arms"]["fast"]["evaluated_tokens"]
        ),
        "existing_shadow_stack_replays_saved_exact_group_metrics": (
            exact_group_stack_exact == int(expected_exact)
            and exact_group_stack_hits == int(expected_hits)
            and int(expected_exact) == 104
            and int(expected_hits) == 442
        ),
        "one_writer_bootstrap_subset_reconciles": len(exact_group_writer_rows) <= 9,
        "all_shadow_outputs_preserve_top5_members": all(
            token in candidate
            for row, trace in zip(formula_rows, traces, strict=True)
            for tokens in (
                row["numeric_tokens"], row["challenger_tokens"],
                row["combined_tokens"], row["terminal_bar_tokens"],
            )
            for token, candidate in zip(tokens, [
                [str(value) for value in symbol["prediction"]["top5"]]
                for symbol in trace["oracle_group_hwr"]["symbols"]
            ], strict=True)
        ),
        "every_change_is_unique_exact_arithmetic_equation": all(
            row["audit"]["status"] == "changed_shadow_only"
            and row["audit"]["valid_candidate_count"] == 1
            and is_exact_arithmetic_equation(tuple(row["challenger_tokens"]))
            for row in changes
        ),
        "terminal_rhs_bar_changes_preserve_candidate_and_strict_pattern": all(
            row["terminal_bar_audit"]["status"] == "changed_shadow_only"
            and row["terminal_bar_audit"]["reason"] == "single_terminal_rhs_bar_is_incomplete_fence"
            and row["terminal_bar_tokens"][-1] == "1"
            for row in formula_rows if row["terminal_bar_only_changed"]
        ),
        "topk20_policy_outputs_preserve_candidates": all(
            token in candidate
            for row in formula_rows
            for token, candidate in zip(
                row["wide_equation_tokens"], row["wide_candidate_sets"], strict=True,
            )
        ),
        "topk20_equation_changes_are_exact_and_unique": all(
            row["wide_equation_audit"]["status"] == "changed_shadow_only"
            and row["wide_equation_audit"]["valid_candidate_count"] == 1
            and is_exact_arithmetic_equation(tuple(row["wide_equation_tokens"]))
            for row in formula_rows
            if row["wide_equation_tokens"] != row["terminal_bar_tokens"]
        ),
    }
    checks["all_checks_pass"] = all(checks.values())
    if not checks["all_checks_pass"]:
        raise AssertionError("unique candidate equation rescue failed verification")
    return {
        "schema": SCHEMA,
        "scope": "frozen consumed-development candidate-lattice shadow; no training, CROHME, or product promotion",
        "inputs": {
            "trace_sha256": trace_sha,
            "ownership_sha256": ownership_sha,
            "cause_matrix_sha256": _sha256(cause_matrix_path),
            "grouping_audit_sha256": grouping_sha,
            "writer_reference_sha256": _sha256(writer_reference_path),
            "semantic_reference_sha256": semantic_reference_sha,
            "topk20_trace_sha256": _sha256(topk20_trace_path) if topk20_trace_path else None,
            "formulas": len(formula_rows),
            "writers": len(writer_rows),
            "tokens": token_count,
            "fast_exact_group_formulas": len(exact_group_rows),
            "fast_exact_group_tokens": exact_group_tokens,
        },
        "policy": {
            "description": "append after the frozen candidate-preserving semantic shadow stack; for flat 3-12-token formulas with one selected '=', replace every non-arithmetic token only when its own Top-5 has arithmetic alternatives and exactly one resulting sequence is a true arithmetic equation",
            "maximum_changed_glyphs": MAX_CHANGED_GLYPHS,
            "candidate_vocabulary": sorted(ARITHMETIC_TOKENS),
            "mathematical_truth_used_as_shadow_signal": True,
            "false_user_written_equations_are_a_known_risk": True,
        },
        "summary": {
            "status_counts": dict(sorted(Counter(
                str(row["status"]) + ":" + str(row["reason"]) for row in audits
            ).items())),
            "changed_formulas": len(changes),
            "raw_hwr_top1": {
                "formula_exact": raw_top1_exact,
                "token_hits": raw_top1_hits,
                "tokens": token_count,
            },
            "existing_shadow_stack": {
                "formula_exact": stack_exact,
                "token_hits": stack_hits,
                "tokens": token_count,
            },
            "numeric_context_only": {
                "formula_exact": numeric_exact,
                "token_hits": numeric_hits,
                "tokens": token_count,
            },
            "new_shadow": {
                "formula_exact": challenger_exact,
                "token_hits": challenger_hits,
                "tokens": token_count,
            },
            "numeric_then_unique_equation": {
                "formula_exact": combined_exact,
                "token_hits": combined_hits,
                "tokens": token_count,
            },
            "combined_then_terminal_rhs_bar": {
                "formula_exact": terminal_bar_exact,
                "token_hits": terminal_bar_hits,
                "tokens": token_count,
                "policy_changes": sum(int(row["terminal_bar_only_changed"]) for row in formula_rows),
            },
            "topk20_unique_equation_after_terminal_bar": {
                "formula_exact": wide_equation_exact,
                "token_hits": wide_equation_hits,
                "tokens": token_count,
                "policy_changes": sum(
                    int(row["wide_equation_tokens"] != row["terminal_bar_tokens"])
                    for row in formula_rows
                ),
            },
            "shadow_arm_comparisons_vs_existing_stack": all_arm_comparisons,
            "delta": {
                "formula_exact": challenger_exact - stack_exact,
                "token_hits": challenger_hits - stack_hits,
                **transitions,
            },
            "writer_stratified": writer_rows,
            "writer_cluster_bootstrap": _writer_bootstrap(formula_rows, writer_by_id),
            "fast_exact_group_subset": {
                "formulas": len(exact_group_rows),
                "tokens": exact_group_tokens,
                "existing_shadow_stack": {
                    "formula_exact": exact_group_stack_exact,
                    "token_hits": exact_group_stack_hits,
                },
                "new_shadow": {
                    "formula_exact": exact_group_challenger_exact,
                    "token_hits": exact_group_challenger_hits,
                },
                "arm_comparisons_vs_existing_stack": exact_group_comparisons,
                "delta": {
                    "formula_exact": exact_group_challenger_exact - exact_group_stack_exact,
                    "token_hits": exact_group_challenger_hits - exact_group_stack_hits,
                    "changed_formula_ids": [
                        row["sample_id"] for row in exact_group_rows if row["changed"]
                    ],
                    "regressed_formula_ids": [
                        row["sample_id"] for row in exact_group_rows
                        if row["baseline_exact"] and not row["challenger_exact"]
                    ],
                },
                "writer_stratified": exact_group_writer_rows,
                "writer_cluster_bootstrap_by_arm": exact_group_bootstrap,
            },
            "changed_cases": [
                {
                    "sample_id": row["sample_id"],
                    "targets": row["targets"],
                    "before": row["baseline_tokens"],
                    "after": row["challenger_tokens"],
                    "audit": row["audit"],
                }
                for row in changes
            ],
            "terminal_rhs_bar_cases": [
                {
                    "sample_id": row["sample_id"],
                    "targets": row["targets"],
                    "before": row["combined_tokens"],
                    "after": row["terminal_bar_tokens"],
                    "audit": row["terminal_bar_audit"],
                }
                for row in formula_rows if row["terminal_bar_only_changed"]
            ],
            "topk20_unique_equation_cases": [
                {
                    "sample_id": row["sample_id"],
                    "targets": row["targets"],
                    "before": row["terminal_bar_tokens"],
                    "after": row["wide_equation_tokens"],
                    "audit": row["wide_equation_audit"],
                }
                for row in formula_rows
                if row["wide_equation_tokens"] != row["terminal_bar_tokens"]
            ],
        },
        "verification": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--ownership", type=Path, required=True)
    parser.add_argument("--cause-matrix", type=Path, required=True)
    parser.add_argument("--grouping-audit", type=Path, required=True)
    parser.add_argument("--writer-reference", type=Path, required=True)
    parser.add_argument("--semantic-reference", type=Path, required=True)
    parser.add_argument("--topk20-trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(
        args.trace, args.ownership, args.cause_matrix, args.grouping_audit,
        args.writer_reference, args.semantic_reference, args.topk20_trace,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "all_checks_pass": report["verification"]["all_checks_pass"],
        "formula_exact": {
            "existing_shadow_stack": report["summary"]["existing_shadow_stack"]["formula_exact"],
            "after_terminal_rhs_bar": report["summary"]["combined_then_terminal_rhs_bar"]["formula_exact"],
            "after_topk20_equation": report["summary"]["topk20_unique_equation_after_terminal_bar"]["formula_exact"],
        },
        "fast_exact_group_terminal_bar_delta": report["summary"]["fast_exact_group_subset"]
        ["arm_comparisons_vs_existing_stack"]["combined_then_terminal_rhs_bar"]["delta"],
        "fast_exact_group_topk20_delta": report["summary"]["fast_exact_group_subset"]
        ["arm_comparisons_vs_existing_stack"]["topk20_unique_equation_after_terminal_bar"]["delta"],
        "topk20_recovered": report["summary"]["shadow_arm_comparisons_vs_existing_stack"]
        ["topk20_unique_equation_after_terminal_bar"]["delta"]["recovered_formula_ids"],
        "topk20_regressed": report["summary"]["shadow_arm_comparisons_vs_existing_stack"]
        ["topk20_unique_equation_after_terminal_bar"]["delta"]["regressed_formula_ids"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    _self_test()
    print('{"self_test":"pass"}')
    main()
