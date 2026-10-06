#!/usr/bin/env python3
"""Conservatively repair flat arithmetic equations inside an HWR candidate lattice."""

from __future__ import annotations

from collections import Counter
from fractions import Fraction
import math

import train_masked_context_reranker_v1 as masked


DIGITS = frozenset(str(value) for value in range(10))
ADDITIVE = frozenset({"+", "-"})
MULTIPLICATIVE = frozenset({r"\times", r"\div", "/", r"\cdot"})
ARITHMETIC_TOKENS = DIGITS | ADDITIVE | MULTIPLICATIVE | {"="}
MIN_FORMULA_LENGTH = 3
MAX_FORMULA_LENGTH = 12
MAX_CHANGED_GLYPHS = 3
BEAM_WIDTH = 64
PREFERRED_TOKEN_BONUS = 0.75
MIN_VALID_MARGIN = 1.5
MIN_VERTICAL_OVERLAP = 0.25


def _number(tokens: tuple[str, ...], index: int) -> tuple[Fraction | None, int]:
    sign = 1
    if index < len(tokens) and tokens[index] == "-":
        sign = -1
        index += 1
    start = index
    value = 0
    while index < len(tokens) and tokens[index] in DIGITS:
        value = value * 10 + int(tokens[index])
        index += 1
    if index == start:
        return None, index
    return Fraction(sign * value), index


def _term(tokens: tuple[str, ...], index: int) -> tuple[Fraction | None, int]:
    value, index = _number(tokens, index)
    if value is None:
        return None, index
    while index < len(tokens) and tokens[index] in MULTIPLICATIVE:
        operator = tokens[index]
        right, following = _number(tokens, index + 1)
        if right is None:
            return None, index
        if operator in {r"\div", "/"}:
            if right == 0:
                return None, index
            value /= right
        else:
            value *= right
        index = following
    return value, index


def _expression(tokens: tuple[str, ...]) -> Fraction | None:
    value, index = _term(tokens, 0)
    if value is None:
        return None
    while index < len(tokens):
        operator = tokens[index]
        if operator not in ADDITIVE:
            return None
        right, following = _term(tokens, index + 1)
        if right is None:
            return None
        value = value + right if operator == "+" else value - right
        index = following
    return value


def is_exact_arithmetic_equation(tokens: tuple[str, ...]) -> bool:
    if tokens.count("=") != 1:
        return False
    equality = tokens.index("=")
    if equality == 0 or equality == len(tokens) - 1:
        return False
    left = _expression(tokens[:equality])
    right = _expression(tokens[equality + 1:])
    return left is not None and right is not None and left == right


def _flat(sequence: list[dict]) -> bool:
    for left, right in zip(sequence, sequence[1:]):
        left_geometry = left["geometry"]
        right_geometry = right["geometry"]
        if float(right_geometry["center_x"]) <= float(left_geometry["center_x"]):
            return False
        left_height = max(float(left_geometry["height_rel"]), 1e-9)
        right_height = max(float(right_geometry["height_rel"]), 1e-9)
        overlap_top = max(
            float(left_geometry["center_y"]) - left_height / 2.0,
            float(right_geometry["center_y"]) - right_height / 2.0,
        )
        overlap_bottom = min(
            float(left_geometry["center_y"]) + left_height / 2.0,
            float(right_geometry["center_y"]) + right_height / 2.0,
        )
        overlap_ratio = (overlap_bottom - overlap_top) / min(
            left_height, right_height
        )
        if overlap_ratio < MIN_VERTICAL_OVERLAP:
            return False
    return True


def _options(row: dict, preferred: str) -> list[tuple[str, float]]:
    output = []
    for token, probability in zip(
        row["final_topk"], row["final_topk_probabilities"], strict=True
    ):
        token = str(token)
        probability = float(probability)
        if token not in ARITHMETIC_TOKENS:
            continue
        score = math.log(max(probability, 1e-12))
        if token == preferred:
            score += PREFERRED_TOKEN_BONUS
        output.append((token, score))
    return output


def apply_semantic_equation_guard(
    rows: list[dict], predictions: dict[str, str],
) -> tuple[dict[str, str], dict]:
    record_ids = {str(row["record_id"]) for row in rows}
    if set(predictions) != record_ids:
        raise ValueError("semantic equation guard prediction coverage mismatch")
    output = {record_id: str(token) for record_id, token in predictions.items()}
    audit = Counter()
    changes = []
    for formula_id, sequence in masked._formulae(rows).items():
        if not MIN_FORMULA_LENGTH <= len(sequence) <= MAX_FORMULA_LENGTH:
            audit["skipped_length"] += 1
            continue
        if not _flat(sequence):
            audit["skipped_nonflat"] += 1
            continue
        baseline = tuple(output[str(row["record_id"])] for row in sequence)
        if is_exact_arithmetic_equation(baseline):
            audit["baseline_already_valid"] += 1
            continue
        if any(token not in ARITHMETIC_TOKENS for token in baseline):
            audit["skipped_symbolic_baseline"] += 1
            continue
        options = [
            _options(row, output[str(row["record_id"])]) for row in sequence
        ]
        if any(not values for values in options):
            audit["skipped_nonarithmetic_slot"] += 1
            continue
        audit["eligible_formulas"] += 1
        beam: list[tuple[float, tuple[str, ...]]] = [(0.0, ())]
        for values in options:
            beam = sorted(
                (
                    (score + option_score, tokens + (token,))
                    for score, tokens in beam
                    for token, option_score in values
                ),
                key=lambda item: (-item[0], item[1]),
            )[:BEAM_WIDTH]
        valid = sorted(
            (item for item in beam if is_exact_arithmetic_equation(item[1])),
            key=lambda item: (-item[0], item[1]),
        )
        if not valid:
            audit["no_valid_equation"] += 1
            continue
        best_score, best = valid[0]
        second_score = valid[1][0] if len(valid) > 1 else None
        margin = best_score - second_score if second_score is not None else None
        changed_positions = [
            index for index, (before, after) in enumerate(zip(baseline, best, strict=True))
            if before != after
        ]
        if not 1 <= len(changed_positions) <= MAX_CHANGED_GLYPHS:
            audit["skipped_change_limit"] += 1
            continue
        if margin is not None and margin < MIN_VALID_MARGIN:
            audit["skipped_ambiguous"] += 1
            continue
        for index in changed_positions:
            row = sequence[index]
            token = best[index]
            if token not in row["final_topk"]:
                raise AssertionError("semantic equation guard invented a candidate")
            output[str(row["record_id"])] = token
        changes.append({
            "formula_id": str(formula_id),
            "before": list(baseline),
            "after": list(best),
            "changed_positions": changed_positions,
            "valid_score_margin": margin,
            "only_valid_equation": second_score is None,
        })
        audit["finalized_formulas"] += 1
        audit["changed_glyphs"] += len(changed_positions)
    if any(
        output[str(row["record_id"])] not in row["final_topk"] for row in rows
    ):
        raise AssertionError("semantic equation guard violated candidate preservation")
    return output, {
        "configuration": {
            "minimum_formula_length": MIN_FORMULA_LENGTH,
            "maximum_formula_length": MAX_FORMULA_LENGTH,
            "maximum_changed_glyphs": MAX_CHANGED_GLYPHS,
            "beam_width": BEAM_WIDTH,
            "preferred_token_bonus": PREFERRED_TOKEN_BONUS,
            "minimum_valid_score_margin": MIN_VALID_MARGIN,
            "flat_relation": (
                "center_x must increase and adjacent vertical intervals must overlap"
            ),
            "minimum_vertical_overlap": MIN_VERTICAL_OVERLAP,
            "arithmetic_tokens": sorted(ARITHMETIC_TOKENS),
            "symbolic_baseline_policy": (
                "skip when any selected token is outside the arithmetic vocabulary"
            ),
        },
        "formulas": len(masked._formulae(rows)),
        **{key: int(value) for key, value in sorted(audit.items())},
        "changes": changes,
        "candidate_preservation_rate": 1.0,
        "new_tokens": 0,
        "deleted_glyphs": 0,
        "grouping_mutations": 0,
    }


def _row(index: int, length: int, candidates: list[str], probabilities: list[float]) -> dict:
    return {
        "record_id": f"r{index}",
        "formula_id": "equation",
        "final_topk": candidates,
        "final_topk_probabilities": probabilities,
        "context": {"index": index, "length": length},
        "geometry": {
            "center_x": float(index), "center_y": 0.5,
            "width_rel": 0.2, "height_rel": 1.0,
        },
    }


def self_test() -> None:
    assert is_exact_arithmetic_equation(("1", "6", r"\div", "8", "=", "2"))
    assert is_exact_arithmetic_equation(("2", "+", "3", r"\times", "4", "=", "1", "4"))
    assert not is_exact_arithmetic_equation(("1", r"\div", "0", "=", "1"))
    candidates = [
        (["6"], [1.0]),
        ([r"\times"], [1.0]),
        (["6"], [1.0]),
        (["="], [1.0]),
        (["3"], [1.0]),
        ([r"\cdot", "6"], [0.7, 0.2]),
    ]
    rows = [_row(index, len(candidates), *values) for index, values in enumerate(candidates)]
    baseline = {row["record_id"]: row["final_topk"][0] for row in rows}
    finalized, audit = apply_semantic_equation_guard(rows, baseline)
    assert [finalized[row["record_id"]] for row in rows] == [
        "6", r"\times", "6", "=", "3", "6",
    ]
    assert audit["changed_glyphs"] == 1
    unchanged, locked = apply_semantic_equation_guard(rows, finalized)
    assert unchanged == finalized and locked["baseline_already_valid"] == 1
    symbolic_candidates = [
        (["0"], [1.0]), ([r"\times"], [1.0]),
        (["q", "7"], [0.7, 0.2]), (["="], [1.0]), (["0"], [1.0]),
    ]
    symbolic_rows = [
        _row(index, len(symbolic_candidates), *values)
        for index, values in enumerate(symbolic_candidates)
    ]
    symbolic = {
        row["record_id"]: row["final_topk"][0] for row in symbolic_rows
    }
    preserved, symbolic_audit = apply_semantic_equation_guard(
        symbolic_rows, symbolic
    )
    assert preserved == symbolic
    assert symbolic_audit["skipped_symbolic_baseline"] == 1


if __name__ == "__main__":
    self_test()
    print('{"self_test":"pass"}')
