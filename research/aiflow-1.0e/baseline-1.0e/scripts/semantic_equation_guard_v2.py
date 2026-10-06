#!/usr/bin/env python3
"""Repair one uniquely solvable symbolic-looking glyph in a flat equation."""

from __future__ import annotations

from collections import Counter
import math

import train_masked_context_reranker_v1 as masked
from semantic_equation_guard_v1 import (
    ARITHMETIC_TOKENS,
    MAX_FORMULA_LENGTH,
    MIN_FORMULA_LENGTH,
    _flat,
    _row,
    apply_semantic_equation_guard,
    is_exact_arithmetic_equation,
)


def apply_semantic_equation_guard_v2(
    rows: list[dict], predictions: dict[str, str],
    candidate_probability_ratio_floor: float,
) -> tuple[dict[str, str], dict]:
    """Run v1, then replace one non-arithmetic slot only when the equation is unique."""
    if not math.isfinite(candidate_probability_ratio_floor) or not (
        0.0 <= candidate_probability_ratio_floor <= 1.0
    ):
        raise ValueError("candidate probability ratio floor must be in [0, 1]")
    output, base_audit = apply_semantic_equation_guard(rows, predictions)
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
        symbolic = [
            index for index, token in enumerate(baseline)
            if token not in ARITHMETIC_TOKENS
        ]
        if len(symbolic) != 1:
            audit["skipped_symbolic_count"] += 1
            continue
        index = symbolic[0]
        row = sequence[index]
        valid = []
        for candidate_index, candidate in enumerate(row["final_topk"]):
            candidate = str(candidate)
            if candidate not in ARITHMETIC_TOKENS:
                continue
            trial = list(baseline)
            trial[index] = candidate
            if is_exact_arithmetic_equation(tuple(trial)):
                valid.append((candidate_index, candidate, tuple(trial)))
        if len(valid) != 1:
            audit["skipped_not_unique"] += 1
            continue
        candidate_index, candidate, trial = valid[0]
        probabilities = row["final_topk_probabilities"]
        ratio = float(probabilities[candidate_index]) / max(
            float(probabilities[0]), 1e-12
        )
        if ratio < candidate_probability_ratio_floor:
            audit["skipped_probability_floor"] += 1
            continue
        output[str(row["record_id"])] = candidate
        changes.append({
            "formula_id": str(formula_id),
            "before": list(baseline),
            "after": list(trial),
            "changed_position": index,
            "candidate_probability_ratio": ratio,
        })
        audit["finalized_formulas"] += 1
        audit["changed_glyphs"] += 1
    if any(
        output[str(row["record_id"])] not in row["final_topk"] for row in rows
    ):
        raise AssertionError("semantic equation guard v2 invented a candidate")
    return output, {
        "configuration": {
            "base_guard": "semantic_equation_guard_v1",
            "symbolic_slots_required": 1,
            "valid_arithmetic_replacements_required": 1,
            "candidate_probability_ratio_floor": candidate_probability_ratio_floor,
        },
        "base_guard": base_audit,
        **{key: int(value) for key, value in sorted(audit.items())},
        "changes": changes,
        "candidate_preservation_rate": 1.0,
        "new_tokens": 0,
        "deleted_glyphs": 0,
        "grouping_mutations": 0,
    }


def self_test() -> None:
    unique = [
        (["1"], [1.0]),
        (["b", "6"], [0.8, 0.08]),
        ([r"\div"], [1.0]),
        (["8"], [1.0]),
        (["="], [1.0]),
        (["2"], [1.0]),
    ]
    rows = [_row(i, len(unique), *values) for i, values in enumerate(unique)]
    baseline = {row["record_id"]: row["final_topk"][0] for row in rows}
    finalized, audit = apply_semantic_equation_guard_v2(rows, baseline, 0.01)
    assert [finalized[row["record_id"]] for row in rows] == [
        "1", "6", r"\div", "8", "=", "2",
    ]
    assert audit["changed_glyphs"] == 1
    preserved, audit = apply_semantic_equation_guard_v2(rows, baseline, 0.2)
    assert preserved == baseline and audit["skipped_probability_floor"] == 1

    ambiguous = [
        (["0"], [1.0]), ([r"\times"], [1.0]),
        (["q", "7", "3"], [0.7, 0.2, 0.1]),
        (["="], [1.0]), (["0"], [1.0]),
    ]
    rows = [_row(i, len(ambiguous), *values) for i, values in enumerate(ambiguous)]
    baseline = {row["record_id"]: row["final_topk"][0] for row in rows}
    preserved, audit = apply_semantic_equation_guard_v2(rows, baseline, 0.01)
    assert preserved == baseline and audit["skipped_not_unique"] == 1


if __name__ == "__main__":
    self_test()
    print('{"self_test":"pass"}')
