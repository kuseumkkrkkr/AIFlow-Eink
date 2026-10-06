#!/usr/bin/env python3
"""Resolve one unambiguous infix operator between horizontal value slots."""

from __future__ import annotations

from collections import Counter

import train_context_decision_layer_v1 as context
import train_masked_context_reranker_v1 as masked


INFIX_OPERATORS = frozenset({"+", "-", "/", r"\times", r"\div", r"\cdot"})
VALUE_ROLES = frozenset({"digit", "operand"})


def apply_semantic_infix_guard(
    rows: list[dict], predictions: dict[str, str], minimum_probability_ratio: float,
) -> tuple[dict[str, str], dict]:
    if not 0.0 <= minimum_probability_ratio <= 1.0:
        raise ValueError("semantic infix guard probability ratio must be in [0, 1]")
    record_ids = {str(row["record_id"]) for row in rows}
    if set(predictions) != record_ids:
        raise ValueError("semantic infix guard prediction coverage mismatch")
    output = {record_id: str(token) for record_id, token in predictions.items()}
    audit = Counter()
    changes = []
    formulae = masked._formulae(rows)
    for formula_id, sequence in formulae.items():
        snapshot = [output[str(row["record_id"])] for row in sequence]
        formula_changed = False
        for index in range(1, len(sequence) - 1):
            if (
                masked._spatial_relation(sequence[index - 1], sequence[index]) != "right"
                or masked._spatial_relation(sequence[index], sequence[index + 1]) != "right"
            ):
                audit["skipped_nonhorizontal"] += 1
                continue
            roles = [
                context._semantic_role(snapshot[position])
                for position in (index - 1, index, index + 1)
            ]
            if (
                roles[0] not in VALUE_ROLES
                or roles[2] not in VALUE_ROLES
                or "digit" not in (roles[0], roles[2])
                or roles[1] in {"operator", "fence"}
                or roles == ["digit", "digit", "digit"]
            ):
                audit["skipped_role_pattern"] += 1
                continue
            row = sequence[index]
            operators = [
                (str(token), float(probability))
                for token, probability in zip(
                    row["final_topk"], row["final_topk_probabilities"], strict=True
                )
                if str(token) in INFIX_OPERATORS
            ]
            if len(operators) != 1:
                audit["skipped_ambiguous_operator"] += 1
                continue
            token, probability = operators[0]
            ratio = probability / max(float(row["final_topk_probabilities"][0]), 1e-12)
            if ratio < minimum_probability_ratio:
                audit["skipped_probability_floor"] += 1
                continue
            record_id = str(row["record_id"])
            before = output[record_id]
            if token == before:
                continue
            if token not in row["final_topk"]:
                raise AssertionError("semantic infix guard invented a candidate")
            output[record_id] = token
            formula_changed = True
            changes.append({
                "formula_id": str(formula_id),
                "record_id": record_id,
                "context_index": index,
                "before": before,
                "after": token,
                "probability_ratio": ratio,
                "neighbor_roles": [roles[0], roles[2]],
            })
            audit["changed_glyphs"] += 1
        audit["finalized_formulas"] += int(formula_changed)
    if any(
        output[str(row["record_id"])] not in row["final_topk"] for row in rows
    ):
        raise AssertionError("semantic infix guard violated candidate preservation")
    return output, {
        "configuration": {
            "infix_operators": sorted(INFIX_OPERATORS),
            "value_roles": sorted(VALUE_ROLES),
            "minimum_probability_ratio": minimum_probability_ratio,
            "spatial_relation": "right on both sides",
            "requires_digit_neighbor": True,
            "all_digit_triplets": "immutable",
        },
        "formulas": len(formulae),
        **{key: int(value) for key, value in sorted(audit.items())},
        "changes": changes,
        "candidate_preservation_rate": 1.0,
        "new_tokens": 0,
        "deleted_glyphs": 0,
        "grouping_mutations": 0,
    }


def _row(
    record_id: str, index: int, candidates: list[str], probabilities: list[float],
    *, center_x: float, center_y: float = 0.5,
) -> dict:
    return {
        "record_id": record_id,
        "formula_id": "infix",
        "final_topk": candidates,
        "final_topk_probabilities": probabilities,
        "context": {"index": index, "length": 3},
        "geometry": {
            "center_x": center_x, "center_y": center_y,
            "width_rel": 0.2, "height_rel": 1.0,
        },
    }


def self_test() -> None:
    rows = [
        _row("left", 0, ["b"], [1.0], center_x=0.0),
        _row("middle", 1, ["4", "+"], [0.8, 0.1], center_x=1.0),
        _row("right", 2, ["4"], [1.0], center_x=2.0),
    ]
    baseline = {row["record_id"]: row["final_topk"][0] for row in rows}
    finalized, audit = apply_semantic_infix_guard(rows, baseline, 0.01)
    assert [finalized[row["record_id"]] for row in rows] == ["b", "+", "4"]
    assert audit["changed_glyphs"] == 1

    variable_rows = [
        _row("x", 0, ["x"], [1.0], center_x=0.0),
        _row("y", 1, ["y", "+"], [0.8, 0.1], center_x=1.0),
        _row("z", 2, ["z"], [1.0], center_x=2.0),
    ]
    variable = {
        row["record_id"]: row["final_topk"][0] for row in variable_rows
    }
    preserved, variable_audit = apply_semantic_infix_guard(
        variable_rows, variable, 0.01
    )
    assert preserved == variable and variable_audit.get("changed_glyphs", 0) == 0


if __name__ == "__main__":
    self_test()
    print('{"self_test":"pass"}')
