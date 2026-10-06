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
    *, ambiguity_policy: str = "skip",
    maximum_competitor_probability_ratio: float = 0.01,
) -> tuple[dict[str, str], dict]:
    if not 0.0 <= minimum_probability_ratio <= 1.0:
        raise ValueError("semantic infix guard probability ratio must be in [0, 1]")
    if not 0.0 <= maximum_competitor_probability_ratio <= 1.0:
        raise ValueError("semantic infix guard competitor ratio must be in [0, 1]")
    if ambiguity_policy not in {
        "skip", "top_probability", "relative_dominance", "numeric_context_dominance",
    }:
        raise ValueError(
            "semantic infix guard ambiguity policy must be skip, top_probability, "
            "relative_dominance, or numeric_context_dominance"
        )
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
            center_token = snapshot[index]
            numeric_operator_role = (
                ambiguity_policy == "numeric_context_dominance"
                and roles[1] == "operator"
                and center_token not in INFIX_OPERATORS
                and center_token not in context.RELATION_BREAK_TOKENS
            )
            if (
                roles[0] not in VALUE_ROLES
                or roles[2] not in VALUE_ROLES
                or "digit" not in (roles[0], roles[2])
                or (roles[1] in {"operator", "fence"} and not numeric_operator_role)
                or roles == ["digit", "digit", "digit"]
            ):
                audit["skipped_role_pattern"] += 1
                continue
            if numeric_operator_role:
                audit["numeric_operator_role_contexts"] += 1
            row = sequence[index]
            operators = [
                (str(token), float(probability))
                for token, probability in zip(
                    row["final_topk"], row["final_topk_probabilities"], strict=True
                )
                if str(token) in INFIX_OPERATORS
            ]
            if not operators:
                audit["skipped_ambiguous_operator"] += 1
                continue
            if len(operators) > 1:
                if ambiguity_policy == "skip":
                    audit["skipped_ambiguous_operator"] += 1
                    continue
                ranked_operators = sorted(operators, key=lambda item: (-item[1], item[0]))
                if ambiguity_policy in {"relative_dominance", "numeric_context_dominance"}:
                    competitor_ratio = ranked_operators[1][1] / max(ranked_operators[0][1], 1e-12)
                    if competitor_ratio > maximum_competitor_probability_ratio:
                        audit["skipped_ambiguous_operator"] += 1
                        continue
                    token, probability = ranked_operators[0]
                    audit["resolved_dominant_operator"] += 1
                else:
                    token, probability = ranked_operators[0]
                    audit["resolved_ambiguous_operator"] += 1
            else:
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
            if numeric_operator_role:
                audit["numeric_operator_role_changes"] += 1
        audit["finalized_formulas"] += int(formula_changed)
    if any(
        output[str(row["record_id"])] not in row["final_topk"] for row in rows
    ):
        raise AssertionError("semantic infix guard violated candidate preservation")
    configuration = {
        "infix_operators": sorted(INFIX_OPERATORS),
        "value_roles": sorted(VALUE_ROLES),
        "minimum_probability_ratio": minimum_probability_ratio,
        "ambiguity_policy": ambiguity_policy,
        "spatial_relation": "right on both sides",
        "requires_digit_neighbor": True,
        "all_digit_triplets": "immutable",
    }
    if ambiguity_policy == "numeric_context_dominance":
        configuration["protected_relation_tokens"] = sorted(context.RELATION_BREAK_TOKENS)
    if ambiguity_policy in {"relative_dominance", "numeric_context_dominance"}:
        configuration["maximum_competitor_probability_ratio"] = maximum_competitor_probability_ratio
    return output, {
        "configuration": configuration,
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

    ambiguous_rows = [
        _row("left", 0, ["b"], [1.0], center_x=0.0),
        _row("middle", 1, ["4", "+", r"\times"], [0.7, 0.2, 0.1], center_x=1.0),
        _row("right", 2, ["4"], [1.0], center_x=2.0),
    ]
    ambiguous_baseline = {
        row["record_id"]: row["final_topk"][0] for row in ambiguous_rows
    }
    unchanged, conservative_audit = apply_semantic_infix_guard(
        ambiguous_rows, ambiguous_baseline, 0.0,
    )
    assert unchanged["middle"] == "4"
    assert conservative_audit["skipped_ambiguous_operator"] == 1
    assert "maximum_competitor_probability_ratio" not in conservative_audit["configuration"]
    resolved, argmax_audit = apply_semantic_infix_guard(
        ambiguous_rows, ambiguous_baseline, 0.0, ambiguity_policy="top_probability",
    )
    assert resolved["middle"] == "+"
    assert argmax_audit["resolved_ambiguous_operator"] == 1

    dominant_rows = [
        _row("left", 0, ["b"], [1.0], center_x=0.0),
        _row("middle", 1, ["4", r"\times", "+"], [0.7, 0.2, 0.00001], center_x=1.0),
        _row("right", 2, ["4"], [1.0], center_x=2.0),
    ]
    dominant_baseline = {
        row["record_id"]: row["final_topk"][0] for row in dominant_rows
    }
    dominant, dominance_audit = apply_semantic_infix_guard(
        dominant_rows, dominant_baseline, 0.0,
        ambiguity_policy="relative_dominance",
        maximum_competitor_probability_ratio=0.01,
    )
    assert dominant["middle"] == r"\times"
    assert dominance_audit["resolved_dominant_operator"] == 1
    assert dominance_audit["configuration"]["maximum_competitor_probability_ratio"] == 0.01

    genuinely_ambiguous, ambiguity_audit = apply_semantic_infix_guard(
        ambiguous_rows, ambiguous_baseline, 0.0,
        ambiguity_policy="relative_dominance",
        maximum_competitor_probability_ratio=0.01,
    )
    assert genuinely_ambiguous["middle"] == "4"
    assert ambiguity_audit["skipped_ambiguous_operator"] == 1

    numeric_operator_rows = [
        _row("left", 0, ["1"], [1.0], center_x=0.0),
        _row("middle", 1, [r"\bot", "+", r"\times"], [0.7, 0.2, 0.0001], center_x=1.0),
        _row("right", 2, ["5"], [1.0], center_x=2.0),
    ]
    numeric_operator_baseline = {
        row["record_id"]: row["final_topk"][0] for row in numeric_operator_rows
    }
    default_numeric_operator, default_numeric_audit = apply_semantic_infix_guard(
        numeric_operator_rows, numeric_operator_baseline, 0.0,
    )
    assert default_numeric_operator["middle"] == r"\bot"
    assert default_numeric_audit.get("numeric_operator_role_contexts", 0) == 0
    ordinary_dominance_numeric_operator, ordinary_dominance_audit = apply_semantic_infix_guard(
        numeric_operator_rows, numeric_operator_baseline, 0.0,
        ambiguity_policy="relative_dominance",
        maximum_competitor_probability_ratio=0.01,
    )
    assert ordinary_dominance_numeric_operator["middle"] == r"\bot"
    assert ordinary_dominance_audit.get("numeric_operator_role_contexts", 0) == 0
    numeric_operator, numeric_operator_audit = apply_semantic_infix_guard(
        numeric_operator_rows, numeric_operator_baseline, 0.0,
        ambiguity_policy="numeric_context_dominance",
        maximum_competitor_probability_ratio=0.01,
    )
    assert numeric_operator["middle"] == "+"
    assert numeric_operator_audit["numeric_operator_role_contexts"] == 1
    assert numeric_operator_audit["numeric_operator_role_changes"] == 1

    protected_arithmetic_rows = [
        _row("left", 0, ["1"], [1.0], center_x=0.0),
        _row("middle", 1, ["+", r"\times"], [0.8, 0.1], center_x=1.0),
        _row("right", 2, ["5"], [1.0], center_x=2.0),
    ]
    protected_arithmetic = {
        row["record_id"]: row["final_topk"][0] for row in protected_arithmetic_rows
    }
    protected_output, protected_audit = apply_semantic_infix_guard(
        protected_arithmetic_rows, protected_arithmetic, 0.0,
        ambiguity_policy="numeric_context_dominance",
        maximum_competitor_probability_ratio=0.01,
    )
    assert protected_output["middle"] == "+"
    assert protected_audit.get("numeric_operator_role_contexts", 0) == 0

    protected_relation_rows = [
        _row("left", 0, ["1"], [1.0], center_x=0.0),
        _row("middle", 1, ["=", r"\div"], [0.8, 0.1], center_x=1.0),
        _row("right", 2, ["5"], [1.0], center_x=2.0),
    ]
    protected_relation = {
        row["record_id"]: row["final_topk"][0] for row in protected_relation_rows
    }
    protected_relation_output, protected_relation_audit = apply_semantic_infix_guard(
        protected_relation_rows, protected_relation, 0.0,
        ambiguity_policy="numeric_context_dominance",
        maximum_competitor_probability_ratio=0.01,
    )
    assert protected_relation_output["middle"] == "="
    assert protected_relation_audit.get("numeric_operator_role_contexts", 0) == 0

    nonhorizontal_numeric_operator_rows = [
        _row("left", 0, ["1"], [1.0], center_x=0.0, center_y=0.0),
        _row("middle", 1, [r"\bot", "+"], [0.7, 0.2], center_x=1.0, center_y=1.0),
        _row("right", 2, ["5"], [1.0], center_x=2.0, center_y=0.0),
    ]
    nonhorizontal_baseline = {
        row["record_id"]: row["final_topk"][0]
        for row in nonhorizontal_numeric_operator_rows
    }
    nonhorizontal_output, nonhorizontal_audit = apply_semantic_infix_guard(
        nonhorizontal_numeric_operator_rows, nonhorizontal_baseline, 0.0,
        ambiguity_policy="numeric_context_dominance",
        maximum_competitor_probability_ratio=0.01,
    )
    assert nonhorizontal_output["middle"] == r"\bot"
    assert nonhorizontal_audit["skipped_nonhorizontal"] == 1

    below_floor, floor_audit = apply_semantic_infix_guard(
        ambiguous_rows, ambiguous_baseline, 0.3, ambiguity_policy="top_probability",
    )
    assert below_floor["middle"] == "4"
    assert floor_audit["skipped_probability_floor"] == 1

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
