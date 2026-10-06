#!/usr/bin/env python3
"""Repair a few high-confidence formula-role typos inside HWR Top-20."""

from __future__ import annotations

from collections import Counter
import math
import string

from operator_cross_slot_rescue_v1 import (
    _candidate_map as _validated_candidate_map,
    _formulae,
)
from wide_candidate_syntax_rescue_v1 import BINARY, DIGITS


SCHEMA = "aiflow-formula-role-typo-rescue/v1"
OUTPUT_SCHEMA = "aiflow-formula-role-typo-finalized/v1"
DEFAULT_CONFIGURATION = {
    "candidate_width": 20,
    "auxiliary_weight": 0.6,
    "allowed_auxiliary_policies": [
        "current_expanded_writer_loo_probability_fusion",
        "old_new_product_probability_fusion",
    ],
    "function_tokens": ["f", "g", "h"],
    "function_candidate_maximum_rank": 3,
    "parenthesized_variables": ["x", "y", "z"],
    "parenthesized_variable_maximum_rank": 4,
    "cross_plus_candidate_maximum_rank": 4,
    "cross_plus_required_stroke_count": 2,
    "cross_plus_maximum_absolute_aspect_log": 0.6,
    "cross_plus_minimum_path_over_diagonal": 1.1,
    "cross_plus_maximum_path_over_diagonal": 2.0,
    "vertical_one_candidate_maximum_rank": 4,
    "vertical_one_required_stroke_count": 1,
    "vertical_one_maximum_aspect_log": -1.5,
    "vertical_one_maximum_path_over_diagonal": 1.15,
    "vertical_one_minimum_absolute_direction_y": 0.9,
    "vertical_one_maximum_relative_width": 0.25,
    "digit_before_binary_top1_only": True,
    "function_rhs_top1_only": True,
    "value_cross_plus_open_fence_enabled": True,
}


def validate_configuration(configuration: dict) -> dict:
    fields = set(configuration)
    current_fields = set(DEFAULT_CONFIGURATION)
    legacy_fields = current_fields - {"value_cross_plus_open_fence_enabled"}
    if fields not in (legacy_fields, current_fields):
        raise ValueError("formula-role typo configuration fields mismatch")
    configuration = {
        **configuration,
        "value_cross_plus_open_fence_enabled": (
            configuration.get("value_cross_plus_open_fence_enabled", False)
        ),
    }
    output = {
        **configuration,
        "candidate_width": int(configuration["candidate_width"]),
        "auxiliary_weight": float(configuration["auxiliary_weight"]),
        "allowed_auxiliary_policies": [
            str(value) for value in configuration["allowed_auxiliary_policies"]
        ],
        "function_tokens": [
            str(value) for value in configuration["function_tokens"]
        ],
        "function_candidate_maximum_rank": int(
            configuration["function_candidate_maximum_rank"]
        ),
        "parenthesized_variables": [
            str(value) for value in configuration["parenthesized_variables"]
        ],
        "parenthesized_variable_maximum_rank": int(
            configuration["parenthesized_variable_maximum_rank"]
        ),
        "cross_plus_candidate_maximum_rank": int(
            configuration["cross_plus_candidate_maximum_rank"]
        ),
        "cross_plus_required_stroke_count": int(
            configuration["cross_plus_required_stroke_count"]
        ),
        "cross_plus_maximum_absolute_aspect_log": float(
            configuration["cross_plus_maximum_absolute_aspect_log"]
        ),
        "cross_plus_minimum_path_over_diagonal": float(
            configuration["cross_plus_minimum_path_over_diagonal"]
        ),
        "cross_plus_maximum_path_over_diagonal": float(
            configuration["cross_plus_maximum_path_over_diagonal"]
        ),
        "vertical_one_candidate_maximum_rank": int(
            configuration["vertical_one_candidate_maximum_rank"]
        ),
        "vertical_one_required_stroke_count": int(
            configuration["vertical_one_required_stroke_count"]
        ),
        "vertical_one_maximum_aspect_log": float(
            configuration["vertical_one_maximum_aspect_log"]
        ),
        "vertical_one_maximum_path_over_diagonal": float(
            configuration["vertical_one_maximum_path_over_diagonal"]
        ),
        "vertical_one_minimum_absolute_direction_y": float(
            configuration["vertical_one_minimum_absolute_direction_y"]
        ),
        "vertical_one_maximum_relative_width": float(
            configuration["vertical_one_maximum_relative_width"]
        ),
    }
    if output["candidate_width"] != 20:
        raise ValueError("formula-role typo rescue requires Top-20 evidence")
    if not 0.0 <= output["auxiliary_weight"] <= 1.0:
        raise ValueError("formula-role typo auxiliary weight is invalid")
    policies = output["allowed_auxiliary_policies"]
    if not policies or len(policies) != len(set(policies)) or any(
        not value for value in policies
    ):
        raise ValueError("formula-role typo candidate policies are invalid")
    if output["function_tokens"] != ["f", "g", "h"]:
        raise ValueError("formula-role typo function token contract is invalid")
    if output["parenthesized_variables"] != ["x", "y", "z"]:
        raise ValueError("formula-role typo variable contract is invalid")
    ranks = (
        output["function_candidate_maximum_rank"],
        output["parenthesized_variable_maximum_rank"],
        output["cross_plus_candidate_maximum_rank"],
        output["vertical_one_candidate_maximum_rank"],
    )
    if any(not 1 <= value <= output["candidate_width"] for value in ranks):
        raise ValueError("formula-role typo candidate rank is invalid")
    numeric = (
        output["cross_plus_maximum_absolute_aspect_log"],
        output["cross_plus_minimum_path_over_diagonal"],
        output["cross_plus_maximum_path_over_diagonal"],
        output["vertical_one_maximum_aspect_log"],
        output["vertical_one_maximum_path_over_diagonal"],
        output["vertical_one_minimum_absolute_direction_y"],
        output["vertical_one_maximum_relative_width"],
    )
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError("formula-role typo numeric configuration is not finite")
    if (
        output["cross_plus_required_stroke_count"] != 2
        or output["cross_plus_maximum_absolute_aspect_log"] < 0.0
        or not 1.0 <= output["cross_plus_minimum_path_over_diagonal"]
        <= output["cross_plus_maximum_path_over_diagonal"]
        or output["vertical_one_required_stroke_count"] != 1
        or output["vertical_one_maximum_aspect_log"] >= 0.0
        or output["vertical_one_maximum_path_over_diagonal"] < 1.0
        or not 0.0 <= output["vertical_one_minimum_absolute_direction_y"] <= 1.0
        or not 0.0 <= output["vertical_one_maximum_relative_width"] <= 1.0
    ):
        raise ValueError("formula-role typo geometry contract is invalid")
    if (
        type(configuration["digit_before_binary_top1_only"]) is not bool
        or configuration["digit_before_binary_top1_only"] is not True
        or type(configuration["function_rhs_top1_only"]) is not bool
        or configuration["function_rhs_top1_only"] is not True
        or type(configuration["value_cross_plus_open_fence_enabled"]) is not bool
    ):
        raise ValueError("formula-role typo Top-1 contract is invalid")
    return output


def _ranked_token(candidate: dict, allowed: set[str], maximum_rank: int) -> str | None:
    return next(
        (
            token for token in candidate["tokens"][:maximum_rank]
            if token in allowed
        ),
        None,
    )


def _geometry(candidate: dict, key: str) -> float:
    try:
        value = float(candidate["geometry"][key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"formula-role typo geometry is invalid: {candidate['record_id']} {key}"
        ) from error
    if not math.isfinite(value):
        raise ValueError(
            f"formula-role typo geometry is not finite: {candidate['record_id']} {key}"
        )
    return value


def _cross_passes(candidate: dict, configuration: dict) -> bool:
    return bool(
        int(round(_geometry(candidate, "stroke_count")))
        == configuration["cross_plus_required_stroke_count"]
        and abs(_geometry(candidate, "aspect_log"))
        <= configuration["cross_plus_maximum_absolute_aspect_log"]
        and configuration["cross_plus_minimum_path_over_diagonal"]
        <= _geometry(candidate, "path_over_diag")
        <= configuration["cross_plus_maximum_path_over_diagonal"]
    )


def _vertical_one_passes(candidate: dict, configuration: dict) -> bool:
    return bool(
        int(round(_geometry(candidate, "stroke_count")))
        == configuration["vertical_one_required_stroke_count"]
        and _geometry(candidate, "aspect_log")
        <= configuration["vertical_one_maximum_aspect_log"]
        and _geometry(candidate, "path_over_diag")
        <= configuration["vertical_one_maximum_path_over_diagonal"]
        and abs(_geometry(candidate, "direction_y"))
        >= configuration["vertical_one_minimum_absolute_direction_y"]
        and _geometry(candidate, "width_rel")
        <= configuration["vertical_one_maximum_relative_width"]
    )


def _balanced_function_call(tokens: list[str], functions: set[str]) -> bool:
    if len(tokens) < 4 or tokens[0] not in functions or tokens[1] != "(" or tokens[-1] != ")":
        return False
    depth = 0
    for token in tokens[1:]:
        if token == "(":
            depth += 1
        elif token == ")":
            depth -= 1
        if depth < 0:
            return False
    return depth == 0


def apply_formula_role_typo_rescue(
    baseline_rows: list[dict], candidate_rows: list[dict],
    configuration: dict | None = None,
) -> tuple[list[dict], dict]:
    configuration = validate_configuration(configuration or DEFAULT_CONFIGURATION)
    candidates = _validated_candidate_map(candidate_rows, configuration)
    baseline_ids = [str(row["record_id"]) for row in baseline_rows]
    if not baseline_ids or len(baseline_ids) != len(set(baseline_ids)):
        raise ValueError("formula-role typo baseline IDs are invalid")
    if set(baseline_ids) != set(candidates):
        raise ValueError("formula-role typo candidate coverage mismatch")
    selected = {
        str(row["record_id"]): str(row["finalized_top1"])
        for row in baseline_rows
    }
    changes: list[dict] = []
    changed_records: set[str] = set()
    skipped = Counter()
    functions = set(configuration["function_tokens"])
    variables = set(configuration["parenthesized_variables"])
    ascii_letters = set(string.ascii_letters)
    value_tokens = set(DIGITS) | ascii_letters | {
        ")", "]", "}", r"\}",
    }
    open_fences = {"(", "[", "{", r"\{"}

    def replace(row: dict, token: str, rule: str) -> None:
        record_id = str(row["record_id"])
        before = selected[record_id]
        if before == token:
            return
        candidate = candidates[record_id]
        if token not in candidate["tokens"]:
            raise AssertionError("formula-role typo rescue invented a token")
        if record_id in changed_records:
            raise AssertionError("formula-role typo rules changed one record twice")
        changed_records.add(record_id)
        selected[record_id] = token
        index = candidate["tokens"].index(token)
        changes.append({
            "formula_id": str(row["formula_id"]),
            "record_id": record_id,
            "context_index": int(row["context_index"]),
            "before": before,
            "after": token,
            "candidate_rank": index + 1,
            "candidate_probability": float(candidate["probabilities"][index]),
            "rule": rule,
        })

    for formula_id, sequence in _formulae(baseline_rows).items():
        tokens = [selected[str(row["record_id"])] for row in sequence]

        for index, row in enumerate(sequence[:-1]):
            candidate = candidates[str(row["record_id"])]
            top1 = candidate["tokens"][0]
            if (
                tokens[index + 1] in BINARY
                and top1 in DIGITS
                and tokens[index] not in DIGITS
            ):
                replace(row, top1, "fused_digit_top1_before_binary")
                tokens[index] = top1

        for index, row in enumerate(sequence[:-1]):
            current = tokens[index]
            nested_digit_call = bool(
                current in DIGITS
                and index >= 2
                and tokens[index - 1] == "("
                and tokens[0] in functions
            )
            if (
                tokens[index + 1] != "("
                or current in functions
                or current in ascii_letters
                or (current in DIGITS and not nested_digit_call)
            ):
                continue
            candidate = candidates[str(row["record_id"])]
            token = _ranked_token(
                candidate, functions,
                configuration["function_candidate_maximum_rank"],
            )
            if token is not None:
                replace(row, token, "function_call_head")
                tokens[index] = token

        for index in range(1, len(sequence) - 1):
            if (
                tokens[index - 1] != "("
                or tokens[index + 1] != ")"
                or tokens[index] in variables
                or tokens[index] in DIGITS
                or tokens[index] in ascii_letters
            ):
                continue
            row = sequence[index]
            candidate = candidates[str(row["record_id"])]
            token = _ranked_token(
                candidate, variables,
                configuration["parenthesized_variable_maximum_rank"],
            )
            if token is not None:
                replace(row, token, "parenthesized_single_variable")
                tokens[index] = token

        if (
            len(sequence) >= 6
            and tokens[-2] == "="
            and _balanced_function_call(tokens[:-2], functions)
            and tokens[-1] not in ascii_letters
        ):
            row = sequence[-1]
            candidate = candidates[str(row["record_id"])]
            top1 = candidate["tokens"][0]
            if top1 in ascii_letters:
                replace(row, top1, "function_equation_rhs_top1")
                tokens[-1] = top1

        for index in range(1, len(sequence) - 1):
            row = sequence[index]
            candidate = candidates[str(row["record_id"])]
            if (
                tokens[index - 1] in BINARY
                and tokens[index + 1] == ")"
                and "1" in candidate["tokens"][
                    :configuration["vertical_one_candidate_maximum_rank"]
                ]
                and _vertical_one_passes(candidate, configuration)
            ):
                replace(row, "1", "binary_vertical_one_before_closer")
                tokens[index] = "1"

        for index in range(1, len(sequence) - 2):
            plus_row = sequence[index]
            one_row = sequence[index + 1]
            plus_candidate = candidates[str(plus_row["record_id"])]
            one_candidate = candidates[str(one_row["record_id"])]
            if (
                tokens[index - 1] in value_tokens
                and tokens[index + 2] == ")"
                and "+" in plus_candidate["tokens"][
                    :configuration["cross_plus_candidate_maximum_rank"]
                ]
                and "1" in one_candidate["tokens"][
                    :configuration["vertical_one_candidate_maximum_rank"]
                ]
                and _cross_passes(plus_candidate, configuration)
                and _vertical_one_passes(one_candidate, configuration)
            ):
                replace(plus_row, "+", "paired_cross_plus_vertical_one")
                replace(one_row, "1", "paired_cross_plus_vertical_one")
                tokens[index:index + 2] = ["+", "1"]

        for index in range(1, len(sequence) - 1):
            row = sequence[index]
            candidate = candidates[str(row["record_id"])]
            if (
                tokens[index] not in BINARY
                and tokens[index - 1] in value_tokens
                and tokens[index + 1] in value_tokens
                and "+" in candidate["tokens"][
                    :configuration["cross_plus_candidate_maximum_rank"]
                ]
                and _cross_passes(candidate, configuration)
            ):
                replace(row, "+", "value_cross_plus_value")
                tokens[index] = "+"

        if configuration["value_cross_plus_open_fence_enabled"]:
            for index in range(1, len(sequence) - 1):
                row = sequence[index]
                candidate = candidates[str(row["record_id"])]
                if (
                    tokens[index] not in BINARY
                    and tokens[index - 1] in value_tokens
                    and tokens[index + 1] in open_fences
                    and "+" in candidate["tokens"][
                        :configuration["cross_plus_candidate_maximum_rank"]
                    ]
                    and _cross_passes(candidate, configuration)
                ):
                    replace(row, "+", "value_cross_plus_open_fence")
                    tokens[index] = "+"

        if any(
            str(row["formula_id"]) != formula_id for row in sequence
        ):
            raise AssertionError("formula-role typo formula identity mismatch")

    output = []
    for row in baseline_rows:
        record_id = str(row["record_id"])
        token = selected[record_id]
        changed = token != str(row["finalized_top1"])
        output.append({
            **row,
            "schema": OUTPUT_SCHEMA,
            "baseline_top1_before_formula_role_typo_rescue": str(
                row["finalized_top1"]
            ),
            "finalized_top1": token,
            "changed_by_formula_role_typo_rescue": changed,
            "decision_source": (
                "formula_role_typo_rescue_v1"
                if changed else str(row.get("decision_source", "baseline"))
            ),
        })
    outside = sum(
        str(row["finalized_top1"])
        not in candidates[str(row["record_id"])]["tokens"]
        for row in output if row["changed_by_formula_role_typo_rescue"]
    )
    if outside:
        raise AssertionError("formula-role typo output left the candidate set")
    return output, {
        "schema": SCHEMA,
        "status": "shadow_runtime_only",
        "configuration": configuration,
        "records": len(output),
        "formulas": len(_formulae(baseline_rows)),
        "changed_glyphs": len(changes),
        "changed_formulas": len({change["formula_id"] for change in changes}),
        "changes": changes,
        "changes_by_rule": dict(sorted(Counter(
            change["rule"] for change in changes
        ).items())),
        "skipped": dict(sorted(skipped.items())),
        "candidate_preservation_rate": 1.0,
        "tokens_outside_candidate_union": 0,
        "inserted_or_deleted_glyphs": 0,
        "glyph_order_mutations": 0,
        "grouping_mutations": 0,
        "layout_mutations": 0,
        "arithmetic_evaluation": False,
        "target_label_or_glyph_count_input": False,
    }


def _self_test() -> None:
    formulas = {
        "nested": [r"\fint", "(", "9", "(", r"\between", ")", ")"],
        "paired": ["(", "x", r"\Psi", r"\prime", ")", r"\times", "2"],
        "nested-plus": ["(", "x", "4", "(", "y", ")", ")"],
    }
    baseline = []
    candidates = []
    replacements = {
        ("nested", 0): ["f"],
        ("nested", 2): ["g"],
        ("nested", 4): ["x"],
        ("paired", 2): ["+"],
        ("paired", 3): ["1"],
        ("nested-plus", 2): ["+"],
    }
    for formula_id, tokens in formulas.items():
        for index, token in enumerate(tokens):
            record_id = f"{formula_id}-{index}"
            baseline.append({
                "record_id": record_id, "formula_id": formula_id,
                "context_index": index, "finalized_top1": token,
            })
            additions = replacements.get((formula_id, index), [])
            topk = [token] + additions
            topk.extend(
                f"z{index}_{offset}" for offset in range(20 - len(topk))
            )
            candidates.append({
                "record_id": record_id, "formula_id": formula_id,
                "final_topk": topk,
                "final_topk_probabilities": [
                    1.0 - offset * 0.04 for offset in range(20)
                ],
                "hwr_policy": "old_new_product_probability_fusion",
                "hwr_fusion_weight": 0.6,
                "geometry": {
                    "stroke_count": 2.0 if (formula_id, index) in {
                        ("paired", 2), ("nested-plus", 2),
                    } else 1.0,
                    "aspect_log": -1.8 if (formula_id, index) == ("paired", 3) else 0.0,
                    "path_over_diag": 1.02 if (formula_id, index) == ("paired", 3) else 1.5,
                    "direction_y": 0.98 if (formula_id, index) == ("paired", 3) else 0.0,
                    "width_rel": 0.03,
                },
            })
    output, audit = apply_formula_role_typo_rescue(baseline, candidates)
    by_formula = _formulae(output)
    assert [row["finalized_top1"] for row in by_formula["nested"]] == [
        "f", "(", "g", "(", "x", ")", ")",
    ]
    assert [row["finalized_top1"] for row in by_formula["paired"]] == [
        "(", "x", "+", "1", ")", r"\times", "2",
    ]
    assert [row["finalized_top1"] for row in by_formula["nested-plus"]] == [
        "(", "x", "+", "(", "y", ")", ")",
    ]
    legacy_configuration = {
        key: value for key, value in DEFAULT_CONFIGURATION.items()
        if key != "value_cross_plus_open_fence_enabled"
    }
    legacy_output, _ = apply_formula_role_typo_rescue(
        baseline, candidates, legacy_configuration,
    )
    assert [
        row["finalized_top1"] for row in _formulae(legacy_output)["nested-plus"]
    ] == ["(", "x", "4", "(", "y", ")", ")"]
    assert audit["changed_glyphs"] == 6
    assert audit["candidate_preservation_rate"] == 1.0
    assert audit["arithmetic_evaluation"] is False


if __name__ == "__main__":
    _self_test()
    print('{"self_test":"pass"}')
