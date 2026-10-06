#!/usr/bin/env python3
"""Rescue two straight `1` operands in one constrained numeric formula."""

from __future__ import annotations

from collections import defaultdict
import math

from wide_candidate_syntax_rescue_v1 import DIGITS, valid_numeric_sequence


SCHEMA = "aiflow-dual-straight-one-slot-rescue/v1"
OUTPUT_SCHEMA = "aiflow-dual-straight-one-slot-finalized/v1"
DEFAULT_CONFIGURATION = {
    "candidate_width": 20,
    "auxiliary_weight": 0.6,
    "allowed_auxiliary_policies": [
        "current_expanded_writer_loo_probability_fusion",
        "old_new_product_probability_fusion",
    ],
    "ambiguous_tokens": ["I", "/", "|", r"\mid", r"\prime", "l"],
    "replacement": "1",
    "required_formula_length": 5,
    "required_binary_operator": r"\div",
    "required_relation": "=",
    "required_change_count": 2,
    "required_stroke_count": 1,
    "maximum_aspect_log": -0.8,
    "maximum_path_over_diagonal": 1.1,
    "minimum_absolute_vertical_direction": 0.9,
    "maximum_relative_width": 0.25,
    "minimum_best_digit_probability_ratio": 10.0,
}


def validate_configuration(configuration: dict) -> dict:
    if set(configuration) != set(DEFAULT_CONFIGURATION):
        raise ValueError("dual-straight-one configuration fields mismatch")
    output = {
        **configuration,
        "candidate_width": int(configuration["candidate_width"]),
        "auxiliary_weight": float(configuration["auxiliary_weight"]),
        "allowed_auxiliary_policies": [
            str(value) for value in configuration["allowed_auxiliary_policies"]
        ],
        "ambiguous_tokens": [str(value) for value in configuration["ambiguous_tokens"]],
        "replacement": str(configuration["replacement"]),
        "required_formula_length": int(configuration["required_formula_length"]),
        "required_binary_operator": str(configuration["required_binary_operator"]),
        "required_relation": str(configuration["required_relation"]),
        "required_change_count": int(configuration["required_change_count"]),
        "required_stroke_count": int(configuration["required_stroke_count"]),
        "maximum_aspect_log": float(configuration["maximum_aspect_log"]),
        "maximum_path_over_diagonal": float(
            configuration["maximum_path_over_diagonal"]
        ),
        "minimum_absolute_vertical_direction": float(
            configuration["minimum_absolute_vertical_direction"]
        ),
        "maximum_relative_width": float(configuration["maximum_relative_width"]),
        "minimum_best_digit_probability_ratio": float(
            configuration["minimum_best_digit_probability_ratio"]
        ),
    }
    numeric = (
        output["auxiliary_weight"], output["maximum_aspect_log"],
        output["maximum_path_over_diagonal"],
        output["minimum_absolute_vertical_direction"],
        output["maximum_relative_width"],
        output["minimum_best_digit_probability_ratio"],
    )
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError("dual-straight-one numeric configuration is not finite")
    if output["candidate_width"] != 20:
        raise ValueError("dual-straight-one rescue requires Top-20 evidence")
    if output["replacement"] != "1":
        raise ValueError("dual-straight-one replacement must be 1")
    if (
        output["required_formula_length"] != 5
        or output["required_change_count"] != 2
        or output["required_stroke_count"] != 1
        or output["required_binary_operator"] != r"\div"
        or output["required_relation"] != "="
    ):
        raise ValueError("dual-straight-one structural contract is invalid")
    if set(output["ambiguous_tokens"]) != {"I", "/", "|", r"\mid", r"\prime", "l"}:
        raise ValueError("dual-straight-one ambiguous token contract is invalid")
    if not 0.0 <= output["auxiliary_weight"] <= 1.0:
        raise ValueError("dual-straight-one auxiliary weight is invalid")
    if output["maximum_path_over_diagonal"] < 1.0:
        raise ValueError("dual-straight-one path limit is invalid")
    if not 0.0 <= output["minimum_absolute_vertical_direction"] <= 1.0:
        raise ValueError("dual-straight-one direction limit is invalid")
    if output["maximum_relative_width"] <= 0.0:
        raise ValueError("dual-straight-one relative width is invalid")
    if output["minimum_best_digit_probability_ratio"] < 1.0:
        raise ValueError("dual-straight-one digit ratio is invalid")
    policies = output["allowed_auxiliary_policies"]
    if not policies or len(policies) != len(set(policies)) or any(not value for value in policies):
        raise ValueError("dual-straight-one candidate policies are invalid")
    return output


def _candidate_map(rows: list[dict], configuration: dict) -> dict[str, dict]:
    output = {}
    for row in rows:
        record_id = str(row.get("record_id", ""))
        tokens = [str(value) for value in row.get("final_topk", [])]
        probabilities = [float(value) for value in row.get("final_topk_probabilities", [])]
        policy = str(row.get("hwr_policy", ""))
        weight = float(row.get("hwr_fusion_weight", -1.0))
        if not record_id or record_id in output:
            raise ValueError(f"dual-straight-one identity is invalid: {record_id}")
        if (
            len(tokens) != configuration["candidate_width"]
            or len(tokens) != len(set(tokens))
            or len(probabilities) != len(tokens)
            or any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in probabilities)
            or any(left < right for left, right in zip(probabilities, probabilities[1:]))
        ):
            raise ValueError(f"dual-straight-one candidate array is invalid: {record_id}")
        if policy not in configuration["allowed_auxiliary_policies"]:
            raise ValueError(f"dual-straight-one candidate policy is invalid: {record_id}")
        if not math.isclose(weight, configuration["auxiliary_weight"], abs_tol=1e-12):
            raise ValueError(f"dual-straight-one candidate weight is invalid: {record_id}")
        output[record_id] = {**row, "tokens": tokens, "probabilities": probabilities}
    return output


def _formulae(rows: list[dict]) -> dict[str, list[dict]]:
    output: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        output[str(row["formula_id"])].append(row)
    for formula_id, sequence in output.items():
        sequence.sort(key=lambda row: int(row["context_index"]))
        if [int(row["context_index"]) for row in sequence] != list(range(len(sequence))):
            raise ValueError(f"dual-straight-one formula order is invalid: {formula_id}")
    return dict(output)


def _geometry_passes(candidate: dict, configuration: dict) -> bool:
    geometry = candidate.get("geometry", {})
    try:
        return bool(
            int(round(float(geometry["stroke_count"])))
            == configuration["required_stroke_count"]
            and float(geometry["aspect_log"]) <= configuration["maximum_aspect_log"]
            and float(geometry["path_over_diag"])
            <= configuration["maximum_path_over_diagonal"]
            and abs(float(geometry["direction_y"]))
            >= configuration["minimum_absolute_vertical_direction"]
            and float(geometry["width_rel"]) <= configuration["maximum_relative_width"]
        )
    except (KeyError, TypeError, ValueError):
        return False


def _best_digit_is_one(candidate: dict, configuration: dict) -> bool:
    ranked = [
        (probability, token)
        for token, probability in zip(
            candidate["tokens"], candidate["probabilities"], strict=True,
        )
        if token in DIGITS
    ]
    if not ranked:
        return False
    ranked.sort(reverse=True)
    if ranked[0][1] != configuration["replacement"]:
        return False
    return bool(
        len(ranked) == 1
        or ranked[0][0] >= configuration["minimum_best_digit_probability_ratio"]
        * max(ranked[1][0], 1e-12)
    )


def apply_dual_straight_one_slot_rescue(
    baseline_rows: list[dict], candidate_rows: list[dict],
    configuration: dict | None = None,
) -> tuple[list[dict], dict]:
    configuration = validate_configuration(configuration or DEFAULT_CONFIGURATION)
    candidates = _candidate_map(candidate_rows, configuration)
    baseline_ids = [str(row["record_id"]) for row in baseline_rows]
    if not baseline_ids or len(baseline_ids) != len(set(baseline_ids)):
        raise ValueError("dual-straight-one baseline IDs are invalid")
    if set(baseline_ids) != set(candidates):
        raise ValueError("dual-straight-one coverage mismatch")
    selected = {
        str(row["record_id"]): str(row["finalized_top1"])
        for row in baseline_rows
    }
    changes = []
    ambiguous = set(configuration["ambiguous_tokens"])
    for formula_id, sequence in _formulae(baseline_rows).items():
        before = [str(row["finalized_top1"]) for row in sequence]
        if (
            len(before) != configuration["required_formula_length"]
            or before[0] not in DIGITS
            or before[1] != configuration["required_binary_operator"]
            or before[2] not in ambiguous
            or before[3] != configuration["required_relation"]
            or before[4] not in ambiguous
        ):
            continue
        eligible = []
        for index in (2, 4):
            row = sequence[index]
            record_id = str(row["record_id"])
            candidate = candidates[record_id]
            if str(candidate.get("formula_id", "")) != formula_id:
                raise ValueError(f"dual-straight-one formula mismatch: {record_id}")
            if (
                before[index] not in candidate["tokens"]
                or configuration["replacement"] not in candidate["tokens"]
                or not _best_digit_is_one(candidate, configuration)
                or not _geometry_passes(candidate, configuration)
            ):
                break
            eligible.append((index, row, candidate))
        if len(eligible) != configuration["required_change_count"]:
            continue
        repaired = list(before)
        for index, _, _ in eligible:
            repaired[index] = configuration["replacement"]
        if not valid_numeric_sequence(repaired):
            continue
        for index, row, candidate in eligible:
            record_id = str(row["record_id"])
            replacement = configuration["replacement"]
            selected[record_id] = replacement
            candidate_index = candidate["tokens"].index(replacement)
            changes.append({
                "formula_id": formula_id,
                "record_id": record_id,
                "context_index": index,
                "before": before[index],
                "after": replacement,
                "candidate_rank": candidate_index + 1,
                "candidate_probability": candidate["probabilities"][candidate_index],
                "geometry": {
                    key: candidate["geometry"][key]
                    for key in (
                        "aspect_log", "path_over_diag", "direction_y",
                        "stroke_count", "width_rel",
                    )
                },
                "rule": "paired_straight_one_numeric_operand_slots",
            })
    output = []
    for row in baseline_rows:
        record_id = str(row["record_id"])
        token = selected[record_id]
        output.append({
            **row,
            "schema": OUTPUT_SCHEMA,
            "baseline_top1_before_dual_straight_one_rescue": str(row["finalized_top1"]),
            "finalized_top1": token,
            "changed_by_dual_straight_one_rescue": token != str(row["finalized_top1"]),
            "decision_source": (
                "dual_straight_one_slot_rescue_v1"
                if token != str(row["finalized_top1"])
                else str(row.get("decision_source", "baseline"))
            ),
        })
    return output, {
        "schema": SCHEMA,
        "status": "shadow_runtime_only",
        "configuration": configuration,
        "records": len(output),
        "formulas": len(_formulae(baseline_rows)),
        "changed_glyphs": len(changes),
        "changed_formulas": len({change["formula_id"] for change in changes}),
        "changes": changes,
        "candidate_preservation_rate": 1.0,
        "inserted_or_deleted_glyphs": 0,
        "glyph_order_mutations": 0,
        "grouping_mutations": 0,
        "layout_mutations": 0,
        "arithmetic_evaluation": False,
    }


def _self_test() -> None:
    before = ["1", r"\div", "I", "=", "/"]

    def candidate(index: int, token: str, *, second_digit_probability: float = 0.0) -> dict:
        if index in (2, 4):
            tokens = [token, "1", "7"] + [f"z{index}_{offset}" for offset in range(17)]
            probabilities = [0.8, 0.1, second_digit_probability] + [0.0] * 17
        else:
            tokens = [token] + [f"z{index}_{offset}" for offset in range(19)]
            probabilities = [1.0] + [0.0] * 19
        return {
            "record_id": f"r{index}", "formula_id": "f",
            "final_topk": tokens,
            "final_topk_probabilities": probabilities,
            "hwr_policy": "old_new_product_probability_fusion",
            "hwr_fusion_weight": 0.6,
            "geometry": {
                "aspect_log": -1.0,
                "path_over_diag": 1.05,
                "direction_y": 0.95,
                "stroke_count": 1.0,
                "width_rel": 0.2,
            },
        }

    baseline = [
        {"record_id": f"r{index}", "formula_id": "f", "context_index": index,
         "finalized_top1": token}
        for index, token in enumerate(before)
    ]
    candidates = [candidate(index, token) for index, token in enumerate(before)]
    output, audit = apply_dual_straight_one_slot_rescue(baseline, candidates)
    assert [row["finalized_top1"] for row in output] == ["1", r"\div", "1", "=", "1"]
    assert audit["changed_glyphs"] == 2 and not audit["arithmetic_evaluation"]
    candidates[4] = candidate(4, "/", second_digit_probability=0.02)
    output, audit = apply_dual_straight_one_slot_rescue(baseline, candidates)
    assert [row["finalized_top1"] for row in output] == before
    assert audit["changed_glyphs"] == 0


if __name__ == "__main__":
    _self_test()
    print('{"self_test":"pass"}')
