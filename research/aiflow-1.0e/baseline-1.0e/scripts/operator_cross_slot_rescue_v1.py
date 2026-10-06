#!/usr/bin/env python3
"""Rescue a plus-shaped cross from an impossible `digit bot digit` slot."""

from __future__ import annotations

from collections import defaultdict
import math

from wide_candidate_syntax_rescue_v1 import BINARY, DIGITS, valid_numeric_sequence


SCHEMA = "aiflow-operator-cross-slot-rescue/v1"
OUTPUT_SCHEMA = "aiflow-operator-cross-slot-finalized/v1"
DEFAULT_CONFIGURATION = {
    "candidate_width": 20,
    "auxiliary_weight": 0.6,
    "allowed_auxiliary_policies": [
        "current_expanded_writer_loo_probability_fusion",
        "old_new_product_probability_fusion",
    ],
    "current_token": r"\bot",
    "required_candidate_prefix": [r"\bot", r"\perp"],
    "replacement": "+",
    "required_formula_length": 3,
    "required_stroke_count": 2,
    "maximum_absolute_aspect_log": 0.3,
    "minimum_path_over_diagonal": 1.2,
    "maximum_path_over_diagonal": 1.8,
}


def validate_configuration(configuration: dict) -> dict:
    if set(configuration) != set(DEFAULT_CONFIGURATION):
        raise ValueError("operator-cross configuration fields mismatch")
    output = {
        **configuration,
        "candidate_width": int(configuration["candidate_width"]),
        "auxiliary_weight": float(configuration["auxiliary_weight"]),
        "allowed_auxiliary_policies": [
            str(value) for value in configuration["allowed_auxiliary_policies"]
        ],
        "current_token": str(configuration["current_token"]),
        "required_candidate_prefix": [
            str(value) for value in configuration["required_candidate_prefix"]
        ],
        "replacement": str(configuration["replacement"]),
        "required_formula_length": int(configuration["required_formula_length"]),
        "required_stroke_count": int(configuration["required_stroke_count"]),
        "maximum_absolute_aspect_log": float(
            configuration["maximum_absolute_aspect_log"]
        ),
        "minimum_path_over_diagonal": float(
            configuration["minimum_path_over_diagonal"]
        ),
        "maximum_path_over_diagonal": float(
            configuration["maximum_path_over_diagonal"]
        ),
    }
    numeric = (
        output["auxiliary_weight"], output["maximum_absolute_aspect_log"],
        output["minimum_path_over_diagonal"], output["maximum_path_over_diagonal"],
    )
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError("operator-cross numeric configuration is not finite")
    if output["candidate_width"] != 20:
        raise ValueError("operator-cross rescue requires Top-20 evidence")
    if output["current_token"] != r"\bot":
        raise ValueError("operator-cross current token must be bot")
    if output["required_candidate_prefix"] != [r"\bot", r"\perp"]:
        raise ValueError("operator-cross prefix contract is invalid")
    if output["replacement"] != "+":
        raise ValueError("operator-cross replacement must be plus")
    if output["required_formula_length"] != 3 or output["required_stroke_count"] != 2:
        raise ValueError("operator-cross structural contract is invalid")
    if not 0.0 <= output["auxiliary_weight"] <= 1.0:
        raise ValueError("operator-cross auxiliary weight is invalid")
    if output["maximum_absolute_aspect_log"] < 0.0:
        raise ValueError("operator-cross aspect limit is invalid")
    if not 1.0 <= output["minimum_path_over_diagonal"] <= output[
        "maximum_path_over_diagonal"
    ]:
        raise ValueError("operator-cross path limits are invalid")
    policies = output["allowed_auxiliary_policies"]
    if not policies or len(policies) != len(set(policies)) or any(not value for value in policies):
        raise ValueError("operator-cross candidate policies are invalid")
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
            raise ValueError(f"operator-cross candidate identity is invalid: {record_id}")
        if (
            len(tokens) != configuration["candidate_width"]
            or len(tokens) != len(set(tokens))
            or len(probabilities) != len(tokens)
            or any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in probabilities)
            or any(left < right for left, right in zip(probabilities, probabilities[1:]))
        ):
            raise ValueError(f"operator-cross candidate array is invalid: {record_id}")
        if policy not in configuration["allowed_auxiliary_policies"]:
            raise ValueError(f"operator-cross candidate policy is invalid: {record_id}")
        if not math.isclose(weight, configuration["auxiliary_weight"], abs_tol=1e-12):
            raise ValueError(f"operator-cross candidate weight is invalid: {record_id}")
        output[record_id] = {**row, "tokens": tokens, "probabilities": probabilities}
    return output


def _formulae(rows: list[dict]) -> dict[str, list[dict]]:
    output: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        output[str(row["formula_id"])].append(row)
    for formula_id, sequence in output.items():
        sequence.sort(key=lambda row: int(row["context_index"]))
        if [int(row["context_index"]) for row in sequence] != list(range(len(sequence))):
            raise ValueError(f"operator-cross formula order is invalid: {formula_id}")
    return dict(output)


def _geometry_passes(candidate: dict, configuration: dict) -> bool:
    geometry = candidate.get("geometry", {})
    try:
        path_ratio = float(geometry["path_over_diag"])
        return bool(
            int(round(float(geometry["stroke_count"])))
            == configuration["required_stroke_count"]
            and abs(float(geometry["aspect_log"]))
            <= configuration["maximum_absolute_aspect_log"]
            and configuration["minimum_path_over_diagonal"]
            <= path_ratio <= configuration["maximum_path_over_diagonal"]
        )
    except (KeyError, TypeError, ValueError):
        return False


def apply_operator_cross_slot_rescue(
    baseline_rows: list[dict], candidate_rows: list[dict],
    configuration: dict | None = None,
) -> tuple[list[dict], dict]:
    configuration = validate_configuration(configuration or DEFAULT_CONFIGURATION)
    candidates = _candidate_map(candidate_rows, configuration)
    baseline_ids = [str(row["record_id"]) for row in baseline_rows]
    if not baseline_ids or len(baseline_ids) != len(set(baseline_ids)):
        raise ValueError("operator-cross baseline IDs are invalid")
    if set(baseline_ids) != set(candidates):
        raise ValueError("operator-cross coverage mismatch")
    selected = {
        str(row["record_id"]): str(row["finalized_top1"])
        for row in baseline_rows
    }
    changes = []
    for formula_id, sequence in _formulae(baseline_rows).items():
        before = [str(row["finalized_top1"]) for row in sequence]
        if (
            len(before) != configuration["required_formula_length"]
            or before[0] not in DIGITS
            or before[1] != configuration["current_token"]
            or before[2] not in DIGITS
        ):
            continue
        row = sequence[1]
        record_id = str(row["record_id"])
        candidate = candidates[record_id]
        if str(candidate.get("formula_id", "")) != formula_id:
            raise ValueError(f"operator-cross formula identity mismatch: {record_id}")
        if candidate["tokens"][:2] != configuration["required_candidate_prefix"]:
            continue
        arithmetic_candidates = [
            token for token in candidate["tokens"] if token in BINARY
        ]
        if arithmetic_candidates != [configuration["replacement"]]:
            continue
        if not _geometry_passes(candidate, configuration):
            continue
        repaired = [before[0], configuration["replacement"], before[2]]
        if not valid_numeric_sequence(repaired):
            continue
        replacement = configuration["replacement"]
        selected[record_id] = replacement
        candidate_index = candidate["tokens"].index(replacement)
        changes.append({
            "formula_id": formula_id,
            "record_id": record_id,
            "context_index": 1,
            "before": before[1],
            "after": replacement,
            "candidate_rank": candidate_index + 1,
            "candidate_probability": candidate["probabilities"][candidate_index],
            "geometry": {
                key: candidate["geometry"][key]
                for key in ("aspect_log", "path_over_diag", "stroke_count")
            },
            "rule": "digit_bot_digit_two_stroke_cross_unique_plus",
        })
    output = []
    for row in baseline_rows:
        record_id = str(row["record_id"])
        token = selected[record_id]
        output.append({
            **row,
            "schema": OUTPUT_SCHEMA,
            "baseline_top1_before_operator_cross_rescue": str(row["finalized_top1"]),
            "finalized_top1": token,
            "changed_by_operator_cross_rescue": token != str(row["finalized_top1"]),
            "decision_source": (
                "operator_cross_slot_rescue_v1"
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
    def candidate(index: int, token: str, *, square: bool = True) -> dict:
        if index == 1:
            tokens = [r"\bot", r"\perp", "+"] + [
                f"z{index}_{offset}" for offset in range(17)
            ]
        else:
            tokens = [token] + [f"z{index}_{offset}" for offset in range(19)]
        return {
            "record_id": f"r{index}", "formula_id": "f",
            "final_topk": tokens,
            "final_topk_probabilities": [1.0] + [0.0] * 19,
            "hwr_policy": "old_new_product_probability_fusion",
            "hwr_fusion_weight": 0.6,
            "geometry": {
                "aspect_log": 0.1 if square else 1.0,
                "path_over_diag": 1.4,
                "stroke_count": 2.0,
            },
        }

    before = ["1", r"\bot", "5"]
    baseline = [
        {"record_id": f"r{index}", "formula_id": "f", "context_index": index,
         "finalized_top1": token}
        for index, token in enumerate(before)
    ]
    candidates = [candidate(index, token) for index, token in enumerate(before)]
    output, audit = apply_operator_cross_slot_rescue(baseline, candidates)
    assert [row["finalized_top1"] for row in output] == ["1", "+", "5"]
    assert audit["changed_glyphs"] == 1 and not audit["arithmetic_evaluation"]
    candidates[1] = candidate(1, r"\bot", square=False)
    output, audit = apply_operator_cross_slot_rescue(baseline, candidates)
    assert [row["finalized_top1"] for row in output] == before
    assert audit["changed_glyphs"] == 0


if __name__ == "__main__":
    _self_test()
    print('{"self_test":"pass"}')
