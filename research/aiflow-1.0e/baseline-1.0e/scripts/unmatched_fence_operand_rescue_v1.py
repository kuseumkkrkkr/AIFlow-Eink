#!/usr/bin/env python3
"""Rescue a straight `1` occupying an impossible unmatched closer slot.

The rule is candidate preserving and placement only.  It does not insert,
delete, reorder, regroup, evaluate arithmetic, or alter layout relations.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import math

from wide_candidate_syntax_rescue_v1 import BINARY, DIGITS, RELATIONS, valid_numeric_sequence


SCHEMA = "aiflow-unmatched-fence-operand-rescue/v1"
OUTPUT_SCHEMA = "aiflow-unmatched-fence-operand-finalized/v1"
CLOSER_TO_OPENER = {
    ")": "(",
    "]": "[",
    r"\}": r"\{",
    r"\rangle": r"\langle",
    r"\rceil": r"\lceil",
    r"\rfloor": r"\lfloor",
    r"\rrbracket": r"\llbracket",
}
DEFAULT_CONFIGURATION = {
    "candidate_width": 20,
    "auxiliary_weight": 0.6,
    "allowed_auxiliary_policies": [
        "current_expanded_writer_loo_probability_fusion",
        "old_new_product_probability_fusion",
    ],
    "replacement": "1",
    "minimum_closer_prefix": 5,
    "minimum_existing_digit_count": 2,
    "maximum_aspect_log": -1.5,
    "maximum_path_over_diagonal": 1.15,
    "minimum_absolute_vertical_direction": 0.9,
    "maximum_relative_width": 0.08,
    "required_stroke_count": 1,
}


def validate_configuration(configuration: dict) -> dict:
    if set(configuration) != set(DEFAULT_CONFIGURATION):
        raise ValueError("unmatched-fence configuration fields mismatch")
    output = {
        **configuration,
        "candidate_width": int(configuration["candidate_width"]),
        "auxiliary_weight": float(configuration["auxiliary_weight"]),
        "allowed_auxiliary_policies": [
            str(value) for value in configuration["allowed_auxiliary_policies"]
        ],
        "replacement": str(configuration["replacement"]),
        "minimum_closer_prefix": int(configuration["minimum_closer_prefix"]),
        "minimum_existing_digit_count": int(
            configuration["minimum_existing_digit_count"]
        ),
        "maximum_aspect_log": float(configuration["maximum_aspect_log"]),
        "maximum_path_over_diagonal": float(
            configuration["maximum_path_over_diagonal"]
        ),
        "minimum_absolute_vertical_direction": float(
            configuration["minimum_absolute_vertical_direction"]
        ),
        "maximum_relative_width": float(configuration["maximum_relative_width"]),
        "required_stroke_count": int(configuration["required_stroke_count"]),
    }
    numeric = (
        output["auxiliary_weight"], output["maximum_aspect_log"],
        output["maximum_path_over_diagonal"],
        output["minimum_absolute_vertical_direction"],
        output["maximum_relative_width"],
    )
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError("unmatched-fence numeric configuration is not finite")
    if output["candidate_width"] != 20:
        raise ValueError("unmatched-fence rescue requires Top-20 evidence")
    if output["replacement"] != "1":
        raise ValueError("unmatched-fence rescue is restricted to straight 1")
    if not 1 <= output["minimum_closer_prefix"] <= 20:
        raise ValueError("unmatched-fence closer prefix is invalid")
    if output["minimum_existing_digit_count"] < 2:
        raise ValueError("unmatched-fence numeric intent is too weak")
    if not 0.0 <= output["auxiliary_weight"] <= 1.0:
        raise ValueError("unmatched-fence auxiliary weight is invalid")
    if not 0.0 <= output["minimum_absolute_vertical_direction"] <= 1.0:
        raise ValueError("unmatched-fence vertical direction is invalid")
    if output["maximum_path_over_diagonal"] < 1.0:
        raise ValueError("unmatched-fence path ratio is invalid")
    if output["maximum_relative_width"] <= 0.0:
        raise ValueError("unmatched-fence relative width is invalid")
    if output["required_stroke_count"] != 1:
        raise ValueError("unmatched-fence rescue requires one stroke")
    policies = output["allowed_auxiliary_policies"]
    if not policies or len(policies) != len(set(policies)) or any(not value for value in policies):
        raise ValueError("unmatched-fence policies are invalid")
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
            raise ValueError(f"unmatched-fence candidate identity is invalid: {record_id}")
        if (
            len(tokens) != configuration["candidate_width"]
            or len(tokens) != len(set(tokens))
            or len(probabilities) != len(tokens)
            or any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in probabilities)
            or any(left < right for left, right in zip(probabilities, probabilities[1:]))
        ):
            raise ValueError(f"unmatched-fence candidate array is invalid: {record_id}")
        if policy not in configuration["allowed_auxiliary_policies"]:
            raise ValueError(f"unmatched-fence candidate policy is invalid: {record_id}")
        if not math.isclose(weight, configuration["auxiliary_weight"], abs_tol=1e-12):
            raise ValueError(f"unmatched-fence candidate weight is invalid: {record_id}")
        output[record_id] = {**row, "tokens": tokens, "probabilities": probabilities}
    return output


def _formulae(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["formula_id"])].append(row)
    for formula_id, sequence in grouped.items():
        sequence.sort(key=lambda row: int(row["context_index"]))
        if [int(row["context_index"]) for row in sequence] != list(range(len(sequence))):
            raise ValueError(f"unmatched-fence formula order is invalid: {formula_id}")
    return dict(grouped)


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


def _eligible(
    tokens: list[str], index: int, candidate: dict, configuration: dict,
) -> tuple[bool, str]:
    before = tokens[index]
    opener = CLOSER_TO_OPENER.get(before)
    if opener is None or candidate["tokens"][0] != before:
        return False, "not_top1_closer"
    prefix = candidate["tokens"][:configuration["minimum_closer_prefix"]]
    if any(token not in CLOSER_TO_OPENER for token in prefix):
        return False, "closer_prefix_not_unanimous"
    if opener in tokens[:index]:
        return False, "possible_matching_opener"
    if index == 0 or index + 1 >= len(tokens):
        return False, "not_between_structure_tokens"
    if tokens[index - 1] not in BINARY or tokens[index + 1] not in RELATIONS:
        return False, "not_binary_to_relation_operand_slot"
    if sum(token in DIGITS for token in tokens) < configuration[
        "minimum_existing_digit_count"
    ]:
        return False, "insufficient_numeric_intent"
    digit_candidates = [token for token in candidate["tokens"] if token in DIGITS]
    if digit_candidates != [configuration["replacement"]]:
        return False, "replacement_not_unique_digit_candidate"
    if not _geometry_passes(candidate, configuration):
        return False, "straight_single_stroke_geometry_failed"
    repaired = list(tokens)
    repaired[index] = configuration["replacement"]
    if not valid_numeric_sequence(repaired):
        return False, "replacement_not_valid_numeric_sequence"
    return True, "eligible"


def apply_unmatched_fence_operand_rescue(
    baseline_rows: list[dict], candidate_rows: list[dict],
    configuration: dict | None = None,
) -> tuple[list[dict], dict]:
    configuration = validate_configuration(configuration or DEFAULT_CONFIGURATION)
    candidates = _candidate_map(candidate_rows, configuration)
    baseline_ids = [str(row["record_id"]) for row in baseline_rows]
    if not baseline_ids or len(baseline_ids) != len(set(baseline_ids)):
        raise ValueError("unmatched-fence baseline IDs are invalid")
    if set(baseline_ids) != set(candidates):
        raise ValueError("unmatched-fence coverage mismatch")
    selected = {
        str(row["record_id"]): str(row["finalized_top1"])
        for row in baseline_rows
    }
    changes = []
    skipped = Counter()
    for formula_id, sequence in _formulae(baseline_rows).items():
        before = [str(row["finalized_top1"]) for row in sequence]
        eligible = []
        for index, row in enumerate(sequence):
            record_id = str(row["record_id"])
            candidate = candidates[record_id]
            if str(candidate.get("formula_id", "")) != formula_id:
                raise ValueError(f"unmatched-fence formula identity mismatch: {record_id}")
            accepted, reason = _eligible(before, index, candidate, configuration)
            if accepted:
                eligible.append((index, row, candidate))
            elif before[index] in CLOSER_TO_OPENER:
                skipped[reason] += 1
        if len(eligible) != 1:
            if len(eligible) > 1:
                skipped["ambiguous_multiple_formula_repairs"] += 1
            continue
        index, row, candidate = eligible[0]
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
            "rule": "unmatched_closer_in_numeric_operand_slot_with_straight_1",
        })
    output = []
    for row in baseline_rows:
        record_id = str(row["record_id"])
        token = selected[record_id]
        output.append({
            **row,
            "schema": OUTPUT_SCHEMA,
            "baseline_top1_before_unmatched_fence_rescue": str(row["finalized_top1"]),
            "finalized_top1": token,
            "changed_by_unmatched_fence_rescue": token != str(row["finalized_top1"]),
            "decision_source": (
                "unmatched_fence_operand_rescue_v1"
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
        "skipped": dict(sorted(skipped.items())),
        "candidate_preservation_rate": 1.0,
        "inserted_or_deleted_glyphs": 0,
        "glyph_order_mutations": 0,
        "grouping_mutations": 0,
        "layout_mutations": 0,
        "arithmetic_evaluation": False,
    }


def _self_test() -> None:
    def candidate(index: int, top1: str, *, straight: bool = True) -> dict:
        closers = [")", "]", r"\}", r"\rangle", r"\rfloor"]
        tokens = [top1] if top1 not in closers else [top1] + [
            token for token in closers if token != top1
        ]
        tokens += ["1"] if "1" not in tokens else []
        tokens += [f"z{index}_{offset}" for offset in range(20 - len(tokens))]
        return {
            "record_id": f"r{index}", "formula_id": "f",
            "final_topk": tokens,
            "final_topk_probabilities": [1.0] + [0.0] * 19,
            "hwr_policy": "old_new_product_probability_fusion",
            "hwr_fusion_weight": 0.6,
            "geometry": {
                "aspect_log": -2.0,
                "path_over_diag": 1.05 if straight else 1.5,
                "direction_y": 0.98,
                "stroke_count": 1.0,
                "width_rel": 0.03,
            },
        }

    before = ["5", r"\times", ")", "=", "5"]
    baseline = [
        {"record_id": f"r{index}", "formula_id": "f", "context_index": index,
         "finalized_top1": token}
        for index, token in enumerate(before)
    ]
    candidates = [candidate(index, token) for index, token in enumerate(before)]
    output, audit = apply_unmatched_fence_operand_rescue(baseline, candidates)
    assert [row["finalized_top1"] for row in output] == ["5", r"\times", "1", "=", "5"]
    assert audit["changed_glyphs"] == 1 and not audit["arithmetic_evaluation"]
    balanced_before = ["(", "5", r"\times", ")", "=", "5"]
    balanced = [
        {"record_id": f"b{index}", "formula_id": "balanced", "context_index": index,
         "finalized_top1": token}
        for index, token in enumerate(balanced_before)
    ]
    balanced_candidates = []
    for index, token in enumerate(balanced_before):
        source = candidate(index, token)
        source["record_id"] = f"b{index}"
        source["formula_id"] = "balanced"
        balanced_candidates.append(source)
    output, audit = apply_unmatched_fence_operand_rescue(balanced, balanced_candidates)
    assert [row["finalized_top1"] for row in output] == balanced_before
    assert audit["changed_glyphs"] == 0


if __name__ == "__main__":
    _self_test()
    print('{"self_test":"pass"}')
