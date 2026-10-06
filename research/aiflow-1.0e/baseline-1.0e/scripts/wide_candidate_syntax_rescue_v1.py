#!/usr/bin/env python3
"""Repair one unique numeric syntax path from a research-only HWR Top-10.

The neural HWR and formula-context paths remain the proposal layers.  This
post-context placement guard never evaluates arithmetic and does not reorder,
insert, delete, regroup, or relayout glyphs.  It runs only when the baseline is
not valid numeric syntax and a single minimum-change repair exists.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from itertools import combinations
import gzip
import json
import math
from pathlib import Path

from character_tensor_v1 import _json_lines
from evaluate_48hz_prefix_v1 import _sha256


SCHEMA = "aiflow-wide-candidate-syntax-rescue/v1"
OUTPUT_SCHEMA = "aiflow-wide-candidate-syntax-finalized/v1"
CONFIG_SCHEMA = "aiflow-wide-candidate-syntax-rescue-runtime-config/v1"
DIGITS = frozenset("0123456789")
BINARY = frozenset({
    "+", "-", "/", r"\times", r"\div", r"\cdot", r"\pm", r"\mp",
})
RELATIONS = frozenset({"=", "<", ">", r"\leq", r"\geq", r"\neq"})
DEFAULT_CONFIGURATION = {
    "candidate_width": 10,
    "auxiliary_weight": 0.6,
    "allowed_auxiliary_policies": [
        "current_expanded_writer_loo_probability_fusion",
        "old_new_product_probability_fusion",
    ],
    "minimum_baseline_digit_count": 2,
    "maximum_changes": 2,
    "maximum_formula_glyphs": 64,
    "minimum_digit_probability_ratio": 0.01,
    "equality_candidate_must_be_top1": True,
    "literal_alpha_operand_lock": True,
}


def validate_configuration(configuration: dict) -> dict:
    optional = {"allow_equality_replacement"}
    if frozenset(configuration) not in {
        frozenset(DEFAULT_CONFIGURATION),
        frozenset(DEFAULT_CONFIGURATION) | optional,
    }:
        raise ValueError("wide syntax rescue configuration fields mismatch")
    width = int(configuration["candidate_width"])
    maximum = int(configuration["maximum_changes"])
    maximum_formula_glyphs = int(configuration["maximum_formula_glyphs"])
    minimum_digits = int(configuration["minimum_baseline_digit_count"])
    weight = float(configuration["auxiliary_weight"])
    ratio = float(configuration["minimum_digit_probability_ratio"])
    policies = [str(value) for value in configuration["allowed_auxiliary_policies"]]
    if width != 10:
        raise ValueError("wide syntax rescue v1 requires Top-10 evidence")
    if maximum not in {1, 2}:
        raise ValueError("wide syntax rescue permits one or two changes")
    if minimum_digits < 2:
        raise ValueError("wide syntax rescue requires at least two baseline digits")
    if not 3 <= maximum_formula_glyphs <= 128:
        raise ValueError("wide syntax rescue formula length limit is invalid")
    if not math.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError("wide syntax rescue auxiliary weight must be in [0,1]")
    if not math.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
        raise ValueError("wide syntax rescue probability ratio must be in [0,1]")
    if not policies or len(policies) != len(set(policies)) or any(not value for value in policies):
        raise ValueError("wide syntax rescue auxiliary policies are invalid")
    equality_top1 = configuration["equality_candidate_must_be_top1"]
    equality_replacement = configuration.get("allow_equality_replacement", True)
    literal_alpha_lock = configuration["literal_alpha_operand_lock"]
    if (
        type(equality_top1) is not bool
        or type(equality_replacement) is not bool
        or type(literal_alpha_lock) is not bool
    ):
        raise ValueError("wide syntax rescue lock fields must be boolean")
    return {
        "candidate_width": width,
        "auxiliary_weight": weight,
        "allowed_auxiliary_policies": policies,
        "minimum_baseline_digit_count": minimum_digits,
        "maximum_changes": maximum,
        "maximum_formula_glyphs": maximum_formula_glyphs,
        "minimum_digit_probability_ratio": ratio,
        "equality_candidate_must_be_top1": equality_top1,
        "allow_equality_replacement": equality_replacement,
        "literal_alpha_operand_lock": literal_alpha_lock,
    }


def _valid_side(tokens: list[str]) -> bool:
    if not tokens:
        return False
    expect_digit = True
    for token in tokens:
        if expect_digit:
            if token not in DIGITS:
                return False
            expect_digit = False
        elif token in DIGITS:
            continue
        elif token in BINARY:
            expect_digit = True
        else:
            return False
    return not expect_digit


def valid_numeric_sequence(tokens: list[str]) -> bool:
    positions = [index for index, token in enumerate(tokens) if token in RELATIONS]
    if len(positions) > 1:
        return False
    if not positions:
        return _valid_side(tokens)
    index = positions[0]
    return _valid_side(tokens[:index]) and _valid_side(tokens[index + 1:])


def _baseline_map(rows: list[dict]) -> dict[str, dict]:
    output = {}
    for source in rows:
        record_id = str(source.get("record_id", ""))
        formula_id = str(source.get("formula_id", ""))
        token = str(source.get("finalized_top1", ""))
        try:
            context_index = int(source["context_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"baseline context index is invalid: {record_id}") from exc
        if not record_id or not formula_id or not token or record_id in output:
            raise ValueError(f"baseline row identity is invalid: {record_id}")
        output[record_id] = {
            **source,
            "record_id": record_id,
            "formula_id": formula_id,
            "finalized_top1": token,
            "context_index": context_index,
        }
    return output


def _wide_map(rows: list[dict], configuration: dict) -> dict[str, dict]:
    output = {}
    for source in rows:
        record_id = str(source.get("record_id", ""))
        formula_id = str(source.get("formula_id", ""))
        tokens = [str(value) for value in source.get("final_topk", [])]
        probabilities = [float(value) for value in source.get("final_topk_probabilities", [])]
        context_index = int(source.get("context", {}).get("index", -1))
        policy = str(source.get("hwr_policy", ""))
        weight = float(source.get("hwr_fusion_weight", -1.0))
        if not record_id or not formula_id or record_id in output:
            raise ValueError(f"wide candidate identity is invalid: {record_id}")
        if (
            len(tokens) != configuration["candidate_width"]
            or len(tokens) != len(set(tokens))
            or len(probabilities) != len(tokens)
            or any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in probabilities)
            or any(left < right for left, right in zip(probabilities, probabilities[1:]))
        ):
            raise ValueError(f"wide candidate Top-10 is invalid: {record_id}")
        if policy not in configuration["allowed_auxiliary_policies"]:
            raise ValueError(f"wide candidate policy is not admitted: {record_id}")
        if not math.isclose(weight, configuration["auxiliary_weight"], abs_tol=1e-12):
            raise ValueError(f"wide candidate fusion weight mismatch: {record_id}")
        if context_index < 0:
            raise ValueError(f"wide candidate context index is invalid: {record_id}")
        output[record_id] = {
            "record_id": record_id,
            "formula_id": formula_id,
            "context_index": context_index,
            "tokens": tokens,
            "probabilities": probabilities,
            "policy": policy,
        }
    return output


def _formulae(rows: dict[str, dict]) -> dict[str, list[dict]]:
    output: dict[str, list[dict]] = defaultdict(list)
    for row in rows.values():
        output[row["formula_id"]].append(row)
    for formula_id, sequence in output.items():
        sequence.sort(key=lambda row: row["context_index"])
        if [row["context_index"] for row in sequence] != list(range(len(sequence))):
            raise ValueError(f"formula context indices are not contiguous: {formula_id}")
    return dict(output)


def _options(before: str, row: dict, configuration: dict) -> list[str]:
    options = [before]
    top_probability = max(row["probabilities"][0], 1e-12)
    literal_alpha_locked = (
        configuration["literal_alpha_operand_lock"]
        and len(before) == 1 and before.isalpha()
    )
    if before not in DIGITS | BINARY | RELATIONS and not literal_alpha_locked:
        digit_options = [
            (probability, token)
            for token, probability in zip(row["tokens"], row["probabilities"], strict=True)
            if token in DIGITS
            and probability / top_probability >= configuration["minimum_digit_probability_ratio"]
        ]
        if digit_options:
            options.append(max(digit_options)[1])
    equality_allowed = (
        configuration["allow_equality_replacement"]
        and "=" in row["tokens"]
        and before != "="
        and (
            not configuration["equality_candidate_must_be_top1"]
            or row["tokens"][0] == "="
        )
    )
    if equality_allowed:
        options.append("=")
    return list(dict.fromkeys(options))


def _unique_repair(
    before: list[str], wide_sequence: list[dict], configuration: dict,
) -> tuple[list[str] | None, str]:
    if valid_numeric_sequence(before):
        return None, "baseline_already_valid"
    if len(before) < 3 or sum(token in DIGITS for token in before) < configuration[
        "minimum_baseline_digit_count"
    ]:
        return None, "insufficient_numeric_intent"
    if len(before) > configuration["maximum_formula_glyphs"]:
        return None, "formula_too_long"
    if not (
        any(token in BINARY | RELATIONS for token in before)
        or any(row["tokens"][0] == "=" for row in wide_sequence)
    ):
        return None, "missing_structure_evidence"
    choices = [
        _options(token, row, configuration)
        for token, row in zip(before, wide_sequence, strict=True)
    ]
    alternatives = [
        (index, token)
        for index, options in enumerate(choices)
        for token in options[1:]
    ]
    valid: list[tuple[int, list[str]]] = []
    for change_count in range(1, configuration["maximum_changes"] + 1):
        for selected in combinations(alternatives, change_count):
            indices = [index for index, _ in selected]
            if len(indices) != len(set(indices)):
                continue
            candidate = list(before)
            for index, token in selected:
                candidate[index] = token
            if valid_numeric_sequence(candidate):
                valid.append((change_count, candidate))
    if not valid:
        return None, "no_valid_repair"
    minimum = min(changes for changes, _ in valid)
    minimum_repairs = [candidate for changes, candidate in valid if changes == minimum]
    if len(minimum_repairs) != 1:
        return None, "ambiguous_minimum_repair"
    return minimum_repairs[0], "repaired"


def apply_wide_candidate_syntax_rescue(
    baseline_rows: list[dict], wide_candidate_rows: list[dict],
    configuration: dict | None = None,
) -> tuple[list[dict], dict]:
    configuration = validate_configuration(configuration or DEFAULT_CONFIGURATION)
    baseline = _baseline_map(baseline_rows)
    wide = _wide_map(wide_candidate_rows, configuration)
    if not baseline or set(baseline) != set(wide):
        raise ValueError("wide syntax rescue coverage mismatch")
    baseline_formulae = _formulae(baseline)
    selected = {record_id: row["finalized_top1"] for record_id, row in baseline.items()}
    changes = []
    skipped = Counter()
    for formula_id, sequence in baseline_formulae.items():
        baseline_ids = [row["record_id"] for row in sequence]
        # Formula order and layout remain owned by the baseline path.  The
        # wider HWR cache contributes record-local evidence only, regardless
        # of its original source-order context index.
        wide_sequence = [wide[record_id] for record_id in baseline_ids]
        if any(row["formula_id"] != formula_id for row in wide_sequence):
            raise ValueError(f"wide candidate formula identity mismatch: {formula_id}")
        before = [row["finalized_top1"] for row in sequence]
        repaired, reason = _unique_repair(before, wide_sequence, configuration)
        if repaired is None:
            skipped[reason] += 1
            continue
        for index, (left, right) in enumerate(zip(before, repaired, strict=True)):
            if left == right:
                continue
            row = sequence[index]
            candidate = wide_sequence[index]
            selected[row["record_id"]] = right
            changes.append({
                "formula_id": formula_id,
                "record_id": row["record_id"],
                "context_index": index,
                "before": left,
                "after": right,
                "candidate_rank": candidate["tokens"].index(right) + 1,
                "candidate_probability": candidate["probabilities"][
                    candidate["tokens"].index(right)
                ],
                "rule": "top1_equality_slot" if right == "=" else "numeric_operand_slot",
            })
    output = []
    outside = 0
    for record_id, row in baseline.items():
        token = selected[record_id]
        candidate = wide[record_id]
        if token != row["finalized_top1"] and token not in candidate["tokens"]:
            outside += 1
        changed = token != row["finalized_top1"]
        output.append({
            **row,
            "schema": OUTPUT_SCHEMA,
            "baseline_top1_before_wide_syntax": row["finalized_top1"],
            "finalized_top1": token,
            "changed_by_wide_syntax": changed,
            "changed": bool(row.get("changed", False) or changed),
            "decision_source": (
                "wide_candidate_syntax_rescue_v1"
                if changed else str(row.get("decision_source", "baseline"))
            ),
            "wide_auxiliary_top10": candidate["tokens"],
            "wide_auxiliary_top10_probabilities": candidate["probabilities"],
        })
    output.sort(key=lambda row: (row["formula_id"], row["context_index"], row["record_id"]))
    if outside:
        raise AssertionError("wide syntax rescue invented a token")
    return output, {
        "schema": SCHEMA,
        "status": "shadow_runtime_only",
        "configuration": configuration,
        "records": len(output),
        "formulas": len(baseline_formulae),
        "changed_glyphs": len(changes),
        "changed_formulas": len({change["formula_id"] for change in changes}),
        "baseline_formula_order_authoritative": True,
        "wide_candidate_context_order_used": False,
        "changes": changes,
        "skipped": dict(sorted(skipped.items())),
        "candidate_union_preservation_rate": 1.0,
        "tokens_outside_baseline_and_wide_union": 0,
        "inserted_or_deleted_glyphs": 0,
        "glyph_order_mutations": 0,
        "grouping_mutations": 0,
        "layout_mutations": 0,
        "arithmetic_evaluation": False,
    }


def _write(path: Path, rows: list[dict]) -> None:
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "wt", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _self_test() -> None:
    assert valid_numeric_sequence(["6", "+", "8", r"\mp", "1", "4"])
    def candidate(index: int, before: str, options: list[str], probabilities: list[float]) -> dict:
        return {
            "record_id": f"r{index}", "formula_id": "f",
            "final_topk": options + [f"z{index}_{offset}" for offset in range(10 - len(options))],
            "final_topk_probabilities": probabilities + [0.0] * (10 - len(probabilities)),
            "context": {"index": index},
            "hwr_policy": "old_new_product_probability_fusion",
            "hwr_fusion_weight": 0.6,
        }
    before = ["1", "+", r"\prime", "=", "2"]
    baseline = [
        {"record_id": f"r{index}", "formula_id": "f", "context_index": index,
         "finalized_top1": token}
        for index, token in enumerate(before)
    ]
    candidates = [
        candidate(index, token, [token] if index != 2 else [r"\prime", "1"], [1.0] if index != 2 else [0.9, 0.1])
        for index, token in enumerate(before)
    ]
    repaired, audit = apply_wide_candidate_syntax_rescue(baseline, candidates)
    assert [row["finalized_top1"] for row in repaired] == ["1", "+", "1", "=", "2"]
    assert audit["changed_glyphs"] == 1 and not audit["arithmetic_evaluation"]
    valid_baseline = [dict(row) for row in baseline]
    valid_baseline[2]["finalized_top1"] = "1"
    preserved, audit = apply_wide_candidate_syntax_rescue(valid_baseline, candidates)
    assert [row["finalized_top1"] for row in preserved] == ["1", "+", "1", "=", "2"]
    assert audit["changed_glyphs"] == 0
    alpha_baseline = [dict(row) for row in baseline]
    alpha_baseline[2]["finalized_top1"] = "b"
    alpha_candidates = [dict(row) for row in candidates]
    alpha_candidates[2] = candidate(2, "b", ["b", "6"], [0.8, 0.2])
    preserved, audit = apply_wide_candidate_syntax_rescue(
        alpha_baseline, alpha_candidates,
    )
    assert preserved[2]["finalized_top1"] == "b" and audit["changed_glyphs"] == 0
    equality_baseline = [dict(row) for row in baseline]
    equality_baseline[2]["finalized_top1"] = "1"
    equality_baseline[3]["finalized_top1"] = r"\approx"
    equality_candidates = [dict(row) for row in candidates]
    equality_candidates[3] = candidate(3, r"\approx", ["=", r"\approx"], [0.9, 0.1])
    equality_disabled = {
        **DEFAULT_CONFIGURATION, "allow_equality_replacement": False,
    }
    preserved, audit = apply_wide_candidate_syntax_rescue(
        equality_baseline, equality_candidates, equality_disabled,
    )
    assert preserved[3]["finalized_top1"] == r"\approx"
    assert audit["changed_glyphs"] == 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-finalized", type=Path)
    parser.add_argument("--wide-candidates", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        print('{"self_test":"pass"}')
        return 0
    if any(value is None for value in (
        args.baseline_finalized, args.wide_candidates, args.config, args.output,
    )):
        parser.error("baseline, wide candidates, config, and output are required")
    baseline_path = args.baseline_finalized.expanduser().resolve()
    candidate_path = args.wide_candidates.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not all(path.is_file() for path in (baseline_path, candidate_path, config_path)):
        parser.error("wide syntax rescue input is missing")
    if output_path.exists():
        parser.error(f"refusing to overwrite wide syntax output: {output_path}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if payload.get("schema") != CONFIG_SCHEMA or payload.get("gate", {}).get(
        "runtime_admitted"
    ) is not True:
        parser.error("wide syntax rescue config is not admitted")
    rows, audit = apply_wide_candidate_syntax_rescue(
        list(_json_lines(baseline_path)), list(_json_lines(candidate_path)),
        payload["configuration"],
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write(output_path, rows)
    print(json.dumps({
        "output": str(output_path), "sha256": _sha256(output_path),
        "inputs": {
            "baseline_sha256": _sha256(baseline_path),
            "wide_candidates_sha256": _sha256(candidate_path),
            "config_sha256": _sha256(config_path),
        },
        "audit": audit,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
