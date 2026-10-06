#!/usr/bin/env python3
"""Annotate formula results with a candidate-preserving review boundary.

The guard never changes a token, stroke group, or order.  It only separates
results with unresolved shape, context, or layout evidence from automatic
acceptance.  Formula truth, writer identity, target length, and arithmetic are
not inputs.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from typing import Any


CONFIG_SCHEMA = "aiflow-formula-acceptance-guard-config/v1"
CONFIG_SCHEMA_V2 = "aiflow-formula-acceptance-guard-config/v2"
GUARD_SCHEMA = "aiflow-formula-acceptance-guard/v1"
DECISION_SCHEMA = "aiflow-formula-acceptance-decision/v1"
AUTO_ACCEPTED = "AUTO_ACCEPTED"
REVIEW_REQUIRED = "REVIEW_REQUIRED"
FENCE_PAIRS = {
    "(": ")", "[": "]", "{": "}", r"\{": r"\}",
    r"\langle": r"\rangle", r"\lceil": r"\rceil",
    r"\lfloor": r"\rfloor", r"\llbracket": r"\rrbracket",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_configuration(payload: dict[str, Any]) -> dict[str, Any]:
    policy = dict(payload.get("policy") or {})
    cross = dict(policy.get("two_stroke_cross_grouping_conflict") or {})
    digit_conflict = dict(policy.get("context_digit_homograph_conflict") or {})
    contracts = dict(payload.get("contracts") or {})
    families = policy.get("unresolved_singleton_homograph_families")
    ratio = float(cross.get("minimum_ranker_probability_ratio", math.nan))
    digit_ratio = float(
        digit_conflict.get("minimum_alternate_digit_probability_ratio", math.nan)
    )
    alternate_tokens = cross.get("alternate_tokens")
    legacy_multidigit = (
        payload.get("schema") == CONFIG_SCHEMA
        and policy.get("review_context_mutated_multidigit") is True
    )
    refined_multidigit = (
        payload.get("schema") == CONFIG_SCHEMA_V2
        and policy.get("review_context_mutated_multidigit") is False
        and digit_conflict.get("enabled") is True
        and digit_conflict.get("require_adjacent_digit") is True
        and math.isfinite(digit_ratio)
        and 0.0 <= digit_ratio <= 1.0
    )
    if (
        payload.get("schema") not in {CONFIG_SCHEMA, CONFIG_SCHEMA_V2}
        or payload.get("guard_schema") != GUARD_SCHEMA
        or payload.get("status") != "development_only_posthoc_shadow"
        or payload.get("posthoc_test_tuning") is not True
        or policy.get("review_singleton_context_override") is not True
        or not (legacy_multidigit or refined_multidigit)
        or policy.get("review_unbalanced_multisymbol_fence") is not True
        or not isinstance(families, list)
        or not families
        or any(
            not isinstance(family, list)
            or len(family) < 2
            or len(family) != len(set(family))
            or any(not isinstance(token, str) or not token for token in family)
            for family in families
        )
        or cross.get("enabled") is not True
        or not isinstance(alternate_tokens, list)
        or len(alternate_tokens) < 2
        or len(alternate_tokens) != len(set(alternate_tokens))
        or any(not isinstance(token, str) or not token for token in alternate_tokens)
        or not math.isfinite(ratio)
        or not 0.0 <= ratio <= 1.0
        or contracts.get("candidate_preserving") is not True
        or contracts.get("token_mutations") != 0
        or contracts.get("grouping_mutations") != 0
        or contracts.get("glyph_order_mutations") != 0
        or contracts.get("target_label_or_glyph_count_input") is not False
        or contracts.get("writer_identity_input") is not False
        or contracts.get("arithmetic_evaluation") is not False
        or payload.get("requires_explicit_shadow_opt_in") is not True
        or payload.get("product_default_enabled") is not False
    ):
        raise ValueError("formula acceptance guard configuration mismatch")
    return {
        **payload,
        "policy": {
            **policy,
            "unresolved_singleton_homograph_families": families,
            "two_stroke_cross_grouping_conflict": {
                **cross,
                "alternate_tokens": alternate_tokens,
                "minimum_ranker_probability_ratio": ratio,
            },
            "context_digit_homograph_conflict": (
                {
                    **digit_conflict,
                    "minimum_alternate_digit_probability_ratio": digit_ratio,
                }
                if refined_multidigit else {}
            ),
        },
        "contracts": contracts,
    }


def load_configuration(path: Path) -> tuple[dict[str, Any], str]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return (
        validate_configuration(json.loads(resolved.read_text(encoding="utf-8"))),
        _sha256(resolved),
    )


def _balanced_fences(tokens: list[str]) -> bool:
    stack: list[str] = []
    closers = set(FENCE_PAIRS.values())
    for token in tokens:
        if token in FENCE_PAIRS:
            stack.append(FENCE_PAIRS[token])
        elif token in closers and (not stack or stack.pop() != token):
            return False
    return not stack


def _two_stroke_cross_conflict(formula: dict[str, Any], policy: dict[str, Any]) -> bool:
    details = list(formula.get("audit", {}).get("candidate_partitions_detail") or [])
    selected_rank = int(formula.get("audit", {}).get("selected_partition_rank", 0))
    selected = next(
        (row for row in details if int(row.get("rank", 0)) == selected_rank), None,
    )
    if selected is None:
        return False
    groups = list(selected.get("groups") or [])
    selected_probability = float(selected.get("ranker_probability", 0.0))
    if (
        len(groups) != 2
        or any(len(group) != 1 for group in groups)
        or selected_probability <= 0.0
    ):
        return False
    strokes = {int(value) for group in groups for value in group}
    cross = policy["two_stroke_cross_grouping_conflict"]
    tokens = set(cross["alternate_tokens"])
    for alternate in details:
        alternate_groups = list(alternate.get("groups") or [])
        alternate_tokens = [str(value) for value in alternate.get("context_tokens") or []]
        if (
            int(alternate.get("rank", 0)) != selected_rank
            and len(alternate_groups) == 1
            and len(alternate_groups[0]) == 2
            and {int(value) for value in alternate_groups[0]} == strokes
            and any(token in tokens for token in alternate_tokens)
            and float(alternate.get("ranker_probability", 0.0)) / selected_probability
            >= cross["minimum_ranker_probability_ratio"]
        ):
            return True
    return False


def review_reasons(
    formula: dict[str, Any], configuration: dict[str, Any],
) -> list[str]:
    configuration = validate_configuration(configuration)
    policy = configuration["policy"]
    tokens = [str(value) for value in formula.get("finalized_tokens") or []]
    symbols = list(formula.get("symbols") or [])
    if not tokens or len(tokens) != len(symbols):
        raise ValueError("formula acceptance guard requires aligned tokens and symbols")
    reasons = set()
    if (
        policy["review_singleton_context_override"]
        and len(tokens) == 1
        and bool(symbols[0].get("changed"))
    ):
        reasons.add("singleton_context_override")
    mutated_multidigit_positions = [
        index for index, (token, symbol) in enumerate(zip(tokens, symbols, strict=True))
        if (
            bool(symbol.get("changed"))
            and token.isascii()
            and token.isdigit()
            and (
                (index > 0 and tokens[index - 1].isascii() and tokens[index - 1].isdigit())
                or (
                    index + 1 < len(tokens)
                    and tokens[index + 1].isascii()
                    and tokens[index + 1].isdigit()
                )
            )
        )
    ]
    if policy["review_context_mutated_multidigit"] and mutated_multidigit_positions:
        reasons.add("context_mutated_multidigit")
    digit_conflict = policy.get("context_digit_homograph_conflict") or {}
    for index in mutated_multidigit_positions if digit_conflict.get("enabled") else []:
        symbol = symbols[index]
        candidates = [str(value) for value in symbol.get("hwr_topk") or []]
        probabilities = [float(value) for value in symbol.get("hwr_topk_probabilities") or []]
        if (
            not candidates
            or len(candidates) != len(probabilities)
            or len(candidates) != len(set(candidates))
            or any(not math.isfinite(value) or value < 0.0 for value in probabilities)
        ):
            raise ValueError("digit homograph review requires valid HWR Top-k evidence")
        token = tokens[index]
        if token not in candidates:
            continue
        selected_probability = probabilities[candidates.index(token)]
        if selected_probability <= 0.0:
            continue
        if any(
            candidate != token
            and candidate.isascii()
            and candidate.isdigit()
            and probability / selected_probability
            >= digit_conflict["minimum_alternate_digit_probability_ratio"]
            for candidate, probability in zip(candidates, probabilities, strict=True)
        ):
            reasons.add("context_digit_homograph_conflict")
            break
    if (
        policy["review_unbalanced_multisymbol_fence"]
        and len(tokens) > 1
        and not _balanced_fences(tokens)
    ):
        reasons.add("unbalanced_multisymbol_fence")
    if len(tokens) == 1 and any(
        tokens[0] in family
        for family in policy["unresolved_singleton_homograph_families"]
    ):
        reasons.add("unresolved_singleton_homograph")
    if _two_stroke_cross_conflict(formula, policy):
        reasons.add("two_stroke_cross_grouping_conflict")
    return sorted(reasons)


def assess_formula(
    formula: dict[str, Any], configuration: dict[str, Any],
) -> dict[str, Any]:
    reasons = review_reasons(formula, configuration)
    return {
        "schema": DECISION_SCHEMA,
        "decision_status": REVIEW_REQUIRED if reasons else AUTO_ACCEPTED,
        "review_reasons": reasons,
        "finalized_tokens_preserved": True,
        "stroke_groups_preserved": True,
        "raw_fallback_required": bool(reasons),
    }


def apply_formula_acceptance_guard(
    payload: dict[str, Any], configuration: dict[str, Any],
    configuration_sha256: str,
) -> dict[str, Any]:
    configuration = validate_configuration(configuration)
    if (
        payload.get("schema") != "aiflow-raw-formula-context-runtime/v1"
        or payload.get("status") != "development_only_posthoc_shadow"
        or payload.get("audit", {}).get("product_default_enabled") is not False
    ):
        raise ValueError("formula acceptance input runtime contract mismatch")
    if len(configuration_sha256) != 64:
        raise ValueError("formula acceptance configuration hash is invalid")
    output = deepcopy(payload)
    decisions = Counter()
    reasons = Counter()
    for formula in output.get("formulas") or []:
        before_tokens = deepcopy(formula.get("finalized_tokens"))
        before_groups = deepcopy(formula.get("groups"))
        decision = assess_formula(formula, configuration)
        formula["decision_status"] = decision["decision_status"]
        formula["acceptance"] = decision
        decisions[decision["decision_status"]] += 1
        reasons.update(decision["review_reasons"])
        if formula.get("finalized_tokens") != before_tokens or formula.get("groups") != before_groups:
            raise AssertionError("formula acceptance guard mutated inference output")
    output.setdefault("audit", {})["formula_acceptance_guard"] = {
        "schema": GUARD_SCHEMA,
        "enabled": True,
        "configuration_sha256": configuration_sha256,
        "formulas": sum(decisions.values()),
        "decision_status": dict(sorted(decisions.items())),
        "review_reasons": dict(sorted(reasons.items())),
        "candidate_preserving": True,
        "token_mutations": 0,
        "grouping_mutations": 0,
        "glyph_order_mutations": 0,
        "target_label_or_glyph_count_input": False,
        "writer_identity_input": False,
        "arithmetic_evaluation": False,
        "product_default_enabled": False,
    }
    return output


def _self_test() -> None:
    configuration = validate_configuration({
        "schema": CONFIG_SCHEMA,
        "guard_schema": GUARD_SCHEMA,
        "status": "development_only_posthoc_shadow",
        "posthoc_test_tuning": True,
        "policy": {
            "review_singleton_context_override": True,
            "review_context_mutated_multidigit": True,
            "review_unbalanced_multisymbol_fence": True,
            "unresolved_singleton_homograph_families": [["0", r"\circlearrowright"]],
            "two_stroke_cross_grouping_conflict": {
                "enabled": True,
                "alternate_tokens": ["x", r"\mathfrak{X}"],
                "minimum_ranker_probability_ratio": 0.35,
            },
        },
        "contracts": {
            "candidate_preserving": True, "token_mutations": 0,
            "grouping_mutations": 0, "glyph_order_mutations": 0,
            "target_label_or_glyph_count_input": False,
            "writer_identity_input": False, "arithmetic_evaluation": False,
        },
        "requires_explicit_shadow_opt_in": True,
        "product_default_enabled": False,
    })
    formula = {
        "finalized_tokens": ["3", "6"],
        "symbols": [
            {"changed": False, "hwr_top1": "3", "finalized_top1": "3"},
            {"changed": True, "hwr_top1": r"\diamond", "finalized_top1": "6"},
        ],
        "audit": {"selected_partition_rank": 1, "candidate_partitions_detail": []},
    }
    assert review_reasons(formula, configuration) == ["context_mutated_multidigit"]
    clean = {**formula, "finalized_tokens": ["3", "+", "6"], "symbols": [
        formula["symbols"][0], {"changed": False}, formula["symbols"][1],
    ]}
    assert assess_formula(clean, configuration)["decision_status"] == AUTO_ACCEPTED
    refined = deepcopy(configuration)
    refined["schema"] = CONFIG_SCHEMA_V2
    refined["policy"]["review_context_mutated_multidigit"] = False
    refined["policy"]["context_digit_homograph_conflict"] = {
        "enabled": True,
        "require_adjacent_digit": True,
        "minimum_alternate_digit_probability_ratio": 0.5,
    }
    formula["symbols"][1].update({
        "hwr_topk": [r"\diamond", "6", "0"],
        "hwr_topk_probabilities": [0.45, 0.25, 0.13],
    })
    assert review_reasons(formula, refined) == ["context_digit_homograph_conflict"]
    formula["symbols"][1]["hwr_topk_probabilities"][2] = 0.01
    assert assess_formula(formula, refined)["decision_status"] == AUTO_ACCEPTED


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-posthoc-shadow", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        print(json.dumps({"self_test": "pass", "schema": GUARD_SCHEMA}))
        return 0
    if args.configuration is None or args.input is None or args.output is None:
        parser.error("--configuration, --input, and --output are required")
    if not args.allow_posthoc_shadow:
        parser.error("--allow-posthoc-shadow is required")
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not input_path.is_file() or input_path.drive.upper() != "D:":
        parser.error("--input must be an existing file on D:")
    if output_path.exists() or output_path.drive.upper() != "D:":
        parser.error("--output must be a new file on D:")
    configuration, configuration_sha256 = load_configuration(args.configuration)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    guarded = apply_formula_acceptance_guard(
        payload, configuration, configuration_sha256,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(guarded, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    print(json.dumps({
        "output": str(output_path), "sha256": _sha256(output_path),
        "audit": guarded["audit"]["formula_acceptance_guard"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
