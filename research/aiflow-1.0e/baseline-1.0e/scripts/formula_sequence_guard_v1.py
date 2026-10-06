#!/usr/bin/env python3
"""Candidate-preserving formula sequence correction over frozen HWR Top-5.

This guard does not evaluate arithmetic.  It only repairs a locally invalid
simple numeric token sequence, using the frozen formula-context candidate
scores and the project-owned role grammar.  Single-glyph input, symbolic
formulae, and already valid numeric expressions are preserved by default.
"""

from __future__ import annotations

from collections import Counter
import math

import train_context_decision_layer_v1 as context
import train_masked_context_reranker_v1 as masked


SCHEMA = "aiflow-formula-sequence-guard/v1"
DIGITS = frozenset("0123456789")
BINARY = frozenset({"+", "-", "/", r"\times", r"\div", r"\cdot"})
RELATIONS = frozenset({"=", "<", ">", r"\leq", r"\geq", r"\neq"})
NUMERIC = DIGITS | BINARY | RELATIONS
VERTICAL_ONE_SOURCES = frozenset({"/", "|", r"\mid", r"\lfloor"})
DEFAULT_CONFIGURATION = {
    "shape_weight": 0.10,
    "context_weight": 1.00,
    "grammar_weight": 0.50,
    "retention_bonus": 0.00,
    "margin": 0.25,
    "max_changes": 1,
    "minimum_candidate_probability_ratio": 0.02,
    "non_numeric_lock_probability": 0.99,
    "repair_valid_missing_structure": False,
}


def validate_configuration(configuration: dict) -> dict:
    if set(configuration) != set(DEFAULT_CONFIGURATION):
        raise ValueError("formula sequence configuration fields mismatch")
    output = {
        key: bool(value) if key == "repair_valid_missing_structure" else value
        for key, value in configuration.items()
    }
    for key in (
        "shape_weight", "context_weight", "grammar_weight",
        "retention_bonus", "margin", "minimum_candidate_probability_ratio",
        "non_numeric_lock_probability",
    ):
        value = float(output[key])
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"invalid formula sequence configuration: {key}")
        output[key] = value
    if any(output[key] > 1.0 for key in (
        "minimum_candidate_probability_ratio", "non_numeric_lock_probability",
    )):
        raise ValueError("formula sequence probabilities must not exceed one")
    output["max_changes"] = int(output["max_changes"])
    if output["max_changes"] != 1:
        raise ValueError("formula sequence guard v1 permits exactly one change")
    return output


def _valid_side(tokens: list[str]) -> bool:
    if not tokens:
        return False
    expect_digit = True
    saw_digit = False
    for token in tokens:
        if expect_digit:
            if token not in DIGITS:
                return False
            saw_digit = True
            expect_digit = False
        elif token in DIGITS:
            continue
        elif token in BINARY:
            expect_digit = True
        else:
            return False
    return saw_digit and not expect_digit


def valid_numeric_sequence(tokens: list[str]) -> bool:
    relation_positions = [index for index, token in enumerate(tokens) if token in RELATIONS]
    if len(relation_positions) > 1:
        return False
    if not relation_positions:
        return _valid_side(tokens)
    index = relation_positions[0]
    return _valid_side(tokens[:index]) and _valid_side(tokens[index + 1:])


def _numeric_intent(
    sequence: list[dict], baseline: list[str], configuration: dict,
) -> bool:
    if len(sequence) < 3 or sum(token in DIGITS for token in baseline) < 2:
        return False
    if not any(
        any(str(token) in BINARY | RELATIONS for token in row["final_topk"])
        for row in sequence[1:-1]
    ):
        return False
    for row, token in zip(sequence, baseline):
        if token in NUMERIC or token != str(row["final_topk"][0]):
            continue
        if float(row["final_topk_probabilities"][0]) >= configuration[
            "non_numeric_lock_probability"
        ]:
            return False
    return all(any(str(token) in NUMERIC for token in row["final_topk"]) for row in sequence)


def _candidate_context(components: dict) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for row_index, row in enumerate(components["rows"]):
        record_id = str(row["record_id"])
        output[record_id] = {
            str(token): float(components["candidate_context"][row_index, candidate_index])
            for candidate_index, token in enumerate(row["final_topk"])
        }
    return output


def _sequence_score(
    sequence: list[dict], tokens: list[str], baseline: list[str],
    candidate_context: dict[str, dict[str, float]], grammar: dict,
    configuration: dict,
) -> float:
    roles = [context._semantic_role(token) for token in tokens]
    total = 0.0
    for index, (row, token) in enumerate(zip(sequence, tokens)):
        candidate_index = [str(value) for value in row["final_topk"]].index(token)
        probability = float(row["final_topk_probabilities"][candidate_index])
        left = roles[index - 1] if index else "boundary"
        right = roles[index + 1] if index + 1 < len(roles) else "boundary"
        total += (
            configuration["shape_weight"] * math.log(probability + 1e-12)
            + configuration["context_weight"]
            * candidate_context[str(row["record_id"])][token]
            + configuration["grammar_weight"]
            * context._grammar_score(grammar, left, right, roles[index])
            + configuration["retention_bonus"] * float(token == baseline[index])
        )
    return total


def _one_change_candidates(
    sequence: list[dict], baseline: list[str], configuration: dict,
) -> list[list[str]]:
    output = [list(baseline)]
    for index, row in enumerate(sequence):
        for token in row["final_topk"]:
            token = str(token)
            if token == baseline[index] or token not in NUMERIC:
                continue
            row_tokens = [str(value) for value in row["final_topk"]]
            probabilities = [float(value) for value in row["final_topk_probabilities"]]
            before_probability = probabilities[row_tokens.index(baseline[index])]
            after_probability = probabilities[row_tokens.index(token)]
            low_ratio_vertical_one = (
                baseline[index] in VERTICAL_ONE_SOURCES and token == "1"
            )
            if (
                after_probability / max(before_probability, 1e-12)
                < configuration["minimum_candidate_probability_ratio"]
                and not low_ratio_vertical_one
            ):
                continue
            candidate = list(baseline)
            candidate[index] = token
            output.append(candidate)
    return output


def apply_formula_sequence_guard(
    rows: list[dict], predictions: dict[str, str], components: dict,
    grammar: dict, configuration: dict | None = None,
) -> tuple[dict[str, str], dict]:
    configuration = validate_configuration(configuration or DEFAULT_CONFIGURATION)
    record_ids = {str(row["record_id"]) for row in rows}
    if set(predictions) != record_ids:
        raise ValueError("formula sequence prediction coverage mismatch")
    for row in rows:
        if str(predictions[str(row["record_id"])]) not in {
            str(token) for token in row["final_topk"]
        }:
            raise ValueError("formula sequence prediction escaped HWR candidates")
    candidate_context = _candidate_context(components)
    if set(candidate_context) != record_ids:
        raise ValueError("formula sequence context coverage mismatch")

    output = dict(predictions)
    changes = []
    skipped = Counter()
    for formula_id, sequence in masked._formulae(rows).items():
        baseline = [str(predictions[str(row["record_id"])]) for row in sequence]
        if len(sequence) == 1:
            skipped["singleton"] += 1
            continue
        if not _numeric_intent(sequence, baseline, configuration):
            skipped["non_numeric_or_ambiguous"] += 1
            continue
        baseline_valid = valid_numeric_sequence(baseline)
        if baseline_valid and not configuration["repair_valid_missing_structure"]:
            skipped["already_valid"] += 1
            continue

        scored = []
        for candidate in _one_change_candidates(sequence, baseline, configuration):
            if not valid_numeric_sequence(candidate):
                continue
            if baseline_valid:
                baseline_structure = sum(token in BINARY | RELATIONS for token in baseline)
                candidate_structure = sum(token in BINARY | RELATIONS for token in candidate)
                if candidate != baseline and candidate_structure <= baseline_structure:
                    continue
            scored.append((
                _sequence_score(
                    sequence, candidate, baseline, candidate_context, grammar,
                    configuration,
                ),
                candidate,
            ))
        if not scored:
            skipped["no_valid_candidate"] += 1
            continue
        scored.sort(key=lambda item: item[0], reverse=True)
        best_score, best = scored[0]
        if best == baseline:
            skipped["retained_by_score"] += 1
            continue
        runner_up = scored[1][0] if len(scored) > 1 else -math.inf
        if best_score - runner_up < configuration["margin"]:
            skipped["insufficient_margin"] += 1
            continue
        changed_indices = [
            index for index, (before, after) in enumerate(zip(baseline, best))
            if before != after
        ]
        if len(changed_indices) != 1:
            raise AssertionError("formula sequence guard exceeded one change")
        index = changed_indices[0]
        row = sequence[index]
        record_id = str(row["record_id"])
        output[record_id] = best[index]
        changes.append({
            "formula_id": str(formula_id),
            "record_id": record_id,
            "before": baseline[index],
            "after": best[index],
            "baseline_valid": baseline_valid,
            "score_margin": (
                None if not math.isfinite(runner_up) else best_score - runner_up
            ),
        })

    if any(output[str(row["record_id"])] not in row["final_topk"] for row in rows):
        raise AssertionError("formula sequence guard invented a candidate")
    return output, {
        "schema": SCHEMA,
        "configuration": configuration,
        "formulas": len(masked._formulae(rows)),
        "changed": len(changes),
        "changes": changes,
        "skipped": dict(sorted(skipped.items())),
        "candidate_preservation_rate": 1.0,
        "new_tokens": 0,
        "deleted_glyphs": 0,
        "grouping_mutations": 0,
        "arithmetic_evaluation": False,
    }


def _self_test() -> None:
    grammar_rows = []
    rows = []
    labels = ["5", "+", "0", "=", "5"]
    candidates = [["5"], ["+"], ["q", "0"], ["="], ["5"]]
    for index, (label, topk) in enumerate(zip(labels, candidates)):
        row = {
            "record_id": f"r{index}", "formula_id": "repair",
            "label": label, "final_topk": topk,
            "final_topk_probabilities": [0.9] if len(topk) == 1 else [0.8, 0.2],
            "context": {"index": index, "length": 5},
            "geometry": {},
        }
        rows.append(row)
        grammar_rows.append(row)
    grammar = context._fit_role_grammar(grammar_rows)
    candidate_context = []
    for row in rows:
        candidate_context.append([0.0] if len(row["final_topk"]) == 1 else [-1.0, 1.0])
    import numpy as np
    width = max(map(len, candidates))
    scores = np.full((len(rows), width), -1e9, dtype=np.float32)
    for index, values in enumerate(candidate_context):
        scores[index, :len(values)] = values
    components = {"rows": rows, "candidate_context": scores}
    predictions = {row["record_id"]: row["final_topk"][0] for row in rows}
    repaired, audit = apply_formula_sequence_guard(rows, predictions, components, grammar)
    assert repaired["r2"] == "0" and audit["changed"] == 1
    wrong_answer = ["1", "+", "1", "=", "3"]
    assert valid_numeric_sequence(wrong_answer)
    assert not valid_numeric_sequence(["1", "+", "=", "3"])


def main() -> int:
    _self_test()
    print('{"self_test":"pass"}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
