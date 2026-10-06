#!/usr/bin/env python3
"""Complete missing directional fences without changing selected fences."""

from __future__ import annotations

from collections import Counter
import math

import train_masked_context_reranker_v1 as masked


FENCE_PAIRS = {
    "(": ")", "[": "]", "{": "}", r"\{": r"\}",
    r"\langle": r"\rangle", r"\lceil": r"\rceil",
    r"\lfloor": r"\rfloor", r"\llbracket": r"\rrbracket",
}
CLOSE_TO_OPEN = {right: left for left, right in FENCE_PAIRS.items()}
DIRECTIONAL_FENCES = frozenset(FENCE_PAIRS) | frozenset(CLOSE_TO_OPEN)
BEAM_WIDTH = 64
PREFERRED_TOKEN_BONUS = 0.75
MIN_VALID_MARGIN = 1.5
MAX_CHANGED_GLYPHS = 2
MAX_FORMULA_LENGTH = 64


def _balanced(tokens: tuple[str, ...]) -> bool:
    stack = []
    pairs = 0
    for token in tokens:
        if token in FENCE_PAIRS:
            stack.append(FENCE_PAIRS[token])
        elif token in CLOSE_TO_OPEN:
            if not stack or stack.pop() != token:
                return False
            pairs += 1
    return not stack and pairs > 0


def _options(row: dict, selected: str) -> list[tuple[str, float]]:
    output = []
    for token, probability in zip(
        row["final_topk"], row["final_topk_probabilities"], strict=True
    ):
        token = str(token)
        if token != selected and (
            selected in DIRECTIONAL_FENCES or token not in DIRECTIONAL_FENCES
        ):
            continue
        score = math.log(max(float(probability), 1e-12))
        if token == selected:
            score += PREFERRED_TOKEN_BONUS
        output.append((token, score))
    return output


def apply_semantic_fence_guard(
    rows: list[dict], predictions: dict[str, str],
) -> tuple[dict[str, str], dict]:
    record_ids = {str(row["record_id"]) for row in rows}
    if set(predictions) != record_ids:
        raise ValueError("semantic fence guard prediction coverage mismatch")
    output = {record_id: str(token) for record_id, token in predictions.items()}
    audit = Counter()
    changes = []
    formulae = masked._formulae(rows)
    for formula_id, sequence in formulae.items():
        if len(sequence) > MAX_FORMULA_LENGTH:
            audit["skipped_length"] += 1
            continue
        baseline = tuple(output[str(row["record_id"])] for row in sequence)
        if not any(token in DIRECTIONAL_FENCES for token in baseline):
            continue
        audit["formulas_with_selected_fence"] += 1
        if _balanced(baseline):
            audit["baseline_already_balanced"] += 1
            continue
        options = [
            _options(row, output[str(row["record_id"])]) for row in sequence
        ]
        if any(not values for values in options):
            raise AssertionError("selected candidate missing from fence options")
        beam: list[tuple[float, tuple[str, ...]]] = [(0.0, ())]
        for values in options:
            beam = sorted(
                (
                    (score + option_score, tokens + (token,))
                    for score, tokens in beam
                    for token, option_score in values
                ),
                key=lambda item: (-item[0], item[1]),
            )[:BEAM_WIDTH]
        valid = sorted(
            (item for item in beam if _balanced(item[1])),
            key=lambda item: (-item[0], item[1]),
        )
        if not valid:
            audit["no_balanced_completion"] += 1
            continue
        best_score, best = valid[0]
        second_score = valid[1][0] if len(valid) > 1 else None
        margin = best_score - second_score if second_score is not None else None
        changed_positions = [
            index for index, (before, after) in enumerate(zip(baseline, best, strict=True))
            if before != after
        ]
        if not 1 <= len(changed_positions) <= MAX_CHANGED_GLYPHS:
            audit["skipped_change_limit"] += 1
            continue
        if margin is not None and margin < MIN_VALID_MARGIN:
            audit["skipped_ambiguous"] += 1
            continue
        for index in changed_positions:
            row = sequence[index]
            token = best[index]
            if baseline[index] in DIRECTIONAL_FENCES:
                raise AssertionError("semantic fence guard changed a selected fence")
            if token not in row["final_topk"]:
                raise AssertionError("semantic fence guard invented a candidate")
            output[str(row["record_id"])] = token
        changes.append({
            "formula_id": str(formula_id),
            "before": list(baseline),
            "after": list(best),
            "changed_positions": changed_positions,
            "valid_score_margin": margin,
            "only_balanced_completion": second_score is None,
        })
        audit["finalized_formulas"] += 1
        audit["changed_glyphs"] += len(changed_positions)
    if any(
        output[str(row["record_id"])] not in row["final_topk"] for row in rows
    ):
        raise AssertionError("semantic fence guard violated candidate preservation")
    return output, {
        "configuration": {
            "beam_width": BEAM_WIDTH,
            "preferred_token_bonus": PREFERRED_TOKEN_BONUS,
            "minimum_valid_score_margin": MIN_VALID_MARGIN,
            "maximum_changed_glyphs": MAX_CHANGED_GLYPHS,
            "maximum_formula_length": MAX_FORMULA_LENGTH,
            "selected_fence_policy": "immutable",
            "fence_pairs": FENCE_PAIRS,
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
    index: int, length: int, candidates: list[str], probabilities: list[float],
) -> dict:
    return {
        "record_id": f"r{index}",
        "formula_id": "fence",
        "final_topk": candidates,
        "final_topk_probabilities": probabilities,
        "context": {"index": index, "length": length},
        "geometry": {
            "center_x": float(index), "center_y": 0.5,
            "width_rel": 0.2, "height_rel": 1.0,
        },
    }


def self_test() -> None:
    candidates = [
        (["f"], [1.0]),
        ([r"\mid", "("], [0.7, 0.2]),
        (["y"], [1.0]),
        ([")", "]"], [0.6, 0.3]),
    ]
    rows = [_row(index, len(candidates), *values) for index, values in enumerate(candidates)]
    baseline = {row["record_id"]: row["final_topk"][0] for row in rows}
    finalized, audit = apply_semantic_fence_guard(rows, baseline)
    assert [finalized[row["record_id"]] for row in rows] == ["f", "(", "y", ")"]
    assert audit["changed_glyphs"] == 1
    unchanged, locked = apply_semantic_fence_guard(rows, finalized)
    assert unchanged == finalized and locked["baseline_already_balanced"] == 1


if __name__ == "__main__":
    self_test()
    print('{"self_test":"pass"}')
