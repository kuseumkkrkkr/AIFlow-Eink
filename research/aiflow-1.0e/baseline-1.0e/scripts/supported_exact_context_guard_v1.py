#!/usr/bin/env python3
"""Accept stronger exact-context choices only with owned support and shape evidence."""

from __future__ import annotations

from collections import Counter

from train_context_decision_layer_v1 import _semantic_role


STRONG_EXACT_CONTEXT_SCALE = 2.0
MINIMUM_PROJECT_SUPPORT = 5
MINIMUM_HWR_PROBABILITY_RATIO = 0.2


def apply_supported_exact_context_guard(
    rows: list[dict], baseline: dict[str, str], stronger: dict[str, str],
    label_support: dict[str, int],
) -> tuple[dict[str, str], dict]:
    record_ids = {str(row["record_id"]) for row in rows}
    if set(baseline) != record_ids or set(stronger) != record_ids:
        raise ValueError("supported exact-context prediction coverage mismatch")
    output = {record_id: str(token) for record_id, token in baseline.items()}
    audit = Counter()
    changes = []
    for row in rows:
        record_id = str(row["record_id"])
        before = output[record_id]
        after = str(stronger[record_id])
        if before == after:
            continue
        if _semantic_role(before) != _semantic_role(after):
            audit["skipped_role_change"] += 1
            continue
        before_support = int(label_support.get(before, 0))
        after_support = int(label_support.get(after, 0))
        if before_support != 0 or after_support < MINIMUM_PROJECT_SUPPORT:
            audit["skipped_support"] += 1
            continue
        if after not in row["final_topk"]:
            raise AssertionError("supported exact-context guard invented a candidate")
        candidate_index = row["final_topk"].index(after)
        probabilities = row["final_topk_probabilities"]
        ratio = float(probabilities[candidate_index]) / max(
            float(probabilities[0]), 1e-12
        )
        if ratio < MINIMUM_HWR_PROBABILITY_RATIO:
            audit["skipped_probability_floor"] += 1
            continue
        output[record_id] = after
        changes.append({
            "record_id": record_id,
            "formula_id": str(row["formula_id"]),
            "context_index": int(row["context"]["index"]),
            "before": before,
            "after": after,
            "before_project_support": before_support,
            "after_project_support": after_support,
            "hwr_probability_ratio": ratio,
        })
        audit["changed_glyphs"] += 1
    if any(
        output[str(row["record_id"])] not in row["final_topk"] for row in rows
    ):
        raise AssertionError("supported exact-context guard violated candidates")
    return output, {
        "configuration": {
            "strong_exact_context_scale": STRONG_EXACT_CONTEXT_SCALE,
            "selected_label_project_support_required": 0,
            "candidate_minimum_project_support": MINIMUM_PROJECT_SUPPORT,
            "minimum_hwr_probability_ratio": MINIMUM_HWR_PROBABILITY_RATIO,
            "same_semantic_role_required": True,
        },
        **{key: int(value) for key, value in sorted(audit.items())},
        "changes": changes,
        "candidate_preservation_rate": 1.0,
        "new_tokens": 0,
        "deleted_glyphs": 0,
        "grouping_mutations": 0,
    }


def _row(probability: float = 0.19) -> dict:
    return {
        "record_id": "r", "formula_id": "f",
        "final_topk": ["h", "b"],
        "final_topk_probabilities": [0.8, probability],
        "context": {"index": 0, "length": 3},
    }


def self_test() -> None:
    finalized, audit = apply_supported_exact_context_guard(
        [_row()], {"r": "h"}, {"r": "b"}, {"b": 5}
    )
    assert finalized == {"r": "b"} and audit["changed_glyphs"] == 1
    preserved, audit = apply_supported_exact_context_guard(
        [_row(0.1)], {"r": "h"}, {"r": "b"}, {"b": 5}
    )
    assert preserved == {"r": "h"} and audit["skipped_probability_floor"] == 1


if __name__ == "__main__":
    self_test()
    print('{"self_test":"pass"}')
