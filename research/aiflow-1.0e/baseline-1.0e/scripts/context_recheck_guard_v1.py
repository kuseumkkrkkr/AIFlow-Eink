#!/usr/bin/env python3
"""Use a second context pass only to retract an unsafe context override."""

from __future__ import annotations

from collections import Counter


def apply_context_recheck_guard(
    rows: list[dict], first_predictions: dict[str, str],
    rechecked_predictions: dict[str, str],
) -> tuple[dict[str, str], dict]:
    record_ids = {str(row["record_id"]) for row in rows}
    if set(first_predictions) != record_ids or set(rechecked_predictions) != record_ids:
        raise ValueError("context recheck prediction coverage mismatch")
    output = {record_id: str(token) for record_id, token in first_predictions.items()}
    audit = Counter()
    changes = []
    for row in rows:
        record_id = str(row["record_id"])
        candidates = [str(token) for token in row["final_topk"]]
        hwr = candidates[0]
        first = str(first_predictions[record_id])
        rechecked = str(rechecked_predictions[record_id])
        if first not in candidates or rechecked not in candidates:
            raise AssertionError("context recheck escaped HWR candidates")
        if rechecked == first:
            audit["stable"] += 1
            continue
        audit["disagreed"] += 1
        if first != hwr and rechecked == hwr:
            output[record_id] = hwr
            audit["retracted_to_hwr_top1"] += 1
            changes.append({
                "record_id": record_id,
                "formula_id": str(row["formula_id"]),
                "context_index": int(row["context"]["index"]),
                "before": first,
                "after": hwr,
            })
        else:
            audit["ignored_novel_change"] += 1
    if any(
        output[str(row["record_id"])] not in row["final_topk"] for row in rows
    ):
        raise AssertionError("context recheck violated candidate preservation")
    return output, {
        "policy": "second pass may only retract a first-pass override to HWR Top-1",
        **{key: int(value) for key, value in sorted(audit.items())},
        "changes": changes,
        "candidate_preservation_rate": 1.0,
        "new_tokens": 0,
        "deleted_glyphs": 0,
        "grouping_mutations": 0,
    }


def self_test() -> None:
    rows = [
        {
            "record_id": "restore", "formula_id": "f", "context": {"index": 0},
            "final_topk": ["x", r"\times"],
        },
        {
            "record_id": "ignore", "formula_id": "f", "context": {"index": 1},
            "final_topk": [r"\sigma", "8"],
        },
    ]
    first = {"restore": r"\times", "ignore": r"\sigma"}
    rechecked = {"restore": "x", "ignore": "8"}
    finalized, audit = apply_context_recheck_guard(rows, first, rechecked)
    assert finalized == {"restore": "x", "ignore": r"\sigma"}
    assert audit["retracted_to_hwr_top1"] == 1
    assert audit["ignored_novel_change"] == 1


if __name__ == "__main__":
    self_test()
    print('{"self_test":"pass"}')
