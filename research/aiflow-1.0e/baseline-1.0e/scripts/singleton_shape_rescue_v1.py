#!/usr/bin/env python3
"""Use admitted auxiliary HWR evidence only for context-free singleton glyphs."""

from __future__ import annotations

from collections import Counter
import math

import train_masked_context_reranker_v1 as masked


SCHEMA = "aiflow-singleton-shape-rescue/v1"
DEFAULT_CONFIGURATION = {
    "auxiliary_policy": "current_expanded_writer_loo_probability_fusion",
    "auxiliary_weight": 0.6,
    "token_confidence_thresholds": {"/": 0.50, r"\times": 0.40},
    "token_candidate_maximum_ranks": {"/": 1, r"\times": 2},
}


def validate_configuration(configuration: dict) -> dict:
    fields = set(configuration)
    current_fields = set(DEFAULT_CONFIGURATION)
    legacy_fields = current_fields - {"token_candidate_maximum_ranks"}
    if fields not in (legacy_fields, current_fields):
        raise ValueError("singleton shape rescue configuration fields mismatch")
    policy = str(configuration["auxiliary_policy"])
    if not policy:
        raise ValueError("singleton shape rescue auxiliary policy must be non-empty")
    weight = float(configuration["auxiliary_weight"])
    if not math.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError("singleton shape rescue auxiliary weight must be in [0,1]")
    thresholds = configuration["token_confidence_thresholds"]
    if not isinstance(thresholds, dict) or set(thresholds) != {"/", r"\times"}:
        raise ValueError("singleton shape rescue token thresholds mismatch")
    clean_thresholds = {}
    for token, raw in thresholds.items():
        value = float(raw)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"invalid singleton shape threshold: {token}")
        clean_thresholds[str(token)] = value
    ranks = configuration.get(
        "token_candidate_maximum_ranks", {"/": 1, r"\times": 1},
    )
    if not isinstance(ranks, dict) or set(ranks) != {"/", r"\times"}:
        raise ValueError("singleton shape rescue token ranks mismatch")
    clean_ranks = {}
    for token, raw in ranks.items():
        if isinstance(raw, bool) or not isinstance(raw, int) or not 1 <= raw <= 5:
            raise ValueError(f"invalid singleton shape candidate rank: {token}")
        clean_ranks[str(token)] = raw
    return {
        "auxiliary_policy": policy,
        "auxiliary_weight": weight,
        "token_confidence_thresholds": clean_thresholds,
        "token_candidate_maximum_ranks": clean_ranks,
    }


def _evidence_by_record(rows: list[dict], configuration: dict) -> dict[str, dict]:
    output = {}
    for row in rows:
        record_id = str(row.get("record_id", ""))
        if not record_id or record_id in output:
            raise ValueError(f"singleton shape evidence record_id is invalid: {record_id!r}")
        if str(row.get("hwr_policy", "")) != configuration["auxiliary_policy"]:
            raise ValueError(f"singleton shape evidence policy mismatch: {record_id}")
        weight = float(row.get("hwr_fusion_weight", -1.0))
        if not math.isclose(weight, configuration["auxiliary_weight"], abs_tol=1e-12):
            raise ValueError(f"singleton shape evidence weight mismatch: {record_id}")
        candidates = [str(value) for value in row.get("final_topk", [])]
        probabilities = [float(value) for value in row.get("final_topk_probabilities", [])]
        if (
            not candidates or len(candidates) != len(probabilities)
            or len(candidates) != len(set(candidates))
            or not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in probabilities)
            or any(left < right for left, right in zip(probabilities, probabilities[1:]))
        ):
            raise ValueError(f"singleton shape evidence candidates are invalid: {record_id}")
        output[record_id] = {
            "formula_id": str(row.get("formula_id", "")),
            "tokens": candidates,
            "probabilities": probabilities,
        }
    return output


def apply_singleton_shape_rescue(
    rows: list[dict], predictions: dict[str, str], auxiliary_rows: list[dict],
    configuration: dict | None = None,
) -> tuple[dict[str, str], dict]:
    """Select one original candidate when auxiliary shape evidence is strong."""
    configuration = validate_configuration(configuration or DEFAULT_CONFIGURATION)
    record_ids = {str(row["record_id"]) for row in rows}
    if set(predictions) != record_ids:
        raise ValueError("singleton shape rescue prediction coverage mismatch")
    evidence = _evidence_by_record(auxiliary_rows, configuration)
    if set(evidence) != record_ids:
        raise ValueError("singleton shape rescue evidence coverage mismatch")
    output = {record_id: str(token) for record_id, token in predictions.items()}
    changes = []
    skipped = Counter()
    thresholds = configuration["token_confidence_thresholds"]
    maximum_ranks = configuration["token_candidate_maximum_ranks"]
    for formula_id, sequence in masked._formulae(rows).items():
        if len(sequence) != 1:
            skipped["context_available"] += 1
            continue
        row = sequence[0]
        record_id = str(row["record_id"])
        item = evidence[record_id]
        if item["formula_id"] != str(formula_id):
            raise ValueError(f"singleton shape evidence formula mismatch: {record_id}")
        ranked = []
        for token, threshold in thresholds.items():
            candidates = item["tokens"][:maximum_ranks[token]]
            if token not in candidates:
                continue
            rank = item["tokens"].index(token) + 1
            confidence = float(item["probabilities"][rank - 1])
            ranked.append((token, confidence, rank, float(threshold)))
        if not ranked:
            skipped["token_not_admitted"] += 1
            continue
        confident = [item for item in ranked if item[1] >= item[3]]
        if not confident:
            skipped["below_confidence_threshold"] += 1
            continue
        preserved = [item for item in confident if item[0] in row["final_topk"]]
        if not preserved:
            skipped["outside_original_top5"] += 1
            continue
        token, confidence, rank, threshold = max(
            preserved, key=lambda item: (item[1], -item[2], item[0]),
        )
        before = output[record_id]
        if before == token:
            skipped["already_selected"] += 1
            continue
        output[record_id] = token
        changes.append({
            "formula_id": str(formula_id),
            "record_id": record_id,
            "before": before,
            "after": token,
            "auxiliary_confidence": confidence,
            "auxiliary_candidate_rank": rank,
            "threshold": float(threshold),
        })
    if any(output[str(row["record_id"])] not in row["final_topk"] for row in rows):
        raise AssertionError("singleton shape rescue invented a candidate")
    return output, {
        "schema": SCHEMA,
        "status": "shadow_runtime_only",
        "configuration": configuration,
        "formulas": len(masked._formulae(rows)),
        "changed": len(changes),
        "changes": changes,
        "skipped": dict(sorted(skipped.items())),
        "candidate_preservation_rate": 1.0,
        "new_tokens": 0,
        "deleted_glyphs": 0,
        "grouping_mutations": 0,
    }


def _self_test() -> None:
    rows = [{
        "record_id": "r", "formula_id": "f",
        "final_topk": ["1", "/"], "final_topk_probabilities": [0.8, 0.2],
        "context": {"index": 0, "length": 1}, "geometry": {},
    }]
    evidence = [{
        "record_id": "r", "formula_id": "f",
        "final_topk": ["/", "1"], "final_topk_probabilities": [0.51, 0.49],
        "hwr_policy": DEFAULT_CONFIGURATION["auxiliary_policy"],
        "hwr_fusion_weight": DEFAULT_CONFIGURATION["auxiliary_weight"],
    }]
    rescued, audit = apply_singleton_shape_rescue(rows, {"r": "1"}, evidence)
    assert rescued == {"r": "/"} and audit["changed"] == 1
    evidence[0]["final_topk_probabilities"] = [0.49, 0.48]
    preserved, _ = apply_singleton_shape_rescue(rows, {"r": "1"}, evidence)
    assert preserved == {"r": "1"}
    times_rows = [{
        **rows[0], "final_topk": ["x", r"\times"],
    }]
    times_evidence = [{
        **evidence[0], "final_topk": ["X", r"\times"],
        "final_topk_probabilities": [0.44, 0.41],
    }]
    rescued, audit = apply_singleton_shape_rescue(
        times_rows, {"r": "x"}, times_evidence,
    )
    assert rescued == {"r": r"\times"}
    assert audit["changes"][0]["auxiliary_candidate_rank"] == 2
    legacy_configuration = {
        key: value for key, value in DEFAULT_CONFIGURATION.items()
        if key != "token_candidate_maximum_ranks"
    }
    preserved, _ = apply_singleton_shape_rescue(
        times_rows, {"r": "x"}, times_evidence, legacy_configuration,
    )
    assert preserved == {"r": "x"}


if __name__ == "__main__":
    _self_test()
    print('{"self_test":"pass"}')
