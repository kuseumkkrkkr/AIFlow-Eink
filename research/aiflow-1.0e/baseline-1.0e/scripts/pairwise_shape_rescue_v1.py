#!/usr/bin/env python3
"""Candidate-preserving binary shape rescue for frozen HWR embeddings.

The expert may replace a finalized ``p`` with ``8`` only when ``8`` already
exists in the original production HWR Top-20 and every fixed gate passes.  It
does not inspect formula truth, writer identity, formula length, or arithmetic.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


MODEL_SCHEMA = "aiflow-pairwise-shape-expert/v1"
CONFIG_SCHEMA = "aiflow-pairwise-shape-rescue-config/v1"
RESCUE_SCHEMA = "aiflow-pairwise-shape-rescue/v1"
FEATURE_DIMENSION = 128


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_expert(payload: dict[str, Any]) -> dict[str, Any]:
    coefficients = np.asarray(payload.get("coefficients"), dtype=np.float64)
    intercept = float(payload.get("intercept", math.nan))
    artifacts = dict(payload.get("artifacts") or {})
    training = dict(payload.get("training") or {})
    if (
        payload.get("schema") != MODEL_SCHEMA
        or payload.get("status") != "development_only_posthoc_shadow"
        or payload.get("model_type") != "binary_logistic_regression"
        or payload.get("feature") != "frozen_hwr_encoder_embedding"
        or payload.get("feature_dimension") != FEATURE_DIMENSION
        or payload.get("negative_class") != "p"
        or payload.get("positive_class") != "8"
        or coefficients.shape != (FEATURE_DIMENSION,)
        or not np.isfinite(coefficients).all()
        or not math.isfinite(intercept)
        or payload.get("input_mode") != "uniform-time"
        or len(str(artifacts.get("hwr_checkpoint_sha256", ""))) != 64
        or training.get("external_train_only") is not True
        or training.get("class_balanced") is not True
        or training.get("formula_truth_input") is not False
        or training.get("writer_identity_input") is not False
        or payload.get("requires_explicit_shadow_opt_in") is not True
        or payload.get("product_default_enabled") is not False
    ):
        raise ValueError("pairwise shape expert contract mismatch")
    return {
        **payload,
        "coefficients": coefficients,
        "intercept": intercept,
        "artifacts": artifacts,
        "training": training,
    }


def validate_configuration(
    configuration: dict[str, Any], *, expert_model_sha256: str | None = None,
) -> dict[str, Any]:
    gate = dict(configuration.get("gate") or {})
    contracts = dict(configuration.get("contracts") or {})
    margin = float(gate.get("minimum_expert_margin", math.nan))
    ratio = float(gate.get("minimum_hwr_probability_ratio", math.nan))
    width = gate.get("hwr_candidate_width")
    if (
        configuration.get("schema") != CONFIG_SCHEMA
        or configuration.get("rescue_schema") != RESCUE_SCHEMA
        or configuration.get("status") != "development_only_posthoc_shadow"
        or gate.get("baseline_token") != "p"
        or gate.get("challenger_token") != "8"
        or not math.isfinite(margin)
        or margin < 0.0
        or not math.isfinite(ratio)
        or ratio < 0.0
        or isinstance(width, bool)
        or not isinstance(width, int)
        or width != 20
        or gate.get("challenger_must_be_in_original_hwr_candidates") is not True
        or contracts.get("candidate_preserving") is not True
        or contracts.get("insertions_or_deletions") != 0
        or contracts.get("glyph_order_mutations") != 0
        or contracts.get("target_label_or_glyph_count_input") is not False
        or contracts.get("writer_identity_input") is not False
        or contracts.get("arithmetic_evaluation") is not False
        or configuration.get("requires_explicit_shadow_opt_in") is not True
        or configuration.get("product_default_enabled") is not False
    ):
        raise ValueError("pairwise shape rescue configuration mismatch")
    configured_hash = str(configuration.get("expert_model_sha256", ""))
    if len(configured_hash) != 64:
        raise ValueError("pairwise shape expert hash is missing")
    if expert_model_sha256 is not None and configured_hash != expert_model_sha256:
        raise ValueError("pairwise shape expert hash mismatch")
    return {
        **configuration,
        "gate": {
            **gate,
            "minimum_expert_margin": margin,
            "minimum_hwr_probability_ratio": ratio,
            "hwr_candidate_width": width,
        },
        "contracts": contracts,
    }


def load_pairwise_shape_artifacts(
    expert_path: Path, configuration_path: Path, *, hwr_checkpoint_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    expert_path = Path(expert_path).expanduser().resolve()
    configuration_path = Path(configuration_path).expanduser().resolve()
    for path in (expert_path, configuration_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    expert_sha256 = _sha256(expert_path)
    configuration_sha256 = _sha256(configuration_path)
    expert = validate_expert(json.loads(expert_path.read_text(encoding="utf-8")))
    configuration = validate_configuration(
        json.loads(configuration_path.read_text(encoding="utf-8")),
        expert_model_sha256=expert_sha256,
    )
    if expert["artifacts"]["hwr_checkpoint_sha256"] != hwr_checkpoint_sha256:
        raise ValueError("pairwise shape expert belongs to another HWR checkpoint")
    return expert, configuration, expert_sha256, configuration_sha256


def expert_margin(embedding: np.ndarray, expert: dict[str, Any]) -> float:
    vector = np.asarray(embedding, dtype=np.float64)
    if vector.shape != (FEATURE_DIMENSION,) or not np.isfinite(vector).all():
        raise ValueError("pairwise shape evidence requires one finite 128-D embedding")
    return float(vector @ expert["coefficients"] + expert["intercept"])


def apply_pairwise_shape_rescue(
    finalized_rows: list[dict[str, Any]],
    evidence_by_record: dict[str, dict[str, Any]],
    expert_payload: dict[str, Any],
    configuration_payload: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expert = validate_expert(expert_payload)
    configuration = validate_configuration(configuration_payload)
    gate = configuration["gate"]
    record_ids = [str(row["record_id"]) for row in finalized_rows]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("duplicate finalized record in pairwise shape rescue")
    if set(record_ids) != set(evidence_by_record):
        raise ValueError("pairwise shape evidence coverage mismatch")

    output = []
    changes = []
    considered = 0
    rejection_counts: dict[str, int] = {
        "baseline_not_p": 0,
        "challenger_outside_top20": 0,
        "probability_ratio_below_gate": 0,
        "expert_margin_below_gate": 0,
    }
    for source in finalized_rows:
        record_id = str(source["record_id"])
        evidence = dict(evidence_by_record[record_id])
        candidates = [str(value) for value in evidence.get("hwr_top20", [])]
        probabilities = [float(value) for value in evidence.get("hwr_top20_probabilities", [])]
        if (
            len(candidates) != gate["hwr_candidate_width"]
            or len(probabilities) != len(candidates)
            or len(set(candidates)) != len(candidates)
            or any(not math.isfinite(value) or value < 0.0 for value in probabilities)
        ):
            raise ValueError(f"invalid original HWR Top-20 evidence: {record_id}")

        token = str(source["finalized_top1"])
        reason = None
        margin = expert_margin(evidence["embedding"], expert)
        challenger_rank = (
            candidates.index(gate["challenger_token"]) + 1
            if gate["challenger_token"] in candidates else None
        )
        probability_by_token = dict(zip(candidates, probabilities, strict=True))
        ratio = (
            probability_by_token.get(gate["challenger_token"], 0.0)
            / max(probability_by_token.get(gate["baseline_token"], 0.0), 1e-12)
        )
        if token != gate["baseline_token"]:
            reason = "baseline_not_p"
        else:
            considered += 1
            if challenger_rank is None:
                reason = "challenger_outside_top20"
            elif ratio < gate["minimum_hwr_probability_ratio"]:
                reason = "probability_ratio_below_gate"
            elif margin < gate["minimum_expert_margin"]:
                reason = "expert_margin_below_gate"
        changed = reason is None
        if changed:
            token = gate["challenger_token"]
            changes.append({
                "record_id": record_id,
                "formula_id": str(source.get("formula_id", "")),
                "from": gate["baseline_token"],
                "to": gate["challenger_token"],
                "expert_margin_8_over_p": margin,
                "hwr_challenger_rank": challenger_rank,
                "hwr_probability_ratio_8_over_p": ratio,
            })
        else:
            rejection_counts[reason] += 1
        output.append({
            **source,
            "finalized_top1": token,
            "changed": token != str(evidence["hwr_top1"]),
            "decision_source": (
                "pairwise_shape_rescue_v1"
                if changed else source.get("decision_source")
            ),
        })
    return output, {
        "schema": RESCUE_SCHEMA,
        "enabled": True,
        "records": len(finalized_rows),
        "considered_finalized_p": considered,
        "changed": len(changes),
        "changed_formula_ids": sorted({row["formula_id"] for row in changes}),
        "changes": changes,
        "rejection_counts": rejection_counts,
        "candidate_preserving": True,
        "insertions_or_deletions": 0,
        "glyph_order_mutations": 0,
        "target_label_or_glyph_count_input": False,
        "writer_identity_input": False,
        "arithmetic_evaluation": False,
        "product_default_enabled": False,
    }


def _self_test() -> None:
    coefficients = [0.0] * FEATURE_DIMENSION
    coefficients[0] = 1.0
    expert = {
        "schema": MODEL_SCHEMA,
        "status": "development_only_posthoc_shadow",
        "model_type": "binary_logistic_regression",
        "feature": "frozen_hwr_encoder_embedding",
        "feature_dimension": FEATURE_DIMENSION,
        "negative_class": "p",
        "positive_class": "8",
        "coefficients": coefficients,
        "intercept": 0.0,
        "input_mode": "uniform-time",
        "artifacts": {"hwr_checkpoint_sha256": "0" * 64},
        "training": {
            "external_train_only": True,
            "class_balanced": True,
            "formula_truth_input": False,
            "writer_identity_input": False,
        },
        "requires_explicit_shadow_opt_in": True,
        "product_default_enabled": False,
    }
    configuration = {
        "schema": CONFIG_SCHEMA,
        "rescue_schema": RESCUE_SCHEMA,
        "status": "development_only_posthoc_shadow",
        "expert_model_sha256": "1" * 64,
        "gate": {
            "baseline_token": "p",
            "challenger_token": "8",
            "minimum_expert_margin": 0.1,
            "hwr_candidate_width": 20,
            "minimum_hwr_probability_ratio": 0.05,
            "challenger_must_be_in_original_hwr_candidates": True,
        },
        "contracts": {
            "candidate_preserving": True,
            "insertions_or_deletions": 0,
            "glyph_order_mutations": 0,
            "target_label_or_glyph_count_input": False,
            "writer_identity_input": False,
            "arithmetic_evaluation": False,
        },
        "requires_explicit_shadow_opt_in": True,
        "product_default_enabled": False,
    }
    row = {
        "record_id": "r", "formula_id": "f", "finalized_top1": "p",
        "changed": False, "decision_source": "formula_context_finalizer",
    }
    candidates = ["p", "8"] + [f"z{index}" for index in range(18)]
    evidence = {
        "r": {
            "embedding": np.asarray([0.5] + [0.0] * 127),
            "hwr_top1": "p",
            "hwr_top20": candidates,
            "hwr_top20_probabilities": [0.5, 0.03] + [0.0] * 18,
        }
    }
    rescued, audit = apply_pairwise_shape_rescue(
        [row], evidence, expert, configuration,
    )
    assert rescued[0]["finalized_top1"] == "8" and audit["changed"] == 1
    evidence["r"]["embedding"][0] = 0.082
    unchanged, audit = apply_pairwise_shape_rescue(
        [row], evidence, expert, configuration,
    )
    assert unchanged[0]["finalized_top1"] == "p" and audit["changed"] == 0


def main() -> int:
    _self_test()
    print(json.dumps({"self_test": "pass", "schema": RESCUE_SCHEMA}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
