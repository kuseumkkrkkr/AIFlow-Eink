#!/usr/bin/env python3
"""Research-only raw online-ink to candidate-preserving formula runtime.

The runtime selects one exact-cover stroke partition from a frozen Top-N
lattice, classifies each selected group with the frozen HWR head, and lets the
frozen masked-context finalizer choose only among supplied HWR candidates.  It
never inserts/deletes a glyph, evaluates arithmetic, or uses a target label or
target glyph count.  The current partition gate was tuned after inspecting its
chronological evaluation set, so callers must explicitly opt into shadow use.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch

from evaluate_48hz_prefix_v1 import DEFAULT_PRODUCT, _load_model, _sha256
from evaluate_joint_hwr_grouping_v1 import _candidate_embeddings, _probabilities
from dual_hwr_numeric_rescue_v1 import (
    apply_dual_hwr_numeric_rescue,
    validate_configuration as validate_dual_numeric_configuration,
)
from wide_candidate_syntax_rescue_v1 import (
    apply_wide_candidate_syntax_rescue,
    validate_configuration as validate_wide_syntax_configuration,
)
from formula_placement_rescue_v1 import (
    CONFIG_SCHEMA as PLACEMENT_CONFIG_SCHEMA, SCHEMA as PLACEMENT_SCHEMA,
    apply_formula_placement_rescue,
    validate_configuration as validate_placement_configuration,
)
from straight_equality_slot_rescue_v1 import (
    CONFIG_SCHEMA as EQUALITY_CONFIG_SCHEMA, SCHEMA as EQUALITY_SCHEMA,
    apply_straight_equality_slot_rescue,
    validate_configuration as validate_equality_configuration,
)
from latin_t_context_rescue_v1 import (
    apply_latin_t_context_rescue, build_latin_auxiliary_rows,
    load_latin_auxiliary_model,
    validate_configuration as validate_latin_t_configuration,
)
from evaluate_partition_context_ranker_v1 import (
    FEATURE_NAMES, POSTHOC_DESIGN_GUARD, SCHEMA as RANKER_SCHEMA,
    _apply_cross_merge_token_locks, _attach_features, _auxiliary_rows,
    _cross_merge_selection, _gate_accept, _geometry_selection, _partition_rows,
    _relation_merge_selection, _repeat_merge_selection, _select,
    _shadow_configuration,
    _validate_cross_merge_configuration, _validate_relation_merge_configuration,
    _validate_repeat_merge_configuration,
)
from finalize_formula_context_v1 import DEFAULT_CONTEXT, OwnedFormulaContextFinalizer
from formula_layout_v1 import finalize_formula_outputs, selected_layout_evidence_rows
from formula_acceptance_guard_v1 import (
    apply_formula_acceptance_guard, load_configuration as load_acceptance_configuration,
)
from singleton_shape_rescue_v1 import (
    SCHEMA as SINGLETON_SCHEMA, apply_singleton_shape_rescue,
    validate_configuration as validate_singleton_configuration,
)
from pairwise_shape_rescue_v1 import (
    apply_pairwise_shape_rescue,
    load_pairwise_shape_artifacts,
)
from stroke_grouping_v1 import build_lattice, candidate_features, enumerate_partitions
from train_project_owned_grouping_v1 import Sample


SCHEMA = "aiflow-raw-formula-context-runtime/v1"
OUTPUT_SCHEMA = "aiflow-raw-formula-context-result/v1"
CANDIDATE_FUSION_CONFIG_SCHEMA = "aiflow-raw-candidate-context-fusion-runtime-config/v1"


def _device(name: str) -> torch.device:
    resolved = "cuda" if name == "auto" and torch.cuda.is_available() else (
        "cpu" if name == "auto" else name
    )
    if resolved == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    return torch.device(resolved)


def _validate_strokes(source: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    formula_id = str(source.get("formula_id") or source.get("sample_id") or "").strip()
    if not formula_id:
        raise ValueError("raw formula requires formula_id or sample_id")
    strokes = list(source.get("strokes") or [])
    if not strokes:
        raise ValueError(f"raw formula has no strokes: {formula_id}")
    ordered = sorted(strokes, key=lambda row: int(row.get("order", -1)))
    orders = [int(row.get("order", -1)) for row in ordered]
    if orders != list(range(len(ordered))):
        raise ValueError(
            f"stroke orders must be unique contiguous zero-based indices: {formula_id}"
        )
    for stroke in ordered:
        points = list(stroke.get("points") or [])
        if not points:
            raise ValueError(f"empty stroke in formula: {formula_id}")
        previous_time = -math.inf
        for point in points:
            x = float(point["x"]); y = float(point["y"])
            time = float(point.get("t_ms", 0.0))
            if not all(math.isfinite(value) for value in (x, y, time)):
                raise ValueError(f"non-finite stroke point in formula: {formula_id}")
            if time < previous_time:
                raise ValueError(f"stroke timestamps must be monotonic: {formula_id}")
            previous_time = time
    return formula_id, ordered


def _load_inputs(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(payload, dict):
        payload = payload.get("formulas", [payload])
    if not isinstance(payload, list) or not payload or any(not isinstance(row, dict) for row in payload):
        raise ValueError("raw input must be one formula, a formula list, or JSONL")
    return payload


def _apply_singleton_finalized_rows(
    runtime_rows: list[dict], finalized: list[dict], auxiliary_rows: list[dict],
    configuration: dict,
) -> tuple[list[dict], dict]:
    predictions = {
        str(row["record_id"]): str(row["finalized_top1"])
        for row in finalized
    }
    rescued, audit = apply_singleton_shape_rescue(
        runtime_rows, predictions, auxiliary_rows, configuration,
    )
    hwr_top1 = {
        str(row["record_id"]): str(row["final_topk"][0])
        for row in runtime_rows
    }
    output = []
    for source in finalized:
        record_id = str(source["record_id"])
        token = str(rescued[record_id])
        changed_by_rescue = token != str(source["finalized_top1"])
        output.append({
            **source,
            "finalized_top1": token,
            "changed": token != hwr_top1[record_id],
            "decision_source": (
                "singleton_shape_rescue"
                if changed_by_rescue
                else source.get("decision_source", "formula_context_finalizer")
            ),
        })
    return output, audit


def _formula_layout_shadow(
    runtime_rows: list[dict], finalized: list[dict],
) -> tuple[dict, dict]:
    predictions = {
        str(row["record_id"]): str(row["finalized_top1"])
        for row in finalized
    }
    if len(predictions) != len(finalized):
        raise ValueError("formula layout finalized record ids must be unique")
    layout_rows, evidence_audit = selected_layout_evidence_rows(runtime_rows, predictions)
    formulae, audit = finalize_formula_outputs(layout_rows, predictions)
    if len(formulae) != 1:
        raise AssertionError("raw runtime must emit exactly one layout formula")
    return formulae[0], {
        **audit,
        "enabled": True,
        **evidence_audit,
        "feedback_into_character_model": False,
        "product_default_enabled": False,
    }


@dataclass(frozen=True)
class RawFormulaContextRuntimeV1:
    ranker_payload: dict[str, Any]
    hwr: Any
    labels: list[str]
    finalizer: OwnedFormulaContextFinalizer
    device: torch.device
    partition_ranker_sha256: str
    singleton_hwr: Any | None
    singleton_configuration: dict[str, Any] | None
    numeric_configuration: dict[str, Any] | None
    wide_syntax_configuration: dict[str, Any] | None
    relation_merge_configuration: dict[str, Any] | None
    cross_merge_configuration: dict[str, Any] | None
    repeat_merge_configuration: dict[str, Any] | None
    placement_configuration: dict[str, Any] | None
    equality_configuration: dict[str, Any] | None
    latin_hwr: Any | None
    latin_labels: list[str] | None
    latin_configuration: dict[str, Any] | None
    latin_hwr_sha256: str | None
    singleton_config_sha256: str | None
    singleton_hwr_sha256: str | None
    pairwise_shape_expert: dict[str, Any] | None
    pairwise_shape_configuration: dict[str, Any] | None
    pairwise_shape_expert_sha256: str | None
    pairwise_shape_config_sha256: str | None
    emit_formula_layout_shadow: bool

    @classmethod
    def from_artifacts(
        cls, partition_ranker: Path, hwr_checkpoint: Path, context_checkpoint: Path,
        *, device: str = "auto", batch_size: int = 128,
        formula_sequence_config: Path | None = None,
        formula_syntax_rescue_config: Path | None = None,
        candidate_context_fusion_config: Path | None = None,
        candidate_context_auxiliary_hwr_checkpoint: Path | None = None,
        formula_placement_config: Path | None = None,
        straight_equality_config: Path | None = None,
        latin_auxiliary_checkpoint: Path | None = None,
        pairwise_shape_expert: Path | None = None,
        pairwise_shape_config: Path | None = None,
        emit_formula_layout_shadow: bool = False,
        allow_posthoc_shadow: bool = False,
    ) -> "RawFormulaContextRuntimeV1":
        ranker_path = Path(partition_ranker).expanduser().resolve()
        hwr_path = Path(hwr_checkpoint).expanduser().resolve()
        context_path = Path(context_checkpoint).expanduser().resolve()
        for path in (ranker_path, hwr_path, context_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        payload = joblib.load(ranker_path)
        if (
            payload.get("schema") != RANKER_SCHEMA
            or tuple(payload.get("feature_names") or ()) != FEATURE_NAMES
            or "model" not in payload
            or "grouping_model" not in payload
        ):
            raise ValueError("partition-context ranker artifact contract mismatch")
        if payload.get("requires_explicit_shadow_opt_in") is not True:
            raise ValueError("partition-context artifact lacks explicit shadow contract")
        if not allow_posthoc_shadow:
            raise ValueError(
                "posthoc partition gate is development-only; explicit shadow opt-in is required"
            )
        if payload.get("product_default_enabled") is not False:
            raise ValueError("partition-context product gate must remain disabled")
        if payload.get("posthoc_test_tuning") is not True:
            raise ValueError("partition-context tuning provenance is missing")
        if payload.get("evaluation_only_writer_loo") is not False:
            raise ValueError("writer-LOO evaluation artifact is not a deployable runtime")
        if dict(payload.get("selection_guard") or {}) != POSTHOC_DESIGN_GUARD:
            raise ValueError("partition-context selection guard mismatch")
        if _sha256(hwr_path) != str(payload.get("hwr_checkpoint_sha256")):
            raise ValueError("partition-context HWR checkpoint hash mismatch")
        if _sha256(context_path) != str(payload.get("context_checkpoint_sha256")):
            raise ValueError("partition-context context checkpoint hash mismatch")
        finalizer_contract = dict(payload.get("finalizer_contract") or {})
        expected_finalizer = {
            "formula_layout": True,
            "semantic_guards": True,
            "equation_correction": False,
        }
        if any(finalizer_contract.get(key) != value for key, value in expected_finalizer.items()):
            raise ValueError("partition-context finalizer contract mismatch")
        if dict(finalizer_contract.get("auxiliary_candidates") or {}).get("enabled") is not False:
            raise ValueError("unimplemented auxiliary candidate contract is not runtime admissible")
        sequence_path = (
            Path(formula_sequence_config).expanduser().resolve()
            if formula_sequence_config is not None else None
        )
        syntax_path = (
            Path(formula_syntax_rescue_config).expanduser().resolve()
            if formula_syntax_rescue_config is not None else None
        )
        for name, path, hash_field in (
            ("formula sequence", sequence_path, "formula_sequence_config_sha256"),
            ("formula syntax rescue", syntax_path, "formula_syntax_rescue_config_sha256"),
        ):
            expected_hash = finalizer_contract.get(hash_field)
            if bool(path) != bool(expected_hash):
                raise ValueError(f"{name} config presence does not match ranker artifact")
            if path is not None and (not path.is_file() or _sha256(path) != expected_hash):
                raise ValueError(f"{name} config hash mismatch")
        resolved_device = _device(device)
        hwr, labels, _ = _load_model(hwr_path, resolved_device)
        if bool(pairwise_shape_expert) != bool(pairwise_shape_config):
            raise ValueError(
                "pairwise shape expert and configuration are required together"
            )
        if bool(candidate_context_fusion_config) != bool(candidate_context_auxiliary_hwr_checkpoint):
            raise ValueError(
                "candidate context fusion config and auxiliary HWR checkpoint are required together"
            )
        if bool(formula_placement_config) != bool(straight_equality_config):
            raise ValueError(
                "formula placement and straight equality configs are required together"
            )
        if formula_placement_config is not None and candidate_context_fusion_config is None:
            raise ValueError("formula placement requires candidate context fusion")
        singleton_hwr = None
        singleton_configuration = None
        numeric_configuration = None
        wide_syntax_configuration = None
        relation_merge_configuration = None
        cross_merge_configuration = None
        repeat_merge_configuration = None
        placement_configuration = None
        equality_configuration = None
        latin_hwr = None
        latin_labels = None
        latin_configuration = None
        latin_hwr_sha256 = None
        singleton_config_sha256 = None
        singleton_hwr_sha256 = None
        pairwise_expert_payload = None
        pairwise_configuration = None
        pairwise_expert_sha256 = None
        pairwise_config_sha256 = None
        if pairwise_shape_expert is not None:
            (
                pairwise_expert_payload,
                pairwise_configuration,
                pairwise_expert_sha256,
                pairwise_config_sha256,
            ) = load_pairwise_shape_artifacts(
                pairwise_shape_expert,
                pairwise_shape_config,
                hwr_checkpoint_sha256=_sha256(hwr_path),
            )
        if candidate_context_fusion_config is not None:
            singleton_config_path = Path(candidate_context_fusion_config).expanduser().resolve()
            singleton_hwr_path = Path(
                candidate_context_auxiliary_hwr_checkpoint,
            ).expanduser().resolve()
            for path in (singleton_config_path, singleton_hwr_path):
                if not path.is_file():
                    raise FileNotFoundError(path)
            singleton_payload = json.loads(
                singleton_config_path.read_text(encoding="utf-8")
            )
            gate = dict(singleton_payload.get("gate") or {})
            provenance = dict(singleton_payload.get("provenance") or {})
            artifacts = dict(singleton_payload.get("artifacts") or {})
            mode = dict(singleton_payload.get("mode") or {})
            rerank_mode = dict(mode.get("restricted_candidate_rerank") or {})
            singleton_mode = dict(mode.get("singleton_shape_rescue") or {})
            numeric_mode = dict(mode.get("dual_numeric_rescue") or {})
            wide_syntax_mode = dict(mode.get("wide_numeric_syntax_rescue") or {})
            relation_merge_mode = dict(mode.get("relation_merge_rescue") or {})
            cross_merge_mode = dict(mode.get("cross_merge_rescue") or {})
            repeat_merge_mode = dict(mode.get("repeat_merge_rescue") or {})
            placement_mode = dict(mode.get("formula_placement_rescue") or {})
            equality_mode = dict(mode.get("straight_equality_rescue") or {})
            latin_mode = dict(mode.get("latin_t_context_rescue") or {})
            legacy_singleton_mode = {
                "enabled": True,
                "preserve_original_candidate_set": True,
            }
            ranked_singleton_mode = {
                **legacy_singleton_mode,
                "token_candidate_maximum_ranks": {"/": 1, r"\times": 2},
            }
            if (
                singleton_payload.get("schema") != CANDIDATE_FUSION_CONFIG_SCHEMA
                or singleton_payload.get("rescue_schema") != SINGLETON_SCHEMA
                or gate.get("shadow_runtime_admitted") is not True
                or gate.get("product_default_admitted") is not False
                or provenance.get("current_96_formula_training_overlap") is not True
                or rerank_mode != {
                    "enabled": True,
                    "candidate_width": 5,
                    "preserve_original_candidate_set": True,
                    "reapply_formula_context_finalizer": True,
                }
                or singleton_mode not in (
                    legacy_singleton_mode, ranked_singleton_mode,
                )
                or numeric_mode != {
                    "enabled": True,
                    "maximum_operand_changes": 2,
                    "candidate_contract": "baseline_and_auxiliary_top5_union",
                    "operator_relation_or_fence_mutations": 0,
                    "arithmetic_evaluation": False,
                }
                or wide_syntax_mode != {
                    "enabled": True,
                    "candidate_width": 10,
                    "maximum_changes": 2,
                    "candidate_contract": "baseline_and_auxiliary_top10_union",
                    "minimum_digit_probability_ratio": 0.005,
                    "allow_equality_replacement": False,
                    "arithmetic_evaluation": False,
                }
                or relation_merge_mode != {
                    "enabled": True,
                    "maximum_candidate_rank": 3,
                    "exact_cover_coarsening_only": True,
                    "required_merged_strokes": 3,
                    "negated_relation_candidate_required": True,
                    "target_label_or_glyph_count_input": False,
                    "arithmetic_evaluation": False,
                }
            ):
                raise ValueError("candidate context fusion shadow contract mismatch")
            cross_payload = singleton_payload.get("cross_merge_configuration")
            if bool(cross_payload) != bool(cross_merge_mode):
                raise ValueError("candidate context cross merge presence mismatch")
            if cross_payload:
                cross_merge_configuration = _validate_cross_merge_configuration(
                    dict(cross_payload),
                )
                legacy_cross_mode = {
                    "enabled": True,
                    "maximum_candidate_rank": 3,
                    "exact_cover_coarsening_only": True,
                    "required_merged_strokes": 2,
                    "merged_hwr_and_context_token": "x",
                    "preserve_merged_x_after_auxiliary_fusion": True,
                    "target_label_or_glyph_count_input": False,
                    "arithmetic_evaluation": False,
                }
                extended_cross_mode = {
                    **legacy_cross_mode,
                    "auxiliary_nested_expression_enabled": True,
                    "auxiliary_candidate_contract": "product_fused_top20",
                    "auxiliary_x_candidate_maximum_rank": 3,
                    "auxiliary_open_fence_candidate_maximum_rank": 1,
                    "auxiliary_plus_geometry_source": "formula_role_typo",
                }
                if cross_merge_mode not in (
                    legacy_cross_mode, extended_cross_mode,
                ):
                    raise ValueError("candidate context cross merge mode mismatch")
                if (
                    cross_merge_configuration["maximum_candidate_rank"]
                    != cross_merge_mode["maximum_candidate_rank"]
                    or cross_merge_configuration["required_merged_strokes"]
                    != cross_merge_mode["required_merged_strokes"]
                    or cross_merge_configuration[
                        "preserve_merged_x_after_auxiliary_fusion"
                    ] is not True
                    or cross_merge_configuration[
                        "auxiliary_nested_expression_enabled"
                    ] != (cross_merge_mode == extended_cross_mode)
                    or (
                        cross_merge_mode == extended_cross_mode
                        and (
                            cross_merge_configuration[
                                "auxiliary_x_candidate_maximum_rank"
                            ] != cross_merge_mode[
                                "auxiliary_x_candidate_maximum_rank"
                            ]
                            or cross_merge_configuration[
                                "auxiliary_open_fence_candidate_maximum_rank"
                            ] != cross_merge_mode[
                                "auxiliary_open_fence_candidate_maximum_rank"
                            ]
                        )
                    )
                ):
                    raise ValueError("candidate context cross merge configuration mismatch")
            repeat_payload = singleton_payload.get("repeat_merge_configuration")
            if bool(repeat_payload) != bool(repeat_merge_mode):
                raise ValueError("candidate context repeat merge presence mismatch")
            if repeat_payload:
                repeat_merge_configuration = _validate_repeat_merge_configuration(
                    dict(repeat_payload),
                )
                if repeat_merge_mode != {
                    "enabled": True,
                    "maximum_candidate_rank": 3,
                    "exact_cover_coarsening_only": True,
                    "required_merged_strokes": 2,
                    "required_formula_glyphs": 3,
                    "same_formula_repeated_shape_dtw_required": True,
                    "hwr_context_agreement_required": True,
                    "preserve_merged_token_after_auxiliary_fusion": True,
                    "target_label_or_glyph_count_input": False,
                    "arithmetic_evaluation": False,
                }:
                    raise ValueError("candidate context repeat merge mode mismatch")
                if (
                    repeat_merge_configuration["maximum_candidate_rank"]
                    != repeat_merge_mode["maximum_candidate_rank"]
                    or repeat_merge_configuration["required_merged_strokes"]
                    != repeat_merge_mode["required_merged_strokes"]
                    or repeat_merge_configuration["required_formula_glyphs"]
                    != repeat_merge_mode["required_formula_glyphs"]
                    or repeat_merge_configuration[
                        "preserve_merged_token_after_auxiliary_fusion"
                    ] is not True
                ):
                    raise ValueError("candidate context repeat merge configuration mismatch")
            placement_paths_present = (
                formula_placement_config is not None
                and straight_equality_config is not None
            )
            if placement_paths_present != bool(placement_mode and equality_mode):
                raise ValueError("candidate context placement config presence mismatch")
            placement_path = equality_path = None
            if placement_paths_present:
                placement_path = Path(formula_placement_config).expanduser().resolve()
                equality_path = Path(straight_equality_config).expanduser().resolve()
                for path in (placement_path, equality_path):
                    if not path.is_file():
                        raise FileNotFoundError(path)
                placement_raw, _ = _shadow_configuration(
                    placement_path, config_schema=PLACEMENT_CONFIG_SCHEMA,
                    rescue_schema=PLACEMENT_SCHEMA,
                )
                equality_raw, _ = _shadow_configuration(
                    equality_path, config_schema=EQUALITY_CONFIG_SCHEMA,
                    rescue_schema=EQUALITY_SCHEMA,
                )
                placement_configuration = validate_placement_configuration(
                    placement_raw,
                )
                equality_configuration = validate_equality_configuration(
                    equality_raw,
                )
                if (
                    cross_merge_configuration is not None
                    and cross_merge_configuration[
                        "auxiliary_nested_expression_enabled"
                    ]
                    and placement_configuration["formula_role_typo"][
                        "value_cross_plus_open_fence_enabled"
                    ] is not True
                ):
                    raise ValueError(
                        "nested cross merge requires open-fence plus placement"
                    )
                if placement_mode != {
                    "enabled": True,
                    "candidate_width": 20,
                    "candidate_contract": "product_fused_top20",
                    "insertions_or_deletions": 0,
                    "arithmetic_evaluation": False,
                } or equality_mode != {
                    "enabled": True,
                    "candidate_width": 20,
                    "candidate_contract": "product_fused_top20",
                    "maximum_changes_per_formula": 1,
                    "arithmetic_evaluation": False,
                }:
                    raise ValueError("candidate context placement mode mismatch")
            latin_payload = singleton_payload.get("latin_t_context_configuration")
            if bool(latin_payload) != bool(latin_mode):
                raise ValueError("candidate context Latin t rescue presence mismatch")
            if bool(latin_payload) != bool(latin_auxiliary_checkpoint):
                raise ValueError(
                    "candidate context Latin t config and checkpoint are required together"
                )
            latin_path = None
            if latin_payload:
                latin_configuration = validate_latin_t_configuration(
                    dict(latin_payload),
                )
                if latin_mode != {
                    "enabled": True,
                    "candidate_width": 5,
                    "candidate_contract": "approved_legacy_latin_auxiliary_top5",
                    "target_token": "t",
                    "single_parenthesized_function_argument_only": True,
                    "insertions_or_deletions": 0,
                    "target_label_or_glyph_count_input": False,
                    "arithmetic_evaluation": False,
                }:
                    raise ValueError("candidate context Latin t mode mismatch")
                latin_path = Path(latin_auxiliary_checkpoint).expanduser().resolve()
                if not latin_path.is_file():
                    raise FileNotFoundError(latin_path)
                latin_hwr, latin_labels = load_latin_auxiliary_model(
                    latin_path, resolved_device,
                )
                latin_hwr_sha256 = _sha256(latin_path)
            expected_hashes = {
                "partition_ranker_sha256": _sha256(ranker_path),
                "hwr_checkpoint_sha256": _sha256(hwr_path),
                "context_checkpoint_sha256": _sha256(context_path),
                "auxiliary_hwr_checkpoint_sha256": _sha256(singleton_hwr_path),
            }
            if placement_paths_present:
                expected_hashes.update({
                    "formula_placement_config_sha256": _sha256(placement_path),
                    "straight_equality_config_sha256": _sha256(equality_path),
                })
            if latin_path is not None:
                expected_hashes["latin_auxiliary_checkpoint_sha256"] = _sha256(
                    latin_path,
                )
            if any(artifacts.get(key) != value for key, value in expected_hashes.items()):
                raise ValueError("candidate context fusion artifact hash mismatch")
            evidence = dict(singleton_payload.get("evidence") or {})
            for name in ("selected_summary", "evaluation_evidence"):
                evidence_path = singleton_config_path.parent / str(
                    evidence.get(name, ""),
                )
                if (
                    not evidence_path.is_file()
                    or _sha256(evidence_path) != evidence.get(f"{name}_sha256")
                ):
                    raise ValueError(
                        f"candidate context fusion {name} hash mismatch"
                    )
            singleton_configuration = validate_singleton_configuration(
                dict(singleton_payload.get("configuration") or {}),
            )
            expected_singleton_ranks = singleton_mode.get(
                "token_candidate_maximum_ranks", {"/": 1, r"\times": 1},
            )
            if (
                singleton_configuration["token_candidate_maximum_ranks"]
                != expected_singleton_ranks
            ):
                raise ValueError("candidate context singleton rank mode mismatch")
            numeric_configuration = validate_dual_numeric_configuration(
                dict(singleton_payload.get("numeric_configuration") or {}),
            )
            wide_syntax_configuration = validate_wide_syntax_configuration(
                dict(singleton_payload.get("wide_syntax_configuration") or {}),
            )
            relation_merge_configuration = _validate_relation_merge_configuration(
                dict(singleton_payload.get("relation_merge_configuration") or {}),
            )
            if (
                relation_merge_configuration["maximum_candidate_rank"]
                != relation_merge_mode["maximum_candidate_rank"]
                or relation_merge_configuration["required_merged_strokes"]
                != relation_merge_mode["required_merged_strokes"]
            ):
                raise ValueError("candidate context relation merge mode mismatch")
            if singleton_configuration["auxiliary_policy"] != "old_new_product_probability_fusion":
                raise ValueError("candidate context fusion policy mismatch")
            if (
                numeric_configuration["auxiliary_policy"]
                != singleton_configuration["auxiliary_policy"]
                or numeric_configuration["auxiliary_weight"]
                != singleton_configuration["auxiliary_weight"]
            ):
                raise ValueError("candidate context fusion numeric policy mismatch")
            if (
                wide_syntax_configuration["allowed_auxiliary_policies"]
                != [singleton_configuration["auxiliary_policy"]]
                or wide_syntax_configuration["auxiliary_weight"]
                != singleton_configuration["auxiliary_weight"]
                or wide_syntax_configuration["candidate_width"] != 10
            ):
                raise ValueError("candidate context fusion wide syntax policy mismatch")
            if placement_configuration is not None and (
                placement_configuration["unmatched_fence_operand"]["auxiliary_weight"]
                != singleton_configuration["auxiliary_weight"]
                or equality_configuration["auxiliary_weight"]
                != singleton_configuration["auxiliary_weight"]
            ):
                raise ValueError("candidate context placement fusion policy mismatch")
            singleton_hwr, singleton_labels, _ = _load_model(
                singleton_hwr_path, resolved_device,
            )
            if singleton_labels != labels:
                raise ValueError("singleton auxiliary HWR vocabulary mismatch")
            singleton_state = singleton_hwr.state_dict()
            for name, value in hwr.state_dict().items():
                if not name.startswith("math_head.") and not torch.equal(
                    value.detach().cpu(), singleton_state[name].detach().cpu()
                ):
                    raise ValueError(f"singleton auxiliary and frozen HWR encoders differ: {name}")
            singleton_config_sha256 = _sha256(singleton_config_path)
            singleton_hwr_sha256 = _sha256(singleton_hwr_path)
        finalizer = OwnedFormulaContextFinalizer(
            context_path, hwr_path, device=str(resolved_device), batch_size=batch_size,
            semantic_guards=True, equation_correction=False, formula_layout=True,
            formula_sequence_config=sequence_path,
            formula_syntax_rescue_config=syntax_path,
        )
        return cls(
            payload, hwr, labels, finalizer, resolved_device, _sha256(ranker_path),
            singleton_hwr, singleton_configuration, numeric_configuration,
            wide_syntax_configuration, relation_merge_configuration,
            cross_merge_configuration, repeat_merge_configuration,
            placement_configuration,
            equality_configuration,
            latin_hwr, latin_labels, latin_configuration, latin_hwr_sha256,
            singleton_config_sha256,
            singleton_hwr_sha256,
            pairwise_expert_payload,
            pairwise_configuration,
            pairwise_expert_sha256,
            pairwise_config_sha256,
            bool(emit_formula_layout_shadow),
        )

    def _sample(self, source: dict[str, Any]) -> Sample:
        formula_id, strokes = _validate_strokes(source)
        lattice_config = dict(self.ranker_payload["lattice_config"])
        candidates = build_lattice(strokes, **lattice_config)
        return Sample(
            formula_id, "runtime", strokes, (), candidates,
            candidate_features(candidates, strokes),
        )

    def infer(self, source: dict[str, Any]) -> dict[str, Any]:
        sample = self._sample(source)
        embeddings, slices = _candidate_embeddings([sample], self.hwr, self.device)
        probabilities = _probabilities(self.hwr.math_head, embeddings, self.device)
        fused_probability = None
        if self.singleton_hwr is not None:
            auxiliary_probability = _probabilities(
                self.singleton_hwr.math_head, embeddings, self.device,
            )
            weight = float(self.singleton_configuration["auxiliary_weight"])
            fused_probability = (
                (1.0 - weight) * probabilities + weight * auxiliary_probability
            )
        local_probability = probabilities[slices[sample.sample_id]]
        grouping_probability = self.ranker_payload["grouping_model"].predict_proba(
            sample.features
        )[:, 1]
        logits = np.log(
            np.clip(grouping_probability, 1e-6, 1 - 1e-6)
            / np.clip(1 - grouping_probability, 1e-6, 1)
        )
        top_n = int(self.ranker_payload["top_n"])
        ranked = enumerate_partitions(
            sample.candidates, logits, len(sample.strokes), top_n=top_n,
        )
        if not ranked:
            raise ValueError(f"no complete stroke partition: {sample.sample_id}")
        candidate_index = {
            frozenset(row["source_indices"]): index
            for index, row in enumerate(sample.candidates)
        }
        best_geometry = float(ranked[0][0])
        partitions = []
        for rank, (score, groups) in enumerate(ranked, 1):
            partitions.append({
                "sample": sample,
                "rank": rank,
                "geometry_score": float(score),
                "geometry_delta": float(score) - best_geometry,
                "groups": groups,
                "rows": _partition_rows(
                    sample, rank, groups, local_probability, self.labels,
                    candidate_index,
                ),
                "truth": False,
            })
        _attach_features(
            partitions, self.finalizer.model, self.finalizer.contract,
            self.finalizer.payload, self.device,
        )
        baseline = _geometry_selection(partitions)[sample.sample_id]
        proposal = _select(
            partitions, self.ranker_payload["model"], len(FEATURE_NAMES),
        )[sample.sample_id]
        accepted = _gate_accept(
            baseline, proposal, self.ranker_payload["selection_guard"],
        )
        selected = proposal if accepted else baseline
        relation_merge_audit = {"enabled": False}
        if self.relation_merge_configuration is not None:
            relation_selected, relation_merge_audit = _relation_merge_selection(
                partitions, {sample.sample_id: selected},
                self.relation_merge_configuration,
            )
            selected = relation_selected[sample.sample_id]
        relation_merge_changed = bool(
            relation_merge_audit.get("changed_formulas", 0)
        )
        cross_merge_audit = {"enabled": False}
        if self.cross_merge_configuration is not None:
            cross_auxiliary = None
            if fused_probability is not None and self.placement_configuration is not None:
                cross_auxiliary = {
                    "probability": fused_probability,
                    "slices": slices,
                    "labels": self.labels,
                    "policy": self.singleton_configuration["auxiliary_policy"],
                    "weight": float(
                        self.singleton_configuration["auxiliary_weight"]
                    ),
                    "placement_configuration": self.placement_configuration,
                }
            cross_selected, cross_merge_audit = _cross_merge_selection(
                partitions, {sample.sample_id: selected},
                self.cross_merge_configuration,
                cross_auxiliary,
            )
            selected = cross_selected[sample.sample_id]
        cross_merge_changed = bool(cross_merge_audit.get("changed_formulas", 0))
        repeat_merge_audit = {"enabled": False}
        if self.repeat_merge_configuration is not None:
            repeat_selected, repeat_merge_audit = _repeat_merge_selection(
                partitions, {sample.sample_id: selected},
                self.repeat_merge_configuration,
            )
            selected = repeat_selected[sample.sample_id]
        repeat_merge_changed = bool(
            repeat_merge_audit.get("changed_formulas", 0)
        )
        runtime_rows = [
            {
                **{
                    key: value for key, value in row.items()
                    if key not in {"group", "full_probability", "grouping_features"}
                },
                "formula_id": sample.sample_id,
            }
            for row in selected["rows"]
        ]
        singleton_audit = {"enabled": False}
        if self.singleton_hwr is not None:
            if fused_probability is None:
                raise AssertionError("singleton fusion probability is missing")
            weight = float(self.singleton_configuration["auxiliary_weight"])
            restricted_rows = _auxiliary_rows(
                [sample], {sample.sample_id: selected}, fused_probability, slices,
                self.labels, width=5,
                policy=self.singleton_configuration["auxiliary_policy"],
                weight=weight, preserve_source_candidates=True,
            )
            finalized, finalizer_audit = self.finalizer.finalize(restricted_rows)
            singleton_rows = _auxiliary_rows(
                [sample], {sample.sample_id: selected}, fused_probability, slices,
                self.labels, width=5,
                policy=self.singleton_configuration["auxiliary_policy"],
                weight=weight,
            )
            finalized, singleton_audit = _apply_singleton_finalized_rows(
                runtime_rows, finalized, singleton_rows,
                self.singleton_configuration,
            )
            auxiliary_finalized, auxiliary_finalizer_audit = self.finalizer.finalize(
                singleton_rows,
            )
            finalized, numeric_audit = apply_dual_hwr_numeric_rescue(
                finalized, singleton_rows, auxiliary_finalized,
                self.numeric_configuration,
            )
            wide_rows = _auxiliary_rows(
                [sample], {sample.sample_id: selected}, fused_probability, slices,
                self.labels, width=10,
                policy=self.singleton_configuration["auxiliary_policy"],
                weight=weight,
            )
            finalized, wide_syntax_audit = apply_wide_candidate_syntax_rescue(
                finalized, wide_rows, self.wide_syntax_configuration,
            )
            placement_audit = {"enabled": False}
            equality_audit = {"enabled": False}
            if self.placement_configuration is not None:
                top20_rows = _auxiliary_rows(
                    [sample], {sample.sample_id: selected}, fused_probability,
                    slices, self.labels, width=20,
                    policy=self.singleton_configuration["auxiliary_policy"],
                    weight=weight,
                )
                finalized, placement_audit = apply_formula_placement_rescue(
                    finalized, top20_rows, self.placement_configuration,
                )
                finalized, equality_audit = apply_straight_equality_slot_rescue(
                    finalized, top20_rows, self.equality_configuration,
                )
            latin_audit = {"enabled": False}
            if self.latin_hwr is not None:
                latin_rows = build_latin_auxiliary_rows(
                    [sample], {sample.sample_id: selected}, self.latin_hwr,
                    self.latin_labels, self.device,
                )
                finalized, latin_audit = apply_latin_t_context_rescue(
                    finalized, latin_rows, self.latin_configuration,
                )
            finalized, cross_lock_audit = _apply_cross_merge_token_locks(
                finalized, {sample.sample_id: selected},
            )
            singleton_audit = {
                **singleton_audit,
                "restricted_candidate_rerank": {
                    "enabled": True,
                    "candidate_set_preserved": True,
                    "candidate_width": 5,
                    "formula_context_finalizer_reapplied": True,
                },
                "dual_numeric_auxiliary_finalizer": auxiliary_finalizer_audit,
                "dual_numeric_rescue": numeric_audit,
                "wide_numeric_syntax_rescue": wide_syntax_audit,
                "formula_placement_rescue": placement_audit,
                "straight_equality_rescue": equality_audit,
                "latin_t_context_rescue": latin_audit,
                "cross_merge_auxiliary_semantic_lock": cross_lock_audit,
            }
        else:
            finalized, finalizer_audit = self.finalizer.finalize(runtime_rows)
            latin_audit = {"enabled": False}
            finalized, cross_lock_audit = _apply_cross_merge_token_locks(
                finalized, {sample.sample_id: selected},
            )
        pairwise_shape_audit = {"enabled": False}
        if self.pairwise_shape_expert is not None:
            sample_slice = slices[sample.sample_id]
            offset = int(sample_slice.start or 0)
            evidence_by_record = {}
            for source in selected["rows"]:
                group = frozenset(int(value) for value in source["group"])
                embedding_index = offset + candidate_index[group]
                original_probability = probabilities[embedding_index]
                top20_indices = np.argsort(-original_probability)[:20]
                evidence_by_record[str(source["record_id"])] = {
                    "embedding": embeddings[embedding_index].numpy(),
                    "hwr_top1": self.labels[int(top20_indices[0])],
                    "hwr_top20": [
                        self.labels[int(value)] for value in top20_indices
                    ],
                    "hwr_top20_probabilities": [
                        float(original_probability[int(value)])
                        for value in top20_indices
                    ],
                }
            finalized, pairwise_shape_audit = apply_pairwise_shape_rescue(
                finalized,
                evidence_by_record,
                self.pairwise_shape_expert,
                self.pairwise_shape_configuration,
            )
        finalized.sort(key=lambda row: int(row["context_index"]))
        runtime_by_record = {str(row["record_id"]): row for row in runtime_rows}
        source_by_group = {
            frozenset(int(value) for value in row["group"]): row
            for row in selected["rows"]
        }
        groups = []
        candidate_map = {
            frozenset(row["source_indices"]): row for row in sample.candidates
        }
        ordered_groups = sorted(
            selected["groups"],
            key=lambda group: (
                candidate_map[frozenset(group)]["box"]["left"],
                candidate_map[frozenset(group)]["box"]["top"], min(group),
            ),
        )
        for index, group in enumerate(ordered_groups):
            box = candidate_map[frozenset(group)]["box"]
            groups.append({
                "partition_input_index": index,
                "record_id": str(source_by_group[frozenset(group)]["record_id"]),
                "stroke_indices": sorted(int(value) for value in group),
                "box": {key: float(value) for key, value in box.items()},
            })
        assigned = sorted(
            index for group in groups for index in group["stroke_indices"]
        )
        if assigned != list(range(len(sample.strokes))):
            raise AssertionError("runtime grouping is not an exact stroke cover")
        hwr_tokens = [
            str(runtime_by_record[str(row["record_id"])]["final_topk"][0])
            for row in finalized
        ]
        finalized_tokens = [str(row["finalized_top1"]) for row in finalized]
        group_by_record = {row["record_id"]: row for row in groups}
        symbols = [
            {
                "record_id": str(row["record_id"]),
                "context_index": int(row["context_index"]),
                "relation_from_previous": row.get("layout_relation_from_previous"),
                "stroke_indices": list(group_by_record[str(row["record_id"])]["stroke_indices"]),
                "hwr_top1": str(runtime_by_record[str(row["record_id"])]["final_topk"][0]),
                "hwr_topk": list(runtime_by_record[str(row["record_id"])]["final_topk"]),
                "hwr_topk_probabilities": list(
                    runtime_by_record[str(row["record_id"])]["final_topk_probabilities"]
                ),
                "finalized_top1": str(row["finalized_top1"]),
                "changed": bool(row["changed"]),
                "context_decision_status": row.get("decision_status"),
                "decision_source": row.get("decision_source"),
            }
            for row in finalized
        ]
        layout_output = layout_audit = None
        if self.emit_formula_layout_shadow:
            layout_output, layout_audit = _formula_layout_shadow(runtime_rows, finalized)
        return {
            "schema": OUTPUT_SCHEMA,
            "status": "development_only_posthoc_shadow",
            "formula_id": sample.sample_id,
            "groups": groups,
            "symbols": symbols,
            "hwr_top1_tokens": hwr_tokens,
            "partition_context_tokens": list(selected["tokens"]),
            "finalized_tokens": finalized_tokens,
            "formula_text": " ".join(finalized_tokens),
            **({
                "formula_latex_shadow": layout_output["latex"],
                "formula_layout_shadow": layout_output,
            } if layout_output is not None else {}),
            "audit": {
                "candidate_groups": len(sample.candidates),
                "candidate_partitions": len(partitions),
                "geometry_baseline_rank": int(baseline["rank"]),
                "ranker_proposal_rank": int(proposal["rank"]),
                "selected_partition_rank": int(selected["rank"]),
                "partition_change_accepted": bool(selected["rank"] != baseline["rank"]),
                "design_gate_partition_change_accepted": bool(accepted),
                "partition_change_source": (
                    "repeat_merge_rescue" if repeat_merge_changed else
                    "relation_merge_rescue" if relation_merge_changed else
                    "cross_merge_rescue" if cross_merge_changed else
                    "posthoc_design_guard" if accepted else "geometry_baseline"
                ),
                "ranker_probability_gain": float(
                    proposal["ranker_probability"] - baseline["ranker_probability"]
                ),
                "selected_ranker_probability_gain": float(
                    selected["ranker_probability"] - baseline["ranker_probability"]
                ),
                "all_strokes_exactly_once": True,
                "target_label_or_glyph_count_input": False,
                "inserted_or_deleted_glyphs": 0,
                "arithmetic_evaluation": False,
                "partition_ranker_sha256": self.partition_ranker_sha256,
                "hwr_checkpoint_sha256": self.finalizer.hwr_checkpoint_sha256,
                "context_checkpoint_sha256": self.finalizer.checkpoint_sha256,
                "candidate_context_fusion": {
                    **singleton_audit,
                    "relation_merge_rescue": relation_merge_audit,
                    "cross_merge_rescue": cross_merge_audit,
                    "repeat_merge_rescue": repeat_merge_audit,
                    "enabled": self.singleton_hwr is not None,
                    "configuration_sha256": self.singleton_config_sha256,
                    "auxiliary_hwr_checkpoint_sha256": self.singleton_hwr_sha256,
                    "current_96_formula_training_overlap": self.singleton_hwr is not None,
                    "product_default_enabled": False,
                },
                "pairwise_shape_rescue": {
                    **pairwise_shape_audit,
                    "enabled": self.pairwise_shape_expert is not None,
                    "expert_sha256": self.pairwise_shape_expert_sha256,
                    "configuration_sha256": self.pairwise_shape_config_sha256,
                    "product_default_enabled": False,
                },
                "product_default_enabled": False,
                "posthoc_test_tuning": True,
                "candidate_partitions_detail": [
                    {
                        "rank": int(partition["rank"]),
                        "geometry_delta": float(partition["geometry_delta"]),
                        "ranker_probability": float(partition["ranker_probability"]),
                        "groups": [sorted(int(value) for value in group) for group in partition["groups"]],
                        "hwr_top1_tokens": [str(row["final_topk"][0]) for row in partition["rows"]],
                        "context_tokens": list(partition["tokens"]),
                    }
                    for partition in partitions
                ],
                "finalizer": finalizer_audit,
                **({"formula_layout_shadow": layout_audit} if layout_audit is not None else {}),
            },
        }


def _self_test() -> None:
    source = {
        "formula_id": "self-test",
        "strokes": [{
            "order": 0,
            "points": [
                {"x": 0.0, "y": 0.0, "t_ms": 0.0},
                {"x": 1.0, "y": 1.0, "t_ms": 20.0},
            ],
        }],
    }
    formula_id, strokes = _validate_strokes(source)
    assert formula_id == "self-test" and len(strokes) == 1
    try:
        _validate_strokes({**source, "strokes": [{**source["strokes"][0], "order": 1}]})
    except ValueError:
        pass
    else:
        raise AssertionError("invalid stroke order was accepted")
    configuration = {
        "auxiliary_policy": "old_new_product_probability_fusion",
        "auxiliary_weight": 0.6,
        "token_confidence_thresholds": {"/": 0.50, r"\times": 0.45},
    }
    runtime_rows = [{
        "record_id": "r", "formula_id": "f", "final_topk": ["1", "/"],
        "final_topk_probabilities": [0.7, 0.2],
        "context": {"index": 0, "length": 1}, "geometry": {},
    }]
    finalized = [{
        **runtime_rows[0], "context_index": 0, "finalized_top1": "1",
        "changed": False,
    }]
    auxiliary_rows = [{
        "record_id": "r", "formula_id": "f", "final_topk": ["/", "1"],
        "final_topk_probabilities": [0.51, 0.49],
        "hwr_policy": configuration["auxiliary_policy"],
        "hwr_fusion_weight": configuration["auxiliary_weight"],
    }]
    rescued, audit = _apply_singleton_finalized_rows(
        runtime_rows, finalized, auxiliary_rows, configuration,
    )
    assert rescued[0]["finalized_top1"] == "/"
    assert rescued[0]["decision_source"] == "singleton_shape_rescue"
    assert audit["changed"] == 1
    layout_rows = [
        {
            "record_id": "x", "formula_id": "layout", "final_topk": ["x"],
            "final_topk_probabilities": [1.0],
            "geometry": {"left": 0, "top": 10, "right": 20, "bottom": 40},
        },
        {
            "record_id": "2", "formula_id": "layout", "final_topk": ["2"],
            "final_topk_probabilities": [1.0],
            "geometry": {"left": 21, "top": 0, "right": 29, "bottom": 14},
        },
    ]
    layout, layout_audit = _formula_layout_shadow(
        layout_rows,
        [
            {"record_id": "x", "finalized_top1": "x"},
            {"record_id": "2", "finalized_top1": "2"},
        ],
    )
    assert layout["latex"] == "x^{2}"
    assert layout_audit["candidate_extensions"] == 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition-ranker", type=Path)
    parser.add_argument("--hwr-checkpoint", type=Path, default=DEFAULT_PRODUCT)
    parser.add_argument("--context-checkpoint", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--formula-sequence-config", type=Path)
    parser.add_argument("--formula-syntax-rescue-config", type=Path)
    parser.add_argument("--candidate-context-fusion-config", type=Path)
    parser.add_argument("--candidate-context-auxiliary-hwr-checkpoint", type=Path)
    parser.add_argument("--formula-placement-config", type=Path)
    parser.add_argument("--straight-equality-config", type=Path)
    parser.add_argument("--latin-auxiliary-checkpoint", type=Path)
    parser.add_argument("--pairwise-shape-expert", type=Path)
    parser.add_argument("--pairwise-shape-config", type=Path)
    parser.add_argument("--formula-acceptance-config", type=Path)
    parser.add_argument("--emit-formula-layout-shadow", action="store_true")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--allow-posthoc-shadow", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        print(json.dumps({"self_test": "pass"}))
        return 0
    if args.partition_ranker is None or args.input is None or args.output is None:
        parser.error("--partition-ranker, --input, and --output are required")
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not input_path.is_file():
        parser.error(f"raw formula input is missing: {input_path}")
    if output_path.exists():
        parser.error(f"refusing to overwrite raw formula output: {output_path}")
    if not args.allow_posthoc_shadow:
        parser.error("--allow-posthoc-shadow is required for this development-only gate")
    runtime = RawFormulaContextRuntimeV1.from_artifacts(
        args.partition_ranker, args.hwr_checkpoint, args.context_checkpoint,
        device=args.device, batch_size=args.batch_size,
        formula_sequence_config=args.formula_sequence_config,
        formula_syntax_rescue_config=args.formula_syntax_rescue_config,
        candidate_context_fusion_config=args.candidate_context_fusion_config,
        candidate_context_auxiliary_hwr_checkpoint=args.candidate_context_auxiliary_hwr_checkpoint,
        formula_placement_config=args.formula_placement_config,
        straight_equality_config=args.straight_equality_config,
        latin_auxiliary_checkpoint=args.latin_auxiliary_checkpoint,
        pairwise_shape_expert=args.pairwise_shape_expert,
        pairwise_shape_config=args.pairwise_shape_config,
        emit_formula_layout_shadow=args.emit_formula_layout_shadow,
        allow_posthoc_shadow=args.allow_posthoc_shadow,
    )
    results = [runtime.infer(row) for row in _load_inputs(input_path)]
    output = {
        "schema": SCHEMA,
        "status": "development_only_posthoc_shadow",
        "formulas": results,
        "audit": {
            "formulas": len(results),
            "product_default_enabled": False,
            "posthoc_test_tuning": True,
        },
    }
    if args.emit_formula_layout_shadow:
        output["audit"]["formula_layout_shadow_emitted"] = len(results)
    if args.formula_acceptance_config is not None:
        acceptance_configuration, acceptance_configuration_sha256 = (
            load_acceptance_configuration(args.formula_acceptance_config)
        )
        output = apply_formula_acceptance_guard(
            output, acceptance_configuration, acceptance_configuration_sha256,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    print(json.dumps({
        "output": str(output_path), "sha256": _sha256(output_path),
        "formulas": len(results), "status": output["status"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
