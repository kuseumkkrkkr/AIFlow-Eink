#!/usr/bin/env python3
"""Chronological Top-5 partition reranking with frozen HWR and context models."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluate_48hz_prefix_v1 import (
    DEFAULT_BASE, DEFAULT_PRODUCT, _load_loo_heads, _load_model, _sha256,
)
from evaluate_joint_hwr_grouping_v1 import _candidate_embeddings, _probabilities
from dual_hwr_numeric_rescue_v1 import (
    apply_dual_hwr_numeric_rescue,
    validate_configuration as validate_dual_numeric_configuration,
)
from wide_candidate_syntax_rescue_v1 import (
    apply_wide_candidate_syntax_rescue,
    validate_configuration as validate_wide_syntax_configuration,
)
from finalize_formula_context_v1 import OwnedFormulaContextFinalizer
from formula_placement_rescue_v1 import (
    CONFIG_SCHEMA as PLACEMENT_CONFIG_SCHEMA, SCHEMA as PLACEMENT_SCHEMA,
    apply_formula_placement_rescue,
)
from singleton_shape_rescue_v1 import (
    SCHEMA as SINGLETON_SCHEMA, apply_singleton_shape_rescue,
    validate_configuration as validate_singleton_configuration,
)
from straight_equality_slot_rescue_v1 import (
    CONFIG_SCHEMA as EQUALITY_CONFIG_SCHEMA, SCHEMA as EQUALITY_SCHEMA,
    apply_straight_equality_slot_rescue,
)
from latin_t_context_rescue_v1 import (
    DEFAULT_CONFIGURATION as PRODUCT_LATIN_T_CONFIGURATION,
    apply_latin_t_context_rescue, build_latin_auxiliary_rows,
    load_latin_auxiliary_model,
)
from stroke_grouping_v1 import (
    FEATURE_NAMES as GROUPING_FEATURE_NAMES, enumerate_partitions,
    group_shape_features, normalized_group_dtw_distance,
)
from train_context_decision_layer_v1 import ROLE_TO_INDEX, _semantic_role
from train_owned_formula_context_v1 import (
    _components, decide_owned_formula_rows_supported_exact, load_owned_formula_context,
)
from train_project_owned_grouping_v1 import Sample, _fit, _samples
from character_tensor_v1 import _digest


SCHEMA = "aiflow-partition-context-ranker/v1"
MODEL_VERSION = "aiflow-partition-context-ranker-1.0-r2-shadow"
TOP_N = 5
LATTICE_CONFIG = {
    "temporal_window": 6,
    "spatial_neighbors": 4,
    "max_long_group_width_fraction": 1.0,
}
FEATURE_NAMES = (
    "geometry_delta", "geometry_per_stroke", "group_count", "group_stroke_ratio",
    "hwr_mean_log_top1", "hwr_min_log_top1", "hwr_mean_top5_mass", "hwr_mean_entropy",
    "context_role_best_mean", "context_role_margin_min", "context_exact_best_mean",
    "context_selected_exact_mean", "context_selected_shape_mean", "grammar_selected_mean",
    "syntax_violations", "fence_imbalance",
)
HWR_ONLY_FEATURE_COUNT = 8
GATE_THRESHOLDS = (0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)
MERGE_GAIN_THRESHOLDS = (-1.0, -0.10, 0.0, 0.05, 0.10, 0.20)
MERGE_GAP_THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 10.0)
POSTHOC_DESIGN_GUARD = {
    "ranker_probability_gain": 0.05,
    "minimum_merge_hwr_gain": 0.10,
    "maximum_merge_pair_gap_ref": 0.10,
    "coarsening_only": True,
}
PRODUCT_SINGLETON_CONFIGURATION = {
    "auxiliary_policy": "old_new_product_probability_fusion",
    "auxiliary_weight": 0.6,
    "token_confidence_thresholds": {"/": 0.50, r"\times": 0.45},
}
PRODUCT_DUAL_NUMERIC_CONFIGURATION = {
    "auxiliary_policy": "old_new_product_probability_fusion",
    "auxiliary_weight": 0.6,
    "maximum_operand_changes": 2,
}
PRODUCT_WIDE_SYNTAX_CONFIGURATION = {
    "candidate_width": 10,
    "auxiliary_weight": 0.6,
    "allowed_auxiliary_policies": ["old_new_product_probability_fusion"],
    "minimum_baseline_digit_count": 2,
    "maximum_changes": 2,
    "maximum_formula_glyphs": 64,
    "minimum_digit_probability_ratio": 0.005,
    "equality_candidate_must_be_top1": True,
    "allow_equality_replacement": False,
    "literal_alpha_operand_lock": True,
}
RELATION_MERGE_CONFIGURATION = {
    "maximum_candidate_rank": 3,
    "minimum_geometry_delta": -5.0,
    "minimum_ranker_probability": 0.05,
    "minimum_component_box_overlap": 0.25,
    "required_merged_strokes": 3,
}
CROSS_MERGE_CONFIGURATION = {
    "maximum_candidate_rank": 3,
    "minimum_geometry_delta": -12.0,
    "maximum_pair_gap_ref": 0.30,
    "minimum_y_overlap": 0.70,
    "maximum_absolute_aspect_log": 0.40,
    "required_merged_strokes": 2,
    "preserve_merged_x_after_auxiliary_fusion": True,
    "auxiliary_nested_expression_enabled": True,
    "auxiliary_x_candidate_maximum_rank": 3,
    "auxiliary_open_fence_candidate_maximum_rank": 1,
}
REPEAT_MERGE_CONFIGURATION = {
    "maximum_candidate_rank": 3,
    "minimum_geometry_delta": -6.0,
    "minimum_ranker_probability": 0.80,
    "maximum_pair_gap_ref": 0.20,
    "minimum_y_overlap": 0.95,
    "maximum_normalized_dtw_distance": 0.10,
    "required_merged_strokes": 2,
    "required_formula_glyphs": 3,
    "relation_token": "=",
    "preserve_merged_token_after_auxiliary_fusion": True,
}
NEGATED_RELATIONS = frozenset({
    r"\neq", r"\notin", r"\nsubset", r"\nsubseteq",
    r"\nsupset", r"\nsupseteq", r"\nleq", r"\ngeq",
})
BASE_RELATION_TOKENS = frozenset({
    "=", "<", ">", r"\leq", r"\geq", r"\approx", r"\simeq", r"\equiv",
})
SLASH_COMPONENT_TOKENS = frozenset({
    "1", "I", "l", "|", "/", r"\backslash", r"\mid", r"\prime",
})


def _validate_relation_merge_configuration(configuration: dict) -> dict:
    expected = {
        "maximum_candidate_rank", "minimum_geometry_delta",
        "minimum_ranker_probability", "minimum_component_box_overlap",
        "required_merged_strokes",
    }
    if set(configuration) != expected:
        raise ValueError("relation merge configuration fields mismatch")
    maximum_rank = configuration["maximum_candidate_rank"]
    required_strokes = configuration["required_merged_strokes"]
    if (
        isinstance(maximum_rank, bool)
        or not isinstance(maximum_rank, int)
        or not 1 <= maximum_rank <= TOP_N
        or isinstance(required_strokes, bool)
        or not isinstance(required_strokes, int)
        or required_strokes < 2
    ):
        raise ValueError("invalid relation merge integer configuration")
    geometry_delta = float(configuration["minimum_geometry_delta"])
    ranker_probability = float(configuration["minimum_ranker_probability"])
    box_overlap = float(configuration["minimum_component_box_overlap"])
    if (
        not all(math.isfinite(value) for value in (
            geometry_delta, ranker_probability, box_overlap,
        ))
        or geometry_delta > 0.0
        or not 0.0 <= ranker_probability <= 1.0
        or not 0.0 <= box_overlap <= 1.0
    ):
        raise ValueError("invalid relation merge threshold configuration")
    return {
        "maximum_candidate_rank": maximum_rank,
        "minimum_geometry_delta": geometry_delta,
        "minimum_ranker_probability": ranker_probability,
        "minimum_component_box_overlap": box_overlap,
        "required_merged_strokes": required_strokes,
    }


def _validate_cross_merge_configuration(configuration: dict) -> dict:
    legacy = {
        "maximum_candidate_rank", "minimum_geometry_delta",
        "maximum_pair_gap_ref", "minimum_y_overlap",
        "maximum_absolute_aspect_log", "required_merged_strokes",
        "preserve_merged_x_after_auxiliary_fusion",
    }
    auxiliary_fields = {
        "auxiliary_nested_expression_enabled",
        "auxiliary_x_candidate_maximum_rank",
        "auxiliary_open_fence_candidate_maximum_rank",
    }
    fields = set(configuration)
    if fields not in (legacy, legacy | auxiliary_fields):
        raise ValueError("cross merge configuration fields mismatch")
    auxiliary_enabled = configuration.get(
        "auxiliary_nested_expression_enabled", False,
    )
    auxiliary_x_rank = configuration.get(
        "auxiliary_x_candidate_maximum_rank", 0,
    )
    auxiliary_fence_rank = configuration.get(
        "auxiliary_open_fence_candidate_maximum_rank", 0,
    )
    maximum_rank = configuration["maximum_candidate_rank"]
    required_strokes = configuration["required_merged_strokes"]
    if (
        isinstance(maximum_rank, bool)
        or not isinstance(maximum_rank, int)
        or not 1 <= maximum_rank <= TOP_N
        or isinstance(required_strokes, bool)
        or required_strokes != 2
        or type(auxiliary_enabled) is not bool
        or isinstance(auxiliary_x_rank, bool)
        or not isinstance(auxiliary_x_rank, int)
        or isinstance(auxiliary_fence_rank, bool)
        or not isinstance(auxiliary_fence_rank, int)
        or (auxiliary_enabled and not 1 <= auxiliary_x_rank <= 20)
        or (auxiliary_enabled and not 1 <= auxiliary_fence_rank <= 20)
        or (not auxiliary_enabled and (auxiliary_x_rank or auxiliary_fence_rank))
    ):
        raise ValueError("invalid cross merge integer configuration")
    preserve_token = configuration["preserve_merged_x_after_auxiliary_fusion"]
    if type(preserve_token) is not bool:
        raise ValueError("invalid cross merge token preservation configuration")
    values = {
        key: float(configuration[key]) for key in (
            "minimum_geometry_delta", "maximum_pair_gap_ref",
            "minimum_y_overlap", "maximum_absolute_aspect_log",
        )
    }
    if (
        not all(math.isfinite(value) for value in values.values())
        or values["minimum_geometry_delta"] > 0.0
        or values["maximum_pair_gap_ref"] < 0.0
        or not 0.0 <= values["minimum_y_overlap"] <= 1.0
        or values["maximum_absolute_aspect_log"] < 0.0
    ):
        raise ValueError("invalid cross merge threshold configuration")
    return {
        "maximum_candidate_rank": maximum_rank,
        **values,
        "required_merged_strokes": required_strokes,
        "preserve_merged_x_after_auxiliary_fusion": preserve_token,
        "auxiliary_nested_expression_enabled": auxiliary_enabled,
        "auxiliary_x_candidate_maximum_rank": auxiliary_x_rank,
        "auxiliary_open_fence_candidate_maximum_rank": auxiliary_fence_rank,
    }


def _validate_repeat_merge_configuration(configuration: dict) -> dict:
    expected = {
        "maximum_candidate_rank", "minimum_geometry_delta",
        "minimum_ranker_probability", "maximum_pair_gap_ref",
        "minimum_y_overlap", "maximum_normalized_dtw_distance",
        "required_merged_strokes", "required_formula_glyphs",
        "relation_token", "preserve_merged_token_after_auxiliary_fusion",
    }
    if set(configuration) != expected:
        raise ValueError("repeat merge configuration fields mismatch")
    maximum_rank = configuration["maximum_candidate_rank"]
    required_strokes = configuration["required_merged_strokes"]
    required_glyphs = configuration["required_formula_glyphs"]
    if (
        isinstance(maximum_rank, bool) or not isinstance(maximum_rank, int)
        or not 1 <= maximum_rank <= TOP_N
        or isinstance(required_strokes, bool) or required_strokes != 2
        or isinstance(required_glyphs, bool) or required_glyphs != 3
        or configuration["relation_token"] != "="
        or type(configuration["preserve_merged_token_after_auxiliary_fusion"])
        is not bool
    ):
        raise ValueError("invalid repeat merge discrete configuration")
    values = {
        key: float(configuration[key]) for key in (
            "minimum_geometry_delta", "minimum_ranker_probability",
            "maximum_pair_gap_ref", "minimum_y_overlap",
            "maximum_normalized_dtw_distance",
        )
    }
    if (
        not all(math.isfinite(value) for value in values.values())
        or values["minimum_geometry_delta"] > 0.0
        or not 0.0 <= values["minimum_ranker_probability"] <= 1.0
        or values["maximum_pair_gap_ref"] < 0.0
        or not 0.0 <= values["minimum_y_overlap"] <= 1.0
        or values["maximum_normalized_dtw_distance"] < 0.0
    ):
        raise ValueError("invalid repeat merge threshold configuration")
    return {
        "maximum_candidate_rank": maximum_rank,
        **values,
        "required_merged_strokes": required_strokes,
        "required_formula_glyphs": required_glyphs,
        "relation_token": "=",
        "preserve_merged_token_after_auxiliary_fusion": bool(
            configuration["preserve_merged_token_after_auxiliary_fusion"]
        ),
    }


def _partition_rows(
    sample: Sample, rank: int, groups: tuple[frozenset[int], ...],
    probability: np.ndarray, labels: list[str], candidate_index: dict[frozenset[int], int],
) -> list[dict]:
    candidates = {frozenset(row["source_indices"]): row for row in sample.candidates}
    formula_left = min(row["box"]["left"] for row in sample.candidates if len(row["source_indices"]) == 1)
    formula_top = min(row["box"]["top"] for row in sample.candidates if len(row["source_indices"]) == 1)
    formula_right = max(row["box"]["right"] for row in sample.candidates if len(row["source_indices"]) == 1)
    formula_bottom = max(row["box"]["bottom"] for row in sample.candidates if len(row["source_indices"]) == 1)
    formula_width = max(formula_right - formula_left, 1e-6); formula_height = max(formula_bottom - formula_top, 1e-6)
    ordered = sorted(groups, key=lambda group: (candidates[group]["box"]["left"], candidates[group]["box"]["top"], min(group)))
    rows = []
    for index, group in enumerate(ordered):
        candidate = candidates[group]; box = candidate["box"]; probs = probability[candidate_index[group]]
        token_indices = np.argsort(-probs)[:5]
        width = float(box["right"] - box["left"]); height = float(box["bottom"] - box["top"])
        rows.append({
            "record_id": f"{sample.sample_id}::p{rank}::g{index}",
            "formula_id": f"{sample.sample_id}::p{rank}",
            "final_topk": [labels[int(value)] for value in token_indices],
            "final_topk_probabilities": [float(probs[int(value)]) for value in token_indices],
            "geometry": {
                "left": float(box["left"]), "top": float(box["top"]),
                "right": float(box["right"]), "bottom": float(box["bottom"]),
                "width": width, "height": height,
                "center_x": ((float(box["left"]) + float(box["right"])) / 2.0 - formula_left) / formula_width,
                "center_y": ((float(box["top"]) + float(box["bottom"])) / 2.0 - formula_top) / formula_height,
                "width_rel": width / formula_width, "height_rel": height / formula_height,
                "stroke_count": float(len(group)),
            },
            "context": {"index": index, "length": len(ordered)},
            "group": sorted(group),
            "full_probability": probs,
            "grouping_features": {
                name: float(sample.features[candidate_index[group], feature_index])
                for feature_index, name in enumerate(GROUPING_FEATURE_NAMES)
            },
        })
    return rows


def _enumerate(
    samples: list[Sample], grouping_models: dict[str, object], probability: np.ndarray,
    slices: dict[str, slice], labels: list[str], *, writer_models: bool,
) -> list[dict]:
    output = []
    for sample in samples:
        model = grouping_models[sample.writer] if writer_models else grouping_models["all"]
        group_probability = model.predict_proba(sample.features)[:, 1]
        logits = np.log(np.clip(group_probability, 1e-6, 1 - 1e-6) / np.clip(1 - group_probability, 1e-6, 1))
        ranked = enumerate_partitions(sample.candidates, logits, len(sample.strokes), top_n=TOP_N)
        local_probability = probability[slices[sample.sample_id]]
        candidate_index = {frozenset(row["source_indices"]): index for index, row in enumerate(sample.candidates)}
        best_geometry = float(ranked[0][0])
        for rank, (score, groups) in enumerate(ranked, 1):
            output.append({
                "sample": sample, "rank": rank, "geometry_score": float(score),
                "geometry_delta": float(score) - best_geometry, "groups": groups,
                "rows": _partition_rows(sample, rank, groups, local_probability, labels, candidate_index),
                "truth": set(groups) == set(sample.truth),
            })
    return output


def _syntax(tokens: list[str]) -> tuple[int, int]:
    roles = [_semantic_role(token) for token in tokens]
    operators = {"binary", "relation", "prefix"}
    violations = int(bool(roles) and roles[0] in {"binary", "relation"})
    violations += int(bool(roles) and roles[-1] in operators)
    violations += sum(left in operators and right in operators for left, right in zip(roles, roles[1:]))
    left = sum(token in {"(", "[", "{"} for token in tokens)
    right = sum(token in {")", "]", "}"} for token in tokens)
    return violations, abs(left - right)


def _attach_features(partitions: list[dict], model, contract: dict, payload: dict, device: torch.device) -> None:
    rows = [{key: value for key, value in row.items() if key not in {"group", "full_probability", "grouping_features"}}
            for partition in partitions for row in partition["rows"]]
    components = _components(model, contract, rows, payload["role_grammar"], device, 256)
    predictions, _ = decide_owned_formula_rows_supported_exact(model, contract, payload, rows, device, 256)
    packed_index = {str(row["record_id"]): index for index, row in enumerate(components["rows"])}
    configuration = payload["configuration"]
    role_scores = (
        float(configuration["context_weight"]) * components["context"]
        + float(configuration["shape_weight"]) * components["shape"]
        + float(configuration["grammar_weight"]) * components["grammar"]
    )
    for partition in partitions:
        role_best = []; role_margin = []; exact_best = []; selected_exact = []
        selected_shape = []; selected_grammar = []; top1_logs = []; top5_mass = []; entropies = []; tokens = []
        for source in partition["rows"]:
            record_id = str(source["record_id"]); index = packed_index[record_id]
            allowed_roles = np.flatnonzero(components["role_mask"][index])
            values = np.sort(role_scores[index, allowed_roles])
            role_best.append(float(values[-1])); role_margin.append(float(values[-1] - values[-2]) if len(values) > 1 else 0.0)
            allowed_candidates = np.flatnonzero(components["candidate_mask"][index])
            exact = (
                float(configuration["exact_context_scale"]) * float(configuration["context_weight"])
                * components["candidate_context"][index, allowed_candidates]
                + float(configuration["shape_weight"]) * components["candidate_shape"][index, allowed_candidates]
            )
            exact_best.append(float(exact.max()))
            token = str(predictions[record_id]); tokens.append(token)
            token_position = source["final_topk"].index(token)
            selected_exact.append(float(components["candidate_context"][index, token_position]))
            selected_shape.append(float(components["candidate_shape"][index, token_position]))
            selected_role = ROLE_TO_INDEX[_semantic_role(token)]
            selected_grammar.append(float(components["grammar"][index, selected_role]))
            probability = np.asarray(source["full_probability"], dtype=np.float64)
            top1_logs.append(math.log(max(float(probability.max()), 1e-12)))
            top5_mass.append(float(np.sort(probability)[-5:].sum()))
            entropies.append(float(-(probability * np.log(np.clip(probability, 1e-12, 1.0))).sum() / math.log(len(probability))))
        violations, imbalance = _syntax(tokens)
        group_count = len(partition["groups"]); stroke_count = len(partition["sample"].strokes)
        partition["tokens"] = tokens
        partition["features"] = np.asarray([
            partition["geometry_delta"], partition["geometry_score"] / max(stroke_count, 1),
            group_count, group_count / max(stroke_count, 1),
            float(np.mean(top1_logs)), float(np.min(top1_logs)), float(np.mean(top5_mass)), float(np.mean(entropies)),
            float(np.mean(role_best)), float(np.min(role_margin)), float(np.mean(exact_best)),
            float(np.mean(selected_exact)), float(np.mean(selected_shape)), float(np.mean(selected_grammar)),
            float(violations), float(imbalance),
        ], dtype=np.float32)
        if partition["features"].shape != (len(FEATURE_NAMES),) or not np.isfinite(partition["features"]).all():
            raise AssertionError("invalid partition ranker features")


def _fit_ranker(partitions: list[dict], width: int):
    usable_formulae = {
        partition["sample"].sample_id for partition in partitions if partition["truth"]
    }
    rows = [partition for partition in partitions if partition["sample"].sample_id in usable_formulae]
    features = np.stack([partition["features"][:width] for partition in rows])
    truth = np.asarray([partition["truth"] for partition in rows], dtype=np.int64)
    if len(set(truth.tolist())) != 2:
        raise ValueError("partition ranker requires positive and negative rows")
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=20260819),
    ).fit(features, truth)
    return model, {"formulas": len(usable_formulae), "rows": len(rows), "positives": int(truth.sum())}


def _select(partitions: list[dict], model, width: int) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = {}
    for partition in partitions:
        grouped.setdefault(partition["sample"].sample_id, []).append(partition)
    selected = {}
    for sample_id, rows in grouped.items():
        probability = model.predict_proba(np.stack([row["features"][:width] for row in rows]))[:, 1]
        for row, value in zip(rows, probability, strict=True):
            row["ranker_probability"] = float(value)
        selected[sample_id] = max(rows, key=lambda row: (row["ranker_probability"], -row["rank"]))
    return selected


def _merge_evidence(baseline: dict, candidate: dict) -> tuple[bool, float, float]:
    baseline_groups = [frozenset(group) for group in baseline["groups"]]
    candidate_groups = [frozenset(group) for group in candidate["groups"]]
    coarsening = all(any(group.issubset(merged) for merged in candidate_groups) for group in baseline_groups)
    if not coarsening or len(candidate_groups) >= len(baseline_groups):
        return False, -math.inf, math.inf
    baseline_rows = {frozenset(row["group"]): row for row in baseline["rows"]}
    candidate_rows = {frozenset(row["group"]): row for row in candidate["rows"]}
    gains = []; gaps = []
    for merged in candidate_groups:
        components = [group for group in baseline_groups if group.issubset(merged)]
        if len(components) <= 1:
            continue
        merged_probability = float(np.max(candidate_rows[merged]["full_probability"]))
        component_probability = float(np.mean([
            np.max(baseline_rows[group]["full_probability"]) for group in components
        ]))
        gains.append(merged_probability - component_probability)
        gaps.append(float(candidate_rows[merged]["grouping_features"]["pair_gap_max_ref"]))
    return bool(gains), min(gains, default=-math.inf), max(gaps, default=math.inf)


def _gate_accept(baseline: dict, candidate: dict, configuration: dict) -> bool:
    gain = float(candidate["ranker_probability"] - baseline["ranker_probability"])
    coarsening, merge_gain, merge_gap = _merge_evidence(baseline, candidate)
    return (
        candidate["rank"] != 1
        and gain >= float(configuration["ranker_probability_gain"])
        and coarsening
        and merge_gain >= float(configuration["minimum_merge_hwr_gain"])
        and merge_gap <= float(configuration["maximum_merge_pair_gap_ref"])
    )


def _gated_selection(partitions: list[dict], model, width: int, configuration: dict) -> dict[str, dict]:
    ranker = _select(partitions, model, width)
    geometry = _geometry_selection(partitions)
    output = {}
    for sample_id, candidate in ranker.items():
        baseline = geometry[sample_id]
        output[sample_id] = candidate if _gate_accept(baseline, candidate, configuration) else baseline
    return output


def _box_overlap_ratio(left: dict, right: dict) -> float:
    width = max(0.0, min(float(left["right"]), float(right["right"])) - max(
        float(left["left"]), float(right["left"]),
    ))
    height = max(0.0, min(float(left["bottom"]), float(right["bottom"])) - max(
        float(left["top"]), float(right["top"]),
    ))
    left_area = max(
        (float(left["right"]) - float(left["left"]))
        * (float(left["bottom"]) - float(left["top"])),
        1e-12,
    )
    right_area = max(
        (float(right["right"]) - float(right["left"]))
        * (float(right["bottom"]) - float(right["top"])),
        1e-12,
    )
    return width * height / min(left_area, right_area)


def _relation_merge_selection(
    partitions: list[dict], selected: dict[str, dict], configuration: dict,
) -> tuple[dict[str, dict], dict]:
    configuration = _validate_relation_merge_configuration(configuration)
    grouped: dict[str, list[dict]] = {}
    for partition in partitions:
        grouped.setdefault(partition["sample"].sample_id, []).append(partition)
    output = dict(selected)
    changes = []
    ambiguous = []
    for sample_id, baseline in selected.items():
        baseline_groups = [frozenset(group) for group in baseline["groups"]]
        baseline_rows = {
            frozenset(row["group"]): row for row in baseline["rows"]
        }
        baseline_tokens = {
            frozenset(row["group"]): str(token)
            for row, token in zip(baseline["rows"], baseline["tokens"], strict=True)
        }
        admitted = []
        for candidate in grouped[sample_id]:
            candidate_groups = [frozenset(group) for group in candidate["groups"]]
            if (
                candidate["rank"] > configuration["maximum_candidate_rank"]
                or float(candidate["geometry_delta"])
                < configuration["minimum_geometry_delta"]
                or float(candidate.get("ranker_probability", 0.0))
                < configuration["minimum_ranker_probability"]
                or len(candidate_groups) != len(baseline_groups) - 1
            ):
                continue
            components = {
                merged: [group for group in baseline_groups if group.issubset(merged)]
                for merged in candidate_groups
            }
            merged_groups = [
                (merged, groups) for merged, groups in components.items()
                if len(groups) > 1
            ]
            if (
                len(merged_groups) != 1
                or len(merged_groups[0][1]) != 2
                or any(not groups for groups in components.values())
            ):
                continue
            merged, pair = merged_groups[0]
            if len(merged) != configuration["required_merged_strokes"]:
                continue
            pair_tokens = [baseline_tokens[group] for group in pair]
            if not (
                sum(token in BASE_RELATION_TOKENS for token in pair_tokens) == 1
                and sum(token in SLASH_COMPONENT_TOKENS for token in pair_tokens) == 1
            ):
                continue
            overlap = _box_overlap_ratio(
                baseline_rows[pair[0]]["geometry"], baseline_rows[pair[1]]["geometry"],
            )
            if overlap < configuration["minimum_component_box_overlap"]:
                continue
            merged_index = next(
                index for index, row in enumerate(candidate["rows"])
                if frozenset(row["group"]) == merged
            )
            relation_positions = [
                index for index, token in enumerate(candidate["tokens"])
                if token in NEGATED_RELATIONS
            ]
            if relation_positions != [merged_index]:
                continue
            relation_index = merged_index
            if (
                relation_index == 0
                or relation_index == len(candidate["tokens"]) - 1
                or _semantic_role(candidate["tokens"][relation_index - 1])
                not in {"digit", "operand"}
                or _semantic_role(candidate["tokens"][relation_index + 1])
                not in {"digit", "operand"}
            ):
                continue
            candidate_row = candidate["rows"][merged_index]
            relation = str(candidate["tokens"][relation_index])
            if relation not in candidate_row["final_topk"]:
                continue
            admitted.append((candidate, overlap, pair_tokens, relation))
        if len(admitted) != 1:
            if len(admitted) > 1:
                ambiguous.append(sample_id)
            continue
        candidate, overlap, pair_tokens, relation = admitted[0]
        output[sample_id] = candidate
        changes.append({
            "formula_id": sample_id,
            "before_rank": int(baseline["rank"]),
            "after_rank": int(candidate["rank"]),
            "before_tokens": list(baseline["tokens"]),
            "after_tokens": list(candidate["tokens"]),
            "merged_component_tokens": pair_tokens,
            "merged_relation": relation,
            "component_box_overlap": overlap,
            "geometry_delta": float(candidate["geometry_delta"]),
            "ranker_probability": float(candidate["ranker_probability"]),
        })
    return output, {
        "enabled": True,
        "status": "development_only_posthoc_shadow",
        "configuration": configuration,
        "changed_formulas": len(changes),
        "changes": changes,
        "ambiguous_formulas": ambiguous,
        "all_strokes_exactly_once": True,
        "target_label_or_glyph_count_input": False,
        "arithmetic_evaluation": False,
    }


def _valid_nested_flat_expression(tokens: list[str]) -> bool:
    pairs = {"(": ")", "[": "]", "{": "}", r"\{": r"\}"}
    closers = set(pairs.values())
    stack: list[str] = []
    expect_value = True
    previous_role = "boundary"
    for token in tokens:
        if token in pairs:
            if not expect_value:
                return False
            stack.append(pairs[token])
            previous_role = "fence"
            continue
        if token in closers:
            if expect_value or not stack or stack.pop() != token:
                return False
            previous_role = "fence"
            continue
        role = _semantic_role(token)
        if expect_value:
            if role not in {"digit", "operand"}:
                return False
            expect_value = False
        elif role == "operator":
            expect_value = True
        elif role == "digit" and previous_role == "digit":
            pass
        else:
            return False
        previous_role = role
    return not expect_value and not stack


def _auxiliary_nested_cross_evidence(
    candidate: dict, merged_index: int, auxiliary: dict, configuration: dict,
) -> dict | None:
    if not configuration["auxiliary_nested_expression_enabled"]:
        return None
    placement = dict(auxiliary.get("placement_configuration") or {})
    role_configuration = dict(placement.get("formula_role_typo") or {})
    if not role_configuration or not role_configuration.get(
        "value_cross_plus_open_fence_enabled", False,
    ):
        return None
    sample = candidate["sample"]
    rows = _auxiliary_rows(
        [sample], {sample.sample_id: candidate}, auxiliary["probability"],
        auxiliary["slices"], auxiliary["labels"], width=20,
        policy=auxiliary["policy"], weight=auxiliary["weight"],
    )
    if len(rows) != len(candidate["rows"]):
        raise AssertionError("auxiliary cross merge row coverage mismatch")
    plus_index = merged_index + 1
    open_index = plus_index + 1
    if open_index >= len(rows):
        return None
    tokens = [str(value) for value in candidate["tokens"]]
    merged_tokens = [str(value) for value in rows[merged_index]["final_topk"]]
    plus_tokens = [str(value) for value in rows[plus_index]["final_topk"]]
    open_tokens = [str(value) for value in rows[open_index]["final_topk"]]
    if (
        merged_index != 1
        or tokens[0] != "("
        or tokens[-2:] != [")", ")"]
        or tokens[merged_index] != "x"
        or "x" not in merged_tokens[
            :configuration["auxiliary_x_candidate_maximum_rank"]
        ]
        or "+" not in plus_tokens[
            :int(role_configuration["cross_plus_candidate_maximum_rank"])
        ]
        or "(" not in open_tokens[
            :configuration["auxiliary_open_fence_candidate_maximum_rank"]
        ]
    ):
        return None
    geometry = rows[plus_index]["geometry"]
    if (
        int(round(float(geometry["stroke_count"])))
        != int(role_configuration["cross_plus_required_stroke_count"])
        or abs(float(geometry["aspect_log"]))
        > float(role_configuration["cross_plus_maximum_absolute_aspect_log"])
        or not float(role_configuration["cross_plus_minimum_path_over_diagonal"])
        <= float(geometry["path_over_diag"])
        <= float(role_configuration["cross_plus_maximum_path_over_diagonal"])
    ):
        return None
    proposed = list(tokens)
    proposed[merged_index] = "x"
    proposed[plus_index] = "+"
    proposed[open_index] = "("
    if proposed.count("(") < 2 or not _valid_nested_flat_expression(proposed):
        return None
    return {
        "policy": "product_fused_nested_expression",
        "proposed_tokens": proposed,
        "x_candidate_rank": merged_tokens.index("x") + 1,
        "plus_candidate_rank": plus_tokens.index("+") + 1,
        "open_fence_candidate_rank": open_tokens.index("(") + 1,
    }


def _cross_merge_selection(
    partitions: list[dict], selected: dict[str, dict], configuration: dict,
    auxiliary: dict | None = None,
) -> tuple[dict[str, dict], dict]:
    configuration = _validate_cross_merge_configuration(configuration)
    grouped: dict[str, list[dict]] = {}
    for partition in partitions:
        grouped.setdefault(partition["sample"].sample_id, []).append(partition)
    output = dict(selected)
    changes = []
    ambiguous = []
    for sample_id, baseline in selected.items():
        baseline_groups = [frozenset(group) for group in baseline["groups"]]
        baseline_tokens = {
            frozenset(row["group"]): str(token)
            for row, token in zip(baseline["rows"], baseline["tokens"], strict=True)
        }
        admitted = []
        for candidate in grouped[sample_id]:
            candidate_groups = [frozenset(group) for group in candidate["groups"]]
            if (
                candidate["rank"] > configuration["maximum_candidate_rank"]
                or float(candidate["geometry_delta"])
                < configuration["minimum_geometry_delta"]
                or len(candidate_groups) != len(baseline_groups) - 1
            ):
                continue
            components = {
                merged: [group for group in baseline_groups if group.issubset(merged)]
                for merged in candidate_groups
            }
            merged_groups = [
                (merged, groups) for merged, groups in components.items()
                if len(groups) > 1
            ]
            if (
                len(merged_groups) != 1
                or len(merged_groups[0][1]) != 2
                or any(not groups for groups in components.values())
            ):
                continue
            merged, pair = merged_groups[0]
            if (
                len(merged) != configuration["required_merged_strokes"]
                or any(len(group) != 1 for group in pair)
            ):
                continue
            merged_index = next(
                index for index, row in enumerate(candidate["rows"])
                if frozenset(row["group"]) == merged
            )
            candidate_row = candidate["rows"][merged_index]
            features = candidate_row["grouping_features"]
            if (
                float(features["temporal_contiguous"]) != 1.0
                or float(features["pair_gap_max_ref"])
                > configuration["maximum_pair_gap_ref"]
                or float(features["y_overlap_mean"])
                < configuration["minimum_y_overlap"]
                or abs(float(features["aspect_log"]))
                > configuration["maximum_absolute_aspect_log"]
            ):
                continue
            standard = (
                str(candidate["tokens"][merged_index]) == "x"
                and str(candidate_row["final_topk"][0]) == "x"
            )
            auxiliary_evidence = None
            if not standard and auxiliary is not None:
                auxiliary_evidence = _auxiliary_nested_cross_evidence(
                    candidate, merged_index, auxiliary, configuration,
                )
            if not standard and auxiliary_evidence is None:
                continue
            admitted.append((
                candidate, pair, features,
                "main_hwr_context_x" if standard else "auxiliary_nested_expression",
                auxiliary_evidence,
            ))
        if len(admitted) != 1:
            if len(admitted) > 1:
                ambiguous.append(sample_id)
            continue
        candidate, pair, features, admission_rule, auxiliary_evidence = admitted[0]
        merged = next(group for group, groups in (
            (
                group,
                [baseline_group for baseline_group in baseline_groups
                 if baseline_group.issubset(group)],
            )
            for group in (frozenset(value) for value in candidate["groups"])
        ) if len(groups) == 2)
        merged_row = next(
            row for row in candidate["rows"]
            if frozenset(row["group"]) == merged
        )
        candidate = {
            **candidate,
            "cross_merge_token_locks": (
                {str(merged_row["record_id"]): "x"}
                if configuration["preserve_merged_x_after_auxiliary_fusion"]
                else {}
            ),
        }
        output[sample_id] = candidate
        changes.append({
            "formula_id": sample_id,
            "before_rank": int(baseline["rank"]),
            "after_rank": int(candidate["rank"]),
            "before_tokens": list(baseline["tokens"]),
            "after_tokens": list(candidate["tokens"]),
            "merged_component_tokens": [baseline_tokens[group] for group in pair],
            "merged_token": "x",
            "merged_record_id": str(merged_row["record_id"]),
            "geometry_delta": float(candidate["geometry_delta"]),
            "ranker_probability": float(candidate["ranker_probability"]),
            "pair_gap_max_ref": float(features["pair_gap_max_ref"]),
            "y_overlap_mean": float(features["y_overlap_mean"]),
            "aspect_log": float(features["aspect_log"]),
            "admission_rule": admission_rule,
            "auxiliary_evidence": auxiliary_evidence,
        })
    return output, {
        "enabled": True,
        "status": "development_only_posthoc_shadow",
        "configuration": configuration,
        "changed_formulas": len(changes),
        "changes": changes,
        "ambiguous_formulas": ambiguous,
        "all_strokes_exactly_once": True,
        "target_label_or_glyph_count_input": False,
        "arithmetic_evaluation": False,
    }


def _repeat_merge_selection(
    partitions: list[dict], selected: dict[str, dict], configuration: dict,
) -> tuple[dict[str, dict], dict]:
    configuration = _validate_repeat_merge_configuration(configuration)
    grouped: dict[str, list[dict]] = {}
    for partition in partitions:
        grouped.setdefault(partition["sample"].sample_id, []).append(partition)
    output = dict(selected)
    changes = []
    ambiguous = []
    value_roles = {"digit", "operand"}
    for sample_id, baseline in selected.items():
        baseline_groups = [frozenset(group) for group in baseline["groups"]]
        admitted = []
        for candidate in grouped[sample_id]:
            candidate_groups = [frozenset(group) for group in candidate["groups"]]
            hwr_tokens = [str(row["final_topk"][0]) for row in candidate["rows"]]
            context_tokens = [str(token) for token in candidate["tokens"]]
            if (
                candidate["rank"] > configuration["maximum_candidate_rank"]
                or float(candidate["geometry_delta"])
                < configuration["minimum_geometry_delta"]
                or float(candidate.get("ranker_probability", 0.0))
                < configuration["minimum_ranker_probability"]
                or len(candidate_groups) != len(baseline_groups) - 1
                or len(context_tokens) != configuration["required_formula_glyphs"]
                or hwr_tokens != context_tokens
                or context_tokens[1] != configuration["relation_token"]
                or _semantic_role(context_tokens[0]) not in value_roles
                or _semantic_role(context_tokens[2]) not in value_roles
            ):
                continue
            components = {
                merged: [group for group in baseline_groups if group.issubset(merged)]
                for merged in candidate_groups
            }
            merged_groups = [
                (merged, groups) for merged, groups in components.items()
                if len(groups) > 1
            ]
            if (
                len(merged_groups) != 1
                or len(merged_groups[0][1]) != 2
                or any(not groups for groups in components.values())
            ):
                continue
            merged, pair = merged_groups[0]
            if (
                len(merged) != configuration["required_merged_strokes"]
                or any(len(group) != 1 for group in pair)
            ):
                continue
            merged_index = next(
                index for index, row in enumerate(candidate["rows"])
                if frozenset(row["group"]) == merged
            )
            if merged_index == 1:
                continue
            merged_row = candidate["rows"][merged_index]
            features = merged_row["grouping_features"]
            if (
                float(features["temporal_contiguous"]) != 1.0
                or float(features["pair_gap_max_ref"])
                > configuration["maximum_pair_gap_ref"]
                or float(features["y_overlap_mean"])
                < configuration["minimum_y_overlap"]
            ):
                continue
            token = context_tokens[merged_index]
            references = [
                frozenset(row["group"])
                for index, row in enumerate(candidate["rows"])
                if index != merged_index
                and context_tokens[index] == token
                and len(row["group"]) == configuration["required_merged_strokes"]
            ]
            if not references:
                continue
            distances = [
                (
                    normalized_group_dtw_distance(
                        candidate["sample"].strokes, sorted(merged), sorted(reference),
                    ),
                    reference,
                )
                for reference in references
            ]
            distance, reference = min(distances, key=lambda value: value[0])
            if distance > configuration["maximum_normalized_dtw_distance"]:
                continue
            admitted.append((candidate, merged, pair, reference, features, token, distance))
        if len(admitted) != 1:
            if len(admitted) > 1:
                ambiguous.append(sample_id)
            continue
        candidate, merged, pair, reference, features, token, distance = admitted[0]
        merged_row = next(
            row for row in candidate["rows"] if frozenset(row["group"]) == merged
        )
        candidate = {
            **candidate,
            "repeat_merge_token_locks": (
                {str(merged_row["record_id"]): token}
                if configuration["preserve_merged_token_after_auxiliary_fusion"]
                else {}
            ),
        }
        output[sample_id] = candidate
        changes.append({
            "formula_id": sample_id,
            "before_rank": int(baseline["rank"]),
            "after_rank": int(candidate["rank"]),
            "before_tokens": list(baseline["tokens"]),
            "after_tokens": list(candidate["tokens"]),
            "merged_component_groups": [sorted(group) for group in pair],
            "merged_group": sorted(merged),
            "reference_group": sorted(reference),
            "repeated_token": token,
            "normalized_dtw_distance": distance,
            "geometry_delta": float(candidate["geometry_delta"]),
            "ranker_probability": float(candidate["ranker_probability"]),
            "pair_gap_max_ref": float(features["pair_gap_max_ref"]),
            "y_overlap_mean": float(features["y_overlap_mean"]),
        })
    return output, {
        "enabled": True,
        "status": "development_only_posthoc_shadow",
        "configuration": configuration,
        "changed_formulas": len(changes),
        "changes": changes,
        "ambiguous_formulas": ambiguous,
        "same_formula_repeated_shape_only": True,
        "all_strokes_exactly_once": True,
        "target_label_or_glyph_count_input": False,
        "arithmetic_evaluation": False,
    }


def _apply_cross_merge_token_locks(
    rows: list[dict], selected: dict[str, dict],
) -> tuple[list[dict], dict]:
    locks = {}
    lock_rules = {}
    for partition in selected.values():
        for field, rule in (
            ("cross_merge_token_locks", "cross_merge"),
            ("repeat_merge_token_locks", "repeat_merge"),
        ):
            for record_id, token in dict(partition.get(field) or {}).items():
                record_id = str(record_id); token = str(token)
                if record_id in locks and locks[record_id] != token:
                    raise AssertionError("grouping merge token lock conflict")
                locks[record_id] = token
                lock_rules[record_id] = rule
    changes = []
    output = []
    seen = set()
    for row in rows:
        record_id = str(row["record_id"])
        token = locks.get(record_id)
        if token is None:
            output.append(row)
            continue
        seen.add(record_id)
        candidates = []
        for field in (
            "final_topk", "baseline_top5", "auxiliary_top5",
            "candidate_union", "wide_auxiliary_top10",
        ):
            candidates.extend(
                str(value) for value in row.get(field, [])
                if str(value) not in candidates
            )
        if token not in candidates:
            raise AssertionError(
                "cross merge token lock left the candidate set: "
                f"{record_id} token={token!r} candidates={candidates!r}"
            )
        before = str(row["finalized_top1"])
        changed = before != token
        output.append({
            **row,
            "finalized_top1": token,
            "changed": token != str(row.get("hwr_top1", candidates[0])),
            "decision_source": (
                f"{lock_rules[record_id]}_auxiliary_semantic_lock_v1"
                if changed else row.get("decision_source")
            ),
        })
        if changed:
            changes.append({
                "formula_id": str(row["formula_id"]),
                "record_id": record_id,
                "before": before,
                "after": token,
            })
    if seen != set(locks):
        raise AssertionError("cross merge token lock coverage mismatch")
    return output, {
        "enabled": bool(locks),
        "status": "development_only_posthoc_shadow",
        "locked_records": len(locks),
        "lock_rules": dict(sorted(Counter(lock_rules.values()).items())),
        "changed_glyphs": len(changes),
        "changes": changes,
        "candidate_preservation_rate": 1.0,
        "inserted_or_deleted_glyphs": 0,
        "grouping_mutations": 0,
        "arithmetic_evaluation": False,
    }


def _select_gate(train_partitions: list[dict], width: int) -> dict:
    writers = sorted({partition["sample"].writer for partition in train_partitions})
    held_rows = []
    for writer in writers:
        fit = [partition for partition in train_partitions if partition["sample"].writer != writer]
        held = [partition for partition in train_partitions if partition["sample"].writer == writer]
        model, _ = _fit_ranker(fit, width)
        _select(held, model, width)
        held_rows.extend(held)
    grouped: dict[str, list[dict]] = {}
    for partition in held_rows:
        grouped.setdefault(partition["sample"].sample_id, []).append(partition)
    trials = []
    for threshold in GATE_THRESHOLDS:
        for merge_threshold in MERGE_GAIN_THRESHOLDS:
            for gap_threshold in MERGE_GAP_THRESHOLDS:
                configuration = {
                    "ranker_probability_gain": threshold,
                    "minimum_merge_hwr_gain": merge_threshold,
                    "maximum_merge_pair_gap_ref": gap_threshold,
                    "coarsening_only": True,
                }
                exact = improved = regressed = changes = 0
                for rows in grouped.values():
                    baseline = next(row for row in rows if row["rank"] == 1)
                    candidate = max(rows, key=lambda row: (row["ranker_probability"], -row["rank"]))
                    selected = candidate if _gate_accept(baseline, candidate, configuration) else baseline
                    exact += int(selected["truth"]); changes += int(selected["rank"] != 1)
                    improved += int(selected["truth"] and not baseline["truth"])
                    regressed += int(baseline["truth"] and not selected["truth"])
                trials.append({
                    "configuration": configuration, "formulas": len(grouped), "exact": exact,
                    "changes": changes, "improved": improved, "regressed": regressed,
                })
    winner = max(trials, key=lambda row: (
        row["exact"], -row["regressed"], row["improved"], -row["changes"],
        row["configuration"]["minimum_merge_hwr_gain"],
        row["configuration"]["ranker_probability_gain"],
        -row["configuration"]["maximum_merge_pair_gap_ref"],
    ))
    return {"winner": winner, "trials": trials, "protocol": "leave-one-writer-out ranker gate on first 47 only"}


def _score(samples: list[Sample], selected: dict[str, dict], truth_labels: dict[str, list[str]]) -> dict:
    grouping = hwr = context = 0
    failures = []
    for sample in samples:
        partition = selected[sample.sample_id]
        grouping_hit = bool(partition["truth"])
        ordered_rows = partition["rows"]
        hwr_tokens = [str(row["final_topk"][0]) for row in ordered_rows]
        context_tokens = list(partition["tokens"])
        hwr_hit = grouping_hit and hwr_tokens == truth_labels[sample.sample_id]
        context_hit = grouping_hit and context_tokens == truth_labels[sample.sample_id]
        grouping += grouping_hit; hwr += hwr_hit; context += context_hit
        if not context_hit:
            failures.append({
                "sample_id": sample.sample_id, "selected_rank": partition["rank"],
                "grouping_exact": grouping_hit, "truth_tokens": truth_labels[sample.sample_id],
                "hwr_tokens": hwr_tokens, "context_tokens": context_tokens,
            })
    total = len(samples)
    return {
        "formulas": total, "grouping_exact_count": grouping, "grouping_exact": grouping / total,
        "raw_hwr_formula_exact_count": hwr, "raw_hwr_formula_exact": hwr / total,
        "context_formula_exact_count": context, "context_formula_exact": context / total,
        "failures": failures,
    }


def _auxiliary_rows(
    samples: list[Sample], selected: dict[str, dict], probability: np.ndarray,
    slices: dict[str, slice], labels: list[str], *, width: int, policy: str,
    weight: float, preserve_source_candidates: bool = False,
) -> list[dict]:
    output = []
    label_index = {label: index for index, label in enumerate(labels)}
    for sample in samples:
        local = probability[slices[sample.sample_id]]
        candidate_index = {
            frozenset(row["source_indices"]): index
            for index, row in enumerate(sample.candidates)
        }
        for source in selected[sample.sample_id]["rows"]:
            group = frozenset(int(value) for value in source["group"])
            values = local[candidate_index[group]]
            if preserve_source_candidates:
                source_indices = [
                    label_index[str(token)] for token in source["final_topk"]
                ]
                indices = np.asarray(sorted(
                    source_indices, key=lambda index: -float(values[index]),
                )[:width])
            else:
                indices = np.argsort(-values)[:width]
            output.append({
                "record_id": str(source["record_id"]),
                "formula_id": sample.sample_id,
                "final_topk": [labels[int(value)] for value in indices],
                "final_topk_probabilities": [float(values[int(value)]) for value in indices],
                "geometry": {
                    **source["geometry"],
                    **group_shape_features(sample.strokes, sorted(group)),
                    "component_shapes": [
                        group_shape_features(sample.strokes, [index])
                        for index in sorted(group)
                    ],
                },
                "context": dict(source["context"]),
                "hwr_policy": policy,
                "hwr_fusion_weight": float(weight),
                "hwr_candidate_width": width,
            })
    return output


def _score_finalized(
    samples: list[Sample], selected: dict[str, dict], truth_labels: dict[str, list[str]],
    finalizer: OwnedFormulaContextFinalizer, auxiliary: dict | None = None,
    singleton_auxiliary: dict | None = None,
    latin_auxiliary: dict | None = None,
) -> dict:
    runtime_rows = []
    for sample in samples:
        partition = selected[sample.sample_id]
        for source in partition["rows"]:
            runtime_rows.append({
                **{
                    key: value for key, value in source.items()
                    if key not in {"group", "full_probability", "grouping_features"}
                },
                "formula_id": sample.sample_id,
            })
    finalized, audit = finalizer.finalize(runtime_rows)
    def score(stage_rows: list[dict], candidate_rows: list[dict]) -> dict:
        by_formula: dict[str, list[dict]] = {}
        for row in stage_rows:
            by_formula.setdefault(str(row["formula_id"]), []).append(row)
        candidates = {str(row["record_id"]): row for row in candidate_rows}
        exact = oracle = 0; failures = []
        for sample in samples:
            rows = sorted(
                by_formula[sample.sample_id], key=lambda row: int(row["context_index"]),
            )
            tokens = [str(row["finalized_top1"]) for row in rows]
            grouping_hit = bool(selected[sample.sample_id]["truth"])
            truth = truth_labels[sample.sample_id]
            candidate_sequences = [
                [str(value) for value in candidates[str(row["record_id"])]["final_topk"]]
                for row in rows
            ]
            oracle_hit = grouping_hit and len(rows) == len(truth) and all(
                token in values
                for values, token in zip(candidate_sequences, truth, strict=True)
            )
            hit = grouping_hit and tokens == truth
            exact += int(hit); oracle += int(oracle_hit)
            if not hit:
                failures.append({
                    "sample_id": sample.sample_id,
                    "selected_rank": int(selected[sample.sample_id]["rank"]),
                    "grouping_exact": grouping_hit,
                    "candidate_oracle": oracle_hit,
                    "truth_tokens": truth,
                    "finalized_tokens": tokens,
                    "missing_from_candidates": [
                        {"index": index, "token": token}
                        for index, (values, token) in enumerate(zip(candidate_sequences, truth))
                        if token not in values
                    ] if grouping_hit and len(rows) == len(truth) else [],
                })
        return {"exact": exact, "oracle": oracle, "failures": failures}

    base = score(finalized, runtime_rows)
    final_rows = finalized
    singleton_audit = {"enabled": False}
    singleton_rows = None
    singleton_score = None
    singleton_improved: list[str] = []
    singleton_regressed: list[str] = []
    restricted_fusion_score = None
    restricted_fusion_audit = {"enabled": False}
    restricted_fusion_improved: list[str] = []
    restricted_fusion_regressed: list[str] = []
    combined_score = None
    combined_audit = {"enabled": False}
    combined_improved: list[str] = []
    combined_regressed: list[str] = []
    numeric_score = None
    numeric_audit = {"enabled": False}
    numeric_finalizer_audit = {"enabled": False}
    numeric_improved: list[str] = []
    numeric_regressed: list[str] = []
    wide_score = None
    wide_audit = {"enabled": False}
    wide_improved: list[str] = []
    wide_regressed: list[str] = []
    if singleton_auxiliary is not None:
        singleton_rows = _auxiliary_rows(
            samples, selected, singleton_auxiliary["probability"],
            singleton_auxiliary["slices"], singleton_auxiliary["labels"],
            width=5, policy=singleton_auxiliary["policy"],
            weight=singleton_auxiliary["weight"],
        )
        predictions = {
            str(row["record_id"]): str(row["finalized_top1"])
            for row in finalized
        }
        rescued, singleton_audit = apply_singleton_shape_rescue(
            runtime_rows, predictions, singleton_rows,
            singleton_auxiliary["configuration"],
        )
        final_rows = [
            {
                **row,
                "finalized_top1": rescued[str(row["record_id"])],
                "decision_source": (
                    "singleton_shape_rescue"
                    if rescued[str(row["record_id"])] != str(row["finalized_top1"])
                    else row.get("decision_source", "formula_context_finalizer")
                ),
            }
            for row in finalized
        ]
        singleton_score = score(final_rows, runtime_rows)
        base_failures = {str(row["sample_id"]) for row in base["failures"]}
        singleton_failures = {
            str(row["sample_id"]) for row in singleton_score["failures"]
        }
        singleton_improved = sorted(base_failures - singleton_failures)
        singleton_regressed = sorted(singleton_failures - base_failures)
        restricted_rows = _auxiliary_rows(
            samples, selected, singleton_auxiliary["probability"],
            singleton_auxiliary["slices"], singleton_auxiliary["labels"],
            width=5, policy=singleton_auxiliary["policy"],
            weight=singleton_auxiliary["weight"],
            preserve_source_candidates=True,
        )
        restricted_finalized, restricted_fusion_audit = finalizer.finalize(
            restricted_rows,
        )
        restricted_fusion_score = score(restricted_finalized, runtime_rows)
        restricted_failures = {
            str(row["sample_id"])
            for row in restricted_fusion_score["failures"]
        }
        restricted_fusion_improved = sorted(base_failures - restricted_failures)
        restricted_fusion_regressed = sorted(restricted_failures - base_failures)
        restricted_predictions = {
            str(row["record_id"]): str(row["finalized_top1"])
            for row in restricted_finalized
        }
        combined_predictions, combined_audit = apply_singleton_shape_rescue(
            runtime_rows, restricted_predictions, singleton_rows,
            singleton_auxiliary["configuration"],
        )
        combined_rows = [
            {
                **row,
                "finalized_top1": combined_predictions[str(row["record_id"])],
            }
            for row in restricted_finalized
        ]
        final_rows = combined_rows
        combined_score = score(combined_rows, runtime_rows)
        combined_failures = {
            str(row["sample_id"]) for row in combined_score["failures"]
        }
        combined_improved = sorted(base_failures - combined_failures)
        combined_regressed = sorted(combined_failures - base_failures)
        auxiliary_finalized, numeric_finalizer_audit = finalizer.finalize(
            singleton_rows,
        )
        numeric_rows, numeric_audit = apply_dual_hwr_numeric_rescue(
            combined_rows, singleton_rows, auxiliary_finalized,
            singleton_auxiliary["numeric_configuration"],
        )
        final_rows = numeric_rows
        union_candidates = [
            {
                "record_id": row["record_id"],
                "final_topk": row["candidate_union"],
            }
            for row in numeric_rows
        ]
        numeric_score = score(numeric_rows, union_candidates)
        numeric_failures = {
            str(row["sample_id"]) for row in numeric_score["failures"]
        }
        numeric_improved = sorted(base_failures - numeric_failures)
        numeric_regressed = sorted(numeric_failures - base_failures)
        wide_rows = _auxiliary_rows(
            samples, selected, singleton_auxiliary["probability"],
            singleton_auxiliary["slices"], singleton_auxiliary["labels"], width=10,
            policy=singleton_auxiliary["policy"],
            weight=singleton_auxiliary["weight"],
        )
        wide_finalized, wide_audit = apply_wide_candidate_syntax_rescue(
            numeric_rows, wide_rows, singleton_auxiliary["wide_configuration"],
        )
        final_rows = wide_finalized
        numeric_candidates = {
            str(row["record_id"]): list(row["candidate_union"])
            for row in numeric_rows
        }
        wide_union_candidates = []
        for row in wide_rows:
            candidates = list(numeric_candidates[str(row["record_id"])])
            candidates.extend(
                token for token in row["final_topk"] if token not in candidates
            )
            wide_union_candidates.append({
                "record_id": row["record_id"], "final_topk": candidates,
            })
        wide_score = score(wide_finalized, wide_union_candidates)
        wide_failures = {
            str(row["sample_id"]) for row in wide_score["failures"]
        }
        wide_improved = sorted(base_failures - wide_failures)
        wide_regressed = sorted(wide_failures - base_failures)
    placement_audit = equality_audit = {"enabled": False}
    auxiliary_rows = None
    placement_score = equality_score = None
    if auxiliary is not None:
        auxiliary_rows = _auxiliary_rows(
            samples, selected, auxiliary["probability"], auxiliary["slices"],
            auxiliary["labels"], width=20, policy=auxiliary["policy"],
            weight=auxiliary["weight"],
        )
        placement_rows, placement_audit = apply_formula_placement_rescue(
            final_rows, auxiliary_rows, auxiliary["placement_configuration"],
        )
        final_rows = placement_rows
        placement_score = score(placement_rows, auxiliary_rows)
        final_rows, equality_audit = apply_straight_equality_slot_rescue(
            final_rows, auxiliary_rows, auxiliary["equality_configuration"],
        )
        equality_score = score(final_rows, auxiliary_rows)
    latin_audit = {"enabled": False}
    latin_score = None
    latin_rows = None
    if latin_auxiliary is not None:
        latin_rows = build_latin_auxiliary_rows(
            samples, selected, latin_auxiliary["model"],
            latin_auxiliary["labels"], latin_auxiliary["device"],
        )
        final_rows, latin_audit = apply_latin_t_context_rescue(
            final_rows, latin_rows, latin_auxiliary["configuration"],
        )
    lock_rows, cross_lock_audit = _apply_cross_merge_token_locks(
        final_rows, selected,
    )
    if auxiliary_rows is not None:
        lock_candidates = auxiliary_rows
    elif wide_score is not None:
        lock_candidates = wide_union_candidates
    elif numeric_score is not None:
        lock_candidates = union_candidates
    else:
        lock_candidates = runtime_rows
    if latin_rows is not None:
        main_candidates = {
            str(row["record_id"]): [str(value) for value in row["final_topk"]]
            for row in lock_candidates
        }
        latin_candidates = {
            str(row["record_id"]): [str(value) for value in row["final_topk"]]
            for row in latin_rows
        }
        lock_candidates = []
        for record_id, tokens in main_candidates.items():
            combined = list(tokens)
            combined.extend(
                token for token in latin_candidates[record_id]
                if token not in combined
            )
            lock_candidates.append({"record_id": record_id, "final_topk": combined})
    final_rows = lock_rows
    final_score = score(final_rows, lock_candidates)
    if latin_rows is not None:
        latin_score = final_score
    total = len(samples)
    result = {
        "formula_exact_count": base["exact"],
        "formula_exact": base["exact"] / total,
        "top5_formula_oracle_count": base["oracle"],
        "top5_formula_oracle": base["oracle"] / total,
        "unresolved_inside_top5_count": base["oracle"] - base["exact"],
        "final_formula_exact_count": final_score["exact"],
        "final_formula_exact": final_score["exact"] / total,
        "failures": final_score["failures"],
        "audit": {
            "pipeline": audit["pipeline"],
            "changed": audit["changed"],
            "candidate_preservation_rate": audit["candidate_preservation_rate"],
            "new_tokens": audit["new_tokens"],
            "deleted_glyphs": audit["deleted_glyphs"],
            "grouping_mutations": audit["grouping_mutations"],
            "equation_correction_enabled": audit["equation_correction_enabled"],
            "formula_layout": audit["formula_layout"],
            "singleton_shape": singleton_audit,
            "restricted_product_fusion_finalizer": restricted_fusion_audit,
            "dual_numeric_auxiliary_finalizer": numeric_finalizer_audit,
            "dual_numeric_rescue": numeric_audit,
            "wide_numeric_syntax_rescue": wide_audit,
            "formula_placement": placement_audit,
            "straight_equality": equality_audit,
            "latin_t_context_rescue": latin_audit,
            "cross_merge_auxiliary_semantic_lock": cross_lock_audit,
        },
    }
    if singleton_rows is not None and singleton_score is not None:
        result.update({
            "singleton_formula_exact_count": singleton_score["exact"],
            "singleton_formula_exact": singleton_score["exact"] / total,
            "singleton_improved": singleton_improved,
            "singleton_regressed": singleton_regressed,
        })
    if restricted_fusion_score is not None:
        result.update({
            "restricted_product_fusion_formula_exact_count": restricted_fusion_score["exact"],
            "restricted_product_fusion_formula_exact": restricted_fusion_score["exact"] / total,
            "restricted_product_fusion_improved": restricted_fusion_improved,
            "restricted_product_fusion_regressed": restricted_fusion_regressed,
        })
    if combined_score is not None:
        result.update({
            "combined_formula_exact_count": combined_score["exact"],
            "combined_formula_exact": combined_score["exact"] / total,
            "combined_improved": combined_improved,
            "combined_regressed": combined_regressed,
            "combined_singleton_audit": combined_audit,
        })
    if numeric_score is not None:
        result.update({
            "dual_numeric_formula_exact_count": numeric_score["exact"],
            "dual_numeric_formula_exact": numeric_score["exact"] / total,
            "dual_numeric_formula_oracle_count": numeric_score["oracle"],
            "dual_numeric_formula_oracle": numeric_score["oracle"] / total,
            "dual_numeric_improved": numeric_improved,
            "dual_numeric_regressed": numeric_regressed,
        })
    if wide_score is not None:
        result.update({
            "wide_numeric_formula_exact_count": wide_score["exact"],
            "wide_numeric_formula_exact": wide_score["exact"] / total,
            "wide_numeric_formula_oracle_count": wide_score["oracle"],
            "wide_numeric_formula_oracle": wide_score["oracle"] / total,
            "wide_numeric_improved": wide_improved,
            "wide_numeric_regressed": wide_regressed,
        })
    if auxiliary_rows is not None and placement_score is not None and equality_score is not None:
        result.update({
            "top20_formula_oracle_count": placement_score["oracle"],
            "top20_formula_oracle": placement_score["oracle"] / total,
            "formula_placement_exact_count": placement_score["exact"],
            "formula_placement_exact": placement_score["exact"] / total,
            "straight_equality_exact_count": equality_score["exact"],
            "straight_equality_exact": equality_score["exact"] / total,
        })
    if latin_score is not None:
        result.update({
            "latin_t_context_exact_count": latin_score["exact"],
            "latin_t_context_exact": latin_score["exact"] / total,
        })
    return result


def _writer_loo_probabilities(
    samples: list[Sample], embeddings: torch.Tensor, slices: dict[str, slice],
    heads: dict[str, torch.nn.Linear], labels: list[str], device: torch.device,
) -> np.ndarray:
    output = np.empty((len(embeddings), len(labels)), dtype=np.float32)
    for writer in sorted({sample.writer for sample in samples}):
        writer_group = _digest(["project_owned_writer", writer])[:24]
        if writer_group not in heads:
            raise ValueError(f"held-writer candidate HWR head missing: {writer}")
        probability = _probabilities(heads[writer_group], embeddings, device)
        for sample in samples:
            if sample.writer == writer:
                output[slices[sample.sample_id]] = probability[slices[sample.sample_id]]
    if not np.isfinite(output).all():
        raise AssertionError("invalid writer-LOO candidate probabilities")
    return output


def _shadow_configuration(
    path: Path, *, config_schema: str, rescue_schema: str,
) -> tuple[dict, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    gate = payload.get("gate", {})
    if (
        payload.get("schema") != config_schema
        or payload.get("rescue_schema") != rescue_schema
        or gate.get("shadow_runtime_admitted") is not True
        or gate.get("product_default_admitted") is not False
    ):
        raise ValueError(f"auxiliary shadow configuration is not admitted: {path}")
    return payload["configuration"], payload


def _geometry_selection(partitions: list[dict]) -> dict[str, dict]:
    return {
        partition["sample"].sample_id: partition
        for partition in partitions if partition["rank"] == 1
    }


def _selection_changes(baseline: dict[str, dict], candidate: dict[str, dict]) -> list[dict]:
    output = []
    for sample_id in sorted(baseline):
        selected = candidate[sample_id]
        if selected["rank"] == baseline[sample_id]["rank"]:
            continue
        output.append({
            "sample_id": sample_id, "selected_rank": selected["rank"],
            "geometry_delta": selected["geometry_delta"],
            "ranker_probability": selected.get("ranker_probability"),
            "baseline_ranker_probability": baseline[sample_id].get("ranker_probability"),
            "ranker_probability_gain": (
                float(selected["ranker_probability"] - baseline[sample_id]["ranker_probability"])
                if "ranker_probability" in selected and "ranker_probability" in baseline[sample_id] else None
            ),
            "baseline_truth": bool(baseline[sample_id]["truth"]),
            "selected_truth": bool(selected["truth"]),
            "baseline_tokens": list(baseline[sample_id]["tokens"]),
            "selected_tokens": list(selected["tokens"]),
            "merge_evidence": {
                "coarsening_only": _merge_evidence(baseline[sample_id], selected)[0],
                "minimum_hwr_gain": _merge_evidence(baseline[sample_id], selected)[1],
                "maximum_pair_gap_ref": _merge_evidence(baseline[sample_id], selected)[2],
            },
        })
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--hwr-checkpoint", type=Path, default=DEFAULT_PRODUCT)
    parser.add_argument("--context-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-base-checkpoint", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--candidate-loo-heads", type=Path)
    parser.add_argument("--formula-sequence-config", type=Path)
    parser.add_argument("--formula-syntax-rescue-config", type=Path)
    parser.add_argument("--candidate-context-auxiliary-hwr-checkpoint", type=Path)
    parser.add_argument("--auxiliary-hwr-checkpoint", type=Path)
    parser.add_argument("--auxiliary-fusion-weight", type=float, default=0.6)
    parser.add_argument("--formula-placement-config", type=Path)
    parser.add_argument("--straight-equality-config", type=Path)
    parser.add_argument("--latin-auxiliary-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.candidate_loo_heads is not None and args.candidate_base_checkpoint is None:
        parser.error("--candidate-loo-heads requires --candidate-base-checkpoint")
    auxiliary_arguments = (
        args.auxiliary_hwr_checkpoint, args.formula_placement_config,
        args.straight_equality_config,
    )
    if any(value is not None for value in auxiliary_arguments) and not all(
        value is not None for value in auxiliary_arguments
    ):
        parser.error("auxiliary HWR, formula placement, and straight equality configs are required together")
    if args.candidate_loo_heads is not None and args.auxiliary_hwr_checkpoint is not None:
        parser.error("product-fusion auxiliary cannot be mixed with writer-LOO evaluation")
    if args.candidate_loo_heads is not None and args.candidate_context_auxiliary_hwr_checkpoint is not None:
        parser.error("product singleton auxiliary cannot be mixed with writer-LOO evaluation")
    if not 0.0 <= args.auxiliary_fusion_weight <= 1.0:
        parser.error("--auxiliary-fusion-weight must be in [0, 1]")
    output = args.output.resolve()
    if output.exists() or output.drive.upper() != "D:":
        parser.error("output must be a new directory on D:")
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    if device_name == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    device = torch.device(device_name)
    dataset_root = args.dataset_root.resolve(); hwr_path = args.hwr_checkpoint.resolve(); context_path = args.context_checkpoint.resolve()
    sequence_path = args.formula_sequence_config.resolve() if args.formula_sequence_config else None
    syntax_path = args.formula_syntax_rescue_config.resolve() if args.formula_syntax_rescue_config else None
    samples, dataset = _samples(dataset_root, lattice_config=LATTICE_CONFIG)
    if len(samples) != 96:
        raise ValueError("chronological partition contract expects 96 ownership formulas")
    train, test = samples[:47], samples[47:]
    truth_labels = {
        str(row["sample_id"]): [str(value) for value in row["labels"]]
        for row in [json.loads(line) for line in (dataset_root / "data" / "ownership_train.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        if row.get("accepted")
    }
    hwr, labels, _ = _load_model(hwr_path, device)
    evaluation_only_writer_loo = args.candidate_loo_heads is not None
    if evaluation_only_writer_loo:
        candidate_base_path = args.candidate_base_checkpoint.resolve()
        candidate_loo_path = args.candidate_loo_heads.resolve()
        candidate_base, candidate_labels, _ = _load_model(candidate_base_path, device)
        if candidate_labels != labels:
            raise ValueError("candidate writer-LOO vocabulary differs from context HWR vocabulary")
        candidate_heads = _load_loo_heads(
            candidate_loo_path, candidate_base_path, candidate_labels, device,
        )
        embeddings, slices = _candidate_embeddings(samples, candidate_base, device)
        probability = _writer_loo_probabilities(
            samples, embeddings, slices, candidate_heads, labels, device,
        )
        candidate_policy = {
            "kind": "writer_loo_evaluation_only",
            "base_checkpoint_sha256": _sha256(candidate_base_path),
            "writer_loo_heads_sha256": _sha256(candidate_loo_path),
            "held_writer_absent_from_candidate_head_fit": True,
        }
    else:
        embeddings, slices = _candidate_embeddings(samples, hwr, device)
        probability = _probabilities(hwr.math_head, embeddings, device)
        candidate_policy = {
            "kind": "frozen_product_checkpoint",
            "checkpoint_sha256": _sha256(hwr_path),
            "held_writer_absent_from_candidate_head_fit": False,
        }
    auxiliary = None
    auxiliary_contract = {"enabled": False}
    singleton_auxiliary = None
    singleton_contract = {"enabled": False}
    latin_auxiliary = None
    latin_contract = {"enabled": False}
    latin_path = None
    if args.latin_auxiliary_checkpoint is not None:
        latin_path = args.latin_auxiliary_checkpoint.resolve()
        if not latin_path.is_file():
            parser.error("Latin auxiliary checkpoint is missing")
        latin_model, latin_labels = load_latin_auxiliary_model(latin_path, device)
        latin_auxiliary = {
            "model": latin_model,
            "labels": latin_labels,
            "device": device,
            "configuration": dict(PRODUCT_LATIN_T_CONFIGURATION),
        }
        latin_contract = {
            "enabled": True,
            "checkpoint_sha256": _sha256(latin_path),
            "candidate_width": 5,
            "target_token": "t",
            "product_default_enabled": False,
        }
    if args.candidate_context_auxiliary_hwr_checkpoint is not None:
        singleton_hwr_path = args.candidate_context_auxiliary_hwr_checkpoint.resolve()
        singleton_hwr, singleton_labels, _ = _load_model(singleton_hwr_path, device)
        if singleton_labels != labels:
            raise ValueError("singleton auxiliary HWR vocabulary differs from frozen HWR vocabulary")
        for name, value in hwr.state_dict().items():
            if not name.startswith("math_head.") and not torch.equal(
                value.detach().cpu(), singleton_hwr.state_dict()[name].detach().cpu()
            ):
                raise ValueError(f"singleton auxiliary and frozen HWR encoders differ: {name}")
        configuration = validate_singleton_configuration(
            PRODUCT_SINGLETON_CONFIGURATION,
        )
        numeric_configuration = validate_dual_numeric_configuration(
            PRODUCT_DUAL_NUMERIC_CONFIGURATION,
        )
        wide_configuration = validate_wide_syntax_configuration(
            PRODUCT_WIDE_SYNTAX_CONFIGURATION,
        )
        singleton_probability = _probabilities(
            singleton_hwr.math_head, embeddings, device,
        )
        singleton_weight = float(configuration["auxiliary_weight"])
        singleton_auxiliary = {
            "probability": (
                (1.0 - singleton_weight) * probability
                + singleton_weight * singleton_probability
            ),
            "slices": slices,
            "labels": labels,
            "policy": configuration["auxiliary_policy"],
            "weight": singleton_weight,
            "configuration": configuration,
            "numeric_configuration": numeric_configuration,
            "wide_configuration": wide_configuration,
        }
        singleton_contract = {
            "enabled": True,
            "rescue_schema": SINGLETON_SCHEMA,
            "configuration": configuration,
            "numeric_configuration": numeric_configuration,
            "wide_configuration": wide_configuration,
            "auxiliary_hwr_checkpoint_sha256": _sha256(singleton_hwr_path),
            "current_96_formula_training_overlap": True,
        }
    if args.auxiliary_hwr_checkpoint is not None:
        auxiliary_hwr_path = args.auxiliary_hwr_checkpoint.resolve()
        placement_path = args.formula_placement_config.resolve()
        equality_path = args.straight_equality_config.resolve()
        auxiliary_hwr, auxiliary_labels, _ = _load_model(auxiliary_hwr_path, device)
        if auxiliary_labels != labels:
            raise ValueError("auxiliary product HWR vocabulary differs from frozen HWR vocabulary")
        for name, value in hwr.state_dict().items():
            if not name.startswith("math_head.") and not torch.equal(
                value.detach().cpu(), auxiliary_hwr.state_dict()[name].detach().cpu()
            ):
                raise ValueError(f"auxiliary and frozen HWR encoders differ: {name}")
        auxiliary_probability = _probabilities(auxiliary_hwr.math_head, embeddings, device)
        weight = float(args.auxiliary_fusion_weight)
        fused_probability = (1.0 - weight) * probability + weight * auxiliary_probability
        placement_configuration, _ = _shadow_configuration(
            placement_path, config_schema=PLACEMENT_CONFIG_SCHEMA,
            rescue_schema=PLACEMENT_SCHEMA,
        )
        equality_configuration, _ = _shadow_configuration(
            equality_path, config_schema=EQUALITY_CONFIG_SCHEMA,
            rescue_schema=EQUALITY_SCHEMA,
        )
        if not math.isclose(float(placement_configuration["unmatched_fence_operand"]["auxiliary_weight"]), weight):
            raise ValueError("formula placement fusion weight mismatch")
        if not math.isclose(float(equality_configuration["auxiliary_weight"]), weight):
            raise ValueError("straight equality fusion weight mismatch")
        auxiliary = {
            "probability": fused_probability,
            "slices": slices,
            "labels": labels,
            "policy": "old_new_product_probability_fusion",
            "weight": weight,
            "placement_configuration": placement_configuration,
            "equality_configuration": equality_configuration,
        }
        auxiliary_contract = {
            "enabled": True,
            "policy": auxiliary["policy"],
            "fusion_weight": weight,
            "auxiliary_hwr_checkpoint_sha256": _sha256(auxiliary_hwr_path),
            "formula_placement_config_sha256": _sha256(placement_path),
            "straight_equality_config_sha256": _sha256(equality_path),
            "current_96_formula_training_overlap": True,
        }
    context_model, contract, payload = load_owned_formula_context(context_path, hwr_path, device, require_d_drive=False)
    finalizer = OwnedFormulaContextFinalizer(
        context_path, hwr_path, device=device_name, batch_size=128,
        semantic_guards=True, equation_correction=False, formula_layout=True,
        formula_sequence_config=sequence_path,
        formula_syntax_rescue_config=syntax_path,
    )

    writer_models = {
        writer: _fit([sample for sample in train if sample.writer != writer])
        for writer in sorted({sample.writer for sample in train})
    }
    chronological_grouping_model = _fit(train)
    train_partitions = _enumerate(train, writer_models, probability, slices, labels, writer_models=True)
    test_partitions = _enumerate(
        test, {"all": chronological_grouping_model}, probability, slices, labels,
        writer_models=False,
    )
    _attach_features(train_partitions + test_partitions, context_model, contract, payload, device)
    hwr_ranker, hwr_training = _fit_ranker(train_partitions, HWR_ONLY_FEATURE_COUNT)
    context_ranker, context_training = _fit_ranker(train_partitions, len(FEATURE_NAMES))
    hwr_gate = _select_gate(train_partitions, HWR_ONLY_FEATURE_COUNT)
    context_gate = _select_gate(train_partitions, len(FEATURE_NAMES))
    geometry_selected = _geometry_selection(test_partitions)
    hwr_selected = _select(test_partitions, hwr_ranker, HWR_ONLY_FEATURE_COUNT)
    context_selected = _select(test_partitions, context_ranker, len(FEATURE_NAMES))
    hwr_gated = _gated_selection(
        test_partitions, hwr_ranker, HWR_ONLY_FEATURE_COUNT, hwr_gate["winner"]["configuration"],
    )
    context_gated = _gated_selection(
        test_partitions, context_ranker, len(FEATURE_NAMES), context_gate["winner"]["configuration"],
    )
    context_design_guard = _gated_selection(
        test_partitions, context_ranker, len(FEATURE_NAMES), POSTHOC_DESIGN_GUARD,
    )
    relation_merge_configuration = _validate_relation_merge_configuration(
        RELATION_MERGE_CONFIGURATION,
    )
    relation_merge_guard, relation_merge_audit = _relation_merge_selection(
        test_partitions, context_design_guard, relation_merge_configuration,
    )
    cross_merge_configuration = _validate_cross_merge_configuration(
        CROSS_MERGE_CONFIGURATION,
    )
    cross_merge_guard, cross_merge_audit = _cross_merge_selection(
        test_partitions, relation_merge_guard, cross_merge_configuration,
        auxiliary,
    )
    repeat_merge_configuration = _validate_repeat_merge_configuration(
        REPEAT_MERGE_CONFIGURATION,
    )
    repeat_merge_guard, repeat_merge_audit = _repeat_merge_selection(
        test_partitions, cross_merge_guard, repeat_merge_configuration,
    )
    scores = {
        "geometry_top1": _score(test, geometry_selected, truth_labels),
        "geometry_hwr_ranker": _score(test, hwr_selected, truth_labels),
        "geometry_hwr_context_ranker": _score(test, context_selected, truth_labels),
        "geometry_hwr_ranker_gated": _score(test, hwr_gated, truth_labels),
        "geometry_hwr_context_ranker_gated": _score(test, context_gated, truth_labels),
        "geometry_hwr_context_ranker_posthoc_design_guard": _score(
            test, context_design_guard, truth_labels,
        ),
        "relation_merge_guard": _score(
            test, relation_merge_guard, truth_labels,
        ),
        "cross_merge_guard": _score(
            test, cross_merge_guard, truth_labels,
        ),
        "repeat_merge_guard": _score(
            test, repeat_merge_guard, truth_labels,
        ),
    }
    finalized_scores = {
        "geometry_top1": _score_finalized(
            test, geometry_selected, truth_labels, finalizer, auxiliary,
            singleton_auxiliary, latin_auxiliary,
        ),
        "posthoc_design_guard": _score_finalized(
            test, context_design_guard, truth_labels, finalizer, auxiliary,
            singleton_auxiliary, latin_auxiliary,
        ),
        "relation_merge_guard": _score_finalized(
            test, relation_merge_guard, truth_labels, finalizer, auxiliary,
            singleton_auxiliary, latin_auxiliary,
        ),
        "cross_merge_guard": _score_finalized(
            test, cross_merge_guard, truth_labels, finalizer, auxiliary,
            singleton_auxiliary, latin_auxiliary,
        ),
        "repeat_merge_guard": _score_finalized(
            test, repeat_merge_guard, truth_labels, finalizer, auxiliary,
            singleton_auxiliary, latin_auxiliary,
        ),
    }
    baseline = scores["geometry_top1"]
    candidate = scores["repeat_merge_guard"]
    baseline_ids = {sample.sample_id for sample in test if geometry_selected[sample.sample_id]["truth"]}
    candidate_ids = {sample.sample_id for sample in test if repeat_merge_guard[sample.sample_id]["truth"]}
    comparison = {
        "grouping_improved": sorted(candidate_ids - baseline_ids),
        "grouping_regressed": sorted(baseline_ids - candidate_ids),
        "grouping_exact_delta": candidate["grouping_exact_count"] - baseline["grouping_exact_count"],
        "context_formula_exact_delta": candidate["context_formula_exact_count"] - baseline["context_formula_exact_count"],
        "finalized_formula_exact_delta": (
            finalized_scores["repeat_merge_guard"]["final_formula_exact_count"]
            - finalized_scores["geometry_top1"]["final_formula_exact_count"]
        ),
    }
    output.mkdir(parents=True)
    model_path = output / "partition_context_ranker.joblib"
    joblib.dump({
        "schema": SCHEMA, "model_version": MODEL_VERSION, "model": context_ranker,
        "grouping_model": chronological_grouping_model,
        "feature_names": FEATURE_NAMES, "top_n": TOP_N, "lattice_config": LATTICE_CONFIG,
        "hwr_checkpoint_sha256": _sha256(hwr_path), "context_checkpoint_sha256": _sha256(context_path),
        "selection_guard": POSTHOC_DESIGN_GUARD,
        "relation_merge_guard": relation_merge_configuration,
        "cross_merge_guard": cross_merge_configuration,
        "repeat_merge_guard": repeat_merge_configuration,
        "finalizer_contract": {
            "formula_layout": True,
            "semantic_guards": True,
            "equation_correction": False,
            "formula_sequence_config_sha256": _sha256(sequence_path) if sequence_path else None,
            "formula_syntax_rescue_config_sha256": _sha256(syntax_path) if syntax_path else None,
            "auxiliary_candidates": auxiliary_contract,
            "singleton_shape_rescue": singleton_contract,
            "latin_auxiliary_candidates": latin_contract,
        },
        "grouping_training_scope": "first 47 accepted ownership formulas",
        "ranker_training_scope": "first 47 writer-LOO grouping candidate partitions",
        "posthoc_test_tuning": True,
        "candidate_policy": candidate_policy,
        "evaluation_only_writer_loo": evaluation_only_writer_loo,
        "requires_explicit_shadow_opt_in": True,
        "product_default_enabled": False,
    }, model_path, compress=3)
    report = {
        "schema": SCHEMA, "generated_at": datetime.now(timezone.utc).isoformat(), "status": "shadow",
        "model_version": MODEL_VERSION, "dataset": dataset,
        "protocol": {
            "fit": "first 47 accepted ownership formulas; writer-LOO grouping candidates",
            "test": "subsequent 49 accepted ownership formulas; frozen model and Top-5",
            "truth_partition_top5_test": sum(any(partition["truth"] for partition in test_partitions if partition["sample"].sample_id == sample.sample_id) for sample in test),
            "hwr_and_context_training_scope": "first 47 ownership formulas plus pre-existing external/synthetic contracts",
            "candidate_policy": candidate_policy,
            "auxiliary_candidates": auxiliary_contract,
            "singleton_shape_rescue": singleton_contract,
            "latin_auxiliary_candidates": latin_contract,
        },
        "training": {
            "hwr_ranker": hwr_training, "context_ranker": context_training,
            "hwr_gate": hwr_gate, "context_gate": context_gate,
            "posthoc_design_guard": {
                "configuration": POSTHOC_DESIGN_GUARD,
                "status": "development_only_after_inspecting_49_formula_failures",
            },
            "relation_merge_guard": relation_merge_audit,
            "cross_merge_guard": cross_merge_audit,
            "repeat_merge_guard": repeat_merge_audit,
        },
        "scores": scores, "finalized_scores": finalized_scores, "comparison": comparison,
        "selection_changes": {
            "hwr_ranker_gated": _selection_changes(geometry_selected, hwr_gated),
            "context_ranker_gated": _selection_changes(geometry_selected, context_gated),
            "context_ranker_posthoc_design_guard": _selection_changes(
                geometry_selected, context_design_guard,
            ),
            "relation_merge_guard": _selection_changes(
                context_design_guard, relation_merge_guard,
            ),
            "cross_merge_guard": _selection_changes(
                relation_merge_guard, cross_merge_guard,
            ),
            "repeat_merge_guard": _selection_changes(
                cross_merge_guard, repeat_merge_guard,
            ),
        },
        "artifact": {"file": model_path.name, "sha256": _sha256(model_path), "feature_names": list(FEATURE_NAMES)},
        "contracts": {
            "truth_label_or_target_count_input": False, "top_n": TOP_N,
            "all_strokes_exactly_once": True, "arithmetic_solver": False,
            "product_default_enabled": False,
            "posthoc_test_tuning": True,
        },
        "limits": [
            "the 49-formula arrival set has already been inspected in later research loops",
            "formula exact is flat token-sequence exact; 2D relation equivalence is outside this score",
        ],
    }
    (output / "partition_context_ranker_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n",
    )
    runtime = {
        "schema": "aiflow-partition-context-runtime-config/v1",
        "status": "development_only_posthoc_shadow",
        "model_version": MODEL_VERSION,
        "partition_ranker": str(model_path),
        "partition_ranker_sha256": _sha256(model_path),
        "hwr_checkpoint": str(hwr_path),
        "hwr_checkpoint_sha256": _sha256(hwr_path),
        "context_checkpoint": str(context_path),
        "context_checkpoint_sha256": _sha256(context_path),
        "selection_guard": POSTHOC_DESIGN_GUARD,
        "finalizer": {
            "formula_layout": True,
            "semantic_guards": True,
            "equation_correction": False,
            "formula_sequence_config": str(sequence_path) if sequence_path else None,
            "formula_sequence_config_sha256": _sha256(sequence_path) if sequence_path else None,
            "formula_syntax_rescue_config": str(syntax_path) if syntax_path else None,
            "formula_syntax_rescue_config_sha256": _sha256(syntax_path) if syntax_path else None,
            "auxiliary_candidates": {
                **auxiliary_contract,
                "auxiliary_hwr_checkpoint": str(auxiliary_hwr_path) if auxiliary is not None else None,
                "formula_placement_config": str(placement_path) if auxiliary is not None else None,
                "straight_equality_config": str(equality_path) if auxiliary is not None else None,
            },
            "singleton_shape_rescue": {
                **singleton_contract,
                "auxiliary_hwr_checkpoint": (
                    str(singleton_hwr_path)
                    if singleton_auxiliary is not None else None
                ),
            },
            "latin_auxiliary_candidates": {
                **latin_contract,
                "checkpoint": str(latin_path) if latin_path is not None else None,
            },
        },
        "requires_explicit_shadow_opt_in": True,
        "product_default_enabled": False,
    }
    if evaluation_only_writer_loo:
        runtime = {
            "schema": runtime["schema"],
            "status": "evaluation_only_writer_loo_not_runtime_admissible",
            "partition_ranker": str(model_path),
            "partition_ranker_sha256": _sha256(model_path),
            "candidate_policy": candidate_policy,
            "requires_writer_identity": True,
            "product_default_enabled": False,
        }
    (output / "partition_context_runtime_config.json").write_text(
        json.dumps(runtime, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    print(json.dumps({
        "output": str(output), "scores": {key: {field: value for field, value in score.items() if field != "failures"} for key, score in scores.items()},
        "finalized_scores": {
            key: {field: value for field, value in score.items() if field not in {"failures", "audit"}}
            for key, score in finalized_scores.items()
        },
        "comparison": comparison, "model_sha256": _sha256(model_path),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
