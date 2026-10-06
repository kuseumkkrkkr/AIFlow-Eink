#!/usr/bin/env python3
"""Replay the deterministic semantic-guard shadow on a hash-verified saved A/B/C/D run.

This reuses already stored HWR Top-5 rows and grouping choices. It does not load
or run the HWR model, train, tune, inspect CROHME, or alter product defaults.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from raw_formula_context_runtime_v1 import (
    SEMANTIC_GUARD_SHADOW_SCHEMA,
    _candidate_preserving_semantic_guard_shadow,
)


SCHEMA = "aiflow-semantic-guard-saved-tournament-audit/v11"
ARMS = (
    "fast", "selective_geometry", "selective_joint_hwr",
    "selective_joint_hwr_geometry_prior",
)
STAGES = (
    "decoder", "after_fence_guard", "after_infix_guard", "after_equation_guard",
    "after_arithmetic_expression_guard",
    "after_unique_bar_equation_guard", "after_unique_exact_equation_candidates_guard",
    "after_boundary_bar_as_unit_guard",
    "after_unique_candidate_arithmetic_equation_guard", "after_terminal_rhs_bar_guard",
)
ACCEPTANCE_THRESHOLDS = {
    "minimum_formula_count": 160,
    "minimum_writer_count": 8,
    "maximum_formulas_per_writer": 20,
    "minimum_device_count": 4,
    "minimum_input_stack_count": 2,
}
DEVICE_METADATA_FIELDS = (
    "device_id", "device_identifier", "device_model", "device_type",
)
INPUT_STACK_METADATA_FIELDS = (
    "input_stack", "input_stack_id", "input_method", "capture_stack",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _partition(groups: list[list[int]]) -> set[frozenset[int]]:
    return {frozenset(int(index) for index in group) for group in groups}


def _point_xy(point: Any) -> tuple[float, float]:
    if isinstance(point, dict):
        return float(point["x"]), float(point["y"])
    return float(point[0]), float(point[1])


def _group_box(strokes: list[dict[str, Any]], indices: list[int]) -> dict[str, float]:
    points = [
        _point_xy(point)
        for index in indices
        for point in (strokes[index].get("points") or [])
    ]
    if not points:
        raise ValueError("saved selected group has no source points")
    xs, ys = zip(*points, strict=True)
    box = {
        "left": min(xs), "top": min(ys),
        "right": max(xs), "bottom": max(ys),
    }
    if not all(math.isfinite(value) for value in box.values()):
        raise ValueError("source group box contains non-finite coordinates")
    return box


def _transition(before: bool, after: bool) -> str:
    if before and after:
        return "both_exact"
    if before:
        return "regressed"
    if after:
        return "recovered"
    return "both_wrong"


def _values_for_field(value: Any, field_name: str) -> set[str]:
    values: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).casefold() == field_name.casefold() and isinstance(
                child, (str, int, float),
            ) and not isinstance(child, bool) and str(child).strip():
                values.add(str(child).strip())
            values.update(_values_for_field(child, field_name))
    elif isinstance(value, list):
        for child in value:
            values.update(_values_for_field(child, field_name))
    return values


def _metadata_count(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> dict[str, Any]:
    for field in fields:
        values = set().union(*(_values_for_field(row, field) for row in rows))
        if values:
            return {"count": len(values), "field": field}
    return {"count": None, "field": None}


def _acceptance_scope(
    records: list[dict[str, Any]], annotations: dict[str, dict[str, Any]],
    formulas: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    writer_counts = Counter(
        str(row.get("writer_id") or "unknown") for row in annotations.values()
    )
    formula_rows = [formulas[sample_id] for sample_id in annotations]
    device_observation = _metadata_count(formula_rows, DEVICE_METADATA_FIELDS)
    input_stack_observation = _metadata_count(
        formula_rows, INPUT_STACK_METADATA_FIELDS,
    )
    writer_count = sum(writer != "unknown" for writer in writer_counts)
    max_per_writer = max(
        (count for writer, count in writer_counts.items() if writer != "unknown"),
        default=0,
    )
    checks: dict[str, bool | None] = {
        "minimum_formula_count": len(records) >= ACCEPTANCE_THRESHOLDS[
            "minimum_formula_count"
        ],
        "minimum_writer_count": writer_count >= ACCEPTANCE_THRESHOLDS[
            "minimum_writer_count"
        ],
        "maximum_formulas_per_writer": max_per_writer <= ACCEPTANCE_THRESHOLDS[
            "maximum_formulas_per_writer"
        ],
        "minimum_device_count": (
            None if device_observation["count"] is None
            else device_observation["count"] >= ACCEPTANCE_THRESHOLDS[
                "minimum_device_count"
            ]
        ),
        "minimum_input_stack_count": (
            None if input_stack_observation["count"] is None
            else input_stack_observation["count"] >= ACCEPTANCE_THRESHOLDS[
                "minimum_input_stack_count"
            ]
        ),
    }
    if sum(writer_counts.values()) != len(annotations):
        raise AssertionError("writer cluster counts do not cover accepted formulas")
    if len(records) != len(annotations):
        raise AssertionError("acceptance formula count differs from accepted labels")
    return {
        "thresholds": ACCEPTANCE_THRESHOLDS,
        "observed": {
            "formula_count": len(records),
            "writer_count": writer_count,
            "max_formulas_per_writer": max_per_writer,
            "writer_formula_counts": dict(sorted(writer_counts.items())),
            "device_count": device_observation["count"],
            "device_metadata_field": device_observation["field"],
            "input_stack_count": input_stack_observation["count"],
            "input_stack_metadata_field": input_stack_observation["field"],
        },
        "checks": checks,
        "formal_acceptance_ready": all(value is True for value in checks.values()),
    }


def _writer_cluster_metrics(
    output_rows: list[dict[str, Any]], annotations: dict[str, dict[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    accumulators: dict[str, dict[str, dict[str, Any]]] = {
        arm: {} for arm in ARMS
    }
    for row in output_rows:
        sample_id = str(row["sample_id"])
        writer_id = str(annotations[sample_id].get("writer_id") or "unknown")
        for arm in ARMS:
            cluster = accumulators[arm].setdefault(writer_id, {
                "formula_count": 0,
                "group_exact_formula_count": 0,
                "strict_formula_exact_by_stage": Counter(),
            })
            metrics = row["arms"][arm]
            cluster["formula_count"] += 1
            cluster["group_exact_formula_count"] += int(bool(metrics["group_exact"]))
            exact_by_stage = metrics.get("formula_exact_by_stage") or {}
            for stage in STAGES:
                cluster["strict_formula_exact_by_stage"][stage] += int(
                    bool(exact_by_stage.get(stage, False))
                )

    output: dict[str, dict[str, dict[str, Any]]] = {}
    for arm, clusters in accumulators.items():
        output[arm] = {}
        if sum(cluster["formula_count"] for cluster in clusters.values()) != len(output_rows):
            raise AssertionError(f"writer clusters do not cover formulas for {arm}")
        for writer_id, cluster in sorted(clusters.items()):
            formula_count = int(cluster["formula_count"])
            exact_counts = dict(cluster["strict_formula_exact_by_stage"])
            output[arm][writer_id] = {
                "formula_count": formula_count,
                "group_exact_formula_count": int(cluster["group_exact_formula_count"]),
                "strict_formula_exact_by_stage": exact_counts,
                "strict_formula_exact_rate_by_stage": {
                    stage: exact_counts[stage] / formula_count
                    for stage in STAGES
                },
            }
    return output


def audit(tournament_path: Path, dataset_root: Path) -> dict[str, Any]:
    tournament_path = tournament_path.resolve()
    dataset_root = dataset_root.resolve()
    tournament = json.loads(tournament_path.read_text(encoding="utf-8"))
    if tournament.get("schema") != "aiflow-selective-2d-research-loop/v1":
        raise ValueError("unsupported saved tournament schema")
    if tournament.get("promotion_eligible") is not False:
        raise ValueError("saved tournament is not marked shadow-only")
    inputs = tournament.get("inputs") or {}
    if Path(str(inputs.get("dataset_root"))).resolve() != dataset_root:
        raise ValueError("dataset root does not match the saved tournament")
    formulas_path = dataset_root / "data" / "formulas_valid.jsonl"
    ownership_path = dataset_root / "data" / "ownership_train.jsonl"
    expected_files = {
        "formulas_valid_sha256": formulas_path,
        "ownership_train_sha256": ownership_path,
    }
    file_hashes = {}
    for key, path in expected_files.items():
        actual = _sha256(path)
        if actual != inputs.get(key):
            raise ValueError(f"saved tournament {key} does not match {path}")
        file_hashes[str(path)] = actual
    for path_key, hash_key in (
        ("checkpoint", "checkpoint_sha256"),
        ("partition_ranker", "partition_ranker_sha256"),
    ):
        path = Path(str(inputs.get(path_key) or ""))
        if not path.is_file() or _sha256(path) != inputs.get(hash_key):
            raise ValueError(f"saved tournament {path_key} artifact hash mismatch")

    formulas = {str(row["sample_id"]): row for row in _jsonl(formulas_path)}
    annotations = {
        str(row["sample_id"]): row
        for row in _jsonl(ownership_path)
        if row.get("accepted")
    }
    records = list(tournament.get("records") or [])
    if len(records) != 149 or len(annotations) != 149:
        raise AssertionError("saved A/B/C/D acceptance scope must contain 149 formulas")
    if {str(row["sample_id"]) for row in records} != set(annotations):
        raise AssertionError("saved tournament and accepted ownership IDs differ")
    if not set(annotations) <= set(formulas):
        raise AssertionError("accepted ownership rows missing from raw formula data")
    for sample_id, annotation in annotations.items():
        formula_writer = str(formulas[sample_id].get("writer_id") or "")
        annotation_writer = str(annotation.get("writer_id") or "")
        if formula_writer and annotation_writer and formula_writer != annotation_writer:
            raise AssertionError(f"writer provenance mismatch: {sample_id}")
    acceptance_scope = _acceptance_scope(records, annotations, formulas)

    arm_formula_counts = {arm: Counter() for arm in ARMS}
    arm_token_hits = {arm: Counter() for arm in ARMS}
    arm_token_totals = Counter()
    arm_transitions = {arm: {
        f"{before}_to_{after}": Counter()
        for before, after in zip(STAGES, STAGES[1:])
    } for arm in ARMS}
    arm_final_vs_base = {arm: Counter() for arm in ARMS}
    arm_shadow_status = {arm: Counter() for arm in ARMS}
    arm_fence_status = {arm: Counter() for arm in ARMS}
    arm_fence_changes = {arm: [] for arm in ARMS}
    arm_expression_status = {arm: Counter() for arm in ARMS}
    arm_expression_changes = {arm: [] for arm in ARMS}
    arm_bar_status = {arm: Counter() for arm in ARMS}
    arm_bar_changes = {arm: [] for arm in ARMS}
    arm_boundary_bar_status = {arm: Counter() for arm in ARMS}
    arm_boundary_bar_changes = {arm: [] for arm in ARMS}
    arm_unique_equation_status = {arm: Counter() for arm in ARMS}
    arm_unique_equation_changes = {arm: [] for arm in ARMS}
    arm_unique_equation_search = {arm: Counter() for arm in ARMS}
    arm_unique_candidate_equation_status = {arm: Counter() for arm in ARMS}
    arm_unique_candidate_equation_changes = {arm: [] for arm in ARMS}
    arm_unique_candidate_equation_search = {arm: Counter() for arm in ARMS}
    arm_terminal_rhs_bar_status = {arm: Counter() for arm in ARMS}
    arm_terminal_rhs_bar_changes = {arm: [] for arm in ARMS}
    arm_target_ranks = {arm: Counter() for arm in ARMS}
    arm_candidate_headroom = {arm: Counter() for arm in ARMS}
    arm_residual_confusions = {arm: Counter() for arm in ARMS}
    arm_residual_target_ranks = {arm: Counter() for arm in ARMS}
    arm_residual_formula_ids = {arm: [] for arm in ARMS}
    output_rows = []
    candidate_checks = 0
    candidate_violations = 0
    group_mutations = 0

    for record in records:
        sample_id = str(record["sample_id"])
        source = formulas[sample_id]
        annotation = annotations[sample_id]
        strokes = sorted(source["strokes"], key=lambda row: int(row["order"]))
        target_groups = [
            [int(index) for index in group]
            for group in annotation["groups"]
        ]
        target_tokens = [str(token) for token in annotation["labels"]]
        if len(target_groups) != len(target_tokens):
            raise AssertionError(f"target group/token mismatch: {sample_id}")
        truth_by_group = {
            frozenset(group): token
            for group, token in zip(target_groups, target_tokens, strict=True)
        }
        row_result = {"sample_id": sample_id, "arms": {}}

        for arm in ARMS:
            arm_row = record["hwr_tournament"][arm]
            groups = [[int(index) for index in group] for group in arm_row["groups"]]
            group_exact = _partition(groups) == _partition(target_groups)
            if bool(arm_row["group_exact"]) != group_exact:
                raise AssertionError(f"stored grouping exactness mismatch: {sample_id}/{arm}")
            symbols = []
            for symbol in arm_row["selected_symbols"]:
                indices = [int(index) for index in symbol["stroke_indices"]]
                symbols.append({
                    "stroke_indices": indices,
                    "hwr_topk": [str(token) for token in symbol["hwr_topk"]],
                    "hwr_topk_probabilities": [
                        float(value) for value in symbol["hwr_topk_probabilities"]
                    ],
                    "geometry": _group_box(strokes, indices),
                })
            if {frozenset(row["stroke_indices"]) for row in symbols} != _partition(groups):
                raise AssertionError(f"stored HWR symbols do not cover groups: {sample_id}/{arm}")
            decoder = {
                "accepted": bool(arm_row["decoder_accepted"]),
                "tokens": [str(token) for token in arm_row["decoder_tokens"]],
            }
            shadow = _candidate_preserving_semantic_guard_shadow(
                sample_id, groups, symbols, decoder,
            )
            arm_shadow_status[arm][str(shadow.get("status"))] += 1
            fence_audit = dict((shadow.get("audits") or {}).get("fence") or {})
            arm_fence_status[arm][str(
                "changed_shadow_only" if fence_audit.get("changes") else "unchanged"
            )] += 1
            if fence_audit.get("changes"):
                arm_fence_changes[arm].append({
                    "sample_id": sample_id,
                    "changes": list(fence_audit["changes"]),
                    "changed_glyphs": int(fence_audit.get("changed_glyphs", 0)),
                    "changed_selected_fences": int(
                        fence_audit.get("changed_selected_fences", 0)
                    ),
                    "formula_exact_before": None,
                    "formula_exact_after": None,
                    "token_hits_before": None,
                    "token_hits_after": None,
                })
            bar_audit = dict(
                (shadow.get("audits") or {}).get("unique_unpaired_bar_equation") or {}
            )
            boundary_bar_audit = dict(
                (shadow.get("audits") or {}).get("boundary_bar_as_unit") or {}
            )
            expression_audit = dict(
                (shadow.get("audits") or {}).get("arithmetic_expression") or {}
            )
            unique_equation_audit = dict(
                (shadow.get("audits") or {}).get(
                    "unique_exact_equation_candidates"
                ) or {}
            )
            unique_candidate_equation_audit = dict(
                (shadow.get("audits") or {}).get(
                    "unique_candidate_arithmetic_equation"
                ) or {}
            )
            terminal_rhs_bar_audit = dict(
                (shadow.get("audits") or {}).get("terminal_rhs_bar") or {}
            )
            arm_bar_status[arm][str(bar_audit.get("status", "missing"))] += 1
            if bar_audit.get("status") == "changed_shadow_only":
                changes = list(bar_audit.get("changes") or [])
                if len(changes) != 1:
                    raise AssertionError("unique-bar rule must change exactly one candidate")
                arm_bar_changes[arm].append({
                    "sample_id": sample_id,
                    "changes": changes,
                    "formula_exact_before": None,
                    "formula_exact_after": None,
                })
            arm_expression_status[arm][str(expression_audit.get("status", "missing"))] += 1
            if expression_audit.get("status") == "changed_shadow_only":
                changes = list(expression_audit.get("changes") or [])
                changed_glyphs = sum(len(row.get("changes") or []) for row in changes)
                if not changes or not 1 <= changed_glyphs <= 3:
                    raise AssertionError(
                        "arithmetic-expression rule must change one to three candidates"
                    )
                arm_expression_changes[arm].append({
                    "sample_id": sample_id,
                    "changes": changes,
                    "formula_exact_before": None,
                    "formula_exact_after": None,
                })
            arm_boundary_bar_status[arm][
                str(boundary_bar_audit.get("status", "missing"))
            ] += 1
            if boundary_bar_audit.get("status") == "changed_shadow_only":
                changes = list(boundary_bar_audit.get("changes") or [])
                if len(changes) != 1:
                    raise AssertionError("boundary-bar rule must change exactly one candidate")
                arm_boundary_bar_changes[arm].append({
                    "sample_id": sample_id,
                    "changes": changes,
                    "formula_exact_before": None,
                    "formula_exact_after": None,
                })
            arm_unique_equation_status[arm][
                str(unique_equation_audit.get("status", "missing"))
            ] += 1
            checked_sequences = int(unique_equation_audit.get("candidate_sequences_checked", 0))
            arm_unique_equation_search[arm]["formulas_with_search"] += int(
                checked_sequences > 0
            )
            arm_unique_equation_search[arm]["candidate_sequences_checked"] += checked_sequences
            arm_unique_equation_search[arm]["ambiguous_formulas"] += int(
                unique_equation_audit.get("reason") == "multiple_exact_equation_candidates"
            )
            arm_unique_equation_search[arm]["no_exact_candidate_formulas"] += int(
                unique_equation_audit.get("reason") == "no_exact_equation_candidate"
            )
            if unique_equation_audit.get("status") == "changed_shadow_only":
                changes = list(unique_equation_audit.get("changes") or [])
                if not 1 <= len(changes) <= 3:
                    raise AssertionError("unique-equation rule must change one to three candidates")
                arm_unique_equation_changes[arm].append({
                    "sample_id": sample_id,
                    "changes": changes,
                    "formula_exact_before": None,
                    "formula_exact_after": None,
                })
            arm_unique_candidate_equation_status[arm][
                str(unique_candidate_equation_audit.get("status", "missing"))
            ] += 1
            candidate_sequences_checked = int(
                unique_candidate_equation_audit.get("candidate_sequences_checked", 0)
            )
            arm_unique_candidate_equation_search[arm]["formulas_with_search"] += int(
                candidate_sequences_checked > 0
            )
            arm_unique_candidate_equation_search[arm][
                "candidate_sequences_checked"
            ] += candidate_sequences_checked
            arm_unique_candidate_equation_search[arm]["ambiguous_formulas"] += int(
                unique_candidate_equation_audit.get("reason")
                == "ambiguous_exact_candidate_equations"
            )
            arm_unique_candidate_equation_search[arm]["no_exact_candidate_formulas"] += int(
                unique_candidate_equation_audit.get("reason")
                == "no_exact_candidate_equation"
            )
            if unique_candidate_equation_audit.get("status") == "changed_shadow_only":
                changes = list(unique_candidate_equation_audit.get("changes") or [])
                if not 1 <= len(changes) <= 3:
                    raise AssertionError(
                        "unique candidate equation rule exceeded its edit budget"
                    )
                arm_unique_candidate_equation_changes[arm].append({
                    "sample_id": sample_id,
                    "changes": changes,
                    "formula_exact_before": None,
                    "formula_exact_after": None,
                })
            arm_terminal_rhs_bar_status[arm][
                str(terminal_rhs_bar_audit.get("status", "missing"))
            ] += 1
            if terminal_rhs_bar_audit.get("status") == "changed_shadow_only":
                changes = list(terminal_rhs_bar_audit.get("changes") or [])
                if (
                    len(changes) != 1
                    or changes[0].get("before") != "|"
                    or changes[0].get("after") != "1"
                ):
                    raise AssertionError("terminal RHS bar rule changed an unexpected token")
                arm_terminal_rhs_bar_changes[arm].append({
                    "sample_id": sample_id,
                    "changes": changes,
                    "formula_exact_before": None,
                    "formula_exact_after": None,
                })
            metrics = {
                "group_exact": group_exact,
                "status": shadow.get("status"),
                "formula_exact_by_stage": None,
                "token_hits_by_stage": None,
                "fence_guard": {
                    "changes": list(fence_audit.get("changes") or []),
                    "changed_glyphs": int(fence_audit.get("changed_glyphs", 0)),
                    "changed_selected_fences": int(
                        fence_audit.get("changed_selected_fences", 0)
                    ),
                },
                "unique_unpaired_bar_equation": bar_audit,
                "arithmetic_expression": expression_audit,
                "boundary_bar_as_unit": boundary_bar_audit,
                "unique_exact_equation_candidates": unique_equation_audit,
                "unique_candidate_arithmetic_equation": unique_candidate_equation_audit,
                "terminal_rhs_bar": terminal_rhs_bar_audit,
            }
            if shadow.get("status") == "applied_shadow_only":
                candidate_checks += sum(len(shadow["stages"][stage]) for stage in STAGES)
                candidate_violations += int(not shadow["candidate_preservation"])
                group_mutations += int(shadow["grouping_mutations"] != 0)
                if shadow["base_decoder_tokens"] != decoder["tokens"]:
                    raise AssertionError(f"shadow changed the saved decoder: {sample_id}/{arm}")
                stage_maps = {
                    stage: {
                        frozenset(int(index) for index in row["stroke_indices"]): str(row["token"])
                        for row in shadow["stages"][stage]
                    }
                    for stage in STAGES
                }
                if any(len(stage_maps[stage]) != len(groups) for stage in STAGES):
                    raise AssertionError(f"semantic guard stage changed group multiplicity: {sample_id}/{arm}")
                if group_exact:
                    arm_formula_counts[arm]["evaluated"] += 1
                    arm_token_totals[arm] += len(target_tokens)
                    symbol_by_group = {
                        frozenset(int(index) for index in symbol["stroke_indices"]): symbol
                        for symbol in symbols
                    }
                    target_ranks = []
                    for target_group, target_token in zip(
                        target_groups, target_tokens, strict=True,
                    ):
                        topk = symbol_by_group[frozenset(target_group)]["hwr_topk"]
                        rank = topk.index(target_token) + 1 if target_token in topk else None
                        target_ranks.append(rank)
                        arm_target_ranks[arm][
                            str(rank) if rank is not None else "missing_or_outside_top5"
                        ] += 1
                    all_targets_top5 = all(rank is not None for rank in target_ranks)
                    all_targets_top1 = all(rank == 1 for rank in target_ranks)
                    arm_candidate_headroom[arm]["evaluated_group_exact_formulas"] += 1
                    arm_candidate_headroom[arm]["all_targets_in_top5"] += int(all_targets_top5)
                    arm_candidate_headroom[arm]["all_targets_top1"] += int(all_targets_top1)
                    exact_by_stage = {}
                    hits_by_stage = {}
                    for stage in STAGES:
                        prediction = [stage_maps[stage][frozenset(group)] for group in target_groups]
                        exact = prediction == target_tokens
                        hits = sum(
                            predicted == truth
                            for predicted, truth in zip(prediction, target_tokens, strict=True)
                        )
                        exact_by_stage[stage] = exact
                        hits_by_stage[stage] = hits
                        arm_formula_counts[arm][f"exact:{stage}"] += int(exact)
                        arm_token_hits[arm][stage] += hits
                    for before, after in zip(STAGES[:-1], STAGES[1:], strict=True):
                        arm_transitions[arm][f"{before}_to_{after}"][_transition(
                            exact_by_stage[before], exact_by_stage[after],
                        )] += 1
                    arm_final_vs_base[arm][_transition(
                        exact_by_stage["decoder"], exact_by_stage["after_equation_guard"],
                    )] += 1
                    metrics["formula_exact_by_stage"] = exact_by_stage
                    metrics["token_hits_by_stage"] = hits_by_stage
                    if fence_audit.get("changes"):
                        change_row = arm_fence_changes[arm][-1]
                        change_row["formula_exact_before"] = exact_by_stage["decoder"]
                        change_row["formula_exact_after"] = exact_by_stage[
                            "after_fence_guard"
                        ]
                        change_row["token_hits_before"] = hits_by_stage["decoder"]
                        change_row["token_hits_after"] = hits_by_stage[
                            "after_fence_guard"
                        ]
                    if expression_audit.get("status") == "changed_shadow_only":
                        change_row = arm_expression_changes[arm][-1]
                        change_row["formula_exact_before"] = exact_by_stage[
                            "after_equation_guard"
                        ]
                        change_row["formula_exact_after"] = exact_by_stage[
                            "after_arithmetic_expression_guard"
                        ]
                    final_exact = exact_by_stage["after_terminal_rhs_bar_guard"]
                    decoder_exact = exact_by_stage["decoder"]
                    residual_glyph_errors = []
                    final_stage = "after_terminal_rhs_bar_guard"
                    for target_group, target_token in zip(
                        target_groups, target_tokens, strict=True,
                    ):
                        group_key = frozenset(target_group)
                        predicted_token = stage_maps[final_stage][group_key]
                        if predicted_token == target_token:
                            continue
                        topk = symbol_by_group[group_key]["hwr_topk"]
                        target_rank = (
                            topk.index(target_token) + 1 if target_token in topk else None
                        )
                        rank_key = (
                            str(target_rank) if target_rank is not None
                            else "missing_or_outside_top5"
                        )
                        arm_residual_confusions[arm][(target_token, predicted_token)] += 1
                        arm_residual_target_ranks[arm][rank_key] += 1
                        residual_glyph_errors.append({
                            "stroke_indices": sorted(target_group),
                            "target": target_token,
                            "prediction": predicted_token,
                            "target_hwr_rank": target_rank,
                        })
                    metrics["residual_glyph_errors_after_guard"] = residual_glyph_errors
                    if residual_glyph_errors:
                        arm_residual_formula_ids[arm].append(sample_id)
                    if all_targets_top5 and final_exact:
                        arm_candidate_headroom[arm]["top5_complete_final_exact"] += 1
                    if all_targets_top1 and final_exact:
                        arm_candidate_headroom[arm]["all_top1_final_exact"] += 1
                    if not final_exact:
                        if all_targets_top5:
                            arm_candidate_headroom[arm]["top5_complete_final_wrong"] += 1
                            if not decoder_exact:
                                arm_candidate_headroom[arm]["decoder_wrong_with_full_top5"] += 1
                                if all_targets_top1:
                                    arm_candidate_headroom[arm][
                                        "all_top1_correct_but_decoder_wrong"
                                    ] += 1
                        else:
                            arm_candidate_headroom[arm]["top5_missing_final_wrong"] += 1
                    if decoder_exact and not final_exact:
                        arm_candidate_headroom[arm]["guard_regressed_decoder_exact"] += 1
                    if bar_audit.get("status") == "changed_shadow_only":
                        change_row = arm_bar_changes[arm][-1]
                        change_row["formula_exact_before"] = exact_by_stage[
                            "after_arithmetic_expression_guard"
                        ]
                        change_row["formula_exact_after"] = exact_by_stage[
                            "after_unique_bar_equation_guard"
                        ]
                    if unique_equation_audit.get("status") == "changed_shadow_only":
                        change_row = arm_unique_equation_changes[arm][-1]
                        change_row["formula_exact_before"] = exact_by_stage[
                            "after_unique_bar_equation_guard"
                        ]
                        change_row["formula_exact_after"] = exact_by_stage[
                            "after_unique_exact_equation_candidates_guard"
                        ]
                    if boundary_bar_audit.get("status") == "changed_shadow_only":
                        change_row = arm_boundary_bar_changes[arm][-1]
                        change_row["formula_exact_before"] = exact_by_stage[
                            "after_unique_exact_equation_candidates_guard"
                        ]
                        change_row["formula_exact_after"] = exact_by_stage[
                            "after_boundary_bar_as_unit_guard"
                        ]
                    if unique_candidate_equation_audit.get("status") == "changed_shadow_only":
                        change_row = arm_unique_candidate_equation_changes[arm][-1]
                        change_row["formula_exact_before"] = exact_by_stage[
                            "after_boundary_bar_as_unit_guard"
                        ]
                        change_row["formula_exact_after"] = exact_by_stage[
                            "after_unique_candidate_arithmetic_equation_guard"
                        ]
                    if terminal_rhs_bar_audit.get("status") == "changed_shadow_only":
                        change_row = arm_terminal_rhs_bar_changes[arm][-1]
                        change_row["formula_exact_before"] = exact_by_stage[
                            "after_unique_candidate_arithmetic_equation_guard"
                        ]
                        change_row["formula_exact_after"] = exact_by_stage[
                            "after_terminal_rhs_bar_guard"
                        ]
            row_result["arms"][arm] = metrics
        output_rows.append(row_result)

    writer_clusters = _writer_cluster_metrics(output_rows, annotations)
    arm_summary = {}
    for arm in ARMS:
        evaluated = arm_formula_counts[arm]["evaluated"]
        exact = {
            stage: arm_formula_counts[arm][f"exact:{stage}"]
            for stage in STAGES
        }
        arm_summary[arm] = {
            "group_exact_formulas_in_saved_run": sum(
                bool(record["hwr_tournament"][arm]["group_exact"])
                for record in records
            ),
            "shadow_applied_formulas": sum(arm_shadow_status[arm].values())
                - arm_shadow_status[arm]["skipped"],
            "shadow_skipped_formulas": arm_shadow_status[arm]["skipped"],
            "evaluated_exact_group_formulas": evaluated,
            "evaluated_tokens": arm_token_totals[arm],
            "formula_exact_by_stage": exact,
            "token_hits_by_stage": {
                stage: arm_token_hits[arm][stage] for stage in STAGES
            },
            "formula_exact_transitions": {
                key: dict(value) for key, value in arm_transitions[arm].items()
            },
            "fence_guard_status": dict(arm_fence_status[arm]),
            "fence_guard_changed_formulas": arm_fence_changes[arm],
            "arithmetic_expression_guard_status": dict(arm_expression_status[arm]),
            "arithmetic_expression_guard_changed_formulas": arm_expression_changes[arm],
            "arithmetic_expression_guard_exact_transition": dict(
                arm_transitions[arm][
                    "after_equation_guard_to_after_arithmetic_expression_guard"
                ]
            ),
            "decoder_to_equation_guard_transition": dict(arm_final_vs_base[arm]),
            "unique_bar_guard_status": dict(arm_bar_status[arm]),
            "unique_bar_guard_changed_formulas": arm_bar_changes[arm],
            "unique_bar_guard_exact_transition": dict(
                arm_transitions[arm][
                    "after_arithmetic_expression_guard_to_after_unique_bar_equation_guard"
                ]
            ),
            "boundary_bar_as_unit_status": dict(arm_boundary_bar_status[arm]),
            "boundary_bar_as_unit_changed_formulas": arm_boundary_bar_changes[arm],
            "boundary_bar_as_unit_exact_transition": dict(
                arm_transitions[arm][
                    "after_unique_exact_equation_candidates_guard_to_after_boundary_bar_as_unit_guard"
                ]
            ),
            "unique_exact_equation_candidates_status": dict(
                arm_unique_equation_status[arm]
            ),
            "unique_exact_equation_candidates_changed_formulas": (
                arm_unique_equation_changes[arm]
            ),
            "unique_exact_equation_candidates_search": dict(
                arm_unique_equation_search[arm]
            ),
            "unique_exact_equation_candidates_exact_transition": dict(
                arm_transitions[arm][
                    "after_unique_bar_equation_guard_to_after_unique_exact_equation_candidates_guard"
                ]
            ),
            "unique_candidate_arithmetic_equation_status": dict(
                arm_unique_candidate_equation_status[arm]
            ),
            "unique_candidate_arithmetic_equation_changed_formulas": (
                arm_unique_candidate_equation_changes[arm]
            ),
            "unique_candidate_arithmetic_equation_search": dict(
                arm_unique_candidate_equation_search[arm]
            ),
            "unique_candidate_arithmetic_equation_exact_transition": dict(
                arm_transitions[arm][
                    "after_boundary_bar_as_unit_guard_to_after_unique_candidate_arithmetic_equation_guard"
                ]
            ),
            "terminal_rhs_bar_status": dict(arm_terminal_rhs_bar_status[arm]),
            "terminal_rhs_bar_changed_formulas": arm_terminal_rhs_bar_changes[arm],
            "terminal_rhs_bar_exact_transition": dict(
                arm_transitions[arm][
                    "after_unique_candidate_arithmetic_equation_guard_to_after_terminal_rhs_bar_guard"
                ]
            ),
            "target_hwr_rank_histogram_on_exact_groups": dict(arm_target_ranks[arm]),
            "candidate_headroom_on_exact_groups": dict(arm_candidate_headroom[arm]),
            "residual_glyph_confusions_after_guard": [
                {"target": target, "prediction": prediction, "count": count}
                for (target, prediction), count in arm_residual_confusions[arm].most_common(20)
            ],
            "residual_glyph_target_rank_histogram_after_guard": dict(
                arm_residual_target_ranks[arm]
            ),
            "residual_formula_ids_after_guard": arm_residual_formula_ids[arm],
        }
    if candidate_violations or group_mutations:
        raise AssertionError("saved tournament semantic shadow violated an invariant")
    return {
        "schema": SCHEMA,
        "semantic_guard_shadow_schema": SEMANTIC_GUARD_SHADOW_SCHEMA,
        "evaluation_status": {
            "role": "posthoc_development_shadow",
            "same_frozen_cohort_inspected_for_rule_selection": True,
            "independent_acceptance_claim_allowed": False,
        },
        "scope": (
            "posthoc shadow replay of the same frozen cohort inspected during lookalike-rule "
            "development; saved A/B/C/D predictions; no HWR inference, training, CROHME, or promotion"
        ),
        "formulas": len(records),
        "acceptance_scope": acceptance_scope,
        "inputs": {
            "tournament_summary": str(tournament_path),
            "tournament_summary_sha256": _sha256(tournament_path),
            "dataset_root": str(dataset_root),
            "dataset_file_sha256": file_hashes,
            "hwr_checkpoint_sha256": inputs["checkpoint_sha256"],
            "partition_ranker_sha256": inputs["partition_ranker_sha256"],
        },
        "audit": {
            "candidate_checks": candidate_checks,
            "candidate_violations": candidate_violations,
            "group_mutations": group_mutations,
            "product_default_enabled": False,
            "crohme_training_or_tuning": False,
            "promotion_eligible": False,
        },
        "arms": arm_summary,
        "writer_cluster_metrics": writer_clusters,
        "formula_level": output_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tournament-summary", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite output: {args.output}")
    report = audit(args.tournament_summary, args.dataset_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    args.output.write_text(serialized, encoding="utf-8", newline="\n")
    compact_arms = {
        arm: {
            "group_exact_formulas_in_saved_run": metrics[
                "group_exact_formulas_in_saved_run"
            ],
            "evaluated_exact_group_formulas": metrics[
                "evaluated_exact_group_formulas"
            ],
            "formula_exact_by_stage": metrics["formula_exact_by_stage"],
        }
        for arm, metrics in report["arms"].items()
    }
    print(json.dumps({
        "output": str(args.output.resolve()),
        "output_sha256": _sha256(args.output),
        "evaluation_status": report["evaluation_status"],
        "acceptance_scope": report["acceptance_scope"],
        "arms": compact_arms,
        "audit": report["audit"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
