#!/usr/bin/env python3
"""Describe numeric infix confusions, geometry, and all-digit controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from audit_widened_semantic_guard_shadow_v1 import _guard_formula, _jsonl


SCHEMA = "aiflow-hwr-numeric-infix-geometry-microscope/v1"
ARITHMETIC_OPERATORS = {"+", "-", "/", r"\times", r"\div", r"\cdot"}
GEOMETRY_ROUTER_THRESHOLDS = (0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 3.0)
DIGIT = re.compile(r"^[0-9]$")
ARMS = {
    "top5_unique": (5, "unique", 0.0, 0.01),
    "top20_unique": (20, "unique", 0.0, 0.01),
    "top20_relative_0003": (20, "operator_relative", 0.0, 0.0003),
    "top20_relative_002": (20, "operator_relative", 0.0, 0.002),
    "top20_numeric_context_002": (20, "operator_numeric_context", 0.0, 0.002),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _percentiles(values: list[float]) -> dict[str, float | None]:
    ordered = sorted(values)
    if not ordered:
        return {"p25": None, "median": None, "p75": None}
    return {
        "p25": ordered[int(0.25 * (len(ordered) - 1))],
        "median": ordered[int(0.50 * (len(ordered) - 1))],
        "p75": ordered[int(0.75 * (len(ordered) - 1))],
    }


def _geometry_router_sweep(contexts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    correct_count = sum(context["top1_center"] == context["target_center"] for context in contexts)
    error_count = len(contexts) - correct_count
    output = []
    for threshold in GEOMETRY_ROUTER_THRESHOLDS:
        routed = [float(context["bbox_aspect_ratio"]) <= threshold for context in contexts]
        routed_errors = sum(
            flag and context["top1_center"] != context["target_center"]
            for context, flag in zip(contexts, routed, strict=True)
        )
        routed_correct = sum(
            flag and context["top1_center"] == context["target_center"]
            for context, flag in zip(contexts, routed, strict=True)
        )
        routed_count = sum(routed)
        output.append({
            "route_if_aspect_at_most": threshold,
            "routed_contexts": routed_count,
            "routed_fraction": routed_count / len(contexts) if contexts else 0.0,
            "error_contexts_captured": routed_errors,
            "error_recall": routed_errors / error_count if error_count else None,
            "correct_contexts_false_routed": routed_correct,
            "false_positive_rate": routed_correct / correct_count if correct_count else None,
            "error_precision_in_routed_set": routed_errors / routed_count if routed_count else None,
        })
    return output


def _collect_contexts(
    traces: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    operator_contexts = []
    equality_contexts = []
    digit_triplets = []
    other_numeric_contexts = []
    for trace in traces:
        targets = [str(token) for token in trace["source"]["target_tokens"]]
        symbols = trace["oracle_group_hwr"]["symbols"]
        if len(targets) != len(symbols):
            raise AssertionError(f"target/symbol length mismatch: {trace['sample_id']}")
        for position in range(1, len(targets) - 1):
            left, middle, right = targets[position - 1:position + 2]
            is_digit_triplet = all(DIGIT.fullmatch(token) for token in (left, middle, right))
            numeric_neighbors = (
                DIGIT.fullmatch(left) is not None
                and DIGIT.fullmatch(right) is not None
            )
            if not numeric_neighbors:
                continue
            is_numeric_infix = middle in ARITHMETIC_OPERATORS

            symbol = symbols[position]
            prediction = symbol["prediction"]
            bbox = symbol["preprocessing"]["raw_bbox"]
            left_bbox = symbols[position - 1]["preprocessing"]["raw_bbox"]
            right_bbox = symbols[position + 1]["preprocessing"]["raw_bbox"]
            width = max(float(bbox["right"]) - float(bbox["left"]), 0.0)
            height = max(float(bbox["bottom"]) - float(bbox["top"]), 1e-6)
            top1_probability = float(prediction["top5_probabilities"][0])
            diagnostic = prediction["diagnostic_topk"]
            diagnostic_tokens = [str(token) for token in diagnostic["tokens"]]
            diagnostic_probabilities = [float(value) for value in diagnostic["probabilities"]]
            target_rank = next(
                (index + 1 for index, token in enumerate(diagnostic_tokens) if token == middle), None,
            )
            record = {
                "sample_id": str(trace["sample_id"]),
                "position": position,
                "target_triplet": [left, middle, right],
                "target_center": middle,
                "top1_center": str(prediction["top5"][0]),
                "target_rank": target_rank,
                "target_probability": (
                    diagnostic_probabilities[target_rank - 1] if target_rank is not None else None
                ),
                "top1_probability": top1_probability,
                "target_to_top1_probability_ratio": (
                    diagnostic_probabilities[target_rank - 1] / max(top1_probability, 1e-12)
                    if target_rank is not None else None
                ),
                "bbox_aspect_ratio": width / height,
                "left_gap_over_center_height": (
                    float(bbox["left"]) - float(left_bbox["right"])
                ) / height,
                "right_gap_over_center_height": (
                    float(right_bbox["left"]) - float(bbox["right"])
                ) / height,
            }
            if is_numeric_infix:
                operator_contexts.append(record)
            elif middle == "=":
                equality_contexts.append(record)
            elif is_digit_triplet:
                digit_triplets.append(record)
            else:
                other_numeric_contexts.append(record)
    return operator_contexts, equality_contexts, digit_triplets, other_numeric_contexts


def _arm_context_metrics(
    contexts: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    rows_by_id = {str(row["sample_id"]): row for row in rows}
    stage_names = ("hwr_top1", "after_infix", "after_expression")
    stage_totals = {stage: {"correct": 0} for stage in stage_names}
    recovered = {stage: [] for stage in stage_names[1:]}
    regressed = {stage: [] for stage in stage_names[1:]}
    details = []
    by_operator: dict[str, dict[str, int]] = {}
    for context in contexts:
        sample_id = str(context["sample_id"])
        position = int(context["position"])
        target = str(context["target_center"])
        row = rows_by_id[sample_id]
        predictions = {
            stage: str(row["pipeline_stages"][stage]["tokens"][position])
            for stage in stage_names
        }
        group = by_operator.setdefault(target, {stage: 0 for stage in stage_names} | {"count": 0})
        group["count"] += 1
        for stage in stage_names:
            if predictions[stage] == target:
                stage_totals[stage]["correct"] += 1
                group[stage] += 1
        for stage in stage_names[1:]:
            before_correct = predictions["hwr_top1"] == target
            after_correct = predictions[stage] == target
            item = {
                "sample_id": sample_id,
                "position": position,
                "target": target,
                "before": predictions["hwr_top1"],
                "after": predictions[stage],
            }
            if not before_correct and after_correct:
                recovered[stage].append(item)
            elif before_correct and not after_correct:
                regressed[stage].append(item)
        if len(set(predictions.values())) > 1 or predictions["hwr_top1"] != target:
            details.append({**context, "predictions": predictions})

    trace_geometry = [
        float(context["bbox_aspect_ratio"])
        for context in contexts
        if context["top1_center"] == context["target_center"]
    ]
    error_geometry = [
        float(context["bbox_aspect_ratio"])
        for context in contexts
        if context["top1_center"] != context["target_center"]
    ]
    return {
        "contexts": len(contexts),
        "target_candidate_recall": {
            "top5": sum(context["target_rank"] is not None and context["target_rank"] <= 5 for context in contexts),
            "top20": sum(context["target_rank"] is not None and context["target_rank"] <= 20 for context in contexts),
        },
        "correct_by_stage": stage_totals,
        "recovered_vs_top1": recovered,
        "regressed_vs_top1": regressed,
        "by_target_operator": by_operator,
        "geometry_aspect_ratio": {
            "top1_correct": _percentiles(trace_geometry),
            "top1_wrong": _percentiles(error_geometry),
        },
        "geometry_aspect_router_sweep": _geometry_router_sweep(contexts),
        "nontrivial_cases": details,
    }


def audit(trace_path: Path, reference_path: Path) -> dict[str, Any]:
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    if reference.get("schema") != "aiflow-hwr-widened-semantic-guard-shadow/v1":
        raise ValueError("unexpected widened-shadow reference schema")
    actual_trace_sha = _sha256(trace_path)
    expected_trace_sha = reference["inputs"]["top20_trace"]["sha256"]
    if actual_trace_sha != expected_trace_sha:
        raise AssertionError("numeric-infix trace hash differs from frozen reference")
    traces = _jsonl(trace_path)
    if len(traces) != 149:
        raise AssertionError(f"expected frozen 149-formula trace, got {len(traces)}")
    operator_contexts, equality_contexts, digit_triplets, other_numeric_contexts = _collect_contexts(traces)
    arm_rows: dict[str, list[dict[str, Any]]] = {}
    arms = {}
    for name, (k, infix_policy, min_ratio, competitor_ratio) in ARMS.items():
        rows = [
            _guard_formula(
                trace, k, infix_policy=infix_policy,
                minimum_infix_probability_ratio=min_ratio,
                maximum_operator_competitor_probability_ratio=competitor_ratio,
            )
            for trace in traces
        ]
        arm_rows[name] = rows
        arms[name] = {
            "candidate_k": k,
            "infix_policy": infix_policy,
            "minimum_infix_probability_ratio": min_ratio,
            "maximum_operator_competitor_probability_ratio": competitor_ratio,
            "numeric_infix_contexts": _arm_context_metrics(
                operator_contexts, rows,
            ),
            "numeric_equality_contexts": _arm_context_metrics(
                equality_contexts, rows,
            ),
            "all_digit_triplet_controls": _arm_context_metrics(
                digit_triplets, rows,
            ),
            "other_numeric_neighbor_contexts": _arm_context_metrics(
                other_numeric_contexts, rows,
            ),
        }

    reference_argmax = reference["pipeline_formula_level_by_k"]["top20_operator_argmax_shadow"]
    argmax_by_id = {str(row["sample_id"]): row for row in reference_argmax}
    current_argmax_matches = all(
        arm_rows["top20_relative_002"][index]["pipeline_stages"]["after_expression"]["tokens"]
        == argmax_by_id[str(arm_rows["top20_relative_002"][index]["sample_id"])]["stages"]["after_expression"]["tokens"]
        for index in range(len(traces))
    )
    other_numeric_context_changes = {
        arm: sum(
            item["predictions"]["hwr_top1"] != item["predictions"]["after_infix"]
            for item in arms[arm]["other_numeric_neighbor_contexts"]["nontrivial_cases"]
        )
        for arm in ARMS
    }
    numeric_context_rows = arm_rows["top20_numeric_context_002"]
    relative_rows = arm_rows["top20_relative_002"]
    numeric_context_delta = {
        "after_infix_changed_formulas_vs_relative_dominance_0.002": [],
        "after_expression_changed_formulas_vs_relative_dominance_0.002": [],
        "changed_token_positions_after_infix": [],
    }
    for numeric_row, relative_row in zip(numeric_context_rows, relative_rows, strict=True):
        sample_id = str(numeric_row["sample_id"])
        numeric_stages = numeric_row["pipeline_stages"]
        relative_stages = relative_row["pipeline_stages"]
        for stage, key in (
            ("after_infix", "after_infix_changed_formulas_vs_relative_dominance_0.002"),
            ("after_expression", "after_expression_changed_formulas_vs_relative_dominance_0.002"),
        ):
            if numeric_stages[stage]["tokens"] != relative_stages[stage]["tokens"]:
                numeric_context_delta[key].append(sample_id)
        changed_positions = [
            position for position, (before, after) in enumerate(zip(
                relative_stages["after_infix"]["tokens"],
                numeric_stages["after_infix"]["tokens"],
                strict=True,
            ))
            if before != after
        ]
        for position in changed_positions:
            numeric_context_delta["changed_token_positions_after_infix"].append({
                "sample_id": sample_id,
                "position": position,
                "from": relative_stages["after_infix"]["tokens"][position],
                "to": numeric_stages["after_infix"]["tokens"][position],
            })
    report = {
        "schema": SCHEMA,
        "scope": "oracle-group Top-k diagnostic on consumed development; descriptive geometry only; no geometry threshold fitting or promotion",
        "inputs": {
            "trace_sha256": actual_trace_sha,
            "reference_report_sha256": _sha256(reference_path),
            "formula_count": len(traces),
            "top20_trace_sha256_expected": expected_trace_sha,
        },
        "definitions": {
            "numeric_infix_context": "target left and right are one-digit tokens; target center is +, -, /, times, div, or cdot",
            "numeric_equality_context": "target left and right are one-digit tokens; target center is =",
            "all_digit_triplet_control": "target left, center, and right are each one digit",
            "other_numeric_neighbor_context": "target left and right are one-digit tokens; center is neither an arithmetic operator, =, nor a digit",
            "geometry": "raw group bounding-box aspect ratio and adjacent horizontal gap divided by center height",
            "geometry_risk_router": "descriptive only; flags a group when its raw bbox aspect ratio is at most the listed threshold",
        },
        "context_counts": {
            "numeric_infix_contexts": len(operator_contexts),
            "numeric_equality_contexts": len(equality_contexts),
            "all_digit_triplet_controls": len(digit_triplets),
            "other_numeric_neighbor_contexts": len(other_numeric_contexts),
            "numeric_infix_errors_at_hwr_top1": sum(
                context["top1_center"] != context["target_center"]
                for context in operator_contexts
            ),
        },
        "arms": arms,
        "numeric_context_policy_delta_vs_relative_dominance_0.002": numeric_context_delta,
        "verification": {
            "trace_hash_matches_frozen_reference": actual_trace_sha == expected_trace_sha,
            "frozen_formula_count_is_149": len(traces) == 149,
            "top20_relative_002_replays_argmax_formula_predictions": current_argmax_matches,
            "all_output_tokens_preserve_candidate_membership": all(
                row["candidate_preservation"]
                for rows in arm_rows.values()
                for row in rows
            ),
            "new_arm_matches_relative_arm_outside_numeric_operator_role_contexts": (
                all(
                    int(numeric_row["pipeline_guard_audits"]["infix"].get(
                        "numeric_operator_role_changes", 0,
                    )) == sum(
                        before != after
                        for before, after in zip(
                            relative_row["pipeline_stages"]["after_infix"]["tokens"],
                            numeric_row["pipeline_stages"]["after_infix"]["tokens"],
                            strict=True,
                        )
                    )
                    for numeric_row, relative_row in zip(
                        numeric_context_rows, relative_rows, strict=True,
                    )
                )
            ),
            "other_numeric_context_change_counts_reconciled": all(
                count == len([
                    item for item in arms[name]["other_numeric_neighbor_contexts"]["nontrivial_cases"]
                    if item["predictions"]["hwr_top1"] != item["predictions"]["after_infix"]
                ])
                for name, count in other_numeric_context_changes.items()
            ),
        },
    }
    report["verification"]["all_checks_pass"] = all(report["verification"].values())
    if not report["verification"]["all_checks_pass"]:
        raise AssertionError("numeric-infix geometry microscope failed verification")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.trace, args.reference_report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "all_checks_pass": report["verification"]["all_checks_pass"],
        "numeric_infix_contexts": report["context_counts"]["numeric_infix_contexts"],
        "top1_errors": report["context_counts"]["numeric_infix_errors_at_hwr_top1"],
        "all_digit_triplet_controls": report["context_counts"]["all_digit_triplet_controls"],
        "numeric_equality_contexts": report["context_counts"]["numeric_equality_contexts"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
