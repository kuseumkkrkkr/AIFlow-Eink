#!/usr/bin/env python3
"""Microscope the frozen HWR layer path for candidate-present residual errors."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


SCHEMA = "aiflow-hwr-candidate-residual-layer-microscope/v1"
STAGES = (
    "encoder.input",
    "encoder.block_0.output",
    "encoder.block_1.output",
    "encoder.block_2.output",
    "encoder.block_3.output",
    "encoder.output",
)
DIGITS = set("0123456789")
OPERATORS = {"+", "-", "/", r"\times", r"\div", r"\cdot"}
RELATIONS = {"=", "<", ">", r"\leq", r"\geq", r"\neq", r"\approx", r"\sim"}
FENCES = {"(", ")", "[", "]", "{", "}", "|", r"\{", r"\}", r"\lfloor", r"\rfloor"}


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stroke_key(values: Any) -> tuple[int, ...]:
    return tuple(int(value) for value in values)


def _coarse_class(token: str) -> str:
    if token in DIGITS:
        return "digit"
    if token in OPERATORS:
        return "arithmetic_operator"
    if token in RELATIONS:
        return "relation"
    if token in FENCES:
        return "fence"
    if len(token) == 1 and token.isascii() and token.isalpha():
        return "latin_letter"
    if token.startswith("\\"):
        return "named_symbol"
    return "other"


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (float(ordered[middle - 1]) + float(ordered[middle])) / 2.0


def audit(candidate_path: Path, trace_path: Path) -> dict[str, Any]:
    candidate = _json(candidate_path)
    traces = _jsonl(trace_path)
    if candidate.get("schema") != "aiflow-hwr-guarded-top5-rescore-shadow/v1":
        raise ValueError("unexpected candidate-rescore artifact schema")
    if candidate.get("status") != "consumed_development_cached_shadow_only":
        raise ValueError("candidate artifact is not a consumed-development shadow")
    if candidate.get("promotion_eligible") is not False:
        raise ValueError("candidate artifact unexpectedly claims promotion eligibility")

    trace_by_id = {str(row["sample_id"]): row for row in traces}
    if len(trace_by_id) != len(traces):
        raise AssertionError("duplicate formula IDs in layer traces")
    if len(traces) != int(candidate["formulas"]):
        raise AssertionError("candidate and trace formula counts differ")

    residual_formulas = [
        row for row in candidate["post_rescore_failure_breakdown"]["formulas"]
        if row["cause"] == "top5_present_but_rescore_sequence_wrong"
    ]
    if len(residual_formulas) != int(
        candidate["post_rescore_failure_breakdown"]["group_exact_cause_counts"][
            "top5_present_but_rescore_sequence_wrong"
        ]
    ):
        raise AssertionError("residual formula count disagrees with cause summary")

    stage_rows: dict[str, list[dict[str, Any]]] = {stage: [] for stage in STAGES}
    rank_histogram: Counter[str] = Counter()
    first_top5_histogram: Counter[str] = Counter()
    class_confusions: Counter[tuple[str, str]] = Counter()
    detail_rows: list[dict[str, Any]] = []
    checks = Counter()

    for formula in residual_formulas:
        sample_id = str(formula["sample_id"])
        trace = trace_by_id.get(sample_id)
        if trace is None:
            raise AssertionError(f"missing layer trace for residual formula {sample_id}")
        symbols_by_group = {
            _stroke_key(symbol["stroke_indices"]): symbol
            for symbol in trace["oracle_group_hwr"]["symbols"]
        }
        for mismatch in formula["mismatched_groups"]:
            key = _stroke_key(mismatch["stroke_indices"])
            symbol = symbols_by_group.get(key)
            if symbol is None:
                raise AssertionError(f"no exact gold-group trace for {sample_id}:{key}")
            target = str(mismatch["target"])
            top1 = str(mismatch["hwr_top1"])
            prediction = symbol["prediction"]
            if str(symbol["target_label"]) != target:
                raise AssertionError(f"target label mismatch for {sample_id}:{key}")
            if str(prediction["top1"]) != top1:
                raise AssertionError(f"HWR Top-1 mismatch for {sample_id}:{key}")
            if [str(token) for token in prediction["top5"]] != [
                str(token) for token in mismatch["top5"]
            ]:
                raise AssertionError(f"Top-5 mismatch for {sample_id}:{key}")
            if int(prediction["target_rank"]) != int(mismatch["target_hwr_rank"]):
                raise AssertionError(f"target rank mismatch for {sample_id}:{key}")

            trajectory = symbol["layer_logit_lens"]
            if tuple(str(row["stage"]) for row in trajectory) != STAGES:
                raise AssertionError(f"layer stage sequence mismatch for {sample_id}:{key}")
            ranks: list[int] = []
            margins: list[float] = []
            for layer in trajectory:
                if layer["target_in_vocab"] is not True:
                    raise AssertionError(f"Top-5 residual is unexpectedly OOV: {sample_id}:{key}")
                ranks.append(int(layer["target_rank"]))
                margins.append(float(layer["target_minus_best_other_logit"]))
                stage_rows[str(layer["stage"])].append({
                    "sample_id": sample_id,
                    "stroke_indices": list(key),
                    "target_rank": ranks[-1],
                    "target_minus_best_other_logit": margins[-1],
                })
            if ranks[-1] != int(prediction["target_rank"]):
                raise AssertionError(f"final-layer target rank differs from saved prediction: {sample_id}:{key}")
            if str(trajectory[-1]["predicted_top1"]) != top1:
                raise AssertionError(f"final-layer Top-1 differs from saved prediction: {sample_id}:{key}")

            target_probability = float(prediction["target_probability"])
            top1_probability = float(prediction["top1_probability"])
            probability_log_gap = math.log(top1_probability) - math.log(target_probability)
            final_logit_gap = -margins[-1]
            if not math.isclose(probability_log_gap, final_logit_gap, rel_tol=1e-4, abs_tol=1e-4):
                raise AssertionError(f"probability/logit gap parity failed: {sample_id}:{key}")

            rank_histogram[str(ranks[-1])] += 1
            first_top5 = next((STAGES[i] for i, rank in enumerate(ranks) if rank <= 5), None)
            if first_top5 is None:
                raise AssertionError(f"final Top-5 target absent from every layer: {sample_id}:{key}")
            first_top5_histogram[first_top5] += 1
            target_class = _coarse_class(target)
            top1_class = _coarse_class(top1)
            class_confusions[(target_class, top1_class)] += 1
            checks["mismatch_groups"] += 1
            checks["final_top1_errors"] += int(target != top1)
            checks["top5_hits"] += int(target in [str(token) for token in prediction["top5"]])

            detail_rows.append({
                "sample_id": sample_id,
                "stroke_indices": list(key),
                "target": target,
                "target_class": target_class,
                "hwr_top1": top1,
                "top1_class": top1_class,
                "target_hwr_rank": ranks[-1],
                "top5": [str(token) for token in prediction["top5"]],
                "target_probability": target_probability,
                "top1_probability": top1_probability,
                "top1_over_target_probability_log_gap": probability_log_gap,
                "layer_trajectory": [
                    {
                        "stage": stage,
                        "target_rank": rank,
                        "predicted_top1": str(layer["predicted_top1"]),
                        "target_minus_best_other_logit": margin,
                    }
                    for stage, rank, margin, layer in zip(STAGES, ranks, margins, trajectory, strict=True)
                ],
                "first_top5_stage": first_top5,
                "best_rank_stage": STAGES[min(range(len(ranks)), key=ranks.__getitem__)],
                "rank_improved_input_to_final": ranks[-1] < ranks[0],
                "rank_best_better_than_final": min(ranks) < ranks[-1],
                "margin_became_positive_at_any_layer": any(value > 0.0 for value in margins),
                "raw_point_count": int(symbol["preprocessing"]["raw_point_count"]),
                "raw_points_per_stroke": [
                    int(value) for value in symbol["preprocessing"]["raw_points_per_stroke"]
                ],
            })

    stage_summary: dict[str, Any] = {}
    for stage in STAGES:
        rows = stage_rows[stage]
        ranks = [int(row["target_rank"]) for row in rows]
        margins = [float(row["target_minus_best_other_logit"]) for row in rows]
        stage_summary[stage] = {
            "mismatch_symbols": len(rows),
            "target_top1": sum(rank == 1 for rank in ranks),
            "target_top5": sum(rank <= 5 for rank in ranks),
            "mean_target_rank": sum(ranks) / len(ranks) if ranks else None,
            "median_target_rank": _median([float(rank) for rank in ranks]),
            "mean_target_minus_best_other_logit": sum(margins) / len(margins) if margins else None,
            "median_target_minus_best_other_logit": _median(margins),
        }

    if checks["mismatch_groups"] != sum(len(row["mismatched_groups"]) for row in residual_formulas):
        raise AssertionError("not every candidate-present mismatch was replayed")
    if checks["top5_hits"] != checks["mismatch_groups"]:
        raise AssertionError("not every target remained in Top-5")

    return {
        "schema": SCHEMA,
        "status": "consumed_development_forensic_shadow_only",
        "scope": "no training, no threshold fitting, no CROHME, no promotion",
        "inputs": {
            "candidate_rescore_path": str(candidate_path),
            "candidate_rescore_sha256": _sha256(candidate_path),
            "layer_trace_path": str(trace_path),
            "layer_trace_sha256": _sha256(trace_path),
            "candidate_formula_count": int(candidate["formulas"]),
            "trace_formula_count": len(traces),
            "residual_formula_count": len(residual_formulas),
        },
        "summary": {
            "mismatch_symbol_count": checks["mismatch_groups"],
            "all_top1_wrong": checks["final_top1_errors"] == checks["mismatch_groups"],
            "all_targets_replayed_in_top5": checks["top5_hits"] == checks["mismatch_groups"],
            "final_target_rank_histogram": dict(sorted(rank_histogram.items())),
            "first_top5_stage_histogram": dict(first_top5_histogram),
            "class_confusions": [
                {"target_class": target, "top1_class": predicted, "count": count}
                for (target, predicted), count in sorted(class_confusions.items())
            ],
            "rank_improved_input_to_final": sum(bool(row["rank_improved_input_to_final"]) for row in detail_rows),
            "best_intermediate_rank_better_than_final": sum(bool(row["rank_best_better_than_final"]) for row in detail_rows),
            "any_intermediate_positive_target_margin": sum(bool(row["margin_became_positive_at_any_layer"]) for row in detail_rows),
            "mean_top1_over_target_log_probability_gap": (
                sum(float(row["top1_over_target_probability_log_gap"]) for row in detail_rows) / len(detail_rows)
                if detail_rows else None
            ),
            "layer_summary": stage_summary,
        },
        "details": detail_rows,
        "checks": {
            "candidate_schema_valid": True,
            "trace_formula_count_matches": len(traces) == int(candidate["formulas"]),
            "every_residual_group_matches_gold_group_trace": checks["mismatch_groups"] == sum(
                len(row["mismatched_groups"]) for row in residual_formulas
            ),
            "top1_top5_target_rank_and_final_layer_all_match": True,
            "final_probability_gap_matches_logit_gap": True,
            "all_targets_in_final_top5": checks["top5_hits"] == checks["mismatch_groups"],
            "all_checks_pass": True,
        },
        "product_default_enabled": False,
        "crohme_training_or_tuning": False,
        "promotion_eligible": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-rescore", type=Path, required=True)
    parser.add_argument("--layer-trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.candidate_rescore, args.layer_trace)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "mismatch_symbol_count": result["summary"]["mismatch_symbol_count"],
        "all_checks_pass": result["checks"]["all_checks_pass"],
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
