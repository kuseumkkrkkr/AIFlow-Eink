#!/usr/bin/env python3
"""Trace valid decoder survivors that lose back through the frozen HWR layers."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


SCHEMA = "aiflow-hwr-decoder-survivor-layer-microscope/v1"
LAYER_STAGES = (
    "encoder.input",
    "encoder.block_0.output",
    "encoder.block_1.output",
    "encoder.block_2.output",
    "encoder.block_3.output",
    "encoder.output",
)
DIGIT = re.compile(r"^[0-9]$")
ARITHMETIC = {"+", "-", "/", r"\times", r"\div", r"\cdot"}
FENCES = {"(", ")", "[", "]", "{", "}", "|", r"\{", r"\}", r"\lfloor", r"\rfloor"}
RELATIONS = {"=", "<", ">", r"\leq", r"\geq", r"\neq", r"\approx", r"\sim"}


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


def _coarse_class(token: str) -> str:
    token = str(token)
    if DIGIT.fullmatch(token):
        return "digit"
    if token in ARITHMETIC:
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


def audit(trace_path: Path, cause_path: Path) -> dict[str, Any]:
    traces = _jsonl(trace_path)
    cause = _json(cause_path)
    trace_sha = _sha256(trace_path)
    cause_sha = _sha256(cause_path)
    trace_by_id = {str(row["sample_id"]): row for row in traces}
    if len(trace_by_id) != len(traces):
        raise AssertionError("duplicate formula IDs in layer traces")
    expected_trace_sha = cause["inputs"]["traces"]["sha256"]
    if trace_sha != expected_trace_sha:
        raise AssertionError("survivor layer trace hash differs from frozen cause matrix")
    if cause.get("schema") != "aiflow-hwr-failure-cause-microscope/v1":
        raise ValueError("unexpected failure-cause matrix schema")

    survivors = [
        row for row in cause["residual_details"]
        if row["cause"] == "survives_but_loses"
    ]
    if len(survivors) != int(cause["summary"]["surviving_valid_targets"]):
        raise AssertionError("survivor count differs from failure-cause summary")

    stage_rows: dict[str, list[dict[str, Any]]] = {stage: [] for stage in LAYER_STAGES}
    signatures: Counter[tuple[str, str]] = Counter()
    mismatch_count = 0
    replay_checks = 0
    score_gaps: list[float] = []
    total_score_gaps: list[float] = []
    relation_gaps: list[float] = []
    details: list[dict[str, Any]] = []
    for formula in survivors:
        sample_id = str(formula["sample_id"])
        trace = trace_by_id[sample_id]
        targets = [str(token) for token in formula["target_tokens"]]
        selected = [str(token) for token in formula["selected_tokens"]]
        symbols = trace["oracle_group_hwr"]["symbols"]
        if len(targets) != len(selected) or len(targets) != len(symbols):
            raise AssertionError(f"formula token/symbol lengths differ: {sample_id}")
        if [str(token) for token in trace["source"]["target_tokens"]] != targets:
            raise AssertionError(f"cause matrix targets differ from trace: {sample_id}")
        token_rows = []
        for position, (target, picked, symbol) in enumerate(zip(targets, selected, symbols, strict=True)):
            if target == picked:
                continue
            mismatch_count += 1
            prediction = symbol["prediction"]
            if str(symbol["target_label"]) != target or str(prediction["top1"]) != picked:
                raise AssertionError(f"selected-token replay differs from trace: {sample_id}:{position}")
            if target not in [str(token) for token in prediction["top5"]]:
                raise AssertionError(f"valid survivor target absent from Top-5: {sample_id}:{position}")
            if int(prediction["target_rank"]) != int(next(
                row["target_hwr_rank"]
                for row in formula["token_level_selection"]
                if int(row["position"]) == position + 1
            )):
                raise AssertionError(f"HWR target rank differs from selection trace: {sample_id}:{position}")
            ranks = []
            margins = []
            trajectory = symbol["layer_logit_lens"]
            if tuple(str(row["stage"]) for row in trajectory) != LAYER_STAGES:
                raise AssertionError(f"layer stage contract differs: {sample_id}:{position}")
            if (
                int(trajectory[-1]["target_rank"]) != int(prediction["target_rank"])
                or str(trajectory[-1]["predicted_top1"]) != str(prediction["top1"])
            ):
                raise AssertionError(f"final layer trace differs from saved HWR output: {sample_id}:{position}")
            for layer in trajectory:
                stage = str(layer["stage"])
                rank = int(layer["target_rank"])
                margin = float(layer["target_minus_best_other_logit"])
                if layer["target_in_vocab"] is not True:
                    raise AssertionError(f"surviving target unexpectedly OOV: {sample_id}:{position}")
                stage_rows[stage].append({
                    "sample_id": sample_id,
                    "position": position,
                    "target_rank": rank,
                    "target_minus_best_other_logit": margin,
                })
                ranks.append(rank)
                margins.append(margin)
            target_class = _coarse_class(target)
            selected_class = _coarse_class(picked)
            signatures[(target_class, selected_class)] += 1
            token_rows.append({
                "position": position,
                "target": target,
                "selected_top1": picked,
                "target_class": target_class,
                "selected_class": selected_class,
                "target_hwr_rank": int(prediction["target_rank"]),
                "target_probability": float(prediction["target_probability"]),
                "selected_probability": float(prediction["top1_probability"]),
                "raw_point_count": int(symbol["preprocessing"]["raw_point_count"]),
                "raw_bbox": symbol["preprocessing"]["raw_bbox"],
                "layer_target_ranks": ranks,
                "layer_target_minus_best_other_logits": margins,
                "first_stage_in_top5": next(
                    (stage for stage, rank in zip(LAYER_STAGES, ranks, strict=True) if rank <= 5),
                    None,
                ),
                "target_rank_improved_input_to_final": ranks[-1] < ranks[0],
                "target_margin_improved_input_to_final": margins[-1] > margins[0],
            })
            replay_checks += 1
        decomposition = formula["score_gap_decomposition"]
        score_gaps.append(float(decomposition["token_log_probability_component"]))
        total_score_gaps.append(float(decomposition["reconstructed_total"]))
        relation_gaps.append(float(decomposition["relation_component"]))
        details.append({
            "sample_id": sample_id,
            "target_tokens": targets,
            "selected_tokens": selected,
            "target_beam_rank": formula["target_beam_rank"],
            "target_to_selected_score_gap": float(decomposition["reconstructed_total"]),
            "token_log_probability_component": float(decomposition["token_log_probability_component"]),
            "relation_component": float(decomposition["relation_component"]),
            "mismatches": token_rows,
        })

    stage_summary = {}
    for stage in LAYER_STAGES:
        rows = stage_rows[stage]
        ranks = [row["target_rank"] for row in rows]
        margins = [row["target_minus_best_other_logit"] for row in rows]
        stage_summary[stage] = {
            "mismatch_glyphs": len(rows),
            "target_top1_count": sum(rank == 1 for rank in ranks),
            "target_top5_count": sum(rank <= 5 for rank in ranks),
            "mean_target_rank": sum(ranks) / len(ranks) if ranks else None,
            "median_target_rank": _median([float(rank) for rank in ranks]),
            "mean_target_minus_best_other_logit": sum(margins) / len(margins) if margins else None,
            "median_target_minus_best_other_logit": _median(margins),
        }

    verification = {
        "cause_matrix_schema_valid": cause.get("schema") == "aiflow-hwr-failure-cause-microscope/v1",
        "trace_hash_matches_cause_matrix": trace_sha == expected_trace_sha,
        "formula_count_is_149": len(traces) == 149,
        "survivor_count_is_11": len(survivors) == 11,
        "mismatch_count_is_12": mismatch_count == 12,
        "all_mismatches_replayed_against_hwr_trace": replay_checks == mismatch_count,
        "all_survivor_relations_contribute_zero": all(abs(value) <= 1e-12 for value in relation_gaps),
        "all_target_score_gaps_are_token_log_probability_only": all(
            abs(total - token_gap) <= 1e-9
            for total, token_gap in zip(total_score_gaps, [
                float(row["token_log_probability_component"])
                for row in details
            ], strict=True)
        ),
    }
    verification["all_checks_pass"] = all(verification.values())
    if not verification["all_checks_pass"]:
        raise AssertionError("decoder survivor layer microscope failed verification")
    return {
        "schema": SCHEMA,
        "scope": "frozen consumed-development forensic replay; no fitting, threshold selection, CROHME, or promotion",
        "inputs": {
            "trace_path": str(trace_path),
            "trace_sha256": trace_sha,
            "cause_matrix_path": str(cause_path),
            "cause_matrix_sha256": cause_sha,
            "formula_count": len(traces),
            "survivor_count": len(survivors),
            "mismatch_count": mismatch_count,
        },
        "findings": {
            "score_gap_source": "HWR token log probability only; relation component is zero for all survivors",
            "mean_formula_token_score_gap": sum(score_gaps) / len(score_gaps) if score_gaps else None,
            "coarse_class_confusions": [
                {"target_class": target, "selected_class": selected, "count": count}
                for (target, selected), count in sorted(signatures.items())
            ],
            "layer_summary": stage_summary,
            "target_rank_improved_input_to_final_count": sum(
                int(row["target_rank_improved_input_to_final"])
                for formula in details for row in formula["mismatches"]
            ),
            "target_margin_improved_input_to_final_count": sum(
                int(row["target_margin_improved_input_to_final"])
                for formula in details for row in formula["mismatches"]
            ),
            "details": details,
        },
        "verification": verification,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--cause-matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.trace, args.cause_matrix)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "all_checks_pass": report["verification"]["all_checks_pass"],
        "survivors": report["inputs"]["survivor_count"],
        "mismatches": report["inputs"]["mismatch_count"],
        "layer_summary": report["findings"]["layer_summary"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
