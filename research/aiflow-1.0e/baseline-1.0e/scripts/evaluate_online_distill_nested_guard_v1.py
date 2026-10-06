#!/usr/bin/env python3
"""Nested writer guard for the raster-free online candidate distillation trial.

Thresholds are selected on writers outside the held writer.  The runtime
decision uses only the candidate cache, adapter confidence, and adapter
margin; labels are used only inside the outer evaluation for calibration and
reporting.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from evaluate_trocr_candidate_guard_v1 import (
    annotate,
    formula_exact,
    metrics,
    read_jsonl_gz,
    write_jsonl_gz,
)
from accuracy_upgrade_contract_v1 import writer_key
from accuracy_lineage_10e import validate_prediction


CONFIDENCE_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)
MARGIN_GRID = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


def apply_thresholds(rows: list[dict], confidence: float, margin: float) -> tuple[list[str], list[bool]]:
    predictions: list[str] = []
    accepted_rows: list[bool] = []
    for row in rows:
        accepted = (
            bool(row["adapter_candidate_eligible"])
            and float(row["adapter_confidence"]) >= confidence
            and float(row["adapter_margin"]) >= margin
        )
        accepted_rows.append(accepted)
        predictions.append(str(row["adapter_token"]) if accepted else str(row["baseline_token"]))
    return predictions, accepted_rows


def select_threshold(train_rows: list[dict]) -> tuple[float, float, dict]:
    baseline = [str(row["baseline_token"]) for row in train_rows]
    best: tuple[tuple[float, float, int, float, float], float, float, dict] | None = None
    for confidence in CONFIDENCE_GRID:
        for margin in MARGIN_GRID:
            predictions, _ = apply_thresholds(train_rows, confidence, margin)
            score = metrics(train_rows, predictions, baseline)
            if score["row_level_regressions"] != 0:
                continue
            # Prefer full-formula correctness, then row correctness, then the
            # least number of changes.  Thresholds are the final deterministic
            # tie-break, favouring stricter acceptance.
            key = (
                float(score["formula_exact_rate"]),
                float(score["top1"]),
                -int(score["changed_rows"]),
                float(confidence),
                float(margin),
            )
            if best is None or key > best[0]:
                best = (key, confidence, margin, score)
    if best is None:
        return 1.0, 1.0, metrics(train_rows, baseline, baseline)
    return best[1], best[2], best[3]


def _provenance_excludes(row: dict, writer: str) -> bool:
    """예측을 만든 학습 집합이 해당 writer를 제외했는지 확인한다."""
    provenance = row.get("prediction_provenance") or {}
    excluded = {str(value) for value in provenance.get("excluded_writers") or []}
    return str(writer) in excluded


def _strict_inner_rows(rows: list[dict], inner_rows: list[dict], outer_writer: str) -> list[dict]:
    """outer held writer까지 제외한 inner-OOF calibration row를 검증한다."""
    train_ids = {
        str(row["record_id"])
        for row in rows
        if str(row["writer_group"]) != str(outer_writer)
    }
    selected = [
        row for row in inner_rows
        if str(row.get("record_id", "")) in train_ids
        and _provenance_excludes(row, outer_writer)
    ]
    selected_ids = [str(row["record_id"]) for row in selected]
    references = {str(row["record_id"]): row for row in rows}
    for row in selected:
        reference = references[str(row["record_id"])]
        if writer_key(row) != writer_key(reference) or str(row["formula_id"]) != str(reference["formula_id"]):
            raise ValueError("inner prediction identity mismatch")
        validate_prediction(row, outer_writer)
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError(f"strict nested calibration has duplicate record IDs for outer writer {outer_writer}")
    if set(selected_ids) != train_ids:
        missing = sorted(train_ids - set(selected_ids))
        raise ValueError(
            "strict nested calibration requires one inner-OOF row per outer-train record; "
            f"outer={outer_writer}, missing={missing[:5]}, count={len(missing)}"
        )
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--inner-predictions",
        type=Path,
        help="inner-OOF predictions whose models also exclude each outer held writer",
    )
    parser.add_argument(
        "--strict-nested",
        action="store_true",
        help="reject legacy threshold calibration without outer-held exclusion provenance",
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")

    rows = [annotate(row) for row in read_jsonl_gz(args.predictions)]
    inner_rows = []
    if args.strict_nested:
        if args.inner_predictions is None:
            parser.error("--strict-nested requires --inner-predictions")
        inner_rows = [annotate(row) for row in read_jsonl_gz(args.inner_predictions)]
    writers = sorted({str(row["writer_group"]) for row in rows})
    baseline = [str(row["baseline_token"]) for row in rows]
    held_predictions = [str(row["baseline_token"]) for row in rows]
    held_acceptance: dict[str, int] = collections.Counter()
    fold_reports: list[dict] = []

    for held_writer in writers:
        train_rows = [row for row in rows if str(row["writer_group"]) != held_writer]
        held_indices = [index for index, row in enumerate(rows) if str(row["writer_group"]) == held_writer]
        if args.strict_nested:
            for index in held_indices:
                validate_prediction(rows[index], held_writer)
        calibration_rows = (
            _strict_inner_rows(rows, inner_rows, held_writer)
            if args.strict_nested else train_rows
        )
        confidence, margin, train_selection = select_threshold(calibration_rows)
        if args.strict_nested and any(not _provenance_excludes(row, held_writer) for row in calibration_rows):
            raise ValueError(f"strict nested provenance failed for outer writer {held_writer}")
        predictions, accepted = apply_thresholds([rows[index] for index in held_indices], confidence, margin)
        for index, prediction, is_accepted in zip(held_indices, predictions, accepted):
            held_predictions[index] = prediction
            if is_accepted:
                held_acceptance[held_writer] += 1
        held_rows = [rows[index] for index in held_indices]
        fold_reports.append(
            {
                "held_writer": held_writer,
                "train_writers": [writer for writer in writers if writer != held_writer],
                "calibration_scope": "inner_oof_excludes_outer_writer" if args.strict_nested else "legacy_outer_train_predictions",
                "confidence_threshold": confidence,
                "margin_threshold": margin,
                "train_selection": train_selection,
                "held": metrics(held_rows, predictions, [str(row["baseline_token"]) for row in held_rows]),
                "held_changed_rows": sum(prediction != str(row["baseline_token"]) for row, prediction in zip(held_rows, predictions)),
                "held_accepted_rows": sum(accepted),
            }
        )

    output_rows = []
    for row, prediction in zip(rows, held_predictions):
        fold = next(item for item in fold_reports if item["held_writer"] == str(row["writer_group"]))
        accepted = prediction != str(row["baseline_token"])
        output_rows.append(
            {
                **row,
                "nested_guard_policy": "writer_loo_threshold_selected_on_other_writers",
                "nested_guard_confidence_threshold": fold["confidence_threshold"],
                "nested_guard_margin_threshold": fold["margin_threshold"],
                "nested_guard_accepted": accepted,
                "nested_guarded_token": prediction,
            }
        )

    raw_predictions = [
        str(row["adapter_token"]) if bool(row["adapter_candidate_eligible"]) else str(row["baseline_token"])
        for row in rows
    ]
    exact, total, exact_rate = formula_exact(rows, held_predictions)
    result = metrics(rows, held_predictions, baseline)
    contract_violations = sum(
        prediction != str(row["baseline_token"])
        and str(row["nested_guarded_token"]) not in [str(value) for value in row["candidates"]]
        for row in output_rows
        for prediction in [str(row["nested_guarded_token"])]
    )
    report = {
        "schema": "aiflow-1.0e-online-candidate-distill-nested-guard/v1",
        "status": "development" if args.strict_nested else "legacy_development",
        "training_performed": False,
        "strict_nested": bool(args.strict_nested),
        "input_predictions": str(args.predictions),
        "rows": len(rows),
        "writers": writers,
        "runtime_ground_truth_used": False,
        "candidate_contract": "existing HWR Top-k only; no candidate creation",
        "candidate_contract_violations": contract_violations,
        "baseline": metrics(rows, baseline, baseline),
        "raw_adapter": metrics(rows, raw_predictions, baseline),
        "nested_guarded": {
            **result,
            "formula_exact": exact,
            "formula_total": total,
            "formula_exact_rate": exact_rate,
        },
        "row_level_regression_gate": result["row_level_regressions"] == 0,
        "folds": fold_reports,
        "selection_rule": "among thresholds with zero calibration-row regressions, maximize formula exact, then Top-1, then minimize changed rows",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    write_jsonl_gz(args.output / "nested_guarded_predictions.jsonl.gz", output_rows)
    (args.output / "nested_guard_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
