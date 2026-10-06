#!/usr/bin/env python3
"""Evaluate a fail-closed guard for the external TrOCR candidate adapter.

The guard may replace the existing HWR Top-1 token only when the adapter's
candidate is already present and its softmax confidence and margin clear fixed
thresholds.  Ground-truth fields are used only for evaluation metrics; they
never participate in the runtime decision.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import io
import json
import math
from pathlib import Path
from typing import Iterable


def read_jsonl_gz(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl_gz(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
            with io.TextIOWrapper(gz, encoding="utf-8", newline="\n") as text:
                for row in rows:
                    text.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
                    text.write("\n")


def annotate(row: dict) -> dict:
    scores = [float(value) for value in row["adapter_scores"]]
    peak = max(scores)
    exp_scores = [math.exp(value - peak) for value in scores]
    denominator = sum(exp_scores)
    probabilities = [value / denominator for value in exp_scores]
    ordered = sorted(probabilities, reverse=True)
    adapter_token = str(row["adapter_token"])
    candidates = [str(value) for value in row["candidates"]]
    return {
        **row,
        "adapter_confidence": ordered[0],
        "adapter_margin": ordered[0] - ordered[1] if len(ordered) > 1 else ordered[0],
        "adapter_candidate_eligible": adapter_token in candidates,
    }


def formula_exact(rows: list[dict], predictions: list[str]) -> tuple[int, int, float]:
    grouped: dict[tuple[str, str], list[bool]] = collections.defaultdict(list)
    for row, prediction in zip(rows, predictions):
        grouped[(str(row["writer_group"]), str(row["formula_id"]))].append(prediction == str(row["label"]))
    total = len(grouped)
    exact = sum(all(values) for values in grouped.values())
    return exact, total, exact / total if total else 0.0


def metrics(rows: list[dict], predictions: list[str], baseline: list[str]) -> dict:
    correct = [prediction == str(row["label"]) for row, prediction in zip(rows, predictions)]
    baseline_correct = [prediction == str(row["label"]) for row, prediction in zip(rows, baseline)]
    exact, total, exact_rate = formula_exact(rows, predictions)
    return {
        "rows": len(rows),
        "top1_correct": sum(correct),
        "top1": sum(correct) / len(rows) if rows else 0.0,
        "formula_exact": exact,
        "formula_total": total,
        "formula_exact_rate": exact_rate,
        "row_level_improvements": sum(not old and new for old, new in zip(baseline_correct, correct)),
        "row_level_regressions": sum(old and not new for old, new in zip(baseline_correct, correct)),
        "changed_rows": sum(old != new for old, new in zip(baseline, predictions)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confidence-threshold", type=float, default=0.30)
    parser.add_argument("--margin-threshold", type=float, default=0.30)
    args = parser.parse_args()
    if not 0.0 <= args.confidence_threshold <= 1.0:
        raise SystemExit("--confidence-threshold must be in [0, 1]")
    if not 0.0 <= args.margin_threshold <= 1.0:
        raise SystemExit("--margin-threshold must be in [0, 1]")

    rows = [annotate(row) for row in read_jsonl_gz(args.predictions)]
    baseline = [str(row["baseline_token"]) for row in rows]
    guarded_rows = []
    guarded_predictions = []
    for row in rows:
        eligible = bool(row["adapter_candidate_eligible"])
        accepted = (
            eligible
            and float(row["adapter_confidence"]) >= args.confidence_threshold
            and float(row["adapter_margin"]) >= args.margin_threshold
        )
        prediction = str(row["adapter_token"]) if accepted else str(row["baseline_token"])
        guarded_predictions.append(prediction)
        guarded_rows.append(
            {
                **row,
                "guard_policy": "adapter_if_candidate_present_and_confidence_margin_pass_else_baseline",
                "guard_accepted": accepted,
                "guarded_token": prediction,
            }
        )

    args.output.mkdir(parents=True, exist_ok=True)
    write_jsonl_gz(args.output / "guarded_predictions.jsonl.gz", guarded_rows)
    raw_predictions = [
        str(row["adapter_token"]) if bool(row["adapter_candidate_eligible"]) else str(row["baseline_token"])
        for row in rows
    ]
    candidate_absent = sum(not bool(row.get("target_in_candidates", False)) for row in rows)
    contract_violations = sum(not bool(row["adapter_candidate_eligible"]) for row in rows if str(row["adapter_token"]) != str(row["baseline_token"]))
    report = {
        "schema": "aiflow-1.0e-trocr-candidate-guard-eval/v1",
        "status": "shadow_only",
        "training_performed": False,
        "input_predictions": str(args.predictions),
        "rows": len(rows),
        "writers": sorted({str(row["writer_group"]) for row in rows}),
        "confidence_threshold": args.confidence_threshold,
        "margin_threshold": args.margin_threshold,
        "runtime_ground_truth_used": False,
        "candidate_contract": "existing HWR Top-k only; no candidate creation",
        "candidate_absent_records_for_evaluation": candidate_absent,
        "candidate_contract_violations": contract_violations,
        "baseline": metrics(rows, baseline, baseline),
        "raw_adapter": metrics(rows, raw_predictions, baseline),
        "guarded": metrics(rows, guarded_predictions, baseline),
        "commercial_interpretation": {
            "row_level_regression_gate": metrics(rows, guarded_predictions, baseline)["row_level_regressions"] == 0,
            "fresh_acceptance_required": True,
            "posthoc_threshold_calibration_not_sufficient_for_adoption": True,
        },
    }
    (args.output / "guard_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
