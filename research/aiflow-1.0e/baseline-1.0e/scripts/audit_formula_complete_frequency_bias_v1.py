#!/usr/bin/env python3
"""Measure class-frequency bias after whole-formula context finalization.

This is a frozen CROHME validation audit.  It never trains, selects, or tunes.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from character_tensor_v1 import ROOT, _json_lines


SCHEMA = "aiflow-formula-complete-frequency-bias-audit/v1"
DEFAULT_CANDIDATES = (
    ROOT / "artifacts" / "crohme_standard_context_20260820_r1_research"
    / "test_candidates.jsonl.gz"
)
DEFAULT_FINALIZED = (
    ROOT / "artifacts" / "commercial_formula_complete_crohme_20260821_r1"
    / "formula_complete_predictions.jsonl.gz"
)
DEFAULT_OUTPUT = (
    ROOT / "artifacts" / "formula_complete_frequency_bias_20260821_r1"
    / "frequency_bias_report.json"
)


def _band(count: int) -> str:
    if count < 10:
        return "1-9"
    if count < 50:
        return "10-49"
    if count < 200:
        return "50-199"
    return "200+"


def _metrics(rows: list[dict]) -> dict:
    if not rows:
        return {"records": 0, "hwr_top1": None, "formula_complete_top1": None}
    baseline = np.asarray([row["hwr_top1"] == row["truth"] for row in rows])
    final = np.asarray([row["finalized_top1"] == row["truth"] for row in rows])
    return {
        "records": len(rows),
        "hwr_top1": float(baseline.mean()),
        "formula_complete_top1": float(final.mean()),
        "delta": float(final.mean() - baseline.mean()),
        "improved": int(np.sum(~baseline & final)),
        "regressed": int(np.sum(baseline & ~final)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--finalized", type=Path, default=DEFAULT_FINALIZED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    paths = [args.candidates.resolve(), args.finalized.resolve(), args.output.resolve()]
    if any(path.drive.upper() != "D:" for path in paths):
        parser.error("all inputs and outputs must remain on D:")
    if args.output.exists() or any(not path.is_file() for path in paths[:-1]):
        parser.error("inputs must exist and output must be new")

    truth_rows = {str(row["record_id"]): row for row in _json_lines(args.candidates)}
    finalized = {str(row["record_id"]): row for row in _json_lines(args.finalized)}
    if set(truth_rows) != set(finalized) or len(truth_rows) != 11991:
        raise ValueError("frozen formula-complete validation coverage mismatch")
    rows = [{
        "record_id": record_id,
        "truth": str(source["label"]),
        "hwr_top1": str(finalized[record_id]["hwr_top1"]),
        "finalized_top1": str(finalized[record_id]["finalized_top1"]),
    } for record_id, source in truth_rows.items()]
    truth_frequency = Counter(row["truth"] for row in rows)
    baseline_frequency = Counter(row["hwr_top1"] for row in rows)
    final_frequency = Counter(row["finalized_top1"] for row in rows)

    by_band: dict[str, list[dict]] = defaultdict(list)
    by_label: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_band[_band(truth_frequency[row["truth"]])].append(row)
        by_label[row["truth"]].append(row)
    regressions = [
        row for row in rows
        if row["hwr_top1"] == row["truth"] and row["finalized_top1"] != row["truth"]
    ]
    regression_to_more_frequent = [
        row for row in regressions
        if truth_frequency[row["finalized_top1"]] > truth_frequency[row["truth"]]
    ]
    improvements = [
        row for row in rows
        if row["hwr_top1"] != row["truth"] and row["finalized_top1"] == row["truth"]
    ]
    labels = []
    for label, selected in sorted(by_label.items(), key=lambda value: (-len(value[1]), value[0])):
        metric = _metrics(selected)
        labels.append({
            "label": label,
            "truth_frequency": truth_frequency[label],
            "hwr_prediction_frequency": baseline_frequency[label],
            "formula_complete_prediction_frequency": final_frequency[label],
            "hwr_overprediction": baseline_frequency[label] - truth_frequency[label],
            "formula_complete_overprediction": final_frequency[label] - truth_frequency[label],
            **metric,
        })

    def overpredicted(counter: Counter) -> list[dict]:
        values = []
        for label in set(counter) | set(truth_frequency):
            excess = counter[label] - truth_frequency[label]
            if excess > 0:
                values.append({
                    "label": label, "truth_count": truth_frequency[label],
                    "prediction_count": counter[label], "excess": excess,
                })
        return sorted(values, key=lambda row: (-row["excess"], row["label"]))[:25]

    report = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "noncommercial_validation_only",
        "training_performed": False,
        "selection_performed": False,
        "crohme_gradient_updates": 0,
        "prefix_or_streaming_accuracy_scored": False,
        "decision_boundary": "formula_complete whole-formula only",
        "overall": _metrics(rows),
        "by_truth_frequency_band": {
            key: _metrics(value) for key, value in sorted(by_band.items())
        },
        "context_changes": {
            "improved": len(improvements),
            "regressed": len(regressions),
            "net_correct_gain": len(improvements) - len(regressions),
            "regressions_to_more_frequent_truth_token": len(regression_to_more_frequent),
            "regressions_to_more_frequent_truth_token_rate": (
                len(regression_to_more_frequent) / len(regressions) if regressions else 0.0
            ),
        },
        "overprediction": {
            "hwr_top1": overpredicted(baseline_frequency),
            "formula_complete": overpredicted(final_frequency),
            "absolute_excess_mass_hwr": sum(
                max(0, baseline_frequency[label] - truth_frequency[label])
                for label in set(baseline_frequency) | set(truth_frequency)
            ),
            "absolute_excess_mass_formula_complete": sum(
                max(0, final_frequency[label] - truth_frequency[label])
                for label in set(final_frequency) | set(truth_frequency)
            ),
        },
        "by_label": labels,
        "interpretation": (
            "Whole-formula context improves aggregate accuracy, but any remaining "
            "regressions toward more frequent tokens are retained as a measured risk; "
            "CROHME is not used to retune the model."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output),
        "overall": report["overall"],
        "context_changes": report["context_changes"],
        "excess_mass": report["overprediction"],
        "crohme_gradient_updates": 0,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
