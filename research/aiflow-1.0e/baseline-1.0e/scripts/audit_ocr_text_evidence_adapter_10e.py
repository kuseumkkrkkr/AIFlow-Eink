#!/usr/bin/env python3
"""Audit the completed AIFlow 1.0e external-text shadow artifact.

This audit does not load or alter a model.  It recomputes full-formula Exact
including candidate-absent rows and verifies that the adapter selected only an
existing HWR Top-k candidate.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT = ROOT / "artifacts" / "ocr_text_evidence_adapter_10e_20260830"


def _read_predictions(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _full_exact(predictions: list[dict], selected_key: str) -> dict:
    by_formula: dict[str, list[dict]] = defaultdict(list)
    for row in predictions:
        by_formula[str(row["formula_id"])].append(row)
    exact_ids = []
    for formula_id, rows in sorted(by_formula.items()):
        if all(row["target_in_candidates"] and row[selected_key] == row["label"] for row in rows):
            exact_ids.append(formula_id)
    total = len(by_formula)
    return {
        "exact_formulas": len(exact_ids),
        "total_formulas": total,
        "rate": len(exact_ids) / total if total else 0.0,
        "formula_ids": exact_ids,
    }


def _teacher_exact(evaluation: dict, predictions: list[dict]) -> dict | None:
    if "teacher_outputs" not in evaluation:
        return None
    by_formula: dict[str, list[str]] = defaultdict(list)
    for row in predictions:
        by_formula[str(row["formula_id"])].append(str(row["label"]))
    exact_ids = []
    for formula_id, expected in sorted(evaluation["teacher_outputs"].items()):
        if by_formula.get(formula_id) == expected["tokens"]:
            exact_ids.append(formula_id)
    total = len(evaluation["teacher_outputs"])
    return {
        "exact_formulas": len(exact_ids),
        "total_formulas": total,
        "rate": len(exact_ids) / total if total else 0.0,
        "formula_ids": exact_ids,
    }


def _folds(predictions: list[dict]) -> list[dict]:
    by_writer: dict[str, list[dict]] = defaultdict(list)
    for row in predictions:
        by_writer[str(row["writer_group"])].append(row)
    output = []
    for writer, rows in sorted(by_writer.items()):
        covered = [row for row in rows if row["target_in_candidates"]]
        output.append({
            "held_writer": writer,
            "records": len(rows),
            "covered_records": len(covered),
            "candidate_recall": len(covered) / len(rows) if rows else 0.0,
            "baseline_top1_covered": sum(row["baseline_token"] == row["label"] for row in covered) / len(covered) if covered else 0.0,
            "adapter_top1_covered": sum(row["adapter_token"] == row["label"] for row in covered) / len(covered) if covered else 0.0,
            "baseline_full_formula_exact": _full_exact(rows, "baseline_token"),
            "adapter_full_formula_exact": _full_exact(rows, "adapter_token"),
        })
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    args = parser.parse_args()
    evaluation_path = args.artifact / "evaluation.json"
    predictions_path = args.artifact / "writer_loo_predictions.jsonl.gz"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    predictions = _read_predictions(predictions_path)
    contract_violations = [
        row["record_id"]
        for row in predictions
        if row["adapter_token"] not in row["candidates"]
    ]
    candidate_absent = [row["record_id"] for row in predictions if not row["target_in_candidates"]]
    audit = {
        "schema": "aiflow-1.0e-ocr-text-evidence-adapter-audit/v1",
        "artifact": str(args.artifact),
        "external_model": evaluation["model"],
        "training": evaluation["training"],
        "aggregate_reported": evaluation["aggregate"],
        "aggregate_full_exact": {
            "baseline": _full_exact(predictions, "baseline_token"),
            "adapter": _full_exact(predictions, "adapter_token"),
            "candidate_absent_records": len(candidate_absent),
            "candidate_absent_record_ids": candidate_absent,
        },
        "teacher_direct_exact": _teacher_exact(evaluation, predictions),
        "writer_loo_full_exact": _folds(predictions),
        "candidate_contract": {
            "violations": len(contract_violations),
            "violation_record_ids": contract_violations,
            "top_k_only_verified": not contract_violations,
        },
        "status": "shadow_only",
    }
    output = args.artifact / "audit.json"
    output.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "baseline_full_exact": audit["aggregate_full_exact"]["baseline"]["rate"],
        "adapter_full_exact": audit["aggregate_full_exact"]["adapter"]["rate"],
        "teacher_direct_exact": audit["teacher_direct_exact"]["rate"] if audit["teacher_direct_exact"] else None,
        "candidate_absent_records": len(candidate_absent),
        "candidate_contract_violations": len(contract_violations),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
