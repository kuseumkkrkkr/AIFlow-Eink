#!/usr/bin/env python3
"""Verify formula-complete gating against the frozen owned 110-formula output."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from character_tensor_v1 import ROOT
from formula_complete_gate_v1 import (
    DEFAULT_CONTEXT, DEFAULT_HWR, DEFAULT_RANKER, FormulaCompleteSessionV1,
)
from raw_formula_context_runtime_v1 import RawFormulaContextRuntimeV1


SCHEMA = "aiflow-formula-complete-owned-nonregression/v1"
DEFAULT_FORMULAS = ROOT / "hf-dataset" / "data" / "formulas_valid.jsonl"
DEFAULT_BASELINE = (
    ROOT / "artifacts" / "context_detection_runtime_smoke_20260820_r1_shadow"
    / "raw_runtime_output.json"
)
DEFAULT_OUTPUT = (
    ROOT / "artifacts" / "formula_complete_owned_nonregression_20260821_r1"
    / "nonregression_report.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _projection(result: dict) -> dict:
    return {
        "groups": [
            {
                "record_id": str(row["record_id"]),
                "stroke_indices": [int(value) for value in row["stroke_indices"]],
            }
            for row in result["groups"]
        ],
        "symbols": [
            {
                "record_id": str(row["record_id"]),
                "stroke_indices": [int(value) for value in row["stroke_indices"]],
                "hwr_top1": str(row["hwr_top1"]),
                "hwr_topk": [str(value) for value in row["hwr_topk"]],
                "finalized_top1": str(row["finalized_top1"]),
            }
            for row in result["symbols"]
        ],
        "hwr_top1_tokens": [str(value) for value in result["hwr_top1_tokens"]],
        "partition_context_tokens": [
            str(value) for value in result["partition_context_tokens"]
        ],
        "finalized_tokens": [str(value) for value in result["finalized_tokens"]],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formulas", type=Path, default=DEFAULT_FORMULAS)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--baseline-ranker", type=Path, default=DEFAULT_RANKER)
    parser.add_argument("--partition-ranker", type=Path, default=DEFAULT_RANKER)
    parser.add_argument("--hwr-checkpoint", type=Path, default=DEFAULT_HWR)
    parser.add_argument("--context-checkpoint", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    paths = [
        args.formulas.resolve(), args.baseline.resolve(), args.output.resolve(),
        args.baseline_ranker.resolve(), args.partition_ranker.resolve(),
        args.hwr_checkpoint.resolve(), args.context_checkpoint.resolve(),
    ]
    if any(path.drive.upper() != "D:" for path in paths):
        parser.error("all inputs and outputs must remain on D:")
    if not args.formulas.is_file() or not args.baseline.is_file() or args.output.exists():
        parser.error("inputs must exist and output must be new")
    formulas = [
        json.loads(line) for line in args.formulas.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    baseline_payload = json.loads(args.baseline.read_text(encoding="utf-8"))
    baseline = {
        str(row["formula_id"]): row for row in baseline_payload.get("formulas") or []
    }
    formula_ids = [str(row["sample_id"]) for row in formulas]
    if len(formulas) != 110 or set(formula_ids) != set(baseline):
        raise ValueError("owned 110-formula baseline coverage mismatch")
    expected_hashes = {
        "partition_ranker_sha256": _sha256(args.baseline_ranker),
        "hwr_checkpoint_sha256": _sha256(args.hwr_checkpoint),
        "context_checkpoint_sha256": _sha256(args.context_checkpoint),
    }
    for row in baseline.values():
        audit = row.get("audit", {})
        if any(str(audit.get(key)) != value for key, value in expected_hashes.items()):
            raise ValueError("owned baseline artifact hash contract mismatch")

    runtime = RawFormulaContextRuntimeV1.from_artifacts(
        args.partition_ranker, args.hwr_checkpoint, args.context_checkpoint,
        device=args.device, emit_formula_layout_shadow=True,
        allow_posthoc_shadow=True,
    )
    mismatches = []
    pre_complete_commits = 0
    stroke_loss_or_duplication = 0
    raw_fallback_mismatches = 0
    product_auto_commits = 0
    for index, source in enumerate(formulas, 1):
        formula_id = str(source["sample_id"])
        session = FormulaCompleteSessionV1(formula_id, runtime.infer)
        for stroke in source["strokes"]:
            event = session.append_stroke(stroke)
            pre_complete_commits += bool(event.get("committed"))
            pre_complete_commits += bool(event.get("finalized_tokens"))
            pre_complete_commits += bool(event.get("accuracy_scored"))
        current = session.complete()
        raw_fallback_mismatches += (
            current.get("raw_fallback", {}).get("strokes") != source["strokes"]
        )
        product_auto_commits += bool(
            current.get("product_decision", {}).get("product_auto_committed")
        )
        assigned = [
            int(value) for group in current["groups"] for value in group["stroke_indices"]
        ]
        if (
            sorted(assigned) != list(range(len(source["strokes"])))
            or len(assigned) != len(set(assigned))
        ):
            stroke_loss_or_duplication += 1
        before = _projection(baseline[formula_id])
        after = _projection(current)
        if before != after:
            mismatches.append({
                "formula_id": formula_id,
                "grouping_changed": before["groups"] != after["groups"],
                "hwr_changed": before["hwr_top1_tokens"] != after["hwr_top1_tokens"],
                "context_changed": (
                    before["partition_context_tokens"]
                    != after["partition_context_tokens"]
                ),
                "final_changed": before["finalized_tokens"] != after["finalized_tokens"],
                "before": before,
                "after": after,
            })
        if index % 25 == 0 or index == len(formulas):
            print(json.dumps({
                "event": "owned_nonregression_progress", "completed": index,
                "total": len(formulas), "mismatches": len(mismatches),
            }, ensure_ascii=False), flush=True)
    if (
        pre_complete_commits or stroke_loss_or_duplication
        or raw_fallback_mismatches or product_auto_commits
    ):
        raise AssertionError("formula-complete gate integrity regression")
    report = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "project_owned_nonregression_pass" if not mismatches else "regression",
        "formulas": len(formulas),
        "baseline_projection_mismatches": len(mismatches),
        "grouping_mismatches": sum(row["grouping_changed"] for row in mismatches),
        "hwr_mismatches": sum(row["hwr_changed"] for row in mismatches),
        "context_mismatches": sum(row["context_changed"] for row in mismatches),
        "final_token_mismatches": sum(row["final_changed"] for row in mismatches),
        "mismatches": mismatches,
        "pre_complete_commits": pre_complete_commits,
        "prefix_accuracy_scored": False,
        "stroke_loss_or_duplication_formulas": stroke_loss_or_duplication,
        "raw_fallback_mismatches": raw_fallback_mismatches,
        "product_auto_commits": product_auto_commits,
        "product_decision": "REVIEW_REQUIRED",
        "formula_complete_output_additions": [
            "formula_complete_gate", "semantic_composition", "formula_layout_shadow",
            "product_decision", "raw_fallback",
        ],
        "product_default_enabled": False,
        "posthoc_shadow_opt_in": True,
        "inputs": {
            "formulas_sha256": _sha256(args.formulas),
            "baseline_sha256": _sha256(args.baseline),
            **expected_hashes,
            "candidate_partition_ranker_sha256": _sha256(args.partition_ranker),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "event": "owned_nonregression_finished", "output": str(args.output),
        "formulas": len(formulas), "mismatches": len(mismatches),
        "pre_complete_commits": pre_complete_commits,
        "stroke_loss_or_duplication": stroke_loss_or_duplication,
        "raw_fallback_mismatches": raw_fallback_mismatches,
        "product_auto_commits": product_auto_commits,
    }, ensure_ascii=False))
    return 0 if not mismatches else 1


if __name__ == "__main__":
    raise SystemExit(main())
