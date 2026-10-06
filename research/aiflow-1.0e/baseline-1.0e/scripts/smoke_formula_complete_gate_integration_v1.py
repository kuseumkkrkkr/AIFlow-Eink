#!/usr/bin/env python3
"""Run the formula-complete gate against the real frozen runtime once."""

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


SCHEMA = "aiflow-formula-complete-gate-integration-smoke/v1"
DEFAULT_FORMULAS = ROOT / "hf-dataset" / "data" / "formulas_valid.jsonl"
DEFAULT_OUTPUT = (
    ROOT / "artifacts" / "formula_complete_gate_20260821_r1_smoke"
    / "integration_report.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formulas", type=Path, default=DEFAULT_FORMULAS)
    parser.add_argument("--sample-id", default="aiflow_0001")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    paths = [
        args.formulas.resolve(), args.output.resolve(), DEFAULT_RANKER.resolve(),
        DEFAULT_HWR.resolve(), DEFAULT_CONTEXT.resolve(),
    ]
    if any(path.drive.upper() != "D:" for path in paths):
        parser.error("all inputs and outputs must remain on D:")
    if not args.formulas.is_file() or args.output.exists():
        parser.error("formula input must exist and output must be new")
    selected = None
    with args.formulas.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if str(row.get("sample_id")) == args.sample_id:
                selected = row
                break
    if selected is None:
        raise ValueError(f"sample not found: {args.sample_id}")
    runtime = RawFormulaContextRuntimeV1.from_artifacts(
        DEFAULT_RANKER, DEFAULT_HWR, DEFAULT_CONTEXT,
        device=args.device, emit_formula_layout_shadow=True,
        allow_posthoc_shadow=True,
    )
    session = FormulaCompleteSessionV1(args.sample_id, runtime.infer)
    stroke_events = [session.append_stroke(stroke) for stroke in selected["strokes"]]
    if any(
        event.get("committed") is not False
        or event.get("finalized_tokens") != []
        or event.get("accuracy_scored") is not False
        for event in stroke_events
    ):
        raise AssertionError("a pre-complete stroke event committed a result")
    result = session.complete()
    replay = session.complete()
    gate = result["formula_complete_gate"]
    if (
        gate.get("pre_complete_commits") != 0
        or gate.get("prefix_accuracy_scored") is not False
        or gate.get("committed") is not True
        or gate.get("product_auto_committed") is not False
        or replay.get("idempotent_replay") is not True
    ):
        raise AssertionError("formula-complete commit contract failed")
    selected_indices = [
        int(value) for group in result["groups"] for value in group["stroke_indices"]
    ]
    expected_indices = list(range(len(selected["strokes"])))
    if sorted(selected_indices) != expected_indices or len(selected_indices) != len(set(selected_indices)):
        raise AssertionError("selected grouping lost or duplicated strokes")
    if result["raw_fallback"]["strokes"] != selected["strokes"]:
        raise AssertionError("raw fallback differs from the submitted strokes")
    target_tokens = [str(row["token"]) for row in selected.get("target_cells") or []]
    report = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "project_owned_runtime_smoke_shadow",
        "sample_id": args.sample_id,
        "stroke_events": len(stroke_events),
        "pre_complete_commits": 0,
        "prefix_accuracy_scored": False,
        "formula_complete_commits": 1,
        "duplicate_complete_idempotent": True,
        "all_input_strokes_exactly_once": True,
        "raw_fallback_exact": True,
        "raw_fallback_sha256": result["raw_fallback"]["canonical_sha256"],
        "group_count": len(result["groups"]),
        "target_tokens_used_by_inference": False,
        "target_tokens": target_tokens,
        "finalized_tokens": result["finalized_tokens"],
        "token_exact_smoke": result["finalized_tokens"] == target_tokens,
        "semantic_composition": result["semantic_composition"],
        "layout": result.get("formula_layout_shadow"),
        "product_default_enabled": False,
        "product_decision": result["product_decision"],
        "posthoc_shadow_opt_in": True,
        "inputs": {
            "formulas_sha256": _sha256(args.formulas),
            "partition_ranker_sha256": _sha256(DEFAULT_RANKER),
            "hwr_checkpoint_sha256": _sha256(DEFAULT_HWR),
            "context_checkpoint_sha256": _sha256(DEFAULT_CONTEXT),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output),
        "sample_id": args.sample_id,
        "pre_complete_commits": 0,
        "all_input_strokes_exactly_once": True,
        "target": target_tokens,
        "prediction": result["finalized_tokens"],
        "token_exact_smoke": report["token_exact_smoke"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
