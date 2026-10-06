#!/usr/bin/env python3
"""Export a formula graph after a frozen character finalizer has run.

This boundary is deliberately one-way: layout may read immutable HWR boxes,
Top-k candidates, and an existing candidate-preserving finalized token, but it
cannot feed a changed order back into character recognition.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path

from character_tensor_v1 import _json_lines
from evaluate_48hz_prefix_v1 import _sha256
from formula_layout_v1 import LayoutConfig, finalize_formula_outputs


REPORT_SCHEMA = "aiflow-formula-layout-export/v1"


def _finalized_predictions(path: Path) -> dict[str, str]:
    predictions: dict[str, str] = {}
    for row in _json_lines(path):
        record_id = str(row.get("record_id", ""))
        token = str(row.get("finalized_top1", ""))
        if not record_id or not token or record_id in predictions:
            raise ValueError("finalized rows require unique record_id and finalized_top1")
        predictions[record_id] = token
    return predictions


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {"mode": "wt", "encoding": "utf-8", "newline": "\n"}
    stream_context = (
        gzip.open(path, **kwargs)
        if path.suffix.lower() == ".gz"
        else path.open(**kwargs)
    )
    with stream_context as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")


def _self_test() -> None:
    rows = [
        {
            "record_id": "x", "formula_id": "f",
            "final_topk": ["x"], "final_topk_probabilities": [1.0],
            "geometry": {"left": 0, "top": 10, "right": 20, "bottom": 40},
        },
        {
            "record_id": "2", "formula_id": "f",
            "final_topk": ["2"], "final_topk_probabilities": [1.0],
            "geometry": {"left": 21, "top": 0, "right": 29, "bottom": 14},
        },
    ]
    formulae, audit = finalize_formula_outputs(rows, {"x": "x", "2": "2"})
    assert formulae[0]["latex"] == "x^{2}"
    assert formulae[0]["latex_status"] == "shadow_single_parent_graph"
    assert audit["multi_parent_nodes"] == 0
    assert audit["candidate_preservation_rate"] == 1.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path)
    parser.add_argument("--finalized", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        print(json.dumps({"self_test": "pass"}))
        return 0
    if None in (args.candidates, args.finalized, args.output, args.report):
        parser.error("--candidates, --finalized, --output, and --report are required")

    candidates_path = args.candidates.expanduser().resolve()
    finalized_path = args.finalized.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    report_path = args.report.expanduser().resolve()
    if not candidates_path.is_file() or not finalized_path.is_file():
        parser.error("candidate and finalized inputs must exist")
    if output_path == report_path:
        parser.error("--output and --report must differ")
    if output_path.exists() or report_path.exists():
        parser.error("refusing to overwrite formula layout output or report")

    candidates = list(_json_lines(candidates_path))
    predictions = _finalized_predictions(finalized_path)
    formulae, audit = finalize_formula_outputs(candidates, predictions)
    _write_jsonl(output_path, formulae)
    report = {
        "schema": REPORT_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "shadow_runtime_only",
        "pipeline_position": "after frozen candidate-preserving character finalization",
        "feedback_into_character_model": False,
        "configuration": asdict(LayoutConfig()),
        "inputs": {
            "candidates": {"path": str(candidates_path), "sha256": _sha256(candidates_path)},
            "finalized": {"path": str(finalized_path), "sha256": _sha256(finalized_path)},
        },
        "output": {"path": str(output_path), "sha256": _sha256(output_path)},
        "audit": audit,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    print(json.dumps({
        "output": str(output_path), "output_sha256": _sha256(output_path),
        "report": str(report_path), "report_sha256": _sha256(report_path),
        "audit": audit,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
