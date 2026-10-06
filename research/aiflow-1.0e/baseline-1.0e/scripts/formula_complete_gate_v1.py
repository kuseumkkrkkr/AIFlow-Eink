#!/usr/bin/env python3
"""Gate formula commitment until the client emits ``formula_complete``.

Stroke events never return a finalized token. Optional previews expose only
provisional HWR candidates. The complete event runs the existing whole-formula
global partition, frozen HWR Top-5, candidate-preserving context, and 2D layout
pipeline exactly once; duplicate complete events are idempotent replays.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

from character_tensor_v1 import ROOT
from formula_semantic_lexicon_v1 import compose_function_spans


SCHEMA = "aiflow-formula-complete-gate/v1"
EVENT_SCHEMA = "aiflow-formula-complete-event/v1"
RAW_FALLBACK_SCHEMA = "aiflow-raw-ink-fallback/v1"
DEFAULT_RANKER = (
    ROOT / "artifacts" / "partition_context_ranker_expanded_hwr_20260820_r1_shadow"
    / "partition_context_ranker.joblib"
)
DEFAULT_HWR = Path(
    r"D:\AIFlow-Workspace\PrivateData\math-ink-data-collector\derived"
    r"\hwr-head-calibration-20260819-r1\final_all_writers_steps250_lr1e-3"
    r"\project_symbol_head_checkpoint.pt"
)
DEFAULT_CONTEXT = (
    ROOT / "artifacts" / "owned_formula_context_expanded_hwr_20260820_r1_shadow"
    / "owned_formula_context_product.pt"
)


def _stroke(value: dict[str, Any], expected_order: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("stroke event requires an object")
    order = int(value.get("order", expected_order))
    if order != expected_order:
        raise ValueError(
            f"stroke order must be contiguous: expected {expected_order}, got {order}"
        )
    points = value.get("points") or []
    if not isinstance(points, list) or not points:
        raise ValueError("stroke requires at least one point")
    return deepcopy({**value, "order": order, "points": points})


def _raw_fallback(formula_id: str, strokes: list[dict[str, Any]]) -> dict[str, Any]:
    preserved = deepcopy(strokes)
    canonical = json.dumps(
        preserved, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "schema": RAW_FALLBACK_SCHEMA,
        "formula_id": formula_id,
        "strokes": preserved,
        "stroke_count": len(preserved),
        "point_count": sum(len(stroke["points"]) for stroke in preserved),
        "canonical_sha256": hashlib.sha256(canonical).hexdigest(),
        "all_input_strokes_preserved": True,
    }


def _product_decision(result: dict[str, Any]) -> dict[str, Any]:
    audit = result.get("audit") or {}
    product_default = audit.get("product_default_enabled") is True
    posthoc = audit.get("posthoc_test_tuning") is True
    auto_commit = product_default and not posthoc
    return {
        "status": "AUTO_ACCEPTED" if auto_commit else "REVIEW_REQUIRED",
        "product_auto_committed": auto_commit,
        "reason": (
            "runtime artifacts passed their independent product admission gate"
            if auto_commit else
            "grouping/context runtime remains a posthoc shadow and raw ink is authoritative"
        ),
        "raw_fallback_available": True,
    }


def _provisional(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": EVENT_SCHEMA,
        "event": "preview",
        "formula_id": str(result["formula_id"]),
        "state": "PROVISIONAL",
        "committed": False,
        "groups": list(result.get("groups") or []),
        "symbols": [
            {
                "record_id": str(row["record_id"]),
                "stroke_indices": list(row["stroke_indices"]),
                "hwr_topk": list(row["hwr_topk"]),
                "hwr_topk_probabilities": list(row["hwr_topk_probabilities"]),
            }
            for row in result.get("symbols") or []
        ],
        "finalized_tokens": [],
        "formula_text": None,
        "formula_latex": None,
        "accuracy_scored": False,
    }


def _semantic_view(result: dict[str, Any]) -> dict:
    symbols = list(result.get("symbols") or [])
    by_id = {str(row["record_id"]): row for row in symbols}
    layout = result.get("formula_layout_shadow") or {}
    ordered_ids = [str(value) for value in layout.get("ordered_record_ids") or []]
    if not ordered_ids:
        ordered_ids = [
            str(row["record_id"])
            for row in sorted(symbols, key=lambda row: int(row["context_index"]))
        ]
    if set(ordered_ids) != set(by_id) or len(ordered_ids) != len(by_id):
        raise ValueError("formula-complete semantic order does not cover symbols")
    tokens = [str(by_id[record_id]["finalized_top1"]) for record_id in ordered_ids]
    return compose_function_spans(tokens, ordered_ids)


class FormulaCompleteSessionV1:
    def __init__(
        self, formula_id: str,
        infer_complete: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> None:
        if not formula_id:
            raise ValueError("formula_id is required")
        self.formula_id = formula_id
        self._infer_complete = infer_complete
        self._strokes: list[dict[str, Any]] = []
        self._completed: dict[str, Any] | None = None

    def append_stroke(self, stroke: dict[str, Any]) -> dict[str, Any]:
        if self._completed is not None:
            raise RuntimeError("cannot append a stroke after formula_complete")
        self._strokes.append(_stroke(stroke, len(self._strokes)))
        return {
            "schema": EVENT_SCHEMA,
            "event": "stroke",
            "formula_id": self.formula_id,
            "state": "COLLECTING",
            "committed": False,
            "stroke_count": len(self._strokes),
            "finalized_tokens": [],
            "accuracy_scored": False,
        }

    def preview(self) -> dict[str, Any]:
        if self._completed is not None:
            raise RuntimeError("preview is unavailable after formula_complete")
        if not self._strokes:
            raise RuntimeError("preview requires at least one stroke")
        return _provisional(self._infer_complete({
            "formula_id": self.formula_id,
            "strokes": deepcopy(self._strokes),
        }))

    def complete(self) -> dict[str, Any]:
        if self._completed is not None:
            return {**self._completed, "idempotent_replay": True}
        if not self._strokes:
            raise RuntimeError("formula_complete requires at least one stroke")
        raw_fallback = _raw_fallback(self.formula_id, self._strokes)
        result = self._infer_complete({
            "formula_id": self.formula_id,
            "strokes": deepcopy(self._strokes),
        })
        product_decision = _product_decision(result)
        result = {
            **result,
            "formula_complete_gate": {
                "schema": SCHEMA,
                "state": "FORMULA_COMPLETE",
                "committed": True,
                "result_finalized": True,
                "product_auto_committed": product_decision["product_auto_committed"],
                "commit_trigger": "explicit_formula_complete_event",
                "pre_complete_commits": 0,
                "prefix_accuracy_scored": False,
                "scored_result": "whole_formula_only",
                "stroke_count": len(self._strokes),
            },
            "semantic_composition": _semantic_view(result),
            "product_decision": product_decision,
            "raw_fallback": raw_fallback,
            "idempotent_replay": False,
        }
        self._completed = result
        return result


def _events(path: Path) -> list[dict]:
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError("event stream must contain JSON objects")
    return rows


def _self_test() -> None:
    def fake(source: dict[str, Any]) -> dict[str, Any]:
        return {
            "formula_id": source["formula_id"],
            "groups": [{"stroke_indices": list(range(len(source["strokes"])))}],
            "symbols": [{
                "record_id": "r0", "context_index": 0,
                "stroke_indices": list(range(len(source["strokes"]))),
                "hwr_topk": ["x"], "hwr_topk_probabilities": [1.0],
                "finalized_top1": "x",
            }],
            "finalized_tokens": ["x"],
        }

    session = FormulaCompleteSessionV1("f", fake)
    stroke = {"order": 0, "points": [{"x": 0, "y": 0}, {"x": 1, "y": 1}]}
    collecting = session.append_stroke(stroke)
    assert collecting["committed"] is False and collecting["finalized_tokens"] == []
    preview = session.preview()
    assert preview["committed"] is False and preview["formula_text"] is None
    complete = session.complete()
    assert complete["formula_complete_gate"]["committed"] is True
    assert complete["formula_complete_gate"]["product_auto_committed"] is False
    assert complete["product_decision"]["status"] == "REVIEW_REQUIRED"
    assert complete["raw_fallback"]["strokes"] == [stroke]
    assert complete["formula_complete_gate"]["pre_complete_commits"] == 0
    assert session.complete()["idempotent_replay"] is True
    try:
        session.append_stroke(stroke)
    except RuntimeError:
        pass
    else:
        raise AssertionError("post-completion stroke was accepted")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition-ranker", type=Path, default=DEFAULT_RANKER)
    parser.add_argument("--hwr-checkpoint", type=Path, default=DEFAULT_HWR)
    parser.add_argument("--context-checkpoint", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--formula-sequence-config", type=Path)
    parser.add_argument("--formula-syntax-rescue-config", type=Path)
    parser.add_argument("--input-events", type=Path)
    parser.add_argument("--output-events", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        print('{"self_test":"pass"}')
        return 0
    if args.input_events is None or args.output_events is None:
        parser.error("--input-events and --output-events are required")
    from raw_formula_context_runtime_v1 import RawFormulaContextRuntimeV1
    input_path = args.input_events.resolve()
    output_path = args.output_events.resolve()
    if input_path.drive.upper() != "D:" or output_path.drive.upper() != "D:":
        parser.error("event input and output must remain on D:")
    if not input_path.is_file() or output_path.exists():
        parser.error("input must exist and output must be new")
    runtime = RawFormulaContextRuntimeV1.from_artifacts(
        args.partition_ranker,
        args.hwr_checkpoint,
        args.context_checkpoint,
        device=args.device,
        formula_sequence_config=args.formula_sequence_config,
        formula_syntax_rescue_config=args.formula_syntax_rescue_config,
        emit_formula_layout_shadow=True,
        allow_posthoc_shadow=True,
    )
    sessions: dict[str, FormulaCompleteSessionV1] = {}
    output = []
    completed = set()
    for event in _events(input_path):
        event_type = str(event.get("type", ""))
        formula_id = str(event.get("formula_id", ""))
        if not formula_id:
            raise ValueError("every event requires formula_id")
        session = sessions.setdefault(
            formula_id, FormulaCompleteSessionV1(formula_id, runtime.infer)
        )
        if event_type == "stroke":
            output.append(session.append_stroke(event.get("stroke") or {}))
        elif event_type == "preview":
            output.append(session.preview())
        elif event_type == "formula_complete":
            output.append(session.complete())
            completed.add(formula_id)
        else:
            raise ValueError(f"unsupported event type: {event_type}")
    incomplete = sorted(set(sessions) - completed)
    if incomplete:
        raise ValueError(f"event stream ended before formula_complete: {incomplete}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in output),
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(output_path),
        "formulas": len(completed),
        "events": len(output),
        "pre_complete_commits": 0,
        "prefix_accuracy_scored": False,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
