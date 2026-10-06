#!/usr/bin/env python3
"""Revision-safe Fast/Selective-2D session contract for mobile hosts.

This module is separate from the historical formula-complete gate so existing
clients remain unchanged.  It is fail-closed: every inference exception emits
the complete immutable raw-ink payload instead of an empty recognition result.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any, Callable


SCHEMA = "aiflow-revisioned-formula-session/v1"
RESULT_SCHEMA = "aiflow-recognition-update/v1"
ROUTES = frozenset(("fast", "local_2d", "fallback"))


def _raw(formula_id: str, strokes: list[dict[str, Any]]) -> dict[str, Any]:
    preserved = deepcopy(strokes)
    encoded = json.dumps(preserved, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "schema": "aiflow-raw-ink-fallback/v1", "formula_id": formula_id,
        "strokes": preserved, "stroke_count": len(preserved),
        "point_count": sum(len(row["points"]) for row in preserved),
        "canonical_sha256": hashlib.sha256(encoded).hexdigest(),
        "all_input_strokes_preserved": True,
    }


def _stroke(value: dict[str, Any], expected_order: int) -> dict[str, Any]:
    if not isinstance(value, dict) or int(value.get("order", expected_order)) != expected_order:
        raise ValueError("stroke order must be contiguous")
    points = list(value.get("points") or ())
    if not points:
        raise ValueError("stroke requires at least one point")
    return deepcopy({**value, "order": expected_order, "points": points})


class RevisionedFormulaSessionV1:
    def __init__(
        self, formula_id: str, fast_infer: Callable[[dict[str, Any]], dict[str, Any]],
        refine_infer: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        if not formula_id:
            raise ValueError("formula_id is required")
        self.formula_id, self.fast_infer, self.refine_infer = formula_id, fast_infer, refine_infer
        self.revision = 0
        self._strokes: list[dict[str, Any]] = []
        # A completion event may be retried with a new transport id.  The
        # actual finalization is keyed by the immutable ink, not that id.
        self._final_by_revision_raw: dict[tuple[int, str], dict[str, Any]] = {}

    def _source(self) -> dict[str, Any]:
        return {"formula_id": self.formula_id, "strokes": deepcopy(self._strokes)}

    def _fallback(self, stage: str, error: Exception | str) -> dict[str, Any]:
        raw = _raw(self.formula_id, self._strokes)
        return {
            "schema": RESULT_SCHEMA, "formula_id": self.formula_id, "revision": self.revision,
            "stage": stage, "route": "fallback", "committed": stage == "final",
            "groups": [], "symbols": [], "formula_latex": None, "formula_text": None,
            "raw_sha256": raw["canonical_sha256"], "raw_fallback": raw,
            "failure_reason": str(error), "timings_ms": {}, "region_strokes": [],
            "candidate_count": 0, "winner_reason": "fallback",
        }

    def _invoke(self, callback: Callable[[dict[str, Any]], dict[str, Any]], stage: str) -> dict[str, Any]:
        try:
            result = dict(callback(self._source()))
        except Exception as error:
            return self._fallback(stage, f"{type(error).__name__}: {error}")
        route = str(result.get("route", "fast"))
        if route not in ROUTES:
            return self._fallback(stage, f"invalid route: {route}")
        raw = _raw(self.formula_id, self._strokes)
        return {
            **result, "schema": RESULT_SCHEMA, "formula_id": self.formula_id,
            "revision": self.revision, "stage": stage, "route": route,
            "committed": stage == "final", "raw_sha256": raw["canonical_sha256"],
            "raw_fallback": raw, "failure_reason": None,
            "timings_ms": dict(result.get("timings_ms", {})),
            "region_strokes": list(result.get("region_strokes", [])),
            "candidate_count": int(result.get("candidate_count", 0)),
            "winner_reason": result.get("winner_reason", route),
        }

    def stroke_end(self, revision: int, stroke: dict[str, Any]) -> dict[str, Any]:
        if revision != self.revision + 1:
            raise ValueError(f"expected revision {self.revision + 1}, got {revision}")
        self._strokes.append(_stroke(stroke, len(self._strokes)))
        self.revision = revision
        self._final_by_revision_raw.clear()
        fast = self._invoke(self.fast_infer, "fast")
        refined = self._invoke(self.refine_infer, "refined") if self.refine_infer else None
        return {"schema": SCHEMA, "event": "stroke_end", "formula_id": self.formula_id,
                "revision": self.revision, "fast": fast, "refined": refined}

    def replace_ink(self, revision: int, strokes: list[dict[str, Any]]) -> dict[str, Any]:
        if revision <= self.revision:
            raise ValueError("replacement revision must advance")
        self._strokes = [_stroke(row, index) for index, row in enumerate(strokes)]
        self.revision = revision
        self._final_by_revision_raw.clear()
        return {"schema": SCHEMA, "event": "replace_ink", "formula_id": self.formula_id,
                "revision": self.revision, "stroke_count": len(self._strokes)}

    def complete(self, revision: int, completion_event_id: str) -> dict[str, Any]:
        if revision != self.revision or not completion_event_id:
            raise ValueError("complete requires current revision and completion_event_id")
        raw_sha = _raw(self.formula_id, self._strokes)["canonical_sha256"]
        key = (revision, raw_sha)
        if key in self._final_by_revision_raw:
            return {
                **self._final_by_revision_raw[key], "completion_event_id": completion_event_id,
                "idempotent_replay": True,
            }
        result = self._invoke(self.refine_infer or self.fast_infer, "final")
        result.update(completion_event_id=completion_event_id, idempotent_replay=False)
        self._final_by_revision_raw[key] = result
        return result


def _self_test() -> None:
    def fast(source: dict[str, Any]) -> dict[str, Any]:
        return {"route": "fast", "groups": [[0]], "symbols": [], "formula_latex": "x"}

    def refined(source: dict[str, Any]) -> dict[str, Any]:
        return {"route": "local_2d", "groups": [[0]], "symbols": [], "formula_latex": "x"}

    session = RevisionedFormulaSessionV1("f", fast, refined)
    update = session.stroke_end(1, {"order": 0, "points": [{"x": 1, "y": 2}]})
    assert update["fast"]["stage"] == "fast" and update["refined"]["route"] == "local_2d"
    final = session.complete(1, "done")
    assert final["committed"] and session.complete(1, "retry")["idempotent_replay"]
    session.replace_ink(2, [{"order": 0, "points": [{"x": 1, "y": 2}]}])
    failed = RevisionedFormulaSessionV1("g", lambda _source: (_ for _ in ()).throw(RuntimeError("x")))
    assert failed.stroke_end(1, {"order": 0, "points": [{"x": 0, "y": 0}]})["fast"]["route"] == "fallback"


if __name__ == "__main__":
    _self_test()
    print('{"self_test":"pass"}')
