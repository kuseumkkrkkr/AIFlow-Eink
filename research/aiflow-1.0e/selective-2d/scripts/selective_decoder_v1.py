#!/usr/bin/env python3
"""Bounded Top-5 plus structure decoder for Selective-2D HWR shadows."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Iterable

from formula_layout_v1 import infer_formula_layout, selected_layout_evidence_rows


ALLOWED_RELATIONS = frozenset({"above", "below", "contains", "superscript", "subscript"})
FRACTION_BARS = frozenset({"-", r"\frac"})
ROOTS = frozenset({r"\sqrt", r"\sqrt{}"})
SCHEMA = "aiflow-selective-2d-decoder/v2"
DEFAULT_TOKEN_BEAM = 32


def _covers_exactly(groups: Iterable[Iterable[int]], stroke_count: int) -> bool:
    assigned = [int(index) for group in groups for index in group]
    return sorted(assigned) == list(range(stroke_count)) and len(assigned) == len(set(assigned))


def _strict_latex(rows: list[dict[str, Any]], predictions: dict[str, str]) -> tuple[str, list[dict[str, Any]], float]:
    evidence, _ = selected_layout_evidence_rows(rows, predictions)
    layout = infer_formula_layout(evidence)
    edges = list(layout["edges"])
    if any(str(edge["type"]) not in ALLOWED_RELATIONS for edge in edges):
        raise ValueError("unsupported relation")
    by_id = {str(row["record_id"]): row for row in evidence}
    parents: dict[str, tuple[str, str]] = {}
    children: dict[str, dict[str, list[str]]] = {}
    for edge in edges:
        parent, child, kind = str(edge["parent"]), str(edge["child"]), str(edge["type"])
        if child in parents:
            raise ValueError("multiple structure parents")
        parents[child] = (parent, kind)
        children.setdefault(parent, {}).setdefault(kind, []).append(child)
    for parent, slots in children.items():
        above, below, inside = slots.get("above", []), slots.get("below", []), slots.get("contains", [])
        token = predictions[parent]
        if bool(above) != bool(below):
            raise ValueError("incomplete fraction")
        if above and token not in FRACTION_BARS:
            raise ValueError("fraction bar outside top5 selection")
        if inside and token not in ROOTS:
            raise ValueError("root outside top5 selection")
        if inside and (above or below):
            raise ValueError("root fraction collision")
    for record_id, token in predictions.items():
        slots = children.get(record_id, {})
        if token in ROOTS and not slots.get("contains"):
            raise ValueError("root without radicand")

    emitted: set[str] = set()
    def sequence(record_ids: Iterable[str], active: frozenset[str]) -> str:
        return "".join(node(value, active) for value in sorted(set(record_ids), key=lambda value: (float(by_id[value]["geometry"]["left"]), value)))
    def node(record_id: str, active: frozenset[str]) -> str:
        if record_id in active or record_id in emitted:
            raise ValueError("cycle or duplicate AST node")
        emitted.add(record_id)
        slots = children.get(record_id, {})
        nested = active | {record_id}
        if slots.get("above"):
            value = r"\frac{" + sequence(slots["above"], nested) + "}{" + sequence(slots["below"], nested) + "}"
        elif slots.get("contains"):
            value = r"\sqrt{" + sequence(slots["contains"], nested) + "}"
        else:
            value = predictions[record_id]
        if slots.get("subscript"):
            value += "_{" + sequence(slots["subscript"], nested) + "}"
        if slots.get("superscript"):
            value += "^{" + sequence(slots["superscript"], nested) + "}"
        return value
    latex = sequence((key for key in by_id if key not in parents), frozenset())
    if emitted != set(by_id) or not latex:
        raise ValueError("incomplete AST")
    relation_score = sum(math.log(max(1e-8, float(edge["confidence"]))) for edge in edges) / max(len(edges), 1)
    return latex, edges, relation_score


def decode_selective_partition(
    formula_id: str, groups: list[list[int]], symbols: list[dict[str, Any]], *,
    stroke_count: int, token_beam: int = DEFAULT_TOKEN_BEAM,
    context_log_probabilities: Mapping[str, Mapping[str, float]] | None = None,
    context_weight: float = 0.0,
) -> dict[str, Any]:
    """Jointly choose existing Top-5 tokens and a valid structural AST.

    Optional context scores are keyed by ``{formula_id}:{group_ordinal}``, then
    by the exact HWR candidate token. They can reweight candidates but can never
    introduce a token. With the default zero weight, the legacy result is
    unchanged. Invalid active context scores fail closed for the caller to use
    its Fast fallback.
    """
    if not _covers_exactly(groups, stroke_count):
        return {"accepted": False, "reason": "exact_cover", "latex": None, "relations": []}
    try:
        context_weight = float(context_weight)
    except (TypeError, ValueError):
        return {"accepted": False, "reason": "context_weight_invalid", "latex": None, "relations": []}
    if not math.isfinite(context_weight) or context_weight < 0.0:
        return {"accepted": False, "reason": "context_weight_invalid", "latex": None, "relations": []}
    if context_weight > 0.0 and context_log_probabilities is None:
        return {"accepted": False, "reason": "context_scores_missing", "latex": None, "relations": []}
    by_group = {tuple(sorted(int(index) for index in row["stroke_indices"])): row for row in symbols}
    if len(by_group) != len(groups):
        return {"accepted": False, "reason": "symbol_coverage", "latex": None, "relations": []}
    rows, options = [], []
    for ordinal, group in enumerate(groups):
        symbol = by_group.get(tuple(sorted(int(index) for index in group)))
        if symbol is None:
            return {"accepted": False, "reason": "missing_symbol", "latex": None, "relations": []}
        tokens = [str(value) for value in symbol.get("hwr_topk") or ()]
        probabilities = [float(value) for value in symbol.get("hwr_topk_probabilities") or ()]
        geometry = dict(symbol.get("geometry") or {})
        if len(tokens) != 5 or len(probabilities) != 5 or not geometry:
            return {"accepted": False, "reason": "top5_or_geometry", "latex": None, "relations": []}
        record_id = f"{formula_id}:{ordinal}"
        rows.append({"record_id": record_id, "formula_id": formula_id, "final_topk": tokens,
                     "final_topk_probabilities": probabilities, "geometry": geometry})
        candidate_scores = [0.0] * len(tokens)
        if context_weight > 0.0:
            score_row = context_log_probabilities.get(record_id)  # type: ignore[union-attr]
            if not isinstance(score_row, Mapping):
                return {"accepted": False, "reason": "context_scores_invalid", "latex": None, "relations": []}
            try:
                candidate_scores = [float(score_row[token]) for token in tokens]
            except (KeyError, TypeError, ValueError):
                return {"accepted": False, "reason": "context_scores_invalid", "latex": None, "relations": []}
            if any(not math.isfinite(value) for value in candidate_scores):
                return {"accepted": False, "reason": "context_scores_invalid", "latex": None, "relations": []}
        options.append([
            (math.log(max(1e-8, probability)) + context_weight * context_score, token)
            for token, probability, context_score in zip(tokens, probabilities, candidate_scores, strict=True)
        ])
    beams: list[tuple[float, list[str]]] = [(0.0, [])]
    for choices in options:
        beams = sorted(((score + value, picked + [token]) for score, picked in beams for value, token in choices), key=lambda row: (-row[0], row[1]))[:token_beam]
    best: tuple[float, str, list[dict[str, Any]], list[str], float] | None = None
    for token_score, tokens in beams:
        predictions = {str(row["record_id"]): token for row, token in zip(rows, tokens, strict=True)}
        try:
            latex, relations, relation_score = _strict_latex(rows, predictions)
        except ValueError:
            continue
        score = token_score / len(rows) + relation_score
        if best is None or score > best[0]:
            best = (score, latex, relations, tokens, relation_score)
    if best is None:
        return {"accepted": False, "reason": "no_valid_top5_ast", "latex": None, "relations": []}
    score, latex, relations, tokens, relation_score = best
    result = {"schema": SCHEMA, "accepted": True, "reason": "joint_top5_structural_ast", "latex": latex,
              "relations": relations, "tokens": tokens, "joint_token_relation_score": score,
              "relation_log_score": relation_score, "top5_preserved": True}
    if context_weight > 0.0:
        result["context_weight"] = float(context_weight)
        result["context_scored_candidates"] = len(rows) * 5
    return result


def _self_test() -> None:
    fraction = decode_selective_partition("f", [[0], [1], [2]], [
        {"stroke_indices": [0], "hwr_topk": ["x", "-", "=", "+", "1"], "hwr_topk_probabilities": [.55, .35, .05, .03, .02], "geometry": {"left": 0, "top": 4, "right": 10, "bottom": 5}},
        {"stroke_indices": [1], "hwr_topk": ["1", "7", "x", "y", "-"], "hwr_topk_probabilities": [.8, .1, .05, .03, .02], "geometry": {"left": 2, "top": 0, "right": 4, "bottom": 2}},
        {"stroke_indices": [2], "hwr_topk": ["2", "7", "x", "y", "-"], "hwr_topk_probabilities": [.8, .1, .05, .03, .02], "geometry": {"left": 2, "top": 7, "right": 4, "bottom": 9}},
    ], stroke_count=3)
    assert fraction["accepted"] and fraction["latex"] == r"\frac{1}{2}" and fraction["tokens"][0] == "-"
    root = decode_selective_partition("r", [[0]], [{"stroke_indices": [0], "hwr_topk": [r"\sqrt", "x", "y", "z", "1"], "hwr_topk_probabilities": [.9, .04, .03, .02, .01], "geometry": {"left": 0, "top": 0, "right": 10, "bottom": 10}}], stroke_count=1)
    assert root["accepted"] and root["tokens"] == ["x"] and root["latex"] == "x"

    candidates = [
        {"stroke_indices": [0], "hwr_topk": ["x", "y", "z", "1", "2"], "hwr_topk_probabilities": [.7, .15, .07, .05, .03], "geometry": {"left": 0, "top": 0, "right": 3, "bottom": 3}},
        {"stroke_indices": [1], "hwr_topk": ["1", "2", "3", "4", "5"], "hwr_topk_probabilities": [.75, .14, .05, .035, .025], "geometry": {"left": 8, "top": 0, "right": 11, "bottom": 3}},
    ]
    fast = decode_selective_partition("ctx", [[0], [1]], candidates, stroke_count=2)
    disabled = decode_selective_partition(
        "ctx", [[0], [1]], candidates, stroke_count=2,
        context_log_probabilities={}, context_weight=0.0,
    )
    assert disabled == fast  # router-off remains byte-for-byte equivalent
    context = {
        "ctx:0": {"x": 0.0, "y": -8.0, "z": -8.0, "1": -8.0, "2": -8.0, "not-in-top5": 1000.0},
        "ctx:1": {"1": -8.0, "2": 0.0, "3": -8.0, "4": -8.0, "5": -8.0, "not-in-top5": 1000.0},
    }
    contextual = decode_selective_partition(
        "ctx", [[0], [1]], candidates, stroke_count=2,
        context_log_probabilities=context, context_weight=0.5,
    )
    assert contextual["accepted"] and contextual["tokens"] == ["x", "2"]
    assert contextual["relations"] == fast["relations"]
    assert all(token in row["hwr_topk"] for token, row in zip(contextual["tokens"], candidates, strict=True))
    assert contextual["context_scored_candidates"] == 10
    invalid_context = decode_selective_partition(
        "ctx", [[0], [1]], candidates, stroke_count=2,
        context_log_probabilities={"ctx:0": context["ctx:0"]}, context_weight=0.5,
    )
    assert not invalid_context["accepted"] and invalid_context["reason"] == "context_scores_invalid"
    invalid_weight = decode_selective_partition(
        "ctx", [[0], [1]], candidates, stroke_count=2,
        context_log_probabilities=context, context_weight="invalid",  # type: ignore[arg-type]
    )
    assert not invalid_weight["accepted"] and invalid_weight["reason"] == "context_weight_invalid"


if __name__ == "__main__":
    _self_test()
    print('{"self_test":"pass"}')
