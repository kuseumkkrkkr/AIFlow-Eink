#!/usr/bin/env python3
"""Evaluate a frozen raw-stroke runtime on every CROHME test InkML row.

This is a research-only coverage and diagnostic evaluator.  It never trains,
selects, or tunes from CROHME.  Unlike the historical raw evaluator, malformed
truth partitions, labels outside the 372-class head, and one-point strokes
remain in the 1,199-formula denominator as explicit failures.

The reported strict score is a local grouping/layout/relation proxy.  It is
not CROHME Expression Rate: this repository has no symLG writer or LgEval
integration.  Function-name span composition is similarly diagnostic-only and
does not mutate the frozen runtime's grouping or token output.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
import torch

from evaluate_48hz_prefix_v1 import _sha256
from formula_layout_v1 import STRUCTURAL
from raw_formula_context_runtime_v1 import RawFormulaContextRuntimeV1
from replay_evaluate_hwr_v1 import _local_name, _trace_points


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CROHME_TEST_ROWS = 1199
TICK_MS = 1000.0 / 48.0
SCHEMA = "aiflow-crohme-raw-full-coverage-research/v1"
CROHME_ALIASES = {
    r"\sqrt": r"\sqrt{}",
    r"\ldots": r"\dots",
    r"\lt": "<",
    r"\gt": ">",
}
FUNCTION_SPANS = {
    ("s", "i", "n"): r"\sin",
    ("c", "o", "s"): r"\cos",
    ("t", "a", "n"): r"\tan",
    ("l", "i", "m"): r"\lim",
    ("l", "o", "g"): r"\log",
}


class _GroupingProbabilityRoute:
    """Replay an artifact-owned grouping blend through an older runtime API.

    The clean runtime calls enumerate_partitions without a group bias. The r11
    artifact route blends its long-formula model and applies that bias. Moving
    the bias into probability logits produces the same candidate score without
    altering the frozen ranker, HWR, context, or runtime source contract.
    """

    def __init__(
        self,
        baseline: Any,
        *,
        long_model: Any | None,
        baseline_probability_weight: float,
        group_bias: float,
    ) -> None:
        self.baseline = baseline
        self.long_model = long_model
        self.baseline_probability_weight = float(baseline_probability_weight)
        self.group_bias = float(group_bias)
        if not 0.0 <= self.baseline_probability_weight <= 1.0:
            raise ValueError("invalid grouping probability blend weight")

    def predict_proba(self, features: Any) -> np.ndarray:
        probability = np.asarray(
            self.baseline.predict_proba(features), dtype=np.float64,
        )[:, 1]
        if self.long_model is not None:
            long_probability = np.asarray(
                self.long_model.predict_proba(features), dtype=np.float64,
            )[:, 1]
            probability = (
                self.baseline_probability_weight * probability
                + (1.0 - self.baseline_probability_weight) * long_probability
            )
        logits = np.log(
            np.clip(probability, 1e-6, 1.0 - 1e-6)
            / np.clip(1.0 - probability, 1e-6, 1.0)
        )
        routed = 1.0 / (1.0 + np.exp(-(logits + self.group_bias)))
        return np.column_stack((1.0 - routed, routed))


def _grouping_route_config(runtime: RawFormulaContextRuntimeV1) -> dict[str, Any]:
    payload = runtime.ranker_payload
    route = dict(payload.get("grouping_long_formula_route") or {})
    if not route:
        return {
            "enabled": False,
            "route": "owned_baseline",
            "minimum_strokes": None,
            "group_bias": float(payload.get("group_bias", 0.0)),
            "baseline_probability_weight": None,
        }
    required = {
        "minimum_strokes",
        "group_bias",
        "baseline_probability_weight",
        "target_label_or_glyph_count_input",
    }
    if set(route) != required or route["target_label_or_glyph_count_input"] is not False:
        raise ValueError("invalid artifact-owned long-formula grouping route")
    if "grouping_long_formula_model" not in payload:
        raise ValueError("long-formula grouping model is missing")
    minimum_strokes = int(route["minimum_strokes"])
    if minimum_strokes < 1:
        raise ValueError("long-formula grouping minimum_strokes is invalid")
    return {
        "enabled": True,
        "route": "owned_blended_long_formula",
        "minimum_strokes": minimum_strokes,
        "group_bias": float(route["group_bias"]),
        "baseline_probability_weight": float(route["baseline_probability_weight"]),
    }


@contextmanager
def _apply_grouping_route(
    runtime: RawFormulaContextRuntimeV1, stroke_count: int,
):
    """Temporarily adapt only the frozen artifact score route for inference."""

    payload = runtime.ranker_payload
    config = _grouping_route_config(runtime)
    baseline = payload["grouping_model"]
    use_long = bool(
        config["enabled"] and stroke_count >= int(config["minimum_strokes"])
    )
    group_bias = (
        float(config["group_bias"])
        if use_long else float(payload.get("group_bias", 0.0))
    )
    if use_long or group_bias:
        payload["grouping_model"] = _GroupingProbabilityRoute(
            baseline,
            long_model=payload["grouping_long_formula_model"] if use_long else None,
            baseline_probability_weight=(
                float(config["baseline_probability_weight"]) if use_long else 1.0
            ),
            group_bias=group_bias,
        )
    try:
        yield {
            "route": str(config["route"]) if use_long else "owned_baseline",
            "group_bias": group_bias,
            "minimum_strokes": config["minimum_strokes"],
            "adapter_enabled": use_long or bool(group_bias),
        }
    finally:
        payload["grouping_model"] = baseline


@dataclass(frozen=True)
class TruthGroup:
    """One symbol-level CROHME trace group, preserving its stroke ownership."""

    label: str
    stroke_indices: tuple[int, ...]
    href: str | None


@dataclass(frozen=True)
class TruthSample:
    strokes: tuple[tuple[tuple[float, float], ...], ...]
    groups: tuple[TruthGroup, ...]
    raw_trace_elements: int
    invalid_raw_trace_elements: int
    duplicate_trace_ids: int
    source_valid: bool
    partition_complete: bool
    parse_status: str
    ignored_unknown_trace_reference_groups: int
    formula_truth: str | None
    root: ET.Element


def _normalise_label(value: str) -> str:
    raw = str(value).strip().replace("$", "")
    return CROHME_ALIASES.get(raw, raw)


def _direct_truth_label(group: ET.Element) -> str:
    for child in group:
        if (
            _local_name(child) == "annotation"
            and child.attrib.get("type") == "truth"
            and child.text
        ):
            return _normalise_label(child.text)
    return ""


def _direct_href(group: ET.Element) -> str | None:
    for child in group:
        if _local_name(child) != "annotationXML":
            continue
        href = str(child.attrib.get("href", "")).strip()
        if href:
            return href
    return None


def _has_nested_semantic_group(group: ET.Element) -> bool:
    """Do not count a container group when it contains named child symbols."""

    for descendant in group.iter():
        if descendant is group or _local_name(descendant) != "traceGroup":
            continue
        label = _direct_truth_label(descendant)
        if label and label != "Closest Strk":
            return True
    return False


def _descendant_trace_refs(group: ET.Element) -> list[str]:
    refs = []
    for descendant in group.iter():
        if _local_name(descendant) != "traceView":
            continue
        ref = str(descendant.attrib.get("traceDataRef", "")).strip()
        if ref:
            refs.append(ref)
    return list(dict.fromkeys(refs))


def _formula_truth(root: ET.Element) -> str | None:
    for child in root:
        if (
            _local_name(child) == "annotation"
            and child.attrib.get("type") == "truth"
            and child.text
        ):
            return str(child.text).strip()
    return None


def _parse_truth_root(root: ET.Element) -> TruthSample:
    """Parse CROHME's direct and nested traceGroup encodings without mutation."""

    trace_ids: list[str] = []
    strokes: list[tuple[tuple[float, float], ...]] = []
    raw_trace_elements = 0
    invalid_raw_trace_elements = 0
    duplicate_trace_ids = 0
    for node in root.iter():
        if _local_name(node) != "trace":
            continue
        raw_trace_elements += 1
        points = tuple((float(x), float(y)) for x, y in _trace_points(node.text))
        if not points:
            invalid_raw_trace_elements += 1
            continue
        trace_id = str(node.attrib.get("id", len(trace_ids))).strip()
        if not trace_id:
            invalid_raw_trace_elements += 1
            continue
        if trace_id in trace_ids:
            duplicate_trace_ids += 1
            continue
        trace_ids.append(trace_id)
        strokes.append(points)
    index_by_id = {trace_id: index for index, trace_id in enumerate(trace_ids)}

    groups: list[TruthGroup] = []
    ignored_unknown_trace_reference_groups = 0
    for group in root.iter():
        if _local_name(group) != "traceGroup":
            continue
        label = _direct_truth_label(group)
        if not label or label == "Closest Strk" or _has_nested_semantic_group(group):
            continue
        refs = _descendant_trace_refs(group)
        if not refs:
            continue
        if any(ref not in index_by_id for ref in refs):
            # Some CROHME files retain a trailing symbol annotation whose
            # trace ID is absent.  Do not turn an otherwise complete,
            # independently parseable source into an omitted evaluation row.
            ignored_unknown_trace_reference_groups += 1
            continue
        groups.append(
            TruthGroup(
                label=label,
                stroke_indices=tuple(index_by_id[ref] for ref in refs),
                href=_direct_href(group),
            )
        )

    assigned = [index for group in groups for index in group.stroke_indices]
    duplicate_ownership = len(assigned) != len(set(assigned))
    source_valid = (
        raw_trace_elements == len(strokes)
        and invalid_raw_trace_elements == 0
        and duplicate_trace_ids == 0
    )
    partition_complete = (
        source_valid
        and bool(groups)
        and not duplicate_ownership
        and set(assigned) == set(range(len(strokes)))
    )
    if not strokes:
        parse_status = "missing_traces"
    elif not source_valid:
        parse_status = "invalid_raw_trace_stream"
    elif not groups:
        parse_status = "missing_symbol_groups"
    elif duplicate_ownership:
        parse_status = "duplicate_stroke_ownership"
    elif not partition_complete:
        parse_status = "incomplete_stroke_ownership"
    elif ignored_unknown_trace_reference_groups:
        parse_status = "ok_with_ignored_unknown_trace_reference"
    else:
        parse_status = "ok"
    return TruthSample(
        strokes=tuple(strokes),
        groups=tuple(groups),
        raw_trace_elements=raw_trace_elements,
        invalid_raw_trace_elements=invalid_raw_trace_elements,
        duplicate_trace_ids=duplicate_trace_ids,
        source_valid=source_valid,
        partition_complete=partition_complete,
        parse_status=parse_status,
        ignored_unknown_trace_reference_groups=(
            ignored_unknown_trace_reference_groups
        ),
        formula_truth=_formula_truth(root),
        root=root,
    )


def _parse_inkml(path: Path) -> TruthSample:
    return _parse_truth_root(ET.parse(path).getroot())


def _source(formula_id: str, sample: TruthSample) -> dict[str, Any]:
    """Build the only payload supplied to runtime inference: ordered raw ink."""

    if not sample.source_valid:
        raise ValueError("runtime source cannot omit malformed raw trace elements")
    tick = 0
    strokes = []
    for order, stroke in enumerate(sample.strokes):
        points = []
        for x, y in stroke:
            points.append({"x": float(x), "y": float(y), "t_ms": tick * TICK_MS})
            tick += 1
        strokes.append({"order": order, "points": points})
    return {"formula_id": formula_id, "strokes": strokes}


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _truth_rows(formula_id: str, sample: TruthSample) -> list[dict[str, Any]]:
    rows = []
    for index, group in enumerate(sample.groups):
        points = [
            point
            for stroke_index in group.stroke_indices
            for point in sample.strokes[stroke_index]
        ]
        if not points:
            continue
        left, right = min(point[0] for point in points), max(point[0] for point in points)
        top, bottom = min(point[1] for point in points), max(point[1] for point in points)
        rows.append({
            "record_id": f"{formula_id}:{index}",
            "group_index": index,
            "label": group.label,
            "geometry": {
                "left": left,
                "right": right,
                "top": top,
                "bottom": bottom,
                "center_x": (left + right) / 2.0,
                "center_y": (top + bottom) / 2.0,
            },
            "href": group.href,
        })
    return rows


def _element_id(element: ET.Element) -> str:
    return str(
        element.attrib.get("{http://www.w3.org/XML/1998/namespace}id")
        or element.attrib.get("id")
        or ""
    ).strip()


def _mapped_ids(element: ET.Element, href_to_record: dict[str, str]) -> list[str]:
    output = []
    for child in element.iter():
        record_id = href_to_record.get(_element_id(child))
        if record_id:
            output.append(record_id)
    return list(dict.fromkeys(output))


def _truth_edges(sample: TruthSample, rows: list[dict[str, Any]]) -> set[tuple[str, str, str]]:
    """Use the repository's local MathML relation proxy with robust group hrefs."""

    href_to_record = {
        str(row["href"]): str(row["record_id"])
        for row in rows
        if row.get("href")
    }
    boxes = {str(row["record_id"]): row["geometry"] for row in rows}
    edges: set[tuple[str, str, str]] = set()
    for element in sample.root.iter():
        tag, children = _local_name(element), list(element)
        slots: list[tuple[int, str]] = []
        if tag == "msup" and len(children) >= 2:
            slots = [(1, "superscript")]
        elif tag == "msub" and len(children) >= 2:
            slots = [(1, "subscript")]
        elif tag == "msubsup" and len(children) >= 3:
            slots = [(1, "subscript"), (2, "superscript")]
        elif tag == "munder" and len(children) >= 2:
            slots = [(1, "subscript")]
        elif tag == "mover" and len(children) >= 2:
            slots = [(1, "superscript")]
        elif tag == "munderover" and len(children) >= 3:
            slots = [(1, "subscript"), (2, "superscript")]
        if slots:
            bases = _mapped_ids(children[0], href_to_record)
            if bases:
                parent = max(bases, key=lambda record_id: float(boxes[record_id]["right"]))
                for slot, relation in slots:
                    edges.update(
                        (parent, child, relation)
                        for child in _mapped_ids(children[slot], href_to_record)
                        if child != parent
                    )
        anchor = href_to_record.get(_element_id(element))
        if not anchor:
            continue
        if tag == "mfrac" and len(children) >= 2:
            for child in _mapped_ids(children[0], href_to_record):
                if child != anchor:
                    edges.add((anchor, child, "above"))
            for child in _mapped_ids(children[1], href_to_record):
                if child != anchor:
                    edges.add((anchor, child, "below"))
        elif tag in {"msqrt", "mroot"} and children:
            content = children[0] if tag == "mroot" else element
            for child in _mapped_ids(content, href_to_record):
                if child != anchor:
                    edges.add((anchor, child, "contains"))
    return edges


def _runtime_exact_cover(result: dict[str, Any], stroke_count: int) -> bool:
    groups = result.get("groups")
    if not isinstance(groups, list):
        return False
    assigned = [
        int(index)
        for group in groups
        for index in list(group.get("stroke_indices") or [])
    ]
    return (
        sorted(assigned) == list(range(stroke_count))
        and len(assigned) == len(set(assigned))
    )


def _ordered_prediction(result: dict[str, Any]) -> tuple[list[str], list[str]]:
    layout = result.get("formula_layout_shadow")
    if isinstance(layout, dict):
        tokens = [str(value) for value in list(layout.get("ordered_tokens") or [])]
        record_ids = [
            str(value) for value in list(layout.get("ordered_record_ids") or [])
        ]
        if tokens and len(tokens) == len(record_ids):
            return tokens, record_ids
    symbols = sorted(
        list(result.get("symbols") or []),
        key=lambda row: int(row.get("context_index", 0)),
    )
    return (
        [str(row.get("finalized_top1", "")) for row in symbols],
        [str(row.get("record_id", "")) for row in symbols],
    )


def _group_boxes(result: dict[str, Any]) -> dict[str, dict[str, float]]:
    output = {}
    for group in list(result.get("groups") or []):
        record_id = str(group.get("record_id", ""))
        box = dict(group.get("box") or {})
        if not record_id or not {"left", "right", "top", "bottom"} <= set(box):
            continue
        output[record_id] = {
            key: float(box[key]) for key in ("left", "right", "top", "bottom")
        }
    return output


def _same_baseline(
    record_ids: list[str],
    boxes: dict[str, dict[str, float]],
    structural_nodes: set[str],
) -> bool:
    """Conservatively reject function triples attached to 2D structure."""

    if any(record_id in structural_nodes for record_id in record_ids):
        return False
    try:
        values = [boxes[record_id] for record_id in record_ids]
    except KeyError:
        return False
    for previous, current in zip(values, values[1:]):
        previous_center_x = (previous["left"] + previous["right"]) / 2.0
        current_center_x = (current["left"] + current["right"]) / 2.0
        previous_center_y = (previous["top"] + previous["bottom"]) / 2.0
        current_center_y = (current["top"] + current["bottom"]) / 2.0
        previous_height = max(previous["bottom"] - previous["top"], 1e-6)
        current_height = max(current["bottom"] - current["top"], 1e-6)
        mean_height = (previous_height + current_height) / 2.0
        if current_center_x < previous_center_x - 0.10 * mean_height:
            return False
        if abs(current_center_y - previous_center_y) > 0.35 * mean_height:
            return False
    return True


def _compose_function_spans(
    tokens: list[str],
    record_ids: list[str],
    boxes: dict[str, dict[str, float]],
    structural_nodes: set[str],
) -> list[dict[str, Any]]:
    """Compose only exact, same-baseline Latin triples for diagnostic scoring."""

    if len(tokens) != len(record_ids):
        return []
    spans = []
    index = 0
    while index < len(tokens):
        triple = tuple(tokens[index:index + 3])
        function = FUNCTION_SPANS.get(triple)
        function = (
            function
            if function and _same_baseline(
                record_ids[index:index + 3], boxes, structural_nodes
            )
            else None
        )
        width = 3 if function else 1
        spans.append({
            "token": function or tokens[index],
            "source_record_ids": record_ids[index:index + width],
            "diagnostic_function_span": bool(function),
        })
        index += width
    return spans


def _safe_error(error: Exception) -> str:
    return f"{type(error).__name__}: {str(error)}"[:400]


def _safe_file_sha256(path: Path) -> str | None:
    try:
        return _sha256(path)
    except OSError:
        return None


def _evaluate_formula(
    formula_id: str,
    sample: TruthSample,
    runtime_labels: set[str],
    result: dict[str, Any] | None,
    inference_error: str | None,
    raw_source_sha256: str | None,
    raw_source_hash_kind: str,
    raw_source_unchanged: bool | None,
) -> dict[str, Any]:
    truth_tokens = [group.label for group in sample.groups]
    truth_rows = _truth_rows(formula_id, sample)
    row: dict[str, Any] = {
        "formula_id": formula_id,
        "inference_status": "error" if inference_error else "ok",
        "inference_error": inference_error,
        "raw_source_sha256": raw_source_sha256,
        "raw_source_hash_kind": raw_source_hash_kind,
        "raw_source_unchanged": raw_source_unchanged,
        "raw_trace_elements": sample.raw_trace_elements,
        "invalid_raw_trace_elements": sample.invalid_raw_trace_elements,
        "duplicate_trace_ids": sample.duplicate_trace_ids,
        "source_valid": sample.source_valid,
        "source_strokes": len(sample.strokes),
        "source_points": sum(len(stroke) for stroke in sample.strokes),
        "single_point_source_strokes": sum(
            len(stroke) < 2 for stroke in sample.strokes
        ),
        "truth_parse_status": sample.parse_status,
        "ignored_unknown_trace_reference_groups": (
            sample.ignored_unknown_trace_reference_groups
        ),
        "truth_partition_complete": sample.partition_complete,
        "truth_formula": sample.formula_truth,
        "truth_tokens": truth_tokens,
        "unsupported_truth_tokens": [
            token for token in truth_tokens if token not in runtime_labels
        ],
        "runtime_groups": 0,
        "truth_groups": len(sample.groups),
        "all_strokes_exactly_once": False,
        "grouping_exact": False,
        "flat_token_sequence_exact": False,
        "layout_order_exact": False,
        "layout_token_sequence_exact": False,
        "semantic_function_span_sequence_exact": False,
        "semantic_function_span_count": 0,
        "relation_parsed": False,
        "relation_exact": False,
        "truth_relations": [],
        "predicted_relations": [],
        "strict_group_layout_relation_character_exact": False,
    }
    if result is None:
        return row

    row["runtime_groups"] = len(list(result.get("groups") or []))
    row["all_strokes_exactly_once"] = _runtime_exact_cover(
        result, len(sample.strokes)
    )
    raw_tokens = [str(value) for value in list(result.get("finalized_tokens") or [])]
    ordered_tokens, ordered_record_ids = _ordered_prediction(result)
    layout = result.get("formula_layout_shadow") or {}
    structural_nodes = {
        str(edge[key])
        for edge in list(layout.get("relations") or [])
        if str(edge.get("type")) in STRUCTURAL
        for key in ("parent", "child")
    }
    semantic_spans = _compose_function_spans(
        ordered_tokens,
        ordered_record_ids,
        _group_boxes(result),
        structural_nodes,
    )
    semantic_tokens = [str(span["token"]) for span in semantic_spans]
    row.update({
        "finalized_tokens": raw_tokens,
        "layout_ordered_tokens": ordered_tokens,
        "semantic_token_spans": semantic_spans,
        "semantic_function_span_count": sum(
            bool(span["diagnostic_function_span"]) for span in semantic_spans
        ),
    })
    if not sample.parse_status.startswith("ok"):
        return row

    truth_groups = {
        frozenset(group.stroke_indices): f"{formula_id}:{index}"
        for index, group in enumerate(sample.groups)
    }
    selected_groups = {
        frozenset(int(value) for value in row_data.get("stroke_indices") or []):
        str(row_data.get("record_id", ""))
        for row_data in list(result.get("groups") or [])
    }
    row["grouping_exact"] = (
        len(selected_groups) == len(list(result.get("groups") or []))
        and set(selected_groups) == set(truth_groups)
    )
    row["flat_token_sequence_exact"] = raw_tokens == truth_tokens
    row["semantic_function_span_sequence_exact"] = semantic_tokens == truth_tokens
    if not row["grouping_exact"]:
        return row

    runtime_to_truth = {
        runtime_id: truth_groups[group]
        for group, runtime_id in selected_groups.items()
    }
    layout_ids = [str(value) for value in list(layout.get("ordered_record_ids") or [])]
    expected_ids = [f"{formula_id}:{index}" for index in range(len(truth_tokens))]
    row["layout_order_exact"] = (
        [runtime_to_truth.get(record_id, "") for record_id in layout_ids]
        == expected_ids
    )
    row["layout_token_sequence_exact"] = (
        [str(value) for value in list(layout.get("ordered_tokens") or [])]
        == truth_tokens
    )
    try:
        truth = _truth_edges(sample, truth_rows)
    except (KeyError, TypeError, ValueError) as error:
        row["relation_parse_error"] = _safe_error(error)
        return row
    predicted = {
        (
            runtime_to_truth[str(edge["parent"])],
            runtime_to_truth[str(edge["child"])],
            str(edge["type"]),
        )
        for edge in list(layout.get("relations") or [])
        if (
            str(edge.get("type")) in STRUCTURAL
            and str(edge.get("parent")) in runtime_to_truth
            and str(edge.get("child")) in runtime_to_truth
        )
    }
    row["relation_parsed"] = True
    row["relation_exact"] = predicted == truth
    row["truth_relations"] = sorted(truth)
    row["predicted_relations"] = sorted(predicted)
    row["strict_group_layout_relation_character_exact"] = bool(
        row["layout_order_exact"]
        and row["layout_token_sequence_exact"]
        and row["relation_exact"]
    )
    return row


def _source_parse_failure_row(
    formula_id: str, error: Exception, raw_inkml_sha256: str | None,
) -> dict[str, Any]:
    """Retain a non-decodable InkML row without claiming runtime validation."""

    return {
        "formula_id": formula_id,
        "inference_status": "error",
        "inference_error": _safe_error(error),
        "raw_source_sha256": raw_inkml_sha256,
        "raw_source_hash_kind": (
            "inkml_bytes" if raw_inkml_sha256 is not None else "unavailable"
        ),
        "raw_source_unchanged": None,
        "raw_trace_elements": 0,
        "invalid_raw_trace_elements": 0,
        "duplicate_trace_ids": 0,
        "source_valid": False,
        "source_strokes": 0,
        "source_points": 0,
        "single_point_source_strokes": 0,
        "truth_parse_status": "source_parse_error",
        "ignored_unknown_trace_reference_groups": 0,
        "truth_partition_complete": False,
        "truth_formula": None,
        "truth_tokens": [],
        "unsupported_truth_tokens": [],
        "runtime_groups": 0,
        "truth_groups": 0,
        "all_strokes_exactly_once": False,
        "grouping_exact": False,
        "flat_token_sequence_exact": False,
        "layout_order_exact": False,
        "layout_token_sequence_exact": False,
        "semantic_function_span_sequence_exact": False,
        "semantic_function_span_count": 0,
        "relation_parsed": False,
        "relation_exact": False,
        "truth_relations": [],
        "predicted_relations": [],
        "strict_group_layout_relation_character_exact": False,
    }


def _ratio(count: int, denominator: int) -> float:
    return count / max(denominator, 1)


def _formula_scores(rows: list[dict[str, Any]], denominator: int) -> dict[str, Any]:
    """Formula scores with an explicit denominator supplied by the caller."""

    return {
        "denominator": denominator,
        "grouping_exact_count": sum(row["grouping_exact"] for row in rows),
        "grouping_exact": _ratio(
            sum(row["grouping_exact"] for row in rows), denominator
        ),
        "flat_token_sequence_exact_count": sum(
            row["flat_token_sequence_exact"] for row in rows
        ),
        "flat_token_sequence_exact": _ratio(
            sum(row["flat_token_sequence_exact"] for row in rows), denominator
        ),
        "semantic_function_span_sequence_exact_count": sum(
            row["semantic_function_span_sequence_exact"] for row in rows
        ),
        "semantic_function_span_sequence_exact": _ratio(
            sum(row["semantic_function_span_sequence_exact"] for row in rows),
            denominator,
        ),
        "layout_order_exact_count": sum(
            row["layout_order_exact"] for row in rows
        ),
        "layout_order_exact": _ratio(
            sum(row["layout_order_exact"] for row in rows), denominator
        ),
        "layout_token_sequence_exact_count": sum(
            row["layout_token_sequence_exact"] for row in rows
        ),
        "layout_token_sequence_exact": _ratio(
            sum(row["layout_token_sequence_exact"] for row in rows), denominator
        ),
        "strict_group_layout_relation_character_exact_count": sum(
            row["strict_group_layout_relation_character_exact"] for row in rows
        ),
        "strict_group_layout_relation_character_exact": _ratio(
            sum(
                row["strict_group_layout_relation_character_exact"] for row in rows
            ),
            denominator,
        ),
    }


def _summarise(
    rows: list[dict[str, Any]], *, full_coverage: bool,
) -> dict[str, Any]:
    total = len(rows)
    parsed = [row for row in rows if row["relation_parsed"]]
    structural = [row for row in parsed if row["truth_relations"]]
    relation_counts: Counter[str] = Counter(tp=0, fp=0, fn=0)
    for row in parsed:
        truth = {tuple(value) for value in row["truth_relations"]}
        predicted = {tuple(value) for value in row["predicted_relations"]}
        relation_counts["tp"] += len(truth & predicted)
        relation_counts["fp"] += len(predicted - truth)
        relation_counts["fn"] += len(truth - predicted)
    precision = _ratio(
        relation_counts["tp"], relation_counts["tp"] + relation_counts["fp"]
    )
    recall = _ratio(
        relation_counts["tp"], relation_counts["tp"] + relation_counts["fn"]
    )
    unsupported = Counter(
        token for row in rows for token in row["unsupported_truth_tokens"]
    )
    return {
        "evaluated_subset_denominator": total,
        "inference": {
            "ok": sum(row["inference_status"] == "ok" for row in rows),
            "error": sum(row["inference_status"] == "error" for row in rows),
            "error_types": dict(sorted(
                Counter(
                    str(row.get("inference_error", "")).split(":", 1)[0]
                    for row in rows if row["inference_status"] == "error"
                ).items()
            )),
            "success_exact_stroke_cover": sum(
                row["inference_status"] == "ok"
                and row["all_strokes_exactly_once"]
                for row in rows
            ),
            "grouping_routes": dict(sorted(
                Counter(
                    str(dict(row.get("grouping_route") or {}).get(
                        "route", "not_run"
                    ))
                    for row in rows
                ).items()
            )),
        },
        "truth": {
            "parse_status": dict(sorted(
                Counter(str(row["truth_parse_status"]) for row in rows).items()
            )),
            "source_valid_rows": sum(bool(row["source_valid"]) for row in rows),
            "raw_trace_elements": sum(
                int(row["raw_trace_elements"]) for row in rows
            ),
            "invalid_raw_trace_elements": sum(
                int(row["invalid_raw_trace_elements"]) for row in rows
            ),
            "duplicate_trace_ids": sum(
                int(row["duplicate_trace_ids"]) for row in rows
            ),
            "partition_complete": sum(
                bool(row["truth_partition_complete"]) for row in rows
            ),
            "ignored_unknown_trace_reference_groups": sum(
                int(row["ignored_unknown_trace_reference_groups"]) for row in rows
            ),
            "formulas_with_unsupported_truth_tokens": sum(
                bool(row["unsupported_truth_tokens"]) for row in rows
            ),
            "unsupported_truth_token_counts": dict(sorted(unsupported.items())),
            "formulas_with_single_point_source_stroke": sum(
                int(row["single_point_source_strokes"]) > 0 for row in rows
            ),
            "single_point_source_strokes": sum(
                int(row["single_point_source_strokes"]) for row in rows
            ),
        },
        "evaluated_subset_scores": _formula_scores(rows, total),
        "all_1199_denominator_scores": (
            _formula_scores(rows, EXPECTED_CROHME_TEST_ROWS)
            if full_coverage else None
        ),
        "relation_proxy_conditioned_on_grouping_exact": {
            "parsed_formulas": len(parsed),
            "structural_formulas": len(structural),
            "formula_exact_count": sum(row["relation_exact"] for row in parsed),
            "formula_exact": _ratio(
                sum(row["relation_exact"] for row in parsed), len(parsed)
            ),
            "formula_exact_on_structural_count": sum(
                row["relation_exact"] for row in structural
            ),
            "formula_exact_on_structural": _ratio(
                sum(row["relation_exact"] for row in structural), len(structural)
            ),
            "micro": {
                **dict(relation_counts),
                "precision": precision,
                "recall": recall,
                "f1": _ratio(2 * precision * recall, precision + recall),
            },
        },
    }


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")


def _self_test() -> None:
    class _FixedProbability:
        def __init__(self, probability: float) -> None:
            self.probability = probability

        def predict_proba(self, features: Any) -> np.ndarray:
            count = len(features)
            return np.asarray(
                [[1.0 - self.probability, self.probability]] * count,
                dtype=np.float64,
            )

    routed = _GroupingProbabilityRoute(
        _FixedProbability(0.6),
        long_model=_FixedProbability(0.2),
        baseline_probability_weight=0.25,
        group_bias=-1.0,
    ).predict_proba([0, 1])[:, 1]
    expected = 1.0 / (1.0 + np.exp(-(
        np.log(0.3 / 0.7) - 1.0
    )))
    assert np.allclose(routed, [expected, expected])
    nested = ET.fromstring(
        """
        <ink>
          <annotation type="truth">$A_i$</annotation>
          <trace id="0">0 0, 1 1</trace>
          <trace id="1">1 0, 2 1</trace>
          <trace id="2">4 1, 5 2</trace>
          <trace id="3">5 2</trace>
          <traceGroup><annotation type="truth">Closest Strk</annotation>
            <traceGroup><annotation type="truth">A</annotation>
              <annotationXML href="A_1"/>
              <traceGroup><annotation type="truth"></annotation>
                <traceView traceDataRef="0"/><traceView traceDataRef="1"/>
              </traceGroup>
            </traceGroup>
            <traceGroup><annotation type="truth">i</annotation>
              <annotationXML href="i_1"/>
              <traceGroup><annotation type="truth"></annotation>
                <traceView traceDataRef="2"/><traceView traceDataRef="3"/>
              </traceGroup>
            </traceGroup>
          </traceGroup>
        </ink>
        """
    )
    sample = _parse_truth_root(nested)
    assert sample.parse_status == "ok"
    assert [group.label for group in sample.groups] == ["A", "i"]
    assert [group.stroke_indices for group in sample.groups] == [(0, 1), (2, 3)]
    invalid_trace = _parse_truth_root(ET.fromstring(
        """
        <ink>
          <trace id="0"></trace>
          <traceGroup><annotation type="truth">x</annotation>
            <traceView traceDataRef="0"/>
          </traceGroup>
        </ink>
        """
    ))
    assert not invalid_trace.source_valid
    assert invalid_trace.invalid_raw_trace_elements == 1
    assert invalid_trace.parse_status == "missing_traces"
    source = _source("nested", sample)
    assert len(source["strokes"][3]["points"]) == 1
    assert source["strokes"][3]["points"][0]["x"] == 5.0
    expected_unsupported = [
        r"\sin", ",", "t", r"\cos", r"\lim", r"\log", r"\tan", ".", "!",
    ]
    assert [_normalise_label(token) for token in expected_unsupported] == expected_unsupported
    boxes = {
        str(index): {
            "left": float(index * 10),
            "right": float(index * 10 + 5),
            "top": 0.0,
            "bottom": 10.0,
        }
        for index in range(8)
    }
    spans = _compose_function_spans(
        ["x", "s", "i", "n", "+", "l", "o", "g"],
        [str(index) for index in range(8)],
        boxes,
        set(),
    )
    assert [span["token"] for span in spans] == ["x", r"\sin", "+", r"\log"]
    assert spans[1]["source_record_ids"] == ["1", "2", "3"]
    blocked = _compose_function_spans(
        ["s", "i", "n"],
        ["1", "2", "3"],
        boxes,
        {"2"},
    )
    assert [span["token"] for span in blocked] == ["s", "i", "n"]
    failed = _evaluate_formula(
        "nested",
        sample,
        {"A", "i"},
        None,
        "ValueError: insufficient_points",
        _canonical_sha256(source),
        "runtime_source_canonical_json",
        True,
    )
    assert failed["inference_status"] == "error"
    assert failed["source_strokes"] == 4
    assert not failed["grouping_exact"]
    assert not failed["strict_group_layout_relation_character_exact"]
    partial_summary = _summarise([failed], full_coverage=False)
    assert partial_summary["evaluated_subset_scores"]["denominator"] == 1
    assert partial_summary["all_1199_denominator_scores"] is None
    duplicate_unsupported = {**failed, "unsupported_truth_tokens": ["t", "t"]}
    duplicate_summary = _summarise(
        [duplicate_unsupported], full_coverage=False,
    )
    assert duplicate_summary["truth"]["unsupported_truth_token_counts"]["t"] == 2
    parse_failure = _source_parse_failure_row(
        "broken", ValueError("bad InkML"), None,
    )
    assert _summarise(
        [parse_failure], full_coverage=False,
    )["inference"]["error"] == 1
    print(json.dumps({"event": "self_test_passed", "schema": SCHEMA}))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--partition-ranker",
        type=Path,
        default=ROOT / "artifacts" / "project_owned_grouping_runtime_20260822_r11_conservative_long_blend_shadow" / "partition_context_ranker.joblib",
    )
    parser.add_argument(
        "--hwr-checkpoint",
        type=Path,
        default=ROOT / "artifacts" / "missing_hwr_checkpoint.pt",
    )
    parser.add_argument(
        "--context-checkpoint",
        type=Path,
        default=ROOT / "artifacts" / "owned_formula_context_expanded_hwr_20260820_r1_shadow" / "owned_formula_context_product.pt",
    )
    parser.add_argument(
        "--crohme",
        type=Path,
        default=ROOT / "datasets" / "30_noncommercial_evaluation" / "crohme2019" / "crohme2019" / "crohme2019" / "test",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "crohme_raw_full_coverage_research_r1",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Run only the first N sorted InkML rows; never use this report as full coverage.",
    )
    parser.add_argument("--self-test", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.self_test:
        _self_test()
        return 0
    if args.limit < 0:
        raise ValueError("--limit must be non-negative")
    paths = {
        "partition_ranker": args.partition_ranker.expanduser().resolve(),
        "hwr_checkpoint": args.hwr_checkpoint.expanduser().resolve(),
        "context_checkpoint": args.context_checkpoint.expanduser().resolve(),
        "crohme": args.crohme.expanduser().resolve(),
        "output": args.output.expanduser().resolve(),
    }
    if paths["output"].exists():
        raise FileExistsError(f"output must be new: {paths['output']}")
    if not paths["crohme"].is_dir():
        raise FileNotFoundError(paths["crohme"])
    for key in ("partition_ranker", "hwr_checkpoint", "context_checkpoint"):
        if not paths[key].is_file():
            raise FileNotFoundError(paths[key])

    inkml_paths = sorted(paths["crohme"].rglob("*.inkml"))
    if not args.limit and len(inkml_paths) != EXPECTED_CROHME_TEST_ROWS:
        raise ValueError(
            f"expected {EXPECTED_CROHME_TEST_ROWS} CROHME test rows, found {len(inkml_paths)}"
        )
    selected_paths = inkml_paths[:args.limit] if args.limit else inkml_paths
    if not selected_paths:
        raise ValueError("no CROHME InkML rows selected")
    record_ids = [path.relative_to(paths["crohme"]).as_posix() for path in selected_paths]
    if len(record_ids) != len(set(record_ids)):
        raise AssertionError("CROHME record IDs are not unique")

    torch.set_grad_enabled(False)
    if torch.is_grad_enabled():
        raise AssertionError("CROHME validation must run with autograd disabled")
    hwr_payload = torch.load(paths["hwr_checkpoint"], map_location="cpu", weights_only=False)
    hwr_policy = dict(hwr_payload.get("report", {}).get("data_policy", {}))
    if hwr_policy.get("project_owned_formula_grouping_training") is not False:
        raise ValueError("HWR training provenance boundary is missing")
    context_payload = torch.load(
        paths["context_checkpoint"], map_location="cpu", weights_only=False
    )
    context_training = str(context_payload.get("training_data", ""))
    if "crohme" in context_training.casefold() or "project-owned" not in context_training:
        raise ValueError("context checkpoint training provenance is invalid")
    runtime = RawFormulaContextRuntimeV1.from_artifacts(
        paths["partition_ranker"],
        paths["hwr_checkpoint"],
        paths["context_checkpoint"],
        device=args.device,
        emit_formula_layout_shadow=True,
        allow_posthoc_shadow=True,
    )
    runtime_labels = set(runtime.labels)
    route_contract = _grouping_route_config(runtime)

    rows: list[dict[str, Any]] = []
    for index, path in enumerate(selected_paths, 1):
        formula_id = path.relative_to(paths["crohme"]).as_posix()
        try:
            sample = _parse_inkml(path)
        except Exception as error:
            # Preserve a row even if the source itself cannot be decoded.  There
            # is no safe way to supply corrupted raw ink to runtime inference.
            rows.append(_source_parse_failure_row(
                formula_id, error, _safe_file_sha256(path),
            ))
            continue
        if not sample.source_valid:
            rows.append(_evaluate_formula(
                formula_id,
                sample,
                runtime_labels,
                None,
                (
                    "SourceValidationError: "
                    f"invalid_trace_elements={sample.invalid_raw_trace_elements};"
                    f"duplicate_trace_ids={sample.duplicate_trace_ids}"
                ),
                _safe_file_sha256(path),
                "inkml_bytes",
                None,
            ))
            continue
        source = _source(formula_id, sample)
        source_hash = _canonical_sha256(source)
        result = None
        inference_error = None
        with _apply_grouping_route(runtime, len(sample.strokes)) as route_audit:
            try:
                result = runtime.infer(source)
            except Exception as error:
                inference_error = _safe_error(error)
        raw_source_unchanged = source_hash == _canonical_sha256(source)
        if not raw_source_unchanged:
            raise AssertionError(f"runtime mutated raw input: {formula_id}")
        evaluated = _evaluate_formula(
            formula_id,
            sample,
            runtime_labels,
            result,
            inference_error,
            source_hash,
            "runtime_source_canonical_json",
            raw_source_unchanged,
        )
        evaluated["grouping_route"] = route_audit
        rows.append(evaluated)
        if index % 25 == 0 or index == len(selected_paths):
            print(json.dumps({
                "event": "crohme_raw_full_coverage_progress",
                "completed": index,
                "total": len(selected_paths),
                "inference_ok": sum(row["inference_status"] == "ok" for row in rows),
                "strict_evaluated_subset_count": sum(
                    row["strict_group_layout_relation_character_exact"]
                    for row in rows
                ),
            }, ensure_ascii=False), flush=True)

    if len(rows) != len(selected_paths):
        raise AssertionError("an InkML input was omitted")
    if len({str(row["formula_id"]) for row in rows}) != len(rows):
        raise AssertionError("prediction rows must have unique formula IDs")
    successful = [row for row in rows if row["inference_status"] == "ok"]
    if any(row["raw_source_unchanged"] is not True for row in successful):
        raise AssertionError("runtime source was not verified immutable")
    if any(not row["all_strokes_exactly_once"] for row in successful):
        raise AssertionError("a successful runtime output lost or duplicated a stroke")

    paths["output"].mkdir(parents=True)
    predictions = paths["output"] / "formula_results.jsonl.gz"
    _write_rows(predictions, rows)
    full_coverage = (
        not args.limit
        and len(rows) == EXPECTED_CROHME_TEST_ROWS
        and len(inkml_paths) == EXPECTED_CROHME_TEST_ROWS
    )
    summary = _summarise(rows, full_coverage=full_coverage)
    report = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "noncommercial_validation_only_posthoc_shadow",
        "training_performed": False,
        "selection_performed": False,
        "threshold_tuning_performed": False,
        "crohme_rows_used_for_training": 0,
        "crohme_gradient_updates": 0,
        "official_lgeval_or_symlg": False,
        "score_boundary": (
            "Local raw grouping/layout/relation proxy only; do not call any "
            "reported score CROHME Expression Rate."
        ),
        "coverage": {
            "discovered_raw_inkml_rows": len(inkml_paths),
            "evaluated_formula_rows": len(rows),
            "expected_full_test_rows": EXPECTED_CROHME_TEST_ROWS,
            "limited_run": bool(args.limit),
            "unique_formula_ids": len({str(row["formula_id"]) for row in rows}),
            "omitted_selected_formula_rows": len(selected_paths) - len(rows),
            "unselected_raw_inkml_rows": len(inkml_paths) - len(selected_paths),
            "full_coverage_verified": full_coverage,
        },
        "summary": summary,
        "historical_non_comparable_reference": {
            "name": "oracle_group_cache_proxy_full_1199",
            "count": 343,
            "denominator": 1199,
            "score": 343 / 1199,
            "source": "reports/CROHME_STANDARD_REINFORCEMENT_LOOP_20260821.md",
            "comparison_rule": (
                "This uses truth groups/cached character candidates and is not "
                "comparable to the raw full-coverage strict score."
            ),
        },
        "contracts": {
            "one_result_per_selected_raw_inkml": True,
            "all_selected_input_rows_retained": len(rows) == len(selected_paths),
            "truth_not_supplied_to_runtime": True,
            "runtime_source_fields": ["formula_id", "strokes"],
            "runtime_source_unchanged_rows": sum(
                row["raw_source_unchanged"] is True for row in rows
            ),
            "successful_runtime_exact_stroke_cover_rows": sum(
                row["inference_status"] == "ok"
                and row["all_strokes_exactly_once"]
                for row in rows
            ),
            "product_default_enabled": False,
            "posthoc_shadow_opt_in": True,
            "autograd_enabled": False,
        },
        "limits": [
            "No symLG writer or official LgEval is installed.",
            "A source/runtime parse failure is retained as a scored failure, not excluded.",
            "Function span composition is an output diagnostic only; it does not alter grouping, runtime tokens, or product behavior.",
            "The frozen ranker is explicitly shadow-only and posthoc-tuned on prior project-owned data.",
        ],
        "inputs": {
            "partition_ranker": str(paths["partition_ranker"]),
            "partition_ranker_sha256": _sha256(paths["partition_ranker"]),
            "hwr_checkpoint": str(paths["hwr_checkpoint"]),
            "hwr_checkpoint_sha256": _sha256(paths["hwr_checkpoint"]),
            "context_checkpoint": str(paths["context_checkpoint"]),
            "context_checkpoint_sha256": _sha256(paths["context_checkpoint"]),
            "context_training_data": context_training,
            "crohme": str(paths["crohme"]),
            "frozen_artifact_grouping_route_adapter": {
                **route_contract,
                "mechanism": (
                    "replays artifact-owned long-formula blend and group bias "
                    "through the clean runtime without modifying its weights"
                ),
            },
        },
        "output": {
            "formula_results": str(predictions),
            "formula_results_sha256": _sha256(predictions),
        },
    }
    report_path = paths["output"] / "validation_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "event": "crohme_raw_full_coverage_finished",
        "report": str(report_path),
        "summary": (
            report["summary"]["all_1199_denominator_scores"]
            or report["summary"]["evaluated_subset_scores"]
        ),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
