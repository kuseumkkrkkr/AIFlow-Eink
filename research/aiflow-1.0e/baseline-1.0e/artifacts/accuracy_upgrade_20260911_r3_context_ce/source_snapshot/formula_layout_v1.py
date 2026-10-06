#!/usr/bin/env python3
"""Infer a conservative formula layout over immutable HWR symbol boxes.

The layer ignores incoming ``context.index`` values. It only reorders existing
boxes and emits spatial relations; it cannot create/delete a glyph or change
stroke ownership. Thresholds are hand-set dimensionless geometry defaults and
were not selected on CROHME or another noncommercial dataset.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
from statistics import median
from typing import Iterable


RELATIONS = frozenset({
    "right", "left", "above", "below", "superscript", "subscript", "overlap",
})
STRUCTURAL = frozenset({"above", "below", "superscript", "subscript", "contains"})
FORMULA_OUTPUT_SCHEMA = "aiflow-formula-layout-finalized/v1"

# These are semantic abstention rules, not character replacements. A strong
# opening fence is a poor script base, while relation/infix operators are poor
# isolated script children. The lower equality threshold covers visually
# confusable multi-bar HWR candidates without depending on finalized Top-1.
OPENING_FENCES = frozenset({"(", "[", r"\{", r"\langle", r"\lceil", r"\lfloor"})
CLOSING_FENCES = frozenset({")", "]", r"\}", r"\rangle", r"\rceil", r"\rfloor"})
FENCE_PAIRS = {
    "(": ")", "[": "]", r"\{": r"\}", r"\langle": r"\rangle",
    r"\lceil": r"\rceil", r"\lfloor": r"\rfloor",
}
EQUALITY_FAMILY = frozenset({
    "=", r"\approx", r"\asymp", r"\doteq", r"\equiv", r"\neq",
    r"\rightleftharpoons", r"\rightrightarrows", r"\simeq",
})
INFIX_SCRIPT_CHILDREN = frozenset({
    "+", "/", r"\ast", r"\cdot", r"\div", r"\mp", r"\pm", r"\times",
})


@dataclass(frozen=True, slots=True)
class LayoutConfig:
    fraction_min_width_height: float = 4.0
    fraction_min_width_reference: float = 0.75
    fraction_min_x_overlap: float = 0.25
    fraction_max_vertical_gap: float = 1.5
    script_min_height_ratio: float = 0.15
    script_max_height_ratio: float = 0.78
    script_min_vertical_shift: float = 0.25
    script_max_horizontal_gap: float = 0.75
    structural_candidate_floor: float = 0.25
    root_inner_left_ratio: float = 0.05
    root_parenthesized_gap_ratio: float = 0.35
    opening_fence_script_veto_floor: float = 0.75
    equality_script_child_veto_floor: float = 0.30
    infix_script_child_veto_floor: float = 0.75


def _box(row: dict) -> dict[str, float]:
    geometry = row.get("geometry") or {}
    required = ("left", "top", "right", "bottom")
    if any(key not in geometry for key in required):
        raise ValueError(f"layout geometry missing for {row.get('record_id')}")
    left, top, right, bottom = (float(geometry[key]) for key in required)
    if not all(math.isfinite(value) for value in (left, top, right, bottom)):
        raise ValueError(f"non-finite layout geometry for {row.get('record_id')}")
    if right < left or bottom < top:
        raise ValueError(f"inverted layout geometry for {row.get('record_id')}")
    width, height = max(right - left, 1e-6), max(bottom - top, 1e-6)
    return {
        "left": left, "top": top, "right": right, "bottom": bottom,
        "width": width, "height": height,
        "cx": (left + right) / 2.0, "cy": (top + bottom) / 2.0,
    }


def _scores(row: dict) -> dict[str, float]:
    candidates = [str(value) for value in row.get("final_topk") or []]
    probabilities = [float(value) for value in row.get("final_topk_probabilities") or []]
    if not candidates or len(candidates) != len(probabilities):
        raise ValueError(f"invalid layout candidates for {row.get('record_id')}")
    return dict(zip(candidates, probabilities, strict=True))


def selected_layout_evidence_rows(
    rows: list[dict], predictions: dict[str, str],
) -> tuple[list[dict], dict]:
    """Copy upstream finalized tokens into layout-only semantic evidence."""
    input_ids = {str(row.get("record_id", "")) for row in rows}
    if "" in input_ids or len(input_ids) != len(rows) or set(predictions) != input_ids:
        raise ValueError("selected layout evidence prediction coverage mismatch")
    output = []
    candidate_extensions = selected_semantic_promotions = 0
    for source in rows:
        row = {**source}
        record_id = str(row["record_id"])
        candidates = [str(value) for value in row.get("final_topk") or []]
        probabilities = [float(value) for value in row.get("final_topk_probabilities") or []]
        if not candidates or len(candidates) != len(probabilities):
            raise ValueError(f"invalid selected layout candidates for {record_id}")
        token = str(predictions[record_id])
        if token not in candidates:
            candidates.append(token)
            probabilities.append(1.0)
            candidate_extensions += 1
            selected_semantic_promotions += 1
        else:
            selected_index = candidates.index(token)
            if probabilities[selected_index] < 1.0:
                probabilities[selected_index] = 1.0
                selected_semantic_promotions += 1
        row["final_topk"] = candidates
        row["final_topk_probabilities"] = probabilities
        output.append(row)
    return output, {
        "candidate_extensions": candidate_extensions,
        "candidate_extension_policy": "upstream finalized token only",
        "selected_semantic_promotions": selected_semantic_promotions,
        "selected_semantic_policy": (
            "upstream finalized token receives layout-only unit evidence"
        ),
    }


def _reference_height(boxes: list[dict[str, float]]) -> float:
    maximum = max(box["height"] for box in boxes)
    body = [box["height"] for box in boxes if box["height"] >= maximum * 0.25]
    return max(float(median(body)), 1e-6)


def _x_overlap(first: dict[str, float], second: dict[str, float]) -> float:
    overlap = max(0.0, min(first["right"], second["right"]) - max(first["left"], second["left"]))
    return overlap / max(min(first["width"], second["width"]), 1e-6)


def _score_any(scores: dict[str, float], labels: Iterable[str]) -> float:
    return max((scores.get(label, 0.0) for label in labels), default=0.0)


def _script_semantic_veto(
    parent_scores: dict[str, float], child_scores: dict[str, float],
    config: LayoutConfig,
) -> bool:
    return (
        _score_any(parent_scores, OPENING_FENCES)
        >= config.opening_fence_script_veto_floor
        or _score_any(child_scores, EQUALITY_FAMILY)
        >= config.equality_script_child_veto_floor
        or _score_any(child_scores, INFIX_SCRIPT_CHILDREN)
        >= config.infix_script_child_veto_floor
    )


def _edge(parent: dict, child: dict, relation: str, confidence: float, source: str) -> dict:
    return {
        "parent": str(parent["record_id"]), "child": str(child["record_id"]),
        "type": relation, "confidence": float(confidence), "source": source,
    }


def infer_formula_layout(
    rows: list[dict], config: LayoutConfig | None = None, *, script_predictor=None,
) -> dict:
    """Return geometry-only order and structural edges for one formula."""
    if not rows:
        raise ValueError("formula layout requires at least one row")
    formula_ids = {str(row.get("formula_id", "")) for row in rows}
    record_ids = [str(row.get("record_id", "")) for row in rows]
    if len(formula_ids) != 1 or "" in formula_ids:
        raise ValueError("formula layout rows must share one non-empty formula_id")
    if "" in record_ids or len(set(record_ids)) != len(record_ids):
        raise ValueError("formula layout record_ids must be unique and non-empty")
    layout = config or LayoutConfig()
    boxes = [_box(row) for row in rows]
    scores = [_scores(row) for row in rows]
    reference = _reference_height(boxes)
    assigned: set[int] = set()
    structural_parents: set[int] = set()
    region_memberships: dict[int, set[tuple[int, str]]] = defaultdict(set)
    parent_by_child: dict[int, int] = {}
    edge_keys: set[tuple[int, int, str]] = set()
    edges: list[dict] = []

    def append_edge(
        parent: int, child: int, relation: str, confidence: float, source: str,
        *, exclusive_child: bool = True,
    ) -> bool:
        key = (parent, child, relation)
        if parent == child or key in edge_keys:
            return False
        if exclusive_child:
            if child in parent_by_child:
                return False
            ancestor = parent
            while ancestor in parent_by_child:
                ancestor = parent_by_child[ancestor]
                if ancestor == child:
                    return False
            parent_by_child[child] = parent
            assigned.add(child)
        else:
            region_memberships[child].add((parent, relation))
        edge_keys.add(key)
        edges.append(_edge(rows[parent], rows[child], relation, confidence, source))
        return True

    # Complete numerator+denominator geometry is required, so a plain minus is
    # never promoted to a fraction bar by its shape alone.
    for parent, bar in sorted(enumerate(boxes), key=lambda item: -item[1]["width"]):
        if (
            bar["width"] < reference * layout.fraction_min_width_reference
            or bar["width"] < bar["height"] * layout.fraction_min_width_height
        ):
            continue
        above, below = [], []
        for child, box in enumerate(boxes):
            # Wider bars are visited first. A smaller nested bar may belong to
            # the wider region, but must not claim an already-seen outer bar
            # back and create an inverse fraction cycle.
            if child == parent or child in structural_parents:
                continue
            overlap = _x_overlap(bar, box)
            if overlap < layout.fraction_min_x_overlap:
                continue
            if box["bottom"] <= bar["cy"] and bar["top"] - box["bottom"] <= reference * layout.fraction_max_vertical_gap:
                above.append((child, overlap))
            elif box["top"] >= bar["cy"] and box["top"] - bar["bottom"] <= reference * layout.fraction_max_vertical_gap:
                below.append((child, overlap))
        if not above or not below:
            continue
        structural_parents.add(parent)
        for relation, children in (("above", above), ("below", below)):
            for child, overlap in children:
                append_edge(
                    parent, child, relation,
                    min(0.95, 0.80 + 0.15 * overlap), "fraction_geometry",
                    exclusive_child=False,
                )

    # Root containment requires independent Top-k evidence; geometry alone is
    # intentionally insufficient for this visually ambiguous relation.
    seen_roots: set[int] = set()
    root_rows = zip(range(len(rows)), boxes, scores, strict=True)
    for parent, root, candidates in sorted(root_rows, key=lambda item: -item[1]["width"]):
        evidence = _score_any(candidates, (r"\sqrt", r"\sqrt{}"))
        if evidence < layout.structural_candidate_floor:
            continue
        inside_children = []
        for child, box in enumerate(boxes):
            # Nested radicals are ordered by enclosing width. The inner root
            # may be a member of the outer one, but never the inverse.
            if child == parent or child in seen_roots:
                continue
            inside = (
                root["left"] + root["width"] * layout.root_inner_left_ratio
                <= box["cx"] <= root["right"] + reference * 0.20
                and root["top"] <= box["cy"] <= root["bottom"]
            )
            if inside:
                inside_children.append(child)

        # Handwritten radicals often stop their overbar just before a complete
        # parenthesized radicand. A strong, immediately adjacent matching fence
        # pair supplies the missing semantic span without extending past it.
        for opening, closing in FENCE_PAIRS.items():
            opening_children = [
                child for child, box in enumerate(boxes)
                if child != parent and child not in seen_roots
                and _score_any(scores[child], (opening,))
                >= layout.opening_fence_script_veto_floor
                and box["cx"] > root["cx"]
                and box["right"] >= root["left"]
                and max(0.0, box["left"] - root["right"])
                <= reference * layout.root_parenthesized_gap_ratio
            ]
            if not opening_children:
                continue
            open_child = min(opening_children, key=lambda child: boxes[child]["left"])
            closing_children = [
                child for child, box in enumerate(boxes)
                if child != parent and child not in seen_roots
                and box["cx"] > boxes[open_child]["cx"]
                and _score_any(scores[child], (closing,))
                >= layout.opening_fence_script_veto_floor
            ]
            if not closing_children:
                continue
            close_child = min(closing_children, key=lambda child: boxes[child]["left"])
            span_left, span_right = boxes[open_child]["left"], boxes[close_child]["right"]
            inside_children.extend(
                child for child, box in enumerate(boxes)
                if child != parent and child not in seen_roots
                and span_left <= box["cx"] <= span_right
            )
            break
        inside_children = sorted(set(inside_children))

        # A lone/unclosed opening fence next to a radical is more safely read
        # as a baseline expression (for example ``sqrt (a)``) than as a
        # partially captured radicand. Complete parenthesized radicands remain
        # eligible when both fence sides are geometrically contained.
        has_open = any(
            _score_any(scores[child], OPENING_FENCES)
            >= layout.opening_fence_script_veto_floor
            for child in inside_children
        )
        has_close = any(
            _score_any(scores[child], CLOSING_FENCES)
            >= layout.opening_fence_script_veto_floor
            for child in inside_children
        )
        if has_open and not has_close:
            inside_children = []
        if inside_children:
            structural_parents.add(parent)
        for child in inside_children:
            append_edge(
                parent, child, "contains",
                min(0.95, 0.70 + 0.25 * evidence), "root_topk_geometry",
                exclusive_child=False,
            )
        seen_roots.add(parent)

    if script_predictor is not None:
        for parent, child, relation, confidence in script_predictor.propose(
            rows, boxes, reference, assigned, structural_parents,
        ):
            if _script_semantic_veto(scores[parent], scores[child], layout):
                continue
            append_edge(parent, child, relation, confidence, "script_network_v1")
    else:
        # Remaining small displaced boxes may be scripts of the nearest box on
        # the left. Region signatures keep numerator, denominator, and radical
        # contents separate while still allowing nested scripts such as x^2.
        for child, box in sorted(enumerate(boxes), key=lambda item: (item[1]["left"], item[1]["top"])):
            if child in assigned or child in structural_parents:
                continue
            signature = frozenset(region_memberships.get(child, ()))

            def compatible_region(parent: int) -> bool:
                parent_signature = frozenset(region_memberships.get(parent, ()))
                if parent_signature == signature:
                    return True
                # A closing fence captured by a radical may carry a script just
                # beyond the handwritten overbar. Keep that script in the same
                # radical region; fraction regions and non-fence bases remain
                # strictly isolated.
                return bool(
                    not signature
                    and parent_signature
                    and all(relation == "contains" for _, relation in parent_signature)
                    and _score_any(scores[parent], CLOSING_FENCES)
                    >= layout.opening_fence_script_veto_floor
                )

            parents = [
                parent for parent, base in enumerate(boxes)
                if parent != child and parent not in structural_parents
                and parent not in assigned
                and base["cx"] < box["cx"]
                and base["height"] >= reference * 0.45
                and compatible_region(parent)
                and max(0.0, box["left"] - base["right"]) <= reference * layout.script_max_horizontal_gap
            ]
            if not parents:
                continue
            parent = max(parents, key=lambda index: boxes[index]["right"])
            base = boxes[parent]
            if _script_semantic_veto(scores[parent], scores[child], layout):
                continue
            ratio = box["height"] / max(base["height"], 1e-6)
            maximum_ratio = 1.25 if signature else layout.script_max_height_ratio
            if not layout.script_min_height_ratio <= ratio <= maximum_ratio:
                continue
            shift = base["height"] * layout.script_min_vertical_shift
            if box["cy"] <= base["cy"] - shift and box["bottom"] <= base["cy"] + base["height"] * 0.05:
                relation = "superscript"
            elif box["cy"] >= base["cy"] + shift and box["top"] >= base["cy"] - base["height"] * 0.05:
                relation = "subscript"
            else:
                continue
            inherited_regions = frozenset(region_memberships.get(parent, ())) - signature
            if not append_edge(parent, child, relation, 0.80, "script_geometry"):
                continue
            for region_parent, region_relation in inherited_regions:
                append_edge(
                    region_parent, child, region_relation, 0.78,
                    "script_region_closure", exclusive_child=False,
                )
            # A script is a span, not an isolated pair. Tiny punctuation such
            # as the minus in e^{-n} may fail the size test but is accepted only
            # when bracketed between the confirmed base and script endpoint.
            for bridge, middle in enumerate(boxes):
                if (
                    bridge == parent or bridge == child or bridge in assigned
                    or bridge in structural_parents
                    or frozenset(region_memberships.get(bridge, ())) != signature
                    or not base["cx"] < middle["cx"] < box["cx"]
                ):
                    continue
                same_side = (
                    middle["bottom"] <= base["cy"] + base["height"] * 0.05
                    if relation == "superscript"
                    else middle["top"] >= base["cy"] - base["height"] * 0.05
                )
                if same_side:
                    if _script_semantic_veto(scores[parent], scores[bridge], layout):
                        continue
                    append_edge(
                        parent, bridge, relation, 0.78,
                        "script_span_geometry",
                    )

    row_by_id = {str(row["record_id"]): row for row in rows}
    box_by_id = {str(row["record_id"]): box for row, box in zip(rows, boxes, strict=True)}
    children: dict[str, list[dict]] = defaultdict(list)
    structural_children = set()
    for edge in edges:
        children[edge["parent"]].append(edge)
        structural_children.add(edge["child"])

    def position(record_id: str) -> tuple[float, float, str]:
        box = box_by_id[record_id]
        return box["cx"], box["cy"], record_id

    ordered_ids: list[str] = []
    active: set[str] = set()

    def visit(record_id: str) -> None:
        if record_id in active:
            raise ValueError(
                f"formula layout relation cycle at {record_id}: {sorted(active)}"
            )
        if record_id in ordered_ids:
            return
        active.add(record_id)
        ordered_ids.append(record_id)
        priority = {"contains": 0, "above": 1, "below": 2, "superscript": 3, "subscript": 4}
        for edge in sorted(children.get(record_id, []), key=lambda value: (priority[value["type"]], position(value["child"]))):
            visit(edge["child"])
        active.remove(record_id)

    roots = sorted((record_id for record_id in record_ids if record_id not in structural_children), key=position)
    for record_id in roots:
        visit(record_id)
    for record_id in sorted(record_ids, key=position):
        visit(record_id)
    if set(ordered_ids) != set(record_ids) or len(ordered_ids) != len(record_ids):
        raise AssertionError("formula layout coverage mismatch")
    return {
        "formula_id": next(iter(formula_ids)),
        "ordered_record_ids": ordered_ids,
        "edges": sorted(edges, key=lambda value: (value["child"], value["parent"], value["type"])),
        "relation_counts": dict(sorted(Counter(edge["type"] for edge in edges).items())),
        "row_by_id": row_by_id,
    }


def _spatial_relation(left: dict, right: dict) -> str:
    first, second = _box(left), _box(right)
    dx, dy = second["cx"] - first["cx"], second["cy"] - first["cy"]
    width = max((first["width"] + second["width"]) / 2.0, 1e-6)
    height = max((first["height"] + second["height"]) / 2.0, 1e-6)
    if abs(dx) <= 0.25 * width and abs(dy) <= 0.25 * height:
        return "overlap"
    if abs(dx) <= 0.60 * width and abs(dy) >= 0.55 * height:
        return "above" if dy < 0.0 else "below"
    if dx >= -0.10 * width and dy <= -0.45 * height:
        return "superscript"
    if dx >= -0.10 * width and dy >= 0.45 * height:
        return "subscript"
    return "right" if dx >= 0.0 else "left"


def recontextualize_formula_rows(
    rows: list[dict], config: LayoutConfig | None = None, *, script_predictor=None,
) -> tuple[list[dict], dict]:
    """Replace any supplied sequence context with geometry-derived context."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for source in rows:
        grouped[str(source.get("formula_id", ""))].append(source)
    output = [{**row} for row in rows]
    by_id = {str(row["record_id"]): row for row in output}
    relation_counts: Counter[str] = Counter()
    edge_count = 0
    for formula_id, sequence in grouped.items():
        result = infer_formula_layout(sequence, config, script_predictor=script_predictor)
        ordered = [by_id[record_id] for record_id in result["ordered_record_ids"]]
        edge_lookup = {(edge["parent"], edge["child"]): edge["type"] for edge in result["edges"]}
        relation_counts.update(result["relation_counts"])
        edge_count += len(result["edges"])
        for index, row in enumerate(ordered):
            previous = ordered[index - 1] if index else None
            following = ordered[index + 1] if index + 1 < len(ordered) else None
            relation = None if previous is None else edge_lookup.get(
                (str(previous["record_id"]), str(row["record_id"])),
                _spatial_relation(previous, row),
            )
            if relation is not None and relation not in RELATIONS:
                # ``contains`` is structural output; the existing context model
                # has no contains token, so its conservative fallback is overlap.
                relation = "overlap"
            row["context"] = {
                "index": index, "length": len(ordered),
                "previous_top1": str(previous["final_topk"][0]) if previous else "<S>",
                "next_top1": str(following["final_topk"][0]) if following else "</S>",
                "previous_dx": float(row["geometry"]["center_x"] - previous["geometry"]["center_x"]) if previous else 0.0,
                "previous_dy": float(row["geometry"]["center_y"] - previous["geometry"]["center_y"]) if previous else 0.0,
                "next_dx": float(following["geometry"]["center_x"] - row["geometry"]["center_x"]) if following else 0.0,
                "next_dy": float(following["geometry"]["center_y"] - row["geometry"]["center_y"]) if following else 0.0,
                "relation_from_previous": relation,
                "source": "formula_layout_v1",
            }
    return output, {
        "enabled": True,
        "policy": "geometry-only ordering and relations over immutable HWR boxes",
        "formulas": len(grouped), "records": len(rows), "structural_edges": edge_count,
        "relation_counts": dict(sorted(relation_counts.items())),
        "candidate_preservation_rate": 1.0, "new_tokens": 0,
        "deleted_glyphs": 0, "grouping_mutations": 0,
        "threshold_provenance": "hand-set dimensionless defaults; no dataset fitting",
        "script_relation_source": (
            "formula_script_network_v1" if script_predictor is not None
            else "conservative_geometry"
        ),
    }


def serialize_formula(
    rows: list[dict], predictions: dict[str, str] | None = None,
    *, script_predictor=None,
) -> str:
    """Serialize the inferred structural graph without changing candidates."""
    result = infer_formula_layout(rows, script_predictor=script_predictor)
    row_by_id = result["row_by_id"]
    labels = {
        record_id: str((predictions or {}).get(record_id, row["final_topk"][0]))
        for record_id, row in row_by_id.items()
    }
    children: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    structural_children = set()
    for edge in result["edges"]:
        children[edge["parent"]][edge["type"]].append(edge["child"])
        structural_children.add(edge["child"])

    def left(record_id: str) -> tuple[float, str]:
        return float(row_by_id[record_id]["geometry"]["left"]), record_id

    emitted: set[str] = set()

    def sequence(record_ids: Iterable[str], active: frozenset[str]) -> str:
        return "".join(node(record_id, active) for record_id in sorted(set(record_ids), key=left))

    def node(record_id: str, active: frozenset[str]) -> str:
        if record_id in active:
            raise ValueError("formula layout serialization cycle")
        if record_id in emitted:
            return ""
        emitted.add(record_id)
        nested = active | {record_id}
        slots = children.get(record_id, {})
        above, below, contained = slots.get("above", []), slots.get("below", []), slots.get("contains", [])
        if above and below:
            base = rf"\frac{{{sequence(above, nested)}}}{{{sequence(below, nested)}}}"
        elif contained:
            base = rf"\sqrt{{{sequence(contained, nested)}}}"
        else:
            base = labels[record_id]
        if slots.get("subscript"):
            base += rf"_{{{sequence(slots['subscript'], nested)}}}"
        if slots.get("superscript"):
            base += rf"^{{{sequence(slots['superscript'], nested)}}}"
        return base

    roots = [record_id for record_id in row_by_id if record_id not in structural_children]
    return sequence(roots, frozenset())


def finalize_formula_outputs(
    rows: list[dict], predictions: dict[str, str], *, script_predictor=None,
) -> tuple[list[dict], dict]:
    """Emit one shadow formula result per immutable candidate grouping."""
    input_ids = {str(row.get("record_id", "")) for row in rows}
    if "" in input_ids or len(input_ids) != len(rows):
        raise ValueError("formula output record_ids must be unique and non-empty")
    if set(predictions) != input_ids:
        raise ValueError("formula output prediction coverage mismatch")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        formula_id = str(row.get("formula_id", ""))
        if not formula_id:
            raise ValueError("formula output formula_id must be non-empty")
        record_id = str(row["record_id"])
        token = str(predictions[record_id])
        if token not in [str(value) for value in row.get("final_topk") or []]:
            raise ValueError(f"formula output token is outside HWR candidates: {record_id}")
        grouped[formula_id].append(row)

    output = []
    relation_counts: Counter[str] = Counter()
    formulas_with_multi_parent_nodes = 0
    multi_parent_node_count = 0
    for formula_id in sorted(grouped):
        sequence = grouped[formula_id]
        layout = infer_formula_layout(sequence, script_predictor=script_predictor)
        ordered = list(layout["ordered_record_ids"])
        relation_counts.update(layout["relation_counts"])
        parents_by_child: dict[str, list[dict]] = defaultdict(list)
        for edge in layout["edges"]:
            parents_by_child[str(edge["child"])].append({
                "parent_record_id": str(edge["parent"]),
                "relation": str(edge["type"]),
            })
        multi_parent_nodes = []
        for record_id, parent_relations in sorted(parents_by_child.items()):
            if len({value["parent_record_id"] for value in parent_relations}) <= 1:
                continue
            multi_parent_nodes.append({
                "record_id": record_id,
                "parent_relations": sorted(
                    parent_relations,
                    key=lambda value: (value["parent_record_id"], value["relation"]),
                ),
            })
        formulas_with_multi_parent_nodes += bool(multi_parent_nodes)
        multi_parent_node_count += len(multi_parent_nodes)
        output.append({
            "schema": FORMULA_OUTPUT_SCHEMA,
            "status": "shadow_runtime_only",
            "formula_id": formula_id,
            "latex": serialize_formula(
                sequence, predictions, script_predictor=script_predictor,
            ),
            "latex_status": (
                "shadow_ambiguous_multi_parent"
                if multi_parent_nodes else "shadow_single_parent_graph"
            ),
            "latex_authoritative": False,
            "relation_graph_semantics": "transitive_nonexclusive",
            "multi_parent_nodes": multi_parent_nodes,
            "ordered_record_ids": ordered,
            "ordered_tokens": [str(predictions[record_id]) for record_id in ordered],
            "relations": list(layout["edges"]),
            "candidate_preservation_rate": 1.0,
            "new_tokens": 0,
            "deleted_glyphs": 0,
            "grouping_mutations": 0,
        })
    return output, {
        "schema": FORMULA_OUTPUT_SCHEMA,
        "status": "shadow_runtime_only",
        "formulas": len(output),
        "records": len(rows),
        "relation_counts": dict(sorted(relation_counts.items())),
        "relation_graph_semantics": "transitive_nonexclusive",
        "formulas_with_multi_parent_nodes": formulas_with_multi_parent_nodes,
        "multi_parent_nodes": multi_parent_node_count,
        "latex_authoritative": False,
        "candidate_preservation_rate": 1.0,
        "new_tokens": 0,
        "deleted_glyphs": 0,
        "grouping_mutations": 0,
    }


def _sample(record_id: str, left: float, top: float, right: float, bottom: float, token: str = "x") -> dict:
    return {
        "record_id": record_id, "formula_id": "f",
        "final_topk": [token], "final_topk_probabilities": [1.0],
        "geometry": {
            "left": left, "top": top, "right": right, "bottom": bottom,
            "center_x": (left + right) / 2.0, "center_y": (top + bottom) / 2.0,
        },
    }


def _self_test() -> None:
    flat = [_sample("b", 20, 0, 30, 20, "+"), _sample("a", 0, 0, 10, 20, "1")]
    ordered = infer_formula_layout(flat)["ordered_record_ids"]
    assert ordered == ["a", "b"]
    scripted = [_sample("x", 0, 10, 20, 40), _sample("2", 21, 0, 29, 14, "2")]
    layout = infer_formula_layout(scripted)
    assert [(edge["parent"], edge["child"], edge["type"]) for edge in layout["edges"]] == [("x", "2", "superscript")]
    assert not infer_formula_layout([
        _sample("open", 0, 10, 20, 40, "("),
        _sample("y", 21, 30, 29, 44, "y"),
    ])["edges"]
    assert not infer_formula_layout([
        _sample("x", 0, 10, 20, 40, "x"),
        _sample("eq", 21, 0, 29, 14, "="),
    ])["edges"]
    assert not infer_formula_layout([
        _sample("root", 0, 0, 40, 40, r"\sqrt{}"),
        _sample("open", 10, 10, 18, 30, "("),
    ])["edges"]
    parenthesized_root = [
        _sample("root", 0, 0, 40, 40, r"\sqrt{}"),
        _sample("open", 41, 2, 48, 38, "("),
        _sample("value", 50, 10, 65, 30, "a"),
        _sample("close", 67, 2, 74, 38, ")"),
    ]
    assert {
        (edge["parent"], edge["child"], edge["type"])
        for edge in infer_formula_layout(parenthesized_root)["edges"]
    } == {
        ("root", "open", "contains"),
        ("root", "value", "contains"),
        ("root", "close", "contains"),
    }
    assert serialize_formula(parenthesized_root) == r"\sqrt{(a)}"
    parenthesized_root_script = [
        *parenthesized_root,
        _sample("power", 72, -2, 79, 10, "2"),
    ]
    root_script_edges = {
        (edge["parent"], edge["child"], edge["type"])
        for edge in infer_formula_layout(parenthesized_root_script)["edges"]
    }
    assert ("close", "power", "superscript") in root_script_edges
    assert ("root", "power", "contains") in root_script_edges
    assert serialize_formula(parenthesized_root_script) == r"\sqrt{(a)^{2}}"
    contextual, audit = recontextualize_formula_rows(list(reversed(scripted)))
    assert sorted((row["record_id"], row["context"]["index"]) for row in contextual) == [("2", 1), ("x", 0)]
    assert serialize_formula(scripted) == "x^{2}" and audit["grouping_mutations"] == 0
    fraction = [
        _sample("bar", 0, 20, 40, 22, "-"),
        _sample("num", 10, 0, 20, 12, "1"),
        _sample("den", 10, 30, 20, 42, "2"),
    ]
    assert serialize_formula(fraction) == r"\frac{1}{2}"
    formulae, formula_audit = finalize_formula_outputs(
        fraction, {"bar": "-", "num": "1", "den": "2"},
    )
    assert formulae[0]["latex"] == r"\frac{1}{2}"
    assert formulae[0]["latex_status"] == "shadow_single_parent_graph"
    assert formula_audit["multi_parent_nodes"] == 0
    assert formula_audit["candidate_preservation_rate"] == 1.0
    nested_fraction = [
        _sample("outer", 0, 20, 100, 22, "-"),
        _sample("inner", 10, 10, 50, 12, "-"),
        _sample("nested_num", 20, 0, 30, 8, "1"),
        _sample("nested_den", 20, 14, 30, 18, "2"),
        _sample("outer_den", 20, 30, 30, 40, "3"),
    ]
    nested_formulae, nested_audit = finalize_formula_outputs(
        nested_fraction,
        {str(row["record_id"]): str(row["final_topk"][0]) for row in nested_fraction},
    )
    assert nested_formulae[0]["latex"] == r"\frac{\frac{1}{2}}{3}"
    assert nested_formulae[0]["latex_status"] == "shadow_ambiguous_multi_parent"
    assert nested_audit["multi_parent_nodes"] == 2


if __name__ == "__main__":
    _self_test()
    print(json.dumps({"self_test": "pass"}))
