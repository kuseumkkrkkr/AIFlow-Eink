#!/usr/bin/env python3
"""Build a private relation-only acceptance set from reviewed owned ink."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


SCHEMA = "aiflow-owned-formula-relation-acceptance/v1"
RELATIONS = frozenset({"above", "below", "contains", "subscript", "superscript"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _geometry(strokes: dict[int, dict], group: list[int]) -> dict[str, float]:
    points = [point for index in group for point in strokes[index]["points"]]
    coordinates = [(float(point["x"]), float(point["y"])) for point in points]
    if not coordinates or not all(math.isfinite(value) for pair in coordinates for value in pair):
        raise ValueError("owned relation group requires finite points")
    xs, ys = zip(*coordinates, strict=True)
    left, top, right, bottom = min(xs), min(ys), max(xs), max(ys)
    return {
        "left": left, "top": top, "right": right, "bottom": bottom,
        "center_x": (left + right) / 2.0, "center_y": (top + bottom) / 2.0,
    }


def _formula(annotation: dict) -> dict:
    source_path = Path(str(annotation["source_path"])).resolve()
    if not source_path.is_file() or _sha256(source_path) != str(annotation["source_sha256"]):
        raise ValueError(f"owned relation source hash mismatch: {source_path}")
    source = json.loads(source_path.read_text(encoding="utf-8-sig"))
    if source.get("consent_scope") != "commercial_model_training:aiflow_math_ink":
        raise ValueError(f"owned relation consent mismatch: {source_path}")
    labels = [str(value) for value in annotation["labels"]]
    target = [str(cell["token"]) for cell in source.get("target_cells") or []]
    groups = [[int(value) for value in group] for group in annotation["groups"]]
    if labels != target or len(groups) != len(labels):
        raise ValueError(f"owned relation label/group mismatch: {source_path}")
    strokes = {int(row["order"]): row for row in source.get("strokes") or []}
    assigned = [value for group in groups for value in group]
    if sorted(assigned) != sorted(strokes) or len(assigned) != len(set(assigned)):
        raise ValueError(f"owned relation stroke ownership mismatch: {source_path}")
    formula_id = str(annotation["public_formula_id"])
    rows = []
    for index, (label, group) in enumerate(zip(labels, groups, strict=True)):
        rows.append({
            "record_id": f"{formula_id}:{index}", "formula_id": formula_id,
            "label": label, "final_topk": [label], "final_topk_probabilities": [1.0],
            "geometry": _geometry(strokes, group),
            "context": {"index": index, "length": len(labels)},
            "evaluation_partition": "owned_relation_character_oracle",
        })
    truth = []
    for relation in annotation["relations"]:
        parent, child = int(relation["parent_index"]), int(relation["child_index"])
        kind = str(relation["type"]).lower()
        if kind not in RELATIONS or parent == child or not (0 <= parent < len(rows) and 0 <= child < len(rows)):
            raise ValueError(f"invalid owned relation edge: {formula_id}")
        truth.append({"type": kind, "parent_index": parent, "child_index": child})
    if len({(row["type"], row["parent_index"], row["child_index"]) for row in truth}) != len(truth):
        raise ValueError(f"duplicate owned relation edge: {formula_id}")
    stored = {
        (str(row["type"]).lower(), int(row["from"]), int(row["to"]))
        for row in source.get("target_relations") or []
    }
    expected = {(row["type"], row["parent_index"], row["child_index"]) for row in truth}
    relation_source = str(annotation["relation_source"])
    if relation_source.startswith("prompt_stored") and stored != expected:
        raise ValueError(f"stored owned relation mismatch: {formula_id}")
    return {
        "formula_id": formula_id, "rows": rows, "truth_relations": truth,
        "relation_source": relation_source, "source_sha256": str(annotation["source_sha256"]),
    }


def _self_test() -> None:
    strokes = {
        0: {"points": [{"x": 1, "y": 2}, {"x": 3, "y": 4}]},
        1: {"points": [{"x": 5, "y": 1}]},
    }
    assert _geometry(strokes, [0, 1]) == {
        "left": 1.0, "top": 1.0, "right": 5.0, "bottom": 4.0,
        "center_x": 3.0, "center_y": 2.5,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        print(json.dumps({"self_test": "pass"}))
        return 0
    if args.annotations is None or args.output is None:
        parser.error("--annotations and --output are required")
    annotations = args.annotations.resolve()
    output = args.output.resolve()
    if not annotations.is_file() or output.exists():
        parser.error("annotations must exist and output must not exist")
    payload = json.loads(annotations.read_text(encoding="utf-8"))
    if payload.get("schema") != "aiflow-owned-formula-relation-annotations/v1":
        raise ValueError("owned relation annotation schema mismatch")
    formulas = [_formula(row) for row in payload.get("rows") or [] if row.get("accepted")]
    if not formulas or len({row["formula_id"] for row in formulas}) != len(formulas):
        raise ValueError("owned relation formulas must be non-empty and unique")
    relation_counts = Counter(
        edge["type"] for formula in formulas for edge in formula["truth_relations"]
    )
    result = {
        "schema": SCHEMA, "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "relation_only_character_oracle",
        "commercial_training_rights": "project_owned_consent_verified",
        "character_model_evaluated": False,
        "formulas": formulas,
        "audit": {
            "formula_count": len(formulas),
            "glyph_count": sum(len(row["rows"]) for row in formulas),
            "relation_count": sum(relation_counts.values()),
            "relation_counts": dict(sorted(relation_counts.items())),
            "annotation_sha256": _sha256(annotations),
            "source_hashes_verified": len(formulas),
            "ownership_coverage": 1.0,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({
        "output": str(output), "sha256": _sha256(output), "audit": result["audit"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
