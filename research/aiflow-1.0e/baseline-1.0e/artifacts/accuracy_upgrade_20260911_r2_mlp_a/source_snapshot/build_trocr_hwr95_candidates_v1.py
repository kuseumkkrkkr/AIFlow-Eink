#!/usr/bin/env python3
"""Build a formula-linked candidate cache for the 1.0e TrOCR shadow loop."""

from __future__ import annotations

import argparse
import gzip
import io
import json
import math
from collections import defaultdict
from pathlib import Path

from accuracy_upgrade_contract_v1 import (
    formula_bounds,
    formula_position,
    relative_context,
    source_bbox,
)
from character_tensor_v1 import iter_direct_ownership_examples


def _rows(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _geometry(row: dict) -> dict[str, float]:
    points = [point for stroke in row["strokes"] for point in stroke["points"]]
    xy = [(float(point[0]), float(point[1])) for point in points]
    xs = [point[0] for point in xy]
    ys = [point[1] for point in xy]
    left, right = min(xs), max(xs)
    top, bottom = min(ys), max(ys)
    width = max(right - left, 1e-6)
    height = max(bottom - top, 1e-6)
    path = 0.0
    dx = dy = 0.0
    for stroke in row["strokes"]:
        for first, second in zip(stroke["points"], stroke["points"][1:]):
            step_x = float(second[0]) - float(first[0])
            step_y = float(second[1]) - float(first[1])
            path += math.hypot(step_x, step_y)
            dx += step_x
            dy += step_y
    diagonal = math.hypot(width, height)
    raw = source_bbox(row)
    return {
        "left": left,
        "top": top,
        "right": right,
        "bottom": bottom,
        "width": width,
        "height": height,
        "cx": (left + right) / 2.0,
        "cy": (top + bottom) / 2.0,
        "width_rel": width,
        "height_rel": height,
        "aspect_log": math.log(width / height),
        "path_over_diag": path / diagonal,
        "direction_x": dx / max(path, 1e-6),
        "direction_y": dy / max(path, 1e-6),
        "stroke_count": float(len(row["strokes"])),
        "point_count_log": math.log(max(1, len(points))),
        "center_x": (left + right) / 2.0,
        "center_y": (top + bottom) / 2.0,
        "raw_left": raw["left"],
        "raw_top": raw["top"],
        "raw_right": raw["right"],
        "raw_bottom": raw["bottom"],
        "raw_width": raw["width"],
        "raw_height": raw["height"],
        "raw_cx": raw["cx"],
        "raw_cy": raw["cy"],
    }


def _formula_map(formulas: Path, ownership: Path, raw_ids: set[str]) -> dict[str, str]:
    generated = list(iter_direct_ownership_examples(formulas, ownership))
    accepted = [row for row in _jsonl(ownership) if row.get("accepted")]
    expected = sum(len(row.get("groups", [])) for row in accepted)
    if expected != len(generated):
        raise ValueError(f"ownership expansion mismatch: {expected} != {len(generated)}")
    output: dict[str, str] = {}
    cursor = 0
    for annotation in accepted:
        sample_id = str(annotation["sample_id"])
        for _ in annotation.get("groups", []):
            record_id = str(generated[cursor]["record_id"])
            if record_id in raw_ids:
                output[record_id] = sample_id
            cursor += 1
    if set(output) != raw_ids:
        raise ValueError(f"formula mapping coverage mismatch: {len(output)} != {len(raw_ids)}")
    return output


def build(raw_path: Path, prediction_path: Path, formulas: Path, ownership: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    raw = _rows(raw_path)
    predictions = _rows(prediction_path)
    raw_by_id = {str(row["record_id"]): row for row in raw}
    prediction_by_id = {str(row["record_id"]): row for row in predictions}
    if len(raw_by_id) != len(raw) or len(prediction_by_id) != len(predictions):
        raise ValueError("record IDs must be unique")
    if set(raw_by_id) != set(prediction_by_id):
        raise ValueError("raw and prediction coverage differs")
    formula_by_record = _formula_map(formulas, ownership, set(raw_by_id))
    grouped: defaultdict[tuple[str, str], list[dict]] = defaultdict(list)
    for record_id in raw_by_id:
        prediction = prediction_by_id[record_id]
        grouped[(str(prediction["writer"]), formula_by_record[record_id])].append(prediction)
    contexts: dict[str, dict[str, float]] = {}
    geometry_by_id: dict[str, dict[str, float]] = {}
    for group in grouped.values():
        centers = [_geometry(raw_by_id[str(row["record_id"])]) for row in group]
        bounds = formula_bounds(
            [{"left": item["raw_left"], "top": item["raw_top"], "right": item["raw_right"], "bottom": item["raw_bottom"]} for item in centers]
        )
        for row, item in zip(group, centers, strict=True):
            geometry_by_id[str(row["record_id"])] = {
                **item,
                **formula_position(
                    {"left": item["raw_left"], "top": item["raw_top"], "right": item["raw_right"], "bottom": item["raw_bottom"], "width": item["raw_width"], "height": item["raw_height"], "cx": item["raw_cx"], "cy": item["raw_cy"]},
                    bounds,
                ),
            }
        raw_centers = [
            {"cx": item["raw_cx"], "cy": item["raw_cy"]}
            for item in centers
        ]
        for index, prediction in enumerate(group):
            current = raw_centers[index]
            previous = raw_centers[index - 1] if index else None
            following = raw_centers[index + 1] if index + 1 < len(group) else None
            context = relative_context(previous, current, following, bounds)
            contexts[str(prediction["record_id"])] = {
                "index": float(index),
                "length": float(len(group)),
                "previous_top1": "<S>" if previous is None else str(group[index - 1]["top_labels"][0]),
                "next_top1": "</S>" if following is None else str(group[index + 1]["top_labels"][0]),
                **context,
            }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as raw_stream, gzip.GzipFile(
        filename="", fileobj=raw_stream, mode="wb", mtime=0,
    ) as zipped, io.TextIOWrapper(zipped, encoding="utf-8", newline="\n") as stream:
        for record_id in raw_by_id:
            source = raw_by_id[record_id]
            prediction = prediction_by_id[record_id]
            stream.write(json.dumps({
                "record_id": record_id,
                "label": str(source["label"]),
                "source": "project_owned",
                "formula_id": formula_by_record[record_id],
                "writer_group": str(prediction["writer"]),
                "raw_writer_group": str(source.get("writer_group", "")),
                "final_topk": list(prediction["top_labels"]),
                "final_topk_probabilities": list(prediction["top_scores"]),
                "geometry": geometry_by_id[record_id],
                "context": contexts[record_id],
            }, ensure_ascii=False, separators=(",", ":")) + "\n")
    return {
        "schema": "aiflow-1.0e-trocr-hwr95-candidates/v2",
        "records": len(raw_by_id),
        "writers": len({str(row["writer"]) for row in predictions}),
        "formulas": len(set(formula_by_record.values())),
        "candidate_contract": "existing HWR Top-k only",
        "geometry_contract": "local_shape_plus_formula_coordinate_context/v1",
        "training_performed": False,
        "output": str(output),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--formulas", type=Path, required=True)
    parser.add_argument("--ownership", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.raw, args.predictions, args.formulas, args.ownership, args.output), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
