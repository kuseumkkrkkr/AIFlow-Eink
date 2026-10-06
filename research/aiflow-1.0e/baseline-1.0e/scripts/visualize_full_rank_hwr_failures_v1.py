#!/usr/bin/env python3
"""Render raw online-ink glyphs whose truth rank is outside Top-20."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from build_expanded_owned_context_candidates_v1 import _items, _legacy_writer_map
from character_tensor_v1 import _json_lines
from evaluate_homograph_context_reranker_v1 import _d_path


def _normalized(strokes: list[np.ndarray]) -> list[np.ndarray]:
    points = np.concatenate([stroke[:, :2] for stroke in strokes], axis=0)
    low = points.min(axis=0)
    high = points.max(axis=0)
    span = np.maximum(high - low, 1e-9)
    scale = float(max(span))
    return [(stroke[:, :2] - low) / scale for stroke in strokes]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--legacy-dataset-root", type=Path,
        default=Path(__file__).resolve().parents[1] / "hf-dataset",
    )
    parser.add_argument("--rank-audit", type=Path, required=True)
    parser.add_argument("--top20-candidates", type=Path, required=True)
    parser.add_argument("--reference-weight", type=float, default=0.6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not math.isfinite(args.reference_weight):
        parser.error("reference weight must be finite")
    candidate_root = _d_path(args.dataset_root, "expanded dataset")
    legacy_root = _d_path(args.legacy_dataset_root, "legacy dataset")
    audit_path = _d_path(args.rank_audit, "full-rank audit")
    candidate_path = _d_path(args.top20_candidates, "Top-20 candidates")
    output_path = _d_path(args.output, "failure visualization")
    manifest_path = output_path.with_suffix(".json")
    if not audit_path.is_file() or not candidate_path.is_file():
        parser.error("failure visualization inputs are missing")
    if output_path.exists() or manifest_path.exists():
        parser.error("refusing to overwrite failure visualization")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    weight_key = str(float(args.reference_weight))
    target_rows = [
        row for row in audit["target_rows"]
        if int(row["truth_rank_by_weight"][weight_key]) > 20
    ]
    target_ids = {str(row["record_id"]) for row in target_rows}
    candidates = {
        str(row["record_id"]): row for row in _json_lines(candidate_path)
    }
    writer_map, legacy_summary = _legacy_writer_map(candidate_root, legacy_root)
    items, raw_strokes, _ = _items(
        candidate_root, writer_map, int(legacy_summary["legacy_formulas"])
    )
    item_map = {str(row["record_id"]): row for row in items}
    if not target_ids or not target_ids <= set(candidates) or not target_ids <= set(raw_strokes):
        raise ValueError("failure visualization target coverage mismatch")
    columns = 4
    rows_count = math.ceil(len(target_rows) / columns)
    figure, axes = plt.subplots(
        rows_count, columns, figsize=(4.2 * columns, 4.0 * rows_count),
        constrained_layout=True, squeeze=False,
    )
    palette = plt.get_cmap("tab10")
    manifest_rows = []
    for axis, target in zip(axes.flat, target_rows, strict=False):
        record_id = str(target["record_id"])
        strokes = _normalized(raw_strokes[record_id])
        for stroke_index, stroke in enumerate(strokes):
            color = palette(stroke_index % 10)
            axis.plot(stroke[:, 0], stroke[:, 1], color=color, linewidth=2.0)
            axis.scatter(stroke[0, 0], stroke[0, 1], color=color, s=22, marker="o")
            axis.scatter(stroke[-1, 0], stroke[-1, 1], color=color, s=26, marker="x")
        candidate = candidates[record_id]
        rank = int(target["truth_rank_by_weight"][weight_key])
        axis.set_title(
            f"{target['formula_id']}\ntruth={target['truth']}  "
            f"top1={candidate['final_topk'][0]}  rank={rank}\n"
            f"strokes={len(strokes)}  points={sum(len(value) for value in strokes)}",
            fontsize=9,
        )
        axis.set_aspect("equal", adjustable="box")
        axis.invert_yaxis()
        axis.grid(True, linewidth=0.4, alpha=0.3)
        axis.set_xticks([])
        axis.set_yticks([])
        manifest_rows.append({
            "record_id": record_id,
            "formula_id": str(target["formula_id"]),
            "truth": str(target["truth"]),
            "top1": str(candidate["final_topk"][0]),
            "truth_rank": rank,
            "truth_probability": float(
                target["truth_probability_by_weight"][weight_key]
            ),
            "writer_group": str(item_map[record_id]["writer_group"]),
            "stroke_count": len(strokes),
            "point_count": sum(len(value) for value in strokes),
        })
    for axis in axes.flat[len(target_rows):]:
        axis.axis("off")
    figure.suptitle(
        "Project-owned HWR truth outside Top-20 — raw ordered strokes\n"
        "circle=start, x=end; diagnostic only",
        fontsize=14,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    manifest_path.write_text(
        json.dumps({
            "schema": "aiflow-full-rank-hwr-failure-visualization/v1",
            "reference_weight": float(args.reference_weight),
            "image": str(output_path),
            "rows": manifest_rows,
            "limits": [
                "project-owned diagnostic visualization",
                "not approved for public dataset release",
            ],
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    print(json.dumps({
        "output": str(output_path), "manifest": str(manifest_path),
        "glyphs": len(manifest_rows),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
