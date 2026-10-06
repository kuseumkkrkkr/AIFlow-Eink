#!/usr/bin/env python3
"""Render remaining project-owned formula failures with raw stroke ownership."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
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
    scale = float(max(np.max(high - low), 1e-9))
    return [(stroke[:, :2] - low) / scale for stroke in strokes]


def _context_index(row: dict) -> int:
    if "context_index" in row:
        return int(row["context_index"])
    return int(row.get("context", {}).get("index", -1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--legacy-dataset-root", type=Path,
        default=Path(__file__).resolve().parents[1] / "hf-dataset",
    )
    parser.add_argument("--top20-candidates", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidate_root = _d_path(args.dataset_root, "expanded dataset")
    legacy_root = _d_path(args.legacy_dataset_root, "legacy dataset")
    candidate_path = _d_path(args.top20_candidates, "Top-20 candidates")
    runtime_path = _d_path(args.runtime, "formula runtime")
    output_path = _d_path(args.output, "remaining-failure visualization")
    manifest_path = output_path.with_suffix(".json")
    if not candidate_path.is_file() or not runtime_path.is_file():
        parser.error("remaining-failure visualization inputs are missing")
    if output_path.exists() or manifest_path.exists():
        parser.error("refusing to overwrite remaining-failure visualization")
    candidates = {
        str(row["record_id"]): row for row in _json_lines(candidate_path)
    }
    runtime = {str(row["record_id"]): row for row in _json_lines(runtime_path)}
    if not candidates or set(candidates) != set(runtime):
        raise ValueError("remaining-failure candidate/runtime coverage mismatch")
    writer_map, legacy_summary = _legacy_writer_map(candidate_root, legacy_root)
    items, raw_strokes, _ = _items(
        candidate_root, writer_map, int(legacy_summary["legacy_formulas"])
    )
    if not set(candidates) <= set(raw_strokes):
        raise ValueError("remaining-failure raw-stroke coverage mismatch")
    by_formula: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        if str(item["record_id"]) not in candidates:
            continue
        by_formula[str(item["formula_id"])].append(item)
    failures = []
    for formula_id, formula_items in sorted(by_formula.items()):
        ordered = sorted(
            formula_items,
            key=lambda row: _context_index(candidates[str(row["record_id"])]),
        )
        wrong = [
            row for row in ordered
            if str(runtime[str(row["record_id"])]["finalized_top1"])
            != str(candidates[str(row["record_id"])]["label"])
        ]
        if wrong:
            failures.append((formula_id, ordered, wrong))
    if not failures:
        raise ValueError("no remaining formula failures to visualize")
    figure = plt.figure(figsize=(18, 3.1 * len(failures)), constrained_layout=True)
    grid = figure.add_gridspec(len(failures), 3, width_ratios=(2.2, 1.5, 1.4))
    palette = plt.get_cmap("tab20")
    manifest_rows = []
    for row_index, (formula_id, ordered, wrong) in enumerate(failures):
        formula_axis = figure.add_subplot(grid[row_index, 0])
        glyph_axis = figure.add_subplot(grid[row_index, 1])
        text_axis = figure.add_subplot(grid[row_index, 2])
        truth_tokens = []
        predicted_tokens = []
        for glyph_index, item in enumerate(ordered):
            record_id = str(item["record_id"])
            truth = str(candidates[record_id]["label"])
            prediction = str(runtime[record_id]["finalized_top1"])
            truth_tokens.append(truth)
            predicted_tokens.append(prediction)
            color = palette(glyph_index % 20)
            points = np.concatenate([stroke[:, :2] for stroke in raw_strokes[record_id]])
            for stroke in raw_strokes[record_id]:
                formula_axis.plot(stroke[:, 0], stroke[:, 1], color=color, linewidth=1.8)
                formula_axis.scatter(stroke[0, 0], stroke[0, 1], color=color, s=12)
            formula_axis.text(
                float(points[:, 0].mean()), float(points[:, 1].min()), truth,
                color=color, fontsize=8, ha="center", va="bottom",
            )
        formula_axis.set_title(
            f"{formula_id}\ntruth: {' '.join(truth_tokens)}\npred:  {' '.join(predicted_tokens)}",
            fontsize=9,
        )
        formula_axis.set_aspect("equal", adjustable="datalim")
        formula_axis.invert_yaxis()
        formula_axis.grid(True, linewidth=0.35, alpha=0.25)
        formula_axis.set_xticks([])
        formula_axis.set_yticks([])
        text_lines = []
        wrong_manifest = []
        for wrong_index, item in enumerate(wrong):
            record_id = str(item["record_id"])
            candidate = candidates[record_id]
            truth = str(candidate["label"])
            prediction = str(runtime[record_id]["finalized_top1"])
            tokens = [str(value) for value in candidate["final_topk"]]
            truth_rank = tokens.index(truth) + 1 if truth in tokens else None
            normalized = _normalized(raw_strokes[record_id])
            offset = wrong_index * 1.35
            for stroke_index, stroke in enumerate(normalized):
                color = palette((wrong_index * 4 + stroke_index) % 20)
                glyph_axis.plot(
                    stroke[:, 0] + offset, stroke[:, 1], color=color, linewidth=2.2,
                )
                glyph_axis.scatter(
                    stroke[0, 0] + offset, stroke[0, 1], color=color, s=18,
                )
                glyph_axis.scatter(
                    stroke[-1, 0] + offset, stroke[-1, 1], color=color, s=20,
                    marker="x",
                )
            glyph_axis.text(
                offset + 0.5, -0.08,
                f"{truth} <- {prediction}\nrank={truth_rank}",
                fontsize=8, ha="center", va="bottom",
            )
            top5 = "  ".join(tokens[:5])
            text_lines.append(
                f"i{_context_index(candidate)}  {prediction} -> {truth}\n"
                f"rank={truth_rank}  top5={top5}"
            )
            wrong_manifest.append({
                "record_id": record_id,
                "context_index": _context_index(candidate),
                "truth": truth,
                "prediction": prediction,
                "truth_rank": truth_rank,
                "top5": tokens[:5],
                "stroke_count": len(raw_strokes[record_id]),
                "point_count": sum(len(stroke) for stroke in raw_strokes[record_id]),
            })
        glyph_axis.set_xlim(-0.15, max(1.1, len(wrong) * 1.35 - 0.2))
        glyph_axis.set_ylim(1.1, -0.18)
        glyph_axis.set_aspect("equal", adjustable="box")
        glyph_axis.grid(True, linewidth=0.35, alpha=0.25)
        glyph_axis.set_xticks([])
        glyph_axis.set_yticks([])
        text_axis.axis("off")
        text_axis.text(0.0, 0.5, "\n\n".join(text_lines), fontsize=8, va="center")
        manifest_rows.append({
            "formula_id": formula_id,
            "truth": truth_tokens,
            "prediction": predicted_tokens,
            "wrong": wrong_manifest,
        })
    figure.suptitle(
        "Remaining project-owned formula failures — raw online strokes\n"
        "whole formula ownership (left), normalized wrong glyphs (center), Top-20 evidence (right)",
        fontsize=14,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=170)
    plt.close(figure)
    manifest_path.write_text(
        json.dumps({
            "schema": "aiflow-remaining-formula-failure-visualization/v1",
            "image": str(output_path),
            "formulas": len(manifest_rows),
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
        "formulas": len(manifest_rows),
        "wrong_glyphs": sum(len(row["wrong"]) for row in manifest_rows),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
