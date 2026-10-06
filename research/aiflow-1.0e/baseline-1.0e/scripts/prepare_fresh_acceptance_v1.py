#!/usr/bin/env python3
"""Prepare a leakage-screened, visually reviewable fresh acceptance set.

This script never trains or evaluates a model.  It identifies formulae from
writers absent from the frozen direct candidate cache, removes exact token
sequences seen by context training, proposes contiguous stroke ownership, and
renders every proposal for independent visual review.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
from typing import Iterator

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = Path(
    r"D:\AIFlow-Workspace\PrivateData\math-ink-data-collector\derived"
    r"\public-candidate-20260820-r4-replay-restored"
)
DEFAULT_DIRECT = Path(
    r"D:\AIFlow-Workspace\PrivateData\math-ink-data-collector\derived"
    r"\hwr-head-calibration-20260819-r1\expanded_writer_loo_candidates_r1.jsonl.gz"
)
DEFAULT_PROMPT = (
    ROOT / "artifacts" / "prompt_context_corpus_20260820_r2"
    / "prompt_context_corpus.jsonl"
)
DEFAULT_BROAD = (
    ROOT / "datasets" / "10_approved_external" / "deepmind_mathematics_dataset"
    / "derived" / "formula_context_v1.jsonl.gz"
)
DEFAULT_OUTPUT = Path(
    r"D:\AIFlow-Workspace\PrivateData\math-ink-data-collector\derived"
    r"\fresh-context-acceptance-20260820-r1"
)
COLORS = (
    "#d32f2f", "#1976d2", "#388e3c", "#7b1fa2",
    "#f57c00", "#00897b", "#5d4037", "#303f9f",
)
STROKE_PRIOR = {
    "+": 2, "=": 2, r"\times": 2, r"\div": 3,
    "/": 1, "-": 1, "(": 1, ")": 1,
    "m": 1, "n": 1, "a": 1, "b": 1,
    "f": 2, "x": 2, "y": 2,
}
KNOWN_SPARSE = ("|", "O", "o")


def _lines(path: Path) -> Iterator[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def _sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _d_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.drive.upper() != "D:" or not resolved.is_file():
        raise ValueError(f"{label} must be an existing D: file: {resolved}")
    return resolved


def _sequence(row: dict) -> tuple[str, ...]:
    return tuple(str(cell["token"]) for cell in row.get("target_cells", []))


def _box(strokes: list[dict], indices: list[int]) -> tuple[float, float, float, float]:
    points = [point for index in indices for point in strokes[index]["points"]]
    return (
        min(float(point["x"]) for point in points),
        min(float(point["y"]) for point in points),
        max(float(point["x"]) for point in points),
        max(float(point["y"]) for point in points),
    )


def _score(strokes: list[dict], labels: list[str], sizes: tuple[int, ...]) -> float:
    groups, start = [], 0
    for size in sizes:
        groups.append(list(range(start, start + size)))
        start += size
    boxes = [_box(strokes, group) for group in groups]
    heights = sorted(max(1.0, box[3] - box[1]) for box in boxes)
    scale = heights[len(heights) // 2]
    score = 0.0
    for label, size, box in zip(labels, sizes, boxes, strict=True):
        score += 2.5 * abs(size - STROKE_PRIOR.get(label, 1))
        score += 0.15 * max(0.0, (box[2] - box[0]) / scale - 1.2)
    for first, second in zip(boxes, boxes[1:]):
        gap = second[0] - first[2]
        score += 5.0 * max(0.0, -gap) / scale
        score -= min(2.0, max(0.0, gap) / scale)
        if second[0] + second[2] < first[0] + first[2]:
            score += 5.0
    return score


def _partition(strokes: list[dict], labels: list[str]) -> list[list[int]]:
    if len(labels) == 1:
        return [list(range(len(strokes)))]
    candidates = (
        sizes for sizes in product(range(1, 5), repeat=len(labels))
        if sum(sizes) == len(strokes)
    )
    try:
        selected = min(candidates, key=lambda sizes: _score(strokes, labels, sizes))
    except ValueError as error:
        raise ValueError(
            f"no contiguous ownership partition for {len(strokes)} strokes and "
            f"{len(labels)} labels"
        ) from error
    groups, start = [], 0
    for size in selected:
        groups.append(list(range(start, start + size)))
        start += size
    return groups


def _render(rows: list[dict], output: Path) -> list[dict]:
    output.mkdir(parents=True, exist_ok=False)
    font = ImageFont.load_default()
    pages = []
    for page_start in range(0, len(rows), 8):
        page_rows = rows[page_start:page_start + 8]
        image = Image.new("RGB", (1400, 900), "white")
        draw = ImageDraw.Draw(image)
        for slot, row in enumerate(page_rows):
            origin_x, origin_y = (slot % 2) * 700, (slot // 2) * 225
            labels = row["labels"]
            draw.text(
                (origin_x + 8, origin_y + 5),
                f"{row['sample_id']}  {' '.join(labels)}",
                fill="black", font=font,
            )
            formula_box = _box(row["strokes"], list(range(len(row["strokes"]))))
            formula_width = max(1.0, formula_box[2] - formula_box[0])
            formula_height = max(1.0, formula_box[3] - formula_box[1])
            scale = min(600 / formula_width, 155 / formula_height)
            offset_x = origin_x + 45 + (600 - formula_width * scale) / 2
            offset_y = origin_y + 38 + (155 - formula_height * scale) / 2

            def location(x: float, y: float) -> tuple[float, float]:
                return (
                    offset_x + (x - formula_box[0]) * scale,
                    offset_y + (y - formula_box[1]) * scale,
                )

            for group_index, (group, label) in enumerate(
                zip(row["groups"], labels, strict=True)
            ):
                color = COLORS[group_index % len(COLORS)]
                for stroke_index in group:
                    points = row["strokes"][stroke_index]["points"]
                    coordinates = [
                        location(float(point["x"]), float(point["y"]))
                        for point in points
                    ]
                    if len(coordinates) > 1:
                        draw.line(coordinates, fill=color, width=3)
                    elif coordinates:
                        x, y = coordinates[0]
                        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
                    if coordinates:
                        draw.text(coordinates[0], str(stroke_index), fill=color, font=font)
                left, top, right, bottom = _box(row["strokes"], group)
                box_left, box_top = location(left, top)
                box_right, box_bottom = location(right, bottom)
                draw.rectangle(
                    (box_left - 3, box_top - 3, box_right + 3, box_bottom + 3),
                    outline=color, width=2,
                )
                draw.text(
                    (box_left + 3, box_bottom + 2), label, fill=color, font=font
                )
        path = output / f"ownership-{page_start // 8 + 1:02d}.png"
        image.save(path)
        pages.append({
            "file": path.name,
            "sha256": _sha(path),
            "sample_ids": [row["sample_id"] for row in page_rows],
        })
    return pages


def prepare(args: argparse.Namespace) -> dict:
    dataset = args.dataset.expanduser().resolve()
    if dataset.drive.upper() != "D:" or not dataset.is_dir():
        raise ValueError(f"dataset must be an existing D: directory: {dataset}")
    formulas_path = _d_file(dataset / "data" / "formulas_valid.jsonl", "valid formulae")
    ownership_path = _d_file(dataset / "data" / "ownership_train.jsonl", "ownership")
    direct_path = _d_file(args.direct, "frozen direct candidates")
    prompt_path = _d_file(args.prompt, "prompt context corpus")
    broad_path = _d_file(args.broad, "broad context corpus")
    output = args.output.expanduser().resolve()
    if output.drive.upper() != "D:":
        raise ValueError(f"output must remain on D:: {output}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite acceptance preparation: {output}")

    formulas = list(_lines(formulas_path))
    formula_by_id = {str(row["sample_id"]): row for row in formulas}
    direct = list(_lines(direct_path))
    used_formulae = {str(row["formula_id"]) for row in direct}
    used_writers = {
        str(formula_by_id[formula_id]["writer_id"])
        for formula_id in used_formulae if formula_id in formula_by_id
    }
    accepted_ids = {
        str(row["sample_id"]) for row in _lines(ownership_path)
        if row.get("accepted")
    }
    prompt_sequences = {
        tuple(str(token) for token in row["labels"])
        for row in _lines(prompt_path)
    }
    broad_sequences = {
        tuple(str(token) for token in row["tokens"])
        for row in _lines(broad_path)
    }

    unused_writer_pool = [
        row for row in formulas
        if str(row["sample_id"]) not in used_formulae
        and str(row["writer_id"]) not in used_writers
    ]
    overlap = [
        row for row in unused_writer_pool
        if _sequence(row) in prompt_sequences or _sequence(row) in broad_sequences
    ]
    selected = [
        row for row in unused_writer_pool
        if _sequence(row)
        and _sequence(row) not in prompt_sequences
        and _sequence(row) not in broad_sequences
    ]
    selected.sort(key=lambda row: (str(row["writer_id"]), str(row["sample_id"])))
    proposals = []
    for row in selected:
        labels = list(_sequence(row))
        strokes = sorted(row["strokes"], key=lambda stroke: int(stroke["order"]))
        groups = _partition(strokes, labels)
        owned = [index for group in groups for index in group]
        if sorted(owned) != list(range(len(strokes))) or len(owned) != len(set(owned)):
            raise AssertionError("proposed ownership is not exhaustive and unique")
        proposals.append({
            "sample_id": str(row["sample_id"]),
            "writer_id": str(row["writer_id"]),
            "canvas": row["canvas"],
            "labels": labels,
            "groups": groups,
            "strokes": strokes,
            "accepted": False,
        })

    output.mkdir(parents=True)
    pages = _render(proposals, output / "inspection")
    compact = [
        {key: value for key, value in row.items() if key != "strokes"}
        for row in proposals
    ]
    proposal_path = output / "ownership_proposals.json"
    proposal_path.write_text(json.dumps({
        "schema": "aiflow-fresh-acceptance-proposals/v1",
        "rows": compact,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    inspection_path = output / "inspection_manifest.json"
    inspection_path.write_text(json.dumps({
        "schema": "aiflow-fresh-acceptance-inspection/v1",
        "pages": pages,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    token_counts = Counter(token for row in selected for token in _sequence(row))
    writer_counts = Counter(str(row["writer_id"]) for row in selected)
    report = {
        "schema": "aiflow-fresh-acceptance-readiness/v1",
        "model_predictions_opened": False,
        "training_performed": False,
        "selection": {
            "frozen_direct_formulae": len(used_formulae),
            "frozen_direct_writers": len(used_writers),
            "unused_writer_pool_formulae": len(unused_writer_pool),
            "exact_training_sequence_overlap_excluded": len(overlap),
            "selected_formulae": len(selected),
            "selected_writers": len(writer_counts),
            "context_formulae": sum(len(_sequence(row)) >= 2 for row in selected),
            "single_symbol_formulae": sum(len(_sequence(row)) == 1 for row in selected),
            "previously_accepted_ownership": sum(
                str(row["sample_id"]) in accepted_ids for row in selected
            ),
            "writer_formula_counts": dict(sorted(writer_counts.items())),
            "known_sparse_truth_counts": {
                token: token_counts[token] for token in KNOWN_SPARSE
            },
        },
        "gates_before_visual_review": {
            "at_least_two_unused_writers": len(writer_counts) >= 2,
            "at_least_fifty_formulae": len(selected) >= 50,
            "all_ownership_independently_reviewed": False,
            "known_sparse_truth_present": all(token_counts[token] > 0 for token in KNOWN_SPARSE),
        },
        "artifacts": {
            "formulae": {"path": str(formulas_path), "sha256": _sha(formulas_path)},
            "ownership": {"path": str(ownership_path), "sha256": _sha(ownership_path)},
            "direct_candidates": {"path": str(direct_path), "sha256": _sha(direct_path)},
            "prompt_context": {"path": str(prompt_path), "sha256": _sha(prompt_path)},
            "broad_context": {"path": str(broad_path), "sha256": _sha(broad_path)},
            "proposals": {"path": str(proposal_path), "sha256": _sha(proposal_path)},
            "inspection_manifest": {"path": str(inspection_path), "sha256": _sha(inspection_path)},
        },
    }
    readiness_path = output / "readiness.json"
    readiness_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--direct", type=Path, default=DEFAULT_DIRECT)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--broad", type=Path, default=DEFAULT_BROAD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(prepare(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
