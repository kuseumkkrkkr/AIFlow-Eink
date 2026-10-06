#!/usr/bin/env python3
"""Audit whether transcript-only formulas admit exact covers of the target size.

This is a geometry-only upper-bound diagnostic. It does not run HWR, train a
model, infer token identity/order, or use CROHME data.
"""

from __future__ import annotations

import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import statistics
from typing import Any

from stroke_grouping_v1 import DEFAULT_LATTICE_CONFIG, build_lattice


SCHEMA = "aiflow-transcript-only-partition-cardinality/v1"
MAX_AUDIT_STROKES = 18


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cover_counts_by_group_count(
    candidates: list[dict[str, Any]], stroke_count: int,
) -> tuple[int, ...]:
    """Count unordered candidate exact covers, indexed by number of groups."""
    if not 1 <= stroke_count <= MAX_AUDIT_STROKES:
        raise ValueError(f"stroke count must be within 1..{MAX_AUDIT_STROKES}")

    by_stroke: list[list[int]] = [[] for _ in range(stroke_count)]
    seen: set[int] = set()
    for candidate in candidates:
        indices = tuple(sorted({int(value) for value in candidate["source_indices"]}))
        if not indices or indices[0] < 0 or indices[-1] >= stroke_count:
            raise ValueError("candidate contains an invalid stroke index")
        mask = sum(1 << value for value in indices)
        if mask in seen:
            continue
        seen.add(mask)
        for value in indices:
            by_stroke[value].append(mask)

    @lru_cache(maxsize=None)
    def count(remaining: int) -> tuple[int, ...]:
        if remaining == 0:
            return (1,) + (0,) * stroke_count
        pivot_bit = remaining & -remaining
        pivot = pivot_bit.bit_length() - 1
        result = [0] * (stroke_count + 1)
        for mask in by_stroke[pivot]:
            if mask & remaining != mask:
                continue
            for suffix_count, ways in enumerate(count(remaining ^ mask)):
                if ways:
                    result[suffix_count + 1] += ways
        return tuple(result)

    return count((1 << stroke_count) - 1)


def _validate_inputs(
    formulas: list[dict[str, Any]], ownership: list[dict[str, Any]], manifest: dict[str, Any],
    formulas_path: Path, ownership_path: Path,
) -> set[str]:
    formula_by_id = {str(row["sample_id"]): row for row in formulas}
    if len(formula_by_id) != len(formulas):
        raise ValueError("duplicate formula sample_id")
    if manifest.get("quality_status", {}).get("valid") != len(formulas):
        raise ValueError("valid formula count differs from manifest")

    for name, path in (("formulas_valid.jsonl", formulas_path), ("ownership_train.jsonl", ownership_path)):
        declared = manifest.get("files", {}).get(name, {})
        if declared and (
            declared.get("bytes") != path.stat().st_size
            or declared.get("sha256") != _sha256(path)
        ):
            raise ValueError(f"manifest integrity mismatch: {name}")

    owned_ids: set[str] = set()
    for annotation in ownership:
        if not annotation.get("accepted"):
            continue
        sample_id = str(annotation["sample_id"])
        if sample_id in owned_ids or sample_id not in formula_by_id:
            raise ValueError("duplicate or missing accepted ownership source")
        owned_ids.add(sample_id)
        formula = formula_by_id[sample_id]
        groups = [[int(value) for value in group] for group in annotation["groups"]]
        flat = [value for group in groups for value in group]
        if sorted(flat) != list(range(len(formula["strokes"]))):
            raise ValueError(f"ownership is not an exact stroke partition: {sample_id}")
        labels = [str(value) for value in annotation["labels"]]
        targets = [str(cell["token"]) for cell in formula.get("target_cells") or []]
        if len(groups) != len(labels) or labels != targets:
            raise ValueError(f"ownership/token contract mismatch: {sample_id}")
    return owned_ids


def audit(dataset_root: Path) -> dict[str, Any]:
    root = dataset_root.resolve()
    if any("crohme" in part.lower() for part in root.parts):
        raise ValueError("this audit accepts project-owned data only; CROHME paths are forbidden")
    formulas_path = root / "data" / "formulas_valid.jsonl"
    ownership_path = root / "data" / "ownership_train.jsonl"
    manifest_path = root / "dataset_info.json"
    for path in (formulas_path, ownership_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    formulas = _read_jsonl(formulas_path)
    ownership = _read_jsonl(ownership_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    owned_ids = _validate_inputs(
        formulas, ownership, manifest, formulas_path, ownership_path,
    )
    transcript_only = [row for row in formulas if str(row["sample_id"]) not in owned_ids]
    if any(not row.get("target_cells") for row in transcript_only):
        raise ValueError("transcript-only formula has no target token sequence")

    candidate_counts: list[int] = []
    stroke_counts: list[int] = []
    target_counts: list[int] = []
    matching_cover_counts: list[int] = []
    relation_edge_count = 0
    feasible = 0
    no_cover = 0
    for formula in transcript_only:
        strokes = sorted(formula["strokes"], key=lambda row: int(row["order"]))
        if len(strokes) > MAX_AUDIT_STROKES:
            raise ValueError(f"formula exceeds audit budget: {formula['sample_id']}")
        candidates = build_lattice(strokes, **DEFAULT_LATTICE_CONFIG)
        target_count = len(formula["target_cells"])
        counts = _cover_counts_by_group_count(candidates, len(strokes))
        target_ways = counts[target_count] if target_count < len(counts) else 0
        candidate_counts.append(len(candidates))
        stroke_counts.append(len(strokes))
        target_counts.append(target_count)
        matching_cover_counts.append(target_ways)
        relation_edge_count += len(formula.get("target_relations") or [])
        feasible += int(target_ways > 0)
        no_cover += int(not any(counts))

    def _range(values: list[int]) -> dict[str, int | float] | None:
        if not values:
            return None
        return {
            "min": min(values),
            "median": statistics.median(values),
            "max": max(values),
        }

    return {
        "schema": SCHEMA,
        "status": "geometry_only_upper_bound",
        "training_performed": False,
        "files_written": False,
        "crohme_rows": 0,
        "source": {
            "valid_formulas": len(formulas),
            "ownership_labeled_formulas": len(owned_ids),
            "transcript_only_formulas": len(transcript_only),
            "transcript_only_relation_edges": relation_edge_count,
            "formulas_sha256": _sha256(formulas_path),
            "ownership_sha256": _sha256(ownership_path),
            "dataset_info_sha256": _sha256(manifest_path),
        },
        "lattice": {
            "config": DEFAULT_LATTICE_CONFIG,
            "candidate_count_min_median_max": _range(candidate_counts),
            "stroke_count_min_median_max": _range(stroke_counts),
            "target_token_count_min_median_max": _range(target_counts),
        },
        "result": {
            "target_group_count_exact_cover_feasible": feasible,
            "target_group_count_exact_cover_rate": feasible / max(len(transcript_only), 1),
            "no_exact_cover_formulas": no_cover,
            "matching_partition_count_min_median_max": _range(matching_cover_counts),
        },
        "limits": [
            "does not use HWR logits or determine token identity",
            "does not check token order or relation correctness",
            "exact-cover cardinality is an optimistic bound, not recognition accuracy",
        ],
    }


def _self_test() -> None:
    candidates = [
        {"source_indices": [0]}, {"source_indices": [1]},
        {"source_indices": [0, 1]},
    ]
    assert _cover_counts_by_group_count(candidates, 2) == (0, 1, 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        print(json.dumps({"self_test": "pass", "crohme_rows": 0}))
        return 0
    if args.dataset_root is None:
        parser.error("--dataset-root is required")
    print(json.dumps(audit(args.dataset_root), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
