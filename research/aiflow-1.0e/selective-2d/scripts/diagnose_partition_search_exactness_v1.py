#!/usr/bin/env python3
"""Forensic comparison of beam and exact bounded partition enumeration."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from stroke_grouping_v1 import (
    build_lattice, candidate_features, enumerate_partitions,
    enumerate_partitions_exact,
)


SCHEMA = "aiflow-partition-search-exactness-diagnostic/v1"
OPTIONS_PER_STROKE = 64
TOP_N = 32


def _read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _partition_key(groups) -> tuple[tuple[int, ...], ...]:
    return tuple(sorted(tuple(sorted(int(index) for index in group)) for group in groups))


def diagnose(dataset_root: Path, ranker_path: Path) -> dict[str, Any]:
    data_dir = dataset_root / "data"
    formulas_path = data_dir / "formulas_valid.jsonl"
    ownership_path = data_dir / "ownership_train.jsonl"
    formulas = {str(row["sample_id"]): row for row in _read_rows(formulas_path)}
    annotations = [row for row in _read_rows(ownership_path) if row.get("accepted")]
    grouping_model = joblib.load(ranker_path)["grouping_model"]
    records = []
    counts: Counter[str] = Counter()
    beam_ranks: Counter[str] = Counter()
    exact_ranks: Counter[str] = Counter()

    for annotation in annotations:
        sample_id = str(annotation["sample_id"])
        source = formulas[sample_id]
        strokes = sorted(source["strokes"], key=lambda row: int(row["order"]))
        target = [tuple(sorted(int(index) for index in group)) for group in annotation["groups"]]
        lattice = build_lattice(strokes, temporal_window=6, spatial_neighbors=4)
        probabilities = grouping_model.predict_proba(candidate_features(lattice, strokes))[:, 1]
        clipped = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
        scores = np.log(clipped / (1.0 - clipped)).tolist()
        score_by_group = {
            tuple(sorted(int(index) for index in row["source_indices"])): float(score)
            for row, score in zip(lattice, scores, strict=True)
        }
        all_target_edges_present = all(group in score_by_group for group in target)
        beam = enumerate_partitions(
            lattice, scores, len(strokes), top_n=TOP_N, beam_width=32,
            options_per_stroke=OPTIONS_PER_STROKE,
        )
        beam_rank = next((
            rank for rank, (_score, groups) in enumerate(beam, 1)
            if _partition_key(groups) == _partition_key(target)
        ), None)
        beam_ranks[str(beam_rank or ">32")] += 1

        exact_rank = None
        exact_status = "exact_stroke_cap_exceeded" if len(strokes) > 12 else "exact_not_run"
        target_edges_survive_cap = None
        if len(strokes) <= 12:
            retained_by_first: dict[int, set[tuple[int, ...]]] = {}
            for first in range(len(strokes)):
                eligible = [group for group in score_by_group if first in group]
                eligible.sort(key=lambda group: (-score_by_group[group], group))
                retained_by_first[first] = set(eligible[:OPTIONS_PER_STROKE])
            target_edges_survive_cap = all(
                group in retained_by_first[min(group)] for group in target
            )
            try:
                exact = enumerate_partitions_exact(
                    lattice, scores, len(strokes), top_n=TOP_N,
                    options_per_stroke=OPTIONS_PER_STROKE, max_strokes=12,
                )
            except ValueError:
                if not all_target_edges_present:
                    exact_status = "target_group_missing_from_lattice"
                elif not target_edges_survive_cap:
                    exact_status = "target_edge_removed_by_per_stroke_cap"
                else:
                    exact_status = "no_cover_with_per_stroke_cap"
            else:
                exact_rank = next((
                    rank for rank, (_score, groups) in enumerate(exact, 1)
                    if _partition_key(groups) == _partition_key(target)
                ), None)
                exact_status = "target_in_exact_top32" if exact_rank is not None else "target_below_exact_top32"
                if not all_target_edges_present:
                    exact_status = "target_group_missing_from_lattice"
                elif not target_edges_survive_cap:
                    exact_status = "target_edge_removed_by_per_stroke_cap"
                elif beam_rank is None and exact_rank is not None:
                    exact_status = "beam_pruned_target_that_exact_search_keeps"
                exact_ranks[str(exact_rank or ">32")] += 1

        counts["formulas"] += 1
        counts["target_groups"] += len(target)
        counts["target_groups_present_in_lattice"] += sum(group in score_by_group for group in target)
        counts["formulas_all_target_edges_survive_per_stroke_cap"] += (
            int(bool(target_edges_survive_cap)) if target_edges_survive_cap is not None else 0
        )
        counts["beam_missed_target"] += int(beam_rank is None)
        counts["beam_miss_exactly_explained_by_pruning"] += int(
            beam_rank is None and exact_rank is not None
        )
        counts["exact_eligible_formulas"] += int(len(strokes) <= 12)
        counts[f"exact_status:{exact_status}"] += 1
        records.append({
            "sample_id": sample_id,
            "stroke_count": len(strokes),
            "lattice_group_count": len(lattice),
            "beam_target_rank": beam_rank,
            "exact_target_rank": exact_rank,
            "target_group_candidate_recall": sum(group in score_by_group for group in target) / max(len(target), 1),
            "target_edges_survive_per_stroke_cap": target_edges_survive_cap,
            "exact_status": exact_status,
        })

    return {
        "schema": SCHEMA,
        "scope": "forensic failure-cause analysis only; frozen data, no fitting, threshold selection, CROHME, or promotion",
        "search_contract": {
            "beam_width": 32,
            "top_n": TOP_N,
            "options_per_stroke": OPTIONS_PER_STROKE,
            "exact_max_strokes": 12,
            "exactness": "exact over the retained per-stroke candidate cap",
        },
        "inputs": {
            "dataset_root": str(dataset_root.resolve()),
            "formulas_sha256": _sha256(formulas_path),
            "ownership_sha256": _sha256(ownership_path),
            "partition_ranker_sha256": _sha256(ranker_path),
        },
        "summary": {
            "counts": dict(sorted(counts.items())),
            "beam_target_rank_histogram": dict(sorted(beam_ranks.items())),
            "exact_target_rank_histogram": dict(sorted(exact_ranks.items())),
        },
        "formula_diagnostics": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--partition-ranker", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = diagnose(args.dataset_root, args.partition_ranker)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
