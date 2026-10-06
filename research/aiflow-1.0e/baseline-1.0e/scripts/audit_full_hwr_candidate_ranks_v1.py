#!/usr/bin/env python3
"""Audit full 372-class HWR truth ranks without widening runtime candidates."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import torch

from build_expanded_owned_context_candidates_v1 import (
    _items,
    _legacy_writer_map,
    _score,
)
from evaluate_48hz_prefix_v1 import DEFAULT_BASE, DEFAULT_LOO, DEFAULT_PRODUCT, _sha256
from evaluate_homograph_context_reranker_v1 import _d_path


SCHEMA = "aiflow-full-hwr-candidate-rank-audit/v1"
CUTOFFS = (5, 10, 20, 50, 100, 372)


def _admitted(items: list[dict], labels: list[str]) -> list[dict]:
    supported = set(labels)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in items:
        grouped[str(row["formula_id"])].append(row)
    admitted_formulae = {
        formula_id for formula_id, rows in grouped.items()
        if all(str(row["label"]) in supported for row in rows)
    }
    return [row for row in items if str(row["formula_id"]) in admitted_formulae]


def _rank(row: dict, scores: dict[str, dict]) -> tuple[int, float]:
    record_id = str(row["record_id"])
    truth = str(row["label"])
    tokens = [str(value) for value in scores[record_id]["final_topk"]]
    probabilities = [
        float(value) for value in scores[record_id]["final_topk_probabilities"]
    ]
    index = tokens.index(truth)
    return index + 1, probabilities[index]


def _metrics(items: list[dict], ranks: dict[str, int]) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in items:
        grouped[str(row["formula_id"])].append(row)
    return {
        "records": len(items),
        "formulas": len(grouped),
        "cutoffs": {
            str(cutoff): {
                "character_truth_count": sum(
                    ranks[str(row["record_id"])] <= cutoff for row in items
                ),
                "formula_oracle_count": sum(
                    all(ranks[str(row["record_id"])] <= cutoff for row in rows)
                    for rows in grouped.values()
                ),
            }
            for cutoff in CUTOFFS
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--legacy-dataset-root", type=Path,
        default=Path(__file__).resolve().parents[1] / "hf-dataset",
    )
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--loo-heads", type=Path, default=DEFAULT_LOO)
    parser.add_argument("--product-checkpoint", type=Path, default=DEFAULT_PRODUCT)
    parser.add_argument("--expanded-loo-heads", type=Path, required=True)
    parser.add_argument("--weights", type=float, nargs="+", default=(0.0, 0.6, 1.0))
    parser.add_argument("--reference-weight", type=float, default=0.6)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        not args.weights
        or any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in args.weights)
        or len(set(args.weights)) != len(args.weights)
    ):
        parser.error("weights must be unique finite values in [0,1]")
    if args.reference_weight not in args.weights:
        parser.error("reference weight must be included in weights")
    candidate_root = _d_path(args.dataset_root, "expanded dataset")
    legacy_root = _d_path(args.legacy_dataset_root, "legacy dataset")
    base_path = _d_path(args.base_checkpoint, "base HWR checkpoint")
    loo_path = _d_path(args.loo_heads, "writer-LOO heads")
    product_path = _d_path(args.product_checkpoint, "product HWR checkpoint")
    expanded_path = _d_path(args.expanded_loo_heads, "expanded writer-LOO heads")
    output_path = _d_path(args.output, "full-rank audit")
    if output_path.exists():
        parser.error(f"refusing to overwrite full-rank audit: {output_path}")
    device_name = (
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    if device_name == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    device = torch.device(device_name)
    writer_map, legacy_summary = _legacy_writer_map(candidate_root, legacy_root)
    items, _, policy = _items(
        candidate_root, writer_map, int(legacy_summary["legacy_formulas"])
    )
    by_weight = {}
    labels = None
    admitted_items = None
    for weight in args.weights:
        scores, checkpoints, current_labels = _score(
            items if admitted_items is None else admitted_items,
            base_path, loo_path, product_path, device,
            expanded_path, float(weight), top_k=372,
        )
        if labels is None:
            labels = current_labels
            admitted_items = _admitted(items, labels)
            admitted_ids = {str(row["record_id"]) for row in admitted_items}
            scores = {key: value for key, value in scores.items() if key in admitted_ids}
        elif current_labels != labels:
            raise ValueError("HWR label order changed across weights")
        ranks = {
            str(row["record_id"]): _rank(row, scores)[0]
            for row in admitted_items
        }
        probabilities = {
            str(row["record_id"]): _rank(row, scores)[1]
            for row in admitted_items
        }
        by_weight[str(weight)] = {
            "weight": float(weight),
            "metrics": _metrics(admitted_items, ranks),
            "ranks": ranks,
            "probabilities": probabilities,
            "checkpoints": checkpoints,
        }
    reference = by_weight[str(args.reference_weight)]
    target_formulae = {
        str(row["formula_id"]) for row in admitted_items
        if reference["ranks"][str(row["record_id"])] > 20
    }
    targets = []
    for row in admitted_items:
        if str(row["formula_id"]) not in target_formulae:
            continue
        record_id = str(row["record_id"])
        targets.append({
            "record_id": record_id,
            "formula_id": str(row["formula_id"]),
            "truth": str(row["label"]),
            "writer_group": str(row["writer_group"]),
            "evaluation_partition": str(row["evaluation_partition"]),
            "truth_rank_by_weight": {
                key: value["ranks"][record_id] for key, value in by_weight.items()
            },
            "truth_probability_by_weight": {
                key: value["probabilities"][record_id]
                for key, value in by_weight.items()
            },
        })
    report = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "training_performed": False,
        "runtime_candidate_width_changed": False,
        "vocabulary_size": len(labels),
        "weights": [float(value) for value in args.weights],
        "reference_weight": float(args.reference_weight),
        "legacy_prefix": legacy_summary,
        **policy,
        "metrics_by_weight": {
            key: value["metrics"] for key, value in by_weight.items()
        },
        "target_formulae": sorted(target_formulae),
        "target_rows": targets,
        "inputs": {
            "dataset_root": str(candidate_root),
            "base_checkpoint_sha256": _sha256(base_path),
            "writer_loo_heads_sha256": _sha256(loo_path),
            "product_checkpoint_sha256": _sha256(product_path),
            "expanded_writer_loo_heads_sha256": _sha256(expanded_path),
        },
        "limits": [
            "full ranks are diagnostic and do not expand runtime candidates",
            "direct formulas are repeatedly observed development evidence",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    print(json.dumps({
        "output": str(output_path), "sha256": _sha256(output_path),
        "runtime_candidate_width_changed": False,
        "metrics_by_weight": report["metrics_by_weight"],
        "target_formulae": report["target_formulae"],
        "target_rows": targets,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
