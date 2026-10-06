#!/usr/bin/env python3
"""Compute a reproducible lower bound for untouched grouping acceptance data."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from statistics import NormalDist

from character_tensor_v1 import ROOT


def wilson_lower(successes: int, total: int, confidence: float) -> float:
    if not 0 <= successes <= total or total < 1:
        raise ValueError("invalid binomial counts")
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    probability = successes / total
    denominator = 1.0 + z * z / total
    center = probability + z * z / (2.0 * total)
    spread = z * math.sqrt(
        (probability * (1.0 - probability) + z * z / (4.0 * total)) / total
    )
    return (center - spread) / denominator


def minimum_formulas(
    observed_target: float, lower_bound_target: float, confidence: float,
) -> tuple[int, int, float]:
    for total in range(1, 1_000_001):
        successes = math.ceil(observed_target * total)
        lower = wilson_lower(successes, total, confidence)
        if successes / total >= observed_target and lower >= lower_bound_target:
            return total, successes, lower
    raise RuntimeError("sample-size search did not converge")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observed-target", type=float, default=0.90)
    parser.add_argument("--lower-bound-target", type=float, default=0.85)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--max-formulas-per-writer", type=int, default=20)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "artifacts" / "grouping_collection_gate_20260822_r1"
        / "collection_gate.json",
    )
    args = parser.parse_args()
    if not (
        0 < args.lower_bound_target < args.observed_target < 1
        and 0 < args.confidence < 1
        and args.max_formulas_per_writer > 0
    ):
        parser.error("invalid statistical gate")
    output = args.output.resolve()
    if output.drive.upper() != "D:" or output.exists():
        parser.error("output must be a new file on D:")
    total, successes, lower = minimum_formulas(
        args.observed_target, args.lower_bound_target, args.confidence,
    )
    practical_total = math.ceil(total / args.max_formulas_per_writer) * args.max_formulas_per_writer
    minimum_writers = math.ceil(total / args.max_formulas_per_writer)
    clustering = {}
    for icc in (0.05, 0.10, 0.20):
        design_effect = 1.0 + (args.max_formulas_per_writer - 1) * icc
        clustered_total = math.ceil(total * design_effect)
        clustering[f"icc_{icc:.2f}"] = {
            "assumption_only": True,
            "design_effect": design_effect,
            "formulas": clustered_total,
            "writers_at_cap": math.ceil(clustered_total / args.max_formulas_per_writer),
        }
    report = {
        "schema": "aiflow-grouping-collection-gate/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "new untouched product-owned/commercial grouping acceptance only",
        "gate": {
            "observed_partition_exact_target": args.observed_target,
            "wilson_lower_bound_target": args.lower_bound_target,
            "confidence": args.confidence,
            "minimum_iid_formulas": total,
            "minimum_successes_at_that_size": successes,
            "achieved_observed_rate": successes / total,
            "achieved_wilson_lower_bound": lower,
            "practical_formula_batch_at_writer_cap": practical_total,
            "max_formulas_per_writer": args.max_formulas_per_writer,
            "minimum_writers_at_cap": minimum_writers,
        },
        "writer_clustering_sensitivity": clustering,
        "collection_contract": {
            "writers_disjoint_from_training_and_prior_inspection": True,
            "exact_stroke_partition_required": True,
            "stroke_order_and_timestamps_required": True,
            "formula_complete_boundary_required": True,
            "two_dimensional_relation_labels_required": True,
            "raw_ink_immutable": True,
            "crohme_substitution_allowed": False,
        },
        "interpretation": {
            "minimum_iid_formulas_is_lower_bound": True,
            "training_quantity_proven_by_this_calculation": False,
            "next_training_tranche_recommendation": practical_total,
            "stop_rule": (
                "repeat project-owned training tranches; never inspect the untouched acceptance set "
                "for model/epoch/threshold selection; stop only when the frozen candidate reaches the gate"
            ),
            "writer_level_requirement": (
                "measure intraclass correlation on new data and apply the reported design-effect formula"
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    print(json.dumps({
        "output": str(output), "minimum_iid_formulas": total,
        "practical_batch": practical_total, "minimum_writers_at_cap": minimum_writers,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
