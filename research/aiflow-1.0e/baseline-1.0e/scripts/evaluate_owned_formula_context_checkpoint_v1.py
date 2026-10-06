#!/usr/bin/env python3
"""Evaluate a frozen owned formula-context checkpoint on candidate caches."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

import torch

from character_tensor_v1 import _json_lines
from evaluate_48hz_prefix_v1 import DEFAULT_PRODUCT, _sha256
from context_recheck_guard_v1 import apply_context_recheck_guard
from evaluate_homograph_context_reranker_v1 import _metrics
from finalize_formula_context_v1 import _apply_equation_correction
from semantic_fence_guard_v1 import apply_semantic_fence_guard
from semantic_infix_guard_v1 import apply_semantic_infix_guard
import train_masked_context_reranker_v1 as masked
from train_owned_formula_context_v1 import (
    _assert_candidate_contract,
    decide_owned_formula_rows_supported_exact,
    load_owned_formula_context,
)


def _scope(
    rows: list[dict], context_predictions: dict[str, str],
    final_predictions: dict[str, str],
) -> dict:
    baseline = {str(row["record_id"]): str(row["final_topk"][0]) for row in rows}
    context_metrics = _metrics(rows, context_predictions)
    final_metrics = _metrics(rows, final_predictions)
    baseline_metrics = _metrics(rows, baseline)
    audit = masked._candidate_audit(rows, final_predictions)
    _assert_candidate_contract("frozen checkpoint evaluation", audit)
    return {
        "baseline": baseline_metrics,
        "owned_context": context_metrics,
        "delta": masked._metric_delta(context_metrics, baseline_metrics),
        "semantic_finalizer": final_metrics,
        "semantic_delta_over_context": masked._metric_delta(
            final_metrics, context_metrics
        ),
        "candidate_audit": audit,
    }


def _subsets(
    rows: list[dict], context_predictions: dict[str, str],
    final_predictions: dict[str, str],
) -> dict:
    output = {"all": _scope(rows, context_predictions, final_predictions)}
    by_partition: dict[str, list[dict]] = defaultdict(list)
    by_writer: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_partition[str(row.get("evaluation_partition", "unspecified"))].append(row)
        by_writer[str(row["writer_group"])].append(row)
    for name, subset in sorted(by_partition.items()):
        output[name] = _scope(subset, context_predictions, final_predictions)
    arrivals = [
        row for row in rows if str(row.get("evaluation_partition")) != "legacy47"
    ]
    if arrivals:
        output["new_arrivals"] = _scope(
            arrivals, context_predictions, final_predictions
        )
    output["by_writer"] = {
        writer: _scope(subset, context_predictions, final_predictions)
        for writer, subset in sorted(by_writer.items())
    }
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--hwr-checkpoint", type=Path, default=DEFAULT_PRODUCT)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions-output", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--enable-equation-correction", action="store_true",
        help="research-only: allow exact arithmetic to override HWR candidates",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch size must be positive")
    checkpoint = masked._d_path(args.checkpoint, "owned context checkpoint")
    hwr_checkpoint = masked._d_path(args.hwr_checkpoint, "HWR checkpoint")
    input_root = masked._d_path(args.input, "candidate input")
    output = masked._d_path(args.output, "evaluation report")
    if output.exists():
        parser.error(f"refusing to overwrite evaluation report: {output}")
    predictions_output = (
        masked._d_path(args.predictions_output, "prediction output")
        if args.predictions_output else None
    )
    if predictions_output is not None and predictions_output.exists():
        parser.error(f"refusing to overwrite predictions: {predictions_output}")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    model, contract, payload = load_owned_formula_context(
        checkpoint, hwr_checkpoint, device
    )
    direct_path = input_root / "direct_candidates.jsonl.gz"
    direct_rows = list(_json_lines(direct_path))
    direct_context, direct_runtime = decide_owned_formula_rows_supported_exact(
        model, contract, payload, direct_rows, device, args.batch_size
    )
    probability_ratio_floor = float(
        payload["configuration"]["candidate_probability_ratio_floor"]
    )
    direct_equation, direct_equation_audit = _apply_equation_correction(
        direct_rows, direct_context, probability_ratio_floor,
        args.enable_equation_correction,
    )
    direct_fence, direct_fence_audit = apply_semantic_fence_guard(
        direct_rows, direct_equation
    )
    direct_first_predictions, direct_infix_audit = apply_semantic_infix_guard(
        direct_rows, direct_fence,
        probability_ratio_floor,
    )
    direct_recheck_context, direct_recheck_runtime = (
        decide_owned_formula_rows_supported_exact(
            model, contract, payload, direct_rows, device, args.batch_size,
            direct_first_predictions,
        )
    )
    direct_recheck_equation, direct_recheck_equation_audit = (
        _apply_equation_correction(
            direct_rows, direct_recheck_context, probability_ratio_floor,
            args.enable_equation_correction,
        )
    )
    direct_recheck_fence, direct_recheck_fence_audit = apply_semantic_fence_guard(
        direct_rows, direct_recheck_equation
    )
    direct_recheck_predictions, direct_recheck_infix_audit = (
        apply_semantic_infix_guard(
            direct_rows, direct_recheck_fence, probability_ratio_floor
        )
    )
    direct_predictions, direct_recheck_audit = apply_context_recheck_guard(
        direct_rows, direct_first_predictions, direct_recheck_predictions
    )
    if predictions_output is not None:
        predictions_output.parent.mkdir(parents=True, exist_ok=True)
        masked._write_prediction_rows(predictions_output, direct_rows, direct_predictions)
    report = {
        "schema": "aiflow-owned-formula-context-checkpoint-evaluation/v4",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "training_performed": False,
        "runtime_pipeline": [
            "owned_formula_context_r6",
            "owned_supported_exact_context_guard_v1",
        ] + (["semantic_equation_guard_v2"] if args.enable_equation_correction else []) + [
            "semantic_fence_guard_v1",
            "semantic_infix_guard_v1",
            "context_recheck_guard_v1",
        ],
        "equation_correction_enabled": args.enable_equation_correction,
        "device": str(device),
        "checkpoint": {"path": str(checkpoint), "sha256": _sha256(checkpoint)},
        "hwr_checkpoint_sha256": _sha256(hwr_checkpoint),
        "direct_candidate_sha256": _sha256(direct_path),
        "direct_predictions": (
            {"path": str(predictions_output), "sha256": _sha256(predictions_output)}
            if predictions_output is not None else None
        ),
        "direct": {
            "runtime_audit": {
                "owned_context": direct_runtime,
                "semantic_equation_guard": direct_equation_audit,
                "semantic_fence_guard": direct_fence_audit,
                "semantic_infix_guard": direct_infix_audit,
                "context_recheck_model": direct_recheck_runtime,
                "context_recheck_semantic": {
                    "equation": direct_recheck_equation_audit,
                    "fence": direct_recheck_fence_audit,
                    "infix": direct_recheck_infix_audit,
                },
                "context_recheck_guard": direct_recheck_audit,
            },
            "evaluation": _subsets(
                direct_rows, direct_context, direct_predictions
            ),
        },
    }
    crohme_path = input_root / "crohme_candidates.jsonl.gz"
    if crohme_path.is_file():
        crohme_rows = list(_json_lines(crohme_path))
        crohme_context, crohme_runtime = decide_owned_formula_rows_supported_exact(
            model, contract, payload, crohme_rows, device, args.batch_size
        )
        crohme_equation, crohme_equation_audit = _apply_equation_correction(
            crohme_rows, crohme_context, probability_ratio_floor,
            args.enable_equation_correction,
        )
        crohme_fence, crohme_fence_audit = apply_semantic_fence_guard(
            crohme_rows, crohme_equation
        )
        crohme_first_predictions, crohme_infix_audit = apply_semantic_infix_guard(
            crohme_rows, crohme_fence,
            probability_ratio_floor,
        )
        crohme_recheck_context, crohme_recheck_runtime = (
            decide_owned_formula_rows_supported_exact(
                model, contract, payload, crohme_rows, device, args.batch_size,
                crohme_first_predictions,
            )
        )
        crohme_recheck_equation, crohme_recheck_equation_audit = (
            _apply_equation_correction(
                crohme_rows, crohme_recheck_context, probability_ratio_floor,
                args.enable_equation_correction,
            )
        )
        crohme_recheck_fence, crohme_recheck_fence_audit = (
            apply_semantic_fence_guard(crohme_rows, crohme_recheck_equation)
        )
        crohme_recheck_predictions, crohme_recheck_infix_audit = (
            apply_semantic_infix_guard(
                crohme_rows, crohme_recheck_fence, probability_ratio_floor
            )
        )
        crohme_predictions, crohme_recheck_audit = apply_context_recheck_guard(
            crohme_rows, crohme_first_predictions, crohme_recheck_predictions
        )
        report["crohme"] = {
            "candidate_sha256": _sha256(crohme_path),
            "runtime_audit": {
                "owned_context": crohme_runtime,
                "semantic_equation_guard": crohme_equation_audit,
                "semantic_fence_guard": crohme_fence_audit,
                "semantic_infix_guard": crohme_infix_audit,
                "context_recheck_model": crohme_recheck_runtime,
                "context_recheck_semantic": {
                    "equation": crohme_recheck_equation_audit,
                    "fence": crohme_recheck_fence_audit,
                    "infix": crohme_recheck_infix_audit,
                },
                "context_recheck_guard": crohme_recheck_audit,
            },
            "evaluation": _scope(
                crohme_rows, crohme_context, crohme_predictions
            ),
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    direct = report["direct"]["evaluation"]
    print(json.dumps({
        "report": str(output),
        "all": direct["all"]["semantic_finalizer"],
        "new_arrivals": direct.get("new_arrivals", {}).get("semantic_finalizer"),
        "candidate_preservation": direct["all"]["candidate_audit"]["candidate_preservation_rate"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
