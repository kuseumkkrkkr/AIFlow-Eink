#!/usr/bin/env python3
"""Writer-LOFO calibration diagnostic for the frozen candidate-preserving mini-LM."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

import audit_mini_lm_context_input_shift_v1 as context_audit
import export_mini_formula_lm_onnx_v1 as export
import train_masked_context_reranker_v1 as masked
import train_prompt_context_reranker_v1 as prompt
from run_prompt_mini_lm_distillation_v1 import MiniFormulaLM


ROOT = Path(__file__).resolve().parents[1]
DISTILL_DIR = ROOT / "artifacts" / "hwr_failure_cause_microscope_20260928" / "mini_lm_distill_20261002_r2"
DEFAULT_CHECKPOINT = DISTILL_DIR / "mini_formula_lm.pt"
DEFAULT_DISTILLATION_REPORT = DISTILL_DIR / "mini_formula_lm_distillation_report.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "hwr_failure_cause_microscope_20260928" / "mini_lm_onnx_20261002" / "writer_nested_calibration_20261003_v2.json"
LAMBDA_GRID = tuple(index / 10 for index in range(11))


def _formula_exact_by_id(rows: list[dict], targets: dict[str, dict], predictions: dict[str, str]) -> dict[str, bool]:
    by_formula: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_formula[str(row["formula_id"])].append(row)
    exact: dict[str, bool] = {}
    for formula_id, target in targets.items():
        sequence = sorted(by_formula[formula_id], key=lambda row: int(row["context"]["index"]))
        target_tokens = [str(token) for token in target["tokens"]]
        exact[formula_id] = bool(
            target["group_exact"]
            and len(sequence) == len(target_tokens)
            and all(predictions.get(str(row["record_id"])) == token for row, token in zip(sequence, target_tokens, strict=True))
        )
    return exact


def _bootstrap_writer_delta(rows: list[dict], targets: dict[str, dict], before: dict[str, bool], after: dict[str, bool], draws: int = 20000) -> list[float]:
    by_writer: dict[str, list[float]] = defaultdict(list)
    for formula_id in targets:
        writer = str(targets[formula_id]["writer_id"])
        by_writer[writer].append(float(after[formula_id]) - float(before[formula_id]))
    writers = sorted(by_writer)
    rng = random.Random(20261003)
    deltas = []
    for _ in range(draws):
        chosen = [rng.choice(writers) for _ in writers]
        numerator = sum(sum(by_writer[writer]) for writer in chosen)
        denominator = sum(len(by_writer[writer]) for writer in chosen)
        deltas.append(100.0 * numerator / denominator)
    deltas.sort()
    return [deltas[int(0.025 * draws)], deltas[int(0.975 * draws) - 1]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--distillation-report", type=Path, default=DEFAULT_DISTILLATION_REPORT)
    parser.add_argument("--summary", type=Path, default=export.DEFAULT_SUMMARY)
    parser.add_argument("--data", type=Path, default=export.DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite nested calibration report: {args.output}")

    checkpoint_path, distill_path = args.checkpoint.resolve(), args.distillation_report.resolve()
    summary_path, data_path = args.summary.resolve(), args.data.resolve()
    for name, path in (("checkpoint", checkpoint_path), ("distillation report", distill_path), ("summary", summary_path), ("formula data", data_path)):
        if not path.is_file():
            parser.error(f"missing {name}: {path}")
    distill = json.loads(distill_path.read_text(encoding="utf-8"))
    protocol = distill.get("protocol", {})
    if protocol.get("crohme_rows_loaded") != 0 or protocol.get("crohme_training_or_tuning") is not False:
        raise ValueError("distillation report does not attest CROHME exclusion")
    if export._sha256(checkpoint_path) != distill["student"]["checkpoint_sha256"]:
        raise ValueError("mini-LM checkpoint hash differs from its report")
    if export._sha256(data_path) != distill["provenance"]["current_formula_data_sha256"]:
        raise ValueError("formula data hash differs from its report")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("crohme_training_or_tuning") is not False or summary.get("product_default_enabled") is not False:
        raise ValueError("frozen summary violates CROHME/product-off guard")
    raw = export._jsonl(data_path)
    if any("crohme" in str(row.get("source_partition", "")).casefold() for row in raw):
        raise ValueError("CROHME-marked row found in diagnostic formula data")
    raw_by_id = {str(row["sample_id"]): row for row in raw}
    all_rows, targets = export._formula_rows(summary, raw_by_id)
    if len(targets) != 149:
        raise ValueError(f"expected the consumed 149-formula diagnostic, found {len(targets)}")

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    labels = [str(value) for value in payload["labels"]]
    model = MiniFormulaLM(len(labels), len(payload["relations"]), int(payload["max_positions"]))
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    # The mini-LM cannot repair grouping, so only exact-group formulas enter its context path.
    exact_group_ids = {formula_id for formula_id, target in targets.items() if target["group_exact"]}
    context_rows = [row for row in all_rows if str(row["formula_id"]) in exact_group_ids]
    context_targets = {formula_id: targets[formula_id] for formula_id in exact_group_ids}
    inputs = context_audit._inputs(context_rows, context_targets, model, labels, oracle=False)
    # This 149-formula diagnostic is small; keep inference deterministic on CPU.
    logits = export._predict_torch(model.eval(), inputs).astype(np.float32, copy=False)
    log_probabilities = logits - np.logaddexp.reduce(logits, axis=1, keepdims=True)
    context_logp = {
        str(row["record_id"]): values
        for row, values in zip(context_rows, log_probabilities, strict=True)
    }

    predictions_by_lambda: dict[float, dict[str, str]] = {}
    exact_by_lambda: dict[float, dict[str, bool]] = {}
    metrics_by_lambda: dict[str, dict] = {}
    for fusion_lambda in LAMBDA_GRID:
        raw_predictions = {
            str(row["record_id"]): str(row["final_topk"][0]) for row in all_rows
        }
        raw_predictions.update(masked._fused_predictions(context_rows, context_logp, labels, fusion_lambda))
        predictions = prompt._strict_lock(all_rows, raw_predictions)
        if any(predictions[str(row["record_id"])] not in row["final_topk"] for row in all_rows):
            raise AssertionError("mini-LM escaped the HWR Top-5 candidate set")
        predictions_by_lambda[fusion_lambda] = predictions
        exact_by_lambda[fusion_lambda] = _formula_exact_by_id(all_rows, targets, predictions)
        metrics_by_lambda[f"{fusion_lambda:.1f}"] = export._formula_metrics(all_rows, targets, predictions)

    writers_by_formula = {formula_id: str(target["writer_id"]) for formula_id, target in targets.items()}
    writers = sorted(set(writers_by_formula.values()))
    if len(writers) < 3:
        raise ValueError("writer-level nested calibration requires at least three writer groups")
    fixed_lambda = float(protocol["fusion_lambda_frozen_from_teacher"])
    if fixed_lambda not in predictions_by_lambda:
        raise ValueError("teacher lambda must be represented on the fixed sweep grid")

    folds = []
    nested_predictions: dict[str, str] = {}
    nested_exact: dict[str, bool] = {}
    for held_writer in writers:
        train_ids = [formula_id for formula_id in targets if writers_by_formula[formula_id] != held_writer]
        held_ids = [formula_id for formula_id in targets if writers_by_formula[formula_id] == held_writer]
        if not train_ids or not held_ids:
            raise AssertionError("empty writer-held-out calibration fold")
        fit_exact_by_lambda = {}
        for fusion_lambda in LAMBDA_GRID:
            hits = sum(exact_by_lambda[fusion_lambda][formula_id] for formula_id in train_ids)
            fit_exact_by_lambda[fusion_lambda] = hits
        best_fit_exact = max(fit_exact_by_lambda.values())
        best_lambda_tie_set = [value for value, hits in fit_exact_by_lambda.items() if hits == best_fit_exact]
        # Favor the Fast model on ties; this selection rule does not depend on the overlapping teacher cache.
        selected_lambda = min(best_lambda_tie_set)
        fold_predictions = predictions_by_lambda[selected_lambda]
        for formula_id in held_ids:
            nested_predictions.update({
                str(row["record_id"]): fold_predictions[str(row["record_id"])]
                for row in all_rows if str(row["formula_id"]) == formula_id
            })
            nested_exact[formula_id] = exact_by_lambda[selected_lambda][formula_id]
        folds.append({
            "held_writer_hash": hashlib.sha256(held_writer.encode("utf-8")).hexdigest()[:12],
            "fit_formula_count": len(train_ids),
            "held_formula_count": len(held_ids),
            "selected_lambda": selected_lambda,
            "fit_exact_count": best_fit_exact,
            "best_lambda_tie_set": best_lambda_tie_set,
            "fit_exact_by_lambda": {f"{value:.1f}": hits for value, hits in fit_exact_by_lambda.items()},
            "held_exact_count": sum(nested_exact[formula_id] for formula_id in held_ids),
            "fixed_lambda_held_exact_count": sum(exact_by_lambda[fixed_lambda][formula_id] for formula_id in held_ids),
        })

    baseline = {formula_id: bool(target["group_exact"]) and all(
        str(row["final_topk"][0]) == str(target_token)
        for row, target_token in zip(
            sorted((row for row in all_rows if str(row["formula_id"]) == formula_id), key=lambda row: int(row["context"]["index"])),
            targets[formula_id]["tokens"], strict=True,
        )
    ) for formula_id, target in targets.items()}
    fixed_exact = exact_by_lambda[fixed_lambda]
    if sum(baseline.values()) != int(metrics_by_lambda["0.0"]["baseline_top1_formula_exact"]):
        raise AssertionError("constructed Fast formula exact count differs from the evaluator")
    fixed_metric = metrics_by_lambda[f"{fixed_lambda:.1f}"]
    nested_hits = sum(nested_exact.values())
    fixed_hits = sum(fixed_exact.values())
    baseline_hits = sum(baseline.values())
    nested_ci = _bootstrap_writer_delta(all_rows, targets, baseline, nested_exact)
    fixed_ci = _bootstrap_writer_delta(all_rows, targets, baseline, fixed_exact)

    report = {
        "schema": "aiflow-mini-lm-writer-nested-calibration/v1",
        "status": "consumed_development_writer_lofo_diagnostic",
        "protocol": {
            "training_performed": False,
            "lambda_selection": "leave-one-writer-out; each held writer's formulas excluded; equal-fit ties choose the smallest context weight (Fast-favoring), independent of teacher lambda",
            "model_or_architecture_selection_performed": False,
            "fusion_grid": list(LAMBDA_GRID),
            "crohme_rows_loaded": 0,
            "current_formula_data_consumed_development": True,
            "teacher_lambda_source_current_formula_overlap": protocol.get("teacher_lambda_source_current_formula_overlap"),
            "promotion_eligible": False,
            "warning": "writer nesting reduces within-cohort tuning leakage but the 149-formula cohort is already consumed development data; this is not independent acceptance",
        },
        "provenance": {
            "checkpoint_sha256": export._sha256(checkpoint_path),
            "distillation_report_sha256": export._sha256(distill_path),
            "summary_sha256": export._sha256(summary_path),
            "formula_data_sha256": export._sha256(data_path),
            "formula_count": len(targets),
            "writer_count": len(writers),
        },
        "metrics": {
            "fast_formula_exact": baseline_hits,
            "fixed_teacher_lambda": fixed_lambda,
            "fixed_lambda_formula_exact": fixed_hits,
            "fixed_lambda_delta_over_fast": fixed_hits - baseline_hits,
            "fixed_lambda_writer_bootstrap_delta_pp_95_ci": fixed_ci,
            "writer_lofo_formula_exact": nested_hits,
            "writer_lofo_delta_over_fast": nested_hits - baseline_hits,
            "writer_lofo_writer_bootstrap_delta_pp_95_ci": nested_ci,
            "candidate_preservation_rate": fixed_metric["candidate_preservation_rate"],
            "fixed_lambda_changed_tokens": fixed_metric["changed_tokens"],
            "fixed_lambda_improved_tokens": fixed_metric["improved_tokens"],
            "fixed_lambda_regressed_tokens": fixed_metric["regressed_tokens"],
        },
        "lambda_sweep": {
            key: {
                "formula_exact": value["reranked_formula_exact"],
                "delta_over_fast": value["delta_formula_exact"],
                "changed_tokens": value["changed_tokens"],
                "improved_tokens": value["improved_tokens"],
                "regressed_tokens": value["regressed_tokens"],
            }
            for key, value in metrics_by_lambda.items()
        },
        "folds": folds,
        "product_adopted": False,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({
        "event": "mini_lm_writer_nested_calibration_complete",
        "report": str(output),
        "writers": len(writers),
        "fast_exact": baseline_hits,
        "fixed_lambda_exact": fixed_hits,
        "writer_lofo_exact": nested_hits,
        "writer_lofo_ci": nested_ci,
        "crohme_rows": 0,
        "promotion_eligible": False,
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
