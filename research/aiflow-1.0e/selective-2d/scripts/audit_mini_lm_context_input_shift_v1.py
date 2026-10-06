#!/usr/bin/env python3
"""Audit the frozen mini-LM's clean-context vs Fast-Top-1 context shift.

This is a consumed-development diagnostic only. The oracle-context arm is an
upper bound, never a model-selection or promotion result. No model is trained.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import export_mini_formula_lm_onnx_v1 as export
import train_masked_context_reranker_v1 as masked
import train_prompt_context_reranker_v1 as prompt
from run_prompt_mini_lm_distillation_v1 import MiniFormulaLM


ROOT = Path(__file__).resolve().parents[1]
DISTILL_DIR = ROOT / "artifacts" / "hwr_failure_cause_microscope_20260928" / "mini_lm_distill_20261002_r2"
DEFAULT_OUTPUT = ROOT / "artifacts" / "hwr_failure_cause_microscope_20260928" / "mini_lm_onnx_20261002" / "context_input_shift_20261002.json"


def _inputs(rows: list[dict], targets: dict[str, dict], model: MiniFormulaLM, labels: list[str], *, oracle: bool) -> dict[str, np.ndarray]:
    label_to_index = {label: index for index, label in enumerate(labels)}
    relation_to_index = {relation: index for index, relation in enumerate(masked.RELATIONS)}
    examples = []
    for formula_id, sequence in masked._formulae(rows).items():
        truth = [str(value) for value in targets[formula_id]["tokens"]]
        if len(truth) != len(sequence):
            raise ValueError(f"oracle target/group count mismatch: {formula_id}")
        for target_index, target in enumerate(sequence):
            ids = [model.cls_id]
            mask_position = -1
            for index, row in enumerate(sequence):
                if index:
                    relation = masked._spatial_relation(sequence[index - 1], row)
                    ids.append(len(labels) + relation_to_index[relation])
                if index == target_index:
                    mask_position = len(ids)
                    ids.append(model.mask_id)
                    continue
                token = truth[index] if oracle else str(row["final_topk"][0])
                if token not in label_to_index:
                    raise ValueError(f"context token outside mini-LM vocabulary: {token}")
                ids.append(label_to_index[token])
            ids.append(model.sep_id)
            if mask_position < 0 or len(ids) > model.position_embedding.num_embeddings:
                raise ValueError(f"invalid masked context sequence: {formula_id}")
            if ids[mask_position] != model.mask_id:
                raise AssertionError("masked target position leaked a context label")
            examples.append((str(target["record_id"]), ids, mask_position))

    if {item[0] for item in examples} != {str(row["record_id"]) for row in rows}:
        raise AssertionError("context input coverage mismatch")
    width = max(len(ids) for _, ids, _ in examples)
    input_ids = np.full((len(examples), width), model.pad_id, dtype=np.int64)
    attention = np.zeros((len(examples), width), dtype=np.bool_)
    positions = np.empty(len(examples), dtype=np.int64)
    for row_index, (_, ids, position) in enumerate(examples):
        input_ids[row_index, :len(ids)] = ids
        attention[row_index, :len(ids)] = True
        positions[row_index] = position
    return {"input_ids": input_ids, "attention_mask": attention, "mask_positions": positions}


def _score(rows: list[dict], targets: dict[str, dict], inputs: dict[str, np.ndarray], model: MiniFormulaLM, labels: list[str], fusion_lambda: float) -> tuple[dict[str, str], dict]:
    logits = export._predict_torch(model, inputs).astype(np.float32, copy=False)
    log_probabilities = logits - np.logaddexp.reduce(logits, axis=1, keepdims=True)
    context = {
        str(row["record_id"]): values
        for row, values in zip(rows, log_probabilities, strict=True)
    }
    raw = masked._fused_predictions(rows, context, labels, fusion_lambda)
    prediction = prompt._strict_lock(rows, raw)
    if any(prediction[str(row["record_id"])] not in row["final_topk"] for row in rows):
        raise AssertionError("mini-LM changed a symbol outside the HWR Top-5")
    return prediction, export._formula_metrics(rows, targets, prediction)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DISTILL_DIR / "mini_formula_lm.pt")
    parser.add_argument("--distillation-report", type=Path, default=DISTILL_DIR / "mini_formula_lm_distillation_report.json")
    parser.add_argument("--summary", type=Path, default=export.DEFAULT_SUMMARY)
    parser.add_argument("--data", type=Path, default=export.DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    checkpoint, report_path = args.checkpoint.resolve(), args.distillation_report.resolve()
    summary_path, data_path, output = args.summary.resolve(), args.data.resolve(), args.output.resolve()
    if output.exists():
        parser.error(f"refusing to overwrite existing report: {output}")
    for name, path in (("checkpoint", checkpoint), ("distillation report", report_path), ("summary", summary_path), ("formula data", data_path)):
        if not path.is_file():
            parser.error(f"missing {name}: {path}")

    distill = json.loads(report_path.read_text(encoding="utf-8"))
    protocol = distill.get("protocol", {})
    if protocol.get("crohme_rows_loaded") != 0 or protocol.get("crohme_training_or_tuning") is not False:
        raise ValueError("mini-LM report does not attest CROHME exclusion")
    if export._sha256(checkpoint) != distill["student"]["checkpoint_sha256"]:
        raise ValueError("mini-LM checkpoint hash mismatch")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("crohme_training_or_tuning") is not False or summary.get("product_default_enabled") is not False:
        raise ValueError("frozen summary violates CROHME/product-off guard")
    if export._sha256(data_path) != summary["inputs"]["formulas_valid_sha256"]:
        raise ValueError("frozen formula data hash mismatch")
    raw = export._jsonl(data_path)
    if any("crohme" in str(row.get("source_partition", "")).casefold() for row in raw):
        raise ValueError("CROHME row found in diagnostic formula data")
    raw_by_id = {str(row["sample_id"]): row for row in raw}
    all_rows, all_targets = export._formula_rows(summary, raw_by_id)

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    labels = [str(value) for value in payload["labels"]]
    label_set = set(labels)
    group_exact_ids = {formula_id for formula_id, target in all_targets.items() if target["group_exact"]}
    unsupported_context_ids = {
        formula_id for formula_id in group_exact_ids
        if any(str(token) not in label_set for token in all_targets[formula_id]["tokens"])
    }
    eligible_ids = group_exact_ids - unsupported_context_ids
    rows = [row for row in all_rows if str(row["formula_id"]) in eligible_ids]
    targets = {formula_id: all_targets[formula_id] for formula_id in eligible_ids}
    if not rows or set(targets) != eligible_ids:
        raise ValueError("no exact-group formulas available for context-shift audit")
    model = MiniFormulaLM(len(labels), len(payload["relations"]), int(payload["max_positions"]))
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()

    predicted_inputs = _inputs(rows, targets, model, labels, oracle=False)
    oracle_inputs = _inputs(rows, targets, model, labels, oracle=True)
    fusion_lambda = float(protocol["fusion_lambda_frozen_from_teacher"])
    predicted, predicted_metrics = _score(rows, targets, predicted_inputs, model, labels, fusion_lambda)
    oracle, oracle_metrics = _score(rows, targets, oracle_inputs, model, labels, fusion_lambda)
    fast = {str(row["record_id"]): str(row["final_topk"][0]) for row in rows}
    fast_metrics = export._formula_metrics(rows, targets, fast)

    context_errors = {}
    for formula_id in sorted(eligible_ids):
        sequence = sorted((row for row in rows if str(row["formula_id"]) == formula_id), key=lambda row: int(row["context"]["index"]))
        truth = targets[formula_id]["tokens"]
        context_errors[formula_id] = sum(str(row["final_topk"][0]) != str(expected) for row, expected in zip(sequence, truth, strict=True))
    context_counts = list(context_errors.values())
    predicted_exact_ids = {
        formula_id for formula_id in eligible_ids
        if [predicted[str(row["record_id"])] for row in sorted((item for item in rows if str(item["formula_id"]) == formula_id), key=lambda item: int(item["context"]["index"]))] == targets[formula_id]["tokens"]
    }
    oracle_exact_ids = {
        formula_id for formula_id in eligible_ids
        if [oracle[str(row["record_id"])] for row in sorted((item for item in rows if str(item["formula_id"]) == formula_id), key=lambda item: int(item["context"]["index"]))] == targets[formula_id]["tokens"]
    }

    report = {
        "schema": "aiflow-mini-lm-context-input-shift-audit/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "consumed_development_diagnostic_only",
        "protocol": {
            "training_performed": False,
            "model_or_threshold_selection": False,
            "crohme_rows_loaded": 0,
            "current_formula_set_consumed_development_data": True,
            "oracle_context_is_upper_bound_only": True,
            "fusion_lambda_frozen": fusion_lambda,
            "teacher_lambda_source_current_formula_overlap": protocol.get("teacher_lambda_source_current_formula_overlap"),
            "warning": "Do not use these consumed-formula diagnostics for promotion or model selection; the oracle arm leaks neighboring truth labels by design.",
        },
        "input_coverage": {
            "group_exact_formulas_before_vocabulary_filter": len(group_exact_ids),
            "group_exact_formulas_audited": len(eligible_ids),
            "excluded_formulas_with_truth_context_outside_vocabulary": len(unsupported_context_ids),
            "excluded_formula_ids": sorted(unsupported_context_ids),
        },
        "provenance": {
            "checkpoint_sha256": export._sha256(checkpoint),
            "distillation_report_sha256": export._sha256(report_path),
            "summary_sha256": export._sha256(summary_path),
            "formula_data_sha256": export._sha256(data_path),
        },
        "eligible_exact_group_formulas": len(eligible_ids),
        "fast": fast_metrics,
        "predicted_top1_context": predicted_metrics,
        "oracle_truth_context_upper_bound": oracle_metrics,
        "context_input_error": {
            "formulas_with_at_least_one_wrong_fast_top1_context": sum(count > 0 for count in context_counts),
            "formula_count": len(context_counts),
            "total_wrong_context_symbols": sum(context_counts),
            "mean_wrong_context_symbols_per_formula": float(np.mean(context_counts)),
            "median_wrong_context_symbols_per_formula": float(np.median(context_counts)),
        },
        "formula_exact_context_intervention": {
            "predicted_context_exact_formula_ids": sorted(predicted_exact_ids),
            "oracle_context_exact_formula_ids": sorted(oracle_exact_ids),
            "oracle_only_exact_count": len(oracle_exact_ids - predicted_exact_ids),
            "predicted_only_exact_count": len(predicted_exact_ids - oracle_exact_ids),
        },
        "decision": {
            "training_input_mismatch_present": bool(sum(context_counts)),
            "oracle_gain_is_attributable_only_to_context_intervention": True,
            "promotion_eligible": False,
            "android_latency_verified": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({
        "event": "mini_lm_context_input_shift_audit_complete",
        "report": str(output),
        "eligible_group_exact_formulas": len(eligible_ids),
        "excluded_unsupported_truth_context_formulas": len(unsupported_context_ids),
        "fast_predicted_oracle_exact": [
            fast_metrics["baseline_top1_formula_exact"],
            predicted_metrics["reranked_formula_exact"],
            oracle_metrics["reranked_formula_exact"],
        ],
        "context_error_formulas_and_total": [sum(count > 0 for count in context_counts), sum(context_counts)],
        "crohme_rows_loaded": 0,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
