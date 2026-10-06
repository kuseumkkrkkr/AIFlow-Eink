#!/usr/bin/env python3
"""Writer-LOFO shadow audit of the mini-LM strict-homograph lock policy.

This compares the existing strict Top-1 lock with a Top-5-preserving unlock.
It does not train a model and only uses the already-consumed 149-formula
development cohort for a diagnostic, never CROHME or product promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch

import audit_mini_lm_writer_nested_calibration_v1 as nested
import audit_mini_lm_context_input_shift_v1 as context_audit
import export_mini_formula_lm_onnx_v1 as export
import run_prompt_mini_lm_distillation_v1 as mini_lm_trainer
import train_masked_context_reranker_v1 as masked
import train_prompt_context_reranker_v1 as prompt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT / "artifacts" / "hwr_failure_cause_microscope_20260928"
    / "mini_lm_onnx_20261002" / "homograph_unlock_onnx_parity_writer_lofo_20261003_v2.json"
)
LAMBDA_GRID = tuple(index / 10 for index in range(11))
MARGIN_THRESHOLDS: tuple[float | None, ...] = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, None)


def _threshold_key(value: float | None) -> str:
    return "strict_lock" if value is None else f"{value:.2f}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bootstrap_delta_by_writer(
    targets: dict[str, dict], baseline: dict[str, bool], candidate: dict[str, bool],
    *, salt: int,
) -> list[float]:
    by_writer: dict[str, list[float]] = {}
    for formula_id, target in targets.items():
        by_writer.setdefault(str(target["writer_id"]), []).append(
            float(candidate[formula_id]) - float(baseline[formula_id])
        )
    writers = sorted(by_writer)
    rng = random.Random(20261003 + salt)
    draws = 20000
    deltas = []
    for _ in range(draws):
        chosen = [rng.choice(writers) for _ in writers]
        numerator = sum(sum(by_writer[writer]) for writer in chosen)
        denominator = sum(len(by_writer[writer]) for writer in chosen)
        deltas.append(100.0 * numerator / denominator)
    deltas.sort()
    return [deltas[int(0.025 * draws)], deltas[int(0.975 * draws) - 1]]


def _formula_exact(rows: list[dict], targets: dict[str, dict], predictions: dict[str, str]) -> dict[str, bool]:
    return nested._formula_exact_by_id(rows, targets, predictions)


def _fuse_with_formula_lambdas(
    logits: np.ndarray, scoring_rows: list[dict], all_rows: list[dict], labels: list[str],
    lambda_by_formula: dict[str, float], fast_predictions: dict[str, str],
    *, output_rows: list[dict] | None = None,
) -> dict[str, str]:
    log_probabilities = logits.astype(np.float32, copy=False)
    log_probabilities = log_probabilities - np.logaddexp.reduce(log_probabilities, axis=1, keepdims=True)
    context_rows = scoring_rows if output_rows is None else output_rows
    context = {
        str(row["record_id"]): values
        for row, values in zip(context_rows, log_probabilities, strict=True)
    }
    predictions = fast_predictions.copy()
    for fusion_lambda in sorted(set(lambda_by_formula.values())):
        lambda_rows = [
            row for row in scoring_rows
            if lambda_by_formula[str(row["formula_id"])] == fusion_lambda
        ]
        predictions.update(masked._fused_predictions(
            lambda_rows,
            {str(row["record_id"]): context[str(row["record_id"])] for row in lambda_rows},
            labels,
            fusion_lambda,
        ))
    if any(predictions[str(row["record_id"])] not in row["final_topk"] for row in all_rows):
        raise AssertionError("iterative backend emitted a symbol outside frozen HWR Top-5")
    return predictions


def _nested_policy(
    rows: list[dict], targets: dict[str, dict], predictions_by_lambda: dict[float, dict[str, str]],
    exact_by_lambda: dict[float, dict[str, bool]], policy: str,
) -> tuple[dict[str, str], dict[str, bool], list[dict], dict[str, float]]:
    writer_by_formula = {
        formula_id: str(target["writer_id"]) for formula_id, target in targets.items()
    }
    writers = sorted(set(writer_by_formula.values()))
    nested_predictions: dict[str, str] = {}
    nested_exact: dict[str, bool] = {}
    lambda_by_formula: dict[str, float] = {}
    folds = []
    for held_writer in writers:
        fit_ids = [key for key in targets if writer_by_formula[key] != held_writer]
        held_ids = [key for key in targets if writer_by_formula[key] == held_writer]
        fit_hits = {
            value: sum(exact_by_lambda[value][key] for key in fit_ids)
            for value in LAMBDA_GRID
        }
        best = max(fit_hits.values())
        # Conservative tie-break: prefer Fast / the smallest context weight.
        selected = min(value for value, hits in fit_hits.items() if hits == best)
        selected_predictions = predictions_by_lambda[selected]
        for row in rows:
            formula_id = str(row["formula_id"])
            if formula_id in held_ids:
                record_id = str(row["record_id"])
                nested_predictions[record_id] = selected_predictions[record_id]
        for formula_id in held_ids:
            nested_exact[formula_id] = exact_by_lambda[selected][formula_id]
            lambda_by_formula[formula_id] = selected
        folds.append({
            "held_writer_hash": hashlib.sha256(held_writer.encode("utf-8")).hexdigest()[:12],
            "fit_formula_count": len(fit_ids),
            "held_formula_count": len(held_ids),
            "selected_lambda": selected,
            "fit_exact_count": best,
            "fit_exact_by_lambda": {f"{value:.1f}": hits for value, hits in fit_hits.items()},
            "held_exact_count": sum(nested_exact[key] for key in held_ids),
        })
    if len(nested_predictions) != len(rows):
        raise AssertionError("writer-LOFO predictions do not cover every row")
    return nested_predictions, nested_exact, folds, lambda_by_formula


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=nested.DEFAULT_CHECKPOINT)
    parser.add_argument("--distillation-report", type=Path, default=nested.DEFAULT_DISTILLATION_REPORT)
    parser.add_argument("--summary", type=Path, default=export.DEFAULT_SUMMARY)
    parser.add_argument("--data", type=Path, default=export.DEFAULT_DATA)
    parser.add_argument("--onnx", type=Path, default=export.DEFAULT_OUTPUT / "mini_formula_lm_fp32.onnx")
    parser.add_argument("--onnx-parity-report", type=Path, default=export.DEFAULT_OUTPUT / "mini_formula_lm_onnx_parity_report.json")
    parser.add_argument("--dynamic-shape-report", type=Path, default=export.DEFAULT_OUTPUT / "dynamic_shape_contract_report.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite report: {args.output}")
    checkpoint, distill_path = args.checkpoint.resolve(), args.distillation_report.resolve()
    summary_path, data_path = args.summary.resolve(), args.data.resolve()
    for name, path in (("checkpoint", checkpoint), ("distillation report", distill_path),
                       ("frozen summary", summary_path), ("formula data", data_path)):
        if not path.is_file():
            parser.error(f"missing {name}: {path}")

    distill = json.loads(distill_path.read_text(encoding="utf-8"))
    protocol = distill["protocol"]
    if protocol.get("crohme_rows_loaded") != 0 or protocol.get("crohme_training_or_tuning") is not False:
        raise ValueError("mini-LM distillation report does not attest CROHME exclusion")
    if _sha256(checkpoint) != distill["student"]["checkpoint_sha256"]:
        raise ValueError("checkpoint hash differs from frozen distillation report")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("crohme_training_or_tuning") is not False or summary.get("product_default_enabled") is not False:
        raise ValueError("frozen summary violates CROHME/product-off contract")
    raw = export._jsonl(data_path)
    if any("crohme" in str(row.get("source_partition", "")).casefold() for row in raw):
        raise ValueError("CROHME-marked row found in consumed development data")
    raw_by_id = {str(row["sample_id"]): row for row in raw}
    rows, targets = export._formula_rows(summary, raw_by_id)
    if len(targets) != 149:
        raise ValueError(f"expected consumed 149-formula diagnostic, found {len(targets)}")

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    labels = [str(value) for value in payload["labels"]]
    layers = int(payload["layers"])
    reported_layers = int(distill["student"]["architecture"]["layers"])
    if layers != reported_layers:
        raise ValueError("checkpoint/report layer count mismatch")
    mini_lm_trainer.LAYERS = layers
    model = mini_lm_trainer.MiniFormulaLM(len(labels), len(payload["relations"]), int(payload["max_positions"]))
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    onnx_path = args.onnx.resolve()
    parity_path = args.onnx_parity_report.resolve()
    dynamic_report_path = args.dynamic_shape_report.resolve()
    for name, path in (("FP32 ONNX model", onnx_path), ("ONNX parity report", parity_path),
                       ("dynamic-shape report", dynamic_report_path)):
        if not path.is_file():
            parser.error(f"missing {name}: {path}")
    onnx_parity = json.loads(parity_path.read_text(encoding="utf-8"))
    dynamic_report = json.loads(dynamic_report_path.read_text(encoding="utf-8"))
    if onnx_parity.get("protocol", {}).get("crohme_rows_loaded") != 0:
        raise ValueError("ONNX parity report does not attest CROHME exclusion")
    if onnx_parity.get("provenance", {}).get("checkpoint_sha256") != _sha256(checkpoint):
        raise ValueError("ONNX parity checkpoint hash differs from audited checkpoint")
    if dynamic_report.get("provenance", {}).get("fp32_onnx_sha256") != _sha256(onnx_path):
        raise ValueError("FP32 ONNX hash differs from dynamic-shape report")
    onnx_session = export.ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    exact_group_ids = {key for key, target in targets.items() if target["group_exact"]}
    context_rows = [row for row in rows if str(row["formula_id"]) in exact_group_ids]
    context_targets = {key: targets[key] for key in exact_group_ids}
    truth_by_record = {}
    for formula_id, target in targets.items():
        sequence = sorted(
            (row for row in rows if str(row["formula_id"]) == formula_id),
            key=lambda row: int(row["context"]["index"]),
        )
        truth_tokens = [str(token) for token in target["tokens"]]
        if target["group_exact"] and len(sequence) == len(truth_tokens):
            truth_by_record.update({
                str(row["record_id"]): truth
                for row, truth in zip(sequence, truth_tokens, strict=True)
            })
    inputs = context_audit._inputs(context_rows, context_targets, model, labels, oracle=False)
    logits = export._predict_torch(model, inputs).astype(np.float32, copy=False)
    initial_onnx_logits = export._predict_ort(onnx_session, inputs).astype(np.float32, copy=False)
    initial_onnx_max_abs_error = float(np.max(np.abs(logits - initial_onnx_logits)))
    if initial_onnx_max_abs_error > float(onnx_parity["fp32"]["max_abs_error_gate"]):
        raise AssertionError(f"iterative initial-context ONNX parity exceeded gate: {initial_onnx_max_abs_error}")
    initial_class_argmax_matches = int(np.sum(np.argmax(logits, axis=1) == np.argmax(initial_onnx_logits, axis=1)))
    context_logp = logits - np.logaddexp.reduce(logits, axis=1, keepdims=True)
    context = {str(row["record_id"]): value for row, value in zip(context_rows, context_logp, strict=True)}

    label_to_index = {label: index for index, label in enumerate(labels)}
    predictions_by_policy: dict[str, dict[float, dict[str, str]]] = {"strict_lock": {}, "top5_unlock": {}}
    exact_by_policy: dict[str, dict[float, dict[str, bool]]] = {"strict_lock": {}, "top5_unlock": {}}
    fused_predictions_by_lambda: dict[float, dict[str, str]] = {}
    override_margin_by_lambda: dict[float, dict[str, float]] = {}
    winner_gap_by_lambda: dict[float, dict[str, float]] = {}
    for fusion_lambda in LAMBDA_GRID:
        raw_predictions = {str(row["record_id"]): str(row["final_topk"][0]) for row in rows}
        raw_predictions.update(masked._fused_predictions(context_rows, context, labels, fusion_lambda))
        fused_predictions_by_lambda[fusion_lambda] = raw_predictions.copy()
        margins = {}
        winner_gaps = {}
        for row in rows:
            record_id = str(row["record_id"])
            if record_id not in context:
                margins[record_id] = 0.0
                winner_gaps[record_id] = 0.0
                continue
            candidate_scores = [
                math.log(max(float(probability), 1e-12))
                + fusion_lambda * float(context[record_id][label_to_index[str(token)]])
                for token, probability in zip(row["final_topk"], row["final_topk_probabilities"], strict=True)
            ]
            margins[record_id] = max(candidate_scores) - candidate_scores[0]
            sorted_scores = sorted(candidate_scores, reverse=True)
            winner_gaps[record_id] = sorted_scores[0] - sorted_scores[1]
        override_margin_by_lambda[fusion_lambda] = margins
        winner_gap_by_lambda[fusion_lambda] = winner_gaps
        for policy in predictions_by_policy:
            predictions = (
                prompt._strict_lock(rows, raw_predictions)
                if policy == "strict_lock" else raw_predictions.copy()
            )
            if any(predictions[str(row["record_id"])] not in row["final_topk"] for row in rows):
                raise AssertionError(f"{policy} emitted a symbol outside the frozen HWR Top-5")
            predictions_by_policy[policy][fusion_lambda] = predictions
            exact_by_policy[policy][fusion_lambda] = _formula_exact(rows, targets, predictions)

    fast_predictions = {str(row["record_id"]): str(row["final_topk"][0]) for row in rows}
    fast_exact = _formula_exact(rows, targets, fast_predictions)
    policy_results = {}
    for index, policy in enumerate(predictions_by_policy):
        predictions, exact, folds, lambda_by_formula = _nested_policy(
            rows, targets, predictions_by_policy[policy], exact_by_policy[policy], policy
        )
        metrics = export._formula_metrics(rows, targets, predictions)
        policy_results[policy] = {
            "formula_exact": sum(exact.values()),
            "delta_over_fast": sum(exact[key] for key in exact) - sum(fast_exact.values()),
            "writer_bootstrap_delta_pp_95_ci": _bootstrap_delta_by_writer(
                targets, fast_exact, exact, salt=index
            ),
            "candidate_preservation_rate": metrics["candidate_preservation_rate"],
            "changed_tokens": metrics["changed_tokens"],
            "improved_tokens": metrics["improved_tokens"],
            "regressed_tokens": metrics["regressed_tokens"],
            "nested_writer_folds": folds,
            "formula_exact_by_id": exact,
            "predictions_by_record_id": predictions,
            "lambda_by_formula": lambda_by_formula,
        }

    # Isolate the lock effect by applying the strict-lock arm's selected lambda
    # to both policies for each held writer. Separate per-policy calibration
    # remains a secondary upper-bound-style diagnostic below.
    strict_lambda_by_formula = policy_results["strict_lock"]["lambda_by_formula"]
    paired_strict_predictions: dict[str, str] = {}
    paired_unlocked_predictions: dict[str, str] = {}
    for row in rows:
        record_id = str(row["record_id"])
        formula_id = str(row["formula_id"])
        fusion_lambda = strict_lambda_by_formula[formula_id]
        paired_strict_predictions[record_id] = predictions_by_policy["strict_lock"][fusion_lambda][record_id]
        paired_unlocked_predictions[record_id] = predictions_by_policy["top5_unlock"][fusion_lambda][record_id]
    paired_strict_exact = _formula_exact(rows, targets, paired_strict_predictions)
    paired_unlocked_exact = _formula_exact(rows, targets, paired_unlocked_predictions)
    onnx_first_pass_predictions = _fuse_with_formula_lambdas(
        initial_onnx_logits, context_rows, rows, labels, strict_lambda_by_formula, fast_predictions
    )
    initial_prediction_disagreements = sum(
        onnx_first_pass_predictions[key] != paired_unlocked_predictions[key]
        for key in paired_unlocked_predictions
    )
    onnx_iterative_rounds = []

    # Calibrate only the minimum fused-score advantage required to override a
    # strict Top-1 homograph. Each held writer inherits lambda and threshold
    # selected from the other writers; the strict-lock option is in the grid.
    gate_predictions_by_threshold: dict[float | None, dict[str, str]] = {}
    gate_exact_by_threshold: dict[float | None, dict[str, bool]] = {}
    for threshold in MARGIN_THRESHOLDS:
        predictions = {}
        for row in rows:
            record_id = str(row["record_id"])
            formula_id = str(row["formula_id"])
            fusion_lambda = strict_lambda_by_formula[formula_id]
            fast = str(row["final_topk"][0])
            fused = fused_predictions_by_lambda[fusion_lambda][record_id]
            if fast in prompt.STRICT_TOKENS and (
                threshold is None or override_margin_by_lambda[fusion_lambda][record_id] < threshold
            ):
                predictions[record_id] = fast
            else:
                predictions[record_id] = fused
        if any(predictions[str(row["record_id"])] not in row["final_topk"] for row in rows):
            raise AssertionError("margin gate emitted a symbol outside the frozen HWR Top-5")
        gate_predictions_by_threshold[threshold] = predictions
        gate_exact_by_threshold[threshold] = _formula_exact(rows, targets, predictions)

    writer_by_formula = {key: str(value["writer_id"]) for key, value in targets.items()}
    margin_gate_predictions: dict[str, str] = {}
    margin_gate_exact: dict[str, bool] = {}
    margin_gate_folds = []
    selected_threshold_by_formula: dict[str, float | None] = {}
    for held_writer in sorted(set(writer_by_formula.values())):
        fit_ids = [key for key in targets if writer_by_formula[key] != held_writer]
        held_ids = [key for key in targets if writer_by_formula[key] == held_writer]
        fit_counts = {
            threshold: sum(gate_exact_by_threshold[threshold][key] for key in fit_ids)
            for threshold in MARGIN_THRESHOLDS
        }
        best_count = max(fit_counts.values())
        tied = [threshold for threshold in MARGIN_THRESHOLDS if fit_counts[threshold] == best_count]
        # Most conservative threshold wins a tie; None means retain strict lock.
        selected_threshold = max(tied, key=lambda value: math.inf if value is None else value)
        for row in rows:
            formula_id = str(row["formula_id"])
            if formula_id in held_ids:
                record_id = str(row["record_id"])
                margin_gate_predictions[record_id] = gate_predictions_by_threshold[selected_threshold][record_id]
        for formula_id in held_ids:
            margin_gate_exact[formula_id] = gate_exact_by_threshold[selected_threshold][formula_id]
            selected_threshold_by_formula[formula_id] = selected_threshold
        margin_gate_folds.append({
            "held_writer_hash": hashlib.sha256(held_writer.encode("utf-8")).hexdigest()[:12],
            "held_formula_count": len(held_ids),
            "shared_lambda": next(iter({strict_lambda_by_formula[key] for key in held_ids})),
            "selected_margin_threshold": selected_threshold,
            "fit_exact_count": best_count,
            "fit_exact_by_threshold": {
                _threshold_key(threshold): fit_counts[threshold] for threshold in MARGIN_THRESHOLDS
            },
            "held_exact_count": sum(margin_gate_exact[key] for key in held_ids),
        })
    if len(margin_gate_predictions) != len(rows):
        raise AssertionError("margin-gated predictions do not cover every formula row")

    margin_gate_metrics = export._formula_metrics(rows, targets, margin_gate_predictions)
    margin_gate_recovered = {
        key: bool(margin_gate_exact[key] and not paired_strict_exact[key]) for key in targets
    }
    margin_gate_regressed = {
        key: bool(paired_strict_exact[key] and not margin_gate_exact[key]) for key in targets
    }
    margin_gate_ci = _bootstrap_delta_by_writer(
        targets, paired_strict_exact, margin_gate_exact, salt=31
    )

    # A second confidence signal asks whether the selected Top-5 challenger is
    # separated from its nearest competing candidate, not merely from Fast.
    gap_predictions_by_threshold: dict[float | None, dict[str, str]] = {}
    gap_exact_by_threshold: dict[float | None, dict[str, bool]] = {}
    for threshold in MARGIN_THRESHOLDS:
        predictions = {}
        for row in rows:
            record_id = str(row["record_id"])
            formula_id = str(row["formula_id"])
            fusion_lambda = strict_lambda_by_formula[formula_id]
            fast = str(row["final_topk"][0])
            fused = fused_predictions_by_lambda[fusion_lambda][record_id]
            if fast in prompt.STRICT_TOKENS and (
                threshold is None or winner_gap_by_lambda[fusion_lambda][record_id] < threshold
            ):
                predictions[record_id] = fast
            else:
                predictions[record_id] = fused
        if any(predictions[str(row["record_id"])] not in row["final_topk"] for row in rows):
            raise AssertionError("winner-gap gate emitted a symbol outside the frozen HWR Top-5")
        gap_predictions_by_threshold[threshold] = predictions
        gap_exact_by_threshold[threshold] = _formula_exact(rows, targets, predictions)

    gap_gate_predictions: dict[str, str] = {}
    gap_gate_exact: dict[str, bool] = {}
    gap_gate_folds = []
    selected_gap_by_formula: dict[str, float | None] = {}
    for held_writer in sorted(set(writer_by_formula.values())):
        fit_ids = [key for key in targets if writer_by_formula[key] != held_writer]
        held_ids = [key for key in targets if writer_by_formula[key] == held_writer]
        fit_counts = {
            threshold: sum(gap_exact_by_threshold[threshold][key] for key in fit_ids)
            for threshold in MARGIN_THRESHOLDS
        }
        best_count = max(fit_counts.values())
        tied = [threshold for threshold in MARGIN_THRESHOLDS if fit_counts[threshold] == best_count]
        selected_threshold = max(tied, key=lambda value: math.inf if value is None else value)
        for row in rows:
            formula_id = str(row["formula_id"])
            if formula_id in held_ids:
                record_id = str(row["record_id"])
                gap_gate_predictions[record_id] = gap_predictions_by_threshold[selected_threshold][record_id]
        for formula_id in held_ids:
            gap_gate_exact[formula_id] = gap_exact_by_threshold[selected_threshold][formula_id]
            selected_gap_by_formula[formula_id] = selected_threshold
        gap_gate_folds.append({
            "held_writer_hash": hashlib.sha256(held_writer.encode("utf-8")).hexdigest()[:12],
            "held_formula_count": len(held_ids),
            "shared_lambda": next(iter({strict_lambda_by_formula[key] for key in held_ids})),
            "selected_winner_runnerup_gap": selected_threshold,
            "fit_exact_count": best_count,
            "fit_exact_by_gap": {
                _threshold_key(threshold): fit_counts[threshold] for threshold in MARGIN_THRESHOLDS
            },
            "held_exact_count": sum(gap_gate_exact[key] for key in held_ids),
        })
    if len(gap_gate_predictions) != len(rows):
        raise AssertionError("winner-gap-gated predictions do not cover every formula row")

    gap_gate_metrics = export._formula_metrics(rows, targets, gap_gate_predictions)
    gap_gate_recovered = {
        key: bool(gap_gate_exact[key] and not paired_strict_exact[key]) for key in targets
    }
    gap_gate_regressed = {
        key: bool(paired_strict_exact[key] and not gap_gate_exact[key]) for key in targets
    }
    gap_gate_ci = _bootstrap_delta_by_writer(
        targets, paired_strict_exact, gap_gate_exact, salt=47
    )

    # Fixed two-round self-context refinement: round 1 is the paired unlocked
    # pass above; subsequent rounds consume only the prior round's predictions.
    # Truth labels are used solely for scoring, never placed into context.
    iterative_rounds = []
    iterative_state = paired_unlocked_predictions.copy()
    iterative_state_history = [(1, iterative_state.copy())]
    for round_number in (2, 3):
        previous_state = iterative_state
        previous_exact = _formula_exact(rows, targets, previous_state)
        refined_context_rows = []
        for row in context_rows:
            item = dict(row)
            current = previous_state[str(row["record_id"])]
            item["final_topk"] = [current] + [
                str(token) for token in row["final_topk"] if str(token) != current
            ]
            refined_context_rows.append(item)
        refined_inputs = context_audit._inputs(
            refined_context_rows, context_targets, model, labels, oracle=False
        )
        refined_logits = export._predict_torch(model, refined_inputs).astype(np.float32, copy=False)
        refined_onnx_logits = export._predict_ort(onnx_session, refined_inputs).astype(np.float32, copy=False)
        round_max_abs_error = float(np.max(np.abs(refined_logits - refined_onnx_logits)))
        if round_max_abs_error > float(onnx_parity["fp32"]["max_abs_error_gate"]):
            raise AssertionError(
                f"iterative round {round_number} ONNX parity exceeded gate: {round_max_abs_error}"
            )
        class_argmax_matches = int(np.sum(np.argmax(refined_logits, axis=1) == np.argmax(refined_onnx_logits, axis=1)))
        refined_logp = refined_logits - np.logaddexp.reduce(refined_logits, axis=1, keepdims=True)
        refined_context = {
            str(row["record_id"]): values
            for row, values in zip(refined_context_rows, refined_logp, strict=True)
        }
        next_state = fast_predictions.copy()
        for fusion_lambda in sorted(set(strict_lambda_by_formula.values())):
            lambda_rows = [
                row for row in context_rows
                if strict_lambda_by_formula[str(row["formula_id"])] == fusion_lambda
            ]
            next_state.update(masked._fused_predictions(
                lambda_rows,
                {str(row["record_id"]): refined_context[str(row["record_id"])] for row in lambda_rows},
                labels,
                fusion_lambda,
            ))
        if any(next_state[str(row["record_id"])] not in row["final_topk"] for row in rows):
            raise AssertionError(f"iterative round {round_number} emitted a symbol outside HWR Top-5")
        onnx_next_state = _fuse_with_formula_lambdas(
            refined_onnx_logits, context_rows, rows, labels,
            strict_lambda_by_formula, fast_predictions, output_rows=refined_context_rows,
        )
        fused_prediction_disagreements = sum(
            onnx_next_state[key] != next_state[key] for key in next_state
        )
        torch_next_exact = _formula_exact(rows, targets, next_state)
        onnx_next_exact = _formula_exact(rows, targets, onnx_next_state)
        refined_row_index = {
            str(row["record_id"]): index for index, row in enumerate(refined_context_rows)
        }
        mismatch_details = []
        torch_logp = refined_logits - np.logaddexp.reduce(refined_logits, axis=1, keepdims=True)
        onnx_logp = refined_onnx_logits - np.logaddexp.reduce(refined_onnx_logits, axis=1, keepdims=True)
        for row in rows:
            record_id = str(row["record_id"])
            if onnx_next_state[record_id] == next_state[record_id]:
                continue
            index = refined_row_index[record_id]
            candidates = [str(token) for token in row["final_topk"]]
            log_prior = [math.log(max(float(value), 1e-12)) for value in row["final_topk_probabilities"]]
            fusion_lambda = strict_lambda_by_formula[str(row["formula_id"])]
            candidate_scores = {}
            for backend, values in (("torch", torch_logp[index]), ("onnx", onnx_logp[index])):
                scores = [
                    prior + fusion_lambda * float(values[label_to_index[token]])
                    for token, prior in zip(candidates, log_prior, strict=True)
                ]
                order = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
                candidate_scores[backend] = {
                    "winner": candidates[order[0]],
                    "winner_runnerup_margin": scores[order[0]] - scores[order[1]] if len(order) > 1 else None,
                }
            mismatch_details.append({
                "record_id": record_id,
                "formula_id": str(row["formula_id"]),
                "candidates": candidates,
                "truth": truth_by_record.get(record_id),
                "fusion_lambda": fusion_lambda,
                "torch_actual_prediction": next_state[record_id],
                "onnx_actual_prediction": onnx_next_state[record_id],
                "torch": candidate_scores["torch"],
                "onnx": candidate_scores["onnx"],
            })
        exact_formula_disagreements = [
            {
                "formula_id": formula_id,
                "torch_exact": bool(torch_next_exact[formula_id]),
                "onnx_exact": bool(onnx_next_exact[formula_id]),
            }
            for formula_id in sorted(targets)
            if torch_next_exact[formula_id] != onnx_next_exact[formula_id]
        ]
        onnx_iterative_rounds.append({
            "round": round_number,
            "input_context_is_same_as_torch": True,
            "max_abs_logit_error": round_max_abs_error,
            "fp32_error_gate": float(onnx_parity["fp32"]["max_abs_error_gate"]),
            "full_vocabulary_argmax_matches": class_argmax_matches,
            "full_vocabulary_argmax_count": len(refined_logits),
            "fused_top5_prediction_disagreements": fused_prediction_disagreements,
            "torch_formula_exact": sum(torch_next_exact.values()),
            "onnx_formula_exact": sum(onnx_next_exact.values()),
            "exact_formula_disagreements": exact_formula_disagreements,
            "fused_prediction_mismatch_details": mismatch_details,
            "onnx_candidate_preservation_rate": export._formula_metrics(rows, targets, onnx_next_state)[
                "candidate_preservation_rate"
            ],
        })
        next_exact = _formula_exact(rows, targets, next_state)
        next_metrics = export._formula_metrics(rows, targets, next_state)
        recovered_vs_previous = sorted(
            key for key in targets if next_exact[key] and not previous_exact[key]
        )
        regressed_vs_previous = sorted(
            key for key in targets if previous_exact[key] and not next_exact[key]
        )
        transitions_vs_previous: Counter[str] = Counter()
        for formula_id in targets:
            transition = (
                ("exact" if previous_exact[formula_id] else "wrong")
                + "_to_"
                + ("exact" if next_exact[formula_id] else "wrong")
            )
            transitions_vs_previous[transition] += 1
        changed_token_details = []
        token_improved_vs_previous = token_regressed_vs_previous = 0
        for row in rows:
            record_id = str(row["record_id"])
            previous_token = previous_state[record_id]
            refined_token = next_state[record_id]
            if previous_token == refined_token:
                continue
            formula_id = str(row["formula_id"])
            truth = truth_by_record.get(record_id)
            improved = truth is not None and previous_token != truth and refined_token == truth
            regressed = truth is not None and previous_token == truth and refined_token != truth
            hwr_candidates = [str(token) for token in row["final_topk"]]
            hwr_probabilities = [float(value) for value in row["final_topk_probabilities"]]
            hwr_probability_by_token = dict(zip(hwr_candidates, hwr_probabilities, strict=True))

            def _hwr_rank(token: str | None) -> int | None:
                return hwr_candidates.index(token) + 1 if token in hwr_candidates else None

            token_improved_vs_previous += int(improved)
            token_regressed_vs_previous += int(regressed)
            changed_token_details.append({
                "record_id": record_id,
                "formula_id": formula_id,
                "writer_hash": hashlib.sha256(
                    writer_by_formula[formula_id].encode("utf-8")
                ).hexdigest()[:12],
                "target": truth,
                "hwr_top1": hwr_candidates[0],
                "hwr_top1_probability": hwr_probabilities[0],
                "hwr_top1_runner_up_probability_gap": (
                    hwr_probabilities[0] - hwr_probabilities[1]
                    if len(hwr_probabilities) > 1 else None
                ),
                "target_hwr_top5_rank": _hwr_rank(truth),
                "target_hwr_probability": hwr_probability_by_token.get(truth),
                "previous": previous_token,
                "previous_hwr_rank": _hwr_rank(previous_token),
                "previous_hwr_probability": hwr_probability_by_token.get(previous_token),
                "refined": refined_token,
                "refined_hwr_rank": _hwr_rank(refined_token),
                "refined_hwr_probability": hwr_probability_by_token.get(refined_token),
                "improved": bool(improved),
                "regressed": bool(regressed),
            })
        repeated_state_round = next(
            (prior_round for prior_round, prior_state in iterative_state_history if next_state == prior_state),
            None,
        )
        round_writer_folds = []
        for held_writer in sorted(set(writer_by_formula.values())):
            held_ids = [key for key in targets if writer_by_formula[key] == held_writer]
            round_writer_folds.append({
                "held_writer_hash": hashlib.sha256(held_writer.encode("utf-8")).hexdigest()[:12],
                "held_formula_count": len(held_ids),
                "strict_lock_exact": sum(paired_strict_exact[key] for key in held_ids),
                "round_exact": sum(next_exact[key] for key in held_ids),
                "recovered_vs_strict_lock": sum(next_exact[key] and not paired_strict_exact[key] for key in held_ids),
                "regressed_vs_strict_lock": sum(paired_strict_exact[key] and not next_exact[key] for key in held_ids),
            })
        iterative_rounds.append({
            "round": round_number,
            "context_source": "previous round predictions only; no truth-label injection",
            "formula_exact": sum(next_exact.values()),
            "formula_exact_before_round": sum(previous_exact.values()),
            "delta_formula_exact_vs_previous_round": (
                sum(next_exact.values()) - sum(previous_exact.values())
            ),
            "formula_exact_transitions_vs_previous_round": dict(sorted(transitions_vs_previous.items())),
            "formula_exact_recovered_vs_previous_round": len(recovered_vs_previous),
            "formula_exact_regressed_vs_previous_round": len(regressed_vs_previous),
            "recovered_formula_ids_vs_previous_round": recovered_vs_previous,
            "regressed_formula_ids_vs_previous_round": regressed_vs_previous,
            "writer_bootstrap_delta_pp_95_ci_vs_previous_round": _bootstrap_delta_by_writer(
                targets, previous_exact, next_exact, salt=round_number + 80
            ),
            "delta_over_fast": sum(next_exact.values()) - sum(fast_exact.values()),
            "delta_over_paired_strict_lock": sum(next_exact.values()) - sum(paired_strict_exact.values()),
            "writer_bootstrap_delta_pp_95_ci_vs_strict_lock": _bootstrap_delta_by_writer(
                targets, paired_strict_exact, next_exact, salt=round_number + 60
            ),
            "formula_exact_recovered_vs_strict_lock": sum(
                next_exact[key] and not paired_strict_exact[key] for key in targets
            ),
            "formula_exact_regressed_vs_strict_lock": sum(
                paired_strict_exact[key] and not next_exact[key] for key in targets
            ),
            "changed_tokens_from_previous_round": sum(
                next_state[str(row["record_id"])] != previous_state[str(row["record_id"])]
                for row in rows
            ),
            "changed_token_improvements_vs_previous_round": token_improved_vs_previous,
            "changed_token_regressions_vs_previous_round": token_regressed_vs_previous,
            "changed_token_details_vs_previous_round": changed_token_details,
            "no_net_exact_gain_but_token_churn": (
                sum(next_exact.values()) == sum(previous_exact.values())
                and next_state != previous_state
            ),
            "fixed_point_reached": next_state == previous_state,
            "repeated_prediction_state_from_round": repeated_state_round,
            "returned_to_round1_tokens": sum(
                next_state[str(row["record_id"])] == paired_unlocked_predictions[str(row["record_id"])]
                and previous_state[str(row["record_id"])] != paired_unlocked_predictions[str(row["record_id"])]
                for row in rows
            ),
            "candidate_preservation_rate": next_metrics["candidate_preservation_rate"],
            "writer_folds": round_writer_folds,
            "changed_improved_regressed_tokens": [
                next_metrics["changed_tokens"],
                next_metrics["improved_tokens"],
                next_metrics["regressed_tokens"],
            ],
        })
        iterative_state = next_state
        iterative_state_history.append((round_number, iterative_state.copy()))
    strict_rows = [row for row in rows if str(row["final_topk"][0]) in prompt.STRICT_TOKENS]
    unlock_changed = [
        row for row in strict_rows
        if paired_strict_predictions[str(row["record_id"])] != paired_unlocked_predictions[str(row["record_id"])]
    ]
    by_target_class: dict[str, Counter] = {}
    for row in strict_rows:
        record_id = str(row["record_id"])
        if record_id not in truth_by_record:
            continue
        truth = truth_by_record[record_id]
        stats = by_target_class.setdefault(truth, Counter())
        stats["rows"] += 1
        candidate_list = [str(token) for token in row["final_topk"]]
        if truth in candidate_list:
            rank = candidate_list.index(truth) + 1
            stats["truth_in_top5"] += 1
            stats[f"truth_rank_{rank}"] += 1
        else:
            stats["truth_missing_from_top5"] += 1
        fast_ok = str(row["final_topk"][0]) == truth
        locked_ok = paired_strict_predictions[record_id] == truth
        unlocked_ok = paired_unlocked_predictions[record_id] == truth
        margin_gate_ok = margin_gate_predictions[record_id] == truth
        gap_gate_ok = gap_gate_predictions[record_id] == truth
        stats["fast_hits"] += int(fast_ok)
        stats["strict_lock_hits"] += int(locked_ok)
        stats["unlock_hits"] += int(unlocked_ok)
        stats["margin_gate_hits"] += int(margin_gate_ok)
        stats["winner_gap_gate_hits"] += int(gap_gate_ok)
        changed = paired_strict_predictions[record_id] != paired_unlocked_predictions[record_id]
        stats["rows_changed"] += int(changed)
        stats["correct_changes"] += int(changed and unlocked_ok)
        stats["wrong_changes"] += int(changed and not unlocked_ok)
    paired_writer_folds = []
    for held_writer in sorted({str(target["writer_id"]) for target in targets.values()}):
        held_ids = [key for key, target in targets.items() if str(target["writer_id"]) == held_writer]
        lambdas = {strict_lambda_by_formula[key] for key in held_ids}
        if len(lambdas) != 1:
            raise AssertionError("held writer received multiple paired lambdas")
        paired_writer_folds.append({
            "held_writer_hash": hashlib.sha256(held_writer.encode("utf-8")).hexdigest()[:12],
            "held_formula_count": len(held_ids),
            "shared_lambda": next(iter(lambdas)),
            "strict_lock_exact": sum(paired_strict_exact[key] for key in held_ids),
            "top5_unlock_exact": sum(paired_unlocked_exact[key] for key in held_ids),
            "unlock_recovered": sum(paired_unlocked_exact[key] and not paired_strict_exact[key] for key in held_ids),
            "unlock_regressed": sum(paired_strict_exact[key] and not paired_unlocked_exact[key] for key in held_ids),
        })
    unlock_vs_lock_formula = {
        key: bool(paired_unlocked_exact[key] and not paired_strict_exact[key]) for key in targets
    }
    lock_vs_unlock_formula = {
        key: bool(paired_strict_exact[key] and not paired_unlocked_exact[key]) for key in targets
    }
    paired_strict_metrics = export._formula_metrics(rows, targets, paired_strict_predictions)
    paired_unlocked_metrics = export._formula_metrics(rows, targets, paired_unlocked_predictions)
    report = {
        "schema": "aiflow-mini-lm-homograph-unlock-confidence-gates-writer-lofo/v5",
        "status": "consumed_development_confidence_gate_shadow_diagnostic",
        "protocol": {
            "training_performed": False,
            "threshold_selection": "direct lock ablation uses the strict-lock arm's leave-one-writer-out lambda for both policies; equal-fit ties choose the smallest lambda",
            "secondary_separate_calibration": "each policy's own LOFO lambda result is reported separately and is not used to estimate the pure lock effect",
            "margin_gate_selection": "each held writer's minimum fused-score override margin is selected on other writers using the strict-lock LOFO lambda; tie-break favors retaining the strict lock",
            "winner_gap_gate_selection": "each held writer's minimum fused winner-versus-runner-up gap is selected on other writers using the same strict-lock LOFO lambda; tie-break favors retaining the strict lock",
            "iterative_refinement": "two fixed self-context rounds; each round reads only previous predictions, with HWR Top-5 membership preserved; no iteration count tuned on this cohort",
            "margin_threshold_grid": [_threshold_key(value) for value in MARGIN_THRESHOLDS],
            "candidate_policy": "both direct-ablation arms use identical per-writer fusion lambda and remain inside the frozen HWR Top-5; only strict HWR Top-1 homograph lock differs",
            "crohme_rows_loaded": 0,
            "current_149_formula_set_consumed_development": True,
            "policy_comparison_cohort_used_for_design_selection": True,
            "product_adopted": False,
            "warning": "Both confidence gates are diagnostic selections on a consumed development cohort; neither this analysis nor the prior lock ablation is independent acceptance.",
            "supersedes_report": "homograph_unlock_writer_lofo_20261003.json; its direct lock comparison used different per-policy lambdas",
        },
        "provenance": {
            "checkpoint_sha256": _sha256(checkpoint),
            "distillation_report_sha256": _sha256(distill_path),
            "summary_sha256": _sha256(summary_path),
            "formula_data_sha256": _sha256(data_path),
            "formula_count": len(targets),
            "writer_count": len({str(target["writer_id"]) for target in targets.values()}),
            "audit_script_sha256": _sha256(Path(__file__).resolve()),
        },
        "fast_formula_exact": sum(fast_exact.values()),
        "strict_top1_lock_scope": {
            "rows_with_fast_top1_in_strict_homograph_set": len(strict_rows),
            "rows_changed_by_unlock_at_same_lambda": len(unlock_changed),
            "changed_rows_with_exact_group_truth": sum(str(row["record_id"]) in truth_by_record for row in unlock_changed),
            "unlocked_changes_correct": sum(
                truth_by_record.get(str(row["record_id"])) == paired_unlocked_predictions[str(row["record_id"])]
                for row in unlock_changed
                if str(row["record_id"]) in truth_by_record
            ),
            "by_target_class": {
                label: dict(stats) for label, stats in sorted(by_target_class.items())
            },
            "unlocked_changes_wrong": sum(
                truth_by_record.get(str(row["record_id"])) != paired_unlocked_predictions[str(row["record_id"])]
                for row in unlock_changed
                if str(row["record_id"]) in truth_by_record
            ),
        },
        "separately_calibrated_policies": {
            policy: {key: value for key, value in metrics.items()
                     if key not in ("formula_exact_by_id", "predictions_by_record_id")}
            for policy, metrics in policy_results.items()
        },
        "paired_same_lambda": {
            "lambda_source": "strict-lock policy LOFO fit excluding each held writer",
            "strict_lock_formula_exact": sum(paired_strict_exact.values()),
            "top5_unlock_formula_exact": sum(paired_unlocked_exact.values()),
            "strict_lock_changed_improved_regressed_tokens": [
                paired_strict_metrics["changed_tokens"],
                paired_strict_metrics["improved_tokens"],
                paired_strict_metrics["regressed_tokens"],
            ],
            "top5_unlock_changed_improved_regressed_tokens": [
                paired_unlocked_metrics["changed_tokens"],
                paired_unlocked_metrics["improved_tokens"],
                paired_unlocked_metrics["regressed_tokens"],
            ],
            "candidate_preservation_rates": {
                "strict_lock": paired_strict_metrics["candidate_preservation_rate"],
                "top5_unlock": paired_unlocked_metrics["candidate_preservation_rate"],
            },
            "nested_writer_folds": paired_writer_folds,
        },
        "margin_gated_unlock": {
            "formula_exact": sum(margin_gate_exact.values()),
            "delta_over_fast": sum(margin_gate_exact.values()) - sum(fast_exact.values()),
            "delta_over_paired_strict_lock": sum(margin_gate_exact.values()) - sum(paired_strict_exact.values()),
            "writer_bootstrap_delta_pp_95_ci_vs_strict_lock": margin_gate_ci,
            "formula_exact_recovered_vs_strict_lock": sum(margin_gate_recovered.values()),
            "formula_exact_regressed_vs_strict_lock": sum(margin_gate_regressed.values()),
            "changed_improved_regressed_tokens": [
                margin_gate_metrics["changed_tokens"],
                margin_gate_metrics["improved_tokens"],
                margin_gate_metrics["regressed_tokens"],
            ],
            "candidate_preservation_rate": margin_gate_metrics["candidate_preservation_rate"],
            "selected_margin_threshold_by_formula": {
                key: selected_threshold_by_formula[key] for key in sorted(selected_threshold_by_formula)
            },
            "nested_writer_folds": margin_gate_folds,
        },
        "winner_runnerup_gap_gated_unlock": {
            "score": "fused score of best candidate minus fused score of second-best candidate within HWR Top-5",
            "formula_exact": sum(gap_gate_exact.values()),
            "delta_over_fast": sum(gap_gate_exact.values()) - sum(fast_exact.values()),
            "delta_over_paired_strict_lock": sum(gap_gate_exact.values()) - sum(paired_strict_exact.values()),
            "writer_bootstrap_delta_pp_95_ci_vs_strict_lock": gap_gate_ci,
            "formula_exact_recovered_vs_strict_lock": sum(gap_gate_recovered.values()),
            "formula_exact_regressed_vs_strict_lock": sum(gap_gate_regressed.values()),
            "changed_improved_regressed_tokens": [
                gap_gate_metrics["changed_tokens"],
                gap_gate_metrics["improved_tokens"],
                gap_gate_metrics["regressed_tokens"],
            ],
            "candidate_preservation_rate": gap_gate_metrics["candidate_preservation_rate"],
            "selected_gap_by_formula": {
                key: selected_gap_by_formula[key] for key in sorted(selected_gap_by_formula)
            },
            "nested_writer_folds": gap_gate_folds,
        },
        "iterative_context_refinement": iterative_rounds,
        "onnx_iterative_parity": {
            "status": "fp32_cpu_execution_provider_shadow_parity",
            "provider": "CPUExecutionProvider",
            "onnx_path": str(onnx_path),
            "onnx_sha256": _sha256(onnx_path),
            "onnx_parity_report_sha256": _sha256(parity_path),
            "dynamic_shape_report_sha256": _sha256(dynamic_report_path),
            "initial_fast_context_max_abs_logit_error": initial_onnx_max_abs_error,
            "initial_fast_context_fp32_error_gate": float(onnx_parity["fp32"]["max_abs_error_gate"]),
            "initial_fast_context_full_vocabulary_argmax_matches": initial_class_argmax_matches,
            "initial_fast_context_full_vocabulary_argmax_count": len(logits),
            "initial_fused_top5_prediction_disagreements": initial_prediction_disagreements,
            "initial_onnx_formula_exact": sum(_formula_exact(rows, targets, onnx_first_pass_predictions).values()),
            "iterative_rounds": onnx_iterative_rounds,
            "recursive_prediction_path_equivalent": (
                initial_prediction_disagreements == 0
                and all(item["fused_top5_prediction_disagreements"] == 0 for item in onnx_iterative_rounds)
            ),
            "device_latency_verified": False,
        },
        "unlock_vs_lock": {
            "formula_exact_recovered": sum(unlock_vs_lock_formula.values()),
            "formula_exact_regressed": sum(lock_vs_unlock_formula.values()),
            "writer_bootstrap_delta_pp_95_ci": _bootstrap_delta_by_writer(
                targets,
                policy_results["strict_lock"]["formula_exact_by_id"],
                policy_results["top5_unlock"]["formula_exact_by_id"],
                salt=17,
            ),
        },
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({
        "event": "mini_lm_homograph_unlock_lofo_complete",
        "report": str(output),
        "fast_exact": report["fast_formula_exact"],
        "strict_lock_exact": policy_results["strict_lock"]["formula_exact"],
        "top5_unlock_exact": policy_results["top5_unlock"]["formula_exact"],
        "paired_same_lambda_strict_exact": report["paired_same_lambda"]["strict_lock_formula_exact"],
        "paired_same_lambda_unlock_exact": report["paired_same_lambda"]["top5_unlock_formula_exact"],
        "margin_gated_unlock_exact": report["margin_gated_unlock"]["formula_exact"],
        "margin_gated_recovered_regressed": [report["margin_gated_unlock"]["formula_exact_recovered_vs_strict_lock"], report["margin_gated_unlock"]["formula_exact_regressed_vs_strict_lock"]],
        "winner_gap_gated_unlock_exact": report["winner_runnerup_gap_gated_unlock"]["formula_exact"],
        "winner_gap_recovered_regressed": [report["winner_runnerup_gap_gated_unlock"]["formula_exact_recovered_vs_strict_lock"], report["winner_runnerup_gap_gated_unlock"]["formula_exact_regressed_vs_strict_lock"]],
        "iterative_round_formula_exact": [row["formula_exact"] for row in report["iterative_context_refinement"]],
        "unlock_scope": report["strict_top1_lock_scope"],
        "unlock_vs_lock": report["unlock_vs_lock"],
        "crohme_rows": 0,
        "product_adopted": False,
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
