#!/usr/bin/env python3
"""Layerwise, no-training diagnostic for the compact formula-context reranker."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import audit_prompt_bert_context_on_frozen149_v1 as context_audit
import export_mini_formula_lm_onnx_v1 as export
import run_prompt_mini_lm_distillation_v1 as mini_lm_trainer
from selective_decoder_v1 import decode_selective_partition

MiniFormulaLM = mini_lm_trainer.MiniFormulaLM

ROOT = Path(__file__).resolve().parents[1]
DISTILL_DIR = ROOT / "artifacts" / "hwr_failure_cause_microscope_20260928" / "mini_lm_distill_20261002_r2"
MODEL_DIR = ROOT / "artifacts" / "hwr_failure_cause_microscope_20260928" / "mini_lm_onnx_20261002"
DEFAULT_OUTPUT = MODEL_DIR / "layer_activation_microscope_20261003_v3.json"
FROZEN_FUSION_WEIGHT = 0.5
PARITY_TOLERANCE = 1e-5


def _stage_logits(model: MiniFormulaLM, inputs: dict[str, np.ndarray], batch_size: int = 96) -> tuple[dict[str, np.ndarray], float]:
    stage_outputs: dict[str, list[np.ndarray]] = defaultdict(list)
    max_final_parity_error = 0.0
    with torch.inference_mode():
        for start in range(0, len(inputs["input_ids"]), batch_size):
            end = min(start + batch_size, len(inputs["input_ids"]))
            input_ids = torch.from_numpy(inputs["input_ids"][start:end])
            attention = torch.from_numpy(inputs["attention_mask"][start:end])
            mask_positions = torch.from_numpy(inputs["mask_positions"][start:end])
            positions = torch.arange(input_ids.shape[1], dtype=torch.long).unsqueeze(0)
            hidden = model.token_embedding(input_ids) + model.position_embedding(positions)
            key_padding_mask = ~attention.bool()

            stage_logits: dict[str, torch.Tensor] = {}
            encoded = hidden
            for layer_index, layer in enumerate(model.encoder.layers, start=1):
                encoded = layer(encoded, src_key_padding_mask=key_padding_mask)
                masked = encoded[torch.arange(len(encoded)), mask_positions]
                stage_logits[f"layer_{layer_index}"] = model.classifier(model.output_norm(masked))

            if model.encoder.norm is not None:
                encoded = model.encoder.norm(encoded)
                masked = encoded[torch.arange(len(encoded)), mask_positions]
                stage_logits[f"layer_{len(model.encoder.layers)}"] = model.classifier(model.output_norm(masked))

            reference = model(input_ids, attention, mask_positions)
            layer_final = stage_logits[f"layer_{len(model.encoder.layers)}"]
            batch_error = float((reference - layer_final).abs().max().item())
            max_final_parity_error = max(max_final_parity_error, batch_error)
            for stage_name, logits in stage_logits.items():
                stage_outputs[stage_name].append(logits.cpu().numpy().astype(np.float32, copy=False))

    return {name: np.concatenate(parts, axis=0) for name, parts in stage_outputs.items()}, max_final_parity_error


def _stable_argmax(candidates: list[str], scores: list[float]) -> str:
    if not candidates or len(candidates) != len(scores):
        raise ValueError("candidate-score alignment failure")
    return candidates[max(range(len(scores)), key=scores.__getitem__)]


def _score_stage(
    rows: list[dict], targets: dict[str, dict], examples: list[dict], labels: list[str], logits: np.ndarray,
) -> tuple[dict, dict[str, dict]]:
    if len(rows) != len(examples) or logits.shape != (len(rows), len(labels)):
        raise ValueError("stage score rows are not aligned")
    log_probs = logits - np.logaddexp.reduce(logits, axis=1, keepdims=True)
    row_by_id = {str(row["record_id"]): row for row in rows}
    if len(row_by_id) != len(rows):
        raise ValueError("duplicate record ID")

    predictions: dict[str, dict] = {}
    aligned_count = 0
    context_raw_hits = 0
    fused_hits = 0
    candidate_hits = 0
    candidate_misses = 0
    context_candidate_ranks: Counter[int] = Counter()
    for index, example in enumerate(examples):
        record_id = str(example["record_id"])
        formula_id, ordinal_raw = record_id.rsplit(":", 1)
        ordinal = int(ordinal_raw)
        target = targets[formula_id]
        row = row_by_id[record_id]
        truth = str(target["tokens"][ordinal]) if target["group_exact"] and ordinal < len(target["tokens"]) else None
        candidates = [str(value) for value in row["final_topk"]]
        hwr_probs = [float(value) for value in row["final_topk_probabilities"]]
        if len(candidates) != len(hwr_probs):
            raise ValueError(f"HWR Top-5 probability alignment failed: {record_id}")

        ranked_label_indices = np.argsort(-log_probs[index], kind="stable")
        raw_context_top1 = labels[int(ranked_label_indices[0])]
        raw_context_rank = None
        if truth is not None:
            aligned_count += 1
            context_raw_hits += int(raw_context_top1 == truth)
            if truth in candidates:
                candidate_hits += 1
                raw_context_rank = 1 + candidates.index(truth)
                context_candidate_ranks[raw_context_rank] += 1
            else:
                candidate_misses += 1

        fused_scores = [
            math.log(max(probability, 1e-12))
            + FROZEN_FUSION_WEIGHT * float(log_probs[index, labels.index(token)])
            for token, probability in zip(candidates, hwr_probs, strict=True)
        ]
        fused_top1 = _stable_argmax(candidates, fused_scores)
        fused_ranked = sorted(range(len(candidates)), key=lambda candidate_index: (-fused_scores[candidate_index], candidate_index))
        fused_truth_rank = (1 + fused_ranked.index(candidates.index(truth))) if truth in candidates else None
        fused_hits += int(truth is not None and fused_top1 == truth)
        predictions[record_id] = {
            "truth": truth,
            "fast_top1": candidates[0],
            "context_unrestricted_top1": raw_context_top1,
            "context_truth_rank_in_hwr_top5": raw_context_rank,
            "fused_top1": fused_top1,
            "fused_truth_rank_in_hwr_top5": fused_truth_rank,
            "top5": candidates,
            "hwr_top5_probabilities": hwr_probs,
            "context_top5_log_probs": [float(log_probs[index, labels.index(token)]) for token in candidates],
            "fused_top5_scores": fused_scores,
        }

    formula_predictions: dict[str, list[str]] = defaultdict(list)
    fast_formula_predictions: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        record_id = str(row["record_id"])
        formula_id, _ = record_id.rsplit(":", 1)
        formula_predictions[formula_id].append(str(predictions[record_id]["fused_top1"]))
        fast_formula_predictions[formula_id].append(str(predictions[record_id]["fast_top1"]))

    formula_exact = set()
    fast_exact = set()
    eligible_formulas = {formula_id for formula_id, target in targets.items() if target["group_exact"]}
    for formula_id in eligible_formulas:
        truth = [str(value) for value in targets[formula_id]["tokens"]]
        if formula_predictions.get(formula_id) == truth:
            formula_exact.add(formula_id)
        if fast_formula_predictions.get(formula_id) == truth:
            fast_exact.add(formula_id)

    stage_metrics = {
        "aligned_exact_group_tokens": aligned_count,
        "context_unrestricted_top1_hits": context_raw_hits,
        "context_unrestricted_top1_accuracy": context_raw_hits / aligned_count if aligned_count else None,
        "target_in_hwr_top5": candidate_hits,
        "target_outside_hwr_top5": candidate_misses,
        "context_target_rank_within_hwr_top5": {str(rank): count for rank, count in sorted(context_candidate_ranks.items())},
        "fused_hwr_top5_top1_hits": fused_hits,
        "fused_token_accuracy_on_exact_groups": fused_hits / aligned_count if aligned_count else None,
        "symbol_sequence_exact_over_all_149_grouping_fails_as_misses": len(formula_exact),
        "symbol_sequence_exact_over_all_149_rate": len(formula_exact) / len(targets) if targets else None,
        "symbol_sequence_exact_on_group_exact_subset": len(formula_exact),
        "group_exact_formula_count": len(eligible_formulas),
        "symbol_sequence_transitions_vs_fast": {
            "wrong_to_exact": len(formula_exact - fast_exact),
            "exact_to_wrong": len(fast_exact - formula_exact),
            "exact_to_exact": len(formula_exact & fast_exact),
            "wrong_to_wrong": len(eligible_formulas - formula_exact - fast_exact),
        },
    }
    return stage_metrics, predictions


def _layer_transition(left: dict[str, dict], right: dict[str, dict], targets: dict[str, dict]) -> dict:
    token_counts = Counter()
    left_exact: set[str] = set()
    right_exact: set[str] = set()
    left_sequences: dict[str, list[str]] = defaultdict(list)
    right_sequences: dict[str, list[str]] = defaultdict(list)
    for record_id, before in left.items():
        after = right[record_id]
        truth = before["truth"]
        if truth is None:
            continue
        token_counts["aligned_tokens"] += 1
        token_counts["changed_predictions"] += int(before["fused_top1"] != after["fused_top1"])
        token_counts["wrong_to_correct"] += int(before["fused_top1"] != truth and after["fused_top1"] == truth)
        token_counts["correct_to_wrong"] += int(before["fused_top1"] == truth and after["fused_top1"] != truth)
        formula_id, _ = record_id.rsplit(":", 1)
        left_sequences[formula_id].append(str(before["fused_top1"]))
        right_sequences[formula_id].append(str(after["fused_top1"]))
    eligible = {formula_id for formula_id, target in targets.items() if target["group_exact"]}
    for formula_id in eligible:
        truth = [str(value) for value in targets[formula_id]["tokens"]]
        if left_sequences[formula_id] == truth:
            left_exact.add(formula_id)
        if right_sequences[formula_id] == truth:
            right_exact.add(formula_id)
    token_counts["symbol_sequence_wrong_to_exact"] = len(right_exact - left_exact)
    token_counts["symbol_sequence_exact_to_wrong"] = len(left_exact - right_exact)
    token_counts["symbol_sequence_exact_to_exact"] = len(right_exact & left_exact)
    token_counts["symbol_sequence_wrong_to_wrong"] = len(eligible - right_exact - left_exact)
    return dict(token_counts)


def _strict_decode_stage(
    rows: list[dict], targets: dict[str, dict], labels: list[str], logits: np.ndarray,
    summary: dict, raw_by_id: dict[str, dict],
) -> tuple[dict, dict[str, dict]]:
    log_probs = logits - np.logaddexp.reduce(logits, axis=1, keepdims=True)
    label_to_index = {label: index for index, label in enumerate(labels)}
    rows_by_formula: dict[str, list[dict]] = defaultdict(list)
    row_index = {}
    for index, row in enumerate(rows):
        formula_id = str(row["formula_id"])
        rows_by_formula[formula_id].append(row)
        row_index[str(row["record_id"])] = index
    summary_by_id = {str(record["sample_id"]): record for record in summary["records"]}
    base_exact: set[str] = set()
    stage_exact: set[str] = set()
    formula_outputs: dict[str, dict] = {}
    stage_accepted = 0
    baseline_accepted = 0
    candidate_preserved = 0
    emitted_tokens = 0
    changed_tokens = 0
    improved_tokens = 0
    regressed_tokens = 0
    relation_graph_changes = 0

    for formula_id, target in targets.items():
        record = summary_by_id[formula_id]
        fast = record["hwr_tournament"]["fast"]
        groups = [[int(value) for value in group] for group in fast["groups"]]
        formula_rows = rows_by_formula[formula_id]
        selected_symbols = fast["selected_symbols"]
        if len(formula_rows) != len(selected_symbols):
            raise ValueError(f"selected symbol rows do not align for {formula_id}")
        decoder_symbols = []
        context_scores = {}
        fallback = []
        for symbol, row in zip(selected_symbols, formula_rows, strict=True):
            record_id = str(row["record_id"])
            topk = [str(value) for value in row["final_topk"]]
            probabilities = [float(value) for value in row["final_topk_probabilities"]]
            decoder_symbols.append({
                "stroke_indices": [int(value) for value in symbol["stroke_indices"]],
                "hwr_topk": topk,
                "hwr_topk_probabilities": probabilities,
                "geometry": dict(row["geometry"]),
            })
            fallback.append(topk[0])
            context_scores[record_id] = {
                token: float(log_probs[row_index[record_id], label_to_index[token]])
                for token in topk
            }

        stroke_count = len(raw_by_id[formula_id]["strokes"])
        base = decode_selective_partition(formula_id, groups, decoder_symbols, stroke_count=stroke_count)
        challenge = decode_selective_partition(
            formula_id, groups, decoder_symbols, stroke_count=stroke_count,
            context_log_probabilities=context_scores, context_weight=FROZEN_FUSION_WEIGHT,
        )
        base_tokens = [str(value) for value in base["tokens"]] if base.get("accepted") else list(fallback)
        challenge_tokens = [str(value) for value in challenge["tokens"]] if challenge.get("accepted") else list(fallback)
        if len(base_tokens) != len(fallback) or len(challenge_tokens) != len(fallback):
            raise ValueError(f"decoder emitted an unexpected token count for {formula_id}")
        baseline_accepted += int(bool(base.get("accepted")))
        stage_accepted += int(bool(challenge.get("accepted")))
        truth = [str(value) for value in target["tokens"]]
        if target["group_exact"] and base_tokens == truth:
            base_exact.add(formula_id)
        if target["group_exact"] and challenge_tokens == truth:
            stage_exact.add(formula_id)
        if target["group_exact"]:
            for ordinal, (before, after, expected) in enumerate(zip(base_tokens, challenge_tokens, truth, strict=True)):
                changed_tokens += int(before != after)
                improved_tokens += int(before != expected and after == expected)
                regressed_tokens += int(before == expected and after != expected)
        for token, symbol in zip(challenge_tokens, decoder_symbols, strict=True):
            emitted_tokens += 1
            candidate_preserved += int(token in symbol["hwr_topk"])
        base_edges = sorted(
            (str(edge.get("parent")), str(edge.get("child")), str(edge.get("type")))
            for edge in base.get("relations", [])
        )
        stage_edges = sorted(
            (str(edge.get("parent")), str(edge.get("child")), str(edge.get("type")))
            for edge in challenge.get("relations", [])
        )
        relation_graph_changes += int(bool(base.get("accepted")) and bool(challenge.get("accepted")) and base_edges != stage_edges)
        formula_outputs[formula_id] = {
            "group_exact": bool(target["group_exact"]),
            "truth": truth,
            "fast_tokens": list(fallback),
            "baseline_decoder_tokens": base_tokens,
            "stage_tokens": challenge_tokens,
            "baseline_decoder_accepted": bool(base.get("accepted")),
            "stage_decoder_accepted": bool(challenge.get("accepted")),
            "symbol_sequence_exact": formula_id in stage_exact,
        }

    strict_transition = {
        "wrong_to_exact_vs_baseline_decoder": len(stage_exact - base_exact),
        "exact_to_wrong_vs_baseline_decoder": len(base_exact - stage_exact),
        "exact_to_exact_vs_baseline_decoder": len(stage_exact & base_exact),
    }
    fast_exact = {
        formula_id for formula_id, result in formula_outputs.items()
        if result["group_exact"] and result["fast_tokens"] == result["truth"]
    }
    stage_metrics = {
        "symbol_sequence_exact_over_all_149_grouping_fails_as_misses": len(stage_exact),
        "symbol_sequence_exact_rate": len(stage_exact) / len(targets) if targets else None,
        "group_exact_formula_count": sum(bool(target["group_exact"]) for target in targets.values()),
        "baseline_decoder_symbol_sequence_exact": len(base_exact),
        "delta_vs_fast_top1_symbol_sequence_exact": len(stage_exact) - len(fast_exact),
        "delta_vs_baseline_decoder_symbol_sequence_exact": len(stage_exact) - len(base_exact),
        "stage_decoder_accepted_formulas": stage_accepted,
        "baseline_decoder_accepted_formulas": baseline_accepted,
        "candidate_preservation_rate": candidate_preserved / emitted_tokens if emitted_tokens else None,
        "changed_improved_regressed_tokens_vs_baseline_decoder": [changed_tokens, improved_tokens, regressed_tokens],
        "relation_graph_changes_vs_baseline_decoder": relation_graph_changes,
        "symbol_sequence_transitions_vs_baseline_decoder": strict_transition,
        "symbol_sequence_transitions_vs_fast_top1": {
            "wrong_to_exact": len(stage_exact - fast_exact),
            "exact_to_wrong": len(fast_exact - stage_exact),
        },
        "grouping_mutations": 0,
    }
    return stage_metrics, formula_outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DISTILL_DIR / "mini_formula_lm.pt")
    parser.add_argument("--distillation-report", type=Path, default=DISTILL_DIR / "mini_formula_lm_distillation_report.json")
    parser.add_argument("--summary", type=Path, default=context_audit.DEFAULT_SUMMARY)
    parser.add_argument("--data", type=Path, default=context_audit.DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    checkpoint, distill_path = args.checkpoint.resolve(), args.distillation_report.resolve()
    summary_path, data_path, output = args.summary.resolve(), args.data.resolve(), args.output.resolve()
    if output.exists():
        parser.error(f"refusing to overwrite existing report: {output}")
    for name, path in (("checkpoint", checkpoint), ("distillation report", distill_path), ("summary", summary_path), ("formula data", data_path)):
        if not path.is_file():
            parser.error(f"missing {name}: {path}")

    distill = json.loads(distill_path.read_text(encoding="utf-8"))
    protocol = distill.get("protocol", {})
    if protocol.get("crohme_rows_loaded") != 0 or protocol.get("crohme_training_or_tuning") is not False:
        raise ValueError("mini-LM distillation provenance does not attest CROHME exclusion")
    checkpoint_hash = export._sha256(checkpoint)
    if checkpoint_hash != distill["student"]["checkpoint_sha256"]:
        raise ValueError("mini-LM checkpoint hash mismatch")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("crohme_training_or_tuning") is not False:
        raise ValueError("summary does not attest CROHME exclusion")
    raw = context_audit._jsonl(data_path)
    if any("crohme" in str(row.get("source_partition", "")).casefold() for row in raw):
        raise ValueError("CROHME row found in diagnostic data")
    raw_by_id = {str(row["sample_id"]): row for row in raw}
    rows, targets = context_audit._formula_rows(summary, raw_by_id)
    if len(targets) != 149 or len({str(row["record_id"]) for row in rows}) != len(rows):
        raise ValueError("unexpected or non-unique consumed diagnostic cohort")

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    labels = [str(value) for value in payload["labels"]]
    if len(labels) != 372 or len(set(labels)) != len(labels):
        raise ValueError("mini-LM label vocabulary contract mismatch")
    layer_count = int(payload["layers"])
    if int(distill["student"]["architecture"]["layers"]) != layer_count or layer_count < 1:
        raise ValueError("mini-LM checkpoint/report depth contract mismatch")
    mini_lm_trainer.LAYERS = layer_count
    model = MiniFormulaLM(len(labels), len(payload["relations"]), int(payload["max_positions"]))
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    inputs, examples = export._current_inputs(rows, model, labels)
    if len(examples) != len(rows):
        raise AssertionError("mini-LM examples do not cover every selected HWR group")

    stage_logits, parity_error = _stage_logits(model, inputs)
    if parity_error > PARITY_TOLERANCE:
        raise AssertionError(f"manual layer trace diverges from model.forward: {parity_error} > {PARITY_TOLERANCE}")

    stage_metrics: dict[str, dict] = {}
    stage_predictions: dict[str, dict[str, dict]] = {}
    strict_stage_metrics: dict[str, dict] = {}
    strict_stage_outputs: dict[str, dict[str, dict]] = {}
    for stage_name, logits in stage_logits.items():
        metrics, predictions = _score_stage(rows, targets, examples, labels, logits)
        stage_metrics[stage_name] = metrics
        stage_predictions[stage_name] = predictions
        strict_metrics, strict_outputs = _strict_decode_stage(rows, targets, labels, logits, summary, raw_by_id)
        strict_stage_metrics[stage_name] = strict_metrics
        strict_stage_outputs[stage_name] = strict_outputs

    stage_names = list(stage_logits)
    transitions = {
        f"{left}_to_{right}": _layer_transition(stage_predictions[left], stage_predictions[right], targets)
        for left, right in zip(stage_names, stage_names[1:], strict=False)
    }
    strict_transitions = {}
    for left, right in zip(stage_names, stage_names[1:], strict=False):
        left_exact = {formula_id for formula_id, result in strict_stage_outputs[left].items() if result["symbol_sequence_exact"]}
        right_exact = {formula_id for formula_id, result in strict_stage_outputs[right].items() if result["symbol_sequence_exact"]}
        strict_transitions[f"{left}_to_{right}"] = {
            "symbol_sequence_wrong_to_exact": len(right_exact - left_exact),
            "symbol_sequence_exact_to_wrong": len(left_exact - right_exact),
            "symbol_sequence_exact_to_exact": len(left_exact & right_exact),
            "symbol_sequence_wrong_to_wrong": sum(bool(target["group_exact"]) for target in targets.values()) - len(left_exact | right_exact),
        }
    if len(stage_names) >= 2:
        transitions["fast_to_first_transformer_layer"] = _layer_transition(
            {record_id: {**item, "fused_top1": item["fast_top1"]} for record_id, item in stage_predictions[stage_names[0]].items()},
            stage_predictions[stage_names[0]], targets,
        )

    aligned_rows = []
    for row in rows:
        record_id = str(row["record_id"])
        detail_by_stage = {stage: stage_predictions[stage][record_id] for stage in stage_names}
        truth = detail_by_stage[stage_names[0]]["truth"]
        if truth is None:
            continue
        fast_top1 = detail_by_stage[stage_names[0]]["fast_top1"]
        fused = [detail_by_stage[stage]["fused_top1"] for stage in stage_names]
        if fast_top1 == truth and all(value == truth for value in fused):
            continue
        aligned_rows.append({
            "record_id": record_id,
            "truth": truth,
            "fast_top1": fast_top1,
            "layers": {
                stage: {
                    "context_unrestricted_top1": detail_by_stage[stage]["context_unrestricted_top1"],
                    "context_truth_rank_in_hwr_top5": detail_by_stage[stage]["context_truth_rank_in_hwr_top5"],
                    "fused_top1": detail_by_stage[stage]["fused_top1"],
                    "fused_truth_rank_in_hwr_top5": detail_by_stage[stage]["fused_truth_rank_in_hwr_top5"],
                    "context_top5_log_probs": detail_by_stage[stage]["context_top5_log_probs"],
                }
                for stage in stage_names
            },
            "hwr_top5": detail_by_stage[stage_names[0]]["top5"],
            "hwr_top5_probabilities": detail_by_stage[stage_names[0]]["hwr_top5_probabilities"],
        })

    report = {
        "schema": "aiflow-mini-lm-layer-activation-microscope/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "consumed_development_activation_diagnostic",
        "protocol": {
            "training_performed": False,
            "threshold_or_architecture_selection_performed": False,
            "crohme_rows_loaded": 0,
            "consumed_formula_count": len(targets),
            "writer_count": len({str(target["writer_id"]) for target in targets.values()}),
            "fusion_weight": FROZEN_FUSION_WEIGHT,
            "fusion_weight_source": "frozen teacher value; not retuned in this layer diagnostic",
            "candidate_policy": "rerank only original HWR Top-5; no new classes or grouping mutations",
            "warning": "Consumed development data only; layerwise projection is diagnostic, not independent acceptance or product evidence.",
            "crohme_training_or_tuning": False,
            "product_adopted": False,
        },
        "provenance": {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_hash,
            "distillation_report": str(distill_path),
            "distillation_report_sha256": export._sha256(distill_path),
            "summary": str(summary_path),
            "summary_sha256": export._sha256(summary_path),
            "formula_data": str(data_path),
            "formula_data_sha256": export._sha256(data_path),
            "checkpoint_parameters": int(distill["student"]["parameters"]),
            "architecture": dict(distill["student"]["architecture"]),
        },
        "verification": {
            "input_group_count": len(rows),
            "exact_group_formula_count": sum(bool(target["group_exact"]) for target in targets.values()),
            "model_forward_vs_manual_layer_trace_max_abs_error": parity_error,
            "parity_tolerance": PARITY_TOLERANCE,
            "finite_logits": all(bool(np.isfinite(logits).all()) for logits in stage_logits.values()),
            "all_checks_passed": parity_error <= PARITY_TOLERANCE and all(bool(np.isfinite(logits).all()) for logits in stage_logits.values()),
        },
        "metrics_by_stage": stage_metrics,
        "decoder_constrained_symbol_sequence_metrics_by_stage": strict_stage_metrics,
        "adjacent_layer_transitions": transitions,
        "decoder_constrained_symbol_sequence_layer_transitions": strict_transitions,
        "nontrivial_token_cases": aligned_rows[:128],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({
        "event": "mini_lm_layer_activation_microscope_complete",
        "report": str(output),
        "layer_metrics": {name: {key: value for key, value in metrics.items() if key in ("context_unrestricted_top1_hits", "fused_hwr_top5_top1_hits", "symbol_sequence_exact_over_all_149_grouping_fails_as_misses", "symbol_sequence_transitions_vs_fast")} for name, metrics in stage_metrics.items()},
        "decoder_constrained_symbol_sequence_metrics": strict_stage_metrics,
        "adjacent_transitions": transitions,
        "decoder_constrained_symbol_sequence_transitions": strict_transitions,
        "trace_max_abs_error": parity_error,
        "crohme_rows_loaded": 0,
        "product_adopted": False,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
