#!/usr/bin/env python3
"""Compare structural decoding with naive and writer-LOFO gated mini-LM scores.

This is a consumed-development shadow audit only. It performs no training,
selects gate thresholds only in writer-LOFO folds, and never loads CROHME.
"""

from __future__ import annotations

import json
import hashlib
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import export_mini_formula_lm_onnx_v1 as export
import audit_prompt_bert_context_on_frozen149_v1 as context_audit
import run_prompt_mini_lm_distillation_v1 as mini_lm_trainer
from selective_decoder_v1 import decode_selective_partition

MiniFormulaLM = mini_lm_trainer.MiniFormulaLM


ROOT = Path(__file__).resolve().parents[1]
DISTILL_DIR = ROOT / "artifacts" / "hwr_failure_cause_microscope_20260928" / "mini_lm_distill_20261002_r2"
MODEL_DIR = ROOT / "artifacts" / "hwr_failure_cause_microscope_20260928" / "mini_lm_onnx_20261002"
DEFAULT_OUTPUT = MODEL_DIR / "joint_decoder_shadow_20261003.json"
FROZEN_CONTEXT_WEIGHT = 0.5
OVERRIDE_MARGIN_THRESHOLDS: tuple[float | None, ...] = (
    0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.0, 1.5, 2.0, None,
)


def _threshold_key(value: float | None) -> str:
    return "strict_lock" if value is None else f"{value:.2f}"


def _decode_with_override_margin_gate(
    case: dict[str, object], context_scores: dict[str, dict[str, float]], margin_threshold: float | None,
    hwr_top1_margin_cap: float | None = None,
) -> tuple[list[str], dict[str, object], int]:
    """Apply context only when it agrees with HWR top-1 or clears confidence-aware gates."""
    formula_id = str(case["formula_id"])
    symbols = case["symbols"]
    assert isinstance(symbols, list)
    gated_scores: dict[str, dict[str, float]] = {}
    unlocked_groups = 0
    for ordinal, symbol in enumerate(symbols):
        assert isinstance(symbol, dict)
        record_id = f"{formula_id}:{ordinal}"
        tokens = [str(token) for token in symbol["hwr_topk"]]
        probabilities = [float(value) for value in symbol["hwr_topk_probabilities"]]
        scores = context_scores[record_id]
        fused = [
            (math.log(max(1e-8, probability)) + FROZEN_CONTEXT_WEIGHT * scores[token], token)
            for token, probability in zip(tokens, probabilities, strict=True)
        ]
        fused.sort(key=lambda pair: (-pair[0], pair[1]))
        winner = fused[0][1]
        hwr_top1 = tokens[0]
        fused_by_token = {token: score for score, token in fused}
        override_margin = fused[0][0] - fused_by_token[hwr_top1]
        hwr_top1_margin = probabilities[0] - max(probabilities[1:], default=0.0)
        allowed = winner == hwr_top1 or (
            margin_threshold is not None and override_margin >= margin_threshold
            and (hwr_top1_margin_cap is None or hwr_top1_margin <= hwr_top1_margin_cap)
        )
        if allowed:
            gated_scores[record_id] = scores
            unlocked_groups += int(winner != hwr_top1)
        else:
            # Uniform scores remove only the mini-LM contribution; HWR logits remain intact.
            gated_scores[record_id] = {token: 0.0 for token in tokens}

    result = decode_selective_partition(
        formula_id, case["groups"], symbols, stroke_count=int(case["stroke_count"]),
        context_log_probabilities=gated_scores, context_weight=FROZEN_CONTEXT_WEIGHT,
    )
    fallback = [str(symbol["hwr_topk"][0]) for symbol in symbols]
    decoded = [str(token) for token in result["tokens"]] if result.get("accepted") else fallback
    if len(decoded) != len(symbols):
        raise AssertionError("gated decoder token count does not match selected groups")
    if any(token not in symbol["hwr_topk"] for token, symbol in zip(decoded, symbols, strict=True)):
        raise AssertionError("gated decoder emitted a token outside HWR Top-5")
    return decoded, result, unlocked_groups


def _paired_formula_arm(
    baseline_rows: list[dict[str, object]], challenger_rows: list[dict[str, object]],
    targets: dict[str, dict], arm: str,
) -> dict[str, object]:
    baseline = {str(row["formula_id"]): row for row in baseline_rows}
    challenger = {str(row["formula_id"]): row for row in challenger_rows}
    if set(baseline) != set(challenger) or set(baseline) != set(targets):
        raise ValueError("paired decoder reports do not contain the same formula IDs")
    transitions: Counter[str] = Counter()
    deltas: dict[str, int] = {}
    for formula_id in baseline:
        before_row, after_row = baseline[formula_id], challenger[formula_id]
        if before_row.get("writer_hash") != after_row.get("writer_hash"):
            raise ValueError(f"writer identity mismatch in paired decoder report: {formula_id}")
        if bool(before_row.get("group_exact")) != bool(after_row.get("group_exact")):
            raise ValueError(f"grouping label mismatch in paired decoder report: {formula_id}")
        before = bool(before_row["arms"][arm]["formula_exact"])
        after = bool(after_row["arms"][arm]["formula_exact"])
        transitions[("exact" if before else "wrong") + "_to_" + ("exact" if after else "wrong")] += 1
        deltas[formula_id] = int(after) - int(before)
    return {
        "arm": arm,
        "baseline_formula_exact": sum(bool(row["arms"][arm]["formula_exact"]) for row in baseline.values()),
        "challenger_formula_exact": sum(bool(row["arms"][arm]["formula_exact"]) for row in challenger.values()),
        "delta_formula_exact": sum(deltas.values()),
        "recovered": sum(value == 1 for value in deltas.values()),
        "regressed": sum(value == -1 for value in deltas.values()),
        "transitions": dict(sorted(transitions.items())),
        "writer_bootstrap_delta_pp_95_ci": context_audit._writer_bootstrap(deltas, targets)[
            "delta_formula_exact_pp_95_ci"
        ],
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DISTILL_DIR / "mini_formula_lm.pt")
    parser.add_argument("--distillation-report", type=Path, default=DISTILL_DIR / "mini_formula_lm_distillation_report.json")
    parser.add_argument("--onnx", type=Path, default=MODEL_DIR / "mini_formula_lm_fp32.onnx")
    parser.add_argument("--onnx-report", type=Path, default=MODEL_DIR / "mini_formula_lm_onnx_parity_report.json")
    parser.add_argument("--summary", type=Path, default=context_audit.DEFAULT_SUMMARY)
    parser.add_argument("--data", type=Path, default=context_audit.DEFAULT_DATA)
    parser.add_argument("--soft-context-mix", type=float, default=None)
    parser.add_argument(
        "--hwr-top1-margin-cap", type=float, default=None,
        help="shadow ablation: forbid a mini-LM override when HWR Top-1 exceeds its runner-up by more than this probability margin",
    )
    parser.add_argument("--soft-mix-selection-report", type=Path, default=None)
    parser.add_argument("--hard-context-baseline-report", type=Path, default=None)
    parser.add_argument(
        "--threshold-tie-break", choices=("conservative", "lowest_tied_finite"), default="conservative",
        help="when multiple margins tie on fit exact count, choose the strictest or lowest finite threshold",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    checkpoint, distill_path, onnx_path = args.checkpoint.resolve(), args.distillation_report.resolve(), args.onnx.resolve()
    onnx_report_path, summary_path, data_path, output = (
        args.onnx_report.resolve(), args.summary.resolve(), args.data.resolve(), args.output.resolve()
    )
    soft_mix_report_path = args.soft_mix_selection_report.resolve() if args.soft_mix_selection_report else None
    hard_context_baseline_path = args.hard_context_baseline_report.resolve() if args.hard_context_baseline_report else None
    if args.soft_context_mix is not None and not 0.0 <= args.soft_context_mix <= 1.0:
        parser.error("--soft-context-mix must be within [0, 1]")
    if args.hwr_top1_margin_cap is not None and not 0.0 <= args.hwr_top1_margin_cap <= 1.0:
        parser.error("--hwr-top1-margin-cap must be within [0, 1]")
    if args.soft_context_mix is None and soft_mix_report_path is not None:
        parser.error("--soft-mix-selection-report requires --soft-context-mix")
    if args.soft_context_mix is not None and soft_mix_report_path is None:
        parser.error("soft-context decoding requires its writer-crossfit selection report")
    if args.soft_context_mix is not None and hard_context_baseline_path is None:
        parser.error("soft-context decoding requires a paired hard-context baseline report")
    if args.soft_context_mix is None and hard_context_baseline_path is not None:
        parser.error("--hard-context-baseline-report requires --soft-context-mix")
    if output.exists():
        parser.error(f"refusing to overwrite report: {output}")
    for name, path in (
        ("mini-LM checkpoint", checkpoint), ("distillation report", distill_path),
        ("FP32 ONNX", onnx_path), ("ONNX parity report", onnx_report_path),
        ("frozen summary", summary_path), ("formula data", data_path),
    ):
        if not path.is_file():
            parser.error(f"missing {name}: {path}")
    if soft_mix_report_path is not None and not soft_mix_report_path.is_file():
        parser.error(f"missing soft-mix selection report: {soft_mix_report_path}")
    if hard_context_baseline_path is not None and not hard_context_baseline_path.is_file():
        parser.error(f"missing hard-context baseline report: {hard_context_baseline_path}")

    distill = json.loads(distill_path.read_text(encoding="utf-8"))
    if distill.get("protocol", {}).get("crohme_rows_loaded") != 0 or distill["protocol"].get("crohme_training_or_tuning") is not False:
        raise ValueError("student provenance does not attest CROHME exclusion")
    if export._sha256(checkpoint) != distill["student"]["checkpoint_sha256"]:
        raise ValueError("mini-LM checkpoint hash mismatch")
    onnx_report = json.loads(onnx_report_path.read_text(encoding="utf-8"))
    if onnx_report.get("protocol", {}).get("crohme_rows_loaded") != 0:
        raise ValueError("ONNX parity report does not attest CROHME exclusion")
    if onnx_report.get("provenance", {}).get("checkpoint_sha256") != export._sha256(checkpoint):
        raise ValueError("ONNX report/checkpoint provenance mismatch")
    if onnx_report.get("provenance", {}).get("onnx_sha256") not in (None, export._sha256(onnx_path)):
        raise ValueError("ONNX parity report/model hash mismatch")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("crohme_training_or_tuning") is not False or summary.get("product_default_enabled") is not False:
        raise ValueError("frozen summary violates CROHME/product-off contract")
    raw = context_audit._jsonl(data_path)
    if any("crohme" in str(row.get("source_partition", "")).casefold() for row in raw):
        raise ValueError("CROHME row found in requested diagnostic input")
    raw_by_id = {str(row["sample_id"]): row for row in raw}
    rows, targets = context_audit._formula_rows(summary, raw_by_id)
    if len(targets) != 149:
        raise ValueError(f"expected consumed 149-formula diagnostic; found {len(targets)}")

    soft_mix_selection_report = None
    if soft_mix_report_path is not None:
        soft_mix_selection_report = json.loads(soft_mix_report_path.read_text(encoding="utf-8"))
        selection_protocol = soft_mix_selection_report.get("protocol", {})
        selection_provenance = soft_mix_selection_report.get("provenance", {})
        selection_counts = soft_mix_selection_report.get("metrics", {}).get(
            "writer_crossfit_soft_mix", {},
        ).get("selected_mix_counts_by_held_writer", {})
        selection_folds = soft_mix_selection_report.get("metrics", {}).get(
            "writer_crossfit_soft_mix", {},
        ).get("folds", [])
        if (
            soft_mix_selection_report.get("status") != "consumed_development_shadow_diagnostic"
            or selection_protocol.get("training_performed") is not False
            or selection_protocol.get("crohme_rows_loaded") != 0
            or selection_protocol.get("consumed_formula_count") != len(targets)
            or selection_protocol.get("writer_count") != len({str(t["writer_id"]) for t in targets.values()})
            or soft_mix_selection_report.get("decision", {}).get("promotion_eligible") is not False
        ):
            raise ValueError("soft-mix selection report violates the expected shadow-only contract")
        if (
            selection_provenance.get("checkpoint_sha256") != export._sha256(checkpoint)
            or selection_provenance.get("summary_sha256") != export._sha256(summary_path)
            or selection_provenance.get("formula_data_sha256") != export._sha256(data_path)
        ):
            raise ValueError("soft-mix selection report does not bind these model/data inputs")
        writer_hashes = {
            hashlib.sha256(str(t["writer_id"]).encode("utf-8")).hexdigest()[:12]
            for t in targets.values()
        }
        selected_writer_hashes = {str(fold.get("held_writer_hash")) for fold in selection_folds}
        selected_key = f"{args.soft_context_mix:.2f}"
        if (
            selected_writer_hashes != writer_hashes
            or len(selection_folds) != len(writer_hashes)
            or selection_counts != {selected_key: len(writer_hashes)}
            or any(float(fold.get("selected_soft_mix", -1.0)) != args.soft_context_mix for fold in selection_folds)
        ):
            raise ValueError("requested soft-context mixture was not selected in every matching held-writer fold")

    hard_context_baseline = None
    if hard_context_baseline_path is not None:
        hard_context_baseline = json.loads(hard_context_baseline_path.read_text(encoding="utf-8"))
        baseline_protocol = hard_context_baseline.get("protocol", {})
        baseline_provenance = hard_context_baseline.get("provenance", {})
        if (
            hard_context_baseline.get("schema") != "aiflow-mini-lm-selective-decoder-shadow/v4"
            or hard_context_baseline.get("status") != "consumed_development_joint_decoder_shadow_diagnostic"
            or baseline_protocol.get("training_performed") is not False
            or baseline_protocol.get("crohme_rows_loaded") != 0
            or baseline_protocol.get("mini_lm_layers") != int(distill["student"]["architecture"]["layers"])
            or baseline_protocol.get("per_formula_predictions_included") is not True
            or baseline_protocol.get("product_adopted") is not False
        ):
            raise ValueError("paired baseline is not an eligible hard-context shadow report")
        if any(
            baseline_provenance.get(key) != value
            for key, value in (
                ("checkpoint_sha256", export._sha256(checkpoint)),
                ("onnx_sha256", export._sha256(onnx_path)),
                ("summary_sha256", export._sha256(summary_path)),
                ("formula_data_sha256", export._sha256(data_path)),
            )
        ):
            raise ValueError("hard-context baseline does not bind the same model/data inputs")
        baseline_arms = hard_context_baseline.get("inference", {}).get("arms", {})
        if any(baseline_arms.get(arm) is None for arm in ("mini_lm_decoder", "mini_lm_margin_gated")):
            raise ValueError("hard-context baseline is missing a paired mini-LM decoder arm")

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    labels = [str(value) for value in payload["labels"]]
    label_to_index = {label: index for index, label in enumerate(labels)}
    if len(labels) != 372:
        raise ValueError(f"expected frozen 372-label contract, found {len(labels)}")
    layer_count = int(distill["student"]["architecture"]["layers"])
    if int(payload.get("layers", -1)) != layer_count or layer_count < 1:
        raise ValueError("mini-LM checkpoint/report depth contract mismatch")
    mini_lm_trainer.LAYERS = layer_count
    model = MiniFormulaLM(len(labels), len(payload["relations"]), int(payload["max_positions"]))
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    inputs, examples = export._current_inputs(rows, model, labels)
    if len(examples) != len(rows):
        raise AssertionError("context-model rows do not align with selected Fast groups")

    session = export.ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_logits = export._predict_ort(session, inputs).astype(np.float32, copy=False)
    soft_context_contract = None
    soft_mix_onehot_error = None
    if args.soft_context_mix is None:
        logits = onnx_logits
    else:
        import audit_mini_lm_soft_top5_context_shadow_v1 as soft_context_audit

        candidate_ids, candidate_weights, soft_positions, mean_top5_mass = soft_context_audit._soft_context_tensors(
            rows, examples, inputs, labels, model,
        )
        onehot_logits = soft_context_audit._predict_soft(
            model, inputs, candidate_ids, candidate_weights, force_top1=True,
        )
        soft_mix_onehot_error = float(np.max(np.abs(onnx_logits - onehot_logits)))
        if soft_mix_onehot_error > 1e-5:
            raise AssertionError(f"soft-context one-hot control failed: {soft_mix_onehot_error}")
        logits = soft_context_audit._predict_soft(
            model, inputs, candidate_ids, candidate_weights, soft_mix=args.soft_context_mix,
        )
        soft_context_contract = {
            "mix": args.soft_context_mix,
            "selection_report_sha256": export._sha256(soft_mix_report_path),
            "selection_method": "writer-LOFO; this held writer's formula labels were excluded from mixture selection",
            "softened_neighbor_positions": soft_positions,
            "top_k": soft_context_audit.TOP_K,
            "mean_original_top5_probability_mass": mean_top5_mass,
            "onehot_control_max_abs_error_vs_onnx_hard_context": soft_mix_onehot_error,
            "onehot_control_error_gate": 1e-5,
        }
    logits = np.asarray(logits, dtype=np.float32)
    if logits.shape != (len(rows), len(labels)) or not np.isfinite(logits).all():
        raise AssertionError("mini-LM emitted invalid logits")
    logp = logits - np.logaddexp.reduce(logits, axis=1, keepdims=True)
    context_scores = {}
    row_by_id = {str(row["record_id"]): row for row in rows}
    for index, example in enumerate(examples):
        record_id = str(example["record_id"])
        context_scores[record_id] = {
            token: float(logp[index, label_to_index[token]])
            for token in row_by_id[record_id]["final_topk"]
        }

    summary_by_id = {str(record["sample_id"]): record for record in summary["records"]}
    raw_by_id = {str(row["sample_id"]): row for row in raw}
    predictions = {"fast": {}, "decoder": {}, "mini_lm_decoder": {}, "mini_lm_margin_gated": {}}
    accepted = Counter()
    relation_graph_changes = 0
    formula_details = []
    gate_cases: dict[str, dict[str, object]] = {}
    for formula_id, target in targets.items():
        fast = summary_by_id[formula_id]["hwr_tournament"]["fast"]
        selected = list(fast["selected_symbols"])
        groups = [[int(value) for value in group] for group in fast["groups"]]
        decoder_symbols = []
        for ordinal, symbol in enumerate(selected):
            record_id = f"{formula_id}:{ordinal}"
            source = row_by_id[record_id]
            decoder_symbols.append({
                "stroke_indices": [int(value) for value in symbol["stroke_indices"]],
                "hwr_topk": [str(value) for value in symbol["hwr_topk"]],
                "hwr_topk_probabilities": [float(value) for value in symbol["hwr_topk_probabilities"]],
                "geometry": dict(source["geometry"]),
            })
            predictions["fast"][record_id] = str(symbol["hwr_topk"][0])

        stroke_count = len(raw_by_id[formula_id]["strokes"])
        base = decode_selective_partition(formula_id, groups, decoder_symbols, stroke_count=stroke_count)
        corrected = decode_selective_partition(
            formula_id, groups, decoder_symbols, stroke_count=stroke_count,
            context_log_probabilities=context_scores, context_weight=FROZEN_CONTEXT_WEIGHT,
        )
        before_edges = []
        after_edges = []
        graph_changed = False
        if base.get("accepted") and corrected.get("accepted"):
            edge_key = lambda result: sorted(
                (str(edge.get("parent")), str(edge.get("child")), str(edge.get("type")))
                for edge in result["relations"]
            )
            before_edges, after_edges = edge_key(base), edge_key(corrected)
            graph_changed = before_edges != after_edges
            relation_graph_changes += int(graph_changed)
        fallback_map = {
            f"{formula_id}:{ordinal}": predictions["fast"][f"{formula_id}:{ordinal}"]
            for ordinal in range(len(decoder_symbols))
        }
        fallback = [fallback_map[f"{formula_id}:{ordinal}"] for ordinal in range(len(decoder_symbols))]
        base_tokens = list(base["tokens"]) if base.get("accepted") else fallback
        corrected_tokens = list(corrected["tokens"]) if corrected.get("accepted") else fallback
        for name, decoded in (("decoder", base), ("mini_lm_decoder", corrected)):
            accepted[name] += int(bool(decoded.get("accepted")))
            for ordinal, symbol in enumerate(decoder_symbols):
                record_id = f"{formula_id}:{ordinal}"
                token = decoded["tokens"][ordinal] if decoded.get("accepted") else fallback_map[record_id]
                if token not in symbol["hwr_topk"]:
                    raise AssertionError("decoder emitted a token outside HWR Top-5")
                predictions[name][record_id] = token
        gate_cases[formula_id] = {
            "formula_id": formula_id,
            "writer_id": str(target["writer_id"]),
            "group_exact": bool(target["group_exact"]),
            "truth": [str(value) for value in target["tokens"]] if target["group_exact"] else [],
            "groups": groups,
            "symbols": decoder_symbols,
            "stroke_count": stroke_count,
            "fast_tokens": fallback,
            "decoder_tokens": base_tokens,
            "mini_lm_tokens": corrected_tokens,
            "mini_lm_relations": list(corrected.get("relations", [])) if corrected.get("accepted") else [],
        }
        if graph_changed or base_tokens != corrected_tokens:
            truth = [str(value) for value in target["tokens"]] if target["group_exact"] else []
            token_edits = []
            if target["group_exact"]:
                for ordinal, (before_token, after_token, true_token, symbol) in enumerate(
                    zip(base_tokens, corrected_tokens, truth, decoder_symbols, strict=True)
                ):
                    if before_token != after_token:
                        token_edits.append({
                            "ordinal": ordinal,
                            "truth": true_token,
                            "decoder": before_token,
                            "mini_lm_decoder": after_token,
                            "hwr_top5": symbol["hwr_topk"],
                        })
            formula_details.append({
                "formula_id": formula_id,
                "writer_hash": hashlib.sha256(str(target["writer_id"]).encode("utf-8")).hexdigest()[:12],
                "group_exact": bool(target["group_exact"]),
                "fast_exact": bool(target["group_exact"] and fallback == truth),
                "decoder_exact": bool(target["group_exact"] and base_tokens == truth),
                "mini_lm_decoder_exact": bool(target["group_exact"] and corrected_tokens == truth),
                "relation_graph_changed": graph_changed,
                "decoder_relation_edges": before_edges,
                "mini_lm_relation_edges": after_edges,
                "token_edits": token_edits,
            })

    gate_outputs: dict[str, dict[str, dict[str, object]]] = {}
    for threshold in OVERRIDE_MARGIN_THRESHOLDS:
        key = _threshold_key(threshold)
        gate_outputs[key] = {}
        for formula_id, case in gate_cases.items():
            tokens, decoded, unlocked_groups = _decode_with_override_margin_gate(
                case, context_scores, threshold, args.hwr_top1_margin_cap,
            )
            gate_outputs[key][formula_id] = {
                "tokens": tokens,
                "accepted": bool(decoded.get("accepted")),
                "relations": list(decoded.get("relations", [])) if decoded.get("accepted") else [],
                "unlocked_groups": unlocked_groups,
            }

    writers_by_formula = {formula_id: str(case["writer_id"]) for formula_id, case in gate_cases.items()}
    selected_threshold_by_writer: dict[str, float | None] = {}
    selected_threshold_by_formula: dict[str, float | None] = {}
    nested_writer_folds = []
    for held_writer in sorted(set(writers_by_formula.values())):
        held_ids = [formula_id for formula_id, writer in writers_by_formula.items() if writer == held_writer]
        fit_ids = [formula_id for formula_id, writer in writers_by_formula.items() if writer != held_writer]
        fit_exact_by_threshold = {}
        for threshold in OVERRIDE_MARGIN_THRESHOLDS:
            key = _threshold_key(threshold)
            fit_exact_by_threshold[key] = sum(
                bool(gate_cases[formula_id]["group_exact"])
                and gate_outputs[key][formula_id]["tokens"] == gate_cases[formula_id]["truth"]
                for formula_id in fit_ids
            )
        best_count = max(fit_exact_by_threshold.values())
        tied_thresholds = [
            threshold for threshold in OVERRIDE_MARGIN_THRESHOLDS
            if fit_exact_by_threshold[_threshold_key(threshold)] == best_count
        ]
        if args.threshold_tie_break == "conservative":
            # Conservative tie-break: prefer full lock, otherwise the largest override gap.
            selected_threshold = max(
                tied_thresholds,
                key=lambda threshold: (threshold is None, math.inf if threshold is None else threshold),
            )
        else:
            finite_ties = [threshold for threshold in tied_thresholds if threshold is not None]
            selected_threshold = min(finite_ties) if finite_ties else None
        selected_threshold_by_writer[held_writer] = selected_threshold
        held_exact = 0
        held_unlocked_groups = 0
        held_accepted = 0
        for formula_id in held_ids:
            selected_threshold_by_formula[formula_id] = selected_threshold
            out = gate_outputs[_threshold_key(selected_threshold)][formula_id]
            case = gate_cases[formula_id]
            held_exact += int(bool(case["group_exact"]) and out["tokens"] == case["truth"])
            held_unlocked_groups += int(out["unlocked_groups"])
            held_accepted += int(bool(out["accepted"]))
            for ordinal, token in enumerate(out["tokens"]):
                predictions["mini_lm_margin_gated"][f"{formula_id}:{ordinal}"] = str(token)
        nested_writer_folds.append({
            "held_writer_hash": hashlib.sha256(held_writer.encode("utf-8")).hexdigest()[:12],
            "held_formula_count": len(held_ids),
            "fit_formula_count": len(fit_ids),
            "selected_override_margin_threshold": _threshold_key(selected_threshold),
            "fit_exact_count": best_count,
            "fit_exact_by_threshold": fit_exact_by_threshold,
            "held_exact_count": held_exact,
            "held_unlocked_group_count": held_unlocked_groups,
            "held_structural_decode_accept_count": held_accepted,
        })

    margin_gate_details = []
    graph_changes_always_to_margin_gated = 0
    override_microscope = []
    for formula_id, case in gate_cases.items():
        threshold = selected_threshold_by_formula[formula_id]
        gated = gate_outputs[_threshold_key(threshold)][formula_id]
        gated_tokens = [str(token) for token in gated["tokens"]]
        always_tokens = [str(token) for token in case["mini_lm_tokens"]]
        truth = [str(token) for token in case["truth"]]
        gated_edges = sorted(
            (str(edge.get("parent")), str(edge.get("child")), str(edge.get("type")))
            for edge in gated["relations"]
        )
        always_edges = sorted(
            (str(edge.get("parent")), str(edge.get("child")), str(edge.get("type")))
            for edge in case["mini_lm_relations"]
        )
        graph_changed = bool(gated["accepted"]) and bool(case["mini_lm_relations"]) and gated_edges != always_edges
        graph_changes_always_to_margin_gated += int(graph_changed)
        fast_exact = bool(case["group_exact"] and case["fast_tokens"] == truth)
        always_exact = bool(case["group_exact"] and always_tokens == truth)
        margin_gated_exact = bool(case["group_exact"] and gated_tokens == truth)
        if fast_exact != always_exact or always_exact != margin_gated_exact:
            group_evidence = []
            for ordinal, symbol in enumerate(case["symbols"]):
                record_id = f"{formula_id}:{ordinal}"
                tokens = [str(token) for token in symbol["hwr_topk"]]
                probabilities = [float(value) for value in symbol["hwr_topk_probabilities"]]
                lm_scores = context_scores[record_id]
                fused_rows = sorted(
                    [
                        (math.log(max(1e-8, probability)) + FROZEN_CONTEXT_WEIGHT * lm_scores[token], token)
                        for token, probability in zip(tokens, probabilities, strict=True)
                    ],
                    key=lambda pair: (-pair[0], pair[1]),
                )
                hwr_top1 = tokens[0]
                fused_winner, fused_winner_token = fused_rows[0]
                fused_by_token = {token: score for score, token in fused_rows}
                override_margin = fused_winner - fused_by_token[hwr_top1]
                hwr_top1_margin = probabilities[0] - max(probabilities[1:], default=0.0)
                fused_gap = fused_rows[0][0] - fused_rows[1][0]
                allowed = fused_winner_token == hwr_top1 or (
                    threshold is not None and override_margin >= threshold
                    and (args.hwr_top1_margin_cap is None or hwr_top1_margin <= args.hwr_top1_margin_cap)
                )
                if (
                    fused_winner_token != hwr_top1
                    or str(case["mini_lm_tokens"][ordinal]) != str(case["fast_tokens"][ordinal])
                    or gated_tokens[ordinal] != str(case["mini_lm_tokens"][ordinal])
                ):
                    hwr_winner_probability = probabilities[tokens.index(fused_winner_token)]
                    group_evidence.append({
                        "ordinal": ordinal,
                        "truth": truth[ordinal],
                        "fast_top1": hwr_top1,
                        "mini_lm_context_winner": fused_winner_token,
                        "mini_lm_structural_decoder": str(case["mini_lm_tokens"][ordinal]),
                        "margin_gated_decoder": gated_tokens[ordinal],
                        "hwr_top1_probability": probabilities[0],
                        "context_winner_hwr_probability": hwr_winner_probability,
                        "hwr_top1_probability_margin": hwr_top1_margin,
                        "context_log_probability_delta_vs_hwr_top1": lm_scores[fused_winner_token] - lm_scores[hwr_top1],
                        "fused_override_margin": override_margin,
                        "fused_winner_runner_up_gap": fused_gap,
                        "selected_override_margin_threshold": _threshold_key(threshold),
                        "override_allowed": allowed,
                        "hwr_top5": tokens,
                    })
            override_microscope.append({
                "formula_id": formula_id,
                "writer_hash": hashlib.sha256(str(case["writer_id"]).encode("utf-8")).hexdigest()[:12],
                "fast_exact": fast_exact,
                "mini_lm_decoder_exact": always_exact,
                "mini_lm_margin_gated_exact": margin_gated_exact,
                "selected_override_margin_threshold": _threshold_key(threshold),
                "changed_groups": group_evidence,
            })
        if gated_tokens != always_tokens or graph_changed:
            token_edits = []
            if case["group_exact"]:
                for ordinal, (before, after, target_token, symbol) in enumerate(
                    zip(always_tokens, gated_tokens, truth, case["symbols"], strict=True)
                ):
                    if before != after:
                        token_edits.append({
                            "ordinal": ordinal,
                            "truth": target_token,
                            "mini_lm_decoder": before,
                            "mini_lm_margin_gated_decoder": after,
                            "hwr_top5": symbol["hwr_topk"],
                        })
            margin_gate_details.append({
                "formula_id": formula_id,
                "writer_hash": hashlib.sha256(str(case["writer_id"]).encode("utf-8")).hexdigest()[:12],
                "selected_override_margin_threshold": _threshold_key(threshold),
                "group_exact": bool(case["group_exact"]),
                "mini_lm_decoder_exact": bool(case["group_exact"] and always_tokens == truth),
                "mini_lm_margin_gated_exact": bool(case["group_exact"] and gated_tokens == truth),
                "unlocked_group_count": int(gated["unlocked_groups"]),
                "relation_graph_changed": graph_changed,
                "mini_lm_relations": always_edges,
                "margin_gated_relations": gated_edges,
                "token_edits": token_edits,
            })

    metrics = {name: export._formula_metrics(rows, targets, values) for name, values in predictions.items()}
    if any(metric["candidate_preservation_rate"] != 1.0 for metric in metrics.values()):
        raise AssertionError("one decoder arm violated Top-5 candidate preservation")
    if any(len(values) != len(rows) for values in predictions.values()):
        raise AssertionError("prediction arm does not cover all selected Fast groups")
    rows_by_formula: dict[str, list[dict]] = {}
    for row in rows:
        rows_by_formula.setdefault(str(row["formula_id"]), []).append(row)
    per_formula_predictions = []
    for formula_id, target in sorted(targets.items()):
        formula_rows = sorted(rows_by_formula[formula_id], key=lambda row: int(row["context"]["index"]))
        truth_tokens = [str(token) for token in target["tokens"]]
        arm_outputs = {}
        for arm_name, arm_predictions in predictions.items():
            tokens = [str(arm_predictions[str(row["record_id"])]) for row in formula_rows]
            arm_outputs[arm_name] = {
                "tokens": tokens,
                "formula_exact": bool(target["group_exact"] and tokens == truth_tokens),
            }
        per_formula_predictions.append({
            "formula_id": formula_id,
            "writer_hash": hashlib.sha256(str(target["writer_id"]).encode("utf-8")).hexdigest()[:12],
            "group_exact": bool(target["group_exact"]),
            "truth_tokens": truth_tokens,
            "arms": arm_outputs,
        })
    paired_baseline_comparison = None
    if hard_context_baseline is not None:
        baseline_rows = hard_context_baseline["inference"]["per_formula_predictions"]
        paired_baseline_comparison = {
            "baseline_report_sha256": export._sha256(hard_context_baseline_path),
            "by_arm": {
                arm: _paired_formula_arm(baseline_rows, per_formula_predictions, targets, arm)
                for arm in ("mini_lm_decoder", "mini_lm_margin_gated")
            },
        }
    report = {
        "schema": "aiflow-mini-lm-selective-decoder-shadow/v5",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "consumed_development_joint_decoder_shadow_diagnostic",
        "protocol": {
            "training_performed": False,
            "threshold_selection_performed": True,
            "threshold_selection": "nested writer-LOFO on this consumed development cohort; held-writer labels are excluded from its threshold fit",
            "threshold_grid": [_threshold_key(value) for value in OVERRIDE_MARGIN_THRESHOLDS],
            "hwr_top1_margin_cap": args.hwr_top1_margin_cap,
            "crohme_rows_loaded": 0,
            "consumed_formula_count": len(targets),
            "writer_count": len({str(target["writer_id"]) for target in targets.values()}),
            "mini_lm_layers": layer_count,
            "context_input_policy": (
                "frozen hard-token ONNX context"
                if soft_context_contract is None
                else "frozen Top-5 probability expected embeddings mixed with hard-token embeddings"
            ),
            "soft_context": soft_context_contract,
            "candidate_policy": "reweight only the frozen HWR Top-5; the gated arm applies context when it agrees with HWR Top-1 or its fused winner score exceeds fused HWR Top-1 score by the held-writer override-margin threshold; invalid/missing scores fail closed to Fast",
            "context_weight": FROZEN_CONTEXT_WEIGHT,
            "context_weight_source": "frozen mini-LM teacher fusion weight; no current-cohort sweep",
            "grouping_mutations": 0,
            "product_adopted": False,
            "per_formula_predictions_included": True,
            "warning": "The 149-formula cohort is consumed development data and was used for nested gate selection; this is not independent acceptance or Android latency evidence.",
        },
        "provenance": {
            "checkpoint_sha256": export._sha256(checkpoint),
            "onnx_sha256": export._sha256(onnx_path),
            "distillation_report_sha256": export._sha256(distill_path),
            "summary_sha256": export._sha256(summary_path),
            "formula_data_sha256": export._sha256(data_path),
            "soft_mix_selection_report_sha256": (
                export._sha256(soft_mix_report_path) if soft_mix_report_path is not None else None
            ),
            "hard_context_baseline_report_sha256": (
                export._sha256(hard_context_baseline_path) if hard_context_baseline_path is not None else None
            ),
        },
        "inference": {
            "formulas": len(targets),
            "selected_groups": len(rows),
            "fast_group_exact_formulas": sum(bool(target["group_exact"]) for target in targets.values()),
            "accepted_structural_decodes": dict(accepted),
            "relation_graph_changes_between_decoders": relation_graph_changes,
            "per_formula_predictions": per_formula_predictions,
            "paired_to_hard_context_baseline": paired_baseline_comparison,
            "changed_formula_details": formula_details,
            "override_margin_gate": {
                "score": "fused winning candidate score minus fused HWR Top-1 score",
                "fit_scope": "per-held-writer threshold fit on all other writers; held writer is scored out of fold",
                "threshold_tie_break": (
                    "prefer strict lock, then the largest override-margin threshold"
                    if args.threshold_tie_break == "conservative"
                    else "prefer the lowest finite override-margin threshold among fit-score ties; use strict lock only if no finite threshold ties"
                ),
                "threshold_tie_break_policy": args.threshold_tie_break,
                "frozen_context_weight": FROZEN_CONTEXT_WEIGHT,
                "selected_threshold_by_writer_hash": {
                    hashlib.sha256(writer.encode("utf-8")).hexdigest()[:12]: _threshold_key(value)
                    for writer, value in sorted(selected_threshold_by_writer.items())
                },
                "nested_writer_folds": nested_writer_folds,
                "relation_graph_changes_always_to_margin_gated": graph_changes_always_to_margin_gated,
                "changed_formula_details": margin_gate_details,
                "exact_transition_microscope": override_microscope,
            },
            "arms": metrics,
        },
        "decision": {
            "automatic_default_replacement": False,
            "android_latency_verified": False,
            "next_gate": "freeze the gate using a development cohort, then replicate on independent project-owned writers and real devices",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({
        "event": "mini_lm_selective_decoder_shadow_complete",
        "report": str(output),
        "fast_exact": metrics["fast"]["reranked_formula_exact"],
        "decoder_exact": metrics["decoder"]["reranked_formula_exact"],
        "mini_lm_decoder_exact": metrics["mini_lm_decoder"]["reranked_formula_exact"],
        "mini_lm_margin_gated_exact": metrics["mini_lm_margin_gated"]["reranked_formula_exact"],
        "soft_context_mix": args.soft_context_mix,
        "soft_context_control_error": soft_mix_onehot_error,
        "crohme_rows_loaded": 0,
        "product_adopted": False,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
