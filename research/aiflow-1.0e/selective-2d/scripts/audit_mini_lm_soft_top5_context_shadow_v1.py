#!/usr/bin/env python3
"""Compare hard Top-1 and soft Top-5 visual context for the frozen mini-LM.

This is a no-training, consumed-development shadow diagnostic. It replaces
neighboring hard token embeddings with expected embeddings under the HWR Top-5
distribution; target positions remain masked and final outputs stay inside the
original HWR Top-5.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import site
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

# Resolve the model stack against the system site before exposing this
# workstation's separate user-site ONNX Runtime installation.
_USER_SITE = site.getusersitepackages()
if _USER_SITE in sys.path:
    sys.path.remove(_USER_SITE)

import numpy as np
import torch

import export_mini_formula_lm_onnx_v1 as export
import run_prompt_mini_lm_distillation_v1 as mini_lm_trainer
import train_masked_context_reranker_v1 as masked
import audit_prompt_bert_context_on_frozen149_v1 as context_audit

if _USER_SITE and _USER_SITE not in sys.path:
    sys.path.append(_USER_SITE)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "hwr_failure_cause_microscope_20260928"
DISTILL_DIR = ARTIFACTS / "mini_lm_distill_20261003_one_layer"
DEFAULT_OUTPUT = (
    ARTIFACTS / "mini_lm_onnx_20261002" / "mini_lm_one_layer_onnx_20261003"
    / "soft_top5_context_embedding_shadow_20261003.json"
)
TOP_K = 5
BATCH_SIZE = 64
SOFT_MIX_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _soft_context_tensors(
    rows: list[dict], examples: list[dict], base_inputs: dict[str, np.ndarray],
    labels: list[str], model: mini_lm_trainer.MiniFormulaLM,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    label_to_index = {label: index for index, label in enumerate(labels)}
    rows_by_id = {str(row["record_id"]): row for row in rows}
    rows_by_formula = masked._formulae(rows)
    width = int(base_inputs["input_ids"].shape[1])
    candidate_ids = np.zeros((len(examples), width, TOP_K), dtype=np.int64)
    candidate_weights = np.zeros((len(examples), width, TOP_K), dtype=np.float32)
    context_positions = 0
    top5_mass_sum = 0.0

    for batch_index, example in enumerate(examples):
        record_id = str(example["record_id"])
        target_row = rows_by_id[record_id]
        formula_id = str(target_row["formula_id"])
        sequence = rows_by_formula[formula_id]
        target_ordinal = int(target_row["context"]["index"])
        if target_ordinal >= len(sequence) or str(sequence[target_ordinal]["record_id"]) != record_id:
            raise ValueError(f"formula/token alignment mismatch: {record_id}")

        ids = [int(value) for value in example["input_ids"]]
        symbol_positions = [position for position, value in enumerate(ids) if value < len(labels)]
        context_rows = [row for index, row in enumerate(sequence) if index != target_ordinal]
        if len(symbol_positions) != len(context_rows):
            raise ValueError(f"soft context position mismatch: {record_id}")
        if ids[int(example["mask_position"])] != model.mask_id:
            raise AssertionError("masked target input was not preserved")

        for position, context_row in zip(symbol_positions, context_rows, strict=True):
            tokens = [str(value) for value in context_row["final_topk"]]
            probabilities = np.asarray(context_row["final_topk_probabilities"], dtype=np.float64)
            if len(tokens) != TOP_K or probabilities.shape != (TOP_K,):
                raise ValueError(f"expected exactly {TOP_K} HWR candidates: {context_row['record_id']}")
            if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
                raise ValueError(f"invalid HWR candidate probabilities: {context_row['record_id']}")
            mass = float(probabilities.sum())
            if mass <= 0:
                raise ValueError(f"empty HWR Top-{TOP_K} probability mass: {context_row['record_id']}")
            if any(token not in label_to_index for token in tokens):
                raise ValueError(f"HWR candidate outside mini-LM vocabulary: {context_row['record_id']}")
            candidate_ids[batch_index, position] = [label_to_index[token] for token in tokens]
            candidate_weights[batch_index, position] = (probabilities / mass).astype(np.float32)
            context_positions += 1
            top5_mass_sum += mass

    mean_top5_probability_mass = top5_mass_sum / context_positions if context_positions else 0.0
    return candidate_ids, candidate_weights, context_positions, mean_top5_probability_mass


@torch.inference_mode()
def _predict_soft(
    model: mini_lm_trainer.MiniFormulaLM,
    base_inputs: dict[str, np.ndarray],
    candidate_ids: np.ndarray,
    candidate_weights: np.ndarray,
    *,
    soft_mix: float = 1.0,
    force_top1: bool = False,
) -> np.ndarray:
    if not 0.0 <= soft_mix <= 1.0:
        raise ValueError("soft_mix must be in [0, 1]")
    outputs = []
    for start in range(0, len(base_inputs["input_ids"]), BATCH_SIZE):
        stop = min(start + BATCH_SIZE, len(base_inputs["input_ids"]))
        input_ids = torch.from_numpy(base_inputs["input_ids"][start:stop])
        attention = torch.from_numpy(base_inputs["attention_mask"][start:stop])
        mask_positions = torch.from_numpy(base_inputs["mask_positions"][start:stop])
        ids = torch.from_numpy(candidate_ids[start:stop])
        weights = torch.from_numpy(candidate_weights[start:stop].copy())
        if force_top1:
            active = weights.sum(dim=-1) > 0
            weights.zero_()
            weights[..., 0] = active.to(weights.dtype)

        hard_embeddings = model.token_embedding(input_ids)
        candidate_embeddings = model.token_embedding(ids)
        expected_embeddings = torch.sum(candidate_embeddings * weights.unsqueeze(-1), dim=-2)
        if soft_mix == 1.0:
            mixed_embeddings = expected_embeddings
        elif soft_mix == 0.0:
            mixed_embeddings = hard_embeddings
        else:
            mixed_embeddings = hard_embeddings + soft_mix * (expected_embeddings - hard_embeddings)
        active_positions = weights.sum(dim=-1) > 0
        token_embeddings = torch.where(
            active_positions.unsqueeze(-1), mixed_embeddings, hard_embeddings,
        )
        positions = torch.arange(input_ids.shape[1]).unsqueeze(0)
        hidden = token_embeddings + model.position_embedding(positions)
        encoded = model.encoder(hidden, src_key_padding_mask=~attention.bool())
        masked = encoded[torch.arange(len(encoded)), mask_positions]
        outputs.append(model.classifier(model.output_norm(masked)).cpu().numpy())
    return np.concatenate(outputs, axis=0).astype(np.float32, copy=False)


def _fuse(rows: list[dict], logits: np.ndarray, labels: list[str], weight: float) -> dict[str, str]:
    log_probabilities = logits - np.logaddexp.reduce(logits, axis=1, keepdims=True)
    context = {
        str(row["record_id"]): values
        for row, values in zip(rows, log_probabilities, strict=True)
    }
    predictions = masked._fused_predictions(rows, context, labels, weight)
    if any(predictions[str(row["record_id"])] not in row["final_topk"] for row in rows):
        raise AssertionError("soft context emitted a symbol outside the frozen HWR Top-5")
    return predictions


def _formula_exact(rows: list[dict], targets: dict[str, dict], predictions: dict[str, str]) -> dict[str, bool]:
    by_formula: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_formula[str(row["formula_id"])].append(row)
    result = {}
    for formula_id, sequence in by_formula.items():
        sequence.sort(key=lambda row: int(row["context"]["index"]))
        target = targets[formula_id]
        result[formula_id] = bool(
            target["group_exact"]
            and [predictions[str(row["record_id"])] for row in sequence] == target["tokens"]
        )
    return result


def _paired_comparison(
    rows: list[dict], targets: dict[str, dict], before: dict[str, str], after: dict[str, str],
) -> dict:
    before_exact = _formula_exact(rows, targets, before)
    after_exact = _formula_exact(rows, targets, after)
    transitions: Counter[str] = Counter()
    deltas = {}
    formula_transitions = []
    for formula_id in targets:
        transitions[("exact" if before_exact[formula_id] else "wrong") + "_to_" +
                    ("exact" if after_exact[formula_id] else "wrong")] += 1
        deltas[formula_id] = int(after_exact[formula_id]) - int(before_exact[formula_id])
        formula_transitions.append({
            "formula_id": formula_id,
            "writer_hash": hashlib.sha256(
                str(targets[formula_id]["writer_id"]).encode("utf-8")
            ).hexdigest()[:12],
            "before_exact": before_exact[formula_id],
            "after_exact": after_exact[formula_id],
        })
    changed = improved = regressed = 0
    changed_details = []
    for row in rows:
        record_id = str(row["record_id"])
        old, new = before[record_id], after[record_id]
        if old == new:
            continue
        changed += 1
        formula_id = str(row["formula_id"])
        target = targets[formula_id]
        truth = target["tokens"][int(row["context"]["index"])] if target["group_exact"] else None
        old_correct = truth is not None and old == truth
        new_correct = truth is not None and new == truth
        improved += int(not old_correct and new_correct)
        regressed += int(old_correct and not new_correct)
        changed_details.append({
            "formula_id": formula_id,
            "record_id": record_id,
            "previous": old,
            "soft_top5": new,
            "target": truth,
            "improved": bool(not old_correct and new_correct),
            "regressed": bool(old_correct and not new_correct),
        })
    return {
        "formula_exact_before": sum(before_exact.values()),
        "formula_exact_after": sum(after_exact.values()),
        "formula_exact_delta": sum(after_exact.values()) - sum(before_exact.values()),
        "formula_exact_transitions": dict(sorted(transitions.items())),
        "formula_exact_recovered": sum(value == 1 for value in deltas.values()),
        "formula_exact_regressed": sum(value == -1 for value in deltas.values()),
        "formula_exact_transitions_by_formula": [
            item for item in formula_transitions if item["before_exact"] != item["after_exact"]
        ],
        "writer_bootstrap_delta_pp_95_ci": context_audit._writer_bootstrap(deltas, targets)[
            "delta_formula_exact_pp_95_ci"
        ],
        "changed_tokens": changed,
        "improved_tokens": improved,
        "regressed_tokens": regressed,
        "changed_token_details": changed_details,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DISTILL_DIR / "mini_formula_lm.pt")
    parser.add_argument("--distillation-report", type=Path, default=DISTILL_DIR / "mini_formula_lm_distillation_report.json")
    parser.add_argument("--summary", type=Path, default=export.DEFAULT_SUMMARY)
    parser.add_argument("--data", type=Path, default=export.DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    checkpoint, distill_path = args.checkpoint.resolve(), args.distillation_report.resolve()
    summary_path, data_path, output = args.summary.resolve(), args.data.resolve(), args.output.resolve()
    if output.exists():
        parser.error(f"refusing to overwrite existing report: {output}")
    for name, path in (("checkpoint", checkpoint), ("distillation report", distill_path),
                       ("frozen summary", summary_path), ("formula data", data_path)):
        if not path.is_file():
            parser.error(f"missing {name}: {path}")

    distill = _read_json(distill_path)
    protocol = distill.get("protocol", {})
    if protocol.get("crohme_rows_loaded") != 0 or protocol.get("crohme_training_or_tuning") is not False:
        raise ValueError("distillation report does not attest CROHME exclusion")
    if _sha256(checkpoint) != distill["student"]["checkpoint_sha256"]:
        raise ValueError("checkpoint hash differs from distillation report")
    summary = _read_json(summary_path)
    if summary.get("crohme_training_or_tuning") is not False or summary.get("product_default_enabled") is not False:
        raise ValueError("summary violates CROHME/product-off guard")
    if _sha256(data_path) != summary["inputs"]["formulas_valid_sha256"]:
        raise ValueError("formula data hash differs from frozen summary")
    raw = export._jsonl(data_path)
    if any("crohme" in str(row.get("source_partition", "")).casefold() for row in raw):
        raise ValueError("CROHME-marked row found in diagnostic data")
    raw_by_id = {str(row["sample_id"]): row for row in raw}
    rows, targets = export._formula_rows(summary, raw_by_id)
    if len(targets) != 149:
        raise ValueError(f"expected the consumed 149-formula shadow set; found {len(targets)}")

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    labels = [str(value) for value in payload["labels"]]
    layers = int(payload["layers"])
    if layers != int(distill["student"]["architecture"]["layers"]):
        raise ValueError("checkpoint/report layer count mismatch")
    mini_lm_trainer.LAYERS = layers
    model = mini_lm_trainer.MiniFormulaLM(
        len(labels), len(payload["relations"]), int(payload["max_positions"]),
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()

    base_inputs, examples = export._current_inputs(rows, model, labels)
    candidate_ids, candidate_weights, soft_positions, mean_top5_mass = _soft_context_tensors(
        rows, examples, base_inputs, labels, model,
    )
    hard_logits = export._predict_torch(model, base_inputs).astype(np.float32, copy=False)
    onehot_soft_logits = _predict_soft(
        model, base_inputs, candidate_ids, candidate_weights, force_top1=True,
    )
    onehot_max_error = float(np.max(np.abs(hard_logits - onehot_soft_logits)))
    if onehot_max_error > 1e-5:
        raise AssertionError(f"one-hot expected-embedding control failed: {onehot_max_error}")
    alpha_logits = {0.0: hard_logits}
    for soft_mix in SOFT_MIX_GRID[1:]:
        alpha_logits[soft_mix] = _predict_soft(
            model, base_inputs, candidate_ids, candidate_weights, soft_mix=soft_mix,
        )
    soft_logits = alpha_logits[1.0]
    fusion_weight = float(protocol["fusion_lambda_frozen_from_teacher"])

    predictions_by_mix = {
        soft_mix: _fuse(rows, alpha_logits[soft_mix], labels, fusion_weight)
        for soft_mix in SOFT_MIX_GRID
    }
    hard_predictions = predictions_by_mix[0.0]
    soft_predictions = predictions_by_mix[1.0]
    fast_predictions = {str(row["record_id"]): str(row["final_topk"][0]) for row in rows}
    hard_metrics = export._formula_metrics(rows, targets, hard_predictions)
    soft_metrics = export._formula_metrics(rows, targets, soft_predictions)
    fast_metrics = export._formula_metrics(rows, targets, fast_predictions)
    if hard_metrics["candidate_preservation_rate"] != 1.0 or soft_metrics["candidate_preservation_rate"] != 1.0:
        raise AssertionError("candidate-preservation invariant failed")
    if soft_metrics["new_tokens"] != 0 or soft_metrics["grouping_mutations"] != 0:
        raise AssertionError("soft context changed token inventory or grouping")

    writers = sorted({str(target["writer_id"]) for target in targets.values()})
    exact_by_mix = {
        soft_mix: _formula_exact(rows, targets, predictions_by_mix[soft_mix])
        for soft_mix in SOFT_MIX_GRID
    }
    writer_by_formula = {
        formula_id: str(target["writer_id"]) for formula_id, target in targets.items()
    }
    crossfit_predictions: dict[str, str] = {}
    crossfit_folds = []
    selected_alpha_counts: Counter[str] = Counter()
    for held_writer in writers:
        fit_formulas = [
            formula_id for formula_id in targets if writer_by_formula[formula_id] != held_writer
        ]
        fit_scores = {
            soft_mix: sum(exact_by_mix[soft_mix][formula_id] for formula_id in fit_formulas)
            for soft_mix in SOFT_MIX_GRID
        }
        # Prefer the least-displaced input on ties to limit the untrained-input shift.
        selected_mix = min(SOFT_MIX_GRID, key=lambda value: (-fit_scores[value], value))
        selected_alpha_counts[f"{selected_mix:.2f}"] += 1
        held_formula_ids = [
            formula_id for formula_id in targets if writer_by_formula[formula_id] == held_writer
        ]
        held_record_ids = {
            str(row["record_id"])
            for row in rows if str(row["formula_id"]) in set(held_formula_ids)
        }
        crossfit_predictions.update({
            record_id: predictions_by_mix[selected_mix][record_id]
            for record_id in held_record_ids
        })
        crossfit_folds.append({
            "held_writer_hash": hashlib.sha256(held_writer.encode("utf-8")).hexdigest()[:12],
            "fit_formula_count": len(fit_formulas),
            "held_formula_count": len(held_formula_ids),
            "selected_soft_mix": selected_mix,
            "fit_formula_exact_by_mix": {f"{value:.2f}": fit_scores[value] for value in SOFT_MIX_GRID},
        })
    if set(crossfit_predictions) != {str(row["record_id"]) for row in rows}:
        raise AssertionError("writer-crossfit predictions do not cover each token exactly once")
    crossfit_metrics = export._formula_metrics(rows, targets, crossfit_predictions)
    crossfit_vs_hard = _paired_comparison(rows, targets, hard_predictions, crossfit_predictions)
    if crossfit_metrics["candidate_preservation_rate"] != 1.0:
        raise AssertionError("crossfit soft mix emitted a symbol outside the frozen HWR Top-5")
    if crossfit_metrics["new_tokens"] != 0 or crossfit_metrics["grouping_mutations"] != 0:
        raise AssertionError("crossfit soft mix changed token inventory or grouping")

    report = {
        "schema": "aiflow-mini-lm-soft-top5-context-shadow/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "consumed_development_shadow_diagnostic",
        "protocol": {
            "training_performed": False,
            "threshold_selection_performed": False,
            "soft_mix_selection": "nested writer-LOFO over a frozen grid; one writer's formulas are never used to choose that writer's mix",
            "soft_mix_grid": list(SOFT_MIX_GRID),
            "crohme_rows_loaded": 0,
            "consumed_formula_count": len(targets),
            "writer_count": len({str(target["writer_id"]) for target in targets.values()}),
            "mini_lm_layers": layers,
            "candidate_policy": "only the original HWR Top-5; no new classes, regrouping, or iterative self-feedback",
            "context_policy": "neighbor token embedding is the normalized HWR Top-5 probability-weighted expected embedding; target stays masked; relation markers are unchanged",
            "top5_probability_normalization": "renormalize within the provided Top-5 candidates",
            "fusion_weight": fusion_weight,
            "fusion_weight_source": "frozen distillation teacher; no current-cohort tuning",
            "product_adopted": False,
            "warning": "The 149-formula set is consumed development data; soft embeddings were not trained or selected on independent writers, and this is not Android latency evidence.",
        },
        "provenance": {
            "checkpoint_sha256": _sha256(checkpoint),
            "distillation_report_sha256": _sha256(distill_path),
            "summary_sha256": _sha256(summary_path),
            "formula_data_sha256": _sha256(data_path),
            "audit_script_sha256": _sha256(Path(__file__).resolve()),
        },
        "input_contract": {
            "formula_examples": len(examples),
            "softened_neighbor_positions": soft_positions,
            "top_k": TOP_K,
            "mean_original_top5_probability_mass": mean_top5_mass,
            "onehot_expected_embedding_max_abs_error_vs_hard_path": onehot_max_error,
            "onehot_control_error_gate": 1e-5,
        },
        "metrics": {
            "fast_top1": fast_metrics,
            "hard_top1_context": hard_metrics,
            "soft_top5_expected_embedding_context": soft_metrics,
            "paired_soft_vs_hard": _paired_comparison(rows, targets, hard_predictions, soft_predictions),
            "writer_crossfit_soft_mix": {
                "formula_metrics": crossfit_metrics,
                "selected_mix_counts_by_held_writer": dict(sorted(selected_alpha_counts.items())),
                "folds": crossfit_folds,
                "paired_vs_hard_top1_context": crossfit_vs_hard,
            },
        },
        "decision": {
            "candidate_preservation_passed": True,
            "onehot_control_passed": True,
            "android_latency_verified": False,
            "promotion_eligible": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({
        "event": "mini_lm_soft_top5_context_shadow_complete",
        "report": str(output),
        "fast_exact": fast_metrics["reranked_formula_exact"],
        "hard_context_exact": hard_metrics["reranked_formula_exact"],
        "soft_context_exact": soft_metrics["reranked_formula_exact"],
        "soft_vs_hard_delta": report["metrics"]["paired_soft_vs_hard"]["formula_exact_delta"],
        "writer_crossfit_soft_mix_exact": crossfit_metrics["reranked_formula_exact"],
        "writer_crossfit_vs_hard_delta": crossfit_vs_hard["formula_exact_delta"],
        "selected_mix_counts": dict(sorted(selected_alpha_counts.items())),
        "onehot_control_error": onehot_max_error,
        "crohme_rows_loaded": 0,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
