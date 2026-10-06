#!/usr/bin/env python3
"""Train BERT-Tiny context weights on project-owned, evaluation-disjoint prompts."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics

import torch

from character_tensor_v1 import ROOT, _json_lines
from evaluate_homograph_context_reranker_v1 import STRICT_FAMILIES, _metrics
from train_masked_context_reranker_v1 import (
    DEFAULT_PRETRAINED,
    DEFAULT_PRODUCT,
    LAMBDA_GRID,
    MODEL_ID,
    MODEL_REVISION,
    RELATIONS,
    SCHEMA as CHECKPOINT_SCHEMA,
    SEED,
    _build_model,
    _candidate_audit,
    _context_log_probabilities,
    _fused_predictions,
    _labels,
    _metric_delta,
    _sha256,
    _train_epochs,
    _verify_pretrained,
    _write_prediction_rows,
)


SCHEMA = "aiflow-prompt-context-reranker/v1"
DEFAULT_CORPUS = ROOT / "artifacts" / "prompt_context_corpus_20260820_r2" / "prompt_context_corpus.jsonl"
DEFAULT_AUDIT = ROOT / "artifacts" / "prompt_context_corpus_20260820_r2" / "prompt_context_corpus_audit.json"
DEFAULT_CANDIDATES = ROOT / "artifacts" / "homograph_context_20260814"
DEFAULT_OUTPUT = ROOT / "artifacts" / "prompt_context_bert_tiny_20260820_r1_shadow"
STRICT_TOKENS = frozenset().union(*STRICT_FAMILIES.values())


def _d_path(path: Path, label: str, *, file: bool = False) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.drive.upper() != "D:" or file and not resolved.is_file():
        raise ValueError(f"{label} must remain on D: {resolved}")
    return resolved


def _prompt_rows(path: Path, labels: list[str]) -> tuple[list[dict], dict]:
    allowed = set(labels)
    output = []
    seen = set()
    excluded = []
    for source in _json_lines(path):
        formula_id = f"prompt::{source['formula_id']}"
        if formula_id in seen:
            raise ValueError(f"duplicate prompt formula: {formula_id}")
        seen.add(formula_id)
        tokens = [str(value) for value in source["labels"]]
        unknown = sorted(set(tokens) - allowed)
        if unknown:
            excluded.append({"formula_id": formula_id, "unsupported_labels": unknown})
            continue
        length = len(tokens)
        for index, token in enumerate(tokens):
            width = 1.0 / length
            output.append({
                "record_id": f"{formula_id}::{index}",
                "formula_id": formula_id,
                "writer_group": "project-owned-prompt-corpus",
                "label": token,
                "final_topk": [token],
                "final_topk_probabilities": [1.0],
                "geometry": {
                    "left": index * width,
                    "top": 0.0,
                    "right": (index + 1) * width,
                    "bottom": 1.0,
                    "width": width,
                    "height": 1.0,
                    "center_x": (index + 0.5) * width,
                    "center_y": 0.5,
                    "width_rel": width,
                    "height_rel": 1.0,
                    "stroke_count": 0.0,
                },
                "context": {"index": index, "length": length},
            })
    if not output:
        raise ValueError("prompt context corpus is empty")
    return output, {
        "admitted_formulas": len({row["formula_id"] for row in output}),
        "admitted_records": len(output),
        "excluded_unsupported_formulas": len(excluded),
        "unsupported_labels": sorted({label for row in excluded for label in row["unsupported_labels"]}),
    }


def _split(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    formulae = sorted({str(row["formula_id"]) for row in rows})
    held = {formula_id for index, formula_id in enumerate(formulae) if index % 5 == 0}
    fit = [row for row in rows if str(row["formula_id"]) not in held]
    validation = [row for row in rows if str(row["formula_id"]) in held]
    if not fit or not validation or {row["formula_id"] for row in fit} & {row["formula_id"] for row in validation}:
        raise AssertionError("invalid prompt formula split")
    return fit, validation


def _masked_metrics(model, contract: dict, rows: list[dict], labels: list[str], device: torch.device, batch_size: int) -> dict:
    context = _context_log_probabilities(model, contract, rows, device, batch_size)
    predictions = {
        str(row["record_id"]): labels[max(range(len(labels)), key=context[str(row["record_id"])].__getitem__)]
        for row in rows
    }
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["formula_id"])].append(row)
    strict = [row for row in rows if str(row["label"]) in STRICT_TOKENS]
    strict_labels = sorted({str(row["label"]) for row in strict})
    return {
        "records": len(rows),
        "formulas": len(grouped),
        "top1": sum(predictions[str(row["record_id"])] == row["label"] for row in rows) / len(rows),
        "formula_exact": sum(all(predictions[str(row["record_id"])] == row["label"] for row in values) for values in grouped.values()) / len(grouped),
        "strict_macro": (
            sum(
                sum(predictions[str(row["record_id"])] == label for row in strict if row["label"] == label)
                / sum(row["label"] == label for row in strict)
                for label in strict_labels
            ) / len(strict_labels)
            if strict_labels else 0.0
        ),
    }


def _strict_lock(rows: list[dict], predictions: dict[str, str]) -> dict[str, str]:
    return {
        str(row["record_id"]): (
            str(row["final_topk"][0])
            if str(row["final_topk"][0]) in STRICT_TOKENS
            else predictions[str(row["record_id"])]
        )
        for row in rows
    }


def _select_lambda(rows: list[dict], context: dict, labels: list[str]) -> tuple[dict, list[dict]]:
    baseline_predictions = {str(row["record_id"]): str(row["final_topk"][0]) for row in rows}
    baseline = _metrics(rows, baseline_predictions)
    trials = []
    for value in LAMBDA_GRID:
        raw = _fused_predictions(rows, context, labels, value)
        predictions = _strict_lock(rows, raw)
        metrics = _metrics(rows, predictions)
        admissible = (
            all(metrics[key] >= baseline[key] for key in ("all_top1", "strict_micro_top1", "strict_macro_top1", "formula_exact"))
            and metrics["improved"] >= metrics["regressed"]
        )
        trials.append({"lambda": value, "metrics": metrics, "admissible": admissible})
    winner = max(
        (trial for trial in trials if trial["admissible"]),
        key=lambda trial: (
            trial["metrics"]["formula_exact"],
            trial["metrics"]["all_top1"],
            trial["metrics"]["strict_macro_top1"],
            trial["metrics"]["improved"] - trial["metrics"]["regressed"],
            -trial["metrics"]["changed"],
            -trial["lambda"],
        ),
    )
    return winner, trials


def _direct_writer_loo(rows: list[dict], context: dict, labels: list[str]) -> tuple[dict[str, str], list[dict]]:
    predictions, folds = {}, []
    for writer in sorted({str(row["writer_group"]) for row in rows}):
        training = [row for row in rows if str(row["writer_group"]) != writer]
        held = [row for row in rows if str(row["writer_group"]) == writer]
        if {row["formula_id"] for row in training} & {row["formula_id"] for row in held}:
            raise AssertionError("formula leakage in prompt-context writer-LOO calibration")
        winner, _ = _select_lambda(training, context, labels)
        raw = _fused_predictions(held, context, labels, float(winner["lambda"]))
        selected = _strict_lock(held, raw)
        predictions.update(selected)
        folds.append({
            "held_writer_group": writer,
            "training_formulas": len({row["formula_id"] for row in training}),
            "held_formulas": len({row["formula_id"] for row in held}),
            "selected_lambda": winner["lambda"],
            "held_metrics": _metrics(held, selected),
        })
    if set(predictions) != {str(row["record_id"]) for row in rows}:
        raise AssertionError("prompt-context writer-LOO coverage mismatch")
    return predictions, folds


def _self_test() -> None:
    rows = [
        {"record_id": "a", "final_topk": ["|", "1"]},
        {"record_id": "b", "final_topk": ["l", "1"]},
    ]
    assert _strict_lock(rows, {"a": "1", "b": "1"}) == {"a": "|", "b": "1"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--corpus-audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--pretrained", type=Path, default=DEFAULT_PRETRAINED)
    parser.add_argument("--hwr-checkpoint", type=Path, default=DEFAULT_PRODUCT)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--predict-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    _self_test()
    if args.self_test:
        print(json.dumps({"event": "self_test", "status": "pass"}))
        return 0
    if min(args.epochs, args.batch_size, args.predict_batch_size) < 1:
        parser.error("epochs and batch sizes must be positive")

    corpus = _d_path(args.corpus, "prompt corpus", file=True)
    corpus_audit_path = _d_path(args.corpus_audit, "prompt corpus audit", file=True)
    pretrained = _d_path(args.pretrained, "pretrained model")
    hwr_checkpoint = _d_path(args.hwr_checkpoint, "HWR checkpoint", file=True)
    candidates = _d_path(args.candidates, "candidate cache")
    output = _d_path(args.output, "output")
    report_path = output / "prompt_context_report.json"
    checkpoint_path = output / "masked_context_product.pt"
    if output.exists():
        parser.error(f"refusing to overwrite output: {output}")
    audit = json.loads(corpus_audit_path.read_text(encoding="utf-8"))
    if (
        audit.get("schema") != "aiflow-owned-prompt-context-audit/v1"
        or audit.get("contracts", {}).get("evaluation_sequence_overlap") != 0
        or audit.get("contracts", {}).get("arithmetic_evaluation_training") is not False
        or audit.get("contracts", {}).get("relation_inference_training") is not False
        or audit.get("contracts", {}).get("commercial_training_rights") is not True
    ):
        raise ValueError("prompt context admission contract failed")
    model_hashes = _verify_pretrained(pretrained)
    labels = _labels(hwr_checkpoint)
    prompt_rows, prompt_admission = _prompt_rows(corpus, labels)
    fit_rows, validation_rows = _split(prompt_rows)
    direct_path = candidates / "direct_candidates.jsonl.gz"
    crohme_path = candidates / "crohme_candidates.jsonl.gz"
    direct_rows = list(_json_lines(direct_path))
    crohme_rows = list(_json_lines(crohme_path))
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")

    selection_model, _, selection_contract = _build_model(pretrained, labels, device, SEED)
    prompt_before = _masked_metrics(selection_model, selection_contract, validation_rows, labels, device, args.predict_batch_size)
    selection_losses = _train_epochs(
        selection_model, selection_contract, fit_rows, device,
        epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
        weight_decay=args.weight_decay, seed=SEED,
    )
    prompt_after = _masked_metrics(selection_model, selection_contract, validation_rows, labels, device, args.predict_batch_size)
    del selection_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    product_model, _, contract = _build_model(pretrained, labels, device, SEED + 1)
    product_losses = _train_epochs(
        product_model, contract, prompt_rows, device,
        epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
        weight_decay=args.weight_decay, seed=SEED + 1,
    )
    direct_context = _context_log_probabilities(product_model, contract, direct_rows, device, args.predict_batch_size)
    direct_predictions, folds = _direct_writer_loo(direct_rows, direct_context, labels)
    direct_baseline_predictions = {str(row["record_id"]): str(row["final_topk"][0]) for row in direct_rows}
    direct_baseline = _metrics(direct_rows, direct_baseline_predictions)
    direct_metrics = _metrics(direct_rows, direct_predictions)
    product_selection, lambda_trials = _select_lambda(direct_rows, direct_context, labels)
    product_lambda = float(product_selection["lambda"])
    crohme_context = _context_log_probabilities(product_model, contract, crohme_rows, device, args.predict_batch_size)
    crohme_raw = _fused_predictions(crohme_rows, crohme_context, labels, product_lambda)
    crohme_predictions = _strict_lock(crohme_rows, crohme_raw)
    crohme_baseline_predictions = {str(row["record_id"]): str(row["final_topk"][0]) for row in crohme_rows}
    crohme_baseline = _metrics(crohme_rows, crohme_baseline_predictions)
    crohme_metrics = _metrics(crohme_rows, crohme_predictions)

    output.mkdir(parents=True, exist_ok=False)
    checkpoint = {
        "schema": CHECKPOINT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_file_sha256": model_hashes,
        "hwr_checkpoint_sha256": _sha256(hwr_checkpoint),
        "math_labels": labels,
        "class_tokens": contract["class_tokens"],
        "relation_tokens": contract["relation_tokens"],
        "candidate_policy": "strict HWR Top-1 lock then log(P_HWR)+lambda*log(P_prompt_MLM) inside HWR Top-5",
        "lambda": product_lambda,
        "epochs": args.epochs,
        "training_formulae": len({row["formula_id"] for row in prompt_rows}),
        "training_records": len(prompt_rows),
        "training_data": "project-owned prompt formulas; current159 exact token sequences, equations, and 2D relations excluded",
        "state_dict": {key: value.detach().cpu() for key, value in product_model.state_dict().items()},
    }
    torch.save(checkpoint, checkpoint_path)
    _write_prediction_rows(output / "direct_writer_loo_predictions.jsonl.gz", direct_rows, direct_predictions)
    product_direct_raw = _fused_predictions(direct_rows, direct_context, labels, product_lambda)
    _write_prediction_rows(output / "direct_product_refit_predictions.jsonl.gz", direct_rows, _strict_lock(direct_rows, product_direct_raw))
    _write_prediction_rows(output / "crohme_transfer_predictions.jsonl.gz", crohme_rows, crohme_predictions)
    direct_audit = _candidate_audit(direct_rows, direct_predictions)
    crohme_audit = _candidate_audit(crohme_rows, crohme_predictions)
    research_gate = (
        prompt_after["top1"] > prompt_before["top1"]
        and all(direct_metrics[key] >= direct_baseline[key] for key in ("all_top1", "strict_micro_top1", "strict_macro_top1", "formula_exact"))
        and all(crohme_metrics[key] >= crohme_baseline[key] for key in ("all_top1", "strict_micro_top1", "strict_macro_top1", "formula_exact"))
        and direct_audit["candidate_preservation_rate"] == 1.0
        and crohme_audit["candidate_preservation_rate"] == 1.0
    )
    report = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "architecture": {
            "context_encoder": {"model_id": MODEL_ID, "revision": MODEL_REVISION, "layers": 2, "hidden_size": 128, "attention_heads": 2},
            "training_objective": "masked class-token prediction over clean project-owned flat prompt formulas",
            "fusion": "strict HWR Top-1 homograph lock then candidate-preserving HWR+MLM log probability",
        },
        "provenance": {
            "prompt_corpus": str(corpus),
            "prompt_corpus_sha256": _sha256(corpus),
            "prompt_corpus_audit": str(corpus_audit_path),
            "prompt_corpus_audit_sha256": _sha256(corpus_audit_path),
            "direct_candidates_sha256": _sha256(direct_path),
            "crohme_candidates_sha256": _sha256(crohme_path),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": _sha256(checkpoint_path),
        },
        "training": {
            "device": str(device),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "prompt_fit_formulas": len({row["formula_id"] for row in fit_rows}),
            "prompt_validation_formulas": len({row["formula_id"] for row in validation_rows}),
            "prompt_admission": prompt_admission,
            "prompt_fit_losses": selection_losses,
            "prompt_validation_before": prompt_before,
            "prompt_validation_after": prompt_after,
            "product_losses": product_losses,
            "product_lambda": product_lambda,
            "lambda_selection": "nested held-writer calibration on project-owned direct OOF; CROHME unused",
            "writer_loo_folds": folds,
            "lambda_trials": lambda_trials,
        },
        "evaluation": {
            "direct_writer_loo": {
                "baseline": direct_baseline,
                "prompt_context": direct_metrics,
                "delta": _metric_delta(direct_metrics, direct_baseline),
                "candidate_audit": direct_audit,
            },
            "crohme_transfer": {
                "scope": "CC BY-NC repeated diagnostic only; zero model training or lambda selection",
                "baseline": crohme_baseline,
                "prompt_context": crohme_metrics,
                "delta": _metric_delta(crohme_metrics, crohme_baseline),
                "candidate_audit": crohme_audit,
            },
        },
        "decision": {
            "research_gate_passed": research_gate,
            "runtime_status": "shadow",
            "automatic_default_replacement": False,
            "promotion_requirement": "fresh commercial writer/formula-disjoint acceptance",
        },
        "contracts": {
            "evaluation_sequence_overlap": 0,
            "arithmetic_evaluation": False,
            "candidate_preserving": True,
            "new_tokens": 0,
            "grouping_mutations": 0,
            "product_default_enabled": False,
        },
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({
        "event": "prompt_context_complete",
        "output": str(output),
        "prompt_validation_top1": prompt_after["top1"],
        "direct_top1": direct_metrics["all_top1"],
        "direct_formula_exact": direct_metrics["formula_exact"],
        "crohme_top1": crohme_metrics["all_top1"],
        "crohme_formula_exact": crohme_metrics["formula_exact"],
        "research_gate": research_gate,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
