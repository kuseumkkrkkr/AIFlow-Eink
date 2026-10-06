#!/usr/bin/env python3
"""Distill the frozen prompt-context BERT into a compact formula-context LM.

Only the project-owned prompt corpus supplies training labels. The current 149
formulas are inference-only diagnostics; CROHME is never loaded or trained on.
"""

from __future__ import annotations

import hashlib
import gzip
import json
import math
import random
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from audit_prompt_bert_context_on_frozen149_v1 import (
    DEFAULT_CONTEXT_REPORT,
    DEFAULT_DATA,
    DEFAULT_SUMMARY,
    _formula_metrics,
    _formula_rows,
    _jsonl,
    _sha256,
)
from train_masked_context_reranker_v1 import (
    RELATIONS,
    _context_log_probabilities,
    _formulae,
    _fused_predictions,
    _spatial_relation,
    load_product_checkpoint,
)
from train_prompt_context_reranker_v1 import STRICT_TOKENS, _prompt_rows, _split, _strict_lock


SCHEMA = "aiflow-prompt-mini-lm-distillation/v1"
CANONICAL_ROOT = Path(r"D:\AIFlow-Workspace\Projects\Aiflow\aiflow-math-ink-1.0")
DEFAULT_PROMPT_CORPUS = CANONICAL_ROOT / "artifacts" / "prompt_context_corpus_20260820_r2" / "prompt_context_corpus.jsonl"
DEFAULT_PRETRAINED_CONTEXT = CANONICAL_ROOT / "research" / "pretrained" / "google-bert-tiny"
DEFAULT_CHECKPOINT = CANONICAL_ROOT / "artifacts" / "prompt_context_bert_tiny_20260820_r3_epoch3_shadow" / "masked_context_product.pt"
DEFAULT_DIRECT_CANDIDATES = CANONICAL_ROOT / "artifacts" / "homograph_context_20260814" / "direct_candidates.jsonl.gz"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "artifacts" / "hwr_failure_cause_microscope_20260928" / "mini_lm_distill_20261002_r2"
HIDDEN = 64
LAYERS = 2
HEADS = 2
FEEDFORWARD = 128
MAX_EPOCHS = 40
PATIENCE = 6
BATCH_SIZE = 64
LEARNING_RATE = 8e-4
WEIGHT_DECAY = 1e-3
DISTILL_TEMPERATURE = 2.0
DISTILL_WEIGHT = 0.5
SEED = 20261002
MAX_FORMULA_TOKENS = 64
MAX_POSITIONS = MAX_FORMULA_TOKENS * 2 + 1


class MiniFormulaLM(nn.Module):
    def __init__(self, class_count: int, relation_count: int, max_positions: int) -> None:
        super().__init__()
        self.pad_id = class_count + relation_count
        self.cls_id = self.pad_id + 1
        self.sep_id = self.pad_id + 2
        self.mask_id = self.pad_id + 3
        vocabulary_size = self.pad_id + 4
        self.token_embedding = nn.Embedding(vocabulary_size, HIDDEN, padding_idx=self.pad_id)
        self.position_embedding = nn.Embedding(max_positions, HIDDEN)
        layer = nn.TransformerEncoderLayer(
            d_model=HIDDEN,
            nhead=HEADS,
            dim_feedforward=FEEDFORWARD,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=LAYERS, enable_nested_tensor=False)
        for encoder_layer in self.encoder.layers:
            for parameter in encoder_layer.parameters():
                if parameter.ndim > 1:
                    nn.init.xavier_uniform_(parameter)
        self.output_norm = nn.LayerNorm(HIDDEN)
        self.classifier = nn.Linear(HIDDEN, class_count)

    def forward(self, input_ids: torch.Tensor, attention: torch.Tensor, mask_positions: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        encoded = self.encoder(hidden, src_key_padding_mask=~attention.bool())
        masked = encoded[torch.arange(len(encoded), device=encoded.device), mask_positions]
        return self.classifier(self.output_norm(masked))


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _teacher_examples(rows: list[dict], labels: list[str], teacher_logp: dict[str, np.ndarray]) -> list[dict]:
    label_to_index = {label: index for index, label in enumerate(labels)}
    relation_to_id = {relation: index for index, relation in enumerate(RELATIONS)}
    examples = []
    for _, sequence in _formulae(rows).items():
        for target_index, target in enumerate(sequence):
            ids = [len(labels) + len(RELATIONS) + 1]
            mask_position = -1
            for index, row in enumerate(sequence):
                if index:
                    relation = _spatial_relation(sequence[index - 1], row)
                    ids.append(len(labels) + relation_to_id[relation])
                if index == target_index:
                    mask_position = len(ids)
                    ids.append(len(labels) + len(RELATIONS) + 3)
                else:
                    ids.append(label_to_index[str(row["label"])])
            ids.append(len(labels) + len(RELATIONS) + 2)
            if mask_position < 0 or str(target["record_id"]) not in teacher_logp:
                raise ValueError("masked LM teacher/student coverage mismatch")
            examples.append({
                "record_id": str(target["record_id"]),
                "formula_id": str(target["formula_id"]),
                "input_ids": ids,
                "mask_position": mask_position,
                "target": label_to_index[str(target["label"])],
                "teacher_logp": np.asarray(teacher_logp[str(target["record_id"])], dtype=np.float32),
            })
    if {example["record_id"] for example in examples} != {str(row["record_id"]) for row in rows}:
        raise AssertionError("student masked-example coverage mismatch")
    return examples


def _pack(examples: list[dict], model: MiniFormulaLM) -> dict:
    width = max(len(example["input_ids"]) for example in examples)
    input_ids = torch.full((len(examples), width), model.pad_id, dtype=torch.long)
    attention = torch.zeros((len(examples), width), dtype=torch.bool)
    mask_positions = torch.empty(len(examples), dtype=torch.long)
    targets = torch.empty(len(examples), dtype=torch.long)
    teacher = torch.empty((len(examples), len(examples[0]["teacher_logp"])), dtype=torch.float32)
    formula_ids = []
    for index, example in enumerate(examples):
        length = len(example["input_ids"])
        input_ids[index, :length] = torch.tensor(example["input_ids"], dtype=torch.long)
        attention[index, :length] = True
        mask_positions[index] = int(example["mask_position"])
        targets[index] = int(example["target"])
        teacher[index] = torch.from_numpy(example["teacher_logp"])
        formula_ids.append(str(example["formula_id"]))
    counts = Counter(targets.tolist())
    weights = torch.tensor([1.0 / math.sqrt(counts[int(target)]) for target in targets], dtype=torch.float32)
    weights /= weights.mean()
    return {
        "input_ids": input_ids,
        "attention": attention,
        "mask_positions": mask_positions,
        "targets": targets,
        "teacher_logp": teacher,
        "weights": weights,
        "formula_ids": formula_ids,
    }


def _batch(pack: dict, indices: torch.Tensor, device: torch.device) -> dict:
    return {
        key: value[indices].to(device) if isinstance(value, torch.Tensor) else value
        for key, value in pack.items()
        if key != "formula_ids"
    }


def _losses(model: MiniFormulaLM, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = model(batch["input_ids"], batch["attention"], batch["mask_positions"])
    per_row = F.cross_entropy(logits, batch["targets"], reduction="none")
    ce = (per_row * batch["weights"]).sum() / batch["weights"].sum()
    temperature = DISTILL_TEMPERATURE
    teacher_probs = torch.softmax(batch["teacher_logp"] / temperature, dim=-1)
    student_log_probs = F.log_softmax(logits / temperature, dim=-1)
    per_row_kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1) * temperature**2
    kl = (per_row_kl * batch["weights"]).sum() / batch["weights"].sum()
    return (1.0 - DISTILL_WEIGHT) * ce + DISTILL_WEIGHT * kl, ce, kl


@torch.inference_mode()
def _evaluate(model: MiniFormulaLM, pack: dict, device: torch.device) -> dict:
    model.eval()
    batch_size = 256
    logits_rows = []
    weighted_loss = 0.0
    weight_sum = 0.0
    for start in range(0, len(pack["targets"]), batch_size):
        indices = torch.arange(start, min(start + batch_size, len(pack["targets"])))
        batch = _batch(pack, indices, device)
        logits = model(batch["input_ids"], batch["attention"], batch["mask_positions"])
        losses = F.cross_entropy(logits, batch["targets"], reduction="none")
        weighted_loss += float((losses * batch["weights"]).sum())
        weight_sum += float(batch["weights"].sum())
        logits_rows.extend(logits.argmax(dim=-1).cpu().tolist())
    correct = sum(int(prediction == int(target)) for prediction, target in zip(logits_rows, pack["targets"].tolist(), strict=True))
    by_formula: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for formula_id, prediction, target in zip(pack["formula_ids"], logits_rows, pack["targets"].tolist(), strict=True):
        by_formula[formula_id].append((prediction, target))
    formula_exact = sum(all(prediction == target for prediction, target in values) for values in by_formula.values())
    return {
        "records": len(logits_rows),
        "formulas": len(by_formula),
        "masked_symbol_top1": correct / len(logits_rows),
        "formula_exact": formula_exact / len(by_formula) if by_formula else None,
        "weighted_ce": weighted_loss / weight_sum,
    }


def _fit(
    examples: list[dict], class_count: int, relation_count: int, max_positions: int, device: torch.device,
    *, epochs: int, seed: int, validation: dict | None = None, patience: int = PATIENCE,
) -> tuple[MiniFormulaLM, dict]:
    _seed_everything(seed)
    model = MiniFormulaLM(class_count, relation_count, max_positions).to(device)
    train_pack = _pack(examples, model)
    validation_pack = _pack(validation, model) if validation else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    best_epoch, best_loss, stale = 0, math.inf, 0
    best_state = None
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(examples))
        train_ce_total = train_kl_total = 0.0
        for start in range(0, len(order), BATCH_SIZE):
            indices = order[start:start + BATCH_SIZE]
            batch = _batch(train_pack, indices, device)
            optimizer.zero_grad(set_to_none=True)
            loss, ce, kl = _losses(model, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite mini-LM distillation loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_ce_total += float(ce.detach()) * len(indices)
            train_kl_total += float(kl.detach()) * len(indices)
        if validation_pack is not None:
            metric = _evaluate(model, validation_pack, device)
            score = float(metric["weighted_ce"])
            history.append({"epoch": epoch, "train_ce": train_ce_total / len(examples), "train_kl": train_kl_total / len(examples), "validation": metric})
            if score < best_loss - 1e-4:
                best_loss, best_epoch, stale = score, epoch, 0
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            else:
                stale += 1
            if epoch >= 4 and stale >= patience:
                break
        else:
            history.append({"epoch": epoch, "train_ce": train_ce_total / len(examples), "train_kl": train_kl_total / len(examples)})
    if validation_pack is not None:
        if best_state is None:
            raise AssertionError("mini-LM epoch selector did not retain a checkpoint")
        model.load_state_dict(best_state, strict=True)
    metadata = {
        "epochs_run": len(history),
        "selected_epoch": best_epoch if validation_pack is not None else epochs,
        "validation": history[best_epoch - 1]["validation"] if validation_pack is not None else None,
        "history": history,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    return model.eval(), metadata


@torch.inference_mode()
def _student_logp(model: MiniFormulaLM, examples: list[dict], device: torch.device) -> dict[str, np.ndarray]:
    pack = _pack(examples, model)
    result = {}
    model.eval()
    for start in range(0, len(examples), 256):
        indices = torch.arange(start, min(start + 256, len(examples)))
        batch = _batch(pack, indices, device)
        logp = model(batch["input_ids"], batch["attention"], batch["mask_positions"]).log_softmax(dim=-1).cpu().numpy()
        for example, scores in zip(examples[start:start + 256], logp, strict=True):
            result[str(example["record_id"])] = scores
    if set(result) != {str(example["record_id"]) for example in examples}:
        raise AssertionError("student inference output coverage mismatch")
    return result


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-corpus", type=Path, default=DEFAULT_PROMPT_CORPUS)
    parser.add_argument("--context-report", type=Path, default=DEFAULT_CONTEXT_REPORT)
    parser.add_argument("--teacher-checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--teacher-lambda-source", type=Path, default=DEFAULT_DIRECT_CANDIDATES)
    parser.add_argument("--pretrained", type=Path, default=DEFAULT_PRETRAINED_CONTEXT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    args = parser.parse_args()

    prompt_path = args.prompt_corpus.resolve()
    context_report_path = args.context_report.resolve()
    teacher_checkpoint = args.teacher_checkpoint.resolve()
    teacher_lambda_source = args.teacher_lambda_source.resolve()
    summary_path, data_path = args.summary.resolve(), args.data.resolve()
    pretrained, output_dir = args.pretrained.resolve(), args.output.resolve()
    if output_dir.exists():
        parser.error(f"refusing to overwrite existing output directory: {output_dir}")
    for label, path in (("prompt corpus", prompt_path), ("context report", context_report_path), ("teacher checkpoint", teacher_checkpoint), ("teacher lambda source", teacher_lambda_source), ("current shadow summary", summary_path), ("current formula data", data_path)):
        if not path.is_file():
            parser.error(f"missing {label}: {path}")

    context_report = json.loads(context_report_path.read_text(encoding="utf-8"))
    if _sha256(prompt_path) != str(context_report["provenance"]["prompt_corpus_sha256"]):
        raise ValueError("prompt corpus hash mismatch")
    if _sha256(teacher_checkpoint) != str(context_report["provenance"]["checkpoint_sha256"]):
        raise ValueError("frozen teacher checkpoint hash mismatch")
    teacher_lambda_source_hash = _sha256(teacher_lambda_source)
    if teacher_lambda_source_hash != str(context_report["provenance"]["direct_candidates_sha256"]):
        raise ValueError("teacher lambda calibration source differs from its frozen report")
    training_meta = context_report.get("training", {})
    if "CROHME unused" not in str(training_meta.get("lambda_selection", "")):
        raise ValueError("teacher report lacks CROHME-exclusion provenance")
    if "project-owned flat prompt formulas" not in str(context_report.get("architecture", {}).get("training_objective", "")):
        raise ValueError("teacher report does not attest project-owned prompt-only training")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    data_hash = _sha256(data_path)
    if data_hash != str(summary["inputs"]["formulas_valid_sha256"]):
        raise ValueError("current frozen formula data hash mismatch")
    if summary.get("crohme_training_or_tuning") is not False:
        raise ValueError("current source summary lacks CROHME exclusion attestation")

    raw_current = _jsonl(data_path)
    raw_by_id = {str(row["sample_id"]): row for row in raw_current}
    current_ids = {str(record["sample_id"]) for record in summary["records"]}
    if len(current_ids) != 149:
        raise ValueError("expected the frozen 149-formula diagnostic set")
    direct_formula_ids = set()
    direct_candidate_rows = 0
    with gzip.open(teacher_lambda_source, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                direct_formula_ids.add(str(json.loads(line)["formula_id"]))
                direct_candidate_rows += 1
    shared_lambda_formula_ids = current_ids & direct_formula_ids
    prompt_rows_raw = _jsonl(prompt_path)
    if any("project-owned" not in str(row.get("commercial_training_rights", "")).casefold() for row in prompt_rows_raw):
        raise ValueError("prompt corpus contains rows without project-owned training rights")
    if any("crohme" in json.dumps(row, ensure_ascii=False).casefold() for row in prompt_rows_raw):
        raise ValueError("CROHME marker found in prompt training rows")
    prompt_sequences = {tuple(str(token) for token in row["labels"]) for row in prompt_rows_raw}
    current_sequences = {
        tuple(str(cell["token"]) for cell in raw_by_id[sample_id]["target_cells"])
        for sample_id in current_ids
    }
    overlap_count = len(prompt_sequences & current_sequences)
    if overlap_count:
        raise ValueError(f"prompt training and current diagnostic formula sequences overlap: {overlap_count}")

    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    if device_name == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    device = torch.device(device_name)
    teacher, teacher_contract, teacher_payload = load_product_checkpoint(pretrained, teacher_checkpoint, device)
    labels = list(teacher_payload["math_labels"])
    label_to_index = {label: index for index, label in enumerate(labels)}
    prompt_rows, admission = _prompt_rows(prompt_path, labels)
    if admission != context_report["training"]["prompt_admission"]:
        raise ValueError("prompt row admission differs from the frozen teacher report")
    fit_rows, validation_rows = _split(prompt_rows)

    teacher_logp = _context_log_probabilities(teacher, teacher_contract, prompt_rows, device, 128)
    examples = _teacher_examples(prompt_rows, labels, teacher_logp)
    fit_ids = {str(row["formula_id"]) for row in fit_rows}
    fit_examples = [example for example in examples if example["formula_id"] in fit_ids]
    validation_examples = [example for example in examples if example["formula_id"] not in fit_ids]
    if {example["formula_id"] for example in fit_examples} & {example["formula_id"] for example in validation_examples}:
        raise AssertionError("mini-LM prompt formula split leakage")
    if max(len(example["input_ids"]) for example in examples) > MAX_POSITIONS:
        raise ValueError("prompt formula exceeds the fixed 64-symbol position contract")
    max_positions = MAX_POSITIONS
    current_rows, current_targets = _formula_rows(summary, raw_by_id)
    current_max_positions = max(2 * int(row["context"]["length"]) + 1 for row in current_rows)
    if current_max_positions > max_positions:
        raise ValueError(f"current formula exceeds fixed 64-symbol position contract: {current_max_positions}>{max_positions}")

    start = time.perf_counter()
    selected_model, selection = _fit(
        fit_examples, len(labels), len(RELATIONS), max_positions, device,
        epochs=MAX_EPOCHS, seed=SEED, validation=validation_examples,
    )
    final_epochs = int(selection["selected_epoch"])
    del selected_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    student, refit = _fit(
        examples, len(labels), len(RELATIONS), max_positions, device,
        epochs=final_epochs, seed=SEED + 1,
    )

    teacher_predictions = _fused_predictions(
        current_rows,
        _context_log_probabilities(teacher, teacher_contract, current_rows, device, 128),
        labels,
        float(teacher_payload["lambda"]),
    )
    teacher_product_predictions = _strict_lock(current_rows, teacher_predictions)
    # Build the compact model's input without exposing labels to its inference path.
    relation_to_id = {relation: index for index, relation in enumerate(RELATIONS)}
    current_examples = []
    for _, sequence in _formulae(current_rows).items():
        for target_index, target in enumerate(sequence):
            ids = [student.cls_id]
            mask_position = -1
            for index, row in enumerate(sequence):
                if index:
                    ids.append(len(labels) + relation_to_id[_spatial_relation(sequence[index - 1], row)])
                if index == target_index:
                    mask_position = len(ids)
                    ids.append(student.mask_id)
                else:
                    ids.append(label_to_index[str(row["final_topk"][0])])
            ids.append(student.sep_id)
            current_examples.append({
                "record_id": str(target["record_id"]), "formula_id": str(target["formula_id"]),
                "input_ids": ids, "mask_position": mask_position, "target": 0,
                "teacher_logp": np.zeros(len(labels), dtype=np.float32),
            })
    student_context = _student_logp(student, current_examples, device)
    student_raw = _fused_predictions(current_rows, student_context, labels, float(teacher_payload["lambda"]))
    student_product = _strict_lock(current_rows, student_raw)
    baseline = {str(row["record_id"]): str(row["final_topk"][0]) for row in current_rows}
    baseline_metrics = _formula_metrics(current_rows, current_targets, baseline)
    teacher_metrics = _formula_metrics(current_rows, current_targets, teacher_product_predictions)
    student_metrics = _formula_metrics(current_rows, current_targets, student_product)
    if any(student_product[str(row["record_id"])] not in row["final_topk"] for row in current_rows):
        raise AssertionError("student changed a candidate outside HWR Top-5")
    experiment_elapsed_ms = 1000.0 * (time.perf_counter() - start)

    output_dir.mkdir(parents=True, exist_ok=False)
    student_checkpoint = output_dir / "mini_formula_lm.pt"
    torch.save({
        "schema": SCHEMA,
        "labels": labels,
        "relations": list(RELATIONS),
        "hidden": HIDDEN,
        "layers": LAYERS,
        "heads": HEADS,
        "feedforward": FEEDFORWARD,
        "max_positions": max_positions,
        "state_dict": {key: value.detach().cpu() for key, value in student.state_dict().items()},
    }, student_checkpoint)
    report = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "shadow_only_consumed_development_diagnostic",
        "protocol": {
            "teacher_frozen": True,
            "student_trained_only_on_project_owned_prompt_corpus": True,
            "current_149_used_for_training_or_epoch_selection": False,
            "prompt_current_exact_sequence_overlap": overlap_count,
            "current_149_threshold_tuning": False,
            "teacher_lambda_selected_on_current_overlap_cache": bool(shared_lambda_formula_ids),
            "teacher_lambda_source_current_formula_overlap": len(shared_lambda_formula_ids),
            "crohme_rows_loaded": 0,
            "crohme_training_or_tuning": False,
            "fusion_lambda_frozen_from_teacher": float(teacher_payload["lambda"]),
            "product_policy": "candidate-preserving HWR Top-5 plus strict HWR Top-1 homograph lock",
            "promotion_eligible": False,
            "warning": "current149 is consumed development data, and the frozen teacher lambda was selected on a direct-candidate cache sharing formula IDs; this is not independent acceptance or mobile latency evidence",
        },
        "provenance": {
            "prompt_corpus": str(prompt_path),
            "prompt_corpus_sha256": _sha256(prompt_path),
            "prompt_corpus_raw_rows": len(prompt_rows_raw),
            "prompt_admission": admission,
            "prompt_fit_formulas": len({row["formula_id"] for row in fit_rows}),
            "prompt_validation_formulas": len({row["formula_id"] for row in validation_rows}),
            "teacher_report": str(context_report_path),
            "teacher_checkpoint": str(teacher_checkpoint),
            "teacher_checkpoint_sha256": _sha256(teacher_checkpoint),
            "teacher_lambda_source": str(teacher_lambda_source),
            "teacher_lambda_source_sha256": teacher_lambda_source_hash,
            "teacher_lambda_source_candidate_rows": direct_candidate_rows,
            "teacher_lambda_source_formulas": len(direct_formula_ids),
            "teacher_lambda_source_shared_formula_ids_with_current149": sorted(shared_lambda_formula_ids),
            "current_summary": str(summary_path),
            "current_summary_sha256": _sha256(summary_path),
            "current_formula_data_sha256": data_hash,
        },
        "student": {
            "architecture": {"layers": LAYERS, "hidden": HIDDEN, "heads": HEADS, "feedforward": FEEDFORWARD},
            "parameters": sum(parameter.numel() for parameter in student.parameters()),
            "checkpoint_bytes": student_checkpoint.stat().st_size,
            "checkpoint_sha256": _sha256(student_checkpoint),
            "training": {
                "seed": SEED,
                "distill_temperature": DISTILL_TEMPERATURE,
                "distill_weight": DISTILL_WEIGHT,
                "max_epochs": MAX_EPOCHS,
                "selected_epoch": final_epochs,
                "experiment_elapsed_ms_host_not_mobile": experiment_elapsed_ms,
                "selection_validation": selection["validation"],
                "refit_epochs": refit["selected_epoch"],
            },
            "validation": selection["validation"],
        },
        "current149_evaluation": {
            "baseline_fast": baseline_metrics,
            "frozen_teacher_product": teacher_metrics,
            "distilled_student_product": student_metrics,
            "student_minus_fast_formula_exact": student_metrics["delta_formula_exact"],
            "student_minus_teacher_formula_exact": student_metrics["reranked_formula_exact"] - teacher_metrics["reranked_formula_exact"],
        },
        "decision": {
            "automatic_default_replacement": False,
            "runtime_status": "shadow",
            "next_gate": "fresh writer/formula-disjoint acceptance, multi-seed stability, FP32 parity, INT8 and target-device profiling",
        },
    }
    report_path = output_dir / "mini_formula_lm_distillation_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({
        "event": "mini_formula_lm_distillation_complete",
        "report": str(report_path),
        "checkpoint": str(student_checkpoint),
        "parameters": report["student"]["parameters"],
        "checkpoint_bytes": report["student"]["checkpoint_bytes"],
        "validation_top1": selection["validation"]["masked_symbol_top1"],
        "formula_exact_fast_teacher_student": [
            baseline_metrics["baseline_top1_formula_exact"],
            teacher_metrics["reranked_formula_exact"],
            student_metrics["reranked_formula_exact"],
        ],
        "student_token_hits": student_metrics["reranked_top1_token_hits"],
        "crohme_rows_loaded": 0,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
