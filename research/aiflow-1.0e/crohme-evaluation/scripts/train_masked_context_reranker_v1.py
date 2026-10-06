#!/usr/bin/env python3
"""Train and evaluate a candidate-preserving BERT-Tiny formula-context reranker.

The unified 372-class trajectory model stays frozen.  This script masks one
glyph location, encodes the remaining HWR Top-1 sequence plus coarse spatial
relations, and fuses BERT-Tiny class-token probabilities with the existing
HWR Top-5 probabilities.  The reranker cannot invent a class, regroup strokes,
or use CROHME for training or hyperparameter selection.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

try:
    from transformers import BertForMaskedLM, BertTokenizerFast
    from transformers.utils import logging as transformers_logging
except ImportError as error:  # pragma: no cover - environment-specific message
    raise SystemExit(
        "transformers import failed. On this workstation run with "
        "PYTHONNOUSERSITE=1 so transformers 4.48.0 uses huggingface-hub 0.27.1."
    ) from error

from character_tensor_v1 import ROOT, _json_lines
from evaluate_48hz_prefix_v1 import DEFAULT_PRODUCT
from evaluate_homograph_context_reranker_v1 import STRICT_FAMILIES, _fold, _metrics


transformers_logging.set_verbosity_error()

SCHEMA = "aiflow-masked-context-reranker/v1"
MODEL_ID = "google/bert_uncased_L-2_H-128_A-2"
MODEL_REVISION = "30b0a37ccaaa32f332884b96992754e246e48c5f"
DEFAULT_PRETRAINED = ROOT / "research" / "pretrained" / "google-bert-tiny"
DEFAULT_INPUT = ROOT / "artifacts" / "homograph_context_20260814"
DEFAULT_LOGISTIC_REPORT = DEFAULT_INPUT / "homograph_context_report.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "masked_context_bert_tiny_20260814"
EXPECTED_MODEL_FILES = {
    "config.json": "508e1f01aae55d73355cbdd82609be2f43ba5a0d3428837adbe56cf8391f8b39",
    "model.safetensors": "7fb69ad9f6866d8983183c930e33828f326470bf6ad8bbb2ad4ed957a92e9414",
    "vocab.txt": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
}
RELATIONS = ("right", "left", "above", "below", "superscript", "subscript", "overlap")
LAMBDA_GRID = (0.0, 0.025, 0.05, 0.10, 0.20, 0.35, 0.50)
SEED = 20260814


def _event(name: str, **values: object) -> None:
    print(json.dumps({"event": name, **values}, ensure_ascii=False), flush=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _d_path(path: Path, kind: str) -> Path:
    resolved = path.resolve()
    if resolved.drive.upper() != "D:":
        raise ValueError(f"{kind} must remain on D:: {resolved}")
    return resolved


def _verify_pretrained(path: Path) -> dict[str, str]:
    hashes = {}
    for name, expected in EXPECTED_MODEL_FILES.items():
        source = path / name
        if not source.is_file():
            raise FileNotFoundError(f"missing pinned BERT-Tiny file: {source}")
        actual = _sha256(source)
        if actual != expected:
            raise ValueError(f"pinned BERT-Tiny hash mismatch for {name}: {actual}")
        hashes[name] = actual
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    expected_architecture = {
        "hidden_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "vocab_size": 30522,
        "max_position_embeddings": 512,
    }
    if {key: config.get(key) for key in expected_architecture} != expected_architecture:
        raise ValueError("pinned BERT-Tiny architecture contract changed")
    return hashes


def _labels(checkpoint: Path) -> list[str]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    labels = list(payload.get("math_labels", []))
    if len(labels) != 372 or len(set(labels)) != 372 or payload.get("auxiliary_labels"):
        raise ValueError("expected the unified 372-class HWR checkpoint")
    return labels


def _class_tokens(labels: list[str]) -> list[str]:
    return [f"[AIFLOW_C_{index:03d}]" for index in range(len(labels))]


def _relation_tokens() -> dict[str, str]:
    return {relation: f"[AIFLOW_R_{relation.upper()}]" for relation in RELATIONS}


ANCHOR_OVERRIDES = {
    "+": "plus", "-": "minus", "/": "slash", "=": "equals", "<": "less", ">": "greater",
    "|": "vertical bar", "[": "left bracket", "]": "right bracket", "(": "left parenthesis",
    ")": "right parenthesis", r"\{": "left brace", r"\}": "right brace", r"\#": "number sign",
    r"\$": "dollar", r"\%": "percent", r"\&": "and", r"\times": "times",
    r"\div": "divide", r"\mid": "middle", r"\prime": "prime", r"\sqrt{}": "square root",
}


def _anchor_text(label: str) -> str:
    if label in ANCHOR_OVERRIDES:
        return ANCHOR_OVERRIDES[label]
    styled = re.fullmatch(r"\\math(?:bb|cal|frak|scr|ds)\{(.+)\}", label)
    if styled:
        return styled.group(1)
    cleaned = label.replace("\\not\\", " not ").replace("\\", " ")
    cleaned = re.sub(r"[{}]", " ", cleaned)
    cleaned = re.sub(r"[^A-Za-z0-9]+", " ", cleaned).strip()
    return cleaned or "symbol"


def _anchor_ids(tokenizer: BertTokenizerFast, text: str, base_vocab_size: int) -> list[int]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    valid = [index for index in ids if index < base_vocab_size and index != tokenizer.unk_token_id]
    return valid or [tokenizer.unk_token_id]


def _build_model(pretrained: Path, labels: list[str], device: torch.device, seed: int) -> tuple[BertForMaskedLM, BertTokenizerFast, dict]:
    _set_seed(seed)
    tokenizer = BertTokenizerFast.from_pretrained(pretrained, local_files_only=True)
    base_vocab_size = len(tokenizer)
    anchors = [_anchor_ids(tokenizer, _anchor_text(label), base_vocab_size) for label in labels]
    relation_anchors = {
        relation: _anchor_ids(tokenizer, relation.replace("script", " script"), base_vocab_size)
        for relation in RELATIONS
    }
    class_tokens = _class_tokens(labels)
    relation_tokens = _relation_tokens()
    added = tokenizer.add_special_tokens({
        "additional_special_tokens": class_tokens + [relation_tokens[name] for name in RELATIONS]
    })
    if added != len(labels) + len(RELATIONS):
        raise ValueError(f"custom token collision: expected {len(labels) + len(RELATIONS)}, added {added}")
    model = BertForMaskedLM.from_pretrained(pretrained, local_files_only=True)
    model.resize_token_embeddings(len(tokenizer))
    class_ids = [tokenizer.convert_tokens_to_ids(token) for token in class_tokens]
    relation_ids = {name: tokenizer.convert_tokens_to_ids(token) for name, token in relation_tokens.items()}
    embeddings = model.get_input_embeddings().weight
    with torch.no_grad():
        for custom_id, source_ids in zip(class_ids, anchors, strict=True):
            embeddings[custom_id].copy_(embeddings[source_ids].mean(dim=0))
        for relation, custom_id in relation_ids.items():
            embeddings[custom_id].copy_(embeddings[relation_anchors[relation]].mean(dim=0))
    model.tie_weights()
    contract = {
        "labels": labels,
        "label_to_index": {label: index for index, label in enumerate(labels)},
        "class_tokens": class_tokens,
        "class_ids": class_ids,
        "relation_tokens": relation_tokens,
        "relation_ids": relation_ids,
        "pad_id": tokenizer.pad_token_id,
        "cls_id": tokenizer.cls_token_id,
        "sep_id": tokenizer.sep_token_id,
        "mask_id": tokenizer.mask_token_id,
        "base_vocab_size": base_vocab_size,
        "expanded_vocab_size": len(tokenizer),
    }
    if None in (contract["pad_id"], contract["cls_id"], contract["sep_id"], contract["mask_id"]):
        raise ValueError("BERT special-token contract is incomplete")
    return model.to(device), tokenizer, contract


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _formulae(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["formula_id"])].append(row)
    for formula_id, sequence in grouped.items():
        sequence.sort(key=lambda row: int(row["context"]["index"]))
        indices = [int(row["context"]["index"]) for row in sequence]
        if indices != list(range(len(sequence))) or any(int(row["context"]["length"]) != len(sequence) for row in sequence):
            raise ValueError(f"invalid formula sequence contract: {formula_id}")
    return dict(grouped)


def _spatial_relation(left: dict, right: dict) -> str:
    a, b = left["geometry"], right["geometry"]
    dx = float(b["center_x"] - a["center_x"])
    dy = float(b["center_y"] - a["center_y"])
    width = max((float(a["width_rel"]) + float(b["width_rel"])) / 2.0, 1e-6)
    height = max((float(a["height_rel"]) + float(b["height_rel"])) / 2.0, 1e-6)
    if abs(dx) <= 0.25 * width and abs(dy) <= 0.25 * height:
        return "overlap"
    if abs(dx) <= 0.60 * width and abs(dy) >= 0.55 * height:
        return "above" if dy < 0.0 else "below"
    if dx >= -0.10 * width and dy <= -0.45 * height:
        return "superscript"
    if dx >= -0.10 * width and dy >= 0.45 * height:
        return "subscript"
    return "right" if dx >= 0.0 else "left"


def _examples(rows: list[dict], contract: dict, context_tokens: dict[str, str] | None = None) -> list[dict]:
    output = []
    label_to_index = contract["label_to_index"]
    for formula_id, sequence in _formulae(rows).items():
        relations = [_spatial_relation(sequence[index - 1], sequence[index]) for index in range(1, len(sequence))]
        for target_index, target in enumerate(sequence):
            ids = [contract["cls_id"]]
            mask_position = -1
            for index, row in enumerate(sequence):
                if index:
                    ids.append(contract["relation_ids"][relations[index - 1]])
                if index == target_index:
                    mask_position = len(ids)
                    ids.append(contract["mask_id"])
                else:
                    token = (context_tokens or {}).get(str(row["record_id"]), row["final_topk"][0])
                    if token not in label_to_index:
                        raise ValueError(f"HWR context token outside 372 classes: {token}")
                    ids.append(contract["class_ids"][label_to_index[token]])
            ids.append(contract["sep_id"])
            if mask_position < 0 or len(ids) > 512:
                raise ValueError(f"masked sequence exceeds BERT contract: {formula_id}")
            target_label = str(target.get("label", target["final_topk"][0]))
            if target_label not in label_to_index:
                raise ValueError(f"masked target outside 372 classes: {target_label}")
            output.append({
                "row": target,
                "input_ids": ids,
                "mask_position": mask_position,
                # Inference rows do not carry truth; this value is unused there.
                "target": label_to_index[target_label],
            })
    if {example["row"]["record_id"] for example in output} != {row["record_id"] for row in rows}:
        raise AssertionError("masked-context example coverage mismatch")
    return output


def _pack(rows: list[dict], contract: dict, context_tokens: dict[str, str] | None = None) -> dict:
    examples = _examples(rows, contract, context_tokens)
    width = max(len(example["input_ids"]) for example in examples)
    input_ids = torch.full((len(examples), width), int(contract["pad_id"]), dtype=torch.long)
    attention_mask = torch.zeros((len(examples), width), dtype=torch.long)
    mask_positions = torch.empty(len(examples), dtype=torch.long)
    targets = torch.empty(len(examples), dtype=torch.long)
    counts = Counter(example["target"] for example in examples)
    weights = torch.empty(len(examples), dtype=torch.float32)
    for index, example in enumerate(examples):
        length = len(example["input_ids"])
        input_ids[index, :length] = torch.tensor(example["input_ids"], dtype=torch.long)
        attention_mask[index, :length] = 1
        mask_positions[index] = example["mask_position"]
        targets[index] = example["target"]
        weights[index] = 1.0 / math.sqrt(counts[example["target"]])
    weights /= weights.mean()
    return {
        "rows": [example["row"] for example in examples],
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "mask_positions": mask_positions,
        "targets": targets,
        "weights": weights,
    }


def _class_logits(model: BertForMaskedLM, input_ids: torch.Tensor, attention_mask: torch.Tensor, mask_positions: torch.Tensor, class_ids: torch.Tensor) -> torch.Tensor:
    hidden = model.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True).last_hidden_state
    masked = hidden[torch.arange(len(hidden), device=hidden.device), mask_positions]
    transformed = model.cls.predictions.transform(masked)
    decoder_weight = model.get_output_embeddings().weight.index_select(0, class_ids)
    decoder_bias = model.cls.predictions.bias.index_select(0, class_ids)
    return F.linear(transformed, decoder_weight, decoder_bias)


def _train_epochs(model: BertForMaskedLM, contract: dict, rows: list[dict], device: torch.device, *, epochs: int, batch_size: int, learning_rate: float, weight_decay: float, seed: int) -> list[float]:
    if epochs == 0:
        return []
    packed = _pack(rows, contract)
    class_ids = torch.tensor(contract["class_ids"], dtype=torch.long, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed)
    losses = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(rows), generator=generator)
        numerator, denominator = 0.0, 0.0
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            input_ids = packed["input_ids"][indices].to(device)
            attention = packed["attention_mask"][indices].to(device)
            mask_positions = packed["mask_positions"][indices].to(device)
            targets = packed["targets"][indices].to(device)
            weights = packed["weights"][indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = _class_logits(model, input_ids, attention, mask_positions, class_ids)
            per_record = F.cross_entropy(logits, targets, reduction="none")
            loss = (per_record * weights).sum() / weights.sum()
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite masked-context training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            numerator += float((per_record.detach() * weights).sum())
            denominator += float(weights.sum())
        losses.append(numerator / denominator)
        _event("context_epoch", epoch=epoch, epochs=epochs, records=len(rows), loss=losses[-1])
    model.eval()
    return losses


@torch.inference_mode()
def _context_log_probabilities(model: BertForMaskedLM, contract: dict, rows: list[dict], device: torch.device, batch_size: int) -> dict[str, np.ndarray]:
    packed = _pack(rows, contract)
    class_ids = torch.tensor(contract["class_ids"], dtype=torch.long, device=device)
    output = {}
    model.eval()
    for start in range(0, len(rows), batch_size):
        stop = min(start + batch_size, len(rows))
        logits = _class_logits(
            model,
            packed["input_ids"][start:stop].to(device),
            packed["attention_mask"][start:stop].to(device),
            packed["mask_positions"][start:stop].to(device),
            class_ids,
        )
        values = logits.log_softmax(dim=1).cpu().numpy()
        for row, scores in zip(packed["rows"][start:stop], values, strict=True):
            output[str(row["record_id"])] = scores
    return output


def _fused_predictions(rows: list[dict], context: dict[str, np.ndarray], labels: list[str], lambda_value: float) -> dict[str, str]:
    label_to_index = {label: index for index, label in enumerate(labels)}
    predictions = {}
    for row in rows:
        candidates = row["final_topk"]
        probabilities = row["final_topk_probabilities"]
        context_scores = context[str(row["record_id"])]
        scores = [
            math.log(max(float(probability), 1e-12)) + lambda_value * float(context_scores[label_to_index[token]])
            for token, probability in zip(candidates, probabilities, strict=True)
        ]
        prediction = candidates[max(range(len(scores)), key=scores.__getitem__)]
        if prediction not in candidates:
            raise AssertionError("context reranker invented a candidate")
        predictions[str(row["record_id"])] = prediction
    return predictions


def load_product_checkpoint(pretrained: Path, checkpoint_path: Path, device: torch.device) -> tuple[BertForMaskedLM, dict, dict]:
    """Load a deployable context checkpoint while revalidating its pinned base."""
    pretrained = _d_path(pretrained, "pretrained model")
    checkpoint_path = _d_path(checkpoint_path, "context checkpoint")
    base_hashes = _verify_pretrained(pretrained)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if payload.get("schema") != SCHEMA or payload.get("model_id") != MODEL_ID or payload.get("model_revision") != MODEL_REVISION:
        raise ValueError("masked-context checkpoint contract mismatch")
    if payload.get("model_file_sha256") != base_hashes:
        raise ValueError("masked-context checkpoint belongs to different BERT-Tiny files")
    labels = list(payload.get("math_labels", []))
    model, _, contract = _build_model(pretrained, labels, device, SEED)
    if payload.get("class_tokens") != contract["class_tokens"] or payload.get("relation_tokens") != contract["relation_tokens"]:
        raise ValueError("masked-context custom-token contract mismatch")
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.eval(), contract, payload


def rerank_formula_rows(model: BertForMaskedLM, contract: dict, checkpoint: dict, rows: list[dict], device: torch.device, batch_size: int = 128) -> dict[str, str]:
    """Rerank complete formula rows without changing their Top-5 candidate sets."""
    context = _context_log_probabilities(model, contract, rows, device, batch_size)
    return _fused_predictions(rows, context, checkpoint["math_labels"], float(checkpoint["lambda"]))


def _selection_key(metrics: dict, lambda_value: float) -> tuple:
    return (
        metrics["all_top1"], metrics["formula_exact"], metrics["strict_macro_top1"],
        metrics["strict_micro_top1"], -metrics["regressed"], -metrics["changed"], -lambda_value,
    )


def _best_lambda(rows: list[dict], context: dict[str, np.ndarray], labels: list[str]) -> tuple[float, dict, list[dict]]:
    trials = []
    for lambda_value in LAMBDA_GRID:
        predictions = _fused_predictions(rows, context, labels, lambda_value)
        metrics = _metrics(rows, predictions)
        trials.append({"lambda": lambda_value, "metrics": metrics})
    best = max(trials, key=lambda trial: _selection_key(trial["metrics"], trial["lambda"]))
    return float(best["lambda"]), best["metrics"], trials


def _inner_split(rows: list[dict], salt: str) -> tuple[list[dict], list[dict]]:
    formula_ids = sorted(_formulae(rows))
    held = {formula_id for formula_id in formula_ids if _fold(formula_id, 5, salt) == 0}
    if not held:
        held = {formula_ids[0]}
    if len(held) == len(formula_ids):
        held = {formula_ids[0]}
    fit = [row for row in rows if row["formula_id"] not in held]
    validation = [row for row in rows if row["formula_id"] in held]
    if {row["formula_id"] for row in fit} & {row["formula_id"] for row in validation}:
        raise AssertionError("formula leakage in context inner split")
    return fit, validation


def _select_epoch_lambda(pretrained: Path, labels: list[str], rows: list[dict], device: torch.device, *, max_epochs: int, patience: int, batch_size: int, predict_batch_size: int, learning_rate: float, weight_decay: float, seed: int, salt: str) -> dict:
    fit, validation = _inner_split(rows, salt)
    model, _, contract = _build_model(pretrained, labels, device, seed)
    packed_fit = _pack(fit, contract)
    class_ids = torch.tensor(contract["class_ids"], dtype=torch.long, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed)
    trials, losses = [], []

    def evaluate(epoch: int) -> tuple:
        context = _context_log_probabilities(model, contract, validation, device, predict_batch_size)
        lambda_value, metrics, lambda_trials = _best_lambda(validation, context, labels)
        trial = {"epoch": epoch, "lambda": lambda_value, "metrics": metrics, "lambda_trials": lambda_trials}
        trials.append(trial)
        return _selection_key(metrics, lambda_value)

    best_key = evaluate(0)
    best_epoch, best_lambda, stale = 0, trials[-1]["lambda"], 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = torch.randperm(len(fit), generator=generator)
        numerator, denominator = 0.0, 0.0
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            input_ids = packed_fit["input_ids"][indices].to(device)
            attention = packed_fit["attention_mask"][indices].to(device)
            mask_positions = packed_fit["mask_positions"][indices].to(device)
            targets = packed_fit["targets"][indices].to(device)
            weights = packed_fit["weights"][indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = _class_logits(model, input_ids, attention, mask_positions, class_ids)
            per_record = F.cross_entropy(logits, targets, reduction="none")
            loss = (per_record * weights).sum() / weights.sum()
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite masked-context selection loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            numerator += float((per_record.detach() * weights).sum())
            denominator += float(weights.sum())
        losses.append(numerator / denominator)
        key = evaluate(epoch)
        current = trials[-1]
        _event(
            "context_selection_epoch", epoch=epoch, fit_records=len(fit), validation_records=len(validation),
            loss=losses[-1], selected_lambda=current["lambda"], validation_top1=current["metrics"]["all_top1"],
        )
        if key > best_key:
            best_key, best_epoch, best_lambda, stale = key, epoch, current["lambda"], 0
        else:
            stale += 1
        if stale >= patience:
            break
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    selected = next(trial for trial in trials if trial["epoch"] == best_epoch and trial["lambda"] == best_lambda)
    return {
        "epochs": best_epoch,
        "lambda": best_lambda,
        "fit_records": len(fit),
        "fit_formulas": len(_formulae(fit)),
        "validation_records": len(validation),
        "validation_formulas": len(_formulae(validation)),
        "validation_metrics": selected["metrics"],
        "searched_epochs": len(losses),
        "losses": losses,
        "trials": trials,
    }


def _fit_model(pretrained: Path, labels: list[str], rows: list[dict], device: torch.device, *, epochs: int, batch_size: int, learning_rate: float, weight_decay: float, seed: int) -> tuple[BertForMaskedLM, dict, list[float]]:
    model, _, contract = _build_model(pretrained, labels, device, seed)
    losses = _train_epochs(
        model, contract, rows, device, epochs=epochs, batch_size=batch_size,
        learning_rate=learning_rate, weight_decay=weight_decay, seed=seed,
    )
    return model, contract, losses


def _writer_loo(pretrained: Path, labels: list[str], rows: list[dict], device: torch.device, args) -> tuple[dict[str, str], list[dict]]:
    predictions, folds = {}, []
    writers = sorted({str(row["writer_group"]) for row in rows})
    for fold_index, writer in enumerate(writers):
        training = [row for row in rows if str(row["writer_group"]) != writer]
        held = [row for row in rows if str(row["writer_group"]) == writer]
        if {row["formula_id"] for row in training} & {row["formula_id"] for row in held}:
            raise AssertionError("formula leakage in writer-LOO context split")
        fold_seed = SEED + 1000 * fold_index
        selection = _select_epoch_lambda(
            pretrained, labels, training, device, max_epochs=args.max_epochs, patience=args.patience,
            batch_size=args.batch_size, predict_batch_size=args.predict_batch_size,
            learning_rate=args.learning_rate, weight_decay=args.weight_decay, seed=fold_seed,
            salt=f"bert-tiny-inner-{writer}",
        )
        model, contract, losses = _fit_model(
            pretrained, labels, training, device, epochs=selection["epochs"], batch_size=args.batch_size,
            learning_rate=args.learning_rate, weight_decay=args.weight_decay, seed=fold_seed,
        )
        context = _context_log_probabilities(model, contract, held, device, args.predict_batch_size)
        fold_predictions = _fused_predictions(held, context, labels, selection["lambda"])
        predictions.update(fold_predictions)
        fold_metrics = _metrics(held, fold_predictions)
        folds.append({
            "held_writer_group": writer,
            "training_formulas": len(_formulae(training)),
            "held_formulas": len(_formulae(held)),
            "selection": selection,
            "refit_losses": losses,
            "held_metrics": fold_metrics,
        })
        _event(
            "context_writer_fold", held_writer=writer, epochs=selection["epochs"],
            lambda_value=selection["lambda"], held_top1=fold_metrics["all_top1"],
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if set(predictions) != {str(row["record_id"]) for row in rows}:
        raise AssertionError("writer-LOO context prediction coverage mismatch")
    return predictions, folds


def _write_prediction_rows(path: Path, rows: list[dict], predictions: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", compresslevel=6, mtime=0) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8", newline="\n") as stream:
                for row in rows:
                    record_id = str(row["record_id"])
                    output = {
                        "record_id": record_id,
                        "formula_id": row["formula_id"],
                        "label": row["label"],
                        "hwr_top1": row["final_topk"][0],
                        "reranked_top1": predictions[record_id],
                        "candidate_preserved": predictions[record_id] in row["final_topk"],
                    }
                    stream.write(json.dumps(output, ensure_ascii=False, separators=(",", ":")) + "\n")


def _candidate_audit(rows: list[dict], predictions: dict[str, str]) -> dict:
    preserved = sum(predictions[str(row["record_id"])] in row["final_topk"] for row in rows)
    return {
        "records": len(rows),
        "candidate_preserved": preserved,
        "candidate_preservation_rate": preserved / len(rows),
        "new_tokens": 0,
        "grouping_mutations": 0,
    }


def _metric_delta(candidate: dict, baseline: dict) -> dict:
    return {
        key: candidate[key] - baseline[key]
        for key in ("all_top1", "strict_micro_top1", "strict_macro_top1", "formula_exact")
    }


def _self_test(pretrained: Path, labels: list[str], device: torch.device) -> None:
    model, tokenizer, contract = _build_model(pretrained, labels, device, SEED)
    geometry = {
        "width_rel": 0.1, "height_rel": 0.2, "center_x": 0.0, "center_y": 0.5,
    }
    rows = []
    for index, (label, topk) in enumerate((
        ("1", ["|", "1", r"\mid", "I", r"\prime"]),
        ("+", ["+", r"\dag", r"\pm", r"\div", r"\Psi"]),
        ("2", ["2", "z", "Z", r"\mathcal{Z}", r"\Sigma"]),
    )):
        row_geometry = dict(geometry, center_x=0.2 * index)
        rows.append({
            "record_id": f"r{index}", "formula_id": "f", "label": label,
            "writer_group": "w", "final_topk": topk,
            "final_topk_probabilities": [0.5, 0.25, 0.12, 0.08, 0.05],
            "geometry": row_geometry, "context": {"index": index, "length": 3},
        })
    examples = _examples(rows, contract)
    assert len(examples) == 3 and all(example["input_ids"].count(contract["mask_id"]) == 1 for example in examples)
    context = _context_log_probabilities(model, contract, rows, device, 3)
    predictions = _fused_predictions(rows, context, labels, 0.0)
    assert predictions == {"r0": "|", "r1": "+", "r2": "2"}
    runtime_rows = [{key: value for key, value in row.items() if key != "label"} for row in rows]
    runtime_predictions = rerank_formula_rows(
        model, contract, {"math_labels": labels, "lambda": 0.0}, runtime_rows, device,
    )
    assert runtime_predictions == predictions
    assert _candidate_audit(rows, predictions)["candidate_preservation_rate"] == 1.0
    assert len(tokenizer) == 30522 + 372 + len(RELATIONS)
    assert sum(parameter.numel() for parameter in model.parameters()) < 5_000_000


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained", type=Path, default=DEFAULT_PRETRAINED)
    parser.add_argument("--hwr-checkpoint", type=Path, default=DEFAULT_PRODUCT)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--logistic-report", type=Path, default=DEFAULT_LOGISTIC_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--predict-batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.max_epochs < 1 or args.patience < 1 or args.batch_size < 1 or args.predict_batch_size < 1:
        parser.error("epoch, patience, and batch sizes must be positive")

    pretrained = _d_path(args.pretrained, "pretrained model")
    hwr_checkpoint = _d_path(args.hwr_checkpoint, "HWR checkpoint")
    input_root = _d_path(args.input, "candidate input")
    output = _d_path(args.output, "output")
    model_hashes = _verify_pretrained(pretrained)
    labels = _labels(hwr_checkpoint)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    if args.self_test:
        _self_test(pretrained, labels, device)
        _event("self_test", status="pass", device=str(device))
        return 0

    report_path = output / "masked_context_report.json"
    checkpoint_path = output / "masked_context_product.pt"
    if report_path.exists() or checkpoint_path.exists():
        parser.error(f"refusing to overwrite existing masked-context output: {output}")
    direct_path = input_root / "direct_candidates.jsonl.gz"
    crohme_path = input_root / "crohme_candidates.jsonl.gz"
    if not direct_path.is_file() or not crohme_path.is_file():
        parser.error("candidate caches missing; run evaluate_homograph_context_reranker_v1.py first")
    direct_rows = list(_json_lines(direct_path))
    crohme_rows = list(_json_lines(crohme_path))
    logistic_report = json.loads(args.logistic_report.read_text(encoding="utf-8"))

    _event("context_training_start", device=str(device), direct_records=len(direct_rows), direct_formulas=len(_formulae(direct_rows)))
    direct_predictions, folds = _writer_loo(pretrained, labels, direct_rows, device, args)
    direct_baseline_predictions = {str(row["record_id"]): row["final_topk"][0] for row in direct_rows}
    direct_baseline = _metrics(direct_rows, direct_baseline_predictions)
    direct_metrics = _metrics(direct_rows, direct_predictions)
    product_epochs = int(statistics.median(fold["selection"]["epochs"] for fold in folds))
    product_lambda = float(statistics.median(fold["selection"]["lambda"] for fold in folds))

    product_model, contract, product_losses = _fit_model(
        pretrained, labels, direct_rows, device, epochs=product_epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate, weight_decay=args.weight_decay, seed=SEED + 9000,
    )
    direct_refit_context = _context_log_probabilities(product_model, contract, direct_rows, device, args.predict_batch_size)
    direct_refit_predictions = _fused_predictions(direct_rows, direct_refit_context, labels, product_lambda)
    crohme_context = _context_log_probabilities(product_model, contract, crohme_rows, device, args.predict_batch_size)
    crohme_predictions = _fused_predictions(crohme_rows, crohme_context, labels, product_lambda)
    crohme_baseline_predictions = {str(row["record_id"]): row["final_topk"][0] for row in crohme_rows}
    crohme_baseline = _metrics(crohme_rows, crohme_baseline_predictions)
    crohme_metrics = _metrics(crohme_rows, crohme_predictions)

    output.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_file_sha256": model_hashes,
        "hwr_checkpoint_sha256": _sha256(hwr_checkpoint),
        "math_labels": labels,
        "class_tokens": contract["class_tokens"],
        "relation_tokens": contract["relation_tokens"],
        "candidate_policy": "log(P_HWR)+lambda*log(P_MLM), argmax limited to original HWR Top-5",
        "lambda": product_lambda,
        "epochs": product_epochs,
        "training_formulae": len(_formulae(direct_rows)),
        "training_records": len(direct_rows),
        "training_data": "project-owned accepted ownership only",
        "state_dict": {key: value.detach().cpu() for key, value in product_model.state_dict().items()},
    }
    torch.save(checkpoint, checkpoint_path)
    checkpoint_sha256 = _sha256(checkpoint_path)
    _write_prediction_rows(output / "direct_writer_loo_predictions.jsonl.gz", direct_rows, direct_predictions)
    _write_prediction_rows(output / "direct_product_refit_predictions.jsonl.gz", direct_rows, direct_refit_predictions)
    _write_prediction_rows(output / "crohme_transfer_predictions.jsonl.gz", crohme_rows, crohme_predictions)

    direct_logistic = logistic_report["direct"]["evaluation"]["variants"]["context"]["metrics"]
    crohme_target_fitted_logistic = logistic_report["crohme"]["evaluation"]["variants"]["context"]["metrics"]
    direct_audit = _candidate_audit(direct_rows, direct_predictions)
    crohme_audit = _candidate_audit(crohme_rows, crohme_predictions)
    direct_label_support = Counter(str(row["label"]) for row in direct_rows)
    strict_labels = set().union(*STRICT_FAMILIES.values())
    missing_strict_labels = sorted(strict_labels - set(direct_label_support))
    reported_metrics = ("all_top1", "strict_micro_top1", "strict_macro_top1", "formula_exact")
    promotion_gate = (
        direct_metrics["all_top1"] > direct_baseline["all_top1"]
        and direct_metrics["regressed"] <= direct_metrics["improved"]
        and all(direct_metrics[key] >= direct_baseline[key] for key in reported_metrics)
        and all(crohme_metrics[key] >= crohme_baseline[key] for key in reported_metrics)
        and not missing_strict_labels
        and direct_audit["candidate_preservation_rate"] == 1.0
        and crohme_audit["candidate_preservation_rate"] == 1.0
    )
    report = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "architecture": {
            "frozen_hwr": "128x5 online-ink Transformer with unified 372-class head",
            "context_encoder": {"model_id": MODEL_ID, "revision": MODEL_REVISION, "license": "Apache-2.0", "layers": 2, "hidden_size": 128, "attention_heads": 2},
            "custom_tokens": {"classes": 372, "spatial_relations": list(RELATIONS)},
            "masked_input": "one target class token replaced by [MASK]; all other glyphs are frozen-HWR Top-1 predictions",
            "fusion": "log(P_HWR)+lambda*log(P_MLM)",
            "candidate_policy": "original HWR Top-5 only; no token invention, deletion, or regrouping",
        },
        "provenance": {
            "pretrained_file_sha256": model_hashes,
            "hwr_checkpoint": str(hwr_checkpoint),
            "hwr_checkpoint_sha256": _sha256(hwr_checkpoint),
            "direct_candidate_sha256": _sha256(direct_path),
            "crohme_candidate_sha256": _sha256(crohme_path),
            "product_checkpoint": str(checkpoint_path),
            "product_checkpoint_sha256": checkpoint_sha256,
            "prediction_files": {
                "direct_writer_loo": "direct_writer_loo_predictions.jsonl.gz",
                "direct_product_refit": "direct_product_refit_predictions.jsonl.gz",
                "crohme_transfer": "crohme_transfer_predictions.jsonl.gz",
            },
        },
        "data_admission": {
            "training": {
                "source": "project-owned accepted ownership", "formulas": len(_formulae(direct_rows)),
                "records": len(direct_rows), "writer_groups": len({row["writer_group"] for row in direct_rows}),
                "target_label_support": dict(sorted(direct_label_support.items())),
                "homograph_labels_without_positive_training_examples": missing_strict_labels,
            },
            "crohme": "evaluation-only CC BY-NC transfer diagnostic; zero training or hyperparameter selection",
            "external_character_10pct": "not applicable to context reranking because rows have no formula sequence or spatial-relation labels",
            "direct_canonical_10": "replay-only whole-formula rows without verified character ownership; excluded from context accuracy",
        },
        "training": {
            "seed": SEED,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "lambda_grid": LAMBDA_GRID,
            "selection": "nested formula-disjoint validation inside each writer-LOO fold; lambda=0 is an explicit abstention baseline",
            "writer_loo_folds": folds,
            "product_refit": {"epochs": product_epochs, "lambda": product_lambda, "losses": product_losses, "aggregation": "median of outer-fold inner-selected settings"},
        },
        "evaluation": {
            "direct_writer_loo": {
                "scope": "47 project-owned formulas, 211 symbols, three held-writer folds",
                "baseline": direct_baseline,
                "masked_context": direct_metrics,
                "delta": _metric_delta(direct_metrics, direct_baseline),
                "candidate_audit": direct_audit,
                "one_hot_context_reference": direct_logistic,
            },
            "direct_product_refit_resubstitution": {
                "warning": "training-set fit only; not acceptance evidence",
                "metrics": _metrics(direct_rows, direct_refit_predictions),
            },
            "crohme_transfer": {
                "scope": "CC BY-NC evaluation only; product refit trained exclusively on project-owned formulae",
                "baseline": crohme_baseline,
                "masked_context": crohme_metrics,
                "delta": _metric_delta(crohme_metrics, crohme_baseline),
                "candidate_audit": crohme_audit,
                "target_fitted_one_hot_diagnostic_not_comparable_product_model": crohme_target_fitted_logistic,
            },
        },
        "decision": {
            "promotion_gate_passed": promotion_gate,
            "runtime_status": "candidate-preserving shadow component" if not promotion_gate else "candidate-preserving product candidate",
            "automatic_default_replacement": False,
            "reason": (
                "homograph macro regression or missing project-owned homograph labels; retain as shadow"
                if not promotion_gate else "new untouched project-owned writer/formula acceptance is still required"
            ),
        },
        "limitations": [
            "Only 47 ownership-verified formulae from three writer groups are available for commercial context training.",
            "The direct folds and CROHME cache were already inspected in earlier experiments and are not untouched final acceptance sets.",
            "Truth grouping is supplied; grouping and final LaTeX/equivalence decoding are outside this reranker evaluation.",
        ],
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    _event(
        "context_complete", report=str(report_path), checkpoint=str(checkpoint_path),
        direct_top1=direct_metrics["all_top1"], crohme_top1=crohme_metrics["all_top1"],
        promotion_gate=promotion_gate,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
