#!/usr/bin/env python3
"""Train a broad-math candidate-validity detector constrained to HWR Top-5.

The detector learns replaced-token validity from an evaluation-disjoint,
Apache-2.0 DeepMind formula corpus.  At inference every existing HWR candidate
is inserted at the target position and scored; the model cannot create tokens,
change stroke grouping, or update the frozen shape classifier.
"""

from __future__ import annotations

from training_data_guard_v1 import (
    assert_training_entrypoint_arguments_clean,
    assert_training_metadata_clean,
    assert_training_path_clean,
    assert_training_rows_clean,
    zero_crohme_training_manifest,
)
if __name__ == "__main__":
    assert_training_entrypoint_arguments_clean()

import argparse
import gc
import gzip
import hashlib
import io
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from character_tensor_v1 import ROOT, _json_lines
from evaluate_homograph_context_reranker_v1 import _metrics
import train_independent_formula_context_v1 as independent
import train_masked_context_reranker_v1 as masked


SCHEMA = "aiflow-candidate-validity-context/v1"
MODEL_FAMILY = "bert-tiny-replaced-token-validity"
SEED = 20260820
BROAD_EPOCHS = (1, 2, 3)
FINETUNE_EPOCHS = (1, 2)
LAMBDA_GRID = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)

DEFAULT_HWR = independent.DEFAULT_HWR
DEFAULT_DIRECT = independent.DEFAULT_DIRECT
DEFAULT_CROHME = independent.DEFAULT_CROHME
DEFAULT_PROMPT = independent.DEFAULT_PROMPT_CORPUS
DEFAULT_PROMPT_AUDIT = independent.DEFAULT_PROMPT_AUDIT
DEFAULT_PRETRAINED = masked.DEFAULT_PRETRAINED
DEFAULT_BROAD_CORPUS = (
    ROOT / "datasets" / "10_approved_external"
    / "deepmind_mathematics_dataset" / "derived"
    / "formula_context_v2.jsonl.gz"
)
DEFAULT_BROAD_AUDIT = DEFAULT_BROAD_CORPUS.with_name("formula_context_v2_audit.json")
DEFAULT_COVERAGE_CORPUS = (
    ROOT / "datasets" / "00_project_owned" / "generated_context"
    / "candidate_validity_coverage_v2.jsonl.gz"
)
DEFAULT_COVERAGE_AUDIT = DEFAULT_COVERAGE_CORPUS.with_name(
    "candidate_validity_coverage_v2_audit.json"
)
DEFAULT_INITIAL_BROAD_CHECKPOINT = (
    ROOT / "artifacts" / "candidate_validity_context_20260820_r1_shadow"
    / "broad_validity_epoch_2.pt"
)
DEFAULT_R2_REPORT = (
    ROOT / "artifacts" / "independent_embedding_typo_context_20260820_r2_shadow"
    / "independent_context_report.json"
)
DEFAULT_R2_DIRECT = DEFAULT_R2_REPORT.with_name("writer_loo_accuracy_predictions.jsonl.gz")
DEFAULT_R2_CROHME = DEFAULT_R2_REPORT.with_name("crohme_accuracy_predictions.jsonl.gz")
DEFAULT_R6_CONTEXT = independent.DEFAULT_R6_CONTEXT
DEFAULT_R6_HWR = independent.DEFAULT_R6_HWR
DEFAULT_OUTPUT = ROOT / "artifacts" / "candidate_validity_context_20260820_r1_shadow"

HOMOGRAPH_FAMILIES = (
    ("0", "O", "o", r"\mathcal{O}", r"\circ"),
    ("1", "|", "l", "/", r"\mathbb{1}"),
    ("x", r"\times", "X", r"\mathcal{X}", r"\chi"),
)
COVERAGE_TOKENS = tuple(token for family in HOMOGRAPH_FAMILIES for token in family)
DIGITS = tuple(str(value) for value in range(10))
VARIABLES = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrsuvwxyz")
BASIC_OPERATORS = ("+", "-", "/", r"\times", "=", "<", ">", r"\neq", r"\leq", r"\geq")
DELIMITERS = ("(", ")", "[", "]", r"\{", r"\}")


def _event(name: str, **values: object) -> None:
    print(json.dumps({"event": name, **values}, ensure_ascii=False), flush=True)


def _seed(salt: str) -> int:
    digest = hashlib.sha256(salt.encode("utf-8")).digest()
    return SEED + int.from_bytes(digest[:4], "big") % 1_000_000


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _d_path(path: Path, label: str, *, file: bool = True) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.drive.upper() != "D:":
        raise ValueError(f"{label} must remain on D: {resolved}")
    if file and not resolved.is_file():
        raise FileNotFoundError(f"missing {label}: {resolved}")
    return resolved


class CandidateValidityModel(nn.Module):
    """BERT encoder plus a target-position binary validity head."""

    def __init__(self, base_mlm: nn.Module):
        super().__init__()
        self.bert = base_mlm.bert
        self.transform = base_mlm.cls.predictions.transform
        hidden = int(base_mlm.config.hidden_size)
        self.validity_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor, target_positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        ).last_hidden_state
        target = hidden[
            torch.arange(len(hidden), device=hidden.device), target_positions
        ]
        return self.validity_head(self.transform(target)).squeeze(1)


def _new_model(
    pretrained: Path, labels: list[str], device: torch.device, seed: int,
) -> tuple[CandidateValidityModel, dict]:
    _set_seed(seed)
    base, _tokenizer, contract = masked._build_model(
        pretrained, labels, device, seed
    )
    model = CandidateValidityModel(base).to(device)
    del base
    return model, contract


def _role(token: str) -> str:
    if token in DIGITS:
        return "digit"
    if token in VARIABLES:
        return "variable"
    if token in BASIC_OPERATORS:
        return "operator"
    if token in DELIMITERS:
        return "delimiter"
    return "symbol"


def _hard_negative(
    truth: str, labels: list[str], rng: random.Random
) -> str:
    allowed = set(labels)
    for family in HOMOGRAPH_FAMILIES:
        if truth in family:
            candidates = [value for value in family if value != truth and value in allowed]
            if candidates:
                return rng.choice(candidates)
    pools = {
        "digit": BASIC_OPERATORS + DELIMITERS,
        "variable": BASIC_OPERATORS + DIGITS,
        "operator": DIGITS + VARIABLES,
        "delimiter": BASIC_OPERATORS + DIGITS,
        "symbol": DIGITS + BASIC_OPERATORS + VARIABLES,
    }
    candidates = [value for value in pools[_role(truth)] if value != truth and value in allowed]
    if not candidates:
        raise ValueError(f"no validity corruption candidate for {truth}")
    return rng.choice(candidates)


def _encode(
    tokens: list[str], relations: list[str], target_index: int,
    candidate: str, contract: dict,
) -> tuple[list[int], list[int], int]:
    if len(relations) != len(tokens) - 1 or not 0 <= target_index < len(tokens):
        raise ValueError("invalid formula validity example")
    label_to_index = contract["label_to_index"]
    if candidate not in label_to_index or any(token not in label_to_index for token in tokens):
        raise ValueError("formula token outside frozen 372 classes")
    ids = [int(contract["cls_id"])]
    token_types = [0]
    target_position = -1
    for index, original in enumerate(tokens):
        if index:
            relation = relations[index - 1]
            if relation not in contract["relation_ids"]:
                raise ValueError(f"unknown spatial relation: {relation}")
            ids.append(int(contract["relation_ids"][relation]))
            token_types.append(0)
        token = candidate if index == target_index else original
        if index == target_index:
            target_position = len(ids)
        ids.append(int(contract["class_ids"][label_to_index[token]]))
        token_types.append(1 if index == target_index else 0)
    ids.append(int(contract["sep_id"]))
    token_types.append(0)
    if target_position < 0 or len(ids) > 512:
        raise ValueError("candidate validity sequence exceeds encoder contract")
    return ids, token_types, target_position


def _batch(
    examples: list[tuple[list[str], list[str], int, str, float]],
    contract: dict,
) -> dict[str, torch.Tensor]:
    encoded = [
        (*_encode(tokens, relations, target, candidate, contract), label)
        for tokens, relations, target, candidate, label in examples
    ]
    width = max(len(value[0]) for value in encoded)
    input_ids = torch.full(
        (len(encoded), width), int(contract["pad_id"]), dtype=torch.long
    )
    attention = torch.zeros((len(encoded), width), dtype=torch.long)
    token_types = torch.zeros((len(encoded), width), dtype=torch.long)
    positions = torch.empty(len(encoded), dtype=torch.long)
    targets = torch.empty(len(encoded), dtype=torch.float32)
    for index, (ids, types, position, label) in enumerate(encoded):
        length = len(ids)
        input_ids[index, :length] = torch.tensor(ids, dtype=torch.long)
        attention[index, :length] = 1
        token_types[index, :length] = torch.tensor(types, dtype=torch.long)
        positions[index] = position
        targets[index] = float(label)
    return {
        "input_ids": input_ids,
        "attention_mask": attention,
        "token_type_ids": token_types,
        "target_positions": positions,
        "targets": targets,
    }


def _forward_batch(
    model: CandidateValidityModel, packed: dict[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    return model(
        packed["input_ids"].to(device),
        packed["attention_mask"].to(device),
        packed["token_type_ids"].to(device),
        packed["target_positions"].to(device),
    )


def _load_broad_corpus(path: Path, labels: list[str]) -> list[dict]:
    allowed = set(labels)
    rows = []
    seen = set()
    for row in _json_lines(path):
        if (
            row.get("schema") != "aiflow-deepmind-formula-context-corpus/v2"
            or row.get("license") != "Apache-2.0"
        ):
            raise ValueError("broad formula corpus contract mismatch")
        tokens = [str(value) for value in row["tokens"]]
        relations = [str(value) for value in row["relations"]]
        sequence = tuple(tokens)
        if (
            not 3 <= len(tokens) <= 64
            or len(relations) != len(tokens) - 1
            or set(tokens) - allowed
            or sequence in seen
        ):
            raise ValueError("invalid broad formula record")
        seen.add(sequence)
        rows.append({
            "formula_id": str(row["formula_id"]),
            "tokens": tokens,
            "relations": relations,
            "source_module": str(row["source_module"]),
        })
    if len(rows) < 50_000:
        raise ValueError(f"broad formula corpus too small: {len(rows)}")
    return rows


def _load_coverage_corpus(path: Path, labels: list[str]) -> list[dict]:
    allowed = set(labels)
    rows = []
    seen = set()
    counts = Counter()
    for row in _json_lines(path):
        if (
            row.get("schema") != "aiflow-candidate-validity-coverage-corpus/v2"
            or row.get("commercial_training_rights")
            != "project-owned generated formula"
        ):
            raise ValueError("coverage formula corpus contract mismatch")
        tokens = [str(value) for value in row["tokens"]]
        relations = [str(value) for value in row["relations"]]
        focus_index = int(row["focus_index"])
        focus_token = str(row["focus_token"])
        formula_id = str(row["formula_id"])
        if (
            formula_id in seen
            or not 3 <= len(tokens) <= 64
            or len(relations) != len(tokens) - 1
            or set(tokens) - allowed
            or focus_token not in COVERAGE_TOKENS
            or not 0 <= focus_index < len(tokens)
            or tokens[focus_index] != focus_token
        ):
            raise ValueError("invalid candidate-validity coverage record")
        seen.add(formula_id)
        counts[focus_token] += 1
        rows.append({
            "formula_id": formula_id,
            "tokens": tokens,
            "relations": relations,
            "focus_index": focus_index,
            "focus_token": focus_token,
            "template": str(row["template"]),
        })
    if set(counts) != set(COVERAGE_TOKENS) or min(counts.values()) < 100:
        raise ValueError(f"incomplete coverage token balance: {dict(counts)}")
    return rows


def _coverage_split(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    fit, validation = [], []
    for row in rows:
        key = f"{row['focus_token']}\0{row['formula_id']}"
        bucket = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) % 10
        (validation if bucket == 0 else fit).append(row)
    fit_tokens = Counter(row["focus_token"] for row in fit)
    validation_tokens = Counter(row["focus_token"] for row in validation)
    if (
        set(fit_tokens) != set(COVERAGE_TOKENS)
        or set(validation_tokens) != set(COVERAGE_TOKENS)
    ):
        raise AssertionError("coverage split omitted a focus token")
    return fit, validation


def _coverage_examples(
    rows: list[dict], labels: list[str], salt: str,
) -> list[tuple[list[str], list[str], int, str, float]]:
    examples = []
    for row in rows:
        truth = row["focus_token"]
        negative = _hard_negative(
            truth, labels,
            random.Random(_seed(f"coverage-{salt}-{row['formula_id']}")),
        )
        examples.append((
            row["tokens"], row["relations"], row["focus_index"], truth, 1.0,
        ))
        examples.append((
            row["tokens"], row["relations"], row["focus_index"], negative, 0.0,
        ))
    return examples


def _coverage_replay_examples(
    rows: list[dict], labels: list[str], per_token: int, salt: str,
) -> list[tuple[list[str], list[str], int, str, float]]:
    if per_token <= 0 or not rows:
        return []
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["focus_token"]].append(row)
    selected = []
    for token in COVERAGE_TOKENS:
        candidates = sorted(
            grouped[token],
            key=lambda row: hashlib.sha256(
                f"coverage-replay-v1\0{row['formula_id']}".encode("utf-8")
            ).hexdigest(),
        )
        if len(candidates) < per_token:
            raise ValueError(
                f"coverage replay shortage for {token}: {len(candidates)}"
            )
        selected.extend(candidates[:per_token])
    return _coverage_examples(selected, labels, f"replay-{salt}")


def _broad_split(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    fit, validation = [], []
    for row in rows:
        bucket = int(hashlib.sha256(row["formula_id"].encode("utf-8")).hexdigest()[:8], 16) % 20
        (validation if bucket == 0 else fit).append(row)
    if not fit or not validation:
        raise AssertionError("invalid broad formula split")
    return fit, validation


def _broad_examples(
    rows: list[dict], labels: list[str], salt: str,
) -> list[tuple[list[str], list[str], int, str, float]]:
    examples = []
    for row in rows:
        rng = random.Random(_seed(f"{salt}-{row['formula_id']}"))
        target = rng.randrange(len(row["tokens"]))
        truth = row["tokens"][target]
        negative = _hard_negative(truth, labels, rng)
        examples.append((row["tokens"], row["relations"], target, truth, 1.0))
        examples.append((row["tokens"], row["relations"], target, negative, 0.0))
    return examples


def _formula_rows(rows: list[dict]) -> dict[str, list[dict]]:
    grouped = masked._formulae(rows)
    return {str(formula_id): sequence for formula_id, sequence in grouped.items()}


def _relations(sequence: list[dict]) -> list[str]:
    return [
        masked._spatial_relation(sequence[index - 1], sequence[index])
        for index in range(1, len(sequence))
    ]


def _direct_examples(
    rows: list[dict], labels: list[str], salt: str,
) -> list[tuple[list[str], list[str], int, str, float]]:
    examples = []
    for formula_id, sequence in _formula_rows(rows).items():
        truth_tokens = [str(row["label"]) for row in sequence]
        runtime_tokens = [str(row["final_topk"][0]) for row in sequence]
        relations = _relations(sequence)
        for target, row in enumerate(sequence):
            truth = str(row["label"])
            wrong = [str(value) for value in row["final_topk"] if str(value) != truth]
            if wrong:
                negative = wrong[0]
            else:
                negative = _hard_negative(
                    truth, labels,
                    random.Random(_seed(f"direct-{salt}-{formula_id}-{target}")),
                )
            for context in (truth_tokens, runtime_tokens):
                examples.append((context, relations, target, truth, 1.0))
                examples.append((context, relations, target, negative, 0.0))
    return examples


def _prompt_examples(
    rows: list[dict], labels: list[str], salt: str,
) -> list[tuple[list[str], list[str], int, str, float]]:
    examples = []
    for formula_id, sequence in _formula_rows(rows).items():
        tokens = [str(row["label"]) for row in sequence]
        relations = _relations(sequence)
        for target, truth in enumerate(tokens):
            rng = random.Random(_seed(f"prompt-{salt}-{formula_id}-{target}"))
            negative = _hard_negative(truth, labels, rng)
            examples.append((tokens, relations, target, truth, 1.0))
            examples.append((tokens, relations, target, negative, 0.0))
    return examples


def _train_one_epoch(
    model: CandidateValidityModel, contract: dict,
    examples: list[tuple[list[str], list[str], int, str, float]],
    optimizer: torch.optim.Optimizer, device: torch.device, batch_size: int,
    seed: int,
) -> float:
    model.train()
    order = list(range(len(examples)))
    random.Random(seed).shuffle(order)
    numerator = 0.0
    for start in range(0, len(order), batch_size):
        selected = [examples[index] for index in order[start:start + batch_size]]
        packed = _batch(selected, contract)
        optimizer.zero_grad(set_to_none=True)
        logits = _forward_batch(model, packed, device)
        targets = packed["targets"].to(device)
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite candidate-validity loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        numerator += float(loss.detach()) * len(selected)
    model.eval()
    return numerator / len(examples)


@torch.inference_mode()
def _validity_metrics(
    model: CandidateValidityModel, contract: dict,
    examples: list[tuple[list[str], list[str], int, str, float]],
    device: torch.device, batch_size: int,
) -> dict:
    model.eval()
    loss_sum = 0.0
    hits = 0
    positives = negatives = positive_hits = negative_hits = 0
    for start in range(0, len(examples), batch_size):
        selected = examples[start:start + batch_size]
        packed = _batch(selected, contract)
        logits = _forward_batch(model, packed, device).cpu()
        targets = packed["targets"]
        loss_sum += float(F.binary_cross_entropy_with_logits(
            logits, targets, reduction="sum"
        ))
        predictions = logits >= 0.0
        truth = targets >= 0.5
        hits += int((predictions == truth).sum())
        positives += int(truth.sum())
        negatives += int((~truth).sum())
        positive_hits += int((predictions & truth).sum())
        negative_hits += int(((~predictions) & (~truth)).sum())
    return {
        "examples": len(examples),
        "loss": loss_sum / len(examples),
        "accuracy": hits / len(examples),
        "positive_recall": positive_hits / positives if positives else None,
        "negative_recall": negative_hits / negatives if negatives else None,
    }


def _state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _pretrain_broad(
    rows: list[dict], labels: list[str], pretrained: Path,
    device: torch.device, args,
) -> tuple[dict[int, dict[str, torch.Tensor]], list[dict]]:
    fit, validation = _broad_split(rows)
    validation_examples = _broad_examples(validation, labels, "broad-validation")
    model, contract = _new_model(pretrained, labels, device, _seed("broad-base"))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.broad_learning_rate,
        weight_decay=args.weight_decay,
    )
    states = {}
    history = []
    for epoch in BROAD_EPOCHS:
        examples = _broad_examples(fit, labels, f"broad-train-{epoch}")
        loss = _train_one_epoch(
            model, contract, examples, optimizer, device,
            args.broad_batch_size, _seed(f"broad-order-{epoch}"),
        )
        validation_metrics = _validity_metrics(
            model, contract, validation_examples, device,
            args.predict_batch_size,
        )
        states[epoch] = _state(model)
        history.append({
            "epoch": epoch,
            "fit_formulas": len(fit),
            "fit_examples": len(examples),
            "train_loss": loss,
            "validation_formulas": len(validation),
            "validation": validation_metrics,
        })
        _event(
            "broad_validity_epoch", epoch=epoch, loss=loss,
            validation_accuracy=validation_metrics["accuracy"],
            validation_loss=validation_metrics["loss"],
        )
        del examples
        gc.collect()
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return states, history


def _load_initial_broad_state(
    path: Path, pretrained: Path, labels: list[str], broad_corpus: Path,
) -> tuple[dict[str, torch.Tensor], dict]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        payload.get("schema") != "aiflow-candidate-validity-broad-base/v1"
        or payload.get("model_family") != MODEL_FAMILY
        or payload.get("model_id") != masked.MODEL_ID
        or payload.get("model_revision") != masked.MODEL_REVISION
        or payload.get("model_file_sha256") != masked._verify_pretrained(pretrained)
        or payload.get("math_labels") != labels
        or payload.get("broad_corpus_sha256") != masked._sha256(broad_corpus)
        or not isinstance(payload.get("state_dict"), dict)
    ):
        raise ValueError("initial broad candidate-validity checkpoint contract failed")
    return payload["state_dict"], {
        key: payload[key]
        for key in (
            "schema", "epoch", "model_family", "model_id", "model_revision",
            "broad_corpus_sha256",
        )
    }


def _finetune_coverage(
    initial_state: dict[str, torch.Tensor], rows: list[dict], labels: list[str],
    pretrained: Path, device: torch.device, args,
) -> tuple[dict[int, dict[str, torch.Tensor]], list[dict]]:
    fit, validation = _coverage_split(rows)
    validation_examples = _coverage_examples(
        validation, labels, "coverage-validation"
    )
    model, contract = _new_from_state(
        initial_state, pretrained, labels, device, _seed("coverage-base")
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.coverage_learning_rate,
        weight_decay=args.weight_decay,
    )
    states = {}
    history = []
    for epoch in BROAD_EPOCHS:
        examples = _coverage_examples(fit, labels, f"coverage-train-{epoch}")
        loss = _train_one_epoch(
            model, contract, examples, optimizer, device,
            args.coverage_batch_size, _seed(f"coverage-order-{epoch}"),
        )
        validation_metrics = _validity_metrics(
            model, contract, validation_examples, device,
            args.predict_batch_size,
        )
        states[epoch] = _state(model)
        history.append({
            "coverage_epoch": epoch,
            "fit_formulas": len(fit),
            "fit_examples": len(examples),
            "train_loss": loss,
            "validation_formulas": len(validation),
            "validation": validation_metrics,
            "fit_by_token": dict(sorted(Counter(
                row["focus_token"] for row in fit
            ).items())),
            "validation_by_token": dict(sorted(Counter(
                row["focus_token"] for row in validation
            ).items())),
        })
        _event(
            "coverage_validity_epoch", epoch=epoch, loss=loss,
            validation_accuracy=validation_metrics["accuracy"],
            positive_recall=validation_metrics["positive_recall"],
            negative_recall=validation_metrics["negative_recall"],
        )
        del examples
        gc.collect()
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return states, history


def _new_from_state(
    state: dict[str, torch.Tensor], pretrained: Path, labels: list[str],
    device: torch.device, seed: int,
) -> tuple[CandidateValidityModel, dict]:
    model, contract = _new_model(pretrained, labels, device, seed)
    model.load_state_dict(state, strict=True)
    return model, contract


def _candidate_examples(
    rows: list[dict],
) -> tuple[
    list[tuple[list[str], list[str], int, str, float]],
    list[tuple[str, str]],
]:
    examples = []
    keys = []
    for _formula_id, sequence in _formula_rows(rows).items():
        runtime_tokens = [str(row["final_topk"][0]) for row in sequence]
        relations = _relations(sequence)
        for target, row in enumerate(sequence):
            record_id = str(row["record_id"])
            for candidate in row["final_topk"]:
                candidate = str(candidate)
                examples.append((runtime_tokens, relations, target, candidate, 0.0))
                keys.append((record_id, candidate))
    return examples, keys


@torch.inference_mode()
def _score_candidates(
    model: CandidateValidityModel, contract: dict, rows: list[dict],
    device: torch.device, batch_size: int,
) -> dict[str, dict[str, float]]:
    examples, keys = _candidate_examples(rows)
    output: dict[str, dict[str, float]] = defaultdict(dict)
    model.eval()
    offset = 0
    for start in range(0, len(examples), batch_size):
        selected = examples[start:start + batch_size]
        packed = _batch(selected, contract)
        logits = _forward_batch(model, packed, device)
        log_validity = F.logsigmoid(logits).cpu().tolist()
        for (record_id, candidate), score in zip(
            keys[offset:offset + len(selected)], log_validity, strict=True
        ):
            output[record_id][candidate] = float(score)
        offset += len(selected)
    if offset != len(keys):
        raise AssertionError("candidate validity score coverage mismatch")
    return dict(output)


def _fused_predictions(
    rows: list[dict], validity: dict[str, dict[str, float]],
    lambda_value: float,
) -> dict[str, str]:
    if not math.isfinite(lambda_value) or lambda_value < 0.0:
        raise ValueError("candidate validity lambda must be finite and non-negative")
    predictions = {}
    for row in rows:
        record_id = str(row["record_id"])
        candidates = [str(value) for value in row["final_topk"]]
        probabilities = [float(value) for value in row["final_topk_probabilities"]]
        if int(row["context"]["length"]) < 2:
            prediction = candidates[0]
        else:
            scores = [
                math.log(max(probability, 1e-12))
                + lambda_value * validity[record_id][candidate]
                for candidate, probability in zip(candidates, probabilities, strict=True)
            ]
            prediction = candidates[max(range(len(scores)), key=scores.__getitem__)]
        if prediction not in candidates:
            raise AssertionError("validity detector invented a candidate")
        predictions[record_id] = prediction
    return predictions


def _compact_metrics(metrics: dict) -> dict:
    return {
        key: metrics[key]
        for key in (
            "all_records", "all_top1", "strict_records", "strict_micro_top1",
            "strict_macro_top1", "changed", "improved", "regressed",
            "formula_exact", "baseline_formula_exact",
        )
    }


def _safe_key(trial: dict) -> tuple:
    metrics = trial["metrics"]
    return (
        trial["admissible"],
        float(metrics["formula_exact"]),
        float(metrics["all_top1"]),
        float(metrics["strict_macro_top1"]),
        float(metrics["strict_micro_top1"]),
        int(metrics["improved"]) - int(metrics["regressed"]),
        -int(metrics["regressed"]),
        -float(trial["lambda"]),
        -int(trial["finetune_epoch"]),
        -int(trial["broad_epoch"]),
    )


def _accuracy_key(trial: dict) -> tuple:
    metrics = trial["metrics"]
    return (
        float(metrics["formula_exact"]),
        float(metrics["all_top1"]),
        float(metrics["strict_macro_top1"]),
        float(metrics["strict_micro_top1"]),
        int(metrics["improved"]) - int(metrics["regressed"]),
        -int(metrics["regressed"]),
        -float(trial["lambda"]),
        -int(trial["finetune_epoch"]),
        -int(trial["broad_epoch"]),
    )


def _selection_fold(
    broad_states: dict[int, dict[str, torch.Tensor]],
    prompt_rows: list[dict], training_rows: list[dict], labels: list[str],
    pretrained: Path, device: torch.device, args, salt: str,
    coverage_replay_rows: list[dict],
) -> dict:
    fit_rows, validation_rows = independent._sequence_group_split(training_rows, salt)
    baseline_predictions = {
        str(row["record_id"]): str(row["final_topk"][0])
        for row in validation_rows
    }
    baseline = _metrics(validation_rows, baseline_predictions)
    trials = []
    training_history = []
    for broad_epoch in BROAD_EPOCHS:
        model, contract = _new_from_state(
            broad_states[broad_epoch], pretrained, labels, device,
            _seed(f"selection-{salt}-broad-{broad_epoch}"),
        )
        examples = (
            _prompt_examples(prompt_rows, labels, f"{salt}-{broad_epoch}")
            + _direct_examples(fit_rows, labels, f"{salt}-{broad_epoch}")
            + _coverage_replay_examples(
                coverage_replay_rows, labels,
                args.coverage_replay_per_token,
                f"selection-{salt}-{broad_epoch}",
            )
        )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.finetune_learning_rate,
            weight_decay=args.weight_decay,
        )
        losses = []
        for finetune_epoch in FINETUNE_EPOCHS:
            losses.append(_train_one_epoch(
                model, contract, examples, optimizer, device,
                args.finetune_batch_size,
                _seed(f"selection-order-{salt}-{broad_epoch}-{finetune_epoch}"),
            ))
            validity = _score_candidates(
                model, contract, validation_rows, device,
                args.predict_batch_size,
            )
            for lambda_value in LAMBDA_GRID:
                predictions = _fused_predictions(
                    validation_rows, validity, lambda_value
                )
                metrics = _metrics(validation_rows, predictions)
                admissible = (
                    metrics["regressed"] == 0
                    and metrics["all_top1"] >= baseline["all_top1"]
                    and metrics["formula_exact"] >= baseline["formula_exact"]
                    and metrics["strict_macro_top1"] >= baseline["strict_macro_top1"]
                    and metrics["strict_micro_top1"] >= baseline["strict_micro_top1"]
                )
                trials.append({
                    "broad_epoch": broad_epoch,
                    "finetune_epoch": finetune_epoch,
                    "lambda": lambda_value,
                    "admissible": admissible,
                    "metrics": _compact_metrics(metrics),
                })
        training_history.append({
            "broad_epoch": broad_epoch,
            "fit_records": len(fit_rows),
            "fit_formulas": len(_formula_rows(fit_rows)),
            "training_examples": len(examples),
            "losses": losses,
        })
        del model, contract, examples, optimizer
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    safe = max(trials, key=_safe_key)
    accuracy = max(trials, key=_accuracy_key)
    return {
        "fit_records": len(fit_rows),
        "fit_formulas": len(_formula_rows(fit_rows)),
        "validation_records": len(validation_rows),
        "validation_formulas": len(_formula_rows(validation_rows)),
        "baseline": _compact_metrics(baseline),
        "safe": safe,
        "accuracy": accuracy,
        "training_history": training_history,
        "trials": trials,
    }


def _fit_configuration(
    broad_states: dict[int, dict[str, torch.Tensor]], configuration: dict,
    prompt_rows: list[dict], training_rows: list[dict], held_rows: list[dict],
    labels: list[str], pretrained: Path, device: torch.device, args, salt: str,
    coverage_replay_rows: list[dict],
) -> tuple[dict[str, str], list[float], dict]:
    broad_epoch = int(configuration["broad_epoch"])
    model, contract = _new_from_state(
        broad_states[broad_epoch], pretrained, labels, device,
        _seed(f"refit-{salt}-{broad_epoch}"),
    )
    examples = (
        _prompt_examples(prompt_rows, labels, f"refit-{salt}-{broad_epoch}")
        + _direct_examples(training_rows, labels, f"refit-{salt}-{broad_epoch}")
        + _coverage_replay_examples(
            coverage_replay_rows, labels, args.coverage_replay_per_token,
            f"refit-{salt}-{broad_epoch}",
        )
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.finetune_learning_rate,
        weight_decay=args.weight_decay,
    )
    losses = []
    for epoch in range(1, int(configuration["finetune_epoch"]) + 1):
        losses.append(_train_one_epoch(
            model, contract, examples, optimizer, device,
            args.finetune_batch_size, _seed(f"refit-order-{salt}-{epoch}"),
        ))
    started = time.perf_counter()
    validity = _score_candidates(
        model, contract, held_rows, device, args.predict_batch_size
    )
    predictions = _fused_predictions(
        held_rows, validity, float(configuration["lambda"])
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    runtime = {
        "records": len(held_rows),
        "candidate_evaluations": sum(len(row["final_topk"]) for row in held_rows),
        "elapsed_ms": elapsed_ms,
        "milliseconds_per_record": elapsed_ms / len(held_rows),
    }
    del model, contract, examples, optimizer, validity
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return predictions, losses, runtime


def _writer_loo(
    broad_states: dict[int, dict[str, torch.Tensor]], prompt_rows: list[dict],
    rows: list[dict], labels: list[str], pretrained: Path,
    device: torch.device, args, coverage_replay_rows: list[dict],
) -> tuple[dict[str, dict[str, str]], list[dict]]:
    predictions = {"safe": {}, "accuracy": {}}
    folds = []
    writers = sorted({str(row["writer_group"]) for row in rows})
    if len(writers) < 2:
        raise ValueError("candidate validity evaluation requires multiple writers")
    for fold_index, writer in enumerate(writers, start=1):
        raw_training = [row for row in rows if str(row["writer_group"]) != writer]
        held = [row for row in rows if str(row["writer_group"]) == writer]
        held_sequences = independent._formula_sequences(held)
        training_sequence_map = independent._formula_sequence_map(raw_training)
        excluded_formulae = {
            formula_id for formula_id, sequence in training_sequence_map.items()
            if sequence in held_sequences
        }
        training = [
            row for row in raw_training
            if str(row["formula_id"]) not in excluded_formulae
        ]
        if independent._formula_sequences(training) & held_sequences:
            raise AssertionError("outer token-sequence leakage in validity model")
        if {str(row["formula_id"]) for row in training} & {
            str(row["formula_id"]) for row in held
        }:
            raise AssertionError("outer formula leakage in validity model")
        selection = _selection_fold(
            broad_states, prompt_rows, training, labels, pretrained, device,
            args, f"writer-{writer}", coverage_replay_rows,
        )
        configurations = {
            name: {
                key: selection[name][key]
                for key in ("broad_epoch", "finetune_epoch", "lambda")
            }
            for name in ("safe", "accuracy")
        }
        cache = {}
        held_report = {}
        for name, configuration in configurations.items():
            key = tuple(configuration[value] for value in (
                "broad_epoch", "finetune_epoch", "lambda"
            ))
            if key not in cache:
                cache[key] = _fit_configuration(
                    broad_states, configuration, prompt_rows, training, held,
                    labels, pretrained, device, args,
                    f"writer-{writer}-{name}", coverage_replay_rows,
                )
            result, losses, runtime = cache[key]
            predictions[name].update(result)
            audit = masked._candidate_audit(held, result)
            if audit["new_tokens"] or audit["grouping_mutations"]:
                raise AssertionError("validity writer-LOO candidate contract failed")
            held_report[name] = {
                "configuration": configuration,
                "metrics": _metrics(held, result),
                "candidate_audit": audit,
                "refit_losses": losses,
                "runtime": runtime,
            }
        fold_report = {
            "held_writer_group": writer,
            "raw_training_records": len(raw_training),
            "sequence_overlap_formulas_excluded": len(excluded_formulae),
            "training_records": len(training),
            "training_formulas": len(_formula_rows(training)),
            "held_records": len(held),
            "held_formulas": len(_formula_rows(held)),
            "selection": selection,
            "held": held_report,
        }
        folds.append(fold_report)
        _event(
            "validity_writer_fold", fold=fold_index, writer=writer,
            safe_top1=held_report["safe"]["metrics"]["all_top1"],
            accuracy_top1=held_report["accuracy"]["metrics"]["all_top1"],
        )
    expected = {str(row["record_id"]) for row in rows}
    if any(set(values) != expected for values in predictions.values()):
        raise AssertionError("validity writer-LOO coverage mismatch")
    return predictions, folds


def _aggregate_configuration(folds: list[dict], name: str) -> dict:
    selected = [fold["held"][name]["configuration"] for fold in folds]
    return {
        "broad_epoch": int(median(int(value["broad_epoch"]) for value in selected)),
        "finetune_epoch": int(median(int(value["finetune_epoch"]) for value in selected)),
        "lambda": float(median(float(value["lambda"]) for value in selected)),
    }


def _fit_products(
    broad_states: dict[int, dict[str, torch.Tensor]], configurations: dict,
    prompt_rows: list[dict], direct_rows: list[dict], labels: list[str],
    pretrained: Path, device: torch.device, args,
    coverage_replay_rows: list[dict],
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, list[float]]]:
    states = {}
    losses = {}
    cache = {}
    for name, configuration in configurations.items():
        key = (int(configuration["broad_epoch"]), int(configuration["finetune_epoch"]))
        if key not in cache:
            broad_epoch, epochs = key
            model, contract = _new_from_state(
                broad_states[broad_epoch], pretrained, labels, device,
                _seed(f"product-{broad_epoch}-{epochs}"),
            )
            examples = (
                _prompt_examples(prompt_rows, labels, f"product-{broad_epoch}")
                + _direct_examples(direct_rows, labels, f"product-{broad_epoch}")
                + _coverage_replay_examples(
                    coverage_replay_rows, labels,
                    args.coverage_replay_per_token,
                    f"product-{broad_epoch}",
                )
            )
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.finetune_learning_rate,
                weight_decay=args.weight_decay,
            )
            fit_losses = []
            for epoch in range(1, epochs + 1):
                fit_losses.append(_train_one_epoch(
                    model, contract, examples, optimizer, device,
                    args.finetune_batch_size,
                    _seed(f"product-order-{broad_epoch}-{epoch}"),
                ))
            cache[key] = (_state(model), fit_losses)
            del model, contract, examples, optimizer
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        states[name] = cache[key][0]
        losses[name] = cache[key][1]
    return states, losses


def _predict_state(
    state: dict[str, torch.Tensor], configuration: dict, rows: list[dict],
    pretrained: Path, labels: list[str], device: torch.device, batch_size: int,
) -> tuple[dict[str, str], dict]:
    model, contract = _new_from_state(
        state, pretrained, labels, device, _seed("product-predict")
    )
    started = time.perf_counter()
    validity = _score_candidates(model, contract, rows, device, batch_size)
    predictions = _fused_predictions(rows, validity, float(configuration["lambda"]))
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    runtime = {
        "records": len(rows),
        "candidate_evaluations": sum(len(row["final_topk"]) for row in rows),
        "elapsed_ms": elapsed_ms,
        "milliseconds_per_record": elapsed_ms / len(rows),
    }
    del model, contract, validity
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return predictions, runtime


def _prediction_file(path: Path, rows: list[dict]) -> dict[str, str]:
    output = {}
    for row in _json_lines(path):
        record_id = str(row["record_id"])
        prediction = str(row["reranked_top1"])
        if record_id in output:
            raise ValueError(f"duplicate comparison prediction: {record_id}")
        output[record_id] = prediction
    expected = {str(row["record_id"]) for row in rows}
    if set(output) != expected:
        raise ValueError("comparison prediction coverage mismatch")
    return output


def _subset_metrics(
    rows: list[dict], predictions: dict[str, str], partition: str,
) -> dict:
    selected = [
        row for row in rows
        if str(row.get("evaluation_partition")) == partition
    ]
    return _metrics(selected, predictions)


def _checkpoint_payload(
    state: dict[str, torch.Tensor], configuration: dict, labels: list[str],
    pretrained: Path, hwr: Path, broad_corpus: Path, broad_audit: Path,
    prompt: Path, direct: Path, training: dict,
) -> dict:
    return {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_family": MODEL_FAMILY,
        "model_id": masked.MODEL_ID,
        "model_revision": masked.MODEL_REVISION,
        "model_file_sha256": masked._verify_pretrained(pretrained),
        "pretrained_license": "Apache-2.0",
        "math_labels": labels,
        "class_tokens": masked._class_tokens(labels),
        "relation_tokens": masked._relation_tokens(),
        "configuration": configuration,
        "hwr_checkpoint_sha256": masked._sha256(hwr),
        "broad_corpus_sha256": masked._sha256(broad_corpus),
        "broad_audit_sha256": masked._sha256(broad_audit),
        "prompt_corpus_sha256": masked._sha256(prompt),
        "direct_candidates_sha256": masked._sha256(direct),
        "candidate_policy": (
            "insert each frozen HWR Top-5 candidate, score log-sigmoid validity, "
            "then argmax log(P_HWR)+lambda*log(P_valid); singleton locks HWR Top-1"
        ),
        "shape_training": False,
        "parent_aiflow_context_checkpoint": None,
        "training": training,
        "state_dict": state,
    }


def load_candidate_validity_context(
    pretrained: Path, checkpoint: Path, hwr: Path, device: torch.device,
) -> tuple[CandidateValidityModel, dict, dict]:
    pretrained = _d_path(pretrained, "pinned pretrained model", file=False)
    checkpoint = _d_path(checkpoint, "candidate validity checkpoint")
    hwr = _d_path(hwr, "HWR checkpoint")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    labels = masked._labels(hwr)
    configuration = payload.get("configuration")
    if (
        payload.get("schema") != SCHEMA
        or payload.get("model_family") != MODEL_FAMILY
        or payload.get("model_id") != masked.MODEL_ID
        or payload.get("model_revision") != masked.MODEL_REVISION
        or payload.get("model_file_sha256") != masked._verify_pretrained(pretrained)
        or payload.get("pretrained_license") != "Apache-2.0"
        or payload.get("math_labels") != labels
        or payload.get("hwr_checkpoint_sha256") != masked._sha256(hwr)
        or payload.get("shape_training") is not False
        or payload.get("parent_aiflow_context_checkpoint") is not None
        or not isinstance(configuration, dict)
        or set(configuration) != {"broad_epoch", "finetune_epoch", "lambda"}
        or int(configuration["broad_epoch"]) not in BROAD_EPOCHS
        or int(configuration["finetune_epoch"]) not in FINETUNE_EPOCHS
        or float(configuration["lambda"]) < 0.0
    ):
        raise ValueError("candidate validity checkpoint contract mismatch")
    model, contract = _new_model(pretrained, labels, device, SEED)
    if (
        payload.get("class_tokens") != contract["class_tokens"]
        or payload.get("relation_tokens") != contract["relation_tokens"]
    ):
        raise ValueError("candidate validity token contract mismatch")
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, contract, payload


def decide_candidate_validity_rows(
    model: CandidateValidityModel, contract: dict, payload: dict,
    rows: list[dict], device: torch.device, batch_size: int = 256,
) -> tuple[dict[str, str], dict]:
    validity = _score_candidates(model, contract, rows, device, batch_size)
    predictions = _fused_predictions(
        rows, validity, float(payload["configuration"]["lambda"])
    )
    audit = masked._candidate_audit(rows, predictions)
    if audit["new_tokens"] or audit["grouping_mutations"]:
        raise AssertionError("loaded candidate validity model violated Top-5")
    return predictions, audit


def _self_test(
    pretrained: Path, labels: list[str], device: torch.device,
) -> None:
    model, contract = _new_model(pretrained, labels, device, SEED)
    tokens = ["1", "+", "2"]
    relations = ["right", "right"]
    negative = _hard_negative("1", labels, random.Random(SEED))
    examples = [
        (tokens, relations, 0, "1", 1.0),
        (tokens, relations, 0, negative, 0.0),
    ]
    packed = _batch(examples, contract)
    logits = _forward_batch(model, packed, device)
    assert logits.shape == (2,) and torch.isfinite(logits).all()
    rows = [
        {
            "record_id": "a", "formula_id": "f", "writer_group": "w",
            "label": "1", "final_topk": ["|", "1"],
            "final_topk_probabilities": [0.7, 0.3],
            "geometry": {
                "center_x": 0.2, "center_y": 0.5,
                "width_rel": 0.3, "height_rel": 1.0,
            },
            "context": {"index": 0, "length": 3},
        },
        {
            "record_id": "b", "formula_id": "f", "writer_group": "w",
            "label": "+", "final_topk": ["+", "1"],
            "final_topk_probabilities": [0.8, 0.2],
            "geometry": {
                "center_x": 0.5, "center_y": 0.5,
                "width_rel": 0.3, "height_rel": 1.0,
            },
            "context": {"index": 1, "length": 3},
        },
        {
            "record_id": "c", "formula_id": "f", "writer_group": "w",
            "label": "2", "final_topk": ["2", "z"],
            "final_topk_probabilities": [0.9, 0.1],
            "geometry": {
                "center_x": 0.8, "center_y": 0.5,
                "width_rel": 0.3, "height_rel": 1.0,
            },
            "context": {"index": 2, "length": 3},
        },
    ]
    validity = _score_candidates(model, contract, rows, device, 8)
    predictions = _fused_predictions(rows, validity, 0.0)
    assert predictions == {"a": "|", "b": "+", "c": "2"}
    audit = masked._candidate_audit(rows, predictions)
    assert audit["new_tokens"] == 0 and audit["grouping_mutations"] == 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hwr-checkpoint", type=Path, default=DEFAULT_HWR)
    parser.add_argument("--direct-candidates", type=Path, default=DEFAULT_DIRECT)
    parser.add_argument("--crohme-candidates", type=Path, default=DEFAULT_CROHME)
    parser.add_argument("--prompt-corpus", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--prompt-audit", type=Path, default=DEFAULT_PROMPT_AUDIT)
    parser.add_argument("--broad-corpus", type=Path, default=DEFAULT_BROAD_CORPUS)
    parser.add_argument("--broad-audit", type=Path, default=DEFAULT_BROAD_AUDIT)
    parser.add_argument("--coverage-finetune", action="store_true")
    parser.add_argument(
        "--coverage-corpus", type=Path, default=DEFAULT_COVERAGE_CORPUS
    )
    parser.add_argument(
        "--coverage-audit", type=Path, default=DEFAULT_COVERAGE_AUDIT
    )
    parser.add_argument(
        "--initial-broad-checkpoint", type=Path,
        default=DEFAULT_INITIAL_BROAD_CHECKPOINT,
    )
    parser.add_argument("--pretrained", type=Path, default=DEFAULT_PRETRAINED)
    parser.add_argument("--r2-report", type=Path, default=DEFAULT_R2_REPORT)
    parser.add_argument("--r2-direct-predictions", type=Path, default=DEFAULT_R2_DIRECT)
    parser.add_argument("--r2-crohme-predictions", type=Path, default=DEFAULT_R2_CROHME)
    parser.add_argument("--r6-context", type=Path, default=DEFAULT_R6_CONTEXT)
    parser.add_argument("--r6-hwr", type=Path, default=DEFAULT_R6_HWR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--broad-batch-size", type=int, default=256)
    parser.add_argument("--coverage-batch-size", type=int, default=128)
    parser.add_argument("--coverage-replay-per-token", type=int, default=0)
    parser.add_argument("--finetune-batch-size", type=int, default=128)
    parser.add_argument("--predict-batch-size", type=int, default=256)
    parser.add_argument("--broad-learning-rate", type=float, default=3e-5)
    parser.add_argument("--coverage-learning-rate", type=float, default=5e-5)
    parser.add_argument("--finetune-learning-rate", type=float, default=8e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if (
        min(
            args.broad_batch_size, args.coverage_batch_size,
            args.finetune_batch_size, args.predict_batch_size,
        ) < 1
        or args.broad_learning_rate <= 0.0
        or args.coverage_learning_rate <= 0.0
        or args.finetune_learning_rate <= 0.0
        or args.weight_decay < 0.0
        or args.coverage_replay_per_token < 0
    ):
        parser.error("batch sizes and learning rates must be positive")
    if args.coverage_replay_per_token and not args.coverage_finetune:
        parser.error("coverage replay requires --coverage-finetune")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")

    pretrained = _d_path(args.pretrained, "pinned BERT-Tiny", file=False)
    pretrained_hashes = masked._verify_pretrained(pretrained)
    hwr = _d_path(args.hwr_checkpoint, "HWR checkpoint")
    labels = masked._labels(hwr)
    if args.self_test:
        _self_test(pretrained, labels, device)
        _event("self_test", status="pass", device=str(device))
        return 0

    direct_path = _d_path(args.direct_candidates, "direct candidates")
    crohme_path = _d_path(args.crohme_candidates, "CROHME candidates")
    prompt_path = _d_path(args.prompt_corpus, "prompt corpus")
    prompt_audit_path = _d_path(args.prompt_audit, "prompt audit")
    broad_path = _d_path(args.broad_corpus, "broad formula corpus")
    broad_audit_path = _d_path(args.broad_audit, "broad formula audit")
    coverage_path = (
        _d_path(args.coverage_corpus, "coverage formula corpus")
        if args.coverage_finetune else None
    )
    coverage_audit_path = (
        _d_path(args.coverage_audit, "coverage formula audit")
        if args.coverage_finetune else None
    )
    initial_broad_path = (
        _d_path(args.initial_broad_checkpoint, "initial broad checkpoint")
        if args.coverage_finetune else None
    )
    r2_report_path = _d_path(args.r2_report, "r2 report")
    r2_direct_path = _d_path(args.r2_direct_predictions, "r2 direct predictions")
    r2_crohme_path = _d_path(args.r2_crohme_predictions, "r2 CROHME predictions")
    r6_context = _d_path(args.r6_context, "r6 context checkpoint")
    r6_hwr = _d_path(args.r6_hwr, "r6 HWR checkpoint")
    output = _d_path(args.output, "candidate validity output", file=False)
    if output.exists():
        parser.error(f"refusing to overwrite candidate validity output: {output}")

    training_immutable_paths = (
        hwr, direct_path, prompt_path, prompt_audit_path,
        broad_path, broad_audit_path,
    ) + tuple(pretrained / name for name in masked.EXPECTED_MODEL_FILES)
    if args.coverage_finetune:
        training_immutable_paths += (
            coverage_path, coverage_audit_path, initial_broad_path,
        )
    report_only_immutable_paths = (
        crohme_path, r2_report_path, r2_direct_path, r2_crohme_path,
        r6_context, r6_hwr,
    )
    immutable_paths = training_immutable_paths + report_only_immutable_paths
    hashes_before = {str(path): masked._sha256(path) for path in immutable_paths}
    training_hashes_before = {
        str(path): masked._sha256(path) for path in training_immutable_paths
    }

    broad_audit = json.loads(broad_audit_path.read_text(encoding="utf-8"))
    assert_training_metadata_clean(broad_audit, label="commercial broad-corpus audit")
    if (
        broad_audit.get("schema") != "aiflow-deepmind-formula-context-audit/v2"
        or broad_audit.get("source", {}).get("license") != "Apache-2.0"
        or broad_audit.get("source", {}).get("revision")
        != "427f45075f84b8b9774950196ad63867ca20ffb3"
        or broad_audit.get("contracts", {}).get("commercial_training_rights") is not True
        or broad_audit.get("contracts", {}).get("project_owned_dev_exact_sequence_overlap") != 0
        or broad_audit.get("contracts", {}).get("crohme_used_for_generation_or_filtering") is not False
        or int(broad_audit.get("generation", {}).get("unique_formulas", 0)) < 50_000
        or broad_audit.get("artifacts", {}).get("corpus_sha256") != masked._sha256(broad_path)
    ):
        raise ValueError("broad formula corpus audit contract failed")
    prompt_audit = json.loads(prompt_audit_path.read_text(encoding="utf-8"))
    assert_training_metadata_clean(prompt_audit, label="project-owned prompt audit")
    if (
        prompt_audit.get("schema") != "aiflow-owned-prompt-context-audit/v1"
        or prompt_audit.get("contracts", {}).get("commercial_training_rights") is not True
    ):
        raise ValueError("project prompt corpus rights contract failed")

    direct_rows = list(_json_lines(direct_path))
    crohme_rows = list(_json_lines(crohme_path))
    if not direct_rows or not crohme_rows:
        raise ValueError("candidate evaluation cache is empty")
    assert_training_path_clean(direct_path, "project-owned candidate selection rows")
    assert_training_path_clean(prompt_path, "project-owned prompt training corpus")
    assert_training_path_clean(broad_path, "commercial broad formula training corpus")
    assert_training_rows_clean(direct_rows)
    training_exclusion_sequences = independent._formula_sequences(direct_rows)
    prompt_rows, prompt_admission = independent._prompt_rows(
        prompt_path, labels, training_exclusion_sequences
    )
    assert_training_rows_clean(prompt_rows)
    if independent._formula_sequences(prompt_rows) & training_exclusion_sequences:
        raise AssertionError("prompt corpus overlaps a project-owned dev token sequence")
    broad_formula_count = int(
        broad_audit.get("generation", {}).get("unique_formulas", 0)
    )
    coverage_audit = None
    coverage_rows: list[dict] = []
    coverage_replay_rows: list[dict] = []
    initial_broad_metadata = None
    if args.coverage_finetune:
        assert_training_path_clean(coverage_path, "project-owned coverage training corpus")
        coverage_audit = json.loads(
            coverage_audit_path.read_text(encoding="utf-8")
        )
        assert_training_metadata_clean(
            coverage_audit, label="project-owned coverage audit"
        )
        if (
            coverage_audit.get("schema")
            != "aiflow-candidate-validity-coverage-audit/v2"
            or coverage_audit.get("contracts", {}).get(
                "commercial_training_rights"
            ) is not True
            or coverage_audit.get("contracts", {}).get(
                "project_owned_generated"
            ) is not True
            or coverage_audit.get("contracts", {}).get(
                "project_owned_dev_exact_sequence_overlap"
            ) != 0
            or coverage_audit.get("contracts", {}).get(
                "crohme_used_for_generation_or_filtering"
            ) is not False
            or coverage_audit.get("artifacts", {}).get("corpus_sha256")
            != masked._sha256(coverage_path)
        ):
            raise ValueError("coverage formula corpus audit contract failed")
        coverage_rows = _load_coverage_corpus(coverage_path, labels)
        assert_training_rows_clean(coverage_rows)
        if any(
            tuple(row["tokens"]) in training_exclusion_sequences
            for row in coverage_rows
        ):
            raise AssertionError(
                "coverage corpus overlaps a project-owned dev token sequence"
            )
        initial_state, initial_broad_metadata = _load_initial_broad_state(
            initial_broad_path, pretrained, labels, broad_path
        )
        broad_states, broad_history = _finetune_coverage(
            initial_state, coverage_rows, labels, pretrained, device, args
        )
        coverage_replay_rows, _coverage_validation_rows = _coverage_split(
            coverage_rows
        )
        del initial_state
        pretraining_mode = "coverage_finetune_from_frozen_broad_epoch_2"
    else:
        broad_rows = _load_broad_corpus(broad_path, labels)
        assert_training_rows_clean(broad_rows)
        if any(
            tuple(row["tokens"]) in training_exclusion_sequences
            for row in broad_rows
        ):
            raise AssertionError(
                "broad corpus overlaps a project-owned dev token sequence"
            )
        broad_states, broad_history = _pretrain_broad(
            broad_rows, labels, pretrained, device, args
        )
        pretraining_mode = "broad_formula_pretraining"
    oof_predictions, folds = _writer_loo(
        broad_states, prompt_rows, direct_rows, labels, pretrained,
        device, args, coverage_replay_rows,
    )
    oof_evaluation = {}
    for name, predictions in oof_predictions.items():
        audit = masked._candidate_audit(direct_rows, predictions)
        if audit["new_tokens"] or audit["grouping_mutations"]:
            raise AssertionError("validity OOF candidate contract failed")
        oof_evaluation[name] = {
            "all": _metrics(direct_rows, predictions),
            "legacy47": _subset_metrics(direct_rows, predictions, "legacy47"),
            "new_known_writer": _subset_metrics(
                direct_rows, predictions, "new_known_writer"
            ),
            "new_writer": _subset_metrics(direct_rows, predictions, "new_writer"),
            "candidate_audit": audit,
        }
    configurations = {
        name: _aggregate_configuration(folds, name)
        for name in ("safe", "accuracy")
    }
    product_states, product_losses = _fit_products(
        broad_states, configurations, prompt_rows, direct_rows, labels,
        pretrained, device, args, coverage_replay_rows,
    )
    product_predictions = {}
    product_evaluation = {}
    for name, state in product_states.items():
        direct_predictions, direct_runtime = _predict_state(
            state, configurations[name], direct_rows, pretrained, labels,
            device, args.predict_batch_size,
        )
        crohme_predictions, crohme_runtime = _predict_state(
            state, configurations[name], crohme_rows, pretrained, labels,
            device, args.predict_batch_size,
        )
        product_predictions[name] = {
            "direct": direct_predictions,
            "crohme": crohme_predictions,
        }
        product_evaluation[name] = {
            "direct_refit_warning": "all direct rows are product fit input",
            "direct_refit": _metrics(direct_rows, direct_predictions),
            "direct_candidate_audit": masked._candidate_audit(
                direct_rows, direct_predictions
            ),
            "direct_runtime": direct_runtime,
            "crohme_selection_role": "none; post-selection research diagnostic only",
            "crohme": _metrics(crohme_rows, crohme_predictions),
            "crohme_candidate_audit": masked._candidate_audit(
                crohme_rows, crohme_predictions
            ),
            "crohme_runtime": crohme_runtime,
        }

    # Comparison models are read only after every candidate-validity fit.
    r2_direct_predictions = _prediction_file(r2_direct_path, direct_rows)
    r2_crohme_predictions = _prediction_file(r2_crohme_path, crohme_rows)
    r2_report = json.loads(r2_report_path.read_text(encoding="utf-8"))
    r2_comparison = {
        "loaded_after_validity_training": True,
        "report_sha256": masked._sha256(r2_report_path),
        "direct_writer_loo": {
            "all": _metrics(direct_rows, r2_direct_predictions),
            "new_writer": _subset_metrics(
                direct_rows, r2_direct_predictions, "new_writer"
            ),
        },
        "crohme_post_selection": _metrics(crohme_rows, r2_crohme_predictions),
        "reported_runtime_status": r2_report.get("decision", {}).get("runtime_status"),
    }
    r6_comparison = independent._r6_comparison(
        r6_context, r6_hwr, direct_rows, crohme_rows, device,
        args.predict_batch_size,
    )

    hashes_after = {str(path): masked._sha256(path) for path in immutable_paths}
    training_hashes_after = {
        str(path): masked._sha256(path) for path in training_immutable_paths
    }
    if hashes_after != hashes_before:
        raise AssertionError("candidate validity training changed a frozen input")

    output.mkdir(parents=True, exist_ok=False)
    broad_checkpoints = {}
    for epoch, state in broad_states.items():
        checkpoint_kind = "coverage" if args.coverage_finetune else "broad"
        path = output / f"{checkpoint_kind}_validity_epoch_{epoch}.pt"
        state_payload = {
            "schema": (
                "aiflow-candidate-validity-coverage-base/v1"
                if args.coverage_finetune else
                "aiflow-candidate-validity-broad-base/v1"
            ),
            "epoch": epoch,
            "model_family": MODEL_FAMILY,
            "model_id": masked.MODEL_ID,
            "model_revision": masked.MODEL_REVISION,
            "model_file_sha256": pretrained_hashes,
            "math_labels": labels,
            "broad_corpus_sha256": masked._sha256(broad_path),
            "state_dict": state,
        }
        if args.coverage_finetune:
            state_payload.update({
                "initial_broad_checkpoint_sha256": masked._sha256(
                    initial_broad_path
                ),
                "coverage_corpus_sha256": masked._sha256(coverage_path),
                "coverage_audit_sha256": masked._sha256(coverage_audit_path),
            })
        torch.save(state_payload, path)
        broad_checkpoints[str(epoch)] = {
            "path": str(path), "sha256": masked._sha256(path)
        }

    checkpoint_reports = {}
    training_contract = {
        "pretraining_mode": pretraining_mode,
        "broad_formula_count": broad_formula_count,
        "state_variant_epochs": list(BROAD_EPOCHS),
        "configuration_field_note": (
            "legacy broad_epoch field denotes coverage_epoch in coverage mode"
            if args.coverage_finetune else
            "broad_epoch denotes broad pretraining epoch"
        ),
        "finetune_epoch_grid": list(FINETUNE_EPOCHS),
        "lambda_grid": list(LAMBDA_GRID),
        "broad_learning_rate": args.broad_learning_rate,
        "coverage_learning_rate": (
            args.coverage_learning_rate if args.coverage_finetune else None
        ),
        "coverage_formula_count": len(coverage_rows),
        "coverage_replay_per_token": args.coverage_replay_per_token,
        "coverage_replay_formula_count": (
            len(COVERAGE_TOKENS) * args.coverage_replay_per_token
        ),
        "coverage_corpus_sha256": (
            masked._sha256(coverage_path) if args.coverage_finetune else None
        ),
        "coverage_audit_sha256": (
            masked._sha256(coverage_audit_path) if args.coverage_finetune else None
        ),
        "initial_broad_checkpoint_sha256": (
            masked._sha256(initial_broad_path)
            if args.coverage_finetune else None
        ),
        "finetune_learning_rate": args.finetune_learning_rate,
        "weight_decay": args.weight_decay,
        "negative_policy": "homograph-first then role-incompatible replacement",
        "direct_contexts": ["truth-clean", "runtime-HWR-Top1"],
        "product_losses": product_losses,
    }
    for name, state in product_states.items():
        checkpoint = output / f"candidate_validity_{name}.pt"
        torch.save(_checkpoint_payload(
            state, configurations[name], labels, pretrained, hwr,
            broad_path, broad_audit_path, prompt_path, direct_path,
            training_contract,
        ), checkpoint)
        model, contract, payload = load_candidate_validity_context(
            pretrained, checkpoint, hwr, device
        )
        reload_direct, direct_audit = decide_candidate_validity_rows(
            model, contract, payload, direct_rows, device,
            args.predict_batch_size,
        )
        reload_crohme, crohme_audit = decide_candidate_validity_rows(
            model, contract, payload, crohme_rows, device,
            args.predict_batch_size,
        )
        mismatches = {
            "direct": sum(
                reload_direct[key] != value
                for key, value in product_predictions[name]["direct"].items()
            ),
            "crohme": sum(
                reload_crohme[key] != value
                for key, value in product_predictions[name]["crohme"].items()
            ),
        }
        if any(mismatches.values()):
            raise AssertionError(f"candidate validity reload mismatch: {mismatches}")
        checkpoint_reports[name] = {
            "path": str(checkpoint),
            "sha256": masked._sha256(checkpoint),
            "reload_mismatches": mismatches,
            "candidate_audit": {
                "direct": direct_audit,
                "crohme": crohme_audit,
            },
        }
        del model, contract, payload
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for name in ("safe", "accuracy"):
        masked._write_prediction_rows(
            output / f"writer_loo_{name}_predictions.jsonl.gz",
            direct_rows, oof_predictions[name],
        )
        masked._write_prediction_rows(
            output / f"direct_refit_{name}_predictions.jsonl.gz",
            direct_rows, product_predictions[name]["direct"],
        )
        masked._write_prediction_rows(
            output / f"crohme_{name}_predictions.jsonl.gz",
            crohme_rows, product_predictions[name]["crohme"],
        )

    baseline_predictions = {
        str(row["record_id"]): str(row["final_topk"][0])
        for row in direct_rows
    }
    baseline = _metrics(direct_rows, baseline_predictions)
    candidate_contract_passed = all(
        checkpoint_reports[name]["candidate_audit"]["direct"]["new_tokens"] == 0
        and checkpoint_reports[name]["candidate_audit"]["direct"]["grouping_mutations"] == 0
        for name in checkpoint_reports
    )
    crohme_candidate_contract_report_only = all(
        checkpoint_reports[name]["candidate_audit"]["crohme"]["new_tokens"] == 0
        and checkpoint_reports[name]["candidate_audit"]["crohme"]["grouping_mutations"] == 0
        for name in checkpoint_reports
    )
    accuracy_new = oof_evaluation["accuracy"]["new_writer"]
    r6_new = r6_comparison["new_writer"]
    r6_nonregression = {
        "all_top1": accuracy_new["all_top1"] >= r6_new["all_top1"],
        "formula_exact": accuracy_new["formula_exact"] >= r6_new["formula_exact"],
        "strict_macro_top1": (
            accuracy_new["strict_macro_top1"] >= r6_new["strict_macro_top1"]
        ),
    }
    metric_promotion_gate = all(r6_nonregression.values())
    report = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "architecture": {
            "family": MODEL_FAMILY,
            "base": masked.MODEL_ID,
            "base_revision": masked.MODEL_REVISION,
            "hidden_size": 128,
            "transformer_layers": 2,
            "attention_heads": 2,
            "target_marker": "BERT token_type_id=1 at inserted candidate",
            "objective": "binary replaced-token validity",
            "parameters": sum(
                tensor.numel()
                for tensor in next(iter(product_states.values())).values()
            ),
            "external_pretrained_weights": True,
            "pretrained_license": "Apache-2.0",
            "parent_aiflow_context_checkpoint": None,
        },
        "data": {
            "broad_corpus": {
                "formulas": broad_formula_count,
                "audit": broad_audit,
            },
            "coverage_corpus": (
                {
                    "formulas": len(coverage_rows),
                    "focus_tokens": list(COVERAGE_TOKENS),
                    "audit": coverage_audit,
                    "initial_broad_checkpoint": initial_broad_metadata,
                }
                if args.coverage_finetune else None
            ),
            "prompt_admission": prompt_admission,
            "direct_records": len(direct_rows),
            "direct_formulas": len(_formula_rows(direct_rows)),
            "direct_writers": len({str(row["writer_group"]) for row in direct_rows}),
            "crohme_training_or_selection": False,
            "crohme_prior_diagnostic_informed_coverage_tokens": False,
            "evaluation_sequence_overlap": 0,
        },
        "training": {
            "pretraining_mode": pretraining_mode,
            "coverage_replay": {
                "per_token": args.coverage_replay_per_token,
                "tokens": len(COVERAGE_TOKENS),
                "formulas_per_finetune_epoch": (
                    len(COVERAGE_TOKENS) * args.coverage_replay_per_token
                ),
                "examples_per_finetune_epoch": (
                    2 * len(COVERAGE_TOKENS)
                    * args.coverage_replay_per_token
                ),
                "selection_basis": (
                    "user-specified homograph families plus project-owned dev; CROHME unused"
                ),
            },
            "state_variant_history": broad_history,
            "variant_grid": {
                "broad_epoch": list(BROAD_EPOCHS),
                "broad_epoch_field_semantics": (
                    "coverage_epoch" if args.coverage_finetune
                    else "broad_pretraining_epoch"
                ),
                "finetune_epoch": list(FINETUNE_EPOCHS),
                "lambda": list(LAMBDA_GRID),
            },
            "outer_split": (
                "leave-one-writer-out; held exact token sequences removed from training"
            ),
            "inner_split": "formula and exact-token-sequence grouped nested selection",
            "folds": folds,
            "product_configuration": configurations,
            "product_losses": product_losses,
        },
        "evaluation": {
            "hwr_writer_loo_baseline": baseline,
            "candidate_validity_writer_loo": oof_evaluation,
            "product_refit_and_crohme": product_evaluation,
            "independent_embedding_r2_comparison": r2_comparison,
            "historical_r6_comparison": r6_comparison,
        },
        "integrity": {
            "shape_training_performed": False,
            "shape_gradient_updates": 0,
            "candidate_contract_passed": candidate_contract_passed,
            "immutable_hashes_before": hashes_before,
            "immutable_hashes_after": hashes_after,
            "immutable_inputs_unchanged": hashes_before == hashes_after,
            "training_immutable_hashes_before": training_hashes_before,
            "training_immutable_hashes_after": training_hashes_after,
            "training_immutable_inputs_unchanged": (
                training_hashes_before == training_hashes_after
            ),
            "comparison_models_loaded_after_all_validity_training": True,
            "crohme_candidate_contract_report_only": crohme_candidate_contract_report_only,
            "training_data_guard": zero_crohme_training_manifest(
                admitted_sources={
                    "commercial_broad_formulas": broad_formula_count,
                    "project_owned_prompt_formulas": len(prompt_rows),
                    "project_owned_direct_candidates": len(direct_rows),
                    "project_owned_coverage_formulas": len(coverage_rows),
                }
            ),
        },
        "checkpoints": {
            "state_variants": broad_checkpoints,
            "product": checkpoint_reports,
        },
        "decision": {
            "research_gate_passed": (
                oof_evaluation["accuracy"]["all"]["all_top1"] > baseline["all_top1"]
                and candidate_contract_passed
                and training_hashes_before == training_hashes_after
            ),
            "r6_new_writer_nonregression": r6_nonregression,
            "metric_promotion_gate_passed": metric_promotion_gate,
            "fresh_untouched_acceptance_available": False,
            "automatic_default_replacement": False,
            "runtime_status": (
                "promotion-candidate shadow" if metric_promotion_gate
                else "standalone shadow candidate-validity detector"
            ),
        },
    }
    report_path = output / "candidate_validity_context_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    _event(
        "candidate_validity_complete", output=str(output),
        baseline_top1=baseline["all_top1"],
        accuracy_top1=oof_evaluation["accuracy"]["all"]["all_top1"],
        accuracy_formula=oof_evaluation["accuracy"]["all"]["formula_exact"],
        new_writer_top1=accuracy_new["all_top1"],
        r6_metric_gate=metric_promotion_gate,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
