#!/usr/bin/env python3
"""Train a candidate-preserving formula-context model with no external weights or corpus."""

from __future__ import annotations

from training_data_guard_v1 import (
    assert_training_entrypoint_arguments_clean, assert_training_path_clean,
    assert_training_rows_clean, zero_crohme_training_manifest,
)
if __name__ == "__main__":
    assert_training_entrypoint_arguments_clean()

import argparse
import gc
import hashlib
import json
import math
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel

from character_tensor_v1 import ROOT, _json_lines
from evaluate_homograph_context_reranker_v1 import _metrics
import train_context_decision_layer_v1 as context
import train_masked_context_reranker_v1 as masked
from supported_exact_context_guard_v1 import (
    STRONG_EXACT_CONTEXT_SCALE,
    apply_supported_exact_context_guard,
)


SCHEMA = "aiflow-owned-formula-context/v1"
SEED = 20260822
DEFAULT_INPUT = ROOT / "artifacts" / "homograph_context_20260814"
DEFAULT_OUTPUT = ROOT / "artifacts" / "owned_formula_context_20260822"
DEFAULT_R2_REPORT = ROOT / "artifacts" / "context_role_finalizer_20260819_r2" / "context_role_finalizer_report.json"
DEFAULT_R7_REPORT = ROOT / "artifacts" / "distilled_math_context_20260819_r7" / "distilled_math_context_report.json"
SPECIAL_IDS = {"pad": 0, "cls": 1, "sep": 2, "mask": 3}
MODEL_CONFIG = {
    "hidden_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "intermediate_size": 256,
    "max_position_embeddings": 256,
    "type_vocab_size": 1,
    "hidden_dropout_prob": 0.1,
    "attention_probs_dropout_prob": 0.1,
}
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-2
EXACT_LOSS_WEIGHT = 0.5
REAL_REPEAT = 8
RUNTIME_CONFIGURATION_KEYS = frozenset({
    "context_weight", "shape_weight", "grammar_weight", "margin",
    "exact_context_scale", "candidate_probability_ratio_floor",
})

COMMON_OPERANDS = (
    "x", "y", "z", "a", "b", "c", "n", "m", "k", "t", "u", "v", "w",
    "f", "g", "h", "p", "q", "r", "s", r"\alpha", r"\beta", r"\gamma",
    r"\theta", r"\lambda", r"\mu", r"\pi", r"\sigma", r"\phi", r"\omega",
)
COMMON_BINARY_OPERATORS = (
    "+", "-", r"\times", "/", r"\div", r"\cdot", r"\pm", r"\cap", r"\cup",
)
COMMON_RELATIONS = (
    "=", "<", ">", r"\leq", r"\geq", r"\neq", r"\approx", r"\in", r"\notin",
)
COMMON_PREFIX_OPERATORS = (
    r"\sqrt{}", r"\sum", r"\prod", r"\int", r"\oint", r"\partial", r"\nabla",
)
FENCE_PAIRS = (
    ("(", ")"), ("[", "]"), (r"\{", r"\}"),
    (r"\langle", r"\rangle"), (r"\lceil", r"\rceil"),
    (r"\lfloor", r"\rfloor"), ("|", "|"),
)


def _event(name: str, **values: object) -> None:
    print(json.dumps({"event": name, **values}, ensure_ascii=False), flush=True)


def _seed(salt: str) -> int:
    return SEED + int.from_bytes(hashlib.sha256(salt.encode("utf-8")).digest()[:4], "big") % 100_000


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _contract(labels: list[str]) -> dict:
    class_start = len(SPECIAL_IDS)
    relation_start = class_start + len(labels)
    return {
        "labels": labels,
        "label_to_index": {label: index for index, label in enumerate(labels)},
        "class_ids": [class_start + index for index in range(len(labels))],
        "relation_ids": {
            relation: relation_start + index
            for index, relation in enumerate(masked.RELATIONS)
        },
        "pad_id": SPECIAL_IDS["pad"],
        "cls_id": SPECIAL_IDS["cls"],
        "sep_id": SPECIAL_IDS["sep"],
        "mask_id": SPECIAL_IDS["mask"],
        "vocab_size": relation_start + len(masked.RELATIONS),
    }


class OwnedFormulaContext(nn.Module):
    def __init__(self, vocab_size: int, class_count: int):
        super().__init__()
        config = BertConfig(
            vocab_size=vocab_size,
            pad_token_id=SPECIAL_IDS["pad"],
            **MODEL_CONFIG,
        )
        self.encoder = BertModel(config, add_pooling_layer=False)
        self.role_head = nn.Linear(config.hidden_size, len(context.ROLES))
        self.class_head = nn.Linear(config.hidden_size, class_count)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, mask_positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        ).last_hidden_state
        masked_hidden = hidden[
            torch.arange(len(hidden), device=hidden.device), mask_positions
        ]
        return self.role_head(masked_hidden), self.class_head(masked_hidden)


class FormulaGenerator:
    """Small owned DSL; it emits tokens directly and never reads a text corpus."""

    def __init__(self, labels: list[str], rows: list[dict], seed: int):
        self.labels = labels
        self.label_to_index = {label: index for index, label in enumerate(labels)}
        self.rng = random.Random(seed)
        support = Counter(str(row["label"]) for row in rows)
        self.pools = {
            role: [index for index, label in enumerate(labels) if context._semantic_role(label) == role]
            for role in context.ROLES
        }
        self.weights = {
            role: [0.02 + float(support[labels[index]]) for index in indices]
            for role, indices in self.pools.items()
        }
        self.common = {
            "operand": self._available(COMMON_OPERANDS),
            "binary": self._available(COMMON_BINARY_OPERATORS),
            "relation": self._available(COMMON_RELATIONS),
            "prefix": self._available(COMMON_PREFIX_OPERATORS),
            "digit": self._available(tuple(str(value) for value in range(10))),
        }
        self.fence_pairs = [
            (self.label_to_index[left], self.label_to_index[right])
            for left, right in FENCE_PAIRS
            if left in self.label_to_index and right in self.label_to_index
        ]
        if not all(self.pools.values()) or not self.common["operand"] or not self.common["binary"]:
            raise ValueError("owned formula generator label coverage incomplete")

    def _available(self, values: tuple[str, ...]) -> list[int]:
        return [self.label_to_index[value] for value in values if value in self.label_to_index]

    def _weighted_role(self, role: str) -> int:
        return self.rng.choices(self.pools[role], weights=self.weights[role], k=1)[0]

    def _pick(self, kind: str, role: str) -> int:
        values = self.common[kind]
        if values and self.rng.random() < 0.82:
            return self.rng.choice(values)
        return self._weighted_role(role)

    @staticmethod
    def _append(tokens: list[int], relations: list[str], token: int, relation: str = "right") -> None:
        if tokens:
            relations.append(relation)
        tokens.append(token)

    def _number(self, tokens: list[int], relations: list[str]) -> None:
        for _ in range(self.rng.randint(1, 5)):
            self._append(tokens, relations, self._pick("digit", "digit"))

    def _atom(self, tokens: list[int], relations: list[str], depth: int) -> None:
        choice = self.rng.random()
        if choice < 0.40:
            self._number(tokens, relations)
        elif choice < 0.78 or depth >= 1:
            self._append(tokens, relations, self._pick("operand", "operand"))
        elif choice < 0.92 and self.fence_pairs:
            left, right = self.rng.choice(self.fence_pairs)
            self._append(tokens, relations, left)
            self._expression(tokens, relations, depth + 1, self.rng.randint(1, 3))
            self._append(tokens, relations, right)
        else:
            self._append(tokens, relations, self._pick("prefix", "operator"))
            self._atom(tokens, relations, depth + 1)
        if len(tokens) < 60 and self.rng.random() < 0.30:
            relation = self.rng.choice(("superscript", "subscript", "above", "below"))
            if self.rng.random() < 0.65:
                self._append(tokens, relations, self._pick("digit", "digit"), relation)
            else:
                self._append(tokens, relations, self._pick("operand", "operand"), relation)

    def _expression(
        self, tokens: list[int], relations: list[str], depth: int, terms: int,
    ) -> None:
        if self.rng.random() < 0.12:
            self._append(tokens, relations, self._pick("prefix", "operator"))
        self._atom(tokens, relations, depth)
        for _ in range(terms - 1):
            self._append(tokens, relations, self._pick("binary", "operator"))
            self._atom(tokens, relations, depth)

    def formula(self) -> tuple[list[int], list[str]]:
        tokens: list[int] = []
        relations: list[str] = []
        self._expression(tokens, relations, 0, self.rng.randint(1, 8))
        if len(tokens) < 55 and self.rng.random() < 0.45:
            self._append(tokens, relations, self._pick("relation", "operator"))
            self._expression(tokens, relations, 0, self.rng.randint(1, 5))
        if len(tokens) < 60 and self.rng.random() < 0.03:
            self._append(tokens, relations, self._weighted_role("other"))
        if len(tokens) > 63:
            tokens = tokens[:63]
            relations = relations[:62]
        if len(relations) != len(tokens) - 1:
            raise AssertionError("owned formula relation contract mismatch")
        return tokens, relations


def _examples(labels: list[str], rows: list[dict], count: int, seed: int) -> dict[str, torch.Tensor]:
    contract = _contract(labels)
    generator = FormulaGenerator(labels, rows, seed)
    role_to_index = {role: index for index, role in enumerate(context.ROLES)}
    examples = []
    while len(examples) < count:
        tokens, relations = generator.formula()
        targets = generator.rng.sample(
            range(len(tokens)), min(len(tokens), generator.rng.randint(1, 3))
        )
        for target in targets:
            ids = [contract["cls_id"]]
            mask_position = -1
            for index, token in enumerate(tokens):
                if index:
                    ids.append(contract["relation_ids"][relations[index - 1]])
                if index == target:
                    mask_position = len(ids)
                    ids.append(contract["mask_id"])
                else:
                    ids.append(contract["class_ids"][token])
            ids.append(contract["sep_id"])
            label = labels[tokens[target]]
            examples.append((
                ids,
                mask_position,
                role_to_index[context._semantic_role(label)],
                tokens[target],
            ))
            if len(examples) == count:
                break
    width = max(len(example[0]) for example in examples)
    output = {
        "input_ids": torch.full((count, width), contract["pad_id"], dtype=torch.long),
        "attention_mask": torch.zeros((count, width), dtype=torch.long),
        "mask_positions": torch.empty(count, dtype=torch.long),
        "role_targets": torch.empty(count, dtype=torch.long),
        "class_targets": torch.empty(count, dtype=torch.long),
    }
    for index, (ids, position, role, target) in enumerate(examples):
        output["input_ids"][index, :len(ids)] = torch.tensor(ids)
        output["attention_mask"][index, :len(ids)] = 1
        output["mask_positions"][index] = position
        output["role_targets"][index] = role
        output["class_targets"][index] = target
    return output


def _append_real(data: dict[str, torch.Tensor], rows: list[dict], contract: dict) -> dict[str, torch.Tensor]:
    if not rows or REAL_REPEAT < 1:
        return data
    packed = masked._pack(rows, contract)
    role_targets = torch.tensor([
        context.ROLE_TO_INDEX[context._semantic_role(contract["labels"][int(target)])]
        for target in packed["targets"]
    ], dtype=torch.long)
    real = {
        "input_ids": packed["input_ids"].repeat(REAL_REPEAT, 1),
        "attention_mask": packed["attention_mask"].repeat(REAL_REPEAT, 1),
        "mask_positions": packed["mask_positions"].repeat(REAL_REPEAT),
        "role_targets": role_targets.repeat(REAL_REPEAT),
        "class_targets": packed["targets"].repeat(REAL_REPEAT),
    }
    width = max(data["input_ids"].shape[1], real["input_ids"].shape[1])
    for key, fill in (("input_ids", contract["pad_id"]), ("attention_mask", 0)):
        data[key] = F.pad(data[key], (0, width - data[key].shape[1]), value=fill)
        real[key] = F.pad(real[key], (0, width - real[key].shape[1]), value=fill)
    return {key: torch.cat((data[key], real[key])) for key in data}


def _new_model(contract: dict, device: torch.device, seed: int) -> OwnedFormulaContext:
    _set_seed(seed)
    return OwnedFormulaContext(contract["vocab_size"], len(contract["labels"])).to(device)


def _batch_loss(
    model: OwnedFormulaContext, data: dict[str, torch.Tensor], indices, device: torch.device,
) -> torch.Tensor:
    role_logits, class_logits = model(
        data["input_ids"][indices].to(device),
        data["attention_mask"][indices].to(device),
        data["mask_positions"][indices].to(device),
    )
    return (
        F.cross_entropy(role_logits, data["role_targets"][indices].to(device))
        + EXACT_LOSS_WEIGHT * F.cross_entropy(
            class_logits, data["class_targets"][indices].to(device)
        )
    )


def _train_epoch(
    model: OwnedFormulaContext, data: dict[str, torch.Tensor], optimizer, device: torch.device,
    batch_size: int, generator: torch.Generator,
) -> float:
    model.train()
    order = torch.randperm(len(data["role_targets"]), generator=generator)
    total = 0.0
    for start in range(0, len(order), batch_size):
        indices = order[start:start + batch_size]
        optimizer.zero_grad(set_to_none=True)
        loss = _batch_loss(model, data, indices, device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += float(loss.detach()) * len(indices)
    return total / len(order)


@torch.inference_mode()
def _validation_loss(
    model: OwnedFormulaContext, data: dict[str, torch.Tensor], device: torch.device,
    batch_size: int,
) -> float:
    model.eval()
    total = 0.0
    for start in range(0, len(data["role_targets"]), batch_size):
        indices = slice(start, start + batch_size)
        loss = _batch_loss(model, data, indices, device)
        total += float(loss) * len(data["role_targets"][indices])
    return total / len(data["role_targets"])


@torch.inference_mode()
def _components(
    model: OwnedFormulaContext, contract: dict, rows: list[dict], grammar: dict,
    device: torch.device, batch_size: int,
    context_tokens: dict[str, str] | None = None,
) -> dict:
    packed = _pack_runtime(rows, contract, context_tokens)
    role_parts, class_parts = [], []
    model.eval()
    for start in range(0, len(rows), batch_size):
        stop = min(start + batch_size, len(rows))
        role_logits, class_logits = model(
            packed["input_ids"][start:stop].to(device),
            packed["attention_mask"][start:stop].to(device),
            packed["mask_positions"][start:stop].to(device),
        )
        role_parts.append(role_logits.cpu())
        class_parts.append(class_logits.cpu())
    role_scores = torch.cat(role_parts).numpy()
    class_scores = torch.cat(class_parts)
    width = max(len(row["final_topk"]) for row in packed["rows"])
    candidate_logits = np.full((len(rows), width), -1e9, dtype=np.float32)
    candidate_mask = np.zeros((len(rows), width), dtype=bool)
    for row_index, row in enumerate(packed["rows"]):
        for candidate_index, token in enumerate(row["final_topk"]):
            candidate_logits[row_index, candidate_index] = float(
                class_scores[row_index, contract["label_to_index"][str(token)]]
            )
            candidate_mask[row_index, candidate_index] = True
    return context._assemble_role_components(
        packed["rows"], candidate_logits, candidate_mask, rows, grammar, role_scores
    )


def _pack_runtime(
    rows: list[dict], contract: dict,
    context_tokens: dict[str, str] | None = None,
) -> dict:
    """Build masked formula inputs without requiring unavailable truth labels."""
    examples = []
    label_to_index = contract["label_to_index"]
    record_ids = {str(row["record_id"]) for row in rows}
    if context_tokens is not None and set(context_tokens) != record_ids:
        raise ValueError("formula context token coverage mismatch")
    if context_tokens is not None:
        for row in rows:
            record_id = str(row["record_id"])
            if str(context_tokens[record_id]) not in row["final_topk"]:
                raise ValueError(
                    f"formula context token outside HWR candidates: {record_id}"
                )
    for formula_id, sequence in masked._formulae(rows).items():
        relations = []
        for index in range(1, len(sequence)):
            explicit = sequence[index].get("context", {}).get("relation_from_previous")
            relation = str(explicit) if explicit is not None else masked._spatial_relation(
                sequence[index - 1], sequence[index]
            )
            if relation not in masked.RELATIONS:
                raise ValueError(f"unsupported formula layout relation: {relation}")
            relations.append(relation)
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
                    record_id = str(row["record_id"])
                    token = str(
                        context_tokens[record_id]
                        if context_tokens is not None else row["final_topk"][0]
                    )
                    if token not in label_to_index:
                        raise ValueError(
                            f"HWR context token outside 372 classes: {token}"
                        )
                    ids.append(contract["class_ids"][label_to_index[token]])
            ids.append(contract["sep_id"])
            if mask_position < 0 or len(ids) > 512:
                raise ValueError(f"masked sequence exceeds context contract: {formula_id}")
            examples.append({
                "row": target,
                "input_ids": ids,
                "mask_position": mask_position,
            })
    if not examples:
        raise ValueError("context finalizer requires at least one candidate row")
    if {str(example["row"]["record_id"]) for example in examples} != {
        str(row["record_id"]) for row in rows
    }:
        raise AssertionError("runtime masked-context example coverage mismatch")
    width = max(len(example["input_ids"]) for example in examples)
    input_ids = torch.full(
        (len(examples), width), int(contract["pad_id"]), dtype=torch.long
    )
    attention_mask = torch.zeros((len(examples), width), dtype=torch.long)
    mask_positions = torch.empty(len(examples), dtype=torch.long)
    for index, example in enumerate(examples):
        length = len(example["input_ids"])
        input_ids[index, :length] = torch.tensor(
            example["input_ids"], dtype=torch.long
        )
        attention_mask[index, :length] = 1
        mask_positions[index] = int(example["mask_position"])
    return {
        "rows": [example["row"] for example in examples],
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "mask_positions": mask_positions,
    }


def _configuration_grid() -> list[dict | None]:
    output: list[dict | None] = [None]
    for context_weight in (0.25, 0.5, 1.0):
        for shape_weight in (0.5, 1.0, 2.0):
            for grammar_weight in (1.0, 2.0):
                for margin in (0.25, 0.5, 1.0):
                    for exact_context_scale in (0.0, 1.0, 2.0, 4.0):
                        output.append({
                            "context_weight": context_weight,
                            "shape_weight": shape_weight,
                            "grammar_weight": grammar_weight,
                            "margin": margin,
                            "exact_context_scale": exact_context_scale,
                        })
    return output


def _selection_key(metrics: dict, configuration: dict | None) -> tuple:
    complexity = 0.0 if configuration is None else sum(float(value) for value in configuration.values())
    return (
        metrics["regressed"] == 0,
        metrics["all_top1"],
        metrics["formula_exact"],
        metrics["strict_macro_top1"],
        metrics["strict_micro_top1"],
        -metrics["changed"],
        -complexity,
    )


def _select(
    rows: list[dict], labels: list[str], device: torch.device, args, salt: str,
) -> dict:
    fit_rows, validation_rows = masked._inner_split(rows, f"owned-{salt}")
    seed = _seed(salt)
    contract = _contract(labels)
    training = _append_real(
        _examples(labels, fit_rows, args.synthetic_train_examples, seed), fit_rows, contract
    )
    validation = _examples(labels, fit_rows, args.synthetic_valid_examples, seed + 1)
    model = _new_model(contract, device, seed)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    generator = torch.Generator().manual_seed(seed)
    grammar = context._fit_role_grammar(fit_rows)
    label_support = dict(Counter(str(row["label"]) for row in fit_rows))
    reliability = context._fit_top1_reliability(fit_rows)
    trials = []
    for epoch in range(1, args.max_epochs + 1):
        train_loss = _train_epoch(
            model, training, optimizer, device, args.batch_size, generator
        )
        validation_loss = _validation_loss(
            model, validation, device, args.predict_batch_size
        )
        components = _components(
            model, contract, validation_rows, grammar, device, args.predict_batch_size
        )
        epoch_trials = []
        for configuration in _configuration_grid():
            predictions, audit = context._predict(
                components, configuration, label_support, reliability
            )
            metrics = _metrics(validation_rows, predictions)
            epoch_trials.append({
                "configuration": configuration, "metrics": metrics, "audit": audit,
            })
        best = max(
            epoch_trials,
            key=lambda trial: _selection_key(trial["metrics"], trial["configuration"]),
        )
        trials.append({
            "epoch": epoch,
            "train_objective": train_loss,
            "synthetic_validation_objective": validation_loss,
            "configuration": best["configuration"],
            "metrics": best["metrics"],
            "audit": best["audit"],
        })
        _event(
            "owned_selection_epoch", salt=salt, epoch=epoch,
            validation_top1=best["metrics"]["all_top1"],
            validation_regressed=best["metrics"]["regressed"],
        )
    selected = max(
        trials,
        key=lambda trial: _selection_key(trial["metrics"], trial["configuration"]),
    )
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "epoch": selected["epoch"],
        "configuration": selected["configuration"],
        "selected_metrics": selected["metrics"],
        "fit_formulas": len(masked._formulae(fit_rows)),
        "validation_formulas": len(masked._formulae(validation_rows)),
        "curve": trials,
    }


def _fit(
    rows: list[dict], labels: list[str], epochs: int, device: torch.device, args, salt: str,
) -> tuple[OwnedFormulaContext, dict, list[float]]:
    seed = _seed(salt)
    contract = _contract(labels)
    training = _append_real(
        _examples(labels, rows, args.synthetic_train_examples, seed), rows, contract
    )
    model = _new_model(contract, device, seed)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    generator = torch.Generator().manual_seed(seed)
    losses = []
    for epoch in range(1, epochs + 1):
        losses.append(_train_epoch(
            model, training, optimizer, device, args.batch_size, generator
        ))
        _event("owned_fit_epoch", salt=salt, epoch=epoch, objective=losses[-1])
    model.eval()
    return model, contract, losses


def _predict(
    model: OwnedFormulaContext, contract: dict, rows: list[dict], training_rows: list[dict],
    configuration: dict | None, device: torch.device, batch_size: int,
) -> tuple[dict[str, str], dict]:
    components = _components(
        model, contract, rows, context._fit_role_grammar(training_rows), device, batch_size
    )
    return context._predict(
        components,
        configuration,
        dict(Counter(str(row["label"]) for row in training_rows)),
        context._fit_top1_reliability(training_rows),
    )


def _writer_loo(
    rows: list[dict], labels: list[str], device: torch.device, args,
) -> tuple[dict[str, str], list[dict]]:
    predictions, folds = {}, []
    writers = sorted({str(row["writer_group"]) for row in rows})
    for index, writer in enumerate(writers):
        training = [row for row in rows if str(row["writer_group"]) != writer]
        held = [row for row in rows if str(row["writer_group"]) == writer]
        selection = _select(training, labels, device, args, f"writer-select-{writer}")
        model, contract, losses = _fit(
            training, labels, selection["epoch"], device, args, f"writer-fit-{writer}"
        )
        fold_predictions, audit = _predict(
            model, contract, held, training, selection["configuration"],
            device, args.predict_batch_size,
        )
        predictions.update(fold_predictions)
        folds.append({
            "held_writer_group": writer,
            "selection": selection,
            "fit_final_objective": losses[-1],
            "metrics": _metrics(held, fold_predictions),
            "audit": audit,
        })
        _event(
            "owned_writer_fold", fold=index + 1, writer=writer,
            top1=folds[-1]["metrics"]["all_top1"],
            epoch=selection["epoch"],
        )
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if set(predictions) != {str(row["record_id"]) for row in rows}:
        raise AssertionError("owned writer-LOO prediction coverage mismatch")
    return predictions, folds


def _aggregate_fold_configuration(folds: list[dict]) -> dict | None:
    """Use the outer writer folds, not one tiny product split, for runtime policy."""
    configurations = [fold["selection"]["configuration"] for fold in folds]
    if not configurations or any(configuration is None for configuration in configurations):
        return None
    keys = tuple(configurations[0])
    if any(tuple(configuration) != keys for configuration in configurations):
        raise ValueError("owned context fold configuration contract mismatch")
    return {
        key: float(median(float(configuration[key]) for configuration in configurations))
        for key in keys
    }


def _aggregate_fold_epochs(folds: list[dict]) -> int:
    epochs = [int(fold["selection"]["epoch"]) for fold in folds]
    if not epochs or any(epoch < 1 for epoch in epochs):
        raise ValueError("owned context fold epoch contract mismatch")
    return int(median(epochs))


def _successful_override_probability_floor(
    rows: list[dict], predictions: dict[str, str],
) -> float:
    """Retain at least the weakest cross-writer correction proven on owned ink."""
    ratios = []
    for row in rows:
        prediction = predictions[str(row["record_id"])]
        if prediction == str(row["final_topk"][0]) or prediction != str(row["label"]):
            continue
        candidate_index = row["final_topk"].index(prediction)
        probabilities = row["final_topk_probabilities"]
        ratios.append(
            float(probabilities[candidate_index]) / max(float(probabilities[0]), 1e-12)
        )
    return min(ratios, default=1.0)


def _valid_runtime_configuration(configuration: object) -> bool:
    if not isinstance(configuration, dict) or set(configuration) != RUNTIME_CONFIGURATION_KEYS:
        return False
    try:
        values = {key: float(configuration[key]) for key in RUNTIME_CONFIGURATION_KEYS}
    except (TypeError, ValueError):
        return False
    return (
        all(math.isfinite(value) for value in values.values())
        and values["context_weight"] > 0.0
        and values["shape_weight"] > 0.0
        and values["grammar_weight"] > 0.0
        and values["margin"] >= 0.0
        and values["exact_context_scale"] >= 0.0
        and 0.0 <= values["candidate_probability_ratio_floor"] <= 1.0
    )


def load_owned_formula_context(
    checkpoint_path: Path, hwr_checkpoint: Path, device: torch.device,
    *, require_d_drive: bool = True,
) -> tuple[OwnedFormulaContext, dict, dict]:
    if require_d_drive:
        checkpoint_path = masked._d_path(checkpoint_path, "owned context checkpoint")
        hwr_checkpoint = masked._d_path(hwr_checkpoint, "HWR checkpoint")
    else:
        checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        hwr_checkpoint = Path(hwr_checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"missing owned context checkpoint: {checkpoint_path}")
    if not hwr_checkpoint.is_file():
        raise FileNotFoundError(f"missing HWR checkpoint: {hwr_checkpoint}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    labels = masked._labels(hwr_checkpoint)
    if (
        payload.get("schema") != SCHEMA
        or payload.get("model_config") != MODEL_CONFIG
        or payload.get("special_ids") != SPECIAL_IDS
        or payload.get("math_labels") != labels
        or payload.get("roles") != list(context.ROLES)
        or payload.get("semantic_role_by_label")
        != {label: context._semantic_role(label) for label in labels}
        or payload.get("hwr_checkpoint_sha256") != masked._sha256(hwr_checkpoint)
        or payload.get("external_pretrained_weights") is not False
        or payload.get("external_text_corpus") is not False
        or payload.get("exact_loss_weight") != EXACT_LOSS_WEIGHT
        or not isinstance(payload.get("selected_epochs"), int)
        or int(payload["selected_epochs"]) < 1
        or not _valid_runtime_configuration(payload.get("configuration"))
    ):
        raise ValueError("owned formula-context checkpoint contract mismatch")
    contract = _contract(labels)
    model = _new_model(contract, device, SEED)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, contract, payload


def decide_owned_formula_rows(
    model: OwnedFormulaContext, contract: dict, payload: dict, rows: list[dict],
    device: torch.device, batch_size: int = 128,
    context_tokens: dict[str, str] | None = None,
) -> tuple[dict[str, str], dict]:
    components = _components(
        model, contract, rows, payload["role_grammar"], device, batch_size,
        context_tokens,
    )
    return context._predict(
        components,
        payload["configuration"],
        payload["label_support"],
        payload["top1_reliability"],
    )


def decide_owned_formula_rows_supported_exact(
    model: OwnedFormulaContext, contract: dict, payload: dict, rows: list[dict],
    device: torch.device, batch_size: int = 128,
    context_tokens: dict[str, str] | None = None,
) -> tuple[dict[str, str], dict]:
    components = _components(
        model, contract, rows, payload["role_grammar"], device, batch_size,
        context_tokens,
    )
    baseline, audit = context._predict(
        components, payload["configuration"], payload["label_support"],
        payload["top1_reliability"],
    )
    stronger, _ = context._predict(
        components,
        {
            **payload["configuration"],
            "exact_context_scale": STRONG_EXACT_CONTEXT_SCALE,
        },
        payload["label_support"], payload["top1_reliability"],
    )
    finalized, guard_audit = apply_supported_exact_context_guard(
        rows, baseline, stronger, payload["label_support"]
    )
    return finalized, {**audit, "supported_exact_context": guard_audit}


def _assert_candidate_contract(name: str, audit: dict) -> None:
    if (
        audit["candidate_preservation_rate"] != 1.0
        or audit["new_tokens"] != 0
        or audit["grouping_mutations"] != 0
    ):
        raise AssertionError(f"{name} candidate contract violated: {audit}")


def _reference(path: Path, field: str) -> tuple[dict | None, str | None]:
    if not path.is_file():
        return None, None
    report = json.loads(path.read_text(encoding="utf-8"))
    return report["evaluation"][field], masked._sha256(path)


def _self_test(labels: list[str], device: torch.device) -> None:
    contract = _contract(labels)
    rows = [{"label": "1"}, {"label": "+"}, {"label": "x"}]
    generator = FormulaGenerator(labels, rows, SEED)
    tokens, relations = generator.formula()
    assert tokens and len(relations) == len(tokens) - 1
    data = _examples(labels, rows, 8, SEED)
    model = _new_model(contract, device, SEED)
    role_logits, class_logits = model(
        data["input_ids"].to(device),
        data["attention_mask"].to(device),
        data["mask_positions"].to(device),
    )
    assert role_logits.shape == (8, len(context.ROLES))
    assert class_logits.shape == (8, len(labels))
    assert sum(parameter.numel() for parameter in model.parameters()) < 1_000_000
    runtime_rows = [
        {
            "record_id": "runtime-0", "formula_id": "runtime", "label": "1",
            "final_topk": ["1", "+"],
            "context": {"index": 0, "length": 2},
            "geometry": {"center_x": 0.0, "center_y": 0.0, "width_rel": 0.5, "height_rel": 1.0},
        },
        {
            "record_id": "runtime-1", "formula_id": "runtime", "label": "+",
            "final_topk": ["+", "1"],
            "context": {"index": 1, "length": 2},
            "geometry": {"center_x": 1.0, "center_y": 0.0, "width_rel": 1.0, "height_rel": 1.0},
        },
    ]
    training_pack = masked._pack(runtime_rows, contract)
    runtime_clean = [
        {key: value for key, value in row.items() if key != "label"}
        for row in runtime_rows
    ]
    runtime_pack = _pack_runtime(runtime_clean, contract)
    for key in ("input_ids", "attention_mask", "mask_positions"):
        assert torch.equal(training_pack[key], runtime_pack[key])
    rechecked_pack = _pack_runtime(
        runtime_clean, contract, {"runtime-0": "+", "runtime-1": "1"}
    )
    assert not torch.equal(runtime_pack["input_ids"], rechecked_pack["input_ids"])
    try:
        _pack_runtime(
            runtime_clean, contract, {"runtime-0": "x", "runtime-1": "1"}
        )
    except ValueError:
        pass
    else:
        raise AssertionError("context token outside HWR candidates was accepted")
    assert _aggregate_fold_configuration([
        {"selection": {"configuration": {"weight": 0.25}}},
        {"selection": {"configuration": {"weight": 1.0}}},
        {"selection": {"configuration": {"weight": 2.0}}},
    ]) == {"weight": 1.0}
    assert _aggregate_fold_epochs([
        {"selection": {"epoch": 1}},
        {"selection": {"epoch": 4}},
        {"selection": {"epoch": 1}},
    ]) == 1
    floor = _successful_override_probability_floor([
        {
            "record_id": "correction", "label": "1", "final_topk": ["|", "1"],
            "final_topk_probabilities": [0.8, 0.08],
        }
    ], {"correction": "1"})
    assert abs(floor - 0.1) < 1e-12
    assert _valid_runtime_configuration({
        "context_weight": 1.0, "shape_weight": 0.5, "grammar_weight": 1.0,
        "margin": 0.25, "exact_context_scale": 1.0,
        "candidate_probability_ratio_floor": 0.01,
    })
    assert not _valid_runtime_configuration({
        "context_weight": 1.0, "shape_weight": 0.5, "grammar_weight": 1.0,
        "margin": 0.25, "exact_context_scale": 1.0,
        "candidate_probability_ratio_floor": float("nan"),
    })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hwr-checkpoint", type=Path, default=masked.DEFAULT_PRODUCT)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--r2-report", type=Path, default=DEFAULT_R2_REPORT)
    parser.add_argument("--r7-report", type=Path, default=DEFAULT_R7_REPORT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-epochs", type=int, default=8)
    parser.add_argument("--synthetic-train-examples", type=int, default=16000)
    parser.add_argument("--synthetic-valid-examples", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--predict-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if min(
        args.max_epochs, args.synthetic_train_examples, args.synthetic_valid_examples,
        args.batch_size, args.predict_batch_size,
    ) < 1:
        parser.error("epochs, examples, and batch sizes must be positive")
    if args.learning_rate <= 0.0 or args.weight_decay < 0.0:
        parser.error("learning rate must be positive and weight decay non-negative")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    hwr_checkpoint = masked._d_path(args.hwr_checkpoint, "HWR checkpoint")
    labels = masked._labels(hwr_checkpoint)
    if args.self_test:
        _self_test(labels, device)
        _event("self_test", status="pass", device=str(device))
        return 0

    input_root = masked._d_path(args.input, "candidate input")
    output = masked._d_path(args.output, "owned context output")
    checkpoint_path = output / "owned_formula_context_product.pt"
    report_path = output / "owned_formula_context_report.json"
    if checkpoint_path.exists() or report_path.exists():
        parser.error(f"refusing to overwrite owned context output: {output}")
    direct_path = assert_training_path_clean(
        input_root / "direct_candidates.jsonl.gz", "owned context fit candidates"
    )
    crohme_path = input_root / "crohme_candidates.jsonl.gz"
    direct_rows = list(_json_lines(direct_path))
    assert_training_rows_clean(direct_rows)
    crohme_rows = list(_json_lines(crohme_path))
    direct_formula_count = len({str(row["formula_id"]) for row in direct_rows})
    direct_writer_count = len({str(row["writer_group"]) for row in direct_rows})
    direct_partitions = dict(Counter(
        str(row.get("evaluation_partition", "unspecified")) for row in direct_rows
    ))
    _set_seed(SEED)
    _event(
        "owned_context_start", device=str(device), direct_records=len(direct_rows),
        synthetic_examples=args.synthetic_train_examples,
    )

    direct_predictions, folds = _writer_loo(direct_rows, labels, device, args)
    direct_baseline_predictions = {
        str(row["record_id"]): str(row["final_topk"][0]) for row in direct_rows
    }
    direct_baseline = _metrics(direct_rows, direct_baseline_predictions)
    direct_metrics = _metrics(direct_rows, direct_predictions)

    product_configuration = _aggregate_fold_configuration(folds)
    probability_ratio_floor = _successful_override_probability_floor(
        direct_rows, direct_predictions
    )
    if product_configuration is not None:
        product_configuration = {
            **product_configuration,
            "candidate_probability_ratio_floor": probability_ratio_floor,
        }
    product_epochs = _aggregate_fold_epochs(folds)
    product, contract, product_losses = _fit(
        direct_rows, labels, product_epochs, device, args, "product-fit"
    )
    direct_refit_predictions, direct_refit_runtime = _predict(
        product, contract, direct_rows, direct_rows, product_configuration,
        device, args.predict_batch_size,
    )
    crohme_predictions, crohme_runtime = _predict(
        product, contract, crohme_rows, direct_rows, product_configuration,
        device, args.predict_batch_size,
    )
    crohme_baseline_predictions = {
        str(row["record_id"]): str(row["final_topk"][0]) for row in crohme_rows
    }
    crohme_baseline = _metrics(crohme_rows, crohme_baseline_predictions)
    crohme_metrics = _metrics(crohme_rows, crohme_predictions)
    direct_audit = masked._candidate_audit(direct_rows, direct_predictions)
    direct_refit_audit = masked._candidate_audit(direct_rows, direct_refit_predictions)
    crohme_audit = masked._candidate_audit(crohme_rows, crohme_predictions)
    _assert_candidate_contract("direct writer-LOO", direct_audit)
    _assert_candidate_contract("direct product refit", direct_refit_audit)
    _assert_candidate_contract("CROHME transfer", crohme_audit)

    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_config": MODEL_CONFIG,
        "special_ids": SPECIAL_IDS,
        "roles": list(context.ROLES),
        "math_labels": labels,
        "semantic_role_by_label": {
            label: context._semantic_role(label) for label in labels
        },
        "hwr_checkpoint_sha256": masked._sha256(hwr_checkpoint),
        "candidate_input_sha256": {
            "direct": masked._sha256(direct_path),
        },
        "external_pretrained_weights": False,
        "external_text_corpus": False,
        "training_data": (
            f"project-owned {direct_formula_count} formulas / {len(direct_rows)} glyphs / "
            f"{direct_writer_count} writers plus deterministic owned formula DSL"
        ),
        "synthetic_seed": SEED,
        "synthetic_train_examples": args.synthetic_train_examples,
        "real_repeat": REAL_REPEAT,
        "exact_loss_weight": EXACT_LOSS_WEIGHT,
        "selected_epochs": product_epochs,
        "selected_epochs_source": "median of outer writer-LOO selected epochs",
        "configuration": product_configuration,
        "configuration_source": "median of outer writer-LOO selected configurations",
        "candidate_probability_ratio_floor_source": (
            "minimum HWR probability ratio among correct outer writer-LOO overrides"
        ),
        "role_grammar": context._fit_role_grammar(direct_rows),
        "label_support": dict(Counter(str(row["label"]) for row in direct_rows)),
        "top1_reliability": context._fit_top1_reliability(direct_rows),
        "state_dict": {
            key: value.detach().cpu() for key, value in product.state_dict().items()
        },
    }
    torch.save(payload, checkpoint_path)
    checkpoint_sha = masked._sha256(checkpoint_path)
    del product
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    reloaded, reloaded_contract, reloaded_payload = load_owned_formula_context(
        checkpoint_path, hwr_checkpoint, device
    )
    reloaded_direct, _ = decide_owned_formula_rows(
        reloaded, reloaded_contract, reloaded_payload, direct_rows,
        device, args.predict_batch_size,
    )
    reloaded_crohme, _ = decide_owned_formula_rows(
        reloaded, reloaded_contract, reloaded_payload, crohme_rows,
        device, args.predict_batch_size,
    )
    direct_reload_mismatches = sum(
        reloaded_direct[record_id] != prediction
        for record_id, prediction in direct_refit_predictions.items()
    )
    crohme_reload_mismatches = sum(
        reloaded_crohme[record_id] != prediction
        for record_id, prediction in crohme_predictions.items()
    )
    if direct_reload_mismatches or crohme_reload_mismatches:
        raise AssertionError(
            f"owned checkpoint reload mismatch: direct={direct_reload_mismatches}, "
            f"crohme={crohme_reload_mismatches}"
        )

    masked._write_prediction_rows(
        output / "direct_writer_loo_predictions.jsonl.gz", direct_rows, direct_predictions
    )
    masked._write_prediction_rows(
        output / "direct_product_refit_predictions.jsonl.gz",
        direct_rows, direct_refit_predictions,
    )
    masked._write_prediction_rows(
        output / "crohme_transfer_predictions.jsonl.gz", crohme_rows, crohme_predictions
    )

    r2_direct_field, r2_hash = _reference(args.r2_report, "direct_writer_loo")
    r2_crohme_field, _ = _reference(args.r2_report, "crohme_transfer")
    r7_direct_field, r7_hash = _reference(args.r7_report, "direct_writer_loo")
    r7_crohme_field, _ = _reference(args.r7_report, "crohme_transfer")
    r2_direct = r2_direct_field.get("context_role_finalizer") if r2_direct_field else None
    r2_crohme = r2_crohme_field.get("context_role_finalizer") if r2_crohme_field else None
    r7_direct = r7_direct_field.get("distilled_context") if r7_direct_field else None
    r7_crohme = r7_crohme_field.get("distilled_context") if r7_crohme_field else None
    direct_reference_comparable = all(
        reference is not None and int(reference.get("all_records", -1)) == len(direct_rows)
        for reference in (r2_direct, r7_direct)
    )
    if not direct_reference_comparable:
        r2_direct = None
        r7_direct = None
    metrics = ("all_top1", "strict_micro_top1", "strict_macro_top1", "formula_exact")
    research_gate = (
        all(direct_metrics[key] >= direct_baseline[key] for key in metrics)
        and direct_audit["candidate_preservation_rate"] == 1.0
    )
    report = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "architecture": {
            "model": "random-initialized BERT encoder with role and 372-class heads",
            "layers": MODEL_CONFIG["num_hidden_layers"],
            "hidden_size": MODEL_CONFIG["hidden_size"],
            "attention_heads": MODEL_CONFIG["num_attention_heads"],
            "parameters": sum(value.numel() for value in payload["state_dict"].values()),
            "external_pretrained_weights": False,
            "external_text_corpus": False,
            "candidate_contract": "HWR Top-5 only; no token, deletion, or grouping mutation",
        },
        "data_admission": {
            "project": {
                "ownership_verified_formulas": direct_formula_count,
                "glyphs": len(direct_rows),
                "writers": direct_writer_count,
                "evaluation_partitions": direct_partitions,
            },
            "synthetic": {
                "source": "deterministic in-repository formula DSL",
                "training_examples_per_fit": args.synthetic_train_examples,
                "validation_examples_per_selection": args.synthetic_valid_examples,
                "seed": SEED,
                "external_corpus": False,
            },
            "crohme": "CC BY-NC transfer diagnostic only; never used for training or selection",
            "training_data_guard": zero_crohme_training_manifest(
                admitted_sources={
                    "project-owned-direct": len(direct_rows),
                    "deterministic-owned-formula-dsl": args.synthetic_train_examples,
                },
            ),
        },
        "training": {
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "exact_loss_weight": EXACT_LOSS_WEIGHT,
            "real_repeat": REAL_REPEAT,
            "selection": "formula-disjoint project validation inside every outer writer fold",
            "writer_loo_folds": folds,
            "product_inner_selection": {
                "executed": False,
                "reason": "outer writer-LOO medians own runtime epoch and configuration",
            },
            "product_runtime_epochs": product_epochs,
            "product_epoch_source": (
                "median of outer writer-LOO selected epochs; no separate product selection"
            ),
            "product_runtime_configuration": product_configuration,
            "candidate_probability_ratio_floor": probability_ratio_floor,
            "candidate_probability_ratio_floor_source": (
                "minimum HWR probability ratio among correct outer writer-LOO overrides; "
                "project-owned data only"
            ),
            "product_configuration_source": (
                "median of outer writer-LOO selected configurations; "
                "no separate product selection"
            ),
            "product_fit_final_objective": product_losses[-1],
        },
        "evaluation": {
            "direct_writer_loo": {
                "baseline": direct_baseline,
                "owned_context": direct_metrics,
                "delta": masked._metric_delta(direct_metrics, direct_baseline),
                "candidate_audit": direct_audit,
                "r2_reference": r2_direct,
                "r7_noncommercial_reference": r7_direct,
                "prior_reference_comparable": direct_reference_comparable,
                "prior_reference_note": (
                    "same direct denominator" if direct_reference_comparable
                    else "omitted because prior reports used a different direct dataset"
                ),
            },
            "direct_product_refit_resubstitution": {
                "warning": "training-set fit only; not acceptance evidence",
                "metrics": _metrics(direct_rows, direct_refit_predictions),
                "runtime_audit": direct_refit_runtime,
                "candidate_audit": direct_refit_audit,
            },
            "crohme_transfer": {
                "baseline": crohme_baseline,
                "owned_context": crohme_metrics,
                "delta": masked._metric_delta(crohme_metrics, crohme_baseline),
                "runtime_audit": crohme_runtime,
                "candidate_audit": crohme_audit,
                "r2_reference": r2_crohme,
                "r7_noncommercial_reference": r7_crohme,
            },
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "bytes": checkpoint_path.stat().st_size,
            "sha256": checkpoint_sha,
            "direct_product_refit_reload_mismatches": direct_reload_mismatches,
            "crohme_transfer_reload_mismatches": crohme_reload_mismatches,
        },
        "provenance": {
            "hwr_checkpoint_sha256": masked._sha256(hwr_checkpoint),
            "direct_candidate_sha256": masked._sha256(direct_path),
            "crohme_candidate_sha256": masked._sha256(crohme_path),
            "r2_report_sha256": r2_hash,
            "r7_report_sha256": r7_hash,
        },
        "decision": {
            "research_gate_passed": research_gate,
            "research_gate_sources": "project-owned outer writer-LOO only",
            "crohme_metrics_used_for_gate": False,
            "crohme_role": "post-checkpoint report-only no-gradient validation",
            "commercial_context_training_rights_gate_passed": True,
            "commercial_accuracy_gate_passed": False,
            "automatic_default_replacement": False,
            "runtime_status": "shadow owned formula-context candidate",
            "reason": (
                "training rights are clean, but both current evaluations are repeatedly observed "
                "and fresh project-owned writer/formula acceptance is still required"
            ),
        },
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _event(
        "owned_context_complete", report=str(report_path), checkpoint=str(checkpoint_path),
        direct_top1=direct_metrics["all_top1"], crohme_top1=crohme_metrics["all_top1"],
        research_gate=research_gate,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
