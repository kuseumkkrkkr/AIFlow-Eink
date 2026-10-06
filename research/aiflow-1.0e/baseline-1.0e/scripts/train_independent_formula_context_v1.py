#!/usr/bin/env python3
"""Tune a standalone embedding-based typo corrector for formula context.

The shape HWR checkpoint, candidate order, and stroke grouping stay frozen.
This model starts from a pinned Apache-2.0 BERT-Tiny language encoder, never
from an AIFlow context checkpoint.  Project-owned clean formulae are paired
with context corrupted from outer-fold HWR confusions, and the final decision
is always restricted to the frozen HWR Top-5 candidates.
"""

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
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

import numpy as np
import torch
import torch.nn.functional as F

from character_tensor_v1 import ROOT, _json_lines
from evaluate_homograph_context_reranker_v1 import _metrics
import train_masked_context_reranker_v1 as masked


SCHEMA = "aiflow-independent-embedding-typo-context/v1"
SEED = 20260820
ARCHITECTURE = {
    "family": "candidate_preserving_masked_token_embedding_corrector",
    "base_model": masked.MODEL_ID,
    "base_revision": masked.MODEL_REVISION,
    "hidden_size": 128,
    "transformer_layers": 2,
    "attention_heads": 2,
    "class_embedding_count": 372,
    "class_embedding_tied_to_decoder": True,
}
LAMBDA_GRID = (0.0, 0.025, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75)

DEFAULT_HWR = Path(
    r"D:\AIFlow-Workspace\PrivateData\math-ink-data-collector\derived"
    r"\hwr-head-calibration-20260819-r1\final_all_writers_steps250_lr1e-3"
    r"\project_symbol_head_checkpoint.pt"
)
DEFAULT_DIRECT = Path(
    r"D:\AIFlow-Workspace\PrivateData\math-ink-data-collector\derived"
    r"\hwr-head-calibration-20260819-r1\expanded_writer_loo_candidates_r1.jsonl.gz"
)
DEFAULT_CROHME = (
    ROOT / "artifacts" / "homograph_context_expanded_hwr_20260820_r1_shadow"
    / "crohme_candidates.jsonl.gz"
)
DEFAULT_PROMPT_CORPUS = (
    ROOT / "artifacts" / "prompt_context_corpus_20260820_r2"
    / "prompt_context_corpus.jsonl"
)
DEFAULT_PROMPT_AUDIT = (
    ROOT / "artifacts" / "prompt_context_corpus_20260820_r2"
    / "prompt_context_corpus_audit.json"
)
DEFAULT_R6_CONTEXT = (
    ROOT / "artifacts" / "owned_formula_context_20260822_r6"
    / "owned_formula_context_product.pt"
)
DEFAULT_R6_HWR = (
    ROOT / "artifacts" / "unified_head_20260814"
    / "uniform_time_final_all_writers" / "project_symbol_head_checkpoint.pt"
)
DEFAULT_OUTPUT = (
    ROOT / "artifacts" / "independent_embedding_typo_context_20260820_r2_shadow"
)
DEFAULT_PRETRAINED = masked.DEFAULT_PRETRAINED


def _event(name: str, **values: object) -> None:
    print(json.dumps({"event": name, **values}, ensure_ascii=False), flush=True)


def _seed(salt: str) -> int:
    digest = hashlib.sha256(salt.encode("utf-8")).digest()
    return SEED + int.from_bytes(digest[:4], "big") % 100_000


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


def _new_model(
    pretrained: Path, labels: list[str], device: torch.device, seed: int
) -> tuple[object, dict]:
    """Build only from the pinned generic encoder, never an AIFlow checkpoint."""
    model, _tokenizer, contract = masked._build_model(
        pretrained, labels, device, seed
    )
    return model, contract


def _formula_sequences(rows: list[dict]) -> set[tuple[str, ...]]:
    return {
        tuple(str(row["label"]) for row in sequence)
        for sequence in masked._formulae(rows).values()
    }


def _formula_sequence_map(rows: list[dict]) -> dict[str, tuple[str, ...]]:
    return {
        str(formula_id): tuple(str(row["label"]) for row in sequence)
        for formula_id, sequence in masked._formulae(rows).items()
    }


def _sequence_group_split(
    rows: list[dict], salt: str
) -> tuple[list[dict], list[dict]]:
    """Keep duplicate token sequences in exactly one inner partition."""
    sequence_by_formula = _formula_sequence_map(rows)
    sequences = sorted(
        set(sequence_by_formula.values()),
        key=lambda sequence: hashlib.sha256(
            (salt + "\0" + json.dumps(sequence, ensure_ascii=False)).encode("utf-8")
        ).digest(),
    )
    if len(sequences) < 2:
        raise ValueError("inner selection requires at least two token sequences")
    held_count = min(len(sequences) - 1, max(1, round(len(sequences) * 0.2)))
    held_sequences = set(sequences[:held_count])
    fit = [
        row for row in rows
        if sequence_by_formula[str(row["formula_id"])] not in held_sequences
    ]
    validation = [
        row for row in rows
        if sequence_by_formula[str(row["formula_id"])] in held_sequences
    ]
    if (
        not fit or not validation
        or {str(row["formula_id"]) for row in fit}
        & {str(row["formula_id"]) for row in validation}
        or _formula_sequences(fit) & _formula_sequences(validation)
    ):
        raise AssertionError("invalid token-sequence-grouped inner split")
    return fit, validation


def _prompt_rows(
    path: Path, labels: list[str], excluded_sequences: set[tuple[str, ...]]
) -> tuple[list[dict], dict]:
    allowed = set(labels)
    output = []
    seen = set()
    unsupported = Counter()
    overlap_excluded = 0
    for source in _json_lines(path):
        if source.get("commercial_training_rights") != (
            "project-owned collector prompt catalog"
        ):
            raise ValueError("prompt formula lacks project-owned training rights")
        source_id = str(source["formula_id"])
        if source_id in seen:
            raise ValueError(f"duplicate prompt formula: {source_id}")
        seen.add(source_id)
        tokens = tuple(str(value) for value in source["labels"])
        unknown = sorted(set(tokens) - allowed)
        if unknown:
            unsupported.update(unknown)
            continue
        if tokens in excluded_sequences:
            overlap_excluded += 1
            continue
        formula_id = f"independent-prompt::{source_id}"
        length = len(tokens)
        if length < 1:
            raise ValueError(f"empty prompt formula: {source_id}")
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
                    "center_x": (index + 0.5) * width,
                    "center_y": 0.5,
                    "width_rel": width,
                    "height_rel": 1.0,
                },
                "context": {"index": index, "length": length},
            })
    if not output:
        raise ValueError("independent prompt corpus is empty")
    return output, {
        "source_formulas": len(seen),
        "admitted_formulas": len(masked._formulae(output)),
        "admitted_records": len(output),
        "evaluation_sequence_overlap_excluded": overlap_excluded,
        "unsupported_labels": dict(sorted(unsupported.items())),
    }


def _confusion_distribution(
    direct_rows: list[dict], labels: list[str]
) -> dict[str, tuple[list[str], list[float]]]:
    """Estimate typo noise only from the current outer-fold training HWR."""
    allowed = set(labels)
    counts: dict[str, Counter] = defaultdict(Counter)
    for row in direct_rows:
        truth = str(row["label"])
        for candidate, probability in zip(
            row["final_topk"], row["final_topk_probabilities"], strict=True
        ):
            candidate = str(candidate)
            if candidate != truth and candidate in allowed:
                counts[truth][candidate] += max(float(probability), 1e-6)
    return {
        truth: (
            sorted(values),
            [float(values[token]) for token in sorted(values)],
        )
        for truth, values in counts.items() if values
    }


def _truth_context(rows: list[dict]) -> dict[str, str]:
    return {str(row["record_id"]): str(row["label"]) for row in rows}


def _corrupt_context(
    rows: list[dict], confusions: dict[str, tuple[list[str], list[float]]],
    noise_rate: float, seed: int,
) -> tuple[dict[str, str], int]:
    rng = random.Random(seed)
    output = _truth_context(rows)
    changed = 0
    for row in rows:
        truth = str(row["label"])
        choices = confusions.get(truth)
        if choices and rng.random() < noise_rate:
            output[str(row["record_id"])] = rng.choices(
                choices[0], weights=choices[1], k=1
            )[0]
            changed += 1
    return output, changed


def _training_data(
    prompt_rows: list[dict], direct_rows: list[dict], contract: dict,
    direct_repeat: int, noise_rate: float, noise_replicas: int, seed: int,
) -> tuple[dict[str, torch.Tensor], dict]:
    labels = list(contract["labels"])
    confusions = _confusion_distribution(direct_rows, labels)
    sources: list[tuple[dict, int, str, int]] = []

    def add(
        rows: list[dict], context: dict[str, str] | None, repeat: int,
        name: str, changed: int = 0,
    ) -> None:
        if rows:
            sources.append((masked._pack(rows, contract, context), repeat, name, changed))

    add(prompt_rows, _truth_context(prompt_rows), 1, "prompt_clean")
    for replica in range(noise_replicas):
        context, changed = _corrupt_context(
            prompt_rows, confusions, noise_rate,
            _seed(f"prompt-noise-{seed}-{replica}"),
        )
        add(prompt_rows, context, 1, f"prompt_noisy_{replica + 1}", changed)
    if direct_rows:
        add(direct_rows, _truth_context(direct_rows), direct_repeat, "direct_clean")
        add(direct_rows, None, direct_repeat, "direct_runtime_hwr_top1")
        for replica in range(noise_replicas):
            context, changed = _corrupt_context(
                direct_rows, confusions, noise_rate,
                _seed(f"direct-noise-{seed}-{replica}"),
            )
            add(
                direct_rows, context, direct_repeat,
                f"direct_noisy_{replica + 1}", changed,
            )
    width = max(pack["input_ids"].shape[1] for pack, _, _, _ in sources)
    combined: dict[str, list[torch.Tensor]] = {
        key: [] for key in (
            "input_ids", "attention_mask", "mask_positions", "targets"
        )
    }
    source_audit = []
    for packed, repeat, name, changed in sources:
        input_ids = F.pad(
            packed["input_ids"], (0, width - packed["input_ids"].shape[1]),
            value=int(contract["pad_id"]),
        )
        attention = F.pad(
            packed["attention_mask"],
            (0, width - packed["attention_mask"].shape[1]), value=0,
        )
        combined["input_ids"].append(input_ids.repeat((repeat, 1)))
        combined["attention_mask"].append(attention.repeat((repeat, 1)))
        combined["mask_positions"].append(packed["mask_positions"].repeat(repeat))
        combined["targets"].append(packed["targets"].repeat(repeat))
        source_audit.append({
            "name": name,
            "records": len(packed["rows"]),
            "repeat": repeat,
            "effective_records": len(packed["rows"]) * repeat,
            "corrupted_context_tokens": changed,
        })
    output = {key: torch.cat(parts) for key, parts in combined.items()}
    counts = torch.bincount(
        output["targets"], minlength=len(contract["labels"])
    ).to(torch.float32)
    output["weights"] = counts[output["targets"]].clamp_min(1.0).rsqrt()
    output["weights"] /= output["weights"].mean()
    return output, {
        "noise_rate": noise_rate,
        "noise_replicas": noise_replicas,
        "confusion_source_records": len(direct_rows),
        "truth_labels_with_confusions": len(confusions),
        "confusion_edges": sum(len(values[0]) for values in confusions.values()),
        "sources": source_audit,
        "effective_training_records": len(output["targets"]),
    }


def _train_epoch(
    model, contract: dict, data: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer, device: torch.device, batch_size: int,
    generator: torch.Generator,
) -> float:
    model.train()
    class_ids = torch.tensor(
        contract["class_ids"], dtype=torch.long, device=device
    )
    order = torch.randperm(len(data["targets"]), generator=generator)
    numerator = denominator = 0.0
    for start in range(0, len(order), batch_size):
        indices = order[start:start + batch_size]
        optimizer.zero_grad(set_to_none=True)
        logits = masked._class_logits(
            model,
            data["input_ids"][indices].to(device),
            data["attention_mask"][indices].to(device),
            data["mask_positions"][indices].to(device),
            class_ids,
        )
        targets = data["targets"][indices].to(device)
        weights = data["weights"][indices].to(device)
        per_record = F.cross_entropy(logits, targets, reduction="none")
        loss = (per_record * weights).sum() / weights.sum()
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite independent context loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        numerator += float((per_record.detach() * weights).sum())
        denominator += float(weights.sum())
    model.eval()
    return numerator / denominator


@torch.inference_mode()
def _context_log_probabilities(
    model, contract: dict, rows: list[dict],
    device: torch.device, batch_size: int,
    context_tokens: dict[str, str] | None = None,
) -> dict[str, np.ndarray]:
    if context_tokens is None:
        return masked._context_log_probabilities(
            model, contract, rows, device, batch_size
        )
    packed = masked._pack(rows, contract, context_tokens)
    class_ids = torch.tensor(
        contract["class_ids"], dtype=torch.long, device=device
    )
    output = {}
    model.eval()
    for start in range(0, len(rows), batch_size):
        stop = min(start + batch_size, len(rows))
        logits = masked._class_logits(
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


def _fused_predictions(
    rows: list[dict], context_scores: dict[str, np.ndarray],
    labels: list[str], lambda_value: float,
) -> dict[str, str]:
    if not math.isfinite(lambda_value) or lambda_value < 0.0:
        raise ValueError("context lambda must be finite and non-negative")
    predictions = masked._fused_predictions(
        rows, context_scores, labels, lambda_value
    )
    # A one-token formula has no neighbouring evidence.  Letting a context
    # model change it is necessarily an embedding prior, not context repair.
    for row in rows:
        if int(row["context"]["length"]) < 2:
            predictions[str(row["record_id"])] = str(row["final_topk"][0])
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
        -int(metrics["changed"]),
        -float(trial["lambda"]),
        -int(trial["epoch"]),
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
        -int(metrics["changed"]),
        -float(trial["lambda"]),
        -int(trial["epoch"]),
    )


def _select_fold_configuration(
    prompt_rows: list[dict], training_rows: list[dict], labels: list[str],
    device: torch.device, args, salt: str,
) -> dict:
    fit_rows, validation_rows = _sequence_group_split(training_rows, salt)
    seed = _seed(f"select-{salt}")
    model, contract = _new_model(args.pretrained, labels, device, seed)
    data, data_audit = _training_data(
        prompt_rows, fit_rows, contract, args.direct_repeat,
        args.noise_rate, args.noise_replicas, seed,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    generator = torch.Generator().manual_seed(seed)
    baseline_predictions = {
        str(row["record_id"]): str(row["final_topk"][0])
        for row in validation_rows
    }
    baseline = _metrics(validation_rows, baseline_predictions)
    trials = []
    losses = []
    for epoch in range(1, args.max_epochs + 1):
        losses.append(_train_epoch(
            model, contract, data, optimizer, device, args.batch_size, generator
        ))
        context_scores = _context_log_probabilities(
            model, contract, validation_rows, device, args.predict_batch_size
        )
        for lambda_value in LAMBDA_GRID:
            predictions = _fused_predictions(
                validation_rows, context_scores, labels, lambda_value
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
                "epoch": epoch,
                "lambda": lambda_value,
                "admissible": admissible,
                "metrics": _compact_metrics(metrics),
            })
        _event(
            "independent_selection_epoch", salt=salt, epoch=epoch,
            objective=losses[-1],
            best_formula=max(
                trial["metrics"]["formula_exact"]
                for trial in trials if trial["epoch"] == epoch
            ),
        )
    safe = max(trials, key=_safe_key)
    accuracy = max(trials, key=_accuracy_key)
    del model, data, optimizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "fit_records": len(fit_rows),
        "fit_formulas": len(masked._formulae(fit_rows)),
        "validation_records": len(validation_rows),
        "validation_formulas": len(masked._formulae(validation_rows)),
        "safe": safe,
        "accuracy": accuracy,
        "losses": losses,
        "training_data": data_audit,
        "trials": trials,
    }


def _fit_fold(
    prompt_rows: list[dict], training_rows: list[dict], held_rows: list[dict],
    labels: list[str], selected: dict[str, dict], device: torch.device, args,
    salt: str,
) -> tuple[dict[str, dict[str, str]], list[float]]:
    seed = _seed(f"fit-{salt}")
    model, contract = _new_model(args.pretrained, labels, device, seed)
    data, _data_audit = _training_data(
        prompt_rows, training_rows, contract, args.direct_repeat,
        args.noise_rate, args.noise_replicas, seed,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    generator = torch.Generator().manual_seed(seed)
    required_epochs = {int(config["epoch"]) for config in selected.values()}
    predictions: dict[str, dict[str, str]] = {}
    losses = []
    for epoch in range(1, max(required_epochs) + 1):
        losses.append(_train_epoch(
            model, contract, data, optimizer, device, args.batch_size, generator
        ))
        names = [
            name for name, config in selected.items()
            if int(config["epoch"]) == epoch
        ]
        if not names:
            continue
        context_scores = _context_log_probabilities(
            model, contract, held_rows, device, args.predict_batch_size
        )
        for name in names:
            predictions[name] = _fused_predictions(
                held_rows, context_scores, labels,
                float(selected[name]["lambda"]),
            )
    del model, data, optimizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if set(predictions) != set(selected):
        raise AssertionError("independent fold configuration coverage mismatch")
    return predictions, losses


def _writer_loo(
    prompt_rows: list[dict], rows: list[dict], labels: list[str],
    device: torch.device, args,
) -> tuple[dict[str, dict[str, str]], list[dict]]:
    predictions = {"safe": {}, "accuracy": {}}
    folds = []
    writers = sorted({str(row["writer_group"]) for row in rows})
    if len(writers) < 2:
        raise ValueError("independent context requires at least two writers")
    for fold_index, writer in enumerate(writers, start=1):
        raw_training_rows = [
            row for row in rows if str(row["writer_group"]) != writer
        ]
        held_rows = [
            row for row in rows if str(row["writer_group"]) == writer
        ]
        held_sequences = _formula_sequences(held_rows)
        raw_sequence_by_formula = _formula_sequence_map(raw_training_rows)
        excluded_formulae = {
            formula_id for formula_id, sequence in raw_sequence_by_formula.items()
            if sequence in held_sequences
        }
        training_rows = [
            row for row in raw_training_rows
            if str(row["formula_id"]) not in excluded_formulae
        ]
        if _formula_sequences(training_rows) & held_sequences:
            raise AssertionError("token-sequence leakage in independent writer-LOO")
        training_formulas = {str(row["formula_id"]) for row in training_rows}
        held_formulas = {str(row["formula_id"]) for row in held_rows}
        if training_formulas & held_formulas:
            raise AssertionError("formula leakage in independent writer-LOO")
        selection = _select_fold_configuration(
            prompt_rows, training_rows, labels, device, args,
            f"writer-{writer}",
        )
        selected = {
            name: selection[name] for name in ("safe", "accuracy")
        }
        fold_predictions, refit_losses = _fit_fold(
            prompt_rows, training_rows, held_rows, labels, selected, device,
            args, f"writer-{writer}",
        )
        fold_report = {
            "held_writer_group": writer,
            "raw_training_records": len(raw_training_rows),
            "sequence_overlap_formulas_excluded": len(excluded_formulae),
            "training_records": len(training_rows),
            "training_formulas": len(training_formulas),
            "held_records": len(held_rows),
            "held_formulas": len(held_formulas),
            "selection": selection,
            "refit_losses": refit_losses,
            "held": {},
        }
        for name, result in fold_predictions.items():
            predictions[name].update(result)
            audit = masked._candidate_audit(held_rows, result)
            if audit["new_tokens"] or audit["grouping_mutations"]:
                raise AssertionError("independent context violated candidate contract")
            fold_report["held"][name] = {
                "metrics": _metrics(held_rows, result),
                "candidate_audit": audit,
            }
        folds.append(fold_report)
        _event(
            "independent_writer_fold", fold=fold_index, writer=writer,
            safe_top1=fold_report["held"]["safe"]["metrics"]["all_top1"],
            accuracy_top1=(
                fold_report["held"]["accuracy"]["metrics"]["all_top1"]
            ),
        )
    expected = {str(row["record_id"]) for row in rows}
    if any(set(values) != expected for values in predictions.values()):
        raise AssertionError("independent writer-LOO prediction coverage mismatch")
    return predictions, folds


def _aggregate_configuration(folds: list[dict], name: str) -> dict:
    epochs = [int(fold["selection"][name]["epoch"]) for fold in folds]
    lambdas = [float(fold["selection"][name]["lambda"]) for fold in folds]
    return {"epoch": int(median(epochs)), "lambda": float(median(lambdas))}


def _fit_product_states(
    prompt_rows: list[dict], direct_rows: list[dict], labels: list[str],
    configurations: dict[str, dict], device: torch.device, args,
) -> tuple[dict[str, dict[str, torch.Tensor]], list[float]]:
    seed = _seed("independent-product")
    model, contract = _new_model(args.pretrained, labels, device, seed)
    data, _data_audit = _training_data(
        prompt_rows, direct_rows, contract, args.direct_repeat,
        args.noise_rate, args.noise_replicas, seed,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    generator = torch.Generator().manual_seed(seed)
    epochs_by_name = {
        name: int(config["epoch"]) for name, config in configurations.items()
    }
    states = {}
    losses = []
    for epoch in range(1, max(epochs_by_name.values()) + 1):
        losses.append(_train_epoch(
            model, contract, data, optimizer, device, args.batch_size, generator
        ))
        for name, selected_epoch in epochs_by_name.items():
            if selected_epoch == epoch:
                states[name] = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
        _event(
            "independent_product_epoch", epoch=epoch, objective=losses[-1]
        )
    del model, data, optimizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if set(states) != set(configurations):
        raise AssertionError("independent product checkpoint coverage mismatch")
    return states, losses


def _predict_state(
    state: dict[str, torch.Tensor], configuration: dict, rows: list[dict],
    labels: list[str], pretrained: Path, device: torch.device, batch_size: int,
) -> tuple[dict[str, str], dict]:
    model, contract = _new_model(pretrained, labels, device, SEED)
    model.load_state_dict(state, strict=True)
    started = time.perf_counter()
    context_scores = _context_log_probabilities(
        model, contract, rows, device, batch_size
    )
    predictions = _fused_predictions(
        rows, context_scores, labels, float(configuration["lambda"])
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return predictions, {
        "records": len(rows),
        "elapsed_ms": elapsed_ms,
        "milliseconds_per_record": elapsed_ms / len(rows),
    }


def _checkpoint_payload(
    state: dict[str, torch.Tensor], configuration: dict, labels: list[str],
    hwr_checkpoint: Path, prompt_corpus: Path, direct_candidates: Path,
    pretrained: Path, training: dict,
) -> dict:
    return {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "architecture": ARCHITECTURE,
        "model_id": masked.MODEL_ID,
        "model_revision": masked.MODEL_REVISION,
        "model_file_sha256": masked._verify_pretrained(pretrained),
        "pretrained_license": "Apache-2.0",
        "relations": list(masked.RELATIONS),
        "class_tokens": masked._class_tokens(labels),
        "relation_tokens": masked._relation_tokens(),
        "math_labels": labels,
        "hwr_checkpoint_sha256": masked._sha256(hwr_checkpoint),
        "prompt_corpus_sha256": masked._sha256(prompt_corpus),
        "direct_candidates_sha256": masked._sha256(direct_candidates),
        "configuration": configuration,
        "training": training,
        "external_pretrained_weights": True,
        "parent_context_checkpoint": None,
        "warm_start": False,
        "candidate_policy": (
            "argmax log(P_HWR)+lambda*log(P_embedding_context) inside frozen HWR Top-5; "
            "single-token formulae lock frozen HWR Top-1"
        ),
        "state_dict": state,
    }


def load_independent_formula_context(
    pretrained: Path, checkpoint_path: Path, hwr_checkpoint: Path,
    device: torch.device,
) -> tuple[object, dict, dict]:
    pretrained = _d_path(pretrained, "pinned pretrained encoder", file=False)
    checkpoint_path = _d_path(checkpoint_path, "independent context checkpoint")
    hwr_checkpoint = _d_path(hwr_checkpoint, "HWR checkpoint")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    labels = masked._labels(hwr_checkpoint)
    configuration = payload.get("configuration")
    if (
        payload.get("schema") != SCHEMA
        or payload.get("architecture") != ARCHITECTURE
        or payload.get("model_id") != masked.MODEL_ID
        or payload.get("model_revision") != masked.MODEL_REVISION
        or payload.get("model_file_sha256") != masked._verify_pretrained(pretrained)
        or payload.get("pretrained_license") != "Apache-2.0"
        or payload.get("relations") != list(masked.RELATIONS)
        or payload.get("math_labels") != labels
        or payload.get("hwr_checkpoint_sha256")
        != masked._sha256(hwr_checkpoint)
        or payload.get("external_pretrained_weights") is not True
        or payload.get("parent_context_checkpoint") is not None
        or payload.get("warm_start") is not False
        or not isinstance(configuration, dict)
        or set(configuration) != {"epoch", "lambda"}
        or int(configuration["epoch"]) < 1
        or float(configuration["lambda"]) < 0.0
    ):
        raise ValueError("independent formula-context checkpoint contract mismatch")
    model, contract = _new_model(pretrained, labels, device, SEED)
    if (
        payload.get("class_tokens") != contract["class_tokens"]
        or payload.get("relation_tokens") != contract["relation_tokens"]
    ):
        raise ValueError("independent embedding token contract mismatch")
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, contract, payload


def decide_independent_formula_rows(
    model, contract: dict, payload: dict,
    rows: list[dict], device: torch.device, batch_size: int = 256,
) -> tuple[dict[str, str], dict]:
    context_scores = _context_log_probabilities(
        model, contract, rows, device, batch_size
    )
    predictions = _fused_predictions(
        rows, context_scores, contract["labels"],
        float(payload["configuration"]["lambda"]),
    )
    audit = masked._candidate_audit(rows, predictions)
    if audit["new_tokens"] or audit["grouping_mutations"]:
        raise AssertionError("loaded independent context violated candidate contract")
    return predictions, audit


def _r6_comparison(
    checkpoint: Path, hwr_checkpoint: Path, direct_rows: list[dict],
    crohme_rows: list[dict], device: torch.device, batch_size: int,
) -> dict:
    import train_owned_formula_context_v1 as owned

    model, contract, payload = owned.load_owned_formula_context(
        checkpoint, hwr_checkpoint, device
    )
    direct_predictions, direct_runtime = owned.decide_owned_formula_rows(
        model, contract, payload, direct_rows, device, batch_size
    )
    crohme_predictions, crohme_runtime = owned.decide_owned_formula_rows(
        model, contract, payload, crohme_rows, device, batch_size
    )
    new_rows = [
        row for row in direct_rows
        if str(row.get("evaluation_partition")) == "new_writer"
    ]
    return {
        "loaded_after_independent_training": True,
        "checkpoint_sha256": masked._sha256(checkpoint),
        "all_direct_warning": "contains rows used by the historical r6 fit",
        "all_direct": _metrics(direct_rows, direct_predictions),
        "new_writer": _metrics(new_rows, direct_predictions),
        "crohme_post_selection": _metrics(crohme_rows, crohme_predictions),
        "candidate_audit": {
            "direct": masked._candidate_audit(direct_rows, direct_predictions),
            "crohme": masked._candidate_audit(crohme_rows, crohme_predictions),
        },
        "runtime_audit": {
            "direct": direct_runtime,
            "crohme": crohme_runtime,
        },
    }


def _subset_metrics(
    rows: list[dict], predictions: dict[str, str], partition: str
) -> dict:
    selected = [
        row for row in rows
        if str(row.get("evaluation_partition")) == partition
    ]
    return _metrics(selected, predictions)


def _self_test(
    labels: list[str], pretrained: Path, device: torch.device
) -> None:
    model, contract = _new_model(pretrained, labels, device, SEED)
    rows = [
        {
            "record_id": "a", "formula_id": "f", "writer_group": "w",
            "label": "1", "final_topk": ["|", "1"],
            "final_topk_probabilities": [0.7, 0.3],
            "geometry": {
                "center_x": 0.25, "center_y": 0.5,
                "width_rel": 0.5, "height_rel": 1.0,
            },
            "context": {"index": 0, "length": 2},
        },
        {
            "record_id": "b", "formula_id": "f", "writer_group": "w",
            "label": "+", "final_topk": ["+", "1"],
            "final_topk_probabilities": [0.8, 0.2],
            "geometry": {
                "center_x": 0.75, "center_y": 0.5,
                "width_rel": 0.5, "height_rel": 1.0,
            },
            "context": {"index": 1, "length": 2},
        },
    ]
    data, audit = _training_data(rows, [], contract, 1, 0.2, 1, SEED)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    loss = _train_epoch(
        model, contract, data, optimizer, device, 2,
        torch.Generator().manual_seed(SEED),
    )
    assert math.isfinite(loss)
    context_scores = _context_log_probabilities(
        model, contract, rows, device, 2
    )
    predictions = _fused_predictions(rows, context_scores, labels, 0.0)
    assert predictions == {"a": "|", "b": "+"}
    assert masked._candidate_audit(rows, predictions)[
        "candidate_preservation_rate"
    ] == 1.0
    assert audit["effective_training_records"] == 4
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    assert 4_000_000 < parameter_count < 6_000_000


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hwr-checkpoint", type=Path, default=DEFAULT_HWR)
    parser.add_argument("--direct-candidates", type=Path, default=DEFAULT_DIRECT)
    parser.add_argument("--crohme-candidates", type=Path, default=DEFAULT_CROHME)
    parser.add_argument("--prompt-corpus", type=Path, default=DEFAULT_PROMPT_CORPUS)
    parser.add_argument("--prompt-audit", type=Path, default=DEFAULT_PROMPT_AUDIT)
    parser.add_argument("--pretrained", type=Path, default=DEFAULT_PRETRAINED)
    parser.add_argument(
        "--comparison-context-checkpoint", type=Path,
        default=DEFAULT_R6_CONTEXT,
    )
    parser.add_argument(
        "--comparison-hwr-checkpoint", type=Path, default=DEFAULT_R6_HWR
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-epochs", type=int, default=3)
    parser.add_argument("--direct-repeat", type=int, default=2)
    parser.add_argument("--noise-rate", type=float, default=0.2)
    parser.add_argument("--noise-replicas", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--predict-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if (
        min(
            args.max_epochs, args.direct_repeat, args.batch_size,
            args.predict_batch_size, args.noise_replicas,
        ) < 1
        or not 0.0 <= args.noise_rate <= 1.0
        or args.learning_rate <= 0.0
        or args.weight_decay < 0.0
    ):
        parser.error("epochs, repeats, batches, and learning rate must be positive")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    pretrained = _d_path(args.pretrained, "pinned pretrained encoder", file=False)
    pretrained_hashes = masked._verify_pretrained(pretrained)
    args.pretrained = pretrained
    hwr_checkpoint = _d_path(args.hwr_checkpoint, "HWR checkpoint")
    labels = masked._labels(hwr_checkpoint)
    if args.self_test:
        _self_test(labels, pretrained, device)
        _event("self_test", status="pass", device=str(device))
        return 0

    direct_path = assert_training_path_clean(
        _d_path(args.direct_candidates, "direct candidates"),
        "independent context direct fit candidates",
    )
    crohme_path = _d_path(args.crohme_candidates, "CROHME candidates")
    prompt_corpus = assert_training_path_clean(
        _d_path(args.prompt_corpus, "prompt corpus"),
        "independent context prompt fit corpus",
    )
    prompt_audit_path = _d_path(args.prompt_audit, "prompt audit")
    r6_context = _d_path(
        args.comparison_context_checkpoint, "r6 comparison checkpoint"
    )
    r6_hwr = _d_path(args.comparison_hwr_checkpoint, "r6 HWR checkpoint")
    output = _d_path(args.output, "output", file=False)
    if output.exists():
        parser.error(f"refusing to overwrite independent context output: {output}")

    immutable_paths = (
        hwr_checkpoint, direct_path, crohme_path, prompt_corpus,
        prompt_audit_path, r6_context, r6_hwr,
    ) + tuple(pretrained / name for name in masked.EXPECTED_MODEL_FILES)
    hashes_before = {
        str(path): masked._sha256(path) for path in immutable_paths
    }
    direct_rows = list(_json_lines(direct_path))
    assert_training_rows_clean(direct_rows)
    crohme_rows = list(_json_lines(crohme_path))
    if not direct_rows or not crohme_rows:
        raise ValueError("independent context candidate caches are empty")
    if len({str(row["record_id"]) for row in direct_rows}) != len(direct_rows):
        raise ValueError("duplicate direct candidate record id")
    prompt_audit = json.loads(prompt_audit_path.read_text(encoding="utf-8"))
    if (
        prompt_audit.get("schema") != "aiflow-owned-prompt-context-audit/v1"
        or prompt_audit.get("contracts", {}).get("commercial_training_rights")
        is not True
        or prompt_audit.get("contracts", {}).get("raster_images") is not False
        or prompt_audit.get("contracts", {}).get("raw_ink") is not False
    ):
        raise ValueError("independent prompt-corpus rights contract failed")
    # Only project-owned direct rows may influence fit-corpus admission.  CROHME
    # is read later for report-only transfer validation and cannot filter fit data.
    evaluation_sequences = _formula_sequences(direct_rows)
    prompt_rows, prompt_admission = _prompt_rows(
        prompt_corpus, labels, evaluation_sequences
    )
    assert_training_rows_clean(prompt_rows)
    if _formula_sequences(prompt_rows) & evaluation_sequences:
        raise AssertionError("independent prompt/evaluation sequence overlap")

    oof_predictions, folds = _writer_loo(
        prompt_rows, direct_rows, labels, device, args
    )
    oof_evaluation = {}
    for name, predictions in oof_predictions.items():
        audit = masked._candidate_audit(direct_rows, predictions)
        if audit["new_tokens"] or audit["grouping_mutations"]:
            raise AssertionError("independent OOF candidate contract failed")
        oof_evaluation[name] = {
            "all": _metrics(direct_rows, predictions),
            "legacy47": _subset_metrics(direct_rows, predictions, "legacy47"),
            "new_known_writer": _subset_metrics(
                direct_rows, predictions, "new_known_writer"
            ),
            "new_writer": _subset_metrics(
                direct_rows, predictions, "new_writer"
            ),
            "candidate_audit": audit,
        }
    configurations = {
        name: _aggregate_configuration(folds, name)
        for name in ("safe", "accuracy")
    }
    product_states, product_losses = _fit_product_states(
        prompt_rows, direct_rows, labels, configurations, device, args
    )
    product_evaluation = {}
    in_memory_predictions = {}
    for name, state in product_states.items():
        direct_predictions, direct_runtime = _predict_state(
            state, configurations[name], direct_rows, labels, pretrained, device,
            args.predict_batch_size,
        )
        crohme_predictions, crohme_runtime = _predict_state(
            state, configurations[name], crohme_rows, labels, pretrained, device,
            args.predict_batch_size,
        )
        in_memory_predictions[name] = {
            "direct": direct_predictions,
            "crohme": crohme_predictions,
        }
        product_evaluation[name] = {
            "direct_refit_warning": "all direct rows are fit input",
            "direct_refit": _metrics(direct_rows, direct_predictions),
            "direct_candidate_audit": masked._candidate_audit(
                direct_rows, direct_predictions
            ),
            "direct_runtime": direct_runtime,
            "crohme_selection_role": (
                "none; post-selection noncommercial transfer diagnostic only"
            ),
            "crohme": _metrics(crohme_rows, crohme_predictions),
            "crohme_candidate_audit": masked._candidate_audit(
                crohme_rows, crohme_predictions
            ),
            "crohme_runtime": crohme_runtime,
        }

    # Historical r6 is loaded only now, after every independent model fit.
    r6_comparison = _r6_comparison(
        r6_context, r6_hwr, direct_rows, crohme_rows, device,
        args.predict_batch_size,
    )
    hashes_after = {
        str(path): masked._sha256(path) for path in immutable_paths
    }
    if hashes_after != hashes_before:
        raise AssertionError("independent context changed a frozen input")

    output.mkdir(parents=True, exist_ok=False)
    checkpoint_paths = {}
    checkpoint_reports = {}
    training_contract = {
        "initialization": "pinned generic BERT-Tiny only",
        "r6_or_aiflow_context_warm_start": False,
        "pretrained_model_file_sha256": pretrained_hashes,
        "prompt_formulas": len(masked._formulae(prompt_rows)),
        "prompt_records": len(prompt_rows),
        "direct_records": len(direct_rows),
        "direct_formulas": len(masked._formulae(direct_rows)),
        "direct_writers": len({str(row["writer_group"]) for row in direct_rows}),
        "direct_repeat": args.direct_repeat,
        "noise_rate": args.noise_rate,
        "noise_replicas": args.noise_replicas,
        "noise_source": "outer-training-fold frozen HWR Top-5 confusions only",
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "product_losses": product_losses,
    }
    for name, state in product_states.items():
        checkpoint_path = output / f"independent_context_{name}.pt"
        torch.save(
            _checkpoint_payload(
                state, configurations[name], labels, hwr_checkpoint,
                prompt_corpus, direct_path, pretrained, training_contract,
            ),
            checkpoint_path,
        )
        checkpoint_paths[name] = checkpoint_path
        model, contract, payload = load_independent_formula_context(
            pretrained, checkpoint_path, hwr_checkpoint, device
        )
        reload_direct, direct_audit = decide_independent_formula_rows(
            model, contract, payload, direct_rows, device,
            args.predict_batch_size,
        )
        reload_crohme, crohme_audit = decide_independent_formula_rows(
            model, contract, payload, crohme_rows, device,
            args.predict_batch_size,
        )
        mismatches = {
            "direct": sum(
                reload_direct[key] != value
                for key, value in in_memory_predictions[name]["direct"].items()
            ),
            "crohme": sum(
                reload_crohme[key] != value
                for key, value in in_memory_predictions[name]["crohme"].items()
            ),
        }
        if any(mismatches.values()):
            raise AssertionError(f"independent checkpoint reload mismatch: {mismatches}")
        checkpoint_reports[name] = {
            "path": str(checkpoint_path),
            "sha256": masked._sha256(checkpoint_path),
            "reload_mismatches": mismatches,
            "candidate_audit": {
                "direct": direct_audit,
                "crohme": crohme_audit,
            },
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    masked._write_prediction_rows(
        output / "writer_loo_safe_predictions.jsonl.gz",
        direct_rows, oof_predictions["safe"],
    )
    masked._write_prediction_rows(
        output / "writer_loo_accuracy_predictions.jsonl.gz",
        direct_rows, oof_predictions["accuracy"],
    )
    for name in ("safe", "accuracy"):
        masked._write_prediction_rows(
            output / f"direct_refit_{name}_predictions.jsonl.gz",
            direct_rows, in_memory_predictions[name]["direct"],
        )
        masked._write_prediction_rows(
            output / f"crohme_{name}_predictions.jsonl.gz",
            crohme_rows, in_memory_predictions[name]["crohme"],
        )

    baseline_predictions = {
        str(row["record_id"]): str(row["final_topk"][0])
        for row in direct_rows
    }
    baseline_metrics = _metrics(direct_rows, baseline_predictions)
    accuracy_metrics = oof_evaluation["accuracy"]["all"]
    candidate_contract_passed = all(
        checkpoint_reports[name]["candidate_audit"]["direct"]["new_tokens"] == 0
        and checkpoint_reports[name]["candidate_audit"]["direct"][
            "grouping_mutations"
        ] == 0
        for name in checkpoint_reports
    )
    crohme_candidate_contract_report_only = all(
        checkpoint_reports[name]["candidate_audit"]["crohme"]["new_tokens"] == 0
        and checkpoint_reports[name]["candidate_audit"]["crohme"][
            "grouping_mutations"
        ] == 0
        for name in checkpoint_reports
    )
    report = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "architecture": {
            **ARCHITECTURE,
            "parameters": sum(
                tensor.numel()
                for tensor in next(iter(product_states.values())).values()
            ),
            "external_pretrained_weights": True,
            "pretrained_license": "Apache-2.0",
            "pretrained_file_sha256": pretrained_hashes,
            "parent_context_checkpoint": None,
            "warm_start": False,
        },
        "data": {
            "prompt_admission": prompt_admission,
            "direct_records": len(direct_rows),
            "direct_formulas": len(masked._formulae(direct_rows)),
            "direct_writers": len({str(row["writer_group"]) for row in direct_rows}),
            "crohme_training_or_selection": False,
            "prompt_direct_exact_sequence_overlap": 0,
            "crohme_used_to_filter_prompt_training_rows": False,
            "training_data_guard": zero_crohme_training_manifest(
                admitted_sources={
                    "project-owned-direct": len(direct_rows),
                    "project-owned-prompt": len(prompt_rows),
                },
            ),
        },
        "training": {
            "outer_split": "leave-one-writer-out; held duplicate token sequences removed from training",
            "inner_split": "formula and exact-token-sequence grouped selection inside outer training fold",
            "configuration_grid": {
                "epochs": list(range(1, args.max_epochs + 1)),
                "lambda": list(LAMBDA_GRID),
            },
            "direct_repeat": args.direct_repeat,
            "noise_rate": args.noise_rate,
            "noise_replicas": args.noise_replicas,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "folds": folds,
            "product_configuration": configurations,
            "product_losses": product_losses,
        },
        "evaluation": {
            "hwr_writer_loo_baseline": baseline_metrics,
            "independent_writer_loo": oof_evaluation,
            "product_refit_and_crohme": product_evaluation,
            "historical_r6_comparison": r6_comparison,
        },
        "integrity": {
            "shape_training_performed": False,
            "shape_gradient_updates": 0,
            "candidate_contract_passed": candidate_contract_passed,
            "crohme_candidate_contract_report_only": (
                crohme_candidate_contract_report_only
            ),
            "immutable_hashes_before": hashes_before,
            "immutable_hashes_after": hashes_after,
            "immutable_inputs_unchanged": hashes_before == hashes_after,
            "r6_loaded_after_all_independent_training": True,
        },
        "checkpoint": checkpoint_reports,
        "decision": {
            "independent_model_created": True,
            "research_gate_passed": (
                accuracy_metrics["all_top1"] > baseline_metrics["all_top1"]
                and accuracy_metrics["formula_exact"]
                >= baseline_metrics["formula_exact"]
                and candidate_contract_passed
                and hashes_before == hashes_after
            ),
            "commercial_accuracy_gate_passed": False,
            "automatic_default_replacement": False,
            "runtime_status": "standalone shadow context model",
            "reason": (
                "outer writer OOF is valid research evidence, but all seven writers "
                "are now observed and fresh untouched commercial acceptance is absent"
            ),
        },
    }
    report_path = output / "independent_context_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    _event(
        "independent_context_complete", output=str(output),
        baseline_top1=baseline_metrics["all_top1"],
        safe_top1=oof_evaluation["safe"]["all"]["all_top1"],
        accuracy_top1=accuracy_metrics["all_top1"],
        accuracy_formula_exact=accuracy_metrics["formula_exact"],
        research_gate=report["decision"]["research_gate_passed"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
