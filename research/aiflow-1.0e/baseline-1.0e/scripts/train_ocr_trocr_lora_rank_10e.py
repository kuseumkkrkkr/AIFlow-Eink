#!/usr/bin/env python3
"""Shadow-only LoRA candidate ranking for AIFlow 1.0e.

The external TrOCR encoder remains a training/evaluation teacher.  Existing
AIFlow HWR candidates, stroke grouping, and token contracts are unchanged.
The selector is trained with formula-balanced sampling and a conservative
margin loss that stabilizes the existing baseline when it is already correct.
This artifact is not a product runtime bundle.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from train_ocr_decision_adapter_10e import (
    DEFAULT_CANDIDATES,
    DEFAULT_FORMULAS,
    OcrDecisionAdapter,
    _batch,
    _load_candidates,
    _load_formula_records,
    _render_formula,
    _set_seed,
)
from train_ocr_trocr_adapter_10e import (
    DEFAULT_MODEL,
    FrozenTrOcrEncoder,
    MODEL_ID,
    MODEL_REVISION,
    _make_examples,
)
from train_ocr_trocr_weight_tuned_10e import _dynamic_features


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts" / "ocr_trocr_lora_rank_10e_20260902"
SCHEMA = "aiflow-1.0e-trocr-lora-rank/v1"


class LoRALinear(nn.Module):
    """A frozen linear layer plus a small low-rank residual."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.scaling = float(alpha) / float(rank)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        update = (inputs @ self.lora_a.t()) @ self.lora_b.t()
        return self.base(inputs) + update * self.scaling


def _get_parent(root: nn.Module, path: str) -> tuple[nn.Module, str]:
    pieces = path.split(".")
    parent = root
    for piece in pieces[:-1]:
        parent = parent[int(piece)] if piece.isdigit() else getattr(parent, piece)
    return parent, pieces[-1]


def _install_lora(
    encoder: nn.Module,
    block_count: int,
    rank: int,
    alpha: float,
    target_modules: tuple[str, ...],
) -> list[LoRALinear]:
    layers = encoder.encoder.layer
    if not 1 <= block_count <= len(layers):
        raise ValueError(f"block_count must be between 1 and {len(layers)}")
    installed: list[LoRALinear] = []
    for layer in layers[-block_count:]:
        for module_name in target_modules:
            module_path = {
                "query": "attention.attention.query",
                "key": "attention.attention.key",
                "value": "attention.attention.value",
                "output": "attention.output.dense",
                "ffn_in": "intermediate.dense",
                "ffn_out": "output.dense",
            }.get(module_name, module_name)
            parent, leaf = _get_parent(layer, module_path)
            base = getattr(parent, leaf)
            if not isinstance(base, nn.Linear):
                raise TypeError(f"{module_path} is {type(base).__name__}, expected Linear")
            adapter = LoRALinear(base, rank, alpha)
            setattr(parent, leaf, adapter)
            installed.append(adapter)
    return installed


def _trainable_lora(installed: list[LoRALinear]) -> list[nn.Parameter]:
    return [parameter for module in installed for parameter in module.parameters() if parameter.requires_grad]


def _balanced_batches(examples: list[dict], steps: int, batch_size: int, seed: int) -> list[list[dict]]:
    eligible = [example for example in examples if int(example["target"]) >= 0]
    if not eligible:
        return []
    by_formula: dict[str, list[dict]] = defaultdict(list)
    for example in eligible:
        by_formula[str(example["formula_id"])].append(example)
    formula_ids = sorted(by_formula)
    rng = random.Random(seed)
    batches: list[list[dict]] = []
    for _ in range(steps):
        batches.append([rng.choice(by_formula[rng.choice(formula_ids)]) for _ in range(batch_size)])
    return batches


def _ranking_loss(scores: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    valid = targets >= 0
    if not bool(valid.any()):
        return scores.sum() * 0.0, {"ce": 0.0, "pairwise": 0.0, "stability": 0.0}
    valid_scores = scores[valid]
    valid_targets = targets[valid]
    ce = nn.functional.cross_entropy(valid_scores, valid_targets)
    rows = torch.arange(valid_scores.shape[0], device=scores.device)
    target_scores = valid_scores[rows, valid_targets]
    other_scores = valid_scores.clone()
    other_scores[rows, valid_targets] = torch.finfo(other_scores.dtype).min
    hardest_other = other_scores.max(dim=1).values
    margin = 0.35
    pairwise = nn.functional.relu(margin - target_scores + hardest_other)
    nonbaseline = valid_targets != 0
    pairwise_term = pairwise[nonbaseline].mean() if bool(nonbaseline.any()) else pairwise.mean() * 0.0
    baseline_correct = ~nonbaseline
    stability_term = (
        nn.functional.relu(0.20 - valid_scores[baseline_correct, 0] + hardest_other[baseline_correct]).mean()
        if bool(baseline_correct.any())
        else pairwise.mean() * 0.0
    )
    loss = ce + 0.50 * pairwise_term + 0.25 * stability_term
    return loss, {"ce": float(ce.detach()), "pairwise": float(pairwise_term.detach()), "stability": float(stability_term.detach())}


def _train_fold(
    teacher: FrozenTrOcrEncoder,
    selector: OcrDecisionAdapter,
    examples: list[dict],
    formula_images: dict[str, object],
    epochs: int,
    seed: int,
    device: torch.device,
    feature_batch_size: int,
    learning_rate: float,
    installed: list[LoRALinear],
) -> dict[str, float]:
    trainable = _trainable_lora(installed) + list(selector.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=1e-2)
    teacher.encoder.eval()
    for module in installed:
        module.train()
    selector.train()
    eligible = [example for example in examples if int(example["target"]) >= 0]
    steps_per_epoch = max(1, math.ceil(len(eligible) / 32))
    totals = {"ce": 0.0, "pairwise": 0.0, "stability": 0.0}
    batches_seen = 0
    for epoch in range(epochs):
        batches = _balanced_batches(examples, steps_per_epoch, 32, seed + epoch)
        for selected in batches:
            formula_ids = list(dict.fromkeys(str(example["formula_id"]) for example in selected))
            feature_map = _dynamic_features(
                teacher, formula_images, formula_ids, device, feature_batch_size, with_grad=True
            )
            numeric, token_ids, _, mask, targets = _batch(selected, device)
            ocr = torch.stack([feature_map[str(example["formula_id"])] for example in selected])
            scores = selector(numeric, token_ids, ocr, mask)
            loss, parts = _ranking_loss(scores, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            for name in totals:
                totals[name] += parts[name]
            batches_seen += 1
    selector.eval()
    for module in installed:
        module.eval()
    for parameter in trainable:
        parameter.requires_grad_(False)
    return {name: value / max(1, batches_seen) for name, value in totals.items()}


def _formula_metrics(rows: list[dict], prediction_key: str, baseline_key: str = "baseline_token") -> dict:
    correct = [str(row[prediction_key]) == str(row["label"]) for row in rows]
    baseline = [str(row[baseline_key]) == str(row["label"]) for row in rows]
    groups: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for row, is_correct in zip(rows, correct):
        groups[(str(row["writer_group"]), str(row["formula_id"]))].append(is_correct)
    exact = sum(all(values) for values in groups.values())
    return {
        "rows": len(rows),
        "top1": sum(correct) / len(rows) if rows else 0.0,
        "top1_correct": sum(correct),
        "formula_exact": exact / len(groups) if groups else 0.0,
        "formula_exact_correct": exact,
        "formula_total": len(groups),
        "row_level_improvements": sum(not old and new for old, new in zip(baseline, correct)),
        "row_level_regressions": sum(old and not new for old, new in zip(baseline, correct)),
        "changed_rows": sum(str(row[prediction_key]) != str(row[baseline_key]) for row in rows),
    }


def _write_predictions(path: Path, rows: list[dict]) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as stream:
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
                    stream.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--formulas", type=Path, default=DEFAULT_FORMULAS)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--unfreeze-blocks", type=int, default=2)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--target-modules", nargs="+", default=["query", "value"])
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    if args.rank < 1 or args.alpha <= 0:
        parser.error("rank must be positive and alpha must be greater than zero")
    _set_seed(args.seed)
    device = torch.device(args.device)
    rows = _load_candidates(args.candidates, None)
    formulas = _load_formula_records(args.formulas)
    formula_ids = list(dict.fromkeys(str(row["formula_id"]) for row in rows))
    missing = [formula_id for formula_id in formula_ids if formula_id not in formulas]
    if missing:
        raise ValueError(f"formula raster source is missing {len(missing)} formula IDs")
    labels = sorted({str(token) for row in rows for token in row["final_topk"]})
    token_to_id = {token: index for index, token in enumerate(labels)}
    formula_images = {formula_id: _render_formula(formulas[formula_id]) for formula_id in formula_ids}
    placeholder_features = {formula_id: np.zeros(384, dtype=np.float32) for formula_id in formula_ids}
    examples = _make_examples(rows, placeholder_features, token_to_id)
    by_writer: defaultdict[str, list[dict]] = defaultdict(list)
    for example in examples:
        by_writer[example["writer_group"]].append(example)
    print(json.dumps({
        "event": "lora_rank_start", "schema": SCHEMA, "model": MODEL_ID,
        "model_revision": MODEL_REVISION, "formula_count": len(formula_ids),
        "record_count": len(rows), "device": str(device), "rank": args.rank,
        "alpha": args.alpha, "target_modules": args.target_modules,
    }, ensure_ascii=False), flush=True)

    all_predictions: list[dict] = []
    folds: list[dict] = []
    loss_reports: list[dict] = []
    lora_parameter_count = 0
    for fold_index, writer in enumerate(sorted(by_writer)):
        teacher = FrozenTrOcrEncoder(args.model, device)
        installed = _install_lora(
            teacher.encoder, args.unfreeze_blocks, args.rank, args.alpha, tuple(args.target_modules)
        )
        teacher.encoder.to(device)
        lora_parameter_count = sum(parameter.numel() for parameter in _trainable_lora(installed))
        train = [
            example for group, values in by_writer.items() if group != writer for example in values
            if int(example["target"]) >= 0
        ]
        test = list(by_writer[writer])
        selector = OcrDecisionAdapter(examples[0]["numeric"].shape[-1], len(labels), 384).to(device)
        loss_report = _train_fold(
            teacher, selector, train, formula_images, args.epochs, args.seed + fold_index,
            device, args.batch_size, args.learning_rate, installed,
        )
        loss_report["held_writer"] = writer
        loss_reports.append(loss_report)
        test_features = _dynamic_features(
            teacher, formula_images, list(dict.fromkeys(str(example["formula_id"]) for example in test)),
            device, args.batch_size, with_grad=False,
        )
        for example in test:
            example["ocr"] = test_features[str(example["formula_id"])].cpu().numpy()
        predictions = _predict_with_scores(selector, test, device)
        all_predictions.extend(predictions)
        folds.append({"held_writer": writer, **_formula_metrics(predictions, "adapter_token")})
        del selector, teacher
        if device.type == "cuda":
            torch.cuda.empty_cache()

    refit_teacher = FrozenTrOcrEncoder(args.model, device)
    refit_installed = _install_lora(
        refit_teacher.encoder, args.unfreeze_blocks, args.rank, args.alpha, tuple(args.target_modules)
    )
    refit_teacher.encoder.to(device)
    refit_selector = OcrDecisionAdapter(examples[0]["numeric"].shape[-1], len(labels), 384).to(device)
    _train_fold(
        refit_teacher, refit_selector, [example for example in examples if int(example["target"]) >= 0],
        formula_images, args.epochs, args.seed + 1000, device, args.batch_size, args.learning_rate, refit_installed,
    )
    args.output.mkdir(parents=True)
    lora_state = {
        name: parameter.detach().cpu()
        for name, parameter in refit_teacher.encoder.named_parameters()
        if ".lora_a" in name or ".lora_b" in name
    }
    checkpoint_data = {
        "schema": SCHEMA,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "weights_source": "huggingface_pretrained_then_lora_partial_finetune",
        "external_weights_tuned": True,
        "lora_rank": args.rank,
        "lora_alpha": args.alpha,
        "lora_target_modules": args.target_modules,
        "lora_block_count": args.unfreeze_blocks,
        "lora_parameter_count": lora_parameter_count,
        "lora_state_dict": lora_state,
        "selector_state_dict": refit_selector.state_dict(),
        "labels": labels,
        "candidate_contract": {
            "top_k_only": True, "token_creation": False, "row_deletion": False,
            "stroke_regrouping": False, "relation_mutation": False,
        },
        "runtime_status": "shadow_only; external raster teacher not bundled for product runtime",
    }
    checkpoint = args.output / "ocr_trocr_lora_rank.pt"
    torch.save(checkpoint_data, checkpoint)
    baseline_metrics = _formula_metrics(all_predictions, "baseline_token")
    adapter_metrics = _formula_metrics(all_predictions, "adapter_token")
    evaluation = {
        "schema": SCHEMA,
        "model": {"name": MODEL_ID, "revision": MODEL_REVISION, "weights_source": checkpoint_data["weights_source"]},
        "data": {"candidate_path": str(args.candidates), "formula_path": str(args.formulas), "formula_count": len(formula_ids), "record_count": len(rows)},
        "training": {
            "epochs": args.epochs, "seed": args.seed, "learning_rate": args.learning_rate,
            "lora_rank": args.rank, "lora_alpha": args.alpha, "lora_block_count": args.unfreeze_blocks,
            "lora_target_modules": args.target_modules, "lora_parameters": lora_parameter_count,
            "selector_parameters": sum(parameter.numel() for parameter in refit_selector.parameters()),
            "loss": "cross_entropy + formula-balanced pairwise margin + baseline stability penalty",
        },
        "writer_loo": folds,
        "loss_reports": loss_reports,
        "baseline": baseline_metrics,
        "adapter": adapter_metrics,
        "candidate_recall": sum(bool(row["target_in_candidates"]) for row in all_predictions) / len(all_predictions),
        "status": "shadow_only",
        "checkpoint": str(checkpoint),
    }
    (args.output / "evaluation.json").write_text(json.dumps(evaluation, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_predictions(args.output / "writer_loo_predictions.jsonl.gz", all_predictions)
    print(json.dumps({"event": "complete", "output": str(args.output), "baseline": baseline_metrics, "adapter": adapter_metrics, "candidate_recall": evaluation["candidate_recall"], "lora_parameters": lora_parameter_count}, ensure_ascii=False), flush=True)
    return 0


@torch.inference_mode()
def _predict_with_scores(model: OcrDecisionAdapter, examples: list[dict], device: torch.device) -> list[dict]:
    predictions: list[dict] = []
    for start in range(0, len(examples), 64):
        selected = examples[start:start + 64]
        numeric, token_ids, ocr, mask, _ = _batch(selected, device)
        scores = model(numeric, token_ids, ocr, mask).cpu().numpy()
        for example, row_scores in zip(selected, scores):
            row_scores = row_scores[:len(example["candidates"])]
            selected_index = int(np.argmax(row_scores))
            predictions.append({
                "record_id": example["record_id"], "formula_id": example["formula_id"],
                "writer_group": example["writer_group"], "label": example["label"],
                "candidates": example["candidates"], "baseline_token": example["candidates"][0],
                "adapter_token": example["candidates"][selected_index],
                "target_in_candidates": example["target"] >= 0,
                "adapter_scores": [float(value) for value in row_scores],
            })
    return predictions


if __name__ == "__main__":
    raise SystemExit(main())
