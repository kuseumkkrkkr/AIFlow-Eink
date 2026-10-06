#!/usr/bin/env python3
"""AIFlow 1.0e single external-weight partial fine-tuning trial.

Only Azu/trocr-handwritten-math is used as the external neural model.  A
configurable trailing slice of its DeiT encoder is fine-tuned together with
the candidate selector on the existing AIFlow writer split.  HWR candidate
generation and structure are untouched.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from train_ocr_decision_adapter_10e import (
    DEFAULT_CANDIDATES,
    DEFAULT_FORMULAS,
    OcrDecisionAdapter,
    _batch,
    _load_candidates,
    _load_formula_records,
    _metrics,
    _predict,
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


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts" / "ocr_trocr_weight_tuned_10e_20260830"
SCHEMA = "aiflow-1.0e-trocr-weight-tuned/v1"


def _dynamic_features(
    teacher: FrozenTrOcrEncoder,
    formula_images: dict[str, object],
    formula_ids: list[str],
    device: torch.device,
    batch_size: int,
    with_grad: bool,
) -> dict[str, torch.Tensor]:
    features: dict[str, torch.Tensor] = {}
    context = torch.enable_grad() if with_grad else torch.inference_mode()
    with context:
        for start in range(0, len(formula_ids), batch_size):
            ids = formula_ids[start:start + batch_size]
            images = [formula_images[formula_id].convert("RGB") for formula_id in ids]
            pixel_values = teacher.processor(images=images, return_tensors="pt").pixel_values
            hidden = teacher.encoder(pixel_values=pixel_values.to(device)).last_hidden_state
            pooled = hidden.mean(dim=1)
            features.update({formula_id: pooled[index] for index, formula_id in enumerate(ids)})
    return features


def _train_fold(
    teacher: FrozenTrOcrEncoder,
    selector: OcrDecisionAdapter,
    examples: list[dict],
    formula_images: dict[str, object],
    epochs: int,
    seed: int,
    device: torch.device,
    batch_size: int,
    learning_rate: float,
    unfreeze_blocks: int,
) -> None:
    layers = teacher.encoder.encoder.layer
    if not 1 <= unfreeze_blocks <= len(layers):
        raise ValueError(f"unfreeze_blocks must be between 1 and {len(layers)}")
    tuned_layers = list(layers[-unfreeze_blocks:])
    tuned = [parameter for layer in tuned_layers for parameter in layer.parameters()]
    for parameter in tuned:
        parameter.requires_grad_(True)
    selector_parameters = list(selector.parameters())
    optimizer = torch.optim.AdamW([
        {"params": tuned, "lr": learning_rate * 0.1},
        {"params": selector_parameters, "lr": learning_rate},
    ], weight_decay=1e-2)
    rng = random.Random(seed)
    teacher.encoder.eval()
    for layer in tuned_layers:
        layer.train()
    selector.train()
    order = list(range(len(examples)))
    for _ in range(epochs):
        rng.shuffle(order)
        for start in range(0, len(order), 32):
            selected = [examples[index] for index in order[start:start + 32]]
            formula_ids = list(dict.fromkeys(str(example["formula_id"]) for example in selected))
            feature_map = _dynamic_features(
                teacher, formula_images, formula_ids, device, batch_size, with_grad=True
            )
            numeric, token_ids, _, mask, targets = _batch(selected, device)
            ocr = torch.stack([feature_map[str(example["formula_id"])] for example in selected])
            scores = selector(numeric, token_ids, ocr, mask)
            loss = torch.nn.functional.cross_entropy(scores, targets, ignore_index=-100)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tuned + selector_parameters, 1.0)
            optimizer.step()
    selector.eval()
    for layer in tuned_layers:
        layer.eval()
    for parameter in tuned:
        parameter.requires_grad_(False)


def _tuned_parameters(teacher: FrozenTrOcrEncoder, unfreeze_blocks: int) -> dict[str, torch.Tensor]:
    layers = teacher.encoder.encoder.layer
    start = len(layers) - unfreeze_blocks
    return {
        f"layer.{layer_index}.{name}": parameter.detach().cpu()
        for layer_index, layer in enumerate(layers[start:], start=start)
        for name, parameter in layer.named_parameters()
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--formulas", type=Path, default=DEFAULT_FORMULAS)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--unfreeze-blocks", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    if args.unfreeze_blocks < 1:
        parser.error("--unfreeze-blocks must be at least 1")
    _set_seed(args.seed)
    device = torch.device(args.device)
    rows = _load_candidates(args.candidates, args.limit)
    formulas = _load_formula_records(args.formulas)
    formula_ids = list(dict.fromkeys(str(row["formula_id"]) for row in rows))
    missing = [formula_id for formula_id in formula_ids if formula_id not in formulas]
    if missing:
        raise ValueError(f"formula raster source is missing {len(missing)} formula IDs")
    labels = sorted({str(token) for row in rows for token in row["final_topk"]})
    token_to_id = {token: index for index, token in enumerate(labels)}
    formula_images = {formula_id: _render_formula(formulas[formula_id]) for formula_id in formula_ids}
    feature_size = 384
    placeholder_features = {
        formula_id: np.zeros(feature_size, dtype=np.float32) for formula_id in formula_ids
    }
    examples = _make_examples(rows, placeholder_features, token_to_id)
    by_writer: defaultdict[str, list[dict]] = defaultdict(list)
    for example in examples:
        by_writer[example["writer_group"]].append(example)
    print(json.dumps({
        "event": "external_weight_tuning_start", "schema": SCHEMA, "model": MODEL_ID,
        "model_revision": MODEL_REVISION, "formula_count": len(formula_ids),
        "record_count": len(rows), "device": str(device),
        "external_layer": f"encoder.encoder.layer[-{args.unfreeze_blocks}:]",
    }, ensure_ascii=False), flush=True)
    all_predictions: list[dict] = []
    folds: list[dict] = []
    external_parameter_count = 0
    for fold_index, writer in enumerate(sorted(by_writer)):
        teacher = FrozenTrOcrEncoder(args.model, device)
        tuned = [
            parameter
            for layer in teacher.encoder.encoder.layer[-args.unfreeze_blocks:]
            for parameter in layer.parameters()
        ]
        external_parameter_count = sum(parameter.numel() for parameter in tuned)
        train = [example for group, values in by_writer.items() if group != writer for example in values]
        test = list(by_writer[writer])
        selector = OcrDecisionAdapter(examples[0]["numeric"].shape[-1], len(labels), feature_size).to(device)
        _train_fold(
            teacher, selector, train, formula_images, args.epochs, args.seed + fold_index,
            device, args.batch_size, args.learning_rate, args.unfreeze_blocks,
        )
        test_features = _dynamic_features(
            teacher, formula_images, list(dict.fromkeys(str(example["formula_id"]) for example in test)),
            device, args.batch_size, with_grad=False,
        )
        for example in test:
            example["ocr"] = test_features[str(example["formula_id"])].cpu().numpy()
        predictions = _predict(selector, test, device)
        all_predictions.extend(predictions)
        folds.append({"held_writer": writer, **_metrics(predictions)})
        del selector, teacher
        if device.type == "cuda":
            torch.cuda.empty_cache()
    refit_teacher = FrozenTrOcrEncoder(args.model, device)
    refit_selector = OcrDecisionAdapter(examples[0]["numeric"].shape[-1], len(labels), feature_size).to(device)
    _train_fold(
        refit_teacher, refit_selector, examples, formula_images, args.epochs, args.seed + 1000,
        device, args.batch_size, args.learning_rate, args.unfreeze_blocks,
    )
    args.output.mkdir(parents=True)
    checkpoint = args.output / "ocr_trocr_weight_tuned.pt"
    tuned_state = _tuned_parameters(refit_teacher, args.unfreeze_blocks)
    checkpoint_data = {
        "schema": SCHEMA,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "weights_source": "huggingface_pretrained_then_partial_finetune",
        "external_weights_tuned": True,
        "external_tuned_parameter_names": [f"encoder.encoder.{name}" for name in tuned_state],
        "external_tuned_parameter_count": external_parameter_count,
        "numeric_size": int(examples[0]["numeric"].shape[-1]),
        "ocr_size": feature_size,
        "labels": labels,
        "external_encoder_layers_state_dict": tuned_state,
        "selector_state_dict": refit_selector.state_dict(),
        "candidate_contract": {
            "top_k_only": True, "token_creation": False, "row_deletion": False,
            "stroke_regrouping": False, "relation_mutation": False,
        },
    }
    if args.unfreeze_blocks == 1:
        checkpoint_data["external_encoder_layer_11_state_dict"] = {
            name.removeprefix("layer.11."): value
            for name, value in tuned_state.items()
        }
    torch.save(checkpoint_data, checkpoint)
    evaluation = {
        "schema": SCHEMA,
        "model": {
            "name": MODEL_ID, "revision": MODEL_REVISION,
            "weights_source": "huggingface_pretrained_then_partial_finetune",
            "external_weights_tuned": True,
            "tuned_scope": f"encoder.encoder.layer[-{args.unfreeze_blocks}:]",
        },
        "data": {
            "candidate_path": str(args.candidates), "formula_path": str(args.formulas),
            "formula_count": len(formula_ids), "record_count": len(rows),
        },
        "training": {
            "epochs": args.epochs, "seed": args.seed, "learning_rate": args.learning_rate,
            "external_tuned_parameters": external_parameter_count,
            "candidate_adapter_parameters": sum(parameter.numel() for parameter in refit_selector.parameters()),
            "unfreeze_blocks": args.unfreeze_blocks,
            "tuning_scope": "external_encoder_last_block_plus_candidate_selector",
        },
        "writer_loo": folds,
        "aggregate": _metrics(all_predictions),
        "status": "shadow_only",
        "checkpoint": str(checkpoint),
    }
    (args.output / "evaluation.json").write_text(json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8")
    with gzip.open(args.output / "writer_loo_predictions.jsonl.gz", "wt", encoding="utf-8") as stream:
        for prediction in all_predictions:
            stream.write(json.dumps(prediction, ensure_ascii=False) + "\n")
    print(json.dumps({"event": "complete", "output": str(args.output), "aggregate": evaluation["aggregate"], "external_tuned_parameters": external_parameter_count}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
