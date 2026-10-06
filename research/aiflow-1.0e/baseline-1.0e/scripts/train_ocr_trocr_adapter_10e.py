#!/usr/bin/env python3
"""AIFlow 1.0e single-model trial using an external TrOCR checkpoint.

The Hugging Face Azu/trocr-handwritten-math weights are the only neural OCR
model used by this trial.  They are frozen as an external representation; a
small candidate-only adapter is tuned on the existing AIFlow project-owned
writer split.  The adapter can select one existing HWR Top-k candidate only.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import ViTImageProcessor, VisionEncoderDecoderModel

from train_ocr_decision_adapter_10e import (
    DEFAULT_CANDIDATES,
    DEFAULT_FORMULAS,
    OcrDecisionAdapter,
    _json_lines,
    _load_candidates,
    _load_formula_records,
    _metrics,
    _predict,
    _render_formula,
    _row_numeric,
    _set_seed,
    _train,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path(
    r"D:\AIFlow-Workspace\Caches\user\.cache\huggingface\hub\models--Azu--trocr-handwritten-math"
    r"\snapshots\fc8dc9829360d42b1d4bc2f2668c831a72c80379"
)
DEFAULT_OUTPUT = ROOT / "artifacts" / "ocr_trocr_adapter_10e_20260830"
MODEL_ID = "Azu/trocr-handwritten-math"
MODEL_REVISION = "fc8dc9829360d42b1d4bc2f2668c831a72c80379"
SCHEMA = "aiflow-1.0e-trocr-candidate-adapter/v1"


class FrozenTrOcrEncoder:
    """Use only the frozen encoder representation from one HF OCR checkpoint."""

    def __init__(self, model_path: Path, device: torch.device):
        if not model_path.exists():
            raise FileNotFoundError(f"TrOCR model snapshot is missing: {model_path}")
        self.device = device
        self.model = VisionEncoderDecoderModel.from_pretrained(
            str(model_path), local_files_only=True, torch_dtype=torch.float32,
        ).to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.encoder = self.model.get_encoder().eval()
        self.processor = ViTImageProcessor(
            do_resize=True,
            size={"height": 384, "width": 384},
            resample=2,
            do_rescale=True,
            rescale_factor=1 / 255.0,
            do_normalize=True,
            image_mean=[0.5, 0.5, 0.5],
            image_std=[0.5, 0.5, 0.5],
        )

    @torch.inference_mode()
    def encode(self, images: list[Image.Image], batch_size: int = 8) -> np.ndarray:
        if not images:
            raise ValueError("at least one formula image is required")
        outputs = []
        for start in range(0, len(images), batch_size):
            batch_images = [image.convert("RGB") for image in images[start:start + batch_size]]
            pixel_values = self.processor(images=batch_images, return_tensors="pt").pixel_values
            hidden = self.encoder(pixel_values=pixel_values.to(self.device)).last_hidden_state
            outputs.append(hidden.mean(dim=1).cpu().numpy().astype(np.float32))
        return np.concatenate(outputs, axis=0)


def _make_examples(rows: list[dict], formula_features: dict[str, np.ndarray], token_to_id: dict[str, int]) -> list[dict]:
    examples: list[dict] = []
    by_formula_index: defaultdict[str, int] = defaultdict(int)
    for row in rows:
        formula_id = str(row["formula_id"])
        position = by_formula_index[formula_id]
        by_formula_index[formula_id] += 1
        if formula_id not in formula_features:
            raise KeyError(f"missing external TrOCR evidence for {formula_id}")
        candidates = [str(token) for token in row["final_topk"]]
        target = str(row["label"])
        examples.append({
            "record_id": str(row.get("record_id", "")),
            "formula_id": formula_id,
            "writer_group": str(row["writer_group"]),
            "numeric": np.asarray(
                [_row_numeric(row, candidate_index) for candidate_index in range(len(candidates))],
                dtype=np.float32,
            ),
            "token_ids": np.asarray([token_to_id[token] for token in candidates], dtype=np.int64),
            "mask": np.ones(len(candidates), dtype=bool),
            "ocr": formula_features[formula_id],
            "target": candidates.index(target) if target in candidates else -1,
            "baseline": 0,
            "candidates": candidates,
            "label": target,
        })
    return examples


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
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
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
    print(json.dumps({
        "event": "external_trocr_load", "schema": SCHEMA, "model": MODEL_ID,
        "model_revision": MODEL_REVISION, "preprocessor": "ViTImageProcessor@384px",
        "formula_count": len(formula_ids),
        "record_count": len(rows), "device": str(device),
    }, ensure_ascii=False), flush=True)
    teacher = FrozenTrOcrEncoder(args.model, device)
    images = [_render_formula(formulas[formula_id]) for formula_id in formula_ids]
    formula_features_array = teacher.encode(images, batch_size=args.batch_size)
    formula_features = dict(zip(formula_ids, formula_features_array))
    examples = _make_examples(rows, formula_features, token_to_id)
    by_writer: defaultdict[str, list[dict]] = defaultdict(list)
    for example in examples:
        by_writer[example["writer_group"]].append(example)
    all_predictions: list[dict] = []
    folds: list[dict] = []
    for fold_index, writer in enumerate(sorted(by_writer)):
        train = [example for group, values in by_writer.items() if group != writer for example in values]
        test = list(by_writer[writer])
        model = OcrDecisionAdapter(
            examples[0]["numeric"].shape[-1], len(labels), int(formula_features_array.shape[-1])
        ).to(device)
        _train(model, train, args.epochs, args.seed + fold_index, device)
        predictions = _predict(model, test, device)
        all_predictions.extend(predictions)
        folds.append({"held_writer": writer, **_metrics(predictions)})
    refit = OcrDecisionAdapter(
        examples[0]["numeric"].shape[-1], len(labels), int(formula_features_array.shape[-1])
    ).to(device)
    _train(refit, examples, args.epochs, args.seed + 1000, device)
    args.output.mkdir(parents=True)
    checkpoint = args.output / "ocr_trocr_adapter.pt"
    torch.save({
        "schema": SCHEMA,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "weights_source": "huggingface_pretrained",
        "weights_frozen": True,
        "preprocessor": {
            "type": "ViTImageProcessor", "size": 384, "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5], "source": "model-card-compatible-fixed-config",
        },
        "numeric_size": int(examples[0]["numeric"].shape[-1]),
        "ocr_size": int(formula_features_array.shape[-1]),
        "labels": labels,
        "state_dict": refit.state_dict(),
        "candidate_contract": {
            "top_k_only": True, "token_creation": False, "row_deletion": False,
            "stroke_regrouping": False, "relation_mutation": False,
        },
    }, checkpoint)
    evaluation = {
        "schema": SCHEMA,
        "model": {
            "name": MODEL_ID, "revision": MODEL_REVISION, "weights_frozen": True,
            "representation": "encoder_mean_pool_only",
        },
        "preprocessor": {
            "type": "ViTImageProcessor", "size": 384, "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5], "source": "model-card-compatible-fixed-config",
        },
        "data": {
            "candidate_path": str(args.candidates), "formula_path": str(args.formulas),
            "formula_count": len(formula_ids), "record_count": len(rows),
        },
        "training": {
            "epochs": args.epochs, "seed": args.seed,
            "trainable_adapter_parameters": sum(
                parameter.numel() for parameter in refit.parameters() if parameter.requires_grad
            ),
            "external_trainable_parameters": 0,
            "tuning_scope": "candidate_selector_only",
        },
        "writer_loo": folds,
        "aggregate": _metrics(all_predictions),
        "status": "shadow_only",
        "checkpoint": str(checkpoint),
    }
    (args.output / "evaluation.json").write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with gzip.open(args.output / "writer_loo_predictions.jsonl.gz", "wt", encoding="utf-8") as stream:
        for prediction in all_predictions:
            stream.write(json.dumps(prediction, ensure_ascii=False) + "\n")
    print(json.dumps({
        "event": "complete", "output": str(args.output), "aggregate": evaluation["aggregate"]
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
