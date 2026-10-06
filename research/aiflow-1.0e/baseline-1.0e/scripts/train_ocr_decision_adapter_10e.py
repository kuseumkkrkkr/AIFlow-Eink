#!/usr/bin/env python3
"""AIFlow Math Ink 1.0e: frozen OCR evidence + candidate-only decision adapter.

This is a shadow experiment.  The external formula OCR model is used only as
an image-level evidence encoder.  Its weights are frozen; only the small
candidate scorer below is trained.  The scorer can select an existing HWR
Top-k candidate, but cannot create a token, remove a row, regroup strokes, or
expand the candidate set.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from online_candidate_features_10e import _row_numeric as _online_row_numeric


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATES = ROOT / "artifacts" / "homograph_context_20260814" / "direct_candidates.jsonl.gz"
DEFAULT_FORMULAS = ROOT / "hf-dataset" / "data" / "formulas_valid.jsonl"
DEFAULT_OUTPUT = ROOT / "artifacts" / "ocr_decision_adapter_10e_20260829"
DEFAULT_TEXTELLER = Path(
    r"C:\Users\user\.cache\huggingface\hub\models--OleehyO--TexTeller"
    r"\snapshots\7b96df06b9d81cdb129c3bef68b7250bc3e2b0ea"
)
TEXTELLER_REVISION = "7b96df06b9d81cdb129c3bef68b7250bc3e2b0ea"
SCHEMA = "aiflow-1.0e-ocr-decision-adapter/v1"
POINT_KEYS = (
    "width_rel", "height_rel", "aspect_log", "path_over_diag", "direction_x",
    "direction_y", "stroke_count", "point_count_log", "center_x", "center_y",
)
CONTEXT_KEYS = ("previous_dx", "previous_dy", "next_dx", "next_dy")
FORMULA_GEOMETRY_KEYS = (
    "formula_left", "formula_top", "formula_width", "formula_height",
    "formula_cx", "formula_cy", "formula_geometry_available",
)
ROLE_NAMES = ("digit", "operator", "fence", "operand", "other")
OPERATOR_TOKENS = frozenset({
    "+", "-", "=", "/", "<", ">", r"\times", r"\div", r"\pm", r"\mp",
    r"\cdot", r"\ast", r"\leq", r"\geq", r"\neq", r"\approx",
})
FENCE_TOKENS = frozenset({"(", ")", "[", "]", "{", "}", "|", r"\{", r"\}"})
OPERAND_TOKENS = frozenset({
    r"\alpha", r"\beta", r"\gamma", r"\delta", r"\epsilon", r"\lambda",
    r"\mu", r"\pi", r"\sigma", r"\theta", r"\omega", r"\infty",
})


def _json_lines(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _semantic_role(token: str) -> str:
    token = str(token)
    if token.isdigit():
        return "digit"
    if token in OPERATOR_TOKENS:
        return "operator"
    if token in FENCE_TOKENS:
        return "fence"
    if token in OPERAND_TOKENS or token.startswith("\\"):
        return "operand"
    return "other"


def _load_candidates(path: Path, limit: int | None) -> list[dict]:
    rows = _json_lines(path)
    if not rows:
        raise ValueError(f"empty candidate file: {path}")
    formula_ids = list(dict.fromkeys(str(row["formula_id"]) for row in rows))
    if limit is not None:
        allowed = set(formula_ids[:limit])
        rows = [row for row in rows if str(row["formula_id"]) in allowed]
    required = {"formula_id", "writer_group", "label", "final_topk", "final_topk_probabilities", "geometry"}
    if any(not required <= set(row) for row in rows):
        raise ValueError("candidate rows do not satisfy the 1.0 candidate contract")
    for row in rows:
        if not row["final_topk"] or len(row["final_topk"]) != len(row["final_topk_probabilities"]):
            raise ValueError(f"invalid candidate list for {row.get('record_id')}")
    return rows


def _load_formula_records(path: Path) -> dict[str, dict]:
    records = {}
    for row in _json_lines(path):
        sample_id = str(row.get("sample_id", row.get("formula_id", "")))
        if sample_id:
            records[sample_id] = row
    return records


def _render_formula(record: dict) -> Image.Image:
    canvas = record.get("canvas") or {}
    width = max(1, int(canvas.get("width", 327)))
    height = max(1, int(canvas.get("height", 310)))
    image = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(image)
    for stroke in record.get("strokes", []):
        points = []
        for point in stroke.get("points", []):
            if isinstance(point, dict):
                points.append((float(point["x"]), float(point["y"])))
            else:
                points.append((float(point[0]), float(point[1])))
        if len(points) == 1:
            x, y = points[0]
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=0)
        elif len(points) > 1:
            draw.line(points, fill=0, width=max(2, round(min(width, height) / 110)), joint="curve")
    return image


class FrozenOcrEncoder:
    """Load a local VisionEncoderDecoder checkpoint without network access."""

    def __init__(self, model_path: Path, device: torch.device):
        from transformers import VisionEncoderDecoderModel
        if not model_path.exists():
            raise FileNotFoundError(f"OCR model snapshot is missing: {model_path}")
        self.device = device
        self.model = VisionEncoderDecoderModel.from_pretrained(
            str(model_path), local_files_only=True, torch_dtype=torch.float32,
        ).to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.encoder = self.model.get_encoder().eval()
        self.mean = 0.9545467
        self.std = 0.15394445
        self.image_size = 448
        self.channels = int(getattr(self.model.config.encoder, "num_channels", 1))

    def _tensor(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("L")
        width, height = image.size
        scale = min(self.image_size / max(width, 1), self.image_size / max(height, 1))
        new_width = max(1, min(self.image_size, round(width * scale)))
        new_height = max(1, min(self.image_size, round(height * scale)))
        resized = image.resize((new_width, new_height), Image.Resampling.BICUBIC)
        values = np.asarray(resized, dtype=np.float32) / 255.0
        values = (values - self.mean) / self.std
        canvas = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        canvas[:new_height, :new_width] = values
        tensor = torch.from_numpy(canvas).unsqueeze(0)
        if self.channels == 3:
            tensor = tensor.repeat(3, 1, 1)
        return tensor.unsqueeze(0)

    @torch.inference_mode()
    def encode(self, images: list[Image.Image], batch_size: int = 1) -> np.ndarray:
        if not images:
            raise ValueError("at least one formula image is required")
        outputs = []
        for start in range(0, len(images), batch_size):
            batch = torch.cat([self._tensor(image) for image in images[start:start + batch_size]], dim=0)
            hidden = self.encoder(pixel_values=batch.to(self.device)).last_hidden_state
            outputs.append(hidden.mean(dim=1).cpu().numpy().astype(np.float32))
        return np.concatenate(outputs, axis=0)


def _row_numeric(row: dict, candidate_index: int, version: str = "formula28") -> list[float]:
    """OCR 학습도 online 추론과 동일한 버전별 특징을 사용한다."""
    return _online_row_numeric(row, candidate_index, version)


def _make_examples(rows: list[dict], formula_features: dict[str, np.ndarray], token_to_id: dict[str, int]) -> list[dict]:
    examples = []
    for row in rows:
        formula_id = str(row["formula_id"])
        if formula_id not in formula_features:
            raise KeyError(f"missing OCR feature for {formula_id}")
        candidates = [str(token) for token in row["final_topk"]]
        target = str(row["label"])
        examples.append({
            "record_id": str(row.get("record_id", "")),
            "formula_id": formula_id,
            "writer_group": str(row.get("raw_writer_group") or row["writer_group"]),
            "numeric": np.asarray([_row_numeric(row, index) for index in range(len(candidates))], dtype=np.float32),
            "token_ids": np.asarray([token_to_id[token] for token in candidates], dtype=np.int64),
            "mask": np.ones(len(candidates), dtype=bool),
            "ocr": formula_features[formula_id],
            "target": candidates.index(target) if target in candidates else -1,
            "baseline": 0,
            "candidates": candidates,
            "label": target,
        })
    return examples


class OcrDecisionAdapter(nn.Module):
    """Scores each candidate independently, conditioned on formula OCR evidence."""

    def __init__(self, numeric_size: int, token_count: int, ocr_size: int, token_width: int = 48, hidden: int = 128):
        super().__init__()
        self.ocr_projection = nn.Sequential(nn.Linear(ocr_size, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.numeric_projection = nn.Sequential(nn.Linear(numeric_size, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.token_embedding = nn.Embedding(token_count, token_width)
        self.fusion = nn.Sequential(
            nn.Linear(hidden + hidden + token_width, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1),
        )

    def forward(self, numeric: torch.Tensor, token_ids: torch.Tensor, ocr: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        candidate = self.numeric_projection(numeric)
        evidence = self.ocr_projection(ocr).unsqueeze(1).expand(-1, numeric.shape[1], -1)
        token = self.token_embedding(token_ids)
        scores = self.fusion(torch.cat((candidate, evidence, token), dim=-1)).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        return scores


def _batch(examples: list[dict], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    width = max(len(example["candidates"]) for example in examples)
    numeric_size = examples[0]["numeric"].shape[-1]
    numeric = np.zeros((len(examples), width, numeric_size), dtype=np.float32)
    token_ids = np.zeros((len(examples), width), dtype=np.int64)
    mask = np.zeros((len(examples), width), dtype=bool)
    targets = np.full(len(examples), -100, dtype=np.int64)
    ocr = np.stack([example["ocr"] for example in examples]).astype(np.float32)
    for row_index, example in enumerate(examples):
        size = len(example["candidates"])
        numeric[row_index, :size] = example["numeric"]
        token_ids[row_index, :size] = example["token_ids"]
        mask[row_index, :size] = True
        if example["target"] >= 0:
            targets[row_index] = example["target"]
    return (
        torch.from_numpy(numeric).to(device), torch.from_numpy(token_ids).to(device),
        torch.from_numpy(ocr).to(device), torch.from_numpy(mask).to(device),
        torch.from_numpy(targets).to(device),
    )


def _train(model: OcrDecisionAdapter, examples: list[dict], epochs: int, seed: int, device: torch.device) -> None:
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise AssertionError("adapter has no trainable parameters")
    optimizer = torch.optim.AdamW(trainable, lr=3e-4, weight_decay=1e-2)
    rng = random.Random(seed)
    model.train()
    for _ in range(epochs):
        order = list(range(len(examples)))
        rng.shuffle(order)
        for start in range(0, len(order), 32):
            selected = [examples[index] for index in order[start:start + 32]]
            numeric, token_ids, ocr, mask, targets = _batch(selected, device)
            scores = model(numeric, token_ids, ocr, mask)
            loss = nn.functional.cross_entropy(scores, targets, ignore_index=-100)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
    model.eval()


@torch.inference_mode()
def _predict(model: OcrDecisionAdapter, examples: list[dict], device: torch.device) -> list[dict]:
    predictions = []
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


def _metrics(predictions: list[dict]) -> dict:
    covered = [row for row in predictions if row["target_in_candidates"]]
    def accuracy(key: str, rows: list[dict]) -> float:
        return sum(row[key] == row["label"] for row in rows) / len(rows) if rows else 0.0
    by_formula = defaultdict(list)
    for row in covered:
        by_formula[row["formula_id"]].append(row)
    return {
        "records": len(predictions), "covered_records": len(covered),
        "candidate_recall": len(covered) / len(predictions) if predictions else 0.0,
        "baseline_top1_covered": accuracy("baseline_token", covered),
        "adapter_top1_covered": accuracy("adapter_token", covered),
        "baseline_formula_exact_covered": sum(all(row["baseline_token"] == row["label"] for row in values) for values in by_formula.values()) / len(by_formula) if by_formula else 0.0,
        "adapter_formula_exact_covered": sum(all(row["adapter_token"] == row["label"] for row in values) for values in by_formula.values()) / len(by_formula) if by_formula else 0.0,
    }


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--formulas", type=Path, default=DEFAULT_FORMULAS)
    parser.add_argument("--ocr-model", type=Path, default=DEFAULT_TEXTELLER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None, help="limit to the first N formula IDs for a smoke run")
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.epochs < 1 or args.epochs > 20:
        parser.error("--epochs must be between 1 and 20")
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    _set_seed(args.seed)

    rows = _load_candidates(args.candidates, args.limit)
    formulas = _load_formula_records(args.formulas)
    required_formula_ids = list(dict.fromkeys(str(row["formula_id"]) for row in rows))
    missing = [formula_id for formula_id in required_formula_ids if formula_id not in formulas]
    if missing:
        raise ValueError(f"formula raster source is missing {len(missing)} formula IDs: {missing[:3]}")
    labels = sorted({str(token) for row in rows for token in row["final_topk"]})
    token_to_id = {token: index for index, token in enumerate(labels)}

    print(json.dumps({
        "event": "ocr_encoder_load", "schema": SCHEMA, "model": str(args.ocr_model),
        "revision": TEXTELLER_REVISION, "formulas": len(required_formula_ids),
        "records": len(rows), "device": str(device),
    }, ensure_ascii=False), flush=True)
    encoder = FrozenOcrEncoder(args.ocr_model, device)
    images = [_render_formula(formulas[formula_id]) for formula_id in required_formula_ids]
    features = encoder.encode(images)
    formula_features = dict(zip(required_formula_ids, features))
    examples = _make_examples(rows, formula_features, token_to_id)
    numeric_size = examples[0]["numeric"].shape[-1]
    ocr_size = int(features.shape[-1])

    by_writer = defaultdict(list)
    for example in examples:
        by_writer[example["writer_group"]].append(example)
    all_predictions = []
    folds = []
    for fold_index, writer in enumerate(sorted(by_writer)):
        train = [example for group, values in by_writer.items() if group != writer for example in values]
        test = list(by_writer[writer])
        model = OcrDecisionAdapter(numeric_size, len(labels), ocr_size).to(device)
        _train(model, train, args.epochs, args.seed + fold_index, device)
        predictions = _predict(model, test, device)
        all_predictions.extend(predictions)
        folds.append({"held_writer": writer, **_metrics(predictions)})

    # A refit checkpoint is useful for a later shadow runtime, but the fold
    # estimates above remain the only reported held-writer evidence.
    refit = OcrDecisionAdapter(numeric_size, len(labels), ocr_size).to(device)
    _train(refit, examples, args.epochs, args.seed + 1000, device)
    args.output.mkdir(parents=True)
    checkpoint = args.output / "ocr_decision_adapter.pt"
    torch.save({
        "schema": SCHEMA, "model_family": "frozen_texteller_encoder_candidate_adapter",
        "texteller_revision": TEXTELLER_REVISION, "numeric_size": numeric_size,
        "ocr_size": ocr_size, "labels": labels, "state_dict": refit.state_dict(),
            "candidate_contract": {"top_k_only": True, "token_creation": False, "row_deletion": False, "stroke_regrouping": False},
            "feature_contract": {"version": "formula-coordinate-context/v1", "point_keys": list(POINT_KEYS), "formula_geometry_keys": list(FORMULA_GEOMETRY_KEYS), "context_keys": list(CONTEXT_KEYS)},
    }, checkpoint)
    evaluation = {
        "schema": SCHEMA, "model": {"name": "OleehyO/TexTeller", "revision": TEXTELLER_REVISION, "weights_frozen": True},
        "data": {"candidate_path": str(args.candidates), "formula_path": str(args.formulas), "formula_count": len(required_formula_ids), "record_count": len(rows)},
        "training": {"epochs": args.epochs, "seed": args.seed, "trainable_adapter_parameters": sum(parameter.numel() for parameter in refit.parameters() if parameter.requires_grad), "external_trainable_parameters": 0},
        "writer_loo": folds, "aggregate": _metrics(all_predictions), "status": "shadow_only", "checkpoint": str(checkpoint),
    }
    (args.output / "evaluation.json").write_text(json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8")
    with gzip.open(args.output / "writer_loo_predictions.jsonl.gz", "wt", encoding="utf-8") as stream:
        for prediction in all_predictions:
            stream.write(json.dumps(prediction, ensure_ascii=False) + "\n")
    print(json.dumps({"event": "complete", "output": str(args.output), "aggregate": evaluation["aggregate"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
