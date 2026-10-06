#!/usr/bin/env python3
"""Extract frozen TexTeller or Microsoft TrOCR formula features offline.

The raster is derived from project-owned ordered strokes and is used only in
this shadow teacher step.  The output contains formula-level feature vectors;
it is never a product-runtime input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from accuracy_upgrade_contract_v1 import canonical_json_sha256
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from train_ocr_decision_adapter_10e import (
    _load_candidates,
    _load_formula_records,
    _render_formula,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATES = ROOT / "artifacts" / "ocr_trocr_hwr95_candidates_20260901_r1" / "candidates.jsonl.gz"
DEFAULT_FORMULAS = Path(
    r"D:\AIFlow-Workspace\PrivateData\math-ink-data-collector\derived"
    r"\public-candidate-20260819-r2\data\formulas_valid.jsonl"
)
SCHEMA = "aiflow-1.0e-offline-teacher-features/v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_files(model_path: Path) -> list[Path]:
    names = ("model.safetensors", "pytorch_model.bin", "config.json")
    return [model_path / name for name in names if (model_path / name).is_file()]


class FrozenTeacher:
    def __init__(self, kind: str, model_path: Path, device: torch.device) -> None:
        self.kind = kind
        self.device = device
        self.model = VisionEncoderDecoderModel.from_pretrained(
            str(model_path), local_files_only=True, torch_dtype=torch.float32
        ).to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.encoder = self.model.get_encoder().eval()
        self.channels = int(getattr(self.model.config.encoder, "num_channels", 3))
        self.processor = None
        if kind == "trocr-small":
            self.processor = TrOCRProcessor.from_pretrained(str(model_path), local_files_only=True)

    def _texteller_tensor(self, image) -> torch.Tensor:
        image = image.convert("L")
        width, height = image.size
        size = 448
        scale = min(size / max(width, 1), size / max(height, 1))
        new_width = max(1, min(size, round(width * scale)))
        new_height = max(1, min(size, round(height * scale)))
        resized = image.resize((new_width, new_height))
        values = np.asarray(resized, dtype=np.float32) / 255.0
        values = (values - 0.9545467) / 0.15394445
        canvas = np.zeros((size, size), dtype=np.float32)
        canvas[:new_height, :new_width] = values
        tensor = torch.from_numpy(canvas).unsqueeze(0)
        if self.channels == 3:
            tensor = tensor.repeat(3, 1, 1)
        return tensor

    @torch.inference_mode()
    def encode(self, images: list, batch_size: int) -> np.ndarray:
        outputs: list[np.ndarray] = []
        for start in range(0, len(images), batch_size):
            selected = images[start:start + batch_size]
            if self.kind == "texteller":
                pixel_values = torch.stack([self._texteller_tensor(image) for image in selected])
            else:
                assert self.processor is not None
                pixel_values = self.processor(images=[image.convert("RGB") for image in selected], return_tensors="pt").pixel_values
            hidden = self.encoder(pixel_values=pixel_values.to(self.device)).last_hidden_state
            outputs.append(hidden.mean(dim=1).float().cpu().numpy())
        return np.concatenate(outputs, axis=0).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("texteller", "trocr-small"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--formulas", type=Path, default=DEFAULT_FORMULAS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    rows = _load_candidates(args.candidates, None)
    formulas = _load_formula_records(args.formulas)
    formula_ids = list(dict.fromkeys(str(row["formula_id"]) for row in rows))
    missing = [formula_id for formula_id in formula_ids if formula_id not in formulas]
    if missing:
        raise ValueError(f"missing {len(missing)} formula records")
    files = checkpoint_files(args.model)
    if not files:
        raise FileNotFoundError(f"no complete local checkpoint in {args.model}")
    print(json.dumps({"event": "load", "kind": args.kind, "formulas": len(formula_ids), "device": args.device}), flush=True)
    teacher = FrozenTeacher(args.kind, args.model, torch.device(args.device))
    images = [_render_formula(formulas[formula_id]) for formula_id in formula_ids]
    features = teacher.encode(images, args.batch_size)
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output / "features.npz", formula_ids=np.asarray(formula_ids), features=features)
    manifest = {
        "schema": SCHEMA,
        "status": "shadow_only",
        "teacher": args.kind,
        "model_path": str(args.model),
        "checkpoint_sha256": {path.name: sha256(path) for path in files},
        "formula_count": len(formula_ids),
        "feature_shape": list(features.shape),
        "feature_dtype": str(features.dtype),
        "source_formula_sha256": {key: canonical_json_sha256({"strokes": formulas[key]["strokes"], "canvas": formulas[key].get("canvas", {})}) for key in formula_ids},
        "feature_file_sha256": sha256(args.output / "features.npz"),
        "preprocessing_sha256": sha256(Path(__file__)),
        "raster_source": "project-owned ordered strokes rendered offline",
        "product_runtime_input": False,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"event": "complete", **manifest}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
