#!/usr/bin/env python3
"""Offline-only TexTeller ViT teacher used to prepare bridge targets."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from safetensors import safe_open
from transformers import ViTConfig, ViTModel

import train_serial_ink_teller_10e as serial


TEACHER_GRID = 4


def render_formula(record: dict) -> Image.Image:
    canvas = record.get("canvas") or {}
    width = max(1, int(round(float(canvas.get("width", 327)))))
    height = max(1, int(round(float(canvas.get("height", 310)))))
    image = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(image)
    for stroke in record.get("strokes", []):
        points = []
        for point in stroke.get("points", []):
            x, y, _ = serial._point_xy(point)
            points.append((x, y))
        if len(points) == 1:
            x, y = points[0]
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=0)
        elif len(points) > 1:
            draw.line(points, fill=0, width=max(2, round(min(width, height) / 110)), joint="curve")
    return image


class OfflineTexTellerEncoder:
    def __init__(self, model_path: Path, device: torch.device):
        config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))["encoder"]
        self.model = ViTModel(ViTConfig.from_dict(config), add_pooling_layer=True)
        loaded = 0
        with safe_open(str(model_path / "model.safetensors"), framework="pt", device="cpu") as source:
            target_state = self.model.state_dict()
            for key in source.keys():
                if not key.startswith("encoder."):
                    continue
                target_key = key[len("encoder."):]
                if target_key not in target_state:
                    raise RuntimeError(f"encoder target key not found: {target_key}")
                target_state[target_key].copy_(source.get_tensor(key))
                loaded += 1
        if loaded < 160:
            raise RuntimeError(f"unexpectedly few TexTeller encoder tensors loaded: {loaded}")
        self.model.to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.device = device
        self.image_size = 448
        self.mean = 0.9545467
        self.std = 0.15394445

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
        return torch.from_numpy(canvas).unsqueeze(0)

    @torch.inference_mode()
    def encode(self, images: list[Image.Image], batch_size: int = 2) -> np.ndarray:
        chunks = []
        for start in range(0, len(images), batch_size):
            batch = torch.stack([self._tensor(image) for image in images[start:start + batch_size]]).to(self.device)
            hidden = self.model(pixel_values=batch).last_hidden_state
            patches = hidden[:, 1:, :]
            grid = int(math.sqrt(patches.shape[1]))
            if grid * grid != patches.shape[1]:
                raise RuntimeError(f"unexpected ViT patch count: {patches.shape[1]}")
            pooled = F.adaptive_avg_pool2d(
                patches.transpose(1, 2).reshape(patches.shape[0], patches.shape[2], grid, grid),
                (TEACHER_GRID, TEACHER_GRID),
            ).flatten(2).transpose(1, 2)
            chunks.append(pooled.cpu().numpy().astype(np.float32))
        return np.concatenate(chunks, axis=0)
