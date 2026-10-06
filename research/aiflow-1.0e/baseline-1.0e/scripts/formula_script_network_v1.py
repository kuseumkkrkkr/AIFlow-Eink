#!/usr/bin/env python3
"""Tiny commercial-safe neural challenger for superscript/subscript geometry."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch import nn


SCHEMA = "aiflow-formula-script-network/v1"
CLASSES = ("none", "superscript", "subscript")
FEATURES = (
    "dx_reference", "dy_reference", "horizontal_gap_reference",
    "log_child_parent_height", "log_child_parent_width",
    "child_height_reference", "parent_height_reference",
    "child_top_parent_center", "child_bottom_parent_center",
    "x_overlap", "parent_aspect_log", "child_aspect_log",
)
MODEL_CONFIG = {"input_size": len(FEATURES), "hidden_size": 24, "classes": len(CLASSES)}


class ScriptRelationNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(MODEL_CONFIG["input_size"], MODEL_CONFIG["hidden_size"]),
            nn.GELU(),
            nn.Linear(MODEL_CONFIG["hidden_size"], MODEL_CONFIG["classes"]),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


def pair_features(parent: dict[str, float], child: dict[str, float], reference: float) -> list[float]:
    reference = max(float(reference), 1e-6)
    parent_height, parent_width = max(parent["height"], 1e-6), max(parent["width"], 1e-6)
    child_height, child_width = max(child["height"], 1e-6), max(child["width"], 1e-6)
    overlap = max(0.0, min(parent["right"], child["right"]) - max(parent["left"], child["left"]))
    return [
        (child["cx"] - parent["cx"]) / reference,
        (child["cy"] - parent["cy"]) / reference,
        max(0.0, child["left"] - parent["right"]) / reference,
        math.log(child_height / parent_height),
        math.log(child_width / parent_width),
        child_height / reference,
        parent_height / reference,
        (child["top"] - parent["cy"]) / parent_height,
        (child["bottom"] - parent["cy"]) / parent_height,
        overlap / max(min(parent_width, child_width), 1e-6),
        math.log(parent_width / parent_height),
        math.log(child_width / child_height),
    ]


class NeuralScriptPredictor:
    def __init__(self, checkpoint: Path, device: str = "cpu") -> None:
        checkpoint = Path(checkpoint).expanduser().resolve()
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (
            payload.get("schema") != SCHEMA
            or payload.get("classes") != list(CLASSES)
            or payload.get("features") != list(FEATURES)
            or payload.get("model_config") != MODEL_CONFIG
            or not 0.0 < float(payload.get("threshold", 0.0)) < 1.0
            or float(payload.get("max_horizontal_gap_reference", 0.0)) <= 0.0
            or payload.get("training_data", {}).get("crohme_used") is not False
        ):
            raise ValueError("script-layout checkpoint contract mismatch")
        self.device = torch.device(device)
        self.model = ScriptRelationNetwork().to(self.device)
        self.model.load_state_dict(payload["state_dict"], strict=True)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.mean = np.asarray(payload["normalization"]["mean"], dtype=np.float32)
        self.scale = np.asarray(payload["normalization"]["scale"], dtype=np.float32)
        if self.mean.shape != (len(FEATURES),) or self.scale.shape != self.mean.shape or np.any(self.scale <= 0):
            raise ValueError("script-layout normalization contract mismatch")
        self.threshold = float(payload["threshold"])
        self.max_gap = float(payload["max_horizontal_gap_reference"])
        self.checkpoint = checkpoint
        self.payload = payload

    @torch.inference_mode()
    def propose(
        self, rows: list[dict], boxes: list[dict[str, float]], reference: float,
        blocked_children: set[int], structural_parents: set[int],
    ) -> list[tuple[int, int, str, float]]:
        pairs = []
        for child, child_box in enumerate(boxes):
            if child in blocked_children or child in structural_parents:
                continue
            for parent, parent_box in enumerate(boxes):
                if parent == child or parent_box["cx"] >= child_box["cx"]:
                    continue
                gap = max(0.0, child_box["left"] - parent_box["right"])
                if gap > reference * self.max_gap:
                    continue
                pairs.append((parent, child, pair_features(parent_box, child_box, reference)))
        if not pairs:
            return []
        matrix = (np.asarray([pair[2] for pair in pairs], dtype=np.float32) - self.mean) / self.scale
        probabilities = self.model(torch.from_numpy(matrix).to(self.device)).softmax(dim=1).cpu().numpy()
        best_by_child: dict[int, tuple[int, int, str, float]] = {}
        for (parent, child, _), probability in zip(pairs, probabilities, strict=True):
            relation_index = int(np.argmax(probability[1:])) + 1
            confidence = float(probability[relation_index])
            if confidence < self.threshold:
                continue
            proposal = (parent, child, CLASSES[relation_index], confidence)
            if child not in best_by_child or confidence > best_by_child[child][3]:
                best_by_child[child] = proposal
        return sorted(best_by_child.values(), key=lambda value: (-value[3], value[1], value[0]))
