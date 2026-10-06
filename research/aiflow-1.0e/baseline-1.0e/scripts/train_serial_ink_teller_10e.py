#!/usr/bin/env python3
"""AIFlow 1.0e serial experiment: online ink -> TexTeller decoder.

This is a research/shadow path.  It deliberately does not import PIL, render a
formula, or execute the TexTeller vision encoder.  The decoder is extracted
from the local TexTeller safetensors checkpoint and receives a learned 19ch
online-ink bridge.  The public AIFlow 0.6 online weights are loaded as the
trajectory prior; only that online adapter and the bridge are tuned.

The 0.6 checkpoint is research-only and its 19-channel contract is not the
final 1.0e five-channel contract.  This script therefore reports feasibility,
not a product promotion decision.
"""

from __future__ import annotations

import argparse
import gc
import gzip
import html
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from torch import nn
from transformers import AutoTokenizer, TrOCRConfig, TrOCRForCausalLM


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATES = ROOT / "artifacts" / "homograph_context_20260814" / "direct_candidates.jsonl.gz"
DEFAULT_FORMULAS = ROOT / "hf-dataset" / "data" / "formulas_valid.jsonl"
DEFAULT_TEXTELLER = Path(
    r"C:\Users\user\.cache\huggingface\hub\models--OleehyO--TexTeller"
    r"\snapshots\7b96df06b9d81cdb129c3bef68b7250bc3e2b0ea"
)
DEFAULT_AI_FLOW_BASE = Path(
    r"D:\AIFlow-Workspace\Caches\user\aiflow_math_ink_06_seed17"
    r"\models\seed17\base_378.pt"
)
DEFAULT_AI_FLOW_ADAPTER = Path(
    r"D:\AIFlow-Workspace\Caches\user\aiflow_math_ink_06_seed17"
    r"\models\seed17\online_adapter.pt"
)
DEFAULT_OUTPUT = ROOT / "artifacts" / "serial_ink_teller_10e_20260830"
TEXTELLER_REVISION = "7b96df06b9d81cdb129c3bef68b7250bc3e2b0ea"
SCHEMA = "aiflow-1.0e-serial-ink-teller/v1"
MAX_EVENTS = 128
SERIAL_STRIDE = 8
SERIAL_MEMORY = MAX_EVENTS // SERIAL_STRIDE
TEXTELLER_HIDDEN = 768
TEXTELLER_START = 2
TEXTELLER_EOS = 2
TEXTELLER_PAD = 1


def _json_lines(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_formula_records(path: Path) -> dict[str, dict]:
    return {
        str(row.get("sample_id", row.get("formula_id", ""))): row
        for row in _json_lines(path)
        if row.get("sample_id", row.get("formula_id", ""))
    }


def _load_candidate_rows(path: Path) -> list[dict]:
    rows = _json_lines(path)
    if not rows:
        raise ValueError(f"empty candidate file: {path}")
    required = {"formula_id", "writer_group", "label", "final_topk"}
    if any(not required <= set(row) for row in rows):
        raise ValueError("candidate rows do not satisfy the expected contract")
    return rows


def _target_to_tex(text: str) -> str:
    value = (
        str(text)
        .replace("×", r"\times")
        .replace("÷", r"\div")
        .replace("−", "-")
        .replace("–", "-")
    )
    # Match TexTeller's own sequence convention; canonical comparison removes
    # these delimiters again.
    return r"\[" + re.sub(r"\s+", "", value) + r"\]"


def _canonical(text: str) -> str:
    value = html.unescape(str(text)).strip()
    value = re.sub(r"\\(?:left|right|displaystyle|textstyle)", "", value)
    value = value.replace(r"\(", "").replace(r"\)", "")
    value = value.replace(r"\[", "").replace(r"\]", "")
    value = value.replace(r"\times", "*").replace("×", "*")
    value = value.replace(r"\div", "/").replace("÷", "/")
    value = value.replace(r"\cdot", "*")
    value = value.replace("−", "-").replace("–", "-")
    value = re.sub(r"\\(?:mathrm|mathbf|mathit)\{([^{}]*)\}", r"\1", value)
    value = re.sub(r"\s+", "", value)
    return value


class _TcnBlock(nn.Module):
    """State-compatible residual TCN block used by the public 0.6 encoder."""

    def __init__(self, hidden: int, kernel: int = 5):
        super().__init__()
        self.network = nn.ModuleList([
            nn.Conv1d(hidden, hidden, kernel, padding=kernel // 2),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Identity(),
            nn.Conv1d(hidden, hidden, 1),
            nn.LayerNorm(hidden),
        ])

    @staticmethod
    def _norm(layer: nn.LayerNorm, value: torch.Tensor) -> torch.Tensor:
        return layer(value.transpose(1, 2)).transpose(1, 2)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = value
        value = self.network[0](value)
        value = self._norm(self.network[1], value)
        value = self.network[2](value)
        value = self.network[4](value)
        value = self._norm(self.network[5], value)
        return residual + value


class _OnlineBlock(nn.Module):
    """State-compatible depthwise online adapter block."""

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.network = nn.ModuleList([
            nn.Conv1d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, 1, bias=False),
            nn.LayerNorm(hidden),
        ])

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = value
        value = self.network[0](value)
        value = self.network[1](value.transpose(1, 2)).transpose(1, 2)
        value = self.network[2](value)
        value = self.network[3](value)
        value = self.network[4](value.transpose(1, 2)).transpose(1, 2)
        return residual + value


class OnlinePrior(nn.Module):
    """AIFlow 0.6 online adapter + shared trajectory encoder, no raster path."""

    def __init__(self):
        super().__init__()
        self.online = nn.ModuleDict({
            "input_projection": nn.Sequential(nn.Conv1d(19, 64, 1)),
            "blocks": nn.ModuleList([_OnlineBlock(64) for _ in range(4)]),
            "output_projection": nn.Conv1d(64, 19, 1),
        })
        self.trajectory_encoder = nn.ModuleDict({
            "input_projection": nn.ModuleList([nn.Conv1d(19, 128, 1), nn.LayerNorm(128)]),
            "blocks": nn.ModuleList([_TcnBlock(128, 5) for _ in range(4)]),
            "attention": nn.Conv1d(128, 1, 1),
        })

    @staticmethod
    def _load(module: nn.Module, state: dict[str, torch.Tensor], prefix: str) -> None:
        selected = {
            key[len(prefix):]: value
            for key, value in state.items()
            if key.startswith(prefix)
        }
        result = module.load_state_dict(selected, strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(f"checkpoint mismatch for {prefix}: {result}")

    @classmethod
    def from_checkpoints(cls, base_path: Path, adapter_path: Path) -> "OnlinePrior":
        if not base_path.exists() or not adapter_path.exists():
            raise FileNotFoundError(f"AIFlow online checkpoints missing: {base_path}, {adapter_path}")
        base = torch.load(base_path, map_location="cpu", weights_only=False)
        adapter = torch.load(adapter_path, map_location="cpu", weights_only=False)
        model = cls()
        base_state = base["state_dict"]
        model._load(model.trajectory_encoder, base_state, "trajectory_encoder.")
        # The public checkpoint documents this order: base -> shared state -> online adapter.
        shared_state = adapter.get("shared_state_dict") or {}
        if shared_state:
            model._load(model.trajectory_encoder, shared_state, "trajectory_encoder.")
        model._load(model.online, adapter["state_dict"], "online.")
        return model

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        value = features.transpose(1, 2)
        correction = self.online["input_projection"](value)
        for block in self.online["blocks"]:
            correction = block(correction)
        value = value + self.online["output_projection"](correction)
        value = self.trajectory_encoder["input_projection"][0](value)
        value = self.trajectory_encoder["input_projection"][1](value.transpose(1, 2)).transpose(1, 2)
        for block in self.trajectory_encoder["blocks"]:
            value = block(value)
        return value.transpose(1, 2)


def _point_xy(point: dict | list | tuple) -> tuple[float, float, float]:
    if isinstance(point, dict):
        return float(point.get("x", 0.0)), float(point.get("y", 0.0)), float(point.get("t_ms", 0.0))
    return float(point[0]), float(point[1]), float(point[2]) if len(point) > 2 else 0.0


def _ink_features(record: dict) -> tuple[np.ndarray, np.ndarray]:
    """Encode real ordered raw points into the public 0.6 128x19 contract."""
    strokes = record.get("strokes") or []
    all_points: list[tuple[float, float, float, int, int, int]] = []
    for stroke_index, stroke in enumerate(strokes):
        points = stroke.get("points") or []
        for point_index, point in enumerate(points):
            x, y, timestamp = _point_xy(point)
            all_points.append((x, y, timestamp, stroke_index, point_index, len(points)))
    if not all_points:
        raise ValueError(f"formula {record.get('sample_id')} has no stroke points")
    if len(all_points) > MAX_EVENTS:
        selected = np.linspace(0, len(all_points) - 1, MAX_EVENTS).round().astype(int).tolist()
        all_points = [all_points[index] for index in selected]

    canvas = record.get("canvas") or {}
    canvas_w = max(float(canvas.get("width", 1.0)), 1.0)
    canvas_h = max(float(canvas.get("height", 1.0)), 1.0)
    xs = np.asarray([point[0] for point in all_points], dtype=np.float32)
    ys = np.asarray([point[1] for point in all_points], dtype=np.float32)
    min_x, max_x = float(xs.min()), float(xs.max())
    min_y, max_y = float(ys.min()), float(ys.max())
    bbox_w, bbox_h = max(max_x - min_x, 1e-3), max(max_y - min_y, 1e-3)
    diag = max(math.hypot(bbox_w, bbox_h), 1e-3)
    rows: list[list[float]] = []
    previous_direction = (0.0, 0.0)
    for index, (x, y, timestamp, stroke_index, point_index, stroke_length) in enumerate(all_points):
        if index and all_points[index - 1][3] == stroke_index:
            px, py, previous_time = all_points[index - 1][:3]
            dx, dy = x - px, y - py
            distance = math.hypot(dx, dy)
            dt = max(timestamp - previous_time, 0.0)
            direction = (dx / max(distance, 1e-5), dy / max(distance, 1e-5))
            curvature = direction[0] * previous_direction[1] - direction[1] * previous_direction[0]
            speed = min(distance / max(dt, 1.0) * 20.0 / diag, 4.0)
        else:
            dx = dy = distance = dt = curvature = speed = 0.0
            direction = (0.0, 0.0)
        previous_direction = direction
        stroke_points = [p for p in all_points if p[3] == stroke_index]
        sx = [p[0] for p in stroke_points]
        sy = [p[1] for p in stroke_points]
        stroke_w = max(max(sx) - min(sx), 1e-3)
        stroke_h = max(max(sy) - min(sy), 1e-3)
        center_y = ((min(sy) + max(sy)) * 0.5) / canvas_h
        progress = point_index / max(stroke_length - 1, 1)
        rows.append([
            (x - (min_x + max_x) * 0.5) / diag,
            (y - (min_y + max_y) * 0.5) / diag,
            x / canvas_w,
            y / canvas_h,
            direction[0],
            direction[1],
            float(np.clip(curvature, -1.0, 1.0)),
            float(point_index == stroke_length - 1),
            progress,
            float(np.clip(math.log(stroke_w / stroke_h), -4.0, 4.0) / 4.0),
            min(sy) / canvas_h,
            max(sy) / canvas_h,
            stroke_h / canvas_h,
            center_y,
            0.0,  # no baseline was observed in the source contract
            min(dt / 100.0, 4.0),
            speed,
            0.0,  # raw timestamps are present in this dataset
            1.0,  # online source modality
        ])
    values = np.asarray(rows, dtype=np.float32)
    mask = np.ones(len(values), dtype=bool)
    if len(values) < MAX_EVENTS:
        padded = np.zeros((MAX_EVENTS, values.shape[1]), dtype=np.float32)
        padded[:len(values)] = values
        values = padded
        mask = np.pad(mask, (0, MAX_EVENTS - len(mask)), constant_values=False)
    return values, mask


class SerialBridge(nn.Module):
    def __init__(self, input_size: int = 128, hidden: int = 512, output_size: int = TEXTELLER_HIDDEN):
        super().__init__()
        self.position = nn.Parameter(torch.zeros(1, SERIAL_MEMORY, input_size))
        nn.init.normal_(self.position, std=0.02)
        self.projection = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, hidden),
            nn.GELU(),
            nn.Linear(hidden, output_size),
            nn.LayerNorm(output_size),
        )

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        return self.projection(encoded + self.position[:, :encoded.shape[1]])


def _serial_tokens(encoded: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep ordered temporal evidence while reducing decoder cross-attention cost."""
    return encoded[:, ::SERIAL_STRIDE, :], mask[:, ::SERIAL_STRIDE]


def _load_decoder_only(model_path: Path, device: torch.device) -> TrOCRForCausalLM:
    config_path = model_path / "config.json"
    weights_path = model_path / "model.safetensors"
    if not config_path.exists() or not weights_path.exists():
        raise FileNotFoundError(f"TexTeller local files missing under {model_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))["decoder"]
    decoder = TrOCRForCausalLM(TrOCRConfig.from_dict(config))
    full_state = load_file(str(weights_path), device="cpu")
    decoder_state = {
        key[len("decoder."):]: value
        for key, value in full_state.items()
        if key.startswith("decoder.")
    }
    decoder_keys = decoder.state_dict()
    unknown = [key for key in decoder_state if key not in decoder_keys]
    if unknown:
        raise RuntimeError(f"decoder target key not found: {unknown[0]}")
    loaded = len(decoder_state)
    result = decoder.load_state_dict(decoder_state, strict=False)
    if result.unexpected_keys or len(result.missing_keys) > 1:
        raise RuntimeError(f"decoder checkpoint mismatch: {result}")
    del decoder_state
    del full_state
    gc.collect()
    decoder.tie_weights()
    decoder.config.use_cache = False
    print(json.dumps({"event": "decoder_state_dict_load_complete", "loaded": loaded}, ensure_ascii=False), flush=True)
    decoder.to(device).eval()
    print(json.dumps({"event": "decoder_device_transfer_complete", "device": str(device)}, ensure_ascii=False), flush=True)
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)
    if loaded < 300:
        raise RuntimeError(f"unexpectedly few TexTeller decoder tensors loaded: {loaded}")
    return decoder


def _enable_decoder_cross_attention(decoder: TrOCRForCausalLM) -> None:
    """Tune only the decoder's interface to online memory, not its LM body."""
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)
    for layer in decoder.model.decoder.layers:
        for parameter in layer.encoder_attn.parameters():
            parameter.requires_grad_(True)
        for parameter in layer.encoder_attn_layer_norm.parameters():
            parameter.requires_grad_(True)


def _cross_attention_state(decoder: TrOCRForCausalLM) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in decoder.state_dict().items()
        if ".encoder_attn." in key or ".encoder_attn_layer_norm." in key
    }


def _restore_cross_attention_state(decoder: TrOCRForCausalLM, state: dict[str, torch.Tensor]) -> None:
    current = decoder.state_dict()
    for key, value in state.items():
        current[key].copy_(value.to(current[key].device))


def _examples(rows: list[dict], formulas: dict[str, dict], include_all_raw: bool = False) -> tuple[list[dict], set[str]]:
    candidate_ids = set(str(row["formula_id"]) for row in rows)
    candidate_writer_groups = {
        str(formulas[str(row["formula_id"])].get("writer_id")): str(row["writer_group"])
        for row in rows
        if str(row["formula_id"]) in formulas
    }
    unique_ids = list(formulas) if include_all_raw else list(dict.fromkeys(str(row["formula_id"]) for row in rows))
    examples = []
    for formula_id in unique_ids:
        if formula_id not in formulas:
            raise KeyError(f"raw formula record missing for {formula_id}")
        record = formulas[formula_id]
        writer = candidate_writer_groups.get(str(record.get("writer_id")), f"aux_{record.get('writer_id', 'unknown')}")
        target = _target_to_tex(record["target_display"])
        values, mask = _ink_features(record)
        examples.append({
            "formula_id": formula_id,
            "writer_group": writer,
            "is_eval": formula_id in candidate_ids,
            "target": target,
            "features": values,
            "mask": mask,
        })
    return examples, candidate_ids


def _token_batch(tokenizer, examples: list[dict], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = tokenizer(
        [example["target"] for example in examples],
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt",
    )
    labels = encoded.input_ids.to(device)
    labels[labels == tokenizer.pad_token_id] = -100
    return labels, encoded.attention_mask.to(device)


def _shift_decoder_inputs(labels: torch.Tensor) -> torch.Tensor:
    """Match VisionEncoderDecoder shift-right for the standalone decoder."""
    shifted = torch.full_like(labels, TEXTELLER_PAD)
    shifted[:, 0] = TEXTELLER_START
    shifted[:, 1:] = labels[:, :-1]
    shifted[shifted == -100] = TEXTELLER_PAD
    return shifted


def _batch(examples: list[dict], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    features = torch.from_numpy(np.stack([example["features"] for example in examples])).to(device)
    mask = torch.from_numpy(np.stack([example["mask"] for example in examples])).to(device)
    return features, mask


def _train(
    online: OnlinePrior,
    bridge: SerialBridge,
    decoder: TrOCRForCausalLM,
    tokenizer,
    train_examples: list[dict],
    epochs: int,
    batch_size: int,
    seed: int,
    device: torch.device,
    online_lr: float,
    bridge_lr: float,
    decoder_lr: float,
) -> None:
    trainable = [parameter for parameter in list(online.parameters()) + list(bridge.parameters()) if parameter.requires_grad]
    trainable_decoder = [parameter for parameter in decoder.parameters() if parameter.requires_grad]
    trainable.extend(trainable_decoder)
    optimizer = torch.optim.AdamW([
        {"params": list(online.parameters()), "lr": online_lr},
        {"params": list(bridge.parameters()), "lr": bridge_lr},
        {"params": trainable_decoder, "lr": decoder_lr},
    ], weight_decay=1e-4)
    rng = random.Random(seed)
    online.train()
    bridge.train()
    decoder.eval()
    for _ in range(epochs):
        order = list(range(len(train_examples)))
        rng.shuffle(order)
        for start in range(0, len(order), batch_size):
            selected = [train_examples[index] for index in order[start:start + batch_size]]
            features, memory_mask = _batch(selected, device)
            labels, _ = _token_batch(tokenizer, selected, device)
            decoder_input_ids = _shift_decoder_inputs(labels)
            encoded, memory_mask = _serial_tokens(online(features), memory_mask)
            memory = bridge(encoded)
            outputs = decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=memory,
                encoder_attention_mask=memory_mask.long(),
                labels=labels,
                use_cache=False,
                return_dict=True,
            )
            if not torch.isfinite(outputs.loss):
                raise FloatingPointError(f"non-finite decoder loss: {outputs.loss.item()}")
            optimizer.zero_grad(set_to_none=True)
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
    online.eval()
    bridge.eval()


@torch.inference_mode()
def _greedy_decode(decoder, tokenizer, memory: torch.Tensor, memory_mask: torch.Tensor, max_length: int) -> str:
    tokens = torch.full((1, 1), TEXTELLER_START, dtype=torch.long, device=memory.device)
    for _ in range(max_length - 1):
        outputs = decoder(
            input_ids=tokens,
            encoder_hidden_states=memory,
            encoder_attention_mask=memory_mask.long(),
            use_cache=False,
            return_dict=True,
        )
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tokens = torch.cat((tokens, next_token), dim=1)
        if int(next_token.item()) == TEXTELLER_EOS:
            break
    return tokenizer.decode(tokens[0], skip_special_tokens=True)


@torch.inference_mode()
def _predict(online, bridge, decoder, tokenizer, examples, device, max_length: int) -> list[dict]:
    online.eval()
    bridge.eval()
    predictions = []
    for example in examples:
        features, memory_mask = _batch([example], device)
        encoded, memory_mask = _serial_tokens(online(features), memory_mask)
        memory = bridge(encoded)
        predicted = _greedy_decode(decoder, tokenizer, memory, memory_mask, max_length)
        predictions.append({
            "formula_id": example["formula_id"],
            "writer_group": example["writer_group"],
            "target": example["target"],
            "prediction": predicted,
            "target_canonical": _canonical(example["target"]),
            "prediction_canonical": _canonical(predicted),
            "exact": _canonical(example["target"]) == _canonical(predicted),
        })
    return predictions


def _baseline_metrics(rows: list[dict], formulas: dict[str, dict]) -> dict:
    grouped: defaultdict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["formula_id"])].append(row)
    exact = 0
    for formula_id, formula_rows in grouped.items():
        prediction = " ".join(str(row["final_topk"][0]) for row in formula_rows)
        target = _target_to_tex(formulas[formula_id]["target_display"])
        exact += int(_canonical(prediction) == _canonical(target))
    return {"formula_exact": exact, "formula_count": len(grouped), "formula_exact_rate": exact / max(1, len(grouped))}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--formulas", type=Path, default=DEFAULT_FORMULAS)
    parser.add_argument("--texteller", type=Path, default=DEFAULT_TEXTELLER)
    parser.add_argument("--ai-flow-base", type=Path, default=DEFAULT_AI_FLOW_BASE)
    parser.add_argument("--ai-flow-adapter", type=Path, default=DEFAULT_AI_FLOW_ADAPTER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=48)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--online-lr", type=float, default=1e-5)
    parser.add_argument("--bridge-lr", type=float, default=1e-3)
    parser.add_argument("--decoder-lr", type=float, default=2e-5)
    parser.add_argument("--include-all-raw", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    device = torch.device(args.device)
    _set_seed(args.seed)
    rows = _load_candidate_rows(args.candidates)
    formulas = _load_formula_records(args.formulas)
    examples, candidate_ids = _examples(rows, formulas, args.include_all_raw)
    tokenizer = AutoTokenizer.from_pretrained(str(args.texteller), local_files_only=True)
    print(json.dumps({
        "event": "serial_model_load",
        "schema": SCHEMA,
        "device": str(device),
        "formulas": len(candidate_ids),
        "train_formulas": len(examples),
        "texteller_revision": TEXTELLER_REVISION,
        "raster_encoder_executed": False,
        "online_checkpoint": str(args.ai_flow_adapter),
    }, ensure_ascii=False), flush=True)
    decoder = _load_decoder_only(args.texteller, device)
    _enable_decoder_cross_attention(decoder)
    initial_cross_attention = _cross_attention_state(decoder)
    online_template = OnlinePrior.from_checkpoints(args.ai_flow_base, args.ai_flow_adapter)
    for parameter in online_template.trajectory_encoder.parameters():
        parameter.requires_grad_(False)
    for parameter in online_template.online.parameters():
        parameter.requires_grad_(True)
    online_template.to(device)
    trainable_online = sum(parameter.numel() for parameter in online_template.online.parameters() if parameter.requires_grad)
    trainable_encoder = sum(parameter.numel() for parameter in decoder.parameters() if parameter.requires_grad)
    bridge_parameter_count = sum(parameter.numel() for parameter in SerialBridge().parameters())
    all_predictions: list[dict] = []
    folds = []
    by_writer: defaultdict[str, list[dict]] = defaultdict(list)
    for example in examples:
        if example["is_eval"]:
            by_writer[example["writer_group"]].append(example)
    for fold_index, held_writer in enumerate(sorted(by_writer)):
        _restore_cross_attention_state(decoder, initial_cross_attention)
        train = [example for example in examples if example["writer_group"] != held_writer]
        test = [example for example in examples if example["is_eval"] and example["writer_group"] == held_writer]
        online = OnlinePrior()
        online.load_state_dict(online_template.state_dict())
        bridge = SerialBridge().to(device)
        online.to(device)
        _train(online, bridge, decoder, tokenizer, train, args.epochs, args.batch_size, args.seed + fold_index, device, args.online_lr, args.bridge_lr, args.decoder_lr)
        predictions = _predict(online, bridge, decoder, tokenizer, test, device, args.max_length)
        all_predictions.extend(predictions)
        folds.append({
            "held_writer": held_writer,
            "formula_count": len(test),
            "exact": sum(int(row["exact"]) for row in predictions),
            "exact_rate": sum(int(row["exact"]) for row in predictions) / max(1, len(predictions)),
        })
        del online, bridge
        if device.type == "cuda":
            torch.cuda.empty_cache()
    exact = sum(int(row["exact"]) for row in all_predictions)
    args.output.mkdir(parents=True)
    checkpoint = args.output / "serial_bridge.pt"
    # Refit only for a shadow runtime artifact; held-writer folds above remain the evidence.
    _restore_cross_attention_state(decoder, initial_cross_attention)
    refit_online = OnlinePrior()
    refit_online.load_state_dict(online_template.state_dict())
    refit_online.to(device)
    refit_bridge = SerialBridge().to(device)
    _train(refit_online, refit_bridge, decoder, tokenizer, examples, args.epochs, args.batch_size, args.seed + 1000, device, args.online_lr, args.bridge_lr, args.decoder_lr)
    torch.save({
        "schema": SCHEMA,
        "texteller_revision": TEXTELLER_REVISION,
        "ai_flow_base": str(args.ai_flow_base),
        "ai_flow_adapter": str(args.ai_flow_adapter),
        "raster_encoder_executed": False,
        "online_state_dict": refit_online.online.state_dict(),
        "bridge_state_dict": refit_bridge.state_dict(),
        "decoder_cross_attention_state_dict": _cross_attention_state(decoder),
        "input_contract": {"source": "ordered raw strokes", "channels": 19, "max_events": MAX_EVENTS},
        "output_contract": {"decoder": "TexTeller decoder-only", "free_form_latex": True},
    }, checkpoint)
    evaluation = {
        "schema": SCHEMA,
        "status": "shadow_only",
        "runtime_contract": {
            "serial": ["ordered_online_ink", "AIFlow_online_prior", "bridge_128_to_768_with_16_temporal_tokens", "TexTeller_decoder"],
            "raster_encoder_executed": False,
            "candidate_only": False,
            "free_form_decoder": True,
        },
        "model": {
            "texteller_revision": TEXTELLER_REVISION,
            "decoder_parameters_loaded": sum(parameter.numel() for parameter in decoder.parameters()),
            "decoder_trainable_parameters": sum(parameter.numel() for parameter in decoder.parameters() if parameter.requires_grad),
            "ai_flow_online_trainable_parameters": trainable_online,
            "bridge_trainable_parameters": bridge_parameter_count,
            "ai_flow_trajectory_frozen": True,
            "ai_flow_product_validation": False,
        },
        "data": {
            "candidate_path": str(args.candidates),
            "formula_path": str(args.formulas),
            "formula_count": len(candidate_ids),
            "train_formula_count": len(examples),
            "auxiliary_formula_count": len(examples) - len(candidate_ids),
            "writer_count": len(by_writer),
            "raw_stroke_input": True,
            "serial_memory_tokens": SERIAL_MEMORY,
        },
        "training": {"epochs": args.epochs, "batch_size": args.batch_size, "seed": args.seed, "online_lr": args.online_lr, "bridge_lr": args.bridge_lr, "decoder_lr": args.decoder_lr, "include_all_raw": args.include_all_raw},
        "baseline": _baseline_metrics(rows, formulas),
        "writer_loo": folds,
        "aggregate": {"formula_exact": exact, "formula_count": len(all_predictions), "formula_exact_rate": exact / max(1, len(all_predictions))},
        "checkpoint": str(checkpoint),
    }
    (args.output / "evaluation.json").write_text(json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8")
    with gzip.open(args.output / "writer_loo_predictions.jsonl.gz", "wt", encoding="utf-8") as stream:
        for row in all_predictions:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"event": "serial_complete", "output": str(args.output), "aggregate": evaluation["aggregate"], "baseline": evaluation["baseline"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
