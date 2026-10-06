#!/usr/bin/env python3
"""Scratch-only writer-disjoint scale probe for the 372-class online HWR encoder.

The UJI inner validation writers come only from the official UJI train writers.
The official UJI test split is never scored, and prior checkpoints/teacher logits
are not loaded. This is a capacity diagnostic, not a product acceptance run.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from build_normalized_ink_v1 import ROOT, _json_lines, _record_id, _sha256
from character_tensor_v1 import CHANNELS, POINTS, tensorize
from evaluate_48hz_prefix_v1 import _load_model
from run_hwr_affine_distillation_experiment_v1 import _synthesize_equal, _tensor_checks
from train_character_classifier_v1 import EXTRA_MATH_LABELS, apply_input_mode


DEFAULT_CANONICAL_ROOT = Path(
    r"D:\AIFlow-Workspace\PrivateData\math-ink-data-collector\derived"
    r"\hwr-augmentation-20261001\canonical-trainpool"
)
DEFAULT_CURATED = ROOT / "datasets" / "10_approved_external" / "uji_pen_characters_v2" / "derived" / "uji_math_curated.jsonl.gz"
DEFAULT_SPLIT = ROOT / "artifacts" / "hwr_augmentation_microscope_20261002" / "uji_writer_group_split_seed20261002.json"
DEFAULT_VOCAB_CHECKPOINT = Path(
    r"D:\AIFlow-Workspace\PrivateData\math-ink-data-collector\derived"
    r"\hwr-head-calibration-20260819-r1\final_all_writers_steps250_lr1e-3"
    r"\project_symbol_head_checkpoint.pt"
)
DEFAULT_CACHE = Path(
    r"D:\AIFlow-Workspace\PrivateData\math-ink-data-collector\derived"
    r"\hwr-architecture-20261002\uji-writer-inner-scratch-cache"
)
DEFAULT_OUTPUT = ROOT / "artifacts" / "hwr_augmentation_microscope_20261002" / "architecture_scale_uji_inner_writer_probe.json"
SYNTHETIC_EQUAL_ROWS = 512


class ScaledInkClassifier(nn.Module):
    def __init__(self, classes: int, width: int, layers: int, heads: int, feedforward: int) -> None:
        super().__init__()
        if width <= 0 or layers <= 0 or heads <= 0 or width % heads:
            raise ValueError("width/layers/heads must be positive and width divisible by heads")
        self.input_projection = nn.Sequential(nn.Linear(len(CHANNELS), width), nn.LayerNorm(width), nn.GELU())
        self.position = nn.Parameter(torch.empty(1, POINTS, width))
        nn.init.trunc_normal_(self.position, std=0.02)
        block = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=feedforward, dropout=0.1,
            activation="gelu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(block, num_layers=layers)
        self.pool_score = nn.Linear(width, 1)
        self.math_head = nn.Linear(width, classes)

    def encode(self, points: torch.Tensor) -> torch.Tensor:
        if points.ndim != 3 or points.shape[1:] != (POINTS, len(CHANNELS)):
            raise ValueError(f"expected batch x {POINTS} x {len(CHANNELS)}, got {tuple(points.shape)}")
        hidden = self.encoder(self.input_projection(points) + self.position)
        weights = F.softmax(self.pool_score(hidden).squeeze(-1), dim=1)
        return torch.sum(hidden * weights.unsqueeze(-1), dim=1)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return self.math_head(self.encode(points))


class NpyDataset(Dataset):
    def __init__(self, features_path: Path, labels_path: Path) -> None:
        self.features = np.load(features_path, mmap_mode="r")
        self.labels = np.load(labels_path, mmap_mode="r")
        if self.features.ndim != 3 or self.features.shape[1:] != (POINTS, len(CHANNELS)) or len(self.features) != len(self.labels):
            raise ValueError(f"invalid cached arrays at {features_path}")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return torch.from_numpy(np.array(self.features[index], copy=True)), int(self.labels[index])


def _writer_digest(writer_key: str) -> str:
    return hashlib.sha256(writer_key.encode("utf-8")).hexdigest()[:16]


def _load_writer_map(curated_path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in _json_lines(curated_path):
        record_id = _record_id("uji_pen_v2", str(row["sample_id"]))
        if record_id in result:
            raise ValueError(f"duplicate curated UJI record ID: {record_id}")
        result[record_id] = str(row["writer_key"])
    return result


def _iter_train_rows(canonical_root: Path, source: str, labels: set[str]):
    for row in _json_lines(canonical_root / f"{source}.jsonl.gz"):
        if row.get("split") == "train" and str(row.get("label")) in labels:
            yield row


def _split_rows(canonical_root: Path, writer_map: dict[str, str], validation_hashes: set[str], labels: set[str]) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    counts = {"train": 0, "validation": 0}
    labels_seen = {"train": Counter(), "validation": Counter()}
    for source in ("hwrt", "uji"):
        for row in _iter_train_rows(canonical_root, source, labels):
            partition = "train"
            if source == "uji":
                writer_key = writer_map.get(str(row["record_id"]))
                if writer_key is None:
                    raise ValueError(f"UJI row missing writer join: {row['record_id']}")
                if _writer_digest(writer_key) in validation_hashes:
                    partition = "validation"
            counts[partition] += 1
            labels_seen[partition][str(row["label"])] += 1
    return counts, {key: dict(sorted(value.items())) for key, value in labels_seen.items()}


def _class_labels(canonical_root: Path) -> list[str]:
    labels = sorted({str(row["label"]) for row in _json_lines(canonical_root / "hwrt.jsonl.gz")})
    if len(labels) != 369 or any(label in labels for label in EXTRA_MATH_LABELS):
        raise ValueError(f"unexpected frozen HWR base vocabulary: {len(labels)} HWRT labels")
    result = labels + list(EXTRA_MATH_LABELS)
    if len(result) != 372 or len(set(result)) != 372:
        raise AssertionError("372-class vocabulary construction failed")
    return result


def _accumulate_counts(counter: Counter, ids: torch.Tensor, classes: int) -> None:
    values = torch.bincount(ids.cpu(), minlength=classes).tolist()
    counter.update({index: count for index, count in enumerate(values) if count})


def _prepare_arrays(
    canonical_root: Path,
    curated_path: Path,
    split_path: Path,
    labels: list[str],
    cache_root: Path,
    equal_count: int,
) -> dict[str, Any]:
    if cache_root.exists() and any(cache_root.iterdir()):
        raise FileExistsError(f"refusing to reuse non-empty architecture cache: {cache_root}")
    cache_root.mkdir(parents=True, exist_ok=True)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if split.get("schema") != "aiflow-hwr-uji-writer-group-split/v1" or split.get("status") != "completed":
        raise ValueError("invalid writer split manifest")
    if split["inputs"]["canonical_uji_sha256"] != _sha256(canonical_root / "uji.jsonl.gz"):
        raise ValueError("writer split was produced from a different canonical UJI source")
    if split["inputs"]["curated_uji_sha256"] != _sha256(curated_path):
        raise ValueError("writer split was produced from a different curated UJI source")
    validation_hashes = set(split["inner_split"]["validation_writer_hashes"])
    writer_map = _load_writer_map(curated_path)
    label_set = set(labels)
    expected_counts, expected_labels = _split_rows(canonical_root, writer_map, validation_hashes, label_set)
    if not expected_counts["train"] or not expected_counts["validation"]:
        raise ValueError("empty architecture train or validation cohort")
    missing_train = sorted(label_set - set(expected_labels["train"]) - {"="})
    if missing_train:
        raise ValueError(f"training cohort missing model labels: {missing_train[:12]}")
    if not set(expected_labels["validation"]) <= label_set:
        raise AssertionError("validation labels escaped the model vocabulary")

    paths = {
        name: {
            "features": cache_root / f"{name}_features.npy",
            "labels": cache_root / f"{name}_labels.npy",
        }
        for name in ("train", "validation")
    }
    arrays = {
        name: (
            np.lib.format.open_memmap(item["features"], mode="w+", dtype=np.float32, shape=(expected_counts[name] + (equal_count if name == "train" else 0), POINTS, len(CHANNELS))),
            np.lib.format.open_memmap(item["labels"], mode="w+", dtype=np.int16, shape=(expected_counts[name] + (equal_count if name == "train" else 0),)),
        )
        for name, item in paths.items()
    }
    label_to_id = {label: index for index, label in enumerate(labels)}
    offsets = Counter()
    counts_by_source: dict[str, Counter] = {
        "train:hwrt": Counter(), "train:uji": Counter(),
        "validation:uji": Counter(), "train:synthetic_equal": Counter(),
    }
    dash_features: list[np.ndarray] = []
    for source in ("hwrt", "uji"):
        for row in _iter_train_rows(canonical_root, source, label_set):
            partition = "train"
            if source == "uji":
                writer_key = writer_map.get(str(row["record_id"]))
                if writer_key is None:
                    raise ValueError(f"UJI row missing writer join: {row['record_id']}")
                if _writer_digest(writer_key) in validation_hashes:
                    partition = "validation"
            feature = apply_input_mode(tensorize(row), "uniform-time")
            _tensor_checks(feature)
            if partition == "train" and source == "hwrt" and str(row["label"]) == "-" and len(dash_features) < 4096:
                dash_features.append(feature.copy())
            target = label_to_id[str(row["label"])]
            feature_array, label_array = arrays[partition]
            feature_array[offsets[partition]] = feature
            label_array[offsets[partition]] = target
            offsets[partition] += 1
            counts_by_source[f"{partition}:{source}"][str(row["label"])] += 1
    if dict(offsets) != expected_counts:
        raise AssertionError(f"writer split/cache count mismatch: {dict(offsets)} != {expected_counts}")
    if equal_count:
        if not dash_features:
            raise ValueError("cannot synthesize '=' without HWRT training dashes")
        equal = _synthesize_equal(np.stack(dash_features), equal_count, 20261002)
        for feature in equal:
            _tensor_checks(feature)
        train_features, train_labels = arrays["train"]
        start = offsets["train"]
        train_features[start:start + equal_count] = equal
        train_labels[start:start + equal_count] = label_to_id["="]
        counts_by_source["train:synthetic_equal"]["="] = equal_count
        offsets["train"] += equal_count
    for feature_array, label_array in arrays.values():
        feature_array.flush()
        label_array.flush()
    del arrays

    return {
        "cache": {key: {name: str(path.resolve()) for name, path in value.items()} for key, value in paths.items()},
        "rows": {"train": int(offsets["train"]), "validation": int(offsets["validation"])},
        "real_rows": expected_counts,
        "rows_by_source_label": {key: dict(sorted(value.items())) for key, value in counts_by_source.items()},
        "train_label_coverage": len(set(expected_labels["train"]) | ({"="} if equal_count else set())),
        "validation_label_coverage": len(expected_labels["validation"]),
        "validation_missing_model_labels": sorted(label_set - set(expected_labels["validation"])),
        "crohme_rows": 0,
    }


def _inspect_prepared_cache(
    canonical_root: Path,
    curated_path: Path,
    split_path: Path,
    labels: list[str],
    cache_root: Path,
    equal_count: int,
) -> dict[str, Any]:
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if split.get("schema") != "aiflow-hwr-uji-writer-group-split/v1" or split.get("status") != "completed":
        raise ValueError("invalid writer split manifest")
    if split["inputs"]["canonical_uji_sha256"] != _sha256(canonical_root / "uji.jsonl.gz"):
        raise ValueError("writer split was produced from a different canonical UJI source")
    if split["inputs"]["curated_uji_sha256"] != _sha256(curated_path):
        raise ValueError("writer split was produced from a different curated UJI source")
    writer_map = _load_writer_map(curated_path)
    expected_counts, expected_labels = _split_rows(
        canonical_root, writer_map, set(split["inner_split"]["validation_writer_hashes"]), set(labels),
    )
    names = ("train_features.npy", "train_labels.npy", "validation_features.npy", "validation_labels.npy")
    if any(not (cache_root / name).is_file() for name in names):
        raise FileNotFoundError(f"prepared architecture cache is incomplete: {cache_root}")
    train = NpyDataset(cache_root / "train_features.npy", cache_root / "train_labels.npy")
    validation = NpyDataset(cache_root / "validation_features.npy", cache_root / "validation_labels.npy")
    expected_train_rows = expected_counts["train"] + equal_count
    if len(train) != expected_train_rows or len(validation) != expected_counts["validation"]:
        raise ValueError(
            f"prepared cache row mismatch: train={len(train)}/{expected_train_rows}, "
            f"validation={len(validation)}/{expected_counts['validation']}"
        )
    train_labels = np.asarray(train.labels, dtype=np.int64)
    validation_labels = np.asarray(validation.labels, dtype=np.int64)
    if train_labels.size and (train_labels.min() < 0 or train_labels.max() >= len(labels)):
        raise ValueError("prepared train labels are outside the frozen vocabulary")
    if validation_labels.size and (validation_labels.min() < 0 or validation_labels.max() >= len(labels)):
        raise ValueError("prepared validation labels are outside the frozen vocabulary")
    train_support = np.bincount(train_labels, minlength=len(labels))
    validation_support = np.bincount(validation_labels, minlength=len(labels))
    if not np.all(train_support > 0) or train_support[labels.index("=")] != equal_count:
        raise ValueError("prepared cache does not preserve all train classes and exact synthetic '=' count")
    val_present = {labels[index] for index in np.flatnonzero(validation_support)}
    return {
        "cache": {
            "train": {"features": str((cache_root / "train_features.npy").resolve()), "labels": str((cache_root / "train_labels.npy").resolve())},
            "validation": {"features": str((cache_root / "validation_features.npy").resolve()), "labels": str((cache_root / "validation_labels.npy").resolve())},
        },
        "rows": {"train": len(train), "validation": len(validation)},
        "real_rows": expected_counts,
        "train_label_coverage": int(np.count_nonzero(train_support)),
        "validation_label_coverage": len(val_present),
        "validation_missing_model_labels": sorted(set(labels) - val_present),
        "prepared_cache_sha256": {name: _sha256(cache_root / name) for name in names},
        "crohme_rows": 0,
    }


def _self_test_gradient_accumulation() -> None:
    torch.manual_seed(203)
    full_batch = nn.Linear(5, 7)
    micro_batch = nn.Linear(5, 7)
    micro_batch.load_state_dict(full_batch.state_dict())
    features = torch.randn(6, 5)
    targets = torch.tensor([0, 3, 2, 1, 6, 5])
    F.cross_entropy(full_batch(features), targets).backward()
    for start in range(0, len(targets), 2):
        stop = min(start + 2, len(targets))
        loss = F.cross_entropy(micro_batch(features[start:stop]), targets[start:stop])
        (loss * ((stop - start) / len(targets))).backward()
    max_error = max(
        float((left.grad - right.grad).abs().max())
        for left, right in zip(full_batch.parameters(), micro_batch.parameters(), strict=True)
    )
    if max_error > 1.0e-6:
        raise AssertionError(f"microbatch gradient accumulation differs from effective-batch gradient: {max_error}")


@torch.inference_mode()
def _evaluate(model: nn.Module, dataset: NpyDataset, labels: list[str], device: torch.device, batch_size: int) -> dict[str, Any]:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    hits1 = hits5 = total = 0
    margins: list[np.ndarray] = []
    class_hits = Counter()
    class_counts = Counter()
    class_target_ranks: dict[int, Counter] = {}
    rivals = Counter()
    for features, target in loader:
        features = features.to(device, non_blocking=device.type == "cuda")
        target_device = target.to(device, non_blocking=device.type == "cuda")
        logits = model(features).float()
        top = logits.topk(5, dim=1).indices
        hits1 += int((top[:, 0] == target_device).sum().item())
        hits5 += int((top == target_device[:, None]).any(dim=1).sum().item())
        target_score = logits.gather(1, target_device[:, None]).squeeze(1)
        competitor = logits.clone()
        competitor.scatter_(1, target_device[:, None], float("-inf"))
        rival_score, rival_id = competitor.max(dim=1)
        margins.append((target_score - rival_score).cpu().numpy())
        _accumulate_counts(rivals, rival_id, len(labels))
        target_in_top = top == target_device[:, None]
        target_rank = target_in_top.to(torch.int64).argmax(dim=1) + 1
        target_rank = torch.where(target_in_top.any(dim=1), target_rank, 6)
        for target_id, correct, rank in zip(
            target.cpu().tolist(),
            (top[:, 0] == target_device).cpu().tolist(),
            target_rank.cpu().tolist(),
        ):
            class_counts[target_id] += 1
            class_hits[target_id] += int(correct)
            class_target_ranks.setdefault(target_id, Counter())[int(rank)] += 1
        total += len(target)
    margin_values = np.concatenate(margins) if margins else np.empty(0, dtype=np.float32)
    macro_top1 = float(np.mean([class_hits[index] / count for index, count in class_counts.items()])) if class_counts else 0.0
    per_class = []
    for class_id, support in sorted(class_counts.items()):
        rank_counts = class_target_ranks[class_id]
        top1_hits = int(rank_counts[1])
        top5_hits = int(sum(rank_counts[rank] for rank in range(1, 6)))
        per_class.append({
            "class_id": int(class_id),
            "label": labels[class_id],
            "support": int(support),
            "top1_hits": top1_hits,
            "top1": top1_hits / support,
            "top5_hits": top5_hits,
            "top5": top5_hits / support,
            "target_rank_histogram": {str(rank): int(rank_counts[rank]) for rank in range(1, 7)},
        })
    return {
        "rows": total,
        "top1_hits": hits1,
        "top5_hits": hits5,
        "top1": hits1 / total if total else 0.0,
        "top5": hits5 / total if total else 0.0,
        "macro_top1_over_present_labels": macro_top1,
        "present_labels": len(class_counts),
        "per_class": per_class,
        "missing_labels": sorted(set(range(len(labels))) - set(class_counts)),
        "target_minus_hardest_rival_margin": {
            "mean": float(margin_values.mean()) if len(margin_values) else None,
            "p10": float(np.quantile(margin_values, 0.10)) if len(margin_values) else None,
            "positive_fraction": float((margin_values > 0).mean()) if len(margin_values) else None,
        },
        "hardest_rivals_top10": [{"label": labels[index], "count": count} for index, count in rivals.most_common(10)],
    }


def _train_arm(
    name: str,
    width: int,
    layers: int,
    heads: int,
    epochs: int,
    batch_size: int,
    microbatch_size: int,
    learning_rate: float,
    seed: int,
    labels: list[str],
    cache: dict[str, Any],
    device: torch.device,
    checkpoint_dir: Path,
    eval_batch_size: int,
) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model = ScaledInkClassifier(len(labels), width, layers, heads, width * 4).to(device)
    params = sum(parameter.numel() for parameter in model.parameters())
    train_dataset = NpyDataset(Path(cache["train"]["features"]), Path(cache["train"]["labels"]))
    val_dataset = NpyDataset(Path(cache["validation"]["features"]), Path(cache["validation"]["labels"]))
    train_label_ids = np.asarray(train_dataset.labels, dtype=np.int64)
    counts = np.bincount(train_label_ids, minlength=len(labels)).astype(np.float64)
    if not np.all(counts > 0):
        raise ValueError("class-balanced training requires support for all model labels")
    sampler = WeightedRandomSampler(
        torch.as_tensor(1.0 / counts[train_label_ids], dtype=torch.double),
        num_samples=len(train_label_ids), replacement=True,
        generator=torch.Generator().manual_seed(seed + 17),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=0, pin_memory=device.type == "cuda")
    # Fused AdamW avoids full-parameter temporary tensors on the 4 GiB CUDA
    # probe device; keep the ordinary implementation for CPU runs.
    fused_adamw = device.type == "cuda"
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1.0e-4,
        fused=fused_adamw, foreach=False,
    )
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    history: list[dict[str, Any]] = []
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        epoch_margin: list[np.ndarray] = []
        rival_counts = Counter()
        epoch_started = time.perf_counter()
        for batch_index, (features, target) in enumerate(train_loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            batch_rows = int(target.shape[0])
            batch_loss = 0.0
            for start in range(0, batch_rows, microbatch_size):
                stop = min(start + microbatch_size, batch_rows)
                micro_features = features[start:stop].to(device, non_blocking=amp)
                micro_target = target[start:stop].to(device, non_blocking=amp)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                    logits = model(micro_features)
                    loss = F.cross_entropy(logits.float(), micro_target)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite CE in {name}, epoch={epoch}, batch={batch_index}")
                weight = (stop - start) / batch_rows
                scaler.scale(loss * weight).backward()
                batch_loss += float(loss.detach().cpu()) * weight
                with torch.no_grad():
                    scores = logits.float().detach()
                    rival_logits = scores.clone()
                    rival_logits.scatter_(1, micro_target[:, None], float("-inf"))
                    rival_score, rival_id = rival_logits.max(dim=1)
                    target_score = scores.gather(1, micro_target[:, None]).squeeze(1)
                    epoch_margin.append((target_score - rival_score).cpu().numpy())
                    _accumulate_counts(rival_counts, rival_id, len(labels))
            scaler.unscale_(optimizer)
            # The foreach implementation allocates temporary buffers for every
            # parameter tensor. Keep this probe viable on 4 GiB mobile-class
            # GPUs while preserving the same global-norm clipping rule.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=False)
            scaler.step(optimizer)
            scaler.update()
            losses.append(batch_loss)
        validation = _evaluate(model, val_dataset, labels, device, eval_batch_size)
        margins = np.concatenate(epoch_margin) if epoch_margin else np.empty(0, dtype=np.float32)
        epoch_result = {
            "epoch": epoch,
            "train_cross_entropy": float(np.mean(losses)),
            "train_target_minus_hardest_rival": {
                "mean": float(margins.mean()) if len(margins) else None,
                "positive_fraction": float((margins > 0).mean()) if len(margins) else None,
            },
            "train_hardest_rivals_top10": [{"label": labels[index], "count": count} for index, count in rival_counts.most_common(10)],
            "validation": validation,
            "seconds": time.perf_counter() - epoch_started,
        }
        history.append(epoch_result)
        print(json.dumps({"event": "architecture_probe_epoch", "arm": name, **epoch_result}, ensure_ascii=False), flush=True)

    checkpoint_path = checkpoint_dir / f"{name}_seed{seed}.pt"
    if checkpoint_path.exists():
        raise FileExistsError(f"refusing to overwrite architecture checkpoint: {checkpoint_path}")
    torch.save({
        "schema": "aiflow-hwr-architecture-capacity-probe/v1",
        "arm": {"name": name, "width": width, "layers": layers, "heads": heads, "feedforward": width * 4, "parameters": params},
        "seed": seed,
        "state_dict": model.state_dict(),
        "initialization": "random; no existing model weights or teacher outputs",
    }, checkpoint_path)
    return {
        "name": name,
        "architecture": {"width": width, "layers": layers, "heads": heads, "feedforward": width * 4, "parameters": params},
        "optimizer": {"name": "AdamW", "learning_rate": learning_rate, "weight_decay": 1.0e-4, "fused": fused_adamw, "foreach": False, "gradient_clip_norm": 1.0, "gradient_clip_foreach": False, "effective_batch_size": batch_size, "microbatch_size": microbatch_size, "loss": "372-way cross-entropy", "teacher_distillation": False, "explicit_margin_loss": False},
        "seed": seed,
        "history": history,
        "final_validation": history[-1]["validation"],
        "checkpoint": str(checkpoint_path.resolve()),
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path, default=DEFAULT_CANONICAL_ROOT)
    parser.add_argument("--curated", type=Path, default=DEFAULT_CURATED)
    parser.add_argument("--writer-split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--reference-checkpoint", type=Path, default=DEFAULT_VOCAB_CHECKPOINT, help="used only by --self-test to verify default-width inference parity")
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--reuse-prepared-cache", action="store_true", help="reuse only after validating source hashes, writer split, row counts, and label range")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--microbatch-size", type=int, default=None, help="split each effective batch for activation-memory-bounded gradient accumulation")
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--arms", default="128:4:4,192:4:4", help="comma-separated width:layers:heads architecture arms")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test_gradient_accumulation()
        torch.manual_seed(23)
        reference = ScaledInkClassifier(372, 128, 4, 4, 512).eval()
        model, reference_labels, _ = _load_model(args.reference_checkpoint, torch.device("cpu"))
        labels = _class_labels(args.canonical_root.resolve())
        if len(reference_labels) != 372 or labels != reference_labels:
            raise AssertionError("vocabulary checkpoint must contain 372 classes")
        reference.load_state_dict(model.state_dict())
        values = torch.randn(2, POINTS, len(CHANNELS))
        with torch.inference_mode():
            if not torch.equal(reference(values), model(values, "math")):
                raise AssertionError("128x4 probe architecture is not checkpoint-compatible")
        print(json.dumps({"self_test": "pass", "default_arm_checkpoint_parity": True, "gradient_accumulation_matches_effective_batch": True}))
        return 0
    if args.epochs < 1 or args.batch_size < 1 or args.eval_batch_size < 1:
        parser.error("epochs and batch sizes must be positive")
    microbatch_size = args.microbatch_size or args.batch_size
    if microbatch_size < 1 or microbatch_size > args.batch_size:
        parser.error("microbatch size must be within [1, batch size]")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite architecture report: {args.output}")
    arms: list[tuple[int, int, int]] = []
    for spec in args.arms.split(","):
        parts = tuple(int(value.strip()) for value in spec.split(":"))
        if len(parts) != 3:
            parser.error(f"invalid architecture arm {spec!r}; expected width:layers:heads")
        width, layers, heads = parts
        if width % heads:
            parser.error(f"architecture arm {spec!r} width must be divisible by heads")
        arms.append(parts)
    if len(set(arms)) != len(arms):
        parser.error("duplicate architecture arms")

    labels = _class_labels(args.canonical_root.resolve())
    if len(labels) != 372 or len(set(labels)) != 372:
        raise ValueError(f"expected the frozen 372-class label order, got {len(labels)}")
    if args.reuse_prepared_cache:
        cache = _inspect_prepared_cache(
            args.canonical_root.resolve(), args.curated.resolve(), args.writer_split.resolve(),
            labels, args.cache_root.resolve(), SYNTHETIC_EQUAL_ROWS,
        )
    else:
        cache = _prepare_arrays(
            args.canonical_root.resolve(), args.curated.resolve(), args.writer_split.resolve(),
            labels, args.cache_root.resolve(), SYNTHETIC_EQUAL_ROWS,
        )
    checkpoint_dir = args.cache_root.resolve() / "checkpoints"
    if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
        raise FileExistsError(f"refusing to reuse non-empty checkpoint directory: {checkpoint_dir}")
    checkpoint_dir.mkdir(exist_ok=True)
    results = []
    for width, layers, heads in arms:
        name = f"d{width}_l{layers}_h{heads}"
        results.append(_train_arm(
            name, width, layers, heads, args.epochs, args.batch_size, microbatch_size,
            args.learning_rate, args.seed, labels, cache["cache"], device,
            checkpoint_dir, args.eval_batch_size,
        ))
        if device.type == "cuda":
            torch.cuda.empty_cache()
    report = {
        "schema": "aiflow-hwr-architecture-scale-writer-probe/v1",
        "status": "completed_exploratory_capacity_probe",
        "data": {
            "writer_split_manifest": str(args.writer_split.resolve()),
            "writer_split_sha256": _sha256(args.writer_split),
            "canonical_hwrt_sha256": _sha256(args.canonical_root / "hwrt.jsonl.gz"),
            "canonical_uji_sha256": _sha256(args.canonical_root / "uji.jsonl.gz"),
            "curated_uji_sha256": _sha256(args.curated),
            "cache": cache,
            "training_policy": "all HWRT official train rows + only non-validation UJI official train writers",
            "validation_policy": "selected 8 writer groups from UJI official train split; 20 official test writers untouched",
            "excluded": ["CROHME", "project-owned holdout", "existing checkpoint weights", "teacher logits", "official UJI test scores"],
        },
        "comparison": {
            "objective": "equal 372-way CE; no teacher distillation and no explicit margin term",
            "input_contract": "128x5 points, uniform-time channel; identical rows and sampler per arm",
            "product_adopted": False,
            "interpretation_limit": "UJI writer-generalization diagnostic for the 97-class overlap only; not full 372-class or formula-level acceptance",
        },
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "microbatch_size": microbatch_size,
        "seed": args.seed,
        "arms": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({
        "event": "architecture_scale_probe_complete",
        "output": str(args.output.resolve()),
        "arms": [{"name": row["name"], "parameters": row["architecture"]["parameters"], "val_top1": row["final_validation"]["top1"], "val_top5": row["final_validation"]["top5"], "macro_top1": row["final_validation"]["macro_top1_over_present_labels"]} for row in results],
        "crohme_rows": 0,
        "product_adopted": False,
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
