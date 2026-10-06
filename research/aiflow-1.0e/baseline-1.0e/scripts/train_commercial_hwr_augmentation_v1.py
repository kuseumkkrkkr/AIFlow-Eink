#!/usr/bin/env python3
"""Select and freeze a commercial-data-only online-HWR augmentation candidate.

The selection loop is project-writer-disjoint. HWRT/UJI/ISGL/UCI rows are
rehearsal data only, and CROHME/MathWriting are rejected before ML imports.
Augmentation preserves the 128 x 5 tensor contract and changes only x/y.
"""

from __future__ import annotations

from training_data_guard_v1 import assert_training_entrypoint_arguments_clean

assert_training_entrypoint_arguments_clean()

import argparse
import copy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from calibrate_project_punctuation_v1 import (
    _checkpoint_layout,
    _direct_rows,
    _load_base,
    _writer_split_indices,
)
from character_tensor_v1 import CHANNELS, POINTS, ROOT, _json_lines
from train_character_classifier_v1 import (
    INPUT_MODES,
    InkClassifierV1,
    _dataset,
    apply_input_mode,
    input_contract,
    load_vocabs,
    prepare_cache,
)
from training_data_guard_v1 import (
    assert_training_path_clean,
    zero_crohme_training_manifest,
)


SCHEMA = "aiflow-commercial-hwr-augmentation/v1"
SEED = 20260822
DEFAULT_CANONICAL = ROOT / "datasets" / "normalized" / "v1"
DEFAULT_CACHE = (
    ROOT / "artifacts" / "unified_head_20260813" / "unified_math_8ep_full" / "cache"
)
DEFAULT_BASE = (
    ROOT / "artifacts" / "time_normalization_20260813"
    / "uniform_time_unified_math_8ep_full" / "classifier_checkpoint.pt"
)
DEFAULT_DIRECT = Path(
    r"D:\AIFlow-Workspace\PrivateData\math-ink-data-collector\derived"
    r"\hwr-head-calibration-20260819-r1\project_owned_ownership_eval_95.jsonl.gz"
)
DEFAULT_LOO_HEADS = Path(
    r"D:\AIFlow-Workspace\PrivateData\math-ink-data-collector\derived"
    r"\hwr-head-calibration-20260819-r1\loo_steps250_lr1e-3\writer_loo_heads.pt"
)
DEFAULT_PRODUCT = Path(
    r"D:\AIFlow-Workspace\PrivateData\math-ink-data-collector\derived"
    r"\hwr-head-calibration-20260819-r1\final_all_writers_steps250_lr1e-3"
    r"\project_symbol_head_checkpoint.pt"
)
DEFAULT_OUTPUT = ROOT / "artifacts" / "commercial_hwr_augmentation_20260822_r1_shadow"

FAMILIES = {
    "vertical_slash": ("1", "|", "/"),
    "circle": ("0", "O", "o"),
    "cross": ("x", "\\times"),
}


@dataclass(frozen=True)
class Variant:
    name: str
    train_scope: str
    learning_rate: float
    rotation_degrees: float
    aspect_log_scale: float
    shear: float
    elastic: float
    top5_margin_weight: float

    @property
    def augmented(self) -> bool:
        return any((
            self.rotation_degrees,
            self.aspect_log_scale,
            self.shear,
            self.elastic,
        ))


VARIANTS = {
    row.name: row for row in (
        Variant("control_head", "head", 1.0e-3, 0.0, 0.0, 0.0, 0.0, 0.0),
        Variant("mild_head", "head", 2.0e-4, 7.0, 0.10, 0.07, 0.010, 0.0),
        Variant("mild_tail", "last_block", 3.0e-5, 8.0, 0.12, 0.08, 0.012, 0.0),
        Variant("mild_tail_top5", "last_block", 3.0e-5, 8.0, 0.12, 0.08, 0.012, 0.35),
        Variant("robust_tail", "last_block", 2.0e-5, 12.0, 0.16, 0.11, 0.018, 0.35),
    )
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _d_path(path: Path, kind: str, *, must_exist: bool = True) -> Path:
    resolved = assert_training_path_clean(path, kind)
    if resolved.drive.upper() != "D:":
        raise ValueError(f"{kind} must remain on D:: {resolved}")
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"missing {kind}: {resolved}")
    return resolved


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _clone_state(model: InkClassifierV1) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _new_model(
    base_state: dict[str, torch.Tensor], labels: list[str], device: torch.device,
    scope: str,
) -> InkClassifierV1:
    model = InkClassifierV1(len(labels), 0).to(device)
    model.load_state_dict(base_state, strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.math_head.requires_grad_(True)
    if scope == "last_block":
        model.encoder.layers[-1].requires_grad_(True)
        model.pool_score.requires_grad_(True)
    elif scope != "head":
        raise ValueError(f"unknown training scope: {scope}")
    model.eval()
    model.math_head.train()
    if scope == "last_block":
        model.encoder.layers[-1].train()
        model.pool_score.train()
    return model


def _letterbox(xy: torch.Tensor) -> torch.Tensor:
    low = xy.amin(dim=1, keepdim=True)
    high = xy.amax(dim=1, keepdim=True)
    center = (low + high) * 0.5
    extent = (high - low).amax(dim=2, keepdim=True).clamp_min(1.0e-5)
    return ((xy - center) / extent + 0.5).clamp(0.0, 1.0)


def augment_online_tensor(
    features: torch.Tensor, variant: Variant, generator: torch.Generator,
) -> torch.Tensor:
    """Apply label-preserving x/y variation; all other channels remain exact."""
    if not variant.augmented:
        return features
    result = features.clone()
    xy = result[:, :, :2]
    batch = len(xy)
    dtype, device = xy.dtype, xy.device
    center = (xy.amin(dim=1, keepdim=True) + xy.amax(dim=1, keepdim=True)) * 0.5
    relative = xy - center

    angle = (
        (torch.rand((batch, 1, 1), generator=generator, device=device, dtype=dtype) * 2.0 - 1.0)
        * math.radians(variant.rotation_degrees)
    )
    aspect = (
        (torch.rand((batch, 1, 1), generator=generator, device=device, dtype=dtype) * 2.0 - 1.0)
        * variant.aspect_log_scale
    )
    shear = (
        (torch.rand((batch, 1, 1), generator=generator, device=device, dtype=dtype) * 2.0 - 1.0)
        * variant.shear
    )
    cosine, sine = angle.cos(), angle.sin()
    x = relative[:, :, 0:1] * aspect.exp()
    y = relative[:, :, 1:2] * (-aspect).exp()
    rotated_x = cosine * x - sine * y + shear * y
    rotated_y = sine * x + cosine * y
    transformed = torch.cat((rotated_x, rotated_y), dim=2) + center

    if variant.elastic:
        noise = torch.randn(
            (batch, 2, POINTS), generator=generator, device=device, dtype=dtype,
        )
        smooth = F.avg_pool1d(F.pad(noise, (8, 8), mode="reflect"), 17, stride=1)
        smooth = smooth / smooth.square().mean(dim=2, keepdim=True).sqrt().clamp_min(1.0e-5)
        transformed = transformed + variant.elastic * smooth.transpose(1, 2)

    result[:, :, :2] = _letterbox(transformed)
    if not torch.equal(result[:, :, 2:], features[:, :, 2:]):
        raise AssertionError("augmentation changed time/stroke/observed channels")
    if not torch.isfinite(result).all() or result[:, :, :2].amin() < 0 or result[:, :, :2].amax() > 1:
        raise AssertionError("augmentation produced an invalid online tensor")
    return result


def _sample_probabilities(labels: np.ndarray, indices: np.ndarray, power: float) -> np.ndarray:
    selected = labels[indices]
    counts = np.bincount(selected, minlength=int(labels.max()) + 1).astype(np.float64)
    weights = np.power(counts[selected], -power)
    return weights / weights.sum()


def _top5_margin_loss(logits: torch.Tensor, target: torch.Tensor, margin: float = 0.15) -> torch.Tensor:
    truth = logits.gather(1, target[:, None]).squeeze(1)
    negatives = logits.clone()
    negatives.scatter_(1, target[:, None], float("-inf"))
    fifth_negative = negatives.topk(k=5, dim=1).values[:, -1]
    return F.relu(fifth_negative + margin - truth).mean()


def _batch(
    features: np.ndarray, indices: np.ndarray, input_mode: str, device: torch.device,
) -> torch.Tensor:
    values = np.asarray(features[indices], dtype=np.float32)
    values = apply_input_mode(values, input_mode)
    return torch.as_tensor(values, device=device)


def _train(
    initial_state: dict[str, torch.Tensor], labels: list[str], variant: Variant,
    direct_features: np.ndarray, direct_labels: np.ndarray, direct_indices: np.ndarray,
    external_features: np.ndarray, external_labels: np.ndarray, steps: int,
    direct_batch_size: int, external_batch_size: int, device: torch.device, seed: int,
) -> tuple[InkClassifierV1, dict]:
    model = _new_model(initial_state, labels, device, variant.train_scope)
    teacher = _new_model(initial_state, labels, device, "head")
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    teacher.eval()
    parameters = [value for value in model.parameters() if value.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=variant.learning_rate, weight_decay=1.0e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    rng = np.random.default_rng(seed)
    direct_probability = _sample_probabilities(direct_labels, direct_indices, 1.0)
    external_indices = np.arange(len(external_labels), dtype=np.int64)
    external_probability = _sample_probabilities(external_labels, external_indices, 0.5)
    direct_schedule = rng.choice(
        direct_indices, size=steps * direct_batch_size, replace=True, p=direct_probability,
    ).reshape(steps, direct_batch_size)
    external_schedule = rng.choice(
        external_indices, size=steps * external_batch_size, replace=True,
        p=external_probability,
    ).reshape(steps, external_batch_size)
    generator = torch.Generator(device=device).manual_seed(seed + 17)
    mutable = torch.zeros(len(labels), dtype=torch.bool, device=device)
    mutable[torch.as_tensor(np.unique(direct_labels[direct_indices]), device=device)] = True
    immutable = ~mutable
    initial_head_weight = initial_state["math_head.weight"].to(device)
    initial_head_bias = initial_state["math_head.bias"].to(device)
    losses: list[float] = []
    started = time.perf_counter()

    for step in range(steps):
        direct = _batch(direct_features, direct_schedule[step], "uniform-time", device)
        external = _batch(external_features, external_schedule[step], "uniform-time", device)
        direct_target = torch.as_tensor(direct_labels[direct_schedule[step]], device=device)
        external_target = torch.as_tensor(external_labels[external_schedule[step]], device=device)
        direct_augmented = augment_online_tensor(direct, variant, generator)
        external_augmented = augment_online_tensor(external, variant, generator)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            direct_raw_logits = model(direct, "math")
            external_raw_logits = model(external, "math")
            with torch.no_grad():
                teacher_external_logits = teacher(external, "math")
            loss = F.cross_entropy(direct_raw_logits, direct_target)
            loss = loss + F.cross_entropy(external_raw_logits, external_target)
            if variant.top5_margin_weight:
                loss = loss + variant.top5_margin_weight * _top5_margin_loss(
                    direct_raw_logits, direct_target,
                )
            loss = loss + 0.5 * F.kl_div(
                F.log_softmax(external_raw_logits, dim=1),
                F.softmax(teacher_external_logits, dim=1), reduction="batchmean",
            )
            if variant.augmented:
                direct_augmented_logits = model(direct_augmented, "math")
                external_augmented_logits = model(external_augmented, "math")
                loss = loss + F.cross_entropy(direct_augmented_logits, direct_target)
                loss = loss + F.cross_entropy(external_augmented_logits, external_target)
                if variant.top5_margin_weight:
                    loss = loss + variant.top5_margin_weight * _top5_margin_loss(
                        direct_augmented_logits, direct_target,
                    )
                loss = loss + 0.15 * F.kl_div(
                    F.log_softmax(external_augmented_logits, dim=1),
                    F.softmax(teacher_external_logits, dim=1), reduction="batchmean",
                )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step + 1}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        scaler.step(optimizer)
        scaler.update()
        if variant.train_scope == "head":
            with torch.no_grad():
                model.math_head.weight[immutable] = initial_head_weight[immutable]
                model.math_head.bias[immutable] = initial_head_bias[immutable]
        losses.append(float(loss.detach()))

    model.eval()
    return model, {
        "steps": steps,
        "direct_batch_size": direct_batch_size,
        "external_batch_size": external_batch_size,
        "learning_rate": variant.learning_rate,
        "train_scope": variant.train_scope,
        "trainable_parameters": sum(value.numel() for value in parameters),
        "mean_loss": float(np.mean(losses)),
        "final_loss": losses[-1],
        "seconds": time.perf_counter() - started,
    }


@torch.inference_mode()
def _logits(
    model: InkClassifierV1, features: np.ndarray, indices: np.ndarray,
    device: torch.device, batch_size: int,
) -> torch.Tensor:
    output: list[torch.Tensor] = []
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start:start + batch_size]
        output.append(model(_batch(features, batch_indices, "uniform-time", device), "math").cpu())
    return torch.cat(output)


def _ece(confidence: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    total = len(correct)
    value = 0.0
    for lower in np.linspace(0.0, 1.0, bins + 1)[:-1]:
        upper = lower + 1.0 / bins
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            value += float(mask.mean()) * abs(float(confidence[mask].mean()) - float(correct[mask].mean()))
    return value


def _metrics(rows: list[dict], label_names: list[str]) -> dict:
    if not rows:
        raise ValueError("cannot score an empty prediction set")
    truth = np.asarray([row["truth_index"] for row in rows], dtype=np.int64)
    top = np.asarray([row["top_indices"] for row in rows], dtype=np.int64)
    confidence = np.asarray([row["confidence"] for row in rows], dtype=np.float64)
    correct = top[:, 0] == truth
    top5 = np.any(top == truth[:, None], axis=1)
    writers = sorted({row["writer"] for row in rows})
    writer_scores = []
    for writer in writers:
        mask = np.asarray([row["writer"] == writer for row in rows])
        writer_scores.append({
            "writer": writer, "records": int(mask.sum()),
            "top1": float(correct[mask].mean()), "top5": float(top5[mask].mean()),
        })
    by_label = {}
    for index in sorted(set(truth.tolist())):
        mask = truth == index
        by_label[label_names[index]] = {
            "records": int(mask.sum()), "top1": float(correct[mask].mean()),
            "top5": float(top5[mask].mean()),
        }
    family = {}
    for name, values in FAMILIES.items():
        indices = {label_names.index(value) for value in values if value in label_names}
        mask = np.asarray([value in indices for value in truth])
        if mask.any():
            family[name] = {
                "records": int(mask.sum()), "top1": float(correct[mask].mean()),
                "top5": float(top5[mask].mean()),
            }
    return {
        "records": len(rows), "top1": float(correct.mean()), "top5": float(top5.mean()),
        "strict_macro_top1": float(np.mean([row["top1"] for row in by_label.values()])),
        "strict_macro_top5": float(np.mean([row["top5"] for row in by_label.values()])),
        "ece_10": _ece(confidence, correct),
        "writer_macro_top1": float(np.mean([row["top1"] for row in writer_scores])),
        "writer_macro_top5": float(np.mean([row["top5"] for row in writer_scores])),
        "worst_writer_top1": min(row["top1"] for row in writer_scores),
        "worst_writer_top5": min(row["top5"] for row in writer_scores),
        "by_writer": writer_scores, "by_label": by_label, "by_family": family,
    }


def _prediction_rows(
    logits: torch.Tensor, truth_indices: np.ndarray, writers: list[str],
) -> list[dict]:
    probabilities = F.softmax(logits, dim=1)
    confidence, top = probabilities.topk(k=5, dim=1)
    return [
        {
            "truth_index": int(truth), "top_indices": candidates.tolist(),
            "confidence": float(scores[0]), "writer": writer,
        }
        for truth, candidates, scores, writer in zip(
            truth_indices, top.numpy(), confidence.numpy(), writers, strict=True,
        )
    ]


def _safe_candidate(candidate: dict, control: dict) -> tuple[bool, list[str]]:
    reasons = []
    if candidate["top1"] < control["top1"] - 0.005:
        reasons.append("pooled_top1_regression")
    if candidate["top5"] < control["top5"]:
        reasons.append("pooled_top5_regression")
    if candidate["worst_writer_top5"] < control["worst_writer_top5"] - 0.03:
        reasons.append("worst_writer_top5_regression")
    if candidate["ece_10"] > control["ece_10"] + 0.03:
        reasons.append("calibration_regression")
    for family, baseline in control["by_family"].items():
        current = candidate["by_family"].get(family)
        if current and current["top5"] < baseline["top5"] - 0.01:
            reasons.append(f"{family}_top5_regression")
    return not reasons, reasons


def _select(results: dict[str, dict]) -> dict:
    control = results["control_head"]["writer_disjoint"]
    candidates = []
    gates = {}
    for name, row in results.items():
        if name == "control_head":
            continue
        passed, reasons = _safe_candidate(row["writer_disjoint"], control)
        gates[name] = {"passed": passed, "reasons": reasons}
        if passed:
            score = row["writer_disjoint"]
            candidates.append((
                score["top5"], score["top1"], score["strict_macro_top1"],
                -score["ece_10"], name,
            ))
    if not candidates:
        return {
            "selected": "control_head", "augmentation_adopted": False,
            "reason": "no augmented variant passed writer-disjoint safety gates",
            "gates": gates,
        }
    selected = max(candidates)[-1]
    return {
        "selected": selected, "augmentation_adopted": True,
        "reason": "best safe writer-disjoint Top-5, then Top-1/macro/calibration",
        "gates": gates,
    }


def _self_test() -> None:
    variant = VARIANTS["mild_head"]
    features = torch.zeros((3, POINTS, len(CHANNELS)), dtype=torch.float32)
    features[:, :, 0] = torch.linspace(0.1, 0.9, POINTS)
    features[:, :, 1] = torch.linspace(0.9, 0.1, POINTS)
    features[:, :, 2] = 1.0 / (POINTS - 1)
    features[:, 0, 2] = 0.0
    features[:, 0, 3] = 1.0
    features[:, :, 4] = 1.0
    generator = torch.Generator().manual_seed(7)
    augmented = augment_online_tensor(features, variant, generator)
    assert augmented.shape == features.shape
    assert torch.equal(augmented[:, :, 2:], features[:, :, 2:])
    assert not torch.equal(augmented[:, :, :2], features[:, :, :2])
    assert float(augmented[:, :, :2].min()) >= 0.0
    assert float(augmented[:, :, :2].max()) <= 1.0
    labels = np.asarray([0, 0, 1, 2], dtype=np.int64)
    probability = _sample_probabilities(labels, np.arange(4), 1.0)
    assert np.isclose(probability.sum(), 1.0) and probability[2] > probability[0]
    logits = torch.tensor([[3.0, 2.0, 1.0, 0.0, -1.0, -2.0]])
    assert _top5_margin_loss(logits, torch.tensor([5])).item() > 0
    assert _top5_margin_loss(logits, torch.tensor([0])).item() == 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path, default=DEFAULT_CANONICAL)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--direct-rows", type=Path, default=DEFAULT_DIRECT)
    parser.add_argument("--loo-heads", type=Path, default=DEFAULT_LOO_HEADS)
    parser.add_argument("--product-checkpoint", type=Path, default=DEFAULT_PRODUCT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--variants", default="control_head,mild_head,mild_tail,mild_tail_top5,robust_tail",
        help="comma-separated built-in variants; control_head is mandatory",
    )
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--direct-batch-size", type=int, default=32)
    parser.add_argument("--external-batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--search-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        print(json.dumps({"self_test": "pass"}))
        return 0
    if min(args.steps, args.direct_batch_size, args.external_batch_size, args.eval_batch_size) < 1:
        parser.error("steps and batch sizes must be positive")

    names = [value.strip() for value in args.variants.split(",") if value.strip()]
    if "control_head" not in names or len(names) != len(set(names)):
        parser.error("variants must be unique and include control_head")
    unknown = sorted(set(names) - set(VARIANTS))
    if unknown:
        parser.error(f"unknown variants: {unknown}")

    canonical = _d_path(args.canonical_root, "canonical root")
    cache_dir = _d_path(args.cache_dir, "training cache")
    base_path = _d_path(args.base_checkpoint, "external-only base checkpoint")
    direct_path = _d_path(args.direct_rows, "project-owned training derivative")
    loo_path = _d_path(args.loo_heads, "writer-LOO calibrated heads")
    product_path = _d_path(args.product_checkpoint, "current product HWR checkpoint")
    output = _d_path(args.output, "output", must_exist=False)
    if output.exists():
        parser.error(f"refusing to overwrite output: {output}")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")

    _seed_everything(SEED)
    labels, available_auxiliary = load_vocabs(canonical)
    auxiliary, head_mode = _checkpoint_layout(base_path, labels, available_auxiliary)
    if auxiliary or head_mode != "unified-math":
        raise ValueError("augmentation v1 requires the fixed unified 372-class base")
    base = _load_base(base_path, labels, auxiliary, device, "uniform-time")
    base_state = _clone_state(base)
    loo_payload = torch.load(loo_path, map_location="cpu", weights_only=False)
    if (
        loo_payload.get("schema") != "aiflow-writer-loo-project-symbol-heads/v1"
        or loo_payload.get("base_checkpoint_sha256") != _sha256(base_path)
        or loo_payload.get("math_labels") != labels
        or loo_payload.get("input_mode") != "uniform-time"
    ):
        raise ValueError("writer-LOO heads do not match the frozen external base")
    product_payload = torch.load(product_path, map_location="cpu", weights_only=False)
    if (
        product_payload.get("schema") != "aiflow-project-symbol-head-calibration/v2"
        or product_payload.get("math_labels") != labels
        or product_payload.get("auxiliary_labels") != []
        or product_payload.get("report", {}).get("base_checkpoint_sha256") != _sha256(base_path)
    ):
        raise ValueError("current product HWR checkpoint does not match the frozen base")
    product_state = {
        key: value.detach().cpu().clone()
        for key, value in product_payload["state_dict"].items()
    }
    cache = prepare_cache(canonical, cache_dir, labels, available_auxiliary, head_mode)
    math_train = _dataset(cache_dir, cache["sets"]["math_train"], "preserve")
    math_eval = _dataset(cache_dir, cache["sets"]["math_eval"], "preserve")
    direct_features, direct_target, writers, _truth = _direct_rows(
        canonical, {label: index for index, label in enumerate(labels)}, direct_path,
    )
    direct_labels = direct_target.numpy().astype(np.int64, copy=False)
    external_labels = np.asarray(math_train.labels, dtype=np.int64)
    folds = _writer_split_indices(writers)

    results: dict[str, dict] = {}
    states: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    started = time.perf_counter()
    for variant_index, name in enumerate(names):
        variant = VARIANTS[name]
        prediction_rows: list[dict] = []
        fold_rows = []
        states[name] = {}
        for fold_index, (writer, train_index, held_index) in enumerate(folds):
            # Keep samples and geometric draws paired across variants so the
            # comparison measures the configuration rather than seed noise.
            seed = SEED + fold_index * 37
            fold_state = {key: value.clone() for key, value in base_state.items()}
            calibrated_head = loo_payload["heads_by_held_writer_group"].get(writer)
            if calibrated_head is None:
                raise ValueError(f"missing calibrated head for held writer: {writer}")
            fold_state["math_head.weight"] = calibrated_head["weight"].clone()
            fold_state["math_head.bias"] = calibrated_head["bias"].clone()
            if name == "control_head":
                model = _new_model(fold_state, labels, device, "head")
                model.eval()
                training = {
                    "steps": 0, "seconds": 0.0, "train_scope": "frozen_control",
                    "initialization": "validated writer-LOO calibrated head",
                }
            else:
                model, training = _train(
                    fold_state, labels, variant, direct_features, direct_labels,
                    train_index.numpy(), math_train.features, external_labels, args.steps,
                    args.direct_batch_size, args.external_batch_size, device, seed,
                )
            held = held_index.numpy()
            logits = _logits(model, direct_features, held, device, args.eval_batch_size)
            prediction_rows.extend(_prediction_rows(
                logits, direct_labels[held], [writers[index] for index in held],
            ))
            fold_rows.append({
                "held_writer": writer, "held_records": len(held), "training": training,
            })
            states[name][writer] = _clone_state(model)
            print(json.dumps({
                "event": "augmentation_fold_complete", "variant": name,
                "fold": fold_index + 1, "folds": len(folds), "writer": writer,
                "held_records": len(held), "seconds": training["seconds"],
            }, ensure_ascii=False), flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        results[name] = {
            "variant": asdict(variant), "folds": fold_rows,
            "writer_disjoint": _metrics(prediction_rows, labels),
        }
        print(json.dumps({
            "event": "augmentation_variant_complete", "variant": name,
            "top1": results[name]["writer_disjoint"]["top1"],
            "top5": results[name]["writer_disjoint"]["top5"],
            "ece": results[name]["writer_disjoint"]["ece_10"],
        }, ensure_ascii=False), flush=True)

    selection = _select(results)
    selected_name = selection["selected"]
    selected_variant = VARIANTS[selected_name]
    final_state = None
    final_training = None
    external_gate = None
    checkpoint_path = None
    if not args.search_only:
        all_indices = np.arange(len(direct_labels), dtype=np.int64)
        if selected_name == "control_head":
            final_model = _new_model(product_state, labels, device, "head")
            final_model.eval()
            final_training = {
                "steps": 0, "seconds": 0.0, "train_scope": "frozen_control",
                "initialization": "current all-writer calibrated product HWR",
            }
        else:
            final_model, final_training = _train(
                product_state, labels, selected_variant, direct_features, direct_labels,
                all_indices, math_train.features, external_labels, args.steps,
                args.direct_batch_size, args.external_batch_size, device, SEED + 9000,
            )
        evaluation_indices = np.arange(len(math_eval.labels), dtype=np.int64)
        product_model = _new_model(product_state, labels, device, "head")
        product_model.eval()
        base_logits = _logits(product_model, math_eval.features, evaluation_indices, device, args.eval_batch_size)
        final_logits = _logits(final_model, math_eval.features, evaluation_indices, device, args.eval_batch_size)
        evaluation_writers = ["technical_external"] * len(evaluation_indices)
        base_external = _metrics(_prediction_rows(
            base_logits, np.asarray(math_eval.labels), evaluation_writers,
        ), labels)
        final_external = _metrics(_prediction_rows(
            final_logits, np.asarray(math_eval.labels), evaluation_writers,
        ), labels)
        external_gate = {
            "base": base_external, "candidate": final_external,
            "top1_regression": base_external["top1"] - final_external["top1"],
            "top5_regression": base_external["top5"] - final_external["top5"],
            "passed": (
                final_external["top1"] >= base_external["top1"] - 0.003
                and final_external["top5"] >= base_external["top5"] - 0.002
            ),
            "role": "fixed external technical non-regression only; not variant selection",
        }
        final_state = _clone_state(final_model)

    gradient_updates = (len(names) - 1) * len(folds) * args.steps
    if not args.search_only and selected_name != "control_head":
        gradient_updates += args.steps
    report = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "search_only" if args.search_only else "frozen_shadow_pending_untouched_acceptance",
        "architecture_unchanged": True,
        "architecture": {
            "input": [POINTS, len(CHANNELS)], "channels": list(CHANNELS),
            "classes": len(labels), "hidden": 128, "transformer_blocks": 4,
            "attention_heads": 4, "single_head": True,
        },
        "input_contract": input_contract("uniform-time"),
        "selection_policy": {
            "project_writer_disjoint": True, "writers": len(set(writers)),
            "records": len(direct_labels), "selection_used_fresh_acceptance": False,
            "selection_used_crohme": False, "selection_used_mathwriting": False,
            "external_rows_role": "class-balanced rehearsal only",
            "hwrt_writer_limit": "source user IDs are not treated as writer-disjoint evidence",
        },
        "augmentation_contract": {
            "space": "normalized online x/y only",
            "allowed": ["rotation", "aspect ratio", "shear", "smooth elastic displacement"],
            "forbidden": ["stroke deletion", "stroke merge", "stroke reorder", "label mutation", "rasterization"],
            "untouched_channels": ["delta_t", "stroke_start", "observed"],
            "crohme_error_labels_used_for_targeting": False,
        },
        "variants": results,
        "selection": selection,
        "final_training": final_training,
        "external_technical_nonregression": external_gate,
        "training_data_guard": zero_crohme_training_manifest(
            admitted_sources={
                **{key: int(value) for key, value in cache["sets"]["math_train"]["sources"].items()},
                "project_owned_legacy_writers": len(direct_labels),
            },
            gradient_updates=gradient_updates,
        ),
        "inputs": {
            "canonical_root": str(canonical),
            "cache_manifest": str((cache_dir / "cache_manifest.json").resolve()),
            "cache_manifest_sha256": _sha256(cache_dir / "cache_manifest.json"),
            "base_checkpoint": str(base_path), "base_checkpoint_sha256": _sha256(base_path),
            "project_owned_rows": str(direct_path), "project_owned_rows_sha256": _sha256(direct_path),
            "writer_loo_heads": str(loo_path), "writer_loo_heads_sha256": _sha256(loo_path),
            "current_product_checkpoint": str(product_path),
            "current_product_checkpoint_sha256": _sha256(product_path),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "frozen_candidate": not args.search_only,
        "product_adopted": False,
        "adoption_limit": (
            "fresh unseen-writer and one-time CROHME evaluation remain pending; "
            "grouping/context/layout are outside this trainer"
        ),
    }
    output.mkdir(parents=True)
    if not args.search_only:
        checkpoint_path = output / "commercial_hwr_augmented_checkpoint.pt"
        torch.save({
            "schema": SCHEMA, "state_dict": final_state, "math_labels": labels,
            "auxiliary_labels": [], "report": report,
        }, checkpoint_path)
        report["checkpoint"] = {
            "path": str(checkpoint_path.resolve()), "sha256": _sha256(checkpoint_path),
            "selected_variant": selected_name,
        }
        torch.save({
            "schema": "aiflow-commercial-hwr-augmentation-writer-loo/v1",
            "math_labels": labels, "selected_variant": selected_name,
            "states_by_held_writer": states[selected_name],
        }, output / "selected_writer_loo_models.pt")
    (output / "augmentation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n",
    )
    print(json.dumps({
        "event": "commercial_hwr_augmentation_complete", "output": str(output),
        "selected": selected_name, "augmentation_adopted": selection["augmentation_adopted"],
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
