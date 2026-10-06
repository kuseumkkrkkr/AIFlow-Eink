#!/usr/bin/env python3
"""Prepare, microscope-audit, and train an exploratory augmentation-distilled HWR student.

The scope is the complete frozen 372-class mathematical-symbol vocabulary,
including digits, Latin/Greek letters, operators, relations, fences, arrows,
large operators, and the remaining math symbols. Only approved HWRT/UJI train
splits are admitted. Missing '=' coverage is filled with explicitly synthetic
two-bar compositions from HWRT dash samples. CROHME is never used.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from character_tensor_v1 import CHANNELS, POINTS, ROOT, _json_lines, tensorize
from evaluate_48hz_prefix_v1 import _load_model
from train_character_classifier_v1 import apply_input_mode, input_contract
from hwr_boundary_distillation_v1 import boundary_training_loss, load_boundary_candidates


SCHEMA = "aiflow-hwr-affine-distillation-experiment/v1"
DEFAULT_CANONICAL_ROOT = Path(
    r"D:\AIFlow-Workspace\PrivateData\math-ink-data-collector\derived"
    r"\hwr-augmentation-20261001\canonical-trainpool"
)
DEFAULT_CHECKPOINT = Path(
    r"D:\AIFlow-Workspace\PrivateData\math-ink-data-collector\derived"
    r"\hwr-head-calibration-20260819-r1\final_all_writers_steps250_lr1e-3"
    r"\project_symbol_head_checkpoint.pt"
)
DEFAULT_WORK_DIR = Path(
    r"D:\AIFlow-Workspace\PrivateData\math-ink-data-collector\derived"
    r"\hwr-augmentation-20261001\affine-distill-v1"
)
DEFAULT_REPORT_DIR = ROOT / "artifacts" / "hwr_augmentation_microscope_20261001" / "affine_distill_v1"
SOURCE_IDS = {"hwrt": 0, "uji": 1, "synthetic_equal": 2}
RNG_SEED = 20261001
QA_SAMPLES_PER_CLASS = 8
SYNTHETIC_EQUAL_ROWS = 512
SAMPLES_PER_CLASS_PER_EPOCH = 128
DISTILL_TEMPERATURE = 2.0
XY_RMS_LIMIT = 0.035
XY_POINT_LIMIT = 0.105
PATH_RATIO_RANGE = (0.78, 1.22)
WIDE_PARAMETER_NAMES = ("rotation", "axis_scale", "shear", "elastic")
SYMBOL_FAMILY_NAMES = (
    "arrows", "digits", "fences", "greek", "large_ops",
    "latin_lower", "latin_upper", "operators", "other_math", "relations",
)


def _parse_wide_parameter_overrides(
    items: list[str],
    *,
    option_name: str,
    minimum: float,
    maximum: float,
) -> dict[str, float]:
    overrides: dict[str, float] = {}
    for item in items:
        name, separator, raw_value = item.partition("=")
        name = name.strip()
        if not separator or name not in WIDE_PARAMETER_NAMES:
            raise ValueError(
                f"{option_name} expects PARAMETER=VALUE where PARAMETER is one of "
                f"{', '.join(WIDE_PARAMETER_NAMES)}; got {item!r}"
            )
        if name in overrides:
            raise ValueError(f"{option_name} repeats parameter {name!r}")
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise ValueError(f"{option_name} has a non-numeric value in {item!r}") from exc
        if not math.isfinite(value) or not minimum <= value <= maximum:
            raise ValueError(f"{option_name} values must be in [{minimum}, {maximum}]")
        overrides[name] = value
    return overrides


def _resolve_wide_parameter_settings(
    default_probability: float,
    default_multiplier: float,
    probability_overrides: list[str],
    multiplier_overrides: list[str],
) -> tuple[dict[str, float], dict[str, float]]:
    probabilities = {name: default_probability for name in WIDE_PARAMETER_NAMES}
    probabilities.update(_parse_wide_parameter_overrides(
        probability_overrides,
        option_name="--wide-probability",
        minimum=0.0,
        maximum=1.0,
    ))
    multipliers = {name: default_multiplier for name in WIDE_PARAMETER_NAMES}
    multipliers.update(_parse_wide_parameter_overrides(
        multiplier_overrides,
        option_name="--wide-multiplier",
        minimum=1.0,
        maximum=3.0,
    ))
    return probabilities, multipliers


def _parse_family_displacement_overrides(items: list[str]) -> dict[str, float]:
    overrides: dict[str, float] = {}
    for item in items:
        family, separator, raw_value = item.partition("=")
        family = family.strip()
        if not separator or family not in SYMBOL_FAMILY_NAMES:
            raise ValueError(
                "--stroke-curve-family-displacement expects FAMILY=VALUE where FAMILY is one of "
                f"{', '.join(SYMBOL_FAMILY_NAMES)}; got {item!r}"
            )
        if family in overrides:
            raise ValueError(f"--stroke-curve-family-displacement repeats family {family!r}")
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise ValueError(f"--stroke-curve-family-displacement has a non-numeric value in {item!r}") from exc
        if not math.isfinite(value) or not 0.0 <= value <= 0.03:
            raise ValueError("--stroke-curve-family-displacement values must be finite and in [0, 0.03]")
        overrides[family] = value
    return overrides


def _parse_family_probability_overrides(items: list[str]) -> dict[str, float]:
    overrides: dict[str, float] = {}
    option_name = "--stroke-curve-family-probability"
    for item in items:
        family, separator, raw_value = item.partition("=")
        family = family.strip()
        if not separator or family not in SYMBOL_FAMILY_NAMES:
            raise ValueError(
                f"{option_name} expects FAMILY=VALUE where FAMILY is one of "
                f"{', '.join(SYMBOL_FAMILY_NAMES)}; got {item!r}"
            )
        if family in overrides:
            raise ValueError(f"{option_name} repeats family {family!r}")
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise ValueError(f"{option_name} has a non-numeric value in {item!r}") from exc
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{option_name} values must be finite and in [0, 1]")
        overrides[family] = value
    return overrides


def _parse_family_max_waves_overrides(items: list[str]) -> dict[str, int]:
    overrides: dict[str, int] = {}
    option_name = "--stroke-curve-family-max-waves"
    for item in items:
        family, separator, raw_value = item.partition("=")
        family = family.strip()
        if not separator or family not in SYMBOL_FAMILY_NAMES:
            raise ValueError(
                f"{option_name} expects FAMILY=VALUE where FAMILY is one of "
                f"{', '.join(SYMBOL_FAMILY_NAMES)}; got {item!r}"
            )
        if family in overrides:
            raise ValueError(f"{option_name} repeats family {family!r}")
        try:
            value = int(raw_value)
        except ValueError as exc:
            raise ValueError(f"{option_name} has a non-integer value in {item!r}") from exc
        if not 1 <= value <= 3:
            raise ValueError(f"{option_name} values must be in [1, 3]")
        overrides[family] = value
    return overrides


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_comparison_sources_eligible(
    policy: dict[str, Any],
    *,
    event: str,
    source_split_rows: dict[str, dict[str, int]] | None = None,
) -> None:
    """Fail closed when a comparison consumes restricted external test splits."""
    split_rows = source_split_rows or {}
    test_rows = {
        source: int(split_rows.get(source, {}).get("test", 0))
        for source in ("hwrt", "uji")
    }
    heldout = str(policy.get("heldout_comparison", "")).casefold()
    restricted = {
        "hwrt": test_rows["hwrt"] > 0 or "hwrt curated test split" in heldout,
        "uji": test_rows["uji"] > 0 or "uji pen v2" in heldout and "test split" in heldout,
    }
    if not any(restricted.values()):
        return
    result = {
        "schema": "aiflow-hwr-comparison-source-policy/v1",
        "status": "blocked_ineligible_external_test_sources",
        "event": event,
        "restricted_test_rows_by_source": test_rows,
        "restricted_source_flags": restricted,
        "crohme_rows": 0,
        "reason": (
            "HWRT splits are approved for training/integrity only, and UJI Pen v2 is approved only for box-local "
            "candidate pretraining. Neither external official test split may score or select this model."
        ),
        "next_step": "Supply a permitted, untouched writer/formula/device-disjoint evaluation source before comparison.",
    }
    print(json.dumps(result, ensure_ascii=False), flush=True)
    raise SystemExit(2)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite experiment report: {path}")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _family(label: str) -> str:
    relation_words = (
        "leq", "geq", "neq", "equiv", "approx", "subset", "supset",
        "notin", "dashv", "vdash", "models", "propto", "cong", "parallel",
        "asymp", "nleq", "ngeq", "nsim", "nsubset", "nsupset", "mid",
    )
    if label in {"=", "<", ">", "≤", "≥", "≠", "≈", "∼", "∝"} or any(key in label.lower() for key in relation_words):
        return "relations"
    if len(label) == 1 and label.isdigit():
        return "digits"
    if len(label) == 1 and label.islower():
        return "latin_lower"
    if len(label) == 1 and label.isupper():
        return "latin_upper"
    if any(key in label.lower() for key in (
        "alpha", "beta", "gamma", "delta", "epsilon", "theta", "lambda",
        "mu", "pi", "rho", "sigma", "tau", "phi", "chi", "psi", "omega",
        "kappa", "zeta", "eta", "iota", "nu", "xi", "upsilon",
    )):
        return "greek"
    if label in {"(", ")", "[", "]", "{", "}", r"\{", r"\}", "|", r"\|", r"\langle", r"\rangle", r"\lfloor", r"\rfloor", r"\lceil", r"\rceil"}:
        return "fences"
    if any(key in label.lower() for key in ("arrow", "mapsto", "longrightarrow", "rightarrow", "leftarrow", "hookrightarrow", "to")):
        return "arrows"
    if any(key in label.lower() for key in ("sqrt", "sum", "prod", "int", "oint", "bigcup", "bigcap", "coprod", "lim")):
        return "large_ops"
    if any(key in label.lower() for key in ("times", "div", "pm", "mp", "cdot", "oplus", "otimes", "cup", "cap", "land", "lor", "wedge", "vee", "circ", "ast", "star", "setminus", "infty")) or label in {"+", "-", "/", "*", "<", ">", "≤", "≥", "≠"}:
        return "operators"
    return "other_math"


def _load_teacher(checkpoint_path: Path, device: torch.device):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    model, labels, checkpoint = _load_model(checkpoint_path, device)
    if len(labels) != 372 or len(set(labels)) != 372:
        raise ValueError(f"expected the frozen 372-class AIFlow checkpoint, got {len(labels)} labels")
    return model, list(labels), checkpoint


def _tensor_checks(features: np.ndarray) -> None:
    if features.shape != (POINTS, len(CHANNELS)):
        raise ValueError(f"invalid tensor shape: {features.shape}")
    if not np.isfinite(features).all():
        raise ValueError("non-finite model input")
    if (features[:, :2] < 0).any() or (features[:, :2] > 1).any():
        raise ValueError("spatial coordinates outside [0,1]")
    if (features[:, 2] < 0).any():
        raise ValueError("negative delta_t")
    starts = np.flatnonzero(features[:, 3] > 0.5)
    if not len(starts) or starts[0] != 0 or len(starts) > 43:
        raise ValueError("invalid stroke-start channel or unsupported stroke count")
    if not np.isfinite(features).all():
        raise ValueError("non-finite tensor after input-mode conversion")


def _scan_source_files(root: Path, labels: list[str]) -> dict[str, Any]:
    label_set = set(labels)
    counts: dict[str, Counter] = defaultdict(Counter)
    label_counts: dict[str, Counter] = defaultdict(Counter)
    cross_split_fingerprints: dict[str, list[str]] = {}
    for source in ("hwrt", "uji"):
        fingerprint_splits: dict[str, str] = {}
        path = root / f"{source}.jsonl.gz"
        if not path.is_file():
            raise FileNotFoundError(path)
        for row in _json_lines(path):
            split = str(row.get("split", "unknown"))
            label = str(row.get("label", ""))
            counts[source][split] += 1
            if split in {"train", "test"} and label in label_set:
                label_counts[f"{source}:{split}"][label] += 1
            fingerprint = str(row.get("source_fingerprint") or row.get("record_id") or "")
            if fingerprint:
                previous = fingerprint_splits.setdefault(fingerprint, split)
                if previous != split:
                    cross_split_fingerprints.setdefault(source, []).append(fingerprint)
        if cross_split_fingerprints.get(source):
            raise ValueError(f"{source} has identical source fingerprints in multiple splits")
    if set(label_counts["hwrt:train"]) != set(labels) - {"(", ")", "="}:
        missing = sorted((set(labels) - {"(", ")", "="}) - set(label_counts["hwrt:train"]))
        extra = sorted(set(label_counts["hwrt:train"]) - set(labels))
        raise ValueError(f"HWRT train vocabulary mismatch; missing={missing[:10]} extra={extra[:10]}")
    if not {"(", ")"} <= set(label_counts["uji:train"]):
        raise ValueError("UJI official train split is missing one or both parentheses")
    return {
        "source_split_rows": {source: dict(sorted(row.items())) for source, row in counts.items()},
        "label_counts": {key: dict(sorted(value.items())) for key, value in label_counts.items()},
        "input_files": {
            source: {
                "path": str((root / f"{source}.jsonl.gz").resolve()),
                "sha256": _sha256(root / f"{source}.jsonl.gz"),
            }
            for source in ("hwrt", "uji")
        },
        "cross_split_source_fingerprint_overlap": cross_split_fingerprints,
    }


def _iter_matching(root: Path, split: str, source: str, label_set: set[str]):
    for row in _json_lines(root / f"{source}.jsonl.gz"):
        if row.get("split") == split and str(row.get("label")) in label_set:
            yield row


def _synthesize_equal(dash_features: np.ndarray, count: int, seed: int) -> np.ndarray:
    if len(dash_features) < 4:
        raise ValueError("need at least four real HWRT dash examples for synthetic '='")
    rng = np.random.default_rng(seed)
    choice_a = rng.integers(0, len(dash_features), size=count)
    choice_b = rng.integers(0, len(dash_features), size=count)
    indices = np.rint(np.linspace(0, POINTS - 1, 64)).astype(np.int64)
    first = dash_features[choice_a][:, indices, :]
    second = dash_features[choice_b][:, indices, :]
    output = np.zeros((count, POINTS, len(CHANNELS)), dtype=np.float32)
    for bar_index, source in enumerate((first, second)):
        xy = source[:, :, :2].copy()
        low = xy[:, :, 0].min(axis=1, keepdims=True)
        high = xy[:, :, 0].max(axis=1, keepdims=True)
        width = np.maximum(high - low, 1.0e-5)
        x = (xy[:, :, 0] - low) / width
        source_mean_y = xy[:, :, 1].mean(axis=1, keepdims=True)
        source_y_deviation = np.clip(xy[:, :, 1] - source_mean_y, -0.035, 0.035) * 0.55
        slope = rng.uniform(-0.018, 0.018, size=(count, 1)).astype(np.float32)
        jitter = rng.uniform(-0.018, 0.018, size=(count, 1)).astype(np.float32)
        center = (0.38 if bar_index == 0 else 0.62) + jitter
        y = center + source_y_deviation + slope * (x - 0.5)
        output[:, bar_index * 64:(bar_index + 1) * 64, 0] = x
        output[:, bar_index * 64:(bar_index + 1) * 64, 1] = y
    # Independently reverse bar directions and vary which bar is written first.
    for index in range(count):
        if rng.random() < 0.5:
            output[index, :64, :2] = output[index, :64, :2][::-1].copy()
        if rng.random() < 0.5:
            output[index, 64:, :2] = output[index, 64:, :2][::-1].copy()
        if rng.random() < 0.5:
            output[index, :, :2] = np.concatenate((output[index, 64:, :2].copy(), output[index, :64, :2].copy()), axis=0)
    output[:, 0, 3] = 1.0
    output[:, 64, 3] = 1.0
    output[:, :, 2] = 1.0 / (POINTS - 1)
    output[:, 0, 2] = 0.0
    output[:, :, 4] = 1.0
    output[:, :, :2] = np.clip(output[:, :, :2], 0.0, 1.0)
    return output


def _prepare_cache(root: Path, checkpoint_path: Path, work_dir: Path, labels: list[str]) -> dict[str, Any]:
    if work_dir.exists() and any(work_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty experiment directory: {work_dir}")
    work_dir.mkdir(parents=True, exist_ok=True)
    source_audit = _scan_source_files(root, labels)
    label_to_id = {label: index for index, label in enumerate(labels)}
    label_set = set(labels)
    real_counts = {
        split: sum(
            sum(source_audit["label_counts"].get(f"{source}:{split}", {}).values())
            for source in ("hwrt", "uji")
        )
        for split in ("train", "test")
    }
    dash_label = "-"
    train_label_counts = source_audit["label_counts"]["hwrt:train"]
    if train_label_counts.get(dash_label, 0) < 4:
        raise ValueError("the approved HWRT train split has too few '-' examples for synthetic '='")
    synthetic_count = SYNTHETIC_EQUAL_ROWS
    output_counts = {
        "train": real_counts["train"] + synthetic_count,
        "test": real_counts["test"],
    }
    written_by_split_source: dict[str, Counter] = defaultdict(Counter)
    label_counts: dict[str, Counter] = defaultdict(Counter)
    dash_features: list[np.ndarray] = []
    for split in ("train", "test"):
        features_path = work_dir / f"{split}_features.npy"
        labels_path = work_dir / f"{split}_labels.npy"
        sources_path = work_dir / f"{split}_sources.npy"
        if any(path.exists() for path in (features_path, labels_path, sources_path)):
            raise FileExistsError(f"cache file already exists for {split}")
        features = np.lib.format.open_memmap(
            features_path, mode="w+", dtype=np.float32,
            shape=(output_counts[split], POINTS, len(CHANNELS)),
        )
        target_ids = np.lib.format.open_memmap(labels_path, mode="w+", dtype=np.int16, shape=(output_counts[split],))
        source_ids = np.lib.format.open_memmap(sources_path, mode="w+", dtype=np.int8, shape=(output_counts[split],))
        offset = 0
        for source in ("hwrt", "uji"):
            for row in _iter_matching(root, split, source, label_set):
                values = apply_input_mode(tensorize(row), "uniform-time")
                _tensor_checks(values)
                features[offset] = values
                target_ids[offset] = label_to_id[str(row["label"])]
                source_ids[offset] = SOURCE_IDS[source]
                written_by_split_source[f"{split}:{source}"][str(row["label"])] += 1
                label_counts[split][str(row["label"])] += 1
                if split == "train" and source == "hwrt" and str(row["label"]) == dash_label and len(dash_features) < 4096:
                    dash_features.append(values.copy())
                offset += 1
        if offset != real_counts[split]:
            raise AssertionError(f"{split} tensorized row count mismatch: expected {real_counts[split]}, got {offset}")
        if split == "train":
            dash_array = np.stack(dash_features).astype(np.float32, copy=False)
            equal_rows = _synthesize_equal(dash_array, synthetic_count, RNG_SEED + 17)
            for values in equal_rows:
                _tensor_checks(values)
            features[offset:offset + synthetic_count] = equal_rows
            target_ids[offset:offset + synthetic_count] = label_to_id["="]
            source_ids[offset:offset + synthetic_count] = SOURCE_IDS["synthetic_equal"]
            label_counts[split]["="] += synthetic_count
            written_by_split_source["train:synthetic_equal"]["="] = synthetic_count
            offset += synthetic_count
        if offset != output_counts[split]:
            raise AssertionError(f"{split} output row count mismatch")
        features.flush()
        target_ids.flush()
        source_ids.flush()
        del features, target_ids, source_ids

    observed_real = set(label_counts["train"]) - {"="}
    missing_real = sorted(set(labels) - observed_real - {"="})
    if missing_real:
        raise ValueError(f"real approved train data is missing unexpected labels: {missing_real}")
    if label_counts["train"]["="] != synthetic_count:
        raise AssertionError("synthetic '=' count mismatch")
    if label_counts["test"].get("=", 0) != 0:
        raise AssertionError("the official external test split unexpectedly contains '='")
    class_rows = [
        {
            "label": label,
            "family": _family(label),
            "real_train": int(label_counts["train"].get(label, 0) - (synthetic_count if label == "=" else 0)),
            "synthetic_train": int(synthetic_count if label == "=" else 0),
            "test": int(label_counts["test"].get(label, 0)),
        }
        for label in labels
    ]
    model_sha = _sha256(checkpoint_path)
    manifest = {
        "schema": SCHEMA,
        "status": "prepared",
        "current_checkpoint": {"path": str(checkpoint_path.resolve()), "sha256": model_sha, "classes": len(labels)},
        "input_policy": {
            "admitted_train": "HWRT curated official train split + UJI Pen v2 official writer-disjoint train split",
            "heldout_comparison": "HWRT curated test split + UJI Pen v2 official writer-disjoint test split; scored once after training",
            "project_owned": "excluded from this student experiment and its comparison",
            "crohme": "zero rows; never trained or scored",
            "input_mode": "uniform-time",
        },
        "source_audit": source_audit,
        "real_train_rows": int(real_counts["train"]),
        "synthetic_rows": {"=": synthetic_count, "generator": "two independently resampled HWRT train dashes, separated and lightly sloped; no CROHME or project holdout input"},
        "heldout_rows": int(real_counts["test"]),
        "class_rows": class_rows,
        "class_coverage": {
            "total_model_classes": len(labels),
            "real_training_classes": len(observed_real),
            "synthetic_only_classes": ["="],
            "external_test_classes": len(label_counts["test"]),
            "classes_without_train_support": missing_real,
        },
        "cache": {
            split: {
                "features": str((work_dir / f"{split}_features.npy").resolve()),
                "labels": str((work_dir / f"{split}_labels.npy").resolve()),
                "sources": str((work_dir / f"{split}_sources.npy").resolve()),
                "rows": output_counts[split],
            }
            for split in ("train", "test")
        },
        "counts_by_split_source": {
            key: dict(sorted(counter.items())) for key, counter in sorted(written_by_split_source.items())
        },
        "product_adopted": False,
    }
    return manifest


def _augment_affine(
    features: torch.Tensor,
    generator: torch.Generator,
    *,
    max_degrees: float | torch.Tensor = 2.5,
    scale_jitter: float | torch.Tensor = 0.055,
    shear_jitter: float | torch.Tensor = 0.025,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if features.ndim != 3 or features.shape[1:] != (POINTS, len(CHANNELS)):
        raise ValueError(f"expected batch x {POINTS} x 5, got {tuple(features.shape)}")
    baseline = features
    xy = baseline[:, :, :2]
    low = xy.amin(dim=1, keepdim=True)
    high = xy.amax(dim=1, keepdim=True)
    centered = xy - (low + high) * 0.5
    batch = len(features)
    def per_row(value: float | torch.Tensor) -> torch.Tensor:
        result = torch.as_tensor(value, device=features.device, dtype=features.dtype)
        if result.ndim == 0:
            return result
        if result.shape != (batch,):
            raise ValueError(f"expected scalar or one affine bound per row, got {tuple(result.shape)}")
        return result

    radians = (torch.rand(batch, device=features.device, generator=generator) * 2.0 - 1.0) * (per_row(max_degrees) * math.pi / 180.0)
    scale_x = 1.0 + (torch.rand(batch, device=features.device, generator=generator) * 2.0 - 1.0) * per_row(scale_jitter)
    scale_y = 1.0 + (torch.rand(batch, device=features.device, generator=generator) * 2.0 - 1.0) * per_row(scale_jitter)
    shear = (torch.rand(batch, device=features.device, generator=generator) * 2.0 - 1.0) * per_row(shear_jitter)
    cosine, sine = torch.cos(radians), torch.sin(radians)
    a = cosine * scale_x
    b = cosine * shear - sine * scale_y
    c = sine * scale_x
    d = sine * shear + cosine * scale_y
    determinant = a * d - b * c
    if torch.any(determinant <= 0.0):
        raise AssertionError("augmentation affine matrix is not orientation-preserving")
    transformed = torch.stack((
        centered[:, :, 0] * a[:, None] + centered[:, :, 1] * b[:, None],
        centered[:, :, 0] * c[:, None] + centered[:, :, 1] * d[:, None],
    ), dim=-1)
    transformed_low = transformed.amin(dim=1, keepdim=True)
    transformed_high = transformed.amax(dim=1, keepdim=True)
    transformed_center = (transformed_low + transformed_high) * 0.5
    extent = (transformed_high - transformed_low).amax(dim=2, keepdim=True).clamp_min(1.0e-6)
    transformed = (transformed - transformed_center) / extent + 0.5
    displacement = transformed - xy
    rms = displacement.square().mean(dim=(1, 2)).sqrt()
    maximum = displacement.norm(dim=2).amax(dim=1)
    original_steps = (xy[:, 1:] - xy[:, :-1]).norm(dim=2)
    transformed_steps = (transformed[:, 1:] - transformed[:, :-1]).norm(dim=2)
    within_stroke = (baseline[:, 1:, 3] < 0.5).to(original_steps.dtype)
    original_path = (original_steps * within_stroke).sum(dim=1).clamp_min(1.0e-7)
    transformed_path = (transformed_steps * within_stroke).sum(dim=1)
    path_ratio = transformed_path / original_path
    valid = (
        torch.isfinite(transformed).all(dim=(1, 2))
        & (transformed >= 0.0).all(dim=(1, 2))
        & (transformed <= 1.0).all(dim=(1, 2))
        & (rms <= XY_RMS_LIMIT)
        & (maximum <= XY_POINT_LIMIT)
        & (path_ratio >= PATH_RATIO_RANGE[0])
        & (path_ratio <= PATH_RATIO_RANGE[1])
    )
    result = baseline.clone()
    result[:, :, :2] = torch.where(valid[:, None, None], transformed, xy)
    if not torch.equal(result[:, :, 2:], baseline[:, :, 2:]):
        raise AssertionError("affine augmentation changed time/stroke/observed channels")
    if not torch.isfinite(result).all():
        raise AssertionError("affine augmentation produced non-finite features")
    diagnostics = {
        "rows": batch,
        "accepted": int(valid.sum().item()),
        "reverted": int((~valid).sum().item()),
        "changed": int((result[:, :, :2] != xy).any(dim=(1, 2)).sum().item()),
        "rms_mean": float(rms.mean().item()),
        "rms_p95": float(torch.quantile(rms, 0.95).item()),
        "max_point_p95": float(torch.quantile(maximum, 0.95).item()),
        "path_ratio_min": float(path_ratio.min().item()),
        "path_ratio_max": float(path_ratio.max().item()),
        "stroke_count_preserved": True,
        "orientation_preserving_affine": True,
    }
    return result, diagnostics


def _augment_elastic(
    features: torch.Tensor,
    generator: torch.Generator,
    *,
    grid_size: int = 5,
    max_control_displacement: float | torch.Tensor = 0.012,
    max_jacobian_frobenius: float = 0.35,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Apply a small smooth, boundary-pinned 2-D displacement field."""
    if features.ndim != 3 or features.shape[1:] != (POINTS, len(CHANNELS)):
        raise ValueError(f"expected batch x {POINTS} x 5, got {tuple(features.shape)}")
    if grid_size < 3:
        raise ValueError("elastic grid must be at least 3")
    baseline = features
    xy = baseline[:, :, :2]
    batch = len(features)
    amplitude = torch.as_tensor(max_control_displacement, device=features.device, dtype=features.dtype)
    if amplitude.ndim == 0:
        amplitude_scale = amplitude
    elif amplitude.shape == (batch,):
        amplitude_scale = amplitude[:, None, None, None]
    else:
        raise ValueError(f"expected scalar or one elastic bound per row, got {tuple(amplitude.shape)}")
    if torch.any(amplitude <= 0.0):
        raise ValueError("elastic displacement limits must be positive")
    field = (
        torch.rand((batch, grid_size, grid_size, 2), device=features.device, generator=generator)
        * 2.0 - 1.0
    ) * amplitude_scale
    # A pinned boundary keeps glyphs touching the normalized bbox from leaking out.
    field[:, 0, :, :] = 0.0
    field[:, -1, :, :] = 0.0
    field[:, :, 0, :] = 0.0
    field[:, :, -1, :] = 0.0

    spacing_scale = float(grid_size - 1)
    grad_x = torch.diff(field, dim=2) * spacing_scale
    grad_y = torch.diff(field, dim=1) * spacing_scale
    # A Frobenius bound below one makes I + grad(displacement) orientation-preserving.
    jacobian_bound = torch.sqrt(
        grad_x.square().amax(dim=(1, 2, 3))
        + grad_y.square().amax(dim=(1, 2, 3))
    )

    gx = xy[:, :, 0].clamp(0.0, 1.0) * spacing_scale
    gy = xy[:, :, 1].clamp(0.0, 1.0) * spacing_scale
    ix = gx.floor().long().clamp_(0, grid_size - 2)
    iy = gy.floor().long().clamp_(0, grid_size - 2)
    fx = (gx - ix.to(gx.dtype)).unsqueeze(-1)
    fy = (gy - iy.to(gy.dtype)).unsqueeze(-1)
    flat = field.reshape(batch, grid_size * grid_size, 2)

    def corner(offset_x: int, offset_y: int) -> torch.Tensor:
        indices = (iy + offset_y) * grid_size + (ix + offset_x)
        return torch.gather(flat, 1, indices.unsqueeze(-1).expand(-1, -1, 2))

    top_left, top_right = corner(0, 0), corner(1, 0)
    bottom_left, bottom_right = corner(0, 1), corner(1, 1)
    top = top_left * (1.0 - fx) + top_right * fx
    bottom = bottom_left * (1.0 - fx) + bottom_right * fx
    displacement = top * (1.0 - fy) + bottom * fy
    transformed = xy + displacement
    rms = (transformed - xy).square().mean(dim=(1, 2)).sqrt()
    maximum = (transformed - xy).norm(dim=2).amax(dim=1)
    original_steps = (xy[:, 1:] - xy[:, :-1]).norm(dim=2)
    transformed_steps = (transformed[:, 1:] - transformed[:, :-1]).norm(dim=2)
    within_stroke = (baseline[:, 1:, 3] < 0.5).to(original_steps.dtype)
    original_path = (original_steps * within_stroke).sum(dim=1).clamp_min(1.0e-7)
    path_ratio = (transformed_steps * within_stroke).sum(dim=1) / original_path
    valid = (
        torch.isfinite(transformed).all(dim=(1, 2))
        & (transformed >= 0.0).all(dim=(1, 2))
        & (transformed <= 1.0).all(dim=(1, 2))
        & (jacobian_bound <= max_jacobian_frobenius)
        & (rms <= XY_RMS_LIMIT)
        & (maximum <= XY_POINT_LIMIT)
        & (path_ratio >= PATH_RATIO_RANGE[0])
        & (path_ratio <= PATH_RATIO_RANGE[1])
    )
    result = baseline.clone()
    result[:, :, :2] = torch.where(valid[:, None, None], transformed, xy)
    if not torch.equal(result[:, :, 2:], baseline[:, :, 2:]):
        raise AssertionError("elastic augmentation changed time/stroke/observed channels")
    return result, {
        "rows": batch,
        "accepted": int(valid.sum().item()),
        "reverted": int((~valid).sum().item()),
        "changed": int((result[:, :, :2] != xy).any(dim=(1, 2)).sum().item()),
        "rms_mean": float(rms.mean().item()),
        "rms_p95": float(torch.quantile(rms, 0.95).item()),
        "max_point_p95": float(maximum.quantile(0.95).item()),
        "jacobian_frobenius_max": float(jacobian_bound.max().item()),
        "path_ratio_min": float(path_ratio.min().item()),
        "path_ratio_max": float(path_ratio.max().item()),
        "stroke_count_preserved": True,
        "orientation_preserving_bound": max_jacobian_frobenius,
        "boundary_pinned": True,
    }


def _augment_stroke_local(
    features: torch.Tensor,
    generator: torch.Generator,
    *,
    probability: float,
    rotation_degrees: float,
    scale_jitter: float,
    translation_jitter: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Apply tiny independent, orientation-preserving affine changes per stroke."""
    if features.ndim != 3 or features.shape[1:] != (POINTS, len(CHANNELS)):
        raise ValueError(f"expected batch x {POINTS} x 5, got {tuple(features.shape)}")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("stroke-local probability must be in [0, 1]")
    if not 0.0 <= rotation_degrees <= 5.0:
        raise ValueError("stroke-local rotation must be in [0, 5] degrees")
    if not 0.0 <= scale_jitter <= 0.15 or not 0.0 <= translation_jitter <= 0.05:
        raise ValueError("stroke-local scale/translation jitter is outside safe bounds")
    if probability == 0.0:
        return features, {
            "rows": len(features), "selected_rows": 0, "accepted_rows": 0,
            "reverted_rows": 0, "changed_rows": 0,
        }

    batch = len(features)
    xy = features[:, :, :2]
    starts = (features[:, :, 3] > 0.5).long()
    if not torch.all(starts[:, 0] == 1) or torch.any(starts.sum(dim=1) < 1):
        raise ValueError("stroke-local transform requires a stroke start at point zero")
    stroke_ids = starts.cumsum(dim=1) - 1
    row_offsets = torch.arange(batch, device=features.device).unsqueeze(1) * POINTS
    flat_ids = (stroke_ids + row_offsets).reshape(-1)
    flat_xy = xy.reshape(-1, 2)
    flat_centers = torch.zeros((batch * POINTS, 2), dtype=xy.dtype, device=xy.device)
    flat_centers.scatter_add_(0, flat_ids[:, None].expand(-1, 2), flat_xy)
    flat_counts = torch.zeros((batch * POINTS,), dtype=xy.dtype, device=xy.device)
    flat_counts.scatter_add_(0, flat_ids, torch.ones_like(flat_ids, dtype=xy.dtype))
    flat_centers /= flat_counts.clamp_min(1.0)[:, None]
    centers = flat_centers[flat_ids].reshape(batch, POINTS, 2)

    random_shape = (batch, POINTS)
    stroke_angles = (torch.rand(random_shape, device=features.device, generator=generator) * 2.0 - 1.0)
    stroke_angles *= rotation_degrees * math.pi / 180.0
    stroke_scale_x = 1.0 + (torch.rand(random_shape, device=features.device, generator=generator) * 2.0 - 1.0) * scale_jitter
    stroke_scale_y = 1.0 + (torch.rand(random_shape, device=features.device, generator=generator) * 2.0 - 1.0) * scale_jitter
    stroke_translate = (torch.rand((batch, POINTS, 2), device=features.device, generator=generator) * 2.0 - 1.0) * translation_jitter
    angles = stroke_angles.gather(1, stroke_ids)
    scale_x = stroke_scale_x.gather(1, stroke_ids)
    scale_y = stroke_scale_y.gather(1, stroke_ids)
    translate = stroke_translate.gather(1, stroke_ids[:, :, None].expand(-1, -1, 2))
    cos_a, sin_a = angles.cos(), angles.sin()
    centered = xy - centers
    local = torch.stack((
        centers[:, :, 0] + scale_x * (cos_a * centered[:, :, 0] - sin_a * centered[:, :, 1]) + translate[:, :, 0],
        centers[:, :, 1] + scale_y * (sin_a * centered[:, :, 0] + cos_a * centered[:, :, 1]) + translate[:, :, 1],
    ), dim=-1)
    local_low = local.amin(dim=1, keepdim=True)
    local_high = local.amax(dim=1, keepdim=True)
    local_center = (local_low + local_high) * 0.5
    local_extent = (local_high - local_low).amax(dim=2, keepdim=True).clamp_min(1.0e-6)
    local = ((local - local_center) / local_extent + 0.5).clamp(0.0, 1.0)
    selected = torch.rand((batch,), device=features.device, generator=generator) < probability
    candidate_xy = torch.where(selected[:, None, None], local, xy)
    displacement = candidate_xy - xy
    rms = displacement.square().mean(dim=(1, 2)).sqrt()
    maximum = displacement.norm(dim=2).amax(dim=1)
    original_steps = (xy[:, 1:] - xy[:, :-1]).norm(dim=2)
    candidate_steps = (candidate_xy[:, 1:] - candidate_xy[:, :-1]).norm(dim=2)
    within_stroke = (features[:, 1:, 3] < 0.5).to(original_steps.dtype)
    original_path = (original_steps * within_stroke).sum(dim=1).clamp_min(1.0e-7)
    path_ratio = (candidate_steps * within_stroke).sum(dim=1) / original_path
    finite = torch.isfinite(candidate_xy).all(dim=(1, 2))
    in_bounds = (candidate_xy >= 0.0).all(dim=(1, 2)) & (candidate_xy <= 1.0).all(dim=(1, 2))
    rms_ok = rms <= XY_RMS_LIMIT
    point_ok = maximum <= XY_POINT_LIMIT
    path_low_ok = path_ratio >= PATH_RATIO_RANGE[0]
    path_high_ok = path_ratio <= PATH_RATIO_RANGE[1]
    valid = finite & in_bounds & rms_ok & point_ok & path_low_ok & path_high_ok
    accepted = selected & valid
    result = features.clone()
    result[:, :, :2] = torch.where(accepted[:, None, None], candidate_xy, xy)
    if not torch.equal(result[:, :, 2:], features[:, :, 2:]):
        raise AssertionError("stroke-local augmentation changed time/stroke/observed channels")
    changed = (result[:, :, :2] != xy).any(dim=(1, 2))
    selected_count = int(selected.sum().item())
    return result, {
        "rows": batch,
        "selected_rows": selected_count,
        "accepted_rows": int(accepted.sum().item()),
        "reverted_rows": int((selected & ~valid).sum().item()),
        "changed_rows": int(changed.sum().item()),
        "rms_mean_selected": float(rms[selected].mean().item()) if selected_count else 0.0,
        "rms_p95_selected": float(rms[selected].quantile(0.95).item()) if selected_count else 0.0,
        "orientation_preserving": bool(torch.all((scale_x > 0.0) & (scale_y > 0.0)).item()),
        "rejection_reasons_on_selected_rows": {
            "non_finite": int((selected & ~finite).sum().item()),
            "out_of_bounds": int((selected & ~in_bounds).sum().item()),
            "rms_limit": int((selected & ~rms_ok).sum().item()),
            "point_limit": int((selected & ~point_ok).sum().item()),
            "path_ratio_low": int((selected & ~path_low_ok).sum().item()),
            "path_ratio_high": int((selected & ~path_high_ok).sum().item()),
        },
    }


def _resolve_stroke_curve_probability_rows(
    probability: float | torch.Tensor,
    *,
    batch: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if torch.is_tensor(probability):
        if probability.ndim != 1 or len(probability) != batch:
            raise ValueError(f"per-row stroke-curve probability must have shape ({batch},)")
        values = probability.detach().to(device=device, dtype=dtype)
    else:
        try:
            value = float(probability)
        except (TypeError, ValueError) as exc:
            raise ValueError("stroke-curve probability must be a scalar or per-row tensor") from exc
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("stroke-curve probability must be finite and in [0, 1]")
        values = torch.full((batch,), value, device=device, dtype=dtype)
    if not torch.isfinite(values).all().item() or torch.any(values < 0.0).item() or torch.any(values > 1.0).item():
        raise ValueError("stroke-curve probability values must be finite and in [0, 1]")
    return values


def _resolve_stroke_curve_max_waves_rows(
    max_waves: int | torch.Tensor,
    *,
    batch: int,
    device: torch.device,
) -> torch.Tensor:
    if torch.is_tensor(max_waves):
        if max_waves.ndim != 1 or len(max_waves) != batch:
            raise ValueError(f"per-row stroke-curve max waves must have shape ({batch},)")
        if torch.is_complex(max_waves):
            raise ValueError("stroke-curve max waves must be integer values in [1, 3]")
        values = max_waves.detach().to(device=device, dtype=torch.float32)
        if (
            not torch.isfinite(values).all().item()
            or torch.any(values < 1.0).item()
            or torch.any(values > 3.0).item()
            or torch.any(values != values.round()).item()
        ):
            raise ValueError("stroke-curve max waves values must be integers in [1, 3]")
        return values.to(dtype=torch.long)
    if isinstance(max_waves, bool):
        raise ValueError("stroke-curve max waves must be an integer in [1, 3]")
    try:
        value = int(max_waves)
    except (TypeError, ValueError) as exc:
        raise ValueError("stroke-curve max waves must be an integer in [1, 3]") from exc
    if value != max_waves or not 1 <= value <= 3:
        raise ValueError("stroke-curve max waves must be an integer in [1, 3]")
    return torch.full((batch,), value, dtype=torch.long, device=device)


def _resolve_stroke_curve_displacement_rows(
    max_displacement: float | torch.Tensor,
    *,
    batch: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if torch.is_tensor(max_displacement):
        if max_displacement.ndim != 1 or len(max_displacement) != batch:
            raise ValueError(f"per-row stroke-curve displacement must have shape ({batch},)")
        values = max_displacement.detach().to(device=device, dtype=dtype)
    else:
        try:
            value = float(max_displacement)
        except (TypeError, ValueError) as exc:
            raise ValueError("stroke-curve max displacement must be a scalar or per-row tensor") from exc
        if not math.isfinite(value) or not 0.0 <= value <= 0.03:
            raise ValueError("stroke-curve max displacement must be finite and in [0, 0.03]")
        values = torch.full((batch,), value, device=device, dtype=dtype)
    if not torch.isfinite(values).all().item() or torch.any(values < 0.0).item() or torch.any(values > 0.03).item():
        raise ValueError("stroke-curve max displacement values must be finite and in [0, 0.03]")
    return values


def _augment_stroke_curve(
    features: torch.Tensor,
    generator: torch.Generator,
    *,
    probability: float | torch.Tensor,
    max_displacement: float | torch.Tensor,
    max_waves: int | torch.Tensor,
    family_by_row: list[str] | None = None,
    selection_generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Bend each selected stroke with a small, smooth normal displacement."""
    if features.ndim != 3 or features.shape[1:] != (POINTS, len(CHANNELS)):
        raise ValueError(f"expected batch x {POINTS} x 5, got {tuple(features.shape)}")
    batch = len(features)
    max_waves_by_row = _resolve_stroke_curve_max_waves_rows(
        max_waves,
        batch=batch,
        device=features.device,
    )
    probability_by_row = _resolve_stroke_curve_probability_rows(
        probability,
        batch=batch,
        device=features.device,
        dtype=features.dtype,
    )
    displacement_by_row = _resolve_stroke_curve_displacement_rows(
        max_displacement,
        batch=batch,
        device=features.device,
        dtype=features.dtype,
    )
    if family_by_row is not None and (
        len(family_by_row) != batch or any(family not in SYMBOL_FAMILY_NAMES for family in family_by_row)
    ):
        raise ValueError("family_by_row must contain one supported symbol family per row")
    if not torch.any(probability_by_row > 0.0).item() or not torch.any(displacement_by_row > 0.0).item():
        return features, {
            "rows": batch, "selected_rows": 0, "accepted_rows": 0,
            "reverted_rows": 0, "changed_rows": 0, "changed_strokes": 0,
            "probability_mean_requested": float(probability_by_row.mean().item()) if batch else 0.0,
            "probability_p05_requested": float(probability_by_row.quantile(0.05).item()) if batch else 0.0,
            "probability_p95_requested": float(probability_by_row.quantile(0.95).item()) if batch else 0.0,
            "rows_by_family": {},
        }

    xy = features[:, :, :2]
    starts = features[:, :, 3] > 0.5
    if not torch.all(starts[:, 0]) or torch.any(starts.sum(dim=1) < 1):
        raise ValueError("stroke-curve transform requires a stroke start at point zero")
    stroke_ids = starts.long().cumsum(dim=1) - 1
    within_stroke = ~starts[:, 1:]
    segment = xy[:, 1:] - xy[:, :-1]
    segment_length = segment.norm(dim=2)
    distance_at_point = torch.cat((
        torch.zeros((batch, 1), dtype=xy.dtype, device=xy.device),
        segment_length * within_stroke.to(xy.dtype),
    ), dim=1).cumsum(dim=1)
    stroke_origin = torch.where(starts, distance_at_point, torch.full_like(distance_at_point, -torch.inf))
    stroke_origin = torch.cummax(stroke_origin, dim=1).values
    local_distance = (distance_at_point - stroke_origin).clamp_min(0.0)
    stroke_lengths = torch.zeros_like(distance_at_point)
    stroke_lengths.scatter_add_(1, stroke_ids, torch.cat((
        torch.zeros((batch, 1), dtype=xy.dtype, device=xy.device),
        segment_length * within_stroke.to(xy.dtype),
    ), dim=1))
    local_fraction = local_distance / stroke_lengths.gather(1, stroke_ids).clamp_min(1.0e-7)
    local_fraction = local_fraction.clamp(0.0, 1.0)

    previous = torch.cat((xy[:, :1], xy[:, :-1]), dim=1)
    following = torch.cat((xy[:, 1:], xy[:, -1:]), dim=1)
    has_previous = torch.cat((torch.zeros((batch, 1), dtype=torch.bool, device=xy.device), ~starts[:, 1:]), dim=1)
    has_following = torch.cat((~starts[:, 1:], torch.zeros((batch, 1), dtype=torch.bool, device=xy.device)), dim=1)
    tangent = (xy - previous) * has_previous.unsqueeze(-1) + (following - xy) * has_following.unsqueeze(-1)
    tangent = tangent / tangent.norm(dim=2, keepdim=True).clamp_min(1.0e-7)
    normal = torch.stack((-tangent[:, :, 1], tangent[:, :, 0]), dim=-1)

    random_shape = (batch, POINTS)
    amplitudes = (torch.rand(random_shape, device=xy.device, generator=generator) * 2.0 - 1.0) * displacement_by_row[:, None]
    phases = torch.rand(random_shape, device=xy.device, generator=generator) * (2.0 * math.pi)
    if torch.all(max_waves_by_row == max_waves_by_row[0]).item():
        wave_limit = int(max_waves_by_row[0].item())
        waves = torch.randint(1, wave_limit + 1, random_shape, device=xy.device, generator=generator)
    else:
        wave_unit = torch.rand(random_shape, device=xy.device, generator=generator)
        waves = (wave_unit * max_waves_by_row[:, None]).floor().long() + 1
    amplitude = amplitudes.gather(1, stroke_ids)
    phase = phases.gather(1, stroke_ids)
    frequency = waves.gather(1, stroke_ids).to(xy.dtype)
    selected = (
        torch.rand(
            (batch,), device=xy.device,
            generator=selection_generator if selection_generator is not None else generator,
        ) < probability_by_row
    ) & (displacement_by_row > 0.0)
    envelope = torch.sin(math.pi * local_fraction)
    displacement_scalar = amplitude * envelope * torch.sin(2.0 * math.pi * frequency * local_fraction + phase)
    stroke_ends = torch.cat((starts[:, 1:], torch.ones((batch, 1), dtype=torch.bool, device=xy.device)), dim=1)
    endpoints = starts | stroke_ends
    displacement_scalar = torch.where(endpoints, torch.zeros_like(displacement_scalar), displacement_scalar)
    edge_distance = torch.stack((xy[:, :, 0], 1.0 - xy[:, :, 0], xy[:, :, 1], 1.0 - xy[:, :, 1]), dim=-1).amin(dim=-1).clamp_min(0.0)
    boundary_scale = edge_distance / (edge_distance + displacement_by_row[:, None]).clamp_min(1.0e-7)
    displacement_scalar *= boundary_scale
    candidate_xy = xy + normal * displacement_scalar.unsqueeze(-1) * selected[:, None, None].to(xy.dtype)

    displacement = candidate_xy - xy
    rms = displacement.square().mean(dim=(1, 2)).sqrt()
    maximum = displacement.norm(dim=2).amax(dim=1)
    candidate_segment = candidate_xy[:, 1:] - candidate_xy[:, :-1]
    candidate_length = candidate_segment.norm(dim=2)
    active_segment = within_stroke & (segment_length > 1.0e-7)
    cosine = (segment * candidate_segment).sum(dim=2) / (segment_length * candidate_length).clamp_min(1.0e-7)
    direction_ok = ((cosine >= 0.0) | ~active_segment).all(dim=1)
    original_path = (segment_length * within_stroke.to(xy.dtype)).sum(dim=1).clamp_min(1.0e-7)
    candidate_path = (candidate_length * within_stroke.to(xy.dtype)).sum(dim=1)
    path_ratio = candidate_path / original_path
    finite = torch.isfinite(candidate_xy).all(dim=(1, 2))
    in_bounds = (candidate_xy >= 0.0).all(dim=(1, 2)) & (candidate_xy <= 1.0).all(dim=(1, 2))
    rms_ok = rms <= XY_RMS_LIMIT
    point_ok = maximum <= XY_POINT_LIMIT
    path_low_ok = path_ratio >= PATH_RATIO_RANGE[0]
    path_high_ok = path_ratio <= PATH_RATIO_RANGE[1]
    valid = finite & in_bounds & rms_ok & point_ok & path_low_ok & path_high_ok & direction_ok
    accepted = selected & valid
    result = features.clone()
    result[:, :, :2] = torch.where(accepted[:, None, None], candidate_xy, xy)
    if not torch.equal(result[:, :, 2:], features[:, :, 2:]):
        raise AssertionError("stroke-curve augmentation changed time/stroke/observed channels")
    point_changed = (result[:, :, :2] != xy).any(dim=2)
    changed_per_stroke = torch.zeros_like(stroke_lengths)
    changed_per_stroke.scatter_add_(1, stroke_ids, point_changed.to(xy.dtype))
    changed_strokes = int((changed_per_stroke > 0).sum().item())
    selected_count = int(selected.sum().item())
    active_cosine = cosine[active_segment & selected[:, None]]
    selected_stroke_starts = starts & selected[:, None]

    def summarize_wave_counts(stroke_mask: torch.Tensor) -> dict[str, Any]:
        histogram = {
            str(wave_count): int(((frequency == wave_count) & stroke_mask).sum().item())
            for wave_count in (1, 2, 3)
        }
        stroke_count = sum(histogram.values())
        weighted_total = sum(int(wave_count) * count for wave_count, count in ((1, histogram["1"]), (2, histogram["2"]), (3, histogram["3"])))
        return {
            "selected_strokes": stroke_count,
            "applied_wave_count_histogram": histogram,
            "mean_applied_wave_count": weighted_total / stroke_count if stroke_count else 0.0,
        }

    selected_wave_summary = summarize_wave_counts(selected_stroke_starts)
    rows_by_family: dict[str, dict[str, Any]] = {}
    if family_by_row is not None:
        family_ids = torch.tensor(
            [SYMBOL_FAMILY_NAMES.index(family) for family in family_by_row],
            dtype=torch.long,
            device=features.device,
        )
        for family_id, family in enumerate(SYMBOL_FAMILY_NAMES):
            rows = family_ids == family_id
            row_count = int(rows.sum().item())
            if not row_count:
                continue
            family_wave_summary = summarize_wave_counts(selected_stroke_starts & rows[:, None])
            rows_by_family[family] = {
                "rows": row_count,
                "probability_mean_requested": float(probability_by_row[rows].mean().item()),
                "max_waves_mean_requested": float(max_waves_by_row[rows].to(torch.float32).mean().item()),
                **family_wave_summary,
                "selected_rows": int((selected & rows).sum().item()),
                "accepted_rows": int((accepted & rows).sum().item()),
                "changed_rows": int((point_changed.any(dim=1) & rows).sum().item()),
                "reverted_rows": int((selected & ~valid & rows).sum().item()),
            }
    return result, {
        "rows": batch,
        "selected_rows": selected_count,
        "accepted_rows": int(accepted.sum().item()),
        "reverted_rows": int((selected & ~valid).sum().item()),
        "changed_rows": int(point_changed.any(dim=1).sum().item()),
        "changed_strokes": changed_strokes,
        "max_displacement_requested": float(displacement_by_row.max().item()),
        "max_displacement_mean_requested": float(displacement_by_row.mean().item()),
        "max_displacement_p05_requested": float(displacement_by_row.quantile(0.05).item()),
        "max_displacement_p95_requested": float(displacement_by_row.quantile(0.95).item()),
        "probability_mean_requested": float(probability_by_row.mean().item()),
        "probability_p05_requested": float(probability_by_row.quantile(0.05).item()),
        "probability_p95_requested": float(probability_by_row.quantile(0.95).item()),
        "rows_by_family": rows_by_family,
        "max_waves_requested": int(max_waves_by_row[0].item()) if torch.all(max_waves_by_row == max_waves_by_row[0]).item() else None,
        "max_waves_min_requested": int(max_waves_by_row.min().item()),
        "max_waves_mean_requested": float(max_waves_by_row.to(torch.float32).mean().item()),
        "max_waves_max_requested": int(max_waves_by_row.max().item()),
        **selected_wave_summary,
        "rms_mean_selected": float(rms[selected].mean().item()) if selected_count else 0.0,
        "rms_p95_selected": float(rms[selected].quantile(0.95).item()) if selected_count else 0.0,
        "direction_reversal_rows": int((selected & ~direction_ok).sum().item()),
        "direction_cosine_p05": float(active_cosine.quantile(0.05).item()) if active_cosine.numel() else 1.0,
        "boundary_attenuated_rows": int((selected & (boundary_scale < 0.99).any(dim=1)).sum().item()),
        "boundary_scale_p05_selected": float(boundary_scale[selected].quantile(0.05).item()) if selected_count else 1.0,
        "rejection_reasons_on_selected_rows": {
            "non_finite": int((selected & ~finite).sum().item()),
            "out_of_bounds": int((selected & ~in_bounds).sum().item()),
            "rms_limit": int((selected & ~rms_ok).sum().item()),
            "point_limit": int((selected & ~point_ok).sum().item()),
            "path_ratio_low": int((selected & ~path_low_ok).sum().item()),
            "path_ratio_high": int((selected & ~path_high_ok).sum().item()),
            "direction_reversal": int((selected & ~direction_ok).sum().item()),
        },
    }


def _augment_with_engine(
    features: torch.Tensor,
    generator: torch.Generator,
    engine: str,
    elastic_probability: float = 0.25,
    *,
    rotation_degrees: float = 2.5,
    axis_scale_jitter: float = 0.055,
    shear_jitter: float = 0.025,
    elastic_control_displacement: float = 0.012,
    wide_parameter_probability: float = 0.0,
    wide_magnitude_multiplier: float = 1.5,
    wide_probability_by_parameter: dict[str, float] | None = None,
    wide_multiplier_by_parameter: dict[str, float] | None = None,
    stroke_local_probability: float = 0.0,
    stroke_local_rotation_degrees: float = 1.0,
    stroke_local_scale_jitter: float = 0.02,
    stroke_local_translation_jitter: float = 0.003,
    stroke_generator: torch.Generator | None = None,
    stroke_curve_probability: float | torch.Tensor = 0.0,
    stroke_curve_max_displacement: float | torch.Tensor = 0.01,
    stroke_curve_max_waves: int | torch.Tensor = 2,
    stroke_curve_family_by_row: list[str] | None = None,
    stroke_curve_selection_generator: torch.Generator | None = None,
    curve_generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if engine not in {"affine", "affine_elastic", "affine_elastic_mix"}:
        raise ValueError(f"unknown augmentation engine: {engine}")
    if not 0.0 <= elastic_probability <= 1.0:
        raise ValueError("elastic_probability must be in [0, 1]")
    if not 0.0 <= wide_parameter_probability <= 1.0 or wide_magnitude_multiplier < 1.0:
        raise ValueError("wide-parameter probability must be in [0, 1] and multiplier must be >= 1")
    unknown_probability_names = set(wide_probability_by_parameter or {}) - set(WIDE_PARAMETER_NAMES)
    unknown_multiplier_names = set(wide_multiplier_by_parameter or {}) - set(WIDE_PARAMETER_NAMES)
    if unknown_probability_names or unknown_multiplier_names:
        raise ValueError("wide parameter overrides contain an unknown transform name")
    probability_settings = {name: wide_parameter_probability for name in WIDE_PARAMETER_NAMES}
    probability_settings.update(wide_probability_by_parameter or {})
    multiplier_settings = {name: wide_magnitude_multiplier for name in WIDE_PARAMETER_NAMES}
    multiplier_settings.update(wide_multiplier_by_parameter or {})
    if any(not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0 for value in probability_settings.values()):
        raise ValueError("wide probabilities by parameter must be finite and in [0, 1]")
    if any(not math.isfinite(float(value)) or not 1.0 <= float(value) <= 3.0 for value in multiplier_settings.values()):
        raise ValueError("wide multipliers by parameter must be finite and in [1, 3]")
    if not 0.0 <= stroke_local_probability <= 1.0:
        raise ValueError("stroke-local probability must be in [0, 1]")
    if not 0.0 <= stroke_local_rotation_degrees <= 5.0:
        raise ValueError("stroke-local rotation must be in [0, 5] degrees")
    if not 0.0 <= stroke_local_scale_jitter <= 0.15 or not 0.0 <= stroke_local_translation_jitter <= 0.05:
        raise ValueError("stroke-local scale/translation jitter is outside safe bounds")
    stroke_curve_probability_rows = _resolve_stroke_curve_probability_rows(
        stroke_curve_probability,
        batch=len(features),
        device=features.device,
        dtype=features.dtype,
    )
    stroke_curve_displacement_rows = _resolve_stroke_curve_displacement_rows(
        stroke_curve_max_displacement,
        batch=len(features),
        device=features.device,
        dtype=features.dtype,
    )
    stroke_curve_max_waves_rows = _resolve_stroke_curve_max_waves_rows(
        stroke_curve_max_waves,
        batch=len(features),
        device=features.device,
    )

    def finish(base: torch.Tensor, report: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        augmented = base
        if stroke_local_probability > 0.0:
            augmented, local_diag = _augment_stroke_local(
                base, stroke_generator if stroke_generator is not None else generator,
                probability=stroke_local_probability,
                rotation_degrees=stroke_local_rotation_degrees,
                scale_jitter=stroke_local_scale_jitter,
                translation_jitter=stroke_local_translation_jitter,
            )
            report["stroke_local"] = local_diag
        if torch.any(stroke_curve_probability_rows > 0.0).item():
            augmented, curve_diag = _augment_stroke_curve(
                augmented, curve_generator if curve_generator is not None else (
                    stroke_generator if stroke_generator is not None else generator
                ),
                probability=stroke_curve_probability_rows,
                max_displacement=stroke_curve_displacement_rows,
                max_waves=stroke_curve_max_waves_rows,
                family_by_row=stroke_curve_family_by_row,
                selection_generator=stroke_curve_selection_generator,
            )
            report["stroke_curve"] = curve_diag
        xy = augmented[:, :, :2]
        original_xy = features[:, :, :2]
        displacement = xy - original_xy
        rms = displacement.square().mean(dim=(1, 2)).sqrt()
        maximum = displacement.norm(dim=2).amax(dim=1)
        original_steps = (original_xy[:, 1:] - original_xy[:, :-1]).norm(dim=2)
        transformed_steps = (xy[:, 1:] - xy[:, :-1]).norm(dim=2)
        within_stroke = (features[:, 1:, 3] < 0.5).to(original_steps.dtype)
        original_path = (original_steps * within_stroke).sum(dim=1).clamp_min(1.0e-7)
        path_ratio = (transformed_steps * within_stroke).sum(dim=1) / original_path
        changed = (augmented[:, :, :2] != features[:, :, :2]).any(dim=(1, 2))
        report["changed"] = int(changed.sum().item())
        report["identity_output_rows"] = int((~changed).sum().item())
        report["reverted"] = int((~changed).sum().item())
        report["rms_mean"] = float(rms.mean().item())
        report["rms_p95"] = float(rms.quantile(0.95).item())
        report["max_point_p95"] = float(maximum.quantile(0.95).item())
        report["path_ratio_min"] = float(path_ratio.min().item())
        report["path_ratio_max"] = float(path_ratio.max().item())
        return augmented, report

    batch = len(features)
    parameter_names = WIDE_PARAMETER_NAMES
    active_parameter_count = 3 if engine == "affine" else 4
    wide_mask = torch.zeros((batch, len(parameter_names)), dtype=torch.bool, device=features.device)
    magnitude = torch.ones((batch, len(parameter_names)), dtype=features.dtype, device=features.device)
    probability_tensor = torch.tensor(
        [probability_settings[name] for name in parameter_names],
        dtype=features.dtype,
        device=features.device,
    )
    multiplier_tensor = torch.tensor(
        [multiplier_settings[name] for name in parameter_names],
        dtype=features.dtype,
        device=features.device,
    )
    if torch.any(probability_tensor[:active_parameter_count] > 0.0):
        wide_mask[:, :active_parameter_count] = (
            torch.rand((batch, active_parameter_count), device=features.device, generator=generator)
            < probability_tensor[:active_parameter_count]
        )
        magnitude = torch.where(wide_mask, multiplier_tensor.unsqueeze(0), 1.0)
    affine, affine_diag = _augment_affine(
        features,
        generator,
        max_degrees=rotation_degrees * magnitude[:, 0],
        scale_jitter=axis_scale_jitter * magnitude[:, 1],
        shear_jitter=shear_jitter * magnitude[:, 2],
    )
    severity_diag = {
        "independent_wide_probability_by_parameter": (
            next(iter(probability_settings.values()))
            if len(set(probability_settings.values())) == 1
            else None
        ),
        "wide_magnitude_multiplier": (
            next(iter(multiplier_settings.values()))
            if len(set(multiplier_settings.values())) == 1 and any(probability_settings.values())
            else None
        ),
        "resolved_wide_probability_by_parameter": probability_settings,
        "resolved_wide_multiplier_by_parameter": multiplier_settings,
        "wide_selected_rows_by_parameter": {
            name: int(wide_mask[:, index].sum().item()) for index, name in enumerate(parameter_names)
        },
        "wide_selected_any_parameter_rows": int(wide_mask.any(dim=1).sum().item()),
    }
    if engine == "affine":
        affine_diag["severity_mixture"] = severity_diag
        return finish(affine, affine_diag)
    elastic, elastic_diag = _augment_elastic(
        affine,
        generator,
        max_control_displacement=elastic_control_displacement * magnitude[:, 3],
    )
    if engine == "affine_elastic":
        selected = torch.ones(len(features), dtype=torch.bool, device=features.device)
    else:
        selected = torch.rand(len(features), device=features.device, generator=generator) < elastic_probability
    wide_mask[:, 3] &= selected
    severity_diag = {
        "independent_wide_probability_by_parameter": (
            next(iter(probability_settings.values()))
            if len(set(probability_settings.values())) == 1
            else None
        ),
        "wide_magnitude_multiplier": (
            next(iter(multiplier_settings.values()))
            if len(set(multiplier_settings.values())) == 1 and any(probability_settings.values())
            else None
        ),
        "resolved_wide_probability_by_parameter": probability_settings,
        "resolved_wide_multiplier_by_parameter": multiplier_settings,
        "wide_selected_rows_by_parameter": {
            name: int(wide_mask[:, index].sum().item()) for index, name in enumerate(parameter_names)
        },
        "wide_selected_any_parameter_rows": int(wide_mask.any(dim=1).sum().item()),
    }
    candidate = torch.where(selected[:, None, None], elastic, affine)
    xy = candidate[:, :, :2]
    original_xy = features[:, :, :2]
    displacement = xy - original_xy
    rms = displacement.square().mean(dim=(1, 2)).sqrt()
    maximum = displacement.norm(dim=2).amax(dim=1)
    original_steps = (original_xy[:, 1:] - original_xy[:, :-1]).norm(dim=2)
    transformed_steps = (xy[:, 1:] - xy[:, :-1]).norm(dim=2)
    within_stroke = (features[:, 1:, 3] < 0.5).to(original_steps.dtype)
    original_path = (original_steps * within_stroke).sum(dim=1).clamp_min(1.0e-7)
    path_ratio = (transformed_steps * within_stroke).sum(dim=1) / original_path
    valid = (
        torch.isfinite(candidate).all(dim=(1, 2))
        & (xy >= 0.0).all(dim=(1, 2))
        & (xy <= 1.0).all(dim=(1, 2))
        & (rms <= XY_RMS_LIMIT)
        & (maximum <= XY_POINT_LIMIT)
        & (path_ratio >= PATH_RATIO_RANGE[0])
        & (path_ratio <= PATH_RATIO_RANGE[1])
    )
    if not torch.equal(candidate[:, :, 2:], features[:, :, 2:]):
        raise AssertionError("composite augmentation changed time/stroke/observed channels")
    result = candidate.clone()
    # Invalid composites retain the already-validated affine view rather than
    # throwing away the entire augmentation opportunity.
    result[:, :, :2] = torch.where(valid[:, None, None], xy, affine[:, :, :2])
    changed = (result[:, :, :2] != original_xy).any(dim=(1, 2))
    report = {
        "rows": len(features),
        "changed": int(changed.sum().item()),
        "reverted": int((~changed).sum().item()),
        "elastic_probability_requested": 1.0 if engine == "affine_elastic" else elastic_probability,
        "elastic_selected_rows": int(selected.sum().item()),
        "elastic_selected_and_changed_rows": int((selected & (elastic[:, :, :2] != affine[:, :, :2]).any(dim=(1, 2))).sum().item()),
        "affine_gate_reverted_rows": int(affine_diag["reverted"]),
        "elastic_gate_reverted_rows": int(elastic_diag["reverted"]),
        "composite_gate_reverted_selected_rows": int((selected & ~valid).sum().item()),
        "identity_output_rows": int((~changed).sum().item()),
        "rms_mean": float(rms.mean().item()),
        "rms_p95": float(rms.quantile(0.95).item()),
        "max_point_p95": float(maximum.quantile(0.95).item()),
        "path_ratio_min": float(path_ratio.min().item()),
        "path_ratio_max": float(path_ratio.max().item()),
        "stroke_count_preserved": True,
        "affine_stage": affine_diag,
        "elastic_stage": elastic_diag,
        "composite_fell_back_to_affine": int((selected & ~valid).sum().item()),
        "severity_mixture": severity_diag,
    }
    return finish(result, report)


def _augmentation_parameters(args) -> dict[str, Any]:
    resolved_curve_probabilities = {
        family: args.stroke_curve_probability_by_family.get(
            family, args.stroke_curve_probability,
        )
        for family in SYMBOL_FAMILY_NAMES
    }
    resolved_curve_displacements = {
        family: args.stroke_curve_displacement_by_family.get(
            family, args.stroke_curve_max_displacement,
        )
        for family in SYMBOL_FAMILY_NAMES
    }
    resolved_curve_max_waves = {
        family: args.stroke_curve_max_waves_by_family.get(
            family, args.stroke_curve_max_waves,
        )
        for family in SYMBOL_FAMILY_NAMES
    }
    return {
        "rotation_degrees": [-args.rotation_degrees, args.rotation_degrees],
        "axis_scale_relative": [-args.axis_scale_jitter, args.axis_scale_jitter],
        "shear": [-args.shear_jitter, args.shear_jitter],
        "elastic_control_grid": [5, 5] if args.augmentation != "affine" else None,
        "elastic_max_control_displacement": args.elastic_control_displacement if args.augmentation != "affine" else None,
        "elastic_boundary": "pinned" if args.augmentation != "affine" else None,
        "elastic_jacobian_frobenius_max": 0.35 if args.augmentation != "affine" else None,
        "elastic_probability": args.elastic_probability if args.augmentation == "affine_elastic_mix" else None,
        "wide_parameter_probability_default": args.wide_parameter_probability,
        "wide_magnitude_multiplier_default": args.wide_magnitude_multiplier,
        "independent_wide_probability_by_parameter": (
            next(iter(args.wide_probability_by_parameter.values()))
            if len(set(args.wide_probability_by_parameter.values())) == 1
            else None
        ),
        "wide_magnitude_multiplier": (
            next(iter(args.wide_multiplier_by_parameter.values()))
            if len(set(args.wide_multiplier_by_parameter.values())) == 1
            and any(args.wide_probability_by_parameter.values())
            else None
        ),
        "resolved_wide_probability_by_parameter": args.wide_probability_by_parameter,
        "resolved_wide_multiplier_by_parameter": args.wide_multiplier_by_parameter,
        "stroke_local_probability": args.stroke_local_probability,
        "stroke_local_rotation_degrees": args.stroke_local_rotation_degrees,
        "stroke_local_scale_jitter": args.stroke_local_scale_jitter,
        "stroke_local_translation_jitter": args.stroke_local_translation_jitter,
        "stroke_curve_probability": args.stroke_curve_probability,
        "stroke_curve_probability_by_family": resolved_curve_probabilities,
        "stroke_curve_max_displacement": args.stroke_curve_max_displacement,
        "stroke_curve_max_displacement_by_family": resolved_curve_displacements,
        "stroke_curve_max_waves": args.stroke_curve_max_waves,
        "stroke_curve_max_waves_by_family": resolved_curve_max_waves,
        "stroke_curve_shape": (
            "arc-length sinusoidal normal offset with pinned stroke endpoints and no-backtracking gate"
            if any(
                resolved_curve_probabilities[family] > 0.0 and resolved_curve_displacements[family] > 0.0
                for family in SYMBOL_FAMILY_NAMES
            )
            else None
        ),
    }


def _load_cache(work_dir: Path, split: str):
    paths = {
        "features": work_dir / f"{split}_features.npy",
        "labels": work_dir / f"{split}_labels.npy",
        "sources": work_dir / f"{split}_sources.npy",
    }
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError(f"{split} cache is incomplete in {work_dir}")
    result = {key: np.load(path, mmap_mode="r") for key, path in paths.items()}
    if result["features"].shape[1:] != (POINTS, len(CHANNELS)):
        raise ValueError(f"invalid {split} cached feature shape: {result['features'].shape}")
    if not (len(result["features"]) == len(result["labels"]) == len(result["sources"])):
        raise ValueError(f"{split} cached row counts disagree")
    return result


@torch.inference_mode()
def _predict_logits(model, features: np.ndarray, device: torch.device, batch_size: int = 64) -> np.ndarray:
    outputs = []
    model.eval()
    for start in range(0, len(features), batch_size):
        batch = torch.from_numpy(np.array(features[start:start + batch_size], dtype=np.float32, copy=True)).to(device)
        logits = model.math_head(model.encode(batch)).float()
        outputs.append(logits.cpu().numpy())
    result = np.concatenate(outputs, axis=0) if outputs else np.empty((0, 372), dtype=np.float32)
    if result.shape != (len(features), 372) or not np.isfinite(result).all():
        raise ValueError("baseline/student returned invalid 372-class logits")
    return result


def _batch_hard_rank_margin_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Push each target above its current highest-scoring non-target class."""
    if logits.ndim != 2 or targets.ndim != 1 or logits.shape[0] != targets.shape[0]:
        raise ValueError("batch-hard margin expects [batch, classes] logits and [batch] targets")
    if logits.shape[1] < 2 or margin < 0.0:
        raise ValueError("batch-hard margin requires at least two classes and a non-negative margin")
    scores = logits.float()
    rivals = scores.clone()
    rivals.scatter_(1, targets.long().unsqueeze(1), float("-inf"))
    rival_logits, rival_ids = rivals.max(dim=1)
    target_logits = scores.gather(1, targets.long().unsqueeze(1)).squeeze(1)
    loss = F.relu(float(margin) - (target_logits - rival_logits)).mean()
    return loss, rival_ids


def _score(logits: np.ndarray, target_ids: np.ndarray, source_ids: np.ndarray, labels: list[str]) -> dict[str, Any]:
    top1 = np.argmax(logits, axis=1)
    top5 = np.argpartition(-logits, kth=4, axis=1)[:, :5]
    target = np.asarray(target_ids, dtype=np.int64)
    rows = []
    for name, source_id in (("hwrt", SOURCE_IDS["hwrt"]), ("uji", SOURCE_IDS["uji"])):
        indices = np.flatnonzero(source_ids == source_id)
        if not len(indices):
            continue
        rows.append((name, indices))

    def compact(indices: np.ndarray) -> dict[str, Any]:
        hits1 = top1[indices] == target[indices]
        hits5 = np.asarray([target[index] in top5[index] for index in indices], dtype=bool)
        by_family: dict[str, dict[str, Any]] = {}
        fam_indices: dict[str, list[int]] = defaultdict(list)
        for index in indices.tolist():
            fam_indices[_family(labels[int(target[index])])].append(index)
        for family, family_rows in sorted(fam_indices.items()):
            family_array = np.asarray(family_rows, dtype=np.int64)
            by_family[family] = {
                "rows": len(family_rows),
                "top1": float(np.mean(top1[family_array] == target[family_array])),
                "top5": float(np.mean([target[index] in top5[index] for index in family_rows])),
            }
        return {
            "rows": int(len(indices)),
            "top1_hits": int(hits1.sum()),
            "top5_hits": int(hits5.sum()),
            "top1": float(hits1.mean()),
            "top5": float(hits5.mean()),
            "by_family": by_family,
        }
    return {
        "overall": compact(np.arange(len(target), dtype=np.int64)),
        "by_source": {name: compact(indices) for name, indices in rows},
    }


def _select_qa_rows(features: np.ndarray, target_ids: np.ndarray, labels: list[str], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected = []
    for class_id in range(len(labels)):
        candidates = np.flatnonzero(target_ids == class_id)
        if not len(candidates):
            raise ValueError(f"no training examples for class {labels[class_id]!r}")
        take = min(QA_SAMPLES_PER_CLASS, len(candidates))
        selected.extend(rng.choice(candidates, size=take, replace=False).tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


def _write_sample_sheet(
    path: Path,
    original: np.ndarray,
    augmented: np.ndarray,
    target_ids: np.ndarray,
    source_ids: np.ndarray,
    labels: list[str],
    original_logits: np.ndarray,
    augmented_logits: np.ndarray,
) -> dict[str, Any]:
    from PIL import Image, ImageDraw, ImageFont

    families = (
        "digits", "latin_lower", "latin_upper", "greek", "relations",
        "operators", "fences", "arrows", "large_ops", "other_math",
    )
    indices_by_class: dict[int, list[int]] = defaultdict(list)
    for index, class_id in enumerate(target_ids.tolist()):
        indices_by_class[int(class_id)].append(index)
    chosen: dict[str, list[int]] = {}
    preferred = {
        "digits": list("0123456789"),
        "relations": ["=", "<", ">", r"\leq", r"\neq", r"\approx", r"\equiv"],
        "operators": ["+", "-", "/", r"\times", r"\div", r"\pm"],
        "fences": ["(", ")", "[", "]", r"\{", r"\}"],
    }
    limits = {"digits": 10, "relations": 8, "operators": 8, "fences": 8}
    for family in families:
        candidates = [index for class_id, label in enumerate(labels) if _family(label) == family for index in indices_by_class.get(class_id, [])]
        seen_labels = set()
        family_indices = []
        for index in candidates:
            label = labels[int(target_ids[index])]
            if label not in seen_labels:
                seen_labels.add(label)
                family_indices.append(index)
        by_label = {labels[int(target_ids[index])]: index for index in family_indices}
        ordered = [by_label[label] for label in preferred.get(family, []) if label in by_label]
        ordered.extend(index for index in family_indices if index not in ordered)
        chosen[family] = ordered[:limits.get(family, 8)]
    cell_width, cell_height, margin, panel = 132, 114, 8, 44
    rows = len(families)
    image = Image.new("RGB", (10 * cell_width + 2 * margin, 34 + rows * cell_height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((margin, 8), "AIFlow 372-class train samples: identity vs bounded global affine view", fill="#102a43", font=font)

    def draw_glyph(feature: np.ndarray, box: tuple[int, int, int, int], color: str) -> None:
        x0, y0, width, height = box
        starts = np.flatnonzero(feature[:, 3] > 0.5).tolist()
        ends = starts[1:] + [len(feature)]
        for start, end in zip(starts, ends, strict=True):
            coords = [
                (x0 + 3 + float(feature[i, 0]) * (width - 6), y0 + 3 + float(feature[i, 1]) * (height - 6))
                for i in range(start, end)
            ]
            if len(coords) > 1:
                draw.line(coords, fill=color, width=2, joint="curve")
            elif coords:
                x, y = coords[0]
                draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=color)

    source_name = {0: "H", 1: "U", 2: "S"}
    for row_index, family in enumerate(families):
        y = 34 + row_index * cell_height
        draw.text((margin, y + 1), family, fill="#243b53", font=font)
        for col, index in enumerate(chosen[family]):
            x = margin + col * cell_width
            label = labels[int(target_ids[index])]
            before = labels[int(np.argmax(original_logits[index]))]
            after = labels[int(np.argmax(augmented_logits[index]))]
            draw.text((x, y + 14), f"{label} [{source_name[int(source_ids[index])]}]", fill="#334e68", font=font)
            left = (x, y + 31, panel, panel)
            right = (x + panel + 10, y + 31, panel, panel)
            draw.rectangle((left[0], left[1], left[0] + panel, left[1] + panel), outline="#d9e2ec")
            draw.rectangle((right[0], right[1], right[0] + panel, right[1] + panel), outline="#d9e2ec")
            draw_glyph(original[index], left, "#1d3557")
            draw_glyph(augmented[index], right, "#d1495b")
            draw.text((x, y + 84), f"{before}>{after}", fill="#52606d", font=font)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return {
        "path": str(path.resolve()),
        "families": {family: [labels[int(target_ids[i])] for i in indices] for family, indices in chosen.items()},
        "synthetic_equal_visible": "=" in [labels[int(target_ids[i])] for i in chosen["relations"]],
        "source_codes": {"H": "HWRT real train", "U": "UJI real train", "S": "synthetic '='"},
        "legend": "each symbol is shown as source ink then affine-augmented ink; captions show teacher Top1 before>after",
    }


def _write_full_vocabulary_sample_pages(
    output_dir: Path,
    originals: np.ndarray,
    augmented_views: list[np.ndarray],
    labels: list[str],
    teacher_logits: list[np.ndarray],
    engine: str,
) -> list[dict[str, Any]]:
    """Render one source example and every audited view for every model class."""
    from PIL import Image, ImageDraw, ImageFont

    if len(originals) != len(labels) or any(len(view) != len(originals) for view in augmented_views):
        raise ValueError("full-vocabulary image audit arrays do not align")
    if len(augmented_views) != len(teacher_logits):
        raise ValueError("teacher logits must be supplied for each augmented view")

    def draw_glyph(draw, feature: np.ndarray, box: tuple[int, int, int, int], color: str) -> None:
        x0, y0, width, height = box
        starts = np.flatnonzero(feature[:, 3] > 0.5).tolist()
        ends = starts[1:] + [len(feature)]
        for start, end in zip(starts, ends, strict=True):
            coords = [
                (x0 + 2 + float(feature[i, 0]) * (width - 4), y0 + 2 + float(feature[i, 1]) * (height - 4))
                for i in range(start, end)
            ]
            if len(coords) > 1:
                draw.line(coords, fill=color, width=2, joint="curve")
            elif coords:
                x, y = coords[0]
                draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=color)

    output_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    cell_width, cell_height, page_classes = 216, 106, 96
    cols = 8
    image_rows = []
    for start in range(0, len(labels), page_classes):
        stop = min(start + page_classes, len(labels))
        count = stop - start
        rows = math.ceil(count / cols)
        image = Image.new("RGB", (cols * cell_width, rows * cell_height + 26), "white")
        draw = ImageDraw.Draw(image)
        draw.text((6, 6), f"Full 372-class coverage: source / {engine} views; page {start // page_classes + 1}", fill="#102a43", font=font)
        for local_index, sample_index in enumerate(range(start, stop)):
            col, row = local_index % cols, local_index // cols
            x, y = col * cell_width, 26 + row * cell_height
            draw.rectangle((x + 1, y + 1, x + cell_width - 2, y + cell_height - 2), outline="#d9e2ec")
            draw.text((x + 5, y + 3), labels[sample_index], fill="#243b53", font=font)
            panels = [originals[sample_index], *(view[sample_index] for view in augmented_views)]
            panel_width = min(39, (cell_width - 12 - (len(panels) - 1) * 3) // len(panels))
            panel_height, gap = 56, 3
            total_width = len(panels) * panel_width + (len(panels) - 1) * gap
            panel_x = x + max(4, (cell_width - total_width) // 2)
            for panel_index, feature in enumerate(panels):
                px = panel_x + panel_index * (panel_width + gap)
                py = y + 19
                draw.rectangle((px, py, px + panel_width, py + panel_height), outline="#d9e2ec")
                color = "#1d3557" if panel_index == 0 else "#d1495b"
                draw_glyph(draw, feature, (px + 1, py + 1, panel_width - 2, panel_height - 2), color)
                prediction = labels[int(np.argmax(teacher_logits[panel_index - 1][sample_index]))] if panel_index else "src"
                draw.text((px + 2, py + panel_height + 2), f"{panel_index}:{prediction[:5]}", fill="#52606d", font=font)
            image_rows.append({"label": labels[sample_index], "page": start // page_classes + 1})
        image_path = output_dir / f"full_domain_{engine}_samples_{start // page_classes + 1:02d}.png"
        if image_path.exists():
            raise FileExistsError(image_path)
        image.save(image_path)
        image_rows.append({"page_image": str(image_path.resolve()), "classes": count})
    return image_rows


def _preflight_full_vocabulary_sample_outputs(output_dir: Path) -> None:
    """Refuse a colliding or mixed sample bundle before expensive audit work starts."""
    prefix, suffix = "full_domain_", "_sample_pages"
    engine = output_dir.name[len(prefix):-len(suffix)] if (
        output_dir.name.startswith(prefix) and output_dir.name.endswith(suffix)
    ) else "affine_elastic_mix"
    longest_sample_path = output_dir / f"full_domain_{engine}_samples_04.png"
    if os.name == "nt" and len(str(longest_sample_path)) >= 260:
        raise ValueError(
            "full-vocabulary sample image path exceeds the Windows MAX_PATH limit "
            f"({len(str(longest_sample_path))} characters): {longest_sample_path}; "
            "rerun with a shorter --report-dir"
        )
    if not output_dir.exists():
        return
    if not output_dir.is_dir():
        raise FileExistsError(f"sample output path is not a directory: {output_dir}")
    if any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to mix full-vocabulary sample output with existing files: {output_dir}")


def _summarize_pairwise_view_diversity(
    views: list[np.ndarray], qa_targets: np.ndarray, labels: list[str],
) -> dict[str, Any]:
    """Measure geometric diversity among views of the same source rows only."""
    if len(views) < 2:
        return {
            "status": "insufficient_views",
            "scope": "training-pool rows only; geometry diagnostic, not model selection",
            "view_count": len(views),
        }
    if any(view.shape != views[0].shape for view in views):
        raise ValueError("pairwise augmentation views must have identical shapes")
    if views[0].ndim != 3 or views[0].shape[0] != len(qa_targets):
        raise ValueError("pairwise augmentation views do not match QA row targets")

    pairwise_rms: list[np.ndarray] = []
    affine_aligned_residual_rms: list[np.ndarray] = []
    pairwise_identical: list[np.ndarray] = []
    for left in range(len(views)):
        for right in range(left + 1, len(views)):
            left_xy = np.asarray(views[left][:, :, :2], dtype=np.float64)
            right_xy = np.asarray(views[right][:, :, :2], dtype=np.float64)
            delta = left_xy - right_xy
            pairwise_rms.append(np.sqrt(np.mean(np.square(delta), axis=(1, 2))))
            pairwise_identical.append(np.max(np.abs(delta), axis=(1, 2)) <= 1.0e-8)
            # Remove each pair's best global affine map. The residual exposes
            # local/elastic shape changes that global rotation, scale, or shear
            # can otherwise dominate in raw XY RMS.
            design = np.concatenate((left_xy, np.ones((*left_xy.shape[:2], 1))), axis=2)
            gram = np.einsum("npi,npj->nij", design, design, optimize=True)
            cross = np.einsum("npi,npj->nij", design, right_xy, optimize=True)
            coefficients = np.einsum("nij,njk->nik", np.linalg.pinv(gram), cross, optimize=True)
            fitted = np.einsum("npi,nij->npj", design, coefficients, optimize=True)
            residual = right_xy - fitted
            affine_aligned_residual_rms.append(np.sqrt(np.mean(np.square(residual), axis=(1, 2))))

    rms = np.stack(pairwise_rms, axis=0)
    aligned_rms = np.stack(affine_aligned_residual_rms, axis=0)
    identical = np.stack(pairwise_identical, axis=0)

    def summarize(rows: np.ndarray) -> dict[str, Any]:
        if len(rows) == 0:
            return {
                "source_rows": 0,
                "pairwise_xy_rms_mean": None,
                "pairwise_xy_rms_median": None,
                "pairwise_xy_rms_p95": None,
                "affine_aligned_residual_xy_rms_mean": None,
                "affine_aligned_residual_xy_rms_median": None,
                "affine_aligned_residual_xy_rms_p95": None,
                "identical_view_pair_fraction": None,
                "source_rows_with_any_identical_pair_fraction": None,
                "source_rows_with_all_views_distinct_fraction": None,
            }
        selected_rms = rms[:, rows]
        selected_aligned_rms = aligned_rms[:, rows]
        selected_identical = identical[:, rows]
        return {
            "source_rows": int(len(rows)),
            "pairwise_xy_rms_mean": float(np.mean(selected_rms)),
            "pairwise_xy_rms_median": float(np.median(selected_rms)),
            "pairwise_xy_rms_p95": float(np.percentile(selected_rms, 95)),
            "affine_aligned_residual_xy_rms_mean": float(np.mean(selected_aligned_rms)),
            "affine_aligned_residual_xy_rms_median": float(np.median(selected_aligned_rms)),
            "affine_aligned_residual_xy_rms_p95": float(np.percentile(selected_aligned_rms, 95)),
            "identical_view_pair_fraction": float(np.mean(selected_identical)),
            "source_rows_with_any_identical_pair_fraction": float(np.mean(np.any(selected_identical, axis=0))),
            "source_rows_with_all_views_distinct_fraction": float(np.mean(~np.any(selected_identical, axis=0))),
        }

    family_by_row = np.asarray([_family(labels[int(class_id)]) for class_id in qa_targets], dtype=object)
    return {
        "status": "measured",
        "scope": "training-pool rows only; raw and global-affine-aligned XY geometry diagnostics, never used for model selection",
        "view_count": len(views),
        "pairs_per_source_row": len(pairwise_rms),
        "identity_tolerance_xy": 1.0e-8,
        "affine_alignment": "unconstrained least-squares 2D affine map fitted independently per source row and view pair",
        "overall": summarize(np.arange(len(qa_targets), dtype=np.int64)),
        "by_class_family": {
            family: summarize(np.flatnonzero(family_by_row == family))
            for family in sorted(set(family_by_row.tolist()))
        },
    }


def _audit_multi_view_augmentation(args) -> int:
    """Pre-train data-unit and full-vocabulary visual audit of a larger view batch."""
    data_dir = args.data_dir or args.work_dir
    manifest_path = data_dir / "prepared_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "pass" or manifest.get("schema") != SCHEMA:
        raise ValueError("the source data audit did not pass")
    if manifest.get("current_checkpoint", {}).get("sha256") != _sha256(args.checkpoint):
        raise ValueError("source audit belongs to a different frozen teacher")
    audit_path = args.report_dir / "full_domain_augmentation_audit.json"
    if audit_path.exists():
        raise FileExistsError(audit_path)

    sample_output_dir = args.report_dir / f"full_domain_{args.augmentation}_sample_pages"
    _preflight_full_vocabulary_sample_outputs(sample_output_dir)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    teacher, labels, _ = _load_teacher(args.checkpoint, device)
    cache = _load_cache(data_dir, "train")
    features, targets, sources = cache["features"], cache["labels"], cache["sources"]
    qa_indices = _select_qa_rows(features, targets, labels, RNG_SEED + 811)
    originals = np.asarray(features[qa_indices], dtype=np.float32).copy()
    qa_targets = np.asarray(targets[qa_indices], dtype=np.int64)
    qa_sources = np.asarray(sources[qa_indices], dtype=np.int8)
    if ((qa_targets < 0) | (qa_targets >= len(labels))).any():
        raise ValueError("cannot resolve family-specific stroke curves with an invalid QA target id")
    qa_families = [_family(labels[int(class_id)]) for class_id in qa_targets]
    curve_probability: float | torch.Tensor = args.stroke_curve_probability
    if args.stroke_curve_probability_by_family:
        curve_probability = torch.tensor(
            [
                args.stroke_curve_probability_by_family.get(family, args.stroke_curve_probability)
                for family in qa_families
            ],
            dtype=torch.float32,
            device=device,
        )
    curve_displacement: float | torch.Tensor = args.stroke_curve_max_displacement
    if args.stroke_curve_displacement_by_family:
        curve_displacement = torch.tensor(
            [
                args.stroke_curve_displacement_by_family.get(
                    _family(labels[int(class_id)]), args.stroke_curve_max_displacement,
                )
                for class_id in qa_targets
            ],
            dtype=torch.float32,
            device=device,
        )
    curve_max_waves: int | torch.Tensor = args.stroke_curve_max_waves
    if args.stroke_curve_max_waves_by_family:
        curve_max_waves = torch.tensor(
            [
                args.stroke_curve_max_waves_by_family.get(family, args.stroke_curve_max_waves)
                for family in qa_families
            ],
            dtype=torch.long,
            device=device,
        )
    generator = torch.Generator(device=device).manual_seed(RNG_SEED + 812)
    stroke_generator = torch.Generator(device=device).manual_seed(RNG_SEED + 813)
    curve_generator = torch.Generator(device=device).manual_seed(RNG_SEED + 814)
    curve_selection_generator = torch.Generator(device=device).manual_seed(RNG_SEED + 815)

    views: list[np.ndarray] = []
    view_reports: list[dict[str, Any]] = []
    unit_failures = Counter()
    for view_index in range(args.audit_views):
        view_tensor, diag = _augment_with_engine(
            torch.from_numpy(originals).to(device), generator, args.augmentation, args.elastic_probability,
            rotation_degrees=args.rotation_degrees,
            axis_scale_jitter=args.axis_scale_jitter,
            shear_jitter=args.shear_jitter,
            elastic_control_displacement=args.elastic_control_displacement,
            wide_parameter_probability=args.wide_parameter_probability,
            wide_magnitude_multiplier=args.wide_magnitude_multiplier,
            wide_probability_by_parameter=args.wide_probability_by_parameter,
            wide_multiplier_by_parameter=args.wide_multiplier_by_parameter,
            stroke_local_probability=args.stroke_local_probability,
            stroke_local_rotation_degrees=args.stroke_local_rotation_degrees,
            stroke_local_scale_jitter=args.stroke_local_scale_jitter,
            stroke_local_translation_jitter=args.stroke_local_translation_jitter,
            stroke_generator=stroke_generator,
            stroke_curve_probability=curve_probability,
            stroke_curve_max_displacement=curve_displacement,
            stroke_curve_max_waves=curve_max_waves,
            stroke_curve_family_by_row=qa_families,
            stroke_curve_selection_generator=curve_selection_generator,
            curve_generator=curve_generator,
        )
        view = view_tensor.cpu().numpy()
        if not np.isfinite(view).all():
            unit_failures["non_finite"] += int((~np.isfinite(view)).any(axis=(1, 2)).sum())
        if (view[:, :, :2] < 0).any() or (view[:, :, :2] > 1).any():
            unit_failures["xy_out_of_range"] += int(((view[:, :, :2] < 0) | (view[:, :, :2] > 1)).any(axis=(1, 2)).sum())
        changed_channels = np.any(view[:, :, 2:] != originals[:, :, 2:], axis=(1, 2))
        unit_failures["nonspatial_channel_changed"] += int(changed_channels.sum())
        changed_starts = np.any((view[:, :, 3] > 0.5) != (originals[:, :, 3] > 0.5), axis=1)
        unit_failures["stroke_start_changed"] += int(changed_starts.sum())
        bad_class_ids = (qa_targets < 0) | (qa_targets >= len(labels))
        unit_failures["invalid_target_id"] += int(bad_class_ids.sum())
        views.append(view)
        view_reports.append({"view_index": view_index + 1, **diag})

    # Image review covers every class, not just a few examples from each family.
    class_first = []
    first_by_class = {}
    for index, class_id in enumerate(qa_targets.tolist()):
        first_by_class.setdefault(int(class_id), index)
    if set(first_by_class) != set(range(len(labels))):
        raise AssertionError("visual QA sample did not cover all frozen model classes")
    for class_id in range(len(labels)):
        class_first.append(first_by_class[class_id])
    representative_originals = originals[class_first]
    representative_views = [view[class_first] for view in views]
    original_logits = _predict_logits(teacher, representative_originals, device, args.eval_batch_size)
    teacher_logits = [
        _predict_logits(teacher, representative_views[index], device, args.eval_batch_size)
        for index in range(len(representative_views))
    ]
    visual_rows = _write_full_vocabulary_sample_pages(
        sample_output_dir,
        representative_originals,
        representative_views,
        labels,
        teacher_logits,
        args.augmentation,
    )

    # Quantify full QA-batch teacher stability as well as the 372-class contact sheet.
    original_qa_logits = _predict_logits(teacher, originals, device, args.eval_batch_size)
    original_qa_top1 = np.argmax(original_qa_logits, axis=1)
    original_qa_top5 = np.argpartition(-original_qa_logits, kth=4, axis=1)[:, :5]

    def robustness_summary(view_logits: np.ndarray) -> dict[str, Any]:
        predicted = np.argmax(view_logits, axis=1)
        top5 = np.argpartition(-view_logits, kth=4, axis=1)[:, :5]
        def compact(indices: np.ndarray) -> dict[str, Any]:
            return {
                "rows": int(len(indices)),
                "teacher_top1_agreement_with_source": float(np.mean(predicted[indices] == original_qa_top1[indices])),
                "source_teacher_top1_label_accuracy": float(np.mean(original_qa_top1[indices] == qa_targets[indices])),
                "augmented_teacher_top1_label_accuracy": float(np.mean(predicted[indices] == qa_targets[indices])),
                "source_teacher_top5_label_accuracy": float(np.mean([
                    qa_targets[index] in original_qa_top5[index] for index in indices
                ])),
                "augmented_teacher_top5_label_accuracy": float(np.mean([
                    qa_targets[index] in top5[index] for index in indices
                ])),
            }
        family_rows: dict[str, list[int]] = defaultdict(list)
        source_rows: dict[str, list[int]] = defaultdict(list)
        source_names = {value: key for key, value in SOURCE_IDS.items()}
        for index, class_id in enumerate(qa_targets.tolist()):
            family_rows[_family(labels[class_id])].append(index)
            source_rows[source_names.get(int(qa_sources[index]), "unknown")].append(index)
        return {
            "overall": compact(np.arange(len(qa_targets), dtype=np.int64)),
            "by_family": {key: compact(np.asarray(value, dtype=np.int64)) for key, value in sorted(family_rows.items())},
            "by_source": {key: compact(np.asarray(value, dtype=np.int64)) for key, value in sorted(source_rows.items())},
        }

    teacher_qa_robustness = []
    for view_index, view in enumerate(views, start=1):
        view_logits = _predict_logits(teacher, view, device, args.eval_batch_size)
        teacher_qa_robustness.append({
            "view_index": view_index,
            **robustness_summary(view_logits),
        })
    pairwise_view_diversity = _summarize_pairwise_view_diversity(views, qa_targets, labels)
    expected = np.arange(len(labels), dtype=np.int64)
    original_predictions = np.argmax(original_logits, axis=1)
    original_hits = original_predictions == expected
    visual_top1_by_view = [{
        "view_index": 0,
        "classes": len(labels),
        "top1_hits": int(original_hits.sum()),
        "prediction_changed_from_original": 0,
    }]
    for index, logits in enumerate(teacher_logits):
        predicted = np.argmax(logits, axis=1)
        visual_top1_by_view.append({
            "view_index": index + 1,
            "classes": len(labels),
            "top1_hits": int((predicted == expected).sum()),
            "prediction_changed_from_original": int((predicted != original_predictions).sum()),
            "top1_delta_from_original": int((predicted == expected).sum() - original_hits.sum()),
        })
    family_coverage = Counter(_family(label) for label in labels)
    total_unit_failures = int(sum(unit_failures.values()))
    training_cache_sha256 = {
        f"train_{name}.npy": _sha256(data_dir / f"train_{name}.npy")
        for name in ("features", "labels", "sources")
    }
    qa_row_indices_sha256 = hashlib.sha256(
        np.asarray(qa_indices, dtype="<i8").tobytes()
    ).hexdigest()
    report = {
        "schema": SCHEMA,
        "status": "pass" if total_unit_failures == 0 else "fail",
        "provenance": {
            "audit_script_sha256": _sha256(Path(__file__).resolve()),
            "prepared_manifest_sha256": _sha256(manifest_path),
            "teacher_checkpoint_sha256": _sha256(args.checkpoint),
            "training_cache_sha256": training_cache_sha256,
            "qa_row_indices_sha256": qa_row_indices_sha256,
            "qa_selection_seed": RNG_SEED + 811,
        },
        "scope": "all 372 frozen mathematical-domain classes: digits, Latin/Greek letters, operators, relations, fences, arrows, large operators, and remaining math symbols",
        "augmentation_engine": args.augmentation,
        "training_pool_rows": int(len(features)),
        "heldout_rows_used": 0,
        "crohme_rows": 0,
        "class_count": len(labels),
        "class_family_coverage": dict(sorted(family_coverage.items())),
        "data_unit_validation": {
            "unique_rows": int(len(qa_indices)),
            "rows_per_class_max": QA_SAMPLES_PER_CLASS,
            "augmented_views_per_row": args.audit_views,
            "augmented_data_units_checked": int(len(qa_indices) * args.audit_views),
            "unit_failures": dict(unit_failures),
            "unit_failure_total": total_unit_failures,
            "all_labels_immutable": True,
            "all_nonspatial_channels_identical": not unit_failures["nonspatial_channel_changed"],
            "stroke_count_and_start_positions_preserved": not unit_failures["stroke_start_changed"],
            "finite_xy_in_unit_square": not unit_failures["non_finite"] and not unit_failures["xy_out_of_range"],
            "per_view_transform_diagnostics": view_reports,
            "selected_rows_by_source": {
                {value: key for key, value in SOURCE_IDS.items()}.get(int(source_id), "unknown"):
                int(np.sum(qa_sources == source_id))
                for source_id in np.unique(qa_sources)
            },
        },
        "teacher_robustness_on_qa_rows": {
            "scope": "training pool only; diagnostic, never used for model selection",
            "views": teacher_qa_robustness,
        },
        "augmentation_diversity_geometry_only": pairwise_view_diversity,
        "teacher_visual_representative_stability": visual_top1_by_view,
        "image_sample_verification": {
            "classes_visually_sampled": len(labels),
            "panels_per_class": 1 + args.audit_views,
            "augmentation_engine": args.augmentation,
            "images": visual_rows,
        },
        "parameters": {
            **_augmentation_parameters(args),
            "seed": RNG_SEED + 812,
            "stroke_local_seed": RNG_SEED + 813,
            "stroke_curve_seed": RNG_SEED + 814,
            "stroke_curve_selection_seed": RNG_SEED + 815,
        },
        "heldout_used_for_selection": False,
        "product_adopted": False,
    }
    _write_json(audit_path, report)
    print(json.dumps({"event": "full_domain_augmentation_volume_audit", "status": report["status"],
                      "engine": args.augmentation, "data_units": len(qa_indices) * args.audit_views, "classes": len(labels),
                      "view_diagnostics": view_reports, "report": str(audit_path.resolve())}, ensure_ascii=False), flush=True)
    return 0 if report["status"] == "pass" else 2


def prepare(args) -> int:
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    teacher, labels, _checkpoint = _load_teacher(args.checkpoint, device)
    manifest = _prepare_cache(args.canonical_root, args.checkpoint, args.work_dir, labels)
    cache = _load_cache(args.work_dir, "train")
    heldout_cache = _load_cache(args.work_dir, "test")
    features = cache["features"]
    targets = cache["labels"]
    sources = cache["sources"]
    if manifest["current_checkpoint"]["sha256"] != _sha256(args.checkpoint):
        raise AssertionError("checkpoint changed while preparing the experiment")

    qa_indices = _select_qa_rows(features, targets, labels, RNG_SEED + 2)
    originals = np.asarray(features[qa_indices], dtype=np.float32).copy()
    generator = torch.Generator(device=device).manual_seed(RNG_SEED + 3)
    stroke_generator = torch.Generator(device=device).manual_seed(RNG_SEED + 4)
    curve_generator = torch.Generator(device=device).manual_seed(RNG_SEED + 5)
    augmented_tensor, transform_report = _augment_with_engine(
        torch.from_numpy(originals).to(device), generator, args.augmentation, args.elastic_probability,
        rotation_degrees=args.rotation_degrees,
        axis_scale_jitter=args.axis_scale_jitter,
        shear_jitter=args.shear_jitter,
        elastic_control_displacement=args.elastic_control_displacement,
        wide_parameter_probability=args.wide_parameter_probability,
        wide_magnitude_multiplier=args.wide_magnitude_multiplier,
        wide_probability_by_parameter=args.wide_probability_by_parameter,
        wide_multiplier_by_parameter=args.wide_multiplier_by_parameter,
        stroke_local_probability=args.stroke_local_probability,
        stroke_local_rotation_degrees=args.stroke_local_rotation_degrees,
        stroke_local_scale_jitter=args.stroke_local_scale_jitter,
        stroke_local_translation_jitter=args.stroke_local_translation_jitter,
        stroke_generator=stroke_generator,
        stroke_curve_probability=args.stroke_curve_probability,
        stroke_curve_max_displacement=args.stroke_curve_max_displacement,
        stroke_curve_max_waves=args.stroke_curve_max_waves,
        curve_generator=curve_generator,
    )
    augmented = augmented_tensor.cpu().numpy()
    if not np.array_equal(originals[:, :, 2:], augmented[:, :, 2:]):
        raise AssertionError("augmentation changed non-spatial channels during microscope QA")
    if not np.array_equal((originals[:, :, 3] > 0.5), (augmented[:, :, 3] > 0.5)):
        raise AssertionError("augmentation changed stroke count or stroke starts")
    if not np.isfinite(augmented).all() or (augmented[:, :, :2] < 0).any() or (augmented[:, :, :2] > 1).any():
        raise AssertionError("augmentation QA contains non-finite or out-of-range XY")

    original_logits = _predict_logits(teacher, originals, device)
    augmented_logits = _predict_logits(teacher, augmented, device)
    qa_targets = np.asarray(targets[qa_indices], dtype=np.int64)
    qa_sources = np.asarray(sources[qa_indices], dtype=np.int8)
    baseline_score = _score(original_logits, qa_targets, qa_sources, labels)
    augmented_score = _score(augmented_logits, qa_targets, qa_sources, labels)
    truth = qa_targets
    top1_original = np.argmax(original_logits, axis=1)
    top1_augmented = np.argmax(augmented_logits, axis=1)
    top5_original = np.argpartition(-original_logits, kth=4, axis=1)[:, :5]
    top5_augmented = np.argpartition(-augmented_logits, kth=4, axis=1)[:, :5]
    label_hits_original = top1_original == truth
    label_hits_augmented = top1_augmented == truth
    label_hits5_original = np.asarray([truth[i] in top5_original[i] for i in range(len(truth))])
    label_hits5_augmented = np.asarray([truth[i] in top5_augmented[i] for i in range(len(truth))])
    class_transition: dict[str, dict[str, int]] = {}
    for class_id, label in enumerate(labels):
        indices = np.flatnonzero(truth == class_id)
        class_transition[label] = {
            "qa_rows": int(len(indices)),
            "top1_before": int(label_hits_original[indices].sum()),
            "top1_after": int(label_hits_augmented[indices].sum()),
            "top5_before": int(label_hits5_original[indices].sum()),
            "top5_after": int(label_hits5_augmented[indices].sum()),
        }
    image_path = args.report_dir / "full_domain_affine_sample_sheet.png"
    visual_report = _write_sample_sheet(
        image_path, originals, augmented, qa_targets, qa_sources, labels,
        original_logits, augmented_logits,
    )
    manifest["status"] = "pass"
    manifest["data_unit_validation"] = {
        "rows_tensorized_and_checked": int(manifest["real_train_rows"] + len(heldout_cache["features"])),
        "real_training_rows": int(manifest["real_train_rows"]),
        "synthetic_rows_checked": int(SYNTHETIC_EQUAL_ROWS),
        "bad_rows": 0,
        "all_labels_in_frozen_vocabulary": True,
        "all_xy_finite_and_bounded": True,
        "uniform_time_contract": True,
        "stroke_start_channel_preserved": True,
    }
    manifest["augmentation_audit"] = {
        "algorithm": (
            "orientation-preserving global affine jitter"
            if args.augmentation == "affine"
            else (
                "orientation-preserving affine jitter + smooth boundary-pinned elastic field"
                if args.augmentation == "affine_elastic"
                else f"orientation-preserving affine jitter + elastic mixture (p={args.elastic_probability:.3f})"
            )
        ),
        "engine": args.augmentation,
        "parameters": _augmentation_parameters(args),
        "hard_limits": {"xy_rms": XY_RMS_LIMIT, "max_point_displacement": XY_POINT_LIMIT, "per_stroke_path_ratio": list(PATH_RATIO_RANGE)},
        "qa_rows": int(len(qa_indices)),
        "qa_rows_per_class_max": QA_SAMPLES_PER_CLASS,
        "unit_checks": transform_report,
        "teacher_identity_metrics": baseline_score,
        "teacher_augmented_metrics": augmented_score,
        "teacher_label_top1_delta": int(label_hits_augmented.sum() - label_hits_original.sum()),
        "teacher_label_top5_delta": int(label_hits5_augmented.sum() - label_hits5_original.sum()),
        "teacher_prediction_changed_rows": int((top1_original != top1_augmented).sum()),
        "per_class": class_transition,
        "image_sample_verification": visual_report,
        "qa_scope": "training pool only; no augmentation parameter selection from heldout metrics",
    }
    manifest["product_adopted"] = False
    _write_json(args.work_dir / "prepared_manifest.json", manifest)
    report_path = args.report_dir / "augmentation_audit.json"
    _write_json(report_path, {
        "schema": SCHEMA,
        "status": "pass",
        "work_dir": str(args.work_dir.resolve()),
        "report": str(report_path.resolve()),
        "real_train_rows": manifest["real_train_rows"],
        "synthetic_equals": SYNTHETIC_EQUAL_ROWS,
        "class_coverage": manifest["class_coverage"],
        "data_unit_validation": manifest["data_unit_validation"],
        "augmentation_audit": manifest["augmentation_audit"],
        "heldout_used_for_selection": False,
        "crohme_rows": 0,
        "product_adopted": False,
    })
    print(json.dumps({
        "event": "augmentation_audit_complete",
        "status": "pass",
        "real_train_rows": manifest["real_train_rows"],
        "synthetic_equals": SYNTHETIC_EQUAL_ROWS,
        "qa_rows": len(qa_indices),
        "teacher_top1_before": baseline_score["overall"]["top1"],
        "teacher_top1_after": augmented_score["overall"]["top1"],
        "report": str(report_path.resolve()),
        "sample_sheet": str(image_path.resolve()),
    }, ensure_ascii=False), flush=True)
    return 0


class CachedRows(Dataset):
    def __init__(self, work_dir: Path, teacher_logits: np.ndarray) -> None:
        self.features = np.load(work_dir / "train_features.npy", mmap_mode="r")
        self.labels = np.load(work_dir / "train_labels.npy", mmap_mode="r")
        self.sources = np.load(work_dir / "train_sources.npy", mmap_mode="r")
        self.teacher_logits = teacher_logits
        if not (len(self.features) == len(self.labels) == len(self.sources) == len(self.teacher_logits)):
            raise ValueError("cached train arrays do not align")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (
            torch.from_numpy(np.array(self.features[index], dtype=np.float32, copy=True)),
            int(self.labels[index]),
            torch.from_numpy(np.array(self.teacher_logits[index], dtype=np.float32, copy=True)),
            int(self.sources[index] == SOURCE_IDS["synthetic_equal"]),
        )


def _precompute_teacher_logits(model, features: np.ndarray, output_path: Path, device: torch.device, batch_size: int) -> np.ndarray:
    if output_path.exists():
        raise FileExistsError(f"teacher-logit cache already exists: {output_path}")
    logits_cache = np.lib.format.open_memmap(
        output_path, mode="w+", dtype=np.float16, shape=(len(features), 372),
    )
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            batch = torch.from_numpy(np.array(features[start:start + batch_size], dtype=np.float32, copy=True)).to(device)
            logits = model.math_head(model.encode(batch)).float().cpu().numpy()
            if not np.isfinite(logits).all():
                raise FloatingPointError(f"non-finite teacher logits at row {start}")
            logits_cache[start:start + len(logits)] = logits.astype(np.float16, copy=False)
            if start and start % max(batch_size * 500, 1) == 0:
                print(json.dumps({"event": "teacher_logits_progress", "rows": start, "total": len(features)}, ensure_ascii=False), flush=True)
    logits_cache.flush()
    return np.load(output_path, mmap_mode="r")


def _train(args) -> int:
    experiment_started = time.perf_counter()
    data_dir = args.data_dir or args.work_dir
    manifest_path = data_dir / "prepared_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"run --mode prepare first: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "pass" or manifest.get("schema") != SCHEMA:
        raise ValueError("data/augmentation microscope report did not pass")
    _assert_comparison_sources_eligible(
        manifest.get("input_policy", {}),
        event="train_and_compare",
        source_split_rows=manifest.get("source_audit", {}).get("source_split_rows", {}),
    )
    if manifest.get("current_checkpoint", {}).get("sha256") != _sha256(args.checkpoint):
        raise ValueError("frozen teacher checkpoint differs from preparation audit")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    if args.report_dir.exists() and (args.report_dir / "comparison.json").exists():
        raise FileExistsError("comparison report already exists")
    for name in ("student_checkpoint.pt", "teacher_train_logits.npy"):
        if (args.work_dir / name).exists():
            raise FileExistsError(f"refusing to overwrite {args.work_dir / name}")
    train = _load_cache(data_dir, "train")
    test = _load_cache(data_dir, "test")
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model, labels, _checkpoint = _load_teacher(args.checkpoint, device)
    boundary_features = boundary_teacher_logits = None
    boundary_replay = {"enabled": args.boundary_loss_weight > 0.0}
    if boundary_replay["enabled"]:
        boundary_report, boundary_x, boundary_logits = load_boundary_candidates(
            args.boundary_candidate_report, _sha256(args.checkpoint), labels,
        )
        boundary_features = torch.from_numpy(boundary_x.copy()).to(device)
        boundary_teacher_logits = torch.from_numpy(boundary_logits.copy()).to(device)
        boundary_replay.update({
            "candidate_report": str(args.boundary_candidate_report.resolve()),
            "candidate_report_sha256": _sha256(args.boundary_candidate_report),
            "teacher_checkpoint_sha256": boundary_report["provenance"]["teacher_checkpoint_sha256"],
            "candidate_rows": len(boundary_x), "pilot_scope": "class-2 only; not full-vocabulary coverage",
            "policy": args.boundary_policy, "temperature": args.boundary_temperature,
            "student_mode": args.boundary_student_mode,
            "loss_weight": args.boundary_loss_weight, "batch_size": args.boundary_batch_size,
            "base_loss_multiplier": 1.0 - args.boundary_loss_weight,
            "hard_synthetic_labels_assigned": 0, "human_labels_available": False,
            "candidate_extra_augmentation": False, "heldout_rows_read": 0, "crohme_rows": 0,
        })
    train_y = np.asarray(train["labels"], dtype=np.int64)
    test_y = np.asarray(test["labels"], dtype=np.int64)
    test_sources = np.asarray(test["sources"], dtype=np.int8)
    train_sources = np.asarray(train["sources"], dtype=np.int8)
    if np.any(train_y < 0) or np.any(train_y >= len(labels)) or np.any(test_y < 0) or np.any(test_y >= len(labels)):
        raise ValueError("class ID outside the frozen 372-class vocabulary")
    label_to_id = {label: index for index, label in enumerate(labels)}
    rank_margin_pairs = []
    for spec in args.rank_margin_pair:
        if ":" not in spec:
            raise ValueError(f"invalid --rank-margin-pair {spec!r}; expected TARGET:RIVAL")
        target_label, rival_label = spec.split(":", 1)
        if target_label not in label_to_id or rival_label not in label_to_id or target_label == rival_label:
            raise ValueError(f"unknown or identical labels in --rank-margin-pair {spec!r}")
        rank_margin_pairs.append((target_label, rival_label, label_to_id[target_label], label_to_id[rival_label]))
    if len({pair[:2] for pair in rank_margin_pairs}) != len(rank_margin_pairs):
        raise ValueError("duplicate --rank-margin-pair entries")
    if args.batch_hard_rank_margin and rank_margin_pairs:
        raise ValueError("batch-hard rank margin cannot be combined with fixed --rank-margin-pair entries")
    if args.rank_margin_weight > 0.0 and not rank_margin_pairs and not args.batch_hard_rank_margin:
        raise ValueError("positive --rank-margin-weight requires fixed pairs or --batch-hard-rank-margin")
    if args.batch_hard_rank_margin and args.rank_margin_weight <= 0.0:
        raise ValueError("--batch-hard-rank-margin requires a positive --rank-margin-weight")
    if not np.isclose(args.hard_loss_weight + args.distill_loss_weight + args.rank_margin_weight, 1.0):
        raise ValueError("hard, distill, and rank-margin loss weights must sum to 1")
    rank_margin_mode = "batch_hard" if args.batch_hard_rank_margin else ("fixed_pairs" if rank_margin_pairs else "none")

    baseline_started = time.perf_counter()
    baseline_logits = _predict_logits(model, test["features"], device, args.eval_batch_size)
    baseline_metrics = _score(baseline_logits, test_y, test_sources, labels)
    baseline_eval_seconds = time.perf_counter() - baseline_started
    teacher_cache_started = time.perf_counter()
    teacher_logits = _precompute_teacher_logits(
        model, train["features"], args.work_dir / "teacher_train_logits.npy",
        device, args.eval_batch_size,
    )
    teacher_cache_seconds = time.perf_counter() - teacher_cache_started
    print(json.dumps({
        "event": "training_start",
        "device": str(device),
        "train_rows": len(train_y),
        "test_rows": len(test_y),
        "classes": len(labels),
        "synthetic_equal_rows": int(np.sum(train_sources == SOURCE_IDS["synthetic_equal"])),
        "samples_per_class_per_epoch": args.samples_per_class_per_epoch,
        "epochs": args.epochs,
        "estimated_augmented_views": args.samples_per_class_per_epoch * len(labels) * args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "rank_margin_pairs": [[pair[0], pair[1]] for pair in rank_margin_pairs],
        "rank_margin_mode": rank_margin_mode,
        "rank_margin": args.rank_margin,
        "rank_margin_weight": args.rank_margin_weight,
        "augmentation_engine": args.augmentation,
        "elastic_probability": args.elastic_probability,
        "augmentation_parameters": _augmentation_parameters(args),
        "boundary_replay": boundary_replay,
    }, ensure_ascii=False), flush=True)

    dataset = CachedRows(data_dir, teacher_logits)
    class_counts = np.bincount(train_y, minlength=len(labels)).astype(np.float64)
    if np.any(class_counts <= 0):
        missing = [labels[index] for index in np.flatnonzero(class_counts <= 0)]
        raise ValueError(f"class-balanced student sampler has empty classes: {missing}")
    sample_weights = torch.as_tensor(1.0 / class_counts[train_y], dtype=torch.double)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    generator = torch.Generator(device=device).manual_seed(RNG_SEED + 71)
    stroke_generator = torch.Generator(device=device).manual_seed(RNG_SEED + 72)
    curve_generator = torch.Generator(device=device).manual_seed(RNG_SEED + 73)
    boundary_generator = torch.Generator().manual_seed(RNG_SEED + 74)
    amp = device.type == "cuda"
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1.0e-4)
    history = []
    total_augmented = 0
    total_changed = 0
    total_reverted = 0
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=args.samples_per_class_per_epoch * len(labels),
            replacement=True,
            generator=torch.Generator().manual_seed(RNG_SEED + epoch),
        )
        loader = DataLoader(
            dataset, batch_size=args.batch_size, sampler=sampler,
            num_workers=0, pin_memory=amp,
        )
        model.train()
        hard_losses, distill_losses, rank_losses = [], [], []
        boundary_losses = []
        epoch_boundary_views = 0
        epoch_changed = 0
        epoch_reverted = 0
        epoch_views = 0
        epoch_ranked_rows = 0
        epoch_hard_negative_rivals: Counter[int] = Counter()
        epoch_wide_selected: Counter[str] = Counter()
        epoch_stroke_local: Counter[str] = Counter()
        epoch_stroke_curve: Counter[str] = Counter()
        epoch_started = time.perf_counter()
        for batch_index, (features, target_ids, teacher_batch, synthetic_mask) in enumerate(loader, start=1):
            features = features.to(device, non_blocking=amp)
            target_ids = target_ids.to(device, non_blocking=amp)
            teacher_batch = teacher_batch.to(device, non_blocking=amp)
            synthetic_mask = synthetic_mask.to(device, non_blocking=amp).bool()
            augmented, diag = _augment_with_engine(
                features, generator, args.augmentation, args.elastic_probability,
                rotation_degrees=args.rotation_degrees,
                axis_scale_jitter=args.axis_scale_jitter,
                shear_jitter=args.shear_jitter,
                elastic_control_displacement=args.elastic_control_displacement,
                wide_parameter_probability=args.wide_parameter_probability,
                wide_magnitude_multiplier=args.wide_magnitude_multiplier,
                wide_probability_by_parameter=args.wide_probability_by_parameter,
                wide_multiplier_by_parameter=args.wide_multiplier_by_parameter,
                stroke_local_probability=args.stroke_local_probability,
                stroke_local_rotation_degrees=args.stroke_local_rotation_degrees,
                stroke_local_scale_jitter=args.stroke_local_scale_jitter,
                stroke_local_translation_jitter=args.stroke_local_translation_jitter,
                stroke_generator=stroke_generator,
                stroke_curve_probability=args.stroke_curve_probability,
                stroke_curve_max_displacement=args.stroke_curve_max_displacement,
                stroke_curve_max_waves=args.stroke_curve_max_waves,
                curve_generator=curve_generator,
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                student_logits = model.math_head(model.encode(augmented))
                hard_each = F.cross_entropy(student_logits.float(), target_ids, reduction="none")
                hard_loss = hard_each.mean()
                real_mask = ~synthetic_mask
                if real_mask.any():
                    distill_each = F.kl_div(
                        F.log_softmax(student_logits[real_mask].float() / DISTILL_TEMPERATURE, dim=1),
                        F.softmax(teacher_batch[real_mask].float() / DISTILL_TEMPERATURE, dim=1),
                        reduction="none",
                    ).sum(dim=1) * (DISTILL_TEMPERATURE ** 2)
                    distill_loss = distill_each.mean()
                else:
                    distill_loss = hard_loss.new_zeros(())
                rank_terms = []
                rank_rows = 0
                if args.batch_hard_rank_margin:
                    rank_loss, rival_ids = _batch_hard_rank_margin_loss(
                        student_logits, target_ids, args.rank_margin,
                    )
                    rank_rows = len(target_ids)
                    epoch_hard_negative_rivals.update(
                        {index: count for index, count in enumerate(
                            torch.bincount(rival_ids.detach(), minlength=len(labels)).cpu().tolist()
                        ) if count}
                    )
                else:
                    for _target_label, _rival_label, target_id, rival_id in rank_margin_pairs:
                        pair_mask = target_ids == target_id
                        if pair_mask.any():
                            pair_margin = student_logits[pair_mask, target_id].float() - student_logits[pair_mask, rival_id].float()
                            rank_terms.append(F.relu(args.rank_margin - pair_margin).mean())
                            rank_rows += int(pair_mask.sum().item())
                    rank_loss = torch.stack(rank_terms).mean() if rank_terms else hard_loss.new_zeros(())
                loss = (
                    args.hard_loss_weight * hard_loss
                    + args.distill_loss_weight * distill_loss
                    + args.rank_margin_weight * rank_loss
                )
                loss, boundary_loss, boundary_indices = boundary_training_loss(
                    loss, model, boundary_features, boundary_teacher_logits, boundary_generator,
                    args.boundary_loss_weight, args.boundary_batch_size,
                    args.boundary_temperature, args.boundary_policy,
                    args.boundary_student_mode,
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}, batch {batch_index}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            hard_losses.append(float(hard_loss.detach().cpu()))
            distill_losses.append(float(distill_loss.detach().cpu()))
            rank_losses.append(float(rank_loss.detach().cpu()))
            boundary_losses.append(float(boundary_loss.detach().cpu()))
            epoch_boundary_views += len(boundary_indices)
            epoch_views += len(features)
            epoch_changed += diag["changed"]
            epoch_reverted += diag["reverted"]
            epoch_ranked_rows += rank_rows
            epoch_wide_selected.update(
                (diag.get("severity_mixture") or {}).get("wide_selected_rows_by_parameter", {})
            )
            local_diag = diag.get("stroke_local", {})
            for key in ("selected_rows", "accepted_rows", "reverted_rows", "changed_rows"):
                epoch_stroke_local[key] += int(local_diag.get(key, 0))
            curve_diag = diag.get("stroke_curve", {})
            for key in ("selected_rows", "accepted_rows", "reverted_rows", "changed_rows", "changed_strokes"):
                epoch_stroke_curve[key] += int(curve_diag.get(key, 0))
            if batch_index % args.log_every == 0 or batch_index == len(loader):
                print(json.dumps({
                    "event": "train_progress",
                    "epoch": epoch,
                    "batch": batch_index,
                    "batches": len(loader),
                    "hard_loss": float(np.mean(hard_losses)),
                    "distill_loss": float(np.mean(distill_losses)),
                    "rank_margin_loss": float(np.mean(rank_losses)),
                    "boundary_kl_loss": float(np.mean(boundary_losses)),
                    "boundary_replay_views": epoch_boundary_views,
                    "ranked_rows": epoch_ranked_rows,
                    "hard_negative_rivals_top10": [
                        {"label": labels[index], "count": count}
                        for index, count in epoch_hard_negative_rivals.most_common(10)
                    ],
                    "wide_selected_rows_by_parameter": dict(epoch_wide_selected),
                    "stroke_local_augmentation": dict(epoch_stroke_local),
                    "stroke_curve_augmentation": dict(epoch_stroke_curve),
                    "augmented_views": epoch_views,
                    "changed_views": epoch_changed,
                    "reverted_views": epoch_reverted,
                }, ensure_ascii=False), flush=True)
        total_augmented += epoch_views
        total_changed += epoch_changed
        total_reverted += epoch_reverted
        history.append({
            "epoch": epoch,
            "views": epoch_views,
            "changed_views": epoch_changed,
            "reverted_views": epoch_reverted,
            "hard_loss": float(np.mean(hard_losses)),
            "distill_loss": float(np.mean(distill_losses)),
            "rank_margin_loss": float(np.mean(rank_losses)),
            "boundary_kl_loss": float(np.mean(boundary_losses)),
            "boundary_replay_views": epoch_boundary_views,
            "ranked_rows": epoch_ranked_rows,
            "hard_negative_rivals_top10": [
                {"label": labels[index], "count": count}
                for index, count in epoch_hard_negative_rivals.most_common(10)
            ],
            "wide_selected_rows_by_parameter": dict(epoch_wide_selected),
            "stroke_local_augmentation": dict(epoch_stroke_local),
            "stroke_curve_augmentation": dict(epoch_stroke_curve),
            "seconds": time.perf_counter() - epoch_started,
        })

    model.eval()
    student_logits = _predict_logits(model, test["features"], device, args.eval_batch_size)
    student_metrics = _score(student_logits, test_y, test_sources, labels)
    student_checkpoint = args.work_dir / "student_checkpoint.pt"
    if student_checkpoint.exists():
        raise FileExistsError(student_checkpoint)
    report = {
        "schema": SCHEMA,
        "status": "completed_exploratory_comparison",
        "teacher": {
            "name": "current AIFlow 1.0e 372-class calibrated checkpoint",
            "path": str(args.checkpoint.resolve()),
            "sha256": _sha256(args.checkpoint),
            "metrics": baseline_metrics,
            "heldout_seconds": baseline_eval_seconds,
        },
        "student": {
            "name": f"{args.augmentation} augmentation + frozen-logit self-distillation challenger",
            "checkpoint": str(student_checkpoint.resolve()),
            "metrics": student_metrics,
            "input_contract": input_contract("uniform-time"),
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "loss": {
                "hard_label_cross_entropy": args.hard_loss_weight,
                "teacher_kl_on_real_rows": args.distill_loss_weight,
                "pairwise_rank_margin": args.rank_margin_weight,
                "rank_margin_mode": rank_margin_mode,
                "rank_margin_value": args.rank_margin,
                "rank_margin_pairs": [[pair[0], pair[1]] for pair in rank_margin_pairs],
                "temperature": DISTILL_TEMPERATURE,
                "synthetic_equal_kl": "disabled; hard label only",
                "boundary_replay": boundary_replay,
            },
            "optimizer": "AdamW",
            "class_balanced_views_per_class_per_epoch": args.samples_per_class_per_epoch,
            "augmented_views": total_augmented,
            "changed_views": total_changed,
            "reverted_views": total_reverted,
            "history": history,
        },
        "paired_delta": {
            "top1_pp": 100.0 * (student_metrics["overall"]["top1"] - baseline_metrics["overall"]["top1"]),
            "top5_pp": 100.0 * (student_metrics["overall"]["top5"] - baseline_metrics["overall"]["top5"]),
            "by_source_top1_pp": {
                name: 100.0 * (student_metrics["by_source"][name]["top1"] - baseline_metrics["by_source"][name]["top1"])
                for name in student_metrics["by_source"]
            },
            "interpretation": "single exploratory official-split comparison; not a fresh writer/device acceptance set",
        },
        "data_policy": manifest["input_policy"],
        "input_contract": input_contract("uniform-time"),
        "train_data": {
            "real_rows": manifest["real_train_rows"],
            "synthetic_equal_rows": SYNTHETIC_EQUAL_ROWS,
            "classes": len(labels),
            "synthetic_only_classes": ["="],
            "heldout_rows_by_class": {
                labels[class_id]: int(np.sum(test_y == class_id))
                for class_id in range(len(labels))
                if np.any(test_y == class_id)
            },
            "classes_without_heldout_support": [
                labels[class_id] for class_id in range(len(labels))
                if not np.any(test_y == class_id)
            ],
            "source_manifest_sha256": _sha256(data_dir / "prepared_manifest.json"),
        },
        "augmentation": {
            "algorithm": (
                "orientation-preserving affine jitter"
                if args.augmentation == "affine"
                else (
                    "orientation-preserving affine jitter + smooth boundary-pinned elastic field"
                    if args.augmentation == "affine_elastic"
                    else f"orientation-preserving affine jitter + elastic mixture (p={args.elastic_probability:.3f})"
                )
            ),
            "engine": args.augmentation,
            "parameters": _augmentation_parameters(args),
            "per_sample_limits": manifest["augmentation_audit"]["hard_limits"],
            "pretraining_volume_audit": str((args.report_dir / "full_domain_augmentation_audit.json").resolve()),
            "unit_validation": "every sampled training batch passed finite/range/spatial-limit gates; rejected transforms reverted to exact identity before optimizer step",
        },
        "crohme_rows": 0,
        "product_adopted": False,
        "training_seconds": time.perf_counter() - started,
        "teacher_logit_precompute_seconds": teacher_cache_seconds,
        "total_experiment_seconds": time.perf_counter() - experiment_started,
    }
    torch.save({
        "schema": "aiflow-augmentation-distilled-hwr-student/v1",
        "state_dict": model.state_dict(),
        "math_labels": labels,
        "auxiliary_labels": [],
        "input_mode": "uniform-time",
        "report": report,
    }, student_checkpoint)
    report_path = args.report_dir / "comparison.json"
    _write_json(report_path, report)
    print(json.dumps({
        "event": "training_complete",
        "baseline_top1": baseline_metrics["overall"]["top1"],
        "student_top1": student_metrics["overall"]["top1"],
        "baseline_top5": baseline_metrics["overall"]["top5"],
        "student_top5": student_metrics["overall"]["top5"],
        "checkpoint": str(student_checkpoint.resolve()),
        "report": str(report_path.resolve()),
        "product_adopted": False,
    }, ensure_ascii=False), flush=True)
    return 0


def _finalize_checkpoint(args) -> int:
    source_path = args.work_dir / "student_checkpoint.pt"
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    destination = args.work_dir / "student_checkpoint_compatible.pt"
    report_path = args.report_dir / "comparison_compatible.json"
    if destination.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite compatibility checkpoint/report")
    payload = torch.load(source_path, map_location="cpu", weights_only=False)
    labels = list(payload.get("math_labels", []))
    if len(labels) != 372 or payload.get("auxiliary_labels"):
        raise ValueError("student checkpoint is not a unified 372-class model")
    report = payload.get("report")
    if not isinstance(report, dict):
        raise ValueError("student checkpoint has no experiment report")
    _assert_comparison_sources_eligible(report.get("data_policy", {}), event="finalize_compatible_checkpoint")
    report["input_contract"] = input_contract("uniform-time")
    report["inference_compatibility"] = "validated by evaluate_48hz_prefix_v1._load_model"
    payload["report"] = report
    torch.save(payload, destination)
    report = json.loads(json.dumps(report))
    report["student"]["checkpoint"] = str(destination.resolve())
    report["student"]["checkpoint_sha256"] = _sha256(destination)
    _write_json(report_path, report)
    print(json.dumps({
        "event": "checkpoint_finalized",
        "checkpoint": str(destination.resolve()),
        "checkpoint_sha256": _sha256(destination),
        "report": str(report_path.resolve()),
    }, ensure_ascii=False), flush=True)
    return 0


def _analyze_pair(args) -> int:
    data_dir = args.data_dir or args.work_dir
    manifest_path = data_dir / "prepared_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _assert_comparison_sources_eligible(
        manifest.get("input_policy", {}),
        event="posthoc_pair_analysis",
        source_split_rows=manifest.get("source_audit", {}).get("source_split_rows", {}),
    )
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    teacher, labels, _ = _load_teacher(args.checkpoint, device)
    student_path = args.work_dir / "student_checkpoint_compatible.pt"
    if not student_path.is_file():
        raise FileNotFoundError(f"run --mode finalize first: {student_path}")
    student, student_labels, _ = _load_teacher(student_path, device)
    if labels != student_labels:
        raise ValueError("teacher/student class order differs")
    cache = _load_cache(data_dir, "test")
    features = cache["features"]
    targets = np.asarray(cache["labels"], dtype=np.int64)
    sources = np.asarray(cache["sources"], dtype=np.int8)
    teacher_logits = _predict_logits(teacher, features, device, args.eval_batch_size)
    student_logits = _predict_logits(student, features, device, args.eval_batch_size)
    teacher_top1 = np.argmax(teacher_logits, axis=1)
    student_top1 = np.argmax(student_logits, axis=1)
    teacher_top5 = np.argpartition(-teacher_logits, kth=4, axis=1)[:, :5]
    student_top5 = np.argpartition(-student_logits, kth=4, axis=1)[:, :5]
    teacher_ranks = np.argsort(np.argsort(-teacher_logits, axis=1), axis=1) + 1
    student_ranks = np.argsort(np.argsort(-student_logits, axis=1), axis=1) + 1

    def rank_histogram(values: np.ndarray) -> dict[str, int]:
        return {
            "rank_1": int(np.sum(values == 1)),
            "rank_2": int(np.sum(values == 2)),
            "rank_3": int(np.sum(values == 3)),
            "rank_4": int(np.sum(values == 4)),
            "rank_5": int(np.sum(values == 5)),
            "rank_6_plus": int(np.sum(values > 5)),
        }

    per_label = {}
    for class_id, label in enumerate(labels):
        indices = np.flatnonzero(targets == class_id)
        before1 = teacher_top1[indices] == class_id
        after1 = student_top1[indices] == class_id
        before5 = np.asarray([class_id in teacher_top5[index] for index in indices], dtype=bool)
        after5 = np.asarray([class_id in student_top5[index] for index in indices], dtype=bool)
        regressed_rows = indices[before1 & ~after1]
        confusion_destinations = Counter(labels[int(student_top1[index])] for index in regressed_rows)
        per_label[label] = {
            "family": _family(label),
            "heldout_rows": int(len(indices)),
            "teacher_top1_hits": int(before1.sum()),
            "student_top1_hits": int(after1.sum()),
            "top1_recovered": int((~before1 & after1).sum()),
            "top1_regressed": int((before1 & ~after1).sum()),
            "teacher_top5_hits": int(before5.sum()),
            "student_top5_hits": int(after5.sum()),
            "top5_recovered": int((~before5 & after5).sum()),
            "top5_regressed": int((before5 & ~after5).sum()),
            "teacher_target_rank_histogram": rank_histogram(teacher_ranks[indices, class_id]),
            "student_target_rank_histogram": rank_histogram(student_ranks[indices, class_id]),
            "top1_regression_destinations": [
                {"label": target_label, "rows": int(rows)}
                for target_label, rows in confusion_destinations.most_common(8)
            ],
        }
    by_source = {}
    for source_name, source_id in (("hwrt", SOURCE_IDS["hwrt"]), ("uji", SOURCE_IDS["uji"])):
        indices = np.flatnonzero(sources == source_id)
        before1 = teacher_top1[indices] == targets[indices]
        after1 = student_top1[indices] == targets[indices]
        before5 = np.asarray([targets[index] in teacher_top5[index] for index in indices], dtype=bool)
        after5 = np.asarray([targets[index] in student_top5[index] for index in indices], dtype=bool)
        by_source[source_name] = {
            "rows": int(len(indices)),
            "teacher_top1_hits": int(before1.sum()),
            "student_top1_hits": int(after1.sum()),
            "top1_recovered": int((~before1 & after1).sum()),
            "top1_regressed": int((before1 & ~after1).sum()),
            "teacher_top5_hits": int(before5.sum()),
            "student_top5_hits": int(after5.sum()),
            "top5_recovered": int((~before5 & after5).sum()),
            "top5_regressed": int((before5 & ~after5).sum()),
        }
    regressions = [
        {"label": label, **metrics}
        for label, metrics in per_label.items()
        if metrics["heldout_rows"] >= 8 and metrics["student_top1_hits"] < metrics["teacher_top1_hits"]
    ]
    regressions.sort(key=lambda row: (
        row["student_top1_hits"] - row["teacher_top1_hits"],
        -row["heldout_rows"],
        row["label"],
    ))
    recoveries = [
        {"label": label, **metrics}
        for label, metrics in per_label.items()
        if metrics["heldout_rows"] >= 8 and metrics["student_top1_hits"] > metrics["teacher_top1_hits"]
    ]
    recoveries.sort(key=lambda row: (
        -(row["student_top1_hits"] - row["teacher_top1_hits"]),
        -row["heldout_rows"],
        row["label"],
    ))
    report_path = args.report_dir / "paired_class_microscope_detailed.json"
    report = {
        "schema": SCHEMA,
        "status": "completed_paired_heldout_microscope",
        "teacher_checkpoint_sha256": _sha256(args.checkpoint),
        "student_checkpoint_sha256": _sha256(student_path),
        "test_source_policy": "HWRT curated test split (source-group split, not proven writer-disjoint) plus UJI official writer-heldout test split",
        "rows": len(targets),
        "crohme_rows": 0,
        "product_adopted": False,
        "by_source": by_source,
        "top1_total": {
            "teacher_hits": int((teacher_top1 == targets).sum()),
            "student_hits": int((student_top1 == targets).sum()),
            "recovered": int(((teacher_top1 != targets) & (student_top1 == targets)).sum()),
            "regressed": int(((teacher_top1 == targets) & (student_top1 != targets)).sum()),
        },
        "top5_total": {
            "teacher_hits": int(sum(targets[i] in teacher_top5[i] for i in range(len(targets)))),
            "student_hits": int(sum(targets[i] in student_top5[i] for i in range(len(targets)))),
            "recovered": int(sum(targets[i] not in teacher_top5[i] and targets[i] in student_top5[i] for i in range(len(targets)))),
            "regressed": int(sum(targets[i] in teacher_top5[i] and targets[i] not in student_top5[i] for i in range(len(targets)))),
        },
        "by_label": per_label,
        "classes_with_no_heldout_rows": [label for label, row in per_label.items() if row["heldout_rows"] == 0],
        "top1_regressions_min_8_rows": regressions[:30],
        "top1_recoveries_min_8_rows": recoveries[:30],
        "interpretation_limit": "paired symbol-level external-split diagnostic only; no formula-exact, fresh-writer, or fresh-device acceptance claim",
    }
    _write_json(report_path, report)
    print(json.dumps({
        "event": "paired_microscope_complete",
        "rows": len(targets),
        "top1": report["top1_total"],
        "top5": report["top5_total"],
        "sources": by_source,
        "classes_without_test": len(report["classes_with_no_heldout_rows"]),
        "report": str(report_path.resolve()),
    }, ensure_ascii=False), flush=True)
    return 0


def _self_test_stroke_curve() -> int:
    """Exercise topology, boundary, determinism, and disabled-path invariants without external data."""
    features = torch.zeros((2, POINTS, len(CHANNELS)), dtype=torch.float32)
    local_t = torch.linspace(0.0, 1.0, 64)
    features[:, :64, 0] = 0.12 + 0.76 * local_t
    features[0, :64, 1] = 0.22 + 0.10 * torch.sin(math.pi * local_t)
    features[1, :64, 1] = 0.25 + 0.08 * torch.sin(math.pi * local_t)
    features[:, 64:, 0] = 0.10 + 0.80 * local_t
    features[0, 64:, 1] = 0.66 + 0.08 * torch.sin(math.pi * local_t)
    features[1, 64:, 1] = 0.70 + 0.06 * torch.sin(math.pi * local_t)
    features[:, 0, 3] = 1.0
    features[:, 64, 3] = 1.0
    features[:, :, 2] = 1.0 / (POINTS - 1)
    features[:, 0, 2] = 0.0
    features[:, :, 4] = 1.0

    seed = 20261002
    first, diagnostics = _augment_stroke_curve(
        features, torch.Generator().manual_seed(seed),
        probability=1.0, max_displacement=0.025, max_waves=1,
    )
    second, _ = _augment_stroke_curve(
        features, torch.Generator().manual_seed(seed),
        probability=1.0, max_displacement=0.025, max_waves=1,
    )
    vector_equivalent, _ = _augment_stroke_curve(
        features, torch.Generator().manual_seed(seed),
        probability=1.0, max_displacement=torch.full((len(features),), 0.025), max_waves=1,
    )
    probability_vector_equivalent, _ = _augment_stroke_curve(
        features, torch.Generator().manual_seed(seed),
        probability=torch.ones(len(features)), max_displacement=0.025, max_waves=1,
    )
    waves_vector_equivalent, _ = _augment_stroke_curve(
        features, torch.Generator().manual_seed(seed),
        probability=1.0, max_displacement=0.025,
        max_waves=torch.ones(len(features), dtype=torch.long),
    )
    if not torch.equal(first, second):
        raise AssertionError("stroke-curve augmentation is not deterministic for a fixed seed")
    if not torch.equal(first, vector_equivalent):
        raise AssertionError("uniform per-row displacement changed the legacy scalar augmentation output")
    if not torch.equal(first, probability_vector_equivalent):
        raise AssertionError("uniform per-row probability changed the scalar augmentation output")
    if not torch.equal(first, waves_vector_equivalent):
        raise AssertionError("uniform per-row max-waves profile changed the scalar augmentation output")
    if not torch.equal(first[:, :, 2:], features[:, :, 2:]):
        raise AssertionError("stroke-curve self-test changed a non-spatial channel")
    if not torch.equal(first[:, :, 3] > 0.5, features[:, :, 3] > 0.5):
        raise AssertionError("stroke-curve self-test changed stroke starts")
    endpoints = torch.tensor([0, 63, 64, 127], dtype=torch.long)
    if not torch.equal(first[:, endpoints, :2], features[:, endpoints, :2]):
        raise AssertionError("stroke-curve self-test moved a stroke endpoint")
    if not torch.isfinite(first).all() or (first[:, :, :2] < 0.0).any() or (first[:, :, :2] > 1.0).any():
        raise AssertionError("stroke-curve self-test produced an invalid coordinate")
    if diagnostics["accepted_rows"] != len(features) or diagnostics["changed_rows"] != len(features):
        raise AssertionError(f"stroke-curve self-test did not change every designed glyph: {diagnostics}")

    per_row_displacement = torch.tensor([0.0, 0.025], dtype=features.dtype)
    family_view, family_diagnostics = _augment_stroke_curve(
        features, torch.Generator().manual_seed(seed),
        probability=1.0, max_displacement=per_row_displacement, max_waves=1,
    )
    if not torch.equal(family_view[0], features[0]) or not torch.equal(family_view[:, :, 2:], features[:, :, 2:]):
        raise AssertionError("family-specific stroke-curve changed a disabled row or non-spatial channel")
    if family_diagnostics["selected_rows"] != 1 or family_diagnostics["changed_rows"] != 1:
        raise AssertionError(f"family-specific stroke-curve did not honor per-row amplitudes: {family_diagnostics}")

    probability_by_row = torch.tensor([1.0, 0.0], dtype=features.dtype)
    probability_view, probability_diagnostics = _augment_stroke_curve(
        features, torch.Generator().manual_seed(seed),
        probability=probability_by_row, max_displacement=0.025, max_waves=1,
        family_by_row=["digits", "greek"],
    )
    if not torch.equal(probability_view[0], first[0]) or not torch.equal(probability_view[1], features[1]):
        raise AssertionError("per-row stroke-curve probability did not select exactly its enabled family row")
    if (
        probability_diagnostics["rows_by_family"].get("digits", {}).get("selected_rows") != 1
        or probability_diagnostics["rows_by_family"].get("greek", {}).get("selected_rows") != 0
    ):
        raise AssertionError(f"per-family stroke-curve selection diagnostics are incorrect: {probability_diagnostics}")

    max_waves_by_row = torch.tensor([1, 3], dtype=torch.long)
    selection_seed = seed + 20
    mixed_waves, mixed_waves_diagnostics = _augment_stroke_curve(
        features, torch.Generator().manual_seed(seed),
        probability=torch.ones(len(features)), max_displacement=0.025,
        max_waves=max_waves_by_row, family_by_row=["digits", "greek"],
        selection_generator=torch.Generator().manual_seed(selection_seed),
    )
    repeated_mixed_waves, _ = _augment_stroke_curve(
        features, torch.Generator().manual_seed(seed),
        probability=torch.ones(len(features)), max_displacement=0.025,
        max_waves=max_waves_by_row, family_by_row=["digits", "greek"],
        selection_generator=torch.Generator().manual_seed(selection_seed),
    )
    uniform_waves, uniform_waves_diagnostics = _augment_stroke_curve(
        features, torch.Generator().manual_seed(seed),
        probability=torch.ones(len(features)), max_displacement=0.025,
        max_waves=2, family_by_row=["digits", "greek"],
        selection_generator=torch.Generator().manual_seed(selection_seed),
    )
    if not torch.equal(mixed_waves, repeated_mixed_waves):
        raise AssertionError("family-specific max-waves profile is not deterministic for fixed seeds")
    if mixed_waves_diagnostics["selected_rows"] != uniform_waves_diagnostics["selected_rows"]:
        raise AssertionError("separate selection generator changed selected rows across wave profiles")
    if (
        mixed_waves_diagnostics["max_waves_mean_requested"] != 2.0
        or mixed_waves_diagnostics["rows_by_family"]["digits"]["max_waves_mean_requested"] != 1.0
        or mixed_waves_diagnostics["rows_by_family"]["greek"]["max_waves_mean_requested"] != 3.0
    ):
        raise AssertionError(f"family-specific max-waves profile diagnostics are incorrect: {mixed_waves_diagnostics}")

    try:
        _parse_family_displacement_overrides(["unknown_family=0.02"])
    except ValueError:
        pass
    else:
        raise AssertionError("family-specific stroke-curve accepted an unknown family")

    for overrides, expected_fragment in (
        (["unknown_family=0.5"], "one of"),
        (["digits=0.5", "digits=0.7"], "repeats family"),
        (["digits=1.1"], "in [0, 1]"),
    ):
        try:
            _parse_family_probability_overrides(overrides)
        except ValueError as exc:
            if expected_fragment not in str(exc):
                raise AssertionError(f"family probability validation returned an unexpected error: {exc}") from exc
        else:
            raise AssertionError(f"family probability parser accepted invalid overrides: {overrides}")

    for overrides, expected_fragment in (
        (["unknown_family=2"], "one of"),
        (["digits=1", "digits=3"], "repeats family"),
        (["digits=1.5"], "non-integer"),
        (["digits=4"], "in [1, 3]"),
    ):
        try:
            _parse_family_max_waves_overrides(overrides)
        except ValueError as exc:
            if expected_fragment not in str(exc):
                raise AssertionError(f"family max-waves validation returned an unexpected error: {exc}") from exc
        else:
            raise AssertionError(f"family max-waves parser accepted invalid overrides: {overrides}")

    disabled, disabled_diagnostics = _augment_stroke_curve(
        features, torch.Generator().manual_seed(seed),
        probability=0.0, max_displacement=0.025, max_waves=1,
    )
    if not torch.equal(disabled, features) or disabled_diagnostics["changed_rows"] != 0:
        raise AssertionError("zero-probability stroke-curve path is not identity")

    generator_a = torch.Generator().manual_seed(seed + 1)
    generator_b = torch.Generator().manual_seed(seed + 1)
    stroke_a = torch.Generator().manual_seed(seed + 2)
    stroke_b = torch.Generator().manual_seed(seed + 2)
    baseline, _ = _augment_with_engine(
        features, generator_a, "affine", stroke_local_probability=0.5,
        stroke_generator=stroke_a,
    )
    router_off, _ = _augment_with_engine(
        features, generator_b, "affine", stroke_local_probability=0.5,
        stroke_generator=stroke_b, stroke_curve_probability=0.0,
        curve_generator=torch.Generator().manual_seed(seed + 3),
    )
    if not torch.equal(baseline, router_off):
        raise AssertionError("disabled stroke-curve option changed the pre-existing augmentation output")

    print(json.dumps({
        "event": "stroke_curve_self_test",
        "status": "pass",
        "batch_rows": len(features),
        "stroke_endpoints_fixed": True,
        "nonspatial_channels_identical": True,
        "boundary_coordinates_valid": True,
        "fixed_seed_deterministic": True,
        "uniform_family_profile_matches_scalar": True,
        "uniform_probability_profile_matches_scalar": True,
        "disabled_path_identity": True,
        "router_off_byte_identical": True,
        "family_specific_displacement_row_mask": True,
        "family_specific_probability_row_mask": True,
        "uniform_max_waves_profile_matches_scalar": True,
        "family_specific_max_waves_profile": True,
        "max_waves_selection_stream_isolated": True,
        "per_family_selection_diagnostics": True,
        "family_override_validation": True,
        "diagnostics": diagnostics,
        "family_diagnostics": family_diagnostics,
        "mixed_waves_diagnostics": mixed_waves_diagnostics,
        "crohme_rows": 0,
    }, ensure_ascii=False), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "audit-volume", "train", "finalize", "analyze", "self-test"), required=True)
    parser.add_argument("--augmentation", choices=("affine", "affine_elastic", "affine_elastic_mix"), default="affine")
    parser.add_argument("--elastic-probability", type=float, default=0.25)
    parser.add_argument("--wide-parameter-probability", type=float, default=0.0, help="independent per-parameter probability of sampling the wider augmentation bound")
    parser.add_argument("--wide-magnitude-multiplier", type=float, default=1.5, help="multiplier applied to each independently selected affine/elastic bound")
    parser.add_argument("--wide-probability", action="append", default=[], metavar="PARAMETER=PROBABILITY", help="override one transform's wide-magnitude probability; repeat for rotation, axis_scale, shear, or elastic")
    parser.add_argument("--wide-multiplier", action="append", default=[], metavar="PARAMETER=MULTIPLIER", help="override one transform's wide-magnitude multiplier; repeat for rotation, axis_scale, shear, or elastic")
    parser.add_argument("--stroke-local-probability", type=float, default=0.0, help="probability of adding small independent affine variation per stroke")
    parser.add_argument("--stroke-local-rotation-degrees", type=float, default=1.0)
    parser.add_argument("--stroke-local-scale-jitter", type=float, default=0.02)
    parser.add_argument("--stroke-local-translation-jitter", type=float, default=0.003)
    parser.add_argument("--stroke-curve-probability", type=float, default=0.0, help="probability of adding a smooth normal bend to each selected glyph")
    parser.add_argument("--stroke-curve-max-displacement", type=float, default=0.01)
    parser.add_argument(
        "--stroke-curve-family-displacement", action="append", default=[], metavar="FAMILY=VALUE",
        help="audit-only smooth-bend amplitude override by mathematical symbol family; repeatable",
    )
    parser.add_argument(
        "--stroke-curve-family-probability", action="append", default=[], metavar="FAMILY=VALUE",
        help="audit-only smooth-bend selection probability override by mathematical symbol family; repeatable",
    )
    parser.add_argument(
        "--stroke-curve-family-max-waves", action="append", default=[], metavar="FAMILY=1|2|3",
        help="audit-only smooth-bend wave-count ceiling override by mathematical symbol family; repeatable",
    )
    parser.add_argument("--stroke-curve-max-waves", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--rotation-degrees", type=float, default=2.5)
    parser.add_argument("--axis-scale-jitter", type=float, default=0.055)
    parser.add_argument("--shear-jitter", type=float, default=0.025)
    parser.add_argument("--elastic-control-displacement", type=float, default=0.012)
    parser.add_argument("--canonical-root", type=Path, default=DEFAULT_CANONICAL_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--data-dir", type=Path, default=None, help="read the audited train/test cache here; defaults to --work-dir")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--samples-per-class-per-epoch", type=int, default=SAMPLES_PER_CLASS_PER_EPOCH)
    parser.add_argument("--learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--hard-loss-weight", type=float, default=0.70)
    parser.add_argument("--distill-loss-weight", type=float, default=0.30)
    parser.add_argument("--rank-margin-weight", type=float, default=0.0)
    parser.add_argument("--rank-margin", type=float, default=1.0)
    parser.add_argument("--rank-margin-pair", action="append", default=[], help="pairwise target-vs-rival class constraint, TARGET:RIVAL (repeatable)")
    parser.add_argument("--batch-hard-rank-margin", action="store_true", help="rank every training row against its current highest-scoring non-target class")
    parser.add_argument("--boundary-candidate-report", type=Path, default=None, help="verified training-only boundary candidate report; opt-in replay")
    parser.add_argument("--boundary-loss-weight", type=float, default=0.0, help="total boundary KL share; remaining share multiplies the original loss")
    parser.add_argument("--boundary-policy", choices=("full", "top5"), default="full")
    parser.add_argument("--boundary-temperature", type=float, default=1.0)
    parser.add_argument("--boundary-student-mode", choices=("current", "deterministic"), default="deterministic", help="deterministic disables dropout only for replay and preserves autograd; current is the stochastic ablation")
    parser.add_argument("--boundary-batch-size", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--audit-views", type=int, default=3)
    args = parser.parse_args()
    if not np.isfinite(args.boundary_loss_weight) or not 0.0 <= args.boundary_loss_weight < 1.0:
        parser.error("--boundary-loss-weight must be finite in [0,1)")
    if not np.isfinite(args.boundary_temperature) or args.boundary_temperature <= 0.0 or args.boundary_batch_size < 1:
        parser.error("boundary temperature and batch size must be positive and finite")
    if args.boundary_loss_weight > 0.0 and (args.boundary_candidate_report is None or args.mode != "train"):
        parser.error("enabled boundary replay requires --mode train and --boundary-candidate-report")
    if args.epochs < 1 or min(args.batch_size, args.eval_batch_size, args.samples_per_class_per_epoch, args.log_every, args.audit_views) < 1:
        parser.error("epochs and batch/step settings must be positive")
    if args.learning_rate <= 0.0:
        parser.error("--learning-rate must be positive")
    if not 0.0 <= args.elastic_probability <= 1.0:
        parser.error("--elastic-probability must be in [0, 1]")
    if not 0.0 <= args.wide_parameter_probability <= 1.0:
        parser.error("--wide-parameter-probability must be in [0, 1]")
    if not 1.0 <= args.wide_magnitude_multiplier <= 3.0:
        parser.error("--wide-magnitude-multiplier must be in [1, 3]")
    try:
        args.wide_probability_by_parameter, args.wide_multiplier_by_parameter = _resolve_wide_parameter_settings(
            args.wide_parameter_probability,
            args.wide_magnitude_multiplier,
            args.wide_probability,
            args.wide_multiplier,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if not 0.0 <= args.stroke_local_probability <= 1.0:
        parser.error("--stroke-local-probability must be in [0, 1]")
    if not 0.0 <= args.stroke_local_rotation_degrees <= 5.0:
        parser.error("--stroke-local-rotation-degrees must be in [0, 5]")
    if not 0.0 <= args.stroke_local_scale_jitter <= 0.15:
        parser.error("--stroke-local-scale-jitter must be in [0, 0.15]")
    if not 0.0 <= args.stroke_local_translation_jitter <= 0.05:
        parser.error("--stroke-local-translation-jitter must be in [0, 0.05]")
    if not 0.0 <= args.stroke_curve_probability <= 1.0:
        parser.error("--stroke-curve-probability must be in [0, 1]")
    if not 0.0 <= args.stroke_curve_max_displacement <= 0.03:
        parser.error("--stroke-curve-max-displacement must be in [0, 0.03]")
    try:
        args.stroke_curve_displacement_by_family = _parse_family_displacement_overrides(
            args.stroke_curve_family_displacement,
        )
        args.stroke_curve_probability_by_family = _parse_family_probability_overrides(
            args.stroke_curve_family_probability,
        )
        args.stroke_curve_max_waves_by_family = _parse_family_max_waves_overrides(
            args.stroke_curve_family_max_waves,
        )
    except ValueError as exc:
        parser.error(str(exc))
    has_family_specific_curve_settings = any((
        args.stroke_curve_displacement_by_family,
        args.stroke_curve_probability_by_family,
        args.stroke_curve_max_waves_by_family,
    ))
    if has_family_specific_curve_settings and args.mode != "audit-volume":
        parser.error("family-specific stroke-curve settings are only supported in --mode audit-volume")
    effective_curve_probabilities = {
        family: args.stroke_curve_probability_by_family.get(family, args.stroke_curve_probability)
        for family in SYMBOL_FAMILY_NAMES
    }
    effective_curve_displacements = {
        family: args.stroke_curve_displacement_by_family.get(family, args.stroke_curve_max_displacement)
        for family in SYMBOL_FAMILY_NAMES
    }
    if has_family_specific_curve_settings and not any(
        effective_curve_probabilities[family] > 0.0 and effective_curve_displacements[family] > 0.0
        for family in SYMBOL_FAMILY_NAMES
    ):
        parser.error("family-specific stroke-curve settings must activate at least one family")
    if not 0.0 <= args.rotation_degrees <= 12.0:
        parser.error("--rotation-degrees must be in [0, 12]")
    if not 0.0 <= args.axis_scale_jitter <= 0.25:
        parser.error("--axis-scale-jitter must be in [0, 0.25]")
    if not 0.0 <= args.shear_jitter <= 0.15:
        parser.error("--shear-jitter must be in [0, 0.15]")
    if not 0.0 <= args.elastic_control_displacement <= 0.05:
        parser.error("--elastic-control-displacement must be in [0, 0.05]")
    if min(args.hard_loss_weight, args.distill_loss_weight, args.rank_margin_weight) < 0.0 or args.rank_margin < 0.0:
        parser.error("loss weights and rank margin must be non-negative")
    if args.mode == "self-test":
        return _self_test_stroke_curve()
    if args.mode == "prepare":
        return prepare(args)
    if args.mode == "audit-volume":
        return _audit_multi_view_augmentation(args)
    if args.mode == "finalize":
        return _finalize_checkpoint(args)
    if args.mode == "analyze":
        return _analyze_pair(args)
    return _train(args)


if __name__ == "__main__":
    raise SystemExit(main())
