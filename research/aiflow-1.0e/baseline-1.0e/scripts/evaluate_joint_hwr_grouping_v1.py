#!/usr/bin/env python3
"""Evaluate geometry plus HWR confidence for raw-stroke grouping.

Each outer writer fold uses the HWR head trained without that writer and a
grouping selector fitted without that writer.  The exact-cover bias is fixed at
zero, so no held-writer threshold selection is hidden in the result.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import joblib
import numpy as np
import torch

from build_normalized_ink_v1 import SourceSample, _canonicalize
from character_tensor_v1 import _digest
from evaluate_48hz_prefix_v1 import (
    DEFAULT_BASE, INPUT_MODE, _load_loo_heads, _load_model, _prefix_tensor,
    _sha256, resample_direct_48hz,
)
from stroke_grouping_v1 import FEATURE_NAMES
from train_character_classifier_v1 import apply_input_mode
from train_project_owned_grouping_v1 import (
    MODEL_VERSION as GEOMETRY_MODEL_VERSION, Sample, _fit, _metrics, _predict,
    _samples,
)


SCHEMA = "aiflow-joint-hwr-stroke-grouping/v1"
MODEL_VERSION = "aiflow-joint-hwr-stroke-grouping-1.0-r1-shadow"
FULL_LATTICE_CONFIG = {
    "temporal_window": 6,
    "spatial_neighbors": 4,
    "max_long_group_width_fraction": 1.0,
}
HWR_FEATURE_NAMES = (
    "hwr_top1", "hwr_margin", "hwr_top5_mass", "hwr_entropy",
    "hwr_component_top1_gain", "hwr_component_entropy_gain",
    "hwr_label_digit", "hwr_label_operator", "hwr_label_alpha", "hwr_label_delimiter",
)
FIXED_GROUP_BIAS = 0.0


def _candidate_tensor(sample: Sample, group: list[int]) -> np.ndarray:
    source = [
        [
            (float(point["x"]), float(point["y"]), float(point["t_ms"]))
            for point in sample.strokes[index]["points"]
        ]
        for index in group
    ]
    record = _canonicalize(SourceSample(
        "project_owned_group_candidate", f"{sample.sample_id}:{','.join(map(str, group))}",
        "?", "grouping_candidate", "candidate_only", source,
    ))
    return _prefix_tensor(resample_direct_48hz(record))


def _candidate_embeddings(samples: list[Sample], model, device: torch.device) -> tuple[torch.Tensor, dict[str, slice]]:
    tensors = []; slices = {}; offset = 0
    for sample in samples:
        rows = [_candidate_tensor(sample, row["source_indices"]) for row in sample.candidates]
        tensors.extend(rows)
        slices[sample.sample_id] = slice(offset, offset + len(rows)); offset += len(rows)
    values = np.stack(tensors).astype(np.float32, copy=False)
    chunks = []
    for start in range(0, len(values), 512):
        batch = torch.from_numpy(apply_input_mode(values[start:start + 512], INPUT_MODE)).to(device)
        with torch.inference_mode():
            chunks.append(model.encode(batch).cpu())
    return torch.cat(chunks), slices


def _probabilities(head, embeddings: torch.Tensor, device: torch.device) -> np.ndarray:
    rows = []
    for start in range(0, len(embeddings), 1024):
        with torch.inference_mode():
            rows.append(head(embeddings[start:start + 1024].to(device)).softmax(dim=1).cpu())
    return torch.cat(rows).numpy()


def _hwr_features(sample: Sample, probability: np.ndarray, labels: list[str]) -> np.ndarray:
    order = np.argsort(-probability, axis=1)
    top = np.take_along_axis(probability, order[:, :5], axis=1)
    entropy = -(probability * np.log(np.clip(probability, 1e-12, 1.0))).sum(axis=1) / math.log(probability.shape[1])
    singleton = {
        int(row["source_indices"][0]): index
        for index, row in enumerate(sample.candidates)
        if len(row["source_indices"]) == 1
    }
    operators = {"+", "-", "=", "/", r"\times", r"\div", r"\neq", "<", ">"}
    delimiters = {"(", ")", "[", "]", "{", "}", "|"}
    rows = []
    for index, candidate in enumerate(sample.candidates):
        components = [singleton[int(value)] for value in candidate["source_indices"]]
        label = labels[int(order[index, 0])]
        rows.append([
            float(top[index, 0]), float(top[index, 0] - top[index, 1]), float(top[index].sum()),
            float(entropy[index]),
            float(top[index, 0] - np.mean(top[components, 0])) if len(components) > 1 else 0.0,
            float(np.mean(entropy[components]) - entropy[index]) if len(components) > 1 else 0.0,
            float(label.isdigit()), float(label in operators),
            float(len(label) == 1 and label.isalpha()), float(label in delimiters),
        ])
    values = np.asarray(rows, dtype=np.float32)
    if values.shape != (len(sample.candidates), len(HWR_FEATURE_NAMES)) or not np.isfinite(values).all():
        raise AssertionError("invalid HWR grouping features")
    return values


def _augment(
    samples: list[Sample], all_probability: np.ndarray, slices: dict[str, slice], labels: list[str],
) -> list[Sample]:
    return [
        replace(sample, features=np.concatenate((
            sample.features, _hwr_features(sample, all_probability[slices[sample.sample_id]], labels),
        ), axis=1))
        for sample in samples
    ]


def _failed(samples: list[Sample], predictions: dict[str, tuple[frozenset[int], ...]]) -> list[dict]:
    return [
        {
            "sample_id": sample.sample_id, "writer": sample.writer,
            "truth": [sorted(group) for group in sample.truth],
            "predicted": [sorted(group) for group in predictions[sample.sample_id]],
        }
        for sample in samples if set(sample.truth) != set(predictions[sample.sample_id])
    ]


def _outer_writer_evaluation(
    samples: list[Sample], embeddings: torch.Tensor, slices: dict[str, slice], labels: list[str],
    heads: dict[str, torch.nn.Linear], device: torch.device,
) -> dict:
    geometry_predictions = {}; joint_predictions = {}; folds = []
    for writer in sorted({sample.writer for sample in samples}):
        writer_group = _digest(["project_owned_writer", writer])[:24]
        if writer_group not in heads:
            raise ValueError(f"held-writer HWR head missing: {writer}")
        probability = _probabilities(heads[writer_group], embeddings, device)
        joint_samples = _augment(samples, probability, slices, labels)
        geometry_train = [sample for sample in samples if sample.writer != writer]
        geometry_test = [sample for sample in samples if sample.writer == writer]
        joint_train = [sample for sample in joint_samples if sample.writer != writer]
        joint_test = [sample for sample in joint_samples if sample.writer == writer]
        geometry_model = _fit(geometry_train)
        joint_model = _fit(joint_train)
        geometry_scores = {
            sample.sample_id: np.log(np.clip(geometry_model.predict_proba(sample.features)[:, 1], 1e-6, 1 - 1e-6)
                                     / np.clip(1 - geometry_model.predict_proba(sample.features)[:, 1], 1e-6, 1))
            for sample in geometry_test
        }
        joint_scores = {
            sample.sample_id: np.log(np.clip(joint_model.predict_proba(sample.features)[:, 1], 1e-6, 1 - 1e-6)
                                     / np.clip(1 - joint_model.predict_proba(sample.features)[:, 1], 1e-6, 1))
            for sample in joint_test
        }
        geometry_fold = _predict(geometry_test, geometry_scores, FIXED_GROUP_BIAS)
        joint_fold = _predict(joint_test, joint_scores, FIXED_GROUP_BIAS)
        geometry_predictions.update(geometry_fold); joint_predictions.update(joint_fold)
        folds.append({
            "held_out_writer": writer, "train_formulas": len(geometry_train), "test_formulas": len(geometry_test),
            "geometry": _metrics(geometry_test, geometry_fold),
            "joint_hwr": _metrics(joint_test, joint_fold),
        })
    return {
        "protocol": "outer writer absent from HWR-head and grouping-model fitting; fixed group_bias=0",
        "geometry": _metrics(samples, geometry_predictions),
        "joint_hwr": _metrics(samples, joint_predictions),
        "folds": folds,
        "geometry_failed": _failed(samples, geometry_predictions),
        "joint_hwr_failed": _failed(samples, joint_predictions),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--loo-heads", type=Path, required=True)
    parser.add_argument("--product-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or output.drive.upper() != "D:":
        parser.error("output must be a new directory on D:")
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    if device_name == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    device = torch.device(device_name)
    samples, dataset = _samples(args.dataset_root.resolve(), lattice_config=FULL_LATTICE_CONFIG)
    base_path = args.base_checkpoint.resolve(); loo_path = args.loo_heads.resolve(); product_path = args.product_checkpoint.resolve()
    base, labels, _ = _load_model(base_path, device)
    heads = _load_loo_heads(loo_path, base_path, labels, device)
    embeddings, slices = _candidate_embeddings(samples, base, device)
    evaluation = _outer_writer_evaluation(samples, embeddings, slices, labels, heads, device)

    product, product_labels, _ = _load_model(product_path, device)
    if product_labels != labels:
        raise ValueError("product HWR vocabulary differs from grouping HWR vocabulary")
    for name, value in base.state_dict().items():
        if not name.startswith("math_head.") and not torch.equal(value.detach().cpu(), product.state_dict()[name].detach().cpu()):
            raise ValueError(f"base and product HWR encoders differ: {name}")
    product_probability = _probabilities(product.math_head, embeddings, device)
    product_samples = _augment(samples, product_probability, slices, labels)
    final_model = _fit(product_samples)

    output.mkdir(parents=True)
    model_path = output / "joint_grouping_selector.joblib"
    feature_names = FEATURE_NAMES + HWR_FEATURE_NAMES
    joblib.dump({
        "schema": SCHEMA, "model_version": MODEL_VERSION, "model": final_model,
        "feature_names": feature_names, "geometry_feature_names": FEATURE_NAMES,
        "hwr_feature_names": HWR_FEATURE_NAMES, "lattice_config": FULL_LATTICE_CONFIG,
        "group_bias": FIXED_GROUP_BIAS, "product_hwr_checkpoint_sha256": _sha256(product_path),
        "dataset": dataset, "product_default_enabled": False,
    }, model_path, compress=3)
    summary = {
        "schema": SCHEMA, "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "shadow", "model_version": MODEL_VERSION,
        "dataset": dataset, "lattice_config": FULL_LATTICE_CONFIG,
        "fixed_group_bias": FIXED_GROUP_BIAS, "feature_names": list(feature_names),
        "evaluation": evaluation,
        "artifacts": {
            "joint_grouping_selector": {"file": model_path.name, "sha256": _sha256(model_path)},
            "base_hwr_checkpoint_sha256": _sha256(base_path),
            "writer_loo_heads_sha256": _sha256(loo_path),
            "product_hwr_checkpoint_sha256": _sha256(product_path),
        },
        "contracts": {
            "target_label_or_cell_count_input": False, "writer_id_feature": False,
            "held_writer_absent_from_hwr_and_grouping_fit": True,
            "all_strokes_exactly_once": True, "product_default_enabled": False,
        },
        "limits": [
            "current writers are development evidence, not untouched product acceptance",
            "the final product artifact is fitted on all current ownership rows and is not scored on them",
        ],
    }
    (output / "joint_grouping_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n",
    )
    print(json.dumps({
        "output": str(output), "geometry": evaluation["geometry"],
        "joint_hwr": evaluation["joint_hwr"], "joint_failed": evaluation["joint_hwr_failed"],
        "model_sha256": _sha256(model_path),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
