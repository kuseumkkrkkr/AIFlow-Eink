#!/usr/bin/env python3
"""Layerwise CE/SupCon gradient-alignment microscope on one attested batch.

This is a diagnostic only: it does not train, update, export, or promote a model.
It compares the same fixed P×K batch at random initialization and at the
hash-pinned completed CE checkpoint. CROHME is excluded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from build_normalized_ink_v1 import ROOT, _sha256
from run_hwr_architecture_capacity_probe_v1 import (
    DEFAULT_CANONICAL_ROOT,
    DEFAULT_CURATED,
    DEFAULT_SPLIT,
    DEFAULT_VOCAB_CHECKPOINT,
    NpyDataset,
    ScaledInkClassifier,
)
from run_hwr_competitive_objective_probe_v1 import (
    DEFAULT_CACHE,
    PKBatchSampler,
    _reuse_hash_attested_cache,
    _seed_everything,
    supervised_contrastive_loss,
)


SEED = 20261004
BATCH_CLASSES = 12
ROWS_PER_CLASS = 4
SUPCON_WEIGHT = 0.1
TEMPERATURE = 0.1
CE_PARTIAL_REPORT = ROOT / "artifacts" / "hwr_augmentation_microscope_20261002" / "objective_competition_ce_vs_supcon_pk_r4_partial.json"
OUTPUT = ROOT / "artifacts" / "hwr_augmentation_microscope_20261002" / "objective_gradient_alignment_microscope_20261002.json"


def _component(name: str) -> str:
    if name.startswith("input_projection."):
        return "input_projection"
    if name == "position":
        return "position_embedding"
    if name.startswith("encoder.layers."):
        return f"encoder_layer_{name.split('.')[2]}"
    if name.startswith("pool_score."):
        return "attention_pool"
    if name.startswith("math_head."):
        return "372_class_head"
    return "other"


def _gradient_stats(
    named_parameters: list[tuple[str, torch.nn.Parameter]],
    ce_gradients: tuple[torch.Tensor | None, ...],
    supcon_gradients: tuple[torch.Tensor | None, ...],
) -> dict[str, Any]:
    aggregate: dict[str, dict[str, float]] = {}
    for (name, _), ce_gradient, supcon_gradient in zip(named_parameters, ce_gradients, supcon_gradients, strict=True):
        if ce_gradient is None or supcon_gradient is None:
            continue
        # CPU float64 accumulation avoids keeping diagnostic reductions on CUDA
        # and makes the small reported norms/cosines numerically stable.
        ce = ce_gradient.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
        supcon = supcon_gradient.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
        weighted = supcon * SUPCON_WEIGHT
        group = aggregate.setdefault(_component(name), {"ce_sq": 0.0, "supcon_sq": 0.0, "weighted_sq": 0.0, "dot": 0.0})
        group["ce_sq"] += float(torch.dot(ce, ce))
        group["supcon_sq"] += float(torch.dot(supcon, supcon))
        group["weighted_sq"] += float(torch.dot(weighted, weighted))
        group["dot"] += float(torch.dot(ce, supcon))

    components: dict[str, Any] = {}
    totals = {key: 0.0 for key in ("ce_sq", "supcon_sq", "weighted_sq", "dot")}
    for key, values in aggregate.items():
        for total_key in totals:
            totals[total_key] += values[total_key]
        ce_norm = values["ce_sq"] ** 0.5
        supcon_norm = values["supcon_sq"] ** 0.5
        weighted_norm = values["weighted_sq"] ** 0.5
        combined_sq = values["ce_sq"] + values["weighted_sq"] + 2.0 * SUPCON_WEIGHT * values["dot"]
        combined_norm = max(combined_sq, 0.0) ** 0.5
        cosine = values["dot"] / (ce_norm * supcon_norm) if ce_norm and supcon_norm else None
        combined_cosine = (values["ce_sq"] + SUPCON_WEIGHT * values["dot"]) / (ce_norm * combined_norm) if ce_norm and combined_norm else None
        components[key] = {
            "ce_gradient_norm": ce_norm,
            "supcon_gradient_norm_unweighted": supcon_norm,
            "weighted_supcon_to_ce_norm_ratio": weighted_norm / ce_norm if ce_norm else None,
            "ce_vs_supcon_cosine": cosine,
            "combined_gradient_norm": combined_norm,
            "combined_vs_ce_cosine": combined_cosine,
        }

    ce_norm = totals["ce_sq"] ** 0.5
    supcon_norm = totals["supcon_sq"] ** 0.5
    weighted_norm = totals["weighted_sq"] ** 0.5
    combined_sq = totals["ce_sq"] + totals["weighted_sq"] + 2.0 * SUPCON_WEIGHT * totals["dot"]
    combined_norm = max(combined_sq, 0.0) ** 0.5
    cosine = totals["dot"] / (ce_norm * supcon_norm) if ce_norm and supcon_norm else None
    combined_cosine = (totals["ce_sq"] + SUPCON_WEIGHT * totals["dot"]) / (ce_norm * combined_norm) if ce_norm and combined_norm else None
    return {
        "global": {
            "ce_gradient_norm": ce_norm,
            "supcon_gradient_norm_unweighted": supcon_norm,
            "weighted_supcon_to_ce_norm_ratio": weighted_norm / ce_norm if ce_norm else None,
            "ce_vs_supcon_cosine": cosine,
            "combined_gradient_norm": combined_norm,
            "combined_vs_ce_cosine": combined_cosine,
            "combined_minus_ce_relative_l2": weighted_norm / ce_norm if ce_norm else None,
        },
        "components": components,
    }


def _measure(model: ScaledInkClassifier, features: torch.Tensor, targets: torch.Tensor, label: str) -> dict[str, Any]:
    model.train()  # Keep the training dropout path; both objectives share one forward graph/mask.
    _seed_everything(SEED + 991)
    embeddings = model.encode(features)
    logits = model.math_head(embeddings)
    ce = F.cross_entropy(logits, targets)
    supcon = supervised_contrastive_loss(embeddings, targets, TEMPERATURE)
    parameters = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    ce_gradients = torch.autograd.grad(ce, tuple(parameter for _, parameter in parameters), retain_graph=True, allow_unused=True)
    supcon_gradients = torch.autograd.grad(supcon, tuple(parameter for _, parameter in parameters), allow_unused=True)
    return {
        "checkpoint": label,
        "ce_loss": float(ce.detach()),
        "supcon_loss": float(supcon.detach()),
        "weighted_supcon_loss": float(SUPCON_WEIGHT * supcon.detach()),
        "gradient_alignment": _gradient_stats(parameters, ce_gradients, supcon_gradients),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    cache_labels, cache_audit, cache_provenance = _reuse_hash_attested_cache(
        DEFAULT_CANONICAL_ROOT.resolve(), DEFAULT_CURATED.resolve(), DEFAULT_SPLIT.resolve(), DEFAULT_CACHE.resolve()
    )
    partial = json.loads(CE_PARTIAL_REPORT.read_text(encoding="utf-8"))
    if partial.get("status") != "partial_ce_arm_complete_supcon_arm_failed" or partial.get("data", {}).get("crohme_rows") != 0:
        raise ValueError("CE partial report is not the expected CROHME-free exploratory run")
    if partial.get("experiment", {}).get("seed") != SEED:
        raise ValueError("CE checkpoint seed does not match this audit")
    partial_data = partial.get("data", {})
    source_hashes = cache_provenance["source_hashes"]
    for key in ("canonical_hwrt_sha256", "canonical_uji_sha256", "curated_uji_sha256", "writer_split_sha256"):
        if partial_data.get(key) != source_hashes.get(key):
            raise ValueError(f"CE partial report source fingerprint mismatch: {key}")
    if partial_data.get("prepared_cache_sha256") != cache_provenance.get("prepared_cache_sha256"):
        raise ValueError("CE partial report cache hashes do not match the attested training arrays")
    if partial_data.get("cache_reuse_mode") != "hash_attested_reuse":
        raise ValueError("CE partial report was not produced from the hash-attested cache")
    if partial_data.get("cache_audit_report_sha256") != cache_provenance.get("audit_report_sha256"):
        raise ValueError("CE partial report points to a different cache audit")
    sampling = partial.get("experiment", {}).get("sampler", {})
    if sampling != {"classes_per_batch": BATCH_CLASSES, "rows_per_class": ROWS_PER_CLASS, "batch_size": BATCH_CLASSES * ROWS_PER_CLASS}:
        raise ValueError("CE checkpoint sampler does not match the fixed diagnostic batch")
    completed = partial["completed_arm"]
    checkpoint_path = Path(completed["checkpoint"]).resolve()
    checkpoint_hash = _sha256(checkpoint_path)
    if checkpoint_hash != completed.get("checkpoint_sha256"):
        raise ValueError("completed CE checkpoint SHA-256 does not match its partial report")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    expected_architecture = {"width": 128, "layers": 4, "heads": 4, "feedforward": 512}
    actual_architecture = checkpoint.get("architecture", {})
    checkpoint_objective = checkpoint.get("objective", {})
    if checkpoint.get("arm") != "ce_pk" or checkpoint.get("seed") != SEED or any(actual_architecture.get(key) != value for key, value in expected_architecture.items()):
        raise ValueError("CE checkpoint arm/seed/architecture mismatch")
    if checkpoint_objective.get("ce_weight") != 1.0 or checkpoint_objective.get("supcon_weight") != 0.0:
        raise ValueError("pinned checkpoint is not a pure CE baseline")

    dataset = NpyDataset(DEFAULT_CACHE / "train_features.npy", DEFAULT_CACHE / "train_labels.npy")
    sampler = PKBatchSampler(np.asarray(dataset.labels, dtype=np.int64), BATCH_CLASSES, ROWS_PER_CLASS, 1, SEED + 17)
    batch_indices = next(iter(sampler))
    batch_items = [dataset[index] for index in batch_indices]
    features = torch.stack([item[0] for item in batch_items])
    targets = torch.tensor([item[1] for item in batch_items], dtype=torch.long)
    if features.shape != (BATCH_CLASSES * ROWS_PER_CLASS, 128, 5) or len(torch.unique(targets)) != BATCH_CLASSES:
        raise AssertionError("fixed first batch violates the r4 P×K contract")
    index_sha256 = hashlib.sha256(np.asarray(batch_indices, dtype="<i8").tobytes()).hexdigest()

    torch.set_num_threads(1)
    _seed_everything(SEED)
    initial_model = ScaledInkClassifier(len(cache_labels), 128, 4, 4, 512).cpu()
    initial_result = _measure(initial_model, features, targets, "random_initialization_seed20261004")
    del initial_model

    trained_model = ScaledInkClassifier(len(cache_labels), 128, 4, 4, 512).cpu()
    trained_model.load_state_dict(checkpoint["state_dict"], strict=True)
    trained_result = _measure(trained_model, features, targets, "completed_ce_epoch4_checkpoint")

    report = {
        "schema": "aiflow-hwr-objective-gradient-alignment-microscope/v1",
        "status": "completed_fixed_batch_gradient_alignment_diagnostic",
        "product_adopted": False,
        "data": {
            "crohme_rows": 0,
            "train_rows": int(len(dataset)),
            "cache_audit": cache_audit,
            "cache_reuse_provenance": cache_provenance,
            "train_features_sha256": _sha256(DEFAULT_CACHE / "train_features.npy"),
            "train_labels_sha256": _sha256(DEFAULT_CACHE / "train_labels.npy"),
        },
        "fixed_batch": {
            "seed": SEED,
            "sampler_seed": SEED + 17,
            "epoch_index": 0,
            "classes_per_batch": BATCH_CLASSES,
            "rows_per_class": ROWS_PER_CLASS,
            "batch_size": BATCH_CLASSES * ROWS_PER_CLASS,
            "sample_index_list_sha256": index_sha256,
            "class_counts": {str(int(label)): int(count) for label, count in zip(*np.unique(targets.numpy(), return_counts=True), strict=True)},
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint_report": str(CE_PARTIAL_REPORT),
        },
        "experiment": {
            "architecture": expected_architecture,
            "objective": "CE versus unweighted SupCon gradient; combined direction uses CE + 0.1 × SupCon",
            "temperature": TEMPERATURE,
            "supcon_weight": SUPCON_WEIGHT,
            "device": "cpu",
            "same_forward_graph_for_both_losses": True,
            "optimizer_step_performed": False,
        },
        "records": [initial_result, trained_result],
        "interpretation_limit": "one fixed train batch; gradient geometry only, not a paired multi-epoch accuracy result or evidence for changing the objective weight",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "output": str(output_path), "crohme_rows": 0, "records": [{"checkpoint": row["checkpoint"], **row["gradient_alignment"]["global"]} for row in report["records"]]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
