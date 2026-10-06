#!/usr/bin/env python3
"""Compare CE with a top-wrong-class margin term on one attested batch.

This microscope performs forward/backward diagnostics only. It does not step an
optimizer or promote a model; the held-out validation set and CROHME are unused.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from build_normalized_ink_v1 import ROOT, _sha256
from run_hwr_architecture_capacity_probe_v1 import (
    DEFAULT_CANONICAL_ROOT,
    DEFAULT_CURATED,
    DEFAULT_SPLIT,
    NpyDataset,
    ScaledInkClassifier,
)
from run_hwr_competitive_objective_probe_v1 import (
    DEFAULT_CACHE,
    PKBatchSampler,
    _reuse_hash_attested_cache,
    _seed_everything,
)


SEED = 20261004
BATCH_CLASSES = 12
ROWS_PER_CLASS = 4
MARGIN = 0.5
WEIGHTS = (0.1, 0.25, 0.5)
CE_PARTIAL_REPORT = ROOT / "artifacts" / "hwr_augmentation_microscope_20261002" / "objective_competition_ce_vs_supcon_pk_r4_partial.json"
OUTPUT = ROOT / "artifacts" / "hwr_augmentation_microscope_20261002" / "objective_hardnegative_gradient_microscope_20261002.json"


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


def _compare_gradients(
    parameters: list[tuple[str, torch.nn.Parameter]],
    ce_gradients: tuple[torch.Tensor | None, ...],
    hard_gradients: tuple[torch.Tensor | None, ...],
) -> dict[str, Any]:
    groups: dict[str, dict[str, float]] = {}
    for (name, _), ce_gradient, hard_gradient in zip(parameters, ce_gradients, hard_gradients, strict=True):
        if ce_gradient is None or hard_gradient is None:
            continue
        ce = ce_gradient.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
        hard = hard_gradient.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
        values = groups.setdefault(_component(name), {"ce_sq": 0.0, "hard_sq": 0.0, "dot": 0.0})
        values["ce_sq"] += float(torch.dot(ce, ce))
        values["hard_sq"] += float(torch.dot(hard, hard))
        values["dot"] += float(torch.dot(ce, hard))

    aggregates = {key: sum(values[key] for values in groups.values()) for key in ("ce_sq", "hard_sq", "dot")}
    per_component: dict[str, Any] = {}
    for name, values in groups.items():
        per_component[name] = _summarize(values)
    return {
        "global": _summarize(aggregates),
        "components": per_component,
    }


def _summarize(values: dict[str, float]) -> dict[str, Any]:
    ce_norm = values["ce_sq"] ** 0.5
    hard_norm = values["hard_sq"] ** 0.5
    cosine = values["dot"] / (ce_norm * hard_norm) if ce_norm and hard_norm else None
    combined: dict[str, Any] = {}
    for weight in WEIGHTS:
        combined_sq = values["ce_sq"] + (weight ** 2) * values["hard_sq"] + 2.0 * weight * values["dot"]
        combined_norm = max(combined_sq, 0.0) ** 0.5
        combined[format(weight, ".2f")] = {
            "weighted_hard_to_ce_norm_ratio": weight * hard_norm / ce_norm if ce_norm else None,
            "combined_gradient_norm": combined_norm,
            "combined_vs_ce_cosine": (values["ce_sq"] + weight * values["dot"]) / (ce_norm * combined_norm) if ce_norm and combined_norm else None,
        }
    return {
        "ce_gradient_norm": ce_norm,
        "hardnegative_gradient_norm": hard_norm,
        "ce_vs_hardnegative_cosine": cosine,
        "weighted_combinations": combined,
    }


def _measure(model: ScaledInkClassifier, features: torch.Tensor, targets: torch.Tensor, label: str) -> dict[str, Any]:
    model.train()
    _seed_everything(SEED + 991)
    embeddings = model.encode(features)
    logits = model.math_head(embeddings)
    ce = F.cross_entropy(logits, targets)
    wrong_logits = logits.clone()
    wrong_logits.scatter_(1, targets[:, None], float("-inf"))
    hardest_wrong = wrong_logits.max(dim=1).values
    correct = logits.gather(1, targets[:, None]).squeeze(1)
    margins = correct - hardest_wrong
    hinge = F.relu(MARGIN - margins)
    hard_loss = hinge.mean()
    parameters = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    parameter_tensors = tuple(parameter for _, parameter in parameters)
    ce_gradients = torch.autograd.grad(ce, parameter_tensors, retain_graph=True, allow_unused=True)
    hard_gradients = torch.autograd.grad(hard_loss, parameter_tensors, allow_unused=True)
    return {
        "checkpoint": label,
        "ce_loss": float(ce.detach()),
        "hardnegative_margin_loss": float(hard_loss.detach()),
        "top1_hits": int((logits.argmax(dim=1) == targets).sum()),
        "top5_hits": int((logits.topk(5, dim=1).indices == targets[:, None]).any(dim=1).sum()),
        "active_hinge_examples": int((hinge > 0).sum()),
        "active_hinge_fraction": float((hinge > 0).float().mean()),
        "correct_minus_hardest_wrong_margin_mean": float(margins.mean()),
        "gradient_alignment": _compare_gradients(parameters, ce_gradients, hard_gradients),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    output_path = parser.parse_args().output.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")

    labels, cache_audit, cache_provenance = _reuse_hash_attested_cache(
        DEFAULT_CANONICAL_ROOT.resolve(), DEFAULT_CURATED.resolve(), DEFAULT_SPLIT.resolve(), DEFAULT_CACHE.resolve()
    )
    partial = json.loads(CE_PARTIAL_REPORT.read_text(encoding="utf-8"))
    if partial.get("status") != "partial_ce_arm_complete_supcon_arm_failed" or partial.get("data", {}).get("crohme_rows") != 0:
        raise ValueError("unexpected CE partial report or CROHME scope")
    partial_data = partial["data"]
    for key, value in cache_provenance["source_hashes"].items():
        if partial_data.get(key) != value:
            raise ValueError(f"CE report source fingerprint mismatch: {key}")
    if partial_data.get("prepared_cache_sha256") != cache_provenance.get("prepared_cache_sha256"):
        raise ValueError("CE report cache hashes do not match current attested arrays")
    if partial_data.get("cache_reuse_mode") != "hash_attested_reuse" or partial_data.get("cache_audit_report_sha256") != cache_provenance.get("audit_report_sha256"):
        raise ValueError("CE report does not point to the current hash-attested cache audit")
    if partial.get("experiment", {}).get("seed") != SEED:
        raise ValueError("CE report seed mismatch")
    sampler_config = partial.get("experiment", {}).get("sampler", {})
    if sampler_config != {"classes_per_batch": BATCH_CLASSES, "rows_per_class": ROWS_PER_CLASS, "batch_size": BATCH_CLASSES * ROWS_PER_CLASS}:
        raise ValueError("CE report sampler does not match the diagnostic batch")

    completed = partial["completed_arm"]
    checkpoint_path = Path(completed["checkpoint"]).resolve()
    checkpoint_hash = _sha256(checkpoint_path)
    if checkpoint_hash != completed.get("checkpoint_sha256"):
        raise ValueError("CE checkpoint SHA-256 mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    expected_architecture = {"width": 128, "layers": 4, "heads": 4, "feedforward": 512}
    actual_architecture = checkpoint.get("architecture", {})
    if checkpoint.get("arm") != "ce_pk" or checkpoint.get("seed") != SEED or any(actual_architecture.get(key) != value for key, value in expected_architecture.items()):
        raise ValueError("checkpoint arm/seed/architecture mismatch")
    if checkpoint.get("objective", {}).get("ce_weight") != 1.0 or checkpoint.get("objective", {}).get("supcon_weight") != 0.0:
        raise ValueError("pinned checkpoint is not a pure CE checkpoint")

    dataset = NpyDataset(DEFAULT_CACHE / "train_features.npy", DEFAULT_CACHE / "train_labels.npy")
    sampler = PKBatchSampler(np.asarray(dataset.labels, dtype=np.int64), BATCH_CLASSES, ROWS_PER_CLASS, 1, SEED + 17)
    indices = next(iter(sampler))
    items = [dataset[index] for index in indices]
    features = torch.stack([item[0] for item in items])
    targets = torch.tensor([item[1] for item in items], dtype=torch.long)
    if features.shape != (BATCH_CLASSES * ROWS_PER_CLASS, 128, 5) or len(torch.unique(targets)) != BATCH_CLASSES:
        raise AssertionError("P×K diagnostic batch contract failed")
    class_ids, class_counts = np.unique(targets.numpy(), return_counts=True)
    if not np.all(class_counts == ROWS_PER_CLASS):
        raise AssertionError("diagnostic batch is not class-balanced")
    torch.set_num_threads(1)

    _seed_everything(SEED)
    initial = ScaledInkClassifier(len(labels), 128, 4, 4, 512).cpu()
    initial_result = _measure(initial, features, targets, "random_initialization_seed20261004")
    del initial
    trained = ScaledInkClassifier(len(labels), 128, 4, 4, 512).cpu()
    trained.load_state_dict(checkpoint["state_dict"], strict=True)
    trained_result = _measure(trained, features, targets, "completed_ce_epoch4_checkpoint")

    report = {
        "schema": "aiflow-hwr-hardnegative-gradient-microscope/v1",
        "status": "completed_fixed_batch_hardnegative_gradient_diagnostic",
        "product_adopted": False,
        "data": {
            "crohme_rows": 0,
            "validation_data_used": False,
            "train_rows": int(len(dataset)),
            "cache_audit": cache_audit,
            "cache_reuse_provenance": cache_provenance,
        },
        "fixed_batch": {
            "seed": SEED,
            "sampler_seed": SEED + 17,
            "epoch_index": 0,
            "classes_per_batch": BATCH_CLASSES,
            "rows_per_class": ROWS_PER_CLASS,
            "batch_size": int(len(targets)),
            "class_counts": {str(int(class_id)): int(count) for class_id, count in zip(class_ids, class_counts, strict=True)},
            "checkpoint_sha256": checkpoint_hash,
        },
        "experiment": {
            "architecture": expected_architecture,
            "hardnegative_definition": "mean(relu(0.5 + max_wrong_logit - target_logit))",
            "weights_against_ce": list(WEIGHTS),
            "same_forward_graph_for_ce_and_hardnegative": True,
            "device": "cpu",
            "optimizer_step_performed": False,
        },
        "records": [initial_result, trained_result],
        "interpretation_limit": "one fixed training batch; gradient geometry only, not accuracy improvement or evidence for changing the training objective",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({
        "status": report["status"],
        "output": str(output_path),
        "crohme_rows": 0,
        "records": [{
            "checkpoint": row["checkpoint"],
            "ce_loss": row["ce_loss"],
            "hardnegative_loss": row["hardnegative_margin_loss"],
            "active_hinge_examples": row["active_hinge_examples"],
            "top1_hits": row["top1_hits"],
            "top5_hits": row["top5_hits"],
            "global_gradient_alignment": row["gradient_alignment"]["global"],
            "class_head_gradient_alignment": row["gradient_alignment"]["components"].get("372_class_head"),
        } for row in report["records"]],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
