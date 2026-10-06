#!/usr/bin/env python3
"""Compare linear and nonlinear readouts over the same frozen HWR embedding."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from run_hwr_affine_distillation_experiment_v1 import (
    SOURCE_IDS,
    _assert_comparison_sources_eligible,
    _load_cache,
    _load_teacher,
    _score,
    _sha256,
    _write_json,
)


class ResidualMLPReadout(nn.Module):
    """A zero-initialized nonlinear residual adapter before the frozen 372-way interface."""

    def __init__(self, output_head: nn.Linear, width: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        dimension = output_head.in_features
        self.expand = nn.Linear(dimension, width)
        self.dropout = nn.Dropout(dropout)
        self.contract = nn.Linear(width, dimension)
        nn.init.zeros_(self.contract.weight)
        nn.init.zeros_(self.contract.bias)
        self.output = deepcopy(output_head)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        residual = self.contract(self.dropout(F.gelu(self.expand(embedding))))
        return self.output(embedding + residual)


def _embed(model, features: np.ndarray, device: torch.device, batch_size: int) -> torch.Tensor:
    outputs = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            batch = torch.from_numpy(
                np.array(features[start:start + batch_size], dtype=np.float32, copy=True)
            ).to(device)
            embedding = model.encode(batch).float()
            if not torch.isfinite(embedding).all():
                raise FloatingPointError(f"non-finite frozen embedding at row {start}")
            outputs.append(embedding)
    if not outputs:
        return torch.empty((0, model.math_head.in_features), device=device)
    return torch.cat(outputs, dim=0)


def _balanced_orders(labels: np.ndarray, classes: int, samples_per_class: int, epochs: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    rows_by_class = [np.flatnonzero(labels == class_id) for class_id in range(classes)]
    empty = [index for index, rows in enumerate(rows_by_class) if not len(rows)]
    if empty:
        raise ValueError(f"class-balanced frozen-head training has empty classes: {empty}")
    orders = []
    for _ in range(epochs):
        order = np.concatenate([
            rng.choice(rows, size=samples_per_class, replace=True)
            for rows in rows_by_class
        ]).astype(np.int64, copy=False)
        rng.shuffle(order)
        orders.append(order)
    return orders


def _pair_counts(before: np.ndarray, after: np.ndarray, targets: np.ndarray, k: int) -> dict[str, Any]:
    if k == 1:
        before_pred = np.argmax(before, axis=1)
        after_pred = np.argmax(after, axis=1)
        before_hit = before_pred == targets
        after_hit = after_pred == targets
    else:
        before_top = np.argpartition(-before, kth=k - 1, axis=1)[:, :k]
        after_top = np.argpartition(-after, kth=k - 1, axis=1)[:, :k]
        before_hit = np.any(before_top == targets[:, None], axis=1)
        after_hit = np.any(after_top == targets[:, None], axis=1)
    recovered = int((~before_hit & after_hit).sum())
    regressed = int((before_hit & ~after_hit).sum())
    discordant = recovered + regressed
    tail = min(recovered, regressed)
    p = 1.0 if not discordant else min(
        1.0,
        2.0 * sum(math.comb(discordant, index) for index in range(tail + 1)) / (2 ** discordant),
    )
    return {
        "rows": int(len(targets)),
        "before_hits": int(before_hit.sum()),
        "after_hits": int(after_hit.sum()),
        "before_accuracy": float(before_hit.mean()) if len(targets) else 0.0,
        "after_accuracy": float(after_hit.mean()) if len(targets) else 0.0,
        "recovered": recovered,
        "regressed": regressed,
        "net_rows": recovered - regressed,
        "exact_mcnemar_two_sided_p": p,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checkpoint-out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--encode-batch-size", type=int, default=128)
    parser.add_argument("--train-batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--samples-per-class-per-epoch", type=int, default=128)
    parser.add_argument("--hidden-width", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--hard-loss-weight", type=float, default=0.70)
    parser.add_argument("--distill-loss-weight", type=float, default=0.30)
    args = parser.parse_args()
    if min(args.encode_batch_size, args.train_batch_size, args.epochs, args.samples_per_class_per_epoch, args.hidden_width) < 1:
        parser.error("batch sizes, epochs, samples per class, and hidden width must be positive")
    if args.learning_rate <= 0.0:
        parser.error("--learning-rate must be positive")
    if min(args.hard_loss_weight, args.distill_loss_weight) < 0.0 or not np.isclose(args.hard_loss_weight + args.distill_loss_weight, 1.0):
        parser.error("hard-label and teacher-KL weights must be non-negative and sum to 1")
    if args.report.exists() or args.checkpoint_out.exists():
        parser.error("refusing to overwrite an existing report or checkpoint")

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    manifest = json.loads((args.data_dir / "prepared_manifest.json").read_text(encoding="utf-8"))
    _assert_comparison_sources_eligible(
        manifest.get("input_policy", {}),
        event="frozen_encoder_readout_capacity_probe",
        source_split_rows=manifest.get("source_audit", {}).get("source_split_rows", {}),
    )
    if manifest.get("status") != "pass" or manifest.get("current_checkpoint", {}).get("sha256") != _sha256(args.checkpoint):
        raise ValueError("data cache must have a passing audit for the supplied frozen checkpoint")
    model, labels, _ = _load_teacher(args.checkpoint, device)
    train = _load_cache(args.data_dir, "train")
    test = _load_cache(args.data_dir, "test")
    train_y = np.asarray(train["labels"], dtype=np.int64)
    train_sources = np.asarray(train["sources"], dtype=np.int8)
    test_y = np.asarray(test["labels"], dtype=np.int64)
    test_sources = np.asarray(test["sources"], dtype=np.int8)
    if len(labels) != 372 or np.any(train_y < 0) or np.any(train_y >= len(labels)) or np.any(test_y < 0) or np.any(test_y >= len(labels)):
        raise ValueError("readout probe requires the frozen 372-class vocabulary")

    started = time.perf_counter()
    train_embeddings = _embed(model, train["features"], device, args.encode_batch_size)
    test_embeddings = _embed(model, test["features"], device, args.encode_batch_size)
    base_head = model.math_head
    teacher_test_logits = base_head(test_embeddings).float().detach().cpu().numpy()
    teacher_train_weight = base_head.weight.detach()
    teacher_train_bias = base_head.bias.detach()

    linear = deepcopy(base_head)
    nonlinear = ResidualMLPReadout(base_head, args.hidden_width).to(device)
    if not torch.allclose(nonlinear(test_embeddings[:8]), teacher_test_logits[:8] if isinstance(teacher_test_logits, torch.Tensor) else torch.from_numpy(teacher_test_logits[:8]).to(device), atol=1.0e-6, rtol=1.0e-6):
        raise AssertionError("zero-initialized nonlinear readout does not preserve the teacher output")
    arms: dict[str, nn.Module] = {"linear": linear, "residual_mlp": nonlinear}
    optimizers = {
        name: torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=1.0e-4)
        for name, head in arms.items()
    }
    sample_orders = _balanced_orders(
        train_y, len(labels), args.samples_per_class_per_epoch, args.epochs, seed=20261003,
    )

    head_weight = teacher_train_weight
    head_bias = teacher_train_bias
    for epoch, order in enumerate(sample_orders, start=1):
        epoch_losses: dict[str, list[float]] = {name: [] for name in arms}
        for start in range(0, len(order), args.train_batch_size):
            row_ids_np = order[start:start + args.train_batch_size]
            row_ids = torch.as_tensor(row_ids_np, dtype=torch.long, device=device)
            embedding = train_embeddings[row_ids]
            target = torch.as_tensor(train_y[row_ids_np], dtype=torch.long, device=device)
            source = torch.as_tensor(train_sources[row_ids_np], dtype=torch.long, device=device)
            teacher_logits = F.linear(embedding, head_weight, head_bias).detach().float()
            real_mask = source != SOURCE_IDS["synthetic_equal"]

            for name, head in arms.items():
                head.train()
                optimizer = optimizers[name]
                optimizer.zero_grad(set_to_none=True)
                logits = head(embedding).float()
                hard_loss = F.cross_entropy(logits, target)
                if real_mask.any():
                    distill_loss = F.kl_div(
                        F.log_softmax(logits[real_mask] / 2.0, dim=1),
                        F.softmax(teacher_logits[real_mask] / 2.0, dim=1),
                        reduction="none",
                    ).sum(dim=1).mean() * 4.0
                else:
                    distill_loss = hard_loss.new_zeros(())
                loss = args.hard_loss_weight * hard_loss + args.distill_loss_weight * distill_loss
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss in {name} at epoch {epoch}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                optimizer.step()
                epoch_losses[name].append(float(loss.detach().cpu()))
        print({
            "event": "readout_probe_epoch",
            "epoch": epoch,
            "rows_per_arm": int(len(order)),
            "linear_loss": float(np.mean(epoch_losses["linear"])),
            "residual_mlp_loss": float(np.mean(epoch_losses["residual_mlp"])),
        }, flush=True)

    def predict(head: nn.Module) -> np.ndarray:
        head.eval()
        outputs = []
        with torch.inference_mode():
            for start in range(0, len(test_embeddings), args.train_batch_size):
                logits = head(test_embeddings[start:start + args.train_batch_size]).float()
                if not torch.isfinite(logits).all():
                    raise FloatingPointError("non-finite readout output")
                outputs.append(logits.cpu().numpy())
        return np.concatenate(outputs, axis=0)

    linear_logits = predict(linear)
    nonlinear_logits = predict(nonlinear)
    teacher_metrics = _score(teacher_test_logits, test_y, test_sources, labels)
    linear_metrics = _score(linear_logits, test_y, test_sources, labels)
    nonlinear_metrics = _score(nonlinear_logits, test_y, test_sources, labels)
    report = {
        "schema": "aiflow-hwr-frozen-encoder-readout-capacity/v1",
        "status": "completed_exploratory_architecture_probe",
        "frozen_encoder": {
            "checkpoint": str(args.checkpoint.resolve()),
            "sha256": _sha256(args.checkpoint),
            "architecture": "128-dim, 4-layer Transformer; frozen during both readout arms",
        },
        "readouts": {
            "teacher_linear": {"parameters": sum(parameter.numel() for parameter in base_head.parameters()), "metrics": teacher_metrics},
            "trained_linear": {"parameters": sum(parameter.numel() for parameter in linear.parameters()), "metrics": linear_metrics},
            "zero_init_residual_mlp": {
                "hidden_width": args.hidden_width,
                "parameters": sum(parameter.numel() for parameter in nonlinear.parameters()),
                "metrics": nonlinear_metrics,
            },
        },
        "paired": {
            "teacher_vs_trained_linear_top1": _pair_counts(teacher_test_logits, linear_logits, test_y, 1),
            "teacher_vs_trained_linear_top5": _pair_counts(teacher_test_logits, linear_logits, test_y, 5),
            "teacher_vs_residual_mlp_top1": _pair_counts(teacher_test_logits, nonlinear_logits, test_y, 1),
            "teacher_vs_residual_mlp_top5": _pair_counts(teacher_test_logits, nonlinear_logits, test_y, 5),
            "trained_linear_vs_residual_mlp_top1": _pair_counts(linear_logits, nonlinear_logits, test_y, 1),
            "trained_linear_vs_residual_mlp_top5": _pair_counts(linear_logits, nonlinear_logits, test_y, 5),
        },
        "training": {
            "epochs": args.epochs,
            "samples_per_class_per_epoch": args.samples_per_class_per_epoch,
            "batch_size": args.train_batch_size,
            "learning_rate": args.learning_rate,
            "optimizer": "AdamW",
            "hard_label_cross_entropy": args.hard_loss_weight,
            "teacher_kl_on_real_rows": args.distill_loss_weight,
            "class_balanced_orders_shared_between_arms": True,
            "encoder_trainable": False,
        },
        "data": {
            "train_rows": int(len(train_y)),
            "test_rows": int(len(test_y)),
            "classes": len(labels),
            "source_manifest_sha256": _sha256(args.data_dir / "prepared_manifest.json"),
            "heldout_policy": "previously consumed exploratory split; not fresh-writer/device acceptance",
        },
        "crohme_rows": 0,
        "product_adopted": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema": "aiflow-hwr-frozen-encoder-readout-probe/v1",
        "base_checkpoint_sha256": _sha256(args.checkpoint),
        "math_labels": labels,
        "linear_state_dict": linear.state_dict(),
        "residual_mlp_state_dict": nonlinear.state_dict(),
        "hidden_width": args.hidden_width,
        "report": report,
    }, args.checkpoint_out)
    _write_json(args.report, report)
    print({"event": "readout_probe_complete", "teacher_top1": teacher_metrics["overall"]["top1"],
           "trained_linear_top1": linear_metrics["overall"]["top1"],
           "residual_mlp_top1": nonlinear_metrics["overall"]["top1"],
           "report": str(args.report.resolve()), "checkpoint_out": str(args.checkpoint_out.resolve())}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
