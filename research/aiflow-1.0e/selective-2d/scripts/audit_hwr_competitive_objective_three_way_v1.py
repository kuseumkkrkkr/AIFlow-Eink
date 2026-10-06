#!/usr/bin/env python3
"""Paired audit of vanilla CE, P×K CE, and CE+SupCon on the same held writers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from build_normalized_ink_v1 import ROOT, _sha256
from audit_hwr_architecture_capacity_paired_v1 import _jsonl_train_uji_writer_hashes, _paired, _writer_bootstrap
from run_hwr_architecture_capacity_probe_v1 import (
    DEFAULT_CANONICAL_ROOT,
    DEFAULT_CURATED,
    DEFAULT_SPLIT,
    NpyDataset,
    ScaledInkClassifier,
    _class_labels,
    _load_writer_map,
)


DEFAULT_CAPACITY_REPORT = ROOT / "artifacts" / "hwr_augmentation_microscope_20261002" / "architecture_scale_uji_inner_writer_probe.json"
DEFAULT_OBJECTIVE_REPORT = ROOT / "artifacts" / "hwr_augmentation_microscope_20261002" / "objective_competition_ce_vs_supcon_pk_r1.json"
DEFAULT_REPORT = ROOT / "artifacts" / "hwr_augmentation_microscope_20261002" / "objective_competition_three_way_paired_r1.json"
DEFAULT_CACHE = Path(
    r"D:\AIFlow-Workspace\PrivateData\math-ink-data-collector\derived"
    r"\hwr-architecture-20261002\uji-writer-inner-scratch-cache-r3"
)


@torch.inference_mode()
def _predict(checkpoint_path: Path, dataset: NpyDataset, labels: list[str], device: torch.device, batch_size: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    schema = checkpoint.get("schema")
    if schema == "aiflow-hwr-architecture-capacity-probe/v1":
        architecture = checkpoint["arm"]
    elif schema == "aiflow-hwr-objective-competition-probe/v1":
        architecture = checkpoint["architecture"]
    else:
        raise ValueError(f"unexpected checkpoint schema: {checkpoint_path}: {schema}")
    model = ScaledInkClassifier(
        len(labels), int(architecture["width"]), int(architecture["layers"]),
        int(architecture["heads"]), int(architecture["feedforward"]),
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    top1: list[np.ndarray] = []
    top5: list[np.ndarray] = []
    for features, target in loader:
        logits = model(features.to(device, non_blocking=device.type == "cuda")).float()
        target = target.to(device, non_blocking=device.type == "cuda")
        indices = logits.topk(5, dim=1).indices
        top1.append((indices[:, 0] == target).cpu().numpy())
        top5.append((indices == target[:, None]).any(dim=1).cpu().numpy())
    metadata = {
        "path": str(checkpoint_path.resolve()),
        "sha256": _sha256(checkpoint_path),
        "seed": checkpoint["seed"],
        "architecture": architecture,
        "checkpoint_schema": schema,
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(top1), np.concatenate(top5), metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path, default=DEFAULT_CANONICAL_ROOT)
    parser.add_argument("--curated", type=Path, default=DEFAULT_CURATED)
    parser.add_argument("--writer-split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--capacity-report", type=Path, default=DEFAULT_CAPACITY_REPORT)
    parser.add_argument("--objective-report", type=Path, default=DEFAULT_OBJECTIVE_REPORT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(f"refusing to overwrite three-way audit: {args.report}")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    if args.batch_size < 1 or args.bootstrap_draws < 100:
        parser.error("batch size must be positive and bootstrap draws at least 100")

    capacity_report = json.loads(args.capacity_report.read_text(encoding="utf-8"))
    objective_report = json.loads(args.objective_report.read_text(encoding="utf-8"))
    if capacity_report.get("schema") != "aiflow-hwr-architecture-scale-writer-probe/v1":
        raise ValueError("unexpected capacity report schema")
    if objective_report.get("schema") != "aiflow-hwr-objective-competition-probe/v1":
        raise ValueError("unexpected objective report schema")
    objective_run = next((item for item in objective_report["seed_runs"] if item["seed"] == args.seed), None)
    if objective_run is None:
        raise ValueError(f"objective report has no seed={args.seed}")
    capacity_arm = next((item for item in capacity_report["arms"] if item["name"] == "d128_l4_h4" and item["seed"] == args.seed), None)
    if capacity_arm is None:
        raise ValueError(f"capacity report has no default architecture seed={args.seed}")

    device = torch.device(args.device)
    labels = _class_labels(args.canonical_root.resolve())
    dataset = NpyDataset(args.cache_root / "validation_features.npy", args.cache_root / "validation_labels.npy")
    split = json.loads(args.writer_split.read_text(encoding="utf-8"))
    if capacity_report["data"]["writer_split_sha256"] != _sha256(args.writer_split):
        raise ValueError("capacity report writer split hash mismatch")
    if objective_report["data"]["cache_audit"]["prepared_cache_sha256"]["validation_features.npy"] != _sha256(args.cache_root / "validation_features.npy"):
        raise ValueError("objective report validation feature hash mismatch")
    if objective_report["data"]["cache_audit"]["prepared_cache_sha256"]["validation_labels.npy"] != _sha256(args.cache_root / "validation_labels.npy"):
        raise ValueError("objective report validation label hash mismatch")

    selected = set(split["inner_split"]["validation_writer_hashes"])
    writer_map = _load_writer_map(args.curated.resolve())
    writer_hashes = _jsonl_train_uji_writer_hashes(args.canonical_root.resolve(), writer_map, selected, set(labels))
    if len(writer_hashes) != len(dataset) or len(set(writer_hashes)) != 8:
        raise ValueError("validation row/writer mapping is not exactly the expected eight-writer split")

    checkpoints = {
        "vanilla_ce_random_sampler": Path(capacity_arm["checkpoint"]),
        "ce_pk": Path(objective_run["arms"]["ce_pk"]["checkpoint"]),
        "ce_plus_supcon_pk": Path(objective_run["arms"]["ce_plus_supcon"]["checkpoint"]),
    }
    predictions: dict[str, dict[str, np.ndarray]] = {}
    checkpoint_metadata = {}
    for name, path in checkpoints.items():
        top1, top5, metadata = _predict(path, dataset, labels, device, args.batch_size)
        if int(metadata["seed"]) != args.seed:
            raise ValueError(f"checkpoint seed mismatch for {name}")
        predictions[name] = {"top1": top1, "top5": top5}
        checkpoint_metadata[name] = metadata

    pairwise: dict[str, Any] = {}
    names = list(checkpoints)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            key = f"{left}_vs_{right}"
            pairwise[key] = {
                "top1": _paired(predictions[left]["top1"], predictions[right]["top1"]),
                "top5": _paired(predictions[left]["top5"], predictions[right]["top5"]),
                "top1_writer_cluster_bootstrap": _writer_bootstrap(
                    predictions[left]["top1"], predictions[right]["top1"], writer_hashes,
                    args.seed + 5000 + left_index * 100 + names.index(right), args.bootstrap_draws,
                ),
            }

    report = {
        "schema": "aiflow-hwr-objective-three-way-paired-audit/v1",
        "status": "completed_exploratory_three_way_paired_audit",
        "data": {
            "capacity_report_sha256": _sha256(args.capacity_report),
            "objective_report_sha256": _sha256(args.objective_report),
            "writer_split_sha256": _sha256(args.writer_split),
            "validation_features_sha256": _sha256(args.cache_root / "validation_features.npy"),
            "validation_labels_sha256": _sha256(args.cache_root / "validation_labels.npy"),
            "validation_rows": len(dataset),
            "writer_clusters": len(set(writer_hashes)),
            "present_labels": int(len(np.unique(dataset.labels))),
            "crohme_rows": 0,
        },
        "seed": args.seed,
        "checkpoints": checkpoint_metadata,
        "pairwise": pairwise,
        "product_adopted": False,
        "interpretation_limit": "single seed, one eight-writer internal split, 66 overlapping labels; exploratory evidence only, no release or CROHME claim",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    compact = {key: {"top1": value["top1"], "top5": value["top5"], "top1_writer_ci": value["top1_writer_cluster_bootstrap"]["delta_percentage_points_percentile_95_interval"]} for key, value in pairwise.items()}
    print(json.dumps({"event": "objective_three_way_paired_audit_complete", "report": str(args.report.resolve()), "pairwise": compact, "crohme_rows": 0, "product_adopted": False}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
