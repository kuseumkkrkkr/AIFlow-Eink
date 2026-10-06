#!/usr/bin/env python3
"""Paired row and writer-cluster comparison of scratch architecture probe checkpoints."""

from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from build_normalized_ink_v1 import ROOT, _json_lines, _sha256
from run_hwr_architecture_capacity_probe_v1 import (
    DEFAULT_CACHE,
    DEFAULT_CANONICAL_ROOT,
    DEFAULT_CURATED,
    DEFAULT_SPLIT,
    NpyDataset,
    ScaledInkClassifier,
    _class_labels,
    _load_writer_map,
    _writer_digest,
)


DEFAULT_CHECKPOINT_DIR = DEFAULT_CACHE / "checkpoints"
DEFAULT_REPORT = ROOT / "artifacts" / "hwr_augmentation_microscope_20261002" / "architecture_scale_paired_rows_v2.json"


def _jsonl_train_uji_writer_hashes(canonical_root: Path, writer_map: dict[str, str], selected_hashes: set[str], label_set: set[str]) -> list[str]:
    result: list[str] = []
    for row in _json_lines(canonical_root / "uji.jsonl.gz"):
        if row.get("split") != "train" or str(row.get("label")) not in label_set:
            continue
        record_id = str(row["record_id"])
        writer_key = writer_map.get(record_id)
        if writer_key is None:
            raise ValueError(f"missing writer identity join: {record_id}")
        writer_hash = _writer_digest(writer_key)
        if writer_hash in selected_hashes:
            result.append(writer_hash)
    return result


@torch.inference_mode()
def _predict(checkpoint_path: Path, dataset: NpyDataset, labels: list[str], device: torch.device, batch_size: int) -> tuple[np.ndarray, np.ndarray, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != "aiflow-hwr-architecture-capacity-probe/v1":
        raise ValueError(f"unexpected checkpoint schema: {checkpoint_path}")
    architecture = checkpoint["arm"]
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
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(top1), np.concatenate(top5), {"architecture": architecture, "sha256": _sha256(checkpoint_path)}


def _mcnemar(recovered: int, regressed: int) -> dict:
    discordant = recovered + regressed
    if discordant == 0:
        return {"recovered": 0, "regressed": 0, "two_sided_exact_p": 1.0}
    tail = min(recovered, regressed)
    probability = Fraction(2 * sum(math.comb(discordant, index) for index in range(tail + 1)), 2**discordant)
    return {
        "recovered": recovered,
        "regressed": regressed,
        "two_sided_exact_p": min(1.0, float(probability)),
    }


def _paired(before: np.ndarray, after: np.ndarray) -> dict:
    if before.shape != after.shape:
        raise ValueError("paired metric arrays have different row counts")
    recovered = int((~before & after).sum())
    regressed = int((before & ~after).sum())
    return {
        "rows": int(len(before)),
        "before_hits": int(before.sum()),
        "after_hits": int(after.sum()),
        "before_rate": float(before.mean()) if len(before) else 0.0,
        "after_rate": float(after.mean()) if len(after) else 0.0,
        "delta_percentage_points": float((after.mean() - before.mean()) * 100.0) if len(before) else 0.0,
        **_mcnemar(recovered, regressed),
    }


def _writer_bootstrap(before: np.ndarray, after: np.ndarray, writer_hashes: list[str], seed: int, draws: int) -> dict:
    if len(before) != len(after) or len(before) != len(writer_hashes):
        raise ValueError("writer assignment is not aligned with validation predictions")
    groups: dict[str, list[int]] = defaultdict(list)
    for index, key in enumerate(writer_hashes):
        groups[key].append(index)
    keys = sorted(groups)
    if len(keys) != 8:
        raise ValueError(f"expected exactly eight validation writers, got {len(keys)}")
    delta_hits = np.asarray([int(after[groups[key]].sum()) - int(before[groups[key]].sum()) for key in keys], dtype=np.float64)
    row_counts = np.asarray([len(groups[key]) for key in keys], dtype=np.float64)
    per_writer = [
        {
            "writer_hash": key,
            "rows": int(row_counts[index]),
            "before_hits": int(before[groups[key]].sum()),
            "after_hits": int(after[groups[key]].sum()),
            "delta_percentage_points": float(delta_hits[index] / row_counts[index] * 100.0),
        }
        for index, key in enumerate(keys)
    ]
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        chosen = rng.integers(0, len(keys), size=len(keys))
        samples[draw] = delta_hits[chosen].sum() / row_counts[chosen].sum() * 100.0
    return {
        "writer_clusters": len(keys),
        "draws": draws,
        "seed": seed,
        "delta_percentage_points_percentile_95_interval": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
        "bootstrap_fraction_delta_positive": float((samples > 0).mean()),
        "per_writer": per_writer,
        "interpretation_limit": "only eight writers in one internal split; diagnostic uncertainty, not product acceptance",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path, default=DEFAULT_CANONICAL_ROOT)
    parser.add_argument("--curated", type=Path, default=DEFAULT_CURATED)
    parser.add_argument("--writer-split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--checkpoint-seed", type=int, default=20261002)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(f"refusing to overwrite paired audit: {args.report}")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    labels = _class_labels(args.canonical_root.resolve())
    dataset = NpyDataset(args.cache_root / "validation_features.npy", args.cache_root / "validation_labels.npy")
    split = json.loads(args.writer_split.read_text(encoding="utf-8"))
    selected = set(split["inner_split"]["validation_writer_hashes"])
    writer_map = _load_writer_map(args.curated.resolve())
    writer_hashes = _jsonl_train_uji_writer_hashes(args.canonical_root.resolve(), writer_map, selected, set(labels))
    if len(writer_hashes) != len(dataset):
        raise ValueError(f"validation writer mapping mismatch: {len(writer_hashes)} != {len(dataset)}")

    before1, before5, before_meta = _predict(args.checkpoint_dir / f"d128_l4_h4_seed{args.checkpoint_seed}.pt", dataset, labels, device, args.batch_size)
    after1, after5, after_meta = _predict(args.checkpoint_dir / f"d192_l4_h4_seed{args.checkpoint_seed}.pt", dataset, labels, device, args.batch_size)
    report = {
        "schema": "aiflow-hwr-architecture-capacity-paired-audit/v1",
        "status": "completed_exploratory_paired_writer_audit",
        "data": {
            "writer_split_sha256": _sha256(args.writer_split),
            "canonical_uji_sha256": _sha256(args.canonical_root / "uji.jsonl.gz"),
            "curated_uji_sha256": _sha256(args.curated),
            "validation_features_sha256": _sha256(args.cache_root / "validation_features.npy"),
            "validation_labels_sha256": _sha256(args.cache_root / "validation_labels.npy"),
            "rows": len(dataset),
            "writer_clusters": len(set(writer_hashes)),
            "checkpoint_seed": args.checkpoint_seed,
            "crohme_rows": 0,
        },
        "arms": {"d128_l4_h4": before_meta, "d192_l4_h4": after_meta},
        "paired_comparison": {
            "top1": _paired(before1, after1),
            "top5": _paired(before5, after5),
            "top1_writer_cluster_bootstrap": _writer_bootstrap(before1, after1, writer_hashes, args.seed, args.bootstrap_draws),
            "top5_writer_cluster_bootstrap": _writer_bootstrap(before5, after5, writer_hashes, args.seed, args.bootstrap_draws),
        },
        "product_adopted": False,
        "interpretation_limit": "single seed, one internal 8-writer split, 66 UJI labels overlapping the 372-class math vocabulary; not full-vocabulary acceptance",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({
        "event": "architecture_paired_writer_audit_complete",
        "report": str(args.report.resolve()),
        "top1": report["paired_comparison"]["top1"],
        "top1_writer_bootstrap": report["paired_comparison"]["top1_writer_cluster_bootstrap"]["delta_percentage_points_percentile_95_interval"],
        "top5_writer_bootstrap": report["paired_comparison"]["top5_writer_cluster_bootstrap"]["delta_percentage_points_percentile_95_interval"],
        "top5": report["paired_comparison"]["top5"],
        "crohme_rows": 0,
        "product_adopted": False,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
