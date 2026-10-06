#!/usr/bin/env python3
"""Class/rank microscope for the paired 128-vs-192 scratch HWR checkpoints."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from build_normalized_ink_v1 import _sha256
from audit_hwr_architecture_capacity_paired_v1 import (
    DEFAULT_REPORT as PAIRED_REPORT,
    _jsonl_train_uji_writer_hashes,
    _load_writer_map,
)
from run_hwr_architecture_capacity_probe_v1 import (
    DEFAULT_CACHE,
    DEFAULT_CANONICAL_ROOT,
    DEFAULT_CURATED,
    DEFAULT_SPLIT,
    NpyDataset,
    ScaledInkClassifier,
    _class_labels,
)


DEFAULT_CACHE = Path(r"D:\AIFlow-Workspace\PrivateData\math-ink-data-collector\derived\hwr-architecture-20261002\uji-writer-inner-scratch-cache-r3")
DEFAULT_CHECKPOINT_DIR = DEFAULT_CACHE / "checkpoints"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "artifacts" / "hwr_augmentation_microscope_20261002" / "architecture_scale_classwise_rankshift_20261003_v2.json"


@torch.inference_mode()
def _top5(checkpoint_path: Path, dataset: NpyDataset, class_count: int, device: torch.device, batch_size: int) -> np.ndarray:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != "aiflow-hwr-architecture-capacity-probe/v1":
        raise ValueError(f"unexpected checkpoint schema: {checkpoint_path}")
    arm = checkpoint["arm"]
    model = ScaledInkClassifier(
        class_count, int(arm["width"]), int(arm["layers"]), int(arm["heads"]), int(arm["feedforward"])
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    output: list[np.ndarray] = []
    for features, _ in loader:
        logits = model(features.to(device, non_blocking=device.type == "cuda")).float()
        output.append(logits.topk(5, dim=1).indices.cpu().numpy())
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(output, axis=0)


def _rank_counts(topk: np.ndarray, targets: np.ndarray) -> dict[str, int]:
    matches = topk == targets[:, None]
    found = matches.any(axis=1)
    ranks = np.where(found, matches.argmax(axis=1) + 1, 6)
    return {
        "top1": int((ranks == 1).sum()),
        "ranks2to5": int(((ranks >= 2) & (ranks <= 5)).sum()),
        "outside_top5": int((ranks == 6).sum()),
    }


def _class_metrics(before: np.ndarray, after: np.ndarray, targets: np.ndarray, labels: list[str]) -> list[dict]:
    rows: list[dict] = []
    for class_id in sorted(np.unique(targets).tolist()):
        mask = targets == class_id
        y = targets[mask]
        b, a = before[mask], after[mask]
        bh, ah = b[:, 0] == y, a[:, 0] == y
        b5, a5 = (b == y[:, None]).any(axis=1), (a == y[:, None]).any(axis=1)
        confusion_before = Counter(labels[int(pred)] for pred in b[~bh, 0])
        confusion_after = Counter(labels[int(pred)] for pred in a[~ah, 0])
        rows.append({
            "label": labels[int(class_id)],
            "support": int(mask.sum()),
            "top1_before": int(bh.sum()),
            "top1_after": int(ah.sum()),
            "top1_recovered": int((~bh & ah).sum()),
            "top1_regressed": int((bh & ~ah).sum()),
            "top5_before": int(b5.sum()),
            "top5_after": int(a5.sum()),
            "top5_recovered": int((~b5 & a5).sum()),
            "top5_regressed": int((b5 & ~a5).sum()),
            "target_rank_before": _rank_counts(b, y),
            "target_rank_after": _rank_counts(a, y),
            "top1_confusions_before": dict(confusion_before.most_common(5)),
            "top1_confusions_after": dict(confusion_after.most_common(5)),
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path, default=DEFAULT_CANONICAL_ROOT)
    parser.add_argument("--curated", type=Path, default=DEFAULT_CURATED)
    parser.add_argument("--writer-split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--paired-report", type=Path, default=PAIRED_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--checkpoint-seed", type=int, default=20261002)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite classwise report: {args.output}")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")

    labels = _class_labels(args.canonical_root.resolve())
    split = json.loads(args.writer_split.read_text(encoding="utf-8"))
    paired = json.loads(args.paired_report.read_text(encoding="utf-8"))
    if paired.get("data", {}).get("crohme_rows") != 0 or int(paired.get("data", {}).get("rows", 0)) < 1:
        raise ValueError("paired report is not a valid CROHME-free writer audit")
    paired_inputs = paired["data"]
    input_paths = {
        "writer_split_sha256": args.writer_split.resolve(),
        "canonical_uji_sha256": args.canonical_root.resolve() / "uji.jsonl.gz",
        "curated_uji_sha256": args.curated.resolve(),
        "validation_features_sha256": args.cache_root / "validation_features.npy",
        "validation_labels_sha256": args.cache_root / "validation_labels.npy",
    }
    for name, path in input_paths.items():
        if _sha256(path) != paired_inputs.get(name):
            raise ValueError(f"input hash differs from paired audit: {name}")
    checkpoint_paths = {
        "d128_l4_h4": args.checkpoint_dir / f"d128_l4_h4_seed{args.checkpoint_seed}.pt",
        "d192_l4_h4": args.checkpoint_dir / f"d192_l4_h4_seed{args.checkpoint_seed}.pt",
    }
    for name, path in checkpoint_paths.items():
        if _sha256(path) != paired["arms"][name]["sha256"]:
            raise ValueError(f"checkpoint hash differs from paired audit: {name}")
    validation_hashes = set(split["inner_split"]["validation_writer_hashes"])
    writer_map = _load_writer_map(args.curated.resolve())
    writer_hashes = _jsonl_train_uji_writer_hashes(args.canonical_root.resolve(), writer_map, validation_hashes, set(labels))
    dataset = NpyDataset(args.cache_root / "validation_features.npy", args.cache_root / "validation_labels.npy")
    if len(writer_hashes) != len(dataset) or len(set(writer_hashes)) != int(paired_inputs.get("writer_clusters", -1)):
        raise ValueError("validation rows/writer clusters do not match the paired audit")

    device = torch.device(args.device)
    before = _top5(checkpoint_paths["d128_l4_h4"], dataset, len(labels), device, args.batch_size)
    after = _top5(checkpoint_paths["d192_l4_h4"], dataset, len(labels), device, args.batch_size)
    targets = np.asarray(dataset.labels, dtype=np.int64)
    if before.shape != after.shape or before.shape != (len(dataset), 5) or len(targets) != len(dataset):
        raise AssertionError("paired Top-5 predictions are not aligned")

    class_rows = _class_metrics(before, after, targets, labels)
    overall = {
        "top1_before": int((before[:, 0] == targets).sum()),
        "top1_after": int((after[:, 0] == targets).sum()),
        "top5_before": int((before == targets[:, None]).any(axis=1).sum()),
        "top5_after": int((after == targets[:, None]).any(axis=1).sum()),
        "top1_predictions_changed": int((before[:, 0] != after[:, 0]).sum()),
        "targets_entering_top5": int((~(before == targets[:, None]).any(axis=1) & (after == targets[:, None]).any(axis=1)).sum()),
        "targets_leaving_top5": int(((before == targets[:, None]).any(axis=1) & ~(after == targets[:, None]).any(axis=1)).sum()),
        "target_rank_before": _rank_counts(before, targets),
        "target_rank_after": _rank_counts(after, targets),
    }
    paired_expected = {
        "top1_before": paired["paired_comparison"]["top1"]["before_hits"],
        "top1_after": paired["paired_comparison"]["top1"]["after_hits"],
        "top5_before": paired["paired_comparison"]["top5"]["before_hits"],
        "top5_after": paired["paired_comparison"]["top5"]["after_hits"],
    }
    if any(overall[key] != int(value) for key, value in paired_expected.items()):
        raise AssertionError("classwise replay does not reproduce paired aggregate metrics")
    if sum(row["support"] for row in class_rows) != len(dataset):
        raise AssertionError("class supports do not cover validation rows")
    if any(sum(row["target_rank_before"].values()) != row["support"] or sum(row["target_rank_after"].values()) != row["support"] for row in class_rows):
        raise AssertionError("per-class Top-5 rank counts do not sum to support")

    report = {
        "schema": "aiflow-hwr-architecture-capacity-classwise-rankshift/v1",
        "status": "completed_exploratory_posthoc_audit",
        "protocol": {
            "training_performed": False,
            "model_selection_performed": False,
            "crohme_rows_loaded": 0,
            "product_adopted": False,
            "warning": "post-hoc class breakdown of one consumed internal 8-writer split; 66 of 372 labels are represented and this is not product acceptance",
        },
        "provenance": {
            "paired_report_sha256": _sha256(args.paired_report.resolve()),
            "writer_split_sha256": _sha256(args.writer_split.resolve()),
            "canonical_uji_sha256": _sha256(args.canonical_root.resolve() / "uji.jsonl.gz"),
            "validation_features_sha256": _sha256(args.cache_root / "validation_features.npy"),
            "validation_labels_sha256": _sha256(args.cache_root / "validation_labels.npy"),
            "baseline_checkpoint_sha256": _sha256(checkpoint_paths["d128_l4_h4"]),
            "larger_checkpoint_sha256": _sha256(checkpoint_paths["d192_l4_h4"]),
            "rows": len(dataset),
            "writer_clusters": len(set(writer_hashes)),
            "checkpoint_seed": args.checkpoint_seed,
            "present_labels": int(len(np.unique(targets))),
        },
        "overall": overall,
        "classes": class_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"event": "architecture_capacity_classwise_audit_complete", "report": str(args.output.resolve()), "overall": report["overall"], "classes": len(report["classes"]), "crohme_rows": 0, "product_adopted": False}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
