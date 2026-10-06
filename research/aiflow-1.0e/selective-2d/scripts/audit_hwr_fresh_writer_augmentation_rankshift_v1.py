#!/usr/bin/env python3
"""Rank-shift microscope for the completed augmentation A/B validation cohort."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from build_normalized_ink_v1 import ROOT, _sha256
from run_hwr_architecture_capacity_probe_v1 import DEFAULT_CANONICAL_ROOT, NpyDataset, ScaledInkClassifier, _class_labels


DEFAULT_INPUT = ROOT / "artifacts" / "hwr_augmentation_microscope_20261002" / "fresh_writer_augmentation_ab_seed20261088.json"
DEFAULT_CACHE = Path(
    r"D:\AIFlow-Workspace\PrivateData\math-ink-data-collector\derived"
    r"\hwr-augmentation-20261002\fresh-writer-paired-cache-seed20261088"
)
DEFAULT_OUTPUT = ROOT / "artifacts" / "hwr_augmentation_microscope_20261002" / "fresh_writer_augmentation_ab_seed20261088_rankshift.json"


@torch.inference_mode()
def _predict_targets(checkpoint_path: Path, dataset: NpyDataset, device: torch.device, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != "aiflow-hwr-architecture-capacity-probe/v1":
        raise ValueError(f"unexpected checkpoint schema: {checkpoint_path}")
    arm = checkpoint["arm"]
    model = ScaledInkClassifier(372, int(arm["width"]), int(arm["layers"]), int(arm["heads"]), int(arm["feedforward"]))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    ranks: list[np.ndarray] = []
    top1_ids: list[np.ndarray] = []
    for features, target in loader:
        logits = model(features.to(device, non_blocking=device.type == "cuda")).float()
        top = logits.topk(5, dim=1).indices
        target_on_device = target.to(device, non_blocking=device.type == "cuda")
        target_rank = torch.full_like(target_on_device, 6)
        for rank in range(5):
            target_rank = torch.where(top[:, rank] == target_on_device, rank + 1, target_rank)
        ranks.append(target_rank.cpu().numpy())
        top1_ids.append(top[:, 0].cpu().numpy())
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(ranks), np.concatenate(top1_ids)


def _transition_table(before: np.ndarray, after: np.ndarray) -> list[dict[str, int]]:
    counts = Counter(zip(before.tolist(), after.tolist()))
    return [
        {"before_target_rank": left, "after_target_rank": right, "rows": count}
        for (left, right), count in sorted(counts.items())
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--canonical-root", type=Path, default=DEFAULT_CANONICAL_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite rank-shift report: {args.output}")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")

    input_path = args.input.resolve()
    ab = json.loads(input_path.read_text(encoding="utf-8"))
    if ab.get("schema") != "aiflow-hwr-fresh-writer-augmentation-ab/v1" or ab.get("status") != "completed_exploratory_paired_writer_audit":
        raise ValueError("rank-shift audit requires a completed augmentation A/B report")
    data = ab["data"]
    if data["crohme_rows"] != 0 or data["prior_validation_writer_overlap"] != 0:
        raise ValueError("input violates the CROHME/fresh-writer boundary")
    canonical_root, cache_root = args.canonical_root.resolve(), args.cache_root.resolve()
    if _sha256(canonical_root / "hwrt.jsonl.gz") != data["canonical_hwrt_sha256"] or _sha256(canonical_root / "uji.jsonl.gz") != data["canonical_uji_sha256"]:
        raise ValueError("canonical source hash differs from the completed experiment")
    for name, expected in data["cache_file_sha256"].items():
        if _sha256(cache_root / name) != expected:
            raise ValueError(f"cache changed since paired run: {name}")
    labels = _class_labels(canonical_root)
    validation = NpyDataset(cache_root / "validation_features.npy", cache_root / "validation_labels.npy")
    targets = np.asarray(validation.labels, dtype=np.int64)
    device = torch.device(args.device)

    target_ranks: list[np.ndarray] = []
    top1_predictions: list[np.ndarray] = []
    for arm in ab["arms"]:
        checkpoint = Path(arm["checkpoint"])
        if _sha256(checkpoint) != arm["checkpoint_sha256"]:
            raise ValueError(f"checkpoint hash differs from the completed experiment: {checkpoint}")
        ranks, predictions = _predict_targets(checkpoint, validation, device, args.batch_size)
        target_ranks.append(ranks)
        top1_predictions.append(predictions)
    before_rank, after_rank = target_ranks
    before_top1, after_top1 = top1_predictions
    if int((before_rank == 1).sum()) != ab["paired_comparison"]["top1"]["before_hits"]:
        raise AssertionError("control Top-1 rank count disagrees with A/B report")
    if int((after_rank == 1).sum()) != ab["paired_comparison"]["top1"]["after_hits"]:
        raise AssertionError("augmented Top-1 rank count disagrees with A/B report")
    if int((before_rank <= 5).sum()) != ab["paired_comparison"]["top5"]["before_hits"]:
        raise AssertionError("control Top-5 rank count disagrees with A/B report")
    if int((after_rank <= 5).sum()) != ab["paired_comparison"]["top5"]["after_hits"]:
        raise AssertionError("augmented Top-5 rank count disagrees with A/B report")

    confusion_changes = Counter()
    per_class: list[dict[str, Any]] = []
    for class_id in sorted(set(targets.tolist())):
        selected = targets == class_id
        per_class.append({
            "target": labels[class_id],
            "support": int(selected.sum()),
            "control_top1": int((before_rank[selected] == 1).sum()),
            "augmented_top1": int((after_rank[selected] == 1).sum()),
            "top1_delta_hits": int((after_rank[selected] == 1).sum() - (before_rank[selected] == 1).sum()),
            "control_target_in_top5": int((before_rank[selected] <= 5).sum()),
            "augmented_target_in_top5": int((after_rank[selected] <= 5).sum()),
            "target_in_top5_delta_hits": int((after_rank[selected] <= 5).sum() - (before_rank[selected] <= 5).sum()),
        })
    for index in np.flatnonzero(before_top1 != after_top1):
        confusion_changes[(labels[int(targets[index])], labels[int(before_top1[index])], labels[int(after_top1[index])])] += 1

    top1_recovered = (before_rank != 1) & (after_rank == 1)
    top1_regressed = (before_rank == 1) & (after_rank != 1)
    top5_recovered = (before_rank == 6) & (after_rank <= 5)
    top5_regressed = (before_rank <= 5) & (after_rank == 6)
    output = {
        "schema": "aiflow-hwr-fresh-writer-augmentation-rankshift/v1",
        "status": "consumed_internal_validation_diagnostic",
        "input_report": str(input_path),
        "input_report_sha256": _sha256(input_path),
        "protocol": {
            "training_performed": False,
            "model_or_augmentation_selection_performed": False,
            "official_uji_test_scored": False,
            "crohme_rows": 0,
            "writer_clusters": len(data["validation_writer_hashes"]),
            "present_classes": len(set(targets.tolist())),
            "interpretation_limit": "post-hoc rank analysis of the same consumed eight-writer validation cohort; not independent acceptance",
        },
        "summary": {
            "rows": len(validation),
            "target_rank_1_to_6_before_after": _transition_table(before_rank, after_rank),
            "top1_recovered_rows": int(top1_recovered.sum()),
            "top1_regressed_rows": int(top1_regressed.sum()),
            "top1_regressions_still_in_top5": int((top1_regressed & (after_rank <= 5)).sum()),
            "top1_recoveries_from_top5": int((top1_recovered & (before_rank <= 5)).sum()),
            "target_entered_top5": int(top5_recovered.sum()),
            "target_left_top5": int(top5_regressed.sum()),
            "top1_changed_rows": int((before_top1 != after_top1).sum()),
            "most_common_changed_confusions": [
                {"target": target, "control_prediction": before, "augmented_prediction": after, "rows": count}
                for (target, before, after), count in confusion_changes.most_common(30)
            ],
        },
        "largest_top1_regressions": sorted((row for row in per_class if row["top1_delta_hits"] < 0), key=lambda row: row["top1_delta_hits"])[:15],
        "largest_top1_recoveries": sorted((row for row in per_class if row["top1_delta_hits"] > 0), key=lambda row: row["top1_delta_hits"], reverse=True)[:15],
    }
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({
        "event": "fresh_writer_aug_rankshift_audit_complete",
        "output": str(output_path),
        "top1_regressed_still_in_top5": output["summary"]["top1_regressions_still_in_top5"],
        "target_entered_top5": output["summary"]["target_entered_top5"],
        "target_left_top5": output["summary"]["target_left_top5"],
        "crohme_rows": 0,
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
