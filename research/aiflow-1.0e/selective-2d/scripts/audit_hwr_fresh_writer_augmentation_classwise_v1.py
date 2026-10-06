#!/usr/bin/env python3
"""Classwise microscope for a completed fresh-writer augmentation A/B report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from audit_hwr_architecture_capacity_paired_v1 import _predict
from build_normalized_ink_v1 import ROOT, _sha256
from run_hwr_architecture_capacity_probe_v1 import DEFAULT_CANONICAL_ROOT, NpyDataset, _class_labels


DEFAULT_INPUT = ROOT / "artifacts" / "hwr_augmentation_microscope_20261002" / "fresh_writer_augmentation_ab_seed20261088.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "hwr_augmentation_microscope_20261002" / "fresh_writer_augmentation_ab_seed20261088_classwise.json"
DEFAULT_CACHE = Path(
    r"D:\AIFlow-Workspace\PrivateData\math-ink-data-collector\derived"
    r"\hwr-augmentation-20261002\fresh-writer-paired-cache-seed20261088"
)


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
        raise FileExistsError(f"refusing to overwrite classwise report: {args.output}")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")

    source_path = args.input.resolve()
    report = json.loads(source_path.read_text(encoding="utf-8"))
    if report.get("schema") != "aiflow-hwr-fresh-writer-augmentation-ab/v1":
        raise ValueError("unexpected A/B report schema")
    if report.get("status") != "completed_exploratory_paired_writer_audit":
        raise ValueError("classwise audit requires a completed paired run")
    data = report["data"]
    if data.get("crohme_rows") != 0 or data.get("prior_validation_writer_overlap") != 0:
        raise ValueError("input report fails the CROHME/fresh-writer boundary")

    canonical_root = args.canonical_root.resolve()
    if _sha256(canonical_root / "hwrt.jsonl.gz") != data["canonical_hwrt_sha256"]:
        raise ValueError("canonical HWRT source hash changed")
    if _sha256(canonical_root / "uji.jsonl.gz") != data["canonical_uji_sha256"]:
        raise ValueError("canonical UJI source hash changed")
    labels = _class_labels(canonical_root)
    cache_root = args.cache_root.resolve()
    for filename, expected in data["cache_file_sha256"].items():
        if _sha256(cache_root / filename) != expected:
            raise ValueError(f"validation/train cache hash changed: {filename}")

    validation = NpyDataset(cache_root / "validation_features.npy", cache_root / "validation_labels.npy")
    validation_targets = np.asarray(validation.labels, dtype=np.int64)
    if len(validation) != data["rows"]["validation"]:
        raise ValueError("validation cache row count changed")
    device = torch.device(args.device)
    arms = report["arms"]
    for arm in arms:
        checkpoint = Path(arm["checkpoint"])
        if _sha256(checkpoint) != arm["checkpoint_sha256"]:
            raise ValueError(f"checkpoint hash changed: {checkpoint}")
    before1, before5, _ = _predict(Path(arms[0]["checkpoint"]), validation, labels, device, args.batch_size)
    after1, after5, _ = _predict(Path(arms[1]["checkpoint"]), validation, labels, device, args.batch_size)

    per_class = []
    for class_id in sorted(set(validation_targets.tolist())):
        selected = validation_targets == class_id
        row = {
            "label": labels[class_id],
            "support": int(selected.sum()),
            "control_top1_hits": int(before1[selected].sum()),
            "augmented_top1_hits": int(after1[selected].sum()),
            "top1_delta_hits": int(after1[selected].sum() - before1[selected].sum()),
            "top1_recovered": int((~before1[selected] & after1[selected]).sum()),
            "top1_regressed": int((before1[selected] & ~after1[selected]).sum()),
            "control_top5_hits": int(before5[selected].sum()),
            "augmented_top5_hits": int(after5[selected].sum()),
            "top5_delta_hits": int(after5[selected].sum() - before5[selected].sum()),
            "top5_recovered": int((~before5[selected] & after5[selected]).sum()),
            "top5_regressed": int((before5[selected] & ~after5[selected]).sum()),
        }
        per_class.append(row)
    totals = {
        "rows": len(validation),
        "present_classes": len(per_class),
        "control_top1_hits": int(before1.sum()),
        "augmented_top1_hits": int(after1.sum()),
        "top1_recovered": int((~before1 & after1).sum()),
        "top1_regressed": int((before1 & ~after1).sum()),
        "control_top5_hits": int(before5.sum()),
        "augmented_top5_hits": int(after5.sum()),
        "top5_recovered": int((~before5 & after5).sum()),
        "top5_regressed": int((before5 & ~after5).sum()),
    }
    if totals["control_top1_hits"] != report["paired_comparison"]["top1"]["before_hits"] or totals["augmented_top1_hits"] != report["paired_comparison"]["top1"]["after_hits"]:
        raise AssertionError("classwise Top-1 totals disagree with the paired report")
    if totals["control_top5_hits"] != report["paired_comparison"]["top5"]["before_hits"] or totals["augmented_top5_hits"] != report["paired_comparison"]["top5"]["after_hits"]:
        raise AssertionError("classwise Top-5 totals disagree with the paired report")

    output = {
        "schema": "aiflow-hwr-fresh-writer-augmentation-classwise/v1",
        "status": "consumed_internal_validation_diagnostic",
        "input_report": str(source_path),
        "input_report_sha256": _sha256(source_path),
        "protocol": {
            "training_performed": False,
            "model_or_augmentation_selection_performed": False,
            "crohme_rows": 0,
            "official_uji_test_scored": False,
            "writer_clusters": len(data["validation_writer_hashes"]),
            "writer_overlap_with_previous_probes": 0,
            "interpretation_limit": "one consumed eight-writer internal train split, 66 validation labels; not independent product acceptance",
        },
        "totals": totals,
        "largest_top1_regressions": sorted((row for row in per_class if row["top1_delta_hits"] < 0), key=lambda row: row["top1_delta_hits"])[:12],
        "largest_top1_recoveries": sorted((row for row in per_class if row["top1_delta_hits"] > 0), key=lambda row: row["top1_delta_hits"], reverse=True)[:12],
        "per_class": per_class,
    }
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({
        "event": "fresh_writer_aug_classwise_audit_complete",
        "output": str(output_path),
        "rows": totals["rows"],
        "classes": totals["present_classes"],
        "top1_delta_hits": totals["augmented_top1_hits"] - totals["control_top1_hits"],
        "top5_delta_hits": totals["augmented_top5_hits"] - totals["control_top5_hits"],
        "crohme_rows": 0,
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
