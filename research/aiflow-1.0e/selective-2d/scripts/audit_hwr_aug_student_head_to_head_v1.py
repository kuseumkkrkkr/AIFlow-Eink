#!/usr/bin/env python3
"""Paired post-hoc comparison of two frozen augmentation-distilled HWR students."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from run_hwr_affine_distillation_experiment_v1 import (
    SOURCE_IDS,
    _family,
    _assert_comparison_sources_eligible,
    _load_cache,
    _load_teacher,
    _predict_logits,
    _sha256,
    _write_json,
)


def _counts(before: np.ndarray, after: np.ndarray, mask: np.ndarray) -> dict[str, int | float]:
    left = before[mask]
    right = after[mask]
    recovered = int((~left & right).sum())
    regressed = int((left & ~right).sum())
    return {
        "rows": int(mask.sum()),
        "before_hits": int(left.sum()),
        "after_hits": int(right.sum()),
        "before_accuracy": float(left.mean()) if len(left) else 0.0,
        "after_accuracy": float(right.mean()) if len(right) else 0.0,
        "recovered": recovered,
        "regressed": regressed,
        "net_rows": recovered - regressed,
    }


def _exact_mcnemar_p(recovered: int, regressed: int) -> float:
    discordant = recovered + regressed
    if not discordant:
        return 1.0
    tail = min(recovered, regressed)
    probability = sum(math.comb(discordant, index) for index in range(tail + 1)) / (2 ** discordant)
    return min(1.0, 2.0 * probability)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--before", type=Path, required=True, help="baseline compatible checkpoint")
    parser.add_argument("--after", type=Path, required=True, help="challenger compatible checkpoint")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    manifest_path = args.data_dir / "prepared_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _assert_comparison_sources_eligible(
        manifest.get("input_policy", {}),
        event="posthoc_student_head_to_head",
        source_split_rows=manifest.get("source_audit", {}).get("source_split_rows", {}),
    )

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    before_model, before_labels, _ = _load_teacher(args.before, device)
    after_model, after_labels, _ = _load_teacher(args.after, device)
    if before_labels != after_labels:
        raise ValueError("checkpoint class order differs")
    cache = _load_cache(args.data_dir, "test")
    features = cache["features"]
    targets = np.asarray(cache["labels"], dtype=np.int64)
    sources = np.asarray(cache["sources"], dtype=np.int8)
    before_logits = _predict_logits(before_model, features, device, args.batch_size)
    after_logits = _predict_logits(after_model, features, device, args.batch_size)
    before_top1 = np.argmax(before_logits, axis=1)
    after_top1 = np.argmax(after_logits, axis=1)
    before_top5 = np.argpartition(-before_logits, kth=4, axis=1)[:, :5]
    after_top5 = np.argpartition(-after_logits, kth=4, axis=1)[:, :5]
    before_correct = before_top1 == targets
    after_correct = after_top1 == targets
    before_in_top5 = np.any(before_top5 == targets[:, None], axis=1)
    after_in_top5 = np.any(after_top5 == targets[:, None], axis=1)

    by_source: dict[str, Any] = {}
    for name, source_id in SOURCE_IDS.items():
        if name == "synthetic_equal":
            continue
        by_source[name] = {
            "top1": _counts(before_correct, after_correct, sources == source_id),
            "top5": _counts(before_in_top5, after_in_top5, sources == source_id),
        }

    family_indices: dict[str, list[int]] = defaultdict(list)
    label_indices: dict[str, list[int]] = defaultdict(list)
    for row, class_id in enumerate(targets.tolist()):
        label = before_labels[class_id]
        family_indices[_family(label)].append(row)
        label_indices[label].append(row)
    by_family = {
        family: {
            "top1": _counts(before_correct, after_correct, np.isin(np.arange(len(targets)), indices)),
            "top5": _counts(before_in_top5, after_in_top5, np.isin(np.arange(len(targets)), indices)),
        }
        for family, indices in sorted(family_indices.items())
    }
    by_label = {
        label: {
            "rows": len(indices),
            "top1": _counts(before_correct, after_correct, np.isin(np.arange(len(targets)), indices)),
            "top5": _counts(before_in_top5, after_in_top5, np.isin(np.arange(len(targets)), indices)),
        }
        for label, indices in sorted(label_indices.items())
    }
    top1 = _counts(before_correct, after_correct, np.ones(len(targets), dtype=bool))
    top5 = _counts(before_in_top5, after_in_top5, np.ones(len(targets), dtype=bool))
    report = {
        "status": "completed_posthoc_paired_diagnostic",
        "rows": len(targets),
        "before_checkpoint": {"path": str(args.before.resolve()), "sha256": _sha256(args.before)},
        "after_checkpoint": {"path": str(args.after.resolve()), "sha256": _sha256(args.after)},
        "top1": {**top1, "exact_mcnemar_two_sided_p": _exact_mcnemar_p(top1["recovered"], top1["regressed"])},
        "top5": {**top5, "exact_mcnemar_two_sided_p": _exact_mcnemar_p(top5["recovered"], top5["regressed"])},
        "by_source": by_source,
        "by_family": by_family,
        "by_label": by_label,
        "split_policy": "previously consumed HWRT curated test + UJI official writer-heldout test; post-hoc diagnostic only",
        "crohme_rows": 0,
        "heldout_used_for_selection": False,
        "product_adopted": False,
        "interpretation_limit": "symbol-level comparison only; not fresh-writer/device or formula-exact acceptance",
    }
    _write_json(args.report, report)
    print({"event": "student_head_to_head_complete", "rows": len(targets), "top1": report["top1"], "top5": report["top5"], "report": str(args.report.resolve())})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
