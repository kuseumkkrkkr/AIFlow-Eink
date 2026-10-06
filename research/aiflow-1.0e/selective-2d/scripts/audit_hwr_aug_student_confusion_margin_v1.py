#!/usr/bin/env python3
"""Compare teacher/student logit margins on known confusion pairs.

Training rows are diagnostic for fit behavior. The already-consumed external
test split is post-hoc diagnosis only and must not be treated as acceptance.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from run_hwr_affine_distillation_experiment_v1 import (
    DEFAULT_CHECKPOINT,
    DEFAULT_WORK_DIR,
    ROOT,
    SCHEMA,
    SOURCE_IDS,
    _assert_comparison_sources_eligible,
    _load_cache,
    _load_teacher,
    _predict_logits,
    _sha256,
    _write_json,
)


DEFAULT_STUDENT = Path(
    r"D:\AIFlow-Workspace\PrivateData\math-ink-data-collector\derived"
    r"\hwr-augmentation-20261001\affine-distill-v2-volume\student_checkpoint_compatible.pt"
)
DEFAULT_DATA = Path(
    r"D:\AIFlow-Workspace\PrivateData\math-ink-data-collector\derived"
    r"\hwr-augmentation-20261001\affine-distill-v1"
)
DEFAULT_ERROR_REPORT = ROOT / "artifacts" / "hwr_augmentation_microscope_20261001" / "affine_distill_v2_volume_checked" / "paired_class_microscope_detailed.json"
DEFAULT_REPORT = ROOT / "artifacts" / "hwr_augmentation_microscope_20261001" / "affine_distill_v2_volume_checked" / "confusion_margin_microscope.json"
SOURCE_NAMES = {value: key for key, value in SOURCE_IDS.items()}


def _pairs_from_report(path: Path, labels: list[str]) -> list[tuple[str, str]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    pairs: set[tuple[str, str]] = set()
    # Pair selection is explicitly post-hoc from the previously consumed
    # diagnostic split; it is for error analysis, not another model search.
    for row in report.get("top1_regressions_min_8_rows", []):
        target = str(row.get("label", ""))
        if target not in labels:
            continue
        for destination in row.get("top1_regression_destinations", []):
            rival = str(destination.get("label", ""))
            if rival in labels and rival != target:
                pairs.add((target, rival))
                pairs.add((rival, target))
    for target, rival in (("V", "v"), ("T", r"\top"), ("(", "C")):
        if target in labels and rival in labels:
            pairs.add((target, rival))
            pairs.add((rival, target))
    return sorted(pairs)


def _margin_summary(values: np.ndarray) -> dict[str, float | int]:
    if not len(values):
        return {"rows": 0}
    return {
        "rows": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p10": float(np.quantile(values, 0.10)),
        "p90": float(np.quantile(values, 0.90)),
        "positive_fraction": float(np.mean(values > 0.0)),
    }


def _score_pair(
    pair: tuple[str, str],
    labels: list[str],
    targets: np.ndarray,
    sources: np.ndarray,
    teacher_logits: np.ndarray,
    student_logits: np.ndarray,
) -> dict[str, Any]:
    target_label, rival_label = pair
    target_id, rival_id = labels.index(target_label), labels.index(rival_label)
    indices = np.flatnonzero(targets == target_id)
    teacher_pred = np.argmax(teacher_logits[indices], axis=1)
    student_pred = np.argmax(student_logits[indices], axis=1)
    teacher_margin = teacher_logits[indices, target_id] - teacher_logits[indices, rival_id]
    student_margin = student_logits[indices, target_id] - student_logits[indices, rival_id]
    teacher_top5 = np.argpartition(-teacher_logits[indices], kth=4, axis=1)[:, :5]
    student_top5 = np.argpartition(-student_logits[indices], kth=4, axis=1)[:, :5]
    target_ranks_teacher = np.argsort(np.argsort(-teacher_logits[indices], axis=1), axis=1)[:, target_id] + 1
    target_ranks_student = np.argsort(np.argsort(-student_logits[indices], axis=1), axis=1)[:, target_id] + 1
    teacher_correct = teacher_pred == target_id
    student_correct = student_pred == target_id
    source_rows = {}
    for source_id, source_name in SOURCE_NAMES.items():
        local = sources[indices] == source_id
        if not local.any():
            continue
        source_rows[source_name] = {
            "rows": int(local.sum()),
            "teacher_target_top1": int(teacher_correct[local].sum()),
            "student_target_top1": int(student_correct[local].sum()),
            "teacher_target_vs_rival_margin": _margin_summary(teacher_margin[local]),
            "student_target_vs_rival_margin": _margin_summary(student_margin[local]),
        }
    return {
        "target": target_label,
        "rival": rival_label,
        "rows": int(len(indices)),
        "teacher_target_top1": int(teacher_correct.sum()),
        "student_target_top1": int(student_correct.sum()),
        "teacher_rival_top1": int(np.sum(teacher_pred == rival_id)),
        "student_rival_top1": int(np.sum(student_pred == rival_id)),
        "teacher_target_top5": int(np.any(teacher_top5 == target_id, axis=1).sum()),
        "student_target_top5": int(np.any(student_top5 == target_id, axis=1).sum()),
        "teacher_target_rank_median": float(np.median(target_ranks_teacher)) if len(indices) else None,
        "student_target_rank_median": float(np.median(target_ranks_student)) if len(indices) else None,
        "teacher_target_vs_rival_margin": _margin_summary(teacher_margin),
        "student_target_vs_rival_margin": _margin_summary(student_margin),
        "paired_margin_delta_student_minus_teacher": _margin_summary(student_margin - teacher_margin),
        "teacher_correct_to_student_wrong": int(np.sum(teacher_correct & ~student_correct)),
        "teacher_wrong_to_student_correct": int(np.sum(~teacher_correct & student_correct)),
        "by_source": source_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--student", type=Path, default=DEFAULT_STUDENT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--error-report", type=Path, default=DEFAULT_ERROR_REPORT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.report.exists():
        raise FileExistsError(f"refusing to overwrite report: {args.report}")

    manifest_path = args.data_dir / "prepared_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _assert_comparison_sources_eligible(
        manifest.get("input_policy", {}),
        event="posthoc_confusion_margin_analysis",
        source_split_rows=manifest.get("source_audit", {}).get("source_split_rows", {}),
    )

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    teacher, labels, _ = _load_teacher(args.teacher, device)
    student, student_labels, _ = _load_teacher(args.student, device)
    if labels != student_labels:
        raise ValueError("teacher/student class order differs")
    pairs = _pairs_from_report(args.error_report, labels)
    if not pairs:
        raise ValueError("no valid confusion pairs found")

    results = {}
    for split in ("train", "test"):
        cache = _load_cache(args.data_dir, split)
        targets = np.asarray(cache["labels"], dtype=np.int64)
        sources = np.asarray(cache["sources"], dtype=np.int8)
        target_ids = sorted({labels.index(target) for target, _ in pairs})
        indices = np.flatnonzero(np.isin(targets, target_ids))
        features = np.asarray(cache["features"][indices], dtype=np.float32)
        teacher_logits = _predict_logits(teacher, features, device, args.batch_size)
        student_logits = _predict_logits(student, features, device, args.batch_size)
        pair_results = []
        for pair in pairs:
            row = _score_pair(
                pair,
                labels,
                targets[indices],
                sources[indices],
                teacher_logits,
                student_logits,
            )
            if row["rows"]:
                pair_results.append(row)
        results[split] = {
            "source_rows": int(len(targets)),
            "rows_scored": int(len(indices)),
            "pairs": pair_results,
        }

    report = {
        "schema": "aiflow-hwr-confusion-margin-microscope/v1",
        "status": "completed_diagnostic_only",
        "teacher_checkpoint_sha256": _sha256(args.teacher),
        "student_checkpoint_sha256": _sha256(args.student),
        "source_manifest_sha256": _sha256(args.data_dir / "prepared_manifest.json"),
        "pair_source": str(args.error_report.resolve()),
        "pair_selection_policy": "post-hoc destination pairs from a previously consumed diagnostic report plus known V/v, T/top, and (/C pairs",
        "test_policy": "previously consumed diagnostic split; not an untouched acceptance set and not used for training",
        "training_policy": "training split used only for fit-behavior diagnostics; no model update occurs",
        "crohme_rows": 0,
        "product_adopted": False,
        "pairs": [list(pair) for pair in pairs],
        "by_split": results,
    }
    _write_json(args.report, report)
    compact = {
        split: {"rows_scored": value["rows_scored"], "pairs_with_rows": len(value["pairs"])}
        for split, value in results.items()
    }
    print(json.dumps({"event": "confusion_margin_microscope_complete", "rows": compact,
                      "pairs": len(pairs), "report": str(args.report.resolve())}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
