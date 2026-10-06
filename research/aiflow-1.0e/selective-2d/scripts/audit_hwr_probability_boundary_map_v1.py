#!/usr/bin/env python3
"""Map frozen-teacher true-label margins across the full HWR training pool.

Diagnostic/mining aid only. The cache contains teacher predictions on training
rows, so these in-sample statistics must not be used as acceptance metrics or
as the sole signal for model/augmentation selection. No training, heldout, or
CROHME data is read.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = Path(
    r"D:\AIFlow-Workspace\PrivateData\math-ink-data-collector\derived"
    r"\hwr-augmentation-20261001\affine-distill-v1"
)
DEFAULT_CHECKPOINT = Path(
    r"D:\AIFlow-Workspace\PrivateData\math-ink-data-collector\derived"
    r"\hwr-head-calibration-20260819-r1\final_all_writers_steps250_lr1e-3"
    r"\project_symbol_head_checkpoint.pt"
)
DEFAULT_REPORT_DIR = ROOT / "artifacts" / "hwr_probability_boundary_map_20261004_r3"
CLASS_COUNT = 372
TOP_K = 5
TEMPERATURE = 2.0
CHUNK_ROWS = 4096
PROBES_PER_GROUP = 2
MIN_COMMIT_HEADROOM_GIB = 1.5
SOURCE_IDS = {"hwrt": 0, "uji": 1, "synthetic_equal": 2}
REAL_SOURCES = ((0, "hwrt"), (1, "uji"))
np: Any = None


def _load_numpy() -> Any:
    global np
    if np is None:
        import numpy as numpy_module
        np = numpy_module
    return np


def _available_commit_gib() -> float:
    if os.name != "nt":
        raise RuntimeError("memory guard supports Windows only")

    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_uint32), ("dwMemoryLoad", ctypes.c_uint32),
            ("ullTotalPhys", ctypes.c_uint64), ("ullAvailPhys", ctypes.c_uint64),
            ("ullTotalPageFile", ctypes.c_uint64), ("ullAvailPageFile", ctypes.c_uint64),
            ("ullTotalVirtual", ctypes.c_uint64), ("ullAvailVirtual", ctypes.c_uint64),
            ("ullAvailExtendedVirtual", ctypes.c_uint64),
        ]

    status = MemoryStatusEx()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("GlobalMemoryStatusEx failed")
    return float(status.ullAvailPageFile) / (1024.0 ** 3)


def _guard_commit(stage: str) -> float | None:
    try:
        available = _available_commit_gib()
    except (OSError, RuntimeError) as exc:
        reason, available = str(exc), None
    else:
        reason = "commit headroom is below the safety floor" if available < MIN_COMMIT_HEADROOM_GIB else ""
    if reason:
        print(json.dumps({
            "event": "probability_boundary_map_refused", "stage": stage,
            "reason": reason, "available_commit_headroom_gib": available,
            "minimum_commit_headroom_gib": MIN_COMMIT_HEADROOM_GIB,
            "numpy_loaded": np is not None,
        }), flush=True)
        return None
    return available


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _score_logits(logits: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, ...]:
    values = np.asarray(logits, dtype=np.float32)
    target_ids = np.asarray(targets, dtype=np.int64)
    if values.ndim != 2 or values.shape[1] != CLASS_COUNT or target_ids.shape != (len(values),):
        raise ValueError("expected aligned [N,372] logits and [N] target IDs")
    if not np.isfinite(values).all() or np.any(target_ids < 0) or np.any(target_ids >= CLASS_COUNT):
        raise ValueError("logits or target IDs are invalid")
    rows = np.arange(len(values))
    predictions = np.argmax(values, axis=1)
    rivals = values.copy()
    target_logits = rivals[rows, target_ids].copy()
    rivals[rows, target_ids] = -np.inf
    rival_ids = np.argmax(rivals, axis=1)
    margins = target_logits - rivals[rows, rival_ids]
    top5 = np.argpartition(-values, kth=TOP_K - 1, axis=1)[:, :TOP_K]
    top5_hit = np.any(top5 == target_ids[:, None], axis=1)
    scaled = values.astype(np.float64) / TEMPERATURE
    maximum = scaled.max(axis=1)
    exp_sum = np.exp(scaled - maximum[:, None]).sum(axis=1)
    true_probability = np.exp(scaled[rows, target_ids] - maximum) / exp_sum
    return predictions, rival_ids, margins, top5_hit, true_probability


def _describe(values: list[np.ndarray]) -> dict[str, float | int] | None:
    if not values:
        return None
    flat = np.concatenate(values).astype(np.float64, copy=False)
    return {
        "rows": int(len(flat)),
        "mean": float(flat.mean()),
        "p10": float(np.quantile(flat, 0.10)),
        "median": float(np.quantile(flat, 0.50)),
        "p90": float(np.quantile(flat, 0.90)),
    }


def _record_probe(
    row_index: int, class_id: int, predicted_id: int, source_name: str,
    margin: float, probability: float, stroke_count: int, kind: str,
) -> dict[str, Any]:
    return {
        "row_index": int(row_index), "class_id": int(class_id),
        "predicted_id": int(predicted_id), "source": source_name,
        "true_label_margin": float(margin),
        "teacher_softmax_true_probability_t2": float(probability),
        "stroke_count": int(stroke_count), "probe_kind": kind,
    }


def _update_probe_pool(
    pool: list[dict[str, Any]], candidates: list[dict[str, Any]], limit: int,
) -> list[dict[str, Any]]:
    unique = {(item["row_index"], item["probe_kind"]): item for item in (*pool, *candidates)}
    values = list(unique.values())
    near = sorted(
        (item for item in values if item["probe_kind"] == "near_zero"),
        key=lambda item: abs(item["true_label_margin"]),
    )[:PROBES_PER_GROUP]
    hard = sorted(
        (item for item in values if item["probe_kind"] == "hard_miss"),
        key=lambda item: item["true_label_margin"],
    )[:1]
    return near + hard[:max(0, limit - len(near))]


def _draw_probe_sheet(
    features: np.ndarray, probes: list[dict[str, Any]], labels: list[str],
    class_families: list[str], path: Path,
) -> dict[str, Any]:
    from PIL import Image, ImageDraw, ImageFont

    by_family: dict[str, list[dict[str, Any]]] = {}
    for probe in probes:
        by_family.setdefault(class_families[probe["class_id"]], []).append(probe)
    visual: list[dict[str, Any]] = []
    for family in sorted(by_family):
        items = by_family[family]
        wrong = sorted(
            (item for item in items if item["probe_kind"] == "hard_miss"),
            key=lambda item: item["true_label_margin"],
        )[:2]
        near = sorted(
            (item for item in items if item["probe_kind"] == "near_zero"),
            key=lambda item: abs(item["true_label_margin"]),
        )[:2]
        visual.extend(wrong + near)
    visual = visual[:40]
    cols, cell_w, cell_h = 4, 190, 146
    rows = (len(visual) + cols - 1) // cols
    image = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
    draw, font = ImageDraw.Draw(image), ImageFont.load_default()
    for index, probe in enumerate(visual):
        x0, y0 = (index % cols) * cell_w, (index // cols) * cell_h
        sample = np.asarray(features[probe["row_index"]], dtype=np.float32)
        starts = np.flatnonzero(sample[:, 3] > 0.5).tolist()
        if not starts or starts[0] != 0:
            starts = [0, *starts]
        ends = starts[1:] + [len(sample)]
        color = "#c0392b" if probe["predicted_id"] != probe["class_id"] else "#176b3a"
        for start, end in zip(starts, ends, strict=True):
            points = [
                (x0 + 10 + float(sample[point, 0]) * 165,
                 y0 + 24 + (1.0 - float(sample[point, 1])) * 86)
                for point in range(start, end)
            ]
            if len(points) > 1:
                draw.line(points, fill=color, width=2, joint="curve")
        true_label = labels[probe["class_id"]].replace("\\", "")
        pred_label = labels[probe["predicted_id"]].replace("\\", "")
        true_label = true_label[:7] + ("~" if len(true_label) > 7 else "")
        pred_label = pred_label[:7] + ("~" if len(pred_label) > 7 else "")
        family = class_families[probe["class_id"]]
        draw.text((x0 + 4, y0 + 3), f"{family} {probe['source']} {probe['probe_kind']}", fill="#222", font=font)
        draw.text((x0 + 4, y0 + 116), f"{true_label}>{pred_label}", fill=color, font=font)
        draw.text((x0 + 4, y0 + 130), f"m={probe['true_label_margin']:+.2f} p={probe['teacher_softmax_true_probability_t2']:.2f}", fill=color, font=font)
    if path.exists():
        raise FileExistsError(path)
    image.save(path)
    return {
        "path": str(path.resolve()), "rows_shown": len(visual), "columns": cols,
        "width": cols * cell_w, "height": rows * cell_h,
    }


def _self_test() -> int:
    logits = np.full((3, CLASS_COUNT), -4.0, dtype=np.float32)
    logits[0, 0], logits[0, 1], logits[0, 2] = 3.0, 1.0, 0.0
    logits[1, 0], logits[1, 1], logits[1, 2] = 2.0, 2.4, 0.0
    logits[2, 0], logits[2, 1], logits[2, 2] = 0.5, 0.0, 0.2
    pred, rival, margins, top5, probability = _score_logits(logits, np.asarray([0, 0, 0]))
    if pred.tolist() != [0, 1, 0] or rival.tolist() != [1, 1, 2]:
        raise AssertionError("top-1/rival extraction changed")
    if not np.allclose(margins, [2.0, -0.4, 0.3], atol=1e-6):
        raise AssertionError("true-label margin calculation changed")
    if not top5.all() or not np.isfinite(probability).all() or not np.all((probability > 0) & (probability < 1)):
        raise AssertionError("Top-5 or temperature-2 probability check failed")
    print(json.dumps({"event": "probability_boundary_map_self_test_pass", "crohme_rows": 0}), flush=True)
    return 0


def _run_audit(args) -> int:
    started = time.perf_counter()
    manifest_path = args.data_dir / "prepared_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "pass":
        raise ValueError("prepared manifest did not pass")
    policy = manifest.get("input_policy", {})
    if "zero rows" not in str(policy.get("crohme", "")).casefold():
        raise ValueError("prepared manifest does not attest CROHME exclusion")
    if manifest.get("current_checkpoint", {}).get("sha256") != _sha256(args.checkpoint):
        raise ValueError("training cache and frozen checkpoint SHA differ")
    if args.report_dir.exists() and any(args.report_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty report directory: {args.report_dir}")

    manifest_labels = [row["label"] for row in manifest.get("class_rows", [])]
    class_families = [row["family"] for row in manifest.get("class_rows", [])]
    if len(manifest_labels) != CLASS_COUNT or len(set(manifest_labels)) != CLASS_COUNT:
        raise ValueError("prepared manifest lacks a unique ordered 372-class vocabulary")
    if manifest.get("current_checkpoint", {}).get("classes") != CLASS_COUNT:
        raise ValueError("frozen checkpoint class count is not 372")

    train_features = np.load(args.data_dir / "train_features.npy", mmap_mode="r", allow_pickle=False)
    train_labels = np.load(args.data_dir / "train_labels.npy", mmap_mode="r", allow_pickle=False)
    train_sources = np.load(args.data_dir / "train_sources.npy", mmap_mode="r", allow_pickle=False)
    teacher_logits = np.load(args.data_dir / "teacher_train_logits.npy", mmap_mode="r", allow_pickle=False)
    n = int(manifest.get("cache", {}).get("train", {}).get("rows", -1))
    if train_features.shape != (n, 128, 5) or train_labels.shape != (n,) or train_sources.shape != (n,) or teacher_logits.shape != (n, CLASS_COUNT):
        raise ValueError("training features/labels/sources/logits are not aligned with manifest")
    if not np.isin(np.unique(train_sources), list(SOURCE_IDS.values())).all():
        raise ValueError("unexpected source ID in the prepared training cache")

    counts = np.zeros((CLASS_COUNT, 2), dtype=np.int64)
    hits1 = np.zeros_like(counts)
    hits5 = np.zeros_like(counts)
    stroke_hist = [[{} for _ in REAL_SOURCES] for _ in range(CLASS_COUNT)]
    margins_by_group: list[list[list[np.ndarray]]] = [[[] for _ in REAL_SOURCES] for _ in range(CLASS_COUNT)]
    probabilities_by_group: list[list[list[np.ndarray]]] = [[[] for _ in REAL_SOURCES] for _ in range(CLASS_COUNT)]
    confusion = np.zeros((2, CLASS_COUNT, CLASS_COUNT), dtype=np.int64)
    probe_pool: list[list[list[dict[str, Any]]]] = [[[] for _ in REAL_SOURCES] for _ in range(CLASS_COUNT)]
    real_rows = synthetic_rows = 0

    for start in range(0, n, CHUNK_ROWS):
        stop = min(n, start + CHUNK_ROWS)
        source_batch = np.asarray(train_sources[start:stop], dtype=np.int64)
        target_batch = np.asarray(train_labels[start:stop], dtype=np.int64)
        if np.any(target_batch < 0) or np.any(target_batch >= CLASS_COUNT):
            raise ValueError(f"target outside frozen vocabulary in rows {start}:{stop}")
        eligible = source_batch < SOURCE_IDS["synthetic_equal"]
        synthetic_rows += int((~eligible).sum())
        real_rows += int(eligible.sum())
        if not eligible.any():
            continue
        local_features = np.asarray(train_features[start:stop][eligible], dtype=np.float32)
        targets = target_batch[eligible]
        sources = source_batch[eligible]
        row_ids = start + np.flatnonzero(eligible)
        if not np.isfinite(local_features).all() or (local_features[:, :, :2] < 0).any() or (local_features[:, :, :2] > 1).any():
            raise ValueError(f"invalid feature values in rows {start}:{stop}")
        stroke_counts = (local_features[:, :, 3] > 0.5).sum(axis=1)
        logits = np.asarray(teacher_logits[start:stop][eligible], dtype=np.float32)
        pred, _rival, margin, top5, true_probability = _score_logits(logits, targets)

        for source_index, (source_id, source_name) in enumerate(REAL_SOURCES):
            source_mask = sources == source_id
            if not source_mask.any():
                continue
            source_targets = targets[source_mask]
            source_pred = pred[source_mask]
            flat = source_targets * CLASS_COUNT + source_pred
            confusion[source_index] += np.bincount(flat, minlength=CLASS_COUNT * CLASS_COUNT).reshape(CLASS_COUNT, CLASS_COUNT)
            for class_id in np.unique(source_targets):
                selected = source_targets == class_id
                local_ids = np.flatnonzero(source_mask)[selected]
                m = margin[local_ids]
                p = true_probability[local_ids]
                counts[class_id, source_index] += len(local_ids)
                hits1[class_id, source_index] += int(np.sum(pred[local_ids] == class_id))
                hits5[class_id, source_index] += int(np.sum(top5[local_ids]))
                margins_by_group[class_id][source_index].append(m.copy())
                probabilities_by_group[class_id][source_index].append(p.copy())
                hist = stroke_hist[class_id][source_index]
                for value, amount in zip(*np.unique(stroke_counts[local_ids], return_counts=True), strict=True):
                    hist[str(int(value))] = int(hist.get(str(int(value)), 0) + int(amount))

                nearest = np.argsort(np.abs(m), kind="stable")[:PROBES_PER_GROUP]
                candidates = [
                    _record_probe(
                        int(row_ids[local_ids[pos]]), int(class_id), int(pred[local_ids[pos]]),
                        source_name, float(m[pos]), float(p[pos]), int(stroke_counts[local_ids[pos]]), "near_zero",
                    )
                    for pos in nearest
                ]
                negative = np.flatnonzero(m <= 0.0)
                if len(negative):
                    worst = int(negative[np.argmin(m[negative])])
                    candidates.append(_record_probe(
                        int(row_ids[local_ids[worst]]), int(class_id), int(pred[local_ids[worst]]),
                        source_name, float(m[worst]), float(p[worst]), int(stroke_counts[local_ids[worst]]), "hard_miss",
                    ))
                probe_pool[class_id][source_index] = _update_probe_pool(
                    probe_pool[class_id][source_index], candidates, PROBES_PER_GROUP + 1,
                )

    if real_rows + synthetic_rows != n:
        raise AssertionError("source partition totals do not equal manifest rows")
    manifest_support = np.asarray(
        [row.get("real_train", 0) for row in manifest["class_rows"]], dtype=np.int64,
    )
    if not np.array_equal(counts.sum(axis=1), manifest_support):
        raise AssertionError("class/source support totals disagree with the prepared manifest")
    if int(counts.sum()) != real_rows or int(hits1.sum()) != int(np.trace(confusion.sum(axis=0))):
        raise AssertionError("class hits/support disagree with the confusion matrix")
    if int(confusion.sum()) != real_rows or np.any(hits5 > counts):
        raise AssertionError("confusion or Top-5 totals failed integrity checks")
    per_class = []
    all_probes: list[dict[str, Any]] = []
    for class_id, label in enumerate(manifest_labels):
        by_source = {}
        class_margin_parts, class_probability_parts = [], []
        for source_index, (_source_id, source_name) in enumerate(REAL_SOURCES):
            mparts = margins_by_group[class_id][source_index]
            pparts = probabilities_by_group[class_id][source_index]
            support = int(counts[class_id, source_index])
            class_margin_parts.extend(mparts)
            class_probability_parts.extend(pparts)
            if support:
                by_source[source_name] = {
                    "rows": support,
                    "top1_hits": int(hits1[class_id, source_index]),
                    "top1_accuracy": float(hits1[class_id, source_index] / support),
                    "top5_hits": int(hits5[class_id, source_index]),
                    "top5_recall": float(hits5[class_id, source_index] / support),
                    "nonpositive_true_margin_rows": int(sum(np.sum(values <= 0.0) for values in mparts)),
                    "true_label_margin": _describe(mparts),
                    "teacher_softmax_true_probability_t2": _describe(pparts),
                    "stroke_count_histogram": stroke_hist[class_id][source_index],
                }
                all_probes.extend(probe_pool[class_id][source_index])
        support = int(counts[class_id].sum())
        class_margins = _describe(class_margin_parts)
        class_probs = _describe(class_probability_parts)
        confusion_row = confusion[:, class_id, :].sum(axis=0)
        confusion_row[class_id] = 0
        rivals = [
            {"label": manifest_labels[int(rival)], "rows": int(confusion_row[rival])}
            for rival in np.argsort(-confusion_row, kind="stable")[:5]
            if confusion_row[rival] > 0
        ]
        per_class.append({
            "class_id": class_id,
            "label": label,
            "family": class_families[class_id],
            "real_train_support": int(manifest["class_rows"][class_id].get("real_train", 0)),
            "scored_real_support": support,
            "top1_hits": int(hits1[class_id].sum()),
            "top1_accuracy": None if not support else float(hits1[class_id].sum() / support),
            "top5_hits": int(hits5[class_id].sum()),
            "top5_recall": None if not support else float(hits5[class_id].sum() / support),
            "nonpositive_true_margin_rows": int(sum(
                np.sum(values <= 0.0)
                for source_index in range(2)
                for values in margins_by_group[class_id][source_index]
            )),
            "true_label_margin": class_margins,
            "teacher_softmax_true_probability_t2": class_probs,
            "top1_confusions": rivals,
            "by_source": by_source,
        })

    confusion_total = confusion.sum(axis=0)
    np.fill_diagonal(confusion_total, 0)
    top_confusions = []
    for flat_id in np.argsort(-confusion_total.ravel(), kind="stable")[:100]:
        count = int(confusion_total.ravel()[flat_id])
        if count <= 0:
            break
        target_id, predicted_id = divmod(int(flat_id), CLASS_COUNT)
        top_confusions.append({
            "true_label": manifest_labels[target_id],
            "predicted_label": manifest_labels[predicted_id],
            "rows": count,
            "by_source": {
                source_name: int(confusion[source_index, target_id, predicted_id])
                for source_index, (_source_id, source_name) in enumerate(REAL_SOURCES)
            },
        })
    all_probes.sort(key=lambda item: (item["class_id"], item["source"], item["probe_kind"], abs(item["true_label_margin"])))
    source_summaries = {}
    for source_index, (_source_id, source_name) in enumerate(REAL_SOURCES):
        source_margins = [
            values for class_groups in margins_by_group
            for values in class_groups[source_index]
        ]
        source_probabilities = [
            values for class_groups in probabilities_by_group
            for values in class_groups[source_index]
        ]
        support = int(counts[:, source_index].sum())
        source_summaries[source_name] = {
            "rows": support,
            "top1_hits": int(hits1[:, source_index].sum()),
            "top1_accuracy": float(hits1[:, source_index].sum() / max(support, 1)),
            "top5_hits": int(hits5[:, source_index].sum()),
            "top5_recall": float(hits5[:, source_index].sum() / max(support, 1)),
            "nonpositive_true_margin_rows": int(sum(np.sum(values <= 0.0) for values in source_margins)),
            "true_label_margin": _describe(source_margins),
            "teacher_softmax_true_probability_t2": _describe(source_probabilities),
        }
    visual_path = args.report_dir / "probability_boundary_probe_samples.png"
    args.report_dir.mkdir(parents=True, exist_ok=True)
    visual = _draw_probe_sheet(train_features, all_probes, manifest_labels, class_families, visual_path)
    visual["sha256"] = _sha256(visual_path)
    visual["bytes"] = visual_path.stat().st_size
    report = {
        "schema": "aiflow-hwr-probability-boundary-map/v1",
        "status": "training_pool_in_sample_boundary_diagnostic_only",
        "scope": "all 372 frozen mathematical-domain classes; real HWRT and UJI train rows; source=synthetic_equal excluded",
        "boundary_definition": "true-label logit minus maximum competing-class logit; zero is the frozen teacher boundary for the observed row",
        "probability_definition": f"teacher softmax probability of the human data label at T={TEMPERATURE}; not calibrated human confidence",
        "provenance": {
            "prepared_manifest_sha256": _sha256(manifest_path),
            "teacher_checkpoint_sha256": _sha256(args.checkpoint),
            "train_features_sha256": _sha256(args.data_dir / "train_features.npy"),
            "train_labels_sha256": _sha256(args.data_dir / "train_labels.npy"),
            "train_sources_sha256": _sha256(args.data_dir / "train_sources.npy"),
            "teacher_train_logits_sha256": _sha256(args.data_dir / "teacher_train_logits.npy"),
            "class_order_from_prepared_manifest": True,
            "training_rows_total": n,
            "real_rows_scored": real_rows,
            "synthetic_equal_rows_excluded": synthetic_rows,
            "heldout_rows_read": 0,
            "crohme_rows": 0,
            "in_sample_logits": True,
            "model_selection_performed": False,
            "student_training_performed": False,
        },
        "summary": {
            "classes": CLASS_COUNT,
            "classes_with_real_support": int(sum(bool(value) for value in counts.sum(axis=1))),
            "real_rows_scored": real_rows,
            "overall_top1_accuracy": float(hits1.sum() / max(real_rows, 1)),
            "overall_top5_recall": float(hits5.sum() / max(real_rows, 1)),
            "by_source": source_summaries,
            "nonpositive_true_margin_rows": int(sum(
                np.sum(values <= 0.0)
                for class_groups in margins_by_group
                for source_groups in class_groups
                for values in source_groups
            )),
            "in_sample_warning": "training predictions can be overconfident; use this map only to locate candidate examples for manual review, never as an unbiased accuracy estimate or as the only augmentation sampler",
        },
        "source_ids": {name: value for name, value in SOURCE_IDS.items()},
        "classwise": per_class,
        "top1_confusion_pairs": top_confusions,
        "boundary_probes": all_probes,
        "visual_sample_sheet": visual,
        "interpretation_boundary": "diagnostic only; no training, heldout tuning, CROHME access, model selection, or product promotion",
        "runtime": {
            "chunk_rows": CHUNK_ROWS,
            "temperature": TEMPERATURE,
            "commit_headroom_gib_before_numpy": args.commit_headroom_gib_before_numpy,
            "commit_headroom_gib_after_numpy": args.commit_headroom_gib_after_numpy,
            "commit_headroom_safety_floor_gib": MIN_COMMIT_HEADROOM_GIB,
        },
        "seconds": time.perf_counter() - started,
    }
    report_path = args.report_dir / "probability_boundary_map.json"
    with report_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({
        "event": "probability_boundary_map_complete",
        "report": str(report_path.resolve()),
        "classes": CLASS_COUNT,
        "real_rows_scored": real_rows,
        "boundary_rows": report["summary"]["nonpositive_true_margin_rows"],
        "top_confusion_pairs": top_confusions[:5],
        "crohme_rows": 0,
    }, ensure_ascii=False), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("self-test", "audit"), required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    args = parser.parse_args()
    before = _guard_commit("before_numpy_import")
    if before is None:
        return 78
    args.commit_headroom_gib_before_numpy = before
    _load_numpy()
    after = _guard_commit("after_numpy_import")
    if after is None:
        return 78
    args.commit_headroom_gib_after_numpy = after
    if args.mode == "self-test":
        return _self_test()
    for path in (args.data_dir / "prepared_manifest.json", args.checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    for filename in ("train_features.npy", "train_labels.npy", "train_sources.npy", "teacher_train_logits.npy"):
        if not (args.data_dir / filename).is_file():
            raise FileNotFoundError(args.data_dir / filename)
    return _run_audit(args)


if __name__ == "__main__":
    raise SystemExit(main())
