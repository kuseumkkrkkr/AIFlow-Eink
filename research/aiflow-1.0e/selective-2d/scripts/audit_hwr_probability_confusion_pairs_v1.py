#!/usr/bin/env python3
"""Inspect real training glyphs on both sides of the largest HWR class boundaries.

Consumes the all-class probability map and the same hash-attested training
cache. This is an in-sample visual diagnostic only; it never selects or trains
a model and does not read heldout data or CROHME.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
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
DEFAULT_BOUNDARY_REPORT = ROOT / "artifacts" / "hwr_probability_boundary_map_20261004_r3" / "probability_boundary_map.json"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "hwr_probability_confusion_pairs_20261004"
CLASS_COUNT = 372
TOP_K = 5
TEMPERATURE = 2.0
CHUNK_ROWS = 4096
MIN_COMMIT_HEADROOM_GIB = 1.5
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
            "event": "probability_pair_audit_refused", "stage": stage,
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


def _softmax_true_probability(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    scaled = np.asarray(logits, dtype=np.float64) / TEMPERATURE
    ids = np.asarray(targets, dtype=np.int64)
    rows = np.arange(len(scaled))
    maximum = scaled.max(axis=1)
    normalizer = np.exp(scaled - maximum[:, None]).sum(axis=1)
    return np.exp(scaled[rows, ids] - maximum) / normalizer


def _score_logits(logits: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(logits, dtype=np.float32)
    target_ids = np.asarray(targets, dtype=np.int64)
    if values.ndim != 2 or values.shape[1] != CLASS_COUNT or target_ids.shape != (len(values),):
        raise ValueError("expected aligned [N,372] logits and [N] targets")
    if not np.isfinite(values).all() or np.any(target_ids < 0) or np.any(target_ids >= CLASS_COUNT):
        raise ValueError("invalid logits or class ID")
    return np.argmax(values, axis=1), _softmax_true_probability(values, target_ids)


def _choose_candidate(
    current: dict[str, Any] | None,
    candidate: dict[str, Any],
    *,
    closest_to_zero: bool,
) -> dict[str, Any]:
    if current is None:
        return candidate
    current_margin = float(current["pair_margin_oriented"])
    candidate_margin = float(candidate["pair_margin_oriented"])
    if closest_to_zero:
        return candidate if abs(candidate_margin) < abs(current_margin) else current
    return candidate if candidate_margin < current_margin else current


def _describe(parts: list[np.ndarray]) -> dict[str, float | int] | None:
    if not parts:
        return None
    values = np.concatenate(parts).astype(np.float64, copy=False)
    return {
        "rows": int(len(values)), "mean": float(values.mean()),
        "p10": float(np.quantile(values, 0.10)),
        "median": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "positive_fraction": float(np.mean(values > 0.0)),
    }


def _pair_definitions(boundary_report: dict[str, Any], labels: list[str], max_pairs: int) -> list[dict[str, Any]]:
    label_to_id = {label: index for index, label in enumerate(labels)}
    buckets: dict[tuple[int, int], dict[str, Any]] = {}
    for row in boundary_report.get("top1_confusion_pairs", []):
        true_label, predicted_label = row["true_label"], row["predicted_label"]
        if true_label not in label_to_id or predicted_label not in label_to_id:
            raise ValueError("boundary report contains a label outside its class vocabulary")
        left, right = label_to_id[true_label], label_to_id[predicted_label]
        key = tuple(sorted((left, right)))
        entry = buckets.setdefault(key, {"directions": {}, "source_errors": {name: 0 for _, name in REAL_SOURCES}})
        direction = (left, right)
        entry["directions"][direction] = int(row["rows"])
        for _, source_name in REAL_SOURCES:
            entry["source_errors"][source_name] += int(row.get("by_source", {}).get(source_name, 0))

    pairs = []
    for (left, right), entry in buckets.items():
        direction, count = max(entry["directions"].items(), key=lambda item: item[1])
        pairs.append({
            "class_a_id": int(direction[0]), "class_b_id": int(direction[1]),
            "reported_a_to_b_errors": int(count),
            "reported_reverse_errors": int(entry["directions"].get((direction[1], direction[0]), 0)),
            "reported_pair_errors": int(sum(entry["directions"].values())),
            "reported_by_source_errors": entry["source_errors"],
        })
    pairs.sort(key=lambda item: (-item["reported_pair_errors"], item["class_a_id"], item["class_b_id"]))
    return pairs[:max_pairs]


def _short_label(label: str, limit: int = 9) -> str:
    token = label.replace("\\", "")
    return token if len(token) <= limit else token[:limit - 1] + "~"


def _draw_glyph(draw, feature: np.ndarray, x: int, y: int, width: int, height: int, color: str) -> None:
    starts = np.flatnonzero(feature[:, 3] > 0.5).tolist()
    if not starts or starts[0] != 0:
        starts = [0, *starts]
    ends = starts[1:] + [len(feature)]
    for start, end in zip(starts, ends, strict=True):
        points = [
            (x + 2 + float(feature[index, 0]) * (width - 4),
             y + 2 + (1.0 - float(feature[index, 1])) * (height - 4))
            for index in range(start, end)
        ]
        if len(points) > 1:
            draw.line(points, fill=color, width=2, joint="curve")


def _draw_pair_sheet(
    features: np.ndarray, pairs: list[dict[str, Any]], labels: list[str],
    class_families: list[str], path: Path,
) -> dict[str, Any]:
    from PIL import Image, ImageDraw, ImageFont

    panel_w, panel_h, cols = 440, 222, 2
    rows = (len(pairs) + cols - 1) // cols
    image = Image.new("RGB", (cols * panel_w, rows * panel_h), "white")
    draw, font = ImageDraw.Draw(image), ImageFont.load_default()
    cell_w, cell_h = 216, 96
    slot_names = (
        ("a_near", "A true: closest to A/B margin 0"),
        ("a_error", "A true: top-1 predicted B"),
        ("b_near", "B true: closest to A/B margin 0"),
        ("b_error", "B true: top-1 predicted A"),
    )
    for pair_index, pair in enumerate(pairs):
        px, py = (pair_index % cols) * panel_w, (pair_index // cols) * panel_h
        a, b = pair["class_a_id"], pair["class_b_id"]
        label_a, label_b = _short_label(labels[a]), _short_label(labels[b])
        source = pair.get("visual_source")
        draw.text((px + 5, py + 4),
                  f"{pair_index + 1}. {label_a} vs {label_b}  {source}  errors={pair['total_observed_errors']}",
                  fill="#111111", font=font)
        for slot_index, (slot_name, title) in enumerate(slot_names):
            col, row = slot_index % 2, slot_index // 2
            x0, y0 = px + col * cell_w, py + 22 + row * cell_h
            draw.rectangle((x0, y0, x0 + cell_w - 3, y0 + cell_h - 3), outline="#d9dfe5")
            probe = pair["visual_probes"].get(slot_name)
            draw.text((x0 + 4, y0 + 2), title, fill="#34495e", font=font)
            if probe is None:
                draw.text((x0 + 5, y0 + 42), "no matching sample", fill="#888888", font=font)
                continue
            feature = np.asarray(features[probe["row_index"]], dtype=np.float32)
            is_correct = probe["predicted_id"] == probe["true_id"]
            color = "#176b3a" if is_correct else "#c0392b"
            _draw_glyph(draw, feature, x0 + 3, y0 + 18, 86, 72, color)
            true_text = _short_label(labels[probe["true_id"]], 6)
            pred_text = _short_label(labels[probe["predicted_id"]], 6)
            draw.text((x0 + 91, y0 + 24), f"{true_text}>{pred_text}", fill=color, font=font)
            draw.text((x0 + 91, y0 + 39), f"m={probe['pair_margin_oriented']:+.2f}", fill=color, font=font)
            draw.text((x0 + 91, y0 + 54), f"pT2={probe['teacher_softmax_true_probability_t2']:.2f}", fill=color, font=font)
            draw.text((x0 + 91, y0 + 69), f"strokes={probe['stroke_count']}", fill="#555555", font=font)
    if path.exists():
        raise FileExistsError(path)
    image.save(path)
    return {"path": str(path.resolve()), "width": cols * panel_w, "height": rows * panel_h, "pairs_shown": len(pairs)}


def _self_test() -> int:
    logits = np.full((3, CLASS_COUNT), -5.0, dtype=np.float32)
    logits[:, 1], logits[:, 4] = np.asarray([2.0, 0.8, 0.1]), np.asarray([1.0, 1.2, 0.2])
    pred, prob = _score_logits(logits, np.asarray([1, 1, 4]))
    if pred.tolist() != [1, 4, 4] or not np.isfinite(prob).all():
        raise AssertionError("pair-audit scoring contract changed")
    current = {"pair_margin_oriented": -0.25}
    candidate = {"pair_margin_oriented": -0.03}
    if _choose_candidate(current, candidate, closest_to_zero=True) is not candidate:
        raise AssertionError("near-boundary sample selection failed")
    if _choose_candidate(current, candidate, closest_to_zero=False) is not current:
        raise AssertionError("hardest-error sample selection failed")
    labels = ["A", "B"] + [f"c{i}" for i in range(CLASS_COUNT - 2)]
    fake = {"top1_confusion_pairs": [
        {"true_label": "A", "predicted_label": "B", "rows": 8, "by_source": {"hwrt": 6, "uji": 2}},
        {"true_label": "B", "predicted_label": "A", "rows": 3, "by_source": {"hwrt": 1, "uji": 2}},
    ]}
    pairs = _pair_definitions(fake, labels, 1)
    if len(pairs) != 1 or pairs[0]["reported_pair_errors"] != 11 or pairs[0]["class_a_id"] != 0:
        raise AssertionError("reciprocal confusion-pair aggregation failed")
    print(json.dumps({"event": "probability_confusion_pairs_self_test_pass", "crohme_rows": 0}), flush=True)
    return 0


def _run_audit(args) -> int:
    started = time.perf_counter()
    boundary_report = json.loads(args.boundary_report.read_text(encoding="utf-8"))
    if boundary_report.get("schema") != "aiflow-hwr-probability-boundary-map/v1":
        raise ValueError("unexpected all-class boundary report schema")
    provenance = boundary_report.get("provenance", {})
    if not provenance.get("in_sample_logits") or provenance.get("heldout_rows_read") != 0 or provenance.get("crohme_rows") != 0:
        raise ValueError("boundary map does not have the expected diagnostic-only scope")
    if args.report_dir.exists() and any(args.report_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty report directory: {args.report_dir}")

    manifest_path = args.data_dir / "prepared_manifest.json"
    actual_boundary_report_sha256 = _sha256(args.boundary_report)
    if actual_boundary_report_sha256.casefold() != args.expected_boundary_report_sha256.casefold():
        raise ValueError("boundary report SHA differs from pinned CLI attestation")
    if _sha256(manifest_path) != provenance.get("prepared_manifest_sha256"):
        raise ValueError("prepared manifest SHA differs from the classwise map")
    input_paths = {
        "teacher_checkpoint_sha256": args.checkpoint,
        "train_features_sha256": args.data_dir / "train_features.npy",
        "train_labels_sha256": args.data_dir / "train_labels.npy",
        "train_sources_sha256": args.data_dir / "train_sources.npy",
        "teacher_train_logits_sha256": args.data_dir / "teacher_train_logits.npy",
    }
    for key, path in input_paths.items():
        if _sha256(path) != provenance.get(key):
            raise ValueError(f"SHA mismatch against boundary map: {key}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "pass" or "zero rows" not in str(manifest.get("input_policy", {}).get("crohme", "")).casefold():
        raise ValueError("prepared manifest fails CROHME/data integrity policy")
    rows = manifest.get("class_rows", [])
    labels = [row["label"] for row in rows]
    families = [row["family"] for row in rows]
    if len(labels) != CLASS_COUNT or len(set(labels)) != CLASS_COUNT:
        raise ValueError("manifest class vocabulary is malformed")
    pairs = _pair_definitions(boundary_report, labels, args.max_pairs)
    if not pairs:
        raise ValueError("all-class map has no confusion pairs")

    features = np.load(args.data_dir / "train_features.npy", mmap_mode="r", allow_pickle=False)
    targets = np.load(args.data_dir / "train_labels.npy", mmap_mode="r", allow_pickle=False)
    sources = np.load(args.data_dir / "train_sources.npy", mmap_mode="r", allow_pickle=False)
    logits_map = np.load(args.data_dir / "teacher_train_logits.npy", mmap_mode="r", allow_pickle=False)
    n = int(manifest.get("cache", {}).get("train", {}).get("rows", -1))
    if features.shape != (n, 128, 5) or targets.shape != (n,) or sources.shape != (n,) or logits_map.shape != (n, CLASS_COUNT):
        raise ValueError("training feature/label/source/logit arrays are misaligned")

    for pair in pairs:
        pair["by_source"] = {}
        for _, source_name in REAL_SOURCES:
            pair["by_source"][source_name] = {
                "a_support": 0, "b_support": 0,
                "a_to_b_errors": 0, "b_to_a_errors": 0,
                "a_top1_hits": 0, "b_top1_hits": 0,
                "a_margins": [], "b_margins": [],
                "a_near": None, "a_error": None,
                "b_near": None, "b_error": None,
            }

    for start in range(0, n, CHUNK_ROWS):
        stop = min(n, start + CHUNK_ROWS)
        source_all = np.asarray(sources[start:stop], dtype=np.int64)
        eligible = source_all < 2
        if not eligible.any():
            continue
        source_batch = source_all[eligible]
        target_batch = np.asarray(targets[start:stop][eligible], dtype=np.int64)
        logits = np.asarray(logits_map[start:stop][eligible], dtype=np.float32)
        predictions, true_probs = _score_logits(logits, target_batch)
        row_ids = start + np.flatnonzero(eligible)
        stroke_counts = (np.asarray(features[start:stop][eligible, :, 3], dtype=np.float32) > 0.5).sum(axis=1)

        for _, source_name in REAL_SOURCES:
            source_mask = source_batch == (0 if source_name == "hwrt" else 1)
            if not source_mask.any():
                continue
            for pair in pairs:
                a, b = pair["class_a_id"], pair["class_b_id"]
                info = pair["by_source"][source_name]
                side_a = np.flatnonzero(source_mask & (target_batch == a))
                side_b = np.flatnonzero(source_mask & (target_batch == b))
                if len(side_a):
                    margin_a = logits[side_a, a] - logits[side_a, b]
                    pred_a = predictions[side_a]
                    info["a_support"] += int(len(side_a))
                    info["a_top1_hits"] += int(np.sum(pred_a == a))
                    info["a_to_b_errors"] += int(np.sum(pred_a == b))
                    info["a_margins"].append(margin_a.copy())
                    near_pos = int(np.argmin(np.abs(margin_a)))
                    near = {
                        "row_index": int(row_ids[side_a[near_pos]]), "true_id": int(a),
                        "predicted_id": int(pred_a[near_pos]), "source": source_name,
                        "pair_margin_oriented": float(margin_a[near_pos]),
                        "teacher_softmax_true_probability_t2": float(true_probs[side_a[near_pos]]),
                        "stroke_count": int(stroke_counts[side_a[near_pos]]),
                    }
                    info["a_near"] = _choose_candidate(info["a_near"], near, closest_to_zero=True)
                    errors = np.flatnonzero((pred_a == b) & (margin_a <= 0.0))
                    if len(errors):
                        error_pos = int(errors[np.argmax(margin_a[errors])])
                        error = {
                            "row_index": int(row_ids[side_a[error_pos]]), "true_id": int(a),
                            "predicted_id": int(pred_a[error_pos]), "source": source_name,
                            "pair_margin_oriented": float(margin_a[error_pos]),
                            "teacher_softmax_true_probability_t2": float(true_probs[side_a[error_pos]]),
                            "stroke_count": int(stroke_counts[side_a[error_pos]]),
                        }
                        info["a_error"] = _choose_candidate(info["a_error"], error, closest_to_zero=True)
                if len(side_b):
                    margin_b = logits[side_b, b] - logits[side_b, a]
                    pred_b = predictions[side_b]
                    info["b_support"] += int(len(side_b))
                    info["b_top1_hits"] += int(np.sum(pred_b == b))
                    info["b_to_a_errors"] += int(np.sum(pred_b == a))
                    info["b_margins"].append(margin_b.copy())
                    near_pos = int(np.argmin(np.abs(margin_b)))
                    near = {
                        "row_index": int(row_ids[side_b[near_pos]]), "true_id": int(b),
                        "predicted_id": int(pred_b[near_pos]), "source": source_name,
                        "pair_margin_oriented": float(margin_b[near_pos]),
                        "teacher_softmax_true_probability_t2": float(true_probs[side_b[near_pos]]),
                        "stroke_count": int(stroke_counts[side_b[near_pos]]),
                    }
                    info["b_near"] = _choose_candidate(info["b_near"], near, closest_to_zero=True)
                    errors = np.flatnonzero((pred_b == a) & (margin_b <= 0.0))
                    if len(errors):
                        error_pos = int(errors[np.argmax(margin_b[errors])])
                        error = {
                            "row_index": int(row_ids[side_b[error_pos]]), "true_id": int(b),
                            "predicted_id": int(pred_b[error_pos]), "source": source_name,
                            "pair_margin_oriented": float(margin_b[error_pos]),
                            "teacher_softmax_true_probability_t2": float(true_probs[side_b[error_pos]]),
                            "stroke_count": int(stroke_counts[side_b[error_pos]]),
                        }
                        info["b_error"] = _choose_candidate(info["b_error"], error, closest_to_zero=True)

    for pair in pairs:
        for source_name, info in pair["by_source"].items():
            info["a_true_vs_b_margin"] = _describe(info.pop("a_margins"))
            info["b_true_vs_a_margin"] = _describe(info.pop("b_margins"))
        source_name = max(
            (name for _, name in REAL_SOURCES),
            key=lambda name: pair["by_source"][name]["a_to_b_errors"] + pair["by_source"][name]["b_to_a_errors"],
        )
        chosen = pair["by_source"][source_name]
        pair["visual_source"] = source_name
        pair["total_observed_errors"] = int(chosen["a_to_b_errors"] + chosen["b_to_a_errors"])
        pair["visual_probes"] = {
            "a_near": chosen["a_near"], "a_error": chosen["a_error"],
            "b_near": chosen["b_near"], "b_error": chosen["b_error"],
        }
        pair["by_source"] = {
            source_name: {
                key: value for key, value in info.items()
                if key not in {"a_near", "a_error", "b_near", "b_error"}
            }
            for source_name, info in pair["by_source"].items()
        }
    pairs.sort(key=lambda item: -item["total_observed_errors"])

    args.report_dir.mkdir(parents=True, exist_ok=True)
    image_path = args.report_dir / "top_confusion_pair_boundaries.png"
    image_info = _draw_pair_sheet(features, pairs, labels, families, image_path)
    image_info["sha256"] = _sha256(image_path)
    image_info["bytes"] = image_path.stat().st_size
    for pair in pairs:
        pair["class_a"] = labels[pair["class_a_id"]]
        pair["class_b"] = labels[pair["class_b_id"]]
        for probes in pair["visual_probes"].values():
            if probes is not None:
                probes["true_label"] = labels[probes["true_id"]]
                probes["predicted_label"] = labels[probes["predicted_id"]]

    report = {
        "schema": "aiflow-hwr-probability-confusion-pairs/v1",
        "status": "training_pool_in_sample_pairwise_boundary_diagnostic_only",
        "input_boundary_report": str(args.boundary_report.resolve()),
        "input_boundary_report_sha256": actual_boundary_report_sha256,
        "scope": "top reciprocal class confusion pairs; same-source real HWRT/UJI training samples only",
        "pair_margin_definition": "logit(A)-logit(B); zero is the frozen teacher pairwise boundary for the observed feature",
        "provenance": {
            "prepared_manifest_sha256": _sha256(manifest_path),
            "teacher_checkpoint_sha256": _sha256(args.checkpoint),
            "train_features_sha256": _sha256(args.data_dir / "train_features.npy"),
            "train_labels_sha256": _sha256(args.data_dir / "train_labels.npy"),
            "train_sources_sha256": _sha256(args.data_dir / "train_sources.npy"),
            "teacher_train_logits_sha256": _sha256(args.data_dir / "teacher_train_logits.npy"),
            "pairs_requested": args.max_pairs,
            "pairs_reported": len(pairs),
            "heldout_rows_read": 0,
            "crohme_rows": 0,
            "in_sample_logits": True,
            "training_or_model_selection_performed": False,
        },
        "pairwise_boundaries": pairs,
        "visual_sample_sheet": image_info,
        "interpretation_limit": "training predictions are in-sample; visual pairs locate candidate regions for human review but do not establish human ambiguity, independent error rates, or an augmentation gain",
        "seconds": time.perf_counter() - started,
    }
    report_path = args.report_dir / "top_confusion_pair_boundaries.json"
    with report_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({
        "event": "probability_confusion_pair_audit_complete",
        "report": str(report_path.resolve()),
        "pairs": len(pairs), "sample_sheet": image_info,
        "top_pairs": [
            {"a": row["class_a"], "b": row["class_b"], "errors": row["total_observed_errors"]}
            for row in pairs[:5]
        ],
        "crohme_rows": 0,
    }, ensure_ascii=False), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("self-test", "audit"), required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--boundary-report", type=Path, default=DEFAULT_BOUNDARY_REPORT)
    parser.add_argument("--expected-boundary-report-sha256", default="")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-pairs", type=int, default=12)
    args = parser.parse_args()
    if args.mode == "audit" and not 1 <= args.max_pairs <= 40:
        parser.error("--max-pairs must be in [1,40]")
    before = _guard_commit("before_numpy_import")
    if before is None:
        return 78
    args.commit_headroom_gib_before_numpy = before
    _load_numpy()
    after = _guard_commit("after_numpy_import")
    if after is None:
        return 78
    if args.mode == "self-test":
        return _self_test()
    args.commit_headroom_gib_after_numpy = after
    if not args.expected_boundary_report_sha256:
        parser.error("audit requires --expected-boundary-report-sha256 from the pinned map report")
    if len(args.expected_boundary_report_sha256) != 64 or any(
        char not in "0123456789abcdefABCDEF" for char in args.expected_boundary_report_sha256
    ):
        parser.error("--expected-boundary-report-sha256 must be 64 hexadecimal characters")
    required = [args.boundary_report, args.checkpoint, args.data_dir / "prepared_manifest.json"]
    required.extend(args.data_dir / filename for filename in (
        "train_features.npy", "train_labels.npy", "train_sources.npy", "teacher_train_logits.npy",
    ))
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    return _run_audit(args)


if __name__ == "__main__":
    raise SystemExit(main())
