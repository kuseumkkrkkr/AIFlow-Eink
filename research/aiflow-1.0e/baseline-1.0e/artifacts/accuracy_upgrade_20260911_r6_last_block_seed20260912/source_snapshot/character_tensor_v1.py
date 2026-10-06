#!/usr/bin/env python3
"""Build fixed 128-point, five-channel inputs for AIFlow Math Ink 1.0.

No model fitting lives here. This module only validates canonical online ink,
creates a privacy-preserving direct-ownership evaluation derivative, and emits
the tensor contract used by a future classifier.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from build_normalized_ink_v1 import SourceSample, _canonicalize


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANONICAL_ROOT = ROOT / "datasets" / "normalized" / "v1"
DEFAULT_OUTPUT = DEFAULT_CANONICAL_ROOT / "character_classifier_v1"
POINTS = 128
CHANNELS = ("x", "y", "delta_t", "stroke_start", "observed")
EXCLUDED_SOURCES = {"bdshwa_raw_online"}
EXTERNAL_TRAINING_SOURCES = ("hwrt", "uji", "isgl", "uci")


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _json_lines(path: Path) -> Iterator[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def iter_current_external_metadata(canonical_root: Path) -> Iterator[dict]:
    """Yield source-qualified labels for the fixed 10% external holdout."""
    for corpus in EXTERNAL_TRAINING_SOURCES:
        head = "math" if corpus == "hwrt" else "auxiliary"
        for row in _json_lines(canonical_root / f"{corpus}.jsonl.gz"):
            source_record_id = str(row["record_id"])
            yield {
                "record_id": f"{corpus}:{source_record_id}",
                "source": corpus,
                "source_record_id": source_record_id,
                "head": head,
                "label": str(row["label"]),
                "holdout_key": f"{corpus}:{row['label']}",
            }


def _strokes(record: dict) -> list[np.ndarray]:
    if record.get("source") in EXCLUDED_SOURCES:
        raise ValueError(f"excluded source: {record['source']}")
    raw = record.get("strokes")
    if not isinstance(raw, list) or not raw:
        raise ValueError("record has no strokes")
    orders = [stroke.get("source_order") for stroke in raw]
    if orders != sorted(orders) or len(orders) != len(set(orders)):
        raise ValueError("strokes are not strictly source-ordered")
    strokes = [np.asarray(stroke.get("points"), dtype=np.float64) for stroke in raw]
    if any(points.ndim != 2 or points.shape[1] != 3 or len(points) == 0 for points in strokes):
        raise ValueError("each stroke must contain N x 3 points")
    points = np.concatenate(strokes, axis=0)
    if len(points) < 2:
        raise ValueError("one-point trajectory is a missing character trajectory")
    if not np.isfinite(points).all() or (points < 0).any() or (points > 1).any():
        raise ValueError("canonical points must be finite and in [0,1]")
    if (np.diff(points[:, 2]) < 0).any():
        raise ValueError("canonical time reversal")
    return strokes


def _stroke_weight(points: np.ndarray) -> float:
    distance = np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1).sum() if len(points) > 1 else 0.0
    return float(distance) if distance > 0 else 1.0


def _allocate_points(strokes: list[np.ndarray], target_points: int) -> list[int]:
    if target_points < len(strokes):
        raise ValueError(f"{len(strokes)} strokes cannot retain stroke starts in {target_points} points")
    remaining = target_points - len(strokes)
    weights = np.asarray([_stroke_weight(stroke) for stroke in strokes], dtype=np.float64)
    exact = remaining * weights / weights.sum()
    base = np.floor(exact).astype(int)
    remainder = remaining - int(base.sum())
    for index in sorted(range(len(strokes)), key=lambda value: (-float(exact[value] - base[value]), value))[:remainder]:
        base[index] += 1
    return [int(value) + 1 for value in base]


def _resample_stroke(points: np.ndarray, count: int) -> np.ndarray:
    if count == 1:
        return points[:1].copy()
    delta = np.diff(points[:, :2], axis=0)
    distance = np.linalg.norm(delta, axis=1)
    domain = np.concatenate(([0.0], np.cumsum(distance)))
    if domain[-1] == 0.0:
        domain = np.arange(len(points), dtype=np.float64)
    grid = np.linspace(domain[0], domain[-1], count, dtype=np.float64)
    return np.column_stack([np.interp(grid, domain, points[:, column]) for column in range(3)])


def tensorize(record: dict, target_points: int = POINTS) -> np.ndarray:
    """Return an exactly ``target_points x 5`` float32 tensor without cropping."""
    strokes = _strokes(record)
    observed = 1.0 if record.get("normalization", {}).get("source_time_available") else 0.0
    output: list[np.ndarray] = []
    previous_time: float | None = None
    for stroke, count in zip(strokes, _allocate_points(strokes, target_points), strict=True):
        sampled = _resample_stroke(stroke, count)
        delta_t = np.maximum(0.0, np.diff(sampled[:, 2], prepend=sampled[0, 2] if previous_time is None else previous_time))
        starts = np.zeros(count, dtype=np.float64); starts[0] = 1.0
        output.append(np.column_stack((sampled[:, 0], sampled[:, 1], delta_t, starts, np.full(count, observed))))
        previous_time = float(sampled[-1, 2])
    tensor = np.concatenate(output, axis=0)
    if tensor.shape != (target_points, len(CHANNELS)):
        raise AssertionError(f"unexpected tensor shape: {tensor.shape}")
    if not np.isfinite(tensor).all() or (tensor[:, :2] < 0).any() or (tensor[:, :2] > 1).any() or (tensor[:, 2] < 0).any():
        raise AssertionError("invalid tensor values")
    return tensor.astype(np.float32, copy=False)


def _source_rows(path: Path) -> dict[str, dict]:
    return {row["sample_id"]: row for row in _json_lines(path)}


def iter_direct_ownership_examples(formulas: Path, ownership: Path) -> Iterator[dict]:
    """Explode manually verified stroke ownership into box-local character rows."""
    source_rows = _source_rows(formulas)
    for annotation in _json_lines(ownership):
        if not annotation.get("accepted"):
            continue
        source = source_rows.get(annotation["sample_id"])
        if source is None:
            raise ValueError(f"ownership sample missing from valid formulas: {annotation['sample_id']}")
        all_strokes = sorted(source["strokes"], key=lambda stroke: stroke["order"])
        groups, labels = annotation.get("groups", []), annotation.get("labels", [])
        if len(groups) != len(labels):
            raise ValueError(f"ownership group/label count mismatch: {annotation['sample_id']}")
        for group_index, (group, label) in enumerate(zip(groups, labels, strict=True)):
            indices = sorted({int(index) for index in group})
            if not indices or indices[-1] >= len(all_strokes):
                raise ValueError(f"invalid ownership stroke group: {annotation['sample_id']}:{group_index}")
            strokes = [[(float(point["x"]), float(point["y"]), float(point["t_ms"])) for point in all_strokes[index]["points"]] for index in indices]
            canonical = _canonicalize(SourceSample(
                "project_owned_ownership", f"{annotation['sample_id']}:{group_index}", str(label), "project_owned_evaluation",
                "box_local_project_owned_evaluation_only", strokes,
            ))
            canonical["formula_group_index"] = group_index
            canonical["writer_group"] = _digest(["project_owned_writer", annotation["writer_id"]])[:24]
            yield canonical


def stratified_holdout(rows: Iterable[dict], ratio: float = 0.10, seed: int = 20260812) -> set[str]:
    """Deterministic holdout by ``holdout_key`` (or label), minimum one row."""
    if not 0.0 < ratio <= 1.0:
        raise ValueError("ratio must be in (0,1]")
    by_key: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        by_key[str(row.get("holdout_key", row["label"]))].append(str(row["record_id"]))
    selected: set[str] = set()
    for key, identifiers in sorted(by_key.items()):
        ordered = sorted(identifiers, key=lambda identifier: _digest([seed, key, identifier]))
        selected.update(ordered[:max(1, math.floor(len(ordered) * ratio))])
    return selected


def _gzip_write(path: Path, rows: Iterable[dict]) -> int:
    temporary = path.with_name(f".{path.name}.tmp")
    count = 0
    try:
        with temporary.open("wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", compresslevel=6, mtime=0) as zipped, io.TextIOWrapper(zipped, encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), sort_keys=True)); stream.write("\n")
                count += 1
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return count


def build_manifests(canonical_root: Path, output: Path) -> dict:
    """Write only the small direct-ownership derivative; large source JSONL stays in place."""
    if output.resolve().drive.upper() != "D:":
        raise ValueError(f"character-corpus output must remain on D:: {output}")
    output.mkdir(parents=True, exist_ok=True)
    direct_path = output / "project_owned_ownership_eval.jsonl.gz"
    formulas = ROOT / "hf-dataset" / "data" / "formulas_valid.jsonl"
    ownership = ROOT / "hf-dataset" / "data" / "ownership_train.jsonl"
    direct_rows = list(iter_direct_ownership_examples(formulas, ownership))
    for row in direct_rows:
        tensorize(row)
    direct_count = _gzip_write(direct_path, direct_rows)
    holdout = stratified_holdout(iter_current_external_metadata(canonical_root))
    source_support: dict[str, dict] = {}
    for row in iter_current_external_metadata(canonical_root):
        support = source_support.setdefault(row["source"], {"records": 0, "labels": set(), "head": row["head"], "holdout_records": 0})
        support["records"] += 1
        support["labels"].add(row["label"])
        support["holdout_records"] += row["record_id"] in holdout
    source_summary = {
        source: {"records": support["records"], "labels": len(support["labels"]), "head": support["head"], "holdout_records": support["holdout_records"]}
        for source, support in source_support.items()
    }
    hwrt_path = canonical_root / "hwrt.jsonl.gz"
    manifest = {
        "schema": "aiflow-character-classifier-corpus/v1",
        "tensor": {"points": POINTS, "channels": list(CHANNELS), "resampling": "per-stroke-arc-length-with-minimum-one-point/v1"},
        "math_pretraining": {"file": str(hwrt_path.resolve()), "records": source_summary["hwrt"]["records"], "labels": source_summary["hwrt"]["labels"], "role": "training_only_external"},
        "auxiliary_pretraining": [
            {"file": str((canonical_root / name).resolve()), "role": "shared_encoder_only"}
            for name in ("uji.jsonl.gz", "isgl.jsonl.gz", "uci.jsonl.gz")
        ],
        "project_owned_evaluation": {"file": direct_path.name, "records": direct_count, "formulas": 47, "role": "all_records_evaluation_only"},
        "technical_holdout": {"scope": "approved_external_classifier_sources", "ratio": 0.10, "minimum_per_source_label": 1, "records": len(holdout), "selection": "deterministic_source_label_stratified", "sources": source_summary, "not_product_evidence": True},
        "excluded": {"bdshwa_raw_online": "no character boundaries or character-to-stroke ownership"},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    (output / "current_external_holdout_ids.json").write_text(json.dumps(sorted(holdout), ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return manifest


def _self_test() -> None:
    record = {
        "source": "test", "normalization": {"source_time_available": True},
        "strokes": [
            {"source_order": 0, "points": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.4], [1.0, 1.0, 0.6]]},
            {"source_order": 1, "points": [[0.0, 1.0, 0.7], [0.0, 0.0, 1.0]]},
        ],
    }
    tensor = tensorize(record, 16)
    assert tensor.shape == (16, 5)
    assert int(tensor[:, 3].sum()) == 2 and tensor[0, 3] == 1 and tensor[:, 4].min() == 1
    assert np.all(tensor[:, 2] >= 0)
    assert tensor[np.flatnonzero(tensor[:, 3])[1], 2] > 0
    try:
        tensorize({"source": "test", "normalization": {}, "strokes": [{"source_order": 0, "points": [[0.5, 0.5, 0.0]]}]})
    except ValueError as error:
        assert "one-point" in str(error)
    else:
        raise AssertionError("one-point trajectory tensorized")
    try:
        tensorize({"source": "bdshwa_raw_online", "normalization": {}, "strokes": record["strokes"]})
    except ValueError as error:
        assert "excluded source" in str(error)
    else:
        raise AssertionError("BDSHWA tensorized")
    rows = [{"label": "a", "record_id": "a1"}, {"label": "a", "record_id": "a2"}, {"label": "b", "record_id": "b1"}]
    assert len(stratified_holdout(rows)) == 2
    assert len(stratified_holdout([{"label": "a", "record_id": "uji", "holdout_key": "uji:a"}, {"label": "a", "record_id": "isgl", "holdout_key": "isgl:a"}])) == 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path, default=DEFAULT_CANONICAL_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--build-manifests", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test(); print(json.dumps({"self_test": "pass"})); return 0
    if not args.build_manifests:
        parser.error("use --build-manifests or --self-test; this script never trains")
    print(json.dumps(build_manifests(args.canonical_root, args.output), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
