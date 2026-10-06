#!/usr/bin/env python3
"""Build source-preserving canonical online-ink derivatives for Math Ink 1.0.

Raw archives and inherited derivatives remain unchanged.  This builder emits
local-only model-input coordinates in a unit square, with an explicit
aspect-preserving bbox/letterbox transform and relative or ordinal time.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import shutil
import tempfile
import zipfile
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "datasets" / "normalized" / "v1"
DEFAULT_BDSHWA_ARCHIVE = ROOT / "datasets" / "10_approved_external" / "bdshwa" / "raw" / "bdshwa_v1.zip"
SCHEMA = "aiflow-canonical-online-ink/v1"
SPATIAL_RULE = "bbox-aspect-preserving-letterbox-unit-square/v1"
TIME_RULE = "relative-duration-or-stitched-stroke-duration-or-ordinal-unit-interval/v1"
CLASSIFIER_SOURCE_ORDER = ("project_owned", "uji", "isgl", "uci", "hwrt")
AUDIT_SOURCE_ORDER = ("bdshwa",)
SOURCE_ORDER = CLASSIFIER_SOURCE_ORDER + AUDIT_SOURCE_ORDER
SOURCE_IDENTIFIERS = {
    "project_owned": "project_owned",
    "uji": "uji_pen_v2",
    "isgl": "isgl_online",
    "uci": "uci_character_trajectories",
    "hwrt": "hwrt_curated",
    "bdshwa": "bdshwa_raw_online",
}
FORBIDDEN_DERIVATIVE_KEYS = {
    "source_id", "sample_id", "session_id", "participant_id", "writer_id", "writer_key",
    "raw_x", "raw_y", "timestamp", "raw_timestamp", "user_id", "user_agent", "pressure",
    "tilt_x", "tilt_y", "age", "gender", "handedness",
}
ALLOWED_BDSHWA_TASK_KEYS = {"mode", "category", "script", "task_label", "task_prompt", "writing_condition", "task_index"}


@dataclass(frozen=True)
class SourceSample:
    source: str
    source_id: str
    label: str
    split: str
    training_role: str
    strokes: list[list[tuple[float, float, float | None]]]
    time_scale_to_ms: float = 1.0
    extra: dict[str, Any] | None = None
    deduplicate_consecutive_exact_xy: bool = False


def _finite(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"non_numeric_value:{value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non_finite_value:{value!r}")
    return number


def _round(value: float) -> float:
    value = round(value, 8)
    return 0.0 if value == 0 else value


def _digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _record_id(source: str, source_id: str) -> str:
    # Do not expose raw file, participant, or source-writer identifiers.
    return f"{source}_{_digest([source, source_id])[:24]}"


def _point(stroke: Any, source: str) -> tuple[float, float, float | None]:
    if isinstance(stroke, dict):
        x, y = stroke.get("x"), stroke.get("y")
        t = stroke.get("t_ms")
    else:
        x, y = stroke[0], stroke[1]
        t = stroke[2] if len(stroke) > 2 else None
    return _finite(x), _finite(y), None if t is None else _finite(t)


def _canonicalize(sample: SourceSample) -> dict[str, Any]:
    cleaned: list[tuple[int, list[tuple[float, float, float | None]]]] = []
    dropped_empty: list[int] = []
    removed_consecutive_exact_xy_points = 0
    for source_order, raw_stroke in enumerate(sample.strokes):
        points = [_point(point, sample.source) for point in raw_stroke]
        if sample.deduplicate_consecutive_exact_xy:
            deduplicated: list[tuple[float, float, float | None]] = []
            for point in points:
                if deduplicated and point[:2] == deduplicated[-1][:2]:
                    # Retain the final timestamp so a real dwell remains in the
                    # following delta-time interval without retaining zero-motion copies.
                    deduplicated[-1] = point
                    removed_consecutive_exact_xy_points += 1
                else:
                    deduplicated.append(point)
            points = deduplicated
        if points:
            cleaned.append((source_order, points))
        else:
            dropped_empty.append(source_order)
    if not cleaned:
        raise ValueError(f"empty_sample:{sample.source}:{sample.source_id}")

    flat = [point for _, stroke in cleaned for point in stroke]
    if len(flat) < 2:
        raise ValueError(f"insufficient_points:{sample.source}:{sample.source_id}")
    xs, ys = [point[0] for point in flat], [point[1] for point in flat]
    left, right, top, bottom = min(xs), max(xs), min(ys), max(ys)
    width, height = right - left, bottom - top
    extent = max(width, height)
    scale = 1.0 / extent if extent > 0 else 1.0
    pad_x, pad_y = (1.0 - width * scale) / 2.0, (1.0 - height * scale) / 2.0

    provided_times = [point[2] for point in flat]
    has_provided_time = all(value is not None for value in provided_times)
    if has_provided_time:
        times = [float(value) * sample.time_scale_to_ms for value in provided_times if value is not None]
        if any(later < earlier for earlier, later in zip(times, times[1:])):
            stroke_times = [[float(point[2]) * sample.time_scale_to_ms for point in stroke] for _, stroke in cleaned]
            if any(any(later < earlier for earlier, later in zip(values, values[1:])) for values in stroke_times):
                raise ValueError(f"timestamp_reversal_within_stroke:{sample.source}:{sample.source_id}")
            # Some BDSHWA files reset/overlap their source clock at every pen-down.
            # Keep intra-stroke timing, but do not invent inter-stroke wall-clock gaps.
            stitched, offset = [], 0.0
            for values in stroke_times:
                local = [value - values[0] for value in values]
                stitched.extend(offset + value for value in local)
                offset += local[-1]
            times = stitched
            time_kind = "stitched_stroke_duration"
        else:
            time_kind = "relative_duration"
        duration_ms = times[-1] - times[0]
    else:
        times, duration_ms = [], 0.0
    ordinal = len(flat) > 1
    if has_provided_time and duration_ms > 0:
        normalized_times = [(value - times[0]) / duration_ms for value in times]
    else:
        denominator = max(len(flat) - 1, 1)
        normalized_times = [index / denominator for index in range(len(flat))]
        time_kind = "ordinal_index"

    global_index = 0
    normalized_strokes = []
    raw_strokes = []
    for source_order, points in cleaned:
        output_points = []
        raw_points = []
        for x, y, t in points:
            nx = min(1.0, max(0.0, (x - left) * scale + pad_x))
            ny = min(1.0, max(0.0, (y - top) * scale + pad_y))
            output_point = [_round(nx), _round(ny), _round(normalized_times[global_index])]
            if sample.deduplicate_consecutive_exact_xy and output_points and output_point[:2] == output_points[-1][:2]:
                # Canonical rounding can make a near-identical source movement
                # exactly equal. Coalesce it as well, keeping the final time.
                output_points[-1] = output_point
                raw_points[-1] = [x, y, t]
                removed_consecutive_exact_xy_points += 1
            else:
                output_points.append(output_point)
                raw_points.append([x, y, t])
            global_index += 1
        normalized_strokes.append({"source_order": source_order, "points": output_points})
        raw_strokes.append([source_order, raw_points])

    final_points = [point for stroke in normalized_strokes for point in stroke["points"]]
    if len(final_points) < 2:
        raise ValueError(f"insufficient_points:{sample.source}:{sample.source_id}")
    if sample.deduplicate_consecutive_exact_xy:
        first_time, last_time = final_points[0][2], final_points[-1][2]
        if last_time > first_time:
            for point in final_points:
                point[2] = _round((point[2] - first_time) / (last_time - first_time))

    record = {
        "schema": SCHEMA,
        "record_id": _record_id(sample.source, sample.source_id),
        "source": sample.source,
        "split": sample.split,
        "training_role": sample.training_role,
        "label": sample.label,
        "source_fingerprint": _digest([sample.source_id, raw_strokes]),
        "normalization": {
            "spatial": SPATIAL_RULE,
            "time": TIME_RULE,
            "time_kind": time_kind,
            "source_time_available": has_provided_time,
            "removed_consecutive_exact_xy_points": removed_consecutive_exact_xy_points,
        },
        "transform": {
            "bbox": {"left": _round(left), "top": _round(top), "right": _round(right), "bottom": _round(bottom)},
            "scale": _round(scale),
            "pad_x": _round(pad_x),
            "pad_y": _round(pad_y),
            "duration_ms": _round(duration_ms) if has_provided_time else None,
        },
        "source_stroke_count": len(sample.strokes),
        "dropped_empty_stroke_orders": dropped_empty,
        "stroke_count": len(normalized_strokes),
        "point_count": len(final_points),
        "strokes": normalized_strokes,
    }
    if sample.extra:
        record["task"] = sample.extra
    return record


def _json_lines(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def _project_owned() -> Iterable[SourceSample]:
    path = ROOT / "hf-dataset" / "data" / "formulas_valid.jsonl"
    for row in _json_lines(path):
        strokes = [[(p["x"], p["y"], p.get("t_ms")) for p in stroke["points"]] for stroke in sorted(row["strokes"], key=lambda s: s["order"])]
        yield SourceSample(
            "project_owned", row["sample_id"], str(row["target_display"]), "unassigned_project_holdout",
            "formula_layout_and_box_local_supervision", strokes,
        )


def _external_jsonl(path: Path, source: str, role: str) -> Iterable[SourceSample]:
    for row in _json_lines(path):
        strokes = [[_point(point, source) for point in stroke["points"]] for stroke in sorted(row["strokes"], key=lambda s: s["order"])]
        yield SourceSample(source, str(row["sample_id"]), str(row["label"]), str(row.get("split", "train")), role, strokes)


def _uji() -> Iterable[SourceSample]:
    return _external_jsonl(
        ROOT / "datasets" / "10_approved_external" / "uji_pen_characters_v2" / "derived" / "uji_math_curated.jsonl.gz",
        "uji_pen_v2", "box_local_character_pretraining",
    )


def _isgl() -> Iterable[SourceSample]:
    return _external_jsonl(
        ROOT / "datasets" / "10_approved_external" / "isgl_online_offline_hwr" / "migrated" / "isgl_online.jsonl.gz",
        "isgl_online", "box_local_character_pretraining_training_only",
    )


def _hwrt() -> Iterable[SourceSample]:
    root = ROOT / "datasets" / "10_approved_external" / "hwrt" / "derived"
    for split in ("train", "validation", "test"):
        yield from _external_jsonl(root / f"{split}.jsonl.gz", "hwrt_curated", "box_local_math_symbol_pretraining_training_only")


def _uci() -> Iterable[SourceSample]:
    path = ROOT / "datasets" / "10_approved_external" / "uci_character_trajectories" / "raw" / "character+trajectories.official.zip"
    with zipfile.ZipFile(path) as archive:
        data = loadmat(io.BytesIO(archive.read("mixoutALL_shifted.mat")), squeeze_me=True, struct_as_record=False)
    labels = data["consts"].key
    class_ids = data["consts"].charlabels
    for index, matrix in enumerate(data["mixout"]):
        points = [(float(matrix[0, point]), float(matrix[1, point]), point * 0.005) for point in range(matrix.shape[1])]
        yield SourceSample(
            "uci_character_trajectories", f"sample-{index}", str(labels[int(class_ids[index]) - 1]), "train",
            "single_writer_lowercase_representation_pretraining", [points], time_scale_to_ms=1000.0,
            deduplicate_consecutive_exact_xy=True,
        )


def _truthy(value: str | None) -> bool:
    return str(value).strip().lower() in {"1", "1.0", "true"}


@contextmanager
def _bdshwa_csv_archive(path: Path) -> Iterator[zipfile.ZipFile]:
    """Open either the supplied Raw_Data ZIP or its checked-in outer archive."""
    if not path.is_file():
        raise FileNotFoundError(f"BDSHWA archive is required: {path}")
    with zipfile.ZipFile(path) as outer:
        if any(name.startswith("Raw_Data/") and name.endswith(".csv") for name in outer.namelist()):
            yield outer
            return
        for nested_name in sorted(name for name in outer.namelist() if name.endswith(".zip")):
            with zipfile.ZipFile(io.BytesIO(outer.read(nested_name))) as nested:
                if any(name.startswith("Raw_Data/") and name.endswith(".csv") for name in nested.namelist()):
                    yield nested
                    return
    raise ValueError(f"BDSHWA Raw_Data ZIP was not found in: {path}")


def _bdshwa(archive_path: Path) -> Iterable[SourceSample]:
    with _bdshwa_csv_archive(archive_path) as archive:
        names = sorted(name for name in archive.namelist() if name.startswith("Raw_Data/") and name.endswith(".csv"))
        if len(names) != 1348:
            raise ValueError(f"unexpected BDSHWA raw CSV count: {len(names)}")
        for name in names:
            by_stroke: dict[int, list[tuple[float, float, float | None]]] = {}
            first_seen: list[int] = []
            task: dict[str, str] | None = None
            with archive.open(name) as raw:
                reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""))
                expected = ["timestamp", "raw_x", "raw_y", "canvas_x", "canvas_y", "pressure", "tilt_x", "tilt_y", "distance", "pen_down", "stroke", "mode", "category", "script", "task_label", "task_prompt", "sample_id", "session_id", "writing_condition", "task_index"]
                if reader.fieldnames != expected:
                    raise ValueError(f"unexpected BDSHWA header: {name}")
                for row in reader:
                    if task is None:
                        task = {key: str(row[key]) for key in ("mode", "category", "script", "task_label", "task_prompt", "writing_condition", "task_index")}
                    if not _truthy(row["pen_down"]):
                        continue
                    stroke = int(_finite(float(row["stroke"])))
                    if stroke not in by_stroke:
                        by_stroke[stroke] = []
                        first_seen.append(stroke)
                    by_stroke[stroke].append((_finite(float(row["canvas_x"])), _finite(float(row["canvas_y"])), _finite(float(row["timestamp"]))))
            strokes = [by_stroke[key] for key in first_seen]
            label = (task or {}).get("task_prompt") or (task or {}).get("task_label") or ""
            yield SourceSample(
                "bdshwa_raw_online", name, label, "train", "general_text_hwr_pretraining_training_only",
                strokes, time_scale_to_ms=1000.0, extra=task,
            )


def _sources(names: set[str], bdshwa_archive: Path) -> dict[str, Iterable[SourceSample]]:
    factories: dict[str, Any] = {
        "project_owned": _project_owned,
        "uji": _uji,
        "isgl": _isgl,
        "uci": _uci,
        "hwrt": _hwrt,
        "bdshwa": lambda: _bdshwa(bdshwa_archive),
    }
    unknown = names - factories.keys()
    if unknown:
        raise ValueError(f"unknown sources: {sorted(unknown)}")
    return {name: factories[name]() for name in SOURCE_ORDER if name in names}


@contextmanager
def _gzip_text(path: Path):
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", compresslevel=6, mtime=0) as zipped, io.TextIOWrapper(zipped, encoding="utf-8", newline="\n") as text:
            yield text
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_source(output: Path, file_key: str, source: str, samples: Iterable[SourceSample]) -> dict[str, Any]:
    path = output / f"{file_key}.jsonl.gz"
    counts, labels = Counter(), Counter()
    record_ids: set[str] = set()
    rejected: list[dict[str, Any]] = []
    with _gzip_text(path) as stream:
        for sample in samples:
            try:
                record = _canonicalize(sample)
            except ValueError as error:
                if str(error).startswith("empty_sample:"):
                    reason = "empty_after_source_pen_down_filter"
                elif str(error).startswith("insufficient_points:"):
                    reason = "insufficient_points_for_character_trajectory"
                else:
                    raise
                rejection = {
                    "schema": "aiflow-canonical-online-ink-rejection/v1",
                    "record_id": _record_id(sample.source, sample.source_id),
                    "source": sample.source,
                    "reason": reason,
                    "source_fingerprint": _digest([sample.source_id, reason]),
                }
                if sample.extra:
                    rejection["task"] = sample.extra
                rejected.append(rejection)
                continue
            if record["source"] != source:
                raise ValueError(f"unexpected source record: {record['source']} != {source}")
            if record["record_id"] in record_ids:
                raise ValueError(f"duplicate record ID: {record['record_id']}")
            record_ids.add(record["record_id"])
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":"), sort_keys=True))
            stream.write("\n")
            counts["records"] += 1
            counts["strokes"] += record["stroke_count"]
            counts["points"] += record["point_count"]
            counts[f"time_{record['normalization']['time_kind']}"] += 1
            counts[f"split_{record['split']}"] += 1
            labels[record["label"]] += 1
    result = {
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "counts": dict(sorted(counts.items())),
        "labels": len(labels),
    }
    if rejected:
        rejection_path = output / f"{file_key}_rejections.jsonl.gz"
        with _gzip_text(rejection_path) as stream:
            for row in rejected:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), sort_keys=True))
                stream.write("\n")
        result["rejections"] = {
            "file": rejection_path.name,
            "bytes": rejection_path.stat().st_size,
            "sha256": _sha256(rejection_path),
            "count": len(rejected),
        }
    return result


def _existing_entry(output: Path, source: str, prior: dict[str, Any]) -> dict[str, Any]:
    """Rebuild manifest metadata after an interrupted local derivative write."""
    path = output / str(prior["file"])
    counts, labels = Counter(), Counter()
    for row in _json_lines(path):
        if row.get("schema") != SCHEMA or row.get("source") != source:
            raise ValueError(f"schema/source mismatch while reconciling: {path}")
        counts["records"] += 1
        counts["strokes"] += int(row["stroke_count"])
        counts["points"] += int(row["point_count"])
        counts[f"time_{row['normalization']['time_kind']}"] += 1
        counts[f"split_{row['split']}"] += 1
        labels[str(row["label"])] += 1
    result = {"file": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path), "counts": dict(sorted(counts.items())), "labels": len(labels)}
    rejection_path = output / path.name.replace(".jsonl.gz", "_rejections.jsonl.gz")
    if rejection_path.is_file():
        rows = list(_json_lines(rejection_path))
        if any(row.get("schema") != "aiflow-canonical-online-ink-rejection/v1" or row.get("source") != source for row in rows):
            raise ValueError(f"rejection schema/source mismatch while reconciling: {rejection_path}")
        result["rejections"] = {"file": rejection_path.name, "bytes": rejection_path.stat().st_size, "sha256": _sha256(rejection_path), "count": len(rows)}
    return result


def reconcile_manifest(output: Path, drop_sources: set[str] | None = None) -> dict[str, Any]:
    """Refresh only manifest metadata; it never changes a source or derivative row."""
    output = output.resolve()
    manifest_path = output / "manifest.json"
    existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    dropped = {SOURCE_IDENTIFIERS.get(source, source) for source in (drop_sources or set())}
    sources = {
        source: _existing_entry(output, source, entry)
        for source, entry in existing.get("sources", {}).items()
        if source not in dropped
    }
    manifest = {
        "schema": "aiflow-canonical-online-ink-manifest/v1",
        "normalizer": {"spatial": SPATIAL_RULE, "time": TIME_RULE, "source_order": [name for name in SOURCE_ORDER if SOURCE_IDENTIFIERS[name] in sources]},
        "sources": sources,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return manifest


def build(output: Path, names: set[str], bdshwa_archive: Path, prune_unselected: bool = False, drop_sources: set[str] | None = None) -> dict[str, Any]:
    output = output.resolve()
    if output.drive.upper() != "D:":
        raise ValueError(f"normalization output must remain on D:: {output}")
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    existing = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    dropped = {SOURCE_IDENTIFIERS.get(source, source) for source in (drop_sources or set())}
    if names & (drop_sources or set()):
        raise ValueError("a source cannot be rebuilt and dropped in one command")
    files = {} if prune_unselected else {
        SOURCE_IDENTIFIERS.get(source, source): entry
        for source, entry in existing.get("sources", {}).items()
        if SOURCE_IDENTIFIERS.get(source, source) not in dropped
    }
    files.update({
        SOURCE_IDENTIFIERS[file_key]: _write_source(output, file_key, SOURCE_IDENTIFIERS[file_key], samples)
        for file_key, samples in _sources(names, bdshwa_archive).items()
    })
    manifest = {
        "schema": "aiflow-canonical-online-ink-manifest/v1",
        "normalizer": {"spatial": SPATIAL_RULE, "time": TIME_RULE, "source_order": [name for name in SOURCE_ORDER if SOURCE_IDENTIFIERS[name] in files]},
        "sources": files,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return manifest


def verify(output: Path) -> dict[str, Any]:
    output = output.resolve()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    seen_ids: set[str] = set()
    totals = Counter()
    for source, expected in manifest["sources"].items():
        path = output / expected["file"]
        if path.stat().st_size != expected["bytes"] or _sha256(path) != expected["sha256"]:
            raise AssertionError(f"artifact hash mismatch: {path}")
        local = Counter()
        for row in _json_lines(path):
            if row["schema"] != SCHEMA or row["source"] != source:
                raise AssertionError(f"schema/source mismatch: {row['record_id']}")
            if set(row) & FORBIDDEN_DERIVATIVE_KEYS:
                raise AssertionError(f"forbidden derivative key: {row['record_id']}")
            if "task" in row and set(row["task"]) - ALLOWED_BDSHWA_TASK_KEYS:
                raise AssertionError(f"forbidden task field: {row['record_id']}")
            if row["record_id"] in seen_ids:
                raise AssertionError(f"duplicate record ID: {row['record_id']}")
            seen_ids.add(row["record_id"])
            if row["normalization"]["spatial"] != SPATIAL_RULE or row["normalization"]["time"] != TIME_RULE:
                raise AssertionError(f"normalization contract mismatch: {row['record_id']}")
            orders = [stroke["source_order"] for stroke in row["strokes"]]
            if orders != sorted(orders) or len(orders) != len(set(orders)):
                raise AssertionError(f"stroke order mismatch: {row['record_id']}")
            points = [point for stroke in row["strokes"] for point in stroke["points"]]
            if len(points) != row["point_count"] or len(points) < 2:
                raise AssertionError(f"point count mismatch: {row['record_id']}")
            if any(len(point) != 3 or not all(math.isfinite(float(value)) and 0.0 <= float(value) <= 1.0 for value in point) for point in points):
                raise AssertionError(f"out-of-range canonical point: {row['record_id']}")
            if points[0][2] != 0.0 or (len(points) > 1 and points[-1][2] != 1.0):
                raise AssertionError(f"time endpoint mismatch: {row['record_id']}")
            if not isinstance(row["normalization"].get("removed_consecutive_exact_xy_points"), int):
                raise AssertionError(f"deduplication metadata missing: {row['record_id']}")
            local["records"] += 1; local["strokes"] += row["stroke_count"]; local["points"] += row["point_count"]
            local[f"time_{row['normalization']['time_kind']}"] += 1; local[f"split_{row['split']}"] += 1
        expected_counts = {key: value for key, value in expected["counts"].items() if not key.startswith("split_")}
        observed_counts = {key: value for key, value in local.items() if not key.startswith("split_")}
        if observed_counts != expected_counts:
            raise AssertionError(f"count mismatch for {source}: {observed_counts} != {expected_counts}")
        if rejection := expected.get("rejections"):
            rejection_path = output / rejection["file"]
            if rejection_path.stat().st_size != rejection["bytes"] or _sha256(rejection_path) != rejection["sha256"]:
                raise AssertionError(f"rejection artifact hash mismatch: {rejection_path}")
            rows = list(_json_lines(rejection_path))
            if len(rows) != rejection["count"] or any(row["schema"] != "aiflow-canonical-online-ink-rejection/v1" for row in rows):
                raise AssertionError(f"rejection schema/count mismatch: {rejection_path}")
            if any(set(row) & FORBIDDEN_DERIVATIVE_KEYS for row in rows):
                raise AssertionError(f"forbidden rejection key: {rejection_path}")
        totals.update(local)
    return {"records": totals["records"], "strokes": totals["strokes"], "points": totals["points"], "sources": len(manifest["sources"])}


def _self_test() -> None:
    sample = SourceSample("test", "A", "x", "train", "test", [[(10, 20, 2), (20, 20, 4)], [], [(20, 30, 6)]])
    first, second = _canonicalize(sample), _canonicalize(sample)
    assert first == second
    assert first["dropped_empty_stroke_orders"] == [1]
    assert [stroke["source_order"] for stroke in first["strokes"]] == [0, 2]
    assert first["strokes"][0]["points"][0] == [0.0, 0.0, 0.0]
    assert first["strokes"][-1]["points"][-1] == [1.0, 1.0, 1.0]
    stitched = _canonicalize(SourceSample("test", "B", "x", "train", "test", [[(0, 0, 10), (1, 0, 12)], [(1, 1, 5), (1, 2, 9)]]))
    assert stitched["normalization"]["time_kind"] == "stitched_stroke_duration"
    deduplicated = _canonicalize(SourceSample("test", "C", "x", "train", "test", [[(0, 0, 1), (0, 0, 2), (1, 0, 3)]], deduplicate_consecutive_exact_xy=True))
    assert deduplicated["normalization"]["removed_consecutive_exact_xy_points"] == 1
    rounded_duplicate = _canonicalize(SourceSample("test", "E", "x", "train", "test", [[(0, 0, 1), (1e-10, 0, 2), (1, 1, 3)]], deduplicate_consecutive_exact_xy=True))
    assert rounded_duplicate["normalization"]["removed_consecutive_exact_xy_points"] == 1
    assert rounded_duplicate["strokes"][0]["points"][0][2] == 0.0
    try:
        _canonicalize(SourceSample("test", "D", "x", "train", "test", [[(0, 0, 1)]]))
    except ValueError as error:
        assert str(error).startswith("insufficient_points:")
    else:
        raise AssertionError("one-point trajectory was accepted")


def _deterministic_replay(output: Path, names: set[str], bdshwa_archive: Path, prune_unselected: bool = False, drop_sources: set[str] | None = None) -> None:
    parent = output.resolve().parent
    replay = Path(tempfile.mkdtemp(prefix=".normalization-replay-", dir=parent))
    try:
        replay_manifest = build(replay, names, bdshwa_archive, prune_unselected=prune_unselected, drop_sources=drop_sources)
        original = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        if original != replay_manifest:
            raise AssertionError("normalization replay manifest differs")
    finally:
        shutil.rmtree(replay, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sources", default=",".join(CLASSIFIER_SOURCE_ORDER), help="comma-separated source names")
    parser.add_argument("--bdshwa-archive", type=Path, default=DEFAULT_BDSHWA_ARCHIVE)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--migrate-manifest", action="store_true")
    parser.add_argument("--reconcile-manifest", action="store_true", help="refresh hashes and counts from existing local derivative files without rewriting rows")
    parser.add_argument("--prune-unselected", action="store_true", help="drop unselected sources from the manifest; raw archives remain untouched")
    parser.add_argument("--drop-sources", default="", help="comma-separated source keys to remove from the manifest without touching raw archives")
    parser.add_argument("--deterministic-replay", action="store_true")
    parser.add_argument("--deterministic-replay-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test(); print(json.dumps({"self_test": "pass"})); return 0
    names = {part.strip() for part in args.sources.split(",") if part.strip()}
    drop_sources = {part.strip() for part in args.drop_sources.split(",") if part.strip()}
    if args.verify_only:
        print(json.dumps(verify(args.output), ensure_ascii=False, sort_keys=True)); return 0
    if args.reconcile_manifest:
        reconcile_manifest(args.output, drop_sources=drop_sources)
        print(json.dumps(verify(args.output), ensure_ascii=False, sort_keys=True)); return 0
    if args.migrate_manifest:
        build(args.output, set(), args.bdshwa_archive, drop_sources=drop_sources)
        print(json.dumps(verify(args.output), ensure_ascii=False, sort_keys=True)); return 0
    if args.deterministic_replay_only:
        result = verify(args.output)
        _deterministic_replay(args.output, names, args.bdshwa_archive, prune_unselected=args.prune_unselected, drop_sources=drop_sources)
        result["deterministic_replay"] = "pass"
        print(json.dumps(result, ensure_ascii=False, sort_keys=True)); return 0
    build(args.output, names, args.bdshwa_archive, prune_unselected=args.prune_unselected, drop_sources=drop_sources)
    result = verify(args.output)
    if args.deterministic_replay:
        _deterministic_replay(args.output, names, args.bdshwa_archive, prune_unselected=args.prune_unselected, drop_sources=drop_sources)
        result["deterministic_replay"] = "pass"
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
