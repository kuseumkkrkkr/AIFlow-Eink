#!/usr/bin/env python3
"""Build and verify the privacy-minimized, source-group-disjoint HWRT derivative."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import os
import tarfile
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    REPO_ROOT
    / "datasets"
    / "10_approved_external"
    / "hwrt"
    / "raw"
    / "2015-01-28-data.tar"
)
DEFAULT_OUTPUT = DEFAULT_SOURCE.parents[1] / "derived"
SOURCE_SHA256 = "B96FEAFD71B01F1623997DFF3CC8AC4D18628D128CEE1B3DF1880518BBA3EA4A"
SOURCE_URL = "https://zenodo.org/records/50022"
USER_ID_EVIDENCE_URL = "https://arxiv.org/abs/1701.08380"
DETEXIFY_AGGREGATE_USER_ID = "16925"
SPLITS = ("train", "validation", "test")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _canonical_number(value: Any) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("non_numeric_point")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non_finite_point")
    return int(number) if number.is_integer() else number


def _normalize_strokes(raw: Any) -> tuple[list[dict[str, Any]], int, str, int]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("empty_sample")

    first_time: int | float | None = None
    previous_time: int | float | None = None
    reversals = 0
    point_count = 0
    normalized: list[dict[str, Any]] = []

    for order, raw_stroke in enumerate(raw):
        if not isinstance(raw_stroke, list) or not raw_stroke:
            raise ValueError("empty_stroke")
        points: list[list[int | float]] = []
        for point in raw_stroke:
            if not isinstance(point, dict) or not {"x", "y", "time"} <= point.keys():
                raise ValueError("invalid_point_schema")
            x = _canonical_number(point["x"])
            y = _canonical_number(point["y"])
            timestamp = _canonical_number(point["time"])
            if first_time is None:
                first_time = timestamp
            if previous_time is not None and timestamp < previous_time:
                reversals += 1
            previous_time = timestamp
            points.append([x, y, _canonical_number(timestamp - first_time)])
            point_count += 1
        normalized.append({"trace_id": str(order), "order": order, "points": points})

    if reversals:
        error = ValueError("timestamp_reversal")
        error.reversals = reversals  # type: ignore[attr-defined]
        raise error
    assert first_time is not None
    time_mode = "epoch_ms" if first_time >= 100_000_000_000 else "relative_ms"
    return normalized, point_count, time_mode, reversals


def _writer_key(raw_user_id: str) -> str:
    if raw_user_id == DETEXIFY_AGGREGATE_USER_ID:
        return "aggregate_detexify_unknown_writers"
    digest = hashlib.sha256(f"hwrt-writer-v1\0{raw_user_id}".encode()).hexdigest()
    return f"writer_{digest[:16]}"


def _writer_split(raw_user_id: str) -> str:
    if raw_user_id == DETEXIFY_AGGREGATE_USER_ID:
        return "train"
    digest = hashlib.sha256(f"hwrt-split-v1\0{raw_user_id}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") / 2**64
    return "train" if bucket < 0.8 else "validation" if bucket < 0.9 else "test"


def _gzip_writer(path: Path) -> io.TextIOWrapper:
    raw = path.open("wb")
    compressed = gzip.GzipFile(filename="", fileobj=raw, mode="wb", compresslevel=6, mtime=0)
    return io.TextIOWrapper(compressed, encoding="utf-8", newline="\n")


def _write_json_line(stream: io.TextIOBase, row: dict[str, Any]) -> None:
    stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), sort_keys=True))
    stream.write("\n")


def _load_symbols(archive: tarfile.TarFile) -> dict[str, str]:
    source = archive.extractfile("symbols.csv")
    if source is None:
        raise ValueError("symbols.csv missing")
    rows = io.TextIOWrapper(source, encoding="utf-8-sig", newline="")
    header = rows.readline().rstrip("\r\n").split(";")
    if header != ["symbol_id", "latex", "training_samples", "test_samples"]:
        raise ValueError(f"unexpected symbols header: {header}")
    symbols: dict[str, str] = {}
    for line in rows:
        symbol_id, label, _train_count, _test_count = line.rstrip("\r\n").split(";", 3)
        symbols[symbol_id] = label
    return symbols


def _iter_source_rows(
    archive: tarfile.TarFile, member_name: str
) -> Iterable[tuple[int, list[str]]]:
    source = archive.extractfile(member_name)
    if source is None:
        raise ValueError(f"{member_name} missing")
    rows = io.TextIOWrapper(source, encoding="utf-8-sig", newline="")
    if rows.readline().rstrip("\r\n") != "symbol_id;user_id;data;user_agent":
        raise ValueError(f"unexpected header in {member_name}")
    for row_number, line in enumerate(rows, start=2):
        yield row_number, line.rstrip("\r\n").split(";", 3)


def build(source: Path, output: Path) -> dict[str, Any]:
    source = source.resolve()
    output = output.resolve()
    if source.drive.upper() != "D:" or output.drive.upper() != "D:":
        raise ValueError("HWRT source and derivative must remain on D:")
    if _sha256(source) != SOURCE_SHA256:
        raise ValueError("HWRT source SHA-256 mismatch")

    output.mkdir(parents=True, exist_ok=True)
    temporary = {name: output / f".{name}.jsonl.gz.tmp" for name in SPLITS}
    temporary["rejections"] = output / ".rejections.jsonl.gz.tmp"
    final = {name: output / f"{name}.jsonl.gz" for name in SPLITS}
    final["rejections"] = output / "rejections.jsonl.gz"

    source_rows = Counter()
    accepted_rows = Counter()
    rejection_reasons = Counter()
    source_time_modes = Counter()
    source_split_to_curated = defaultdict(Counter)
    split_writers = {name: set() for name in SPLITS}
    split_labels = {name: set() for name in SPLITS}
    seen_samples: dict[str, str] = {}
    accepted_strokes = 0
    accepted_points = 0
    reversal_events = 0
    group_type_counts = Counter()

    try:
        with tarfile.open(source, "r") as archive, ExitStack() as stack:
            symbols = _load_symbols(archive)
            writers = {
                name: stack.enter_context(_gzip_writer(path)) for name, path in temporary.items()
            }
            for source_split, member in (("train", "train-data.csv"), ("test", "test-data.csv")):
                for row_number, parts in _iter_source_rows(archive, member):
                    source_rows[source_split] += 1
                    rejection: dict[str, Any] | None = None
                    if len(parts) != 4:
                        rejection = {"reason": "malformed_source_row"}
                    else:
                        symbol_id, raw_user_id, stroke_json, _user_agent = parts
                        try:
                            if symbol_id not in symbols:
                                raise ValueError("unknown_symbol")
                            raw_strokes = json.loads(stroke_json)
                            strokes, point_count, time_mode, _ = _normalize_strokes(raw_strokes)
                            source_time_modes[time_mode] += 1
                            canonical = json.dumps(
                                [symbol_id, strokes], separators=(",", ":"), sort_keys=True
                            ).encode()
                            fingerprint = hashlib.sha256(canonical).hexdigest()
                            sample_id = f"hwrt_{fingerprint[:24]}"
                            if fingerprint in seen_samples:
                                rejection = {
                                    "reason": "duplicate_sample",
                                    "duplicate_of": seen_samples[fingerprint],
                                    "sample_fingerprint": fingerprint,
                                }
                            else:
                                seen_samples[fingerprint] = sample_id
                                split = _writer_split(raw_user_id)
                                writer_key = _writer_key(raw_user_id)
                                group_type = (
                                    "detexify_aggregate_unknown_writers"
                                    if raw_user_id == DETEXIFY_AGGREGATE_USER_ID
                                    else "source_user_id_unverified"
                                )
                                row = {
                                    "contributor_group_type": group_type,
                                    "eligible_for_training": True,
                                    "eligible_for_final_evaluation": False,
                                    "eligible_for_model_selection": False,
                                    "label": symbols[symbol_id],
                                    "license_id": "ODbL-1.0",
                                    "modality": "isolated_trajectory",
                                    "point_count": point_count,
                                    "sample_id": sample_id,
                                    "schema_version": "aiflow-external-trajectory-v1",
                                    "source": "zenodo-hwrt-2015-01-28",
                                    "source_partition": source_split,
                                    "source_url": SOURCE_URL,
                                    "split": split,
                                    "stroke_count": len(strokes),
                                    "strokes": strokes,
                                    "symbol_id": int(symbol_id),
                                    "timestamp_mode": "relative_ms",
                                    "timestamp_origin_removed": True,
                                    "track": "approved_external",
                                    "training_role": "supervised_math_symbol_candidate",
                                    "writer_key": writer_key,
                                }
                                _write_json_line(writers[split], row)
                                accepted_rows[split] += 1
                                source_split_to_curated[source_split][split] += 1
                                split_writers[split].add(writer_key)
                                split_labels[split].add(symbol_id)
                                accepted_strokes += len(strokes)
                                accepted_points += point_count
                                group_type_counts[group_type] += 1
                        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                            reason = str(error) or error.__class__.__name__
                            if reason == "timestamp_reversal":
                                reversal_events += int(getattr(error, "reversals", 1))
                            rejection = {"reason": reason}

                    if rejection is not None:
                        rejection_reasons[rejection["reason"]] += 1
                        rejection.update({"source_partition": source_split, "source_row": row_number})
                        _write_json_line(writers["rejections"], rejection)

        for name in (*SPLITS, "rejections"):
            os.replace(temporary[name], final[name])
    finally:
        for path in temporary.values():
            path.unlink(missing_ok=True)

    overlaps = {
        "train_validation": len(split_writers["train"] & split_writers["validation"]),
        "train_test": len(split_writers["train"] & split_writers["test"]),
        "validation_test": len(split_writers["validation"] & split_writers["test"]),
    }
    if any(overlaps.values()):
        raise AssertionError(f"writer split overlap: {overlaps}")

    files = {
        name: {"bytes": final[name].stat().st_size, "sha256": _sha256(final[name])}
        for name in (*SPLITS, "rejections")
    }
    manifest = {
        "schema_version": "aiflow-hwrt-curation-manifest-v1",
        "source": {
            "artifact": "raw/2015-01-28-data.tar",
            "sha256": SOURCE_SHA256,
            "url": SOURCE_URL,
            "license_id": "ODbL-1.0",
        },
        "policy": {
            "duplicate_key": "SHA-256(symbol_id + normalized ordered strokes)",
            "privacy": "drop user_agent, pseudonymize user_id, remove absolute timestamp origin",
            "rejection": "reject malformed/empty/non-finite samples and any cross-point time reversal",
            "split": "Detexify aggregate 16925 is train-only; other source IDs use SHA-256 80/10/10 group assignment",
            "split_limit": "source user IDs are unreliable; splits are group-disjoint but not proven writer-disjoint and cannot select or evaluate models",
            "role": "box-local mathematical-symbol candidate model only",
        },
        "source_user_id_caveat": {
            "evidence": USER_ID_EVIDENCE_URL,
            "finding": "HASYv2 reports that ID 16925 combines many Detexify contributors and that source user IDs cannot identify exact writers",
            "true_writer_disjoint": "unverifiable",
        },
        "counts": {
            "source_rows": dict(source_rows),
            "accepted_rows": dict(accepted_rows),
            "accepted_total": sum(accepted_rows.values()),
            "rejected_total": sum(rejection_reasons.values()),
            "rejection_reasons": dict(sorted(rejection_reasons.items())),
            "rejected_timestamp_reversal_events": reversal_events,
            "accepted_strokes": accepted_strokes,
            "accepted_points": accepted_points,
            "contributor_group_types": dict(group_type_counts),
            "source_time_modes": dict(source_time_modes),
            "source_to_curated_split": {
                source_name: dict(counts)
                for source_name, counts in source_split_to_curated.items()
            },
            "source_groups": {name: len(split_writers[name]) for name in SPLITS},
            "labels": {name: len(split_labels[name]) for name in SPLITS},
            "source_group_overlap": overlaps,
            "output_timestamp_reversals": 0,
        },
        "files": files,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def verify(output: Path) -> dict[str, Any]:
    output = output.resolve()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    split_writers = {name: set() for name in SPLITS}
    sample_ids: set[str] = set()
    counts = Counter()

    for split in SPLITS:
        path = output / f"{split}.jsonl.gz"
        expected = manifest["files"][split]
        if path.stat().st_size != expected["bytes"] or _sha256(path) != expected["sha256"]:
            raise AssertionError(f"artifact mismatch: {path}")
        with gzip.open(path, "rt", encoding="utf-8") as rows:
            for line in rows:
                row = json.loads(line)
                if row["split"] != split or row["timestamp_mode"] != "relative_ms":
                    raise AssertionError(f"split/time contract violation: {row['sample_id']}")
                if (
                    not row["eligible_for_training"]
                    or row["eligible_for_model_selection"]
                    or row["eligible_for_final_evaluation"]
                ):
                    raise AssertionError(f"eligibility contract violation: {row['sample_id']}")
                if "user_id" in row or "user_agent" in row:
                    raise AssertionError(f"privacy field leaked: {row['sample_id']}")
                if (
                    row["contributor_group_type"] == "detexify_aggregate_unknown_writers"
                    and split != "train"
                ):
                    raise AssertionError("Detexify aggregate escaped the training split")
                if row["sample_id"] in sample_ids:
                    raise AssertionError(f"duplicate sample ID: {row['sample_id']}")
                sample_ids.add(row["sample_id"])
                split_writers[split].add(row["writer_key"])
                previous: int | float | None = None
                if row["strokes"][0]["points"][0][2] != 0:
                    raise AssertionError(f"timestamp origin retained: {row['sample_id']}")
                for stroke in row["strokes"]:
                    for _x, _y, timestamp in stroke["points"]:
                        if previous is not None and timestamp < previous:
                            raise AssertionError(f"timestamp reversal: {row['sample_id']}")
                        previous = timestamp
                counts[split] += 1

    overlaps = {
        "train_validation": len(split_writers["train"] & split_writers["validation"]),
        "train_test": len(split_writers["train"] & split_writers["test"]),
        "validation_test": len(split_writers["validation"] & split_writers["test"]),
    }
    if any(overlaps.values()):
        raise AssertionError(f"writer split overlap: {overlaps}")
    if dict(counts) != manifest["counts"]["accepted_rows"]:
        raise AssertionError(f"row count mismatch: {dict(counts)}")

    rejection_path = output / "rejections.jsonl.gz"
    expected = manifest["files"]["rejections"]
    if (
        rejection_path.stat().st_size != expected["bytes"]
        or _sha256(rejection_path) != expected["sha256"]
    ):
        raise AssertionError("rejection artifact mismatch")
    with gzip.open(rejection_path, "rt", encoding="utf-8") as rows:
        rejected = sum(1 for line in rows if line.strip())
    if rejected != manifest["counts"]["rejected_total"]:
        raise AssertionError(f"rejection count mismatch: {rejected}")
    return {
        "rows": dict(counts),
        "source_groups": {k: len(v) for k, v in split_writers.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        result = verify(args.output)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return
    manifest = build(args.source, args.output)
    result = verify(args.output)
    print(json.dumps({"counts": manifest["counts"], "verified": result}, ensure_ascii=False))


if __name__ == "__main__":
    main()
