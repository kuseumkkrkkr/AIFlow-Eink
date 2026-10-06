"""Build the public, pseudonymized AIFlow Math Ink 0.9 dataset."""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path


DROP_POINT_FIELDS = {"time_origin_ms", "raw_pressure"}


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite(value):
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _points(record: dict) -> tuple[list[dict], int]:
    ordered = sorted(record["strokes"], key=lambda row: int(row["order"]))
    all_points = [point for stroke in ordered for point in stroke["points"]]
    origin = min((float(point.get("t_ms", 0.0)) for point in all_points), default=0.0)
    output, total = [], 0
    for stroke_order, stroke in enumerate(ordered):
        clean = []
        for point in stroke["points"]:
            row = {
                key: value for key, value in point.items()
                if key not in DROP_POINT_FIELDS and key != "t_ms" and (_finite(value) or isinstance(value, str))
            }
            row["t_ms"] = round(float(point.get("t_ms", 0.0)) - origin, 3)
            clean.append(row)
        output.append({"stroke_id": stroke_order, "order": stroke_order, "points": clean})
        total += len(clean)
    return output, total


def _aliases(values: list[str], prefix: str) -> dict[str, str]:
    return {value: f"{prefix}_{index:03d}" for index, value in enumerate(sorted(set(values)), 1)}


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows)
    path.write_bytes(payload.encode("utf-8"))


def build(args: argparse.Namespace) -> dict:
    arrival = _json(args.arrival_index)["records"]
    source_paths = sorted(args.archive_root.glob("samples/*/*.json"))
    raw = [(path, _json(path)) for path in source_paths]
    writers = _aliases([str(row.get("contributor_id") or row["session_id"]) for _, row in raw], "writer")
    sessions = _aliases([str(row["session_id"]) for _, row in raw], "session")
    path_to_id, key_to_id, original_key_to_id = {}, {}, {}
    output = {"valid": [], "pending": [], "reject": []}
    source_counts, label_counts = Counter(), Counter()

    for number, (path, record) in enumerate(raw, 1):
        relative = "samples/" + "/".join(path.parts[-2:])
        review = arrival[relative]
        status = str(review.get("decision") or "pending") if review.get("reviewStatus") == "reviewed" else "pending"
        if status not in output:
            raise ValueError(f"unexpected review status: {status}")
        sample_id = f"aiflow_{number:04d}"
        strokes, point_count = _points(record)
        writer_key = str(record.get("contributor_id") or record["session_id"])
        row = {
            "schema": "aiflow-public-math-ink/v1",
            "sample_id": sample_id,
            "writer_id": writers[writer_key],
            "session_group": sessions[str(record["session_id"])],
            "quality_status": status,
            "source_partition": str(record.get("source") or "unknown"),
            "target_display": str(record.get("target_display") or ""),
            "target_cells": record.get("target_cells") or [],
            "target_relations": record.get("target_relations") or [],
            "formula_cells": record.get("formula_cells") or [],
            "ownership_status": str(record.get("ownership_status") or "unreviewed"),
            "label_status": str(record.get("label_status") or "unknown"),
            "canvas": record["canvas"],
            "strokes": strokes,
            "stroke_count": len(strokes),
            "point_count": point_count,
            "pressure_available": any(_finite(point.get("pressure")) for stroke in strokes for point in stroke["points"]),
        }
        output[status].append(row)
        source_counts[row["source_partition"]] += 1
        label_counts.update(str(cell["token"]) for cell in row["target_cells"])
        path_to_id[str(path.resolve()).lower()] = sample_id
        key_to_id[(str(record["session_id"]), str(record["prompt_id"]))] = sample_id
        original_session_hash = str(record.get("original_session_hash") or "")
        original_prompt_id = str(record.get("original_prompt_id") or "")
        if original_session_hash and original_prompt_id:
            original_key = (original_session_hash, original_prompt_id)
            if original_key in original_key_to_id:
                raise ValueError("duplicate original replay identity")
            original_key_to_id[original_key] = sample_id

    ownership = _json(args.ownership_annotations)
    owned_rows = []
    by_id = {row["sample_id"]: row for rows in output.values() for row in rows}
    for row in ownership["rows"]:
        if not row.get("accepted"):
            continue
        sample_id = path_to_id[str(Path(row["path"]).resolve()).lower()]
        owned_rows.append({
            "schema": "aiflow-public-ownership/v1", "sample_id": sample_id,
            "writer_id": by_id[sample_id]["writer_id"], "groups": row["groups"],
            "labels": row["labels"], "accepted": True,
        })

    replay_rows = []
    replay_match_counts = Counter()
    replay_unmatched = 0
    if args.replay_source and args.replay_annotations:
        replay_annotations = _json(args.replay_annotations)
        excluded = set(replay_annotations.get("excluded", {}))
        for line in args.replay_source.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["prompt_id"] in excluded:
                continue
            key = (str(row["session_id"]), str(row["prompt_id"]))
            sample_id = key_to_id.get(key)
            match_method = "direct"
            if sample_id is None:
                original_key = (
                    sha256(str(row["session_id"]).encode("utf-8")).hexdigest(),
                    str(row["prompt_id"]),
                )
                sample_id = original_key_to_id.get(original_key)
                match_method = "original_session_sha256_prompt"
            if sample_id is None:
                replay_unmatched += 1
                continue
            replay_match_counts[match_method] += 1
            replay_rows.append({
                "schema": "aiflow-public-ownership/v1", "sample_id": sample_id,
                "writer_id": by_id[sample_id]["writer_id"],
                "groups": replay_annotations["groups"][row["prompt_id"]],
                "labels": [str(value) for value in row["expected"]], "accepted": True,
            })

    for status, rows in output.items():
        _write_jsonl(args.output / "data" / f"formulas_{status}.jsonl", rows)
    _write_jsonl(args.output / "data" / "ownership_train.jsonl", owned_rows)
    replay_path = args.output / "data" / "ownership_replay.jsonl"
    if replay_rows:
        _write_jsonl(replay_path, replay_rows)
    elif replay_path.exists():
        replay_path.unlink()
    manifest = {
        "schema": "aiflow-public-math-ink-manifest/v1",
        "records": sum(map(len, output.values())),
        "quality_status": {key: len(value) for key, value in output.items()},
        "sources": dict(sorted(source_counts.items())),
        "writers": len(writers), "sessions": len(sessions),
        "ownership_train_formulas": len(owned_rows), "ownership_replay_formulas": len(replay_rows),
        "ownership_replay_match": {
            "direct": replay_match_counts["direct"],
            "original_session_sha256_prompt": replay_match_counts["original_session_sha256_prompt"],
            "unmatched": replay_unmatched,
        },
        "target_label_counts": dict(sorted(label_counts.items())),
        "privacy": {
            "removed": ["contributor_id", "session_id", "prompt_id", "recorded_at", "ink_png_data_url", "time_origin_ms", "raw_pressure", "source_path"],
            "time_normalization": "each formula starts at t_ms=0",
            "aliases": "dataset-local writer and session groups; private mapping is not published",
        },
        "files": {},
    }
    for path in sorted((args.output / "data").glob("*.jsonl")):
        manifest["files"][path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path.read_bytes()).hexdigest()}
    (args.output / "dataset_info.json").write_bytes((json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--arrival-index", type=Path, required=True)
    parser.add_argument("--ownership-annotations", type=Path, required=True)
    parser.add_argument("--replay-source", type=Path)
    parser.add_argument("--replay-annotations", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
