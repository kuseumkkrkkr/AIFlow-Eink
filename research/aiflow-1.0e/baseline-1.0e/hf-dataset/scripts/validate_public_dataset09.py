"""Fail-closed checks for the published AIFlow Math Ink dataset."""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path


FORBIDDEN = {"contributor_id", "session_id", "prompt_id", "recorded_at", "ink_png_data_url", "time_origin_ms", "raw_pressure", "source_path"}
STATUSES = ("valid", "pending", "reject")


def _walk(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk(item)


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate(root: Path) -> dict:
    manifest = json.loads((root / "dataset_info.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "aiflow-public-math-ink-manifest/v1"
    assert set(manifest["quality_status"]) == set(STATUSES)
    assert manifest["records"] > 0
    formulas = []
    for status in STATUSES:
        path = root / "data" / f"formulas_{status}.jsonl"
        rows = _rows(path)
        assert len(rows) == manifest["quality_status"][status]
        assert all(row["quality_status"] == status for row in rows)
        formulas.extend(rows)
    assert len(formulas) == manifest["records"] == sum(manifest["quality_status"].values())
    assert len({row["sample_id"] for row in formulas}) == len(formulas)
    assert not (FORBIDDEN & set(_walk(formulas)))
    assert all(row["stroke_count"] == len(row["strokes"]) for row in formulas)
    assert all(row["point_count"] == sum(len(stroke["points"]) for stroke in row["strokes"]) for row in formulas)
    assert all(min(float(point["t_ms"]) for stroke in row["strokes"] for point in stroke["points"]) == 0.0 for row in formulas if row["point_count"])
    assert len({row["writer_id"] for row in formulas}) == manifest["writers"]
    assert len({row["session_group"] for row in formulas}) == manifest["sessions"]
    assert Counter(row["source_partition"] for row in formulas) == manifest["sources"]
    assert Counter(
        str(cell["token"])
        for row in formulas
        for cell in row["target_cells"]
    ) == manifest["target_label_counts"]
    ownership = _rows(root / "data" / "ownership_train.jsonl")
    by_id = {row["sample_id"]: row for row in formulas}
    assert len(ownership) == manifest["ownership_train_formulas"]
    assert len({row["sample_id"] for row in ownership}) == len(ownership)
    for row in ownership:
        formula = by_id[row["sample_id"]]
        groups = row["groups"]
        stroke_ids = [stroke_id for group in groups for stroke_id in group]
        assert row["schema"] == "aiflow-public-ownership/v1"
        assert row["accepted"] is True
        assert formula["quality_status"] == "valid"
        assert row["writer_id"] == formula["writer_id"]
        assert len(groups) == len(row["labels"])
        assert all(group for group in groups)
        assert len(stroke_ids) == len(set(stroke_ids))
        assert set(stroke_ids) == {stroke["stroke_id"] for stroke in formula["strokes"]}
    data_files = {path.name for path in (root / "data").glob("*.jsonl")}
    assert set(manifest["files"]) == data_files
    for name, expected in manifest["files"].items():
        path = root / "data" / name
        assert path.stat().st_size == expected["bytes"]
        assert sha256(path.read_bytes()).hexdigest() == expected["sha256"]
    return {
        "records": len(formulas),
        "quality_status": manifest["quality_status"],
        "ownership": len(ownership),
        "forbidden_keys": 0,
        "status": "pass",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.root), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
