#!/usr/bin/env python3
"""Create an append-only project-owned dataset with audited stroke regrouping."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


SCHEMA = "aiflow-project-owned-grouping-repair/v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _group_geometry(formula: dict, groups: list[list[int]]) -> list[dict]:
    strokes = sorted(formula["strokes"], key=lambda row: int(row["order"]))
    output = []
    for group in groups:
        points = [
            point for index in group for point in strokes[int(index)]["points"]
        ]
        xs = [float(point["x"]) for point in points]
        ys = [float(point["y"]) for point in points]
        output.append({
            "stroke_indices": [int(value) for value in group],
            "points": len(points),
            "bbox": [min(xs), min(ys), max(xs), max(ys)],
            "bbox_center_x": (min(xs) + max(xs)) / 2.0,
        })
    return output


def _validate_groups(
    sample_id: str, formula: dict, labels: list[str], groups: list[list[int]],
) -> list[dict]:
    if len(labels) != len(groups) or not groups or any(not group for group in groups):
        raise ValueError(f"invalid label/group cardinality: {sample_id}")
    stroke_count = len(formula["strokes"])
    flattened = [int(value) for group in groups for value in group]
    if sorted(flattened) != list(range(stroke_count)):
        raise ValueError(
            f"groups must partition every stroke exactly once: {sample_id}"
        )
    target_labels = [str(cell["token"]) for cell in formula["target_cells"]]
    if [str(value) for value in labels] != target_labels:
        raise ValueError(f"ownership labels differ from target cells: {sample_id}")
    geometry = _group_geometry(formula, groups)
    centers = [float(row["bbox_center_x"]) for row in geometry]
    if any(right <= left for left, right in zip(centers, centers[1:])):
        raise ValueError(f"repaired glyph centers are not left-to-right: {sample_id}")
    return geometry


def _verified_manifest_files(root: Path, info: dict) -> None:
    for name, expected in info.get("files", {}).items():
        path = root / "data" / name
        if not path.is_file():
            raise FileNotFoundError(f"manifest file missing: {path}")
        if path.stat().st_size != int(expected["bytes"]):
            raise ValueError(f"manifest byte mismatch: {path}")
        if _sha256(path) != str(expected["sha256"]):
            raise ValueError(f"manifest hash mismatch: {path}")


def repair(source_root: Path, repairs_path: Path, output_root: Path) -> dict:
    source_root = source_root.expanduser().resolve()
    repairs_path = repairs_path.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if source_root.drive.upper() != "D:" or output_root.drive.upper() != "D:":
        raise ValueError("source and output datasets must remain on D:")
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite output dataset: {output_root}")
    info_path = source_root / "dataset_info.json"
    formulas_path = source_root / "data" / "formulas_valid.jsonl"
    ownership_path = source_root / "data" / "ownership_train.jsonl"
    for path in (info_path, formulas_path, ownership_path, repairs_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    source_info = json.loads(info_path.read_text(encoding="utf-8"))
    _verified_manifest_files(source_root, source_info)
    repair_manifest = json.loads(repairs_path.read_text(encoding="utf-8"))
    if repair_manifest.get("schema") != SCHEMA:
        raise ValueError("unsupported repair manifest schema")
    repair_rows = repair_manifest.get("repairs", [])
    repair_by_id = {str(row["sample_id"]): row for row in repair_rows}
    if not repair_by_id or len(repair_by_id) != len(repair_rows):
        raise ValueError("repair sample IDs must be non-empty and unique")
    formulas = {str(row["sample_id"]): row for row in _rows(formulas_path)}
    missing = sorted(set(repair_by_id) - set(formulas))
    if missing:
        raise ValueError(f"repair formulas missing: {missing}")

    original_lines = ownership_path.read_text(encoding="utf-8").splitlines()
    output_lines = []
    audit = []
    seen_repairs: set[str] = set()
    for line in original_lines:
        annotation = json.loads(line)
        sample_id = str(annotation["sample_id"])
        requested = repair_by_id.get(sample_id)
        if requested is None:
            output_lines.append(line)
            continue
        if [str(value) for value in annotation["labels"]] != [
            str(value) for value in requested["labels"]
        ]:
            raise ValueError(f"repair label precondition failed: {sample_id}")
        old_groups = [[int(value) for value in group] for group in annotation["groups"]]
        expected_old = [
            [int(value) for value in group] for group in requested["old_groups"]
        ]
        if old_groups != expected_old:
            raise ValueError(f"repair old-group precondition failed: {sample_id}")
        new_groups = [
            [int(value) for value in group] for group in requested["new_groups"]
        ]
        formula = formulas[sample_id]
        old_geometry = _group_geometry(formula, old_groups)
        new_geometry = _validate_groups(
            sample_id, formula, annotation["labels"], new_groups,
        )
        annotation["groups"] = new_groups
        output_lines.append(json.dumps(
            annotation, ensure_ascii=False, separators=(",", ":"),
        ))
        seen_repairs.add(sample_id)
        audit.append({
            "sample_id": sample_id,
            "labels": annotation["labels"],
            "old_groups": old_groups,
            "new_groups": new_groups,
            "old_geometry": old_geometry,
            "new_geometry": new_geometry,
            "labels_changed": False,
            "strokes_changed": False,
            "stroke_order_changed": False,
            "formula_changed": False,
            "reason": requested.get("reason", ""),
        })
    if seen_repairs != set(repair_by_id):
        raise ValueError(f"repair annotations missing: {sorted(set(repair_by_id)-seen_repairs)}")

    output_data = output_root / "data"
    output_data.mkdir(parents=True)
    for source in sorted((source_root / "data").iterdir()):
        if source.name == "ownership_train.jsonl":
            continue
        if source.is_file():
            shutil.copy2(source, output_data / source.name)
    output_ownership = output_data / "ownership_train.jsonl"
    output_ownership.write_text(
        "\n".join(output_lines) + "\n", encoding="utf-8", newline="\n",
    )
    output_info = json.loads(json.dumps(source_info))
    for name in output_info["files"]:
        path = output_data / name
        output_info["files"][name] = {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    generated_at = datetime.now(timezone.utc).isoformat()
    output_info["curation"] = {
        "schema": SCHEMA,
        "generated_at": generated_at,
        "source_dataset_info_sha256": _sha256(info_path),
        "repair_manifest_sha256": _sha256(repairs_path),
        "ownership_groups_changed": len(audit),
        "labels_changed": 0,
        "strokes_changed": 0,
        "formulas_changed": 0,
        "policy": "audited ownership regrouping with complete stroke partition",
    }
    output_info_path = output_root / "dataset_info.json"
    output_info_path.write_text(
        json.dumps(output_info, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    report = {
        "schema": SCHEMA,
        "generated_at": generated_at,
        "source_dataset_info_sha256": _sha256(info_path),
        "output_dataset_info_sha256": _sha256(output_info_path),
        "repair_manifest_sha256": _sha256(repairs_path),
        "formulas_valid_sha256_unchanged": (
            _sha256(formulas_path)
            == _sha256(output_data / "formulas_valid.jsonl")
        ),
        "ownership_before_sha256": _sha256(ownership_path),
        "ownership_after_sha256": _sha256(output_ownership),
        "repair_count": len(audit),
        "repairs": audit,
        "invariants": {
            "labels_changed": 0,
            "strokes_changed": 0,
            "stroke_order_changed": 0,
            "formulas_changed": 0,
            "every_repaired_stroke_owned_exactly_once": True,
            "repaired_group_centers_left_to_right": True,
        },
    }
    report_path = output_root / "grouping_repair_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    return {
        "output_root": str(output_root),
        "dataset_info_sha256": _sha256(output_info_path),
        "report": str(report_path),
        "report_sha256": _sha256(report_path),
        "repair_count": len(audit),
        "invariants": report["invariants"],
    }


def _self_test() -> None:
    formula = {
        "target_cells": [{"token": "a"}, {"token": "b"}],
        "strokes": [
            {"order": 0, "points": [{"x": 0, "y": 0}, {"x": 1, "y": 1}]},
            {"order": 1, "points": [{"x": 10, "y": 0}, {"x": 11, "y": 1}]},
        ],
    }
    geometry = _validate_groups("test", formula, ["a", "b"], [[0], [1]])
    assert [row["bbox_center_x"] for row in geometry] == [0.5, 10.5]
    try:
        _validate_groups("bad", formula, ["a", "b"], [[0], [0]])
    except ValueError as error:
        assert "exactly once" in str(error)
    else:
        raise AssertionError("duplicate stroke ownership was accepted")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--repairs", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        print('{"self_test":"pass"}')
        return 0
    if args.source_root is None or args.repairs is None or args.output_root is None:
        parser.error("--source-root, --repairs, and --output-root are required")
    print(json.dumps(
        repair(args.source_root, args.repairs, args.output_root), ensure_ascii=False,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
