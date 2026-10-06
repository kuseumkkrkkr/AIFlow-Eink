#!/usr/bin/env python3
"""Audit the 1.0e accuracy-upgrade input and split contracts.

이 명령은 학습하지 않는다. raw online ink와 후보 cache의 record coverage,
원본 writer 계보, 복원 가능한 formula-coordinate context, nested split 규모를
검사하고 기존 산출물을 덮어쓰지 않는 JSON 감사 결과를 생성한다.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path

from accuracy_upgrade_contract_v1 import (
    SCHEMA,
    nested_writer_splits,
    source_bbox,
    writer_key,
    formula_bounds,
    formula_position,
)


def _read_gz(path: Path) -> list[dict]:
    """UTF-8 JSONL gzip 파일을 읽는다."""
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def audit(raw_path: Path, candidates_path: Path, inner_folds: int = 3) -> dict:
    """입력 coverage와 위치 문맥 계약을 감사한다."""
    raw = _read_gz(raw_path)
    candidates = _read_gz(candidates_path)
    raw_by_id = {str(row["record_id"]): row for row in raw}
    candidate_by_id = {str(row["record_id"]): row for row in candidates}
    if len(raw_by_id) != len(raw) or len(candidate_by_id) != len(candidates):
        raise ValueError("raw and candidate record IDs must be unique")
    if set(raw_by_id) != set(candidate_by_id):
        raise ValueError("raw and candidate record coverage differs")

    formula_groups: dict[str, list[dict]] = {}
    for record_id, row in raw_by_id.items():
        formula_groups.setdefault(str(candidate_by_id[record_id]["formula_id"]), []).append(row)
        source_bbox(row)

    nonzero_context = 0
    context_rows = 0
    raw_lineage_mismatches = 0
    candidate_namespace_mismatches = 0
    invalid_geometry = []
    aliases: dict[str, set[str]] = {}
    reverse_aliases: dict[str, set[str]] = {}
    bounds_by_formula = {key: formula_bounds([source_bbox(row) for row in group]) for key, group in formula_groups.items()}
    for record_id, candidate in candidate_by_id.items():
        context = candidate.get("context") or {}
        context_rows += 1
        if any(abs(float(context.get(key, 0.0))) > 1e-9 for key in ("previous_dx", "previous_dy", "next_dx", "next_dy")):
            nonzero_context += 1
        if str(candidate.get("raw_writer_group")) != str(raw_by_id[record_id].get("writer_group")):
            raw_lineage_mismatches += 1
        if str(candidate.get("writer_group")) != str(raw_by_id[record_id].get("writer_group")):
            candidate_namespace_mismatches += 1
        expected = formula_position(source_bbox(raw_by_id[record_id]), bounds_by_formula[str(candidate['formula_id'])])
        geometry = candidate.get('geometry') or {}
        if any(not math.isfinite(float(geometry.get(key, float('nan')))) or abs(float(geometry.get(key, 0)) - value) > 1e-5 for key, value in expected.items()):
            invalid_geometry.append(record_id)
        if any(not math.isfinite(float(context.get(key, float('nan')))) for key in ('previous_dx', 'previous_dy', 'next_dx', 'next_dy')):
            invalid_geometry.append(record_id)
        alias, writer = str(candidate['writer_group']), writer_key(raw_by_id[record_id])
        aliases.setdefault(alias, set()).add(writer)
        reverse_aliases.setdefault(writer, set()).add(alias)
    alias_bijection = all(len(values) == 1 for values in list(aliases.values()) + list(reverse_aliases.values()))

    split_rows = [{"record_id": record_id, "writer_group": writer_key(raw_by_id[record_id])} for record_id in raw_by_id]
    splits = nested_writer_splits(split_rows, inner_folds=inner_folds)
    return {
        "schema": SCHEMA,
        "audit_schema": "aiflow-1.0e-accuracy-upgrade-audit/v1",
        "inputs": {"raw": str(raw_path), "candidates": str(candidates_path)},
        "coverage": {
            "raw_records": len(raw),
            "candidate_records": len(candidates),
            "formulas": len(formula_groups),
            "writers": len({writer_key(row) for row in raw}),
            "raw_lineage_mismatches": raw_lineage_mismatches,
            "candidate_namespace_mismatches_preserved": candidate_namespace_mismatches,
            "writer_alias_bijection": alias_bijection,
        },
        "geometry": {
            "source_bbox_rows": len(raw),
            "context_rows": context_rows,
            "nonzero_relative_context_rows": nonzero_context,
            "all_context_zero": nonzero_context == 0 and context_rows > 1,
            "invalid_formula_geometry_records": sorted(set(invalid_geometry)),
            "singleton_formulas": sum(len(group) == 1 for group in formula_groups.values()),
        },
        "nested_split": {
            "inner_folds": inner_folds,
            "outer_writers": len(splits),
            "splits": splits,
        },
        "status": (
            "pass"
            if raw_lineage_mismatches == 0 and alias_bijection and not invalid_geometry
            else "fail_geometry_or_lineage"
        ),
    }


def main() -> int:
    """CLI 진입점."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--inner-folds", type=int, default=3)
    args = parser.parse_args()
    report = audit(args.raw, args.candidates, args.inner_folds)
    if args.output is not None:
        if args.output.exists():
            parser.error(f"refusing to overwrite existing output: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
