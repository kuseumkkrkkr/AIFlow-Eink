#!/usr/bin/env python3
"""Freeze a strict writer/formula-disjoint auxiliary distillation bank.

The previously consumed two-writer acceptance set is admitted for training
only.  The original 95-formula / seven-writer bank remains the sole outer
evaluation partition.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path

from character_tensor_v1 import iter_direct_ownership_examples


SCHEMA = "aiflow-1.0e-auxiliary-distillation-data/v1"


def read_gz(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_gz(path: Path, rows: list[dict]) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8", newline="\n") as stream:
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
                    stream.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-raw", type=Path, required=True)
    parser.add_argument("--original-candidates", type=Path, required=True)
    parser.add_argument("--expanded-candidates", type=Path, required=True)
    parser.add_argument("--formulas", type=Path, required=True)
    parser.add_argument("--ownership", type=Path, required=True)
    parser.add_argument("--acceptance-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")

    original_raw = read_gz(args.original_raw)
    original = read_gz(args.original_candidates)
    expanded = read_gz(args.expanded_candidates)
    formulas = {str(row["sample_id"]): row for row in read_jsonl(args.formulas)}
    original_formula_ids = {str(row["formula_id"]) for row in original}
    original_displays = {str(formulas[formula_id]["target_display"]) for formula_id in original_formula_ids}

    auxiliary_all = [row for row in expanded if row.get("evaluation_partition") == "new_writer"]
    auxiliary = [
        row for row in auxiliary_all
        if str(formulas[str(row["formula_id"])]["target_display"]) not in original_displays
    ]
    excluded_ids = sorted({str(row["formula_id"]) for row in auxiliary_all} - {str(row["formula_id"]) for row in auxiliary})
    auxiliary_ids = {str(row["record_id"]) for row in auxiliary}
    generated = list(iter_direct_ownership_examples(args.formulas, args.ownership))
    auxiliary_raw = [row for row in generated if str(row["record_id"]) in auxiliary_ids]

    original_ids = {str(row["record_id"]) for row in original}
    original_raw_ids = {str(row["record_id"]) for row in original_raw}
    original_writers = {str(row["writer_group"]) for row in original}
    auxiliary_writers = {str(row["writer_group"]) for row in auxiliary}
    auxiliary_formula_ids = {str(row["formula_id"]) for row in auxiliary}
    auxiliary_displays = {str(formulas[formula_id]["target_display"]) for formula_id in auxiliary_formula_ids}

    checks = {
        "original_candidate_raw_ids_match": original_ids == original_raw_ids,
        "auxiliary_generated_raw_coverage": auxiliary_ids == {str(row["record_id"]) for row in auxiliary_raw},
        "record_id_overlap_zero": not (original_ids & auxiliary_ids),
        "formula_id_overlap_zero": not (original_formula_ids & auxiliary_formula_ids),
        "formula_display_overlap_zero": not (original_displays & auxiliary_displays),
        "writer_overlap_zero": not (original_writers & auxiliary_writers),
        "auxiliary_writers_two": len(auxiliary_writers) == 2,
    }
    if not all(checks.values()):
        raise ValueError(f"distillation boundary failed: {checks}")

    args.output.mkdir(parents=True)
    outputs = {
        "auxiliary_candidates": args.output / "auxiliary_candidates.jsonl.gz",
        "combined_candidates": args.output / "combined_candidates.jsonl.gz",
        "combined_raw": args.output / "combined_raw.jsonl.gz",
    }
    write_gz(outputs["auxiliary_candidates"], auxiliary)
    write_gz(outputs["combined_candidates"], original + auxiliary)
    write_gz(outputs["combined_raw"], original_raw + auxiliary_raw)

    manifest = {
        "schema": SCHEMA,
        "status": "training_only_shadow",
        "lineage": {
            "auxiliary_source_role": "previously consumed acceptance bank; never a fresh acceptance claim",
            "outer_evaluation": "original seven-writer 95-formula partition only",
            "exact_display_overlap_policy": "exclude complete auxiliary formula",
        },
        "counts": {
            "original": {"rows": len(original), "formulas": len(original_formula_ids), "writers": len(original_writers)},
            "auxiliary_before_formula_filter": {"rows": len(auxiliary_all), "formulas": len({str(row['formula_id']) for row in auxiliary_all})},
            "auxiliary_training": {"rows": len(auxiliary), "formulas": len(auxiliary_formula_ids), "writers": len(auxiliary_writers)},
            "combined": {"rows": len(original) + len(auxiliary), "formulas": len(original_formula_ids | auxiliary_formula_ids), "writers": len(original_writers | auxiliary_writers)},
        },
        "excluded_exact_display_formula_ids": excluded_ids,
        "checks": checks,
        "inputs": {str(path): sha256(path) for path in (
            args.original_raw, args.original_candidates, args.expanded_candidates,
            args.formulas, args.ownership, args.acceptance_manifest,
        )},
        "outputs": {name: {"path": str(path), "sha256": sha256(path)} for name, path in outputs.items()},
        "product_runtime_changed": False,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
