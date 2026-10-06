#!/usr/bin/env python3
"""Check whether unused project-owned formula data can support independent HWR acceptance."""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = ROOT / "artifacts" / "hwr_failure_cause_microscope_20260928" / "group_mean_geometry_prior_shadow_20261001" / "summary.json"
DEFAULT_DATA = Path(r"D:\AIFlow-Workspace\PrivateData\math-ink-data-collector\derived\fresh-context-acceptance-20260820-r2\frozen_acceptance\frozen_dataset\data\formulas_valid.jsonl")
DEFAULT_CANDIDATE_CACHE = Path(r"D:\AIFlow-Workspace\Projects\Aiflow\aiflow-math-ink-1.0\artifacts\homograph_context_20260814\direct_candidates.jsonl.gz")
DEFAULT_OUTPUT = ROOT / "artifacts" / "hwr_augmentation_microscope_20261002" / "formula_cohort_readiness_20261003.json"
ALLOWED_SOURCES = {"owned-phone-replay", "public-web-collection"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_no}") from exc
    return rows


def _writer_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def _candidate_cache_readiness(path: Path, consumed_formula_ids: set[str]) -> dict:
    """Count candidate-training rows left after formula and writer isolation."""
    by_formula: dict[str, list[dict]] = {}
    record_ids: set[str] = set()
    source_counts: Counter[str] = Counter()
    missing_top5 = 0
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid candidate-cache JSONL at {path}:{line_no}") from exc
            if "crohme" in json.dumps(row, ensure_ascii=False).casefold():
                raise ValueError("CROHME marker found in candidate cache; refusing readiness audit")
            source = str(row.get("source", "unknown"))
            source_counts[source] += 1
            if source != "project_owned":
                raise ValueError(f"unexpected candidate-cache source: {source}")
            record_id = str(row.get("record_id", ""))
            formula_id = str(row.get("formula_id", ""))
            writer_group = str(row.get("writer_group", ""))
            if not record_id or not formula_id or not writer_group:
                raise ValueError(f"candidate-cache row lacks identity fields at line {line_no}")
            if record_id in record_ids:
                raise ValueError(f"duplicate candidate-cache record ID at line {line_no}")
            record_ids.add(record_id)
            if row.get("label") not in row.get("final_topk", []):
                missing_top5 += 1
            by_formula.setdefault(formula_id, []).append(row)

    formula_writer_groups = {
        formula_id: {str(row["writer_group"]) for row in rows}
        for formula_id, rows in by_formula.items()
    }
    if any(len(groups) != 1 for groups in formula_writer_groups.values()):
        raise ValueError("one formula ID maps to multiple candidate-cache writer groups")
    overlapped_formula_ids = set(by_formula) & consumed_formula_ids
    consumed_writer_groups = {
        group
        for formula_id in overlapped_formula_ids
        for group in formula_writer_groups[formula_id]
    }
    formula_disjoint = [
        row for formula_id, rows in by_formula.items()
        if formula_id not in consumed_formula_ids
        for row in rows
    ]
    writer_and_formula_disjoint = [
        row for row in formula_disjoint
        if str(row["writer_group"]) not in consumed_writer_groups
    ]
    fresh_formula_ids = {str(row["formula_id"]) for row in writer_and_formula_disjoint}
    fresh_writer_groups = {str(row["writer_group"]) for row in writer_and_formula_disjoint}
    trainable_candidates = [
        row for row in writer_and_formula_disjoint
        if row.get("label") in row.get("final_topk", [])
    ]
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "rows": len(record_ids),
        "formula_count": len(by_formula),
        "writer_group_count": len({group for groups in formula_writer_groups.values() for group in groups}),
        "source_counts": dict(sorted(source_counts.items())),
        "crohme_rows": 0,
        "target_missing_from_top5_rows": missing_top5,
        "consumed_formula_overlap_count": len(overlapped_formula_ids),
        "remaining_formula_disjoint_rows": len(formula_disjoint),
        "remaining_formula_disjoint_formulas": len(set(by_formula) - consumed_formula_ids),
        "remaining_formula_disjoint_writer_groups": len({str(row["writer_group"]) for row in formula_disjoint}),
        "writer_groups_touched_by_consumed_formulas": len(consumed_writer_groups),
        "fresh_writer_and_formula_disjoint_rows": len(writer_and_formula_disjoint),
        "fresh_writer_and_formula_disjoint_formulas": len(fresh_formula_ids),
        "fresh_writer_and_formula_disjoint_writer_groups": len(fresh_writer_groups),
        "fresh_rankable_rows_with_target_in_top5": len(trainable_candidates),
        "independent_candidate_reranker_training_available": bool(fresh_formula_ids and fresh_writer_groups and trainable_candidates),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--candidate-cache", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite cohort readiness report: {args.output}")

    summary_path, data_path = args.summary.resolve(), args.data.resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("crohme_training_or_tuning") is not False or summary.get("product_default_enabled") is not False:
        raise ValueError("frozen evaluation summary lacks CROHME/product-off attestations")
    data_hash = _sha256(data_path)
    if data_hash != summary["inputs"]["formulas_valid_sha256"]:
        raise ValueError("formula data hash differs from the frozen evaluation summary")

    rows = _jsonl(data_path)
    raw_by_id = {str(row["sample_id"]): row for row in rows}
    if len(raw_by_id) != len(rows):
        raise ValueError("duplicate sample IDs in formula data")
    sources = Counter(str(row.get("source_partition", "unknown")) for row in rows)
    if not set(sources) <= ALLOWED_SOURCES:
        raise ValueError(f"unexpected formula source partition(s): {sorted(set(sources) - ALLOWED_SOURCES)}")
    crohme_rows = sum("crohme" in source.casefold() for source in sources.elements())
    if crohme_rows:
        raise ValueError("CROHME-marked source row found; refusing readiness audit")

    evaluated_ids = {str(record["sample_id"]) for record in summary["records"]}
    if len(evaluated_ids) != int(summary.get("formulas", len(evaluated_ids))):
        raise ValueError("frozen evaluation summary formula count does not match unique IDs")
    if not evaluated_ids <= set(raw_by_id):
        raise ValueError("frozen evaluation IDs are missing from formula data")
    evaluated_writers = {str(raw_by_id[sample_id]["writer_id"]) for sample_id in evaluated_ids}
    extras = [row for sample_id, row in raw_by_id.items() if sample_id not in evaluated_ids]
    extra_statuses = Counter(str(row.get("ownership_status", "unknown")) for row in extras)
    extra_sources = Counter(str(row.get("source_partition", "unknown")) for row in extras)
    unseen_writer_rows = [row for row in extras if str(row["writer_id"]) not in evaluated_writers]
    unseen_writer_hashes = sorted({_writer_hash(str(row["writer_id"])) for row in unseen_writer_rows})
    evaluable_extras = [row for row in extras if str(row.get("ownership_status", "")).startswith("accepted")]
    candidate_cache = (
        _candidate_cache_readiness(args.candidate_cache.resolve(), evaluated_ids)
        if args.candidate_cache is not None else None
    )

    report = {
        "schema": "aiflow-formula-cohort-readiness/v1",
        "status": "no_independent_formula_acceptance_rows_available",
        "protocol": {
            "training_performed": False,
            "model_selection_performed": False,
            "crohme_rows_loaded": 0,
            "product_adopted": False,
            "note": "the evaluation summary's 149 formulas are treated as consumed development data; unowned/pending rows are not scored as strict formula acceptance",
        },
        "provenance": {
            "summary_sha256": _sha256(summary_path),
            "formula_data_sha256": data_hash,
            "summary_consumed_formula_count": len(evaluated_ids),
            "raw_valid_formula_count": len(rows),
            "formula_data_writer_count": len({str(row["writer_id"]) for row in rows}),
            "consumed_writer_count": len(evaluated_writers),
            "consumed_writer_hashes": sorted(_writer_hash(writer) for writer in evaluated_writers),
        },
        "unused_rows": {
            "count": len(extras),
            "ownership_status_counts": dict(sorted(extra_statuses.items())),
            "source_partition_counts": dict(sorted(extra_sources.items())),
            "unseen_writer_row_count": len(unseen_writer_rows),
            "unseen_writer_count": len(unseen_writer_hashes),
            "unseen_writer_hashes": unseen_writer_hashes,
            "accepted_ownership_rows": len(evaluable_extras),
            "independent_formula_acceptance_available": bool(unseen_writer_rows and evaluable_extras),
        },
        "candidate_reranker_training_readiness": candidate_cache,
        "product_adopted": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({
        "event": "formula_cohort_readiness_audit_complete",
        "report": str(args.output.resolve()),
        "consumed": len(evaluated_ids),
        "valid": len(rows),
        "unused": len(extras),
        "unseen_writer_rows": len(unseen_writer_rows),
        "accepted_unused_rows": len(evaluable_extras),
        "crohme_rows": 0,
        "independent_acceptance_available": report["unused_rows"]["independent_formula_acceptance_available"],
        "independent_candidate_reranker_training_available": (
            candidate_cache["independent_candidate_reranker_training_available"] if candidate_cache else None
        ),
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
