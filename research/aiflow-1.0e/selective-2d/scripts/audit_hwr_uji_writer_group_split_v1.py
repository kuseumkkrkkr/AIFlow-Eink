#!/usr/bin/env python3
"""Audit a reproducible writer-disjoint inner split using UJI's official train writers.

This split is intended only for scratch-initialized architecture experiments. It
must not be used with checkpoints or teacher logits trained on the full official
UJI train partition. The official UJI test writers remain untouched.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from build_normalized_ink_v1 import ROOT, _json_lines, _record_id, _sha256


DEFAULT_CANONICAL_ROOT = Path(
    r"D:\AIFlow-Workspace\PrivateData\math-ink-data-collector\derived"
    r"\hwr-augmentation-20261001\canonical-trainpool"
)
DEFAULT_CURATED = ROOT / "datasets" / "10_approved_external" / "uji_pen_characters_v2" / "derived" / "uji_math_curated.jsonl.gz"
DEFAULT_REJECTIONS = Path(
    r"D:\AIFlow-Workspace\PrivateData\math-ink-data-collector\derived"
    r"\hwr-augmentation-20261001\canonical-trainpool\uji_rejections.jsonl.gz"
)
DEFAULT_OUTPUT = ROOT / "artifacts" / "hwr_augmentation_microscope_20261002" / "uji_writer_group_split_seed20261002.json"
DEFAULT_CONSUMPTION_LEDGER = ROOT / "artifacts/hwr_human_boundary_readiness_20261005/uji_validation_consumption.json"


def _consumed_split_paths(ledger_path: Path) -> tuple[list[Path], int]:
    """Trust completed, hash-attested runs rather than split filenames alone."""
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    if ledger.get("schema") != "aiflow-hwr-uji-validation-consumption/v1" or ledger.get("status") != "verified_completed_cohorts":
        raise ValueError("invalid validation-consumption ledger")
    directory = (ROOT / ledger["evidence_directory"]).resolve()
    paths, writers, pool = [], set(), set()
    for item in ledger["evidence"]:
        split_path, report_path = directory / item["split"], directory / item["completed_report"]
        if _sha256(split_path) != item["split_sha256"] or _sha256(report_path) != item["completed_report_sha256"]:
            raise ValueError("validation-consumption proof hash mismatch")
        proof = json.loads(report_path.read_text(encoding="utf-8"))
        if not proof.get("status", "").startswith("completed") or proof.get("data", {}).get("writer_split_sha256") != item["split_sha256"]:
            raise ValueError("validation cohort has no completed matching experiment")
        split = json.loads(split_path.read_text(encoding="utf-8"))
        validation = set(split["inner_split"]["validation_writer_hashes"])
        full = validation | set(split["inner_split"]["training_writer_hashes"])
        if pool and pool != full:
            raise ValueError("consumed cohorts use inconsistent writer pools")
        pool = full
        writers.update(validation)
        paths.append(split_path)
    remaining = len(pool - writers)
    if len(pool) != ledger["official_train_writer_pool"] or len(writers) != ledger["consumed_validation_writers"] or remaining != ledger["remaining_unconsumed_writers"]:
        raise ValueError("validation-consumption ledger counts differ from completed evidence")
    return paths, remaining


def _writer_digest(writer_key: str) -> str:
    return hashlib.sha256(writer_key.encode("utf-8")).hexdigest()[:16]


def _count_rows(rows: list[dict[str, Any]], selected: set[str]) -> tuple[int, dict[str, int]]:
    labels = Counter(str(row["label"]) for row in rows if str(row["writer_key"]) in selected)
    return sum(labels.values()), dict(sorted(labels.items()))


def audit_partition(
    canonical_root: Path,
    curated_path: Path,
    rejection_path: Path,
    validation_writer_count: int,
    seed: int,
    excluded_validation_writer_hashes: set[str] | None = None,
) -> dict[str, Any]:
    curated_by_id: dict[str, dict[str, str]] = {}
    for row in _json_lines(curated_path):
        record_id = _record_id("uji_pen_v2", str(row["sample_id"]))
        if record_id in curated_by_id:
            raise ValueError(f"duplicate curated UJI record ID: {record_id}")
        curated_by_id[record_id] = {
            "writer_key": str(row["writer_key"]),
            "split": str(row["split"]),
            "label": str(row["label"]),
        }

    canonical_rows: list[dict[str, Any]] = []
    canonical_ids: set[str] = set()
    for row in _json_lines(canonical_root / "uji.jsonl.gz"):
        record_id = str(row["record_id"])
        if record_id in canonical_ids:
            raise ValueError(f"duplicate canonical UJI record ID: {record_id}")
        canonical_ids.add(record_id)
        source_row = curated_by_id.get(record_id)
        if source_row is None:
            raise ValueError(f"canonical row cannot be joined to curated source: {record_id}")
        if str(row["label"]) != source_row["label"] or str(row["split"]) != source_row["split"]:
            raise ValueError(f"canonical/source metadata mismatch: {record_id}")
        canonical_rows.append({
            "record_id": record_id,
            "writer_key": source_row["writer_key"],
            "split": source_row["split"],
            "label": source_row["label"],
        })

    rejected_ids = {str(row["record_id"]) for row in _json_lines(rejection_path)}
    if canonical_ids & rejected_ids:
        raise ValueError("canonical accepted rows overlap the rejection set")
    if canonical_ids | rejected_ids != set(curated_by_id):
        raise ValueError(
            "curated rows are not fully accounted for by canonical acceptance plus rejection "
            f"(curated={len(curated_by_id)}, accepted={len(canonical_ids)}, rejected={len(rejected_ids)})"
        )

    train_rows = [row for row in canonical_rows if row["split"] == "train"]
    test_rows = [row for row in canonical_rows if row["split"] == "test"]
    train_writers = {str(row["writer_key"]) for row in train_rows}
    test_writers = {str(row["writer_key"]) for row in test_rows}
    if len(train_writers) != 40 or len(test_writers) != 20:
        raise ValueError(f"unexpected official writer counts: train={len(train_writers)}, test={len(test_writers)}")
    if train_writers & test_writers:
        raise ValueError("official UJI train/test writer overlap")
    if not 1 <= validation_writer_count < len(train_writers):
        raise ValueError("validation writer count must leave at least one training writer")
    if any(not writer.startswith("trn_") for writer in train_writers):
        raise ValueError("official train split contains an unexpected writer key")
    if any(not writer.startswith("tst_") for writer in test_writers):
        raise ValueError("official test split contains an unexpected writer key")

    excluded_hashes = set(excluded_validation_writer_hashes or ())
    train_hash_to_writer = {_writer_digest(writer): writer for writer in train_writers}
    unknown_excluded_hashes = excluded_hashes - set(train_hash_to_writer)
    if unknown_excluded_hashes:
        raise ValueError(f"prior validation hashes not present in official train writers: {sorted(unknown_excluded_hashes)}")
    eligible_validation_writers = train_writers - {
        train_hash_to_writer[value] for value in excluded_hashes
    }
    if validation_writer_count > len(eligible_validation_writers):
        raise ValueError(
            "requested validation cohort is larger than the remaining unconsumed writers: "
            f"requested={validation_writer_count}, available={len(eligible_validation_writers)}"
        )

    ordered = sorted(
        eligible_validation_writers,
        key=lambda writer: hashlib.sha256(f"{seed}:{writer}".encode("utf-8")).digest(),
    )
    validation_writers = set(ordered[:validation_writer_count])
    inner_train_writers = train_writers - validation_writers
    inner_train_rows, inner_train_labels = _count_rows(train_rows, inner_train_writers)
    validation_rows, validation_labels = _count_rows(train_rows, validation_writers)
    full_labels = sorted({str(row["label"]) for row in canonical_rows})
    train_missing = sorted(set(full_labels) - set(inner_train_labels))
    validation_missing = sorted(set(full_labels) - set(validation_labels))

    report = {
        "schema": "aiflow-hwr-uji-writer-group-split/v1",
        "status": "completed",
        "purpose": "scratch-only architecture comparison; not product or CROHME acceptance",
        "inputs": {
            "canonical_uji_path": str((canonical_root / "uji.jsonl.gz").resolve()),
            "canonical_uji_sha256": _sha256(canonical_root / "uji.jsonl.gz"),
            "curated_uji_path": str(curated_path.resolve()),
            "curated_uji_sha256": _sha256(curated_path),
            "rejections_path": str(rejection_path.resolve()),
            "rejections_sha256": _sha256(rejection_path),
        },
        "join_audit": {
            "curated_rows": len(curated_by_id),
            "canonical_accepted_rows": len(canonical_ids),
            "rejected_rows": len(rejected_ids),
            "all_source_rows_accounted_for": True,
            "label_and_official_split_match": True,
        },
        "official_split": {
            "train_writer_count": len(train_writers),
            "train_rows": len(train_rows),
            "test_writer_count": len(test_writers),
            "test_rows": len(test_rows),
            "train_test_writer_overlap": 0,
            "official_test_writers_used_for_inner_selection": False,
        },
        "inner_split": {
            "seed": seed,
            "validation_writer_count": len(validation_writers),
            "training_writer_count": len(inner_train_writers),
            "validation_row_count": validation_rows,
            "training_row_count": inner_train_rows,
            "validation_writer_hashes": sorted(_writer_digest(key) for key in validation_writers),
            "training_writer_hashes": sorted(_writer_digest(key) for key in inner_train_writers),
            "excluded_prior_validation_writer_count": len(excluded_hashes),
            "excluded_prior_validation_writer_hashes": sorted(excluded_hashes),
            "writer_overlap": 0,
            "class_count": len(full_labels),
            "training_label_counts": inner_train_labels,
            "validation_label_counts": validation_labels,
            "training_missing_labels": train_missing,
            "validation_missing_labels": validation_missing,
        },
        "candidate_protocol": {
            "initialization": "random initialization; do not load existing checkpoints",
            "teacher_distillation": "disabled; historical teacher may have seen these official train writers",
            "training_admission": "HWRT curated official train plus UJI official train writers outside this validation group",
            "selection_data": "inner UJI writer-disjoint validation rows only",
            "official_test": "reserved; do not score during architecture selection",
            "crohme_rows": 0,
            "product_adopted": False,
        },
    }
    if inner_train_writers & validation_writers or train_missing:
        raise AssertionError("invalid writer-disjoint inner split")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path, default=DEFAULT_CANONICAL_ROOT)
    parser.add_argument("--curated", type=Path, default=DEFAULT_CURATED)
    parser.add_argument("--rejections", type=Path, default=DEFAULT_REJECTIONS)
    parser.add_argument("--validation-writers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument(
        "--prior-split", type=Path, action="append", default=[],
        help="exclude validation writers from a previously completed split; may be repeated",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--require-unused-writers", action="store_true", help="enforce completed-run consumption evidence before claiming a fresh validation cohort")
    parser.add_argument("--consumption-ledger", type=Path, default=DEFAULT_CONSUMPTION_LEDGER)
    args = parser.parse_args()
    canonical_root = args.canonical_root.resolve()
    curated_path = args.curated.resolve()
    rejection_path = args.rejections.resolve()
    source_hashes = {
        "canonical_uji_sha256": _sha256(canonical_root / "uji.jsonl.gz"),
        "curated_uji_sha256": _sha256(curated_path),
        "rejections_sha256": _sha256(rejection_path),
    }
    if args.require_unused_writers:
        consumed_paths, remaining = _consumed_split_paths(args.consumption_ledger)
        for path in consumed_paths:
            prior_inputs = json.loads(path.read_text(encoding="utf-8"))["inputs"]
            if any(prior_inputs.get(key) != value for key, value in source_hashes.items()):
                raise ValueError("consumption ledger belongs to a different source; do not infer fresh-writer availability")
        if args.validation_writers > remaining:
            print(json.dumps({"status": "blocked_no_unused_validation_writers", "requested": args.validation_writers, "available": remaining, "raw_source_rows_read": 0, "crohme_rows": 0}))
            return 2
        args.prior_split = list(dict.fromkeys([*args.prior_split, *consumed_paths]))
    excluded_hashes: set[str] = set()
    prior_evidence = []
    for prior_path in (path.resolve() for path in args.prior_split):
        prior = json.loads(prior_path.read_text(encoding="utf-8"))
        if prior.get("schema") != "aiflow-hwr-uji-writer-group-split/v1" or prior.get("status") != "completed":
            raise ValueError(f"prior split manifest schema/status mismatch: {prior_path}")
        if prior.get("candidate_protocol", {}).get("crohme_rows") != 0:
            raise ValueError(f"prior split manifest lacks CROHME exclusion: {prior_path}")
        for key, value in source_hashes.items():
            if prior.get("inputs", {}).get(key) != value:
                raise ValueError(f"prior split manifest uses a different source ({key}): {prior_path}")
        hashes = set(str(value) for value in prior["inner_split"]["validation_writer_hashes"])
        overlap = excluded_hashes & hashes
        if overlap:
            raise ValueError(f"prior split manifests repeat validation writers: {sorted(overlap)}")
        excluded_hashes.update(hashes)
        prior_evidence.append({
            "path": str(prior_path),
            "sha256": _sha256(prior_path),
            "validation_writers": len(hashes),
        })

    report = audit_partition(
        canonical_root, curated_path, rejection_path, args.validation_writers, args.seed,
        excluded_validation_writer_hashes=excluded_hashes,
    )
    report["prior_validation_split_evidence"] = prior_evidence
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({
        "event": "uji_writer_group_split_audit_complete",
        "output": str(args.output.resolve()),
        "official_train_writers": report["official_split"]["train_writer_count"],
        "inner_train_writers": report["inner_split"]["training_writer_count"],
        "inner_validation_writers": report["inner_split"]["validation_writer_count"],
        "excluded_prior_validation_writers": report["inner_split"]["excluded_prior_validation_writer_count"],
        "inner_train_rows": report["inner_split"]["training_row_count"],
        "inner_validation_rows": report["inner_split"]["validation_row_count"],
        "validation_missing_labels": report["inner_split"]["validation_missing_labels"],
        "crohme_rows": report["candidate_protocol"]["crohme_rows"],
        "product_adopted": report["candidate_protocol"]["product_adopted"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
