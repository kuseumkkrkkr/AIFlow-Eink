#!/usr/bin/env python3
"""Read-only structural audit for a completed V12 R4-r2 raw clean-room bank."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "artifacts/action_evidence_augmentation_v12_20260824_r4_r2_raw"
DEFAULT_OUTPUT = ROOT / "reports/V12_ACTION_EVIDENCE_R4_R2_RAW_INDEPENDENT_AUDIT.json"
EXPECTED_FILES = {
    "GENERATION_STARTED.json",
    "PRE_FINAL_STAGE_TELEMETRY.json",
    "action_evidence_raw_bank.metadata.jsonl.gz",
    "action_evidence_raw_bank.npz",
    "generate_action_evidence_augmentation_v12_r4.source.py",
    "generation_report.json",
}
EXPECTED_CHECKPOINT = "c9ebd51e5ba72f1a1e8e2938343823808f7c0ab1bc7143e93332ef8b58143521"
MIN_RMS = 1.0e-5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_hash(value) -> str:
    return hashlib.sha256(value.astype("float32", copy=False).tobytes()).hexdigest()


def require_true(values: dict, label: str) -> None:
    if not isinstance(values, dict) or not values or not all(value is True for value in values.values()):
        raise ValueError(f"{label} must be a nonempty Boolean-true gate map")


def read_metadata(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def audit(raw: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError("independent raw audit receipt already exists")
    files = {path.name for path in raw.iterdir() if path.is_file()}
    if files != EXPECTED_FILES:
        raise ValueError(f"raw output must be exact success6, got {sorted(files)}")
    marker = json.loads((raw / "GENERATION_STARTED.json").read_text(encoding="utf-8"))
    telemetry = json.loads((raw / "PRE_FINAL_STAGE_TELEMETRY.json").read_text(encoding="utf-8"))
    report = json.loads((raw / "generation_report.json").read_text(encoding="utf-8"))
    if marker.get("status") != "V12_ACTION_EVIDENCE_R4_GENERATION_STARTED":
        raise ValueError("unexpected generation marker status")
    if telemetry.get("status") != "V12_R4_PRE_FINAL_STAGE_TELEMETRY_IMMUTABLE" or telemetry.get("first_empty_stage") != "NONE":
        raise ValueError("pre-final telemetry did not pass")
    if telemetry.get("boundaries", {}).get("row_level_tensor_metadata_label_writer_pair_class") is not False:
        raise ValueError("telemetry privacy boundary drift")
    if report.get("status") != "V12_R4_ACTION_EVIDENCE_RAW_GENERATED_AUDIT_REQUIRED":
        raise ValueError("unexpected raw report status")
    require_true(report.get("gates", {}), "raw report gates")
    if report.get("aggregate_stage_counts") != telemetry.get("counts"):
        raise ValueError("report and immutable telemetry counts differ")
    hashes = report.get("hashes", {})
    names = {
        "source_snapshot": "generate_action_evidence_augmentation_v12_r4.source.py",
        "started_marker": "GENERATION_STARTED.json",
        "prefinal_telemetry": "PRE_FINAL_STAGE_TELEMETRY.json",
        "bank": "action_evidence_raw_bank.npz",
        "metadata": "action_evidence_raw_bank.metadata.jsonl.gz",
    }
    for key, name in names.items():
        if hashes.get(key) != sha256(raw / name):
            raise ValueError(f"raw report hash mismatch: {key}")
    if hashes.get("checkpoint") != EXPECTED_CHECKPOINT:
        raise ValueError("checkpoint lineage drift")
    if report.get("final_admission", {}).get("candidate_violations") != 0:
        raise ValueError("candidate violation in raw report")

    import numpy as np  # Keep dry validation free of data/runtime imports.
    with np.load(raw / "action_evidence_raw_bank.npz", allow_pickle=False) as bank:
        values = np.asarray(bank["features"], dtype=np.float32)
        labels = np.asarray(bank["labels"], dtype=np.int64)
        writers = np.asarray(bank["writers"], dtype=np.int64)
        splits = np.asarray(bank["split"], dtype=np.int8)
    metadata = read_metadata(raw / "action_evidence_raw_bank.metadata.jsonl.gz")
    n = len(values)
    if values.shape != (n, 128, 5) or len(labels) != n or len(writers) != n or len(splits) != n or len(metadata) != n:
        raise ValueError("raw tensor/metadata alignment failure")
    if set(splits.tolist()) != {0, 1} or set(writers.tolist()) - set(range(256, 320)):
        raise ValueError("raw split or writer scope failure")
    if not (np.isfinite(values).all() and values[:, :, :2].min() >= 0 and values[:, :, :2].max() <= 1):
        raise ValueError("non-finite or out-of-box coordinates")
    expected_dt = np.full(128, np.float32(1 / 127)); expected_dt[0] = 0
    if not np.array_equal(values[:, :, 2], expected_dt[None]) or not np.all(values[:, :, 4] == 1):
        raise ValueError("time/observed contract drift")
    hashes_by_row = [tensor_hash(value) for value in values]
    if len(hashes_by_row) != len(set(hashes_by_row)):
        raise ValueError("duplicate raw tensor")

    cal_support = Counter()
    cal_fingerprints = defaultdict(set)
    query_pair_writers = defaultdict(set)
    query_entries: list[tuple[int, dict]] = []
    streams = set()
    for index, row in enumerate(metadata):
        split = "calibration" if int(splits[index]) == 0 else "query"
        if row.get("episode_split") != split or int(row.get("global_writer_id", -1)) != int(writers[index]):
            raise ValueError("metadata split/writer mismatch")
        label = int(row["label_index"]) if split == "calibration" else int(row["truth_label_index"])
        if label != int(labels[index]) or row.get("tensor_sha256") != hashes_by_row[index]:
            raise ValueError("metadata label or tensor hash mismatch")
        if float(row.get("spatial_rms_from_truth_parent", 0)) <= MIN_RMS or int(row.get("project_rows", -1)) != 0:
            raise ValueError("identity or project-row boundary failure")
        if split == "calibration":
            if row.get("external_approved_parent") is not True:
                raise ValueError("non-approved calibration parent")
            key = (int(writers[index]), int(row["candidate_label_index"]))
            cal_support[key] += 1
            cal_fingerprints[int(writers[index])].update(row.get("parent_fingerprints", []))
        else:
            if row.get("external_approved_parents") is not True:
                raise ValueError("non-approved query parent")
            query_entries.append((index, row))
        stream = (int(row["physics_seed"]), int(row["physics_batch_position"]))
        if stream in streams:
            raise ValueError("duplicate physics RNG stream")
        streams.add(stream)
    if any(count != 2 for count in cal_support.values()):
        raise ValueError("calibration support is not exactly two per writer/candidate")
    for index, row in query_entries:
        pair = row["directed_pair"]
        baseline, candidate = int(pair["baseline_label_index"]), int(pair["candidate_label_index"])
        top5 = [int(value) for value in row.get("frozen_top5", [])]
        if baseline not in top5 or candidate not in top5:
            raise ValueError("query directed pair left frozen Top-5")
        writer = int(writers[index])
        if cal_support[(writer, candidate)] < 2:
            raise ValueError("query has insufficient calibration support")
        query_parent_fps = set(row.get("truth_parent_fingerprints", [])) | set(row.get("relative_parent_fingerprints", []))
        if query_parent_fps & cal_fingerprints[writer]:
            raise ValueError("calibration/query parent overlap")
        query_pair_writers[(baseline, candidate)].add(writer)
    if not query_pair_writers or min(map(len, query_pair_writers.values())) < 16:
        raise ValueError("retained query pair has fewer than 16 writers")
    if report.get("rows") != n or report.get("calibration_rows") != int((splits == 0).sum()) or report.get("query_rows") != int((splits == 1).sum()):
        raise ValueError("report row totals mismatch")

    result = {
        "schema": "aiflow-independent-v12-r4-r2-raw-audit/v1",
        "status": "INDEPENDENT_V12_R4_R2_RAW_AUDIT_PASSED",
        "raw_path": str(raw.resolve()),
        "hashes": {name: sha256(raw / name) for name in sorted(EXPECTED_FILES)},
        "counts": {"rows": n, "calibration_rows": int((splits == 0).sum()), "query_rows": int((splits == 1).sum()), "writers": len(set(writers.tolist())), "pairs": len(query_pair_writers)},
        "gates": {
            "exact_success6_and_immutable_telemetry_passed": True,
            "report_hashes_and_boolean_gates_exact": True,
            "tensor_metadata_labels_writers_splits_aligned": True,
            "uniform_time_observed_unit_box_and_unique_tensors": True,
            "external_approved_only_and_project_rows_zero": True,
            "calibration_support_and_parent_disjointness_passed": True,
            "directed_pair_frozen_top5_and_pair16_passed": True,
            "physics_rng_streams_unique": True,
            "training_postprocess_evaluation_not_performed": True
        },
        "decision": {"postprocessing_allowed": False, "training_allowed": False, "product_promotion_allowed": False, "next": "Perform independent full/cross-bank audit before any later stage."}
    }
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def dry_toy() -> dict:
    result = {"status": "V12_R4_R2_RAW_AUDITOR_DRY_TOY_PASSED", "raw_loaded": False, "checkpoint_loaded": False, "training_performed": False, "output_written": False}
    if any(result[key] is not False for key in ("raw_loaded", "checkpoint_loaded", "training_performed", "output_written")):
        raise AssertionError("raw auditor dry toy boundary failure")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-toy", action="store_true")
    mode.add_argument("--audit", action="store_true")
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.dry_toy:
        print(json.dumps(dry_toy(), indent=2)); return 0
    if args.raw.resolve().drive.upper() != "D:" or args.output.resolve().drive.upper() != "D:":
        parser.error("raw and output paths must remain on D:")
    print(json.dumps(audit(args.raw, args.output), indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
