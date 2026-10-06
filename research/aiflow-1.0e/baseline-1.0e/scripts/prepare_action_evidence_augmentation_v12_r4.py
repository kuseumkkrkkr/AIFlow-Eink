#!/usr/bin/env python3
"""Prepare the V12 R4 telemetry-first contract without generation or model access."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
R3_RAW = ROOT / "artifacts/action_evidence_augmentation_v12_20260824_r3_raw"
R3_FAILURE_AUDIT = ROOT / "reports/V12_ACTION_EVIDENCE_R3_INDEPENDENT_FAILURE_AUDIT.json"
R4_DESIGN = ROOT / "reports/V12_ACTION_EVIDENCE_R4_TELEMETRY_FIRST_DESIGN_20260824.md"
R4_SUMMARY = ROOT / "reports/V12_ACTION_EVIDENCE_R4_TELEMETRY_FIRST_DRY_SUMMARY.json"
R4_RAW = ROOT / "artifacts/action_evidence_augmentation_v12_20260824_r4_raw"
R4_FINAL = ROOT / "artifacts/action_evidence_augmentation_v12_20260824_r4_final"

EXPECTED_R3_EXACT_FILES = (
    "GENERATION_FAILED.json",
    "GENERATION_STARTED.json",
    "generate_action_evidence_augmentation_v12_r3.source.py",
)
EXPECTED_R3_SOURCE_SHA256 = "23aef377213e07bc46629546caec3bb02f2f4b0891df466968f28e5eeacc9d4f"
EXPECTED_R3_MARKER_SHA256 = "b964b9d224b268b6fef3eb1952ecf3da9884e4b7029025d571fd513ad897ba0f"
EXPECTED_R3_FAILURE_SHA256 = "0cfbdd85693f4984a90198ee9a9390603979fa433e8e4e2a886a79953c24368b"
EXPECTED_R3_FAILURE_AUDIT_SHA256 = "793ee301ce01f1a3bb0dddfd51cb05d925b4c6e611a99bf81844c3bdc6d7a069"
EXPECTED_R3_FAILURE_AUDIT_STATUS = "INDEPENDENT_V12_R3_FAILURE_AUDIT_CONFIRMED"
EXPECTED_R4_DESIGN_SHA256 = "99d3dfb2a996390ea24f458b6d2ff1c413314a99bbed374996d03dd73ab360ad"
EXPECTED_R3_POLICY_CATALOG_SHA256 = "66390f17ca243b5f779a7065d483b955e6c5f3038e5cc8e593e3e7e559d2b360"
EXPECTED_R3_PAIR_COUNT = 9_867
EXPECTED_R3_ACTION_COUNT = 197_340
WRITER_RANGE = (256, 319)

CALIBRATION_SURVIVORS = (
    "requested_specs",
    "specs_with_parent_candidates",
    "specs_parent_disjoint",
    "specs_topology_valid",
    "specs_direct_top5",
    "specs_homograph_clear",
    "specs_duplicate_clear",
    "specs_support2_complete",
)
CALIBRATION_AUXILIARY = (
    "parent_candidate_rows",
    "topology_rejects",
    "invalid_tensor_rejects",
    "direct_top5_rejects",
    "homograph_rejects",
    "duplicate_rejects",
)
QUERY_SURVIVORS = (
    "planned_actions",
    "actions_calibration_supported",
    "actions_shared_topology",
    "actions_truth_parent_selected",
    "actions_relative_parent_selected",
    "actions_with_morph_candidate",
    "actions_direct_membership",
    "actions_risk_exact",
    "actions_homograph_clear",
    "actions_accepted",
)
QUERY_AUXILIARY = (
    "morph_attempts",
    "morph_topology_drops",
    "auxiliary_unsupported_telemetry",
    "invalid_tensor_rejects",
    "duplicate_rejects",
)
FINAL_SURVIVORS = (
    "query_rows_entering",
    "rows_stored_top5_exact",
    "rows_after_support_filter",
    "rows_after_pair16_filter",
)
FINAL_COUNTERS = (
    "calibration_rows_entering",
    "query_rows_entering",
    "rows_stored_top5_exact",
    "stored_top5_mismatch_drops",
    "rows_after_support_filter",
    "support_drops",
    "rows_after_pair16_filter",
    "pair_lt16_rows_dropped",
    "candidate_violations",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_lineage() -> dict:
    files = tuple(sorted(path.name for path in R3_RAW.iterdir() if path.is_file()))
    if files != EXPECTED_R3_EXACT_FILES:
        raise ValueError("R3 immutable failure output is not exact3")
    expected = {
        R3_RAW / "generate_action_evidence_augmentation_v12_r3.source.py": EXPECTED_R3_SOURCE_SHA256,
        R3_RAW / "GENERATION_FAILED.json": EXPECTED_R3_FAILURE_SHA256,
        R3_RAW / "GENERATION_STARTED.json": EXPECTED_R3_MARKER_SHA256,
        R3_FAILURE_AUDIT: EXPECTED_R3_FAILURE_AUDIT_SHA256,
        R4_DESIGN: EXPECTED_R4_DESIGN_SHA256,
    }
    for path, wanted in expected.items():
        if sha256(path) != wanted:
            raise ValueError(f"pinned lineage drift: {path.name}")
    failure = json.loads((R3_RAW / "GENERATION_FAILED.json").read_text(encoding="utf-8"))
    audit = json.loads(R3_FAILURE_AUDIT.read_text(encoding="utf-8"))
    if failure.get("retry_allowed") is not False:
        raise ValueError("R3 retry boundary drift")
    if failure.get("status") != "V12_ACTION_EVIDENCE_R3_GENERATION_FAILED_IMMUTABLE":
        raise ValueError("R3 failure status drift")
    if audit.get("status") != EXPECTED_R3_FAILURE_AUDIT_STATUS:
        raise ValueError("R3 failure audit status drift")
    if not audit.get("gates") or not all(value is True for value in audit["gates"].values()):
        raise ValueError("R3 failure audit is not all-gates PASS")
    if audit.get("failure", {}).get("first_empty_stage") != "INDETERMINATE_CALIBRATION_OR_QUERY":
        raise ValueError("R3 indeterminate bottleneck boundary drift")
    if audit.get("decision", {}).get("r3_retry_allowed") is not False:
        raise ValueError("R3 independent retry boundary drift")
    if R4_RAW.exists() or R4_FINAL.exists():
        raise FileExistsError("R4 generation outputs must remain absent during static design")
    return {
        "r3_source_sha256": EXPECTED_R3_SOURCE_SHA256,
        "r3_started_marker_sha256": EXPECTED_R3_MARKER_SHA256,
        "r3_failure_sha256": EXPECTED_R3_FAILURE_SHA256,
        "r3_failure_audit_sha256": EXPECTED_R3_FAILURE_AUDIT_SHA256,
        "r4_design_sha256": EXPECTED_R4_DESIGN_SHA256,
    }


def first_empty_stage(telemetry: dict) -> str:
    for section, names in (
        ("calibration", CALIBRATION_SURVIVORS),
        ("query", QUERY_SURVIVORS),
        ("final", FINAL_SURVIVORS),
    ):
        values = telemetry.get(section, {})
        previous = None
        for name in names:
            value = values.get(name)
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"missing or invalid aggregate counter: {section}.{name}")
            if previous is not None and value > previous:
                raise ValueError(f"non-monotone survivor counter: {section}.{name}")
            previous = value
            if value == 0:
                return f"{section}.{name}"
    violations = telemetry.get("final", {}).get("candidate_violations")
    if not isinstance(violations, int) or violations < 0:
        raise ValueError("missing or invalid aggregate counter: final.candidate_violations")
    return "final.candidate_violations" if violations else "NONE"


def blank_telemetry(fill: int = 1) -> dict:
    if fill < 0:
        raise ValueError("fill must be nonnegative")
    return {
        "calibration": {name: fill for name in CALIBRATION_SURVIVORS + CALIBRATION_AUXILIARY},
        "query": {name: fill for name in QUERY_SURVIVORS + QUERY_AUXILIARY},
        "final": {name: fill for name in FINAL_COUNTERS},
    }


def build_summary() -> dict:
    lineage = validate_lineage()
    calibration_empty = blank_telemetry()
    for name in CALIBRATION_SURVIVORS[4:]:
        calibration_empty["calibration"][name] = 0
    for name in QUERY_SURVIVORS:
        query_empty_value = 0
        calibration_empty["query"][name] = query_empty_value
    for name in FINAL_SURVIVORS:
        calibration_empty["final"][name] = 0
    query_empty = blank_telemetry()
    for name in QUERY_SURVIVORS[2:]:
        query_empty["query"][name] = 0
    for name in FINAL_SURVIVORS:
        query_empty["final"][name] = 0
    final_empty = blank_telemetry(); final_empty["final"]["rows_after_pair16_filter"] = 0
    success = blank_telemetry(); success["final"]["candidate_violations"] = 0
    candidate_violation = blank_telemetry()
    nonmonotone = blank_telemetry(); nonmonotone["query"]["actions_shared_topology"] = 2
    try:
        first_empty_stage(nonmonotone)
    except ValueError:
        nonmonotone_rejected = True
    else:
        nonmonotone_rejected = False
    tests = {
        "calibration_empty": first_empty_stage(calibration_empty),
        "query_empty": first_empty_stage(query_empty),
        "final_empty": first_empty_stage(final_empty),
        "candidate_violation": first_empty_stage(candidate_violation),
        "success": first_empty_stage(success),
        "nonmonotone_rejected": nonmonotone_rejected,
    }
    if tests != {
        "calibration_empty": "calibration.specs_direct_top5",
        "query_empty": "query.actions_shared_topology",
        "final_empty": "final.rows_after_pair16_filter",
        "candidate_violation": "final.candidate_violations",
        "success": "NONE",
        "nonmonotone_rejected": True,
    }:
        raise AssertionError("first_empty_stage deterministic toy failed")
    return {
        "schema": "aiflow-v12-r4-telemetry-first-static/v1",
        "status": "V12_R4_TELEMETRY_FIRST_STATIC_DESIGN_READY_AUDIT_REQUIRED",
        "scope": "design and aggregate telemetry schema only; no generation, physics, model forward, or training",
        "lineage": lineage,
        "fixed_policy": {
            "source": "V12 R3 policy unchanged",
            "r3_policy_catalog_sha256": EXPECTED_R3_POLICY_CATALOG_SHA256,
            "pairs": EXPECTED_R3_PAIR_COUNT,
            "planned_actions": EXPECTED_R3_ACTION_COUNT,
            "support_required": 2,
            "minimum_pair_writers": 16,
            "new_writer_range_proposed": list(WRITER_RANGE),
            "result_specific_threshold_class_writer_tuning": False,
        },
        "telemetry_contract": {
            "filename": "PRE_FINAL_STAGE_TELEMETRY.json",
            "publish": "exclusive temp create, flush, fsync, non-replacing atomic rename",
            "must_precede": ["compound empty guard", "GENERATION_FAILED.json"],
            "calibration_survivors": list(CALIBRATION_SURVIVORS),
            "calibration_auxiliary": list(CALIBRATION_AUXILIARY),
            "query_survivors": list(QUERY_SURVIVORS),
            "query_auxiliary": list(QUERY_AUXILIARY),
            "final_counters": list(FINAL_COUNTERS),
            "first_empty_stage_source": "aggregate counters only",
            "row_level_tensor_metadata_labels": False,
        },
        "deterministic_toys": tests,
        "gates": {
            "r3_exact3_failure_lineage_pinned": True,
            "r3_first_bottleneck_not_overclaimed": True,
            "r3_writer192_255_retry_forbidden": True,
            "r4_writer256_319_generation_not_started": True,
            "prefinal_aggregate_telemetry_required": True,
            "first_empty_stage_deterministic": True,
            "row_level_labels_excluded": True,
            "r3_policy_threshold_class_writer_rules_unchanged": True,
            "writers096_127_legacy_real_crohme_mathwriting_closed": True,
            "hwr_checkpoint_runtime_unchanged": True,
            "optimizer_backward_training_not_performed": True,
            "r4_raw_and_final_absent": True,
        },
        "decision": {
            "static_independent_audit_required": True,
            "generation_source_implementation_allowed": False,
            "generation_execution_allowed": False,
            "training_allowed": False,
            "product_promotion_allowed": False,
        },
    }


def write_summary(path: Path) -> dict:
    if path.exists():
        raise FileExistsError(f"static summary already exists: {path}")
    payload = build_summary()
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry", action="store_true")
    mode.add_argument("--write-summary", action="store_true")
    parser.add_argument("--summary", type=Path, default=R4_SUMMARY)
    args = parser.parse_args()
    payload = build_summary() if args.dry else write_summary(args.summary)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
