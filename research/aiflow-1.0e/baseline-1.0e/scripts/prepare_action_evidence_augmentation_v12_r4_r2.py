#!/usr/bin/env python3
"""Prepare the corrected V12 R4-r2 aggregate telemetry contract only."""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
R3_RAW = ROOT / "artifacts/action_evidence_augmentation_v12_20260824_r3_raw"
R3_FAILURE_AUDIT = ROOT / "reports/V12_ACTION_EVIDENCE_R3_INDEPENDENT_FAILURE_AUDIT.json"
R3_POLICY_CATALOG = ROOT / "reports/V12_ACTION_EVIDENCE_R3_DRY_POLICY_CATALOG.json.gz"
R4_R1_SOURCE = ROOT / "scripts/prepare_action_evidence_augmentation_v12_r4.py"
R4_R1_DESIGN = ROOT / "reports/V12_ACTION_EVIDENCE_R4_TELEMETRY_FIRST_DESIGN_20260824.md"
R4_R1_SUMMARY = ROOT / "reports/V12_ACTION_EVIDENCE_R4_TELEMETRY_FIRST_DRY_SUMMARY.json"
R4_R2_DESIGN = ROOT / "reports/V12_ACTION_EVIDENCE_R4_R2_TELEMETRY_FIRST_DESIGN_20260824.md"
R4_R2_SUMMARY = ROOT / "reports/V12_ACTION_EVIDENCE_R4_R2_TELEMETRY_FIRST_DRY_SUMMARY.json"
R4_R2_RAW = ROOT / "artifacts/action_evidence_augmentation_v12_20260824_r4_r2_raw"
R4_R2_FINAL = ROOT / "artifacts/action_evidence_augmentation_v12_20260824_r4_r2_final"

EXPECTED = {
    "r3_source": "23aef377213e07bc46629546caec3bb02f2f4b0891df466968f28e5eeacc9d4f",
    "r3_marker": "b964b9d224b268b6fef3eb1952ecf3da9884e4b7029025d571fd513ad897ba0f",
    "r3_failure": "0cfbdd85693f4984a90198ee9a9390603979fa433e8e4e2a886a79953c24368b",
    "r3_failure_audit": "793ee301ce01f1a3bb0dddfd51cb05d925b4c6e611a99bf81844c3bdc6d7a069",
    "r3_policy_catalog": "66390f17ca243b5f779a7065d483b955e6c5f3038e5cc8e593e3e7e559d2b360",
    "r4_r1_source": "4a7adf04878a000f98ce3b6185108b39f3f1d660c81dd33f1d911ef1dcc07bcf",
    "r4_r1_design": "99d3dfb2a996390ea24f458b6d2ff1c413314a99bbed374996d03dd73ab360ad",
    "r4_r1_summary": "d462fd14833e3151a59dd5d7c0a76fce6476410bce3ce6d825f7cbc959713fa9",
    "r4_r2_design": "1ea91413489999648f9af7a9ce500b9b10a1dc36f808b3c99342a9b94274c129",
}
R3_EXACT_FILES = (
    "GENERATION_FAILED.json",
    "GENERATION_STARTED.json",
    "generate_action_evidence_augmentation_v12_r3.source.py",
)
EXPECTED_PAIRS = 9_867
EXPECTED_ACTIONS = 197_340
R3_WRITERS = tuple(range(192, 256))
R4_R2_WRITERS = tuple(range(256, 320))

CALIBRATION_CHAIN = (
    "requested_specs", "specs_with_parent_candidates", "specs_parent_disjoint",
    "specs_topology_valid", "specs_direct_top5", "specs_homograph_clear",
    "specs_duplicate_clear", "specs_support2_complete",
)
CALIBRATION_AUXILIARY = (
    "parent_candidate_rows", "topology_rejects", "invalid_tensor_rejects",
    "direct_top5_rejects", "homograph_rejects", "duplicate_rejects",
)
QUERY_CHAIN = (
    "planned_actions", "actions_calibration_supported", "actions_shared_topology",
    "actions_truth_parent_selected", "actions_relative_parent_selected",
    "actions_with_morph_candidate", "actions_direct_membership", "actions_risk_exact",
    "actions_homograph_clear", "actions_accepted",
)
QUERY_AUXILIARY = (
    "morph_attempts", "morph_topology_drops", "auxiliary_unsupported_telemetry",
    "invalid_tensor_rejects", "duplicate_rejects",
)
FINAL_CALIBRATION_CHAIN = (
    "calibration_rows_entering", "calibration_rows_stored_top5_exact",
    "calibration_rows_after_support_retention",
)
FINAL_QUERY_CHAIN = (
    "query_rows_entering", "query_rows_stored_top5_exact",
    "query_rows_after_support_filter", "query_rows_after_pair16_filter",
)
FINAL_REJECTIONS = (
    "calibration_stored_top5_mismatch_drops", "calibration_support_retention_drops",
    "query_stored_top5_mismatch_drops", "query_support_drops",
    "query_pair_lt16_rows_dropped", "candidate_violations",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_file(path: Path, expected: str) -> None:
    if not path.is_file() or sha256(path) != expected:
        raise ValueError(f"pinned file drift: {path.name}")


def validate_lineage_and_catalog() -> dict:
    files = tuple(sorted(path.name for path in R3_RAW.iterdir() if path.is_file()))
    if files != R3_EXACT_FILES:
        raise ValueError("R3 immutable failure output is not exact3")
    paths = {
        "r3_source": R3_RAW / "generate_action_evidence_augmentation_v12_r3.source.py",
        "r3_marker": R3_RAW / "GENERATION_STARTED.json",
        "r3_failure": R3_RAW / "GENERATION_FAILED.json",
        "r3_failure_audit": R3_FAILURE_AUDIT,
        "r3_policy_catalog": R3_POLICY_CATALOG,
        "r4_r1_source": R4_R1_SOURCE,
        "r4_r1_design": R4_R1_DESIGN,
        "r4_r1_summary": R4_R1_SUMMARY,
        "r4_r2_design": R4_R2_DESIGN,
    }
    for name, path in paths.items():
        _verify_file(path, EXPECTED[name])
    audit = json.loads(R3_FAILURE_AUDIT.read_text(encoding="utf-8"))
    if (audit.get("status") != "INDEPENDENT_V12_R3_FAILURE_AUDIT_CONFIRMED"
            or not audit.get("gates") or not all(value is True for value in audit["gates"].values())):
        raise ValueError("R3 failure audit boundary drift")
    if audit.get("failure", {}).get("first_empty_stage") != "INDETERMINATE_CALIBRATION_OR_QUERY":
        raise ValueError("R3 first bottleneck was overclaimed")
    with gzip.open(R3_POLICY_CATALOG, "rt", encoding="utf-8") as stream:
        catalog = json.load(stream)
    global_policy = catalog.get("global_policy", {})
    expected_acceptance = [
        "baseline_in_generated_frozen_top5",
        "candidate_in_generated_frozen_top5",
        "catalog_risk_stratum_exact",
        "candidate_calibration_support_at_least_2",
        "nonidentity_spatial_change",
        "truth_topology_preserved",
        "calibration_query_parent_fingerprint_disjoint",
        "finite_unit_box_uniform_time_external_only",
    ]
    if global_policy.get("action_acceptance") != expected_acceptance:
        raise ValueError("R3 live catalog action acceptance drift")
    if global_policy.get("auxiliary_top5_unadmitted") != "telemetry_and_runtime_unsupported_identity_not_query_drop":
        raise ValueError("R3 live catalog auxiliary policy drift")
    if global_policy.get("homograph_collision") != "context_owned_whole_row_identity_not_action":
        raise ValueError("R3 live catalog homograph policy drift")
    if global_policy.get("bank_candidate_violations") != "must_equal_zero_or_fail_closed":
        raise ValueError("R3 live catalog candidate policy drift")
    pairs = catalog.get("pairs", [])
    if len(pairs) != EXPECTED_PAIRS:
        raise ValueError("R3 live catalog pair count drift")
    action_count = 0
    orientation = Counter()
    for row in pairs:
        actions = row.get("planned_actions", [])
        writers = [int(action["writer_id"]) for action in actions]
        if len(actions) != 20 or len(set(writers)) != 20 or not set(writers).issubset(R3_WRITERS):
            raise ValueError("R3 live catalog pair writer contract drift")
        local = Counter(str(action["orientation"]) for action in actions)
        if local != {"candidate_truth_promotion": 10, "baseline_truth_veto": 10}:
            raise ValueError("R3 live catalog orientation drift")
        orientation.update(local)
        action_count += len(actions)
    if action_count != EXPECTED_ACTIONS:
        raise ValueError("R3 live catalog action count drift")
    if R4_R2_RAW.exists() or R4_R2_FINAL.exists():
        raise FileExistsError("R4-r2 generation outputs must remain absent")
    return {
        "hashes": {name: EXPECTED[name] for name in paths},
        "live_catalog": {
            "pairs": len(pairs), "planned_actions": action_count,
            "orientation": dict(sorted(orientation.items())),
            "pair_unique_writers": 20, "support_required": 2, "minimum_pair_writers": 16,
            "global_policy_sha256": hashlib.sha256(
                json.dumps(global_policy, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        },
    }


def _validate_chain(section: str, values: dict, names: tuple[str, ...]) -> str | None:
    previous = None
    first_empty = None
    for name in names:
        value = values.get(name)
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"missing or invalid aggregate counter: {section}.{name}")
        if previous is not None and value > previous:
            raise ValueError(f"non-monotone survivor counter: {section}.{name}")
        previous = value
        if value == 0 and first_empty is None:
            first_empty = f"{section}.{name}"
    return first_empty


def first_empty_stage(telemetry: dict) -> str:
    for section, names in (
        ("calibration", CALIBRATION_CHAIN),
        ("query", QUERY_CHAIN),
        ("final_calibration", FINAL_CALIBRATION_CHAIN),
        ("final_query", FINAL_QUERY_CHAIN),
    ):
        empty = _validate_chain(section, telemetry.get(section, {}), names)
        if empty is not None:
            return empty
    violations = telemetry.get("final", {}).get("candidate_violations")
    if not isinstance(violations, int) or violations < 0:
        raise ValueError("missing or invalid aggregate counter: final.candidate_violations")
    return "final.candidate_violations" if violations else "NONE"


def blank_telemetry(fill: int = 1) -> dict:
    return {
        "calibration": {name: fill for name in CALIBRATION_CHAIN + CALIBRATION_AUXILIARY},
        "query": {name: fill for name in QUERY_CHAIN + QUERY_AUXILIARY},
        "final_calibration": {name: fill for name in FINAL_CALIBRATION_CHAIN},
        "final_query": {name: fill for name in FINAL_QUERY_CHAIN},
        "final": {name: (0 if name == "candidate_violations" else fill) for name in FINAL_REJECTIONS},
    }


def _zero_suffix(values: dict, names: tuple[str, ...], start: int) -> None:
    for name in names[start:]:
        values[name] = 0


def deterministic_toys() -> dict:
    calibration = blank_telemetry(); _zero_suffix(calibration["calibration"], CALIBRATION_CHAIN, 4)
    query = blank_telemetry(); _zero_suffix(query["query"], QUERY_CHAIN, 2)
    final_calibration = blank_telemetry(); _zero_suffix(final_calibration["final_calibration"], FINAL_CALIBRATION_CHAIN, 0)
    final_query = blank_telemetry(); _zero_suffix(final_query["final_query"], FINAL_QUERY_CHAIN, 0)
    candidate = blank_telemetry(); candidate["final"]["candidate_violations"] = 1
    success = blank_telemetry()
    nonmonotone = blank_telemetry(); nonmonotone["final_query"]["query_rows_stored_top5_exact"] = 2
    try:
        first_empty_stage(nonmonotone)
    except ValueError:
        nonmonotone_rejected = True
    else:
        nonmonotone_rejected = False
    result = {
        "calibration_empty": first_empty_stage(calibration),
        "query_empty": first_empty_stage(query),
        "final_calibration_empty": first_empty_stage(final_calibration),
        "final_query_empty": first_empty_stage(final_query),
        "candidate_violation": first_empty_stage(candidate),
        "success": first_empty_stage(success),
        "branch_nonmonotone_rejected": nonmonotone_rejected,
    }
    expected = {
        "calibration_empty": "calibration.specs_direct_top5",
        "query_empty": "query.actions_shared_topology",
        "final_calibration_empty": "final_calibration.calibration_rows_entering",
        "final_query_empty": "final_query.query_rows_entering",
        "candidate_violation": "final.candidate_violations",
        "success": "NONE",
        "branch_nonmonotone_rejected": True,
    }
    if result != expected:
        raise AssertionError(f"R4-r2 deterministic telemetry toys failed: {result}")
    return result


def build_summary() -> dict:
    provenance = validate_lineage_and_catalog()
    return {
        "schema": "aiflow-v12-r4-r2-telemetry-first-static/v1",
        "status": "V12_R4_R2_TELEMETRY_FIRST_STATIC_READY_AUDIT_REQUIRED",
        "source_sha256": sha256(Path(__file__)),
        "scope": "aggregate telemetry design only; no generation, physics, forward, training, or evaluation corpus",
        "provenance": provenance,
        "fixed_policy": {
            "writers_proposed": [min(R4_R2_WRITERS), max(R4_R2_WRITERS)],
            "r3_policy_unchanged": True, "threshold_class_writer_result_tuning": False,
        },
        "telemetry": {
            "filename": "PRE_FINAL_STAGE_TELEMETRY.json",
            "publish_before": ["compound empty guard", "GENERATION_FAILED.json"],
            "atomic_publish": "exclusive temp, flush, fsync, non-replacing rename",
            "calibration_chain": list(CALIBRATION_CHAIN),
            "query_chain": list(QUERY_CHAIN),
            "final_calibration_chain": list(FINAL_CALIBRATION_CHAIN),
            "final_query_chain": list(FINAL_QUERY_CHAIN),
            "final_rejections": list(FINAL_REJECTIONS),
            "cross_branch_size_comparison": False,
            "aggregate_counts_and_hashes_only": True,
            "row_level_tensor_metadata_label_writer_pair_class": False,
        },
        "deterministic_toys": deterministic_toys(),
        "gates": {
            "r3_failure_exact3_and_audit_pinned": True,
            "r3_live_policy_catalog_hash_and_counts_exact": True,
            "r4_r1_preserved_as_rejected_static_evidence": True,
            "calibration_and_query_final_chains_separate": True,
            "final_calibration_zero_detected_independently": True,
            "final_query_zero_detected_independently": True,
            "branch_local_monotonicity_enforced": True,
            "aggregate_only_atomic_prefinal_telemetry_required": True,
            "r3_policy_unchanged": True,
            "r4_r2_generation_outputs_absent": True,
            "writers096_127_legacy_real_crohme_mathwriting_closed": True,
            "hwr_checkpoint_runtime_unchanged": True,
            "generation_physics_forward_training_not_performed": True,
        },
        "decision": {
            "static_independent_audit_required": True,
            "generation_source_implementation_allowed": False,
            "generation_execution_allowed": False,
            "training_allowed": False,
            "product_promotion_allowed": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry", action="store_true")
    group.add_argument("--write-summary", action="store_true")
    parser.add_argument("--summary", type=Path, default=R4_R2_SUMMARY)
    args = parser.parse_args()
    payload = build_summary()
    if args.write_summary:
        if args.summary.exists():
            raise FileExistsError(f"R4-r2 summary already exists: {args.summary}")
        args.summary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
