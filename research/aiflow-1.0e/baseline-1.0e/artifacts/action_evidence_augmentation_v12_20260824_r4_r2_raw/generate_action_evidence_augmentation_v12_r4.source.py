#!/usr/bin/env python3
"""R4 generator r2: strict Boolean audit gates; no policy or generation changes."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
R1_SOURCE = ROOT / "scripts/generate_action_evidence_augmentation_v12_r4.py"
EXPECTED_R1_SOURCE_SHA256 = "9e8767be92d1713053e571657de3f774fc67e7ef680741b39e6d89276700f50d"
R3_FAILURE_AUDIT = ROOT / "reports/V12_ACTION_EVIDENCE_R3_INDEPENDENT_FAILURE_AUDIT.json"
R4_R2_STATIC_AUDIT = ROOT / "reports/V12_ACTION_EVIDENCE_R4_R2_INDEPENDENT_STATIC_AUDIT.json"
EXPECTED_R3_FAILURE_AUDIT_SHA256 = "793ee301ce01f1a3bb0dddfd51cb05d925b4c6e611a99bf81844c3bdc6d7a069"
EXPECTED_R4_R2_STATIC_AUDIT_SHA256 = "c3329815d8e287c20b872fa58e6d392e57680e6a329f8767aabb372c6059640d"
EXPECTED_R3_FAILURE_STATUS = "INDEPENDENT_V12_R3_FAILURE_AUDIT_CONFIRMED"
EXPECTED_R4_STATIC_STATUS = "INDEPENDENT_V12_R4_R2_TELEMETRY_FIRST_STATIC_AUDIT_PASSED"
EXPECTED_PREGEN_STATUS = "INDEPENDENT_V12_R4_R2_GENERATOR_PREGENERATION_AUDIT_PASSED"
DEFAULT_RAW_OUTPUT = ROOT / "artifacts/action_evidence_augmentation_v12_20260824_r4_r2_raw"
DEFAULT_POSTPROCESSED_OUTPUT = ROOT / "artifacts/action_evidence_augmentation_v12_20260824_r4_r2_final"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if sha256(R1_SOURCE) != EXPECTED_R1_SOURCE_SHA256:
    raise RuntimeError("R4 generator r1 source drift before import")

r1gen = importlib.import_module("generate_action_evidence_augmentation_v12_r4")


def _strict_all_true(payload: dict, expected_status: str, label: str) -> None:
    gates = payload.get("gates")
    if payload.get("status") != expected_status:
        raise ValueError(f"{label} status mismatch")
    if not isinstance(gates, dict) or not gates:
        raise ValueError(f"{label} gates empty or non-object")
    if not all(value is True for value in gates.values()):
        raise ValueError(f"{label} gates must be Boolean true")


def _validate_static_lineage_strict() -> dict:
    if sha256(R3_FAILURE_AUDIT) != EXPECTED_R3_FAILURE_AUDIT_SHA256:
        raise ValueError("R3 failure audit drift")
    if sha256(R4_R2_STATIC_AUDIT) != EXPECTED_R4_R2_STATIC_AUDIT_SHA256:
        raise ValueError("R4-r2 static audit drift")
    r3_payload = json.loads(R3_FAILURE_AUDIT.read_text(encoding="utf-8"))
    r4_payload = json.loads(R4_R2_STATIC_AUDIT.read_text(encoding="utf-8"))
    _strict_all_true(r3_payload, EXPECTED_R3_FAILURE_STATUS, "R3 failure audit")
    _strict_all_true(r4_payload, EXPECTED_R4_STATIC_STATUS, "R4-r2 static audit")
    decision = r4_payload.get("decision", {})
    if decision.get("generation_source_implementation_allowed") is not True:
        raise ValueError("R4 generation source implementation not authorized")
    if decision.get("generation_execution_allowed") is not False:
        raise ValueError("R4 static generation execution boundary drift")
    if decision.get("training_allowed") is not False:
        raise ValueError("R4 static training boundary drift")
    return {
        "r3_failure_audit_sha256": EXPECTED_R3_FAILURE_AUDIT_SHA256,
        "r4_r2_static_audit_sha256": EXPECTED_R4_R2_STATIC_AUDIT_SHA256,
        "r4_generator_r1_source_sha256": EXPECTED_R1_SOURCE_SHA256,
        "strict_boolean_gate_validation": True,
    }


def _validate_pregeneration_audit_strict(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _strict_all_true(payload, EXPECTED_PREGEN_STATUS, "R4-r2 generator pre-generation audit")
    if payload.get("generator_source_sha256") != sha256(Path(__file__)):
        raise ValueError("R4-r2 pre-generation source mismatch")
    if payload.get("r4_r2_static_audit_sha256") != EXPECTED_R4_R2_STATIC_AUDIT_SHA256:
        raise ValueError("R4-r2 pre-generation lineage mismatch")
    decision = payload.get("decision", {})
    if decision.get("generation_allowed") is not True:
        raise ValueError("R4-r2 generation not authorized")
    if decision.get("training_allowed") is not False:
        raise ValueError("R4-r2 training boundary mismatch")
    return {"path": str(path.resolve()), "sha256": sha256(path), "status": payload["status"]}


def generate(output: Path, postprocessed: Path, audit_path: Path) -> None:
    original_file = r1gen.__file__
    original_static = r1gen._validate_static_lineage
    original_pregen = r1gen._validate_pregeneration_audit
    r1gen.__file__ = __file__
    r1gen._validate_static_lineage = _validate_static_lineage_strict
    r1gen._validate_pregeneration_audit = _validate_pregeneration_audit_strict
    try:
        r1gen.generate(output, postprocessed, audit_path)
    finally:
        r1gen.__file__ = original_file
        r1gen._validate_static_lineage = original_static
        r1gen._validate_pregeneration_audit = original_pregen


def dry_toy() -> dict:
    _validate_static_lineage_strict()
    good = {"status": "PASS", "gates": {"a": True, "b": True}}
    _strict_all_true(good, "PASS", "toy")
    rejected = {}
    for name, gates in {
        "empty": {}, "string": {"a": "true"}, "integer": {"a": 1},
        "false": {"a": False}, "null": {"a": None},
    }.items():
        try:
            _strict_all_true({"status": "PASS", "gates": gates}, "PASS", f"toy-{name}")
        except ValueError:
            rejected[name] = True
        else:
            rejected[name] = False
    inherited = r1gen.dry_toy()
    result = {
        "status": "V12_R4_R2_GENERATOR_DRY_TOY_PASSED",
        "strict_boolean_good_pass": True,
        "strict_rejection_matrix": rejected,
        "all_bad_gate_types_rejected": all(rejected.values()),
        "inherited_telemetry_contract_passed": inherited["status"] == "V12_R4_GENERATOR_DRY_TOY_PASSED",
        "catalog_loaded": False, "external_bank_loaded": False, "checkpoint_loaded": False,
        "physics_executed": False, "hwr_forward_executed": False, "training_performed": False,
        "policy_changed": False, "writers096_127_opened": False,
    }
    positive = {"strict_boolean_good_pass", "all_bad_gate_types_rejected", "inherited_telemetry_contract_passed"}
    negative = {
        "catalog_loaded", "external_bank_loaded", "checkpoint_loaded", "physics_executed",
        "hwr_forward_executed", "training_performed", "policy_changed", "writers096_127_opened",
    }
    if not all(result[key] is True for key in positive) or not all(result[key] is False for key in negative):
        raise AssertionError(f"R4-r2 dry toy failed: {result}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-toy", action="store_true")
    mode.add_argument("--generate", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_RAW_OUTPUT)
    parser.add_argument("--postprocessed-output", type=Path, default=DEFAULT_POSTPROCESSED_OUTPUT)
    parser.add_argument("--independent-audit", type=Path)
    args = parser.parse_args()
    if args.dry_toy:
        print(json.dumps(dry_toy(), indent=2)); return 0
    if args.independent_audit is None:
        parser.error("--generate requires --independent-audit")
    for path in (args.output.resolve(), args.postprocessed_output.resolve(), args.independent_audit.resolve()):
        if path.drive.upper() != "D:":
            parser.error("generation and audit paths must remain on D:")
    generate(args.output, args.postprocessed_output, args.independent_audit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
