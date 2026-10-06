"""Build the v12 R3 static action-acceptance policy catalog.

This script is deliberately dry-only.  It reads the immutable pre-generation
catalog and the R2 generation report, but never opens the R2 NPZ/metadata,
external tensors, checkpoint, or any evaluation corpus.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "reports/V12_ACTION_EVIDENCE_R3_DESIGN_20260824.md"
SOURCE_SUMMARY = ROOT / "reports/V12_ACTION_EVIDENCE_DRY_CATALOG_SUMMARY_R3.json"
SOURCE_CATALOG = ROOT / "reports/V12_ACTION_EVIDENCE_DRY_CATALOG_R3.json.gz"
R2_REPORT = ROOT / "artifacts/action_evidence_augmentation_v12_20260823_r2_raw/generation_report.json"
R3_RAW_OUTPUT = ROOT / "artifacts/action_evidence_augmentation_v12_20260824_r3_raw"

DEFAULT_SUMMARY = ROOT / "reports/V12_ACTION_EVIDENCE_R3_DRY_POLICY_SUMMARY.json"
DEFAULT_CATALOG = ROOT / "reports/V12_ACTION_EVIDENCE_R3_DRY_POLICY_CATALOG.json.gz"

EXPECTED_DESIGN_SHA256 = "7d7cc1d7f655bc1d1aaa9600c029119331227f32ac506e210b76a855fddce7c1"
EXPECTED_SOURCE_SUMMARY_SHA256 = "548c777a8f1007687c6e9b782c40d0d28b4eafad2747f78c9c97332c70b87b68"
EXPECTED_SOURCE_CATALOG_SHA256 = "7643ba1dbd67d4699d4b8a81999c756cb323540bbd941fdcb058f2614654337d"
EXPECTED_R2_REPORT_SHA256 = "34e7dba80ef37bca7046239a26e2265e0c650d5d7c3a2b9ec014056d3270f1a9"
EXPECTED_R2_BANK_SHA256 = "c72bc6e731194eca69522b0224dde6cf351d4c35ed0f669fd082c5f63f2062b1"
EXPECTED_R2_METADATA_SHA256 = "e507a41413b0ee0473b37e440dfbdbd7bbf4882ad9262d7eb0cc0bacc0c3468c"
EXPECTED_R2_AUDITOR_SHA256 = "f7bed63aedc7f9df8ca68f04343573f86f9f0d5b1a1e8ca76de04ec14e7eb319"

WRITER_START = 192
WRITER_COUNT = 64
WRITERS_PER_PAIR = 20
MIN_ADMITTED_WRITERS = 16


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def load_gzip_json(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_deterministic_gzip_json(path: Path, payload: dict) -> None:
    with path.open("xb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as text:
                json.dump(payload, text, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def risk_tuple(entry: dict) -> tuple[int, int, int, int]:
    risk = entry["risk_stratum"]
    return (
        int(risk["candidate_rank"]),
        int(risk["top1_margin_bin"]),
        int(risk["top5_entropy_bin"]),
        int(risk["stroke_count_bucket"]),
    )


def risk_dict(risk: tuple[int, int, int, int]) -> dict:
    return {
        "candidate_rank": risk[0],
        "top1_margin_bin": risk[1],
        "top5_entropy_bin": risk[2],
        "stroke_count_bucket": risk[3],
    }


def pair_writer_ids(baseline: int, candidate: int) -> list[int]:
    digest = hashlib.sha256(f"v12-r3:{baseline}:{candidate}".encode("ascii")).digest()
    start = int.from_bytes(digest[:4], "big") % WRITER_COUNT
    step = (int.from_bytes(digest[4:8], "big") % (WRITER_COUNT // 2)) * 2 + 1
    assert math.gcd(step, WRITER_COUNT) == 1
    writers = [WRITER_START + ((start + index * step) % WRITER_COUNT) for index in range(WRITERS_PER_PAIR)]
    if len(set(writers)) != WRITERS_PER_PAIR:
        raise AssertionError("pair writer allocation is not unique")
    return writers


def validate_inputs() -> tuple[dict, dict]:
    expected = {
        DESIGN: EXPECTED_DESIGN_SHA256,
        SOURCE_SUMMARY: EXPECTED_SOURCE_SUMMARY_SHA256,
        SOURCE_CATALOG: EXPECTED_SOURCE_CATALOG_SHA256,
        R2_REPORT: EXPECTED_R2_REPORT_SHA256,
    }
    for path, expected_hash in expected.items():
        if not path.is_file() or sha256(path) != expected_hash:
            raise ValueError(f"pinned input mismatch: {path}")

    source_summary = load_json(SOURCE_SUMMARY)
    source_catalog = load_gzip_json(SOURCE_CATALOG)
    report = load_json(R2_REPORT)
    if source_summary.get("status") != "V12_DRY_CATALOG_ONLY_BANK_UNGENERATED":
        raise ValueError("source dry summary status mismatch")
    if len(source_catalog.get("entries", [])) != 68_958:
        raise ValueError("source dry catalog entry count mismatch")
    if report.get("status") != "V12_ACTION_EVIDENCE_RAW_GENERATED_POSTPROCESS_REQUIRED":
        raise ValueError("R2 report status mismatch")
    if report.get("query_rows") != 0 or report.get("final_admission", {}).get("retained_pairs") != 0:
        raise ValueError("R2 rejection boundary mismatch")
    if report.get("final_admission", {}).get("identity_only_pairs") != 9_867:
        raise ValueError("R2 identity-only pair count mismatch")
    if report.get("final_admission", {}).get("candidate_violations") != 1:
        raise ValueError("R2 candidate violation boundary mismatch")
    if report.get("gates", {}).get("candidate_action_within_frozen_top5") is not False:
        raise ValueError("R2 candidate gate mismatch")
    if report.get("hashes", {}).get("bank") != EXPECTED_R2_BANK_SHA256:
        raise ValueError("R2 bank receipt mismatch")
    if report.get("hashes", {}).get("metadata") != EXPECTED_R2_METADATA_SHA256:
        raise ValueError("R2 metadata receipt mismatch")
    return source_summary, source_catalog


def build_policy_catalog() -> tuple[dict, dict]:
    source_summary, source_catalog = validate_inputs()
    grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
    all_risks: set[tuple[int, int, int, int]] = set()
    for entry in source_catalog["entries"]:
        pair = entry["pair"]
        key = (int(pair["baseline_label_index"]), int(pair["candidate_label_index"]))
        grouped[key].append(entry)
        all_risks.add(risk_tuple(entry))

    pair_rows: list[dict] = []
    writer_action_counts: Counter[int] = Counter()
    assigned_unit_count = 0
    pairs_all_units_covered = 0
    orientation_counts: Counter[str] = Counter()
    for baseline, candidate in sorted(grouped):
        entries = sorted(grouped[(baseline, candidate)], key=risk_tuple)
        risks = [risk_tuple(entry) for entry in entries]
        writers = pair_writer_ids(baseline, candidate)
        actions = []
        used_risks: set[tuple[int, int, int, int]] = set()
        for slot, writer in enumerate(writers):
            risk = risks[slot % len(risks)]
            orientation = "candidate_truth_promotion" if slot < 10 else "baseline_truth_veto"
            actions.append(
                {
                    "writer_id": writer,
                    "orientation": orientation,
                    "risk_stratum": risk_dict(risk),
                    "action_acceptance": "direct_frozen_reinference_required",
                }
            )
            used_risks.add(risk)
            writer_action_counts[writer] += 1
            orientation_counts[orientation] += 1
        assigned_unit_count += len(used_risks)
        if len(used_risks) == len(risks):
            pairs_all_units_covered += 1
        pair_rows.append(
            {
                "pair": {
                    "baseline_label_index": baseline,
                    "candidate_label_index": candidate,
                },
                "observed_risk_units": [
                    {
                        "risk_stratum": risk_dict(risk_tuple(entry)),
                        "observed": entry["observed"],
                    }
                    for entry in entries
                ],
                "planned_actions": actions,
                "pair_policy": {
                    "pair_writer_unique_20": True,
                    "pair_writer_max_action_1": True,
                    "orientation_candidate_truth_10": True,
                    "orientation_baseline_truth_10": True,
                    "minimum_final_admitted_writers": MIN_ADMITTED_WRITERS,
                    "below_minimum_policy": "drop_pair_identity_only",
                },
            }
        )

    if len(pair_rows) != 9_867:
        raise AssertionError("directed pair count mismatch")
    if set(writer_action_counts) != set(range(WRITER_START, WRITER_START + WRITER_COUNT)):
        raise AssertionError("proposed writer coverage mismatch")
    if any(len({a["writer_id"] for a in row["planned_actions"]}) != WRITERS_PER_PAIR for row in pair_rows):
        raise AssertionError("pair writer uniqueness failure")
    if orientation_counts != {
        "candidate_truth_promotion": len(pair_rows) * 10,
        "baseline_truth_veto": len(pair_rows) * 10,
    }:
        raise AssertionError("orientation balance failure")

    catalog = {
        "schema": "aiflow-v12-action-evidence-r3-dry-policy/v1",
        "status": "R3_STATIC_POLICY_ONLY_BANK_UNGENERATED",
        "pairs": pair_rows,
        "global_policy": {
            "action_acceptance": [
                "baseline_in_generated_frozen_top5",
                "candidate_in_generated_frozen_top5",
                "catalog_risk_stratum_exact",
                "candidate_calibration_support_at_least_2",
                "nonidentity_spatial_change",
                "truth_topology_preserved",
                "calibration_query_parent_fingerprint_disjoint",
                "finite_unit_box_uniform_time_external_only",
            ],
            "auxiliary_top5_unadmitted": "telemetry_and_runtime_unsupported_identity_not_query_drop",
            "homograph_collision": "context_owned_whole_row_identity_not_action",
            "pre_save_candidate_membership_mismatch": "drop_row_then_recompute_pair_minimum",
            "bank_candidate_violations": "must_equal_zero_or_fail_closed",
            "writer_pair_token_label_features": 0,
        },
    }
    summary = {
        "schema": "aiflow-v12-action-evidence-r3-dry-policy-summary/v1",
        "status": "R3_DESIGN_STATIC_FEASIBILITY_ONLY_GENERATION_FORBIDDEN",
        "inputs": {
            "design_sha256": EXPECTED_DESIGN_SHA256,
            "source_dry_summary_sha256": EXPECTED_SOURCE_SUMMARY_SHA256,
            "source_dry_catalog_sha256": EXPECTED_SOURCE_CATALOG_SHA256,
            "r2_report_sha256": EXPECTED_R2_REPORT_SHA256,
            "r2_bank_receipt_sha256": EXPECTED_R2_BANK_SHA256,
            "r2_metadata_receipt_sha256": EXPECTED_R2_METADATA_SHA256,
            "r2_independent_auditor_sha256": EXPECTED_R2_AUDITOR_SHA256,
            "r2_npz_or_metadata_opened": False,
        },
        "r2_rejection": {
            "query_rows": 0,
            "retained_pairs": 0,
            "identity_only_pairs": 9_867,
            "candidate_violations": 1,
            "independent_classification": "REJECT_SPLIT0_ONLY",
        },
        "source_coverage": {
            "admitted_labels": source_summary["catalog"]["admitted_labels"],
            "directed_pairs": len(pair_rows),
            "risk_units": sum(len(row["observed_risk_units"]) for row in pair_rows),
            "label_free_risk_strata": len(all_risks),
        },
        "proposed_static_allocation": {
            "writer_range": [WRITER_START, WRITER_START + WRITER_COUNT - 1],
            "writers": WRITER_COUNT,
            "writers_per_pair": WRITERS_PER_PAIR,
            "planned_query_actions": len(pair_rows) * WRITERS_PER_PAIR,
            "pair_unique_writer_min": WRITERS_PER_PAIR,
            "pair_unique_writer_max": WRITERS_PER_PAIR,
            "pair_writer_max_actions": 1,
            "candidate_truth_actions": orientation_counts["candidate_truth_promotion"],
            "baseline_truth_actions": orientation_counts["baseline_truth_veto"],
            "writer_planned_actions_min": min(writer_action_counts.values()),
            "writer_planned_actions_max": max(writer_action_counts.values()),
            "risk_units_with_planned_action": assigned_unit_count,
            "pairs_all_risk_units_covered": pairs_all_units_covered,
        },
        "feasibility": {
            "pair_writer20_all_pairs": True,
            "orientation10_10_all_pairs": True,
            "pair_writer1_action_max": True,
            "candidate_membership_direct_reinference_specified": True,
            "auxiliary_unsupported_separated_from_action_acceptance": True,
            "homograph_context_identity_separated_from_query_drop": True,
            "pre_save_candidate_violation_zero_fail_closed": True,
            "actual_parent_availability": "UNPROVEN_REQUIRES_SEPARATE_STATIC_PREGENERATION_AUDIT",
            "actual_query_yield": "UNPROVEN_NO_GENERATION_OR_FORWARD_PERFORMED",
        },
        "boundaries": {
            "r3_raw_output": str(R3_RAW_OUTPUT.resolve()),
            "r3_raw_output_exists": R3_RAW_OUTPUT.exists(),
            "bank_generated": False,
            "physics_run": False,
            "frozen_hwr_forward_run": False,
            "optimizer_backward_run": False,
            "r2_raw_npz_metadata_accessed": False,
            "writers096_127_opened": False,
            "legacy_real_crohme_mathwriting_opened": False,
            "hwr_checkpoint_runtime_changed": False,
            "product_promotion": False,
            "static_independent_audit_required_before_generation": True,
        },
    }
    if summary["boundaries"]["r3_raw_output_exists"]:
        raise FileExistsError("R3 raw output must remain absent during dry policy build")
    return summary, catalog


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-policy", action="store_true")
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    args = parser.parse_args()
    if not args.dry_policy:
        raise SystemExit("only --dry-policy is permitted")
    if args.summary.exists() or args.catalog.exists():
        raise FileExistsError("refusing to overwrite R3 dry policy outputs")
    summary, catalog = build_policy_catalog()
    write_deterministic_gzip_json(args.catalog, catalog)
    summary["outputs"] = {
        "catalog": str(args.catalog.resolve()),
        "catalog_sha256": sha256(args.catalog),
        "source_sha256": sha256(Path(__file__)),
    }
    write_json(args.summary, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
