#!/usr/bin/env python3
"""Prepare the v12 clean-room action-evidence catalog and immutable build plan.

This source never generates trajectory tensors or trains a model.  It reads
only consumed synthetic development writers000..095 and approved external
parent manifests.  Writers096..127 remain unopened.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import io
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
V11_SOURCE = ROOT / "scripts/train_dual_view_writer_adapter_v11.py"
CHECKPOINT = ROOT / "artifacts/commercial_hwr_cleanroom_physics_20260823_r1_shadow/commercial_hwr_cleanroom_physics_checkpoint.pt"
EXTERNAL_ROOT = ROOT / "artifacts/commercial_hwr_cleanroom_dataset_v3_20260823_r1"
EXTERNAL_BANK = EXTERNAL_ROOT / "external_profiled_augmented.npz"
EXTERNAL_METADATA = EXTERNAL_ROOT / "external_profiled_augmented.metadata.jsonl.gz"
V11_R3_AUDIT = ROOT / "reports/V11_DUAL_VIEW_R3_REJECTION_INDEPENDENT_AUDIT.json"
WRITERS096_ADMISSION = ROOT / "reports/V11_WRITER_STYLE_096_127_ADMISSION_RECEIPT.json"
DEFAULT_DRY_SUMMARY = ROOT / "reports/V12_ACTION_EVIDENCE_DRY_CATALOG_SUMMARY_R3.json"
DEFAULT_DRY_CATALOG = ROOT / "reports/V12_ACTION_EVIDENCE_DRY_CATALOG_R3.json.gz"
DESIGN = ROOT / "reports/V12_ACTION_EVIDENCE_AUGMENTATION_DESIGN_20260823.md"
DEFAULT_OUTPUT = ROOT / "artifacts/action_evidence_augmentation_v12_20260823_r1_prepared"

EXPECTED_V11_R3_AUDIT_SHA256 = "e7b17080c7f99df46309bc2284928fd36e5342d6fda09ea2a0835f7935551938"
EXPECTED_WRITERS096_ADMISSION_SHA256 = "d655be9fad848667fde3a6483bd7c6a5597c1e60c5ceeb7883947f7990738ca2"
EXPECTED_STATIC_AUDIT_STATUS = "INDEPENDENT_V12_ACTION_EVIDENCE_PREPARE_STATIC_AUDIT_PASSED"
EXPECTED_V11_SOURCE_SHA256 = "fb43d0ce519546ff00e779c97ad7ed6dbdf3f067f5958bb214da38a923b5640f"
EXPECTED_CHECKPOINT_SHA256 = "c9ebd51e5ba72f1a1e8e2938343823808f7c0ab1bc7143e93332ef8b58143521"
EXPECTED_EXTERNAL_BANK_SHA256 = "31eb534d125f38cfbe5aa33d38545489d71737461f9f605da5304ebab8d3d577"
EXPECTED_EXTERNAL_METADATA_SHA256 = "411097fe5846ec3da6c7c6c82c47dc3da66cf2cb0e8cc1ee691edca18ec5c87b"
WRITER_IDS = tuple(range(128, 192))
WRITER_SEED_ROOT = 20260824
WRITER_SEED_MULTIPLIER = 4099
MIN_GLOBAL_ROWS = 4
MIN_GLOBAL_WRITERS = 4
MIN_ZERO_REG_ACTIONS = 16
TARGET_ACTIONS_PER_PAIR = 20
RAW_ATTEMPTS_PER_ACTION = 3
CALIBRATION_ROWS_PER_WRITER_CANDIDATE = 2
MARGIN_EDGES = (0.5, 1.5, 3.0)
ENTROPY_EDGES = (0.35, 0.60, 0.80, 1.0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_v11():
    spec = importlib.util.spec_from_file_location("v11_consumed", V11_SOURCE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _stroke_counts(v11, trajectories: np.ndarray) -> np.ndarray:
    return np.asarray([len(v11.trajectory_signature(row)) for row in trajectories], np.int8)


def _bin(value: float, edges: tuple[float, ...]) -> int:
    return int(np.digitize(value, np.asarray(edges, np.float64)))


def _pair_writer_ids(pair: tuple[int, int]) -> list[int]:
    ranked = sorted(
        WRITER_IDS,
        key=lambda writer: hashlib.sha256(
            f"{WRITER_SEED_ROOT}:pair-writer:{pair[0]}:{pair[1]}:{writer}".encode()
        ).hexdigest(),
    )
    return ranked[:TARGET_ACTIONS_PER_PAIR]


def build_catalog() -> tuple[dict, list[dict]]:
    if sha256(V11_R3_AUDIT) != EXPECTED_V11_R3_AUDIT_SHA256:
        raise ValueError("v11 r3 rejection audit drift")
    if sha256(WRITERS096_ADMISSION) != EXPECTED_WRITERS096_ADMISSION_SHA256:
        raise ValueError("writers096..127 admission drift")
    expected_live = ((V11_SOURCE, EXPECTED_V11_SOURCE_SHA256), (CHECKPOINT, EXPECTED_CHECKPOINT_SHA256),
                     (EXTERNAL_BANK, EXPECTED_EXTERNAL_BANK_SHA256), (EXTERNAL_METADATA, EXPECTED_EXTERNAL_METADATA_SHA256))
    for path, expected in expected_live:
        if sha256(path) != expected: raise ValueError(f"pinned input drift: {path.name}")
    v11 = _load_v11(); data, inventory = v11.load_development96()
    labels = data["labels"].astype(np.int64); logits = data["logits"].astype(np.float32)
    writers = data["writers"].astype(np.int16); trajectories = data["trajectories"].astype(np.float32)
    checkpoint = v11.torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    label_names = [str(value) for value in checkpoint["math_labels"]]
    homographs = v11._resolve_homographs(label_names)
    classes = logits.shape[1]
    counts = np.bincount(labels, minlength=classes)
    writer_counts = np.asarray([len(np.unique(writers[labels == label])) for label in range(classes)], np.int16)
    admitted = (counts >= MIN_GLOBAL_ROWS) & (writer_counts >= MIN_GLOBAL_WRITERS)
    stroke_counts = _stroke_counts(v11, trajectories)
    topology_by_label: dict[int, set[int]] = defaultdict(set)
    for label, strokes in zip(labels.tolist(), stroke_counts.tolist(), strict=True):
        topology_by_label[int(label)].add(int(strokes))

    pair_counts: dict[tuple[int, int], list[int]] = defaultdict(lambda: [0, 0, 0])
    unit_counts: dict[tuple[int, int, int, int, int, int], list[int]] = defaultdict(lambda: [0, 0, 0])
    counters = defaultdict(int)
    for index in range(len(labels)):
        order = np.argsort(logits[index])[-5:][::-1].astype(int)
        if not bool(np.all(admitted[order])):
            counters["unsupported_rows"] += 1; continue
        if v11._collision(order, homographs):
            counters["homograph_rows"] += 1; continue
        counters["eligible_rows"] += 1
        values = logits[index, order].astype(np.float64)
        shifted = values - values.max(); probability = np.exp(shifted); probability /= probability.sum()
        entropy = float(-np.sum(probability * np.log(np.maximum(probability, 1e-12))))
        margin_bin = _bin(float(values[0] - values[1]), MARGIN_EDGES)
        entropy_bin = _bin(entropy, ENTROPY_EDGES)
        stroke_bucket = min(int(stroke_counts[index]), 4)
        baseline = int(order[0]); truth = int(labels[index])
        for rank, candidate in enumerate(order[1:], 1):
            candidate = int(candidate)
            if int(stroke_counts[index]) not in topology_by_label[candidate]:
                counters["topology_filtered_actions"] += 1; continue
            counters["eligible_actions"] += 1
            outcome = (int(truth == candidate), int(truth == baseline))
            pair = (baseline, candidate); pair_counts[pair][0] += 1
            pair_counts[pair][1] += outcome[0]; pair_counts[pair][2] += outcome[1]
            unit = (baseline, candidate, rank, margin_bin, entropy_bin, stroke_bucket)
            unit_counts[unit][0] += 1; unit_counts[unit][1] += outcome[0]; unit_counts[unit][2] += outcome[1]

    assignments: dict[tuple[int, int, int, int, int, int], list[int]] = defaultdict(list); units_by_pair = defaultdict(list)
    for unit in unit_counts:
        units_by_pair[(unit[0], unit[1])].append(unit)
    for pair, units in sorted(units_by_pair.items()):
        ordered = sorted(units, key=lambda unit: hashlib.sha256(f"{WRITER_SEED_ROOT}:stratum:{pair}:{unit[2:]}".encode()).hexdigest())
        for slot, writer in enumerate(_pair_writer_ids(pair)):
            assignments[ordered[slot % len(ordered)]].append(writer)

    entries = []
    writer_candidate_anchors = set()
    for unit, observed in sorted(unit_counts.items()):
        baseline, candidate, rank, margin_bin, entropy_bin, stroke_bucket = unit
        stratum = (rank, margin_bin, entropy_bin, stroke_bucket)
        assigned = assignments[unit]
        target_actions = len(assigned)
        writer_candidate_anchors.update((writer, candidate) for writer in assigned)
        entries.append({
            "pair": {"baseline_label_index": baseline, "candidate_label_index": candidate},
            "risk_stratum": {"candidate_rank": rank, "top1_margin_bin": margin_bin,
                              "top5_entropy_bin": entropy_bin, "stroke_count_bucket": stroke_bucket},
            "observed": {"rows": observed[0], "improvement_opportunities": observed[1],
                         "regression_opportunities": observed[2]},
            "assigned_writer_ids": assigned,
            "target_admitted_query_actions": target_actions,
            "raw_attempts_per_action": RAW_ATTEMPTS_PER_ACTION,
        })

    planned_query = sum(len(value) for value in assignments.values())
    planned_calibration = len(writer_candidate_anchors) * CALIBRATION_ROWS_PER_WRITER_CANDIDATE
    summary = {
        "schema": "aiflow-v12-action-evidence-dry-catalog/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "V12_DRY_CATALOG_ONLY_BANK_UNGENERATED",
        "source": {"consumed_writer_ids": "000..095", "rows": int(len(labels)),
                   "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256, "v11_source_sha256": EXPECTED_V11_SOURCE_SHA256,
                   "v11_r3_audit_sha256": EXPECTED_V11_R3_AUDIT_SHA256},
        "catalog": {"admitted_labels": int(admitted.sum()), "eligible_rows": counters["eligible_rows"],
                    "unsupported_rows": counters["unsupported_rows"], "homograph_rows": counters["homograph_rows"],
                    "topology_filtered_actions": counters["topology_filtered_actions"],
                    "eligible_actions": counters["eligible_actions"], "directed_pairs": len(pair_counts),
                    "pair_risk_units": len(entries),
                    "label_free_risk_strata": len({(row["risk_stratum"]["candidate_rank"], row["risk_stratum"]["top1_margin_bin"], row["risk_stratum"]["top5_entropy_bin"], row["risk_stratum"]["stroke_count_bucket"]) for row in entries}),
                    "pairs_with_improvement": sum(value[1] > 0 for value in pair_counts.values()),
                    "pairs_with_regression": sum(value[2] > 0 for value in pair_counts.values()),
                    "pairs_with_both": sum(value[1] > 0 and value[2] > 0 for value in pair_counts.values())},
        "build_plan": {"writer_ids": list(WRITER_IDS), "writer_seed_root": WRITER_SEED_ROOT,
                       "writer_seed_formula": "seed_root + global_writer_id * 4099",
                       "minimum_zero_reg_actions": MIN_ZERO_REG_ACTIONS,
                       "target_actions_per_directed_pair": TARGET_ACTIONS_PER_PAIR,
                       "pair_actions_distributed_round_robin_over_observed_risk_strata": True,
                       "raw_attempts_per_action": RAW_ATTEMPTS_PER_ACTION,
                       "planned_query_rows": planned_query, "shared_calibration_anchor_rows": planned_calibration,
                       "expected_final_rows_before_post_admission_drop": planned_query + planned_calibration,
                       "planned_raw_attempt_rows": planned_query * RAW_ATTEMPTS_PER_ACTION + planned_calibration},
        "boundaries": {"catalog_pair_ids_are_provenance_and_grouping_only": True,
                       "candidate_token_writer_id_features": 0, "writers096_127_opened": False,
                       "project_rows": 0, "crohme_rows": 0, "mathwriting_rows": 0, "legacy_rows": 0,
                       "bank_generated": False, "adapter_trained": False, "hwr_changed": False,
                       "product_promotion": False},
        "external_parent_inputs": {"bank": str(EXTERNAL_BANK.resolve()), "bank_sha256": EXPECTED_EXTERNAL_BANK_SHA256,
                                   "metadata": str(EXTERNAL_METADATA.resolve()), "metadata_sha256": EXPECTED_EXTERNAL_METADATA_SHA256},
        "consumed_inventory": inventory,
    }
    return summary, entries


def write_dry_catalog(summary_path: Path, catalog_path: Path) -> None:
    if summary_path.exists() or catalog_path.exists():
        raise FileExistsError("refusing to overwrite dry catalog")
    summary, entries = build_catalog()
    catalog_payload = {"summary_sha256_pending": True, "entries": entries}
    with catalog_path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as stream:
                json.dump(catalog_payload, stream, separators=(",", ":"))
    summary["catalog_file"] = str(catalog_path.resolve())
    summary["catalog_sha256"] = sha256(catalog_path)
    summary["prepare_source_sha256"] = sha256(Path(__file__))
    summary["design_sha256"] = sha256(DESIGN)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def validate_static_audit(path: Path, summary_path: Path, catalog_path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    gates = payload.get("gates", {})
    if payload.get("status") != EXPECTED_STATIC_AUDIT_STATUS or not gates or not all(value is True for value in gates.values()):
        raise ValueError("v12 independent static audit is not all-gates PASS")
    if payload.get("script_sha256") != sha256(Path(__file__)):
        raise ValueError("v12 static audit source mismatch")
    expected = {"dry_summary_sha256": sha256(summary_path), "dry_catalog_sha256": sha256(catalog_path),
                "design_sha256": sha256(DESIGN)}
    for key, value in expected.items():
        if payload.get(key) != value: raise ValueError(f"v12 static audit {key} mismatch")
    decision = payload.get("decision", {})
    if decision.get("prepare_allowed") is not True or decision.get("bank_generation_allowed") is not False:
        raise ValueError("v12 static audit decision boundary mismatch")
    return {"path": str(path.resolve()), "sha256": sha256(path), "status": payload["status"]}


def prepare(output: Path, static_audit: Path, summary_path: Path, catalog_path: Path) -> None:
    if output.exists(): raise FileExistsError("refusing to overwrite prepared output")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "V12_DRY_CATALOG_ONLY_BANK_UNGENERATED": raise ValueError("dry summary status mismatch")
    if summary.get("prepare_source_sha256") != sha256(Path(__file__)) or summary.get("catalog_sha256") != sha256(catalog_path) or summary.get("design_sha256") != sha256(DESIGN):
        raise ValueError("dry catalog/source binding mismatch")
    boundaries = summary.get("boundaries", {}); plan = summary.get("build_plan", {})
    if (boundaries.get("bank_generated") is not False or boundaries.get("adapter_trained") is not False or
            boundaries.get("writers096_127_opened") is not False or boundaries.get("candidate_token_writer_id_features") != 0):
        raise ValueError("dry summary boundary mismatch")
    if plan.get("writer_ids") != list(WRITER_IDS) or plan.get("target_actions_per_directed_pair") != TARGET_ACTIONS_PER_PAIR or plan.get("minimum_zero_reg_actions") != MIN_ZERO_REG_ACTIONS:
        raise ValueError("dry summary writer/action contract mismatch")
    for path, expected in ((V11_SOURCE, EXPECTED_V11_SOURCE_SHA256), (CHECKPOINT, EXPECTED_CHECKPOINT_SHA256),
                           (EXTERNAL_BANK, EXPECTED_EXTERNAL_BANK_SHA256), (EXTERNAL_METADATA, EXPECTED_EXTERNAL_METADATA_SHA256)):
        if sha256(path) != expected: raise ValueError(f"pinned live input drift: {path.name}")
    audit = validate_static_audit(static_audit, summary_path, catalog_path)
    output.mkdir(parents=True)
    snapshot = output / "prepare_action_evidence_augmentation_v12.source.py"
    snapshot.write_bytes(Path(__file__).read_bytes())
    manifest = {
        "status": "V12_ACTION_EVIDENCE_PREPARED_BANK_UNGENERATED",
        "source_sha256": sha256(snapshot), "static_audit": audit,
        "dry_summary_sha256": sha256(summary_path), "dry_catalog_sha256": sha256(catalog_path),
        "design_sha256": sha256(DESIGN), "v11_source_sha256": EXPECTED_V11_SOURCE_SHA256,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256, "external_bank_sha256": EXPECTED_EXTERNAL_BANK_SHA256,
        "external_metadata_sha256": EXPECTED_EXTERNAL_METADATA_SHA256,
        "writers096_127_opened": False, "bank_generated": False, "adapter_trained": False,
        "next_action": "independent prepared receipt audit before any bank generation",
    }
    (output / "PREPARED_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-catalog", action="store_true")
    mode.add_argument("--prepare", action="store_true")
    parser.add_argument("--summary", type=Path, default=DEFAULT_DRY_SUMMARY)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_DRY_CATALOG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--independent-audit", type=Path)
    args = parser.parse_args()
    if args.dry_catalog:
        write_dry_catalog(args.summary, args.catalog); return 0
    if args.independent_audit is None:
        parser.error("--prepare requires --independent-audit")
    prepare(args.output, args.independent_audit, args.summary, args.catalog); return 0


if __name__ == "__main__":
    raise SystemExit(main())
