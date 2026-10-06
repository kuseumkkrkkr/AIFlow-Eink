#!/usr/bin/env python3
"""V12 R3 generation-only builder.

R2 is an immutable split0-only rejection.  This source separates direct
baseline/candidate action acceptance from auxiliary unsupported and homograph
fallbacks, then performs a second frozen-HWR admission pass before saving.
The source may be dry-tested, but generation requires an independent
pre-generation audit receipt.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import functools
import hashlib
import importlib
import json
import os
from pathlib import Path
import tempfile
import traceback
from typing import Callable


_ROOT = Path(__file__).resolve().parents[1]
_BOOTSTRAP_SOURCES = {
    "generate_action_evidence_augmentation_v12_r2.py": (
        _ROOT / "scripts/generate_action_evidence_augmentation_v12_r2.py",
        "df34f10e149b3f5c5fa6a797903f1daa87f9acfd346179856048cb2bc10e60a1",
    ),
    "generate_action_evidence_augmentation_v12.py": (
        _ROOT / "scripts/generate_action_evidence_augmentation_v12.py",
        "e82064485a8ed10abcbb978746319a3a7063e78477133ce75f8bb6fe9d503733",
    ),
    "build_cleanroom_writer_style_bank_v5.py": (
        _ROOT / "scripts/build_cleanroom_writer_style_bank_v5.py",
        "a28f6f399f6075d31870d6b7906d13ddc343a4d3396c47266a490425f8b3ebbf",
    ),
    "cleanroom_writer_style_simulator_v4.py": (
        _ROOT / "scripts/cleanroom_writer_style_simulator_v4.py",
        "49de0704301c93f25d20b3c8c847aa5ca8ba70c561a3487c5e484c23aab89c9f",
    ),
    "cleanroom_online_ink_augmentation_v2.py": (
        _ROOT / "scripts/cleanroom_online_ink_augmentation_v2.py",
        "283b8b26b0843a9d096ca8fe8e615dfad1c169204d372910db9b16808c77e0f2",
    ),
    "cleanroom_pen_physics_v3.py": (
        _ROOT / "scripts/cleanroom_pen_physics_v3.py",
        "b1afcbac828c20b9f860097260e1e2e3c20a0201b057f7b27c65fb2b491b2ffd",
    ),
    "cleanroom_trajectory_profiles_v3.py": (
        _ROOT / "scripts/cleanroom_trajectory_profiles_v3.py",
        "a15ffd8d61cee69091a934e581a03b0b9e713b87dccc8cae55ecab2b21d65f42",
    ),
    "train_character_classifier_v1.py": (
        _ROOT / "scripts/train_character_classifier_v1.py",
        "c97bba856ad966fd7de84c688ed2591ed5568b5957d696f35e7a070a3c031835",
    ),
    "character_tensor_v1.py": (
        _ROOT / "scripts/character_tensor_v1.py",
        "9bd62259bef4b872e1a57052575c7f3e316541ac23688e753c116886c022f99f",
    ),
    "training_data_guard_v1.py": (
        _ROOT / "scripts/training_data_guard_v1.py",
        "42c047453bce454733b94c3fd085f0ae150259aa59e74ec05f6b4e32f28e274b",
    ),
    "replay_evaluate_hwr_v1.py": (
        _ROOT / "scripts/replay_evaluate_hwr_v1.py",
        "a4f48526f428f967a7ac4266f8b61952499f1114e0106fdeef9b0ca15f5e65ee",
    ),
    "build_normalized_ink_v1.py": (
        _ROOT / "scripts/build_normalized_ink_v1.py",
        "9e95fa77aa9fda0b2c37a7e9d4a8c5cf9a94068eaad69ab41e77949eb181a291",
    ),
}


def _bootstrap_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


for _name, (_path, _expected) in _BOOTSTRAP_SOURCES.items():
    if _bootstrap_sha256(_path) != _expected:
        raise RuntimeError(f"pre-import generator/helper drift: {_name}")

r2 = importlib.import_module("generate_action_evidence_augmentation_v12_r2")
r1 = r2.r1
np = r2.np


ROOT = _ROOT
R2_OUTPUT = ROOT / "artifacts/action_evidence_augmentation_v12_20260823_r2_raw"
R2_REPORT = R2_OUTPUT / "generation_report.json"
R3_DESIGN = ROOT / "reports/V12_ACTION_EVIDENCE_R3_DESIGN_20260824.md"
R3_DRY_SUMMARY = ROOT / "reports/V12_ACTION_EVIDENCE_R3_DRY_POLICY_SUMMARY.json"
R3_DRY_CATALOG = ROOT / "reports/V12_ACTION_EVIDENCE_R3_DRY_POLICY_CATALOG.json.gz"
R3_STATIC_AUDIT = ROOT / "reports/V12_ACTION_EVIDENCE_R3_INDEPENDENT_STATIC_AUDIT.json"

DEFAULT_RAW_OUTPUT = ROOT / "artifacts/action_evidence_augmentation_v12_20260824_r3_raw"
DEFAULT_POSTPROCESSED_OUTPUT = ROOT / "artifacts/action_evidence_augmentation_v12_20260824_r3_final"

EXPECTED_R2_FILES = [
    "GENERATION_STARTED.json",
    "action_evidence_raw_bank.metadata.jsonl.gz",
    "action_evidence_raw_bank.npz",
    "generate_action_evidence_augmentation_v12.source.py",
    "generation_report.json",
]
EXPECTED_R2_REPORT_SHA256 = "34e7dba80ef37bca7046239a26e2265e0c650d5d7c3a2b9ec014056d3270f1a9"
EXPECTED_R2_BANK_RECEIPT_SHA256 = "c72bc6e731194eca69522b0224dde6cf351d4c35ed0f669fd082c5f63f2062b1"
EXPECTED_R2_METADATA_RECEIPT_SHA256 = "e507a41413b0ee0473b37e440dfbdbd7bbf4882ad9262d7eb0cc0bacc0c3468c"
EXPECTED_R3_DESIGN_SHA256 = "7d7cc1d7f655bc1d1aaa9600c029119331227f32ac506e210b76a855fddce7c1"
EXPECTED_R3_DRY_SUMMARY_SHA256 = "c646c2571e300dbe0262b774f21fa7703756476857f2ecf0837f5cd131f55863"
EXPECTED_R3_DRY_CATALOG_SHA256 = "66390f17ca243b5f779a7065d483b955e6c5f3038e5cc8e593e3e7e559d2b360"
EXPECTED_R3_STATIC_AUDIT_SHA256 = "e9f04366b2e7a0136c5f16ffd482f1dde506952268af107d84bc55b853d85121"
EXPECTED_R3_STATIC_AUDIT_STATUS = "INDEPENDENT_V12_R3_STATIC_AUDIT_PASSED"
EXPECTED_R3_PREGEN_AUDIT_STATUS = "INDEPENDENT_V12_R3_GENERATOR_PREGENERATION_AUDIT_PASSED"

WRITER_IDS = tuple(range(192, 256))
MIN_PAIR_WRITERS = 16
EXPECTED_PAIR_COUNT = 9_867
EXPECTED_ACTION_COUNT = 197_340
SCHEMA = "aiflow-v12-action-evidence-r3-raw/v1"


def sha256(path: Path) -> str:
    return _bootstrap_sha256(path)


def _validate_r2_rejection() -> dict:
    if sorted(path.name for path in R2_OUTPUT.iterdir() if path.is_file()) != EXPECTED_R2_FILES:
        raise ValueError("R2 immutable rejected output is not exact5")
    if sha256(R2_REPORT) != EXPECTED_R2_REPORT_SHA256:
        raise ValueError("R2 report drift")
    report = json.loads(R2_REPORT.read_text(encoding="utf-8"))
    if report.get("query_rows") != 0 or report.get("final_admission", {}).get("retained_pairs") != 0:
        raise ValueError("R2 split0-only rejection boundary mismatch")
    if report.get("hashes", {}).get("bank") != EXPECTED_R2_BANK_RECEIPT_SHA256:
        raise ValueError("R2 bank receipt drift")
    if report.get("hashes", {}).get("metadata") != EXPECTED_R2_METADATA_RECEIPT_SHA256:
        raise ValueError("R2 metadata receipt drift")
    return {
        "report_sha256": EXPECTED_R2_REPORT_SHA256,
        "bank_receipt_sha256": EXPECTED_R2_BANK_RECEIPT_SHA256,
        "metadata_receipt_sha256": EXPECTED_R2_METADATA_RECEIPT_SHA256,
        "reuse_allowed": False,
    }


def _validate_r3_static_lineage() -> dict:
    expected = {
        R3_DESIGN: EXPECTED_R3_DESIGN_SHA256,
        R3_DRY_SUMMARY: EXPECTED_R3_DRY_SUMMARY_SHA256,
        R3_DRY_CATALOG: EXPECTED_R3_DRY_CATALOG_SHA256,
        R3_STATIC_AUDIT: EXPECTED_R3_STATIC_AUDIT_SHA256,
    }
    for path, wanted in expected.items():
        if sha256(path) != wanted:
            raise ValueError(f"R3 static lineage drift: {path.name}")
    audit = json.loads(R3_STATIC_AUDIT.read_text(encoding="utf-8"))
    gates = audit.get("gates", {})
    if audit.get("status") != EXPECTED_R3_STATIC_AUDIT_STATUS or not gates or not all(gates.values()):
        raise ValueError("R3 static audit is not all-gates PASS")
    if audit.get("decision", {}).get("generation_source_run_allowed") is not False:
        raise ValueError("R3 static audit generation boundary mismatch")
    return {
        "design_sha256": EXPECTED_R3_DESIGN_SHA256,
        "dry_summary_sha256": EXPECTED_R3_DRY_SUMMARY_SHA256,
        "dry_catalog_sha256": EXPECTED_R3_DRY_CATALOG_SHA256,
        "static_audit_sha256": EXPECTED_R3_STATIC_AUDIT_SHA256,
    }


def _validate_pregeneration_audit(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    gates = payload.get("gates", {})
    if payload.get("status") != EXPECTED_R3_PREGEN_AUDIT_STATUS or not gates or not all(gates.values()):
        raise ValueError("R3 pre-generation audit is not all-gates PASS")
    if payload.get("generator_source_sha256") != sha256(Path(__file__)):
        raise ValueError("R3 pre-generation audit source mismatch")
    if payload.get("r3_static_audit_sha256") != EXPECTED_R3_STATIC_AUDIT_SHA256:
        raise ValueError("R3 pre-generation audit lineage mismatch")
    decision = payload.get("decision", {})
    if decision.get("generation_allowed") is not True or decision.get("training_allowed") is not False:
        raise ValueError("R3 pre-generation decision mismatch")
    return {"path": str(path.resolve()), "sha256": sha256(path), "status": payload["status"]}


@functools.lru_cache(maxsize=1)
def _latents_r3():
    return tuple(r1._sample_latents(256))


def _catalog_entries_r3() -> list[dict]:
    payload = r1._load_json_gzip(R3_DRY_CATALOG)
    pairs = payload.get("pairs", [])
    if len(pairs) != EXPECTED_PAIR_COUNT:
        raise ValueError("R3 policy pair count drift")
    entries = []
    action_keys = set()
    for pair_row in pairs:
        pair = {key: int(value) for key, value in pair_row["pair"].items()}
        writers = []
        orientations = Counter()
        for action in pair_row["planned_actions"]:
            writer = int(action["writer_id"])
            orientation = str(action["orientation"])
            risk = {key: int(value) for key, value in action["risk_stratum"].items()}
            key = (pair["baseline_label_index"], pair["candidate_label_index"], writer)
            if key in action_keys:
                raise ValueError("R3 pair-writer action duplicated")
            action_keys.add(key)
            writers.append(writer)
            orientations[orientation] += 1
            entries.append({"pair": pair, "risk_stratum": risk, "writer_id": writer, "orientation": orientation})
        if len(writers) != 20 or len(set(writers)) != 20 or not set(writers).issubset(WRITER_IDS):
            raise ValueError("R3 pair writer allocation drift")
        if orientations != {"candidate_truth_promotion": 10, "baseline_truth_veto": 10}:
            raise ValueError("R3 pair orientation drift")
    if len(entries) != EXPECTED_ACTION_COUNT or len(action_keys) != EXPECTED_ACTION_COUNT:
        raise ValueError("R3 action count drift")
    return entries


def _calibration_specs(entries: list[dict]) -> set[tuple[int, int]]:
    return {
        (int(entry["writer_id"]), int(entry["pair"]["candidate_label_index"]))
        for entry in entries
    }


def _action_decision(
    actual: dict, expected: dict, baseline: int, candidate: int,
    homographs: tuple[frozenset[int], ...],
) -> dict:
    """Classify one direct action without treating auxiliary labels as blockers."""

    top5 = [int(value) for value in actual["top5"]]
    auxiliary_unadmitted = sorted({
        value for value in top5
        if value not in (baseline, candidate) and value not in r1.ADMITTED_LABELS
    })
    risk_exact = all(
        int(actual[key]) == int(expected[key])
        for key in ("candidate_rank", "top1_margin_bin", "top5_entropy_bin", "stroke_count_bucket")
    )
    direct_membership = baseline in top5 and candidate in top5
    homograph = r1._homograph_collision(top5, homographs)
    if homograph:
        disposition = "context_owned_identity"
    elif not direct_membership:
        disposition = "direct_membership_reject"
    elif not risk_exact:
        disposition = "risk_mismatch_reject"
    else:
        disposition = "action"
    return {
        "disposition": disposition,
        "direct_membership": direct_membership,
        "risk_exact": risk_exact,
        "homograph_collision": homograph,
        "auxiliary_unadmitted_labels": auxiliary_unadmitted,
        "auxiliary_unsupported_present": bool(auxiliary_unadmitted),
    }


def _build_calibration_r3(
    specs: set[tuple[int, int]], features: np.ndarray, labels: np.ndarray, metadata: list[dict],
    pool: dict[tuple[int, int], list[int]], model, homographs: tuple[frozenset[int], ...],
    seen_hashes: set[str],
) -> tuple[list[np.ndarray], list[dict], dict[tuple[int, int], set[str]], list[dict], dict]:
    del labels
    rows: list[np.ndarray] = []
    rows_meta: list[dict] = []
    fingerprints_by_spec: dict[tuple[int, int], set[str]] = {}
    unavailable: list[dict] = []
    counters: Counter = Counter()
    for writer, label in sorted(specs):
        candidates = [
            (strokes, index)
            for (pool_label, strokes), indices in sorted(pool.items()) if pool_label == label
            for index in indices
        ]
        candidates.sort(key=lambda item: hashlib.sha256(
            f"v12-r3:cal:{writer}:{label}:{metadata[item[1]]['synthetic_id']}".encode()
        ).hexdigest())
        accepted: list[tuple[np.ndarray, dict]] = []
        used_fingerprints: set[str] = set()
        local_hashes: set[str] = set()
        for strokes, index in candidates:
            fingerprints = r1._parent_fingerprints(metadata[index])
            if fingerprints & used_fingerprints:
                continue
            source = features[index]
            seed_offset = r1._seed_offset(
                f"r3:cal:{writer}:{label}:{metadata[index]['synthetic_id']}:{len(accepted)}"
            )
            generated, audits = r1._style_and_physics([source], writer, seed_offset)
            value = generated[0]
            logits = r1._predict(model, value[None])[0]
            top5 = np.argsort(logits)[-5:][::-1].astype(int).tolist()
            valid, rms = r1._valid_tensor(value, source)
            digest = r1.tensor_hash(value)
            if r1._homograph_collision(top5, homographs):
                counters["calibration_context_owned_identity"] += 1
                continue
            if not valid or label not in top5 or digest in seen_hashes or digest in local_hashes:
                counters["calibration_direct_reject"] += 1
                continue
            auxiliary = sorted(value for value in top5 if value != label and value not in r1.ADMITTED_LABELS)
            local_hashes.add(digest)
            used_fingerprints |= fingerprints
            accepted.append((value, {
                "schema": SCHEMA, "episode_split": "calibration",
                "synthetic_writer_id": f"synthetic_writer_{writer:03d}", "global_writer_id": writer,
                "label_index": label, "candidate_label_index": label,
                "support_role": "candidate_calibration_anchor",
                "parent_synthetic_id": metadata[index]["synthetic_id"],
                "parent_fingerprints": sorted(fingerprints), "source_index": index,
                "source_topology": metadata[index]["topology"], "attempt": len(accepted),
                "physics_seed": r1.WRITER_SEED_ROOT + writer * r1.WRITER_SEED_MULTIPLIER + seed_offset,
                "physics_batch_position": 0, "writer_latent": asdict(_latents_r3()[writer]),
                "style_audit": audits[0], "spatial_rms_from_truth_parent": rms,
                "tensor_sha256": digest, "frozen_top5": top5,
                "auxiliary_unadmitted_labels": auxiliary,
                "auxiliary_unsupported_present": bool(auxiliary),
                "external_approved_parent": True, "project_rows": 0,
            }))
            if len(accepted) == r1.CALIBRATION_SUPPORT:
                break
        if len(accepted) != r1.CALIBRATION_SUPPORT:
            unavailable.append({"global_writer_id": writer, "candidate_label_index": label,
                                "admitted_support": len(accepted), "policy": "identity_only"})
            continue
        fingerprints_by_spec[(writer, label)] = used_fingerprints
        seen_hashes.update(local_hashes)
        for value, row in accepted:
            rows.append(value)
            rows_meta.append(row)
    counters["admitted_rows"] = len(rows)
    counters["unavailable_specs"] = len(unavailable)
    return rows, rows_meta, fingerprints_by_spec, unavailable, dict(counters)


def _build_queries_r3(
    entries: list[dict], features: np.ndarray, metadata: list[dict],
    pool: dict[tuple[int, int], list[int]], model, homographs: tuple[frozenset[int], ...],
    calibration_fingerprints: dict[tuple[int, int], set[str]], seen_hashes: set[str],
) -> tuple[list[np.ndarray], list[dict], dict]:
    rows: list[np.ndarray] = []
    rows_meta: list[dict] = []
    counters: Counter = Counter()
    pair_admitted: dict[tuple[int, int], set[int]] = defaultdict(set)
    all_calibration_fingerprints: dict[int, set[str]] = defaultdict(set)
    for (writer, _candidate), fingerprints in calibration_fingerprints.items():
        all_calibration_fingerprints[writer].update(fingerprints)

    for entry in entries:
        baseline = int(entry["pair"]["baseline_label_index"])
        candidate = int(entry["pair"]["candidate_label_index"])
        writer = int(entry["writer_id"])
        orientation = str(entry["orientation"])
        expected = {key: int(value) for key, value in entry["risk_stratum"].items()}
        pair = (baseline, candidate)
        if (writer, candidate) not in calibration_fingerprints:
            counters["missing_candidate_calibration_support"] += 1
            continue
        truth = candidate if orientation == "candidate_truth_promotion" else baseline
        relative = baseline if truth == candidate else candidate
        target_strokes = expected["stroke_count_bucket"]
        available_strokes = sorted({
            strokes for label, strokes in pool
            if label == truth and (relative, strokes) in pool and min(int(strokes), 4) == target_strokes
        })
        if not available_strokes:
            counters["missing_shared_topology"] += 1
            continue
        strokes = available_strokes[0]
        forbidden = set(all_calibration_fingerprints[writer])
        truth_index = r1._select_parent(
            pool, metadata, truth, strokes, f"r3-query-truth:{pair}:{writer}", forbidden, set()
        )
        if truth_index is None:
            counters["missing_truth_parent"] += 1
            continue
        truth_fp = r1._parent_fingerprints(metadata[truth_index])
        forbidden |= truth_fp
        relative_index = r1._select_parent(
            pool, metadata, relative, strokes, f"r3-query-relative:{pair}:{writer}", forbidden, set()
        )
        if relative_index is None:
            counters["missing_relative_parent"] += 1
            continue
        relative_fp = r1._parent_fingerprints(metadata[relative_index])
        source = features[truth_index]
        relative_source = features[relative_index]
        attempt_rows, attempt_specs = r2._morph_attempts(source, relative_source, counters)
        if not attempt_rows:
            counters["all_morph_attempts_topology_rejected"] += 1
            counters["all_attempts_drop"] += 1
            continue
        action_key = f"r3-query:{baseline}:{candidate}:{writer}:{r1.canonical_hash(expected)}"
        seed_offset = r1._seed_offset(action_key)
        generated, style_audits = r1._style_and_physics(attempt_rows, writer, seed_offset)
        logits = r1._predict(model, generated)
        accepted = None
        for physics_batch_position, (value, row_logits, attempt_spec, style_audit) in enumerate(
            zip(generated, logits, attempt_specs, style_audits, strict=True)
        ):
            actual = r1._risk(row_logits, strokes, candidate)
            decision = _action_decision(actual, expected, baseline, candidate, homographs)
            counters[f"disposition_{decision['disposition']}"] += 1
            if decision["auxiliary_unsupported_present"]:
                counters["auxiliary_unsupported_telemetry"] += 1
            if decision["disposition"] != "action":
                continue
            valid, rms = r1._valid_tensor(value, source)
            digest = r1.tensor_hash(value)
            if not valid or digest in seen_hashes:
                counters["invalid_or_duplicate"] += 1
                continue
            accepted = (value, {
                "schema": SCHEMA, "episode_split": "query",
                "synthetic_writer_id": f"synthetic_writer_{writer:03d}", "global_writer_id": writer,
                "directed_pair": {"baseline_label_index": baseline, "candidate_label_index": candidate},
                "risk_stratum": expected, "orientation": orientation, "truth_label_index": truth,
                "relative_label_index": relative,
                "truth_parent_synthetic_id": metadata[truth_index]["synthetic_id"],
                "relative_parent_synthetic_id": metadata[relative_index]["synthetic_id"],
                "truth_parent_fingerprints": sorted(truth_fp),
                "relative_parent_fingerprints": sorted(relative_fp),
                "calibration_parent_fingerprints": sorted(calibration_fingerprints[(writer, candidate)]),
                "all_writer_calibration_parent_fingerprints_sha256": r1.canonical_hash(
                    sorted(all_calibration_fingerprints[writer])
                ),
                "truth_source_index": truth_index, "relative_source_index": relative_index,
                "truth_topology": metadata[truth_index]["topology"],
                "relative_topology": metadata[relative_index]["topology"],
                "attempt": attempt_spec["morph_attempt_index"],
                "morph_attempt_index": attempt_spec["morph_attempt_index"],
                "morph_strength": attempt_spec["morph_strength"],
                "physics_seed": r1.WRITER_SEED_ROOT + writer * r1.WRITER_SEED_MULTIPLIER + seed_offset,
                "physics_batch_position": physics_batch_position,
                "writer_latent": asdict(_latents_r3()[writer]), "style_audit": style_audit,
                "spatial_rms_from_truth_parent": rms, "tensor_sha256": digest,
                "frozen_top5": actual["top5"], "frozen_top1": actual["top5"][0],
                "candidate_action_inside_frozen_top5": True,
                "baseline_action_inside_frozen_top5": True,
                "auxiliary_unadmitted_labels": decision["auxiliary_unadmitted_labels"],
                "auxiliary_unsupported_present": decision["auxiliary_unsupported_present"],
                "runtime_unsupported_option_policy": "whole_row_identity",
                "external_approved_parents": True, "project_rows": 0,
            })
            break
        if accepted is None:
            counters["all_attempts_drop"] += 1
            continue
        value, row = accepted
        seen_hashes.add(row["tensor_sha256"])
        rows.append(value)
        rows_meta.append(row)
        pair_admitted[pair].add(writer)
        counters[orientation] += 1

    all_pairs = {
        (int(entry["pair"]["baseline_label_index"]), int(entry["pair"]["candidate_label_index"]))
        for entry in entries
    }
    failed_pairs = [
        {"baseline_label_index": pair[0], "candidate_label_index": pair[1],
         "unique_admitted_writers": len(pair_admitted.get(pair, set())), "policy": "identity_only"}
        for pair in sorted(all_pairs) if len(pair_admitted.get(pair, set())) < MIN_PAIR_WRITERS
    ]
    failed_set = {(row["baseline_label_index"], row["candidate_label_index"]) for row in failed_pairs}
    keep = [
        index for index, row in enumerate(rows_meta)
        if (int(row["directed_pair"]["baseline_label_index"]),
            int(row["directed_pair"]["candidate_label_index"])) not in failed_set
    ]
    return [rows[index] for index in keep], [rows_meta[index] for index in keep], {
        "counters": dict(counters), "failed_pairs": failed_pairs,
        "retained_before_final_reinference": len(keep),
        "recovery_contract": {"exact_topology_rejection": r2.TOPOLOGY_REJECTION,
                              "broad_exception_catch": False,
                              "morph_attempt_index_separate_from_physics_batch_position": True},
    }


def _final_reinference_filter(
    cal_rows: list[np.ndarray], cal_meta: list[dict],
    query_rows: list[np.ndarray], query_meta: list[dict],
    model, homographs: tuple[frozenset[int], ...], all_pair_keys: set[tuple[int, int]],
) -> tuple[list[np.ndarray], list[dict], list[np.ndarray], list[dict], dict]:
    if not cal_rows or not query_rows:
        raise ValueError("R3 generated no calibration or query rows before final admission")
    all_rows = cal_rows + query_rows
    all_meta = cal_meta + query_meta
    values = np.stack(all_rows).astype(np.float32, copy=False)
    logits = r1._predict(model, values)
    kept_cal_indices: list[int] = []
    kept_query_indices: list[int] = []
    counters: Counter = Counter()

    for index, row in enumerate(all_meta):
        top5 = np.argsort(logits[index])[-5:][::-1].astype(int).tolist()
        if top5 != [int(value) for value in row["frozen_top5"]]:
            counters["stored_top5_mismatch_drop"] += 1
            continue
        if row["episode_split"] == "calibration":
            candidate = int(row["candidate_label_index"])
            if candidate not in top5 or r1._homograph_collision(top5, homographs):
                counters["calibration_direct_or_context_drop"] += 1
                continue
            kept_cal_indices.append(index)
            continue
        pair = row["directed_pair"]
        baseline = int(pair["baseline_label_index"])
        candidate = int(pair["candidate_label_index"])
        stroke_count = r1._topology(values[index])[0]
        actual = r1._risk(logits[index], stroke_count, candidate)
        decision = _action_decision(actual, row["risk_stratum"], baseline, candidate, homographs)
        if decision["disposition"] != "action":
            counters[f"final_{decision['disposition']}_drop"] += 1
            continue
        kept_query_indices.append(index - len(cal_rows))

    support_counts = Counter(
        (int(all_meta[index]["global_writer_id"]), int(all_meta[index]["candidate_label_index"]))
        for index in kept_cal_indices
    )
    supported_query_indices = []
    for index in kept_query_indices:
        row = query_meta[index]
        key = (int(row["global_writer_id"]), int(row["directed_pair"]["candidate_label_index"]))
        if support_counts[key] < r1.CALIBRATION_SUPPORT:
            counters["final_calibration_support_drop"] += 1
            continue
        supported_query_indices.append(index)

    pair_writers: dict[tuple[int, int], set[int]] = defaultdict(set)
    for index in supported_query_indices:
        row = query_meta[index]
        pair = (int(row["directed_pair"]["baseline_label_index"]),
                int(row["directed_pair"]["candidate_label_index"]))
        pair_writers[pair].add(int(row["global_writer_id"]))
    retained_pairs = {pair for pair, writers in pair_writers.items() if len(writers) >= MIN_PAIR_WRITERS}
    final_query_indices = [
        index for index in supported_query_indices
        if (int(query_meta[index]["directed_pair"]["baseline_label_index"]),
            int(query_meta[index]["directed_pair"]["candidate_label_index"])) in retained_pairs
    ]
    counters["pair_minimum_drop"] += len(supported_query_indices) - len(final_query_indices)
    used_specs = {
        (int(query_meta[index]["global_writer_id"]),
         int(query_meta[index]["directed_pair"]["candidate_label_index"]))
        for index in final_query_indices
    }
    final_cal_indices = [
        index for index in kept_cal_indices
        if (int(all_meta[index]["global_writer_id"]), int(all_meta[index]["candidate_label_index"])) in used_specs
    ]
    final_cal_rows = [all_rows[index] for index in final_cal_indices]
    final_cal_meta = [all_meta[index] for index in final_cal_indices]
    final_query_rows = [query_rows[index] for index in final_query_indices]
    final_query_meta = [query_meta[index] for index in final_query_indices]

    violations = 0
    if final_query_rows:
        query_values = np.stack(final_query_rows).astype(np.float32, copy=False)
        final_logits = r1._predict(model, query_values)
        for index, row in enumerate(final_query_meta):
            pair = row["directed_pair"]
            candidate = int(pair["candidate_label_index"])
            actual = r1._risk(final_logits[index], r1._topology(query_values[index])[0], candidate)
            decision = _action_decision(
                actual, row["risk_stratum"], int(pair["baseline_label_index"]), candidate, homographs
            )
            violations += int(decision["disposition"] != "action")
    if violations:
        raise AssertionError(f"R3 final candidate violations: {violations}")
    failed_pairs = sorted(all_pair_keys - retained_pairs)
    minimum_writers = min((len(pair_writers[pair]) for pair in retained_pairs), default=0)
    pair_gate = (
        all(len(pair_writers[pair]) >= MIN_PAIR_WRITERS for pair in retained_pairs)
        and retained_pairs.isdisjoint(set(failed_pairs))
        and retained_pairs | set(failed_pairs) == all_pair_keys
    )
    return final_cal_rows, final_cal_meta, final_query_rows, final_query_meta, {
        "counters": dict(counters), "candidate_violations": violations,
        "retained_pairs": len(retained_pairs), "identity_only_pairs": len(failed_pairs),
        "minimum_retained_pair_writers": minimum_writers,
        "pair_writer_min16_or_identity_only": pair_gate,
        "failed_pairs": [
            {"baseline_label_index": pair[0], "candidate_label_index": pair[1],
             "unique_admitted_writers": len(pair_writers.get(pair, set())), "policy": "identity_only"}
            for pair in failed_pairs
        ],
    }


def _exclusive_bytes(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _write_failure(output: Path, independent_audit: Path, error: BaseException) -> None:
    marker = output / "GENERATION_STARTED.json"
    if not marker.exists():
        return
    payload = {
        "status": "V12_ACTION_EVIDENCE_R3_GENERATION_FAILED_IMMUTABLE",
        "failed_at": datetime.now(timezone.utc).isoformat(),
        "exception_type": type(error).__name__, "exception_message": str(error),
        "traceback_sha256": hashlib.sha256(traceback.format_exc().encode()).hexdigest(),
        "source_sha256": sha256(Path(__file__)), "started_marker_sha256": sha256(marker),
        "independent_audit": {"path": str(independent_audit.resolve()), "sha256": sha256(independent_audit)},
        "r2_rejection": _validate_r2_rejection(), "r3_static_lineage": _validate_r3_static_lineage(),
        "retry_allowed": False, "postprocessing_performed": False, "training_performed": False,
        "writers096_127_opened": False, "legacy_real_crohme_mathwriting_opened": False,
        "hwr_checkpoint_runtime_changed": False, "product_promotion": False,
    }
    r2._atomic_exclusive_json(output / "GENERATION_FAILED.json", payload)


def generate(output: Path, postprocessed: Path, independent_audit: Path) -> None:
    r2_rejection = _validate_r2_rejection()
    static_lineage = _validate_r3_static_lineage()
    pregen_audit = _validate_pregeneration_audit(independent_audit)
    if output.exists() or postprocessed.exists():
        raise FileExistsError("R3 raw or postprocessed output already exists; generation is one-shot")
    output.mkdir(parents=True)
    source_snapshot = output / "generate_action_evidence_augmentation_v12_r3.source.py"
    _exclusive_bytes(source_snapshot, Path(__file__).read_bytes())
    marker = {
        "status": "V12_ACTION_EVIDENCE_R3_GENERATION_STARTED",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "source_sha256": sha256(source_snapshot), "r2_rejection": r2_rejection,
        "r3_static_lineage": static_lineage, "pregeneration_audit": pregen_audit,
        "bootstrap_source_sha256": {
            name: expected for name, (_path, expected) in sorted(_BOOTSTRAP_SOURCES.items())
        },
        "external_bank_not_loaded_before_this_marker": True,
        "checkpoint_not_loaded_before_this_marker": True,
        "catalog_entries_not_loaded_before_this_marker": True,
        "retry_allowed": False,
    }
    r2._atomic_exclusive_json(output / "GENERATION_STARTED.json", marker)

    try:
        # No model, external tensor, or catalog entry load may move above STARTED.
        r1._validate_live_generation_inputs()
        helper_hashes = r1._validate_helper_sources()
        entries = _catalog_entries_r3()
        with np.load(r1.EXTERNAL_BANK, allow_pickle=False) as payload:
            features = np.asarray(payload["features"], dtype=np.float32)
            labels = np.asarray(payload["labels"], dtype=np.int64)
        metadata = r1._load_metadata(r1.EXTERNAL_METADATA)
        if len(features) != len(labels) or len(labels) != len(metadata):
            raise ValueError("external bank row alignment failed")
        model, tokens = r1._model(r1.CHECKPOINT)
        homographs = r1._resolve_homographs(tokens)
        pool = r1._parent_pool(labels, metadata)
        seen_hashes: set[str] = set()
        original_latents = r1._latents
        r1._latents = _latents_r3
        try:
            cal_rows, cal_meta, cal_fingerprints, unavailable, cal_audit = _build_calibration_r3(
                _calibration_specs(entries), features, labels, metadata, pool, model, homographs, seen_hashes,
            )
            query_rows, query_meta, query_audit = _build_queries_r3(
                entries, features, metadata, pool, model, homographs, cal_fingerprints, seen_hashes,
            )
        finally:
            r1._latents = original_latents
        all_pair_keys = {
            (int(entry["pair"]["baseline_label_index"]), int(entry["pair"]["candidate_label_index"]))
            for entry in entries
        }
        cal_rows, cal_meta, query_rows, query_meta, final_audit = _final_reinference_filter(
            cal_rows, cal_meta, query_rows, query_meta, model, homographs, all_pair_keys,
        )
        if not query_rows or final_audit["retained_pairs"] == 0:
            raise ValueError("R3 final admission retained no action-evidence query pairs")
        values = np.stack(cal_rows + query_rows).astype(np.float32, copy=False)
        row_meta = cal_meta + query_meta
        row_labels = np.asarray([
            int(row["label_index"]) if row["episode_split"] == "calibration"
            else int(row["truth_label_index"]) for row in row_meta
        ], dtype=np.int64)
        writers = np.asarray([int(row["global_writer_id"]) for row in row_meta], dtype=np.int16)
        splits = np.asarray([0 if row["episode_split"] == "calibration" else 1 for row in row_meta], dtype=np.int8)
        if set(np.unique(splits).tolist()) != {0, 1}:
            raise AssertionError("R3 final bank must contain calibration and query")
        final_hashes = [r1.tensor_hash(value) for value in values]
        if len(final_hashes) != len(set(final_hashes)):
            raise AssertionError("R3 final admitted tensor duplicate")
        if final_hashes != [str(row["tensor_sha256"]) for row in row_meta]:
            raise AssertionError("R3 final tensor/metadata hash mismatch")
        physics_streams = [
            (int(row["physics_seed"]), int(row["physics_batch_position"])) for row in row_meta
        ]
        if len(physics_streams) != len(set(physics_streams)):
            raise AssertionError("R3 admitted rows reused a physics RNG substream")
        expected_dt = np.full(128, np.float32(1.0 / 127.0), dtype=np.float32)
        expected_dt[0] = 0.0
        source_indices = [
            int(row["source_index"] if row["episode_split"] == "calibration" else row["truth_source_index"])
            for row in row_meta
        ]
        nonspatial_exact = all(
            np.array_equal(values[index, :, 2:], features[source_index, :, 2:])
            for index, source_index in enumerate(source_indices)
        )
        topology_parent_exact = all(
            r1._topology(values[index]) == r1._topology(features[source_index])
            for index, source_index in enumerate(source_indices)
        )
        external_approved_only = all(
            bool(row.get("external_approved_parent", row.get("external_approved_parents", False)))
            and int(row.get("project_rows", -1)) == 0 for row in row_meta
        )
        uniform_time = bool(np.all(values[:, :, 2] == expected_dt[None]))
        observed_exact = bool(np.all(values[:, :, 4] == 1.0))
        finite_unit_box = bool(
            np.isfinite(values).all() and values[:, :, :2].min() >= 0.0 and values[:, :, :2].max() <= 1.0
        )
        identity_zero = all(float(row["spatial_rms_from_truth_parent"]) > r1.MIN_REAL_RMS for row in row_meta)
        pair_gate = bool(final_audit["pair_writer_min16_or_identity_only"])
        candidate_zero = int(final_audit["candidate_violations"]) == 0
        computed_gates = {
            "calibration_and_query_present": bool(len(cal_rows) and len(query_rows)),
            "finite_unit_box": finite_unit_box,
            "within_new_raw_bank_duplicates_zero": len(final_hashes) == len(set(final_hashes)),
            "within_new_raw_bank_identity_zero": identity_zero,
            "uniform_time_exact": uniform_time, "observed_exact": observed_exact,
            "nonspatial_parent_exact": nonspatial_exact,
            "topology_parent_exact": topology_parent_exact,
            "external_approved_only": external_approved_only,
            "pair_writer_min16_or_identity_only": pair_gate,
            "candidate_action_within_frozen_top5": candidate_zero,
            "bank_candidate_violations_zero": candidate_zero,
            "admitted_physics_rng_substreams_unique": len(physics_streams) == len(set(physics_streams)),
            "adapter_training_not_performed": True, "hwr_checkpoint_runtime_unchanged": True,
            "writers096_127_remained_closed": True,
            "legacy_real_crohme_mathwriting_remained_closed": True,
            "postprocessing_not_performed": True, "product_promotion_not_performed": True,
        }
        if not all(computed_gates.values()):
            raise AssertionError(f"R3 final computed gate failure: {computed_gates}")
        np.savez_compressed(
            output / "action_evidence_raw_bank.npz",
            features=values, labels=row_labels, writers=writers, split=splits,
        )
        r1._write_jsonl_gzip(output / "action_evidence_raw_bank.metadata.jsonl.gz", row_meta)
        report = {
            "schema": SCHEMA, "status": "V12_R3_ACTION_EVIDENCE_RAW_GENERATED_AUDIT_REQUIRED",
            "rows": len(values), "calibration_rows": len(cal_rows), "query_rows": len(query_rows),
            "writers": len(set(writers.tolist())), "writer_range": [min(writers), max(writers)],
            "classes": len(set(row_labels.tolist())),
            "orientations": dict(Counter(row.get("orientation", "calibration") for row in row_meta)),
            "calibration_generation": cal_audit, "query_generation": query_audit,
            "unavailable_calibration_writer_candidates": unavailable,
            "final_admission": final_audit, "gates": computed_gates,
            "boundaries": {
                "raw_count_not_forced": True, "postprocessed_output": str(postprocessed.resolve()),
                "postprocessed_output_exists": postprocessed.exists(),
                "auxiliary_unadmitted_policy": "telemetry_and_runtime_whole_row_identity_not_query_drop",
                "homograph_policy": "context_owned_whole_row_identity_not_action",
                "cross_bank_000_127_duplicate_identity_audit": "pending_independent_raw_full_cross_audit",
                "adapter_trained": False, "hwr_changed": False, "writers096_127_opened": False,
                "legacy_rows": 0, "real_rows": 0, "crohme_rows": 0, "mathwriting_rows": 0,
            },
            "hashes": {
                "source_snapshot": sha256(source_snapshot),
                "started_marker": sha256(output / "GENERATION_STARTED.json"),
                "bank": sha256(output / "action_evidence_raw_bank.npz"),
                "metadata": sha256(output / "action_evidence_raw_bank.metadata.jsonl.gz"),
                "checkpoint": r1.EXPECTED_CHECKPOINT_SHA256,
                "external_bank": r1.EXPECTED_EXTERNAL_BANK_SHA256,
                "external_metadata": r1.EXPECTED_EXTERNAL_METADATA_SHA256,
                "r3_dry_catalog": EXPECTED_R3_DRY_CATALOG_SHA256,
                "r3_static_audit": EXPECTED_R3_STATIC_AUDIT_SHA256,
                "r3_pregeneration_audit": pregen_audit["sha256"],
            },
            "helper_source_sha256": helper_hashes,
            "bootstrap_source_sha256": {
                name: expected for name, (_path, expected) in sorted(_BOOTSTRAP_SOURCES.items())
            },
            "admitted_label_provenance": {
                "count": len(r1.ADMITTED_LABELS), "sha256": r1.ADMITTED_LABELS_SHA256,
                "auxiliary_labels_are_telemetry_only": True,
            },
            "tokens_sha256": r1.canonical_hash(tokens),
            "writer_latents_sha256": r1.canonical_hash([asdict(value) for value in _latents_r3()[192:256]]),
            "seen_hashes_including_dropped_rows": len(seen_hashes), "final_hashes": len(final_hashes),
        }
        (output / "generation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    except BaseException as error:
        _write_failure(output, independent_audit, error)
        raise


def dry_toy() -> dict:
    """Policy-only test; no catalog, external tensor, checkpoint, physics, or forward."""

    homographs = (frozenset({20, 21}),)
    expected = {"candidate_rank": 1, "top1_margin_bin": 2,
                "top5_entropy_bin": 3, "stroke_count_bucket": 1}
    accepted_actual = {"top5": [10, 11, 999, 12, 13], **expected}
    accepted = _action_decision(accepted_actual, expected, 10, 11, homographs)
    homograph_actual = {"top5": [10, 11, 20, 21, 13], **expected}
    context_identity = _action_decision(homograph_actual, expected, 10, 11, homographs)
    missing_candidate = _action_decision(
        {"top5": [10, 12, 13, 14, 15], **expected}, expected, 10, 11, homographs
    )
    risk_mismatch = _action_decision(
        {"top5": [10, 11, 12, 13, 14], **{**expected, "top1_margin_bin": 4}},
        expected, 10, 11, homographs,
    )
    pair_writers = {
        (1, 2): set(range(192, 208)),
        (2, 3): set(range(192, 207)),
    }
    retained = {pair for pair, writers in pair_writers.items() if len(writers) >= MIN_PAIR_WRITERS}
    with tempfile.TemporaryDirectory(prefix="v12-r3-exclusive-toy-") as directory:
        receipt = Path(directory) / "receipt.json"
        payload = {"status": "TOY", "retry_allowed": False}
        r2._atomic_exclusive_json(receipt, payload)
        atomic_exact = json.loads(receipt.read_text(encoding="utf-8")) == payload
        try:
            r2._atomic_exclusive_json(receipt, {"status": "MUST_NOT_REPLACE"})
        except FileExistsError:
            nonreplacement = json.loads(receipt.read_text(encoding="utf-8")) == payload
        else:
            nonreplacement = False
    result = {
        "status": "V12_R3_GENERATOR_DRY_TOY_PASSED",
        "catalog_loaded": False, "external_tensor_loaded": False, "checkpoint_loaded": False,
        "physics_executed": False, "frozen_hwr_forward_executed": False,
        "auxiliary_unadmitted_action_accepted": accepted["disposition"] == "action",
        "auxiliary_unadmitted_telemetry_exact": accepted["auxiliary_unadmitted_labels"] == [999],
        "homograph_context_owned_identity": context_identity["disposition"] == "context_owned_identity",
        "missing_candidate_direct_reject": missing_candidate["disposition"] == "direct_membership_reject",
        "risk_mismatch_reject": risk_mismatch["disposition"] == "risk_mismatch_reject",
        "pair_16_retained": (1, 2) in retained, "pair_15_identity_only": (2, 3) not in retained,
        "atomic_receipt_exact": atomic_exact, "existing_receipt_not_replaced": nonreplacement,
        "writers096_127_opened": False, "training_performed": False,
    }
    positive_keys = {
        "auxiliary_unadmitted_action_accepted", "auxiliary_unadmitted_telemetry_exact",
        "homograph_context_owned_identity", "missing_candidate_direct_reject",
        "risk_mismatch_reject", "pair_16_retained", "pair_15_identity_only",
        "atomic_receipt_exact", "existing_receipt_not_replaced",
    }
    negative_keys = {
        "catalog_loaded", "external_tensor_loaded", "checkpoint_loaded", "physics_executed",
        "frozen_hwr_forward_executed", "writers096_127_opened", "training_performed",
    }
    if (not all(result[key] is True for key in positive_keys)
            or not all(result[key] is False for key in negative_keys)):
        raise AssertionError(f"R3 dry policy contract failed: {result}")
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
        print(json.dumps(dry_toy(), indent=2))
        return 0
    if args.independent_audit is None:
        parser.error("--generate requires --independent-audit")
    for path in (args.output.resolve(), args.postprocessed_output.resolve(), args.independent_audit.resolve()):
        if path.drive.upper() != "D:":
            parser.error("generation and audit paths must remain on D:")
    generate(args.output, args.postprocessed_output, args.independent_audit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
