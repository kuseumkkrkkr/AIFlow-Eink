#!/usr/bin/env python3
"""V12 R4 telemetry-first generation-only builder; execution requires a later audit."""

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


ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = {
    "generate_action_evidence_augmentation_v12_r3.py": (
        ROOT / "scripts/generate_action_evidence_augmentation_v12_r3.py",
        "23aef377213e07bc46629546caec3bb02f2f4b0891df466968f28e5eeacc9d4f",
    ),
    "prepare_action_evidence_augmentation_v12_r4_r2.py": (
        ROOT / "scripts/prepare_action_evidence_augmentation_v12_r4_r2.py",
        "07f1019a96705bab7bdcbe459b10e7454a65b661d9fadfc15dc625c829f63005",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


for _name, (_path, _wanted) in BOOTSTRAP.items():
    if sha256(_path) != _wanted:
        raise RuntimeError(f"pre-import R4 dependency drift: {_name}")

r3 = importlib.import_module("generate_action_evidence_augmentation_v12_r3")
prepare = importlib.import_module("prepare_action_evidence_augmentation_v12_r4_r2")
r2 = r3.r2
r1 = r3.r1
np = r3.np


R3_RAW = ROOT / "artifacts/action_evidence_augmentation_v12_20260824_r3_raw"
R3_FAILURE_AUDIT = ROOT / "reports/V12_ACTION_EVIDENCE_R3_INDEPENDENT_FAILURE_AUDIT.json"
R4_R2_STATIC_AUDIT = ROOT / "reports/V12_ACTION_EVIDENCE_R4_R2_INDEPENDENT_STATIC_AUDIT.json"
DEFAULT_RAW_OUTPUT = ROOT / "artifacts/action_evidence_augmentation_v12_20260824_r4_raw"
DEFAULT_POSTPROCESSED_OUTPUT = ROOT / "artifacts/action_evidence_augmentation_v12_20260824_r4_final"

EXPECTED_R3_FAILURE_AUDIT_SHA256 = "793ee301ce01f1a3bb0dddfd51cb05d925b4c6e611a99bf81844c3bdc6d7a069"
EXPECTED_R4_R2_STATIC_AUDIT_SHA256 = "c3329815d8e287c20b872fa58e6d392e57680e6a329f8767aabb372c6059640d"
EXPECTED_R4_R2_STATIC_AUDIT_STATUS = "INDEPENDENT_V12_R4_R2_TELEMETRY_FIRST_STATIC_AUDIT_PASSED"
EXPECTED_PREGEN_AUDIT_STATUS = "INDEPENDENT_V12_R4_GENERATOR_PREGENERATION_AUDIT_PASSED"
EXPECTED_PAIR_COUNT = 9_867
EXPECTED_ACTION_COUNT = 197_340
WRITER_IDS = tuple(range(256, 320))
MIN_PAIR_WRITERS = 16
SCHEMA = "aiflow-v12-action-evidence-r4-raw/v1"


def _validate_static_lineage() -> dict:
    if sha256(R3_FAILURE_AUDIT) != EXPECTED_R3_FAILURE_AUDIT_SHA256:
        raise ValueError("R3 failure audit drift")
    if sha256(R4_R2_STATIC_AUDIT) != EXPECTED_R4_R2_STATIC_AUDIT_SHA256:
        raise ValueError("R4-r2 static audit drift")
    r3_audit = json.loads(R3_FAILURE_AUDIT.read_text(encoding="utf-8"))
    r4_audit = json.loads(R4_R2_STATIC_AUDIT.read_text(encoding="utf-8"))
    if (r3_audit.get("status") != "INDEPENDENT_V12_R3_FAILURE_AUDIT_CONFIRMED"
            or not r3_audit.get("gates") or not all(r3_audit["gates"].values())):
        raise ValueError("R3 failure boundary is not all-gates PASS")
    if (r4_audit.get("status") != EXPECTED_R4_R2_STATIC_AUDIT_STATUS
            or not r4_audit.get("gates") or not all(r4_audit["gates"].values())):
        raise ValueError("R4-r2 static audit is not all-gates PASS")
    decision = r4_audit.get("decision", {})
    if decision.get("generation_source_implementation_allowed") is not True:
        raise ValueError("R4 generation source implementation not authorized")
    if decision.get("generation_execution_allowed") is not False:
        raise ValueError("R4 static audit execution boundary drift")
    return {
        "r3_failure_audit_sha256": EXPECTED_R3_FAILURE_AUDIT_SHA256,
        "r4_r2_static_audit_sha256": EXPECTED_R4_R2_STATIC_AUDIT_SHA256,
    }


def _validate_pregeneration_audit(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    gates = payload.get("gates", {})
    if payload.get("status") != EXPECTED_PREGEN_AUDIT_STATUS or not gates or not all(gates.values()):
        raise ValueError("R4 generator pre-generation audit is not all-gates PASS")
    if payload.get("generator_source_sha256") != sha256(Path(__file__)):
        raise ValueError("R4 generator pre-generation source mismatch")
    if payload.get("r4_r2_static_audit_sha256") != EXPECTED_R4_R2_STATIC_AUDIT_SHA256:
        raise ValueError("R4 generator pre-generation lineage mismatch")
    decision = payload.get("decision", {})
    if decision.get("generation_allowed") is not True or decision.get("training_allowed") is not False:
        raise ValueError("R4 generator pre-generation decision mismatch")
    return {"path": str(path.resolve()), "sha256": sha256(path), "status": payload["status"]}


@functools.lru_cache(maxsize=1)
def _latents_r4():
    return tuple(r1._sample_latents(320))


def _catalog_entries_r4() -> list[dict]:
    source = r3._catalog_entries_r3()
    entries = []
    keys = set()
    for row in source:
        writer = int(row["writer_id"]) + 64
        entry = {
            "pair": {key: int(value) for key, value in row["pair"].items()},
            "risk_stratum": {key: int(value) for key, value in row["risk_stratum"].items()},
            "writer_id": writer, "orientation": str(row["orientation"]),
        }
        key = (entry["pair"]["baseline_label_index"], entry["pair"]["candidate_label_index"], writer)
        if key in keys:
            raise ValueError("R4 pair-writer action duplicated")
        keys.add(key); entries.append(entry)
    if len(entries) != EXPECTED_ACTION_COUNT or len(keys) != EXPECTED_ACTION_COUNT:
        raise ValueError("R4 action count drift")
    pair_writers: dict[tuple[int, int], set[int]] = defaultdict(set)
    orientations: dict[tuple[int, int], Counter] = defaultdict(Counter)
    for entry in entries:
        pair = (entry["pair"]["baseline_label_index"], entry["pair"]["candidate_label_index"])
        pair_writers[pair].add(entry["writer_id"]); orientations[pair][entry["orientation"]] += 1
    if len(pair_writers) != EXPECTED_PAIR_COUNT:
        raise ValueError("R4 pair count drift")
    for pair, writers in pair_writers.items():
        if len(writers) != 20 or not writers.issubset(WRITER_IDS):
            raise ValueError(f"R4 writer allocation drift: {pair}")
        if orientations[pair] != {"candidate_truth_promotion": 10, "baseline_truth_veto": 10}:
            raise ValueError(f"R4 orientation drift: {pair}")
    return entries


def _calibration_specs(entries: list[dict]) -> set[tuple[int, int]]:
    return {(int(row["writer_id"]), int(row["pair"]["candidate_label_index"])) for row in entries}


def _blank_stage_counts() -> dict:
    return prepare.blank_telemetry(0)


def _telemetry_payload(counts: dict, lineage: dict, pregen: dict) -> dict:
    first_empty = prepare.first_empty_stage(counts)
    return {
        "schema": "aiflow-v12-r4-prefinal-stage-telemetry/v1",
        "status": "V12_R4_PRE_FINAL_STAGE_TELEMETRY_IMMUTABLE",
        "published_at": datetime.now(timezone.utc).isoformat(),
        "first_empty_stage": first_empty,
        "counts": counts,
        "lineage": {
            **lineage, "pregeneration_audit_sha256": pregen["sha256"],
            "source_sha256": sha256(Path(__file__)),
            "r3_policy_catalog_sha256": prepare.EXPECTED["r3_policy_catalog"],
        },
        "boundaries": {
            "aggregate_counts_and_hashes_only": True,
            "row_level_tensor_metadata_label_writer_pair_class": False,
            "writer_range": [256, 319], "training_performed": False,
            "writers096_127_opened": False, "legacy_real_crohme_mathwriting_opened": False,
        },
    }


def _publish_telemetry(output: Path, counts: dict, lineage: dict, pregen: dict) -> Path:
    path = output / "PRE_FINAL_STAGE_TELEMETRY.json"
    if path.exists():
        return path
    r2._atomic_exclusive_json(path, _telemetry_payload(counts, lineage, pregen))
    return path


def _build_calibration(
    specs: set[tuple[int, int]], features: np.ndarray, metadata: list[dict],
    pool: dict[tuple[int, int], list[int]], model, homographs: tuple[frozenset[int], ...],
    seen_hashes: set[str], counts: dict,
) -> tuple[list[np.ndarray], list[dict], dict[tuple[int, int], set[str]]]:
    stage = counts["calibration"]
    stage["requested_specs"] = len(specs)
    rows: list[np.ndarray] = []
    rows_meta: list[dict] = []
    fingerprints_by_spec: dict[tuple[int, int], set[str]] = {}
    for writer, label in sorted(specs):
        candidates = [
            (strokes, index)
            for (pool_label, strokes), indices in sorted(pool.items()) if pool_label == label
            for index in indices
        ]
        if not candidates:
            continue
        stage["specs_with_parent_candidates"] += 1
        stage["parent_candidate_rows"] += len(candidates)
        candidates.sort(key=lambda item: hashlib.sha256(
            f"v12-r4:cal:{writer}:{label}:{metadata[item[1]]['synthetic_id']}".encode()
        ).hexdigest())
        accepted: list[tuple[np.ndarray, dict]] = []
        used_fingerprints: set[str] = set()
        local_hashes: set[str] = set()
        flags = {"parent": False, "valid": False, "top5": False, "homograph": False, "duplicate": False}
        for _strokes, index in candidates:
            fingerprints = r1._parent_fingerprints(metadata[index])
            if fingerprints & used_fingerprints:
                continue
            flags["parent"] = True
            source = features[index]
            seed_offset = r1._seed_offset(
                f"r4:cal:{writer}:{label}:{metadata[index]['synthetic_id']}:{len(accepted)}"
            )
            generated, audits = r1._style_and_physics([source], writer, seed_offset)
            value = generated[0]
            row_logits = r1._predict(model, value[None])[0]
            top5 = np.argsort(row_logits)[-5:][::-1].astype(int).tolist()
            valid, rms = r1._valid_tensor(value, source)
            digest = r1.tensor_hash(value)
            if not valid:
                stage["invalid_tensor_rejects"] += 1
                if r1._topology(value) != r1._topology(source):
                    stage["topology_rejects"] += 1
                continue
            flags["valid"] = True
            if label not in top5:
                stage["direct_top5_rejects"] += 1
                continue
            flags["top5"] = True
            if r1._homograph_collision(top5, homographs):
                stage["homograph_rejects"] += 1
                continue
            flags["homograph"] = True
            if digest in seen_hashes or digest in local_hashes:
                stage["duplicate_rejects"] += 1
                continue
            flags["duplicate"] = True
            auxiliary = sorted(value for value in top5 if value != label and value not in r1.ADMITTED_LABELS)
            local_hashes.add(digest); used_fingerprints |= fingerprints
            accepted.append((value, {
                "schema": SCHEMA, "episode_split": "calibration",
                "synthetic_writer_id": f"synthetic_writer_{writer:03d}", "global_writer_id": writer,
                "label_index": label, "candidate_label_index": label,
                "support_role": "candidate_calibration_anchor",
                "parent_synthetic_id": metadata[index]["synthetic_id"],
                "parent_fingerprints": sorted(fingerprints), "source_index": index,
                "source_topology": metadata[index]["topology"], "attempt": len(accepted),
                "physics_seed": r1.WRITER_SEED_ROOT + writer * r1.WRITER_SEED_MULTIPLIER + seed_offset,
                "physics_batch_position": 0, "writer_latent": asdict(_latents_r4()[writer]),
                "style_audit": audits[0], "spatial_rms_from_truth_parent": rms,
                "tensor_sha256": digest, "frozen_top5": top5,
                "auxiliary_unadmitted_labels": auxiliary,
                "auxiliary_unsupported_present": bool(auxiliary),
                "external_approved_parent": True, "project_rows": 0,
            }))
            if len(accepted) == r1.CALIBRATION_SUPPORT:
                break
        stage["specs_parent_disjoint"] += int(flags["parent"])
        stage["specs_topology_valid"] += int(flags["valid"])
        stage["specs_direct_top5"] += int(flags["top5"])
        stage["specs_homograph_clear"] += int(flags["homograph"])
        stage["specs_duplicate_clear"] += int(flags["duplicate"])
        if len(accepted) != r1.CALIBRATION_SUPPORT:
            continue
        stage["specs_support2_complete"] += 1
        fingerprints_by_spec[(writer, label)] = used_fingerprints
        seen_hashes.update(local_hashes)
        for value, row in accepted:
            rows.append(value); rows_meta.append(row)
    return rows, rows_meta, fingerprints_by_spec


def _build_queries(
    entries: list[dict], features: np.ndarray, metadata: list[dict],
    pool: dict[tuple[int, int], list[int]], model, homographs: tuple[frozenset[int], ...],
    calibration_fingerprints: dict[tuple[int, int], set[str]], seen_hashes: set[str], counts: dict,
) -> tuple[list[np.ndarray], list[dict]]:
    stage = counts["query"]
    stage["planned_actions"] = len(entries)
    rows: list[np.ndarray] = []
    rows_meta: list[dict] = []
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
            continue
        stage["actions_calibration_supported"] += 1
        truth = candidate if orientation == "candidate_truth_promotion" else baseline
        relative = baseline if truth == candidate else candidate
        target_strokes = expected["stroke_count_bucket"]
        available_strokes = sorted({
            strokes for label, strokes in pool
            if label == truth and (relative, strokes) in pool and min(int(strokes), 4) == target_strokes
        })
        if not available_strokes:
            continue
        stage["actions_shared_topology"] += 1
        strokes = available_strokes[0]
        forbidden = set(all_calibration_fingerprints[writer])
        truth_index = r1._select_parent(
            pool, metadata, truth, strokes, f"r4-query-truth:{pair}:{writer}", forbidden, set()
        )
        if truth_index is None:
            continue
        stage["actions_truth_parent_selected"] += 1
        truth_fp = r1._parent_fingerprints(metadata[truth_index]); forbidden |= truth_fp
        relative_index = r1._select_parent(
            pool, metadata, relative, strokes, f"r4-query-relative:{pair}:{writer}", forbidden, set()
        )
        if relative_index is None:
            continue
        stage["actions_relative_parent_selected"] += 1
        relative_fp = r1._parent_fingerprints(metadata[relative_index])
        source = features[truth_index]; relative_source = features[relative_index]
        morph_counters: Counter = Counter()
        attempt_rows, attempt_specs = r2._morph_attempts(source, relative_source, morph_counters)
        stage["morph_attempts"] += len(attempt_rows)
        stage["morph_topology_drops"] += int(morph_counters["morph_topology_attempt_drop"])
        if not attempt_rows:
            continue
        stage["actions_with_morph_candidate"] += 1
        seed_offset = r1._seed_offset(
            f"r4-query:{baseline}:{candidate}:{writer}:{r1.canonical_hash(expected)}"
        )
        generated, style_audits = r1._style_and_physics(attempt_rows, writer, seed_offset)
        logits = r1._predict(model, generated)
        accepted = None
        flags = {"membership": False, "risk": False, "homograph": False}
        for physics_position, (value, row_logits, spec, style_audit) in enumerate(
            zip(generated, logits, attempt_specs, style_audits, strict=True)
        ):
            actual = r1._risk(row_logits, strokes, candidate)
            decision = r3._action_decision(actual, expected, baseline, candidate, homographs)
            stage["auxiliary_unsupported_telemetry"] += int(decision["auxiliary_unsupported_present"])
            valid, rms = r1._valid_tensor(value, source)
            digest = r1.tensor_hash(value)
            if not valid:
                stage["invalid_tensor_rejects"] += 1
                continue
            if digest in seen_hashes:
                stage["duplicate_rejects"] += 1
                continue
            flags["membership"] |= bool(decision["direct_membership"])
            flags["risk"] |= bool(decision["direct_membership"] and decision["risk_exact"])
            flags["homograph"] |= bool(
                decision["direct_membership"] and decision["risk_exact"] and not decision["homograph_collision"]
            )
            if decision["disposition"] != "action":
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
                "truth_source_index": truth_index, "relative_source_index": relative_index,
                "truth_topology": metadata[truth_index]["topology"],
                "relative_topology": metadata[relative_index]["topology"],
                "morph_attempt_index": spec["morph_attempt_index"],
                "morph_strength": spec["morph_strength"],
                "physics_seed": r1.WRITER_SEED_ROOT + writer * r1.WRITER_SEED_MULTIPLIER + seed_offset,
                "physics_batch_position": physics_position,
                "writer_latent": asdict(_latents_r4()[writer]), "style_audit": style_audit,
                "spatial_rms_from_truth_parent": rms, "tensor_sha256": digest,
                "frozen_top5": actual["top5"], "frozen_top1": actual["top5"][0],
                "auxiliary_unadmitted_labels": decision["auxiliary_unadmitted_labels"],
                "auxiliary_unsupported_present": decision["auxiliary_unsupported_present"],
                "external_approved_parents": True, "project_rows": 0,
            })
            break
        stage["actions_direct_membership"] += int(flags["membership"])
        stage["actions_risk_exact"] += int(flags["risk"])
        stage["actions_homograph_clear"] += int(flags["homograph"])
        if accepted is None:
            continue
        value, row = accepted; seen_hashes.add(row["tensor_sha256"])
        rows.append(value); rows_meta.append(row); stage["actions_accepted"] += 1
    return rows, rows_meta


def _final_admission(
    cal_rows: list[np.ndarray], cal_meta: list[dict],
    query_rows: list[np.ndarray], query_meta: list[dict],
    model, homographs: tuple[frozenset[int], ...], counts: dict,
) -> tuple[list[np.ndarray], list[dict], list[np.ndarray], list[dict], dict]:
    cal_stage = counts["final_calibration"]
    query_stage = counts["final_query"]
    final = counts["final"]
    cal_stage["calibration_rows_entering"] = len(cal_rows)
    query_stage["query_rows_entering"] = len(query_rows)

    exact_cal_indices: list[int] = []
    if cal_rows:
        cal_values = np.stack(cal_rows).astype(np.float32, copy=False)
        cal_logits = r1._predict(model, cal_values)
        for index, row in enumerate(cal_meta):
            top5 = np.argsort(cal_logits[index])[-5:][::-1].astype(int).tolist()
            if top5 == [int(value) for value in row["frozen_top5"]]:
                exact_cal_indices.append(index)
    cal_stage["calibration_rows_stored_top5_exact"] = len(exact_cal_indices)
    final["calibration_stored_top5_mismatch_drops"] = len(cal_rows) - len(exact_cal_indices)

    exact_query_indices: list[int] = []
    action_query_indices: list[int] = []
    if query_rows:
        query_values = np.stack(query_rows).astype(np.float32, copy=False)
        query_logits = r1._predict(model, query_values)
        for index, row in enumerate(query_meta):
            top5 = np.argsort(query_logits[index])[-5:][::-1].astype(int).tolist()
            if top5 != [int(value) for value in row["frozen_top5"]]:
                continue
            exact_query_indices.append(index)
            pair = row["directed_pair"]
            candidate = int(pair["candidate_label_index"])
            actual = r1._risk(query_logits[index], r1._topology(query_values[index])[0], candidate)
            decision = r3._action_decision(
                actual, row["risk_stratum"], int(pair["baseline_label_index"]), candidate, homographs
            )
            if decision["disposition"] == "action":
                action_query_indices.append(index)
    query_stage["query_rows_stored_top5_exact"] = len(exact_query_indices)
    final["query_stored_top5_mismatch_drops"] = len(query_rows) - len(exact_query_indices)
    final["query_action_contract_drops"] = len(exact_query_indices) - len(action_query_indices)

    support_counts = Counter(
        (int(cal_meta[index]["global_writer_id"]), int(cal_meta[index]["candidate_label_index"]))
        for index in exact_cal_indices
    )
    supported_query_indices = []
    for index in action_query_indices:
        row = query_meta[index]
        key = (int(row["global_writer_id"]), int(row["directed_pair"]["candidate_label_index"]))
        if support_counts[key] >= r1.CALIBRATION_SUPPORT:
            supported_query_indices.append(index)
    query_stage["query_rows_after_support_filter"] = len(supported_query_indices)
    final["query_support_drops"] = len(action_query_indices) - len(supported_query_indices)

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
    query_stage["query_rows_after_pair16_filter"] = len(final_query_indices)
    final["query_pair_lt16_rows_dropped"] = len(supported_query_indices) - len(final_query_indices)

    used_specs = {
        (int(query_meta[index]["global_writer_id"]),
         int(query_meta[index]["directed_pair"]["candidate_label_index"]))
        for index in final_query_indices
    }
    final_cal_indices = [
        index for index in exact_cal_indices
        if (int(cal_meta[index]["global_writer_id"]), int(cal_meta[index]["candidate_label_index"])) in used_specs
    ]
    cal_stage["calibration_rows_after_support_retention"] = len(final_cal_indices)
    final["calibration_support_retention_drops"] = len(exact_cal_indices) - len(final_cal_indices)

    final_cal_rows = [cal_rows[index] for index in final_cal_indices]
    final_cal_meta = [cal_meta[index] for index in final_cal_indices]
    final_query_rows = [query_rows[index] for index in final_query_indices]
    final_query_meta = [query_meta[index] for index in final_query_indices]
    violations = 0
    if final_query_rows:
        values = np.stack(final_query_rows).astype(np.float32, copy=False)
        logits = r1._predict(model, values)
        for index, row in enumerate(final_query_meta):
            pair = row["directed_pair"]
            candidate = int(pair["candidate_label_index"])
            actual = r1._risk(logits[index], r1._topology(values[index])[0], candidate)
            violations += int(r3._action_decision(
                actual, row["risk_stratum"], int(pair["baseline_label_index"]),
                candidate, homographs,
            )["disposition"] != "action")
    final["candidate_violations"] = violations
    return final_cal_rows, final_cal_meta, final_query_rows, final_query_meta, {
        "retained_pairs": len(retained_pairs),
        "identity_only_pairs": EXPECTED_PAIR_COUNT - len(retained_pairs),
        "minimum_retained_pair_writers": min((len(pair_writers[pair]) for pair in retained_pairs), default=0),
        "candidate_violations": violations,
    }


def _exclusive_bytes(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())


def _write_failure(
    output: Path, audit_path: Path, error: BaseException,
    telemetry_path: Path, first_empty: str,
) -> None:
    marker = output / "GENERATION_STARTED.json"
    if not marker.exists() or not telemetry_path.exists():
        raise RuntimeError("R4 failure receipt forbidden before STARTED and telemetry") from error
    payload = {
        "status": "V12_ACTION_EVIDENCE_R4_GENERATION_FAILED_IMMUTABLE",
        "failed_at": datetime.now(timezone.utc).isoformat(),
        "exception_type": type(error).__name__, "exception_message": str(error),
        "traceback_sha256": hashlib.sha256(traceback.format_exc().encode()).hexdigest(),
        "source_sha256": sha256(Path(__file__)), "started_marker_sha256": sha256(marker),
        "prefinal_telemetry_sha256": sha256(telemetry_path), "first_empty_stage": first_empty,
        "independent_audit_sha256": sha256(audit_path),
        "retry_allowed": False, "postprocessing_performed": False, "training_performed": False,
        "writers096_127_opened": False, "legacy_real_crohme_mathwriting_opened": False,
        "hwr_checkpoint_runtime_changed": False, "product_promotion": False,
    }
    r2._atomic_exclusive_json(output / "GENERATION_FAILED.json", payload)


def generate(output: Path, postprocessed: Path, audit_path: Path) -> None:
    lineage = _validate_static_lineage()
    pregen = _validate_pregeneration_audit(audit_path)
    if output.exists() or postprocessed.exists():
        raise FileExistsError("R4 raw or postprocessed output already exists; generation is one-shot")
    output.mkdir(parents=True)
    source_snapshot = output / "generate_action_evidence_augmentation_v12_r4.source.py"
    _exclusive_bytes(source_snapshot, Path(__file__).read_bytes())
    marker = {
        "status": "V12_ACTION_EVIDENCE_R4_GENERATION_STARTED",
        "started_at": datetime.now(timezone.utc).isoformat(), "source_sha256": sha256(source_snapshot),
        "static_lineage": lineage, "pregeneration_audit": pregen,
        "telemetry_not_yet_published": True,
        "catalog_external_checkpoint_not_loaded_before_marker": True,
        "retry_allowed": False,
    }
    r2._atomic_exclusive_json(output / "GENERATION_STARTED.json", marker)
    counts = _blank_stage_counts()
    telemetry_path = output / "PRE_FINAL_STAGE_TELEMETRY.json"
    try:
        # All data/model/catalog loads remain after durable STARTED.
        r1._validate_live_generation_inputs()
        helper_hashes = r1._validate_helper_sources()
        entries = _catalog_entries_r4()
        specs = _calibration_specs(entries)
        counts["calibration"]["requested_specs"] = len(specs)
        counts["query"]["planned_actions"] = len(entries)
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
        r1._latents = _latents_r4
        try:
            cal_rows, cal_meta, cal_fingerprints = _build_calibration(
                specs, features, metadata, pool, model, homographs, seen_hashes, counts,
            )
            query_rows, query_meta = _build_queries(
                entries, features, metadata, pool, model, homographs,
                cal_fingerprints, seen_hashes, counts,
            )
        finally:
            r1._latents = original_latents
        cal_rows, cal_meta, query_rows, query_meta, final_audit = _final_admission(
            cal_rows, cal_meta, query_rows, query_meta, model, homographs, counts,
        )
        telemetry_path = _publish_telemetry(output, counts, lineage, pregen)
        first_empty = json.loads(telemetry_path.read_text(encoding="utf-8"))["first_empty_stage"]
        if first_empty != "NONE":
            raise ValueError(f"R4 pre-final gate failed at {first_empty}")
        if not cal_rows or not query_rows or final_audit["retained_pairs"] == 0:
            raise ValueError("R4 final admission unexpectedly empty after telemetry NONE")

        values = np.stack(cal_rows + query_rows).astype(np.float32, copy=False)
        row_meta = cal_meta + query_meta
        row_labels = np.asarray([
            int(row["label_index"]) if row["episode_split"] == "calibration"
            else int(row["truth_label_index"]) for row in row_meta
        ], dtype=np.int64)
        writers = np.asarray([int(row["global_writer_id"]) for row in row_meta], dtype=np.int16)
        splits = np.asarray([0 if row["episode_split"] == "calibration" else 1 for row in row_meta], dtype=np.int8)
        final_hashes = [r1.tensor_hash(value) for value in values]
        physics_streams = [(int(row["physics_seed"]), int(row["physics_batch_position"])) for row in row_meta]
        expected_dt = np.full(128, np.float32(1.0 / 127.0), dtype=np.float32); expected_dt[0] = 0.0
        source_indices = [
            int(row["source_index"] if row["episode_split"] == "calibration" else row["truth_source_index"])
            for row in row_meta
        ]
        gates = {
            "calibration_and_query_present": set(np.unique(splits).tolist()) == {0, 1},
            "finite_unit_box": bool(np.isfinite(values).all() and values[:, :, :2].min() >= 0 and values[:, :, :2].max() <= 1),
            "within_new_raw_bank_duplicates_zero": len(final_hashes) == len(set(final_hashes)),
            "metadata_tensor_hash_exact": final_hashes == [str(row["tensor_sha256"]) for row in row_meta],
            "within_new_raw_bank_identity_zero": all(float(row["spatial_rms_from_truth_parent"]) > r1.MIN_REAL_RMS for row in row_meta),
            "uniform_time_exact": bool(np.all(values[:, :, 2] == expected_dt[None])),
            "observed_exact": bool(np.all(values[:, :, 4] == 1.0)),
            "nonspatial_parent_exact": all(np.array_equal(values[i, :, 2:], features[s, :, 2:]) for i, s in enumerate(source_indices)),
            "topology_parent_exact": all(r1._topology(values[i]) == r1._topology(features[s]) for i, s in enumerate(source_indices)),
            "external_approved_only": all(bool(row.get("external_approved_parent", row.get("external_approved_parents", False))) and int(row.get("project_rows", -1)) == 0 for row in row_meta),
            "pair_writer_min16": final_audit["minimum_retained_pair_writers"] >= MIN_PAIR_WRITERS,
            "candidate_violations_zero": final_audit["candidate_violations"] == 0,
            "physics_rng_substreams_unique": len(physics_streams) == len(set(physics_streams)),
            "prefinal_telemetry_published_before_compound_guard": telemetry_path.exists(),
            "adapter_training_not_performed": True, "hwr_checkpoint_runtime_unchanged": True,
            "writers096_127_legacy_real_crohme_mathwriting_closed": True,
            "postprocessing_not_performed": True, "product_promotion_not_performed": True,
        }
        if not all(gates.values()):
            raise AssertionError(f"R4 final computed gates failed: {gates}")
        np.savez_compressed(output / "action_evidence_raw_bank.npz", features=values, labels=row_labels, writers=writers, split=splits)
        r1._write_jsonl_gzip(output / "action_evidence_raw_bank.metadata.jsonl.gz", row_meta)
        report = {
            "schema": SCHEMA, "status": "V12_R4_ACTION_EVIDENCE_RAW_GENERATED_AUDIT_REQUIRED",
            "rows": len(values), "calibration_rows": len(cal_rows), "query_rows": len(query_rows),
            "writers": len(set(writers.tolist())), "writer_range": [min(writers), max(writers)],
            "classes": len(set(row_labels.tolist())), "final_admission": final_audit,
            "aggregate_stage_counts": counts, "gates": gates,
            "boundaries": {
                "raw_count_not_forced": True, "postprocessed_output_exists": postprocessed.exists(),
                "cross_bank_000_255_audit": "pending independent raw/full/cross audit",
                "adapter_trained": False, "writers096_127_opened": False,
                "legacy_real_crohme_mathwriting_rows": 0, "hwr_changed": False,
            },
            "hashes": {
                "source_snapshot": sha256(source_snapshot),
                "started_marker": sha256(output / "GENERATION_STARTED.json"),
                "prefinal_telemetry": sha256(telemetry_path),
                "bank": sha256(output / "action_evidence_raw_bank.npz"),
                "metadata": sha256(output / "action_evidence_raw_bank.metadata.jsonl.gz"),
                "checkpoint": r1.EXPECTED_CHECKPOINT_SHA256,
                "external_bank": r1.EXPECTED_EXTERNAL_BANK_SHA256,
                "external_metadata": r1.EXPECTED_EXTERNAL_METADATA_SHA256,
                "r3_policy_catalog": prepare.EXPECTED["r3_policy_catalog"],
                "pregeneration_audit": pregen["sha256"],
            },
            "helper_source_sha256": helper_hashes,
            "bootstrap_source_sha256": {name: wanted for name, (_path, wanted) in sorted(BOOTSTRAP.items())},
            "writer_latents_sha256": r1.canonical_hash([asdict(value) for value in _latents_r4()[256:320]]),
            "tokens_sha256": r1.canonical_hash(tokens),
        }
        (output / "generation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    except BaseException as error:
        if not telemetry_path.exists():
            telemetry_path = _publish_telemetry(output, counts, lineage, pregen)
        first_empty = json.loads(telemetry_path.read_text(encoding="utf-8"))["first_empty_stage"]
        _write_failure(output, audit_path, error, telemetry_path, first_empty)
        raise


def dry_toy() -> dict:
    """No catalog, bank, checkpoint, physics, HWR forward, or training."""

    final_cal = _blank_stage_counts()
    for name in prepare.CALIBRATION_CHAIN:
        final_cal["calibration"][name] = 1
    for name in prepare.QUERY_CHAIN:
        final_cal["query"][name] = 1
    for name in prepare.FINAL_CALIBRATION_CHAIN:
        final_cal["final_calibration"][name] = 0
    for name in prepare.FINAL_QUERY_CHAIN:
        final_cal["final_query"][name] = 1
    final_query = _blank_stage_counts()
    for name in prepare.CALIBRATION_CHAIN:
        final_query["calibration"][name] = 1
    for name in prepare.QUERY_CHAIN:
        final_query["query"][name] = 1
    for name in prepare.FINAL_CALIBRATION_CHAIN:
        final_query["final_calibration"][name] = 1
    for name in prepare.FINAL_QUERY_CHAIN:
        final_query["final_query"][name] = 0
    lineage = {"toy_static": True}
    pregen = {"sha256": "0" * 64}
    with tempfile.TemporaryDirectory(prefix="v12-r4-telemetry-toy-") as directory:
        output = Path(directory)
        path = _publish_telemetry(output, final_cal, lineage, pregen)
        payload = json.loads(path.read_text(encoding="utf-8"))
        exact_first = payload["first_empty_stage"] == "final_calibration.calibration_rows_entering"
        aggregate_only = (
            "counts" in payload and "features" not in payload and "labels" not in payload
            and "metadata" not in payload and "rows" not in payload
        )
        try:
            r2._atomic_exclusive_json(path, {"status": "MUST_NOT_REPLACE"})
        except FileExistsError:
            nonreplacement = json.loads(path.read_text(encoding="utf-8")) == payload
        else:
            nonreplacement = False
        marker = output / "GENERATION_STARTED.json"
        r2._atomic_exclusive_json(marker, {"status": "TOY_STARTED"})
        missing_telemetry_output = output / "missing"
        missing_telemetry_output.mkdir()
        r2._atomic_exclusive_json(missing_telemetry_output / "GENERATION_STARTED.json", {"status": "TOY"})
        try:
            _write_failure(
                missing_telemetry_output, R4_R2_STATIC_AUDIT, RuntimeError("toy"),
                missing_telemetry_output / "PRE_FINAL_STAGE_TELEMETRY.json", "toy",
            )
        except RuntimeError as error:
            failure_before_telemetry_rejected = "forbidden before" in str(error)
        else:
            failure_before_telemetry_rejected = False
    result = {
        "status": "V12_R4_GENERATOR_DRY_TOY_PASSED",
        "catalog_loaded": False, "external_bank_loaded": False, "checkpoint_loaded": False,
        "physics_executed": False, "hwr_forward_executed": False, "training_performed": False,
        "writer_range_remap": [192 + 64, 255 + 64] == [256, 319],
        "final_calibration_zero_distinct": exact_first,
        "final_query_zero_distinct": prepare.first_empty_stage(final_query) == "final_query.query_rows_entering",
        "telemetry_aggregate_only": aggregate_only,
        "telemetry_nonreplacement": nonreplacement,
        "failure_before_telemetry_rejected": failure_before_telemetry_rejected,
        "writers096_127_opened": False,
    }
    positive = {
        "writer_range_remap", "final_calibration_zero_distinct", "final_query_zero_distinct",
        "telemetry_aggregate_only", "telemetry_nonreplacement", "failure_before_telemetry_rejected",
    }
    negative = {
        "catalog_loaded", "external_bank_loaded", "checkpoint_loaded", "physics_executed",
        "hwr_forward_executed", "training_performed", "writers096_127_opened",
    }
    if not all(result[key] is True for key in positive) or not all(result[key] is False for key in negative):
        raise AssertionError(f"R4 generator dry toy failed: {result}")
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
