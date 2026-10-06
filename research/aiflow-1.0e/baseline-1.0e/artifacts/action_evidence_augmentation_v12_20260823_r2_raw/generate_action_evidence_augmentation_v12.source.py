#!/usr/bin/env python3
"""V12 R2 generation-only builder with narrow per-strength topology recovery.

R1 is immutable.  This wrapper pins the complete R1 source and replaces only
the query builder so the expected ``morph changed truth topology`` rejection
drops that fixed morph attempt instead of aborting the whole one-shot run.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import tempfile
import traceback
from typing import Callable


_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[1]
_BOOTSTRAP_SOURCES = {
    "generate_action_evidence_augmentation_v12.py": (_BOOTSTRAP_ROOT / "scripts/generate_action_evidence_augmentation_v12.py", "e82064485a8ed10abcbb978746319a3a7063e78477133ce75f8bb6fe9d503733"),
    "build_cleanroom_writer_style_bank_v5.py": (_BOOTSTRAP_ROOT / "scripts/build_cleanroom_writer_style_bank_v5.py", "a28f6f399f6075d31870d6b7906d13ddc343a4d3396c47266a490425f8b3ebbf"),
    "cleanroom_writer_style_simulator_v4.py": (_BOOTSTRAP_ROOT / "scripts/cleanroom_writer_style_simulator_v4.py", "49de0704301c93f25d20b3c8c847aa5ca8ba70c561a3487c5e484c23aab89c9f"),
    "cleanroom_online_ink_augmentation_v2.py": (_BOOTSTRAP_ROOT / "scripts/cleanroom_online_ink_augmentation_v2.py", "283b8b26b0843a9d096ca8fe8e615dfad1c169204d372910db9b16808c77e0f2"),
    "cleanroom_pen_physics_v3.py": (_BOOTSTRAP_ROOT / "scripts/cleanroom_pen_physics_v3.py", "b1afcbac828c20b9f860097260e1e2e3c20a0201b057f7b27c65fb2b491b2ffd"),
    "cleanroom_trajectory_profiles_v3.py": (_BOOTSTRAP_ROOT / "scripts/cleanroom_trajectory_profiles_v3.py", "a15ffd8d61cee69091a934e581a03b0b9e713b87dccc8cae55ecab2b21d65f42"),
    "train_character_classifier_v1.py": (_BOOTSTRAP_ROOT / "scripts/train_character_classifier_v1.py", "c97bba856ad966fd7de84c688ed2591ed5568b5957d696f35e7a070a3c031835"),
    "character_tensor_v1.py": (_BOOTSTRAP_ROOT / "scripts/character_tensor_v1.py", "9bd62259bef4b872e1a57052575c7f3e316541ac23688e753c116886c022f99f"),
    "training_data_guard_v1.py": (_BOOTSTRAP_ROOT / "scripts/training_data_guard_v1.py", "42c047453bce454733b94c3fd085f0ae150259aa59e74ec05f6b4e32f28e274b"),
    "replay_evaluate_hwr_v1.py": (_BOOTSTRAP_ROOT / "scripts/replay_evaluate_hwr_v1.py", "a4f48526f428f967a7ac4266f8b61952499f1114e0106fdeef9b0ca15f5e65ee"),
    "build_normalized_ink_v1.py": (_BOOTSTRAP_ROOT / "scripts/build_normalized_ink_v1.py", "9e95fa77aa9fda0b2c37a7e9d4a8c5cf9a94068eaad69ab41e77949eb181a291"),
}


def _bootstrap_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


for _bootstrap_name, (_bootstrap_path, _bootstrap_expected) in _BOOTSTRAP_SOURCES.items():
    if _bootstrap_sha256(_bootstrap_path) != _bootstrap_expected:
        raise RuntimeError(f"pre-import generator/helper drift: {_bootstrap_name}")

r1 = importlib.import_module("generate_action_evidence_augmentation_v12")
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
R1_SOURCE = ROOT / "scripts/generate_action_evidence_augmentation_v12.py"
R1_OUTPUT = ROOT / "artifacts/action_evidence_augmentation_v12_20260823_r1_raw"
R1_MARKER = R1_OUTPUT / "GENERATION_STARTED.json"
R1_SNAPSHOT = R1_OUTPUT / "generate_action_evidence_augmentation_v12.source.py"
R1_FAILURE_AUDIT = ROOT / "reports/V12_ACTION_EVIDENCE_R1_GENERATION_FAILURE_INDEPENDENT_AUDIT.json"

DEFAULT_RAW_OUTPUT = ROOT / "artifacts/action_evidence_augmentation_v12_20260823_r2_raw"
DEFAULT_POSTPROCESSED_OUTPUT = ROOT / "artifacts/action_evidence_augmentation_v12_20260823_r2_final"

EXPECTED_R1_SOURCE_SHA256 = "e82064485a8ed10abcbb978746319a3a7063e78477133ce75f8bb6fe9d503733"
EXPECTED_R1_MARKER_SHA256 = "c414a8995c26dd3e90f800cad1fd162b0d04f41b43730cb744a403a84dd5ec8c"
EXPECTED_R1_FAILURE_AUDIT_SHA256 = "ddf43abc3513d1f7eccf0a91819978460449a2cf2777b2c98f0b4159639af8e0"
EXPECTED_R1_FAILURE_STATUS = "INDEPENDENT_V12_R1_GENERATION_FAILURE_CONFIRMED"
EXPECTED_R2_AUDIT_STATUS = "INDEPENDENT_V12_ACTION_EVIDENCE_R2_GENERATOR_STATIC_AUDIT_PASSED"
TOPOLOGY_REJECTION = "morph changed truth topology"


def _validate_r1_lineage() -> dict[str, str]:
    expected = {
        R1_SOURCE: EXPECTED_R1_SOURCE_SHA256,
        R1_SNAPSHOT: EXPECTED_R1_SOURCE_SHA256,
        R1_MARKER: EXPECTED_R1_MARKER_SHA256,
        R1_FAILURE_AUDIT: EXPECTED_R1_FAILURE_AUDIT_SHA256,
    }
    for path, wanted in expected.items():
        if r1.sha256(path) != wanted:
            raise ValueError(f"immutable R1 lineage drift: {path.name}")
    files = sorted(path.name for path in R1_OUTPUT.iterdir() if path.is_file())
    if files != ["GENERATION_STARTED.json", "generate_action_evidence_augmentation_v12.source.py"]:
        raise ValueError("R1 failed output is not immutable exact2")
    audit = json.loads(R1_FAILURE_AUDIT.read_text(encoding="utf-8"))
    if audit.get("status") != EXPECTED_R1_FAILURE_STATUS:
        raise ValueError("R1 independent failure status mismatch")
    gates = audit.get("gates", {})
    if not gates or not all(value is True for value in gates.values()):
        raise ValueError("R1 independent failure audit is not all-gates PASS")
    return {
        "r1_source_sha256": EXPECTED_R1_SOURCE_SHA256,
        "r1_started_marker_sha256": EXPECTED_R1_MARKER_SHA256,
        "r1_failure_audit_sha256": EXPECTED_R1_FAILURE_AUDIT_SHA256,
    }


def _validate_r2_audit(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    gates = payload.get("gates", {})
    if payload.get("status") != EXPECTED_R2_AUDIT_STATUS or not gates or not all(value is True for value in gates.values()):
        raise ValueError("R2 generator static audit is not all-gates PASS")
    if payload.get("generator_source_sha256") != r1.sha256(Path(__file__)):
        raise ValueError("R2 generator audit source mismatch")
    if payload.get("r1_failure_audit_sha256") != EXPECTED_R1_FAILURE_AUDIT_SHA256:
        raise ValueError("R2 generator audit R1 failure lineage mismatch")
    decision = payload.get("decision", {})
    if decision.get("generation_allowed") is not True or decision.get("training_allowed") is not False:
        raise ValueError("R2 generator audit decision mismatch")
    return {"path": str(path.resolve()), "sha256": r1.sha256(path), "status": payload["status"]}


def _morph_attempts(
    source: np.ndarray,
    relative: np.ndarray,
    counters: Counter,
    morph: Callable[[np.ndarray, np.ndarray, float], np.ndarray] = r1.morph_truth_toward_relative,
) -> tuple[list[np.ndarray], list[dict]]:
    rows: list[np.ndarray] = []
    attempts: list[dict] = []
    for morph_attempt_index, strength in enumerate(r1.MORPH_STRENGTHS):
        try:
            value = morph(source, relative, strength)
        except ValueError as error:
            if str(error) != TOPOLOGY_REJECTION:
                raise
            counters["morph_topology_attempt_drop"] += 1
            continue
        attempts.append({"morph_attempt_index": morph_attempt_index, "morph_strength": strength})
        rows.append(value)
    return rows, attempts


def _build_queries_r2(
    entries: list[dict], features: np.ndarray, labels: np.ndarray, metadata: list[dict],
    pool: dict[tuple[int, int], list[int]], model: r1.InkClassifierV1,
    homographs: tuple[frozenset[int], ...],
    calibration_fingerprints: dict[tuple[int, int], set[str]], existing_hashes: set[str],
) -> tuple[list[np.ndarray], list[dict], dict]:
    del labels
    rows: list[np.ndarray] = []
    rows_meta: list[dict] = []
    pair_admitted: dict[tuple[int, int], set[int]] = defaultdict(set)
    counters: Counter = Counter()
    all_calibration_fingerprints: dict[int, set[str]] = defaultdict(set)
    for (writer, _candidate), fingerprints in calibration_fingerprints.items():
        all_calibration_fingerprints[writer].update(fingerprints)

    for entry in entries:
        baseline = int(entry["pair"]["baseline_label_index"])
        candidate = int(entry["pair"]["candidate_label_index"])
        expected = {key: int(value) for key, value in entry["risk_stratum"].items()}
        pair = (baseline, candidate)
        for writer_value in entry["assigned_writer_ids"]:
            writer = int(writer_value)
            orientation = r1.orientation_for(pair, writer)
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
            truth_index = r1._select_parent(pool, metadata, truth, strokes, f"query-truth:{pair}:{writer}", forbidden, set())
            if truth_index is None:
                counters["missing_truth_parent"] += 1
                continue
            truth_fp = r1._parent_fingerprints(metadata[truth_index])
            forbidden |= truth_fp
            relative_index = r1._select_parent(pool, metadata, relative, strokes, f"query-relative:{pair}:{writer}", forbidden, set())
            if relative_index is None:
                counters["missing_relative_parent"] += 1
                continue
            relative_fp = r1._parent_fingerprints(metadata[relative_index])
            source = features[truth_index]
            relative_source = features[relative_index]
            attempt_rows, attempt_specs = _morph_attempts(source, relative_source, counters)
            if not attempt_rows:
                counters["all_morph_attempts_topology_rejected"] += 1
                counters["three_attempt_drop"] += 1
                continue
            action_key = f"query:{baseline}:{candidate}:{writer}:{r1.canonical_hash(expected)}"
            seed_offset = r1._seed_offset(action_key)
            generated, style_audits = r1._style_and_physics(attempt_rows, writer, seed_offset)
            logits = r1._predict(model, generated)
            accepted = None
            for physics_batch_position, (value, row_logits, attempt_spec, style_audit) in enumerate(
                zip(generated, logits, attempt_specs, style_audits, strict=True)
            ):
                actual = r1._risk(row_logits, strokes, candidate)
                valid, rms = r1._valid_tensor(value, source)
                digest = r1.tensor_hash(value)
                if not all(label in r1.ADMITTED_LABELS for label in actual["top5"]):
                    counters["generated_top5_unadmitted"] += 1
                    continue
                if r1._homograph_collision(actual["top5"], homographs):
                    counters["generated_top5_homograph_collision"] += 1
                    continue
                if not valid or digest in existing_hashes or not r1._risk_matches(actual, expected, baseline, candidate, homographs):
                    continue
                accepted = (value, {
                    "schema": r1.SCHEMA, "episode_split": "query",
                    "synthetic_writer_id": f"synthetic_writer_{writer:03d}", "global_writer_id": writer,
                    "directed_pair": {"baseline_label_index": baseline, "candidate_label_index": candidate},
                    "risk_stratum": expected, "orientation": orientation, "truth_label_index": truth,
                    "relative_label_index": relative,
                    "truth_parent_synthetic_id": metadata[truth_index]["synthetic_id"],
                    "relative_parent_synthetic_id": metadata[relative_index]["synthetic_id"],
                    "truth_parent_fingerprints": sorted(truth_fp), "relative_parent_fingerprints": sorted(relative_fp),
                    "calibration_parent_fingerprints": sorted(calibration_fingerprints[(writer, candidate)]),
                    "all_writer_calibration_parent_fingerprints_sha256": r1.canonical_hash(sorted(all_calibration_fingerprints[writer])),
                    "truth_source_index": truth_index, "relative_source_index": relative_index,
                    "truth_topology": metadata[truth_index]["topology"], "relative_topology": metadata[relative_index]["topology"],
                    "attempt": attempt_spec["morph_attempt_index"],
                    "morph_attempt_index": attempt_spec["morph_attempt_index"],
                    "morph_strength": attempt_spec["morph_strength"],
                    "physics_seed": r1.WRITER_SEED_ROOT + writer * r1.WRITER_SEED_MULTIPLIER + seed_offset,
                    "physics_batch_position": physics_batch_position,
                    "writer_latent": asdict(r1._latents()[writer]), "style_audit": style_audit,
                    "spatial_rms_from_truth_parent": rms, "tensor_sha256": digest,
                    "frozen_top5": actual["top5"], "frozen_top1": actual["top5"][0],
                    "candidate_action_inside_frozen_top5": True, "external_approved_parents": True, "project_rows": 0,
                })
                break
            if accepted is None:
                counters["three_attempt_drop"] += 1
                continue
            value, row = accepted
            existing_hashes.add(row["tensor_sha256"])
            rows.append(value)
            rows_meta.append(row)
            pair_admitted[pair].add(writer)
            counters[orientation] += 1

    failed_pairs = sorted([
        {"baseline_label_index": pair[0], "candidate_label_index": pair[1],
         "unique_admitted_writers": len(writers), "policy": "identity_only"}
        for pair, writers in pair_admitted.items() if len(writers) < r1.MIN_ADMITTED_WRITERS_PER_PAIR
    ], key=lambda row: (row["baseline_label_index"], row["candidate_label_index"]))
    all_pairs = {(int(e["pair"]["baseline_label_index"]), int(e["pair"]["candidate_label_index"])) for e in entries}
    for pair in sorted(all_pairs - set(pair_admitted)):
        failed_pairs.append({"baseline_label_index": pair[0], "candidate_label_index": pair[1],
                             "unique_admitted_writers": 0, "policy": "identity_only"})
    failed_set = {(row["baseline_label_index"], row["candidate_label_index"]) for row in failed_pairs}
    keep = [index for index, row in enumerate(rows_meta)
            if (row["directed_pair"]["baseline_label_index"], row["directed_pair"]["candidate_label_index"]) not in failed_set]
    return [rows[i] for i in keep], [rows_meta[i] for i in keep], {
        "counters": dict(counters), "failed_pairs": failed_pairs,
        "recovery_contract": {"exact_rejection": TOPOLOGY_REJECTION, "broad_exception_catch": False,
                              "morph_attempt_index_separate_from_physics_batch_position": True},
    }


def _atomic_exclusive_json(path: Path, payload: dict) -> None:
    """Publish JSON once; neither an existing final nor stale temp is replaced."""

    temporary = path.with_name(path.name + ".tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"immutable receipt path already exists: {path.name}")
    encoded = json.dumps(payload, indent=2).encode("utf-8")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        # Preserve the exclusive temp as crash evidence; retry remains fail-closed.
        raise
    os.rename(temporary, path)  # On Windows this fails instead of replacing an existing final.


def _write_immutable_failure(output: Path, independent_audit: Path, error: BaseException) -> None:
    marker = output / "GENERATION_STARTED.json"
    if not marker.exists():
        return
    failure = {
        "status": "V12_ACTION_EVIDENCE_R2_GENERATION_FAILED_IMMUTABLE",
        "failed_at": datetime.now(timezone.utc).isoformat(),
        "exception_type": type(error).__name__, "exception_message": str(error),
        "traceback_sha256": r1.hashlib.sha256(traceback.format_exc().encode()).hexdigest(),
        "source_sha256": r1.sha256(Path(__file__)), "started_marker_sha256": r1.sha256(marker),
        "independent_audit": {"path": str(independent_audit.resolve()), "sha256": r1.sha256(independent_audit)},
        "r1_lineage": _validate_r1_lineage(), "retry_allowed": False,
        "training_performed": False, "postprocessing_performed": False,
        "writers096_127_opened": False, "legacy_real_eval_product_opened": False,
    }
    _atomic_exclusive_json(output / "GENERATION_FAILED.json", failure)


def generate(output: Path, postprocessed: Path, independent_audit: Path) -> None:
    lineage = _validate_r1_lineage()
    original_file = r1.__file__
    original_build_queries = r1._build_queries
    original_validate_audit = r1._validate_generator_audit
    original_validate_helpers = r1._validate_helper_sources

    def validate_helpers_r2() -> dict[str, str]:
        hashes = original_validate_helpers()
        if r1.sha256(R1_SOURCE) != EXPECTED_R1_SOURCE_SHA256:
            raise ValueError("R1 generator helper drift")
        hashes["generate_action_evidence_augmentation_v12.py"] = EXPECTED_R1_SOURCE_SHA256
        return hashes

    r1.__file__ = __file__
    r1._build_queries = _build_queries_r2
    r1._validate_generator_audit = _validate_r2_audit
    r1._validate_helper_sources = validate_helpers_r2
    try:
        r1.generate(output, postprocessed, independent_audit)
        report_path = output / "generation_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["r2_lineage"] = lineage
        report["boundaries"]["r1_failed_run_immutable"] = True
        report["boundaries"]["r2_only_change"] = "narrow per-strength topology rejection recovery"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    except BaseException as error:
        if output.exists():
            _write_immutable_failure(output, independent_audit, error)
        raise
    finally:
        r1.__file__ = original_file
        r1._build_queries = original_build_queries
        r1._validate_generator_audit = original_validate_audit
        r1._validate_helper_sources = original_validate_helpers


def dry_toy() -> dict:
    truth = np.zeros((128, 5), np.float32)
    relative = np.zeros((128, 5), np.float32)
    truth[:, 0] = np.linspace(0.2, 0.8, 128); truth[:, 1] = 0.35
    relative[:, 0] = np.linspace(0.25, 0.75, 128); relative[:, 1] = 0.65
    for value in (truth, relative):
        value[0, 3] = 1.0; value[:, 4] = 1.0; value[1:, 2] = 1.0 / 127.0
    calls = {"count": 0}

    def first_rejected(source: np.ndarray, target: np.ndarray, strength: float) -> np.ndarray:
        calls["count"] += 1
        if calls["count"] == 1:
            raise ValueError(TOPOLOGY_REJECTION)
        return r1.morph_truth_toward_relative(source, target, strength)

    counters: Counter = Counter()
    rows, attempts = _morph_attempts(truth, relative, counters, first_rejected)
    try:
        _morph_attempts(truth, relative, Counter(), lambda *_: (_ for _ in ()).throw(ValueError("unrelated error")))
    except ValueError as error:
        unrelated_propagated = str(error) == "unrelated error"
    else:
        unrelated_propagated = False
    with tempfile.TemporaryDirectory(prefix="v12-r2-atomic-toy-") as directory:
        receipt = Path(directory) / "GENERATION_FAILED.json"
        toy_payload = {"status": "TOY_FAILURE", "retry_allowed": False}
        _atomic_exclusive_json(receipt, toy_payload)
        atomic_exact = json.loads(receipt.read_text(encoding="utf-8")) == toy_payload
        try:
            _atomic_exclusive_json(receipt, {"status": "MUST_NOT_REPLACE"})
        except FileExistsError:
            existing_final_not_replaced = json.loads(receipt.read_text(encoding="utf-8")) == toy_payload
        else:
            existing_final_not_replaced = False
        temp_absent_after_publish = not receipt.with_name(receipt.name + ".tmp").exists()
    result = {
        "status": "V12_R2_GENERATOR_DRY_TOY_PASSED", "external_tensor_loaded": False,
        "checkpoint_loaded": False, "catalog_loaded": False,
        "fixed_strengths_unchanged": tuple(item["morph_strength"] for item in attempts) == r1.MORPH_STRENGTHS[1:],
        "topology_drop_count": counters["morph_topology_attempt_drop"],
        "surviving_attempt_indices": [item["morph_attempt_index"] for item in attempts],
        "surviving_rows": len(rows), "unrelated_value_error_propagated": unrelated_propagated,
        "morph_index_physics_position_example": {"morph_attempt_index": attempts[0]["morph_attempt_index"], "physics_batch_position": 0},
        "r1_source_and_helpers_preverified_before_import": True,
        "r1_artifact_and_failure_audit_loaded_in_toy": False,
        "atomic_failure_json_exact": atomic_exact,
        "existing_failure_json_not_replaced": existing_final_not_replaced,
        "failure_temp_absent_after_publish": temp_absent_after_publish,
    }
    if (result["topology_drop_count"] != 1 or result["surviving_attempt_indices"] != [1, 2]
            or not unrelated_propagated or not atomic_exact or not existing_final_not_replaced
            or not temp_absent_after_publish):
        raise AssertionError("R2 narrow recovery contract failed")
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
