"""Evaluate the frozen global prototype scorer as a calibration-only Legacy policy.

Evaluation truth is passed only to scoring after predictions are fixed.  The
known replay, fresh acceptance, CROHME, and MathWriting are never loaded.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import inspect
import json
from pathlib import Path

import numpy as np

from build_cleanroom_writer_style_bank_v5 import sha256
from evaluate_writer_adaptation_v8 import (
    DEFAULT_CHECKPOINT,
    _aggregate,
    _candidate_masked,
    _frozen_outputs,
    _load_v7,
    _score,
    _writer_regressions,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCORER = ROOT / "artifacts/global_prototype_shape_scorer_v9_20260823_r3_frozen"
DEFAULT_V7 = ROOT / "artifacts/hierarchical_writer_model_v7_20260823_r1"
DEFAULT_SPLIT = ROOT / "artifacts/writer_adaptation_v8_20260823_r1_shadow/split_manifest.json"
DEFAULT_OUTPUT = ROOT / "artifacts/global_prototype_legacy_policy_v9_20260823_r2"
DEFAULT_CACHE = ROOT / "artifacts/global_prototype_legacy_policy_v9_20260823_r2_cache.npz"
RESERVE_ROOT = ROOT / "artifacts/global_prototype_shape_scorer_v9_reserve_outer_20260823_r4"
RESERVE_CACHE = ROOT / "artifacts/writer_adaptation_v9_global_prototype_reserve_cache_20260823_r4.npz"
RESERVE_OPENER = ROOT / "scripts/evaluate_global_prototype_reserve_v9.py"
EXPECTED_WRITERS = ("writer_001", "writer_002", "writer_003", "writer_004", "writer_006", "writer_007", "writer_008")
EXPECTED_SPLIT_COUNTS = {
    "raw_catalog_rows": 393, "supported_rows": 392,
    "calibration_formulae": 23, "evaluation_formulae": 73,
    "calibration_glyphs": 92, "evaluation_glyphs": 300,
}


def _verify_external_gates(independent_audit_path: Path, reserve_decision_path: Path) -> dict:
    audit = json.loads(independent_audit_path.read_text(encoding="utf-8"))
    positive = (
        "live_generator_equals_snapshot", "old_fit20_dev4_disjoint",
        "alpha_selected_before_consumed_diagnostic", "writer_residual_disabled",
        "any_unsupported_top5_identity", "candidate_set_exact",
        "independent_metrics_match", "independent_reproduction_exact",
    )
    closed = ("reserve_loaded", "legacy_loaded", "known_replay_loaded", "crohme_loaded", "mathwriting_loaded")
    if audit.get("status") != "INDEPENDENT_GLOBAL_PROTOTYPE_SCORER_AUDIT_PASSED" or not all(audit.get("gates", {}).get(key) is True for key in positive) or not all(audit.get("gates", {}).get(key) is False for key in closed):
        raise ValueError("independent scorer audit gate failed")
    for key, path in {
        "artifact_report": DEFAULT_SCORER / "development_report.json",
        "artifact_receipt": DEFAULT_SCORER / "freeze_receipt.json",
        "prototype": DEFAULT_SCORER / "global_class_prototypes.npz",
        "config": DEFAULT_SCORER / "frozen_config.json",
        "source_snapshot": DEFAULT_SCORER / "freeze_global_prototype_shape_scorer_v9.source.py",
    }.items():
        if audit.get("hashes", {}).get(key) != sha256(path):
            raise ValueError(f"independent scorer audit live hash mismatch: {key}")

    decision = json.loads(reserve_decision_path.read_text(encoding="utf-8"))
    required = (
        "reserve_outer_pass", "all_writer_top1_pseudo_exact_nonregression",
        "aggregate_top1_positive", "aggregate_pseudo_exact_nonregression",
        "common_class_nonregression", "extension_only_identity", "candidate_set_exact",
    )
    if decision.get("status") != "INDEPENDENT_GLOBAL_PROTOTYPE_RESERVE_AUDIT_PASSED" or not all(decision.get("gates", {}).get(key) is True for key in required):
        raise ValueError("independent reserve decision gate failed")
    reserve_report = RESERVE_ROOT / "outer_report.json"
    prepare_receipt = RESERVE_ROOT / "reserve_open_receipt.json"
    trace = RESERVE_ROOT / "changed_prediction_trace.jsonl"
    snapshot = RESERVE_ROOT / "evaluate_global_prototype_reserve_v9.source.py"
    for key, path in {
        "reserve_report_sha256": reserve_report,
        "prepare_receipt_sha256": prepare_receipt,
        "cache_sha256": RESERVE_CACHE,
        "trace_sha256": trace,
        "opener_script_sha256": RESERVE_OPENER,
        "opener_source_snapshot_sha256": snapshot,
    }.items():
        if decision.get("hashes", {}).get(key) != sha256(path):
            raise ValueError(f"independent reserve decision live hash mismatch: {key}")
    report = json.loads(reserve_report.read_text(encoding="utf-8"))
    report_required = (
        "all_writer_top1_pseudo_exact_nonregression", "aggregate_top1_positive",
        "common_class_nonregression", "extension_only_identity", "candidate_set_exact",
    )
    if report.get("status") != "SYNTHETIC_RESERVE_OUTER_PASS" or not all(report.get("gates", {}).get(key) is True for key in report_required):
        raise ValueError("live reserve outer report is not a pass")
    if sha256(RESERVE_OPENER) != sha256(snapshot):
        raise ValueError("reserve opener live/source snapshot drift")
    return {"independent_scorer_audit_sha256": sha256(independent_audit_path), "independent_reserve_decision_sha256": sha256(reserve_decision_path)}


def _legacy_receipt(args: argparse.Namespace, external_hashes: dict) -> dict:
    return {
        "schema": "aiflow-global-prototype-legacy-one-shot/v9",
        "status": "PREPARED_LEGACY_UNOPENED",
        "scorer": str(args.scorer.resolve()), "v7": str(args.v7.resolve()),
        "split": str(args.split.resolve()), "checkpoint": str(args.checkpoint.resolve()),
        **external_hashes,
        "scorer_freeze_receipt_sha256": sha256(args.scorer / "freeze_receipt.json"),
        "scorer_report_sha256": sha256(args.scorer / "development_report.json"),
        "scorer_prototype_sha256": sha256(args.scorer / "global_class_prototypes.npz"),
        "scorer_config_sha256": sha256(args.scorer / "frozen_config.json"),
        "v7_catalog_sha256": sha256(args.v7 / "raw_legacy_writer_catalog.npz"),
        "v7_metadata_sha256": sha256(args.v7 / "raw_legacy_writer_catalog.metadata.jsonl"),
        "v7_report_sha256": sha256(args.v7 / "report.json"),
        "source_split_sha256": sha256(args.split), "checkpoint_sha256": sha256(args.checkpoint),
        "reserve_report_sha256": sha256(RESERVE_ROOT / "outer_report.json"),
        "reserve_prepare_receipt_sha256": sha256(RESERVE_ROOT / "reserve_open_receipt.json"),
        "reserve_cache_sha256": sha256(RESERVE_CACHE),
        "reserve_trace_sha256": sha256(RESERVE_ROOT / "changed_prediction_trace.jsonl"),
        "reserve_opener_script_sha256": sha256(RESERVE_OPENER),
        "reserve_opener_source_snapshot_sha256": sha256(RESERVE_ROOT / "evaluate_global_prototype_reserve_v9.source.py"),
        "legacy_cache_path": str(args.cache.resolve()),
        "expected_writer_ids": list(EXPECTED_WRITERS),
        "expected_split_counts": EXPECTED_SPLIT_COUNTS,
        "boundary": {
            "legacy_features_metadata_labels_logits_predictions_opened": False,
            "legacy_split_content_opened": False,
            "known_replay_opened": False, "fresh_acceptance_opened": False,
            "crohme_opened": False, "mathwriting_opened": False,
            "product_promotion": False,
        },
    }


def _verify_scorer(root: Path) -> tuple[dict, dict[str, np.ndarray]]:
    receipt = json.loads((root / "freeze_receipt.json").read_text(encoding="utf-8"))
    report = json.loads((root / "development_report.json").read_text(encoding="utf-8"))
    config = json.loads((root / "frozen_config.json").read_text(encoding="utf-8"))
    checks = {
        "development_report_sha256": root / "development_report.json",
        "prototype_sha256": root / "global_class_prototypes.npz",
        "config_sha256": root / "frozen_config.json",
        "generator_source_snapshot_sha256": root / "freeze_global_prototype_shape_scorer_v9.source.py",
    }
    for key, path in checks.items():
        if receipt.get(key) != sha256(path):
            raise ValueError(f"frozen scorer receipt mismatch: {key}")
    if receipt.get("status") != "GLOBAL_PROTOTYPE_SCORER_FROZEN_RESERVE_UNOPENED":
        raise ValueError("frozen scorer status is not admissible")
    if report.get("status") != "GLOBAL_PROTOTYPE_DEVELOPMENT_PASS_RESERVE_CLOSED":
        raise ValueError("frozen scorer development gate did not pass")
    required = (
        "old_dev_nonregression_positive", "consumed_r1_all_writer_nonregression",
        "common_all_writer_nonregression", "extension_only_full_identity", "candidate_set_exact",
    )
    if not all(report["gates"].get(key) is True for key in required):
        raise ValueError("frozen scorer required gates are incomplete")
    if report["gates"].get("reserve_opened") is not False or config.get("writer_residual_enabled") is not False:
        raise ValueError("reserve boundary or disabled residual contract changed")
    with np.load(root / "global_class_prototypes.npz", allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name]).copy() for name in ("prototypes", "counts", "reliability", "admitted")}
    return config, arrays


def _label_blind_api_toy() -> dict:
    parameters = tuple(inspect.signature(_apply_without_truth).parameters)
    if parameters != ("embeddings", "logits", "prototypes", "admitted", "alpha"):
        raise AssertionError(f"prediction API accepts unexpected inputs: {parameters}")
    embeddings = np.asarray([[1.0, 0.0]], np.float32)
    logits = np.asarray([[5.0, 4.0, 3.0, 2.0, 1.0, -1.0]], np.float32)
    prototypes = np.zeros((6, 2), np.float32); prototypes[1] = embeddings[0]
    admitted = np.ones(6, bool)
    adapted, gate, _trace = _apply_without_truth(embeddings, logits, prototypes, admitted, 16.0)
    old = set(np.argsort(logits[0])[-5:].tolist()); new = set(np.argsort(adapted[0])[-5:].tolist())
    if old != new or gate["candidate_set_violations"] != 0:
        raise AssertionError("label-blind candidate-set toy failed")
    return {"parameters": list(parameters), "candidate_set_exact": True, "labels_or_metadata_inputs": False}


def _apply_without_truth(
    embeddings: np.ndarray, logits: np.ndarray, prototypes: np.ndarray,
    admitted: np.ndarray, alpha: float,
) -> tuple[np.ndarray, dict, list[dict]]:
    """Produce predictions without accepting labels or sample metadata."""
    norm = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-8)
    top5 = np.argsort(logits, axis=1)[:, -5:][:, ::-1]
    adapted = _candidate_masked(logits, top5)
    traces = []
    identity_unsupported = 0
    for row in range(len(logits)):
        candidates = top5[row]
        if not np.all(admitted[candidates]):
            identity_unsupported += 1
            continue
        values = logits[row, candidates].astype(np.float64)
        z = (values - values.mean()) / max(float(values.std()), 1e-6)
        cosine = norm[row] @ prototypes[candidates].T
        scores = z + alpha * cosine
        adapted[row, candidates] = scores.astype(adapted.dtype)
        selected = int(candidates[int(np.argmax(scores))])
        baseline = int(candidates[0])
        if selected != baseline:
            traces.append({"local_row": row, "baseline_top5": candidates.astype(int).tolist(), "baseline": baseline, "selected": selected})
    adapted_top5 = np.argsort(adapted, axis=1)[:, -5:][:, ::-1]
    violations = int(sum(set(adapted_top5[row].tolist()) != set(top5[row].tolist()) for row in range(len(top5))))
    return adapted, {"rows": len(logits), "identity_unsupported_top5": identity_unsupported, "candidate_set_violations": violations}, traces


def _paired(truth: np.ndarray, baseline: np.ndarray, adapted: np.ndarray) -> dict:
    old = np.argmax(baseline, axis=1)
    new = np.argmax(adapted, axis=1)
    return {
        "changed": int(np.sum(old != new)),
        "improved": int(np.sum((old != truth) & (new == truth))),
        "regressed": int(np.sum((old == truth) & (new != truth))),
        "wrong_to_wrong": int(np.sum((old != truth) & (new != truth) & (old != new))),
    }


def _candidate_set_mismatches(baseline: np.ndarray, adapted: np.ndarray) -> int:
    old = np.argsort(baseline, axis=1)[:, -5:]
    new = np.argsort(adapted, axis=1)[:, -5:]
    return int(sum(set(old[row].tolist()) != set(new[row].tolist()) for row in range(len(old))))


def _indices(metadata: list[dict], writer: str, sample_ids: set[str]) -> list[int]:
    return [
        index for index, row in enumerate(metadata)
        if row["writer_id"] == writer and row["sample_id"] in sample_ids and int(row["label_index"]) >= 0
    ]


def _validate_split_contract(metadata: list[dict], legacy_split: dict) -> dict:
    if tuple(sorted(legacy_split)) != tuple(sorted(EXPECTED_WRITERS)):
        raise ValueError("Legacy writer set changed")
    supported = [row for row in metadata if int(row["label_index"]) >= 0]
    if len(metadata) != EXPECTED_SPLIT_COUNTS["raw_catalog_rows"] or len(supported) != EXPECTED_SPLIT_COUNTS["supported_rows"]:
        raise ValueError("Legacy raw/supported row count changed")
    formula_counts = {"calibration_formulae": 0, "evaluation_formulae": 0}
    glyph_counts = {"calibration_glyphs": 0, "evaluation_glyphs": 0}
    per_writer = {}
    assigned = set()
    for writer in EXPECTED_WRITERS:
        calibration_ids = set(legacy_split[writer]["calibration_formulae"])
        evaluation_ids = set(legacy_split[writer]["evaluation_formulae"])
        if calibration_ids & evaluation_ids:
            raise ValueError(f"formula overlap for {writer}")
        calibration = _indices(metadata, writer, calibration_ids); evaluation = _indices(metadata, writer, evaluation_ids)
        if set(calibration) & set(evaluation) or assigned & (set(calibration) | set(evaluation)):
            raise ValueError("row overlap across Legacy split")
        assigned.update(calibration); assigned.update(evaluation)
        formula_counts["calibration_formulae"] += len(calibration_ids); formula_counts["evaluation_formulae"] += len(evaluation_ids)
        glyph_counts["calibration_glyphs"] += len(calibration); glyph_counts["evaluation_glyphs"] += len(evaluation)
        per_writer[writer] = {"calibration_formulae": len(calibration_ids), "evaluation_formulae": len(evaluation_ids), "calibration_glyphs": len(calibration), "evaluation_glyphs": len(evaluation)}
    actual = {"raw_catalog_rows": len(metadata), "supported_rows": len(supported), **formula_counts, **glyph_counts}
    if actual != EXPECTED_SPLIT_COUNTS or len(assigned) != len(supported):
        raise ValueError(f"Legacy split count/coverage changed: {actual}")
    return {"writer_ids_exact": True, "formula_disjoint": True, "supported_rows_assigned_exactly_once": True, "counts": actual, "per_writer": per_writer}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scorer", type=Path, default=DEFAULT_SCORER)
    parser.add_argument("--v7", type=Path, default=DEFAULT_V7)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--independent-audit", type=Path, required=True)
    parser.add_argument("--reserve-decision", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()
    if args.prepare == args.open:
        parser.error("choose exactly one of --prepare or --open")

    external_gate_hashes = _verify_external_gates(args.independent_audit, args.reserve_decision)
    scorer_config, scorer = _verify_scorer(args.scorer)
    expected_receipt = _legacy_receipt(args, external_gate_hashes)
    receipt_path = args.output / "legacy_open_receipt.json"
    source_snapshot = args.output / "evaluate_global_prototype_legacy_policy_v9.source.py"
    opening_marker = args.output / "legacy_open_started.json"
    if args.prepare:
        if args.output.exists() or args.cache.exists():
            parser.error(f"refusing to overwrite output: {args.output}")
        expected_receipt["label_blind_api_toy"] = _label_blind_api_toy()
        args.output.mkdir(parents=True)
        source_snapshot.write_bytes(Path(__file__).read_bytes())
        expected_receipt["evaluator_script_sha256"] = sha256(Path(__file__))
        expected_receipt["evaluator_source_snapshot_sha256"] = sha256(source_snapshot)
        receipt_path.write_text(json.dumps(expected_receipt, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"output": str(args.output), "status": expected_receipt["status"]}, ensure_ascii=False))
        return 0
    forbidden = ("legacy_open_started.json", "report.json", "calibration_policy.json", "changed_prediction_trace.jsonl", "split_manifest.json", "decision_receipt.json")
    prepared_files = {path.name for path in args.output.iterdir() if path.is_file()} if args.output.exists() else set()
    if not receipt_path.exists() or args.cache.exists() or any((args.output / name).exists() for name in forbidden) or prepared_files != {"legacy_open_receipt.json", "evaluate_global_prototype_legacy_policy_v9.source.py"}:
        raise ValueError("legacy prepare missing or one-shot already opened/interrupted")
    expected_receipt["label_blind_api_toy"] = _label_blind_api_toy()
    expected_receipt["evaluator_script_sha256"] = sha256(Path(__file__))
    expected_receipt["evaluator_source_snapshot_sha256"] = sha256(source_snapshot)
    actual_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if actual_receipt != expected_receipt or expected_receipt["evaluator_script_sha256"] != expected_receipt["evaluator_source_snapshot_sha256"]:
        raise ValueError("legacy prepared receipt/source/live input drift")
    opening_marker.write_text(json.dumps({"status": "LEGACY_ONE_SHOT_OPEN_STARTED", "prepare_receipt_sha256": sha256(receipt_path), "started_at": datetime.now(timezone.utc).isoformat()}, indent=2), encoding="utf-8")
    source_split = json.loads(args.split.read_text(encoding="utf-8"))
    legacy_split = source_split["legacy"]
    features, metadata = _load_v7(args.v7)
    split_audit = _validate_split_contract(metadata, legacy_split)
    labels, embeddings, logits = _frozen_outputs(features, args.checkpoint)
    np.savez_compressed(args.cache, embeddings=embeddings.astype(np.float32), logits=logits.astype(np.float32), checkpoint_sha256=np.asarray(sha256(args.checkpoint)), catalog_sha256=np.asarray(sha256(args.v7 / "raw_legacy_writer_catalog.npz")), metadata_sha256=np.asarray(sha256(args.v7 / "raw_legacy_writer_catalog.metadata.jsonl")))
    alpha = float(scorer_config["alpha"])

    per_writer = []
    policy = {}
    changed_trace = []
    all_eval_indices, all_truth, all_baseline, all_adapted = [], [], [], []
    for writer, split in sorted(legacy_split.items()):
        calibration = _indices(metadata, writer, set(split["calibration_formulae"]))
        evaluation = _indices(metadata, writer, set(split["evaluation_formulae"]))
        if not calibration or not evaluation or set(split["calibration_formulae"]) & set(split["evaluation_formulae"]):
            raise ValueError(f"invalid formula-disjoint split for {writer}")

        calibration_adapted, calibration_apply, _ = _apply_without_truth(
            embeddings[calibration], logits[calibration], scorer["prototypes"], scorer["admitted"], alpha,
        )
        calibration_truth = np.asarray([metadata[index]["label_index"] for index in calibration], np.int64)
        calibration_score = _score(metadata, calibration, calibration_truth, logits[calibration], calibration_adapted)
        calibration_score["candidate_set_mismatches"] = _candidate_set_mismatches(logits[calibration], calibration_adapted)
        calibration_paired = _paired(calibration_truth, logits[calibration], calibration_adapted)
        enabled = (
            calibration_apply["candidate_set_violations"] == 0
            and calibration_score["candidate_set_mismatches"] == 0
            and calibration_paired["improved"] > 0
            and calibration_paired["regressed"] == 0
            and calibration_score["adapted"]["top1"] >= calibration_score["baseline"]["top1"]
            and calibration_score["adapted"]["formula_exact"] >= calibration_score["baseline"]["formula_exact"]
        )
        policy[writer] = {
            "mode": "global_prototype" if enabled else "identity",
            "selection_source": "calibration labels and formula IDs only",
            "calibration_glyphs": len(calibration),
            "calibration_formulae": len(split["calibration_formulae"]),
            "calibration": calibration_score,
            "paired": calibration_paired,
            "apply_gate": calibration_apply,
        }

        proposed, apply_gate, raw_trace = _apply_without_truth(
            embeddings[evaluation], logits[evaluation], scorer["prototypes"], scorer["admitted"], alpha,
        )
        adapted = proposed if enabled else _candidate_masked(logits[evaluation], np.argsort(logits[evaluation], axis=1)[:, -5:][:, ::-1])
        eval_truth = np.asarray([metadata[index]["label_index"] for index in evaluation], np.int64)
        result = _score(metadata, evaluation, eval_truth, logits[evaluation], adapted)
        result["candidate_set_mismatches"] = _candidate_set_mismatches(logits[evaluation], adapted)
        result.update({
            "writer_id": writer, "policy_mode": policy[writer]["mode"],
            "calibration_glyphs": len(calibration), "evaluation_glyphs": len(evaluation),
            "paired": _paired(eval_truth, logits[evaluation], adapted), "apply_gate": apply_gate,
        })
        per_writer.append(result)
        if enabled:
            for trace in raw_trace:
                local = int(trace.pop("local_row")); index = evaluation[local]
                truth = int(metadata[index]["label_index"])
                changed_trace.append({
                    "writer_id": writer, "sample_id": metadata[index]["sample_id"],
                    "canonical_record_id": metadata[index]["canonical_record_id"], "token": metadata[index]["token"],
                    "truth_index": truth, "truth_token": labels[truth],
                    "baseline_token": labels[trace["baseline"]], "selected_token": labels[trace["selected"]], **trace,
                })
        all_eval_indices.extend(evaluation)
        all_truth.append(eval_truth); all_baseline.append(logits[evaluation]); all_adapted.append(adapted)

    aggregate = _aggregate(per_writer)
    regressions = _writer_regressions(per_writer)
    truth = np.concatenate(all_truth); baseline = np.concatenate(all_baseline); adapted = np.concatenate(all_adapted)
    overall = _score(metadata, all_eval_indices, truth, baseline, adapted)
    overall["candidate_set_mismatches"] = _candidate_set_mismatches(baseline, adapted)
    paired = _paired(truth, baseline, adapted)
    positive = aggregate["adapted"]["top1"] > aggregate["baseline"]["top1"] or aggregate["adapted"]["formula_exact"] > aggregate["baseline"]["formula_exact"]
    apply_gate_violations = sum(int(row["apply_gate"]["candidate_set_violations"]) for row in per_writer)
    direct_candidate_set_mismatches = sum(int(row["candidate_set_mismatches"]) for row in per_writer)
    passed = not regressions and positive and aggregate["candidate_violations"] == 0 and apply_gate_violations == 0 and direct_candidate_set_mismatches == 0

    split_manifest = {
        "schema": "aiflow-global-prototype-legacy-policy-split/v9",
        "source_split_sha256": sha256(args.split), "legacy": legacy_split,
        "known_replay_loaded": False, "fresh_acceptance_loaded": False,
    }
    split_path = args.output / "split_manifest.json"
    split_path.write_text(json.dumps(split_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    policy_path = args.output / "calibration_policy.json"
    policy_path.write_text(json.dumps({
        "alpha": alpha, "writer_residual_enabled": False,
        "activation": "calibration paired improved > 0, regressed = 0, Top1/formula exact nonregression",
        "unsupported": scorer_config["unsupported_contract"], "per_writer": policy,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    trace_path = args.output / "changed_prediction_trace.jsonl"
    with trace_path.open("w", encoding="utf-8") as stream:
        for row in changed_trace:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "schema": "aiflow-global-prototype-legacy-calibration-policy/v9",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "LEGACY_DEVELOPMENT_GATE_PASS_ACCEPTANCE_CLOSED" if passed else "PRE_ACCEPTANCE_FAIL_CLOSED",
        "boundary": "Legacy is known development evidence, not promotion; actual formula exact uses sample_id groups",
        "selection": "writer policy selected only from calibration formulae; evaluation labels passed only to scoring",
        "aggregate": aggregate, "paired": paired, "per_writer": per_writer, "split_audit": split_audit,
        "candidate_contract": {"direct_top5_set_mismatches": direct_candidate_set_mismatches, "apply_gate_violations": apply_gate_violations},
        "overall_per_class": overall["per_class"], "overall_confusion": overall["confusion"],
        "writer_regressions": regressions,
        "gates": {
            "formula_disjoint": bool(split_audit["writer_ids_exact"] and split_audit["formula_disjoint"] and split_audit["supported_rows_assigned_exactly_once"]), "evaluation_labels_scoring_only": True,
            "all_writer_top1_formula_exact_nonregression": not regressions,
            "aggregate_positive": positive, "candidate_set_exact": direct_candidate_set_mismatches == 0 and apply_gate_violations == 0 and aggregate["candidate_violations"] == 0,
            "baseline_top5_unchanged": direct_candidate_set_mismatches == 0 and apply_gate_violations == 0,
            "known_replay_opened": False, "fresh_acceptance_opened": False,
            "crohme_opened": False, "mathwriting_opened": False,
            "checkpoint_changed": False, "runtime_changed": False, "hwr_changed": False,
            "product_promotion": False,
        },
        "inputs": {
            **external_gate_hashes,
            "scorer_freeze_receipt_sha256": sha256(args.scorer / "freeze_receipt.json"),
            "v7_catalog_sha256": sha256(args.v7 / "raw_legacy_writer_catalog.npz"),
            "v7_metadata_sha256": sha256(args.v7 / "raw_legacy_writer_catalog.metadata.jsonl"),
            "checkpoint_sha256": sha256(args.checkpoint), "source_split_sha256": sha256(args.split),
        },
        "outputs": {
            "split_manifest_sha256": sha256(split_path), "policy_sha256": sha256(policy_path),
            "trace_sha256": sha256(trace_path), "script_sha256": sha256(Path(__file__)),
            "source_snapshot_sha256": sha256(source_snapshot),
            "legacy_open_receipt_sha256": sha256(receipt_path),
            "legacy_open_marker_sha256": sha256(opening_marker), "legacy_cache_sha256": sha256(args.cache),
        },
    }
    report_path = args.output / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "decision_receipt.json").write_text(json.dumps({
        "status": report["status"], "report_sha256": sha256(report_path), **report["inputs"], **report["outputs"],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "status": report["status"], "aggregate": aggregate, "active_writers": [writer for writer, row in policy.items() if row["mode"] != "identity"]}, ensure_ascii=False))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
