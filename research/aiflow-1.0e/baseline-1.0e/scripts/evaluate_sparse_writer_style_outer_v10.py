#!/usr/bin/env python3
"""Prepare, then one-shot evaluate frozen v10 r3 on writers 064..095."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCORER = ROOT / "artifacts/sparse_writer_style_scorer_v10_20260823_r3_frozen"
BANK = ROOT / "artifacts/cleanroom_writer_style_v5_20260823_writers064_095_r4_r1"
ADMISSION = ROOT / "reports/V10_WRITER_STYLE_064_095_ADMISSION_RECEIPT.json"
R3_AUDIT = ROOT / "reports/V10_SPARSE_WRITER_STYLE_R3_ARTIFACT_INDEPENDENT_AUDIT.json"
FULL_AUDIT = ROOT / "reports/V10_WRITER_STYLE_064_095_FULL_INDEPENDENT_AUDIT.json"
CROSS_AUDIT = ROOT / "reports/V10_WRITER_STYLE_064_095_CROSS_BANK_AUDIT.json"
CHECKPOINT = ROOT / "artifacts/commercial_hwr_cleanroom_physics_20260823_r1_shadow/commercial_hwr_cleanroom_physics_checkpoint.pt"
DEFAULT_OUTPUT = ROOT / "artifacts/sparse_writer_style_scorer_v10_writers064_095_outer_r3"
DEFAULT_CACHE = ROOT / "artifacts/sparse_writer_style_scorer_v10_writers064_095_outer_r3_cache.npz"

EXPECTED_ADMISSION_SHA256 = "acba95f9b789e192bec4f6857a557206e608f2e208efb564ffc4f939c178b6be"
EXPECTED_R3_AUDIT_SHA256 = "e0360a7918fb3785be87c5cc0dc338f79199381efab107190c39e4c12afe0182"
EXPECTED_PREOPEN_AUDIT_STATUS = "INDEPENDENT_V10_WRITERS064_095_OUTER_PREOPEN_AUDIT_PASSED"
SUPPORT_SIZES = (4, 6, 10, 20, 22, 24)
EXPECTED_GLOBAL_WRITERS = tuple(range(64, 96))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _all_true(values: object) -> bool:
    return isinstance(values, dict) and bool(values) and all(value is True for value in values.values())


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_semantic_inputs() -> dict:
    if sha256(ADMISSION) != EXPECTED_ADMISSION_SHA256:
        raise ValueError("new style admission receipt hash drift")
    admission = _read_json(ADMISSION)
    if admission.get("status") != "INDEPENDENT_V10_NEW_STYLE_BANK_ADMISSION_PASSED" or not _all_true(admission.get("gates")):
        raise ValueError("new style admission is not an all-gates PASS")
    decision = admission.get("decision", {})
    if decision.get("outer_prepare_allowed") is not True or decision.get("outer_open_allowed") is not False:
        raise ValueError("new style admission boundary does not allow prepare-only")
    if admission.get("artifact", {}).get("writer_ids") != "synthetic_writer_064..095":
        raise ValueError("admission writer namespace drift")
    components = admission.get("component_audits", {})
    for path, key in ((FULL_AUDIT, "full_bank_sha256"), (CROSS_AUDIT, "cross_bank_sha256")):
        if components.get(key) != sha256(path):
            raise ValueError(f"component audit hash drift: {path.name}")
        component = _read_json(path)
        # Component auditors use typed gates (zero counts and empty failure
        # lists as well as booleans). Their semantics are normalized to the
        # 17 boolean gates in the independently built admission receipt.
        if not str(component.get("status", "")).endswith("PASSED"):
            raise ValueError(f"component audit status is not PASS: {path.name}")
    if sha256(R3_AUDIT) != EXPECTED_R3_AUDIT_SHA256:
        raise ValueError("r3 artifact audit hash drift")
    r3_audit = _read_json(R3_AUDIT)
    if r3_audit.get("status") != "INDEPENDENT_V10_R3_SYNTHETIC64_ARTIFACT_AUDIT_PASSED" or not _all_true(r3_audit.get("gates")):
        raise ValueError("r3 frozen artifact audit is not an all-gates PASS")
    if r3_audit.get("decision", {}).get("new_style_bank_064_095_generation_allowed") is not True:
        raise ValueError("r3 audit decision drift")
    freeze = _read_json(SCORER / "freeze_receipt.json")
    artifact_map = {
        "report_sha256": "development_report.json",
        "model_sha256": "candidate_scorer.pt",
        "scaler_sha256": "feature_scaler.npz",
        "statistics_sha256": "global_statistics_synthetic64.npz",
        "config_sha256": "frozen_config.json",
        "split_sha256": "writer_split_manifest.json",
        "started_sha256": "R3_DEVELOPMENT_STARTED.json",
        "source_snapshot_sha256": "train_sparse_writer_style_scorer_v10_r3.source.py",
    }
    for key, name in artifact_map.items():
        if freeze.get(key) != sha256(SCORER / name):
            raise ValueError(f"r3 frozen artifact drift: {name}")
    config = _read_json(SCORER / "frozen_config.json")
    if config.get("model_kind") != "monotone_logistic" or config.get("action_cost") != 4.0 or config.get("threshold") != 0.98:
        raise ValueError("frozen v10 policy changed")
    if tuple(config.get("feature_names", ())) == () or config.get("minimum_calibration_class_support") != 2:
        raise ValueError("frozen feature/support contract changed")
    final_hashes = admission.get("artifact", {}).get("final_hashes", {})
    live_final = {
        "report": sha256(BANK / "report.json"),
        "bank": sha256(BANK / "synthetic_writer_style_bank_v5_r4.npz"),
        "metadata": sha256(BANK / "synthetic_writer_style_bank_v5_r4.metadata.jsonl.gz"),
        "latents": sha256(BANK / "writer_latents.json"),
    }
    if final_hashes != live_final:
        raise ValueError("admitted bank hash drift")
    return {"admission": admission, "r3_audit": r3_audit, "freeze": freeze, "bank_hashes": live_final}


def _bound_hashes() -> dict:
    files = {
        "admission_receipt": ADMISSION,
        "r3_independent_audit": R3_AUDIT,
        "full_bank_audit": FULL_AUDIT,
        "cross_bank_audit": CROSS_AUDIT,
        "bank_report": BANK / "report.json",
        "bank_npz": BANK / "synthetic_writer_style_bank_v5_r4.npz",
        "bank_metadata": BANK / "synthetic_writer_style_bank_v5_r4.metadata.jsonl.gz",
        "writer_latents": BANK / "writer_latents.json",
        "r3_freeze_receipt": SCORER / "freeze_receipt.json",
        "r3_model": SCORER / "candidate_scorer.pt",
        "r3_scaler": SCORER / "feature_scaler.npz",
        "r3_statistics": SCORER / "global_statistics_synthetic64.npz",
        "r3_config": SCORER / "frozen_config.json",
        "r3_split": SCORER / "writer_split_manifest.json",
        "r3_report": SCORER / "development_report.json",
        "r3_source_snapshot": SCORER / "train_sparse_writer_style_scorer_v10_r3.source.py",
        "checkpoint": CHECKPOINT,
        "hwr_output_helper": ROOT / "scripts/evaluate_writer_adaptation_v8.py",
        "hwr_model_source": ROOT / "scripts/train_character_classifier_v1.py",
        "bank_builder_source": ROOT / "scripts/build_cleanroom_writer_style_bank_v5.py",
        "bank_postprocessor_source": ROOT / "scripts/postprocess_cleanroom_writer_style_bank_v5_r4.py",
    }
    return {name: {"path": str(path.resolve()), "sha256": sha256(path)} for name, path in files.items()}


def _expected_receipt(snapshot: Path) -> dict:
    _verify_semantic_inputs()
    return {
        "schema": "aiflow-v10-sparse-writer-style-outer-prepare/v1",
        "status": "PREPARED_V10_WRITERS064_095_OUTER_UNOPENED",
        "outer_local_writer_ids": list(range(32)),
        "outer_global_writer_ids": list(EXPECTED_GLOBAL_WRITERS),
        "local_to_global_contract": "global_writer_id = local_writer_index + 64",
        "support_sizes": list(SUPPORT_SIZES),
        "query_scope": "all episode_split=1 rows for every writer and support size",
        "frozen_policy": {"model": "monotone_logistic", "action_cost": 4.0, "threshold": 0.98, "candidate_class_support": 2},
        "hashes": _bound_hashes(),
        "opener_script_sha256": sha256(Path(__file__)),
        "opener_source_snapshot_sha256": sha256(snapshot),
        "boundary": {
            "bank_npz_content_opened_by_prepare": False,
            "labels_features_logits_predictions_created_by_prepare": False,
            "cache_created_by_prepare": False,
            "outer_opened": False,
            "training_performed": False,
            "legacy_opened": False,
            "known_replay_opened": False,
            "fresh_acceptance_opened": False,
            "crohme_opened": False,
            "mathwriting_opened": False,
            "hwr_checkpoint_runtime_changed": False,
            "product_promotion": False,
        },
    }


def _verify_preopen_audit(path: Path, snapshot: Path, receipt_path: Path) -> dict:
    if not path.is_file():
        raise ValueError("independent pre-open audit is required")
    payload = _read_json(path)
    if payload.get("status") != EXPECTED_PREOPEN_AUDIT_STATUS or not _all_true(payload.get("gates")):
        raise ValueError("independent pre-open audit is not an all-gates PASS")
    hashes = payload.get("hashes", {})
    expected = {
        "evaluator_live_sha256": sha256(Path(__file__)),
        "evaluator_source_snapshot_sha256": sha256(snapshot),
        "prepared_receipt_sha256": sha256(receipt_path),
    }
    if any(hashes.get(key) != value for key, value in expected.items()):
        raise ValueError("independent pre-open audit source/receipt hash mismatch")
    if payload.get("decision", {}).get("outer_open_allowed") is not True:
        raise ValueError("independent pre-open audit does not authorize one-shot open")
    return {"path": str(path.resolve()), "sha256": sha256(path), "status": payload["status"], "verified_hashes": expected}


def _load_frozen_v10():
    path = SCORER / "train_sparse_writer_style_scorer_v10_r3.source.py"
    spec = importlib.util.spec_from_file_location("aiflow_frozen_v10_r3", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen v10 source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _episode_indices_without_query_truth(writer: int, size: int, prediction_data: dict, calibration_truth: dict[int, int]):
    np = prediction_data["np"]
    calibration_all = np.flatnonzero((prediction_data["writers"] == writer) & (prediction_data["splits"] == 0))
    query = np.flatnonzero((prediction_data["writers"] == writer) & (prediction_data["splits"] == 1))
    by_label = defaultdict(list)
    for index in calibration_all.tolist():
        by_label[int(calibration_truth[index])].append(index)
    ordered = sorted(by_label, key=lambda label: hashlib.sha256(f"20260823:{writer}:{size}:{label}".encode()).hexdigest())
    pair_count = max(1, size // 4)
    pair_labels = ordered[:pair_count]
    single_labels = ordered[pair_count:pair_count + size - 2 * pair_count]
    selected = [index for label in pair_labels for index in by_label[label][:2]] + [by_label[label][0] for label in single_labels]
    if len(selected) != size:
        raise AssertionError("sparse calibration episode size mismatch")
    return np.asarray(selected, np.int64), query


def _episode_context_without_query_truth(calibration, prediction_data: dict, calibration_truth: dict[int, int], stats: dict, exclude: int | None = None, eligibility_support: dict[int, int] | None = None):
    np = prediction_data["np"]
    selected = calibration if exclude is None else calibration[calibration != exclude]
    norm = prediction_data["norm_embeddings"]
    selected_labels = np.asarray([calibration_truth[int(index)] for index in selected], np.int64)
    supported_mask = stats["admitted"][selected_labels]
    supported = selected[supported_mask]
    supported_labels = selected_labels[supported_mask]
    if len(supported):
        residual_rows = norm[supported] - stats["prototypes"][supported_labels]
        residual = np.median(residual_rows, axis=0)
        dispersion = float(np.median(np.linalg.norm(residual_rows - residual, axis=1)))
    else:
        residual = np.zeros(norm.shape[1], np.float32); dispersion = 0.0
    shrinkage = len(supported) / max(len(supported) + 8.0 * (1.0 + dispersion), 1e-8)
    adjusted = v10_normalize((stats["prototypes"] + shrinkage * residual[None]).astype(np.float32), np)
    by_class = defaultdict(list)
    for index, label in zip(selected.tolist(), selected_labels.tolist(), strict=True):
        by_class[int(label)].append(index)
    selected_support = {label: len(rows) for label, rows in by_class.items()}
    support = selected_support if eligibility_support is None else dict(eligibility_support)
    cal_prototypes = {}
    for label, rows in by_class.items():
        if int(support.get(label, 0)) >= 2 and len(rows) >= 1:
            centroid = norm[rows].mean(axis=0)
            centroid /= max(float(np.linalg.norm(centroid)), 1e-8)
            cal_prototypes[label] = centroid
    return {"adjusted": adjusted, "support": support, "cal_prototypes": cal_prototypes, "shrinkage": float(shrinkage), "dispersion": dispersion}


def v10_normalize(values, np):
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)


def _outer_episode(writer: int, size: int, prediction_data: dict, calibration_truth: dict[int, int], stats: dict, groups: list[list[int]], v10):
    calibration, query = _episode_indices_without_query_truth(writer, size, prediction_data, calibration_truth)
    context = _episode_context_without_query_truth(calibration, prediction_data, calibration_truth, stats)
    records = []
    features = []
    for index in query.tolist():
        values, candidates, ranks = v10.inference_row_features(prediction_data["norm_embeddings"][index], prediction_data["logits"][index], context, stats, groups)
        for feature, candidate, rank in zip(values, candidates, ranks, strict=True):
            features.append(feature)
            records.append((writer, size, writer * 100 + size, index, candidate, rank))
    full_support = defaultdict(int)
    for index in calibration.tolist():
        full_support[int(calibration_truth[index])] += 1
    self_features = []
    self_records = []
    for index in calibration.tolist():
        local_context = _episode_context_without_query_truth(calibration, prediction_data, calibration_truth, stats, index, full_support)
        values, candidates, ranks = v10.inference_row_features(prediction_data["norm_embeddings"][index], prediction_data["logits"][index], local_context, stats, groups)
        for feature, candidate, rank in zip(values, candidates, ranks, strict=True):
            self_features.append(feature)
            self_records.append((writer * 100 + size, index, candidate, rank))
    supported = sum(count >= 2 for count in context["support"].values())
    deficiency = "none" if supported == 0 else "one" if supported == 1 else "multiple"
    np = v10.np
    return {
        "writer": writer, "support_size": size, "episode_id": writer * 100 + size,
        "calibration": calibration, "query": query, "context": context,
        "features": np.asarray(features, np.float32).reshape(-1, len(v10.FEATURE_NAMES)), "records": records,
        "self_features": np.asarray(self_features, np.float32).reshape(-1, len(v10.FEATURE_NAMES)), "self_records": self_records,
        "deficiency": deficiency,
    }


def _predict_without_query_truth(episode: dict, prediction_data: dict, calibration_truth_by_index: dict[int, int], model, mean, scale, threshold: float, v10):
    np = v10.np
    calibration = episode["calibration"]
    self_probability = v10._probabilities(model, episode["self_features"], mean, scale)
    local_map = {index: position for position, index in enumerate(calibration.tolist())}
    local_options = [(local_map[row], candidate, rank) for _episode, row, candidate, rank in episode["self_records"]]
    calibration_logits = prediction_data["logits"][calibration]
    preliminary = v10.apply_predictions(calibration_logits, local_options, self_probability, threshold, True)
    calibration_truth = np.asarray([calibration_truth_by_index[int(index)] for index in calibration], np.int64)
    calibration_baseline = np.argmax(calibration_logits, axis=1)
    evidence = {}
    allowed = set()
    for candidate in sorted(set(preliminary[preliminary != calibration_baseline].tolist())):
        selected = (preliminary == candidate) & (preliminary != calibration_baseline)
        changed = int(np.sum(selected))
        correct = int(np.sum(selected & (preliminary == calibration_truth) & (calibration_baseline != calibration_truth)))
        regressed = int(np.sum(selected & (calibration_baseline == calibration_truth) & (preliminary != calibration_truth)))
        lower = v10.wilson_lower(correct, changed)
        accepted = correct > 0 and regressed == 0 and lower >= 0.20
        evidence[str(int(candidate))] = {"changed": changed, "correct_promotions": correct, "regressions": regressed, "wilson_lower": lower, "accepted": accepted}
        if accepted:
            allowed.add(int(candidate))
    selected_self = [i for i, (_row, candidate, _rank) in enumerate(local_options) if candidate in allowed]
    self_options = [local_options[i] for i in selected_self]
    self_prob = self_probability[np.asarray(selected_self, np.int64)] if selected_self else np.empty(0, np.float32)
    self_prediction = v10.apply_predictions(calibration_logits, self_options, self_prob, threshold, bool(allowed))
    self_chunks = [np.arange(start, min(start + 4, len(calibration))) for start in range(0, len(calibration), 4)]
    base_exact = sum(bool(np.all(calibration_baseline[c] == calibration_truth[c])) for c in self_chunks)
    adapted_exact = sum(bool(np.all(self_prediction[c] == calibration_truth[c])) for c in self_chunks)
    self_regressed = int(np.sum((calibration_baseline == calibration_truth) & (self_prediction != calibration_truth)))
    enabled = bool(allowed) and self_regressed == 0 and adapted_exact >= base_exact
    query = episode["query"]
    query_map = {index: position for position, index in enumerate(query.tolist())}
    selected_query = [i for i, row in enumerate(episode["records"]) if int(row[4]) in allowed]
    query_options = [(query_map[episode["records"][i][3]], episode["records"][i][4], episode["records"][i][5]) for i in selected_query]
    probability = v10._probabilities(model, episode["features"], mean, scale)
    query_probability = probability[np.asarray(selected_query, np.int64)] if selected_query else np.empty(0, np.float32)
    prediction = v10.apply_predictions(prediction_data["logits"][query], query_options, query_probability, threshold, enabled)
    return prediction, {"enabled": enabled, "allowed_candidates": sorted(allowed), "candidate_evidence": evidence}


def _score_episode(episode: dict, prediction, state: dict, prediction_data: dict, scoring_labels, writer_names: dict[int, str], v10):
    np = v10.np
    query = episode["query"]
    logits = prediction_data["logits"][query]
    truth = scoring_labels[query]  # scoring-only access after prediction is frozen
    baseline_top5 = np.argsort(logits, axis=1)[:, -5:][:, ::-1]
    baseline = baseline_top5[:, 0]
    old_correct = baseline == truth
    new_correct = prediction == truth
    chunks = [np.arange(start, min(start + 8, len(query))) for start in range(0, len(query), 8)]
    trace = []
    set_mismatches = 0
    for local in np.flatnonzero(prediction != baseline).tolist():
        original = baseline_top5[local].astype(int).tolist()
        selected = int(prediction[local])
        adapted = [selected] + [value for value in original if value != selected]
        mismatch = set(original) != set(adapted)
        set_mismatches += int(mismatch)
        trace.append({
            "global_writer_id": writer_names[episode["writer"]], "global_writer_index": episode["writer"],
            "support_size": episode["support_size"], "episode_id": episode["episode_id"],
            "query_source_index": int(query[local]), "truth": int(truth[local]), "baseline": int(baseline[local]), "adapted": selected,
            "original_top5": original, "adapted_top5": adapted, "top5_set_exact": not mismatch,
        })
    row = {
        "writer": episode["writer"], "support_size": episode["support_size"], "deficiency": episode["deficiency"], "enabled": state["enabled"],
        "rows": len(query), "baseline_correct": int(old_correct.sum()), "adapted_correct": int(new_correct.sum()),
        "formulae": len(chunks), "baseline_exact": int(sum(bool(np.all(old_correct[c])) for c in chunks)), "adapted_exact": int(sum(bool(np.all(new_correct[c])) for c in chunks)),
        "improved": int(np.sum(~old_correct & new_correct)), "regressed": int(np.sum(old_correct & ~new_correct)), "changed": int(np.sum(baseline != prediction)),
        "candidate_violations": int(sum(int(prediction[i]) not in baseline_top5[i] for i in range(len(query)))),
        "direct_top5_set_mismatches": set_mismatches, "allowed_candidate_count": len(state["allowed_candidates"]), "candidate_evidence": state["candidate_evidence"],
    }
    return row, trace


def _open(receipt: dict, output: Path, cache: Path, audit_binding: dict) -> int:
    marker = output / "OUTER_OPEN_STARTED.json"
    forbidden = [marker, output / "outer_report.json", output / "outer_decision.json", output / "changed_prediction_trace.jsonl", cache]
    if any(path.exists() for path in forbidden):
        raise ValueError("one-shot outer is already opened, interrupted, or consumed")
    if sorted(path.name for path in output.iterdir()) != ["evaluate_sparse_writer_style_outer_v10.source.py", "outer_prepare_receipt.json"]:
        raise ValueError("pre-open output must contain the exact prepared two files")
    marker.write_text(json.dumps({
        "status": "V10_WRITERS064_095_OUTER_OPEN_STARTED_IMMUTABLE",
        "prepared_receipt_sha256": sha256(output / "outer_prepare_receipt.json"),
        "opener_script_sha256": sha256(Path(__file__)),
        "opener_source_snapshot_sha256": receipt["opener_source_snapshot_sha256"],
        "independent_preopen_audit": audit_binding,
        "outer_global_writer_ids": list(EXPECTED_GLOBAL_WRITERS),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    # No NPZ or checkpoint is loaded before the immutable marker above.
    import numpy as np
    import torch
    from evaluate_writer_adaptation_v8 import _frozen_outputs

    v10 = _load_frozen_v10()
    latents = _read_json(BANK / "writer_latents.json")
    writer_names = {index + 64: str(row["synthetic_writer_id"]) for index, row in enumerate(latents)}
    if writer_names != {index: f"synthetic_writer_{index:03d}" for index in EXPECTED_GLOBAL_WRITERS}:
        raise ValueError("local writer 0..31 to global 64..95 mapping drift")
    with np.load(BANK / "synthetic_writer_style_bank_v5_r4.npz", allow_pickle=False) as payload:
        features = np.asarray(payload["features"], np.float32)
        labels = np.asarray(payload["labels"], np.int64)
        local_writers = np.asarray(payload["writer_index"], np.int16)
        splits = np.asarray(payload["episode_split"], np.int8)
    if set(local_writers.tolist()) != set(range(32)) or len(labels) != 40960 or set(splits.tolist()) != {0, 1}:
        raise ValueError("outer bank inventory drift")
    label_names, embeddings, logits = _frozen_outputs(features, CHECKPOINT)
    if len(label_names) != 372 or logits.shape != (40960, 372):
        raise ValueError("frozen HWR output contract drift")
    prediction_data = {
        "np": np, "writers": (local_writers + 64).astype(np.int16), "splits": splits,
        "embeddings": embeddings.astype(np.float32), "norm_embeddings": v10._normalize(embeddings.astype(np.float32)), "logits": logits.astype(np.float32),
    }
    calibration_truth = {int(index): int(labels[index]) for index in np.flatnonzero(splits == 0).tolist()}
    np.savez_compressed(cache, labels=labels, writers=prediction_data["writers"], local_writers=local_writers, splits=splits,
                        embeddings=prediction_data["embeddings"], logits=prediction_data["logits"], bank_sha256=np.asarray(sha256(BANK / "synthetic_writer_style_bank_v5_r4.npz")),
                        checkpoint_sha256=np.asarray(sha256(CHECKPOINT)))

    payload = torch.load(SCORER / "candidate_scorer.pt", map_location="cpu", weights_only=False)
    model = v10.MonotoneLogistic(len(v10.FEATURE_NAMES)); model.load_state_dict(payload["state_dict"], strict=True); model.eval()
    with np.load(SCORER / "feature_scaler.npz", allow_pickle=False) as values:
        mean = np.asarray(values["mean"], np.float32); scale = np.asarray(values["scale"], np.float32)
    with np.load(SCORER / "global_statistics_synthetic64.npz", allow_pickle=False) as values:
        stats = {name: np.asarray(values[name]).copy() for name in values.files}
    config = _read_json(SCORER / "frozen_config.json")
    groups = v10._resolve_homographs(label_names)
    episodes = [_outer_episode(writer, size, prediction_data, calibration_truth, stats, groups, v10) for writer in EXPECTED_GLOBAL_WRITERS for size in SUPPORT_SIZES]
    rows = []
    trace = []
    for episode in episodes:
        prediction, state = _predict_without_query_truth(episode, prediction_data, calibration_truth, model, mean, scale, float(config["threshold"]), v10)
        row, changed = _score_episode(episode, prediction, state, prediction_data, labels, writer_names, v10)
        rows.append(row); trace.extend(changed)
    result = v10.summarize_episode_rows(rows)
    direct_set_mismatches = sum(row["direct_top5_set_mismatches"] for row in rows)
    gates = {
        "aggregate_positive": result["positive"],
        "all_writer_support_deficiency_nonregression": result["safe"],
        "all_writer_by_support_nonregression": result["writer_by_support_episode_nonregression"],
        "candidate_violations_zero": result["overall"]["candidate_violations"] == 0,
        "direct_adapted_top5_set_exact": direct_set_mismatches == 0,
        "writer_namespace_exact_064_095": set(prediction_data["writers"].tolist()) == set(EXPECTED_GLOBAL_WRITERS),
        "support_sizes_exact": sorted(set(row["support_size"] for row in rows)) == list(SUPPORT_SIZES),
        "training_not_performed": True,
        "legacy_remained_closed": True, "known_replay_remained_closed": True, "fresh_acceptance_remained_closed": True,
        "crohme_remained_closed": True, "mathwriting_remained_closed": True, "hwr_checkpoint_runtime_unchanged": True,
    }
    passed = all(gates.values())
    trace_path = output / "changed_prediction_trace.jsonl"
    with trace_path.open("w", encoding="utf-8") as stream:
        for record in trace:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    report = {
        "schema": "aiflow-v10-sparse-writer-style-outer/v1", "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "V10_WRITERS064_095_SYNTHETIC_OUTER_PASS" if passed else "V10_WRITERS064_095_SYNTHETIC_OUTER_REJECTED",
        "evaluation": v10.compact_evaluation(result), "direct_top5_set_mismatches": direct_set_mismatches,
        "changed_trace_rows": len(trace), "gates": gates,
        "boundary": "one-shot synthetic known-external-parent-manifold evidence only; pseudo exact is an 8-row diagnostic, not real formula exact or product evidence",
        "hashes": {"prepare_receipt": sha256(output / "outer_prepare_receipt.json"), "open_marker": sha256(marker), "cache": sha256(cache), "change_trace": sha256(trace_path)},
        "independent_preopen_audit": audit_binding,
    }
    report_path = output / "outer_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    decision = {
        "status": report["status"], "passed": passed, "outer_consumed": True, "retry_allowed": False,
        "product_promotion_allowed": False, "real_corpus_open_allowed": False,
        "independent_preopen_audit": audit_binding,
        "report_sha256": sha256(report_path), "cache_sha256": sha256(cache), "trace_sha256": sha256(trace_path), "marker_sha256": sha256(marker),
    }
    (output / "outer_decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "overall": result["overall"], "output": str(output)}, ensure_ascii=False))
    return 0 if passed else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--independent-audit", type=Path)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()
    if args.prepare == args.open:
        parser.error("choose exactly one of --prepare or --open")
    receipt_path = args.output / "outer_prepare_receipt.json"
    snapshot = args.output / "evaluate_sparse_writer_style_outer_v10.source.py"
    if args.prepare:
        if args.output.exists() or args.cache.exists():
            parser.error("prepare output/cache must not already exist")
        _verify_semantic_inputs()
        args.output.mkdir(parents=True)
        shutil.copyfile(Path(__file__), snapshot)
        receipt = _expected_receipt(snapshot)
        if receipt["opener_script_sha256"] != receipt["opener_source_snapshot_sha256"]:
            raise ValueError("prepare source snapshot mismatch")
        receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"status": receipt["status"], "output": str(args.output), "outer_global_writer_ids": receipt["outer_global_writer_ids"]}, ensure_ascii=False))
        return 0
    if not receipt_path.is_file() or not snapshot.is_file() or args.cache.exists():
        raise ValueError("valid unopened prepare receipt/source/cache boundary missing")
    actual = _read_json(receipt_path)
    expected = _expected_receipt(snapshot)
    if actual != expected or expected["opener_script_sha256"] != expected["opener_source_snapshot_sha256"]:
        raise ValueError("prepared receipt, live source, or bound input drift")
    if args.independent_audit is None:
        raise ValueError("--open requires --independent-audit")
    audit_binding = _verify_preopen_audit(args.independent_audit, snapshot, receipt_path)
    return _open(actual, args.output, args.cache, audit_binding)


if __name__ == "__main__":
    raise SystemExit(main())
