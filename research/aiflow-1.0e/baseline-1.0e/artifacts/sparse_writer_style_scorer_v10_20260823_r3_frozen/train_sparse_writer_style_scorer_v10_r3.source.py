"""Train a leakage-free sparse-calibration writer style scorer on synthetic64.

No Legacy evaluation, replay, fresh acceptance, CROHME, or MathWriting data is
loaded.  All fold statistics exclude the held-out writer set.  The scorer can
only select an item from each frozen baseline Top-5 and otherwise abstains.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
from pathlib import Path
import subprocess

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from build_cleanroom_writer_style_bank_v5 import sha256
from hierarchical_writer_model_v7 import DEFAULT_CHECKPOINT
from train_cleanroom_writer_adapter_v9 import SUPPORT_SIZES, _episode_indices


ROOT = Path(__file__).resolve().parents[1]
OLD_CACHE = ROOT / "artifacts/writer_adaptation_v9_frozen_cache_20260823_r1.npz"
EXT_CONSUMED_CACHE = ROOT / "artifacts/writer_adaptation_v9_extension32_frozen_cache_20260823_r1.npz"
EXT_RESERVE_CACHE = ROOT / "artifacts/writer_adaptation_v9_global_prototype_reserve_cache_20260823_r4.npz"
OLD_BANK = ROOT / "artifacts/cleanroom_writer_style_v5_20260823_smoke32_r4_r2/synthetic_writer_style_bank_v5_r4.npz"
EXT_BANK = ROOT / "artifacts/cleanroom_writer_style_v5_20260823_extension32_r4_r1/synthetic_writer_style_bank_v5_r4.npz"
R2_CONSUMED_OUTPUT = ROOT / "artifacts/sparse_writer_style_scorer_v10_20260823_r2_frozen"
R3_OUTPUT = ROOT / "artifacts/sparse_writer_style_scorer_v10_20260823_r3_frozen"
DEFAULT_OUTPUT = R3_OUTPUT
R3_FIXED_ACTION_COST = 4.0
R3_FIXED_THRESHOLD = 0.98
R2_COMMIT = "6d29b866e61eebaf728d0e3c5322b94956a2b117"
R2_MARKER = R2_CONSUMED_OUTPUT / "CHECK_OPEN_STARTED.json"
R2_REJECTION = R2_CONSUMED_OUTPUT / "pre_new_style_rejection.json"
R2_AUDIT = ROOT / "reports/V10_SPARSE_WRITER_STYLE_R2_CHECK_REJECTION_INDEPENDENT_AUDIT.json"
R2_EXPECTED_HASHES = {"marker": "f249651971b8e48972262d0b197bd72bc8bbe0030c32ccdfe8ca17ee1721f45f", "rejection": "8044be54d338f5a291f9cad7432375ad9c2a658efc3b53b5abaacef9b3534ac1", "audit": "c860075c61085c81bae99ba63b475292bc735e3247c3018984d25c599dff1f81"}
EXPECTED_STATIC_AUDIT_STATUS = "INDEPENDENT_V10_PREDEVELOPMENT_STATIC_AUDIT_PASSED"
SEED = 20260823
MIN_GLOBAL_SUPPORT = 4
MIN_CAL_CLASS_SUPPORT = 2
SHRINKAGE_PRIOR = 8.0
SELF_CHECK_CHUNK_SIZE = 4
MIN_SELF_PROMOTION_WILSON_LOWER = 0.20
ACTION_COSTS = (4.0, 8.0)
THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98)
FEATURE_NAMES = (
    "candidate_relative_logit", "global_cosine_advantage", "writer_adjusted_cosine_advantage",
    "candidate_calibration_vs_baseline_global_cosine", "candidate_calibration_support_log1p",
    "candidate_global_reliability", "candidate_class_precision", "negative_candidate_class_fpr",
    "writer_residual_shrinkage", "negative_writer_residual_dispersion", "row_entropy",
    "negative_top1_margin", "candidate_minus_baseline_reliability", "candidate_support_eligible",
)
HOMOGRAPH_TOKEN_GROUPS = (
    ("1", "|", "/"), ("0", "O", "o"), ("x", "\\times"),
    ("Z", "\\mathcal{Z}"), ("\\epsilon", "\\varepsilon"),
    ("\\setminus", "\\backslash"), ("\\Rightarrow", "\\Longrightarrow"),
    ("P", "\\mathcal{P}"), ("\\parallel", "|"), ("\\mathfrak{M}", "\\ohm"),
)


class MonotoneLogistic(nn.Module):
    def __init__(self, dimensions: int) -> None:
        super().__init__(); self.raw_weight = nn.Parameter(torch.zeros(dimensions)); self.bias = nn.Parameter(torch.tensor(-2.0))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values @ F.softplus(self.raw_weight) + self.bias


class SharedTinyMLP(nn.Module):
    def __init__(self, dimensions: int) -> None:
        super().__init__(); self.layers = nn.Sequential(nn.Linear(dimensions, 12), nn.Tanh(), nn.Linear(12, 1))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values).squeeze(-1)


def _normalize(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)


def _scalar(payload: np.lib.npyio.NpzFile, name: str) -> str:
    return str(np.asarray(payload[name]).item())


def validate_independent_audit(path: Path, source: Path) -> dict:
    if not path.is_file(): raise ValueError(f"independent audit missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != EXPECTED_STATIC_AUDIT_STATUS: raise ValueError("independent audit status is not PASS")
    gates = payload.get("gates")
    if not isinstance(gates, dict) or not gates or not all(value is True for value in gates.values()): raise ValueError("independent audit gates are not all true")
    if payload.get("script_sha256") != sha256(source): raise ValueError("independent audit script hash mismatch")
    decision = payload.get("decision", {})
    if decision.get("development_run_allowed") is not True or decision.get("new_styles_allowed") is not False: raise ValueError("independent audit decision boundary mismatch")
    return {"path": str(path.resolve()), "sha256": sha256(path), "status": payload["status"], "script_sha256": payload["script_sha256"]}


def validate_r2_consumed_lineage() -> dict:
    live = {"marker": sha256(R2_MARKER), "rejection": sha256(R2_REJECTION), "audit": sha256(R2_AUDIT)}
    if live != R2_EXPECTED_HASHES: raise ValueError(f"r2 consumed lineage hash mismatch: {live}")
    result = subprocess.run(["git", "cat-file", "-e", f"{R2_COMMIT}^{{commit}}"], cwd=ROOT, capture_output=True, text=True, check=False)
    if result.returncode != 0: raise ValueError("r2 immutable commit is unavailable")
    return {"commit": R2_COMMIT, "marker_sha256": live["marker"], "rejection_sha256": live["rejection"], "independent_audit_sha256": live["audit"]}


def rejection_envelope(status: str, audit_binding: dict, fit_coverage: dict | None = None, development_coverage: dict | None = None, check_opened: bool = False, **details: object) -> dict:
    return {
        "status": status,
        "script_sha256": sha256(Path(__file__)),
        "independent_audit": audit_binding,
        "structural_coverage": {"fit40": fit_coverage, "development12": development_coverage},
        "check_opened": check_opened,
        **details,
    }


def _load_cache(path: Path, source: str, offset: int) -> tuple[dict[str, np.ndarray], list[dict]]:
    with np.load(path, allow_pickle=False) as payload:
        arrays = {key: np.asarray(payload[key]).copy() for key in ("labels", "writers", "splits", "embeddings", "logits")}
        embedded = {key: _scalar(payload, key) for key in ("bank_sha256", "checkpoint_sha256")}
    local = arrays["writers"].astype(np.int16); arrays["local_writers"] = local.copy(); arrays["writers"] = (local + offset).astype(np.int16)
    manifest = [{"source_cache": source, "local_writer_id": int(writer), "global_writer_id": int(writer + offset), "rows": int(np.sum(local == writer))} for writer in sorted(set(local.tolist()))]
    manifest.append({"source_cache": source, "cache_sha256": sha256(path), **embedded})
    return arrays, manifest


def load_synthetic64(checkpoint: Path) -> tuple[dict[str, np.ndarray], list[dict], dict]:
    old, old_manifest = _load_cache(OLD_CACHE, "old32", 0)
    consumed, consumed_manifest = _load_cache(EXT_CONSUMED_CACHE, "extension_consumed16", 32)
    reserve, reserve_manifest = _load_cache(EXT_RESERVE_CACHE, "extension_reserve16", 32)
    writers = [set(block["writers"].tolist()) for block in (old, consumed, reserve)]
    if writers[1] & writers[2] or writers[0] & (writers[1] | writers[2]) or writers[0] | writers[1] | writers[2] != set(range(64)):
        raise ValueError("global writer namespace is not exact 0..63 disjoint")
    checkpoint_hash = sha256(checkpoint)
    if old_manifest[-1]["checkpoint_sha256"] != checkpoint_hash or consumed_manifest[-1]["checkpoint_sha256"] != checkpoint_hash or reserve_manifest[-1]["checkpoint_sha256"] != checkpoint_hash:
        raise ValueError("cache checkpoint mismatch")
    if old_manifest[-1]["bank_sha256"] != sha256(OLD_BANK) or consumed_manifest[-1]["bank_sha256"] != sha256(EXT_BANK) or reserve_manifest[-1]["bank_sha256"] != sha256(EXT_BANK):
        raise ValueError("cache bank mismatch")
    data = {key: np.concatenate([old[key], consumed[key], reserve[key]]) for key in ("labels", "writers", "local_writers", "splits", "embeddings", "logits")}
    data["norm_embeddings"] = _normalize(data["embeddings"])
    inventory = {
        "rows": len(data["labels"]), "writers": len(set(data["writers"].tolist())),
        "split_counts": {str(value): int(np.sum(data["splits"] == value)) for value in (0, 1)},
        "checkpoint_sha256": checkpoint_hash, "old_bank_sha256": sha256(OLD_BANK), "extension_bank_sha256": sha256(EXT_BANK),
    }
    if inventory["rows"] != 81280 or inventory["writers"] != 64 or inventory["split_counts"] != {"0": 40640, "1": 40640}:
        raise ValueError(f"synthetic64 inventory changed: {inventory}")
    return data, old_manifest + consumed_manifest + reserve_manifest, inventory


def _writer_order() -> list[int]:
    return sorted(range(64), key=lambda writer: hashlib.sha256(f"{SEED}:v10-writer:{writer}".encode()).hexdigest())


def writer_splits() -> dict[str, list[int]]:
    ordered = _writer_order()
    meta = {"fit": ordered[:40], "development": ordered[40:52], "check": ordered[52:64]}
    if set(meta["fit"]) & set(meta["development"]) or set(meta["fit"]) & set(meta["check"]) or set(meta["development"]) & set(meta["check"]):
        raise AssertionError("meta writer split overlap")
    if set().union(*map(set, meta.values())) != set(range(64)): raise AssertionError("meta writer split incomplete")
    return meta


def partition_writers(writers: list[int], folds: int = 4) -> list[list[int]]:
    output = [writers[index::folds] for index in range(folds)]
    if any(set(output[i]) & set(output[j]) for i in range(folds) for j in range(i + 1, folds)) or set().union(*map(set, output)) != set(writers):
        raise AssertionError("writer fold overlap")
    return output


def _resolve_homographs(labels: list[str]) -> list[list[int]]:
    output = []
    for group in HOMOGRAPH_TOKEN_GROUPS:
        resolved = [labels.index(token) for token in group if token in labels]
        if len(resolved) >= 2: output.append(resolved)
    return output


def fold_statistics(data: dict[str, np.ndarray], training_writers: list[int]) -> dict[str, np.ndarray]:
    mask = np.isin(data["writers"], np.asarray(training_writers, np.int16)); norm = data["norm_embeddings"]
    classes = data["logits"].shape[1]; dimensions = norm.shape[1]
    sums = np.zeros((classes, dimensions), np.float64); counts = np.zeros(classes, np.int64)
    np.add.at(sums, data["labels"][mask], norm[mask]); np.add.at(counts, data["labels"][mask], 1)
    prototypes = _normalize((sums / np.maximum(counts[:, None], 1)).astype(np.float32))
    # Reliability and class error rates are writer-LOO estimates: every row is
    # scored against prototypes that exclude all rows from that row's writer.
    reliability_sum = np.zeros(classes, np.float64); reliability_count = np.zeros(classes, np.int64)
    predicted_count = np.zeros(classes, np.int64); true_positive = np.zeros(classes, np.int64)
    for writer in training_writers:
        writer_rows = np.flatnonzero(mask & (data["writers"] == writer)); writer_labels = data["labels"][writer_rows]
        writer_sums = np.zeros_like(sums); writer_counts = np.zeros_like(counts)
        np.add.at(writer_sums, writer_labels, norm[writer_rows]); np.add.at(writer_counts, writer_labels, 1)
        loo_counts = counts - writer_counts; loo_sums = sums - writer_sums
        loo_prototypes = _normalize((loo_sums / np.maximum(loo_counts[:, None], 1)).astype(np.float32))
        valid_truth = loo_counts[writer_labels] >= MIN_GLOBAL_SUPPORT
        truth_cosine = np.einsum("nd,nd->n", norm[writer_rows], loo_prototypes[writer_labels])
        np.add.at(reliability_sum, writer_labels[valid_truth], truth_cosine[valid_truth]); np.add.at(reliability_count, writer_labels[valid_truth], 1)
        top5 = np.argsort(data["logits"][writer_rows], axis=1)[:, -5:][:, ::-1]
        admitted_loo = loo_counts >= MIN_GLOBAL_SUPPORT
        cosine = np.einsum("nd,nkd->nk", norm[writer_rows], loo_prototypes[top5]); cosine[~admitted_loo[top5]] = -np.inf
        usable = np.any(admitted_loo[top5], axis=1)
        prediction = top5[:, 0].copy(); prediction[usable] = top5[usable, np.argmax(cosine[usable], axis=1)]
        np.add.at(predicted_count, prediction, 1); correct = prediction == writer_labels; np.add.at(true_positive, prediction[correct], 1)
    reliability = (reliability_sum / np.maximum(reliability_count, 1)).astype(np.float32)
    admitted = (counts >= MIN_GLOBAL_SUPPORT) & (reliability_count > 0) & np.isfinite(reliability) & (reliability > 0)
    precision = (true_positive + 1.0) / (predicted_count + 2.0); fpr = (predicted_count - true_positive + 1.0) / (predicted_count + 2.0)
    return {"prototypes": prototypes, "counts": counts, "reliability": reliability, "reliability_loo_counts": reliability_count, "admitted": admitted, "class_precision": precision.astype(np.float32), "class_fpr": fpr.astype(np.float32)}


def _homograph_collision(candidates: np.ndarray, groups: list[list[int]]) -> bool:
    values = set(candidates.tolist()); return any(len(values & set(group)) >= 2 for group in groups)


def _episode_context(calibration: np.ndarray, data: dict[str, np.ndarray], stats: dict[str, np.ndarray], exclude: int | None = None, eligibility_support: dict[int, int] | None = None) -> dict:
    selected = calibration if exclude is None else calibration[calibration != exclude]
    norm = data["norm_embeddings"]; supported = selected[stats["admitted"][data["labels"][selected]]]
    if len(supported):
        residual_rows = norm[supported] - stats["prototypes"][data["labels"][supported]]
        residual = np.median(residual_rows, axis=0); dispersion = float(np.median(np.linalg.norm(residual_rows - residual, axis=1)))
    else: residual = np.zeros(norm.shape[1], np.float32); dispersion = 0.0
    shrinkage = len(supported) / max(len(supported) + SHRINKAGE_PRIOR * (1.0 + dispersion), 1e-8)
    adjusted = _normalize((stats["prototypes"] + shrinkage * residual[None]).astype(np.float32))
    by_class = defaultdict(list)
    for index in selected.tolist(): by_class[int(data["labels"][index])].append(index)
    selected_support = {label: len(rows) for label, rows in by_class.items()}
    support = selected_support if eligibility_support is None else dict(eligibility_support)
    cal_prototypes = {}
    for label, rows in by_class.items():
        if int(support.get(label, 0)) >= MIN_CAL_CLASS_SUPPORT and len(rows) >= 1:
            centroid = norm[rows].mean(axis=0); centroid /= max(float(np.linalg.norm(centroid)), 1e-8); cal_prototypes[label] = centroid
    return {"adjusted": adjusted, "support": support, "cal_prototypes": cal_prototypes, "shrinkage": float(shrinkage), "dispersion": dispersion}


def inference_row_features(embedding: np.ndarray, logits: np.ndarray, context: dict, stats: dict[str, np.ndarray], groups: list[list[int]]) -> tuple[list[list[float]], list[int], list[int]]:
    candidates = np.argsort(logits)[-5:][::-1].astype(int)
    if not np.all(stats["admitted"][candidates]) or _homograph_collision(candidates, groups): return [], [], []
    norm = embedding; values = logits[candidates].astype(np.float64); spread = max(float(values.std()), 1e-6)
    shifted = values - values.max(); probability = np.exp(shifted); probability /= probability.sum(); entropy = float(-np.sum(probability * np.log(np.maximum(probability, 1e-12))))
    global_cos = norm @ stats["prototypes"][candidates].T; adjusted_cos = norm @ context["adjusted"][candidates].T
    baseline = int(candidates[0]); baseline_global_cos = float(global_cos[0])
    features = []; option_labels = []; option_ranks = []
    for rank in range(1, 5):
        candidate = int(candidates[rank])
        support = int(context["support"].get(candidate, 0))
        if support < MIN_CAL_CLASS_SUPPORT or candidate not in context["cal_prototypes"]: continue
        candidate_cal = context["cal_prototypes"][candidate]; candidate_cal_cos = float(norm @ candidate_cal)
        features.append([
            float((values[rank] - values[0]) / spread), float(global_cos[rank] - global_cos[0]), float(adjusted_cos[rank] - adjusted_cos[0]),
            float(candidate_cal_cos - baseline_global_cos), float(np.log1p(support)), float(stats["reliability"][candidate]),
            float(stats["class_precision"][candidate]), float(-stats["class_fpr"][candidate]), float(context["shrinkage"]), float(-context["dispersion"]),
            entropy, float(-(values[0] - values[1]) / spread), float(stats["reliability"][candidate] - stats["reliability"][baseline]), float(support >= MIN_CAL_CLASS_SUPPORT),
        ]); option_labels.append(candidate); option_ranks.append(rank)
    return features, option_labels, option_ranks


def _row_features(index: int, context: dict, data: dict[str, np.ndarray], stats: dict[str, np.ndarray], groups: list[list[int]]) -> tuple[list[list[float]], list[int], list[int]]:
    return inference_row_features(data["norm_embeddings"][index], data["logits"][index], context, stats, groups)


def build_episode(writer: int, size: int, data: dict[str, np.ndarray], stats: dict[str, np.ndarray], groups: list[list[int]]) -> dict:
    calibration, query = _episode_indices(writer, size, data["labels"], data["writers"], data["splits"])
    context = _episode_context(calibration, data, stats); features = []; targets = []; records = []
    episode_id = writer * 100 + size
    for index in query.tolist():
        values, candidates, ranks = _row_features(index, context, data, stats, groups)
        for feature, candidate, rank in zip(values, candidates, ranks, strict=True):
            features.append(feature); targets.append(float(int(data["labels"][index]) == candidate)); records.append((writer, size, episode_id, index, candidate, rank))
    full_support = defaultdict(int)
    for index in calibration.tolist(): full_support[int(data["labels"][index])] += 1
    self_features = []; self_targets = []; self_records = []
    for index in calibration.tolist():
        local_context = _episode_context(calibration, data, stats, index, full_support); values, candidates, ranks = _row_features(index, local_context, data, stats, groups)
        for feature, candidate, rank in zip(values, candidates, ranks, strict=True): self_features.append(feature); self_targets.append(float(int(data["labels"][index]) == candidate)); self_records.append((episode_id, index, candidate, rank))
    supported_classes = sum(count >= MIN_CAL_CLASS_SUPPORT for count in context["support"].values())
    deficiency = "none" if supported_classes == 0 else "one" if supported_classes == 1 else "multiple"
    return {"writer": writer, "support_size": size, "episode_id": episode_id, "calibration": calibration, "query": query, "features": np.asarray(features, np.float32).reshape(-1, len(FEATURE_NAMES)), "targets": np.asarray(targets, np.float32), "records": records, "self_features": np.asarray(self_features, np.float32).reshape(-1, len(FEATURE_NAMES)), "self_targets": np.asarray(self_targets, np.float32), "self_records": self_records, "deficiency": deficiency}


def _probabilities(model: nn.Module, features: np.ndarray, mean: np.ndarray, scale: np.ndarray, feature_mask: np.ndarray | None = None) -> np.ndarray:
    if not len(features): return np.empty(0, np.float32)
    normalized = ((features - mean) / scale).astype(np.float32)
    if feature_mask is not None: normalized[:, ~feature_mask] = 0.0
    with torch.inference_mode(): return torch.sigmoid(model(torch.from_numpy(normalized))).numpy()


def apply_predictions(logits: np.ndarray, row_options: list[tuple[int, int, int]], probabilities: np.ndarray, threshold: float, enabled: bool) -> np.ndarray:
    prediction = np.argmax(logits, axis=1).astype(np.int64)
    if not enabled: return prediction
    options = defaultdict(list)
    for probability, (row, candidate, rank) in zip(probabilities.tolist(), row_options, strict=True): options[int(row)].append((float(probability), -int(rank), int(candidate)))
    for row, values in options.items():
        best = max(values)
        if best[0] >= threshold: prediction[row] = best[2]
    return prediction


def wilson_lower(successes: int, trials: int, z: float = 1.96) -> float:
    if trials <= 0: return 0.0
    proportion = successes / trials; denominator = 1.0 + z * z / trials
    centre = proportion + z * z / (2.0 * trials)
    radius = z * math.sqrt(proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials))
    return (centre - radius) / denominator


def _episode_result(episode: dict, data: dict[str, np.ndarray], model: nn.Module, mean: np.ndarray, scale: np.ndarray, threshold: float, feature_mask: np.ndarray | None = None) -> dict:
    self_prob = _probabilities(model, episode["self_features"], mean, scale, feature_mask); self_options = [(record[1], record[2], record[3]) for record in episode["self_records"]]
    calibration = episode["calibration"]
    candidate_evidence = {}; allowed_candidates: set[int] = set()
    if len(calibration):
        local_map = {index: position for position, index in enumerate(calibration.tolist())}; local_options = [(local_map[row], candidate, rank) for row, candidate, rank in self_options]
        preliminary = apply_predictions(data["logits"][calibration], local_options, self_prob, threshold, True); self_truth = data["labels"][calibration]; self_base = np.argmax(data["logits"][calibration], axis=1)
        for candidate in sorted(set(preliminary[preliminary != self_base].tolist())):
            selected = (preliminary == candidate) & (preliminary != self_base)
            changed = int(np.sum(selected)); correct = int(np.sum(selected & (preliminary == self_truth) & (self_base != self_truth)))
            regression = int(np.sum(selected & (self_base == self_truth) & (preliminary != self_truth))); lower = wilson_lower(correct, changed)
            accepted = correct > 0 and regression == 0 and lower >= MIN_SELF_PROMOTION_WILSON_LOWER
            candidate_evidence[str(int(candidate))] = {"changed": changed, "correct_promotions": correct, "regressions": regression, "wilson_lower": lower, "accepted": accepted}
            if accepted: allowed_candidates.add(int(candidate))
        filtered_positions = [position for position, (_row, candidate, _rank) in enumerate(local_options) if candidate in allowed_candidates]
        filtered_options = [local_options[position] for position in filtered_positions]; filtered_prob = self_prob[np.asarray(filtered_positions, np.int64)] if filtered_positions else np.empty(0, np.float32)
        self_prediction = apply_predictions(data["logits"][calibration], filtered_options, filtered_prob, threshold, bool(allowed_candidates))
        self_improved = int(np.sum((self_base != self_truth) & (self_prediction == self_truth))); self_regressed = int(np.sum((self_base == self_truth) & (self_prediction != self_truth)))
        self_chunks = [np.arange(start, min(start + SELF_CHECK_CHUNK_SIZE, len(calibration))) for start in range(0, len(calibration), SELF_CHECK_CHUNK_SIZE)]
        self_base_exact = int(sum(bool(np.all(self_base[chunk] == self_truth[chunk])) for chunk in self_chunks))
        self_adapted_exact = int(sum(bool(np.all(self_prediction[chunk] == self_truth[chunk])) for chunk in self_chunks))
        self_changed = int(np.sum(self_base != self_prediction)); self_correct_promotions = int(np.sum((self_base != self_prediction) & (self_prediction == self_truth)))
        self_wilson_lower = wilson_lower(self_correct_promotions, self_changed)
    else:
        self_improved = self_regressed = self_base_exact = self_adapted_exact = self_changed = self_correct_promotions = 0; self_wilson_lower = 0.0
    enabled = bool(allowed_candidates) and self_regressed == 0 and self_adapted_exact >= self_base_exact
    probability = _probabilities(model, episode["features"], mean, scale, feature_mask); query = episode["query"]
    index_map = {index: position for position, index in enumerate(query.tolist())}; positions = [position for position, record in enumerate(episode["records"]) if int(record[4]) in allowed_candidates]
    options = [(index_map[episode["records"][position][3]], episode["records"][position][4], episode["records"][position][5]) for position in positions]
    filtered_probability = probability[np.asarray(positions, np.int64)] if positions else np.empty(0, np.float32)
    prediction = apply_predictions(data["logits"][query], options, filtered_probability, threshold, enabled); baseline = np.argmax(data["logits"][query], axis=1); truth = data["labels"][query]
    old_correct = baseline == truth; new_correct = prediction == truth; chunks = [np.arange(start, min(start + 8, len(query))) for start in range(0, len(query), 8)]
    top5 = np.argsort(data["logits"][query], axis=1)[:, -5:]
    return {"writer": episode["writer"], "support_size": episode["support_size"], "deficiency": episode["deficiency"], "enabled": enabled,
            "rows": len(query), "baseline_correct": int(old_correct.sum()), "adapted_correct": int(new_correct.sum()),
            "formulae": len(chunks), "baseline_exact": int(sum(bool(np.all(old_correct[chunk])) for chunk in chunks)), "adapted_exact": int(sum(bool(np.all(new_correct[chunk])) for chunk in chunks)),
            "improved": int(np.sum(~old_correct & new_correct)), "regressed": int(np.sum(old_correct & ~new_correct)), "changed": int(np.sum(baseline != prediction)),
            "candidate_violations": int(sum(int(prediction[row]) not in top5[row] for row in range(len(query)))), "self_improved": self_improved, "self_regressed": self_regressed,
            "self_baseline_pseudo_exact": self_base_exact, "self_adapted_pseudo_exact": self_adapted_exact, "self_changed": self_changed, "self_correct_promotions": self_correct_promotions, "self_wilson_lower": self_wilson_lower,
            "allowed_candidate_count": len(allowed_candidates), "candidate_evidence": candidate_evidence}


def evaluate_episodes(episodes: list[dict], data: dict[str, np.ndarray], model: nn.Module, mean: np.ndarray, scale: np.ndarray, threshold: float, feature_mask: np.ndarray | None = None) -> dict:
    rows = [_episode_result(episode, data, model, mean, scale, threshold, feature_mask) for episode in episodes]
    return summarize_episode_rows(rows)


def summarize_episode_rows(rows: list[dict]) -> dict:
    def aggregate(group: list[dict]) -> dict:
        n = sum(row["rows"] for row in group); f = sum(row["formulae"] for row in group)
        return {"rows": n, "baseline_top1": sum(row["baseline_correct"] for row in group) / n, "adapted_top1": sum(row["adapted_correct"] for row in group) / n,
                "baseline_pseudo_exact": sum(row["baseline_exact"] for row in group) / f, "adapted_pseudo_exact": sum(row["adapted_exact"] for row in group) / f,
                "improved": sum(row["improved"] for row in group), "regressed": sum(row["regressed"] for row in group), "changed": sum(row["changed"] for row in group), "candidate_violations": sum(row["candidate_violations"] for row in group), "activation_coverage": sum(row["enabled"] for row in group) / len(group)}
    overall = aggregate(rows); per_writer = {str(writer): aggregate([row for row in rows if row["writer"] == writer]) for writer in sorted(set(row["writer"] for row in rows))}
    per_support = {str(size): aggregate([row for row in rows if row["support_size"] == size]) for size in SUPPORT_SIZES}
    per_deficiency = {key: aggregate([row for row in rows if row["deficiency"] == key]) for key in sorted(set(row["deficiency"] for row in rows))}
    def nonreg(values: dict) -> bool: return values["adapted_top1"] >= values["baseline_top1"] and values["adapted_pseudo_exact"] >= values["baseline_pseudo_exact"] and values["candidate_violations"] == 0
    episode_nonregression = all(row["adapted_correct"] >= row["baseline_correct"] and row["adapted_exact"] >= row["baseline_exact"] and row["candidate_violations"] == 0 for row in rows)
    safe = episode_nonregression and all(nonreg(value) for value in per_writer.values()) and all(nonreg(value) for value in per_support.values()) and all(nonreg(value) for value in per_deficiency.values()) and nonreg(overall)
    positive = overall["adapted_top1"] > overall["baseline_top1"] or overall["adapted_pseudo_exact"] > overall["baseline_pseudo_exact"]
    return {"overall": overall, "per_writer": per_writer, "per_support": per_support, "per_class_support_deficiency": per_deficiency, "writer_by_support_episode_nonregression": episode_nonregression, "safe": safe, "positive": positive, "episodes": rows}


def compact_evaluation(result: dict) -> dict:
    return {key: value for key, value in result.items() if key != "episodes"}


def structural_coverage(episodes: list[dict]) -> dict:
    query_records = [record for episode in episodes for record in episode["records"]]
    targets = np.concatenate([episode["targets"] for episode in episodes]) if episodes else np.empty(0, np.float32)
    self_targets = np.concatenate([episode["self_targets"] for episode in episodes]) if episodes else np.empty(0, np.float32)
    return {
        "episodes": len(episodes),
        "query_rows": int(sum(len(episode["query"]) for episode in episodes)),
        "query_rows_with_options": len({(int(record[0]), int(record[1]), int(record[3])) for record in query_records}),
        "query_options": len(query_records),
        "positive_targets": int(np.sum(targets == 1.0)),
        "negative_targets": int(np.sum(targets == 0.0)),
        "self_options": int(sum(len(episode["self_records"]) for episode in episodes)),
        "episodes_with_self_options": int(sum(bool(len(episode["self_records"])) for episode in episodes)),
        "self_positive_targets": int(np.sum(self_targets == 1.0)),
        "self_negative_targets": int(np.sum(self_targets == 0.0)),
        "episodes_with_self_positive": int(sum(bool(np.any(episode["self_targets"] == 1.0)) for episode in episodes)),
    }


def structural_coverage_passed(coverage: dict) -> bool:
    return all(coverage[key] > 0 for key in ("query_rows_with_options", "query_options", "positive_targets", "negative_targets", "self_options", "episodes_with_self_options", "self_positive_targets", "self_negative_targets", "episodes_with_self_positive"))


def candidate_option_permutation_test(episodes: list[dict], data: dict[str, np.ndarray], model: nn.Module, mean: np.ndarray, scale: np.ndarray, threshold: float) -> dict:
    for episode in episodes:
        grouped = defaultdict(list)
        for feature_index, record in enumerate(episode["records"]): grouped[int(record[3])].append((feature_index, record))
        for source_index, members in grouped.items():
            if len(members) < 2: continue
            feature_indices = np.asarray([member[0] for member in members], np.int64)
            records = [member[1] for member in members]
            ranks = [int(record[5]) for record in records]
            if len(set(ranks)) != len(ranks): raise AssertionError("candidate ranks are not distinct")
            probabilities = _probabilities(model, episode["features"][feature_indices], mean, scale)
            options = [(0, int(record[4]), int(record[5])) for record in records]
            logits = data["logits"][source_index:source_index + 1]
            original = apply_predictions(logits, options, probabilities, threshold, True)
            permutation = np.arange(len(options))[::-1]
            permuted = apply_predictions(logits, [options[index] for index in permutation], probabilities[permutation], threshold, True)
            return {"source_index": source_index, "option_count": len(options), "distinct_ranks": sorted(ranks), "prediction_exact": bool(np.array_equal(original, permuted))}
    raise ValueError("no development glyph has at least two promotion options")


def _train(model_kind: str, features: np.ndarray, targets: np.ndarray, mean: np.ndarray, scale: np.ndarray, action_cost: float) -> tuple[nn.Module, list[float]]:
    torch.manual_seed(SEED + int(action_cost * 10) + (0 if model_kind == "logistic" else 1000)); model = MonotoneLogistic(len(FEATURE_NAMES)) if model_kind == "logistic" else SharedTinyMLP(len(FEATURE_NAMES))
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3 if model_kind == "mlp" else 5e-3, weight_decay=0.02)
    x = torch.from_numpy(((features - mean) / scale).astype(np.float32)); y = torch.from_numpy(targets.astype(np.float32)); history = []
    for _epoch in range(16):
        order = torch.randperm(len(x)); losses = []
        for start in range(0, len(order), 8192):
            batch = order[start:start + 8192]; output = model(x[batch]); weight = torch.where(y[batch] > 0.5, 1.0, action_cost); loss = F.binary_cross_entropy_with_logits(output, y[batch], weight=weight)
            optimizer.zero_grad(); loss.backward(); optimizer.step(); losses.append(float(loss))
        history.append(float(np.mean(losses)))
    model.eval(); return model, history


def _mcnemar_p(improved: int, regressed: int) -> float:
    total = improved + regressed
    if total == 0: return 1.0
    tail = sum(math.comb(total, index) for index in range(0, min(improved, regressed) + 1)) / (2 ** total)
    return min(1.0, 2.0 * tail)


def build_writer_episodes(writers: list[int], data: dict[str, np.ndarray], stats: dict[str, np.ndarray], groups: list[list[int]]) -> list[dict]:
    return [build_episode(writer, size, data, stats, groups) for writer in writers for size in SUPPORT_SIZES]


def build_oof_training_episodes(writer_pool: list[int], data: dict[str, np.ndarray], groups: list[list[int]], stage: str) -> tuple[list[dict], list[dict]]:
    episodes = []; provenance = []
    for fold_index, heldout in enumerate(partition_writers(writer_pool)):
        training = sorted(set(writer_pool) - set(heldout)); stats = fold_statistics(data, training)
        exclusion_payload = {"stage": stage, "training": training, "heldout": heldout}
        provenance.append({
            "stage": stage, "fold": fold_index, "training_writers": training, "heldout_writers": heldout,
            "training_rows": int(np.isin(data["writers"], training).sum()), "heldout_rows": int(np.isin(data["writers"], heldout).sum()),
            "writer_exclusion_sha256": hashlib.sha256(json.dumps(exclusion_payload, separators=(",", ":")).encode()).hexdigest(),
            "class_statistics_writer_loo": True,
        })
        episodes.extend(build_writer_episodes(heldout, data, stats, groups))
    return episodes, provenance


def main() -> int:
    raise RuntimeError(f"v10 r2 is consumed and immutable at commit {R2_COMMIT}; use main_r3 only")
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT); parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT); parser.add_argument("--independent-audit", type=Path, required=True); args = parser.parse_args()
    if args.output.exists(): parser.error(f"refusing to overwrite output: {args.output}")
    audit_binding = validate_independent_audit(args.independent_audit, Path(__file__))
    r2_lineage = validate_r2_consumed_lineage()
    data, cache_manifest, inventory = load_synthetic64(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False); label_names = [str(value) for value in checkpoint["math_labels"]]; groups = _resolve_homographs(label_names)
    meta = writer_splits()
    fit_episodes, fit_oof_provenance = build_oof_training_episodes(meta["fit"], data, groups, "fit40_oof")
    development_stats = fold_statistics(data, meta["fit"])
    dev_episodes = build_writer_episodes(meta["development"], data, development_stats, groups)
    fit_coverage = structural_coverage(fit_episodes); development_coverage = structural_coverage(dev_episodes)
    if not structural_coverage_passed(fit_coverage) or not structural_coverage_passed(development_coverage):
        payload = rejection_envelope("PRE_TRAINING_STRUCTURAL_FAIL_CLOSED", audit_binding, fit_coverage, development_coverage)
        args.output.mkdir(parents=True); (args.output / "pre_training_structural_rejection.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"); print(json.dumps({"status": payload["status"]})); return 2
    train_features = np.concatenate([episode["features"] for episode in fit_episodes]); train_targets = np.concatenate([episode["targets"] for episode in fit_episodes]); mean = train_features.mean(axis=0).astype(np.float32); scale = np.maximum(train_features.std(axis=0), 1e-4).astype(np.float32)
    candidates = []
    for kind in ("logistic", "mlp"):
        for cost in ACTION_COSTS:
            model, history = _train(kind, train_features, train_targets, mean, scale, cost)
            for threshold in THRESHOLDS:
                result = evaluate_episodes(dev_episodes, data, model, mean, scale, threshold)
                candidates.append({"kind": kind, "action_cost": cost, "threshold": threshold, "result": result, "history": history, "model": model})
    eligible = [row for row in candidates if row["result"]["safe"] and row["result"]["positive"]]
    if not eligible:
        summaries = [{"kind": row["kind"], "action_cost": row["action_cost"], "threshold": row["threshold"], "result": compact_evaluation(row["result"]), "history": row["history"]} for row in candidates]
        payload = rejection_envelope("PRE_NEW_STYLE_FAIL_CLOSED", audit_binding, fit_coverage, development_coverage, candidate_summaries=summaries)
        args.output.mkdir(parents=True); (args.output / "pre_new_style_rejection.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"); print(json.dumps({"status": payload["status"]})); return 2
    best_by_kind = {}
    for kind in ("logistic", "mlp"):
        values = [row for row in eligible if row["kind"] == kind]
        if values: best_by_kind[kind] = max(values, key=lambda row: (row["result"]["overall"]["adapted_top1"] - row["result"]["overall"]["baseline_top1"], row["result"]["overall"]["adapted_pseudo_exact"] - row["result"]["overall"]["baseline_pseudo_exact"], -row["result"]["overall"]["regressed"], -row["result"]["overall"]["changed"], row["threshold"]))
    if "logistic" not in best_by_kind:
        payload = rejection_envelope("PRE_CHECK_FAIL_CLOSED_LOGISTIC_INELIGIBLE", audit_binding, fit_coverage, development_coverage, mlp_diagnostic=compact_evaluation(best_by_kind["mlp"]["result"]) if "mlp" in best_by_kind else None)
        args.output.mkdir(parents=True); (args.output / "pre_check_rejection.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"); print(json.dumps({"status": payload["status"]})); return 2
    selected = best_by_kind["logistic"]
    complexity_decision = "monotone_logistic_preferred_unless_paired_mlp_superiority_is_proven"
    if "logistic" in best_by_kind and "mlp" in best_by_kind:
        logistic = best_by_kind["logistic"]; mlp = best_by_kind["mlp"]
        delta = mlp["result"]["overall"]["adapted_top1"] - logistic["result"]["overall"]["adapted_top1"]
        complexity_decision += f"; observed_dev_delta={delta:.8f}; no_independent_paired_superiority_test_so_logistic_frozen"
    permutation_test = candidate_option_permutation_test(dev_episodes, data, selected["model"], mean, scale, selected["threshold"])
    permutation_exact = permutation_test["prediction_exact"]
    inference_parameters = list(inspect.signature(inference_row_features).parameters); prediction_parameters = list(inspect.signature(apply_predictions).parameters)
    api_safe = not any(value in inference_parameters + prediction_parameters for value in ("labels", "metadata", "writer", "writer_id", "data"))
    if not permutation_exact or not api_safe:
        payload = rejection_envelope("PRE_CHECK_API_FAIL_CLOSED", audit_binding, fit_coverage, development_coverage, permutation_test=permutation_test, api_safe=api_safe)
        args.output.mkdir(parents=True); (args.output / "pre_check_api_rejection.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"); print(json.dumps({"status": payload["status"]})); return 2
    # Hyperparameters are now frozen. Refit on fit+development writer-OOF
    # features, then open the synthetic check12 exactly once with statistics
    # that exclude the entire check writer set.
    refit_writers = meta["fit"] + meta["development"]
    refit_episodes, refit_oof_provenance = build_oof_training_episodes(refit_writers, data, groups, "fit_plus_development52_oof")
    refit_features = np.concatenate([episode["features"] for episode in refit_episodes]); refit_targets = np.concatenate([episode["targets"] for episode in refit_episodes])
    final_mean = refit_features.mean(axis=0).astype(np.float32); final_scale = np.maximum(refit_features.std(axis=0), 1e-4).astype(np.float32)
    final_model, refit_history = _train(selected["kind"], refit_features, refit_targets, final_mean, final_scale, selected["action_cost"])
    check_stats = fold_statistics(data, refit_writers)
    args.output.mkdir(parents=True)
    check_marker = args.output / "CHECK_OPEN_STARTED.json"
    check_marker.write_text(json.dumps({"status": "CHECK_OPEN_STARTED_IMMUTABLE", "script_sha256": sha256(Path(__file__)), "independent_audit": audit_binding, "check_writers": meta["check"], "frozen_hyperparameters": {"kind": selected["kind"], "action_cost": selected["action_cost"], "threshold": selected["threshold"]}}, ensure_ascii=False, indent=2), encoding="utf-8")
    check_episodes = build_writer_episodes(meta["check"], data, check_stats, groups)
    check = evaluate_episodes(check_episodes, data, final_model, final_mean, final_scale, selected["threshold"])
    if not check["safe"] or not check["positive"]:
        selected_summary = {"kind": selected["kind"], "action_cost": selected["action_cost"], "threshold": selected["threshold"], "result": compact_evaluation(selected["result"]), "history": selected["history"]}
        payload = rejection_envelope("PRE_NEW_STYLE_FAIL_CLOSED", audit_binding, fit_coverage, development_coverage, check_opened=True, selected=selected_summary, check=compact_evaluation(check), check_marker_sha256=sha256(check_marker))
        (args.output / "pre_new_style_rejection.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"); print(json.dumps({"status": payload["status"], "check": check["overall"]})); return 2
    # The exact check52 combination is preserved separately. Only after the
    # one-shot check passes is the outer candidate retrained on all64 OOF
    # features. Its first evidence must be untouched writers064..095.
    check_model_path = args.output / "check_gate_candidate_scorer.pt"; torch.save({"kind": selected["kind"], "state_dict": final_model.state_dict(), "feature_names": FEATURE_NAMES}, check_model_path)
    check_scaler_path = args.output / "check_gate_feature_scaler.npz"; np.savez(check_scaler_path, mean=final_mean, scale=final_scale)
    check_stats_path = args.output / "check_gate_statistics_fit_dev52.npz"; np.savez_compressed(check_stats_path, **check_stats)
    outer_episodes, outer_oof_provenance = build_oof_training_episodes(list(range(64)), data, groups, "outer_all64_oof")
    outer_features = np.concatenate([episode["features"] for episode in outer_episodes]); outer_targets = np.concatenate([episode["targets"] for episode in outer_episodes])
    outer_mean = outer_features.mean(axis=0).astype(np.float32); outer_scale = np.maximum(outer_features.std(axis=0), 1e-4).astype(np.float32)
    outer_model, outer_history = _train(selected["kind"], outer_features, outer_targets, outer_mean, outer_scale, selected["action_cost"])
    final_stats = fold_statistics(data, list(range(64)))
    source = args.output / "train_sparse_writer_style_scorer_v10.source.py"; source.write_bytes(Path(__file__).read_bytes())
    model_path = args.output / "candidate_scorer.pt"; torch.save({"kind": selected["kind"], "state_dict": outer_model.state_dict(), "feature_names": FEATURE_NAMES}, model_path)
    scaler_path = args.output / "feature_scaler.npz"; np.savez(scaler_path, mean=outer_mean, scale=outer_scale)
    stats_path = args.output / "global_statistics_synthetic64.npz"; np.savez_compressed(stats_path, **final_stats)
    split_path = args.output / "writer_split_manifest.json"; split_path.write_text(json.dumps({"schema": "aiflow-writer-v10-synthetic64-split/v1", "cache_namespace_manifest": cache_manifest, "inventory": inventory, "fit40_oof_folds": fit_oof_provenance, "fit_plus_development52_oof_folds": refit_oof_provenance, "outer_all64_oof_folds": outer_oof_provenance, "meta_split": meta, "development_statistics_writers": meta["fit"], "check_statistics_writers": refit_writers, "final_outer_statistics_writers": list(range(64))}, ensure_ascii=False, indent=2), encoding="utf-8")
    config = {"model_kind": selected["kind"], "action_cost": selected["action_cost"], "threshold": selected["threshold"], "minimum_global_support": MIN_GLOBAL_SUPPORT, "minimum_calibration_class_support": MIN_CAL_CLASS_SUPPORT, "shrinkage_prior": SHRINKAGE_PRIOR, "self_check_chunk_size": SELF_CHECK_CHUNK_SIZE, "minimum_self_promotion_wilson_lower": MIN_SELF_PROMOTION_WILSON_LOWER, "homograph_token_groups": HOMOGRAPH_TOKEN_GROUPS, "resolved_homograph_indices": groups, "feature_names": FEATURE_NAMES, "new_styles_opened": False}
    config_path = args.output / "frozen_config.json"; config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    masks = {
        "global_only": (0, 1, 5, 6, 7, 10, 11, 12),
        "writer_residual_only": (0, 2, 8, 9, 10, 11),
        "calibration_prototype_only": (0, 3, 4, 10, 11, 13),
        "full": tuple(range(len(FEATURE_NAMES))),
    }
    ablations = {}
    for name, indices in masks.items():
        mask = np.zeros(len(FEATURE_NAMES), bool); mask[list(indices)] = True
        ablations[name] = {"feature_names": [FEATURE_NAMES[index] for index in indices], "check": compact_evaluation(evaluate_episodes(check_episodes, data, final_model, final_mean, final_scale, selected["threshold"], mask))}
    model_comparison = {kind: {"development": compact_evaluation(row["result"]), "action_cost": row["action_cost"], "threshold": row["threshold"]} for kind, row in best_by_kind.items()}
    report = {"schema": "aiflow-sparse-writer-style-scorer/v10", "generated_at": datetime.now(timezone.utc).isoformat(), "status": "V10_DEVELOPMENT_FROZEN_NEW_STYLES_CLOSED", "structural_coverage": {"fit40": fit_coverage, "development12": development_coverage}, "selection": {"model_kind": selected["kind"], "action_cost": selected["action_cost"], "threshold": selected["threshold"], "complexity_decision": complexity_decision, "development": compact_evaluation(selected["result"]), "refit_history": refit_history, "check": compact_evaluation(check), "outer_all64_refit_history": outer_history, "outer_all64_evaluated": False, "model_comparison": model_comparison}, "ablations": ablations, "gates": {"independent_static_audit_bound": True, "fit_and_development_structural_coverage_nonzero": structural_coverage_passed(fit_coverage) and structural_coverage_passed(development_coverage), "check_marker_precedes_check_episode_build": True, "writer_oof_disjoint": True, "fit_statistics_exclude_development_and_check": True, "development_statistics_fit40_only": True, "check_statistics_fit_plus_development52_only": True, "candidate_or_writer_id_features": False, "candidate_option_permutation_exact": permutation_exact, "class_stats_writer_loo_and_exclude_heldout_query": True, "candidate_calibration_support_hard_gate": True, "calibration_pseudo_exact_and_confidence_gate": True, "frozen_action_threshold_support_shrinkage_homographs": True, "direct_adapted_top5_set_exact": check["overall"]["candidate_violations"] == 0, "unsupported_global_or_homograph_whole_row_identity": True, "writer_by_support_episode_nonregression": check["writer_by_support_episode_nonregression"], "support_macro_nonregression": check["safe"], "eval_api_label_metadata_writer_free": api_safe, "legacy_eval300_opened": False, "known_replay_opened": False, "fresh_acceptance_opened": False, "crohme_opened": False, "mathwriting_opened": False, "checkpoint_changed": False, "hwr_changed": False, "runtime_changed": False, "new_styles_opened": False}, "boundaries": {"parent_overlap": "known external parent manifold; tracked separately from writer-disjoint folds", "class_statistics": "fit OOF excludes held fold; development uses fit40 only; check uses fit+development52 only", "outer_all64_artifact": "retrained only after check pass; exact all64 model/scaler/stats has no evaluation evidence yet", "outer_first_evidence": "untouched synthetic writers064..095 after a second independent audit", "pseudo_formula_exact": "fixed synthetic 8-row diagnostic only, never a real formula metric", "next": "independent static and artifact audit before generating writers 064..095"}, "hashes": {"independent_static_audit": audit_binding["sha256"], "check_open_marker": sha256(check_marker), "old_cache": sha256(OLD_CACHE), "extension_consumed_cache": sha256(EXT_CONSUMED_CACHE), "extension_reserve_cache": sha256(EXT_RESERVE_CACHE), "checkpoint": sha256(args.checkpoint)}}
    report["candidate_option_permutation_test"] = permutation_test
    report_path = args.output / "development_report.json"; report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    receipt = {"status": report["status"], "independent_static_audit_sha256": audit_binding["sha256"], "check_open_marker_sha256": sha256(check_marker), "check_gate_model_sha256": sha256(check_model_path), "check_gate_scaler_sha256": sha256(check_scaler_path), "check_gate_statistics_sha256": sha256(check_stats_path), "report_sha256": sha256(report_path), "model_sha256": sha256(model_path), "scaler_sha256": sha256(scaler_path), "statistics_sha256": sha256(stats_path), "split_sha256": sha256(split_path), "config_sha256": sha256(config_path), "script_sha256": sha256(Path(__file__)), "source_snapshot_sha256": sha256(source)}
    (args.output / "freeze_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "status": report["status"], "selected": {"kind": selected["kind"], "cost": selected["action_cost"], "threshold": selected["threshold"]}, "check": check["overall"]}, ensure_ascii=False)); return 0


def main() -> int:
    raise RuntimeError(f"v10 r2 is consumed and immutable at commit {R2_COMMIT}; use main_r3 only")


def main_r3() -> int:
    parser = argparse.ArgumentParser(description="v10 r3 fixed candidate-class writer policy")
    parser.add_argument("--output", type=Path, default=R3_OUTPUT); parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT); parser.add_argument("--independent-audit", type=Path, required=True); args = parser.parse_args()
    if args.output.exists(): parser.error(f"refusing to overwrite output: {args.output}")
    audit_binding = validate_independent_audit(args.independent_audit, Path(__file__))
    r2_lineage = validate_r2_consumed_lineage()
    data, cache_manifest, inventory = load_synthetic64(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False); label_names = [str(value) for value in checkpoint["math_labels"]]; groups = _resolve_homographs(label_names)
    api_parameters = list(inspect.signature(inference_row_features).parameters) + list(inspect.signature(apply_predictions).parameters)
    api_safe = not any(value in api_parameters for value in ("labels", "metadata", "writer", "writer_id", "data"))
    if not api_safe: raise ValueError("inference API is not label/writer blind")
    args.output.mkdir(parents=True)
    source = args.output / "train_sparse_writer_style_scorer_v10_r3.source.py"; source.write_bytes(Path(__file__).read_bytes())
    started = args.output / "R3_DEVELOPMENT_STARTED.json"
    started.write_text(json.dumps({"status": "R3_DEVELOPMENT_STARTED_IMMUTABLE", "script_sha256": sha256(Path(__file__)), "source_snapshot_sha256": sha256(source), "independent_audit": audit_binding, "r2_consumed_lineage": r2_lineage, "writers": list(range(64)), "new_styles_opened": False}, ensure_ascii=False, indent=2), encoding="utf-8")
    masks = {
        "global_reference": (0, 1, 5, 6, 7, 10, 11, 12),
        "writer_residual": (0, 2, 8, 9, 10, 11),
        "calibration_prototype": (0, 3, 4, 10, 11, 13),
        "full": tuple(range(len(FEATURE_NAMES))),
    }
    outer_rows = []; ablation_rows = {name: [] for name in masks}; fold_reports = []
    for fold_index, heldout in enumerate(partition_writers(list(range(64)), 8)):
        training = sorted(set(range(64)) - set(heldout))
        training_episodes, training_provenance = build_oof_training_episodes(training, data, groups, f"r3_outer{fold_index}_train56_oof")
        training_coverage = structural_coverage(training_episodes)
        if not structural_coverage_passed(training_coverage): raise ValueError(f"outer fold {fold_index} training coverage failed")
        features = np.concatenate([episode["features"] for episode in training_episodes]); targets = np.concatenate([episode["targets"] for episode in training_episodes])
        mean = features.mean(axis=0).astype(np.float32); scale = np.maximum(features.std(axis=0), 1e-4).astype(np.float32)
        model, history = _train("logistic", features, targets, mean, scale, R3_FIXED_ACTION_COST)
        outer_stats = fold_statistics(data, training); evaluation_episodes = build_writer_episodes(heldout, data, outer_stats, groups)
        fold_rows = [_episode_result(episode, data, model, mean, scale, R3_FIXED_THRESHOLD) for episode in evaluation_episodes]; outer_rows.extend(fold_rows)
        for name, indices in masks.items():
            mask = np.zeros(len(FEATURE_NAMES), bool); mask[list(indices)] = True
            ablation_rows[name].extend(_episode_result(episode, data, model, mean, scale, R3_FIXED_THRESHOLD, mask) for episode in evaluation_episodes)
        fold_reports.append({"fold": fold_index, "training_writers": training, "heldout_writers": heldout, "training_coverage": training_coverage, "training_oof_provenance": training_provenance, "history": history, "evaluation": compact_evaluation(summarize_episode_rows(fold_rows))})
    nested = summarize_episode_rows(outer_rows)
    ablations = {name: compact_evaluation(summarize_episode_rows(rows)) for name, rows in ablation_rows.items()}
    if not nested["safe"] or not nested["positive"]:
        payload = rejection_envelope("R3_NESTED_OOF_FAIL_CLOSED", audit_binding, check_opened=False, nested_oof=compact_evaluation(nested), ablations=ablations, folds=fold_reports, source_snapshot_sha256=sha256(source), development_started_sha256=sha256(started), r2_consumed_lineage=r2_lineage)
        rejection = args.output / "r3_nested_oof_rejection.json"; rejection.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"status": payload["status"], "nested": nested["overall"]}, ensure_ascii=False)); return 2
    final_episodes, final_provenance = build_oof_training_episodes(list(range(64)), data, groups, "r3_final_all64_oof")
    final_coverage = structural_coverage(final_episodes)
    if not structural_coverage_passed(final_coverage): raise ValueError("final all64 OOF coverage failed")
    final_features = np.concatenate([episode["features"] for episode in final_episodes]); final_targets = np.concatenate([episode["targets"] for episode in final_episodes])
    final_mean = final_features.mean(axis=0).astype(np.float32); final_scale = np.maximum(final_features.std(axis=0), 1e-4).astype(np.float32)
    final_model, final_history = _train("logistic", final_features, final_targets, final_mean, final_scale, R3_FIXED_ACTION_COST); final_stats = fold_statistics(data, list(range(64)))
    model_path = args.output / "candidate_scorer.pt"; torch.save({"kind": "logistic", "state_dict": final_model.state_dict(), "feature_names": FEATURE_NAMES}, model_path)
    scaler_path = args.output / "feature_scaler.npz"; np.savez(scaler_path, mean=final_mean, scale=final_scale)
    stats_path = args.output / "global_statistics_synthetic64.npz"; np.savez_compressed(stats_path, **final_stats)
    config_path = args.output / "frozen_config.json"; config_path.write_text(json.dumps({"model_kind": "monotone_logistic", "action_cost": R3_FIXED_ACTION_COST, "threshold": R3_FIXED_THRESHOLD, "minimum_global_support": MIN_GLOBAL_SUPPORT, "minimum_calibration_class_support": MIN_CAL_CLASS_SUPPORT, "shrinkage_prior": SHRINKAGE_PRIOR, "self_check_chunk_size": SELF_CHECK_CHUNK_SIZE, "minimum_candidate_wilson_lower": MIN_SELF_PROMOTION_WILSON_LOWER, "candidate_class_gate": "correct>0_and_regression=0_and_wilson>=0.20", "homograph_token_groups": HOMOGRAPH_TOKEN_GROUPS, "feature_names": FEATURE_NAMES, "new_styles_opened": False}, ensure_ascii=False, indent=2), encoding="utf-8")
    split_path = args.output / "writer_split_manifest.json"; split_path.write_text(json.dumps({"schema": "aiflow-writer-v10-r3-nested-oof/v1", "cache_namespace_manifest": cache_manifest, "inventory": inventory, "outer_folds": fold_reports, "final_all64_oof_provenance": final_provenance, "final_coverage": final_coverage}, ensure_ascii=False, indent=2), encoding="utf-8")
    report = {"schema": "aiflow-sparse-writer-style-scorer/v10-r3", "generated_at": datetime.now(timezone.utc).isoformat(), "status": "V10_R3_SYNTHETIC64_DEVELOPMENT_FROZEN_NEW_STYLES_CLOSED", "fixed_policy": {"model": "monotone_logistic", "action_cost": R3_FIXED_ACTION_COST, "threshold": R3_FIXED_THRESHOLD}, "r2_consumed_lineage": r2_lineage, "nested_oof": compact_evaluation(nested), "ablations": ablations, "final_all64": {"history": final_history, "coverage": final_coverage, "evaluated": False}, "gates": {"all_writer_top1_and_pseudo_exact_nonregression": nested["safe"], "all_writer_by_support_nonregression": nested["writer_by_support_episode_nonregression"], "aggregate_positive": nested["positive"], "candidate_violations_zero": nested["overall"]["candidate_violations"] == 0, "candidate_class_calibration_gate": True, "inference_label_writer_blind": api_safe, "new_styles_opened": False, "legacy_eval300_opened": False, "known_replay_opened": False, "fresh_acceptance_opened": False, "crohme_opened": False, "mathwriting_opened": False, "hwr_checkpoint_runtime_changed": False}, "boundaries": {"synthetic64": "consumed development only", "all64_artifact": "unevaluated after final fit", "first_outer_evidence": "writers064..095 after independent artifact audit", "promotion": False}, "hashes": {"script": sha256(Path(__file__)), "source_snapshot": sha256(source), "independent_audit": audit_binding["sha256"], "checkpoint": sha256(args.checkpoint)}}
    report_path = args.output / "development_report.json"; report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    receipt = {"status": report["status"], "r2_consumed_lineage": r2_lineage, "report_sha256": sha256(report_path), "model_sha256": sha256(model_path), "scaler_sha256": sha256(scaler_path), "statistics_sha256": sha256(stats_path), "config_sha256": sha256(config_path), "split_sha256": sha256(split_path), "started_sha256": sha256(started), "script_sha256": sha256(Path(__file__)), "source_snapshot_sha256": sha256(source), "independent_audit_sha256": audit_binding["sha256"]}
    receipt_path = args.output / "freeze_receipt.json"; receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "nested": nested["overall"], "output": str(args.output)}, ensure_ascii=False)); return 0


if __name__ == "__main__":
    raise SystemExit(main_r3())
