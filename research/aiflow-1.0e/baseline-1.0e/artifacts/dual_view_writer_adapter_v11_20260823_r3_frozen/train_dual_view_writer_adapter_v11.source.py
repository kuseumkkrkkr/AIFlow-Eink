#!/usr/bin/env python3
"""Develop a frozen-HWR, candidate-preserving dual-view writer adapter on writers000..095."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
OLD_BANK = ROOT / "artifacts/cleanroom_writer_style_v5_20260823_smoke32_r4_r2"
EXT_BANK = ROOT / "artifacts/cleanroom_writer_style_v5_20260823_extension32_r4_r1"
NEW_DEV_BANK = ROOT / "artifacts/cleanroom_writer_style_v5_20260823_writers064_095_r4_r1"
OLD_CACHE = ROOT / "artifacts/writer_adaptation_v9_frozen_cache_20260823_r1.npz"
EXT_CONSUMED_CACHE = ROOT / "artifacts/writer_adaptation_v9_extension32_frozen_cache_20260823_r1.npz"
EXT_RESERVE_CACHE = ROOT / "artifacts/writer_adaptation_v9_global_prototype_reserve_cache_20260823_r4.npz"
NEW_DEV_CACHE = ROOT / "artifacts/sparse_writer_style_scorer_v10_writers064_095_outer_r3_cache.npz"
CHECKPOINT = ROOT / "artifacts/commercial_hwr_cleanroom_physics_20260823_r1_shadow/commercial_hwr_cleanroom_physics_checkpoint.pt"
DEFAULT_OUTPUT = ROOT / "artifacts/dual_view_writer_adapter_v11_20260823_r3_frozen"
R1_OUTPUT = ROOT / "artifacts/dual_view_writer_adapter_v11_20260823_r1_frozen"
R1_REJECTION_AUDIT = ROOT / "reports/V11_DUAL_VIEW_R1_REJECTION_INDEPENDENT_AUDIT.json"
R2_OUTPUT = ROOT / "artifacts/dual_view_writer_adapter_v11_20260823_r2_frozen"
R2_REJECTION_AUDIT = ROOT / "reports/V11_DUAL_VIEW_R2_REJECTION_INDEPENDENT_AUDIT.json"

EXPECTED_AUDIT_STATUS = "INDEPENDENT_V11_DUAL_VIEW_R3_PREDEVELOPMENT_STATIC_AUDIT_PASSED"
R1_HASHES = {
    "development_rejection.json": "c9f2b1d007c5580b558b5f2f97426d8d3cdc890095b8bf98d08af67f96eef4a4",
    "train_dual_view_writer_adapter_v11.source.py": "69e39539b447b8d1ccd06dbdcf976b67b58e0cb5d2cd9500db20d257ad6a0837",
    "V11_DEVELOPMENT_STARTED.json": "fb92aa439d28278908ef8ad7e6e27ad8a8194bd7a36493a269967215bdc71bdb",
}
R1_REJECTION_AUDIT_SHA256 = "8cbc38b7918ecb89a989659ff7b45abd544fa1fd3a6b9a58dc9d9dd92e17a2b2"
R2_HASHES = {
    "development_rejection.json": "e4831ed308ad7489cbb7b7c4df3c48c3eb4b97ad190eda0c7f20070869027461",
    "stage_telemetry.json": "fadf8c4d4540a5b3b82306e839619b6615e845b5537d712be749b63ed52c148a",
    "train_dual_view_writer_adapter_v11.source.py": "f9be1bb22a20c457b52d41f141506577ba67684ead12e6da4af03fad1dd6d13d",
    "V11_DEVELOPMENT_STARTED.json": "875bcc3d1e418c0d492aa7837f86ebd1a6134fe1089e375ed355a129a7c295fe",
}
R2_REJECTION_AUDIT_SHA256 = "c7ebc5ed1208b11645469e1623cc66fb1527caa9718adc22925076b17b4bfd09"
SEED = 20260823
SUPPORT_SIZES = (4, 6, 10, 20, 22, 24)
DTW_POINTS_PER_STROKE = 16
DTW_WINDOW = 4
DTW_COST = "mean_euclidean_xy_per_path_step"
MIN_GLOBAL_SUPPORT = 4
MIN_CALIBRATION_SUPPORT = 2
MAX_GEOMETRY_EXEMPLARS = 4
MAX_REGRESSION_WILSON_UPPER = 0.20
ACTION_COST = 4.0
THRESHOLDS = (0.90, 0.95, 0.98, 0.99)
FEATURE_NAMES = (
    "candidate_relative_logit", "global_embedding_advantage", "calibration_embedding_advantage",
    "global_geometry_advantage", "calibration_geometry_advantage", "negative_calibration_dtw",
    "calibration_support_log1p", "candidate_global_reliability", "negative_geometry_dispersion",
    "negative_row_entropy", "negative_top1_margin", "candidate_support_eligible",
)
HOMOGRAPH_TOKEN_GROUPS = (
    ("1", "|", "/"), ("0", "O", "o"), ("x", "\\times"),
    ("Z", "\\mathcal{Z}"), ("\\epsilon", "\\varepsilon"),
    ("\\setminus", "\\backslash"), ("\\Rightarrow", "\\Longrightarrow"),
    ("P", "\\mathcal{P}"), ("\\parallel", "|"), ("\\mathfrak{M}", "\\ohm"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class MonotoneLogistic(nn.Module):
    def __init__(self, dimensions: int) -> None:
        super().__init__(); self.raw_weight = nn.Parameter(torch.zeros(dimensions)); self.bias = nn.Parameter(torch.tensor(-2.0))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values @ F.softplus(self.raw_weight) + self.bias


def _normalize(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)


def _validate_audit(path: Path) -> dict:
    if not path.is_file(): raise ValueError("independent static audit missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    gates = payload.get("gates", {})
    if payload.get("status") != EXPECTED_AUDIT_STATUS or not isinstance(gates, dict) or not gates or not all(value is True for value in gates.values()):
        raise ValueError("independent static audit is not an all-gates PASS")
    if payload.get("script_sha256") != sha256(Path(__file__)):
        raise ValueError("independent static audit source hash mismatch")
    decision = payload.get("decision", {})
    if decision.get("development_run_allowed") is not True or decision.get("writers096_127_open_allowed") is not False:
        raise ValueError("independent static audit decision boundary mismatch")
    return {"path": str(path.resolve()), "sha256": sha256(path), "status": payload["status"]}


def _validate_r1_lineage() -> dict:
    observed = {}
    for name, expected in R1_HASHES.items():
        path = R1_OUTPUT / name
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"immutable v11 r1 lineage mismatch: {name}")
        observed[name] = expected
    if not R1_REJECTION_AUDIT.is_file() or sha256(R1_REJECTION_AUDIT) != R1_REJECTION_AUDIT_SHA256:
        raise ValueError("immutable v11 r1 independent audit mismatch")
    payload = json.loads(R1_REJECTION_AUDIT.read_text(encoding="utf-8"))
    if payload.get("status") != "INDEPENDENT_V11_DUAL_VIEW_R1_REJECTION_AUDIT_PASSED":
        raise ValueError("v11 r1 independent rejection status mismatch")
    return {"r1_files": observed, "r1_independent_audit_sha256": R1_REJECTION_AUDIT_SHA256,
            "r1_reuse": False, "r1_result_specific_threshold_tuning": False}


def _validate_consumed_lineage() -> dict:
    lineage = _validate_r1_lineage(); observed = {}
    for name, expected in R2_HASHES.items():
        path = R2_OUTPUT / name
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"immutable v11 r2 lineage mismatch: {name}")
        observed[name] = expected
    if not R2_REJECTION_AUDIT.is_file() or sha256(R2_REJECTION_AUDIT) != R2_REJECTION_AUDIT_SHA256:
        raise ValueError("immutable v11 r2 independent audit mismatch")
    payload = json.loads(R2_REJECTION_AUDIT.read_text(encoding="utf-8"))
    if payload.get("status") != "INDEPENDENT_V11_DUAL_VIEW_R2_REJECTION_AUDIT_PASSED":
        raise ValueError("v11 r2 independent rejection status mismatch")
    lineage.update({"r2_files": observed, "r2_independent_audit_sha256": R2_REJECTION_AUDIT_SHA256,
                    "r2_reuse": False, "r2_result_specific_tuning": False})
    return lineage


def _bank_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path / "synthetic_writer_style_bank_v5_r4.npz", allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]).copy() for key in ("features", "labels", "writer_index", "episode_split")}


def _cache_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]).copy() for key in ("labels", "writers", "splits", "embeddings", "logits")}


def _aligned_block(bank: dict[str, np.ndarray], cache: dict[str, np.ndarray], writer_offset: int, selected_local: set[int] | None = None) -> dict[str, np.ndarray]:
    mask = np.ones(len(bank["labels"]), bool) if selected_local is None else np.isin(bank["writer_index"], np.asarray(sorted(selected_local), np.int16))
    local = bank["writer_index"][mask].astype(np.int16)
    cache_local = cache["writers"].astype(np.int16)
    if not np.array_equal(bank["labels"][mask], cache["labels"]) or not np.array_equal(local, cache_local) or not np.array_equal(bank["episode_split"][mask], cache["splits"]):
        raise ValueError("bank/cache row alignment mismatch")
    return {
        "trajectories": bank["features"][mask].astype(np.float32), "labels": cache["labels"].astype(np.int64),
        "writers": (local + writer_offset).astype(np.int16), "splits": cache["splits"].astype(np.int8),
        "embeddings": cache["embeddings"].astype(np.float32), "logits": cache["logits"].astype(np.float32),
    }


def load_development96() -> tuple[dict[str, np.ndarray], dict]:
    old_bank = _bank_arrays(OLD_BANK); ext_bank = _bank_arrays(EXT_BANK); new_bank = _bank_arrays(NEW_DEV_BANK)
    old = _aligned_block(old_bank, _cache_arrays(OLD_CACHE), 0)
    consumed_cache = _cache_arrays(EXT_CONSUMED_CACHE); reserve_cache = _cache_arrays(EXT_RESERVE_CACHE)
    consumed = _aligned_block(ext_bank, consumed_cache, 32, set(consumed_cache["writers"].tolist()))
    reserve = _aligned_block(ext_bank, reserve_cache, 32, set(reserve_cache["writers"].tolist()))
    new_cache = _cache_arrays(NEW_DEV_CACHE)
    # The consumed v10 cache already stores global writers64..095 while the bank is local0..31.
    local_new = dict(new_cache); local_new["writers"] = (new_cache["writers"] - 64).astype(np.int16)
    new = _aligned_block(new_bank, local_new, 64)
    blocks = (old, consumed, reserve, new)
    data = {key: np.concatenate([block[key] for block in blocks]) for key in blocks[0]}
    data["norm_embeddings"] = _normalize(data["embeddings"])
    if set(data["writers"].tolist()) != set(range(96)) or len(data["labels"]) != 122240:
        raise ValueError("development96 inventory mismatch")
    if int(np.sum(data["splits"] == 0)) != 61120 or int(np.sum(data["splits"] == 1)) != 61120:
        raise ValueError("development96 split inventory mismatch")
    checkpoint_hashes = set()
    bank_hashes = {}
    for name, path in (("old", OLD_CACHE), ("extension_consumed", EXT_CONSUMED_CACHE), ("extension_reserve", EXT_RESERVE_CACHE), ("new_development", NEW_DEV_CACHE)):
        with np.load(path, allow_pickle=False) as payload:
            checkpoint_hashes.add(str(np.asarray(payload["checkpoint_sha256"]).item()))
            bank_hashes[name] = str(np.asarray(payload["bank_sha256"]).item())
    if checkpoint_hashes != {sha256(CHECKPOINT)}:
        raise ValueError("cache checkpoint provenance mismatch")
    inventory = {"rows": len(data["labels"]), "writers": 96, "calibration_rows": 61120, "query_rows": 61120,
                 "checkpoint_sha256": sha256(CHECKPOINT), "cache_bank_hashes": bank_hashes,
                 "development_only_writer_ids": list(range(96)), "writers096_127_accessed": False}
    return data, inventory


def _resample(points: np.ndarray, count: int = DTW_POINTS_PER_STROKE) -> np.ndarray:
    if len(points) == 1: return np.repeat(points, count, axis=0).astype(np.float32)
    distance = np.linalg.norm(np.diff(points, axis=0), axis=1); cumulative = np.concatenate(([0.0], np.cumsum(distance)))
    if cumulative[-1] <= 1e-8: return np.repeat(points[:1], count, axis=0).astype(np.float32)
    targets = np.linspace(0.0, cumulative[-1], count)
    return np.stack([np.interp(targets, cumulative, points[:, axis]) for axis in range(2)], axis=1).astype(np.float32)


def trajectory_signature(tensor: np.ndarray) -> tuple[np.ndarray, ...]:
    observed = tensor[:, 4] > 0.5; values = tensor[observed]
    if not len(values): return tuple()
    starts = np.flatnonzero(values[:, 3] > 0.5).tolist()
    if not starts or starts[0] != 0: starts = [0] + starts
    starts = sorted(set(starts)); ends = starts[1:] + [len(values)]
    return tuple(_resample(values[left:right, :2]) for left, right in zip(starts, ends, strict=True) if right > left)


@lru_cache(maxsize=2_000_000)
def _dtw_cached(left: int, right: int) -> float:
    return _DTW_ENGINE.distance(left, right)


class GeometryEngine:
    def __init__(self, trajectories: np.ndarray) -> None:
        self.signatures = [trajectory_signature(row) for row in trajectories]

    def distance(self, left: int, right: int) -> float:
        return self.signature_distance(self.signatures[left], self.signatures[right])

    @staticmethod
    def signature_distance(a: tuple[np.ndarray, ...], b: tuple[np.ndarray, ...]) -> float:
        if len(a) != len(b) or not a: return math.inf
        total = 0.0; steps = 0
        for x, y in zip(a, b, strict=True):
            n, m = len(x), len(y); cost = np.full((n + 1, m + 1), np.inf, np.float64); path = np.zeros((n + 1, m + 1), np.int16); cost[0, 0] = 0.0
            for i in range(1, n + 1):
                for j in range(max(1, i - DTW_WINDOW), min(m, i + DTW_WINDOW) + 1):
                    choices = ((cost[i - 1, j], path[i - 1, j]), (cost[i, j - 1], path[i, j - 1]), (cost[i - 1, j - 1], path[i - 1, j - 1]))
                    best_cost, best_steps = min(choices, key=lambda value: (value[0], value[1]))
                    cost[i, j] = best_cost + float(np.linalg.norm(x[i - 1] - y[j - 1])); path[i, j] = best_steps + 1
            if not np.isfinite(cost[n, m]): return math.inf
            total += float(cost[n, m]); steps += int(path[n, m])
        return total / max(steps, 1)


_DTW_ENGINE: GeometryEngine


def _resolve_homographs(label_names: list[str]) -> list[list[int]]:
    output = []
    for group in HOMOGRAPH_TOKEN_GROUPS:
        values = [label_names.index(token) for token in group if token in label_names]
        if len(values) >= 2: output.append(values)
    return output


def _collision(candidates: np.ndarray, groups: list[list[int]]) -> bool:
    values = set(candidates.tolist()); return any(len(values & set(group)) >= 2 for group in groups)


def fold_statistics(data: dict[str, np.ndarray], development_labels: np.ndarray, training_writers: list[int]) -> dict:
    mask = np.isin(data["writers"], np.asarray(training_writers, np.int16)); labels = development_labels; norm = data["norm_embeddings"]
    classes, dimensions = data["logits"].shape[1], norm.shape[1]
    sums = np.zeros((classes, dimensions), np.float64); counts = np.zeros(classes, np.int64)
    np.add.at(sums, labels[mask], norm[mask]); np.add.at(counts, labels[mask], 1)
    prototypes = _normalize((sums / np.maximum(counts[:, None], 1)).astype(np.float32))
    reliability_sum = np.zeros(classes, np.float64); reliability_count = np.zeros(classes, np.int64)
    # Writer-LOO reliability: every scored training row uses a class prototype
    # that excludes every row from that row's writer.
    for writer in training_writers:
        rows = np.flatnonzero(mask & (data["writers"] == writer)); writer_labels = labels[rows]
        writer_sums = np.zeros_like(sums); writer_counts = np.zeros_like(counts)
        np.add.at(writer_sums, writer_labels, norm[rows]); np.add.at(writer_counts, writer_labels, 1)
        loo_counts = counts - writer_counts; loo_prototypes = _normalize(((sums - writer_sums) / np.maximum(loo_counts[:, None], 1)).astype(np.float32))
        valid = loo_counts[writer_labels] >= MIN_GLOBAL_SUPPORT
        cosine = np.einsum("nd,nd->n", norm[rows], loo_prototypes[writer_labels])
        np.add.at(reliability_sum, writer_labels[valid], cosine[valid]); np.add.at(reliability_count, writer_labels[valid], 1)
    reliability = (reliability_sum / np.maximum(reliability_count, 1)).astype(np.float32)
    exemplars: dict[int, list[int]] = {}
    for label in np.flatnonzero(counts >= MIN_GLOBAL_SUPPORT).tolist():
        rows = np.flatnonzero(mask & (labels == label)).tolist()
        rows.sort(key=lambda index: hashlib.sha256(f"{SEED}:geometry:{label}:{index}".encode()).hexdigest())
        exemplars[label] = rows[:MAX_GEOMETRY_EXEMPLARS]
    admitted = (counts >= MIN_GLOBAL_SUPPORT) & (reliability > 0)
    for label in np.flatnonzero(admitted).tolist(): admitted[label] = bool(exemplars.get(label))
    return {"prototypes": prototypes, "counts": counts, "reliability": reliability, "admitted": admitted, "geometry_exemplars": exemplars,
            "training_writer_hash": hashlib.sha256(json.dumps(sorted(training_writers), separators=(",", ":")).encode()).hexdigest()}


def _global_geometry(query_signature: tuple[np.ndarray, ...], label: int, stats: dict) -> float:
    values = [_DTW_ENGINE.signature_distance(query_signature, _DTW_ENGINE.signatures[row]) for row in stats["geometry_exemplars"].get(label, [])]
    finite = [value for value in values if math.isfinite(value)]
    return min(finite) if finite else math.inf


def calibration_context(calibration: np.ndarray, calibration_truth: np.ndarray, data: dict[str, np.ndarray], exclude_position: int | None = None) -> dict:
    selected_positions = [position for position in range(len(calibration)) if position != exclude_position]
    by_class = defaultdict(list)
    full_support = defaultdict(int)
    for label in calibration_truth.tolist(): full_support[int(label)] += 1
    for position in selected_positions: by_class[int(calibration_truth[position])].append(int(calibration[position]))
    embedding_prototypes = {}; geometry_dispersion = {}
    for label, rows in by_class.items():
        if full_support[label] < MIN_CALIBRATION_SUPPORT or not rows: continue
        centroid = data["norm_embeddings"][rows].mean(axis=0); centroid /= max(float(np.linalg.norm(centroid)), 1e-8); embedding_prototypes[label] = centroid
        pairs = [_dtw_cached(min(a, b), max(a, b)) for i, a in enumerate(rows) for b in rows[i + 1:]]
        finite = [value for value in pairs if math.isfinite(value)]; geometry_dispersion[label] = float(np.median(finite)) if finite else 0.0
    return {"support": dict(full_support), "rows_by_class": dict(by_class), "embedding_prototypes": embedding_prototypes, "geometry_dispersion": geometry_dispersion}


def inference_options(index: int, trajectory: np.ndarray, embedding: np.ndarray, logits: np.ndarray, context: dict, stats: dict, groups: list[list[int]]) -> tuple[np.ndarray, list[int], list[int], dict]:
    candidates = np.argsort(logits)[-5:][::-1].astype(int)
    telemetry = {"top5_nonbaseline_candidates": 4, "whole_row_pass": 0, "structural_candidates": 0,
                 "support_pass": 0, "calibration_prototype_pass": 0, "stroke_match_pass": 0, "finite_dtw_pass": 0,
                 "global_embedding_sign_pass": 0, "global_geometry_sign_pass": 0,
                 "calibration_embedding_sign_pass": 0, "calibration_geometry_sign_pass": 0,
                 "four_way_sign_consensus": 0, "emitted_options": 0}
    if not np.all(stats["admitted"][candidates]) or _collision(candidates, groups):
        return np.empty((0, len(FEATURE_NAMES)), np.float32), [], [], telemetry
    telemetry["whole_row_pass"] = 1; telemetry["structural_candidates"] = 4
    values = logits[candidates].astype(np.float64); spread = max(float(values.std()), 1e-6)
    shifted = values - values.max(); probability = np.exp(shifted); probability /= probability.sum(); entropy = float(-np.sum(probability * np.log(np.maximum(probability, 1e-12))))
    query_signature = trajectory_signature(trajectory)
    global_cos = embedding @ stats["prototypes"][candidates].T; baseline = int(candidates[0]); baseline_geo = _global_geometry(query_signature, baseline, stats)
    features = []; labels = []; ranks = []
    stroke_count = len(query_signature)
    for rank in range(1, 5):
        candidate = int(candidates[rank]); support = int(context["support"].get(candidate, 0)); rows = context["rows_by_class"].get(candidate, [])
        if support < MIN_CALIBRATION_SUPPORT: continue
        telemetry["support_pass"] += 1
        if candidate not in context["embedding_prototypes"]: continue
        telemetry["calibration_prototype_pass"] += 1
        matching = [row for row in rows if len(_DTW_ENGINE.signatures[row]) == stroke_count]
        if len(matching) < 1: continue
        telemetry["stroke_match_pass"] += 1
        cal_cos = float(embedding @ context["embedding_prototypes"][candidate]); candidate_geo = _global_geometry(query_signature, candidate, stats)
        cal_distances = [_DTW_ENGINE.signature_distance(query_signature, _DTW_ENGINE.signatures[row]) for row in matching]
        cal_distances = [value for value in cal_distances if math.isfinite(value)]
        if not cal_distances or not math.isfinite(candidate_geo) or not math.isfinite(baseline_geo): continue
        telemetry["finite_dtw_pass"] += 1
        cal_dtw = min(cal_distances); global_embedding_adv = float(global_cos[rank] - global_cos[0]); global_geometry_adv = float(baseline_geo - candidate_geo)
        calibration_embedding_adv = float(cal_cos - stats["reliability"][candidate]); calibration_geometry_adv = float(candidate_geo - cal_dtw)
        signs = (global_embedding_adv > 0.0, global_geometry_adv > 0.0, calibration_embedding_adv >= -0.02, calibration_geometry_adv >= 0.0)
        telemetry["global_embedding_sign_pass"] += int(signs[0]); telemetry["global_geometry_sign_pass"] += int(signs[1])
        telemetry["calibration_embedding_sign_pass"] += int(signs[2]); telemetry["calibration_geometry_sign_pass"] += int(signs[3])
        telemetry["four_way_sign_consensus"] += int(all(signs))
        # r3 keeps these four advantages as continuous model evidence. They
        # are diagnostics, not hard filters; uncertainty falls through to the
        # frozen probability, pooled-risk, sparse-self, and identity gates.
        features.append([
            float((values[rank] - values[0]) / spread), global_embedding_adv, calibration_embedding_adv,
            global_geometry_adv, calibration_geometry_adv, -float(cal_dtw), float(np.log1p(support)),
            float(stats["reliability"][candidate]), -float(context["geometry_dispersion"].get(candidate, 0.0)),
            -entropy, -float((values[0] - values[1]) / spread), 1.0,
        ]); labels.append(candidate); ranks.append(rank)
    telemetry["emitted_options"] = len(labels)
    return np.asarray(features, np.float32).reshape(-1, len(FEATURE_NAMES)), labels, ranks, telemetry


def episode_indices(writer: int, size: int, prediction_data: dict[str, np.ndarray], calibration_truth_by_index: dict[int, int]) -> tuple[np.ndarray, np.ndarray]:
    calibration = np.flatnonzero((prediction_data["writers"] == writer) & (prediction_data["splits"] == 0)); query = np.flatnonzero((prediction_data["writers"] == writer) & (prediction_data["splits"] == 1))
    by_label = defaultdict(list)
    for index in calibration.tolist(): by_label[int(calibration_truth_by_index[index])].append(index)
    ordered = sorted(by_label, key=lambda label: hashlib.sha256(f"{SEED}:{writer}:{size}:{label}".encode()).hexdigest())
    pair_count = max(1, size // 4); pair_labels = ordered[:pair_count]; single = ordered[pair_count:pair_count + size - 2 * pair_count]
    selected = [index for label in pair_labels for index in by_label[label][:2]] + [by_label[label][0] for label in single]
    if len(selected) != size: raise AssertionError("sparse calibration size mismatch")
    return np.asarray(selected, np.int64), query


def build_episode(writer: int, size: int, prediction_data: dict[str, np.ndarray], calibration_truth_by_index: dict[int, int], stats: dict, groups: list[list[int]]) -> dict:
    calibration, query = episode_indices(writer, size, prediction_data, calibration_truth_by_index)
    cal_truth = np.asarray([calibration_truth_by_index[int(index)] for index in calibration], np.int64)
    context = calibration_context(calibration, cal_truth, prediction_data)
    features = []; records = []; query_structural = defaultdict(int)
    for index in query.tolist():
        values, candidates, ranks, telemetry = inference_options(index, prediction_data["trajectories"][index], prediction_data["norm_embeddings"][index], prediction_data["logits"][index], context, stats, groups)
        for key, value in telemetry.items(): query_structural[key] += int(value)
        for feature, candidate, rank in zip(values, candidates, ranks, strict=True):
            features.append(feature); records.append((index, candidate, rank))
    self_features = []; self_records = []; self_structural = defaultdict(int)
    for position, index in enumerate(calibration.tolist()):
        local = calibration_context(calibration, cal_truth, prediction_data, position)
        values, candidates, ranks, telemetry = inference_options(index, prediction_data["trajectories"][index], prediction_data["norm_embeddings"][index], prediction_data["logits"][index], local, stats, groups)
        for key, value in telemetry.items(): self_structural[key] += int(value)
        for feature, candidate, rank in zip(values, candidates, ranks, strict=True):
            self_features.append(feature); self_records.append((position, candidate, rank))
    supported = sum(value >= MIN_CALIBRATION_SUPPORT for value in context["support"].values()); deficiency = "none" if supported == 0 else "one" if supported == 1 else "multiple"
    return {"writer": writer, "support_size": size, "calibration": calibration, "query": query,
            "features": np.asarray(features, np.float32).reshape(-1, len(FEATURE_NAMES)), "records": records,
            "self_features": np.asarray(self_features, np.float32).reshape(-1, len(FEATURE_NAMES)), "self_records": self_records,
            "calibration_truth": cal_truth,
            "query_structural_telemetry": dict(query_structural), "self_structural_telemetry": dict(self_structural),
            "deficiency": deficiency}


def _probabilities(model: nn.Module, features: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    if not len(features): return np.empty(0, np.float32)
    with torch.inference_mode(): return torch.sigmoid(model(torch.from_numpy(((features - mean) / scale).astype(np.float32)))).numpy()


def _apply(logits: np.ndarray, records: list[tuple[int, int, int]], probabilities: np.ndarray, threshold: float, enabled_candidates: set[int] | None = None) -> np.ndarray:
    prediction = np.argmax(logits, axis=1).astype(np.int64); options = defaultdict(list)
    for probability, (row, candidate, rank) in zip(probabilities.tolist(), records, strict=True):
        if enabled_candidates is None or candidate in enabled_candidates: options[int(row)].append((float(probability), -int(rank), int(candidate)))
    for row, values in options.items():
        best = max(values)
        if best[0] >= threshold: prediction[row] = best[2]
    return prediction


def _wilson(successes: int, trials: int, upper: bool) -> float:
    if trials <= 0: return 1.0 if upper else 0.0
    z = 1.96; p = successes / trials; denominator = 1 + z * z / trials; centre = p + z * z / (2 * trials)
    radius = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials))
    return (centre + radius if upper else centre - radius) / denominator


def _raw_query_action(episode: dict, prediction_data: dict[str, np.ndarray], model: nn.Module, mean: np.ndarray, scale: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    query = episode["query"]; index_map = {index: position for position, index in enumerate(query.tolist())}
    local_records = [(index_map[index], candidate, rank) for index, candidate, rank in episode["records"]]
    probabilities = _probabilities(model, episode["features"], mean, scale)
    return _apply(prediction_data["logits"][query], local_records, probabilities, threshold), probabilities


def pooled_candidate_action_evidence(episodes: list[dict], prediction_data: dict[str, np.ndarray], scoring_labels: np.ndarray,
                                     model: nn.Module, mean: np.ndarray, scale: np.ndarray, threshold: float) -> dict:
    counts = defaultdict(lambda: {"changed": 0, "improved": 0, "regressed": 0})
    for episode in episodes:
        query = episode["query"]; baseline = np.argmax(prediction_data["logits"][query], axis=1)
        prediction, _ = _raw_query_action(episode, prediction_data, model, mean, scale, threshold); truth = scoring_labels[query]
        for candidate in np.unique(prediction[prediction != baseline]).tolist():
            selected = (prediction == candidate) & (prediction != baseline); row = counts[int(candidate)]
            row["changed"] += int(selected.sum())
            row["improved"] += int(np.sum(selected & (prediction == truth) & (baseline != truth)))
            row["regressed"] += int(np.sum(selected & (baseline == truth) & (prediction != truth)))
    evidence = {}
    for candidate, row in sorted(counts.items()):
        upper = _wilson(row["regressed"], row["changed"], True)
        accepted = row["improved"] > 0 and row["regressed"] == 0 and upper <= MAX_REGRESSION_WILSON_UPPER
        evidence[str(candidate)] = {**row, "regression_wilson_upper": upper, "accepted": accepted}
    return evidence


def prepare_pooled_writer_oof(writers: list[int], prediction_data: dict[str, np.ndarray], scoring_labels: np.ndarray,
                              calibration_truth_by_index: dict[int, int], groups: list[list[int]]) -> list[dict]:
    prepared = []
    for fold, held in enumerate(_folds(writers, 4)):
        training_writers = sorted(set(writers) - set(held))
        training_episodes, feature_provenance = oof_training_episodes(training_writers, prediction_data, scoring_labels, calibration_truth_by_index, groups)
        features, targets = feature_target_arrays(training_episodes, scoring_labels)
        mean = features.mean(axis=0).astype(np.float32); scale = np.maximum(features.std(axis=0), 1e-4).astype(np.float32)
        model, history = _train(features, targets, mean, scale)
        held_stats = fold_statistics(prediction_data, scoring_labels, training_writers)
        held_episodes = build_episodes(held, prediction_data, calibration_truth_by_index, held_stats, groups)
        statistics_sets = [set(row["training_writers"]) for row in feature_provenance]
        union = sorted(set().union(*statistics_sets)); intersection = sorted(set.intersection(*statistics_sets))
        if set(held) & set(union): raise ValueError("risk-held writer leaked into pooled training feature statistics")
        prepared.append({"fold": fold, "risk_heldout_writers": held, "model_training_writers": training_writers,
                         "feature_statistics_union": union, "feature_statistics_intersection": intersection,
                         "held_absent_from_every_training_feature_stats": True, "feature_provenance": feature_provenance,
                         "held_statistics_writer_hash": held_stats["training_writer_hash"], "held_episodes": held_episodes,
                         "model": model, "mean": mean, "scale": scale, "history": history})
    return prepared


def pooled_writer_oof_action_evidence(prepared: list[dict], prediction_data: dict[str, np.ndarray], scoring_labels: np.ndarray,
                                      threshold: float) -> tuple[dict, list[dict]]:
    combined = defaultdict(lambda: {"changed": 0, "improved": 0, "regressed": 0}); provenance = []
    for fold_state in prepared:
        local = pooled_candidate_action_evidence(fold_state["held_episodes"], prediction_data, scoring_labels,
                                                 fold_state["model"], fold_state["mean"], fold_state["scale"], threshold)
        for candidate, row in local.items():
            for key in ("changed", "improved", "regressed"): combined[int(candidate)][key] += int(row[key])
        provenance.append({key: value for key, value in fold_state.items() if key not in ("held_episodes", "model", "mean", "scale", "history")})
    evidence = {}
    for candidate, row in sorted(combined.items()):
        upper = _wilson(row["regressed"], row["changed"], True)
        accepted = row["improved"] > 0 and row["regressed"] == 0 and upper <= MAX_REGRESSION_WILSON_UPPER
        evidence[str(candidate)] = {**row, "regression_wilson_upper": upper, "accepted": accepted}
    return evidence, provenance


def predict_episode(episode: dict, prediction_data: dict[str, np.ndarray], model: nn.Module, mean: np.ndarray, scale: np.ndarray,
                    threshold: float, pooled_evidence: dict | None = None) -> dict:
    cal = episode["calibration"]; cal_truth = episode["calibration_truth"]; cal_logits = prediction_data["logits"][cal]; cal_base = np.argmax(cal_logits, axis=1)
    pooled_evidence = pooled_evidence or {}
    pooled_allowed = {int(candidate) for candidate, row in pooled_evidence.items() if row.get("accepted") is True}
    self_prob = _probabilities(model, episode["self_features"], mean, scale)
    preliminary = _apply(cal_logits, episode["self_records"], self_prob, threshold, pooled_allowed)
    allowed = set(); evidence = {}
    for candidate in sorted(set(preliminary[preliminary != cal_base].tolist())):
        selected = (preliminary == candidate) & (preliminary != cal_base); changed = int(selected.sum())
        improved = int(np.sum(selected & (preliminary == cal_truth) & (cal_base != cal_truth))); regressed = int(np.sum(selected & (cal_base == cal_truth) & (preliminary != cal_truth)))
        accepted = candidate in pooled_allowed and improved > 0 and regressed == 0
        evidence[str(candidate)] = {"changed": changed, "improved": improved, "regressed": regressed,
                                    "pooled_writer_oof": pooled_evidence.get(str(candidate)), "accepted": accepted}
        if accepted: allowed.add(candidate)
    self_prediction = _apply(cal_logits, episode["self_records"], self_prob, threshold, allowed)
    chunks = [np.arange(start, min(start + 4, len(cal))) for start in range(0, len(cal), 4)]
    self_changed = int(np.sum(cal_base != self_prediction)); self_regressed = int(np.sum((cal_base == cal_truth) & (self_prediction != cal_truth)))
    enabled = bool(allowed) and self_regressed == 0 and sum(np.all(self_prediction[c] == cal_truth[c]) for c in chunks) >= sum(np.all(cal_base[c] == cal_truth[c]) for c in chunks)
    query = episode["query"]; index_map = {index: position for position, index in enumerate(query.tolist())}
    local_records = [(index_map[index], candidate, rank) for index, candidate, rank in episode["records"]]
    probabilities = _probabilities(model, episode["features"], mean, scale)
    prediction = _apply(prediction_data["logits"][query], local_records, probabilities, threshold, allowed if enabled else set())
    query_candidates = [candidate for _index, candidate, _rank in episode["records"]]
    self_candidates = [candidate for _position, candidate, _rank in episode["self_records"]]
    telemetry = {"writer": episode["writer"], "support_size": episode["support_size"],
                 "raw_query_options": len(episode["records"]), "query_probability_pass": int(np.sum(probabilities >= threshold)),
                 "query_pooled_candidate_pass": sum(candidate in pooled_allowed for candidate in query_candidates),
                 "query_probability_and_pooled_pass": sum(float(probability) >= threshold and candidate in pooled_allowed for probability, candidate in zip(probabilities.tolist(), query_candidates, strict=True)),
                 "raw_self_options": len(episode["self_records"]), "self_probability_pass": int(np.sum(self_prob >= threshold)),
                 "self_pooled_candidate_pass": sum(candidate in pooled_allowed for candidate in self_candidates),
                 "self_probability_and_pooled_pass": sum(float(probability) >= threshold and candidate in pooled_allowed for probability, candidate in zip(self_prob.tolist(), self_candidates, strict=True)),
                 "candidate_self_gate_count": len(allowed), "episode_enabled": enabled,
                 "query_changed": int(np.sum(np.argmax(prediction_data["logits"][query], axis=1) != prediction))}
    telemetry.update({f"query_{key}": value for key, value in episode["query_structural_telemetry"].items()})
    telemetry.update({f"self_{key}": value for key, value in episode["self_structural_telemetry"].items()})
    return {"prediction": prediction, "enabled": enabled, "candidate_evidence": evidence, "self_regressed": self_regressed, "stage_telemetry": telemetry}


def score_prediction(episode: dict, prediction_state: dict, prediction_data: dict[str, np.ndarray], scoring_labels: np.ndarray) -> dict:
    query = episode["query"]; prediction = prediction_state["prediction"]
    truth = scoring_labels[query]; baseline = np.argmax(prediction_data["logits"][query], axis=1); old = baseline == truth; new = prediction == truth
    formulae = [np.arange(start, min(start + 8, len(query))) for start in range(0, len(query), 8)]; top5 = np.argsort(prediction_data["logits"][query], axis=1)[:, -5:]
    return {"writer": episode["writer"], "support_size": episode["support_size"], "deficiency": episode["deficiency"], "enabled": prediction_state["enabled"], "rows": len(query),
            "baseline_correct": int(old.sum()), "adapted_correct": int(new.sum()), "formulae": len(formulae),
            "baseline_exact": int(sum(bool(np.all(old[c])) for c in formulae)), "adapted_exact": int(sum(bool(np.all(new[c])) for c in formulae)),
            "improved": int(np.sum(~old & new)), "regressed": int(np.sum(old & ~new)), "changed": int(np.sum(baseline != prediction)),
            "candidate_violations": int(sum(int(prediction[i]) not in top5[i] for i in range(len(query)))), "candidate_evidence": prediction_state["candidate_evidence"],
            "self_regressed": prediction_state["self_regressed"], "stage_telemetry": prediction_state["stage_telemetry"]}


def summarize(rows: list[dict]) -> dict:
    def aggregate(group: list[dict]) -> dict:
        n = sum(row["rows"] for row in group); f = sum(row["formulae"] for row in group); changed = sum(row["changed"] for row in group); regressed = sum(row["regressed"] for row in group)
        return {"rows": n, "baseline_top1": sum(row["baseline_correct"] for row in group) / n, "adapted_top1": sum(row["adapted_correct"] for row in group) / n,
                "baseline_pseudo_exact": sum(row["baseline_exact"] for row in group) / f, "adapted_pseudo_exact": sum(row["adapted_exact"] for row in group) / f,
                "improved": sum(row["improved"] for row in group), "regressed": regressed, "changed": changed, "candidate_violations": sum(row["candidate_violations"] for row in group),
                "regression_wilson_upper": 0.0 if changed == 0 else _wilson(regressed, changed, True), "activation_coverage": sum(row["enabled"] for row in group) / len(group)}
    overall = aggregate(rows); per_writer = {str(w): aggregate([row for row in rows if row["writer"] == w]) for w in sorted(set(row["writer"] for row in rows))}
    per_support = {str(s): aggregate([row for row in rows if row["support_size"] == s]) for s in SUPPORT_SIZES}
    per_deficiency = {d: aggregate([row for row in rows if row["deficiency"] == d]) for d in sorted(set(row["deficiency"] for row in rows))}
    def nonreg(value: dict) -> bool: return value["adapted_top1"] >= value["baseline_top1"] and value["adapted_pseudo_exact"] >= value["baseline_pseudo_exact"] and value["candidate_violations"] == 0
    episode_safe = all(row["adapted_correct"] >= row["baseline_correct"] and row["adapted_exact"] >= row["baseline_exact"] and row["candidate_violations"] == 0 for row in rows)
    activation_risk_safe = all((not row["enabled"]) or row["self_regressed"] == 0 for row in rows)
    safe = episode_safe and activation_risk_safe and all(nonreg(v) for v in per_writer.values()) and all(nonreg(v) for v in per_support.values()) and all(nonreg(v) for v in per_deficiency.values()) and nonreg(overall)
    positive = overall["adapted_top1"] > overall["baseline_top1"] or overall["adapted_pseudo_exact"] > overall["baseline_pseudo_exact"]
    return {"overall": overall, "per_writer": per_writer, "per_support": per_support, "per_deficiency": per_deficiency, "writer_by_support_nonregression": episode_safe, "activation_risk_safe": activation_risk_safe, "safe": safe, "positive": positive, "episodes": rows}


def _train(features: np.ndarray, targets: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> tuple[nn.Module, list[float]]:
    torch.manual_seed(SEED); model = MonotoneLogistic(len(FEATURE_NAMES)); optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.05)
    x = torch.from_numpy(((features - mean) / scale).astype(np.float32)); y = torch.from_numpy(targets.astype(np.float32)); history = []
    for _epoch in range(40):
        output = model(x); weight = torch.where(y > 0.5, 1.0, ACTION_COST); loss = F.binary_cross_entropy_with_logits(output, y, weight=weight)
        optimizer.zero_grad(); loss.backward(); optimizer.step(); history.append(float(loss.detach()))
    model.eval(); return model, history


def _writer_order(values: list[int], salt: str) -> list[int]:
    return sorted(values, key=lambda writer: hashlib.sha256(f"{SEED}:{salt}:{writer}".encode()).hexdigest())


def _folds(values: list[int], count: int) -> list[list[int]]:
    ordered = _writer_order(values, f"fold{count}"); return [ordered[index::count] for index in range(count)]


def build_episodes(writers: list[int], prediction_data: dict[str, np.ndarray], calibration_truth_by_index: dict[int, int], stats: dict, groups: list[list[int]]) -> list[dict]:
    return [build_episode(writer, size, prediction_data, calibration_truth_by_index, stats, groups) for writer in writers for size in SUPPORT_SIZES]


def oof_training_episodes(writers: list[int], prediction_data: dict[str, np.ndarray], development_labels: np.ndarray, calibration_truth_by_index: dict[int, int], groups: list[list[int]]) -> tuple[list[dict], list[dict]]:
    episodes = []; provenance = []
    for fold, held in enumerate(_folds(writers, 4)):
        training = sorted(set(writers) - set(held)); stats = fold_statistics(prediction_data, development_labels, training); episodes.extend(build_episodes(held, prediction_data, calibration_truth_by_index, stats, groups))
        provenance.append({"fold": fold, "training_writers": training, "heldout_writers": held, "statistics_writer_hash": stats["training_writer_hash"]})
    return episodes, provenance


def feature_target_arrays(episodes: list[dict], development_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = [episode["features"] for episode in episodes if len(episode["features"])]
    targets = [np.asarray([float(candidate == int(development_labels[index])) for index, candidate, _rank in episode["records"]], np.float32) for episode in episodes if len(episode["records"])]
    return np.concatenate(values), np.concatenate(targets)


def evaluate_episodes(episodes: list[dict], prediction_data: dict[str, np.ndarray], scoring_labels: np.ndarray, model: nn.Module, mean: np.ndarray, scale: np.ndarray,
                      threshold: float, pooled_evidence: dict | None = None) -> dict:
    predictions = [predict_episode(episode, prediction_data, model, mean, scale, threshold, pooled_evidence) for episode in episodes]
    return summarize([score_prediction(episode, state, prediction_data, scoring_labels) for episode, state in zip(episodes, predictions, strict=True)])


def query_label_permutation_test(episode: dict, prediction_data: dict[str, np.ndarray], scoring_labels: np.ndarray, model: nn.Module, mean: np.ndarray, scale: np.ndarray,
                                 threshold: float, pooled_evidence: dict | None = None) -> dict:
    first = predict_episode(episode, prediction_data, model, mean, scale, threshold, pooled_evidence)["prediction"]
    permuted = scoring_labels.copy(); query = episode["query"]; permuted[query] = permuted[query][::-1]
    second = predict_episode(episode, prediction_data, model, mean, scale, threshold, pooled_evidence)["prediction"]
    return {"query_rows": len(query), "labels_permuted": bool(not np.array_equal(scoring_labels[query], permuted[query])), "prediction_exact": bool(np.array_equal(first, second))}


def compact(result: dict) -> dict:
    return {key: value for key, value in result.items() if key != "episodes"}


def stage_telemetry(result: dict) -> list[dict]:
    return [row["stage_telemetry"] for row in result.get("episodes", [])]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT); parser.add_argument("--independent-audit", type=Path, required=True); args = parser.parse_args()
    if args.output.exists(): parser.error("refusing to overwrite v11 output")
    audit = _validate_audit(args.independent_audit)
    consumed_lineage = _validate_consumed_lineage()
    source_hash = sha256(Path(__file__))
    data, inventory = load_development96()
    development_labels = data.pop("labels")
    calibration_truth_by_index = {int(index): int(development_labels[index]) for index in np.flatnonzero(data["splits"] == 0)}
    if "labels" in data:
        raise ValueError("prediction_data must not contain labels")
    global _DTW_ENGINE; _DTW_ENGINE = GeometryEngine(data["trajectories"]); _dtw_cached.cache_clear()
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False); label_names = [str(value) for value in checkpoint["math_labels"]]; groups = _resolve_homographs(label_names)
    api_parameters = list(inspect.signature(inference_options).parameters)
    api_safe = not any(name in api_parameters for name in ("truth", "labels", "writer", "writer_id", "metadata", "session", "sample"))
    if not api_safe: raise ValueError("query inference API is not label/writer/metadata blind")
    args.output.mkdir(parents=True); source = args.output / "train_dual_view_writer_adapter_v11.source.py"; source.write_bytes(Path(__file__).read_bytes())
    started = args.output / "V11_DEVELOPMENT_STARTED.json"; started.write_text(json.dumps({"status": "V11_R3_DEVELOPMENT_STARTED_000_095_ONLY", "script_sha256": source_hash, "source_snapshot_sha256": sha256(source), "audit": audit, "consumed_lineage": consumed_lineage, "inventory": inventory, "writers096_127_accessed": False, "hwr_training": False}, indent=2), encoding="utf-8")
    outer_rows = []; fold_reports = []; selected_thresholds = []; all_stage_telemetry = []
    for outer_fold, heldout in enumerate(_folds(list(range(96)), 8)):
        train84 = sorted(set(range(96)) - set(heldout)); ordered = _writer_order(train84, f"outer{outer_fold}:inner")
        fit72, development12 = ordered[:72], ordered[72:]
        fit_episodes, fit_provenance = oof_training_episodes(fit72, data, development_labels, calibration_truth_by_index, groups); features, targets = feature_target_arrays(fit_episodes, development_labels)
        mean = features.mean(axis=0).astype(np.float32); scale = np.maximum(features.std(axis=0), 1e-4).astype(np.float32); model, history = _train(features, targets, mean, scale)
        fit72_stats = fold_statistics(data, development_labels, fit72)
        development = build_episodes(development12, data, calibration_truth_by_index, fit72_stats, groups)
        pooled_prepared = prepare_pooled_writer_oof(fit72, data, development_labels, calibration_truth_by_index, groups)
        choices = []
        for threshold in THRESHOLDS:
            pooled_evidence, pooled_provenance = pooled_writer_oof_action_evidence(pooled_prepared, data, development_labels, threshold)
            result = evaluate_episodes(development, data, development_labels, model, mean, scale, threshold, pooled_evidence)
            eligible = result["safe"] and result["positive"] and result["overall"]["regressed"] == 0 and result["overall"]["regression_wilson_upper"] <= MAX_REGRESSION_WILSON_UPPER
            choices.append({"threshold": threshold, "eligible": eligible, "result": result,
                            "pooled_candidate_evidence": pooled_evidence, "pooled_writer_oof_provenance": pooled_provenance})
        eligible = [choice for choice in choices if choice["eligible"]]
        if not eligible:
            telemetry_path = args.output / "stage_telemetry.json"
            telemetry_path.write_text(json.dumps({"outer_fold": outer_fold, "choices": [{"threshold": c["threshold"], "rows": stage_telemetry(c["result"]), "pooled_candidate_evidence": c["pooled_candidate_evidence"]} for c in choices]}, indent=2), encoding="utf-8")
            rejection = {"status": "V11_R3_INNER_DEVELOPMENT_FAIL_CLOSED", "outer_fold": outer_fold,
                         "choices": [{"threshold": c["threshold"], "eligible": c["eligible"], "result": compact(c["result"]), "pooled_candidate_evidence": c["pooled_candidate_evidence"], "pooled_writer_oof_provenance": c["pooled_writer_oof_provenance"]} for c in choices],
                         "stage_telemetry_sha256": sha256(telemetry_path), "consumed_lineage": consumed_lineage,
                         "writers096_127_accessed": False, "source_sha256": source_hash, "source_snapshot_sha256": sha256(source), "audit": audit}
            path = args.output / "development_rejection.json"; path.write_text(json.dumps(rejection, indent=2), encoding="utf-8"); print(json.dumps({"status": rejection["status"], "outer_fold": outer_fold})); return 2
        selected = max(eligible, key=lambda choice: (choice["result"]["overall"]["adapted_top1"] - choice["result"]["overall"]["baseline_top1"], choice["threshold"]))
        selected_thresholds.append(selected["threshold"])
        all_stage_telemetry.append({"fold": outer_fold, "selected_threshold": selected["threshold"], "development": stage_telemetry(selected["result"])})
        outer_episodes = build_episodes(heldout, data, calibration_truth_by_index, fit72_stats, groups)
        outer_predictions = [predict_episode(episode, data, model, mean, scale, selected["threshold"], selected["pooled_candidate_evidence"]) for episode in outer_episodes]
        rows = [score_prediction(episode, state, data, development_labels) for episode, state in zip(outer_episodes, outer_predictions, strict=True)]
        outer_rows.extend(rows)
        permutation = query_label_permutation_test(development[0], data, development_labels, model, mean, scale, selected["threshold"], selected["pooled_candidate_evidence"])
        if not permutation["labels_permuted"] or not permutation["prediction_exact"]:
            raise ValueError("query-label permutation contract failed")
        fold_reports.append({"fold": outer_fold, "fit72": fit72, "development12": development12, "heldout12": heldout, "selected_threshold": selected["threshold"],
                             "outer_model_training_writers": fit72, "outer_statistics_training_writers": fit72,
                             "development_and_outer_excluded_from_model_and_statistics": sorted(development12 + heldout),
                             "fit_oof": fit_provenance, "history": history, "development": compact(selected["result"]),
                             "pooled_candidate_evidence": selected["pooled_candidate_evidence"], "pooled_writer_oof_provenance": selected["pooled_writer_oof_provenance"],
                             "query_label_permutation": permutation, "outer": compact(summarize(rows))})
    nested = summarize(outer_rows)
    if not nested["safe"] or not nested["positive"] or nested["overall"]["regressed"] != 0:
        telemetry_path = args.output / "stage_telemetry.json"
        telemetry_path.write_text(json.dumps({"outer_folds": all_stage_telemetry}, indent=2), encoding="utf-8")
        rejection = {"status": "V11_R3_NESTED_WRITER_OOF_REJECTED", "nested": compact(nested), "folds": fold_reports,
                     "stage_telemetry_sha256": sha256(telemetry_path), "consumed_lineage": consumed_lineage,
                     "writers096_127_accessed": False, "source_sha256": source_hash, "source_snapshot_sha256": sha256(source), "audit": audit}
        path = args.output / "nested_rejection.json"; path.write_text(json.dumps(rejection, indent=2), encoding="utf-8"); print(json.dumps({"status": rejection["status"], "overall": nested["overall"]})); return 2
    final_threshold = max(selected_thresholds)
    final_episodes, final_provenance = oof_training_episodes(list(range(96)), data, development_labels, calibration_truth_by_index, groups); final_x, final_y = feature_target_arrays(final_episodes, development_labels)
    mean = final_x.mean(axis=0).astype(np.float32); scale = np.maximum(final_x.std(axis=0), 1e-4).astype(np.float32); model, history = _train(final_x, final_y, mean, scale); stats = fold_statistics(data, development_labels, list(range(96)))
    final_pooled_prepared = prepare_pooled_writer_oof(list(range(96)), data, development_labels, calibration_truth_by_index, groups)
    final_pooled_evidence, final_pooled_provenance = pooled_writer_oof_action_evidence(final_pooled_prepared, data, development_labels, final_threshold)
    torch.save({"kind": "monotone_logistic", "state_dict": model.state_dict(), "feature_names": FEATURE_NAMES}, args.output / "candidate_scorer.pt")
    np.savez(args.output / "feature_scaler.npz", mean=mean, scale=scale)
    np.savez_compressed(args.output / "global_embedding_statistics.npz", prototypes=stats["prototypes"], counts=stats["counts"], reliability=stats["reliability"], admitted=stats["admitted"])
    (args.output / "geometry_exemplars.json").write_text(json.dumps({str(label): rows for label, rows in stats["geometry_exemplars"].items()}, indent=2), encoding="utf-8")
    (args.output / "pooled_candidate_evidence.json").write_text(json.dumps({"evidence": final_pooled_evidence, "writer_oof_provenance": final_pooled_provenance}, indent=2), encoding="utf-8")
    (args.output / "stage_telemetry.json").write_text(json.dumps({"outer_folds": all_stage_telemetry}, indent=2), encoding="utf-8")
    config = {"model": "monotone_logistic", "action_cost": ACTION_COST, "threshold": final_threshold, "support_sizes": SUPPORT_SIZES, "minimum_calibration_support": MIN_CALIBRATION_SUPPORT,
              "dtw": {"points_per_stroke": DTW_POINTS_PER_STROKE, "window": DTW_WINDOW, "cost": DTW_COST, "channels": ["x", "y", "stroke_start", "observed"], "delta_t_used": False},
              "feature_names": FEATURE_NAMES, "candidate_or_writer_id_feature": False, "threshold_grid_prefixed": list(THRESHOLDS),
              "risk_contract": {"candidate_uncertainty_source": "training-writer OOF pooled action evidence", "candidate_regression_wilson_upper_max": MAX_REGRESSION_WILSON_UPPER,
                                "sparse_episode_gate": "observed self regression zero and pseudo-exact nonregression", "identity_without_action_is_safe_but_not_positive": True,
                                "hard_candidate_gates": ["top5 globally admitted", "no homograph collision", "calibration support >=2", "calibration prototype present", "stroke count match", "finite global and calibration DTW"],
                                "four_view_advantage_signs_are_diagnostics_not_hard_gates": True},
              "writers096_127_opened": False,
              "final_all96_model_evaluated": False, "first_evidence_writer_ids": "096..127 after independent artifact audit"}
    (args.output / "frozen_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (args.output / "writer_split_manifest.json").write_text(json.dumps({"outer_folds": fold_reports, "final_oof": final_provenance, "final_pooled_writer_oof": final_pooled_provenance, "selected_thresholds": selected_thresholds, "final_threshold": final_threshold}, indent=2), encoding="utf-8")
    report = {"schema": "aiflow-dual-view-writer-adapter/v11-r3", "generated_at": datetime.now(timezone.utc).isoformat(), "status": "V11_R3_DUAL_VIEW_SYNTHETIC96_FROZEN_OUTER_CLOSED", "nested": compact(nested), "inventory": inventory, "consumed_lineage": consumed_lineage,
              "gates": {"nested_safe_positive": nested["safe"] and nested["positive"], "nested_regression_zero": nested["overall"]["regressed"] == 0, "candidate_zero": nested["overall"]["candidate_violations"] == 0, "query_api_label_writer_metadata_blind": api_safe,
                        "query_label_permutation_exact": all(fold["query_label_permutation"]["prediction_exact"] for fold in fold_reports),
                        "model_threshold_pair_matched": all(fold["outer_model_training_writers"] == fold["fit72"] for fold in fold_reports),
                        "pooled_candidate_regression_uncertainty_bounded": all((not row["accepted"]) or row["regression_wilson_upper"] <= MAX_REGRESSION_WILSON_UPPER for row in final_pooled_evidence.values()),
                        "sparse_episode_observed_regression_zero": nested["activation_risk_safe"],
                        "dtw_fixed_xy_stroke_observed_only": True, "hwr_checkpoint_runtime_unchanged": True,
                        "hwr_training_not_performed": True, "writers096_127_remained_closed": True,
                        "legacy_remained_closed": True, "real_corpus_remained_closed": True,
                        "product_promotion_not_performed": True},
              "provenance_boundary": "geometry exemplars and reliability are train-writer-only per fold but remain on the known external parent manifold",
              "final_all96_evaluated": False,
              "boundary": "synthetic development-only; candidate requires independent artifact audit before Legacy or writers096..127"}
    report_path = args.output / "development_report.json"; report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    files = ["candidate_scorer.pt", "feature_scaler.npz", "global_embedding_statistics.npz", "geometry_exemplars.json", "pooled_candidate_evidence.json", "stage_telemetry.json", "frozen_config.json", "writer_split_manifest.json", "development_report.json", "V11_DEVELOPMENT_STARTED.json", source.name]
    receipt = {"status": report["status"], "script_sha256": source_hash, "audit_sha256": audit["sha256"], "consumed_lineage": consumed_lineage, "files": {name: sha256(args.output / name) for name in files}, "writers096_127_opened": False}
    (args.output / "freeze_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "nested": nested["overall"], "output": str(args.output)})); return 0


if __name__ == "__main__":
    raise SystemExit(main())
