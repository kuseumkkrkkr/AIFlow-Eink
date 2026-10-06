#!/usr/bin/env python3
"""Paired writer-heldout probes for CE, contrastive, and candidate-preserving objectives.

Both arms use the same 128-d Transformer, initialization, class-balanced P×K
batches, rows, and training exposure. This isolates an explicit embedding-level
competition term from ordinary 372-way softmax classification.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Iterator

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Sampler

from build_normalized_ink_v1 import ROOT, _sha256
from audit_hwr_architecture_capacity_paired_v1 import (
    _jsonl_train_uji_writer_hashes,
    _paired,
    _writer_bootstrap,
)
from run_hwr_architecture_capacity_probe_v1 import (
    DEFAULT_CANONICAL_ROOT,
    DEFAULT_CURATED,
    DEFAULT_SPLIT,
    DEFAULT_VOCAB_CHECKPOINT,
    NpyDataset,
    ScaledInkClassifier,
    _class_labels,
    _inspect_prepared_cache,
    _load_writer_map,
)


DEFAULT_CACHE = Path(
    r"D:\AIFlow-Workspace\PrivateData\math-ink-data-collector\derived"
    r"\hwr-architecture-20261002\uji-writer-inner-scratch-cache-r3"
)
DEFAULT_RUN_ROOT = Path(
    r"D:\AIFlow-Workspace\PrivateData\math-ink-data-collector\derived"
    r"\hwr-objective-competition-20261002\ce-vs-supcon-pk-r1"
)
DEFAULT_REPORT = ROOT / "artifacts" / "hwr_augmentation_microscope_20261002" / "objective_competition_ce_vs_supcon_pk_r1.json"
DEFAULT_CACHE_AUDIT = ROOT / "artifacts" / "hwr_augmentation_microscope_20261002" / "objective_competition_three_way_paired_r1.json"
DEFAULT_CACHE_CAPACITY_REPORT = ROOT / "artifacts" / "hwr_augmentation_microscope_20261002" / "architecture_scale_uji_inner_writer_probe.json"
DEFAULT_CACHE_OBJECTIVE_REPORT = ROOT / "artifacts" / "hwr_augmentation_microscope_20261002" / "objective_competition_ce_vs_supcon_pk_r1.json"
DEFAULT_CACHE_AUDIT_SHA256 = "23c7db2a4e8fab1d590c4f452cef400aaaba55bb125d8265cc2b96b84c80473e"
SYNTHETIC_EQUAL_ROWS = 512


def _host_memory_snapshot() -> dict[str, Any]:
    """Read host physical/commit headroom without adding a runtime dependency."""
    if os.name != "nt":
        return {"host_memory_query": "windows_only"}

    class _MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = _MemoryStatusEx()
    status.dwLength = ctypes.sizeof(_MemoryStatusEx)
    try:
        ok = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    except (AttributeError, OSError):
        return {"host_memory_query": "failed"}
    if not ok:
        return {"host_memory_query": "failed"}
    return {
        "host_memory_load_percent": int(status.dwMemoryLoad),
        "host_available_physical_mib": round(status.ullAvailPhys / 1024**2, 2),
        "host_available_commit_mib": round(status.ullAvailPageFile / 1024**2, 2),
        "host_commit_limit_mib": round(status.ullTotalPageFile / 1024**2, 2),
    }


def _reuse_hash_attested_cache(
    canonical_root: Path,
    curated_path: Path,
    split_path: Path,
    cache_root: Path,
) -> tuple[list[str], dict[str, Any], dict[str, Any]]:
    """Reuse only the exact arrays and sources bound by the completed paired audit."""
    audit_path = DEFAULT_CACHE_AUDIT.resolve()
    capacity_path = DEFAULT_CACHE_CAPACITY_REPORT.resolve()
    objective_path = DEFAULT_CACHE_OBJECTIVE_REPORT.resolve()
    if _sha256(audit_path) != DEFAULT_CACHE_AUDIT_SHA256:
        raise ValueError("trusted cache audit fingerprint changed; use the full cache audit path")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    capacity = json.loads(capacity_path.read_text(encoding="utf-8"))
    objective = json.loads(objective_path.read_text(encoding="utf-8"))
    if audit.get("schema") != "aiflow-hwr-objective-three-way-paired-audit/v1" or audit.get("status") != "completed_exploratory_three_way_paired_audit":
        raise ValueError("trusted cache audit schema/status mismatch")
    if capacity.get("schema") != "aiflow-hwr-architecture-scale-writer-probe/v1" or capacity.get("status") != "completed_exploratory_capacity_probe":
        raise ValueError("trusted capacity report schema/status mismatch")
    if objective.get("schema") != "aiflow-hwr-objective-competition-probe/v1" or objective.get("status") != "completed_exploratory_objective_ablation":
        raise ValueError("trusted objective report schema/status mismatch")
    audit_data = audit["data"]
    capacity_data = capacity["data"]
    objective_data = objective["data"]
    capacity_cache = capacity_data["cache"]
    objective_cache = objective_data["cache_audit"]
    pinned = {
        "capacity_report_sha256": _sha256(capacity_path),
        "objective_report_sha256": _sha256(objective_path),
    }
    if any(audit_data.get(key) != value for key, value in pinned.items()):
        raise ValueError("paired audit does not pin the supplied capacity/objective reports")
    if audit.get("product_adopted") is not False or audit_data.get("crohme_rows") != 0 or objective_data.get("crohme_rows") != 0:
        raise ValueError("cache reuse reference violates shadow-only/CROHME-excluded policy")

    source_hashes = {
        "canonical_hwrt_sha256": _sha256(canonical_root / "hwrt.jsonl.gz"),
        "canonical_uji_sha256": _sha256(canonical_root / "uji.jsonl.gz"),
        "curated_uji_sha256": _sha256(curated_path),
        "writer_split_sha256": _sha256(split_path),
    }
    expected_source_hashes = {
        "canonical_hwrt_sha256": capacity_data["canonical_hwrt_sha256"],
        "canonical_uji_sha256": capacity_data["canonical_uji_sha256"],
        "curated_uji_sha256": capacity_data["curated_uji_sha256"],
        "writer_split_sha256": capacity_data["writer_split_sha256"],
    }
    if source_hashes != expected_source_hashes:
        raise ValueError("source/split fingerprint mismatch; refusing attested-cache reuse")
    if objective_data.get("canonical_uji_sha256") != source_hashes["canonical_uji_sha256"] or objective_data.get("curated_uji_sha256") != source_hashes["curated_uji_sha256"] or objective_data.get("writer_split_sha256") != source_hashes["writer_split_sha256"]:
        raise ValueError("objective report source fingerprints do not reconcile")
    if audit_data.get("writer_split_sha256") != source_hashes["writer_split_sha256"]:
        raise ValueError("paired audit writer split does not match the current split")

    prepared_hashes = objective_cache.get("prepared_cache_sha256", {})
    if prepared_hashes != capacity_cache.get("prepared_cache_sha256", {}):
        raise ValueError("objective and capacity reports disagree on prepared cache hashes")
    if not prepared_hashes:
        raise ValueError("trusted reports do not contain prepared cache hashes")
    cache_files = {
        "train_features.npy": objective_cache["cache"]["train"]["features"],
        "train_labels.npy": objective_cache["cache"]["train"]["labels"],
        "validation_features.npy": objective_cache["cache"]["validation"]["features"],
        "validation_labels.npy": objective_cache["cache"]["validation"]["labels"],
    }
    current_cache_hashes: dict[str, str] = {}
    for name, recorded_path in cache_files.items():
        current_path = (cache_root / name).resolve()
        if Path(recorded_path).resolve() != current_path:
            raise ValueError(f"prepared cache path mismatch for {name}")
        current_cache_hashes[name] = _sha256(current_path)
    if current_cache_hashes != prepared_hashes:
        raise ValueError("prepared cache array hash mismatch; refusing reuse")

    checkpoint = torch.load(DEFAULT_VOCAB_CHECKPOINT, map_location="cpu", weights_only=False)
    labels = list(checkpoint.get("math_labels", []))
    if len(labels) != 372 or len(set(labels)) != 372:
        raise ValueError("pinned vocabulary checkpoint does not provide a unique 372-class label set")
    train_dataset = NpyDataset(cache_root / "train_features.npy", cache_root / "train_labels.npy")
    validation_dataset = NpyDataset(cache_root / "validation_features.npy", cache_root / "validation_labels.npy")
    if train_dataset.features.dtype != np.float32 or validation_dataset.features.dtype != np.float32:
        raise ValueError("prepared feature arrays are not float32")
    if train_dataset.labels.dtype.kind not in "iu" or validation_dataset.labels.dtype.kind not in "iu":
        raise ValueError("prepared label arrays are not integer-valued")
    if len(train_dataset) != objective_cache["rows"]["train"] or len(validation_dataset) != objective_cache["rows"]["validation"]:
        raise ValueError("prepared cache row counts do not match the attested report")
    train_ids = np.asarray(train_dataset.labels, dtype=np.int64)
    validation_ids = np.asarray(validation_dataset.labels, dtype=np.int64)
    if train_ids.min() < 0 or train_ids.max() >= len(labels) or validation_ids.min() < 0 or validation_ids.max() >= len(labels):
        raise ValueError("prepared cache label IDs escape the 372-class vocabulary")
    if np.count_nonzero(np.bincount(train_ids, minlength=len(labels))) != 372 or len(np.unique(validation_ids)) != objective_cache["validation_label_coverage"]:
        raise ValueError("prepared cache label support does not match the attested report")
    if len(validation_dataset) != audit_data["validation_rows"] or len(np.unique(validation_ids)) != audit_data["present_labels"]:
        raise ValueError("validation arrays do not match the paired audit")

    cache_audit = dict(objective_cache)
    provenance = {
        "mode": "hash_attested_reuse",
        "audit_report": str(audit_path),
        "audit_report_sha256": DEFAULT_CACHE_AUDIT_SHA256,
        "capacity_report_sha256": pinned["capacity_report_sha256"],
        "objective_report_sha256": pinned["objective_report_sha256"],
        "source_hashes": source_hashes,
        "prepared_cache_sha256": current_cache_hashes,
        "vocabulary_checkpoint_sha256": _sha256(DEFAULT_VOCAB_CHECKPOINT),
        "full_jsonl_rebuild_scan": False,
    }
    return labels, cache_audit, provenance


def _reuse_objective_report_attested_cache(
    canonical_root: Path,
    curated_path: Path,
    split_path: Path,
    cache_root: Path,
    report_path: Path,
) -> tuple[list[str], dict[str, Any], dict[str, Any]]:
    """Reuse prepared arrays only when a completed exploratory report binds their bytes and sources."""
    report_file = report_path.resolve()
    if not report_file.is_file():
        raise FileNotFoundError("completed objective report is required to attest cache reuse")
    report = json.loads(report_file.read_text(encoding="utf-8"))
    if (
        report.get("schema") != "aiflow-hwr-objective-competition-probe/v2"
        or report.get("status") != "completed_exploratory_objective_ablation"
    ):
        raise ValueError("cache attestation report schema/status mismatch")
    data = report.get("data", {})
    experiment = report.get("experiment", {})
    cache_audit = data.get("cache_audit", {})
    if data.get("crohme_rows") != 0 or cache_audit.get("crohme_rows") != 0:
        raise ValueError("cache attestation report includes CROHME rows")
    if experiment.get("product_adopted") is not False:
        raise ValueError("cache attestation report is not explicitly shadow-only")

    source_hashes = {
        "canonical_hwrt_sha256": _sha256(canonical_root / "hwrt.jsonl.gz"),
        "canonical_uji_sha256": _sha256(canonical_root / "uji.jsonl.gz"),
        "curated_uji_sha256": _sha256(curated_path),
        "writer_split_sha256": _sha256(split_path),
    }
    report_source_hashes = {
        "canonical_hwrt_sha256": data.get("canonical_hwrt_sha256"),
        "canonical_uji_sha256": data.get("canonical_uji_sha256"),
        "curated_uji_sha256": data.get("curated_uji_sha256"),
        "writer_split_sha256": data.get("writer_split_sha256"),
    }
    if source_hashes != report_source_hashes:
        raise ValueError("source/split fingerprint mismatch; refusing objective-report cache reuse")

    prepared_hashes = cache_audit.get("prepared_cache_sha256", {})
    cache_files = {
        "train_features.npy": cache_audit.get("cache", {}).get("train", {}).get("features"),
        "train_labels.npy": cache_audit.get("cache", {}).get("train", {}).get("labels"),
        "validation_features.npy": cache_audit.get("cache", {}).get("validation", {}).get("features"),
        "validation_labels.npy": cache_audit.get("cache", {}).get("validation", {}).get("labels"),
    }
    if not prepared_hashes or any(not value for value in cache_files.values()):
        raise ValueError("completed report lacks prepared-cache paths or fingerprints")
    current_hashes: dict[str, str] = {}
    for name, recorded_path in cache_files.items():
        current_path = (cache_root / name).resolve()
        if Path(recorded_path).resolve() != current_path:
            raise ValueError(f"prepared cache path mismatch for {name}")
        current_hashes[name] = _sha256(current_path)
    if current_hashes != prepared_hashes:
        raise ValueError("prepared cache array hash mismatch; refusing objective-report reuse")

    labels = _class_labels(canonical_root)
    if len(labels) != 372 or len(set(labels)) != 372:
        raise ValueError("canonical sources do not provide the expected unique 372-class vocabulary")
    train_dataset = NpyDataset(cache_root / "train_features.npy", cache_root / "train_labels.npy")
    validation_dataset = NpyDataset(cache_root / "validation_features.npy", cache_root / "validation_labels.npy")
    if (
        train_dataset.features.dtype != np.float32
        or validation_dataset.features.dtype != np.float32
        or train_dataset.features.shape[1:] != (128, 5)
        or validation_dataset.features.shape[1:] != (128, 5)
    ):
        raise ValueError("prepared arrays have an unexpected dtype or HWR input shape")
    if train_dataset.labels.dtype.kind not in "iu" or validation_dataset.labels.dtype.kind not in "iu":
        raise ValueError("prepared labels are not integer-valued")
    rows = cache_audit.get("rows", {})
    if len(train_dataset) != rows.get("train") or len(validation_dataset) != rows.get("validation"):
        raise ValueError("prepared cache row counts differ from the completed report")
    train_ids = np.asarray(train_dataset.labels, dtype=np.int64)
    validation_ids = np.asarray(validation_dataset.labels, dtype=np.int64)
    if (
        train_ids.size == 0
        or validation_ids.size == 0
        or train_ids.min() < 0
        or train_ids.max() >= len(labels)
        or validation_ids.min() < 0
        or validation_ids.max() >= len(labels)
        or len(np.unique(train_ids)) != 372
        or len(np.unique(validation_ids)) != data.get("validation_present_labels")
    ):
        raise ValueError("prepared label IDs/support do not match the completed report")
    if data.get("validation_rows") != len(validation_dataset) or data.get("validation_writer_clusters") != 8:
        raise ValueError("completed report does not attest the expected eight-writer validation set")

    provenance = {
        "mode": "completed_objective_report_hash_attested_reuse",
        "attestation_report": str(report_file),
        "attestation_report_sha256": _sha256(report_file),
        "source_hashes": source_hashes,
        "prepared_cache_sha256": current_hashes,
        "full_jsonl_rebuild_scan": False,
    }
    return labels, dict(cache_audit), provenance


def _reuse_partial_report_attested_cache(
    canonical_root: Path,
    curated_path: Path,
    split_path: Path,
    cache_root: Path,
    report_path: Path,
) -> tuple[list[str], dict[str, Any], dict[str, Any]]:
    """Resume a completed arm from a failed pair only when report-bound bytes still match."""
    report_file = report_path.resolve()
    if not report_file.is_file():
        raise FileNotFoundError("partial CE report is required to attest cache reuse")
    report = json.loads(report_file.read_text(encoding="utf-8"))
    if (
        report.get("schema") != "aiflow-hwr-objective-competition-partial-run/v1"
        or report.get("status") != "partial_ce_arm_complete_supcon_arm_failed"
        or report.get("product_adopted") is not False
    ):
        raise ValueError("partial report does not attest a shadow-only completed CE arm")
    data = report.get("data", {})
    if data.get("crohme_rows") != 0:
        raise ValueError("partial report cache includes CROHME rows")
    source_hashes = {
        "canonical_hwrt_sha256": _sha256(canonical_root / "hwrt.jsonl.gz"),
        "canonical_uji_sha256": _sha256(canonical_root / "uji.jsonl.gz"),
        "curated_uji_sha256": _sha256(curated_path),
        "writer_split_sha256": _sha256(split_path),
    }
    reported_source_hashes = {
        key: data.get(key) for key in source_hashes
    }
    if source_hashes != reported_source_hashes:
        raise ValueError("partial CE source/split hashes differ from the current inputs")
    prepared_hashes = data.get("prepared_cache_sha256", {})
    cache_paths = {
        "train_features.npy": cache_root / "train_features.npy",
        "train_labels.npy": cache_root / "train_labels.npy",
        "validation_features.npy": cache_root / "validation_features.npy",
        "validation_labels.npy": cache_root / "validation_labels.npy",
    }
    if set(prepared_hashes) != set(cache_paths):
        raise ValueError("partial report lacks all prepared-array fingerprints")
    current_cache_hashes = {name: _sha256(path.resolve()) for name, path in cache_paths.items()}
    if current_cache_hashes != prepared_hashes:
        raise ValueError("partial report prepared-array hashes differ from the current cache")

    labels = _class_labels(canonical_root)
    train_dataset = NpyDataset(cache_paths["train_features.npy"], cache_paths["train_labels.npy"])
    validation_dataset = NpyDataset(cache_paths["validation_features.npy"], cache_paths["validation_labels.npy"])
    if (
        train_dataset.features.dtype != np.float32
        or validation_dataset.features.dtype != np.float32
        or train_dataset.features.shape[1:] != (128, 5)
        or validation_dataset.features.shape[1:] != (128, 5)
        or train_dataset.labels.dtype.kind not in "iu"
        or validation_dataset.labels.dtype.kind not in "iu"
    ):
        raise ValueError("partial report cache has unexpected HWR tensor or label types")
    train_ids = np.asarray(train_dataset.labels, dtype=np.int64)
    validation_ids = np.asarray(validation_dataset.labels, dtype=np.int64)
    if (
        len(train_dataset) != data.get("train_rows")
        or len(validation_dataset) != data.get("validation_rows")
        or train_ids.size == 0
        or validation_ids.size == 0
        or train_ids.min() < 0
        or train_ids.max() >= len(labels)
        or validation_ids.min() < 0
        or validation_ids.max() >= len(labels)
        or len(np.unique(train_ids)) != 372
        or len(np.unique(validation_ids)) != data.get("validation_present_labels")
        or data.get("validation_writer_clusters") != 8
    ):
        raise ValueError("partial report cache row/label/writer support differs from current arrays")
    cache_audit = {
        "rows": {"train": len(train_dataset), "validation": len(validation_dataset)},
        "prepared_cache_sha256": current_cache_hashes,
        "validation_label_coverage": int(len(np.unique(validation_ids))),
        "crohme_rows": 0,
    }
    provenance = {
        "mode": "partial_report_hash_attested_reuse",
        "attestation_report": str(report_file),
        "attestation_report_sha256": _sha256(report_file),
        "audit_report_sha256": None,
        "source_hashes": source_hashes,
        "prepared_cache_sha256": current_cache_hashes,
        "full_jsonl_rebuild_scan": False,
    }
    return labels, cache_audit, provenance


class PKBatchSampler(Sampler[list[int]]):
    """Yield P classes × K distinct rows; the same seeded batches feed both arms."""

    def __init__(self, labels: np.ndarray, classes_per_batch: int, rows_per_class: int, batches: int, seed: int) -> None:
        self.labels = np.asarray(labels, dtype=np.int64)
        self.classes_per_batch = int(classes_per_batch)
        self.rows_per_class = int(rows_per_class)
        self.batches = int(batches)
        self.seed = int(seed)
        self.epoch = 0
        self.classes = np.unique(self.labels)
        self.indices_by_class = {int(label): np.flatnonzero(self.labels == label) for label in self.classes}
        if self.classes_per_batch < 2 or self.rows_per_class < 2 or self.batches < 1:
            raise ValueError("P, K, and batch count must be positive; P and K must be at least two")
        if self.classes_per_batch > len(self.classes):
            raise ValueError("classes_per_batch exceeds the number of train classes")
        too_small = {label: len(indices) for label, indices in self.indices_by_class.items() if len(indices) < self.rows_per_class}
        if too_small:
            raise ValueError(f"classes lack enough distinct rows for P×K sampling: {too_small}")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.batches

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        class_ids = self.classes
        for _ in range(self.batches):
            chosen_classes = rng.choice(class_ids, size=self.classes_per_batch, replace=False)
            batch: list[int] = []
            for label in chosen_classes:
                rows = self.indices_by_class[int(label)]
                batch.extend(int(index) for index in rng.choice(rows, size=self.rows_per_class, replace=False))
            rng.shuffle(batch)
            if len(batch) != self.classes_per_batch * self.rows_per_class or len(set(batch)) != len(batch):
                raise AssertionError("P×K batch construction produced an invalid batch")
            yield batch


def supervised_contrastive_loss(embeddings: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
    """Pull same-label examples together and compete against other batch classes."""
    if embeddings.ndim != 2 or labels.ndim != 1 or len(embeddings) != len(labels):
        raise ValueError("expected embeddings [batch, dim] and labels [batch]")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    vectors = F.normalize(embeddings.float(), dim=1)
    similarities = vectors @ vectors.T / float(temperature)
    batch_size = len(labels)
    off_diagonal = ~torch.eye(batch_size, dtype=torch.bool, device=labels.device)
    positives = labels[:, None].eq(labels[None, :]) & off_diagonal
    positive_counts = positives.sum(dim=1)
    if torch.any(positive_counts == 0):
        raise ValueError("every anchor must have at least one distinct same-label positive")
    similarities = similarities.masked_fill(~off_diagonal, float("-inf"))
    log_prob = similarities - torch.logsumexp(similarities, dim=1, keepdim=True)
    positive_log_prob = log_prob.masked_fill(~positives, 0.0).sum(dim=1) / positive_counts
    loss = -positive_log_prob.mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite supervised contrastive loss")
    return loss


def hardnegative_margin_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    margin: float,
    target_top_k: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Penalize the strongest wrong class, optionally only for target-in-Top-K rows."""
    if logits.ndim != 2 or labels.ndim != 1 or len(logits) != len(labels):
        raise ValueError("expected logits [batch, classes] and labels [batch]")
    if margin <= 0.0 or logits.shape[1] < 2:
        raise ValueError("margin must be positive and at least two classes are required")
    values = logits.float()
    if target_top_k is None:
        eligible = torch.ones_like(labels, dtype=torch.bool)
    else:
        if target_top_k < 1 or target_top_k > values.shape[1]:
            raise ValueError("target_top_k must be between one and the number of classes")
        eligible = (values.topk(k=target_top_k, dim=1).indices == labels[:, None]).any(dim=1)
    wrong = values.clone()
    wrong.scatter_(1, labels[:, None], float("-inf"))
    strongest_wrong = wrong.max(dim=1).values
    correct = values.gather(1, labels[:, None]).squeeze(1)
    losses = F.relu(float(margin) + strongest_wrong - correct)
    losses = torch.where(eligible, losses, torch.zeros_like(losses))
    if not torch.isfinite(losses).all():
        raise FloatingPointError("non-finite hard-negative margin loss")
    return losses.mean(), (losses > 0.0).sum(), eligible.sum()


def top5_set_preservation_loss(
    logits: torch.Tensor,
    reference_logits: torch.Tensor,
    margin: float = 0.0,
    top_k: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep every frozen-reference Top-K class above all classes outside its set."""
    if logits.ndim != 2 or reference_logits.shape != logits.shape:
        raise ValueError("student and reference logits must have the same [batch, classes] shape")
    if not math.isfinite(margin) or margin < 0.0:
        raise ValueError("Top-K preservation margin must be finite and non-negative")
    if top_k < 1 or top_k >= logits.shape[1]:
        raise ValueError("Top-K preservation requires 1 <= top_k < number of classes")

    student = logits.float()
    reference = reference_logits.detach().float()
    reference_indices = reference.topk(k=top_k, dim=1).indices
    reference_mask = torch.zeros_like(student, dtype=torch.bool)
    reference_mask.scatter_(1, reference_indices, True)
    outside_max = student.masked_fill(reference_mask, float("-inf")).amax(dim=1)
    candidate_scores = student.gather(1, reference_indices)
    violations = F.relu(float(margin) + outside_max[:, None] - candidate_scores)
    loss = violations.mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite Top-K preservation loss")
    active_examples = (violations > 0.0).any(dim=1).sum()
    return loss, active_examples


def _backward_microbatched_ce_hardnegative(
    model: nn.Module,
    features: torch.Tensor,
    target: torch.Tensor,
    microbatch_size: int,
    *,
    use_hardnegative: bool,
    hardnegative_weight: float,
    hardnegative_margin: float,
    hardnegative_top_k: int | None,
    top5_reference_model: nn.Module | None,
    top5_preservation_weight: float,
    top5_preservation_margin: float,
    amp: bool,
    scaler: torch.amp.GradScaler,
    capture_outputs: bool,
    phase_state: dict[str, str],
) -> dict[str, Any]:
    """Backpropagate separable CE/HN loss in slices, preserving one effective batch."""
    batch_size = len(target)
    if batch_size == 0 or microbatch_size <= 0 or microbatch_size >= batch_size:
        raise ValueError("microbatch size must be positive and smaller than the effective batch")
    if batch_size % microbatch_size:
        raise ValueError("effective batch must be divisible by microbatch size")

    ce_total = torch.zeros((), device=features.device, dtype=torch.float32)
    hardnegative_total = torch.zeros_like(ce_total)
    top5_preservation_total = torch.zeros_like(ce_total)
    top5_preservation_active_total = torch.zeros((), device=target.device, dtype=torch.long)
    active_total = torch.zeros((), device=target.device, dtype=torch.long)
    eligible_total = torch.zeros((), device=target.device, dtype=torch.long)
    correct_total = torch.zeros((), device=target.device, dtype=torch.long)
    finite = torch.ones((), device=features.device, dtype=torch.bool)
    embedding_chunks: list[torch.Tensor] = []
    logits_chunks: list[torch.Tensor] = []
    microbatch_count = batch_size // microbatch_size

    try:
        for microbatch_index, start in enumerate(range(0, batch_size, microbatch_size), start=1):
            end = start + microbatch_size
            size = end - start
            weight = size / batch_size
            phase_state["phase"] = f"forward_microbatch_{microbatch_index}_of_{microbatch_count}"
            micro_features = features[start:end]
            micro_target = target[start:end]
            reference_logits = None
            if top5_reference_model is not None:
                with torch.no_grad(), torch.autocast(device_type=features.device.type, enabled=False):
                    reference_logits = top5_reference_model.math_head(
                        top5_reference_model.encode(micro_features.float())
                    )
            with torch.autocast(device_type=features.device.type, dtype=torch.float16, enabled=amp):
                embedding = model.encode(micro_features)
                logits = model.math_head(embedding)
                ce = F.cross_entropy(logits.float(), micro_target)
                if use_hardnegative:
                    hardnegative, active, eligible = hardnegative_margin_loss(
                        logits, micro_target, hardnegative_margin, hardnegative_top_k,
                    )
                    objective = ce + hardnegative_weight * hardnegative
                    active_total.add_(active.detach())
                    eligible_total.add_(eligible.detach())
                    hardnegative_total.add_(hardnegative.detach() * weight)
                else:
                    hardnegative = ce.new_zeros(())
                    active = torch.zeros((), device=target.device, dtype=torch.long)
                    eligible = torch.zeros((), device=target.device, dtype=torch.long)
                    objective = ce
                if reference_logits is not None:
                    top5_preservation, top5_active = top5_set_preservation_loss(
                        logits, reference_logits, top5_preservation_margin,
                    )
                    objective = objective + top5_preservation_weight * top5_preservation
                    top5_preservation_total.add_(top5_preservation.detach() * weight)
                    top5_preservation_active_total.add_(top5_active.detach())
                else:
                    top5_preservation = ce.new_zeros(())
                weighted_objective = objective * weight
            ce_total.add_(ce.detach() * weight)
            finite.logical_and_(torch.isfinite(weighted_objective.detach()))
            correct_total.add_((logits.detach().argmax(dim=1) == micro_target).sum())
            phase_state["phase"] = f"backward_microbatch_{microbatch_index}_of_{microbatch_count}"
            if amp:
                scaler.scale(weighted_objective).backward()
            else:
                weighted_objective.backward()
            if capture_outputs:
                embedding_chunks.append(embedding.detach().float().cpu())
                logits_chunks.append(logits.detach().float().cpu())
            del micro_features, micro_target, embedding, logits, ce, hardnegative, objective, weighted_objective, top5_preservation, reference_logits
        phase_state["phase"] = "validate_accumulated_objective"
        if not bool(finite.detach().cpu()):
            raise FloatingPointError("non-finite microbatched objective")
        ce_value = float(ce_total.detach().cpu())
        hardnegative_value = float(hardnegative_total.detach().cpu())
        top5_preservation_value = float(top5_preservation_total.detach().cpu())
        active_value = int(active_total.detach().cpu())
        correct_value = int(correct_total.detach().cpu())
        return {
            "ce": ce_value,
            "hardnegative": hardnegative_value,
            "loss": ce_value + hardnegative_weight * hardnegative_value + top5_preservation_weight * top5_preservation_value,
            "active": active_value,
            "eligible": int(eligible_total.detach().cpu()),
            "top5_preservation": top5_preservation_value,
            "top5_preservation_active": int(top5_preservation_active_total.detach().cpu()),
            "correct": correct_value,
            "rows": batch_size,
            "embedding": torch.cat(embedding_chunks, dim=0) if capture_outputs else None,
            "logits": torch.cat(logits_chunks, dim=0) if capture_outputs else None,
        }
    except Exception as exc:
        setattr(exc, "hwr_microbatch_phase", phase_state.get("phase", "unknown"))
        setattr(exc, "hwr_microbatch_index", microbatch_index if "microbatch_index" in locals() else 0)
        raise


def _capture_rng_state(device: torch.device) -> tuple[torch.Tensor, torch.Tensor | None]:
    cpu_state = torch.get_rng_state().clone()
    cuda_state = torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
    return cpu_state, cuda_state


def _restore_rng_state(state: tuple[torch.Tensor, torch.Tensor | None], device: torch.device) -> None:
    torch.set_rng_state(state[0])
    if device.type == "cuda":
        if state[1] is None:
            raise AssertionError("CUDA RNG replay state is missing")
        torch.cuda.set_rng_state(state[1], device)


def _backward_microbatched_ce_supcon(
    model: nn.Module,
    features: torch.Tensor,
    target: torch.Tensor,
    microbatch_size: int,
    *,
    contrastive_weight: float,
    temperature: float,
    amp: bool,
    scaler: torch.amp.GradScaler,
    capture_outputs: bool,
    phase_state: dict[str, str],
) -> dict[str, Any]:
    """Keep global P×K SupCon pairs while recomputing encoder activations by slice.

    The first no-grad pass stores only embeddings and the per-slice RNG snapshots.
    SupCon is differentiated over the complete effective batch; its embedding
    gradient is then replayed through each recomputed encoder slice alongside
    that slice's averaged CE loss. This avoids the full-batch encoder graph.
    """
    batch_size = len(target)
    if batch_size == 0 or microbatch_size <= 0 or microbatch_size >= batch_size:
        raise ValueError("microbatch size must be positive and smaller than the effective batch")
    if batch_size % microbatch_size:
        raise ValueError("effective batch must be divisible by microbatch size")

    embedding_chunks: list[torch.Tensor] = []
    logits_chunks: list[torch.Tensor] = []
    rng_states: list[tuple[torch.Tensor, torch.Tensor | None]] = []
    microbatch_count = batch_size // microbatch_size
    try:
        for microbatch_index, start in enumerate(range(0, batch_size, microbatch_size), start=1):
            end = start + microbatch_size
            phase_state["phase"] = f"supcon_probe_forward_{microbatch_index}_of_{microbatch_count}"
            rng_states.append(_capture_rng_state(features.device))
            with torch.no_grad(), torch.autocast(
                device_type=features.device.type, dtype=torch.float16, enabled=amp,
            ):
                embedding = model.encode(features[start:end])
                logits = model.math_head(embedding)
            embedding_chunks.append(embedding.detach().float())
            if capture_outputs:
                logits_chunks.append(logits.detach().float().cpu())
            del embedding, logits

        rng_after_probe = _capture_rng_state(features.device)
        phase_state["phase"] = "global_supcon_loss"
        joint_embeddings = torch.cat(embedding_chunks, dim=0).detach().requires_grad_(True)
        supcon = supervised_contrastive_loss(joint_embeddings, target, temperature)
        if not torch.isfinite(supcon):
            raise FloatingPointError("non-finite global SupCon loss")
        phase_state["phase"] = "global_supcon_embedding_backward"
        scaled_supcon = scaler.scale(contrastive_weight * supcon) if amp else contrastive_weight * supcon
        scaled_supcon.backward()
        if joint_embeddings.grad is None or not torch.isfinite(joint_embeddings.grad).all():
            raise FloatingPointError("global SupCon embedding gradient is missing or non-finite")
        supcon_embedding_gradient = joint_embeddings.grad.detach()

        ce_total = torch.zeros((), device=features.device, dtype=torch.float32)
        correct_total = 0
        replay_error_max = 0.0
        for microbatch_index, start in enumerate(range(0, batch_size, microbatch_size), start=1):
            end = start + microbatch_size
            phase_state["phase"] = f"supcon_recompute_forward_{microbatch_index}_of_{microbatch_count}"
            _restore_rng_state(rng_states[microbatch_index - 1], features.device)
            with torch.autocast(device_type=features.device.type, dtype=torch.float16, enabled=amp):
                embedding = model.encode(features[start:end])
                logits = model.math_head(embedding)
                ce = F.cross_entropy(logits.float(), target[start:end])
                weighted_ce = ce * ((end - start) / batch_size)
            expected_embedding = embedding_chunks[microbatch_index - 1]
            replay_error = float((embedding.detach().float() - expected_embedding).abs().max().cpu())
            replay_error_max = max(replay_error_max, replay_error)
            if replay_error > 1.0e-5:
                raise AssertionError(
                    f"dropout RNG replay changed microbatch {microbatch_index} embeddings: {replay_error}"
                )
            phase_state["phase"] = f"supcon_recompute_backward_{microbatch_index}_of_{microbatch_count}"
            ce_for_backward = scaler.scale(weighted_ce) if amp else weighted_ce
            torch.autograd.backward(
                (ce_for_backward, embedding),
                (None, supcon_embedding_gradient[start:end].to(dtype=embedding.dtype)),
            )
            ce_total.add_(ce.detach().float() * ((end - start) / batch_size))
            correct_total += int((logits.detach().argmax(dim=1) == target[start:end]).sum().cpu())
            if capture_outputs:
                embedding_chunks[microbatch_index - 1] = embedding.detach().float().cpu()
            del embedding, logits, ce, weighted_ce, ce_for_backward

        _restore_rng_state(rng_after_probe, features.device)
        positive_mask = target[:, None].eq(target[None, :]) & ~torch.eye(
            batch_size, dtype=torch.bool, device=target.device,
        )
        ce_value = float(ce_total.detach().cpu())
        supcon_value = float(supcon.detach().cpu())
        return {
            "ce": ce_value,
            "supcon": supcon_value,
            "hardnegative": 0.0,
            "loss": ce_value + contrastive_weight * supcon_value,
            "positive_anchors": int(positive_mask.any(dim=1).sum().cpu()),
            "correct": correct_total,
            "rows": batch_size,
            "replay_max_abs_embedding_error": replay_error_max,
            "embedding": torch.cat(embedding_chunks, dim=0).detach().cpu() if capture_outputs else None,
            "logits": torch.cat(logits_chunks, dim=0) if capture_outputs else None,
        }
    except Exception as exc:
        setattr(exc, "hwr_microbatch_phase", phase_state.get("phase", "unknown"))
        setattr(exc, "hwr_microbatch_index", microbatch_index if "microbatch_index" in locals() else 0)
        raise


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _gradient_component_norms(model: nn.Module) -> dict[str, float]:
    """Report pre-clip gradient norms at the model's input, encoder, and heads."""
    squared_norms: dict[str, float] = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        if name.startswith("input_projection."):
            component = "input_projection"
        elif name == "position":
            component = "position_embedding"
        elif name.startswith("encoder.layers."):
            component = f"encoder_layer_{name.split('.')[2]}"
        elif name.startswith("pool_score."):
            component = "attention_pool"
        elif name.startswith("math_head."):
            component = "372_class_head"
        else:
            component = "other"
        # Copy one gradient tensor at a time to CPU so diagnostics do not raise
        # the peak CUDA allocation of the model being measured.
        component_norm = float(torch.linalg.vector_norm(parameter.grad.detach().float().cpu(), ord=2))
        squared_norms[component] = squared_norms.get(component, 0.0) + component_norm * component_norm
    return {name: round(value ** 0.5, 8) for name, value in sorted(squared_norms.items())}


def _gradient_health_microscope(model: nn.Module) -> dict[str, Any]:
    """Locate non-finite gradients by model component without allocating on CUDA."""
    components: dict[str, dict[str, Any]] = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        if name.startswith("input_projection."):
            component = "input_projection"
        elif name == "position":
            component = "position_embedding"
        elif name.startswith("encoder.layers."):
            component = f"encoder_layer_{name.split('.')[2]}"
        elif name.startswith("pool_score."):
            component = "attention_pool"
        elif name.startswith("math_head."):
            component = "372_class_head"
        else:
            component = "other"
        gradient = parameter.grad.detach().to(device="cpu", dtype=torch.float32)
        finite = torch.isfinite(gradient)
        finite_values = gradient[finite]
        report = components.setdefault(component, {
            "parameter_tensors": 0,
            "gradient_elements": 0,
            "nonfinite_elements": 0,
            "nan_elements": 0,
            "positive_inf_elements": 0,
            "negative_inf_elements": 0,
            "finite_l2_squared": 0.0,
            "finite_abs_max": 0.0,
            "nonfinite_parameters": [],
        })
        report["parameter_tensors"] += 1
        report["gradient_elements"] += int(gradient.numel())
        report["nonfinite_elements"] += int((~finite).sum().item())
        report["nan_elements"] += int(torch.isnan(gradient).sum().item())
        report["positive_inf_elements"] += int(torch.isposinf(gradient).sum().item())
        report["negative_inf_elements"] += int(torch.isneginf(gradient).sum().item())
        if finite_values.numel():
            report["finite_l2_squared"] += float(torch.sum(finite_values * finite_values).item())
            report["finite_abs_max"] = max(
                report["finite_abs_max"], float(finite_values.abs().max().item())
            )
        if not bool(finite.all()):
            report["nonfinite_parameters"].append(name)
        del gradient, finite, finite_values
    for report in components.values():
        report["finite_l2_norm"] = round(report.pop("finite_l2_squared") ** 0.5, 8)
        report["finite_abs_max"] = round(report["finite_abs_max"], 8)
    return {
        "nonfinite_gradient_element_count": sum(
            int(value["nonfinite_elements"]) for value in components.values()
        ),
        "by_component": dict(sorted(components.items())),
    }


def _clip_grad_norm_low_memory(model: nn.Module, max_norm: float) -> torch.Tensor:
    """Compute the global norm one tensor at a time on CPU, then scale in place.

    On the GTX 1650 runtime, the stock CUDA norm-stack path intermittently fails
    after backward. This preserves the same global L2 clipping rule without
    stacking per-parameter CUDA norm tensors or allocating a flattened gradient.
    """
    if max_norm <= 0.0:
        raise ValueError("max_norm must be positive")
    squared_norm = 0.0
    parameters = [parameter for parameter in model.parameters() if parameter.grad is not None]
    for parameter in parameters:
        gradient_cpu = parameter.grad.detach().to(device="cpu", dtype=torch.float32)
        parameter_norm = float(torch.linalg.vector_norm(gradient_cpu, ord=2))
        squared_norm += parameter_norm * parameter_norm
        del gradient_cpu
    total_norm = squared_norm ** 0.5
    if not np.isfinite(total_norm):
        exc = FloatingPointError("non-finite global gradient norm")
        exc.hwr_gradient_diagnostics = _gradient_health_microscope(model)
        raise exc
    coefficient = min(1.0, float(max_norm) / (total_norm + 1.0e-6))
    if coefficient < 1.0:
        for parameter in parameters:
            parameter.grad.mul_(coefficient)
    return torch.tensor(total_norm, dtype=torch.float32)


@torch.inference_mode()
def _predict_correctness(model: nn.Module, dataset: NpyDataset, device: torch.device, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    top1: list[np.ndarray] = []
    top5: list[np.ndarray] = []
    for features, target in loader:
        logits = model(features.to(device, non_blocking=device.type == "cuda")).float()
        target = target.to(device, non_blocking=device.type == "cuda")
        indices = logits.topk(5, dim=1).indices
        top1.append((indices[:, 0] == target).cpu().numpy())
        top5.append((indices == target[:, None]).any(dim=1).cpu().numpy())
    return np.concatenate(top1), np.concatenate(top5)


def _train_arm(
    arm: str,
    seed: int,
    train_dataset: NpyDataset,
    validation_dataset: NpyDataset,
    labels: list[str],
    device: torch.device,
    run_root: Path,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    classes_per_batch: int,
    rows_per_class: int,
    contrastive_weight: float,
    temperature: float,
    hardnegative_weight: float,
    hardnegative_margin: float,
    hardnegative_top_k: int | None,
    top5_preservation_weight: float,
    top5_preservation_margin: float,
    top5_reference_checkpoint: Path | None,
    top5_reference_checkpoint_sha256: str | None,
    eval_batch_size: int,
    precision: str,
    microbatch_size: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    _seed_everything(seed)
    if device.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
    model = ScaledInkClassifier(len(labels), 128, 4, 4, 512).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    top5_reference_model = None
    top5_reference_metadata = None
    if top5_preservation_weight > 0.0:
        if top5_reference_checkpoint is None or top5_reference_checkpoint_sha256 is None:
            raise ValueError("Top-5 preservation requires a hash-attested CE reference checkpoint")
        reference_path = top5_reference_checkpoint.resolve()
        if _sha256(reference_path) != top5_reference_checkpoint_sha256:
            raise ValueError("Top-5 reference checkpoint changed after CE attestation")
        with torch.random.fork_rng(devices=[torch.cuda.current_device()] if device.type == "cuda" else []):
            top5_reference_model = ScaledInkClassifier(len(labels), 128, 4, 4, 512).to(device)
        reference_payload = torch.load(reference_path, map_location="cpu", weights_only=False)
        if (
            reference_payload.get("schema") != "aiflow-hwr-objective-competition-probe/v1"
            or reference_payload.get("arm") != "ce_pk"
            or reference_payload.get("seed") != seed
            or reference_payload.get("precision") != ("amp_fp16" if device.type == "cuda" and precision == "amp" else "fp32")
        ):
            raise ValueError("Top-5 reference checkpoint schema, arm, seed, or precision mismatch")
        reference_objective = reference_payload.get("objective", {})
        if reference_objective.get("ce_weight") != 1.0 or reference_objective.get("supcon_weight") != 0.0:
            raise ValueError("Top-5 reference checkpoint must be the paired pure-CE arm")
        if reference_payload.get("architecture", {}).get("parameters") != parameter_count:
            raise ValueError("Top-5 reference checkpoint architecture mismatch")
        top5_reference_model.load_state_dict(reference_payload["state_dict"], strict=True)
        top5_reference_model.eval()
        top5_reference_model.requires_grad_(False)
        top5_reference_metadata = {
            "checkpoint": str(reference_path),
            "checkpoint_sha256": top5_reference_checkpoint_sha256,
            "set_source": "paired frozen CE model Top-5 on the same input batch",
        }
    labels_train = np.asarray(train_dataset.labels, dtype=np.int64)
    batch_count = (len(train_dataset) + batch_size - 1) // batch_size
    sampler = PKBatchSampler(labels_train, classes_per_batch, rows_per_class, batch_count, seed + 17)
    loader = DataLoader(train_dataset, batch_sampler=sampler, num_workers=0, pin_memory=device.type == "cuda")
    # The default CUDA foreach implementation keeps extra parameter-sized
    # temporaries during the optimizer step. That caused an OOM on the 4 GB
    # GTX 1650 after forward/backward had already fit. Use the lower-peak-memory
    # per-tensor path and record it in both the checkpoint and report.
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1.0e-4, foreach=False)
    amp = device.type == "cuda" and precision == "amp"
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    history: list[dict[str, Any]] = []
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        sampler.set_epoch(epoch - 1)
        model.train()
        ce_losses: list[float] = []
        contrastive_losses: list[float] = []
        hardnegative_losses: list[float] = []
        top5_preservation_losses: list[float] = []
        total_losses: list[float] = []
        amp_overflow_skips = 0
        positive_anchors = 0
        total_anchors = 0
        active_hardnegative_examples = 0
        eligible_hardnegative_examples = 0
        total_hardnegative_examples = 0
        top5_preservation_active_examples = 0
        total_top5_preservation_examples = 0
        train_top1_hits = 0
        train_rows = 0
        supcon_replay_errors: list[float] = []
        epoch_started = time.perf_counter()
        for batch_index, (features, target) in enumerate(loader, start=1):
            microscope_step = batch_index == 1 or batch_index % 500 == 0
            features = features.to(device, non_blocking=device.type == "cuda")
            target = target.to(device, non_blocking=device.type == "cuda")
            optimizer.zero_grad(set_to_none=True)
            microbatch_mode = (
                microbatch_size < len(target)
                and arm in {"ce_pk", "ce_plus_hardnegative", "ce_plus_supcon"}
            )
            microbatch_stats = None
            prebackward_done = False
            if microbatch_mode:
                phase_state = {"phase": "microbatch_forward"}
                try:
                    if arm == "ce_plus_supcon":
                        microbatch_stats = _backward_microbatched_ce_supcon(
                            model, features, target, microbatch_size,
                            contrastive_weight=contrastive_weight,
                            temperature=temperature,
                            amp=amp,
                            scaler=scaler,
                            capture_outputs=microscope_step,
                            phase_state=phase_state,
                        )
                    else:
                        microbatch_stats = _backward_microbatched_ce_hardnegative(
                            model, features, target, microbatch_size,
                            use_hardnegative=arm == "ce_plus_hardnegative",
                            hardnegative_weight=hardnegative_weight,
                            hardnegative_margin=hardnegative_margin,
                            hardnegative_top_k=hardnegative_top_k,
                            top5_reference_model=top5_reference_model,
                            top5_preservation_weight=top5_preservation_weight,
                            top5_preservation_margin=top5_preservation_margin,
                            amp=amp,
                            scaler=scaler,
                            capture_outputs=microscope_step,
                            phase_state=phase_state,
                        )
                except RuntimeError as exc:
                    exc.hwr_failed_location = (
                        f"epoch={epoch},batch={batch_index},phase={getattr(exc, 'hwr_microbatch_phase', phase_state['phase'])}"
                    )
                    exc.hwr_completed_epochs = len(history)
                    raise
                ce = torch.tensor(microbatch_stats["ce"], dtype=torch.float32)
                supcon = torch.tensor(microbatch_stats.get("supcon", 0.0), dtype=torch.float32)
                hardnegative = torch.tensor(microbatch_stats["hardnegative"], dtype=torch.float32)
                top5_preservation = torch.tensor(microbatch_stats["top5_preservation"], dtype=torch.float32)
                hardnegative_active = torch.tensor(microbatch_stats.get("active", 0), dtype=torch.long)
                hardnegative_eligible = torch.tensor(microbatch_stats.get("eligible", 0), dtype=torch.long)
                loss = torch.tensor(microbatch_stats["loss"], dtype=torch.float32)
                embedding = microbatch_stats["embedding"]
                logits = microbatch_stats["logits"]
                prebackward_done = True
                if arm == "ce_plus_supcon":
                    positive_anchors += microbatch_stats["positive_anchors"]
                    total_anchors += microbatch_stats["rows"]
                    supcon_replay_errors.append(microbatch_stats["replay_max_abs_embedding_error"])
                if arm == "ce_plus_hardnegative":
                    active_hardnegative_examples += microbatch_stats["active"]
                    eligible_hardnegative_examples += microbatch_stats["eligible"]
                    total_hardnegative_examples += len(target)
                    if top5_reference_model is not None:
                        top5_preservation_active_examples += microbatch_stats["top5_preservation_active"]
                        total_top5_preservation_examples += len(target)
            else:
                reference_logits = None
                if top5_reference_model is not None:
                    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
                        reference_logits = top5_reference_model.math_head(
                            top5_reference_model.encode(features.float())
                        )
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                    embedding = model.encode(features)
                    logits = model.math_head(embedding)
                    ce = F.cross_entropy(logits.float(), target)
                    supcon = ce.new_zeros(())
                    hardnegative = ce.new_zeros(())
                    top5_preservation = ce.new_zeros(())
                    top5_preservation_active = ce.new_zeros((), dtype=torch.long)
                    hardnegative_active = ce.new_zeros((), dtype=torch.long)
                    hardnegative_eligible = ce.new_zeros((), dtype=torch.long)
                    if arm == "ce_plus_supcon":
                        try:
                            supcon = supervised_contrastive_loss(embedding, target, temperature)
                        except RuntimeError as exc:
                            exc.hwr_failed_location = f"epoch={epoch},batch={batch_index},phase=supcon_loss"
                            exc.hwr_completed_epochs = len(history)
                            if device.type == "cuda":
                                try:
                                    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
                                    memory = {
                                        "allocated_mib": round(torch.cuda.memory_allocated(device) / 1024**2, 2),
                                        "reserved_mib": round(torch.cuda.memory_reserved(device) / 1024**2, 2),
                                        "driver_free_mib": round(free_bytes / 1024**2, 2),
                                        "driver_total_mib": round(total_bytes / 1024**2, 2),
                                    }
                                except RuntimeError:
                                    memory = {"cuda_memory_query": "failed_after_loss_error"}
                                print(json.dumps({
                                    "event": "competitive_objective_loss_failure",
                                    "arm": arm, "seed": seed, "epoch": epoch, "batch": batch_index,
                                    "phase": "supcon_loss", "batch_size": len(target),
                                    **memory, **_host_memory_snapshot(),
                                    "error_type": type(exc).__name__, "error": str(exc),
                                }, ensure_ascii=False), flush=True)
                            raise
                        loss = ce + contrastive_weight * supcon
                        with torch.no_grad():
                            positive_anchors += int((target[:, None].eq(target[None, :]) & ~torch.eye(len(target), dtype=torch.bool, device=target.device)).any(dim=1).sum())
                            total_anchors += len(target)
                    elif arm == "ce_plus_hardnegative":
                        hardnegative, hardnegative_active, hardnegative_eligible = hardnegative_margin_loss(
                            logits, target, hardnegative_margin, hardnegative_top_k,
                        )
                        loss = ce + hardnegative_weight * hardnegative
                        active_hardnegative_examples += int(hardnegative_active.detach().cpu())
                        eligible_hardnegative_examples += int(hardnegative_eligible.detach().cpu())
                        total_hardnegative_examples += len(target)
                    elif arm == "ce_pk":
                        loss = ce
                    else:
                        raise ValueError(f"unknown objective arm: {arm}")
                    if reference_logits is not None:
                        top5_preservation, top5_preservation_active = top5_set_preservation_loss(
                            logits, reference_logits, top5_preservation_margin,
                        )
                        loss = loss + top5_preservation_weight * top5_preservation
                        top5_preservation_active_examples += int(top5_preservation_active.detach().cpu())
                        total_top5_preservation_examples += len(target)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite objective loss in {arm}, epoch={epoch}, batch={batch_index}")
            phase = "backward"
            try:
                if amp:
                    if not prebackward_done:
                        scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    if not prebackward_done:
                        loss.backward()
                phase = "gradient_norm_clip"
                amp_overflow_skipped = False
                try:
                    gradient_norm = _clip_grad_norm_low_memory(model, 1.0)
                except FloatingPointError as exc:
                    gradient_diagnostics = getattr(exc, "hwr_gradient_diagnostics", None)
                    nonfinite_count = (
                        gradient_diagnostics.get("nonfinite_gradient_element_count", 0)
                        if isinstance(gradient_diagnostics, dict) else 0
                    )
                    if not amp or nonfinite_count <= 0:
                        raise
                    scale_before_backoff = float(scaler.get_scale())
                    phase = "amp_overflow_skip"
                    # unscale_ has already recorded the inf gradients; GradScaler
                    # skips optimizer.step and backs off its scale here.
                    scaler.step(optimizer)
                    scaler.update()
                    scale_after_backoff = float(scaler.get_scale())
                    if scale_after_backoff >= scale_before_backoff:
                        raise FloatingPointError("GradScaler did not back off after detected inf gradients")
                    amp_overflow_skipped = True
                    amp_overflow_skips += 1
                    print(json.dumps({
                        "event": "competitive_objective_amp_overflow_skipped",
                        "arm": arm, "seed": seed, "epoch": epoch, "batch": batch_index,
                        "phase": phase, "loss": float(loss.detach().float().cpu()),
                        "cross_entropy": float(ce.detach().float().cpu()),
                        "supcon": float(supcon.detach().float().cpu()),
                        "grad_scaler_scale_before": scale_before_backoff,
                        "grad_scaler_scale_after": scale_after_backoff,
                        "gradient_diagnostics": gradient_diagnostics,
                    }, ensure_ascii=False), flush=True)
                if microscope_step and not amp_overflow_skipped:
                    component_gradient_norms = _gradient_component_norms(model)
                    probe_parameters = {
                        "input_projection": model.input_projection[0].weight,
                        "encoder_layer_0_attention": model.encoder.layers[0].self_attn.in_proj_weight,
                        "372_class_head": model.math_head.weight,
                    }
                    probe_before = {name: parameter.detach().clone() for name, parameter in probe_parameters.items()}
                if amp and not amp_overflow_skipped:
                    phase = "optimizer_step"
                    scaler.step(optimizer)
                    scaler.update()
                elif not amp:
                    phase = "optimizer_step"
                    optimizer.step()
                if microscope_step and not amp_overflow_skipped:
                    parameter_update_norms = {
                        name: round(float(torch.linalg.vector_norm(parameter.detach() - probe_before[name]).cpu()), 8)
                        for name, parameter in probe_parameters.items()
                    }
                    print(json.dumps({
                        "event": "competitive_objective_gradient_microscope",
                        "arm": arm, "seed": seed, "epoch": epoch, "batch": batch_index,
                        "loss": float(loss.detach().cpu()),
                        "cross_entropy": float(ce.detach().cpu()),
                        "supcon": float(supcon.detach().cpu()),
                        "hardnegative": float(hardnegative.detach().cpu()),
                        "hardnegative_active_examples": int(hardnegative_active.detach().cpu()),
                        "hardnegative_eligible_examples": int(hardnegative_eligible.detach().cpu()),
                        "embedding_norm_mean": round(float(embedding.detach().float().norm(dim=1).mean().cpu()), 8),
                        "logit_std": round(float(logits.detach().float().std().cpu()), 8),
                        "global_gradient_norm_preclip": round(float(gradient_norm.detach().cpu()), 8),
                        "gradient_norm_by_component_preclip": component_gradient_norms,
                        "parameter_update_norm_by_component": parameter_update_norms,
                    }, ensure_ascii=False), flush=True)
            except FloatingPointError as exc:
                exc.hwr_failed_location = f"epoch={epoch},batch={batch_index},phase={phase}"
                exc.hwr_completed_epochs = epoch - 1
                exc.hwr_completed_epoch_history = list(history)
                failure_probe = {
                    "event": "competitive_objective_nonfinite_gradient_failure",
                    "arm": arm, "seed": seed, "epoch": epoch, "batch": batch_index,
                    "phase": phase, "precision": "amp_fp16" if amp else "fp32",
                    "grad_scaler_scale": float(scaler.get_scale()) if amp else None,
                    "loss": float(loss.detach().float().cpu()) if "loss" in locals() else None,
                    "cross_entropy": float(ce.detach().float().cpu()) if "ce" in locals() else None,
                    "supcon": float(supcon.detach().float().cpu()) if "supcon" in locals() else None,
                    "gradient_diagnostics": getattr(exc, "hwr_gradient_diagnostics", None),
                    "error_type": type(exc).__name__, "error": str(exc),
                }
                if device.type == "cuda":
                    try:
                        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
                        failure_probe.update({
                            "allocated_mib": round(torch.cuda.memory_allocated(device) / 1024**2, 2),
                            "reserved_mib": round(torch.cuda.memory_reserved(device) / 1024**2, 2),
                            "driver_free_mib": round(free_bytes / 1024**2, 2),
                            "driver_total_mib": round(total_bytes / 1024**2, 2),
                        })
                    except RuntimeError:
                        failure_probe["cuda_memory_query"] = "failed_after_nonfinite_gradient"
                print(json.dumps(failure_probe, ensure_ascii=False), flush=True)
                raise
            except torch.cuda.OutOfMemoryError as exc:
                exc.hwr_failed_location = f"epoch={epoch},batch={batch_index},phase={phase}"
                exc.hwr_completed_epochs = epoch - 1
                if device.type == "cuda":
                    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
                    print(json.dumps({
                        "event": "competitive_objective_cuda_oom",
                        "arm": arm, "seed": seed, "epoch": epoch, "batch": batch_index,
                        "precision": "amp_fp16" if amp else "fp32",
                        "batch_size": len(target),
                        "allocated_mib": round(torch.cuda.memory_allocated(device) / 1024**2, 2),
                        "reserved_mib": round(torch.cuda.memory_reserved(device) / 1024**2, 2),
                        "peak_allocated_mib": round(torch.cuda.max_memory_allocated(device) / 1024**2, 2),
                        "driver_free_mib": round(free_bytes / 1024**2, 2),
                        "driver_total_mib": round(total_bytes / 1024**2, 2),
                        **_host_memory_snapshot(),
                        "error": str(exc),
                    }, ensure_ascii=False), flush=True)
                raise
            except RuntimeError as exc:
                exc.hwr_failed_location = f"epoch={epoch},batch={batch_index},phase={phase}"
                exc.hwr_completed_epochs = epoch - 1
                if device.type == "cuda":
                    try:
                        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
                        memory = {
                            "allocated_mib": round(torch.cuda.memory_allocated(device) / 1024**2, 2),
                            "reserved_mib": round(torch.cuda.memory_reserved(device) / 1024**2, 2),
                            "driver_free_mib": round(free_bytes / 1024**2, 2),
                            "driver_total_mib": round(total_bytes / 1024**2, 2),
                        }
                    except RuntimeError:
                        memory = {"cuda_memory_query": "failed_after_runtime_error"}
                    print(json.dumps({
                        "event": "competitive_objective_runtime_failure",
                        "arm": arm, "seed": seed, "epoch": epoch, "batch": batch_index,
                        "phase": phase, "batch_size": len(target), "precision": "amp_fp16" if amp else "fp32",
                        **memory, **_host_memory_snapshot(),
                        "error_type": type(exc).__name__, "error": str(exc),
                    }, ensure_ascii=False), flush=True)
                raise
            ce_losses.append(float(ce.detach().cpu()))
            contrastive_losses.append(float(supcon.detach().cpu()))
            hardnegative_losses.append(float(hardnegative.detach().cpu()))
            top5_preservation_losses.append(float(top5_preservation.detach().cpu()))
            total_losses.append(float(loss.detach().cpu()))
            if microbatch_mode:
                train_top1_hits += microbatch_stats["correct"]
                train_rows += microbatch_stats["rows"]
            else:
                with torch.no_grad():
                    train_top1_hits += int((logits.argmax(dim=1) == target).sum())
                    train_rows += len(target)
            if device.type == "cuda" and batch_index % 500 == 0:
                free_bytes, total_bytes = torch.cuda.mem_get_info(device)
                print(json.dumps({
                    "event": "competitive_objective_batch_progress",
                    "arm": arm, "seed": seed, "epoch": epoch, "batch": batch_index,
                    "mean_ce": float(np.mean(ce_losses)),
                    "mean_supcon": float(np.mean(contrastive_losses)),
                    "mean_hardnegative": float(np.mean(hardnegative_losses)),
                    "mean_top5_preservation": float(np.mean(top5_preservation_losses)),
                    "amp_overflow_skipped_steps": amp_overflow_skips,
                    "grad_scaler_scale": float(scaler.get_scale()) if amp else None,
                    "hardnegative_eligible_example_fraction": eligible_hardnegative_examples / max(total_hardnegative_examples, 1),
                    "top5_preservation_active_example_fraction": top5_preservation_active_examples / max(total_top5_preservation_examples, 1) if total_top5_preservation_examples else None,
                    "allocated_mib": round(torch.cuda.memory_allocated(device) / 1024**2, 2),
                    "reserved_mib": round(torch.cuda.memory_reserved(device) / 1024**2, 2),
                    "driver_free_mib": round(free_bytes / 1024**2, 2),
                    "driver_total_mib": round(total_bytes / 1024**2, 2),
                }, ensure_ascii=False), flush=True)
        try:
            validation = _evaluate_validation(model, validation_dataset, device, eval_batch_size)
        except RuntimeError as exc:
            exc.hwr_failed_location = f"epoch={epoch},batch=validation,phase=validation,eval_batch_size={eval_batch_size}"
            exc.hwr_completed_epochs = epoch - 1
            if device.type == "cuda":
                try:
                    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
                    memory = {
                        "allocated_mib": round(torch.cuda.memory_allocated(device) / 1024**2, 2),
                        "reserved_mib": round(torch.cuda.memory_reserved(device) / 1024**2, 2),
                        "driver_free_mib": round(free_bytes / 1024**2, 2),
                        "driver_total_mib": round(total_bytes / 1024**2, 2),
                    }
                except RuntimeError:
                    memory = {"cuda_memory_query": "failed_after_validation_error"}
                print(json.dumps({
                    "event": "competitive_objective_validation_failure",
                    "arm": arm, "seed": seed, "epoch": epoch,
                    "eval_batch_size": eval_batch_size,
                    **memory, **_host_memory_snapshot(),
                    "error_type": type(exc).__name__, "error": str(exc),
                }, ensure_ascii=False), flush=True)
            raise
        epoch_result = {
            "epoch": epoch,
            "train_ce": float(np.mean(ce_losses)),
            "train_supcon": float(np.mean(contrastive_losses)),
            "train_hardnegative": float(np.mean(hardnegative_losses)),
            "train_top5_preservation": float(np.mean(top5_preservation_losses)),
            "train_total_objective": float(np.mean(total_losses)),
            "amp_overflow_skipped_steps": amp_overflow_skips,
            "amp_overflow_skip_fraction": amp_overflow_skips / max(batch_index, 1),
            "final_grad_scaler_scale": float(scaler.get_scale()) if amp else None,
            "train_top1_on_sampled_batches": train_top1_hits / max(train_rows, 1),
            "supcon_positive_anchor_fraction": positive_anchors / max(total_anchors, 1) if arm == "ce_plus_supcon" else None,
            "supcon_replay_max_abs_embedding_error": max(supcon_replay_errors) if supcon_replay_errors else None,
            "hardnegative_active_example_fraction": active_hardnegative_examples / max(total_hardnegative_examples, 1) if arm == "ce_plus_hardnegative" else None,
            "hardnegative_eligible_example_fraction": eligible_hardnegative_examples / max(total_hardnegative_examples, 1) if arm == "ce_plus_hardnegative" else None,
            "top5_preservation_active_example_fraction": top5_preservation_active_examples / max(total_top5_preservation_examples, 1) if total_top5_preservation_examples else None,
            "validation": validation,
            "seconds": time.perf_counter() - epoch_started,
        }
        history.append(epoch_result)
        print(json.dumps({"event": "competitive_objective_epoch", "arm": arm, "seed": seed, **epoch_result}, ensure_ascii=False), flush=True)

    checkpoint_path = run_root / "checkpoints" / f"{arm}_seed{seed}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint_path.exists():
        raise FileExistsError(f"refusing to overwrite objective checkpoint: {checkpoint_path}")
    torch.save({
        "schema": "aiflow-hwr-objective-competition-probe/v1",
        "arm": arm,
        "architecture": {"width": 128, "layers": 4, "heads": 4, "feedforward": 512, "parameters": parameter_count},
        "objective": {"ce_weight": 1.0, "supcon_weight": contrastive_weight if arm == "ce_plus_supcon" else 0.0, "temperature": temperature if arm == "ce_plus_supcon" else None, "hardnegative_weight": hardnegative_weight if arm == "ce_plus_hardnegative" else 0.0, "hardnegative_margin": hardnegative_margin if arm == "ce_plus_hardnegative" else None, "hardnegative_gate": "target_in_current_top5" if arm == "ce_plus_hardnegative" and hardnegative_top_k == 5 else "all_examples" if arm == "ce_plus_hardnegative" else None, "top5_preservation_weight": top5_preservation_weight, "top5_preservation_margin": top5_preservation_margin if top5_reference_model is not None else None, "top5_reference_checkpoint_sha256": top5_reference_checkpoint_sha256 if top5_reference_model is not None else None, "batch_classes": classes_per_batch, "rows_per_class": rows_per_class},
        "precision": "amp_fp16" if amp else "fp32",
        "optimizer": {"name": "AdamW", "foreach": False, "learning_rate": learning_rate, "weight_decay": 1.0e-4, "gradient_clip_foreach": False},
        "seed": seed,
        "state_dict": model.state_dict(),
        "initialization": "random; frozen paired CE outputs provide only the in-batch Top-5 set-preservation constraint" if top5_reference_model is not None else "random; no existing model weights, teacher outputs, CROHME, or project-owned evaluation data",
    }, checkpoint_path)
    top1, top5 = _predict_correctness(model, validation_dataset, device, eval_batch_size)
    result = {
        "arm": arm,
        "seed": seed,
        "architecture": {"width": 128, "layers": 4, "heads": 4, "feedforward": 512, "parameters": parameter_count},
        "objective": {"ce_weight": 1.0, "supcon_weight": contrastive_weight if arm == "ce_plus_supcon" else 0.0, "temperature": temperature if arm == "ce_plus_supcon" else None, "hardnegative_weight": hardnegative_weight if arm == "ce_plus_hardnegative" else 0.0, "hardnegative_margin": hardnegative_margin if arm == "ce_plus_hardnegative" else None, "hardnegative_gate": "target_in_current_top5" if arm == "ce_plus_hardnegative" and hardnegative_top_k == 5 else "all_examples" if arm == "ce_plus_hardnegative" else None, "top5_preservation_weight": top5_preservation_weight, "top5_preservation_margin": top5_preservation_margin if top5_reference_model is not None else None, "top5_reference_checkpoint_sha256": top5_reference_checkpoint_sha256 if top5_reference_model is not None else None},
        "sampling": {"classes_per_batch": classes_per_batch, "rows_per_class": rows_per_class, "batch_size": batch_size, "batches_per_epoch": batch_count, "rows_per_epoch": batch_count * batch_size, "train_rows": len(train_dataset), "distinct_rows_within_batch": True},
        "precision": "amp_fp16" if amp else "fp32",
        "optimizer": {"name": "AdamW", "foreach": False, "learning_rate": learning_rate, "weight_decay": 1.0e-4, "gradient_clip_norm": 1.0, "gradient_clip_foreach": False},
        "history": history,
        "final_validation": history[-1]["validation"],
        "top5_reference": top5_reference_metadata,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "elapsed_seconds": time.perf_counter() - started,
    }
    if device.type == "cuda":
        del model, optimizer, scaler, loader, sampler, features, target, embedding, logits, ce, supcon, hardnegative, top5_preservation, hardnegative_active, loss
        if top5_reference_model is not None:
            del top5_reference_model
        gc.collect()
        torch.cuda.empty_cache()
    return result, top1, top5


@torch.inference_mode()
def _evaluate_validation(model: nn.Module, dataset: NpyDataset, device: torch.device, batch_size: int) -> dict[str, Any]:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    top1_hits = top5_hits = 0
    rows = 0
    for features, target in loader:
        logits = model(features.to(device, non_blocking=device.type == "cuda")).float()
        target = target.to(device, non_blocking=device.type == "cuda")
        indices = logits.topk(5, dim=1).indices
        top1_hits += int((indices[:, 0] == target).sum())
        top5_hits += int((indices == target[:, None]).any(dim=1).sum())
        rows += len(target)
    return {"rows": rows, "top1_hits": top1_hits, "top5_hits": top5_hits, "top1": top1_hits / rows, "top5": top5_hits / rows}


def _reuse_completed_ce_arm(
    checkpoint_path: Path,
    attestation_report_path: Path,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    classes_per_batch: int,
    rows_per_class: int,
    learning_rate: float,
    precision: str,
    labels: list[str],
    train_dataset: NpyDataset,
    validation_dataset: NpyDataset,
    device: torch.device,
    eval_batch_size: int,
    microbatch_size: int,
    cache_provenance: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Reuse a completed CE arm only when its source, config, checkpoint, and metrics reconcile."""
    report_path = attestation_report_path.resolve()
    checkpoint_file = checkpoint_path.resolve()
    if not report_path.is_file() or not checkpoint_file.is_file():
        raise FileNotFoundError("CE attestation report and checkpoint must both exist")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report_schema = report.get("schema")
    if report_schema == "aiflow-hwr-objective-competition-partial-run/v1":
        if report.get("status") != "partial_ce_arm_complete_supcon_arm_failed":
            raise ValueError("partial report does not attest a completed CE arm")
        partial_data = report.get("data", {})
        experiment = report.get("experiment", {})
        completed = report.get("completed_arm", {})
        failed = report.get("failed_arm", {})
        if failed.get("arm") != "ce_plus_supcon" or failed.get("paired_comparison") is not None:
            raise ValueError("partial report does not contain the expected unpaired CE checkpoint")
        cache_rows = {"train": partial_data.get("train_rows"), "validation": partial_data.get("validation_rows")}
        prepared_hashes = partial_data.get("prepared_cache_sha256")
        source_report_hashes = partial_data
        cache_audit_report_hash = partial_data.get("cache_audit_report_sha256")
        is_full_report = False
    elif report_schema == "aiflow-hwr-objective-competition-probe/v2":
        if report.get("status") != "completed_exploratory_objective_ablation":
            raise ValueError("completed objective report status mismatch")
        report_data = report.get("data", {})
        full_experiment = report.get("experiment", {})
        if full_experiment.get("product_adopted") is not False:
            raise ValueError("completed objective report is not explicitly shadow-only")
        if seed not in full_experiment.get("seeds", []):
            raise ValueError("completed objective report does not include the requested seed")
        seed_matches = [item for item in report.get("seed_runs", []) if item.get("seed") == seed]
        if len(seed_matches) != 1:
            raise ValueError("completed objective report must contain exactly one matching seed run")
        seed_run = seed_matches[0]
        completed = seed_run.get("arms", {}).get("ce_pk", {})
        failed = {}
        cache_audit = report_data.get("cache_audit", {})
        experiment = {
            "seed": seed,
            "architecture": full_experiment.get("architecture_fixed"),
            "precision": full_experiment.get("precision"),
            "sampler": full_experiment.get("sampler_fixed", {}),
            "epochs_requested": full_experiment.get("epochs"),
            "eval_batch_size": full_experiment.get("eval_batch_size"),
            "optimizer": full_experiment.get("optimizer", {}),
            "product_adopted": full_experiment.get("product_adopted"),
        }
        partial_data = {
            "crohme_rows": report_data.get("crohme_rows"),
            "cache_reuse_mode": report_data.get("cache_reuse_provenance", {}).get("mode"),
            "canonical_hwrt_sha256": report_data.get("canonical_hwrt_sha256"),
            "canonical_uji_sha256": report_data.get("canonical_uji_sha256"),
            "curated_uji_sha256": report_data.get("curated_uji_sha256"),
            "writer_split_sha256": report_data.get("writer_split_sha256"),
            "prepared_cache_sha256": cache_audit.get("prepared_cache_sha256"),
            "train_rows": cache_audit.get("rows", {}).get("train"),
            "validation_rows": cache_audit.get("rows", {}).get("validation"),
        }
        source_report_hashes = partial_data
        prepared_hashes = partial_data.get("prepared_cache_sha256")
        cache_rows = {"train": partial_data.get("train_rows"), "validation": partial_data.get("validation_rows")}
        cache_audit_report_hash = None
        is_full_report = True
        if report_data.get("cache_audit", {}).get("crohme_rows") != 0:
            raise ValueError("completed objective report cache audit includes CROHME rows")
    else:
        raise ValueError("reusable CE report schema mismatch")

    if partial_data.get("crohme_rows") != 0:
        raise ValueError("reusable CE report did not exclude CROHME")
    source_hashes = cache_provenance.get("source_hashes", {})
    source_fields = {
        "canonical_hwrt_sha256": "canonical_hwrt_sha256",
        "canonical_uji_sha256": "canonical_uji_sha256",
        "curated_uji_sha256": "curated_uji_sha256",
        "writer_split_sha256": "writer_split_sha256",
    }
    for report_key, provenance_key in source_fields.items():
        if source_report_hashes.get(report_key) != source_hashes.get(provenance_key):
            raise ValueError(f"reusable CE source fingerprint mismatch: {report_key}")
    if prepared_hashes != cache_provenance.get("prepared_cache_sha256"):
        raise ValueError("reusable CE prepared-array fingerprints differ from the current cache")
    if not is_full_report and cache_audit_report_hash != cache_provenance.get("audit_report_sha256"):
        raise ValueError("reusable CE cache-audit fingerprint differs from the current attestation")
    if cache_rows.get("train") != len(train_dataset) or cache_rows.get("validation") != len(validation_dataset):
        raise ValueError("reusable CE row counts differ from the current prepared arrays")

    if experiment.get("product_adopted", report.get("product_adopted")) is not False:
        raise ValueError("reusable CE report is not explicitly shadow-only")
    expected_architecture = {"width": 128, "layers": 4, "heads": 4, "feedforward": 512}
    expected_sampler = {
        "classes_per_batch": classes_per_batch,
        "rows_per_class": rows_per_class,
        "batch_size": batch_size,
    }
    if experiment.get("seed") != seed or experiment.get("architecture") != expected_architecture:
        raise ValueError("reusable CE seed or architecture differs from the requested run")
    if experiment.get("precision") != precision:
        raise ValueError("reusable CE precision differs from the requested run")
    reported_sampler = experiment.get("sampler", {})
    if any(reported_sampler.get(key) != value for key, value in expected_sampler.items()):
        raise ValueError("reusable CE P×K sampler differs from the requested run")
    expected_microbatch = {
        "microbatch_size": microbatch_size,
        "gradient_accumulation_steps": batch_size // microbatch_size,
    }
    if any(reported_sampler.get(key) != value for key, value in expected_microbatch.items()):
        raise ValueError("reusable CE microbatch differs from the requested run")
    if is_full_report and reported_sampler.get("same_seeded_batch_schedule_across_arms") is not True:
        raise ValueError("completed CE report does not attest the shared seeded batch schedule")
    if experiment.get("epochs_requested") != epochs:
        raise ValueError("reusable CE epoch count differs from the requested run")
    if is_full_report and experiment.get("eval_batch_size") != eval_batch_size:
        raise ValueError("reusable CE evaluation batch size differs from the requested run")
    optimizer_report = experiment.get("optimizer", {})
    expected_optimizer = {
        "name": "AdamW",
        "learning_rate": learning_rate,
        "weight_decay": 1.0e-4,
        "foreach": False,
        "gradient_clip_norm": 1.0,
        "gradient_clip_foreach": False,
    }
    if optimizer_report != expected_optimizer:
        raise ValueError("reusable CE optimizer settings differ from the requested run")

    if completed.get("arm") != "ce_pk" or completed.get("seed") != seed:
        raise ValueError("attestation report does not contain the expected CE arm and seed")
    recorded_checkpoint = Path(completed.get("checkpoint", "")).resolve()
    if recorded_checkpoint != checkpoint_file:
        raise ValueError("reusable CE checkpoint path does not match the attestation report")
    checkpoint_hash = _sha256(checkpoint_file)
    if checkpoint_hash != completed.get("checkpoint_sha256"):
        raise ValueError("reusable CE checkpoint hash differs from the attestation report")

    payload = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    if payload.get("schema") != "aiflow-hwr-objective-competition-probe/v1" or payload.get("arm") != "ce_pk":
        raise ValueError("reusable checkpoint schema or arm mismatch")
    if payload.get("seed") != seed or payload.get("precision") != precision:
        raise ValueError("reusable checkpoint seed or precision mismatch")
    if payload.get("architecture", {}) | {"parameters": None} != expected_architecture | {"parameters": None}:
        raise ValueError("reusable checkpoint architecture mismatch")
    checkpoint_objective = payload.get("objective", {})
    if checkpoint_objective.get("ce_weight") != 1.0 or checkpoint_objective.get("supcon_weight") != 0.0:
        raise ValueError("reusable checkpoint is not a pure CE arm")
    if checkpoint_objective.get("batch_classes") != classes_per_batch or checkpoint_objective.get("rows_per_class") != rows_per_class:
        raise ValueError("reusable checkpoint P×K objective metadata mismatch")
    checkpoint_optimizer = payload.get("optimizer", {})
    if (
        checkpoint_optimizer.get("name") != "AdamW"
        or checkpoint_optimizer.get("foreach") is not False
        or checkpoint_optimizer.get("learning_rate") != learning_rate
        or checkpoint_optimizer.get("weight_decay") != 1.0e-4
        or checkpoint_optimizer.get("gradient_clip_foreach") is not False
    ):
        raise ValueError("reusable checkpoint optimizer metadata mismatch")

    model = ScaledInkClassifier(len(labels), 128, 4, 4, 512).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    measured_top1, measured_top5 = _predict_correctness(model, validation_dataset, device, eval_batch_size)
    measured_validation = {
        "rows": int(len(measured_top1)),
        "top1_hits": int(measured_top1.sum()),
        "top5_hits": int(measured_top5.sum()),
        "top1": float(measured_top1.mean()),
        "top5": float(measured_top5.mean()),
    }
    reported_validation = completed.get("final_validation", {})
    history = completed.get("history", [])
    last_epoch = history[-1] if history else {}
    expected_metrics = {
        "rows": reported_validation.get("rows", last_epoch.get("validation_rows")),
        "top1_hits": reported_validation.get("top1_hits", last_epoch.get("validation_top1_hits")),
        "top5_hits": reported_validation.get("top5_hits", last_epoch.get("validation_top5_hits")),
    }
    if any(expected_metrics[key] is None or expected_metrics[key] != measured_validation[key] for key in expected_metrics):
        raise ValueError(f"reusable checkpoint metrics do not reproduce the attestation report: {expected_metrics} vs {measured_validation}")
    if len(labels) != 372:
        raise ValueError("reusable CE checkpoint requires the pinned 372-class vocabulary")

    result = {
        "arm": "ce_pk",
        "seed": seed,
        "architecture": {**expected_architecture, "parameters": sum(parameter.numel() for parameter in model.parameters())},
        "objective": {"ce_weight": 1.0, "supcon_weight": 0.0, "temperature": None},
        "sampling": {**expected_sampler, "microbatch_size": microbatch_size, "gradient_accumulation_steps": batch_size // microbatch_size, "train_rows": len(train_dataset), "distinct_rows_within_batch": True},
        # This describes checkpoint training precision, not this FP32 validation replay.
        "precision": payload["precision"],
        "optimizer": expected_optimizer,
        "history": history,
        "final_validation": measured_validation,
        "checkpoint": str(checkpoint_file),
        "checkpoint_sha256": checkpoint_hash,
        "elapsed_seconds": 0.0,
        "reused_completed_checkpoint": True,
        "reuse_source_attestation_report": str(report_path),
        "reuse_source_attestation_report_sha256": _sha256(report_path),
    }
    del model, payload
    if device.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
    return result, measured_top1, measured_top5


def _self_test() -> None:
    torch.manual_seed(31)
    embeddings = torch.randn(8, 16, requires_grad=True)
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    loss = supervised_contrastive_loss(embeddings, labels, 0.1)
    loss.backward()
    if not torch.isfinite(loss) or embeddings.grad is None or not torch.isfinite(embeddings.grad).all():
        raise AssertionError("SupCon self-test did not produce finite gradients")
    margin_logits = torch.tensor([[2.0, 0.0, 1.0], [0.0, 1.0, 2.0]], requires_grad=True)
    margin_labels = torch.tensor([0, 1])
    margin_loss, active, eligible = hardnegative_margin_loss(margin_logits, margin_labels, 0.5)
    margin_loss.backward()
    if abs(float(margin_loss.detach()) - 0.75) > 1.0e-6 or int(active) != 1 or int(eligible) != 2 or margin_logits.grad is None or not torch.isfinite(margin_logits.grad).all():
        raise AssertionError("hard-negative margin self-test failed its expected loss/gradient")
    gate_logits = torch.tensor([[3.0, 2.8, 1.0, 0.0], [4.0, 3.0, 2.0, 1.0]], requires_grad=True)
    gate_labels = torch.tensor([0, 2])
    gate_loss, gate_active, gate_eligible = hardnegative_margin_loss(gate_logits, gate_labels, 0.5, target_top_k=2)
    gate_loss.backward()
    if abs(float(gate_loss.detach()) - 0.15) > 1.0e-6 or int(gate_active) != 1 or int(gate_eligible) != 1:
        raise AssertionError("Top-5-gated hard-negative self-test violated its eligibility/loss counts")
    if gate_logits.grad is None or not torch.isfinite(gate_logits.grad).all() or not torch.equal(gate_logits.grad[1], torch.zeros_like(gate_logits.grad[1])):
        raise AssertionError("Top-5-gated hard-negative leaked gradient into an ineligible row")

    reference_set_logits = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0, 0.0]])
    preserved_logits = reference_set_logits.clone().requires_grad_(True)
    preserved_loss, preserved_active = top5_set_preservation_loss(preserved_logits, reference_set_logits)
    if float(preserved_loss.detach()) != 0.0 or int(preserved_active) != 0:
        raise AssertionError("Top-5 set-preservation penalized an unchanged reference candidate set")
    intrusion_logits = torch.tensor([[5.0, 4.0, 3.0, 2.0, 0.5, 1.5]], requires_grad=True)
    intrusion_loss, intrusion_active = top5_set_preservation_loss(intrusion_logits, reference_set_logits)
    intrusion_loss.backward()
    if abs(float(intrusion_loss.detach()) - 0.2) > 1.0e-6 or int(intrusion_active) != 1:
        raise AssertionError("Top-5 set-preservation did not detect an outside-class intrusion")
    if intrusion_logits.grad is None or not torch.isfinite(intrusion_logits.grad).all() or intrusion_logits.grad[0, 5] <= 0.0 or intrusion_logits.grad[0, 4] >= 0.0:
        raise AssertionError("Top-5 set-preservation gradient did not push the displaced candidate set back together")

    class _ToyClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Linear(3, 5)
            self.math_head = nn.Linear(5, 6)

        def encode(self, points: torch.Tensor) -> torch.Tensor:
            return torch.tanh(self.encoder(points))

    torch.manual_seed(73)
    full_model = _ToyClassifier().eval()
    micro_model = _ToyClassifier().eval()
    micro_model.load_state_dict(full_model.state_dict())
    reference_model = _ToyClassifier().eval()
    test_features = torch.randn(8, 3)
    test_labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    full_logits = full_model.math_head(full_model.encode(test_features))
    full_ce = F.cross_entropy(full_logits, test_labels)
    full_hn, full_active, full_eligible = hardnegative_margin_loss(full_logits, test_labels, 0.5)
    with torch.no_grad():
        reference_logits = reference_model.math_head(reference_model.encode(test_features))
    full_top5, _ = top5_set_preservation_loss(full_logits, reference_logits)
    (full_ce + 0.1 * full_hn + 0.05 * full_top5).backward()
    micro_state = {"phase": "test"}
    micro_stats = _backward_microbatched_ce_hardnegative(
        micro_model, test_features, test_labels, 2,
        use_hardnegative=True, hardnegative_weight=0.1, hardnegative_margin=0.5,
        hardnegative_top_k=None,
        top5_reference_model=reference_model,
        top5_preservation_weight=0.05,
        top5_preservation_margin=0.0,
        amp=False, scaler=torch.amp.GradScaler("cpu", enabled=False),
        capture_outputs=True, phase_state=micro_state,
    )
    expected_combined_loss = full_ce + 0.1 * full_hn + 0.05 * full_top5
    if abs(micro_stats["loss"] - float(expected_combined_loss.detach())) > 1.0e-6:
        raise AssertionError("microbatch accumulation changed the effective CE+hard-negative+Top-5-preservation loss")
    if abs(micro_stats["top5_preservation"] - float(full_top5.detach())) > 1.0e-6:
        raise AssertionError("microbatch accumulation changed the Top-5-preservation component")
    if micro_stats["active"] != int(full_active) or micro_stats["eligible"] != int(full_eligible) or micro_stats["correct"] != int((full_logits.argmax(dim=1) == test_labels).sum()):
        raise AssertionError("microbatch accumulation changed active-hinge or Top-1 counts")
    for full_parameter, micro_parameter in zip(full_model.parameters(), micro_model.parameters(), strict=True):
        if full_parameter.grad is None or micro_parameter.grad is None or not torch.allclose(full_parameter.grad, micro_parameter.grad, atol=1.0e-6, rtol=1.0e-6):
            raise AssertionError("microbatch CE+hard-negative+Top-5-preservation gradient differs from full-batch gradient")

    class _ToySupConClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.projection = nn.Linear(3, 6)
            self.dropout = nn.Dropout(0.25)
            self.math_head = nn.Linear(6, 4)

        def encode(self, points: torch.Tensor) -> torch.Tensor:
            return self.dropout(torch.tanh(self.projection(points)))

    torch.manual_seed(89)
    full_supcon_model = _ToySupConClassifier().eval()
    micro_supcon_model = _ToySupConClassifier().eval()
    micro_supcon_model.load_state_dict(full_supcon_model.state_dict())
    supcon_features = torch.randn(8, 3)
    supcon_labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    supcon_weight, supcon_temperature = 0.07, 0.2
    full_embedding = full_supcon_model.encode(supcon_features)
    full_logits = full_supcon_model.math_head(full_embedding)
    full_ce = F.cross_entropy(full_logits, supcon_labels)
    full_supcon = supervised_contrastive_loss(full_embedding, supcon_labels, supcon_temperature)
    full_combined = full_ce + supcon_weight * full_supcon
    full_combined.backward()
    supcon_phase = {"phase": "self_test"}
    micro_supcon_stats = _backward_microbatched_ce_supcon(
        micro_supcon_model, supcon_features, supcon_labels, 2,
        contrastive_weight=supcon_weight,
        temperature=supcon_temperature,
        amp=False,
        scaler=torch.amp.GradScaler("cpu", enabled=False),
        capture_outputs=True,
        phase_state=supcon_phase,
    )
    if abs(micro_supcon_stats["loss"] - float(full_combined.detach())) > 1.0e-6:
        raise AssertionError("microbatched CE+SupCon changed the full P×K objective value")
    if micro_supcon_stats["positive_anchors"] != 8 or micro_supcon_stats["replay_max_abs_embedding_error"] > 1.0e-6:
        raise AssertionError("microbatched SupCon lost positive pairs or failed deterministic replay")
    for full_parameter, micro_parameter in zip(full_supcon_model.parameters(), micro_supcon_model.parameters(), strict=True):
        if full_parameter.grad is None or micro_parameter.grad is None or not torch.allclose(full_parameter.grad, micro_parameter.grad, atol=1.0e-5, rtol=1.0e-5):
            raise AssertionError("microbatched CE+SupCon gradient differs from the full-batch gradient")

    dropout_replay_model = _ToySupConClassifier().train()
    torch.manual_seed(97)
    dropout_replay_stats = _backward_microbatched_ce_supcon(
        dropout_replay_model, supcon_features, supcon_labels, 2,
        contrastive_weight=supcon_weight,
        temperature=supcon_temperature,
        amp=False,
        scaler=torch.amp.GradScaler("cpu", enabled=False),
        capture_outputs=True,
        phase_state={"phase": "dropout_replay_self_test"},
    )
    if dropout_replay_stats["replay_max_abs_embedding_error"] > 1.0e-6:
        raise AssertionError("SupCon checkpoint recomputation did not reproduce dropout embeddings")
    if not all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in dropout_replay_model.parameters()):
        raise AssertionError("dropout replay did not produce finite SupCon model gradients")

    torch.manual_seed(101)
    encoder_smoke_model = ScaledInkClassifier(372, 128, 4, 4, 512).train()
    encoder_smoke_features = torch.randn(8, 128, 5)
    encoder_smoke_labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    encoder_smoke_stats = _backward_microbatched_ce_supcon(
        encoder_smoke_model, encoder_smoke_features, encoder_smoke_labels, 2,
        contrastive_weight=supcon_weight,
        temperature=supcon_temperature,
        amp=False,
        scaler=torch.amp.GradScaler("cpu", enabled=False),
        capture_outputs=True,
        phase_state={"phase": "production_encoder_smoke_test"},
    )
    if encoder_smoke_stats["replay_max_abs_embedding_error"] > 1.0e-5:
        raise AssertionError("production Transformer encoder failed SupCon activation replay")
    if not all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in encoder_smoke_model.parameters()):
        raise AssertionError("production Transformer encoder failed microbatched SupCon backward")

    clip_model = nn.Linear(2, 1, bias=False)
    clip_model.weight.grad = torch.tensor([[3.0, 4.0]])
    clipped_norm = _clip_grad_norm_low_memory(clip_model, 1.0)
    if abs(float(clipped_norm) - 5.0) > 1.0e-6 or not torch.allclose(clip_model.weight.grad, torch.tensor([[0.6, 0.8]]), atol=1.0e-6):
        raise AssertionError("low-memory global gradient clipping failed its exact norm/scale check")
    nonfinite_model = nn.Linear(2, 1, bias=False)
    nonfinite_model.weight.grad = torch.tensor([[float("inf"), float("nan")]])
    try:
        _clip_grad_norm_low_memory(nonfinite_model, 1.0)
    except FloatingPointError as exc:
        diagnostics = getattr(exc, "hwr_gradient_diagnostics", {})
        other = diagnostics.get("by_component", {}).get("other", {})
        if (
            diagnostics.get("nonfinite_gradient_element_count") != 2
            or other.get("nan_elements") != 1
            or other.get("positive_inf_elements") != 1
            or other.get("nonfinite_parameters") != ["weight"]
        ):
            raise AssertionError("non-finite gradient microscope lost component/element identity") from exc
    else:
        raise AssertionError("non-finite gradient microscope did not fail closed")
    scaler_model = nn.Linear(1, 1, bias=False)
    scaler_optimizer = torch.optim.SGD(scaler_model.parameters(), lr=0.1)
    scaler_probe = torch.amp.GradScaler("cpu", init_scale=8.0, backoff_factor=0.5, growth_interval=1)
    scaler_before = scaler_model.weight.detach().clone()
    scaler_probe.scale(scaler_model(torch.ones(1, 1)).sum()).backward()
    scaler_model.weight.grad.fill_(float("inf"))
    scaler_probe.unscale_(scaler_optimizer)
    scaler_probe.step(scaler_optimizer)
    scaler_probe.update()
    if not torch.equal(scaler_model.weight.detach(), scaler_before) or scaler_probe.get_scale() != 4.0:
        raise AssertionError("AMP overflow was not skipped with the expected scaler backoff")
    sampler = PKBatchSampler(np.repeat(np.arange(8), 8), 4, 2, 3, 7)
    first = list(sampler)
    sampler2 = PKBatchSampler(np.repeat(np.arange(8), 8), 4, 2, 3, 7)
    second = list(sampler2)
    if first != second or any(len(batch) != 8 or len(set(batch)) != 8 for batch in first):
        raise AssertionError("P×K sampler is not deterministic or contains duplicates")
    for batch in first:
        counts = np.bincount(np.repeat(np.arange(8), 8)[batch], minlength=8)
        if sorted(counts[counts > 0].tolist()) != [2, 2, 2, 2]:
            raise AssertionError("P×K sampler violated exact class balance")
    print(json.dumps({
        "self_test": "pass",
        "supcon_finite_gradient": True,
        "hardnegative_finite_gradient": True,
        "top5_set_preservation_exact_gate_and_gradient": True,
        "microbatch_loss_gradient_equivalent": True,
        "microbatch_top5_preservation_loss_gradient_equivalent": True,
        "supcon_microbatch_global_pairs": True,
        "supcon_microbatch_gradient_equivalent": True,
        "supcon_dropout_rng_replay": True,
        "production_transformer_microbatch_backward": True,
        "nonfinite_gradient_component_microscope": True,
        "amp_overflow_skip_and_backoff": True,
        "supcon_replay_max_abs_embedding_error": max(
            micro_supcon_stats["replay_max_abs_embedding_error"],
            dropout_replay_stats["replay_max_abs_embedding_error"],
            encoder_smoke_stats["replay_max_abs_embedding_error"],
        ),
        "low_memory_clip_exact": True,
        "pk_sampler_deterministic": True,
        "pk_exact_balance": True,
        "host_memory": _host_memory_snapshot(),
    }))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path, default=DEFAULT_CANONICAL_ROOT)
    parser.add_argument("--curated", type=Path, default=DEFAULT_CURATED)
    parser.add_argument("--writer-split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--classes-per-batch", type=int, default=24)
    parser.add_argument("--rows-per-class", type=int, default=4)
    parser.add_argument("--microbatch-size", type=int, default=0, help="activation-recompute slice size; SupCon pairs still span the full effective P×K batch; 0 keeps full-batch execution")
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--contrastive-weight", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--auxiliary-objective", choices=("supcon", "hardnegative"), default="supcon")
    parser.add_argument("--hardnegative-weight", type=float, default=0.1)
    parser.add_argument("--hardnegative-margin", type=float, default=0.5)
    parser.add_argument("--hardnegative-gate", choices=("all", "target_top5"), default="all", help="apply the margin to all examples or only rows whose target is currently inside Top-5")
    parser.add_argument("--top5-preservation-weight", type=float, default=0.0, help="hinge weight preserving the paired frozen CE model's Top-5 candidate set")
    parser.add_argument("--top5-preservation-margin", type=float, default=0.0, help="minimum logit margin between each reference Top-5 class and every outside class")
    parser.add_argument("--seeds", default="20261002", help="comma-separated matched seeds; begin with one diagnostic seed")
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--precision", choices=("amp", "fp32"), default="amp", help="FP32 bypasses CUDA AMP unscale failures on this 4 GB GTX 1650")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--reuse-audited-cache", action="store_true", help="reuse only cache/source bytes pinned by the completed paired audit; fails closed on any hash mismatch")
    parser.add_argument("--preflight-only", action="store_true", help="validate data/split/sampler support without training or writing files")
    parser.add_argument("--reuse-ce-checkpoint", type=Path, help="reuse a completed CE arm only when paired with its hash-bound attestation report")
    parser.add_argument("--reuse-ce-partial-report", "--reuse-ce-report", dest="reuse_ce_report", type=Path, help="partial or completed report that attests the CE checkpoint and data/config provenance")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return 0
    if args.epochs < 1 or args.batch_size < 1 or args.eval_batch_size < 1 or args.bootstrap_draws < 100:
        parser.error("epochs and batch sizes must be positive; bootstrap draws must be at least 100")
    if args.batch_size != args.classes_per_batch * args.rows_per_class:
        parser.error("batch-size must equal classes-per-batch × rows-per-class")
    if args.microbatch_size < 0 or args.microbatch_size > args.batch_size:
        parser.error("microbatch-size must be 0 or between 1 and batch-size")
    if args.microbatch_size and args.batch_size % args.microbatch_size:
        parser.error("batch-size must be divisible by microbatch-size")
    if args.learning_rate <= 0.0 or args.temperature <= 0.0 or args.contrastive_weight < 0.0 or args.hardnegative_weight < 0.0 or args.hardnegative_margin <= 0.0:
        parser.error("learning rate/temperature/margin must be positive; auxiliary weights cannot be negative")
    if args.top5_preservation_weight < 0.0 or args.top5_preservation_margin < 0.0:
        parser.error("Top-5 preservation weight and margin cannot be negative")
    if args.top5_preservation_weight > 0.0 and args.auxiliary_objective != "hardnegative":
        parser.error("--top5-preservation-weight currently applies only to the hardnegative objective")
    if args.auxiliary_objective != "hardnegative" and args.hardnegative_gate != "all":
        parser.error("--hardnegative-gate applies only to the hardnegative objective")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    if args.device == "cpu" and args.precision == "amp":
        parser.error("--precision amp requires --device cuda; use --precision fp32 for CPU")
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if not seeds or len(set(seeds)) != len(seeds):
        parser.error("seeds must be a non-empty list without duplicates")
    if bool(args.reuse_ce_checkpoint) != bool(args.reuse_ce_report):
        parser.error("--reuse-ce-checkpoint and its --reuse-ce-report attestation must be supplied together")
    if args.top5_preservation_weight > 0.0 and len(seeds) != 1:
        parser.error("Top-5 preservation currently requires one paired seed per run")
    if args.reuse_ce_checkpoint:
        if len(seeds) != 1:
            parser.error("CE checkpoint reuse is limited to one explicitly matched seed")
    if args.report.exists() or args.run_root.exists():
        raise FileExistsError("refusing to overwrite an existing objective run root or report")
    challenger_arm = "ce_plus_supcon" if args.auxiliary_objective == "supcon" else "ce_plus_hardnegative"
    hardnegative_top_k = 5 if args.auxiliary_objective == "hardnegative" and args.hardnegative_gate == "target_top5" else None
    device = torch.device(args.device)
    report_schema = None
    if args.reuse_ce_report:
        if not args.reuse_ce_report.is_file():
            raise FileNotFoundError("CE report attestation does not exist")
        report_schema = json.loads(args.reuse_ce_report.read_text(encoding="utf-8")).get("schema")
    if report_schema == "aiflow-hwr-objective-competition-probe/v2":
        labels, cache_audit, cache_reuse_provenance = _reuse_objective_report_attested_cache(
            args.canonical_root.resolve(), args.curated.resolve(), args.writer_split.resolve(),
            args.cache_root.resolve(), args.reuse_ce_report,
        )
    elif report_schema == "aiflow-hwr-objective-competition-partial-run/v1":
        labels, cache_audit, cache_reuse_provenance = _reuse_partial_report_attested_cache(
            args.canonical_root.resolve(), args.curated.resolve(), args.writer_split.resolve(),
            args.cache_root.resolve(), args.reuse_ce_report,
        )
    elif args.reuse_audited_cache:
        labels, cache_audit, cache_reuse_provenance = _reuse_hash_attested_cache(
            args.canonical_root.resolve(), args.curated.resolve(), args.writer_split.resolve(), args.cache_root.resolve(),
        )
    else:
        if args.reuse_ce_report:
            raise ValueError("CE report schema cannot attest cache reuse")
        labels = _class_labels(args.canonical_root.resolve())
        cache_audit = _inspect_prepared_cache(
            args.canonical_root.resolve(), args.curated.resolve(), args.writer_split.resolve(),
            labels, args.cache_root.resolve(), SYNTHETIC_EQUAL_ROWS,
        )
        cache_reuse_provenance = {"mode": "full_source_scan", "full_jsonl_rebuild_scan": True}
    train_dataset = NpyDataset(args.cache_root / "train_features.npy", args.cache_root / "train_labels.npy")
    validation_dataset = NpyDataset(args.cache_root / "validation_features.npy", args.cache_root / "validation_labels.npy")
    if args.batch_size != args.classes_per_batch * args.rows_per_class:
        raise AssertionError("invalid P×K contract")
    split = json.loads(args.writer_split.read_text(encoding="utf-8"))
    selected_writer_hashes = set(split["inner_split"]["validation_writer_hashes"])
    writer_map = _load_writer_map(args.curated.resolve())
    writer_hashes = _jsonl_train_uji_writer_hashes(args.canonical_root.resolve(), writer_map, selected_writer_hashes, set(labels))
    if len(writer_hashes) != len(validation_dataset) or len(set(writer_hashes)) != 8:
        raise ValueError("validation examples do not align to exactly eight held-out writers")
    if args.preflight_only:
        support = np.bincount(np.asarray(train_dataset.labels, dtype=np.int64), minlength=len(labels))
        summary = {
            "preflight": "pass",
            "train_rows": len(train_dataset),
            "validation_rows": len(validation_dataset),
            "train_classes": int(np.count_nonzero(support)),
            "minimum_class_support": int(support.min()),
            "required_rows_per_class": args.rows_per_class,
            "validation_writer_clusters": len(set(writer_hashes)),
            "validation_present_labels": int(len(np.unique(validation_dataset.labels))),
            "batch_size": args.batch_size,
            "classes_per_batch": args.classes_per_batch,
            "rows_per_class": args.rows_per_class,
            "microbatch_size": args.microbatch_size or args.batch_size,
            "gradient_accumulation_steps": args.batch_size // (args.microbatch_size or args.batch_size),
            "batches_per_epoch": (len(train_dataset) + args.batch_size - 1) // args.batch_size,
            "precision": "amp_fp16" if args.precision == "amp" and device.type == "cuda" else "fp32",
            "crohme_rows": 0,
            "host_memory": _host_memory_snapshot(),
            "torch_num_threads": torch.get_num_threads(),
            "report_exists": args.report.exists(),
            "run_root_exists": args.run_root.exists(),
        }
        if args.reuse_ce_checkpoint:
            reused_result, _, _ = _reuse_completed_ce_arm(
                args.reuse_ce_checkpoint, args.reuse_ce_report,
                seed=seeds[0], epochs=args.epochs, batch_size=args.batch_size,
                classes_per_batch=args.classes_per_batch, rows_per_class=args.rows_per_class,
                learning_rate=args.learning_rate,
                precision="amp_fp16" if args.precision == "amp" and device.type == "cuda" else "fp32",
                labels=labels,
                train_dataset=train_dataset, validation_dataset=validation_dataset,
                device=device, eval_batch_size=args.eval_batch_size,
                microbatch_size=args.microbatch_size or args.batch_size,
                cache_provenance=cache_reuse_provenance,
            )
            summary["ce_checkpoint_reuse"] = {
                "status": "pass",
                "checkpoint_sha256": reused_result["checkpoint_sha256"],
                "reproduced_validation": reused_result["final_validation"],
            }
        print(json.dumps(summary, ensure_ascii=False))
        return 0
    args.run_root.mkdir(parents=True, exist_ok=False)

    report_data = {
        "cache_audit": cache_audit,
        "cache_reuse_provenance": cache_reuse_provenance,
        "writer_split_sha256": _sha256(args.writer_split),
        "canonical_hwrt_sha256": _sha256(args.canonical_root / "hwrt.jsonl.gz"),
        "canonical_uji_sha256": _sha256(args.canonical_root / "uji.jsonl.gz"),
        "curated_uji_sha256": _sha256(args.curated),
        "validation_rows": len(validation_dataset),
        "validation_writer_clusters": len(set(writer_hashes)),
        "validation_present_labels": int(len(np.unique(validation_dataset.labels))),
        "crohme_rows": 0,
        "project_owned_holdout_used": False,
    }
    report_experiment = {
        "arms": ["ce_pk", challenger_arm],
        "architecture_fixed": {"width": 128, "layers": 4, "heads": 4, "feedforward": 512},
        "sampler_fixed": {"classes_per_batch": args.classes_per_batch, "rows_per_class": args.rows_per_class, "batch_size": args.batch_size, "microbatch_size": args.microbatch_size or args.batch_size, "gradient_accumulation_steps": args.batch_size // (args.microbatch_size or args.batch_size), "same_seeded_batch_schedule_across_arms": True},
        "activation_memory_strategy": {
            "enabled": 0 < args.microbatch_size < args.batch_size,
            "method": (
                "two_pass_global_supcon_embedding_gradient_recompute"
                if args.auxiliary_objective == "supcon" and 0 < args.microbatch_size < args.batch_size
                else "per_slice_separable_loss_gradient_accumulation"
                if 0 < args.microbatch_size < args.batch_size
                else "full_effective_batch_graph"
            ),
            "supcon_pair_scope": "entire_effective_PxK_batch" if args.auxiliary_objective == "supcon" else None,
            "dropout_replay": "per-slice RNG snapshot and restore" if args.auxiliary_objective == "supcon" and 0 < args.microbatch_size < args.batch_size else None,
            "embedding_replay_error_max_abs_gate": 1.0e-5 if args.auxiliary_objective == "supcon" and 0 < args.microbatch_size < args.batch_size else None,
        },
        "objective_choice": args.auxiliary_objective,
        "objective": {
            "ce": "372-way cross-entropy",
            "supcon": "normalized embedding supervised contrastive" if args.auxiliary_objective == "supcon" else None,
            "contrastive_weight": args.contrastive_weight if args.auxiliary_objective == "supcon" else 0.0,
            "temperature": args.temperature if args.auxiliary_objective == "supcon" else None,
            "hardnegative": "mean(relu(margin + max_wrong_logit - target_logit))" if args.auxiliary_objective == "hardnegative" else None,
            "hardnegative_weight": args.hardnegative_weight if args.auxiliary_objective == "hardnegative" else 0.0,
            "hardnegative_margin": args.hardnegative_margin if args.auxiliary_objective == "hardnegative" else None,
            "hardnegative_gate": "target_in_current_top5" if hardnegative_top_k == 5 else "all_examples" if args.auxiliary_objective == "hardnegative" else None,
            "top5_preservation_loss": "mean(relu(margin + strongest_student_logit_outside_frozen_CE_Top5 - each_frozen_CE_Top5_logit))" if args.top5_preservation_weight > 0.0 else None,
            "top5_preservation_weight": args.top5_preservation_weight,
            "top5_preservation_margin": args.top5_preservation_margin if args.top5_preservation_weight > 0.0 else None,
        },
        "optimizer": {"name": "AdamW", "foreach": False, "learning_rate": args.learning_rate, "weight_decay": 1.0e-4, "gradient_clip_norm": 1.0, "gradient_clip_foreach": False},
        "precision": "amp_fp16" if args.precision == "amp" and device.type == "cuda" else "fp32",
        "device": str(device),
        "runtime_environment": {
            "torch_version": torch.__version__,
            "torch_num_threads": torch.get_num_threads(),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "host_memory_at_report_time": _host_memory_snapshot(),
        },
        "seeds": seeds,
        "epochs": args.epochs,
        "eval_batch_size": args.eval_batch_size,
        "product_adopted": False,
        "ce_arm_reuse": None if not args.reuse_ce_checkpoint else {
            "checkpoint": str(args.reuse_ce_checkpoint.resolve()),
            "attestation_report": str(args.reuse_ce_report.resolve()),
            "verified_before_auxiliary_training": True,
        },
    }

    seeds_report: list[dict[str, Any]] = []
    for seed in seeds:
        arm_results = {}
        predictions = {}
        if args.reuse_ce_checkpoint:
            result, top1, top5 = _reuse_completed_ce_arm(
                args.reuse_ce_checkpoint, args.reuse_ce_report,
                seed=seed, epochs=args.epochs, batch_size=args.batch_size,
                classes_per_batch=args.classes_per_batch, rows_per_class=args.rows_per_class,
                learning_rate=args.learning_rate,
                precision="amp_fp16" if args.precision == "amp" and device.type == "cuda" else "fp32",
                labels=labels,
                train_dataset=train_dataset, validation_dataset=validation_dataset,
                device=device, eval_batch_size=args.eval_batch_size,
                microbatch_size=args.microbatch_size or args.batch_size,
                cache_provenance=cache_reuse_provenance,
            )
            arm_results["ce_pk"] = result
            predictions["ce_pk"] = {"top1": top1, "top5": top5}
            arms_to_train = (challenger_arm,)
        else:
            arms_to_train = ("ce_pk", challenger_arm)
        for arm in arms_to_train:
            try:
                reference_result = arm_results.get("ce_pk") if arm == challenger_arm and args.top5_preservation_weight > 0.0 else None
                result, top1, top5 = _train_arm(
                    arm, seed, train_dataset, validation_dataset, labels, device, args.run_root,
                    args.epochs, args.learning_rate, args.batch_size, args.classes_per_batch,
                    args.rows_per_class, args.contrastive_weight, args.temperature,
                    args.hardnegative_weight, args.hardnegative_margin, hardnegative_top_k,
                    args.top5_preservation_weight if arm == challenger_arm else 0.0,
                    args.top5_preservation_margin,
                    Path(reference_result["checkpoint"]) if reference_result else None,
                    reference_result.get("checkpoint_sha256") if reference_result else None,
                    args.eval_batch_size, args.precision,
                    args.microbatch_size or args.batch_size,
                )
            except Exception as exc:
                if "ce_pk" in arm_results or arm == "ce_pk":
                    partial_report = {
                        "schema": "aiflow-hwr-objective-competition-partial-run/v1",
                        "status": f"partial_ce_arm_complete_{args.auxiliary_objective}_arm_failed" if "ce_pk" in arm_results else "partial_first_arm_failed",
                        "product_adopted": False,
                        "data": {
                            "train_rows": len(train_dataset),
                            "validation_rows": len(validation_dataset),
                            "validation_writer_clusters": len(set(writer_hashes)),
                            "validation_present_labels": int(len(np.unique(validation_dataset.labels))),
                            "crohme_rows": 0,
                            "cache_reuse_mode": cache_reuse_provenance.get("mode"),
                            "cache_reuse_provenance": cache_reuse_provenance,
                            "cache_audit_report": cache_reuse_provenance.get("audit_report"),
                            "cache_audit_report_sha256": cache_reuse_provenance.get("audit_report_sha256"),
                            "writer_split_sha256": report_data.get("writer_split_sha256"),
                            "canonical_hwrt_sha256": report_data.get("canonical_hwrt_sha256"),
                            "canonical_uji_sha256": report_data.get("canonical_uji_sha256"),
                            "curated_uji_sha256": report_data.get("curated_uji_sha256"),
                            "prepared_cache_sha256": cache_audit.get("prepared_cache_sha256"),
                        },
                        "experiment": {
                            "seed": seed,
                            "architecture": report_experiment["architecture_fixed"],
                            "precision": report_experiment["precision"],
                            "sampler": {
                                "classes_per_batch": args.classes_per_batch,
                                "rows_per_class": args.rows_per_class,
                                "batch_size": args.batch_size,
                                "microbatch_size": args.microbatch_size or args.batch_size,
                                "gradient_accumulation_steps": args.batch_size // (args.microbatch_size or args.batch_size),
                            },
                            "activation_memory_strategy": report_experiment["activation_memory_strategy"],
                            "epochs_requested": args.epochs,
            "eval_batch_size": args.eval_batch_size,
                            "optimizer": report_experiment["optimizer"],
                            "objective": report_experiment["objective"],
                            "product_adopted": False,
                        },
                        "completed_arm": arm_results.get("ce_pk"),
                        "failed_arm": {
                            "arm": arm,
                            "completed_epochs": getattr(exc, "hwr_completed_epochs", None),
                            "completed_epoch_history": getattr(exc, "hwr_completed_epoch_history", None),
                            "failed_location": getattr(exc, "hwr_failed_location", "training_or_validation"),
                            "error_type": type(exc).__name__,
                            "error": str(exc)[:1000],
                            "gradient_diagnostics": getattr(exc, "hwr_gradient_diagnostics", None),
                            "paired_comparison": None,
                        },
                        "interpretation_limit": "partial exploratory arm; not a paired result and not product accuracy",
                    }
                    args.report.parent.mkdir(parents=True, exist_ok=True)
                    with args.report.open("x", encoding="utf-8", newline="\n") as stream:
                        json.dump(partial_report, stream, ensure_ascii=False, indent=2)
                        stream.write("\n")
                raise
            arm_results[arm] = result
            predictions[arm] = {"top1": top1, "top5": top5}
        paired = {
            "top1": _paired(predictions["ce_pk"]["top1"], predictions[challenger_arm]["top1"]),
            "top5": _paired(predictions["ce_pk"]["top5"], predictions[challenger_arm]["top5"]),
            "top1_writer_cluster_bootstrap": _writer_bootstrap(predictions["ce_pk"]["top1"], predictions[challenger_arm]["top1"], writer_hashes, seed + 3000, args.bootstrap_draws),
            "top5_writer_cluster_bootstrap": _writer_bootstrap(predictions["ce_pk"]["top5"], predictions[challenger_arm]["top5"], writer_hashes, seed + 4000, args.bootstrap_draws),
        }
        seeds_report.append({"seed": seed, "arms": arm_results, "paired_comparison": paired})

    report = {
        "schema": "aiflow-hwr-objective-competition-probe/v2" if args.auxiliary_objective == "hardnegative" else "aiflow-hwr-objective-competition-probe/v1",
        "status": "completed_exploratory_objective_ablation",
        "data": report_data,
        "experiment": report_experiment,
        "seed_runs": seeds_report,
        "interpretation_limit": "one exploratory writer-disjoint UJI validation split; objective selection on this cohort is not fresh product acceptance",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    compact = [{"seed": item["seed"], "top1": item["paired_comparison"]["top1"], "top1_writer_ci": item["paired_comparison"]["top1_writer_cluster_bootstrap"]["delta_percentage_points_percentile_95_interval"], "top5": item["paired_comparison"]["top5"]} for item in seeds_report]
    print(json.dumps({"event": "competitive_objective_probe_complete", "report": str(args.report.resolve()), "seeds": compact, "crohme_rows": 0, "product_adopted": False}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
