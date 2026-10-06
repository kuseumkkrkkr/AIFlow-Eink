#!/usr/bin/env python3
"""Generate the v12 external-only action-evidence raw bank.

The default path is generation-only.  It cannot train or mutate the frozen HWR,
and it never opens writers096..127, Legacy, CROHME, MathWriting, or project data.
Run ``--dry-toy`` for a dependency-light contract test that reads no bank tensor.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Iterable
from functools import lru_cache

import numpy as np
import torch

from build_cleanroom_writer_style_bank_v5 import (
    _parent_fingerprints,
    _sample_latents,
    _writer_physics,
    apply_writer_style,
)
from cleanroom_pen_physics_v3 import GENERIC_CLASS_PROFILE, simulate_pen_physics_v3
from cleanroom_trajectory_profiles_v3 import trajectory_descriptor
from train_character_classifier_v1 import InkClassifierV1


ROOT = Path(__file__).resolve().parents[1]
PREPARED = ROOT / "artifacts/action_evidence_augmentation_v12_20260823_r1_prepared"
PREPARED_MANIFEST = PREPARED / "PREPARED_MANIFEST.json"
PREPARED_SOURCE = PREPARED / "prepare_action_evidence_augmentation_v12.source.py"
PREGEN_AUDIT = ROOT / "reports/V12_ACTION_EVIDENCE_PREGENERATION_INDEPENDENT_AUDIT.json"
DRY_SUMMARY = ROOT / "reports/V12_ACTION_EVIDENCE_DRY_CATALOG_SUMMARY_R3.json"
DRY_CATALOG = ROOT / "reports/V12_ACTION_EVIDENCE_DRY_CATALOG_R3.json.gz"
DESIGN = ROOT / "reports/V12_ACTION_EVIDENCE_AUGMENTATION_DESIGN_20260823.md"
EXTERNAL_BANK = ROOT / "artifacts/commercial_hwr_cleanroom_dataset_v3_20260823_r1/external_profiled_augmented.npz"
EXTERNAL_METADATA = ROOT / "artifacts/commercial_hwr_cleanroom_dataset_v3_20260823_r1/external_profiled_augmented.metadata.jsonl.gz"
CHECKPOINT = ROOT / "artifacts/commercial_hwr_cleanroom_physics_20260823_r1_shadow/commercial_hwr_cleanroom_physics_checkpoint.pt"
HELPER_SOURCES = {
    "build_cleanroom_writer_style_bank_v5.py": (ROOT / "scripts/build_cleanroom_writer_style_bank_v5.py", "a28f6f399f6075d31870d6b7906d13ddc343a4d3396c47266a490425f8b3ebbf"),
    "cleanroom_writer_style_simulator_v4.py": (ROOT / "scripts/cleanroom_writer_style_simulator_v4.py", "49de0704301c93f25d20b3c8c847aa5ca8ba70c561a3487c5e484c23aab89c9f"),
    "cleanroom_online_ink_augmentation_v2.py": (ROOT / "scripts/cleanroom_online_ink_augmentation_v2.py", "283b8b26b0843a9d096ca8fe8e615dfad1c169204d372910db9b16808c77e0f2"),
    "cleanroom_pen_physics_v3.py": (ROOT / "scripts/cleanroom_pen_physics_v3.py", "b1afcbac828c20b9f860097260e1e2e3c20a0201b057f7b27c65fb2b491b2ffd"),
    "cleanroom_trajectory_profiles_v3.py": (ROOT / "scripts/cleanroom_trajectory_profiles_v3.py", "a15ffd8d61cee69091a934e581a03b0b9e713b87dccc8cae55ecab2b21d65f42"),
    "train_character_classifier_v1.py": (ROOT / "scripts/train_character_classifier_v1.py", "c97bba856ad966fd7de84c688ed2591ed5568b5957d696f35e7a070a3c031835"),
    "character_tensor_v1.py": (ROOT / "scripts/character_tensor_v1.py", "9bd62259bef4b872e1a57052575c7f3e316541ac23688e753c116886c022f99f"),
    "training_data_guard_v1.py": (ROOT / "scripts/training_data_guard_v1.py", "42c047453bce454733b94c3fd085f0ae150259aa59e74ec05f6b4e32f28e274b"),
    "replay_evaluate_hwr_v1.py": (ROOT / "scripts/replay_evaluate_hwr_v1.py", "a4f48526f428f967a7ac4266f8b61952499f1114e0106fdeef9b0ca15f5e65ee"),
    "build_normalized_ink_v1.py": (ROOT / "scripts/build_normalized_ink_v1.py", "9e95fa77aa9fda0b2c37a7e9d4a8c5cf9a94068eaad69ab41e77949eb181a291"),
}
DEFAULT_RAW_OUTPUT = ROOT / "artifacts/action_evidence_augmentation_v12_20260823_r1_raw"
DEFAULT_POSTPROCESSED_OUTPUT = ROOT / "artifacts/action_evidence_augmentation_v12_20260823_r1_final"

SCHEMA = "aiflow-v12-action-evidence-raw-bank/v1"
EXPECTED_PREPARED_MANIFEST_SHA256 = "ac6e5153ce6cdb8b7c20979b031f85bdd05edce77786b31cf35084d744acb5a7"
EXPECTED_PREPARED_SOURCE_SHA256 = "9fed8bc4512f99e7ba1e12af109fc688b93b0522cb63faf0e3d07f3ea76ca1ba"
EXPECTED_PREGEN_AUDIT_SHA256 = "bcdae625acafaa7d89b4f822298e1bee70a3357e1d3bb0ba56c5eb0a7b51b714"
EXPECTED_PREGEN_STATUS = "INDEPENDENT_V12_ACTION_EVIDENCE_PREGENERATION_AUDIT_PASSED"
EXPECTED_DRY_SUMMARY_SHA256 = "548c777a8f1007687c6e9b782c40d0d28b4eafad2747f78c9c97332c70b87b68"
EXPECTED_DRY_CATALOG_SHA256 = "7643ba1dbd67d4699d4b8a81999c756cb323540bbd941fdcb058f2614654337d"
EXPECTED_DESIGN_SHA256 = "1b9e41933ca7efed225aaa589ccde7c2d0130022c4b0c380637c6ec1445338d5"
EXPECTED_CHECKPOINT_SHA256 = "c9ebd51e5ba72f1a1e8e2938343823808f7c0ab1bc7143e93332ef8b58143521"
EXPECTED_EXTERNAL_BANK_SHA256 = "31eb534d125f38cfbe5aa33d38545489d71737461f9f605da5304ebab8d3d577"
EXPECTED_EXTERNAL_METADATA_SHA256 = "411097fe5846ec3da6c7c6c82c47dc3da66cf2cb0e8cc1ee691edca18ec5c87b"
EXPECTED_GENERATOR_AUDIT_STATUS = "INDEPENDENT_V12_ACTION_EVIDENCE_GENERATOR_STATIC_AUDIT_PASSED"

WRITER_IDS = tuple(range(128, 192))
WRITER_SEED_ROOT = 20260824
WRITER_SEED_MULTIPLIER = 4099
TARGET_ACTIONS_PER_PAIR = 20
ORIENTATION_ROWS = 10
RAW_ATTEMPTS = 3
MIN_ADMITTED_WRITERS_PER_PAIR = 16
CALIBRATION_SUPPORT = 2
MARGIN_EDGES = (0.5, 1.5, 3.0)
ENTROPY_EDGES = (0.35, 0.60, 0.80, 1.0)
MORPH_STRENGTHS = (0.10, 0.16, 0.22)
MIN_REAL_RMS = 1.0e-5
ADMITTED_LABELS = (
    0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,49,50,51,52,53,55,56,57,58,59,60,61,62,63,64,66,68,69,70,71,72,74,75,76,77,78,80,81,82,83,85,86,87,88,89,90,91,92,93,94,97,100,101,102,103,104,105,107,108,109,110,111,112,113,114,116,117,118,119,120,121,125,126,127,128,129,130,131,132,133,134,135,136,139,140,141,142,143,144,145,146,150,151,152,153,154,156,157,158,160,161,162,164,165,166,167,168,169,170,171,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,192,193,194,195,197,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,223,224,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,243,244,245,246,247,248,249,251,252,253,254,256,257,258,259,262,263,264,265,266,267,269,270,272,273,274,275,276,278,279,280,281,282,283,284,285,286,288,289,291,292,293,294,295,296,297,298,299,300,301,302,304,305,306,307,308,309,310,311,312,313,314,315,316,317,318,319,320,321,322,323,324,325,326,327,328,329,332,333,334,335,336,337,338,339,340,341,342,343,344,345,346,347,348,350,352,353,354,355,356,357,358,359,360,361,362,363,364,365,366,367,369,370,371,
)
ADMITTED_LABELS_SHA256 = "15c87d72f3113eec71ba8e0502ce3a20979eaa172216ef1e6bb6cc68cb52ddae"
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


def tensor_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value, dtype=np.float32).tobytes()).hexdigest()


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@lru_cache(maxsize=1)
def _latents():
    return tuple(_sample_latents(192))


def _seed_offset(key: str) -> int:
    return int(hashlib.sha256(key.encode()).hexdigest()[:16], 16) & ((1 << 63) - 1)


def _bin(value: float, edges: tuple[float, ...]) -> int:
    return int(np.digitize(value, np.asarray(edges, np.float64)))


def _stroke_bounds(features: np.ndarray) -> list[tuple[int, int]]:
    starts = np.flatnonzero(np.asarray(features)[:, 3] > 0.5).tolist()
    if not starts or starts[0] != 0:
        raise ValueError("trajectory must begin with stroke_start")
    bounds = list(zip(starts, starts[1:] + [len(features)], strict=True))
    if any(end <= start for start, end in bounds):
        raise ValueError("empty stroke")
    return bounds


def _topology(features: np.ndarray) -> tuple[int, int, int]:
    descriptor = trajectory_descriptor(np.asarray(features, np.float32))
    return int(descriptor.stroke_count), int(descriptor.loop_stroke), int(descriptor.loop_direction)


def _letterbox(xy: np.ndarray) -> np.ndarray:
    low = xy.min(axis=0); high = xy.max(axis=0)
    extent = max(float(np.max(high - low)), 1.0e-7)
    return np.clip((xy - (low + high) * 0.5) / extent + 0.5, 0.0, 1.0)


def _resample(points: np.ndarray, count: int) -> np.ndarray:
    if len(points) == count:
        return points.astype(np.float64, copy=True)
    if len(points) == 1:
        return np.repeat(points.astype(np.float64), count, axis=0)
    delta = np.linalg.norm(np.diff(points, axis=0), axis=1)
    progress = np.concatenate(([0.0], np.cumsum(delta)))
    if progress[-1] <= 1.0e-9:
        return np.repeat(points[:1].astype(np.float64), count, axis=0)
    progress /= progress[-1]
    target = np.linspace(0.0, 1.0, count)
    return np.stack([np.interp(target, progress, points[:, axis]) for axis in range(2)], axis=1)


def morph_truth_toward_relative(truth: np.ndarray, relative: np.ndarray, strength: float) -> np.ndarray:
    """Morph XY only; truth stroke boundaries and uniform-time channels are immutable."""

    truth = np.asarray(truth, np.float32); relative = np.asarray(relative, np.float32)
    truth_bounds = _stroke_bounds(truth); relative_bounds = _stroke_bounds(relative)
    if len(truth_bounds) != len(relative_bounds):
        raise ValueError("stroke-count mismatch")
    aligned = np.zeros((len(truth), 2), dtype=np.float64)
    for (ts, te), (rs, re) in zip(truth_bounds, relative_bounds, strict=True):
        aligned[ts:te] = _resample(relative[rs:re, :2], te - ts)
    output = truth.copy()
    output[:, :2] = _letterbox((1.0 - strength) * truth[:, :2] + strength * aligned).astype(np.float32)
    if not np.array_equal(output[:, 2:], truth[:, 2:]):
        raise AssertionError("morph changed non-spatial channels")
    if _topology(output) != _topology(truth):
        raise ValueError("morph changed truth topology")
    return output


def _writer_order(pair: tuple[int, int]) -> list[int]:
    return sorted(WRITER_IDS, key=lambda writer: hashlib.sha256(
        f"{WRITER_SEED_ROOT}:pair-writer:{pair[0]}:{pair[1]}:{writer}".encode()
    ).hexdigest())[:TARGET_ACTIONS_PER_PAIR]


def orientation_for(pair: tuple[int, int], writer: int) -> str:
    slot = _writer_order(pair).index(writer)
    return "candidate_truth_promotion" if slot < ORIENTATION_ROWS else "baseline_truth_veto"


def _risk(logits: np.ndarray, stroke_count: int, candidate: int) -> dict:
    order = np.argsort(logits)[-5:][::-1].astype(int)
    matches = np.flatnonzero(order == int(candidate))
    if not len(matches):
        rank = -1
    else:
        rank = int(matches[0])
    values = logits[order].astype(np.float64)
    probability = np.exp(values - values.max()); probability /= probability.sum()
    entropy = float(-np.sum(probability * np.log(np.maximum(probability, 1.0e-12))))
    return {
        "top5": order.tolist(), "candidate_rank": rank,
        "top1_margin_bin": _bin(float(values[0] - values[1]), MARGIN_EDGES),
        "top5_entropy_bin": _bin(entropy, ENTROPY_EDGES),
        "stroke_count_bucket": min(int(stroke_count), 4),
    }


def _resolve_homographs(tokens: list[str]) -> tuple[frozenset[int], ...]:
    return tuple(
        frozenset(tokens.index(token) for token in group if token in tokens)
        for group in HOMOGRAPH_TOKEN_GROUPS
        if sum(token in tokens for token in group) >= 2
    )


def _homograph_collision(top5: list[int], groups: tuple[frozenset[int], ...]) -> bool:
    values = set(int(value) for value in top5)
    return any(len(values & group) >= 2 for group in groups)


def _risk_matches(
    actual: dict, expected: dict, baseline: int, candidate: int,
    homographs: tuple[frozenset[int], ...],
) -> bool:
    return (
        actual["top5"][0] == baseline
        and candidate in actual["top5"]
        and all(int(value) in ADMITTED_LABELS for value in actual["top5"])
        and not _homograph_collision(actual["top5"], homographs)
        and actual["candidate_rank"] == expected["candidate_rank"]
        and actual["top1_margin_bin"] == expected["top1_margin_bin"]
        and actual["top5_entropy_bin"] == expected["top5_entropy_bin"]
        and actual["stroke_count_bucket"] == expected["stroke_count_bucket"]
    )


def _load_json_gzip(path: Path) -> object:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def _load_metadata(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write_jsonl_gzip(path: Path, rows: Iterable[dict]) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as stream:
                for row in rows:
                    stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def _validate_prepared() -> dict:
    files = sorted(path.name for path in PREPARED.iterdir() if path.is_file())
    if files != ["PREPARED_MANIFEST.json", "prepare_action_evidence_augmentation_v12.source.py"]:
        raise ValueError("prepared artifact is not exact2")
    expected = (
        (PREPARED_MANIFEST, EXPECTED_PREPARED_MANIFEST_SHA256),
        (PREPARED_SOURCE, EXPECTED_PREPARED_SOURCE_SHA256),
        (PREGEN_AUDIT, EXPECTED_PREGEN_AUDIT_SHA256),
        (DRY_SUMMARY, EXPECTED_DRY_SUMMARY_SHA256),
        (DRY_CATALOG, EXPECTED_DRY_CATALOG_SHA256),
        (DESIGN, EXPECTED_DESIGN_SHA256),
    )
    for path, wanted in expected:
        if sha256(path) != wanted:
            raise ValueError(f"pinned input drift: {path.name}")
    manifest = json.loads(PREPARED_MANIFEST.read_text(encoding="utf-8"))
    audit = json.loads(PREGEN_AUDIT.read_text(encoding="utf-8"))
    if manifest.get("status") != "V12_ACTION_EVIDENCE_PREPARED_BANK_UNGENERATED":
        raise ValueError("prepared status mismatch")
    if any(manifest.get(key) is not False for key in ("writers096_127_opened", "bank_generated", "adapter_trained")):
        raise ValueError("prepared closed boundary mismatch")
    if audit.get("status") != EXPECTED_PREGEN_STATUS or not audit.get("gates") or not all(audit["gates"].values()):
        raise ValueError("pre-generation independent audit is not all-gates PASS")
    if audit.get("decision", {}).get("bank_builder_source_implementation_allowed") is not True:
        raise ValueError("builder implementation was not authorized")
    if audit.get("decision", {}).get("bank_generation_execution_allowed") is not False:
        raise ValueError("unexpected pre-generation execution permission")
    if manifest.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("prepared checkpoint binding mismatch")
    if manifest.get("external_bank_sha256") != EXPECTED_EXTERNAL_BANK_SHA256:
        raise ValueError("prepared external bank binding mismatch")
    if manifest.get("external_metadata_sha256") != EXPECTED_EXTERNAL_METADATA_SHA256:
        raise ValueError("prepared external metadata binding mismatch")
    return manifest


def _validate_live_generation_inputs() -> None:
    """Called only after STARTED is durable; this reads the live model/bank files."""

    for path, wanted in (
        (CHECKPOINT, EXPECTED_CHECKPOINT_SHA256),
        (EXTERNAL_BANK, EXPECTED_EXTERNAL_BANK_SHA256),
        (EXTERNAL_METADATA, EXPECTED_EXTERNAL_METADATA_SHA256),
    ):
        if sha256(path) != wanted:
            raise ValueError(f"pinned generation input drift: {path.name}")


def _validate_helper_sources() -> dict[str, str]:
    actual = {}
    for name, (path, wanted) in HELPER_SOURCES.items():
        value = sha256(path)
        if value != wanted:
            raise ValueError(f"generation helper drift: {name}")
        actual[name] = value
    if canonical_hash(list(ADMITTED_LABELS)) != ADMITTED_LABELS_SHA256 or len(ADMITTED_LABELS) != 325:
        raise ValueError("admitted label provenance drift")
    return actual


def _validate_generator_audit(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    gates = payload.get("gates", {})
    if payload.get("status") != EXPECTED_GENERATOR_AUDIT_STATUS or not gates or not all(value is True for value in gates.values()):
        raise ValueError("generator static audit is not all-gates PASS")
    if payload.get("generator_source_sha256") != sha256(Path(__file__)):
        raise ValueError("generator audit source mismatch")
    if payload.get("prepared_manifest_sha256") != EXPECTED_PREPARED_MANIFEST_SHA256:
        raise ValueError("generator audit prepared receipt mismatch")
    if payload.get("pregen_audit_sha256") != EXPECTED_PREGEN_AUDIT_SHA256:
        raise ValueError("generator audit pre-generation audit mismatch")
    decision = payload.get("decision", {})
    if decision.get("generation_allowed") is not True or decision.get("training_allowed") is not False:
        raise ValueError("generator audit decision mismatch")
    return {"path": str(path.resolve()), "sha256": sha256(path), "status": payload["status"]}


def _parent_pool(labels: np.ndarray, metadata: list[dict]) -> dict[tuple[int, int], list[int]]:
    output: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, (label, row) in enumerate(zip(labels.tolist(), metadata, strict=True)):
        if row.get("cross_writer") or row.get("parent_writer_fingerprints"):
            raise ValueError("non-external writer parent detected")
        if int(row.get("label_index", -1)) != int(label) or not row.get("synthetic_id") or not row.get("parents"):
            raise ValueError("external metadata provenance mismatch")
        strokes = int(row.get("topology", {}).get("stroke_count", 0))
        if strokes < 1:
            raise ValueError("invalid external topology")
        output[(int(label), strokes)].append(index)
    for key, rows in output.items():
        rows.sort(key=lambda index: hashlib.sha256(f"v12:parent:{key}:{metadata[index]['synthetic_id']}".encode()).hexdigest())
    return output


def _select_parent(
    pool: dict[tuple[int, int], list[int]], metadata: list[dict], label: int, strokes: int,
    key: str, forbidden: set[str], used_ids: set[str],
) -> int | None:
    rows = sorted(pool.get((label, strokes), []), key=lambda index: hashlib.sha256(
        f"{WRITER_SEED_ROOT}:{key}:{metadata[index]['synthetic_id']}".encode()
    ).hexdigest())
    for index in rows:
        synthetic_id = str(metadata[index]["synthetic_id"])
        fingerprints = _parent_fingerprints(metadata[index])
        if synthetic_id not in used_ids and not (fingerprints & forbidden):
            return index
    return None


def _model(checkpoint_path: Path) -> tuple[InkClassifierV1, list[str]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    tokens = [str(value) for value in checkpoint["math_labels"]]
    model = InkClassifierV1(len(tokens), None)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, tokens


def _predict(model: InkClassifierV1, values: np.ndarray, batch_size: int = 512) -> np.ndarray:
    outputs = []
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            outputs.append(model(torch.from_numpy(values[start:start + batch_size]), "math").cpu().numpy())
    return np.concatenate(outputs).astype(np.float32, copy=False)


def _style_and_physics(rows: list[np.ndarray], writer: int, attempt_offset: int = 0) -> tuple[np.ndarray, list[dict]]:
    latent = _latents()[writer]
    styled_rows = []; audits = []
    for row in rows:
        styled, audit = apply_writer_style(row, latent.v4())
        styled_rows.append(styled); audits.append(audit)
    values = np.stack(styled_rows).astype(np.float32, copy=False)
    physical = simulate_pen_physics_v3(
        torch.from_numpy(values), _writer_physics(latent.v4()),
        torch.Generator().manual_seed(WRITER_SEED_ROOT + writer * WRITER_SEED_MULTIPLIER + attempt_offset),
        row_profiles=[GENERIC_CLASS_PROFILE] * len(values), return_diagnostics=False,
    ).numpy().astype(np.float32, copy=False)
    physical[:, :, 2:] = values[:, :, 2:]
    return physical, audits


def _valid_tensor(value: np.ndarray, source: np.ndarray) -> tuple[bool, float]:
    rms = float(np.sqrt(np.mean(np.square(value[:, :2] - source[:, :2]))))
    expected_dt = np.full(128, np.float32(1.0 / 127.0), dtype=np.float32); expected_dt[0] = 0.0
    uniform_time = np.array_equal(value[:, 2], expected_dt)
    observed = bool(np.all(value[:, 4] == 1.0))
    stroke_start = bool(value[0, 3] == 1.0 and np.all((value[:, 3] == 0.0) | (value[:, 3] == 1.0)))
    valid = (
        value.shape == (128, 5) and np.isfinite(value).all()
        and float(value[:, :2].min()) >= 0.0 and float(value[:, :2].max()) <= 1.0
        and np.array_equal(value[:, 2:], source[:, 2:])
        and np.array_equal(value[:, 3], source[:, 3])
        and np.array_equal(value[:, 4], source[:, 4])
        and uniform_time and observed and stroke_start
        and _topology(value) == _topology(source)
        and rms > MIN_REAL_RMS
    )
    return bool(valid), rms


def _catalog_entries() -> list[dict]:
    payload = _load_json_gzip(DRY_CATALOG)
    entries = payload.get("entries", [])
    if len(entries) != 68958:
        raise ValueError("R3 catalog entry count drift")
    pair_writers: dict[tuple[int, int], list[int]] = defaultdict(list)
    for entry in entries:
        pair = entry["pair"]
        key = (int(pair["baseline_label_index"]), int(pair["candidate_label_index"]))
        pair_writers[key].extend(int(value) for value in entry["assigned_writer_ids"])
    if len(pair_writers) != 9867:
        raise ValueError("R3 directed pair count drift")
    for pair, writers in pair_writers.items():
        if sorted(writers) != sorted(_writer_order(pair)) or not (len(writers) == len(set(writers)) == TARGET_ACTIONS_PER_PAIR):
            raise ValueError(f"pair writer contract failed: {pair}")
    return entries


def _calibration_specs(entries: list[dict]) -> set[tuple[int, int]]:
    return {
        (int(writer), int(entry["pair"]["candidate_label_index"]))
        for entry in entries for writer in entry["assigned_writer_ids"]
    }


def _build_calibration(
    specs: set[tuple[int, int]], features: np.ndarray, labels: np.ndarray, metadata: list[dict],
    pool: dict[tuple[int, int], list[int]], model: InkClassifierV1,
    homographs: tuple[frozenset[int], ...], existing_hashes: set[str],
) -> tuple[list[np.ndarray], list[dict], dict[tuple[int, int], set[str]], list[dict]]:
    rows = []; rows_meta = []; forbidden_by_writer: dict[tuple[int, int], set[str]] = {}
    unavailable = []
    for writer, label in sorted(specs):
        candidates = []
        for (pool_label, strokes), indices in sorted(pool.items()):
            if pool_label != label:
                continue
            for index in indices:
                candidates.append((strokes, index))
        candidates.sort(key=lambda item: hashlib.sha256(
            f"v12:cal:{writer}:{label}:{metadata[item[1]]['synthetic_id']}".encode()
        ).hexdigest())
        accepted = []
        used_fingerprints: set[str] = set(); used_ids: set[str] = set(); local_hashes: set[str] = set()
        for strokes, index in candidates:
            fingerprints = _parent_fingerprints(metadata[index])
            if fingerprints & used_fingerprints:
                continue
            source = features[index]
            seed_offset = _seed_offset(f"cal:{writer}:{label}:{metadata[index]['synthetic_id']}:{len(accepted)}")
            generated, audits = _style_and_physics([source], writer, seed_offset)
            value = generated[0]
            logits = _predict(model, value[None])[0]
            top5 = np.argsort(logits)[-5:][::-1].astype(int).tolist()
            valid, rms = _valid_tensor(value, source)
            digest = tensor_hash(value)
            if (not valid or label not in top5 or digest in existing_hashes or digest in local_hashes
                    or not all(value in ADMITTED_LABELS for value in top5)
                    or _homograph_collision(top5, homographs)):
                continue
            local_hashes.add(digest); used_fingerprints |= fingerprints; used_ids.add(str(metadata[index]["synthetic_id"]))
            accepted.append((value, {
                "schema": SCHEMA, "episode_split": "calibration", "synthetic_writer_id": f"synthetic_writer_{writer:03d}",
                "global_writer_id": writer, "label_index": label, "candidate_label_index": label,
                "support_role": "candidate_calibration_anchor", "parent_synthetic_id": metadata[index]["synthetic_id"],
                "parent_fingerprints": sorted(fingerprints), "source_index": index, "source_topology": metadata[index]["topology"],
                "attempt": len(accepted), "physics_seed": WRITER_SEED_ROOT + writer * WRITER_SEED_MULTIPLIER + seed_offset,
                "physics_batch_position": 0,
                "writer_latent": asdict(_latents()[writer]), "style_audit": audits[0],
                "spatial_rms_from_truth_parent": rms, "tensor_sha256": digest, "frozen_top5": top5,
                "external_approved_parent": True, "project_rows": 0,
            }))
            if len(accepted) == CALIBRATION_SUPPORT:
                break
        if len(accepted) != CALIBRATION_SUPPORT:
            unavailable.append({"global_writer_id": writer, "candidate_label_index": label,
                                "admitted_support": len(accepted), "policy": "identity_only"})
            continue
        forbidden_by_writer[(writer, label)] = used_fingerprints
        existing_hashes.update(local_hashes)
        for value, row in accepted:
            rows.append(value); rows_meta.append(row)
    return rows, rows_meta, forbidden_by_writer, unavailable


def _build_queries(
    entries: list[dict], features: np.ndarray, labels: np.ndarray, metadata: list[dict],
    pool: dict[tuple[int, int], list[int]], model: InkClassifierV1,
    homographs: tuple[frozenset[int], ...],
    calibration_fingerprints: dict[tuple[int, int], set[str]], existing_hashes: set[str],
) -> tuple[list[np.ndarray], list[dict], dict]:
    rows = []; rows_meta = []; pair_admitted: dict[tuple[int, int], set[int]] = defaultdict(set)
    counters = Counter()
    all_calibration_fingerprints: dict[int, set[str]] = defaultdict(set)
    for (writer, _candidate), fingerprints in calibration_fingerprints.items():
        all_calibration_fingerprints[writer].update(fingerprints)
    for entry in entries:
        baseline = int(entry["pair"]["baseline_label_index"]); candidate = int(entry["pair"]["candidate_label_index"])
        expected = {key: int(value) for key, value in entry["risk_stratum"].items()}
        pair = (baseline, candidate)
        for writer_value in entry["assigned_writer_ids"]:
            writer = int(writer_value); orientation = orientation_for(pair, writer)
            if (writer, candidate) not in calibration_fingerprints:
                counters["missing_candidate_calibration_support"] += 1; continue
            truth = candidate if orientation == "candidate_truth_promotion" else baseline
            relative = baseline if truth == candidate else candidate
            target_strokes = expected["stroke_count_bucket"]
            available_strokes = sorted({
                strokes for label, strokes in pool
                if label == truth and (relative, strokes) in pool
                and min(int(strokes), 4) == target_strokes
            })
            if not available_strokes:
                counters["missing_shared_topology"] += 1; continue
            strokes = available_strokes[0]
            forbidden = set(all_calibration_fingerprints[writer])
            truth_index = _select_parent(pool, metadata, truth, strokes, f"query-truth:{pair}:{writer}", forbidden, set())
            if truth_index is None:
                counters["missing_truth_parent"] += 1; continue
            truth_fp = _parent_fingerprints(metadata[truth_index]); forbidden |= truth_fp
            relative_index = _select_parent(pool, metadata, relative, strokes, f"query-relative:{pair}:{writer}", forbidden, set())
            if relative_index is None:
                counters["missing_relative_parent"] += 1; continue
            relative_fp = _parent_fingerprints(metadata[relative_index])
            source = features[truth_index]; relative_source = features[relative_index]
            attempt_rows = []
            for strength in MORPH_STRENGTHS:
                morphed = morph_truth_toward_relative(source, relative_source, strength)
                attempt_rows.append(morphed)
            action_key = f"query:{baseline}:{candidate}:{writer}:{canonical_hash(expected)}"
            seed_offset = _seed_offset(action_key)
            generated, style_audits = _style_and_physics(attempt_rows, writer, seed_offset)
            logits = _predict(model, generated)
            accepted = None
            for attempt, (value, row_logits, strength, style_audit) in enumerate(zip(generated, logits, MORPH_STRENGTHS, style_audits, strict=True)):
                actual = _risk(row_logits, strokes, candidate)
                valid, rms = _valid_tensor(value, source); digest = tensor_hash(value)
                if not all(value in ADMITTED_LABELS for value in actual["top5"]):
                    counters["generated_top5_unadmitted"] += 1; continue
                if _homograph_collision(actual["top5"], homographs):
                    counters["generated_top5_homograph_collision"] += 1; continue
                if not valid or digest in existing_hashes or not _risk_matches(actual, expected, baseline, candidate, homographs):
                    continue
                accepted = (value, {
                    "schema": SCHEMA, "episode_split": "query", "synthetic_writer_id": f"synthetic_writer_{writer:03d}",
                    "global_writer_id": writer, "directed_pair": {"baseline_label_index": baseline, "candidate_label_index": candidate},
                    "risk_stratum": expected, "orientation": orientation, "truth_label_index": truth,
                    "relative_label_index": relative, "truth_parent_synthetic_id": metadata[truth_index]["synthetic_id"],
                    "relative_parent_synthetic_id": metadata[relative_index]["synthetic_id"],
                    "truth_parent_fingerprints": sorted(truth_fp), "relative_parent_fingerprints": sorted(relative_fp),
                    "calibration_parent_fingerprints": sorted(calibration_fingerprints[(writer, candidate)]),
                    "all_writer_calibration_parent_fingerprints_sha256": canonical_hash(sorted(all_calibration_fingerprints[writer])),
                    "truth_source_index": truth_index, "relative_source_index": relative_index,
                    "truth_topology": metadata[truth_index]["topology"], "relative_topology": metadata[relative_index]["topology"],
                    "attempt": attempt, "morph_strength": strength,
                    "physics_seed": WRITER_SEED_ROOT + writer * WRITER_SEED_MULTIPLIER + seed_offset,
                    "physics_batch_position": attempt,
                    "writer_latent": asdict(_latents()[writer]), "style_audit": style_audit,
                    "spatial_rms_from_truth_parent": rms, "tensor_sha256": digest,
                    "frozen_top5": actual["top5"], "frozen_top1": actual["top5"][0],
                    "candidate_action_inside_frozen_top5": True, "external_approved_parents": True,
                    "project_rows": 0,
                })
                break
            if accepted is None:
                counters["three_attempt_drop"] += 1; continue
            value, row = accepted; digest = row["tensor_sha256"]
            existing_hashes.add(digest)
            rows.append(value); rows_meta.append(row); pair_admitted[pair].add(writer); counters[orientation] += 1
    failed_pairs = sorted([
        {"baseline_label_index": pair[0], "candidate_label_index": pair[1], "unique_admitted_writers": len(writers), "policy": "identity_only"}
        for pair, writers in pair_admitted.items() if len(writers) < MIN_ADMITTED_WRITERS_PER_PAIR
    ], key=lambda row: (row["baseline_label_index"], row["candidate_label_index"]))
    all_pairs = {(int(e["pair"]["baseline_label_index"]), int(e["pair"]["candidate_label_index"])) for e in entries}
    for pair in sorted(all_pairs - set(pair_admitted)):
        failed_pairs.append({"baseline_label_index": pair[0], "candidate_label_index": pair[1], "unique_admitted_writers": 0, "policy": "identity_only"})
    failed_set = {(row["baseline_label_index"], row["candidate_label_index"]) for row in failed_pairs}
    keep = [index for index, row in enumerate(rows_meta) if (row["directed_pair"]["baseline_label_index"], row["directed_pair"]["candidate_label_index"]) not in failed_set]
    return [rows[i] for i in keep], [rows_meta[i] for i in keep], {"counters": dict(counters), "failed_pairs": failed_pairs}


def generate(output: Path, postprocessed: Path, independent_audit: Path) -> None:
    _validate_prepared()
    helper_hashes = _validate_helper_sources()
    generator_audit = _validate_generator_audit(independent_audit)
    if output.exists() or postprocessed.exists():
        raise FileExistsError("raw or postprocessed output already exists; generation is one-shot")
    output.mkdir(parents=True)
    source_snapshot = output / "generate_action_evidence_augmentation_v12.source.py"
    source_snapshot.write_bytes(Path(__file__).read_bytes())
    marker = {
        "status": "V12_ACTION_EVIDENCE_GENERATION_STARTED",
        "started_at": datetime.now(timezone.utc).isoformat(), "source_sha256": sha256(source_snapshot),
        "prepared_manifest_sha256": EXPECTED_PREPARED_MANIFEST_SHA256,
        "pregen_audit_sha256": EXPECTED_PREGEN_AUDIT_SHA256, "generator_static_audit": generator_audit,
        "helper_source_sha256": helper_hashes, "admitted_labels_sha256": ADMITTED_LABELS_SHA256,
        "external_bank_not_loaded_before_this_marker": True, "checkpoint_not_loaded_before_this_marker": True,
        "catalog_entries_not_loaded_before_this_marker": True, "retry_allowed": False,
    }
    (output / "GENERATION_STARTED.json").write_text(json.dumps(marker, indent=2), encoding="utf-8")

    # Do not move any load above the immutable marker.
    _validate_live_generation_inputs()
    entries = _catalog_entries()
    with np.load(EXTERNAL_BANK, allow_pickle=False) as payload:
        features = np.asarray(payload["features"], dtype=np.float32)
        labels = np.asarray(payload["labels"], dtype=np.int64)
    metadata = _load_metadata(EXTERNAL_METADATA)
    if len(features) != len(labels) or len(labels) != len(metadata):
        raise ValueError("external bank row alignment failed")
    model, tokens = _model(CHECKPOINT)
    homographs = _resolve_homographs(tokens)
    pool = _parent_pool(labels, metadata)
    seen_hashes: set[str] = set()
    cal_rows, cal_meta, cal_fingerprints, unavailable_calibration = _build_calibration(
        _calibration_specs(entries), features, labels, metadata, pool, model, homographs, seen_hashes,
    )
    query_rows, query_meta, query_audit = _build_queries(
        entries, features, labels, metadata, pool, model, homographs, cal_fingerprints, seen_hashes,
    )
    values = np.stack(cal_rows + query_rows).astype(np.float32, copy=False)
    row_meta = cal_meta + query_meta
    row_labels = np.asarray([row["label_index"] if row["episode_split"] == "calibration" else row["truth_label_index"] for row in row_meta], np.int64)
    writers = np.asarray([row["global_writer_id"] for row in row_meta], np.int16)
    splits = np.asarray([0 if row["episode_split"] == "calibration" else 1 for row in row_meta], np.int8)
    final_hashes = [tensor_hash(value) for value in values]
    if len(final_hashes) != len(set(final_hashes)):
        raise AssertionError("final admitted tensor duplicate")
    if final_hashes != [str(row["tensor_sha256"]) for row in row_meta]:
        raise AssertionError("final tensor/metadata hash mismatch")
    physics_streams = [(int(row["physics_seed"]), int(row["physics_batch_position"])) for row in row_meta]
    if len(physics_streams) != len(set(physics_streams)):
        raise AssertionError("admitted rows reused a physics RNG substream")
    expected_dt = np.full(128, np.float32(1.0 / 127.0), dtype=np.float32); expected_dt[0] = 0.0
    source_indices = [int(row["source_index"] if row["episode_split"] == "calibration" else row["truth_source_index"]) for row in row_meta]
    nonspatial_exact = all(np.array_equal(values[index, :, 2:], features[source_index, :, 2:]) for index, source_index in enumerate(source_indices))
    uniform_time = bool(np.all(values[:, :, 2] == expected_dt[None]))
    observed_exact = bool(np.all(values[:, :, 4] == 1.0))
    final_logits = _predict(model, values)
    candidate_violations = 0
    for index, row in enumerate(row_meta):
        top5 = np.argsort(final_logits[index])[-5:][::-1].astype(int).tolist()
        if top5 != row["frozen_top5"] or not all(value in ADMITTED_LABELS for value in top5) or _homograph_collision(top5, homographs):
            candidate_violations += 1; continue
        if row["episode_split"] == "calibration":
            candidate_violations += int(int(row["candidate_label_index"]) not in top5)
        else:
            pair = row["directed_pair"]
            actual = _risk(final_logits[index], _topology(values[index])[0], int(pair["candidate_label_index"]))
            candidate_violations += int(not _risk_matches(
                actual, row["risk_stratum"], int(pair["baseline_label_index"]),
                int(pair["candidate_label_index"]), homographs,
            ))
    retained_pair_writers: dict[tuple[int, int], set[int]] = defaultdict(set)
    for row in query_meta:
        pair = (int(row["directed_pair"]["baseline_label_index"]), int(row["directed_pair"]["candidate_label_index"]))
        retained_pair_writers[pair].add(int(row["global_writer_id"]))
    failed_pair_keys = {(int(row["baseline_label_index"]), int(row["candidate_label_index"])) for row in query_audit["failed_pairs"]}
    catalog_pair_keys = {(int(entry["pair"]["baseline_label_index"]), int(entry["pair"]["candidate_label_index"])) for entry in entries}
    pair_gate = all(len(writer_ids) >= MIN_ADMITTED_WRITERS_PER_PAIR for writer_ids in retained_pair_writers.values())
    pair_gate = pair_gate and not (set(retained_pair_writers) & failed_pair_keys)
    pair_gate = pair_gate and (set(retained_pair_writers) | failed_pair_keys) == catalog_pair_keys
    np.savez_compressed(output / "action_evidence_raw_bank.npz", features=values, labels=row_labels, writers=writers, split=splits)
    _write_jsonl_gzip(output / "action_evidence_raw_bank.metadata.jsonl.gz", row_meta)
    orientations = Counter(row.get("orientation", "calibration") for row in row_meta)
    report = {
        "schema": SCHEMA, "status": "V12_ACTION_EVIDENCE_RAW_GENERATED_POSTPROCESS_REQUIRED",
        "rows": len(values), "calibration_rows": len(cal_rows), "query_rows": len(query_rows),
        "writers": len(set(writers.tolist())), "writer_range": [128, 191], "classes": len(set(row_labels.tolist())),
        "orientations": dict(orientations), "query_generation": query_audit,
        "unavailable_calibration_writer_candidates": unavailable_calibration,
        "gates": {
            "finite_unit_box": bool(np.isfinite(values).all() and values[:, :, :2].min() >= 0 and values[:, :, :2].max() <= 1),
            "within_new_raw_bank_duplicates_zero": len(final_hashes) == len(set(final_hashes)),
            "within_new_raw_bank_identity_zero": all(row["spatial_rms_from_truth_parent"] > MIN_REAL_RMS for row in row_meta),
            "uniform_time_exact": uniform_time, "observed_exact": observed_exact,
            "nonspatial_parent_exact": nonspatial_exact, "external_approved_only": True,
            "pair_writer_min16_or_identity_only": pair_gate,
            "candidate_action_within_frozen_top5": candidate_violations == 0,
            "admitted_physics_rng_substreams_unique": len(physics_streams) == len(set(physics_streams)),
            "adapter_training_not_performed": True, "hwr_checkpoint_runtime_unchanged": True,
            "writers096_127_remained_closed": True, "legacy_real_crohme_mathwriting_remained_closed": True,
            "postprocessing_not_performed": True, "product_promotion_not_performed": True,
        },
        "boundaries": {"raw_attempt_upper_plan": 633416, "raw_count_not_forced": True, "postprocessed_output": str(postprocessed.resolve()),
                       "postprocessed_output_exists": postprocessed.exists(),
                       "cross_bank_000_127_duplicate_identity_audit": "pending_independent_postprocess_audit",
                       "adapter_trained": False, "hwr_changed": False,
                       "writers096_127_opened": False, "legacy_rows": 0, "real_rows": 0, "crohme_rows": 0, "mathwriting_rows": 0},
        "hashes": {"source_snapshot": sha256(source_snapshot), "started_marker": sha256(output / "GENERATION_STARTED.json"),
                   "bank": sha256(output / "action_evidence_raw_bank.npz"),
                   "metadata": sha256(output / "action_evidence_raw_bank.metadata.jsonl.gz"),
                   "checkpoint": EXPECTED_CHECKPOINT_SHA256, "external_bank": EXPECTED_EXTERNAL_BANK_SHA256,
                   "external_metadata": EXPECTED_EXTERNAL_METADATA_SHA256, "dry_catalog": EXPECTED_DRY_CATALOG_SHA256},
        "final_admission": {"candidate_violations": candidate_violations,
                            "retained_pairs": len(retained_pair_writers), "identity_only_pairs": len(failed_pair_keys),
                            "minimum_retained_pair_writers": min(map(len, retained_pair_writers.values()), default=0),
                            "seen_hashes_including_dropped_rows": len(seen_hashes), "final_hashes": len(final_hashes)},
        "helper_source_sha256": helper_hashes,
        "admitted_label_provenance": {"count": len(ADMITTED_LABELS), "sha256": ADMITTED_LABELS_SHA256,
                                      "rule": "consumed000..095 count>=4 and writers>=4; independently pinned before generation"},
        "tokens_sha256": canonical_hash(tokens), "writer_latents_sha256": canonical_hash([asdict(x) for x in _latents()[128:192]]),
    }
    (output / "generation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def dry_toy() -> dict:
    """Pure toy: no prepared artifact, external tensor, checkpoint, or catalog load."""

    truth = np.zeros((128, 5), np.float32); relative = np.zeros((128, 5), np.float32)
    truth[:, 0] = np.linspace(0.2, 0.8, 128); truth[:, 1] = 0.35
    relative[:, 0] = np.linspace(0.25, 0.75, 128); relative[:, 1] = 0.65
    for value in (truth, relative):
        value[0, 3] = 1.0; value[:, 4] = 1.0; value[1:, 2] = 1.0 / 127.0
    output = morph_truth_toward_relative(truth, relative, MORPH_STRENGTHS[1])
    pair = (11, 22); writers = _writer_order(pair)
    orientations = Counter(orientation_for(pair, writer) for writer in writers)
    result = {
        "status": "V12_GENERATOR_DRY_TOY_PASSED",
        "external_tensor_loaded": False, "checkpoint_loaded": False, "catalog_loaded": False,
        "pair_unique_writers": len(set(writers)), "pair_writer_max_actions": 1,
        "orientations": dict(orientations), "topology_preserved": _topology(output) == _topology(truth),
        "non_spatial_exact": bool(np.array_equal(output[:, 2:], truth[:, 2:])),
        "finite_unit_box": bool(np.isfinite(output).all() and output[:, :2].min() >= 0 and output[:, :2].max() <= 1),
        "spatial_rms": float(np.sqrt(np.mean(np.square(output[:, :2] - truth[:, :2])))),
    }
    if result["pair_unique_writers"] != 20 or result["orientations"] != {"candidate_truth_promotion": 10, "baseline_truth_veto": 10}:
        raise AssertionError("orientation contract failed")
    if not result["topology_preserved"] or not result["non_spatial_exact"] or not result["finite_unit_box"]:
        raise AssertionError("trajectory contract failed")
    return result


def dry_parent_coverage() -> dict:
    """Metadata/catalog-only estimate; external NPZ and checkpoint remain unopened."""

    if sha256(EXTERNAL_METADATA) != EXPECTED_EXTERNAL_METADATA_SHA256 or sha256(DRY_CATALOG) != EXPECTED_DRY_CATALOG_SHA256:
        raise ValueError("coverage inputs drift")
    metadata = _load_metadata(EXTERNAL_METADATA)
    labels = np.asarray([int(row["label_index"]) for row in metadata], np.int64)
    pool = _parent_pool(labels, metadata)
    entries = _catalog_entries()
    disjoint_counts = {}
    for key, indices in pool.items():
        used: set[str] = set(); count = 0
        for index in indices:
            fingerprints = _parent_fingerprints(metadata[index])
            if fingerprints & used:
                continue
            used |= fingerprints; count += 1
        disjoint_counts[key] = count
    pair_shared = {}; unit_shared = 0; action_shared = 0
    anchor_specs = _calibration_specs(entries); anchor_supported = 0
    for writer, label in anchor_specs:
        anchor_supported += int(any(key[0] == label and count >= CALIBRATION_SUPPORT for key, count in disjoint_counts.items()))
    for entry in entries:
        baseline = int(entry["pair"]["baseline_label_index"]); candidate = int(entry["pair"]["candidate_label_index"])
        bucket = int(entry["risk_stratum"]["stroke_count_bucket"])
        shared = sorted(strokes for label, strokes in pool if label == baseline and (candidate, strokes) in pool and min(strokes, 4) == bucket)
        if shared:
            unit_shared += 1; action_shared += len(entry["assigned_writer_ids"]); pair_shared[(baseline, candidate)] = True
        else:
            pair_shared.setdefault((baseline, candidate), False)
    result = {
        "status": "V12_GENERATOR_PARENT_COVERAGE_DRY_ONLY",
        "external_tensor_loaded": False, "checkpoint_loaded": False, "writers096_127_opened": False,
        "external_metadata_rows": len(metadata), "catalog_units": len(entries),
        "directed_pairs": len(pair_shared), "pairs_with_shared_exact_stroke_parent": sum(pair_shared.values()),
        "pair_risk_units_with_shared_exact_stroke_parent": unit_shared,
        "planned_actions_with_shared_exact_stroke_parent": action_shared,
        "writer_candidate_anchor_specs": len(anchor_specs), "anchor_specs_with_metadata_support2": anchor_supported,
        "orientation_plan": {"candidate_truth_promotion_per_pair": 10, "baseline_truth_veto_per_pair": 10},
        "limits": "metadata-only upper coverage; frozen-HWR/risk/topology/duplicate/identity admission can only reduce rows",
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-toy", action="store_true")
    mode.add_argument("--dry-parent-coverage", action="store_true")
    mode.add_argument("--generate", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_RAW_OUTPUT)
    parser.add_argument("--postprocessed-output", type=Path, default=DEFAULT_POSTPROCESSED_OUTPUT)
    parser.add_argument("--independent-audit", type=Path)
    args = parser.parse_args()
    if args.dry_toy:
        print(json.dumps(dry_toy(), indent=2)); return 0
    if args.dry_parent_coverage:
        print(json.dumps(dry_parent_coverage(), indent=2)); return 0
    if args.independent_audit is None:
        parser.error("--generate requires --independent-audit")
    for path in (args.output.resolve(), args.postprocessed_output.resolve(), args.independent_audit.resolve()):
        if path.drive.upper() != "D:":
            parser.error("generation and audit paths must remain on D:")
    generate(args.output, args.postprocessed_output, args.independent_audit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
