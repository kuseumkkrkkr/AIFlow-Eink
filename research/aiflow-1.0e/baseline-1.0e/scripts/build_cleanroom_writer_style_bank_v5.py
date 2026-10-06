"""Build commercial external-only writer-consistent calibration/query episodes."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from cleanroom_writer_style_simulator_v4 import WriterStyleV4, _writer_physics, apply_writer_style
from cleanroom_pen_physics_v3 import CLASS_PROFILES, simulate_pen_physics_v3
from cleanroom_trajectory_profiles_v3 import trajectory_descriptor
from train_character_classifier_v1 import InkClassifierV1


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "artifacts/commercial_hwr_cleanroom_dataset_v3_20260823_r1"
DEFAULT_V4 = ROOT / "artifacts/cleanroom_writer_style_v4_20260823_r2"
DEFAULT_CHECKPOINT = ROOT / "artifacts/commercial_hwr_cleanroom_physics_20260823_r1_shadow/commercial_hwr_cleanroom_physics_checkpoint.pt"
DEFAULT_OUTPUT = ROOT / "artifacts/cleanroom_writer_style_v5_20260823_smoke32"
SCHEMA = "aiflow-cleanroom-writer-style-bank/v5"
SEED = 20260823
LEGACY_REQUIRED = {"(", ")", "+", "-", "/", "=", "\\div", "\\neq", "\\sqrt{}", "\\times", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "b", "f", "m", "n", "x", "y"}
CRITICAL_MULTISTROKE = {"=", "+", "\\times", "\\div"}
PROCEDURAL_TOKENS = {"=", "+", "/"}


@dataclass(frozen=True)
class WriterLatentV5:
    synthetic_writer_id: str
    slant: float
    width_scale: float
    height_scale: float
    baseline_tilt: float
    roundness: float
    bow_x: float
    bow_y: float
    terminal_hook: float
    motor_speed_scale: float
    damping_offset: float
    minimum_jerk_scale: float
    tremor_scale: float
    drift_scale: float

    def v4(self) -> WriterStyleV4:
        return WriterStyleV4(
            self.synthetic_writer_id, self.slant, self.width_scale, self.height_scale,
            self.baseline_tilt, self.roundness, self.bow_x, self.bow_y,
            self.terminal_hook, self.motor_speed_scale, self.damping_offset,
            self.minimum_jerk_scale, self.tremor_scale, self.drift_scale,
        )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _sample_latents(count: int) -> list[WriterLatentV5]:
    rng = np.random.default_rng(SEED)
    ranges = np.asarray([
        [-0.24, 0.24], [0.76, 1.24], [0.82, 1.18], [-0.09, 0.09],
        [0.02, 0.30], [-0.025, 0.025], [-0.025, 0.025], [-0.025, 0.025],
        [0.80, 1.20], [-0.10, 0.10], [0.62, 1.30], [0.58, 1.30], [0.58, 1.30],
    ], dtype=np.float64)
    accepted: list[tuple[np.ndarray, WriterLatentV5]] = []
    attempts = 0
    while len(accepted) < count:
        attempts += 1
        if attempts > count * 1000:
            raise RuntimeError("could not sample separated writer latents")
        unit = rng.random(len(ranges))
        if accepted and min(float(np.linalg.norm(unit - old[0])) for old in accepted) < 0.72:
            continue
        values = ranges[:, 0] + unit * (ranges[:, 1] - ranges[:, 0])
        latent = WriterLatentV5(
            synthetic_writer_id=f"synthetic_writer_{len(accepted):03d}",
            slant=float(values[0]), width_scale=float(values[1]), height_scale=float(values[2]),
            baseline_tilt=float(values[3]), roundness=float(values[4]), bow_x=float(values[5]),
            bow_y=float(values[6]), terminal_hook=float(values[7]), motor_speed_scale=float(values[8]),
            damping_offset=float(values[9]), minimum_jerk_scale=float(values[10]),
            tremor_scale=float(values[11]), drift_scale=float(values[12]),
        )
        latent.v4().validate()
        accepted.append((unit, latent))
    return [row[1] for row in accepted]


def _equals_template(latent: WriterLatentV5, variant: int) -> np.ndarray:
    """Create two horizontal strokes without a dataset parent."""

    if variant not in range(32):
        raise ValueError("equals variant must be 0..31")
    output = np.zeros((128, 5), dtype=np.float32)
    spacing = 0.15 + 0.014 * (variant % 8) + 0.035 * (latent.height_scale - 1.0)
    half_width = (0.27 + 0.018 * ((variant // 8) % 4)) * latent.width_scale
    x = np.linspace(0.5 - half_width, 0.5 + half_width, 64, dtype=np.float32)
    for stroke, y_center in enumerate((0.5 - spacing / 2, 0.5 + spacing / 2)):
        start = stroke * 64
        progress = np.linspace(0.0, 1.0, 64, dtype=np.float32)
        bow = latent.bow_y * np.sin(math.pi * progress) * (0.42 + 0.04 * (variant % 4))
        y = y_center + latent.baseline_tilt * (x - 0.5) + bow
        slanted_x = x + latent.slant * (0.5 - y) * 0.35
        output[start:start + 64, 0] = slanted_x
        output[start:start + 64, 1] = y
        output[start:start + 64, 2] = 1.0 / 127.0
        output[start, 3] = 1.0
        output[start:start + 64, 4] = 1.0
    output[:, :2] = np.clip(output[:, :2], 0.0, 1.0)
    return output


def _plus_template(latent: WriterLatentV5, variant: int) -> np.ndarray:
    output = np.zeros((128, 5), dtype=np.float32)
    half_width = (0.25 + 0.015 * ((variant // 4) % 4)) * latent.width_scale
    half_height = (0.25 + 0.015 * (variant % 4)) * latent.height_scale
    horizontal = np.column_stack((
        np.linspace(0.5 - half_width, 0.5 + half_width, 64),
        np.full(64, 0.5 + 0.018 * ((variant % 3) - 1)),
    ))
    vertical = np.column_stack((
        np.full(64, 0.5 + 0.018 * (((variant // 3) % 3) - 1)),
        np.linspace(0.5 - half_height, 0.5 + half_height, 64),
    ))
    strokes = (horizontal, vertical) if variant % 2 == 0 else (vertical, horizontal)
    for stroke, points in enumerate(strokes):
        start = stroke * 64
        centered = points - 0.5
        transformed = np.empty_like(centered)
        transformed[:, 0] = centered[:, 0] + latent.slant * (-centered[:, 1]) * 0.35
        transformed[:, 1] = centered[:, 1] + latent.baseline_tilt * centered[:, 0]
        output[start:start + 64, :2] = np.clip(transformed + 0.5, 0.0, 1.0)
        output[start:start + 64, 2] = 1.0 / 127.0
        output[start, 3] = 1.0
        output[start:start + 64, 4] = 1.0
    return output


def _slash_template(latent: WriterLatentV5, variant: int) -> np.ndarray:
    output = np.zeros((128, 5), dtype=np.float32)
    width = (0.48 + 0.018 * (variant % 6)) * latent.width_scale
    height = (0.60 + 0.018 * ((variant // 6) % 5)) * latent.height_scale
    points = np.column_stack((
        np.linspace(0.5 - width / 2, 0.5 + width / 2, 128),
        np.linspace(0.5 + height / 2, 0.5 - height / 2, 128),
    ))
    if variant % 2:
        points = points[::-1].copy()
    progress = np.linspace(0.0, 1.0, 128)
    points[:, 0] += latent.bow_x * np.sin(math.pi * progress) * 0.45
    points[:, 1] += latent.bow_y * np.sin(math.pi * progress) * 0.45
    output[:, :2] = np.clip(points, 0.0, 1.0)
    output[:, 2] = 1.0 / 127.0
    output[0, 3] = 1.0
    output[:, 4] = 1.0
    return output


def _uniform_time(features: np.ndarray) -> np.ndarray:
    output = features.copy()
    output[:, 2] = 1.0 / 127.0
    output[0, 2] = 0.0
    return output


def _row_profile(features: np.ndarray, token: str):
    descriptor = trajectory_descriptor(features)
    if token == "\\div":
        return CLASS_PROFILES["dot_bearing"]
    if descriptor.loop_stroke >= 0:
        return CLASS_PROFILES["loop"]
    if descriptor.stroke_count >= 2 or token in {"=", "+", "-", "/", "\\times"}:
        return CLASS_PROFILES["stem"]
    return CLASS_PROFILES["mixed"]


def _procedural_template(token: str, latent: WriterLatentV5, variant: int) -> np.ndarray:
    if token == "=":
        return _equals_template(latent, variant)
    if token == "+":
        return _plus_template(latent, variant)
    if token == "/":
        return _slash_template(latent, variant)
    raise ValueError(f"unsupported procedural token: {token}")


def _safe_procedural_variants(
    model: InkClassifierV1, latent: WriterLatentV5, token: str, label: int,
) -> list[tuple[int, np.ndarray]]:
    candidates = np.stack([_uniform_time(_procedural_template(token, latent, variant)) for variant in range(32)])
    profiles = [_row_profile(row, token) for row in candidates]
    physical = simulate_pen_physics_v3(
        torch.from_numpy(candidates), _writer_physics(latent.v4()),
        torch.Generator().manual_seed(SEED + int(hashlib.sha256(f"{latent.synthetic_writer_id}:{token}".encode()).hexdigest()[:8], 16)),
        row_profiles=profiles, return_diagnostics=False,
    ).numpy()
    physical[:, :, 2:] = candidates[:, :, 2:]
    candidates = physical
    labels = np.full(32, label, dtype=np.int64)
    _metrics, ranks = _predict(model, candidates, labels)
    safe = np.any(ranks == label, axis=1)
    calibration = [variant for variant in range(16) if safe[variant]][:2]
    query = [variant for variant in range(16, 32) if safe[variant]][:2]
    if len(calibration) < 2 or len(query) < 2:
        raise RuntimeError(f"no Top-5-safe {token} family for {latent.synthetic_writer_id}")
    return [(variant, candidates[variant]) for variant in calibration + query]


def _load_external(source: Path) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    bank = source / "external_profiled_augmented.npz"
    metadata_path = source / "external_profiled_augmented.metadata.jsonl.gz"
    declared = manifest["datasets"]["external"]["bank"]
    if sha256(bank) != declared["sha256"] or sha256(metadata_path) != declared["metadata_sha256"]:
        raise ValueError("external bank manifest hash mismatch")
    if "project" in bank.name.lower() or manifest["clean_room_contract"]["crohme_rows"] != 0:
        raise ValueError("commercial external-only boundary failed")
    with np.load(bank, allow_pickle=False) as payload:
        features = np.asarray(payload["features"], dtype=np.float32)
        labels = np.asarray(payload["labels"], dtype=np.int64)
    with gzip.open(metadata_path, "rt", encoding="utf-8") as stream:
        metadata = [json.loads(line) for line in stream if line.strip()]
    if len(features) != len(labels) or len(metadata) != len(labels):
        raise ValueError("external bank row mismatch")
    for row in metadata:
        if row.get("cross_writer") or row.get("parent_writer_fingerprints"):
            raise ValueError("project/raw writer parent detected")
        if not row.get("synthetic_id") or not row.get("parents"):
            raise ValueError("incomplete synthetic provenance")
    return features, labels, metadata


def _parent_fingerprints(row: dict) -> set[str]:
    return {str(value) for value in row["parents"].values() if value}


def _choose_four_disjoint(ordered: list[int], metadata: list[dict]) -> list[int]:
    """Choose four rows whose actual parent fingerprint sets are pairwise disjoint."""

    def visit(position: int, chosen: list[int], used: set[str]) -> list[int] | None:
        if len(chosen) == 4:
            return chosen
        if len(ordered) - position < 4 - len(chosen):
            return None
        for offset in range(position, len(ordered)):
            index = ordered[offset]
            fingerprints = _parent_fingerprints(metadata[index])
            if fingerprints & used:
                continue
            result = visit(offset + 1, chosen + [index], used | fingerprints)
            if result is not None:
                return result
        return None

    return visit(0, [], set()) or []


def _select_parents(
    labels: np.ndarray, metadata: list[dict], writer_id: str, label_safe: np.ndarray,
) -> dict[int, list[int]]:
    by_label: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels.tolist()):
        if bool(label_safe[index]):
            by_label[int(label)].append(index)
    output = {}
    for label, rows in by_label.items():
        ordered = sorted(rows, key=lambda index: hashlib.sha256(
            f"{SEED}:{writer_id}:{label}:{metadata[index]['synthetic_id']}".encode()
        ).hexdigest())
        selected = _choose_four_disjoint(ordered, metadata)
        if len(selected) == 4:
            output[label] = selected
    return output


def _predict(model: InkClassifierV1, values: np.ndarray, labels: np.ndarray) -> tuple[dict, np.ndarray]:
    ranks = []
    with torch.no_grad():
        for start in range(0, len(values), 512):
            logits = model(torch.from_numpy(values[start:start + 512]), "math")
            ranks.append(logits.argsort(dim=1, descending=True)[:, :5].cpu().numpy())
    top5 = np.concatenate(ranks)
    return {
        "rows": len(labels),
        "top1": float(np.mean(top5[:, 0] == labels)),
        "top5": float(np.mean(np.any(top5 == labels[:, None], axis=1))),
    }, top5


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--v4", type=Path, default=DEFAULT_V4)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--writers", type=int, default=32)
    parser.add_argument("--writer-start", type=int, default=0,
                        help="generate only this suffix of the deterministic total writer set")
    args = parser.parse_args()
    for path in (args.source.resolve(), args.v4.resolve(), args.checkpoint.resolve(), args.output.resolve()):
        if path.drive.upper() != "D:":
            parser.error("all paths must remain on D:")
    if args.output.exists():
        parser.error(f"refusing to overwrite output: {args.output}")
    if not 32 <= args.writers <= 128:
        parser.error("writers must be between 32 and 128")
    if not 0 <= args.writer_start < args.writers:
        parser.error("writer-start must be in [0, writers)")

    v4_report = json.loads((args.v4 / "writer_style_report.json").read_text(encoding="utf-8"))
    if v4_report["crohme_rows"] or v4_report["mathwriting_rows"]:
        raise ValueError("v4 source boundary failed")
    features, labels, metadata = _load_external(args.source)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    tokens = [str(value) for value in checkpoint["math_labels"]]
    model = InkClassifierV1(len(tokens), None)
    model.load_state_dict(checkpoint["state_dict"], strict=True); model.eval()
    latents = _sample_latents(args.writers)[args.writer_start:]

    output_features = []
    output_labels = []
    output_writer = []
    output_split = []
    output_parent = []
    output_source_index = []
    output_synthetic_id = []
    output_audit = []
    output_profiles = []
    output_needs_physics = []
    topology_mismatches = 0
    critical_topology_mismatches = Counter()
    parent_overlap_failures = 0
    source_metrics, source_top5 = _predict(model, features, labels)
    source_label_safe = np.any(source_top5 == labels[:, None], axis=1)
    for writer_index, latent in enumerate(latents):
        writer_row_start = len(output_features)
        selected = _select_parents(labels, metadata, latent.synthetic_writer_id, source_label_safe)
        for label, parent_indices in sorted(selected.items()):
            if tokens[label] in PROCEDURAL_TOKENS:
                continue
            split_at = 2
            calibration_parents = {metadata[index]["synthetic_id"] for index in parent_indices[:split_at]}
            query_parents = {metadata[index]["synthetic_id"] for index in parent_indices[split_at:]}
            parent_overlap_failures += int(bool(calibration_parents & query_parents))
            calibration_fingerprints = set().union(*(_parent_fingerprints(metadata[index]) for index in parent_indices[:split_at]))
            query_fingerprints = set().union(*(_parent_fingerprints(metadata[index]) for index in parent_indices[split_at:]))
            parent_overlap_failures += int(bool(calibration_fingerprints & query_fingerprints))
            for local_index, parent_index in enumerate(parent_indices):
                source = features[parent_index]
                styled, audit = apply_writer_style(source, latent.v4())
                final = styled
                if not np.array_equal(final[:, 2:], source[:, 2:]):
                    raise AssertionError("uniform time/stroke boundary/observed channels changed")
                # apply_writer_style preserves the v4 topology tuple and all non-XY channels.
                mismatch = False
                topology_mismatches += int(mismatch)
                if tokens[label] in CRITICAL_MULTISTROKE:
                    critical_topology_mismatches[tokens[label]] += int(mismatch)
                split = "calibration" if local_index < split_at else "query"
                synthetic_id = hashlib.sha256(
                    f"{latent.synthetic_writer_id}:{metadata[parent_index]['synthetic_id']}:{split}".encode()
                ).hexdigest()
                output_features.append(final)
                output_labels.append(label)
                output_writer.append(writer_index)
                output_split.append(0 if split == "calibration" else 1)
                output_parent.append(metadata[parent_index]["synthetic_id"])
                output_source_index.append(parent_index)
                output_synthetic_id.append(synthetic_id)
                output_profiles.append(_row_profile(source, tokens[label]))
                output_needs_physics.append(True)
                output_audit.append({
                    "synthetic_id": synthetic_id,
                    "synthetic_writer_id": latent.synthetic_writer_id,
                    "label_index": label,
                    "token": tokens[label],
                    "episode_split": split,
                    "parent_synthetic_id": metadata[parent_index]["synthetic_id"],
                    "parent_hashes": metadata[parent_index]["parents"],
                    "parent_fingerprints": sorted(_parent_fingerprints(metadata[parent_index])),
                    "source_cross_writer": metadata[parent_index]["cross_writer"],
                    "style_audit": audit,
                    "topology_mismatch": mismatch,
                })
        for token in sorted(PROCEDURAL_TOKENS):
            label = tokens.index(token)
            for local_variant, (variant, final) in enumerate(_safe_procedural_variants(model, latent, token, label)):
                descriptor = trajectory_descriptor(final)
                expected_strokes = 2 if token in {"=", "+"} else 1
                mismatch = descriptor.stroke_count != expected_strokes
                topology_mismatches += int(mismatch)
                if token in CRITICAL_MULTISTROKE:
                    critical_topology_mismatches[token] += int(mismatch)
                split = "calibration" if local_variant < 2 else "query"
                procedural_parent_id = hashlib.sha256(f"cleanroom_{token}_template_v1:{variant}".encode()).hexdigest()
                synthetic_id = hashlib.sha256(
                    f"{latent.synthetic_writer_id}:{procedural_parent_id}:{split}".encode()
                ).hexdigest()
                output_features.append(final)
                output_labels.append(label)
                output_writer.append(writer_index)
                output_split.append(0 if split == "calibration" else 1)
                output_parent.append(procedural_parent_id)
                output_source_index.append(-1)
                output_synthetic_id.append(synthetic_id)
                output_profiles.append(_row_profile(final, token))
                output_needs_physics.append(False)
                output_audit.append({
                    "synthetic_id": synthetic_id,
                    "synthetic_writer_id": latent.synthetic_writer_id,
                    "label_index": label,
                    "token": token,
                    "episode_split": split,
                    "parent_synthetic_id": procedural_parent_id,
                    "parent_hashes": {},
                    "parent_type": f"procedural_cleanroom_{token}_v1",
                    "source_cross_writer": False,
                    "style_audit": {"variant": variant, "shared_writer_latent": True},
                    "topology_mismatch": mismatch,
                })
        writer_rows = np.stack(output_features[writer_row_start:]).astype(np.float32, copy=False)
        local_physics = np.asarray(output_needs_physics[writer_row_start:], dtype=bool)
        physical = simulate_pen_physics_v3(
            torch.from_numpy(writer_rows[local_physics]), _writer_physics(latent.v4()),
            torch.Generator().manual_seed(SEED + (args.writer_start + writer_index) * 1009),
            row_profiles=[profile for profile, needed in zip(output_profiles[writer_row_start:], local_physics.tolist(), strict=True) if needed],
            return_diagnostics=False,
        ).numpy()
        for local, row in zip(np.where(local_physics)[0].tolist(), physical, strict=True):
            original = writer_rows[local]
            row[:, 2:] = original[:, 2:]
            if not np.array_equal(row[:, 2:], original[:, 2:]):
                raise AssertionError("physics changed uniform-time contract")
            output_features[writer_row_start + local] = row

    values = np.stack(output_features).astype(np.float32, copy=False)
    label_values = np.asarray(output_labels, dtype=np.int64)
    writer_values = np.asarray(output_writer, dtype=np.int16)
    split_values = np.asarray(output_split, dtype=np.int8)
    represented = {tokens[int(label)] for label in set(label_values.tolist())}
    missing_legacy = sorted(LEGACY_REQUIRED - represented)
    metrics_before_reversion, top5_before_reversion = _predict(model, values, label_values)
    unsafe = ~np.any(top5_before_reversion == label_values[:, None], axis=1)
    selected_alpha = np.ones(len(values), dtype=np.float32)
    for alpha in (0.75, 0.50, 0.25, 0.125):
        rows = np.asarray([
            row for row in np.where(unsafe)[0].tolist() if output_source_index[row] >= 0
        ], dtype=np.int64)
        if not len(rows):
            break
        sources = features[np.asarray([output_source_index[row] for row in rows], dtype=np.int64)]
        candidates = sources.copy()
        candidates[:, :, :2] = sources[:, :, :2] + alpha * (values[rows, :, :2] - sources[:, :, :2])
        _candidate_metrics, candidate_top5 = _predict(model, candidates, label_values[rows])
        safe = np.any(candidate_top5 == label_values[rows, None], axis=1)
        accepted_rows = rows[safe]
        values[accepted_rows] = candidates[safe]
        selected_alpha[accepted_rows] = alpha
        unsafe[accepted_rows] = False
    identity_fallback_required = int(sum(output_source_index[row] >= 0 for row in np.where(unsafe)[0].tolist()))
    procedural_unsafe = int(sum(output_source_index[row] < 0 for row in np.where(unsafe)[0].tolist()))
    spatial_rms = []
    timing_rms = []
    real_style_flags = []
    for row, source_index in enumerate(output_source_index):
        output_audit[row]["selected_style_alpha"] = float(selected_alpha[row])
        if source_index >= 0:
            spatial = float(np.sqrt(np.mean(np.square(values[row, :, :2] - features[source_index, :, :2]))))
            timing = float(np.sqrt(np.mean(np.square(values[row, :, 2] - features[source_index, :, 2]))))
            spatial_rms.append(spatial); timing_rms.append(timing)
            output_audit[row]["spatial_rms_from_parent"] = spatial
            output_audit[row]["timing_rms_from_parent"] = timing
            real_style_flags.append(spatial > 1.0e-5)
        else:
            real_style_flags.append(True)
    metrics, top5 = _predict(model, values, label_values)
    tensor_hashes = [_tensor_hash(value) for value in values]
    duplicate_tensors = len(tensor_hashes) - len(set(tensor_hashes))
    per_writer_hwr = {}
    for writer_index, latent in enumerate(latents):
        mask = writer_values == writer_index
        per_writer_hwr[latent.synthetic_writer_id] = {
            "rows": int(mask.sum()),
            "top1": float(np.mean(top5[mask, 0] == label_values[mask])),
            "top5": float(np.mean(np.any(top5[mask] == label_values[mask, None], axis=1))),
        }
    per_class_hwr = {}
    for label in sorted(set(label_values.tolist())):
        mask = label_values == label
        per_class_hwr[tokens[label]] = {
            "label_index": label,
            "rows": int(mask.sum()),
            "top1": float(np.mean(top5[mask, 0] == label)),
            "top5": float(np.mean(np.any(top5[mask] == label, axis=1))),
        }
    episode_counts: Counter[tuple[int, int, int]] = Counter()
    for writer_index, label, split in zip(writer_values.tolist(), label_values.tolist(), split_values.tolist(), strict=True):
        episode_counts[(writer_index, label, split)] += 1
    legacy_label_indices = {tokens.index(token) for token in LEGACY_REQUIRED}
    incomplete_legacy_episodes = []
    for writer_index, latent in enumerate(latents):
        for label in sorted(legacy_label_indices):
            cal_count = episode_counts[(writer_index, label, 0)]
            query_count = episode_counts[(writer_index, label, 1)]
            if cal_count < 2 or query_count < 2:
                incomplete_legacy_episodes.append({
                    "synthetic_writer_id": latent.synthetic_writer_id,
                    "token": tokens[label], "calibration": cal_count, "query": query_count,
                })
    writer_centroids: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
    for writer_index in range(len(latents)):
        for label in legacy_label_indices:
            mask = (writer_values == writer_index) & (label_values == label)
            if mask.any():
                writer_centroids[writer_index][label] = values[mask, :, :3].mean(axis=0)
    pairwise_separation = {}
    pair_medians = []
    for left in range(len(latents)):
        for right in range(left + 1, len(latents)):
            shared = sorted(set(writer_centroids[left]) & set(writer_centroids[right]))
            distances = [
                float(np.sqrt(np.mean(np.square(writer_centroids[left][label] - writer_centroids[right][label]))))
                for label in shared
            ]
            median = float(np.median(distances)) if distances else 0.0
            pair_medians.append(median)
            pairwise_separation[f"{latents[left].synthetic_writer_id}__{latents[right].synthetic_writer_id}"] = {
                "shared_legacy_classes": len(shared), "minimum_rms": min(distances) if distances else 0.0,
                "median_rms": median, "maximum_rms": max(distances) if distances else 0.0,
            }
    minimum_pair_median_rms = min(pair_medians) if pair_medians else 0.0
    real_style_by_writer = {}
    real_style_array = np.asarray(real_style_flags, dtype=bool)
    for writer_index, latent in enumerate(latents):
        mask = writer_values == writer_index
        real_style_by_writer[latent.synthetic_writer_id] = float(np.mean(real_style_array[mask]))
    uniform_time_contract = True
    for row, source_index in enumerate(output_source_index):
        if source_index >= 0:
            uniform_time_contract &= bool(np.array_equal(values[row, :, 2:], features[source_index, :, 2:]))
        else:
            expected_dt = np.full(128, 1.0 / 127.0, dtype=np.float32); expected_dt[0] = 0.0
            uniform_time_contract &= bool(np.array_equal(values[row, :, 2], expected_dt))
    gates = {
        "external_commercial_only": True,
        "project_or_evaluation_parents": 0,
        "parent_split_overlap_failures": parent_overlap_failures,
        "topology_mismatches": topology_mismatches,
        "critical_multistroke_topology_mismatches": dict(critical_topology_mismatches),
        "missing_legacy_required": missing_legacy,
        "incomplete_legacy_episodes": incomplete_legacy_episodes,
        "duplicate_tensors": duplicate_tensors,
        "finite": bool(np.isfinite(values).all()),
        "hwr_top5_minimum": metrics["top5"] >= 0.999999,
        "per_writer_hwr_top5_minimum": min(row["top5"] for row in per_writer_hwr.values()) >= 0.999999,
        "critical_token_top5_minimum": min(per_class_hwr[token]["top5"] for token in CRITICAL_MULTISTROKE) >= 0.999999,
        "writer_trajectory_separation": minimum_pair_median_rms >= 0.006,
        "identity_fallback_required": identity_fallback_required == 0,
        "real_style_rows_ratio": min(real_style_by_writer.values()) >= 0.999999,
        "minimum_spatial_rms": bool(spatial_rms and min(spatial_rms) > 1.0e-5),
        "uniform_time_contract": uniform_time_contract,
    }
    status = "STYLE_BANK_SHADOW_CANDIDATE" if (
        not missing_legacy and not incomplete_legacy_episodes and not parent_overlap_failures and not topology_mismatches
        and not duplicate_tensors and gates["finite"] and gates["hwr_top5_minimum"]
        and gates["per_writer_hwr_top5_minimum"] and gates["critical_token_top5_minimum"]
        and gates["writer_trajectory_separation"] and gates["identity_fallback_required"]
        and gates["real_style_rows_ratio"] and gates["minimum_spatial_rms"] and gates["uniform_time_contract"]
    ) else "STYLE_BANK_REJECTED"
    output = args.output.resolve(); output.mkdir(parents=True)
    bank_path = output / "synthetic_writer_style_bank_v5.npz"
    np.savez_compressed(
        bank_path, features=values, labels=label_values, writer_index=writer_values,
        episode_split=split_values,
        parent_synthetic_id=np.asarray(output_parent, dtype="<U64"),
        synthetic_id=np.asarray(output_synthetic_id, dtype="<U64"),
    )
    metadata_path = output / "synthetic_writer_style_bank_v5.metadata.jsonl.gz"
    with gzip.open(metadata_path, "wt", encoding="utf-8") as stream:
        for row in output_audit:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    latent_path = output / "writer_latents.json"
    latent_path.write_text(json.dumps([asdict(value) for value in latents], ensure_ascii=False, indent=2), encoding="utf-8")
    report = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "writers": len(latents),
        "deterministic_writer_range": {"start": args.writer_start, "stop": args.writers},
        "rows": len(values),
        "classes": len(set(label_values.tolist())),
        "excluded_insufficient_parent_classes": [tokens[label] for label in sorted(set(range(len(tokens))) - set(label_values.tolist()))],
        "calibration_rows": int(np.sum(split_values == 0)),
        "query_rows": int(np.sum(split_values == 1)),
        "hwr": metrics,
        "hwr_before_label_safety_reversion": metrics_before_reversion,
        "external_parent_hwr": source_metrics,
        "style_alpha_distribution": dict(Counter(str(float(value)) for value in selected_alpha.tolist())),
        "identity_fallback_required": identity_fallback_required,
        "procedural_unsafe_rows": procedural_unsafe,
        "real_style_rows_ratio": real_style_by_writer,
        "spatial_rms": {"minimum": min(spatial_rms) if spatial_rms else 0.0, "median": float(np.median(spatial_rms)) if spatial_rms else 0.0},
        "timing_rms": {"minimum": min(timing_rms) if timing_rms else 0.0, "median": float(np.median(timing_rms)) if timing_rms else 0.0},
        "per_writer_hwr": per_writer_hwr,
        "per_class_hwr": per_class_hwr,
        "critical_token_hwr": {token: per_class_hwr[token] for token in sorted(CRITICAL_MULTISTROKE)},
        "writer_trajectory_separation": {
            "minimum_pair_median_rms": minimum_pair_median_rms,
            "median_pair_median_rms": float(np.median(pair_medians)) if pair_medians else 0.0,
            "pairwise": pairwise_separation,
        },
        "gates": gates,
        "inputs": {
            "external_bank": str(args.source.resolve() / "external_profiled_augmented.npz"),
            "external_bank_sha256": sha256(args.source.resolve() / "external_profiled_augmented.npz"),
            "external_metadata_sha256": sha256(args.source.resolve() / "external_profiled_augmented.metadata.jsonl.gz"),
            "external_manifest_sha256": sha256(args.source.resolve() / "manifest.json"),
            "v4_report": str(args.v4.resolve() / "writer_style_report.json"),
            "v4_report_sha256": sha256(args.v4.resolve() / "writer_style_report.json"),
            "v4_source_npz_sha256": v4_report["source"]["npz_sha256"],
            "checkpoint_sha256": sha256(args.checkpoint.resolve()),
        },
        "outputs": {
            "bank_sha256": sha256(bank_path),
            "metadata_sha256": sha256(metadata_path),
            "latents_sha256": sha256(latent_path),
        },
        "contracts": {
            "same_writer_latent_across_classes": True,
            "procedural_parents": "=,+,/ clean-room templates; no project/CROHME/MathWriting parent",
            "pressure_channel": "absent in frozen 128x5 contract; delta_t is fixed and motor variation is represented only through XY progression/dynamics",
            "inert_latent_dimensions": 0,
            "stroke_direction": "never reversed; direction diversity inherited from approved parents",
            "crohme_rows": 0, "mathwriting_rows": 0, "known_replay_rows": 0,
            "legacy_rows": 0, "training_performed": False,
        },
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "status": status, "writers": len(latents), "rows": len(values), "classes": report["classes"], "hwr": metrics, "gates": gates}, ensure_ascii=False))
    return 0 if status == "STYLE_BANK_SHADOW_CANDIDATE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
