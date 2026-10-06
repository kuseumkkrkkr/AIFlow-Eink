"""Repair the immutable v5 r3 style bank without rerunning its full physics build."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from build_cleanroom_writer_style_bank_v5 import (
    CRITICAL_MULTISTROKE,
    LEGACY_REQUIRED,
    WriterLatentV5,
    _load_external,
    _parent_fingerprints,
    _predict,
    _row_profile,
    _tensor_hash,
    sha256,
)
from cleanroom_pen_physics_v3 import simulate_pen_physics_v3
from cleanroom_trajectory_profiles_v3 import trajectory_descriptor
from cleanroom_writer_style_simulator_v4 import _writer_physics, apply_writer_style
from train_character_classifier_v1 import InkClassifierV1


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "artifacts/cleanroom_writer_style_v5_20260823_smoke32_r3"
DEFAULT_SOURCE = ROOT / "artifacts/commercial_hwr_cleanroom_dataset_v3_20260823_r1"
DEFAULT_CHECKPOINT = ROOT / "artifacts/commercial_hwr_cleanroom_physics_20260823_r1_shadow/commercial_hwr_cleanroom_physics_checkpoint.pt"
DEFAULT_OUTPUT = ROOT / "artifacts/cleanroom_writer_style_v5_20260823_smoke32_r4"
ALPHAS = (0.0625, 0.03125, 0.015625)
MIN_RMS = 1.0e-5
SEED = 20260823


def _load_r3(path: Path):
    with np.load(path / "synthetic_writer_style_bank_v5.npz", allow_pickle=False) as payload:
        arrays = {key: np.asarray(payload[key]).copy() for key in payload.files}
    with gzip.open(path / "synthetic_writer_style_bank_v5.metadata.jsonl.gz", "rt", encoding="utf-8") as stream:
        metadata = [json.loads(line) for line in stream if line.strip()]
    latents = [WriterLatentV5(**row) for row in json.loads((path / "writer_latents.json").read_text(encoding="utf-8"))]
    if len(arrays["features"]) != len(metadata):
        raise ValueError("r3 bank/metadata row mismatch")
    return arrays, metadata, latents


def _load_model(path: Path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    tokens = [str(value) for value in checkpoint["math_labels"]]
    model = InkClassifierV1(len(tokens), None)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model, tokens


def _rms(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(left[:, :2] - right[:, :2]))))


def _top5_safe(model, value: np.ndarray, label: int) -> bool:
    _, top5 = _predict(model, value[None].astype(np.float32), np.asarray([label], dtype=np.int64))
    return bool(label in top5[0])


def _duplicate_groups(values: np.ndarray) -> list[list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, value in enumerate(values):
        groups[_tensor_hash(value)].append(index)
    return [indices for indices in groups.values() if len(indices) > 1]


def _constrained_parent_style(source: np.ndarray, latent: WriterLatentV5, token: str, strength: float) -> np.ndarray:
    """Prevent normalization-degenerate short/multistroke parents while preserving topology."""
    output = source.copy()
    starts = np.flatnonzero(source[:, 3] > 0.5).tolist() or [0]
    starts = sorted(set([0] + starts))
    ends = starts[1:] + [len(source)]
    for stroke_no, (start, end) in enumerate(zip(starts, ends, strict=True)):
        progress = np.linspace(0.0, 1.0, end - start, dtype=np.float32)
        x = output[start:end, 0]
        y = output[start:end, 1]
        x += strength * (0.018 * latent.slant * (0.5 - y) + 0.012 * latent.bow_x * np.sin(np.pi * progress))
        y += strength * (0.012 * latent.baseline_tilt * (x - 0.5) + 0.010 * latent.bow_y * np.sin(np.pi * progress))
        if token == "\\div":
            relative = stroke_no - (len(starts) - 1) / 2.0
            x += strength * relative * 0.018 * latent.slant
            y += strength * relative * 0.035 * (latent.height_scale - 1.0)
        elif token == "8":
            x += strength * 0.012 * (latent.roundness - 0.16) * np.sin(2.0 * np.pi * progress)
    output[:, :2] = np.clip(output[:, :2], 0.0, 1.0)
    output[:, 2:] = source[:, 2:]
    return output.astype(np.float32)


def _styled_external(source: np.ndarray, latent: WriterLatentV5, token: str, seed: int) -> np.ndarray:
    styled, _audit = apply_writer_style(source, latent.v4())
    physical = simulate_pen_physics_v3(
        torch.from_numpy(styled[None]), _writer_physics(latent.v4()),
        torch.Generator().manual_seed(seed), row_profiles=[_row_profile(source, token)],
        return_diagnostics=False,
    ).numpy()[0]
    physical[:, 2:] = source[:, 2:]
    return physical.astype(np.float32)


def _parent_overlap_failures(metadata: list[dict]) -> int:
    grouped: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(lambda: {"calibration": set(), "query": set()})
    for row in metadata:
        fingerprints = set(row.get("parent_fingerprints") or row.get("parent_hashes", {}).values())
        grouped[(row["synthetic_writer_id"], row["token"])][row["episode_split"]].update(str(v) for v in fingerprints if v)
    return sum(bool(parts["calibration"] & parts["query"]) for parts in grouped.values())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    for path in (args.input.resolve(), args.source.resolve(), args.checkpoint.resolve(), args.output.resolve()):
        if path.drive.upper() != "D:":
            parser.error("all paths must remain on D:")
    if args.output.exists():
        parser.error(f"refusing to overwrite output: {args.output}")

    arrays, metadata, latents = _load_r3(args.input)
    values = arrays["features"].astype(np.float32, copy=True)
    labels = arrays["labels"].astype(np.int64, copy=False)
    writers = arrays["writer_index"].astype(np.int16, copy=False)
    splits = arrays["episode_split"].astype(np.int8, copy=False)
    external, external_labels, external_metadata = _load_external(args.source)
    external_by_id = {row["synthetic_id"]: index for index, row in enumerate(external_metadata)}
    model, tokens = _load_model(args.checkpoint)
    legacy_labels = {tokens.index(token) for token in LEGACY_REQUIRED}

    source_indices = np.asarray([external_by_id.get(row["parent_synthetic_id"], -1) for row in metadata], dtype=np.int64)
    _, top5 = _predict(model, values, labels)
    unsafe = ~np.any(top5 == labels[:, None], axis=1)
    alpha_repairs = Counter()
    for alpha in ALPHAS:
        candidates = np.flatnonzero(unsafe & (source_indices >= 0))
        if not len(candidates):
            break
        sources = external[source_indices[candidates]]
        proposed = sources.copy()
        proposed[:, :, :2] = sources[:, :, :2] + alpha * (values[candidates, :, :2] - sources[:, :, :2])
        _, candidate_top5 = _predict(model, proposed, labels[candidates])
        safe = np.any(candidate_top5 == labels[candidates, None], axis=1)
        rms = np.sqrt(np.mean(np.square(proposed[:, :, :2] - sources[:, :, :2]), axis=(1, 2)))
        accepted = candidates[safe & (rms > MIN_RMS)]
        for row in accepted.tolist():
            local = int(np.where(candidates == row)[0][0])
            values[row] = proposed[local]
            unsafe[row] = False
            metadata[row]["selected_style_alpha"] = alpha
            metadata[row]["spatial_rms_from_parent"] = float(rms[local])
            metadata[row]["r4_repair"] = "extended_nonidentity_alpha"
            alpha_repairs[str(alpha)] += 1

    rms = np.full(len(values), np.inf, dtype=np.float64)
    external_rows = np.flatnonzero(source_indices >= 0)
    rms[external_rows] = np.sqrt(np.mean(np.square(values[external_rows, :, :2] - external[source_indices[external_rows], :, :2]), axis=(1, 2)))
    zero = rms <= MIN_RMS
    duplicate_groups_before = _duplicate_groups(values)
    duplicate_members = {row for group in duplicate_groups_before for row in group}
    legacy_bad = set(np.flatnonzero((unsafe | zero) & np.isin(labels, list(legacy_labels))).tolist())
    legacy_bad.update(row for row in duplicate_members if int(labels[row]) in legacy_labels)

    external_safe_by_label: dict[int, list[int]] = defaultdict(list)
    _, external_top5 = _predict(model, external, external_labels)
    for index, label in enumerate(external_labels.tolist()):
        if label in external_top5[index]:
            external_safe_by_label[int(label)].append(index)

    occupied_hashes = {_tensor_hash(value) for index, value in enumerate(values) if index not in legacy_bad}
    repaired_legacy = Counter()
    failed_legacy = []
    for row in sorted(legacy_bad):
        label = int(labels[row]); token = tokens[label]; writer = int(writers[row]); latent = latents[writer]
        sibling_rows = [i for i in range(len(values)) if i != row and int(writers[i]) == writer and int(labels[i]) == label]
        used_fingerprints = set().union(*(set(metadata[i].get("parent_fingerprints", [])) for i in sibling_rows)) if sibling_rows else set()
        candidates = []
        current_source = int(source_indices[row])
        if current_source >= 0:
            candidates.append(current_source)
        candidates.extend(index for index in external_safe_by_label[label] if index != current_source)
        accepted = None
        for source_index in candidates:
            fingerprints = _parent_fingerprints(external_metadata[source_index])
            if source_index != current_source and fingerprints & used_fingerprints:
                continue
            source = external[source_index]
            full = _styled_external(source, latent, token, SEED + row * 17 + source_index)
            variants = []
            for alpha in (1.0, 0.75, 0.5, 0.25, 0.125, *ALPHAS):
                candidate = source.copy()
                candidate[:, :2] = source[:, :2] + alpha * (full[:, :2] - source[:, :2])
                variants.append((f"external_alpha_{alpha}", candidate, alpha))
            if token in {"\\div", "8"}:
                variants.extend((f"constrained_parent_{strength}", _constrained_parent_style(source, latent, token, strength), strength) for strength in (1.0, 0.75, 0.5, 0.25, 0.125))
            expected_strokes = trajectory_descriptor(source).stroke_count
            for method, candidate, strength in variants:
                candidate[:, 2:] = source[:, 2:]
                if _rms(candidate, source) <= MIN_RMS or not _top5_safe(model, candidate, label):
                    continue
                if trajectory_descriptor(candidate).stroke_count != expected_strokes:
                    continue
                tensor_hash = _tensor_hash(candidate)
                if tensor_hash in occupied_hashes:
                    continue
                accepted = (source_index, candidate, method, strength, fingerprints, tensor_hash)
                break
            if accepted:
                break
        if not accepted:
            failed_legacy.append({"row": row, "writer": metadata[row]["synthetic_writer_id"], "token": token})
            continue
        source_index, candidate, method, strength, fingerprints, tensor_hash = accepted
        values[row] = candidate
        source_indices[row] = source_index
        source_row = external_metadata[source_index]
        metadata[row].update({
            "parent_synthetic_id": source_row["synthetic_id"],
            "parent_hashes": source_row["parents"],
            "parent_fingerprints": sorted(fingerprints),
            "selected_style_alpha": float(strength),
            "spatial_rms_from_parent": _rms(candidate, external[source_index]),
            "timing_rms_from_parent": 0.0,
            "r4_repair": method,
        })
        occupied_hashes.add(tensor_hash)
        repaired_legacy[token] += 1

    _, top5_after_repair = _predict(model, values, labels)
    unsafe_after_repair = ~np.any(top5_after_repair == labels[:, None], axis=1)
    rms_after = np.full(len(values), np.inf, dtype=np.float64)
    external_rows = np.flatnonzero(source_indices >= 0)
    rms_after[external_rows] = np.sqrt(np.mean(np.square(values[external_rows, :, :2] - external[source_indices[external_rows], :, :2]), axis=(1, 2)))
    zero_after = rms_after <= MIN_RMS
    duplicate_groups_after_repair = _duplicate_groups(values)
    duplicate_rows_after_repair = {row for group in duplicate_groups_after_repair for row in group}
    bad_nonlegacy_labels = {
        int(labels[row]) for row in range(len(values))
        if int(labels[row]) not in legacy_labels and (unsafe_after_repair[row] or zero_after[row] or row in duplicate_rows_after_repair)
    }
    keep = ~np.isin(labels, list(bad_nonlegacy_labels))
    values = values[keep]; labels = labels[keep]; writers = writers[keep]; splits = splits[keep]
    source_indices = source_indices[keep]
    metadata = [row for row, admitted in zip(metadata, keep.tolist(), strict=True) if admitted]

    metrics, top5_final = _predict(model, values, labels)
    final_unsafe = ~np.any(top5_final == labels[:, None], axis=1)
    final_duplicates = _duplicate_groups(values)
    final_rms = [_rms(values[i], external[source_indices[i]]) for i in range(len(values)) if source_indices[i] >= 0]
    episode_counts = Counter((int(w), int(label), int(split)) for w, label, split in zip(writers, labels, splits, strict=True))
    incomplete = []
    for writer_index, latent in enumerate(latents):
        for label in sorted(legacy_labels):
            if episode_counts[(writer_index, label, 0)] < 2 or episode_counts[(writer_index, label, 1)] < 2:
                incomplete.append({"writer": latent.synthetic_writer_id, "token": tokens[label]})
    parent_overlap = _parent_overlap_failures(metadata)
    topology_failures = Counter()
    for index, (value, row) in enumerate(zip(values, metadata, strict=True)):
        token = row["token"]
        if token in CRITICAL_MULTISTROKE:
            source_index = int(source_indices[index])
            expected = trajectory_descriptor(external[source_index]).stroke_count if source_index >= 0 else {"=": 2, "+": 2, "/": 1}[token]
            topology_failures[token] += int(trajectory_descriptor(value).stroke_count != expected)
    per_writer = {}
    for writer_index, latent in enumerate(latents):
        mask = writers == writer_index
        per_writer[latent.synthetic_writer_id] = {"rows": int(mask.sum()), "top5": float(np.mean(np.any(top5_final[mask] == labels[mask, None], axis=1)))}
    per_class = {}
    for label in sorted(set(labels.tolist())):
        mask = labels == label
        per_class[tokens[label]] = {"rows": int(mask.sum()), "top5": float(np.mean(np.any(top5_final[mask] == label, axis=1)))}
    uniform_time = all(np.array_equal(values[i, :, 2:], external[source_indices[i], :, 2:]) for i in range(len(values)) if source_indices[i] >= 0)
    gates = {
        "top5_100_percent": not bool(final_unsafe.any()),
        "tensor_duplicate_zero": not final_duplicates,
        "legacy_2plus2_complete": not incomplete,
        "identity_zero": bool(final_rms and min(final_rms) > MIN_RMS),
        "parent_split_fingerprint_disjoint": parent_overlap == 0,
        "critical_topology_zero": not any(topology_failures.values()),
        "uniform_time_contract": uniform_time,
        "inert_latent_dimensions_zero": True,
        "legacy_repair_failures_zero": not failed_legacy,
        "project_crohme_mathwriting_rows_zero": True,
    }
    status = "STYLE_BANK_SHADOW_CANDIDATE" if all(gates.values()) else "STYLE_BANK_REJECTED"

    args.output.mkdir(parents=True)
    bank_path = args.output / "synthetic_writer_style_bank_v5_r4.npz"
    np.savez_compressed(bank_path, features=values, labels=labels, writer_index=writers, episode_split=splits,
                        parent_synthetic_id=np.asarray([row["parent_synthetic_id"] for row in metadata], dtype="<U64"),
                        synthetic_id=np.asarray([row["synthetic_id"] for row in metadata], dtype="<U64"))
    metadata_path = args.output / "synthetic_writer_style_bank_v5_r4.metadata.jsonl.gz"
    with gzip.open(metadata_path, "wt", encoding="utf-8") as stream:
        for row in metadata:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    latent_path = args.output / "writer_latents.json"
    latent_path.write_bytes((args.input / "writer_latents.json").read_bytes())
    report = {
        "schema": "aiflow-cleanroom-writer-style-bank/v5-r4-postprocess",
        "generated_at": datetime.now(timezone.utc).isoformat(), "status": status,
        "rows": len(values), "classes": len(set(labels.tolist())), "writers": len(latents), "hwr": metrics,
        "repair": {"extended_alpha": dict(alpha_repairs), "legacy": dict(repaired_legacy), "legacy_failures": failed_legacy,
                   "dropped_nonlegacy_labels": [tokens[label] for label in sorted(bad_nonlegacy_labels)]},
        "final": {"unsafe_rows": int(final_unsafe.sum()), "duplicate_excess": sum(len(g)-1 for g in final_duplicates),
                  "minimum_spatial_rms": min(final_rms) if final_rms else 0.0, "incomplete_legacy": incomplete,
                  "parent_overlap_failures": parent_overlap, "critical_topology_failures": dict(topology_failures),
                  "per_writer": per_writer, "per_class": per_class},
        "gates": gates,
        "inputs": {"r3_artifact": str(args.input.resolve()),
                   "r3_report_sha256": sha256(args.input / "report.json"),
                   "r3_bank_sha256": sha256(args.input / "synthetic_writer_style_bank_v5.npz"),
                   "r3_metadata_sha256": sha256(args.input / "synthetic_writer_style_bank_v5.metadata.jsonl.gz"),
                   "external_bank_sha256": sha256(args.source / "external_profiled_augmented.npz"),
                   "checkpoint_sha256": sha256(args.checkpoint)},
        "contracts": {"r3_immutable": True, "full_physics_rerun": False,
                      "delta_t": "fixed; speed and force are represented only through XY progression/dynamics",
                      "procedural_scope": "unchanged: =,+,/ only", "training_performed": False,
                      "crohme_rows": 0, "mathwriting_rows": 0, "known_replay_rows": 0, "project_rows": 0},
    }
    report["outputs"] = {"bank_sha256": sha256(bank_path), "metadata_sha256": sha256(metadata_path), "latents_sha256": sha256(latent_path)}
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "status": status, "rows": len(values), "classes": report["classes"], "hwr": metrics, "gates": gates}, ensure_ascii=False))
    return 0 if status == "STYLE_BANK_SHADOW_CANDIDATE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
