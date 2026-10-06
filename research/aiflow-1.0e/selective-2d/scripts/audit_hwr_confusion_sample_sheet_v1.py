#!/usr/bin/env python3
"""Render hash-attested model-input strokes for selected HWR rank flips.

This is a read-only post-hoc audit. It never trains, selects, or promotes a
model. It verifies the paired Top-1 counts before saving an anonymized contact
sheet and a small provenance report. CROHME is excluded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from build_normalized_ink_v1 import _sha256
from run_hwr_architecture_capacity_probe_v1 import (
    DEFAULT_CANONICAL_ROOT,
    DEFAULT_CURATED,
    NpyDataset,
    ScaledInkClassifier,
    _class_labels,
)


DEFAULT_INPUT = Path(__file__).resolve().parents[1] / "artifacts" / "hwr_augmentation_microscope_20261002" / "objective_hardnegative_top5gate_seed20261101_last8.json"
DEFAULT_SPLIT = Path(__file__).resolve().parents[1] / "artifacts" / "hwr_augmentation_microscope_20261002" / "uji_writer_group_split_seed20261101_last8_fresh.json"
DEFAULT_CACHE = Path(
    r"D:\AIFlow-Workspace\PrivateData\math-ink-data-collector\derived"
    r"\hwr-objective-competition-20261002\last8-split-seed20261101-cache"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "artifacts" / "hwr_augmentation_microscope_20261002" / "hardnegative-confusion-stroke-samples-v1.png"
CONFUSION_PAIRS = (("l", r"\ell"), ("L", r"\iota"), ("Z", "z"), ("p", r"\wp"))


@torch.inference_mode()
def _predict(checkpoint: Path, validation: NpyDataset, labels: list[str], device: torch.device, batch_size: int) -> np.ndarray:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("schema") != "aiflow-hwr-objective-competition-probe/v1":
        raise ValueError(f"unexpected checkpoint schema: {checkpoint}")
    architecture = payload.get("architecture", {})
    expected = {"width": 128, "layers": 4, "heads": 4, "feedforward": 512}
    if any(architecture.get(key) != value for key, value in expected.items()):
        raise ValueError(f"unexpected checkpoint architecture: {checkpoint}")
    model = ScaledInkClassifier(len(labels), 128, 4, 4, 512).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    loader = DataLoader(validation, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    predictions: list[np.ndarray] = []
    for features, _targets in loader:
        logits = model(features.to(device, non_blocking=device.type == "cuda")).float()
        predictions.append(logits.argmax(dim=1).cpu().numpy())
    del model, payload
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(predictions)


def _draw_strokes(axis: Any, feature: np.ndarray) -> None:
    starts = np.flatnonzero(feature[:, 3] > 0.5).tolist()
    if not starts or starts[0] != 0:
        starts = [0] + starts
    ends = starts[1:] + [len(feature)]
    for start, end in zip(starts, ends, strict=True):
        xy = feature[start:end, :2]
        if len(xy) == 0:
            continue
        axis.plot(xy[:, 0], xy[:, 1], color="#26374a", linewidth=1.8, solid_capstyle="round", solid_joinstyle="round")
        axis.scatter([xy[0, 0]], [xy[0, 1]], s=11, color="#ad5a38", zorder=3)
    axis.set_xlim(-0.04, 1.04)
    axis.set_ylim(1.04, -0.04)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--canonical-root", type=Path, default=DEFAULT_CANONICAL_ROOT)
    parser.add_argument("--curated", type=Path, default=DEFAULT_CURATED)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch size must be positive")

    input_path, split_path = args.input.resolve(), args.split.resolve()
    output_path = args.output.resolve()
    report_path = output_path.with_suffix(".json")
    if output_path.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite an existing sample sheet or provenance report")
    source = json.loads(input_path.read_text(encoding="utf-8"))
    if source.get("schema") != "aiflow-hwr-objective-competition-probe/v2" or source.get("status") != "completed_exploratory_objective_ablation":
        raise ValueError("expected a completed paired objective report")
    data = source["data"]
    if data.get("crohme_rows") != 0 or data.get("project_owned_holdout_used") is not False:
        raise ValueError("objective report violates the CROHME/project-holdout boundary")
    if data.get("validation_writer_clusters") != 8 or source.get("experiment", {}).get("product_adopted") is not False:
        raise ValueError("expected the shadow-only eight-writer diagnostic cohort")
    objective = source.get("experiment", {}).get("objective", {})
    if objective.get("hardnegative_gate") != "target_in_current_top5":
        raise ValueError("sample sheet expects Top-5-gated hard-negative training")

    split = json.loads(split_path.read_text(encoding="utf-8"))
    if split.get("schema") != "aiflow-hwr-uji-writer-group-split/v1" or split.get("status") != "completed":
        raise ValueError("invalid writer split manifest")
    if _sha256(split_path) != data.get("writer_split_sha256"):
        raise ValueError("writer split fingerprint mismatch")
    if split.get("official_split", {}).get("official_test_writers_used_for_inner_selection") is not False:
        raise ValueError("official UJI test writers were not explicitly excluded")
    if split.get("candidate_protocol", {}).get("crohme_rows") != 0:
        raise ValueError("writer split does not attest CROHME exclusion")

    canonical_root, curated_path, cache_root = args.canonical_root.resolve(), args.curated.resolve(), args.cache_root.resolve()
    expected_sources = {
        canonical_root / "hwrt.jsonl.gz": data.get("canonical_hwrt_sha256"),
        canonical_root / "uji.jsonl.gz": data.get("canonical_uji_sha256"),
        curated_path: data.get("curated_uji_sha256"),
    }
    for path, expected_hash in expected_sources.items():
        if not expected_hash or _sha256(path) != expected_hash:
            raise ValueError(f"source fingerprint mismatch: {path.name}")
    cache_hashes = data.get("cache_audit", {}).get("prepared_cache_sha256", {})
    for filename, expected_hash in cache_hashes.items():
        if _sha256(cache_root / filename) != expected_hash:
            raise ValueError(f"prepared cache fingerprint mismatch: {filename}")

    labels = _class_labels(canonical_root)
    validation = NpyDataset(cache_root / "validation_features.npy", cache_root / "validation_labels.npy")
    targets = np.asarray(validation.labels, dtype=np.int64)
    if len(validation) != data.get("validation_rows") or validation.features.shape[1:] != (128, 5):
        raise ValueError("validation model-input contract differs from the report")
    device_name = args.device or source.get("experiment", {}).get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA selected by CLI/report but unavailable")
    device = torch.device(device_name)

    seed_run = source["seed_runs"][0]
    predictions: dict[str, np.ndarray] = {}
    checkpoint_hashes: dict[str, str] = {}
    for arm_name in ("ce_pk", "ce_plus_hardnegative"):
        arm = seed_run["arms"][arm_name]
        checkpoint = Path(arm["checkpoint"]).resolve()
        checkpoint_hashes[arm_name] = _sha256(checkpoint)
        if checkpoint_hashes[arm_name] != arm.get("checkpoint_sha256"):
            raise ValueError(f"checkpoint fingerprint mismatch: {arm_name}")
        predictions[arm_name] = _predict(checkpoint, validation, labels, device, args.batch_size)
        if len(predictions[arm_name]) != len(validation):
            raise AssertionError(f"prediction row count mismatch: {arm_name}")
    comparison = seed_run["paired_comparison"]
    for arm_name, key in (("ce_pk", "before_hits"), ("ce_plus_hardnegative", "after_hits")):
        hits = int((predictions[arm_name] == targets).sum())
        if hits != comparison["top1"][key]:
            raise AssertionError(f"{arm_name} Top-1 count differs from its paired report: {hits}")

    selected: list[dict[str, Any]] = []
    for truth, rival in CONFUSION_PAIRS:
        truth_id, rival_id = labels.index(truth), labels.index(rival)
        indices = np.flatnonzero(
            (targets == truth_id)
            & (predictions["ce_pk"] == truth_id)
            & (predictions["ce_plus_hardnegative"] == rival_id)
        )
        if len(indices) < 2:
            raise ValueError(f"fewer than two verified CE-to-HN flips for {truth!r} -> {rival!r}")
        for example_number, index in enumerate(indices[:2], start=1):
            selected.append({"truth": truth, "ce_prediction": truth, "hardnegative_prediction": rival, "example_number": example_number, "validation_row_index": int(index)})

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(len(CONFUSION_PAIRS), 2, figsize=(8.4, 9.3), squeeze=False)
    for row_index, (truth, rival) in enumerate(CONFUSION_PAIRS):
        figure.text(0.5, 0.985 - row_index * 0.234, f"true {truth}  |  CE {truth}  →  hard-negative {rival}", ha="center", va="top", fontsize=11)
        examples = [item for item in selected if item["truth"] == truth]
        for column_index, item in enumerate(examples):
            axis = axes[row_index, column_index]
            feature = np.asarray(validation.features[item["validation_row_index"]], dtype=np.float32)
            _draw_strokes(axis, feature)
            axis.set_title(f"held-out sample {item['example_number']}", fontsize=9, pad=1)
    figure.suptitle("Near-homograph Top-1 flips", fontsize=13, y=0.998)
    figure.text(0.5, 0.008, "Normalized 128-point model inputs; one dot marks each stroke start. No writer IDs shown.", ha="center", va="bottom", fontsize=9)
    figure.subplots_adjust(left=0.08, right=0.92, top=0.94, bottom=0.04, hspace=0.48, wspace=0.4)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as stream:
        figure.savefig(stream, format="png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    audit = {
        "schema": "aiflow-hwr-confusion-sample-sheet-audit/v1",
        "status": "completed_posthoc_sample_visualization",
        "input_report": str(input_path),
        "input_report_sha256": _sha256(input_path),
        "writer_split_sha256": _sha256(split_path),
        "validation_cache_sha256": {name: _sha256(cache_root / name) for name in cache_hashes},
        "checkpoint_sha256": checkpoint_hashes,
        "protocol": {"training_performed": False, "model_or_objective_selection_performed": False, "evaluation_device": str(device), "crohme_rows": 0, "official_uji_test_scored": False, "product_adopted": False},
        "validation_rows": len(validation),
        "samples": selected,
        "image": str(output_path),
    }
    with report_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(audit, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"status": audit["status"], "image": str(output_path), "report": str(report_path), "samples": len(selected), "crohme_rows": 0, "evaluation_device": str(device)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
