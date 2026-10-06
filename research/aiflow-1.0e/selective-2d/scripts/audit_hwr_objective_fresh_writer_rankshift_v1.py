#!/usr/bin/env python3
"""Post-hoc class/rank microscope for a completed paired HWR objective probe."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from build_normalized_ink_v1 import ROOT, _sha256
from run_hwr_architecture_capacity_probe_v1 import (
    DEFAULT_CANONICAL_ROOT,
    NpyDataset,
    ScaledInkClassifier,
    _class_labels,
)


DEFAULT_INPUT = ROOT / "artifacts" / "hwr_augmentation_microscope_20261002" / "objective_ce_vs_hardnegative_weight005_freshsplit_seed20263315_2ep_micro4.json"
DEFAULT_SPLIT = ROOT / "artifacts" / "hwr_augmentation_microscope_20261002" / "uji_writer_group_split_seed20263315_fresh.json"
DEFAULT_CACHE = Path(
    r"D:\AIFlow-Workspace\PrivateData\math-ink-data-collector\derived"
    r"\hwr-objective-hardnegative-20261002\uji-writer-inner-cache-seed20263315"
)
DEFAULT_OUTPUT = ROOT / "artifacts" / "hwr_augmentation_microscope_20261002" / "objective_ce_vs_hardnegative_weight005_seed20263315_classwise_rankshift.json"


@torch.inference_mode()
def _predict(checkpoint_path: Path, dataset: NpyDataset, labels: list[str], device: torch.device, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "aiflow-hwr-objective-competition-probe/v1":
        raise ValueError(f"unexpected objective checkpoint schema: {checkpoint_path}")
    architecture = payload.get("architecture", {})
    model = ScaledInkClassifier(
        len(labels), int(architecture["width"]), int(architecture["layers"]),
        int(architecture["heads"]), int(architecture["feedforward"]),
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device).eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    target_ranks: list[np.ndarray] = []
    top1_predictions: list[np.ndarray] = []
    for features, target in loader:
        logits = model(features.to(device, non_blocking=device.type == "cuda")).float()
        top = logits.topk(5, dim=1).indices
        target_device = target.to(device, non_blocking=device.type == "cuda")
        rank = torch.full_like(target_device, 6)
        for position in range(5):
            rank = torch.where(top[:, position] == target_device, position + 1, rank)
        target_ranks.append(rank.cpu().numpy())
        top1_predictions.append(top[:, 0].cpu().numpy())
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(target_ranks), np.concatenate(top1_predictions)


def _transition_table(before: np.ndarray, after: np.ndarray) -> list[dict[str, int]]:
    counts = Counter(zip(before.tolist(), after.tolist()))
    return [
        {"before_target_rank": int(left), "after_target_rank": int(right), "rows": int(count)}
        for (left, right), count in sorted(counts.items())
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--canonical-root", type=Path, default=DEFAULT_CANONICAL_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None, help="default to the reported training device, or CUDA when available")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite rank-shift report: {args.output}")
    if args.batch_size < 1:
        parser.error("batch size must be positive")
    input_path, split_path = args.input.resolve(), args.split.resolve()
    source = json.loads(input_path.read_text(encoding="utf-8"))
    if source.get("schema") != "aiflow-hwr-objective-competition-probe/v2" or source.get("status") != "completed_exploratory_objective_ablation":
        raise ValueError("rank-shift audit requires a completed objective report")
    data = source["data"]
    reported_device = source.get("experiment", {}).get("device")
    device_name = args.device or reported_device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested by CLI/report but unavailable")
    device = torch.device(device_name)
    if data.get("crohme_rows") != 0 or data.get("project_owned_holdout_used") is not False:
        raise ValueError("objective report violates the CROHME/project-holdout boundary")
    if data.get("validation_writer_clusters") != 8:
        raise ValueError("expected the pinned eight-writer internal validation split")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if split.get("schema") != "aiflow-hwr-uji-writer-group-split/v1" or split.get("status") != "completed":
        raise ValueError("invalid writer split manifest")
    if _sha256(split_path) != data.get("writer_split_sha256"):
        raise ValueError("writer split hash differs from the completed objective report")
    if split.get("official_split", {}).get("official_test_writers_used_for_inner_selection") is not False:
        raise ValueError("official UJI test writers were not explicitly excluded")
    if split.get("candidate_protocol", {}).get("crohme_rows") != 0:
        raise ValueError("writer split does not attest CROHME exclusion")

    canonical_root, cache_root = args.canonical_root.resolve(), args.cache_root.resolve()
    for filename, expected in (
        ("hwrt.jsonl.gz", data["canonical_hwrt_sha256"]),
        ("uji.jsonl.gz", data["canonical_uji_sha256"]),
    ):
        if _sha256(canonical_root / filename) != expected:
            raise ValueError(f"canonical source hash changed: {filename}")
    cache_hashes = data["cache_audit"]["prepared_cache_sha256"]
    for filename, expected in cache_hashes.items():
        if _sha256(cache_root / filename) != expected:
            raise ValueError(f"prepared cache hash changed: {filename}")

    labels = _class_labels(canonical_root)
    validation = NpyDataset(cache_root / "validation_features.npy", cache_root / "validation_labels.npy")
    targets = np.asarray(validation.labels, dtype=np.int64)
    if len(validation) != data["validation_rows"] or len(validation) != data["cache_audit"]["rows"]["validation"]:
        raise ValueError("validation cache row count differs from the report")
    seed_run = source["seed_runs"][0]
    if not {"ce_pk", "ce_plus_hardnegative"} <= set(seed_run["arms"]):
        raise ValueError("paired CE and hard-negative arms are missing")
    ranks: dict[str, np.ndarray] = {}
    predictions: dict[str, np.ndarray] = {}
    checkpoint_hashes: dict[str, str] = {}
    for arm_name in ("ce_pk", "ce_plus_hardnegative"):
        arm = seed_run["arms"][arm_name]
        checkpoint = Path(arm["checkpoint"]).resolve()
        digest = _sha256(checkpoint)
        if digest != arm["checkpoint_sha256"]:
            raise ValueError(f"checkpoint hash changed: {checkpoint}")
        ranks[arm_name], predictions[arm_name] = _predict(checkpoint, validation, labels, device, args.batch_size)
        checkpoint_hashes[arm_name] = digest

    before, after = ranks["ce_pk"], ranks["ce_plus_hardnegative"]
    if int((before == 1).sum()) != seed_run["paired_comparison"]["top1"]["before_hits"]:
        raise AssertionError("CE Top-1 rank counts disagree with the paired report")
    if int((after == 1).sum()) != seed_run["paired_comparison"]["top1"]["after_hits"]:
        raise AssertionError("challenger Top-1 rank counts disagree with the paired report")
    if int((before <= 5).sum()) != seed_run["paired_comparison"]["top5"]["before_hits"]:
        raise AssertionError("CE Top-5 rank counts disagree with the paired report")
    if int((after <= 5).sum()) != seed_run["paired_comparison"]["top5"]["after_hits"]:
        raise AssertionError("challenger Top-5 rank counts disagree with the paired report")

    per_class: list[dict[str, Any]] = []
    for class_id in sorted(set(targets.tolist())):
        selected = targets == class_id
        before_class, after_class = before[selected], after[selected]
        per_class.append({
            "label": labels[class_id],
            "support": int(selected.sum()),
            "ce_top1_hits": int((before_class == 1).sum()),
            "hardnegative_top1_hits": int((after_class == 1).sum()),
            "top1_delta_hits": int((after_class == 1).sum() - (before_class == 1).sum()),
            "top1_recovered": int(((before_class != 1) & (after_class == 1)).sum()),
            "top1_regressed": int(((before_class == 1) & (after_class != 1)).sum()),
            "ce_top5_hits": int((before_class <= 5).sum()),
            "hardnegative_top5_hits": int((after_class <= 5).sum()),
            "top5_delta_hits": int((after_class <= 5).sum() - (before_class <= 5).sum()),
            "top5_recovered": int(((before_class == 6) & (after_class <= 5)).sum()),
            "top5_regressed": int(((before_class <= 5) & (after_class == 6)).sum()),
        })

    changed_confusions = Counter()
    before_predictions, after_predictions = predictions["ce_pk"], predictions["ce_plus_hardnegative"]
    for index in np.flatnonzero(before_predictions != after_predictions):
        changed_confusions[(labels[int(targets[index])], labels[int(before_predictions[index])], labels[int(after_predictions[index])])] += 1

    output = {
        "schema": "aiflow-hwr-objective-fresh-writer-rankshift/v1",
        "status": "posthoc_consumed_internal_validation_diagnostic",
        "input_report": str(input_path),
        "input_report_sha256": _sha256(input_path),
        "protocol": {
            "training_performed": False,
            "model_or_objective_selection_performed": False,
            "evaluation_device": str(device),
            "evaluation_device_source": "cli" if args.device else "objective_report" if reported_device else "cuda_if_available_fallback",
            "crohme_rows": 0,
            "official_uji_test_scored": False,
            "writer_clusters": len(split["inner_split"]["validation_writer_hashes"]),
            "present_labels": len(set(targets.tolist())),
            "interpretation_limit": "post-hoc class/rank analysis of one eight-writer internal split; not independent product acceptance",
        },
        "provenance": {
            "writer_split_sha256": _sha256(split_path),
            "validation_cache_sha256": {
                name: _sha256(cache_root / name)
                for name in ("validation_features.npy", "validation_labels.npy")
            },
            "checkpoint_sha256": checkpoint_hashes,
        },
        "summary": {
            "rows": len(validation),
            "ce_top1": int((before == 1).sum()),
            "hardnegative_top1": int((after == 1).sum()),
            "top1_recovered": int(((before != 1) & (after == 1)).sum()),
            "top1_regressed": int(((before == 1) & (after != 1)).sum()),
            "top1_regressions_still_in_top5": int(((before == 1) & (after >= 2) & (after <= 5)).sum()),
            "top1_recoveries_from_top5": int(((before >= 2) & (before <= 5) & (after == 1)).sum()),
            "ce_top5": int((before <= 5).sum()),
            "hardnegative_top5": int((after <= 5).sum()),
            "target_entered_top5": int(((before == 6) & (after <= 5)).sum()),
            "target_left_top5": int(((before <= 5) & (after == 6)).sum()),
            "target_rank_1_to_6_before_after": _transition_table(before, after),
            "changed_top1_prediction_rows": int((before_predictions != after_predictions).sum()),
            "most_common_changed_confusions": [
                {"target": target, "ce_prediction": ce_pred, "hardnegative_prediction": hn_pred, "rows": count}
                for (target, ce_pred, hn_pred), count in changed_confusions.most_common(30)
            ],
        },
        "largest_top1_regressions": sorted((row for row in per_class if row["top1_delta_hits"] < 0), key=lambda row: row["top1_delta_hits"])[:15],
        "largest_top1_recoveries": sorted((row for row in per_class if row["top1_delta_hits"] > 0), key=lambda row: row["top1_delta_hits"], reverse=True)[:15],
        "per_class": per_class,
        "product_adopted": False,
    }
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({
        "event": "objective_fresh_writer_rankshift_audit_complete",
        "report": str(output_path),
        "rows": len(validation),
        "present_labels": len(per_class),
        "top1_recovered": output["summary"]["top1_recovered"],
        "top1_regressed": output["summary"]["top1_regressed"],
        "top5_entered": output["summary"]["target_entered_top5"],
        "top5_left": output["summary"]["target_left_top5"],
        "crohme_rows": 0,
        "product_adopted": False,
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
