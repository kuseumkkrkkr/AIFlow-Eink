#!/usr/bin/env python3
"""Train a tiny script-relation network without noncommercial training data."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn

from character_tensor_v1 import _json_lines
from evaluate_48hz_prefix_v1 import _sha256
from formula_layout_v1 import _box, _reference_height
from formula_script_network_v1 import (
    CLASSES, FEATURES, MODEL_CONFIG, SCHEMA, ScriptRelationNetwork, pair_features,
)


SEED = 20260819
MAX_GAP_REFERENCE = 1.25


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _geometry(left: float, top: float, width: float, height: float) -> dict[str, float]:
    return {
        "left": left, "top": top, "right": left + width, "bottom": top + height,
        "width": width, "height": height,
        "cx": left + width / 2.0, "cy": top + height / 2.0,
    }


def _synthetic(count_per_class: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    features, labels = [], []
    reference = 100.0
    for label in range(len(CLASSES)):
        for _ in range(count_per_class):
            parent_width = rng.uniform(45.0, 105.0)
            parent_height = rng.uniform(75.0, 125.0)
            parent = _geometry(0.0, 0.0, parent_width, parent_height)
            if label in (1, 2):
                child_height = parent_height * rng.uniform(0.22, 0.70)
                child_width = parent_width * rng.uniform(0.22, 0.80)
                gap = reference * rng.uniform(-0.08, 0.72)
                child_left = parent["right"] + gap
                displacement = parent_height * rng.uniform(0.32, 0.88)
                child_cy = parent["cy"] + (-displacement if label == 1 else displacement)
            else:
                mode = rng.randrange(4)
                gap = reference * rng.uniform(-0.08, 1.15)
                child_left = parent["right"] + gap
                if mode == 0:  # ordinary baseline neighbor
                    child_height = parent_height * rng.uniform(0.65, 1.25)
                    child_width = parent_width * rng.uniform(0.45, 1.20)
                    child_cy = parent["cy"] + parent_height * rng.uniform(-0.16, 0.16)
                elif mode == 1:  # small but not vertically displaced
                    child_height = parent_height * rng.uniform(0.20, 0.68)
                    child_width = parent_width * rng.uniform(0.20, 0.75)
                    child_cy = parent["cy"] + parent_height * rng.uniform(-0.18, 0.18)
                elif mode == 2:  # displaced, but body-sized
                    child_height = parent_height * rng.uniform(0.82, 1.30)
                    child_width = parent_width * rng.uniform(0.45, 1.10)
                    child_cy = parent["cy"] + parent_height * rng.choice((-1, 1)) * rng.uniform(0.30, 0.75)
                else:  # ambiguous overlap near the baseline
                    child_height = parent_height * rng.uniform(0.45, 0.85)
                    child_width = parent_width * rng.uniform(0.35, 0.90)
                    child_left = parent["right"] - child_width * rng.uniform(0.10, 0.80)
                    child_cy = parent["cy"] + parent_height * rng.uniform(-0.22, 0.22)
            child_top = child_cy - child_height / 2.0
            child = _geometry(child_left, child_top, child_width, child_height)
            row = pair_features(parent, child, reference)
            row = [value + rng.gauss(0.0, 0.015) for value in row]
            features.append(row)
            labels.append(label)
    order = list(range(len(labels)))
    rng.shuffle(order)
    return (
        np.asarray([features[index] for index in order], dtype=np.float32),
        np.asarray([labels[index] for index in order], dtype=np.int64),
    )


def _verify_flat_owned(dataset_root: Path, formula_ids: set[str]) -> dict:
    formulas = {
        str(row["sample_id"]): row
        for row in _json_lines(dataset_root / "data" / "formulas_valid.jsonl")
    }
    accepted = {
        str(row["sample_id"])
        for row in _json_lines(dataset_root / "data" / "ownership_train.jsonl")
        if row.get("accepted")
    }
    missing = sorted(formula_ids - accepted)
    relation_bearing = sorted(
        formula_id for formula_id in formula_ids
        if formula_id in formulas and formulas[formula_id].get("target_relations")
    )
    if missing or relation_bearing:
        raise ValueError(
            f"direct negative contract failed: missing_accepted={missing[:5]}, "
            f"relation_bearing={relation_bearing[:5]}"
        )
    return {
        "accepted_flat_formulas": len(formula_ids),
        "relation_bearing_formulas": 0,
    }


def _direct_pairs(rows: list[dict]) -> tuple[np.ndarray, list[str], list[str]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["formula_id"])].append(row)
    features, formula_ids, partitions = [], [], []
    for formula_id, sequence in grouped.items():
        boxes = [_box(row) for row in sequence]
        reference = _reference_height(boxes)
        partition = str(sequence[0].get("evaluation_partition", ""))
        if any(str(row.get("evaluation_partition", "")) != partition for row in sequence):
            raise ValueError(f"mixed direct partition inside formula: {formula_id}")
        for child, child_box in enumerate(boxes):
            for parent, parent_box in enumerate(boxes):
                if parent == child or parent_box["cx"] >= child_box["cx"]:
                    continue
                gap = max(0.0, child_box["left"] - parent_box["right"])
                if gap > reference * MAX_GAP_REFERENCE:
                    continue
                features.append(pair_features(parent_box, child_box, reference))
                formula_ids.append(formula_id)
                partitions.append(partition)
    return np.asarray(features, dtype=np.float32), formula_ids, partitions


def _relation_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict:
    result, f1_values = {}, []
    for index, name in enumerate(CLASSES[1:], start=1):
        tp = int(np.sum((truth == index) & (prediction == index)))
        fp = int(np.sum((truth != index) & (prediction == index)))
        fn = int(np.sum((truth == index) & (prediction != index)))
        precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        result[name] = {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}
        f1_values.append(f1)
    return {"macro_f1": sum(f1_values) / len(f1_values), "by_relation": result}


def _threshold_prediction(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    relation = probabilities[:, 1:].argmax(axis=1) + 1
    confidence = probabilities[np.arange(len(probabilities)), relation]
    return np.where(confidence >= threshold, relation, 0)


def _negative_metrics(probabilities: np.ndarray, formula_ids: list[str], threshold: float) -> dict:
    prediction = _threshold_prediction(probabilities, threshold)
    false_pairs = int(np.sum(prediction != 0))
    false_formulas = len({formula_id for formula_id, value in zip(formula_ids, prediction, strict=True) if value != 0})
    return {
        "pairs": len(prediction), "false_positive_pairs": false_pairs,
        "pair_false_positive_rate": false_pairs / max(len(prediction), 1),
        "formulas": len(set(formula_ids)), "false_positive_formulas": false_formulas,
        "formula_false_positive_rate": false_formulas / max(len(set(formula_ids)), 1),
    }


@torch.inference_mode()
def _probabilities(model: nn.Module, values: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    return model(torch.from_numpy(values).to(device)).softmax(dim=1).cpu().numpy()


def _self_test() -> None:
    values, labels = _synthetic(4, SEED)
    assert values.shape == (12, len(FEATURES)) and set(labels.tolist()) == {0, 1, 2}
    parent, child = _geometry(0, 0, 50, 100), _geometry(50, -20, 20, 30)
    assert len(pair_features(parent, child, 100)) == len(FEATURES)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-candidates", type=Path)
    parser.add_argument("--owned-dataset-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        print(json.dumps({"self_test": "pass"}))
        return 0
    if None in (args.direct_candidates, args.owned_dataset_root, args.checkpoint, args.report):
        parser.error("direct candidates, owned dataset root, checkpoint, and report are required")
    if args.checkpoint.exists() or args.report.exists():
        parser.error("refusing to overwrite script-layout artifact")
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    if device_name == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    device = torch.device(device_name)
    _set_seed(SEED)
    direct_rows = list(_json_lines(args.direct_candidates))
    flat_audit = _verify_flat_owned(
        args.owned_dataset_root, {str(row["formula_id"]) for row in direct_rows},
    )
    direct_x, direct_formulas, partitions = _direct_pairs(direct_rows)
    legacy_mask = np.asarray([partition == "legacy47" for partition in partitions])
    holdout_mask = ~legacy_mask
    if not legacy_mask.any() or not holdout_mask.any():
        raise ValueError("direct flat negatives require legacy and new-arrival partitions")
    synthetic_x, synthetic_y = _synthetic(6000, SEED)
    validation_x, validation_y = _synthetic(1500, SEED + 1)
    train_x = np.concatenate((synthetic_x, np.repeat(direct_x[legacy_mask], 4, axis=0)))
    train_y = np.concatenate((synthetic_y, np.zeros(int(legacy_mask.sum()) * 4, dtype=np.int64)))
    mean, scale = train_x.mean(axis=0), train_x.std(axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale).astype(np.float32)
    mean = mean.astype(np.float32)
    train_x = (train_x - mean) / scale
    validation_x = (validation_x - mean) / scale
    direct_normalized = (direct_x - mean) / scale
    model = ScriptRelationNetwork().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    generator = torch.Generator().manual_seed(SEED)
    best = None
    curve = []
    train_tensor = torch.from_numpy(train_x)
    train_targets = torch.from_numpy(train_y)
    validation_tensor = torch.from_numpy(validation_x).to(device)
    validation_targets = torch.from_numpy(validation_y).to(device)
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(train_targets), generator=generator)
        total = 0.0
        for start in range(0, len(order), 512):
            indices = order[start:start + 512]
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(
                model(train_tensor[indices].to(device)), train_targets[indices].to(device),
            )
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(indices)
        model.eval()
        with torch.inference_mode():
            validation_loss = float(nn.functional.cross_entropy(model(validation_tensor), validation_targets))
        curve.append({"epoch": epoch, "train_loss": total / len(order), "validation_loss": validation_loss})
        if best is None or validation_loss < best["validation_loss"]:
            best = {
                "epoch": epoch, "validation_loss": validation_loss,
                "state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
            }
    model.load_state_dict(best["state_dict"], strict=True)
    validation_probability = _probabilities(model, validation_x, device)
    legacy_probability = _probabilities(model, direct_normalized[legacy_mask], device)
    holdout_probability = _probabilities(model, direct_normalized[holdout_mask], device)
    maximum_legacy = float(legacy_probability[:, 1:].max())
    thresholds = sorted(set(
        [float(value) for value in np.linspace(0.50, 0.99, 50)]
        + [min(0.9999, maximum_legacy + 1e-6)]
    ))
    trials = []
    for threshold in thresholds:
        prediction = _threshold_prediction(validation_probability, threshold)
        metrics = _relation_metrics(validation_y, prediction)
        legacy = _negative_metrics(
            legacy_probability, [formula for formula, keep in zip(direct_formulas, legacy_mask, strict=True) if keep], threshold,
        )
        trials.append({"threshold": threshold, "synthetic_validation": metrics, "legacy_flat": legacy})
    eligible = [row for row in trials if row["legacy_flat"]["false_positive_formulas"] == 0]
    if not eligible:
        raise AssertionError("no zero-false-positive threshold on legacy flat formulas")
    selected = max(eligible, key=lambda row: (
        row["synthetic_validation"]["macro_f1"],
        min(value["precision"] for value in row["synthetic_validation"]["by_relation"].values()),
        -row["threshold"],
    ))
    threshold = float(selected["threshold"])
    holdout = _negative_metrics(
        holdout_probability, [formula for formula, keep in zip(direct_formulas, holdout_mask, strict=True) if keep], threshold,
    )
    minimum_precision = min(
        value["precision"] for value in selected["synthetic_validation"]["by_relation"].values()
    )
    admitted = (
        holdout["false_positive_formulas"] == 0
        and selected["synthetic_validation"]["macro_f1"] >= 0.90
        and minimum_precision >= 0.95
    )
    payload = {
        "schema": SCHEMA, "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_config": MODEL_CONFIG, "features": list(FEATURES), "classes": list(CLASSES),
        "state_dict": best["state_dict"],
        "normalization": {"mean": mean.tolist(), "scale": scale.tolist()},
        "threshold": threshold, "max_horizontal_gap_reference": MAX_GAP_REFERENCE,
        "selected_epoch": int(best["epoch"]), "shadow_admitted": admitted,
        "training_data": {
            "crohme_used": False,
            "synthetic_source": "deterministic in-repository geometry generator",
            "synthetic_training_examples": len(synthetic_y),
            "direct_negative_partition": "legacy47 only",
            "direct_candidates_sha256": _sha256(args.direct_candidates),
        },
    }
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.checkpoint)
    report = {
        "schema": "aiflow-formula-script-network-training/v1",
        "generated_at": payload["generated_at"], "device": str(device),
        "architecture": {**MODEL_CONFIG, "parameters": sum(parameter.numel() for parameter in model.parameters())},
        "data": {
            **flat_audit, "synthetic_train": len(synthetic_y), "synthetic_validation": len(validation_y),
            "legacy_negative_pairs": int(legacy_mask.sum()), "holdout_negative_pairs": int(holdout_mask.sum()),
            "crohme_used": False,
        },
        "selection": {"epoch": best["epoch"], "threshold": threshold, "trials": trials},
        "selected_synthetic_validation": selected["synthetic_validation"],
        "legacy_flat": selected["legacy_flat"], "new_arrival_flat_holdout": holdout,
        "shadow_admitted": admitted,
        "admission_rule": "zero new-arrival flat formula false positives, synthetic macro F1 >=0.90, per-relation precision >=0.95",
        "curve": curve,
        "checkpoint": {"path": str(args.checkpoint), "sha256": _sha256(args.checkpoint)},
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({
        "checkpoint": report["checkpoint"], "selected_epoch": best["epoch"],
        "threshold": threshold, "synthetic": selected["synthetic_validation"],
        "legacy_flat": selected["legacy_flat"], "new_arrival_flat_holdout": holdout,
        "shadow_admitted": admitted, "report": str(args.report),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
