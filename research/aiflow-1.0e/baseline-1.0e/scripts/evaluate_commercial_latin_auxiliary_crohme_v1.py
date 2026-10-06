#!/usr/bin/env python3
"""Validate the frozen commercial Latin auxiliary HWR on CROHME test ink.

CROHME is non-commercial validation input only.  This command performs no
training, epoch selection, threshold selection, or gradient update.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from build_crohme_standard_candidates_v1 import _crohme_items_tolerant
from character_tensor_v1 import ROOT, _json_lines
from evaluate_48hz_prefix_v1 import _prefix_tensor
from train_character_classifier_v1 import InkClassifierV1, apply_input_mode


SCHEMA = "aiflow-commercial-latin-auxiliary-crohme-validation/v1"
CHECKPOINT_SCHEMA = "aiflow-commercial-latin-auxiliary/v1"
EXPANSION_LABELS = frozenset({"t", ",", ".", "!"})
DEFAULT_CHECKPOINT = (
    ROOT / "artifacts" / "commercial_latin_auxiliary_20260821_r1_shadow"
    / "latin_auxiliary_checkpoint.pt"
)
DEFAULT_CROHME = (
    ROOT / "datasets" / "30_noncommercial_evaluation" / "crohme2019"
    / "crohme2019" / "crohme2019" / "test"
)
DEFAULT_OUTSIDE = (
    ROOT / "artifacts" / "crohme_stream_rank_context_expanded_20260820_r2_shadow"
    / "crohme_test_truth_outside_top5.jsonl.gz"
)
DEFAULT_OUTPUT = (
    ROOT / "artifacts" / "commercial_latin_auxiliary_crohme_20260821_r1"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _summary(rows: list[dict]) -> dict:
    if not rows:
        return {"records": 0, "top1": None, "top5": None}
    return {
        "records": len(rows),
        "top1": sum(row["top1"] == row["truth"] for row in rows) / len(rows),
        "top5": sum(row["truth"] in row["top5"] for row in rows) / len(rows),
    }


def _category(label: str) -> str:
    if label.isascii() and len(label) == 1 and label.isdigit():
        return "digit"
    if label.isascii() and len(label) == 1 and label.isalpha():
        return "latin"
    return "punctuation_or_operator"


def _write_rows(path: Path, rows: list[dict]) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


@torch.inference_mode()
def _predict(
    model: InkClassifierV1, features: np.ndarray, labels: list[str],
    device: torch.device, batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    top5: list[np.ndarray] = []
    ranks: list[np.ndarray] = []
    for start in range(0, len(features), batch_size):
        batch = torch.from_numpy(features[start:start + batch_size]).to(device)
        probability = model(batch, "auxiliary").softmax(dim=1)
        top5.append(probability.topk(5, dim=1).indices.cpu().numpy())
        ranks.append(probability.argsort(dim=1, descending=True).cpu().numpy())
    return np.concatenate(top5), np.concatenate(ranks)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--crohme-root", type=Path, default=DEFAULT_CROHME)
    parser.add_argument("--outside-top5", type=Path, default=DEFAULT_OUTSIDE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    paths = [
        args.checkpoint.resolve(), args.crohme_root.resolve(),
        args.outside_top5.resolve(), args.output.resolve(),
    ]
    if any(path.drive.upper() != "D:" for path in paths):
        parser.error("all inputs and outputs must remain on D:")
    if args.output.exists() or args.batch_size < 1:
        parser.error("output must be new and batch size positive")
    if any(not path.exists() for path in paths[:-1]):
        parser.error("one or more frozen validation inputs are missing")

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("unexpected Latin auxiliary checkpoint schema")
    training_report = payload.get("report", {})
    guard = training_report.get("training_data_guard", {})
    if (
        guard.get("crohme_rows") != 0
        or guard.get("crohme_gradient_updates") != 0
        or training_report.get("crohme_scored_during_training_or_selection") is not False
        or int(training_report.get("data", {}).get("selection", {}).get("writer_overlap", -1)) != 0
    ):
        raise ValueError("Latin auxiliary checkpoint violates validation boundary")
    labels = [str(value) for value in payload.get("auxiliary_labels", [])]
    if len(labels) != 95 or not EXPANSION_LABELS <= set(labels):
        raise ValueError("unexpected Latin auxiliary vocabulary")

    items, coverage, formula_ids, failed_formula_ids, fallbacks, _writers = (
        _crohme_items_tolerant(args.crohme_root, set(labels), "test")
    )
    features = np.stack([_prefix_tensor(row["strokes"]) for row in items]).astype(
        np.float32, copy=False
    )
    features = apply_input_mode(features, "uniform-time")
    resolved_device = (
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    if resolved_device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    device = torch.device(resolved_device)
    model = InkClassifierV1(1, len(labels)).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    top5_indices, rank_indices = _predict(
        model, features, labels, device, args.batch_size,
    )
    label_to_index = {label: index for index, label in enumerate(labels)}
    predictions = []
    for item, top_indices, order in zip(items, top5_indices, rank_indices, strict=True):
        truth = str(item["label"])
        truth_index = label_to_index[truth]
        rank = int(np.flatnonzero(order == truth_index)[0]) + 1
        predictions.append({
            "record_id": str(item["record_id"]),
            "formula_id": str(item["formula_id"]),
            "truth": truth,
            "category": _category(truth),
            "top1": labels[int(top_indices[0])],
            "top5": [labels[int(index)] for index in top_indices],
            "truth_rank": rank,
        })

    outside = {str(row["record_id"]): row for row in _json_lines(args.outside_top5)}
    if len(outside) != 749:
        raise ValueError("frozen main-head outside-Top-5 set is not 749 rows")
    outside_predictions = [row for row in predictions if row["record_id"] in outside]
    outside_ids = {row["record_id"] for row in outside_predictions}
    outside_eligible = {
        record_id for record_id, row in outside.items()
        if str(row["truth"]) in label_to_index
    }
    if outside_ids != outside_eligible:
        raise AssertionError("Latin auxiliary CROHME record coverage mismatch")

    by_category: dict[str, list[dict]] = defaultdict(list)
    by_label: dict[str, list[dict]] = defaultdict(list)
    for row in predictions:
        by_category[row["category"]].append(row)
        by_label[row["truth"]].append(row)
    expansion = [row for row in predictions if row["truth"] in EXPANSION_LABELS]
    outside_metrics = _summary(outside_predictions)
    outside_metrics.update({
        "main_outside_total": 749,
        "auxiliary_vocabulary_eligible": len(outside_predictions),
        "auxiliary_top1_rescue": sum(
            row["top1"] == row["truth"] for row in outside_predictions
        ),
        "auxiliary_top5_rescue": sum(
            row["truth"] in row["top5"] for row in outside_predictions
        ),
        "not_claimed_as_product_rescue": True,
    })
    label_metrics = []
    for label, selected in sorted(by_label.items(), key=lambda value: (-len(value[1]), value[0])):
        metric = _summary(selected)
        metric.update({"label": label})
        label_metrics.append(metric)

    args.output.mkdir(parents=True)
    prediction_path = args.output / "crohme_latin_auxiliary_predictions.jsonl.gz"
    _write_rows(prediction_path, predictions)
    report = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "noncommercial_validation_only_shadow",
        "training_performed": False,
        "selection_performed": False,
        "threshold_tuning_performed": False,
        "crohme_rows_used_for_training": 0,
        "crohme_gradient_updates": 0,
        "checkpoint_selected_before_crohme_scoring": True,
        "truth_grouping_supplied": True,
        "official_end_to_end_expression_rate": False,
        "metrics": {
            "all_auxiliary_vocabulary": _summary(predictions),
            "by_category": {
                key: _summary(value) for key, value in sorted(by_category.items())
            },
            "new_main_head_expansion_labels": {
                **_summary(expansion),
                "labels": sorted(EXPANSION_LABELS),
                "by_label": {
                    label: _summary(by_label.get(label, []))
                    for label in sorted(EXPANSION_LABELS)
                },
            },
            "frozen_749_main_outside_top5": outside_metrics,
            "by_label": label_metrics,
        },
        "coverage": {
            **coverage,
            "eligible_formulas": len(formula_ids),
            "failed_or_partly_unsupported_formulas": len(failed_formula_ids),
            "parse_fallback_files": len(fallbacks),
        },
        "decision": {
            "product_adopted": False,
            "reason": (
                "CROHME can quantify transfer only; adoption needs untouched project-owned "
                "and commercial-rights formula-complete acceptance data"
            ),
            "safe_integration_candidate": (
                "formula-complete candidate expansion for ASCII glyphs, followed by the "
                "candidate-preserving context and 2D layers"
            ),
        },
        "training_provenance": {
            "training_report": training_report,
            "crohme_used_for_model_epoch_or_threshold_selection": False,
        },
        "inputs": {
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": _sha256(args.checkpoint),
            "crohme_root": str(args.crohme_root.resolve()),
            "outside_top5_sha256": _sha256(args.outside_top5),
        },
        "output": {
            "predictions": str(prediction_path),
            "predictions_sha256": _sha256(prediction_path),
        },
    }
    report_path = args.output / "validation_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "report": str(report_path),
        "all": report["metrics"]["all_auxiliary_vocabulary"],
        "expansion": report["metrics"]["new_main_head_expansion_labels"],
        "outside_749": outside_metrics,
        "crohme_gradient_updates": 0,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
