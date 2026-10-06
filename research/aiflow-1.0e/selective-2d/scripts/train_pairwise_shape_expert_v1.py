#!/usr/bin/env python3
"""Train and audit the external-data-only 8/p frozen-embedding expert."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
import torch

from calibrate_project_punctuation_v1 import (
    _collision_free_eval_indices,
    _encode,
)
from character_tensor_v1 import ROOT
from evaluate_48hz_prefix_v1 import INPUT_MODE, _load_model, _sha256
from pairwise_shape_rescue_v1 import (
    CONFIG_SCHEMA,
    FEATURE_DIMENSION,
    MODEL_SCHEMA,
    RESCUE_SCHEMA,
    validate_configuration,
    validate_expert,
)


SEED = 20260820
NEGATIVE_TOKEN = "p"
POSITIVE_TOKEN = "8"
LOGISTIC_C = 0.01
MINIMUM_EXPERT_MARGIN = 0.10
HWR_CANDIDATE_WIDTH = 20
MINIMUM_HWR_PROBABILITY_RATIO = 0.05
DEFAULT_CACHE = (
    ROOT / "artifacts" / "unified_head_20260813" / "unified_math_8ep_full" / "cache"
)
DEFAULT_HWR = (
    ROOT / "artifacts" / "unified_head_20260814"
    / "uniform_time_final_all_writers" / "project_symbol_head_checkpoint.pt"
)
DEFAULT_OUTPUT = (
    ROOT / "artifacts" / "pairwise_shape_expert_20260820_r1_shadow"
)


def _d_path(path: Path, kind: str) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.drive.upper() != "D:":
        raise ValueError(f"{kind} must remain on D:: {resolved}")
    return resolved


def _json_lines(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _device(name: str) -> torch.device:
    resolved = "cuda" if name == "auto" and torch.cuda.is_available() else (
        "cpu" if name == "auto" else name
    )
    if resolved == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    return torch.device(resolved)


@torch.inference_mode()
def _head_logits(model, embeddings: torch.Tensor, device: torch.device) -> np.ndarray:
    chunks = []
    for start in range(0, len(embeddings), 1024):
        chunks.append(model.math_head(embeddings[start:start + 1024].to(device)).cpu())
    return torch.cat(chunks).numpy()


def _classification_counts(before: np.ndarray, after: np.ndarray, truth: np.ndarray) -> dict:
    changed = before != after
    improved = changed & (before != truth) & (after == truth)
    regressed = changed & (before == truth) & (after != truth)
    return {
        "records": int(len(truth)),
        "baseline_top1_hits": int((before == truth).sum()),
        "challenger_top1_hits": int((after == truth).sum()),
        "changed": int(changed.sum()),
        "improved": int(improved.sum()),
        "regressed": int(regressed.sum()),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )


def train_and_evaluate(
    cache_dir: Path, hwr_checkpoint: Path, output_dir: Path,
    *, device: torch.device, batch_size: int,
) -> dict[str, Any]:
    cache_dir = _d_path(cache_dir, "cache")
    hwr_checkpoint = _d_path(hwr_checkpoint, "HWR checkpoint")
    output_dir = _d_path(output_dir, "output")
    manifest_path = cache_dir / "cache_manifest.json"
    if not manifest_path.is_file() or not hwr_checkpoint.is_file():
        raise FileNotFoundError("cache manifest or HWR checkpoint is missing")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = dict(manifest.get("config") or {})
    sets = dict(manifest.get("sets") or {})
    labels = list(config.get("math_labels") or [])
    if (
        config.get("schema") != "aiflow-character-classifier-cache/v2"
        or config.get("points") != 128
        or config.get("channels") != [
            "x", "y", "delta_t", "stroke_start", "observed",
        ]
        or len(labels) != 372
        or config.get("head_mode") != "unified-math"
        or any(token not in labels for token in (NEGATIVE_TOKEN, POSITIVE_TOKEN))
    ):
        raise ValueError("unsupported external cache contract")
    train_info = dict(sets.get("math_train") or {})
    eval_info = dict(sets.get("math_eval") or {})
    train_feature_path = cache_dir / str(train_info.get("features", ""))
    train_label_path = cache_dir / str(train_info.get("labels", ""))
    eval_feature_path = cache_dir / str(eval_info.get("features", ""))
    eval_label_path = cache_dir / str(eval_info.get("labels", ""))
    eval_truth_path = cache_dir / str(eval_info.get("truth", ""))
    for path in (
        train_feature_path, train_label_path, eval_feature_path,
        eval_label_path, eval_truth_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    train_features = np.load(train_feature_path, mmap_mode="r")
    train_labels = np.load(train_label_path, mmap_mode="r")
    eval_features = np.load(eval_feature_path, mmap_mode="r")
    eval_labels = np.load(eval_label_path, mmap_mode="r")
    eval_truth = _json_lines(eval_truth_path)
    if (
        train_features.shape != (int(train_info["records"]), 128, 5)
        or eval_features.shape != (int(eval_info["records"]), 128, 5)
        or len(train_labels) != len(train_features)
        or len(eval_labels) != len(eval_features)
        or len(eval_truth) != len(eval_features)
    ):
        raise ValueError("external cache array coverage mismatch")

    model, checkpoint_labels, _ = _load_model(hwr_checkpoint, device)
    if checkpoint_labels != labels:
        raise ValueError("cache and production HWR vocabularies differ")
    label_index = {token: index for index, token in enumerate(labels)}
    negative_index = label_index[NEGATIVE_TOKEN]
    positive_index = label_index[POSITIVE_TOKEN]
    pair_indices = np.flatnonzero(
        np.isin(np.asarray(train_labels), [negative_index, positive_index])
    )
    pair_features = np.asarray(train_features[pair_indices])
    pair_targets = (
        np.asarray(train_labels[pair_indices]) == positive_index
    ).astype(np.int64)
    support = {
        NEGATIVE_TOKEN: int((pair_targets == 0).sum()),
        POSITIVE_TOKEN: int((pair_targets == 1).sum()),
    }
    if support != {NEGATIVE_TOKEN: 389, POSITIVE_TOKEN: 325}:
        raise ValueError(f"unexpected external pair support: {support}")
    pair_embeddings = _encode(
        model, pair_features, device, INPUT_MODE, batch_size=batch_size,
    ).numpy()
    classifier = LogisticRegression(
        C=LOGISTIC_C,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=2000,
        random_state=SEED,
    )
    classifier.fit(pair_embeddings, pair_targets)
    if classifier.classes_.tolist() != [0, 1] or classifier.coef_.shape != (1, 128):
        raise ValueError("unexpected pairwise logistic model shape")

    clean_indices, collision_audit = _collision_free_eval_indices(
        train_features, eval_features, INPUT_MODE,
    )
    clean_numpy = clean_indices.numpy()
    clean_features = np.asarray(eval_features[clean_numpy])
    clean_truth_indices = np.asarray(eval_labels[clean_numpy])
    clean_truth_rows = [eval_truth[index] for index in clean_numpy.tolist()]
    clean_embeddings = _encode(
        model, clean_features, device, INPUT_MODE, batch_size=batch_size,
    )
    logits = _head_logits(model, clean_embeddings, device)
    probabilities = torch.from_numpy(logits).softmax(dim=1).numpy()
    baseline_top1 = logits.argmax(axis=1)
    expert_margins = classifier.decision_function(clean_embeddings.numpy())

    pair_eval_mask = np.isin(
        clean_truth_indices, [negative_index, positive_index],
    )
    pair_truth = clean_truth_indices[pair_eval_mask]
    baseline_pair = np.where(
        logits[pair_eval_mask, positive_index]
        >= logits[pair_eval_mask, negative_index],
        positive_index, negative_index,
    )
    expert_pair = np.where(
        expert_margins[pair_eval_mask] >= 0.0,
        positive_index, negative_index,
    )
    pair_errors = []
    pair_clean_positions = np.flatnonzero(pair_eval_mask)
    for local_index, clean_position in enumerate(pair_clean_positions):
        if expert_pair[local_index] == pair_truth[local_index]:
            continue
        row = clean_truth_rows[int(clean_position)]
        p8 = float(probabilities[clean_position, positive_index])
        pp = float(probabilities[clean_position, negative_index])
        pair_errors.append({
            "collision_free_eval_index": int(clean_position),
            "record_id": str(row["record_id"]),
            "source": str(row["source"]),
            "truth": labels[int(pair_truth[local_index])],
            "expert_prediction": labels[int(expert_pair[local_index])],
            "expert_margin_8_over_p": float(expert_margins[clean_position]),
            "hwr_8_rank": int(1 + (probabilities[clean_position] > p8).sum()),
            "hwr_probability_8": p8,
            "hwr_probability_p": pp,
            "hwr_probability_ratio_8_over_p": p8 / max(pp, 1e-12),
        })

    p8 = probabilities[:, positive_index]
    pp = probabilities[:, negative_index]
    rank8 = 1 + (probabilities > p8[:, None]).sum(axis=1)
    ratio8p = p8 / np.maximum(pp, 1e-12)
    gate_mask = (
        (baseline_top1 == negative_index)
        & (expert_margins >= MINIMUM_EXPERT_MARGIN)
        & (rank8 <= HWR_CANDIDATE_WIDTH)
        & (ratio8p >= MINIMUM_HWR_PROBABILITY_RATIO)
    )
    challenger_top1 = baseline_top1.copy()
    challenger_top1[gate_mask] = positive_index
    external_gate = _classification_counts(
        baseline_top1, challenger_top1, clean_truth_indices,
    )
    changed_truth = {
        token: int(((clean_truth_indices == index) & gate_mask).sum())
        for index, token in enumerate(labels)
        if ((clean_truth_indices == index) & gate_mask).any()
    }
    external_gate["changed_truth"] = changed_truth
    if external_gate["regressed"]:
        raise ValueError("pairwise shape gate regresses the fixed external holdout")

    generated_at = datetime.now(timezone.utc).isoformat()
    input_hashes = {
        "cache_manifest_sha256": _sha256(manifest_path),
        "train_features_sha256": _sha256(train_feature_path),
        "train_labels_sha256": _sha256(train_label_path),
        "eval_features_sha256": _sha256(eval_feature_path),
        "eval_labels_sha256": _sha256(eval_label_path),
        "eval_truth_sha256": _sha256(eval_truth_path),
        "hwr_checkpoint_sha256": _sha256(hwr_checkpoint),
    }
    expert_payload = {
        "schema": MODEL_SCHEMA,
        "generated_at": generated_at,
        "status": "development_only_posthoc_shadow",
        "model_type": "binary_logistic_regression",
        "feature": "frozen_hwr_encoder_embedding",
        "feature_dimension": FEATURE_DIMENSION,
        "negative_class": NEGATIVE_TOKEN,
        "positive_class": POSITIVE_TOKEN,
        "coefficients": classifier.coef_[0].astype(float).tolist(),
        "intercept": float(classifier.intercept_[0]),
        "input_mode": INPUT_MODE,
        "hyperparameters": {
            "C": LOGISTIC_C,
            "class_weight": "balanced",
            "solver": "lbfgs",
            "max_iter": 2000,
            "random_state": SEED,
            "scaler": None,
        },
        "artifacts": input_hashes,
        "training": {
            "external_train_only": True,
            "records": int(len(pair_targets)),
            "support": support,
            "class_balanced": True,
            "encoder_frozen": True,
            "formula_truth_input": False,
            "writer_identity_input": False,
        },
        "requires_explicit_shadow_opt_in": True,
        "product_default_enabled": False,
    }
    validate_expert(expert_payload)
    expert_path = output_dir / "pairwise_shape_expert.json"
    _write_json(expert_path, expert_payload)
    expert_sha256 = _sha256(expert_path)
    configuration_payload = {
        "schema": CONFIG_SCHEMA,
        "rescue_schema": RESCUE_SCHEMA,
        "generated_at": generated_at,
        "status": "development_only_posthoc_shadow",
        "expert_model_sha256": expert_sha256,
        "gate": {
            "baseline_token": NEGATIVE_TOKEN,
            "challenger_token": POSITIVE_TOKEN,
            "minimum_expert_margin": MINIMUM_EXPERT_MARGIN,
            "hwr_candidate_width": HWR_CANDIDATE_WIDTH,
            "minimum_hwr_probability_ratio": MINIMUM_HWR_PROBABILITY_RATIO,
            "challenger_must_be_in_original_hwr_candidates": True,
        },
        "threshold_provenance": {
            "external_false_positive_margin_observed": (
                max((row["expert_margin_8_over_p"] for row in pair_errors), default=None)
            ),
            "current_159_formula_inspected": True,
            "external_gate_positive_coverage": external_gate["changed"],
            "fresh_writer_formula_disjoint_acceptance_required": True,
        },
        "contracts": {
            "candidate_preserving": True,
            "insertions_or_deletions": 0,
            "glyph_order_mutations": 0,
            "target_label_or_glyph_count_input": False,
            "writer_identity_input": False,
            "arithmetic_evaluation": False,
        },
        "requires_explicit_shadow_opt_in": True,
        "product_default_enabled": False,
    }
    validate_configuration(
        configuration_payload, expert_model_sha256=expert_sha256,
    )
    configuration_path = output_dir / "pairwise_shape_rescue_config.json"
    _write_json(configuration_path, configuration_payload)
    configuration_sha256 = _sha256(configuration_path)
    report = {
        "schema": "aiflow-pairwise-shape-expert-training-report/v1",
        "generated_at": generated_at,
        "status": "development_only_posthoc_shadow",
        "decision": "continue_to_formula_and_crohme_no_regression_gates",
        "training": expert_payload["training"],
        "collision_audit": collision_audit,
        "fixed_external_pair_eval": {
            "records": int(pair_eval_mask.sum()),
            "support": {
                NEGATIVE_TOKEN: int((pair_truth == negative_index).sum()),
                POSITIVE_TOKEN: int((pair_truth == positive_index).sum()),
            },
            "production_hwr_pairwise_hits": int((baseline_pair == pair_truth).sum()),
            "expert_pairwise_hits": int((expert_pair == pair_truth).sum()),
            "expert_errors": pair_errors,
        },
        "fixed_collision_free_external_gate": external_gate,
        "artifacts": {
            "expert": expert_path.name,
            "expert_sha256": expert_sha256,
            "configuration": configuration_path.name,
            "configuration_sha256": configuration_sha256,
            **input_hashes,
        },
        "software": {
            "numpy": np.__version__,
            "pytorch": torch.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "limits": [
            "the expert standalone pair decision makes one fixed external p-to-8 error",
            "the deployed gate has zero changed external rows and therefore no external positive coverage",
            "the threshold was inspected against the current 159 formulas",
            "fresh commercial writer/formula-disjoint p and 8 acceptance remains required",
        ],
        "product_default_enabled": False,
    }
    report_path = output_dir / "training_evaluation_report.json"
    _write_json(report_path, report)
    return {
        "output": str(output_dir),
        "expert_sha256": expert_sha256,
        "configuration_sha256": configuration_sha256,
        "report_sha256": _sha256(report_path),
        "training_support": support,
        "pair_eval": report["fixed_external_pair_eval"],
        "external_gate": external_gate,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--hwr-checkpoint", type=Path, default=DEFAULT_HWR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    result = train_and_evaluate(
        args.cache_dir, args.hwr_checkpoint, args.output,
        device=_device(args.device), batch_size=args.batch_size,
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
