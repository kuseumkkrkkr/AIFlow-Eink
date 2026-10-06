#!/usr/bin/env python3
"""Paired scratch CE pilot for online stroke augmentation on fresh UJI writers.

Both arms share the same 372-way model, initialization, sampled rows, optimizer,
and validation set. The only treatment is train-time geometric augmentation.
Official UJI test writers, project-owned formula holdouts, and CROHME are excluded.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from audit_hwr_architecture_capacity_paired_v1 import (
    _paired,
    _writer_bootstrap,
    _jsonl_train_uji_writer_hashes,
    _predict,
)
from build_normalized_ink_v1 import ROOT, _sha256
from run_hwr_affine_distillation_experiment_v1 import (
    _augment_with_engine,
    _resolve_wide_parameter_settings,
)
from run_hwr_architecture_capacity_probe_v1 import (
    DEFAULT_CANONICAL_ROOT,
    DEFAULT_CURATED,
    NpyDataset,
    ScaledInkClassifier,
    SYNTHETIC_EQUAL_ROWS,
    _class_labels,
    _evaluate,
    _load_writer_map,
    _prepare_arrays,
)


DEFAULT_SPLIT = ROOT / "artifacts" / "hwr_augmentation_microscope_20261002" / "uji_writer_group_split_seed20261088_fresh.json"
DEFAULT_CACHE = Path(
    r"D:\AIFlow-Workspace\PrivateData\math-ink-data-collector\derived"
    r"\hwr-augmentation-20261002\fresh-writer-paired-cache-seed20261088"
)
DEFAULT_OUTPUT = ROOT / "artifacts" / "hwr_augmentation_microscope_20261002" / "fresh_writer_augmentation_ab_seed20261088.json"
PRIOR_SPLITS = (
    ROOT / "artifacts" / "hwr_augmentation_microscope_20261002" / "uji_writer_group_split_seed20261002.json",
    ROOT / "artifacts" / "hwr_augmentation_microscope_20261002" / "uji_writer_group_split_seed20261019_nooverlap.json",
)
SEED = 20261088
BOOTSTRAP_DRAWS = 20_000
DEFAULT_AUGMENTATION_PARAMETERS: dict[str, float | int] = {
    "elastic_probability": 0.5,
    "rotation_degrees": 3.0,
    "axis_scale_jitter": 0.08,
    "shear_jitter": 0.04,
    "elastic_control_displacement": 0.018,
    "stroke_local_probability": 0.65,
    "stroke_local_rotation_degrees": 2.0,
    "stroke_local_scale_jitter": 0.04,
    "stroke_local_translation_jitter": 0.006,
    "stroke_curve_probability": 0.65,
    "stroke_curve_max_displacement": 0.025,
    "stroke_curve_max_waves": 1,
}


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _validate_augmentation_parameters(parameters: dict[str, float | int]) -> None:
    bounds = {
        "elastic_probability": (0.0, 1.0),
        "rotation_degrees": (0.0, 12.0),
        "axis_scale_jitter": (0.0, 0.25),
        "shear_jitter": (0.0, 0.15),
        "elastic_control_displacement": (0.0, 0.05),
        "stroke_local_probability": (0.0, 1.0),
        "stroke_local_rotation_degrees": (0.0, 5.0),
        "stroke_local_scale_jitter": (0.0, 0.15),
        "stroke_local_translation_jitter": (0.0, 0.05),
        "stroke_curve_probability": (0.0, 1.0),
        "stroke_curve_max_displacement": (0.0, 0.03),
    }
    for name, (minimum, maximum) in bounds.items():
        value = parameters[name]
        if not minimum <= value <= maximum:
            raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    if parameters["stroke_curve_max_waves"] not in (1, 2, 3):
        raise ValueError("stroke_curve_max_waves must be 1, 2, or 3")


def _augment(
    features: torch.Tensor,
    generators: tuple[torch.Generator, ...],
    *,
    wide_parameter_probability: float = 0.0,
    wide_magnitude_multiplier: float = 1.5,
    wide_probability_by_parameter: dict[str, float] | None = None,
    wide_multiplier_by_parameter: dict[str, float] | None = None,
    augmentation_parameters: dict[str, float | int] | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    parameters = {**DEFAULT_AUGMENTATION_PARAMETERS, **(augmentation_parameters or {})}
    _validate_augmentation_parameters(parameters)
    return _augment_with_engine(
        features,
        generators[0],
        "affine_elastic_mix",
        parameters["elastic_probability"],
        wide_parameter_probability=wide_parameter_probability,
        wide_magnitude_multiplier=wide_magnitude_multiplier,
        wide_probability_by_parameter=wide_probability_by_parameter,
        wide_multiplier_by_parameter=wide_multiplier_by_parameter,
        rotation_degrees=parameters["rotation_degrees"],
        axis_scale_jitter=parameters["axis_scale_jitter"],
        shear_jitter=parameters["shear_jitter"],
        elastic_control_displacement=parameters["elastic_control_displacement"],
        stroke_local_probability=parameters["stroke_local_probability"],
        stroke_local_rotation_degrees=parameters["stroke_local_rotation_degrees"],
        stroke_local_scale_jitter=parameters["stroke_local_scale_jitter"],
        stroke_local_translation_jitter=parameters["stroke_local_translation_jitter"],
        stroke_generator=generators[1],
        stroke_curve_probability=parameters["stroke_curve_probability"],
        stroke_curve_max_displacement=parameters["stroke_curve_max_displacement"],
        stroke_curve_max_waves=parameters["stroke_curve_max_waves"],
        curve_generator=generators[2],
    )


def _new_augmentation_generators(device: torch.device, seed: int) -> tuple[torch.Generator, ...]:
    return tuple(torch.Generator(device=device).manual_seed(seed + offset) for offset in (101, 211, 307))


def _augment_contract_smoke(
    train: NpyDataset,
    device: torch.device,
    seed: int,
    *,
    wide_parameter_probability: float,
    wide_magnitude_multiplier: float,
    wide_probability_by_parameter: dict[str, float],
    wide_multiplier_by_parameter: dict[str, float],
    augmentation_parameters: dict[str, float | int],
) -> dict[str, Any]:
    count = min(32, len(train))
    features = torch.from_numpy(np.array(train.features[:count], dtype=np.float32, copy=True)).to(device)
    augmented, diagnostics = _augment(
        features,
        _new_augmentation_generators(device, seed),
        wide_parameter_probability=wide_parameter_probability,
        wide_magnitude_multiplier=wide_magnitude_multiplier,
        wide_probability_by_parameter=wide_probability_by_parameter,
        wide_multiplier_by_parameter=wide_multiplier_by_parameter,
        augmentation_parameters=augmentation_parameters,
    )
    if augmented.shape != features.shape or not torch.isfinite(augmented).all():
        raise AssertionError("augmentation produced invalid shape or non-finite values")
    if not torch.equal(augmented[:, :, 2:], features[:, :, 2:]):
        raise AssertionError("augmentation changed time, stroke-start, or observed channels")
    if torch.any(augmented[:, :, :2] < 0.0) or torch.any(augmented[:, :, :2] > 1.0):
        raise AssertionError("augmentation escaped the normalized canvas")
    if not diagnostics.get("stroke_count_preserved", False):
        raise AssertionError("augmentation did not attest stroke-count preservation")
    if diagnostics.get("changed", 0) == 0:
        raise AssertionError("augmentation smoke batch did not exercise a changed example")
    return {
        "rows": count,
        "changed_rows": diagnostics["changed"],
        "stroke_local_accepted_rows": diagnostics["stroke_local"]["accepted_rows"],
        "stroke_curve_accepted_rows": diagnostics["stroke_curve"]["accepted_rows"],
        "severity_mixture": diagnostics["severity_mixture"],
        "time_stroke_observed_channels_identical": True,
        "normalized_bounds_valid": True,
    }


def _train_arm(
    name: str,
    *,
    augmented: bool,
    labels: list[str],
    cache: dict[str, Any],
    device: torch.device,
    checkpoint_dir: Path,
    seed: int,
    epochs: int,
    batch_size: int,
    eval_batch_size: int,
    learning_rate: float,
    wide_parameter_probability: float,
    wide_magnitude_multiplier: float,
    wide_probability_by_parameter: dict[str, float],
    wide_multiplier_by_parameter: dict[str, float],
    augmentation_parameters: dict[str, float | int],
) -> dict[str, Any]:
    _seed(seed)
    model = ScaledInkClassifier(len(labels), 128, 4, 4, 512).to(device)
    train = NpyDataset(Path(cache["train"]["features"]), Path(cache["train"]["labels"]))
    validation = NpyDataset(Path(cache["validation"]["features"]), Path(cache["validation"]["labels"]))
    target_ids = np.asarray(train.labels, dtype=np.int64)
    counts = np.bincount(target_ids, minlength=len(labels)).astype(np.float64)
    if not np.all(counts > 0):
        raise ValueError("class-balanced training requires support for every model class")
    sampler = WeightedRandomSampler(
        torch.as_tensor(1.0 / counts[target_ids], dtype=torch.double),
        num_samples=len(train), replacement=True,
        generator=torch.Generator().manual_seed(seed + 17),
    )
    loader = DataLoader(
        train, batch_size=batch_size, sampler=sampler, num_workers=0,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1.0e-4)
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    history: list[dict[str, Any]] = []
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        diag_totals = {
            "rows": 0,
            "changed_rows": 0,
            "elastic_selected_rows": 0,
            "composite_gate_reverted_selected_rows": 0,
            "stroke_local_selected_rows": 0,
            "stroke_local_accepted_rows": 0,
            "stroke_curve_selected_rows": 0,
            "stroke_curve_accepted_rows": 0,
            "stroke_curve_direction_reversal_rows": 0,
            "wide_selected_any_parameter_rows": 0,
            "wide_selected_rows_by_parameter": {
                "rotation": 0,
                "axis_scale": 0,
                "shear": 0,
                "elastic": 0,
            },
        }
        generators = _new_augmentation_generators(device, seed + epoch * 1009)
        for batch_index, (features, target) in enumerate(loader, start=1):
            features = features.to(device, non_blocking=amp)
            target = target.to(device, non_blocking=amp)
            if augmented:
                features, diag = _augment(
                    features,
                    generators,
                    wide_parameter_probability=wide_parameter_probability,
                    wide_magnitude_multiplier=wide_magnitude_multiplier,
                    wide_probability_by_parameter=wide_probability_by_parameter,
                    wide_multiplier_by_parameter=wide_multiplier_by_parameter,
                    augmentation_parameters=augmentation_parameters,
                )
                diag_totals["rows"] += int(diag["rows"])
                diag_totals["changed_rows"] += int(diag["changed"])
                diag_totals["elastic_selected_rows"] += int(diag["elastic_selected_rows"])
                diag_totals["composite_gate_reverted_selected_rows"] += int(diag["composite_gate_reverted_selected_rows"])
                local = diag["stroke_local"]
                curve = diag["stroke_curve"]
                diag_totals["stroke_local_selected_rows"] += int(local["selected_rows"])
                diag_totals["stroke_local_accepted_rows"] += int(local["accepted_rows"])
                diag_totals["stroke_curve_selected_rows"] += int(curve["selected_rows"])
                diag_totals["stroke_curve_accepted_rows"] += int(curve["accepted_rows"])
                diag_totals["stroke_curve_direction_reversal_rows"] += int(curve["direction_reversal_rows"])
                severity = diag["severity_mixture"]
                diag_totals["wide_selected_any_parameter_rows"] += int(
                    severity["wide_selected_any_parameter_rows"]
                )
                for parameter_name, selected_rows in severity["wide_selected_rows_by_parameter"].items():
                    diag_totals["wide_selected_rows_by_parameter"][parameter_name] += int(selected_rows)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits = model(features)
                loss = F.cross_entropy(logits.float(), target)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite CE in {name}, epoch={epoch}, batch={batch_index}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))

        metrics = _evaluate(model, validation, labels, device, eval_batch_size)
        epoch_result = {
            "epoch": epoch,
            "train_cross_entropy": float(np.mean(losses)),
            "validation": metrics,
            "augmentation": diag_totals if augmented else {"enabled": False},
        }
        history.append(epoch_result)
        print(json.dumps({"event": "fresh_writer_aug_ab_epoch", "arm": name, **epoch_result}, ensure_ascii=False), flush=True)

    checkpoint_path = checkpoint_dir / f"{name}_seed{seed}.pt"
    if checkpoint_path.exists():
        raise FileExistsError(f"refusing to overwrite paired checkpoint: {checkpoint_path}")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    torch.save({
        "schema": "aiflow-hwr-architecture-capacity-probe/v1",
        "arm": {"name": name, "width": 128, "layers": 4, "heads": 4, "feedforward": 512, "parameters": parameter_count},
        "seed": seed,
        "state_dict": model.state_dict(),
        "initialization": "random; paired CE scratch arm; no checkpoint or teacher logits",
    }, checkpoint_path)
    return {
        "name": name,
        "treatment": "online_stroke_augmentation" if augmented else "none",
        "architecture": {"width": 128, "layers": 4, "heads": 4, "feedforward": 512, "parameters": parameter_count},
        "optimizer": {"name": "AdamW", "learning_rate": learning_rate, "weight_decay": 1.0e-4, "loss": "372-way cross-entropy", "teacher_distillation": False},
        "seed": seed,
        "history": history,
        "final_validation": history[-1]["validation"],
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path, default=DEFAULT_CANONICAL_ROOT)
    parser.add_argument("--curated", type=Path, default=DEFAULT_CURATED)
    parser.add_argument("--writer-split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument(
        "--wide-parameter-probability",
        type=float,
        default=0.0,
        help="independent per-parameter probability of sampling the wider affine/elastic bound",
    )
    parser.add_argument(
        "--wide-magnitude-multiplier",
        type=float,
        default=1.5,
        help="multiplier applied to each independently selected affine/elastic bound",
    )
    parser.add_argument(
        "--wide-probability",
        action="append",
        default=[],
        metavar="PARAMETER=PROBABILITY",
        help="override one transform's wide-magnitude probability; repeat for rotation, axis_scale, shear, or elastic",
    )
    parser.add_argument(
        "--wide-multiplier",
        action="append",
        default=[],
        metavar="PARAMETER=MULTIPLIER",
        help="override one transform's wide-magnitude multiplier; repeat for rotation, axis_scale, shear, or elastic",
    )
    parser.add_argument("--elastic-probability", type=float, default=DEFAULT_AUGMENTATION_PARAMETERS["elastic_probability"])
    parser.add_argument("--rotation-degrees", type=float, default=DEFAULT_AUGMENTATION_PARAMETERS["rotation_degrees"])
    parser.add_argument("--axis-scale-jitter", type=float, default=DEFAULT_AUGMENTATION_PARAMETERS["axis_scale_jitter"])
    parser.add_argument("--shear-jitter", type=float, default=DEFAULT_AUGMENTATION_PARAMETERS["shear_jitter"])
    parser.add_argument("--elastic-control-displacement", type=float, default=DEFAULT_AUGMENTATION_PARAMETERS["elastic_control_displacement"])
    parser.add_argument("--stroke-local-probability", type=float, default=DEFAULT_AUGMENTATION_PARAMETERS["stroke_local_probability"])
    parser.add_argument("--stroke-local-rotation-degrees", type=float, default=DEFAULT_AUGMENTATION_PARAMETERS["stroke_local_rotation_degrees"])
    parser.add_argument("--stroke-local-scale-jitter", type=float, default=DEFAULT_AUGMENTATION_PARAMETERS["stroke_local_scale_jitter"])
    parser.add_argument("--stroke-local-translation-jitter", type=float, default=DEFAULT_AUGMENTATION_PARAMETERS["stroke_local_translation_jitter"])
    parser.add_argument("--stroke-curve-probability", type=float, default=DEFAULT_AUGMENTATION_PARAMETERS["stroke_curve_probability"])
    parser.add_argument("--stroke-curve-max-displacement", type=float, default=DEFAULT_AUGMENTATION_PARAMETERS["stroke_curve_max_displacement"])
    parser.add_argument("--stroke-curve-max-waves", type=int, choices=(1, 2, 3), default=DEFAULT_AUGMENTATION_PARAMETERS["stroke_curve_max_waves"])
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--bootstrap-draws", type=int, default=BOOTSTRAP_DRAWS)
    args = parser.parse_args()
    if args.epochs < 1 or min(args.batch_size, args.eval_batch_size, args.bootstrap_draws) < 1:
        parser.error("epochs, batch sizes, and bootstrap draws must be positive")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite experiment report: {args.output}")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    augmentation_parameters = {
        "elastic_probability": args.elastic_probability,
        "rotation_degrees": args.rotation_degrees,
        "axis_scale_jitter": args.axis_scale_jitter,
        "shear_jitter": args.shear_jitter,
        "elastic_control_displacement": args.elastic_control_displacement,
        "stroke_local_probability": args.stroke_local_probability,
        "stroke_local_rotation_degrees": args.stroke_local_rotation_degrees,
        "stroke_local_scale_jitter": args.stroke_local_scale_jitter,
        "stroke_local_translation_jitter": args.stroke_local_translation_jitter,
        "stroke_curve_probability": args.stroke_curve_probability,
        "stroke_curve_max_displacement": args.stroke_curve_max_displacement,
        "stroke_curve_max_waves": args.stroke_curve_max_waves,
    }
    try:
        _validate_augmentation_parameters(augmentation_parameters)
    except ValueError as exc:
        parser.error(str(exc))
    if not 0.0 <= args.wide_parameter_probability <= 1.0:
        parser.error("--wide-parameter-probability must be in [0, 1]")
    if not 1.0 <= args.wide_magnitude_multiplier <= 3.0:
        parser.error("--wide-magnitude-multiplier must be in [1, 3]")
    try:
        wide_probability_by_parameter, wide_multiplier_by_parameter = _resolve_wide_parameter_settings(
            args.wide_parameter_probability,
            args.wide_magnitude_multiplier,
            args.wide_probability,
            args.wide_multiplier,
        )
    except ValueError as exc:
        parser.error(str(exc))

    canonical_root = args.canonical_root.resolve()
    curated_path = args.curated.resolve()
    split_path = args.writer_split.resolve()
    cache_root = args.cache_root.resolve()
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if split.get("schema") != "aiflow-hwr-uji-writer-group-split/v1" or split.get("status") != "completed":
        raise ValueError("invalid writer split manifest")
    validation_hashes = set(split["inner_split"]["validation_writer_hashes"])
    prior_hashes: set[str] = set()
    prior_split_evidence = []
    for path in PRIOR_SPLITS:
        if not path.is_file():
            raise FileNotFoundError(f"prior split audit is missing: {path}")
        previous = json.loads(path.read_text(encoding="utf-8"))
        previous_hashes = set(previous["inner_split"]["validation_writer_hashes"])
        prior_hashes.update(previous_hashes)
        prior_split_evidence.append({"path": str(path.resolve()), "sha256": _sha256(path), "writers": len(previous_hashes)})
    overlap = sorted(validation_hashes & prior_hashes)
    if overlap:
        raise ValueError(f"fresh validation split overlaps prior architecture/objective probes: {overlap}")
    if len(validation_hashes) != 8:
        raise ValueError("expected exactly eight fresh validation writers")
    if cache_root.exists() and any(cache_root.iterdir()):
        raise FileExistsError(f"refusing to reuse non-empty paired cache: {cache_root}")
    if cache_root == Path(cache_root.anchor) or len(cache_root.parts) < 4:
        raise ValueError("unsafe cache root")

    labels = _class_labels(canonical_root)
    print(json.dumps({"event": "fresh_writer_aug_ab_prepare_start", "train_labels": len(labels), "fresh_validation_writers": len(validation_hashes), "crohme_rows": 0}), flush=True)
    cache = _prepare_arrays(
        canonical_root, curated_path, split_path, labels, cache_root, SYNTHETIC_EQUAL_ROWS,
    )
    print(json.dumps({"event": "fresh_writer_aug_ab_prepare_complete", "rows": cache["rows"], "validation_present_labels": cache["validation_label_coverage"]}), flush=True)
    device = torch.device(args.device)
    train = NpyDataset(Path(cache["cache"]["train"]["features"]), Path(cache["cache"]["train"]["labels"]))
    smoke = _augment_contract_smoke(
        train,
        device,
        args.seed + 991,
        wide_parameter_probability=args.wide_parameter_probability,
        wide_magnitude_multiplier=args.wide_magnitude_multiplier,
        wide_probability_by_parameter=wide_probability_by_parameter,
        wide_multiplier_by_parameter=wide_multiplier_by_parameter,
        augmentation_parameters=augmentation_parameters,
    )
    validation = NpyDataset(Path(cache["cache"]["validation"]["features"]), Path(cache["cache"]["validation"]["labels"]))
    checkpoint_dir = cache_root / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=False)

    common = {
        "labels": labels,
        "cache": cache["cache"],
        "device": device,
        "checkpoint_dir": checkpoint_dir,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "learning_rate": args.learning_rate,
        "wide_parameter_probability": args.wide_parameter_probability,
        "wide_magnitude_multiplier": args.wide_magnitude_multiplier,
        "wide_probability_by_parameter": wide_probability_by_parameter,
        "wide_multiplier_by_parameter": wide_multiplier_by_parameter,
        "augmentation_parameters": augmentation_parameters,
    }
    arms = [
        _train_arm("control_ce", augmented=False, **common),
        _train_arm("affine_local_curve_ce", augmented=True, **common),
    ]
    writer_map = _load_writer_map(curated_path)
    writer_rows = _jsonl_train_uji_writer_hashes(canonical_root, writer_map, validation_hashes, set(labels))
    predictions = [_predict(Path(arm["checkpoint"]), validation, labels, device, args.eval_batch_size) for arm in arms]
    comparison = {
        "top1": _paired(predictions[0][0], predictions[1][0]),
        "top5": _paired(predictions[0][1], predictions[1][1]),
        "top1_writer_cluster_bootstrap": _writer_bootstrap(predictions[0][0], predictions[1][0], writer_rows, args.seed + 1, args.bootstrap_draws),
        "top5_writer_cluster_bootstrap": _writer_bootstrap(predictions[0][1], predictions[1][1], writer_rows, args.seed + 2, args.bootstrap_draws),
        "product_adopted": False,
    }
    report = {
        "schema": "aiflow-hwr-fresh-writer-augmentation-ab/v1",
        "status": "completed_exploratory_paired_writer_audit",
        "data": {
            "writer_split_manifest": str(split_path),
            "writer_split_sha256": _sha256(split_path),
            "validation_writer_hashes": sorted(validation_hashes),
            "validation_writers_disjoint_from_prior_probe_splits": True,
            "prior_split_evidence": prior_split_evidence,
            "prior_validation_writer_overlap": 0,
            "canonical_hwrt_sha256": _sha256(canonical_root / "hwrt.jsonl.gz"),
            "canonical_uji_sha256": _sha256(canonical_root / "uji.jsonl.gz"),
            "curated_uji_sha256": _sha256(curated_path),
            "cache": cache,
            "cache_file_sha256": {name: _sha256(cache_root / name) for name in ("train_features.npy", "train_labels.npy", "validation_features.npy", "validation_labels.npy")},
            "rows": cache["rows"],
            "real_rows": cache["real_rows"],
            "validation_present_labels": cache["validation_label_coverage"],
            "crohme_rows": 0,
        },
        "comparison": {
            "paired_controls": ["same 128x4 architecture", "same seed and initialization", "same class-balanced sampled rows", "same AdamW/CE schedule", "same held-writer validation rows"],
            "control": "no online geometric augmentation",
            "challenger": "affine_elastic_mix + bounded stroke-local affine/curve + optional independent per-parameter magnitude mixture; train only",
            "augmentation_parameters": {
                **augmentation_parameters,
                "wide_parameter_probability": args.wide_parameter_probability,
                "wide_magnitude_multiplier": args.wide_magnitude_multiplier,
                "wide_probability_by_parameter": wide_probability_by_parameter,
                "wide_multiplier_by_parameter": wide_multiplier_by_parameter,
            },
            "augmentation_contract_smoke": smoke,
            "product_adopted": False,
            "official_uji_test_scored": False,
            "project_owned_formula_holdout_scored": False,
            "crohme_used_for_training_selection_or_evaluation": False,
            "interpretation_limit": "single seed and one fresh eight-writer UJI-train split; character-level only; not product acceptance or full 372-class generalization",
        },
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "arms": arms,
        "paired_comparison": comparison,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({
        "event": "fresh_writer_augmentation_ab_complete",
        "output": str(args.output.resolve()),
        "top1_delta_pp": comparison["top1"]["delta_percentage_points"],
        "top5_delta_pp": comparison["top5"]["delta_percentage_points"],
        "top1_writer_bootstrap_ci": comparison["top1_writer_cluster_bootstrap"]["delta_percentage_points_percentile_95_interval"],
        "crohme_rows": 0,
        "product_adopted": False,
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
