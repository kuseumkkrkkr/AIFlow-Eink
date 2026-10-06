#!/usr/bin/env python3
"""상업권리 온라인 잉크만으로 DTW·물리 증강 HWR 후보를 선택한다."""

from __future__ import annotations

from training_data_guard_v1 import assert_training_entrypoint_arguments_clean

assert_training_entrypoint_arguments_clean()

import argparse
import copy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from calibrate_project_punctuation_v1 import _direct_rows, _writer_split_indices
from character_tensor_v1 import CHANNELS, POINTS, ROOT
from cleanroom_online_ink_augmentation_v2 import (
    DtwBank,
    LIGHT_PHYSICS,
    MEDIUM_PHYSICS,
    NO_PHYSICS,
    PhysicalConfig,
    build_dtw_bank,
    save_preview,
    simulate_pen_physics,
)
from train_character_classifier_v1 import (
    InkClassifierV1,
    _dataset,
    input_contract,
    load_vocabs,
    prepare_cache,
)
from train_commercial_hwr_augmentation_v1 import (
    Variant as GeometryVariant,
    _batch,
    _clone_state,
    _logits,
    _metrics,
    _new_model,
    _prediction_rows,
    _sample_probabilities,
    _select,
    _sha256,
    _top5_margin_loss,
    augment_online_tensor,
)
from training_data_guard_v1 import (
    assert_training_path_clean,
    zero_crohme_training_manifest,
)


SCHEMA = "aiflow-commercial-hwr-cleanroom-physics/v2"
LOO_SCHEMA = "aiflow-commercial-hwr-cleanroom-physics-writer-loo/v2"
SEED = 20260823
DEFAULT_CANONICAL = ROOT / "datasets" / "normalized" / "v1"
DEFAULT_CACHE = (
    ROOT / "artifacts" / "unified_head_20260813" / "unified_math_8ep_full" / "cache"
)
DEFAULT_PARENT = (
    ROOT / "artifacts" / "commercial_hwr_augmentation_20260822_r1_shadow"
    / "commercial_hwr_augmented_checkpoint.pt"
)
DEFAULT_PARENT_LOO = (
    ROOT / "artifacts" / "commercial_hwr_augmentation_20260822_r1_shadow"
    / "selected_writer_loo_models.pt"
)
DEFAULT_DIRECT = Path(
    r"D:\AIFlow-Workspace\PrivateData\math-ink-data-collector\derived"
    r"\hwr-head-calibration-20260819-r1\project_owned_ownership_eval_95.jsonl.gz"
)
DEFAULT_OUTPUT = ROOT / "artifacts" / "commercial_hwr_cleanroom_physics_20260823_r1_shadow"


@dataclass(frozen=True)
class Variant:
    """학습 범위와 독립 증강 조합을 한 실험 단위로 선언한다."""

    name: str
    train_scope: str
    learning_rate: float
    geometry: GeometryVariant
    physics: PhysicalConfig
    use_dtw: bool
    top5_margin_weight: float


NO_GEOMETRY = GeometryVariant("none", "last_block", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
MILD_GEOMETRY = GeometryVariant("mild_v2", "last_block", 0.0, 5.0, 0.08, 0.05, 0.008, 0.0)

VARIANTS = {
    row.name: row for row in (
        Variant("control_head", "head", 0.0, NO_GEOMETRY, NO_PHYSICS, False, 0.0),
        Variant("continuation_geometry", "last_block", 1.0e-5, MILD_GEOMETRY, NO_PHYSICS, False, 0.25),
        Variant("physics_light", "last_block", 1.0e-5, NO_GEOMETRY, LIGHT_PHYSICS, False, 0.25),
        Variant("dtw_physics_light", "last_block", 1.0e-5, MILD_GEOMETRY, LIGHT_PHYSICS, True, 0.25),
        Variant("dtw_physics_medium", "last_block", 7.5e-6, MILD_GEOMETRY, MEDIUM_PHYSICS, True, 0.25),
        Variant("dtw_only_fast", "last_block", 2.0e-5, MILD_GEOMETRY, NO_PHYSICS, True, 0.35),
        Variant("dtw_physics_light_fast", "last_block", 2.0e-5, MILD_GEOMETRY, LIGHT_PHYSICS, True, 0.35),
        Variant("dtw_physics_light_top5", "last_block", 1.5e-5, MILD_GEOMETRY, LIGHT_PHYSICS, True, 0.50),
    )
}


def _d_path(path: Path, kind: str, *, must_exist: bool = True) -> Path:
    """모든 입력·출력을 D:로 제한하고 비상업 평가 경로를 차단한다."""

    resolved = assert_training_path_clean(path, kind)
    if resolved.drive.upper() != "D:":
        raise ValueError(f"{kind} must remain on D:: {resolved}")
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"missing {kind}: {resolved}")
    return resolved


def _seed_everything(seed: int) -> None:
    """CPU와 GPU 난수를 고정해 variant 간 비교를 재현 가능하게 만든다."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_parent(path: Path, labels: list[str]) -> tuple[dict[str, torch.Tensor], dict]:
    """이전 상업권리 challenger의 단일 372-class 상태를 검증해 읽는다."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    report = payload.get("report", {})
    if (
        payload.get("schema") != "aiflow-commercial-hwr-augmentation/v1"
        or payload.get("math_labels") != labels
        or payload.get("auxiliary_labels") != []
        or report.get("input_contract", {}).get("observed_channel_mode") != "uniform-time"
    ):
        raise ValueError("parent HWR checkpoint contract mismatch")
    state = {
        key: value.detach().cpu().clone()
        for key, value in payload.get("state_dict", {}).items()
    }
    model = InkClassifierV1(len(labels), 0)
    model.load_state_dict(state, strict=True)
    return state, report


def _load_parent_loo(path: Path, labels: list[str]) -> dict[str, dict[str, torch.Tensor]]:
    """이전 증강에서 선택된 writer-LOO 전체 모델 상태를 검증해 읽는다."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    states = payload.get("states_by_held_writer", {})
    if (
        payload.get("schema") != "aiflow-commercial-hwr-augmentation-writer-loo/v1"
        or payload.get("math_labels") != labels
        or len(states) < 2
    ):
        raise ValueError("parent writer-LOO checkpoint contract mismatch")
    expected = set(InkClassifierV1(len(labels), 0).state_dict())
    output = {}
    for writer, state in states.items():
        if set(state) != expected:
            raise ValueError(f"writer-LOO state keys differ: {writer}")
        output[str(writer)] = {
            key: value.detach().cpu().clone() for key, value in state.items()
        }
    return output


def _bank_batch(bank: DtwBank, schedule: np.ndarray, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """합성은행의 한 배치를 GPU 텐서와 정답 인덱스로 변환한다."""

    features = torch.as_tensor(np.asarray(bank.features[schedule], dtype=np.float32), device=device)
    labels = torch.as_tensor(np.asarray(bank.labels[schedule], dtype=np.int64), device=device)
    return features, labels


def _augment(
    features: torch.Tensor,
    variant: Variant,
    generator: torch.Generator,
) -> torch.Tensor:
    """사전 선언된 기하 변형 뒤 독립 물리 시뮬레이션을 좌표에 적용한다."""

    geometric = augment_online_tensor(features, variant.geometry, generator)
    return simulate_pen_physics(geometric, variant.physics, generator)


def _train(
    initial_state: dict[str, torch.Tensor],
    labels: list[str],
    variant: Variant,
    direct_features: np.ndarray,
    direct_labels: np.ndarray,
    direct_indices: np.ndarray,
    external_features: np.ndarray,
    external_labels: np.ndarray,
    direct_bank: DtwBank,
    external_bank: DtwBank,
    steps: int,
    direct_batch_size: int,
    external_batch_size: int,
    device: torch.device,
    seed: int,
) -> tuple[InkClassifierV1, dict]:
    """실제 표본 보존 손실과 합성 일관성 손실로 한 writer-LOO 모델을 갱신한다."""

    model = _new_model(initial_state, labels, device, variant.train_scope)
    teacher = _new_model(initial_state, labels, device, "head")
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    teacher.eval()
    parameters = [value for value in model.parameters() if value.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=variant.learning_rate, weight_decay=1.0e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    rng = np.random.default_rng(seed)
    direct_probability = _sample_probabilities(direct_labels, direct_indices, 1.0)
    external_indices = np.arange(len(external_labels), dtype=np.int64)
    external_probability = _sample_probabilities(external_labels, external_indices, 0.5)
    direct_schedule = rng.choice(
        direct_indices, size=steps * direct_batch_size, replace=True, p=direct_probability,
    ).reshape(steps, direct_batch_size)
    external_schedule = rng.choice(
        external_indices, size=steps * external_batch_size, replace=True, p=external_probability,
    ).reshape(steps, external_batch_size)
    direct_bank_schedule = rng.integers(
        0, len(direct_bank.labels), size=(steps, direct_batch_size), endpoint=False,
    )
    external_bank_schedule = rng.integers(
        0, len(external_bank.labels), size=(steps, external_batch_size), endpoint=False,
    )
    generator = torch.Generator(device=device).manual_seed(seed + 17)
    losses: list[float] = []
    started = time.perf_counter()

    for step in range(steps):
        direct = _batch(direct_features, direct_schedule[step], "uniform-time", device)
        external = _batch(external_features, external_schedule[step], "uniform-time", device)
        direct_target = torch.as_tensor(direct_labels[direct_schedule[step]], device=device)
        external_target = torch.as_tensor(external_labels[external_schedule[step]], device=device)
        if variant.use_dtw:
            direct_augmented, direct_augmented_target = _bank_batch(
                direct_bank, direct_bank_schedule[step], device,
            )
            external_augmented, external_augmented_target = _bank_batch(
                external_bank, external_bank_schedule[step], device,
            )
        else:
            direct_augmented, direct_augmented_target = direct, direct_target
            external_augmented, external_augmented_target = external, external_target
        direct_augmented = _augment(direct_augmented, variant, generator)
        external_augmented = _augment(external_augmented, variant, generator)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            direct_raw_logits = model(direct, "math")
            external_raw_logits = model(external, "math")
            direct_augmented_logits = model(direct_augmented, "math")
            external_augmented_logits = model(external_augmented, "math")
            with torch.no_grad():
                teacher_external_logits = teacher(external, "math")
                teacher_external_augmented_logits = teacher(external_augmented, "math")
            loss = F.cross_entropy(direct_raw_logits, direct_target)
            loss = loss + F.cross_entropy(external_raw_logits, external_target)
            loss = loss + 0.85 * F.cross_entropy(direct_augmented_logits, direct_augmented_target)
            loss = loss + 0.85 * F.cross_entropy(external_augmented_logits, external_augmented_target)
            if variant.top5_margin_weight:
                loss = loss + variant.top5_margin_weight * _top5_margin_loss(
                    direct_raw_logits, direct_target,
                )
                loss = loss + variant.top5_margin_weight * _top5_margin_loss(
                    direct_augmented_logits, direct_augmented_target,
                )
            loss = loss + 0.50 * F.kl_div(
                F.log_softmax(external_raw_logits, dim=1),
                F.softmax(teacher_external_logits, dim=1), reduction="batchmean",
            )
            loss = loss + 0.15 * F.kl_div(
                F.log_softmax(external_augmented_logits, dim=1),
                F.softmax(teacher_external_augmented_logits, dim=1), reduction="batchmean",
            )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step + 1}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach()))

    model.eval()
    return model, {
        "steps": steps,
        "direct_batch_size": direct_batch_size,
        "external_batch_size": external_batch_size,
        "learning_rate": variant.learning_rate,
        "train_scope": variant.train_scope,
        "use_dtw": variant.use_dtw,
        "physics": asdict(variant.physics),
        "trainable_parameters": sum(value.numel() for value in parameters),
        "mean_loss": float(np.mean(losses)),
        "final_loss": losses[-1],
        "seconds": time.perf_counter() - started,
    }


def _save_bank(path: Path, bank: DtwBank) -> None:
    """재현 가능한 DTW 합성 텐서와 라벨을 압축 NPZ로 기록한다."""

    np.savez_compressed(path, features=bank.features, labels=bank.labels)


def _self_test() -> None:
    """variant 계약과 증강 조합의 비공간 채널 보존을 검증한다."""

    features = torch.zeros((2, POINTS, len(CHANNELS)), dtype=torch.float32)
    features[:, :, 0] = torch.linspace(0.1, 0.9, POINTS)
    features[:, :, 1] = torch.linspace(0.9, 0.1, POINTS)
    features[:, :, 2] = 1.0 / (POINTS - 1)
    features[:, 0, 2] = 0.0
    features[:, 0, 3] = 1.0
    features[:, :, 4] = 1.0
    variant = VARIANTS["dtw_physics_light"]
    result = _augment(features, variant, torch.Generator().manual_seed(7))
    assert result.shape == features.shape
    assert torch.equal(result[:, :, 2:], features[:, :, 2:])
    assert not torch.equal(result[:, :, :2], features[:, :, :2])
    assert VARIANTS["control_head"].physics == NO_PHYSICS


def main() -> int:
    """CLI 입력을 검증하고 writer-LOO 탐색과 동결 최종 학습을 실행한다."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path, default=DEFAULT_CANONICAL)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--parent-checkpoint", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--parent-loo", type=Path, default=DEFAULT_PARENT_LOO)
    parser.add_argument("--direct-rows", type=Path, default=DEFAULT_DIRECT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--variants",
        default="control_head,continuation_geometry,physics_light,dtw_physics_light,dtw_physics_medium",
        help="comma-separated built-in variants; control_head is mandatory",
    )
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--direct-batch-size", type=int, default=24)
    parser.add_argument("--external-batch-size", type=int, default=48)
    parser.add_argument("--external-bank-size", type=int, default=3072)
    parser.add_argument("--direct-bank-size", type=int, default=384)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--search-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        print(json.dumps({"self_test": "pass"}))
        return 0
    positive = (
        args.steps, args.direct_batch_size, args.external_batch_size,
        args.external_bank_size, args.direct_bank_size, args.eval_batch_size,
    )
    if min(positive) < 1:
        parser.error("steps, batch sizes, and bank sizes must be positive")
    names = [value.strip() for value in args.variants.split(",") if value.strip()]
    if "control_head" not in names or len(names) != len(set(names)):
        parser.error("variants must be unique and include control_head")
    unknown = sorted(set(names) - set(VARIANTS))
    if unknown:
        parser.error(f"unknown variants: {unknown}")

    canonical = _d_path(args.canonical_root, "canonical root")
    cache_dir = _d_path(args.cache_dir, "training cache")
    parent_path = _d_path(args.parent_checkpoint, "parent commercial HWR checkpoint")
    parent_loo_path = _d_path(args.parent_loo, "parent writer-LOO checkpoint")
    direct_path = _d_path(args.direct_rows, "project-owned training derivative")
    output = _d_path(args.output, "output", must_exist=False)
    if output.exists():
        parser.error(f"refusing to overwrite output: {output}")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")

    _seed_everything(SEED)
    labels, available_auxiliary = load_vocabs(canonical)
    parent_state, parent_report = _load_parent(parent_path, labels)
    parent_loo = _load_parent_loo(parent_loo_path, labels)
    cache = prepare_cache(canonical, cache_dir, labels, available_auxiliary, "unified-math")
    math_train = _dataset(cache_dir, cache["sets"]["math_train"], "preserve")
    math_eval = _dataset(cache_dir, cache["sets"]["math_eval"], "preserve")
    direct_features, direct_target, writers, _truth = _direct_rows(
        canonical, {label: index for index, label in enumerate(labels)}, direct_path,
    )
    direct_labels = direct_target.numpy().astype(np.int64, copy=False)
    external_labels = np.asarray(math_train.labels, dtype=np.int64)
    folds = _writer_split_indices(writers)
    if set(parent_loo) != {writer for writer, _, _ in folds}:
        raise ValueError("parent writer-LOO coverage differs from the current direct rows")

    started = time.perf_counter()
    print(json.dumps({"event": "external_dtw_bank_start", "records": len(external_labels)}), flush=True)
    external_bank = build_dtw_bank(
        math_train.features, external_labels, args.external_bank_size, SEED + 100,
    )
    direct_banks: dict[str, DtwBank] = {}
    for fold_index, (writer, train_index, _held_index) in enumerate(folds):
        direct_banks[writer] = build_dtw_bank(
            direct_features, direct_labels, args.direct_bank_size,
            SEED + 200 + fold_index,
            allowed_indices=train_index.numpy(), writer_groups=writers,
        )
    print(json.dumps({
        "event": "dtw_banks_complete",
        "external": len(external_bank.labels),
        "direct_by_fold": {key: len(value.labels) for key, value in direct_banks.items()},
    }, ensure_ascii=False), flush=True)

    results: dict[str, dict] = {}
    states: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    for name in names:
        variant = VARIANTS[name]
        prediction_rows: list[dict] = []
        fold_rows = []
        states[name] = {}
        for fold_index, (writer, train_index, held_index) in enumerate(folds):
            seed = SEED + fold_index * 37
            initial_state = {
                key: value.clone() for key, value in parent_loo[writer].items()
            }
            if name == "control_head":
                model = _new_model(initial_state, labels, device, "head")
                model.eval()
                training = {
                    "steps": 0, "seconds": 0.0, "train_scope": "frozen_control",
                    "initialization": "previous commercial-rights writer-LOO challenger",
                }
            else:
                model, training = _train(
                    initial_state, labels, variant,
                    direct_features, direct_labels, train_index.numpy(),
                    math_train.features, external_labels,
                    direct_banks[writer], external_bank,
                    args.steps, args.direct_batch_size, args.external_batch_size,
                    device, seed,
                )
            held = held_index.numpy()
            logits = _logits(model, direct_features, held, device, args.eval_batch_size)
            prediction_rows.extend(_prediction_rows(
                logits, direct_labels[held], [writers[index] for index in held],
            ))
            fold_rows.append({
                "held_writer": writer,
                "held_records": len(held),
                "training": training,
                "direct_dtw_bank": direct_banks[writer].audit,
            })
            states[name][writer] = _clone_state(model)
            print(json.dumps({
                "event": "cleanroom_fold_complete", "variant": name,
                "fold": fold_index + 1, "folds": len(folds), "writer": writer,
                "held_records": len(held), "seconds": training["seconds"],
            }, ensure_ascii=False), flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        results[name] = {
            "variant": asdict(variant),
            "folds": fold_rows,
            "writer_disjoint": _metrics(prediction_rows, labels),
        }
        print(json.dumps({
            "event": "cleanroom_variant_complete", "variant": name,
            "top1": results[name]["writer_disjoint"]["top1"],
            "top5": results[name]["writer_disjoint"]["top5"],
            "ece": results[name]["writer_disjoint"]["ece_10"],
        }, ensure_ascii=False), flush=True)

    selection = _select(results)
    selected_name = selection["selected"]
    selected_variant = VARIANTS[selected_name]
    final_state = None
    final_training = None
    external_gate = None
    final_direct_bank = build_dtw_bank(
        direct_features, direct_labels, args.direct_bank_size, SEED + 8000,
        writer_groups=writers,
    )
    if not args.search_only:
        if selected_name == "control_head":
            final_model = _new_model(parent_state, labels, device, "head")
            final_model.eval()
            final_training = {
                "steps": 0, "seconds": 0.0, "train_scope": "frozen_control",
                "initialization": "previous frozen commercial-rights challenger",
            }
        else:
            final_model, final_training = _train(
                parent_state, labels, selected_variant,
                direct_features, direct_labels, np.arange(len(direct_labels), dtype=np.int64),
                math_train.features, external_labels,
                final_direct_bank, external_bank,
                args.steps, args.direct_batch_size, args.external_batch_size,
                device, SEED + 9000,
            )
        evaluation_indices = np.arange(len(math_eval.labels), dtype=np.int64)
        parent_model = _new_model(parent_state, labels, device, "head")
        parent_model.eval()
        parent_logits = _logits(
            parent_model, math_eval.features, evaluation_indices, device, args.eval_batch_size,
        )
        final_logits = _logits(
            final_model, math_eval.features, evaluation_indices, device, args.eval_batch_size,
        )
        evaluation_writers = ["technical_external"] * len(evaluation_indices)
        before = _metrics(_prediction_rows(
            parent_logits, np.asarray(math_eval.labels), evaluation_writers,
        ), labels)
        after = _metrics(_prediction_rows(
            final_logits, np.asarray(math_eval.labels), evaluation_writers,
        ), labels)
        external_gate = {
            "base": before,
            "candidate": after,
            "top1_regression": before["top1"] - after["top1"],
            "top5_regression": before["top5"] - after["top5"],
            "passed": (
                after["top1"] >= before["top1"] - 0.003
                and after["top5"] >= before["top5"] - 0.002
            ),
            "role": "fixed external technical non-regression only; not variant selection",
        }
        final_state = _clone_state(final_model)

    gradient_updates = (len(names) - 1) * len(folds) * args.steps
    if not args.search_only and selected_name != "control_head":
        gradient_updates += args.steps
    report = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "search_only" if args.search_only else "frozen_shadow_pending_untouched_acceptance",
        "architecture_unchanged": True,
        "architecture": {
            "input": [POINTS, len(CHANNELS)], "channels": list(CHANNELS),
            "classes": len(labels), "hidden": 128, "transformer_blocks": 4,
            "attention_heads": 4, "single_head": True,
        },
        "input_contract": input_contract("uniform-time"),
        "selection_policy": {
            "project_writer_disjoint": True,
            "writers": len(set(writers)),
            "records": len(direct_labels),
            "selection_used_fresh_acceptance": False,
            "selection_used_crohme": False,
            "selection_used_mathwriting": False,
            "external_rows_role": "class-balanced rehearsal and same-label DTW parents only",
        },
        "clean_room_contract": {
            "purpose": "independent rights-compliant physical simulation, not licence circumvention",
            "admitted_parent_sources": sorted(cache["sets"]["math_train"]["sources"]),
            "excluded_sources": ["BDSHWA: no character boundary", "noncommercial evaluation corpora"],
            "same_label_dtw_only": True,
            "same_stroke_count_dtw_only": True,
            "project_dtw_cross_writer_only": True,
            "near_parent_filter": "DTW-aligned RMS [0.035, 0.45] and synthetic-parent RMS >= 0.008",
            "physical_parameters": "predeclared generic spring, damping, inertia, tremor, drift, monotonic speed warp",
            "physical_parameters_fitted_from_external_evaluation": False,
            "evaluation_error_labels_used_for_targeting": False,
            "non_spatial_channels_unchanged": ["delta_t", "stroke_start", "observed"],
            "time_quality": "uniform delta_t; speed variation represented only by monotonic XY progression",
        },
        "dtw_banks": {
            "external": external_bank.audit,
            "project_final": final_direct_bank.audit,
        },
        "variants": results,
        "selection": selection,
        "final_training": final_training,
        "external_technical_nonregression": external_gate,
        "training_data_guard": zero_crohme_training_manifest(
            admitted_sources={
                **{key: int(value) for key, value in cache["sets"]["math_train"]["sources"].items()},
                "project_owned": len(direct_labels),
            },
            gradient_updates=gradient_updates,
        ),
        "inputs": {
            "canonical_root": str(canonical),
            "cache_manifest": str((cache_dir / "cache_manifest.json").resolve()),
            "cache_manifest_sha256": _sha256(cache_dir / "cache_manifest.json"),
            "parent_checkpoint": str(parent_path),
            "parent_checkpoint_sha256": _sha256(parent_path),
            "parent_writer_loo": str(parent_loo_path),
            "parent_writer_loo_sha256": _sha256(parent_loo_path),
            "project_owned_rows": str(direct_path),
            "project_owned_rows_sha256": _sha256(direct_path),
        },
        "parent_summary": {
            "selected_variant": parent_report.get("selection", {}).get("selected"),
            "architecture_unchanged": parent_report.get("architecture_unchanged"),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "frozen_candidate": not args.search_only,
        "product_adopted": False,
        "adoption_limit": (
            "fresh unseen-writer acceptance and final research evaluation remain pending; "
            "grouping/context/layout are outside this trainer"
        ),
    }

    output.mkdir(parents=True)
    external_bank_path = output / "external_same_label_dtw_bank.npz"
    direct_bank_path = output / "project_cross_writer_dtw_bank.npz"
    _save_bank(external_bank_path, external_bank)
    _save_bank(direct_bank_path, final_direct_bank)
    preview_path = output / "cleanroom_dtw_physics_preview.png"
    preview_config = selected_variant.physics if selected_variant.physics.enabled else LIGHT_PHYSICS
    save_preview(preview_path, external_bank.previews, labels, preview_config)
    report["generated_data"] = {
        "external_dtw_bank": {
            "path": str(external_bank_path.resolve()), "sha256": _sha256(external_bank_path),
        },
        "project_dtw_bank": {
            "path": str(direct_bank_path.resolve()), "sha256": _sha256(direct_bank_path),
        },
        "visual_audit": {
            "path": str(preview_path.resolve()), "sha256": _sha256(preview_path),
        },
    }
    checkpoint_path = None
    if not args.search_only:
        checkpoint_path = output / "commercial_hwr_cleanroom_physics_checkpoint.pt"
        torch.save({
            "schema": SCHEMA,
            "state_dict": final_state,
            "math_labels": labels,
            "auxiliary_labels": [],
            "report": report,
        }, checkpoint_path)
        report["checkpoint"] = {
            "path": str(checkpoint_path.resolve()),
            "sha256": _sha256(checkpoint_path),
            "selected_variant": selected_name,
        }
        torch.save({
            "schema": LOO_SCHEMA,
            "math_labels": labels,
            "selected_variant": selected_name,
            "parent_checkpoint_sha256": _sha256(parent_path),
            "states_by_held_writer": states[selected_name],
        }, output / "selected_writer_loo_models.pt")
    (output / "augmentation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    print(json.dumps({
        "event": "commercial_hwr_cleanroom_physics_complete",
        "output": str(output),
        "selected": selected_name,
        "augmentation_adopted": selection["augmentation_adopted"],
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
