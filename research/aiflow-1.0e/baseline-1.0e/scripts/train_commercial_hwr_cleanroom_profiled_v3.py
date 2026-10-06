#!/usr/bin/env python3
"""물질화된 clean-room 프로파일 데이터셋으로 상업 HWR v3 후보를 선택한다."""

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
from cleanroom_trajectory_profiles_v3 import (
    ProfiledBank,
    _sha256,
    _writer_fingerprint,
    load_profiled_bank,
)
from train_character_classifier_v1 import (
    InkClassifierV1,
    _dataset,
    input_contract,
    load_vocabs,
    prepare_cache,
)
from train_commercial_hwr_augmentation_v1 import (
    _batch,
    _clone_state,
    _logits,
    _metrics,
    _new_model,
    _prediction_rows,
    _sample_probabilities,
    _select,
    _top5_margin_loss,
)
from training_data_guard_v1 import assert_training_path_clean, zero_crohme_training_manifest


SCHEMA = "aiflow-commercial-hwr-cleanroom-profiled/v3"
LOO_SCHEMA = "aiflow-commercial-hwr-cleanroom-profiled-writer-loo/v3"
DATASET_SCHEMA = "aiflow-cleanroom-hwr-dataset/v3"
PARENT_SCHEMA = "aiflow-commercial-hwr-cleanroom-physics/v2"
PARENT_LOO_SCHEMA = "aiflow-commercial-hwr-cleanroom-physics-writer-loo/v2"
SEED = 20260823
DEFAULT_CANONICAL = ROOT / "datasets" / "normalized" / "v1"
DEFAULT_CACHE = (
    ROOT / "artifacts" / "unified_head_20260813" / "unified_math_8ep_full" / "cache"
)
DEFAULT_PARENT = (
    ROOT / "artifacts" / "commercial_hwr_cleanroom_physics_20260823_r1_shadow"
    / "commercial_hwr_cleanroom_physics_checkpoint.pt"
)
DEFAULT_PARENT_LOO = (
    ROOT / "artifacts" / "commercial_hwr_cleanroom_physics_20260823_r1_shadow"
    / "selected_writer_loo_models.pt"
)
DEFAULT_DATASET = ROOT / "artifacts" / "commercial_hwr_cleanroom_dataset_v3_20260823_r1"
DEFAULT_DIRECT = Path(
    r"D:\AIFlow-Workspace\PrivateData\math-ink-data-collector\derived"
    r"\hwr-head-calibration-20260819-r1\project_owned_ownership_eval_95.jsonl.gz"
)
DEFAULT_OUTPUT = ROOT / "artifacts" / "commercial_hwr_cleanroom_profiled_20260823_r1_shadow"


@dataclass(frozen=True)
class Variant:
    """물질화 합성 재생의 학습 범위·속도·손실 비중을 선언한다."""

    name: str
    train_scope: str
    learning_rate: float
    synthetic_weight: float
    top5_margin_weight: float
    step_multiplier: float


VARIANTS = {
    row.name: row for row in (
        Variant("control_head", "head", 0.0, 0.0, 0.0, 0.0),
        Variant("profiled_last_block", "last_block", 1.0e-5, 0.85, 0.25, 1.0),
        Variant("profiled_last_block_fast", "last_block", 2.0e-5, 0.80, 0.35, 1.0),
        Variant("profiled_last_block_long", "last_block", 8.0e-6, 0.90, 0.35, 1.5),
        Variant("profiled_head", "head", 2.0e-5, 0.85, 0.25, 1.0),
    )
}


def _d_path(path: Path, kind: str, *, must_exist: bool = True) -> Path:
    """모든 입출력을 D:로 제한하고 평가 전용 경로를 학습에서 차단한다."""

    resolved = assert_training_path_clean(path, kind)
    if resolved.drive.upper() != "D:":
        raise ValueError(f"{kind} must remain on D:: {resolved}")
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"missing {kind}: {resolved}")
    return resolved


def _seed_everything(seed: int) -> None:
    """CPU·GPU 난수를 고정해 variant와 writer-fold 비교를 재현 가능하게 한다."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_parent(path: Path, labels: list[str]) -> tuple[dict[str, torch.Tensor], dict]:
    """동결 v2 clean-room 체크포인트의 372-class 단일 헤드 계약을 검증한다."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    report = payload.get("report", {})
    if (
        payload.get("schema") != PARENT_SCHEMA
        or payload.get("math_labels") != labels
        or payload.get("auxiliary_labels") != []
        or report.get("input_contract", {}).get("observed_channel_mode") != "uniform-time"
    ):
        raise ValueError("parent clean-room HWR checkpoint contract mismatch")
    state = {
        key: value.detach().cpu().clone()
        for key, value in payload.get("state_dict", {}).items()
    }
    model = InkClassifierV1(len(labels), 0)
    model.load_state_dict(state, strict=True)
    return state, report


def _load_parent_loo(path: Path, labels: list[str]) -> dict[str, dict[str, torch.Tensor]]:
    """동결 v2 writer-LOO 모델 전체의 스키마·키·라벨 순서를 검증한다."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    states = payload.get("states_by_held_writer", {})
    if (
        payload.get("schema") != PARENT_LOO_SCHEMA
        or payload.get("math_labels") != labels
        or len(states) < 2
    ):
        raise ValueError("parent clean-room writer-LOO contract mismatch")
    expected = set(InkClassifierV1(len(labels), 0).state_dict())
    output = {}
    for writer, state in states.items():
        if set(state) != expected:
            raise ValueError(f"writer-LOO state keys differ: {writer}")
        output[str(writer)] = {
            key: value.detach().cpu().clone() for key, value in state.items()
        }
    return output


def _load_dataset_manifest(root: Path) -> dict:
    """v3 물질화 데이터셋의 clean-room·zero-CROHME 계약을 검증한다."""

    path = root / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    contract = payload.get("clean_room_contract", {})
    guard = payload.get("training_data_guard", {})
    if (
        payload.get("schema") != DATASET_SCHEMA
        or payload.get("training_performed") is not False
        or payload.get("selection_used_crohme") is not False
        or int(contract.get("crohme_rows", -1)) != 0
        or int(contract.get("crohme_statistics", -1)) != 0
        or not guard.get("passed")
    ):
        raise ValueError("clean-room dataset manifest contract mismatch")
    return payload


def _resolve_dataset_file(root: Path, relative: str) -> Path:
    """manifest 상대경로가 데이터셋 루트를 벗어나지 않는지 확인한다."""

    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"dataset file escapes its root: {relative}")
    return path


def _load_manifest_bank(root: Path, record: dict) -> ProfiledBank:
    """manifest에 고정된 NPZ 해시를 사용해 합성은행을 읽는다."""

    bank = record.get("bank", record)
    path = _resolve_dataset_file(root, str(bank["path"]))
    return load_profiled_bank(path, str(bank["sha256"]))


def _bank_batch(
    bank: ProfiledBank,
    schedule: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """물질화 합성행을 GPU 입력과 라벨 텐서로 변환한다."""

    features = torch.as_tensor(np.asarray(bank.features[schedule], dtype=np.float32), device=device)
    labels = torch.as_tensor(np.asarray(bank.labels[schedule], dtype=np.int64), device=device)
    return features, labels


def _train(
    initial_state: dict[str, torch.Tensor],
    labels: list[str],
    variant: Variant,
    direct_features: np.ndarray,
    direct_labels: np.ndarray,
    direct_indices: np.ndarray,
    external_features: np.ndarray,
    external_labels: np.ndarray,
    direct_bank: ProfiledBank,
    external_bank: ProfiledBank,
    base_steps: int,
    direct_batch_size: int,
    external_batch_size: int,
    device: torch.device,
    seed: int,
) -> tuple[InkClassifierV1, dict]:
    """실제 승인행 보존 손실과 고정 합성행 손실로 한 fold 모델을 갱신한다."""

    steps = max(1, int(round(base_steps * variant.step_multiplier)))
    model = _new_model(initial_state, labels, device, variant.train_scope)
    teacher = _new_model(initial_state, labels, device, "head")
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    teacher.eval()
    parameters = [value for value in model.parameters() if value.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=variant.learning_rate, weight_decay=1.0e-3,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    rng = np.random.default_rng(seed)
    direct_probability = _sample_probabilities(direct_labels, direct_indices, 1.0)
    external_indices = np.arange(len(external_labels), dtype=np.int64)
    external_probability = _sample_probabilities(external_labels, external_indices, 0.5)
    direct_schedule = rng.choice(
        direct_indices, size=steps * direct_batch_size,
        replace=True, p=direct_probability,
    ).reshape(steps, direct_batch_size)
    external_schedule = rng.choice(
        external_indices, size=steps * external_batch_size,
        replace=True, p=external_probability,
    ).reshape(steps, external_batch_size)
    direct_bank_schedule = rng.integers(
        0, len(direct_bank.labels), size=(steps, direct_batch_size), endpoint=False,
    )
    external_bank_schedule = rng.integers(
        0, len(external_bank.labels), size=(steps, external_batch_size), endpoint=False,
    )
    losses: list[float] = []
    started = time.perf_counter()
    for step in range(steps):
        direct = _batch(direct_features, direct_schedule[step], "uniform-time", device)
        external = _batch(external_features, external_schedule[step], "uniform-time", device)
        direct_target = torch.as_tensor(direct_labels[direct_schedule[step]], device=device)
        external_target = torch.as_tensor(external_labels[external_schedule[step]], device=device)
        direct_synthetic, direct_synthetic_target = _bank_batch(
            direct_bank, direct_bank_schedule[step], device,
        )
        external_synthetic, external_synthetic_target = _bank_batch(
            external_bank, external_bank_schedule[step], device,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            direct_logits = model(direct, "math")
            external_logits = model(external, "math")
            direct_synthetic_logits = model(direct_synthetic, "math")
            external_synthetic_logits = model(external_synthetic, "math")
            with torch.no_grad():
                teacher_external = teacher(external, "math")
                teacher_synthetic = teacher(external_synthetic, "math")
            loss = F.cross_entropy(direct_logits, direct_target)
            loss = loss + F.cross_entropy(external_logits, external_target)
            loss = loss + variant.synthetic_weight * F.cross_entropy(
                direct_synthetic_logits, direct_synthetic_target,
            )
            loss = loss + variant.synthetic_weight * F.cross_entropy(
                external_synthetic_logits, external_synthetic_target,
            )
            if variant.top5_margin_weight:
                loss = loss + variant.top5_margin_weight * _top5_margin_loss(
                    direct_logits, direct_target,
                )
                loss = loss + variant.top5_margin_weight * _top5_margin_loss(
                    direct_synthetic_logits, direct_synthetic_target,
                )
            loss = loss + 0.50 * F.kl_div(
                F.log_softmax(external_logits, dim=1),
                F.softmax(teacher_external, dim=1), reduction="batchmean",
            )
            loss = loss + 0.18 * F.kl_div(
                F.log_softmax(external_synthetic_logits, dim=1),
                F.softmax(teacher_synthetic, dim=1), reduction="batchmean",
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
        "synthetic_weight": variant.synthetic_weight,
        "top5_margin_weight": variant.top5_margin_weight,
        "trainable_parameters": sum(value.numel() for value in parameters),
        "mean_loss": float(np.mean(losses)),
        "final_loss": losses[-1],
        "seconds": time.perf_counter() - started,
    }


def _self_test() -> None:
    """variant 수와 기준선 이름, 합성 손실 활성화 계약을 검증한다."""

    assert VARIANTS["control_head"].learning_rate == 0.0
    assert VARIANTS["control_head"].synthetic_weight == 0.0
    assert all(
        row.synthetic_weight > 0.0 and row.step_multiplier > 0.0
        for name, row in VARIANTS.items() if name != "control_head"
    )


def main() -> int:
    """writer-LOO 후보 선택, 전체 재학습, 외부 기술 게이트와 동결 저장을 실행한다."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path, default=DEFAULT_CANONICAL)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--parent-checkpoint", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--parent-loo", type=Path, default=DEFAULT_PARENT_LOO)
    parser.add_argument("--cleanroom-dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--direct-rows", type=Path, default=DEFAULT_DIRECT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--variants",
        default="control_head,profiled_last_block,profiled_last_block_fast,profiled_head",
        help="comma-separated built-in variants; control_head is mandatory",
    )
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--direct-batch-size", type=int, default=24)
    parser.add_argument("--external-batch-size", type=int, default=48)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--search-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        print(json.dumps({"self_test": "pass", "schema": SCHEMA}))
        return 0
    if min(args.steps, args.direct_batch_size, args.external_batch_size, args.eval_batch_size) < 1:
        parser.error("steps and batch sizes must be positive")
    names = [value.strip() for value in args.variants.split(",") if value.strip()]
    if "control_head" not in names or len(names) != len(set(names)):
        parser.error("variants must be unique and include control_head")
    unknown = sorted(set(names) - set(VARIANTS))
    if unknown:
        parser.error(f"unknown variants: {unknown}")

    canonical = _d_path(args.canonical_root, "canonical root")
    cache_dir = _d_path(args.cache_dir, "training cache")
    parent_path = _d_path(args.parent_checkpoint, "parent clean-room checkpoint")
    parent_loo_path = _d_path(args.parent_loo, "parent clean-room writer-LOO")
    dataset_root = _d_path(args.cleanroom_dataset, "clean-room materialized dataset")
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
    manifest = _load_dataset_manifest(dataset_root)
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
        raise ValueError("parent writer-LOO coverage differs from project rows")
    fold_manifest = manifest["datasets"]["project_writer_loo"]
    expected_fold_hashes = {_writer_fingerprint(str(writer)) for writer, _, _ in folds}
    if set(fold_manifest) != expected_fold_hashes:
        raise ValueError("materialized writer-LOO dataset coverage differs from project rows")
    external_bank = _load_manifest_bank(dataset_root, manifest["datasets"]["external"])
    final_direct_bank = _load_manifest_bank(dataset_root, manifest["datasets"]["project_final"])
    direct_banks = {
        str(writer): _load_manifest_bank(
            dataset_root,
            fold_manifest[_writer_fingerprint(str(writer))]["bank"],
        )
        for writer, _, _ in folds
    }

    started = time.perf_counter()
    results: dict[str, dict] = {}
    states: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    for name in names:
        variant = VARIANTS[name]
        prediction_rows: list[dict] = []
        fold_rows = []
        states[name] = {}
        for fold_index, (writer, train_index, held_index) in enumerate(folds):
            seed = SEED + fold_index * 41
            initial_state = {key: value.clone() for key, value in parent_loo[writer].items()}
            if name == "control_head":
                model = _new_model(initial_state, labels, device, "head")
                model.eval()
                training = {
                    "steps": 0, "seconds": 0.0, "train_scope": "frozen_control",
                    "initialization": "v2 clean-room writer-LOO candidate",
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
            writer_hash = _writer_fingerprint(str(writer))
            fold_rows.append({
                "held_writer_fingerprint": writer_hash,
                "held_records": len(held),
                "profile_fit_held_rows": 0,
                "training": training,
                "materialized_bank_sha256": fold_manifest[writer_hash]["bank"]["sha256"],
            })
            states[name][writer] = _clone_state(model)
            print(json.dumps({
                "event": "profiled_fold_complete", "variant": name,
                "fold": fold_index + 1, "folds": len(folds),
                "held_writer": writer_hash, "held_records": len(held),
                "seconds": training["seconds"],
            }), flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        results[name] = {
            "variant": asdict(variant),
            "folds": fold_rows,
            "writer_disjoint": _metrics(prediction_rows, labels),
        }
        print(json.dumps({
            "event": "profiled_variant_complete", "variant": name,
            "top1": results[name]["writer_disjoint"]["top1"],
            "top5": results[name]["writer_disjoint"]["top5"],
            "ece": results[name]["writer_disjoint"]["ece_10"],
        }), flush=True)

    selection = _select(results)
    selected_name = selection["selected"]
    selected_variant = VARIANTS[selected_name]
    final_state = None
    final_training = None
    external_gate = None
    if not args.search_only:
        if selected_name == "control_head":
            final_model = _new_model(parent_state, labels, device, "head")
            final_model.eval()
            final_training = {
                "steps": 0, "seconds": 0.0, "train_scope": "frozen_control",
                "initialization": "v2 clean-room frozen candidate",
            }
        else:
            final_model, final_training = _train(
                parent_state, labels, selected_variant,
                direct_features, direct_labels,
                np.arange(len(direct_labels), dtype=np.int64),
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

    gradient_updates = sum(
        int(round(args.steps * VARIANTS[name].step_multiplier)) * len(folds)
        for name in names if name != "control_head"
    )
    if not args.search_only and selected_name != "control_head":
        gradient_updates += int(round(args.steps * selected_variant.step_multiplier))
    dataset_manifest_path = dataset_root / "manifest.json"
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
            "profile_fit_excludes_held_writer": True,
            "selection_used_fresh_acceptance": False,
            "selection_used_crohme": False,
            "selection_used_mathwriting": False,
            "external_evaluation_role": "fixed technical non-regression after selection only",
        },
        "clean_room_contract": {
            "training_sources": [
                "approved external real rows", "project-owned real rows",
                "materialized class-profile synthetic rows",
            ],
            "crohme_rows": 0,
            "crohme_statistics": 0,
            "crohme_error_labels": 0,
            "trajectory_reversal": False,
            "profile_features": manifest["clean_room_contract"]["profile_features"],
            "materialized_dataset_manifest_sha256": _sha256(dataset_manifest_path),
            "row_level_parent_provenance": True,
        },
        "variants": results,
        "selection": selection,
        "final_training": final_training,
        "external_technical_nonregression": external_gate,
        "training_data_guard": zero_crohme_training_manifest(
            admitted_sources={
                **{key: int(value) for key, value in cache["sets"]["math_train"]["sources"].items()},
                "project_owned": len(direct_labels),
                "cleanroom_external_synthetic": len(external_bank.labels),
                "cleanroom_project_synthetic": len(final_direct_bank.labels),
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
            "cleanroom_dataset": str(dataset_root),
            "cleanroom_dataset_manifest_sha256": _sha256(dataset_manifest_path),
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
            "fresh unseen-writer acceptance and one-time CROHME evaluation remain pending; "
            "no post-CROHME training is permitted"
        ),
    }

    output.mkdir(parents=True)
    checkpoint_path = None
    if not args.search_only:
        checkpoint_path = output / "commercial_hwr_cleanroom_profiled_checkpoint.pt"
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
            "dataset_manifest_sha256": _sha256(dataset_manifest_path),
            "states_by_held_writer": states[selected_name],
        }, output / "selected_writer_loo_models.pt")
    (output / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    print(json.dumps({
        "event": "commercial_hwr_cleanroom_profiled_complete",
        "output": str(output),
        "selected": selected_name,
        "augmentation_adopted": selection["augmentation_adopted"],
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "crohme_rows": 0,
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
