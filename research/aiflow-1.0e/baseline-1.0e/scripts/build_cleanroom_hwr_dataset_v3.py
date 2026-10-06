#!/usr/bin/env python3
"""상업권리 학습 분할로 재사용 가능한 clean-room HWR 증강 데이터셋을 만든다."""

from __future__ import annotations

from training_data_guard_v1 import assert_training_entrypoint_arguments_clean

assert_training_entrypoint_arguments_clean()

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import numpy as np

from calibrate_project_punctuation_v1 import _direct_rows, _writer_split_indices
from character_tensor_v1 import ROOT
from cleanroom_online_ink_augmentation_v2 import LIGHT_PHYSICS, NO_PHYSICS
from cleanroom_trajectory_profiles_v3 import (
    PROFILE_SCHEMA,
    SCHEMA as BANK_SCHEMA,
    ProfiledBank,
    _sha256,
    _writer_fingerprint,
    build_profiled_bank,
    fit_trajectory_profiles,
    save_profile_preview,
    save_profiled_bank,
    trajectory_descriptor,
)
from train_character_classifier_v1 import _dataset, load_vocabs, prepare_cache
from training_data_guard_v1 import assert_training_path_clean, zero_crohme_training_manifest


SCHEMA = "aiflow-cleanroom-hwr-dataset/v3"
SEED = 20260823
DEFAULT_CANONICAL = ROOT / "datasets" / "normalized" / "v1"
DEFAULT_CACHE = (
    ROOT / "artifacts" / "unified_head_20260813" / "unified_math_8ep_full" / "cache"
)
DEFAULT_DIRECT = Path(
    r"D:\AIFlow-Workspace\PrivateData\math-ink-data-collector\derived"
    r"\hwr-head-calibration-20260819-r1\project_owned_ownership_eval_95.jsonl.gz"
)
DEFAULT_OUTPUT = ROOT / "artifacts" / "commercial_hwr_cleanroom_dataset_v3_20260823_r1"


def _d_path(path: Path, kind: str, *, must_exist: bool = True) -> Path:
    """입출력을 D:로 고정하고 평가 데이터가 학습 경로에 들어오는 것을 차단한다."""

    resolved = assert_training_path_clean(path, kind)
    if resolved.drive.upper() != "D:":
        raise ValueError(f"{kind} must remain on D:: {resolved}")
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"missing {kind}: {resolved}")
    return resolved


def _write_json(path: Path, value: dict) -> None:
    """UTF-8과 LF를 고정해 감사 가능한 JSON 파일을 기록한다."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _relative_bank_record(record: dict, root: Path) -> dict:
    """staging 절대경로를 제거하고 최종 데이터셋 기준 상대경로만 남긴다."""

    return {
        "path": Path(record["path"]).relative_to(root).as_posix(),
        "sha256": record["sha256"],
        "metadata_path": Path(record["metadata_path"]).relative_to(root).as_posix(),
        "metadata_sha256": record["metadata_sha256"],
        "records": record["records"],
        "labels": record["labels"],
    }


def _special_profiles(profiles: dict) -> dict:
    """폐곡선 방향 검수에 중요한 숫자·영문 원형 클래스만 발췌한다."""

    classes = profiles.get("classes", {})
    return {
        label: classes[label]
        for label in ("0", "O", "o", r"\circ", r"\mathcal{O}")
        if label in classes
    }


def _validate_bank(bank: ProfiledBank, expected_cross_writer: bool) -> dict:
    """합성은행의 좌표·시간·방향·메타데이터 계약을 전체 행에서 검증한다."""

    features, labels = bank.features, bank.labels
    if features.shape != (len(labels), 128, 5) or len(bank.metadata) != len(labels):
        raise ValueError("generated clean-room bank shape or metadata coverage mismatch")
    if not np.isfinite(features).all():
        raise ValueError("generated clean-room bank contains non-finite values")
    if float(features[:, :, :2].min()) < 0.0 or float(features[:, :, :2].max()) > 1.0:
        raise ValueError("generated clean-room coordinates are outside [0,1]")
    if not np.allclose(features[:, 1:, 2], 1.0 / 127.0, atol=1.0e-7):
        raise ValueError("generated clean-room delta_t is not uniform")
    if not np.all(features[:, 0, 2] == 0.0) or not np.all(features[:, :, 4] == 1.0):
        raise ValueError("generated clean-room time or observed channels are invalid")
    mismatches = 0
    for row, metadata in zip(features, bank.metadata, strict=True):
        descriptor = trajectory_descriptor(row)
        topology = metadata["topology"]
        if (
            descriptor.stroke_count != int(topology["stroke_count"])
            or descriptor.loop_stroke != int(topology["loop_stroke"])
            or descriptor.loop_direction != int(topology["loop_direction"])
        ):
            mismatches += 1
        if bool(metadata["cross_writer"]) != expected_cross_writer:
            raise ValueError("generated clean-room cross-writer metadata mismatch")
        if expected_cross_writer:
            writers = metadata["parent_writer_fingerprints"][:2]
            if len(writers) != 2 or len(set(writers)) != 2:
                raise ValueError("project synthetic parents are not writer-disjoint")
    if mismatches:
        raise ValueError(f"generated clean-room topology mismatches: {mismatches}")
    return {
        "records": len(labels),
        "labels": len(set(labels.tolist())),
        "finite": True,
        "coordinates_in_unit_box": True,
        "uniform_time": True,
        "topology_mismatches": 0,
        "cross_writer_verified": expected_cross_writer,
    }


def _physics(name: str):
    """CLI 이름을 사전 선언된 물리 시뮬레이션 설정으로 변환한다."""

    return {"none": NO_PHYSICS, "light": LIGHT_PHYSICS}[name]


def _self_test() -> None:
    """경로 상대화와 특수 프로파일 발췌의 최소 계약을 검증한다."""

    profiles = {"classes": {"0": {"records": 3}, "x": {"records": 4}}}
    assert _special_profiles(profiles) == {"0": {"records": 3}}
    root = Path(r"D:\dataset")
    record = {
        "path": str(root / "a.npz"), "sha256": "a",
        "metadata_path": str(root / "a.metadata.jsonl.gz"),
        "metadata_sha256": "b", "records": 1, "labels": 1,
    }
    assert _relative_bank_record(record, root)["path"] == "a.npz"


def main() -> int:
    """프로파일 적합, fold별 합성, 무결성 기록을 원자적 staging 흐름으로 실행한다."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path, default=DEFAULT_CANONICAL)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--direct-rows", type=Path, default=DEFAULT_DIRECT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--external-size", type=int, default=4096)
    parser.add_argument("--project-size", type=int, default=512)
    parser.add_argument("--physics", choices=("none", "light"), default="light")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        print(json.dumps({"self_test": "pass", "schema": SCHEMA}))
        return 0
    if min(args.external_size, args.project_size) < 1:
        parser.error("external-size and project-size must be positive")

    canonical = _d_path(args.canonical_root, "canonical root")
    cache_dir = _d_path(args.cache_dir, "training cache")
    direct_path = _d_path(args.direct_rows, "project-owned training derivative")
    output = _d_path(args.output, "output", must_exist=False)
    staging = output.with_name(f"{output.name}.building-{os.getpid()}")
    if output.exists() or staging.exists():
        parser.error(f"refusing to overwrite output or staging path: {output}")

    labels, available_auxiliary = load_vocabs(canonical)
    cache = prepare_cache(canonical, cache_dir, labels, available_auxiliary, "unified-math")
    math_train = _dataset(cache_dir, cache["sets"]["math_train"], "preserve")
    direct_features, direct_target, writers, _truth = _direct_rows(
        canonical, {label: index for index, label in enumerate(labels)}, direct_path,
    )
    external_labels = np.asarray(math_train.labels, dtype=np.int64)
    direct_labels = direct_target.numpy().astype(np.int64, copy=False)
    folds = _writer_split_indices(writers)
    physics = _physics(args.physics)
    staging.mkdir(parents=True)

    print(json.dumps({
        "event": "external_profile_start", "records": len(external_labels),
    }), flush=True)
    external_catalog = fit_trajectory_profiles(
        math_train.features, external_labels, labels,
        source_role="approved_external_training_split_only",
    )
    external_bank = build_profiled_bank(
        math_train.features, external_labels, external_catalog,
        args.external_size, SEED + 100, physics=physics,
    )
    external_validation = _validate_bank(external_bank, False)
    external_record = _relative_bank_record(
        save_profiled_bank(staging / "external_profiled_augmented.npz", external_bank),
        staging,
    )
    _write_json(staging / "profiles" / "external_training_profiles.json", external_catalog.profiles)
    print(json.dumps({
        "event": "external_profile_complete", "generated": len(external_bank.labels),
        "labels": len(set(external_bank.labels.tolist())),
    }), flush=True)

    project_catalog = fit_trajectory_profiles(
        direct_features, direct_labels, labels,
        source_role="project_owned_all_writers_for_final_fit_only",
    )
    project_bank = build_profiled_bank(
        direct_features, direct_labels, project_catalog,
        args.project_size, SEED + 200, writer_groups=writers, physics=physics,
    )
    project_validation = _validate_bank(project_bank, True)
    project_record = _relative_bank_record(
        save_profiled_bank(staging / "project_profiled_augmented_final.npz", project_bank),
        staging,
    )
    _write_json(staging / "profiles" / "project_final_profiles.json", project_catalog.profiles)

    fold_records = {}
    for fold_index, (writer, train_index, held_index) in enumerate(folds):
        writer_hash = _writer_fingerprint(str(writer))
        catalog = fit_trajectory_profiles(
            direct_features, direct_labels, labels,
            allowed_indices=train_index.numpy(),
            source_role=f"project_owned_writer_loo_train:{writer_hash}",
        )
        bank = build_profiled_bank(
            direct_features, direct_labels, catalog, args.project_size,
            SEED + 1000 + fold_index,
            writer_groups=writers, physics=physics,
        )
        validation = _validate_bank(bank, True)
        bank_path = staging / "project_writer_loo" / f"held_{writer_hash}.npz"
        bank_record = _relative_bank_record(save_profiled_bank(bank_path, bank), staging)
        profile_path = staging / "profiles" / f"project_loo_held_{writer_hash}.json"
        _write_json(profile_path, catalog.profiles)
        fold_records[writer_hash] = {
            "held_writer_fingerprint": writer_hash,
            "held_records": len(held_index),
            "profile_fit_records": len(train_index),
            "held_rows_in_profile": 0,
            "bank": bank_record,
            "bank_audit": bank.audit,
            "validation": validation,
            "profile_path": profile_path.relative_to(staging).as_posix(),
            "profile_sha256": _sha256(profile_path),
        }
        print(json.dumps({
            "event": "project_fold_dataset_complete", "fold": fold_index + 1,
            "folds": len(folds), "held_writer": writer_hash,
            "generated": len(bank.labels),
        }), flush=True)

    preview_path = staging / "cleanroom_profiled_preview.png"
    save_profile_preview(preview_path, external_bank.previews, labels)
    cache_manifest_path = cache_dir / "cache_manifest.json"
    manifest = {
        "schema": SCHEMA,
        "bank_schema": BANK_SCHEMA,
        "profile_schema": PROFILE_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "materialized_cleanroom_training_dataset",
        "training_performed": False,
        "selection_used_crohme": False,
        "selection_used_fresh_acceptance": False,
        "clean_room_contract": {
            "purpose": "rights-cleared class-conditional online-ink augmentation",
            "admitted_external_sources": dict(sorted(cache["sets"]["math_train"]["sources"].items())),
            "admitted_project_owned_rows": len(direct_labels),
            "excluded_sources": [
                "CROHME: noncommercial evaluation only",
                "BDSHWA: no character boundary",
                "external technical evaluation split",
                "fresh unseen-writer acceptance",
            ],
            "profile_features": [
                "stroke_count", "closed_loop", "loop_direction", "loop_start_phase",
                "aspect_ratio", "path_length", "closure", "turning",
            ],
            "direction_policy": (
                "sample only same-direction approved parents; never reverse a trajectory; "
                "start phase comes from an approved same-class style parent"
            ),
            "time_quality": "uniform delta_t; physical speed is represented by monotonic XY progression",
            "row_provenance": "hashed parents plus deterministic transform metadata",
            "crohme_rows": 0,
            "crohme_statistics": 0,
        },
        "inputs": {
            "canonical_root": str(canonical),
            "cache_manifest": str(cache_manifest_path.resolve()),
            "cache_manifest_sha256": _sha256(cache_manifest_path),
            "project_owned_rows": str(direct_path),
            "project_owned_rows_sha256": _sha256(direct_path),
        },
        "profiles": {
            "external_path": "profiles/external_training_profiles.json",
            "external_sha256": _sha256(staging / "profiles" / "external_training_profiles.json"),
            "project_final_path": "profiles/project_final_profiles.json",
            "project_final_sha256": _sha256(staging / "profiles" / "project_final_profiles.json"),
            "external_special_loop_classes": _special_profiles(external_catalog.profiles),
            "project_special_loop_classes": _special_profiles(project_catalog.profiles),
        },
        "datasets": {
            "external": {
                "bank": external_record,
                "bank_audit": external_bank.audit,
                "validation": external_validation,
            },
            "project_final": {
                "bank": project_record,
                "bank_audit": project_bank.audit,
                "validation": project_validation,
            },
            "project_writer_loo": fold_records,
        },
        "visual_audit": {
            "path": preview_path.relative_to(staging).as_posix(),
            "sha256": _sha256(preview_path),
        },
        "training_data_guard": zero_crohme_training_manifest(
            admitted_sources={
                **{key: int(value) for key, value in cache["sets"]["math_train"]["sources"].items()},
                "project_owned": len(direct_labels),
                "cleanroom_external_synthetic": len(external_bank.labels),
                "cleanroom_project_synthetic_final": len(project_bank.labels),
            },
            gradient_updates=0,
        ),
    }
    _write_json(staging / "manifest.json", manifest)
    staging.rename(output)
    print(json.dumps({
        "event": "cleanroom_hwr_dataset_complete",
        "output": str(output),
        "manifest_sha256": _sha256(output / "manifest.json"),
        "external_records": len(external_bank.labels),
        "project_final_records": len(project_bank.labels),
        "writer_loo_folds": len(fold_records),
        "crohme_rows": 0,
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
