#!/usr/bin/env python3
"""상업권리 온라인 잉크만으로 클래스별 필기동역학 증강 데이터를 만든다."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from cleanroom_online_ink_augmentation_v2 import (
    CHANNELS,
    LIGHT_PHYSICS,
    NO_PHYSICS,
    POINTS,
    PhysicalConfig,
    _letterbox_numpy,
    blend_dtw_pair,
    simulate_pen_physics,
    stroke_bounds,
)


SCHEMA = "aiflow-cleanroom-profiled-augmentation/v3"
PROFILE_SCHEMA = "aiflow-cleanroom-trajectory-profiles/v3"


@dataclass(frozen=True)
class TrajectoryDescriptor:
    """한 온라인 궤적의 획 구조와 폐곡선 필기동역학을 요약한다."""

    stroke_count: int
    aspect_ratio: float
    total_length: float
    loop_stroke: int
    loop_direction: int
    loop_area: float
    closure_ratio: float
    start_phase: float
    initial_angle: float
    final_angle: float
    signed_turn: float
    absolute_turn: float


@dataclass(frozen=True)
class TrajectoryCatalog:
    """학습 분할에서만 계산한 행별 기술자 배열과 클래스 프로파일이다."""

    allowed_indices: np.ndarray
    stroke_count: np.ndarray
    loop_stroke: np.ndarray
    loop_direction: np.ndarray
    aspect_ratio: np.ndarray
    start_phase: np.ndarray
    descriptors: tuple[TrajectoryDescriptor, ...]
    profiles: dict


@dataclass(frozen=True)
class ProfiledBank:
    """물질화된 clean-room 합성 텐서와 재현 메타데이터를 묶는다."""

    features: np.ndarray
    labels: np.ndarray
    audit: dict
    metadata: tuple[dict, ...]
    previews: tuple[dict, ...]


def _wrapped_angle_delta(values: np.ndarray) -> np.ndarray:
    """연속 접선 각도 차이를 -pi에서 pi 범위로 접는다."""

    return (values + math.pi) % math.tau - math.pi


def _angle(points: np.ndarray, first: bool) -> float:
    """획 앞쪽 또는 뒤쪽의 안정된 단위 접선 각도를 계산한다."""

    if len(points) < 2:
        return 0.0
    count = min(5, len(points) - 1)
    vector = (
        points[count] - points[0]
        if first else points[-1] - points[-1 - count]
    )
    if float(np.linalg.norm(vector)) < 1.0e-8:
        return 0.0
    return float(math.atan2(float(vector[1]), float(vector[0])))


def _stroke_geometry(points: np.ndarray) -> dict:
    """한 획의 길이·폐합도·방향·곡률을 좌표계 독립 비율로 계산한다."""

    if len(points) < 2:
        return {
            "length": 0.0, "closure_ratio": float("inf"), "area": 0.0,
            "direction": 0, "start_phase": float("nan"),
            "initial_angle": 0.0, "final_angle": 0.0,
            "signed_turn": 0.0, "absolute_turn": 0.0, "closed": False,
        }
    delta = np.diff(points, axis=0)
    lengths = np.linalg.norm(delta, axis=1)
    path_length = float(lengths.sum())
    width, height = np.ptp(points, axis=0)
    extent = max(float(width), float(height), 1.0e-6)
    closure = float(np.linalg.norm(points[-1] - points[0]) / extent)
    x, y = points[:, 0], points[:, 1]
    area = float(0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))
    normalized_area = area / (extent * extent)
    valid_delta = delta[lengths > 1.0e-8]
    if len(valid_delta) >= 2:
        tangent_angles = np.arctan2(valid_delta[:, 1], valid_delta[:, 0])
        turns = _wrapped_angle_delta(np.diff(tangent_angles))
        signed_turn = float(turns.sum())
        absolute_turn = float(np.abs(turns).sum())
    else:
        signed_turn = absolute_turn = 0.0
    closed = (
        len(points) >= 8
        and path_length >= 1.15 * extent
        and closure <= 0.32
        and abs(normalized_area) >= 0.055
    )
    center = (points.min(axis=0) + points.max(axis=0)) * 0.5
    start_vector = points[0] - center
    phase = (
        float(math.atan2(float(start_vector[1]), float(start_vector[0])) % math.tau)
        if closed and float(np.linalg.norm(start_vector)) > 1.0e-8
        else float("nan")
    )
    return {
        "length": path_length,
        "closure_ratio": closure,
        "area": normalized_area,
        "direction": (1 if normalized_area > 0.0 else -1) if closed else 0,
        "start_phase": phase,
        "initial_angle": _angle(points, True),
        "final_angle": _angle(points, False),
        "signed_turn": signed_turn,
        "absolute_turn": absolute_turn,
        "closed": closed,
    }


def trajectory_descriptor(features: np.ndarray) -> TrajectoryDescriptor:
    """128x5 온라인 텐서에서 대표 폐곡선과 전체 형태 기술자를 계산한다."""

    values = np.asarray(features, dtype=np.float64)
    if values.shape != (POINTS, CHANNELS) or not np.isfinite(values).all():
        raise ValueError("trajectory descriptor expects a finite 128x5 tensor")
    bounds = stroke_bounds(values)
    xy = values[:, :2]
    width, height = np.ptp(xy, axis=0)
    aspect = float(max(width, 1.0e-6) / max(height, 1.0e-6))
    geometries = [_stroke_geometry(xy[start:end]) for start, end in bounds]
    closed = [
        (index, row) for index, row in enumerate(geometries) if bool(row["closed"])
    ]
    if closed:
        loop_stroke, primary = max(closed, key=lambda row: abs(float(row[1]["area"])))
    else:
        loop_stroke, primary = -1, {
            "direction": 0, "area": 0.0, "closure_ratio": float("nan"),
            "start_phase": float("nan"), "initial_angle": 0.0,
            "final_angle": 0.0, "signed_turn": 0.0, "absolute_turn": 0.0,
        }
    return TrajectoryDescriptor(
        stroke_count=len(bounds),
        aspect_ratio=aspect,
        total_length=float(sum(float(row["length"]) for row in geometries)),
        loop_stroke=loop_stroke,
        loop_direction=int(primary["direction"]),
        loop_area=float(primary["area"]),
        closure_ratio=float(primary["closure_ratio"]),
        start_phase=float(primary["start_phase"]),
        initial_angle=float(primary["initial_angle"]),
        final_angle=float(primary["final_angle"]),
        signed_turn=float(primary["signed_turn"]),
        absolute_turn=float(primary["absolute_turn"]),
    )


def _finite_summary(values: Sequence[float]) -> dict:
    """유한한 값만 남겨 강건 분위수 요약을 JSON 숫자로 반환한다."""

    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if not len(finite):
        return {"records": 0}
    return {
        "records": len(finite),
        "q10": float(np.quantile(finite, 0.10)),
        "median": float(np.median(finite)),
        "q90": float(np.quantile(finite, 0.90)),
    }


def _circular_summary(values: Sequence[float]) -> dict:
    """폐곡선 시작 각도의 원형 평균과 집중도를 계산한다."""

    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if not len(finite):
        return {"records": 0}
    vector = np.exp(1j * finite).mean()
    return {
        "records": len(finite),
        "mean_radians": float(np.angle(vector) % math.tau),
        "concentration": float(abs(vector)),
    }


def fit_trajectory_profiles(
    features: np.ndarray,
    labels: np.ndarray,
    label_names: Sequence[str],
    *,
    allowed_indices: np.ndarray | None = None,
    source_role: str,
) -> TrajectoryCatalog:
    """지정된 학습 행에서만 행별 기술자와 클래스별 분포를 적합한다."""

    targets = np.asarray(labels, dtype=np.int64)
    allowed = (
        np.arange(len(targets), dtype=np.int64)
        if allowed_indices is None else np.asarray(allowed_indices, dtype=np.int64)
    )
    if len(features) != len(targets) or not len(allowed):
        raise ValueError("profile fitting requires non-empty aligned features and labels")
    if allowed.min() < 0 or allowed.max() >= len(targets):
        raise ValueError("profile fitting indices are outside the source arrays")
    if targets.min() < 0 or targets.max() >= len(label_names):
        raise ValueError("profile labels are outside the vocabulary")

    stroke_count = np.full(len(targets), -1, dtype=np.int16)
    loop_stroke = np.full(len(targets), -2, dtype=np.int16)
    loop_direction = np.full(len(targets), 0, dtype=np.int8)
    aspect_ratio = np.full(len(targets), np.nan, dtype=np.float32)
    start_phase = np.full(len(targets), np.nan, dtype=np.float32)
    descriptor_rows: list[TrajectoryDescriptor] = []
    by_label: dict[int, list[TrajectoryDescriptor]] = defaultdict(list)
    for index in allowed.tolist():
        descriptor = trajectory_descriptor(np.asarray(features[index]))
        stroke_count[index] = descriptor.stroke_count
        loop_stroke[index] = descriptor.loop_stroke
        loop_direction[index] = descriptor.loop_direction
        aspect_ratio[index] = descriptor.aspect_ratio
        start_phase[index] = descriptor.start_phase
        descriptor_rows.append(descriptor)
        by_label[int(targets[index])].append(descriptor)

    profiles = {}
    for label_index, rows in sorted(by_label.items()):
        directions = Counter(row.loop_direction for row in rows if row.loop_direction)
        loops = [row for row in rows if row.loop_direction]
        profiles[str(label_names[label_index])] = {
            "label_index": label_index,
            "records": len(rows),
            "stroke_counts": dict(sorted(Counter(row.stroke_count for row in rows).items())),
            "loop_records": len(loops),
            "loop_direction": {
                "clockwise_y_up": int(directions.get(-1, 0)),
                "counterclockwise_y_up": int(directions.get(1, 0)),
            },
            "aspect_ratio": _finite_summary([row.aspect_ratio for row in rows]),
            "total_length": _finite_summary([row.total_length for row in rows]),
            "closure_ratio": _finite_summary([row.closure_ratio for row in loops]),
            "absolute_turn": _finite_summary([row.absolute_turn for row in rows]),
            "start_phase": _circular_summary([row.start_phase for row in loops]),
        }
    profile_payload = {
        "schema": PROFILE_SCHEMA,
        "source_role": source_role,
        "records": len(allowed),
        "labels_present": len(by_label),
        "evaluation_rows": 0,
        "crohme_rows": 0,
        "direction_convention": "signed polygon area in normalized y-up coordinates",
        "classes": profiles,
    }
    return TrajectoryCatalog(
        allowed_indices=allowed,
        stroke_count=stroke_count,
        loop_stroke=loop_stroke,
        loop_direction=loop_direction,
        aspect_ratio=aspect_ratio,
        start_phase=start_phase,
        descriptors=tuple(descriptor_rows),
        profiles=profile_payload,
    )


def _profile_groups(
    labels: np.ndarray,
    catalog: TrajectoryCatalog,
    writer_groups: Sequence[str] | None,
) -> dict[tuple[int, int, int, int], np.ndarray]:
    """라벨·획 수·대표 폐곡선 위치·방향이 같은 부모를 묶는다."""

    groups: dict[tuple[int, int, int, int], list[int]] = defaultdict(list)
    for index in catalog.allowed_indices.tolist():
        key = (
            int(labels[index]), int(catalog.stroke_count[index]),
            int(catalog.loop_stroke[index]), int(catalog.loop_direction[index]),
        )
        groups[key].append(index)
    output = {}
    for key, values in groups.items():
        if len(values) < 2:
            continue
        if writer_groups is not None and len({str(writer_groups[index]) for index in values}) < 2:
            continue
        output[key] = np.asarray(values, dtype=np.int64)
    return output


def _canonical_parent(features: np.ndarray) -> np.ndarray:
    """부모 텐서의 좌표·획 경계를 유지하고 시간품질만 uniform-time으로 고정한다."""

    output = np.asarray(features, dtype=np.float32).copy()
    output[:, 2] = 1.0 / (POINTS - 1)
    output[0, 2] = 0.0
    output[:, 4] = 1.0
    return output


def _fingerprint(features: np.ndarray, label: int) -> str:
    """원시 식별자를 노출하지 않는 좌표·라벨 기반 부모 지문을 만든다."""

    payload = np.asarray(np.round(features[:, :2], 5), dtype=np.float32).tobytes()
    payload += np.asarray([label], dtype=np.int64).tobytes()
    return hashlib.sha256(payload).hexdigest()


def _writer_fingerprint(writer: str) -> str:
    """writer 문자열을 내보내지 않고 분리 검증용 짧은 지문으로 바꾼다."""

    return hashlib.sha256(f"aiflow-cleanroom-v3:{writer}".encode("utf-8")).hexdigest()[:20]


def _retarget_aspect(features: np.ndarray, target: float, strength: float) -> tuple[np.ndarray, float]:
    """실제 승인 표본의 종횡비 쪽으로 좌표를 제한적으로 이동한다."""

    output = np.asarray(features, dtype=np.float32).copy()
    xy = np.asarray(output[:, :2], dtype=np.float64)
    width, height = np.ptp(xy, axis=0)
    current = float(max(width, 1.0e-6) / max(height, 1.0e-6))
    target = float(np.clip(target, 0.16, 6.25))
    desired = math.exp((1.0 - strength) * math.log(current) + strength * math.log(target))
    scale = math.sqrt(desired / current)
    center = (xy.min(axis=0, keepdims=True) + xy.max(axis=0, keepdims=True)) * 0.5
    adjusted = xy - center
    adjusted[:, 0] *= scale
    adjusted[:, 1] /= scale
    output[:, :2] = _letterbox_numpy(adjusted + center).astype(np.float32)
    return output, desired


def _retarget_loop_start(
    features: np.ndarray,
    loop_stroke: int,
    target_phase: float,
) -> tuple[np.ndarray, int]:
    """폐곡선 진행 방향은 보존하며 실제 관측 시작 각도에 가장 가까운 점으로 순환한다."""

    output = np.asarray(features, dtype=np.float32).copy()
    if loop_stroke < 0 or not np.isfinite(target_phase):
        return output, 0
    bounds = stroke_bounds(output)
    if loop_stroke >= len(bounds):
        raise ValueError("profile loop stroke is outside the trajectory topology")
    start, end = bounds[loop_stroke]
    points = np.asarray(output[start:end, :2], dtype=np.float64)
    if len(points) < 8:
        return output, 0
    center = (points.min(axis=0) + points.max(axis=0)) * 0.5
    angles = np.mod(np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0]), math.tau)
    distance = np.abs(_wrapped_angle_delta(angles - target_phase))
    shift = int(np.argmin(distance))
    if shift:
        output[start:end, :2] = np.roll(output[start:end, :2], -shift, axis=0)
    return output, shift


def _profile_transform(
    mixed: np.ndarray,
    style: np.ndarray,
    key: tuple[int, int, int, int],
    strength: float,
) -> tuple[np.ndarray, dict]:
    """동일 클래스 승인 표본의 비율과 폐곡선 시작 위치를 합성 궤적에 이식한다."""

    style_descriptor = trajectory_descriptor(style)
    output, desired_aspect = _retarget_aspect(mixed, style_descriptor.aspect_ratio, strength)
    output, phase_shift = _retarget_loop_start(
        output, key[2], style_descriptor.start_phase,
    )
    descriptor = trajectory_descriptor(output)
    if (
        descriptor.stroke_count != key[1]
        or descriptor.loop_stroke != key[2]
        or descriptor.loop_direction != key[3]
    ):
        raise ValueError("profile transform changed stroke or loop topology")
    if not np.array_equal(output[:, 2:], mixed[:, 2:]):
        raise AssertionError("profile transform changed non-spatial channels")
    return output, {
        "strength": float(strength),
        "target_aspect_ratio": style_descriptor.aspect_ratio,
        "realized_aspect_ratio": desired_aspect,
        "target_start_phase": (
            style_descriptor.start_phase if np.isfinite(style_descriptor.start_phase) else None
        ),
        "phase_shift_points": phase_shift,
        "loop_direction": key[3],
    }


def _apply_physics(
    values: np.ndarray,
    expected: Sequence[tuple[int, int, int]],
    config: PhysicalConfig,
    seed: int,
) -> tuple[np.ndarray, list[bool], Counter[str]]:
    """합성은행 전체에 결정적 물리 시뮬레이션을 적용하고 구조 변화 행은 원복한다."""

    output = np.asarray(values, dtype=np.float32).copy()
    applied = [False] * len(output)
    rejections: Counter[str] = Counter()
    if not config.enabled:
        return output, applied, rejections
    generator = torch.Generator().manual_seed(seed)
    for start in range(0, len(output), 256):
        end = min(start + 256, len(output))
        batch = torch.from_numpy(output[start:end])
        candidate = simulate_pen_physics(batch, config, generator).numpy()
        for offset, row in enumerate(candidate):
            index = start + offset
            descriptor = trajectory_descriptor(row)
            if (
                descriptor.stroke_count,
                descriptor.loop_stroke,
                descriptor.loop_direction,
            ) != expected[index]:
                rejections["physics_topology_reverted"] += 1
                continue
            output[index] = row
            applied[index] = True
    return output, applied, rejections


def build_profiled_bank(
    features: np.ndarray,
    labels: np.ndarray,
    catalog: TrajectoryCatalog,
    size: int,
    seed: int,
    *,
    writer_groups: Sequence[str] | None = None,
    physics: PhysicalConfig = LIGHT_PHYSICS,
    preview_count: int = 12,
) -> ProfiledBank:
    """학습 분할의 동일 라벨·동일 동역학 부모만으로 물질화 합성은행을 생성한다."""

    targets = np.asarray(labels, dtype=np.int64)
    if len(features) != len(targets) or size < 1:
        raise ValueError("profiled bank requires aligned non-empty input")
    if writer_groups is not None and len(writer_groups) != len(targets):
        raise ValueError("writer group count does not match profiled parents")
    groups = _profile_groups(targets, catalog, writer_groups)
    by_label: dict[int, list[tuple[int, int, int, int]]] = defaultdict(list)
    for key in groups:
        by_label[key[0]].append(key)
    eligible_labels = sorted(by_label)
    if not eligible_labels:
        raise ValueError("no clean-room trajectory profile group can create a pair")

    rng = np.random.default_rng(seed)
    label_cycle = np.asarray(eligible_labels, dtype=np.int64)
    rng.shuffle(label_cycle)
    values: list[np.ndarray] = []
    output_labels: list[int] = []
    metadata: list[dict] = []
    previews: list[dict] = []
    fingerprints: set[str] = set()
    parent_fingerprints: set[str] = set()
    rejections: Counter[str] = Counter()
    expected_topology: list[tuple[int, int, int]] = []
    attempts = 0
    max_attempts = max(size * 120, 2000)
    while len(values) < size and attempts < max_attempts:
        label = int(label_cycle[attempts % len(label_cycle)])
        key = by_label[label][int(rng.integers(0, len(by_label[label])))]
        candidates = groups[key]
        parent_indices = rng.choice(candidates, size=2, replace=False).tolist()
        anchor_index, partner_index = int(parent_indices[0]), int(parent_indices[1])
        attempts += 1
        if (
            writer_groups is not None
            and str(writer_groups[anchor_index]) == str(writer_groups[partner_index])
        ):
            rejections["same_writer"] += 1
            continue
        style_index = int(candidates[int(rng.integers(0, len(candidates)))])
        anchor = _canonical_parent(np.asarray(features[anchor_index]))
        partner = _canonical_parent(np.asarray(features[partner_index]))
        style = _canonical_parent(np.asarray(features[style_index]))
        try:
            mixed, blend_audit = blend_dtw_pair(
                anchor, partner, float(rng.uniform(0.28, 0.72)),
            )
            profiled, transform = _profile_transform(
                mixed, style, key, float(rng.uniform(0.25, 0.55)),
            )
        except ValueError as error:
            rejections[str(error)] += 1
            continue
        synthetic_fingerprint = _fingerprint(profiled, label)
        if synthetic_fingerprint in fingerprints:
            rejections["duplicate_synthetic"] += 1
            continue
        fingerprints.add(synthetic_fingerprint)
        anchor_fingerprint = _fingerprint(anchor, label)
        partner_fingerprint = _fingerprint(partner, label)
        style_fingerprint = _fingerprint(style, label)
        parent_fingerprints.update((anchor_fingerprint, partner_fingerprint, style_fingerprint))
        values.append(profiled)
        output_labels.append(label)
        expected_topology.append((key[1], key[2], key[3]))
        row = {
            "synthetic_id": synthetic_fingerprint,
            "label_index": label,
            "topology": {
                "stroke_count": key[1], "loop_stroke": key[2],
                "loop_direction": key[3],
            },
            "parents": {
                "anchor": anchor_fingerprint,
                "partner": partner_fingerprint,
                "style": style_fingerprint,
            },
            "cross_writer": writer_groups is not None,
            "parent_writer_fingerprints": (
                [
                    _writer_fingerprint(str(writer_groups[anchor_index])),
                    _writer_fingerprint(str(writer_groups[partner_index])),
                    _writer_fingerprint(str(writer_groups[style_index])),
                ]
                if writer_groups is not None else []
            ),
            "dtw": blend_audit,
            "profile_transform": transform,
            "physics": {"name": physics.name, "applied": False},
        }
        metadata.append(row)
        if len(previews) < preview_count:
            previews.append({
                "label": label, "anchor": anchor, "mixed": mixed,
                "profiled": profiled,
            })

    if len(values) < size:
        raise ValueError(
            f"profiled bank generated {len(values)} of {size} rows after {attempts} attempts"
        )
    base = np.stack(values).astype(np.float32, copy=False)
    physical, applied, physics_rejections = _apply_physics(
        base, expected_topology, physics, seed + 1_000_003,
    )
    rejections.update(physics_rejections)
    final_fingerprints: set[str] = set()
    for index, row in enumerate(physical):
        if not np.array_equal(row[:, 2:], base[index, :, 2:]):
            raise AssertionError("materialized physics changed non-spatial channels")
        fingerprint = _fingerprint(row, output_labels[index])
        if fingerprint in final_fingerprints:
            raise ValueError("materialized physics produced a duplicate synthetic row")
        final_fingerprints.add(fingerprint)
        metadata[index]["synthetic_id"] = fingerprint
        metadata[index]["physics"]["applied"] = bool(applied[index])
        if index < len(previews):
            previews[index]["physical"] = row
    direction_counts = Counter(key[2] for key in expected_topology)
    audit = {
        "schema": SCHEMA,
        "requested": size,
        "generated": len(physical),
        "attempts": attempts,
        "eligible_labels": len(eligible_labels),
        "generated_labels": len(set(output_labels)),
        "eligible_profile_groups": len(groups),
        "unique_parent_fingerprints": len(parent_fingerprints),
        "same_label_only": True,
        "same_stroke_count_only": True,
        "same_loop_direction_only": True,
        "cross_writer_required": writer_groups is not None,
        "profile_fitted_rows": len(catalog.allowed_indices),
        "evaluation_rows": 0,
        "crohme_rows": 0,
        "non_spatial_channels_preserved": True,
        "physics": asdict(physics),
        "physics_applied_rows": int(sum(applied)),
        "loop_direction_rows": {
            "clockwise_y_up": int(direction_counts.get(-1, 0)),
            "counterclockwise_y_up": int(direction_counts.get(1, 0)),
            "no_primary_loop": int(direction_counts.get(0, 0)),
        },
        "rejections": dict(sorted(rejections.items())),
    }
    return ProfiledBank(
        physical,
        np.asarray(output_labels, dtype=np.int64),
        audit,
        tuple(metadata),
        tuple(previews),
    )


def _sha256(path: Path) -> str:
    """파일을 스트리밍해 SHA-256 무결성 지문을 계산한다."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_profiled_bank(path: Path, bank: ProfiledBank) -> dict:
    """합성 텐서 NPZ와 행별 gzip JSONL 메타데이터를 함께 기록한다."""

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, features=bank.features, labels=bank.labels)
    metadata_path = path.with_suffix(".metadata.jsonl.gz")
    with gzip.open(metadata_path, "wt", encoding="utf-8", newline="\n") as stream:
        for row in bank.metadata:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return {
        "path": str(path.resolve()), "sha256": _sha256(path),
        "metadata_path": str(metadata_path.resolve()),
        "metadata_sha256": _sha256(metadata_path),
        "records": len(bank.labels), "labels": len(set(bank.labels.tolist())),
    }


def load_profiled_bank(path: Path, expected_sha256: str | None = None) -> ProfiledBank:
    """물질화 합성은행의 해시·형상·채널 계약을 검증해 읽는다."""

    if not path.is_file():
        raise FileNotFoundError(f"missing profiled bank: {path}")
    if expected_sha256 and _sha256(path).lower() != expected_sha256.lower():
        raise ValueError(f"profiled bank hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as payload:
        features = np.asarray(payload["features"], dtype=np.float32)
        labels = np.asarray(payload["labels"], dtype=np.int64)
    if features.shape != (len(labels), POINTS, CHANNELS) or not len(labels):
        raise ValueError(f"profiled bank shape mismatch: {features.shape}")
    if not np.isfinite(features).all():
        raise ValueError("profiled bank contains non-finite values")
    if float(features[:, :, :2].min()) < 0.0 or float(features[:, :, :2].max()) > 1.0:
        raise ValueError("profiled bank coordinates are outside [0,1]")
    if not np.allclose(features[:, 1:, 2], 1.0 / (POINTS - 1), atol=1.0e-7):
        raise ValueError("profiled bank does not use uniform delta_t")
    if not np.all(features[:, 0, 2] == 0.0) or not np.all(features[:, :, 4] == 1.0):
        raise ValueError("profiled bank observed/time boundary contract mismatch")
    return ProfiledBank(features, labels, {}, (), ())


def save_profile_preview(
    path: Path,
    previews: Sequence[dict],
    label_names: Sequence[str],
) -> None:
    """부모·DTW·프로파일·물리 결과를 한 장에 그려 육안 검수를 지원한다."""

    from PIL import Image, ImageDraw

    rows = list(previews)[:8]
    if not rows:
        raise ValueError("profile preview requires at least one row")
    width, header, row_height = 1040, 70, 220
    image = Image.new("RGB", (width, header + row_height * len(rows)), "white")
    draw = ImageDraw.Draw(image)
    draw.text((20, 14), "Clean-room class-profile augmentation v3", fill="#172033")
    columns = ("approved parent", "same-label DTW", "class profile", "physics")
    for column, title in enumerate(columns):
        draw.text((78 + column * 245, 43), title, fill="#43516a")
    colors = ("#2667ff", "#7a4ce0", "#de6b35", "#16856d")
    for row_index, row in enumerate(rows):
        top = header + row_index * row_height
        draw.text((12, top + 92), str(label_names[int(row["label"])]), fill="#172033")
        for column, values in enumerate((
            row["anchor"], row["mixed"], row["profiled"], row["physical"],
        )):
            left = 62 + column * 245
            draw.rounded_rectangle(
                (left, top + 12, left + 205, top + 207), radius=10,
                fill="#f7f9fc", outline="#d6dce7", width=2,
            )
            for start, end in stroke_bounds(np.asarray(values)):
                points = [
                    (
                        int(left + 18 + float(x) * 169),
                        int(top + 25 + float(y) * 169),
                    )
                    for x, y in np.asarray(values)[start:end, :2]
                ]
                if len(points) >= 2:
                    draw.line(points, fill=colors[column], width=4, joint="curve")
                    radius = 4
                    draw.ellipse(
                        (
                            points[0][0] - radius, points[0][1] - radius,
                            points[0][0] + radius, points[0][1] + radius,
                        ),
                        fill="#172033",
                    )
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _circle_tensor(aspect: float, phase: float, wobble: float) -> np.ndarray:
    """자체검사용 반시계 폐곡선 128x5 텐서를 만든다."""

    angle = np.linspace(phase, phase + math.tau, POINTS)
    radius = 1.0 + wobble * np.sin(3.0 * angle)
    output = np.zeros((POINTS, CHANNELS), dtype=np.float32)
    output[:, 0] = 0.5 + 0.38 * aspect * radius * np.cos(angle)
    output[:, 1] = 0.5 + 0.38 / aspect * radius * np.sin(angle)
    output[:, :2] = _letterbox_numpy(output[:, :2])
    output[:, 2] = 1.0 / (POINTS - 1)
    output[0, 2] = 0.0
    output[0, 3] = 1.0
    output[:, 4] = 1.0
    return output


def _self_test() -> None:
    """폐곡선 방향·시작점·비공간 채널 보존과 저장 로더 계약을 빠르게 검증한다."""

    features = np.stack([
        _circle_tensor(0.78, 0.0, 0.03),
        _circle_tensor(1.22, 0.7, 0.08),
        _circle_tensor(0.88, 1.5, 0.12),
        _circle_tensor(1.12, 2.4, 0.16),
    ])
    labels = np.zeros(4, dtype=np.int64)
    catalog = fit_trajectory_profiles(
        features, labels, ["0"], source_role="self_test_approved_train",
    )
    assert catalog.profiles["classes"]["0"]["loop_records"] == 4
    assert set(catalog.loop_direction.tolist()) == {1}
    bank = build_profiled_bank(
        features, labels, catalog, 3, 7,
        writer_groups=["a", "b", "c", "d"], physics=NO_PHYSICS,
    )
    assert bank.features.shape == (3, POINTS, CHANNELS)
    assert np.array_equal(bank.features[:, :, 2:], np.repeat(features[:1, :, 2:], 3, axis=0))
    assert all(trajectory_descriptor(row).loop_direction == 1 for row in bank.features)


def main() -> int:
    """독립 모듈 자체검사 CLI를 실행한다."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test:
        parser.error("this module exposes only --self-test; use the dataset builder")
    _self_test()
    print(json.dumps({"self_test": "pass", "schema": SCHEMA}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
