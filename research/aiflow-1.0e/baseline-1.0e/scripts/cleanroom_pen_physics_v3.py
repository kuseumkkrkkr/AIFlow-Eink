#!/usr/bin/env python3
"""48 Hz 온라인 잉크용 clean-room 펜·손 운동 시뮬레이션 v3."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from typing import Sequence

import torch
import torch.nn.functional as F


POINTS = 128
CHANNELS = 5
SCHEMA = "aiflow-cleanroom-pen-physics/v3"


@dataclass(frozen=True)
class PenPhysicsConfigV3:
    """운동계·신경운동 잡음·장치 샘플링의 제한 범위를 선언한다."""

    name: str
    sample_rate_hz: float
    oversample: int
    natural_frequency_hz_min: float
    natural_frequency_hz_max: float
    damping_ratio_min: float
    damping_ratio_max: float
    minimum_jerk_strength: float
    curvature_slowdown: float
    tremor_hz_min: float
    tremor_hz_max: float
    tremor_amplitude: float
    drift_amplitude: float
    quantization_levels: int
    max_rms_displacement: float
    max_point_displacement: float
    min_path_length_ratio: float
    max_path_length_ratio: float

    @property
    def enabled(self) -> bool:
        """동역학 변형이 활성화된 설정인지 반환한다."""

        return self.name != "none"

    def validate(self) -> None:
        """수치 적분과 샘플링에 필요한 설정 범위를 검증한다."""

        if not self.enabled:
            return
        if self.sample_rate_hz <= 0.0 or self.oversample < 2:
            raise ValueError("sample rate and oversample must be positive")
        if not 0.0 < self.natural_frequency_hz_min <= self.natural_frequency_hz_max:
            raise ValueError("invalid natural-frequency range")
        if not 0.35 <= self.damping_ratio_min <= self.damping_ratio_max <= 1.5:
            raise ValueError("invalid damping-ratio range")
        if not 0.0 <= self.minimum_jerk_strength <= 1.0:
            raise ValueError("minimum-jerk strength must remain in [0,1]")
        nyquist = self.sample_rate_hz * 0.5
        if not 0.0 < self.tremor_hz_min <= self.tremor_hz_max < nyquist:
            raise ValueError("tremor frequency must remain below Nyquist")
        if min(
            self.tremor_amplitude,
            self.drift_amplitude,
            self.max_rms_displacement,
            self.max_point_displacement,
        ) < 0.0:
            raise ValueError("amplitudes and displacement gates cannot be negative")
        if self.quantization_levels < 256:
            raise ValueError("quantization must retain at least 8-bit coordinate quality")
        if not 0.0 < self.min_path_length_ratio <= 1.0 <= self.max_path_length_ratio:
            raise ValueError("invalid path-length preservation range")


@dataclass(frozen=True)
class PenClassProfileV3:
    """문자 형태군에 따라 기본 펜 동역학의 상대 강도를 조정한다."""

    family: str
    natural_frequency_scale: float
    damping_offset: float
    minimum_jerk_scale: float
    curvature_scale: float
    tremor_scale: float
    drift_scale: float
    protect_short_stroke_points: int

    def validate(self) -> None:
        """문자 프로파일이 형태를 과도하게 바꾸지 않는 제한 안인지 확인한다."""

        if not self.family:
            raise ValueError("class profile family is required")
        if not 0.7 <= self.natural_frequency_scale <= 1.3:
            raise ValueError("class natural-frequency scale is outside the safe range")
        if not -0.2 <= self.damping_offset <= 0.2:
            raise ValueError("class damping offset is outside the safe range")
        if not all(0.35 <= value <= 1.4 for value in (
            self.minimum_jerk_scale,
            self.curvature_scale,
            self.tremor_scale,
            self.drift_scale,
        )):
            raise ValueError("class motion scale is outside the safe range")
        if not 0 <= self.protect_short_stroke_points <= 12:
            raise ValueError("short-stroke protection is outside the safe range")


GENERIC_CLASS_PROFILE = PenClassProfileV3("mixed", 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2)
LOOP_CLASS_PROFILE = PenClassProfileV3("loop", 1.05, 0.08, 0.85, 1.25, 0.70, 0.65, 2)
STEM_CLASS_PROFILE = PenClassProfileV3("stem", 1.10, 0.06, 0.72, 0.55, 0.72, 0.60, 3)
DOT_CLASS_PROFILE = PenClassProfileV3("dot_bearing", 1.02, 0.10, 0.70, 0.80, 0.55, 0.55, 8)
CLASS_PROFILES = {
    profile.family: profile for profile in (
        GENERIC_CLASS_PROFILE, LOOP_CLASS_PROFILE, STEM_CLASS_PROFILE, DOT_CLASS_PROFILE,
    )
}


NO_MOTOR_PHYSICS = PenPhysicsConfigV3(
    "none", 48.0, 4, 5.0, 5.0, 0.8, 0.8, 0.0, 0.0,
    8.0, 8.0, 0.0, 0.0, 8192, 0.0, 0.0, 1.0, 1.0,
)
MOTOR_LIGHT_PHYSICS = PenPhysicsConfigV3(
    "motor_light", 48.0, 4, 5.0, 7.5, 0.72, 1.02, 0.22, 0.40,
    7.0, 12.0, 0.0024, 0.0032, 8192, 0.035, 0.105, 0.78, 1.22,
)
MOTOR_MEDIUM_PHYSICS = PenPhysicsConfigV3(
    "motor_medium", 48.0, 4, 4.2, 7.0, 0.62, 1.08, 0.35, 0.65,
    6.5, 12.5, 0.0036, 0.0048, 4096, 0.052, 0.145, 0.74, 1.28,
)


def _stroke_bounds(features: torch.Tensor) -> list[tuple[int, int]]:
    """한 행의 stroke_start 채널을 반열림 획 구간으로 복원한다."""

    starts = torch.nonzero(features[:, 3] > 0.5, as_tuple=False).flatten().tolist()
    if not starts or starts[0] != 0:
        raise ValueError("online tensor must begin with stroke_start")
    ends = starts[1:] + [POINTS]
    bounds = list(zip(starts, ends, strict=True))
    if any(end <= start for start, end in bounds):
        raise ValueError("online tensor contains an empty stroke")
    return bounds


def _letterbox(xy: torch.Tensor) -> torch.Tensor:
    """종횡비를 보존하며 배치 좌표를 단위 정사각형으로 정규화한다."""

    low = xy.amin(dim=1, keepdim=True)
    high = xy.amax(dim=1, keepdim=True)
    center = (low + high) * 0.5
    extent = (high - low).amax(dim=2, keepdim=True).clamp_min(1.0e-6)
    return ((xy - center) / extent + 0.5).clamp(0.0, 1.0)


def _uniform(generator: torch.Generator, reference: torch.Tensor, low: float, high: float) -> torch.Tensor:
    """참조 텐서와 같은 장치·dtype의 단일 균등 난수를 만든다."""

    value = torch.rand((), generator=generator, device=reference.device, dtype=reference.dtype)
    return low + value * (high - low)


def _minimum_jerk_progress(
    points: torch.Tensor,
    strength: float,
    slowdown: float,
) -> torch.Tensor:
    """획 양끝 감속과 곡률 체류시간을 반영한 단조 진행 위치를 계산한다."""

    count = len(points)
    if count < 3:
        return torch.arange(count, device=points.device, dtype=points.dtype)
    delta = points[1:] - points[:-1]
    lengths = delta.norm(dim=1).clamp_min(1.0e-7)
    tangent = delta / lengths[:, None]
    curvature = torch.zeros(count - 1, device=points.device, dtype=points.dtype)
    if count > 2:
        turns = (tangent[1:] - tangent[:-1]).norm(dim=1).clamp_max(2.0)
        curvature[1:] = turns
        curvature[:-1] = torch.maximum(curvature[:-1], turns)
    travel_cost = lengths * (1.0 + slowdown * curvature)
    cumulative = torch.cat((torch.zeros(1, device=points.device, dtype=points.dtype), travel_cost.cumsum(0)))
    cumulative = cumulative / cumulative[-1].clamp_min(1.0e-7)
    clock = torch.linspace(0.0, 1.0, count, device=points.device, dtype=points.dtype)
    progress = 10.0 * clock.pow(3) - 15.0 * clock.pow(4) + 6.0 * clock.pow(5)
    upper = torch.searchsorted(cumulative, progress, right=False).clamp(1, count - 1)
    lower = upper - 1
    span = (cumulative[upper] - cumulative[lower]).clamp_min(1.0e-7)
    fraction = (progress - cumulative[lower]) / span
    profiled = lower.to(points.dtype) + fraction
    original = torch.arange(count, device=points.device, dtype=points.dtype)
    return original + strength * (profiled - original)


def _stroke_topology(points: torch.Tensor) -> tuple[bool, int, float]:
    """폐곡선 여부·회전방향·정규화 면적으로 획 구조를 요약한다."""

    if len(points) < 2:
        return False, 0, 0.0
    delta = points[1:] - points[:-1]
    path_length = float(delta.norm(dim=1).sum())
    width_height = points.amax(dim=0) - points.amin(dim=0)
    extent = max(float(width_height.max()), 1.0e-6)
    closure = float((points[-1] - points[0]).norm()) / extent
    x, y = points[:, 0], points[:, 1]
    area = float(0.5 * torch.sum(x * torch.roll(y, -1) - y * torch.roll(x, -1)))
    normalized_area = area / (extent * extent)
    closed = (
        len(points) >= 8
        and path_length >= 1.15 * extent
        and closure <= 0.32
        and abs(normalized_area) >= 0.055
    )
    return closed, (1 if normalized_area > 0.0 else -1) if closed else 0, normalized_area


def _row_topology(features: torch.Tensor, xy: torch.Tensor) -> tuple[int, int, int]:
    """대표 폐곡선 획과 방향을 기존 프로파일 기술자와 같은 규칙으로 찾는다."""

    rows = []
    for stroke_index, (start, end) in enumerate(_stroke_bounds(features)):
        closed, direction, area = _stroke_topology(xy[start:end])
        if closed:
            rows.append((stroke_index, direction, abs(area)))
    if not rows:
        return len(_stroke_bounds(features)), -1, 0
    stroke_index, direction, _area = max(rows, key=lambda row: row[2])
    return len(_stroke_bounds(features)), stroke_index, direction


def _interpolate(points: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """연속 인덱스 위치에서 획 좌표를 선형 보간한다."""

    lower = positions.floor().long().clamp(0, len(points) - 1)
    upper = (lower + 1).clamp_max(len(points) - 1)
    weight = (positions - lower.to(positions.dtype))[:, None]
    return points[lower] * (1.0 - weight) + points[upper] * weight


def _track_second_order(
    target: torch.Tensor,
    natural_hz: torch.Tensor,
    damping: torch.Tensor,
    sample_rate_hz: float,
    oversample: int,
) -> torch.Tensor:
    """감쇠 2차 추종계를 semi-implicit Euler로 안정 적분한다."""

    count = len(target)
    if count < 2:
        return target.clone()
    dt = 1.0 / (sample_rate_hz * oversample)
    omega = math.tau * natural_hz
    tip = target[0].clone()
    velocity = torch.zeros_like(tip)
    output = target.clone()
    for point_index in range(1, count):
        previous_target = target[point_index - 1]
        next_target = target[point_index]
        for substep in range(1, oversample + 1):
            fraction = substep / oversample
            command = previous_target + fraction * (next_target - previous_target)
            acceleration = omega.square() * (command - tip) - 2.0 * damping * omega * velocity
            velocity = velocity + acceleration * dt
            tip = tip + velocity * dt
        output[point_index] = tip
    clock = torch.linspace(0.0, 1.0, count, device=target.device, dtype=target.dtype)
    correction = clock.square() * (3.0 - 2.0 * clock)
    output = output + correction[:, None] * (target[-1] - output[-1])
    output[0], output[-1] = target[0], target[-1]
    return output


def _lateral_tremor(
    points: torch.Tensor,
    amplitude: float,
    frequency_hz: torch.Tensor,
    phase: torch.Tensor,
    sample_rate_hz: float,
) -> torch.Tensor:
    """획 접선의 수직 방향에 시간기반 대역제한 떨림을 더한다."""

    count = len(points)
    if count < 3 or amplitude <= 0.0:
        return torch.zeros_like(points)
    tangent = torch.gradient(points, dim=0)[0]
    tangent = tangent / tangent.norm(dim=1, keepdim=True).clamp_min(1.0e-7)
    normal = torch.stack((-tangent[:, 1], tangent[:, 0]), dim=1)
    clock = torch.arange(count, device=points.device, dtype=points.dtype) / sample_rate_hz
    envelope_clock = torch.linspace(0.0, 1.0, count, device=points.device, dtype=points.dtype)
    envelope = torch.sin(math.pi * envelope_clock).square()
    wave = torch.sin(math.tau * frequency_hz * clock + phase)
    return amplitude * envelope[:, None] * wave[:, None] * normal


def _low_frequency_drift(
    points: torch.Tensor,
    amplitude: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """양끝이 0인 저주파 손 위치 드리프트를 생성한다."""

    count = len(points)
    if count < 4 or amplitude <= 0.0:
        return torch.zeros_like(points)
    controls = max(4, math.ceil(count / 18) + 2)
    noise = torch.randn(
        (1, 2, controls), generator=generator,
        device=points.device, dtype=points.dtype,
    )
    smooth = F.interpolate(noise, size=count, mode="linear", align_corners=True)[0].transpose(0, 1)
    clock = torch.linspace(0.0, 1.0, count, device=points.device, dtype=points.dtype)[:, None]
    bridge = smooth - ((1.0 - clock) * smooth[0] + clock * smooth[-1])
    rms = bridge.square().mean().sqrt().clamp_min(1.0e-7)
    envelope = torch.sin(math.pi * clock).square()
    return amplitude * envelope * bridge / rms


def _path_length(points: torch.Tensor) -> torch.Tensor:
    """한 행 또는 획의 누적 유클리드 길이를 계산한다."""

    if len(points) < 2:
        return torch.zeros((), device=points.device, dtype=points.dtype)
    return (points[1:] - points[:-1]).norm(dim=1).sum()


def simulate_pen_physics_v3(
    features: torch.Tensor,
    config: PenPhysicsConfigV3,
    generator: torch.Generator,
    *,
    row_profiles: Sequence[PenClassProfileV3] | None = None,
    return_diagnostics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, list[dict]]:
    """128x5 계약을 보존하며 48 Hz 기반 펜·손 운동 변형을 적용한다."""

    if features.ndim != 3 or features.shape[1:] != (POINTS, CHANNELS):
        raise ValueError(f"expected batch x {POINTS} x {CHANNELS}, got {tuple(features.shape)}")
    if not torch.isfinite(features).all():
        raise ValueError("physics input contains non-finite values")
    if row_profiles is not None and len(row_profiles) != len(features):
        raise ValueError("row profile count does not match physics batch")
    if not config.enabled:
        output = features.clone()
        return (output, []) if return_diagnostics else output
    config.validate()
    baseline = features.clone()
    source_xy = _letterbox(baseline[:, :, :2])
    simulated_rows: list[torch.Tensor] = []
    diagnostics: list[dict] = []
    for row_index in range(len(features)):
        source = source_xy[row_index]
        result = source.clone()
        profile = GENERIC_CLASS_PROFILE if row_profiles is None else row_profiles[row_index]
        profile.validate()
        natural_hz = _uniform(
            generator, source, config.natural_frequency_hz_min,
            config.natural_frequency_hz_max,
        ) * profile.natural_frequency_scale
        damping = _uniform(
            generator, source, config.damping_ratio_min, config.damping_ratio_max,
        ) + profile.damping_offset
        tremor_hz = _uniform(generator, source, config.tremor_hz_min, config.tremor_hz_max)
        tremor_phase = _uniform(generator, source, 0.0, math.tau)
        for start, end in _stroke_bounds(features[row_index]):
            stroke = source[start:end]
            if len(stroke) <= profile.protect_short_stroke_points:
                result[start:end] = stroke
                continue
            positions = _minimum_jerk_progress(
                stroke,
                config.minimum_jerk_strength * profile.minimum_jerk_scale,
                config.curvature_slowdown * profile.curvature_scale,
            )
            command = _interpolate(stroke, positions)
            tracked = _track_second_order(
                command, natural_hz, damping,
                config.sample_rate_hz, config.oversample,
            )
            tracked = tracked + _lateral_tremor(
                tracked, config.tremor_amplitude * profile.tremor_scale, tremor_hz,
                tremor_phase, config.sample_rate_hz,
            )
            tracked = tracked + _low_frequency_drift(
                tracked, config.drift_amplitude * profile.drift_scale, generator,
            )
            tracked[0], tracked[-1] = stroke[0], stroke[-1]
            result[start:end] = tracked
        result = _letterbox(result[None])[0]
        delta = result - source
        rms_before_gate = delta.square().mean().sqrt()
        max_before_gate = delta.norm(dim=1).max()
        scale = torch.ones((), device=source.device, dtype=source.dtype)
        if config.max_rms_displacement > 0.0:
            scale = torch.minimum(
                scale,
                torch.as_tensor(config.max_rms_displacement, device=source.device, dtype=source.dtype)
                / rms_before_gate.clamp_min(1.0e-8),
            )
        if config.max_point_displacement > 0.0:
            scale = torch.minimum(
                scale,
                torch.as_tensor(config.max_point_displacement, device=source.device, dtype=source.dtype)
                / max_before_gate.clamp_min(1.0e-8),
            )
        result = source + scale.clamp_max(1.0) * delta
        levels = float(config.quantization_levels - 1)
        result = torch.round(result.clamp(0.0, 1.0) * levels) / levels
        topology_changed = _row_topology(features[row_index], result) != _row_topology(
            features[row_index], source,
        )
        path_ratio_before_revert = float(
            _path_length(result) / _path_length(source).clamp_min(1.0e-7)
        )
        path_length_changed = not (
            config.min_path_length_ratio <= path_ratio_before_revert
            <= config.max_path_length_ratio
        )
        if topology_changed or path_length_changed:
            result = source.clone()
        simulated_rows.append(result)
        final_delta = result - source
        diagnostics.append({
            "class_profile": profile.family,
            "natural_frequency_hz": float(natural_hz),
            "damping_ratio": float(damping),
            "tremor_hz": float(tremor_hz),
            "gate_scale": float(scale.clamp_max(1.0)),
            "rms_displacement": float(final_delta.square().mean().sqrt()),
            "max_point_displacement": float(final_delta.norm(dim=1).max()),
            "path_length_ratio": float(
                _path_length(result) / _path_length(source).clamp_min(1.0e-7)
            ),
            "topology_reverted": bool(topology_changed),
            "path_length_reverted": bool(path_length_changed),
            "path_length_ratio_before_revert": path_ratio_before_revert,
        })
    output = baseline.clone()
    output[:, :, :2] = torch.stack(simulated_rows)
    if not torch.equal(output[:, :, 2:], features[:, :, 2:]):
        raise AssertionError("v3 physics changed time, stroke, or observed channels")
    if not torch.isfinite(output).all():
        raise AssertionError("v3 physics produced non-finite values")
    return (output, diagnostics) if return_diagnostics else output


def _self_test() -> None:
    """결정성·채널 불변성·수치 상한·다획 리셋을 작은 궤적으로 검증한다."""

    values = torch.zeros((3, POINTS, CHANNELS), dtype=torch.float32)
    progress = torch.linspace(0.0, 1.0, POINTS)
    values[:, :, 0] = progress
    values[:, :, 1] = 0.5 + 0.18 * torch.sin(math.tau * progress)[None]
    values[:, :, 2] = 1.0 / (POINTS - 1)
    values[:, 0, 2] = 0.0
    values[:, 0, 3] = 1.0
    values[1:, 64, 3] = 1.0
    values[:, :, 4] = 1.0
    first, diagnostics = simulate_pen_physics_v3(
        values, MOTOR_LIGHT_PHYSICS, torch.Generator().manual_seed(17),
        return_diagnostics=True,
    )
    second = simulate_pen_physics_v3(
        values, MOTOR_LIGHT_PHYSICS, torch.Generator().manual_seed(17),
    )
    assert torch.equal(first, second)
    assert torch.equal(first[:, :, 2:], values[:, :, 2:])
    assert first[:, :, :2].min() >= 0.0 and first[:, :, :2].max() <= 1.0
    assert max(row["rms_displacement"] for row in diagnostics) <= 0.0355
    assert max(row["max_point_displacement"] for row in diagnostics) <= 0.106
    assert not torch.equal(first[:, :, :2], values[:, :, :2])


def main() -> int:
    """독립 물리 엔진 자체검사를 실행한다."""

    _self_test()
    print(json.dumps({
        "self_test": "pass", "schema": SCHEMA,
        "config": asdict(MOTOR_LIGHT_PHYSICS),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
