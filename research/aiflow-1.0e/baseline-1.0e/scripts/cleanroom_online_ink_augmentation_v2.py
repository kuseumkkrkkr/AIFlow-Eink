#!/usr/bin/env python3
"""CROHME와 분리된 온라인 필기 DTW 혼합·물리 증강 연산."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F


POINTS = 128
CHANNELS = 5


@dataclass(frozen=True)
class PhysicalConfig:
    """필기구 응답과 손 움직임의 허용 변동 범위를 고정한다."""

    name: str
    time_warp_log: float
    response_min: float
    response_max: float
    momentum_min: float
    momentum_max: float
    tremor: float
    drift: float

    @property
    def enabled(self) -> bool:
        """하나 이상의 물리 변형값이 활성화됐는지 반환한다."""

        return any((self.time_warp_log, self.tremor, self.drift)) or (
            self.response_min != 1.0 or self.response_max != 1.0
            or self.momentum_min != 0.0 or self.momentum_max != 0.0
        )


NO_PHYSICS = PhysicalConfig("none", 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0)
LIGHT_PHYSICS = PhysicalConfig("light", 0.12, 0.62, 0.82, 0.03, 0.14, 0.0035, 0.0045)
MEDIUM_PHYSICS = PhysicalConfig("medium", 0.20, 0.52, 0.78, 0.05, 0.20, 0.0050, 0.0070)


@dataclass(frozen=True)
class DtwBank:
    """동일 라벨의 권리 확인 궤적만으로 만든 합성 텐서 묶음이다."""

    features: np.ndarray
    labels: np.ndarray
    audit: dict
    previews: tuple[dict, ...]


def stroke_bounds(features: np.ndarray) -> list[tuple[int, int]]:
    """stroke_start 채널에서 각 획의 반열림 인덱스 구간을 복원한다."""

    if features.shape != (POINTS, CHANNELS):
        raise ValueError(f"expected {POINTS} x {CHANNELS}, got {features.shape}")
    starts = np.flatnonzero(features[:, 3] > 0.5).astype(int).tolist()
    if not starts or starts[0] != 0:
        raise ValueError("online tensor does not start with a stroke boundary")
    ends = starts[1:] + [POINTS]
    bounds = list(zip(starts, ends, strict=True))
    if any(end <= start for start, end in bounds):
        raise ValueError("online tensor contains an empty stroke")
    return bounds


def _letterbox_numpy(xy: np.ndarray) -> np.ndarray:
    """좌표를 종횡비 보존 단위 정사각형으로 다시 정규화한다."""

    low = xy.min(axis=0, keepdims=True)
    high = xy.max(axis=0, keepdims=True)
    center = (low + high) * 0.5
    extent = max(float((high - low).max()), 1.0e-6)
    return np.clip((xy - center) / extent + 0.5, 0.0, 1.0)


def _tangents(points: np.ndarray) -> np.ndarray:
    """DTW 비용에 사용할 단위 접선 벡터를 계산한다."""

    if len(points) == 1:
        return np.zeros_like(points)
    tangent = np.gradient(points, axis=0)
    norm = np.linalg.norm(tangent, axis=1, keepdims=True)
    return tangent / np.maximum(norm, 1.0e-6)


def _dtw_stroke(anchor: np.ndarray, partner: np.ndarray, band_ratio: float) -> tuple[np.ndarray, float]:
    """한 획을 제한폭 DTW로 정렬해 anchor 길이의 partner 좌표를 만든다."""

    n, m = len(anchor), len(partner)
    if n < 1 or m < 1:
        raise ValueError("DTW cannot align an empty stroke")
    band = max(abs(n - m), int(math.ceil(max(n, m) * band_ratio)), 2)
    cumulative = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    direction = np.zeros((n + 1, m + 1), dtype=np.uint8)
    cumulative[0, 0] = 0.0
    anchor_tangent, partner_tangent = _tangents(anchor), _tangents(partner)
    for i in range(1, n + 1):
        lower, upper = max(1, i - band), min(m, i + band)
        for j in range(lower, upper + 1):
            spatial = float(np.linalg.norm(anchor[i - 1] - partner[j - 1]))
            tangent = float(np.linalg.norm(anchor_tangent[i - 1] - partner_tangent[j - 1]))
            previous = (
                cumulative[i - 1, j - 1],
                cumulative[i - 1, j],
                cumulative[i, j - 1],
            )
            move = int(np.argmin(previous))
            cumulative[i, j] = spatial + 0.12 * tangent + previous[move]
            direction[i, j] = move
    if not np.isfinite(cumulative[n, m]):
        raise ValueError("DTW band could not connect the two strokes")

    pairs: list[tuple[int, int]] = []
    i, j = n, m
    while i > 0 and j > 0:
        pairs.append((i - 1, j - 1))
        move = int(direction[i, j])
        if move == 0:
            i, j = i - 1, j - 1
        elif move == 1:
            i -= 1
        else:
            j -= 1
    while i > 0:
        pairs.append((i - 1, 0)); i -= 1
    while j > 0:
        pairs.append((0, j - 1)); j -= 1
    pairs.reverse()

    buckets: list[list[np.ndarray]] = [[] for _ in range(n)]
    for anchor_index, partner_index in pairs:
        buckets[anchor_index].append(partner[partner_index])
    aligned = np.stack([
        np.mean(bucket, axis=0) if bucket else partner[min(index, m - 1)]
        for index, bucket in enumerate(buckets)
    ])
    return aligned, float(cumulative[n, m] / max(len(pairs), 1))


def dtw_align_partner(anchor: np.ndarray, partner: np.ndarray, band_ratio: float = 0.22) -> tuple[np.ndarray, float]:
    """획 수와 순서를 유지한 채 partner 전체를 anchor의 128점 격자에 정렬한다."""

    anchor_bounds, partner_bounds = stroke_bounds(anchor), stroke_bounds(partner)
    if len(anchor_bounds) != len(partner_bounds):
        raise ValueError("DTW mixing requires equal stroke counts")
    aligned = np.empty((POINTS, 2), dtype=np.float64)
    weighted_cost = 0.0
    for (a_start, a_end), (b_start, b_end) in zip(
        anchor_bounds, partner_bounds, strict=True,
    ):
        values, cost = _dtw_stroke(
            np.asarray(anchor[a_start:a_end, :2], dtype=np.float64),
            np.asarray(partner[b_start:b_end, :2], dtype=np.float64),
            band_ratio,
        )
        aligned[a_start:a_end] = values
        weighted_cost += cost * (a_end - a_start)
    return aligned, weighted_cost / POINTS


def blend_dtw_pair(
    anchor: np.ndarray,
    partner: np.ndarray,
    alpha: float,
    *,
    min_pair_rms: float = 0.035,
    max_pair_rms: float = 0.45,
) -> tuple[np.ndarray, dict]:
    """서로 충분히 다른 동일형상 두 궤적을 DTW 보간해 한 합성 표본을 만든다."""

    if not 0.0 < alpha < 1.0:
        raise ValueError("DTW blend alpha must be in (0, 1)")
    aligned, dtw_cost = dtw_align_partner(anchor, partner)
    anchor_xy = np.asarray(anchor[:, :2], dtype=np.float64)
    pair_rms = float(np.sqrt(np.mean(np.square(anchor_xy - aligned))))
    if not min_pair_rms <= pair_rms <= max_pair_rms:
        raise ValueError("parent distance is outside the clean-room blend gate")
    mixed_xy = _letterbox_numpy((1.0 - alpha) * anchor_xy + alpha * aligned)
    output = np.asarray(anchor, dtype=np.float32).copy()
    output[:, :2] = mixed_xy.astype(np.float32)
    anchor_distance = float(np.sqrt(np.mean(np.square(mixed_xy - _letterbox_numpy(anchor_xy)))))
    partner_distance = float(np.sqrt(np.mean(np.square(mixed_xy - _letterbox_numpy(aligned)))))
    if min(anchor_distance, partner_distance) < 0.008:
        raise ValueError("synthetic trajectory is too close to a parent")
    if not np.array_equal(output[:, 2:], np.asarray(anchor)[:, 2:]):
        raise AssertionError("DTW mixing changed a non-spatial channel")
    return output, {
        "alpha": float(alpha),
        "dtw_cost": dtw_cost,
        "pair_rms": pair_rms,
        "anchor_rms": anchor_distance,
        "partner_rms": partner_distance,
    }


def _topology_groups(
    features: np.ndarray,
    labels: np.ndarray,
    allowed_indices: np.ndarray,
) -> dict[tuple[int, int], np.ndarray]:
    """라벨과 획 수가 같은 부모 후보 인덱스를 메모리 절약형으로 묶는다."""

    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for start in range(0, len(allowed_indices), 4096):
        indices = allowed_indices[start:start + 4096]
        starts = np.count_nonzero(np.asarray(features[indices, :, 3]) > 0.5, axis=1)
        for index, strokes in zip(indices.tolist(), starts.tolist(), strict=True):
            groups[(int(labels[index]), int(strokes))].append(int(index))
    return {
        key: np.asarray(value, dtype=np.int64)
        for key, value in groups.items() if len(value) >= 2
    }


def build_dtw_bank(
    features: np.ndarray,
    labels: np.ndarray,
    size: int,
    seed: int,
    *,
    allowed_indices: np.ndarray | None = None,
    writer_groups: Sequence[str] | None = None,
    preview_count: int = 12,
) -> DtwBank:
    """학습 분할 안에서만 동일 라벨 DTW 합성은행을 결정적으로 생성한다."""

    labels = np.asarray(labels, dtype=np.int64)
    if len(features) != len(labels) or size < 1:
        raise ValueError("invalid DTW bank input")
    allowed = (
        np.arange(len(labels), dtype=np.int64)
        if allowed_indices is None else np.asarray(allowed_indices, dtype=np.int64)
    )
    if not len(allowed) or allowed.min() < 0 or allowed.max() >= len(labels):
        raise ValueError("invalid DTW parent indices")
    if writer_groups is not None and len(writer_groups) != len(labels):
        raise ValueError("writer group count does not match DTW parents")

    groups = _topology_groups(features, labels, allowed)
    if writer_groups is not None:
        groups = {
            key: indices for key, indices in groups.items()
            if len({str(writer_groups[index]) for index in indices}) >= 2
        }
    by_label: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for key in groups:
        by_label[key[0]].append(key)
    eligible_labels = sorted(by_label)
    if not eligible_labels:
        raise ValueError("no same-label topology group can produce a DTW blend")

    rng = np.random.default_rng(seed)
    output: list[np.ndarray] = []
    output_labels: list[int] = []
    diagnostics: list[dict] = []
    previews: list[dict] = []
    fingerprints: set[bytes] = set()
    parents: set[int] = set()
    rejections: Counter[str] = Counter()
    attempts = 0
    max_attempts = max(size * 80, 1000)
    shuffled_labels = np.asarray(eligible_labels, dtype=np.int64)
    rng.shuffle(shuffled_labels)

    while len(output) < size and attempts < max_attempts:
        label = int(shuffled_labels[attempts % len(shuffled_labels)])
        key = by_label[label][int(rng.integers(0, len(by_label[label])))]
        candidates = groups[key]
        anchor_index, partner_index = rng.choice(candidates, size=2, replace=False).tolist()
        attempts += 1
        if writer_groups is not None and str(writer_groups[anchor_index]) == str(writer_groups[partner_index]):
            rejections["same_writer"] += 1
            continue
        anchor = np.asarray(features[anchor_index], dtype=np.float32).copy()
        partner = np.asarray(features[partner_index], dtype=np.float32).copy()
        anchor[:, 2] = 1.0 / (POINTS - 1); anchor[0, 2] = 0.0; anchor[:, 4] = 1.0
        partner[:, 2] = 1.0 / (POINTS - 1); partner[0, 2] = 0.0; partner[:, 4] = 1.0
        try:
            mixed, diagnostic = blend_dtw_pair(
                anchor, partner, float(rng.uniform(0.30, 0.70)),
            )
        except ValueError as error:
            rejections[str(error)] += 1
            continue
        fingerprint = hashlib.sha256(
            np.asarray(np.round(mixed[:, :2], 5), dtype=np.float32).tobytes()
            + np.asarray([label], dtype=np.int64).tobytes()
        ).digest()
        if fingerprint in fingerprints:
            rejections["duplicate_synthetic"] += 1
            continue
        fingerprints.add(fingerprint)
        output.append(mixed)
        output_labels.append(label)
        diagnostics.append(diagnostic)
        parents.update((anchor_index, partner_index))
        if len(previews) < preview_count:
            aligned, _ = dtw_align_partner(anchor, partner)
            previews.append({
                "label": label,
                "anchor": anchor,
                "partner_aligned": np.column_stack((aligned, anchor[:, 2:])),
                "mixed": mixed,
            })

    if not output:
        raise ValueError("DTW blend bank could not pass distance and topology gates")
    values = np.stack(output).astype(np.float32, copy=False)
    target = np.asarray(output_labels, dtype=np.int64)
    pair_rms = np.asarray([row["pair_rms"] for row in diagnostics], dtype=np.float64)
    dtw_cost = np.asarray([row["dtw_cost"] for row in diagnostics], dtype=np.float64)
    audit = {
        "requested": int(size),
        "generated": len(values),
        "attempts": attempts,
        "eligible_labels": len(eligible_labels),
        "generated_labels": len(set(output_labels)),
        "eligible_topology_groups": len(groups),
        "unique_parent_records": len(parents),
        "same_label_pairs_only": True,
        "same_stroke_count_only": True,
        "cross_writer_required": writer_groups is not None,
        "non_spatial_channels_preserved": True,
        "rejections": dict(sorted(rejections.items())),
        "pair_rms": {
            "minimum": float(pair_rms.min()),
            "median": float(np.median(pair_rms)),
            "maximum": float(pair_rms.max()),
        },
        "dtw_cost": {
            "minimum": float(dtw_cost.min()),
            "median": float(np.median(dtw_cost)),
            "maximum": float(dtw_cost.max()),
        },
    }
    return DtwBank(values, target, audit, tuple(previews))


def _torch_letterbox(xy: torch.Tensor) -> torch.Tensor:
    """배치 좌표를 종횡비 보존 단위 정사각형으로 정규화한다."""

    low = xy.amin(dim=1, keepdim=True)
    high = xy.amax(dim=1, keepdim=True)
    center = (low + high) * 0.5
    extent = (high - low).amax(dim=2, keepdim=True).clamp_min(1.0e-6)
    return ((xy - center) / extent + 0.5).clamp(0.0, 1.0)


def _time_warp_xy(xy: torch.Tensor, starts: torch.Tensor, amount: float, generator: torch.Generator) -> torch.Tensor:
    """각 획 내부의 진행률만 단조 변형해 필기 속도 차이를 재현한다."""

    if amount <= 0:
        return xy
    output = xy.clone()
    for batch_index in range(len(xy)):
        boundaries = torch.nonzero(starts[batch_index] > 0.5, as_tuple=False).flatten().tolist()
        boundaries.append(POINTS)
        for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
            count = end - start
            if count < 3:
                continue
            gamma = torch.exp(
                (torch.rand((), generator=generator, device=xy.device, dtype=xy.dtype) * 2.0 - 1.0)
                * amount
            )
            progress = torch.linspace(0.0, 1.0, count, device=xy.device, dtype=xy.dtype).pow(gamma)
            position = progress * (count - 1)
            lower = position.floor().long()
            upper = position.ceil().long().clamp_max(count - 1)
            weight = (position - lower.to(position.dtype)).unsqueeze(1)
            source = xy[batch_index, start:end]
            output[batch_index, start:end] = source[lower] * (1.0 - weight) + source[upper] * weight
    return output


def simulate_pen_physics(
    features: torch.Tensor,
    config: PhysicalConfig,
    generator: torch.Generator,
) -> torch.Tensor:
    """질량–스프링 응답, 관성, 손떨림과 드리프트를 좌표에만 적용한다."""

    if features.ndim != 3 or features.shape[1:] != (POINTS, CHANNELS):
        raise ValueError(f"expected batch x {POINTS} x {CHANNELS}, got {tuple(features.shape)}")
    if not config.enabled:
        return features.clone()
    result = features.clone()
    xy = _time_warp_xy(result[:, :, :2], result[:, :, 3], config.time_warp_log, generator)
    batch, dtype, device = len(xy), xy.dtype, xy.device
    response = torch.rand((batch, 1), generator=generator, device=device, dtype=dtype)
    response = config.response_min + response * (config.response_max - config.response_min)
    momentum = torch.rand((batch, 1), generator=generator, device=device, dtype=dtype)
    momentum = config.momentum_min + momentum * (config.momentum_max - config.momentum_min)
    tip = xy[:, 0].clone()
    velocity = torch.zeros_like(tip)
    simulated = xy.clone()
    for point_index in range(1, POINTS):
        velocity = momentum * velocity + response * (xy[:, point_index] - tip)
        tip = tip + velocity
        reset = result[:, point_index, 3:4] > 0.5
        tip = torch.where(reset, xy[:, point_index], tip)
        velocity = torch.where(reset, torch.zeros_like(velocity), velocity)
        simulated[:, point_index] = tip

    progress = torch.linspace(0.0, 1.0, POINTS, device=device, dtype=dtype)[None, None, :]
    frequency = 4.0 + 5.0 * torch.rand((batch, 2, 1), generator=generator, device=device, dtype=dtype)
    phase = math.tau * torch.rand((batch, 2, 1), generator=generator, device=device, dtype=dtype)
    tremor = config.tremor * torch.sin(math.tau * frequency * progress + phase).transpose(1, 2)
    noise = torch.randn((batch, 2, POINTS), generator=generator, device=device, dtype=dtype)
    smooth = F.avg_pool1d(F.pad(noise, (15, 15), mode="reflect"), 31, stride=1)
    smooth = smooth / smooth.square().mean(dim=2, keepdim=True).sqrt().clamp_min(1.0e-6)
    drift = config.drift * smooth.transpose(1, 2)
    result[:, :, :2] = _torch_letterbox(simulated + tremor + drift)
    if not torch.equal(result[:, :, 2:], features[:, :, 2:]):
        raise AssertionError("physical simulation changed time, stroke, or observed channels")
    if not torch.isfinite(result).all():
        raise AssertionError("physical simulation produced non-finite values")
    return result


def save_preview(path: Path, previews: Sequence[dict], label_names: Sequence[str], config: PhysicalConfig) -> None:
    """부모·DTW 혼합·물리 시뮬레이션을 한 화면에 저장해 육안 검수를 돕는다."""

    from PIL import Image, ImageDraw

    rows = list(previews)[:8]
    if not rows:
        raise ValueError("preview requires at least one DTW sample")
    width, header, row_height = 1040, 70, 220
    image = Image.new("RGB", (width, header + row_height * len(rows)), "white")
    draw = ImageDraw.Draw(image)
    columns = ("parent A", "parent B aligned", "DTW blend", "physics")
    draw.text((20, 14), f"Clean-room online ink augmentation - {config.name}", fill="#172033")
    for column, title in enumerate(columns):
        draw.text((90 + column * 245, 43), title, fill="#43516a")
    generator = torch.Generator().manual_seed(20260823)
    for row_index, row in enumerate(rows):
        mixed = torch.from_numpy(np.asarray(row["mixed"], dtype=np.float32))[None]
        physical = simulate_pen_physics(mixed, config, generator)[0].numpy()
        top = header + row_index * row_height
        draw.text((12, top + 92), str(label_names[int(row["label"])]), fill="#172033")
        for column, values in enumerate((
            row["anchor"], row["partner_aligned"], row["mixed"], physical,
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
                    for x, y in np.asarray(values[start:end, :2])
                ]
                if len(points) == 1:
                    x, y = points[0]
                    draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill="#2856a8")
                else:
                    draw.line(points, fill="#2856a8", width=3, joint="curve")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=True)


def _self_test() -> None:
    """라벨·획·채널 보존과 writer 분리 조건을 작은 합성 예제로 검증한다."""

    base = np.zeros((4, POINTS, CHANNELS), dtype=np.float32)
    progress = np.linspace(0.0, 1.0, POINTS, dtype=np.float32)
    for index in range(4):
        base[index, :, 0] = progress
        base[index, :, 1] = 0.50 + (index - 1.5) * 0.04 + 0.05 * np.sin(math.tau * progress + index)
        base[index, :, 2] = 1.0 / (POINTS - 1)
        base[index, 0, 2] = 0.0
        base[index, 0, 3] = 1.0
        base[index, :, 4] = 1.0
    labels = np.asarray([2, 2, 2, 2], dtype=np.int64)
    writers = ["a", "b", "c", "d"]
    bank = build_dtw_bank(base, labels, 4, 7, writer_groups=writers)
    assert len(bank.features) == 4 and np.all(bank.labels == 2)
    assert np.array_equal(bank.features[:, :, 2:], np.repeat(base[:1, :, 2:], 4, axis=0))
    tensor = torch.from_numpy(bank.features)
    physical = simulate_pen_physics(tensor, LIGHT_PHYSICS, torch.Generator().manual_seed(9))
    assert physical.shape == tensor.shape
    assert torch.equal(physical[:, :, 2:], tensor[:, :, 2:])
    assert not torch.equal(physical[:, :, :2], tensor[:, :, :2])
    assert asdict(NO_PHYSICS)["name"] == "none"


if __name__ == "__main__":
    _self_test()
    print('{"self_test":"pass"}')
