#!/usr/bin/env python3
"""고정 writer latent로 영숫자 전체에 일관된 합성 필체를 생성한다."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
import torch

from cleanroom_online_ink_augmentation_v2 import stroke_bounds
from cleanroom_pen_physics_v3 import (
    CLASS_PROFILES,
    MOTOR_LIGHT_PHYSICS,
    PenClassProfileV3,
    simulate_pen_physics_v3,
)
from cleanroom_trajectory_profiles_v3 import trajectory_descriptor
from train_character_classifier_v1 import InkClassifierV1


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "artifacts" / "cleanroom_alphanumeric_pen_physics_v3_20260823_r5"
DEFAULT_CHECKPOINT = (
    ROOT / "artifacts" / "commercial_hwr_cleanroom_physics_20260823_r1_shadow"
    / "commercial_hwr_cleanroom_physics_checkpoint.pt"
)
DEFAULT_OUTPUT = ROOT / "artifacts" / "cleanroom_writer_style_v4_20260823_r1"
SCHEMA = "aiflow-cleanroom-writer-style/v4"


@dataclass(frozen=True)
class WriterStyleV4:
    """한 합성 작가가 모든 문자에서 공유하는 공간·운동 성향이다."""

    writer_id: str
    slant: float
    width_scale: float
    height_scale: float
    baseline_tilt: float
    roundness: float
    bow_x: float
    bow_y: float
    terminal_hook: float
    motor_speed_scale: float
    damping_offset: float
    minimum_jerk_scale: float
    tremor_scale: float
    drift_scale: float

    def validate(self) -> None:
        """라벨 형태를 뒤집지 않는 보수적 writer latent 범위를 검증한다."""

        if not self.writer_id:
            raise ValueError("writer_id is required")
        if not -0.28 <= self.slant <= 0.28 or not -0.12 <= self.baseline_tilt <= 0.12:
            raise ValueError("writer affine tilt is outside the safe range")
        if not 0.72 <= self.width_scale <= 1.28 or not 0.78 <= self.height_scale <= 1.22:
            raise ValueError("writer aspect scale is outside the safe range")
        if not 0.0 <= self.roundness <= 0.32:
            raise ValueError("writer roundness is outside the safe range")
        if max(abs(self.bow_x), abs(self.bow_y), abs(self.terminal_hook)) > 0.035:
            raise ValueError("writer stroke warp is outside the safe range")
        if not 0.78 <= self.motor_speed_scale <= 1.22:
            raise ValueError("writer motor speed is outside the safe range")
        if not -0.12 <= self.damping_offset <= 0.12:
            raise ValueError("writer damping offset is outside the safe range")
        if not all(0.50 <= value <= 1.35 for value in (
            self.minimum_jerk_scale, self.tremor_scale, self.drift_scale,
        )):
            raise ValueError("writer motor scale is outside the safe range")


WRITER_STYLES = (
    WriterStyleV4(
        "writer_compact_upright", -0.03, 0.82, 1.10, -0.025, 0.08,
        -0.006, 0.004, -0.006, 0.92, 0.07, 0.86, 0.72, 0.68,
    ),
    WriterStyleV4(
        "writer_slanted_round", 0.22, 0.98, 1.02, 0.035, 0.28,
        0.016, 0.006, 0.014, 0.88, 0.09, 1.08, 0.62, 0.72,
    ),
    WriterStyleV4(
        "writer_wide_relaxed", 0.04, 1.22, 0.88, -0.045, 0.18,
        -0.012, 0.015, -0.012, 0.82, -0.04, 1.18, 0.82, 1.12,
    ),
    WriterStyleV4(
        "writer_narrow_quick", 0.13, 0.76, 1.16, 0.055, 0.04,
        0.008, -0.012, 0.008, 1.18, -0.06, 0.76, 1.22, 0.88,
    ),
)


def _sha256(path: Path) -> str:
    """파일의 SHA-256 지문을 계산한다."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _letterbox(xy: np.ndarray) -> np.ndarray:
    """종횡비를 보존하며 좌표를 단위 정사각형에 맞춘다."""

    low = xy.min(axis=0, keepdims=True)
    high = xy.max(axis=0, keepdims=True)
    center = (low + high) * 0.5
    extent = max(float((high - low).max()), 1.0e-7)
    return np.clip((xy - center) / extent + 0.5, 0.0, 1.0)


def _smooth(points: np.ndarray, strength: float) -> np.ndarray:
    """획 양끝을 고정한 5점 저역통과로 writer roundness를 적용한다."""

    if len(points) < 5 or strength <= 0.0:
        return points.copy()
    padded = np.pad(points, ((2, 2), (0, 0)), mode="edge")
    kernel = np.asarray([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float64) / 9.0
    filtered = np.stack([
        np.convolve(padded[:, axis], kernel, mode="valid") for axis in range(2)
    ], axis=1)
    output = (1.0 - strength) * points + strength * filtered
    output[0], output[-1] = points[0], points[-1]
    return output


def _style_once(features: np.ndarray, style: WriterStyleV4, strength: float) -> np.ndarray:
    """한 writer latent를 지정 강도로 좌표에 적용한다."""

    output = np.asarray(features, dtype=np.float32).copy()
    xy = np.asarray(output[:, :2], dtype=np.float64)
    center = xy - 0.5
    width = math.exp(strength * math.log(style.width_scale))
    height = math.exp(strength * math.log(style.height_scale))
    transformed = np.empty_like(center)
    transformed[:, 0] = width * center[:, 0] + strength * style.slant * (-center[:, 1])
    transformed[:, 1] = height * center[:, 1] + strength * style.baseline_tilt * center[:, 0]
    transformed += 0.5
    for start, end in stroke_bounds(output):
        points = transformed[start:end]
        count = len(points)
        if count < 2:
            continue
        points = _smooth(points, strength * style.roundness)
        progress = np.linspace(0.0, 1.0, count, dtype=np.float64)
        body = np.sin(math.pi * progress)
        points[:, 0] += strength * style.bow_x * body
        points[:, 1] += strength * style.bow_y * body
        tangent = points[-1] - points[max(0, count - 5)]
        norm = max(float(np.linalg.norm(tangent)), 1.0e-7)
        normal = np.asarray([-tangent[1], tangent[0]]) / norm
        terminal = np.clip((progress - 0.72) / 0.28, 0.0, 1.0) ** 2
        points += strength * style.terminal_hook * terminal[:, None] * normal[None]
        transformed[start:end] = points
    output[:, :2] = _letterbox(transformed).astype(np.float32)
    return output


def _topology(features: np.ndarray) -> tuple[int, int, int]:
    """획 수·대표 폐곡선·회전방향의 label-safe 구조를 반환한다."""

    descriptor = trajectory_descriptor(features)
    return descriptor.stroke_count, descriptor.loop_stroke, descriptor.loop_direction


def _path_length(features: np.ndarray) -> float:
    """stroke 경계를 넘는 가상 선을 제외한 전체 경로 길이를 계산한다."""

    total = 0.0
    for start, end in stroke_bounds(features):
        total += float(np.linalg.norm(np.diff(features[start:end, :2], axis=0), axis=1).sum())
    return total


def apply_writer_style(features: np.ndarray, style: WriterStyleV4) -> tuple[np.ndarray, dict]:
    """구조 게이트를 통과하는 최대 writer-style 강도를 결정한다."""

    style.validate()
    source = np.asarray(features, dtype=np.float32)
    source_topology = _topology(source)
    source_length = max(_path_length(source), 1.0e-7)
    for strength in (1.0, 0.82, 0.64, 0.46, 0.28):
        candidate = _style_once(source, style, strength)
        length_ratio = _path_length(candidate) / source_length
        rms = float(np.sqrt(np.mean(np.square(candidate[:, :2] - source[:, :2]))))
        if (
            _topology(candidate) == source_topology
            and 0.74 <= length_ratio <= 1.28
            and rms <= 0.17
        ):
            if not np.array_equal(candidate[:, 2:], source[:, 2:]):
                raise AssertionError("writer style changed non-spatial channels")
            return candidate, {
                "applied_strength": strength,
                "path_length_ratio": length_ratio,
                "rms_displacement": rms,
                "reverted": False,
            }
    return source.copy(), {
        "applied_strength": 0.0,
        "path_length_ratio": 1.0,
        "rms_displacement": 0.0,
        "reverted": True,
    }


def _writer_physics(style: WriterStyleV4):
    """writer motor latent를 v3 물리 설정에 고정 결합한다."""

    return replace(
        MOTOR_LIGHT_PHYSICS,
        name=f"motor_light:{style.writer_id}",
        natural_frequency_hz_min=MOTOR_LIGHT_PHYSICS.natural_frequency_hz_min * style.motor_speed_scale,
        natural_frequency_hz_max=MOTOR_LIGHT_PHYSICS.natural_frequency_hz_max * style.motor_speed_scale,
        damping_ratio_min=MOTOR_LIGHT_PHYSICS.damping_ratio_min + style.damping_offset,
        damping_ratio_max=MOTOR_LIGHT_PHYSICS.damping_ratio_max + style.damping_offset,
        minimum_jerk_strength=MOTOR_LIGHT_PHYSICS.minimum_jerk_strength * style.minimum_jerk_scale,
        tremor_amplitude=MOTOR_LIGHT_PHYSICS.tremor_amplitude * style.tremor_scale,
        drift_amplitude=MOTOR_LIGHT_PHYSICS.drift_amplitude * style.drift_scale,
    )


def _class_profiles(tokens: list[str], source_report: dict) -> list[PenClassProfileV3]:
    """v3에서 확정한 문자별 형태군을 writer simulation에 재사용한다."""

    features = source_report["token_features"]
    return [CLASS_PROFILES[str(features[token]["family"])] for token in tokens]


def _predict_metrics(
    model: InkClassifierV1,
    values: np.ndarray,
    labels: np.ndarray,
) -> dict:
    """동결 HWR에서 합성 작가의 Top-1/Top-5 label 보존율을 측정한다."""

    predictions = []
    with torch.no_grad():
        for start in range(0, len(values), 128):
            logits = model(torch.from_numpy(values[start:start + 128]), "math")
            predictions.append(logits.argsort(dim=1, descending=True)[:, :5].cpu().numpy())
    top5 = np.concatenate(predictions, axis=0)
    return {
        "records": len(labels),
        "top1": float(np.mean(top5[:, 0] == labels)),
        "top5": float(np.mean(np.any(top5 == labels[:, None], axis=1))),
    }


def _draw_strokes(draw, features: np.ndarray, box: tuple[int, int, int, int], color: str) -> None:
    """canonical y-down 좌표로 stroke 경계를 보존해 문자를 그린다."""

    left, top, width, height = box
    scale = min(width - 24, height - 24)
    offset_x = left + (width - scale) / 2
    offset_y = top + (height - scale) / 2
    for start, end in stroke_bounds(features):
        points = [
            (int(offset_x + x * scale), int(offset_y + y * scale))
            for x, y in features[start:end, :2]
        ]
        if len(points) == 1:
            x, y = points[0]
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
        else:
            draw.line(points, fill=color, width=3, joint="curve")


def _preview(
    path: Path,
    title: str,
    tokens: list[str],
    baseline: np.ndarray,
    writers: np.ndarray,
) -> None:
    """같은 문자를 baseline과 네 합성 작가가 쓴 결과를 나란히 그린다."""

    from PIL import Image, ImageDraw

    rows = len(tokens)
    cell_width, row_height, header = 205, 190, 78
    image = Image.new("RGB", (70 + cell_width * 5, header + rows * row_height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((16, 14), title, fill="#172033")
    headers = ("baseline", "compact", "slanted", "wide", "quick")
    for column, name in enumerate(headers):
        draw.text((95 + column * cell_width, 49), name, fill="#43516a")
    colors = ("#596579", "#2358a5", "#9a4d9d", "#b15a2c", "#167568")
    for row_index, token in enumerate(tokens):
        top = header + row_index * row_height
        draw.text((15, top + 82), token, fill="#172033")
        rows_values = [baseline[row_index]] + [writers[index, row_index] for index in range(4)]
        for column, values in enumerate(rows_values):
            left = 65 + column * cell_width
            draw.rounded_rectangle(
                (left, top + 8, left + cell_width - 12, top + row_height - 8),
                9, fill="#f7f9fc", outline="#d6dce7", width=2,
            )
            _draw_strokes(
                draw, values,
                (left + 4, top + 12, cell_width - 20, row_height - 24),
                colors[column],
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=True)


def _markdown(report: dict) -> str:
    """합성 작가 파라미터·분리도·인식 보존 결과를 문서화한다."""

    lines = [
        "# Clean-room writer-style simulator v4",
        "",
        "## 결론",
        "",
        f"- 동일 영숫자 {report['classes']}종을 서로 다른 고정 writer latent {report['writers']}명으로 생성했다.",
        f"- 생성 행: {report['generated_rows']}행; topology mismatch: {report['topology_mismatches']}행.",
        f"- writer 쌍 최소 RMS 중앙값: {report['writer_separation']['minimum_pair_median_rms']:.4f}.",
        "- 글자마다 무작위 작가를 바꾸지 않고, 한 작가의 공간·운동 파라미터를 전체 문자에 고정했다.",
        "- 학습, CROHME/MathWriting 사용, 제품 체크포인트/runtime 변경은 없다.",
        "",
        "## 합성 작가",
        "",
        "| writer | slant | 폭 | 높이 | roundness | 속도 | 감쇠 | 떨림 | 드리프트 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for style in report["writer_styles"]:
        lines.append(
            f"| {style['writer_id']} | {style['slant']:+.2f} | {style['width_scale']:.2f} | "
            f"{style['height_scale']:.2f} | {style['roundness']:.2f} | "
            f"{style['motor_speed_scale']:.2f} | {style['damping_offset']:+.2f} | "
            f"{style['tremor_scale']:.2f} | {style['drift_scale']:.2f} |"
        )
    lines.extend([
        "",
        "## HWR label 보존",
        "",
        "| writer | Top-1 | Top-5 | 스타일 원복 |",
        "|---|---:|---:|---:|",
        f"| baseline | {100*report['baseline_hwr']['top1']:.2f}% | {100*report['baseline_hwr']['top5']:.2f}% | - |",
    ])
    for writer_id, row in report["per_writer"].items():
        lines.append(
            f"| {writer_id} | {100*row['hwr']['top1']:.2f}% | {100*row['hwr']['top5']:.2f}% | "
            f"{row['style_reverted_rows']} |"
        )
    lines.extend([
        "",
        "## 해석",
        "",
        "- v3는 동일 필체의 미세 운동 변동, v4는 글자 집합 전체에 고정되는 작가 스타일 변동을 담당한다.",
        "- writer latent는 slant·aspect·roundness·bow·terminal hook과 motor 응답을 묶는다.",
        "- topology 또는 경로 길이·RMS 게이트를 넘으면 해당 문자만 스타일 강도를 낮추거나 원복한다.",
        "- 합성 작가는 실제 개인을 복제한 것이 아니라 범위가 고정된 clean-room latent다.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    """네 합성 작가를 생성하고 시각·구조·HWR 보존 감사를 저장한다."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if source.drive.upper() != "D:" or output.drive.upper() != "D:":
        parser.error("source and output must remain on D:")
    if output.exists():
        parser.error(f"refusing to overwrite output: {output}")
    source_report = json.loads((source / "generation_report.json").read_text(encoding="utf-8"))
    with np.load(source / "alphanumeric_class_aware_physics.npz", allow_pickle=False) as payload:
        source_features = np.asarray(payload["features"], dtype=np.float32)
        source_labels = np.asarray(payload["labels"], dtype=np.int64)
        source_tokens = np.asarray(payload["tokens"]).astype(str)
    first_by_token = {}
    for index, token in enumerate(source_tokens.tolist()):
        first_by_token.setdefault(token, index)
    tokens = list(dict.fromkeys(source_tokens.tolist()))
    selected = np.asarray([first_by_token[token] for token in tokens], dtype=np.int64)
    baseline = source_features[selected]
    labels = source_labels[selected]
    class_profiles = _class_profiles(tokens, source_report)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = InkClassifierV1(len(checkpoint["math_labels"]), None)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    baseline_hwr = _predict_metrics(model, baseline, labels)
    writer_outputs = []
    writer_metadata = []
    per_writer = {}
    topology_mismatches = 0
    for writer_index, style in enumerate(WRITER_STYLES):
        styled_rows, style_audit = [], []
        for row in baseline:
            styled, audit = apply_writer_style(row, style)
            styled_rows.append(styled)
            style_audit.append(audit)
        styled_np = np.stack(styled_rows).astype(np.float32, copy=False)
        physical, physics_audit = simulate_pen_physics_v3(
            torch.from_numpy(styled_np), _writer_physics(style),
            torch.Generator().manual_seed(20260825 + writer_index * 1009),
            row_profiles=class_profiles, return_diagnostics=True,
        )
        final = physical.numpy()
        if not np.array_equal(final[:, :, 2:], baseline[:, :, 2:]):
            raise AssertionError("writer simulation changed non-spatial channels")
        writer_topology_mismatches = 0
        for before, after in zip(baseline, final, strict=True):
            if _topology(before) != _topology(after):
                writer_topology_mismatches += 1
        topology_mismatches += writer_topology_mismatches
        hwr = _predict_metrics(model, final, labels)
        per_writer[style.writer_id] = {
            "hwr": hwr,
            "topology_mismatches": writer_topology_mismatches,
            "style_reverted_rows": int(sum(row["reverted"] for row in style_audit)),
            "style_strength": {
                "minimum": float(min(row["applied_strength"] for row in style_audit)),
                "median": float(np.median([row["applied_strength"] for row in style_audit])),
            },
        }
        writer_outputs.append(final)
        writer_metadata.append({
            "writer": asdict(style), "style_audit": style_audit,
            "physics_audit": physics_audit,
        })
    outputs = np.stack(writer_outputs).astype(np.float32, copy=False)

    pairwise = {}
    pair_medians = []
    for left, right in itertools.combinations(range(len(WRITER_STYLES)), 2):
        rms = np.sqrt(np.mean(np.square(outputs[left, :, :, :2] - outputs[right, :, :, :2]), axis=(1, 2)))
        key = f"{WRITER_STYLES[left].writer_id}__{WRITER_STYLES[right].writer_id}"
        pairwise[key] = {
            "median_rms": float(np.median(rms)),
            "minimum_rms": float(rms.min()),
            "maximum_rms": float(rms.max()),
        }
        pair_medians.append(float(np.median(rms)))
    gates = {
        "non_spatial_channels": bool(np.array_equal(
            outputs[:, :, :, 2:], np.repeat(baseline[None, :, :, 2:], len(WRITER_STYLES), axis=0),
        )),
        "topology": topology_mismatches == 0,
        "writer_separation": min(pair_medians) >= 0.012,
        "hwr_top5": min(row["hwr"]["top5"] for row in per_writer.values()) >= 0.88,
        "finite_unit_box": bool(
            np.isfinite(outputs).all() and outputs[:, :, :, :2].min() >= 0.0
            and outputs[:, :, :, :2].max() <= 1.0
        ),
    }
    output.mkdir(parents=True)
    npz_path = output / "synthetic_writers_alphanumeric.npz"
    np.savez_compressed(
        npz_path, features=outputs, labels=labels,
        tokens=np.asarray(tokens, dtype="<U4"),
        writer_ids=np.asarray([style.writer_id for style in WRITER_STYLES], dtype="<U32"),
    )
    metadata_path = output / "writer_metadata.json"
    metadata_path.write_text(json.dumps(writer_metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    token_index = {token: index for index, token in enumerate(tokens)}
    previews = {}
    for name, preview_tokens in {
        "digits": list("0123456789"),
        "uppercase": list("ABGHKMOQTX"),
        "lowercase": list("abcdfghijx"),
    }.items():
        indices = [token_index[token] for token in preview_tokens]
        path = output / f"preview_writer_styles_{name}.png"
        _preview(path, f"Clean-room synthetic writers v4 - {name}", preview_tokens, baseline[indices], outputs[:, indices])
        previews[name] = {"path": str(path), "sha256": _sha256(path)}
    report = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "candidate_pass" if all(gates.values()) else "candidate_rejected",
        "classes": len(tokens),
        "writers": len(WRITER_STYLES),
        "generated_rows": int(outputs.shape[0] * outputs.shape[1]),
        "shape": list(outputs.shape),
        "writer_styles": [asdict(style) for style in WRITER_STYLES],
        "baseline_hwr": baseline_hwr,
        "per_writer": per_writer,
        "writer_separation": {
            "minimum_pair_median_rms": min(pair_medians),
            "pairwise": pairwise,
        },
        "topology_mismatches": topology_mismatches,
        "gates": gates,
        "source": {"path": str(source), "npz_sha256": _sha256(source / "alphanumeric_class_aware_physics.npz")},
        "output": {"path": str(npz_path), "sha256": _sha256(npz_path)},
        "metadata": {"path": str(metadata_path), "sha256": _sha256(metadata_path)},
        "previews": previews,
        "training_performed": False,
        "crohme_rows": 0,
        "mathwriting_rows": 0,
        "product_runtime_changed": False,
        "hwr_checkpoint_changed": False,
        "context_checkpoint_changed": False,
    }
    report_path = output / "writer_style_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path = output / "WRITER_STYLE_V4.md"
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "shape": report["shape"],
        "gates": gates, "report": str(report_path),
    }, ensure_ascii=False))
    return 0 if all(gates.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
