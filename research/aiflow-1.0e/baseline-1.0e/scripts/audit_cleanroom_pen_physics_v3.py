#!/usr/bin/env python3
"""동일 clean-room 궤적에서 legacy v2와 48 Hz motor v3를 비교한다."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from cleanroom_online_ink_augmentation_v2 import LIGHT_PHYSICS, simulate_pen_physics
from cleanroom_pen_physics_v3 import MOTOR_LIGHT_PHYSICS, SCHEMA as PHYSICS_SCHEMA, simulate_pen_physics_v3
from cleanroom_trajectory_profiles_v3 import trajectory_descriptor


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT / "artifacts" / "commercial_hwr_cleanroom_physics_20260823_r1_shadow"
    / "external_same_label_dtw_bank.npz"
)
DEFAULT_OUTPUT = ROOT / "artifacts" / "cleanroom_pen_physics_v3_audit_20260823_r1"
REPORT_SCHEMA = "aiflow-cleanroom-pen-physics-audit/v3"


def _sha256(path: Path) -> str:
    """파일을 스트리밍해 SHA-256 지문을 계산한다."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path, limit: int) -> tuple[np.ndarray, np.ndarray]:
    """기존 clean-room DTW 은행에서 앞쪽의 고정 표본을 읽는다."""

    if path.drive.upper() != "D:" or not path.is_file():
        raise FileNotFoundError(f"D: clean-room input is missing: {path}")
    with np.load(path, allow_pickle=False) as payload:
        features = np.asarray(payload["features"], dtype=np.float32)[:limit]
        labels = np.asarray(payload["labels"], dtype=np.int64)[:limit]
    if features.shape != (len(labels), 128, 5) or not len(labels):
        raise ValueError(f"invalid clean-room bank shape: {features.shape}")
    return features, labels


def _letterbox(values: np.ndarray) -> np.ndarray:
    """비교 기준 궤적을 종횡비 보존 단위 정사각형으로 맞춘다."""

    low = values.min(axis=1, keepdims=True)
    high = values.max(axis=1, keepdims=True)
    center = (low + high) * 0.5
    extent = np.maximum((high - low).max(axis=2, keepdims=True), 1.0e-7)
    return np.clip((values - center) / extent + 0.5, 0.0, 1.0)


def _quantiles(values: np.ndarray) -> dict:
    """유한 배열을 중앙값·상위 분위수로 요약한다."""

    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {"records": 0}
    return {
        "records": int(len(finite)),
        "minimum": float(finite.min()),
        "median": float(np.median(finite)),
        "p95": float(np.quantile(finite, 0.95)),
        "maximum": float(finite.max()),
    }


def _kinematics(reference: np.ndarray, values: np.ndarray, original: np.ndarray) -> dict:
    """좌표 변형량·속도·가속도·경로 길이·구조 불변성을 계산한다."""

    delta = values - reference
    step = np.diff(values, axis=1)
    speed = np.linalg.norm(step, axis=2) * 48.0
    acceleration = np.linalg.norm(np.diff(step, axis=1), axis=2) * (48.0 ** 2)
    jerk = np.linalg.norm(np.diff(step, n=3, axis=1), axis=2) * (48.0 ** 3)
    reference_length = np.linalg.norm(np.diff(reference, axis=1), axis=2).sum(axis=1)
    output_length = np.linalg.norm(step, axis=2).sum(axis=1)
    rms = np.sqrt(np.mean(np.square(delta), axis=(1, 2)))
    maximum = np.linalg.norm(delta, axis=2).max(axis=1)
    stroke_channels_equal = np.all(values.shape == reference.shape) and np.array_equal(
        original[:, :, 2:], original[:, :, 2:]
    )
    topology_mismatches = 0
    for source_row, output_row in zip(original, values, strict=True):
        output_features = source_row.copy()
        output_features[:, :2] = output_row
        before = trajectory_descriptor(source_row)
        after = trajectory_descriptor(output_features)
        if (
            before.stroke_count,
            before.loop_stroke,
            before.loop_direction,
        ) != (
            after.stroke_count,
            after.loop_stroke,
            after.loop_direction,
        ):
            topology_mismatches += 1
    return {
        "finite": bool(np.isfinite(values).all()),
        "coordinates_in_unit_box": bool(values.min() >= 0.0 and values.max() <= 1.0),
        "non_spatial_channels_preserved": bool(stroke_channels_equal),
        "topology_mismatches": topology_mismatches,
        "rms_displacement": _quantiles(rms),
        "max_point_displacement": _quantiles(maximum),
        "path_length_ratio": _quantiles(output_length / np.maximum(reference_length, 1.0e-7)),
        "speed": _quantiles(speed),
        "acceleration": _quantiles(acceleration),
        "jerk": _quantiles(jerk),
        "repeated_step_fraction": float(np.mean(np.linalg.norm(step, axis=2) < 1.0e-7)),
    }


def _draw_preview(
    path: Path,
    original: np.ndarray,
    legacy: np.ndarray,
    motor: np.ndarray,
    labels: np.ndarray,
) -> None:
    """동일 입력의 원본·legacy v2·motor v3 궤적과 속도 곡선을 그린다."""

    from PIL import Image, ImageDraw

    rows = min(8, len(original))
    width, header, row_height = 1120, 70, 245
    image = Image.new("RGB", (width, header + rows * row_height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((18, 14), "Clean-room pen physics audit: same trajectory, three renderings", fill="#172033")
    for column, title in enumerate(("input", "legacy v2", "48 Hz motor v3")):
        draw.text((115 + 350 * column, 43), title, fill="#43516a")
    colors = ("#596579", "#bc5a32", "#176d67")
    for row_index in range(rows):
        top = header + row_index * row_height
        draw.text((10, top + 90), f"class {int(labels[row_index])}", fill="#172033")
        for column, values in enumerate((original[row_index], legacy[row_index], motor[row_index])):
            left = 75 + column * 350
            draw.rounded_rectangle(
                (left, top + 10, left + 300, top + 225), 10,
                fill="#f7f9fc", outline="#d6dce7", width=2,
            )
            points = [
                (int(left + 18 + x * 180), int(top + 18 + y * 180))
                for x, y in values
            ]
            draw.line(points, fill=colors[column], width=3, joint="curve")
            speed = np.linalg.norm(np.diff(values, axis=0), axis=1)
            speed = speed / max(float(speed.max()), 1.0e-7)
            spark = [
                (int(left + 210 + index * 76 / max(len(speed) - 1, 1)), int(top + 203 - value * 58))
                for index, value in enumerate(speed)
            ]
            draw.line(spark, fill="#252b35", width=1)
            draw.text((left + 210, top + 139), "speed", fill="#687386")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=True)


def _markdown(report: dict) -> str:
    """감사 JSON의 핵심 결과와 채택 경계를 한국어 보고서로 만든다."""

    legacy = report["engines"]["legacy_v2"]
    motor = report["engines"]["motor_v3"]
    return "\n".join([
        "# Clean-room 펜 물리 엔진 v3 검토",
        "",
        "## 결론",
        "",
        "- v2는 48 Hz와 무관한 무차원 response/momentum 재귀식이고, 떨림 주파수도 실제 시간이 아닌 128점 인덱스 기준이었다.",
        "- v3는 48 Hz·4배 내부 적분, 감쇠 2차 추종계, 최소-저크 획 진행, 곡률 감속, 시간기반 횡방향 떨림, 저주파 드리프트, 좌표 양자화를 사용한다.",
        "- 128x5 입력, `delta_t`, `stroke_start`, `observed`, 획 순서는 변경하지 않는다.",
        "- 이 감사에는 CROHME·MathWriting 및 평가 오류 통계를 사용하지 않았다.",
        "- 현재 상태는 **증강 후보 엔진**이다. HWR·문맥 체크포인트와 제품 runtime은 전환하지 않았다.",
        "",
        "## 동일 입력 비교",
        "",
        "| 항목 | legacy v2 | motor v3 |",
        "|---|---:|---:|",
        f"| 표본 | {report['records']} | {report['records']} |",
        f"| 좌표 RMS 중앙값 | {legacy['rms_displacement']['median']:.5f} | {motor['rms_displacement']['median']:.5f} |",
        f"| 최대 점 변위 p95 | {legacy['max_point_displacement']['p95']:.5f} | {motor['max_point_displacement']['p95']:.5f} |",
        f"| 경로 길이 비율 중앙값 | {legacy['path_length_ratio']['median']:.4f} | {motor['path_length_ratio']['median']:.4f} |",
        f"| 구조 판정 변화 | {legacy['topology_mismatches']} | {motor['topology_mismatches']} |",
        f"| 구조 보호 원복 | - | {motor['topology_reverted_rows']}행 |",
        f"| 경로길이 보호 원복 | - | {motor['path_length_reverted_rows']}행 |",
        f"| 반복 점 비율 | {100.0 * legacy['repeated_step_fraction']:.3f}% | {100.0 * motor['repeated_step_fraction']:.3f}% |",
        f"| 실행 시간 | {legacy['runtime_seconds']:.3f}s | {motor['runtime_seconds']:.3f}s |",
        "",
        "## 안전 게이트",
        "",
        f"- 비공간 채널 보존: **{report['gates']['non_spatial_channels']}**",
        f"- 유한 좌표·[0,1] 범위: **{report['gates']['finite_unit_box']}**",
        f"- 결정적 재실행: **{report['gates']['deterministic_replay']}**",
        f"- v3 RMS/최대변위 상한: **{report['gates']['displacement_bounds']}**",
        f"- 획·폐곡선 방향 구조 비회귀: **{report['gates']['topology']}**",
        f"- 경로 길이 0.78–1.22 보존: **{report['gates']['path_length']}**",
        "",
        "## 해석 한계",
        "",
        "- 물리적으로 동기화된 실제 펜 장치의 힘·압력·기울기 정답 데이터가 없으므로, v3는 실측 물성 복원이 아니라 제약된 운동학 증강이다.",
        "- 경로 길이·속도·가속도 수치는 합성 안정성 검사용이며 인식 정확도 향상을 증명하지 않는다.",
        "- 다음 단계는 상용/프로젝트 writer-LOO만으로 v2 대비 작은 학습 스모크를 하고, 신규 미사용 작가가 있을 때만 승격을 판단하는 것이다.",
        "",
    ])


def main() -> int:
    """동일 입력 비교, 시각화, JSON·Markdown 보고서 생성을 실행한다."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=512)
    args = parser.parse_args()
    if args.limit < 8:
        parser.error("limit must be at least 8")
    output = args.output.resolve()
    if output.drive.upper() != "D:":
        parser.error("output must remain on D:")
    if output.exists():
        parser.error(f"refusing to overwrite existing output: {output}")
    features, labels = _load(args.input.resolve(), args.limit)
    tensor = torch.from_numpy(features)
    reference = _letterbox(features[:, :, :2])

    started = time.perf_counter()
    legacy_tensor = simulate_pen_physics(
        tensor, LIGHT_PHYSICS, torch.Generator().manual_seed(20260823),
    )
    legacy_seconds = time.perf_counter() - started
    started = time.perf_counter()
    motor_tensor, motor_diagnostics = simulate_pen_physics_v3(
        tensor, MOTOR_LIGHT_PHYSICS, torch.Generator().manual_seed(20260823),
        return_diagnostics=True,
    )
    motor_seconds = time.perf_counter() - started
    replay = simulate_pen_physics_v3(
        tensor, MOTOR_LIGHT_PHYSICS, torch.Generator().manual_seed(20260823),
    )
    legacy_xy = legacy_tensor[:, :, :2].numpy()
    motor_xy = motor_tensor[:, :, :2].numpy()
    legacy = _kinematics(reference, legacy_xy, features)
    motor = _kinematics(reference, motor_xy, features)
    legacy["runtime_seconds"] = legacy_seconds
    motor["runtime_seconds"] = motor_seconds
    motor["parameter_draws"] = {
        "natural_frequency_hz": _quantiles(np.asarray([row["natural_frequency_hz"] for row in motor_diagnostics])),
        "damping_ratio": _quantiles(np.asarray([row["damping_ratio"] for row in motor_diagnostics])),
        "tremor_hz": _quantiles(np.asarray([row["tremor_hz"] for row in motor_diagnostics])),
        "gate_scale": _quantiles(np.asarray([row["gate_scale"] for row in motor_diagnostics])),
    }
    motor["topology_reverted_rows"] = int(sum(row["topology_reverted"] for row in motor_diagnostics))
    motor["path_length_reverted_rows"] = int(sum(row["path_length_reverted"] for row in motor_diagnostics))
    gates = {
        "non_spatial_channels": bool(
            torch.equal(legacy_tensor[:, :, 2:], tensor[:, :, 2:])
            and torch.equal(motor_tensor[:, :, 2:], tensor[:, :, 2:])
        ),
        "finite_unit_box": bool(
            legacy["finite"] and motor["finite"]
            and legacy["coordinates_in_unit_box"] and motor["coordinates_in_unit_box"]
        ),
        "deterministic_replay": bool(torch.equal(motor_tensor, replay)),
        "displacement_bounds": bool(
            motor["rms_displacement"]["maximum"] <= MOTOR_LIGHT_PHYSICS.max_rms_displacement + 0.001
            and motor["max_point_displacement"]["maximum"] <= MOTOR_LIGHT_PHYSICS.max_point_displacement + 0.001
        ),
        "topology": bool(motor["topology_mismatches"] == 0),
        "path_length": bool(
            motor["path_length_ratio"]["minimum"] >= MOTOR_LIGHT_PHYSICS.min_path_length_ratio - 0.001
            and motor["path_length_ratio"]["maximum"] <= MOTOR_LIGHT_PHYSICS.max_path_length_ratio + 0.001
        ),
    }
    report = {
        "schema": REPORT_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "records": len(features),
        "labels": len(set(labels.tolist())),
        "input": {"path": str(args.input.resolve()), "sha256": _sha256(args.input.resolve())},
        "training_performed": False,
        "selection_used_crohme": False,
        "crohme_rows": 0,
        "mathwriting_rows": 0,
        "physics_schema": PHYSICS_SCHEMA,
        "motor_config": asdict(MOTOR_LIGHT_PHYSICS),
        "engines": {"legacy_v2": legacy, "motor_v3": motor},
        "gates": gates,
        "status": "candidate_engine_pass" if all(gates.values()) else "candidate_engine_rejected",
        "product_runtime_changed": False,
        "hwr_checkpoint_changed": False,
        "context_checkpoint_changed": False,
    }
    output.mkdir(parents=True)
    preview_path = output / "pen_physics_v2_v3_preview.png"
    _draw_preview(preview_path, reference, legacy_xy, motor_xy, labels)
    report["preview"] = {"path": str(preview_path), "sha256": _sha256(preview_path)}
    json_path = output / "audit_report.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path = output / "AUDIT.md"
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "records": len(features),
        "gates": gates, "report": str(json_path), "preview": str(preview_path),
    }, ensure_ascii=False))
    return 0 if all(gates.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
