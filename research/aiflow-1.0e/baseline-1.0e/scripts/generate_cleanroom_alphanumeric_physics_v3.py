#!/usr/bin/env python3
"""실제 숫자·영문 라벨로 클래스 인지 clean-room 펜 궤적을 생성한다."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from cleanroom_online_ink_augmentation_v2 import stroke_bounds
from cleanroom_pen_physics_v3 import (
    CLASS_PROFILES,
    GENERIC_CLASS_PROFILE,
    MOTOR_LIGHT_PHYSICS,
    PenClassProfileV3,
    simulate_pen_physics_v3,
)
from cleanroom_trajectory_profiles_v3 import trajectory_descriptor
from train_character_classifier_v1 import InkClassifierV1


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT / "artifacts" / "commercial_hwr_cleanroom_physics_20260823_r1_shadow"
    / "external_same_label_dtw_bank.npz"
)
DEFAULT_CHECKPOINT = (
    ROOT / "artifacts" / "commercial_hwr_cleanroom_physics_20260823_r1_shadow"
    / "commercial_hwr_cleanroom_physics_checkpoint.pt"
)
DEFAULT_OUTPUT = ROOT / "artifacts" / "cleanroom_alphanumeric_pen_physics_v3_20260823_r1"
SCHEMA = "aiflow-cleanroom-alphanumeric-pen-physics/v3"
DEFAULT_GROUPS = {
    "digits": list("0123456789"),
    "uppercase": list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
    "lowercase": list("abcdefghijklmnopqrsuvwxyz"),
}
KNOWN_MISSING_ALPHANUMERIC = ("t",)


def _sha256(path: Path) -> str:
    """파일을 스트리밍해 SHA-256 지문을 계산한다."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _row_features(values: np.ndarray) -> dict:
    """한 궤적에서 획 수·폐곡선·직선성·짧은 획·회전량을 계산한다."""

    descriptor = trajectory_descriptor(values)
    lengths = []
    endpoints = []
    turning = []
    for start, end in stroke_bounds(values):
        points = np.asarray(values[start:end, :2], dtype=np.float64)
        if len(points) < 2:
            lengths.append(0.0)
            endpoints.append(0.0)
            turning.append(0.0)
            continue
        delta = np.diff(points, axis=0)
        step = np.linalg.norm(delta, axis=1)
        length = float(step.sum())
        lengths.append(length)
        endpoints.append(float(np.linalg.norm(points[-1] - points[0])))
        valid = delta[step > 1.0e-8]
        if len(valid) >= 2:
            angles = np.arctan2(valid[:, 1], valid[:, 0])
            change = (np.diff(angles) + math.pi) % math.tau - math.pi
            turning.append(float(np.abs(change).sum()))
        else:
            turning.append(0.0)
    total = max(sum(lengths), 1.0e-7)
    return {
        "stroke_count": descriptor.stroke_count,
        "has_loop": descriptor.loop_stroke >= 0,
        "straightness": float(sum(endpoints) / total),
        "short_stroke_fraction": float(sum(length / total < 0.12 for length in lengths) / len(lengths)),
        "turning_per_length": float(sum(turning) / total),
        "aspect_ratio": descriptor.aspect_ratio,
    }


def _class_profile(token: str, rows: list[dict]) -> tuple[PenClassProfileV3, dict]:
    """승인 clean-room 부모 통계로 문자별 운동 형태군을 결정한다."""

    stroke_count = float(np.median([row["stroke_count"] for row in rows]))
    loop_fraction = float(np.mean([row["has_loop"] for row in rows]))
    straightness = float(np.median([row["straightness"] for row in rows]))
    short_fraction = float(np.median([row["short_stroke_fraction"] for row in rows]))
    turning = float(np.median([row["turning_per_length"] for row in rows]))
    aspect = float(np.median([row["aspect_ratio"] for row in rows]))
    loop_tokens = set("0689BDOPQRabdegopq")
    stem_tokens = set("147AEFHIJKLMNTVWXYZfhijklxy")
    if token in {"i", "j"} and stroke_count >= 2.0:
        family = "dot_bearing"
    elif loop_fraction >= 0.50 or (token in loop_tokens and loop_fraction >= 0.20):
        family = "loop"
    elif token in stem_tokens and straightness >= 0.50 and loop_fraction < 0.35:
        family = "stem"
    else:
        family = "mixed"
    return CLASS_PROFILES[family], {
        "records": len(rows),
        "median_stroke_count": stroke_count,
        "loop_fraction": loop_fraction,
        "median_straightness": straightness,
        "median_short_stroke_fraction": short_fraction,
        "median_turning_per_length": turning,
        "median_aspect_ratio": aspect,
        "family": family,
    }


def _representative_indices(
    candidates: list[int],
    features: np.ndarray,
    truth_ranks: np.ndarray,
    variants: int,
) -> list[int]:
    """HWR Top-5 안의 부모 중 클래스 내 좌표 medoid에 가까운 표본을 고른다."""

    indices = np.asarray(candidates, dtype=np.int64)
    xy = np.asarray(features[indices, :, :2], dtype=np.float64)
    pairwise = np.sqrt(np.mean(np.square(xy[:, None] - xy[None, :]), axis=(2, 3)))
    medoid_distance = pairwise.mean(axis=1)
    eligible = np.flatnonzero(truth_ranks[indices] <= 5)
    pool = eligible if len(eligible) >= variants else np.arange(len(indices))
    ordered = sorted(
        pool.tolist(),
        key=lambda offset: (
            float(medoid_distance[offset]),
            int(truth_ranks[indices[offset]]),
            int(indices[offset]),
        ),
    )
    return [int(indices[offset]) for offset in ordered[:variants]]


def _draw_trajectory(draw, values: np.ndarray, left: int, top: int, color: str) -> None:
    """stroke 경계를 연결하지 않고 한 온라인 문자를 그린다."""

    for start, end in stroke_bounds(values):
        points = [
            (int(left + 20 + x * 170), int(top + 22 + y * 170))
            for x, y in values[start:end, :2]
        ]
        if len(points) == 1:
            x, y = points[0]
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
        else:
            draw.line(points, fill=color, width=4, joint="curve")


def _draw_page(
    path: Path,
    title: str,
    tokens: list[str],
    original: np.ndarray,
    generic: np.ndarray,
    aware: np.ndarray,
    profiles: list[PenClassProfileV3],
) -> None:
    """실제 문자명·형태군과 원본/일반/클래스 인지 결과를 한 장에 저장한다."""

    from PIL import Image, ImageDraw

    rows = len(tokens)
    width, header, row_height = 1080, 76, 220
    image = Image.new("RGB", (width, header + rows * row_height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((18, 14), title, fill="#172033")
    for column, label in enumerate(("clean-room parent", "generic motor", "class-aware motor")):
        draw.text((115 + 330 * column, 48), label, fill="#43516a")
    for row_index, token in enumerate(tokens):
        top = header + row_index * row_height
        draw.text((12, top + 83), token, fill="#172033")
        draw.text((12, top + 105), profiles[row_index].family, fill="#687386")
        for column, (values, color) in enumerate((
            (original[row_index], "#596579"),
            (generic[row_index], "#bc5a32"),
            (aware[row_index], "#176d67"),
        )):
            left = 70 + column * 330
            draw.rounded_rectangle(
                (left, top + 10, left + 290, top + 205), 10,
                fill="#f7f9fc", outline="#d6dce7", width=2,
            )
            _draw_trajectory(draw, values, left, top, color)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=True)


def _markdown(report: dict) -> str:
    """확정 파라미터와 문자별 특징을 표로 정리한다."""

    config = report["physics_config"]
    parameter_rows = [
        ("입력 샘플링", f"{config['sample_rate_hz']:.0f} Hz", "입력 시간품질 기준"),
        ("내부 적분", f"{config['oversample']}x", "semi-implicit Euler 안정화"),
        ("고유주파수", f"{config['natural_frequency_hz_min']:.1f}–{config['natural_frequency_hz_max']:.1f} Hz", "펜 끝 추종 응답"),
        ("감쇠비", f"{config['damping_ratio_min']:.2f}–{config['damping_ratio_max']:.2f}", "과도한 overshoot 제한"),
        ("최소-저크 혼합", f"{config['minimum_jerk_strength']:.2f}", "원래 점 진행률에 제한 혼합"),
        ("곡률 감속", f"{config['curvature_slowdown']:.2f}", "급회전 구간 감속"),
        ("떨림", f"{config['tremor_hz_min']:.1f}–{config['tremor_hz_max']:.1f} Hz / {config['tremor_amplitude']:.4f}", "획 수직 방향, 양끝 0"),
        ("드리프트", f"{config['drift_amplitude']:.4f}", "저주파 Brownian bridge"),
        ("좌표 양자화", str(config["quantization_levels"]), "13-bit 수준"),
        ("RMS/점 변위 상한", f"{config['max_rms_displacement']:.3f} / {config['max_point_displacement']:.3f}", "초과 시 강도 축소"),
        ("경로 길이", f"{config['min_path_length_ratio']:.2f}–{config['max_path_length_ratio']:.2f}", "범위 밖 행 원복"),
    ]
    lines = [
        "# 숫자·영문 class-aware 펜 물리 v3",
        "",
        "## 결과",
        "",
        f"- 실제 라벨 {report['classes']}종, 합성 {report['generated_rows']}행을 생성했다.",
        f"- 변화 행: {report['changed_rows']}행; topology 변화: {report['topology_mismatches']}행.",
        "- 클래스 번호가 아니라 `0–9`, `A–Z`, 사용 가능한 소문자 전체를 문자명으로 시각화했다.",
        f"- 현재 출력 어휘에 없는 소문자: {', '.join(f'`{token}`' for token in report['missing_output_tokens'])}.",
        "- CROHME·MathWriting 행/통계/오류 라벨은 사용하지 않았고 학습도 실행하지 않았다.",
        "",
        "## 확정 기본 파라미터",
        "",
        "| 변수 | 값 | 역할 |",
        "|---|---:|---|",
    ]
    lines.extend(f"| {name} | {value} | {role} |" for name, value, role in parameter_rows)
    lines.extend([
        "",
        "## 형태군 보정",
        "",
        "| 형태군 | 주파수 배율 | 감쇠 추가 | 최소-저크 | 곡률 | 떨림 | 드리프트 | 짧은 획 보호 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for family, profile in report["class_profiles"].items():
        lines.append(
            f"| {family} | {profile['natural_frequency_scale']:.2f} | {profile['damping_offset']:+.2f} | "
            f"{profile['minimum_jerk_scale']:.2f} | {profile['curvature_scale']:.2f} | "
            f"{profile['tremor_scale']:.2f} | {profile['drift_scale']:.2f} | "
            f"≤{profile['protect_short_stroke_points']}점 |"
        )
    lines.extend([
        "",
        "## 문자별 관측 특징",
        "",
        "| 문자 | 형태군 | 획 중앙값 | loop 비율 | 직선성 | 짧은 획 | 종횡비 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for token, row in report["token_features"].items():
        lines.append(
            f"| `{token}` | {row['family']} | {row['median_stroke_count']:.1f} | "
            f"{row['loop_fraction']:.2f} | {row['median_straightness']:.2f} | "
            f"{row['median_short_stroke_fraction']:.2f} | {row['median_aspect_ratio']:.2f} |"
        )
    lines.extend([
        "",
        "## 결정 경계",
        "",
        "- loop: 폐곡선 비율 0.50 이상, 또는 loop 의미 문자이며 관측 폐곡선 비율 0.20 이상.",
        "- dot_bearing: 실제 `i/j`이고 관측 획 중앙값이 2 이상.",
        "- stem: stem 의미 문자이며 직선성 0.50 이상, loop 비율 0.35 미만.",
        "- 나머지는 mixed. 이 분류는 라벨 정답을 바꾸지 않고 물리 강도만 낮춘다.",
        "- HWR·문맥 체크포인트와 제품 runtime은 변경하지 않았다.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    """문자 프로파일 적합, class-aware 생성, 시각화와 보고서 저장을 실행한다."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--variants", type=int, default=2)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.drive.upper() != "D:" or args.input.resolve().drive.upper() != "D:":
        parser.error("all inputs and outputs must remain on D:")
    if output.exists():
        parser.error(f"refusing to overwrite output: {output}")
    if args.variants < 1:
        parser.error("variants must be positive")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    label_names = list(checkpoint["math_labels"])
    with np.load(args.input, allow_pickle=False) as payload:
        features = np.asarray(payload["features"], dtype=np.float32)
        labels = np.asarray(payload["labels"], dtype=np.int64)
    all_tokens = [token for group in DEFAULT_GROUPS.values() for token in group]
    label_index = {label: index for index, label in enumerate(label_names)}
    missing = [token for token in all_tokens if token not in label_index]
    if missing:
        raise ValueError(f"requested alphanumeric labels are unavailable: {missing}")
    model = InkClassifierV1(len(label_names), None)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    prediction_order = []
    with torch.no_grad():
        for start in range(0, len(features), 256):
            logits = model(torch.from_numpy(features[start:start + 256]), "math")
            prediction_order.append(logits.argsort(dim=1, descending=True).cpu().numpy())
    prediction_order_np = np.concatenate(prediction_order, axis=0)
    truth_ranks = np.empty(len(features), dtype=np.int64)
    for index, label in enumerate(labels.tolist()):
        truth_ranks[index] = int(np.flatnonzero(prediction_order_np[index] == int(label))[0]) + 1

    row_features: dict[int, list[dict]] = defaultdict(list)
    by_label: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels.tolist()):
        by_label[int(label)].append(index)
        if label_names[int(label)] in all_tokens:
            row_features[int(label)].append(_row_features(features[index]))
    profiles_by_token = {}
    features_by_token = {}
    selected_indices = []
    selected_tokens = []
    row_profiles = []
    for token in all_tokens:
        index = label_index[token]
        if len(by_label[index]) < args.variants or not row_features[index]:
            raise ValueError(f"insufficient clean-room parents for token {token}")
        profile, summary = _class_profile(token, row_features[index])
        profiles_by_token[token] = profile
        features_by_token[token] = summary
        parent_indices = _representative_indices(
            by_label[index], features, truth_ranks, args.variants,
        )
        for parent_index in parent_indices:
            selected_indices.append(parent_index)
            selected_tokens.append(token)
            row_profiles.append(profile)
    selected = features[np.asarray(selected_indices, dtype=np.int64)]
    tensor = torch.from_numpy(selected)
    generic = simulate_pen_physics_v3(
        tensor, MOTOR_LIGHT_PHYSICS, torch.Generator().manual_seed(20260824),
    )
    aware, diagnostics = simulate_pen_physics_v3(
        tensor, MOTOR_LIGHT_PHYSICS, torch.Generator().manual_seed(20260824),
        row_profiles=row_profiles, return_diagnostics=True,
    )
    aware_np = aware.numpy()
    topology_mismatches = 0
    for before, after in zip(selected, aware_np, strict=True):
        candidate = before.copy()
        candidate[:, :2] = after[:, :2]
        left, right = trajectory_descriptor(before), trajectory_descriptor(candidate)
        if (left.stroke_count, left.loop_stroke, left.loop_direction) != (
            right.stroke_count, right.loop_stroke, right.loop_direction,
        ):
            topology_mismatches += 1
    if not np.array_equal(aware_np[:, :, 2:], selected[:, :, 2:]):
        raise AssertionError("class-aware generation changed non-spatial channels")
    output.mkdir(parents=True)
    npz_path = output / "alphanumeric_class_aware_physics.npz"
    np.savez_compressed(
        npz_path, features=aware_np, labels=labels[np.asarray(selected_indices)],
        tokens=np.asarray(selected_tokens, dtype="<U4"),
    )
    metadata_path = output / "generation_metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8", newline="\n") as stream:
        for token, parent_index, diagnostic in zip(
            selected_tokens, selected_indices, diagnostics, strict=True,
        ):
            stream.write(json.dumps({
                "token": token,
                "label_index": label_index[token],
                "parent_fingerprint": hashlib.sha256(
                    np.asarray(np.round(features[parent_index, :, :2], 5), dtype=np.float32).tobytes()
                ).hexdigest(),
                "parent_truth_rank": int(truth_ranks[parent_index]),
                "class_features": features_by_token[token],
                "class_profile": asdict(profiles_by_token[token]),
                "physics": diagnostic,
            }, ensure_ascii=False, sort_keys=True) + "\n")
    previews = {}
    cursor = 0
    for group, tokens in DEFAULT_GROUPS.items():
        count = len(tokens) * args.variants
        pages = []
        for page_index, page_start in enumerate(range(0, len(tokens), 10), start=1):
            page_tokens = tokens[page_start:page_start + 10]
            group_indices = [
                cursor + (page_start + index) * args.variants
                for index in range(len(page_tokens))
            ]
            preview_path = output / f"preview_{group}_{page_index}.png"
            _draw_page(
                preview_path,
                f"Class-aware clean-room pen physics v3 - {group} {page_index}",
                page_tokens,
                selected[group_indices],
                generic.numpy()[group_indices],
                aware_np[group_indices],
                [profiles_by_token[token] for token in page_tokens],
            )
            pages.append({"path": str(preview_path), "sha256": _sha256(preview_path)})
        previews[group] = pages
        cursor += count
    changed_rows = int(np.any(np.abs(aware_np[:, :, :2] - selected[:, :, :2]) > 1.0e-7, axis=(1, 2)).sum())
    report = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "classes": len(all_tokens),
        "missing_output_tokens": list(KNOWN_MISSING_ALPHANUMERIC),
        "generated_rows": len(aware_np),
        "variants_per_class": args.variants,
        "changed_rows": changed_rows,
        "topology_mismatches": topology_mismatches,
        "physics_config": asdict(MOTOR_LIGHT_PHYSICS),
        "class_profiles": {name: asdict(profile) for name, profile in CLASS_PROFILES.items()},
        "token_features": features_by_token,
        "family_counts": {
            family: sum(profile.family == family for profile in profiles_by_token.values())
            for family in CLASS_PROFILES
        },
        "representative_parent_selection": {
            "rule": "class medoid among frozen-HWR Top-5 parents; medoid fallback",
            "top1_rows": int(sum(truth_ranks[index] == 1 for index in selected_indices)),
            "top5_rows": int(sum(truth_ranks[index] <= 5 for index in selected_indices)),
        },
        "input": {"path": str(args.input.resolve()), "sha256": _sha256(args.input.resolve())},
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
    report_path = output / "generation_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path = output / "ALPHANUMERIC_PHYSICS_V3.md"
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({
        "status": "pass" if topology_mismatches == 0 else "fail",
        "classes": len(all_tokens), "rows": len(aware_np),
        "changed_rows": changed_rows, "report": str(report_path),
    }, ensure_ascii=False))
    return 0 if topology_mismatches == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
