#!/usr/bin/env python3
"""Shared contracts for the AIFlow 1.0e accuracy-upgrade pipeline.

이 모듈은 학습을 실행하지 않는다. 원본 획의 좌표 복원, writer 기준 분할,
후보 마스킹 손실, 수식 문자열 비교처럼 여러 실험이 공유해야 하는 계약만
한 곳에서 검사한다. 입력 row에는 정답을 주입하지 않으며, 원본 transform이
없는 자료는 위치 문맥을 0으로 위장하지 않고 실패시킨다.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
import re
from typing import Any, Iterable, Mapping

import torch


SCHEMA = "aiflow-1.0e-accuracy-upgrade-contract/v1"
FEATURE_VERSION = "formula-coordinate-context/v1"
LATEX_NORMALIZATION_VERSION = "math-tokens-preserve-text-unicode/v3"


def _finite(value: Any, name: str) -> float:
    """수치가 유한한지 확인하고 float으로 변환한다."""
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def source_bbox(row: Mapping[str, Any]) -> dict[str, float]:
    """정규화된 문자 row에서 원본 좌표 bounding box를 읽는다."""
    transform = row.get("transform") or {}
    bbox = transform.get("bbox") or {}
    required = ("left", "top", "right", "bottom")
    if any(key not in bbox for key in required):
        raise ValueError(
            "formula-coordinate context requires transform.bbox; "
            f"record={row.get('record_id', '<unknown>')}"
        )
    values = {key: _finite(bbox[key], f"transform.bbox.{key}") for key in required}
    if values["right"] < values["left"] or values["bottom"] < values["top"]:
        raise ValueError("transform.bbox is inverted")
    values["width"] = max(values["right"] - values["left"], 1e-9)
    values["height"] = max(values["bottom"] - values["top"], 1e-9)
    values["cx"] = (values["left"] + values["right"]) / 2.0
    values["cy"] = (values["top"] + values["bottom"]) / 2.0
    return values


def formula_bounds(boxes: Iterable[Mapping[str, float]]) -> dict[str, float]:
    """여러 문자 box에서 수식 전체 원본 좌표 범위를 계산한다."""
    values = list(boxes)
    if not values:
        raise ValueError("formula bounds require at least one box")
    left = min(float(box["left"]) for box in values)
    top = min(float(box["top"]) for box in values)
    right = max(float(box["right"]) for box in values)
    bottom = max(float(box["bottom"]) for box in values)
    width = max(right - left, 1e-9)
    height = max(bottom - top, 1e-9)
    return {
        "left": left,
        "top": top,
        "right": right,
        "bottom": bottom,
        "width": width,
        "height": height,
    }


def formula_position(box: Mapping[str, float], bounds: Mapping[str, float]) -> dict[str, float]:
    """문자 원본 box를 수식 전체 기준 상대 위치로 변환한다."""
    width = max(float(bounds["width"]), 1e-9)
    height = max(float(bounds["height"]), 1e-9)
    cx = (float(box["left"]) + float(box["right"])) / 2.0
    cy = (float(box["top"]) + float(box["bottom"])) / 2.0
    return {
        "formula_left": (float(box["left"]) - float(bounds["left"])) / width,
        "formula_top": (float(box["top"]) - float(bounds["top"])) / height,
        "formula_width": float(box["width"]) / width,
        "formula_height": float(box["height"]) / height,
        "formula_cx": (cx - float(bounds["left"])) / width,
        "formula_cy": (cy - float(bounds["top"])) / height,
        "formula_geometry_available": 1.0,
    }


def relative_context(
    previous: Mapping[str, float] | None,
    current: Mapping[str, float],
    following: Mapping[str, float] | None,
    bounds: Mapping[str, float],
) -> dict[str, float | int]:
    """수식 전체 좌표에서 앞뒤 문자 상대 위치를 계산한다."""
    width = max(float(bounds["width"]), 1e-9)
    height = max(float(bounds["height"]), 1e-9)

    def delta(other: Mapping[str, float] | None) -> tuple[float, float]:
        if other is None:
            return 0.0, 0.0
        return (
            (float(other["cx"]) - float(current["cx"])) / width,
            (float(other["cy"]) - float(current["cy"])) / height,
        )

    previous_dx, previous_dy = delta(previous)
    next_dx, next_dy = delta(following)
    return {
        "previous_dx": previous_dx,
        "previous_dy": previous_dy,
        "next_dx": next_dx,
        "next_dy": next_dy,
        "formula_coordinate_context": 1,
    }


def writer_key(row: Mapping[str, Any]) -> str:
    """원본 writer 계보를 우선하여 fold leakage를 막는다."""
    value = row.get("raw_writer_group") or row.get("source_writer_group") or row.get("writer_group")
    if value in (None, ""):
        raise ValueError(f"missing writer lineage for {row.get('record_id', '<unknown>')}")
    return str(value)


def nested_writer_splits(rows: Iterable[Mapping[str, Any]], inner_folds: int = 3) -> dict[str, dict[str, list[str]]]:
    """writer 단위 outer LOO와 deterministic inner fold를 생성한다."""
    if inner_folds < 2:
        raise ValueError("inner_folds must be at least 2")
    writers = sorted({writer_key(row) for row in rows})
    if len(writers) < inner_folds + 1:
        raise ValueError("not enough writers for nested split")
    result: dict[str, dict[str, list[str]]] = {}
    for outer in writers:
        train_writers = [writer for writer in writers if writer != outer]
        folds: dict[str, list[str]] = {str(index): [] for index in range(inner_folds)}
        for index, writer in enumerate(train_writers):
            folds[str(index % inner_folds)].append(writer)
        result[outer] = {
            "outer_train_writers": train_writers,
            "inner_fold_held_writers": [folds[str(index)] for index in range(inner_folds)],
        }
    return result


def assert_outer_exclusion(rows: Iterable[Mapping[str, Any]], held_writer: str) -> None:
    """학습 rows가 outer held writer를 포함하지 않는지 검사한다."""
    forbidden = str(held_writer)
    leaked = [str(row.get("record_id", "<unknown>")) for row in rows if writer_key(row) == forbidden]
    if leaked:
        raise ValueError(f"outer writer leakage detected: {forbidden} ({len(leaked)} rows)")


def masked_candidate_kl(
    student_scores: torch.Tensor,
    teacher_scores: torch.Tensor,
    mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """padding을 제외하고 후보 축을 합산한 KL 평균을 계산한다."""
    if student_scores.shape != teacher_scores.shape or student_scores.shape != mask.shape:
        raise ValueError("student, teacher, and candidate mask shapes must match")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive")
    if mask.dtype != torch.bool:
        raise ValueError("candidate mask must be boolean")
    if not torch.isfinite(student_scores[mask]).all() or not torch.isfinite(teacher_scores[mask]).all():
        raise ValueError("valid candidate logits must be finite")
    valid = mask.any(dim=-1)
    if not bool(valid.any()):
        return student_scores[mask].sum()
    student = student_scores[valid] / temperature
    teacher = teacher_scores[valid] / temperature
    valid_mask = mask[valid]
    floor = torch.finfo(student.dtype).min
    student = student.masked_fill(~valid_mask, floor)
    teacher = teacher.masked_fill(~valid_mask, floor)
    teacher_probability = torch.softmax(teacher, dim=-1)
    teacher_log_probability = torch.log_softmax(teacher, dim=-1)
    student_log_probability = torch.log_softmax(student, dim=-1)
    difference = (teacher_log_probability - student_log_probability).masked_fill(~valid_mask, 0.0)
    values = (teacher_probability * difference).sum(dim=-1)
    return values.mean() * temperature * temperature


def normalize_latex(value: str) -> str:
    """평가용 최소 LaTeX 정규화 규칙을 적용한다."""
    text = str(value).strip()
    for left, right in ((r"\[", r"\]"), (r"\(", r"\)"), ("$$", "$$"), ("$", "$")):
        if len(text) >= len(left) + len(right) and text.startswith(left) and text.endswith(right):
            text = text[len(left):-len(right)].strip()
            break
    # control word의 종료 공백은 보존해야 '\\alpha x'가 '\\alphax'로 합쳐지지 않는다.
    tokens = re.findall(r"\\[A-Za-z]+|\\.|\s+|.", text, flags=re.DOTALL)
    output: list[str] = []
    literal_depth = 0
    pending_literal = False
    for index, token in enumerate(tokens):
        if literal_depth:
            output.append(token)
            if token == "{":
                literal_depth += 1
            elif token == "}":
                literal_depth -= 1
            continue
        if token.isspace():
            following = next((item for item in tokens[index + 1:] if not item.isspace()), "")
            if output and re.fullmatch(r"\\[A-Za-z]+", output[-1]) and re.match(r"[A-Za-z]", following):
                output.append(" ")
            continue
        if pending_literal and token == "{":
            literal_depth = 1
        pending_literal = token in (r"\text", r"\textrm", r"\textbf", r"\textit", r"\mbox", r"\operatorname")
        aliases = {'×': r'\times', '÷': r'\div', '≠': r'\neq', '≤': r'\leq', '≥': r'\geq',
                   '²': '^{2}', '³': '^{3}', r'\ne': r'\neq', r'\le': r'\leq', r'\ge': r'\geq'}
        original = token
        token = aliases.get(token, token)
        output.append(token)
        if original in aliases and re.fullmatch(r"\\[A-Za-z]+", token):
            following = next((item for item in tokens[index + 1:] if not item.isspace()), '')
            if re.match(r'[A-Za-z]', following):
                output.append(' ')
    return "".join(output)


def sha256_file(path: str) -> str:
    """파일 SHA-256을 계산한다."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    """manifest용 deterministic JSON SHA-256을 계산한다."""
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _self_test() -> None:
    """계약 모듈의 위치·분할·손실 불변식을 확인한다."""
    rows = [
        {"record_id": "a", "writer_group": "w1", "transform": {"bbox": {"left": 0, "top": 0, "right": 10, "bottom": 10}}},
        {"record_id": "b", "writer_group": "w2", "transform": {"bbox": {"left": 20, "top": 5, "right": 30, "bottom": 15}}},
        {"record_id": "c", "writer_group": "w3", "transform": {"bbox": {"left": 40, "top": 0, "right": 50, "bottom": 10}}},
        {"record_id": "d", "writer_group": "w4", "transform": {"bbox": {"left": 60, "top": 0, "right": 70, "bottom": 10}}},
    ]
    boxes = [source_bbox(row) for row in rows]
    bounds = formula_bounds(boxes)
    position = formula_position(boxes[1], bounds)
    assert position["formula_cx"] > position["formula_left"]
    context = relative_context(boxes[0], boxes[1], boxes[2], bounds)
    assert context["previous_dx"] < 0 and context["next_dx"] > 0
    splits = nested_writer_splits(rows)
    assert set(splits) == {"w1", "w2", "w3", "w4"}
    assert all(len(value["inner_fold_held_writers"]) == 3 for value in splits.values())
    student = torch.tensor([[2.0, 1.0, 0.0], [0.0, 1.0, 2.0]])
    teacher = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    mask = torch.tensor([[True, True, False], [True, True, True]])
    assert float(masked_candidate_kl(student, teacher, mask, 2.0)) >= 0.0
    assert normalize_latex(r"\[  x + 1  \]") == "x+1"


if __name__ == "__main__":
    _self_test()
    print(json.dumps({"schema": SCHEMA, "self_test": "pass"}, ensure_ascii=False))
