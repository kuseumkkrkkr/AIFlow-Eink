"""OCR 의존성이 없는 버전별 online 후보 특징과 UTF-8 입력 함수."""
from __future__ import annotations

import gzip
import json
import math
from pathlib import Path
from accuracy_temporal_context_10e import KEYS as TEMPORAL_KEYS

POINT_KEYS = ("width_rel", "height_rel", "aspect_log", "path_over_diag", "direction_x",
              "direction_y", "stroke_count", "point_count_log", "center_x", "center_y")
CONTEXT_KEYS = ("previous_dx", "previous_dy", "next_dx", "next_dy")
FORMULA_GEOMETRY_KEYS = ("formula_left", "formula_top", "formula_width", "formula_height",
                         "formula_cx", "formula_cy", "formula_geometry_available")
ROLE_NAMES = ("digit", "operator", "fence", "operand", "other")
OPERATOR_TOKENS = frozenset({"+", "-", "=", "/", "<", ">", r"\times", r"\div", r"\pm",
                            r"\mp", r"\cdot", r"\ast", r"\leq", r"\geq", r"\neq", r"\approx"})
FENCE_TOKENS = frozenset({"(", ")", "[", "]", "{", "}", "|", r"\{", r"\}"})
OPERAND_TOKENS = frozenset({r"\alpha", r"\beta", r"\gamma", r"\delta", r"\epsilon",
                           r"\lambda", r"\mu", r"\pi", r"\sigma", r"\theta", r"\omega", r"\infty"})


def _json_lines(path: Path) -> list[dict]:
    """압축 여부에 맞춰 UTF-8 JSONL을 읽는다."""
    with (gzip.open if path.suffix == ".gz" else open)(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _semantic_role(token: str) -> str:
    """정답이 아닌 후보 token의 고정 문자 종류만 반환한다."""
    if token.isdigit():
        return "digit"
    if token in OPERATOR_TOKENS:
        return "operator"
    if token in FENCE_TOKENS:
        return "fence"
    return "operand" if token in OPERAND_TOKENS or token.startswith("\\") else "other"


def _load_candidates(path: Path, limit: int | None = None) -> list[dict]:
    """학습용 후보 파일을 읽되 후보 누락 정답 row도 유지한다."""
    rows = _json_lines(path)
    if limit is not None:
        allowed = set(list(dict.fromkeys(str(row["formula_id"]) for row in rows))[:limit])
        rows = [row for row in rows if str(row["formula_id"]) in allowed]
    if not rows:
        raise ValueError("empty candidate file")
    for row in rows:
        if not row["final_topk"] or len(row["final_topk"]) != len(row["final_topk_probabilities"]):
            raise ValueError("invalid candidate list")
    return rows


def _load_formula_records(path: Path) -> dict[str, dict]:
    """수식 JSONL을 원본 ID로 색인한다."""
    rows = _json_lines(path)
    result = {str(row.get("sample_id", row.get("formula_id", ""))): row for row in rows}
    if "" in result or len(result) != len(rows):
        raise ValueError("formula IDs must be nonempty and unique")
    return result


def feature_names(version: str = "formula28") -> tuple[str, ...]:
    """체크포인트에 기록할 순서가 고정된 특징 계약을 반환한다."""
    if version not in ("legacy21", "formula28", "formula40"):
        raise ValueError(f"unknown feature version: {version}")
    return (("probability", "candidate_rank") + POINT_KEYS
            + (FORMULA_GEOMETRY_KEYS if version != "legacy21" else ())
            + CONTEXT_KEYS + tuple("role_" + name for name in ROLE_NAMES)
            + (TEMPORAL_KEYS if version == "formula40" else ()))


def _row_numeric(row: dict, candidate_index: int, version: str = "formula28") -> list[float]:
    """정답이나 writer 정보 없이 legacy21 또는 formula28 특징을 생성한다."""
    feature_names(version)
    probabilities = [float(value) for value in row["final_topk_probabilities"]]
    if len(probabilities) != len(row["final_topk"]) or not 0 <= candidate_index < len(probabilities):
        raise ValueError("candidate probabilities and tokens must align")
    geometry = row.get("geometry") or {}
    context = row.get("context") or {}
    result = [probabilities[candidate_index], candidate_index / max(1, len(probabilities) - 1)]
    result.extend(float(geometry.get(key, 0.0)) for key in POINT_KEYS)
    if version in ("formula28", "formula40"):
        available = geometry.get("formula_geometry_available", 0.0)
        if available not in (0, 1, 0.0, 1.0):
            raise ValueError("formula geometry availability must be 0 or 1")
        result.extend(float(geometry.get(key, 0.0)) if available else 0.0 for key in FORMULA_GEOMETRY_KEYS)
    result.extend(float(context.get(key, 0.0)) for key in CONTEXT_KEYS)
    role = _semantic_role(str(row["final_topk"][candidate_index]))
    result.extend(float(role == name) for name in ROLE_NAMES)
    if version == "formula40":
        if any(key not in context for key in TEMPORAL_KEYS):
            raise ValueError('formula40 requires explicit source/time/overlap features and masks')
        result.extend(float(context[key]) for key in TEMPORAL_KEYS)
    if not all(math.isfinite(value) for value in result):
        raise ValueError("online candidate features must be finite")
    return result


def checkpoint_feature_version(payload: dict) -> str:
    """기존 체크포인트의 차원을 확인하고 새 계약 불일치는 거부한다."""
    size = int(payload.get("numeric_size", -1))
    version = payload.get("feature_version") or {21: "legacy21", 28: "formula28", 40: "formula40"}.get(size)
    if version is None or len(feature_names(version)) != size:
        raise ValueError("checkpoint feature version/dimension mismatch")
    if "feature_names" in payload and list(feature_names(version)) != payload["feature_names"]:
        raise ValueError("checkpoint feature order mismatch")
    return version
