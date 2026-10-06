#!/usr/bin/env python3
"""v2와 v3 사이의 최소 가중치 이동으로 보수적 HWR v4 후보를 선택한다."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from calibrate_project_punctuation_v1 import _direct_rows, _writer_split_indices
from character_tensor_v1 import CHANNELS, POINTS, ROOT
from train_character_classifier_v1 import (
    InkClassifierV1,
    _dataset,
    apply_input_mode,
    input_contract,
    load_vocabs,
    prepare_cache,
)


SCHEMA = "aiflow-commercial-hwr-stability/v4"
LOO_SCHEMA = "aiflow-commercial-hwr-stability-writer-loo/v4"
BASE_SCHEMA = "aiflow-commercial-hwr-cleanroom-physics/v2"
BASE_LOO_SCHEMA = "aiflow-commercial-hwr-cleanroom-physics-writer-loo/v2"
CANDIDATE_SCHEMA = "aiflow-commercial-hwr-cleanroom-profiled/v3"
CANDIDATE_LOO_SCHEMA = "aiflow-commercial-hwr-cleanroom-profiled-writer-loo/v3"
DEFAULT_CANONICAL = ROOT / "datasets" / "normalized" / "v1"
DEFAULT_CACHE = ROOT / "artifacts" / "unified_head_20260813" / "unified_math_8ep_full" / "cache"
DEFAULT_DIRECT = Path(
    r"D:\AIFlow-Workspace\PrivateData\math-ink-data-collector\derived"
    r"\hwr-head-calibration-20260819-r1\project_owned_ownership_eval_95.jsonl.gz"
)
DEFAULT_BASE = (
    ROOT / "artifacts" / "commercial_hwr_cleanroom_physics_20260823_r1_shadow"
    / "commercial_hwr_cleanroom_physics_checkpoint.pt"
)
DEFAULT_BASE_LOO = (
    ROOT / "artifacts" / "commercial_hwr_cleanroom_physics_20260823_r1_shadow"
    / "selected_writer_loo_models.pt"
)
DEFAULT_CANDIDATE = (
    ROOT / "artifacts" / "commercial_hwr_cleanroom_profiled_20260823_r1_shadow"
    / "commercial_hwr_cleanroom_profiled_checkpoint.pt"
)
DEFAULT_CANDIDATE_LOO = (
    ROOT / "artifacts" / "commercial_hwr_cleanroom_profiled_20260823_r1_shadow"
    / "selected_writer_loo_models.pt"
)
DEFAULT_OUTPUT = ROOT / "artifacts" / "commercial_hwr_stability_20260823_r3_shadow"
FAMILIES = {
    "vertical_slash": ("1", "|", "/"),
    "circle": ("0", "O", "o"),
    "cross": ("x", "\\times"),
    "descender": ("g", "q", "y"),
}
FORBIDDEN_SELECTION_MARKERS = ("30_noncommercial_evaluation", "crohme", "mathwriting", "fresh_acceptance")


def _assert_cli_clean() -> None:
    """평가 전용 데이터 경로가 v4 선택 인자로 들어오는 것을 시작 전에 차단한다."""

    argument_text = " ".join(sys.argv[1:]).replace("\\", "/").lower()
    found = [marker for marker in FORBIDDEN_SELECTION_MARKERS if marker in argument_text]
    if found:
        raise ValueError(f"evaluation-only marker in v4 selection arguments: {found}")


_assert_cli_clean()


def _d_path(path: Path, kind: str, *, must_exist: bool = True) -> Path:
    """평가·출력 경로를 D:로 제한하고 금지된 데이터 이름을 차단한다."""

    resolved = path.resolve()
    normalized = str(resolved).replace("\\", "/").lower()
    found = [marker for marker in FORBIDDEN_SELECTION_MARKERS if marker in normalized]
    if found:
        raise ValueError(f"evaluation-only marker in {kind}: {found}")
    if resolved.drive.upper() != "D:":
        raise ValueError(f"{kind} must remain on D:: {resolved}")
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"missing {kind}: {resolved}")
    return resolved


def _sha256(path: Path) -> str:
    """큰 산출물도 메모리에 올리지 않고 SHA-256을 계산한다."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _writer_fingerprint(writer: str) -> str:
    """writer 문자열을 내보내지 않고 분리 검증용 짧은 지문으로 바꾼다."""

    return hashlib.sha256(f"aiflow-cleanroom-v3:{writer}".encode("utf-8")).hexdigest()[:20]


def _training_guard() -> dict:
    """추가 학습이 없는 v4 선택의 평가 데이터 사용량을 명시적으로 0으로 기록한다."""

    return {
        "schema": "aiflow-training-data-guard/v1",
        "policy": "CROHME and MathWriting are validation-only",
        "forbidden_markers": ["30_noncommercial_evaluation", "crohme", "mathwriting"],
        "admitted_sources": {},
        "crohme_rows": 0,
        "mathwriting_rows": 0,
        "crohme_gradient_updates": 0,
        "mathwriting_gradient_updates": 0,
        "total_gradient_updates": 0,
        "total_gradient_updates_recorded": True,
        "passed": True,
    }


def _json_lines(path: Path) -> list[dict]:
    """UTF-8 JSONL 또는 JSONL.GZ를 행 순서 그대로 읽는다."""

    opener = gzip.open if path.suffix == ".gz" else path.open
    with opener(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _load_checkpoint(path: Path, labels: list[str], expected_schema: str) -> tuple[dict, dict]:
    """단일 372-class·uniform-time 체크포인트 계약과 상태 키를 검증한다."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    report = payload.get("report", {})
    if (
        payload.get("schema") != expected_schema
        or payload.get("math_labels") != labels
        or payload.get("auxiliary_labels") != []
        or report.get("input_contract", {}).get("observed_channel_mode") != "uniform-time"
    ):
        raise ValueError(f"checkpoint contract mismatch: {path}")
    state = {key: value.detach().cpu().clone() for key, value in payload["state_dict"].items()}
    model = InkClassifierV1(len(labels), 0)
    model.load_state_dict(state, strict=True)
    return state, report


def _load_loo(path: Path, labels: list[str], expected_schema: str) -> dict[str, dict[str, torch.Tensor]]:
    """writer-LOO 상태 묶음의 스키마·라벨·모델 키를 검증한다."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    states = payload.get("states_by_held_writer", {})
    if payload.get("schema") != expected_schema or payload.get("math_labels") != labels or len(states) < 2:
        raise ValueError(f"writer-LOO checkpoint contract mismatch: {path}")
    expected = set(InkClassifierV1(len(labels), 0).state_dict())
    output = {}
    for writer, state in states.items():
        if set(state) != expected:
            raise ValueError(f"writer-LOO state key mismatch: {writer}")
        output[str(writer)] = {key: value.detach().cpu().clone() for key, value in state.items()}
    return output


def _interpolate_state(base: dict[str, torch.Tensor], candidate: dict[str, torch.Tensor], alpha: float) -> dict[str, torch.Tensor]:
    """동일 계보의 두 모델을 가중치 공간에서 선형 보간한다."""

    if set(base) != set(candidate):
        raise ValueError("checkpoint state keys differ")
    output = {}
    for key, left in base.items():
        right = candidate[key]
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError(f"checkpoint tensor contract differs: {key}")
        if left.is_floating_point():
            output[key] = torch.lerp(left, right, alpha)
        else:
            if not torch.equal(left, right):
                raise ValueError(f"non-floating checkpoint tensor differs: {key}")
            output[key] = left.clone()
    return output


def _model(state: dict[str, torch.Tensor], labels: list[str], device: torch.device) -> InkClassifierV1:
    """보간 상태를 추론 전용 단일 헤드 모델로 복원한다."""

    model = InkClassifierV1(len(labels), 0).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def _batch(features: np.ndarray, indices: np.ndarray, device: torch.device) -> torch.Tensor:
    """선택된 원시 특징행에 uniform-time 입력 계약을 적용한다."""

    values = np.asarray(features[indices], dtype=np.float32)
    return torch.as_tensor(apply_input_mode(values, "uniform-time"), device=device)


@torch.inference_mode()
def _logits(
    model: InkClassifierV1,
    features: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    """메모리를 제한한 배치 단위로 372-class logit을 계산한다."""

    output = []
    for start in range(0, len(indices), batch_size):
        selected = indices[start:start + batch_size]
        output.append(model(_batch(features, selected, device), "math").cpu())
    return torch.cat(output)


def _ece(confidence: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    """동일 폭 confidence bin으로 expected calibration error를 계산한다."""

    value = 0.0
    for lower in np.linspace(0.0, 1.0, bins + 1)[:-1]:
        upper = lower + 1.0 / bins
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            value += float(mask.mean()) * abs(float(confidence[mask].mean()) - float(correct[mask].mean()))
    return value


def _metrics(rows: list[dict], labels: list[str]) -> dict:
    """문자·writer·클래스의 Top-1·Top-5·macro·calibration 지표를 계산한다."""

    if not rows:
        raise ValueError("cannot score an empty prediction set")
    truth = np.asarray([row["truth_index"] for row in rows], dtype=np.int64)
    top = np.asarray([row["top_indices"] for row in rows], dtype=np.int64)
    confidence = np.asarray([row["confidence"] for row in rows], dtype=np.float64)
    correct = top[:, 0] == truth
    top5 = np.any(top == truth[:, None], axis=1)
    writer_scores = []
    for writer in sorted({row["writer"] for row in rows}):
        mask = np.asarray([row["writer"] == writer for row in rows])
        writer_scores.append({
            "writer": writer,
            "records": int(mask.sum()),
            "top1": float(correct[mask].mean()),
            "top5": float(top5[mask].mean()),
        })
    by_label = {}
    for index in sorted(set(truth.tolist())):
        mask = truth == index
        by_label[labels[index]] = {
            "records": int(mask.sum()),
            "top1": float(correct[mask].mean()),
            "top5": float(top5[mask].mean()),
        }
    return {
        "records": len(rows),
        "top1": float(correct.mean()),
        "top5": float(top5.mean()),
        "strict_macro_top1": float(np.mean([row["top1"] for row in by_label.values()])),
        "strict_macro_top5": float(np.mean([row["top5"] for row in by_label.values()])),
        "ece_10": _ece(confidence, correct),
        "writer_macro_top1": float(np.mean([row["top1"] for row in writer_scores])),
        "writer_macro_top5": float(np.mean([row["top5"] for row in writer_scores])),
        "worst_writer_top1": min(row["top1"] for row in writer_scores),
        "worst_writer_top5": min(row["top5"] for row in writer_scores),
        "by_writer": writer_scores,
        "by_label": by_label,
        "by_family": {},
    }


def _prediction_rows(
    logits: torch.Tensor,
    truth_indices: np.ndarray,
    writers: list[str],
    metadata: list[dict],
    labels: list[str],
) -> list[dict]:
    """paired 비교와 식 exact 계산에 필요한 최소 예측 정보를 만든다."""

    probabilities = F.softmax(logits, dim=1)
    scores, top = probabilities.topk(k=5, dim=1)
    rows = []
    for truth, indices, confidence, writer, meta in zip(
        truth_indices, top.numpy(), scores.numpy(), writers, metadata, strict=True,
    ):
        rows.append({
            "record_id": str(meta["record_id"]),
            "formula_id": str(meta["formula_id"]),
            "truth_index": int(truth),
            "truth": labels[int(truth)],
            "top_indices": indices.tolist(),
            "top_labels": [labels[int(index)] for index in indices],
            "top_scores": [float(value) for value in confidence],
            "confidence": float(confidence[0]),
            "writer": _writer_fingerprint(str(writer)),
        })
    return rows


def _direct_metadata(rows: list[dict]) -> list[dict]:
    """연속된 formula_group_index의 재시작을 이용해 원래 식 경계를 복원한다."""

    output = []
    previous_writer = None
    previous_group = None
    formula_number = -1
    for row in rows:
        writer = str(row["writer_group"])
        group = int(row["formula_group_index"])
        if previous_writer is None or writer != previous_writer or group <= int(previous_group):
            formula_number += 1
        formula_id = hashlib.sha256(
            f"aiflow-stability-v4:{writer}:{formula_number}".encode("utf-8")
        ).hexdigest()[:20]
        output.append({"record_id": str(row["record_id"]), "formula_id": formula_id})
        previous_writer = writer
        previous_group = group
    return output


def _formula_metrics(rows: list[dict]) -> dict:
    """동일 source formula의 모든 글자가 맞는 exact와 Top-5 oracle을 계산한다."""

    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["formula_id"], []).append(row)
    exact = [all(row["top_indices"][0] == row["truth_index"] for row in group) for group in grouped.values()]
    oracle = [all(row["truth_index"] in row["top_indices"] for row in group) for group in grouped.values()]
    return {
        "available": True,
        "formulas": len(grouped),
        "exact": float(np.mean(exact)),
        "top5_oracle": float(np.mean(oracle)),
    }


def _family_metrics(rows: list[dict], labels: list[str]) -> dict:
    """사전에 선언한 동형·근접 형상 계열의 Top-1과 Top-5를 계산한다."""

    output = {}
    for name, family_labels in FAMILIES.items():
        indices = {labels.index(label) for label in family_labels if label in labels}
        selected = [row for row in rows if row["truth_index"] in indices]
        if selected:
            output[name] = {
                "records": len(selected),
                "top1": float(np.mean([row["top_indices"][0] == row["truth_index"] for row in selected])),
                "top5": float(np.mean([row["truth_index"] in row["top_indices"] for row in selected])),
            }
    return output


def _score(rows: list[dict], labels: list[str], *, formula_available: bool = True) -> dict:
    """기존 문자 지표에 식 exact와 확장 동형계열 지표를 결합한다."""

    score = _metrics(rows, labels)
    score["formula"] = (
        _formula_metrics(rows)
        if formula_available
        else {"available": False, "reason": "external character holdout has no formula grouping"}
    )
    score["by_family"] = _family_metrics(rows, labels)
    return score


def _paired(base: list[dict], candidate: list[dict]) -> dict:
    """같은 레코드에서 후보가 구조·회귀시킨 Top-1·Top-5 건수를 센다."""

    before = {row["record_id"]: row for row in base}
    after = {row["record_id"]: row for row in candidate}
    if set(before) != set(after):
        raise ValueError("paired prediction coverage differs")
    result = {"top1_improved": 0, "top1_regressed": 0, "top5_rescued": 0, "top5_lost": 0, "top1_changed": 0}
    changes = []
    for record_id in sorted(before):
        old, new = before[record_id], after[record_id]
        old_hit = old["top_indices"][0] == old["truth_index"]
        new_hit = new["top_indices"][0] == new["truth_index"]
        old_top5 = old["truth_index"] in old["top_indices"]
        new_top5 = new["truth_index"] in new["top_indices"]
        result["top1_improved"] += int(not old_hit and new_hit)
        result["top1_regressed"] += int(old_hit and not new_hit)
        result["top5_rescued"] += int(not old_top5 and new_top5)
        result["top5_lost"] += int(old_top5 and not new_top5)
        result["top1_changed"] += int(old["top_indices"][0] != new["top_indices"][0])
        if old["top_indices"][0] != new["top_indices"][0]:
            changes.append({
                "record_id": record_id,
                "formula_id": old["formula_id"],
                "writer": old["writer"],
                "truth": old["truth"],
                "base_top1": old["top_labels"][0],
                "candidate_top1": new["top_labels"][0],
                "base_confidence": old["confidence"],
                "candidate_confidence": new["confidence"],
            })
    return {**result, "changes": changes}


def _safe_candidate(score: dict, paired: dict, control: dict) -> tuple[bool, list[str]]:
    """writer-LOO에서 개별 회귀와 핵심 집계 회귀를 모두 차단한다."""

    reasons = []
    checks = (
        (paired["top1_regressed"] == 0, "top1_row_regression"),
        (paired["top5_lost"] == 0, "top5_row_regression"),
        (score["top1"] >= control["top1"], "pooled_top1_regression"),
        (score["top5"] >= control["top5"], "pooled_top5_regression"),
        (score["strict_macro_top1"] >= control["strict_macro_top1"], "macro_top1_regression"),
        (score["strict_macro_top5"] >= control["strict_macro_top5"], "macro_top5_regression"),
        (score["formula"]["exact"] >= control["formula"]["exact"], "formula_exact_regression"),
        (score["formula"]["top5_oracle"] >= control["formula"]["top5_oracle"], "formula_top5_regression"),
    )
    reasons.extend(reason for passed, reason in checks if not passed)
    for family, baseline in control["by_family"].items():
        current = score["by_family"].get(family)
        if current and current["top5"] < baseline["top5"]:
            reasons.append(f"{family}_top5_regression")
    return not reasons, reasons


def _select(grid: list[dict]) -> dict:
    """개선 건수를 우선하고 같은 개선이면 이동량 alpha가 작은 후보를 고른다."""

    control = grid[0]
    candidates = []
    gates = {}
    for row in grid[1:]:
        passed, reasons = _safe_candidate(row["metrics"], row["paired"], control["metrics"])
        key = f"{row['alpha']:.4f}"
        gates[key] = {"passed": passed, "reasons": reasons}
        if passed and (row["paired"]["top1_improved"] or row["paired"]["top5_rescued"]):
            candidates.append((
                row["paired"]["top1_improved"],
                row["paired"]["top5_rescued"],
                row["metrics"]["formula"]["exact"],
                row["metrics"]["strict_macro_top1"],
                row["metrics"]["top1"],
                -row["alpha"],
                row,
            ))
    if not candidates:
        return {
            "selected_alpha": 0.0,
            "changed": False,
            "reason": "no interpolated candidate improved writer-LOO without any row-level regression",
            "gates": gates,
        }
    selected = max(candidates, key=lambda item: item[:-1])[-1]
    return {
        "selected_alpha": selected["alpha"],
        "changed": True,
        "reason": "maximum regression-free writer-LOO rescue with minimum weight movement on ties",
        "gates": gates,
    }


def _external_gate(base: dict, candidate: dict, paired: dict) -> dict:
    """선택 후 고정 외부 holdout의 집계·macro·동형 Top-5 비회귀를 확인한다."""

    reasons = []
    checks = (
        (candidate["top1"] >= base["top1"], "external_top1_regression"),
        (candidate["top5"] >= base["top5"], "external_top5_regression"),
        (candidate["strict_macro_top5"] >= base["strict_macro_top5"], "external_macro_top5_regression"),
    )
    reasons.extend(reason for passed, reason in checks if not passed)
    for family, baseline in base["by_family"].items():
        current = candidate["by_family"].get(family)
        if current and current["top5"] < baseline["top5"]:
            reasons.append(f"external_{family}_top5_regression")
    return {"passed": not reasons, "reasons": reasons, "paired": {key: value for key, value in paired.items() if key != "changes"}}


def _write_predictions(path: Path, rows: list[dict]) -> None:
    """선택된 paired 예측을 재감사 가능한 UTF-8 gzip JSONL로 저장한다."""

    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _parse_alphas(value: str) -> list[float]:
    """0과 1을 포함하는 중복 없는 alpha 그리드를 검증한다."""

    values = sorted({round(float(item.strip()), 6) for item in value.split(",") if item.strip()})
    if not values or values[0] != 0.0 or values[-1] != 1.0 or any(item < 0.0 or item > 1.0 for item in values):
        raise ValueError("alphas must be unique values in [0,1] including 0 and 1")
    return values


def _self_test() -> None:
    """보간 끝점과 paired 회귀 집계의 핵심 계약을 빠르게 검증한다."""

    left = {"x": torch.tensor([0.0, 2.0]), "n": torch.tensor([3])}
    right = {"x": torch.tensor([2.0, 4.0]), "n": torch.tensor([3])}
    assert torch.equal(_interpolate_state(left, right, 0.0)["x"], left["x"])
    assert torch.equal(_interpolate_state(left, right, 1.0)["x"], right["x"])
    assert torch.equal(_interpolate_state(left, right, 0.5)["x"], torch.tensor([1.0, 3.0]))
    assert _parse_alphas("0,0.5,1") == [0.0, 0.5, 1.0]
    metadata = _direct_metadata([
        {"record_id": "a", "writer_group": "w", "formula_group_index": 0},
        {"record_id": "b", "writer_group": "w", "formula_group_index": 1},
        {"record_id": "c", "writer_group": "w", "formula_group_index": 0},
    ])
    assert metadata[0]["formula_id"] == metadata[1]["formula_id"]
    assert metadata[1]["formula_id"] != metadata[2]["formula_id"]


def main() -> int:
    """writer-LOO 선택, 외부 비회귀, 단일 v4 체크포인트 동결을 순서대로 실행한다."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path, default=DEFAULT_CANONICAL)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--direct-rows", type=Path, default=DEFAULT_DIRECT)
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--base-loo", type=Path, default=DEFAULT_BASE_LOO)
    parser.add_argument("--candidate-checkpoint", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--candidate-loo", type=Path, default=DEFAULT_CANDIDATE_LOO)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--alphas", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1")
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        print(json.dumps({"self_test": "pass", "schema": SCHEMA}))
        return 0
    if args.eval_batch_size < 1:
        parser.error("eval-batch-size must be positive")
    try:
        alphas = _parse_alphas(args.alphas)
    except ValueError as error:
        parser.error(str(error))

    canonical = _d_path(args.canonical_root, "canonical root")
    cache_dir = _d_path(args.cache_dir, "training cache")
    direct_path = _d_path(args.direct_rows, "project-owned writer-LOO rows")
    base_path = _d_path(args.base_checkpoint, "v2 base checkpoint")
    base_loo_path = _d_path(args.base_loo, "v2 base writer-LOO")
    candidate_path = _d_path(args.candidate_checkpoint, "v3 profiled checkpoint")
    candidate_loo_path = _d_path(args.candidate_loo, "v3 profiled writer-LOO")
    output = _d_path(args.output, "v4 output", must_exist=False)
    if output.exists():
        parser.error(f"refusing to overwrite output: {output}")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")

    labels, available_auxiliary = load_vocabs(canonical)
    base_state, base_report = _load_checkpoint(base_path, labels, BASE_SCHEMA)
    candidate_state, candidate_report = _load_checkpoint(candidate_path, labels, CANDIDATE_SCHEMA)
    base_loo = _load_loo(base_loo_path, labels, BASE_LOO_SCHEMA)
    candidate_loo = _load_loo(candidate_loo_path, labels, CANDIDATE_LOO_SCHEMA)
    if set(base_loo) != set(candidate_loo):
        raise ValueError("base and candidate writer-LOO coverage differs")
    cache = prepare_cache(canonical, cache_dir, labels, available_auxiliary, "unified-math")
    math_eval = _dataset(cache_dir, cache["sets"]["math_eval"], "preserve")
    direct_features, direct_target, writers, truths = _direct_rows(
        canonical, {label: index for index, label in enumerate(labels)}, direct_path,
    )
    direct_source = _json_lines(direct_path)
    if [str(row["record_id"]) for row in direct_source] != [row["record_id"] for row in truths]:
        raise ValueError("project-owned metadata order differs from tensorized rows")
    metadata = _direct_metadata(direct_source)
    direct_labels = direct_target.numpy().astype(np.int64, copy=False)
    folds = _writer_split_indices(writers)
    if set(base_loo) != {str(writer) for writer, _, _ in folds}:
        raise ValueError("writer-LOO state coverage differs from project rows")

    started = time.perf_counter()
    grid = []
    predictions_by_alpha: dict[float, list[dict]] = {}
    states_by_alpha: dict[float, dict[str, dict[str, torch.Tensor]]] = {}
    for alpha in alphas:
        rows = []
        fold_states = {}
        for writer, _train_index, held_index in folds:
            state = _interpolate_state(base_loo[str(writer)], candidate_loo[str(writer)], alpha)
            model = _model(state, labels, device)
            held = held_index.numpy()
            logits = _logits(model, direct_features, held, device, args.eval_batch_size)
            rows.extend(_prediction_rows(
                logits,
                direct_labels[held],
                [writers[index] for index in held],
                [metadata[index] for index in held],
                labels,
            ))
            fold_states[str(writer)] = state
            del model
        predictions_by_alpha[alpha] = rows
        states_by_alpha[alpha] = fold_states
        metrics = _score(rows, labels)
        paired = _paired(predictions_by_alpha[0.0], rows)
        grid.append({"alpha": alpha, "metrics": metrics, "paired": paired})
        print(json.dumps({
            "event": "stability_alpha_complete",
            "alpha": alpha,
            "top1": metrics["top1"],
            "top5": metrics["top5"],
            "formula_exact": metrics["formula"]["exact"],
            "top1_improved": paired["top1_improved"],
            "top1_regressed": paired["top1_regressed"],
        }), flush=True)

    selection = _select(grid)
    selected_alpha = float(selection["selected_alpha"])
    selected_grid = next(row for row in grid if row["alpha"] == selected_alpha)
    final_state = _interpolate_state(base_state, candidate_state, selected_alpha)
    base_model = _model(base_state, labels, device)
    final_model = _model(final_state, labels, device)
    evaluation_indices = np.arange(len(math_eval.labels), dtype=np.int64)
    evaluation_truth = np.asarray(math_eval.labels, dtype=np.int64)
    external_metadata = [
        {"record_id": f"external_{index:06d}", "formula_id": f"external_{index:06d}"}
        for index in evaluation_indices
    ]
    external_writers = ["technical_external"] * len(evaluation_indices)
    base_external_rows = _prediction_rows(
        _logits(base_model, math_eval.features, evaluation_indices, device, args.eval_batch_size),
        evaluation_truth, external_writers, external_metadata, labels,
    )
    candidate_external_rows = _prediction_rows(
        _logits(final_model, math_eval.features, evaluation_indices, device, args.eval_batch_size),
        evaluation_truth, external_writers, external_metadata, labels,
    )
    base_external = _score(base_external_rows, labels, formula_available=False)
    candidate_external = _score(candidate_external_rows, labels, formula_available=False)
    paired_external = _paired(base_external_rows, candidate_external_rows)
    external_gate = _external_gate(base_external, candidate_external, paired_external)
    frozen = bool(selection["changed"] and external_gate["passed"])

    report = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_shadow_pending_new_untouched_acceptance" if frozen else "no_safe_improvement",
        "training_performed": False,
        "gradient_updates": 0,
        "architecture_unchanged": True,
        "architecture": {
            "input": [POINTS, len(CHANNELS)],
            "channels": list(CHANNELS),
            "classes": len(labels),
            "hidden": 128,
            "transformer_blocks": 4,
            "attention_heads": 4,
            "single_head": True,
        },
        "input_contract": input_contract("uniform-time"),
        "method": {
            "kind": "single-checkpoint weight-space interpolation",
            "alphas": alphas,
            "selected_alpha": selected_alpha,
            "runtime_models": 1,
            "new_parameters": 0,
        },
        "selection_policy": {
            "writer_loo_only": True,
            "writers": len(folds),
            "glyphs": len(direct_labels),
            "formulas": selected_grid["metrics"]["formula"]["formulas"],
            "row_level_top1_regression_required": 0,
            "row_level_top5_loss_required": 0,
            "selection_used_external_holdout": False,
            "selection_used_known_fresh_acceptance": False,
            "selection_used_crohme": False,
            "selection_used_mathwriting": False,
        },
        "selection": selection,
        "writer_loo": {
            "base": grid[0]["metrics"],
            "candidate": selected_grid["metrics"],
            "paired": selected_grid["paired"],
            "grid": [
                {
                    "alpha": row["alpha"],
                    "top1": row["metrics"]["top1"],
                    "top5": row["metrics"]["top5"],
                    "strict_macro_top1": row["metrics"]["strict_macro_top1"],
                    "strict_macro_top5": row["metrics"]["strict_macro_top5"],
                    "formula_exact": row["metrics"]["formula"]["exact"],
                    "formula_top5_oracle": row["metrics"]["formula"]["top5_oracle"],
                    "paired": {key: value for key, value in row["paired"].items() if key != "changes"},
                }
                for row in grid
            ],
        },
        "external_technical_nonregression": {
            "role": "fixed post-selection technical holdout only",
            "passed": external_gate["passed"],
            "base": base_external,
            "candidate": candidate_external,
            "gate": external_gate,
        },
        "clean_room_contract": {
            "new_training_rows": 0,
            "crohme_rows": 0,
            "crohme_statistics": 0,
            "crohme_error_labels": 0,
            "known_fresh_acceptance_rows": 0,
            "known_fresh_acceptance_statistics": 0,
            "parent_v3_was_frozen_before_its_prior_evaluations": True,
            "post_evaluation_gradient_updates": 0,
            "crohme_reuse_for_v4_selection_or_scoring": False,
        },
        "training_data_guard": _training_guard(),
        "inputs": {
            "canonical_root": str(canonical),
            "cache_manifest": str((cache_dir / "cache_manifest.json").resolve()),
            "cache_manifest_sha256": _sha256(cache_dir / "cache_manifest.json"),
            "project_owned_rows": str(direct_path),
            "project_owned_rows_sha256": _sha256(direct_path),
            "base_checkpoint": str(base_path),
            "base_checkpoint_sha256": _sha256(base_path),
            "base_writer_loo": str(base_loo_path),
            "base_writer_loo_sha256": _sha256(base_loo_path),
            "candidate_checkpoint": str(candidate_path),
            "candidate_checkpoint_sha256": _sha256(candidate_path),
            "candidate_writer_loo": str(candidate_loo_path),
            "candidate_writer_loo_sha256": _sha256(candidate_loo_path),
        },
        "parent_summaries": {
            "base_status": base_report.get("status"),
            "candidate_status": candidate_report.get("status"),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "frozen_candidate": frozen,
        "product_adopted": False,
        "adoption_limit": (
            "a genuinely new writer/formula acceptance set is absent; the known 186 glyphs and CROHME "
            "cannot promote this candidate, and the context checkpoint remains bound to the prior HWR hash"
        ),
    }

    output.mkdir(parents=True)
    _write_predictions(output / "writer_loo_base_predictions.jsonl.gz", predictions_by_alpha[0.0])
    _write_predictions(output / "writer_loo_candidate_predictions.jsonl.gz", predictions_by_alpha[selected_alpha])
    checkpoint_path = None
    if frozen:
        checkpoint_path = output / "commercial_hwr_stability_checkpoint.pt"
        torch.save({
            "schema": SCHEMA,
            "state_dict": final_state,
            "math_labels": labels,
            "auxiliary_labels": [],
            "report": report,
        }, checkpoint_path)
        torch.save({
            "schema": LOO_SCHEMA,
            "math_labels": labels,
            "selected_alpha": selected_alpha,
            "base_checkpoint_sha256": _sha256(base_path),
            "candidate_checkpoint_sha256": _sha256(candidate_path),
            "states_by_held_writer": states_by_alpha[selected_alpha],
        }, output / "selected_writer_loo_models.pt")
        report["checkpoint"] = {
            "path": str(checkpoint_path.resolve()),
            "sha256": _sha256(checkpoint_path),
            "selected_alpha": selected_alpha,
        }
    (output / "stability_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({
        "event": "commercial_hwr_stability_complete",
        "output": str(output),
        "selected_alpha": selected_alpha,
        "external_gate": external_gate["passed"],
        "frozen_candidate": frozen,
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "product_adopted": False,
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
