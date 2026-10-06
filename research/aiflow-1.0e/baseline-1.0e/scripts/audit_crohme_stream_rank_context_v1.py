#!/usr/bin/env python3
"""Audit CROHME replay clocks, full HWR ranks, and required neural context.

The synthetic CROHME clock is selected only against timestamped project ink.
CROHME labels are never used to choose that clock or train/select a model.  The
formula context stage is mandatory in this evaluation and may only reorder the
frozen HWR Top-5 candidates.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gc
import gzip
import json
import math
from pathlib import Path
import statistics

import numpy as np
import torch

from audit_crohme_error_layers_v1 import HOMOGRAPH_FAMILIES, _pair_family
from character_tensor_v1 import ROOT, _json_lines, iter_direct_ownership_examples
from evaluate_48hz_prefix_v1 import (
    DEFAULT_PRODUCT,
    INPUT_MODE,
    _crohme_items,
    _direct_items,
    _load_model,
    _prefix_tensor,
    _sha256,
)
from evaluate_homograph_context_reranker_v1 import _decorate, _geometry
from train_character_classifier_v1 import apply_input_mode
from train_owned_formula_context_v1 import (
    decide_owned_formula_rows,
    decide_owned_formula_rows_supported_exact,
    load_owned_formula_context,
)


SCHEMA = "aiflow-crohme-stream-rank-context-audit/v1"
DEFAULT_CROHME = (
    ROOT / "datasets" / "30_noncommercial_evaluation" / "crohme2019"
    / "crohme2019" / "crohme2019" / "test"
)
DEFAULT_PREFIX = (
    ROOT / "artifacts" / "crohme2019_test_streaming_20260820_r1_shadow"
    / "crohme_prefix_predictions.jsonl.gz"
)
DEFAULT_DIRECT_PREFIX = (
    ROOT / "artifacts" / "prefix_48hz_20260814_direct"
    / "direct_prefix_predictions.jsonl.gz"
)
DEFAULT_CONTEXT = (
    ROOT / "artifacts" / "owned_formula_context_20260822_r6"
    / "owned_formula_context_product.pt"
)
CLOCKS = (
    "point_ordinal",
    "pen_down_arc",
    "global_chord_arc",
    "stroke_equal_arc",
    "ordinal_chord_hybrid_50",
)
DECILES = tuple(range(10, 101, 10))
RANK_CUTOFFS = (5, 10, 20, 50, 100, 200, 372)


def _d_path(path: Path, kind: str, *, must_exist: bool = True) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.drive.upper() != "D:":
        raise ValueError(f"{kind} must remain on D:: {resolved}")
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"missing {kind}: {resolved}")
    return resolved


def _arrays(strokes: list[dict] | list[np.ndarray]) -> list[np.ndarray]:
    output = []
    for stroke in strokes:
        value = (
            np.asarray(stroke["points"], dtype=np.float64)
            if isinstance(stroke, dict)
            else np.asarray(stroke, dtype=np.float64)
        )
        if value.ndim != 2 or value.shape[1] < 2 or not len(value):
            raise ValueError("invalid stroke array")
        output.append(value)
    if sum(len(value) for value in output) < 2:
        raise ValueError("clock requires at least two points")
    return output


def _normalize_progress(values: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    span = float(values[-1] - values[0])
    output = (values - values[0]) / span if span > 1e-12 else fallback.copy()
    output = np.maximum.accumulate(np.clip(output, 0.0, 1.0))
    output[0], output[-1] = 0.0, 1.0
    return output


def _clock_progress(strokes: list[dict] | list[np.ndarray]) -> dict[str, np.ndarray]:
    arrays = _arrays(strokes)
    xy = np.concatenate([value[:, :2] for value in arrays], axis=0)
    ordinal = np.linspace(0.0, 1.0, len(xy), dtype=np.float64)

    pen_down = []
    distance = 0.0
    for stroke_index, stroke in enumerate(arrays):
        if stroke_index:
            pen_down.append(distance)
        else:
            pen_down.append(0.0)
        for index in range(1, len(stroke)):
            distance += float(np.linalg.norm(stroke[index, :2] - stroke[index - 1, :2]))
            pen_down.append(distance)
    pen_down_progress = _normalize_progress(np.asarray(pen_down), ordinal)

    chord = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    chord_progress = _normalize_progress(chord, ordinal)

    stroke_equal = []
    stroke_count = len(arrays)
    for stroke_index, stroke in enumerate(arrays):
        local = np.r_[
            0.0,
            np.cumsum(np.linalg.norm(np.diff(stroke[:, :2], axis=0), axis=1)),
        ]
        local = _normalize_progress(local, np.linspace(0.0, 1.0, len(stroke)))
        stroke_equal.extend((stroke_index + local) / stroke_count)
    stroke_equal_progress = np.asarray(stroke_equal, dtype=np.float64)
    stroke_equal_progress[0], stroke_equal_progress[-1] = 0.0, 1.0

    return {
        "point_ordinal": ordinal,
        "pen_down_arc": pen_down_progress,
        "global_chord_arc": chord_progress,
        "stroke_equal_arc": stroke_equal_progress,
        "ordinal_chord_hybrid_50": 0.5 * (ordinal + chord_progress),
    }


def _actual_progress(strokes: list[dict] | list[np.ndarray]) -> np.ndarray:
    arrays = _arrays(strokes)
    if any(value.shape[1] < 3 for value in arrays):
        raise ValueError("actual timestamp channel is absent")
    values = np.concatenate([value[:, 2] for value in arrays])
    if np.any(np.diff(values) < -1e-9):
        raise ValueError("actual timestamps are not monotonic")
    return _normalize_progress(values, np.linspace(0.0, 1.0, len(values)))


def _record_mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    indices = np.arange(1, len(actual) - 1) if len(actual) > 2 else np.arange(len(actual))
    return float(np.mean(np.abs(actual[indices] - predicted[indices])))


def _clock_calibration(direct_rows: list[dict]) -> dict:
    records = []
    for row in direct_rows:
        actual = _actual_progress(row["strokes"])
        candidates = _clock_progress(row["strokes"])
        records.append({
            "record_id": str(row["record_id"]),
            "writer_group": str(row["writer_group"]),
            "points": len(actual),
            "mae": {name: _record_mae(actual, values) for name, values in candidates.items()},
        })
    writers = sorted({row["writer_group"] for row in records})
    methods = {}
    for name in CLOCKS:
        values = [row["mae"][name] for row in records]
        per_writer = {
            writer: float(np.mean([
                row["mae"][name] for row in records if row["writer_group"] == writer
            ]))
            for writer in writers
        }
        methods[name] = {
            "records": len(values),
            "mean_record_mae": float(np.mean(values)),
            "median_record_mae": float(np.median(values)),
            "p90_record_mae": float(np.quantile(values, 0.90)),
            "macro_writer_mae": float(np.mean(list(per_writer.values()))),
            "worst_writer_mae": float(max(per_writer.values())),
            "by_writer": per_writer,
        }
    selected = min(
        CLOCKS,
        key=lambda name: (
            methods[name]["macro_writer_mae"],
            methods[name]["mean_record_mae"],
            CLOCKS.index(name),
        ),
    )
    folds = []
    for held in writers:
        fit = [row for row in records if row["writer_group"] != held]
        chosen = min(
            CLOCKS,
            key=lambda name: (
                float(np.mean([row["mae"][name] for row in fit])),
                CLOCKS.index(name),
            ),
        )
        held_values = [row["mae"][chosen] for row in records if row["writer_group"] == held]
        folds.append({
            "held_writer_group": held,
            "selected_on_other_writers": chosen,
            "held_records": len(held_values),
            "held_mean_record_mae": float(np.mean(held_values)),
        })
    return {
        "scope": "211 project-owned timestamped glyphs; no CROHME truth used",
        "selection_objective": "lowest macro-writer interior-point normalized timestamp MAE",
        "methods": methods,
        "selected_clock": selected,
        "writer_leave_one_out_diagnostic": folds,
        "timestamp_quality": {
            "records": len(records),
            "writers": len(writers),
            "source_timestamp_available": len(records),
            "median_duration_ms": float(statistics.median(
                float(row["transform"]["duration_ms"]) for row in direct_rows
            )),
        },
    }


def _time_weighted_accuracy(
    row: dict, progress: np.ndarray,
) -> tuple[float, float, dict[str, bool], float | None]:
    points = [int(value) for value in row["prefix_points"]]
    top1 = [str(value) == str(row["label"]) for value in row["top1_tokens"]]
    top5 = [bool(value) for value in row["top5_hits"]]
    if not points or len(points) != len(top1) or max(points) > len(progress):
        raise ValueError(f"prefix/clock mismatch: {row['record_id']}")
    event_progress = np.asarray([progress[value - 1] for value in points], dtype=np.float64)
    next_progress = np.r_[event_progress[1:], 1.0]
    spans = np.maximum(0.0, next_progress - event_progress)
    top1_auc = float(np.sum(spans * np.asarray(top1, dtype=np.float64)))
    top5_auc = float(np.sum(spans * np.asarray(top5, dtype=np.float64)))
    by_decile = {}
    for decile in DECILES:
        target = decile / 100.0
        indices = np.flatnonzero(event_progress <= target + 1e-12)
        by_decile[str(decile)] = bool(top1[int(indices[-1])]) if len(indices) else False
    stable = None
    if top1[-1]:
        index = len(top1) - 1
        while index > 0 and top1[index - 1]:
            index -= 1
        stable = float(event_progress[index])
    return top1_auc, top5_auc, by_decile, stable


def _stream_metrics(
    rows: list[dict], progress_by_id: dict[str, dict[str, np.ndarray]], method: str,
) -> dict:
    top1_aucs, top5_aucs, stable = [], [], []
    decile_hits = Counter()
    for row in rows:
        progress = progress_by_id[str(row["record_id"])][method]
        top1_auc, top5_auc, by_decile, stable_progress = _time_weighted_accuracy(
            row, progress
        )
        top1_aucs.append(top1_auc)
        top5_aucs.append(top5_auc)
        decile_hits.update({key: int(value) for key, value in by_decile.items()})
        if stable_progress is not None:
            stable.append(stable_progress)
    return {
        "records": len(rows),
        "final_top1": sum(
            str(row["final_topk"][0]) == str(row["label"]) for row in rows
        ) / len(rows),
        "final_top5": sum(
            str(row["label"]) in [str(value) for value in row["final_topk"][:5]]
            for row in rows
        ) / len(rows),
        "time_weighted_top1_auc": float(np.mean(top1_aucs)),
        "time_weighted_top5_auc": float(np.mean(top5_aucs)),
        "median_stable_correct_progress": (
            float(np.median(stable)) if stable else None
        ),
        "by_progress": {
            str(value): decile_hits[str(value)] / len(rows) for value in DECILES
        },
        "pre_first_prediction_is_scored_incorrect": True,
    }


def _direct_replay_metrics(prefix_rows: list[dict]) -> dict:
    items = _direct_items()
    by_id = {str(item["record_id"]): item for item in items}
    if set(by_id) != {str(row["record_id"]) for row in prefix_rows}:
        raise ValueError("direct replay cache coverage mismatch")
    progress = {}
    for record_id, item in by_id.items():
        clocks = _clock_progress(item["strokes"])
        clocks["actual_timestamp"] = _actual_progress(item["strokes"])
        progress[record_id] = clocks
    return {
        name: _stream_metrics(prefix_rows, progress, name)
        for name in ("actual_timestamp",) + CLOCKS
    }


@torch.inference_mode()
def _score_full_ranks(
    items: list[dict], model, labels: list[str], device: torch.device,
    batch_size: int,
) -> tuple[dict[str, dict], list[dict]]:
    label_to_index = {label: index for index, label in enumerate(labels)}
    scores: dict[str, dict] = {}
    ranks = []
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        raw = np.stack([
            _prefix_tensor(item["strokes"]) for item in batch
        ]).astype(np.float32, copy=False)
        features = torch.from_numpy(apply_input_mode(raw, INPUT_MODE)).to(device)
        logits = model.math_head(model.encode(features))
        probabilities = logits.softmax(dim=1)
        top_values, top_indices = probabilities.topk(5, dim=1)
        truth_indices = torch.tensor(
            [label_to_index[str(item["label"])] for item in batch],
            dtype=torch.long, device=device,
        )
        truth_logits = logits.gather(1, truth_indices[:, None])
        truth_ranks = (logits > truth_logits).sum(dim=1) + 1
        truth_probabilities = probabilities.gather(1, truth_indices[:, None])[:, 0]
        for item, indices, values, rank, truth_probability in zip(
            batch,
            top_indices.cpu().tolist(),
            top_values.cpu().tolist(),
            truth_ranks.cpu().tolist(),
            truth_probabilities.cpu().tolist(),
            strict=True,
        ):
            record_id = str(item["record_id"])
            topk = [labels[index] for index in indices]
            scores[record_id] = {
                "final_topk": topk,
                "final_topk_probabilities": [float(value) for value in values],
            }
            ranks.append({
                "record_id": record_id,
                "formula_id": str(item["formula_id"]),
                "truth": str(item["label"]),
                "top1": topk[0],
                "top5": topk,
                "truth_rank": int(rank),
                "truth_probability": float(truth_probability),
                "homograph_family": _pair_family(str(item["label"]), topk[0]),
            })
    return scores, ranks


def _distribution(values: list[int]) -> dict:
    if not values:
        return {"count": 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "maximum": int(array.max()),
        "within_cutoff": {
            str(cutoff): {
                "count": int(np.count_nonzero(array <= cutoff)),
                "rate": float(np.mean(array <= cutoff)),
            }
            for cutoff in RANK_CUTOFFS
        },
    }


def _rank_summary(rows: list[dict]) -> dict:
    wrong = [row for row in rows if row["truth_rank"] > 1]
    outside = [row for row in wrong if row["truth_rank"] > 5]
    homograph = [row for row in wrong if row["homograph_family"] is not None]
    homograph_outside = [row for row in homograph if row["truth_rank"] > 5]
    same_family_candidate = []
    for row in outside:
        truth_families = [
            name for name, values in HOMOGRAPH_FAMILIES.items()
            if row["truth"] in values
        ]
        same_family_candidate.append(bool(
            truth_families and any(
                candidate in HOMOGRAPH_FAMILIES[truth_families[0]]
                for candidate in row["top5"]
            )
        ))
    by_truth = []
    for truth, count in Counter(row["truth"] for row in outside).most_common():
        selected = [row for row in outside if row["truth"] == truth]
        by_truth.append({
            "truth": truth,
            "outside_top5": count,
            "mean_truth_rank": float(np.mean([row["truth_rank"] for row in selected])),
            "median_truth_rank": float(np.median([row["truth_rank"] for row in selected])),
            "top1_confusions": [
                {"prediction": prediction, "count": value}
                for prediction, value in Counter(
                    row["top1"] for row in selected
                ).most_common(5)
            ],
        })
    by_family = []
    for family, count in Counter(row["homograph_family"] for row in homograph).most_common():
        selected = [row for row in homograph if row["homograph_family"] == family]
        top5 = sum(row["truth_rank"] <= 5 for row in selected)
        by_family.append({
            "family": family,
            "wrong_homograph_pairs": count,
            "truth_in_top5": top5,
            "truth_in_top5_rate": top5 / len(selected),
            "outside_top5_rank": _distribution([
                row["truth_rank"] for row in selected if row["truth_rank"] > 5
            ]),
        })
    return {
        "symbols": len(rows),
        "top1_wrong": len(wrong),
        "truth_in_top5": len(rows) - len(outside),
        "truth_in_top5_rate": (len(rows) - len(outside)) / len(rows),
        "top1_wrong_truth_in_top5": len(wrong) - len(outside),
        "top1_wrong_truth_outside_top5": len(outside),
        "all_wrong_truth_rank": _distribution([row["truth_rank"] for row in wrong]),
        "outside_top5_truth_rank": _distribution([
            row["truth_rank"] for row in outside
        ]),
        "homograph_wrong": {
            "count": len(homograph),
            "truth_in_top5": len(homograph) - len(homograph_outside),
            "truth_in_top5_rate": (
                (len(homograph) - len(homograph_outside)) / len(homograph)
                if homograph else None
            ),
            "outside_top5": len(homograph_outside),
            "outside_top5_rank": _distribution([
                row["truth_rank"] for row in homograph_outside
            ]),
            "by_family": by_family,
        },
        "outside_top5_has_same_family_candidate": {
            "eligible_truths": len(same_family_candidate),
            "count": sum(same_family_candidate),
            "rate": (
                sum(same_family_candidate) / len(same_family_candidate)
                if same_family_candidate else None
            ),
        },
        "by_truth_outside_top5": by_truth,
    }


def _candidate_rows(items: list[dict], scores: dict[str, dict]) -> list[dict]:
    saved = [
        {
            key: item[key]
            for key in ("record_id", "label", "source", "formula_id")
        } | {"final_topk": scores[str(item["record_id"])]["final_topk"]}
        for item in items
    ]
    return _decorate(saved, items, scores, _geometry(items))


def _context_metrics(
    rows: list[dict], predictions: dict[str, str], all_formula_ids: set[str],
    failed_formula_ids: set[str],
) -> dict:
    row_by_id = {str(row["record_id"]): row for row in rows}
    baseline = {str(row["record_id"]): str(row["final_topk"][0]) for row in rows}
    truth = {str(row["record_id"]): str(row["label"]) for row in rows}
    violations = [
        record_id for record_id, prediction in predictions.items()
        if prediction not in row_by_id[record_id]["final_topk"]
    ]
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["formula_id"])].append(row)
    eligible = all_formula_ids - failed_formula_ids

    def formula_hits(values: dict[str, str]) -> set[str]:
        return {
            formula_id for formula_id in eligible
            if formula_id in grouped and all(
                values[str(row["record_id"])] == str(row["label"])
                for row in grouped[formula_id]
            )
        }

    baseline_formulas = formula_hits(baseline)
    context_formulas = formula_hits(predictions)
    family_rows = [
        row for row in rows
        if any(str(row["label"]) in values for values in HOMOGRAPH_FAMILIES.values())
    ]
    by_family = {}
    for family, labels in HOMOGRAPH_FAMILIES.items():
        selected = [row for row in rows if str(row["label"]) in labels]
        if selected:
            by_family[family] = {
                "records": len(selected),
                "baseline_top1": sum(
                    baseline[str(row["record_id"])] == str(row["label"])
                    for row in selected
                ) / len(selected),
                "context_top1": sum(
                    predictions[str(row["record_id"])] == str(row["label"])
                    for row in selected
                ) / len(selected),
            }
    changed = [record_id for record_id in predictions if predictions[record_id] != baseline[record_id]]
    improved = [record_id for record_id in changed if predictions[record_id] == truth[record_id]]
    regressed = [record_id for record_id in changed if baseline[record_id] == truth[record_id]]
    return {
        "symbols": len(rows),
        "baseline_top1_hits": sum(baseline[key] == truth[key] for key in truth),
        "baseline_top1": sum(baseline[key] == truth[key] for key in truth) / len(rows),
        "context_top1_hits": sum(predictions[key] == truth[key] for key in truth),
        "context_top1": sum(predictions[key] == truth[key] for key in truth) / len(rows),
        "changed": len(changed),
        "improved": len(improved),
        "regressed": len(regressed),
        "candidate_contract_violations": len(violations),
        "homograph_truth_scope": {
            "records": len(family_rows),
            "baseline_top1": sum(
                baseline[str(row["record_id"])] == str(row["label"])
                for row in family_rows
            ) / len(family_rows),
            "context_top1": sum(
                predictions[str(row["record_id"])] == str(row["label"])
                for row in family_rows
            ) / len(family_rows),
            "by_family": by_family,
        },
        "formula_exact": {
            "eligible_formulas": len(eligible),
            "baseline_hits": len(baseline_formulas),
            "baseline_rate": len(baseline_formulas) / len(eligible),
            "context_hits": len(context_formulas),
            "context_rate": len(context_formulas) / len(eligible),
            "improved_formulas": len(context_formulas - baseline_formulas),
            "regressed_formulas": len(baseline_formulas - context_formulas),
            "protocol_formulas": len(all_formula_ids),
            "protocol_context_rate": len(context_formulas) / len(all_formula_ids),
        },
    }


def _write_gzip(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{100.0 * value:.2f}%"


def _markdown(report: dict) -> str:
    clock = report["clock_calibration"]
    selected = clock["selected_clock"]
    stream = report["crohme_streaming"]
    rank = report["full_rank"]
    neural = report["mandatory_neural_context"]["supported_exact_runtime"]
    lines = [
        "# CROHME 스트리밍 시계·전체 순위·필수 문맥 신경망 감사",
        "",
        f"- 생성 시각(UTC): `{report['generated_at']}`",
        "- CROHME 용도: 비상업 연구 진단 전용",
        "- 문맥 계약: HWR Top-5 고정, 신경망 호출 필수, 새 토큰·삭제·그룹 변경 금지",
        "",
        "## 0. 실제 사용자 시간과 가장 가까운 CROHME 합성 시계",
        "",
        "| 시계 | 전체 MAE | writer macro MAE | 최악 writer MAE |",
        "|---|---:|---:|---:|",
    ]
    for name in CLOCKS:
        item = clock["methods"][name]
        lines.append(
            f"| `{name}` | {item['mean_record_mae']:.4f} | "
            f"{item['macro_writer_mae']:.4f} | {item['worst_writer_mae']:.4f} |"
        )
    lines.extend([
        "",
        f"선택: **`{selected}`**. 직접수집 211글자·3 writer의 실제 timestamp에 대한 writer-macro MAE가 가장 낮았다. CROHME 정답은 선택에 쓰지 않았다.",
        "",
        "### 선택 시계로 다시 계산한 CROHME 스트리밍",
        "",
        f"- 최종 Top-1 / Top-5: **{_pct(stream['selected']['final_top1'])} / {_pct(stream['selected']['final_top5'])}** (시계와 무관)",
        f"- 시간가중 Top-1 / Top-5 AUC: **{_pct(stream['selected']['time_weighted_top1_auc'])} / {_pct(stream['selected']['time_weighted_top5_auc'])}**",
        f"- 안정 정답 도달 중앙값: **{_pct(stream['selected']['median_stable_correct_progress'])}**",
        "",
        "| 진행률 | Top-1 |",
        "|---:|---:|",
    ])
    for value, score in stream["selected"]["by_progress"].items():
        lines.append(f"| {value}% | {_pct(score)} |")
    outside = rank["outside_top5_truth_rank"]
    homograph = rank["homograph_wrong"]
    lines.extend([
        "",
        "## 1. 전체 372개 후보에서 정답 위치",
        "",
        f"- Top-1 오답: `{rank['top1_wrong']}`",
        f"- 그중 정답 Top-5 안/밖: `{rank['top1_wrong_truth_in_top5']}` / `{rank['top1_wrong_truth_outside_top5']}`",
        f"- Top-5 밖 정답 순위 평균/중앙값/p90/최대: **{outside['mean']:.2f} / {outside['median']:.0f} / {outside['p90']:.0f} / {outside['maximum']}**",
        f"- 동형 오답의 정답 Top-5 포함률: **{_pct(homograph['truth_in_top5_rate'])}** (`{homograph['truth_in_top5']}/{homograph['count']}`)",
        "",
        "## 2. 필수 문맥 신경망",
        "",
        f"- 문자 Top-1: **{_pct(neural['baseline_top1'])} → {_pct(neural['context_top1'])}**",
        f"- 변경/개선/회귀: `{neural['changed']} / {neural['improved']} / {neural['regressed']}`",
        f"- 평가 가능 식 exact: **{_pct(neural['formula_exact']['baseline_rate'])} → {_pct(neural['formula_exact']['context_rate'])}**",
        f"- Top-5 계약 위반: `{neural['candidate_contract_violations']}`",
        "",
        "문맥 모델은 분류기 encoder/head와 가중치를 주고받지 않는다. 분류기의 Top-5를 detection 후보로 받고, 식 순서·공간 관계를 attention으로 읽어 그 다섯 개 중 하나만 확정한다.",
        "",
        "## 판정",
        "",
        report["decision"],
        "",
        "## 한계",
        "",
        "- CROHME에는 timestamp가 없으므로 선택 시계는 실제 필기시간 복원이 아니라 직접수집과 가장 가까운 합성 근사다.",
        "- CROHME truth grouping을 사용한 문자/식 진단이며 공식 symLG/LgEval 점수가 아니다.",
        "- CROHME는 학습·시계 선택·모델 선택에 사용하지 않았고 상용 승격 근거가 아니다.",
    ])
    return "\n".join(lines) + "\n"


def _self_test() -> None:
    strokes = [
        {"points": [[0.0, 0.0, 0.0], [2.0, 0.0, 0.25], [3.0, 0.0, 0.5]]},
        {"points": [[3.0, 2.0, 0.75], [3.0, 3.0, 1.0]]},
    ]
    clocks = _clock_progress(strokes)
    assert set(clocks) == set(CLOCKS)
    for values in clocks.values():
        assert len(values) == 5 and values[0] == 0.0 and values[-1] == 1.0
        assert np.all(np.diff(values) >= -1e-12)
    assert np.allclose(_actual_progress(strokes), [0.0, 0.25, 0.5, 0.75, 1.0])
    row = {
        "record_id": "x", "label": "a", "prefix_points": [2, 3, 4, 5],
        "top1_tokens": ["b", "a", "a", "a"],
        "top5_hits": [False, True, True, True], "final_topk": ["a"],
    }
    top1, top5, deciles, stable = _time_weighted_accuracy(
        row, np.linspace(0.0, 1.0, 5)
    )
    assert math.isclose(top1, 0.5) and math.isclose(top5, 0.5)
    assert deciles["10"] is False and stable == 0.5
    print(json.dumps({"self_test": "pass", "clocks": list(CLOCKS)}))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crohme-root", type=Path, default=DEFAULT_CROHME)
    parser.add_argument("--prefix-predictions", type=Path, default=DEFAULT_PREFIX)
    parser.add_argument("--direct-prefix-predictions", type=Path, default=DEFAULT_DIRECT_PREFIX)
    parser.add_argument("--hwr-checkpoint", type=Path, default=DEFAULT_PRODUCT)
    parser.add_argument("--context-checkpoint", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--output", type=Path, required=False)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--context-batch-size", type=int, default=128)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return 0
    if args.output is None:
        parser.error("--output is required unless --self-test is used")
    if args.batch_size < 1 or args.context_batch_size < 1:
        parser.error("batch sizes must be positive")

    crohme_root = _d_path(args.crohme_root, "CROHME root")
    prefix_path = _d_path(args.prefix_predictions, "CROHME prefix predictions")
    direct_prefix_path = _d_path(
        args.direct_prefix_predictions, "direct prefix predictions"
    )
    hwr_path = _d_path(args.hwr_checkpoint, "HWR checkpoint")
    context_path = _d_path(args.context_checkpoint, "context checkpoint")
    output = _d_path(args.output, "output", must_exist=False)
    if output.exists():
        parser.error(f"refusing to overwrite output: {output}")
    output.mkdir(parents=True)
    device_name = (
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    if device_name == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    device = torch.device(device_name)

    direct_rows = list(iter_direct_ownership_examples(
        ROOT / "hf-dataset" / "data" / "formulas_valid.jsonl",
        ROOT / "hf-dataset" / "data" / "ownership_train.jsonl",
    ))
    clock_calibration = _clock_calibration(direct_rows)
    direct_prefix_rows = list(_json_lines(direct_prefix_path))
    direct_replay = _direct_replay_metrics(direct_prefix_rows)

    hwr_model, labels, _ = _load_model(hwr_path, device)
    items, coverage, all_formula_ids, failed_formula_ids = _crohme_items(
        crohme_root, set(labels)
    )
    prefix_rows = list(_json_lines(prefix_path))
    prefix_by_id = {str(row["record_id"]): row for row in prefix_rows}
    if set(prefix_by_id) != {str(item["record_id"]) for item in items}:
        raise ValueError("CROHME prefix cache coverage mismatch")
    progress_by_id = {
        str(item["record_id"]): _clock_progress(item["strokes"]) for item in items
    }
    stream_by_clock = {
        name: _stream_metrics(prefix_rows, progress_by_id, name) for name in CLOCKS
    }
    selected_clock = str(clock_calibration["selected_clock"])

    scores, rank_rows = _score_full_ranks(
        items, hwr_model, labels, device, args.batch_size
    )
    mismatches = [
        record_id for record_id, score in scores.items()
        if score["final_topk"] != [str(value) for value in prefix_by_id[record_id]["final_topk"]]
    ]
    if mismatches:
        raise ValueError(f"full-rank replay changed cached Top-5: {mismatches[:5]}")
    candidate_rows = _candidate_rows(items, scores)
    rank_summary = _rank_summary(rank_rows)
    _write_gzip(output / "crohme_test_context_candidates.jsonl.gz", candidate_rows)
    _write_gzip(
        output / "crohme_test_truth_outside_top5.jsonl.gz",
        [row for row in rank_rows if row["truth_rank"] > 5],
    )

    del hwr_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    context_model, contract, payload = load_owned_formula_context(
        context_path, hwr_path, device
    )
    eligible_formula_ids = all_formula_ids - failed_formula_ids
    eligible_context_rows = [
        row for row in candidate_rows if str(row["formula_id"]) in eligible_formula_ids
    ]
    baseline_predictions = {
        str(row["record_id"]): str(row["final_topk"][0]) for row in candidate_rows
    }
    eligible_neural_predictions, neural_audit = decide_owned_formula_rows(
        context_model, contract, payload, eligible_context_rows, device,
        args.context_batch_size,
    )
    eligible_runtime_predictions, runtime_audit = decide_owned_formula_rows_supported_exact(
        context_model, contract, payload, eligible_context_rows, device,
        args.context_batch_size,
    )
    neural_predictions = {**baseline_predictions, **eligible_neural_predictions}
    runtime_predictions = {**baseline_predictions, **eligible_runtime_predictions}
    neural_metrics = _context_metrics(
        candidate_rows, neural_predictions, all_formula_ids, failed_formula_ids
    )
    runtime_metrics = _context_metrics(
        candidate_rows, runtime_predictions, all_formula_ids, failed_formula_ids
    )
    if neural_metrics["candidate_contract_violations"] or runtime_metrics["candidate_contract_violations"]:
        raise AssertionError("mandatory context selected a token outside HWR Top-5")
    _write_gzip(
        output / "crohme_test_context_decisions.jsonl.gz",
        [
            {
                "record_id": str(row["record_id"]),
                "formula_id": str(row["formula_id"]),
                "truth": str(row["label"]),
                "hwr_top5": [str(value) for value in row["final_topk"]],
                "neural_context": neural_predictions[str(row["record_id"])],
                "supported_exact_runtime": runtime_predictions[str(row["record_id"])],
            }
            for row in candidate_rows
        ],
    )

    report = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "noncommercial_research_diagnostic_only",
        "training_performed": False,
        "device": str(device),
        "clock_calibration": clock_calibration,
        "direct_timestamp_replay": direct_replay,
        "crohme_streaming": {
            "timestamp_available": False,
            "selected_clock": selected_clock,
            "selected": stream_by_clock[selected_clock],
            "by_clock": stream_by_clock,
            "interpretation": "progress-weighted accuracy, not real-time latency",
        },
        "full_rank": rank_summary,
        "mandatory_neural_context": {
            "required": True,
            "failure_policy": "hard error if neural checkpoint is missing or incompatible",
            "classifier_gradient_coupling": False,
            "candidate_width": 5,
            "processed_complete_formula_rows": len(eligible_context_rows),
            "abstained_incomplete_formula_rows": len(candidate_rows) - len(eligible_context_rows),
            "neural_only": neural_metrics,
            "supported_exact_runtime": runtime_metrics,
            "neural_audit": neural_audit,
            "runtime_audit": runtime_audit,
            "architecture": {
                "encoder": "random-initialized BERT-like Transformer",
                "layers": 2,
                "hidden_size": 128,
                "attention_heads": 2,
                "selection": "one token from immutable HWR Top-5",
            },
        },
        "crohme_coverage": coverage,
        "artifacts": {
            "hwr_checkpoint": {"path": str(hwr_path), "sha256": _sha256(hwr_path)},
            "context_checkpoint": {"path": str(context_path), "sha256": _sha256(context_path)},
            "prefix_predictions": {"path": str(prefix_path), "sha256": _sha256(prefix_path)},
            "candidate_cache": {
                "path": str(output / "crohme_test_context_candidates.jsonl.gz"),
                "sha256": _sha256(output / "crohme_test_context_candidates.jsonl.gz"),
            },
            "outside_top5_rows": {
                "path": str(output / "crohme_test_truth_outside_top5.jsonl.gz"),
                "sha256": _sha256(output / "crohme_test_truth_outside_top5.jsonl.gz"),
            },
            "context_decisions": {
                "path": str(output / "crohme_test_context_decisions.jsonl.gz"),
                "sha256": _sha256(output / "crohme_test_context_decisions.jsonl.gz"),
            },
        },
        "decision": (
            "문맥 신경망은 구조상 필수로 고정할 수 있고 Top-5 계약도 지켰다. "
            "다만 CROHME는 반복 관찰한 비상업 진단이므로 이 결과만으로 제품 기본값을 "
            "승격하지 않으며, 새 프로젝트 소유 writer/formula acceptance가 필요하다."
        ),
        "limits": [
            "CROHME truth grouping is used; official symLG/LgEval is not implemented",
            "CROHME is excluded from training, clock selection, and checkpoint selection",
            "the 211 project-owned glyphs are repeatedly observed development evidence",
        ],
    }
    report_path = output / "stream_rank_context_audit.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    markdown_path = output / "stream_rank_context_audit.md"
    markdown_path.write_text(_markdown(report), encoding="utf-8", newline="\n")
    print(json.dumps({
        "event": "stream_rank_context_audit_complete",
        "report": str(report_path),
        "markdown": str(markdown_path),
        "selected_clock": selected_clock,
        "outside_top5": rank_summary["top1_wrong_truth_outside_top5"],
        "context_top1": runtime_metrics["context_top1"],
        "context_formula_exact": runtime_metrics["formula_exact"]["context_rate"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
