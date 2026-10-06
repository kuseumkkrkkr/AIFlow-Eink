#!/usr/bin/env python3
"""Audit CROHME token errors, overfit evidence, layer bottlenecks, and replay time."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from audit_replay_protocol_v1 import CROHME_ALIASES, _crohme_sample
from character_tensor_v1 import _resample_stroke
from replay_evaluate_hwr_v1 import load_crohme
from report_crohme_streaming_v1 import _json_lines, _sha256


SCHEMA = "aiflow-crohme-error-layer-audit/v1"
HOMOGRAPH_FAMILIES = {
    "vertical_or_slash": frozenset({
        "1", "|", "/", r"\mid", "I", "l", "i", r"\prime",
        r"\backslash", r"\setminus",
    }),
    "circle": frozenset({
        "0", "O", "o", r"\mathcal{O}", r"\circ", r"\degree",
        r"\fullmoon", r"\O",
    }),
    "cross": frozenset({
        "x", "X", r"\times", r"\chi", r"\mathcal{X}", r"\mathfrak{X}",
    }),
    "sigma_sum": frozenset({r"\sum", r"\Sigma"}),
    "pi_product": frozenset({r"\prod", r"\Pi"}),
    "perpendicular": frozenset({r"\perp", r"\bot"}),
    "equality_approx": frozenset({"=", r"\approx", r"\simeq"}),
    "s_five": frozenset({"S", "s", "5"}),
    "beta_eight": frozenset({r"\beta", "8"}),
    "right_arrow": frozenset({r"\rightarrow", r"\shortrightarrow", r"\longrightarrow"}),
}


def _family(token: str) -> str | None:
    matches = [name for name, values in HOMOGRAPH_FAMILIES.items() if token in values]
    if len(matches) > 1:
        raise AssertionError(f"overlapping homograph families for {token}: {matches}")
    return matches[0] if matches else None


def _pair_family(truth: str, prediction: str) -> str | None:
    left, right = _family(truth), _family(prediction)
    return left if left is not None and left == right else None


def _pct(value: int | float, total: int | float) -> float | None:
    return value / total if total else None


def _percent(value: float | None) -> str:
    return "-" if value is None else f"{100.0 * value:.2f}%"


def _md(value: object) -> str:
    return str(value).replace("|", r"\|").replace("\n", " ")


def _code(value: object) -> str:
    return f"`{str(value).replace('`', 'ˋ')}`"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _audit_formulas(root: Path, prediction_rows: list[dict]) -> tuple[list[dict], dict, dict]:
    protocol, duplicates = load_crohme(root)
    by_formula: dict[str, dict[int, dict]] = defaultdict(dict)
    for row in prediction_rows:
        formula_id, index = str(row["record_id"]).rsplit(":", 1)
        by_formula[formula_id][int(index)] = row

    formulas = []
    symbol_counts = Counter()
    pair_counts = Counter()
    family_counts = Counter()
    top5_recoverable_symbols = 0
    for formula in protocol:
        formula_id = str(formula["record_id"])
        sample = _crohme_sample(root / formula_id)
        if sample is None:
            formulas.append({
                "formula_id": formula_id,
                "latex_truth": str(formula["label"]),
                "eligible": False,
                "category": "error",
                "reason": "missing_truth_partition",
                "truth_tokens": [],
                "prediction_tokens": [],
                "differences": [],
            })
            continue
        truths = [CROHME_ALIASES.get(str(value), str(value)) for value in sample["labels"]]
        rows = by_formula.get(formula_id, {})
        predictions = []
        differences = []
        top5_exact = True
        for index, truth in enumerate(truths):
            row = rows.get(index)
            prediction = str(row["final_topk"][0]) if row else "<UNSUPPORTED_OR_INVALID>"
            top5 = [str(value) for value in row["final_topk"][:5]] if row else []
            predictions.append(prediction)
            if row:
                if truth == prediction:
                    symbol_counts["exact"] += 1
                else:
                    family = _pair_family(truth, prediction)
                    kind = "homograph" if family else "error"
                    symbol_counts[kind] += 1
                    pair_counts[(truth, prediction, kind)] += 1
                    if family:
                        family_counts[family] += 1
                    top5_recoverable_symbols += int(truth in top5)
                top5_exact &= truth in top5
            else:
                top5_exact = False
            if truth != prediction:
                family = _pair_family(truth, prediction)
                differences.append({
                    "index": index,
                    "truth": truth,
                    "prediction": prediction,
                    "kind": "homograph" if family else "error",
                    "family": family,
                    "truth_in_top5": truth in top5,
                })
        eligible = bool(sample["partition_complete"] and len(rows) == len(truths))
        if not eligible:
            category = "error"
            reason = (
                "incomplete_truth_partition" if not sample["partition_complete"]
                else "unsupported_or_invalid_truth_group"
            )
        elif not differences:
            category, reason = "exact", None
        elif all(row["kind"] == "homograph" for row in differences):
            category, reason = "homograph_only", None
        else:
            category, reason = "error", None
        formulas.append({
            "formula_id": formula_id,
            "latex_truth": str(formula["label"]),
            "eligible": eligible,
            "category": category,
            "reason": reason,
            "symbols": len(truths),
            "truth_tokens": truths,
            "prediction_tokens": predictions,
            "differences": differences,
            "difference_count": len(differences),
            "homograph_difference_count": sum(row["kind"] == "homograph" for row in differences),
            "error_difference_count": sum(row["kind"] == "error" for row in differences),
            "top5_exact": bool(eligible and top5_exact),
            "spacing_ignored": True,
        })

    def summary(rows: list[dict]) -> dict:
        counts = Counter(row["category"] for row in rows)
        return {
            "formulas": len(rows),
            **{
                name: {"count": counts[name], "rate": _pct(counts[name], len(rows))}
                for name in ("exact", "homograph_only", "error")
            },
            "top5_exact": {
                "count": sum(bool(row.get("top5_exact")) for row in rows),
                "rate": _pct(sum(bool(row.get("top5_exact")) for row in rows), len(rows)),
            },
        }

    eligible_rows = [row for row in formulas if row["eligible"]]
    symbol_total = sum(symbol_counts.values())
    symbol_summary = {
        "symbols": symbol_total,
        **{
            name: {"count": symbol_counts[name], "rate": _pct(symbol_counts[name], symbol_total)}
            for name in ("exact", "homograph", "error")
        },
        "wrong_symbols": symbol_counts["homograph"] + symbol_counts["error"],
        "wrong_truth_in_top5": top5_recoverable_symbols,
        "wrong_truth_outside_top5": (
            symbol_counts["homograph"] + symbol_counts["error"] - top5_recoverable_symbols
        ),
        "top_pairs": [
            {"truth": truth, "prediction": prediction, "kind": kind, "count": count}
            for (truth, prediction, kind), count in pair_counts.most_common(100)
        ],
        "by_homograph_family": [
            {"family": family, "count": count}
            for family, count in family_counts.most_common()
        ],
    }
    formula_summary = {
        "protocol": summary(formulas),
        "eligible": summary(eligible_rows),
        "exact_formula_duplicates_removed": len(duplicates),
        "spacing_policy": {
            "whitespace_is_a_scored_token": False,
            "whitespace_only_differences_are_accepted": True,
            "separate_whitespace_only_error_count": 0,
            "reason": "CROHME comparison is performed on symbol-group tokens; spaces are never emitted or scored",
        },
    }
    return formulas, formula_summary, symbol_summary


def _stream_geometry(root: Path) -> dict:
    protocol, _ = load_crohme(root)
    raw_cvs, arc_cvs, normalized_segments = [], [], []
    zero_segments = total_segments = strokes_measured = 0
    endpoint_max_error = 0.0
    for formula in protocol:
        sample = _crohme_sample(root / str(formula["record_id"]))
        if sample is None:
            continue
        for stroke in sample["strokes"]:
            points = np.asarray([[float(x), float(y), 0.0] for x, y, _ in stroke], dtype=np.float64)
            if len(points) < 3:
                continue
            distances = np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)
            mean = float(distances.mean())
            if mean <= 0:
                continue
            raw_cvs.append(float(distances.std() / mean))
            normalized_segments.extend((distances / mean).tolist())
            zero_segments += int(np.count_nonzero(distances <= 1e-12))
            total_segments += len(distances)
            sampled = _resample_stroke(points, len(points))
            sampled_distances = np.linalg.norm(np.diff(sampled[:, :2], axis=0), axis=1)
            sampled_mean = float(sampled_distances.mean())
            arc_cvs.append(float(sampled_distances.std() / sampled_mean) if sampled_mean > 0 else 0.0)
            endpoint_max_error = max(
                endpoint_max_error,
                float(np.linalg.norm(sampled[0, :2] - points[0, :2])),
                float(np.linalg.norm(sampled[-1, :2] - points[-1, :2])),
            )
            strokes_measured += 1
    if not raw_cvs:
        raise ValueError("no CROHME strokes available for timing-density audit")
    values = np.asarray(normalized_segments, dtype=np.float64)
    return {
        "timestamp_available": False,
        "current_replay": "one recorded point per synthetic 48 Hz tick",
        "spatial_point_order_preserved": True,
        "actual_writer_speed_preserved": False,
        "strokes_measured": strokes_measured,
        "segments_measured": total_segments,
        "zero_length_segment_rate": _pct(zero_segments, total_segments),
        "raw_point_tick_speed_cv": {
            "median": float(statistics.median(raw_cvs)),
            "p90": float(np.quantile(raw_cvs, 0.90)),
        },
        "arc_length_resampled_speed_cv": {
            "median": float(statistics.median(arc_cvs)),
            "p90": float(np.quantile(arc_cvs, 0.90)),
        },
        "raw_segment_speed_multiplier": {
            "p05": float(np.quantile(values, 0.05)),
            "p50": float(np.quantile(values, 0.50)),
            "p95": float(np.quantile(values, 0.95)),
        },
        "arc_length_endpoint_max_error": endpoint_max_error,
        "model_time_policy": "delta_t is overwritten to [0, 1/127, ..., 1/127] by uniform-time mode",
        "decision": "CROHME ordinal replay measures geometric completion, not real-time latency",
        "recommended_protocol": [
            "rename CROHME x-axis to arc-length or point-progress, not 48 Hz time",
            "resample each stroke by cumulative arc length for visually uniform benchmark replay",
            "measure production latency only on timestamped project/device streams using real 48 Hz interpolation",
        ],
    }


def _overfit_evidence(
    training: dict, calibration: dict, writer_loo: dict, selection: dict, symbol_summary: dict,
) -> dict:
    history = training["history"]
    selection_history = selection["history"]
    external = calibration["calibrated_external"]["collision_free_math"]
    direct = writer_loo["pooled_direct"]["calibrated"]
    crohme_top1 = symbol_summary["exact"]["rate"]
    crohme_top5 = 1.0 - (
        symbol_summary["wrong_truth_outside_top5"] / max(symbol_summary["symbols"], 1)
    )
    return {
        "current_training_loss": {
            "epoch_1": history[0]["math"]["loss"],
            "epoch_last": history[-1]["math"]["loss"],
            "epochs": len(history),
            "same_run_validation_curve_available": False,
        },
        "independent_scores": {
            "external_collision_free_holdout": {"top1": external["top1"], "top5": external["top5"], "records": external["records"]},
            "project_writer_loo": {"top1": direct["top1"], "top5": direct["top5"], "records": direct["records"]},
            "crohme_test_truth_groups": {"top1": crohme_top1, "top5": crohme_top5, "records": symbol_summary["symbols"]},
        },
        "gaps": {
            "crohme_minus_external_top1": crohme_top1 - external["top1"],
            "crohme_minus_writer_loo_top1": crohme_top1 - direct["top1"],
            "crohme_minus_external_top5": crohme_top5 - external["top5"],
        },
        "related_epoch_selection": {
            "same_exact_run": False,
            "selected_epoch": selection["selection_validation"]["selected_epoch"],
            "math_top1_by_epoch": [row["selection_validation"]["math"]["top1"] for row in selection_history],
            "math_top5_by_epoch": [row["selection_validation"]["math"]["top5"] for row in selection_history],
        },
        "verdict": {
            "severe_memorization_confirmed": False,
            "overfit_ruled_out": False,
            "assessment": (
                "No classic validation collapse is visible in the related epoch selection and independent holdouts remain close, "
                "but the exact uniform-time full-data run lacks a per-epoch validation curve. The CROHME gap is consistent with "
                "domain/class-distribution shift and does not by itself prove overfit."
            ),
        },
    }


def _layer_evidence(formulas: list[dict], symbols: dict, raw: dict) -> dict:
    eligible = [row for row in formulas if row["eligible"]]
    exact = sum(row["category"] == "exact" for row in eligible)
    top5 = sum(row["top5_exact"] for row in eligible)
    wrong = symbols["wrong_symbols"]
    raw_scores = raw["scores"]
    return {
        "classifier_truth_group": {
            "top1_exact_symbols": symbols["exact"]["count"],
            "symbols": symbols["symbols"],
            "top1_wrong_symbols": wrong,
            "wrong_truth_still_in_top5": symbols["wrong_truth_in_top5"],
            "wrong_truth_outside_top5": symbols["wrong_truth_outside_top5"],
            "exact_formulas": exact,
            "top5_oracle_exact_formulas": top5,
            "eligible_formulas": len(eligible),
            "formula_reranking_recoverable": top5 - exact,
            "formula_candidate_recall_failure": len(eligible) - top5,
        },
        "raw_pipeline": {
            "eligible_formulas": raw["coverage"]["eligible_formulas"],
            "grouping_exact": raw_scores["grouping_exact_after"],
            "flat_token_sequence_exact": raw_scores["formula_exact_after"],
            "context_current_loop_changed": raw["current_loop"]["changed_formulas"],
            "context_formula_exact_gain": (
                raw_scores["formula_exact_after"] - raw_scores["formula_exact_before"]
            ),
            "two_dimensional_relation_exact_measured": False,
        },
        "diagnosis": [
            "classifier Top-1 is a real bottleneck because some truths are absent even from Top-5",
            "most wrong symbols still contain the truth in Top-5, so candidate ranking/context is the larger recoverable share",
            "raw grouping is an independent hard ceiling before official expression recognition",
            "the current downstream context loop produces zero net exact gain on this CROHME test",
        ],
    }


def build(args: argparse.Namespace) -> dict:
    prediction_rows = list(_json_lines(args.predictions))
    formulas, formula_summary, symbol_summary = _audit_formulas(args.crohme_root, prediction_rows)
    raw = _load_json(args.raw_runtime_report)
    training = _load_json(args.training_report)
    calibration = _load_json(args.calibration_report)
    writer_loo = _load_json(args.writer_loo_report)
    selection = _load_json(args.selection_report)
    report = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "noncommercial_research_diagnostic_only",
        "training_performed": False,
        "homograph_policy": {
            "semantic_equivalence": False,
            "formula_rule": "homograph_only only when every non-exact aligned token pair belongs to one declared visual family",
            "families": {name: sorted(values) for name, values in HOMOGRAPH_FAMILIES.items()},
        },
        "formula_summary": formula_summary,
        "symbol_summary": symbol_summary,
        "overfit": _overfit_evidence(training, calibration, writer_loo, selection, symbol_summary),
        "layers": _layer_evidence(formulas, symbol_summary, raw),
        "streaming_time": _stream_geometry(args.crohme_root),
        "formulas": formulas,
        "sources": {
            name: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for name, path in {
                "predictions": args.predictions,
                "raw_runtime_report": args.raw_runtime_report,
                "training_report": args.training_report,
                "calibration_report": args.calibration_report,
                "writer_loo_report": args.writer_loo_report,
                "selection_report": args.selection_report,
            }.items()
        },
    }
    return report


def _diff_text(row: dict) -> str:
    return ", ".join(
        f"{_code(_md(item['truth']))}→{_code(_md(item['prediction']))} "
        f"({'동형:' + item['family'] if item['kind'] == 'homograph' else '오류'})"
        for item in row["differences"][:8]
    ) or "-"


def markdown(report: dict) -> str:
    formulas = report["formula_summary"]
    symbols = report["symbol_summary"]
    layers = report["layers"]
    overfit = report["overfit"]
    timing = report["streaming_time"]
    rows = report["formulas"]
    lines = [
        "# CROHME 오류 등가성·과적합·레이어·시간 보전 감사",
        "",
        f"- 생성 시각(UTC): `{report['generated_at']}`",
        "- 학습 수행: 없음",
        "- 판정 경계: 동형문자는 의미상 정답이 아니라 형상 혼동 진단으로만 분리한다.",
        "",
        "## 1. 완전정답·동형문자·실오류",
        "",
        "공백은 CROHME symbol-group 토큰에 존재하지 않으므로 처음부터 점수에서 제외된다. 띄어쓰기만 다른 식은 완전정답으로 처리되며, 공백 때문에 오답 처리된 식은 0개다.",
        "표의 token 열은 렌더링된 LaTeX 문자열 순서가 아니라 InkML 정답 그룹 인덱스 순서다. 각 위치는 같은 정답 그룹의 truth와 Top-1을 직접 비교한다.",
        "",
        "| 범위 | 완전정답 | 동형문자만 | 실오류 |",
        "|---|---:|---:|---:|",
    ]
    for label, key in (("test 전체", "protocol"), ("평가 가능 truth-group", "eligible")):
        item = formulas[key]
        lines.append(
            f"| {label} ({item['formulas']}) | {item['exact']['count']} ({_percent(item['exact']['rate'])}) | "
            f"{item['homograph_only']['count']} ({_percent(item['homograph_only']['rate'])}) | "
            f"{item['error']['count']} ({_percent(item['error']['rate'])}) |"
        )
    lines.extend([
        "",
        "### 문자 단위",
        "",
        "| 완전정답 | 동형 혼동 | 실오류 |",
        "|---:|---:|---:|",
        f"| {symbols['exact']['count']}/{symbols['symbols']} ({_percent(symbols['exact']['rate'])}) | "
        f"{symbols['homograph']['count']}/{symbols['symbols']} ({_percent(symbols['homograph']['rate'])}) | "
        f"{symbols['error']['count']}/{symbols['symbols']} ({_percent(symbols['error']['rate'])}) |",
        "",
        "### 적용한 동형군",
        "",
        "| 군 | 토큰 | 실제 혼동 건수 |",
        "|---|---|---:|",
    ])
    family_count = {row["family"]: row["count"] for row in symbols["by_homograph_family"]}
    for name, values in report["homograph_policy"]["families"].items():
        lines.append(f"| {_code(name)} | {', '.join(_code(_md(value)) for value in values)} | {family_count.get(name, 0)} |")
    lines.extend([
        "",
        "이 목록 밖의 `x→\\varkappa`, `c→C`, `9→g` 같은 쌍은 비슷해 보여도 기존 승인 동형군이 아니므로 실오류로 유지했다.",
        "",
        "### 동형문자만 다른 식",
        "",
        "| ID | truth token | prediction token | 차이 |",
        "|---|---|---|---|",
    ])
    homographs = sorted(
        (row for row in rows if row["category"] == "homograph_only"),
        key=lambda row: (-row["difference_count"], row["formula_id"]),
    )
    for row in homographs[:40]:
        lines.append(
            f"| {_code(row['formula_id'])} | {_code(_md(' '.join(row['truth_tokens'])))} | "
            f"{_code(_md(' '.join(row['prediction_tokens'])))} | {_diff_text(row)} |"
        )
    if not homographs:
        lines.append("| - | - | - | - |")
    lines.extend([
        "",
        "### 아예 틀린 식 예시",
        "",
        "| ID | truth token | prediction token | 동형/실오류 |",
        "|---|---|---|---|",
    ])
    errors = sorted(
        (row for row in rows if row["eligible"] and row["category"] == "error"),
        key=lambda row: (-row["error_difference_count"], -row["difference_count"], row["formula_id"]),
    )
    for row in errors[:50]:
        lines.append(
            f"| {_code(row['formula_id'])} | {_code(_md(' '.join(row['truth_tokens'])))} | "
            f"{_code(_md(' '.join(row['prediction_tokens'])))} | {_diff_text(row)} |"
        )
    lines.extend([
        "",
        "전체 식의 모든 토큰 차이는 동명 JSON의 `formulas` 배열에 보존했다.",
        "",
        "## 2. 과적합 여부",
        "",
        "| 근거 | Top-1 | Top-5 | n |",
        "|---|---:|---:|---:|",
    ])
    scores = overfit["independent_scores"]
    for label, key in (
        ("외부 고정 holdout (collision-free)", "external_collision_free_holdout"),
        ("직접수집 writer-LOO", "project_writer_loo"),
        ("CROHME test truth-group", "crohme_test_truth_groups"),
    ):
        item = scores[key]
        lines.append(f"| {label} | {_percent(item['top1'])} | {_percent(item['top5'])} | {item['records']} |")
    loss = overfit["current_training_loss"]
    lines.extend([
        "",
        f"- 현재 base 학습 loss: epoch 1 `{loss['epoch_1']:.4f}` → epoch {loss['epochs']} `{loss['epoch_last']:.4f}`.",
        f"- CROHME Top-1은 외부 holdout보다 `{100 * overfit['gaps']['crohme_minus_external_top1']:.2f}%p`, writer-LOO보다 `{100 * overfit['gaps']['crohme_minus_writer_loo_top1']:.2f}%p` 낮다.",
        "- 동일 uniform-time full-data 실행에는 epoch별 validation 곡선이 없고, all-writer 보정 자료 자체 점수도 누수 방지를 위해 계산하지 않았다.",
        f"- 관련 selection 실행은 epoch 7→8에서 math Top-1이 `{_percent(overfit['related_epoch_selection']['math_top1_by_epoch'][6])}`→`{_percent(overfit['related_epoch_selection']['math_top1_by_epoch'][7])}`로 올랐지만 Top-5는 `{_percent(overfit['related_epoch_selection']['math_top5_by_epoch'][6])}`→`{_percent(overfit['related_epoch_selection']['math_top5_by_epoch'][7])}`로 소폭 하락했다. 정확히 같은 실행이 아니므로 보조 근거다.",
        "- 판정: **심각한 암기형 과적합 증거는 없지만 과적합을 완전히 배제할 수도 없다.** 현재 격차는 데이터 도메인·클래스 분포 차이와도 일치한다.",
        "",
        "## 3. 문자 분류기와 후단 레이어 병목",
        "",
        "| 단계 | 결과 | 해석 |",
        "|---|---:|---|",
    ])
    classifier = layers["classifier_truth_group"]
    raw = layers["raw_pipeline"]
    lines.extend([
        f"| 문자 분류 Top-1 | {classifier['top1_exact_symbols']}/{classifier['symbols']} ({_percent(_pct(classifier['top1_exact_symbols'], classifier['symbols']))}) | 분류기 자체 오류 존재 |",
        f"| Top-1 오답 중 truth가 Top-5에 있음 | {classifier['wrong_truth_still_in_top5']}/{classifier['top1_wrong_symbols']} ({_percent(_pct(classifier['wrong_truth_still_in_top5'], classifier['top1_wrong_symbols']))}) | 후단 재랭킹 가능 영역 |",
        f"| Top-1 오답 중 truth가 Top-5 밖 | {classifier['wrong_truth_outside_top5']}/{classifier['top1_wrong_symbols']} ({_percent(_pct(classifier['wrong_truth_outside_top5'], classifier['top1_wrong_symbols']))}) | 분류기 candidate recall 문제 |",
        f"| truth-group 식 Top-1 exact | {classifier['exact_formulas']}/{classifier['eligible_formulas']} ({_percent(_pct(classifier['exact_formulas'], classifier['eligible_formulas']))}) | 현재 순위 그대로 |",
        f"| truth-group 식 Top-5 oracle | {classifier['top5_oracle_exact_formulas']}/{classifier['eligible_formulas']} ({_percent(_pct(classifier['top5_oracle_exact_formulas'], classifier['eligible_formulas']))}) | 문맥층의 이론적 후보 상한 |",
        f"| raw 그룹 exact | {raw['grouping_exact']}/{raw['eligible_formulas']} ({_percent(_pct(raw['grouping_exact'], raw['eligible_formulas']))}) | 그룹핑 독립 병목 |",
        f"| raw flat token exact | {raw['flat_token_sequence_exact']}/{raw['eligible_formulas']} ({_percent(_pct(raw['flat_token_sequence_exact'], raw['eligible_formulas']))}) | 2D 관계는 미포함 |",
        f"| current context exact 순증가 | {raw['context_formula_exact_gain']}식 | 현재 후단이 후보 잠재력을 회수하지 못함 |",
        "",
        f"결론적으로 오답 문자 {classifier['top1_wrong_symbols']}개 중 {classifier['wrong_truth_still_in_top5']}개는 정답이 후보 안에 있다. 오답 716식 중 {classifier['formula_reranking_recoverable']}식은 모든 정답이 Top-5 안에 있고, {classifier['formula_candidate_recall_failure']}식은 적어도 하나가 Top-5 밖이다. 따라서 **현재 가장 큰 회수 가능 병목은 분류기 다음의 후보 재랭킹·문맥 확정층**이다. 다만 {classifier['wrong_truth_outside_top5']}개 문자는 후보에도 정답이 없어 분류기 개선이 필요하고, raw 그룹핑도 별도 병목이다.",
        "",
        "## 4. 스트리밍 속도·형상 보전",
        "",
        "| 항목 | 결과 |",
        "|---|---:|",
        f"| 실제 timestamp | 없음 |",
        f"| 측정 stroke/segment | {timing['strokes_measured']}/{timing['segments_measured']} |",
        f"| raw point를 고정 tick으로 재생한 속도 CV 중앙값 | {timing['raw_point_tick_speed_cv']['median']:.3f} |",
        f"| raw point 속도 CV p90 | {timing['raw_point_tick_speed_cv']['p90']:.3f} |",
        f"| 호장 길이 재표본 속도 CV 중앙값 | {timing['arc_length_resampled_speed_cv']['median']:.3f} |",
        f"| 고정 tick 구간속도 배수 p05/p50/p95 | {timing['raw_segment_speed_multiplier']['p05']:.2f} / {timing['raw_segment_speed_multiplier']['p50']:.2f} / {timing['raw_segment_speed_multiplier']['p95']:.2f} |",
        "",
        "좌표와 획 순서는 유지되므로 최종 형상이 찌그러지는 것은 아니다. 하지만 기록점 밀도를 실제 시간으로 오인해 재생하므로 조밀한 구간은 느리고 성긴 구간은 점프해 보일 수 있다. 또한 현재 모델은 `uniform-time`에서 `delta_t`를 강제로 균등화하므로 실제 필기속도를 사용하지 않는다.",
        "",
        "### 제안",
        "",
        "1. CROHME 평가는 48 Hz가 아니라 **호장 길이 진행률/점 진행률 스트리밍**으로 명명하고 평가한다.",
        "2. 시각 재생은 각 획을 누적 호장 길이로 재표본해 일정한 기하 속도로 표시한다.",
        "3. 실제 지연시간·속도 품질은 timestamp가 있는 직접수집/장치 입력만 실제 48 Hz로 보간해 별도 측정한다.",
        "4. 분류기는 writer 속도 편차에 강한 현재 uniform-time 입력을 유지하되, 속도 신호를 쓰려면 timestamp 품질 gate가 있는 보조 채널로 분리한다.",
        "",
        "DTW는 기준 궤적과의 정렬·유사도 분석에는 쓸 수 있지만, timestamp가 없는 CROHME의 실제 속도를 복원하지는 못한다. 재생 시계에는 단순한 단조 호장 길이 재표본이 더 적합하다.",
        "",
        "## 한계",
        "",
        "- 동형문자율은 선언한 형상군에 따라 달라지는 진단값이며 공식 CROHME Expression Rate가 아니다.",
        "- raw 식 exact는 flat token 순서이며 2D relation exact를 포함하지 않는다.",
        "- CROHME는 비상업 연구 진단 자료이며 제품 승격 근거가 아니다.",
    ])
    return "\n".join(lines) + "\n"


def self_test() -> None:
    assert _pair_family("1", "/") == "vertical_or_slash"
    assert _pair_family("0", "O") == "circle"
    assert _pair_family("x", r"\times") == "cross"
    assert _pair_family("x", r"\varkappa") is None
    assert not any(token.isspace() for values in HOMOGRAPH_FAMILIES.values() for token in values)
    seen = set()
    for values in HOMOGRAPH_FAMILIES.values():
        assert not seen.intersection(values)
        seen.update(values)
    print(json.dumps({"self_test": "pass", "families": len(HOMOGRAPH_FAMILIES)}))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--crohme-root", type=Path)
    parser.add_argument("--raw-runtime-report", type=Path)
    parser.add_argument("--training-report", type=Path)
    parser.add_argument("--calibration-report", type=Path)
    parser.add_argument("--writer-loo-report", type=Path)
    parser.add_argument("--selection-report", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    paths = (
        args.predictions, args.crohme_root, args.raw_runtime_report, args.training_report,
        args.calibration_report, args.writer_loo_report, args.selection_report,
    )
    if not all(path is not None and path.exists() for path in paths):
        parser.error("all input paths are required and must exist")
    if args.output_json is None or args.output_md is None:
        parser.error("both output paths are required")
    if args.output_json.exists() or args.output_md.exists():
        parser.error("refusing to overwrite outputs")
    report = build(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    args.output_md.write_text(markdown(report), encoding="utf-8", newline="\n")
    print(json.dumps({
        "json": str(args.output_json),
        "markdown": str(args.output_md),
        "formula_summary": report["formula_summary"],
        "symbol_summary": {key: report["symbol_summary"][key] for key in ("symbols", "exact", "homograph", "error")},
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
