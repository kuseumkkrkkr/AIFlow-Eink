#!/usr/bin/env python3
"""Summarize point-prefix CROHME replay into JSON and Markdown evidence.

This is an evaluation-only adapter.  It consumes saved prefix predictions and
never trains, selects thresholds, or changes recognition output.
"""

from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Iterable

from audit_replay_protocol_v1 import CROHME_ALIASES, _crohme_sample
from replay_evaluate_hwr_v1 import load_crohme


SCHEMA = "aiflow-crohme-streaming-usability/v1"
DECILES = tuple(range(10, 101, 10))
PHASES = ((0, 25), (25, 50), (50, 75), (75, 100))
OFFICIAL_PAPER = "https://www.cs.rit.edu/~rlaz/files/CROHME_TFD_2019.pdf"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_lines(path: Path) -> Iterable[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def _pct(value: int | float, total: int | float) -> float | None:
    return float(value) / float(total) if total else None


def _percent(value: float | None) -> str:
    return "-" if value is None else f"{100.0 * value:.2f}%"


def _md(value: object) -> str:
    return str(value).replace("|", r"\|").replace("\n", " ")


def _code(value: object) -> str:
    return "`" + str(value).replace("`", "'").replace("\n", " ") + "`"


def _progress_index(points: list[int], percentage: int) -> int:
    target = max(2, math.ceil(points[-1] * percentage / 100.0))
    return min(bisect_left(points, target), len(points) - 1)


def _row_diagnostic(row: dict) -> dict:
    truth = str(row["label"])
    points = [int(value) for value in row["prefix_points"]]
    tokens = [str(value) for value in row["top1_tokens"]]
    top5_hits = [bool(value) for value in row["top5_hits"]]
    if not points or not (len(points) == len(tokens) == len(top5_hits)):
        raise ValueError(f"invalid prefix row: {row.get('record_id')}")
    if any(right <= left for left, right in zip(points, points[1:])):
        raise ValueError(f"non-increasing prefix points: {row.get('record_id')}")
    correct = [token == truth for token in tokens]
    stable = None
    if correct[-1]:
        start = len(correct) - 1
        while start and correct[start - 1]:
            start -= 1
        stable = points[start] / points[-1]
    first = next((points[index] / points[-1] for index, hit in enumerate(correct) if hit), None)
    phases = {}
    for low, high in PHASES:
        indices = [
            index for index, point in enumerate(points)
            if low < 100.0 * point / points[-1] <= high
        ]
        phases[f"{low + 1:02d}-{high:03d}"] = {
            "prefixes": len(indices),
            "hits": sum(correct[index] for index in indices),
        }
    return {
        "record_id": str(row["record_id"]),
        "formula_id": str(row["formula_id"]),
        "label": truth,
        "final_top1": str(row["final_topk"][0]),
        "final_top5_hit": truth in [str(value) for value in row["final_topk"][:5]],
        "prefixes": len(points),
        "prefix_hits": sum(correct),
        "prefix_top5_hits": sum(top5_hits),
        "last_20_count": max(1, math.ceil(len(points) * 0.2)),
        "last_20_hits": sum(correct[-max(1, math.ceil(len(points) * 0.2)):]),
        "first_correct_progress": first,
        "stable_correct_progress": stable,
        "prediction_flips": sum(left != right for left, right in zip(tokens, tokens[1:])),
        "ever_correct": any(correct),
        "end_regression": any(correct) and not correct[-1],
        "decile_hits": {
            str(decile): correct[_progress_index(points, decile)] for decile in DECILES
        },
        "phases": phases,
    }


def _label_metrics(rows: list[dict], diagnostics: dict[str, dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["label"])].append(row)
    output = []
    for label in sorted(grouped):
        selected = grouped[label]
        details = [diagnostics[str(row["record_id"])] for row in selected]
        confusions = Counter(
            detail["final_top1"] for detail in details if detail["final_top1"] != label
        )
        stable = [
            detail["stable_correct_progress"] for detail in details
            if detail["stable_correct_progress"] is not None
        ]
        phases = {}
        for low, high in PHASES:
            key = f"{low + 1:02d}-{high:03d}"
            count = sum(detail["phases"][key]["prefixes"] for detail in details)
            hits = sum(detail["phases"][key]["hits"] for detail in details)
            phases[key] = {"prefixes": count, "accuracy": _pct(hits, count)}
        weakest_phase = min(
            (key for key, value in phases.items() if value["prefixes"]),
            key=lambda key: phases[key]["accuracy"],
        )
        output.append({
            "label": label,
            "records": len(selected),
            "final_top1_hits": sum(detail["final_top1"] == label for detail in details),
            "final_top1": _pct(sum(detail["final_top1"] == label for detail in details), len(details)),
            "final_top5": _pct(sum(detail["final_top5_hit"] for detail in details), len(details)),
            "all_prefix_top1": _pct(
                sum(detail["prefix_hits"] for detail in details),
                sum(detail["prefixes"] for detail in details),
            ),
            "last_20pct_top1": _pct(
                sum(detail["last_20_hits"] for detail in details),
                sum(detail["last_20_count"] for detail in details),
            ),
            "median_stable_correct_progress": statistics.median(stable) if stable else None,
            "mean_prediction_flips": statistics.fmean(detail["prediction_flips"] for detail in details),
            "end_regressions": sum(detail["end_regression"] for detail in details),
            "top_confusions": [
                {"prediction": token, "count": count}
                for token, count in confusions.most_common(3)
            ],
            "phases": phases,
            "weakest_phase": weakest_phase,
            "weakest_phase_accuracy": phases[weakest_phase]["accuracy"],
        })
    return output


def _snapshot(sample: dict, rows: dict[int, dict], percentage: int) -> dict:
    group_by_stroke: dict[int, int] = {}
    for group_index, group in enumerate(sample["groups"]):
        for stroke_index in group:
            if stroke_index in group_by_stroke:
                raise ValueError("truth groups overlap")
            group_by_stroke[int(stroke_index)] = group_index
    lookups = {
        index: dict(zip(row["prefix_points"], row["top1_tokens"], strict=True))
        for index, row in rows.items()
    }
    total_points = sum(len(stroke) for stroke in sample["strokes"])
    target = max(1, math.ceil(total_points * percentage / 100.0))
    local_counts = Counter()
    predictions: dict[int, str] = {}
    completed: set[int] = set()
    last_stroke = {index: max(group) for index, group in enumerate(sample["groups"])}
    seen = 0
    stop = False
    for stroke_index, stroke in enumerate(sample["strokes"]):
        group_index = group_by_stroke.get(stroke_index)
        stroke_finished = False
        for point_index, _point in enumerate(stroke):
            seen += 1
            if group_index is not None:
                local_counts[group_index] += 1
                token = lookups.get(group_index, {}).get(local_counts[group_index])
                if token is not None:
                    predictions[group_index] = str(token)
            stroke_finished = point_index == len(stroke) - 1
            if seen >= target:
                stop = True
                break
        if group_index is not None and stroke_index == last_stroke[group_index] and stroke_finished:
            completed.add(group_index)
        if stop:
            break
    truths = [str(value) for value in sample["labels"]]
    visible = sorted(predictions)
    done = sorted(completed & predictions.keys())
    return {
        "coverage": _pct(len(visible), len(truths)),
        "visible_count": len(visible),
        "visible_hits": sum(predictions[index] == truths[index] for index in visible),
        "completed_count": len(done),
        "completed_hits": sum(predictions[index] == truths[index] for index in done),
        "expression_exact": len(visible) == len(truths) and all(
            predictions[index] == truths[index] for index in range(len(truths))
        ),
    }


def _formula_inventory(
    root: Path, prediction_rows: list[dict], diagnostics: dict[str, dict],
) -> tuple[list[dict], dict, dict]:
    protocol, duplicates = load_crohme(root)
    rows_by_formula: dict[str, dict[int, dict]] = defaultdict(dict)
    for row in prediction_rows:
        formula_id, index = str(row["record_id"]).rsplit(":", 1)
        rows_by_formula[formula_id][int(index)] = row
    formulas = []
    stream_totals = {
        str(decile): Counter(
            formulas=0, visible_count=0, visible_hits=0,
            completed_count=0, completed_hits=0, expression_exact=0,
        ) for decile in DECILES
    }
    missing_partition = incomplete_partition = 0
    for formula in protocol:
        formula_id = str(formula["record_id"])
        sample = _crohme_sample(root / formula_id)
        if sample is None:
            missing_partition += 1
            formulas.append({
                "formula_id": formula_id, "truth": str(formula["label"]),
                "symbols": 0, "eligible": False, "errors": 1,
                "top5_errors": 1, "exact": False, "top5_exact": False,
                "failure_reason": "missing_truth_partition", "error_details": [],
                "prediction_flips": 0,
            })
            continue
        incomplete_partition += int(not sample["partition_complete"])
        rows = rows_by_formula.get(formula_id, {})
        truths = [CROHME_ALIASES.get(str(value), str(value)) for value in sample["labels"]]
        error_details = []
        errors = top5_errors = 0
        for index, truth in enumerate(truths):
            row = rows.get(index)
            prediction = str(row["final_topk"][0]) if row else None
            top5 = [str(value) for value in row["final_topk"][:5]] if row else []
            if prediction != truth:
                errors += 1
                error_details.append({
                    "index": index, "truth": truth,
                    "prediction": prediction or "<UNSUPPORTED_OR_INVALID>",
                })
            top5_errors += truth not in top5
        eligible = bool(sample["partition_complete"] and len(rows) == len(truths))
        if not sample["partition_complete"]:
            reason = "incomplete_truth_partition"
        elif len(rows) != len(truths):
            reason = "unsupported_or_invalid_truth_group"
        else:
            reason = None
        formula_row = {
            "formula_id": formula_id,
            "truth": str(formula["label"]),
            "symbols": len(truths),
            "eligible": eligible,
            "errors": errors,
            "top5_errors": top5_errors,
            "exact": eligible and errors == 0,
            "top5_exact": eligible and top5_errors == 0,
            "failure_reason": reason,
            "error_details": error_details,
            "prediction_flips": sum(
                diagnostics[str(row["record_id"])]["prediction_flips"] for row in rows.values()
            ),
        }
        formulas.append(formula_row)
        if eligible:
            for decile in DECILES:
                snap = _snapshot(sample, rows, decile)
                bucket = stream_totals[str(decile)]
                bucket["formulas"] += 1
                for key in (
                    "visible_count", "visible_hits", "completed_count", "completed_hits",
                ):
                    bucket[key] += snap[key]
                bucket["expression_exact"] += snap["expression_exact"]
                bucket["coverage_sum_ppm"] += round((snap["coverage"] or 0.0) * 1_000_000)
    streaming = {}
    for decile, bucket in stream_totals.items():
        formulas_count = bucket["formulas"]
        streaming[decile] = {
            "formulas": formulas_count,
            "mean_visible_symbol_coverage": _pct(bucket["coverage_sum_ppm"], formulas_count * 1_000_000),
            "visible_symbol_accuracy": _pct(bucket["visible_hits"], bucket["visible_count"]),
            "completed_symbol_accuracy": _pct(bucket["completed_hits"], bucket["completed_count"]),
            "expression_exact": _pct(bucket["expression_exact"], formulas_count),
        }
    coverage = {
        "raw_formulas": len(protocol) + len(duplicates),
        "protocol_formulas": len(protocol),
        "exact_duplicates_removed": len(duplicates),
        "eligible_formulas": sum(row["eligible"] for row in formulas),
        "missing_truth_partition_formulas": missing_partition,
        "incomplete_truth_partition_formulas": incomplete_partition,
    }
    return formulas, coverage, streaming


def _length_bucket(length: int) -> str:
    if length <= 3:
        return "1-3"
    if length <= 6:
        return "4-6"
    if length <= 10:
        return "7-10"
    if length <= 20:
        return "11-20"
    return "21+"


def _formula_metrics(formulas: list[dict]) -> dict:
    eligible = [row for row in formulas if row["eligible"]]
    exact = [row for row in eligible if row["exact"]]
    all_count = len(formulas)
    length = defaultdict(Counter)
    for row in eligible:
        bucket = _length_bucket(int(row["symbols"]))
        length[bucket]["formulas"] += 1
        length[bucket]["exact"] += row["exact"]
    repeated = defaultdict(list)
    for row in formulas:
        key = "".join(str(row["truth"]).replace("$", "").split())
        repeated[key].append(row)
    repeated_stats = []
    for truth, values in repeated.items():
        if len(values) < 2:
            continue
        wrong = sum(not row["exact"] for row in values)
        repeated_stats.append({
            "truth": truth, "instances": len(values), "wrong": wrong,
            "exact": len(values) - wrong, "wrong_rate": wrong / len(values),
        })
    repeated_stats.sort(key=lambda row: (-row["wrong"], -row["wrong_rate"], row["truth"]))
    hardest = sorted(
        (row for row in formulas if not row["exact"]),
        key=lambda row: (-row["errors"], -row["prediction_flips"], row["formula_id"]),
    )
    return {
        "all_protocol": {
            "formulas": all_count,
            "strict_token_exact_count": len(exact),
            "strict_token_exact": _pct(len(exact), all_count),
            "within_1_symbol_error": _pct(sum(row["errors"] <= 1 for row in formulas), all_count),
            "within_2_symbol_errors": _pct(sum(row["errors"] <= 2 for row in formulas), all_count),
            "within_3_symbol_errors": _pct(sum(row["errors"] <= 3 for row in formulas), all_count),
        },
        "eligible_oracle_group": {
            "formulas": len(eligible),
            "strict_token_exact_count": len(exact),
            "strict_token_exact": _pct(len(exact), len(eligible)),
            "top5_oracle_exact_count": sum(row["top5_exact"] for row in eligible),
            "top5_oracle_exact": _pct(sum(row["top5_exact"] for row in eligible), len(eligible)),
            "within_1_symbol_error": _pct(sum(row["errors"] <= 1 for row in eligible), len(eligible)),
            "within_2_symbol_errors": _pct(sum(row["errors"] <= 2 for row in eligible), len(eligible)),
            "within_3_symbol_errors": _pct(sum(row["errors"] <= 3 for row in eligible), len(eligible)),
        },
        "exact_formula_ids": [row["formula_id"] for row in exact],
        "exact_formula_examples": [
            {"formula_id": row["formula_id"], "truth": row["truth"], "symbols": row["symbols"]}
            for row in sorted(exact, key=lambda row: (row["symbols"], row["formula_id"]))[:30]
        ],
        "length_distribution": {
            key: {
                "formulas": value["formulas"], "exact": value["exact"],
                "exact_rate": _pct(value["exact"], value["formulas"]),
            }
            for key in ("1-3", "4-6", "7-10", "11-20", "21+")
            if (value := length.get(key)) is not None
        },
        "frequently_wrong_repeated_truths": repeated_stats,
        "hardest_formula_instances": hardest,
    }


def _load_optional(path: Path | None) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path else None


def build_report(args: argparse.Namespace) -> dict:
    rows = list(_json_lines(args.predictions))
    if not rows:
        raise ValueError("prefix prediction file is empty")
    if len({str(row["record_id"]) for row in rows}) != len(rows):
        raise ValueError("prefix record_id values must be unique")
    diagnostics = {str(row["record_id"]): _row_diagnostic(row) for row in rows}
    final_hits = sum(detail["final_top1"] == detail["label"] for detail in diagnostics.values())
    final_top5 = sum(detail["final_top5_hit"] for detail in diagnostics.values())
    prefix_report = json.loads(args.prefix_report.read_text(encoding="utf-8"))
    prefix_section = prefix_report.get("crohme", {})
    expected_prefixes = sum(detail["prefixes"] for detail in diagnostics.values())
    if prefix_report.get("schema") != "aiflow-48hz-prefix-evaluation/v1":
        raise ValueError("unexpected prefix report schema")
    if int(prefix_section.get("final", {}).get("records", -1)) != len(rows):
        raise ValueError("prefix report record count mismatch")
    if int(prefix_section.get("prefixes", -1)) != expected_prefixes:
        raise ValueError("prefix report prefix count mismatch")
    if not math.isclose(
        float(prefix_section.get("final", {}).get("top1", -1.0)),
        final_hits / len(rows), rel_tol=0.0, abs_tol=1e-12,
    ):
        raise ValueError("prefix report Top-1 mismatch")
    formulas, coverage, formula_stream = _formula_inventory(args.crohme_root, rows, diagnostics)
    formula_metrics = _formula_metrics(formulas)
    confusion = Counter(
        (detail["label"], detail["final_top1"])
        for detail in diagnostics.values() if detail["label"] != detail["final_top1"]
    )
    stable = [
        detail["stable_correct_progress"] for detail in diagnostics.values()
        if detail["stable_correct_progress"] is not None
    ]
    prefix_total = sum(detail["prefixes"] for detail in diagnostics.values())
    report = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "noncommercial_academic_diagnostic_only",
        "training_performed": False,
        "dataset": {
            "name": args.dataset_name,
            "split": args.split,
            "license_boundary": "CROHME noncommercial evaluation only",
            **coverage,
        },
        "academic_protocol": {
            "official_reference": OFFICIAL_PAPER,
            "official_test_formula_count": 1199,
            "official_primary_metric": "Symbol Layout Tree expression recognition rate",
            "implemented_primary_proxy": "truth-group token exact; unsupported/invalid groups fail strict all-protocol score",
            "not_implemented": "official symLG conversion and LgEval relationship/structure error scoring",
        },
        "replay_contract": {
            "tick_hz": 48,
            "crohme_timestamp_mode": "one recorded point per synthetic tick",
            "real_latency_measured": False,
            "future_bbox_used": False,
            "target_points": 128,
            "truth_grouping_used": True,
            "whole_formula_stream_reconstructed_from_source_stroke_order": True,
            "checkpoint_sha256": prefix_report.get("crohme_checkpoint", {}).get("sha256"),
        },
        "symbols": {
            "records": len(rows),
            "final_top1_hits": final_hits,
            "final_top1": _pct(final_hits, len(rows)),
            "final_top5_hits": final_top5,
            "final_top5": _pct(final_top5, len(rows)),
            "prefixes": prefix_total,
            "all_prefix_top1": _pct(sum(detail["prefix_hits"] for detail in diagnostics.values()), prefix_total),
            "last_20pct_top1": _pct(
                sum(detail["last_20_hits"] for detail in diagnostics.values()),
                sum(detail["last_20_count"] for detail in diagnostics.values()),
            ),
            "median_stable_correct_progress": statistics.median(stable) if stable else None,
            "mean_prediction_flips": statistics.fmean(detail["prediction_flips"] for detail in diagnostics.values()),
            "ever_correct_but_final_wrong": sum(detail["end_regression"] for detail in diagnostics.values()),
            "by_progress": {
                str(decile): _pct(
                    sum(detail["decile_hits"][str(decile)] for detail in diagnostics.values()), len(rows),
                ) for decile in DECILES
            },
            "top_confusions": [
                {"truth": truth, "prediction": prediction, "count": count}
                for (truth, prediction), count in confusion.most_common(50)
            ],
            "by_label": _label_metrics(rows, diagnostics),
        },
        "formulas": formula_metrics,
        "whole_formula_stream": formula_stream,
        "current_shadow_evidence": {
            "raw_runtime": _load_optional(args.raw_runtime_report),
            "layout_runtime": _load_optional(args.layout_report),
            "context_runtime": _load_optional(args.context_report),
        },
        "sources": {
            "predictions": {"path": str(args.predictions.resolve()), "sha256": _sha256(args.predictions)},
            "prefix_report": {"path": str(args.prefix_report.resolve()), "sha256": _sha256(args.prefix_report)},
            "crohme_root": str(args.crohme_root.resolve()),
            **({
                "raw_runtime_report": {
                    "path": str(args.raw_runtime_report.resolve()),
                    "sha256": _sha256(args.raw_runtime_report),
                }
            } if args.raw_runtime_report else {}),
        },
        "decision": {
            "official_crohme_expression_rate_claimable": False,
            "reason": "current replay supplies truth groups and does not emit official symLG/LgEval output",
            "commercial_product_validation": False,
        },
    }
    return report


def _markdown(report: dict) -> str:
    dataset = report["dataset"]
    symbols = report["symbols"]
    formulas = report["formulas"]
    eligible = formulas["eligible_oracle_group"]
    all_formula = formulas["all_protocol"]
    raw = report["current_shadow_evidence"]["raw_runtime"]
    raw_conclusion = ""
    if raw:
        raw_coverage = raw.get("coverage", {})
        raw_scores = raw.get("scores", {})
        raw_eligible = int(raw_coverage.get("eligible_formulas", 0))
        raw_exact = int(raw_scores.get("formula_exact_after", 0))
        raw_conclusion = (
            f" 정답 그룹을 주지 않은 current raw-stroke shadow의 flat token-sequence exact는 "
            f"**{raw_exact}/{raw_eligible} ({_percent(_pct(raw_exact, raw_eligible))})**였다."
        )
    lines = [
        "# CROHME 2019 실시간 스트리밍 정확도·사용성 평가",
        "",
        f"- 생성 시각(UTC): `{report['generated_at']}`",
        f"- 데이터: `{dataset['name']}` / `{dataset['split']}` / 원본 {dataset['raw_formulas']:,}식",
        "- 판정: **현재 결과는 연구 진단용이며, 상용 사용 승인이나 공식 CROHME 순위 점수가 아니다.**",
        "",
        "## 결론",
        "",
        f"최종 문자 Top-1은 **{symbols['final_top1_hits']:,}/{symbols['records']:,} ({_percent(symbols['final_top1'])})**, Top-5는 **{symbols['final_top5_hits']:,}/{symbols['records']:,} ({_percent(symbols['final_top5'])})**다. "
        f"전체 식을 엄격히 보면 truth-group token exact는 **{all_formula['strict_token_exact_count']:,}/{all_formula['formulas']:,} ({_percent(all_formula['strict_token_exact'])})**, "
        f"지원 라벨·유효 그룹만 남긴 {eligible['formulas']:,}식에서는 **{eligible['strict_token_exact_count']:,}/{eligible['formulas']:,} ({_percent(eligible['strict_token_exact'])})**다.{raw_conclusion}",
        "",
        f"모든 중간 타점의 문자 Top-1은 {_percent(symbols['all_prefix_top1'])}, 마지막 20% 구간은 {_percent(symbols['last_20pct_top1'])}다. "
        f"최종 정답 문자는 중앙값 기준 전체 타점의 {_percent(symbols['median_stable_correct_progress'])} 이후에야 계속 정답으로 유지됐다. "
        "따라서 타점마다 결과를 즉시 확정하는 UI에는 부적합하고, 후보 표시 후 획 종료·문맥 확정 시점에 커밋하는 방식이 필요하다.",
        "",
        "## 학계 표준과 이번 점수의 경계",
        "",
        f"CROHME 2019 공식 Task 1은 온라인 획을 Symbol Layout Tree로 변환하고 1,199식 test에서 Expression Rate로 순위를 정한다. 기호·관계 오류는 symLG/LgEval로 분석한다([공식 논문]({OFFICIAL_PAPER})).",
        "",
        "이번 신규 평가는 공식 `test` 1,199식을 사용했지만 문자마다 InkML 정답 그룹을 공급했다. 현재 출력은 공식 symLG가 아니므로 아래 식 exact는 **oracle-group token exact proxy**이며 공식 end-to-end Expression Rate로 인용하면 안 된다. 구조·그룹핑을 포함한 raw shadow 결과는 별도 표에 분리한다.",
        "",
        "## 평가 범위",
        "",
        "| 항목 | 값 |",
        "|---|---:|",
        f"| 원본 식 | {dataset['raw_formulas']:,} |",
        f"| exact 중복 제거 후 식 | {dataset['protocol_formulas']:,} |",
        f"| 모든 문자 그룹이 평가 가능한 식 | {dataset['eligible_formulas']:,} |",
        f"| 평가 문자 | {symbols['records']:,} |",
        f"| 평가 prefix | {symbols['prefixes']:,} |",
        "| CROHME 시간 | 실제 timestamp 없음; 기록점 1개를 48 Hz 1 tick으로 가정 |",
        "| 미래 bbox | 사용 안 함; 매 prefix를 보이는 점만으로 재정규화 |",
        "",
        "## 학계식 핵심 지표",
        "",
        "| 범위 | Exact | ≤1 문자 오류 | ≤2 | ≤3 | Top-5 식 oracle |",
        "|---|---:|---:|---:|---:|---:|",
        f"| test 전체(미지원 포함 실패) | {all_formula['strict_token_exact_count']}/{all_formula['formulas']} ({_percent(all_formula['strict_token_exact'])}) | {_percent(all_formula['within_1_symbol_error'])} | {_percent(all_formula['within_2_symbol_errors'])} | {_percent(all_formula['within_3_symbol_errors'])} | - |",
        f"| 평가 가능 truth-group 식 | {eligible['strict_token_exact_count']}/{eligible['formulas']} ({_percent(eligible['strict_token_exact'])}) | {_percent(eligible['within_1_symbol_error'])} | {_percent(eligible['within_2_symbol_errors'])} | {_percent(eligible['within_3_symbol_errors'])} | {eligible['top5_oracle_exact_count']}/{eligible['formulas']} ({_percent(eligible['top5_oracle_exact'])}) |",
        "",
        "`≤N`은 공식 LgEval 구조 오류가 아니라 현재 분류기의 **문자 오분류 개수 proxy**다.",
        "",
        "## 실시간 스트리밍",
        "",
        "### 문자 내부 진행률",
        "",
        "| 타점 진행률 | Top-1 |",
        "|---:|---:|",
    ]
    lines.extend(
        f"| {decile}% | {_percent(symbols['by_progress'][str(decile)])} |" for decile in DECILES
    )
    lines.extend([
        "",
        "### 식 전체 원본 획 순서",
        "",
        "아래 표는 식의 원본 획을 한 점씩 공개하면서, 이미 시작된 문자와 완료된 문자의 상태를 재구성한 것이다. 그룹 소유권은 정답을 사용하므로 실제 온라인 그룹핑 성능은 포함하지 않는다.",
        "",
        "| 식 진행률 | 시작된 문자 비율 | 보이는 문자 정확도 | 완료 문자 정확도 | 식 전체 exact |",
        "|---:|---:|---:|---:|---:|",
    ])
    for decile in DECILES:
        row = report["whole_formula_stream"][str(decile)]
        lines.append(
            f"| {decile}% | {_percent(row['mean_visible_symbol_coverage'])} | {_percent(row['visible_symbol_accuracy'])} | {_percent(row['completed_symbol_accuracy'])} | {_percent(row['expression_exact'])} |"
        )
    lines.extend([
        "",
        f"- 문자당 평균 Top-1 전환: **{symbols['mean_prediction_flips']:.2f}회**",
        f"- 중간에는 정답이었으나 마지막에 오답으로 끝난 문자: **{symbols['ever_correct_but_final_wrong']:,}개**",
        "- 위 시간축은 인식 안정성 시뮬레이션이다. 실제 장치 입력 지연·GPU/CPU 추론 latency는 측정하지 않았다.",
        "",
        "## 문자별 오류 위치와 혼동",
        "",
        "`초/중초/중후/말`은 각각 문자 타점의 1-25/26-50/51-75/76-100% 구간이다. 표본이 적은 클래스는 비율보다 건수를 우선 봐야 한다.",
        "",
        "| 문자 | n | 최종 Top-1 | Top-5 | 전체 prefix | 마지막 20% | 최약 구간 | 안정화 중앙값 | 전환/문자 | 끝회귀 | 주요 최종 혼동 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ])
    for row in symbols["by_label"]:
        confusion = ", ".join(
            f"{_code(_md(item['prediction']))} {item['count']}" for item in row["top_confusions"]
        ) or "-"
        lines.append(
            f"| {_code(_md(row['label']))} | {row['records']} | {_percent(row['final_top1'])} | {_percent(row['final_top5'])} | {_percent(row['all_prefix_top1'])} | {_percent(row['last_20pct_top1'])} | {row['weakest_phase']} ({_percent(row['weakest_phase_accuracy'])}) | {_percent(row['median_stable_correct_progress'])} | {row['mean_prediction_flips']:.2f} | {row['end_regressions']} | {confusion} |"
        )
    lines.extend([
        "",
        "### 오류 건수가 큰 최종 혼동",
        "",
        "| 정답 | 예측 | 건수 |",
        "|---|---|---:|",
    ])
    for row in symbols["top_confusions"][:30]:
        lines.append(f"| {_code(_md(row['truth']))} | {_code(_md(row['prediction']))} | {row['count']} |")
    lines.extend([
        "",
        "## 수식 통계",
        "",
        "### 길이별 전부 정답",
        "",
        "| 문자 수 | 식 | 전부 정답 | Exact |",
        "|---|---:|---:|---:|",
    ])
    for bucket, row in formulas["length_distribution"].items():
        lines.append(f"| {bucket} | {row['formulas']} | {row['exact']} | {_percent(row['exact_rate'])} |")
    lines.extend([
        "",
        "### 전부 맞춘 수식 예시 (truth-group token 기준)",
        "",
        "| ID | truth | 문자 수 |",
        "|---|---|---:|",
    ])
    for row in formulas["exact_formula_examples"]:
        lines.append(
            f"| {_code(_md(row['formula_id']))} | {_code(_md(row['truth']))} | {row['symbols']} |"
        )
    lines.extend([
        "",
        "### 반복 출현하면서 자주 틀린 수식",
        "",
        "| 정규화 truth | 출현 | 오답 | 오답률 |",
        "|---|---:|---:|---:|",
    ])
    repeated = formulas["frequently_wrong_repeated_truths"][:30]
    if repeated:
        for row in repeated:
            lines.append(f"| {_code(_md(row['truth']))} | {row['instances']} | {row['wrong']} | {_percent(row['wrong_rate'])} |")
    else:
        lines.append("| - | 0 | 0 | - |")
    lines.extend([
        "",
        "### 가장 많이 틀린 개별 수식",
        "",
        "| ID | truth | 문자 오류 | 예측 전환 | 주요 오류 |",
        "|---|---|---:|---:|---|",
    ])
    for row in formulas["hardest_formula_instances"][:40]:
        detail = ", ".join(
            f"{_code(_md(item['truth']))}→{_code(_md(item['prediction']))}"
            for item in row["error_details"][:5]
        ) or str(row.get("failure_reason") or "-")
        lines.append(
            f"| {_code(_md(row['formula_id']))} | {_code(_md(row['truth']))} | {row['errors']} | {row['prediction_flips']} | {detail} |"
        )
    lines.extend([
        "",
        f"전부 정답인 식 ID {len(formulas['exact_formula_ids']):,}개 전체 목록은 동명 JSON의 `formulas.exact_formula_ids`에 보존했다.",
        "",
        "## 현재 raw/문맥 shadow와의 관계",
        "",
    ])
    if raw:
        coverage = raw.get("coverage", {})
        scores = raw.get("scores", {})
        acceptance = raw.get("formula_acceptance_guard", {})
        eligible_raw = int(coverage.get("eligible_formulas", 0))
        exact_raw = int(scores.get("formula_exact_after", 0))
        grouping_raw = int(scores.get("grouping_exact_after", 0))
        lines.extend([
            f"- raw-stroke current shadow: 그룹 exact **{grouping_raw}/{eligible_raw} ({_percent(_pct(grouping_raw, eligible_raw))})**, flat token-sequence exact **{exact_raw}/{eligible_raw} ({_percent(_pct(exact_raw, eligible_raw))})**.",
            f"- 자동수락 gate: **{acceptance.get('auto_accepted', 0)}/{eligible_raw} ({_percent(acceptance.get('auto_accept_coverage'))})**를 자동수락했지만, 그중 정답은 **{acceptance.get('auto_accepted_exact', 0)}/{acceptance.get('auto_accepted', 0)} ({_percent(acceptance.get('auto_accepted_precision'))})**뿐이다.",
            "- 이 값은 정답 그룹 없이 시작하는 현재 파이프라인 진단이지만 2D 관계 exact를 포함하지 않는다. 공식 symLG/LgEval Expression Rate와 동일시하지 않는다.",
        ])
    else:
        lines.append("- 신규 test에 대한 raw-stroke end-to-end 보고서가 연결되지 않았다. 위 수치는 truth-group 문자/식 proxy만 나타낸다.")
    lines.extend([
        "- 현재 prompt-context·공식 배치 계층은 shadow이며 제품 기본값은 OFF다. 문맥층이 full-trace에서 좋아져도 중간 타점 안정성이나 실제 그룹핑 정확도를 대신하지 않는다.",
        "",
        "## 사용성 판정",
        "",
        "- **가능:** 획 종료 후 Top-5 후보를 유지하는 보조 입력, 사용자가 최종 수식을 검토하는 연구용 UI.",
        "- **조건부:** 문자 단위 실시간 미리보기. 진행 중 예측 전환이 잦으므로 debounce와 획 종료 커밋이 필요하다.",
        "- **불가:** 현재 수치로 자동 확정·무검토 수식 입력, 공식 CROHME SOTA 비교, 상업 승격 근거 사용.",
        "- **다음 검증:** 공식 symLG/LgEval 출력 연결, 실제 장치에서 p50/p95 추론 latency 측정, 새 상업 이용 가능 writer/formula-disjoint acceptance.",
        "",
        "## 재현",
        "",
        "```powershell",
        "python scripts\\report_crohme_streaming_v1.py `",
        f"  --predictions '{report['sources']['predictions']['path']}' `",
        f"  --prefix-report '{report['sources']['prefix_report']['path']}' `",
        f"  --crohme-root '{report['sources']['crohme_root']}' `",
        *(
            [f"  --raw-runtime-report '{report['sources']['raw_runtime_report']['path']}' `"]
            if "raw_runtime_report" in report["sources"] else []
        ),
        "  --output-json <output.json> --output-md <output.md>",
        "```",
        "",
        f"- prefix 예측 SHA-256: `{report['sources']['predictions']['sha256']}`",
        f"- prefix 기본 보고서 SHA-256: `{report['sources']['prefix_report']['sha256']}`",
        *(
            [f"- raw runtime 보고서 SHA-256: `{report['sources']['raw_runtime_report']['sha256']}`"]
            if "raw_runtime_report" in report["sources"] else []
        ),
    ])
    return "\n".join(lines) + "\n"


def _self_test() -> None:
    row = {
        "record_id": "f:0", "formula_id": "f", "label": "x",
        "prefix_points": [2, 3, 4], "top1_tokens": ["y", "x", "x"],
        "top5_hits": [False, True, True], "final_topk": ["x", "y"],
    }
    result = _row_diagnostic(row)
    assert result["prediction_flips"] == 1
    assert result["stable_correct_progress"] == 0.75
    assert result["final_top5_hit"]
    sample = {
        "strokes": [[(0, 0, None), (1, 1, None)], [(2, 0, None), (3, 1, None)]],
        "groups": [[0], [1]], "labels": ["x", "y"],
    }
    rows = {
        0: {"prefix_points": [2], "top1_tokens": ["x"]},
        1: {"prefix_points": [2], "top1_tokens": ["y"]},
    }
    assert _snapshot(sample, rows, 50)["completed_count"] == 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--prefix-report", type=Path)
    parser.add_argument("--crohme-root", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--raw-runtime-report", type=Path)
    parser.add_argument("--layout-report", type=Path)
    parser.add_argument("--context-report", type=Path)
    parser.add_argument("--dataset-name", default="CROHME 2019 online HME")
    parser.add_argument("--split", default="test")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        print('{"self_test":"pass"}')
        return 0
    required = (args.predictions, args.prefix_report, args.crohme_root, args.output_json, args.output_md)
    if any(path is None for path in required):
        parser.error("predictions, prefix report, CROHME root, and both outputs are required")
    if not all(path.exists() for path in (args.predictions, args.prefix_report, args.crohme_root)):
        parser.error("one or more required inputs are missing")
    if args.output_json.exists() or args.output_md.exists():
        parser.error("refusing to overwrite report output")
    report = build_report(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n",
    )
    args.output_md.write_text(_markdown(report), encoding="utf-8", newline="\n")
    print(json.dumps({
        "json": str(args.output_json), "json_sha256": _sha256(args.output_json),
        "markdown": str(args.output_md), "markdown_sha256": _sha256(args.output_md),
        "symbol_top1": report["symbols"]["final_top1"],
        "strict_formula_exact": report["formulas"]["all_protocol"]["strict_token_exact"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
