#!/usr/bin/env python3
"""동결 v3 HWR의 신규 작가·CROHME 결과와 평가 전용 궤적 방향을 분석한다."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import json
import math
from pathlib import Path

import numpy as np
import torch

from build_crohme_standard_candidates_v1 import _candidate_metrics, _crohme_items_tolerant
from cleanroom_trajectory_profiles_v3 import _sha256, trajectory_descriptor
from evaluate_48hz_prefix_v1 import _prefix_tensor


SCHEMA = "aiflow-cleanroom-profiled-final-analysis/v3"
FAMILIES = {
    "vertical": ("1", "|", "/"),
    "circle": ("0", "O", "o"),
    "cross": ("x", r"\times"),
}


def _d_path(path: Path, kind: str, *, must_exist: bool = True) -> Path:
    """최종 분석의 모든 입력·출력이 D:에 있는지 확인한다."""

    resolved = path.resolve()
    if resolved.drive.upper() != "D:":
        raise ValueError(f"{kind} must remain on D:: {resolved}")
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"missing {kind}: {resolved}")
    return resolved


def _json(path: Path) -> dict:
    """UTF-8 JSON 객체를 읽고 최상위 자료형을 검증한다."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _rows(path: Path) -> list[dict]:
    """gzip JSONL 후보 캐시를 중복 record_id 없이 읽는다."""

    with gzip.open(path, "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    identifiers = [str(row["record_id"]) for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"duplicate candidate record ids: {path}")
    return rows


def _score_rows(rows: list[dict]) -> dict:
    """문자·식·writer·클래스·동형계열 Top-1/Top-5를 한 번에 계산한다."""

    if not rows:
        raise ValueError("cannot score empty candidate rows")
    by_formula: dict[str, list[dict]] = defaultdict(list)
    by_writer: dict[str, list[dict]] = defaultdict(list)
    by_label: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_formula[str(row["formula_id"])].append(row)
        by_writer[str(row.get("writer_group", "unknown"))].append(row)
        by_label[str(row["label"])].append(row)

    def subset_score(values: list[dict]) -> dict:
        """한 부분집합의 레코드 수와 Top-1/Top-5를 계산한다."""

        return {
            "records": len(values),
            "top1": sum(row["final_topk"][0] == row["label"] for row in values) / len(values),
            "top5": sum(row["label"] in row["final_topk"] for row in values) / len(values),
        }

    label_scores = {key: subset_score(value) for key, value in sorted(by_label.items())}
    writer_scores = {key: subset_score(value) for key, value in sorted(by_writer.items())}
    family_scores = {}
    for name, labels in FAMILIES.items():
        subset = [row for row in rows if str(row["label"]) in labels]
        if subset:
            family_scores[name] = subset_score(subset)
    top1_hits = sum(row["final_topk"][0] == row["label"] for row in rows)
    top5_hits = sum(row["label"] in row["final_topk"] for row in rows)
    standard = _candidate_metrics(rows)
    return {
        "records": len(rows),
        "formulas": len(by_formula),
        "writers": len(by_writer),
        "top1": top1_hits / len(rows),
        "top5": top5_hits / len(rows),
        "outside_top5": len(rows) - top5_hits,
        "strict_macro_top1": standard["strict_macro_top1"],
        "strict_micro_top1": standard["strict_micro_top1"],
        "class_macro_top1": float(np.mean([row["top1"] for row in label_scores.values()])),
        "class_macro_top5": float(np.mean([row["top5"] for row in label_scores.values()])),
        "writer_macro_top1": float(np.mean([row["top1"] for row in writer_scores.values()])),
        "writer_macro_top5": float(np.mean([row["top5"] for row in writer_scores.values()])),
        "formula_exact": sum(
            all(row["final_topk"][0] == row["label"] for row in values)
            for values in by_formula.values()
        ) / len(by_formula),
        "formula_top5_oracle": sum(
            all(row["label"] in row["final_topk"] for row in values)
            for values in by_formula.values()
        ) / len(by_formula),
        "by_label": label_scores,
        "by_writer": writer_scores,
        "by_family": family_scores,
    }


def _paired(baseline: list[dict], candidate: list[dict]) -> dict:
    """같은 CROHME 레코드에서 개선·회귀·Top-5 구조·유실을 짝지어 센다."""

    before = {str(row["record_id"]): row for row in baseline}
    after = {str(row["record_id"]): row for row in candidate}
    if set(before) != set(after):
        raise ValueError("baseline and candidate CROHME coverage differs")
    result = Counter()
    changes = []
    for record_id in sorted(before):
        old, new = before[record_id], after[record_id]
        if old["label"] != new["label"] or old["formula_id"] != new["formula_id"]:
            raise ValueError(f"CROHME truth changed: {record_id}")
        truth = str(new["label"])
        old_top1 = old["final_topk"][0] == truth
        new_top1 = new["final_topk"][0] == truth
        old_top5 = truth in old["final_topk"]
        new_top5 = truth in new["final_topk"]
        result["top1_improved"] += not old_top1 and new_top1
        result["top1_regressed"] += old_top1 and not new_top1
        result["top5_rescued"] += not old_top5 and new_top5
        result["top5_lost"] += old_top5 and not new_top5
        result["persistent_outside_top5"] += not old_top5 and not new_top5
        if old["final_topk"] != new["final_topk"]:
            changes.append({
                "record_id": record_id,
                "truth": truth,
                "baseline_top1": old["final_topk"][0],
                "candidate_top1": new["final_topk"][0],
                "baseline_top5_hit": old_top5,
                "candidate_top5_hit": new_top5,
            })
    return {**dict(result), "changed_candidate_sets": len(changes), "changes": changes}


def _quadrant(phase: float) -> str:
    """폐곡선 시작 각도를 렌더링 기준 우·상·좌·하 사분면으로 나눈다."""

    degrees = math.degrees(phase % math.tau)
    if degrees >= 315.0 or degrees < 45.0:
        return "right"
    if degrees < 135.0:
        return "top"
    if degrees < 225.0:
        return "left"
    return "bottom"


def _direction_name(value: int) -> str:
    """부호 방향을 렌더링 y-up 좌표계의 사람이 읽을 이름으로 바꾼다."""

    return {-1: "clockwise_y_up", 0: "no_primary_loop", 1: "counterclockwise_y_up"}[value]


def _direction_audit(
    split_root: Path,
    checkpoint_path: Path,
    candidate_rows: list[dict],
    dataset_manifest: dict,
) -> dict:
    """모델 동결 뒤 CROHME 원형문자의 방향·시작점과 정확도를 평가 전용으로 계산한다."""

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    labels = list(checkpoint.get("math_labels", []))
    if len(labels) != 372:
        raise ValueError("direction audit requires the frozen 372-class checkpoint")
    items, _coverage, _formula_ids, _failed, _fallbacks, _writers = _crohme_items_tolerant(
        split_root, set(labels), "test",
    )
    predictions = {str(row["record_id"]): row for row in candidate_rows}
    target_labels = {"0", "O", "o", r"\circ", r"\mathcal{O}"}
    grouped: dict[str, list[dict]] = defaultdict(list)
    directions: Counter[str] = Counter()
    quadrants: Counter[str] = Counter()
    missing_predictions = 0
    for item in items:
        label = str(item["label"])
        if label not in target_labels:
            continue
        record_id = str(item["record_id"])
        prediction = predictions.get(record_id)
        if prediction is None:
            missing_predictions += 1
            continue
        descriptor = trajectory_descriptor(_prefix_tensor(item["strokes"]))
        direction = _direction_name(descriptor.loop_direction)
        quadrant = (
            _quadrant(descriptor.start_phase)
            if descriptor.loop_direction and np.isfinite(descriptor.start_phase)
            else "none"
        )
        row = {
            "record_id": record_id,
            "label": label,
            "direction": direction,
            "start_quadrant": quadrant,
            "top1": prediction["final_topk"][0] == label,
            "top5": label in prediction["final_topk"],
        }
        grouped[label].append(row)
        directions[direction] += 1
        quadrants[quadrant] += 1

    by_label = {}
    for label, rows in sorted(grouped.items()):
        direction_counts = Counter(row["direction"] for row in rows)
        quadrant_counts = Counter(row["start_quadrant"] for row in rows)
        by_label[label] = {
            "records": len(rows),
            "loop_detected": sum(row["direction"] != "no_primary_loop" for row in rows),
            "direction": dict(sorted(direction_counts.items())),
            "start_quadrant": dict(sorted(quadrant_counts.items())),
            "top1": sum(row["top1"] for row in rows) / len(rows),
            "top5": sum(row["top5"] for row in rows) / len(rows),
        }
    by_direction = {}
    all_rows = [row for rows in grouped.values() for row in rows]
    for direction in sorted({row["direction"] for row in all_rows}):
        rows = [row for row in all_rows if row["direction"] == direction]
        by_direction[direction] = {
            "records": len(rows),
            "top1": sum(row["top1"] for row in rows) / len(rows),
            "top5": sum(row["top5"] for row in rows) / len(rows),
        }
    return {
        "role": "post-freeze evaluation diagnostic only",
        "training_performed": False,
        "used_for_training": False,
        "used_for_variant_selection": False,
        "used_for_augmentation_parameters": False,
        "post_evaluation_retraining_permitted": False,
        "coordinate_convention": (
            "signed polygon area after model-input normalization; rendered with y increasing upward"
        ),
        "records": len(all_rows),
        "missing_predictions": missing_predictions,
        "direction": dict(sorted(directions.items())),
        "start_quadrant": dict(sorted(quadrants.items())),
        "by_label": by_label,
        "by_direction": by_direction,
        "approved_training_profiles": {
            "external": dataset_manifest["profiles"]["external_special_loop_classes"],
            "project_final": dataset_manifest["profiles"]["project_special_loop_classes"],
        },
    }


def _percentage(value: float) -> str:
    """비율을 Markdown 표에 쓸 소수 둘째 자리 백분율로 바꾼다."""

    return f"{100.0 * value:.2f}%"


def _markdown(report: dict) -> str:
    """최종 JSON의 핵심 결과와 방향 감사를 간결한 한국어 Markdown으로 렌더링한다."""

    before, after = report["crohme"]["baseline"], report["crohme"]["candidate"]
    fresh = report["fresh_acceptance"]
    writer = report["writer_loo"]
    paired = report["crohme"]["paired"]
    fresh_changes = report["fresh_acceptance"]["key_changes"]
    lines = [
        "# AIFlow Math Ink 1.0 clean-room 프로파일 증강 v3 결과",
        "",
        "## 결론",
        "",
        f"- CROHME 문자 Top-1: **{_percentage(before['top1'])} → {_percentage(after['top1'])}**",
        f"- CROHME Top-5: **{_percentage(before['top5'])} → {_percentage(after['top5'])}**",
        f"- Top-5 밖: **{before['outside_top5']} → {after['outside_top5']}건**",
        f"- 신규 작가 Top-1: **{_percentage(fresh['baseline']['top1'])} → {_percentage(fresh['candidate']['top1'])}**",
        f"- 신규 작가 식 exact: **{_percentage(fresh['baseline']['formula_exact'])} → {_percentage(fresh['candidate']['formula_exact'])}**",
        f"- writer-LOO Top-1: **{_percentage(writer['baseline']['top1'])} → {_percentage(writer['candidate']['top1'])}**",
        "- 모델·증강 데이터의 CROHME 학습 행, 통계, 오류 라벨 사용량은 모두 0이다.",
        "- 아래 CROHME 방향 통계는 체크포인트 동결 뒤 계산한 평가 진단이며 재학습에 사용하지 않는다.",
        "",
        "## 데이터셋",
        "",
        f"- 외부 프로파일 합성: {report['dataset']['external_records']:,}행",
        f"- 프로젝트 final 합성: {report['dataset']['project_final_records']:,}행",
        f"- 프로젝트 writer-LOO 합성: fold당 {report['dataset']['project_fold_records']:,}행",
        f"- 프로파일 학습 특징: {', '.join(report['dataset']['profile_features'])}",
        "- 모든 합성행에 부모 지문, DTW 거리, 클래스 프로파일 변형, 물리 시뮬레이션 적용 여부를 기록했다.",
        "",
        "## 게이트",
        "",
        "| 지표 | v2 | v3 | 변화 |",
        "|---|---:|---:|---:|",
        f"| writer-LOO Top-1 | {_percentage(writer['baseline']['top1'])} | {_percentage(writer['candidate']['top1'])} | {100*(writer['candidate']['top1']-writer['baseline']['top1']):+.2f}%p |",
        f"| writer-LOO Top-5 | {_percentage(writer['baseline']['top5'])} | {_percentage(writer['candidate']['top5'])} | {100*(writer['candidate']['top5']-writer['baseline']['top5']):+.2f}%p |",
        f"| 외부 holdout Top-1 | {_percentage(report['external_holdout']['baseline']['top1'])} | {_percentage(report['external_holdout']['candidate']['top1'])} | {100*(report['external_holdout']['candidate']['top1']-report['external_holdout']['baseline']['top1']):+.2f}%p |",
        f"| 신규 작가 Top-1 | {_percentage(fresh['baseline']['top1'])} | {_percentage(fresh['candidate']['top1'])} | {100*(fresh['candidate']['top1']-fresh['baseline']['top1']):+.2f}%p |",
        f"| 신규 작가 Top-5 | {_percentage(fresh['baseline']['top5'])} | {_percentage(fresh['candidate']['top5'])} | {100*(fresh['candidate']['top5']-fresh['baseline']['top5']):+.2f}%p |",
        "",
        f"- 신규 작가 Top-1 개선/회귀: **{fresh['delta']['improved_top1']} / {fresh['delta']['regressed_top1']}건**",
        "- 핵심 교환: " + "; ".join(
            f"`{row['truth']}`: `{row['baseline_top1']}` → `{row['candidate_top1']}`"
            for row in fresh_changes
        ),
        "",
        "## CROHME 최종 평가",
        "",
        "| 지표 | v2 | v3 | 변화 |",
        "|---|---:|---:|---:|",
        f"| 문자 Top-1 | {_percentage(before['top1'])} | {_percentage(after['top1'])} | {100*(after['top1']-before['top1']):+.2f}%p |",
        f"| 문자 Top-5 | {_percentage(before['top5'])} | {_percentage(after['top5'])} | {100*(after['top5']-before['top5']):+.2f}%p |",
        f"| strict macro Top-1 | {_percentage(before['strict_macro_top1'])} | {_percentage(after['strict_macro_top1'])} | {100*(after['strict_macro_top1']-before['strict_macro_top1']):+.2f}%p |",
        f"| 식 exact | {_percentage(before['formula_exact'])} | {_percentage(after['formula_exact'])} | {100*(after['formula_exact']-before['formula_exact']):+.2f}%p |",
        f"| 식 Top-5 oracle | {_percentage(before['formula_top5_oracle'])} | {_percentage(after['formula_top5_oracle'])} | {100*(after['formula_top5_oracle']-before['formula_top5_oracle']):+.2f}%p |",
        "",
        f"- Top-1 개선/회귀: **{paired['top1_improved']} / {paired['top1_regressed']}건**",
        f"- Top-5 구조/유실: **{paired['top5_rescued']} / {paired['top5_lost']}건**",
        f"- 지속 Top-5 밖: **{paired['persistent_outside_top5']}건**",
        "",
        "## 동결 후 원형문자 방향 진단",
        "",
        "| 문자 | 표본 | 폐곡선 검출 | Top-1 | Top-5 | 방향 분포 | 시작 사분면 |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for label, row in report["crohme_direction_audit"]["by_label"].items():
        lines.append(
            f"| `{label}` | {row['records']} | {row['loop_detected']} | "
            f"{_percentage(row['top1'])} | {_percentage(row['top5'])} | "
            f"{json.dumps(row['direction'], ensure_ascii=False)} | "
            f"{json.dumps(row['start_quadrant'], ensure_ascii=False)} |"
        )
    approved_zero = report["crohme_direction_audit"]["approved_training_profiles"]["external"].get("0")
    crohme_zero = report["crohme_direction_audit"]["by_label"].get("0")
    if approved_zero and crohme_zero:
        approved_direction = approved_zero["loop_direction"]
        lines.extend([
            "",
            "### `0` 방향 비교",
            "",
            f"- 승인 학습 분할: 폐곡선 {approved_zero['loop_records']}/{approved_zero['records']}건, "
            f"방향 {json.dumps(approved_direction, ensure_ascii=False)}",
            f"- CROHME 평가 분할: 폐곡선 {crohme_zero['loop_detected']}/{crohme_zero['records']}건, "
            f"방향 {json.dumps(crohme_zero['direction'], ensure_ascii=False)}",
            f"- 승인 학습 `0` 시작각 원형 평균: {approved_zero['start_phase']['mean_radians']:.3f} rad, "
            f"집중도 {approved_zero['start_phase']['concentration']:.3f}",
        ])
    lines.extend([
        "",
        "- 이 표는 CROHME를 증강 파라미터로 사용했다는 뜻이 아니다. 모델 동결 후 분포 차이를 설명하기 위한 격리된 평가 결과다.",
        "- 다음 모델 개선에 이 수치를 직접 쓰려면 CROHME benchmark를 폐기해야 하므로, 현재 상용 clean-room 계보에서는 사용하지 않는다.",
        "",
        "## 상태",
        "",
        f"- 신규 작가 acceptance: **{'통과' if fresh['passed'] else '실패'}**",
        f"- 외부 기술 비회귀: **{'통과' if report['external_holdout']['passed'] else '실패'}**",
        f"- 제품 runtime 전환: **아니오**",
        f"- v3 상태: **{report['adoption']['status']}**",
        "- 신규 작가 식 exact 회귀가 있으므로 문맥 결합 이전에도 제품 채택할 수 없다.",
        "",
    ])
    return "\n".join(lines)


def _self_test() -> None:
    """짝지은 개선 집계와 시작 사분면 경계를 검증한다."""

    base = [{"record_id": "a", "label": "0", "formula_id": "f", "final_topk": ["O", "0"]}]
    candidate = [{"record_id": "a", "label": "0", "formula_id": "f", "final_topk": ["0", "O"]}]
    paired = _paired(base, candidate)
    assert paired["top1_improved"] == 1 and paired["top1_regressed"] == 0
    assert _quadrant(0.0) == "right" and _quadrant(math.pi / 2.0) == "top"


def main() -> int:
    """입력 해시와 동결 경계를 검증한 뒤 최종 JSON·Markdown을 기록한다."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-candidates", type=Path)
    parser.add_argument("--candidate-candidates", type=Path)
    parser.add_argument("--candidate-build", type=Path)
    parser.add_argument("--training-report", type=Path)
    parser.add_argument("--dataset-manifest", type=Path)
    parser.add_argument("--fresh-acceptance", type=Path)
    parser.add_argument("--crohme-split-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        print(json.dumps({"self_test": "pass", "schema": SCHEMA}))
        return 0
    required = (
        args.baseline_candidates, args.candidate_candidates, args.candidate_build,
        args.training_report, args.dataset_manifest, args.fresh_acceptance,
        args.crohme_split_root, args.output, args.markdown,
    )
    if any(value is None for value in required):
        parser.error("all paths are required unless --self-test is used")

    baseline_path = _d_path(args.baseline_candidates, "baseline CROHME candidates")
    candidate_path = _d_path(args.candidate_candidates, "candidate CROHME candidates")
    build_path = _d_path(args.candidate_build, "candidate build report")
    training_path = _d_path(args.training_report, "v3 training report")
    dataset_path = _d_path(args.dataset_manifest, "v3 dataset manifest")
    fresh_path = _d_path(args.fresh_acceptance, "fresh acceptance")
    split_root = _d_path(args.crohme_split_root, "CROHME test split")
    output = _d_path(args.output, "analysis output", must_exist=False)
    markdown = _d_path(args.markdown, "analysis markdown", must_exist=False)
    if output.exists() or markdown.exists():
        parser.error("refusing to overwrite final analysis evidence")

    training = _json(training_path)
    dataset = _json(dataset_path)
    fresh = _json(fresh_path)
    build = _json(build_path)
    checkpoint = training.get("checkpoint", {})
    checkpoint_path = _d_path(Path(str(checkpoint.get("path", ""))), "frozen v3 checkpoint")
    if checkpoint.get("sha256") != _sha256(checkpoint_path):
        raise ValueError("frozen v3 checkpoint hash mismatch")
    if build.get("inputs", {}).get("hwr_checkpoint_sha256") != checkpoint.get("sha256"):
        raise ValueError("CROHME candidate build does not use the frozen v3 checkpoint")
    if build.get("training_performed") is not False or build.get("shape_hwr_frozen") is not True:
        raise ValueError("CROHME candidate build was not inference-only")
    if (
        training.get("selection_policy", {}).get("selection_used_crohme") is not False
        or training.get("clean_room_contract", {}).get("crohme_rows") != 0
        or training.get("clean_room_contract", {}).get("crohme_statistics") != 0
    ):
        raise ValueError("v3 training report is not CROHME-clean")
    if training.get("inputs", {}).get("cleanroom_dataset_manifest_sha256") != _sha256(dataset_path):
        raise ValueError("training report and materialized dataset manifest differ")
    if (
        fresh.get("training_performed") is not False
        or fresh.get("selection_used_fresh_acceptance") is not False
    ):
        raise ValueError("fresh acceptance evidence was used for training or selection")

    baseline_rows = _rows(baseline_path)
    candidate_rows = _rows(candidate_path)
    baseline_metrics = _score_rows(baseline_rows)
    candidate_metrics = _score_rows(candidate_rows)
    paired = _paired(baseline_rows, candidate_rows)
    direction = _direction_audit(split_root, checkpoint_path, candidate_rows, dataset)
    writer_baseline = training["variants"]["control_head"]["writer_disjoint"]
    selected_name = training["selection"]["selected"]
    writer_candidate = training["variants"][selected_name]["writer_disjoint"]
    report = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "training_performed": False,
        "checkpoint_frozen_before_crohme": True,
        "post_crohme_training_performed": False,
        "crohme_used_for_training": False,
        "crohme_used_for_variant_selection": False,
        "dataset": {
            "manifest_sha256": _sha256(dataset_path),
            "external_records": dataset["datasets"]["external"]["bank"]["records"],
            "project_final_records": dataset["datasets"]["project_final"]["bank"]["records"],
            "project_fold_records": next(iter(dataset["datasets"]["project_writer_loo"].values()))["bank"]["records"],
            "profile_features": dataset["clean_room_contract"]["profile_features"],
            "crohme_rows": dataset["clean_room_contract"]["crohme_rows"],
            "crohme_statistics": dataset["clean_room_contract"]["crohme_statistics"],
        },
        "writer_loo": {"baseline": writer_baseline, "candidate": writer_candidate},
        "external_holdout": {
            "baseline": training["external_technical_nonregression"]["base"],
            "candidate": training["external_technical_nonregression"]["candidate"],
            "passed": training["external_technical_nonregression"]["passed"],
        },
        "fresh_acceptance": {
            "baseline": fresh["baseline"], "candidate": fresh["candidate"],
            "delta": fresh["delta"], "gates": fresh["gates"], "passed": fresh["passed"],
            "key_changes": [
                row for row in fresh.get("changes", [])
                if row.get("improved") or row.get("regressed")
            ],
        },
        "crohme": {
            "baseline": baseline_metrics,
            "candidate": candidate_metrics,
            "paired": paired,
        },
        "crohme_direction_audit": direction,
        "adoption": {
            "frozen_cleanroom_challenger": bool(
                fresh["passed"] and training["external_technical_nonregression"]["passed"]
            ),
            "product_runtime_switched": False,
            "status": (
                "frozen_cleanroom_challenger"
                if fresh["passed"] else "shadow_rejected_by_fresh_acceptance"
            ),
            "reason": (
                "context checkpoint remains bound to the previous HWR hash"
                if fresh["passed"] else
                "fresh formula exact regressed and one Top-1 regression violated the fixed gate"
            ),
        },
        "inputs": {
            str(path): _sha256(path)
            for path in (
                baseline_path, candidate_path, build_path, training_path,
                dataset_path, fresh_path, checkpoint_path,
            )
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text(_markdown(report), encoding="utf-8", newline="\n")
    print(json.dumps({
        "event": "cleanroom_profiled_final_analysis_complete",
        "crohme_top1": candidate_metrics["top1"],
        "crohme_top5": candidate_metrics["top5"],
        "outside_top5": candidate_metrics["outside_top5"],
        "fresh_passed": fresh["passed"],
        "output": str(output),
        "markdown": str(markdown),
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
