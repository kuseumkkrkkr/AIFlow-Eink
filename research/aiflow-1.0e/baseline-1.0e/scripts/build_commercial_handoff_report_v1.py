#!/usr/bin/env python3
"""Consolidate the frozen HWR, whole-formula, and boundary evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from character_tensor_v1 import ROOT


INPUTS = {
    "outside_top5": ROOT / "artifacts" / "hwr_top5_outside_audit_20260821_r1" / "top5_outside_audit.json",
    "latin_training": ROOT / "artifacts" / "commercial_latin_auxiliary_20260821_r1_shadow" / "training_report.json",
    "latin_transfer": ROOT / "artifacts" / "commercial_latin_auxiliary_crohme_20260821_r1" / "validation_report.json",
    "formula_complete": ROOT / "artifacts" / "commercial_formula_complete_crohme_20260822_r2" / "formula_complete_validation_report.json",
    "frequency": ROOT / "artifacts" / "formula_complete_frequency_bias_20260821_r1" / "frequency_bias_report.json",
    "raw_formula_complete": ROOT / "artifacts" / "formula_complete_raw_crohme_20260822_r2" / "validation_report.json",
    "owned_nonregression": ROOT / "artifacts" / "formula_complete_owned_nonregression_20260822_r2" / "nonregression_report.json",
    "training_boundary": ROOT / "artifacts" / "training_boundary_audit_20260822_r4" / "training_boundary_audit.json",
    "collection_gate": ROOT / "artifacts" / "grouping_collection_gate_20260822_r1" / "collection_gate.json",
    "deepmind_v2": ROOT / "datasets" / "10_approved_external" / "deepmind_mathematics_dataset" / "derived" / "formula_context_v2_audit.json",
    "coverage_v2": ROOT / "datasets" / "00_project_owned" / "generated_context" / "candidate_validity_coverage_v2_audit.json",
}


def _load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _markdown(report: dict) -> str:
    shape = report["shape_classifier"]
    complete = report["whole_formula"]
    raw = report["raw_formula"]
    latin = report["commercial_latin_auxiliary"]
    frequency = report["frequency_bias"]
    collection = report["collection_requirement"]
    lines = [
        "# AIFlow Math Ink 1.0 상용화 경계 및 전체 수식 평가",
        "",
        "## 결론",
        "",
        "- **형태 HWR는 유지**한다. 다만 단독 Top-1 정확도를 상용 완성도로 보지 않는다.",
        "- **전체 수식 완료 뒤 한 번만** global grouping → 고정 HWR Top-5 → 후보 보존 문맥 → 2D 구조를 실행한다.",
        "- 현재 grouping/context 산출물은 **shadow**다. 결과는 제공하지만 제품 자동 확정은 하지 않고 `REVIEW_REQUIRED`와 원시 잉크 fallback을 반환한다.",
        "- 상업권리 영문 보조 헤드는 writer-disjoint dev가 부족하여 **미채택**이다.",
        "- CROHME는 모든 결과에서 **학습 0, 선택 0, gradient update 0**인 검증 전용이다.",
        "",
        "## 1. 형태 분류기와 Top-5 밖 749건",
        "",
        f"- 진실 그룹 기준 HWR Top-1: **{_percent(shape['top1'])}**",
        f"- Top-5: **{_percent(shape['top5'])}**",
        f"- 정답이 Top-5 밖: **{shape['outside_top5']} / {shape['records']}건**",
        f"- 상업권리/프로젝트 학습 지원 0건인 오류: **{shape['zero_support_errors']}건**",
        "- 따라서 749건은 ‘아예 학습하지 않은 문자’ 문제가 아니다. 같은 문자의 필기 도메인 차이, 과도한 372-class 경쟁, 동형·유사형 혼동이 핵심이다.",
        "",
        "| 문자 | Top-5 밖 | 정답 중앙 순위 | 상업권리+소유 학습 지원 |",
        "|---|---:|---:|---:|",
    ]
    for row in shape["largest_failures"]:
        lines.append(
            f"| `{row['truth']}` | {row['outside_top5']} | {row['median_truth_rank']} | {row['training_support']} |"
        )
    lines.extend([
        "",
        "## 2. 상업권리 영문 필기 보강",
        "",
        f"- fit **{latin['fit_records']}건 / {latin['fit_writers']} writers**, dev **{latin['dev_records']}건 / {latin['dev_writers']} writers**, writer overlap **{latin['writer_overlap']}**",
        f"- writer-disjoint dev Top-1/Top-5: **{_percent(latin['dev_top1'])} / {_percent(latin['dev_top5'])}**",
        f"- 최악 writer Top-1/Top-5: **{_percent(latin['worst_writer_top1'])} / {_percent(latin['worst_writer_top5'])}**",
        f"- CROHME의 기존 749건 중 보조 어휘로 평가 가능한 586건에서 Top-1 rescue **{latin['outside_top5_top1_rescue']}건**, Top-5 rescue **{latin['outside_top5_top5_rescue']}건**",
        "- 이 수치는 고정 후 전이 진단일 뿐 채택 근거가 아니다. 대소문자·숫자·점 형태 충돌이 커서 별도 제품 헤드로 승격하지 않는다.",
        "",
        "## 3. 스트리밍 대신 전체 수식 완료 기준",
        "",
        f"- 진실 grouping 제공 문자 Top-1: **{_percent(complete['hwr_top1'])} → {_percent(complete['context_top1'])}** ({complete['top1_delta_pp']:+.2f}%p)",
        f"- 문자열 exact proxy: **{_percent(complete['hwr_formula_exact'])} → {_percent(complete['context_formula_exact'])}** ({complete['formula_delta_pp']:+.2f}%p)",
        f"- 개선/퇴행: **{complete['improved']} / {complete['regressed']}**, 순증 **{complete['net_gain']}건**",
        "- 완료 전 확정과 prefix 정확도 평가는 모두 0이다.",
        "",
        "### 빈도 편향",
        "",
        f"- 전체 정확도는 +{frequency['overall_delta_pp']:.2f}%p, 과다예측 질량은 {frequency['excess_before']} → {frequency['excess_after']}로 감소했다.",
        f"- 그러나 진실 빈도 10~49 구간은 **{frequency['rare_mid_delta_pp']:+.2f}%p** 퇴행했다.",
        f"- 전체 퇴행 {frequency['regressed']}건 중 더 빈번한 문자로 이동한 경우는 {frequency['to_more_frequent']}건({frequency['to_more_frequent_rate']:.2f}%)이다.",
        "- 즉 전체 수식 판정은 스트리밍보다 낫지만 빈도 편향을 완전히 해결하지 못했다. 이 때문에 자동 확정이 아니라 review 경계를 유지한다.",
        "",
        "## 4. 원시 획부터 grouping·2D까지",
        "",
        f"- 대상: **{raw['eligible_formulas']}식**; truth label/writer/glyph count 입력 없음",
        f"- grouping exact: **{_percent(raw['grouping_exact'])}** ({raw['grouping_exact_count']}식)",
        f"- layout order exact: **{_percent(raw['layout_order_exact'])}**",
        f"- relation micro F1: **{_percent(raw['relation_f1'])}**",
        f"- strict grouping+layout+relation+문자 exact: **{_percent(raw['strict_exact'])}** ({raw['strict_exact_count']}식)",
        f"- 원시 획 fallback 일치: **{raw['raw_fallback_exact_formulas']} / {raw['eligible_formulas']}식**, 제품 자동 확정 **{raw['product_auto_commits']}건**",
        "- 병목은 문자 문맥이 아니라 **상업권리 grouping/2D 정답 부족**이다. CROHME truth를 학습에 넣어 수치를 올리지는 않는다.",
        "",
        "## 5. 보유 데이터 비회귀와 경계 감사",
        "",
        f"- 보유 110식: grouping/HWR/context/final projection mismatch **{report['owned_nonregression']['projection_mismatches']}건**",
        f"- 완료 전 확정 **{report['owned_nonregression']['precomplete_commits']}건**, 획 유실·중복 **{report['owned_nonregression']['stroke_failures']}건**, 원시 fallback mismatch **{report['owned_nonregression']['raw_fallback_mismatches']}건**",
        f"- 훈련 진입점 **{report['training_boundary']['trainers']}개** + 코퍼스 준비 **{report['training_boundary']['builders']}개**: 실패 **{report['training_boundary']['failed']}**, CROHME 오염 선택식 **{report['training_boundary']['selection_findings']}**",
        f"- CROHME 비참조 v2 코퍼스: DeepMind **{report['clean_corpora']['deepmind_formulas']}식**, 동형문자 coverage **{report['clean_corpora']['coverage_formulas']}식**",
        "- CROHME로 48식을 제외했던 DeepMind v1과 CROHME를 읽었던 coverage v1은 격리했다.",
        "",
        "## 6. 추가 수집량과 승격 조건",
        "",
        f"- 가정한 상용 grouping gate: 관측 exact ≥ **{_percent(collection['observed_target'])}**, 95% Wilson 하한 ≥ **{_percent(collection['lower_target'])}**",
        f"- 독립식 가정 최소 untouched acceptance: **{collection['minimum_iid_formulas']}식**",
        f"- writer당 최대 {collection['max_per_writer']}식으로 묶으면 실무 배치 **{collection['practical_batch']}식 / 최소 {collection['minimum_writers']} writers**",
        "- 이는 통계적 acceptance 하한이지 충분한 학습량 증명이 아니다. 새 training tranche도 160식 단위로 추가하고, 이미 본 acceptance 식을 모델·epoch·임계값 선택에 재사용하지 않는다.",
        "",
        "## 채택 상태",
        "",
        "| 구성요소 | 상태 | 이유 |",
        "|---|---|---|",
        "| 기존 상업권리 372-class HWR | 유지 | Top-5 93.75%; 전체 수식 후보 공급기로 유효 |",
        "| 상업권리 영문 auxiliary | 미채택 shadow | writer-disjoint dev Top-1 58.52% |",
        "| formula-complete 문맥 | shadow 적용 | 후보 보존 상태에서 aggregate 개선, rare-mid 편향 잔존 |",
        "| raw grouping/2D | 미채택 shadow | grouping exact 39.01%, strict exact 6.50% |",
        "| 제품 자동 확정 | 비활성 | `REVIEW_REQUIRED` + 원시 잉크 fallback |",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "artifacts" / "formula_complete_commercial_handoff_20260822_r1",
    )
    parser.add_argument(
        "--report", type=Path,
        default=ROOT / "reports" / "FORMULA_COMPLETE_COMMERCIAL_HWR_LOOP_20260822.md",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    report_path = args.report.resolve()
    if output.exists() or report_path.exists():
        parser.error("handoff output and report must be new")
    data = {name: _load(path) for name, path in INPUTS.items()}
    outside = data["outside_top5"]
    latin = data["latin_training"]
    transfer = data["latin_transfer"]
    complete = data["formula_complete"]
    frequency = data["frequency"]
    raw = data["raw_formula_complete"]
    owned = data["owned_nonregression"]
    boundary = data["training_boundary"]
    collection = data["collection_gate"]
    largest = outside["outside_top5"]["labels"][:12]
    report = {
        "schema": "aiflow-formula-complete-commercial-handoff/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "bounded_shadow_not_product_auto_commit",
        "shape_classifier": {
            "records": outside["baseline"]["supported_records"],
            "top1": outside["baseline"]["top1"],
            "top5": outside["baseline"]["top5"],
            "outside_top5": outside["baseline"]["outside_top5"],
            "zero_support_errors": outside["outside_top5"]["zero_commercial_support_rows"],
            "by_category": outside["outside_top5"]["by_category"],
            "largest_failures": [{
                "truth": row["truth"],
                "outside_top5": row["outside_top5"],
                "median_truth_rank": row["truth_rank"]["median"],
                "training_support": row["commercial_training_support"]["total"],
            } for row in largest],
            "unsupported_truth_groups": outside["unsupported_truth_groups"],
            "decision": "retain as frozen candidate generator; not a standalone product decision",
        },
        "commercial_latin_auxiliary": {
            "fit_records": latin["data"]["fit_records"],
            "dev_records": latin["data"]["dev_records"],
            "fit_writers": latin["data"]["selection"]["fit_writers"],
            "dev_writers": latin["data"]["selection"]["dev_writers"],
            "writer_overlap": latin["data"]["selection"]["writer_overlap"],
            "dev_top1": latin["selected_writer_disjoint_dev"]["top1"],
            "dev_top5": latin["selected_writer_disjoint_dev"]["top5"],
            "worst_writer_top1": latin["selected_writer_disjoint_dev"]["worst_writer_top1"],
            "worst_writer_top5": latin["selected_writer_disjoint_dev"]["worst_writer_top5"],
            "outside_top5_top1_rescue": transfer["metrics"]["frozen_749_main_outside_top5"]["auxiliary_top1_rescue"],
            "outside_top5_top5_rescue": transfer["metrics"]["frozen_749_main_outside_top5"]["auxiliary_top5_rescue"],
            "product_adopted": False,
        },
        "whole_formula": {
            "hwr_top1": complete["metrics"]["hwr_top1"]["all_top1"],
            "context_top1": complete["metrics"]["formula_complete_context"]["all_top1"],
            "top1_delta_pp": 100 * complete["metrics"]["delta"]["all_top1"],
            "hwr_formula_exact": complete["metrics"]["hwr_top1"]["formula_exact"],
            "context_formula_exact": complete["metrics"]["formula_complete_context"]["formula_exact"],
            "formula_delta_pp": 100 * complete["metrics"]["delta"]["formula_exact"],
            "improved": complete["metrics"]["formula_complete_context"]["improved"],
            "regressed": complete["metrics"]["formula_complete_context"]["regressed"],
            "net_gain": (
                complete["metrics"]["formula_complete_context"]["improved"]
                - complete["metrics"]["formula_complete_context"]["regressed"]
            ),
            "truth_grouping_supplied": True,
            "official_end_to_end": False,
        },
        "frequency_bias": {
            "overall_delta_pp": 100 * frequency["overall"]["delta"],
            "rare_mid_delta_pp": 100 * frequency["by_truth_frequency_band"]["10-49"]["delta"],
            "regressed": frequency["context_changes"]["regressed"],
            "to_more_frequent": frequency["context_changes"]["regressions_to_more_frequent_truth_token"],
            "to_more_frequent_rate": 100 * frequency["context_changes"]["regressions_to_more_frequent_truth_token_rate"],
            "excess_before": frequency["overprediction"]["absolute_excess_mass_hwr"],
            "excess_after": frequency["overprediction"]["absolute_excess_mass_formula_complete"],
        },
        "raw_formula": {
            "eligible_formulas": raw["coverage"]["eligible_formulas"],
            "grouping_exact_count": raw["scores"]["grouping_exact_count"],
            "grouping_exact": raw["scores"]["grouping_exact"],
            "layout_order_exact": raw["scores"]["layout_order_exact"],
            "relation_f1": raw["scores"]["relation_micro"]["f1"],
            "strict_exact_count": raw["scores"]["strict_group_layout_relation_character_exact_count"],
            "strict_exact": raw["scores"]["strict_group_layout_relation_character_exact"],
            "raw_fallback_exact_formulas": raw["contracts"]["raw_fallback_exact_formulas"],
            "product_auto_commits": raw["contracts"]["product_auto_commits"],
            "crohme_gradient_updates": raw["crohme_gradient_updates"],
        },
        "owned_nonregression": {
            "formulas": owned["formulas"],
            "projection_mismatches": owned["baseline_projection_mismatches"],
            "precomplete_commits": owned["pre_complete_commits"],
            "stroke_failures": owned["stroke_loss_or_duplication_formulas"],
            "raw_fallback_mismatches": owned["raw_fallback_mismatches"],
            "product_auto_commits": owned["product_auto_commits"],
        },
        "training_boundary": {
            "passed": boundary["passed"],
            "trainers": boundary["summary"]["trainers"],
            "builders": boundary["summary"]["corpus_builders"],
            "failed": boundary["summary"]["failed"],
            "selection_findings": len(boundary["selection_findings"]),
        },
        "clean_corpora": {
            "deepmind_formulas": data["deepmind_v2"]["generation"]["unique_formulas"],
            "deepmind_crohme_used": data["deepmind_v2"]["contracts"]["crohme_used_for_generation_or_filtering"],
            "coverage_formulas": data["coverage_v2"]["generation"]["records"],
            "coverage_crohme_used": data["coverage_v2"]["contracts"]["crohme_used_for_generation_or_filtering"],
        },
        "collection_requirement": {
            "observed_target": collection["gate"]["observed_partition_exact_target"],
            "lower_target": collection["gate"]["wilson_lower_bound_target"],
            "minimum_iid_formulas": collection["gate"]["minimum_iid_formulas"],
            "practical_batch": collection["gate"]["practical_formula_batch_at_writer_cap"],
            "max_per_writer": collection["gate"]["max_formulas_per_writer"],
            "minimum_writers": collection["gate"]["minimum_writers_at_cap"],
            "training_quantity_proven": False,
        },
        "adoption": {
            "frozen_hwr": "retain",
            "commercial_latin_auxiliary": "shadow_not_adopted",
            "formula_complete_context": "shadow_candidate_preserving",
            "raw_grouping_2d": "shadow_not_adopted",
            "product_auto_commit": False,
            "fallback": "REVIEW_REQUIRED_with_immutable_raw_ink",
        },
        "inputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in INPUTS.items()
        },
    }
    output.mkdir(parents=True)
    json_path = output / "commercial_handoff.json"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_markdown(report), encoding="utf-8", newline="\n")
    print(json.dumps({
        "handoff": str(json_path), "report": str(report_path),
        "status": report["status"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
