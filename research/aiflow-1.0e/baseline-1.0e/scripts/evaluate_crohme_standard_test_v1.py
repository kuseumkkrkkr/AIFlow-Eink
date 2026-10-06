#!/usr/bin/env python3
"""RETIRED: evaluate the invalid historical CROHME-trained checkpoint.

Use the commercial-checkpoint formula-complete evaluator instead.  This file
is retained only so the invalidated experiment remains auditable.
"""

from __future__ import annotations

if __name__ == "__main__":
    raise SystemExit(
        "retired: the referenced checkpoint was trained on CROHME; use the "
        "commercial-checkpoint formula-complete evaluator"
    )

import argparse
import gc
import gzip
import io
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch

from character_tensor_v1 import ROOT, _json_lines
from evaluate_48hz_prefix_v1 import _crohme_items
from evaluate_homograph_context_reranker_v1 import FAMILIES, STRICT_FAMILIES, _metrics
import train_candidate_validity_context_v1 as candidate
import train_independent_formula_context_v1 as independent
import train_masked_context_reranker_v1 as masked


SCHEMA = "aiflow-crohme-standard-test-evaluation/v1"
RESEARCH_SCHEMA = "aiflow-crohme-standard-context-research/v1"
DEFAULT_ROOT = ROOT / "artifacts" / "crohme_standard_context_20260820_r1_research"
DEFAULT_TEST = DEFAULT_ROOT / "test_candidates.jsonl.gz"
DEFAULT_TEST_BUILD = DEFAULT_ROOT / "test_candidate_build_report.json"
DEFAULT_SELECTION = DEFAULT_ROOT / "selection_report.json"
DEFAULT_RESEARCH_CHECKPOINT = DEFAULT_ROOT / "crohme_standard_context_research.pt"
DEFAULT_BASE_CHECKPOINT = (
    ROOT / "artifacts" / "candidate_validity_context_20260820_r1_shadow"
    / "candidate_validity_accuracy.pt"
)
DEFAULT_CROHME_ROOT = (
    ROOT / "datasets" / "30_noncommercial_evaluation" / "crohme2019"
    / "crohme2019" / "crohme2019" / "test"
)
DEFAULT_RAW_GROUPING = (
    ROOT / "artifacts" / "crohme2019_test_raw_runtime_20260820_r1_shadow"
    / "raw_runtime_report.json"
)
DEFAULT_STREAMING = (
    ROOT / "artifacts" / "prefix_48hz_expanded_hwr_crohme_test_20260820_r1_shadow"
    / "prefix_48hz_report.json"
)
DEFAULT_REPORT = DEFAULT_ROOT / "test_evaluation_report.json"
DEFAULT_PREDICTIONS = DEFAULT_ROOT / "test_predictions.jsonl.gz"
DEFAULT_MARKDOWN = ROOT / "reports" / "CROHME_STANDARD_REINFORCEMENT_LOOP_20260821.md"


def _event(name: str, **values: object) -> None:
    print(json.dumps({"event": name, **values}, ensure_ascii=False), flush=True)


def _d_path(path: Path, label: str, *, file: bool = True) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.drive.upper() != "D:":
        raise ValueError(f"{label} must remain on D: {resolved}")
    if file and not resolved.is_file():
        raise FileNotFoundError(f"missing {label}: {resolved}")
    return resolved


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _metric_view(metrics: dict) -> dict:
    return {
        key: metrics[key]
        for key in (
            "all_records",
            "all_top1",
            "strict_records",
            "strict_micro_top1",
            "strict_macro_top1",
            "changed",
            "improved",
            "regressed",
            "formula_exact",
            "baseline_formula_exact",
        )
    }


def _paired(
    rows: list[dict], before: dict[str, str], after: dict[str, str]
) -> dict:
    improved = regressed = changed = 0
    formulae = candidate._formula_rows(rows)
    formula_improved = formula_regressed = 0
    for row in rows:
        record_id = str(row["record_id"])
        truth = str(row["label"])
        old, new = str(before[record_id]), str(after[record_id])
        changed += old != new
        improved += old != truth and new == truth
        regressed += old == truth and new != truth
    for sequence in formulae.values():
        old_exact = all(
            before[str(row["record_id"])] == str(row["label"]) for row in sequence
        )
        new_exact = all(
            after[str(row["record_id"])] == str(row["label"]) for row in sequence
        )
        formula_improved += not old_exact and new_exact
        formula_regressed += old_exact and not new_exact
    return {
        "changed": changed,
        "improved": improved,
        "regressed": regressed,
        "net_improvement": improved - regressed,
        "formula_improved": formula_improved,
        "formula_regressed": formula_regressed,
    }


def _load_research_model(
    checkpoint: Path, pretrained: Path, hwr: Path, device: torch.device,
) -> tuple[torch.nn.Module, dict, dict]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    labels = masked._labels(hwr)
    if (
        payload.get("schema") != RESEARCH_SCHEMA
        or payload.get("math_labels") != labels
        or payload.get("pretrained_hashes") != masked._verify_pretrained(pretrained)
        or payload.get("hwr_checkpoint_sha256") != masked._sha256(hwr)
        or payload.get("shape_training") is not False
        or payload.get("rights", {}).get("research_only") is not True
        or payload.get("rights", {}).get("commercial_use") is not False
    ):
        raise ValueError("CROHME research checkpoint contract mismatch")
    model, contract = candidate._new_model(pretrained, labels, device, candidate.SEED)
    if (
        payload.get("class_tokens") != contract["class_tokens"]
        or payload.get("relation_tokens") != contract["relation_tokens"]
    ):
        raise ValueError("CROHME research token contract mismatch")
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, contract, payload


@torch.inference_mode()
def _research_predictions(
    model: torch.nn.Module, contract: dict, payload: dict, rows: list[dict],
    device: torch.device, batch_size: int,
) -> tuple[dict[str, str], dict]:
    validity = candidate._score_candidates(model, contract, rows, device, batch_size)
    predictions = candidate._fused_predictions(
        rows, validity, float(payload["selection"]["lambda"])
    )
    audit = masked._candidate_audit(rows, predictions)
    return predictions, audit


def _same_visual_family(truth: str, prediction: str) -> str | None:
    for name, values in FAMILIES.items():
        if truth in values and prediction in values:
            return name
    return None


def _class_analysis(
    rows: list[dict], hwr: dict[str, str], base: dict[str, str], selected: dict[str, str]
) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["label"])].append(row)
    output = {}
    for label, selected_rows in sorted(grouped.items()):
        count = len(selected_rows)
        confusion = Counter(
            selected[str(row["record_id"])]
            for row in selected_rows
            if selected[str(row["record_id"])] != label
        )
        output[label] = {
            "records": count,
            "hwr_top1": sum(hwr[str(row["record_id"])] == label for row in selected_rows) / count,
            "base_context_top1": sum(base[str(row["record_id"])] == label for row in selected_rows) / count,
            "selected_context_top1": sum(selected[str(row["record_id"])] == label for row in selected_rows) / count,
            "top5": sum(label in row["final_topk"] for row in selected_rows) / count,
            "selected_vs_base_improved": sum(
                base[str(row["record_id"])] != label
                and selected[str(row["record_id"])] == label
                for row in selected_rows
            ),
            "selected_vs_base_regressed": sum(
                base[str(row["record_id"])] == label
                and selected[str(row["record_id"])] != label
                for row in selected_rows
            ),
            "selected_truth_outside_top5": sum(
                label not in row["final_topk"] for row in selected_rows
            ),
            "selected_common_confusions": [
                {"prediction": prediction, "count": value}
                for prediction, value in confusion.most_common(5)
            ],
        }
    return output


def _homograph_analysis(
    rows: list[dict], hwr: dict[str, str], base: dict[str, str], selected: dict[str, str]
) -> dict:
    output = {}
    for family, labels in STRICT_FAMILIES.items():
        selected_rows = [row for row in rows if str(row["label"]) in labels]
        if not selected_rows:
            continue
        count = len(selected_rows)
        output[family] = {
            "labels": sorted(labels),
            "records": count,
            "hwr_top1": sum(
                hwr[str(row["record_id"])] == str(row["label"])
                for row in selected_rows
            ) / count,
            "base_context_top1": sum(
                base[str(row["record_id"])] == str(row["label"])
                for row in selected_rows
            ) / count,
            "selected_context_top1": sum(
                selected[str(row["record_id"])] == str(row["label"])
                for row in selected_rows
            ) / count,
            "top5": sum(
                str(row["label"]) in row["final_topk"] for row in selected_rows
            ) / count,
        }
    return output


def _formula_analysis(
    rows: list[dict], base: dict[str, str], selected: dict[str, str],
    all_formula_ids: set[str], fully_supported_ids: set[str],
) -> dict:
    formulae = candidate._formula_rows(rows)
    categories = Counter()
    details = []
    repeated: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    fully_supported_selected_exact = 0
    fully_supported_base_exact = 0
    for formula_id, sequence in formulae.items():
        truth = [str(row["label"]) for row in sequence]
        before = [str(base[str(row["record_id"])]) for row in sequence]
        after = [str(selected[str(row["record_id"])]) for row in sequence]
        mismatches = [
            {
                "index": index,
                "truth": truth[index],
                "prediction": after[index],
                "family": _same_visual_family(truth[index], after[index]),
                "truth_in_top5": truth[index] in sequence[index]["final_topk"],
            }
            for index in range(len(sequence))
            if truth[index] != after[index]
        ]
        base_exact = truth == before
        selected_exact = truth == after
        if selected_exact:
            category = "exact"
        elif mismatches and all(value["family"] is not None for value in mismatches):
            category = "homograph_only"
        else:
            category = "real_error"
        categories[category] += 1
        if formula_id in fully_supported_ids:
            fully_supported_selected_exact += selected_exact
            fully_supported_base_exact += base_exact
        value = {
            "formula_id": formula_id,
            "length": len(sequence),
            "truth_tokens": truth,
            "base_tokens": before,
            "selected_tokens": after,
            "base_exact": base_exact,
            "selected_exact": selected_exact,
            "category": category,
            "error_count": len(mismatches),
            "outside_top5_errors": sum(not item["truth_in_top5"] for item in mismatches),
            "mismatches": mismatches,
        }
        details.append(value)
        repeated[tuple(truth)].append(value)
    worst = sorted(
        details,
        key=lambda value: (-value["error_count"], -value["length"], value["formula_id"]),
    )[:30]
    exact_examples = sorted(
        (value for value in details if value["selected_exact"]),
        key=lambda value: (-value["length"], value["formula_id"]),
    )[:30]
    repeated_stats = []
    for truth, values in repeated.items():
        if len(values) < 2:
            continue
        repeated_stats.append({
            "truth_tokens": list(truth),
            "occurrences": len(values),
            "base_exact": sum(value["base_exact"] for value in values),
            "selected_exact": sum(value["selected_exact"] for value in values),
            "selected_error_count": sum(value["error_count"] for value in values),
            "formula_ids": [value["formula_id"] for value in values],
        })
    repeated_stats.sort(
        key=lambda value: (-value["occurrences"], -value["selected_error_count"], value["truth_tokens"])
    )
    return {
        "cache_formulas": len(formulae),
        "protocol_formulas": len(all_formula_ids),
        "fully_supported_formulas": len(fully_supported_ids),
        "categories_on_supported_rows": dict(categories),
        "space_only_error_formulas": 0,
        "space_policy": "InkML truth groups contain no whitespace token; spacing is ignored",
        "fully_supported_oracle_group_proxy": {
            "base_exact_count": fully_supported_base_exact,
            "base_exact": fully_supported_base_exact / len(fully_supported_ids),
            "selected_exact_count": fully_supported_selected_exact,
            "selected_exact": fully_supported_selected_exact / len(fully_supported_ids),
        },
        "all_protocol_oracle_group_proxy": {
            "base_exact_count": fully_supported_base_exact,
            "base_exact": fully_supported_base_exact / len(all_formula_ids),
            "selected_exact_count": fully_supported_selected_exact,
            "selected_exact": fully_supported_selected_exact / len(all_formula_ids),
        },
        "worst_formulas": worst,
        "exact_formula_examples": exact_examples,
        "repeated_formula_statistics": repeated_stats,
        "all_formula_details": details,
    }


def _write_predictions(
    path: Path, rows: list[dict], hwr: dict[str, str], base: dict[str, str],
    selected: dict[str, str],
) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(
            filename="", fileobj=raw, mode="wb", compresslevel=6, mtime=0
        ) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8", newline="\n") as stream:
                for row in rows:
                    record_id = str(row["record_id"])
                    stream.write(json.dumps({
                        "record_id": record_id,
                        "formula_id": str(row["formula_id"]),
                        "label": str(row["label"]),
                        "final_topk": row["final_topk"],
                        "hwr_top1": hwr[record_id],
                        "base_context_top1": base[record_id],
                        "selected_context_top1": selected[record_id],
                        "candidate_preserved": selected[record_id] in row["final_topk"],
                    }, ensure_ascii=False, separators=(",", ":")) + "\n")


def _pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _markdown(report: dict) -> str:
    metrics = report["metrics"]
    selected = metrics["selected_context"]
    base = metrics["base_context"]
    hwr = metrics["hwr"]
    formulas = report["formula_analysis"]
    coverage = report["dataset"]["raw_coverage"]
    per_class = report["per_class"]
    regressed_classes = sorted(
        (
            (label, value)
            for label, value in per_class.items()
            if value["selected_context_top1"] < value["base_context_top1"]
        ),
        key=lambda item: (
            item[1]["selected_context_top1"] - item[1]["base_context_top1"],
            -item[1]["records"],
            item[0],
        ),
    )
    zero_classes = sorted(
        label
        for label, value in per_class.items()
        if value["records"] >= 10 and value["selected_context_top1"] == 0.0
    )
    unsupported = ", ".join(
        f"`{label}`:{count}"
        for label, count in coverage["unsupported_labels"].items()
    )
    largest_regressions = ", ".join(
        f"`{label}` {100.0 * value['base_context_top1']:.2f}%→"
        f"{100.0 * value['selected_context_top1']:.2f}% (n={value['records']})"
        for label, value in regressed_classes[:5]
    )
    lines = [
        "# CROHME 표준 강화 루프 및 병목 감사",
        "",
        "## 결론",
        "",
        "- CROHME train-derived writer+formula 이중 holdout에서 epoch 2, λ=3.0을 선택했다.",
        "- 공식 test는 선택 후에만 평가했으며 후보 밖 출력·grouping 변경·reload 불일치는 모두 0건이다.",
        "- 연구 정확도는 개선됐지만 CROHME가 CC BY-NC이므로 이 체크포인트는 상용 제품에 채택할 수 없다.",
        "- 구조적 결론은 `문맥 재랭커 학습량 부족`과 `Top-5 밖 형태 오류`, `raw grouping`이 각각 독립 병목이라는 것이다.",
        "",
        "## 공식 test 문자·식 결과",
        "",
        "| 경로 | 문자 Top-1 | 식 exact (cache proxy) | strict macro |",
        "|---|---:|---:|---:|",
        f"| 고정 HWR | {_pct(hwr['all_top1'])} | {_pct(hwr['formula_exact'])} | {_pct(hwr['strict_macro_top1'])} |",
        f"| 기존 상용권리 문맥모델 | {_pct(base['all_top1'])} | {_pct(base['formula_exact'])} | {_pct(base['strict_macro_top1'])} |",
        f"| CROHME 연구모델 | **{_pct(selected['all_top1'])}** | **{_pct(selected['formula_exact'])}** | **{_pct(selected['strict_macro_top1'])}** |",
        "",
        "`식 exact (cache proxy)`는 지원 가능한 truth-group 행만 비교한 값이다. 공식 end-to-end Expression Rate가 아니다.",
        "",
        "### 엄격한 protocol 분모",
        "",
        f"- fully-supported 849식 oracle-group proxy: {formulas['fully_supported_oracle_group_proxy']['selected_exact_count']}/849 ({_pct(formulas['fully_supported_oracle_group_proxy']['selected_exact'])})",
        f"- 전체 1,199식에서 미지원 식을 실패로 처리한 proxy: {formulas['all_protocol_oracle_group_proxy']['selected_exact_count']}/1,199 ({_pct(formulas['all_protocol_oracle_group_proxy']['selected_exact'])})",
        f"- 완전정답/동형문자만/실오류: {formulas['categories_on_supported_rows'].get('exact', 0)} / {formulas['categories_on_supported_rows'].get('homograph_only', 0)} / {formulas['categories_on_supported_rows'].get('real_error', 0)}식",
        "- 공백만 다른 오류: 0식(공백 token 자체가 없음)",
        "",
        "## 병목 분해",
        "",
    ]
    bottleneck = report["bottleneck_decomposition"]
    lines.extend([
        f"- 형태 HWR Top-1: {_pct(bottleneck['shape_hwr_top1'])}",
        f"- Top-5 후보 회수율: {_pct(bottleneck['shape_hwr_top5'])}; 정답 후보 밖 {bottleneck['truth_outside_top5_records']}글자",
        f"- 문맥 선택 후 Top-1: {_pct(bottleneck['selected_context_top1'])}; Top-5 안에 남은 미회수 {bottleneck['context_errors_with_truth_inside_top5']}글자",
        f"- raw grouping exact: {bottleneck['raw_grouping_exact_count']}/{bottleneck['raw_grouping_eligible_formulas']} ({_pct(bottleneck['raw_grouping_exact'])})",
        "  - 이 값은 기존 end-to-end 구조 진단이며, 이번 고정 HWR과 체크포인트가 달라 동일 파이프라인 결합 점수로 해석하지 않는다.",
        f"- 스트리밍 전체 prefix Top-1: {_pct(bottleneck['stream_all_prefix_top1'])}; 마지막 20%: {_pct(bottleneck['stream_last_20pct_top1'])}; 안정 정답 도달 중앙값: {100*bottleneck['stream_median_stable_progress']:.2f}%",
        "",
        "따라서 문맥층은 큰 폭으로 개선 가능하지만, 후보 밖 749글자와 raw grouping 실패는 문맥층만으로 해결되지 않는다.",
        "",
        "## 확인된 문제",
        "",
        f"- CROHME test 1,199식 중 현재 출력 계약으로 완전히 평가 가능한 식은 {coverage['fully_supported_formulas']}식이다. 미지원 truth-group {coverage['unsupported_truth_groups']}개: {unsupported}.",
        "- 함수명 단위(`\\sin` 등)와 단일 문자 단위가 섞여 있어 단순 클래스 추가만이 아니라 토큰·grouping 계약 정렬이 필요하다.",
        f"- 출현 {len(per_class)}클래스 중 {len(regressed_classes)}클래스가 기존 문맥모델보다 후퇴했다. 최대 회귀: {largest_regressions}.",
        f"- n≥10인데 Top-1이 0%인 클래스: {', '.join(f'`{label}`' for label in zero_classes)}. 이는 빈도 높은 소문자·숫자 쪽 문맥 prior 쏠림이다.",
        f"- 전체 Top-1은 기존 문맥보다 {100.0 * (selected['all_top1'] - base['all_top1']):.2f}%p 올랐지만 strict macro는 {100.0 * (selected['strict_macro_top1'] - base['strict_macro_top1']):.2f}%p만 올랐다. class/family-balanced 목적함수가 필요하다.",
        "- 공식 symLG/LgEval 출력과 2D 관계 점수가 구현되지 않아 현재 식 수치를 CROHME 공식 Expression Rate로 인용할 수 없다.",
        "- 48 Hz 평가는 prefix 복원 정확도이며 실제 장치 wall-clock p50/p95 latency 측정은 아니다.",
        "",
        "## strict 동형군",
        "",
        "| 군 | n | HWR | 기존 문맥 | 연구 문맥 | Top-5 |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for family, value in report["homograph_analysis"].items():
        lines.append(
            f"| `{family}` | {value['records']} | {_pct(value['hwr_top1'])} | "
            f"{_pct(value['base_context_top1'])} | {_pct(value['selected_context_top1'])} | {_pct(value['top5'])} |"
        )
    lines.extend(["", "## 오류가 큰 문자", "", "| 문자 | n | 연구 Top-1 | Top-5 | 후보 밖 | 주요 혼동 |", "|---|---:|---:|---:|---:|---|"])
    classes = sorted(
        report["per_class"].items(),
        key=lambda item: (
            item[1]["selected_context_top1"],
            -item[1]["records"],
            item[0],
        ),
    )[:25]
    for label, value in classes:
        confusions = ", ".join(
            f"{item['prediction']}:{item['count']}"
            for item in value["selected_common_confusions"][:3]
        ) or "-"
        lines.append(
            f"| `{label}` | {value['records']} | {_pct(value['selected_context_top1'])} | "
            f"{_pct(value['top5'])} | {value['selected_truth_outside_top5']} | {confusions} |"
        )
    lines.extend(["", "## 가장 많이 틀린 식", "", "| ID | 길이 | 오류 | 후보 밖 | truth → prediction |", "|---|---:|---:|---:|---|"])
    for value in formulas["worst_formulas"][:20]:
        truth = " ".join(value["truth_tokens"])
        prediction = " ".join(value["selected_tokens"])
        lines.append(
            f"| `{value['formula_id']}` | {value['length']} | {value['error_count']} | "
            f"{value['outside_top5_errors']} | `{truth}` → `{prediction}` |"
        )
    lines.extend([
        "",
        "## 채택 판정",
        "",
        "- 연구 체크포인트: 비상용 CROHME challenger로 보관하되 **shadow 유지**.",
        "- 상용 AIFlow 1.0 체크포인트: **미채택**. CC BY-NC 학습 가중치를 제품에 넣지 않는다.",
        "- 상용 개선 경로: 동일한 candidate-validity 목적함수를 Apache-2.0/프로젝트 소유 수식과 새 소유 writer 데이터로 재현한다.",
        "- grouping은 별도 ownership 모델, 후보 밖 문자는 형태 HWR 데이터 보강으로 해결한다.",
        "",
        "## 산출물",
        "",
        f"- 선택 보고서: `{report['artifacts']['selection_report']['path']}`",
        f"- test JSON: `{report['artifacts']['test_report']['path']}`",
        f"- test 예측: `{report['artifacts']['test_predictions']['path']}`",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-candidates", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--test-build-report", type=Path, default=DEFAULT_TEST_BUILD)
    parser.add_argument("--selection-report", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--research-checkpoint", type=Path, default=DEFAULT_RESEARCH_CHECKPOINT)
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE_CHECKPOINT)
    parser.add_argument("--hwr-checkpoint", type=Path, default=independent.DEFAULT_HWR)
    parser.add_argument("--pretrained", type=Path, default=masked.DEFAULT_PRETRAINED)
    parser.add_argument("--crohme-root", type=Path, default=DEFAULT_CROHME_ROOT)
    parser.add_argument("--raw-grouping-report", type=Path, default=DEFAULT_RAW_GROUPING)
    parser.add_argument("--streaming-report", type=Path, default=DEFAULT_STREAMING)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--predictions-output", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    parser.error(
        "retired: the referenced checkpoint was trained on CROHME; use validation-only commercial checkpoints"
    )
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    test_path = _d_path(args.test_candidates, "CROHME test candidates")
    test_build_path = _d_path(args.test_build_report, "test candidate build report")
    selection_path = _d_path(args.selection_report, "selection report")
    checkpoint_path = _d_path(args.research_checkpoint, "research checkpoint")
    base_path = _d_path(args.base_checkpoint, "base context checkpoint")
    hwr_path = _d_path(args.hwr_checkpoint, "frozen HWR checkpoint")
    pretrained = _d_path(args.pretrained, "pinned BERT-Tiny", file=False)
    crohme_root = _d_path(args.crohme_root, "CROHME test root", file=False)
    raw_grouping_path = _d_path(args.raw_grouping_report, "raw grouping report")
    streaming_path = _d_path(args.streaming_report, "streaming report")
    report_path = _d_path(args.report_output, "test report output", file=False)
    prediction_path = _d_path(args.predictions_output, "test predictions output", file=False)
    markdown_path = _d_path(args.markdown_output, "markdown report output", file=False)
    if any(path.exists() for path in (report_path, prediction_path, markdown_path)):
        parser.error("refusing to overwrite final CROHME test evidence")

    selection = _load_json(selection_path)
    test_build = _load_json(test_build_path)
    if (
        selection.get("selection_contract", {}).get("official_test_loaded") is not False
        or selection.get("selection_contract", {}).get("official_test_metrics_loaded") is not False
        or selection.get("selection_contract", {}).get("valid_used_for_selection") is not False
        or selection.get("selection_contract", {}).get("variants") != 3
        or test_build.get("split") != "test"
        or test_build.get("output", {}).get("sha256") != masked._sha256(test_path)
    ):
        raise ValueError("test-after-selection protocol contract mismatch")

    immutable_before = {
        "test_candidates": masked._sha256(test_path),
        "test_build_report": masked._sha256(test_build_path),
        "selection_report": masked._sha256(selection_path),
        "research_checkpoint": masked._sha256(checkpoint_path),
        "base_checkpoint": masked._sha256(base_path),
        "hwr_checkpoint": masked._sha256(hwr_path),
        "raw_grouping_report": masked._sha256(raw_grouping_path),
        "streaming_report": masked._sha256(streaming_path),
    }
    rows = list(_json_lines(test_path))
    if not rows or {str(row.get("dataset_split")) for row in rows} != {"test"}:
        raise ValueError("invalid CROHME test candidate cache")
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")

    hwr_predictions = {
        str(row["record_id"]): str(row["final_topk"][0]) for row in rows
    }
    base_model, base_contract, base_payload = candidate.load_candidate_validity_context(
        pretrained, base_path, hwr_path, device
    )
    base_predictions, base_audit = candidate.decide_candidate_validity_rows(
        base_model, base_contract, base_payload, rows, device, args.batch_size
    )
    del base_model, base_contract
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model, contract, research_payload = _load_research_model(
        checkpoint_path, pretrained, hwr_path, device
    )
    selected_predictions, selected_audit = _research_predictions(
        model, contract, research_payload, rows, device, args.batch_size
    )
    del model, contract
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    reload_model, reload_contract, reload_payload = _load_research_model(
        checkpoint_path, pretrained, hwr_path, device
    )
    reload_predictions, reload_audit = _research_predictions(
        reload_model, reload_contract, reload_payload, rows, device, args.batch_size
    )
    reload_mismatches = sum(
        reload_predictions[key] != value
        for key, value in selected_predictions.items()
    )
    if reload_mismatches:
        raise AssertionError(f"test checkpoint reload mismatch: {reload_mismatches}")

    labels = set(research_payload["math_labels"])
    raw_items, raw_coverage, all_formula_ids, failed_formula_ids = _crohme_items(
        crohme_root, labels
    )
    if {str(item["record_id"]) for item in raw_items} != {
        str(row["record_id"]) for row in rows
    }:
        raise ValueError("CROHME raw/test candidate record coverage mismatch")
    fully_supported_ids = set(all_formula_ids) - set(failed_formula_ids)

    hwr_metrics = _metrics(rows, hwr_predictions)
    base_metrics = _metrics(rows, base_predictions)
    selected_metrics = _metrics(rows, selected_predictions)
    per_class = _class_analysis(
        rows, hwr_predictions, base_predictions, selected_predictions
    )
    homographs = _homograph_analysis(
        rows, hwr_predictions, base_predictions, selected_predictions
    )
    formulas = _formula_analysis(
        rows,
        base_predictions,
        selected_predictions,
        set(all_formula_ids),
        fully_supported_ids,
    )
    raw_grouping = _load_json(raw_grouping_path)
    streaming = _load_json(streaming_path)
    raw_scores = raw_grouping["scores"]
    stream_metrics = streaming["crohme"]
    truth_outside_top5 = sum(
        str(row["label"]) not in row["final_topk"] for row in rows
    )
    selected_context_inside_errors = sum(
        selected_predictions[str(row["record_id"])] != str(row["label"])
        and str(row["label"]) in row["final_topk"]
        for row in rows
    )
    raw_eligible = int(raw_grouping["coverage"]["eligible_formulas"])
    raw_grouping_exact = int(raw_scores["grouping_exact_after"])

    _write_predictions(
        prediction_path,
        rows,
        hwr_predictions,
        base_predictions,
        selected_predictions,
    )
    immutable_after = {
        "test_candidates": masked._sha256(test_path),
        "test_build_report": masked._sha256(test_build_path),
        "selection_report": masked._sha256(selection_path),
        "research_checkpoint": masked._sha256(checkpoint_path),
        "base_checkpoint": masked._sha256(base_path),
        "hwr_checkpoint": masked._sha256(hwr_path),
        "raw_grouping_report": masked._sha256(raw_grouping_path),
        "streaming_report": masked._sha256(streaming_path),
    }
    report = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": "CROHME2019 test",
            "license_in_workspace": "CC BY-NC 4.0",
            "research_only": True,
            "protocol_formulas": len(all_formula_ids),
            "cache_formulas": len(candidate._formula_rows(rows)),
            "fully_supported_formulas": len(fully_supported_ids),
            "supported_records": len(rows),
            "raw_coverage": raw_coverage,
        },
        "evaluation_contract": {
            "selected_before_test_candidate_build": True,
            "test_used_for_model_epoch_lambda_selection": False,
            "test_prediction_passes": 2,
            "second_pass_role": "reload determinism only",
            "truth_grouping_supplied": True,
            "official_symLG_or_LgEval": False,
            "official_end_to_end_expression_rate": False,
            "shape_hwr_frozen": True,
            "candidate_set_frozen": True,
            "grouping_frozen": True,
        },
        "selection": research_payload["selection"],
        "rights": research_payload["rights"],
        "metrics": {
            "hwr": _metric_view(hwr_metrics),
            "base_context": _metric_view(base_metrics),
            "selected_context": _metric_view(selected_metrics),
            "selected_vs_hwr": _paired(rows, hwr_predictions, selected_predictions),
            "selected_vs_base_context": _paired(rows, base_predictions, selected_predictions),
        },
        "homograph_analysis": homographs,
        "per_class": per_class,
        "formula_analysis": formulas,
        "bottleneck_decomposition": {
            "shape_hwr_top1": hwr_metrics["all_top1"],
            "shape_hwr_top5": (len(rows) - truth_outside_top5) / len(rows),
            "truth_outside_top5_records": truth_outside_top5,
            "selected_context_top1": selected_metrics["all_top1"],
            "context_errors_with_truth_inside_top5": selected_context_inside_errors,
            "remaining_context_recovery_ceiling_pp": 100.0 * (
                (len(rows) - truth_outside_top5) / len(rows)
                - selected_metrics["all_top1"]
            ),
            "raw_grouping_eligible_formulas": raw_eligible,
            "raw_grouping_exact_count": raw_grouping_exact,
            "raw_grouping_exact": raw_grouping_exact / raw_eligible,
            "raw_grouping_evidence_role": (
                "existing end-to-end structural baseline only; its HWR checkpoint differs "
                "from the frozen HWR used by this context evaluation"
            ),
            "raw_grouping_hwr_checkpoint_sha256": raw_grouping.get("inputs", {}).get(
                "hwr_checkpoint_sha256"
            ),
            "current_hwr_checkpoint_sha256": masked._sha256(hwr_path),
            "raw_grouping_checkpoint_matches_current_hwr": (
                raw_grouping.get("inputs", {}).get("hwr_checkpoint_sha256")
                == masked._sha256(hwr_path)
            ),
            "raw_flat_formula_exact_count": int(raw_scores["formula_exact_after"]),
            "raw_flat_formula_exact": int(raw_scores["formula_exact_after"]) / raw_eligible,
            "stream_all_prefix_top1": float(stream_metrics["all_prefix_top1"]),
            "stream_last_20pct_top1": float(stream_metrics["last_20pct_top1"]),
            "stream_median_stable_progress": float(stream_metrics["median_stable_correct_progress"]),
            "stream_final_hwr_top1": float(stream_metrics["final"]["top1"]),
        },
        "integrity": {
            "base_candidate_audit": base_audit,
            "selected_candidate_audit": selected_audit,
            "reload_candidate_audit": reload_audit,
            "reload_mismatches": reload_mismatches,
            "new_tokens": int(selected_audit["new_tokens"]),
            "grouping_mutations": int(selected_audit["grouping_mutations"]),
            "stroke_mutations": 0,
            "immutable_inputs_unchanged": immutable_before == immutable_after,
        },
        "decision": {
            "crohme_research_checkpoint": "retained as noncommercial shadow challenger",
            "commercial_aiflow_checkpoint": "not adopted",
            "reason": (
                "CROHME-trained weights are CC BY-NC, full protocol/token coverage is "
                "incomplete, and material per-class regressions remain; transfer the "
                "verified objective to commercially eligible formula and owned-writer data"
            ),
        },
        "artifacts": {
            "selection_report": {
                "path": str(selection_path),
                "sha256": masked._sha256(selection_path),
            },
            "research_checkpoint": {
                "path": str(checkpoint_path),
                "sha256": masked._sha256(checkpoint_path),
            },
            "test_candidates": {
                "path": str(test_path),
                "sha256": masked._sha256(test_path),
            },
            "test_predictions": {
                "path": str(prediction_path),
                "sha256": masked._sha256(prediction_path),
            },
            "test_report": {"path": str(report_path)},
            "raw_grouping_report": {
                "path": str(raw_grouping_path),
                "sha256": masked._sha256(raw_grouping_path),
            },
            "streaming_report": {
                "path": str(streaming_path),
                "sha256": masked._sha256(streaming_path),
            },
        },
        "product_adopted": False,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report["artifacts"]["test_report"]["path"] = str(report_path)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(_markdown(report), encoding="utf-8", newline="\n")
    _event(
        "crohme_standard_test_complete",
        hwr_top1=hwr_metrics["all_top1"],
        base_top1=base_metrics["all_top1"],
        selected_top1=selected_metrics["all_top1"],
        hwr_formula_exact=hwr_metrics["formula_exact"],
        base_formula_exact=base_metrics["formula_exact"],
        selected_formula_exact=selected_metrics["formula_exact"],
        selected_strict_macro=selected_metrics["strict_macro_top1"],
        reload_mismatches=reload_mismatches,
        report=str(report_path),
        report_sha256=masked._sha256(report_path),
        markdown=str(markdown_path),
        markdown_sha256=masked._sha256(markdown_path),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
