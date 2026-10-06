#!/usr/bin/env python3
"""Analyze the one-time frozen CROHME HWR augmentation evaluation."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path


CORE_OPERATORS = frozenset({
    "+", "-", "=", "/", "<", ">", "|", r"\times", r"\div",
    r"\pm", r"\neq", r"\leq", r"\geq",
})
DELIMITERS = frozenset({"(", ")", "[", "]", r"\{", r"\}"})
FAMILIES = {
    "vertical_slash": ("1", "|", "/"),
    "circle": ("0", "O", "o"),
    "cross": ("x", r"\times"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _d_path(path: Path, kind: str, *, must_exist: bool = True) -> Path:
    resolved = path.resolve()
    if resolved.drive.upper() != "D:":
        raise ValueError(f"{kind} must remain on D:: {resolved}")
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"missing {kind}: {resolved}")
    return resolved


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def _rows(path: Path) -> dict[str, dict]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    result = {str(row["record_id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate candidate record IDs: {path}")
    return result


def _category(label: str) -> str:
    if len(label) == 1 and label.isascii() and label.isdigit():
        return "digit"
    if len(label) == 1 and label.isascii() and label.isalpha():
        return "latin"
    if label in CORE_OPERATORS:
        return "core_operator"
    if label in DELIMITERS:
        return "delimiter"
    return "other_math"


def _subset_metrics(rows: list[dict]) -> dict:
    top1 = [row["final_topk"][0] == row["label"] for row in rows]
    top5 = [row["label"] in row["final_topk"] for row in rows]
    return {
        "records": len(rows), "top1": sum(top1) / len(rows),
        "top5": sum(top5) / len(rows), "outside_top5": len(rows) - sum(top5),
    }


def _paired_analysis(baseline: dict[str, dict], candidate: dict[str, dict]) -> dict:
    if set(baseline) != set(candidate):
        raise ValueError("baseline/candidate record coverage differs")
    transitions = Counter()
    labels: dict[str, Counter] = defaultdict(Counter)
    categories: dict[str, Counter] = defaultdict(Counter)
    writers: dict[str, list[str]] = defaultdict(list)
    confusion = Counter()
    persistent_confusion = Counter()
    for record_id in sorted(baseline):
        old, new = baseline[record_id], candidate[record_id]
        for key in ("label", "formula_id", "writer_group"):
            if old.get(key) != new.get(key):
                raise ValueError(f"frozen truth contract changed at {record_id}: {key}")
        truth = str(new["label"])
        old_top1 = old["final_topk"][0] == truth
        new_top1 = new["final_topk"][0] == truth
        old_top5 = truth in old["final_topk"]
        new_top5 = truth in new["final_topk"]
        if not old_top1 and new_top1:
            transitions["top1_improved"] += 1
        elif old_top1 and not new_top1:
            transitions["top1_regressed"] += 1
        if not old_top5 and new_top5:
            transitions["top5_rescued"] += 1
        elif old_top5 and not new_top5:
            transitions["top5_lost"] += 1
        elif not old_top5 and not new_top5:
            transitions["top5_persistent_outside"] += 1
        if not new_top1:
            confusion[(truth, str(new["final_topk"][0]))] += 1
        if not old_top5 and not new_top5:
            persistent_confusion[(truth, str(new["final_topk"][0]))] += 1
        for group in (labels[truth], categories[_category(truth)]):
            group["records"] += 1
            group["baseline_top1"] += old_top1
            group["candidate_top1"] += new_top1
            group["baseline_top5"] += old_top5
            group["candidate_top5"] += new_top5
            group["rescued"] += not old_top5 and new_top5
            group["lost"] += old_top5 and not new_top5
        writers[str(new["writer_group"])].append(record_id)

    def finalize(values: dict[str, Counter]) -> dict[str, dict]:
        output = {}
        for key, row in values.items():
            count = row["records"]
            output[key] = {
                **dict(row),
                "baseline_top1_rate": row["baseline_top1"] / count,
                "candidate_top1_rate": row["candidate_top1"] / count,
                "baseline_top5_rate": row["baseline_top5"] / count,
                "candidate_top5_rate": row["candidate_top5"] / count,
                "baseline_outside": count - row["baseline_top5"],
                "candidate_outside": count - row["candidate_top5"],
            }
        return output

    label_metrics = finalize(labels)
    category_metrics = finalize(categories)
    family_metrics = {}
    for name, family in FAMILIES.items():
        old_rows = [row for row in baseline.values() if row["label"] in family]
        new_rows = [candidate[row["record_id"]] for row in old_rows]
        family_metrics[name] = {
            "baseline": _subset_metrics(old_rows), "candidate": _subset_metrics(new_rows),
        }
    writer_metrics = {}
    for writer, record_ids in writers.items():
        writer_metrics[writer] = {
            "baseline": _subset_metrics([baseline[key] for key in record_ids]),
            "candidate": _subset_metrics([candidate[key] for key in record_ids]),
        }
    return {
        "transitions": dict(transitions),
        "by_label": dict(sorted(label_metrics.items())),
        "by_category": dict(sorted(category_metrics.items())),
        "by_family": family_metrics,
        "by_writer": dict(sorted(writer_metrics.items())),
        "writer_macro": {
            "baseline_top1": sum(row["baseline"]["top1"] for row in writer_metrics.values()) / len(writer_metrics),
            "candidate_top1": sum(row["candidate"]["top1"] for row in writer_metrics.values()) / len(writer_metrics),
            "baseline_top5": sum(row["baseline"]["top5"] for row in writer_metrics.values()) / len(writer_metrics),
            "candidate_top5": sum(row["candidate"]["top5"] for row in writer_metrics.values()) / len(writer_metrics),
        },
        "top_confusions": [
            {"truth": pair[0], "prediction": pair[1], "records": count}
            for pair, count in confusion.most_common(30)
        ],
        "persistent_outside_confusions": [
            {"truth": pair[0], "prediction": pair[1], "records": count}
            for pair, count in persistent_confusion.most_common(30)
        ],
    }


def _markdown(report: dict) -> str:
    base = report["crohme"]["baseline"]
    new = report["crohme"]["candidate"]
    delta = report["crohme"]["delta"]
    transitions = report["crohme"]["paired"]["transitions"]
    fresh = report["fresh_acceptance"]
    lines = [
        "# AIFlow Math Ink 1.0 HWR 궤적 증강 결과",
        "",
        "## 결론",
        "",
        f"- 동결 HWR Top-1: **{base['all_top1']:.2%} → {new['all_top1']:.2%}** ({delta['top1']:+.2%}p)",
        f"- 동결 HWR Top-5: **{base['top5']:.2%} → {new['top5']:.2%}** ({delta['top5']:+.2%}p)",
        f"- Top-5 밖: **{base['outside_top5']} → {new['outside_top5']}건** ({delta['outside_top5']:+d}, {report['crohme']['outside_reduction_rate']:.2%} 감소)",
        f"- 기존 누락 구조/신규 누락: **{transitions.get('top5_rescued', 0)} / {transitions.get('top5_lost', 0)}건**",
        f"- Top-1 개선/회귀: **{transitions.get('top1_improved', 0)} / {transitions.get('top1_regressed', 0)}건**",
        "- CROHME는 동결 뒤 1회 평가에만 사용했으며 학습·variant·epoch·threshold 선택에는 사용하지 않았다.",
        "- 661건이 남아 있으므로 완전 해소는 아니다. 이 결과를 보고 재학습하지 않고 잔여 오류를 수집 우선순위로만 남긴다.",
        "",
        "## 상업/프로젝트 데이터 게이트",
        "",
        f"- writer-LOO Top-1: **{report['writer_loo']['baseline_top1']:.2%} → {report['writer_loo']['candidate_top1']:.2%}**",
        f"- writer-LOO Top-5: **{report['writer_loo']['baseline_top5']:.2%} → {report['writer_loo']['candidate_top5']:.2%}**",
        f"- 신규 2명 186글자 Top-1: **{fresh['baseline']['top1']:.2%} → {fresh['candidate']['top1']:.2%}**",
        f"- 신규 2명 186글자 Top-5: **{fresh['baseline']['top5']:.2%} → {fresh['candidate']['top5']:.2%}**",
        f"- 신규 데이터 개선/회귀: **{fresh['delta']['improved_top1']} / {fresh['delta']['regressed_top1']}건**",
        f"- 학습 경계 감사: **{report['training_boundary']['trainers']}개 trainer, 실패 {report['training_boundary']['failed']}개, 오염 선택식 {report['training_boundary']['selection_findings']}개**",
        "",
        "## CROHME 학계표준 truth-group 평가",
        "",
        "| 지표 | 기존 | 증강 후보 | 변화 |",
        "|---|---:|---:|---:|",
        f"| 문자 Top-1 | {base['all_top1']:.2%} | {new['all_top1']:.2%} | {delta['top1']:+.2%}p |",
        f"| 문자 Top-5 | {base['top5']:.2%} | {new['top5']:.2%} | {delta['top5']:+.2%}p |",
        f"| 문자 Top-5 밖 | {base['outside_top5']} | {new['outside_top5']} | {delta['outside_top5']:+d} |",
        f"| 문자 strict macro Top-1 | {base['strict_macro_top1']:.2%} | {new['strict_macro_top1']:.2%} | {delta['strict_macro_top1']:+.2%}p |",
        f"| 식 exact (truth grouping) | {base['formula_exact']:.2%} | {new['formula_exact']:.2%} | {delta['formula_exact']:+.2%}p |",
        f"| 식 Top-5 oracle | {base['formula_top5_oracle']:.2%} | {new['formula_top5_oracle']:.2%} | {delta['formula_top5_oracle']:+.2%}p |",
        "",
        "## Top-5 밖 문자 분포",
        "",
        "| 문자 | 기존 밖 | 증강 뒤 밖 | 구조 | 신규 누락 |",
        "|---|---:|---:|---:|---:|",
    ]
    labels = report["crohme"]["paired"]["by_label"]
    ordered = sorted(labels.items(), key=lambda item: (-item[1]["baseline_outside"], item[0]))[:25]
    for label, row in ordered:
        lines.append(
            f"| `{label}` | {row['baseline_outside']} | {row['candidate_outside']} | "
            f"{row['rescued']} | {row['lost']} |"
        )
    lines.extend([
        "",
        "## 동형 계열",
        "",
        "| 계열 | 표본 | Top-1 기존→증강 | Top-5 기존→증강 |",
        "|---|---:|---:|---:|",
    ])
    for family, row in report["crohme"]["paired"]["by_family"].items():
        lines.append(
            f"| `{family}` | {row['baseline']['records']} | "
            f"{row['baseline']['top1']:.2%}→{row['candidate']['top1']:.2%} | "
            f"{row['baseline']['top5']:.2%}→{row['candidate']['top5']:.2%} |"
        )
    lines.extend([
        "",
        "## 채택 경계",
        "",
        "- **채택:** 상업권리 HWR 증강 challenger 및 다음 문맥 후보 생성용 동결 checkpoint.",
        "- **미전환:** 현재 제품 runtime. 기존 문맥 checkpoint가 이전 HWR hash에 묶여 있으므로 그대로 교체하면 안 된다.",
        "- **불변:** 128×5 입력, 4-block Transformer, 단일 372-class head, grouping/context/layout, raw fallback.",
        "- **다음 조건:** 새 HWR Top-5로 문맥 후보를 프로젝트 소유 데이터에서 다시 생성하고, 문맥 비회귀 후 runtime pair hash를 함께 갱신한다.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-candidates", type=Path, required=True)
    parser.add_argument("--candidate-candidates", type=Path, required=True)
    parser.add_argument("--baseline-build", type=Path, required=True)
    parser.add_argument("--candidate-build", type=Path, required=True)
    parser.add_argument("--augmentation-report", type=Path, required=True)
    parser.add_argument("--fresh-acceptance", type=Path, required=True)
    parser.add_argument("--boundary-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    paths = [
        _d_path(args.baseline_candidates, "baseline candidates"),
        _d_path(args.candidate_candidates, "candidate candidates"),
        _d_path(args.baseline_build, "baseline build"),
        _d_path(args.candidate_build, "candidate build"),
        _d_path(args.augmentation_report, "augmentation report"),
        _d_path(args.fresh_acceptance, "fresh acceptance"),
        _d_path(args.boundary_audit, "training boundary audit"),
        _d_path(args.output, "output", must_exist=False),
        _d_path(args.markdown, "markdown", must_exist=False),
    ]
    if paths[7].exists() or paths[8].exists():
        parser.error("refusing to overwrite analysis output")
    baseline_rows, candidate_rows = _rows(paths[0]), _rows(paths[1])
    baseline_build, candidate_build = _json(paths[2]), _json(paths[3])
    training, fresh, boundary = _json(paths[4]), _json(paths[5]), _json(paths[6])
    if baseline_build["inputs"]["split_directory_sha256"] != candidate_build["inputs"]["split_directory_sha256"]:
        raise ValueError("CROHME split changed between baseline and candidate")
    if candidate_build.get("training_performed") or not candidate_build.get("shape_hwr_frozen"):
        raise ValueError("candidate CROHME build is not a frozen evaluation")
    if candidate_build["inputs"]["hwr_checkpoint_sha256"] != training["checkpoint"]["sha256"]:
        raise ValueError("candidate build/checkpoint hash mismatch")
    if not fresh.get("passed") or boundary.get("summary", {}).get("failed"):
        raise ValueError("pre-CROHME acceptance gates are not clean")

    paired = _paired_analysis(baseline_rows, candidate_rows)
    base = baseline_build["hwr_metrics"]
    new = candidate_build["hwr_metrics"]
    base_view = {
        **base, "outside_top5": int(base["truth_rank_counts"]["outside"]),
    }
    new_view = {
        **new, "outside_top5": int(new["truth_rank_counts"]["outside"]),
    }
    delta = {
        "top1": new["all_top1"] - base["all_top1"],
        "top5": new["top5"] - base["top5"],
        "outside_top5": new_view["outside_top5"] - base_view["outside_top5"],
        "strict_macro_top1": new["strict_macro_top1"] - base["strict_macro_top1"],
        "formula_exact": new["formula_exact"] - base["formula_exact"],
        "formula_top5_oracle": new["formula_top5_oracle"] - base["formula_top5_oracle"],
    }
    control = training["variants"]["control_head"]["writer_disjoint"]
    selected = training["variants"][training["selection"]["selected"]]["writer_disjoint"]
    report = {
        "schema": "aiflow-hwr-augmentation-crohme-analysis/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "training_after_crohme_evaluation": False,
        "crohme_used_for_training_or_selection": False,
        "architecture_unchanged": bool(training.get("architecture_unchanged")),
        "writer_loo": {
            "baseline_top1": control["top1"], "candidate_top1": selected["top1"],
            "baseline_top5": control["top5"], "candidate_top5": selected["top5"],
        },
        "fresh_acceptance": {
            "baseline": fresh["baseline"], "candidate": fresh["candidate"],
            "delta": fresh["delta"], "passed": fresh["passed"],
        },
        "training_boundary": {
            "trainers": boundary["summary"]["trainers"],
            "failed": boundary["summary"]["failed"],
            "selection_findings": len(boundary.get("selection_findings", [])),
        },
        "crohme": {
            "role": "one-time post-freeze research evaluation only",
            "baseline": base_view, "candidate": new_view, "delta": delta,
            "outside_reduction_rate": (base_view["outside_top5"] - new_view["outside_top5"]) / base_view["outside_top5"],
            "paired": paired,
            "coverage": candidate_build["coverage"],
        },
        "decision": {
            "augmentation_effective": delta["top1"] > 0 and delta["top5"] > 0,
            "outside_top5_reduced": new_view["outside_top5"] < base_view["outside_top5"],
            "outside_top5_fully_resolved": new_view["outside_top5"] == 0,
            "hwr_challenger_frozen": True,
            "hwr_challenger_adopted_for_context_rebuild": True,
            "product_runtime_switched": False,
            "reason_runtime_not_switched": "context checkpoint and runtime pair are bound to the previous HWR hash",
        },
        "inputs": {
            str(path): _sha256(path) for path in paths[:7]
        },
    }
    paths[7].parent.mkdir(parents=True, exist_ok=True)
    paths[8].parent.mkdir(parents=True, exist_ok=True)
    paths[7].write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    paths[8].write_text(_markdown(report), encoding="utf-8", newline="\n")
    print(json.dumps({
        "event": "hwr_augmentation_crohme_analysis_complete",
        "top1": new["all_top1"], "top5": new["top5"],
        "outside_top5": new_view["outside_top5"],
        "output": str(paths[7]), "markdown": str(paths[8]),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
