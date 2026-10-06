#!/usr/bin/env python3
"""Trace CROHME validation Top-5 misses to commercial training support.

This command never trains or selects a model.  CROHME supplies validation
truth only; support counts come exclusively from approved commercial or
project-owned character corpora.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
from statistics import mean, median

from character_tensor_v1 import ROOT, _json_lines


SCHEMA = "aiflow-hwr-top5-outside-audit/v1"
SOURCES = ("hwrt", "uji", "isgl", "uci")
CORE_OPERATORS = frozenset({
    "+", "-", "=", "/", "<", ">", "|", r"\times", r"\div",
    r"\pm", r"\neq", r"\leq", r"\geq",
})
DELIMITERS = frozenset({"(", ")", "[", "]", r"\{", r"\}"})
COMMERCIAL_SINGLE_GLYPH_EXPANSION = ("t", ",", ".", "!")
SEMANTIC_FUNCTION_GROUPS = (r"\sin", r"\cos", r"\lim", r"\log", r"\tan")

DEFAULT_OUTSIDE = (
    ROOT / "artifacts" / "crohme_stream_rank_context_expanded_20260820_r2_shadow"
    / "crohme_test_truth_outside_top5.jsonl.gz"
)
DEFAULT_BUILD = (
    ROOT / "artifacts" / "crohme_standard_context_20260820_r1_research"
    / "test_candidate_build_report.json"
)
DEFAULT_CANONICAL = ROOT / "datasets" / "normalized" / "v1"
DEFAULT_OUTPUT = (
    ROOT / "artifacts" / "hwr_top5_outside_audit_20260821_r1"
    / "top5_outside_audit.json"
)
DEFAULT_MARKDOWN = ROOT / "reports" / "HWR_TOP5_OUTSIDE_AUDIT_20260821.md"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


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


def _support_band(count: int) -> str:
    if count == 0:
        return "zero"
    if count < 50:
        return "1-49"
    if count < 200:
        return "50-199"
    if count < 500:
        return "200-499"
    return "500+"


def _commercial_support(canonical_root: Path) -> tuple[dict[str, Counter], Counter]:
    holdout_path = canonical_root / "character_classifier_v1" / "current_external_holdout_ids.json"
    holdout = set(json.loads(holdout_path.read_text(encoding="utf-8")))
    source_counts: dict[str, Counter] = {}
    for source in SOURCES:
        counts: Counter = Counter()
        for row in _json_lines(canonical_root / f"{source}.jsonl.gz"):
            if f"{source}:{row['record_id']}" not in holdout:
                counts[str(row["label"])] += 1
        source_counts[source] = counts
    owned: Counter = Counter()
    for row in _json_lines(
        canonical_root / "character_classifier_v1"
        / "project_owned_ownership_eval.jsonl.gz"
    ):
        owned[str(row["label"])] += 1
    return source_counts, owned


def _rank_summary(values: list[int]) -> dict:
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "mean": mean(ordered),
        "median": median(ordered),
        "maximum": max(ordered),
        "rank_6_to_10": sum(value <= 10 for value in ordered),
        "rank_11_to_20": sum(10 < value <= 20 for value in ordered),
        "rank_21_plus": sum(value > 20 for value in ordered),
    }


def audit(outside_path: Path, build_path: Path, canonical_root: Path) -> dict:
    outside_rows = list(_json_lines(outside_path))
    build = _json(build_path)
    if (
        build.get("training_performed") is not False
        or build.get("rights", {}).get("product_training_eligible") is not False
        or int(build.get("hwr_metrics", {}).get("truth_rank_counts", {}).get("outside", -1))
        != len(outside_rows)
    ):
        raise ValueError("CROHME validation boundary or outside-row count mismatch")
    if len(outside_rows) != 749:
        raise ValueError(f"expected frozen 749 outside-Top-5 rows, got {len(outside_rows)}")

    source_counts, owned_counts = _commercial_support(canonical_root)
    by_label: dict[str, list[dict]] = defaultdict(list)
    categories = Counter()
    support_bands = Counter()
    zero_support_rows = 0
    for row in outside_rows:
        label = str(row["truth"])
        rank = int(row["truth_rank"])
        if rank <= 5:
            raise ValueError(f"outside row has rank <= 5: {row['record_id']}")
        by_label[label].append(row)
        categories[_category(label)] += 1
        support = sum(source_counts[source][label] for source in SOURCES)
        total = support + owned_counts[label]
        support_bands[_support_band(total)] += 1
        zero_support_rows += total == 0

    labels = []
    for label, rows in sorted(
        by_label.items(), key=lambda item: (-len(item[1]), item[0])
    ):
        external = {source: source_counts[source][label] for source in SOURCES}
        external_total = sum(external.values())
        project = owned_counts[label]
        total = external_total + project
        confusions = Counter(str(row["top1"]) for row in rows)
        labels.append({
            "truth": label,
            "category": _category(label),
            "outside_top5": len(rows),
            "truth_rank": _rank_summary([int(row["truth_rank"]) for row in rows]),
            "commercial_training_support": {
                "external_by_source": external,
                "external_total": external_total,
                "project_owned_pool": project,
                "total": total,
                "band": _support_band(total),
                "interpretation": (
                    "output row had no admitted training examples"
                    if total == 0 else
                    "external row absent; project-owned calibration only"
                    if external_total == 0 else
                    "admitted commercial training examples exist"
                ),
            },
            "top1_confusions": [
                {"prediction": token, "count": count}
                for token, count in confusions.most_common(8)
            ],
        })

    unsupported = {
        str(label): int(count)
        for label, count in build["coverage"]["unsupported_labels"].items()
    }
    single_expansion = {
        label: {
            "crohme_validation_groups": unsupported.get(label, 0),
            "commercial_training_support": {
                source: source_counts[source][label] for source in SOURCES
            },
        }
        for label in COMMERCIAL_SINGLE_GLYPH_EXPANSION
    }
    semantic_functions = {
        label: unsupported.get(label, 0) for label in SEMANTIC_FUNCTION_GROUPS
    }
    if sum(unsupported.values()) != 542:
        raise ValueError("frozen unsupported CROHME coverage changed")

    return {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "validation_only_diagnostic",
        "training_performed": False,
        "selection_performed": False,
        "crohme_gradient_updates": 0,
        "baseline": {
            "hwr_checkpoint_sha256": build["inputs"]["hwr_checkpoint_sha256"],
            "supported_records": build["counts"]["supported_records"],
            "top1": build["hwr_metrics"]["all_top1"],
            "top5": build["hwr_metrics"]["top5"],
            "outside_top5": len(outside_rows),
        },
        "outside_top5": {
            "records": len(outside_rows),
            "by_category": dict(sorted(categories.items())),
            "by_training_support_band": dict(sorted(support_bands.items())),
            "zero_commercial_support_rows": zero_support_rows,
            "labels": labels,
        },
        "unsupported_truth_groups": {
            "records": sum(unsupported.values()),
            "labels": unsupported,
            "commercial_single_glyph_expansion": single_expansion,
            "commercial_single_glyph_recoverable_groups": sum(
                unsupported.get(label, 0)
                for label in COMMERCIAL_SINGLE_GLYPH_EXPANSION
            ),
            "semantic_function_composition": semantic_functions,
            "semantic_function_groups": sum(semantic_functions.values()),
        },
        "conclusion": {
            "outside_top5_is_untrained_vocabulary": zero_support_rows == len(outside_rows),
            "outside_top5_zero_support_records": zero_support_rows,
            "main_cause": (
                "shape/domain generalization and class confusion inside the existing vocabulary; "
                "not wholesale absence of English training"
            ),
            "separate_missing_vocabulary": list(COMMERCIAL_SINGLE_GLYPH_EXPANSION),
            "formula_complete_policy": (
                "score final whole-formula decisions; prefix predictions are provisional and excluded"
            ),
        },
        "inputs": {
            "outside_rows": str(outside_path),
            "outside_rows_sha256": _sha256(outside_path),
            "candidate_build_report": str(build_path),
            "candidate_build_report_sha256": _sha256(build_path),
            "canonical_root": str(canonical_root),
        },
    }


def _markdown(report: dict) -> str:
    baseline = report["baseline"]
    outside = report["outside_top5"]
    unsupported = report["unsupported_truth_groups"]
    lines = [
        "# HWR Top-5 밖 749건 추적",
        "",
        "## 결론",
        "",
        "- 749건은 영문을 통째로 학습하지 않아 생긴 오류가 아니다.",
        f"- 상업·소유 학습 표본이 0개인 실패는 `{outside['zero_commercial_support_rows']}/749`건이다.",
        "- 주원인은 기존 372 어휘 안에서의 형태·도메인 일반화와 클래스 혼동이다.",
        "- 별도 미지원 어휘는 `t`, `,`, `.`, `!`이며 UJI/ISGL로 상업권리 학습이 가능하다.",
        "- `\\sin/\\cos/\\lim/\\log/\\tan`은 한 글자 shape class가 아니라 전체 수식 완료 뒤 문자열을 의미 토큰으로 합성해야 한다.",
        "",
        "## 동결 기준선",
        "",
        f"- 지원 문자: `{baseline['supported_records']:,}`",
        f"- Top-1 / Top-5: `{baseline['top1']:.2%}` / `{baseline['top5']:.2%}`",
        f"- 정답 Top-5 밖: `{baseline['outside_top5']:,}`",
        "- CROHME 학습·선택·gradient update: `0`",
        "",
        "## 실패 구성",
        "",
        "| 구분 | 건수 |",
        "|---|---:|",
    ]
    for category, count in outside["by_category"].items():
        lines.append(f"| {category} | {count} |")
    lines += [
        "",
        "## 상위 실패 문자와 실제 학습량",
        "",
        "| 문자 | Top-5 밖 | HWRT | UJI | ISGL | UCI | 소유 pool | rank 중앙값 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in outside["labels"][:25]:
        support = row["commercial_training_support"]
        source = support["external_by_source"]
        lines.append(
            f"| `{row['truth']}` | {row['outside_top5']} | {source['hwrt']} | "
            f"{source['uji']} | {source['isgl']} | {source['uci']} | "
            f"{support['project_owned_pool']} | {row['truth_rank']['median']} |"
        )
    lines += [
        "",
        "## 현재 출력 어휘 밖",
        "",
        f"- 전체 미지원 truth group: `{unsupported['records']}`",
        f"- 상업 데이터로 직접 확장 가능한 단일 문자: `{unsupported['commercial_single_glyph_recoverable_groups']}`",
        f"- 함수명 의미 합성 대상: `{unsupported['semantic_function_groups']}`",
        "",
        "## 판정 기준 변경",
        "",
        "- 중간 prefix 정확도는 제품 정확도에서 제외한다.",
        "- 획 입력 중에는 후보만 provisional로 유지한다.",
        "- `formula_complete` 뒤 global grouping → HWR Top-5 → 전체 수식 문맥 → 2D 배치 결과만 확정·채점한다.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outside", type=Path, default=DEFAULT_OUTSIDE)
    parser.add_argument("--candidate-build", type=Path, default=DEFAULT_BUILD)
    parser.add_argument("--canonical-root", type=Path, default=DEFAULT_CANONICAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    paths = [
        args.outside.resolve(), args.candidate_build.resolve(),
        args.canonical_root.resolve(), args.output.resolve(), args.markdown.resolve(),
    ]
    if any(path.drive.upper() != "D:" for path in paths):
        parser.error("all inputs and outputs must remain on D:")
    if args.output.exists() or args.markdown.exists():
        parser.error("refusing to overwrite audit output")
    report = audit(paths[0], paths[1], paths[2])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.markdown.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "markdown": str(args.markdown.resolve()),
        "outside_top5": report["outside_top5"]["records"],
        "zero_support": report["outside_top5"]["zero_commercial_support_rows"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
