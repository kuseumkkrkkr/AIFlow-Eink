#!/usr/bin/env python3
"""Audit every training entrypoint for the CROHME evaluation-only boundary."""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
RETIRED = "train_crohme_standard_context_loop_v1.py"
CORPUS_BUILDERS = (
    "build_deepmind_formula_context_corpus_v1.py",
    "build_candidate_validity_coverage_corpus_v1.py",
)
SELECTION_TARGET_MARKERS = (
    "gate", "selected", "selection", "best", "winner", "promotion",
    "decision", "adopt",
)
TRAINING_CALL_MARKERS = (
    "fit", "train", "select", "split", "prompt_rows", "examples",
    "load_broad", "load_coverage",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _target_names(node: ast.AST) -> set[str]:
    return {item.id for item in ast.walk(node) if isinstance(item, ast.Name)}


def _value_names(node: ast.AST) -> set[str]:
    names = {item.id.casefold() for item in ast.walk(node) if isinstance(item, ast.Name)}
    names.update(
        item.attr.casefold() for item in ast.walk(node) if isinstance(item, ast.Attribute)
    )
    return names


def _call_name(call: ast.Call) -> str:
    if isinstance(call.func, ast.Name):
        return call.func.id.casefold()
    if isinstance(call.func, ast.Attribute):
        return call.func.attr.casefold()
    return ""


def _selection_findings(path: Path, tree: ast.AST) -> list[dict]:
    findings = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = set().union(*(_target_names(value) for value in node.targets))
            value = node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = _target_names(node.target), node.value
        else:
            continue
        if not any(
            marker in target.casefold()
            for target in targets for marker in SELECTION_TARGET_MARKERS
        ):
            continue
        names = _value_names(value)
        if any("crohme" in name for name in names):
            findings.append({
                "file": path.name,
                "line": int(getattr(node, "lineno", 0)),
                "kind": "crohme_tainted_selection_assignment",
                "targets": sorted(targets),
            })
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        call_name = _call_name(node)
        is_training_call = (
            call_name.startswith(("_fit", "fit_", "_train", "train_", "_select", "select_"))
            or call_name in {"_prompt_rows", "_training_data", "_direct_examples", "_broad_examples", "_coverage_examples"}
        )
        if not is_training_call:
            continue
        argument_names = set().union(*(_value_names(value) for value in [*node.args, *(item.value for item in node.keywords)]))
        if any("crohme" in name for name in argument_names):
            findings.append({
                "file": path.name,
                "line": int(getattr(node, "lineno", 0)),
                "kind": "crohme_tainted_training_or_selection_call",
                "call": call_name,
            })
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = set().union(*(_target_names(value) for value in node.targets))
            value = node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = _target_names(node.target), node.value
        else:
            continue
        if not any(
            marker in target.casefold()
            for target in targets
            for marker in ("training_exclusion", "evaluation_sequences", "fit_sequences")
        ):
            continue
        if any("crohme" in name for name in _value_names(value)):
            findings.append({
                "file": path.name,
                "line": int(getattr(node, "lineno", 0)),
                "kind": "crohme_tainted_fit_corpus_filter",
                "targets": sorted(targets),
            })
    return findings


def _dynamic_rejection(path: Path) -> dict:
    forbidden = r"D:\validation\crohme2019\train.jsonl.gz"
    result = subprocess.run(
        [sys.executable, "-s", str(path), "--input", forbidden],
        cwd=ROOT, capture_output=True, text=True, timeout=30, check=False,
    )
    combined = (result.stdout + "\n" + result.stderr).casefold()
    passed = result.returncode != 0 and (
        "validation-only" in combined or "retired" in combined
    )
    return {
        "passed": passed,
        "return_code": result.returncode,
        "message_has_boundary_reason": "validation-only" in combined or "retired" in combined,
    }


def _audit_trainers() -> tuple[list[dict], list[dict]]:
    rows, findings = [], []
    for path in sorted(SCRIPTS.glob("train_*.py")):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        retired = path.name == RETIRED
        selection_findings = [] if retired else _selection_findings(path, tree)
        dynamic = _dynamic_rejection(path)
        checks = {
            "retired_fail_closed": (
                "retired:" in source.casefold() and "raise SystemExit" in source
            ) if retired else None,
            "early_cli_guard": retired or "assert_training_entrypoint_arguments_clean()" in source,
            "path_guard": retired or "assert_training_path_clean" in source,
            "zero_crohme_manifest": retired or "zero_crohme_training_manifest" in source,
            "no_crohme_tainted_selection": not selection_findings,
            "dynamic_forbidden_path_rejected": dynamic["passed"],
        }
        row = {
            "file": path.name,
            "sha256": _sha256(path),
            "retired": retired,
            "checks": checks,
            "dynamic": dynamic,
            "passed": all(value is not False for value in checks.values()),
        }
        rows.append(row)
        findings.extend(selection_findings)
    return rows, findings


def _audit_builders() -> list[dict]:
    rows = []
    for name in CORPUS_BUILDERS:
        path = SCRIPTS / name
        source = path.read_text(encoding="utf-8")
        dynamic = _dynamic_rejection(path)
        checks = {
            "early_cli_guard": "assert_training_entrypoint_arguments_clean()" in source,
            "path_guard": "assert_training_path_clean" in source,
            "no_crohme_candidate_argument": "--crohme-candidates" not in source,
            "explicit_no_crohme_filter_contract": (
                '"crohme_used_for_generation_or_filtering": False' in source
            ),
            "dynamic_forbidden_path_rejected": dynamic["passed"],
        }
        rows.append({
            "file": name,
            "sha256": _sha256(path),
            "checks": checks,
            "dynamic": dynamic,
            "passed": all(checks.values()),
        })
    return rows


def _markdown(report: dict) -> str:
    lines = [
        "# CROHME 학습 경계 전수 감사",
        "",
        f"- 생성 시각(UTC): `{report['generated_at']}`",
        f"- 훈련 진입점: **{report['summary']['trainers']}개**",
        f"- 코퍼스 준비 진입점: **{report['summary']['corpus_builders']}개**",
        f"- 실패: **{report['summary']['failed']}개**",
        f"- CROHME 오염 선택식: **{len(report['selection_findings'])}개**",
        "",
        "## 훈련 진입점",
        "",
        "| 파일 | 상태 | 비고 |",
        "|---|---:|---|",
    ]
    for row in report["trainers"]:
        failed = [key for key, value in row["checks"].items() if value is False]
        note = "격리됨" if row["retired"] else (", ".join(failed) if failed else "통과")
        lines.append(f"| `{row['file']}` | {'통과' if row['passed'] else '실패'} | {note} |")
    lines.extend(["", "## 코퍼스 준비", "", "| 파일 | 상태 | 비고 |", "|---|---:|---|"])
    for row in report["corpus_builders"]:
        failed = [key for key, value in row["checks"].items() if not value]
        lines.append(
            f"| `{row['file']}` | {'통과' if row['passed'] else '실패'} | "
            f"{', '.join(failed) if failed else 'CROHME 비참조 v2'} |"
        )
    lines.extend([
        "",
        "## 결론",
        "",
        "CROHME 경로는 일반 훈련 인수에서 실행 전에 거부된다. CROHME 점수나 후보 감사값은 모델, epoch, 임계값, 채택 게이트에 사용할 수 없고 별도 무경사 보고만 허용한다.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "artifacts" / "training_boundary_audit_20260822_r1",
    )
    parser.add_argument(
        "--report", type=Path,
        default=ROOT / "reports" / "TRAINING_BOUNDARY_AUDIT_20260822.md",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    report_path = args.report.resolve()
    if output.exists():
        parser.error(f"refusing to overwrite audit output: {output}")
    trainers, findings = _audit_trainers()
    builders = _audit_builders()
    failed = sum(not row["passed"] for row in [*trainers, *builders])
    report = {
        "schema": "aiflow-training-boundary-audit/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "policy": "CROHME and MathWriting are no-gradient validation-only",
        "summary": {
            "trainers": len(trainers),
            "corpus_builders": len(builders),
            "failed": failed,
        },
        "trainers": trainers,
        "corpus_builders": builders,
        "selection_findings": findings,
        "passed": failed == 0 and not findings,
    }
    output.mkdir(parents=True)
    json_path = output / "training_boundary_audit.json"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_markdown(report), encoding="utf-8", newline="\n")
    print(json.dumps({
        "passed": report["passed"], "failed": failed,
        "selection_findings": len(findings), "audit": str(json_path),
    }, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
