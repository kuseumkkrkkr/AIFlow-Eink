#!/usr/bin/env python3
"""Validate frozen commercial-rights models after full formula completion.

CROHME contributes truth-group and 2D-relation validation only.  This command
does not train, select epochs, tune thresholds, or score intermediate prefixes.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

import torch

from character_tensor_v1 import ROOT, _json_lines
from evaluate_formula_layout_v1 import _order_metrics, _relation_metrics
from evaluate_homograph_context_reranker_v1 import _metrics
from finalize_formula_context_v1 import OwnedFormulaContextFinalizer


SCHEMA = "aiflow-crohme-formula-complete-validation/v1"
DEFAULT_ROOT = (
    ROOT / "artifacts" / "commercial_formula_complete_crohme_20260821_r1"
)
DEFAULT_CANDIDATES = (
    ROOT / "artifacts" / "crohme_standard_context_20260820_r1_research"
    / "test_candidates.jsonl.gz"
)
DEFAULT_BUILD = DEFAULT_CANDIDATES.with_name("test_candidate_build_report.json")
DEFAULT_HWR = Path(
    r"D:\AIFlow-Workspace\PrivateData\math-ink-data-collector\derived"
    r"\hwr-head-calibration-20260819-r1\final_all_writers_steps250_lr1e-3"
    r"\project_symbol_head_checkpoint.pt"
)
DEFAULT_CONTEXT = (
    ROOT / "artifacts" / "owned_formula_context_expanded_hwr_20260820_r1_shadow"
    / "owned_formula_context_product.pt"
)
DEFAULT_CROHME = (
    ROOT / "datasets" / "30_noncommercial_evaluation" / "crohme2019"
    / "crohme2019" / "crohme2019" / "test"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _save_rows(path: Path, rows: list[dict]) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _metric_view(value: dict) -> dict:
    keys = (
        "all_records", "all_top1", "baseline_all_top1", "strict_records",
        "strict_micro_top1", "strict_macro_top1", "formula_exact",
        "baseline_formula_exact", "changed", "improved", "regressed",
    )
    return {key: value.get(key) for key in keys if key in value}


def _markdown(report: dict) -> str:
    hwr = report["metrics"]["hwr_top1"]
    final = report["metrics"]["formula_complete_context"]
    order = report["layout"]["tracegroup_order_proxy"]
    relation = report["layout"]["relation_graph"]
    return "\n".join([
        "# CROHME 전체 수식 완료 후 검증",
        "",
        "## 검증 경계",
        "",
        "- CROHME 학습·epoch 선택·임계값 선택·gradient update: `0`",
        "- 중간 prefix/스트리밍 정확도: 측정하지 않음",
        "- 입력 종료(`formula_complete`) 후 전체 수식 문맥과 2D 배치만 채점",
        "- truth character grouping을 사용하므로 공식 end-to-end Expression Rate는 아님",
        "",
        "## 문자·식 결과",
        "",
        "| 단계 | 문자 Top-1 | 식 exact proxy |",
        "|---|---:|---:|",
        f"| 고정 HWR | {hwr['all_top1']:.2%} | {hwr['formula_exact']:.2%} |",
        f"| 전체 수식 문맥 | {final['all_top1']:.2%} | {final['formula_exact']:.2%} |",
        "",
        "## 2D 배치",
        "",
        f"- traceGroup 순서 exact: `{order['layout_order_exact']:.2%}`",
        f"- 순서+문자 식 exact: `{order['layout_and_character_formula_exact']['context_finalized']:.2%}`",
        f"- relation formula exact: `{relation['formula_exact']:.2%}`",
        f"- relation+문자 식 exact: `{relation['relation_and_character_formula_exact']['context_finalized']:.2%}`",
        "",
        "## 해석",
        "",
        "- 실시간 prefix 수치가 낮았던 현상은 더 이상 제품 정확도 정의에 포함하지 않는다.",
        "- 문맥층은 Top-5 밖 749건을 복구할 수 없으므로 영문 보조 HWR와 형태 데이터 보강은 별도 필요하다.",
        "- raw-stroke 자동 grouping 성능은 이 truth-group 검증과 분리해 보고해야 한다.",
        "",
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--candidate-build", type=Path, default=DEFAULT_BUILD)
    parser.add_argument("--hwr-checkpoint", type=Path, default=DEFAULT_HWR)
    parser.add_argument("--context-checkpoint", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--crohme-root", type=Path, default=DEFAULT_CROHME)
    parser.add_argument("--output", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    paths = [
        args.candidates.resolve(), args.candidate_build.resolve(),
        args.hwr_checkpoint.resolve(), args.context_checkpoint.resolve(),
        args.crohme_root.resolve(), args.output.resolve(),
    ]
    if any(path.drive.upper() != "D:" for path in paths):
        parser.error("all inputs and outputs must remain on D:")
    if args.output.exists() or args.batch_size < 1:
        parser.error("output must be new and batch size positive")
    if any(not path.exists() for path in paths[:-1]):
        parser.error("one or more frozen validation inputs are missing")
    torch.set_grad_enabled(False)
    if torch.is_grad_enabled():
        raise AssertionError("CROHME validation must run with autograd disabled")
    build = json.loads(args.candidate_build.read_text(encoding="utf-8"))
    if (
        build.get("training_performed") is not False
        or build.get("inputs", {}).get("hwr_checkpoint_sha256") != _sha256(args.hwr_checkpoint)
        or build.get("rights", {}).get("product_training_eligible") is not False
    ):
        raise ValueError("candidate cache is outside the frozen validation contract")
    hwr_payload = torch.load(args.hwr_checkpoint, map_location="cpu", weights_only=False)
    if hwr_payload.get("report", {}).get("data_policy", {}).get(
        "project_owned_formula_grouping_training"
    ) is not False:
        raise ValueError("HWR checkpoint training boundary is missing")
    context_payload = torch.load(args.context_checkpoint, map_location="cpu", weights_only=False)
    training_data = str(context_payload.get("training_data", ""))
    if "crohme" in training_data.casefold() or "project-owned" not in training_data:
        raise ValueError("context checkpoint training provenance is invalid")

    rows = list(_json_lines(args.candidates))
    baseline_predictions = {
        str(row["record_id"]): str(row["final_topk"][0]) for row in rows
    }
    finalizer = OwnedFormulaContextFinalizer(
        args.context_checkpoint, args.hwr_checkpoint,
        device=args.device, batch_size=args.batch_size,
        semantic_guards=True, equation_correction=False, formula_layout=True,
    )
    finalized, finalizer_audit = finalizer.finalize(rows)
    predictions = {
        str(row["record_id"]): str(row["finalized_top1"]) for row in finalized
    }
    if set(predictions) != set(baseline_predictions):
        raise AssertionError("formula-complete prediction coverage mismatch")
    baseline_metrics = _metrics(rows, baseline_predictions)
    final_metrics = _metrics(rows, predictions)
    order = _order_metrics(rows, predictions)
    relation = _relation_metrics(rows, args.crohme_root, None, predictions)
    if finalizer_audit.get("new_tokens") or finalizer_audit.get("grouping_mutations"):
        raise AssertionError("formula-complete context violated candidate contract")

    args.output.mkdir(parents=True)
    predictions_path = args.output / "formula_complete_predictions.jsonl.gz"
    _save_rows(predictions_path, finalized)
    report = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "noncommercial_validation_only",
        "training_performed": False,
        "selection_performed": False,
        "crohme_rows_used_for_training": 0,
        "crohme_gradient_updates": 0,
        "prefix_or_streaming_accuracy_scored": False,
        "decision_boundary": "formula_complete whole-formula result only",
        "truth_grouping_supplied": True,
        "official_end_to_end_expression_rate": False,
        "metrics": {
            "hwr_top1": _metric_view(baseline_metrics),
            "formula_complete_context": _metric_view(final_metrics),
            "delta": {
                "all_top1": final_metrics["all_top1"] - baseline_metrics["all_top1"],
                "formula_exact": final_metrics["formula_exact"] - baseline_metrics["formula_exact"],
            },
        },
        "layout": {
            "tracegroup_order_proxy": order,
            "relation_graph": relation,
        },
        "contracts": {
            "candidate_preservation_rate": finalizer_audit["candidate_preservation_rate"],
            "new_tokens": finalizer_audit["new_tokens"],
            "deleted_glyphs": finalizer_audit["deleted_glyphs"],
            "grouping_mutations": finalizer_audit["grouping_mutations"],
            "pre_complete_commits": 0,
            "autograd_enabled": False,
        },
        "finalizer_audit": finalizer_audit,
        "training_provenance": {
            "hwr": hwr_payload["report"]["data_policy"],
            "context_training_data": training_data,
            "context_selected_epochs_source": context_payload.get("selected_epochs_source"),
            "context_configuration_source": context_payload.get("configuration_source"),
            "crohme_used_for_model_or_threshold_selection": False,
        },
        "inputs": {
            "candidates": str(args.candidates.resolve()),
            "candidates_sha256": _sha256(args.candidates),
            "candidate_build_sha256": _sha256(args.candidate_build),
            "hwr_checkpoint_sha256": _sha256(args.hwr_checkpoint),
            "context_checkpoint_sha256": _sha256(args.context_checkpoint),
        },
        "output": {
            "predictions": str(predictions_path),
            "predictions_sha256": _sha256(predictions_path),
        },
    }
    report_path = args.output / "formula_complete_validation_report.json"
    markdown_path = ROOT / "reports" / "CROHME_FORMULA_COMPLETE_VALIDATION_20260822.md"
    if markdown_path.exists():
        raise FileExistsError(markdown_path)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({
        "report": str(report_path),
        "markdown": str(markdown_path),
        "hwr_top1": baseline_metrics["all_top1"],
        "complete_top1": final_metrics["all_top1"],
        "hwr_formula_exact": baseline_metrics["formula_exact"],
        "complete_formula_exact": final_metrics["formula_exact"],
        "crohme_gradient_updates": 0,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
