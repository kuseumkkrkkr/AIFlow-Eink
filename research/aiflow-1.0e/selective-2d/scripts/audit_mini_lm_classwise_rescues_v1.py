#!/usr/bin/env python3
"""Measure where the frozen mini-LM shadow changes HWR Top-1 decisions."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import export_mini_formula_lm_onnx_v1 as export


ROOT = Path(__file__).resolve().parents[1]
DISTILL_DIR = ROOT / "artifacts" / "hwr_failure_cause_microscope_20260928" / "mini_lm_distill_20261002_r2"
MODEL_DIR = ROOT / "artifacts" / "hwr_failure_cause_microscope_20260928" / "mini_lm_onnx_20261002"
DEFAULT_OUTPUT = ROOT / "artifacts" / "hwr_failure_cause_microscope_20260928" / "mini_lm_onnx_20261002" / "classwise_rescues_20261002.json"


def main() -> int:
    import argparse

    import numpy as np
    import onnxruntime as ort
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DISTILL_DIR / "mini_formula_lm.pt")
    parser.add_argument("--distillation-report", type=Path, default=DISTILL_DIR / "mini_formula_lm_distillation_report.json")
    parser.add_argument("--onnx", type=Path, default=MODEL_DIR / "mini_formula_lm_fp32.onnx")
    parser.add_argument("--summary", type=Path, default=export.DEFAULT_SUMMARY)
    parser.add_argument("--data", type=Path, default=export.DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    checkpoint, report_path, onnx_path = args.checkpoint.resolve(), args.distillation_report.resolve(), args.onnx.resolve()
    summary_path, data_path, output = args.summary.resolve(), args.data.resolve(), args.output.resolve()
    if output.exists():
        parser.error(f"refusing to overwrite existing report: {output}")
    for name, path in (("student checkpoint", checkpoint), ("distillation report", report_path), ("FP32 ONNX", onnx_path), ("frozen summary", summary_path), ("formula data", data_path)):
        if not path.is_file():
            parser.error(f"missing {name}: {path}")

    distill = json.loads(report_path.read_text(encoding="utf-8"))
    protocol = distill["protocol"]
    if protocol.get("crohme_rows_loaded") != 0 or protocol.get("crohme_training_or_tuning") is not False:
        raise ValueError("mini-LM report lacks CROHME exclusion attestation")
    if export._sha256(checkpoint) != distill["student"]["checkpoint_sha256"]:
        raise ValueError("mini-LM checkpoint hash mismatch")
    if _sha256(onnx_path) != _sha256(MODEL_DIR / "mini_formula_lm_fp32.onnx"):
        raise ValueError("unexpected ONNX model file")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("crohme_training_or_tuning") is not False:
        raise ValueError("formula summary lacks CROHME exclusion attestation")
    if export._sha256(data_path) != summary["inputs"]["formulas_valid_sha256"]:
        raise ValueError("frozen formula dataset hash mismatch")
    raw = export._jsonl(data_path)
    if any("crohme" in str(row.get("source_partition", "")).casefold() for row in raw):
        raise ValueError("CROHME row found in diagnostic data")
    raw_by_id = {str(row["sample_id"]): row for row in raw}
    rows, targets = export._formula_rows(summary, raw_by_id)

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    labels = [str(label) for label in payload["labels"]]
    model = export.MiniFormulaLM(len(labels), len(payload["relations"]), int(payload["max_positions"]))
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    inputs, _ = export._current_inputs(rows, model, labels)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    logits = export._predict_ort(session, inputs)
    logits = logits.astype(np.float32, copy=False)
    log_prob = logits - np.logaddexp.reduce(logits, axis=1, keepdims=True)
    context = {
        str(row["record_id"]): values
        for row, values in zip(rows, log_prob, strict=True)
    }
    fusion_lambda = float(protocol["fusion_lambda_frozen_from_teacher"])
    raw_predictions = export._fused_predictions(rows, context, labels, fusion_lambda)
    student_predictions = export._strict_lock(rows, raw_predictions)
    fast_predictions = {str(row["record_id"]): str(row["final_topk"][0]) for row in rows}
    if any(student_predictions[str(row["record_id"])] not in row["final_topk"] for row in rows):
        raise AssertionError("student introduced a symbol outside HWR Top-5")

    by_formula: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_formula[str(row["formula_id"])].append(row)
    label_stats: dict[str, Counter] = defaultdict(Counter)
    formula_transitions = Counter()
    confusion_before: Counter[tuple[str, str]] = Counter()
    confusion_after: Counter[tuple[str, str]] = Counter()
    top5_missing_changes = []
    changed = improved = regressed = 0
    for formula_id, target in targets.items():
        sequence = sorted(by_formula[formula_id], key=lambda row: int(row["context"]["index"]))
        truths = [str(token) for token in target["tokens"]]
        before = [fast_predictions[str(row["record_id"])] for row in sequence]
        after = [student_predictions[str(row["record_id"])] for row in sequence]
        before_exact = bool(target["group_exact"] and before == truths)
        after_exact = bool(target["group_exact"] and after == truths)
        formula_transitions[("exact" if before_exact else "wrong", "exact" if after_exact else "wrong")] += 1
        if not target["group_exact"]:
            continue
        for row, truth, old, new in zip(sequence, truths, before, after, strict=True):
            stats = label_stats[truth]
            old_ok, new_ok = old == truth, new == truth
            stats["tokens"] += 1
            stats["fast_hits"] += int(old_ok)
            stats["student_hits"] += int(new_ok)
            stats["changed"] += int(old != new)
            stats["improved"] += int(not old_ok and new_ok)
            stats["regressed"] += int(old_ok and not new_ok)
            changed += int(old != new)
            improved += int(not old_ok and new_ok)
            regressed += int(old_ok and not new_ok)
            if not old_ok:
                confusion_before[(truth, old)] += 1
            if not new_ok:
                confusion_after[(truth, new)] += 1
            if truth not in row["final_topk"]:
                top5_missing_changes.append({"target": truth, "fast": old, "student": new, "top5": row["final_topk"]})

    class_rows = []
    for label, stats in label_stats.items():
        total = int(stats["tokens"])
        class_rows.append({
            "label": label,
            "tokens": total,
            "fast_hits": int(stats["fast_hits"]),
            "student_hits": int(stats["student_hits"]),
            "delta_hits": int(stats["student_hits"] - stats["fast_hits"]),
            "changed": int(stats["changed"]),
            "improved": int(stats["improved"]),
            "regressed": int(stats["regressed"]),
            "fast_top1_rate": float(stats["fast_hits"] / total),
            "student_top1_rate": float(stats["student_hits"] / total),
        })
    class_rows.sort(key=lambda row: (-row["delta_hits"], -row["changed"], row["label"]))
    formula_metrics_fast = export._formula_metrics(rows, targets, fast_predictions)
    formula_metrics_student = export._formula_metrics(rows, targets, student_predictions)
    report = {
        "schema": "aiflow-mini-lm-classwise-rescues/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "consumed_development_diagnostic_only",
        "protocol": {
            "training_performed": False,
            "model_or_threshold_selection": False,
            "crohme_rows_loaded": 0,
            "formula_set_consumed_development_data": True,
            "teacher_lambda_selected_on_current_overlap_cache": protocol.get("teacher_lambda_selected_on_current_overlap_cache"),
            "teacher_lambda_current_formula_overlap": protocol.get("teacher_lambda_source_current_formula_overlap"),
            "warning": "Classwise results describe the existing frozen shadow configuration only; they cannot support fresh performance claims or promote the model.",
        },
        "provenance": {
            "checkpoint_sha256": export._sha256(checkpoint),
            "onnx_sha256": _sha256(onnx_path),
            "distillation_report_sha256": export._sha256(report_path),
            "summary_sha256": export._sha256(summary_path),
            "formula_data_sha256": export._sha256(data_path),
        },
        "inference": {
            "formula_count": len(targets),
            "group_exact_formulas": formula_metrics_fast["group_exact_formulas"],
            "groups_changed": False,
            "hwr_candidates_preserved": True,
            "fusion_lambda_frozen": fusion_lambda,
            "formula_exact_fast": formula_metrics_fast["baseline_top1_formula_exact"],
            "formula_exact_student": formula_metrics_student["reranked_formula_exact"],
            "fast_top1_token_hits": formula_metrics_fast["baseline_top1_token_hits"],
            "student_top1_token_hits": formula_metrics_student["reranked_top1_token_hits"],
            "changed_tokens": changed,
            "improved_tokens": improved,
            "regressed_tokens": regressed,
            "formula_exact_transitions": {
                f"{before}_to_{after}": count for (before, after), count in sorted(formula_transitions.items())
            },
            "top5_missing_tokens_unchanged_or_not": top5_missing_changes,
            "top1_confusions_before": [
                {"target": target, "predicted": predicted, "count": int(count)}
                for (target, predicted), count in sorted(confusion_before.items(), key=lambda item: (-item[1], item[0]))[:30]
            ],
            "top1_confusions_after": [
                {"target": target, "predicted": predicted, "count": int(count)}
                for (target, predicted), count in sorted(confusion_after.items(), key=lambda item: (-item[1], item[0]))[:30]
            ],
            "by_target_class": class_rows,
        },
        "decision": {"automatic_default_replacement": False, "android_inference_verified": False},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({
        "event": "mini_lm_classwise_rescue_audit_complete",
        "report": str(output),
        "formula_exact_fast_student": [formula_metrics_fast["baseline_top1_formula_exact"], formula_metrics_student["reranked_formula_exact"]],
        "token_hits_fast_student": [formula_metrics_fast["baseline_top1_token_hits"], formula_metrics_student["reranked_top1_token_hits"]],
        "changed_improved_regressed": [changed, improved, regressed],
        "leading_student_corrections": report["inference"]["top1_confusions_before"][:5],
        "crohme_rows_loaded": 0,
    }, ensure_ascii=False))
    return 0


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
