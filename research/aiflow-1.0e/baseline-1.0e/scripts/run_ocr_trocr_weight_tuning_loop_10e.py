#!/usr/bin/env python3
"""Run a bounded single-model weight-tuning loop for AIFlow 1.0e.

The loop varies only fine-tuning scope and hyperparameters.  It never adds a
second OCR model, never trains on a new HF dataset, and selects only existing
HWR Top-k candidates.  Each trial is audited for full-formula Exact and
candidate-contract violations before ranking.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
TRAIN = SCRIPTS / "train_ocr_trocr_weight_tuned_10e.py"
AUDIT = SCRIPTS / "audit_ocr_text_evidence_adapter_10e.py"
LOOP_OUTPUT = ROOT / "artifacts" / "ocr_trocr_weight_tuning_loop_20260830"
PREVIOUS_TRIAL = ROOT / "artifacts" / "ocr_trocr_weight_tuned_10e_20260830_r1"
HF_COMPAT = Path(r"C:\Users\user\Documents\Codex\2026-08-30\ai\work\hfcompat")

TRIALS = [
    {"name": "last1_lr1e-4_e5", "epochs": 5, "learning_rate": 1e-4, "unfreeze_blocks": 1},
    {"name": "last1_lr2e-4_e5", "epochs": 5, "learning_rate": 2e-4, "unfreeze_blocks": 1},
    {"name": "last1_lr5e-4_e2", "epochs": 2, "learning_rate": 5e-4, "unfreeze_blocks": 1},
    {"name": "last2_lr1e-4_e3", "epochs": 3, "learning_rate": 1e-4, "unfreeze_blocks": 2},
]


def _env() -> dict[str, str]:
    env = os.environ.copy()
    paths = [str(HF_COMPAT), str(SCRIPTS)]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def _audit(artifact: Path) -> dict:
    subprocess.run(
        [sys.executable, str(AUDIT), "--artifact", str(artifact)],
        cwd=ROOT, env=_env(), check=True,
    )
    return json.loads((artifact / "audit.json").read_text(encoding="utf-8"))


def _summary(name: str, artifact: Path, audit: dict, reused: bool = False) -> dict:
    exact = audit["aggregate_full_exact"]["adapter"]
    covered = audit["aggregate_reported"]["adapter_formula_exact_covered"]
    regressions = []
    fold_rates = []
    for fold in audit["writer_loo_full_exact"]:
        baseline = fold["baseline_full_formula_exact"]["rate"]
        tuned = fold["adapter_full_formula_exact"]["rate"]
        fold_rates.append(tuned)
        if tuned < baseline:
            regressions.append(fold["held_writer"])
    result = {
        "name": name,
        "artifact": str(artifact),
        "reused": reused,
        "full_exact": exact["rate"],
        "full_exact_formulas": exact["exact_formulas"],
        "total_formulas": exact["total_formulas"],
        "covered_formula_exact": covered,
        "candidate_recall": audit["aggregate_reported"]["candidate_recall"],
        "candidate_absent_records": audit["aggregate_full_exact"]["candidate_absent_records"],
        "candidate_contract_violations": audit["candidate_contract"]["violations"],
        "writer_fold_rates": fold_rates,
        "writer_fold_regressions": len(regressions),
        "writer_fold_regression_ids": regressions,
    }
    return result


def main() -> int:
    LOOP_OUTPUT.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    if PREVIOUS_TRIAL.exists():
        results.append(_summary("previous_last1_lr3e-4_e3", PREVIOUS_TRIAL, _audit(PREVIOUS_TRIAL), reused=True))
    for trial in TRIALS:
        artifact = LOOP_OUTPUT / trial["name"]
        if not artifact.exists():
            print(json.dumps({"event": "trial_start", **trial, "output": str(artifact)}, ensure_ascii=False), flush=True)
            subprocess.run([
                sys.executable, str(TRAIN),
                "--output", str(artifact),
                "--epochs", str(trial["epochs"]),
                "--learning-rate", str(trial["learning_rate"]),
                "--unfreeze-blocks", str(trial["unfreeze_blocks"]),
                "--device", "cuda",
                "--batch-size", "8",
            ], cwd=ROOT, env=_env(), check=True)
        results.append(_summary(trial["name"], artifact, _audit(artifact)))
    ranked = sorted(
        results,
        key=lambda item: (
            item["candidate_contract_violations"] == 0,
            item["full_exact"],
            item["writer_fold_regressions"] == 0,
            min(item["writer_fold_rates"]),
            item["covered_formula_exact"],
        ),
        reverse=True,
    )
    report = {
        "schema": "aiflow-1.0e-trocr-weight-tuning-loop/v1",
        "model": "Azu/trocr-handwritten-math",
        "model_revision": "fc8dc9829360d42b1d4bc2f2668c831a72c80379",
        "external_model_count": 1,
        "new_hf_dataset_training": False,
        "selection_rule": "full_formula_exact_then_no_writer_regression_then_min_writer_exact",
        "trials": ranked,
        "selected": ranked[0] if ranked else None,
        "status": "shadow_only",
    }
    output = LOOP_OUTPUT / "loop_report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"event": "loop_complete", "output": str(output), "selected": report["selected"], "trials": len(results)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
