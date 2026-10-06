#!/usr/bin/env python3
"""Select a prompt-context checkpoint only if the mature runtime does not regress."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from character_tensor_v1 import ROOT
from evaluate_48hz_prefix_v1 import _sha256


SCHEMA = "aiflow-prompt-context-selection/v1"
DEFAULT_OUTPUT = ROOT / "artifacts" / "prompt_context_selection_20260820_r2_shadow" / "prompt_context_selection.json"
DEFAULT_RUNS = {
    2: (
        ROOT / "artifacts" / "prompt_context_bert_tiny_20260820_r1_shadow" / "prompt_context_report.json",
        ROOT / "artifacts" / "masked_context_runtime_probe_20260820_r2_prompt_epoch2_shadow" / "masked_context_runtime_probe.json",
    ),
    3: (
        ROOT / "artifacts" / "prompt_context_bert_tiny_20260820_r3_epoch3_shadow" / "prompt_context_report.json",
        ROOT / "artifacts" / "masked_context_runtime_probe_20260820_r4_prompt_epoch3_shadow" / "masked_context_runtime_probe.json",
    ),
    4: (
        ROOT / "artifacts" / "prompt_context_bert_tiny_20260820_r2_epoch4_shadow" / "prompt_context_report.json",
        ROOT / "artifacts" / "masked_context_runtime_probe_20260820_r3_prompt_epoch4_shadow" / "masked_context_runtime_probe.json",
    ),
}


def _load(path: Path, schema: str) -> tuple[Path, dict]:
    resolved = path.resolve()
    if resolved.drive.upper() != "D:" or not resolved.is_file():
        raise ValueError(f"selection input must be an existing D: file: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if payload.get("schema") != schema:
        raise ValueError(f"unexpected schema: {resolved}")
    return resolved, payload


def _commercial_gate(report: dict) -> bool:
    training = report["training"]
    direct = report["evaluation"]["direct_writer_loo"]
    baseline = direct["baseline"]
    candidate = direct["prompt_context"]
    return (
        training["prompt_validation_after"]["top1"] > training["prompt_validation_before"]["top1"]
        and all(
            candidate[key] >= baseline[key]
            for key in ("all_top1", "strict_micro_top1", "strict_macro_top1", "formula_exact")
        )
        and direct["candidate_audit"]["candidate_preservation_rate"] == 1.0
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.drive.upper() != "D:" or output.exists():
        parser.error(f"output must be a new D: file: {output}")
    runs = []
    for epochs, (report_source, probe_source) in sorted(DEFAULT_RUNS.items()):
        report_path, report = _load(report_source, "aiflow-prompt-context-reranker/v1")
        probe_path, probe = _load(probe_source, "aiflow-masked-context-runtime-probe/v1")
        if int(report["training"]["epochs"]) != epochs:
            raise ValueError(f"epoch/report mismatch: {report_path}")
        checkpoint_hash = str(report["provenance"]["checkpoint_sha256"])
        if str(probe["provenance"]["masked_context_checkpoint_sha256"]) != checkpoint_hash:
            raise ValueError(f"checkpoint/probe mismatch: epoch {epochs}")
        current = probe["evaluations"]["current_layout"]["score"]
        rescue = probe["evaluations"]["bert_gated_unresolved_rescue"]
        admissible = (
            _commercial_gate(report)
            and rescue["score"]["exact_count"] >= current["exact_count"]
            and not rescue["regressed_formulas"]
        )
        runs.append({
            "epochs": epochs,
            "admissible": admissible,
            "prompt_validation_top1": report["training"]["prompt_validation_after"]["top1"],
            "direct_formula_exact": report["evaluation"]["direct_writer_loo"]["prompt_context"]["formula_exact"],
            "crohme_diagnostic_formula_exact": report["evaluation"]["crohme_transfer"]["prompt_context"]["formula_exact"],
            "runtime_baseline_exact_count": current["exact_count"],
            "runtime_rescue_exact_count": rescue["score"]["exact_count"],
            "runtime_improved_formulas": rescue["improved_formulas"],
            "runtime_regressed_formulas": rescue["regressed_formulas"],
            "checkpoint": report["provenance"]["checkpoint"],
            "checkpoint_sha256": checkpoint_hash,
            "report": {"path": str(report_path), "sha256": _sha256(report_path)},
            "runtime_probe": {"path": str(probe_path), "sha256": _sha256(probe_path)},
        })
    admitted = [run for run in runs if run["admissible"]]
    if not admitted:
        raise ValueError("no prompt-context checkpoint passed the mature-runtime non-regression gate")
    selected = max(
        admitted,
        key=lambda run: (
            run["runtime_rescue_exact_count"],
            run["direct_formula_exact"],
            run["prompt_validation_top1"],
            -run["epochs"],
        ),
    )
    payload = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "shadow_only",
        "selection_policy": "commercial-only: mature runtime zero-regression, direct formula exact, owned-prompt validation, then fewer epochs; CROHME diagnostic excluded",
        "runs": runs,
        "selected": selected,
        "decision": (
            "selected_shadow_checkpoint_without_current159_exact_gain"
            if selected["runtime_rescue_exact_count"] == selected["runtime_baseline_exact_count"]
            else "selected_shadow_checkpoint_with_repeated-development_gain"
        ),
        "automatic_default_replacement": False,
        "promotion_requirement": "fresh commercial writer/formula-disjoint acceptance",
    }
    output.parent.mkdir(parents=True, exist_ok=False)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({
        "event": "prompt_context_selection_complete",
        "output": str(output),
        "selected_epochs": selected["epochs"],
        "runtime_exact": selected["runtime_rescue_exact_count"],
        "runtime_regressions": len(selected["runtime_regressed_formulas"]),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
