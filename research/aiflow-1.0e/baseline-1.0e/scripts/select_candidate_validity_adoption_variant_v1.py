#!/usr/bin/env python3
"""Freeze candidate-validity selection using nested writer-OOF evidence only."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATES = (
    ROOT / "artifacts" / "candidate_validity_context_20260820_r1_shadow",
    ROOT / "artifacts" / "candidate_validity_adoption_20260820_v1_cov1e6_shadow",
    ROOT / "artifacts" / "candidate_validity_adoption_20260820_v2_cov3e6_shadow",
    ROOT / "artifacts" / "candidate_validity_adoption_20260820_v3_cov1e6_replay5_shadow",
)
DEFAULT_ACCEPTANCE = Path(
    r"D:\AIFlow-Workspace\PrivateData\math-ink-data-collector\derived"
    r"\fresh-context-acceptance-20260820-r2\frozen_acceptance"
    r"\frozen_acceptance_manifest.json"
)
DEFAULT_OUTPUT = (
    ROOT / "artifacts" / "candidate_validity_adoption_selection_20260820_r1"
)
THRESHOLDS = {
    "all_top1": 0.8863049095607235,
    "formula_exact": 0.7052631578947368,
    "strict_macro_top1": 0.8915311653116532,
    "new_writer_all_top1": 0.8837209302325582,
    "new_writer_formula_exact": 0.7272727272727273,
    "new_writer_strict_macro_top1": 0.975,
    "new_writer_regressed_max": 0,
}


def _sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metrics(report: dict) -> tuple[dict, dict, dict]:
    # Deliberately do not access product_refit_and_crohme or comparison fields.
    accuracy = report["evaluation"]["candidate_validity_writer_loo"]["accuracy"]
    return accuracy["all"], accuracy["new_writer"], accuracy["candidate_audit"]


def select(args: argparse.Namespace) -> dict:
    output = args.output.expanduser().resolve()
    if output.drive.upper() != "D:":
        raise ValueError(f"selection output must remain on D:: {output}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen selection: {output}")
    acceptance_path = args.acceptance.expanduser().resolve()
    if acceptance_path.drive.upper() != "D:" or not acceptance_path.is_file():
        raise ValueError(f"missing frozen acceptance manifest: {acceptance_path}")
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    if (
        acceptance.get("schema") != "aiflow-fresh-context-acceptance-freeze/v1"
        or acceptance.get("training_performed") is not False
        or acceptance.get("model_predictions_opened_before_freeze") is not False
        or int(acceptance.get("acceptance", {}).get("formulae", 0)) < 50
        or int(acceptance.get("acceptance", {}).get("writers", 0)) < 2
    ):
        raise ValueError("fresh acceptance was not validly frozen before selection")

    candidates = []
    for directory in args.candidates:
        directory = directory.expanduser().resolve()
        report_path = directory / "candidate_validity_context_report.json"
        if directory.drive.upper() != "D:" or not report_path.is_file():
            raise ValueError(f"missing D: candidate report: {report_path}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        overall, new_writer, candidate_audit = _metrics(report)
        product = report["checkpoints"]["product"]["accuracy"]
        checkpoint_path = Path(product["path"]).resolve()
        integrity = report["integrity"]
        metrics = {
            "all_top1": float(overall["all_top1"]),
            "formula_exact": float(overall["formula_exact"]),
            "strict_macro_top1": float(overall["strict_macro_top1"]),
            "regressed": int(overall["regressed"]),
            "new_writer_all_top1": float(new_writer["all_top1"]),
            "new_writer_formula_exact": float(new_writer["formula_exact"]),
            "new_writer_strict_macro_top1": float(new_writer["strict_macro_top1"]),
            "new_writer_regressed": int(new_writer["regressed"]),
        }
        metric_gates = {
            "all_top1": metrics["all_top1"] >= THRESHOLDS["all_top1"],
            "formula_exact": metrics["formula_exact"] >= THRESHOLDS["formula_exact"],
            "strict_macro_top1": metrics["strict_macro_top1"] >= THRESHOLDS["strict_macro_top1"],
            "new_writer_all_top1": metrics["new_writer_all_top1"] >= THRESHOLDS["new_writer_all_top1"],
            "new_writer_formula_exact": metrics["new_writer_formula_exact"] >= THRESHOLDS["new_writer_formula_exact"],
            "new_writer_strict_macro_top1": metrics["new_writer_strict_macro_top1"] >= THRESHOLDS["new_writer_strict_macro_top1"],
            "new_writer_regressed": metrics["new_writer_regressed"] <= THRESHOLDS["new_writer_regressed_max"],
        }
        integrity_gates = {
            "shape_training_performed_false": integrity["shape_training_performed"] is False,
            "shape_gradient_updates_zero": int(integrity["shape_gradient_updates"]) == 0,
            "immutable_inputs_unchanged": integrity["immutable_inputs_unchanged"] is True,
            "candidate_new_tokens_zero": int(candidate_audit["new_tokens"]) == 0,
            "grouping_mutations_zero": int(candidate_audit["grouping_mutations"]) == 0,
            "direct_reload_mismatches_zero": int(product["reload_mismatches"]["direct"]) == 0,
            "checkpoint_hash_matches": checkpoint_path.is_file() and _sha(checkpoint_path) == product["sha256"],
        }
        eligible = all(metric_gates.values()) and all(integrity_gates.values())
        candidates.append({
            "name": directory.name,
            "report": {"path": str(report_path), "sha256": _sha(report_path)},
            "checkpoint": {"path": str(checkpoint_path), "sha256": product["sha256"]},
            "configuration": report["training"]["product_configuration"]["accuracy"],
            "metrics": metrics,
            "metric_gates": metric_gates,
            "integrity_gates": integrity_gates,
            "eligible": eligible,
        })
    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    if not eligible:
        raise ValueError("no candidate satisfies the frozen writer-OOF gate")
    selected = max(eligible, key=lambda candidate: (
        candidate["metrics"]["formula_exact"],
        candidate["metrics"]["new_writer_formula_exact"],
        candidate["metrics"]["all_top1"],
        candidate["metrics"]["strict_macro_top1"],
        -candidate["metrics"]["regressed"],
    ))
    payload = {
        "schema": "aiflow-candidate-validity-adoption-selection/v1",
        "selected_at": datetime.now(timezone.utc).isoformat(),
        "selection_contract": {
            "nested_writer_and_exact_token_sequence_oof_only": True,
            "crohme_metrics_loaded": False,
            "fresh_acceptance_predictions_loaded": False,
            "fresh_acceptance_frozen_before_selection": True,
            "shape_hwr_frozen": True,
            "stroke_grouping_frozen": True,
            "candidate_set_frozen": True,
        },
        "thresholds": THRESHOLDS,
        "fresh_acceptance_manifest": {
            "path": str(acceptance_path), "sha256": _sha(acceptance_path)
        },
        "candidates": candidates,
        "selected": {
            "name": selected["name"],
            "checkpoint": selected["checkpoint"],
            "configuration": selected["configuration"],
            "metrics": selected["metrics"],
        },
    }
    output.mkdir(parents=True)
    path = output / "selection.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, nargs="+", default=DEFAULT_CANDIDATES)
    parser.add_argument("--acceptance", type=Path, default=DEFAULT_ACCEPTANCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = select(args)
    print(json.dumps({
        "selected": payload["selected"],
        "eligible": [row["name"] for row in payload["candidates"] if row["eligible"]],
        "crohme_metrics_loaded": False,
        "fresh_acceptance_predictions_loaded": False,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
