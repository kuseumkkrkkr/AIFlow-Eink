#!/usr/bin/env python3
"""Preserve the frozen runtime below a preselected long-formula boundary."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import joblib


SCHEMA = "aiflow-conservative-grouping-finalization/v1"
MODEL_VERSION = "aiflow-stroke-grouping-1.0-r7-conservative-long-blend-shadow"
SELECTED_CONFIGURATION = (
    "all_features|mass=0.50|base_weight=0.25|bias=-1.0|route_strokes>=16"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ranker", type=Path, required=True)
    parser.add_argument("--blended-candidate", type=Path, required=True)
    parser.add_argument("--blended-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline_path = args.baseline_ranker.resolve()
    candidate_path = args.blended_candidate.resolve()
    summary_path = args.blended_summary.resolve()
    output = args.output.resolve()
    if any(path.drive.upper() != "D:" for path in (baseline_path, candidate_path, summary_path, output)):
        parser.error("all inputs and outputs must remain on D:")
    if output.exists() or any(not path.is_file() for path in (baseline_path, candidate_path, summary_path)):
        parser.error("inputs must exist and output must be new")
    baseline = joblib.load(baseline_path)
    candidate = joblib.load(candidate_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for key in ("schema", "feature_names", "lattice_config", "top_n"):
        if baseline.get(key) != candidate.get(key):
            raise ValueError(f"baseline/candidate contract differs: {key}")
    if (
        baseline.get("hwr_checkpoint_sha256") != candidate.get("hwr_checkpoint_sha256")
        or baseline.get("context_checkpoint_sha256") != candidate.get("context_checkpoint_sha256")
    ):
        raise ValueError("baseline/candidate HWR or context contract differs")
    trials = summary["writer_disjoint_development_audit"]["trials"]
    selected = next(
        (row for row in trials if row["configuration"] == SELECTED_CONFIGURATION),
        None,
    )
    if selected is None:
        raise ValueError("preselected 16-stroke development trial is missing")
    baseline_metrics = summary["writer_disjoint_development_audit"]["baseline"]
    if (
        selected["real"]["partition_exact"]
        < baseline_metrics["real"]["partition_exact"]
        or selected["real"]["pair_f1"]
        < baseline_metrics["real"]["pair_f1"] - 0.002
    ):
        raise ValueError("preselected long-formula trial violates owned nonregression")
    payload = dict(candidate)
    payload.update({
        "model_version": MODEL_VERSION,
        "grouping_model": baseline["grouping_model"],
        "group_bias": float(baseline.get("group_bias", 0.0)),
        "grouping_long_formula_route": {
            "minimum_strokes": 16,
            "group_bias": -1.0,
            "baseline_probability_weight": 0.25,
            "target_label_or_glyph_count_input": False,
        },
        "grouping_short_formula_preservation": {
            "policy": "byte-source baseline grouping model below 16 input strokes",
            "baseline_ranker_sha256": _sha256(baseline_path),
            "minimum_long_formula_strokes": 16,
        },
    })
    output.mkdir(parents=True)
    artifact_path = output / "partition_context_ranker.joblib"
    joblib.dump(payload, artifact_path, compress=3)
    report = {
        "schema": SCHEMA, "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "shadow_frozen_candidate", "model_version": MODEL_VERSION,
        "training_performed": False, "selection_performed": False,
        "selected_before_untouched_acceptance": True,
        "configuration": {
            "short_formula": "unchanged frozen baseline below 16 input strokes",
            "long_formula": SELECTED_CONFIGURATION,
        },
        "development_evidence": {
            "baseline": baseline_metrics,
            "selected_16_stroke_trial": selected,
        },
        "artifact": {
            "file": artifact_path.name, "sha256": _sha256(artifact_path),
            "baseline_ranker_sha256": _sha256(baseline_path),
            "blended_candidate_sha256": _sha256(candidate_path),
            "blended_summary_sha256": _sha256(summary_path),
        },
        "contracts": {
            "all_strokes_exactly_once": True,
            "target_label_writer_or_glyph_count_input": False,
            "hwr_or_context_checkpoint_modified": False,
            "product_default_enabled": False,
            "fresh_acceptance_rows_used_for_selection": 0,
        },
    }
    report_path = output / "conservative_finalization_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    print(json.dumps({
        "output": str(output), "artifact_sha256": report["artifact"]["sha256"],
        "configuration": report["configuration"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
