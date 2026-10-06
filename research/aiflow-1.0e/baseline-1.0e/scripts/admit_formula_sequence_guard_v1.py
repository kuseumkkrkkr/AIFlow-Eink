#!/usr/bin/env python3
"""Bind direct selection and external no-regression evidence into a runtime config."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from evaluate_48hz_prefix_v1 import _sha256
from formula_sequence_guard_v1 import SCHEMA as GUARD_SCHEMA, validate_configuration


SELECTION_SCHEMA = "aiflow-formula-sequence-guard-config/v1"
EVALUATION_SCHEMA = "aiflow-formula-sequence-guard-evaluation/v1"
LAYOUT_EVALUATION_SCHEMA = "aiflow-formula-layout-evaluation/v1"
RUNTIME_SCHEMA = "aiflow-formula-sequence-guard-runtime-config/v1"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-config", type=Path, required=True)
    parser.add_argument("--selection-report", type=Path, required=True)
    parser.add_argument("--external-evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {
        name: Path(value).expanduser().resolve()
        for name, value in {
            "selection_config": args.selection_config,
            "selection_report": args.selection_report,
            "external_evaluation": args.external_evaluation,
            "output": args.output,
        }.items()
    }
    for name in ("selection_config", "selection_report", "external_evaluation"):
        if not paths[name].is_file():
            parser.error(f"missing evidence: {paths[name]}")
    if paths["output"].exists():
        parser.error(f"refusing to overwrite runtime config: {paths['output']}")

    selection = _load(paths["selection_config"])
    report = _load(paths["selection_report"])
    external = _load(paths["external_evaluation"])
    if (
        selection.get("schema") != SELECTION_SCHEMA
        or report.get("schema") != EVALUATION_SCHEMA
        or external.get("schema") != LAYOUT_EVALUATION_SCHEMA
        or selection.get("guard_schema") != GUARD_SCHEMA
        or selection.get("gate", {}).get("shadow_admitted") is not True
    ):
        raise ValueError("formula sequence selection evidence contract mismatch")
    configuration = validate_configuration(selection["configuration"])
    if configuration != report["selection"]["configuration"]:
        raise ValueError("selected configuration differs from selection report")
    if selection["direct_candidate_sha256"] != external["direct"]["candidates_sha256"]:
        raise ValueError("direct candidate evidence hash mismatch")
    selected_direct = report["evaluation"]["all"]["challenger"]
    external_direct = external["direct"]["metrics"]
    if not (
        math.isclose(
            float(selected_direct["all_top1"]),
            float(external_direct["character_accuracy"]["context_finalized"]),
            abs_tol=1e-12,
        )
        and math.isclose(
            float(selected_direct["formula_exact"]),
            float(external_direct["character_formula_exact"]["context_finalized"]),
            abs_tol=1e-12,
        )
    ):
        raise ValueError("direct runtime does not reproduce selected metrics")
    delta = external.get("crohme_noncommercial", {}).get("finalizer_delta") or {}
    external_pass = bool(
        int(delta.get("improved_glyphs", 0)) > 0
        and int(delta.get("regressed_glyphs", 0)) == 0
        and int(delta.get("regressed_formulas", 0)) == 0
    )
    if not external_pass:
        raise ValueError("external formula sequence no-regression gate failed")

    payload = {
        "schema": RUNTIME_SCHEMA,
        "guard_schema": GUARD_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "shadow_runtime_only",
        "configuration": configuration,
        "checkpoint_sha256": selection["checkpoint_sha256"],
        "hwr_checkpoint_sha256": selection["hwr_checkpoint_sha256"],
        "direct_candidate_sha256": selection["direct_candidate_sha256"],
        "gate": {
            "runtime_admitted": True,
            "direct_selection": selection["gate"],
            "external_no_regression": {
                "passed": True,
                "dataset": "CROHME noncommercial evaluation only",
                "changed_glyphs": int(delta["changed_glyphs"]),
                "improved_glyphs": int(delta["improved_glyphs"]),
                "regressed_glyphs": int(delta["regressed_glyphs"]),
                "improved_formulas": int(delta["improved_formulas"]),
                "regressed_formulas": int(delta["regressed_formulas"]),
                "interpretation": (
                    "used for conservative hardening; not an untouched commercial acceptance set"
                ),
            },
        },
        "evidence": {
            name: {"path": str(paths[name]), "sha256": _sha256(paths[name])}
            for name in ("selection_config", "selection_report", "external_evaluation")
        },
    }
    paths["output"].parent.mkdir(parents=True, exist_ok=True)
    paths["output"].write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    print(json.dumps({
        "output": str(paths["output"]),
        "sha256": _sha256(paths["output"]),
        "gate": payload["gate"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
