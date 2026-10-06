#!/usr/bin/env python3
"""Evaluate the unmatched-closer numeric operand rescue on both scopes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from character_tensor_v1 import _json_lines
from evaluate_48hz_prefix_v1 import _sha256
from evaluate_wide_candidate_syntax_rescue_v1 import _scope
from unmatched_fence_operand_rescue_v1 import (
    DEFAULT_CONFIGURATION,
    SCHEMA as RESCUE_SCHEMA,
    _self_test as rescue_self_test,
    apply_unmatched_fence_operand_rescue,
)
from wide_candidate_syntax_rescue_v1 import _write


SCHEMA = "aiflow-unmatched-fence-operand-rescue-evaluation/v1"
CONFIG_SCHEMA = "aiflow-unmatched-fence-operand-rescue-runtime-config/v1"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-baseline", type=Path)
    parser.add_argument("--direct-top20", type=Path)
    parser.add_argument("--crohme-baseline", type=Path)
    parser.add_argument("--crohme-top20", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config-output", type=Path)
    parser.add_argument("--direct-runtime-output", type=Path)
    parser.add_argument("--crohme-runtime-output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        rescue_self_test()
        print('{"self_test":"pass"}')
        return 0
    required = (
        args.direct_baseline, args.direct_top20,
        args.crohme_baseline, args.crohme_top20,
        args.output, args.config_output,
    )
    if any(value is None for value in required):
        parser.error("all baseline, Top-20, report, and config paths are required")
    paths = [value.expanduser().resolve() for value in required]
    if not all(path.is_file() for path in paths[:4]):
        parser.error("one or more unmatched-fence inputs are missing")
    report_path, config_path = paths[4:]
    requested_runtime_paths = [
        value.expanduser().resolve()
        for value in (args.direct_runtime_output, args.crohme_runtime_output)
        if value is not None
    ]
    if report_path.exists() or config_path.exists() or any(
        path.exists() for path in requested_runtime_paths
    ):
        parser.error("refusing to overwrite unmatched-fence evidence")
    direct_baseline = list(_json_lines(paths[0]))
    direct_top20 = list(_json_lines(paths[1]))
    crohme_baseline = list(_json_lines(paths[2]))
    crohme_top20 = list(_json_lines(paths[3]))
    direct_rows, direct_audit = apply_unmatched_fence_operand_rescue(
        direct_baseline, direct_top20, DEFAULT_CONFIGURATION,
    )
    crohme_rows, crohme_audit = apply_unmatched_fence_operand_rescue(
        crohme_baseline, crohme_top20, DEFAULT_CONFIGURATION,
    )
    direct = _scope(direct_top20, direct_baseline, direct_rows)
    crohme = _scope(crohme_top20, crohme_baseline, crohme_rows)
    invariant_keys = (
        "inserted_or_deleted_glyphs", "glyph_order_mutations",
        "grouping_mutations", "layout_mutations",
    )
    candidate_contract_passed = bool(
        direct_audit["candidate_preservation_rate"] == 1.0
        and crohme_audit["candidate_preservation_rate"] == 1.0
        and all(
            direct_audit[key] == crohme_audit[key] == 0
            for key in invariant_keys
        )
    )
    shadow_runtime_admitted = bool(
        direct["formula_improved"] > 0
        and direct["formula_regressed"] == direct["glyph_regressed"] == 0
        and crohme["formula_regressed"] == crohme["glyph_regressed"] == 0
        and candidate_contract_passed
    )
    generated_at = datetime.now(timezone.utc).isoformat()
    gate = {
        "shadow_runtime_admitted": shadow_runtime_admitted,
        "product_default_admitted": False,
        "criteria": {
            "direct_formula_improvement": direct["formula_improved"] > 0,
            "direct_regressions_zero": (
                direct["formula_regressed"] == direct["glyph_regressed"] == 0
            ),
            "crohme_repeated_diagnostic_regressions_zero": (
                crohme["formula_regressed"] == crohme["glyph_regressed"] == 0
            ),
            "candidate_and_layout_contract_passed": candidate_contract_passed,
        },
        "promotion_requirement": (
            "untouched project-owned writer-disjoint and formula-disjoint "
            "commercial acceptance evidence"
        ),
    }
    report = {
        "schema": SCHEMA,
        "generated_at": generated_at,
        "training_performed": False,
        "runtime_admitted": False,
        "configuration": DEFAULT_CONFIGURATION,
        "gate": gate,
        "direct_project_owned_development": {
            "evaluation": direct, "audit": direct_audit,
        },
        "crohme_repeated_noncommercial_diagnostic": {
            "evaluation": crohme, "audit": crohme_audit,
            "product_validation": False,
        },
        "inputs": {
            name: _sha256(path)
            for name, path in zip((
                "direct_baseline", "direct_top20",
                "crohme_baseline", "crohme_top20",
            ), paths[:4], strict=True)
        },
        "limits": [
            "the direct 95 formulas are repeatedly observed development evidence",
            "CROHME is repeated noncommercial diagnostic evidence",
            "the candidate contract expands to research-only Top-20 retrieval",
            "fresh commercial acceptance remains required before product default",
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    config = {
        "schema": CONFIG_SCHEMA,
        "rescue_schema": RESCUE_SCHEMA,
        "generated_at": generated_at,
        "status": "shadow_runtime_only",
        "runtime_admitted": False,
        "product_default_admitted": False,
        "configuration": DEFAULT_CONFIGURATION,
        "gate": gate,
        "evidence": {"path": str(report_path), "sha256": _sha256(report_path)},
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    runtime_outputs = {}
    if shadow_runtime_admitted:
        for name, path, rows in (
            ("direct", args.direct_runtime_output, direct_rows),
            ("crohme", args.crohme_runtime_output, crohme_rows),
        ):
            if path is None:
                continue
            runtime_path = path.expanduser().resolve()
            runtime_path.parent.mkdir(parents=True, exist_ok=True)
            _write(runtime_path, rows)
            runtime_outputs[name] = {
                "path": str(runtime_path), "sha256": _sha256(runtime_path),
            }
    print(json.dumps({
        "report": str(report_path), "report_sha256": _sha256(report_path),
        "config": str(config_path), "config_sha256": _sha256(config_path),
        "gate": gate, "direct": direct, "crohme": crohme,
        "runtime_outputs": runtime_outputs,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
