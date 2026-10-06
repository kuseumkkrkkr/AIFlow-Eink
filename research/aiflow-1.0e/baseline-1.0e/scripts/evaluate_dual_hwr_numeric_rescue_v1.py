#!/usr/bin/env python3
"""Evaluate the dual-HWR numeric operand gate on direct and CROHME scopes."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

from character_tensor_v1 import _json_lines
from dual_hwr_numeric_rescue_v1 import (
    CONFIG_SCHEMA, DEFAULT_CONFIGURATION, SCHEMA as RESCUE_SCHEMA,
    _self_test as rescue_self_test, apply_dual_hwr_numeric_rescue,
)
from evaluate_48hz_prefix_v1 import _sha256


SCHEMA = "aiflow-dual-hwr-numeric-rescue-evaluation/v1"


def _decorate_auxiliary(rows: list[dict]) -> list[dict]:
    return [{
        **row,
        "hwr_policy": DEFAULT_CONFIGURATION["auxiliary_policy"],
        "hwr_fusion_weight": DEFAULT_CONFIGURATION["auxiliary_weight"],
    } for row in rows]


def _scope(truth_rows: list[dict], baseline_rows: list[dict], merged_rows: list[dict]) -> dict:
    truth = {str(row["record_id"]): str(row["label"]) for row in truth_rows}
    baseline = {str(row["record_id"]): str(row["finalized_top1"]) for row in baseline_rows}
    merged = {str(row["record_id"]): str(row["finalized_top1"]) for row in merged_rows}
    formula_rows: dict[str, list[dict]] = defaultdict(list)
    for row in baseline_rows:
        formula_rows[str(row["formula_id"])].append(row)
    formulae = {
        formula_id: [str(row["record_id"]) for row in sorted(
            rows, key=lambda item: int(item["context_index"]),
        )]
        for formula_id, rows in formula_rows.items()
    }
    if set(truth) != set(baseline) or set(truth) != set(merged):
        raise ValueError("dual HWR evaluation coverage mismatch")
    character_before = sum(baseline[key] == truth[key] for key in truth)
    character_after = sum(merged[key] == truth[key] for key in truth)
    glyph_improved = sum(baseline[key] != truth[key] and merged[key] == truth[key] for key in truth)
    glyph_regressed = sum(baseline[key] == truth[key] and merged[key] != truth[key] for key in truth)
    formula_before = formula_after = improved = regressed = 0
    outcomes = []
    for formula_id, record_ids in formulae.items():
        before_ok = all(baseline[key] == truth[key] for key in record_ids)
        after_ok = all(merged[key] == truth[key] for key in record_ids)
        formula_before += before_ok
        formula_after += after_ok
        improved += not before_ok and after_ok
        regressed += before_ok and not after_ok
        if before_ok != after_ok:
            outcomes.append({
                "formula_id": formula_id,
                "direction": "improved" if after_ok else "regressed",
                "truth": [truth[key] for key in record_ids],
                "before": [baseline[key] for key in record_ids],
                "after": [merged[key] for key in record_ids],
            })
    return {
        "records": len(truth),
        "formulas": len(formulae),
        "baseline_character_top1_count": character_before,
        "challenger_character_top1_count": character_after,
        "challenger_character_top1": character_after / len(truth),
        "glyph_improved": glyph_improved,
        "glyph_regressed": glyph_regressed,
        "baseline_formula_exact_count": formula_before,
        "challenger_formula_exact_count": formula_after,
        "challenger_formula_exact": formula_after / len(formulae),
        "formula_improved": improved,
        "formula_regressed": regressed,
        "formula_outcomes": outcomes,
    }


def _evaluate_scope(
    truth_path: Path, baseline_path: Path, auxiliary_candidate_path: Path,
    auxiliary_finalized_path: Path, *, decorate_policy: bool,
) -> tuple[dict, dict]:
    truth_rows = list(_json_lines(truth_path))
    baseline_rows = list(_json_lines(baseline_path))
    auxiliary_candidates = list(_json_lines(auxiliary_candidate_path))
    if decorate_policy:
        auxiliary_candidates = _decorate_auxiliary(auxiliary_candidates)
    merged, audit = apply_dual_hwr_numeric_rescue(
        baseline_rows, auxiliary_candidates,
        list(_json_lines(auxiliary_finalized_path)), DEFAULT_CONFIGURATION,
    )
    return _scope(truth_rows, baseline_rows, merged), audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for prefix in ("direct", "crohme"):
        parser.add_argument(f"--{prefix}-truth", type=Path)
        parser.add_argument(f"--{prefix}-baseline", type=Path)
        parser.add_argument(f"--{prefix}-auxiliary-candidates", type=Path)
        parser.add_argument(f"--{prefix}-auxiliary-finalized", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config-output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        rescue_self_test()
        print('{"self_test":"pass"}')
        return 0
    values = [
        args.direct_truth, args.direct_baseline, args.direct_auxiliary_candidates,
        args.direct_auxiliary_finalized, args.crohme_truth, args.crohme_baseline,
        args.crohme_auxiliary_candidates, args.crohme_auxiliary_finalized,
        args.output, args.config_output,
    ]
    if any(value is None for value in values):
        parser.error("all direct, CROHME, and output paths are required")
    paths = [Path(value).expanduser().resolve() for value in values]
    if not all(path.is_file() for path in paths[:8]):
        parser.error("dual HWR evaluation input is missing")
    output, config_output = paths[8:]
    if output.exists() or config_output.exists():
        parser.error("refusing to overwrite dual HWR evaluation evidence")
    direct, direct_audit = _evaluate_scope(*paths[:4], decorate_policy=False)
    crohme, crohme_audit = _evaluate_scope(*paths[4:8], decorate_policy=True)
    runtime_admitted = bool(
        direct["formula_improved"] > 0
        and direct["formula_regressed"] == direct["glyph_regressed"] == 0
        and crohme["formula_improved"] > 0
        and crohme["formula_regressed"] == crohme["glyph_regressed"] == 0
        and direct_audit["tokens_outside_dual_hwr_union"] == 0
        and crohme_audit["tokens_outside_dual_hwr_union"] == 0
    )
    gate = {
        "runtime_admitted": runtime_admitted,
        "status": "shadow_runtime_only",
        "criteria": {
            "direct_formula_improvement": direct["formula_improved"] > 0,
            "direct_regressions_zero": direct["formula_regressed"] == direct["glyph_regressed"] == 0,
            "crohme_formula_improvement": crohme["formula_improved"] > 0,
            "crohme_regressions_zero": crohme["formula_regressed"] == crohme["glyph_regressed"] == 0,
            "dual_hwr_union_preserved": (
                direct_audit["tokens_outside_dual_hwr_union"]
                == crohme_audit["tokens_outside_dual_hwr_union"] == 0
            ),
        },
        "limits": (
            "direct thresholds and heads use repeatedly observed project-owned formulas; "
            "CROHME is noncommercial repeated diagnostic evidence; untouched commercial "
            "formula acceptance remains required"
        ),
    }
    generated_at = datetime.now(timezone.utc).isoformat()
    report = {
        "schema": SCHEMA,
        "generated_at": generated_at,
        "training_performed": False,
        "configuration": DEFAULT_CONFIGURATION,
        "gate": gate,
        "direct_project_owned_writer_loo": {"evaluation": direct, "audit": direct_audit},
        "crohme_noncommercial_diagnostic": {
            "evaluation": crohme, "audit": crohme_audit, "product_validation": False,
        },
        "inputs": {
            name: _sha256(path)
            for name, path in zip((
                "direct_truth", "direct_baseline", "direct_auxiliary_candidates",
                "direct_auxiliary_finalized", "crohme_truth", "crohme_baseline",
                "crohme_auxiliary_candidates", "crohme_auxiliary_finalized",
            ), paths[:8], strict=True)
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    config = {
        "schema": CONFIG_SCHEMA,
        "rescue_schema": RESCUE_SCHEMA,
        "generated_at": generated_at,
        "status": "shadow_runtime_only",
        "product_default_admitted": False,
        "configuration": DEFAULT_CONFIGURATION,
        "gate": gate,
        "evidence": {"path": str(output), "sha256": _sha256(output)},
    }
    config_output.parent.mkdir(parents=True, exist_ok=True)
    config_output.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({
        "report": str(output), "report_sha256": _sha256(output),
        "config": str(config_output), "config_sha256": _sha256(config_output),
        "gate": gate, "direct": direct, "crohme": crohme,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
