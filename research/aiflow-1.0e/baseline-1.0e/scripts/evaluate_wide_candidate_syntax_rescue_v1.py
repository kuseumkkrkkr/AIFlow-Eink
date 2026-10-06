#!/usr/bin/env python3
"""Evaluate the Top-10 syntax-placement rescue on direct and CROHME scopes."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

from character_tensor_v1 import _json_lines
from dual_hwr_numeric_rescue_v1 import (
    DEFAULT_CONFIGURATION as DUAL_CONFIGURATION,
    apply_dual_hwr_numeric_rescue,
)
from evaluate_48hz_prefix_v1 import _sha256
from evaluate_dual_hwr_numeric_rescue_v1 import _decorate_auxiliary
from wide_candidate_syntax_rescue_v1 import (
    CONFIG_SCHEMA, DEFAULT_CONFIGURATION, SCHEMA as RESCUE_SCHEMA,
    _self_test as rescue_self_test, _write, apply_wide_candidate_syntax_rescue,
)


SCHEMA = "aiflow-wide-candidate-syntax-rescue-evaluation/v1"


def _scope(truth_rows: list[dict], baseline_rows: list[dict], challenger_rows: list[dict]) -> dict:
    truth = {str(row["record_id"]): str(row["label"]) for row in truth_rows}
    baseline = {str(row["record_id"]): str(row["finalized_top1"]) for row in baseline_rows}
    challenger = {str(row["record_id"]): str(row["finalized_top1"]) for row in challenger_rows}
    if not truth or set(truth) != set(baseline) or set(truth) != set(challenger):
        raise ValueError("wide syntax evaluation coverage mismatch")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in baseline_rows:
        grouped[str(row["formula_id"])].append(row)
    formulae = {
        formula_id: [str(row["record_id"]) for row in sorted(
            rows, key=lambda value: int(value["context_index"]),
        )]
        for formula_id, rows in grouped.items()
    }
    character_before = sum(baseline[key] == truth[key] for key in truth)
    character_after = sum(challenger[key] == truth[key] for key in truth)
    glyph_outcomes = [
        {
            "record_id": key,
            "formula_id": next(
                str(row["formula_id"]) for row in baseline_rows
                if str(row["record_id"]) == key
            ),
            "direction": "improved" if challenger[key] == truth[key] else "regressed",
            "truth": truth[key], "before": baseline[key], "after": challenger[key],
        }
        for key in truth
        if baseline[key] != challenger[key]
        and (baseline[key] == truth[key] or challenger[key] == truth[key])
    ]
    outcomes = []
    formula_before = formula_after = improved = regressed = 0
    for formula_id, record_ids in formulae.items():
        before_ok = all(baseline[key] == truth[key] for key in record_ids)
        after_ok = all(challenger[key] == truth[key] for key in record_ids)
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
                "after": [challenger[key] for key in record_ids],
            })
    return {
        "records": len(truth), "formulas": len(formulae),
        "baseline_character_top1_count": character_before,
        "challenger_character_top1_count": character_after,
        "challenger_character_top1": character_after / len(truth),
        "glyph_improved": sum(
            baseline[key] != truth[key] and challenger[key] == truth[key] for key in truth
        ),
        "glyph_regressed": sum(
            baseline[key] == truth[key] and challenger[key] != truth[key] for key in truth
        ),
        "baseline_formula_exact_count": formula_before,
        "challenger_formula_exact_count": formula_after,
        "challenger_formula_exact": formula_after / len(formulae),
        "formula_improved": improved, "formula_regressed": regressed,
        "formula_outcomes": outcomes, "glyph_outcomes": glyph_outcomes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-baseline", type=Path)
    parser.add_argument("--direct-wide-candidates", type=Path)
    parser.add_argument("--crohme-pre-dual-baseline", type=Path)
    parser.add_argument("--crohme-auxiliary-top5", type=Path)
    parser.add_argument("--crohme-auxiliary-finalized", type=Path)
    parser.add_argument("--crohme-wide-candidates", type=Path)
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
    values = [
        args.direct_baseline, args.direct_wide_candidates,
        args.crohme_pre_dual_baseline, args.crohme_auxiliary_top5,
        args.crohme_auxiliary_finalized, args.crohme_wide_candidates,
        args.output, args.config_output,
    ]
    if any(value is None for value in values):
        parser.error("all direct, CROHME, and output paths are required")
    paths = [Path(value).expanduser().resolve() for value in values]
    if not all(path.is_file() for path in paths[:6]):
        parser.error("wide syntax evaluation input is missing")
    output_path, config_path = paths[6:]
    if output_path.exists() or config_path.exists():
        parser.error("refusing to overwrite wide syntax evaluation evidence")
    requested_runtime_paths = [
        value.expanduser().resolve() for value in (
            args.direct_runtime_output, args.crohme_runtime_output,
        ) if value is not None
    ]
    if any(path.exists() for path in requested_runtime_paths):
        parser.error("refusing to overwrite wide syntax runtime output")
    direct_truth = list(_json_lines(paths[1]))
    direct_baseline = list(_json_lines(paths[0]))
    direct_challenger, direct_audit = apply_wide_candidate_syntax_rescue(
        direct_baseline, direct_truth, DEFAULT_CONFIGURATION,
    )
    crohme_truth = list(_json_lines(paths[5]))
    crohme_pre_dual = list(_json_lines(paths[2]))
    crohme_dual, crohme_dual_audit = apply_dual_hwr_numeric_rescue(
        crohme_pre_dual,
        _decorate_auxiliary(list(_json_lines(paths[3]))),
        list(_json_lines(paths[4])),
        DUAL_CONFIGURATION,
    )
    crohme_challenger, crohme_audit = apply_wide_candidate_syntax_rescue(
        crohme_dual, crohme_truth, DEFAULT_CONFIGURATION,
    )
    direct = _scope(direct_truth, direct_baseline, direct_challenger)
    crohme = _scope(crohme_truth, crohme_dual, crohme_challenger)
    runtime_admitted = bool(
        direct["formula_improved"] > 0
        and direct["formula_regressed"] == direct["glyph_regressed"] == 0
        and crohme["formula_improved"] > 0
        and crohme["formula_regressed"] == crohme["glyph_regressed"] == 0
        and direct_audit["tokens_outside_baseline_and_wide_union"] == 0
        and crohme_audit["tokens_outside_baseline_and_wide_union"] == 0
    )
    gate = {
        "runtime_admitted": runtime_admitted,
        "status": "shadow_runtime_only",
        "criteria": {
            "direct_formula_improvement": direct["formula_improved"] > 0,
            "direct_regressions_zero": direct["formula_regressed"] == direct["glyph_regressed"] == 0,
            "crohme_formula_improvement": crohme["formula_improved"] > 0,
            "crohme_regressions_zero": crohme["formula_regressed"] == crohme["glyph_regressed"] == 0,
            "candidate_union_preserved": (
                direct_audit["tokens_outside_baseline_and_wide_union"]
                == crohme_audit["tokens_outside_baseline_and_wide_union"] == 0
            ),
        },
        "limits": (
            "Top-10 and thresholds were inspected on repeated project-owned development data; "
            "CROHME is noncommercial repeated diagnostic evidence; untouched commercial "
            "writer/formula acceptance remains required"
        ),
    }
    generated_at = datetime.now(timezone.utc).isoformat()
    report = {
        "schema": SCHEMA, "generated_at": generated_at,
        "training_performed": False, "configuration": DEFAULT_CONFIGURATION,
        "gate": gate,
        "direct_project_owned_writer_loo": {"evaluation": direct, "audit": direct_audit},
        "crohme_noncommercial_diagnostic": {
            "evaluation": crohme, "audit": crohme_audit,
            "preceding_dual_hwr_audit": crohme_dual_audit,
            "product_validation": False,
        },
        "inputs": {
            name: _sha256(path)
            for name, path in zip((
                "direct_baseline", "direct_wide_candidates",
                "crohme_pre_dual_baseline", "crohme_auxiliary_top5",
                "crohme_auxiliary_finalized", "crohme_wide_candidates",
            ), paths[:6], strict=True)
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    config = {
        "schema": CONFIG_SCHEMA, "rescue_schema": RESCUE_SCHEMA,
        "generated_at": generated_at, "status": "shadow_runtime_only",
        "product_default_admitted": False,
        "configuration": DEFAULT_CONFIGURATION, "gate": gate,
        "evidence": {"path": str(output_path), "sha256": _sha256(output_path)},
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    runtime_outputs = {}
    for name, path, rows in (
        ("direct", args.direct_runtime_output, direct_challenger),
        ("crohme", args.crohme_runtime_output, crohme_challenger),
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
        "report": str(output_path), "report_sha256": _sha256(output_path),
        "config": str(config_path), "config_sha256": _sha256(config_path),
        "gate": gate, "direct": direct, "crohme": crohme,
        "runtime_outputs": runtime_outputs,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
