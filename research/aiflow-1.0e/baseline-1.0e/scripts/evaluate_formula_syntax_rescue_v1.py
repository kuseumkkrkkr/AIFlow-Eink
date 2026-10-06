#!/usr/bin/env python3
"""Evaluate the post-context syntax rescue without using CROHME for selection."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

from character_tensor_v1 import _json_lines
from evaluate_48hz_prefix_v1 import DEFAULT_PRODUCT, _sha256
from evaluate_formula_sequence_guard_v1 import _scope
from finalize_formula_context_v1 import (
    DEFAULT_CONTEXT, OwnedFormulaContextFinalizer, _runtime_rows,
)
from formula_layout_v1 import recontextualize_formula_rows
from formula_syntax_rescue_v1 import (
    DEFAULT_CONFIGURATION, SCHEMA as RESCUE_SCHEMA,
    _horizontal_dash_evidence, _self_test as rescue_self_test,
    apply_formula_syntax_rescue, validate_configuration,
)
import train_masked_context_reranker_v1 as masked
from train_owned_formula_context_v1 import _components


SCHEMA = "aiflow-formula-syntax-rescue-evaluation/v2"
CONFIG_SCHEMA = "aiflow-formula-syntax-rescue-runtime-config/v2"


def _predictions(path: Path) -> dict[str, str]:
    return {
        str(row["record_id"]): str(row["finalized_top1"])
        for row in _json_lines(path)
    }


def _horizontal_dash_diagnostic(
    rows: list[dict], predictions: dict[str, str], configuration: dict,
) -> dict:
    configuration = validate_configuration(configuration)
    eligible = []
    outcomes = Counter()
    truth_distribution = Counter()
    for formula_id, sequence in masked._formulae(rows).items():
        for index, row in enumerate(sequence):
            if index == 0 or index + 1 == len(sequence):
                continue
            record_id = str(row["record_id"])
            candidates = [str(value) for value in row["final_topk"]]
            before = str(predictions[record_id])
            if before != "/" or "-" not in candidates:
                continue
            truth = str(row["label"])
            truth_distribution[truth] += 1
            probabilities = [float(value) for value in row["final_topk_probabilities"]]
            probability_ratio = probabilities[candidates.index("-")] / max(
                probabilities[candidates.index(before)], 1e-12,
            )
            evidence = _horizontal_dash_evidence(
                row, before, "-", configuration,
            )
            selected = bool(
                evidence is not None
                and probability_ratio >= configuration["minimum_candidate_probability_ratio"]
            )
            outcome = "not_selected"
            if selected:
                outcome = (
                    "improved" if truth == "-"
                    else "regressed" if truth == "/"
                    else "neutral"
                )
            outcomes[outcome] += 1
            eligible.append({
                "formula_id": str(formula_id), "record_id": record_id,
                "truth": truth, "selected": selected, "outcome": outcome,
                "candidate_probability_ratio": probability_ratio,
                "geometry": evidence,
            })
    return {
        "eligible": len(eligible),
        "selected": sum(bool(row["selected"]) for row in eligible),
        "truth_distribution": dict(sorted(truth_distribution.items())),
        "outcomes": dict(sorted(outcomes.items())),
        "rows": eligible,
    }


def _evaluate(
    candidate_path: Path, finalized_path: Path,
    finalizer: OwnedFormulaContextFinalizer,
) -> tuple[dict, dict]:
    rows = list(_json_lines(candidate_path))
    baseline = _predictions(finalized_path)
    if set(baseline) != {str(row["record_id"]) for row in rows}:
        raise ValueError("formula syntax rescue finalized coverage mismatch")
    runtime, layout_audit = recontextualize_formula_rows(
        _runtime_rows(rows, finalizer.labels, require_context=False)
    )
    truth = {str(row["record_id"]): str(row["label"]) for row in rows}
    evaluation_rows = [
        {**row, "label": truth[str(row["record_id"])]} for row in runtime
    ]
    components = _components(
        finalizer.model, finalizer.contract, runtime,
        finalizer.payload["role_grammar"], finalizer.device,
        finalizer.batch_size, baseline,
    )
    challenger, audit = apply_formula_syntax_rescue(
        runtime, baseline, components, finalizer.payload["role_grammar"],
        DEFAULT_CONFIGURATION,
    )
    changed_outcomes = []
    glyph_improved = glyph_regressed = 0
    for change in audit["changes"]:
        record_id = str(change["record_id"])
        before, after, expected = (
            baseline[record_id], challenger[record_id], truth[record_id]
        )
        outcome = (
            "improved" if before != expected and after == expected
            else "regressed" if before == expected and after != expected
            else "neutral"
        )
        glyph_improved += outcome == "improved"
        glyph_regressed += outcome == "regressed"
        changed_outcomes.append({**change, "truth": expected, "outcome": outcome})
    return {
        "input": {
            "candidates_path": str(candidate_path),
            "candidates_sha256": _sha256(candidate_path),
            "finalized_path": str(finalized_path),
            "finalized_sha256": _sha256(finalized_path),
        },
        "evaluation": _scope(evaluation_rows, baseline, challenger),
        "audit": {**audit, "changes": changed_outcomes},
        "horizontal_dash_diagnostic": _horizontal_dash_diagnostic(
            evaluation_rows, baseline, DEFAULT_CONFIGURATION,
        ),
        "glyph_improved": glyph_improved,
        "glyph_regressed": glyph_regressed,
        "layout_audit": layout_audit,
    }, challenger


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-candidates", type=Path)
    parser.add_argument("--direct-finalized", type=Path)
    parser.add_argument("--crohme-candidates", type=Path)
    parser.add_argument("--crohme-finalized", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--hwr-checkpoint", type=Path, default=DEFAULT_PRODUCT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config-output", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        rescue_self_test()
        print('{"self_test":"pass"}')
        return 0
    required = (
        args.direct_candidates, args.direct_finalized, args.crohme_candidates,
        args.crohme_finalized, args.output, args.config_output,
    )
    if any(value is None for value in required):
        parser.error("direct, CROHME, output, and config paths are required")
    paths = [Path(value).expanduser().resolve() for value in required]
    direct_candidates, direct_finalized, crohme_candidates, crohme_finalized, output, config_output = paths
    if not all(path.is_file() for path in paths[:4]):
        parser.error("formula syntax rescue inputs must exist")
    if output.exists() or config_output.exists():
        parser.error("refusing to overwrite formula syntax rescue evidence")

    finalizer = OwnedFormulaContextFinalizer(
        args.checkpoint, args.hwr_checkpoint,
        device=args.device, batch_size=args.batch_size, formula_layout=True,
    )
    direct, _ = _evaluate(direct_candidates, direct_finalized, finalizer)
    crohme, _ = _evaluate(crohme_candidates, crohme_finalized, finalizer)
    direct_outcomes = direct["evaluation"]["formula_outcomes"]
    crohme_outcomes = crohme["evaluation"]["formula_outcomes"]
    contract_passed = all(
        scope["audit"]["candidate_preservation_rate"] == 1.0
        and scope["audit"]["new_tokens"] == 0
        and scope["audit"]["deleted_glyphs"] == 0
        and scope["audit"]["grouping_mutations"] == 0
        for scope in (direct, crohme)
    )
    runtime_admitted = bool(
        direct_outcomes["improved"] > 0
        and direct_outcomes["regressed"] == 0
        and crohme_outcomes["regressed"] == 0
        and direct["glyph_regressed"] == 0
        and crohme["glyph_regressed"] == 0
        and contract_passed
    )
    gate = {
        "runtime_admitted": runtime_admitted,
        "status": "shadow_runtime_only",
        "criteria": {
            "direct_formula_improvement": direct_outcomes["improved"] > 0,
            "direct_formula_regressions_zero": direct_outcomes["regressed"] == 0,
            "crohme_formula_regressions_zero": crohme_outcomes["regressed"] == 0,
            "glyph_regressions_zero": (
                direct["glyph_regressed"] == crohme["glyph_regressed"] == 0
            ),
            "candidate_and_grouping_contract": contract_passed,
        },
        "limits": (
            "configuration was selected on repeatedly observed project-owned formulas; "
            "CROHME is noncommercial repeated diagnostic evidence; neither is an untouched product acceptance set"
        ),
    }
    generated_at = datetime.now(timezone.utc).isoformat()
    report = {
        "schema": SCHEMA, "generated_at": generated_at,
        "training_performed": False,
        "selection_scope": "project-owned direct research corpus only",
        "configuration": DEFAULT_CONFIGURATION,
        "checkpoint_sha256": finalizer.checkpoint_sha256,
        "hwr_checkpoint_sha256": finalizer.hwr_checkpoint_sha256,
        "gate": gate,
        "direct_project_owned": direct,
        "crohme_noncommercial_diagnostic": {
            **crohme, "product_validation": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    config = {
        "schema": CONFIG_SCHEMA,
        "rescue_schema": RESCUE_SCHEMA,
        "generated_at": generated_at,
        "status": "shadow_runtime_only",
        "configuration": DEFAULT_CONFIGURATION,
        "checkpoint_sha256": finalizer.checkpoint_sha256,
        "hwr_checkpoint_sha256": finalizer.hwr_checkpoint_sha256,
        "direct_candidate_sha256": _sha256(direct_candidates),
        "gate": gate,
        "evidence": {"path": str(output), "sha256": _sha256(output)},
    }
    config_output.parent.mkdir(parents=True, exist_ok=True)
    config_output.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    print(json.dumps({
        "report": str(output), "report_sha256": _sha256(output),
        "config": str(config_output), "config_sha256": _sha256(config_output),
        "gate": gate,
        "direct_formula_outcomes": direct_outcomes,
        "crohme_formula_outcomes": crohme_outcomes,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
