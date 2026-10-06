#!/usr/bin/env python3
"""Select and evaluate the conservative formula sequence guard.

Only the legacy47 project-owned partition selects scalar weights.  New-arrival
partitions are a fixed gate and CROHME is not used here for fitting or model
selection.  The guard never evaluates arithmetic and never escapes HWR Top-5.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

from character_tensor_v1 import _json_lines
from evaluate_48hz_prefix_v1 import DEFAULT_PRODUCT, _sha256
from evaluate_homograph_context_reranker_v1 import _metrics
from finalize_formula_context_v1 import (
    DEFAULT_CONTEXT, OwnedFormulaContextFinalizer, _runtime_rows,
)
from formula_layout_v1 import recontextualize_formula_rows
from formula_sequence_guard_v1 import (
    DEFAULT_CONFIGURATION, SCHEMA as GUARD_SCHEMA, apply_formula_sequence_guard,
)
import train_masked_context_reranker_v1 as masked
from train_owned_formula_context_v1 import _components


SCHEMA = "aiflow-formula-sequence-guard-evaluation/v1"
CONFIG_SCHEMA = "aiflow-formula-sequence-guard-config/v1"


def _formula_outcomes(
    rows: list[dict], baseline: dict[str, str], challenger: dict[str, str],
) -> dict:
    improved = regressed = baseline_exact = challenger_exact = 0
    failures = []
    for formula_id, sequence in masked._formulae(rows).items():
        before = all(
            baseline[str(row["record_id"])] == str(row["label"])
            for row in sequence
        )
        after = all(
            challenger[str(row["record_id"])] == str(row["label"])
            for row in sequence
        )
        baseline_exact += before
        challenger_exact += after
        improved += after and not before
        regressed += before and not after
        if before != after and len(failures) < 30:
            failures.append({
                "formula_id": str(formula_id),
                "direction": "improved" if after else "regressed",
                "truth": [str(row["label"]) for row in sequence],
                "before": [baseline[str(row["record_id"])] for row in sequence],
                "after": [challenger[str(row["record_id"])] for row in sequence],
            })
    count = len(masked._formulae(rows))
    return {
        "formulas": count,
        "baseline_exact": baseline_exact,
        "challenger_exact": challenger_exact,
        "baseline_formula_exact": baseline_exact / max(count, 1),
        "challenger_formula_exact": challenger_exact / max(count, 1),
        "delta_formula_exact": (challenger_exact - baseline_exact) / max(count, 1),
        "improved": improved,
        "regressed": regressed,
        "changed_outcomes": failures,
    }


def _scope(
    rows: list[dict], baseline: dict[str, str], challenger: dict[str, str],
) -> dict:
    return {
        "baseline": _metrics(rows, baseline),
        "challenger": _metrics(rows, challenger),
        "formula_outcomes": _formula_outcomes(rows, baseline, challenger),
    }


def _partitions(
    rows: list[dict], baseline: dict[str, str], challenger: dict[str, str],
) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("evaluation_partition", "unspecified"))].append(row)
    output = {"all": _scope(rows, baseline, challenger)}
    for name, subset in sorted(grouped.items()):
        output[name] = _scope(subset, baseline, challenger)
    arrivals = [
        row for row in rows
        if str(row.get("evaluation_partition", "unspecified")) != "legacy47"
    ]
    if arrivals:
        output["new_arrivals"] = _scope(arrivals, baseline, challenger)
    return output


def _grid() -> list[dict]:
    output = []
    for shape_weight in (0.0, 0.05, 0.10, 0.25, 0.50):
        for context_weight in (0.25, 0.50, 1.00, 2.00):
            for grammar_weight in (0.0, 0.25, 0.50, 1.00):
                for retention_bonus in (0.0, 0.25, 0.50):
                    for margin in (0.0, 0.25, 0.50, 1.00):
                        output.append({
                            **DEFAULT_CONFIGURATION,
                            "shape_weight": shape_weight,
                            "context_weight": context_weight,
                            "grammar_weight": grammar_weight,
                            "retention_bonus": retention_bonus,
                            "margin": margin,
                        })
    return output


def _selection_key(result: dict) -> tuple:
    metrics = result["legacy_metrics"]
    outcomes = result["legacy_outcomes"]
    return (
        -outcomes["regressed"],
        outcomes["challenger_exact"],
        metrics["all_top1"],
        metrics["strict_macro_top1"],
        -result["legacy_changed"],
        -result["index"],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--hwr-checkpoint", type=Path, default=DEFAULT_PRODUCT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config-output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    input_path = Path(args.input).expanduser().resolve()
    if input_path.is_dir():
        input_path = input_path / "direct_candidates.jsonl.gz"
    output = Path(args.output).expanduser().resolve()
    config_output = Path(args.config_output).expanduser().resolve()
    if not input_path.is_file():
        parser.error(f"direct candidate input is missing: {input_path}")
    if output.exists() or config_output.exists():
        parser.error("refusing to overwrite sequence guard evidence")

    rows = list(_json_lines(input_path))
    finalizer = OwnedFormulaContextFinalizer(
        args.checkpoint, args.hwr_checkpoint,
        device=args.device, batch_size=args.batch_size,
        formula_layout=True,
    )
    baseline_rows, baseline_audit = finalizer.finalize(rows)
    baseline = {
        str(row["record_id"]): str(row["finalized_top1"])
        for row in baseline_rows
    }
    runtime_rows, layout_audit = recontextualize_formula_rows(
        _runtime_rows(rows, finalizer.labels, require_context=False)
    )
    truth_by_record = {
        str(row["record_id"]): str(row["label"])
        for row in rows
    }
    evaluation_rows = [
        {**row, "label": truth_by_record[str(row["record_id"])]}
        for row in runtime_rows
    ]
    components = _components(
        finalizer.model, finalizer.contract, runtime_rows,
        finalizer.payload["role_grammar"], finalizer.device,
        finalizer.batch_size, baseline,
    )
    legacy = [
        row for row in evaluation_rows
        if str(row.get("evaluation_partition")) == "legacy47"
    ]
    arrivals = [
        row for row in evaluation_rows
        if str(row.get("evaluation_partition")) != "legacy47"
    ]
    if not legacy or not arrivals:
        raise ValueError("sequence selection requires legacy47 and new-arrival partitions")
    legacy_formula_ids = {str(row["formula_id"]) for row in legacy}

    results = []
    for index, configuration in enumerate(_grid()):
        predictions, audit = apply_formula_sequence_guard(
            runtime_rows, baseline, components, finalizer.payload["role_grammar"],
            configuration,
        )
        results.append({
            "index": index,
            "configuration": configuration,
            "predictions": predictions,
            "audit": audit,
            "legacy_changed": sum(
                str(change["formula_id"]) in legacy_formula_ids
                for change in audit["changes"]
            ),
            "legacy_metrics": _metrics(legacy, predictions),
            "legacy_outcomes": _formula_outcomes(legacy, baseline, predictions),
        })
    selected = max(results, key=_selection_key)
    predictions = selected.pop("predictions")
    evaluation = _partitions(evaluation_rows, baseline, predictions)
    legacy_gate = evaluation["legacy47"]["formula_outcomes"]
    arrival_gate = evaluation["new_arrivals"]["formula_outcomes"]
    candidate_preserved = all(
        predictions[str(row["record_id"])] in row["final_topk"]
        for row in runtime_rows
    )
    shadow_admitted = bool(
        legacy_gate["improved"] > 0
        and legacy_gate["regressed"] == 0
        and arrival_gate["regressed"] == 0
        and candidate_preserved
    )
    gate = {
        "shadow_admitted": shadow_admitted,
        "criteria": {
            "legacy_formula_improvement": legacy_gate["improved"] > 0,
            "legacy_formula_regressions_zero": legacy_gate["regressed"] == 0,
            "new_arrival_formula_regressions_zero": arrival_gate["regressed"] == 0,
            "candidate_preservation": candidate_preserved,
        },
        "limits": (
            "legacy47 trained the context model and selected these scalar weights; "
            "new arrivals are a fixed observed gate, not an untouched commercial acceptance set"
        ),
    }
    config_payload = {
        "schema": CONFIG_SCHEMA,
        "guard_schema": GUARD_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "configuration": selected["configuration"],
        "selection_scope": "legacy47 only",
        "checkpoint_sha256": finalizer.checkpoint_sha256,
        "hwr_checkpoint_sha256": finalizer.hwr_checkpoint_sha256,
        "direct_candidate_sha256": _sha256(input_path),
        "gate": gate,
    }
    report = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "training_performed": False,
        "arithmetic_evaluation": False,
        "input": {"path": str(input_path), "sha256": _sha256(input_path)},
        "checkpoint_sha256": finalizer.checkpoint_sha256,
        "hwr_checkpoint_sha256": finalizer.hwr_checkpoint_sha256,
        "selection": {
            "grid_candidates": len(results),
            **selected,
        },
        "gate": gate,
        "evaluation": evaluation,
        "baseline_runtime_audit": baseline_audit,
        "layout_audit": layout_audit,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    config_output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    config_output.write_text(
        json.dumps(config_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    print(json.dumps({
        "report": str(output),
        "config": str(config_output),
        "gate": gate,
        "legacy": evaluation["legacy47"]["formula_outcomes"],
        "new_arrivals": evaluation["new_arrivals"]["formula_outcomes"],
        "all": evaluation["all"]["formula_outcomes"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
