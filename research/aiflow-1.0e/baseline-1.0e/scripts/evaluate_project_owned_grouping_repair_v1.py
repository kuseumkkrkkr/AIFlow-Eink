#!/usr/bin/env python3
"""Evaluate audited ownership regrouping on the fixed 95-formula denominator."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path


SCHEMA = "aiflow-project-owned-grouping-repair-evaluation/v1"


def _rows(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _truth(rows: list[dict]) -> tuple[dict[str, str], dict[str, str]]:
    labels = {str(row["record_id"]): str(row["label"]) for row in rows}
    formulae = {str(row["record_id"]): str(row["formula_id"]) for row in rows}
    if not labels or len(labels) != len(rows) or set(labels) != set(formulae):
        raise ValueError("candidate truth identities are invalid")
    return labels, formulae


def _predictions(rows: list[dict], key: str) -> dict[str, str]:
    output = {}
    for row in rows:
        record_id = str(row["record_id"])
        if record_id in output:
            raise ValueError(f"duplicate prediction identity: {record_id}")
        value = row[key]
        output[record_id] = str(value[0] if key == "final_topk" else value)
    return output


def _metrics(
    truth: dict[str, str], formula_by_id: dict[str, str], predictions: dict[str, str],
) -> tuple[dict, dict[str, bool]]:
    if set(truth) != set(predictions):
        raise ValueError("truth and prediction coverage differ")
    grouped: dict[str, list[str]] = defaultdict(list)
    for record_id, formula_id in formula_by_id.items():
        grouped[formula_id].append(record_id)
    exact = {
        formula_id: all(predictions[record_id] == truth[record_id] for record_id in ids)
        for formula_id, ids in grouped.items()
    }
    characters = sum(predictions[key] == value for key, value in truth.items())
    formulas = sum(exact.values())
    return {
        "records": len(truth),
        "character_top1_count": characters,
        "character_top1": characters / len(truth),
        "formulas": len(grouped),
        "formula_exact_count": formulas,
        "formula_exact": formulas / len(grouped),
    }, exact


def evaluate(args: argparse.Namespace) -> dict:
    named_paths = {
        "old_candidates": args.old_candidates,
        "new_candidates": args.new_candidates,
        "old_final": args.old_final,
        "new_context_syntax": args.new_context_syntax,
        "new_singleton": args.new_singleton,
        "new_dual_hwr": args.new_dual_hwr,
        "new_wide_syntax": args.new_wide_syntax,
        "new_formula_placement": args.new_formula_placement,
        "repair_report": args.repair_report,
        "full_rank_before": args.full_rank_before,
        "full_rank_after": args.full_rank_after,
    }
    paths = {name: value.expanduser().resolve() for name, value in named_paths.items()}
    if not all(path.is_file() for path in paths.values()):
        missing = [name for name, path in paths.items() if not path.is_file()]
        raise FileNotFoundError(f"grouping evaluation inputs missing: {missing}")
    old_truth, old_formulae = _truth(_rows(paths["old_candidates"]))
    new_candidate_rows = _rows(paths["new_candidates"])
    truth, formulae = _truth(new_candidate_rows)
    if old_truth != truth or old_formulae != formulae:
        raise ValueError("grouping repair changed record IDs, formula IDs, or labels")
    stage_inputs = [
        ("raw_current_hwr", paths["new_candidates"], "final_topk"),
        ("context_layout_sequence_syntax", paths["new_context_syntax"], "finalized_top1"),
        ("singleton_shape", paths["new_singleton"], "finalized_top1"),
        ("dual_hwr_numeric", paths["new_dual_hwr"], "finalized_top1"),
        ("wide_syntax", paths["new_wide_syntax"], "finalized_top1"),
        ("formula_placement", paths["new_formula_placement"], "finalized_top1"),
    ]
    stages = {}
    stage_predictions = {}
    stage_exact = {}
    for name, path, key in stage_inputs:
        predictions = _predictions(_rows(path), key)
        stages[name], stage_exact[name] = _metrics(truth, formulae, predictions)
        stage_predictions[name] = predictions
    old_predictions = _predictions(_rows(paths["old_final"]), "finalized_top1")
    old_metrics, old_exact = _metrics(truth, formulae, old_predictions)
    new_predictions = stage_predictions["formula_placement"]
    new_exact = stage_exact["formula_placement"]
    improved_glyphs = [
        record_id for record_id in truth
        if old_predictions[record_id] != truth[record_id]
        and new_predictions[record_id] == truth[record_id]
    ]
    regressed_glyphs = [
        record_id for record_id in truth
        if old_predictions[record_id] == truth[record_id]
        and new_predictions[record_id] != truth[record_id]
    ]
    improved_formulae = sorted(
        formula_id for formula_id in new_exact
        if not old_exact[formula_id] and new_exact[formula_id]
    )
    regressed_formulae = sorted(
        formula_id for formula_id in new_exact
        if old_exact[formula_id] and not new_exact[formula_id]
    )
    repair = json.loads(paths["repair_report"].read_text(encoding="utf-8"))
    repaired_ids = sorted(str(row["sample_id"]) for row in repair["repairs"])
    before_rank = json.loads(paths["full_rank_before"].read_text(encoding="utf-8"))
    after_rank = json.loads(paths["full_rank_after"].read_text(encoding="utf-8"))
    reference = "0.6"
    return {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "training_performed": False,
        "fixed_denominator": {"records": len(truth), "formulas": len(set(formulae.values()))},
        "truth_contract": {
            "record_ids_unchanged": True,
            "formula_ids_unchanged": True,
            "labels_unchanged": True,
            "formulas_deleted": 0,
            "glyphs_deleted": 0,
        },
        "repair_invariants": repair["invariants"],
        "repaired_formulae": repaired_ids,
        "old_formula_placement": old_metrics,
        "new_stage_metrics": stages,
        "old_to_new_formula_placement": {
            "glyph_improved": len(improved_glyphs),
            "glyph_regressed": len(regressed_glyphs),
            "formula_improved": len(improved_formulae),
            "formula_regressed": len(regressed_formulae),
            "improved_formulae": improved_formulae,
            "regressed_formulae": regressed_formulae,
            "all_improved_formulae_are_repaired": set(improved_formulae) == set(repaired_ids),
        },
        "full_rank_recall_at_weight_0_6": {
            "before": before_rank["metrics_by_weight"][reference]["cutoffs"],
            "after": after_rank["metrics_by_weight"][reference]["cutoffs"],
            "runtime_candidate_width_changed": False,
        },
        "inputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in paths.items()
        },
        "limits": [
            "the 95 project-owned formulas are repeatedly observed development evidence",
            "regrouping accuracy gain is data-quality correction, not model retraining",
            "untouched commercial writer/formula acceptance remains required",
        ],
    }


def _self_test() -> None:
    truth = {"a": "1", "b": "="}
    formulae = {"a": "f", "b": "f"}
    metrics, exact = _metrics(truth, formulae, {"a": "1", "b": "="})
    assert metrics["character_top1_count"] == 2
    assert metrics["formula_exact_count"] == 1 and exact == {"f": True}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "old_candidates", "new_candidates", "old_final", "new_context_syntax",
        "new_singleton", "new_dual_hwr", "new_wide_syntax",
        "new_formula_placement", "repair_report", "full_rank_before",
        "full_rank_after",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", dest=name, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        print('{"self_test":"pass"}')
        return 0
    required = [getattr(args, name) for name in (
        "old_candidates", "new_candidates", "old_final", "new_context_syntax",
        "new_singleton", "new_dual_hwr", "new_wide_syntax",
        "new_formula_placement", "repair_report", "full_rank_before",
        "full_rank_after", "output",
    )]
    if any(value is None for value in required):
        parser.error("all evaluation inputs and --output are required")
    output = args.output.expanduser().resolve()
    if output.exists():
        parser.error(f"refusing to overwrite grouping evaluation: {output}")
    report = evaluate(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    print(json.dumps({
        "output": str(output), "sha256": _sha256(output),
        "fixed_denominator": report["fixed_denominator"],
        "old": report["old_formula_placement"],
        "new": report["new_stage_metrics"]["formula_placement"],
        "comparison": report["old_to_new_formula_placement"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
