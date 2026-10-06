#!/usr/bin/env python3
"""Audit the frozen semantic equation guard on an existing HWR microscope trace.

This is a post-hoc shadow diagnostic. It does not train, tune, load CROHME, or
alter the runtime default. Every proposed token is constrained to the captured
HWR Top-5 candidates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from semantic_equation_guard_v1 import (
    apply_semantic_equation_guard,
    is_exact_arithmetic_equation,
)


SCHEMA = "aiflow-hwr-semantic-equation-microscope-audit/v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(trace: dict[str, Any]) -> list[dict[str, Any]]:
    sample_id = str(trace["sample_id"])
    symbols = trace["oracle_group_hwr"]["symbols"]
    rows = []
    for index, symbol in enumerate(symbols):
        box = symbol["preprocessing"]["raw_bbox"]
        prediction = symbol["prediction"]
        left, right = float(box["left"]), float(box["right"])
        top, bottom = float(box["top"]), float(box["bottom"])
        rows.append({
            "record_id": f"{sample_id}:{index}",
            "formula_id": sample_id,
            "final_topk": list(prediction["top5"]),
            "final_topk_probabilities": list(prediction["top5_probabilities"]),
            "context": {"index": index, "length": len(symbols)},
            "geometry": {
                "left": left,
                "right": right,
                "top": top,
                "bottom": bottom,
                "center_x": (left + right) / 2.0,
                "center_y": (top + bottom) / 2.0,
                "width_rel": right - left,
                "height_rel": bottom - top,
            },
        })
    return rows


def _metric(rows: list[dict[str, Any]], tokens: list[str], target: list[str]) -> dict[str, Any]:
    if len(tokens) != len(target):
        raise AssertionError("token/target count mismatch")
    return {
        "token_count": len(target),
        "token_hits": sum(token == label for token, label in zip(tokens, target, strict=True)),
        "formula_exact": tokens == target,
        "candidate_preservation": all(
            token in row["final_topk"] for token, row in zip(tokens, rows, strict=True)
        ),
    }


def audit(input_path: Path) -> dict[str, Any]:
    traces = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    before_hits = after_hits = after_infix_hits = 0
    before_exact = after_exact = after_infix_exact = 0
    recovered_vs_hwr = regressed_vs_hwr = 0
    recovered_formulas_vs_hwr = regressed_formulas_vs_hwr = 0
    token_count = 0
    candidate_preserved_tokens = 0
    totals: Counter[str] = Counter()
    formula_changes = []
    guard_equation_outputs_valid = True
    guard_change_limits_respected = True

    for trace in traces:
        sample_id = str(trace["sample_id"])
        target = [str(token) for token in trace["source"]["target_tokens"]]
        rows = _rows(trace)
        if len(rows) != len(target):
            raise AssertionError(f"HWR rows/labels mismatch: {sample_id}")
        before_tokens = list(trace["semantic_guard_shadow"]["stage_tokens"]["hwr_top1"])
        after_infix_tokens = list(trace["semantic_guard_shadow"]["stage_tokens"]["after_infix_guard"])
        if len(before_tokens) != len(target) or len(after_infix_tokens) != len(target):
            raise AssertionError(f"semantic stage token count mismatch: {sample_id}")
        predictions = {row["record_id"]: token for row, token in zip(rows, after_infix_tokens, strict=True)}
        after, guard_audit = apply_semantic_equation_guard(rows, predictions)
        after_tokens = [after[row["record_id"]] for row in rows]
        for change in guard_audit.get("changes", []):
            guard_equation_outputs_valid &= is_exact_arithmetic_equation(
                tuple(change["after"])
            )
            guard_change_limits_respected &= 1 <= len(change["changed_positions"]) <= 3

        if not all(token in row["final_topk"] for token, row in zip(after_tokens, rows, strict=True)):
            raise AssertionError(f"Top-5 candidate preservation failed: {sample_id}")
        candidate_preserved_tokens += sum(
            token in row["final_topk"]
            for token, row in zip(after_tokens, rows, strict=True)
        )
        before_metric = _metric(rows, before_tokens, target)
        infix_metric = _metric(rows, after_infix_tokens, target)
        after_metric = _metric(rows, after_tokens, target)
        before_hits += before_metric["token_hits"]
        after_infix_hits += infix_metric["token_hits"]
        after_hits += after_metric["token_hits"]
        before_exact += int(before_metric["formula_exact"])
        after_infix_exact += int(infix_metric["formula_exact"])
        after_exact += int(after_metric["formula_exact"])
        recovered_vs_hwr += sum(
            baseline != truth and final == truth
            for baseline, final, truth in zip(before_tokens, after_tokens, target, strict=True)
        )
        regressed_vs_hwr += sum(
            baseline == truth and final != truth
            for baseline, final, truth in zip(before_tokens, after_tokens, target, strict=True)
        )
        recovered_formulas_vs_hwr += int(
            after_metric["formula_exact"] and not before_metric["formula_exact"]
        )
        regressed_formulas_vs_hwr += int(
            not after_metric["formula_exact"] and before_metric["formula_exact"]
        )
        token_count += len(target)
        totals.update({key: int(value) for key, value in guard_audit.items() if isinstance(value, int)})

        changed = []
        for index, (old, new, truth, row) in enumerate(
            zip(after_infix_tokens, after_tokens, target, rows, strict=True)
        ):
            if old == new:
                continue
            changed.append({
                "position": index,
                "before": old,
                "after": new,
                "target": truth,
                "before_correct": old == truth,
                "after_correct": new == truth,
                "candidate_rank": row["final_topk"].index(new) + 1,
                "candidate_probability": row["final_topk_probabilities"][row["final_topk"].index(new)],
                "top5": row["final_topk"],
            })
        if changed or guard_audit.get("eligible_formulas"):
            formula_changes.append({
                "sample_id": sample_id,
                "target_tokens": target,
                "hwr_top1_tokens": before_tokens,
                "after_infix_tokens": after_infix_tokens,
                "after_equation_guard_tokens": after_tokens,
                "exact": {
                    "hwr_top1": before_metric["formula_exact"],
                    "after_infix": infix_metric["formula_exact"],
                    "after_equation_guard": after_metric["formula_exact"],
                },
                "changes": changed,
                "guard_audit": guard_audit,
            })

    changed_glyphs = [
        change
        for formula in formula_changes
        for change in formula["changes"]
    ]
    recovered_glyphs = sum(
        not change["before_correct"] and change["after_correct"]
        for change in changed_glyphs
    )
    regressed_glyphs = sum(
        change["before_correct"] and not change["after_correct"]
        for change in changed_glyphs
    )
    recovered_formulas = sum(
        item["exact"]["after_equation_guard"] and not item["exact"]["after_infix"]
        for item in formula_changes
    )
    regressed_formulas = sum(
        not item["exact"]["after_equation_guard"] and item["exact"]["after_infix"]
        for item in formula_changes
    )
    checks = {
        "frozen_formula_count_149": len(traces) == 149,
        "frozen_token_count_579": token_count == 579,
        "hwr_top1_baseline_470_tokens_76_exact": before_hits == 470 and before_exact == 76,
        "infix_stage_baseline_485_tokens_84_exact": (
            after_infix_hits == 485 and after_infix_exact == 84
        ),
        "candidate_preservation_all_tokens": candidate_preserved_tokens == token_count,
        "no_formula_regressions_vs_infix": regressed_formulas == 0,
        "no_formula_regressions_vs_hwr_top1": regressed_formulas_vs_hwr == 0,
        "all_changed_glyphs_recovered": (
            recovered_glyphs == len(changed_glyphs) and regressed_glyphs == 0
        ),
        "no_hwr_top1_glyph_regressions": regressed_vs_hwr == 0,
        "guard_outputs_are_exact_equations": guard_equation_outputs_valid,
        "guard_change_limit_respected": guard_change_limits_respected,
        "no_group_or_glyph_invention": (
            totals.get("new_tokens", 0) == 0
            and totals.get("deleted_glyphs", 0) == 0
            and totals.get("grouping_mutations", 0) == 0
        ),
    }
    if not all(checks.values()):
        raise AssertionError(f"semantic equation microscope checks failed: {checks}")
    return {
        "schema": SCHEMA,
        "scope": "frozen microscope diagnostic only; existing guard constants; no training/tuning/CROHME; no runtime promotion",
        "input": {"path": str(input_path), "sha256": _sha256(input_path)},
        "formulas": len(traces),
        "tokens": token_count,
        "guard_configuration": {
            "minimum_formula_length": 3,
            "maximum_formula_length": 12,
            "maximum_changed_glyphs": 3,
            "beam_width": 64,
            "preferred_token_bonus": 0.75,
            "minimum_valid_score_margin": 1.5,
            "minimum_vertical_overlap": 0.25,
        },
        "metrics": {
            "hwr_top1": {"token_hits": before_hits, "formula_exact": before_exact},
            "after_infix": {"token_hits": after_infix_hits, "formula_exact": after_infix_exact},
            "after_equation_guard": {"token_hits": after_hits, "formula_exact": after_exact},
            "delta_vs_infix": {
                "token_hits": after_hits - after_infix_hits,
                "formula_exact": after_exact - after_infix_exact,
                "recovered_glyphs": recovered_glyphs,
                "regressed_glyphs": regressed_glyphs,
                "recovered_formulas": recovered_formulas,
                "regressed_formulas": regressed_formulas,
                "changed_glyphs": len(changed_glyphs),
            },
            "delta_vs_hwr_top1": {
                "token_hits": after_hits - before_hits,
                "formula_exact": after_exact - before_exact,
                "recovered_glyphs": recovered_vs_hwr,
                "regressed_glyphs": regressed_vs_hwr,
                "recovered_formulas": recovered_formulas_vs_hwr,
                "regressed_formulas": regressed_formulas_vs_hwr,
            },
            "fixed_guard_audit_totals": dict(sorted(totals.items())),
            "candidate_preservation_rate": 1.0,
            "product_default_enabled": False,
        },
        "verification": {"checks": checks, "all_checks_pass": all(checks.values())},
        "formula_level_trace": formula_changes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-traces", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.input_traces)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"metrics": report["metrics"], "verification": report["verification"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
