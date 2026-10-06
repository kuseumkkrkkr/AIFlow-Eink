#!/usr/bin/env python3
"""Compose the owned-prompt n-gram with a conservative function-call layout guard."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable

from evaluate_48hz_prefix_v1 import _sha256
from evaluate_raw_formula_layout_runtime_v1 import _truth


SCHEMA = "aiflow-ngram-layout-guard-evaluation/v1"
RUNTIME_SCHEMA = "aiflow-raw-formula-context-runtime/v1"
NGRAM_SCHEMA = "aiflow-owned-prompt-ngram-reranker/v1"
FUNCTION_TOKENS = frozenset({"f", "g", "h", "F", "G", "H"})
DEFAULT_RUNTIME = Path(r"D:\AIFlow-Workspace\PrivateData\candidate-context-runtime-20260820-r43-layout-shadow-selected.json")
DEFAULT_TRUTH = Path(r"D:\AIFlow-Workspace\PrivateData\math-ink-data-collector\derived\public-candidate-20260819-r3\data\formulas_valid.jsonl")
DEFAULT_NGRAM = Path(r"D:\AIFlow-Workspace\Projects\Aiflow\aiflow-math-ink-1.0\artifacts\owned_prompt_ngram_20260820_r5_writer_loo_shadow\owned_prompt_ngram_report.json")
DEFAULT_OUTPUT = Path(r"D:\AIFlow-Workspace\Projects\Aiflow\aiflow-math-ink-1.0\artifacts\ngram_layout_guard_20260820_r3_writer_loo_shadow\ngram_layout_guard_evaluation.json")


def _d_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.drive.upper() != "D:" or not resolved.is_file():
        raise ValueError(f"{label} must be an existing D: file: {resolved}")
    return resolved


def _box_overlap(first: dict[str, float], second: dict[str, float]) -> float:
    overlap = max(0.0, min(float(first["bottom"]), float(second["bottom"])) - max(float(first["top"]), float(second["top"])))
    first_height = max(float(first["bottom"]) - float(first["top"]), 1e-6)
    second_height = max(float(second["bottom"]) - float(second["top"]), 1e-6)
    return overlap / min(first_height, second_height)


def _guard_relations(
    formula: dict[str, Any], labels: dict[str, str], ordered: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    position = {record_id: index for index, record_id in enumerate(ordered)}
    boxes = {str(row["record_id"]): row["box"] for row in formula["groups"]}
    retained = []
    removed = []
    for edge in formula["formula_layout_shadow"]["relations"]:
        parent = str(edge["parent"])
        child = str(edge["child"])
        relation = str(edge["type"])
        suppress = (
            relation in {"superscript", "subscript"}
            and labels.get(parent) in FUNCTION_TOKENS
            and labels.get(child) == "("
            and position.get(child) == position.get(parent, -2) + 1
            and _box_overlap(boxes[parent], boxes[child]) >= 0.7
            and any(labels.get(record_id) == ")" for record_id in ordered[position[child] + 1:])
        )
        if suppress:
            removed.append({
                **edge,
                "guard": "adjacent_function_open_fence_with_high_vertical_overlap",
                "vertical_overlap": _box_overlap(boxes[parent], boxes[child]),
            })
        else:
            retained.append(edge)
    return retained, removed


def _serialize(
    labels: dict[str, str], ordered: list[str], relations: list[dict[str, Any]],
) -> str:
    children: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    structural_children = set()
    for edge in relations:
        parent = str(edge["parent"])
        child = str(edge["child"])
        children[parent][str(edge["type"])].append(child)
        structural_children.add(child)
    position = {record_id: index for index, record_id in enumerate(ordered)}
    emitted: set[str] = set()

    def sequence(record_ids: Iterable[str], active: frozenset[str]) -> str:
        return "".join(node(record_id, active) for record_id in sorted(set(record_ids), key=position.__getitem__))

    def node(record_id: str, active: frozenset[str]) -> str:
        if record_id in active:
            raise ValueError("layout guard serialization cycle")
        if record_id in emitted:
            return ""
        emitted.add(record_id)
        nested = active | {record_id}
        slots = children.get(record_id, {})
        above = slots.get("above", [])
        below = slots.get("below", [])
        contained = slots.get("contains", [])
        if above and below:
            base = rf"\frac{{{sequence(above, nested)}}}{{{sequence(below, nested)}}}"
        elif contained:
            base = rf"\sqrt{{{sequence(contained, nested)}}}"
        else:
            base = labels[record_id]
        if slots.get("subscript"):
            base += rf"_{{{sequence(slots['subscript'], nested)}}}"
        if slots.get("superscript"):
            base += rf"^{{{sequence(slots['superscript'], nested)}}}"
        return base

    roots = [record_id for record_id in ordered if record_id not in structural_children]
    return sequence(roots, frozenset())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--truth", type=Path, default=DEFAULT_TRUTH)
    parser.add_argument("--ngram", type=Path, default=DEFAULT_NGRAM)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    runtime_path = _d_file(args.runtime, "runtime")
    truth_path = _d_file(args.truth, "truth")
    ngram_path = _d_file(args.ngram, "n-gram report")
    output = args.output.expanduser().resolve()
    if output.drive.upper() != "D:" or output.exists():
        parser.error("output must be a new D: file")
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    ngram = json.loads(ngram_path.read_text(encoding="utf-8"))
    if (
        runtime.get("schema") != RUNTIME_SCHEMA
        or ngram.get("schema") != NGRAM_SCHEMA
        or ngram.get("decision", {}).get("direct_nonregression") is not True
        or ngram.get("decision", {}).get("current_zero_regression") is not True
        or ngram.get("decision", {}).get("current_exact_gain") is not True
        or ngram.get("contracts", {}).get("current_truth_used_for_weight_selection") is not False
        or ngram.get("contracts", {}).get("arithmetic_evaluation") is not False
    ):
        raise ValueError("n-gram layout guard input contract failed")
    truth, truth_inventory = _truth(truth_path)
    token_changes = {
        str(row["formula_id"]): list(row["after"])
        for row in ngram["current159"]["changed_formulas"]
    }
    formulas = []
    relation_counts: Counter[str] = Counter()
    for formula in runtime["formulas"]:
        formula_id = str(formula["formula_id"])
        ordered = [str(value) for value in formula["formula_layout_shadow"]["ordered_record_ids"]]
        symbol_order = sorted(formula["symbols"], key=lambda row: int(row["context_index"]))
        tokens = token_changes.get(formula_id, list(formula["finalized_tokens"]))
        if len(tokens) != len(symbol_order):
            raise ValueError(f"n-gram token coverage mismatch: {formula_id}")
        labels = {
            str(symbol["record_id"]): str(token)
            for symbol, token in zip(symbol_order, tokens, strict=True)
        }
        if set(labels) != set(ordered):
            raise ValueError(f"layout record coverage mismatch: {formula_id}")
        retained, removed = _guard_relations(formula, labels, ordered)
        relation_counts.update(str(row["type"]) for row in retained)
        challenger = _serialize(labels, ordered, retained)
        baseline = str(formula["formula_layout_shadow"]["latex"])
        formulas.append({
            "formula_id": formula_id,
            "decision_status": formula["decision_status"],
            "baseline_latex": baseline,
            "challenger_latex": challenger,
            "truth_latex": truth[formula_id],
            "baseline_exact": baseline == truth[formula_id],
            "challenger_exact": challenger == truth[formula_id],
            "token_changed": formula_id in token_changes,
            "removed_relations": removed,
        })
    improved = [row["formula_id"] for row in formulas if not row["baseline_exact"] and row["challenger_exact"]]
    regressed = [row["formula_id"] for row in formulas if row["baseline_exact"] and not row["challenger_exact"]]
    changed = [row for row in formulas if row["baseline_latex"] != row["challenger_latex"]]
    subsets = {
        "all": formulas,
        "auto_accepted": [row for row in formulas if row["decision_status"] == "AUTO_ACCEPTED"],
        "review_required": [row for row in formulas if row["decision_status"] == "REVIEW_REQUIRED"],
    }
    payload = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "shadow_only",
        "scores": {
            name: {
                "formulas": len(rows),
                "baseline_exact": sum(row["baseline_exact"] for row in rows),
                "challenger_exact": sum(row["challenger_exact"] for row in rows),
            }
            for name, rows in subsets.items()
        },
        "changes": {
            "changed_formulas": changed,
            "improved_formula_ids": improved,
            "regressed_formula_ids": regressed,
            "removed_relation_count": sum(len(row["removed_relations"]) for row in formulas),
            "retained_relation_counts": dict(sorted(relation_counts.items())),
        },
        "truth_inventory": truth_inventory,
        "decision": {
            "current_exact_gain": len(improved) > 0,
            "current_zero_regression": not regressed,
            "auto_accepted_zero_regression": all(row["challenger_exact"] for row in subsets["auto_accepted"]),
            "runtime_status": "shadow",
            "automatic_default_replacement": False,
            "promotion_requirement": "fresh commercial writer/formula-disjoint acceptance",
        },
        "contracts": {
            "candidate_tokens_from_ngram_top32_only": True,
            "relation_guard_uses_tokens_and_geometry_only": True,
            "target_relation_used_by_guard": False,
            "arithmetic_evaluation": False,
            "grouping_mutations": 0,
            "inserted_or_deleted_glyphs": 0,
            "product_default_enabled": False,
        },
        "sources": {
            "runtime": {"path": str(runtime_path), "sha256": _sha256(runtime_path)},
            "truth": {"path": str(truth_path), "sha256": _sha256(truth_path)},
            "ngram": {"path": str(ngram_path), "sha256": _sha256(ngram_path)},
        },
    }
    output.parent.mkdir(parents=True, exist_ok=False)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({
        "event": "ngram_layout_guard_complete", "output": str(output),
        "sha256": _sha256(output), "scores": payload["scores"],
        "improved": improved, "regressed": regressed,
        "removed_relations": payload["changes"]["removed_relation_count"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
