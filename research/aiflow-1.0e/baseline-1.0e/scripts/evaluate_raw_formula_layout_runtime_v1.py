#!/usr/bin/env python3
"""Evaluate opt-in formula-layout output without changing HWR decisions."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from evaluate_48hz_prefix_v1 import _sha256


SCHEMA = "aiflow-raw-formula-layout-runtime-evaluation/v1"
RUNTIME_SCHEMA = "aiflow-raw-formula-context-runtime/v1"


def _d_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.drive.upper() != "D:" or not resolved.is_file():
        raise ValueError(f"{label} must be an existing file on D: {resolved}")
    return resolved


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _truth(path: Path) -> tuple[dict[str, str], dict[str, Any]]:
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 159:
        raise ValueError(f"current project truth must contain 159 formulas, got {len(rows)}")
    expected: dict[str, str] = {}
    root_formulas: list[str] = []
    relation_formulas: list[str] = []
    relation_types: set[str] = set()
    for row in rows:
        formula_id = str(row["sample_id"])
        cells = row.get("target_cells") or []
        tokens = [str(cell["token"]) for cell in cells]
        relations = row.get("target_relations") or []
        roots = [index for index, token in enumerate(tokens) if token == "\\sqrt{}"]
        if roots:
            if (
                roots != [0] or relations or len(tokens) != 4
                or tokens[1] != "(" or tokens[3] != ")"
            ):
                raise ValueError(f"unsupported project root truth pattern: {formula_id}")
            expected[formula_id] = "\\sqrt{" + "".join(tokens[1:]) + "}"
            root_formulas.append(formula_id)
            continue

        children: set[int] = set()
        by_parent: dict[int, tuple[str, int]] = {}
        for relation in relations:
            relation_type = str(relation["type"])
            if relation_type not in {"SUPERSCRIPT", "SUBSCRIPT"}:
                raise ValueError(
                    f"unsupported project relation truth: {formula_id} {relation_type}"
                )
            parent = int(relation["from"])
            child = int(relation["to"])
            if (
                parent < 0 or parent >= len(tokens) or child < 0 or child >= len(tokens)
                or parent in by_parent or child in children or parent == child
            ):
                raise ValueError(f"ambiguous project relation truth: {formula_id}")
            by_parent[parent] = (relation_type, child)
            children.add(child)
            relation_types.add(relation_type)
        if relations:
            relation_formulas.append(formula_id)
        output: list[str] = []
        for index, token in enumerate(tokens):
            if index in children:
                continue
            output.append(token)
            if index in by_parent:
                relation_type, child = by_parent[index]
                marker = "^" if relation_type == "SUPERSCRIPT" else "_"
                output.append(f"{marker}{{{tokens[child]}}}")
        expected[formula_id] = "".join(output)
    if len(expected) != len(rows):
        raise ValueError("duplicate project truth formula ids")
    return expected, {
        "formulas": len(rows),
        "root_formulas": sorted(root_formulas),
        "relation_formulas": sorted(relation_formulas),
        "relation_types": sorted(relation_types),
        "serialization_policy": (
            "target cells plus declared SUPERSCRIPT/SUBSCRIPT; project root wrapper pattern"
        ),
    }


def _runtime(path: Path, expected: set[str]) -> tuple[dict[str, Any], dict[str, dict]]:
    payload = _json(path)
    if (
        payload.get("schema") != RUNTIME_SCHEMA
        or payload.get("status") != "development_only_posthoc_shadow"
    ):
        raise ValueError(f"unexpected runtime contract: {path}")
    rows = payload.get("formulas") or []
    mapped = {str(row["formula_id"]): row for row in rows}
    if len(mapped) != len(rows) or set(mapped) != expected:
        raise ValueError(f"runtime formula coverage mismatch: {path}")
    return payload, mapped


def _without_layout(row: dict[str, Any]) -> dict[str, Any]:
    clean = {key: value for key, value in row.items() if key not in {
        "formula_latex_shadow", "formula_layout_shadow",
    }}
    audit = dict(clean["audit"])
    audit.pop("formula_layout_shadow", None)
    clean["audit"] = audit
    return clean


def _score(predictions: dict[str, str], truth: dict[str, str], ids: list[str]) -> dict:
    exact = [formula_id for formula_id in ids if predictions[formula_id] == truth[formula_id]]
    return {
        "formulas": len(ids),
        "exact_count": len(exact),
        "exact": len(exact) / max(len(ids), 1),
        "failures": [formula_id for formula_id in ids if formula_id not in exact],
    }


def evaluate(
    truth_path: Path, baseline_path: Path, layout_path: Path,
    crohme_layout_path: Path, crohme_layout_baseline_path: Path | None = None,
) -> dict[str, Any]:
    paths = {
        "truth": _d_file(truth_path, "project truth"),
        "baseline_runtime": _d_file(baseline_path, "baseline runtime"),
        "layout_runtime": _d_file(layout_path, "layout runtime"),
        "crohme_layout_evaluation": _d_file(
            crohme_layout_path, "CROHME layout evaluation",
        ),
    }
    if crohme_layout_baseline_path is not None:
        paths["crohme_layout_baseline_evaluation"] = _d_file(
            crohme_layout_baseline_path, "CROHME layout baseline evaluation",
        )
    truth, truth_inventory = _truth(paths["truth"])
    expected_ids = set(truth)
    baseline_payload, baseline = _runtime(paths["baseline_runtime"], expected_ids)
    layout_payload, layout = _runtime(paths["layout_runtime"], expected_ids)
    formula_ids = sorted(truth)

    if any(
        "formula_latex_shadow" in row or "formula_layout_shadow" in row
        for row in baseline.values()
    ) or "formula_layout_shadow_emitted" in baseline_payload.get("audit", {}):
        raise ValueError("baseline runtime unexpectedly emitted formula layout")
    missing_layout = [
        formula_id for formula_id, row in layout.items()
        if "formula_latex_shadow" not in row or "formula_layout_shadow" not in row
    ]
    if missing_layout:
        raise ValueError(f"layout runtime output missing: {missing_layout}")
    if layout_payload.get("audit", {}).get("formula_layout_shadow_emitted") != 159:
        raise ValueError("layout runtime top-level emission audit mismatch")

    mutations = [
        formula_id for formula_id in formula_ids
        if _without_layout(baseline[formula_id]) != _without_layout(layout[formula_id])
    ]
    if mutations:
        raise ValueError(f"formula layout mutated upstream output: {mutations}")

    layout_audits = [row["audit"]["formula_layout_shadow"] for row in layout.values()]
    if any(
        audit.get("enabled") is not True
        or audit.get("feedback_into_character_model") is not False
        or audit.get("product_default_enabled") is not False
        for audit in layout_audits
    ):
        raise ValueError("formula layout shadow boundary mismatch")

    flat_predictions = {
        formula_id: "".join(row["finalized_tokens"])
        for formula_id, row in baseline.items()
    }
    layout_predictions = {
        formula_id: str(row["formula_latex_shadow"])
        for formula_id, row in layout.items()
    }
    improved = [
        formula_id for formula_id in formula_ids
        if flat_predictions[formula_id] != truth[formula_id]
        and layout_predictions[formula_id] == truth[formula_id]
    ]
    regressed = [
        formula_id for formula_id in formula_ids
        if flat_predictions[formula_id] == truth[formula_id]
        and layout_predictions[formula_id] != truth[formula_id]
    ]
    changed = [
        formula_id for formula_id in formula_ids
        if flat_predictions[formula_id] != layout_predictions[formula_id]
    ]
    auto_ids = [
        formula_id for formula_id in formula_ids
        if layout[formula_id]["decision_status"] == "AUTO_ACCEPTED"
    ]
    review_ids = [
        formula_id for formula_id in formula_ids
        if layout[formula_id]["decision_status"] == "REVIEW_REQUIRED"
    ]
    if len(auto_ids) + len(review_ids) != len(formula_ids):
        raise ValueError("unexpected formula acceptance decision")

    flat_score = _score(flat_predictions, truth, formula_ids)
    layout_score = _score(layout_predictions, truth, formula_ids)
    accepted_score = _score(layout_predictions, truth, auto_ids)
    review_score = _score(layout_predictions, truth, review_ids)
    if regressed or layout_score["exact_count"] <= flat_score["exact_count"]:
        raise ValueError("formula layout selection gate failed")
    if accepted_score["exact_count"] != accepted_score["formulas"]:
        raise ValueError("formula layout auto-accepted precision gate failed")

    crohme = _json(paths["crohme_layout_evaluation"])
    if (
        crohme.get("schema") != "aiflow-formula-layout-evaluation/v1"
        or crohme.get("crohme_noncommercial", {}).get("product_validation") is not False
    ):
        raise ValueError("CROHME layout evidence boundary mismatch")
    relation = crohme["crohme_noncommercial"]["relation_graph"]
    crohme_delta = None
    if "crohme_layout_baseline_evaluation" in paths:
        crohme_baseline = _json(paths["crohme_layout_baseline_evaluation"])
        if (
            crohme_baseline.get("schema") != "aiflow-formula-layout-evaluation/v1"
            or crohme_baseline.get("crohme_noncommercial", {}).get(
                "product_validation"
            ) is not False
        ):
            raise ValueError("CROHME layout baseline evidence boundary mismatch")
        baseline_relation = crohme_baseline["crohme_noncommercial"]["relation_graph"]
        crohme_delta = {
            "relation_formula_exact_count": (
                int(relation["formula_exact_count"])
                - int(baseline_relation["formula_exact_count"])
            ),
            "relation_and_context_character_formula_exact_count": (
                int(relation["relation_and_character_formula_exact"][
                    "context_finalized_count"
                ])
                - int(baseline_relation["relation_and_character_formula_exact"][
                    "context_finalized_count"
                ])
            ),
            "relation_micro_f1": (
                float(relation["micro"]["f1"])
                - float(baseline_relation["micro"]["f1"])
            ),
        }
        if any(value < 0 for value in crohme_delta.values()):
            raise ValueError(f"CROHME wrapper regression: {crohme_delta}")

    return {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "development_only_posthoc_shadow",
        "decision": "retain_opt_in_shadow_only_not_product_default",
        "project_truth_inventory": truth_inventory,
        "upstream_invariance": {
            "formulas": len(formula_ids),
            "mutated_formulas": len(mutations),
            "character_group_order_acceptance_unchanged": not mutations,
            "baseline_default_emitted_layout": False,
        },
        "project_formula_latex_exact": {
            "flat_finalized_tokens": flat_score,
            "layout_shadow": layout_score,
            "net_exact_gain": layout_score["exact_count"] - flat_score["exact_count"],
            "improved_formulas": improved,
            "regressed_formulas": regressed,
            "changed_formulas": changed,
        },
        "project_acceptance_boundary": {
            "auto_accepted": accepted_score,
            "review_required": review_score,
            "raw_fallback_formula_ids": review_ids,
        },
        "layout_runtime_audit": {
            "candidate_extensions": sum(
                int(audit["candidate_extensions"]) for audit in layout_audits
            ),
            "selected_semantic_promotions": sum(
                int(audit["selected_semantic_promotions"]) for audit in layout_audits
            ),
            "feedback_into_character_model": False,
            "product_default_enabled": False,
        },
        "crohme_repeated_noncommercial_layout_evidence": {
            "formulas": int(relation["formulas"]),
            "relation_formula_exact_count": int(relation["formula_exact_count"]),
            "relation_formula_exact": float(relation["formula_exact"]),
            "relation_and_context_character_formula_exact_count": int(
                relation["relation_and_character_formula_exact"]["context_finalized_count"]
            ),
            "relation_micro": relation["micro"],
            "license_boundary": relation["license_boundary"],
        },
        **(
            {"crohme_wrapper_delta_vs_unpromoted_baseline": crohme_delta}
            if crohme_delta is not None else {}
        ),
        "contracts": {
            "truth_not_read_by_runtime": True,
            "upstream_finalized_candidates_only": True,
            "grouping_mutations": 0,
            "character_feedback": False,
            "review_required_uses_raw_fallback": True,
            "training_performed_in_this_loop": False,
        },
        "sources": {
            key: {"path": str(path), "sha256": _sha256(path)}
            for key, path in paths.items()
        },
        "limits": [
            "project 159-formula result is posthoc development evidence",
            "CROHME is noncommercial repeated diagnostic evidence only",
            "fixed replay ownership/canonical content is not an independent set",
            "fresh commercial writer/formula-disjoint acceptance is required before promotion",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--crohme-layout-evaluation", type=Path, required=True)
    parser.add_argument("--crohme-layout-baseline-evaluation", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.drive.upper() != "D:" or output.exists():
        parser.error("--output must be a new file on D:")
    report = evaluate(
        args.truth, args.baseline, args.layout, args.crohme_layout_evaluation,
        args.crohme_layout_baseline_evaluation,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps({
        "output": str(output),
        "sha256": _sha256(output),
        "flat_exact": report["project_formula_latex_exact"]["flat_finalized_tokens"]["exact_count"],
        "layout_exact": report["project_formula_latex_exact"]["layout_shadow"]["exact_count"],
        "accepted_exact": report["project_acceptance_boundary"]["auto_accepted"]["exact_count"],
        "accepted_formulas": report["project_acceptance_boundary"]["auto_accepted"]["formulas"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
