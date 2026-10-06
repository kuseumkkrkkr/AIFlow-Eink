#!/usr/bin/env python3
"""Audit the 8/p pairwise expert across project formulas and CROHME."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from evaluate_48hz_prefix_v1 import _sha256
from pairwise_shape_rescue_v1 import load_pairwise_shape_artifacts


SCHEMA = "aiflow-pairwise-shape-runtime-evaluation/v1"


def _d_file(path: Path, kind: str) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.drive.upper() != "D:" or not resolved.is_file():
        raise ValueError(f"{kind} must be an existing file on D:: {resolved}")
    return resolved


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _truth(path: Path) -> tuple[dict[str, list[str]], dict[str, dict[str, Any]]]:
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    truth = {
        str(row["sample_id"]): [str(cell["token"]) for cell in row["target_cells"]]
        for row in rows
    }
    if len(truth) != len(rows):
        raise ValueError("duplicate formula ids in project truth")
    return truth, {str(row["sample_id"]): row for row in rows}


def _runtime(payload: dict[str, Any], expected: set[str], kind: str) -> dict[str, dict]:
    if (
        payload.get("schema") != "aiflow-raw-formula-context-runtime/v1"
        or payload.get("status") != "development_only_posthoc_shadow"
        or payload.get("audit", {}).get("product_default_enabled") is not False
    ):
        raise ValueError(f"{kind} runtime contract mismatch")
    rows = {str(row["formula_id"]): row for row in payload.get("formulas", [])}
    if set(rows) != expected or len(rows) != len(payload.get("formulas", [])):
        raise ValueError(f"{kind} formula coverage mismatch")
    return rows


def _score(
    predictions: dict[str, dict], truth: dict[str, list[str]], ids: list[str],
) -> dict[str, Any]:
    hits = sum(predictions[formula_id]["finalized_tokens"] == truth[formula_id] for formula_id in ids)
    return {"exact": hits, "total": len(ids), "rate": hits / max(len(ids), 1)}


def evaluate(
    truth_path: Path, baseline_path: Path, challenger_path: Path,
    default_off_compat_path: Path,
    crohme_path: Path, expert_path: Path, configuration_path: Path,
    training_report_path: Path,
) -> dict[str, Any]:
    paths = {
        "truth": _d_file(truth_path, "project truth"),
        "baseline": _d_file(baseline_path, "baseline runtime"),
        "challenger": _d_file(challenger_path, "challenger runtime"),
        "default_off_compat": _d_file(
            default_off_compat_path, "default-off compatibility runtime",
        ),
        "crohme": _d_file(crohme_path, "CROHME diagnostic"),
        "expert": _d_file(expert_path, "pairwise expert"),
        "configuration": _d_file(configuration_path, "pairwise configuration"),
        "training_report": _d_file(training_report_path, "training report"),
    }
    training_report = _json(paths["training_report"])
    hwr_sha256 = str(
        training_report.get("artifacts", {}).get("hwr_checkpoint_sha256", "")
    )
    expert, configuration, expert_sha256, configuration_sha256 = (
        load_pairwise_shape_artifacts(
            paths["expert"], paths["configuration"],
            hwr_checkpoint_sha256=hwr_sha256,
        )
    )
    external = dict(training_report.get("fixed_collision_free_external_gate") or {})
    if (
        training_report.get("schema")
        != "aiflow-pairwise-shape-expert-training-report/v1"
        or training_report.get("status") != "development_only_posthoc_shadow"
        or external.get("records") != 18330
        or external.get("changed") != 0
        or external.get("regressed") != 0
        or training_report.get("artifacts", {}).get("expert_sha256") != expert_sha256
        or training_report.get("artifacts", {}).get("configuration_sha256")
        != configuration_sha256
    ):
        raise ValueError("fixed external gate evidence mismatch")

    truth, metadata = _truth(paths["truth"])
    baseline = _runtime(_json(paths["baseline"]), set(truth), "baseline")
    challenger = _runtime(_json(paths["challenger"]), set(truth), "challenger")
    default_off_compat = _runtime(
        _json(paths["default_off_compat"]), set(truth), "default-off compatibility",
    )
    formula_ids = sorted(truth)
    default_off_token_differences = sum(
        baseline[formula_id]["finalized_tokens"]
        != default_off_compat[formula_id]["finalized_tokens"]
        for formula_id in formula_ids
    )
    default_off_group_differences = sum(
        baseline[formula_id]["groups"] != default_off_compat[formula_id]["groups"]
        for formula_id in formula_ids
    )
    if (
        default_off_token_differences
        or default_off_group_differences
        or any(
            row["audit"]["pairwise_shape_rescue"].get("enabled") is not False
            for row in default_off_compat.values()
        )
    ):
        raise ValueError("pairwise default-off compatibility failed")
    changed = []
    improved = []
    regressed = []
    changes = []
    for formula_id in formula_ids:
        before = baseline[formula_id]
        after = challenger[formula_id]
        if before["groups"] != after["groups"]:
            raise ValueError(f"pairwise rescue mutated grouping: {formula_id}")
        if before["finalized_tokens"] == after["finalized_tokens"]:
            if after["audit"]["pairwise_shape_rescue"].get("changed") != 0:
                raise ValueError(f"spurious pairwise change audit: {formula_id}")
            continue
        changed.append(formula_id)
        before_exact = before["finalized_tokens"] == truth[formula_id]
        after_exact = after["finalized_tokens"] == truth[formula_id]
        if after_exact and not before_exact:
            improved.append(formula_id)
        if before_exact and not after_exact:
            regressed.append(formula_id)
        differing = [
            index for index, (left, right) in enumerate(zip(
                before["finalized_tokens"], after["finalized_tokens"], strict=True,
            )) if left != right
        ]
        audit = after["audit"]["pairwise_shape_rescue"]
        changed_position = differing[0] if len(differing) == 1 else None
        if (
            changed_position is None
            or before["finalized_tokens"][changed_position] != "p"
            or after["finalized_tokens"][changed_position] != "8"
            or audit.get("changed") != 1
            or audit.get("candidate_preserving") is not True
            or audit.get("insertions_or_deletions") != 0
            or audit.get("glyph_order_mutations") != 0
        ):
            raise ValueError(f"unexpected pairwise formula mutation: {formula_id}")
        changes.append({
            "formula_id": formula_id,
            "truth": truth[formula_id],
            "before": before["finalized_tokens"],
            "after": after["finalized_tokens"],
            "audit": audit["changes"],
        })
    if regressed or len(changed) != 1 or changed != improved:
        raise ValueError("project formula pairwise gate is not a one-gain zero-regression change")
    for formula_id, result in challenger.items():
        audit = result["audit"]
        if (
            audit.get("all_strokes_exactly_once") is not True
            or audit.get("target_label_or_glyph_count_input") is not False
            or audit.get("inserted_or_deleted_glyphs") != 0
            or audit.get("arithmetic_evaluation") is not False
            or audit.get("product_default_enabled") is not False
        ):
            raise ValueError(f"challenger runtime safety contract failed: {formula_id}")

    crohme = _json(paths["crohme"])
    crohme_scores = dict(crohme.get("scores") or {})
    pairwise_crohme = dict(crohme.get("pairwise_shape_rescue") or {})
    if (
        crohme.get("schema") != "aiflow-formula-context-crohme-diagnostic/v1"
        or crohme.get("status") != "repeated_noncommercial_diagnostic_only"
        or crohme.get("coverage", {}).get("eligible_formulas") != 722
        or pairwise_crohme.get("enabled") is not True
        or pairwise_crohme.get("changed_formulas") != 0
        or crohme_scores.get("grouping_exact_before") != 331
        or crohme_scores.get("grouping_exact_after") != 331
        or crohme_scores.get("formula_exact_before") != 77
        or crohme_scores.get("formula_exact_after") != 77
        or crohme_scores.get("grouping_regressed")
        or crohme_scores.get("formula_regressed")
    ):
        raise ValueError("CROHME pairwise no-regression gate failed")

    subsets = {
        "all_159": formula_ids,
        "public_web_110": [
            formula_id for formula_id in formula_ids
            if metadata[formula_id].get("source_partition") == "public-web-collection"
        ],
        "owned_phone_replay_49": [
            formula_id for formula_id in formula_ids
            if metadata[formula_id].get("source_partition") == "owned-phone-replay"
        ],
    }
    return {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "development_only_posthoc_shadow",
        "decision": "retain_as_shadow_only_not_product",
        "scores": {
            name: {
                "baseline": _score(baseline, truth, ids),
                "challenger": _score(challenger, truth, ids),
            }
            for name, ids in subsets.items()
        },
        "project_formula_changes": {
            "changed_formula_ids": changed,
            "improved_formula_ids": improved,
            "regressed_formula_ids": regressed,
            "changes": changes,
        },
        "fixed_external_collision_free_gate": external,
        "default_off_compatibility": {
            "formulas": len(formula_ids),
            "finalized_token_differences": default_off_token_differences,
            "group_differences": default_off_group_differences,
            "pairwise_shape_enabled": False,
        },
        "crohme_repeated_noncommercial_diagnostic": {
            "coverage": crohme["coverage"],
            "scores": crohme_scores,
            "pairwise_shape_rescue": pairwise_crohme,
            "product_validation": False,
        },
        "model": {
            "negative_class": expert["negative_class"],
            "positive_class": expert["positive_class"],
            "feature_dimension": expert["feature_dimension"],
            "gate": configuration["gate"],
        },
        "contracts": {
            "candidate_preserving": True,
            "all_strokes_exactly_once": True,
            "insertions_or_deletions": 0,
            "glyph_order_mutations": 0,
            "target_label_or_glyph_count_input": False,
            "writer_identity_input": False,
            "arithmetic_evaluation": False,
            "product_default_enabled": False,
        },
        "artifacts": {
            key: {"path": str(path), "sha256": _sha256(path)}
            for key, path in paths.items()
        },
        "limits": [
            "the fixed external gate changed zero rows and has no positive external coverage",
            "the threshold was inspected against the current 159 formulas",
            "CROHME is repeated noncommercial diagnostic evidence only",
            "fresh commercial writer/formula-disjoint 8 and p acceptance is required before promotion",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--challenger", type=Path, required=True)
    parser.add_argument("--default-off-compat", type=Path, required=True)
    parser.add_argument("--crohme", type=Path, required=True)
    parser.add_argument("--expert", type=Path, required=True)
    parser.add_argument("--configuration", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.drive.upper() != "D:" or output.exists():
        parser.error("--output must be a new file on D:")
    report = evaluate(
        args.truth, args.baseline, args.challenger, args.default_off_compat,
        args.crohme,
        args.expert, args.configuration, args.training_report,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    print(json.dumps({
        "output": str(output), "sha256": _sha256(output),
        "decision": report["decision"], "scores": report["scores"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
