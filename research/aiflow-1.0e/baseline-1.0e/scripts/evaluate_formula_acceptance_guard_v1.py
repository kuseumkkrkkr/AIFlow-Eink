#!/usr/bin/env python3
"""Evaluate the formula review boundary without treating abstention as Top-1."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from character_tensor_v1 import ROOT
from evaluate_48hz_prefix_v1 import _sha256
from formula_acceptance_guard_v1 import (
    AUTO_ACCEPTED, REVIEW_REQUIRED, load_configuration,
)


SCHEMA = "aiflow-formula-acceptance-guard-evaluation/v1"


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


def _fixed_protocol_ids(
    ownership_path: Path, canonical_path: Path,
    truth: dict[str, list[str]], metadata: dict[str, dict[str, Any]],
) -> tuple[list[str], list[str]]:
    ownership_rows = [
        json.loads(line) for line in ownership_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ownership_ids = sorted(
        str(row["sample_id"]) for row in ownership_rows if row.get("accepted") is True
    )
    canonical_rows = [
        json.loads(line) for line in canonical_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    canonical_ids = [str(row["record_id"]) for row in canonical_rows]
    if (
        len(ownership_ids) != 47 or len(set(ownership_ids)) != 47
        or len(canonical_ids) != 10 or len(set(canonical_ids)) != 10
    ):
        raise ValueError("fixed ownership/canonical protocol coverage mismatch")
    missing = (set(ownership_ids) | set(canonical_ids)) - set(truth)
    if missing:
        raise ValueError(f"fixed protocol ids missing from project truth: {sorted(missing)}")
    mismatched_ownership = [
        str(row["sample_id"]) for row in ownership_rows
        if row.get("accepted") is True and (
            str(row.get("writer_id")) != str(metadata[str(row["sample_id"])].get("writer_id"))
            or [str(token) for token in row.get("labels") or []]
            != truth[str(row["sample_id"])]
        )
    ]
    mismatched_canonical = [
        str(row["record_id"]) for row in canonical_rows
        if str(row.get("label")) != str(metadata[str(row["record_id"])].get("target_display"))
    ]
    if mismatched_ownership or mismatched_canonical:
        raise ValueError(
            "fixed protocol identity mismatch: "
            f"ownership={mismatched_ownership}, canonical={mismatched_canonical}"
        )
    return ownership_ids, canonical_ids


def _ink_identity(row: dict[str, Any]) -> str:
    payload = {
        key: row.get(key)
        for key in (
            "canvas", "strokes", "target_cells", "target_relations", "formula_cells",
        )
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _content_overlap(
    reference: dict[str, dict[str, Any]], comparison_path: Path,
) -> dict[str, Any]:
    comparison_rows = [
        json.loads(line) for line in comparison_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    comparison = {str(row["sample_id"]): row for row in comparison_rows}
    if len(comparison) != len(comparison_rows):
        raise ValueError("duplicate ids in comparison truth")
    by_identity: dict[str, list[str]] = {}
    for formula_id, row in comparison.items():
        by_identity.setdefault(_ink_identity(row), []).append(formula_id)
    mappings = {
        formula_id: sorted(by_identity.get(_ink_identity(row), []))
        for formula_id, row in reference.items()
    }
    ambiguous = {
        formula_id: matches for formula_id, matches in mappings.items() if len(matches) > 1
    }
    missing = sorted(formula_id for formula_id, matches in mappings.items() if not matches)
    if ambiguous or missing:
        raise ValueError(
            f"fixed replay content mapping mismatch: missing={missing}, ambiguous={ambiguous}"
        )
    one_to_one = {formula_id: matches[0] for formula_id, matches in mappings.items()}
    shared_ids = set(reference) & set(comparison)
    shared_same_content = sum(
        _ink_identity(reference[formula_id]) == _ink_identity(comparison[formula_id])
        for formula_id in shared_ids
    )
    return {
        "identity_policy": "sha256(canvas,strokes,target_cells,target_relations,formula_cells)",
        "reference_formulas": len(reference),
        "comparison_formulas": len(comparison),
        "content_matches": len(one_to_one),
        "content_match_rate": len(one_to_one) / max(len(reference), 1),
        "content_id_remaps": sum(old_id != new_id for old_id, new_id in one_to_one.items()),
        "shared_sample_ids": len(shared_ids),
        "shared_ids_same_content": shared_same_content,
        "shared_ids_reused_for_different_content": len(shared_ids) - shared_same_content,
        "mapping": dict(sorted(one_to_one.items())),
        "independent_evaluation": False if len(one_to_one) == len(reference) else None,
    }


def _runtime(payload: dict[str, Any], expected: set[str], guarded: bool) -> dict[str, dict]:
    if (
        payload.get("schema") != "aiflow-raw-formula-context-runtime/v1"
        or payload.get("status") != "development_only_posthoc_shadow"
        or payload.get("audit", {}).get("product_default_enabled") is not False
    ):
        raise ValueError("formula runtime contract mismatch")
    rows = {str(row["formula_id"]): row for row in payload.get("formulas") or []}
    if set(rows) != expected or len(rows) != len(payload.get("formulas") or []):
        raise ValueError("formula runtime coverage mismatch")
    if guarded and any(
        row.get("decision_status") not in {AUTO_ACCEPTED, REVIEW_REQUIRED}
        for row in rows.values()
    ):
        raise ValueError("guarded runtime decision coverage mismatch")
    return rows


def _score(
    rows: dict[str, dict], truth: dict[str, list[str]], formula_ids: list[str],
) -> dict[str, Any]:
    exact_ids = [
        formula_id for formula_id in formula_ids
        if rows[formula_id]["finalized_tokens"] == truth[formula_id]
    ]
    error_ids = [formula_id for formula_id in formula_ids if formula_id not in exact_ids]
    accepted = [
        formula_id for formula_id in formula_ids
        if rows[formula_id]["decision_status"] == AUTO_ACCEPTED
    ]
    reviewed = [formula_id for formula_id in formula_ids if formula_id not in accepted]
    accepted_exact = [formula_id for formula_id in accepted if formula_id in exact_ids]
    accepted_errors = [formula_id for formula_id in accepted if formula_id in error_ids]
    reviewed_errors = [formula_id for formula_id in reviewed if formula_id in error_ids]
    return {
        "formula_exact": len(exact_ids),
        "total": len(formula_ids),
        "formula_exact_rate": len(exact_ids) / max(len(formula_ids), 1),
        "auto_accepted": len(accepted),
        "auto_accept_coverage": len(accepted) / max(len(formula_ids), 1),
        "auto_accepted_exact": len(accepted_exact),
        "auto_accepted_errors": len(accepted_errors),
        "auto_accepted_precision": len(accepted_exact) / max(len(accepted), 1),
        "review_required": len(reviewed),
        "reviewed_errors": len(reviewed_errors),
        "reviewed_correct": len(reviewed) - len(reviewed_errors),
        "error_capture_rate": len(reviewed_errors) / max(len(error_ids), 1),
        "reviewed_error_ids": reviewed_errors,
        "remaining_auto_accept_error_ids": accepted_errors,
    }


def evaluate(
    truth_path: Path, baseline_path: Path, guarded_path: Path,
    configuration_path: Path, crohme_path: Path,
    project_protocol: str = "current_159",
    ownership_path: Path | None = None, canonical_path: Path | None = None,
    comparison_truth_path: Path | None = None,
) -> dict[str, Any]:
    paths = {
        "truth": _d_file(truth_path, "project truth"),
        "baseline": _d_file(baseline_path, "baseline runtime"),
        "guarded": _d_file(guarded_path, "guarded runtime"),
        "configuration": _d_file(configuration_path, "acceptance configuration"),
        "crohme": _d_file(crohme_path, "CROHME diagnostic"),
    }
    if project_protocol == "fixed_replay_110":
        if ownership_path is None or canonical_path is None:
            raise ValueError("fixed replay protocol paths are required")
        paths["ownership_protocol"] = _d_file(ownership_path, "ownership protocol")
        paths["canonical_protocol"] = _d_file(canonical_path, "canonical protocol")
    elif project_protocol != "current_159":
        raise ValueError(f"unsupported project protocol: {project_protocol}")
    if comparison_truth_path is not None:
        paths["comparison_truth"] = _d_file(comparison_truth_path, "comparison truth")
    _configuration, configuration_sha256 = load_configuration(paths["configuration"])
    truth, metadata = _truth(paths["truth"])
    baseline_payload = _json(paths["baseline"])
    guarded_payload = _json(paths["guarded"])
    baseline = _runtime(baseline_payload, set(truth), False)
    guarded = _runtime(guarded_payload, set(truth), True)
    guard_audit = dict(guarded_payload.get("audit", {}).get("formula_acceptance_guard") or {})
    if (
        guard_audit.get("schema") != "aiflow-formula-acceptance-guard/v1"
        or guard_audit.get("configuration_sha256") != configuration_sha256
        or guard_audit.get("product_default_enabled") is not False
        or guard_audit.get("token_mutations") != 0
        or guard_audit.get("grouping_mutations") != 0
        or guard_audit.get("arithmetic_evaluation") is not False
    ):
        raise ValueError("formula acceptance guard audit mismatch")
    for formula_id in truth:
        before = baseline[formula_id]
        after = guarded[formula_id]
        if (
            before["finalized_tokens"] != after["finalized_tokens"]
            or before["groups"] != after["groups"]
            or before["symbols"] != after["symbols"]
        ):
            raise ValueError(f"formula acceptance guard mutated output: {formula_id}")

    formula_ids = sorted(truth)
    if project_protocol == "current_159":
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
        expected_subset_sizes = {
            "all_159": 159, "public_web_110": 110, "owned_phone_replay_49": 49,
        }
    else:
        ownership_ids, canonical_ids = _fixed_protocol_ids(
            paths["ownership_protocol"], paths["canonical_protocol"], truth, metadata,
        )
        subsets = {
            "all_fixed_replay_110": formula_ids,
            "accepted_ownership_47": ownership_ids,
            "canonical_replay_10": canonical_ids,
        }
        expected_subset_sizes = {
            "all_fixed_replay_110": 110,
            "accepted_ownership_47": 47,
            "canonical_replay_10": 10,
        }
    if {name: len(ids) for name, ids in subsets.items()} != expected_subset_sizes:
        raise ValueError("project evaluation subset coverage mismatch")
    project_scores = {
        name: _score(guarded, truth, ids) for name, ids in subsets.items()
    }
    content_overlap = (
        _content_overlap(metadata, paths["comparison_truth"])
        if "comparison_truth" in paths else None
    )
    reason_counts = Counter(
        reason for row in guarded.values()
        for reason in row["acceptance"]["review_reasons"]
    )

    crohme = _json(paths["crohme"])
    crohme_acceptance = dict(crohme.get("formula_acceptance_guard") or {})
    if (
        crohme.get("schema") != "aiflow-formula-context-crohme-diagnostic/v1"
        or crohme.get("status") != "repeated_noncommercial_diagnostic_only"
        or crohme.get("product_validation") is not False
        or crohme_acceptance.get("enabled") is not True
        or crohme_acceptance.get("configuration_sha256") != configuration_sha256
        or crohme_acceptance.get("candidate_preserving") is not True
        or crohme_acceptance.get("token_mutations") != 0
        or crohme_acceptance.get("grouping_mutations") != 0
        or crohme_acceptance.get("arithmetic_evaluation") is not False
        or crohme_acceptance.get("product_validation") is not False
    ):
        raise ValueError("CROHME formula acceptance diagnostic mismatch")

    limits = [
        "project thresholds were inspected against the same 159 formulas",
        "selective precision is not a replacement for all-formula Top-1 exact",
        "CROHME is repeated noncommercial diagnostic evidence only",
        "fresh commercial writer/formula-disjoint acceptance is required before promotion",
    ]
    if project_protocol == "fixed_replay_110":
        limits.append("fixed replay scores are final-step exact scores and are not independent data")
    if content_overlap and content_overlap["content_match_rate"] == 1.0:
        limits.append("all fixed replay formulas are exact content duplicates of the comparison set")

    return {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "development_only_posthoc_shadow",
        "decision": "retain_as_shadow_only_not_product",
        "top1_formula_exact_unchanged": True,
        "project_selective_scores": project_scores,
        **({
            "project_protocol": project_protocol,
            "fixed_replay_content_overlap": content_overlap,
        } if project_protocol == "fixed_replay_110" else {}),
        "project_review_reasons": dict(sorted(reason_counts.items())),
        "crohme_repeated_noncommercial_diagnostic": crohme_acceptance,
        "contracts": {
            "candidate_preserving": True,
            "token_mutations": 0,
            "grouping_mutations": 0,
            "glyph_order_mutations": 0,
            "target_label_or_glyph_count_input": False,
            "writer_identity_input": False,
            "arithmetic_evaluation": False,
            "raw_fallback_on_review_required": True,
            "product_default_enabled": False,
        },
        "artifacts": {
            key: {"path": str(path), "sha256": _sha256(path)}
            for key, path in paths.items()
        },
        "limits": limits,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--guarded", type=Path, required=True)
    parser.add_argument("--configuration", type=Path, required=True)
    parser.add_argument("--crohme", type=Path, required=True)
    parser.add_argument(
        "--project-protocol", choices=("current_159", "fixed_replay_110"),
        default="current_159",
    )
    parser.add_argument(
        "--ownership", type=Path,
        default=ROOT / "hf-dataset" / "data" / "ownership_train.jsonl",
    )
    parser.add_argument(
        "--canonical", type=Path,
        default=ROOT / "artifacts" / "hwr_replay_evaluation_v1" / "direct_canonical_10_truth.jsonl",
    )
    parser.add_argument("--comparison-truth", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.drive.upper() != "D:" or output.exists():
        parser.error("--output must be a new file on D:")
    report = evaluate(
        args.truth, args.baseline, args.guarded, args.configuration, args.crohme,
        args.project_protocol, args.ownership, args.canonical, args.comparison_truth,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    print(json.dumps({
        "output": str(output), "sha256": _sha256(output),
        "decision": report["decision"],
        "project_selective_scores": report["project_selective_scores"],
        "crohme": report["crohme_repeated_noncommercial_diagnostic"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
