#!/usr/bin/env python3
"""Posthoc sensitivity audit for the frozen Top-5 decoder beam width.

This is a cached-prediction shadow comparison only. It does not train, tune a
release threshold, touch CROHME, or change the product decoder default.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from audit_decoder_failure_beam_v1 import _decoder_rows, _jsonl
from selective_decoder_v1 import DEFAULT_TOKEN_BEAM, decode_selective_partition


SCHEMA = "aiflow-hwr-decoder-beam-width-sensitivity/v1"
WIDTHS = (DEFAULT_TOKEN_BEAM, 64, 128)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(traces_path: Path, shadow_path: Path) -> dict[str, Any]:
    traces = _jsonl(traces_path)
    shadow = json.loads(shadow_path.read_text(encoding="utf-8"))
    if shadow.get("evaluation_status", {}).get(
        "independent_acceptance_claim_allowed"
    ) is not False:
        raise ValueError("shadow input lacks posthoc/non-acceptance guard")
    shadow_rows = {str(row["sample_id"]): row for row in shadow["formula_level"]}
    if len(shadow_rows) != len(traces):
        raise AssertionError("trace/shadow formula count mismatch")

    records = []
    width_counts = {width: Counter() for width in WIDTHS}
    width_ids = {width: {"exact_oracle_groups": [], "exact_fast_groups": []}
                 for width in WIDTHS}
    parity_count = 0
    for trace in traces:
        sample_id = str(trace["sample_id"])
        groups, symbols = _decoder_rows(trace)
        targets = [str(value) for value in trace["source"]["target_tokens"]]
        fast_group_exact = bool(
            shadow_rows[sample_id]["arms"]["fast"]["group_exact"]
        )
        row = {
            "sample_id": sample_id,
            "fast_group_exact": fast_group_exact,
            "target_tokens": targets,
            "widths": {},
        }
        for width in WIDTHS:
            decoded = decode_selective_partition(
                sample_id, groups, symbols,
                stroke_count=int(trace["source"]["stroke_count"]),
                token_beam=width,
            )
            saved = trace["oracle_group_hwr"]["decoder"]
            parity = (
                bool(decoded.get("accepted")) == bool(saved["accepted"])
                and decoded.get("tokens") == saved.get("tokens")
                and decoded.get("latex") == saved.get("latex")
            ) if width == DEFAULT_TOKEN_BEAM else None
            parity_count += int(bool(parity))
            exact = bool(decoded.get("accepted")) and decoded.get("tokens") == targets
            candidate_preserved = bool(decoded.get("top5_preserved", False))
            width_counts[width]["formulas"] += 1
            width_counts[width]["accepted"] += int(bool(decoded.get("accepted")))
            width_counts[width]["strict_exact_oracle_groups"] += int(exact)
            width_counts[width]["candidate_preservation_failures"] += int(
                not candidate_preserved
            )
            if fast_group_exact:
                width_counts[width]["fast_group_exact_formulas"] += 1
                width_counts[width]["strict_exact_fast_groups"] += int(exact)
            if exact:
                width_ids[width]["exact_oracle_groups"].append(sample_id)
                if fast_group_exact:
                    width_ids[width]["exact_fast_groups"].append(sample_id)
            row["widths"][str(width)] = {
                "accepted": bool(decoded.get("accepted")),
                "exact_with_oracle_groups": exact,
                "tokens": decoded.get("tokens"),
                "latex": decoded.get("latex"),
                "joint_score": decoded.get("joint_token_relation_score"),
                "beam32_parity_with_saved_trace": parity,
            }
        records.append(row)

    base_ids = set(width_ids[DEFAULT_TOKEN_BEAM]["exact_oracle_groups"])
    comparisons = {}
    for width in WIDTHS[1:]:
        challenger_ids = set(width_ids[width]["exact_oracle_groups"])
        comparisons[str(width)] = {
            "recovered_vs_32": sorted(challenger_ids - base_ids),
            "regressed_vs_32": sorted(base_ids - challenger_ids),
            "unchanged_exact_vs_32": len(challenger_ids & base_ids),
        }

    checks = {
        "149_trace_rows_match_shadow": len(traces) == shadow.get("formulas"),
        "beam32_matches_saved_decoder_for_all_formulas": parity_count == len(traces),
        "all_widths_accept_all_oracle_group_formulas": all(
            width_counts[width]["accepted"] == len(traces) for width in WIDTHS
        ),
        "all_widths_preserve_existing_top5_candidates": all(
            width_counts[width]["candidate_preservation_failures"] == 0
            for width in WIDTHS
        ),
        "fast_group_exact_denominator_is_constant": all(
            width_counts[width]["fast_group_exact_formulas"]
            == width_counts[DEFAULT_TOKEN_BEAM]["fast_group_exact_formulas"]
            for width in WIDTHS
        ),
        "product_default_stays_disabled": shadow.get("audit", {}).get(
            "product_default_enabled"
        ) is False,
        "crohme_training_or_tuning_stays_disabled": shadow.get("audit", {}).get(
            "crohme_training_or_tuning"
        ) is False,
    }
    if not all(checks.values()):
        raise AssertionError(f"beam-width audit failed: {checks}")

    return {
        "schema": SCHEMA,
        "scope": (
            "posthoc cached Top-5 sensitivity on the same frozen development cohort; "
            "not independent acceptance and not a selected product setting"
        ),
        "inputs": {
            "traces_path": str(traces_path),
            "traces_sha256": _sha256(traces_path),
            "shadow_path": str(shadow_path),
            "shadow_sha256": _sha256(shadow_path),
            "decoder_path": "scripts/selective_decoder_v1.py",
            "decoder_sha256": _sha256(Path(__file__).with_name("selective_decoder_v1.py")),
            "audit_script_sha256": _sha256(Path(__file__)),
        },
        "widths": list(WIDTHS),
        "summary_by_width": {
            str(width): dict(width_counts[width]) for width in WIDTHS
        },
        "comparisons_vs_32": comparisons,
        "exact_sample_ids_by_width": {
            str(width): width_ids[width] for width in WIDTHS
        },
        "verification": {"checks": checks, "all_checks_pass": all(checks.values())},
        "formula_level": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--shadow-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite output: {args.output}")
    report = audit(args.traces, args.shadow_audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "summary_by_width": report["summary_by_width"],
        "comparisons_vs_32": report["comparisons_vs_32"],
        "verification": report["verification"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
