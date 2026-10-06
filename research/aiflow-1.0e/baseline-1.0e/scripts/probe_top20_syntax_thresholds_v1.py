#!/usr/bin/env python3
"""Probe Top-20 numeric-syntax thresholds without producing runtime output."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from character_tensor_v1 import _json_lines
from evaluate_48hz_prefix_v1 import _sha256
from evaluate_wide_candidate_syntax_rescue_v1 import _scope
from wide_candidate_syntax_rescue_v1 import (
    DEFAULT_CONFIGURATION,
    _baseline_map,
    _formulae,
    _unique_repair,
    _wide_map,
)


SCHEMA = "aiflow-top20-syntax-threshold-probe/v1"
RATIOS = (0.01, 0.005, 0.001, 0.0001, 0.00001, 0.000001)


def _prefix_audit(top10_rows: list[dict], top20_rows: list[dict]) -> dict:
    top10 = {str(row["record_id"]): row for row in top10_rows}
    top20 = {str(row["record_id"]): row for row in top20_rows}
    if len(top10) != len(top10_rows) or len(top20) != len(top20_rows):
        raise ValueError("candidate IDs are not unique")
    if set(top10) != set(top20):
        raise ValueError("Top-10 and Top-20 coverage differ")
    token_changes = 0
    maximum_probability_difference = 0.0
    for record_id, narrow in top10.items():
        wide = top20[record_id]
        tokens = [str(value) for value in narrow["final_topk"]]
        probabilities = [float(value) for value in narrow["final_topk_probabilities"]]
        if len(tokens) != 10 or len(wide["final_topk"]) != 20:
            raise ValueError(f"candidate widths are invalid: {record_id}")
        token_changes += tokens != [str(value) for value in wide["final_topk"][:10]]
        maximum_probability_difference = max(
            maximum_probability_difference,
            max(
                abs(left - float(right))
                for left, right in zip(
                    probabilities, wide["final_topk_probabilities"][:10], strict=True,
                )
            ),
        )
    return {
        "records": len(top10),
        "top10_prefix_token_changes": token_changes,
        "top10_prefix_max_probability_difference": maximum_probability_difference,
    }


def _apply_probe(
    baseline_rows: list[dict], candidate_rows: list[dict], ratio: float,
) -> tuple[list[dict], dict]:
    configuration = {
        **DEFAULT_CONFIGURATION,
        "candidate_width": 20,
        "minimum_digit_probability_ratio": float(ratio),
    }
    baseline = _baseline_map(baseline_rows)
    wide = _wide_map(candidate_rows, configuration)
    if not baseline or set(baseline) != set(wide):
        raise ValueError("Top-20 syntax probe coverage mismatch")
    selected = {
        record_id: str(row["finalized_top1"])
        for record_id, row in baseline.items()
    }
    changes = []
    skipped = Counter()
    for formula_id, sequence in _formulae(baseline).items():
        record_ids = [str(row["record_id"]) for row in sequence]
        wide_sequence = [wide[record_id] for record_id in record_ids]
        before = [str(row["finalized_top1"]) for row in sequence]
        repaired, reason = _unique_repair(before, wide_sequence, configuration)
        if repaired is None:
            skipped[reason] += 1
            continue
        for index, (left, right) in enumerate(zip(before, repaired, strict=True)):
            if left == right:
                continue
            record_id = record_ids[index]
            candidate = wide_sequence[index]
            selected[record_id] = right
            candidate_index = candidate["tokens"].index(right)
            changes.append({
                "formula_id": formula_id,
                "record_id": record_id,
                "context_index": index,
                "before": left,
                "after": right,
                "candidate_rank": candidate_index + 1,
                "candidate_probability": candidate["probabilities"][candidate_index],
            })
    output = [
        {**row, "finalized_top1": selected[str(row["record_id"])]}
        for row in baseline_rows
    ]
    return output, {
        "configuration": configuration,
        "changed_glyphs": len(changes),
        "changed_formulas": len({change["formula_id"] for change in changes}),
        "changes": changes,
        "skipped": dict(sorted(skipped.items())),
    }


def _compact(metrics: dict) -> dict:
    keys = (
        "baseline_character_top1_count", "challenger_character_top1_count",
        "glyph_improved", "glyph_regressed",
        "baseline_formula_exact_count", "challenger_formula_exact_count",
        "formula_improved", "formula_regressed",
    )
    return {key: metrics[key] for key in keys}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-baseline", type=Path, required=True)
    parser.add_argument("--direct-top10", type=Path, required=True)
    parser.add_argument("--direct-top20", type=Path, required=True)
    parser.add_argument("--crohme-baseline", type=Path, required=True)
    parser.add_argument("--crohme-top10", type=Path, required=True)
    parser.add_argument("--crohme-top20", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {
        name: value.expanduser().resolve()
        for name, value in vars(args).items()
    }
    if not all(path.is_file() for name, path in paths.items() if name != "output"):
        parser.error("one or more Top-20 probe inputs are missing")
    if paths["output"].exists():
        parser.error(f"refusing to overwrite probe output: {paths['output']}")
    direct_baseline = list(_json_lines(paths["direct_baseline"]))
    direct_top10 = list(_json_lines(paths["direct_top10"]))
    direct_top20 = list(_json_lines(paths["direct_top20"]))
    crohme_baseline = list(_json_lines(paths["crohme_baseline"]))
    crohme_top10 = list(_json_lines(paths["crohme_top10"]))
    crohme_top20 = list(_json_lines(paths["crohme_top20"]))
    variants = []
    for ratio in RATIOS:
        direct_rows, direct_audit = _apply_probe(
            direct_baseline, direct_top20, ratio,
        )
        crohme_rows, crohme_audit = _apply_probe(
            crohme_baseline, crohme_top20, ratio,
        )
        direct_metrics = _scope(direct_top20, direct_baseline, direct_rows)
        crohme_metrics = _scope(crohme_top20, crohme_baseline, crohme_rows)
        variants.append({
            "minimum_digit_probability_ratio": ratio,
            "runtime_admitted": False,
            "zero_regression_probe": bool(
                direct_metrics["glyph_regressed"] == 0
                and direct_metrics["formula_regressed"] == 0
                and crohme_metrics["glyph_regressed"] == 0
                and crohme_metrics["formula_regressed"] == 0
            ),
            "direct": {"metrics": direct_metrics, "audit": direct_audit},
            "crohme": {"metrics": crohme_metrics, "audit": crohme_audit},
        })
    report = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "training_performed": False,
        "runtime_admitted": False,
        "candidate_status": "research_recall_and_threshold_probe_only",
        "prefix_audit": {
            "direct": _prefix_audit(direct_top10, direct_top20),
            "crohme": _prefix_audit(crohme_top10, crohme_top20),
        },
        "variants": variants,
        "inputs": {
            name: _sha256(path)
            for name, path in paths.items() if name != "output"
        },
        "limits": [
            "the direct formulas are repeatedly observed development evidence",
            "CROHME is repeated noncommercial diagnostic evidence",
            "no threshold in this report authorizes runtime promotion",
        ],
    }
    paths["output"].parent.mkdir(parents=True, exist_ok=True)
    paths["output"].write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    print(json.dumps({
        "output": str(paths["output"]),
        "sha256": _sha256(paths["output"]),
        "runtime_admitted": False,
        "prefix_audit": report["prefix_audit"],
        "variants": [
            {
                "minimum_digit_probability_ratio": variant[
                    "minimum_digit_probability_ratio"
                ],
                "zero_regression_probe": variant["zero_regression_probe"],
                "direct": _compact(variant["direct"]["metrics"]),
                "crohme": _compact(variant["crohme"]["metrics"]),
            }
            for variant in variants
        ],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
