#!/usr/bin/env python3
"""Probe two placement-only formula slot rescue hypotheses."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from character_tensor_v1 import _json_lines
from evaluate_48hz_prefix_v1 import _sha256
from evaluate_wide_candidate_syntax_rescue_v1 import _scope
from wide_candidate_syntax_rescue_v1 import BINARY, DIGITS, valid_numeric_sequence


SCHEMA = "aiflow-formula-slot-rescue-hypothesis-probe/v1"
STRAIGHT_ONE_AMBIGUOUS = frozenset({"I", "/", "|", r"\mid", r"\prime", "l"})


def _formulae(rows: list[dict]) -> dict[str, list[dict]]:
    output: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        output[str(row["formula_id"])].append(row)
    for formula_id, sequence in output.items():
        sequence.sort(key=lambda row: int(row["context_index"]))
        if [int(row["context_index"]) for row in sequence] != list(range(len(sequence))):
            raise ValueError(f"formula order is invalid: {formula_id}")
    return dict(output)


def _candidate_map(rows: list[dict]) -> dict[str, dict]:
    output = {str(row["record_id"]): row for row in rows}
    if len(output) != len(rows) or any(len(row.get("final_topk", [])) != 20 for row in rows):
        raise ValueError("Top-20 candidate coverage is invalid")
    return output


def _cross_geometry(candidate: dict) -> bool:
    geometry = candidate.get("geometry", {})
    try:
        return bool(
            int(round(float(geometry["stroke_count"]))) == 2
            and abs(float(geometry["aspect_log"])) <= 0.3
            and 1.2 <= float(geometry["path_over_diag"]) <= 1.8
        )
    except (KeyError, TypeError, ValueError):
        return False


def _straight_geometry(candidate: dict) -> bool:
    geometry = candidate.get("geometry", {})
    try:
        return bool(
            int(round(float(geometry["stroke_count"]))) == 1
            and float(geometry["aspect_log"]) <= -0.8
            and float(geometry["path_over_diag"]) <= 1.1
            and abs(float(geometry["direction_y"])) >= 0.9
            and float(geometry["width_rel"]) <= 0.25
        )
    except (KeyError, TypeError, ValueError):
        return False


def _best_digit_is_one(candidate: dict) -> bool:
    ranked = [
        (float(probability), str(token))
        for token, probability in zip(
            candidate["final_topk"], candidate["final_topk_probabilities"], strict=True,
        )
        if str(token) in DIGITS
    ]
    if not ranked:
        return False
    ranked.sort(reverse=True)
    if ranked[0][1] != "1":
        return False
    return len(ranked) == 1 or ranked[0][0] >= 10.0 * max(ranked[1][0], 1e-12)


def _operator_cross(sequence: list[dict], candidates: dict[str, dict]) -> dict[str, str]:
    before = [str(row["finalized_top1"]) for row in sequence]
    if len(before) != 3 or before[0] not in DIGITS or before[2] not in DIGITS:
        return {}
    if before[1] != r"\bot":
        return {}
    row = sequence[1]
    candidate = candidates[str(row["record_id"])]
    topk = [str(token) for token in candidate["final_topk"]]
    arithmetic = [token for token in topk if token in BINARY]
    if topk[:2] != [r"\bot", r"\perp"] or arithmetic != ["+"]:
        return {}
    if not _cross_geometry(candidate):
        return {}
    repaired = [before[0], "+", before[2]]
    if not valid_numeric_sequence(repaired):
        return {}
    return {str(row["record_id"]): "+"}


def _dual_straight_one(sequence: list[dict], candidates: dict[str, dict]) -> dict[str, str]:
    before = [str(row["finalized_top1"]) for row in sequence]
    if (
        len(before) != 5
        or before[0] not in DIGITS
        or before[1] != r"\div"
        or before[2] not in STRAIGHT_ONE_AMBIGUOUS
        or before[3] != "="
        or before[4] not in STRAIGHT_ONE_AMBIGUOUS
    ):
        return {}
    output = {}
    for index in (2, 4):
        row = sequence[index]
        candidate = candidates[str(row["record_id"])]
        topk = [str(value) for value in candidate["final_topk"]]
        if (
            before[index] not in topk
            or "1" not in topk
            or not _best_digit_is_one(candidate)
            or not _straight_geometry(candidate)
        ):
            return {}
        output[str(row["record_id"])] = "1"
    repaired = [before[0], before[1], "1", before[3], "1"]
    return output if valid_numeric_sequence(repaired) else {}


def _apply(
    baseline_rows: list[dict], candidate_rows: list[dict], rules: tuple[str, ...],
) -> tuple[list[dict], dict]:
    candidates = _candidate_map(candidate_rows)
    if set(candidates) != {str(row["record_id"]) for row in baseline_rows}:
        raise ValueError("formula slot probe coverage mismatch")
    selected = {
        str(row["record_id"]): str(row["finalized_top1"])
        for row in baseline_rows
    }
    changes = []
    for formula_id, sequence in _formulae(baseline_rows).items():
        proposals = {}
        for rule in rules:
            proposal = (
                _operator_cross(sequence, candidates)
                if rule == "operator_cross" else
                _dual_straight_one(sequence, candidates)
            )
            for record_id, token in proposal.items():
                if record_id in proposals and proposals[record_id] != token:
                    raise ValueError(f"formula slot hypotheses conflict: {formula_id}")
                proposals[record_id] = token
        for record_id, token in proposals.items():
            before = selected[record_id]
            selected[record_id] = token
            candidate = candidates[record_id]
            candidate_index = [str(value) for value in candidate["final_topk"]].index(token)
            changes.append({
                "formula_id": formula_id, "record_id": record_id,
                "rule": next(
                    rule for rule in rules
                    if record_id in (
                        _operator_cross(sequence, candidates)
                        if rule == "operator_cross" else
                        _dual_straight_one(sequence, candidates)
                    )
                ),
                "before": before, "after": token,
                "candidate_rank": candidate_index + 1,
                "candidate_probability": float(
                    candidate["final_topk_probabilities"][candidate_index]
                ),
            })
    return [
        {**row, "finalized_top1": selected[str(row["record_id"])]}
        for row in baseline_rows
    ], {"changes": changes, "changed_glyphs": len(changes)}


def _compact(metrics: dict) -> dict:
    return {
        key: metrics[key] for key in (
            "baseline_character_top1_count", "challenger_character_top1_count",
            "glyph_improved", "glyph_regressed",
            "baseline_formula_exact_count", "challenger_formula_exact_count",
            "formula_improved", "formula_regressed",
        )
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-baseline", type=Path, required=True)
    parser.add_argument("--direct-top20", type=Path, required=True)
    parser.add_argument("--crohme-baseline", type=Path, required=True)
    parser.add_argument("--crohme-top20", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {name: value.expanduser().resolve() for name, value in vars(args).items()}
    if not all(path.is_file() for name, path in paths.items() if name != "output"):
        parser.error("one or more formula slot probe inputs are missing")
    if paths["output"].exists():
        parser.error(f"refusing to overwrite probe output: {paths['output']}")
    direct_baseline = list(_json_lines(paths["direct_baseline"]))
    direct_top20 = list(_json_lines(paths["direct_top20"]))
    crohme_baseline = list(_json_lines(paths["crohme_baseline"]))
    crohme_top20 = list(_json_lines(paths["crohme_top20"]))
    variants = []
    for name, rules in (
        ("operator_cross", ("operator_cross",)),
        ("dual_straight_one", ("dual_straight_one",)),
        ("combined", ("operator_cross", "dual_straight_one")),
    ):
        direct_rows, direct_audit = _apply(direct_baseline, direct_top20, rules)
        crohme_rows, crohme_audit = _apply(crohme_baseline, crohme_top20, rules)
        variants.append({
            "name": name, "runtime_admitted": False,
            "direct": {
                "metrics": _scope(direct_top20, direct_baseline, direct_rows),
                "audit": direct_audit,
            },
            "crohme": {
                "metrics": _scope(crohme_top20, crohme_baseline, crohme_rows),
                "audit": crohme_audit,
            },
        })
    report = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "training_performed": False,
        "runtime_admitted": False,
        "variants": variants,
        "inputs": {
            name: _sha256(path) for name, path in paths.items() if name != "output"
        },
        "limits": [
            "direct formulas are repeatedly observed development evidence",
            "CROHME is repeated noncommercial diagnostic evidence",
            "the probe cannot authorize runtime promotion",
        ],
    }
    paths["output"].parent.mkdir(parents=True, exist_ok=True)
    paths["output"].write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    print(json.dumps({
        "output": str(paths["output"]), "sha256": _sha256(paths["output"]),
        "variants": [
            {
                "name": variant["name"],
                "direct": _compact(variant["direct"]["metrics"]),
                "crohme": _compact(variant["crohme"]["metrics"]),
                "direct_changes": variant["direct"]["audit"]["changes"],
                "crohme_changes": len(variant["crohme"]["audit"]["changes"]),
            }
            for variant in variants
        ],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
