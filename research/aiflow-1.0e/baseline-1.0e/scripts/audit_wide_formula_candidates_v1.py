#!/usr/bin/env python3
"""Audit formula failures against a wider research-only HWR candidate cache."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

from character_tensor_v1 import _json_lines


def _index(row: dict) -> int:
    if "context_index" in row:
        return int(row["context_index"])
    return int(row.get("context", {}).get("index", -1))


def audit(candidate_rows: list[dict], runtime_rows: list[dict]) -> dict:
    candidates = {str(row["record_id"]): row for row in candidate_rows}
    if len(candidates) != len(candidate_rows):
        raise ValueError("candidate record IDs are not unique")
    formulae: dict[str, list[dict]] = defaultdict(list)
    for row in runtime_rows:
        formulae[str(row["formula_id"])].append(row)
    if set(candidates) != {str(row["record_id"]) for row in runtime_rows}:
        raise ValueError("candidate and runtime coverage differ")
    failures = []
    exact = 0
    oracle = 0
    for formula_id, unordered in sorted(formulae.items()):
        rows = sorted(unordered, key=_index)
        truth = [str(candidates[str(row["record_id"])]["label"]) for row in rows]
        prediction = [str(row["finalized_top1"]) for row in rows]
        exact += truth == prediction
        oracle += all(
            truth_token in [str(value) for value in candidates[str(row["record_id"])]["final_topk"]]
            for row, truth_token in zip(rows, truth, strict=True)
        )
        if truth == prediction:
            continue
        wrong = []
        for row, truth_token, predicted_token in zip(rows, truth, prediction, strict=True):
            if truth_token == predicted_token:
                continue
            candidate = candidates[str(row["record_id"])]
            tokens = [str(value) for value in candidate["final_topk"]]
            probabilities = [float(value) for value in candidate["final_topk_probabilities"]]
            truth_rank = tokens.index(truth_token) + 1 if truth_token in tokens else None
            wrong.append({
                "context_index": _index(row),
                "truth": truth_token,
                "prediction": predicted_token,
                "truth_rank": truth_rank,
                "truth_probability": (
                    probabilities[truth_rank - 1] if truth_rank is not None else None
                ),
                "topk": [
                    {"token": token, "probability": probability}
                    for token, probability in zip(tokens, probabilities, strict=True)
                ],
            })
        failures.append({
            "formula_id": formula_id,
            "truth": truth,
            "prediction": prediction,
            "wide_candidate_oracle": all(item["truth_rank"] is not None for item in wrong),
            "wrong": wrong,
        })
    return {
        "formulas": len(formulae),
        "runtime_formula_exact_count": exact,
        "wide_candidate_formula_oracle_count": oracle,
        "remaining_failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(
        list(_json_lines(args.candidates)), list(_json_lines(args.runtime)),
    )
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        output = args.output.expanduser().resolve()
        if output.exists():
            parser.error(f"refusing to overwrite audit: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8", newline="\n")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
