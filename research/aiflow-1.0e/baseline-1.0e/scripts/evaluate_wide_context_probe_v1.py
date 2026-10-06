#!/usr/bin/env python3
"""Probe the frozen owned formula-context network over auxiliary Top-k rows.

This is evaluation only.  It does not train, write a runtime configuration, or
authorize candidate selection.  Formula order and relations come from the
current baseline path; the wide cache contributes record-local candidates.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import torch

from character_tensor_v1 import _json_lines
from evaluate_48hz_prefix_v1 import DEFAULT_PRODUCT, _sha256
from train_owned_formula_context_v1 import (
    decide_owned_formula_rows,
    load_owned_formula_context,
)


SCHEMA = "aiflow-wide-context-probe/v1"
DEFAULT_CONTEXT = (
    Path(__file__).resolve().parents[1]
    / "artifacts" / "owned_formula_context_20260822_r6"
    / "owned_formula_context_product.pt"
)


def _baseline_formulae(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["formula_id"])].append(row)
    for formula_id, sequence in grouped.items():
        sequence.sort(key=lambda row: int(row["context_index"]))
        if [int(row["context_index"]) for row in sequence] != list(range(len(sequence))):
            raise ValueError(f"baseline context indices are not contiguous: {formula_id}")
    return dict(grouped)


def _runtime_rows(baseline_rows: list[dict], candidate_rows: list[dict]) -> tuple[list[dict], dict]:
    candidates = {str(row["record_id"]): row for row in candidate_rows}
    if len(candidates) != len(candidate_rows):
        raise ValueError("wide candidate IDs are not unique")
    if set(candidates) != {str(row["record_id"]) for row in baseline_rows}:
        raise ValueError("wide context probe coverage mismatch")
    output = []
    baseline_context = {}
    baseline_missing = []
    for formula_id, sequence in _baseline_formulae(baseline_rows).items():
        length = len(sequence)
        for index, baseline in enumerate(sequence):
            record_id = str(baseline["record_id"])
            source = candidates[record_id]
            if str(source["formula_id"]) != formula_id:
                raise ValueError(f"wide candidate formula mismatch: {record_id}")
            tokens = [str(value) for value in source["final_topk"]]
            probabilities = [float(value) for value in source["final_topk_probabilities"]]
            if not 1 <= len(tokens) <= 20 or len(tokens) != len(set(tokens)):
                raise ValueError(f"wide candidate width is invalid: {record_id}")
            if (
                len(probabilities) != len(tokens)
                or any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in probabilities)
                or any(left < right for left, right in zip(probabilities, probabilities[1:]))
            ):
                raise ValueError(f"wide candidate probability coverage is invalid: {record_id}")
            baseline_token = str(baseline["finalized_top1"])
            baseline_context[record_id] = baseline_token
            if baseline_token not in tokens:
                baseline_missing.append(record_id)
            output.append({
                **source,
                "record_id": record_id,
                "formula_id": formula_id,
                "final_topk": tokens,
                "final_topk_probabilities": probabilities,
                "context": {
                    "index": index,
                    "length": length,
                    "relation_from_previous": baseline.get(
                        "layout_relation_from_previous"
                    ),
                },
            })
    return output, {
        "baseline_context_tokens": baseline_context,
        "baseline_tokens_outside_wide_candidates": baseline_missing,
    }


def _metrics(
    rows: list[dict], baseline_rows: list[dict], predictions: dict[str, str],
) -> dict:
    truth = {str(row["record_id"]): str(row["label"]) for row in rows}
    baseline = {
        str(row["record_id"]): str(row["finalized_top1"])
        for row in baseline_rows
    }
    formula_by_record = {
        str(row["record_id"]): str(row["formula_id"]) for row in baseline_rows
    }
    if set(truth) != set(baseline) or set(truth) != set(predictions):
        raise ValueError("wide context probe metric coverage mismatch")
    formulae = _baseline_formulae(baseline_rows)
    baseline_char = sum(baseline[key] == truth[key] for key in truth)
    predicted_char = sum(predictions[key] == truth[key] for key in truth)
    baseline_formula = predicted_formula = formula_improved = formula_regressed = 0
    outcomes = []
    for formula_id, sequence in formulae.items():
        record_ids = [str(row["record_id"]) for row in sequence]
        before_ok = all(baseline[key] == truth[key] for key in record_ids)
        after_ok = all(predictions[key] == truth[key] for key in record_ids)
        baseline_formula += before_ok
        predicted_formula += after_ok
        formula_improved += not before_ok and after_ok
        formula_regressed += before_ok and not after_ok
        if before_ok != after_ok:
            outcomes.append({
                "formula_id": formula_id,
                "direction": "improved" if after_ok else "regressed",
                "truth": [truth[key] for key in record_ids],
                "before": [baseline[key] for key in record_ids],
                "after": [predictions[key] for key in record_ids],
            })
    changes = [
        {
            "record_id": key,
            "formula_id": formula_by_record[key],
            "direction": (
                "improved" if predictions[key] == truth[key]
                else "regressed" if baseline[key] == truth[key]
                else "changed_still_wrong"
            ),
            "truth": truth[key], "before": baseline[key], "after": predictions[key],
        }
        for key in truth if predictions[key] != baseline[key]
    ]
    return {
        "records": len(truth), "formulas": len(formulae),
        "baseline_character_top1_count": baseline_char,
        "challenger_character_top1_count": predicted_char,
        "challenger_character_top1": predicted_char / len(truth),
        "glyph_improved": sum(change["direction"] == "improved" for change in changes),
        "glyph_regressed": sum(change["direction"] == "regressed" for change in changes),
        "changed_still_wrong": sum(
            change["direction"] == "changed_still_wrong" for change in changes
        ),
        "changed_glyphs": len(changes),
        "baseline_formula_exact_count": baseline_formula,
        "challenger_formula_exact_count": predicted_formula,
        "challenger_formula_exact": predicted_formula / len(formulae),
        "formula_improved": formula_improved,
        "formula_regressed": formula_regressed,
        "formula_outcomes": outcomes,
        "changes": changes,
    }


def _restricted_gate(
    baseline_rows: list[dict], predictions: dict[str, str],
    before: str, after: str,
) -> dict[str, str]:
    output = {
        str(row["record_id"]): str(row["finalized_top1"])
        for row in baseline_rows
    }
    for record_id, prediction in predictions.items():
        if output[record_id] == before and prediction == after:
            output[record_id] = after
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--wide-candidates", type=Path)
    parser.add_argument("--context-checkpoint", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--hwr-checkpoint", type=Path, default=DEFAULT_PRODUCT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if any(value is None for value in (args.baseline, args.wide_candidates, args.output)):
        parser.error("baseline, wide candidates, and output are required")
    paths = [
        value.expanduser().resolve() for value in (
            args.baseline, args.wide_candidates, args.context_checkpoint,
            args.hwr_checkpoint, args.output,
        )
    ]
    if not all(path.is_file() for path in paths[:4]):
        parser.error("wide context probe input is missing")
    if paths[4].exists():
        parser.error(f"refusing to overwrite probe output: {paths[4]}")
    if args.batch_size < 1:
        parser.error("batch size must be positive")
    device_name = (
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    if device_name == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    device = torch.device(device_name)
    baseline_rows = list(_json_lines(paths[0]))
    candidate_rows = list(_json_lines(paths[1]))
    runtime_rows, preparation = _runtime_rows(baseline_rows, candidate_rows)
    model, contract, payload = load_owned_formula_context(
        paths[2], paths[3], device, require_d_drive=False,
    )
    variants = {}
    predictions, audit = decide_owned_formula_rows(
        model, contract, payload, runtime_rows, device, args.batch_size,
    )
    variants["auxiliary_top1_context"] = {
        "metrics": _metrics(runtime_rows, baseline_rows, predictions),
        "model_audit": audit,
        "restricted_gates": {
            "uppercase_X_to_lowercase_x": _metrics(
                runtime_rows, baseline_rows,
                _restricted_gate(baseline_rows, predictions, "X", "x"),
            ),
        },
    }
    if not preparation["baseline_tokens_outside_wide_candidates"]:
        predictions, audit = decide_owned_formula_rows(
            model, contract, payload, runtime_rows, device, args.batch_size,
            preparation["baseline_context_tokens"],
        )
        variants["current_baseline_context"] = {
            "metrics": _metrics(runtime_rows, baseline_rows, predictions),
            "model_audit": audit,
            "restricted_gates": {
                "uppercase_X_to_lowercase_x": _metrics(
                    runtime_rows, baseline_rows,
                    _restricted_gate(baseline_rows, predictions, "X", "x"),
                ),
            },
        }
    report = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "training_performed": False,
        "runtime_admitted": False,
        "candidate_status": "probe_only_no_runtime_configuration",
        "formula_order": "current baseline order and layout relations",
        "preparation": {
            "records": len(runtime_rows),
            "formulas": len(_baseline_formulae(baseline_rows)),
            "candidate_widths": sorted({len(row["final_topk"]) for row in runtime_rows}),
            "baseline_tokens_outside_wide_candidates": len(
                preparation["baseline_tokens_outside_wide_candidates"]
            ),
        },
        "variants": variants,
        "inputs": {
            "baseline_sha256": _sha256(paths[0]),
            "wide_candidates_sha256": _sha256(paths[1]),
            "context_checkpoint_sha256": _sha256(paths[2]),
            "hwr_checkpoint_sha256": _sha256(paths[3]),
        },
        "limits": [
            "the checkpoint was selected under a Top-5 contract, not Top-10",
            "the direct formulas are repeatedly observed development evidence",
            "this probe cannot authorize a runtime candidate-contract expansion",
        ],
    }
    paths[4].parent.mkdir(parents=True, exist_ok=True)
    paths[4].write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    summary_fields = (
        "baseline_character_top1_count", "challenger_character_top1_count",
        "glyph_improved", "glyph_regressed", "changed_still_wrong",
        "changed_glyphs", "baseline_formula_exact_count",
        "challenger_formula_exact_count", "formula_improved", "formula_regressed",
    )
    print(json.dumps({
        "output": str(paths[4]), "sha256": _sha256(paths[4]),
        "runtime_admitted": False,
        "variants": {
            name: {
                "full_probe": {
                    key: value["metrics"][key] for key in summary_fields
                },
                "uppercase_X_to_lowercase_x": {
                    key: value["restricted_gates"]["uppercase_X_to_lowercase_x"][key]
                    for key in summary_fields
                },
            }
            for name, value in variants.items()
        },
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
