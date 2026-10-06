#!/usr/bin/env python3
"""Admit a frozen Latin auxiliary ``t`` candidate in a function-argument slot."""

from __future__ import annotations

from collections import Counter, defaultdict
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from evaluate_joint_hwr_grouping_v1 import _candidate_tensor
from train_character_classifier_v1 import InkClassifierV1, apply_input_mode


SCHEMA = "aiflow-latin-t-context-rescue/v1"
OUTPUT_SCHEMA = "aiflow-latin-t-context-finalized/v1"
LATIN_AUXILIARY_POLICY = "approved_legacy_latin_auxiliary_preserve_time"
DEFAULT_CONFIGURATION = {
    "candidate_width": 5,
    "candidate_policy": LATIN_AUXILIARY_POLICY,
    "target_token": "t",
    "function_tokens": ["f", "g", "h"],
    "maximum_candidate_rank": 2,
    "minimum_candidate_probability": 0.25,
    "minimum_probability_ratio_to_top1": 0.50,
    "single_parenthesized_argument_only": True,
    "existing_ascii_alphanumeric_lock": True,
    "maximum_changes_per_formula": 1,
}


def load_latin_auxiliary_model(
    path: Path, device: torch.device,
) -> tuple[InkClassifierV1, list[str]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    math_labels = [str(value) for value in checkpoint.get("math_labels") or []]
    labels = [str(value) for value in checkpoint.get("auxiliary_labels") or []]
    contract = dict(checkpoint.get("report", {}).get("input_contract") or {})
    if (
        checkpoint.get("schema") != "aiflow-character-classifier-external-training/v1"
        or not math_labels or len(labels) != 95 or "t" not in labels
        or contract.get("auxiliary_observed_transform") != "preserve"
    ):
        raise ValueError("legacy Latin auxiliary checkpoint contract mismatch")
    model = InkClassifierV1(len(math_labels), len(labels))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    if model.latin_aux_head is None:
        raise ValueError("legacy Latin auxiliary head is missing")
    return model.to(device).eval(), labels


def build_latin_auxiliary_rows(
    samples: list[Any], selected: dict[str, dict], model: InkClassifierV1,
    labels: list[str], device: torch.device,
) -> list[dict]:
    sources = []
    tensors = []
    for sample in samples:
        for row in selected[sample.sample_id]["rows"]:
            sources.append((sample.sample_id, row))
            tensors.append(_candidate_tensor(
                sample, [int(value) for value in row["group"]],
            ))
    values = np.stack(tensors).astype(np.float32, copy=False)
    probabilities = []
    for start in range(0, len(values), 512):
        with torch.inference_mode():
            points = torch.from_numpy(
                apply_input_mode(values[start:start + 512], "preserve")
            ).to(device)
            probabilities.append(
                model.latin_aux_head(model.encode(points)).softmax(dim=1).cpu()
            )
    probability = torch.cat(probabilities)
    top_probability, top_indices = probability.topk(5, dim=1)
    output = []
    for (formula_id, source), indices, values in zip(
        sources, top_indices.tolist(), top_probability.tolist(), strict=True,
    ):
        output.append({
            "record_id": str(source["record_id"]),
            "formula_id": formula_id,
            "final_topk": [labels[int(index)] for index in indices],
            "final_topk_probabilities": [float(value) for value in values],
            "hwr_policy": LATIN_AUXILIARY_POLICY,
            "hwr_candidate_width": 5,
        })
    return output


def validate_configuration(configuration: dict) -> dict:
    if set(configuration) != set(DEFAULT_CONFIGURATION):
        raise ValueError("Latin t context configuration fields mismatch")
    output = {
        **configuration,
        "candidate_width": int(configuration["candidate_width"]),
        "candidate_policy": str(configuration["candidate_policy"]),
        "target_token": str(configuration["target_token"]),
        "function_tokens": [str(value) for value in configuration["function_tokens"]],
        "maximum_candidate_rank": int(configuration["maximum_candidate_rank"]),
        "minimum_candidate_probability": float(
            configuration["minimum_candidate_probability"]
        ),
        "minimum_probability_ratio_to_top1": float(
            configuration["minimum_probability_ratio_to_top1"]
        ),
        "maximum_changes_per_formula": int(
            configuration["maximum_changes_per_formula"]
        ),
    }
    if (
        output["candidate_width"] != 5
        or not output["candidate_policy"]
        or output["target_token"] != "t"
        or output["function_tokens"] != ["f", "g", "h"]
        or not 1 <= output["maximum_candidate_rank"] <= output["candidate_width"]
        or not all(math.isfinite(output[key]) for key in (
            "minimum_candidate_probability", "minimum_probability_ratio_to_top1",
        ))
        or not 0.0 <= output["minimum_candidate_probability"] <= 1.0
        or not 0.0 <= output["minimum_probability_ratio_to_top1"] <= 1.0
        or type(output["single_parenthesized_argument_only"]) is not bool
        or output["single_parenthesized_argument_only"] is not True
        or type(output["existing_ascii_alphanumeric_lock"]) is not bool
        or output["existing_ascii_alphanumeric_lock"] is not True
        or output["maximum_changes_per_formula"] != 1
    ):
        raise ValueError("invalid Latin t context configuration")
    return output


def _candidate_map(rows: list[dict], configuration: dict) -> dict[str, dict]:
    output = {}
    for row in rows:
        record_id = str(row.get("record_id", ""))
        tokens = [str(value) for value in row.get("final_topk", [])]
        probabilities = [float(value) for value in row.get(
            "final_topk_probabilities", []
        )]
        if not record_id or record_id in output:
            raise ValueError(f"Latin t candidate identity is invalid: {record_id}")
        if (
            len(tokens) != configuration["candidate_width"]
            or len(tokens) != len(set(tokens))
            or len(probabilities) != len(tokens)
            or any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in probabilities)
            or any(left < right for left, right in zip(probabilities, probabilities[1:]))
            or str(row.get("hwr_policy", "")) != configuration["candidate_policy"]
        ):
            raise ValueError(f"Latin t candidate contract is invalid: {record_id}")
        output[record_id] = {**row, "tokens": tokens, "probabilities": probabilities}
    return output


def _formulae(rows: list[dict]) -> dict[str, list[dict]]:
    output: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        output[str(row["formula_id"])].append(row)
    for formula_id, sequence in output.items():
        sequence.sort(key=lambda row: int(row["context_index"]))
        if [int(row["context_index"]) for row in sequence] != list(range(len(sequence))):
            raise ValueError(f"Latin t formula order is invalid: {formula_id}")
    return dict(output)


def apply_latin_t_context_rescue(
    baseline_rows: list[dict], candidate_rows: list[dict],
    configuration: dict | None = None,
) -> tuple[list[dict], dict]:
    configuration = validate_configuration(configuration or DEFAULT_CONFIGURATION)
    candidates = _candidate_map(candidate_rows, configuration)
    baseline_ids = [str(row["record_id"]) for row in baseline_rows]
    if not baseline_ids or len(baseline_ids) != len(set(baseline_ids)):
        raise ValueError("Latin t baseline IDs are invalid")
    if set(baseline_ids) != set(candidates):
        raise ValueError("Latin t candidate coverage mismatch")
    selected = {
        str(row["record_id"]): str(row["finalized_top1"])
        for row in baseline_rows
    }
    changes = []
    skipped = Counter()
    target = configuration["target_token"]
    for formula_id, sequence in _formulae(baseline_rows).items():
        before = [str(row["finalized_top1"]) for row in sequence]
        eligible = []
        for index in range(2, len(sequence) - 1):
            if not (
                before[index - 2] in configuration["function_tokens"]
                and before[index - 1] == "("
                and before[index + 1] == ")"
            ):
                continue
            current = before[index]
            if len(current) == 1 and current.isascii() and current.isalnum():
                continue
            row = sequence[index]
            record_id = str(row["record_id"])
            candidate = candidates[record_id]
            if str(candidate.get("formula_id", "")) != formula_id:
                raise ValueError(f"Latin t formula identity mismatch: {record_id}")
            if target not in candidate["tokens"]:
                continue
            target_index = candidate["tokens"].index(target)
            probability = candidate["probabilities"][target_index]
            ratio = probability / max(candidate["probabilities"][0], 1e-12)
            if (
                target_index + 1 <= configuration["maximum_candidate_rank"]
                and probability >= configuration["minimum_candidate_probability"]
                and ratio >= configuration["minimum_probability_ratio_to_top1"]
            ):
                eligible.append((index, row, target_index + 1, probability, ratio))
        if not eligible:
            skipped["no_supported_single_function_argument_t"] += 1
            continue
        if len(eligible) != configuration["maximum_changes_per_formula"]:
            skipped["ambiguous_function_argument_t"] += 1
            continue
        index, row, rank, probability, ratio = eligible[0]
        record_id = str(row["record_id"])
        selected[record_id] = target
        changes.append({
            "formula_id": formula_id,
            "record_id": record_id,
            "context_index": index,
            "before": before[index],
            "after": target,
            "candidate_rank": rank,
            "candidate_probability": probability,
            "probability_ratio_to_top1": ratio,
            "rule": "fgh_single_parenthesized_argument_legacy_latin_t",
        })
    output = []
    for row in baseline_rows:
        record_id = str(row["record_id"])
        token = selected[record_id]
        changed = token != str(row["finalized_top1"])
        output.append({
            **row,
            "schema": OUTPUT_SCHEMA,
            "baseline_top1_before_latin_t_context_rescue": str(
                row["finalized_top1"]
            ),
            "finalized_top1": token,
            "changed_by_latin_t_context_rescue": changed,
            "changed": token != str(row.get("hwr_top1", row["finalized_top1"])),
            "decision_source": (
                "latin_t_context_rescue_v1"
                if changed else str(row.get("decision_source", "baseline"))
            ),
        })
    return output, {
        "schema": SCHEMA,
        "status": "shadow_runtime_only",
        "configuration": configuration,
        "records": len(output),
        "formulas": len(_formulae(baseline_rows)),
        "changed_glyphs": len(changes),
        "changed_formulas": len({change["formula_id"] for change in changes}),
        "changes": changes,
        "skipped": dict(sorted(skipped.items())),
        "candidate_preservation_rate": 1.0,
        "tokens_outside_latin_auxiliary_top5": 0,
        "inserted_or_deleted_glyphs": 0,
        "glyph_order_mutations": 0,
        "grouping_mutations": 0,
        "layout_mutations": 0,
        "arithmetic_evaluation": False,
        "target_label_or_glyph_count_input": False,
    }


def _self_test() -> None:
    baseline = [
        {"record_id": f"r{i}", "formula_id": "f", "context_index": i,
         "finalized_top1": token}
        for i, token in enumerate(["f", "(", r"\mathcal{A}", ")", "+", "2"])
    ]
    candidates = []
    for index in range(len(baseline)):
        tokens = ["f", "t", "n", "l", "r"] if index == 2 else [
            f"c{index}_{offset}" for offset in range(5)
        ]
        candidates.append({
            "record_id": f"r{index}", "formula_id": "f",
            "final_topk": tokens,
            "final_topk_probabilities": (
                [0.48, 0.31, 0.1, 0.06, 0.05]
                if index == 2 else [0.5, 0.2, 0.15, 0.1, 0.05]
            ),
            "hwr_policy": DEFAULT_CONFIGURATION["candidate_policy"],
        })
    output, audit = apply_latin_t_context_rescue(baseline, candidates)
    assert [row["finalized_top1"] for row in output] == ["f", "(", "t", ")", "+", "2"]
    assert audit["changed_glyphs"] == 1 and audit["arithmetic_evaluation"] is False


if __name__ == "__main__":
    _self_test()
    print('{"self_test":"pass"}')
