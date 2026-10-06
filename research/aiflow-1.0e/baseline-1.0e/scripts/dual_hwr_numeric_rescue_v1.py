#!/usr/bin/env python3
"""Merge two finalized HWR paths only for unambiguous numeric operands.

The gate never solves arithmetic and never changes operators, relations,
fences, layout, or grouping.  It may replace at most two baseline operands
with digits selected by an independently finalized auxiliary HWR path, and
only when that turns an invalid simple numeric sequence into a valid one.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import json
import math
from pathlib import Path

from character_tensor_v1 import _json_lines
from evaluate_48hz_prefix_v1 import _sha256
from formula_sequence_guard_v1 import (
    BINARY, DIGITS, RELATIONS, valid_numeric_sequence,
)
from train_context_decision_layer_v1 import _semantic_role


SCHEMA = "aiflow-dual-hwr-numeric-rescue/v1"
OUTPUT_SCHEMA = "aiflow-dual-hwr-numeric-finalized/v1"
CONFIG_SCHEMA = "aiflow-dual-hwr-numeric-rescue-runtime-config/v1"
DEFAULT_CONFIGURATION = {
    "auxiliary_policy": "current_expanded_writer_loo_probability_fusion",
    "auxiliary_weight": 0.6,
    "maximum_operand_changes": 2,
}


def validate_configuration(configuration: dict) -> dict:
    if set(configuration) != set(DEFAULT_CONFIGURATION):
        raise ValueError("dual HWR numeric rescue configuration fields mismatch")
    policy = str(configuration["auxiliary_policy"])
    weight = float(configuration["auxiliary_weight"])
    maximum = int(configuration["maximum_operand_changes"])
    if not policy:
        raise ValueError("dual HWR auxiliary policy must be non-empty")
    if not math.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError("dual HWR auxiliary weight must be in [0,1]")
    if maximum not in {1, 2}:
        raise ValueError("dual HWR numeric rescue permits one or two operand changes")
    return {
        "auxiliary_policy": policy,
        "auxiliary_weight": weight,
        "maximum_operand_changes": maximum,
    }


def _candidate_row(row: dict, kind: str) -> dict:
    record_id = str(row.get("record_id", ""))
    formula_id = str(row.get("formula_id", ""))
    candidates = [str(value) for value in row.get("final_topk", [])]
    probabilities = [float(value) for value in row.get("final_topk_probabilities", [])]
    if not record_id or not formula_id:
        raise ValueError(f"{kind} candidate identity is missing")
    if (
        not 1 <= len(candidates) <= 5
        or len(candidates) != len(set(candidates))
        or len(candidates) != len(probabilities)
        or any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in probabilities)
        or any(left < right for left, right in zip(probabilities, probabilities[1:]))
    ):
        raise ValueError(f"{kind} candidates are invalid: {record_id}")
    return {
        "record_id": record_id,
        "formula_id": formula_id,
        "final_topk": candidates,
        "final_topk_probabilities": probabilities,
    }


def _candidate_map(rows: list[dict], configuration: dict) -> dict[str, dict]:
    output = {}
    for source in rows:
        row = _candidate_row(source, "auxiliary")
        record_id = row["record_id"]
        if record_id in output:
            raise ValueError(f"duplicate auxiliary candidate: {record_id}")
        if str(source.get("hwr_policy", "")) != configuration["auxiliary_policy"]:
            raise ValueError(f"auxiliary HWR policy mismatch: {record_id}")
        weight = float(source.get("hwr_fusion_weight", -1.0))
        if not math.isclose(weight, configuration["auxiliary_weight"], abs_tol=1e-12):
            raise ValueError(f"auxiliary HWR weight mismatch: {record_id}")
        output[record_id] = row
    return output


def _finalized_map(rows: list[dict], kind: str) -> dict[str, dict]:
    output = {}
    for source in rows:
        row = _candidate_row(source, kind)
        record_id = row["record_id"]
        if record_id in output:
            raise ValueError(f"duplicate {kind} finalized row: {record_id}")
        token = str(source.get("finalized_top1", ""))
        if token not in row["final_topk"]:
            raise ValueError(f"{kind} finalized token escaped its HWR Top-5: {record_id}")
        try:
            context_index = int(source["context_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{kind} context index is invalid: {record_id}") from exc
        output[record_id] = {
            **row,
            "finalized_top1": token,
            "context_index": context_index,
            "layout_relation_from_previous": source.get("layout_relation_from_previous"),
            "context_available": bool(source.get("context_available", context_index > 0)),
            "decision_status": str(source.get("decision_status", "finalized")),
            "decision_source": str(source.get("decision_source", kind)),
            "changed": bool(source.get("changed", False)),
        }
    return output


def _formulae(rows: dict[str, dict]) -> dict[str, list[dict]]:
    output: dict[str, list[dict]] = defaultdict(list)
    for row in rows.values():
        output[row["formula_id"]].append(row)
    for formula_id, sequence in output.items():
        sequence.sort(key=lambda row: row["context_index"])
        indices = [row["context_index"] for row in sequence]
        if indices != list(range(len(sequence))):
            raise ValueError(f"formula context indices are not contiguous: {formula_id}")
    return dict(output)


def _candidate_union(baseline: dict, auxiliary: dict) -> tuple[list[str], list[float]]:
    candidates = list(baseline["final_topk"])
    probabilities = list(baseline["final_topk_probabilities"])
    for token, probability in zip(
        auxiliary["final_topk"], auxiliary["final_topk_probabilities"], strict=True,
    ):
        if token in candidates:
            index = candidates.index(token)
            probabilities[index] = max(probabilities[index], probability)
        else:
            candidates.append(token)
            probabilities.append(probability)
    return candidates, probabilities


def apply_dual_hwr_numeric_rescue(
    baseline_rows: list[dict], auxiliary_candidate_rows: list[dict],
    auxiliary_finalized_rows: list[dict], configuration: dict | None = None,
) -> tuple[list[dict], dict]:
    configuration = validate_configuration(configuration or DEFAULT_CONFIGURATION)
    baseline = _finalized_map(baseline_rows, "baseline")
    auxiliary = _finalized_map(auxiliary_finalized_rows, "auxiliary")
    evidence = _candidate_map(auxiliary_candidate_rows, configuration)
    if not baseline or set(baseline) != set(auxiliary) or set(baseline) != set(evidence):
        raise ValueError("dual HWR path coverage mismatch")
    for record_id in baseline:
        if (
            baseline[record_id]["formula_id"] != auxiliary[record_id]["formula_id"]
            or baseline[record_id]["formula_id"] != evidence[record_id]["formula_id"]
            or auxiliary[record_id]["final_topk"] != evidence[record_id]["final_topk"]
            or auxiliary[record_id]["final_topk_probabilities"]
            != evidence[record_id]["final_topk_probabilities"]
        ):
            raise ValueError(f"dual HWR evidence mismatch: {record_id}")
    baseline_formulae = _formulae(baseline)
    auxiliary_formulae = _formulae(auxiliary)
    if set(baseline_formulae) != set(auxiliary_formulae):
        raise ValueError("dual HWR formula coverage mismatch")

    selected = {record_id: row["finalized_top1"] for record_id, row in baseline.items()}
    changes = []
    skipped = Counter()
    auxiliary_order_mismatch_formulas = 0
    for formula_id, sequence in baseline_formulae.items():
        baseline_ids = [row["record_id"] for row in sequence]
        auxiliary_ids = [row["record_id"] for row in auxiliary_formulae[formula_id]]
        if baseline_ids != auxiliary_ids:
            auxiliary_order_mismatch_formulas += 1
            skipped["auxiliary_order_mismatch"] += 1
            continue
        # Formula order and layout remain owned by the baseline path.  The
        # auxiliary path contributes only record-local digit evidence.
        auxiliary_sequence = [auxiliary[record_id] for record_id in baseline_ids]
        before = [row["finalized_top1"] for row in sequence]
        proposed = list(before)
        changed_indices = []
        for index, (base_row, auxiliary_row) in enumerate(
            zip(sequence, auxiliary_sequence, strict=True)
        ):
            challenger = auxiliary_row["finalized_top1"]
            if (
                challenger != before[index]
                and _semantic_role(before[index]) == "operand"
                and challenger in DIGITS
            ):
                proposed[index] = challenger
                changed_indices.append(index)
        if not changed_indices:
            skipped["no_operand_to_digit_change"] += 1
            continue
        if len(changed_indices) > configuration["maximum_operand_changes"]:
            skipped["too_many_operand_changes"] += 1
            continue
        if valid_numeric_sequence(before):
            skipped["baseline_already_valid"] += 1
            continue
        numeric_intent = (
            len(sequence) >= 3
            and sum(token in DIGITS for token in proposed) >= 2
            and any(token in BINARY | RELATIONS for token in proposed)
        )
        if not numeric_intent or not valid_numeric_sequence(proposed):
            skipped["proposal_not_valid_numeric"] += 1
            continue
        for index in changed_indices:
            row = sequence[index]
            record_id = row["record_id"]
            selected[record_id] = proposed[index]
            changes.append({
                "formula_id": formula_id,
                "record_id": record_id,
                "context_index": index,
                "before": before[index],
                "after": proposed[index],
                "outside_baseline_top5": proposed[index] not in row["final_topk"],
            })

    output = []
    outside_union = 0
    for record_id, row in baseline.items():
        auxiliary_row = auxiliary[record_id]
        candidate_union, union_probabilities = _candidate_union(row, auxiliary_row)
        token = selected[record_id]
        if token not in candidate_union:
            outside_union += 1
        changed = token != row["finalized_top1"]
        output.append({
            "schema": OUTPUT_SCHEMA,
            "record_id": record_id,
            "formula_id": row["formula_id"],
            "context_index": row["context_index"],
            "layout_relation_from_previous": row["layout_relation_from_previous"],
            "context_available": row["context_available"],
            "decision_status": row["decision_status"],
            "decision_source": (
                "dual_hwr_numeric_rescue_v1" if changed else row["decision_source"]
            ),
            "baseline_top1": row["finalized_top1"],
            "auxiliary_top1": auxiliary_row["finalized_top1"],
            "finalized_top1": token,
            "changed": bool(row["changed"] or changed),
            "baseline_top5": row["final_topk"],
            "baseline_top5_probabilities": row["final_topk_probabilities"],
            "auxiliary_top5": auxiliary_row["final_topk"],
            "auxiliary_top5_probabilities": auxiliary_row["final_topk_probabilities"],
            "candidate_union": candidate_union,
            "candidate_union_path_scores": union_probabilities,
        })
    output.sort(key=lambda row: (row["formula_id"], row["context_index"], row["record_id"]))
    if outside_union:
        raise AssertionError("dual HWR numeric rescue invented a token")
    return output, {
        "schema": SCHEMA,
        "status": "shadow_runtime_only",
        "configuration": configuration,
        "records": len(output),
        "formulas": len(baseline_formulae),
        "changed_glyphs": len(changes),
        "changed_formulas": len({change["formula_id"] for change in changes}),
        "auxiliary_order_mismatch_formulas": auxiliary_order_mismatch_formulas,
        "changes": changes,
        "skipped": dict(sorted(skipped.items())),
        "baseline_top5_expansions": sum(change["outside_baseline_top5"] for change in changes),
        "candidate_union_preservation_rate": 1.0,
        "tokens_outside_dual_hwr_union": 0,
        "deleted_glyphs": 0,
        "grouping_mutations": 0,
        "operator_relation_or_fence_mutations": 0,
        "arithmetic_evaluation": False,
    }


def _write(path: Path, rows: list[dict]) -> None:
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "wt", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _self_test() -> None:
    before = ["4", r"\mathscr{C}", r"\div", r"\mathscr{C}", "=", "6"]
    after = ["4", "8", r"\div", "8", "=", "6"]
    baseline_rows = []
    auxiliary_rows = []
    evidence_rows = []
    for index, (base_token, auxiliary_token) in enumerate(zip(before, after, strict=True)):
        record_id = f"r{index}"
        baseline_top5 = [base_token]
        auxiliary_top5 = [auxiliary_token]
        baseline_rows.append({
            "record_id": record_id, "formula_id": "f", "context_index": index,
            "finalized_top1": base_token, "final_topk": baseline_top5,
            "final_topk_probabilities": [1.0],
        })
        auxiliary_rows.append({
            "record_id": record_id, "formula_id": "f", "context_index": index,
            "finalized_top1": auxiliary_token, "final_topk": auxiliary_top5,
            "final_topk_probabilities": [1.0],
        })
        evidence_rows.append({
            "record_id": record_id, "formula_id": "f", "final_topk": auxiliary_top5,
            "final_topk_probabilities": [1.0],
            "hwr_policy": DEFAULT_CONFIGURATION["auxiliary_policy"],
            "hwr_fusion_weight": DEFAULT_CONFIGURATION["auxiliary_weight"],
        })
    merged, audit = apply_dual_hwr_numeric_rescue(
        baseline_rows, evidence_rows, auxiliary_rows,
    )
    assert [row["finalized_top1"] for row in merged] == after
    assert audit["changed_glyphs"] == 2
    assert audit["baseline_top5_expansions"] == 2
    assert all("label" not in row for row in merged)
    bad_evidence = [dict(row) for row in evidence_rows]
    bad_evidence[0]["hwr_policy"] = "untrusted"
    try:
        apply_dual_hwr_numeric_rescue(baseline_rows, bad_evidence, auxiliary_rows)
    except ValueError:
        pass
    else:
        raise AssertionError("untrusted auxiliary HWR policy was accepted")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-finalized", type=Path)
    parser.add_argument("--auxiliary-candidates", type=Path)
    parser.add_argument("--auxiliary-finalized", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        print('{"self_test":"pass"}')
        return 0
    required = (
        args.baseline_finalized, args.auxiliary_candidates,
        args.auxiliary_finalized, args.config, args.output,
    )
    if any(value is None for value in required):
        parser.error("baseline, auxiliary, config, and output paths are required")
    paths = [Path(value).expanduser().resolve() for value in required]
    baseline_path, auxiliary_candidate_path, auxiliary_finalized_path, config_path, output_path = paths
    if not all(path.is_file() for path in paths[:-1]):
        parser.error("dual HWR numeric rescue input is missing")
    if output_path.exists():
        parser.error(f"refusing to overwrite dual HWR output: {output_path}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if (
        payload.get("schema") != CONFIG_SCHEMA
        or payload.get("gate", {}).get("runtime_admitted") is not True
    ):
        parser.error("dual HWR numeric rescue config is not admitted")
    rows, audit = apply_dual_hwr_numeric_rescue(
        list(_json_lines(baseline_path)),
        list(_json_lines(auxiliary_candidate_path)),
        list(_json_lines(auxiliary_finalized_path)),
        payload["configuration"],
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write(output_path, rows)
    print(json.dumps({
        "output": str(output_path), "sha256": _sha256(output_path),
        "inputs": {
            "baseline_finalized_sha256": _sha256(baseline_path),
            "auxiliary_candidates_sha256": _sha256(auxiliary_candidate_path),
            "auxiliary_finalized_sha256": _sha256(auxiliary_finalized_path),
            "config_sha256": _sha256(config_path),
        },
        "audit": audit,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
