#!/usr/bin/env python3
"""Compose conservative candidate-preserving formula placement rescues."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from character_tensor_v1 import _json_lines
from dual_straight_one_slot_rescue_v1 import (
    DEFAULT_CONFIGURATION as DUAL_ONE_CONFIGURATION,
    apply_dual_straight_one_slot_rescue,
    validate_configuration as validate_dual_one_configuration,
)
from formula_role_typo_rescue_v1 import (
    DEFAULT_CONFIGURATION as FORMULA_ROLE_TYPO_CONFIGURATION,
    apply_formula_role_typo_rescue,
    validate_configuration as validate_formula_role_typo_configuration,
)
from operator_cross_slot_rescue_v1 import (
    DEFAULT_CONFIGURATION as OPERATOR_CROSS_CONFIGURATION,
    apply_operator_cross_slot_rescue,
    validate_configuration as validate_operator_cross_configuration,
)
from unmatched_fence_operand_rescue_v1 import (
    DEFAULT_CONFIGURATION as UNMATCHED_FENCE_CONFIGURATION,
    apply_unmatched_fence_operand_rescue,
    validate_configuration as validate_unmatched_fence_configuration,
)
from evaluate_48hz_prefix_v1 import _sha256
from wide_candidate_syntax_rescue_v1 import _write


SCHEMA = "aiflow-formula-placement-rescue/v1"
OUTPUT_SCHEMA = "aiflow-formula-placement-finalized/v1"
CONFIG_SCHEMA = "aiflow-formula-placement-rescue-runtime-config/v1"
STAGE_ORDER = (
    "unmatched_fence_operand",
    "operator_cross_slot",
    "dual_straight_one_slot",
    "formula_role_typo",
)
DEFAULT_CONFIGURATION = {
    "stage_order": list(STAGE_ORDER),
    "unmatched_fence_operand": UNMATCHED_FENCE_CONFIGURATION,
    "operator_cross_slot": OPERATOR_CROSS_CONFIGURATION,
    "dual_straight_one_slot": DUAL_ONE_CONFIGURATION,
    "formula_role_typo": FORMULA_ROLE_TYPO_CONFIGURATION,
}


def validate_configuration(configuration: dict) -> dict:
    if set(configuration) != set(DEFAULT_CONFIGURATION):
        raise ValueError("formula placement configuration fields mismatch")
    stage_order = [str(value) for value in configuration["stage_order"]]
    if stage_order != list(STAGE_ORDER):
        raise ValueError("formula placement stage order mismatch")
    return {
        "stage_order": stage_order,
        "unmatched_fence_operand": validate_unmatched_fence_configuration(
            configuration["unmatched_fence_operand"]
        ),
        "operator_cross_slot": validate_operator_cross_configuration(
            configuration["operator_cross_slot"]
        ),
        "dual_straight_one_slot": validate_dual_one_configuration(
            configuration["dual_straight_one_slot"]
        ),
        "formula_role_typo": validate_formula_role_typo_configuration(
            configuration["formula_role_typo"]
        ),
    }


def _identity(rows: list[dict]) -> list[tuple[str, str, int]]:
    return [
        (
            str(row["record_id"]), str(row["formula_id"]),
            int(row["context_index"]),
        )
        for row in rows
    ]


def apply_formula_placement_rescue(
    baseline_rows: list[dict], candidate_rows: list[dict],
    configuration: dict | None = None,
) -> tuple[list[dict], dict]:
    configuration = validate_configuration(configuration or DEFAULT_CONFIGURATION)
    if not baseline_rows:
        raise ValueError("formula placement baseline is empty")
    original_identity = _identity(baseline_rows)
    original = {
        str(row["record_id"]): str(row["finalized_top1"])
        for row in baseline_rows
    }
    candidate_map = {str(row["record_id"]): row for row in candidate_rows}
    if len(candidate_map) != len(candidate_rows) or set(candidate_map) != set(original):
        raise ValueError("formula placement candidate coverage mismatch")
    rows = baseline_rows
    stage_audits = {}
    stages = (
        (
            "unmatched_fence_operand",
            apply_unmatched_fence_operand_rescue,
        ),
        ("operator_cross_slot", apply_operator_cross_slot_rescue),
        ("dual_straight_one_slot", apply_dual_straight_one_slot_rescue),
        ("formula_role_typo", apply_formula_role_typo_rescue),
    )
    changed_record_ids = set()
    for name, apply_stage in stages:
        rows, audit = apply_stage(rows, candidate_rows, configuration[name])
        if _identity(rows) != original_identity:
            raise AssertionError(f"formula placement identity changed at {name}")
        stage_ids = {str(change["record_id"]) for change in audit["changes"]}
        if changed_record_ids & stage_ids:
            raise AssertionError("formula placement stages changed one record twice")
        changed_record_ids.update(stage_ids)
        stage_audits[name] = audit
    outside = 0
    output = []
    for row in rows:
        record_id = str(row["record_id"])
        token = str(row["finalized_top1"])
        changed = token != original[record_id]
        if changed and token not in [
            str(value) for value in candidate_map[record_id]["final_topk"]
        ]:
            outside += 1
        output.append({
            **row,
            "schema": OUTPUT_SCHEMA,
            "baseline_top1_before_formula_placement": original[record_id],
            "changed_by_formula_placement": changed,
        })
    if outside:
        raise AssertionError("formula placement rescue invented a token")
    changed_formulas = {
        str(row["formula_id"])
        for row in output if row["changed_by_formula_placement"]
    }
    return output, {
        "schema": SCHEMA,
        "status": "shadow_runtime_only",
        "configuration": configuration,
        "records": len(output),
        "formulas": len({str(row["formula_id"]) for row in output}),
        "changed_glyphs": len(changed_record_ids),
        "changed_formulas": len(changed_formulas),
        "stage_audits": stage_audits,
        "stage_changed_glyphs": {
            name: stage_audits[name]["changed_glyphs"] for name in STAGE_ORDER
        },
        "candidate_preservation_rate": 1.0,
        "tokens_outside_candidate_union": 0,
        "inserted_or_deleted_glyphs": 0,
        "glyph_order_mutations": 0,
        "grouping_mutations": 0,
        "layout_mutations": 0,
        "arithmetic_evaluation": False,
    }


def _self_test() -> None:
    baseline = [
        {
            "record_id": f"r{index}", "formula_id": "f",
            "context_index": index, "finalized_top1": token,
        }
        for index, token in enumerate(("2", "+", "2"))
    ]
    candidates = []
    for index, token in enumerate(("2", "+", "2")):
        candidates.append({
            "record_id": f"r{index}", "formula_id": "f",
            "final_topk": [token] + [f"z{index}_{offset}" for offset in range(19)],
            "final_topk_probabilities": [1.0] + [0.0] * 19,
            "hwr_policy": "old_new_product_probability_fusion",
            "hwr_fusion_weight": 0.6,
            "geometry": {
                "aspect_log": 0.0, "path_over_diag": 1.0,
                "direction_y": 0.0, "stroke_count": 1.0, "width_rel": 0.1,
            },
        })
    output, audit = apply_formula_placement_rescue(baseline, candidates)
    assert [row["finalized_top1"] for row in output] == ["2", "+", "2"]
    assert audit["changed_glyphs"] == 0
    assert audit["stage_changed_glyphs"] == {
        "unmatched_fence_operand": 0,
        "operator_cross_slot": 0,
        "dual_straight_one_slot": 0,
        "formula_role_typo": 0,
    }
    assert not audit["arithmetic_evaluation"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-finalized", type=Path)
    parser.add_argument("--top20-candidates", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        print('{"self_test":"pass"}')
        return 0
    if any(value is None for value in (
        args.baseline_finalized, args.top20_candidates, args.config, args.output,
    )):
        parser.error("baseline, Top-20 candidates, config, and output are required")
    baseline_path = args.baseline_finalized.expanduser().resolve()
    candidate_path = args.top20_candidates.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not all(path.is_file() for path in (baseline_path, candidate_path, config_path)):
        parser.error("formula placement runtime input is missing")
    if output_path.exists():
        parser.error(f"refusing to overwrite formula placement output: {output_path}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    gate = payload.get("gate", {})
    if (
        payload.get("schema") != CONFIG_SCHEMA
        or payload.get("rescue_schema") != SCHEMA
        or gate.get("shadow_runtime_admitted") is not True
        or gate.get("product_default_admitted") is not False
    ):
        parser.error("formula placement shadow runtime config is not admitted")
    rows, audit = apply_formula_placement_rescue(
        list(_json_lines(baseline_path)), list(_json_lines(candidate_path)),
        payload["configuration"],
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write(output_path, rows)
    print(json.dumps({
        "output": str(output_path), "sha256": _sha256(output_path),
        "status": "shadow_runtime_only",
        "changed_glyphs": audit["changed_glyphs"],
        "changed_formulas": audit["changed_formulas"],
        "stage_changed_glyphs": audit["stage_changed_glyphs"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
