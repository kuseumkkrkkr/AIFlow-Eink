#!/usr/bin/env python3
"""Research-only raw online-ink to candidate-preserving formula runtime.

The runtime selects one exact-cover stroke partition from a frozen Top-N
lattice, classifies each selected group with the frozen HWR head, and lets the
frozen masked-context finalizer choose only among supplied HWR candidates. Its
default inference path never inserts/deletes a glyph, evaluates arithmetic, or
uses target labels/counts. The opt-in semantic shadow includes narrow flat-
arithmetic, unique-equation, and boundary-syntax diagnostics but cannot mutate default output.
The current partition gate was tuned after inspecting its chronological
evaluation set, so callers must explicitly opt into shadow use.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch

from evaluate_48hz_prefix_v1 import DEFAULT_PRODUCT, _load_model, _sha256
from evaluate_joint_hwr_grouping_v1 import _candidate_embeddings, _probabilities
from dual_hwr_numeric_rescue_v1 import (
    apply_dual_hwr_numeric_rescue,
    validate_configuration as validate_dual_numeric_configuration,
)
from wide_candidate_syntax_rescue_v1 import (
    apply_wide_candidate_syntax_rescue,
    validate_configuration as validate_wide_syntax_configuration,
)
from formula_placement_rescue_v1 import (
    CONFIG_SCHEMA as PLACEMENT_CONFIG_SCHEMA, SCHEMA as PLACEMENT_SCHEMA,
    apply_formula_placement_rescue,
    validate_configuration as validate_placement_configuration,
)
from straight_equality_slot_rescue_v1 import (
    CONFIG_SCHEMA as EQUALITY_CONFIG_SCHEMA, SCHEMA as EQUALITY_SCHEMA,
    apply_straight_equality_slot_rescue,
    validate_configuration as validate_equality_configuration,
)
from latin_t_context_rescue_v1 import (
    apply_latin_t_context_rescue, build_latin_auxiliary_rows,
    load_latin_auxiliary_model,
    validate_configuration as validate_latin_t_configuration,
)
from evaluate_partition_context_ranker_v1 import (
    FEATURE_NAMES, POSTHOC_DESIGN_GUARD, SCHEMA as RANKER_SCHEMA,
    _apply_cross_merge_token_locks, _attach_features, _auxiliary_rows,
    _cross_merge_selection, _gate_accept, _geometry_selection, _partition_rows,
    _relation_merge_selection, _repeat_merge_selection, _select,
    _shadow_configuration,
    _validate_cross_merge_configuration, _validate_relation_merge_configuration,
    _validate_repeat_merge_configuration,
)
from finalize_formula_context_v1 import DEFAULT_CONTEXT, OwnedFormulaContextFinalizer
from formula_layout_v1 import finalize_formula_outputs, selected_layout_evidence_rows
from formula_acceptance_guard_v1 import (
    apply_formula_acceptance_guard, load_configuration as load_acceptance_configuration,
)
from singleton_shape_rescue_v1 import (
    SCHEMA as SINGLETON_SCHEMA, apply_singleton_shape_rescue,
    validate_configuration as validate_singleton_configuration,
)
from pairwise_shape_rescue_v1 import (
    apply_pairwise_shape_rescue,
    load_pairwise_shape_artifacts,
)
from stroke_grouping_v1 import build_lattice, candidate_features, enumerate_partitions
from selective_2d_anytime_v1 import Selective2DAnytimeSolverV1, Selective2DConfigV1
from selective_decoder_v1 import decode_selective_partition
from train_project_owned_grouping_v1 import Sample


SCHEMA = "aiflow-raw-formula-context-runtime/v1"
OUTPUT_SCHEMA = "aiflow-raw-formula-context-result/v1"
CANDIDATE_FUSION_CONFIG_SCHEMA = "aiflow-raw-candidate-context-fusion-runtime-config/v1"
SEMANTIC_GUARD_SHADOW_SCHEMA = "aiflow-selective-semantic-guard-shadow/v9"
NUMERIC_LOOKALIKE_CANDIDATES = {
    "|": "digit",
    "(": "digit",
    ")": "digit",
    "/": "digit",
    r"\rfloor": "digit",
    r"\lfloor": "digit",
    "O": "digit",
    "o": "digit",
    r"\mathcal{O}": "digit",
    r"\circ": "digit",
    r"\hbar": "digit",
    r"\eta": "digit",
    r"\parr": "digit",
    r"\simeq": "equality",
    r"\approx": "equality",
    r"\asymp": "equality",
    r"\equiv": "equality",
    r"\neq": "equality",
    r"\triangledown": "equality",
}


def _unique_unpaired_bar_arithmetic_shadow(
    rows: list[dict[str, Any]], predictions: dict[str, str],
) -> tuple[dict[str, str], dict[str, Any]]:
    """Test a lone bare vertical-stroke token only if one Top-5 replacement
    makes a complete exact equation in a flat arithmetic formula.

    This is an explicit post-hoc shadow because arithmetic plausibility must
    never override a user's intentionally incorrect written equation.
    """
    from semantic_equation_guard_v1 import (
        ARITHMETIC_TOKENS, _flat, is_exact_arithmetic_equation,
    )

    output = dict(predictions)
    ordered = sorted(rows, key=lambda row: int(row["context"]["index"]))
    baseline = [output[str(row["record_id"])] for row in ordered]
    # Explicit TeX relation/norm tokens are not rewritten by this experiment.
    bar_tokens = frozenset({"|"})
    bar_positions = [index for index, token in enumerate(baseline) if token in bar_tokens]
    audit: dict[str, Any] = {
        "status": "skipped",
        "reason": "not_a_single_numeric_equation_bar_case",
        "changes": [],
        "new_tokens": 0,
        "deleted_glyphs": 0,
        "grouping_mutations": 0,
        "arithmetic_evaluation": True,
    }
    if (
        not 3 <= len(ordered) <= 12
        or not _flat(ordered)
        or len(bar_positions) != 1
        or baseline.count("=") != 1
        or any(token not in ARITHMETIC_TOKENS and token not in bar_tokens for token in baseline)
    ):
        return output, audit

    index = bar_positions[0]
    row = ordered[index]
    candidates = [str(token) for token in row["final_topk"]]
    valid_replacements = []
    for token in candidates:
        if token == baseline[index] or token not in ARITHMETIC_TOKENS:
            continue
        proposed = list(baseline)
        proposed[index] = token
        if is_exact_arithmetic_equation(tuple(proposed)):
            valid_replacements.append(token)
    if len(valid_replacements) != 1:
        audit["reason"] = (
            "no_unique_exact_equation_candidate"
            if not valid_replacements else "multiple_exact_equation_candidates"
        )
        audit["valid_replacement_count"] = len(valid_replacements)
        return output, audit

    selected = valid_replacements[0]
    output[str(row["record_id"])] = selected
    audit.update({
        "status": "changed_shadow_only",
        "reason": "unique_exact_equation_candidate",
        "valid_replacement_count": 1,
        "changes": [{
            "record_id": str(row["record_id"]),
            "before": baseline[index],
            "after": selected,
            "candidate_rank": candidates.index(selected) + 1,
        }],
    })
    return output, audit


def _boundary_bar_as_unit_shadow(
    rows: list[dict[str, Any]], predictions: dict[str, str],
) -> tuple[dict[str, str], dict[str, Any]]:
    """Replace a lone boundary bar with an existing `1` candidate before/after
    an operator, where a binary relation reading has no operand on one side.

    This is a syntax-only diagnostic, not a truth/answer check. It remains
    shadow-only because partial ink and author-intended notation can be valid.
    """
    from semantic_equation_guard_v1 import ARITHMETIC_TOKENS, DIGITS, _flat

    output = dict(predictions)
    ordered = sorted(rows, key=lambda row: int(row["context"]["index"]))
    baseline = [output[str(row["record_id"])] for row in ordered]
    audit: dict[str, Any] = {
        "status": "skipped",
        "reason": "not_a_single_boundary_bar_before_or_after_operator",
        "changes": [],
        "new_tokens": 0,
        "deleted_glyphs": 0,
        "grouping_mutations": 0,
    }
    bars = [index for index, token in enumerate(baseline) if token == "|"]
    operators = frozenset({"+", "-", r"\times", r"\div", "/", r"\cdot"})
    if not 3 <= len(ordered) <= 12 or not _flat(ordered) or len(bars) != 1:
        return output, audit
    if baseline.count("=") == 1 and all(
        token in ARITHMETIC_TOKENS or token == "|" for token in baseline
    ):
        audit["reason"] = "numeric_equation_deferred_to_exact_equation_guard"
        return output, audit

    index = bars[0]
    leading_numeric_slot = (
        index == 0
        and len(baseline) > 1
        and baseline[1] in DIGITS
        and not any(token in {
            "=", r"\simeq", r"\approx", r"\asymp", r"\equiv", r"\neq",
            "<", ">", r"\leq", r"\geq",
        } for token in baseline)
    )
    boundary_operator = (
        index == 0 and len(baseline) > 1 and baseline[1] in operators
    ) or (
        index == len(baseline) - 1 and index > 0 and baseline[index - 1] in operators
    )
    row = ordered[index]
    candidates = [str(token) for token in row["final_topk"]]
    if not boundary_operator and not leading_numeric_slot:
        return output, audit
    if "1" not in candidates:
        audit["reason"] = "unit_candidate_missing_from_top5"
        return output, audit

    output[str(row["record_id"])] = "1"
    audit.update({
        "status": "changed_shadow_only",
        "reason": (
            "lone_bar_at_leading_numeric_slot"
            if leading_numeric_slot else "lone_bar_at_unary_expression_boundary"
        ),
        "changes": [{
            "record_id": str(row["record_id"]),
            "before": "|",
            "after": "1",
            "candidate_rank": candidates.index("1") + 1,
        }],
    })
    return output, audit


def _unique_exact_equation_candidates_shadow(
    rows: list[dict[str, Any]], predictions: dict[str, str], *,
    max_changed_glyphs: int = 3,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Repair a flat equation only when its Top-5 has one
    exact numeric reading after at most three known lookalike corrections.

    The rule is deliberately shadow-only: a mathematically false user-written
    equation can be legitimate, and exact arithmetic must not override intent.
    """
    from itertools import combinations, product

    from semantic_equation_guard_v1 import (
        ARITHMETIC_TOKENS, DIGITS, _flat, is_exact_arithmetic_equation,
    )

    output = dict(predictions)
    ordered = sorted(rows, key=lambda row: int(row["context"]["index"]))
    baseline = [output[str(row["record_id"])] for row in ordered]
    audit: dict[str, Any] = {
        "status": "skipped",
        "reason": "not_a_flat_candidate_equation",
        "changes": [],
        "new_tokens": 0,
        "deleted_glyphs": 0,
        "grouping_mutations": 0,
        "max_changed_glyphs": int(max_changed_glyphs),
        "candidate_sequences_checked": 0,
        "arithmetic_evaluation": True,
    }
    if (
        isinstance(max_changed_glyphs, bool)
        or not isinstance(max_changed_glyphs, int)
        or not 1 <= max_changed_glyphs <= 3
    ):
        raise ValueError("max_changed_glyphs must be in [1, 3]")
    if (
        not 3 <= len(ordered) <= 12
        or not _flat(ordered)
    ):
        return output, audit
    change_options: dict[int, list[str]] = {}
    for index, token in enumerate(baseline):
        category = NUMERIC_LOOKALIKE_CANDIDATES.get(token)
        if token in ARITHMETIC_TOKENS and category is None:
            continue
        if category is None:
            audit["reason"] = "unsupported_symbol_in_baseline"
            audit["unsupported_tokens"] = sorted({
                value for value in baseline if value not in ARITHMETIC_TOKENS
                and value not in NUMERIC_LOOKALIKE_CANDIDATES
            })
            return output, audit
        symbol = ordered[index]
        candidates = [str(value) for value in symbol["final_topk"]]
        allowed = DIGITS if category == "digit" else frozenset({"="})
        alternatives = [
            value for value in candidates
            if value != token and value in allowed
        ]
        if not alternatives:
            audit["reason"] = "lookalike_has_no_admitted_arithmetic_candidate"
            audit["blocked_record_id"] = str(symbol["record_id"])
            return output, audit
        change_options[index] = alternatives

    if not 1 <= len(change_options) <= max_changed_glyphs:
        audit["reason"] = "required_changes_outside_budget"
        audit["required_changed_glyphs"] = len(change_options)
        return output, audit
    valid_sequences: set[tuple[str, ...]] = set()
    indices = tuple(change_options)
    for changed_count in range(1, len(indices) + 1):
        for changed_indices in combinations(indices, changed_count):
            replacements = [change_options[index] for index in changed_indices]
            for replacement_tokens in product(*replacements):
                proposed = list(baseline)
                for index, token in zip(changed_indices, replacement_tokens, strict=True):
                    proposed[index] = token
                audit["candidate_sequences_checked"] += 1
                if is_exact_arithmetic_equation(tuple(proposed)):
                    valid_sequences.add(tuple(proposed))
                    if len(valid_sequences) > 1:
                        break
            if len(valid_sequences) > 1:
                break
        if len(valid_sequences) > 1:
            break

    if len(valid_sequences) != 1:
        audit["reason"] = (
            "no_exact_equation_candidate"
            if not valid_sequences else "multiple_exact_equation_candidates"
        )
        audit["valid_candidate_count"] = len(valid_sequences)
        return output, audit

    selected = next(iter(valid_sequences))
    changes = []
    for index, (before, after) in enumerate(zip(baseline, selected, strict=True)):
        if before == after:
            continue
        row = ordered[index]
        candidates = [str(value) for value in row["final_topk"]]
        output[str(row["record_id"])] = after
        changes.append({
            "record_id": str(row["record_id"]),
            "before": before,
            "after": after,
            "candidate_rank": candidates.index(after) + 1,
        })
    if len(changes) > max_changed_glyphs:
        raise AssertionError("unique equation repair exceeded its edit contract")
    audit.update({
        "status": "changed_shadow_only",
        "reason": "unique_exact_equation_candidate",
        "valid_candidate_count": 1,
        "changes": changes,
    })
    return output, audit


def _unique_candidate_arithmetic_equation_shadow(
    rows: list[dict[str, Any]], predictions: dict[str, str], *,
    max_changed_glyphs: int = 3,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Complete a flat equation only when Top-5 yields one exact numeric reading.

    This deliberately explores unsupported baseline symbols only in the
    optional shadow; the accepted decoder result and product output stay intact.
    """
    from itertools import product

    from semantic_equation_guard_v1 import (
        ARITHMETIC_TOKENS, _flat, is_exact_arithmetic_equation,
    )

    output = dict(predictions)
    ordered = sorted(rows, key=lambda row: int(row["context"]["index"]))
    baseline = [output[str(row["record_id"])] for row in ordered]
    audit: dict[str, Any] = {
        "status": "skipped",
        "reason": "not_flat_single_equality_candidate_formula",
        "changes": [],
        "new_tokens": 0,
        "deleted_glyphs": 0,
        "grouping_mutations": 0,
        "candidate_sequences_checked": 0,
        "valid_candidate_count": 0,
        "max_changed_glyphs": int(max_changed_glyphs),
        "arithmetic_evaluation": True,
        "false_user_equation_risk": True,
    }
    if (
        isinstance(max_changed_glyphs, bool)
        or not isinstance(max_changed_glyphs, int)
        or not 1 <= max_changed_glyphs <= 3
    ):
        raise ValueError("max_changed_glyphs must be in [1, 3]")
    if (
        not 3 <= len(ordered) <= 12
        or not _flat(ordered)
        or baseline.count("=") != 1
    ):
        return output, audit

    unknown_positions = [
        index for index, token in enumerate(baseline)
        if token not in ARITHMETIC_TOKENS
    ]
    if not unknown_positions:
        audit["reason"] = "baseline_already_in_arithmetic_vocabulary"
        return output, audit
    if len(unknown_positions) > max_changed_glyphs:
        audit.update({
            "reason": "unsupported_symbol_count_exceeds_edit_budget",
            "unknown_positions": unknown_positions,
        })
        return output, audit

    options: dict[int, list[str]] = {}
    for index in unknown_positions:
        row = ordered[index]
        topk = [str(token) for token in row["final_topk"]]
        alternatives = [
            token for token in topk
            if token != baseline[index] and token in ARITHMETIC_TOKENS
        ]
        if not alternatives:
            audit.update({
                "reason": "unsupported_symbol_has_no_arithmetic_top5_candidate",
                "blocked_position": index,
                "blocked_token": baseline[index],
            })
            return output, audit
        options[index] = alternatives

    valid: set[tuple[str, ...]] = set()
    for replacements in product(*(options[index] for index in unknown_positions)):
        proposal = list(baseline)
        for index, token in zip(unknown_positions, replacements, strict=True):
            proposal[index] = token
        audit["candidate_sequences_checked"] += 1
        if is_exact_arithmetic_equation(tuple(proposal)):
            valid.add(tuple(proposal))
            if len(valid) > 1:
                break
    audit["valid_candidate_count"] = len(valid)
    if len(valid) != 1:
        audit["reason"] = (
            "no_exact_candidate_equation" if not valid
            else "ambiguous_exact_candidate_equations"
        )
        return output, audit

    selected = next(iter(valid))
    changes = []
    for index, (before, after) in enumerate(zip(baseline, selected, strict=True)):
        if before == after:
            continue
        row = ordered[index]
        topk = [str(token) for token in row["final_topk"]]
        probabilities = [float(value) for value in row["final_topk_probabilities"]]
        rank = topk.index(after)
        changes.append({
            "record_id": str(row["record_id"]),
            "before": before,
            "after": after,
            "candidate_rank": rank + 1,
            "candidate_probability": probabilities[rank],
            "selected_minus_top1_log_probability": (
                math.log(max(probabilities[rank], 1e-12))
                - math.log(max(probabilities[0], 1e-12))
            ),
        })
        output[str(row["record_id"])] = after
    if not 1 <= len(changes) <= max_changed_glyphs:
        raise AssertionError("unique candidate equation exceeded its edit budget")
    audit.update({
        "status": "changed_shadow_only",
        "reason": "unique_exact_top5_arithmetic_equation",
        "unknown_positions": unknown_positions,
        "changes": changes,
        "false_user_equation_risk": True,
    })
    return output, audit


def _terminal_rhs_bar_shadow(
    rows: list[dict[str, Any]], predictions: dict[str, str],
) -> tuple[dict[str, str], dict[str, Any]]:
    """Log a candidate-only challenger for a lone bar in the terminal RHS slot."""
    output = dict(predictions)
    ordered = sorted(rows, key=lambda row: int(row["context"]["index"]))
    baseline = [output[str(row["record_id"])] for row in ordered]
    audit: dict[str, Any] = {
        "status": "skipped",
        "reason": "not_single_terminal_rhs_bar",
        "changes": [],
        "new_tokens": 0,
        "deleted_glyphs": 0,
        "grouping_mutations": 0,
        "truth_evaluation": False,
        "user_intent_risk": True,
    }
    equalities = [index for index, token in enumerate(baseline) if token == "="]
    if (
        len(ordered) < 3
        or len(equalities) != 1
        or equalities[0] != len(ordered) - 2
        or baseline[-1] != "|"
        or baseline.count("|") != 1
    ):
        return output, audit
    row = ordered[-1]
    candidates = [str(token) for token in row["final_topk"]]
    if "1" not in candidates:
        audit["reason"] = "unit_candidate_missing_from_top5"
        return output, audit
    rank = candidates.index("1")
    probabilities = [float(value) for value in row["final_topk_probabilities"]]
    output[str(row["record_id"])] = "1"
    audit.update({
        "status": "changed_shadow_only",
        "reason": "unique_unit_candidate_in_terminal_rhs_slot",
        "changes": [{
            "record_id": str(row["record_id"]),
            "before": "|",
            "after": "1",
            "candidate_rank": rank + 1,
            "candidate_probability": probabilities[rank],
        }],
    })
    return output, audit


def _candidate_preserving_semantic_guard_shadow(
    formula_id: str, groups: list[list[int]], symbols: list[dict[str, Any]],
    decoder: dict[str, Any],
) -> dict[str, Any]:
    """Apply frozen deterministic guards to a copy of an accepted Top-5 decode.

    The result is an optional diagnostic only. It never changes the decoder,
    serialized formula, selected groups, or runtime defaults.
    """
    if not decoder.get("accepted"):
        return {
            "schema": SEMANTIC_GUARD_SHADOW_SCHEMA,
            "status": "skipped",
            "reason": "decoder_not_accepted",
            "product_default_enabled": False,
        }
    tokens = [str(token) for token in decoder.get("tokens") or []]
    if len(groups) != len(symbols) or len(tokens) != len(groups):
        raise AssertionError("semantic guard shadow group/symbol/token count mismatch")
    symbols_by_group = {
        tuple(sorted(int(index) for index in symbol.get("stroke_indices") or [])): symbol
        for symbol in symbols
    }
    if len(symbols_by_group) != len(groups):
        raise AssertionError("semantic guard shadow has duplicate or missing groups")

    rows = []
    ordered_groups = []
    for index, (group, token) in enumerate(zip(groups, tokens, strict=True)):
        key = tuple(sorted(int(value) for value in group))
        symbol = symbols_by_group.get(key)
        if symbol is None:
            raise AssertionError("semantic guard shadow group has no HWR symbol")
        topk = [str(value) for value in symbol.get("hwr_topk") or []]
        probabilities = [float(value) for value in symbol.get("hwr_topk_probabilities") or []]
        geometry = dict(symbol.get("geometry") or {})
        if len(topk) != 5 or len(set(topk)) != 5 or len(probabilities) != 5:
            raise AssertionError("semantic guard shadow requires the frozen HWR Top-5")
        if token not in topk:
            raise AssertionError("base decoder token is outside the HWR Top-5")
        try:
            left, top, right, bottom = (
                float(geometry[name]) for name in ("left", "top", "right", "bottom")
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("semantic guard shadow requires a finite symbol box") from error
        if not all(math.isfinite(value) for value in (left, top, right, bottom)):
            raise ValueError("semantic guard shadow requires a finite symbol box")
        if right < left or bottom < top:
            raise ValueError("semantic guard shadow received an inverted symbol box")
        rows.append({
            "record_id": f"{formula_id}:{index}",
            "formula_id": formula_id,
            "final_topk": topk,
            "final_topk_probabilities": probabilities,
            "context": {"index": index, "length": len(groups)},
            "geometry": {
                "left": left, "top": top, "right": right, "bottom": bottom,
                "center_x": (left + right) / 2.0,
                "center_y": (top + bottom) / 2.0,
                "width_rel": max(right - left, 1e-6),
                "height_rel": max(bottom - top, 1e-6),
            },
        })
        ordered_groups.append(list(key))

    # The semantic guards consume geometry-ordered formula rows, not stroke
    # arrival order. This only changes context metadata on copies of these rows.
    from formula_layout_v1 import recontextualize_formula_rows
    from semantic_equation_guard_v1 import (
        apply_semantic_equation_guard,
        apply_semantic_expression_guard,
    )
    from semantic_fence_guard_v1 import apply_semantic_fence_guard
    from semantic_infix_guard_v1 import apply_semantic_infix_guard

    contextual_rows, layout_audit = recontextualize_formula_rows(rows)
    record_order = [str(row["record_id"]) for row in rows]
    predictions = {
        record_id: token for record_id, token in zip(record_order, tokens, strict=True)
    }
    after_fence, fence_audit = apply_semantic_fence_guard(contextual_rows, predictions)
    after_infix, infix_audit = apply_semantic_infix_guard(
        contextual_rows, after_fence, minimum_probability_ratio=0.0,
    )
    after_equation, equation_audit = apply_semantic_equation_guard(
        contextual_rows, after_infix,
    )
    after_expression, expression_audit = apply_semantic_expression_guard(
        contextual_rows, after_equation,
    )
    after_unpaired_bar, unpaired_bar_audit = _unique_unpaired_bar_arithmetic_shadow(
        contextual_rows, after_expression,
    )
    after_unique_equation, unique_equation_audit = _unique_exact_equation_candidates_shadow(
        contextual_rows, after_unpaired_bar,
    )
    after_boundary_bar, boundary_bar_audit = _boundary_bar_as_unit_shadow(
        contextual_rows, after_unique_equation,
    )
    after_unique_candidate_equation, unique_candidate_equation_audit = (
        _unique_candidate_arithmetic_equation_shadow(contextual_rows, after_boundary_bar)
    )
    after_terminal_rhs_bar, terminal_rhs_bar_audit = _terminal_rhs_bar_shadow(
        contextual_rows, after_unique_candidate_equation,
    )
    prediction_stages = {
        "decoder": predictions,
        "after_fence_guard": after_fence,
        "after_infix_guard": after_infix,
        "after_equation_guard": after_equation,
        "after_arithmetic_expression_guard": after_expression,
        "after_unique_bar_equation_guard": after_unpaired_bar,
        "after_unique_exact_equation_candidates_guard": after_unique_equation,
        "after_boundary_bar_as_unit_guard": after_boundary_bar,
        "after_unique_candidate_arithmetic_equation_guard": after_unique_candidate_equation,
        "after_terminal_rhs_bar_guard": after_terminal_rhs_bar,
    }
    rows_by_id = {str(row["record_id"]): row for row in contextual_rows}
    stage_rows = {
        stage: [
            {
                "stroke_indices": group,
                "token": str(stage_predictions[record_id]),
            }
            for group, record_id in zip(ordered_groups, record_order, strict=True)
        ]
        for stage, stage_predictions in prediction_stages.items()
    }
    candidate_preserved = all(
        str(stage_predictions[record_id]) in rows_by_id[record_id]["final_topk"]
        for stage_predictions in prediction_stages.values()
        for record_id in record_order
    )
    if not candidate_preserved:
        raise AssertionError("semantic guard shadow escaped the HWR Top-5")
    return {
        "schema": SEMANTIC_GUARD_SHADOW_SCHEMA,
        "status": "applied_shadow_only",
        "scope": "frozen Top-5; geometry-ordered; no HWR, group, or product mutation",
        "base_decoder_tokens": tokens,
        "stages": stage_rows,
        "audits": {
            "layout": layout_audit,
            "fence": fence_audit,
            "infix": infix_audit,
            "equation": equation_audit,
            "arithmetic_expression": expression_audit,
            "unique_unpaired_bar_equation": unpaired_bar_audit,
            "unique_exact_equation_candidates": unique_equation_audit,
            "boundary_bar_as_unit": boundary_bar_audit,
            "unique_candidate_arithmetic_equation": unique_candidate_equation_audit,
            "terminal_rhs_bar": terminal_rhs_bar_audit,
        },
        "candidate_preservation": True,
        "new_tokens": 0,
        "deleted_glyphs": 0,
        "grouping_mutations": 0,
        "product_default_enabled": False,
    }


def _device(name: str) -> torch.device:
    resolved = "cuda" if name == "auto" and torch.cuda.is_available() else (
        "cpu" if name == "auto" else name
    )
    if resolved == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    return torch.device(resolved)


def _validate_strokes(source: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    formula_id = str(source.get("formula_id") or source.get("sample_id") or "").strip()
    if not formula_id:
        raise ValueError("raw formula requires formula_id or sample_id")
    strokes = list(source.get("strokes") or [])
    if not strokes:
        raise ValueError(f"raw formula has no strokes: {formula_id}")
    ordered = sorted(strokes, key=lambda row: int(row.get("order", -1)))
    orders = [int(row.get("order", -1)) for row in ordered]
    if orders != list(range(len(ordered))):
        raise ValueError(
            f"stroke orders must be unique contiguous zero-based indices: {formula_id}"
        )
    for stroke in ordered:
        points = list(stroke.get("points") or [])
        if not points:
            raise ValueError(f"empty stroke in formula: {formula_id}")
        previous_time = -math.inf
        for point in points:
            x = float(point["x"]); y = float(point["y"])
            time = float(point.get("t_ms", 0.0))
            if not all(math.isfinite(value) for value in (x, y, time)):
                raise ValueError(f"non-finite stroke point in formula: {formula_id}")
            if time < previous_time:
                raise ValueError(f"stroke timestamps must be monotonic: {formula_id}")
            previous_time = time
    return formula_id, ordered


def _load_inputs(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(payload, dict):
        payload = payload.get("formulas", [payload])
    if not isinstance(payload, list) or not payload or any(not isinstance(row, dict) for row in payload):
        raise ValueError("raw input must be one formula, a formula list, or JSONL")
    return payload


def _apply_singleton_finalized_rows(
    runtime_rows: list[dict], finalized: list[dict], auxiliary_rows: list[dict],
    configuration: dict,
) -> tuple[list[dict], dict]:
    predictions = {
        str(row["record_id"]): str(row["finalized_top1"])
        for row in finalized
    }
    rescued, audit = apply_singleton_shape_rescue(
        runtime_rows, predictions, auxiliary_rows, configuration,
    )
    hwr_top1 = {
        str(row["record_id"]): str(row["final_topk"][0])
        for row in runtime_rows
    }
    output = []
    for source in finalized:
        record_id = str(source["record_id"])
        token = str(rescued[record_id])
        changed_by_rescue = token != str(source["finalized_top1"])
        output.append({
            **source,
            "finalized_top1": token,
            "changed": token != hwr_top1[record_id],
            "decision_source": (
                "singleton_shape_rescue"
                if changed_by_rescue
                else source.get("decision_source", "formula_context_finalizer")
            ),
        })
    return output, audit


def _formula_layout_shadow(
    runtime_rows: list[dict], finalized: list[dict],
) -> tuple[dict, dict]:
    predictions = {
        str(row["record_id"]): str(row["finalized_top1"])
        for row in finalized
    }
    if len(predictions) != len(finalized):
        raise ValueError("formula layout finalized record ids must be unique")
    layout_rows, evidence_audit = selected_layout_evidence_rows(runtime_rows, predictions)
    formulae, audit = finalize_formula_outputs(layout_rows, predictions)
    if len(formulae) != 1:
        raise AssertionError("raw runtime must emit exactly one layout formula")
    return formulae[0], {
        **audit,
        "enabled": True,
        **evidence_audit,
        "feedback_into_character_model": False,
        "product_default_enabled": False,
    }


@dataclass(frozen=True)
class RawFormulaContextRuntimeV1:
    ranker_payload: dict[str, Any]
    hwr: Any
    labels: list[str]
    finalizer: OwnedFormulaContextFinalizer
    device: torch.device
    partition_ranker_sha256: str
    singleton_hwr: Any | None
    singleton_configuration: dict[str, Any] | None
    numeric_configuration: dict[str, Any] | None
    wide_syntax_configuration: dict[str, Any] | None
    relation_merge_configuration: dict[str, Any] | None
    cross_merge_configuration: dict[str, Any] | None
    repeat_merge_configuration: dict[str, Any] | None
    placement_configuration: dict[str, Any] | None
    equality_configuration: dict[str, Any] | None
    latin_hwr: Any | None
    latin_labels: list[str] | None
    latin_configuration: dict[str, Any] | None
    latin_hwr_sha256: str | None
    singleton_config_sha256: str | None
    singleton_hwr_sha256: str | None
    pairwise_shape_expert: dict[str, Any] | None
    pairwise_shape_configuration: dict[str, Any] | None
    pairwise_shape_expert_sha256: str | None
    pairwise_shape_config_sha256: str | None
    emit_formula_layout_shadow: bool

    @classmethod
    def from_artifacts(
        cls, partition_ranker: Path, hwr_checkpoint: Path, context_checkpoint: Path,
        *, device: str = "auto", batch_size: int = 128,
        formula_sequence_config: Path | None = None,
        formula_syntax_rescue_config: Path | None = None,
        candidate_context_fusion_config: Path | None = None,
        candidate_context_auxiliary_hwr_checkpoint: Path | None = None,
        formula_placement_config: Path | None = None,
        straight_equality_config: Path | None = None,
        latin_auxiliary_checkpoint: Path | None = None,
        pairwise_shape_expert: Path | None = None,
        pairwise_shape_config: Path | None = None,
        emit_formula_layout_shadow: bool = False,
        allow_posthoc_shadow: bool = False,
    ) -> "RawFormulaContextRuntimeV1":
        ranker_path = Path(partition_ranker).expanduser().resolve()
        hwr_path = Path(hwr_checkpoint).expanduser().resolve()
        context_path = Path(context_checkpoint).expanduser().resolve()
        for path in (ranker_path, hwr_path, context_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        payload = joblib.load(ranker_path)
        if (
            payload.get("schema") != RANKER_SCHEMA
            or tuple(payload.get("feature_names") or ()) != FEATURE_NAMES
            or "model" not in payload
            or "grouping_model" not in payload
        ):
            raise ValueError("partition-context ranker artifact contract mismatch")
        if payload.get("requires_explicit_shadow_opt_in") is not True:
            raise ValueError("partition-context artifact lacks explicit shadow contract")
        if not allow_posthoc_shadow:
            raise ValueError(
                "posthoc partition gate is development-only; explicit shadow opt-in is required"
            )
        if payload.get("product_default_enabled") is not False:
            raise ValueError("partition-context product gate must remain disabled")
        if payload.get("posthoc_test_tuning") is not True:
            raise ValueError("partition-context tuning provenance is missing")
        if payload.get("evaluation_only_writer_loo") is not False:
            raise ValueError("writer-LOO evaluation artifact is not a deployable runtime")
        if dict(payload.get("selection_guard") or {}) != POSTHOC_DESIGN_GUARD:
            raise ValueError("partition-context selection guard mismatch")
        if _sha256(hwr_path) != str(payload.get("hwr_checkpoint_sha256")):
            raise ValueError("partition-context HWR checkpoint hash mismatch")
        if _sha256(context_path) != str(payload.get("context_checkpoint_sha256")):
            raise ValueError("partition-context context checkpoint hash mismatch")
        finalizer_contract = dict(payload.get("finalizer_contract") or {})
        expected_finalizer = {
            "formula_layout": True,
            "semantic_guards": True,
            "equation_correction": False,
        }
        if any(finalizer_contract.get(key) != value for key, value in expected_finalizer.items()):
            raise ValueError("partition-context finalizer contract mismatch")
        if dict(finalizer_contract.get("auxiliary_candidates") or {}).get("enabled") is not False:
            raise ValueError("unimplemented auxiliary candidate contract is not runtime admissible")
        sequence_path = (
            Path(formula_sequence_config).expanduser().resolve()
            if formula_sequence_config is not None else None
        )
        syntax_path = (
            Path(formula_syntax_rescue_config).expanduser().resolve()
            if formula_syntax_rescue_config is not None else None
        )
        for name, path, hash_field in (
            ("formula sequence", sequence_path, "formula_sequence_config_sha256"),
            ("formula syntax rescue", syntax_path, "formula_syntax_rescue_config_sha256"),
        ):
            expected_hash = finalizer_contract.get(hash_field)
            if bool(path) != bool(expected_hash):
                raise ValueError(f"{name} config presence does not match ranker artifact")
            if path is not None and (not path.is_file() or _sha256(path) != expected_hash):
                raise ValueError(f"{name} config hash mismatch")
        resolved_device = _device(device)
        hwr, labels, _ = _load_model(hwr_path, resolved_device)
        if bool(pairwise_shape_expert) != bool(pairwise_shape_config):
            raise ValueError(
                "pairwise shape expert and configuration are required together"
            )
        if bool(candidate_context_fusion_config) != bool(candidate_context_auxiliary_hwr_checkpoint):
            raise ValueError(
                "candidate context fusion config and auxiliary HWR checkpoint are required together"
            )
        if bool(formula_placement_config) != bool(straight_equality_config):
            raise ValueError(
                "formula placement and straight equality configs are required together"
            )
        if formula_placement_config is not None and candidate_context_fusion_config is None:
            raise ValueError("formula placement requires candidate context fusion")
        singleton_hwr = None
        singleton_configuration = None
        numeric_configuration = None
        wide_syntax_configuration = None
        relation_merge_configuration = None
        cross_merge_configuration = None
        repeat_merge_configuration = None
        placement_configuration = None
        equality_configuration = None
        latin_hwr = None
        latin_labels = None
        latin_configuration = None
        latin_hwr_sha256 = None
        singleton_config_sha256 = None
        singleton_hwr_sha256 = None
        pairwise_expert_payload = None
        pairwise_configuration = None
        pairwise_expert_sha256 = None
        pairwise_config_sha256 = None
        if pairwise_shape_expert is not None:
            (
                pairwise_expert_payload,
                pairwise_configuration,
                pairwise_expert_sha256,
                pairwise_config_sha256,
            ) = load_pairwise_shape_artifacts(
                pairwise_shape_expert,
                pairwise_shape_config,
                hwr_checkpoint_sha256=_sha256(hwr_path),
            )
        if candidate_context_fusion_config is not None:
            singleton_config_path = Path(candidate_context_fusion_config).expanduser().resolve()
            singleton_hwr_path = Path(
                candidate_context_auxiliary_hwr_checkpoint,
            ).expanduser().resolve()
            for path in (singleton_config_path, singleton_hwr_path):
                if not path.is_file():
                    raise FileNotFoundError(path)
            singleton_payload = json.loads(
                singleton_config_path.read_text(encoding="utf-8")
            )
            gate = dict(singleton_payload.get("gate") or {})
            provenance = dict(singleton_payload.get("provenance") or {})
            artifacts = dict(singleton_payload.get("artifacts") or {})
            mode = dict(singleton_payload.get("mode") or {})
            rerank_mode = dict(mode.get("restricted_candidate_rerank") or {})
            singleton_mode = dict(mode.get("singleton_shape_rescue") or {})
            numeric_mode = dict(mode.get("dual_numeric_rescue") or {})
            wide_syntax_mode = dict(mode.get("wide_numeric_syntax_rescue") or {})
            relation_merge_mode = dict(mode.get("relation_merge_rescue") or {})
            cross_merge_mode = dict(mode.get("cross_merge_rescue") or {})
            repeat_merge_mode = dict(mode.get("repeat_merge_rescue") or {})
            placement_mode = dict(mode.get("formula_placement_rescue") or {})
            equality_mode = dict(mode.get("straight_equality_rescue") or {})
            latin_mode = dict(mode.get("latin_t_context_rescue") or {})
            legacy_singleton_mode = {
                "enabled": True,
                "preserve_original_candidate_set": True,
            }
            ranked_singleton_mode = {
                **legacy_singleton_mode,
                "token_candidate_maximum_ranks": {"/": 1, r"\times": 2},
            }
            if (
                singleton_payload.get("schema") != CANDIDATE_FUSION_CONFIG_SCHEMA
                or singleton_payload.get("rescue_schema") != SINGLETON_SCHEMA
                or gate.get("shadow_runtime_admitted") is not True
                or gate.get("product_default_admitted") is not False
                or provenance.get("current_96_formula_training_overlap") is not True
                or rerank_mode != {
                    "enabled": True,
                    "candidate_width": 5,
                    "preserve_original_candidate_set": True,
                    "reapply_formula_context_finalizer": True,
                }
                or singleton_mode not in (
                    legacy_singleton_mode, ranked_singleton_mode,
                )
                or numeric_mode != {
                    "enabled": True,
                    "maximum_operand_changes": 2,
                    "candidate_contract": "baseline_and_auxiliary_top5_union",
                    "operator_relation_or_fence_mutations": 0,
                    "arithmetic_evaluation": False,
                }
                or wide_syntax_mode != {
                    "enabled": True,
                    "candidate_width": 10,
                    "maximum_changes": 2,
                    "candidate_contract": "baseline_and_auxiliary_top10_union",
                    "minimum_digit_probability_ratio": 0.005,
                    "allow_equality_replacement": False,
                    "arithmetic_evaluation": False,
                }
                or relation_merge_mode != {
                    "enabled": True,
                    "maximum_candidate_rank": 3,
                    "exact_cover_coarsening_only": True,
                    "required_merged_strokes": 3,
                    "negated_relation_candidate_required": True,
                    "target_label_or_glyph_count_input": False,
                    "arithmetic_evaluation": False,
                }
            ):
                raise ValueError("candidate context fusion shadow contract mismatch")
            cross_payload = singleton_payload.get("cross_merge_configuration")
            if bool(cross_payload) != bool(cross_merge_mode):
                raise ValueError("candidate context cross merge presence mismatch")
            if cross_payload:
                cross_merge_configuration = _validate_cross_merge_configuration(
                    dict(cross_payload),
                )
                legacy_cross_mode = {
                    "enabled": True,
                    "maximum_candidate_rank": 3,
                    "exact_cover_coarsening_only": True,
                    "required_merged_strokes": 2,
                    "merged_hwr_and_context_token": "x",
                    "preserve_merged_x_after_auxiliary_fusion": True,
                    "target_label_or_glyph_count_input": False,
                    "arithmetic_evaluation": False,
                }
                extended_cross_mode = {
                    **legacy_cross_mode,
                    "auxiliary_nested_expression_enabled": True,
                    "auxiliary_candidate_contract": "product_fused_top20",
                    "auxiliary_x_candidate_maximum_rank": 3,
                    "auxiliary_open_fence_candidate_maximum_rank": 1,
                    "auxiliary_plus_geometry_source": "formula_role_typo",
                }
                if cross_merge_mode not in (
                    legacy_cross_mode, extended_cross_mode,
                ):
                    raise ValueError("candidate context cross merge mode mismatch")
                if (
                    cross_merge_configuration["maximum_candidate_rank"]
                    != cross_merge_mode["maximum_candidate_rank"]
                    or cross_merge_configuration["required_merged_strokes"]
                    != cross_merge_mode["required_merged_strokes"]
                    or cross_merge_configuration[
                        "preserve_merged_x_after_auxiliary_fusion"
                    ] is not True
                    or cross_merge_configuration[
                        "auxiliary_nested_expression_enabled"
                    ] != (cross_merge_mode == extended_cross_mode)
                    or (
                        cross_merge_mode == extended_cross_mode
                        and (
                            cross_merge_configuration[
                                "auxiliary_x_candidate_maximum_rank"
                            ] != cross_merge_mode[
                                "auxiliary_x_candidate_maximum_rank"
                            ]
                            or cross_merge_configuration[
                                "auxiliary_open_fence_candidate_maximum_rank"
                            ] != cross_merge_mode[
                                "auxiliary_open_fence_candidate_maximum_rank"
                            ]
                        )
                    )
                ):
                    raise ValueError("candidate context cross merge configuration mismatch")
            repeat_payload = singleton_payload.get("repeat_merge_configuration")
            if bool(repeat_payload) != bool(repeat_merge_mode):
                raise ValueError("candidate context repeat merge presence mismatch")
            if repeat_payload:
                repeat_merge_configuration = _validate_repeat_merge_configuration(
                    dict(repeat_payload),
                )
                if repeat_merge_mode != {
                    "enabled": True,
                    "maximum_candidate_rank": 3,
                    "exact_cover_coarsening_only": True,
                    "required_merged_strokes": 2,
                    "required_formula_glyphs": 3,
                    "same_formula_repeated_shape_dtw_required": True,
                    "hwr_context_agreement_required": True,
                    "preserve_merged_token_after_auxiliary_fusion": True,
                    "target_label_or_glyph_count_input": False,
                    "arithmetic_evaluation": False,
                }:
                    raise ValueError("candidate context repeat merge mode mismatch")
                if (
                    repeat_merge_configuration["maximum_candidate_rank"]
                    != repeat_merge_mode["maximum_candidate_rank"]
                    or repeat_merge_configuration["required_merged_strokes"]
                    != repeat_merge_mode["required_merged_strokes"]
                    or repeat_merge_configuration["required_formula_glyphs"]
                    != repeat_merge_mode["required_formula_glyphs"]
                    or repeat_merge_configuration[
                        "preserve_merged_token_after_auxiliary_fusion"
                    ] is not True
                ):
                    raise ValueError("candidate context repeat merge configuration mismatch")
            placement_paths_present = (
                formula_placement_config is not None
                and straight_equality_config is not None
            )
            if placement_paths_present != bool(placement_mode and equality_mode):
                raise ValueError("candidate context placement config presence mismatch")
            placement_path = equality_path = None
            if placement_paths_present:
                placement_path = Path(formula_placement_config).expanduser().resolve()
                equality_path = Path(straight_equality_config).expanduser().resolve()
                for path in (placement_path, equality_path):
                    if not path.is_file():
                        raise FileNotFoundError(path)
                placement_raw, _ = _shadow_configuration(
                    placement_path, config_schema=PLACEMENT_CONFIG_SCHEMA,
                    rescue_schema=PLACEMENT_SCHEMA,
                )
                equality_raw, _ = _shadow_configuration(
                    equality_path, config_schema=EQUALITY_CONFIG_SCHEMA,
                    rescue_schema=EQUALITY_SCHEMA,
                )
                placement_configuration = validate_placement_configuration(
                    placement_raw,
                )
                equality_configuration = validate_equality_configuration(
                    equality_raw,
                )
                if (
                    cross_merge_configuration is not None
                    and cross_merge_configuration[
                        "auxiliary_nested_expression_enabled"
                    ]
                    and placement_configuration["formula_role_typo"][
                        "value_cross_plus_open_fence_enabled"
                    ] is not True
                ):
                    raise ValueError(
                        "nested cross merge requires open-fence plus placement"
                    )
                if placement_mode != {
                    "enabled": True,
                    "candidate_width": 20,
                    "candidate_contract": "product_fused_top20",
                    "insertions_or_deletions": 0,
                    "arithmetic_evaluation": False,
                } or equality_mode != {
                    "enabled": True,
                    "candidate_width": 20,
                    "candidate_contract": "product_fused_top20",
                    "maximum_changes_per_formula": 1,
                    "arithmetic_evaluation": False,
                }:
                    raise ValueError("candidate context placement mode mismatch")
            latin_payload = singleton_payload.get("latin_t_context_configuration")
            if bool(latin_payload) != bool(latin_mode):
                raise ValueError("candidate context Latin t rescue presence mismatch")
            if bool(latin_payload) != bool(latin_auxiliary_checkpoint):
                raise ValueError(
                    "candidate context Latin t config and checkpoint are required together"
                )
            latin_path = None
            if latin_payload:
                latin_configuration = validate_latin_t_configuration(
                    dict(latin_payload),
                )
                if latin_mode != {
                    "enabled": True,
                    "candidate_width": 5,
                    "candidate_contract": "approved_legacy_latin_auxiliary_top5",
                    "target_token": "t",
                    "single_parenthesized_function_argument_only": True,
                    "insertions_or_deletions": 0,
                    "target_label_or_glyph_count_input": False,
                    "arithmetic_evaluation": False,
                }:
                    raise ValueError("candidate context Latin t mode mismatch")
                latin_path = Path(latin_auxiliary_checkpoint).expanduser().resolve()
                if not latin_path.is_file():
                    raise FileNotFoundError(latin_path)
                latin_hwr, latin_labels = load_latin_auxiliary_model(
                    latin_path, resolved_device,
                )
                latin_hwr_sha256 = _sha256(latin_path)
            expected_hashes = {
                "partition_ranker_sha256": _sha256(ranker_path),
                "hwr_checkpoint_sha256": _sha256(hwr_path),
                "context_checkpoint_sha256": _sha256(context_path),
                "auxiliary_hwr_checkpoint_sha256": _sha256(singleton_hwr_path),
            }
            if placement_paths_present:
                expected_hashes.update({
                    "formula_placement_config_sha256": _sha256(placement_path),
                    "straight_equality_config_sha256": _sha256(equality_path),
                })
            if latin_path is not None:
                expected_hashes["latin_auxiliary_checkpoint_sha256"] = _sha256(
                    latin_path,
                )
            if any(artifacts.get(key) != value for key, value in expected_hashes.items()):
                raise ValueError("candidate context fusion artifact hash mismatch")
            evidence = dict(singleton_payload.get("evidence") or {})
            for name in ("selected_summary", "evaluation_evidence"):
                evidence_path = singleton_config_path.parent / str(
                    evidence.get(name, ""),
                )
                if (
                    not evidence_path.is_file()
                    or _sha256(evidence_path) != evidence.get(f"{name}_sha256")
                ):
                    raise ValueError(
                        f"candidate context fusion {name} hash mismatch"
                    )
            singleton_configuration = validate_singleton_configuration(
                dict(singleton_payload.get("configuration") or {}),
            )
            expected_singleton_ranks = singleton_mode.get(
                "token_candidate_maximum_ranks", {"/": 1, r"\times": 1},
            )
            if (
                singleton_configuration["token_candidate_maximum_ranks"]
                != expected_singleton_ranks
            ):
                raise ValueError("candidate context singleton rank mode mismatch")
            numeric_configuration = validate_dual_numeric_configuration(
                dict(singleton_payload.get("numeric_configuration") or {}),
            )
            wide_syntax_configuration = validate_wide_syntax_configuration(
                dict(singleton_payload.get("wide_syntax_configuration") or {}),
            )
            relation_merge_configuration = _validate_relation_merge_configuration(
                dict(singleton_payload.get("relation_merge_configuration") or {}),
            )
            if (
                relation_merge_configuration["maximum_candidate_rank"]
                != relation_merge_mode["maximum_candidate_rank"]
                or relation_merge_configuration["required_merged_strokes"]
                != relation_merge_mode["required_merged_strokes"]
            ):
                raise ValueError("candidate context relation merge mode mismatch")
            if singleton_configuration["auxiliary_policy"] != "old_new_product_probability_fusion":
                raise ValueError("candidate context fusion policy mismatch")
            if (
                numeric_configuration["auxiliary_policy"]
                != singleton_configuration["auxiliary_policy"]
                or numeric_configuration["auxiliary_weight"]
                != singleton_configuration["auxiliary_weight"]
            ):
                raise ValueError("candidate context fusion numeric policy mismatch")
            if (
                wide_syntax_configuration["allowed_auxiliary_policies"]
                != [singleton_configuration["auxiliary_policy"]]
                or wide_syntax_configuration["auxiliary_weight"]
                != singleton_configuration["auxiliary_weight"]
                or wide_syntax_configuration["candidate_width"] != 10
            ):
                raise ValueError("candidate context fusion wide syntax policy mismatch")
            if placement_configuration is not None and (
                placement_configuration["unmatched_fence_operand"]["auxiliary_weight"]
                != singleton_configuration["auxiliary_weight"]
                or equality_configuration["auxiliary_weight"]
                != singleton_configuration["auxiliary_weight"]
            ):
                raise ValueError("candidate context placement fusion policy mismatch")
            singleton_hwr, singleton_labels, _ = _load_model(
                singleton_hwr_path, resolved_device,
            )
            if singleton_labels != labels:
                raise ValueError("singleton auxiliary HWR vocabulary mismatch")
            singleton_state = singleton_hwr.state_dict()
            for name, value in hwr.state_dict().items():
                if not name.startswith("math_head.") and not torch.equal(
                    value.detach().cpu(), singleton_state[name].detach().cpu()
                ):
                    raise ValueError(f"singleton auxiliary and frozen HWR encoders differ: {name}")
            singleton_config_sha256 = _sha256(singleton_config_path)
            singleton_hwr_sha256 = _sha256(singleton_hwr_path)
        finalizer = OwnedFormulaContextFinalizer(
            context_path, hwr_path, device=str(resolved_device), batch_size=batch_size,
            semantic_guards=True, equation_correction=False, formula_layout=True,
            formula_sequence_config=sequence_path,
            formula_syntax_rescue_config=syntax_path,
        )
        return cls(
            payload, hwr, labels, finalizer, resolved_device, _sha256(ranker_path),
            singleton_hwr, singleton_configuration, numeric_configuration,
            wide_syntax_configuration, relation_merge_configuration,
            cross_merge_configuration, repeat_merge_configuration,
            placement_configuration,
            equality_configuration,
            latin_hwr, latin_labels, latin_configuration, latin_hwr_sha256,
            singleton_config_sha256,
            singleton_hwr_sha256,
            pairwise_expert_payload,
            pairwise_configuration,
            pairwise_expert_sha256,
            pairwise_config_sha256,
            bool(emit_formula_layout_shadow),
        )

    def _sample(self, source: dict[str, Any]) -> Sample:
        formula_id, strokes = _validate_strokes(source)
        lattice_config = dict(self.ranker_payload["lattice_config"])
        candidates = build_lattice(strokes, **lattice_config)
        return Sample(
            formula_id, "runtime", strokes, (), candidates,
            candidate_features(candidates, strokes),
        )

    def selective_grouping_preview(
        self, source: dict[str, Any], *, config: Selective2DConfigV1 = Selective2DConfigV1(),
        symbol_margins: dict[frozenset[int], float] | None = None,
        include_hwr: bool = True,
        joint_hwr: bool = False,
        max_joint_hwr_candidates: int = 64,
        joint_geometry_prior_weight: float = 0.0,
        joint_geometry_prior_normalization: str = "stroke_count",
        allow_local: bool = True,
        include_semantic_guard_shadow: bool = False,
    ) -> dict[str, Any]:
        """Return a bounded Fast-vs-local-2D grouping tournament.

        This is an explicit development-only seam.  It reuses the frozen
        grouping model but never changes HWR weights, Top-k vocabulary, raw
        strokes, or the legacy ``infer`` output.  A mobile host calls this
        before deciding which local candidates deserve HWR encoding.

        ``allow_local=False`` returns the Fast A/B arm without running the
        local search. ``joint_hwr`` is an opt-in challenger: among geometry-shortlisted local
        partitions, it compares the existing Fast incumbent with candidates
        scored by the same HWR Top-5/strict-structure decoder.  It only encodes
        previously unseen local group candidates, with a hard per-formula cap.
        ``joint_geometry_prior_weight`` optionally adds the frozen grouping
        logit sum normalized by raw stroke count; it defaults to zero and is
        only intended for shadow comparisons.
        ``joint_geometry_prior_normalization`` is a shadow-only ablation between
        raw-stroke and selected-group denominators; the product default remains
        unchanged and uses ``stroke_count``.
        ``include_semantic_guard_shadow`` runs candidate-preserving semantic
        guards as a diagnostic copy, including narrow flat-arithmetic and
        boundary single-bar rules; it never changes this preview's decoder,
        LaTeX, groups, or default behavior.
        This remains shadow-only and does not alter ``infer`` or defaults.
        """
        if include_semantic_guard_shadow and not include_hwr:
            raise ValueError("semantic_guard_shadow requires include_hwr=True")
        if not 1 <= max_joint_hwr_candidates <= 64:
            raise ValueError("max_joint_hwr_candidates must be in [1, 64]")
        if joint_hwr and not include_hwr:
            raise ValueError("joint_hwr requires include_hwr=True")
        if not math.isfinite(float(joint_geometry_prior_weight)) or joint_geometry_prior_weight < 0:
            raise ValueError("joint_geometry_prior_weight must be finite and non-negative")
        if joint_geometry_prior_weight and not joint_hwr:
            raise ValueError("joint_geometry_prior_weight requires joint_hwr=True")
        if joint_geometry_prior_normalization not in {"stroke_count", "group_count"}:
            raise ValueError("joint_geometry_prior_normalization must be stroke_count or group_count")
        if joint_geometry_prior_normalization != "stroke_count" and not joint_hwr:
            raise ValueError("non-default geometry-prior normalization requires joint_hwr=True")
        formula_id, strokes = _validate_strokes(source)
        grouping_model = self.ranker_payload["grouping_model"]

        def score(candidate: dict[str, Any], source_strokes: list[dict[str, Any]]) -> float:
            features = candidate_features([candidate], source_strokes)
            probability = float(grouping_model.predict_proba(features)[0, 1])
            probability = min(max(probability, 1e-6), 1.0 - 1e-6)
            return math.log(probability / (1.0 - probability))

        def score_batch(candidates: list[dict[str, Any]], source_strokes: list[dict[str, Any]]) -> list[float]:
            probability = grouping_model.predict_proba(candidate_features(candidates, source_strokes))[:, 1]
            probability = np.clip(probability, 1e-6, 1.0 - 1e-6)
            return np.log(probability / (1.0 - probability)).tolist()

        fast_tournament = Selective2DAnytimeSolverV1(
            config=config, group_scorer=score, group_score_batch=score_batch,
        )
        fast_result = fast_tournament.solve(strokes, allow_local=False)
        result = fast_result
        hwr_symbols: list[dict[str, Any]] = []
        hwr_encoded = 0
        joint_hwr_encoded = 0
        joint_hwr_budget_exhausted = False
        joint_hwr_partition_diagnostics = {
            "partitions_evaluated": 0,
            "candidate_row_missing": 0,
            "candidate_budget_rejected": 0,
            "decoder_rejected": 0,
            "decoder_rejection_reasons": {},
            "decoder_accepted": 0,
            "cached_group_uses": 0,
            "geometry_prior_score_sum": 0.0,
        }
        joint_score_breakdowns: dict[tuple[tuple[int, ...], ...], dict[str, float]] = {}
        if include_hwr:
            symbol_cache: dict[tuple[int, ...], dict[str, Any]] = {}

            def classify(rows: list[dict[str, Any]]) -> tuple[dict[frozenset[int], float], list[dict[str, Any]]]:
                nonlocal hwr_encoded
                sample = Sample(
                    formula_id, "runtime_selective", strokes, (), rows,
                    candidate_features(rows, strokes),
                )
                embeddings, slices = _candidate_embeddings([sample], self.hwr, self.device)
                probability = _probabilities(self.hwr.math_head, embeddings, self.device)[slices[formula_id]]
                hwr_encoded += len(rows)
                margins: dict[frozenset[int], float] = {}
                symbols: list[dict[str, Any]] = []
                for row, values in zip(rows, probability, strict=True):
                    rank = np.argsort(-values)[:5]
                    group = frozenset(int(value) for value in row["source_indices"])
                    margins[group] = float(values[rank[0]] - values[rank[1]])
                    box = dict(row["box"])
                    symbols.append({
                        "stroke_indices": sorted(group), "hwr_topk": [self.labels[int(value)] for value in rank],
                        "hwr_topk_probabilities": [float(values[int(value)]) for value in rank],
                        "hwr_margin": margins[group], "geometry": box,
                    })
                for symbol in symbols:
                    key = tuple(int(value) for value in symbol["stroke_indices"])
                    symbol_cache[key] = symbol
                return margins, symbols

            margins, hwr_symbols = classify(fast_result["candidate_rows"])
            fast_symbols = hwr_symbols
            fast_decoder = decode_selective_partition(
                formula_id, fast_result["groups"], fast_symbols, stroke_count=len(strokes),
            )

            def joint_hwr_score(
                groups: list[frozenset[int]], local_rows: list[dict[str, Any]],
                local_scores: list[float],
            ) -> float:
                nonlocal joint_hwr_encoded, joint_hwr_budget_exhausted
                joint_hwr_partition_diagnostics["partitions_evaluated"] += 1
                rows_by_group = {
                    tuple(int(value) for value in row["source_indices"]): row
                    for row in local_rows
                }
                keys = [tuple(sorted(int(value) for value in group)) for group in groups]
                if any(key not in rows_by_group for key in keys):
                    joint_hwr_partition_diagnostics["candidate_row_missing"] += 1
                    return -math.inf
                geometry_scores_by_group = {
                    tuple(int(value) for value in row["source_indices"]): float(score_value)
                    for row, score_value in zip(local_rows, local_scores, strict=True)
                }
                geometry_prior_denominator = (
                    len(strokes)
                    if joint_geometry_prior_normalization == "stroke_count"
                    else len(keys)
                )
                geometry_prior_score = sum(geometry_scores_by_group[key] for key in keys) / max(
                    geometry_prior_denominator, 1,
                )
                joint_hwr_partition_diagnostics["geometry_prior_score_sum"] += geometry_prior_score
                missing = [key for key in keys if key not in symbol_cache]
                joint_hwr_partition_diagnostics["cached_group_uses"] += len(keys) - len(missing)
                if joint_hwr_encoded + len(missing) > max_joint_hwr_candidates:
                    joint_hwr_budget_exhausted = True
                    joint_hwr_partition_diagnostics["candidate_budget_rejected"] += 1
                    return -math.inf
                if missing:
                    classify([rows_by_group[key] for key in missing])
                    joint_hwr_encoded += len(missing)
                symbols = [symbol_cache[key] for key in keys]
                decoded = decode_selective_partition(
                    formula_id, [list(group) for group in groups], symbols,
                    stroke_count=len(strokes),
                )
                if not decoded.get("accepted"):
                    joint_hwr_partition_diagnostics["decoder_rejected"] += 1
                    rejection_reasons = joint_hwr_partition_diagnostics[
                        "decoder_rejection_reasons"
                    ]
                    reason = str(decoded.get("reason") or "unknown")
                    rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
                    return -math.inf
                joint_hwr_partition_diagnostics["decoder_accepted"] += 1
                hwr_score = float(decoded["joint_token_relation_score"])
                total_score = hwr_score + float(joint_geometry_prior_weight) * geometry_prior_score
                partition_key = tuple(sorted(tuple(sorted(group)) for group in groups))
                joint_score_breakdowns[partition_key] = {
                    "hwr_score": hwr_score,
                    "geometry_prior_score": geometry_prior_score,
                    "geometry_prior_normalization": joint_geometry_prior_normalization,
                    "geometry_prior_denominator": geometry_prior_denominator,
                    "total_score": total_score,
                }
                return total_score

            if joint_hwr:
                tournament = Selective2DAnytimeSolverV1(
                    config=config, group_scorer=score, group_score_batch=score_batch,
                    joint_scorer=joint_hwr_score,
                )
            else:
                tournament = fast_tournament
            margins_for_route = margins if symbol_margins is None else symbol_margins
            if not joint_hwr and symbol_margins is not None:
                result = fast_result
            else:
                result = tournament.solve(
                    strokes, symbol_margins=margins_for_route,
                    allow_local=allow_local,
                )
            if [row["source_indices"] for row in result["candidate_rows"]] != [
                row["stroke_indices"] for row in hwr_symbols
            ]:
                if joint_hwr:
                    group_keys = [
                        tuple(sorted(int(value) for value in group))
                        for group in result["groups"]
                    ]
                    if all(key in symbol_cache for key in group_keys):
                        hwr_symbols = [symbol_cache[key] for key in group_keys]
                    else:
                        result, hwr_symbols = fast_result, fast_symbols
                        result["audit"] = {
                            **result["audit"], "local_joint_candidate_cache_miss": True,
                        }
                else:
                    _margins, hwr_symbols = classify(result["candidate_rows"])
            decoder = decode_selective_partition(
                formula_id, result["groups"], hwr_symbols, stroke_count=len(strokes),
            )
            if result["route"] == "local_2d" and not decoder["accepted"]:
                result, hwr_symbols = fast_result, fast_symbols
                decoder = fast_decoder
                result["audit"] = {**result["audit"], "local_decoder_rejected": True}
            elif (
                joint_hwr and result["route"] == "local_2d"
                and not bool(result["audit"].get("joint_score_improved", False))
            ):
                result, hwr_symbols = fast_result, fast_symbols
                decoder = fast_decoder
                result["audit"] = {**result["audit"], "local_joint_score_rejected": True}
            elif (
                not joint_hwr
                and result["route"] == "local_2d" and fast_decoder["accepted"]
                and decoder["joint_token_relation_score"] < fast_decoder["joint_token_relation_score"] + config.acceptance_margin
            ):
                result, hwr_symbols, decoder = fast_result, fast_symbols, fast_decoder
                result["audit"] = {**result["audit"], "local_joint_score_rejected": True}
        else:
            decoder = {"accepted": False, "reason": "hwr_not_requested", "latex": None, "relations": []}
        result["audit"] = {
            **result["audit"],
            "joint_hwr_scoring_enabled": bool(joint_hwr),
            "joint_hwr_candidates_encoded": joint_hwr_encoded,
            "joint_hwr_candidate_budget": max_joint_hwr_candidates if joint_hwr else 0,
            "joint_geometry_prior_weight": float(joint_geometry_prior_weight),
            "joint_geometry_prior_normalization": joint_geometry_prior_normalization,
            "joint_hwr_budget_exhausted": joint_hwr_budget_exhausted,
            "joint_hwr_partition_diagnostics": (
                joint_hwr_partition_diagnostics if joint_hwr else None
            ),
            "local_search_enabled": bool(allow_local),
        }
        if joint_hwr:
            def _score_key(groups: list[list[int]]) -> tuple[tuple[int, ...], ...]:
                return tuple(sorted(tuple(sorted(int(value) for value in group)) for group in groups))

            fast_key = _score_key(result["audit"].get("fast_incumbent_groups") or [])
            selected_breakdown = joint_score_breakdowns.get(_score_key(result["groups"]))
            incumbent_breakdown = joint_score_breakdowns.get(fast_key)
            alternatives = sorted(
                (
                    (key, breakdown) for key, breakdown in joint_score_breakdowns.items()
                    if key != fast_key
                ),
                key=lambda item: (-item[1]["total_score"], item[0]),
            )
            best_challenger = None
            if incumbent_breakdown is not None and alternatives:
                challenger_key, challenger_breakdown = alternatives[0]
                best_challenger = {
                    "groups": [list(group) for group in challenger_key],
                    "score_breakdown": challenger_breakdown,
                    "hwr_score_delta_vs_fast": (
                        challenger_breakdown["hwr_score"] - incumbent_breakdown["hwr_score"]
                    ),
                    "geometry_prior_delta_vs_fast": (
                        challenger_breakdown["geometry_prior_score"]
                        - incumbent_breakdown["geometry_prior_score"]
                    ),
                    "total_score_delta_vs_fast": (
                        challenger_breakdown["total_score"] - incumbent_breakdown["total_score"]
                    ),
                }
            result["audit"] = {
                **result["audit"],
                "joint_selected_score_breakdown": selected_breakdown,
                "joint_incumbent_score_breakdown": incumbent_breakdown,
                "joint_best_challenger_vs_fast": best_challenger,
            }
        selected_candidates = result.pop("candidate_rows")
        preview = {
            "schema": "aiflow-selective-2d-grouping-preview/v1",
            "status": "development_only_posthoc_shadow",
            "formula_id": formula_id,
            **result,
            "symbols": hwr_symbols,
            "formula_latex": decoder["latex"], "relation_graph": decoder["relations"],
            "decoder": decoder,
            "audit": {
                **result["audit"],
                "frozen_grouping_model": True,
                "hwr_encoded_candidates": hwr_encoded,
                "selected_candidate_count": len(selected_candidates),
                "crohme_training_or_tuning": False,
                "product_default_enabled": False,
            },
        }
        if include_semantic_guard_shadow:
            preview["semantic_guard_shadow"] = _candidate_preserving_semantic_guard_shadow(
                formula_id, preview["groups"], hwr_symbols, decoder,
            )
            preview["audit"]["semantic_guard_shadow_enabled"] = True
        return preview

    def infer(self, source: dict[str, Any]) -> dict[str, Any]:
        sample = self._sample(source)
        embeddings, slices = _candidate_embeddings([sample], self.hwr, self.device)
        probabilities = _probabilities(self.hwr.math_head, embeddings, self.device)
        fused_probability = None
        if self.singleton_hwr is not None:
            auxiliary_probability = _probabilities(
                self.singleton_hwr.math_head, embeddings, self.device,
            )
            weight = float(self.singleton_configuration["auxiliary_weight"])
            fused_probability = (
                (1.0 - weight) * probabilities + weight * auxiliary_probability
            )
        local_probability = probabilities[slices[sample.sample_id]]
        grouping_probability = self.ranker_payload["grouping_model"].predict_proba(
            sample.features
        )[:, 1]
        logits = np.log(
            np.clip(grouping_probability, 1e-6, 1 - 1e-6)
            / np.clip(1 - grouping_probability, 1e-6, 1)
        )
        top_n = int(self.ranker_payload["top_n"])
        ranked = enumerate_partitions(
            sample.candidates, logits, len(sample.strokes), top_n=top_n,
        )
        if not ranked:
            raise ValueError(f"no complete stroke partition: {sample.sample_id}")
        candidate_index = {
            frozenset(row["source_indices"]): index
            for index, row in enumerate(sample.candidates)
        }
        best_geometry = float(ranked[0][0])
        partitions = []
        for rank, (score, groups) in enumerate(ranked, 1):
            partitions.append({
                "sample": sample,
                "rank": rank,
                "geometry_score": float(score),
                "geometry_delta": float(score) - best_geometry,
                "groups": groups,
                "rows": _partition_rows(
                    sample, rank, groups, local_probability, self.labels,
                    candidate_index,
                ),
                "truth": False,
            })
        _attach_features(
            partitions, self.finalizer.model, self.finalizer.contract,
            self.finalizer.payload, self.device,
        )
        baseline = _geometry_selection(partitions)[sample.sample_id]
        proposal = _select(
            partitions, self.ranker_payload["model"], len(FEATURE_NAMES),
        )[sample.sample_id]
        accepted = _gate_accept(
            baseline, proposal, self.ranker_payload["selection_guard"],
        )
        selected = proposal if accepted else baseline
        relation_merge_audit = {"enabled": False}
        if self.relation_merge_configuration is not None:
            relation_selected, relation_merge_audit = _relation_merge_selection(
                partitions, {sample.sample_id: selected},
                self.relation_merge_configuration,
            )
            selected = relation_selected[sample.sample_id]
        relation_merge_changed = bool(
            relation_merge_audit.get("changed_formulas", 0)
        )
        cross_merge_audit = {"enabled": False}
        if self.cross_merge_configuration is not None:
            cross_auxiliary = None
            if fused_probability is not None and self.placement_configuration is not None:
                cross_auxiliary = {
                    "probability": fused_probability,
                    "slices": slices,
                    "labels": self.labels,
                    "policy": self.singleton_configuration["auxiliary_policy"],
                    "weight": float(
                        self.singleton_configuration["auxiliary_weight"]
                    ),
                    "placement_configuration": self.placement_configuration,
                }
            cross_selected, cross_merge_audit = _cross_merge_selection(
                partitions, {sample.sample_id: selected},
                self.cross_merge_configuration,
                cross_auxiliary,
            )
            selected = cross_selected[sample.sample_id]
        cross_merge_changed = bool(cross_merge_audit.get("changed_formulas", 0))
        repeat_merge_audit = {"enabled": False}
        if self.repeat_merge_configuration is not None:
            repeat_selected, repeat_merge_audit = _repeat_merge_selection(
                partitions, {sample.sample_id: selected},
                self.repeat_merge_configuration,
            )
            selected = repeat_selected[sample.sample_id]
        repeat_merge_changed = bool(
            repeat_merge_audit.get("changed_formulas", 0)
        )
        runtime_rows = [
            {
                **{
                    key: value for key, value in row.items()
                    if key not in {"group", "full_probability", "grouping_features"}
                },
                "formula_id": sample.sample_id,
            }
            for row in selected["rows"]
        ]
        singleton_audit = {"enabled": False}
        if self.singleton_hwr is not None:
            if fused_probability is None:
                raise AssertionError("singleton fusion probability is missing")
            weight = float(self.singleton_configuration["auxiliary_weight"])
            restricted_rows = _auxiliary_rows(
                [sample], {sample.sample_id: selected}, fused_probability, slices,
                self.labels, width=5,
                policy=self.singleton_configuration["auxiliary_policy"],
                weight=weight, preserve_source_candidates=True,
            )
            finalized, finalizer_audit = self.finalizer.finalize(restricted_rows)
            singleton_rows = _auxiliary_rows(
                [sample], {sample.sample_id: selected}, fused_probability, slices,
                self.labels, width=5,
                policy=self.singleton_configuration["auxiliary_policy"],
                weight=weight,
            )
            finalized, singleton_audit = _apply_singleton_finalized_rows(
                runtime_rows, finalized, singleton_rows,
                self.singleton_configuration,
            )
            auxiliary_finalized, auxiliary_finalizer_audit = self.finalizer.finalize(
                singleton_rows,
            )
            finalized, numeric_audit = apply_dual_hwr_numeric_rescue(
                finalized, singleton_rows, auxiliary_finalized,
                self.numeric_configuration,
            )
            wide_rows = _auxiliary_rows(
                [sample], {sample.sample_id: selected}, fused_probability, slices,
                self.labels, width=10,
                policy=self.singleton_configuration["auxiliary_policy"],
                weight=weight,
            )
            finalized, wide_syntax_audit = apply_wide_candidate_syntax_rescue(
                finalized, wide_rows, self.wide_syntax_configuration,
            )
            placement_audit = {"enabled": False}
            equality_audit = {"enabled": False}
            if self.placement_configuration is not None:
                top20_rows = _auxiliary_rows(
                    [sample], {sample.sample_id: selected}, fused_probability,
                    slices, self.labels, width=20,
                    policy=self.singleton_configuration["auxiliary_policy"],
                    weight=weight,
                )
                finalized, placement_audit = apply_formula_placement_rescue(
                    finalized, top20_rows, self.placement_configuration,
                )
                finalized, equality_audit = apply_straight_equality_slot_rescue(
                    finalized, top20_rows, self.equality_configuration,
                )
            latin_audit = {"enabled": False}
            if self.latin_hwr is not None:
                latin_rows = build_latin_auxiliary_rows(
                    [sample], {sample.sample_id: selected}, self.latin_hwr,
                    self.latin_labels, self.device,
                )
                finalized, latin_audit = apply_latin_t_context_rescue(
                    finalized, latin_rows, self.latin_configuration,
                )
            finalized, cross_lock_audit = _apply_cross_merge_token_locks(
                finalized, {sample.sample_id: selected},
            )
            singleton_audit = {
                **singleton_audit,
                "restricted_candidate_rerank": {
                    "enabled": True,
                    "candidate_set_preserved": True,
                    "candidate_width": 5,
                    "formula_context_finalizer_reapplied": True,
                },
                "dual_numeric_auxiliary_finalizer": auxiliary_finalizer_audit,
                "dual_numeric_rescue": numeric_audit,
                "wide_numeric_syntax_rescue": wide_syntax_audit,
                "formula_placement_rescue": placement_audit,
                "straight_equality_rescue": equality_audit,
                "latin_t_context_rescue": latin_audit,
                "cross_merge_auxiliary_semantic_lock": cross_lock_audit,
            }
        else:
            finalized, finalizer_audit = self.finalizer.finalize(runtime_rows)
            latin_audit = {"enabled": False}
            finalized, cross_lock_audit = _apply_cross_merge_token_locks(
                finalized, {sample.sample_id: selected},
            )
        pairwise_shape_audit = {"enabled": False}
        if self.pairwise_shape_expert is not None:
            sample_slice = slices[sample.sample_id]
            offset = int(sample_slice.start or 0)
            evidence_by_record = {}
            for source in selected["rows"]:
                group = frozenset(int(value) for value in source["group"])
                embedding_index = offset + candidate_index[group]
                original_probability = probabilities[embedding_index]
                top20_indices = np.argsort(-original_probability)[:20]
                evidence_by_record[str(source["record_id"])] = {
                    "embedding": embeddings[embedding_index].numpy(),
                    "hwr_top1": self.labels[int(top20_indices[0])],
                    "hwr_top20": [
                        self.labels[int(value)] for value in top20_indices
                    ],
                    "hwr_top20_probabilities": [
                        float(original_probability[int(value)])
                        for value in top20_indices
                    ],
                }
            finalized, pairwise_shape_audit = apply_pairwise_shape_rescue(
                finalized,
                evidence_by_record,
                self.pairwise_shape_expert,
                self.pairwise_shape_configuration,
            )
        finalized.sort(key=lambda row: int(row["context_index"]))
        runtime_by_record = {str(row["record_id"]): row for row in runtime_rows}
        source_by_group = {
            frozenset(int(value) for value in row["group"]): row
            for row in selected["rows"]
        }
        groups = []
        candidate_map = {
            frozenset(row["source_indices"]): row for row in sample.candidates
        }
        ordered_groups = sorted(
            selected["groups"],
            key=lambda group: (
                candidate_map[frozenset(group)]["box"]["left"],
                candidate_map[frozenset(group)]["box"]["top"], min(group),
            ),
        )
        for index, group in enumerate(ordered_groups):
            box = candidate_map[frozenset(group)]["box"]
            groups.append({
                "partition_input_index": index,
                "record_id": str(source_by_group[frozenset(group)]["record_id"]),
                "stroke_indices": sorted(int(value) for value in group),
                "box": {key: float(value) for key, value in box.items()},
            })
        assigned = sorted(
            index for group in groups for index in group["stroke_indices"]
        )
        if assigned != list(range(len(sample.strokes))):
            raise AssertionError("runtime grouping is not an exact stroke cover")
        hwr_tokens = [
            str(runtime_by_record[str(row["record_id"])]["final_topk"][0])
            for row in finalized
        ]
        finalized_tokens = [str(row["finalized_top1"]) for row in finalized]
        group_by_record = {row["record_id"]: row for row in groups}
        symbols = [
            {
                "record_id": str(row["record_id"]),
                "context_index": int(row["context_index"]),
                "relation_from_previous": row.get("layout_relation_from_previous"),
                "stroke_indices": list(group_by_record[str(row["record_id"])]["stroke_indices"]),
                "hwr_top1": str(runtime_by_record[str(row["record_id"])]["final_topk"][0]),
                "hwr_topk": list(runtime_by_record[str(row["record_id"])]["final_topk"]),
                "hwr_topk_probabilities": list(
                    runtime_by_record[str(row["record_id"])]["final_topk_probabilities"]
                ),
                "finalized_top1": str(row["finalized_top1"]),
                "changed": bool(row["changed"]),
                "context_decision_status": row.get("decision_status"),
                "decision_source": row.get("decision_source"),
            }
            for row in finalized
        ]
        layout_output = layout_audit = None
        if self.emit_formula_layout_shadow:
            layout_output, layout_audit = _formula_layout_shadow(runtime_rows, finalized)
        return {
            "schema": OUTPUT_SCHEMA,
            "status": "development_only_posthoc_shadow",
            "formula_id": sample.sample_id,
            "groups": groups,
            "symbols": symbols,
            "hwr_top1_tokens": hwr_tokens,
            "partition_context_tokens": list(selected["tokens"]),
            "finalized_tokens": finalized_tokens,
            "formula_text": " ".join(finalized_tokens),
            **({
                "formula_latex_shadow": layout_output["latex"],
                "formula_layout_shadow": layout_output,
            } if layout_output is not None else {}),
            "audit": {
                "candidate_groups": len(sample.candidates),
                "candidate_partitions": len(partitions),
                "geometry_baseline_rank": int(baseline["rank"]),
                "ranker_proposal_rank": int(proposal["rank"]),
                "selected_partition_rank": int(selected["rank"]),
                "partition_change_accepted": bool(selected["rank"] != baseline["rank"]),
                "design_gate_partition_change_accepted": bool(accepted),
                "partition_change_source": (
                    "repeat_merge_rescue" if repeat_merge_changed else
                    "relation_merge_rescue" if relation_merge_changed else
                    "cross_merge_rescue" if cross_merge_changed else
                    "posthoc_design_guard" if accepted else "geometry_baseline"
                ),
                "ranker_probability_gain": float(
                    proposal["ranker_probability"] - baseline["ranker_probability"]
                ),
                "selected_ranker_probability_gain": float(
                    selected["ranker_probability"] - baseline["ranker_probability"]
                ),
                "all_strokes_exactly_once": True,
                "target_label_or_glyph_count_input": False,
                "inserted_or_deleted_glyphs": 0,
                "arithmetic_evaluation": False,
                "partition_ranker_sha256": self.partition_ranker_sha256,
                "hwr_checkpoint_sha256": self.finalizer.hwr_checkpoint_sha256,
                "context_checkpoint_sha256": self.finalizer.checkpoint_sha256,
                "candidate_context_fusion": {
                    **singleton_audit,
                    "relation_merge_rescue": relation_merge_audit,
                    "cross_merge_rescue": cross_merge_audit,
                    "repeat_merge_rescue": repeat_merge_audit,
                    "enabled": self.singleton_hwr is not None,
                    "configuration_sha256": self.singleton_config_sha256,
                    "auxiliary_hwr_checkpoint_sha256": self.singleton_hwr_sha256,
                    "current_96_formula_training_overlap": self.singleton_hwr is not None,
                    "product_default_enabled": False,
                },
                "pairwise_shape_rescue": {
                    **pairwise_shape_audit,
                    "enabled": self.pairwise_shape_expert is not None,
                    "expert_sha256": self.pairwise_shape_expert_sha256,
                    "configuration_sha256": self.pairwise_shape_config_sha256,
                    "product_default_enabled": False,
                },
                "product_default_enabled": False,
                "posthoc_test_tuning": True,
                "candidate_partitions_detail": [
                    {
                        "rank": int(partition["rank"]),
                        "geometry_delta": float(partition["geometry_delta"]),
                        "ranker_probability": float(partition["ranker_probability"]),
                        "groups": [sorted(int(value) for value in group) for group in partition["groups"]],
                        "hwr_top1_tokens": [str(row["final_topk"][0]) for row in partition["rows"]],
                        "context_tokens": list(partition["tokens"]),
                    }
                    for partition in partitions
                ],
                "finalizer": finalizer_audit,
                **({"formula_layout_shadow": layout_audit} if layout_audit is not None else {}),
            },
        }


def _self_test() -> None:
    semantic_symbols = [
        {
            "stroke_indices": [0], "hwr_topk": ["5", "6", "7", "8", "9"],
            "hwr_topk_probabilities": [0.90, 0.04, 0.03, 0.02, 0.01],
            "geometry": {"left": 0, "top": 0, "right": 8, "bottom": 10},
        },
        {
            "stroke_indices": [1], "hwr_topk": ["x", r"\times", "X", r"\chi", "m"],
            "hwr_topk_probabilities": [0.70, 0.20, 0.05, 0.03, 0.02],
            "geometry": {"left": 20, "top": 0, "right": 28, "bottom": 10},
        },
        {
            "stroke_indices": [2], "hwr_topk": ["0", "1", "2", "3", "4"],
            "hwr_topk_probabilities": [0.90, 0.04, 0.03, 0.02, 0.01],
            "geometry": {"left": 40, "top": 0, "right": 48, "bottom": 10},
        },
    ]
    semantic_shadow = _candidate_preserving_semantic_guard_shadow(
        "semantic-shadow-self-test", [[0], [1], [2]], semantic_symbols,
        {"accepted": True, "tokens": ["5", "x", "0"]},
    )
    assert semantic_shadow["status"] == "applied_shadow_only"
    assert [row["token"] for row in semantic_shadow["stages"]["decoder"]] == ["5", "x", "0"]
    assert [row["token"] for row in semantic_shadow["stages"]["after_infix_guard"]] == [
        "5", r"\times", "0",
    ]
    assert semantic_shadow["candidate_preservation"] is True
    assert semantic_shadow["new_tokens"] == semantic_shadow["deleted_glyphs"] == 0
    assert semantic_shadow["grouping_mutations"] == 0
    expression_symbols = [
        {
            "stroke_indices": [index],
            "hwr_topk": topk,
            "hwr_topk_probabilities": probabilities,
            "geometry": {
                "left": index * 20, "top": 0,
                "right": index * 20 + 8, "bottom": 10,
            },
        }
        for index, (topk, probabilities) in enumerate((
            (["/", r"\prime", "1", "j", "i"], [0.975, 0.023, 0.001, 0.0005, 0.0005]),
            (["2", "z", "Z", "3", r"\mathcal{Z}"], [0.992, 0.004, 0.003, 0.001, 0.0]),
            (["-", "+", r"\div", r"\times", "4"], [0.998, 0.001, 0.0005, 0.0003, 0.0002]),
            (["/", "1", r"\prime", "(", "i"], [0.685, 0.238, 0.044, 0.009, 0.008]),
        ))
    ]
    expression_shadow = _candidate_preserving_semantic_guard_shadow(
        "arithmetic-expression-shadow-self-test",
        [[0], [1], [2], [3]], expression_symbols,
        {"accepted": True, "tokens": ["/", "2", "-", "/"]},
    )
    assert [row["token"] for row in expression_shadow["stages"]["after_equation_guard"]] == [
        "/", "2", "-", "/",
    ]
    assert [row["token"] for row in expression_shadow["stages"]["after_arithmetic_expression_guard"]] == [
        "1", "2", "-", "1",
    ]
    assert expression_shadow["audits"]["arithmetic_expression"]["finalized_formulas"] == 1
    assert expression_shadow["candidate_preservation"] is True
    bar_symbols = [
        {
            "stroke_indices": [index],
            "hwr_topk": topk,
            "hwr_topk_probabilities": probs,
            "geometry": {"left": index * 20, "top": 0, "right": index * 20 + 8, "bottom": 10},
        }
        for index, (topk, probs) in enumerate((
            (["|", r"\mid", "1", "i", "l"], [0.44, 0.29, 0.15, 0.07, 0.05]),
            (["6", "b", "G", "o", "0"], [0.70, 0.12, 0.08, 0.06, 0.04]),
            ([r"\div", "/", "-", "+", "4"], [0.90, 0.05, 0.02, 0.02, 0.01]),
            (["8", "B", "3", "0", "6"], [0.80, 0.08, 0.06, 0.04, 0.02]),
            (["=", r"\approx", r"\asymp", r"\equiv", r"\neq"], [0.90, 0.04, 0.03, 0.02, 0.01]),
            (["2", "Z", "z", "3", "7"], [0.90, 0.04, 0.03, 0.02, 0.01]),
        ))
    ]
    bar_shadow = _candidate_preserving_semantic_guard_shadow(
        "bar-equation-self-test", [[index] for index in range(6)], bar_symbols,
        {"accepted": True, "tokens": ["|", "6", r"\div", "8", "=", "2"]},
    )
    assert [row["token"] for row in bar_shadow["stages"]["after_equation_guard"]] == [
        "|", "6", r"\div", "8", "=", "2",
    ]
    assert [row["token"] for row in bar_shadow["stages"]["after_unique_bar_equation_guard"]] == [
        "1", "6", r"\div", "8", "=", "2",
    ]
    assert bar_shadow["audits"]["unique_unpaired_bar_equation"]["status"] == "changed_shadow_only"

    ambiguous_bar_symbols = [
        {
            "stroke_indices": [index],
            "hwr_topk": topk,
            "hwr_topk_probabilities": [0.70, 0.12, 0.08, 0.06, 0.04],
            "geometry": {
                "left": index * 20, "top": 0,
                "right": index * 20 + 8, "bottom": 10,
            },
        }
        for index, topk in enumerate((
            ["|", "1", "2", "3", "i"],
            [r"\times", "x", "+", "-", "4"],
            ["0", "1", "2", "3", "4"],
            ["=", r"\approx", r"\asymp", r"\equiv", r"\neq"],
            ["0", "1", "2", "3", "4"],
        ))
    ]
    ambiguous_bar_shadow = _candidate_preserving_semantic_guard_shadow(
        "ambiguous-bar-equation-self-test", [[index] for index in range(5)],
        ambiguous_bar_symbols,
        {"accepted": True, "tokens": ["|", r"\times", "0", "=", "0"]},
    )
    assert ambiguous_bar_shadow["stages"]["after_unique_bar_equation_guard"] == (
        ambiguous_bar_shadow["stages"]["after_equation_guard"]
    )
    assert ambiguous_bar_shadow["audits"]["unique_unpaired_bar_equation"]["reason"] == (
        "multiple_exact_equation_candidates"
    )
    assert ambiguous_bar_shadow["stages"]["after_boundary_bar_as_unit_guard"] == (
        ambiguous_bar_shadow["stages"]["after_unique_bar_equation_guard"]
    )
    assert ambiguous_bar_shadow["audits"]["boundary_bar_as_unit"]["reason"] == (
        "numeric_equation_deferred_to_exact_equation_guard"
    )

    multi_bar_symbols = [
        {
            "stroke_indices": [index],
            "hwr_topk": topk,
            "hwr_topk_probabilities": [0.70, 0.12, 0.08, 0.06, 0.04],
            "geometry": {
                "left": index * 20, "top": 0,
                "right": index * 20 + 8, "bottom": 10,
            },
        }
        for index, topk in enumerate((
            ["|", "1", "0", "i", "l"],
            ["+", "-", "x", r"\times", "4"],
            ["|", "1", "0", "i", "l"],
            ["=", r"\approx", r"\asymp", r"\equiv", r"\neq"],
            ["2", "3", "0", "1", "6"],
        ))
    ]
    multi_bar_shadow = _candidate_preserving_semantic_guard_shadow(
        "multi-bar-equation-self-test", [[index] for index in range(5)],
        multi_bar_symbols,
        {"accepted": True, "tokens": ["|", "+", "|", "=", "2"]},
    )
    assert [row["token"] for row in multi_bar_shadow["stages"][
        "after_unique_exact_equation_candidates_guard"
    ]] == ["1", "+", "1", "=", "2"]
    assert multi_bar_shadow["audits"]["unique_exact_equation_candidates"]["status"] == (
        "changed_shadow_only"
    )

    no_bar_equation_symbols = [
        {
            "stroke_indices": [index],
            "hwr_topk": topk,
            "hwr_topk_probabilities": [0.70, 0.12, 0.08, 0.06, 0.04],
            "geometry": {
                "left": index * 20, "top": 0,
                "right": index * 20 + 8, "bottom": 10,
            },
        }
        for index, topk in enumerate((
            ["2", "3", "0", "1", "6"],
            [r"\times", "x", "X", "+", "-"],
            ["5", "6", "3", "0", "8"],
            ["=", r"\approx", r"\asymp", r"\equiv", r"\neq"],
            ["(", "i", "I", "1", "l"],
            ["0", "6", "8", "2", "3"],
        ))
    ]
    no_bar_equation_shadow = _candidate_preserving_semantic_guard_shadow(
        "no-bar-exact-equation-self-test", [[index] for index in range(6)],
        no_bar_equation_symbols,
        {"accepted": True, "tokens": ["2", r"\times", "5", "=", "(", "0"]},
    )
    assert [row["token"] for row in no_bar_equation_shadow["stages"][
        "after_unique_exact_equation_candidates_guard"
    ]] == ["2", r"\times", "5", "=", "1", "0"]
    assert no_bar_equation_shadow["audits"][
        "unique_exact_equation_candidates"
    ]["status"] == "changed_shadow_only"

    slash_digit_symbols = [
        {
            "stroke_indices": [index],
            "hwr_topk": topk,
            "hwr_topk_probabilities": [0.70, 0.12, 0.08, 0.06, 0.04],
            "geometry": {
                "left": index * 20, "top": 0,
                "right": index * 20 + 8, "bottom": 10,
            },
        }
        for index, topk in enumerate((
            ["1", "2", "3", "4", "5"],
            [r"\div", "/", "-", "+", "4"],
            ["(", r"\prime", "1", r"\lceil", "/"],
            ["=", r"\approx", r"\asymp", r"\equiv", r"\neq"],
            ["/", r"\prime", "1", "i", "j"],
        ))
    ]
    slash_digit_shadow = _candidate_preserving_semantic_guard_shadow(
        "slash-as-digit-exact-equation-self-test",
        [[index] for index in range(5)], slash_digit_symbols,
        {"accepted": True, "tokens": ["1", r"\div", "(", "=", "/"]},
    )
    assert [row["token"] for row in slash_digit_shadow["stages"][
        "after_unique_exact_equation_candidates_guard"
    ]] == ["1", r"\div", "1", "=", "1"]
    assert len(slash_digit_shadow["audits"][
        "unique_exact_equation_candidates"
    ]["changes"]) == 2

    zero_lookalike_topks = [
        ["6", "8", "0", "2", "3"],
        [r"\times", "x", "X", "+", "-"],
        ["O", "0", "o", r"\mathcal{O}", r"\circ"],
        ["=", r"\approx", r"\asymp", r"\equiv", r"\neq"],
        [r"\mathcal{O}", "0", "O", "o", r"\leftmoon"],
    ]
    zero_lookalike_symbols = [
        {
            "stroke_indices": [index],
            "hwr_topk": topk,
            "hwr_topk_probabilities": [0.70, 0.12, 0.08, 0.06, 0.04],
            "geometry": {
                "left": index * 20, "top": 0,
                "right": index * 20 + 8, "bottom": 10,
            },
        }
        for index, topk in enumerate(zero_lookalike_topks)
    ]
    zero_lookalike_shadow = _candidate_preserving_semantic_guard_shadow(
        "zero-lookalike-exact-equation-self-test",
        [[index] for index in range(5)], zero_lookalike_symbols,
        {"accepted": True, "tokens": ["6", r"\times", "O", "=", r"\mathcal{O}"]},
    )
    assert [row["token"] for row in zero_lookalike_shadow["stages"][
        "after_unique_exact_equation_candidates_guard"
    ]] == ["6", r"\times", "0", "=", "0"]

    triangle_equality_topks = [
        ["6", "8", "0", "2", "3"],
        ["+", "-", "=", "4", "5"],
        ["8", "3", "0", "2", "5"],
        [r"\triangledown", r"\tau", r"\mp", "=", "T"],
        ["1", "7", "2", "4", "6"],
        ["4", "1", "2", "7", "9"],
    ]
    triangle_equality_symbols = [
        {
            "stroke_indices": [index],
            "hwr_topk": topk,
            "hwr_topk_probabilities": [0.70, 0.12, 0.08, 0.06, 0.04],
            "geometry": {
                "left": index * 20, "top": 0,
                "right": index * 20 + 8, "bottom": 10,
            },
        }
        for index, topk in enumerate(triangle_equality_topks)
    ]
    triangle_equality_shadow = _candidate_preserving_semantic_guard_shadow(
        "triangle-equality-exact-equation-self-test",
        [[index] for index in range(6)], triangle_equality_symbols,
        {"accepted": True, "tokens": ["6", "+", "8", r"\triangledown", "1", "4"]},
    )
    assert [row["token"] for row in triangle_equality_shadow["stages"][
        "after_unique_exact_equation_candidates_guard"
    ]] == ["6", "+", "8", "=", "1", "4"]

    greek_like_equation_topks = [
        [r"\hbar", "5", "E", r"\Pi", "N"],
        ["+", "-", "=", "4", "5"],
        ["2", "3", "0", "1", "6"],
        ["+", "-", "=", "4", "5"],
        [r"\eta", "7", r"\uparrow", "}", "9"],
        ["+", "-", "=", "4", "5"],
        [r"\parr", "8", "p", r"\wp", r"\heartsuit"],
        ["=", r"\approx", r"\asymp", r"\equiv", r"\neq"],
        ["2", "3", "0", "1", "6"],
        ["2", "3", "0", "1", "6"],
    ]
    greek_like_equation_symbols = [
        {
            "stroke_indices": [index],
            "hwr_topk": topk,
            "hwr_topk_probabilities": [0.70, 0.12, 0.08, 0.06, 0.04],
            "geometry": {
                "left": index * 20, "top": 0,
                "right": index * 20 + 8, "bottom": 10,
            },
        }
        for index, topk in enumerate(greek_like_equation_topks)
    ]
    greek_like_equation_shadow = _candidate_preserving_semantic_guard_shadow(
        "greek-like-digit-exact-equation-self-test",
        [[index] for index in range(10)], greek_like_equation_symbols,
        {"accepted": True, "tokens": [
            r"\hbar", "+", "2", "+", r"\eta", "+", r"\parr", "=", "2", "2",
        ]},
    )
    assert [row["token"] for row in greek_like_equation_shadow["stages"][
        "after_unique_exact_equation_candidates_guard"
    ]] == ["5", "+", "2", "+", "7", "+", "8", "=", "2", "2"]
    assert len(greek_like_equation_shadow["audits"][
        "unique_exact_equation_candidates"
    ]["changes"]) == 3

    ambiguous_multi_bar_symbols = [
        {
            "stroke_indices": [index],
            "hwr_topk": topk,
            "hwr_topk_probabilities": [0.70, 0.12, 0.08, 0.06, 0.04],
            "geometry": {
                "left": index * 20, "top": 0,
                "right": index * 20 + 8, "bottom": 10,
            },
        }
        for index, topk in enumerate((
            ["|", "0", "1", "i", "l"],
            [r"\times", "X", "x", "+", "-"],
            ["0", "1", "2", "3", "4"],
            ["=", r"\approx", r"\asymp", r"\equiv", r"\neq"],
            ["|", "0", "1", "i", "l"],
        ))
    ]
    ambiguous_multi_bar_shadow = _candidate_preserving_semantic_guard_shadow(
        "ambiguous-multi-bar-equation-self-test", [[index] for index in range(5)],
        ambiguous_multi_bar_symbols,
        {"accepted": True, "tokens": ["|", r"\times", "0", "=", "|"]},
    )
    assert ambiguous_multi_bar_shadow["audits"][
        "unique_exact_equation_candidates"
    ]["reason"] == "multiple_exact_equation_candidates"
    assert ambiguous_multi_bar_shadow["stages"][
        "after_unique_exact_equation_candidates_guard"
    ] == ambiguous_multi_bar_shadow["stages"]["after_unique_bar_equation_guard"]

    boundary_symbols = [
        {
            "stroke_indices": [index],
            "hwr_topk": topk,
            "hwr_topk_probabilities": probabilities,
            "geometry": {
                "left": index * 20, "top": 0,
                "right": index * 20 + 8, "bottom": 10,
            },
        }
        for index, (topk, probabilities) in enumerate((
            (["4", "5", "6", "7", "8"], [0.80, 0.08, 0.06, 0.04, 0.02]),
            ([r"\times", "X", "x", "+", "-"], [0.80, 0.08, 0.06, 0.04, 0.02]),
            (["a", "o", "A", "x", "5"], [0.80, 0.08, 0.06, 0.04, 0.02]),
            (["+", "-", "=", "4", "5"], [0.80, 0.08, 0.06, 0.04, 0.02]),
            (["|", r"\mid", "1", "i", "l"], [0.44, 0.29, 0.15, 0.07, 0.05]),
        ))
    ]
    boundary_shadow = _candidate_preserving_semantic_guard_shadow(
        "boundary-bar-self-test", [[index] for index in range(5)], boundary_symbols,
        {"accepted": True, "tokens": ["4", r"\times", "a", "+", "|"]},
    )
    assert [row["token"] for row in boundary_shadow["stages"][
        "after_boundary_bar_as_unit_guard"
    ]] == ["4", r"\times", "a", "+", "1"]
    assert boundary_shadow["audits"]["boundary_bar_as_unit"]["status"] == (
        "changed_shadow_only"
    )

    unique_equation_symbols = [
        {
            "stroke_indices": [index],
            "hwr_topk": topk,
            "hwr_topk_probabilities": [0.70, 0.18, 0.06, 0.04, 0.02],
            "geometry": {
                "left": index * 20, "top": 0,
                "right": index * 20 + 8, "bottom": 10,
            },
        }
        for index, topk in enumerate((
            [r"\lfloor", "1", "|", "l", "("],
            [r"\mathscr{C}", "8", "S", "C", "B"],
            [r"\div", "/", "-", "+", "4"],
            ["3", "8", "5", "0", "1"],
            ["=", r"\approx", r"\asymp", r"\equiv", r"\neq"],
            ["6", "8", "5", "0", "1"],
        ))
    ]
    unique_equation_shadow = _candidate_preserving_semantic_guard_shadow(
        "unique-candidate-arithmetic-equation-self-test",
        [[index] for index in range(6)], unique_equation_symbols,
        {"accepted": True, "tokens": [r"\lfloor", r"\mathscr{C}", r"\div", "3", "=", "6"]},
    )
    assert unique_equation_shadow["stages"]["decoder"][0]["token"] == r"\lfloor"
    assert [row["token"] for row in unique_equation_shadow["stages"][
        "after_unique_candidate_arithmetic_equation_guard"
    ]] == ["1", "8", r"\div", "3", "=", "6"]
    assert unique_equation_shadow["audits"][
        "unique_candidate_arithmetic_equation"
    ]["status"] == "changed_shadow_only"
    assert unique_equation_shadow["audits"][
        "unique_candidate_arithmetic_equation"
    ]["candidate_sequences_checked"] == 1
    assert unique_equation_shadow["candidate_preservation"] is True
    assert unique_equation_shadow["product_default_enabled"] is False

    ambiguous_equation_rows = [
        {
            "record_id": f"ambiguous-equation-{index}",
            "context": {"index": index},
            "final_topk": topk,
            "final_topk_probabilities": [0.70, 0.18, 0.06, 0.04, 0.02],
            "geometry": {
                "left": index * 20, "top": 0,
                "right": index * 20 + 8, "bottom": 10,
                "center_x": index * 20 + 4, "center_y": 5,
                "width_rel": 8, "height_rel": 10,
            },
        }
        for index, topk in enumerate((
            ["A", "1", "2", "a", "x"],
            [r"\div", "/", "-", "+", "4"],
            ["B", "1", "2", "b", "y"],
            ["=", r"\approx", r"\asymp", r"\equiv", r"\neq"],
            ["1", "2", "3", "4", "5"],
        ))
    ]
    ambiguous_equation_baseline = {
        "ambiguous-equation-0": "A",
        "ambiguous-equation-1": r"\div",
        "ambiguous-equation-2": "B",
        "ambiguous-equation-3": "=",
        "ambiguous-equation-4": "1",
    }
    ambiguous_equation_output, ambiguous_equation_audit = (
        _unique_candidate_arithmetic_equation_shadow(
            ambiguous_equation_rows, ambiguous_equation_baseline,
        )
    )
    assert ambiguous_equation_output == ambiguous_equation_baseline
    assert ambiguous_equation_audit["reason"] == "ambiguous_exact_candidate_equations"
    assert ambiguous_equation_audit["valid_candidate_count"] == 2

    terminal_rhs_symbols = [
        {
            "stroke_indices": [index],
            "hwr_topk": topk,
            "hwr_topk_probabilities": [0.70, 0.18, 0.06, 0.04, 0.02],
            "geometry": {
                "left": index * 20, "top": 0,
                "right": index * 20 + 8, "bottom": 10,
            },
        }
        for index, topk in enumerate((
            ["f", "F", "g", "r", "s"],
            ["(", r"\langle", "[", "{", "c"],
            ["y", "Y", "v", "u", "x"],
            [")", r"\rangle", "]", "}", "c"],
            ["=", r"\approx", r"\asymp", r"\equiv", r"\neq"],
            ["|", r"\mid", "1", "i", "l"],
        ))
    ]
    terminal_rhs_shadow = _candidate_preserving_semantic_guard_shadow(
        "terminal-rhs-bar-self-test", [[index] for index in range(6)],
        terminal_rhs_symbols,
        {"accepted": True, "tokens": ["f", "(", "y", ")", "=", "|"]},
    )
    assert [row["token"] for row in terminal_rhs_shadow["stages"][
        "after_terminal_rhs_bar_guard"
    ]] == ["f", "(", "y", ")", "=", "1"]
    assert terminal_rhs_shadow["audits"]["terminal_rhs_bar"]["status"] == (
        "changed_shadow_only"
    )
    assert terminal_rhs_shadow["candidate_preservation"] is True
    assert terminal_rhs_shadow["product_default_enabled"] is False

    leading_rows = [
        {
            "record_id": f"leading-{index}",
            "context": {"index": index},
            "final_topk": topk,
            "geometry": {
                "left": index * 20, "top": 0,
                "right": index * 20 + 8, "bottom": 10,
                "center_x": index * 20 + 4, "center_y": 5,
                "height_rel": 1,
            },
        }
        for index, topk in enumerate((
            ["|", "1", "0", "i", "l"],
            ["2", "3", "0", "1", "6"],
            ["+", "-", "=", "4", "5"],
            ["3", "8", "5", "0", "1"],
            ["4", "9", "0", "1", "6"],
        ))
    ]
    leading_predictions = {
        f"leading-{index}": token
        for index, token in enumerate(("|", "2", "+", "3", "4"))
    }
    leading_output, leading_audit = _boundary_bar_as_unit_shadow(
        leading_rows, leading_predictions,
    )
    assert leading_output["leading-0"] == "1"
    assert leading_audit["reason"] == "lone_bar_at_leading_numeric_slot"

    relation_predictions = {
        **leading_predictions,
        "leading-2": "=",
    }
    relation_output, relation_audit = _boundary_bar_as_unit_shadow(
        leading_rows, relation_predictions,
    )
    assert relation_output == relation_predictions
    assert relation_audit["status"] == "skipped"

    source = {
        "formula_id": "self-test",
        "strokes": [{
            "order": 0,
            "points": [
                {"x": 0.0, "y": 0.0, "t_ms": 0.0},
                {"x": 1.0, "y": 1.0, "t_ms": 20.0},
            ],
        }],
    }
    formula_id, strokes = _validate_strokes(source)
    assert formula_id == "self-test" and len(strokes) == 1
    try:
        _validate_strokes({**source, "strokes": [{**source["strokes"][0], "order": 1}]})
    except ValueError:
        pass
    else:
        raise AssertionError("invalid stroke order was accepted")
    configuration = {
        "auxiliary_policy": "old_new_product_probability_fusion",
        "auxiliary_weight": 0.6,
        "token_confidence_thresholds": {"/": 0.50, r"\times": 0.45},
    }
    runtime_rows = [{
        "record_id": "r", "formula_id": "f", "final_topk": ["1", "/"],
        "final_topk_probabilities": [0.7, 0.2],
        "context": {"index": 0, "length": 1}, "geometry": {},
    }]
    finalized = [{
        **runtime_rows[0], "context_index": 0, "finalized_top1": "1",
        "changed": False,
    }]
    auxiliary_rows = [{
        "record_id": "r", "formula_id": "f", "final_topk": ["/", "1"],
        "final_topk_probabilities": [0.51, 0.49],
        "hwr_policy": configuration["auxiliary_policy"],
        "hwr_fusion_weight": configuration["auxiliary_weight"],
    }]
    rescued, audit = _apply_singleton_finalized_rows(
        runtime_rows, finalized, auxiliary_rows, configuration,
    )
    assert rescued[0]["finalized_top1"] == "/"
    assert rescued[0]["decision_source"] == "singleton_shape_rescue"
    assert audit["changed"] == 1
    layout_rows = [
        {
            "record_id": "x", "formula_id": "layout", "final_topk": ["x"],
            "final_topk_probabilities": [1.0],
            "geometry": {"left": 0, "top": 10, "right": 20, "bottom": 40},
        },
        {
            "record_id": "2", "formula_id": "layout", "final_topk": ["2"],
            "final_topk_probabilities": [1.0],
            "geometry": {"left": 21, "top": 0, "right": 29, "bottom": 14},
        },
    ]
    layout, layout_audit = _formula_layout_shadow(
        layout_rows,
        [
            {"record_id": "x", "finalized_top1": "x"},
            {"record_id": "2", "finalized_top1": "2"},
        ],
    )
    assert layout["latex"] == "x^{2}"
    assert layout_audit["candidate_extensions"] == 0
    class _GroupingModel:
        def predict_proba(self, features: np.ndarray) -> np.ndarray:
            probability = np.where(features[:, 0] >= 3.0, 0.9, 0.4).astype(np.float32)
            return np.column_stack((1.0 - probability, probability))

    preview_runtime = object.__new__(RawFormulaContextRuntimeV1)
    object.__setattr__(preview_runtime, "ranker_payload", {"grouping_model": _GroupingModel()})
    preview = preview_runtime.selective_grouping_preview({
        "formula_id": "selective", "strokes": [
            {"order": 0, "points": [{"x": 0, "y": 0}, {"x": 0, "y": 1}]},
            {"order": 1, "points": [{"x": 2, "y": 0}, {"x": 2, "y": 1}]},
            {"order": 2, "points": [{"x": 1, "y": 0}, {"x": 1, "y": 1}]},
        ],
    }, include_hwr=False, allow_local=False)
    assert preview["audit"]["all_strokes_exactly_once"] is True
    assert preview["route"] == "fast"
    assert preview["audit"]["local_search_enabled"] is False
    assert preview["audit"]["candidate_groups_local"] == 0
    assert preview["audit"]["crohme_training_or_tuning"] is False

    class _JointGroupingModel:
        def predict_proba(self, features: np.ndarray) -> np.ndarray:
            probability = np.where(features[:, 0] == 1.0, 0.9, 0.1).astype(np.float32)
            return np.column_stack((1.0 - probability, probability))

    joint_runtime = object.__new__(RawFormulaContextRuntimeV1)
    object.__setattr__(joint_runtime, "ranker_payload", {
        "lattice_config": {"temporal_window": 6, "spatial_neighbors": 4},
        "grouping_model": _JointGroupingModel(),
    })
    object.__setattr__(joint_runtime, "hwr", type("_HWR", (), {"math_head": object()})())
    object.__setattr__(joint_runtime, "labels", [f"L{index}" for index in range(372)])
    object.__setattr__(joint_runtime, "device", torch.device("cpu"))
    original_candidate_embeddings = globals()["_candidate_embeddings"]
    original_probabilities = globals()["_probabilities"]
    original_decoder = globals()["decode_selective_partition"]

    def fake_candidate_embeddings(samples, _model, _device):
        rows = samples[0].candidates
        values = torch.tensor(
            [[float(len(row["source_indices"]))] for row in rows], dtype=torch.float32,
        )
        return values, {samples[0].sample_id: slice(0, len(rows))}

    def fake_probabilities(_head, embeddings, _device):
        values = np.zeros((len(embeddings), 372), dtype=np.float32)
        for index, group_size in enumerate(embeddings[:, 0].tolist()):
            if group_size == 1.0:
                values[index, :5] = [0.20, 0.19, 0.18, 0.17, 0.16]
                values[index, 5:] = 0.10 / 367
            else:
                values[index, :5] = [0.90, 0.04, 0.03, 0.02, 0.01]
        return values

    globals()["_candidate_embeddings"] = fake_candidate_embeddings
    globals()["_probabilities"] = fake_probabilities
    try:
        synthetic_source = {
            "formula_id": "selective-joint-self-test",
            "strokes": [{
                "order": index,
                "points": [
                    {"x": float(index * 20), "y": 0.0},
                    {"x": float(index * 20 + 4), "y": 8.0},
                ],
            } for index in range(4)],
        }
        fast_only_preview = joint_runtime.selective_grouping_preview(
            synthetic_source, allow_local=False,
        )
        fast_semantic_preview = joint_runtime.selective_grouping_preview(
            synthetic_source, allow_local=False, include_semantic_guard_shadow=True,
        )
        fast_preview = joint_runtime.selective_grouping_preview(
            synthetic_source, allow_local=True,
        )
        local_symbol_margins = {frozenset({0}): 0.0}
        joint_preview = joint_runtime.selective_grouping_preview(
            synthetic_source, joint_hwr=True, max_joint_hwr_candidates=16,
            symbol_margins=local_symbol_margins,
        )
        geometry_prior_preview = joint_runtime.selective_grouping_preview(
            synthetic_source, joint_hwr=True, max_joint_hwr_candidates=16,
            joint_geometry_prior_weight=1.0,
            symbol_margins=local_symbol_margins,
        )
        group_mean_prior_preview = joint_runtime.selective_grouping_preview(
            synthetic_source, joint_hwr=True, max_joint_hwr_candidates=16,
            joint_geometry_prior_weight=1.0,
            joint_geometry_prior_normalization="group_count",
            symbol_margins=local_symbol_margins,
        )
        budget_source = {
            "formula_id": "selective-joint-budget-self-test",
            "strokes": [{
                "order": index,
                "points": [
                    {"x": float(index * 20), "y": 0.0},
                    {"x": float(index * 20 + 4), "y": 8.0},
                ],
            } for index in range(8)],
        }
        budget_preview = joint_runtime.selective_grouping_preview(
            budget_source, joint_hwr=True, max_joint_hwr_candidates=1,
            symbol_margins={frozenset({0}): 0.0, frozenset({1}): 0.0},
        )
        assert fast_only_preview["route"] == "fast"
        assert fast_only_preview["audit"]["local_search_enabled"] is False
        assert "semantic_guard_shadow" not in fast_only_preview
        assert fast_semantic_preview["audit"]["semantic_guard_shadow_enabled"] is True
        assert fast_semantic_preview["groups"] == fast_only_preview["groups"]
        assert fast_semantic_preview["decoder"] == fast_only_preview["decoder"]
        assert fast_semantic_preview["formula_latex"] == fast_only_preview["formula_latex"]
        assert fast_only_preview["groups"] == fast_preview["groups"]
        assert fast_preview["route"] == "fast" and len(fast_preview["groups"]) == 4
        assert fast_preview["audit"]["skip_reason"] == "region_budget"
        assert fast_preview["audit"]["region_budget_violations"] == ["max_region_fraction"]
        assert fast_preview["audit"]["local_processed_stroke_fraction"] == 0.0
        assert joint_preview["route"] == "local_2d"
        assert joint_preview["groups"] == [[0, 1], [2], [3]]
        assert joint_preview["audit"]["local_processed_stroke_fraction"] <= 0.50
        assert joint_preview["audit"]["joint_hwr_candidates_encoded"] <= 16
        assert geometry_prior_preview["route"] == "fast"
        assert geometry_prior_preview["groups"] == fast_preview["groups"]
        assert geometry_prior_preview["audit"]["joint_geometry_prior_normalization"] == "stroke_count"
        selected_breakdown = geometry_prior_preview["audit"]["joint_selected_score_breakdown"]
        assert selected_breakdown is not None
        assert selected_breakdown["geometry_prior_score"] > 0
        assert selected_breakdown["geometry_prior_normalization"] == "stroke_count"
        assert selected_breakdown["geometry_prior_denominator"] == len(synthetic_source["strokes"])
        assert geometry_prior_preview["audit"]["joint_incumbent_score_breakdown"] == selected_breakdown
        group_mean_incumbent = group_mean_prior_preview["audit"]["joint_incumbent_score_breakdown"]
        assert group_mean_prior_preview["audit"]["joint_geometry_prior_normalization"] == "group_count"
        assert group_mean_incumbent is not None
        assert group_mean_incumbent["geometry_prior_normalization"] == "group_count"
        assert group_mean_incumbent["geometry_prior_denominator"] == len(
            group_mean_prior_preview["audit"]["fast_incumbent_groups"]
        )
        best_challenger = geometry_prior_preview["audit"]["joint_best_challenger_vs_fast"]
        assert best_challenger is not None
        assert math.isclose(
            best_challenger["total_score_delta_vs_fast"],
            best_challenger["hwr_score_delta_vs_fast"]
            + best_challenger["geometry_prior_delta_vs_fast"],
        )
        joint_diagnostics = joint_preview["audit"]["joint_hwr_partition_diagnostics"]
        joint_score_search = joint_preview["audit"]["joint_score_search"]
        assert joint_diagnostics["partitions_evaluated"] > 0
        assert joint_diagnostics["decoder_accepted"] > 0
        assert joint_diagnostics["partitions_evaluated"] == (
            joint_diagnostics["candidate_row_missing"]
            + joint_diagnostics["candidate_budget_rejected"]
            + joint_diagnostics["decoder_rejected"]
            + joint_diagnostics["decoder_accepted"]
        )
        assert sum(joint_diagnostics["decoder_rejection_reasons"].values()) == (
            joint_diagnostics["decoder_rejected"]
        )
        assert joint_score_search["evaluations"] == joint_diagnostics["partitions_evaluated"]
        assert joint_score_search["finite"] == joint_diagnostics["decoder_accepted"]
        assert joint_score_search["evaluations"] == (
            joint_score_search["finite"] + joint_score_search["non_finite"]
        )
        budget_diagnostics = budget_preview["audit"]["joint_hwr_partition_diagnostics"]
        assert budget_preview["audit"]["joint_hwr_budget_exhausted"]
        assert budget_diagnostics["candidate_budget_rejected"] > 0

        def reject_synthetic_partition(*_args, **_kwargs):
            return {
                "accepted": False, "reason": "synthetic_reject",
                "latex": None, "relations": [],
            }

        globals()["decode_selective_partition"] = reject_synthetic_partition
        rejected_preview = joint_runtime.selective_grouping_preview(
            synthetic_source, joint_hwr=True, max_joint_hwr_candidates=16,
            symbol_margins=local_symbol_margins,
        )
        rejected_diagnostics = rejected_preview["audit"]["joint_hwr_partition_diagnostics"]
        assert rejected_preview["route"] == "fast"
        assert rejected_diagnostics["decoder_rejection_reasons"] == {
            "synthetic_reject": rejected_diagnostics["decoder_rejected"],
        }
        assert rejected_diagnostics["decoder_rejected"] > 0
        assert joint_preview["decoder"]["accepted"]
        assert joint_preview["decoder"]["top5_preserved"]
        assert sorted(index for group in joint_preview["groups"] for index in group) == list(range(4))
    finally:
        globals()["_candidate_embeddings"] = original_candidate_embeddings
        globals()["_probabilities"] = original_probabilities
        globals()["decode_selective_partition"] = original_decoder


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition-ranker", type=Path)
    parser.add_argument("--hwr-checkpoint", type=Path, default=DEFAULT_PRODUCT)
    parser.add_argument("--context-checkpoint", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--formula-sequence-config", type=Path)
    parser.add_argument("--formula-syntax-rescue-config", type=Path)
    parser.add_argument("--candidate-context-fusion-config", type=Path)
    parser.add_argument("--candidate-context-auxiliary-hwr-checkpoint", type=Path)
    parser.add_argument("--formula-placement-config", type=Path)
    parser.add_argument("--straight-equality-config", type=Path)
    parser.add_argument("--latin-auxiliary-checkpoint", type=Path)
    parser.add_argument("--pairwise-shape-expert", type=Path)
    parser.add_argument("--pairwise-shape-config", type=Path)
    parser.add_argument("--formula-acceptance-config", type=Path)
    parser.add_argument("--emit-formula-layout-shadow", action="store_true")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--allow-posthoc-shadow", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        print(json.dumps({"self_test": "pass"}))
        return 0
    if args.partition_ranker is None or args.input is None or args.output is None:
        parser.error("--partition-ranker, --input, and --output are required")
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not input_path.is_file():
        parser.error(f"raw formula input is missing: {input_path}")
    if output_path.exists():
        parser.error(f"refusing to overwrite raw formula output: {output_path}")
    if not args.allow_posthoc_shadow:
        parser.error("--allow-posthoc-shadow is required for this development-only gate")
    runtime = RawFormulaContextRuntimeV1.from_artifacts(
        args.partition_ranker, args.hwr_checkpoint, args.context_checkpoint,
        device=args.device, batch_size=args.batch_size,
        formula_sequence_config=args.formula_sequence_config,
        formula_syntax_rescue_config=args.formula_syntax_rescue_config,
        candidate_context_fusion_config=args.candidate_context_fusion_config,
        candidate_context_auxiliary_hwr_checkpoint=args.candidate_context_auxiliary_hwr_checkpoint,
        formula_placement_config=args.formula_placement_config,
        straight_equality_config=args.straight_equality_config,
        latin_auxiliary_checkpoint=args.latin_auxiliary_checkpoint,
        pairwise_shape_expert=args.pairwise_shape_expert,
        pairwise_shape_config=args.pairwise_shape_config,
        emit_formula_layout_shadow=args.emit_formula_layout_shadow,
        allow_posthoc_shadow=args.allow_posthoc_shadow,
    )
    results = [runtime.infer(row) for row in _load_inputs(input_path)]
    output = {
        "schema": SCHEMA,
        "status": "development_only_posthoc_shadow",
        "formulas": results,
        "audit": {
            "formulas": len(results),
            "product_default_enabled": False,
            "posthoc_test_tuning": True,
        },
    }
    if args.emit_formula_layout_shadow:
        output["audit"]["formula_layout_shadow_emitted"] = len(results)
    if args.formula_acceptance_config is not None:
        acceptance_configuration, acceptance_configuration_sha256 = (
            load_acceptance_configuration(args.formula_acceptance_config)
        )
        output = apply_formula_acceptance_guard(
            output, acceptance_configuration, acceptance_configuration_sha256,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    print(json.dumps({
        "output": str(output_path), "sha256": _sha256(output_path),
        "formulas": len(results), "status": output["status"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
