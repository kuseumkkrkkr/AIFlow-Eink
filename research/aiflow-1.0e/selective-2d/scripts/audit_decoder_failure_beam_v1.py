#!/usr/bin/env python3
"""Microscope the frozen decoder beam on current Fast-shadow residuals.

Replays saved Top-5 probabilities and the exact decoder code against oracle
groups. This isolates decoding from grouping and does not run/train the HWR,
select thresholds, access CROHME, or alter product behavior.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from selective_decoder_v1 import (
    DEFAULT_TOKEN_BEAM,
    _strict_latex,
    decode_selective_partition,
)


SCHEMA = "aiflow-hwr-decoder-failure-beam-microscope/v2"
SHADOW_ARMS = (
    "fast", "selective_geometry", "selective_joint_hwr",
    "selective_joint_hwr_geometry_prior",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _decoder_rows(trace: dict[str, Any]) -> tuple[list[list[int]], list[dict[str, Any]]]:
    groups = [
        [int(index) for index in group]
        for group in trace["source"]["target_grouping"]
    ]
    symbols = []
    for symbol in trace["oracle_group_hwr"]["symbols"]:
        prediction = symbol["prediction"]
        symbols.append({
            "stroke_indices": [int(index) for index in symbol["stroke_indices"]],
            "hwr_topk": [str(token) for token in prediction["top5"]],
            "hwr_topk_probabilities": [
                float(value) for value in prediction["top5_probabilities"]
            ],
            "geometry": {
                key: float(value)
                for key, value in symbol["preprocessing"]["raw_bbox"].items()
            },
        })
    return groups, symbols


def _beam_audit(
    trace: dict[str, Any], groups: list[list[int]], symbols: list[dict[str, Any]],
    beam_width: int,
) -> dict[str, Any]:
    by_group = {
        tuple(sorted(symbol["stroke_indices"])): symbol for symbol in symbols
    }
    ordered_symbols = [by_group[tuple(sorted(group))] for group in groups]
    targets = [str(token) for token in trace["source"]["target_tokens"]]
    if len(targets) != len(ordered_symbols):
        raise AssertionError(f"target/group count mismatch: {trace['sample_id']}")

    rows = []
    options = []
    first_lost_position = None
    first_lost_rank = None
    first_lost_cutoff_gap = None
    beams: list[tuple[float, tuple[str, ...]]] = [(0.0, ())]
    candidate_complete = True
    target_path_available = True
    first_lost_reason = None
    for ordinal, (symbol, target) in enumerate(
        zip(ordered_symbols, targets, strict=True)
    ):
        choices = list(zip(
            symbol["hwr_topk"], symbol["hwr_topk_probabilities"], strict=True,
        ))
        rows.append({
            "record_id": f"{trace['sample_id']}:{ordinal}",
            "formula_id": str(trace["sample_id"]),
            "final_topk": symbol["hwr_topk"],
            "final_topk_probabilities": symbol["hwr_topk_probabilities"],
            "geometry": symbol["geometry"],
        })
        options.append(choices)
        if target not in symbol["hwr_topk"]:
            candidate_complete = False
            if first_lost_position is None:
                first_lost_position = ordinal + 1
                first_lost_reason = "target_missing_from_top5"
            target_path_available = False

        expanded = [
            (
                score + math.log(max(1e-8, probability)),
                prefix + (token,),
            )
            for score, prefix in beams
            for token, probability in choices
        ]
        expanded.sort(key=lambda item: (-item[0], item[1]))
        target_prefix = tuple(targets[:ordinal + 1])
        target_row = None
        if target_path_available:
            target_row = next(
                ((rank, score) for rank, (score, prefix) in enumerate(expanded, 1)
                 if prefix == target_prefix),
                None,
            )
            if target_row is None:
                raise AssertionError("Top-5 target prefix absent before beam pruning")
            if len(expanded) >= beam_width:
                cutoff_score = expanded[beam_width - 1][0]
                cutoff_gap = target_row[1] - cutoff_score
            else:
                cutoff_gap = None
        else:
            cutoff_gap = None
        beams = expanded[:beam_width]
        if target_path_available and first_lost_position is None and not any(
            prefix == target_prefix for _, prefix in beams
        ):
            first_lost_position = ordinal + 1
            first_lost_rank = target_row[0]
            first_lost_cutoff_gap = cutoff_gap
            first_lost_reason = "pruned_by_score_only_beam"
            target_path_available = False

    target_tokens = tuple(targets)
    beam_target_rank = next(
        (rank for rank, (_, tokens) in enumerate(beams, 1) if tokens == target_tokens),
        None,
    )
    target_score = None
    if candidate_complete:
        target_score = sum(
            math.log(max(1e-8, ordered_symbols[index]["hwr_topk_probabilities"][
                ordered_symbols[index]["hwr_topk"].index(target)
            ]))
            for index, target in enumerate(targets)
        )

    target_valid = None
    target_latex = None
    target_relation_score = None
    if candidate_complete:
        target_predictions = {
            str(row["record_id"]): token
            for row, token in zip(rows, targets, strict=True)
        }
        try:
            target_latex, _target_edges, target_relation_score = _strict_latex(
                rows, target_predictions,
            )
            target_valid = True
        except ValueError as error:
            target_valid = False
            target_latex = str(error)

    valid_candidates = []
    for beam_rank, (token_score, tokens) in enumerate(beams, 1):
        predictions = {
            str(row["record_id"]): token
            for row, token in zip(rows, tokens, strict=True)
        }
        try:
            latex, relations, relation_score = _strict_latex(rows, predictions)
        except ValueError:
            continue
        valid_candidates.append({
            "beam_rank": beam_rank,
            "tokens": list(tokens),
            "latex": latex,
            "token_score": token_score,
            "relation_score": relation_score,
            "joint_score": token_score / len(rows) + relation_score,
            "relation_count": len(relations),
        })

    # The production decoder scans the token-score-sorted beam and replaces
    # the incumbent only on a strict score increase; ties keep the earlier row.
    selected = None
    for candidate in valid_candidates:
        if selected is None or candidate["joint_score"] > selected["joint_score"]:
            selected = candidate
    target_valid_beam_rank = next(
        (rank for rank, candidate in enumerate(
            sorted(valid_candidates, key=lambda item: (-item["joint_score"], item["beam_rank"])),
            1,
        ) if candidate["tokens"] == list(target_tokens)),
        None,
    )
    target_candidate = next(
        (candidate for candidate in valid_candidates
         if candidate["tokens"] == list(target_tokens)),
        None,
    )
    token_level = []
    if selected is not None:
        for ordinal, (symbol, target, picked) in enumerate(
            zip(ordered_symbols, targets, selected["tokens"], strict=True),
        ):
            target_index = symbol["hwr_topk"].index(target) if target in symbol["hwr_topk"] else None
            picked_index = symbol["hwr_topk"].index(picked) if picked in symbol["hwr_topk"] else None
            target_probability = (
                symbol["hwr_topk_probabilities"][target_index]
                if target_index is not None else None
            )
            picked_probability = (
                symbol["hwr_topk_probabilities"][picked_index]
                if picked_index is not None else None
            )
            token_level.append({
                "position": ordinal + 1,
                "target": target,
                "selected": picked,
                "mismatch": target != picked,
                "target_hwr_rank": target_index + 1 if target_index is not None else None,
                "selected_hwr_rank": picked_index + 1 if picked_index is not None else None,
                "target_probability": target_probability,
                "selected_probability": picked_probability,
                "selected_minus_target_log_probability": (
                    math.log(max(1e-8, picked_probability))
                    - math.log(max(1e-8, target_probability))
                    if picked_probability is not None and target_probability is not None
                    else None
                ),
            })
    decomposed_gap = None
    if selected is not None and target_candidate is not None:
        token_gap = sum(
            float(row["selected_minus_target_log_probability"])
            for row in token_level
        ) / len(token_level)
        relation_gap = selected["relation_score"] - float(target_relation_score)
        decomposed_gap = token_gap + relation_gap
        if not math.isclose(
            decomposed_gap,
            selected["joint_score"] - target_candidate["joint_score"],
            rel_tol=1e-10,
            abs_tol=1e-10,
        ):
            raise AssertionError("selected-vs-target score decomposition mismatch")

    replay = decode_selective_partition(
        str(trace["sample_id"]), groups, symbols,
        stroke_count=int(trace["source"]["stroke_count"]),
        token_beam=beam_width,
    )
    saved = trace["oracle_group_hwr"]["decoder"]
    parity = (
        bool(replay.get("accepted")) == bool(saved["accepted"])
        and replay.get("tokens") == saved.get("tokens")
        and replay.get("latex") == saved.get("latex")
    )
    if not parity:
        raise AssertionError(f"current decoder replay mismatch: {trace['sample_id']}")

    return {
        "sample_id": str(trace["sample_id"]),
        "target_tokens": list(targets),
        "candidate_complete": candidate_complete,
        "target_beam_rank": beam_target_rank,
        "first_lost_position": first_lost_position,
        "first_lost_reason": first_lost_reason,
        "first_lost_rank_before_prune": first_lost_rank,
        "first_lost_score_gap_vs_beam_cutoff": first_lost_cutoff_gap,
        "target_structurally_valid": target_valid,
        "target_latex_or_invalid_reason": target_latex,
        "target_joint_score": (
            target_score / len(rows) + target_relation_score
            if target_valid and target_score is not None else None
        ),
        "target_rank_among_valid_beam_candidates": target_valid_beam_rank,
        "valid_beam_candidate_count": len(valid_candidates),
        "selected_beam_rank": selected["beam_rank"] if selected else None,
        "selected_joint_score": selected["joint_score"] if selected else None,
        "selected_relation_score": selected["relation_score"] if selected else None,
        "selected_relation_count": selected["relation_count"] if selected else None,
        "selected_latex": selected["latex"] if selected else None,
        "selected_tokens": selected["tokens"] if selected else None,
        "target_to_selected_score_gap": (
            selected["joint_score"] - target_candidate["joint_score"]
            if selected and target_candidate else None
        ),
        "score_gap_decomposition": {
            "token_log_probability_component": (
                sum(float(row["selected_minus_target_log_probability"])
                    for row in token_level) / len(token_level)
                if token_level and all(
                    row["selected_minus_target_log_probability"] is not None
                    for row in token_level
                ) else None
            ),
            "relation_component": (
                selected["relation_score"] - float(target_relation_score)
                if selected and target_candidate else None
            ),
            "reconstructed_total": decomposed_gap,
        },
        "token_level_selection": token_level,
        "decoder_replay_matches_saved_trace": parity,
    }


def audit(
    traces_path: Path, shadow_path: Path, beam_width: int, arm: str,
) -> dict[str, Any]:
    if arm not in SHADOW_ARMS:
        raise ValueError(f"unsupported shadow arm: {arm}")
    traces = _jsonl(traces_path)
    shadow = json.loads(shadow_path.read_text(encoding="utf-8"))
    if shadow.get("evaluation_status", {}).get(
        "independent_acceptance_claim_allowed"
    ) is not False:
        raise ValueError("shadow input lacks the posthoc/non-acceptance guard")
    shadow_by_id = {
        str(row["sample_id"]): row for row in shadow["formula_level"]
    }
    if len(traces) != len(shadow_by_id):
        raise AssertionError("trace/shadow formula count mismatch")

    records = []
    decoder_parity = 0
    for trace in traces:
        groups, symbols = _decoder_rows(trace)
        row = _beam_audit(trace, groups, symbols, beam_width)
        arms = shadow_by_id[row["sample_id"]].get("arms") or {}
        if arm not in arms:
            raise AssertionError(f"shadow arm missing for {row['sample_id']}: {arm}")
        current = arms[arm]
        row["evaluated_arm"] = arm
        row["arm_group_exact"] = bool(current["group_exact"])
        stages = current.get("formula_exact_by_stage") or {}
        row["arm_shadow_exact"] = bool(
            stages.get("after_boundary_bar_as_unit_guard", False)
        )
        row["current_arm_shadow_residual"] = (
            row["arm_group_exact"] and not row["arm_shadow_exact"]
        )
        row["residual_glyph_errors"] = list(
            current.get("residual_glyph_errors_after_guard") or []
        ) if row["current_arm_shadow_residual"] else []
        decoder_parity += int(row["decoder_replay_matches_saved_trace"])
        records.append(row)

    residuals = [row for row in records if row["current_arm_shadow_residual"]]
    complete_residuals = [row for row in residuals if row["candidate_complete"]]
    pruned = [row for row in complete_residuals if row["target_beam_rank"] is None]
    surviving = [row for row in complete_residuals if row["target_beam_rank"] is not None]
    valid_survivors = [row for row in surviving if row["target_structurally_valid"]]
    gaps = [float(row["target_to_selected_score_gap"]) for row in valid_survivors
            if row["target_to_selected_score_gap"] is not None]
    mismatched_positions = [
        token
        for row in valid_survivors
        for token in row["token_level_selection"]
        if token["mismatch"]
    ]

    checks = {
        "formula_counts_match": len(traces) == shadow.get("formulas"),
        "selected_arm_present_for_every_formula": all(
            arm in (row.get("arms") or {}) for row in shadow["formula_level"]
        ),
        "decoder_replay_matches_saved_trace_for_every_formula": decoder_parity == len(traces),
        "current_residuals_partition_into_top5_complete_or_incomplete": (
            len(residuals) == len(complete_residuals)
            + sum(not row["candidate_complete"] for row in residuals)
        ),
        "top5_complete_residuals_partition_by_beam_survival": (
            len(complete_residuals) == len(pruned) + len(surviving)
        ),
        "surviving_valid_target_rows_have_selected_candidate_score": all(
            row["target_to_selected_score_gap"] is not None for row in valid_survivors
        ),
        "shadow_product_default_disabled": shadow.get("audit", {}).get(
            "product_default_enabled"
        ) is False,
        "shadow_crohme_training_or_tuning_disabled": shadow.get("audit", {}).get(
            "crohme_training_or_tuning"
        ) is False,
    }
    if not all(checks.values()):
        raise AssertionError(f"decoder microscope verification failed: {checks}")

    return {
        "schema": SCHEMA,
        "scope": (
            "frozen posthoc oracle-group decoder diagnostic for the selected shadow arm; "
            "no HWR inference, training, CROHME, threshold selection, or product promotion"
        ),
        "arm": arm,
        "inputs": {
            "traces_path": str(traces_path),
            "traces_sha256": _sha256(traces_path),
            "shadow_path": str(shadow_path),
            "shadow_sha256": _sha256(shadow_path),
            "decoder_path": "scripts/selective_decoder_v1.py",
            "decoder_sha256": _sha256(Path(__file__).with_name("selective_decoder_v1.py")),
            "audit_script_sha256": _sha256(Path(__file__)),
        },
        "beam_width": beam_width,
        "summary": {
            "formulas": len(traces),
            "decoder_replay_parity_formulas": decoder_parity,
            "current_arm_group_exact_shadow_residuals": len(residuals),
            "residuals_with_target_missing_from_top5": sum(
                not row["candidate_complete"] for row in residuals
            ),
            "top5_complete_residuals": len(complete_residuals),
            "top5_complete_residuals_pruned_by_beam": len(pruned),
            "top5_complete_residuals_reaching_beam": len(surviving),
            "surviving_targets_valid_under_strict_ast": len(valid_survivors),
            "surviving_valid_targets_not_selected": sum(
                row["selected_tokens"] != row["target_tokens"]
                for row in valid_survivors
            ),
            "surviving_valid_target_token_mismatch_positions": len(mismatched_positions),
            "mismatch_target_hwr_rank_histogram": dict(sorted(Counter(
                str(row["target_hwr_rank"])
                for row in mismatched_positions
            ).items())),
            "mismatch_selected_hwr_rank_histogram": dict(sorted(Counter(
                str(row["selected_hwr_rank"])
                for row in mismatched_positions
            ).items())),
            "mismatch_mean_selected_minus_target_log_probability": (
                sum(float(row["selected_minus_target_log_probability"])
                    for row in mismatched_positions)
                / len(mismatched_positions)
                if mismatched_positions else None
            ),
            "valid_target_to_selected_score_gap": {
                "count": len(gaps),
                "min": min(gaps) if gaps else None,
                "max": max(gaps) if gaps else None,
                "mean": sum(gaps) / len(gaps) if gaps else None,
            },
            "strict_ast_rejection_count_across_all_decoder_replays": sum(
                row["selected_beam_rank"] is None for row in records
            ),
        },
        "verification": {"checks": checks, "all_checks_pass": all(checks.values())},
        "formula_level": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--shadow-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=SHADOW_ARMS, default="fast")
    parser.add_argument("--beam-width", type=int, default=DEFAULT_TOKEN_BEAM)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite output: {args.output}")
    if args.beam_width < 1:
        parser.error("beam width must be positive")
    report = audit(args.traces, args.shadow_audit, args.beam_width, args.arm)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "summary": report["summary"],
        "verification": report["verification"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
