#!/usr/bin/env python3
"""Replay frozen Top-5 rescoring shadows on the selected guarded partitions."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from audit_unique_candidate_equation_rescue_v1 import (
    _search_unique_equation,
    _terminal_rhs_bar_rescue,
)
from formula_layout_v1 import recontextualize_formula_rows
from raw_formula_context_runtime_v1 import _candidate_preserving_semantic_guard_shadow
from semantic_infix_guard_v1 import apply_semantic_infix_guard


SCHEMA = "aiflow-hwr-guarded-top5-rescore-shadow/v1"
GUARDED_ARM = "selective_joint_hwr_geometry_prior_group_count_guard"
FAST_ARM = "fast"
BASE_STAGE = "after_boundary_bar_as_unit_guard"
REPLAY_STAGES = (
    "decoder", "after_fence_guard", "after_infix_guard", "after_equation_guard",
    "after_unique_bar_equation_guard", "after_unique_exact_equation_candidates_guard",
    "after_boundary_bar_as_unit_guard",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _group_key(group: list[int] | tuple[int, ...]) -> tuple[int, ...]:
    return tuple(sorted(int(index) for index in group))


def _group_box(strokes: list[dict], group: tuple[int, ...]) -> dict[str, float]:
    points = [
        point for index in group for point in (strokes[index].get("points") or [])
    ]
    if not points:
        raise ValueError("selected group has no raw stroke points")
    xs = [float(point["x"]) for point in points]
    ys = [float(point["y"]) for point in points]
    left, right, top, bottom = min(xs), max(xs), min(ys), max(ys)
    return {
        "left": left, "top": top, "right": right, "bottom": bottom,
        "center_x": (left + right) / 2.0,
        "center_y": (top + bottom) / 2.0,
        "width_rel": max(right - left, 1e-6),
        "height_rel": max(bottom - top, 1e-6),
    }


def _writer_bootstrap(
    rows: list[dict[str, Any]], writer_by_id: dict[str, str], *,
    iterations: int = 10_000, seed: int = 20261001,
) -> dict[str, Any]:
    by_writer: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_writer.setdefault(writer_by_id[row["sample_id"]], []).append(row)
    writers = sorted(by_writer)
    rng = random.Random(seed)
    deltas = []
    for _ in range(iterations):
        sampled = [rng.choice(writers) for _ in writers]
        denominator = sum(len(by_writer[writer]) for writer in sampled)
        before = sum(int(row["baseline_exact"]) for writer in sampled for row in by_writer[writer])
        after = sum(int(row["challenger_exact"]) for writer in sampled for row in by_writer[writer])
        deltas.append(100.0 * (after - before) / denominator)
    deltas.sort()
    return {
        "method": "paired writer-cluster bootstrap; writers sampled with replacement",
        "iterations": iterations,
        "seed": seed,
        "delta_exact_rate_pp_95_interval": [
            deltas[int(0.025 * (iterations - 1))],
            deltas[int(0.975 * (iterations - 1))],
        ],
        "interpretation": "consumed-development exploratory shadow only",
    }


def _transition(rows: list[dict[str, Any]], stage: str, base_stage: str) -> dict[str, Any]:
    counts = Counter()
    recovered, regressed = [], []
    for row in rows:
        before, after = bool(row[base_stage]), bool(row[stage])
        if before and after:
            counts["both_exact"] += 1
        elif before:
            counts["baseline_only_exact"] += 1
            regressed.append(row["sample_id"])
        elif after:
            counts["shadow_only_recovery"] += 1
            recovered.append(row["sample_id"])
        else:
            counts["both_not_exact"] += 1
    return {
        "counts": dict(sorted(counts.items())),
        "recovered_ids": recovered,
        "regressed_ids": regressed,
    }


def audit(
    summary_path: Path, *, output_path: Path | None = None,
    arm: str = GUARDED_ARM,
) -> dict[str, Any]:
    source = summary_path.resolve()
    summary = json.loads(source.read_text(encoding="utf-8"))
    if summary.get("schema") != "aiflow-selective-2d-research-loop/v1":
        raise ValueError("unexpected source summary schema")
    if summary.get("product_default_enabled") is not False or summary.get("crohme_training_or_tuning") is not False:
        raise AssertionError("source replay crossed a product or CROHME boundary")
    if summary.get("promotion_eligible") is not False:
        raise AssertionError("source replay is not shadow-only")
    if not summary.get("run_config", {}).get("group_count_increase_guard_shadow"):
        raise ValueError("source replay lacks the guarded partition arm")

    dataset_root = Path(summary["inputs"]["dataset_root"])
    formulas_path = dataset_root / "data" / "formulas_valid.jsonl"
    ownership_path = dataset_root / "data" / "ownership_train.jsonl"
    formula_sha = _sha256(formulas_path)
    ownership_sha = _sha256(ownership_path)
    if formula_sha != summary["inputs"]["formulas_valid_sha256"]:
        raise AssertionError("formula source hash differs from guarded replay")
    if ownership_sha != summary["inputs"]["ownership_train_sha256"]:
        raise AssertionError("ownership source hash differs from guarded replay")
    formulas = {str(row["sample_id"]): row for row in _jsonl(formulas_path)}
    ownership = [row for row in _jsonl(ownership_path) if row.get("accepted")]
    writer_by_id = {str(row["sample_id"]): str(row["writer_id"]) for row in ownership}
    records = summary.get("records") or []
    if set(writer_by_id) != {str(row["sample_id"]) for row in records}:
        raise AssertionError("accepted ownership IDs differ from replay formula IDs")

    if arm not in summary.get("semantic_guard_shadow", {}).get("arms", {}):
        raise ValueError(f"semantic shadow arm missing: {arm}")
    summary_arm = summary["semantic_guard_shadow"]["arms"][arm]
    stage_counts = {stage: Counter() for stage in REPLAY_STAGES}
    branch_counts = {
        stage: Counter() for stage in (
            "existing_stack", "numeric_context", "unique_equation",
            "numeric_then_unique_equation", "terminal_rhs_bar",
        )
    }
    formula_rows: list[dict[str, Any]] = []
    changed_cases = []
    post_rescore_failure_details = []
    total_exact_group_tokens = 0
    guard_rejection_ids = []

    for record in records:
        sample_id = str(record["sample_id"])
        metrics = record["hwr_tournament"][arm]
        if int(metrics.get("local_group_count_increase_rejections", 0)):
            guard_rejection_ids.append(sample_id)
        groups = [list(map(int, group)) for group in metrics["groups"]]
        stroke_count = len(formulas[sample_id]["strokes"])
        assigned = [index for group in groups for index in group]
        if sorted(assigned) != list(range(stroke_count)) or len(set(assigned)) != len(assigned):
            raise AssertionError(f"selected groups violate exact stroke cover: {sample_id}")
        if not metrics["group_exact"]:
            formula_rows.append({
                "sample_id": sample_id,
                "baseline_exact": False,
                **{stage: False for stage in (
                    "existing_stack", "numeric_context", "unique_equation",
                    "numeric_then_unique_equation", "terminal_rhs_bar",
                )},
            })
            continue

        strokes = formulas[sample_id]["strokes"]
        symbol_by_group = {
            _group_key(symbol["stroke_indices"]): symbol
            for symbol in metrics["selected_symbols"]
        }
        if len(symbol_by_group) != len(groups):
            raise AssertionError(f"group/symbol count mismatch: {sample_id}")
        selected_symbols = []
        rows = []
        record_to_group = {}
        for index, group in enumerate(groups):
            key = _group_key(group)
            symbol = symbol_by_group.get(key)
            if symbol is None:
                raise AssertionError(f"missing Top-5 candidate row: {sample_id}:{key}")
            top5 = [str(token) for token in symbol["hwr_topk"]]
            probabilities = [float(value) for value in symbol["hwr_topk_probabilities"]]
            if len(top5) != 5 or len(set(top5)) != 5 or len(probabilities) != 5:
                raise AssertionError(f"invalid frozen Top-5: {sample_id}:{key}")
            geometry = _group_box(strokes, key)
            selected_symbols.append({
                "stroke_indices": list(key),
                "hwr_topk": top5,
                "hwr_topk_probabilities": probabilities,
                "geometry": {name: geometry[name] for name in ("left", "top", "right", "bottom")},
            })
            record_id = f"{sample_id}:{index}"
            record_to_group[record_id] = key
            rows.append({
                "record_id": record_id,
                "formula_id": sample_id,
                "context": {"index": index, "length": len(groups)},
                "final_topk": top5,
                "final_topk_probabilities": probabilities,
                "geometry": geometry,
            })

        decoder = {"accepted": metrics["decoder_accepted"], "tokens": metrics["decoder_tokens"]}
        if not decoder["accepted"] or len(decoder["tokens"]) != len(groups):
            raise AssertionError(f"accepted exact grouping lacks a full decoder sequence: {sample_id}")
        replay = _candidate_preserving_semantic_guard_shadow(
            sample_id, groups, selected_symbols, decoder,
        )
        if replay.get("status") != "applied_shadow_only":
            raise AssertionError(f"base semantic stack did not replay: {sample_id}")

        target_by_group = {
            _group_key(group): str(token)
            for group, token in zip(metrics["target_groups"], metrics["target_tokens"], strict=True)
        }
        if set(target_by_group) != set(record_to_group.values()):
            raise AssertionError(f"target mapping mismatch on exact partition: {sample_id}")
        semantic_metrics = metrics["semantic_guard_shadow_if_groups_exact"]
        if not isinstance(semantic_metrics, dict):
            raise AssertionError(f"source semantic metrics unavailable: {sample_id}")

        predictions_by_stage: dict[str, dict[tuple[int, ...], str]] = {}
        for stage in REPLAY_STAGES:
            stage_rows = replay["stages"].get(stage) or []
            by_group = {_group_key(row["stroke_indices"]): str(row["token"]) for row in stage_rows}
            if set(by_group) != set(record_to_group.values()):
                raise AssertionError(f"semantic stage changed ownership: {sample_id}:{stage}")
            predictions_by_stage[stage] = by_group
            hits = sum(by_group[group] == target for group, target in target_by_group.items())
            exact = hits == len(target_by_group)
            stage_counts[stage]["token_hits"] += hits
            stage_counts[stage]["formula_exact"] += int(exact)
            if (
                hits != int(semantic_metrics["token_hits_by_stage"][stage])
                or exact != bool(semantic_metrics["formula_exact_by_stage"][stage])
            ):
                raise AssertionError(f"semantic stage replay mismatch: {sample_id}:{stage}")

        contextual_rows, _layout_audit = recontextualize_formula_rows(rows)
        record_order = [str(row["record_id"]) for row in rows]
        existing_by_record = {
            record_id: predictions_by_stage[BASE_STAGE][record_to_group[record_id]]
            for record_id in record_order
        }
        numeric_by_record, numeric_audit = apply_semantic_infix_guard(
            contextual_rows, existing_by_record, minimum_probability_ratio=0.0,
            ambiguity_policy="numeric_context_dominance",
            maximum_competitor_probability_ratio=0.002,
        )
        ordered_rows = sorted(contextual_rows, key=lambda row: int(row["context"]["index"]))
        ordered_ids = [str(row["record_id"]) for row in ordered_rows]
        top5_candidates = [[str(token) for token in row["final_topk"]] for row in ordered_rows]

        def token_sequence(by_record: dict[str, str]) -> list[str]:
            return [str(by_record[record_id]) for record_id in ordered_ids]

        existing_sequence = token_sequence(existing_by_record)
        numeric_sequence = token_sequence(numeric_by_record)
        unique_output, unique_audit = _search_unique_equation(
            ordered_rows, existing_sequence, top5_candidates,
        )
        combined_output, combined_audit = _search_unique_equation(
            ordered_rows, numeric_sequence, top5_candidates,
        )
        unique_sequence = unique_output or existing_sequence
        combined_sequence = combined_output or numeric_sequence
        terminal_sequence, terminal_audit = _terminal_rhs_bar_rescue(
            combined_sequence, top5_candidates,
        )

        by_stage_sequences = {
            "existing_stack": existing_sequence,
            "numeric_context": numeric_sequence,
            "unique_equation": unique_sequence,
            "numeric_then_unique_equation": combined_sequence,
            "terminal_rhs_bar": terminal_sequence,
        }
        token_by_record_by_stage = {
            stage: dict(zip(ordered_ids, sequence, strict=True))
            for stage, sequence in by_stage_sequences.items()
        }
        for stage, predictions in token_by_record_by_stage.items():
            if any(
                predictions[record_id] not in row["final_topk"]
                for record_id, row in zip(ordered_ids, ordered_rows, strict=True)
            ):
                raise AssertionError(f"{stage} shadow escaped frozen Top-5: {sample_id}")

        # Map geometry-ordered token sequences back to their immutable groups.
        predicted_by_group_by_stage = {
            stage: {
                record_to_group[record_id]: token
                for record_id, token in token_by_record_by_stage[stage].items()
            }
            for stage in by_stage_sequences
        }
        stage_exact = {
            stage: all(
                predicted_by_group_by_stage[stage][group] == target
                for group, target in target_by_group.items()
            )
            for stage in by_stage_sequences
        }
        for stage, predicted_by_group in predicted_by_group_by_stage.items():
            hits = sum(
                predicted_by_group[group] == target
                for group, target in target_by_group.items()
            )
            branch_counts[stage]["token_hits"] += hits
            branch_counts[stage]["formula_exact"] += int(stage_exact[stage])
            if stage != "existing_stack":
                changed_tokens = sum(
                    predicted_by_group[group]
                    != predicted_by_group_by_stage["existing_stack"][group]
                    for group in target_by_group
                )
                branch_counts[stage]["changed_token_count"] += changed_tokens
                branch_counts[stage]["changed_formula_count"] += int(changed_tokens > 0)
        total_exact_group_tokens += len(target_by_group)
        base_exact = stage_exact["existing_stack"]
        formula_row = {
            "sample_id": sample_id,
            "baseline_exact": base_exact,
            **stage_exact,
        }
        formula_rows.append(formula_row)

        if not stage_exact["terminal_rhs_bar"]:
            mismatches = []
            all_targets_in_top5 = True
            for record_id, prediction in zip(ordered_ids, terminal_sequence, strict=True):
                group = record_to_group[record_id]
                target = target_by_group[group]
                row = next(row for row in ordered_rows if str(row["record_id"]) == record_id)
                top5 = [str(token) for token in row["final_topk"]]
                target_rank = top5.index(target) + 1 if target in top5 else None
                all_targets_in_top5 = all_targets_in_top5 and target_rank is not None
                if prediction != target:
                    mismatches.append({
                        "stroke_indices": list(group),
                        "target": target,
                        "target_hwr_rank": target_rank,
                        "hwr_top1": top5[0],
                        "terminal_rescore_token": prediction,
                        "top5": top5,
                    })
            post_rescore_failure_details.append({
                "sample_id": sample_id,
                "cause": (
                    "target_outside_hwr_top5"
                    if not all_targets_in_top5 else "top5_present_but_rescore_sequence_wrong"
                ),
                "target_tokens": [target_by_group[record_to_group[record_id]] for record_id in ordered_ids],
                "terminal_rescore_tokens": terminal_sequence,
                "mismatched_groups": mismatches,
                "all_target_tokens_in_top5": all_targets_in_top5,
                "numeric_context_audit": numeric_audit,
                "unique_equation_audit": unique_audit,
                "combined_equation_audit": combined_audit,
                "terminal_rhs_bar_audit": terminal_audit,
            })

        changed = any(
            any(
                predicted_by_group_by_stage[stage][group]
                != predicted_by_group_by_stage["existing_stack"][group]
                for group in target_by_group
            )
            for stage in by_stage_sequences if stage != "existing_stack"
        )
        if changed:
            changed_cases.append({
                "sample_id": sample_id,
                "target_tokens": metrics["target_tokens"],
                "existing_stack_tokens": existing_sequence,
                "numeric_context_tokens": numeric_sequence,
                "unique_equation_tokens": unique_sequence,
                "numeric_then_unique_equation_tokens": combined_sequence,
                "terminal_rhs_bar_tokens": terminal_sequence,
                "formula_exact_by_stage": stage_exact,
                "numeric_context_audit": numeric_audit,
                "unique_equation_audit": unique_audit,
                "combined_equation_audit": combined_audit,
                "terminal_rhs_bar_audit": terminal_audit,
            })

    # Reconcile the independently replayed current stack with the tournament.
    for stage in REPLAY_STAGES:
        recorded = summary_arm["formula_exact_by_stage"][stage]
        replayed = stage_counts[stage]["formula_exact"]
        recorded_hits = summary_arm["token_hits_by_stage"][stage]
        replayed_hits = stage_counts[stage]["token_hits"]
        if recorded != replayed or recorded_hits != replayed_hits:
            raise AssertionError(f"aggregate semantic replay mismatch: {stage}")

    stages = (
        "existing_stack", "numeric_context", "unique_equation",
        "numeric_then_unique_equation", "terminal_rhs_bar",
    )
    stage_summaries = {}
    for stage in stages:
        exact_count = sum(bool(row[stage]) for row in formula_rows)
        transitions = _transition(formula_rows, stage, "existing_stack")
        stage_summaries[stage] = {
            "group_and_token_sequence_exact": exact_count,
            "token_hits_on_exact_groups": int(branch_counts[stage]["token_hits"]),
            "changed_token_count_vs_existing_stack": int(
                branch_counts[stage].get("changed_token_count", 0)
            ),
            "changed_formula_count_vs_existing_stack": int(
                branch_counts[stage].get("changed_formula_count", 0)
            ),
            "transition_vs_existing_stack": transitions,
            "paired_writer_cluster_bootstrap_vs_existing_stack": _writer_bootstrap(
                [
                    {"sample_id": row["sample_id"], "baseline_exact": int(row["existing_stack"]),
                     "challenger_exact": int(row[stage])}
                    for row in formula_rows
                ], writer_by_id,
            ),
        }

    formula_row_by_id = {row["sample_id"]: row for row in formula_rows}
    fast_comparison_rows = []
    for record in records:
        sample_id = str(record["sample_id"])
        fast_metrics = record["hwr_tournament"][FAST_ARM]
        fast_semantic = fast_metrics.get("semantic_guard_shadow_if_groups_exact") or {}
        fast_exact = bool(
            fast_metrics["group_exact"]
            and (fast_semantic.get("formula_exact_by_stage") or {}).get(BASE_STAGE, False)
        )
        fast_comparison_rows.append({
            "sample_id": sample_id,
            "baseline_exact": int(fast_exact),
            "challenger_exact": int(formula_row_by_id[sample_id]["terminal_rhs_bar"]),
        })
    fast_transition = _transition(
        fast_comparison_rows, "challenger_exact", "baseline_exact",
    )
    fast_formula_exact = int(
        summary["semantic_guard_shadow"]["arms"][FAST_ARM]["formula_exact_by_stage"][BASE_STAGE]
    )
    if sum(row["baseline_exact"] for row in fast_comparison_rows) != fast_formula_exact:
        raise AssertionError("Fast formula exact count does not reconcile with paired rows")
    if sum(row["challenger_exact"] for row in fast_comparison_rows) != int(
        stage_summaries["terminal_rhs_bar"]["group_and_token_sequence_exact"]
    ):
        raise AssertionError("rescored formula exact count does not reconcile with paired rows")

    source_arm = summary["hwr_tournament"]["arms"][arm]
    group_partition_wrong = len(records) - int(source_arm["group_exact"])
    group_residual_count = len(post_rescore_failure_details)
    top5_missing_residuals = sum(
        row["cause"] == "target_outside_hwr_top5"
        for row in post_rescore_failure_details
    )
    candidate_present_residuals = sum(
        row["cause"] == "top5_present_but_rescore_sequence_wrong"
        for row in post_rescore_failure_details
    )
    post_rescore_exact = int(
        stage_summaries["terminal_rhs_bar"]["group_and_token_sequence_exact"]
    )
    if group_residual_count != int(source_arm["group_exact"]) - post_rescore_exact:
        raise AssertionError("post-rescore residual formula count does not reconcile")
    if group_partition_wrong + top5_missing_residuals + candidate_present_residuals + post_rescore_exact != len(records):
        raise AssertionError("post-rescore disjoint failure causes do not reconcile")
    return {
        "schema": SCHEMA,
        "status": "consumed_development_cached_shadow_only",
        "source_summary": str(source.resolve()),
        "source_summary_sha256": _sha256(source),
        "source_data_sha256": {
            "formulas_valid": formula_sha,
            "ownership_train": ownership_sha,
        },
        "formulas": len(records),
        "arm": arm,
        "guard_group_exact": int(source_arm["group_exact"]),
        "base_semantic_replay_reconciled": True,
        "guard_rejected_group_count_increase_ids": guard_rejection_ids,
        "candidate_rescore_policy": {
            "uses_existing_hwr_top5_only": True,
            "numeric_context_maximum_competitor_probability_ratio": 0.002,
            "unique_equation_max_changed_glyphs": 3,
            "unique_equation_uses_exact_arithmetic_truth": True,
            "terminal_rhs_bar_rule": "existing frozen posthoc rule",
            "top20_candidates_used": False,
            "product_default_enabled": False,
            "crohme_training_or_tuning": False,
        },
        "group_and_token_sequence_stage_metrics": stage_summaries,
        "exact_group_semantic_stage_metrics": {
            stage: {
                "formula_exact": int(branch_counts[stage]["formula_exact"]),
                "token_hits": int(branch_counts[stage]["token_hits"]),
            }
            for stage in stages
        },
        "paired_terminal_rescore_vs_fast": {
            "fast_formula_exact": fast_formula_exact,
            "rescored_formula_exact": int(
                stage_summaries["terminal_rhs_bar"]["group_and_token_sequence_exact"]
            ),
            "transition": fast_transition,
            "paired_writer_cluster_bootstrap": _writer_bootstrap(
                fast_comparison_rows, writer_by_id,
            ),
            "interpretation": "consumed-development descriptive shadow comparison; not acceptance or promotion evidence",
        },
        "exact_group_tokens": total_exact_group_tokens,
        "candidate_preservation_rate": 1.0,
        "grouping_mutations": 0,
        "changed_cases": changed_cases,
        "post_rescore_failure_breakdown": {
            "group_partition_wrong_formulas": group_partition_wrong,
            "group_exact_residual_formulas": len(post_rescore_failure_details),
            "group_exact_cause_counts": dict(sorted(Counter(
                row["cause"] for row in post_rescore_failure_details
            ).items())),
            "disjoint_formula_causes": {
                "group_partition_wrong": group_partition_wrong,
                "group_exact_target_outside_hwr_top5": top5_missing_residuals,
                "group_exact_top5_present_but_sequence_wrong": candidate_present_residuals,
                "group_and_rescored_token_sequence_exact": post_rescore_exact,
            },
            "formulas": post_rescore_failure_details,
        },
        "promotion_eligible": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", default=GUARDED_ARM)
    args = parser.parse_args()
    report = audit(args.summary, arm=args.arm)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "formulas": report["formulas"],
        "guard_group_exact": report["guard_group_exact"],
        "stages": {
            key: value["group_and_token_sequence_exact"]
            for key, value in report["group_and_token_sequence_stage_metrics"].items()
        },
        "output": str(args.output.resolve()),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
