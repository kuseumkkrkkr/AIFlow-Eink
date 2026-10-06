#!/usr/bin/env python3
"""Compare the frozen arithmetic-equation guard on Top-5 and Top-20 shadows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from formula_layout_v1 import recontextualize_formula_rows
from semantic_equation_guard_v1 import (
    apply_semantic_equation_guard,
    apply_semantic_expression_guard,
    is_exact_arithmetic_equation,
)
from semantic_fence_guard_v1 import apply_semantic_fence_guard
from semantic_infix_guard_v1 import apply_semantic_infix_guard


SCHEMA = "aiflow-hwr-widened-semantic-guard-shadow/v1"


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _guard_formula(
    trace: dict[str, Any], k: int, *, pipeline_prefix_k: int | None = None,
    infix_policy: str = "unique", minimum_infix_probability_ratio: float = 0.0,
    maximum_operator_competitor_probability_ratio: float = 0.01,
) -> dict[str, Any]:
    sample_id = str(trace["sample_id"])
    symbols = trace["oracle_group_hwr"]["symbols"]
    targets = [str(token) for token in trace["source"]["target_tokens"]]
    if len(symbols) != len(targets):
        raise AssertionError(f"symbol/target count mismatch: {sample_id}")

    rows: list[dict[str, Any]] = []
    prefix_rows: list[dict[str, Any]] = []
    target_by_id: dict[str, str] = {}
    hwr_top1: dict[str, str] = {}
    candidate_complete = True
    target_ranks: list[int | None] = []
    for index, (symbol, target) in enumerate(zip(symbols, targets, strict=True)):
        prediction = symbol["prediction"]
        diagnostic = prediction.get("diagnostic_topk") or {}
        all_tokens = [str(token) for token in diagnostic.get("tokens", [])]
        all_probabilities = [float(value) for value in diagnostic.get("probabilities", [])]
        if len(all_tokens) < k or len(all_tokens) != len(all_probabilities):
            raise ValueError(f"missing Top-{k} rows: {sample_id}:{index}")
        tokens, probabilities = all_tokens[:k], all_probabilities[:k]
        if len(set(tokens)) != k or any(a < b for a, b in zip(probabilities, probabilities[1:])):
            raise AssertionError(f"invalid candidate ordering: {sample_id}:{index}")
        if k == 5 and (
            tokens != [str(value) for value in prediction["top5"]]
            or probabilities != [float(value) for value in prediction["top5_probabilities"]]
        ):
            raise AssertionError(f"Top-5 trace differs from saved output: {sample_id}:{index}")
        rank = next((position + 1 for position, token in enumerate(all_tokens) if token == target), None)
        target_ranks.append(rank)
        candidate_complete &= rank is not None and rank <= k

        box = symbol["preprocessing"]["raw_bbox"]
        left, right = float(box["left"]), float(box["right"])
        top, bottom = float(box["top"]), float(box["bottom"])
        record_id = f"{sample_id}:{index}"
        rows.append({
            "record_id": record_id,
            "formula_id": sample_id,
            "final_topk": tokens,
            "final_topk_probabilities": probabilities,
            "geometry": {
                "left": left, "right": right, "top": top, "bottom": bottom,
                "center_x": (left + right) / 2.0,
                "center_y": (top + bottom) / 2.0,
                "width_rel": max(right - left, 1e-6),
                "height_rel": max(bottom - top, 1e-6),
            },
        })
        if pipeline_prefix_k is not None:
            if pipeline_prefix_k > k:
                raise ValueError("pipeline prefix K cannot exceed candidate K")
            prefix_rows.append({
                **rows[-1],
                "final_topk": tokens[:pipeline_prefix_k],
                "final_topk_probabilities": probabilities[:pipeline_prefix_k],
            })
        target_by_id[record_id] = target
        hwr_top1[record_id] = tokens[0]

    wide_contextual_rows, layout_audit = recontextualize_formula_rows(rows)
    before, guard_audit = apply_semantic_equation_guard(wide_contextual_rows, hwr_top1)
    if pipeline_prefix_k is None:
        early_rows = wide_contextual_rows
        pipeline_rows = wide_contextual_rows
    else:
        early_rows, layout_audit = recontextualize_formula_rows(prefix_rows)
        early_by_id = {str(row["record_id"]): row for row in early_rows}
        pipeline_rows = [
            {**row, "context": dict(early_by_id[str(row["record_id"])]["context"])}
            for row in rows
        ]
    after_fence, fence_audit = apply_semantic_fence_guard(early_rows, hwr_top1)
    if infix_policy == "unique":
        after_infix, infix_audit = apply_semantic_infix_guard(
            early_rows, after_fence,
            minimum_probability_ratio=minimum_infix_probability_ratio,
        )
    elif infix_policy == "operator_argmax":
        after_infix, infix_audit = apply_semantic_infix_guard(
            early_rows, after_fence,
            minimum_probability_ratio=minimum_infix_probability_ratio,
            ambiguity_policy="top_probability",
        )
    elif infix_policy == "operator_relative":
        after_infix, infix_audit = apply_semantic_infix_guard(
            early_rows, after_fence,
            minimum_probability_ratio=minimum_infix_probability_ratio,
            ambiguity_policy="relative_dominance",
            maximum_competitor_probability_ratio=maximum_operator_competitor_probability_ratio,
        )
    elif infix_policy == "operator_numeric_context":
        after_infix, infix_audit = apply_semantic_infix_guard(
            early_rows, after_fence,
            minimum_probability_ratio=minimum_infix_probability_ratio,
            ambiguity_policy="numeric_context_dominance",
            maximum_competitor_probability_ratio=maximum_operator_competitor_probability_ratio,
        )
    else:
        raise ValueError(f"unsupported infix policy: {infix_policy}")
    after_equation, equation_audit = apply_semantic_equation_guard(pipeline_rows, after_infix)
    after_expression, expression_audit = apply_semantic_expression_guard(
        pipeline_rows, after_equation,
    )
    ordered_ids = [str(row["record_id"]) for row in early_rows]
    before_tokens = [hwr_top1[record_id] for record_id in ordered_ids]
    after_tokens = [before[record_id] for record_id in ordered_ids]
    ordered_targets = [target_by_id[record_id] for record_id in ordered_ids]
    wide_by_id = {str(row["record_id"]): row for row in wide_contextual_rows}
    if not all(token in wide_by_id[record_id]["final_topk"] for record_id, token in zip(ordered_ids, after_tokens, strict=True)):
        raise AssertionError(f"candidate preservation failed: {sample_id}")
    for change in guard_audit.get("changes", []):
        if not is_exact_arithmetic_equation(tuple(str(token) for token in change["after"])):
            raise AssertionError(f"guard emitted a non-equation: {sample_id}")

    stage_predictions = {
        "hwr_top1": hwr_top1,
        "after_fence": after_fence,
        "after_infix": after_infix,
        "after_equation": after_equation,
        "after_expression": after_expression,
    }
    pipeline_stages = {}
    for stage, predictions in stage_predictions.items():
        tokens = [predictions[record_id] for record_id in ordered_ids]
        if not all(token in wide_by_id[record_id]["final_topk"] for record_id, token in zip(ordered_ids, tokens, strict=True)):
            raise AssertionError(f"{stage} candidate preservation failed: {sample_id}")
        pipeline_stages[stage] = {
            "token_hits": sum(a == b for a, b in zip(tokens, ordered_targets, strict=True)),
            "formula_exact": tokens == ordered_targets,
            "tokens": tokens,
            "changed_positions": [
                index for index, (old, new) in enumerate(zip(before_tokens, tokens, strict=True))
                if old != new
            ],
        }

    changed_positions = [index for index, pair in enumerate(zip(before_tokens, after_tokens, strict=True)) if pair[0] != pair[1]]
    return {
        "sample_id": sample_id,
        "candidate_complete": candidate_complete,
        "target_ranks": target_ranks,
        "before_exact": before_tokens == ordered_targets,
        "after_exact": after_tokens == ordered_targets,
        "before_token_hits": sum(a == b for a, b in zip(before_tokens, ordered_targets, strict=True)),
        "after_token_hits": sum(a == b for a, b in zip(after_tokens, ordered_targets, strict=True)),
        "changed_positions": [
            {
                "position": index,
                "before": before_tokens[index],
                "after": after_tokens[index],
                "target": ordered_targets[index],
                "target_rank": target_ranks[int(ordered_ids[index].rsplit(":", 1)[1])],
            }
            for index in changed_positions
        ],
        "guard_audit": guard_audit,
        "candidate_preservation": True,
        "pipeline_stages": pipeline_stages,
        "pipeline_guard_audits": {
            "fence": fence_audit,
            "infix": infix_audit,
            "equation": equation_audit,
            "expression": expression_audit,
        },
        "layout_audit": layout_audit,
    }


def audit(wide_path: Path, baseline_path: Path) -> dict[str, Any]:
    wide = _jsonl(wide_path)
    baseline = _jsonl(baseline_path)
    wide_by_id = {str(row["sample_id"]): row for row in wide}
    baseline_by_id = {str(row["sample_id"]): row for row in baseline}
    if set(wide_by_id) != set(baseline_by_id):
        raise AssertionError("Top-20 and baseline formula IDs differ")

    by_k: dict[str, list[dict[str, Any]]] = {
        "5": [], "20": [], "hybrid_5_front_20_equation": [],
        "top20_operator_argmax_shadow": [],
    }
    for sample_id, trace in wide_by_id.items():
        prior = baseline_by_id[sample_id]
        old_by_group = {tuple(row["stroke_indices"]): row for row in prior["oracle_group_hwr"]["symbols"]}
        for symbol in trace["oracle_group_hwr"]["symbols"]:
            old = old_by_group[tuple(symbol["stroke_indices"])]
            prediction = symbol["prediction"]
            if prediction["top5"] != old["prediction"]["top5"] or prediction["top5_probabilities"] != old["prediction"]["top5_probabilities"]:
                raise AssertionError(f"Top-5 logits changed: {sample_id}")
        by_k["5"].append(_guard_formula(trace, 5))
        by_k["20"].append(_guard_formula(trace, 20))
        by_k["hybrid_5_front_20_equation"].append(
            _guard_formula(trace, 20, pipeline_prefix_k=5)
        )
        by_k["top20_operator_argmax_shadow"].append(
            _guard_formula(trace, 20, infix_policy="operator_argmax")
        )

    metrics: dict[str, dict[str, Any]] = {}
    pipeline_stage_names = ("hwr_top1", "after_fence", "after_infix", "after_equation", "after_expression")
    for k, rows in by_k.items():
        metrics[k] = {
            "candidate_complete_formulas": sum(bool(row["candidate_complete"]) for row in rows),
            "hwr_top1_token_hits": sum(int(row["before_token_hits"]) for row in rows),
            "after_guard_token_hits": sum(int(row["after_token_hits"]) for row in rows),
            "hwr_top1_formula_exact": sum(bool(row["before_exact"]) for row in rows),
            "after_guard_formula_exact": sum(bool(row["after_exact"]) for row in rows),
            "guard_changed_formulas": sum(bool(row["changed_positions"]) for row in rows),
            "guard_audit_totals": {
                key: sum(int(row["guard_audit"].get(key, 0)) for row in rows)
                for key in ("eligible_formulas", "finalized_formulas", "changed_glyphs", "no_valid_equation", "skipped_ambiguous", "skipped_symbolic_baseline")
            },
            "existing_guard_pipeline": {
                stage: {
                    "token_hits": sum(int(row["pipeline_stages"][stage]["token_hits"]) for row in rows),
                    "formula_exact": sum(bool(row["pipeline_stages"][stage]["formula_exact"]) for row in rows),
                    "formulas_changed_from_hwr_top1": sum(bool(row["pipeline_stages"][stage]["changed_positions"]) for row in rows),
                }
                for stage in pipeline_stage_names
            },
        }
    exact_ids = {
        k: {row["sample_id"] for row in rows if row["after_exact"]}
        for k, rows in by_k.items()
    }
    k5_changed = {row["sample_id"] for row in by_k["5"] if row["changed_positions"]}
    k20_changed = {row["sample_id"] for row in by_k["20"] if row["changed_positions"]}
    k5_complete = {row["sample_id"] for row in by_k["5"] if row["candidate_complete"]}
    k20_complete = {row["sample_id"] for row in by_k["20"] if row["candidate_complete"]}
    newly_complete_details = []
    for sample_id in sorted(k20_complete - k5_complete):
        trace = wide_by_id[sample_id]
        missing_slots = []
        for index, (symbol, target) in enumerate(zip(
            trace["oracle_group_hwr"]["symbols"], trace["source"]["target_tokens"], strict=True,
        )):
            diagnostic = symbol["prediction"]["diagnostic_topk"]
            tokens = [str(token) for token in diagnostic["tokens"]]
            target = str(target)
            rank = next((position + 1 for position, token in enumerate(tokens) if token == target), None)
            if rank is not None and 5 < rank <= 20:
                missing_slots.append({
                    "position": index,
                    "target": target,
                    "target_rank": rank,
                    "top1": tokens[0],
                    "top5": tokens[:5],
                })
        guarded = next(row for row in by_k["20"] if row["sample_id"] == sample_id)
        newly_complete_details.append({
            "sample_id": sample_id,
            "top5_missing_slots_recovered_by_candidate_expansion": missing_slots,
            "equation_guard_changed_formula": sample_id in k20_changed,
            "formula_exact_after_guard": guarded["after_exact"],
            "guard_changed_positions": guarded["changed_positions"],
            "existing_guard_audit_counts": guarded["pipeline_guard_audits"],
        })
    pipeline_comparison = {}
    for stage in pipeline_stage_names:
        exact_by_k = {
            k: {
                row["sample_id"] for row in rows
                if row["pipeline_stages"][stage]["formula_exact"]
            }
            for k, rows in by_k.items()
        }
        pipeline_comparison[stage] = {
            "k20_recovered_vs_k5": sorted(exact_by_k["20"] - exact_by_k["5"]),
            "k20_lost_vs_k5": sorted(exact_by_k["5"] - exact_by_k["20"]),
            "k5_infix_ambiguous_operator_count": sum(
                int(row["pipeline_guard_audits"]["infix"].get("skipped_ambiguous_operator", 0))
                for row in by_k["5"]
            ),
            "k20_infix_ambiguous_operator_count": sum(
                int(row["pipeline_guard_audits"]["infix"].get("skipped_ambiguous_operator", 0))
                for row in by_k["20"]
            ),
        }
    operator_argmax_comparison = {}
    for stage in pipeline_stage_names:
        exact_by_arm = {
            arm: {
                row["sample_id"] for row in rows
                if row["pipeline_stages"][stage]["formula_exact"]
            }
            for arm, rows in by_k.items()
            if arm in {"5", "20", "top20_operator_argmax_shadow"}
        }
        operator_argmax_comparison[stage] = {
            "argmax_recovered_vs_top5": sorted(
                exact_by_arm["top20_operator_argmax_shadow"] - exact_by_arm["5"]
            ),
            "argmax_lost_vs_top5": sorted(
                exact_by_arm["5"] - exact_by_arm["top20_operator_argmax_shadow"]
            ),
            "argmax_recovered_vs_top20_unique": sorted(
                exact_by_arm["top20_operator_argmax_shadow"] - exact_by_arm["20"]
            ),
            "argmax_lost_vs_top20_unique": sorted(
                exact_by_arm["20"] - exact_by_arm["top20_operator_argmax_shadow"]
            ),
            "resolved_ambiguous_operator_slots": sum(
                int(row["pipeline_guard_audits"]["infix"].get("resolved_ambiguous_operator", 0))
                for row in by_k["top20_operator_argmax_shadow"]
            ),
        }
    checks = {
        "formula_ID_sets_match": set(wide_by_id) == set(baseline_by_id),
        "top5_logits_identical_to_baseline": True,
        "candidate_completeness_monotonic": metrics["5"]["candidate_complete_formulas"] <= metrics["20"]["candidate_complete_formulas"],
        "all_selected_tokens_preserved": all(
            row["guard_audit"].get("candidate_preservation_rate") == 1.0
            and row["guard_audit"].get("new_tokens") == 0
            and row["guard_audit"].get("deleted_glyphs") == 0
            and row["guard_audit"].get("grouping_mutations") == 0
            for rows in by_k.values() for row in rows
        ),
        "formula_exact_recovery_regression_ids_reconcile": (
            len(exact_ids["20"] - exact_ids["5"]) == len(set(exact_ids["20"] - exact_ids["5"]))
            and len(exact_ids["5"] - exact_ids["20"]) == len(set(exact_ids["5"] - exact_ids["20"]))
        ),
        "newly_candidate_complete_details_reconcile": len(newly_complete_details) == len(k20_complete - k5_complete),
        "no_pipeline_formula_regressions_vs_hwr_top1": all(
            not row["pipeline_stages"]["hwr_top1"]["formula_exact"]
            or all(row["pipeline_stages"][stage]["formula_exact"] for stage in pipeline_stage_names[1:])
            for rows in by_k.values() for row in rows
        ),
        "hybrid_prefix5_pipeline_is_identical_to_top5": all(
            by_k["hybrid_5_front_20_equation"][index]["pipeline_stages"]
            == by_k["5"][index]["pipeline_stages"]
            for index in range(len(by_k["5"]))
        ),
        "operator_argmax_has_no_formula_regressions_vs_top5": all(
            not by_k["5"][index]["pipeline_stages"][stage]["formula_exact"]
            or by_k["top20_operator_argmax_shadow"][index]["pipeline_stages"][stage]["formula_exact"]
            for index in range(len(by_k["5"]))
            for stage in pipeline_stage_names
        ),
    }
    changed_details = {
        k: [row for row in rows if row["changed_positions"]]
        for k, rows in by_k.items()
    }
    return {
        "schema": SCHEMA,
        "scope": "consumed-development posthoc shadow only; frozen equation-guard constants; no training, CROHME, tuning, or runtime promotion",
        "inputs": {
            "top20_trace": {"path": str(wide_path), "sha256": _sha256(wide_path)},
            "top5_baseline_trace": {"path": str(baseline_path), "sha256": _sha256(baseline_path)},
        },
        "configuration": {
            "candidate_ks": [5, 20],
            "guard": "existing frozen guard defaults",
            "base_prediction": "HWR rank-1",
            "ordering": "formula_layout_v1 geometry order",
            "hybrid_arm": "fence and infix use Top-5; equation and expression use Top-20",
            "operator_argmax_shadow": "Top-20; retains geometry/role checks; resolves multiple operators by HWR probability",
        },
        "summary": {
            "formulas": len(wide),
            "tokens": sum(len(row["source"]["target_tokens"]) for row in wide),
            "metrics_by_k": metrics,
            "recovered_after_guard_at_k20_vs_k5": sorted(exact_ids["20"] - exact_ids["5"]),
            "regressed_after_guard_at_k20_vs_k5": sorted(exact_ids["5"] - exact_ids["20"]),
            "newly_candidate_complete_from_k5_to_k20": newly_complete_details,
            "guard_changed_formula_sets_identical_at_k5_and_k20": k5_changed == k20_changed,
            "existing_guard_pipeline_k20_vs_k5": pipeline_comparison,
            "operator_argmax_shadow_vs_existing": operator_argmax_comparison,
            "guard_changed_formula_ids_by_k": {k: sorted(row["sample_id"] for row in rows) for k, rows in changed_details.items()},
            "changed_formula_details_by_k": changed_details,
        },
        "pipeline_formula_level_by_k": {
            k: [
                {
                    "sample_id": row["sample_id"],
                    "stages": row["pipeline_stages"],
                    "guard_changes": {
                        name: audit.get("changes", [])
                        for name, audit in row["pipeline_guard_audits"].items()
                    },
                    "guard_audit_counts": {
                        name: {
                            key: value for key, value in audit.items()
                            if isinstance(value, int)
                        }
                        for name, audit in row["pipeline_guard_audits"].items()
                    },
                }
                for row in rows
            ]
            for k, rows in by_k.items()
        },
        "verification": {"checks": checks, "all_checks_pass": all(checks.values())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--baseline-traces", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite output: {args.output}")
    report = audit(args.traces, args.baseline_traces)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": report["summary"]["metrics_by_k"], "verification": report["verification"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
