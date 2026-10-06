#!/usr/bin/env python3
"""Shadow-test the frozen formula context scorer on current-HWR candidates.

The context checkpoint is loaded against its original HWR hash. Current HWR
Top-5 rows are then supplied only as an explicitly mismatched transfer probe;
this never changes runtime defaults or model weights.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA = "aiflow-hwr-context-transfer-microscope/v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _writer_group_digest(writer_id: str) -> str:
    encoded = json.dumps(
        ["project_owned_writer", writer_id], ensure_ascii=False,
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def _rows(trace: dict[str, Any], *, fast_partition: bool = False) -> list[dict[str, Any]]:
    sample_id = str(trace["sample_id"])
    symbols = (
        trace["grouping"]["selected_symbols"]
        if fast_partition else trace["oracle_group_hwr"]["symbols"]
    )
    boxes = [symbol["preprocessing"]["raw_bbox"] for symbol in symbols]
    left = min(float(box["left"]) for box in boxes)
    right = max(float(box["right"]) for box in boxes)
    top = min(float(box["top"]) for box in boxes)
    bottom = max(float(box["bottom"]) for box in boxes)
    width, height = max(right - left, 1e-8), max(bottom - top, 1e-8)
    rows = []
    for index, (symbol, box) in enumerate(zip(symbols, boxes, strict=True)):
        l, r = float(box["left"]), float(box["right"])
        t, b = float(box["top"]), float(box["bottom"])
        prediction = symbol["hwr_prediction"] if fast_partition else symbol["prediction"]
        rows.append({
            "record_id": f"{sample_id}:{'fast:' if fast_partition else ''}{index}",
            "formula_id": sample_id,
            "final_topk": [str(value) for value in prediction["top5"]],
            "final_topk_probabilities": [float(value) for value in prediction["top5_probabilities"]],
            "context": {"index": index, "length": len(symbols)},
            "geometry": {
                "left": l, "right": r, "top": t, "bottom": b,
                "center_x": ((l + r) / 2.0 - left) / width,
                "center_y": ((t + b) / 2.0 - top) / height,
                "width_rel": (r - l) / width,
                "height_rel": (b - t) / height,
            },
        })
    return rows


def _scores(rows: list[dict[str, Any]], target: list[str], predictions: dict[str, str]) -> dict[str, Any]:
    tokens = [str(predictions[row["record_id"]]) for row in rows]
    candidate_violations = sum(
        token not in row["final_topk"] for token, row in zip(tokens, rows, strict=True)
    )
    return {
        "token_hits": sum(token == truth for token, truth in zip(tokens, target, strict=True)),
        "formula_exact": tokens == target,
        "tokens": tokens,
        "candidate_violations": candidate_violations,
    }


def _sequence_score(
    rows: list[dict[str, Any]], target: list[str], predictions: dict[str, str],
) -> dict[str, Any]:
    tokens = [str(predictions[row["record_id"]]) for row in rows]
    same_length = len(tokens) == len(target)
    return {
        "formula_exact": tokens == target,
        "output_symbol_count": len(tokens),
        "target_symbol_count": len(target),
        "length_delta": len(tokens) - len(target),
        "token_hits_if_aligned": (
            sum(token == truth for token, truth in zip(tokens, target)) if same_length else None
        ),
        "candidate_violations": sum(
            token not in row["final_topk"] for token, row in zip(tokens, rows, strict=True)
        ),
    }


def _checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload.get("state_dict")
    if not isinstance(state, dict) or not state or not all(
        isinstance(value, torch.Tensor) for value in state.values()
    ):
        raise ValueError(f"checkpoint has no tensor state_dict: {path}")
    return state


def _state_transfer_audit(old_path: Path, current_path: Path) -> dict[str, Any]:
    import torch

    old, current = _checkpoint_state(old_path), _checkpoint_state(current_path)
    if set(old) != set(current):
        raise AssertionError("HWR checkpoint state_dict keys differ")
    changed = sorted(key for key in old if not torch.equal(old[key], current[key]))
    expected = ["math_head.bias", "math_head.weight"]
    shared_encoder_identical = set(changed) == set(expected)
    return {
        "changed_state_keys": changed,
        "only_math_head_changed": shared_encoder_identical,
        "shared_encoder_identical": shared_encoder_identical,
    }


def _training_identity_audit(
    training_candidates_path: Path, traces: list[dict[str, Any]],
    evaluation_ownership_path: Path,
) -> dict[str, Any]:
    trained: dict[str, list[tuple[int, str]]] = {}
    cache_keys: set[str] = set()
    training_writer_groups: set[str] = set()
    training_rows = 0
    training_rows_with_writer_group = 0
    with gzip.open(training_candidates_path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            training_rows += 1
            cache_keys.update(row)
            trained.setdefault(str(row["formula_id"]), []).append(
                (int(row["context"]["index"]), str(row["label"]))
            )
            if row.get("writer_group") is not None:
                training_rows_with_writer_group += 1
                training_writer_groups.add(str(row["writer_group"]))
    current = {
        str(trace["sample_id"]): [str(token) for token in trace["source"]["target_tokens"]]
        for trace in traces
    }
    evaluation_ownership = [
        json.loads(line)
        for line in evaluation_ownership_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    accepted_ownership = [row for row in evaluation_ownership if row.get("accepted")]
    evaluation_writer_by_id = {
        str(row["sample_id"]): _writer_group_digest(str(row["writer_id"]))
        for row in accepted_ownership
    }
    evaluation_writer_groups = set(evaluation_writer_by_id.values())
    evaluation_sample_ids_match_trace_ids = set(evaluation_writer_by_id) == set(current)
    training_writer_groups_complete = (
        training_rows > 0 and training_rows_with_writer_group == training_rows
        and bool(training_writer_groups)
    )
    writer_group_overlap = sorted(training_writer_groups & evaluation_writer_groups)
    writer_group_overlap_auditable = (
        evaluation_sample_ids_match_trace_ids and training_writer_groups_complete
    )
    id_intersection = sorted(set(trained) & set(current))
    same_target = [
        sample_id for sample_id in id_intersection
        if [token for _, token in sorted(trained[sample_id])] == current[sample_id]
    ]
    mismatched = [sample_id for sample_id in id_intersection if sample_id not in set(same_target)]
    return {
        "training_formula_count": len(trained),
        "evaluation_formula_count": len(current),
        "shared_formula_id_strings": len(id_intersection),
        "same_ordered_target_sequences_on_shared_ids": len(same_target),
        "shared_ids_with_different_target_sequences": len(mismatched),
        "target_mismatch_examples": [
            {
                "formula_id_string": sample_id,
                "training_target": [token for _, token in sorted(trained[sample_id])],
                "evaluation_target": current[sample_id],
            }
            for sample_id in mismatched[:8]
        ],
        "raw_ink_overlap_auditable": bool({"strokes", "raw_ink_sha256"} & cache_keys),
        "writer_id_overlap_auditable": writer_group_overlap_auditable,
        "writer_group_hash_contract": "character_tensor_v1._digest(['project_owned_writer', writer_id])[:24]",
        "training_writer_group_count": len(training_writer_groups),
        "evaluation_writer_group_count": len(evaluation_writer_groups),
        "evaluation_accepted_ownership_rows": len(accepted_ownership),
        "evaluation_sample_ids_match_trace_ids": evaluation_sample_ids_match_trace_ids,
        "training_rows_have_writer_group": training_writer_groups_complete,
        "writer_group_overlap_auditable": writer_group_overlap_auditable,
        "writer_group_overlap_count": len(writer_group_overlap),
        "writer_group_overlap_hashes": writer_group_overlap,
        "writer_group_disjointness_verified": writer_group_overlap_auditable and not writer_group_overlap,
        "evaluation_ownership_sha256": _sha256(evaluation_ownership_path),
        "same_target_sequence_is_proof_of_same_ink": False,
        "interpretation": (
            "training and evaluation share writer groups; this transfer result is not writer-disjoint"
            if writer_group_overlap else
            "writer groups are disjoint but raw-ink overlap remains unverified"
            if writer_group_disjointness_verified else
            "formula_id strings are not a valid join key here; writer-group disjointness is unverified"
        ),
    }


def audit(
    traces_path: Path, residual_path: Path,
    context_checkpoint: Path, compatible_hwr_checkpoint: Path,
    current_hwr_checkpoint: Path, training_candidates_path: Path,
    evaluation_ownership_path: Path,
) -> dict[str, Any]:
    import torch
    from semantic_equation_guard_v1 import apply_semantic_equation_guard
    from semantic_fence_guard_v1 import apply_semantic_fence_guard
    from semantic_infix_guard_v1 import apply_semantic_infix_guard
    from train_owned_formula_context_v1 import (
        decide_owned_formula_rows_supported_exact,
        load_owned_formula_context,
    )

    torch.set_num_threads(1)
    device = torch.device("cpu")
    traces = [
        json.loads(line)
        for line in traces_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    residual = json.loads(residual_path.read_text(encoding="utf-8"))
    residual_by_id = {str(row["sample_id"]): row for row in residual["formula_level"]}
    candidate_rank_failures = {
        str(row["sample_id"])
        for row in residual["formula_level"]
        if row["fast_group_exact"] and row["top5_complete"]
        and not row["after_equation_guard_exact"]
    }
    model, contract, payload = load_owned_formula_context(
        context_checkpoint, compatible_hwr_checkpoint, device,
    )
    original_expected_sha = str(payload["hwr_checkpoint_sha256"])
    current_hwr_sha = str(
        json.loads((traces_path.parent / "summary.json").read_text(encoding="utf-8"))
        ["runtime"]["checkpoint_sha256"]
    )
    if original_expected_sha != _sha256(compatible_hwr_checkpoint):
        raise AssertionError("context checkpoint did not load against its bound HWR")
    if current_hwr_sha != _sha256(current_hwr_checkpoint):
        raise AssertionError("trace checkpoint hash does not match supplied current HWR")
    transfer_audit = _state_transfer_audit(compatible_hwr_checkpoint, current_hwr_checkpoint)
    identity_audit = _training_identity_audit(
        training_candidates_path, traces, evaluation_ownership_path,
    )

    cases = []
    fast_partition_cases = []
    stage_totals = {name: {"hits": 0, "exact": 0} for name in (
        "hwr_top1", "existing_fence_infix", "context", "context_fence_infix",
        "context_fence_infix_equation_shadow",
    )}
    changes_vs_current_top1 = []
    candidate_violations_by_stage = {name: 0 for name in (
        "hwr_top1", "existing_fence_infix", "context", "context_fence_infix",
        "context_fence_infix_equation_shadow",
    )}
    context_audits = []
    formula_stage_exact: dict[str, dict[str, bool]] = {}

    for trace in traces:
        sample_id = str(trace["sample_id"])
        target = [str(value) for value in trace["source"]["target_tokens"]]
        rows = _rows(trace)
        hwr_top1 = {
            row["record_id"]: row["final_topk"][0] for row in rows
        }
        current_fence = trace["semantic_guard_shadow"]["stage_tokens"]["after_fence_guard"]
        current_infix = trace["semantic_guard_shadow"]["stage_tokens"]["after_infix_guard"]
        current_guard = {
            row["record_id"]: token
            for row, token in zip(rows, current_infix, strict=True)
        }
        context_prediction, context_audit = decide_owned_formula_rows_supported_exact(
            model, contract, payload, rows, device, batch_size=128,
        )
        context_fence, fence_audit = apply_semantic_fence_guard(rows, context_prediction)
        context_infix, infix_audit = apply_semantic_infix_guard(
            rows, context_fence,
            minimum_probability_ratio=float(payload["configuration"]["candidate_probability_ratio_floor"]),
        )
        equation, equation_audit = apply_semantic_equation_guard(rows, context_infix)
        stages = {
            "hwr_top1": hwr_top1,
            "existing_fence_infix": current_guard,
            "context": context_prediction,
            "context_fence_infix": context_infix,
            "context_fence_infix_equation_shadow": equation,
        }
        stage_scores = {
            name: _scores(rows, target, predictions)
            for name, predictions in stages.items()
        }
        formula_stage_exact[sample_id] = {
            name: bool(score["formula_exact"]) for name, score in stage_scores.items()
        }
        for name, score in stage_scores.items():
            stage_totals[name]["hits"] += score["token_hits"]
            stage_totals[name]["exact"] += int(score["formula_exact"])
            candidate_violations_by_stage[name] += score["candidate_violations"]

        changed = []
        for index, row in enumerate(rows):
            before = hwr_top1[row["record_id"]]
            after = context_infix[row["record_id"]]
            if before == after:
                continue
            top5 = row["final_topk"]
            target_token = target[index]
            changed.append({
                "position": index,
                "stroke_indices": trace["oracle_group_hwr"]["symbols"][index]["stroke_indices"],
                "before": before,
                "after": after,
                "target": target_token,
                "before_correct": before == target_token,
                "after_correct": after == target_token,
                "candidate_rank": top5.index(after) + 1,
                "candidate_probability": row["final_topk_probabilities"][top5.index(after)],
                "top5": top5,
            })
        if changed:
            changes_vs_current_top1.append({"sample_id": sample_id, "changes": changed})
        if sample_id in candidate_rank_failures or changed:
            cases.append({
                "sample_id": sample_id,
                "target_tokens": target,
                "current_top1_tokens": stage_scores["hwr_top1"]["tokens"],
                "current_fence_infix_tokens": stage_scores["existing_fence_infix"]["tokens"],
                "context_tokens": stage_scores["context"]["tokens"],
                "context_fence_infix_tokens": stage_scores["context_fence_infix"]["tokens"],
                "context_equation_shadow_tokens": stage_scores["context_fence_infix_equation_shadow"]["tokens"],
                "stage_exact": {name: score["formula_exact"] for name, score in stage_scores.items()},
                "group_exact": bool(trace["grouping"]["fast_group_exact"]),
                "target_in_current_top5": all(symbol["prediction"]["target_in_top5"] for symbol in trace["oracle_group_hwr"]["symbols"]),
                "context_changes": changed,
                "context_audit": context_audit,
                "fence_audit": fence_audit,
                "infix_audit": infix_audit,
                "equation_audit": equation_audit,
                "failure_class": _failure_class(residual_by_id[sample_id]),
            })
        context_audits.append(context_audit)

        # Counterfactual: preserve the actual Fast partition and only rerank tokens.
        fast_rows = _rows(trace, fast_partition=True)
        fast_base = {
            row["record_id"]: str(symbol["selected_token"])
            for row, symbol in zip(fast_rows, trace["grouping"]["selected_symbols"], strict=True)
        }
        fast_fence, fast_fence_audit = apply_semantic_fence_guard(fast_rows, fast_base)
        fast_infix, fast_infix_audit = apply_semantic_infix_guard(
            fast_rows, fast_fence,
            minimum_probability_ratio=float(payload["configuration"]["candidate_probability_ratio_floor"]),
        )
        fast_equation, fast_equation_audit = apply_semantic_equation_guard(fast_rows, fast_infix)
        fast_context, fast_context_audit = decide_owned_formula_rows_supported_exact(
            model, contract, payload, fast_rows, device, batch_size=128,
        )
        fast_context_fence, _ = apply_semantic_fence_guard(fast_rows, fast_context)
        fast_context_infix, _ = apply_semantic_infix_guard(
            fast_rows, fast_context_fence,
            minimum_probability_ratio=float(payload["configuration"]["candidate_probability_ratio_floor"]),
        )
        fast_context_equation, fast_context_equation_audit = apply_semantic_equation_guard(
            fast_rows, fast_context_infix,
        )
        fast_stages = {
            "fast_selected": fast_base,
            "fast_fence_infix_equation": fast_equation,
            "context_fence_infix_equation": fast_context_equation,
        }
        fast_stage_scores = {
            name: _sequence_score(fast_rows, target, predictions)
            for name, predictions in fast_stages.items()
        }
        fast_partition_cases.append({
            "sample_id": sample_id,
            "fast_group_exact": bool(trace["grouping"]["fast_group_exact"]),
            "fast_group_count": len(fast_rows),
            "target_token_count": len(target),
            "group_count_delta": len(fast_rows) - len(target),
            "stage_scores": fast_stage_scores,
            "context_changes": sum(
                fast_base[row["record_id"]] != fast_context_equation[row["record_id"]]
                for row in fast_rows
            ),
            "fast_guard_audits": {
                "fence": fast_fence_audit,
                "infix": fast_infix_audit,
                "equation": fast_equation_audit,
                "context": fast_context_audit,
                "context_equation": fast_context_equation_audit,
            },
        })

    metric_names = ["hwr_top1", "existing_fence_infix", "context", "context_fence_infix", "context_fence_infix_equation_shadow"]
    metrics = {
        name: {
            "token_hits": stage_totals[name]["hits"],
            "token_accuracy": stage_totals[name]["hits"] / 579,
            "formula_exact": stage_totals[name]["exact"],
            "formula_exact_rate": stage_totals[name]["exact"] / len(traces),
        }
        for name in metric_names
    }
    base = metrics["existing_fence_infix"]
    shadow = metrics["context_fence_infix"]
    equations = metrics["context_fence_infix_equation_shadow"]
    for name in stage_totals:
        stage_totals[name]["candidate_violations"] = candidate_violations_by_stage[name]
    failure_class_rows: dict[str, list[str]] = {}
    for row in residual["formula_level"]:
        label = _failure_class(row)
        failure_class_rows.setdefault(label, []).append(str(row["sample_id"]))
    formula_by_id = {str(row["sample_id"]): row for row in residual["formula_level"]}
    by_class = {}
    for label, sample_ids in failure_class_rows.items():
        baseline_exact = sum(bool(formula_by_id[key]["after_equation_guard_exact"]) for key in sample_ids)
        context_exact = sum(formula_stage_exact[key]["context_fence_infix_equation_shadow"] for key in sample_ids)
        by_class[label] = {
            "formulas": len(sample_ids),
            "equation_guard_baseline_exact": baseline_exact,
            "context_shadow_exact": context_exact,
            "exact_delta": context_exact - baseline_exact,
            "sample_ids": sample_ids,
        }
    evaluation_ownership_rows = [
        json.loads(line)
        for line in evaluation_ownership_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    writer_by_sample = {
        str(row["sample_id"]): _writer_group_digest(str(row["writer_id"]))
        for row in evaluation_ownership_rows if row.get("accepted")
    }
    if set(writer_by_sample) != set(formula_by_id):
        raise AssertionError("failure-class writer join does not match residual formula IDs")
    overlapping_writer_groups = set(identity_audit["writer_group_overlap_hashes"])
    failure_class_writer_cohorts = {}
    for label, sample_ids in failure_class_rows.items():
        cohort_ids = {
            "context_training_writer_disjoint": [
                sample_id for sample_id in sample_ids
                if writer_by_sample[sample_id] not in overlapping_writer_groups
            ],
            "context_training_writer_overlap": [
                sample_id for sample_id in sample_ids
                if writer_by_sample[sample_id] in overlapping_writer_groups
            ],
        }
        failure_class_writer_cohorts[label] = {
            cohort: {
                "formulas": len(ids),
                "baseline_exact": sum(
                    bool(formula_by_id[sample_id]["after_equation_guard_exact"])
                    for sample_id in ids
                ),
                "context_exact": sum(
                    formula_stage_exact[sample_id]["context_fence_infix_equation_shadow"]
                    for sample_id in ids
                ),
                "formula_gains": sum(
                    not bool(formula_by_id[sample_id]["after_equation_guard_exact"])
                    and formula_stage_exact[sample_id]["context_fence_infix_equation_shadow"]
                    for sample_id in ids
                ),
                "formula_regressions": sum(
                    bool(formula_by_id[sample_id]["after_equation_guard_exact"])
                    and not formula_stage_exact[sample_id]["context_fence_infix_equation_shadow"]
                    for sample_id in ids
                ),
            }
            for cohort, ids in cohort_ids.items()
        }
    checks = {
        "all_149_formulas_processed": len(traces) == 149,
        "context_checkpoint_original_hwr_hash_verified": original_expected_sha == _sha256(compatible_hwr_checkpoint),
        "current_hwr_differs_from_context_bound_hwr": current_hwr_sha != original_expected_sha,
        "shared_encoder_checkpoint_diff_is_math_head_only": transfer_audit["only_math_head_changed"],
        "current_top1_baseline_matches_microscope": metrics["hwr_top1"]["token_hits"] == 470 and metrics["hwr_top1"]["formula_exact"] == 76,
        "current_fence_infix_matches_microscope": base["token_hits"] == 485 and base["formula_exact"] == 84,
        "all_context_outputs_preserve_current_top5": candidate_violations_by_stage["context"] == 0,
        "all_guarded_context_outputs_preserve_current_top5": candidate_violations_by_stage["context_fence_infix_equation_shadow"] == 0,
        "writer_group_overlap_audit_matches_trace_formula_ids": identity_audit[
            "writer_group_overlap_auditable"
        ],
        "failure_class_writer_cohort_rows_reconcile": (
            all(
                sum(result["formulas"] for result in cohorts.values()) == len(failure_class_rows[label])
                for label, cohorts in failure_class_writer_cohorts.items()
            )
            and sum(len(ids) for ids in failure_class_rows.values()) == len(formula_by_id)
        ),
        "fast_partition_counterfactual_preserves_top5": sum(
            int(item["stage_scores"]["context_fence_infix_equation"]["candidate_violations"])
            for item in fast_partition_cases
        ) == 0,
    }
    fast_stage_names = ("fast_selected", "fast_fence_infix_equation", "context_fence_infix_equation")
    fast_stage_summary = {}
    for name in fast_stage_names:
        all_rows = [item["stage_scores"][name] for item in fast_partition_cases]
        exact_group_rows = [
            item["stage_scores"][name] for item in fast_partition_cases
            if item["fast_group_exact"]
        ]
        aligned_group_rows = [row for row in exact_group_rows if row["token_hits_if_aligned"] is not None]
        fast_stage_summary[name] = {
            "formula_exact_all_149": sum(bool(row["formula_exact"]) for row in all_rows),
            "candidate_violations": sum(int(row["candidate_violations"]) for row in all_rows),
            "wrong_output_length_formulas": sum(int(row["length_delta"] != 0) for row in all_rows),
            "formula_exact_on_fast_group_exact_subset": sum(bool(row["formula_exact"]) for row in exact_group_rows),
            "fast_group_exact_subset_formulas": len(exact_group_rows),
            "fast_group_exact_subset_token_hits": sum(int(row["token_hits_if_aligned"]) for row in aligned_group_rows),
            "fast_group_exact_subset_target_tokens": sum(int(row["target_symbol_count"]) for row in aligned_group_rows),
        }
    group_count_deltas = Counter(item["group_count_delta"] for item in fast_partition_cases)
    fast_counterfactual_summary = {
        "scope": "actual Fast-selected groups; token reranking only; no regrouping; conditional shadow",
        "selected_fast_groups": sum(item["fast_group_count"] for item in fast_partition_cases),
        "gold_tokens": sum(item["target_token_count"] for item in fast_partition_cases),
        "fast_group_exact_formulas": sum(bool(item["fast_group_exact"]) for item in fast_partition_cases),
        "group_count_delta_histogram": dict(sorted(group_count_deltas.items())),
        "stages": fast_stage_summary,
    }
    return {
        "schema": SCHEMA,
        "scope": "current-HWR Top-5 transfer probe conditioned on oracle groups; 149-formula diagnostic only; no training/CROHME/runtime activation",
        "checkpoints": {
            "context_checkpoint": str(context_checkpoint),
            "context_checkpoint_sha256": _sha256(context_checkpoint),
            "context_bound_hwr_sha256": original_expected_sha,
            "context_bound_hwr_checkpoint": str(compatible_hwr_checkpoint),
            "current_hwr_sha256": current_hwr_sha,
            "state_transfer_audit": transfer_audit,
            "training_identity_audit": identity_audit,
        },
        "promotion_gate": {
            "eligible": False,
            "reasons": [
                "the 149-formula set is consumed development data",
                *(["training/evaluation writer groups overlap"] if identity_audit["writer_group_overlap_count"] else []),
                *(["raw-ink overlap is not auditable"] if not identity_audit["raw_ink_overlap_auditable"] else []),
            ],
        },
        "data": {"formulas": len(traces), "tokens": 579, "candidate_rank_failure_subset": len(candidate_rank_failures)},
        "metrics": metrics,
        "delta_context_fence_infix_vs_current_fence_infix": {
            "token_hits": shadow["token_hits"] - base["token_hits"],
            "formula_exact": shadow["formula_exact"] - base["formula_exact"],
            "candidate_violations": candidate_violations_by_stage["context_fence_infix"],
        },
        "delta_context_equation_shadow_vs_current_fence_infix": {
            "token_hits": equations["token_hits"] - base["token_hits"],
            "formula_exact": equations["formula_exact"] - base["formula_exact"],
            "candidate_violations": candidate_violations_by_stage["context_fence_infix_equation_shadow"],
        },
        "failure_class_context_transfer": {
            "conditioning": "all HWR/context predictions use oracle target groups; only group-exact slices model the actual fast grouping input",
            "classes": by_class,
        },
        "failure_class_writer_cohort_context_transfer": {
            "conditioning": "writer disjointness is relative only to context-scorer training groups; HWR writer exposure is not established; raw-ink overlap is unauditable",
            "classes": failure_class_writer_cohorts,
        },
        "actual_fast_partition_counterfactual": fast_counterfactual_summary,
        "context_model_audit_totals": {
            "changed_token_rows": sum(len(item["changes"]) for item in changes_vs_current_top1),
            "candidate_preservation_rate": 1.0 - candidate_violations_by_stage["context"] / sum(len(_rows(trace)) for trace in traces),
            "candidate_violations": candidate_violations_by_stage["context"],
            "role_switches": sum(int(audit.get("role_switches", 0)) for audit in context_audits),
            "exact_candidate_switches": sum(int(audit.get("exact_candidate_switches", 0)) for audit in context_audits),
            "probability_floor_reverts": sum(int(audit.get("probability_floor_reverts", 0)) for audit in context_audits),
        },
        "verification": {"checks": checks, "all_checks_pass": all(checks.values())},
        "hypothesis_result": {
            "improved_equation_guard_exact": equations["formula_exact"] > residual["summary"]["stage_formula_exact"]["after_equation"],
            "no_equation_guard_exact_regression": equations["formula_exact"] >= residual["summary"]["stage_formula_exact"]["after_equation"],
            "exact_delta_vs_equation_guard": equations["formula_exact"] - residual["summary"]["stage_formula_exact"]["after_equation"],
        },
        "candidate_rank_failure_cases_and_changes": cases,
        "actual_fast_partition_formula_cases": fast_partition_cases,
    }


def _failure_class(row: dict[str, Any]) -> str:
    group_ok, candidates_ok = bool(row["fast_group_exact"]), bool(row["top5_complete"])
    exact = bool(row["after_equation_guard_exact"])
    if group_ok and candidates_ok and exact:
        return "group_exact_top5_complete_formula_exact"
    if group_ok and candidates_ok:
        return "group_exact_top5_complete_decoder_or_rank_failure"
    if group_ok:
        return "group_exact_but_top5_missing"
    if candidates_ok:
        return "group_wrong_but_top5_complete"
    return "group_wrong_and_top5_incomplete"


def _identity_only_report(
    traces_path: Path, training_candidates_path: Path,
    evaluation_ownership_path: Path, context_transfer_path: Path | None = None,
) -> dict[str, Any]:
    traces = [
        json.loads(line)
        for line in traces_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    identity = _training_identity_audit(
        training_candidates_path, traces, evaluation_ownership_path,
    )
    checks = {
        "evaluation_ownership_ids_match_trace_ids": identity["evaluation_sample_ids_match_trace_ids"],
        "all_training_rows_have_writer_group": identity["training_rows_have_writer_group"],
        "writer_group_overlap_comparison_ran": identity["writer_group_overlap_auditable"],
    }
    cohort_replay = None
    if context_transfer_path is not None:
        cohort_replay = _writer_cohort_replay(
            context_transfer_path, training_candidates_path, evaluation_ownership_path,
        )
        checks["stored_replay_cohort_audit_passes"] = cohort_replay["verification"]["all_checks_pass"]
    reasons = ["evaluation is the consumed 149-formula development set"]
    if identity["writer_group_overlap_count"]:
        reasons.append("training and evaluation writer groups overlap")
    if not identity["raw_ink_overlap_auditable"]:
        reasons.append("raw-ink overlap cannot be audited from these candidate rows")
    return {
        "schema": "aiflow-hwr-context-lineage-audit/v1",
        "scope": "provenance-only audit; no model loading, inference, training, CROHME, or runtime changes",
        "inputs": {
            "traces": {"path": str(traces_path), "sha256": _sha256(traces_path)},
            "training_candidates": {
                "path": str(training_candidates_path),
                "sha256": _sha256(training_candidates_path),
            },
            "evaluation_ownership": {
                "path": str(evaluation_ownership_path),
                "sha256": _sha256(evaluation_ownership_path),
            },
        },
        "training_identity_audit": identity,
        "writer_cohort_replay": cohort_replay,
        "promotion_gate": {"eligible": False, "reasons": reasons},
        "verification": {"checks": checks, "all_checks_pass": all(checks.values())},
    }


def _writer_cohort_replay(
    context_transfer_path: Path, training_candidates_path: Path,
    evaluation_ownership_path: Path,
) -> dict[str, Any]:
    """Describe an existing replay by context-training writer overlap, without rerunning a model."""
    evaluation_rows = [
        json.loads(line)
        for line in evaluation_ownership_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    writer_by_sample = {
        str(row["sample_id"]): _writer_group_digest(str(row["writer_id"]))
        for row in evaluation_rows if row.get("accepted")
    }
    training_writer_groups = set()
    with gzip.open(training_candidates_path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if row.get("writer_group") is not None:
                    training_writer_groups.add(str(row["writer_group"]))

    replay = json.loads(context_transfer_path.read_text(encoding="utf-8"))
    cases = replay["actual_fast_partition_formula_cases"]
    if {str(row["sample_id"]) for row in cases} != set(writer_by_sample):
        raise AssertionError("context replay sample IDs do not match accepted ownership IDs")
    grouped: dict[str, list[dict[str, Any]]] = {"writer_disjoint": [], "training_writer_overlap": []}
    grouped_by_writer: dict[str, dict[str, list[dict[str, Any]]]] = {
        key: defaultdict(list) for key in grouped
    }
    for row in cases:
        sample_id = str(row["sample_id"])
        writer_group = writer_by_sample[sample_id]
        cohort = "training_writer_overlap" if writer_group in training_writer_groups else "writer_disjoint"
        grouped[cohort].append(row)
        grouped_by_writer[cohort][writer_group].append(row)

    baseline_stage = "fast_fence_infix_equation"
    context_stage = "context_fence_infix_equation"

    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        baseline = [row["stage_scores"][baseline_stage] for row in rows]
        context = [row["stage_scores"][context_stage] for row in rows]
        exact_group_rows = [row for row in rows if row["fast_group_exact"]]
        base_gains = sum(
            not row["stage_scores"][baseline_stage]["formula_exact"]
            and row["stage_scores"][context_stage]["formula_exact"]
            for row in rows
        )
        base_regressions = sum(
            row["stage_scores"][baseline_stage]["formula_exact"]
            and not row["stage_scores"][context_stage]["formula_exact"]
            for row in rows
        )
        exact_group_gains = sum(
            not row["stage_scores"][baseline_stage]["formula_exact"]
            and row["stage_scores"][context_stage]["formula_exact"]
            for row in exact_group_rows
        )
        exact_group_regressions = sum(
            row["stage_scores"][baseline_stage]["formula_exact"]
            and not row["stage_scores"][context_stage]["formula_exact"]
            for row in exact_group_rows
        )
        baseline_hits = sum(
            int(row["stage_scores"][baseline_stage]["token_hits_if_aligned"] or 0)
            for row in exact_group_rows
        )
        context_hits = sum(
            int(row["stage_scores"][context_stage]["token_hits_if_aligned"] or 0)
            for row in exact_group_rows
        )
        target_tokens = sum(
            int(row["stage_scores"][baseline_stage]["target_symbol_count"])
            for row in exact_group_rows
        )
        return {
            "formula_count": len(rows),
            "writer_group_count": len({writer_by_sample[str(row["sample_id"])] for row in rows}),
            "fast_group_exact_formulas": len(exact_group_rows),
            "baseline_formula_exact": sum(bool(score["formula_exact"]) for score in baseline),
            "context_formula_exact": sum(bool(score["formula_exact"]) for score in context),
            "formula_gain_transitions": base_gains,
            "formula_regressions": base_regressions,
            "exact_group_baseline_formula_exact": sum(bool(row["stage_scores"][baseline_stage]["formula_exact"]) for row in exact_group_rows),
            "exact_group_context_formula_exact": sum(bool(row["stage_scores"][context_stage]["formula_exact"]) for row in exact_group_rows),
            "exact_group_gain_transitions": exact_group_gains,
            "exact_group_regressions": exact_group_regressions,
            "exact_group_baseline_token_hits": baseline_hits,
            "exact_group_context_token_hits": context_hits,
            "exact_group_target_tokens": target_tokens,
            "candidate_violations": sum(int(score["candidate_violations"]) for score in context),
        }

    cohort_summary = {name: summarize(rows) for name, rows in grouped.items()}
    per_writer = {
        cohort: {
            writer: summarize(rows)
            for writer, rows in sorted(writer_rows.items())
        }
        for cohort, writer_rows in grouped_by_writer.items()
    }
    disjoint_writer_rows = grouped_by_writer["writer_disjoint"]
    writer_disjoint_loo_folds = []
    for excluded_writer, excluded_rows in sorted(disjoint_writer_rows.items()):
        remaining_rows = [
            row
            for writer, writer_rows in disjoint_writer_rows.items()
            if writer != excluded_writer
            for row in writer_rows
        ]
        summary = summarize(remaining_rows)
        delta_pp = 100.0 * (
            summary["context_formula_exact"] - summary["baseline_formula_exact"]
        ) / max(summary["formula_count"], 1)
        writer_disjoint_loo_folds.append({
            "excluded_writer_group": excluded_writer,
            "excluded_formula_count": len(excluded_rows),
            "remaining_formula_count": summary["formula_count"],
            "baseline_formula_exact": summary["baseline_formula_exact"],
            "context_formula_exact": summary["context_formula_exact"],
            "formula_exact_delta": summary["context_formula_exact"] - summary["baseline_formula_exact"],
            "formula_exact_delta_pp": delta_pp,
        })
    writer_disjoint_loo = {
        "method": "leave-one-context-training-disjoint-evaluation-writer-group-out; descriptive sensitivity, not a confidence interval",
        "fold_count": len(writer_disjoint_loo_folds),
        "all_formula_exact_deltas_positive": all(
            fold["formula_exact_delta"] > 0 for fold in writer_disjoint_loo_folds
        ),
        "minimum_formula_exact_delta_pp": min(
            (fold["formula_exact_delta_pp"] for fold in writer_disjoint_loo_folds),
            default=None,
        ),
        "maximum_formula_exact_delta_pp": max(
            (fold["formula_exact_delta_pp"] for fold in writer_disjoint_loo_folds),
            default=None,
        ),
        "folds": writer_disjoint_loo_folds,
    }
    verification = {
        "replay_formula_ids_match_accepted_ownership": len(cases) == len(writer_by_sample),
        "training_and_disjoint_cohorts_cover_all_replay_rows": (
            sum(len(rows) for rows in grouped.values()) == len(cases)
        ),
        "top5_candidate_contract_preserved": all(
            summary["candidate_violations"] == 0 for summary in cohort_summary.values()
        ),
        "per_writer_formula_counts_reconcile": all(
            sum(writer["formula_count"] for writer in per_writer[cohort].values())
            == cohort_summary[cohort]["formula_count"]
            for cohort in cohort_summary
        ),
        "writer_disjoint_loo_covers_each_disjoint_writer": (
            writer_disjoint_loo["fold_count"] == cohort_summary["writer_disjoint"]["writer_group_count"]
        ),
    }
    return {
        "scope": "post-hoc subgroup analysis of stored replay; consumed development data; no model selection",
        "inputs": {
            "context_transfer_report_sha256": _sha256(context_transfer_path),
            "training_writer_group_count": len(training_writer_groups),
            "evaluation_writer_group_count": len(set(writer_by_sample.values())),
        },
        "stages": {"baseline": baseline_stage, "context_shadow": context_stage},
        "cohorts": cohort_summary,
        "per_writer": per_writer,
        "writer_disjoint_loo_sensitivity": writer_disjoint_loo,
        "verification": {"checks": verification, "all_checks_pass": all(verification.values())},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-traces", type=Path, required=True)
    parser.add_argument("--residual-report", type=Path)
    parser.add_argument("--context-checkpoint", type=Path)
    parser.add_argument("--compatible-hwr-checkpoint", type=Path)
    parser.add_argument("--current-hwr-checkpoint", type=Path)
    parser.add_argument("--context-training-candidates", type=Path, required=True)
    parser.add_argument("--evaluation-ownership", type=Path, required=True)
    parser.add_argument("--context-transfer-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--identity-only", action="store_true")
    args = parser.parse_args()
    if args.identity_only:
        report = _identity_only_report(
            args.input_traces, args.context_training_candidates,
            args.evaluation_ownership, args.context_transfer_report,
        )
    else:
        required = {
            "--residual-report": args.residual_report,
            "--context-checkpoint": args.context_checkpoint,
            "--compatible-hwr-checkpoint": args.compatible_hwr_checkpoint,
            "--current-hwr-checkpoint": args.current_hwr_checkpoint,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            parser.error("missing required arguments: " + ", ".join(missing))
        report = audit(
            args.input_traces, args.residual_report,
            args.context_checkpoint, args.compatible_hwr_checkpoint,
            args.current_hwr_checkpoint, args.context_training_candidates,
            args.evaluation_ownership,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.identity_only:
        print(json.dumps({
            "training_identity_audit": report["training_identity_audit"],
            "writer_cohort_replay": report["writer_cohort_replay"],
            "promotion_gate": report["promotion_gate"],
            "verification": report["verification"],
        }, ensure_ascii=False))
    else:
        print(json.dumps({"metrics": report["metrics"], "failure_class_context_transfer": report["failure_class_context_transfer"], "failure_class_writer_cohort_context_transfer": report["failure_class_writer_cohort_context_transfer"], "training_identity_audit": report["checkpoints"]["training_identity_audit"], "promotion_gate": report["promotion_gate"], "hypothesis_result": report["hypothesis_result"], "verification": report["verification"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
