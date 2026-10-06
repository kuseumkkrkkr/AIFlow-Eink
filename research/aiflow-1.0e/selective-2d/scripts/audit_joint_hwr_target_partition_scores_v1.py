#!/usr/bin/env python3
"""Replay gold-partition scores against the saved Fast joint-HWR incumbent."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from selective_2d_anytime_v1 import (
    Selective2DAnytimeSolverV1,
    Selective2DConfigV1,
    build_local_hypergraph,
)
from selective_decoder_v1 import decode_selective_partition
from stroke_grouping_v1 import candidate_features


ARM = "selective_joint_hwr_geometry_prior_group_count_guard"
SCORE_TOLERANCE = 1e-5


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _key(groups: list[list[int]]) -> tuple[tuple[int, ...], ...]:
    return tuple(sorted(tuple(sorted(int(index) for index in group)) for group in groups))


def _symbol(
    group: list[int], top5: list[str], probabilities: list[float], row: dict[str, Any],
) -> dict[str, Any]:
    return {
        "stroke_indices": sorted(int(index) for index in group),
        "hwr_topk": [str(token) for token in top5],
        "hwr_topk_probabilities": [float(value) for value in probabilities],
        "geometry": dict(row["box"]),
    }


def audit(
    summary_path: Path,
    root_cause_path: Path,
    trace_path: Path,
    trace_summary_path: Path,
) -> dict[str, Any]:
    summary = _read(summary_path)
    root_cause = _read(root_cause_path)
    trace_summary = _read(trace_summary_path)
    config_data = summary["run_config"]["selective_2d_config"]
    config = Selective2DConfigV1(**{
        **config_data,
        "partition_schedule": tuple(config_data["partition_schedule"]),
        "reject_local_group_count_increase": True,
    })

    inputs = summary["inputs"]
    dataset_root = Path(inputs["dataset_root"])
    formulas_path = dataset_root / "data" / "formulas_valid.jsonl"
    ranker_path = Path(inputs["partition_ranker"])
    if _sha256(summary_path) != root_cause["inputs"]["guarded_summary"]["sha256"]:
        raise AssertionError("root-cause audit does not match guarded summary")
    if _sha256(formulas_path) != inputs["formulas_valid_sha256"]:
        raise AssertionError("formula source hash changed")
    if _sha256(ranker_path) != inputs["partition_ranker_sha256"]:
        raise AssertionError("partition ranker hash changed")
    trace_fingerprint = trace_summary["reproducibility"]["inputs"]
    if trace_fingerprint["dataset_files"]["data/formulas_valid.jsonl"] != inputs["formulas_valid_sha256"]:
        raise AssertionError("oracle trace uses a different formula source")
    if trace_fingerprint["artifacts"]["hwr_checkpoint_sha256"] != inputs["checkpoint_sha256"]:
        raise AssertionError("oracle trace uses a different HWR checkpoint")
    if trace_fingerprint["artifacts"]["partition_ranker_sha256"] != inputs["partition_ranker_sha256"]:
        raise AssertionError("oracle trace uses a different partition ranker")

    formulas = {str(row["sample_id"]): row for row in _read_jsonl(formulas_path)}
    traces = {str(row["sample_id"]): row for row in _read_jsonl(trace_path)}
    records = {str(row["sample_id"]): row for row in summary["records"]}
    if len(traces) != 149 or len(records) != 149:
        raise AssertionError("saved traces and guarded summary must cover 149 formulas")

    reachable_ids = [
        row["sample_id"]
        for row in root_cause["grouping_failure_causes"]["formulas"]
        if row["cause"] == "gold_partition_reachable_but_not_promoted"
    ]
    ranker_payload = joblib.load(ranker_path)
    ranker = ranker_payload["grouping_model"]

    def score_rows(rows: list[dict[str, Any]], strokes: list[dict[str, Any]]) -> list[float]:
        probabilities = np.clip(ranker.predict_proba(candidate_features(rows, strokes))[:, 1], 1e-6, 1 - 1e-6)
        return np.log(probabilities / (1.0 - probabilities)).tolist()

    solver = Selective2DAnytimeSolverV1(config=config, group_score_batch=score_rows)
    details = []
    checks = {
        "gold_partition_candidates_present": True,
        "fast_decoder_replay_matches_saved_tokens_latex_score": True,
        "fast_geometry_score_matches_saved_breakdown": True,
        "oracle_group_top5_count_matches_gold_groups": True,
        "candidate_row_counts_match_saved_runtime": True,
        "target_search_rank_reconciled": True,
    }

    for sample_id in reachable_ids:
        if sample_id not in formulas or sample_id not in traces or sample_id not in records:
            raise AssertionError(f"missing source row for {sample_id}")
        metrics = records[sample_id]["hwr_tournament"][ARM]
        if metrics.get("route") != "fast":
            raise AssertionError(f"expected retained Fast incumbent for {sample_id}")
        strokes = sorted(formulas[sample_id]["strokes"], key=lambda row: int(row["order"]))
        local = build_local_hypergraph(
            strokes,
            metrics["local_region_strokes"],
            metrics["locked_fast_groups"],
            config,
        )
        if len(local) != int(metrics["candidate_groups_local"]):
            checks["candidate_row_counts_match_saved_runtime"] = False
            raise AssertionError(f"local candidate row count mismatch: {sample_id}")
        row_by_group = {
            tuple(sorted(int(index) for index in row["source_indices"])): row
            for row in local
        }
        group_scores = score_rows(local, strokes)
        score_by_group = {
            tuple(sorted(int(index) for index in row["source_indices"])): float(score)
            for row, score in zip(local, group_scores, strict=True)
        }
        fast_groups = [
            [int(index) for index in group] for group in metrics["groups"]
        ]
        target_groups = [
            [int(index) for index in group] for group in metrics["target_groups"]
        ]
        target_keys = _key(target_groups)
        fast_keys = _key(fast_groups)
        missing_target = [group for group in target_keys if group not in row_by_group]
        if missing_target:
            checks["gold_partition_candidates_present"] = False
            raise AssertionError(f"reachable target group missing from current graph: {sample_id}")
        if any(group not in row_by_group for group in fast_keys):
            raise AssertionError(f"Fast group missing from local/locked graph: {sample_id}")

        fast_symbols = []
        saved_symbols = {
            tuple(sorted(int(index) for index in item["stroke_indices"])): item
            for item in metrics["selected_symbols"]
        }
        for group in fast_groups:
            key = tuple(sorted(group))
            item = saved_symbols[key]
            fast_symbols.append(_symbol(
                group, item["hwr_topk"], item["hwr_topk_probabilities"], row_by_group[key],
            ))
        fast_decode = decode_selective_partition(
            sample_id, fast_groups, fast_symbols, stroke_count=len(strokes),
        )
        if (
            not fast_decode.get("accepted")
            or fast_decode.get("tokens") != metrics["decoder_tokens"]
            or fast_decode.get("latex") != metrics["decoder_latex"]
            or abs(float(fast_decode["joint_token_relation_score"]) - float(metrics["decoder_joint_score"])) > SCORE_TOLERANCE
        ):
            checks["fast_decoder_replay_matches_saved_tokens_latex_score"] = False
            raise AssertionError(f"Fast decoder replay mismatch: {sample_id}")

        trace_symbols = traces[sample_id]["oracle_group_hwr"]["symbols"]
        oracle_by_group = {
            tuple(sorted(int(index) for index in item["stroke_indices"])): item
            for item in trace_symbols
        }
        if set(oracle_by_group) != set(target_keys):
            checks["oracle_group_top5_count_matches_gold_groups"] = False
            raise AssertionError(f"oracle group rows do not match gold partition: {sample_id}")
        target_symbols = []
        for group in target_groups:
            key = tuple(sorted(group))
            prediction = oracle_by_group[key]["prediction"]
            top5 = [str(token) for token in prediction["top5"]]
            probabilities = [float(value) for value in prediction["top5_probabilities"]]
            if len(top5) != 5 or len(probabilities) != 5:
                raise AssertionError(f"invalid saved Top-5 prediction: {sample_id} {key}")
            target_symbols.append(_symbol(group, top5, probabilities, row_by_group[key]))
        target_decode = decode_selective_partition(
            sample_id, target_groups, target_symbols, stroke_count=len(strokes),
        )

        fast_geometry_score = sum(score_by_group[group] for group in fast_keys) / max(len(strokes), 1)
        target_geometry_score = sum(score_by_group[group] for group in target_keys) / max(len(strokes), 1)
        saved_fast = metrics["joint_incumbent_score_breakdown"]
        if abs(fast_geometry_score - float(saved_fast["geometry_prior_score"])) > SCORE_TOLERANCE:
            checks["fast_geometry_score_matches_saved_breakdown"] = False
            raise AssertionError(f"Fast geometry score mismatch: {sample_id}")
        if abs(float(fast_decode["joint_token_relation_score"]) - float(saved_fast["hwr_score"])) > SCORE_TOLERANCE:
            raise AssertionError(f"Fast HWR score mismatch: {sample_id}")

        fast_total = float(saved_fast["total_score"])
        target_hwr_score = (
            float(target_decode["joint_token_relation_score"])
            if target_decode.get("accepted") else None
        )
        target_total = (
            target_hwr_score + float(summary["run_config"]["joint_geometry_prior_weight"]) * target_geometry_score
            if target_hwr_score is not None else None
        )
        score_delta = target_total - fast_total if target_total is not None else None
        ranked, search_mode = solver._rank_partitions(
            local, group_scores, len(strokes), top_n=32,
        )
        target_rank = next(
            (rank for rank, (_score, groups) in enumerate(ranked, start=1)
             if _key([list(group) for group in groups]) == target_keys),
            None,
        )
        expected_rank = next(
            row["search"]["target_rank"]
            for row in root_cause["grouping_failure_causes"]["formulas"]
            if row["sample_id"] == sample_id
        )
        if target_rank != expected_rank:
            checks["target_search_rank_reconciled"] = False
            raise AssertionError(
                f"geometry-search rank mismatch for {sample_id}: current={target_rank} prior={expected_rank}"
            )

        target_tokens = [str(token) for token in metrics["target_tokens"]]
        details.append({
            "sample_id": sample_id,
            "fast_group_count": len(fast_groups),
            "target_group_count": len(target_groups),
            "guard_would_reject_target_for_group_increase": len(target_groups) > len(fast_groups),
            "target_partition_in_local_top32": target_rank is not None,
            "target_partition_rank": target_rank,
            "geometry_search_mode": search_mode,
            "fast": {
                "hwr_score": float(fast_decode["joint_token_relation_score"]),
                "geometry_prior_score": fast_geometry_score,
                "total_score": fast_total,
            },
            "gold_partition_counterfactual": {
                "decoder_accepted": bool(target_decode.get("accepted")),
                "decoder_rejection_reason": target_decode.get("reason") if not target_decode.get("accepted") else None,
                "predicted_tokens": target_decode.get("tokens"),
                "target_tokens": target_tokens,
                "token_sequence_exact": target_decode.get("tokens") == target_tokens,
                "latex": target_decode.get("latex"),
                "hwr_score": target_hwr_score,
                "geometry_prior_score": target_geometry_score,
                "total_score": target_total,
                "score_delta_vs_fast": score_delta,
                "beats_fast_by_acceptance_margin": (
                    score_delta is not None
                    and score_delta >= float(config.acceptance_margin)
                ),
                "misses_hwr_top5_count": sum(
                    token not in [str(value) for value in oracle_by_group[tuple(sorted(group))]["prediction"]["top5"]]
                    for group, token in zip(target_groups, target_tokens, strict=True)
                ),
            },
        })

    if not all(checks.values()):
        raise AssertionError(f"joint target replay checks failed: {checks}")
    score_deltas = [
        row["gold_partition_counterfactual"]["score_delta_vs_fast"]
        for row in details
        if row["gold_partition_counterfactual"]["score_delta_vs_fast"] is not None
    ]
    return {
        "schema": "aiflow-hwr-joint-target-partition-score-audit/v1",
        "scope": "frozen consumed-development replay; Top-5 logits and strokes reused; no HWR inference, training, CROHME, or promotion",
        "inputs": {
            "guarded_summary": {"path": str(summary_path), "sha256": _sha256(summary_path)},
            "root_cause_audit": {"path": str(root_cause_path), "sha256": _sha256(root_cause_path)},
            "layer_trace": {"path": str(trace_path), "sha256": _sha256(trace_path)},
            "layer_trace_summary": {"path": str(trace_summary_path), "sha256": _sha256(trace_summary_path)},
            "dataset_formulas_sha256": _sha256(formulas_path),
            "partition_ranker_sha256": _sha256(ranker_path),
            "hwr_checkpoint_sha256": inputs["checkpoint_sha256"],
        },
        "formula_count": len(details),
        "gold_decoder_acceptance_count": sum(
            int(row["gold_partition_counterfactual"]["decoder_accepted"])
            for row in details
        ),
        "gold_total_score_beats_fast_count": sum(
            int(row["gold_partition_counterfactual"]["beats_fast_by_acceptance_margin"])
            for row in details
        ),
        "gold_partition_in_local_top32_count": sum(
            int(row["target_partition_in_local_top32"]) for row in details
        ),
        "gold_partition_guard_rejected_count": sum(
            int(row["guard_would_reject_target_for_group_increase"]) for row in details
        ),
        "score_delta_summary": {
            "scored_count": len(score_deltas),
            "mean_gold_minus_fast": sum(score_deltas) / len(score_deltas) if score_deltas else None,
            "positive_count": sum(value > 0.0 for value in score_deltas),
            "acceptance_margin": float(config.acceptance_margin),
        },
        "formulas": details,
        "validation": {
            "current_dataset_and_model_hashes_match_saved_layer_trace": True,
            "target_groups_present_in_current_local_candidate_graph": checks[
                "gold_partition_candidates_present"
            ],
            "fast_decoder_output_and_score_reconciled": checks[
                "fast_decoder_replay_matches_saved_tokens_latex_score"
            ],
            "fast_geometry_score_reconciled": checks[
                "fast_geometry_score_matches_saved_breakdown"
            ],
            "top5_targets_match_gold_groups": checks[
                "oracle_group_top5_count_matches_gold_groups"
            ],
            "current_candidate_counts_reconciled": checks[
                "candidate_row_counts_match_saved_runtime"
            ],
            "local_top32_target_ranks_reconciled": checks[
                "target_search_rank_reconciled"
            ],
            "crohme_training_or_tuning": False,
            "product_default_enabled": False,
            "promotion_eligible": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--root-cause-audit", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--trace-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite output: {args.output}")
    result = audit(
        args.summary.resolve(), args.root_cause_audit.resolve(),
        args.trace.resolve(), args.trace_summary.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "formula_count": result["formula_count"],
        "gold_decoder_acceptance_count": result["gold_decoder_acceptance_count"],
        "gold_total_score_beats_fast_count": result["gold_total_score_beats_fast_count"],
        "gold_partition_in_local_top32_count": result["gold_partition_in_local_top32_count"],
        "score_delta_summary": result["score_delta_summary"],
        "validation": result["validation"],
        "output": str(args.output.resolve()),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
