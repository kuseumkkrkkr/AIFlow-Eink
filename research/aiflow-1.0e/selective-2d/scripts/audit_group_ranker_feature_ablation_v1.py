#!/usr/bin/env python3
"""Label-blind per-formula feature ablations for the frozen Fast partition ranker."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np

from selective_2d_anytime_v1 import Selective2DAnytimeSolverV1, Selective2DConfigV1
from stroke_grouping_v1 import (
    FEATURE_NAMES, build_lattice, candidate_features, mask_candidate_features,
)


SCHEMA = "aiflow-hwr-group-ranker-feature-ablation/v1"


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _key(groups: Sequence[Sequence[int] | frozenset[int]]) -> tuple[tuple[int, ...], ...]:
    return tuple(sorted(tuple(sorted(int(index) for index in group)) for group in groups))


def _is_exact_cover(groups: Sequence[Sequence[int] | frozenset[int]], stroke_count: int) -> bool:
    flattened = [int(index) for group in groups for index in group]
    return len(flattened) == stroke_count and sorted(flattened) == list(range(stroke_count))


def _logits(model: Any, features: np.ndarray) -> np.ndarray:
    probabilities = np.clip(model.predict_proba(features)[:, 1], 1e-6, 1 - 1e-6)
    return np.log(probabilities / (1 - probabilities))


def audit(
    trace_path: Path,
    reachability_path: Path,
    budget_replay_path: Path,
    ranker_path: Path,
    provenance_path: Path | None = None,
) -> dict[str, Any]:
    trace_sha = _sha256(trace_path)
    reachability = _json(reachability_path)
    budget_replay = _json(budget_replay_path)
    if reachability.get("inputs", {}).get("traces_sha256") != trace_sha:
        raise ValueError("reachability report and trace hash differ")
    replay_trace_sha = budget_replay.get("inputs", {}).get("trace", {}).get("sha256")
    if replay_trace_sha != trace_sha:
        raise ValueError("budget replay report and trace hash differ")
    if reachability.get("inputs", {}).get("partition_ranker_sha256") != _sha256(ranker_path):
        raise ValueError("reachability report and ranker hash differ")
    formula_path = Path(reachability["inputs"]["formulas"])
    if _sha256(formula_path) != reachability["inputs"].get("formulas_sha256"):
        raise ValueError("raw formula source hash differs from reachability report")

    traces = _jsonl(trace_path)
    formulas = {str(row["sample_id"]): row for row in _jsonl(formula_path)}
    if len({str(row["sample_id"]) for row in traces}) != len(traces):
        raise ValueError("duplicate trace sample_id")
    trace_by_id = {str(row["sample_id"]): row for row in traces}
    if not set(trace_by_id) <= set(formulas):
        raise ValueError("trace rows missing from raw formula source")

    provenance = None
    accepted_ids: set[str] = set()
    exact_ink_duplicate_ids: set[str] = set()
    retained_ids = set(trace_by_id)
    if provenance_path is not None:
        provenance = _json(provenance_path)
        if not provenance.get("verification", {}).get("all_provenance_reconciliation_checks_pass"):
            raise ValueError("provenance report did not pass all reconciliation checks")
        formula_source = provenance["inputs"]["formula_source"]
        if formula_source["sha256"] != _sha256(formula_path):
            raise ValueError("provenance and grouping audit refer to different raw formula sources")
        ownership_ref = provenance["inputs"]["frozen_acceptance_ownership"]
        ownership_path = Path(ownership_ref["path"])
        if _sha256(ownership_path) != ownership_ref["sha256"]:
            raise ValueError("frozen acceptance ownership hash differs from provenance report")
        ownership_rows = _jsonl(ownership_path)
        if any(not row.get("accepted") for row in ownership_rows):
            raise ValueError("frozen acceptance ownership includes a non-accepted row")
        accepted_ids = {str(row["sample_id"]) for row in ownership_rows}
        if len(accepted_ids) != int(provenance["policy"]["frozen_acceptance_subset"]["formulas"]):
            raise ValueError("frozen acceptance ownership count differs from provenance report")
        ink_audit = provenance["policy"]["raw_ink_overlap_audit"]
        for signature in ("raw_xy", "raw_xyt", "normalized_xy"):
            exact_ink_duplicate_ids.update(
                str(pair["evaluation_id"])
                for pair in ink_audit[signature]["matching_formula_pairs"]
            )
        if accepted_ids & exact_ink_duplicate_ids:
            raise ValueError("frozen acceptance unexpectedly overlaps exact training ink")
        if not accepted_ids <= set(trace_by_id) or not exact_ink_duplicate_ids <= set(trace_by_id):
            raise ValueError("provenance exclusions are not fully represented in the trace cohort")
        retained_ids -= accepted_ids | exact_ink_duplicate_ids
        if not retained_ids:
            raise ValueError("provenance filtering left no formulas to audit")
        traces = [row for row in traces if str(row["sample_id"]) in retained_ids]

    payload = joblib.load(ranker_path)
    model = payload["grouping_model"]
    if getattr(model, "n_features_in_", len(FEATURE_NAMES)) != len(FEATURE_NAMES):
        raise ValueError("ranker feature count does not match source contract")
    predictors = getattr(model, "_predictors", None)
    if predictors is None:
        raise ValueError("ranker does not expose the expected frozen-tree diagnostics")
    split_counts = np.zeros(len(FEATURE_NAMES), dtype=np.int64)
    split_gain = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    for iteration in predictors:
        for tree in iteration:
            nodes = tree.nodes
            split_mask = nodes["is_leaf"] == 0
            split_indices = nodes["feature_idx"][split_mask].astype(int)
            np.add.at(split_counts, split_indices, 1)
            np.add.at(split_gain, split_indices, nodes["gain"][split_mask])
    ranker_feature_usage = [
        {
            "feature": name,
            "split_count": int(split_counts[index]),
            "training_split_gain_sum": float(split_gain[index]),
        }
        for index, name in enumerate(FEATURE_NAMES)
    ]
    ranker_feature_usage.sort(key=lambda row: (-row["training_split_gain_sum"], row["feature"]))
    config_values = dict(reachability["inputs"]["selective_2d_config"])
    config_values["partition_schedule"] = tuple(config_values["partition_schedule"])
    config = Selective2DConfigV1(**config_values)
    solver = Selective2DAnytimeSolverV1(config=config)

    prepared: list[dict[str, Any]] = []
    feature_blocks: list[np.ndarray] = []
    total_candidates = 0
    singleton_feature_contract_ok = True
    for trace in traces:
        sample_id = str(trace["sample_id"])
        strokes = sorted(formulas[sample_id]["strokes"], key=lambda row: int(row["order"]))
        candidates = build_lattice(
            strokes,
            temporal_window=config.temporal_window,
            spatial_neighbors=config.spatial_neighbors,
        )
        features = candidate_features(candidates, strokes)
        if features.shape != (len(candidates), len(FEATURE_NAMES)):
            raise AssertionError(f"candidate feature shape mismatch: {sample_id}")
        singleton_feature_contract_ok &= bool(np.array_equal(
            features[:, FEATURE_NAMES.index("singleton")],
            (features[:, FEATURE_NAMES.index("stroke_count")] == 1.0).astype(np.float32),
        ))
        prepared.append({
            "sample_id": sample_id,
            "strokes": strokes,
            "candidates": candidates,
            "features": features,
            "target_key": _key(trace["source"]["target_grouping"]),
            "trace_fast_key": _key(trace["grouping"].get(
                "fast_incumbent_groups", trace["grouping"]["fast_groups"],
            )),
            "writer_id": str(formulas[sample_id].get("writer_id", "unknown")),
        })
        feature_blocks.append(features)
        total_candidates += len(candidates)
    joined_features = np.concatenate(feature_blocks, axis=0)

    def score_predictions(joined: np.ndarray) -> list[list[float]]:
        logits = _logits(model, joined)
        output: list[list[float]] = []
        offset = 0
        for row in prepared:
            count = len(row["candidates"])
            output.append(logits[offset:offset + count].tolist())
            offset += count
        if offset != len(logits):
            raise AssertionError("candidate score split did not consume all rows")
        return output

    base_scores = score_predictions(joined_features)

    exact_cover_prediction_count = 0

    def predict_fast(groups_scores: Sequence[Sequence[float]], row: Mapping[str, Any]) -> tuple[tuple[int, ...], ...]:
        nonlocal exact_cover_prediction_count
        ranked, _mode = solver._rank_partitions(
            row["candidates"], groups_scores, len(row["strokes"]), top_n=2,
        )
        if not ranked:
            raise ValueError(f"no exact-cover Fast partition: {row['sample_id']}")
        key = _key(ranked[0][1])
        if not _is_exact_cover(key, len(row["strokes"])):
            raise AssertionError(f"ranker returned a non-exact cover: {row['sample_id']}")
        exact_cover_prediction_count += 1
        return key

    base_predictions = [predict_fast(scores, row) for scores, row in zip(base_scores, prepared, strict=True)]
    baseline_parity = {
        row["sample_id"]: prediction == row["trace_fast_key"]
        for row, prediction in zip(prepared, base_predictions, strict=True)
    }
    if not all(baseline_parity.values()):
        mismatches = [sample_id for sample_id, passed in baseline_parity.items() if not passed]
        raise AssertionError(f"frozen Fast replay mismatch on {len(mismatches)} rows: {mismatches[:8]}")

    baseline_exact_ids = {
        row["sample_id"] for row, prediction in zip(prepared, base_predictions, strict=True)
        if prediction == row["target_key"]
    }
    budget_skip_ids = {
        str(row["sample_id"]) for row in budget_replay.get("formula_level", [])
    }
    expected_skip_count = int(budget_replay["summary"]["region_budget_skips"])
    if len(budget_skip_ids) != expected_skip_count:
        raise AssertionError("region-budget skip cohort does not reconcile")
    budget_skip_error_indices = [
        index for index, row in enumerate(prepared)
        if row["sample_id"] in budget_skip_ids and base_predictions[index] != row["target_key"]
    ]
    expected_budget_skip_errors = {
        str(row["sample_id"])
        for row in budget_replay.get("formula_level", [])
        if row.get("target_posthoc_audit", {}).get("target_is_fast_incumbent") is False
    }
    retained_budget_skip_errors = {
        prepared[index]["sample_id"] for index in budget_skip_error_indices
    }
    if retained_budget_skip_errors != expected_budget_skip_errors & retained_ids:
        raise AssertionError("region-budget Fast grouping error cohort does not reconcile")

    ablation_rows: list[dict[str, Any]] = []
    for feature_index, feature_name in enumerate(FEATURE_NAMES):
        ablated_blocks = []
        for row in prepared:
            features = row["features"].copy()
            features[:, feature_index] = float(features[:, feature_index].mean())
            ablated_blocks.append(features)
        ablated_scores = score_predictions(np.concatenate(ablated_blocks, axis=0))
        predictions = [
            predict_fast(scores, row)
            for scores, row in zip(ablated_scores, prepared, strict=True)
        ]
        exact_ids = {
            row["sample_id"] for row, prediction in zip(prepared, predictions, strict=True)
            if prediction == row["target_key"]
        }
        recovered = sorted(exact_ids - baseline_exact_ids)
        regressed = sorted(baseline_exact_ids - exact_ids)
        changed = [
            row["sample_id"] for row, before, after in zip(prepared, base_predictions, predictions, strict=True)
            if before != after
        ]
        error_case_effects = []
        for index in budget_skip_error_indices:
            row = prepared[index]
            before = base_predictions[index]
            after = predictions[index]
            error_case_effects.append({
                "sample_id": row["sample_id"],
                "writer_id": row["writer_id"],
                "target_groups": row["target_key"],
                "baseline_fast_groups": before,
                "ablated_fast_groups": after,
                "partition_changed": before != after,
                "exact_before": before == row["target_key"],
                "exact_after": after == row["target_key"],
            })
        writer_deltas: dict[str, dict[str, int]] = {}
        for row, before, after in zip(prepared, base_predictions, predictions, strict=True):
            writer_id = row["writer_id"]
            stats = writer_deltas.setdefault(writer_id, {"baseline_exact": 0, "ablated_exact": 0})
            stats["baseline_exact"] += int(before == row["target_key"])
            stats["ablated_exact"] += int(after == row["target_key"])
        ablation_rows.append({
            "feature": feature_name,
            "formula_exact": len(exact_ids),
            "delta_exact": len(exact_ids) - len(baseline_exact_ids),
            "recovered_formula_ids": recovered,
            "regressed_formula_ids": regressed,
            "changed_partition_formula_ids": changed,
            "budget_skip_error_case_effects": error_case_effects,
            "writer_deltas": {
                writer: {
                    **stats,
                    "delta": stats["ablated_exact"] - stats["baseline_exact"],
                }
                for writer, stats in sorted(writer_deltas.items())
            },
        })

    singleton_neutralizations: list[dict[str, Any]] = []
    for mode, constant in (("constant_zero", 0.0), ("constant_one", 1.0)):
        neutralized_blocks = []
        for row in prepared:
            neutralized_blocks.append(mask_candidate_features(
                row["features"], ("singleton",), value=constant,
            ))
        neutralized_scores = score_predictions(np.concatenate(neutralized_blocks, axis=0))
        predictions = [
            predict_fast(scores, row)
            for scores, row in zip(neutralized_scores, prepared, strict=True)
        ]
        exact_ids = {
            row["sample_id"] for row, prediction in zip(prepared, predictions, strict=True)
            if prediction == row["target_key"]
        }
        writer_deltas: dict[str, dict[str, int]] = {}
        for row, prediction in zip(prepared, predictions, strict=True):
            writer_id = row["writer_id"]
            stats = writer_deltas.setdefault(writer_id, {"baseline_exact": 0, "neutralized_exact": 0})
            stats["baseline_exact"] += int(row["sample_id"] in baseline_exact_ids)
            stats["neutralized_exact"] += int(prediction == row["target_key"])
        changed_cases = []
        for row, before, after in zip(prepared, base_predictions, predictions, strict=True):
            if before == after:
                continue
            changed_cases.append({
                "sample_id": row["sample_id"],
                "writer_id": row["writer_id"],
                "target_groups": row["target_key"],
                "baseline_groups": before,
                "neutralized_groups": after,
                "exact_before": before == row["target_key"],
                "exact_after": after == row["target_key"],
            })
        singleton_neutralizations.append({
            "mode": mode,
            "constant_value": constant,
            "formula_exact": len(exact_ids),
            "delta_exact": len(exact_ids) - len(baseline_exact_ids),
            "recovered_formula_ids": sorted(exact_ids - baseline_exact_ids),
            "regressed_formula_ids": sorted(baseline_exact_ids - exact_ids),
            "changed_partition_formula_ids": [row["sample_id"] for row in changed_cases],
            "changed_cases": changed_cases,
            "writer_deltas": {
                writer: {
                    **stats,
                    "delta": stats["neutralized_exact"] - stats["baseline_exact"],
                }
                for writer, stats in sorted(writer_deltas.items())
            },
            "interpretation_limit": (
                "feature-mask shadow only; injects a value inconsistent with the binary feature contract; "
                "not a deployable setting"
            ),
        })

    ablation_rows.sort(key=lambda row: (-row["delta_exact"], row["feature"]))
    budget_skip_error_ids = [prepared[index]["sample_id"] for index in budget_skip_error_indices]
    current_skip_error_ids = set(budget_skip_error_ids)
    top_by_effect = [
        {
            "feature": row["feature"],
            "delta_exact": row["delta_exact"],
            "recovered_budget_skip_ids": sorted(set(row["recovered_formula_ids"]) & current_skip_error_ids),
            "regressed_budget_skip_ids": sorted(set(row["regressed_formula_ids"]) & current_skip_error_ids),
        }
        for row in ablation_rows[:8]
    ]
    checks = {
        "trace_hash_matches_reachability": reachability["inputs"]["traces_sha256"] == trace_sha,
        "trace_hash_matches_budget_replay": replay_trace_sha == trace_sha,
        "ranker_hash_matches_reachability": reachability["inputs"]["partition_ranker_sha256"] == _sha256(ranker_path),
        "raw_formula_hash_matches_reachability": _sha256(formula_path) == reachability["inputs"]["formulas_sha256"],
        "baseline_fast_partition_replays_all_formulas": all(baseline_parity.values()),
        "all_baseline_and_ablation_partitions_preserve_exact_cover": (
            exact_cover_prediction_count == len(prepared) * (1 + len(FEATURE_NAMES) + len(singleton_neutralizations))
        ),
        "singleton_is_exact_stroke_count_indicator": singleton_feature_contract_ok,
        "region_budget_skip_cohort_reconciles": len(budget_skip_ids) == expected_skip_count,
        "region_budget_fast_error_cohort_reconciles": (
            retained_budget_skip_errors == expected_budget_skip_errors & retained_ids
        ),
        "provenance_filtered_rows_exclude_acceptance_and_exact_training_ink": (
            provenance is None or not (retained_ids & (accepted_ids | exact_ink_duplicate_ids))
        ),
        "no_target_data_used_to_construct_ablated_features": True,
    }
    return {
        "schema": SCHEMA,
        "scope": (
            "post-hoc exploratory development slice with frozen acceptance and exact training-ink duplicates excluded; "
            "not writer-disjoint or acceptance evidence; no fitting, CROHME, threshold selection, or promotion"
            if provenance is not None else
            "frozen 149-formula consumed-development score sensitivity; no fitting, CROHME, threshold selection, or product promotion"
        ),
        "inputs": {
            "trace": {"path": str(trace_path), "sha256": trace_sha},
            "reachability": {"path": str(reachability_path), "sha256": _sha256(reachability_path)},
            "budget_replay": {"path": str(budget_replay_path), "sha256": _sha256(budget_replay_path)},
            "raw_formula_source": {"path": str(formula_path), "sha256": _sha256(formula_path)},
            "ranker": {"path": str(ranker_path), "sha256": _sha256(ranker_path)},
            "provenance_audit": (
                {"path": str(provenance_path), "sha256": _sha256(provenance_path)}
                if provenance_path is not None else None
            ),
            "audit_script_sha256": _sha256(Path(__file__)),
        },
        "data_scope": {
            "trace_formulas_available": len(trace_by_id),
            "frozen_acceptance_formulas_excluded": len(accepted_ids),
            "exact_training_ink_duplicates_excluded": len(exact_ink_duplicate_ids),
            "formulas_retained": len(prepared),
            "retained_formula_ids": sorted(retained_ids),
            "writer_independence_verified": False,
            "cohort_already_inspected_during_development": provenance is not None,
        },
        "policy": {
            "ablation": "for one feature at a time, replace each formula's candidate values by that formula's candidate mean; leave frozen ranker and all other features unchanged",
            "interpretation_limit": "sensitivity-only counterfactual; per-formula mean replacement may leave the feature manifold and is not causal evidence or a deployable setting",
            "selection_is_label_blind": True,
            "posthoc_metric": "Fast partition exact match against owned target grouping",
            "writer_ids": sorted({row["writer_id"] for row in prepared}),
        },
        "ranker_feature_usage": {
            "interpretation_limit": "frozen HistGradientBoosting split gain summarizes fitted-tree usage; it is not causal importance",
            "feature_count": len(FEATURE_NAMES),
            "split_count_total": int(split_counts.sum()),
            "features_by_training_split_gain": ranker_feature_usage,
        },
        "summary": {
            "formulas": len(prepared),
            "candidate_rows": total_candidates,
            "baseline_fast_group_exact": len(baseline_exact_ids),
            "fast_group_errors": len(prepared) - len(baseline_exact_ids),
            "budget_skip_error_ids": budget_skip_error_ids,
            "best_ablation_delta_exact": max(row["delta_exact"] for row in ablation_rows),
            "ablation_features_with_positive_exact_delta": sum(row["delta_exact"] > 0 for row in ablation_rows),
            "ablation_features_with_zero_regressions_and_positive_delta": sum(
                row["delta_exact"] > 0 and not row["regressed_formula_ids"] for row in ablation_rows
            ),
            "top_ablation_effects": top_by_effect,
        },
        "ablations": ablation_rows,
        "singleton_feature_neutralization_shadows": singleton_neutralizations,
        "verification": {"checks": checks, "all_checks_pass": all(checks.values())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--reachability", type=Path, required=True)
    parser.add_argument("--budget-replay", type=Path, required=True)
    parser.add_argument("--ranker", type=Path, required=True)
    parser.add_argument(
        "--provenance-audit", type=Path,
        help="exclude frozen acceptance and exact raw-ink training duplicates from the exploratory cohort",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite output: {args.output}")
    report = audit(
        args.trace, args.reachability, args.budget_replay, args.ranker,
        args.provenance_audit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "summary": report["summary"],
        "verification": report["verification"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
