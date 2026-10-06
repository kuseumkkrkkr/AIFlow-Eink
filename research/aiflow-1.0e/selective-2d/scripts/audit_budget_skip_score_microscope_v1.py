#!/usr/bin/env python3
"""Posthoc score-feature microscope for Fast grouping errors skipped by area budget."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np

from audit_region_budget_skip_replay_v1 import _run_search_without_labels
from selective_2d_anytime_v1 import Selective2DConfigV1, build_local_hypergraph
from stroke_grouping_v1 import FEATURE_NAMES, candidate_features


SCHEMA = "aiflow-hwr-budget-skip-score-microscope/v1"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _key(groups: Sequence[Sequence[int] | frozenset[int]]) -> tuple[tuple[int, ...], ...]:
    return tuple(sorted(tuple(sorted(int(index) for index in group)) for group in groups))


def _logits(model: Any, matrix: np.ndarray) -> np.ndarray:
    probabilities = np.clip(model.predict_proba(matrix)[:, 1], 1e-6, 1 - 1e-6)
    return np.log(probabilities / (1 - probabilities))


def _feature_microscope(
    model: Any,
    rows: Sequence[Mapping[str, Any]],
    strokes: Sequence[Mapping[str, Any]],
    target_groups: Sequence[frozenset[int]],
    incumbent_groups: Sequence[frozenset[int]],
) -> dict[str, Any]:
    features = candidate_features(rows, strokes)
    if features.shape[1] != len(FEATURE_NAMES):
        raise ValueError("group feature count differs from current ranker contract")
    row_index = {
        frozenset(int(index) for index in row["source_indices"]): index
        for index, row in enumerate(rows)
    }
    scores = _logits(model, features)
    score_by_group = {group: float(scores[index]) for group, index in row_index.items()}
    feature_by_group = {group: features[index].astype(float) for group, index in row_index.items()}
    changed: list[dict[str, Any]] = []
    unchanged_count = 0
    additive_group_deltas = 0.0
    for target in target_groups:
        fast_overlap = [group for group in incumbent_groups if group & target]
        if target not in row_index or any(group not in row_index for group in fast_overlap):
            raise ValueError("reachable target/fast group missing from local candidates")
        delta = score_by_group[target] - sum(score_by_group[group] for group in fast_overlap)
        additive_group_deltas += delta
        if len(fast_overlap) == 1 and fast_overlap[0] == target:
            unchanged_count += 1
            continue

        comparison_groups = [target, *fast_overlap]
        comparison = np.asarray([feature_by_group[group] for group in comparison_groups], dtype=np.float32)
        observed_logits = _logits(model, comparison)
        observed_delta = float(observed_logits[0] - observed_logits[1:].sum())
        if abs(observed_delta - delta) > 1e-5:
            raise AssertionError("candidate score decomposition does not reconcile")

        counterfactuals = []
        for index, name in enumerate(FEATURE_NAMES):
            neutralized = comparison.copy()
            neutralized[:, index] = float(comparison[:, index].mean())
            neutralized_logits = _logits(model, neutralized)
            neutralized_delta = float(neutralized_logits[0] - neutralized_logits[1:].sum())
            counterfactuals.append({
                "feature": name,
                "delta_change": neutralized_delta - observed_delta,
                "counterfactual_target_minus_fast": neutralized_delta,
                "flips_nonnegative": neutralized_delta >= -1e-6,
            })
        counterfactuals.sort(key=lambda row: (-row["delta_change"], row["feature"]))
        changed.append({
            "target_group": sorted(target),
            "target_group_score": score_by_group[target],
            "target_group_features": dict(zip(FEATURE_NAMES, feature_by_group[target].tolist(), strict=True)),
            "overlapping_fast_groups": [sorted(group) for group in fast_overlap],
            "overlapping_fast_group_scores": [score_by_group[group] for group in fast_overlap],
            "overlapping_fast_group_features": [
                dict(zip(FEATURE_NAMES, feature_by_group[group].tolist(), strict=True))
                for group in fast_overlap
            ],
            "target_minus_fast_group_score": delta,
            "counterfactual_scope": "posthoc correlated-feature neutralization; diagnostic only, not causal proof or a tuning rule",
            "largest_counterfactual_effects": counterfactuals[:5],
            "counterfactual_nonnegative_flips": [
                row["feature"] for row in counterfactuals if row["flips_nonnegative"]
            ],
        })
    target_total = sum(score_by_group[group] for group in target_groups)
    incumbent_total = sum(score_by_group[group] for group in incumbent_groups)
    if abs(additive_group_deltas - (target_total - incumbent_total)) > 1e-5:
        raise AssertionError("groupwise score deltas do not add to partition delta")
    return {
        "unchanged_target_groups": unchanged_count,
        "changed_target_groups": changed,
        "target_partition_score": target_total,
        "fast_partition_score": incumbent_total,
        "target_minus_fast_partition_score": target_total - incumbent_total,
        "sum_target_group_deltas": additive_group_deltas,
        "score_decomposition_reconciled": True,
    }


def audit(
    trace_path: Path,
    reachability_path: Path,
    budget_replay_path: Path,
    ranker_path: Path,
) -> dict[str, Any]:
    trace_sha = _sha256(trace_path)
    reachability = _read_json(reachability_path)
    budget = _read_json(budget_replay_path)
    if reachability.get("inputs", {}).get("traces_sha256") != trace_sha:
        raise ValueError("reachability report and current trace hash differ")
    if budget.get("inputs", {}).get("trace", {}).get("sha256") != trace_sha:
        raise ValueError("budget replay report and current trace hash differ")
    if budget.get("inputs", {}).get("ranker", {}).get("sha256") != _sha256(ranker_path):
        raise ValueError("budget replay and ranker hashes differ")
    formula_path = Path(reachability["inputs"]["formulas"])
    if _sha256(formula_path) != reachability["inputs"].get("formulas_sha256"):
        raise ValueError("raw formula source hash differs from reachability report")

    traces = {str(row["sample_id"]): row for row in _read_jsonl(trace_path)}
    formulas = {str(row["sample_id"]): row for row in _read_jsonl(formula_path)}
    reach_rows = {str(row["sample_id"]): row for row in reachability["formula_level"]}
    if set(traces) != set(reach_rows) or not set(traces) <= set(formulas):
        raise ValueError("trace, raw ink, and reachability formula IDs do not reconcile")

    payload = joblib.load(ranker_path)
    model = payload["grouping_model"]
    if getattr(model, "n_features_in_", len(FEATURE_NAMES)) != len(FEATURE_NAMES):
        raise ValueError("ranker feature contract mismatch")
    config_values = dict(reachability["inputs"]["selective_2d_config"])
    config_values["partition_schedule"] = tuple(config_values["partition_schedule"])
    config = Selective2DConfigV1(**config_values)

    report_rows = {str(row["sample_id"]): row for row in budget["formula_level"]}
    if len(report_rows) != len(budget["formula_level"]):
        raise ValueError("duplicate formula in budget replay")
    failures = [
        row for row in budget["formula_level"]
        if reach_rows[str(row["sample_id"])].get("fast_group_exact") is False
    ]
    if len(failures) != 6:
        raise ValueError(f"expected six Fast grouping errors in budget skips, found {len(failures)}")

    formula_results: list[dict[str, Any]] = []
    for replay_row in failures:
        sample_id = str(replay_row["sample_id"])
        trace = traces[sample_id]
        grouping = trace["grouping"]
        strokes = sorted(formulas[sample_id]["strokes"], key=lambda row: int(row["order"]))
        region = set(int(index) for index in grouping.get("local_region_strokes", []))
        incumbent = [
            frozenset(int(index) for index in group)
            for group in grouping.get("fast_incumbent_groups", grouping["fast_groups"])
        ]
        target_groups = [
            frozenset(int(index) for index in group)
            for group in trace["source"]["target_grouping"]
        ]

        # Posthoc oracle-only opening determines whether the current region/locks,
        # rather than the edge generator, prevent the target grouping from entering.
        opened_region = set(region)
        changed = True
        while changed:
            changed = False
            for target in target_groups:
                if target.intersection(opened_region) and not target <= opened_region:
                    for fast_group in incumbent:
                        if fast_group.intersection(target - opened_region) and not fast_group <= opened_region:
                            opened_region.update(fast_group)
                            changed = True

        current_locked = [group for group in incumbent if not group.intersection(region)]
        opened_locked = [group for group in incumbent if not group.intersection(opened_region)]
        current_crossings = [
            sorted(group) for group in target_groups
            if group.intersection(region) and not group <= region
        ]
        opened_crossings = [
            sorted(group) for group in target_groups
            if group.intersection(opened_region) and not group <= opened_region
        ]
        opened_outside_targets = {group for group in target_groups if not group.intersection(opened_region)}
        opened_locks_match = set(opened_locked) == opened_outside_targets
        target_local = [group for group in target_groups if group <= opened_region]

        if len(opened_region) > config.max_region_strokes:
            raise ValueError(f"oracle diagnostic region exceeds 12 strokes: {sample_id}")
        local_rows = build_local_hypergraph(strokes, opened_region, opened_locked, config)
        candidate_groups = {
            frozenset(int(index) for index in row["source_indices"])
            for row in local_rows
        }
        missing_local_targets = [sorted(group) for group in target_local if group not in candidate_groups]
        opened_reachable = not opened_crossings and opened_locks_match and not missing_local_targets

        search = _run_search_without_labels(strokes, opened_region, incumbent, opened_locked, config, model)
        target_key = _key(target_groups)
        target_rank = next(
            (index for index, groups in enumerate(search["top32_partitions"], start=1) if _key(groups) == target_key),
            None,
        )
        decomposition = None
        if opened_reachable:
            decomposition = _feature_microscope(
                model, local_rows, strokes, target_groups, incumbent,
            )
        formula_results.append({
            "sample_id": sample_id,
            "current_region_strokes": sorted(region),
            "current_region_fraction": len(region) / max(len(strokes), 1),
            "current_target_crossing_groups": current_crossings,
            "current_locked_groups_match_target_outside": set(current_locked)
            == {group for group in target_groups if not group.intersection(region)},
            "oracle_opened_region_strokes": sorted(opened_region),
            "oracle_added_fast_groups": [sorted(group) for group in incumbent if group & opened_region and not group & region],
            "oracle_opened_target_reachable": opened_reachable,
            "oracle_opened_crossing_groups": opened_crossings,
            "oracle_opened_locked_groups_match_target_outside": opened_locks_match,
            "oracle_opened_missing_local_target_groups": missing_local_targets,
            "target_partition_rank_in_opened_top32": target_rank,
            "target_partition_search_winner_in_opened_region": _key(search["winner_groups"]) == target_key,
            "search_winner_unchanged_from_fast": search["winner_matches_incumbent"],
            "search_target_score_delta": (
                decomposition["target_minus_fast_partition_score"] if decomposition else None
            ),
            "group_score_decomposition": decomposition,
        })

    checks = {
        "current_trace_hash_matches_both_upstream_audits": reachability["inputs"]["traces_sha256"]
        == budget["inputs"]["trace"]["sha256"] == trace_sha,
        "current_ranker_hash_matches_budget_replay": budget["inputs"]["ranker"]["sha256"] == _sha256(ranker_path),
        "six_current_fast_group_failures_joined": len(formula_results) == 6,
        "feature_score_decompositions_reconcile": all(
            row["group_score_decomposition"] is None
            or row["group_score_decomposition"]["score_decomposition_reconciled"]
            for row in formula_results
        ),
        "posthoc_oracle_opening_does_not_change_search_scoring": all(
            row["search_winner_unchanged_from_fast"]
            for row in formula_results
        ),
    }
    return {
        "schema": SCHEMA,
        "scope": "posthoc score/candidate microscope on frozen consumed development; oracle region opening is diagnostic only; no fitting, CROHME, threshold selection, or product promotion",
        "inputs": {
            "trace": {"path": str(trace_path), "sha256": trace_sha},
            "reachability": {"path": str(reachability_path), "sha256": _sha256(reachability_path)},
            "budget_replay": {"path": str(budget_replay_path), "sha256": _sha256(budget_replay_path)},
            "raw_formula_source": {"path": str(formula_path), "sha256": _sha256(formula_path)},
            "ranker": {"path": str(ranker_path), "sha256": _sha256(ranker_path)},
            "audit_script_sha256": _sha256(Path(__file__)),
        },
        "summary": {
            "fast_group_errors_in_budget_skip_cohort": len(formula_results),
            "target_reachable_after_oracle_opening": sum(row["oracle_opened_target_reachable"] for row in formula_results),
            "target_in_top32_after_oracle_opening": sum(row["target_partition_rank_in_opened_top32"] is not None for row in formula_results),
            "target_won_after_oracle_opening": sum(row["target_partition_search_winner_in_opened_region"] for row in formula_results),
            "wrong_fast_outputs_corrected_by_current_score": sum(
                row["target_partition_search_winner_in_opened_region"] for row in formula_results
            ),
            "current_boundary_blocked_count": sum(bool(row["current_target_crossing_groups"]) for row in formula_results),
            "score_delta_for_reachable_wrong_targets": [
                {
                    "sample_id": row["sample_id"],
                    "delta": row["search_target_score_delta"],
                    "rank": row["target_partition_rank_in_opened_top32"],
                }
                for row in formula_results
                if row["search_target_score_delta"] is not None
            ],
        },
        "formula_level": formula_results,
        "verification": {"checks": checks, "all_checks_pass": all(checks.values())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--reachability", type=Path, required=True)
    parser.add_argument("--budget-replay", type=Path, required=True)
    parser.add_argument("--ranker", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite output: {args.output}")
    report = audit(args.trace, args.reachability, args.budget_replay, args.ranker)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "summary": report["summary"],
        "verification": report["verification"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
