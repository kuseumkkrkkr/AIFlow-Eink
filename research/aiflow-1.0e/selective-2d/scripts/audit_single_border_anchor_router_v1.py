#!/usr/bin/env python3
"""Shadow-audit one label-blind border anchor per already-routed formula."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np

from selective_2d_anytime_v1 import (
    Selective2DConfigV1,
    build_local_hypergraph,
)
from stroke_grouping_v1 import FEATURE_NAMES, build_lattice, candidate_features


SCHEMA = "aiflow-hwr-single-border-anchor-microscope/v1"


def _read_json(path: Path) -> Any:
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


def _groups(values: Iterable[Iterable[int]]) -> list[frozenset[int]]:
    return [frozenset(int(index) for index in group) for group in values]


def _select_one_anchor(
    region: set[int],
    incumbent_groups: Sequence[frozenset[int]],
    candidates: Sequence[Mapping[str, Any]],
    scores: Sequence[float],
    score_by_group: Mapping[frozenset[int], float],
    stroke_count: int,
    config: Selective2DConfigV1,
    score_mode: str,
    closure_mode: str,
) -> dict[str, Any] | None:
    """Select by frozen candidate score and budget only; gold labels are absent."""
    ranked: list[tuple[float, tuple[int, ...], set[int], float, float, list[list[int]]]] = []
    for candidate, score in zip(candidates, scores, strict=True):
        indices = tuple(sorted(int(index) for index in candidate["source_indices"]))
        candidate_set = set(indices)
        if not candidate_set.intersection(region) or candidate_set <= region:
            continue
        expanded = set(region)
        added_fast_groups: list[frozenset[int]] = []
        for group in incumbent_groups:
            if group.intersection(candidate_set - region) and not group <= region:
                expanded.update(group)
                added_fast_groups.append(group)
        added = expanded - region
        if closure_mode == "one_raw_stroke" and len(added) != 1:
            continue
        if closure_mode == "one_fast_group" and len(added_fast_groups) != 1:
            continue
        if len(expanded) > config.max_region_strokes:
            continue
        if len(expanded) / max(stroke_count, 1) > config.max_region_fraction:
            continue
        replaced = [group for group in incumbent_groups if group.intersection(candidate_set)]
        replaced_score = sum(score_by_group[group] for group in replaced)
        selection_score = (
            float(score) - replaced_score if score_mode == "net_gain" else float(score)
        )
        ranked.append((
            selection_score, indices, expanded, float(score), replaced_score,
            [sorted(group) for group in replaced], [sorted(group) for group in added_fast_groups],
        ))
    if not ranked:
        return None
    # Stable tie-break: higher frozen ranker score, then lexicographic edge.
    ranked.sort(key=lambda item: (-item[0], item[1]))
    (
        selection_score, indices, expanded, candidate_score, replaced_score,
        replaced_groups, added_fast_groups,
    ) = ranked[0]
    return {
        "candidate_group": list(indices),
        "candidate_logit": candidate_score,
        "replaced_fast_groups": replaced_groups,
        "replaced_fast_groups_logit_sum": replaced_score,
        "selection_score_mode": score_mode,
        "selection_score": selection_score,
        "closure_mode": closure_mode,
        "added_fast_groups": added_fast_groups,
        "added_strokes": sorted(expanded - region),
        "expanded_region": sorted(expanded),
    }


def _target_reachable(
    trace: Mapping[str, Any],
    region: set[int],
    incumbent_groups: Sequence[frozenset[int]],
    strokes: Sequence[Mapping[str, Any]],
    config: Selective2DConfigV1,
) -> dict[str, Any]:
    """Post-selection label audit; never called by the anchor selector."""
    target_groups = _groups(trace["source"]["target_grouping"])
    target_outside = {group for group in target_groups if not group.intersection(region)}
    locked = [group for group in incumbent_groups if not group.intersection(region)]
    crossing = [
        sorted(group) for group in target_groups
        if group.intersection(region) and not group <= region
    ]
    local_targets = [group for group in target_groups if group <= region]
    local_rows = build_local_hypergraph(strokes, region, locked, config)
    local_groups = {
        frozenset(int(index) for index in row["source_indices"])
        for row in local_rows
    }
    found = sum(group in local_groups for group in local_targets)
    locked_match = set(locked) == target_outside
    return {
        "reachable": not crossing and locked_match and found == len(local_targets),
        "crossing_target_groups": crossing,
        "locked_groups_match_target_outside": locked_match,
        "target_local_group_count": len(local_targets),
        "target_local_groups_found": found,
        "local_candidate_count": len(local_rows),
        "local_candidates_respect_region_and_locks": all(
            set(row["source_indices"]) <= region
            or frozenset(int(index) for index in row["source_indices"]) in set(locked)
            for row in local_rows
        ),
    }


def audit(
    trace_path: Path, reachability_path: Path, ranker_path: Path,
    score_mode: str = "ranker_logit", closure_mode: str = "one_raw_stroke",
) -> dict[str, Any]:
    if score_mode not in {"ranker_logit", "net_gain"}:
        raise ValueError(f"unsupported score mode: {score_mode}")
    if closure_mode not in {"one_raw_stroke", "one_fast_group"}:
        raise ValueError(f"unsupported closure mode: {closure_mode}")
    traces = _read_jsonl(trace_path)
    reachability = _read_json(reachability_path)
    trace_sha = _sha256(trace_path)
    if reachability.get("inputs", {}).get("traces_sha256") != trace_sha:
        raise ValueError("reachability report does not match current trace hash")
    formula_path = Path(reachability["inputs"]["formulas"])
    formula_sha = _sha256(formula_path)
    if formula_sha != reachability["inputs"].get("formulas_sha256"):
        raise ValueError("raw formula source does not match reachability input hash")
    formulas = {str(row["sample_id"]): row for row in _read_jsonl(formula_path)}
    reachability_rows = {
        str(row["sample_id"]): row for row in reachability["formula_level"]
    }
    trace_by_id = {str(row["sample_id"]): row for row in traces}
    if len(trace_by_id) != len(traces) or set(trace_by_id) != set(reachability_rows):
        raise ValueError("trace and reachability formula IDs do not reconcile")
    if not set(trace_by_id) <= set(formulas):
        raise ValueError("trace formula IDs are missing from raw ink source")

    payload = joblib.load(ranker_path)
    model = payload["grouping_model"]
    if getattr(model, "n_features_in_", len(FEATURE_NAMES)) != len(FEATURE_NAMES):
        raise ValueError("partition-ranker feature contract mismatch")
    config_values = dict(reachability["inputs"]["selective_2d_config"])
    config_values["partition_schedule"] = tuple(config_values["partition_schedule"])
    config = Selective2DConfigV1(**config_values)

    selected_rows: list[dict[str, Any]] = []
    skip_reasons: Counter[str] = Counter()
    total_strokes = 0
    processed_before = 0
    processed_after = 0
    routed_fractions_before: list[float] = []
    routed_fractions_after: list[float] = []
    topology_recovered: list[str] = []
    topology_lost: list[str] = []
    invariant_errors: list[str] = []

    for trace in traces:
        sample_id = str(trace["sample_id"])
        source = formulas[sample_id]
        strokes = sorted(source["strokes"], key=lambda row: int(row["order"]))
        stroke_count = len(strokes)
        if stroke_count != int(trace["source"]["stroke_count"]):
            raise ValueError(f"raw stroke count does not match trace: {sample_id}")
        total_strokes += stroke_count
        grouping = trace["grouping"]
        incumbent = _groups(
            grouping.get("fast_incumbent_groups", grouping.get("fast_groups", []))
        )
        region = set(int(index) for index in grouping.get("local_region_strokes", []))
        searched = bool(grouping.get("partition_schedule_completed")) and int(
            grouping.get("candidate_groups_local") or 0
        ) > 0
        base_reach = reachability_rows[sample_id]

        if searched:
            processed_before += len(region)
            routed_fractions_before.append(len(region) / max(stroke_count, 1))
            processed_after += len(region)
            routed_fractions_after.append(len(region) / max(stroke_count, 1))
        else:
            skip_reasons[str(base_reach.get("outcome", "not_searched"))] += 1

        anchor = None
        after_audit = None
        if searched and region and len(region) / max(stroke_count, 1) <= config.max_region_fraction:
            candidates = build_lattice(
                strokes,
                temporal_window=config.temporal_window,
                spatial_neighbors=config.spatial_neighbors,
            )
            features = candidate_features(candidates, strokes)
            probabilities = np.clip(model.predict_proba(features)[:, 1], 1e-6, 1 - 1e-6)
            scores = np.log(probabilities / (1 - probabilities))
            score_by_group = {
                frozenset(int(index) for index in row["source_indices"]): float(score)
                for row, score in zip(candidates, scores, strict=True)
            }
            anchor = _select_one_anchor(
                region, incumbent, candidates, scores.tolist(), score_by_group,
                stroke_count, config, score_mode, closure_mode,
            )
            if anchor is None:
                skip_reasons["no_budget_eligible_crossing_candidate"] += 1
            else:
                expanded = set(int(index) for index in anchor["expanded_region"])
                if closure_mode == "one_raw_stroke" and len(expanded - region) != 1:
                    invariant_errors.append(f"{sample_id}: anchor added other than one stroke")
                if closure_mode == "one_fast_group" and len(anchor["added_fast_groups"]) != 1:
                    invariant_errors.append(f"{sample_id}: anchor added other than one Fast group")
                if len(expanded) > config.max_region_strokes:
                    invariant_errors.append(f"{sample_id}: anchor exceeded max_region_strokes")
                if len(expanded) / max(stroke_count, 1) > config.max_region_fraction:
                    invariant_errors.append(f"{sample_id}: anchor exceeded max_region_fraction")
                if not (set(anchor["candidate_group"]) & region):
                    invariant_errors.append(f"{sample_id}: selected candidate did not cross boundary")
                if not (set(anchor["candidate_group"]) - region):
                    invariant_errors.append(f"{sample_id}: selected candidate did not add boundary stroke")
                processed_after += len(expanded) - len(region)
                routed_fractions_after[-1] = len(expanded) / max(stroke_count, 1)
                after_audit = _target_reachable(
                    trace, expanded, incumbent, strokes, config,
                )
                if base_reach.get("target_partition_reachable_in_local_graph") is False and after_audit["reachable"]:
                    topology_recovered.append(sample_id)
                if base_reach.get("target_partition_reachable_in_local_graph") is True and not after_audit["reachable"]:
                    topology_lost.append(sample_id)
                if not after_audit["local_candidates_respect_region_and_locks"]:
                    invariant_errors.append(f"{sample_id}: local candidate violates region/lock boundary")

        selected_rows.append({
            "sample_id": sample_id,
            "prior_outcome": base_reach.get("outcome"),
            "prior_region_strokes": len(region),
            "formula_strokes": stroke_count,
            "anchor": anchor,
            "post_anchor_target_reachability": after_audit,
        })

    checks = {
        "trace_ids_unique_and_match_reachability": len(trace_by_id) == len(traces)
        and set(trace_by_id) == set(reachability_rows),
        "reachability_trace_sha_matches": reachability["inputs"]["traces_sha256"] == trace_sha,
        "ranker_feature_contract_matches": getattr(model, "n_features_in_", len(FEATURE_NAMES)) == len(FEATURE_NAMES),
        "anchor_added_within_selected_closure_mode": all(
            (
                len(row["anchor"]["added_strokes"]) == 1
                if closure_mode == "one_raw_stroke"
                else len(row["anchor"]["added_fast_groups"]) == 1
            )
            for row in selected_rows if row["anchor"] is not None
        ),
        "no_anchor_or_extra_work_on_unsearched_regions": all(
            row["anchor"] is None or row["prior_region_strokes"] > 0
            for row in selected_rows
        ),
        "all_anchor_and_region_invariants_pass": not invariant_errors,
        "exact_cover_reachability_checks_preserve_region_boundary": not topology_lost,
    }
    processed_before_fraction = processed_before / max(total_strokes, 1)
    processed_after_fraction = processed_after / max(total_strokes, 1)
    routed_median_before = median(routed_fractions_before) if routed_fractions_before else 0.0
    routed_median_after = median(routed_fractions_after) if routed_fractions_after else 0.0
    return {
        "schema": SCHEMA,
        "scope": "frozen consumed-development shadow simulation; no fitting, CROHME, candidate-label routing, or product promotion",
        "selection_policy": {
            "description": "for each already-searched region only, select at most one boundary-crossing full-lattice candidate and expand by the configured Fast-group closure; rank by the selected score mode, tie-break lexicographically; labels are consulted only after selection",
            "score_mode": score_mode,
            "closure_mode": closure_mode,
            "maximum_added_fast_groups_per_formula": 1 if closure_mode == "one_fast_group" else None,
            "maximum_added_raw_strokes_per_formula": 1 if closure_mode == "one_raw_stroke" else None,
            "region_fraction_cap": config.max_region_fraction,
            "region_stroke_cap": config.max_region_strokes,
        },
        "inputs": {
            "trace": {"path": str(trace_path), "sha256": trace_sha},
            "reachability": {"path": str(reachability_path), "sha256": _sha256(reachability_path)},
            "raw_formula_source": {"path": str(formula_path), "sha256": formula_sha},
            "ranker": {"path": str(ranker_path), "sha256": _sha256(ranker_path)},
            "audit_script_sha256": _sha256(Path(__file__)),
        },
        "summary": {
            "formulas": len(traces),
            "total_strokes": total_strokes,
            "already_searched_formulas": len(routed_fractions_before),
            "anchors_added": sum(row["anchor"] is not None for row in selected_rows),
            "additional_strokes_processed": processed_after - processed_before,
            "processed_strokes_before": processed_before,
            "processed_strokes_after": processed_after,
            "processed_fraction_before": processed_before_fraction,
            "processed_fraction_after": processed_after_fraction,
            "routed_region_median_before": routed_median_before,
            "routed_region_median_after": routed_median_after,
            "routed_region_median_gate_pass": routed_median_after <= 0.50,
            "total_2d_stroke_fraction_gate_pass": processed_after_fraction <= 0.35,
            "target_topology_recovered": topology_recovered,
            "target_topology_lost": topology_lost,
            "skip_reasons": dict(sorted(skip_reasons.items())),
        },
        "formula_level": selected_rows,
        "invariant_errors": invariant_errors,
        "verification": {"checks": checks, "all_checks_pass": all(checks.values())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--reachability", type=Path, required=True)
    parser.add_argument("--ranker", type=Path, required=True)
    parser.add_argument("--score-mode", choices=("ranker_logit", "net_gain"), default="ranker_logit")
    parser.add_argument("--closure-mode", choices=("one_raw_stroke", "one_fast_group"), default="one_raw_stroke")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite output: {args.output}")
    report = audit(args.trace, args.reachability, args.ranker, args.score_mode, args.closure_mode)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "summary": report["summary"],
        "verification": report["verification"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
