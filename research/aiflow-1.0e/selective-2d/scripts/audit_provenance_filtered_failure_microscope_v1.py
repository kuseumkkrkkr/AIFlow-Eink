#!/usr/bin/env python3
"""Join saved HWR failure evidence after excluding acceptance and raw-ink duplicates."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "aiflow-hwr-provenance-filtered-failure-microscope/v1"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _layer_evidence(symbol: dict[str, Any]) -> dict[str, Any]:
    prediction = symbol["prediction"]
    return {
        "target": symbol["target_label"],
        "stroke_indices": symbol["stroke_indices"],
        "raw_point_count": symbol["preprocessing"]["raw_point_count"],
        "raw_points_per_stroke": symbol["preprocessing"]["raw_points_per_stroke"],
        "top1": prediction["top1"],
        "top1_probability": prediction["top1_probability"],
        "target_rank": prediction["target_rank"],
        "target_probability": prediction["target_probability"],
        "top5": prediction["top5"],
        "top5_probabilities": prediction["top5_probabilities"],
        "layer_trajectory": [
            {
                "stage": stage["stage"],
                "top1": stage["predicted_top1"],
                "target_rank": stage["target_rank"],
                "target_minus_best_other_logit": stage["target_minus_best_other_logit"],
            }
            for stage in symbol["layer_logit_lens"]
        ],
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _group_evidence(
    row: dict[str, Any],
    trace: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    replay = row.get("local_partition_search_replay") or {}
    trace_grouping = trace["grouping"]
    source = trace["source"]
    total_strokes = int(source["stroke_count"])
    local_strokes = int(row.get("local_region_stroke_count") or 0)
    replay_region_cap_violations = []
    if local_strokes > int(config["max_region_strokes"]):
        replay_region_cap_violations.append("max_region_strokes")
    if total_strokes and local_strokes / total_strokes > float(config["max_region_fraction"]):
        replay_region_cap_violations.append("max_region_fraction")
    saved_skip_reason = row.get("local_search_skip_reason")
    derived_skip_reason = None
    skip_reason_source = "saved_trace"
    if row["outcome"] == "local_search_not_run" and saved_skip_reason is None:
        if replay_region_cap_violations:
            derived_skip_reason = "region_budget_under_replay_config"
            skip_reason_source = "inferred_not_saved_in_runtime_trace"
        elif not row.get("partition_schedule_completed"):
            derived_skip_reason = "no_partition_schedule"
            skip_reason_source = "derived_from_saved_search_state"
        elif not row.get("local_candidate_count_stored"):
            derived_skip_reason = "no_local_candidates"
            skip_reason_source = "derived_from_saved_search_state"
    return {
        "outcome": row["outcome"],
        "partition_error_shape": row["partition_error_shape"],
        "fast_groups": trace_grouping.get("fast_groups", trace_grouping.get("groups")),
        "target_groups": source.get("target_grouping"),
        "target_partition_rank_in_fast_top32": trace_grouping.get("target_partition_rank_in_top32"),
        "target_partition_score": trace_grouping.get("target_partition_score"),
        "fast_partition_score": trace_grouping.get("fast_partition_score"),
        "target_group_candidate_recall": trace_grouping.get("target_group_candidate_recall"),
        "risk_reasons": row.get("risk_reasons", []),
        "risk_strokes": trace_grouping.get("risk_strokes", []),
        "local_region_seed_strokes": row.get("local_region_seed_stroke_count"),
        "local_region_strokes": row.get("local_region_stroke_count"),
        "local_region_fraction": local_strokes / total_strokes if total_strokes else None,
        "local_candidate_count": row.get("local_candidate_count_stored"),
        "target_partition_reachable": row.get("target_partition_reachable_in_local_graph"),
        "target_groups_crossing_region": row.get("target_groups_crossing_region", []),
        "saved_skip_reason": saved_skip_reason,
        "derived_skip_reason": derived_skip_reason,
        "skip_reason_source": skip_reason_source,
        "replay_region_cap_violations": (
            replay_region_cap_violations if row["outcome"] == "local_search_not_run" else []
        ),
        "instrumentation_gap": (
            "skip_reason_missing_in_source_trace"
            if row["outcome"] == "local_search_not_run" and saved_skip_reason is None
            else None
        ),
        "router_cues": row.get("router_cue_audit"),
        "search": {
            "outcome": replay.get("replay_outcome"),
            "target_rank": replay.get("target_partition_rank_in_local_top32"),
            "target_score_delta_vs_fast": replay.get(
                "target_partition_score_delta_vs_fast_incumbent"
            ),
            "target_in_geometry_shortlist": replay.get(
                "target_partition_was_in_geometry_shortlist"
            ),
        },
    }


def audit(
    failure_matrix_path: Path,
    grouping_replay_path: Path,
    provenance_path: Path,
    traces_path: Path,
    semantic_guard_path: Path,
) -> dict[str, Any]:
    matrix = _read_json(failure_matrix_path)
    grouping = _read_json(grouping_replay_path)
    provenance = _read_json(provenance_path)
    traces = _read_jsonl(traces_path)
    semantic_guard = _read_json(semantic_guard_path)

    if matrix.get("schema") != "aiflow-hwr-failure-cause-microscope/v1":
        raise ValueError("unexpected failure-matrix schema")
    if grouping.get("schema") != "aiflow-hwr-local-partition-reachability/v4":
        raise ValueError("unexpected grouping-replay schema")
    if provenance.get("verification", {}).get("all_provenance_reconciliation_checks_pass") is not True:
        raise AssertionError("provenance reconciliation has not passed")
    if semantic_guard.get("schema") != "aiflow-semantic-guard-saved-tournament-audit/v11":
        raise ValueError("unexpected semantic-guard audit schema")

    ownership_path = Path(provenance["inputs"]["frozen_acceptance_ownership"]["path"])
    ownership_rows = _read_jsonl(ownership_path)
    acceptance_ids = {str(row["sample_id"]) for row in ownership_rows}
    if len(acceptance_ids) != len(ownership_rows) or len(acceptance_ids) != 53:
        raise AssertionError("frozen acceptance ownership must contain 53 unique formula IDs")

    overlap_audit = provenance["policy"]["raw_ink_overlap_audit"]
    duplicate_ids: set[str] = set()
    for signature_name in ("raw_xy", "raw_xyt", "normalized_xy"):
        for pair in overlap_audit[signature_name]["matching_formula_pairs"]:
            duplicate_ids.add(str(pair["evaluation_id"]))
    if len(duplicate_ids) != 47:
        raise AssertionError(f"expected 47 unique raw-ink duplicate formulas, found {len(duplicate_ids)}")
    if acceptance_ids & duplicate_ids:
        raise AssertionError("acceptance IDs unexpectedly overlap exact raw-ink duplicate IDs")

    matrix_ids = {
        str(row["sample_id"])
        for row in matrix.get("residual_details", [])
    }
    grouping_rows = grouping.get("formula_level", [])
    grouping_by_id = {str(row["sample_id"]): row for row in grouping_rows}
    grouping_config = grouping.get("inputs", {}).get("selective_2d_config", {})
    if not {"max_region_strokes", "max_region_fraction"} <= set(grouping_config):
        raise AssertionError("grouping replay does not include region-budget configuration")
    if len(grouping_by_id) != len(grouping_rows) or len(grouping_rows) != 149:
        raise AssertionError("grouping replay must contain 149 unique formula rows")
    all_ids = set(grouping_by_id)
    if not acceptance_ids <= all_ids or not duplicate_ids <= all_ids:
        raise AssertionError("provenance exclusion IDs do not join to the audited 149 formulas")
    if int(matrix.get("summary", {}).get("formulas", -1)) != len(all_ids):
        raise AssertionError("failure matrix and grouping replay formula counts disagree")
    semantic_rows = semantic_guard.get("formula_level", [])
    semantic_by_id = {str(row["sample_id"]): row for row in semantic_rows}
    if len(semantic_by_id) != len(semantic_rows) or set(semantic_by_id) != all_ids:
        raise AssertionError("semantic guard audit must contain the same 149 unique formula IDs")
    trace_by_id = {str(row["sample_id"]): row for row in traces}
    expected_trace_sha = matrix["inputs"]["traces"]["sha256"]
    if len(trace_by_id) != len(traces) or set(trace_by_id) != all_ids:
        raise AssertionError("layer traces must have the same 149 unique formula IDs")
    if _sha256(traces_path) != expected_trace_sha:
        raise AssertionError("layer trace hash does not match failure matrix")

    retained_ids = all_ids - acceptance_ids - duplicate_ids
    if len(retained_ids) != 49:
        raise AssertionError(f"expected 49 provenance-filtered exploratory formulas, found {len(retained_ids)}")
    residual_by_id = {
        str(row["sample_id"]): row
        for row in matrix.get("residual_details", [])
    }
    if len(residual_by_id) != len(matrix.get("residual_details", [])):
        raise AssertionError("duplicate residual formula IDs in failure matrix")

    grouping_causes: Counter[str] = Counter()
    grouping_shapes: Counter[str] = Counter()
    decoder_causes: Counter[str] = Counter()
    guard_exact_by_stage: Counter[str] = Counter()
    guard_group_exact_count = 0
    failure_rows: list[dict[str, Any]] = []
    residual_group_errors = 0
    for sample_id in sorted(retained_ids):
        group_row = grouping_by_id[sample_id]
        group = _group_evidence(group_row, trace_by_id[sample_id], grouping_config)
        grouping_causes[group["outcome"]] += 1
        if group["outcome"] != "fast_partition_exact":
            grouping_shapes[group["partition_error_shape"]] += 1

        fast_guard = semantic_by_id[sample_id]["arms"]["fast"]
        guard_exact = bool(fast_guard["group_exact"])
        if guard_exact:
            guard_group_exact_count += 1
            for stage, exact in fast_guard["formula_exact_by_stage"].items():
                if exact:
                    guard_exact_by_stage[stage] += 1

        decoder_row = residual_by_id.get(sample_id)
        if decoder_row is None:
            decoder = {"outcome": "decoder_arm_exact_with_oracle_groups"}
        else:
            cause = str(decoder_row["cause"])
            decoder_causes[cause] += 1
            if group["outcome"] != "fast_partition_exact":
                residual_group_errors += 1
            symbols = trace_by_id[sample_id]["oracle_group_hwr"]["symbols"]
            target_tokens = [str(token) for token in decoder_row["target_tokens"]]
            if len(symbols) != len(target_tokens) or [
                str(symbol["target_label"]) for symbol in symbols
            ] != target_tokens:
                raise AssertionError(f"oracle HWR symbols do not align with decoder target: {sample_id}")
            token_microscope = []
            for mismatch in decoder_row.get("token_level_selection", []):
                if not mismatch.get("mismatch"):
                    continue
                position = int(mismatch["position"])
                if position < 1 or position > len(symbols):
                    raise AssertionError(f"decoder mismatch position out of range: {sample_id}")
                token_microscope.append({
                    **mismatch,
                    "hwr_layer_evidence": _layer_evidence(symbols[position - 1]),
                })
            missing_glyphs = []
            for glyph in decoder_row.get("missing_glyphs", []):
                matches = [
                    symbol for symbol in symbols
                    if symbol["target_label"] == glyph["target"]
                    and symbol["stroke_indices"] == glyph["stroke_indices"]
                ]
                if len(matches) != 1:
                    raise AssertionError(f"Top-5 miss does not uniquely join to layer trace: {sample_id}")
                missing_glyphs.append({
                    **glyph,
                    "hwr_layer_evidence": _layer_evidence(matches[0]),
                })
            decoder = {
                "outcome": cause,
                "candidate_complete": decoder_row.get("candidate_complete"),
                "target_beam_rank": decoder_row.get("target_beam_rank"),
                "first_lost_position": decoder_row.get("first_lost_position"),
                "first_lost_reason": decoder_row.get("first_lost_reason"),
                "target_tokens": decoder_row.get("target_tokens"),
                "selected_tokens": decoder_row.get("selected_tokens"),
                "score_gap_decomposition": decoder_row.get("score_gap_decomposition"),
                "missing_glyphs": missing_glyphs,
                "token_level_selection": token_microscope,
            }
        guard = {
            "fast_group_exact": guard_exact,
            "formula_exact_by_stage": fast_guard["formula_exact_by_stage"],
            "final_guard_stage": "after_terminal_rhs_bar_guard",
            "residual_glyph_errors_after_guard": fast_guard.get(
                "residual_glyph_errors_after_guard", []
            ),
        }
        if decoder_row is not None and not guard_exact:
            raise AssertionError(
                f"oracle-group decoder residual is not fast-group-exact in guard replay: {sample_id}"
            )
        if group["outcome"] != "fast_partition_exact" or decoder_row is not None:
            failure_rows.append({
                "sample_id": sample_id,
                "grouping": group,
                "decoder_with_oracle_groups": decoder,
                "fixed_semantic_guard_replay": guard,
            })

    expected_grouping_count = len(retained_ids)
    expected_decoder_residuals = sum(decoder_causes.values())
    checks = {
        "provenance_reconciliation_passes": True,
        "acceptance_ids_unique_and_count_53": len(acceptance_ids) == 53,
        "raw_ink_duplicate_ids_union_count_47": len(duplicate_ids) == 47,
        "acceptance_and_raw_ink_exclusions_disjoint": not (acceptance_ids & duplicate_ids),
        "source_formula_ids_reconcile_at_149": len(all_ids) == 149,
        "retained_formula_ids_reconcile_at_49": len(retained_ids) == 49,
        "retained_ids_exclude_acceptance_and_ink_duplicates": not (
            retained_ids & (acceptance_ids | duplicate_ids)
        ),
        "grouping_causes_reconcile": sum(grouping_causes.values()) == expected_grouping_count,
        "decoder_residual_causes_reconcile": expected_decoder_residuals
        == sum(1 for sample_id in retained_ids if sample_id in residual_by_id),
        "semantic_guard_ids_reconcile_at_149": set(semantic_by_id) == all_ids,
        "oracle_decoder_residuals_have_exact_fast_groups": all(
            semantic_by_id[sample_id]["arms"]["fast"]["group_exact"]
            for sample_id in retained_ids if sample_id in residual_by_id
        ),
        "layer_trace_hash_matches_failure_matrix": _sha256(traces_path) == expected_trace_sha,
        "layer_trace_ids_reconcile_at_149": set(trace_by_id) == all_ids,
        "decoder_mismatches_join_to_layer_trace": all(
            item.get("hwr_layer_evidence")
            for row in failure_rows
            for item in row["decoder_with_oracle_groups"].get("token_level_selection", [])
        ),
        "top5_misses_join_to_layer_trace": all(
            item.get("hwr_layer_evidence")
            for row in failure_rows
            for item in row["decoder_with_oracle_groups"].get("missing_glyphs", [])
        ),
        "all_retained_failure_rows_join_both_sources": all(
            row["sample_id"] in retained_ids for row in failure_rows
        ),
    }

    return {
        "schema": SCHEMA,
        "scope": (
            "posthoc forensic replay on the 49 formulas remaining after excluding the 53 frozen "
            "acceptance formulas and 47 exact raw-ink duplicates; not independent acceptance, "
            "not for tuning or promotion; CROHME not accessed"
        ),
        "interpretation_limits": {
            "decoder": (
                "decoder causes use oracle/target grouping to isolate the HWR/decoder ceiling; "
                "they are not current Fast end-to-end formula accuracy"
            ),
            "local_search_skip": (
                "the sole local_search_not_run case has no saved runtime skip_reason; its "
                "region_budget cause is inferred from region size and replay configuration, "
                "not confirmed by the original runtime trace"
            ),
            "cohort": "all retained cases were already seen during development; posthoc only",
        },
        "inputs": {
            "failure_matrix": {
                "path": str(failure_matrix_path.resolve()),
                "sha256": _sha256(failure_matrix_path),
            },
            "grouping_replay": {
                "path": str(grouping_replay_path.resolve()),
                "sha256": _sha256(grouping_replay_path),
            },
            "provenance_audit": {
                "path": str(provenance_path.resolve()),
                "sha256": _sha256(provenance_path),
            },
            "layer_traces": {
                "path": str(traces_path.resolve()),
                "sha256": _sha256(traces_path),
            },
            "semantic_guard_audit": {
                "path": str(semantic_guard_path.resolve()),
                "sha256": _sha256(semantic_guard_path),
            },
            "acceptance_ownership": {
                "path": str(ownership_path.resolve()),
                "sha256": _sha256(ownership_path),
            },
        },
        "cohort": {
            "source_formulas": len(all_ids),
            "excluded_frozen_acceptance": len(acceptance_ids),
            "excluded_exact_raw_ink_duplicates": len(duplicate_ids),
            "retained_posthoc_formulas": len(retained_ids),
            "retained_ids": sorted(retained_ids),
        },
        "grouping_failure_microscope": {
            "outcome_counts": dict(sorted(grouping_causes.items())),
            "error_shape_counts": dict(sorted(grouping_shapes.items())),
            "fast_partition_exact": grouping_causes.get("fast_partition_exact", 0),
        },
        "decoder_failure_microscope_with_oracle_groups": {
            "outcome_counts": dict(sorted(decoder_causes.items())),
            "oracle_group_decoder_exact": len(retained_ids) - expected_decoder_residuals,
            "residuals_also_having_fast_group_error": residual_group_errors,
            "interpretation": (
                "decoder causes are measured with target/oracle grouping, so they isolate the "
                "HWR/decoder ceiling and are not an end-to-end Fast prediction"
            ),
        },
        "fixed_semantic_guard_replay": {
            "fast_group_exact_formulas": guard_group_exact_count,
            "strict_exact_counts_by_stage": dict(sorted(guard_exact_by_stage.items())),
            "interpretation": (
                "hash-linked replay of the already-frozen shadow guard; no threshold or policy "
                "was selected on this cohort"
            ),
        },
        "failure_rows": failure_rows,
        "verification": {"checks": checks, "all_checks_pass": all(checks.values())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failure-matrix", type=Path, required=True)
    parser.add_argument("--grouping-replay", type=Path, required=True)
    parser.add_argument("--provenance-audit", type=Path, required=True)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--semantic-guard-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite output: {args.output}")

    report = audit(
        args.failure_matrix, args.grouping_replay, args.provenance_audit,
        args.traces, args.semantic_guard_audit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "cohort": report["cohort"],
        "grouping": report["grouping_failure_microscope"],
        "decoder": report["decoder_failure_microscope_with_oracle_groups"],
        "verification": report["verification"],
        "output": str(args.output.resolve()),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
