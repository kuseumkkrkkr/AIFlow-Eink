#!/usr/bin/env python3
"""Writer-stratified posthoc audit of the Top-20 infix argmax shadow."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from formula_layout_v1 import recontextualize_formula_rows


SCHEMA = "aiflow-semantic-infix-writer-stratified-shadow/v1"
STAGE = "after_expression"
BASELINE_ARM = "5"
CHALLENGER_ARM = "top20_operator_argmax_shadow"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _exact_set(rows: list[dict[str, Any]]) -> set[str]:
    return {
        str(row["sample_id"])
        for row in rows
        if bool(row["stages"][STAGE]["formula_exact"])
    }


def _cluster_bootstrap(
    per_writer: dict[str, dict[str, Any]], *, iterations: int, seed: int,
) -> dict[str, Any]:
    writers = sorted(per_writer)
    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(iterations):
        sampled = [rng.choice(writers) for _ in writers]
        baseline_exact = sum(per_writer[writer]["baseline_exact"] for writer in sampled)
        challenger_exact = sum(per_writer[writer]["challenger_exact"] for writer in sampled)
        formula_count = sum(per_writer[writer]["formulas"] for writer in sampled)
        deltas.append(100.0 * (challenger_exact - baseline_exact) / formula_count)
    ordered = sorted(deltas)
    low = ordered[int(0.025 * (iterations - 1))]
    high = ordered[int(0.975 * (iterations - 1))]
    return {
        "method": "paired writer-cluster bootstrap; writers sampled with replacement",
        "iterations": iterations,
        "seed": seed,
        "delta_exact_rate_pp_95_interval": [low, high],
        "interpretation": "descriptive only; this consumed cohort was already used to inspect the challenger",
    }


def audit(
    *, trace_path: Path, trace_summary_path: Path, shadow_path: Path,
    dataset_root: Path, bootstrap_iterations: int, bootstrap_seed: int,
) -> dict[str, Any]:
    trace_summary = _read_json(trace_summary_path)
    shadow = _read_json(shadow_path)
    traces = _read_jsonl(trace_path)
    valid_path = dataset_root / "data" / "formulas_valid.jsonl"
    ownership_path = dataset_root / "data" / "ownership_train.jsonl"
    formula_rows = _read_jsonl(valid_path)
    ownership_rows = _read_jsonl(ownership_path)

    expected_sources = trace_summary["dataset"]["source_files"]
    valid_hash = _sha256(valid_path)
    ownership_hash = _sha256(ownership_path)
    if valid_hash != expected_sources["formulas_valid.jsonl"]["sha256"]:
        raise ValueError("formulas_valid source SHA256 does not match the trace summary")
    if ownership_hash != expected_sources["ownership_train.jsonl"]["sha256"]:
        raise ValueError("ownership_train source SHA256 does not match the trace summary")

    trace_by_id = {str(row["sample_id"]): row for row in traces}
    if len(trace_by_id) != len(traces):
        raise ValueError("duplicate sample IDs in traces")
    owner_by_id = {str(row["sample_id"]): row for row in ownership_rows}
    formula_by_id = {str(row["sample_id"]): row for row in formula_rows}
    if len(owner_by_id) != len(ownership_rows) or len(formula_by_id) != len(formula_rows):
        raise ValueError("duplicate sample IDs in source datasets")

    pipeline_rows = shadow["pipeline_formula_level_by_k"]
    baseline_rows = {str(row["sample_id"]): row for row in pipeline_rows[BASELINE_ARM]}
    challenger_rows = {str(row["sample_id"]): row for row in pipeline_rows[CHALLENGER_ARM]}
    if set(trace_by_id) != set(baseline_rows) or set(trace_by_id) != set(challenger_rows):
        raise ValueError("trace and shadow report formula IDs do not match")

    per_writer_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    formula_level_token_deltas = []
    source_writer_mismatches = []
    target_ownership_mismatches = []
    for sample_id, trace in trace_by_id.items():
        owner = owner_by_id.get(sample_id)
        source = formula_by_id.get(sample_id)
        if owner is None or source is None:
            raise ValueError(f"missing writer/source row for {sample_id}")
        writer_id = str(owner["writer_id"])
        if str(source["writer_id"]) != writer_id:
            source_writer_mismatches.append(sample_id)
        if (
            [str(token) for token in owner["labels"]]
            != [str(token) for token in trace["source"]["target_tokens"]]
            or owner["groups"] != trace["source"]["target_grouping"]
        ):
            target_ownership_mismatches.append(sample_id)
        base = baseline_rows[sample_id]["stages"][STAGE]
        challenger = challenger_rows[sample_id]["stages"][STAGE]
        symbols = trace["oracle_group_hwr"]["symbols"]
        target_tokens = [str(token) for token in trace["source"]["target_tokens"]]
        if len(symbols) != len(target_tokens):
            raise ValueError(f"symbol/target count mismatch for {sample_id}")
        layout_rows = []
        target_by_id: dict[str, str] = {}
        for index, (symbol, target) in enumerate(zip(symbols, target_tokens, strict=True)):
            prediction = symbol["prediction"]
            box = symbol["preprocessing"]["raw_bbox"]
            left, right = float(box["left"]), float(box["right"])
            top, bottom = float(box["top"]), float(box["bottom"])
            record_id = f"{sample_id}:{index}"
            layout_rows.append({
                "record_id": record_id,
                "formula_id": sample_id,
                "final_topk": [str(token) for token in prediction["top5"]],
                "final_topk_probabilities": [float(value) for value in prediction["top5_probabilities"]],
                "geometry": {
                    "left": left, "right": right, "top": top, "bottom": bottom,
                    "center_x": (left + right) / 2.0,
                    "center_y": (top + bottom) / 2.0,
                    "width_rel": max(right - left, 1e-6),
                    "height_rel": max(bottom - top, 1e-6),
                },
            })
            target_by_id[record_id] = target
        contextual_rows, _layout_audit = recontextualize_formula_rows(layout_rows)
        ordered_ids = [str(row["record_id"]) for row in contextual_rows]
        ordered_targets = [target_by_id[record_id] for record_id in ordered_ids]
        base_tokens = [str(token) for token in base["tokens"]]
        challenger_tokens = [str(token) for token in challenger["tokens"]]
        if not (len(ordered_targets) == len(base_tokens) == len(challenger_tokens)):
            raise ValueError(f"geometry-ordered token length mismatch for {sample_id}")
        token_changes = []
        for index, (target, old, new) in enumerate(zip(
            ordered_targets, base_tokens, challenger_tokens, strict=True,
        )):
            if old == new:
                continue
            token_changes.append({
                "position": index,
                "target": target,
                "baseline": old,
                "challenger": new,
                "recovered": old != target and new == target,
                "regressed": old == target and new != target,
                "wrong_to_wrong_change": old != target and new != target,
            })
        recovered_tokens = sum(change["recovered"] for change in token_changes)
        regressed_tokens = sum(change["regressed"] for change in token_changes)
        wrong_to_wrong_changes = sum(change["wrong_to_wrong_change"] for change in token_changes)
        formula_level_token_deltas.append({
            "sample_id": sample_id,
            "writer_id": writer_id,
            "baseline_token_hits": int(base["token_hits"]),
            "challenger_token_hits": int(challenger["token_hits"]),
            "recovered_tokens": recovered_tokens,
            "regressed_tokens": regressed_tokens,
            "wrong_to_wrong_changes": wrong_to_wrong_changes,
            "changes": token_changes,
        })
        per_writer_rows[writer_id].append({
            "sample_id": sample_id,
            "baseline_exact": bool(base["formula_exact"]),
            "challenger_exact": bool(challenger["formula_exact"]),
            "baseline_token_hits": int(base["token_hits"]),
            "challenger_token_hits": int(challenger["token_hits"]),
            "recovered_tokens": recovered_tokens,
            "regressed_tokens": regressed_tokens,
            "wrong_to_wrong_changes": wrong_to_wrong_changes,
        })

    writer_metrics: dict[str, dict[str, Any]] = {}
    for writer_id, rows in sorted(per_writer_rows.items()):
        recovered = sorted(
            row["sample_id"] for row in rows
            if row["challenger_exact"] and not row["baseline_exact"]
        )
        regressed = sorted(
            row["sample_id"] for row in rows
            if row["baseline_exact"] and not row["challenger_exact"]
        )
        writer_metrics[writer_id] = {
            "formulas": len(rows),
            "baseline_exact": sum(row["baseline_exact"] for row in rows),
            "challenger_exact": sum(row["challenger_exact"] for row in rows),
            "baseline_token_hits": sum(row["baseline_token_hits"] for row in rows),
            "challenger_token_hits": sum(row["challenger_token_hits"] for row in rows),
            "recovered_tokens": sum(row["recovered_tokens"] for row in rows),
            "regressed_tokens": sum(row["regressed_tokens"] for row in rows),
            "wrong_to_wrong_changes": sum(row["wrong_to_wrong_changes"] for row in rows),
            "recovered_formula_ids": recovered,
            "regressed_formula_ids": regressed,
        }

    baseline_exact_ids = _exact_set(pipeline_rows[BASELINE_ARM])
    challenger_exact_ids = _exact_set(pipeline_rows[CHALLENGER_ARM])
    recovered_ids = sorted(challenger_exact_ids - baseline_exact_ids)
    regressed_ids = sorted(baseline_exact_ids - challenger_exact_ids)
    formula_count = len(trace_by_id)
    token_count = sum(len(row["source"]["target_tokens"]) for row in traces)
    baseline_tokens = sum(
        int(row["stages"][STAGE]["token_hits"])
        for row in pipeline_rows[BASELINE_ARM]
    )
    challenger_tokens = sum(
        int(row["stages"][STAGE]["token_hits"])
        for row in pipeline_rows[CHALLENGER_ARM]
    )
    baseline_formulas = len(baseline_exact_ids)
    challenger_formulas = len(challenger_exact_ids)
    recovered_token_count = sum(row["recovered_tokens"] for row in formula_level_token_deltas)
    regressed_token_count = sum(row["regressed_tokens"] for row in formula_level_token_deltas)
    wrong_to_wrong_change_count = sum(
        row["wrong_to_wrong_changes"] for row in formula_level_token_deltas
    )

    writer_cluster_input = {
        writer_id: {
            "formulas": value["formulas"],
            "baseline_exact": value["baseline_exact"],
            "challenger_exact": value["challenger_exact"],
        }
        for writer_id, value in writer_metrics.items()
    }
    checks = {
        "source_fingerprints_match_frozen_trace": True,
        "all_149_formula_ids_join_to_writer_metadata": len(per_writer_rows) > 0
        and sum(len(rows) for rows in per_writer_rows.values()) == 149,
        "source_and_ownership_writer_ids_agree": not source_writer_mismatches,
        "ownership_labels_and_groups_match_trace_targets": not target_ownership_mismatches,
        "nine_writer_groups_reconciled": len(per_writer_rows) == 9,
        "baseline_reproduces_89_exact_491_tokens": baseline_formulas == 89 and baseline_tokens == 491,
        "challenger_reproduces_90_exact_493_tokens": challenger_formulas == 90 and challenger_tokens == 493,
        "formula_recovery_and_regression_ids_reconcile": (
            sum(len(row["recovered_formula_ids"]) for row in writer_metrics.values()) == len(recovered_ids)
            and sum(len(row["regressed_formula_ids"]) for row in writer_metrics.values()) == len(regressed_ids)
        ),
        "challenger_formula_regression_count_is_zero": len(regressed_ids) == 0,
        "token_level_deltas_reconcile_to_two_net_hits": (
            recovered_token_count - regressed_token_count == challenger_tokens - baseline_tokens == 2
        ),
    }
    return {
        "schema": SCHEMA,
        "scope": "consumed-development cohort; writer-stratified diagnostic only; not independent acceptance or promotion evidence",
        "inputs": {
            "trace": {"path": str(trace_path), "sha256": _sha256(trace_path)},
            "trace_summary": {"path": str(trace_summary_path), "sha256": _sha256(trace_summary_path)},
            "shadow_report": {"path": str(shadow_path), "sha256": _sha256(shadow_path)},
            "formulas_valid": {"path": str(valid_path), "sha256": valid_hash, "rows": len(formula_rows)},
            "ownership_train": {"path": str(ownership_path), "sha256": ownership_hash, "rows": len(ownership_rows)},
        },
        "comparison": {
            "stage": STAGE,
            "baseline_arm": BASELINE_ARM,
            "challenger_arm": CHALLENGER_ARM,
            "formulas": formula_count,
            "tokens": token_count,
            "baseline": {"formula_exact": baseline_formulas, "token_hits": baseline_tokens},
            "challenger": {"formula_exact": challenger_formulas, "token_hits": challenger_tokens},
            "delta": {
                "formula_exact": challenger_formulas - baseline_formulas,
                "formula_exact_pp": 100.0 * (challenger_formulas - baseline_formulas) / formula_count,
                "token_hits": challenger_tokens - baseline_tokens,
                "recovered_formula_ids": recovered_ids,
                "regressed_formula_ids": regressed_ids,
                "recovered_token_count": recovered_token_count,
                "regressed_token_count": regressed_token_count,
                "wrong_to_wrong_prediction_changes": wrong_to_wrong_change_count,
            },
            "writer_cluster_bootstrap": _cluster_bootstrap(
                writer_cluster_input, iterations=bootstrap_iterations, seed=bootstrap_seed,
            ),
        },
        "per_writer": writer_metrics,
        "formula_level_token_deltas": [row for row in formula_level_token_deltas if row["changes"]],
        "join_diagnostics": {
            "source_writer_mismatch_ids": source_writer_mismatches,
            "target_or_group_mismatch_ids": target_ownership_mismatches,
        },
        "verification": {"checks": checks, "all_checks_pass": all(checks.values())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--trace-summary", type=Path, required=True)
    parser.add_argument("--shadow-report", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261001)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite output: {args.output}")
    if args.bootstrap_iterations < 100:
        parser.error("bootstrap iterations must be at least 100")
    report = audit(
        trace_path=args.trace,
        trace_summary_path=args.trace_summary,
        shadow_path=args.shadow_report,
        dataset_root=args.dataset_root,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=args.bootstrap_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "comparison": report["comparison"],
        "per_writer": report["per_writer"],
        "token_delta": report["comparison"]["delta"],
        "verification": report["verification"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
