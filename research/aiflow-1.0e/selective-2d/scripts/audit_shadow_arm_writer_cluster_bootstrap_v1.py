#!/usr/bin/env python3
"""Paired writer-cluster uncertainty audit for saved HWR shadow arms."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "aiflow-hwr-shadow-arm-writer-bootstrap/v1"
FINAL_STAGE = "after_boundary_bar_as_unit_guard"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _rate(rows: list[dict[str, Any]], arm: str, *, exact: bool) -> bool:
    result = rows[arm]
    if not result["group_exact"]:
        return False
    if exact:
        shadow = result.get("semantic_guard_shadow_if_groups_exact")
        return bool(shadow and shadow["formula_exact_by_stage"].get(FINAL_STAGE, False))
    return True


def audit(
    summary_path: Path,
    *,
    baseline_arm: str,
    challenger_arm: str,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    summary = _json(summary_path)
    inputs = summary["inputs"]
    dataset_root = Path(inputs["dataset_root"])
    formulas_path = dataset_root / "data" / "formulas_valid.jsonl"
    ownership_path = dataset_root / "data" / "ownership_train.jsonl"
    checkpoint_path = Path(inputs["checkpoint"])
    ranker_path = Path(inputs["partition_ranker"])
    trace_summary_path = Path(inputs["oracle_group_decoder_ceiling_source"]["path"])

    checked_paths = {
        "summary": summary_path,
        "formulas_valid": formulas_path,
        "ownership_train": ownership_path,
        "checkpoint": checkpoint_path,
        "partition_ranker": ranker_path,
        "trace_summary": trace_summary_path,
    }
    file_hashes = {name: _sha256(path) for name, path in checked_paths.items()}
    expected_hashes = {
        "formulas_valid": inputs["formulas_valid_sha256"],
        "ownership_train": inputs["ownership_train_sha256"],
        "checkpoint": inputs["checkpoint_sha256"],
        "partition_ranker": inputs["partition_ranker_sha256"],
        "trace_summary": inputs["oracle_group_decoder_ceiling_source"]["sha256"],
    }
    for name, expected in expected_hashes.items():
        if file_hashes[name] != expected:
            raise ValueError(f"{name} hash differs from saved shadow summary")

    formulas = _jsonl(formulas_path)
    formula_by_id = {str(row["sample_id"]): row for row in formulas}
    if len(formula_by_id) != len(formulas):
        raise ValueError("duplicate sample_id in formula source")
    records = summary["records"]
    record_by_id = {str(row["sample_id"]): row for row in records}
    if len(record_by_id) != len(records):
        raise ValueError("duplicate sample_id in shadow records")
    if not set(record_by_id) <= set(formula_by_id):
        raise ValueError("shadow records are missing from formula source")

    arm_keys = set(records[0]["hwr_tournament"])
    if baseline_arm not in arm_keys or challenger_arm not in arm_keys:
        raise ValueError(f"requested arms missing; available: {sorted(arm_keys)}")

    rows: list[dict[str, Any]] = []
    for record in records:
        sample_id = str(record["sample_id"])
        writer_id = str(formula_by_id[sample_id].get("writer_id", ""))
        if not writer_id:
            raise ValueError(f"missing writer_id for {sample_id}")
        arms = record["hwr_tournament"]
        rows.append({
            "sample_id": sample_id,
            "writer_id": writer_id,
            "baseline_group_exact": _rate(arms, baseline_arm, exact=False),
            "challenger_group_exact": _rate(arms, challenger_arm, exact=False),
            "baseline_formula_exact": _rate(arms, baseline_arm, exact=True),
            "challenger_formula_exact": _rate(arms, challenger_arm, exact=True),
        })

    if len(rows) != int(summary["formulas"]):
        raise AssertionError("formula count does not reconcile with shadow summary")
    writers = sorted({row["writer_id"] for row in rows})
    if len(writers) < 2:
        raise AssertionError("writer-cluster bootstrap requires multiple writers")

    transition_names = ("both_exact", "baseline_only", "challenger_only", "both_wrong")

    def transitions(metric: str) -> dict[str, int]:
        result = {name: 0 for name in transition_names}
        for row in rows:
            baseline = row[f"baseline_{metric}"]
            challenger = row[f"challenger_{metric}"]
            name = (
                "both_exact" if baseline and challenger else
                "baseline_only" if baseline else
                "challenger_only" if challenger else "both_wrong"
            )
            result[name] += 1
        return result

    grouped: dict[str, list[dict[str, Any]]] = {writer: [] for writer in writers}
    for row in rows:
        grouped[row["writer_id"]].append(row)

    def bootstrap_delta(metric: str) -> dict[str, Any]:
        import numpy as np

        rng = np.random.default_rng(seed)
        sampled_deltas = np.empty(iterations, dtype=np.float64)
        for index in range(iterations):
            sampled_writers = rng.choice(writers, size=len(writers), replace=True)
            numerator = 0
            denominator = 0
            for writer in sampled_writers:
                cluster = grouped[str(writer)]
                denominator += len(cluster)
                numerator += sum(
                    int(row[f"challenger_{metric}"]) - int(row[f"baseline_{metric}"])
                    for row in cluster
                )
            sampled_deltas[index] = numerator / denominator
        return {
            "method": "paired writer-cluster bootstrap; writers sampled with replacement, formula-weighted rate difference",
            "iterations": iterations,
            "seed": seed,
            "delta_rate_pp_95_interval": [
                float(np.quantile(sampled_deltas, 0.025) * 100),
                float(np.quantile(sampled_deltas, 0.975) * 100),
            ],
            "delta_rate_pp_point_estimate": (
                sum(int(row[f"challenger_{metric}"]) - int(row[f"baseline_{metric}"]) for row in rows)
                / len(rows) * 100
            ),
        }

    group_base = sum(row["baseline_group_exact"] for row in rows)
    group_challenger = sum(row["challenger_group_exact"] for row in rows)
    formula_base = sum(row["baseline_formula_exact"] for row in rows)
    formula_challenger = sum(row["challenger_formula_exact"] for row in rows)
    writer_rows = []
    for writer in writers:
        cluster = grouped[writer]
        writer_rows.append({
            "writer_id": writer,
            "formulas": len(cluster),
            "baseline_group_exact": sum(row["baseline_group_exact"] for row in cluster),
            "challenger_group_exact": sum(row["challenger_group_exact"] for row in cluster),
            "baseline_formula_exact": sum(row["baseline_formula_exact"] for row in cluster),
            "challenger_formula_exact": sum(row["challenger_formula_exact"] for row in cluster),
        })

    if group_base != int(summary["hwr_tournament"]["arms"][baseline_arm]["group_exact"]):
        raise AssertionError("baseline grouping total does not reconcile")
    if group_challenger != int(summary["hwr_tournament"]["arms"][challenger_arm]["group_exact"]):
        raise AssertionError("challenger grouping total does not reconcile")
    if formula_base != int(summary["semantic_guard_shadow"]["arms"][baseline_arm]["formula_exact_by_stage"][FINAL_STAGE]):
        raise AssertionError("baseline final exact total does not reconcile")
    if formula_challenger != int(summary["semantic_guard_shadow"]["arms"][challenger_arm]["formula_exact_by_stage"][FINAL_STAGE]):
        raise AssertionError("challenger final exact total does not reconcile")

    return {
        "schema": SCHEMA,
        "scope": "paired uncertainty audit of a saved consumed-development shadow run; no rerun, fitting, threshold selection, CROHME, or product promotion",
        "inputs": {
            "summary": {"path": str(summary_path), "sha256": file_hashes["summary"]},
            "formulas_valid": {"path": str(formulas_path), "sha256": file_hashes["formulas_valid"]},
            "ownership_train": {"path": str(ownership_path), "sha256": file_hashes["ownership_train"]},
            "checkpoint": {"path": str(checkpoint_path), "sha256": file_hashes["checkpoint"]},
            "partition_ranker": {"path": str(ranker_path), "sha256": file_hashes["partition_ranker"]},
            "trace_summary": {"path": str(trace_summary_path), "sha256": file_hashes["trace_summary"]},
            "source_run_fingerprint": summary.get("reproducibility", {}).get("sha256"),
        },
        "arms": {"baseline": baseline_arm, "challenger": challenger_arm},
        "policy": {
            "strict_formula_exact": f"grouping exact AND semantic shadow {FINAL_STAGE} exact",
            "bootstrap_unit": "writer cluster",
            "selection_independent": False,
            "interpretation_limit": "writers and formulas are from a consumed-development cohort; interval is descriptive and cannot establish fresh acceptance",
        },
        "summary": {
            "formulas": len(rows),
            "writers": len(writers),
            "grouping_exact": {"baseline": group_base, "challenger": group_challenger, "transitions": transitions("group_exact")},
            "strict_formula_exact": {"baseline": formula_base, "challenger": formula_challenger, "transitions": transitions("formula_exact")},
            "writer_cluster_bootstrap": {
                "grouping_exact_delta": bootstrap_delta("group_exact"),
                "strict_formula_exact_delta": bootstrap_delta("formula_exact"),
            },
            "writer_rows": writer_rows,
        },
        "verification": {
            "inputs_hash_match_saved_summary": all(file_hashes[name] == expected for name, expected in expected_hashes.items()),
            "unique_formula_records": len(record_by_id) == len(records),
            "all_shadow_formula_ids_have_writer": len(rows) == len(record_by_id),
            "baseline_and_challenger_aggregate_counts_reconcile": True,
            "all_checks_pass": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--baseline-arm", default="fast")
    parser.add_argument("--challenger-arm", default="selective_joint_hwr_geometry_prior_group_mean_group_count_guard")
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations < 100:
        parser.error("--iterations must be at least 100")
    if args.output.exists():
        parser.error(f"refusing to overwrite output: {args.output}")
    report = audit(
        args.summary,
        baseline_arm=args.baseline_arm,
        challenger_arm=args.challenger_arm,
        iterations=args.iterations,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "summary": report["summary"], "verification": report["verification"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
