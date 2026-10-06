#!/usr/bin/env python3
"""Counterfactual audit of rejecting local partitions that add groups."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any


SCHEMA = "aiflow-hwr-local-group-count-guard-audit/v1"
ARM = "selective_joint_hwr_geometry_prior"
ITERATIONS = 10_000
SEED = 20261001


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_groups(groups: list[list[int]]) -> tuple[tuple[int, ...], ...]:
    return tuple(sorted(tuple(sorted(int(index) for index in group)) for group in groups))


def _covers_same_strokes(groups: list[list[int]], reference: list[list[int]]) -> bool:
    flattened = [int(index) for group in groups for index in group]
    expected = [int(index) for group in reference for index in group]
    return len(flattened) == len(set(flattened)) and sorted(flattened) == sorted(expected)


def _metrics(rows: list[dict[str, Any]], selection_key: str) -> dict[str, Any]:
    return {
        "formulas": len(rows),
        "group_exact": sum(int(row[selection_key]["group_exact"]) for row in rows),
        "token_sequence_exact": sum(int(row[selection_key]["token_exact"]) for row in rows),
        "group_and_token_exact": sum(int(row[selection_key]["group_token_exact"]) for row in rows),
        "local_routes": sum(int(row[selection_key]["route"] == "local_2d") for row in rows),
    }


def _transition(rows: list[dict[str, Any]], before: str, after: str, metric: str) -> dict[str, Any]:
    recovered = [row["sample_id"] for row in rows if not row[before][metric] and row[after][metric]]
    regressed = [row["sample_id"] for row in rows if row[before][metric] and not row[after][metric]]
    return {"recovered": recovered, "regressed": regressed, "delta": len(recovered) - len(regressed)}


def _writer_bootstrap(
    rows: list[dict[str, Any]], writer_by_id: dict[str, str], metric: str,
) -> dict[str, Any]:
    by_writer: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_writer.setdefault(writer_by_id[row["sample_id"]], []).append(row)
    writers = sorted(by_writer)
    rng = random.Random(SEED)
    deltas: list[float] = []
    for _ in range(ITERATIONS):
        sampled = [rng.choice(writers) for _ in writers]
        denominator = sum(len(by_writer[writer]) for writer in sampled)
        before = sum(int(row["fast"][metric]) for writer in sampled for row in by_writer[writer])
        after = sum(int(row["gated"][metric]) for writer in sampled for row in by_writer[writer])
        deltas.append(100.0 * (after - before) / denominator)
    deltas.sort()
    return {
        "method": "paired writer-cluster bootstrap; writers sampled with replacement",
        "iterations": ITERATIONS,
        "seed": SEED,
        "delta_rate_pp_95_interval": [
            deltas[int(0.025 * (ITERATIONS - 1))],
            deltas[int(0.975 * (ITERATIONS - 1))],
        ],
        "interpretation": "consumed-development counterfactual; not acceptance evidence",
    }


def audit(report_path: Path, ownership_path: Path) -> dict[str, Any]:
    report = _read_json(report_path)
    dataset_root = Path(report.get("inputs", {}).get("dataset_root", ""))
    formulas_path = dataset_root / "data" / "formulas_valid.jsonl"
    if not formulas_path.is_file():
        raise FileNotFoundError(f"frozen formulas file not found: {formulas_path}")
    formulas = [
        json.loads(row) for row in formulas_path.read_text(encoding="utf-8").splitlines()
        if row.strip()
    ]
    ownership_rows = [
        row for row in Path(ownership_path).read_text(encoding="utf-8").splitlines()
        if row.strip()
    ]
    ownership = [json.loads(row) for row in ownership_rows]
    if report.get("schema") != "aiflow-selective-2d-research-loop/v1":
        raise ValueError("unexpected selective-2D research report schema")
    if not report.get("hwr_tournament"):
        raise ValueError("source report has no HWR tournament")
    if float(report["hwr_tournament"].get("joint_geometry_prior_weight", -1)) != 1.0:
        raise ValueError("counterfactual requires the frozen weight-1.0 prior arm")
    if report.get("product_default_enabled") or report.get("promotion_eligible"):
        raise AssertionError("source report unexpectedly indicates product promotion")

    writer_by_id = {
        str(row["sample_id"]): str(row["writer_id"])
        for row in ownership if row.get("accepted")
    }
    records = report["records"]
    ids = [str(row["sample_id"]) for row in records]
    formula_ids = {str(row["sample_id"]) for row in formulas}
    if len(ids) != 149 or len(set(ids)) != len(ids) or set(ids) != set(writer_by_id):
        raise AssertionError("source report and accepted ownership IDs do not match 149 unique formulas")

    formula_rows: list[dict[str, Any]] = []
    rejected_increases: list[dict[str, Any]] = []
    checks = {
        "all_fast_and_challenger_partitions_cover_same_strokes": True,
        "all_fast_and_challenger_decoders_accepted": True,
        "all_local_group_increases_are_rejected": True,
        "source_formula_hash_matches_frozen_cohort": (
            report["inputs"]["formulas_valid_sha256"] == _sha256(formulas_path)
        ),
        "all_report_ids_exist_in_hashed_formula_source": set(ids).issubset(formula_ids),
        "ownership_hash_matches_frozen_cohort": (
            report["inputs"]["ownership_train_sha256"] == _sha256(ownership_path)
        ),
    }
    for record in records:
        sample_id = str(record["sample_id"])
        arms = record["hwr_tournament"]
        fast, challenger = arms["fast"], arms[ARM]
        target_groups = challenger["target_groups"]
        target_tokens = [str(token) for token in challenger["target_tokens"]]
        if not _covers_same_strokes(fast["groups"], target_groups):
            checks["all_fast_and_challenger_partitions_cover_same_strokes"] = False
        if not _covers_same_strokes(challenger["groups"], target_groups):
            checks["all_fast_and_challenger_partitions_cover_same_strokes"] = False
        checks["all_fast_and_challenger_decoders_accepted"] &= bool(
            fast["decoder_accepted"] and challenger["decoder_accepted"]
        )
        group_exact = _canonical_groups(challenger["groups"]) == _canonical_groups(target_groups)
        fast_token_exact = [str(token) for token in fast["decoder_tokens"]] == target_tokens
        challenger_token_exact = [str(token) for token in challenger["decoder_tokens"]] == target_tokens
        fast_group_token_exact = bool(fast["group_exact"] and fast_token_exact)
        challenger_group_token_exact = bool(group_exact and challenger_token_exact)
        if bool(challenger["group_exact"]) != group_exact:
            raise AssertionError(f"stored group-exact flag disagrees with groups: {sample_id}")

        increase = (
            challenger["route"] == "local_2d"
            and len(challenger["groups"]) > len(fast["groups"])
        )
        selected = fast if increase else challenger
        if increase:
            rejected_increases.append({
                "sample_id": sample_id,
                "fast_group_count": len(fast["groups"]),
                "challenger_group_count": len(challenger["groups"]),
                "challenger_group_exact": group_exact,
                "challenger_token_exact": challenger_token_exact,
                "challenger_group_and_token_exact": challenger_group_token_exact,
                "score_delta": challenger.get("winner_score_delta"),
            })
        formula_rows.append({
            "sample_id": sample_id,
            "fast": {
                "group_exact": bool(fast["group_exact"]),
                "token_exact": fast_token_exact,
                "group_token_exact": fast_group_token_exact,
                "route": fast["route"],
            },
            "challenger": {
                "group_exact": group_exact,
                "token_exact": challenger_token_exact,
                "group_token_exact": challenger_group_token_exact,
                "route": challenger["route"],
            },
            "gated": {
                "group_exact": bool(selected["group_exact"]),
                "token_exact": [str(token) for token in selected["decoder_tokens"]] == target_tokens,
                "group_token_exact": bool(selected["group_exact"])
                and [str(token) for token in selected["decoder_tokens"]] == target_tokens,
                "route": selected["route"],
                "group_count": len(selected["groups"]),
                "fast_group_count": len(fast["groups"]),
            },
        })

    gated_ids = [row["sample_id"] for row in formula_rows if row["gated"]["route"] == "local_2d"]
    checks["all_local_group_increases_are_rejected"] = all(
        row["gated"]["group_count"] <= row["gated"]["fast_group_count"]
        for row in formula_rows if row["gated"]["route"] == "local_2d"
    )
    checks["formula_count_is_149"] = len(formula_rows) == 149
    checks["accepted_writer_count_is_9"] = len(set(writer_by_id.values())) == 9
    checks["gated_group_exact_never_regresses_vs_fast"] = not _transition(
        formula_rows, "fast", "gated", "group_exact",
    )["regressed"]
    checks["gated_group_and_token_exact_never_regresses_vs_fast"] = not _transition(
        formula_rows, "fast", "gated", "group_token_exact",
    )["regressed"]
    checks["all_checks_pass"] = all(checks.values())
    if not checks["all_checks_pass"]:
        raise AssertionError("group-count guard counterfactual failed verification")

    metrics = {key: _metrics(formula_rows, key) for key in ("fast", "challenger", "gated")}
    return {
        "schema": SCHEMA,
        "scope": "post-hoc shadow counterfactual only; no training, CROHME, runtime change, or promotion",
        "inputs": {
            "source_report": str(report_path.resolve()),
            "source_report_sha256": _sha256(report_path),
            "ownership": str(ownership_path.resolve()),
            "ownership_sha256": _sha256(ownership_path),
            "formulas": str(formulas_path.resolve()),
            "formulas_sha256": _sha256(formulas_path),
            "formula_rows": len(formula_rows),
            "writer_count": len(set(writer_by_id.values())),
            "challenger_arm": ARM,
            "geometry_prior_weight": 1.0,
        },
        "policy": "reject a local_2d challenger only when it emits more groups than the Fast incumbent; otherwise preserve the existing challenger result",
        "summary": {
            "metrics": metrics,
            "fast_to_challenger": {
                metric: _transition(formula_rows, "fast", "challenger", metric)
                for metric in ("group_exact", "token_exact", "group_token_exact")
            },
            "fast_to_gated": {
                metric: _transition(formula_rows, "fast", "gated", metric)
                for metric in ("group_exact", "token_exact", "group_token_exact")
            },
            "rejected_group_increases": rejected_increases,
            "writer_cluster_bootstrap": {
                metric: _writer_bootstrap(formula_rows, writer_by_id, metric)
                for metric in ("group_exact", "group_token_exact")
            },
        },
        "verification": checks,
    }


def _self_test() -> None:
    fast = {"groups": [[0, 1], [2]], "route": "fast"}
    challenger = {"groups": [[0], [1], [2]], "route": "local_2d"}
    reject = challenger["route"] == "local_2d" and len(challenger["groups"]) > len(fast["groups"])
    assert reject and fast["groups"] == [[0, 1], [2]]
    preserve = {"groups": [[0], [1, 2]], "route": "local_2d"}
    assert not (preserve["route"] == "local_2d" and len(preserve["groups"]) > len(fast["groups"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--ownership", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.source_report, args.ownership)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "all_checks_pass": report["verification"]["all_checks_pass"],
        "metrics": report["summary"]["metrics"],
        "fast_to_gated_group_exact": report["summary"]["fast_to_gated"]["group_exact"],
        "fast_to_gated_group_and_token_exact": report["summary"]["fast_to_gated"]["group_token_exact"],
        "rejected_group_increases": [
            row["sample_id"] for row in report["summary"]["rejected_group_increases"]
        ],
        "writer_cluster_bootstrap": report["summary"]["writer_cluster_bootstrap"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    _self_test()
    print('{"self_test":"pass"}')
    main()
