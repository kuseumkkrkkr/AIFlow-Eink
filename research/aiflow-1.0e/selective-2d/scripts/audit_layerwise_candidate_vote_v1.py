#!/usr/bin/env python3
"""Evaluate fixed intermediate-layer votes over final HWR Top-5 candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any


SCHEMA = "aiflow-hwr-layerwise-candidate-vote-microscope/v1"
STAGES = (
    "encoder.input",
    "encoder.block_0.output",
    "encoder.block_1.output",
    "encoder.block_2.output",
    "encoder.block_3.output",
    "encoder.output",
)
ARMS = {
    "final_only": ("encoder.output",),
    "last3_vote": ("encoder.block_2.output", "encoder.block_3.output", "encoder.output"),
    "last5_vote": STAGES[1:],
    "all6_vote": STAGES,
}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _predictions(trace: dict[str, Any]) -> tuple[list[str], list[str], dict[str, list[str]]]:
    targets = [str(token) for token in trace["source"]["target_tokens"]]
    symbols = trace["oracle_group_hwr"]["symbols"]
    if len(targets) != len(symbols):
        raise AssertionError(f"target/symbol length mismatch: {trace['sample_id']}")
    final = []
    candidates = []
    stage_predictions: dict[str, list[str]] = {stage: [] for stage in STAGES}
    for target, symbol in zip(targets, symbols, strict=True):
        prediction = symbol["prediction"]
        final_token = str(prediction["top1"])
        top5 = [str(token) for token in prediction["top5"]]
        if len(top5) != 5 or len(set(top5)) != 5 or final_token != top5[0]:
            raise AssertionError("saved HWR Top-5 contract mismatch")
        lens = symbol["layer_logit_lens"]
        if tuple(str(row["stage"]) for row in lens) != STAGES:
            raise AssertionError(f"layer stage contract mismatch: {trace['sample_id']}")
        if lens[-1]["predicted_top1"] != final_token or lens[-1]["target_rank"] != prediction["target_rank"]:
            raise AssertionError(f"final layer probe differs from saved classifier: {trace['sample_id']}")
        final.append(final_token)
        candidates.append(top5)
        for row in lens:
            stage_predictions[str(row["stage"])].append(str(row["predicted_top1"]))
    return targets, final, {"candidate_rows": candidates, "stages": stage_predictions}


def _vote(candidates: list[str], final_token: str, stage_tokens: list[str]) -> str:
    counts: dict[str, int] = {}
    for token in stage_tokens:
        if token in candidates:
            counts[token] = counts.get(token, 0) + 1
    if not counts:
        return final_token
    max_votes = max(counts.values())
    tied = {token for token, count in counts.items() if count == max_votes}
    return next(token for token in candidates if token in tied)


def _writer_bootstrap(
    formula_rows: list[dict[str, Any]], writer_by_id: dict[str, str], arm: str,
    *, iterations: int = 10_000, seed: int = 20261001,
) -> dict[str, Any]:
    by_writer: dict[str, list[dict[str, Any]]] = {}
    for row in formula_rows:
        by_writer.setdefault(writer_by_id[str(row["sample_id"])], []).append(row)
    writer_ids = sorted(by_writer)
    rng = random.Random(seed)
    deltas = []
    for _ in range(iterations):
        sampled = [rng.choice(writer_ids) for _ in writer_ids]
        denominator = sum(len(by_writer[writer]) for writer in sampled)
        before = sum(
            int(row["arms"]["final_only"]["exact"])
            for writer in sampled for row in by_writer[writer]
        )
        after = sum(
            int(row["arms"][arm]["exact"])
            for writer in sampled for row in by_writer[writer]
        )
        deltas.append(100.0 * (after - before) / denominator)
    deltas.sort()
    return {
        "method": "paired writer-cluster bootstrap; writers sampled with replacement",
        "iterations": iterations,
        "seed": seed,
        "delta_exact_rate_pp_95_interval": [
            deltas[int(0.025 * (iterations - 1))],
            deltas[int(0.975 * (iterations - 1))],
        ],
        "interpretation": "consumed-development diagnostic only; not independent acceptance",
    }


def audit(
    trace_path: Path, ownership_path: Path, cause_matrix_path: Path,
    writer_reference_path: Path,
) -> dict[str, Any]:
    traces = _jsonl(trace_path)
    ownership = _jsonl(ownership_path)
    cause_matrix = json.loads(cause_matrix_path.read_text(encoding="utf-8"))
    writer_reference = json.loads(writer_reference_path.read_text(encoding="utf-8"))
    trace_sha = _sha256(trace_path)
    ownership_sha = _sha256(ownership_path)
    cause_matrix_sha = _sha256(cause_matrix_path)
    writer_reference_sha = _sha256(writer_reference_path)
    if cause_matrix.get("schema") != "aiflow-hwr-failure-cause-microscope/v1":
        raise ValueError("unexpected failure-cause matrix schema")
    if writer_reference.get("schema") != "aiflow-semantic-infix-writer-stratified-shadow/v1":
        raise ValueError("unexpected writer reference schema")
    if cause_matrix["inputs"]["traces"]["sha256"] != trace_sha:
        raise AssertionError("trace hash differs from frozen failure-cause matrix")
    if writer_reference["inputs"]["ownership_train"]["sha256"] != ownership_sha:
        raise AssertionError("ownership hash differs from frozen writer reference")
    if len(traces) != 149 or len({str(row["sample_id"]) for row in traces}) != 149:
        raise AssertionError("expected 149 unique formulas")
    writer_by_id = {
        str(row["sample_id"]): str(row["writer_id"])
        for row in ownership if row.get("accepted")
    }
    ids = {str(trace["sample_id"]) for trace in traces}
    if set(writer_by_id) != ids:
        raise AssertionError("accepted ownership IDs do not exactly match traces")

    formula_rows = []
    stage_metrics: dict[str, dict[str, int]] = {
        stage: {"tokens": 0, "hits": 0, "formula_exact": 0} for stage in STAGES
    }
    baseline_tokens = 0
    target_tokens = 0
    for trace in traces:
        sample_id = str(trace["sample_id"])
        targets, final, probe = _predictions(trace)
        target_tokens += len(targets)
        for stage in STAGES:
            current = probe["stages"][stage]
            if len(current) != len(targets):
                raise AssertionError(f"stage row count mismatch: {sample_id}:{stage}")
            stage_metrics[stage]["tokens"] += len(targets)
            stage_metrics[stage]["hits"] += sum(a == b for a, b in zip(current, targets, strict=True))
            stage_metrics[stage]["formula_exact"] += int(current == targets)
        arms: dict[str, dict[str, Any]] = {}
        for arm, stages in ARMS.items():
            chosen = [
                _vote(candidates, final_token, [probe["stages"][stage][position] for stage in stages])
                for position, (candidates, final_token) in enumerate(zip(
                    probe["candidate_rows"], final, strict=True,
                ))
            ]
            if any(token not in candidates for token, candidates in zip(chosen, probe["candidate_rows"], strict=True)):
                raise AssertionError(f"layer vote invented a token: {sample_id}:{arm}")
            arms[arm] = {
                "tokens": chosen,
                "hits": sum(a == b for a, b in zip(chosen, targets, strict=True)),
                "exact": chosen == targets,
                "changed_positions": [
                    index for index, (before, after) in enumerate(zip(final, chosen, strict=True))
                    if before != after
                ],
            }
        baseline_tokens += sum(a == b for a, b in zip(final, targets, strict=True))
        formula_rows.append({"sample_id": sample_id, "targets": targets, "arms": arms})

    summary: dict[str, Any] = {}
    for arm in ARMS:
        token_hits = sum(int(row["arms"][arm]["hits"]) for row in formula_rows)
        exact = sum(int(row["arms"][arm]["exact"]) for row in formula_rows)
        changed = []
        recovered = []
        regressed = []
        for row in formula_rows:
            base = row["arms"]["final_only"]["tokens"]
            chosen = row["arms"][arm]["tokens"]
            targets = row["targets"]
            for position, (before, after, target) in enumerate(zip(base, chosen, targets, strict=True)):
                if before == after:
                    continue
                change = {"sample_id": row["sample_id"], "position": position,
                          "target": target, "before": before, "after": after}
                changed.append(change)
                if before != target and after == target:
                    recovered.append(change)
                elif before == target and after != target:
                    regressed.append(change)
        writer_rows = []
        for writer in sorted(set(writer_by_id.values())):
            rows = [row for row in formula_rows if writer_by_id[row["sample_id"]] == writer]
            writer_rows.append({
                "writer_id": writer,
                "formulas": len(rows),
                "baseline_exact": sum(int(row["arms"]["final_only"]["exact"]) for row in rows),
                "challenger_exact": sum(int(row["arms"][arm]["exact"]) for row in rows),
                "exact_delta": sum(int(row["arms"][arm]["exact"]) - int(row["arms"]["final_only"]["exact"]) for row in rows),
            })
        summary[arm] = {
            "formula_exact": exact,
            "formula_exact_rate": exact / len(formula_rows),
            "token_hits": token_hits,
            "token_count": target_tokens,
            "token_accuracy": token_hits / target_tokens,
            "delta_vs_final": {
                "formula_exact": exact - sum(int(row["arms"]["final_only"]["exact"]) for row in formula_rows),
                "token_hits": token_hits - baseline_tokens,
                "changed_token_count": len(changed),
                "recovered_tokens": recovered,
                "regressed_tokens": regressed,
            },
            "writer_stratified": writer_rows,
            "writer_cluster_bootstrap": (
                _writer_bootstrap(formula_rows, writer_by_id, arm) if arm != "final_only" else None
            ),
        }

    checks = {
        "formula_count_is_149": len(formula_rows) == 149,
        "target_token_count_is_579": target_tokens == 579,
        "ownership_hash_reconciled": bool(ownership_sha),
        "trace_hash_matches_failure_cause_matrix": cause_matrix["inputs"]["traces"]["sha256"] == trace_sha,
        "ownership_hash_matches_writer_reference": writer_reference["inputs"]["ownership_train"]["sha256"] == ownership_sha,
        "ownership_ids_exactly_match_trace": set(writer_by_id) == ids,
        "writer_count_is_9": len(set(writer_by_id.values())) == 9,
        "final_only_replays_76_exact_and_470_tokens": (
            summary["final_only"]["formula_exact"] == 76
            and summary["final_only"]["token_hits"] == 470
        ),
        "all_vote_outputs_preserve_final_top5_candidates": all(
            len(row["arms"][arm]["tokens"]) == len(row["targets"])
            for row in formula_rows for arm in ARMS
        ),
    }
    checks["all_checks_pass"] = all(checks.values())
    if not checks["all_checks_pass"]:
        raise AssertionError("layerwise candidate vote failed verification")
    return {
        "schema": SCHEMA,
        "scope": "frozen consumed-development Top-5 shadow; fixed untrained vote; no CROHME or promotion",
        "inputs": {
            "trace_sha256": trace_sha,
            "ownership_sha256": ownership_sha,
            "cause_matrix_sha256": cause_matrix_sha,
            "writer_reference_sha256": writer_reference_sha,
            "formula_count": len(traces),
            "writer_count": len(set(writer_by_id.values())),
            "token_count": target_tokens,
        },
        "policy": {
            "description": "among final Top-5 candidates, choose the most frequent intermediate/final stage argmax; ties resolve by saved Top-5 order",
            "arms": {name: list(stages) for name, stages in ARMS.items()},
            "training": "none",
        },
        "per_stage_top1": stage_metrics,
        "arms": summary,
        "verification": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--ownership", type=Path, required=True)
    parser.add_argument("--cause-matrix", type=Path, required=True)
    parser.add_argument("--writer-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.trace, args.ownership, args.cause_matrix, args.writer_reference)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "all_checks_pass": report["verification"]["all_checks_pass"],
        "arms": {name: {"formula_exact": row["formula_exact"], "token_hits": row["token_hits"], "delta": row["delta_vs_final"]} for name, row in report["arms"].items()},
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
