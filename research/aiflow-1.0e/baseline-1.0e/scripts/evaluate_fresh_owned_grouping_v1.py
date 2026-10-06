#!/usr/bin/env python3
"""One-shot untouched owned grouping acceptance for a frozen runtime candidate."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from formula_complete_gate_v1 import (
    DEFAULT_CONTEXT, DEFAULT_HWR, DEFAULT_RANKER, FormulaCompleteSessionV1,
)
from raw_formula_context_runtime_v1 import RawFormulaContextRuntimeV1


SCHEMA = "aiflow-fresh-owned-grouping-acceptance/v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _pairs(groups: set[frozenset[int]]) -> set[tuple[int, int]]:
    output = set()
    for group in groups:
        values = sorted(group)
        output.update(
            (first, second)
            for offset, first in enumerate(values)
            for second in values[offset + 1:]
        )
    return output


def _classify_partition(
    truth: set[frozenset[int]], predicted: set[frozenset[int]],
) -> tuple[bool, bool]:
    overmerge = any(
        sum(bool(group & item) for item in truth) > 1 for group in predicted
    )
    oversplit = any(
        sum(bool(group & item) for item in predicted) > 1 for group in truth
    )
    return overmerge, oversplit


def _evaluate(
    name: str, runtime: RawFormulaContextRuntimeV1,
    formulas: dict[str, dict], annotations: list[dict],
) -> tuple[dict, list[dict]]:
    exact = pair_tp = pair_fp = pair_fn = overmerge = oversplit = 0
    precomplete = cover_failures = fallback_failures = auto_commits = 0
    hwr_top1 = hwr_top5 = hwr_total = 0
    route_counts = Counter(); failures = []; rows = []
    for index, annotation in enumerate(annotations, 1):
        sample_id = str(annotation["sample_id"])
        source = formulas[sample_id]
        truth = {
            frozenset(int(value) for value in group)
            for group in annotation["groups"]
        }
        truth_labels = {
            frozenset(int(value) for value in group): str(label)
            for group, label in zip(
                annotation["groups"], annotation["labels"], strict=True,
            )
        }
        session = FormulaCompleteSessionV1(sample_id, runtime.infer)
        for stroke in source["strokes"]:
            event = session.append_stroke(stroke)
            precomplete += bool(event.get("committed"))
            precomplete += bool(event.get("finalized_tokens"))
            precomplete += bool(event.get("accuracy_scored"))
        result = session.complete()
        predicted = {
            frozenset(int(value) for value in group["stroke_indices"])
            for group in result["groups"]
        }
        assigned = [value for group in predicted for value in group]
        cover_ok = (
            sorted(assigned) == list(range(len(source["strokes"])))
            and len(assigned) == len(set(assigned))
        )
        cover_failures += not cover_ok
        fallback_failures += (
            result.get("raw_fallback", {}).get("strokes") != source["strokes"]
        )
        auto_commits += bool(
            result.get("product_decision", {}).get("product_auto_committed")
        )
        hit = predicted == truth; exact += hit
        truth_pairs = _pairs(truth); predicted_pairs = _pairs(predicted)
        pair_tp += len(truth_pairs & predicted_pairs)
        pair_fp += len(predicted_pairs - truth_pairs)
        pair_fn += len(truth_pairs - predicted_pairs)
        merged, split = _classify_partition(truth, predicted)
        overmerge += merged; oversplit += split
        route = str(result.get("audit", {}).get("grouping_model_route", "owned_baseline"))
        route_counts[route] += 1
        if hit:
            symbol_by_group = {
                frozenset(int(value) for value in row["stroke_indices"]): row
                for row in result["symbols"]
            }
            for group, label in truth_labels.items():
                symbol = symbol_by_group[group]
                hwr_total += 1
                hwr_top1 += str(symbol["hwr_top1"]) == label
                hwr_top5 += label in {str(value) for value in symbol["hwr_topk"]}
        if not hit:
            failures.append({
                "sample_id": sample_id, "writer": str(annotation["writer_id"]),
                "strokes": len(source["strokes"]),
                "truth": [sorted(group) for group in sorted(truth, key=lambda x: min(x))],
                "predicted": [sorted(group) for group in sorted(predicted, key=lambda x: min(x))],
                "overmerge": merged, "oversplit": split, "route": route,
            })
        rows.append({
            "sample_id": sample_id, "writer": str(annotation["writer_id"]),
            "strokes": len(source["strokes"]), "grouping_exact": hit,
            "overmerge": merged, "oversplit": split, "route": route,
        })
        if index % 20 == 0 or index == len(annotations):
            print(json.dumps({
                "event": "fresh_grouping_progress", "runtime": name,
                "completed": index, "total": len(annotations), "exact": exact,
            }), flush=True)
    precision = pair_tp / max(pair_tp + pair_fp, 1)
    recall = pair_tp / max(pair_tp + pair_fn, 1)
    total = len(annotations)
    if precomplete or cover_failures or fallback_failures or auto_commits:
        raise AssertionError(f"{name} formula-complete integrity failure")
    return {
        "formulas": total, "partition_exact_count": exact,
        "partition_exact": exact / max(total, 1),
        "pair_precision": precision, "pair_recall": recall,
        "pair_f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "overmerge_rate": overmerge / max(total, 1),
        "oversplit_rate": oversplit / max(total, 1),
        "hwr_truth_groups_scored": hwr_total,
        "hwr_top1_on_exact_grouping": hwr_top1 / max(hwr_total, 1),
        "hwr_top5_on_exact_grouping": hwr_top5 / max(hwr_total, 1),
        "route_counts": dict(sorted(route_counts.items())),
        "pre_complete_commits": precomplete,
        "stroke_cover_failures": cover_failures,
        "raw_fallback_failures": fallback_failures,
        "product_auto_commits": auto_commits,
        "failures": failures,
    }, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--fresh-ownership", type=Path, required=True)
    parser.add_argument("--baseline-ranker", type=Path, default=DEFAULT_RANKER)
    parser.add_argument("--candidate-ranker", type=Path, required=True)
    parser.add_argument("--hwr-checkpoint", type=Path, default=DEFAULT_HWR)
    parser.add_argument("--context-checkpoint", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    paths = [
        args.dataset_root.resolve(), args.fresh_ownership.resolve(),
        args.baseline_ranker.resolve(), args.candidate_ranker.resolve(),
        args.hwr_checkpoint.resolve(), args.context_checkpoint.resolve(),
        args.output.resolve(),
    ]
    if any(path.drive.upper() != "D:" for path in paths):
        parser.error("all inputs and outputs must remain on D:")
    if args.output.exists() or any(not path.exists() for path in paths[:-1]):
        parser.error("inputs must exist and output must be new")
    formulas_path = args.dataset_root / "data" / "formulas_valid.jsonl"
    formulas = {str(row["sample_id"]): row for row in _rows(formulas_path)}
    annotations = [row for row in _rows(args.fresh_ownership) if row.get("accepted")]
    if len(annotations) != 53 or len({str(row["writer_id"]) for row in annotations}) != 2:
        raise ValueError("frozen fresh acceptance contract mismatch")
    for row in annotations:
        sample_id = str(row["sample_id"])
        source = formulas.get(sample_id)
        if source is None:
            raise ValueError(f"fresh source missing: {sample_id}")
        assigned = [int(value) for group in row["groups"] for value in group]
        if sorted(assigned) != list(range(len(source["strokes"]))) or len(assigned) != len(set(assigned)):
            raise ValueError(f"fresh truth is not an exact cover: {sample_id}")
    baseline_runtime = RawFormulaContextRuntimeV1.from_artifacts(
        args.baseline_ranker, args.hwr_checkpoint, args.context_checkpoint,
        device=args.device, emit_formula_layout_shadow=True,
        allow_posthoc_shadow=True,
    )
    candidate_runtime = RawFormulaContextRuntimeV1.from_artifacts(
        args.candidate_ranker, args.hwr_checkpoint, args.context_checkpoint,
        device=args.device, emit_formula_layout_shadow=True,
        allow_posthoc_shadow=True,
    )
    baseline, baseline_rows = _evaluate(
        "baseline", baseline_runtime, formulas, annotations,
    )
    candidate, candidate_rows = _evaluate(
        "candidate", candidate_runtime, formulas, annotations,
    )
    by_id = {row["sample_id"]: row for row in baseline_rows}
    improved = [
        row["sample_id"] for row in candidate_rows
        if row["grouping_exact"] and not by_id[row["sample_id"]]["grouping_exact"]
    ]
    regressed = [
        row["sample_id"] for row in candidate_rows
        if not row["grouping_exact"] and by_id[row["sample_id"]]["grouping_exact"]
    ]
    report = {
        "schema": SCHEMA, "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_untouched_acceptance_complete",
        "selection_performed": False, "threshold_tuning_performed": False,
        "training_performed": False, "gradient_updates": 0,
        "frozen_acceptance": {"formulas": 53, "writers": 2},
        "baseline": baseline, "candidate": candidate,
        "comparison": {
            "partition_exact_delta": (
                candidate["partition_exact"] - baseline["partition_exact"]
            ),
            "improved": improved, "regressed": regressed,
            "net_exact_formulas": len(improved) - len(regressed),
        },
        "contracts": {
            "all_strokes_exactly_once": True, "pre_complete_commits": 0,
            "raw_fallback_exact": True, "product_auto_commits": 0,
            "target_label_writer_or_glyph_count_input": False,
            "fresh_rows_used_for_training_or_selection": 0,
        },
        "inputs": {
            "formulas_sha256": _sha256(formulas_path),
            "fresh_ownership_sha256": _sha256(args.fresh_ownership),
            "baseline_ranker_sha256": _sha256(args.baseline_ranker),
            "candidate_ranker_sha256": _sha256(args.candidate_ranker),
            "hwr_checkpoint_sha256": _sha256(args.hwr_checkpoint),
            "context_checkpoint_sha256": _sha256(args.context_checkpoint),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n",
    )
    print(json.dumps({
        "event": "fresh_grouping_finished", "output": str(args.output),
        "baseline": baseline["partition_exact"],
        "candidate": candidate["partition_exact"],
        "comparison": report["comparison"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
