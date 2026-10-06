#!/usr/bin/env python3
"""Evaluate the current formula-complete runtime from raw CROHME strokes.

CROHME is validation-only.  No truth, writer, or glyph count is supplied to
inference.  Scores are local grouping/order/relation proxies, not official
LgEval/symLG Expression Rate.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import torch

from audit_replay_protocol_v1 import CROHME_ALIASES, _crohme_sample
from character_tensor_v1 import ROOT
from evaluate_48hz_prefix_v1 import _sha256
from evaluate_formula_layout_v1 import STRUCTURAL, _truth_edges
from formula_complete_gate_v1 import (
    DEFAULT_CONTEXT, DEFAULT_HWR, DEFAULT_RANKER, FormulaCompleteSessionV1,
)
from raw_formula_context_runtime_v1 import RawFormulaContextRuntimeV1
from replay_evaluate_hwr_v1 import load_crohme


SCHEMA = "aiflow-formula-complete-raw-crohme-validation/v1"
TICK_MS = 1000.0 / 48.0
DEFAULT_CROHME = (
    ROOT / "datasets" / "30_noncommercial_evaluation" / "crohme2019"
    / "crohme2019" / "crohme2019" / "test"
)
DEFAULT_OUTPUT = (
    ROOT / "artifacts" / "formula_complete_raw_crohme_20260821_r1"
)


def _source(formula_id: str, sample: dict) -> dict:
    tick = 0
    strokes = []
    for order, stroke in enumerate(sample["strokes"]):
        points = []
        for x, y, _time in stroke:
            points.append({"x": float(x), "y": float(y), "t_ms": tick * TICK_MS})
            tick += 1
        strokes.append({"order": order, "points": points})
    return {"formula_id": formula_id, "strokes": strokes}


def _truth_rows(formula_id: str, sample: dict, labels: list[str]) -> list[dict]:
    rows = []
    for index, (group, label) in enumerate(zip(sample["groups"], labels, strict=True)):
        points = [
            point for stroke_index in group for point in sample["strokes"][stroke_index]
        ]
        left = min(float(point[0]) for point in points)
        right = max(float(point[0]) for point in points)
        top = min(float(point[1]) for point in points)
        bottom = max(float(point[1]) for point in points)
        rows.append({
            "record_id": f"{formula_id}:{index}", "formula_id": formula_id,
            "label": label, "final_topk": [label],
            "final_topk_probabilities": [1.0],
            "geometry": {
                "left": left, "right": right, "top": top, "bottom": bottom,
                "center_x": (left + right) / 2.0,
                "center_y": (top + bottom) / 2.0,
            },
            "context": {"index": index, "length": len(labels)},
        })
    return rows


def _evaluate_result(
    formula_id: str, path: Path, sample: dict, truth_tokens: list[str], result: dict,
) -> dict:
    truth_groups = {
        frozenset(int(value) for value in group): f"{formula_id}:{index}"
        for index, group in enumerate(sample["groups"])
    }
    selected_groups = {
        frozenset(int(value) for value in row["stroke_indices"]): str(row["record_id"])
        for row in result["groups"]
    }
    grouping_exact = set(selected_groups) == set(truth_groups)
    all_strokes = [
        int(value) for row in result["groups"] for value in row["stroke_indices"]
    ]
    exact_cover = (
        sorted(all_strokes) == list(range(len(sample["strokes"])))
        and len(all_strokes) == len(set(all_strokes))
    )
    row = {
        "formula_id": formula_id,
        "truth_tokens": truth_tokens,
        "finalized_tokens": list(result["finalized_tokens"]),
        "groups": len(result["groups"]),
        "truth_groups": len(sample["groups"]),
        "grouping_exact": grouping_exact,
        "flat_token_sequence_exact": list(result["finalized_tokens"]) == truth_tokens,
        "all_strokes_exactly_once": exact_cover,
        "pre_complete_commits": int(
            result["formula_complete_gate"]["pre_complete_commits"]
        ),
        "prefix_accuracy_scored": bool(
            result["formula_complete_gate"]["prefix_accuracy_scored"]
        ),
        "layout_order_exact": False,
        "layout_token_sequence_exact": False,
        "relation_parsed": False,
        "relation_exact": False,
        "truth_relations": [],
        "predicted_relations": [],
        "strict_group_layout_relation_character_exact": False,
    }
    if not grouping_exact:
        return row
    runtime_to_truth = {
        runtime_id: truth_groups[group]
        for group, runtime_id in selected_groups.items()
    }
    layout = result.get("formula_layout_shadow") or {}
    ordered_ids = [
        runtime_to_truth[str(value)] for value in layout.get("ordered_record_ids") or []
    ]
    expected_ids = [f"{formula_id}:{index}" for index in range(len(truth_tokens))]
    row["layout_order_exact"] = ordered_ids == expected_ids
    row["layout_token_sequence_exact"] = (
        [str(value) for value in layout.get("ordered_tokens") or []] == truth_tokens
    )
    try:
        truth = _truth_edges(path, _truth_rows(formula_id, sample, truth_tokens))
    except ET.ParseError:
        return row
    predicted = {
        (
            runtime_to_truth[str(edge["parent"])],
            runtime_to_truth[str(edge["child"])],
            str(edge["type"]),
        )
        for edge in layout.get("relations") or []
        if str(edge.get("type")) in STRUCTURAL
    }
    row["relation_parsed"] = True
    row["relation_exact"] = predicted == truth
    row["truth_relations"] = sorted(truth)
    row["predicted_relations"] = sorted(predicted)
    row["strict_group_layout_relation_character_exact"] = bool(
        row["layout_order_exact"]
        and row["layout_token_sequence_exact"]
        and row["relation_exact"]
    )
    return row


def _write_rows(path: Path, rows: list[dict]) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition-ranker", type=Path, default=DEFAULT_RANKER)
    parser.add_argument("--hwr-checkpoint", type=Path, default=DEFAULT_HWR)
    parser.add_argument("--context-checkpoint", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--crohme", type=Path, default=DEFAULT_CROHME)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    paths = [
        args.partition_ranker.resolve(), args.hwr_checkpoint.resolve(),
        args.context_checkpoint.resolve(), args.crohme.resolve(), args.output.resolve(),
    ]
    if any(path.drive.upper() != "D:" for path in paths):
        parser.error("all inputs and outputs must remain on D:")
    if args.output.exists() or args.limit < 0:
        parser.error("output must be new and limit non-negative")
    if any(not path.exists() for path in paths[:-1]):
        parser.error("one or more frozen inputs are missing")
    torch.set_grad_enabled(False)
    if torch.is_grad_enabled():
        raise AssertionError("CROHME validation must run with autograd disabled")

    hwr_payload = torch.load(args.hwr_checkpoint, map_location="cpu", weights_only=False)
    hwr_policy = hwr_payload.get("report", {}).get("data_policy", {})
    if hwr_policy.get("project_owned_formula_grouping_training") is not False:
        raise ValueError("HWR training provenance boundary is missing")
    context_payload = torch.load(args.context_checkpoint, map_location="cpu", weights_only=False)
    context_training = str(context_payload.get("training_data", ""))
    if "crohme" in context_training.casefold() or "project-owned" not in context_training:
        raise ValueError("context checkpoint training provenance is invalid")
    runtime = RawFormulaContextRuntimeV1.from_artifacts(
        args.partition_ranker, args.hwr_checkpoint, args.context_checkpoint,
        device=args.device, emit_formula_layout_shadow=True,
        allow_posthoc_shadow=True,
    )
    protocol, duplicates = load_crohme(args.crohme)
    labels = set(runtime.labels)
    eligible = []
    skipped = Counter()
    for formula in protocol:
        formula_id = str(formula["record_id"])
        sample = _crohme_sample(args.crohme / formula_id)
        if sample is None:
            skipped["missing_truth_partition"] += 1
            continue
        if not sample["partition_complete"]:
            skipped["incomplete_truth_partition"] += 1
            continue
        truth_tokens = [
            CROHME_ALIASES.get(str(value), str(value)) for value in sample["labels"]
        ]
        if any(token not in labels for token in truth_tokens):
            skipped["unsupported_truth_label"] += 1
            continue
        if any(len(stroke) < 2 for stroke in sample["strokes"]):
            skipped["single_point_source_stroke"] += 1
            continue
        eligible.append((formula_id, sample, truth_tokens))
    if args.limit:
        eligible = eligible[:args.limit]

    rows = []
    for index, (formula_id, sample, truth_tokens) in enumerate(eligible, 1):
        source = _source(formula_id, sample)
        session = FormulaCompleteSessionV1(formula_id, runtime.infer)
        events = [session.append_stroke(stroke) for stroke in source["strokes"]]
        if any(
            event.get("committed") is not False
            or event.get("finalized_tokens") != []
            or event.get("accuracy_scored") is not False
            for event in events
        ):
            raise AssertionError("pre-complete event committed a result")
        result = session.complete()
        evaluated = _evaluate_result(
            formula_id, args.crohme / formula_id, sample, truth_tokens, result,
        )
        evaluated["raw_fallback_exact"] = (
            result.get("raw_fallback", {}).get("strokes") == source["strokes"]
        )
        evaluated["raw_fallback_sha256"] = result.get("raw_fallback", {}).get(
            "canonical_sha256"
        )
        evaluated["product_auto_committed"] = bool(
            result.get("product_decision", {}).get("product_auto_committed")
        )
        evaluated["product_decision"] = result.get("product_decision", {}).get("status")
        rows.append(evaluated)
        if index % 25 == 0 or index == len(eligible):
            print(json.dumps({
                "event": "raw_formula_complete_progress", "completed": index,
                "total": len(eligible),
                "grouping_exact": sum(row["grouping_exact"] for row in rows),
                "strict_exact": sum(
                    row["strict_group_layout_relation_character_exact"] for row in rows
                ),
            }, ensure_ascii=False), flush=True)

    if any(not row["all_strokes_exactly_once"] for row in rows):
        raise AssertionError("stroke loss or duplication detected")
    if any(row["pre_complete_commits"] for row in rows):
        raise AssertionError("pre-complete commit detected")
    if any(not row["raw_fallback_exact"] for row in rows):
        raise AssertionError("raw fallback differs from CROHME source strokes")
    if any(row["product_auto_committed"] for row in rows):
        raise AssertionError("shadow formula result was product-auto-committed")
    parsed = [row for row in rows if row["relation_parsed"]]
    structural = [row for row in parsed if row["truth_relations"]]
    relation_counts = Counter(tp=0, fp=0, fn=0)
    for row in parsed:
        truth = {tuple(value) for value in row["truth_relations"]}
        predicted = {tuple(value) for value in row["predicted_relations"]}
        relation_counts["tp"] += len(truth & predicted)
        relation_counts["fp"] += len(predicted - truth)
        relation_counts["fn"] += len(truth - predicted)
    precision = relation_counts["tp"] / max(
        relation_counts["tp"] + relation_counts["fp"], 1
    )
    recall = relation_counts["tp"] / max(
        relation_counts["tp"] + relation_counts["fn"], 1
    )
    args.output.mkdir(parents=True)
    predictions = args.output / "formula_results.jsonl.gz"
    _write_rows(predictions, rows)
    total = max(len(rows), 1)
    report = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "noncommercial_validation_only_posthoc_shadow",
        "training_performed": False,
        "selection_performed": False,
        "threshold_tuning_performed": False,
        "crohme_rows_used_for_training": 0,
        "crohme_gradient_updates": 0,
        "prefix_or_streaming_accuracy_scored": False,
        "decision_boundary": "formula_complete whole-formula only",
        "official_lgeval_or_symlg": False,
        "coverage": {
            "protocol_formulas": len(protocol),
            "exact_formula_duplicates_removed": len(duplicates),
            "eligible_formulas": len(rows),
            "limited_run": bool(args.limit),
            "skipped": dict(sorted(skipped.items())),
        },
        "scores": {
            "grouping_exact_count": sum(row["grouping_exact"] for row in rows),
            "grouping_exact": sum(row["grouping_exact"] for row in rows) / total,
            "flat_token_sequence_exact_count": sum(
                row["flat_token_sequence_exact"] for row in rows
            ),
            "flat_token_sequence_exact": sum(
                row["flat_token_sequence_exact"] for row in rows
            ) / total,
            "layout_order_exact_count": sum(row["layout_order_exact"] for row in rows),
            "layout_order_exact": sum(row["layout_order_exact"] for row in rows) / total,
            "layout_token_sequence_exact_count": sum(
                row["layout_token_sequence_exact"] for row in rows
            ),
            "layout_token_sequence_exact": sum(
                row["layout_token_sequence_exact"] for row in rows
            ) / total,
            "relation_exact_on_grouping_exact_parsed": (
                sum(row["relation_exact"] for row in parsed) / max(len(parsed), 1)
            ),
            "relation_exact_on_structural_formulas": (
                sum(row["relation_exact"] for row in structural) / max(len(structural), 1)
            ),
            "strict_group_layout_relation_character_exact_count": sum(
                row["strict_group_layout_relation_character_exact"] for row in rows
            ),
            "strict_group_layout_relation_character_exact": sum(
                row["strict_group_layout_relation_character_exact"] for row in rows
            ) / total,
            "relation_micro": {
                **dict(relation_counts), "precision": precision, "recall": recall,
                "f1": 2 * precision * recall / max(precision + recall, 1e-12),
            },
        },
        "contracts": {
            "all_input_strokes_exactly_once": True,
            "pre_complete_commits": 0,
            "prefix_accuracy_scored": False,
            "target_label_writer_or_glyph_count_input": False,
            "product_default_enabled": False,
            "posthoc_shadow_opt_in": True,
            "raw_fallback_exact_formulas": sum(
                row["raw_fallback_exact"] for row in rows
            ),
            "product_auto_commits": sum(
                row["product_auto_committed"] for row in rows
            ),
            "product_decision": "REVIEW_REQUIRED",
            "autograd_enabled": False,
        },
        "limits": [
            "local structural proxy only; official CROHME LgEval/symLG is not installed",
            "unsupported labels and single-point source strokes are excluded",
            "partition ranker was tuned on previously inspected project-owned formulas",
        ],
        "inputs": {
            "partition_ranker_sha256": _sha256(args.partition_ranker),
            "hwr_checkpoint_sha256": _sha256(args.hwr_checkpoint),
            "context_checkpoint_sha256": _sha256(args.context_checkpoint),
            "context_training_data": context_training,
        },
        "output": {
            "predictions": str(predictions),
            "predictions_sha256": _sha256(predictions),
        },
    }
    report_path = args.output / "validation_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "event": "raw_formula_complete_finished", "report": str(report_path),
        "scores": report["scores"], "crohme_gradient_updates": 0,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
