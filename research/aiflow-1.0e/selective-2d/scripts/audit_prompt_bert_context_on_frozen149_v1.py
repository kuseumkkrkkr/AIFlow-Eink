#!/usr/bin/env python3
"""Replay the frozen BERT-Tiny candidate reranker on the current 149 formulas.

This is evaluation only: it neither trains nor tunes on the frozen 149-formula
set, and it never loads CROHME data. Grouping errors count as formula failures.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch

from train_masked_context_reranker_v1 import (
    DEFAULT_PRETRAINED,
    load_product_checkpoint,
    rerank_formula_rows,
)
from train_prompt_context_reranker_v1 import STRICT_TOKENS, _strict_lock


SCHEMA = "aiflow-prompt-bert-current149-shadow/v1"
ROOT = Path(__file__).resolve().parents[1]
CANONICAL_ROOT = Path(r"D:\AIFlow-Workspace\Projects\Aiflow\aiflow-math-ink-1.0")
DEFAULT_SUMMARY = ROOT / "artifacts" / "hwr_failure_cause_microscope_20260928" / "group_mean_geometry_prior_shadow_20261001" / "summary.json"
DEFAULT_DATA = Path(r"D:\AIFlow-Workspace\PrivateData\math-ink-data-collector\derived\fresh-context-acceptance-20260820-r2\frozen_acceptance\frozen_dataset\data\formulas_valid.jsonl")
DEFAULT_CONTEXT_REPORT = CANONICAL_ROOT / "artifacts" / "prompt_context_bert_tiny_20260820_r3_epoch3_shadow" / "prompt_context_report.json"
DEFAULT_PRETRAINED_CONTEXT = CANONICAL_ROOT / "research" / "pretrained" / "google-bert-tiny"
DEFAULT_CHECKPOINT = CANONICAL_ROOT / "artifacts" / "prompt_context_bert_tiny_20260820_r3_epoch3_shadow" / "masked_context_product.pt"
DEFAULT_OUTPUT = ROOT / "artifacts" / "hwr_failure_cause_microscope_20260928" / "mini_lm_top5_shadow_20261002" / "mini_lm_top5_shadow_r4.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _finite_points(strokes: list[dict], stroke_ids: list[int]) -> list[tuple[float, float]]:
    points = []
    for stroke_id in stroke_ids:
        if stroke_id not in strokes:
            raise ValueError(f"candidate references missing stroke {stroke_id}")
        for point in strokes[stroke_id]["points"]:
            x, y = float(point["x"]), float(point["y"])
            if not math.isfinite(x) or not math.isfinite(y):
                raise ValueError("non-finite ink coordinate")
            points.append((x, y))
    if not points:
        raise ValueError("candidate group has no ink points")
    return points


def _formula_rows(summary: dict, raw_by_id: dict[str, dict]) -> tuple[list[dict], dict[str, dict]]:
    rows: list[dict] = []
    targets: dict[str, dict] = {}
    seen_records: set[str] = set()
    for record in summary["records"]:
        sample_id = str(record["sample_id"])
        raw = raw_by_id.get(sample_id)
        if raw is None:
            raise ValueError(f"frozen data missing shadow formula {sample_id}")
        if "crohme" in str(raw.get("source_partition", "")).casefold():
            raise ValueError("CROHME row found in requested evaluation input")

        fast = record["hwr_tournament"]["fast"]
        symbols = fast["selected_symbols"]
        if not symbols:
            raise ValueError(f"empty HWR sequence: {sample_id}")
        stroke_map = {int(stroke["stroke_id"]): stroke for stroke in raw["strokes"]}
        if len(stroke_map) != len(raw["strokes"]):
            raise ValueError(f"duplicate raw stroke ID: {sample_id}")
        all_stroke_ids = sorted(stroke_map)
        all_points = _finite_points(stroke_map, all_stroke_ids)
        formula_left = min(x for x, _ in all_points)
        formula_top = min(y for _, y in all_points)
        formula_width = max(max(x for x, _ in all_points) - formula_left, 1e-6)
        formula_height = max(max(y for _, y in all_points) - formula_top, 1e-6)

        group_exact = bool(record["fast_group_exact"])
        target_tokens = [str(value) for value in fast.get("target_tokens", [])]
        raw_target_tokens = [str(cell["token"]) for cell in raw["target_cells"]]
        if group_exact:
            if target_tokens != raw_target_tokens:
                raise ValueError(f"target order mismatch between summary and raw source: {sample_id}")
            target_groups = [[int(index) for index in group] for group in fast.get("target_groups", [])]
            actual_groups = [[int(index) for index in symbol["stroke_indices"]] for symbol in symbols]
            if actual_groups != target_groups or len(symbols) != len(target_tokens):
                raise ValueError(f"group-exact alignment contract failed: {sample_id}")
            targets[sample_id] = {
                "tokens": target_tokens,
                "writer_id": str(raw["writer_id"]),
                "group_exact": True,
            }
        else:
            targets[sample_id] = {
                "tokens": raw_target_tokens,
                "writer_id": str(raw["writer_id"]),
                "group_exact": False,
            }

        for index, symbol in enumerate(symbols):
            record_id = f"{sample_id}:{index}"
            if record_id in seen_records:
                raise ValueError(f"duplicate HWR record ID: {record_id}")
            seen_records.add(record_id)
            stroke_ids = [int(value) for value in symbol["stroke_indices"]]
            points = _finite_points(stroke_map, stroke_ids)
            left, right = min(x for x, _ in points), max(x for x, _ in points)
            top, bottom = min(y for _, y in points), max(y for _, y in points)
            width, height = right - left, bottom - top
            topk = [str(value) for value in symbol["hwr_topk"]]
            probabilities = [float(value) for value in symbol["hwr_topk_probabilities"]]
            if not topk or len(topk) != len(probabilities) or any(not math.isfinite(p) or p < 0 for p in probabilities):
                raise ValueError(f"invalid HWR Top-K contract: {record_id}")
            rows.append({
                "record_id": record_id,
                "formula_id": sample_id,
                "final_topk": topk,
                "final_topk_probabilities": probabilities,
                "geometry": {
                    "left": left,
                    "top": top,
                    "right": right,
                    "bottom": bottom,
                    "width": width,
                    "height": height,
                    "center_x": ((left + right) / 2.0 - formula_left) / formula_width,
                    "center_y": ((top + bottom) / 2.0 - formula_top) / formula_height,
                    "width_rel": width / formula_width,
                    "height_rel": height / formula_height,
                    "stroke_count": float(len(stroke_ids)),
                },
                # Saved Fast group order is the existing deterministic linearization.
                "context": {"index": index, "length": len(symbols)},
            })
    if set(targets) != {str(record["sample_id"]) for record in summary["records"]}:
        raise AssertionError("formula-target coverage mismatch")
    return rows, targets


def _formula_metrics(rows: list[dict], targets: dict[str, dict], predictions: dict[str, str]) -> dict:
    by_formula: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_formula[str(row["formula_id"])].append(row)
    base_exact_ids, candidate_exact_ids = set(), set()
    group_exact_count = 0
    token_total = token_base_hits = token_candidate_hits = token_top5_hits = 0
    token_changed = token_improved = token_regressed = 0
    rank_histogram: Counter[str] = Counter()
    top5_complete_ids = set()
    candidate_preserved = 0
    transitions = Counter()
    deltas: dict[str, int] = {}
    for formula_id, sequence in by_formula.items():
        sequence.sort(key=lambda row: int(row["context"]["index"]))
        target = targets[formula_id]
        group_exact = bool(target["group_exact"])
        before = [str(row["final_topk"][0]) for row in sequence]
        after = [str(predictions[str(row["record_id"])]) for row in sequence]
        base_ok = group_exact and before == target["tokens"]
        candidate_ok = group_exact and after == target["tokens"]
        base_exact_ids.add(formula_id) if base_ok else None
        candidate_exact_ids.add(formula_id) if candidate_ok else None
        transitions[("exact" if base_ok else "wrong", "exact" if candidate_ok else "wrong")] += 1
        deltas[formula_id] = int(candidate_ok) - int(base_ok)
        if not group_exact:
            continue
        group_exact_count += 1
        all_candidates_hit = True
        for row, expected, old, new in zip(sequence, target["tokens"], before, after, strict=True):
            token_total += 1
            old_ok, new_ok = old == expected, new == expected
            token_base_hits += int(old_ok)
            token_candidate_hits += int(new_ok)
            token_changed += int(old != new)
            token_improved += int(not old_ok and new_ok)
            token_regressed += int(old_ok and not new_ok)
            candidate_preserved += int(new in row["final_topk"])
            if expected in row["final_topk"]:
                token_top5_hits += 1
                rank_histogram[str(row["final_topk"].index(expected) + 1)] += 1
            else:
                all_candidates_hit = False
        if all_candidates_hit:
            top5_complete_ids.add(formula_id)
    formula_count = len(by_formula)
    return {
        "formulas": formula_count,
        "group_exact_formulas": group_exact_count,
        "baseline_top1_formula_exact": len(base_exact_ids),
        "reranked_formula_exact": len(candidate_exact_ids),
        "delta_formula_exact": len(candidate_exact_ids) - len(base_exact_ids),
        "formula_exact_transitions": {
            f"{before}_to_{after}": count for (before, after), count in sorted(transitions.items())
        },
        "writer_bootstrap": _writer_bootstrap(deltas, targets),
        "exact_group_tokens": token_total,
        "baseline_top1_token_hits": token_base_hits,
        "reranked_top1_token_hits": token_candidate_hits,
        "token_top5_hits": token_top5_hits,
        "token_top5_recall": token_top5_hits / token_total if token_total else None,
        "target_rank_histogram_on_exact_groups": dict(sorted(rank_histogram.items(), key=lambda item: int(item[0]))),
        "top5_complete_formula_count_on_exact_groups": len(top5_complete_ids),
        "top5_complete_formula_count_over_all_149": len(top5_complete_ids),
        "changed_tokens": token_changed,
        "improved_tokens": token_improved,
        "regressed_tokens": token_regressed,
        "candidate_preserved_tokens": candidate_preserved,
        "candidate_preservation_rate": candidate_preserved / token_total if token_total else None,
        "new_tokens": 0,
        "grouping_mutations": 0,
        "grouping_wrong_formulas_counted_as_failures": formula_count - group_exact_count,
    }


def _writer_bootstrap(deltas: dict[str, int], targets: dict[str, dict], *, replicates: int = 20000) -> dict:
    by_writer: dict[str, list[int]] = defaultdict(list)
    for formula_id, delta in deltas.items():
        by_writer[str(targets[formula_id]["writer_id"])].append(delta)
    writers = sorted(by_writer)
    if not writers:
        return {"writers": 0, "replicates": 0, "delta_formula_exact_pp_95_ci": None}
    rng = random.Random(20261002)
    samples = []
    for _ in range(replicates):
        selected = [rng.choice(writers) for _ in writers]
        total_delta = sum(sum(by_writer[writer]) for writer in selected)
        total_formulas = sum(len(by_writer[writer]) for writer in selected)
        samples.append(100.0 * total_delta / total_formulas)
    samples.sort()
    return {
        "writers": len(writers),
        "formulas_by_writer": {writer: len(by_writer[writer]) for writer in writers},
        "replicates": replicates,
        "delta_formula_exact_pp_95_ci": [samples[int(0.025 * replicates)], samples[int(0.975 * replicates) - 1]],
    }


def _case_audit(rows: list[dict], targets: dict[str, dict], baseline: dict[str, str], predictions: dict[str, str]) -> dict:
    by_formula: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_formula[str(row["formula_id"])].append(row)
    recovered, regressed, changed_symbols = [], [], []
    correction_types: Counter[tuple[str, str, str]] = Counter()
    for formula_id, sequence in by_formula.items():
        sequence.sort(key=lambda row: int(row["context"]["index"]))
        target = targets[formula_id]
        if not target["group_exact"]:
            continue
        truth = target["tokens"]
        before = [baseline[str(row["record_id"])] for row in sequence]
        after = [predictions[str(row["record_id"])] for row in sequence]
        before_exact, after_exact = before == truth, after == truth
        if not before_exact and after_exact:
            recovered.append(formula_id)
        elif before_exact and not after_exact:
            regressed.append(formula_id)
        for row, expected, old, new in zip(sequence, truth, before, after, strict=True):
            if old == new:
                continue
            topk = row["final_topk"]
            probabilities = row["final_topk_probabilities"]
            probability_map = {token: float(probability) for token, probability in zip(topk, probabilities, strict=True)}
            correction_types[(expected, old, new)] += 1
            changed_symbols.append({
                "record_id": str(row["record_id"]),
                "expected": expected,
                "before": old,
                "after": new,
                "expected_original_rank": topk.index(expected) + 1 if expected in topk else None,
                "before_probability": probability_map[old],
                "after_probability": probability_map[new],
                "formula_exact_before": before_exact,
                "formula_exact_after": after_exact,
            })
    return {
        "group_exact_formula_recoveries": recovered,
        "group_exact_formula_regressions": regressed,
        "changed_symbol_count": len(changed_symbols),
        "changed_symbols": changed_symbols,
        "most_common_expected_before_after": [
            {"expected": expected, "before": before, "after": after, "count": count}
            for (expected, before, after), count in correction_types.most_common(30)
        ],
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--context-report", type=Path, default=DEFAULT_CONTEXT_REPORT)
    parser.add_argument("--pretrained", type=Path, default=DEFAULT_PRETRAINED_CONTEXT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    args = parser.parse_args()

    summary_path, data_path = args.summary.resolve(), args.data.resolve()
    context_report_path = args.context_report.resolve()
    pretrained, checkpoint, output = args.pretrained.resolve(), args.checkpoint.resolve(), args.output.resolve()
    for label, path in (("summary", summary_path), ("frozen formula data", data_path), ("context report", context_report_path), ("checkpoint", checkpoint)):
        if not path.is_file():
            parser.error(f"{label} file is missing: {path}")
    if output.exists():
        parser.error(f"refusing to overwrite existing report: {output}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("schema") != "aiflow-selective-2d-research-loop/v1":
        raise ValueError(f"unexpected saved-summary schema: {summary.get('schema')}")
    if summary.get("crohme_training_or_tuning") is not False or summary.get("product_default_enabled") is not False:
        raise ValueError("source summary violates CROHME/product-off guard")
    expected_data_hash = str(summary["inputs"]["formulas_valid_sha256"])
    data_hash = _sha256(data_path)
    if data_hash != expected_data_hash:
        raise ValueError("frozen formulas_valid hash differs from saved shadow provenance")
    context_report = json.loads(context_report_path.read_text(encoding="utf-8"))
    if context_report.get("schema") != "aiflow-prompt-context-reranker/v1":
        raise ValueError("unexpected frozen context report schema")
    if "CROHME unused" not in str(context_report["training"].get("lambda_selection", "")):
        raise ValueError("frozen context checkpoint report does not attest CROHME exclusion")
    expected_checkpoint_hash = str(context_report["provenance"]["checkpoint_sha256"])
    checkpoint_hash = _sha256(checkpoint)
    if checkpoint_hash != expected_checkpoint_hash:
        raise ValueError("context checkpoint hash differs from its saved report")
    prompt_corpus_path = Path(context_report["provenance"]["prompt_corpus"]).resolve()
    if not prompt_corpus_path.is_file():
        raise ValueError("checkpoint's project-owned prompt corpus is missing")
    prompt_corpus_hash = _sha256(prompt_corpus_path)
    if prompt_corpus_hash != str(context_report["provenance"]["prompt_corpus_sha256"]):
        raise ValueError("prompt corpus hash differs from the checkpoint report")
    prompt_rows = _jsonl(prompt_corpus_path)

    raw_rows = _jsonl(data_path)
    raw_by_id = {str(row["sample_id"]): row for row in raw_rows}
    if len(raw_by_id) != len(raw_rows):
        raise ValueError("duplicate sample IDs in frozen formula file")
    expected_ids = {str(record["sample_id"]) for record in summary["records"]}
    if len(summary["records"]) != 149 or len(expected_ids) != 149 or not expected_ids <= set(raw_by_id):
        raise ValueError("expected exact coverage of the 149 saved shadow formulas")
    if any("crohme" in str(raw_by_id[sample_id].get("source_partition", "")).casefold() for sample_id in expected_ids):
        raise ValueError("CROHME-marked source entered this evaluation")
    prompt_sequences = {tuple(str(token) for token in row["labels"]) for row in prompt_rows}
    current_sequences = {
        tuple(str(cell["token"]) for cell in raw_by_id[sample_id]["target_cells"])
        for sample_id in expected_ids
    }
    exact_training_sequence_overlaps = len(prompt_sequences & current_sequences)

    rows, targets = _formula_rows(summary, raw_by_id)
    if len(rows) != len({str(row["record_id"]) for row in rows}):
        raise ValueError("duplicate candidate rows generated")
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    if device_name == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    device = torch.device(device_name)
    model, contract, payload = load_product_checkpoint(pretrained, checkpoint, device)
    if float(payload["lambda"]) != float(context_report["training"]["product_lambda"]):
        raise ValueError("checkpoint fusion weight differs from attested report")

    baseline = {str(row["record_id"]): str(row["final_topk"][0]) for row in rows}
    start = time.perf_counter()
    raw_predictions = rerank_formula_rows(model, contract, payload, rows, device)
    inference_elapsed_ms = 1000.0 * (time.perf_counter() - start)
    product_predictions = _strict_lock(rows, raw_predictions)
    baseline_metrics = _formula_metrics(rows, targets, baseline)
    raw_metrics = _formula_metrics(rows, targets, raw_predictions)
    product_metrics = _formula_metrics(rows, targets, product_predictions)
    if any(product_predictions[str(row["record_id"])] not in row["final_topk"] for row in rows):
        raise AssertionError("candidate-preservation invariant violated")
    report = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "diagnostic_shadow_only_consumed_development",
        "protocol": {
            "training_performed": False,
            "current_149_lambda_or_threshold_tuning": False,
            "crohme_rows_loaded": 0,
            "crohme_training_or_tuning": False,
            "model_input_contains_truth": False,
            "candidate_policy": "fuse the existing HWR Top-5 with fixed masked-context scores, then apply the product strict-homograph Top-1 lock; no new classes",
            "strict_homograph_locked_tokens": sorted(STRICT_TOKENS),
            "context_order": "saved Fast selected-symbol order; group-exact alignment checked against saved target groups",
            "grouping_wrong_formula_policy": "count as formula-level failure; exclude its misaligned groups from symbol metrics",
            "promotion_eligible": False,
            "warning": "149 formulas have been inspected during development; results are diagnostic, not independent acceptance evidence",
        },
        "provenance": {
            "saved_shadow_summary": str(summary_path),
            "saved_shadow_summary_sha256": _sha256(summary_path),
            "frozen_formulas_valid": str(data_path),
            "frozen_formulas_valid_sha256": data_hash,
            "context_model_report": str(context_report_path),
            "context_model_report_sha256": _sha256(context_report_path),
            "project_owned_prompt_corpus": str(prompt_corpus_path),
            "project_owned_prompt_corpus_sha256": prompt_corpus_hash,
            "project_owned_prompt_corpus_rows": len(prompt_rows),
            "exact_formula_token_sequence_overlap_with_prompt_corpus": exact_training_sequence_overlaps,
            "context_checkpoint": str(checkpoint),
            "context_checkpoint_sha256": checkpoint_hash,
        },
        "model": {
            "model_id": payload["model_id"],
            "model_revision": payload["model_revision"],
            "hidden_size": 128,
            "layers": 2,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "checkpoint_bytes": checkpoint.stat().st_size,
            "fixed_lambda": float(payload["lambda"]),
            "device": device_name,
            "host_inference_elapsed_ms_not_mobile_latency": inference_elapsed_ms,
        },
        "evaluation": {
            "baseline_fast_top1": baseline_metrics,
            "unguarded_reranker_diagnostic_upper_bound": raw_metrics,
            "frozen_product_contract_with_strict_homograph_lock": product_metrics,
            "formula_exact_delta": product_metrics["delta_formula_exact"],
            "symbol_hit_delta": product_metrics["reranked_top1_token_hits"] - baseline_metrics["baseline_top1_token_hits"],
            "correction_microscope": {
                "product_contract": _case_audit(rows, targets, baseline, product_predictions),
                "unguarded_upper_bound": _case_audit(rows, targets, baseline, raw_predictions),
            },
        },
        "decision": {
            "automatic_default_replacement": False,
            "runtime_status": "shadow",
            "next_gate": "replicate on fresh writer/formula-disjoint project-owned data and on-device latency/memory acceptance",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({
        "event": "prompt_bert_frozen149_shadow_complete",
        "output": str(output),
        "baseline_exact": baseline_metrics["baseline_top1_formula_exact"],
        "product_locked_exact": product_metrics["reranked_formula_exact"],
        "product_locked_delta": product_metrics["delta_formula_exact"],
        "unguarded_exact_upper_bound": raw_metrics["reranked_formula_exact"],
        "symbol_hits_before_after_product": [baseline_metrics["baseline_top1_token_hits"], product_metrics["reranked_top1_token_hits"]],
        "product_changed_improved_regressed_tokens": [product_metrics["changed_tokens"], product_metrics["improved_tokens"], product_metrics["regressed_tokens"]],
        "product_candidate_preservation_rate": product_metrics["candidate_preservation_rate"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
