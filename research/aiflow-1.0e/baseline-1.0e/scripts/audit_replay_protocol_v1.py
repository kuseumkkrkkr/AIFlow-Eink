#!/usr/bin/env python3
"""Audit the fixed replay protocols with an existing 372-class checkpoint.

The script never trains.  Formula-level results below are deliberately named
``oracle_group_all_token`` because truth stroke groups are supplied; they are
not end-to-end grouping, relation, or LaTeX-decoder accuracy.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from build_normalized_ink_v1 import SourceSample, _canonicalize
from character_tensor_v1 import _json_lines, iter_direct_ownership_examples, tensorize
from replay_evaluate_hwr_v1 import _local_name, _trace_points, load_crohme, score_glyphs_from_rows
from train_character_classifier_v1 import ArrayDataset, InkClassifierV1, predict


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "artifacts" / "unified_head_20260813" / "unified_math_8ep_full"
DEFAULT_PROTOCOL = ROOT / "artifacts" / "hwr_replay_evaluation_v1"
DEFAULT_CROHME = ROOT / "datasets" / "30_noncommercial_evaluation" / "crohme2019" / "crohme2019" / "crohme2019" / "valid"
DEFAULT_OUTPUT = ROOT / "artifacts" / "replay_mid_audit_20260813"
CROHME_ALIASES = {r"\sqrt": r"\sqrt{}", r"\ldots": r"\dots", r"\lt": "<", r"\gt": ">"}


def _rows(path: Path) -> list[dict]:
    return list(_json_lines(path))


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _protocol_integrity(protocol: Path) -> dict:
    files = {
        "direct_protocol": protocol / "direct_protocol.json",
        "direct_ownership_truth": protocol / "direct_ownership_truth.jsonl",
        "direct_canonical_10_truth": protocol / "direct_canonical_10_truth.jsonl",
        "crohme_protocol": protocol / "crohme_protocol.json",
        "crohme_truth": protocol / "crohme_truth.jsonl",
        "current_data_10pct_truth": protocol / "current_data_10pct_truth.jsonl.gz",
    }
    return {
        "files": {name: {"bytes": path.stat().st_size, "sha256": _sha256(path)} for name, path in files.items()},
        "replay_html_counts": {
            "crohme": len(list((protocol / "crohme_replays").glob("*.html"))),
            "direct_ownership": len(list((protocol / "direct_ownership_replays").glob("*.html"))),
            "direct_canonical_10": len(list((protocol / "direct_canonical_10_replays").glob("*.html"))),
        },
    }


def _load_checkpoint(path: Path, device: torch.device) -> tuple[InkClassifierV1, list[str], dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    labels = list(checkpoint["math_labels"])
    if len(labels) != 372 or checkpoint.get("auxiliary_labels"):
        raise ValueError("expected the unified 372-class checkpoint")
    report = checkpoint.get("report", {})
    if report.get("input_contract", {}).get("math_observed_transform") != "force-observed-one":
        raise ValueError("unexpected checkpoint input contract")
    model = InkClassifierV1(len(labels))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device), labels, report


def _formula_rollup(rows: list[dict]) -> dict:
    return {
        "formulas": len(rows),
        "all_token_top1_hits": sum(row["all_tokens_top1"] for row in rows),
        "all_token_top1": sum(row["all_tokens_top1"] for row in rows) / len(rows) if rows else None,
        "all_token_top5_hits": sum(row["all_tokens_top5"] for row in rows),
        "all_token_top5": sum(row["all_tokens_top5"] for row in rows) / len(rows) if rows else None,
    }


def _direct(run: Path, output: Path) -> tuple[dict, list[dict], dict[str, dict]]:
    formulas = ROOT / "hf-dataset" / "data" / "formulas_valid.jsonl"
    ownership = ROOT / "hf-dataset" / "data" / "ownership_train.jsonl"
    annotations = [row for row in _json_lines(ownership) if row.get("accepted")]
    glyphs = list(iter_direct_ownership_examples(formulas, ownership))
    predictions = {row["record_id"]: row for row in _json_lines(run / "direct_ownership_predictions.jsonl")}
    glyph_truth = [{"record_id": row["record_id"], "label": row["label"]} for row in glyphs]
    glyph_score = score_glyphs_from_rows(glyph_truth, predictions)
    formula_rows: list[dict] = []
    by_sample: dict[str, dict] = {}
    offset = 0
    for annotation in annotations:
        count = len(annotation["labels"])
        group_rows = glyphs[offset:offset + count]
        offset += count
        truth = [str(value) for value in annotation["labels"]]
        if [row["label"] for row in group_rows] != truth:
            raise AssertionError(f"ownership order mismatch: {annotation['sample_id']}")
        topk = [predictions[row["record_id"]]["topk"] for row in group_rows]
        row = {
            "record_id": annotation["sample_id"],
            "symbols": count,
            "truth_tokens": truth,
            "top1_tokens": [values[0] for values in topk],
            "all_tokens_top1": all(values[0] == label for values, label in zip(topk, truth, strict=True)),
            "all_tokens_top5": all(label in values[:5] for values, label in zip(topk, truth, strict=True)),
            "contains_untrained_equals": "=" in truth,
        }
        formula_rows.append(row)
        by_sample[row["record_id"]] = row
    if offset != len(glyphs):
        raise AssertionError("ownership formula rollup did not consume all glyphs")
    _write_jsonl(output / "direct_ownership_formula_predictions.jsonl", formula_rows)
    return {
        "replay_formulas": len(annotations),
        "replay_html_files": len(list((DEFAULT_PROTOCOL / "direct_ownership_replays").glob("*.html"))),
        "glyph": glyph_score,
        "oracle_group_all_token": _formula_rollup(formula_rows),
        "formulas_containing_untrained_equals": sum(row["contains_untrained_equals"] for row in formula_rows),
        "interpretation_limit": "truth stroke ownership supplied; not end-to-end formula accuracy",
    }, formula_rows, by_sample


def _canonical_10(by_sample: dict[str, dict]) -> dict:
    truth = _rows(DEFAULT_PROTOCOL / "direct_canonical_10_truth.jsonl")
    available = [by_sample[row["record_id"]] for row in truth if row["record_id"] in by_sample]
    return {
        "replay_formulas": len(truth),
        "replay_html_files": len(list((DEFAULT_PROTOCOL / "direct_canonical_10_replays").glob("*.html"))),
        "fixed_ids": [row["record_id"] for row in truth],
        "ownership_available": len(available),
        "ownership_missing": len(truth) - len(available),
        "oracle_group_subset": _formula_rollup(available),
        "formula_exact": None,
        "status": "not_scorable_end_to_end",
        "reason": "four fixed formulas lack character ownership; all ten lack a current grouping/relation/formula decoder",
    }


def _crohme_sample(path: Path) -> dict | None:
    root = ET.parse(path).getroot()
    trace_ids: list[str] = []
    traces: dict[str, list[tuple[float, float, None]]] = {}
    for node in root.iter():
        if _local_name(node) != "trace":
            continue
        points = _trace_points(node.text)
        if points:
            trace_id = node.attrib.get("id", str(len(trace_ids))).strip()
            trace_ids.append(trace_id)
            traces[trace_id] = [(point[0], point[1], None) for point in points]
    index_by_id = {trace_id: index for index, trace_id in enumerate(trace_ids)}
    groups: list[list[int]] = []
    labels: list[str] = []
    for group in root.iter():
        if _local_name(group) != "traceGroup":
            continue
        annotation = next((child for child in group if _local_name(child) == "annotation" and child.attrib.get("type") == "truth" and child.text), None)
        refs = [child.attrib.get("traceDataRef", "").strip() for child in group if _local_name(child) == "traceView"]
        if annotation is None or not refs or any(ref not in index_by_id for ref in refs):
            continue
        labels.append(CROHME_ALIASES.get(annotation.text.strip().replace("$", ""), annotation.text.strip().replace("$", "")))
        groups.append([index_by_id[ref] for ref in refs])
    if not groups:
        return None
    return {
        "strokes": [traces[trace_id] for trace_id in trace_ids], "groups": groups, "labels": labels,
        "partition_complete": set().union(*(set(group) for group in groups)) == set(range(len(trace_ids))),
    }


def _crohme(model: InkClassifierV1, labels: list[str], device: torch.device, root: Path, output: Path) -> dict:
    protocol_rows, duplicates = load_crohme(root)
    vocabulary = set(labels)
    features: list[np.ndarray] = []
    truth_rows: list[dict] = []
    formula_rows: list[dict] = []
    unsupported = Counter()
    invalid_single_point = Counter()
    missing_truth_partition = 0
    incomplete_truth_partition = 0
    for formula in protocol_rows:
        sample = _crohme_sample(root / formula["record_id"])
        if sample is None:
            missing_truth_partition += 1
            formula_rows.append({
                "record_id": formula["record_id"], "supported_record_ids": [], "unsupported_labels": [],
                "invalid_single_point_labels": [], "truth_partition_complete": False,
            })
            continue
        incomplete_truth_partition += int(not sample["partition_complete"])
        formula_record = {
            "record_id": formula["record_id"], "supported_record_ids": [], "unsupported_labels": [],
            "invalid_single_point_labels": [], "truth_partition_complete": sample["partition_complete"],
        }
        for index, (group, label) in enumerate(zip(sample["groups"], sample["labels"], strict=True)):
            if label not in vocabulary:
                unsupported[label] += 1
                formula_record["unsupported_labels"].append(label)
                continue
            strokes = [sample["strokes"][stroke_index] for stroke_index in group]
            if sum(len(stroke) for stroke in strokes) < 2:
                invalid_single_point[label] += 1
                formula_record["invalid_single_point_labels"].append(label)
                continue
            record = _canonicalize(SourceSample(
                "crohme2019_valid", f"{formula['record_id']}:{index}", label,
                "noncommercial_evaluation", "truth_group_character_evaluation_only",
                strokes,
            ))
            features.append(tensorize(record))
            truth_rows.append({"record_id": record["record_id"], "label": label, "formula_id": formula["record_id"]})
            formula_record["supported_record_ids"].append(record["record_id"])
        formula_rows.append(formula_record)
    index = {label: position for position, label in enumerate(labels)}
    dataset = ArrayDataset(np.stack(features).astype(np.float32), np.asarray([index[row["label"]] for row in truth_rows], dtype=np.int64), "force-observed-one")
    predictions = predict(model, dataset, truth_rows, labels, "math", device, 512)
    glyph = score_glyphs_from_rows(truth_rows, predictions)
    formula_predictions: list[dict] = []
    truth_by_id = {row["record_id"]: row for row in truth_rows}
    for row in formula_rows:
        record_ids = row["supported_record_ids"]
        topk = [predictions[record_id]["topk"] for record_id in record_ids]
        truths = [truth_by_id[record_id]["label"] for record_id in record_ids]
        fully_supported = row["truth_partition_complete"] and not row["unsupported_labels"] and not row["invalid_single_point_labels"]
        formula_predictions.append({
            "record_id": row["record_id"],
            "fully_supported": fully_supported,
            "supported_symbols": len(record_ids),
            "unsupported_labels": row["unsupported_labels"],
            "invalid_single_point_labels": row["invalid_single_point_labels"],
            "truth_partition_complete": row["truth_partition_complete"],
            "all_tokens_top1": fully_supported and all(values[0] == label for values, label in zip(topk, truths, strict=True)),
            "all_tokens_top5": fully_supported and all(label in values[:5] for values, label in zip(topk, truths, strict=True)),
        })
    _write_jsonl(output / "crohme_truth_group_predictions.jsonl", [
        {"record_id": row["record_id"], "label": row["label"], **predictions[row["record_id"]]} for row in truth_rows
    ])
    _write_jsonl(output / "crohme_formula_coverage.jsonl", formula_predictions)
    fully_supported_rows = [row for row in formula_predictions if row["fully_supported"]]
    return {
        "protocol_formulas": len(protocol_rows),
        "raw_formulas": len(protocol_rows) + len(duplicates),
        "exact_duplicates_removed": len(duplicates),
        "replay_html_files": len(list((DEFAULT_PROTOCOL / "crohme_replays").glob("*.html"))),
        "truth_groups": len(truth_rows) + sum(unsupported.values()) + sum(invalid_single_point.values()),
        "supported_truth_groups": len(truth_rows),
        "unsupported_truth_groups": sum(unsupported.values()),
        "unsupported_labels": dict(unsupported.most_common()),
        "invalid_single_point_truth_groups": sum(invalid_single_point.values()),
        "invalid_single_point_labels": dict(invalid_single_point.most_common()),
        "missing_truth_partition_formulas": missing_truth_partition,
        "incomplete_truth_partition_formulas": incomplete_truth_partition,
        "fully_supported_formulas": len(fully_supported_rows),
        "glyph_on_supported_truth_groups": glyph,
        "oracle_group_all_token_all_985_strict": _formula_rollup(formula_predictions),
        "oracle_group_all_token_fully_supported_751": _formula_rollup(fully_supported_rows),
        "formula_exact": None,
        "interpretation_limit": "CROHME is noncommercial evaluation only; truth grouping supplied and relation/LaTeX decoding absent",
    }


def _external_10pct(run: Path) -> dict:
    truth = _rows(run / "cache" / "math_eval_truth.jsonl")
    predictions = {row["record_id"]: row for row in _json_lines(run / "external_holdout_predictions.jsonl")}
    old_path = DEFAULT_PROTOCOL / "current_data_10pct_truth.jsonl.gz"
    with gzip.open(old_path, "rt", encoding="utf-8") as stream:
        original = [json.loads(line) for line in stream if line.strip()]
    ids = {row["record_id"] for row in truth}
    original_ids = {row["record_id"] for row in original}
    if not ids <= original_ids or set(predictions) != ids:
        raise AssertionError("checkpoint holdout is not the fixed protocol subset")
    score = score_glyphs_from_rows(truth, predictions)
    hits1 = round(score["top1"] * len(truth))
    hits5 = round(score["top5"] * len(truth))
    support = Counter(row["label"] for row in truth)
    unsupported_rows = [row for row in original if row["record_id"] not in ids]
    return {
        "fixed_protocol_records": len(original),
        "unified_head_in_scope_records": len(truth),
        "output_vocabulary": 372,
        "labels_present": len(support),
        "missing_external_label": "=",
        "minimum_records_per_present_label": min(support.values()),
        "maximum_records_per_present_label": max(support.values()),
        "in_scope": score,
        "outside_unified_head_records": len(unsupported_rows),
        "outside_unified_head_by_source": dict(Counter(row["source"] for row in unsupported_rows)),
        "strict_original_protocol": {
            "records": len(original), "top1": hits1 / len(original), "top5": hits5 / len(original),
            "policy": "rows outside the 372-class output count as failures",
        },
        "fixed_subset_verified": True,
    }


def _fixed_holdout_collision_audit(run: Path, labels: list[str]) -> dict:
    """Hash exact model inputs across the fixed train/eval boundary."""
    cache = run / "cache"
    manifest = json.loads((cache / "cache_manifest.json").read_text(encoding="utf-8"))
    seen: dict[bytes, tuple[str, str] | list[tuple[str, str]]] = {}
    groups: dict[bytes, list[tuple[str, str]]] = {}
    train_hashes: dict[bytes, set[str]] = defaultdict(set)
    total = 0
    for split, key in (("train", "math_train"), ("eval", "math_eval")):
        entry = manifest["sets"][key]
        features = np.load(cache / entry["features"], mmap_mode="r")
        targets = np.load(cache / entry["labels"], mmap_mode="r")
        for start in range(0, len(features), 1024):
            batch = np.array(features[start:start + 1024], dtype=np.float32, copy=True)
            batch[:, :, 4] = 1.0
            for tensor, target in zip(batch, targets[start:start + len(batch)], strict=True):
                digest = hashlib.sha256(np.ascontiguousarray(tensor).tobytes()).digest()
                current = (split, labels[int(target)])
                if split == "train":
                    train_hashes[digest].add(labels[int(target)])
                previous = seen.get(digest)
                if previous is None:
                    seen[digest] = current
                elif isinstance(previous, tuple):
                    seen[digest] = [previous, current]
                    groups[digest] = seen[digest]
                else:
                    previous.append(current)
                total += 1
    cross_split_same_label = cross_split_cross_label = within_split_cross_label = 0
    for group in groups.values():
        by_split: dict[str, set[str]] = defaultdict(set)
        for split, label in group:
            by_split[split].add(label)
        all_labels = set().union(*by_split.values())
        cross_split = len(by_split) > 1
        cross_split_same_label += int(cross_split and bool(by_split.get("train", set()) & by_split.get("eval", set())))
        cross_split_cross_label += int(cross_split and len(all_labels) > 1)
        within_split_cross_label += int(any(len(values) > 1 for values in by_split.values()))
    evaluation = manifest["sets"]["math_eval"]
    eval_features = np.load(cache / evaluation["features"], mmap_mode="r")
    eval_targets = np.load(cache / evaluation["labels"], mmap_mode="r")
    affected = affected_top1 = affected_top5 = 0
    prediction_rows = {row["record_id"]: row["topk"] for row in _json_lines(run / "external_holdout_predictions.jsonl")}
    truth_rows = list(_json_lines(cache / evaluation["truth"]))
    for start in range(0, len(eval_features), 1024):
        batch = np.array(eval_features[start:start + 1024], dtype=np.float32, copy=True)
        batch[:, :, 4] = 1.0
        for local_index, (tensor, target) in enumerate(zip(batch, eval_targets[start:start + len(batch)], strict=True)):
            if hashlib.sha256(np.ascontiguousarray(tensor).tobytes()).digest() not in train_hashes:
                continue
            truth = truth_rows[start + local_index]
            topk = prediction_rows[truth["record_id"]]
            affected += 1
            affected_top1 += int(topk[0] == labels[int(target)])
            affected_top5 += int(labels[int(target)] in topk[:5])
    return {
        "input_mode": "force-observed-one",
        "records": total,
        "unique_tensors": total - sum(len(group) - 1 for group in groups.values()),
        "duplicate_tensor_groups": len(groups),
        "duplicate_rows_beyond_first": sum(len(group) - 1 for group in groups.values()),
        "cross_split_same_label_groups": cross_split_same_label,
        "cross_split_cross_label_groups": cross_split_cross_label,
        "within_split_cross_label_groups": within_split_cross_label,
        "eval_rows_with_exact_train_input": affected,
        "eval_rows_with_exact_train_input_top1": affected_top1 / affected if affected else None,
        "eval_rows_with_exact_train_input_top5": affected_top5 / affected if affected else None,
        "eval_rows_without_exact_train_input": len(eval_features) - affected,
        "decision": "rebuild the fixed split by exact input hash before final model claims",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--crohme", type=Path, default=DEFAULT_CROHME)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    checkpoint_path = args.run / "classifier_checkpoint.pt"
    model, labels, checkpoint_report = _load_checkpoint(checkpoint_path, device)
    direct, _formula_rows, by_sample = _direct(args.run, args.output)
    report = {
        "schema": "aiflow-replay-mid-audit/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "training_performed": False,
        "device": str(device),
        "checkpoint": {
            "path": str(checkpoint_path), "sha256": _sha256(checkpoint_path),
            "epochs": checkpoint_report.get("trained_epochs"), "output_classes": len(labels),
            "input_mode": checkpoint_report.get("input_contract", {}).get("observed_channel_mode"),
        },
        "protocol": {
            "point_step_contract": "one slider increment reveals exactly one source point; final-step prediction is scored",
            "integrity": _protocol_integrity(DEFAULT_PROTOCOL),
            "current_10pct": _external_10pct(args.run),
            "current_10pct_exact_input_collisions": _fixed_holdout_collision_audit(args.run, labels),
            "direct_ownership_47": direct,
            "direct_canonical_10": _canonical_10(by_sample),
            "crohme_valid_985": _crohme(model, labels, device, args.crohme, args.output),
        },
        "decision": "character metrics are valid; end-to-end formula accuracy remains blocked until grouping, relation, and formula decoding are connected",
    }
    report_path = args.output / "mid_audit_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
