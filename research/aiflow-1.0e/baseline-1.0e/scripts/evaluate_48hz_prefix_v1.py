#!/usr/bin/env python3
"""Evaluate Math Ink character predictions after every 48 Hz replay point.

Direct project ink uses real-time interpolation and writer-LOO output heads.
CROHME has no timestamps, so each recorded point is exposed for one 48 Hz tick.
Every prefix is spatially normalized from only the points visible at that tick.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import torch
from torch import nn

from audit_replay_protocol_v1 import CROHME_ALIASES, DEFAULT_CROHME, _crohme_sample
from character_tensor_v1 import ROOT, _json_lines, iter_direct_ownership_examples, tensorize
from replay_evaluate_hwr_v1 import load_crohme, score_glyphs_from_rows
from train_character_classifier_v1 import InkClassifierV1, apply_input_mode


TICK_HZ = 48
TICK_MS = 1000.0 / TICK_HZ
INPUT_MODE = "uniform-time"
DEFAULT_BASE = ROOT / "artifacts" / "time_normalization_20260813" / "uniform_time_unified_math_8ep_full" / "classifier_checkpoint.pt"
DEFAULT_LOO = ROOT / "artifacts" / "unified_head_20260814" / "uniform_time_writer_loo_calibration" / "writer_loo_heads.pt"
DEFAULT_PRODUCT = ROOT / "artifacts" / "unified_head_20260814" / "uniform_time_final_all_writers" / "project_symbol_head_checkpoint.pt"
DEFAULT_OUTPUT = ROOT / "artifacts" / "prefix_48hz_20260814"
VISUAL_FAMILIES = {
    "vertical_stroke": frozenset({"1", "|", r"\mid", "I", "l", "i", r"\prime"}),
    "slash_stroke": frozenset({"/", r"\backslash", r"\setminus"}),
    "cross": frozenset({"x", "X", r"\times", r"\chi"}),
    "circle": frozenset({"0", "O", "o", r"\mathcal{O}"}),
    "sigma_sum": frozenset({r"\sum", r"\Sigma"}),
    "pi_product": frozenset({r"\prod", r"\Pi"}),
    "perpendicular": frozenset({r"\perp", r"\bot"}),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _d_path(path: Path, kind: str) -> Path:
    resolved = path.resolve()
    if resolved.drive.upper() != "D:":
        raise ValueError(f"{kind} must remain on D:: {resolved}")
    return resolved


def _load_model(path: Path, device: torch.device) -> tuple[InkClassifierV1, list[str], dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    labels = list(checkpoint.get("math_labels", []))
    if len(labels) != 372 or checkpoint.get("auxiliary_labels"):
        raise ValueError(f"expected a unified 372-class checkpoint: {path}")
    report = checkpoint.get("report", {})
    if report.get("input_contract", {}).get("observed_channel_mode") != INPUT_MODE:
        raise ValueError(f"expected {INPUT_MODE} input contract: {path}")
    model = InkClassifierV1(len(labels))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval(), labels, report


class PrefixPredictor:
    def __init__(self, model: InkClassifierV1, labels: list[str], device: torch.device, heads: dict[str, nn.Linear] | None = None) -> None:
        self.model, self.labels, self.device, self.heads = model, labels, device, heads

    @torch.inference_mode()
    def __call__(self, features: list[np.ndarray], writer_groups: list[str]) -> list[list[str]]:
        raw = np.stack(features).astype(np.float32, copy=False)
        points = torch.from_numpy(apply_input_mode(raw, INPUT_MODE)).to(self.device)
        embeddings = self.model.encode(points)
        if self.heads is None:
            logits = self.model.math_head(embeddings)
        else:
            logits = torch.empty((len(features), len(self.labels)), device=self.device)
            for writer in sorted(set(writer_groups)):
                if writer not in self.heads:
                    raise ValueError(f"missing writer-LOO head: {writer}")
                indices = torch.tensor([index for index, value in enumerate(writer_groups) if value == writer], device=self.device)
                logits[indices] = self.heads[writer](embeddings[indices])
        return [[self.labels[index] for index in row] for row in logits.topk(5, dim=1).indices.cpu().tolist()]


def _load_loo_heads(path: Path, base_path: Path, labels: list[str], device: torch.device) -> dict[str, nn.Linear]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != "aiflow-writer-loo-project-symbol-heads/v1":
        raise ValueError("unexpected writer-LOO head schema")
    if checkpoint.get("math_labels") != labels or checkpoint.get("input_mode") != INPUT_MODE:
        raise ValueError("writer-LOO head contract mismatch")
    if checkpoint.get("base_checkpoint_sha256") != _sha256(base_path):
        raise ValueError("writer-LOO heads do not belong to the supplied base checkpoint")
    heads: dict[str, nn.Linear] = {}
    for writer, state in checkpoint["heads_by_held_writer_group"].items():
        head = nn.Linear(128, len(labels)).to(device)
        head.load_state_dict(state, strict=True)
        heads[str(writer)] = head.eval()
    return heads


def _deduplicate_time(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return points
    keep = np.r_[np.flatnonzero(np.diff(points[:, 2]) > 0), len(points) - 1]
    return points[keep]


def resample_direct_48hz(record: dict) -> list[np.ndarray]:
    """Interpolate each pen-down stroke on a real 20.833 ms grid."""
    duration = record.get("transform", {}).get("duration_ms")
    if not isinstance(duration, (int, float)) or duration <= 0:
        raise ValueError(f"direct record lacks usable duration: {record['record_id']}")
    result: list[np.ndarray] = []
    for stroke in sorted(record["strokes"], key=lambda row: row["source_order"]):
        points = np.asarray(stroke["points"], dtype=np.float64)
        points[:, 2] *= float(duration)
        points = _deduplicate_time(points)
        if len(points) == 1:
            result.append(points)
            continue
        start, end = float(points[0, 2]), float(points[-1, 2])
        grid = np.arange(start, end, TICK_MS, dtype=np.float64)
        if not len(grid) or not math.isclose(float(grid[-1]), end, abs_tol=1e-8):
            grid = np.append(grid, end)
        result.append(np.column_stack((
            np.interp(grid, points[:, 2], points[:, 0]),
            np.interp(grid, points[:, 2], points[:, 1]),
            grid,
        )))
    if sum(len(stroke) for stroke in result) < 2:
        raise ValueError(f"48 Hz replay has fewer than two points: {record['record_id']}")
    return result


def _ordinal_48hz(strokes: list[list[tuple[float, float, None]]]) -> list[np.ndarray]:
    tick = 0
    result: list[np.ndarray] = []
    for stroke in strokes:
        rows = []
        for x, y, _ in stroke:
            rows.append((float(x), float(y), tick * TICK_MS))
            tick += 1
        result.append(np.asarray(rows, dtype=np.float64))
    return result


def _prefix_tensor(strokes: list[np.ndarray]) -> np.ndarray:
    flat = np.concatenate(strokes, axis=0)
    if len(flat) < 2:
        raise ValueError("prefix needs at least two points")
    left, top = flat[:, :2].min(axis=0)
    right, bottom = flat[:, :2].max(axis=0)
    extent = max(float(right - left), float(bottom - top))
    scale = 1.0 / extent if extent > 0 else 1.0
    pad_x = (1.0 - float(right - left) * scale) / 2.0
    pad_y = (1.0 - float(bottom - top) * scale) / 2.0
    first_time, last_time = float(flat[0, 2]), float(flat[-1, 2])
    duration = last_time - first_time
    output = []
    ordinal = 0
    denominator = max(len(flat) - 1, 1)
    for source_order, stroke in enumerate(strokes):
        points = []
        for x, y, time_ms in stroke:
            relative_time = (float(time_ms) - first_time) / duration if duration > 0 else ordinal / denominator
            points.append([
                round((float(x) - left) * scale + pad_x, 8),
                round((float(y) - top) * scale + pad_y, 8),
                round(relative_time, 8),
            ])
            ordinal += 1
        output.append({"source_order": source_order, "points": points})
    return tensorize({
        "source": "48hz_prefix",
        "normalization": {"source_time_available": True},
        "strokes": output,
    })


def _prefix_features(strokes: list[np.ndarray]) -> Iterator[tuple[int, np.ndarray]]:
    visible: list[np.ndarray] = []
    points_seen = 0
    for stroke in strokes:
        current: list[np.ndarray] = []
        visible.append(np.empty((0, 3), dtype=np.float64))
        for point in stroke:
            current.append(point)
            visible[-1] = np.asarray(current, dtype=np.float64)
            points_seen += 1
            if points_seen >= 2:
                yield points_seen, _prefix_tensor(visible)


def _visual_family_diagnostic(rows: list[dict]) -> dict:
    if sum(map(len, VISUAL_FAMILIES.values())) != len(set().union(*VISUAL_FAMILIES.values())):
        raise AssertionError("visual families overlap")
    result: dict[str, dict] = {}
    for name, family in VISUAL_FAMILIES.items():
        selected = [row for row in rows if row["label"] in family]
        if not selected:
            continue
        result[name] = {
            "labels": sorted(family),
            "records": len(selected),
            "exact_top1": sum(row["final_topk"][0] == row["label"] for row in selected) / len(selected),
            "family_top1": sum(row["final_topk"][0] in family for row in selected) / len(selected),
            "exact_top5": sum(row["label"] in row["final_topk"] for row in selected) / len(selected),
            "top1_family_ambiguous": sum(row["final_topk"][0] in family and any(value in family for value in row["final_topk"][1:]) for row in selected) / len(selected),
        }
    return {
        "scope": "diagnostic_only_not_relabeled_accuracy",
        "runtime_policy": "retain_372_class_top5_and_defer_same_family_candidates_to_formula_context",
        "families": result,
    }


def _summarize(rows: list[dict]) -> dict:
    truths = [{"record_id": row["record_id"], "label": row["label"], "source": row.get("source", "all")} for row in rows]
    final_predictions = {row["record_id"]: {"topk": row["final_topk"]} for row in rows}
    final = score_glyphs_from_rows(truths, final_predictions)
    prefix_count = top1_hits = top5_hits = tail_count = tail_top1 = tail_top5 = 0
    stable_ratios: list[float] = []
    for row in rows:
        correct = [value == row["label"] for value in row["top1_tokens"]]
        prefix_count += len(correct); top1_hits += sum(correct); top5_hits += sum(row["top5_hits"])
        tail = max(1, math.ceil(len(correct) * 0.2))
        tail_count += tail; tail_top1 += sum(correct[-tail:]); tail_top5 += sum(row["top5_hits"][-tail:])
        if correct[-1]:
            stable = len(correct) - 1
            for index in range(len(correct) - 2, -1, -1):
                if not correct[index]:
                    break
                stable = index
            stable_ratios.append((stable + 1) / len(correct))
    by_label: dict[str, dict] = {}
    for label in sorted({row["label"] for row in rows}):
        selected = [row for row in rows if row["label"] == label]
        by_label[label] = {
            "records": len(selected),
            "top1": sum(row["final_topk"][0] == label for row in selected) / len(selected),
            "top5": sum(label in row["final_topk"] for row in selected) / len(selected),
        }
    return {
        "final": final,
        "prefixes": prefix_count,
        "all_prefix_top1": top1_hits / prefix_count,
        "all_prefix_top5": top5_hits / prefix_count,
        "last_20pct_top1": tail_top1 / tail_count,
        "last_20pct_top5": tail_top5 / tail_count,
        "stable_correct_records": len(stable_ratios),
        "median_stable_correct_progress": float(np.median(stable_ratios)) if stable_ratios else None,
        "by_label": by_label,
        "visual_family_diagnostic": _visual_family_diagnostic(rows),
    }


def _evaluate(items: Iterable[dict], predictor: PrefixPredictor, output: Path, batch_size: int = 512) -> tuple[dict, list[dict]]:
    states: dict[str, dict] = {}
    features: list[np.ndarray] = []
    references: list[tuple[str, int, str]] = []

    def flush() -> None:
        if not features:
            return
        predictions = predictor(features, [writer for _, _, writer in references])
        for (record_id, point_count, _), topk in zip(references, predictions, strict=True):
            state = states[record_id]
            state["prefix_points"].append(point_count)
            state["top1_tokens"].append(topk[0])
            state["top5_hits"].append(state["label"] in topk)
            state["final_topk"] = topk
        features.clear(); references.clear()

    for item in items:
        record_id = str(item["record_id"])
        states[record_id] = {
            key: item[key] for key in ("record_id", "label", "source", "formula_id", "writer_group") if key in item
        } | {"prefix_points": [], "top1_tokens": [], "top5_hits": [], "final_topk": []}
        for point_count, feature in _prefix_features(item["strokes"]):
            features.append(feature)
            references.append((record_id, point_count, str(item.get("writer_group", ""))))
            if len(features) >= batch_size:
                flush()
    flush()
    rows = list(states.values())
    output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output, "wt", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return _summarize(rows), rows


def _direct_items() -> list[dict]:
    formulas = ROOT / "hf-dataset" / "data" / "formulas_valid.jsonl"
    ownership = ROOT / "hf-dataset" / "data" / "ownership_train.jsonl"
    annotations = [row for row in _json_lines(ownership) if row.get("accepted")]
    glyphs = list(iter_direct_ownership_examples(formulas, ownership))
    items: list[dict] = []
    offset = 0
    for annotation in annotations:
        for glyph in glyphs[offset:offset + len(annotation["labels"])]:
            items.append({
                "record_id": glyph["record_id"], "label": glyph["label"], "source": "project_owned",
                "formula_id": annotation["sample_id"], "writer_group": glyph["writer_group"],
                "strokes": resample_direct_48hz(glyph),
            })
        offset += len(annotation["labels"])
    if offset != len(glyphs):
        raise AssertionError("direct ownership order mismatch")
    return items


def _formula_rollup(
    rows: list[dict],
    formula_ids: set[str] | None = None,
    *,
    require_every_formula: bool = False,
    failed_formula_ids: set[str] | None = None,
) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if formula_ids is None or row["formula_id"] in formula_ids:
            grouped[row["formula_id"]].append(row)
    expected = set(grouped) if formula_ids is None or not require_every_formula else set(formula_ids)
    failed = failed_formula_ids or set()
    top1_hits = sum(
        formula_id not in failed and formula_id in grouped
        and all(row["final_topk"][0] == row["label"] for row in grouped[formula_id])
        for formula_id in expected
    )
    top5_hits = sum(
        formula_id not in failed and formula_id in grouped
        and all(row["label"] in row["final_topk"] for row in grouped[formula_id])
        for formula_id in expected
    )
    return {
        "formulas": len(expected),
        "all_token_top1_hits": top1_hits,
        "all_token_top1": top1_hits / len(expected) if expected else None,
        "all_token_top5_hits": top5_hits,
        "all_token_top5": top5_hits / len(expected) if expected else None,
    }


def _crohme_items(root: Path, labels: set[str]) -> tuple[list[dict], dict, set[str], set[str]]:
    formulas, duplicates = load_crohme(root)
    items: list[dict] = []
    unsupported: Counter[str] = Counter()
    invalid: Counter[str] = Counter()
    all_formula_ids = {str(formula["record_id"]) for formula in formulas}
    failed_formula_ids: set[str] = set()
    incomplete_truth_partition_formulas = 0
    for formula in formulas:
        formula_id = str(formula["record_id"])
        sample = _crohme_sample(root / formula["record_id"])
        if sample is None:
            failed_formula_ids.add(formula_id)
            continue
        if not sample["partition_complete"]:
            failed_formula_ids.add(formula_id)
            incomplete_truth_partition_formulas += 1
        for index, (group, raw_label) in enumerate(zip(sample["groups"], sample["labels"], strict=True)):
            label = CROHME_ALIASES.get(raw_label, raw_label)
            if label not in labels:
                unsupported[label] += 1
                failed_formula_ids.add(formula_id)
                continue
            strokes = [sample["strokes"][stroke_index] for stroke_index in group]
            if sum(len(stroke) for stroke in strokes) < 2:
                invalid[label] += 1
                failed_formula_ids.add(formula_id)
                continue
            items.append({
                "record_id": f"{formula_id}:{index}", "label": label, "source": "crohme2019_valid",
                "formula_id": formula_id, "strokes": _ordinal_48hz(strokes),
            })
    return items, {
        "protocol_formulas": len(formulas), "exact_formula_duplicates_removed": len(duplicates),
        "unsupported_truth_groups": sum(unsupported.values()), "unsupported_labels": dict(unsupported.most_common()),
        "invalid_single_point_truth_groups": sum(invalid.values()),
        "incomplete_truth_partition_formulas": incomplete_truth_partition_formulas,
        "fully_supported_formulas": len(all_formula_ids - failed_formula_ids),
    }, all_formula_ids, failed_formula_ids


def _self_test() -> None:
    record = {
        "record_id": "x", "transform": {"duration_ms": 100.0},
        "strokes": [{"source_order": 0, "points": [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]}],
    }
    replay = resample_direct_48hz(record)
    assert len(replay[0]) == 6 and replay[0][-1, 2] == 100.0
    prefixes = list(_prefix_features(replay))
    assert prefixes[0][0] == 2 and prefixes[-1][0] == 6
    # The first prefix is normalized to its own bbox, not the future full bbox.
    assert prefixes[0][1][:, 0].min() == 0.0 and prefixes[0][1][:, 0].max() == 1.0
    rows = [{"formula_id": "a", "label": "x", "final_topk": ["x"]}]
    strict = _formula_rollup(rows, {"a", "b"}, require_every_formula=True, failed_formula_ids={"b"})
    assert strict["formulas"] == 2 and strict["all_token_top1"] == 0.5


def _prediction_rows(path: Path) -> list[dict]:
    return list(_json_lines(path))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--loo-heads", type=Path, default=DEFAULT_LOO)
    parser.add_argument("--product-checkpoint", type=Path, default=DEFAULT_PRODUCT)
    parser.add_argument("--crohme", type=Path, default=DEFAULT_CROHME)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scope", choices=("direct", "crohme", "all"), default="all")
    parser.add_argument("--summarize-existing", action="store_true", help="reuse saved prefix predictions instead of running model inference")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test(); print(json.dumps({"self_test": "pass"})); return 0
    output = _d_path(args.output, "output")
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "prefix_48hz_report.json"
    if report_path.exists() and not args.summarize_existing:
        parser.error(f"refusing to overwrite report: {report_path}")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    report = {
        "schema": "aiflow-48hz-prefix-evaluation/v1", "generated_at": datetime.now(timezone.utc).isoformat(),
        "training_performed": False, "tick_hz": TICK_HZ, "tick_ms": TICK_MS,
        "input_contract": {"model_input_mode": INPUT_MODE, "future_bbox_used": False, "target_points": 128},
    }
    if args.scope in {"direct", "all"}:
        base_path = _d_path(args.base_checkpoint, "base checkpoint")
        prediction_path = output / "direct_prefix_predictions.jsonl.gz"
        if args.summarize_existing:
            rows = _prediction_rows(prediction_path)
            direct = _summarize(rows)
        else:
            model, labels, _ = _load_model(base_path, device)
            heads = _load_loo_heads(_d_path(args.loo_heads, "writer-LOO heads"), base_path, labels, device)
            direct, rows = _evaluate(_direct_items(), PrefixPredictor(model, labels, device, heads), prediction_path)
        canonical_ids = {row["record_id"] for row in _json_lines(ROOT / "artifacts" / "hwr_replay_evaluation_v1" / "direct_canonical_10_truth.jsonl")}
        direct["writer_disjoint"] = True
        direct["ownership_47_oracle_group"] = _formula_rollup(rows)
        direct["canonical_10"] = {
            "fixed_formulas": 10, "ownership_available": len({row["formula_id"] for row in rows} & canonical_ids),
            "oracle_group_subset": _formula_rollup(rows, canonical_ids),
            "status": "not_scorable_end_to_end", "reason": "four formulas lack character ownership and no grouping/relation decoder exists",
        }
        report["direct_ownership"] = direct
        report["direct_checkpoint"] = {"base_sha256": _sha256(base_path), "loo_heads_sha256": _sha256(args.loo_heads)}
    if args.scope in {"crohme", "all"}:
        product_path = _d_path(args.product_checkpoint, "product checkpoint")
        prediction_path = output / "crohme_prefix_predictions.jsonl.gz"
        model, labels, _ = _load_model(product_path, device)
        items, coverage, formula_ids, failed_formula_ids = _crohme_items(args.crohme, set(labels))
        if args.summarize_existing:
            rows = _prediction_rows(prediction_path)
            crohme = _summarize(rows)
        else:
            crohme, rows = _evaluate(items, PrefixPredictor(model, labels, device), prediction_path)
        crohme.update(coverage)
        fully_supported_ids = formula_ids - failed_formula_ids
        crohme["oracle_group_all_token_all_strict"] = _formula_rollup(
            rows, formula_ids, require_every_formula=True, failed_formula_ids=failed_formula_ids,
        )
        crohme["oracle_group_all_token_fully_supported"] = _formula_rollup(
            rows, fully_supported_ids, require_every_formula=True,
        )
        crohme["license_scope"] = "noncommercial_evaluation_only"
        report["crohme"] = crohme
        report["crohme_checkpoint"] = {"sha256": _sha256(product_path)}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"event": "prefix_48hz_complete", "output": str(output), "sections": sorted(set(report) & {"direct_ownership", "crohme"})}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
