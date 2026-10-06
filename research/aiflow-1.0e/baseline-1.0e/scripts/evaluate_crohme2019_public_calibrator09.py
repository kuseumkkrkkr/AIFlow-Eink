"""Evaluate the public-data calibrator on CROHME2019 without training on it."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from math_grid_drawer.aiflow_ocr05 import AIFlowOCR05
from math_grid_drawer.research.formula_gridder06 import FormulaGridder06
from scripts.audit_math_ink_06_case_context import _load_model06
from scripts.train_crohme_segmentation_lattice_selector import _samples
from scripts.train_normalized_balanced_touch_calibrator09 import (
    AdditiveObservedHead09, _baseline, _forward, _materialize_local, _public_formula_samples,
)


def _metrics(logits, truths, labels):
    index = {label: value for value, label in enumerate(labels)}
    truth = torch.tensor([index[value] for value in truths])
    top = logits.topk(5, dim=1).indices
    return {
        "symbols": len(truths),
        "top1": float(top[:, 0].eq(truth).float().mean()),
        "top5": float(top.eq(truth[:, None]).any(1).float().mean()),
    }, top[:, 0].eq(truth), top.eq(truth[:, None]).any(1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--crohme-root", type=Path, required=True)
    parser.add_argument("--gridder", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--calibrator", type=Path, required=True)
    parser.add_argument("--public-formulas", type=Path, required=True)
    parser.add_argument("--public-ownership", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    engine, online = _load_model06(args.base, args.adapter, device)
    checkpoint = torch.load(args.calibrator, map_location="cpu", weights_only=False)
    head = AdditiveObservedHead09(engine.model.exact_head.in_features, list(checkpoint["observed_existing"])).to(device)
    head.load_state_dict(checkpoint["state_dict"])
    labels = tuple(engine.labels) + ("=",)
    allowed = set(labels)
    training = _public_formula_samples(args.public_formulas, args.public_ownership)
    _x, training_labels, _f, _w = _materialize_local(training)
    support = Counter(training_labels)

    raw_samples = _samples(args.crohme_root, "median_height_32")
    gridder = FormulaGridder06.from_artifact(args.gridder)
    prepared, predicted_by_formula, truth_by_formula = [], {}, {}
    exact_partition = exact_groups = truth_groups = lattice_groups = 0
    fully_supported = set()
    for sample in raw_samples:
        sample_id = str(sample["sample_id"])
        strokes = sample["profiled_strokes"]
        truth = tuple(frozenset(group) for group in sample["truth_groups"])
        truth_labels = tuple(str(value) for value in sample["truth_labels"])
        predicted = tuple(frozenset(group) for group in gridder.grid(strokes).groups)
        candidates = {frozenset(row["source_indices"]) for row in AIFlowOCR05.build_segmentation_lattice(strokes)}
        truth_set, predicted_set = set(truth), set(predicted)
        exact_partition += int(truth_set == predicted_set)
        exact_groups += len(truth_set & predicted_set)
        truth_groups += len(truth)
        lattice_groups += len(truth_set & candidates)
        predicted_by_formula[sample_id] = predicted_set
        truth_by_formula[sample_id] = truth_set
        if all(label in allowed for label in truth_labels):
            fully_supported.add(sample_id)
        supported_pairs = [(group, label) for group, label in zip(truth, truth_labels, strict=True) if label in allowed]
        if supported_pairs:
            prepared.append({
                "sample_id": sample_id, "writer": str(sample["writer_key"]), "strokes": strokes,
                "truth_groups": [group for group, _ in supported_pairs],
                "truth_labels": [label for _, label in supported_pairs],
            })

    features, truths, formula_ids, _writers = _materialize_local(prepared)
    baseline = _baseline(engine, online, features, device)
    candidate = _forward(engine, online, head, features, device)
    base_probability = baseline.softmax(1)
    base_top = baseline.argmax(1)
    preserve = torch.tensor([
        support.get(labels[int(index)], 0) < 3 for index in base_top
    ]) & base_probability.max(1).values.ge(.95)
    guarded = torch.where(preserve[:, None], baseline, candidate)
    base_metrics, base_hit1, base_hit5 = _metrics(baseline, truths, labels)
    candidate_metrics, candidate_hit1, candidate_hit5 = _metrics(candidate, truths, labels)
    guard_metrics, guard_hit1, guard_hit5 = _metrics(guarded, truths, labels)

    def combined(hit1, hit5):
        by_formula1, by_formula5 = {}, {}
        matched1 = matched5 = 0
        for formula, truth_group, one, five in zip(
            formula_ids,
            [group for sample in prepared for group in sample["truth_groups"]],
            hit1.tolist(), hit5.tolist(), strict=True,
        ):
            present = truth_group in predicted_by_formula[formula]
            matched1 += int(present and one); matched5 += int(present and five)
            if formula in fully_supported:
                by_formula1[formula] = by_formula1.get(formula, True) and present and one
                by_formula5[formula] = by_formula5.get(formula, True) and present and five
        return {
            "group_token_top1_recall": matched1 / max(truth_groups, 1),
            "group_token_top5_recall": matched5 / max(truth_groups, 1),
            "formula_exact": sum(by_formula1.values()) / max(len(fully_supported), 1),
            "formula_top5_oracle": sum(by_formula5.values()) / max(len(fully_supported), 1),
        }

    report = {
        "experiment": "R-CROHME2019-PUBLIC-CALIBRATOR09-001",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "track": "R_noncommercial_only", "product_validation": False,
        "dataset": {
            "source": "Kaggle ntcuong2103/crohme2019 valid",
            "license": "CC BY-NC 4.0", "formulas": len(raw_samples),
            "fully_supported_formulas": len(fully_supported),
            "truth_groups": truth_groups, "supported_symbol_groups": len(truths),
        },
        "layout": {
            "partition_exact": exact_partition / max(len(raw_samples), 1),
            "exact_group_recall": exact_groups / max(truth_groups, 1),
            "lattice_group_ceiling": lattice_groups / max(truth_groups, 1),
        },
        "truth_group_classifier": {
            "baseline": base_metrics, "candidate": candidate_metrics, "guarded_candidate": guard_metrics,
            "guard_preserved_symbols": int(preserve.sum()),
            "guard_policy": "preserve baseline when its top1 confidence >=0.95 and public-train support <3",
        },
        "end_to_end": {
            "baseline": combined(base_hit1, base_hit5),
            "candidate": combined(candidate_hit1, candidate_hit5),
            "guarded_candidate": combined(guard_hit1, guard_hit5),
        },
        "limits": [
            "CROHME2019 is evaluation-only and noncommercial",
            "the 0.6 geometry gridder is evaluated because later noncommercial gridder artifacts were removed",
            "decision-layer semantic equivalence is not scored",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
