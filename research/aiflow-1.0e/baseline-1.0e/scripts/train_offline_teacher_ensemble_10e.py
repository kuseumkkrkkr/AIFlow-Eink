#!/usr/bin/env python3
"""Compare frozen OCR features and emit an OOF ensemble teacher distribution.

Every model is evaluated with the same outer writer-LOO split.  The joint
teacher may only reorder the existing HWR Top-k candidates.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from train_ocr_decision_adapter_10e import (
    OcrDecisionAdapter,
    _load_candidates,
    _make_examples,
    _predict,
    _set_seed,
    _train,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATES = ROOT / "artifacts" / "ocr_trocr_hwr95_candidates_20260901_r1" / "candidates.jsonl.gz"
SCHEMA = "aiflow-1.0e-offline-teacher-ensemble/v1"


def read_features(path: Path) -> dict[str, np.ndarray]:
    payload = np.load(path, allow_pickle=False)
    ids = [str(value) for value in payload["formula_ids"].tolist()]
    values = payload["features"].astype(np.float32)
    if len(ids) != len(values) or len(set(ids)) != len(ids):
        raise ValueError(f"invalid feature cache: {path}")
    return dict(zip(ids, values))


def metric(rows: list[dict], key: str) -> dict:
    by_formula: defaultdict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_formula[str(row["formula_id"])].append(row)
    correct = sum(str(row[key]) == str(row["label"]) for row in rows)
    exact = sum(all(str(row[key]) == str(row["label"]) for row in group) for group in by_formula.values())
    baseline_correct = sum(str(row["baseline_token"]) == str(row["label"]) for row in rows)
    return {
        "rows": len(rows),
        "top1_correct": correct,
        "top1": correct / len(rows),
        "formula_exact_correct": exact,
        "formula_total": len(by_formula),
        "formula_exact": exact / len(by_formula),
        "changed_rows": sum(str(row[key]) != str(row["baseline_token"]) for row in rows),
        "row_level_improvements": sum(str(row["baseline_token"]) != str(row["label"]) and str(row[key]) == str(row["label"]) for row in rows),
        "row_level_regressions": sum(str(row["baseline_token"]) == str(row["label"]) and str(row[key]) != str(row["label"]) for row in rows),
        "baseline_top1_correct": baseline_correct,
    }


def write_gz(path: Path, rows: list[dict]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def run_variant(
    name: str,
    rows: list[dict],
    features: dict[str, np.ndarray],
    labels: list[str],
    epochs: int,
    seed: int,
    device: torch.device,
) -> tuple[list[dict], dict, dict]:
    token_to_id = {token: index for index, token in enumerate(labels)}
    examples = _make_examples(rows, features, token_to_id)
    by_writer: defaultdict[str, list[dict]] = defaultdict(list)
    for example in examples:
        by_writer[str(example["writer_group"])].append(example)
    predictions: list[dict] = []
    folds: list[dict] = []
    for fold_index, writer in enumerate(sorted(by_writer)):
        train = [item for group, values in by_writer.items() if group != writer for item in values]
        held = list(by_writer[writer])
        model = OcrDecisionAdapter(
            int(examples[0]["numeric"].shape[-1]), len(labels), int(examples[0]["ocr"].shape[-1])
        ).to(device)
        _train(model, train, epochs, seed + fold_index, device)
        fold_rows = _predict(model, held, device)
        for row in fold_rows:
            row["teacher_variant"] = name
        predictions.extend(fold_rows)
        folds.append({"held_writer": writer, **metric(fold_rows, "adapter_token")})
        del model
    refit = OcrDecisionAdapter(
        int(examples[0]["numeric"].shape[-1]), len(labels), int(examples[0]["ocr"].shape[-1])
    ).to(device)
    _train(refit, examples, epochs, seed + 1000, device)
    checkpoint = {
        "variant": name,
        "labels": labels,
        "numeric_size": int(examples[0]["numeric"].shape[-1]),
        "ocr_size": int(examples[0]["ocr"].shape[-1]),
        "state_dict": refit.state_dict(),
        "candidate_contract": {"top_k_only": True, "token_creation": False, "stroke_regrouping": False},
    }
    return predictions, {"aggregate": metric(predictions, "adapter_token"), "writer_loo": folds}, checkpoint


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--texteller", type=Path, required=True)
    parser.add_argument("--unimernet", type=Path, required=True)
    parser.add_argument("--trocr-small", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    if not 1 <= args.epochs <= 20:
        parser.error("--epochs must be between 1 and 20")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    _set_seed(args.seed)
    device = torch.device(args.device)
    rows = _load_candidates(args.candidates, None)
    labels = sorted({str(token) for row in rows for token in row["final_topk"]})
    sources = {
        "texteller": read_features(args.texteller),
        "unimernet_tiny": read_features(args.unimernet),
        "microsoft_trocr_small": read_features(args.trocr_small),
    }
    required = {str(row["formula_id"]) for row in rows}
    for name, feature_map in sources.items():
        missing = required - set(feature_map)
        if missing:
            raise ValueError(f"{name} missing {len(missing)} formulas")
    joint = {
        formula_id: np.concatenate([sources[name][formula_id] for name in sorted(sources)]).astype(np.float32)
        for formula_id in sorted(required)
    }
    variants = {**sources, "ensemble_joint": joint}
    args.output.mkdir(parents=True)
    reports: dict[str, dict] = {}
    checkpoints: dict[str, dict] = {}
    predictions_by_variant: dict[str, list[dict]] = {}
    for index, (name, feature_map) in enumerate(variants.items()):
        print(json.dumps({"event": "variant_start", "variant": name, "feature_size": len(next(iter(feature_map.values())))}), flush=True)
        predictions, report, checkpoint = run_variant(
            name, rows, feature_map, labels, args.epochs, args.seed + index * 100, device
        )
        reports[name] = report
        checkpoints[name] = checkpoint
        predictions_by_variant[name] = predictions
        write_gz(args.output / f"{name}_writer_loo_predictions.jsonl.gz", predictions)
        print(json.dumps({"event": "variant_complete", "variant": name, **report["aggregate"]}), flush=True)
    source_names = ("texteller", "unimernet_tiny", "microsoft_trocr_small")
    indexed = {
        name: {str(row["record_id"]): row for row in predictions_by_variant[name]}
        for name in source_names
    }
    probability_mean_rows: list[dict] = []
    for base in predictions_by_variant["texteller"]:
        record_id = str(base["record_id"])
        probabilities = []
        for name in source_names:
            row = indexed[name][record_id]
            if [str(value) for value in row["candidates"]] != [str(value) for value in base["candidates"]]:
                raise ValueError(f"candidate mismatch in ensemble: {record_id}")
            scores = np.asarray(row["adapter_scores"], dtype=np.float64)
            scores -= scores.max()
            values = np.exp(scores)
            probabilities.append(values / values.sum())
        mean_probability = np.mean(probabilities, axis=0)
        selected = int(np.argmax(mean_probability))
        probability_mean_rows.append({
            **base,
            "teacher_variant": "ensemble_probability_mean",
            "adapter_token": str(base["candidates"][selected]),
            "adapter_scores": [float(np.log(max(value, 1e-12))) for value in mean_probability],
            "ensemble_members": list(source_names),
        })
    probability_report = {
        "aggregate": metric(probability_mean_rows, "adapter_token"),
        "writer_loo": [
            {"held_writer": writer, **metric([row for row in probability_mean_rows if str(row["writer_group"]) == writer], "adapter_token")}
            for writer in sorted({str(row["writer_group"]) for row in probability_mean_rows})
        ],
        "combination": "fixed uniform mean of per-model softmax probabilities; no label-tuned weights",
    }
    reports["ensemble_probability_mean"] = probability_report
    predictions_by_variant["ensemble_probability_mean"] = probability_mean_rows
    write_gz(args.output / "ensemble_probability_mean_writer_loo_predictions.jsonl.gz", probability_mean_rows)
    print(json.dumps({"event": "variant_complete", "variant": "ensemble_probability_mean", **probability_report["aggregate"]}), flush=True)
    torch.save({"schema": SCHEMA, "status": "shadow_only", "variants": checkpoints}, args.output / "teacher_ensemble_adapters.pt")
    baseline_rows = []
    with gzip.open(args.output / "ensemble_joint_writer_loo_predictions.jsonl.gz", "rt", encoding="utf-8") as stream:
        baseline_rows = [json.loads(line) for line in stream if line.strip()]
    report = {
        "schema": SCHEMA,
        "status": "shadow_only",
        "data": {"candidates": str(args.candidates), "rows": len(rows), "formulas": len(required), "writers": len({str(row['writer_group']) for row in rows})},
        "training": {"epochs": args.epochs, "seed": args.seed, "external_weights_frozen": True},
        "baseline": metric(baseline_rows, "baseline_token"),
        "candidate_recall": sum(str(row["label"]) in [str(value) for value in row["candidates"]] for row in baseline_rows) / len(baseline_rows),
        "variants": reports,
        "teacher_for_distillation": "ensemble_probability_mean_writer_loo_predictions.jsonl.gz",
        "candidate_contract": "existing HWR Top-k only; no candidate creation or stroke regrouping",
        "product_runtime_changed": False,
    }
    (args.output / "evaluation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"event": "complete", "output": str(args.output)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
