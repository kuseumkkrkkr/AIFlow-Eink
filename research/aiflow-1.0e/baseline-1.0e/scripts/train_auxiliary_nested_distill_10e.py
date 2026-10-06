#!/usr/bin/env python3
"""Train with a disjoint auxiliary bank and evaluate only frozen writers."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

import torch

import train_online_candidate_distill_10e as base
from train_teacher_ensemble_raster_free_reranker_10e import teacher_heavy_loss
from train_ocr_decision_adapter_10e import _load_candidates
from accuracy_upgrade_contract_v1 import writer_key


SCHEMA = "aiflow-1.0e-auxiliary-nested-distill/v1"


def read_gz(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-candidates", type=Path, required=True)
    parser.add_argument("--auxiliary-candidates", type=Path, required=True)
    parser.add_argument("--combined-raw", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--hwr-checkpoint", type=Path, default=base.DEFAULT_HWR)
    parser.add_argument("--hwr-loo-checkpoint", type=Path, help="writer-LOO HWR state bundle")
    parser.add_argument("--strict-lineage", action="store_true")
    parser.add_argument("--input-mode", choices=("preserve", "uniform-time"), default="uniform-time")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")

    base._set_seed(args.seed)
    base._loss = teacher_heavy_loss
    device = torch.device(args.device)
    original = _load_candidates(args.original_candidates, None)
    auxiliary = _load_candidates(args.auxiliary_candidates, None)
    raw_rows = read_gz(args.combined_raw)
    teacher_rows = read_gz(args.teacher)
    raw_by_id = {str(row["record_id"]): row for row in raw_rows}
    teacher_by_id = {str(row["record_id"]): row for row in teacher_rows}
    all_candidates = original + auxiliary
    all_ids = {str(row["record_id"]) for row in all_candidates}
    if all_ids != set(raw_by_id) or all_ids != set(teacher_by_id):
        raise ValueError("combined raw/teacher coverage differs from candidates")

    original_writers = {writer_key(raw_by_id[str(row["record_id"])]) for row in original}
    auxiliary_writers = {writer_key(raw_by_id[str(row["record_id"])]) for row in auxiliary}
    original_formulas = {str(row["formula_id"]) for row in original}
    auxiliary_formulas = {str(row["formula_id"]) for row in auxiliary}
    if original_writers & auxiliary_writers or original_formulas & auxiliary_formulas:
        raise ValueError("auxiliary bank is not writer/formula-disjoint")

    if args.strict_lineage and args.hwr_loo_checkpoint is None:
        parser.error("--strict-lineage requires --hwr-loo-checkpoint")
    embeddings_by_writer = None
    if args.hwr_loo_checkpoint is not None:
        embeddings_by_writer, hwr_sha, _ = base._load_hwr_embeddings_by_held_writer(
            raw_by_id, args.hwr_loo_checkpoint, device, args.input_mode
        )
        embeddings = next(iter(embeddings_by_writer.values()))
    else:
        embeddings, hwr_sha, _ = base._load_hwr_embeddings(raw_by_id, args.hwr_checkpoint, device, args.input_mode)
    labels = sorted({str(token) for row in all_candidates for token in row["final_topk"]})
    token_to_id = {token: index for index, token in enumerate(labels)}
    original_samples = base._make_samples(original, raw_by_id, teacher_by_id, embeddings, token_to_id)
    auxiliary_samples = base._make_samples(auxiliary, raw_by_id, teacher_by_id, embeddings, token_to_id)
    by_writer: defaultdict[str, list[dict]] = defaultdict(list)
    for sample in original_samples:
        by_writer[str(sample["writer_group"])].append(sample)

    predictions: list[dict] = []
    folds: list[dict] = []
    losses: list[dict] = []
    numeric_size = int(original_samples[0]["numeric"].shape[-1])
    for fold_index, held_writer in enumerate(sorted(by_writer)):
        train = auxiliary_samples + [
            sample for writer, rows in by_writer.items() if writer != held_writer for sample in rows
        ]
        held = list(by_writer[held_writer])
        if embeddings_by_writer is not None:
            if held_writer not in embeddings_by_writer:
                raise ValueError(f"HWR LOO checkpoint has no state for held writer {held_writer}")
            fold_embeddings = embeddings_by_writer[held_writer]
            train = [{**sample, "ink": fold_embeddings[sample["record_id"]]} for sample in train]
            held = [{**sample, "ink": fold_embeddings[sample["record_id"]]} for sample in held]
        if args.strict_lineage:
            base._assert_nested_teacher_exclusion(train, held_writer)
        model = base.OnlineCandidateRanker(numeric_size, len(labels)).to(device)
        loss = base._train(
            model, train, args.epochs, args.seed + fold_index, device,
            args.learning_rate, args.temperature,
        )
        loss["held_writer"] = held_writer
        losses.append(loss)
        fold_rows = base._predict(model, held, device, excluded_writers=[held_writer])
        predictions.extend(fold_rows)
        folds.append({
            "held_writer": held_writer,
            "train_original_writers": sorted(original_writers - {held_writer}),
            "train_auxiliary_writers": sorted(auxiliary_writers),
            **base._metrics(fold_rows, "adapter_token"),
        })
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    refit = base.OnlineCandidateRanker(numeric_size, len(labels)).to(device)
    refit_loss = base._train(
        refit, original_samples + auxiliary_samples, args.epochs, args.seed + 1000,
        device, args.learning_rate, args.temperature,
    )
    args.output.mkdir(parents=True)
    checkpoint = args.output / "online_candidate_ranker.pt"
    torch.save({
        "schema": SCHEMA,
        "status": "shadow_only",
        "labels": labels,
        "numeric_size": numeric_size,
        "hwr_checkpoint_sha256": hwr_sha,
        "state_dict": refit.state_dict(),
        "runtime_inputs": ["5-channel ordered online ink", "frozen HWR Top-k", "geometry/context", "candidate token identity"],
        "external_teacher_in_runtime": False,
        "candidate_contract": {"top_k_only": True, "token_creation": False, "stroke_regrouping": False},
    }, checkpoint)

    baseline = base._metrics(predictions, "baseline_token")
    adapter = base._metrics(predictions, "adapter_token")
    top5 = sum(bool(row["target_in_candidates"]) for row in predictions) / len(predictions)
    baseline_top5 = sum(bool(row["target_in_candidates"]) for row in predictions) / len(predictions)
    candidate_contract_violations = sum(
        str(row["adapter_token"]) not in {str(value) for value in row["candidates"]}
        for row in predictions
    )
    report = {
        "schema": SCHEMA,
        "status": "shadow_only",
        "outer_evaluation": "frozen original seven-writer 95-formula partition only",
        "training_boundary": "six non-held original writers plus two disjoint auxiliary writers per fold",
        "data": {
            "original": {"rows": len(original_samples), "formulas": len(original_formulas), "writers": len(original_writers)},
            "auxiliary": {"rows": len(auxiliary_samples), "formulas": len(auxiliary_formulas), "writers": len(auxiliary_writers)},
        },
        "training": {
            "epochs": args.epochs, "seed": args.seed, "learning_rate": args.learning_rate,
            "temperature": args.temperature,
            "loss": "0.20 CE + 0.70 teacher KL + 0.10 baseline stability + 0.10 pairwise margin",
            "refit_loss": refit_loss,
        },
        "baseline": baseline,
        "adapter": adapter,
        "top5_candidate_recall": top5,
            "top5_non_regression": top5 >= baseline_top5,
        "writer_loo": folds,
        "loss_reports": losses,
        "gates": {
            "top1_non_regression": adapter["top1"] >= baseline["top1"],
            "formula_exact_non_regression": adapter["formula_exact"] >= baseline["formula_exact"],
            "row_regressions_zero": adapter["row_level_regressions"] == 0,
            "held_writer_row_regressions_zero": all(fold["row_level_regressions"] == 0 for fold in folds),
            "candidate_contract_violations_zero": candidate_contract_violations == 0,
        },
        "external_teacher_in_runtime": False,
        "hwr_embedding_scope": "writer_loo" if embeddings_by_writer is not None else "legacy_all_writer_final",
        "strict_lineage": bool(args.strict_lineage),
        "product_runtime_changed": False,
        "checkpoint": str(checkpoint),
    }
    (args.output / "evaluation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    base._write_gz(args.output / "writer_loo_predictions.jsonl.gz", predictions)
    print(json.dumps({"event": "complete", "baseline": baseline, "adapter": adapter, "gates": report["gates"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
