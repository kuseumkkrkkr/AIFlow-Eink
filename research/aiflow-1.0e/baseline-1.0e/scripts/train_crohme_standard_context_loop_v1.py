#!/usr/bin/env python3
"""RETIRED: historical CROHME-trained context experiment.

AIFlow Math Ink 1.0 now enforces CROHME as validation-only.  The historical
implementation remains readable for audit reproduction, but its command-line
entrypoint fails closed and cannot perform gradient updates.
"""

from __future__ import annotations

if __name__ == "__main__":
    raise SystemExit(
        "retired: CROHME/MathWriting are validation-only; this command cannot train"
    )

import argparse
import copy
import gzip
import hashlib
import io
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch

from character_tensor_v1 import ROOT, _json_lines
from evaluate_homograph_context_reranker_v1 import _metrics
import train_candidate_validity_context_v1 as candidate
import train_independent_formula_context_v1 as independent
import train_masked_context_reranker_v1 as masked


SCHEMA = "aiflow-crohme-standard-context-research/v1"
REPORT_SCHEMA = "aiflow-crohme-standard-context-selection/v1"
EXPECTED_VALID_BASELINE = {
    "all_top1": 0.7745149449396959,
    "formula_exact": 0.21747967479674796,
    "strict_macro_top1": 0.5542332669755218,
}
LAMBDA_GRID = (0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0)
MAX_SEQUENCE_GLYPHS = 255

DEFAULT_BASE_CHECKPOINT = (
    ROOT / "artifacts" / "candidate_validity_context_20260820_r1_shadow"
    / "candidate_validity_accuracy.pt"
)
DEFAULT_TRAIN = (
    ROOT / "artifacts" / "crohme_standard_context_20260820_r1_research"
    / "train_candidates.jsonl.gz"
)
DEFAULT_VALID = (
    ROOT / "artifacts" / "homograph_context_expanded_hwr_20260820_r1_shadow"
    / "crohme_candidates.jsonl.gz"
)
DEFAULT_OUTPUT = (
    ROOT / "artifacts" / "crohme_standard_context_20260820_r1_research"
)


def _event(name: str, **values: object) -> None:
    print(json.dumps({"event": name, **values}, ensure_ascii=False), flush=True)


def _d_path(path: Path, label: str, *, file: bool = True) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.drive.upper() != "D:":
        raise ValueError(f"{label} must remain on D: {resolved}")
    if file and not resolved.is_file():
        raise FileNotFoundError(f"missing {label}: {resolved}")
    return resolved


def _hash_bucket(value: str, modulus: int, salt: str) -> int:
    payload = f"{salt}\0{value}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16) % modulus


def _sequence_key(sequence: list[dict]) -> str:
    return "\0".join(str(row["label"]) for row in sequence)


def _effective_writer(formula_id: str, sequence: list[dict]) -> tuple[str | None, str]:
    writers = {str(row.get("writer_group", "")) for row in sequence}
    if len(writers) != 1 or "" in writers:
        return None, "invalid_annotation"
    writer = next(iter(writers))
    if not writer.startswith("missing-writer:"):
        return writer, "annotation"
    match = re.match(r"^(\d+-\d+)-\d+\.inkml$", formula_id)
    if match:
        return f"inferred-session:{match.group(1)}", "filename_session"
    return None, "unresolved"


def _eligible_formulae(rows: list[dict]) -> tuple[dict[str, list[dict]], list[str]]:
    formulae = candidate._formula_rows(rows)
    too_long = [
        formula_id
        for formula_id, sequence in formulae.items()
        if len(sequence) > MAX_SEQUENCE_GLYPHS
    ]
    return {
        formula_id: sequence
        for formula_id, sequence in formulae.items()
        if formula_id not in set(too_long)
    }, sorted(too_long)


def _double_holdout(rows: list[dict]) -> tuple[list[dict], list[dict], dict]:
    formulae, too_long = _eligible_formulae(rows)
    writer_by_formula = {}
    sequence_by_formula = {}
    writer_sources = Counter()
    unresolved_writer_formula_ids = []
    for formula_id, sequence in formulae.items():
        writer, source = _effective_writer(formula_id, sequence)
        writer_sources[source] += 1
        if writer is None:
            unresolved_writer_formula_ids.append(formula_id)
            continue
        writer_by_formula[formula_id] = writer
        sequence_by_formula[formula_id] = _sequence_key(sequence)
    formulae = {
        formula_id: sequence
        for formula_id, sequence in formulae.items()
        if formula_id in writer_by_formula
    }

    chosen = None
    for modulus in (5, 4, 3):
        held_writers = {
            writer
            for writer in set(writer_by_formula.values())
            if _hash_bucket(writer, modulus, "crohme-held-writer") == 0
        }
        held_sequences = {
            sequence
            for sequence in set(sequence_by_formula.values())
            if _hash_bucket(sequence, modulus, "crohme-held-sequence") == 0
        }
        dev_ids = {
            formula_id
            for formula_id in formulae
            if writer_by_formula[formula_id] in held_writers
            and sequence_by_formula[formula_id] in held_sequences
        }
        fit_ids = {
            formula_id
            for formula_id in formulae
            if writer_by_formula[formula_id] not in held_writers
            and sequence_by_formula[formula_id] not in held_sequences
        }
        if len(dev_ids) >= 100 and len(fit_ids) >= 1000:
            chosen = modulus, held_writers, held_sequences, fit_ids, dev_ids
            break
    if chosen is None:
        raise ValueError("unable to construct a useful writer-and-sequence holdout")

    modulus, held_writers, held_sequences, fit_ids, dev_ids = chosen
    fit = [row for row in rows if str(row["formula_id"]) in fit_ids]
    dev = [row for row in rows if str(row["formula_id"]) in dev_ids]
    fit_writers = {writer_by_formula[formula_id] for formula_id in fit_ids}
    dev_writers = {writer_by_formula[formula_id] for formula_id in dev_ids}
    fit_sequences = {
        _sequence_key(sequence)
        for formula_id, sequence in formulae.items()
        if formula_id in fit_ids
    }
    dev_sequences = {
        _sequence_key(sequence)
        for formula_id, sequence in formulae.items()
        if formula_id in dev_ids
    }
    if fit_writers & dev_writers or fit_sequences & dev_sequences:
        raise AssertionError("CROHME fit/dev writer or sequence leakage")
    if {str(row["record_id"]) for row in fit} & {
        str(row["record_id"]) for row in dev
    }:
        raise AssertionError("CROHME fit/dev record leakage")
    audit = {
        "method": "double holdout: dev requires held writer AND held exact token sequence",
        "hash_modulus": modulus,
        "all_eligible_formulas": len(formulae),
        "all_eligible_records": sum(len(value) for value in formulae.values()),
        "fit_formulas": len(fit_ids),
        "fit_records": len(fit),
        "fit_writers": len(fit_writers),
        "dev_formulas": len(dev_ids),
        "dev_records": len(dev),
        "dev_writers": len(dev_writers),
        "held_writer_pool": len(held_writers),
        "held_sequence_pool": len(held_sequences),
        "discarded_cross_quadrant_formulas": len(formulae) - len(fit_ids) - len(dev_ids),
        "too_long_formulas_excluded": len(too_long),
        "too_long_formula_ids": too_long,
        "writer_source_counts": dict(sorted(writer_sources.items())),
        "unresolved_writer_formulas_excluded": len(unresolved_writer_formula_ids),
        "unresolved_writer_formula_ids": sorted(unresolved_writer_formula_ids),
        "writer_overlap": 0,
        "exact_token_sequence_overlap": 0,
        "record_overlap": 0,
    }
    return fit, dev, audit


def _metric_view(metrics: dict) -> dict:
    return {
        key: metrics[key]
        for key in (
            "all_records",
            "all_top1",
            "strict_records",
            "strict_micro_top1",
            "strict_macro_top1",
            "changed",
            "improved",
            "regressed",
            "formula_exact",
            "baseline_formula_exact",
        )
    }


def _paired(
    rows: list[dict], before: dict[str, str], after: dict[str, str]
) -> dict:
    improved = regressed = changed = 0
    by_formula: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        record_id = str(row["record_id"])
        truth = str(row["label"])
        old = str(before[record_id])
        new = str(after[record_id])
        changed += old != new
        improved += old != truth and new == truth
        regressed += old == truth and new != truth
        by_formula[str(row["formula_id"])].append(row)
    formula_improved = formula_regressed = 0
    for sequence in by_formula.values():
        old_exact = all(
            before[str(row["record_id"])] == str(row["label"]) for row in sequence
        )
        new_exact = all(
            after[str(row["record_id"])] == str(row["label"]) for row in sequence
        )
        formula_improved += not old_exact and new_exact
        formula_regressed += old_exact and not new_exact
    return {
        "changed": changed,
        "improved": improved,
        "regressed": regressed,
        "net_improvement": improved - regressed,
        "formula_improved": formula_improved,
        "formula_regressed": formula_regressed,
    }


def _oracle(rows: list[dict]) -> dict:
    formulae = candidate._formula_rows(rows)
    top5_hits = sum(str(row["label"]) in row["final_topk"] for row in rows)
    formula_hits = sum(
        all(str(row["label"]) in row["final_topk"] for row in sequence)
        for sequence in formulae.values()
    )
    return {
        "records": len(rows),
        "top5_count": top5_hits,
        "top5": top5_hits / len(rows),
        "formula_top5_oracle_count": formula_hits,
        "formula_top5_oracle": formula_hits / len(formulae),
    }


def _admissible(metrics: dict, baseline: dict) -> tuple[bool, bool]:
    keys = ("all_top1", "formula_exact", "strict_macro_top1")
    nonregression = all(float(metrics[key]) + 1e-12 >= float(baseline[key]) for key in keys)
    improvement = any(float(metrics[key]) > float(baseline[key]) + 1e-12 for key in keys)
    return nonregression, improvement


def _trial_key(trial: dict) -> tuple:
    metrics = trial["metrics"]
    paired = trial["paired_vs_base"]
    return (
        float(metrics["formula_exact"]),
        float(metrics["all_top1"]),
        float(metrics["strict_macro_top1"]),
        float(metrics["strict_micro_top1"]),
        int(paired["net_improvement"]),
        -int(paired["regressed"]),
        -int(trial["epoch"]),
        -float(trial["lambda"]),
    )


def _state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _predict_state(
    state: dict[str, torch.Tensor], lambda_value: float, rows: list[dict],
    pretrained: Path, labels: list[str], device: torch.device, batch_size: int,
) -> tuple[dict[str, str], dict]:
    model, contract = candidate._new_model(
        pretrained, labels, device, candidate.SEED
    )
    model.load_state_dict(state, strict=True)
    validity = candidate._score_candidates(model, contract, rows, device, batch_size)
    predictions = candidate._fused_predictions(rows, validity, lambda_value)
    audit = masked._candidate_audit(rows, predictions)
    del model, contract, validity
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return predictions, audit


def _write_predictions(
    path: Path, rows: list[dict], baseline: dict[str, str], selected: dict[str, str]
) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(
            filename="", fileobj=raw, mode="wb", compresslevel=6, mtime=0
        ) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8", newline="\n") as stream:
                for row in rows:
                    record_id = str(row["record_id"])
                    stream.write(json.dumps({
                        "record_id": record_id,
                        "formula_id": str(row["formula_id"]),
                        "writer_group": str(row.get("writer_group", "")),
                        "label": str(row["label"]),
                        "hwr_top1": str(row["final_topk"][0]),
                        "base_context_top1": str(baseline[record_id]),
                        "selected_context_top1": str(selected[record_id]),
                        "candidate_preserved": selected[record_id] in row["final_topk"],
                    }, ensure_ascii=False, separators=(",", ":")) + "\n")


def _save_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-candidates", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--valid-candidates", type=Path, default=DEFAULT_VALID)
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE_CHECKPOINT)
    parser.add_argument("--hwr-checkpoint", type=Path, default=independent.DEFAULT_HWR)
    parser.add_argument("--pretrained", type=Path, default=masked.DEFAULT_PRETRAINED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--predict-batch-size", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    parser.error(
        "retired: CROHME/MathWriting are validation-only; this command cannot train"
    )
    if args.epochs != 3:
        parser.error("this protocol fixes exactly three epoch-snapshot variants")
    if (
        args.learning_rate <= 0.0
        or args.weight_decay < 0.0
        or min(args.batch_size, args.predict_batch_size) < 1
    ):
        parser.error("invalid optimizer or batch configuration")

    train_path = _d_path(args.train_candidates, "CROHME train candidates")
    valid_path = _d_path(args.valid_candidates, "CROHME valid candidates")
    base_checkpoint = _d_path(args.base_checkpoint, "base context checkpoint")
    hwr_checkpoint = _d_path(args.hwr_checkpoint, "frozen HWR checkpoint")
    pretrained = _d_path(args.pretrained, "pinned BERT-Tiny", file=False)
    output = _d_path(args.output, "CROHME research output", file=False)
    if output.exists():
        existing = [
            output / "baseline_freeze.json",
            output / "selection_report.json",
            output / "crohme_standard_context_research.pt",
        ]
        if any(path.exists() for path in existing):
            parser.error(f"refusing to overwrite CROHME loop evidence: {output}")
    output.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")

    train_rows = list(_json_lines(train_path))
    valid_rows = list(_json_lines(valid_path))
    if not train_rows or not valid_rows:
        raise ValueError("empty CROHME candidate cache")
    if {str(row.get("dataset_split")) for row in train_rows} != {"train"}:
        raise ValueError("train cache is not the audited CROHME train split")
    if any("test" in str(row.get("source", "")) for row in train_rows + valid_rows):
        raise ValueError("official test rows are forbidden during model selection")

    fit_rows, dev_rows, split_audit = _double_holdout(train_rows)
    model, contract, base_payload = candidate.load_candidate_validity_context(
        pretrained, base_checkpoint, hwr_checkpoint, device
    )
    labels = list(base_payload["math_labels"])
    base_dev, base_dev_audit = candidate.decide_candidate_validity_rows(
        model, contract, base_payload, dev_rows, device, args.predict_batch_size
    )
    base_valid, base_valid_audit = candidate.decide_candidate_validity_rows(
        model, contract, base_payload, valid_rows, device, args.predict_batch_size
    )
    base_dev_metrics = _metrics(dev_rows, base_dev)
    base_valid_metrics = _metrics(valid_rows, base_valid)
    for key, expected in EXPECTED_VALID_BASELINE.items():
        if not math.isclose(
            float(base_valid_metrics[key]), expected, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                f"frozen CROHME valid baseline drift for {key}: "
                f"{base_valid_metrics[key]} != {expected}"
            )
    base_state = _state(model)
    for parameter in model.parameters():
        parameter.requires_grad_(True)

    baseline_freeze = {
        "schema": "aiflow-crohme-standard-baseline-freeze/v1",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "selection_protocol": {
            "official_test_loaded": False,
            "official_test_metrics_loaded": False,
            "selection_uses_train_derived_double_holdout_only": True,
            "valid_used_for_selection": False,
            "maximum_variants": 3,
        },
        "rights": {
            "CROHME": "CC BY-NC research-only",
            "product_training_eligible": False,
        },
        "split_audit": split_audit,
        "baseline": {
            "dev": _metric_view(base_dev_metrics),
            "valid_reproduction": _metric_view(base_valid_metrics),
            "dev_top5_oracle": _oracle(dev_rows),
            "valid_top5_oracle": _oracle(valid_rows),
            "candidate_audit": {
                "dev": base_dev_audit,
                "valid": base_valid_audit,
            },
        },
        "inputs": {
            "train_candidates": str(train_path),
            "train_candidates_sha256": masked._sha256(train_path),
            "valid_candidates": str(valid_path),
            "valid_candidates_sha256": masked._sha256(valid_path),
            "base_checkpoint": str(base_checkpoint),
            "base_checkpoint_sha256": masked._sha256(base_checkpoint),
            "hwr_checkpoint": str(hwr_checkpoint),
            "hwr_checkpoint_sha256": masked._sha256(hwr_checkpoint),
            "pretrained_hashes": masked._verify_pretrained(pretrained),
        },
    }
    freeze_path = output / "baseline_freeze.json"
    _save_json(freeze_path, baseline_freeze)
    freeze_sha = masked._sha256(freeze_path)
    _event(
        "crohme_baseline_frozen",
        dev_records=len(dev_rows),
        dev_formulas=len(candidate._formula_rows(dev_rows)),
        valid_top1=base_valid_metrics["all_top1"],
        valid_formula_exact=base_valid_metrics["formula_exact"],
        valid_strict_macro=base_valid_metrics["strict_macro_top1"],
        freeze_sha256=freeze_sha,
    )

    examples = candidate._direct_examples(fit_rows, labels, "crohme-standard-r1")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    variants = []
    states = {}
    for epoch in range(1, args.epochs + 1):
        loss = candidate._train_one_epoch(
            model,
            contract,
            examples,
            optimizer,
            device,
            args.batch_size,
            candidate._seed(f"crohme-standard-epoch-{epoch}"),
        )
        states[epoch] = _state(model)
        validity = candidate._score_candidates(
            model, contract, dev_rows, device, args.predict_batch_size
        )
        trials = []
        for lambda_value in LAMBDA_GRID:
            predictions = candidate._fused_predictions(
                dev_rows, validity, lambda_value
            )
            metrics = _metrics(dev_rows, predictions)
            nonregression, improvement = _admissible(metrics, base_dev_metrics)
            trials.append({
                "epoch": epoch,
                "lambda": lambda_value,
                "metrics": _metric_view(metrics),
                "paired_vs_base": _paired(dev_rows, base_dev, predictions),
                "nonregression": nonregression,
                "improvement": improvement,
                "eligible": nonregression and improvement,
            })
        eligible = [trial for trial in trials if trial["eligible"]]
        best = max(eligible, key=_trial_key) if eligible else max(trials, key=_trial_key)
        variants.append({
            "variant": f"epoch_{epoch}",
            "epoch": epoch,
            "train_loss": loss,
            "best_trial": best,
            "eligible_trials": len(eligible),
            "trials": trials,
        })
        _event(
            "crohme_context_variant",
            variant=f"epoch_{epoch}",
            train_loss=loss,
            selected_lambda=best["lambda"],
            dev_top1=best["metrics"]["all_top1"],
            dev_formula_exact=best["metrics"]["formula_exact"],
            dev_strict_macro=best["metrics"]["strict_macro_top1"],
            eligible=best["eligible"],
        )
        del validity

    eligible_trials = [
        trial
        for variant in variants
        for trial in variant["trials"]
        if trial["eligible"]
    ]
    if eligible_trials:
        selected = max(eligible_trials, key=_trial_key)
        selected_kind = "crohme_research_finetune"
        selected_state = states[int(selected["epoch"])]
    else:
        selected_kind = "no_op_parent"
        selected_state = base_state
        selected = {
            "epoch": 0,
            "lambda": float(base_payload["configuration"]["lambda"]),
            "metrics": _metric_view(base_dev_metrics),
            "paired_vs_base": _paired(dev_rows, base_dev, base_dev),
            "nonregression": True,
            "improvement": False,
            "eligible": False,
        }

    selected_dev, selected_dev_audit = _predict_state(
        selected_state,
        float(selected["lambda"]),
        dev_rows,
        pretrained,
        labels,
        device,
        args.predict_batch_size,
    )
    selected_valid, selected_valid_audit = _predict_state(
        selected_state,
        float(selected["lambda"]),
        valid_rows,
        pretrained,
        labels,
        device,
        args.predict_batch_size,
    )
    selected_dev_metrics = _metrics(dev_rows, selected_dev)
    selected_valid_metrics = _metrics(valid_rows, selected_valid)

    checkpoint_payload = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_family": candidate.MODEL_FAMILY,
        "model_id": masked.MODEL_ID,
        "model_revision": masked.MODEL_REVISION,
        "math_labels": labels,
        "class_tokens": contract["class_tokens"],
        "relation_tokens": contract["relation_tokens"],
        "pretrained_hashes": masked._verify_pretrained(pretrained),
        "hwr_checkpoint_sha256": masked._sha256(hwr_checkpoint),
        "parent_checkpoint_sha256": masked._sha256(base_checkpoint),
        "baseline_freeze_sha256": freeze_sha,
        "selection": {
            "kind": selected_kind,
            "epoch": int(selected["epoch"]),
            "lambda": float(selected["lambda"]),
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
        },
        "rights": {
            "CROHME_license": "CC BY-NC 4.0",
            "research_only": True,
            "commercial_use": False,
            "may_replace_product_checkpoint": False,
        },
        "shape_training": False,
        "candidate_policy": "select one token from immutable HWR Top-5",
        "state_dict": selected_state,
    }
    checkpoint_path = output / "crohme_standard_context_research.pt"
    torch.save(checkpoint_payload, checkpoint_path)
    checkpoint_sha = masked._sha256(checkpoint_path)

    reloaded_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    reload_dev, reload_dev_audit = _predict_state(
        reloaded_payload["state_dict"],
        float(reloaded_payload["selection"]["lambda"]),
        dev_rows,
        pretrained,
        labels,
        device,
        args.predict_batch_size,
    )
    reload_valid, reload_valid_audit = _predict_state(
        reloaded_payload["state_dict"],
        float(reloaded_payload["selection"]["lambda"]),
        valid_rows,
        pretrained,
        labels,
        device,
        args.predict_batch_size,
    )
    reload_mismatches = {
        "dev": sum(
            reload_dev[key] != value for key, value in selected_dev.items()
        ),
        "valid": sum(
            reload_valid[key] != value for key, value in selected_valid.items()
        ),
    }
    if any(reload_mismatches.values()):
        raise AssertionError(f"CROHME research checkpoint reload mismatch: {reload_mismatches}")

    dev_prediction_path = output / "dev_predictions.jsonl.gz"
    valid_prediction_path = output / "valid_predictions.jsonl.gz"
    _write_predictions(dev_prediction_path, dev_rows, base_dev, selected_dev)
    _write_predictions(valid_prediction_path, valid_rows, base_valid, selected_valid)

    immutable_after = {
        "train_candidates_sha256": masked._sha256(train_path),
        "valid_candidates_sha256": masked._sha256(valid_path),
        "base_checkpoint_sha256": masked._sha256(base_checkpoint),
        "hwr_checkpoint_sha256": masked._sha256(hwr_checkpoint),
    }
    immutable_unchanged = all(
        immutable_after[key] == baseline_freeze["inputs"][key]
        for key in immutable_after
    )
    report = {
        "schema": REPORT_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rights": checkpoint_payload["rights"],
        "selection_contract": {
            "official_test_loaded": False,
            "official_test_metrics_loaded": False,
            "valid_used_for_selection": False,
            "train_derived_double_holdout_only": True,
            "variants": 3,
            "shape_hwr_frozen": True,
            "candidate_set_frozen": True,
            "grouping_frozen": True,
        },
        "split_audit": split_audit,
        "training": {
            "fit_records": len(fit_rows),
            "fit_formulas": len(candidate._formula_rows(fit_rows)),
            "training_examples": len(examples),
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "variants": variants,
        },
        "selection": selected,
        "selected_kind": selected_kind,
        "evaluation": {
            "dev": {
                "hwr": _metric_view(_metrics(
                    dev_rows,
                    {str(row["record_id"]): str(row["final_topk"][0]) for row in dev_rows},
                )),
                "parent": _metric_view(base_dev_metrics),
                "selected": _metric_view(selected_dev_metrics),
                "paired_selected_vs_parent": _paired(dev_rows, base_dev, selected_dev),
                "top5_oracle": _oracle(dev_rows),
            },
            "valid_post_selection": {
                "selection_role": "none; post-selection repeated noncommercial standard diagnostic",
                "hwr": _metric_view(_metrics(
                    valid_rows,
                    {str(row["record_id"]): str(row["final_topk"][0]) for row in valid_rows},
                )),
                "parent": _metric_view(base_valid_metrics),
                "selected": _metric_view(selected_valid_metrics),
                "paired_selected_vs_parent": _paired(valid_rows, base_valid, selected_valid),
                "top5_oracle": _oracle(valid_rows),
            },
        },
        "integrity": {
            "candidate_audits": {
                "selected_dev": selected_dev_audit,
                "selected_valid": selected_valid_audit,
                "reload_dev": reload_dev_audit,
                "reload_valid": reload_valid_audit,
            },
            "reload_mismatches": reload_mismatches,
            "immutable_inputs_unchanged": immutable_unchanged,
            "shape_gradient_updates": 0,
            "grouping_mutations": 0,
        },
        "artifacts": {
            "baseline_freeze": {"path": str(freeze_path), "sha256": freeze_sha},
            "checkpoint": {"path": str(checkpoint_path), "sha256": checkpoint_sha},
            "dev_predictions": {
                "path": str(dev_prediction_path),
                "sha256": masked._sha256(dev_prediction_path),
            },
            "valid_predictions": {
                "path": str(valid_prediction_path),
                "sha256": masked._sha256(valid_prediction_path),
            },
        },
        "decision": (
            "research candidate selected for one final CROHME test diagnostic"
            if selected_kind != "no_op_parent"
            else "no valid train/dev improvement; retain parent"
        ),
        "product_adopted": False,
    }
    report_path = output / "selection_report.json"
    _save_json(report_path, report)
    _event(
        "crohme_context_selection_complete",
        selected_kind=selected_kind,
        selected_epoch=selected["epoch"],
        selected_lambda=selected["lambda"],
        dev_top1=selected_dev_metrics["all_top1"],
        dev_formula_exact=selected_dev_metrics["formula_exact"],
        dev_strict_macro=selected_dev_metrics["strict_macro_top1"],
        valid_top1=selected_valid_metrics["all_top1"],
        valid_formula_exact=selected_valid_metrics["formula_exact"],
        valid_strict_macro=selected_valid_metrics["strict_macro_top1"],
        checkpoint=str(checkpoint_path),
        checkpoint_sha256=checkpoint_sha,
        report=str(report_path),
        report_sha256=masked._sha256(report_path),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
