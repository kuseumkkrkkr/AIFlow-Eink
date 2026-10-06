#!/usr/bin/env python3
"""Fine-tune only the formula-context network with frozen HWR Top-5 inputs."""

from __future__ import annotations

from training_data_guard_v1 import (
    assert_training_entrypoint_arguments_clean,
    assert_training_path_clean,
    assert_training_rows_clean,
    zero_crohme_training_manifest,
)
if __name__ == "__main__":
    assert_training_entrypoint_arguments_clean()

import argparse
import gc
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from character_tensor_v1 import ROOT, _json_lines
from evaluate_homograph_context_reranker_v1 import _metrics
import train_masked_context_reranker_v1 as masked
import train_owned_formula_context_v1 as owned


SCHEMA = "aiflow-context-only-incremental-loop/v1"
DEFAULT_BASE_CONTEXT = (
    ROOT / "artifacts" / "owned_formula_context_20260822_r6"
    / "owned_formula_context_product.pt"
)
DEFAULT_BASE_HWR = (
    ROOT / "artifacts" / "unified_head_20260814"
    / "uniform_time_final_all_writers" / "project_symbol_head_checkpoint.pt"
)
NEW_WRITER_PARTITION = "new_writer"


def _event(name: str, **values: object) -> None:
    print(json.dumps({"event": name, **values}, ensure_ascii=False), flush=True)


def _seed(salt: str) -> int:
    digest = hashlib.sha256(salt.encode("utf-8")).digest()
    return owned.SEED + int.from_bytes(digest[:4], "big") % 100_000


def _paired(rows: list[dict], baseline: dict[str, str], candidate: dict[str, str]) -> dict:
    improved = regressed = changed = 0
    formulae: dict[str, list[dict]] = {}
    for row in rows:
        record_id = str(row["record_id"])
        truth = str(row["label"])
        before = str(baseline[record_id])
        after = str(candidate[record_id])
        changed += before != after
        improved += before != truth and after == truth
        regressed += before == truth and after != truth
        formulae.setdefault(str(row["formula_id"]), []).append(row)
    formula_improved = formula_regressed = 0
    for formula_rows in formulae.values():
        before_exact = all(
            str(baseline[str(row["record_id"])]) == str(row["label"])
            for row in formula_rows
        )
        after_exact = all(
            str(candidate[str(row["record_id"])]) == str(row["label"])
            for row in formula_rows
        )
        formula_improved += not before_exact and after_exact
        formula_regressed += before_exact and not after_exact
    return {
        "changed": changed,
        "improved": improved,
        "regressed": regressed,
        "net_improvement": improved - regressed,
        "formula_improved": formula_improved,
        "formula_regressed": formula_regressed,
    }


def _candidate_audit(rows: list[dict], predictions: dict[str, str]) -> dict:
    audit = masked._candidate_audit(rows, predictions)
    owned._assert_candidate_contract("context-only loop", audit)
    return audit


def _trial_key(trial: dict) -> tuple:
    metrics = trial["metrics"]
    paired = trial["paired_vs_base"]
    no_regression = paired["regressed"] == 0 and paired["formula_regressed"] == 0
    return (
        no_regression,
        float(metrics["all_top1"]),
        float(metrics["formula_exact"]),
        float(metrics["strict_macro_top1"]),
        float(metrics["strict_micro_top1"]),
        -int(paired["changed"]),
        trial["kind"] == "no_op",
    )


def _accuracy_key(trial: dict) -> tuple:
    metrics = trial["metrics"]
    paired = trial["paired_vs_base"]
    return (
        float(metrics["all_top1"]),
        float(metrics["formula_exact"]),
        float(metrics["strict_macro_top1"]),
        float(metrics["strict_micro_top1"]),
        -int(paired["formula_regressed"]),
        -int(paired["regressed"]),
        -int(paired["changed"]),
        -int(trial["epochs"]),
        -float(trial["learning_rate"]),
    )


def _new_from_state(
    contract: dict, state_dict: dict[str, torch.Tensor], device: torch.device, seed: int,
    trainable_scope: str,
) -> owned.OwnedFormulaContext:
    model = owned._new_model(contract, device, seed)
    model.load_state_dict(state_dict, strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    modules = {
        "all": (model,),
        "heads": (model.role_head, model.class_head),
        "top_layer_heads": (
            model.encoder.encoder.layer[-1], model.role_head, model.class_head,
        ),
    }.get(trainable_scope)
    if modules is None:
        raise ValueError(f"unsupported context trainable scope: {trainable_scope}")
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    return model


def _training_data(
    labels: list[str], rows: list[dict], examples: int, seed: int,
) -> dict[str, torch.Tensor]:
    contract = owned._contract(labels)
    return owned._append_real(owned._examples(labels, rows, examples, seed), rows, contract)


def _predict(
    model: owned.OwnedFormulaContext, contract: dict, payload: dict, rows: list[dict],
    device: torch.device, batch_size: int,
) -> tuple[dict[str, str], dict]:
    predictions, runtime = owned.decide_owned_formula_rows(
        model, contract, payload, rows, device, batch_size
    )
    _candidate_audit(rows, predictions)
    return predictions, runtime


def _selection_trials(
    rows: list[dict], labels: list[str], contract: dict, base_state: dict[str, torch.Tensor],
    base_payload: dict, base_predictions: dict[str, str], learning_rates: list[float],
    epochs: list[int], trainable_scopes: list[str], synthetic_examples: int,
    batch_size: int, predict_batch_size: int, weight_decay: float,
    device: torch.device,
) -> tuple[list[dict], dict[tuple[str, float, int], dict[str, str]]]:
    new_rows = [
        row for row in rows
        if str(row.get("evaluation_partition")) == NEW_WRITER_PARTITION
    ]
    writers = sorted({str(row["writer_group"]) for row in new_rows})
    if len(writers) < 2:
        raise ValueError("context-only loop requires at least two unseen writer groups")
    requested_epochs = sorted(set(epochs))
    predictions = {
        (scope, learning_rate, epoch): {}
        for scope in trainable_scopes
        for learning_rate in learning_rates for epoch in requested_epochs
    }
    fold_reports: dict[tuple[str, float, int], list[dict]] = {
        key: [] for key in predictions
    }
    max_epoch = max(requested_epochs)
    for scope in trainable_scopes:
        for learning_rate in learning_rates:
            for fold_index, writer in enumerate(writers, start=1):
                training_rows = [
                    row for row in rows if str(row["writer_group"]) != writer
                ]
                held_rows = [
                    row for row in new_rows if str(row["writer_group"]) == writer
                ]
                seed = _seed(f"incremental-{scope}-{learning_rate:.12g}-{writer}")
                owned._set_seed(seed)
                data = _training_data(labels, training_rows, synthetic_examples, seed)
                model = _new_from_state(contract, base_state, device, seed, scope)
                optimizer = torch.optim.AdamW(
                    (parameter for parameter in model.parameters() if parameter.requires_grad),
                    lr=learning_rate, weight_decay=weight_decay,
                )
                generator = torch.Generator().manual_seed(seed)
                for epoch in range(1, max_epoch + 1):
                    loss = owned._train_epoch(
                        model, data, optimizer, device, batch_size, generator
                    )
                    if epoch not in requested_epochs:
                        continue
                    fold_predictions, runtime = _predict(
                        model, contract, base_payload, held_rows, device,
                        predict_batch_size,
                    )
                    predictions[(scope, learning_rate, epoch)].update(fold_predictions)
                    fold_reports[(scope, learning_rate, epoch)].append({
                        "held_writer_group": writer,
                        "records": len(held_rows),
                        "formulas": len({str(row["formula_id"]) for row in held_rows}),
                        "objective": loss,
                        "metrics": _metrics(held_rows, fold_predictions),
                        "paired_vs_base": _paired(
                            held_rows, base_predictions, fold_predictions
                        ),
                        "runtime_audit": runtime,
                    })
                    _event(
                        "context_only_fold", learning_rate=learning_rate,
                        epoch=epoch, trainable_scope=scope, fold=fold_index,
                        writer=writer,
                        top1=fold_reports[(scope, learning_rate, epoch)][-1][
                            "metrics"
                        ]["all_top1"],
                    )
                del model, optimizer, data
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    expected = {str(row["record_id"]) for row in new_rows}
    trials = []
    for (scope, learning_rate, epoch), trial_predictions in predictions.items():
        if set(trial_predictions) != expected:
            raise AssertionError("new-writer OOF prediction coverage mismatch")
        trials.append({
            "kind": "fine_tune",
            "trainable_scope": scope,
            "learning_rate": learning_rate,
            "epochs": epoch,
            "metrics": _metrics(new_rows, trial_predictions),
            "paired_vs_base": _paired(new_rows, base_predictions, trial_predictions),
            "candidate_audit": _candidate_audit(new_rows, trial_predictions),
            "folds": fold_reports[(scope, learning_rate, epoch)],
        })
    return trials, predictions


def _fit_product(
    selected: dict, rows: list[dict], labels: list[str], contract: dict,
    base_state: dict[str, torch.Tensor], synthetic_examples: int, batch_size: int,
    weight_decay: float, device: torch.device,
) -> tuple[owned.OwnedFormulaContext, list[float]]:
    seed = _seed("incremental-product")
    scope = (
        "all" if selected["kind"] == "no_op"
        else str(selected.get("trainable_scope", "all"))
    )
    model = _new_from_state(contract, base_state, device, seed, scope)
    if selected["kind"] == "no_op":
        model.eval()
        return model, []
    data = _training_data(labels, rows, synthetic_examples, seed)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(selected["learning_rate"]), weight_decay=weight_decay,
    )
    generator = torch.Generator().manual_seed(seed)
    losses = []
    for epoch in range(1, int(selected["epochs"]) + 1):
        losses.append(owned._train_epoch(
            model, data, optimizer, device, batch_size, generator
        ))
        _event("context_only_product_epoch", epoch=epoch, objective=losses[-1])
    model.eval()
    return model, losses


def _checkpoint_payload(
    base_payload: dict, model: owned.OwnedFormulaContext, selected: dict,
    rows: list[dict], base_context: Path, target_hwr: Path, direct_path: Path,
) -> dict:
    payload = dict(base_payload)
    payload.update({
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hwr_checkpoint_sha256": masked._sha256(target_hwr),
        "candidate_input_sha256": {
            "direct": masked._sha256(direct_path),
        },
        "training_data": (
            f"context-only warm start; {len(rows)} owned glyphs, "
            f"{len({str(row['formula_id']) for row in rows})} formulas"
        ),
        "selected_epochs": (
            int(selected["epochs"]) if selected["kind"] != "no_op"
            else int(base_payload["selected_epochs"])
        ),
        "state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "parent_context_checkpoint_sha256": masked._sha256(base_context),
        "context_only_incremental_loop": {
            "schema": SCHEMA,
            "selected_kind": selected["kind"],
            "trainable_scope": selected.get("trainable_scope", "none"),
            "learning_rate": float(selected["learning_rate"]),
            "fine_tune_epochs": int(selected["epochs"]),
            "training_performed": selected["kind"] != "no_op",
            "shape_gradient_updates": 0,
            "runtime_configuration_frozen_from_parent": True,
        },
    })
    return payload


def _self_test() -> None:
    rows = [
        {"record_id": "a", "formula_id": "f", "label": "1"},
        {"record_id": "b", "formula_id": "f", "label": "+"},
    ]
    base = {"a": "1", "b": "1"}
    better = {"a": "1", "b": "+"}
    worse = {"a": "+", "b": "+"}
    assert _paired(rows, base, better) == {
        "changed": 1, "improved": 1, "regressed": 0, "net_improvement": 1,
        "formula_improved": 1, "formula_regressed": 0,
    }
    assert _paired(rows, base, worse)["regressed"] == 1
    template = {
        "metrics": {
            "all_top1": 0.5, "formula_exact": 0.5,
            "strict_macro_top1": 0.5, "strict_micro_top1": 0.5,
        },
        "paired_vs_base": {"regressed": 0, "formula_regressed": 0, "changed": 0},
    }
    no_op = {
        **template, "kind": "no_op", "trainable_scope": "none",
        "learning_rate": 0.0, "epochs": 0,
    }
    regressed = {
        **template, "kind": "fine_tune", "trainable_scope": "all",
        "learning_rate": 1e-4, "epochs": 1,
        "metrics": {**template["metrics"], "all_top1": 0.9},
        "paired_vs_base": {"regressed": 1, "formula_regressed": 0, "changed": 2},
    }
    assert max((no_op, regressed), key=_trial_key)["kind"] == "no_op"
    assert max((no_op, regressed), key=_accuracy_key)["kind"] == "fine_tune"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-context-checkpoint", type=Path, default=DEFAULT_BASE_CONTEXT)
    parser.add_argument("--base-hwr-checkpoint", type=Path, default=DEFAULT_BASE_HWR)
    parser.add_argument("--target-hwr-checkpoint", type=Path)
    parser.add_argument("--direct-candidates", type=Path)
    parser.add_argument("--crohme-candidates", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--learning-rates", type=float, nargs="+", default=(1e-5, 3e-5, 1e-4))
    parser.add_argument("--epochs", type=int, nargs="+", default=(1, 2, 3))
    parser.add_argument(
        "--trainable-scopes", nargs="+",
        choices=("all", "heads", "top_layer_heads"), default=("all",),
    )
    parser.add_argument("--synthetic-train-examples", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--predict-batch-size", type=int, default=256)
    parser.add_argument("--weight-decay", type=float, default=owned.WEIGHT_DECAY)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        _event("self_test", status="pass")
        return 0
    required = (args.target_hwr_checkpoint, args.direct_candidates, args.output)
    if any(value is None for value in required):
        parser.error("--target-hwr-checkpoint, --direct-candidates, and --output are required")
    if (
        min(args.learning_rates) <= 0.0 or min(args.epochs) < 1
        or min(args.synthetic_train_examples, args.batch_size, args.predict_batch_size) < 1
        or args.weight_decay < 0.0
    ):
        parser.error("learning rates, epochs, examples, and batches must be positive")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    base_context = masked._d_path(args.base_context_checkpoint, "base context checkpoint")
    base_hwr = masked._d_path(args.base_hwr_checkpoint, "base HWR checkpoint")
    target_hwr = masked._d_path(args.target_hwr_checkpoint, "target HWR checkpoint")
    direct_path = masked._d_path(args.direct_candidates, "direct candidate cache")
    crohme_path = (
        masked._d_path(args.crohme_candidates, "CROHME candidate cache")
        if args.crohme_candidates else None
    )
    output = masked._d_path(args.output, "context-only loop output")
    checkpoint_path = output / "context_only_product.pt"
    challenger_checkpoint_path = output / "context_only_challenger.pt"
    report_path = output / "context_only_loop_report.json"
    if checkpoint_path.exists() or challenger_checkpoint_path.exists() or report_path.exists():
        parser.error(f"refusing to overwrite context-only loop output: {output}")
    immutable_paths = [base_context, base_hwr, target_hwr, direct_path]
    if crohme_path is not None:
        immutable_paths.append(crohme_path)
    hashes_before = {str(path): masked._sha256(path) for path in immutable_paths}

    rows = list(_json_lines(direct_path))
    assert_training_path_clean(direct_path, "project-owned context training candidates")
    assert_training_rows_clean(rows)
    new_rows = [
        row for row in rows
        if str(row.get("evaluation_partition")) == NEW_WRITER_PARTITION
    ]
    if not rows or not new_rows:
        raise ValueError("direct candidates require populated legacy and new_writer partitions")
    base_model, contract, base_payload = owned.load_owned_formula_context(
        base_context, base_hwr, device
    )
    labels = masked._labels(target_hwr)
    if labels != contract["labels"]:
        raise ValueError("base and target HWR label vocabularies differ")
    base_predictions, base_runtime = _predict(
        base_model, contract, base_payload, rows, device, args.predict_batch_size
    )
    base_state = {
        key: value.detach().cpu().clone() for key, value in base_model.state_dict().items()
    }
    del base_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    no_op_predictions = {
        str(row["record_id"]): base_predictions[str(row["record_id"])] for row in new_rows
    }
    no_op = {
        "kind": "no_op",
        "trainable_scope": "none",
        "learning_rate": 0.0,
        "epochs": 0,
        "metrics": _metrics(new_rows, no_op_predictions),
        "paired_vs_base": _paired(new_rows, base_predictions, no_op_predictions),
        "candidate_audit": _candidate_audit(new_rows, no_op_predictions),
        "folds": [],
    }
    trials, trial_predictions = _selection_trials(
        rows, labels, contract, base_state, base_payload, base_predictions,
        sorted(set(args.learning_rates)), sorted(set(args.epochs)),
        list(dict.fromkeys(args.trainable_scopes)), args.synthetic_train_examples,
        args.batch_size, args.predict_batch_size, args.weight_decay, device,
    )
    all_trials = [no_op, *trials]
    selected = max(all_trials, key=_trial_key)
    challenger = max(trials, key=_accuracy_key)
    selected_predictions = (
        no_op_predictions if selected["kind"] == "no_op"
        else trial_predictions[(
            str(selected["trainable_scope"]), float(selected["learning_rate"]),
            int(selected["epochs"]),
        )]
    )
    challenger_predictions = trial_predictions[
        (
            str(challenger["trainable_scope"]), float(challenger["learning_rate"]),
            int(challenger["epochs"]),
        )
    ]
    selected_non_regressive = (
        selected["paired_vs_base"]["regressed"] == 0
        and selected["paired_vs_base"]["formula_regressed"] == 0
        and selected["metrics"]["strict_macro_top1"]
        >= no_op["metrics"]["strict_macro_top1"]
    )
    product, product_losses = _fit_product(
        selected, rows, labels, contract, base_state, args.synthetic_train_examples,
        args.batch_size, args.weight_decay, device,
    )
    product_predictions, product_runtime = _predict(
        product, contract, base_payload, rows, device, args.predict_batch_size
    )
    output.mkdir(parents=True, exist_ok=True)
    product_payload = _checkpoint_payload(
        base_payload, product, selected, rows, base_context, target_hwr,
        direct_path,
    )
    torch.save(product_payload, checkpoint_path)
    checkpoint_sha = masked._sha256(checkpoint_path)
    del product
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    reloaded, reloaded_contract, reloaded_payload = owned.load_owned_formula_context(
        checkpoint_path, target_hwr, device
    )
    reloaded_predictions, _ = _predict(
        reloaded, reloaded_contract, reloaded_payload, rows,
        device, args.predict_batch_size,
    )
    reload_mismatches = sum(
        reloaded_predictions[record_id] != prediction
        for record_id, prediction in product_predictions.items()
    )
    if reload_mismatches:
        raise AssertionError(f"context-only checkpoint reload mismatch: {reload_mismatches}")
    crohme_evaluation = None
    if crohme_path is not None:
        crohme_rows = list(_json_lines(crohme_path))
        crohme_predictions, crohme_runtime = _predict(
            reloaded, reloaded_contract, reloaded_payload, crohme_rows,
            device, args.predict_batch_size,
        )
        crohme_evaluation = {
            "selection_role": "none; post-selection noncommercial transfer diagnostic only",
            "metrics": _metrics(crohme_rows, crohme_predictions),
            "candidate_audit": _candidate_audit(crohme_rows, crohme_predictions),
            "runtime_audit": crohme_runtime,
        }
    challenger_product, challenger_losses = _fit_product(
        challenger, rows, labels, contract, base_state, args.synthetic_train_examples,
        args.batch_size, args.weight_decay, device,
    )
    challenger_product_predictions, challenger_product_runtime = _predict(
        challenger_product, contract, base_payload, rows, device, args.predict_batch_size
    )
    challenger_payload = _checkpoint_payload(
        base_payload, challenger_product, challenger, rows, base_context, target_hwr,
        direct_path,
    )
    torch.save(challenger_payload, challenger_checkpoint_path)
    challenger_checkpoint_sha = masked._sha256(challenger_checkpoint_path)
    del challenger_product
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    challenger_reloaded, challenger_contract, challenger_reloaded_payload = (
        owned.load_owned_formula_context(challenger_checkpoint_path, target_hwr, device)
    )
    challenger_reloaded_predictions, _ = _predict(
        challenger_reloaded, challenger_contract, challenger_reloaded_payload,
        rows, device, args.predict_batch_size,
    )
    challenger_reload_mismatches = sum(
        challenger_reloaded_predictions[record_id] != prediction
        for record_id, prediction in challenger_product_predictions.items()
    )
    if challenger_reload_mismatches:
        raise AssertionError(
            f"context-only challenger reload mismatch: {challenger_reload_mismatches}"
        )
    challenger_crohme_evaluation = None
    if crohme_path is not None:
        challenger_crohme_predictions, challenger_crohme_runtime = _predict(
            challenger_reloaded, challenger_contract, challenger_reloaded_payload,
            crohme_rows, device, args.predict_batch_size,
        )
        challenger_crohme_evaluation = {
            "selection_role": "none; post-selection noncommercial transfer diagnostic only",
            "metrics": _metrics(crohme_rows, challenger_crohme_predictions),
            "candidate_audit": _candidate_audit(
                crohme_rows, challenger_crohme_predictions
            ),
            "runtime_audit": challenger_crohme_runtime,
        }
    masked._write_prediction_rows(
        output / "new_writer_oof_predictions.jsonl.gz", new_rows, selected_predictions
    )
    masked._write_prediction_rows(
        output / "new_writer_challenger_oof_predictions.jsonl.gz",
        new_rows, challenger_predictions,
    )
    masked._write_prediction_rows(
        output / "direct_product_predictions.jsonl.gz", rows, product_predictions
    )
    hashes_after = {str(path): masked._sha256(path) for path in immutable_paths}
    if hashes_after != hashes_before:
        raise AssertionError("frozen HWR or candidate input changed during context-only training")
    report = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "frozen_shape_contract": {
            "hwr_training_performed": False,
            "shape_gradient_updates": 0,
            "target_hwr_checkpoint_sha256": hashes_before[str(target_hwr)],
            "direct_candidate_sha256": hashes_before[str(direct_path)],
            "immutable_hashes_before": hashes_before,
            "immutable_hashes_after": hashes_after,
            "unchanged": hashes_before == hashes_after,
        },
        "base_context": {
            "checkpoint_sha256": hashes_before[str(base_context)],
            "hwr_checkpoint_sha256": hashes_before[str(base_hwr)],
            "runtime_configuration_frozen": base_payload["configuration"],
            "all_direct_metrics": _metrics(rows, base_predictions),
            "new_writer_metrics": no_op["metrics"],
            "candidate_audit": _candidate_audit(rows, base_predictions),
            "runtime_audit": base_runtime,
        },
        "data": {
            "records": len(rows),
            "formulas": len({str(row["formula_id"]) for row in rows}),
            "writers": len({str(row["writer_group"]) for row in rows}),
            "new_writer_records": len(new_rows),
            "new_writer_formulas": len({str(row["formula_id"]) for row in new_rows}),
            "new_writer_groups": len({str(row["writer_group"]) for row in new_rows}),
            "crohme_training_or_selection": False,
        },
        "loop": {
            "selection_scope": "new_writer outer leave-one-writer-out only",
            "no_op_is_candidate": True,
            "learning_rates": sorted(set(args.learning_rates)),
            "epochs": sorted(set(args.epochs)),
            "trainable_scopes": list(dict.fromkeys(args.trainable_scopes)),
            "synthetic_train_examples_per_fit": args.synthetic_train_examples,
            "weight_decay": args.weight_decay,
            "trials": all_trials,
            "safe_selected": selected,
            "best_accuracy_challenger": challenger,
            "selected_non_regressive": selected_non_regressive,
            "training_data_guard": zero_crohme_training_manifest(
                admitted_sources={"project_owned_context_candidates": len(rows)}
            ),
        },
        "product_refit": {
            "warning": "all owned rows are fit input; not acceptance evidence",
            "training_performed": selected["kind"] != "no_op",
            "losses": product_losses,
            "metrics": _metrics(rows, product_predictions),
            "candidate_audit": _candidate_audit(rows, product_predictions),
            "runtime_audit": product_runtime,
        },
        "crohme_transfer": crohme_evaluation,
        "accuracy_challenger_refit": {
            "warning": "all owned rows are fit input; not acceptance evidence",
            "losses": challenger_losses,
            "metrics": _metrics(rows, challenger_product_predictions),
            "candidate_audit": _candidate_audit(rows, challenger_product_predictions),
            "runtime_audit": challenger_product_runtime,
        },
        "accuracy_challenger_crohme_transfer": challenger_crohme_evaluation,
        "checkpoint": {
            "safe": {
                "path": str(checkpoint_path),
                "sha256": checkpoint_sha,
                "reload_mismatches": reload_mismatches,
            },
            "accuracy_challenger": {
                "path": str(challenger_checkpoint_path),
                "sha256": challenger_checkpoint_sha,
                "reload_mismatches": challenger_reload_mismatches,
            },
        },
        "decision": {
            "selected_kind": selected["kind"],
            "accuracy_challenger_available": True,
            "research_loop_passed": selected_non_regressive,
            "commercial_accuracy_gate_passed": False,
            "automatic_default_replacement": False,
            "runtime_status": "shadow context-only candidate",
            "reason": (
                "the four unseen writer groups are now model-selection data; "
                "fresh untouched writer/formula acceptance is still required"
            ),
        },
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    _event(
        "context_only_complete", selected_kind=selected["kind"],
        learning_rate=selected["learning_rate"], epochs=selected["epochs"],
        new_writer_top1=selected["metrics"]["all_top1"],
        challenger_top1=challenger["metrics"]["all_top1"],
        report=str(report_path), checkpoint_sha256=checkpoint_sha,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
