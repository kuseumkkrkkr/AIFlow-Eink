#!/usr/bin/env python3
"""Distil a frozen MathBERTa formula teacher into the deployable BERT-Tiny context layer."""

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
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import RobertaForMaskedLM, RobertaTokenizerFast

from character_tensor_v1 import ROOT, _json_lines
from evaluate_homograph_context_reranker_v1 import _metrics
import train_context_decision_layer_v1 as context
import train_masked_context_reranker_v1 as masked


SCHEMA = "aiflow-distilled-math-context/v1"
TEACHER_ID = "witiko/mathberta"
TEACHER_REVISION = "4cb18380847a27c6d0d1d3db3459a78cd1b602cd"
TEACHER_LICENSE = "MIT"
TEACHER_CORPUS_RIGHTS_STATUS = (
    "not cleared for commercial use: arXMLiv 2020 is restricted to research/tool development; "
    "Math StackExchange attribution/share-alike obligations require separate review"
)
CONFIGURATION_PROVENANCE = (
    "frozen after prior project-only r2/r4 development; direct writer-LOO is repeated-development "
    "evidence, not untouched acceptance"
)
TEACHER_FILES = (
    "README.md", "config.json", "pytorch_model.bin", "tokenizer.json",
    "tokenizer_config.json", "vocab.json", "merges.txt", "added_tokens.json",
)
PINNED_TEACHER_SHA256 = {
    "README.md": "a150068a68427de1d3ed57a1e6fcb6ac1ef996b9f81b6d00e6de4fcb527f2d78",
    "config.json": "ce5475d17adee98923939caf894c499c0b2660623c4cd38279dbf38fbda018c3",
    "pytorch_model.bin": "8d80d16ee130a9f3156cc13609208c69b30de3353622b5cd078885499c5ded88",
    "tokenizer.json": "b2123ba84f8bdd3658ddc7670b7c2bc3bca59dfb66dc9a1ee1a97dac624b790c",
    "tokenizer_config.json": "2fca8e1f5d7512f7ec5c22d74e0098cc3e7318466307faa2cba65236263f6ece",
    "vocab.json": "ed19656ea1707df69134c4af35c8ceda2cc9860bf2c3495026153a133670ab5e",
    "merges.txt": "fe36cab26d4f4421ed725e10a2e9ddb7f799449c603a96e7f29b5a3c82a95862",
    "added_tokens.json": "ba6ac69a9f365a9940de43b2ab2f377a39bc656aea6390c28b93db8a65106762",
}
DEFAULT_TEACHER = ROOT / "research" / "pretrained" / "witiko_mathberta_4cb1838"
DEFAULT_INPUT = ROOT / "artifacts" / "homograph_context_20260814"
DEFAULT_REFERENCE = ROOT / "artifacts" / "context_role_finalizer_20260819_r2" / "context_role_finalizer_report.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "distilled_math_context_20260819"
SEED = 20260819
TEMPERATURE = 2.0
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-3
SUPERVISED_WEIGHT = 0.1
MAX_EPOCHS = 40
PATIENCE = 8
MIN_DELTA = 1e-5
PRODUCT_CONFIGURATION = {**context.PINNED_ROLE_CONFIGURATION, "exact_context_scale": 2.0}


def _event(name: str, **values: object) -> None:
    print(json.dumps({"event": name, **values}, ensure_ascii=False), flush=True)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _salt_seed(salt: str) -> int:
    return SEED + int.from_bytes(salt.encode("utf-8"), "little", signed=False) % 100_000


def _verify_teacher(path: Path) -> dict[str, str]:
    path = masked._d_path(path, "MathBERTa teacher")
    missing = [name for name in TEACHER_FILES if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"MathBERTa teacher files missing: {missing}")
    card = (path / "README.md").read_text(encoding="utf-8")
    if "license: mit" not in card.lower():
        raise ValueError("MathBERTa MIT license marker missing")
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    expected = {
        "model_type": "roberta", "hidden_size": 768, "num_hidden_layers": 12,
        "num_attention_heads": 12, "vocab_size": 78680,
    }
    if {key: config.get(key) for key in expected} != expected:
        raise ValueError("pinned MathBERTa architecture contract changed")
    hashes = {name: masked._sha256(path / name) for name in TEACHER_FILES}
    if hashes != PINNED_TEACHER_SHA256:
        raise ValueError("pinned MathBERTa file hashes changed")
    return hashes


def _teacher_anchor_ids(tokenizer, text: str, base_vocab_size: int) -> list[int]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    valid = [index for index in ids if index < base_vocab_size and index != tokenizer.unk_token_id]
    return valid or [tokenizer.unk_token_id]


def _build_teacher(path: Path, labels: list[str], device: torch.device):
    tokenizer = RobertaTokenizerFast.from_pretrained(path, local_files_only=True)
    base_vocab_size = len(tokenizer)
    anchors = [_teacher_anchor_ids(tokenizer, label, base_vocab_size) for label in labels]
    relation_anchors = {
        relation: _teacher_anchor_ids(tokenizer, relation.replace("script", " script"), base_vocab_size)
        for relation in masked.RELATIONS
    }
    class_tokens = masked._class_tokens(labels)
    relation_tokens = masked._relation_tokens()
    added = tokenizer.add_special_tokens({
        "additional_special_tokens": class_tokens + [relation_tokens[name] for name in masked.RELATIONS]
    })
    if added != len(labels) + len(masked.RELATIONS):
        raise ValueError("MathBERTa custom-token collision")
    model = RobertaForMaskedLM.from_pretrained(path, local_files_only=True)
    model.resize_token_embeddings(len(tokenizer))
    class_ids = [tokenizer.convert_tokens_to_ids(token) for token in class_tokens]
    relation_ids = {name: tokenizer.convert_tokens_to_ids(token) for name, token in relation_tokens.items()}
    embeddings = model.get_input_embeddings().weight
    with torch.no_grad():
        for custom_id, source_ids in zip(class_ids, anchors, strict=True):
            embeddings[custom_id].copy_(embeddings[source_ids].mean(dim=0))
        for relation, custom_id in relation_ids.items():
            embeddings[custom_id].copy_(embeddings[relation_anchors[relation]].mean(dim=0))
    model.tie_weights()
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    contract = {
        "labels": labels,
        "label_to_index": {label: index for index, label in enumerate(labels)},
        "class_tokens": class_tokens,
        "class_ids": class_ids,
        "relation_tokens": relation_tokens,
        "relation_ids": relation_ids,
        "pad_id": tokenizer.pad_token_id,
        "cls_id": tokenizer.cls_token_id,
        "sep_id": tokenizer.sep_token_id,
        "mask_id": tokenizer.mask_token_id,
        "base_vocab_size": base_vocab_size,
        "expanded_vocab_size": len(tokenizer),
    }
    if None in (contract["pad_id"], contract["cls_id"], contract["sep_id"], contract["mask_id"]):
        raise ValueError("MathBERTa special-token contract incomplete")
    return model, contract


def _teacher_class_logits(model, input_ids, attention_mask, mask_positions, class_ids):
    hidden = model.roberta(
        input_ids=input_ids, attention_mask=attention_mask, return_dict=True
    ).last_hidden_state
    masked_hidden = hidden[torch.arange(len(hidden), device=hidden.device), mask_positions]
    transformed = model.lm_head.layer_norm(F.gelu(model.lm_head.dense(masked_hidden)))
    weight = model.get_output_embeddings().weight.index_select(0, class_ids)
    bias = model.lm_head.bias.index_select(0, class_ids)
    return F.linear(transformed, weight, bias)


@torch.inference_mode()
def _teacher_targets(model, contract: dict, rows: list[dict], device: torch.device, batch_size: int) -> dict[str, torch.Tensor]:
    packed = masked._pack(rows, contract)
    class_ids = torch.tensor(contract["class_ids"], dtype=torch.long, device=device)
    output = []
    for start in range(0, len(rows), batch_size):
        stop = min(start + batch_size, len(rows))
        output.append(_teacher_class_logits(
            model,
            packed["input_ids"][start:stop].to(device),
            packed["attention_mask"][start:stop].to(device),
            packed["mask_positions"][start:stop].to(device),
            class_ids,
        ).cpu())
    logits = torch.cat(output)
    return {
        str(row["record_id"]): logits[index]
        for index, row in enumerate(packed["rows"])
    }


def _student_pack(rows: list[dict], contract: dict, targets: dict[str, torch.Tensor]) -> tuple[dict, torch.Tensor]:
    packed = masked._pack(rows, contract)
    ordered = torch.stack([targets[str(row["record_id"])] for row in packed["rows"]])
    return packed, F.softmax(ordered / TEMPERATURE, dim=1)


def _new_student(labels: list[str], device: torch.device, seed: int):
    model, _, contract = masked._build_model(masked.DEFAULT_PRETRAINED, labels, device, seed)
    return model, contract


def _distill_epoch(model, contract: dict, packed: dict, targets: torch.Tensor, optimizer, device: torch.device, batch_size: int, generator) -> float:
    model.train()
    class_ids = torch.tensor(contract["class_ids"], dtype=torch.long, device=device)
    order = torch.randperm(len(targets), generator=generator)
    losses = []
    for start in range(0, len(order), batch_size):
        indices = order[start:start + batch_size]
        optimizer.zero_grad(set_to_none=True)
        logits = masked._class_logits(
            model,
            packed["input_ids"][indices].to(device),
            packed["attention_mask"][indices].to(device),
            packed["mask_positions"][indices].to(device),
            class_ids,
        )
        distillation_loss = F.kl_div(
            F.log_softmax(logits / TEMPERATURE, dim=1),
            targets[indices].to(device),
            reduction="batchmean",
        ) * TEMPERATURE**2
        supervised_loss = F.cross_entropy(logits, packed["targets"][indices].to(device))
        loss = distillation_loss + SUPERVISED_WEIGHT * supervised_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    return float(np.mean(losses))


@torch.inference_mode()
def _distill_loss(model, contract: dict, packed: dict, targets: torch.Tensor, device: torch.device, batch_size: int) -> float:
    model.eval()
    class_ids = torch.tensor(contract["class_ids"], dtype=torch.long, device=device)
    total, count = 0.0, 0
    for start in range(0, len(targets), batch_size):
        stop = min(start + batch_size, len(targets))
        logits = masked._class_logits(
            model,
            packed["input_ids"][start:stop].to(device),
            packed["attention_mask"][start:stop].to(device),
            packed["mask_positions"][start:stop].to(device),
            class_ids,
        )
        distillation_loss = F.kl_div(
            F.log_softmax(logits / TEMPERATURE, dim=1),
            targets[start:stop].to(device),
            reduction="sum",
        ) * TEMPERATURE**2
        total += float(distillation_loss)
        count += stop - start
    return total / max(count, 1)


def _select_epochs(
    rows: list[dict],
    targets: dict[str, torch.Tensor],
    labels: list[str],
    device: torch.device,
    batch_size: int,
    salt: str,
) -> dict:
    fit_rows, validation_rows = masked._inner_split(rows, f"distill-{salt}")
    seed = _salt_seed(salt)
    model, contract = _new_student(labels, device, seed)
    fit_pack, fit_targets = _student_pack(fit_rows, contract, targets)
    validation_pack, validation_targets = _student_pack(validation_rows, contract, targets)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    generator = torch.Generator().manual_seed(seed)
    best_epoch, best_loss, stale, curve = 1, float("inf"), 0, []
    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = _distill_epoch(
            model, contract, fit_pack, fit_targets, optimizer, device, batch_size, generator
        )
        validation_loss = _distill_loss(
            model, contract, validation_pack, validation_targets, device, batch_size
        )
        curve.append({"epoch": epoch, "train_objective": train_loss, "validation_kl": validation_loss})
        if validation_loss < best_loss - MIN_DELTA:
            best_epoch, best_loss, stale = epoch, validation_loss, 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "epoch": best_epoch,
        "validation_kl": best_loss,
        "fit_formulas": len(masked._formulae(fit_rows)),
        "validation_formulas": len(masked._formulae(validation_rows)),
        "curve": curve,
    }


def _fit_student(
    rows: list[dict],
    targets: dict[str, torch.Tensor],
    labels: list[str],
    device: torch.device,
    batch_size: int,
    epochs: int,
    salt: str,
):
    seed = _salt_seed(salt)
    model, contract = _new_student(labels, device, seed)
    packed, probabilities = _student_pack(rows, contract, targets)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    generator = torch.Generator().manual_seed(seed)
    losses = [
        _distill_epoch(model, contract, packed, probabilities, optimizer, device, batch_size, generator)
        for _ in range(epochs)
    ]
    model.eval()
    return model, contract, losses


def _predict(model, contract: dict, rows: list[dict], training_rows: list[dict], device: torch.device, batch_size: int):
    grammar = context._fit_role_grammar(training_rows)
    label_support = dict(Counter(str(row["label"]) for row in training_rows))
    reliability = context._fit_top1_reliability(training_rows)
    components = context._role_components(model, contract, rows, grammar, device, batch_size)
    return context._predict(components, PRODUCT_CONFIGURATION, label_support, reliability)


def _writer_loo(
    targets: dict[str, torch.Tensor],
    rows: list[dict],
    labels: list[str],
    device: torch.device,
    batch_size: int,
):
    predictions, folds = {}, []
    for index, writer in enumerate(sorted({str(row["writer_group"]) for row in rows})):
        training = [row for row in rows if str(row["writer_group"]) != writer]
        held = [row for row in rows if str(row["writer_group"]) == writer]
        selection = _select_epochs(training, targets, labels, device, batch_size, f"writer-{writer}")
        model, contract, losses = _fit_student(
            training, targets, labels, device, batch_size, selection["epoch"], f"writer-fit-{writer}"
        )
        fold_predictions, audit = _predict(model, contract, held, training, device, batch_size)
        predictions.update(fold_predictions)
        metrics = _metrics(held, fold_predictions)
        folds.append({
            "held_writer_group": writer,
            "selection": selection,
            "fit_final_objective": losses[-1],
            "metrics": metrics,
            "audit": audit,
        })
        _event("distilled_writer_fold", fold=index + 1, writer=writer, epoch=selection["epoch"], top1=metrics["all_top1"])
        del model
        gc.collect()
        torch.cuda.empty_cache()
    if set(predictions) != {str(row["record_id"]) for row in rows}:
        raise AssertionError("distilled writer-LOO coverage mismatch")
    return predictions, folds


def load_distilled_context_student(pretrained: Path, checkpoint_path: Path, device: torch.device):
    pretrained = masked._d_path(pretrained, "BERT-Tiny base")
    checkpoint_path = masked._d_path(checkpoint_path, "distilled context checkpoint")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (
        payload.get("schema") != SCHEMA
        or payload.get("context_schema") != context.SCHEMA
        or payload.get("teacher_id") != TEACHER_ID
        or payload.get("teacher_revision") != TEACHER_REVISION
        or payload.get("teacher_license") != TEACHER_LICENSE
        or payload.get("teacher_corpus_rights_status") != TEACHER_CORPUS_RIGHTS_STATUS
        or payload.get("teacher_file_sha256") != PINNED_TEACHER_SHA256
        or payload.get("student_base_sha256") != masked._verify_pretrained(pretrained)
        or payload.get("roles") != list(context.ROLES)
        or payload.get("math_labels") != masked._labels(masked.DEFAULT_PRODUCT)
        or payload.get("configuration") != PRODUCT_CONFIGURATION
        or payload.get("configuration_provenance") != CONFIGURATION_PROVENANCE
        or payload.get("temperature") != TEMPERATURE
        or payload.get("supervised_weight") != SUPERVISED_WEIGHT
    ):
        raise ValueError("distilled context checkpoint contract mismatch")
    model, _, contract = masked._build_model(pretrained, list(payload["math_labels"]), device, SEED)
    current_roles = {label: context._semantic_role(label) for label in contract["labels"]}
    if (
        payload.get("class_tokens") != contract["class_tokens"]
        or payload.get("relation_tokens") != contract["relation_tokens"]
        or payload.get("semantic_role_by_label") != current_roles
    ):
        raise ValueError("distilled context token or semantic-role contract mismatch")
    model.load_state_dict(payload["student_state_dict"], strict=True)
    context._freeze(model)
    return model, contract, payload


def predict_distilled_context_student(
    model, contract: dict, payload: dict, rows: list[dict], device: torch.device, batch_size: int = 128,
) -> tuple[dict[str, str], dict]:
    components = context._role_components(
        model, contract, rows, payload["role_grammar"], device, batch_size
    )
    return context._predict(
        components,
        payload["configuration"],
        payload["label_support"],
        payload["top1_reliability"],
    )


def _assert_candidate_contract(name: str, audit: dict) -> None:
    if (
        audit["candidate_preservation_rate"] != 1.0
        or audit["new_tokens"] != 0
        or audit["grouping_mutations"] != 0
    ):
        raise AssertionError(f"{name} candidate contract violated: {audit}")


def _self_test() -> None:
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    probabilities = F.softmax(logits / TEMPERATURE, dim=1)
    assert probabilities.shape == logits.shape
    assert PRODUCT_CONFIGURATION["exact_context_scale"] > 0.0
    assert tuple(PINNED_TEACHER_SHA256) == TEACHER_FILES
    assert context._semantic_role(r"\sqrt{}") == "operator"
    assert context._semantic_role(r"\varkappa") == "operand"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--reference-report", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        _event("self_test", status="pass")
        return 0
    if args.batch_size < 1:
        parser.error("batch size must be positive")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    teacher_path = masked._d_path(args.teacher, "MathBERTa teacher")
    input_root = masked._d_path(args.input, "candidate input")
    output = masked._d_path(args.output, "distilled context output")
    report_path = output / "distilled_math_context_report.json"
    checkpoint_path = output / "distilled_math_context_product.pt"
    if report_path.exists() or checkpoint_path.exists():
        parser.error(f"refusing to overwrite distilled context output: {output}")
    teacher_hashes = _verify_teacher(teacher_path)
    student_base_hashes = masked._verify_pretrained(masked.DEFAULT_PRETRAINED)
    direct_path = input_root / "direct_candidates.jsonl.gz"
    crohme_path = input_root / "crohme_candidates.jsonl.gz"
    direct_rows, crohme_rows = list(_json_lines(direct_path)), list(_json_lines(crohme_path))
    assert_training_path_clean(teacher_path, "distillation teacher")
    assert_training_path_clean(direct_path, "project-owned distillation candidates")
    assert_training_rows_clean(direct_rows)
    labels = masked._labels(masked.DEFAULT_PRODUCT)
    _set_seed(SEED)
    _event("teacher_start", records=len(direct_rows), device=str(device))
    teacher, teacher_contract = _build_teacher(teacher_path, labels, device)
    targets = _teacher_targets(teacher, teacher_contract, direct_rows, device, args.batch_size)
    del teacher
    gc.collect()
    torch.cuda.empty_cache()

    direct_predictions, folds = _writer_loo(targets, direct_rows, labels, device, args.batch_size)
    direct_baseline_predictions = {str(row["record_id"]): str(row["final_topk"][0]) for row in direct_rows}
    direct_baseline = _metrics(direct_rows, direct_baseline_predictions)
    direct_metrics = _metrics(direct_rows, direct_predictions)
    product_selection = _select_epochs(direct_rows, targets, labels, device, args.batch_size, "product")
    product, product_contract, product_losses = _fit_student(
        direct_rows, targets, labels, device, args.batch_size, product_selection["epoch"], "product-fit"
    )
    direct_refit_predictions, direct_refit_audit = _predict(
        product, product_contract, direct_rows, direct_rows, device, args.batch_size
    )
    crohme_predictions, crohme_audit = _predict(
        product, product_contract, crohme_rows, direct_rows, device, args.batch_size
    )
    crohme_baseline_predictions = {str(row["record_id"]): str(row["final_topk"][0]) for row in crohme_rows}
    crohme_baseline = _metrics(crohme_rows, crohme_baseline_predictions)
    crohme_metrics = _metrics(crohme_rows, crohme_predictions)

    direct_audit = masked._candidate_audit(direct_rows, direct_predictions)
    direct_refit_candidate_audit = masked._candidate_audit(direct_rows, direct_refit_predictions)
    crohme_candidate_audit = masked._candidate_audit(crohme_rows, crohme_predictions)
    _assert_candidate_contract("direct writer-LOO", direct_audit)
    _assert_candidate_contract("direct product refit", direct_refit_candidate_audit)
    _assert_candidate_contract("CROHME transfer", crohme_candidate_audit)
    training_input_hashes = {
        "direct_candidates.jsonl.gz": masked._sha256(direct_path),
    }
    evaluation_input_hashes = {
        "crohme_candidates.jsonl.gz": masked._sha256(crohme_path),
    }
    reference_hash = masked._sha256(args.reference_report) if args.reference_report.is_file() else None

    output.mkdir(parents=True, exist_ok=True)
    grammar = context._fit_role_grammar(direct_rows)
    label_support = dict(Counter(str(row["label"]) for row in direct_rows))
    reliability = context._fit_top1_reliability(direct_rows)
    payload = {
        "schema": SCHEMA,
        "context_schema": context.SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "teacher_id": TEACHER_ID,
        "teacher_revision": TEACHER_REVISION,
        "teacher_license": TEACHER_LICENSE,
        "teacher_corpus_rights_status": TEACHER_CORPUS_RIGHTS_STATUS,
        "teacher_file_sha256": teacher_hashes,
        "student_base_sha256": student_base_hashes,
        "candidate_input_sha256": training_input_hashes,
        "reference_report_sha256": reference_hash,
        "math_labels": labels,
        "class_tokens": product_contract["class_tokens"],
        "relation_tokens": product_contract["relation_tokens"],
        "roles": list(context.ROLES),
        "semantic_role_by_label": {label: context._semantic_role(label) for label in labels},
        "student_state_dict": {key: value.detach().cpu() for key, value in product.state_dict().items()},
        "role_grammar": grammar,
        "configuration": PRODUCT_CONFIGURATION,
        "configuration_provenance": CONFIGURATION_PROVENANCE,
        "label_support": label_support,
        "top1_reliability": reliability,
        "training_records": len(direct_rows),
        "training_formulas": len(masked._formulae(direct_rows)),
        "selected_epochs": product_selection["epoch"],
        "temperature": TEMPERATURE,
        "supervised_weight": SUPERVISED_WEIGHT,
        "contract": (
            "research-only MathBERTa KL plus 0.1 project truth CE; runtime uses BERT-Tiny and "
            "preserves HWR Top-5; commercial corpus audit failed"
        ),
    }
    torch.save(payload, checkpoint_path)
    checkpoint_sha256 = masked._sha256(checkpoint_path)

    del product
    gc.collect()
    torch.cuda.empty_cache()
    reloaded, reloaded_contract, reloaded_payload = load_distilled_context_student(
        masked.DEFAULT_PRETRAINED, checkpoint_path, device
    )
    reloaded_direct, _ = predict_distilled_context_student(
        reloaded, reloaded_contract, reloaded_payload, direct_rows, device, args.batch_size
    )
    reloaded_crohme, _ = predict_distilled_context_student(
        reloaded, reloaded_contract, reloaded_payload, crohme_rows, device, args.batch_size
    )
    direct_reload_mismatches = sum(
        reloaded_direct[record_id] != prediction
        for record_id, prediction in direct_refit_predictions.items()
    )
    crohme_reload_mismatches = sum(
        reloaded_crohme[record_id] != prediction
        for record_id, prediction in crohme_predictions.items()
    )
    reloaded_direct_audit = masked._candidate_audit(direct_rows, reloaded_direct)
    reloaded_crohme_audit = masked._candidate_audit(crohme_rows, reloaded_crohme)
    _assert_candidate_contract("reloaded direct product", reloaded_direct_audit)
    _assert_candidate_contract("reloaded CROHME", reloaded_crohme_audit)
    if direct_reload_mismatches or crohme_reload_mismatches:
        raise AssertionError(
            f"checkpoint reload mismatch: direct={direct_reload_mismatches}, "
            f"crohme={crohme_reload_mismatches}"
        )
    del reloaded
    gc.collect()
    torch.cuda.empty_cache()

    masked._write_prediction_rows(output / "direct_writer_loo_predictions.jsonl.gz", direct_rows, direct_predictions)
    masked._write_prediction_rows(output / "direct_product_refit_predictions.jsonl.gz", direct_rows, direct_refit_predictions)
    masked._write_prediction_rows(output / "crohme_transfer_predictions.jsonl.gz", crohme_rows, crohme_predictions)
    reference = json.loads(args.reference_report.read_text(encoding="utf-8")) if args.reference_report.is_file() else None
    reference_direct = reference["evaluation"]["direct_writer_loo"]["context_role_finalizer"] if reference else None
    reference_crohme = reference["evaluation"]["crohme_transfer"]["context_role_finalizer"] if reference else None
    comparison_metrics = ("all_top1", "strict_micro_top1", "strict_macro_top1", "formula_exact")
    research_gate = (
        reference_direct is not None
        and all(direct_metrics[key] >= reference_direct[key] for key in comparison_metrics)
        and direct_audit["candidate_preservation_rate"] == 1.0
    )
    report = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "teacher": {
            "id": TEACHER_ID, "revision": TEACHER_REVISION, "license": TEACHER_LICENSE,
            "architecture": "RoBERTa 12 layers / hidden 768 / 12 heads; training-only teacher",
            "training_corpus_rights_status": TEACHER_CORPUS_RIGHTS_STATUS,
            "file_sha256": teacher_hashes,
        },
        "student": {
            "architecture": "Google BERT-Tiny 2 layers / hidden 128 / 2 heads",
            "student_base_sha256": student_base_hashes,
            "temperature": TEMPERATURE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "supervised_weight": SUPERVISED_WEIGHT,
            "max_epochs": MAX_EPOCHS,
            "product_configuration": PRODUCT_CONFIGURATION,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_bytes": checkpoint_path.stat().st_size,
        },
        "data_admission": {
            "distillation": "project-owned ownership only: 47 formulae / 211 glyphs / three writers; teacher KL plus 0.1 project-truth CE",
            "teacher_targets": "frozen MIT MathBERTa logits over all 372 classes",
            "crohme": "CC BY-NC transfer diagnostic only; never used for training or epoch selection",
            "training_candidate_input_sha256": training_input_hashes,
            "evaluation_candidate_input_sha256": evaluation_input_hashes,
            "r2_reference_report_sha256": reference_hash,
        },
        "selection": {
            "method": "formula-disjoint teacher-KL early stopping inside every outer writer fold",
            "configuration_provenance": CONFIGURATION_PROVENANCE,
            "writer_loo_folds": folds,
            "product": product_selection,
            "product_fit_final_objective": product_losses[-1],
            "training_data_guard": zero_crohme_training_manifest(
                admitted_sources={"project_owned_distillation_candidates": len(direct_rows)}
            ),
        },
        "evaluation": {
            "direct_writer_loo": {
                "baseline": direct_baseline,
                "distilled_context": direct_metrics,
                "delta": masked._metric_delta(direct_metrics, direct_baseline),
                "candidate_audit": direct_audit,
                "r2_reference": reference_direct,
            },
            "direct_product_refit_resubstitution": {
                "warning": "training-set fit only; not acceptance evidence",
                "metrics": _metrics(direct_rows, direct_refit_predictions),
                "runtime_audit": direct_refit_audit,
                "candidate_audit": direct_refit_candidate_audit,
            },
            "crohme_transfer": {
                "baseline": crohme_baseline,
                "distilled_context": crohme_metrics,
                "delta": masked._metric_delta(crohme_metrics, crohme_baseline),
                "candidate_audit": crohme_candidate_audit,
                "runtime_audit": crohme_audit,
                "r2_reference": reference_crohme,
            },
        },
        "checkpoint_reload": {
            "direct_product_refit_mismatches": direct_reload_mismatches,
            "crohme_transfer_mismatches": crohme_reload_mismatches,
            "direct_candidate_audit": reloaded_direct_audit,
            "crohme_candidate_audit": reloaded_crohme_audit,
        },
        "decision": {
            "research_gate_passed": research_gate,
            "research_gate_sources": ["project_owned_writer_loo"],
            "crohme_metrics_used_for_gate": False,
            "commercial_training_corpus_audit_passed": False,
            "commercial_gate_passed": False,
            "runtime_status": "shadow distilled context finalizer",
            "automatic_default_replacement": False,
            "reason": (
                "MathBERTa training-corpus rights are not commercially cleared, both evaluation sets "
                "are repeatedly observed, and a fresh project-owned writer/formula acceptance set is required"
            ),
        },
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _event(
        "distillation_complete", report=str(report_path), checkpoint=str(checkpoint_path),
        direct_top1=direct_metrics["all_top1"], crohme_top1=crohme_metrics["all_top1"],
        crohme_strict_macro=crohme_metrics["strict_macro_top1"], research_gate=research_gate,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
