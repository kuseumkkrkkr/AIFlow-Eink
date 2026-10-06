#!/usr/bin/env python3
"""A/B probe for adding owned synthetic 2-D relations to the mini formula LM.

This experiment uses only the project-owned prompt-context corpus for real
examples and the local formula DSL for synthetic examples. It never reads the
consumed 149-formula development set or CROHME. Synthetic evaluation is a
generator-disjoint diagnostic, not handwritten acceptance evidence.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from run_prompt_mini_lm_distillation_v1 import (
    BATCH_SIZE,
    DEFAULT_OUTPUT as BASE_STUDENT_DIR,
    FEEDFORWARD,
    HIDDEN,
    HEADS,
    LEARNING_RATE,
    LAYERS,
    MAX_POSITIONS,
    MiniFormulaLM,
    _pack,
)
from train_masked_context_reranker_v1 import RELATIONS, _formulae, _spatial_relation
from train_owned_formula_context_v1 import FormulaGenerator
from train_prompt_context_reranker_v1 import _prompt_rows, _split


SCHEMA = "aiflow-mini-formula-lm-relation-augmentation-probe/v1"
ROOT = Path(__file__).resolve().parents[1]
CANONICAL_ROOT = Path(r"D:\AIFlow-Workspace\Projects\Aiflow\aiflow-math-ink-1.0")
DEFAULT_CORPUS = CANONICAL_ROOT / "artifacts" / "prompt_context_corpus_20260820_r2" / "prompt_context_corpus.jsonl"
DEFAULT_AUDIT = CANONICAL_ROOT / "artifacts" / "prompt_context_corpus_20260820_r2" / "prompt_context_corpus_audit.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "hwr_failure_cause_microscope_20260928" / "mini_lm_relation_aug_20261003"
MAX_EPOCHS = 24
PATIENCE = 5
SYNTHETIC_TRAIN_FORMULAS = 2500
SYNTHETIC_VALID_FORMULAS = 400
SYNTHETIC_TEST_FORMULAS = 600
DATA_SEED = 20261003
MODEL_SEEDS = (20261003, 20261004)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _event(name: str, **values: Any) -> None:
    print(json.dumps({"event": name, **values}, ensure_ascii=False), flush=True)


class BalancedFormulaGenerator(FormulaGenerator):
    """Keep common notation while giving long-tail semantic classes real mass."""

    def __init__(self, labels: list[str], rows: list[dict], seed: int) -> None:
        super().__init__(labels, rows, seed)
        self.weights = {role: [1.0] * len(indices) for role, indices in self.pools.items()}
        # This HWR vocabulary represents the drawn fraction bar as the same
        # glyph class as minus; it has no dedicated \frac class.
        self.stacked_bar_token = self.label_to_index.get("-")
        for token in (r"\llbracket", r"\rrbracket", r"\|"):
            if token in self.label_to_index:
                pair = (self.label_to_index[token], self.label_to_index[token])
                if pair not in self.fence_pairs:
                    self.fence_pairs.append(pair)

    def _pick(self, kind: str, role: str) -> int:
        values = self.common[kind]
        if values and self.rng.random() < 0.5:
            return self.rng.choice(values)
        return self._weighted_role(role)

    def _atom(self, tokens: list[int], relations: list[str], depth: int) -> None:
        if self.stacked_bar_token is not None and depth < 2 and len(tokens) < 58 and self.rng.random() < 0.12:
            self._append(tokens, relations, self._pick("operand", "operand"))
            self._append(tokens, relations, self.stacked_bar_token, "below")
            self._append(tokens, relations, self._pick("operand", "operand"), "below")
            return
        super()._atom(tokens, relations, depth)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"expected an object on line {line_number}")
                rows.append(row)
    return rows


def _prompt_contract(corpus: Path, audit_path: Path, labels: list[str]) -> tuple[list[dict], list[dict], dict]:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    contracts = audit.get("contracts", {})
    if audit.get("schema") != "aiflow-owned-prompt-context-audit/v1":
        raise ValueError("prompt corpus audit schema mismatch")
    if (
        contracts.get("evaluation_sequence_overlap") != 0
        or contracts.get("arithmetic_evaluation_training") is not False
        or contracts.get("relation_inference_training") is not False
        or contracts.get("commercial_training_rights") is not True
    ):
        raise ValueError("project-owned prompt corpus admission contract failed")
    raw = _read_jsonl(corpus)
    if any("crohme" in json.dumps(row, ensure_ascii=False).casefold() for row in raw):
        raise ValueError("CROHME marker found in prompt training corpus")
    if any("project-owned" not in str(row.get("commercial_training_rights", "")).casefold() for row in raw):
        raise ValueError("prompt corpus row lacks project-owned training rights")
    rows, admission = _prompt_rows(corpus, labels)
    fit_rows, validation_rows = _split(rows)
    fit_ids = {str(row["formula_id"]) for row in fit_rows}
    validation_ids = {str(row["formula_id"]) for row in validation_rows}
    if fit_ids & validation_ids:
        raise AssertionError("prompt formula split leakage")
    return fit_rows, validation_rows, {
        "corpus_sha256": _sha256(corpus),
        "audit_sha256": _sha256(audit_path),
        "raw_formulas": len(raw),
        "admitted_formulas": admission["admitted_formulas"],
        "admitted_records": admission["admitted_records"],
        "fit_formulas": len(fit_ids),
        "validation_formulas": len(validation_ids),
        "fit_records": len(fit_rows),
        "validation_records": len(validation_rows),
        "fit_formula_ids_sha256": hashlib.sha256("\n".join(sorted(fit_ids)).encode()).hexdigest(),
        "validation_formula_ids_sha256": hashlib.sha256("\n".join(sorted(validation_ids)).encode()).hexdigest(),
        "admission": admission,
    }


def _prompt_masked_examples(rows: list[dict], labels: list[str], relation_count: int) -> list[dict]:
    label_to_index = {label: index for index, label in enumerate(labels)}
    relation_to_id = {relation: index for index, relation in enumerate(RELATIONS)}
    class_count = len(labels)
    pad_id = class_count + relation_count
    cls_id, sep_id, mask_id = pad_id + 1, pad_id + 2, pad_id + 3
    examples = []
    for formula_id, sequence in _formulae(rows).items():
        for target_index, row in enumerate(sequence):
            ids = [cls_id]
            target_relations = set()
            for index, current in enumerate(sequence):
                if index:
                    relation = _spatial_relation(sequence[index - 1], current)
                    ids.append(class_count + relation_to_id[relation])
                    if index == target_index or index == target_index + 1:
                        target_relations.add(relation)
                ids.append(mask_id if index == target_index else label_to_index[str(current["label"])])
            ids.append(sep_id)
            examples.append({
                "record_id": str(row["record_id"]),
                "formula_id": str(formula_id),
                "input_ids": ids,
                "mask_position": 1 + 2 * target_index,
                "target": label_to_index[str(row["label"])],
                "teacher_logp": np.zeros(class_count, dtype=np.float32),
                "target_relations": sorted(target_relations or {"right"}),
            })
    return examples


def _synthetic_formulae(
    labels: list[str], source_rows: list[dict], *, count: int, seed: int,
    split_name: str, occupied: set[tuple[int, ...]],
    all_positions: bool,
) -> tuple[list[dict], dict[str, Any]]:
    generator = BalancedFormulaGenerator(labels, source_rows, seed)
    selection_rng = random.Random(seed ^ 0x5A17)
    label_to_index = {label: index for index, label in enumerate(labels)}
    relation_to_id = {relation: index for index, relation in enumerate(RELATIONS)}
    class_count, relation_count = len(labels), len(RELATIONS)
    pad_id = class_count + relation_count
    cls_id, sep_id, mask_id = pad_id + 1, pad_id + 2, pad_id + 3
    examples: list[dict] = []
    formula_count = attempts = rejected_duplicates = 0
    formulas_with_stacked_bar = formulas_with_root = formulas_with_script = 0
    token_counts: Counter[str] = Counter()
    relation_counts: Counter[str] = Counter()
    formulas_with_relation: Counter[str] = Counter()
    formula_hashes = []
    max_attempts = max(1000, count * 8)
    while formula_count < count and attempts < max_attempts:
        attempts += 1
        token_ids, relations = generator.formula()
        if len(token_ids) < 1 or len(token_ids) > 63 or len(relations) != len(token_ids) - 1:
            raise ValueError("synthetic generator emitted a malformed sequence")
        if any(relation not in relation_to_id for relation in relations):
            raise ValueError("synthetic generator emitted a relation outside the mini-LM vocabulary")
        formula_key = tuple(int(value) for value in token_ids)
        if formula_key in occupied:
            rejected_duplicates += 1
            continue
        occupied.add(formula_key)
        identity = {"tokens": formula_key, "relations": tuple(str(value) for value in relations)}
        formula_hash = hashlib.sha256(json.dumps(identity, separators=(",", ":")).encode()).hexdigest()[:20]
        formula_id = f"{split_name}:{formula_hash}"
        formula_hashes.append(formula_hash)
        token_strings = [labels[index] for index in token_ids]
        token_counts.update(token_strings)
        relation_counts.update(relations)
        formulas_with_relation.update(set(relations))
        formulas_with_stacked_bar += int(any(
            token_strings[index] == "-"
            and relations[index - 1:index + 1] == ["below", "below"]
            for index in range(1, len(token_strings) - 1)
        ))
        formulas_with_root += int(any(token in {r"\sqrt", r"\sqrt{}"} for token in token_strings))
        formulas_with_script += int(any(relation in {"superscript", "subscript"} for relation in relations))
        target_positions = list(range(len(token_ids))) if all_positions else [selection_rng.randrange(len(token_ids))]
        for target_index in target_positions:
            ids = [cls_id]
            target_relations = set()
            for index, token_id in enumerate(token_ids):
                if index:
                    relation = relations[index - 1]
                    ids.append(class_count + relation_to_id[relation])
                    if index == target_index or index == target_index + 1:
                        target_relations.add(relation)
                ids.append(mask_id if index == target_index else int(token_id))
            ids.append(sep_id)
            if len(ids) > MAX_POSITIONS:
                raise ValueError(f"synthetic sequence exceeds position contract: {len(ids)}")
            examples.append({
                "record_id": f"{formula_id}:{target_index}",
                "formula_id": formula_id,
                "input_ids": ids,
                "mask_position": 1 + 2 * target_index,
                "target": int(token_ids[target_index]),
                "teacher_logp": np.zeros(class_count, dtype=np.float32),
                "target_relations": sorted(target_relations or {"right"}),
            })
        formula_count += 1
    if formula_count != count:
        raise RuntimeError(f"generated only {formula_count}/{count} unique {split_name} formulas")
    return examples, {
        "formulas": formula_count,
        "masked_examples": len(examples),
        "generation_attempts": attempts,
        "rejected_duplicate_formulas": rejected_duplicates,
        "unique_formula_hashes_sha256": hashlib.sha256("\n".join(sorted(formula_hashes)).encode()).hexdigest(),
        "distinct_labels": len(token_counts),
        "label_count": len(labels),
        "label_coverage_rate": len(token_counts) / len(labels),
        "missing_label_count": len(set(labels) - set(token_counts)),
        "missing_labels": sorted(set(labels) - set(token_counts)),
        "relation_edge_counts": dict(sorted(relation_counts.items())),
        "formulas_with_relation_type": dict(sorted(formulas_with_relation.items())),
        "distinct_relation_types": len(formulas_with_relation),
        "formulas_with_stacked_bar_template": formulas_with_stacked_bar,
        "formulas_with_root_token": formulas_with_root,
        "formulas_with_subscript_or_superscript": formulas_with_script,
    }


def _metrics(model: MiniFormulaLM, examples: list[dict], labels: list[str], device: torch.device) -> dict[str, Any]:
    pack = _pack(examples, model)
    model.eval()
    predictions: list[int] = []
    with torch.inference_mode():
        for start in range(0, len(examples), 256):
            stop = min(start + 256, len(examples))
            logits = model(
                pack["input_ids"][start:stop].to(device),
                pack["attention"][start:stop].to(device),
                pack["mask_positions"][start:stop].to(device),
            )
            predictions.extend(logits.argmax(dim=-1).cpu().tolist())
    targets = pack["targets"].tolist()
    by_formula: dict[str, list[tuple[int, int]]] = defaultdict(list)
    by_relation: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for example, prediction, target in zip(examples, predictions, targets, strict=True):
        by_formula[str(example["formula_id"])].append((prediction, target))
        for relation in example["target_relations"]:
            by_relation[str(relation)].append((prediction, target))
    logits_loss = 0.0
    weight_sum = 0.0
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(examples), 256):
            stop = min(start + 256, len(examples))
            logits = model(
                pack["input_ids"][start:stop].to(device),
                pack["attention"][start:stop].to(device),
                pack["mask_positions"][start:stop].to(device),
            )
            losses = F.cross_entropy(logits, pack["targets"][start:stop].to(device), reduction="none")
            weights = pack["weights"][start:stop].to(device)
            logits_loss += float((losses * weights).sum())
            weight_sum += float(weights.sum())
    return {
        "context_mode": "each target masked separately with all other symbols kept at ground truth",
        "masked_examples": len(examples),
        "formulas": len(by_formula),
        "masked_symbol_top1": sum(prediction == target for prediction, target in zip(predictions, targets, strict=True)) / max(1, len(targets)),
        "formula_exact": sum(all(prediction == target for prediction, target in values) for values in by_formula.values()) / max(1, len(by_formula)),
        "weighted_ce": logits_loss / max(weight_sum, 1e-12),
        "accuracy_by_target_context_relation": {
            relation: {
                "examples": len(values),
                "top1": sum(prediction == target for prediction, target in values) / len(values),
            }
            for relation, values in sorted(by_relation.items())
        },
    }


def _fit_arm(
    train_examples: list[dict], prompt_validation: list[dict], synthetic_validation: list[dict],
    labels: list[str], *, seed: int, device: torch.device,
) -> tuple[MiniFormulaLM, dict[str, Any]]:
    _seed_everything(seed)
    model = MiniFormulaLM(len(labels), len(RELATIONS), MAX_POSITIONS).to(device)
    train_pack = _pack(train_examples, model)
    prompt_validation_pack = _pack(prompt_validation, model)
    synthetic_validation_pack = _pack(synthetic_validation, model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    best_score, best_epoch, stale, best_state = float("inf"), 0, 0, None
    history = []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        order = torch.randperm(len(train_examples))
        loss_sum = 0.0
        for start in range(0, len(order), BATCH_SIZE):
            indices = order[start:start + BATCH_SIZE]
            batch_indices = indices.to("cpu")
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                train_pack["input_ids"][batch_indices].to(device),
                train_pack["attention"][batch_indices].to(device),
                train_pack["mask_positions"][batch_indices].to(device),
            )
            losses = F.cross_entropy(logits, train_pack["targets"][batch_indices].to(device), reduction="none")
            weights = train_pack["weights"][batch_indices].to(device)
            loss = (losses * weights).sum() / weights.sum()
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite relation-augmentation CE loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(indices)

        prompt_metric = _metrics(model, prompt_validation, labels, device)
        synthetic_metric = _metrics(model, synthetic_validation, labels, device)
        score = 0.5 * (prompt_metric["weighted_ce"] + synthetic_metric["weighted_ce"])
        history.append({
            "epoch": epoch,
            "train_weighted_ce": loss_sum / len(train_examples),
            "selection_score_equal_prompt_and_synthetic_ce": score,
            "prompt_validation": prompt_metric,
            "synthetic_validation": synthetic_metric,
        })
        if score < best_score - 1e-4:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        _event("mini_lm_relation_augmentation_epoch", seed=seed, epoch=epoch, max_epochs=MAX_EPOCHS, train_weighted_ce=round(loss_sum / len(train_examples), 5), prompt_val_top1=round(prompt_metric["masked_symbol_top1"], 5), synthetic_val_top1=round(synthetic_metric["masked_symbol_top1"], 5), selection_score=round(score, 5))
        if epoch >= 5 and stale >= PATIENCE:
            break
    if best_state is None:
        raise AssertionError("validation did not select a mini-LM checkpoint")
    model.load_state_dict(best_state, strict=True)
    model.eval()
    return model, {"epochs_run": len(history), "selected_epoch": best_epoch, "best_selection_score": best_score, "history": history}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--prompt-audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--synthetic-train-formulas", type=int, default=SYNTHETIC_TRAIN_FORMULAS)
    parser.add_argument(
        "--synthetic-train-mask-policy",
        choices=("single_per_formula", "all_positions"),
        default="single_per_formula",
        help="Number of masked-token training examples drawn from each generated formula.",
    )
    parser.add_argument("--synthetic-valid-formulas", type=int, default=SYNTHETIC_VALID_FORMULAS)
    parser.add_argument("--synthetic-test-formulas", type=int, default=SYNTHETIC_TEST_FORMULAS)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(MODEL_SEEDS))
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    if min(args.synthetic_train_formulas, args.synthetic_valid_formulas, args.synthetic_test_formulas) < 1:
        parser.error("synthetic formula counts must be positive")
    if not args.seeds:
        parser.error("at least one model seed is required")
    corpus, audit_path = args.prompt_corpus.resolve(), args.prompt_audit.resolve()
    base_checkpoint = BASE_STUDENT_DIR / "mini_formula_lm.pt"
    base_report_path = BASE_STUDENT_DIR / "mini_formula_lm_distillation_report.json"
    if not base_checkpoint.is_file() or not base_report_path.is_file():
        parser.error("frozen mini-LM label contract/report is missing")
    base_report = json.loads(base_report_path.read_text(encoding="utf-8"))
    if base_report.get("protocol", {}).get("crohme_rows_loaded") != 0 or base_report.get("protocol", {}).get("student_trained_only_on_project_owned_prompt_corpus") is not True:
        raise ValueError("frozen mini-LM does not attest owned prompt-only training and CROHME exclusion")
    base_payload = torch.load(base_checkpoint, map_location="cpu", weights_only=True)
    labels = [str(value) for value in base_payload.get("labels", [])]
    if base_payload.get("relations") != list(RELATIONS):
        raise ValueError("frozen mini-LM relation vocabulary mismatch")
    prompt_fit_rows, prompt_validation_rows, source_meta = _prompt_contract(corpus, audit_path, labels)
    if len(labels) != 372:
        raise ValueError(f"expected the existing 372-class contract, found {len(labels)}")
    if max(2 * int(row["context"]["length"]) + 1 for row in prompt_fit_rows + prompt_validation_rows) > MAX_POSITIONS:
        raise ValueError("prompt formula exceeds mini-LM position contract")

    label_to_index = {label: index for index, label in enumerate(labels)}
    prompt_formula_keys = {
        tuple(label_to_index[str(row["label"])] for row in sequence)
        for sequence in _formulae(prompt_fit_rows + prompt_validation_rows).values()
    }
    occupied: set[tuple[int, ...]] = set(prompt_formula_keys)
    synthetic_train, train_meta = _synthetic_formulae(
        labels, prompt_fit_rows, count=args.synthetic_train_formulas, seed=DATA_SEED,
        split_name="synthetic_train", occupied=occupied,
        all_positions=args.synthetic_train_mask_policy == "all_positions",
    )
    if args.synthetic_train_mask_policy == "all_positions" and train_meta["masked_examples"] <= train_meta["formulas"]:
        raise AssertionError("all-position synthetic masking must yield more than one target per formula")
    synthetic_validation, validation_meta = _synthetic_formulae(
        labels, prompt_fit_rows, count=args.synthetic_valid_formulas, seed=DATA_SEED + 1,
        split_name="synthetic_validation", occupied=occupied, all_positions=True,
    )
    synthetic_test, test_meta = _synthetic_formulae(
        labels, prompt_fit_rows, count=args.synthetic_test_formulas, seed=DATA_SEED + 2,
        split_name="synthetic_test", occupied=occupied, all_positions=True,
    )
    if len(occupied) != len(prompt_formula_keys) + args.synthetic_train_formulas + args.synthetic_valid_formulas + args.synthetic_test_formulas:
        raise AssertionError("synthetic train/validation/test formula overlap")
    prompt_train_examples = _prompt_masked_examples(prompt_fit_rows, labels, len(RELATIONS))
    prompt_validation_examples = _prompt_masked_examples(prompt_validation_rows, labels, len(RELATIONS))
    if not prompt_train_examples or not prompt_validation_examples:
        raise ValueError("empty prompt train or validation set")
    # The flat prompt corpus is expected to train only the rightward relation.
    prompt_relations = Counter(
        relation
        for example in prompt_train_examples
        for relation in example["target_relations"]
    )
    if set(prompt_relations) - {"right"}:
        raise ValueError("prompt relation baseline is no longer flat; rerun the relation-distribution audit")
    prompt_edge_counts = Counter(
        _spatial_relation(sequence[index - 1], sequence[index])
        for sequence in _formulae(prompt_fit_rows).values()
        for index in range(1, len(sequence))
    )

    if args.preflight_only:
        print(json.dumps({
            "event": "mini_lm_relation_augmentation_preflight",
            "prompt_fit_formulas": source_meta["fit_formulas"],
            "prompt_validation_formulas": source_meta["validation_formulas"],
            "prompt_fit_relation_edges": dict(sorted(prompt_edge_counts.items())),
            "synthetic_train": train_meta,
            "synthetic_validation": validation_meta,
            "synthetic_test": test_meta,
            "prompt_formula_overlap": 0,
            "synthetic_split_overlap": 0,
            "crohme_rows": 0,
        }, ensure_ascii=False))
        return 0

    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    if device_name == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    device = torch.device(device_name)
    candidate_train = prompt_train_examples + synthetic_train
    _event("mini_lm_relation_augmentation_start", device=device_name, prompt_fit_examples=len(prompt_train_examples), synthetic_train_formulas=train_meta["formulas"], synthetic_train_examples=len(synthetic_train), synthetic_train_mask_policy=args.synthetic_train_mask_policy, prompt_validation_formulas=source_meta["validation_formulas"], synthetic_validation_formulas=validation_meta["formulas"], synthetic_test_formulas=test_meta["formulas"], crohme_rows=0)

    arms = {"prompt_only": {}, "prompt_plus_synthetic_2d": {}}
    checkpoint_payloads = {}
    for seed in args.seeds:
        for arm_name, train_examples in (("prompt_only", prompt_train_examples), ("prompt_plus_synthetic_2d", candidate_train)):
            started = time.perf_counter()
            model, fit_meta = _fit_arm(
                train_examples, prompt_validation_examples, synthetic_validation,
                labels, seed=int(seed), device=device,
            )
            metrics = {
                "prompt_validation": _metrics(model, prompt_validation_examples, labels, device),
                "synthetic_validation": _metrics(model, synthetic_validation, labels, device),
                "synthetic_test": _metrics(model, synthetic_test, labels, device),
            }
            key = f"{arm_name}_seed{int(seed)}"
            arms[arm_name][str(seed)] = {
                "fit": fit_meta,
                "metrics": metrics,
                "training_examples": len(train_examples),
                "elapsed_seconds": time.perf_counter() - started,
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "test_relation_rows": metrics["synthetic_test"]["accuracy_by_target_context_relation"],
            }
            checkpoint_payloads[key] = {
                "schema": SCHEMA,
                "arm": arm_name,
                "seed": int(seed),
                "labels": labels,
                "relations": list(RELATIONS),
                "hidden": HIDDEN,
                "layers": LAYERS,
                "heads": HEADS,
                "feedforward": FEEDFORWARD,
                "max_positions": MAX_POSITIONS,
                "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            }
            _event("mini_lm_relation_augmentation_arm_complete", arm=arm_name, seed=int(seed), prompt_validation_top1=metrics["prompt_validation"]["masked_symbol_top1"], synthetic_test_top1=metrics["synthetic_test"]["masked_symbol_top1"], synthetic_test_formula_exact=metrics["synthetic_test"]["formula_exact"], elapsed_seconds=round(arms[arm_name][str(seed)]["elapsed_seconds"], 2), crohme_rows=0)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    per_seed = {}
    for seed in args.seeds:
        base = arms["prompt_only"][str(seed)]["metrics"]
        candidate = arms["prompt_plus_synthetic_2d"][str(seed)]["metrics"]
        per_seed[str(seed)] = {
            "prompt_validation_top1_delta_pp": 100.0 * (candidate["prompt_validation"]["masked_symbol_top1"] - base["prompt_validation"]["masked_symbol_top1"]),
            "prompt_validation_weighted_ce_delta": candidate["prompt_validation"]["weighted_ce"] - base["prompt_validation"]["weighted_ce"],
            "synthetic_test_top1_delta_pp": 100.0 * (candidate["synthetic_test"]["masked_symbol_top1"] - base["synthetic_test"]["masked_symbol_top1"]),
            "synthetic_test_formula_exact_delta_pp": 100.0 * (candidate["synthetic_test"]["formula_exact"] - base["synthetic_test"]["formula_exact"]),
            "candidate_test_relation_accuracy": candidate["synthetic_test"]["accuracy_by_target_context_relation"],
            "baseline_test_relation_accuracy": base["synthetic_test"]["accuracy_by_target_context_relation"],
        }

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    checkpoint_meta = {}
    for key, payload in checkpoint_payloads.items():
        path = output / f"{key}.pt"
        torch.save(payload, path)
        checkpoint_meta[key] = {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
    report = {
        "schema": SCHEMA,
        "status": "exploratory_synthetic_relation_ablation_not_acceptance",
        "protocol": {
            "crohme_rows_loaded": 0,
            "crohme_training_or_tuning": False,
            "consumed_149_formula_set_loaded": False,
            "project_owned_prompt_corpus_only_for_real_examples": True,
            "synthetic_generator": "balanced subclass of train_owned_formula_context_v1.FormulaGenerator; uniform within semantic roles and a 0.5 common-token mix",
            "synthetic_train_mask_policy": args.synthetic_train_mask_policy,
            "synthetic_train_masks_per_formula": "all eligible token positions" if args.synthetic_train_mask_policy == "all_positions" else "one seeded token position",
            "training_objective": "masked-symbol cross-entropy; no teacher and no external pretrained context weights",
            "early_stopping": "equal mean weighted cross-entropy on held-out project-owned prompt formulas and a separate generated relation split",
            "synthetic_test_used_for_selection": False,
            "handwritten_acceptance_evidence": False,
            "product_adopted": False,
            "synthetic_structure_note": "A stacked fraction-like template uses the existing '-' HWR glyph class with adjacent below relations; the vocabulary has no dedicated \\frac class, and this chain is only a coarse proxy for a 2-D relation graph.",
            "root_structure_note": "Root tokens are generated, but the frozen mini-LM relation vocabulary has no contains edge; root/radicand containment is not tested by this probe.",
            "warning": "This probes 2-D relation representation on generated contexts only; it is not evidence of handwritten formula improvement or independent product acceptance.",
        },
        "data": {
            **source_meta,
            "base_label_contract_checkpoint_sha256": _sha256(base_checkpoint),
            "base_distillation_report_sha256": _sha256(base_report_path),
            "crohme_rows": 0,
            "prompt_training_relation_edge_counts": dict(sorted(prompt_edge_counts.items())),
            "prompt_training_target_context_relation_counts": dict(sorted(prompt_relations.items())),
            "synthetic_train": train_meta,
            "synthetic_validation": validation_meta,
            "synthetic_test": test_meta,
            "probe_source_sha256": _sha256(Path(__file__).resolve()),
            "generator_source_sha256": _sha256(ROOT / "scripts" / "train_owned_formula_context_v1.py"),
            "cross_split_formula_overlap": 0,
            "synthetic_prompt_formula_overlap": 0,
        },
        "architecture": {
            "layers": LAYERS, "hidden": HIDDEN, "heads": HEADS,
            "feedforward": FEEDFORWARD, "max_positions": MAX_POSITIONS,
            "parameters": sum(parameter.numel() for parameter in MiniFormulaLM(len(labels), len(RELATIONS), MAX_POSITIONS).parameters()),
            "output_classes": len(labels), "relations": list(RELATIONS),
        },
        "arms": arms,
        "paired_by_seed": per_seed,
        "checkpoints": checkpoint_meta,
        "decision": {
            "handwritten_performance_claim": False,
            "product_promotion_eligible": False,
            "next_required_gate": "independent project-owned formula cohort from unseen writers; retain candidate-preserving inference and then measure on-device latency",
        },
    }
    report_path = output / "relation_augmentation_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    _event("mini_lm_relation_augmentation_complete", report=str(report_path), output=str(output), seeds=list(args.seeds), prompt_only_vs_augmented_test_deltas={seed: values["synthetic_test_top1_delta_pp"] for seed, values in per_seed.items()}, crohme_rows=0, product_adopted=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
