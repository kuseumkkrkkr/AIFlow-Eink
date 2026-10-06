#!/usr/bin/env python3
"""Test a tiny learned token-embedding network over frozen HWR Top-5 candidates."""

from __future__ import annotations

import argparse
import copy
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from character_tensor_v1 import ROOT, _json_lines
from evaluate_homograph_context_reranker_v1 import FAMILIES, _fold, _metrics, _options, _role


SCHEMA = "aiflow-mini-embedding-reranker-evaluation/v1"
SEEDS = (17, 31, 47)
EMBEDDING_DIM = 8
HIDDEN_DIM = 32
LEARNING_RATE = 0.01
WEIGHT_DECAY = 1e-3
MAX_EPOCHS = 200
PATIENCE = 25
MARGIN_THRESHOLDS = (None, 4.0, 2.0, 1.0, 0.5, 0.25, 0.0)
DEFAULT_INPUT = ROOT / "artifacts" / "homograph_context_20260814"
DEFAULT_LOGISTIC_REPORT = DEFAULT_INPUT / "homograph_context_report.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "mini_embedding_context_20260814"
SPECIAL_TOKENS = ("<UNK>", "<S>", "</S>")
ROLES = ("digit", "operator", "fence", "operand", "boundary", "other")
NUMERIC_DIM = 8 + len(ROLES) * 3 + len(FAMILIES)


class MiniEmbeddingRanker(nn.Module):
    def __init__(self, vocabulary_size: int) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocabulary_size, EMBEDDING_DIM)
        self.mlp = nn.Sequential(
            nn.Linear(EMBEDDING_DIM * 3 + NUMERIC_DIM, HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(HIDDEN_DIM, 1),
        )

    def forward(
        self,
        candidates: torch.Tensor,
        previous: torch.Tensor,
        following: torch.Tensor,
        numeric: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        option_count = candidates.shape[1]
        previous_embedding = self.token_embedding(previous).unsqueeze(1).expand(-1, option_count, -1)
        following_embedding = self.token_embedding(following).unsqueeze(1).expand(-1, option_count, -1)
        features = torch.cat((self.token_embedding(candidates), previous_embedding, following_embedding, numeric), dim=2)
        return self.mlp(features).squeeze(2).masked_fill(~mask, -torch.inf)


def _d_path(path: Path, kind: str) -> Path:
    resolved = path.resolve()
    if resolved.drive.upper() != "D:":
        raise ValueError(f"{kind} must remain on D:: {resolved}")
    return resolved


def _vocabulary(rows: list[dict]) -> dict[str, int]:
    tokens = set(SPECIAL_TOKENS)
    for row in rows:
        tokens.update(row["final_topk"])
        tokens.add(row["context"]["previous_top1"])
        tokens.add(row["context"]["next_top1"])
    ordered = list(SPECIAL_TOKENS) + sorted(tokens - set(SPECIAL_TOKENS))
    return {token: index for index, token in enumerate(ordered)}


def _one_hot(value: str, values: tuple[str, ...]) -> list[float]:
    return [float(value == candidate) for candidate in values]


def _numeric(row: dict, token: str, rank: int, probability: float) -> list[float]:
    context = row["context"]
    previous_role = _role(context["previous_top1"])
    next_role = _role(context["next_top1"])
    family = next(name for name, members in FAMILIES.items() if token in members)
    top_probability = row["final_topk_probabilities"][0]
    values = [
        rank / 4.0,
        probability,
        max(math.log(max(probability, 1e-12)), -12.0) / 12.0,
        probability - top_probability,
        context["index"] / max(context["length"] - 1, 1),
        min(math.log1p(context["length"]) / 5.0, 1.0),
        float(previous_role == next_role == "digit"),
        float(previous_role == next_role == "operand"),
    ]
    values += _one_hot(_role(token), ROLES) + _one_hot(previous_role, ROLES) + _one_hot(next_role, ROLES)
    values += _one_hot(family, tuple(FAMILIES))
    if len(values) != NUMERIC_DIM:
        raise AssertionError("mini-embedding numeric feature contract changed")
    return values


def _pack(rows: list[dict], vocabulary: dict[str, int], *, require_truth: bool) -> dict | None:
    selected = []
    for row in rows:
        options = _options(row)
        if len(options) < 2 or require_truth and row["label"] not in {token for token, _, _ in options}:
            continue
        selected.append((row, options))
    if not selected:
        return None
    width = max(len(options) for _, options in selected)
    unknown = vocabulary["<UNK>"]
    candidates = np.full((len(selected), width), unknown, dtype=np.int64)
    previous = np.empty(len(selected), dtype=np.int64)
    following = np.empty(len(selected), dtype=np.int64)
    numeric = np.zeros((len(selected), width, NUMERIC_DIM), dtype=np.float32)
    mask = np.zeros((len(selected), width), dtype=bool)
    targets = np.full(len(selected), -1, dtype=np.int64)
    for row_index, (row, options) in enumerate(selected):
        previous[row_index] = vocabulary.get(row["context"]["previous_top1"], unknown)
        following[row_index] = vocabulary.get(row["context"]["next_top1"], unknown)
        for option_index, (token, rank, probability) in enumerate(options):
            candidates[row_index, option_index] = vocabulary.get(token, unknown)
            numeric[row_index, option_index] = _numeric(row, token, rank, probability)
            mask[row_index, option_index] = True
            if token == row["label"]:
                targets[row_index] = option_index
    if require_truth and (targets < 0).any():
        raise AssertionError("training pack contains a row without its truth candidate")
    return {
        "rows": [row for row, _ in selected], "options": [options for _, options in selected],
        "candidates": torch.from_numpy(candidates), "previous": torch.from_numpy(previous),
        "following": torch.from_numpy(following), "numeric": torch.from_numpy(numeric),
        "mask": torch.from_numpy(mask), "targets": torch.from_numpy(targets),
    }


def _to_device(pack: dict, device: torch.device) -> dict:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in pack.items()}


def _row_weights(pack: dict, device: torch.device) -> torch.Tensor:
    counts = Counter(row["label"] for row in pack["rows"])
    weights = [len(pack["rows"]) / (len(counts) * counts[row["label"]]) for row in pack["rows"]]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _loss(model: MiniEmbeddingRanker, pack: dict, weights: torch.Tensor) -> torch.Tensor:
    scores = model(pack["candidates"], pack["previous"], pack["following"], pack["numeric"], pack["mask"])
    losses = F.cross_entropy(scores, pack["targets"], reduction="none")
    return (losses * weights).sum() / weights.sum()


def _choices(scores: torch.Tensor, threshold: float | None) -> torch.Tensor:
    if threshold is None:
        return torch.zeros(len(scores), dtype=torch.long, device=scores.device)
    best = scores.argmax(dim=1)
    margin = scores.gather(1, best[:, None]).squeeze(1) - scores[:, 0]
    return torch.where(margin >= threshold, best, torch.zeros_like(best))


def _choice_score(pack: dict, choices: torch.Tensor) -> tuple[float, float, int]:
    predicted = [options[int(choice)][0] for options, choice in zip(pack["options"], choices.cpu().tolist(), strict=True)]
    labels = sorted({row["label"] for row in pack["rows"]})
    hits = [prediction == row["label"] for prediction, row in zip(predicted, pack["rows"], strict=True)]
    micro = sum(hits) / len(hits)
    macro = sum(
        sum(hit for hit, row in zip(hits, pack["rows"], strict=True) if row["label"] == label)
        / sum(row["label"] == label for row in pack["rows"])
        for label in labels
    ) / len(labels)
    changes = sum(choice != 0 for choice in choices.cpu().tolist())
    return macro, micro, changes


@torch.inference_mode()
def _select_threshold(model: MiniEmbeddingRanker, validation: dict) -> tuple[float | None, list[dict]]:
    model.eval()
    scores = model(validation["candidates"], validation["previous"], validation["following"], validation["numeric"], validation["mask"])
    trials = []
    for threshold in MARGIN_THRESHOLDS:
        macro, micro, changes = _choice_score(validation, _choices(scores, threshold))
        trials.append({"threshold": threshold, "strict_macro": macro, "strict_micro": micro, "changes": changes})
    best = max(
        trials,
        key=lambda row: (row["strict_macro"], row["strict_micro"], -row["changes"], math.inf if row["threshold"] is None else row["threshold"]),
    )
    return best["threshold"], trials


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _select_epochs(rows: list[dict], vocabulary: dict[str, int], device: torch.device, seed: int, salt: str) -> dict:
    fit_rows = [row for row in rows if _fold(row["formula_id"], 5, salt) != 0]
    validation_rows = [row for row in rows if _fold(row["formula_id"], 5, salt) == 0]
    fit = _pack(fit_rows, vocabulary, require_truth=True)
    validation = _pack(validation_rows, vocabulary, require_truth=True)
    if fit is None or validation is None or len(validation["rows"]) < 2:
        return {"epochs": 60, "fit_records": len(fit["rows"]) if fit else 0, "validation_records": len(validation["rows"]) if validation else 0, "fallback": True}
    fit, validation = _to_device(fit, device), _to_device(validation, device)
    _set_seed(seed)
    model = MiniEmbeddingRanker(len(vocabulary)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    fit_weights, validation_weights = _row_weights(fit, device), _row_weights(validation, device)
    best_loss, best_epoch, best_state, stale = math.inf, 1, copy.deepcopy(model.state_dict()), 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        loss = _loss(model, fit, fit_weights)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite mini-embedding training loss")
        loss.backward(); optimizer.step()
        model.eval()
        with torch.inference_mode():
            validation_loss = float(_loss(model, validation, validation_weights))
        if validation_loss < best_loss - 1e-5:
            best_loss, best_epoch, best_state, stale = validation_loss, epoch, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if epoch >= 20 and stale >= PATIENCE:
            break
    model.load_state_dict(best_state)
    threshold, threshold_trials = _select_threshold(model, validation)
    return {
        "epochs": best_epoch, "fit_records": len(fit["rows"]), "validation_records": len(validation["rows"]),
        "validation_loss": best_loss, "searched_epochs": epoch, "threshold": threshold,
        "threshold_trials": threshold_trials, "fallback": False,
    }


def _fit(rows: list[dict], device: torch.device, seed: int, salt: str) -> tuple[MiniEmbeddingRanker, dict[str, int], dict] | None:
    vocabulary = _vocabulary(rows)
    packed = _pack(rows, vocabulary, require_truth=True)
    if packed is None:
        return None
    selection = _select_epochs(rows, vocabulary, device, seed, salt)
    packed = _to_device(packed, device)
    _set_seed(seed)
    model = MiniEmbeddingRanker(len(vocabulary)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    weights = _row_weights(packed, device)
    for _ in range(selection["epochs"]):
        model.train(); optimizer.zero_grad(set_to_none=True)
        loss = _loss(model, packed, weights)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite mini-embedding final loss")
        loss.backward(); optimizer.step()
    model.eval()
    metadata = selection | {
        "training_records": len(packed["rows"]), "vocabulary": len(vocabulary),
        "parameters": sum(parameter.numel() for parameter in model.parameters()), "final_loss": float(loss.detach()),
    }
    return model, vocabulary, metadata


@torch.inference_mode()
def _apply(bundle, rows: list[dict], device: torch.device) -> dict[str, str]:
    output = {row["record_id"]: row["final_topk"][0] for row in rows}
    if bundle is None:
        return output
    model, vocabulary, _ = bundle
    packed = _pack(rows, vocabulary, require_truth=False)
    if packed is None:
        return output
    packed = _to_device(packed, device)
    scores = model(packed["candidates"], packed["previous"], packed["following"], packed["numeric"], packed["mask"])
    choices = _choices(scores, bundle[2].get("threshold", 0.0)).cpu().tolist()
    for row, options, choice in zip(packed["rows"], packed["options"], choices, strict=True):
        output[row["record_id"]] = options[choice][0]
    return output


def _cross_validated(rows: list[dict], scope: str, device: torch.device, seed: int) -> tuple[dict[str, str], list[dict]]:
    if scope == "direct":
        groups = sorted({row["writer_group"] for row in rows})
        held = lambda row, group: row["writer_group"] == group
    else:
        groups = list(range(5))
        held = lambda row, group: _fold(row["formula_id"], 5, "outer-crohme") == group
    predictions, folds = {}, []
    for fold_index, group in enumerate(groups):
        train_rows = [row for row in rows if not held(row, group)]
        held_rows = [row for row in rows if held(row, group)]
        if {row["formula_id"] for row in train_rows} & {row["formula_id"] for row in held_rows}:
            raise AssertionError("formula leakage across mini-embedding fold")
        fold_seed = seed + fold_index * 1000
        bundle = _fit(train_rows, device, fold_seed, f"mini-validation-{scope}-{group}-{seed}")
        predictions.update(_apply(bundle, held_rows, device))
        folds.append({
            "held_group": str(group), "train_formulas": len({row["formula_id"] for row in train_rows}),
            "held_formulas": len({row["formula_id"] for row in held_rows}), "seed": fold_seed,
            "fit": bundle[2] if bundle else {"training_records": 0},
        })
    if set(predictions) != {row["record_id"] for row in rows}:
        raise AssertionError("mini-embedding cross-validation coverage mismatch")
    return predictions, folds


def _consensus(rows: list[dict], predictions: list[dict[str, str]]) -> tuple[dict[str, str], dict]:
    output, unanimous = {}, 0
    for row in rows:
        record_id = row["record_id"]
        values = [prediction[record_id] for prediction in predictions]
        counts = Counter(values)
        maximum = max(counts.values())
        winners = [token for token, count in counts.items() if count == maximum]
        output[record_id] = winners[0] if len(winners) == 1 else row["final_topk"][0]
        unanimous += len(counts) == 1
    eligible = [row for row in rows if len(_options(row)) >= 2]
    eligible_unanimous = sum(len({prediction[row["record_id"]] for prediction in predictions}) == 1 for row in eligible)
    return output, {
        "all_unanimous": unanimous / len(rows),
        "eligible_unanimous": eligible_unanimous / len(eligible) if eligible else None,
    }


def _aggregate(seed_results: list[dict]) -> dict:
    keys = ("all_top1", "strict_micro_top1", "strict_macro_top1", "formula_exact", "changed", "improved", "regressed")
    return {
        key: {
            "mean": float(np.mean([result["metrics"][key] for result in seed_results])),
            "min": float(np.min([result["metrics"][key] for result in seed_results])),
            "max": float(np.max([result["metrics"][key] for result in seed_results])),
        }
        for key in keys
    }


def _scope_report(rows: list[dict], scope: str, device: torch.device, logistic: dict) -> dict:
    seed_results, seed_predictions = [], []
    for seed in SEEDS:
        predictions, folds = _cross_validated(rows, scope, device, seed)
        seed_predictions.append(predictions)
        seed_results.append({"seed": seed, "metrics": _metrics(rows, predictions), "folds": folds})
    consensus, stability = _consensus(rows, seed_predictions)
    baseline_predictions = {row["record_id"]: row["final_topk"][0] for row in rows}
    baseline = _metrics(rows, baseline_predictions)
    consensus_metrics = _metrics(rows, consensus)
    reported_metrics = ("all_top1", "strict_micro_top1", "strict_macro_top1", "formula_exact")
    return {
        "baseline": baseline,
        "logistic_context_reference": logistic,
        "seeds": seed_results,
        "seed_aggregate": _aggregate(seed_results),
        "consensus": {"metrics": consensus_metrics, "stability": stability},
        "hypothesis": {
            "improves_frozen_baseline_all_reported_metrics": all(consensus_metrics[key] > baseline[key] for key in reported_metrics),
            "dominates_logistic_context": (
                all(consensus_metrics[key] >= logistic[key] for key in reported_metrics)
                and consensus_metrics["regressed"] <= logistic["regressed"]
            ),
            "delta_from_logistic_context": {key: consensus_metrics[key] - logistic[key] for key in reported_metrics},
        },
    }


def _self_test() -> None:
    rows = [{
        "record_id": "r", "label": "1", "formula_id": "f", "final_topk": ["|", "1", "7", "I", "/"],
        "final_topk_probabilities": [0.5, 0.3, 0.1, 0.06, 0.04],
        "context": {"index": 1, "length": 3, "previous_top1": "2", "next_top1": "+"},
    }]
    vocabulary = _vocabulary(rows)
    packed = _pack(rows, vocabulary, require_truth=True)
    assert packed is not None and packed["candidates"].shape == (1, 4)
    model = MiniEmbeddingRanker(len(vocabulary))
    scores = model(packed["candidates"], packed["previous"], packed["following"], packed["numeric"], packed["mask"])
    assert scores.shape == (1, 4) and sum(parameter.numel() for parameter in model.parameters()) < 20_000
    consensus, _ = _consensus(rows, [{"r": "1"}, {"r": "1"}, {"r": "|"}])
    assert consensus == {"r": "1"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--logistic-report", type=Path, default=DEFAULT_LOGISTIC_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test(); print(json.dumps({"self_test": "pass"})); return 0
    input_root = _d_path(args.input, "candidate input")
    output = _d_path(args.output, "output")
    report_path = output / "mini_embedding_report.json"
    if report_path.exists():
        parser.error(f"refusing to overwrite report: {report_path}")
    direct_path, crohme_path = input_root / "direct_candidates.jsonl.gz", input_root / "crohme_candidates.jsonl.gz"
    if not direct_path.is_file() or not crohme_path.is_file():
        parser.error("candidate caches missing; run evaluate_homograph_context_reranker_v1.py first")
    logistic_report = json.loads(args.logistic_report.read_text(encoding="utf-8"))
    direct_rows, crohme_rows = list(_json_lines(direct_path)), list(_json_lines(crohme_path))
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    direct_evaluation = _scope_report(direct_rows, "direct", device, logistic_report["direct"]["evaluation"]["variants"]["context"]["metrics"])
    crohme_evaluation = _scope_report(crohme_rows, "crohme", device, logistic_report["crohme"]["evaluation"]["variants"]["context"]["metrics"])
    report = {
        "schema": SCHEMA, "generated_at": datetime.now(timezone.utc).isoformat(),
        "hypothesis": "a shared 8-dimensional token embedding plus a 32-unit MLP can improve frozen Top-5 homograph ranking",
        "architecture": {
            "shared_token_embedding": EMBEDDING_DIM, "hidden_units": HIDDEN_DIM, "external_pretraining": False,
            "inputs": "candidate, previous Top-1, next Top-1, frozen emission, token roles, formula position",
            "candidate_policy": "same-family Top-5 reorder only; no token invention or grouping mutation",
        },
        "training": {
            "seeds": SEEDS, "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
            "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
            "margin_thresholds": MARGIN_THRESHOLDS,
            "epoch_selection": "outer-train-only formula-disjoint validation; final refit on all outer training formulas",
        },
        "direct": {
            "scope": "project-owned three-writer LOO; no O, o, or vertical-bar truth",
            "evaluation": direct_evaluation,
        },
        "crohme": {
            "scope": "CC BY-NC research-only five-fold formula-disjoint; never a product model",
            "evaluation": crohme_evaluation,
        },
        "decision": {
            "mini_embedding_viable_over_frozen_head": all(
                evaluation["hypothesis"]["improves_frozen_baseline_all_reported_metrics"]
                for evaluation in (direct_evaluation, crohme_evaluation)
            ),
            "replace_logistic_context_reranker": all(
                evaluation["hypothesis"]["dominates_logistic_context"]
                for evaluation in (direct_evaluation, crohme_evaluation)
            ),
        },
        "limitations": [
            "The evaluation datasets were already inspected in prior reranker development and are not untouched final acceptance sets.",
            "Only truth-group character ranking is tested; grouping, spatial relations, and formula decoding remain outside this experiment.",
            "A neural gain on CROHME cannot authorize commercial training or deployment.",
        ],
        "product_adopted": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"event": "mini_embedding_complete", "output": str(report_path), "product_adopted": False}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
