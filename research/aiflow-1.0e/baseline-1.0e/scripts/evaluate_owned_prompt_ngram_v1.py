#!/usr/bin/env python3
"""Evaluate a project-owned bidirectional trigram formula reranker."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
from typing import Any

import numpy as np
import torch

from audit_residual_shape_experts_v1 import _current_groups
from character_tensor_v1 import ROOT, _json_lines
from evaluate_48hz_prefix_v1 import _load_model, _sha256
from evaluate_homograph_context_reranker_v1 import STRICT_FAMILIES, _metrics
from train_masked_context_reranker_v1 import LAMBDA_GRID


SCHEMA = "aiflow-owned-prompt-ngram-reranker/v1"
STRICT_TOKENS = frozenset().union(*STRICT_FAMILIES.values())
FUNCTION_TOKENS = frozenset({"f", "g", "h", "F", "G", "H"})
ALPHAS = (0.01, 0.1, 0.5, 1.0)
WEIGHTS = tuple(sorted({*(float(value) for value in LAMBDA_GRID), 0.75, 1.0, 1.5, 2.0, 3.0}))
DEFAULT_CORPUS = ROOT / "artifacts" / "prompt_context_corpus_20260820_r2" / "prompt_context_corpus.jsonl"
DEFAULT_CANDIDATES = ROOT / "artifacts" / "homograph_context_20260814" / "direct_candidates.jsonl.gz"
DEFAULT_CROHME_CANDIDATES = ROOT / "artifacts" / "homograph_context_20260814" / "crohme_candidates.jsonl.gz"
DEFAULT_HWR = ROOT / "artifacts" / "unified_head_20260814" / "uniform_time_final_all_writers" / "project_symbol_head_checkpoint.pt"
DEFAULT_RUNTIME = Path(r"D:\AIFlow-Workspace\PrivateData\candidate-context-runtime-20260820-r43-layout-shadow-selected.json")
DEFAULT_TRUTH = Path(r"D:\AIFlow-Workspace\PrivateData\math-ink-data-collector\derived\public-candidate-20260819-r3\data\formulas_valid.jsonl")
DEFAULT_OUTPUT = ROOT / "artifacts" / "owned_prompt_ngram_20260820_r5_writer_loo_shadow" / "owned_prompt_ngram_report.json"


class BidirectionalTrigram:
    def __init__(self, sequences: list[list[str]], *, alpha: float, vocabulary_size: int) -> None:
        self.alpha = alpha
        self.vocabulary_size = vocabulary_size
        self.forward_counts, self.forward_context = self._fit(sequences)
        self.backward_counts, self.backward_context = self._fit([list(reversed(row)) for row in sequences])

    @staticmethod
    def _fit(sequences: list[list[str]]) -> tuple[Counter, Counter]:
        counts: Counter = Counter()
        contexts: Counter = Counter()
        for sequence in sequences:
            padded = ["<B>", "<B>", *sequence, "<E>"]
            for index in range(2, len(padded)):
                context = (padded[index - 2], padded[index - 1])
                counts[(*context, padded[index])] += 1
                contexts[context] += 1
        return counts, contexts

    def _log_probability(self, left: str, middle: str, token: str, *, backward: bool = False) -> float:
        counts = self.backward_counts if backward else self.forward_counts
        contexts = self.backward_context if backward else self.forward_context
        return math.log(
            (counts[(left, middle, token)] + self.alpha)
            / (contexts[(left, middle)] + self.alpha * self.vocabulary_size)
        )

    def forward_step(self, sequence: list[str], token: str) -> float:
        left = sequence[-2] if len(sequence) >= 2 else "<B>"
        middle = sequence[-1] if sequence else "<B>"
        return self._log_probability(left, middle, token)

    def forward_end(self, sequence: list[str]) -> float:
        return self.forward_step(sequence, "<E>")

    def backward_score(self, sequence: list[str]) -> float:
        reversed_sequence = list(reversed(sequence))
        score = 0.0
        prefix: list[str] = []
        for token in reversed_sequence:
            left = prefix[-2] if len(prefix) >= 2 else "<B>"
            middle = prefix[-1] if prefix else "<B>"
            score += self._log_probability(left, middle, token, backward=True)
            prefix.append(token)
        left = prefix[-2] if len(prefix) >= 2 else "<B>"
        middle = prefix[-1] if prefix else "<B>"
        return score + self._log_probability(left, middle, "<E>", backward=True)

    def score(self, sequence: list[str]) -> float:
        prefix: list[str] = []
        score = 0.0
        for token in sequence:
            score += self.forward_step(prefix, token)
            prefix.append(token)
        return score + self.forward_end(prefix) + self.backward_score(sequence)


def _d_path(path: Path, label: str, *, file: bool = False) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.drive.upper() != "D:" or file and not resolved.is_file():
        raise ValueError(f"{label} must remain on D: {resolved}")
    return resolved


def _prompt_sequences(path: Path, labels: list[str]) -> tuple[list[list[str]], dict[str, Any]]:
    allowed = set(labels)
    admitted = []
    excluded = []
    for row in _json_lines(path):
        sequence = [str(value) for value in row["labels"]]
        unknown = sorted(set(sequence) - allowed)
        if unknown:
            excluded.append({"formula_id": row["formula_id"], "unsupported": unknown})
        else:
            admitted.append(sequence)
    return admitted, {
        "formulas": len(admitted),
        "tokens": sum(map(len, admitted)),
        "excluded_formulas": len(excluded),
        "unsupported_labels": sorted({value for row in excluded for value in row["unsupported"]}),
    }


def _select_alpha(sequences: list[list[str]], vocabulary: list[str], labels: list[str]) -> tuple[float, list[dict[str, Any]]]:
    held_indices = {index for index in range(len(sequences)) if index % 5 == 0}
    fit = [row for index, row in enumerate(sequences) if index not in held_indices]
    held = [row for index, row in enumerate(sequences) if index in held_indices]
    trials = []
    for alpha in ALPHAS:
        model = BidirectionalTrigram(fit, alpha=alpha, vocabulary_size=len(labels) + 1)
        records = hits = formula_hits = 0
        for sequence in held:
            exact = True
            for index, truth in enumerate(sequence):
                winner = max(
                    vocabulary,
                    key=lambda token: model.score([*sequence[:index], token, *sequence[index + 1:]]),
                )
                records += 1
                hits += winner == truth
                exact = exact and winner == truth
            formula_hits += exact
        trials.append({
            "alpha": alpha,
            "records": records,
            "top1": hits / records,
            "formulas": len(held),
            "formula_exact": formula_hits / len(held),
        })
    winner = max(trials, key=lambda row: (row["formula_exact"], row["top1"], -row["alpha"]))
    return float(winner["alpha"]), trials


def _candidate_options(row: dict[str, Any], width: int, *, anchor_baseline: bool) -> list[tuple[str, float]]:
    baseline = str(row["baseline"])
    tokens = [str(value) for value in row["candidates"][:width]]
    probabilities = [float(value) for value in row["probabilities"][:width]]
    by_token = dict(zip(tokens, probabilities, strict=True))
    if baseline not in by_token:
        by_token[baseline] = max(probabilities, default=1.0)
    if anchor_baseline:
        by_token[baseline] = max(by_token.values())
    if baseline in STRICT_TOKENS:
        return [(baseline, by_token[baseline])]
    return sorted(by_token.items(), key=lambda item: (-item[1], item[0]))


def _rerank_formula(
    rows: list[dict[str, Any]], model: BidirectionalTrigram, weight: float,
    *, width: int, anchor_baseline: bool, beam_width: int = 256,
    maximum_changes: int | None = None,
) -> tuple[list[str], dict[str, Any]]:
    baseline = [str(row["baseline"]) for row in rows]
    if len(baseline) < 2 or "=" in baseline:
        return baseline, {
            "baseline_score": None,
            "selected_score": None,
            "score_margin": 0.0,
            "changes": 0,
            "guard": "singleton_or_untrained_equation",
        }
    beams: list[tuple[float, list[str]]] = [(0.0, [])]
    for row in rows:
        expanded = []
        for score, sequence in beams:
            for token, probability in _candidate_options(row, width, anchor_baseline=anchor_baseline):
                expanded.append((
                    score + math.log(max(probability, 1e-12)) + weight * model.forward_step(sequence, token),
                    [*sequence, token],
                ))
        beams = sorted(expanded, key=lambda item: item[0], reverse=True)[:beam_width]
    finalized = [
        (
            score + weight * (model.forward_end(sequence) + model.backward_score(sequence)),
            sequence,
        )
        for score, sequence in beams
    ]
    best_score, prediction = max(finalized, key=lambda item: item[0])
    baseline_hwr = sum(
        math.log(max(dict(_candidate_options(row, width, anchor_baseline=anchor_baseline))[token], 1e-12))
        for row, token in zip(rows, baseline, strict=True)
    )
    baseline_score = baseline_hwr + weight * model.score(baseline)
    changes = sum(left != right for left, right in zip(baseline, prediction, strict=True))
    if best_score <= baseline_score or maximum_changes is not None and changes > maximum_changes:
        prediction = baseline
        best_score = baseline_score
        changes = 0
    return prediction, {
        "baseline_score": baseline_score,
        "selected_score": best_score,
        "score_margin": best_score - baseline_score,
        "changes": changes,
    }


def _function_call_guard(baseline: list[str], proposal: list[str]) -> bool:
    changed = [index for index, (left, right) in enumerate(zip(baseline, proposal, strict=True)) if left != right]
    if not changed or len(changed) > 2:
        return False
    restored_open_fence = any(
        baseline[index] in FUNCTION_TOKENS
        and baseline[index + 1] != "("
        and proposal[index + 1] == "("
        and ")" in proposal[index + 2:]
        for index in range(len(proposal) - 1)
    )
    if not restored_open_fence:
        return False
    depth = 0
    for token in proposal:
        if token == "(":
            depth += 1
        elif token == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def _group_direct(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source in rows:
        grouped[str(source["formula_id"])].append({
            **source,
            "baseline": str(source["final_topk"][0]),
            "candidates": list(source["final_topk"]),
            "probabilities": list(source["final_topk_probabilities"]),
        })
    for values in grouped.values():
        values.sort(key=lambda row: int(row["context"]["index"]))
    return grouped


def _predictions(
    grouped: dict[str, list[dict[str, Any]]], model: BidirectionalTrigram,
    weight: float, *, width: int, anchor_baseline: bool,
    function_call_only: bool = False,
) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    predictions = {}
    audits = {}
    for formula_id, rows in grouped.items():
        tokens, audit = _rerank_formula(
            rows, model, weight, width=width,
            anchor_baseline=anchor_baseline,
        )
        baseline = [str(row["baseline"]) for row in rows]
        if function_call_only and tokens != baseline and not _function_call_guard(baseline, tokens):
            tokens = baseline
            audit = {**audit, "deployment_guard": "rejected_non_function_call_change", "changes": 0}
        elif function_call_only and tokens != baseline:
            audit = {**audit, "deployment_guard": "accepted_function_call_repair"}
        predictions.update({str(row["record_id"]): token for row, token in zip(rows, tokens, strict=True)})
        audits[formula_id] = audit
    return predictions, audits


def _select_weight(rows: list[dict[str, Any]], predictions: dict[float, dict[str, str]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    baseline = {str(row["record_id"]): str(row["final_topk"][0]) for row in rows}
    baseline_metrics = _metrics(rows, baseline)
    trials = []
    for weight in WEIGHTS:
        metrics = _metrics(rows, predictions[float(weight)])
        admissible = (
            all(metrics[key] >= baseline_metrics[key] for key in ("all_top1", "strict_micro_top1", "strict_macro_top1", "formula_exact"))
            and metrics["improved"] >= metrics["regressed"]
        )
        trials.append({"weight": float(weight), "metrics": metrics, "admissible": admissible})
    winner = max(
        (row for row in trials if row["admissible"]),
        key=lambda row: (
            row["metrics"]["formula_exact"], row["metrics"]["all_top1"],
            row["metrics"]["strict_macro_top1"],
            row["metrics"]["improved"] - row["metrics"]["regressed"],
            -row["metrics"]["changed"], -row["weight"],
        ),
    )
    return winner, trials


def _writer_loo(
    rows: list[dict[str, Any]], predictions: dict[float, dict[str, str]],
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    output = {}
    folds = []
    for writer in sorted({str(row["writer_group"]) for row in rows}):
        training = [row for row in rows if str(row["writer_group"]) != writer]
        held = [row for row in rows if str(row["writer_group"]) == writer]
        winner, _ = _select_weight(training, predictions)
        selected = predictions[float(winner["weight"])]
        output.update({str(row["record_id"]): selected[str(row["record_id"])] for row in held})
        folds.append({
            "writer_group": writer,
            "selected_weight": winner["weight"],
            "training_formulas": len({row["formula_id"] for row in training}),
            "held_formulas": len({row["formula_id"] for row in held}),
        })
    return output, folds


def _current_grouped(
    runtime: dict[str, Any], records: list[dict[str, Any]],
    probability: np.ndarray, labels: list[str], *, width: int,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, record in enumerate(records):
        order = np.argsort(-probability[index])[:width]
        grouped[str(record["formula_id"])].append({
            "record_id": record["record_id"],
            "context": {"index": record["index"]},
            "baseline": record["final"],
            "candidates": [labels[int(value)] for value in order],
            "probabilities": [float(probability[index, int(value)]) for value in order],
        })
    for formula_id, values in grouped.items():
        values.sort(key=lambda row: int(row["context"]["index"]))
        expected = next(row for row in runtime["formulas"] if row["formula_id"] == formula_id)
        if [row["baseline"] for row in values] != list(expected["finalized_tokens"]):
            raise AssertionError(f"current formula order mismatch: {formula_id}")
    return grouped


def _score_current(
    runtime: dict[str, Any], truth_rows: list[dict[str, Any]],
    grouped: dict[str, list[dict[str, Any]]], model: BidirectionalTrigram,
    weight: float,
) -> dict[str, Any]:
    truth = {
        str(row["sample_id"]): [str(cell["token"]) for cell in row.get("target_cells") or []]
        for row in truth_rows
    }
    results = []
    for formula in runtime["formulas"]:
        formula_id = str(formula["formula_id"])
        before = list(formula["finalized_tokens"])
        if formula["decision_status"] == "REVIEW_REQUIRED":
            after, audit = _rerank_formula(
                grouped[formula_id], model, weight, width=32,
                anchor_baseline=True, maximum_changes=2,
            )
            if after != before and not _function_call_guard(before, after):
                after = before
                audit = {**audit, "deployment_guard": "rejected_non_function_call_change", "changes": 0}
            elif after != before:
                audit = {**audit, "deployment_guard": "accepted_function_call_repair"}
        else:
            after, audit = before, {"baseline_score": None, "selected_score": None, "score_margin": 0.0, "changes": 0}
        results.append({
            "formula_id": formula_id,
            "decision_status": formula["decision_status"],
            "before": before,
            "after": after,
            "truth": truth[formula_id],
            "before_exact": before == truth[formula_id],
            "after_exact": after == truth[formula_id],
            "audit": audit,
        })
    return {
        "formulas": len(results),
        "baseline_exact": sum(row["before_exact"] for row in results),
        "challenger_exact": sum(row["after_exact"] for row in results),
        "changed_formulas": [row for row in results if row["before"] != row["after"]],
        "improved_formulas": [row["formula_id"] for row in results if not row["before_exact"] and row["after_exact"]],
        "regressed_formulas": [row["formula_id"] for row in results if row["before_exact"] and not row["after_exact"]],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--direct-candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--crohme-candidates", type=Path, default=DEFAULT_CROHME_CANDIDATES)
    parser.add_argument("--hwr", type=Path, default=DEFAULT_HWR)
    parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--truth", type=Path, default=DEFAULT_TRUTH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    corpus = _d_path(args.corpus, "corpus", file=True)
    direct_path = _d_path(args.direct_candidates, "direct candidates", file=True)
    crohme_path = _d_path(args.crohme_candidates, "CROHME candidates", file=True)
    hwr = _d_path(args.hwr, "HWR", file=True)
    runtime_path = _d_path(args.runtime, "runtime", file=True)
    truth_path = _d_path(args.truth, "truth", file=True)
    output = _d_path(args.output, "output")
    if output.exists():
        parser.error("output must be new")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    model, labels, _ = _load_model(hwr, device)
    sequences, admission = _prompt_sequences(corpus, labels)
    vocabulary = sorted({token for sequence in sequences for token in sequence})
    alpha, alpha_trials = _select_alpha(sequences, vocabulary, labels)
    ngram = BidirectionalTrigram(sequences, alpha=alpha, vocabulary_size=len(labels) + 1)

    direct_rows = list(_json_lines(direct_path))
    direct_grouped = _group_direct(direct_rows)
    all_predictions = {
        float(weight): _predictions(
            direct_grouped, ngram, float(weight), width=5,
            anchor_baseline=False,
        )[0]
        for weight in WEIGHTS
    }
    loo_predictions, folds = _writer_loo(direct_rows, all_predictions)
    direct_baseline = {
        str(row["record_id"]): str(row["final_topk"][0]) for row in direct_rows
    }
    direct_baseline_metrics = _metrics(direct_rows, direct_baseline)
    direct_metrics = _metrics(direct_rows, loo_predictions)
    product_selection, weight_trials = _select_weight(direct_rows, all_predictions)
    product_weight = float(product_selection["weight"])
    deployment_weight = float(statistics.median(row["selected_weight"] for row in folds))
    guarded_direct_predictions, _ = _predictions(
        direct_grouped, ngram, deployment_weight, width=5,
        anchor_baseline=False, function_call_only=True,
    )
    guarded_direct_metrics = _metrics(direct_rows, guarded_direct_predictions)

    crohme_rows = list(_json_lines(crohme_path))
    crohme_grouped = _group_direct(crohme_rows)
    crohme_predictions, _ = _predictions(
        crohme_grouped, ngram, deployment_weight, width=5,
        anchor_baseline=False, function_call_only=True,
    )
    crohme_baseline_predictions = {
        str(row["record_id"]): str(row["final_topk"][0]) for row in crohme_rows
    }
    crohme_baseline = _metrics(crohme_rows, crohme_baseline_predictions)
    crohme_metrics = _metrics(crohme_rows, crohme_predictions)

    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    truth_rows = list(_json_lines(truth_path))
    current_records, _, current_probability = _current_groups(
        runtime, truth_rows, model, device,
    )
    current_grouped = _current_grouped(
        runtime, current_records, current_probability, labels, width=32,
    )
    current = _score_current(
        runtime, truth_rows, current_grouped, ngram, deployment_weight,
    )
    current_diagnostics = {
        str(weight): _score_current(
            runtime, truth_rows, current_grouped, ngram, float(weight),
        )
        for weight in WEIGHTS
    }

    payload = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "diagnostic_shadow",
        "architecture": {
            "model": "project-owned bidirectional additive-smoothed trigram",
            "candidate_policy": "strict-homograph lock; direct Top-5; current REVIEW_REQUIRED anchored Top-32",
            "maximum_current_changes_per_formula": 2,
            "guards": "singletons and equations are immutable because neither has admitted prompt-context training",
            "deployment_gate": "accept only a balanced maximum-two-change proposal that restores a missing open fence immediately after a function token",
        },
        "prompt_training": {
            "admission": admission,
            "vocabulary": vocabulary,
            "alpha": alpha,
            "alpha_trials": alpha_trials,
        },
        "direct_writer_loo": {
            "formulas": len(direct_grouped),
            "records": len(direct_rows),
            "baseline": direct_baseline_metrics,
            "ngram": direct_metrics,
            "folds": folds,
            "product_weight": product_weight,
            "deployment_weight": deployment_weight,
            "deployment_weight_selection": "median of writer-LOO selected weights; no writer identity at inference",
            "weight_trials": weight_trials,
            "function_call_guarded_product": guarded_direct_metrics,
        },
        "current159": current,
        "crohme_repeated_noncommercial_diagnostic": {
            "formulas": len(crohme_grouped),
            "records": len(crohme_rows),
            "baseline": crohme_baseline,
            "ngram": crohme_metrics,
            "used_for_training_or_selection": False,
        },
        "current159_posthoc_weight_diagnostics": {
            weight: {
                "baseline_exact": value["baseline_exact"],
                "challenger_exact": value["challenger_exact"],
                "improved_formulas": value["improved_formulas"],
                "regressed_formulas": value["regressed_formulas"],
            }
            for weight, value in current_diagnostics.items()
        },
        "decision": {
            "direct_nonregression": all(
                direct_metrics[key] >= direct_baseline_metrics[key]
                for key in ("all_top1", "strict_micro_top1", "strict_macro_top1", "formula_exact")
            ),
            "current_zero_regression": not current["regressed_formulas"],
            "current_exact_gain": current["challenger_exact"] > current["baseline_exact"],
            "crohme_formula_nonregression": crohme_metrics["formula_exact"] >= crohme_baseline["formula_exact"],
            "automatic_default_replacement": False,
        },
        "contracts": {
            "prompt_only_training": True,
            "current_truth_used_for_weight_selection": False,
            "CROHME_used_for_training_or_selection": False,
            "arithmetic_evaluation": False,
            "candidate_invention": False,
            "grouping_mutations": 0,
            "product_default_enabled": False,
        },
        "sources": {
            "corpus": {"path": str(corpus), "sha256": _sha256(corpus)},
            "direct_candidates": {"path": str(direct_path), "sha256": _sha256(direct_path)},
            "crohme_candidates": {"path": str(crohme_path), "sha256": _sha256(crohme_path)},
            "hwr": {"path": str(hwr), "sha256": _sha256(hwr)},
            "runtime": {"path": str(runtime_path), "sha256": _sha256(runtime_path)},
            "truth": {"path": str(truth_path), "sha256": _sha256(truth_path)},
        },
    }
    output.parent.mkdir(parents=True, exist_ok=False)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({
        "event": "owned_prompt_ngram_complete",
        "output": str(output), "sha256": _sha256(output),
        "alpha": alpha, "product_weight": product_weight,
        "deployment_weight": deployment_weight,
        "direct_formula_exact": direct_metrics["formula_exact"],
        "crohme_formula_exact": crohme_metrics["formula_exact"],
        "crohme_formula_regressed": crohme_metrics["regressed"],
        "current_baseline_exact": current["baseline_exact"],
        "current_challenger_exact": current["challenger_exact"],
        "current_improved": current["improved_formulas"],
        "current_regressed": current["regressed_formulas"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
