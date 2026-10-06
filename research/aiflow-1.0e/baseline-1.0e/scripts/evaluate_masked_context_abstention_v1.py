"""Select a writer-disjoint abstention gate for the masked-context reranker."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

from character_tensor_v1 import ROOT, _json_lines
from evaluate_homograph_context_reranker_v1 import STRICT_FAMILIES, _metrics


SCHEMA = "aiflow-masked-context-abstention/v1"
DEFAULT_CANDIDATES = ROOT / "artifacts" / "homograph_context_20260814"
DEFAULT_MASKED = ROOT / "artifacts" / "masked_context_bert_tiny_20260814_r3"
DEFAULT_OUTPUT = ROOT / "artifacts" / "masked_context_abstention_20260820_r1_shadow"
STRICT_TOKENS = frozenset().union(*STRICT_FAMILIES.values())
TOP1_MAX = (0.15, 0.25, 0.35, 0.5, 0.7, 1.0)
RATIO_MIN = (0.0, 0.1, 0.25, 0.5)
MAX_RANK = (1, 2, 4)
METRIC_KEYS = ("all_top1", "strict_micro_top1", "strict_macro_top1", "formula_exact")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prediction_map(path: Path) -> dict[str, str]:
    rows = list(_json_lines(path))
    result = {str(row["record_id"]): str(row["reranked_top1"]) for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate prediction IDs: {path}")
    return result


def _configs() -> list[dict]:
    baseline = {"enabled": False, "max_top1_probability": 0.0, "min_candidate_ratio": 1.0, "max_candidate_rank": 0}
    candidates = [
        {
            "enabled": True,
            "max_top1_probability": top1,
            "min_candidate_ratio": ratio,
            "max_candidate_rank": rank,
        }
        for top1, ratio, rank in itertools.product(TOP1_MAX, RATIO_MIN, MAX_RANK)
    ]
    return [baseline, *candidates]


def _apply(rows: list[dict], reranked: dict[str, str], config: dict) -> dict[str, str]:
    output = {}
    for row in rows:
        record_id = str(row["record_id"])
        topk = [str(token) for token in row["final_topk"]]
        probabilities = [float(value) for value in row["final_topk_probabilities"]]
        baseline = topk[0]
        candidate = reranked[record_id]
        if candidate not in topk:
            raise ValueError(f"masked-context candidate escaped HWR Top-k: {record_id}")
        selected = baseline
        if config["enabled"] and candidate != baseline and baseline not in STRICT_TOKENS:
            rank = topk.index(candidate)
            ratio = probabilities[rank] / max(probabilities[0], 1e-12)
            if (
                probabilities[0] <= config["max_top1_probability"]
                and rank <= config["max_candidate_rank"]
                and ratio >= config["min_candidate_ratio"]
            ):
                selected = candidate
        output[record_id] = selected
    return output


def _delta(candidate: dict, baseline: dict) -> dict:
    return {key: candidate[key] - baseline[key] for key in METRIC_KEYS}


def _admissible(metrics: dict, baseline: dict) -> bool:
    return (
        all(metrics[key] >= baseline[key] for key in METRIC_KEYS)
        and metrics["improved"] >= metrics["regressed"]
    )


def _select(rows: list[dict], reranked: dict[str, str]) -> tuple[dict, list[dict]]:
    baseline_predictions = {str(row["record_id"]): str(row["final_topk"][0]) for row in rows}
    baseline = _metrics(rows, baseline_predictions)
    trials = []
    for config in _configs():
        metrics = _metrics(rows, _apply(rows, reranked, config))
        trials.append({"configuration": config, "metrics": metrics, "admissible": _admissible(metrics, baseline)})
    admissible = [trial for trial in trials if trial["admissible"]]
    winner = max(
        admissible,
        key=lambda trial: (
            trial["metrics"]["formula_exact"],
            trial["metrics"]["all_top1"],
            trial["metrics"]["strict_macro_top1"],
            trial["metrics"]["improved"] - trial["metrics"]["regressed"],
            -trial["metrics"]["changed"],
        ),
    )
    return winner, trials


def _nested_writer_loo(rows: list[dict], reranked: dict[str, str]) -> tuple[dict[str, str], list[dict]]:
    predictions, folds = {}, []
    writers = sorted({str(row["writer_group"]) for row in rows})
    for writer in writers:
        training = [row for row in rows if str(row["writer_group"]) != writer]
        held = [row for row in rows if str(row["writer_group"]) == writer]
        if {row["formula_id"] for row in training} & {row["formula_id"] for row in held}:
            raise AssertionError("formula leakage in abstention writer-LOO")
        winner, _ = _select(training, reranked)
        held_predictions = _apply(held, reranked, winner["configuration"])
        predictions.update(held_predictions)
        baseline = _metrics(held, {str(row["record_id"]): str(row["final_topk"][0]) for row in held})
        metrics = _metrics(held, held_predictions)
        folds.append({
            "held_writer_group": writer,
            "training_records": len(training),
            "held_records": len(held),
            "selected_configuration": winner["configuration"],
            "held_baseline": baseline,
            "held_gated": metrics,
            "held_delta": _delta(metrics, baseline),
        })
    if set(predictions) != {str(row["record_id"]) for row in rows}:
        raise AssertionError("abstention writer-LOO coverage mismatch")
    return predictions, folds


def _audit(rows: list[dict], predictions: dict[str, str]) -> dict:
    return {
        "records": len(rows),
        "candidate_preserved": all(
            predictions[str(row["record_id"])] in row["final_topk"] for row in rows
        ),
        "strict_baseline_lock": all(
            str(row["final_topk"][0]) not in STRICT_TOKENS
            or predictions[str(row["record_id"])] == str(row["final_topk"][0])
            for row in rows
        ),
        "new_tokens": 0,
        "grouping_mutations": 0,
    }


def _self_test() -> None:
    rows = [
        {"record_id": "a", "formula_id": "f1", "writer_group": "w1", "label": "|", "final_topk": ["|", "1"], "final_topk_probabilities": [0.6, 0.4]},
        {"record_id": "b", "formula_id": "f2", "writer_group": "w2", "label": "1", "final_topk": ["l", "1"], "final_topk_probabilities": [0.55, 0.45]},
    ]
    config = {"enabled": True, "max_top1_probability": 1.0, "min_candidate_ratio": 0.0, "max_candidate_rank": 1}
    predictions = _apply(rows, {"a": "1", "b": "1"}, config)
    assert predictions == {"a": "|", "b": "1"}
    assert _audit(rows, predictions)["strict_baseline_lock"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--masked", type=Path, default=DEFAULT_MASKED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    _self_test()
    if args.self_test:
        print(json.dumps({"event": "self_test", "status": "pass"}))
        return 0

    candidates = args.candidates.resolve()
    masked = args.masked.resolve()
    output = args.output.resolve()
    if any(path.drive.upper() != "D:" for path in (candidates, masked, output)):
        raise ValueError("all model data and output must remain on D:")
    report_path = output / "masked_context_abstention_report.json"
    if report_path.exists():
        parser.error(f"refusing to overwrite existing output: {report_path}")

    direct_path = candidates / "direct_candidates.jsonl.gz"
    crohme_path = candidates / "crohme_candidates.jsonl.gz"
    direct_prediction_path = masked / "direct_writer_loo_predictions.jsonl.gz"
    crohme_prediction_path = masked / "crohme_transfer_predictions.jsonl.gz"
    direct_rows = list(_json_lines(direct_path))
    crohme_rows = list(_json_lines(crohme_path))
    direct_reranked = _prediction_map(direct_prediction_path)
    crohme_reranked = _prediction_map(crohme_prediction_path)
    if set(direct_reranked) != {str(row["record_id"]) for row in direct_rows}:
        raise ValueError("direct prediction coverage mismatch")
    if set(crohme_reranked) != {str(row["record_id"]) for row in crohme_rows}:
        raise ValueError("CROHME prediction coverage mismatch")

    direct_baseline_predictions = {str(row["record_id"]): str(row["final_topk"][0]) for row in direct_rows}
    crohme_baseline_predictions = {str(row["record_id"]): str(row["final_topk"][0]) for row in crohme_rows}
    direct_baseline = _metrics(direct_rows, direct_baseline_predictions)
    crohme_baseline = _metrics(crohme_rows, crohme_baseline_predictions)
    nested_predictions, folds = _nested_writer_loo(direct_rows, direct_reranked)
    nested_metrics = _metrics(direct_rows, nested_predictions)
    final_winner, trials = _select(direct_rows, direct_reranked)
    final_config = final_winner["configuration"]
    direct_selected_predictions = _apply(direct_rows, direct_reranked, final_config)
    direct_selected = _metrics(direct_rows, direct_selected_predictions)
    crohme_selected_predictions = _apply(crohme_rows, crohme_reranked, final_config)
    crohme_selected = _metrics(crohme_rows, crohme_selected_predictions)
    direct_audit = _audit(direct_rows, nested_predictions)
    crohme_audit = _audit(crohme_rows, crohme_selected_predictions)

    research_gate = (
        nested_metrics["all_top1"] > direct_baseline["all_top1"]
        and _admissible(nested_metrics, direct_baseline)
        and _admissible(crohme_selected, crohme_baseline)
        and direct_audit["candidate_preserved"]
        and direct_audit["strict_baseline_lock"]
        and crohme_audit["candidate_preserved"]
        and crohme_audit["strict_baseline_lock"]
    )
    report = {
        "schema": SCHEMA,
        "method": {
            "base_model": "BERT-Tiny masked-context candidate reranker",
            "selection": "nested writer-LOO on project-owned OOF predictions only",
            "strict_safety": "never replace an HWR Top-1 token in a declared strict homograph family",
            "confidence_features": ["HWR Top-1 probability", "selected-candidate probability ratio", "selected-candidate HWR rank"],
            "crohme_used_for_selection": False,
        },
        "provenance": {
            "direct_candidates_sha256": _sha256(direct_path),
            "crohme_candidates_sha256": _sha256(crohme_path),
            "direct_predictions_sha256": _sha256(direct_prediction_path),
            "crohme_predictions_sha256": _sha256(crohme_prediction_path),
        },
        "selection": {
            "grid_size": len(trials),
            "final_configuration": final_config,
            "final_training_metrics": final_winner["metrics"],
            "writer_loo_folds": folds,
        },
        "evaluation": {
            "direct_nested_writer_loo": {
                "baseline": direct_baseline,
                "gated": nested_metrics,
                "delta": _delta(nested_metrics, direct_baseline),
                "audit": direct_audit,
            },
            "direct_full_oof_selection_diagnostic": {
                "warning": "configuration selected on these OOF rows; not acceptance evidence",
                "gated": direct_selected,
                "delta": _delta(direct_selected, direct_baseline),
            },
            "crohme_transfer_diagnostic": {
                "scope": "CC BY-NC diagnostic only; zero gate selection",
                "baseline": crohme_baseline,
                "gated": crohme_selected,
                "delta": _delta(crohme_selected, crohme_baseline),
                "audit": crohme_audit,
            },
        },
        "decision": {
            "research_gate_passed": research_gate,
            "runtime_status": "shadow",
            "automatic_default_replacement": False,
            "promotion_requirement": "fresh commercial project-owned writer/formula-disjoint acceptance",
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({
        "event": "masked_context_abstention_complete",
        "report": str(report_path),
        "direct_nested_top1": nested_metrics["all_top1"],
        "crohme_top1": crohme_selected["all_top1"],
        "crohme_strict_macro": crohme_selected["strict_macro_top1"],
        "research_gate": research_gate,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
