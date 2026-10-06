#!/usr/bin/env python3
"""Evaluate the frozen candidate-validity selection on untouched acceptance."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
from hashlib import sha256
import io
import json
import os
from pathlib import Path

import torch

from character_tensor_v1 import _json_lines
from evaluate_homograph_context_reranker_v1 import _metrics
import train_candidate_validity_context_v1 as candidate


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SELECTION = (
    ROOT / "artifacts" / "candidate_validity_adoption_selection_20260820_r1"
    / "selection.json"
)
DEFAULT_ACCEPTANCE_ROOT = Path(
    r"D:\AIFlow-Workspace\PrivateData\math-ink-data-collector\derived"
    r"\fresh-context-acceptance-20260820-r2\frozen_acceptance"
)
DEFAULT_CANDIDATES = DEFAULT_ACCEPTANCE_ROOT / "fresh_acceptance_candidates.jsonl.gz"
DEFAULT_OUTPUT = ROOT / "artifacts" / "candidate_validity_fresh_acceptance_20260820_r1"
CROHME_THRESHOLDS = {
    "all_top1": 0.794861038280021,
    "formula_exact": 0.27439024390243905,
    "strict_macro_top1": 0.6142421481171703,
}
KNOWN_SPARSE = ("|", "O", "o")


def _sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_predictions(path: Path, rows: list[dict]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as raw:
            with gzip.GzipFile(
                filename="", fileobj=raw, mode="wb", compresslevel=6, mtime=0
            ) as zipped:
                with io.TextIOWrapper(zipped, encoding="utf-8", newline="\n") as stream:
                    for row in rows:
                        stream.write(json.dumps(
                            row, ensure_ascii=False, sort_keys=True,
                            separators=(",", ":"),
                        ))
                        stream.write("\n")
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _homograph_regressions(rows: list[dict], predictions: dict[str, str]) -> int:
    labels = {token for family in candidate.HOMOGRAPH_FAMILIES for token in family}
    return sum(
        str(row["label"]) in labels
        and str(row["final_topk"][0]) == str(row["label"])
        and predictions[str(row["record_id"])] != str(row["label"])
        for row in rows
    )


def evaluate(args: argparse.Namespace) -> dict:
    selection_path = args.selection.expanduser().resolve()
    acceptance_root = args.acceptance_root.expanduser().resolve()
    candidate_path = args.candidates.expanduser().resolve()
    crohme_path = args.crohme_candidates.expanduser().resolve()
    output = args.output.expanduser().resolve()
    required = (selection_path, candidate_path, crohme_path)
    if any(path.drive.upper() != "D:" or not path.is_file() for path in required):
        raise ValueError("selection and candidate caches must be existing D: files")
    if acceptance_root.drive.upper() != "D:" or not acceptance_root.is_dir():
        raise ValueError("acceptance root must be an existing D: directory")
    if output.drive.upper() != "D:":
        raise ValueError(f"evaluation output must remain on D:: {output}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite acceptance evaluation: {output}")

    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if (
        selection.get("schema") != "aiflow-candidate-validity-adoption-selection/v1"
        or selection["selection_contract"]["crohme_metrics_loaded"] is not False
        or selection["selection_contract"]["fresh_acceptance_predictions_loaded"] is not False
        or selection["selection_contract"]["fresh_acceptance_frozen_before_selection"] is not True
    ):
        raise ValueError("model selection was not frozen independently of acceptance")
    checkpoint = Path(selection["selected"]["checkpoint"]["path"]).resolve()
    if (
        not checkpoint.is_file()
        or _sha(checkpoint) != selection["selected"]["checkpoint"]["sha256"]
    ):
        raise ValueError("selected checkpoint hash mismatch")

    acceptance_manifest_path = acceptance_root / "frozen_acceptance_manifest.json"
    fresh_ownership_path = acceptance_root / "ownership_fresh_acceptance.jsonl"
    cache_report_path = candidate_path.with_suffix(candidate_path.suffix + ".report.json")
    acceptance_manifest = json.loads(
        acceptance_manifest_path.read_text(encoding="utf-8")
    )
    cache_report = json.loads(cache_report_path.read_text(encoding="utf-8"))
    if (
        acceptance_manifest.get("model_predictions_opened_before_freeze") is not False
        or int(acceptance_manifest["acceptance"]["formulae"]) < 50
        or int(acceptance_manifest["acceptance"]["writers"]) < 2
        or cache_report.get("training_performed") is not False
        or cache_report["output"]["sha256"] != _sha(candidate_path)
    ):
        raise ValueError("frozen acceptance candidate contract failed")
    ownership = list(_json_lines(fresh_ownership_path))
    fresh_formula_ids = {str(row["sample_id"]) for row in ownership}
    rows = [
        row for row in _json_lines(candidate_path)
        if str(row["formula_id"]) in fresh_formula_ids
    ]
    if (
        len({str(row["formula_id"]) for row in rows}) != len(fresh_formula_ids)
        or len(fresh_formula_ids) != int(acceptance_manifest["acceptance"]["formulae"])
        or len({str(row["writer_group"]) for row in rows}) != 2
        or any(str(row.get("hwr_policy")) != "unseen_writer_product" for row in rows)
    ):
        raise ValueError("fresh acceptance candidate coverage mismatch")

    hwr = args.hwr_checkpoint.expanduser().resolve()
    pretrained = args.pretrained.expanduser().resolve()
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    immutable_paths = (
        selection_path, checkpoint, acceptance_manifest_path,
        fresh_ownership_path, candidate_path, cache_report_path,
        crohme_path, hwr,
    )
    hashes_before = {str(path): _sha(path) for path in immutable_paths}

    model, contract, payload = candidate.load_candidate_validity_context(
        pretrained, checkpoint, hwr, device
    )
    baseline_predictions = {
        str(row["record_id"]): str(row["final_topk"][0]) for row in rows
    }
    predictions, audit = candidate.decide_candidate_validity_rows(
        model, contract, payload, rows, device, args.batch_size
    )
    repeat_predictions, repeat_audit = candidate.decide_candidate_validity_rows(
        model, contract, payload, rows, device, args.batch_size
    )
    reload_mismatches = sum(
        predictions[key] != repeat_predictions[key] for key in predictions
    )
    if audit != repeat_audit:
        raise AssertionError("acceptance repeat audit mismatch")
    baseline = _metrics(rows, baseline_predictions)
    context_metrics = _metrics(rows, predictions)
    homograph_regressions = _homograph_regressions(rows, predictions)

    crohme_rows = list(_json_lines(crohme_path))
    crohme_predictions, crohme_audit = candidate.decide_candidate_validity_rows(
        model, contract, payload, crohme_rows, device, args.batch_size
    )
    crohme_metrics = _metrics(crohme_rows, crohme_predictions)
    fresh_sparse_counts = {
        token: sum(str(row["label"]) == token for row in rows)
        for token in KNOWN_SPARSE
    }
    fresh_gates = {
        "top1_improved": context_metrics["all_top1"] > baseline["all_top1"],
        "formula_exact_improved": context_metrics["formula_exact"] > baseline["formula_exact"],
        "strict_macro_improved": context_metrics["strict_macro_top1"] > baseline["strict_macro_top1"],
        "homograph_regressions_zero": homograph_regressions == 0,
        "known_sparse_truth_present": all(value > 0 for value in fresh_sparse_counts.values()),
    }
    crohme_gates = {
        key: float(crohme_metrics[key]) >= threshold
        for key, threshold in CROHME_THRESHOLDS.items()
    }
    integrity_gates = {
        "candidate_new_tokens_zero": int(audit["new_tokens"]) == 0,
        "grouping_mutations_zero": int(audit["grouping_mutations"]) == 0,
        "checkpoint_repeat_mismatches_zero": reload_mismatches == 0,
        "crohme_candidate_new_tokens_zero": int(crohme_audit["new_tokens"]) == 0,
        "crohme_grouping_mutations_zero": int(crohme_audit["grouping_mutations"]) == 0,
    }

    output.mkdir(parents=True)
    prediction_path = output / "fresh_acceptance_predictions.jsonl.gz"
    _write_predictions(prediction_path, [{
        "record_id": str(row["record_id"]),
        "formula_id": str(row["formula_id"]),
        "writer_group": str(row["writer_group"]),
        "label": str(row["label"]),
        "hwr_top1": str(row["final_topk"][0]),
        "context_top1": predictions[str(row["record_id"])],
        "candidate_preserved": predictions[str(row["record_id"])]
        in [str(value) for value in row["final_topk"]],
    } for row in rows])
    hashes_after = {str(path): _sha(path) for path in immutable_paths}
    report = {
        "schema": "aiflow-candidate-validity-fresh-acceptance/v1",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "training_performed": False,
        "selection_frozen_before_evaluation": True,
        "selection_used_crohme": False,
        "selection_used_fresh_acceptance": False,
        "device": str(device),
        "fresh_acceptance": {
            "formulas": len(fresh_formula_ids),
            "glyphs": len(rows),
            "writers": len({str(row["writer_group"]) for row in rows}),
            "hwr_baseline": baseline,
            "selected_context": context_metrics,
            "known_sparse_truth_counts": fresh_sparse_counts,
            "homograph_regressions": homograph_regressions,
            "gates": fresh_gates,
        },
        "crohme_post_selection": {
            "role": "post-selection diagnostic only; never used for variant choice",
            "metrics": crohme_metrics,
            "thresholds": CROHME_THRESHOLDS,
            "gates": crohme_gates,
        },
        "integrity": {
            "fresh_candidate_audit": audit,
            "crohme_candidate_audit": crohme_audit,
            "checkpoint_repeat_mismatches": reload_mismatches,
            "gates": integrity_gates,
            "immutable_hashes_before": hashes_before,
            "immutable_hashes_after": hashes_after,
            "immutable_inputs_unchanged": hashes_before == hashes_after,
        },
        "decision": {
            "fresh_overall_gate_passed": all(fresh_gates.values()),
            "crohme_gate_passed": all(crohme_gates.values()),
            "integrity_gate_passed": all(integrity_gates.values()) and hashes_before == hashes_after,
            "adoption_gate_passed": (
                all(fresh_gates.values())
                and all(crohme_gates.values())
                and all(integrity_gates.values())
                and hashes_before == hashes_after
            ),
        },
        "artifacts": {
            "selection": {"path": str(selection_path), "sha256": _sha(selection_path)},
            "checkpoint": {"path": str(checkpoint), "sha256": _sha(checkpoint)},
            "acceptance_manifest": {"path": str(acceptance_manifest_path), "sha256": _sha(acceptance_manifest_path)},
            "candidate_cache": {"path": str(candidate_path), "sha256": _sha(candidate_path)},
            "predictions": {"path": str(prediction_path), "sha256": _sha(prediction_path)},
        },
    }
    report_path = output / "fresh_acceptance_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--acceptance-root", type=Path, default=DEFAULT_ACCEPTANCE_ROOT)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--crohme-candidates", type=Path, default=candidate.DEFAULT_CROHME)
    parser.add_argument("--hwr-checkpoint", type=Path, default=candidate.DEFAULT_HWR)
    parser.add_argument("--pretrained", type=Path, default=candidate.DEFAULT_PRETRAINED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    report = evaluate(args)
    print(json.dumps({
        "fresh": {
            "baseline_top1": report["fresh_acceptance"]["hwr_baseline"]["all_top1"],
            "context_top1": report["fresh_acceptance"]["selected_context"]["all_top1"],
            "baseline_formula": report["fresh_acceptance"]["hwr_baseline"]["formula_exact"],
            "context_formula": report["fresh_acceptance"]["selected_context"]["formula_exact"],
            "baseline_strict": report["fresh_acceptance"]["hwr_baseline"]["strict_macro_top1"],
            "context_strict": report["fresh_acceptance"]["selected_context"]["strict_macro_top1"],
            "homograph_regressions": report["fresh_acceptance"]["homograph_regressions"],
            "gates": report["fresh_acceptance"]["gates"],
        },
        "crohme": {
            "top1": report["crohme_post_selection"]["metrics"]["all_top1"],
            "formula": report["crohme_post_selection"]["metrics"]["formula_exact"],
            "strict": report["crohme_post_selection"]["metrics"]["strict_macro_top1"],
            "gates": report["crohme_post_selection"]["gates"],
        },
        "decision": report["decision"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
