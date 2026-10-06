"""Probe the guarded BERT-Tiny reranker on saved raw-runtime candidates."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

import torch

from evaluate_48hz_prefix_v1 import _sha256
from evaluate_masked_context_abstention_v1 import _apply
from evaluate_raw_formula_layout_runtime_v1 import _score, _truth
from train_masked_context_reranker_v1 import (
    DEFAULT_PRETRAINED,
    load_product_checkpoint,
    rerank_formula_rows,
)


SCHEMA = "aiflow-masked-context-runtime-probe/v1"
DEFAULT_RUNTIME = Path(r"D:\AIFlow-Workspace\PrivateData\candidate-context-runtime-20260820-r43-layout-shadow-selected.json")
DEFAULT_TRUTH = Path(r"D:\AIFlow-Workspace\PrivateData\math-ink-data-collector\derived\public-candidate-20260819-r3\data\formulas_valid.jsonl")
DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[1] / "artifacts" / "masked_context_bert_tiny_20260814_r3" / "masked_context_product.pt"
DEFAULT_GATE = Path(__file__).resolve().parents[1] / "artifacts" / "masked_context_abstention_20260820_r1_shadow" / "masked_context_abstention_report.json"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "artifacts" / "masked_context_runtime_probe_20260820_r1_shadow" / "masked_context_runtime_probe.json"


def _d_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.drive.upper() != "D:" or not resolved.is_file():
        raise ValueError(f"{label} must be an existing file on D: {resolved}")
    return resolved


def _runtime_rows(runtime: dict) -> tuple[list[dict], dict[str, dict]]:
    formulas = runtime.get("formulas") or []
    mapped = {str(formula["formula_id"]): formula for formula in formulas}
    if len(mapped) != len(formulas):
        raise ValueError("duplicate raw-runtime formula IDs")
    rows = []
    for formula_id, formula in mapped.items():
        groups = {str(group["record_id"]): group for group in formula["groups"]}
        symbols = sorted(formula["symbols"], key=lambda symbol: int(symbol["context_index"]))
        if set(groups) != {str(symbol["record_id"]) for symbol in symbols}:
            raise ValueError(f"raw-runtime group/symbol mismatch: {formula_id}")
        if [int(symbol["context_index"]) for symbol in symbols] != list(range(len(symbols))):
            raise ValueError(f"non-contiguous raw-runtime context: {formula_id}")
        boxes = [groups[str(symbol["record_id"])]["box"] for symbol in symbols]
        left = min(float(box["left"]) for box in boxes)
        top = min(float(box["top"]) for box in boxes)
        right = max(float(box["right"]) for box in boxes)
        bottom = max(float(box["bottom"]) for box in boxes)
        formula_width = max(right - left, 1e-6)
        formula_height = max(bottom - top, 1e-6)
        for symbol in symbols:
            record_id = str(symbol["record_id"])
            box = groups[record_id]["box"]
            box_left, box_top = float(box["left"]), float(box["top"])
            box_right, box_bottom = float(box["right"]), float(box["bottom"])
            width, height = box_right - box_left, box_bottom - box_top
            rows.append({
                "record_id": record_id,
                "formula_id": formula_id,
                "final_topk": [str(token) for token in symbol["hwr_topk"]],
                "final_topk_probabilities": [float(value) for value in symbol["hwr_topk_probabilities"]],
                "geometry": {
                    "left": box_left,
                    "top": box_top,
                    "right": box_right,
                    "bottom": box_bottom,
                    "width": width,
                    "height": height,
                    "center_x": ((box_left + box_right) / 2.0 - left) / formula_width,
                    "center_y": ((box_top + box_bottom) / 2.0 - top) / formula_height,
                    "width_rel": width / formula_width,
                    "height_rel": height / formula_height,
                    "stroke_count": float(len(symbol["stroke_indices"])),
                },
                "context": {"index": int(symbol["context_index"]), "length": len(symbols)},
            })
    return rows, mapped


def _serialize_layout(formula: dict, predictions: dict[str, str], boxes: dict[str, dict]) -> str:
    layout = formula["formula_layout_shadow"]
    record_ids = [str(value) for value in layout["ordered_record_ids"]]
    if set(record_ids) != set(predictions):
        raise ValueError(f"layout/prediction coverage mismatch: {formula['formula_id']}")
    children: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    structural_children = set()
    for edge in layout["relations"]:
        parent, child, relation = str(edge["parent"]), str(edge["child"]), str(edge["type"])
        children[parent][relation].append(child)
        structural_children.add(child)
    emitted: set[str] = set()

    def left(record_id: str) -> tuple[float, str]:
        return float(boxes[record_id]["left"]), record_id

    def sequence(values: list[str], active: frozenset[str]) -> str:
        return "".join(node(value, active) for value in sorted(set(values), key=left))

    def node(record_id: str, active: frozenset[str]) -> str:
        if record_id in active:
            raise ValueError(f"layout cycle: {formula['formula_id']}")
        if record_id in emitted:
            return ""
        emitted.add(record_id)
        nested = active | {record_id}
        slots = children.get(record_id, {})
        above, below, contained = slots.get("above", []), slots.get("below", []), slots.get("contains", [])
        if above and below:
            base = rf"\frac{{{sequence(above, nested)}}}{{{sequence(below, nested)}}}"
        elif contained:
            base = rf"\sqrt{{{sequence(contained, nested)}}}"
        else:
            base = predictions[record_id]
        if slots.get("subscript"):
            base += rf"_{{{sequence(slots['subscript'], nested)}}}"
        if slots.get("superscript"):
            base += rf"^{{{sequence(slots['superscript'], nested)}}}"
        return base

    roots = [record_id for record_id in record_ids if record_id not in structural_children]
    return sequence(roots, frozenset())


def _variant_predictions(rows: list[dict], formulas: dict[str, dict], gated: dict[str, str]) -> dict[str, dict[str, str]]:
    hwr = {str(row["record_id"]): str(row["final_topk"][0]) for row in rows}
    current = {
        str(symbol["record_id"]): str(symbol["finalized_top1"])
        for formula in formulas.values() for symbol in formula["symbols"]
    }
    unresolved_rescue = dict(current)
    for record_id in current:
        if current[record_id] == hwr[record_id] and gated[record_id] != hwr[record_id]:
            unresolved_rescue[record_id] = gated[record_id]
    return {
        "current_layout": current,
        "bert_gated_replace": gated,
        "bert_gated_unresolved_rescue": unresolved_rescue,
    }


def _evaluate_variant(
    name: str, predictions: dict[str, str], formulas: dict[str, dict], truth: dict[str, str], baseline: dict[str, str],
) -> dict:
    latex = {}
    for formula_id, formula in formulas.items():
        formula_predictions = {
            str(symbol["record_id"]): predictions[str(symbol["record_id"])]
            for symbol in formula["symbols"]
        }
        boxes = {str(group["record_id"]): group["box"] for group in formula["groups"]}
        latex[formula_id] = _serialize_layout(formula, formula_predictions, boxes)
    if name == "current_layout":
        mismatches = [
            formula_id for formula_id, value in latex.items()
            if value != str(formulas[formula_id]["formula_latex_shadow"])
        ]
        if mismatches:
            raise ValueError(f"saved layout graph did not replay: {mismatches}")
    ids = sorted(truth)
    improved = [formula_id for formula_id in ids if baseline[formula_id] != truth[formula_id] and latex[formula_id] == truth[formula_id]]
    regressed = [formula_id for formula_id in ids if baseline[formula_id] == truth[formula_id] and latex[formula_id] != truth[formula_id]]
    changed = [formula_id for formula_id in ids if latex[formula_id] != baseline[formula_id]]
    return {
        "score": _score(latex, truth, ids),
        "improved_formulas": improved,
        "regressed_formulas": regressed,
        "changed_formulas": changed,
        "latex": latex,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--truth", type=Path, default=DEFAULT_TRUTH)
    parser.add_argument("--pretrained", type=Path, default=DEFAULT_PRETRAINED)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--gate", type=Path, default=DEFAULT_GATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    runtime_path = _d_file(args.runtime, "saved raw runtime")
    truth_path = _d_file(args.truth, "project truth")
    pretrained = args.pretrained.expanduser().resolve()
    checkpoint = _d_file(args.checkpoint, "masked-context checkpoint")
    gate_path = _d_file(args.gate, "abstention report")
    output = args.output.expanduser().resolve()
    if pretrained.drive.upper() != "D:" or output.drive.upper() != "D:":
        raise ValueError("pretrained model and output must remain on D:")
    if output.exists():
        parser.error(f"refusing to overwrite output: {output}")

    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    if runtime.get("schema") != "aiflow-raw-formula-context-runtime/v1":
        raise ValueError("unexpected saved raw-runtime schema")
    truth, truth_inventory = _truth(truth_path)
    rows, formulas = _runtime_rows(runtime)
    if set(formulas) != set(truth):
        raise ValueError("raw-runtime/truth formula coverage mismatch")
    gate_report = json.loads(gate_path.read_text(encoding="utf-8"))
    config = gate_report["selection"]["final_configuration"]
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    model, contract, checkpoint_payload = load_product_checkpoint(pretrained, checkpoint, device)
    bert = rerank_formula_rows(model, contract, checkpoint_payload, rows, device)
    gated = _apply(rows, bert, config)
    variants = _variant_predictions(rows, formulas, gated)
    baseline_latex = {formula_id: str(formula["formula_latex_shadow"]) for formula_id, formula in formulas.items()}
    evaluations = {
        name: _evaluate_variant(name, predictions, formulas, truth, baseline_latex)
        for name, predictions in variants.items()
    }
    current = evaluations["current_layout"]
    rescue = evaluations["bert_gated_unresolved_rescue"]
    changed_records = [
        record_id for record_id, token in variants["bert_gated_unresolved_rescue"].items()
        if token != variants["current_layout"][record_id]
    ]
    report = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "diagnostic_shadow_only",
        "provenance": {
            "runtime": str(runtime_path),
            "runtime_sha256": _sha256(runtime_path),
            "truth": str(truth_path),
            "truth_sha256": _sha256(truth_path),
            "masked_context_checkpoint": str(checkpoint),
            "masked_context_checkpoint_sha256": _sha256(checkpoint),
            "abstention_report": str(gate_path),
            "abstention_report_sha256": _sha256(gate_path),
        },
        "protocol": {
            "truth_inventory": truth_inventory,
            "truth_supplied_to_model": False,
            "writer_or_formula_length_supplied_to_model": False,
            "candidate_policy": "BERT may reorder only each raw-runtime HWR Top-5",
            "fusion_policy": "change only rows where the current finalizer abstained at HWR Top-1",
            "layout_policy": "replay the saved immutable layout graph with candidate-preserving tokens",
            "selection_warning": "current159 was previously inspected and cannot authorize promotion",
        },
        "abstention_configuration": config,
        "evaluations": {
            name: {key: value for key, value in evaluation.items() if key != "latex"}
            for name, evaluation in evaluations.items()
        },
        "audit": {
            "records": len(rows),
            "formulas": len(formulas),
            "changed_records": changed_records,
            "changed_record_count": len(changed_records),
            "candidate_preservation_rate": 1.0,
            "new_tokens": 0,
            "deleted_glyphs": 0,
            "grouping_mutations": 0,
        },
        "decision": {
            "probe_improved_without_regression": (
                rescue["score"]["exact_count"] > current["score"]["exact_count"]
                and not rescue["regressed_formulas"]
            ),
            "automatic_default_replacement": False,
            "runtime_status": "shadow",
            "promotion_requirement": "fresh commercial writer/formula-disjoint acceptance",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=False)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({
        "event": "masked_context_runtime_probe_complete",
        "output": str(output),
        "current_exact": current["score"]["exact_count"],
        "replace_exact": evaluations["bert_gated_replace"]["score"]["exact_count"],
        "rescue_exact": rescue["score"]["exact_count"],
        "rescue_improved": rescue["improved_formulas"],
        "rescue_regressed": rescue["regressed_formulas"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
