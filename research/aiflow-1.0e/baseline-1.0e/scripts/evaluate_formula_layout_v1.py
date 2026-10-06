#!/usr/bin/env python3
"""Evaluate geometry-derived formula layout separately from character context.

Direct order uses accepted ownership label order only as held-out evaluation
truth. CROHME contributes noncommercial relation diagnostics only and is never
used to choose thresholds or train a product model.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET

from character_tensor_v1 import _json_lines
from evaluate_48hz_prefix_v1 import _sha256
from formula_layout_v1 import (
    STRUCTURAL, infer_formula_layout, selected_layout_evidence_rows,
)
from formula_script_network_v1 import NeuralScriptPredictor


SCHEMA = "aiflow-formula-layout-evaluation/v1"
XML_ID = "{http://www.w3.org/XML/1998/namespace}id"


def _grouped(rows: Iterable[dict]) -> dict[str, list[dict]]:
    output: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        output[str(row["formula_id"])].append(row)
    return dict(output)


def _predictions(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    return {
        str(row["record_id"]): str(row["finalized_top1"])
        for row in _json_lines(path)
    }


def _order_metrics(
    rows: list[dict], predictions: dict[str, str], script_predictor=None,
) -> dict:
    formulas = _grouped(rows)
    order_exact = pair_hits = pair_total = 0
    hwr_exact = final_exact = oracle_exact = 0
    hwr_hits = final_hits = oracle_hits = 0
    combined_hwr = combined_final = combined_oracle = 0
    failures = []
    for formula_id, sequence in formulas.items():
        truth = sorted(sequence, key=lambda row: int(row["context"]["index"]))
        truth_ids = [str(row["record_id"]) for row in truth]
        layout = infer_formula_layout(sequence, script_predictor=script_predictor)
        predicted_ids = list(layout["ordered_record_ids"])
        exact = predicted_ids == truth_ids
        order_exact += exact
        positions = {record_id: index for index, record_id in enumerate(predicted_ids)}
        for index, first in enumerate(truth_ids):
            for second in truth_ids[index + 1:]:
                pair_total += 1
                pair_hits += positions[first] < positions[second]
        top1_ok = all(str(row["final_topk"][0]) == str(row["label"]) for row in sequence)
        final_ok = bool(predictions) and all(
            predictions.get(str(row["record_id"])) == str(row["label"])
            for row in sequence
        )
        oracle_ok = all(str(row["label"]) in row["final_topk"] for row in sequence)
        hwr_hits += sum(str(row["final_topk"][0]) == str(row["label"]) for row in sequence)
        final_hits += sum(
            predictions.get(str(row["record_id"])) == str(row["label"])
            for row in sequence
        )
        oracle_hits += sum(str(row["label"]) in row["final_topk"] for row in sequence)
        hwr_exact += top1_ok
        final_exact += final_ok
        oracle_exact += oracle_ok
        combined_hwr += exact and top1_ok
        combined_final += exact and final_ok
        combined_oracle += exact and oracle_ok
        if not exact and len(failures) < 25:
            by_id = {str(row["record_id"]): row for row in sequence}
            failures.append({
                "formula_id": formula_id,
                "truth_labels": [str(by_id[record_id]["label"]) for record_id in truth_ids],
                "layout_labels": [str(by_id[record_id]["label"]) for record_id in predicted_ids],
                "edges": layout["edges"],
                "top5_oracle": oracle_ok,
            })
    count = len(formulas)
    return {
        "formulas": count,
        "layout_order_exact": order_exact / max(count, 1),
        "layout_order_exact_count": order_exact,
        "ordered_pair_accuracy": pair_hits / max(pair_total, 1),
        "ordered_pairs": pair_total,
        "character_accuracy": {
            "records": len(rows),
            "hwr_top1": hwr_hits / max(len(rows), 1),
            "context_finalized": final_hits / max(len(rows), 1) if predictions else None,
            "top5_oracle": oracle_hits / max(len(rows), 1),
        },
        "character_formula_exact": {
            "hwr_top1": hwr_exact / max(count, 1),
            "context_finalized": final_exact / max(count, 1) if predictions else None,
            "top5_oracle": oracle_exact / max(count, 1),
        },
        "layout_and_character_formula_exact": {
            "hwr_top1": combined_hwr / max(count, 1),
            "context_finalized": combined_final / max(count, 1) if predictions else None,
            "top5_oracle": combined_oracle / max(count, 1),
        },
        "order_failures": failures,
    }


def _flat_structure_gate(
    rows: list[dict], script_predictor=None,
    relation_formula_ids: frozenset[str] = frozenset(),
) -> dict:
    """Require no invented structure in the accepted direct flat corpus."""
    edges = 0
    affected = []
    grouped = _grouped(rows)
    for formula_id, sequence in grouped.items():
        if formula_id in relation_formula_ids:
            continue
        layout = infer_formula_layout(sequence, script_predictor=script_predictor)
        structural = [edge for edge in layout["edges"] if edge["type"] in STRUCTURAL]
        edges += len(structural)
        if structural:
            by_id = {str(row["record_id"]): row for row in sequence}
            affected.append({
                "formula_id": formula_id,
                "truth_labels": [
                    str(row["label"])
                    for row in sorted(sequence, key=lambda item: int(item["context"]["index"]))
                ],
                "edges": [
                    {
                        **edge,
                        "parent_label": str(by_id[edge["parent"]]["label"]),
                        "child_label": str(by_id[edge["child"]]["label"]),
                    }
                    for edge in structural
                ],
            })
    return {
        "expected": "zero structural edges in accepted direct flat formulas",
        "passed": edges == 0,
        "formulas": len(grouped) - len(set(grouped) & relation_formula_ids),
        "excluded_relation_formulas": sorted(set(grouped) & relation_formula_ids),
        "false_positive_edges": edges,
        "affected_formulas": len(affected),
        "failures": affected[:25],
    }


def _owned_relation_acceptance(path: Path | None) -> dict | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "aiflow-owned-formula-relation-acceptance/v1":
        raise ValueError("owned relation acceptance schema mismatch")
    formulas = payload.get("formulas") or []
    if not formulas or len({str(row["formula_id"]) for row in formulas}) != len(formulas):
        raise ValueError("owned relation acceptance formulas must be non-empty and unique")
    return payload


def _indexed_truth(formula: dict, rows: list[dict]) -> set[tuple[str, str, str]]:
    ordered = sorted(rows, key=lambda row: int(row["context"]["index"]))
    if [int(row["context"]["index"]) for row in ordered] != list(range(len(ordered))):
        raise ValueError(f"owned relation indices are not contiguous: {formula['formula_id']}")
    identifiers = [str(row["record_id"]) for row in ordered]
    truth = set()
    for edge in formula["truth_relations"]:
        parent, child = int(edge["parent_index"]), int(edge["child_index"])
        if not (0 <= parent < len(identifiers) and 0 <= child < len(identifiers)):
            raise ValueError(f"owned relation truth index out of range: {formula['formula_id']}")
        truth.add((identifiers[parent], identifiers[child], str(edge["type"])))
    return truth


def _owned_relation_metrics(
    acceptance: dict, rows_by_formula: dict[str, list[dict]] | None = None,
    predictions: dict[str, str] | None = None, script_predictor=None,
) -> dict:
    totals = Counter(tp=0, fp=0, fn=0)
    exact = hwr_exact = final_exact = strict_exact = 0
    failures = []
    evaluated = 0
    for formula in acceptance["formulas"]:
        formula_id = str(formula["formula_id"])
        rows = (
            formula["rows"] if rows_by_formula is None
            else rows_by_formula.get(formula_id)
        )
        if rows is None:
            continue
        truth = _indexed_truth(formula, rows)
        predicted = {
            (str(edge["parent"]), str(edge["child"]), str(edge["type"]))
            for edge in infer_formula_layout(rows, script_predictor=script_predictor)["edges"]
            if edge["type"] in STRUCTURAL
        }
        relation_ok = predicted == truth
        top1_ok = all(str(row["final_topk"][0]) == str(row["label"]) for row in rows)
        finalized_ok = bool(predictions) and all(
            predictions.get(str(row["record_id"])) == str(row["label"])
            for row in rows
        )
        tp, fp, fn = len(predicted & truth), len(predicted - truth), len(truth - predicted)
        totals.update(tp=tp, fp=fp, fn=fn)
        evaluated += 1
        exact += relation_ok
        hwr_exact += top1_ok
        final_exact += finalized_ok
        strict_exact += relation_ok and finalized_ok
        if not relation_ok and len(failures) < 25:
            failures.append({
                "formula_id": formula_id,
                "truth": sorted(truth), "predicted": sorted(predicted),
            })
    precision = totals["tp"] / max(totals["tp"] + totals["fp"], 1)
    recall = totals["tp"] / max(totals["tp"] + totals["fn"], 1)
    return {
        "declared_formulas": len(acceptance["formulas"]),
        "evaluated_formulas": evaluated,
        "formula_exact_count": exact,
        "formula_exact": exact / max(evaluated, 1),
        "micro": {
            **dict(totals), "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        },
        "character_model_evaluated": rows_by_formula is not None,
        "character_formula_exact": {
            "hwr_top1_count": hwr_exact,
            "context_finalized_count": final_exact if predictions else None,
        } if rows_by_formula is not None else None,
        "relation_and_character_formula_exact_count": (
            strict_exact if predictions and rows_by_formula is not None else None
        ),
        "failures": failures,
    }


def _prediction_delta(
    rows: list[dict], baseline: dict[str, str], challenger: dict[str, str],
) -> dict | None:
    if not baseline or not challenger:
        return None
    improved = regressed = changed = 0
    formula_improved = formula_regressed = 0
    changes = []
    for row in rows:
        record_id = str(row["record_id"])
        before = baseline[record_id]
        after = challenger[record_id]
        changed += before != after
        improved += before != str(row["label"]) and after == str(row["label"])
        regressed += before == str(row["label"]) and after != str(row["label"])
        if before != after and len(changes) < 30:
            outcome = (
                "improved" if before != str(row["label"]) and after == str(row["label"])
                else (
                    "regressed" if before == str(row["label"]) and after != str(row["label"])
                    else "changed_still_wrong"
                )
            )
            changes.append({
                "formula_id": str(row["formula_id"]),
                "record_id": record_id,
                "before": before,
                "after": after,
                "truth": str(row["label"]),
                "outcome": outcome,
                "before_probability": float(row["final_topk_probabilities"][
                    [str(token) for token in row["final_topk"]].index(before)
                ]),
                "after_probability": float(row["final_topk_probabilities"][
                    [str(token) for token in row["final_topk"]].index(after)
                ]),
            })
    for sequence in _grouped(rows).values():
        before = all(
            baseline[str(row["record_id"])] == str(row["label"])
            for row in sequence
        )
        after = all(
            challenger[str(row["record_id"])] == str(row["label"])
            for row in sequence
        )
        formula_improved += after and not before
        formula_regressed += before and not after
    return {
        "changed_glyphs": changed,
        "improved_glyphs": improved,
        "regressed_glyphs": regressed,
        "net_glyph_improvement": improved - regressed,
        "improved_formulas": formula_improved,
        "regressed_formulas": formula_regressed,
        "changes": changes,
    }


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _mapped_ids(element: ET.Element, href_to_record: dict[str, str]) -> list[str]:
    output = []
    for child in element.iter():
        identifier = child.attrib.get(XML_ID) or child.attrib.get("id")
        if identifier in href_to_record:
            output.append(href_to_record[identifier])
    return list(dict.fromkeys(output))


def _truth_edges(path: Path, rows: list[dict]) -> set[tuple[str, str, str]]:
    root = ET.parse(path).getroot()
    trace_ids = {
        str(node.attrib.get("id", "")).strip()
        for node in root.iter() if _local_name(node.tag) == "trace"
    }
    by_group_index = {
        int(str(row["record_id"]).rsplit(":", 1)[1]): str(row["record_id"])
        for row in rows
    }
    href_to_record: dict[str, str] = {}
    group_index = 0
    for group in root.iter():
        if _local_name(group.tag) != "traceGroup":
            continue
        annotation = next((
            child for child in group
            if _local_name(child.tag) == "annotation"
            and child.attrib.get("type") == "truth" and child.text
        ), None)
        refs = [
            str(child.attrib.get("traceDataRef", "")).strip()
            for child in group if _local_name(child.tag) == "traceView"
        ]
        if annotation is None or not refs or any(ref not in trace_ids for ref in refs):
            continue
        href = next((
            str(child.attrib.get("href", "")).strip()
            for child in group
            if _local_name(child.tag) == "annotationXML" and child.attrib.get("href")
        ), "")
        if href and group_index in by_group_index:
            href_to_record[href] = by_group_index[group_index]
        group_index += 1

    boxes = {str(row["record_id"]): row["geometry"] for row in rows}
    edges: set[tuple[str, str, str]] = set()
    for element in root.iter():
        tag, children = _local_name(element.tag), list(element)
        slots: list[tuple[int, str]] = []
        if tag == "msup" and len(children) >= 2:
            slots = [(1, "superscript")]
        elif tag == "msub" and len(children) >= 2:
            slots = [(1, "subscript")]
        elif tag == "msubsup" and len(children) >= 3:
            slots = [(1, "subscript"), (2, "superscript")]
        elif tag == "munder" and len(children) >= 2:
            slots = [(1, "subscript")]
        elif tag == "mover" and len(children) >= 2:
            slots = [(1, "superscript")]
        elif tag == "munderover" and len(children) >= 3:
            slots = [(1, "subscript"), (2, "superscript")]
        if slots:
            bases = _mapped_ids(children[0], href_to_record)
            if bases:
                parent = max(bases, key=lambda record_id: float(boxes[record_id]["right"]))
                for slot, relation in slots:
                    edges.update(
                        (parent, child, relation)
                        for child in _mapped_ids(children[slot], href_to_record)
                        if child != parent
                    )
        identifier = element.attrib.get(XML_ID) or element.attrib.get("id")
        anchor = href_to_record.get(str(identifier))
        if anchor is None:
            continue
        if tag == "mfrac" and len(children) >= 2:
            for child in _mapped_ids(children[0], href_to_record):
                if child != anchor:
                    edges.add((anchor, child, "above"))
            for child in _mapped_ids(children[1], href_to_record):
                if child != anchor:
                    edges.add((anchor, child, "below"))
        elif tag in {"msqrt", "mroot"} and children:
            content = children[0] if tag == "mroot" else element
            for child in _mapped_ids(content, href_to_record):
                if child != anchor:
                    edges.add((anchor, child, "contains"))
    return edges


def _relation_metrics(
    rows: list[dict], root: Path, script_predictor=None,
    predictions: dict[str, str] | None = None,
) -> dict:
    formulas = _grouped(rows)
    totals = {relation: Counter(tp=0, fp=0, fn=0) for relation in sorted(STRUCTURAL)}
    exact = parsed = 0
    hwr_exact = final_exact = oracle_exact = 0
    relation_hwr_exact = relation_final_exact = relation_oracle_exact = 0
    failures = []
    root = root.resolve()
    for formula_id, sequence in formulas.items():
        path = (root / formula_id).resolve()
        if root not in path.parents or not path.is_file():
            continue
        truth = _truth_edges(path, sequence)
        layout = infer_formula_layout(sequence, script_predictor=script_predictor)
        predicted = {
            (str(edge["parent"]), str(edge["child"]), str(edge["type"]))
            for edge in layout["edges"] if edge["type"] in STRUCTURAL
        }
        parsed += 1
        relation_ok = predicted == truth
        exact += relation_ok
        hwr_ok = all(str(row["final_topk"][0]) == str(row["label"]) for row in sequence)
        final_ok = bool(predictions) and all(
            predictions.get(str(row["record_id"])) == str(row["label"])
            for row in sequence
        )
        oracle_ok = all(str(row["label"]) in row["final_topk"] for row in sequence)
        hwr_exact += hwr_ok
        final_exact += final_ok
        oracle_exact += oracle_ok
        relation_hwr_exact += relation_ok and hwr_ok
        relation_final_exact += relation_ok and final_ok
        relation_oracle_exact += relation_ok and oracle_ok
        for relation in totals:
            expected = {edge for edge in truth if edge[2] == relation}
            actual = {edge for edge in predicted if edge[2] == relation}
            totals[relation]["tp"] += len(actual & expected)
            totals[relation]["fp"] += len(actual - expected)
            totals[relation]["fn"] += len(expected - actual)
        if predicted != truth and len(failures) < 25:
            failures.append({
                "formula_id": formula_id,
                "truth": sorted(truth), "predicted": sorted(predicted),
            })
    by_relation, micro = {}, Counter(tp=0, fp=0, fn=0)
    for relation, values in totals.items():
        tp, fp, fn = values["tp"], values["fp"], values["fn"]
        precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        by_relation[relation] = {
            **dict(values), "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        }
        micro.update(values)
    precision = micro["tp"] / max(micro["tp"] + micro["fp"], 1)
    recall = micro["tp"] / max(micro["tp"] + micro["fn"], 1)
    return {
        "formulas": parsed, "formula_exact": exact / max(parsed, 1),
        "formula_exact_count": exact,
        "character_formula_exact": {
            "hwr_top1": hwr_exact / max(parsed, 1),
            "context_finalized": final_exact / max(parsed, 1) if predictions else None,
            "top5_oracle": oracle_exact / max(parsed, 1),
        },
        "relation_and_character_formula_exact": {
            "hwr_top1": relation_hwr_exact / max(parsed, 1),
            "context_finalized": relation_final_exact / max(parsed, 1) if predictions else None,
            "top5_oracle": relation_oracle_exact / max(parsed, 1),
            "context_finalized_count": relation_final_exact if predictions else None,
        },
        "micro": {
            **dict(micro), "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        },
        "by_relation": by_relation, "failures": failures,
        "license_boundary": "CROHME noncommercial evaluation only; no fitting or threshold selection",
    }


def _self_test() -> None:
    rows = [
        {
            "record_id": record_id, "formula_id": "f", "label": label,
            "final_topk": [label], "final_topk_probabilities": [1.0],
            "geometry": {
                "left": left, "top": 0.0, "right": left + 5.0, "bottom": 10.0,
                "center_x": left + 2.5, "center_y": 5.0,
            },
            "context": {"index": truth_index, "length": 2},
        }
        for record_id, label, left, truth_index in (("b", "+", 20.0, 1), ("a", "1", 0.0, 0))
    ]
    metrics = _order_metrics(rows, {"a": "1", "b": "+"})
    assert metrics["layout_order_exact"] == 1.0
    assert metrics["layout_and_character_formula_exact"]["context_finalized"] == 1.0
    assert _flat_structure_gate(rows)["passed"]
    acceptance = {
        "formulas": [{
            "formula_id": "f", "rows": rows,
            "truth_relations": [],
        }],
    }
    assert _owned_relation_metrics(acceptance)["formula_exact_count"] == 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-candidates", type=Path)
    parser.add_argument("--direct-finalized", type=Path)
    parser.add_argument("--crohme-candidates", type=Path)
    parser.add_argument("--crohme-finalized", type=Path)
    parser.add_argument("--crohme-baseline-finalized", type=Path)
    parser.add_argument("--crohme-root", type=Path)
    parser.add_argument("--script-layout-checkpoint", type=Path)
    parser.add_argument("--owned-relation-acceptance", type=Path)
    parser.add_argument("--promote-finalized-layout-evidence", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        print(json.dumps({"self_test": "pass"}))
        return 0
    if args.direct_candidates is None or args.output is None:
        parser.error("--direct-candidates and --output are required")
    direct = list(_json_lines(args.direct_candidates))
    finalized = _predictions(args.direct_finalized)
    script_predictor = (
        NeuralScriptPredictor(args.script_layout_checkpoint)
        if args.script_layout_checkpoint else None
    )
    if finalized and set(finalized) != {str(row["record_id"]) for row in direct}:
        raise ValueError("direct finalized prediction coverage mismatch")
    evidence_audits = {}
    if args.promote_finalized_layout_evidence:
        direct, evidence_audits["direct"] = selected_layout_evidence_rows(
            direct, finalized,
        )
    acceptance = _owned_relation_acceptance(args.owned_relation_acceptance)
    relation_formula_ids = frozenset(
        str(row["formula_id"]) for row in (acceptance or {}).get("formulas", [])
    )
    report = {
        "schema": SCHEMA, "generated_at": datetime.now(timezone.utc).isoformat(),
        "layer": {
            "input": "immutable symbol boxes plus HWR Top-5",
            "output": "geometry-derived order and spatial relation graph",
            "grouping_mutations": 0, "threshold_selection": "none",
            **(
                {"selected_finalized_layout_evidence": evidence_audits}
                if args.promote_finalized_layout_evidence else {}
            ),
        },
        "direct": {
            "metrics": _order_metrics(direct, finalized, script_predictor),
            "flat_structure_gate": _flat_structure_gate(
                direct, script_predictor, relation_formula_ids,
            ),
            "candidates_sha256": _sha256(args.direct_candidates),
            "finalized_sha256": _sha256(args.direct_finalized) if args.direct_finalized else None,
            "interpretation": "accepted ownership order and explicit relation overlay are evaluation truth only and are not read by the layout layer",
        },
    }
    if acceptance is not None:
        report["owned_relation_acceptance"] = {
            "artifact_sha256": _sha256(args.owned_relation_acceptance),
            "status": str(acceptance.get("status")),
            "relation_only_character_oracle": _owned_relation_metrics(
                acceptance, script_predictor=script_predictor,
            ),
            "matched_direct_hwr": _owned_relation_metrics(
                acceptance, _grouped(direct), finalized, script_predictor,
            ),
            "commercial_training_rights": str(acceptance.get("commercial_training_rights")),
        }
    if args.crohme_candidates is not None:
        crohme = list(_json_lines(args.crohme_candidates))
        crohme_finalized = _predictions(args.crohme_finalized)
        crohme_baseline = _predictions(args.crohme_baseline_finalized)
        if crohme_finalized and set(crohme_finalized) != {
            str(row["record_id"]) for row in crohme
        }:
            raise ValueError("CROHME finalized prediction coverage mismatch")
        if crohme_baseline and set(crohme_baseline) != {
            str(row["record_id"]) for row in crohme
        }:
            raise ValueError("CROHME baseline prediction coverage mismatch")
        if args.promote_finalized_layout_evidence:
            crohme, evidence_audits["crohme"] = selected_layout_evidence_rows(
                crohme, crohme_finalized,
            )
        report["crohme_noncommercial"] = {
            "tracegroup_order_proxy": _order_metrics(
                crohme, crohme_finalized, script_predictor
            ),
            "relation_graph": _relation_metrics(
                crohme, args.crohme_root, script_predictor, crohme_finalized
            ) if args.crohme_root else None,
            "baseline_tracegroup_order_proxy": (
                _order_metrics(crohme, crohme_baseline, script_predictor)
                if crohme_baseline else None
            ),
            "finalizer_delta": _prediction_delta(
                crohme, crohme_baseline, crohme_finalized
            ),
            "candidates_sha256": _sha256(args.crohme_candidates),
            "finalized_sha256": (
                _sha256(args.crohme_finalized) if args.crohme_finalized else None
            ),
            "baseline_finalized_sha256": (
                _sha256(args.crohme_baseline_finalized)
                if args.crohme_baseline_finalized else None
            ),
            "product_validation": False,
        }
    report["script_layout_checkpoint"] = (
        {"path": str(args.script_layout_checkpoint), "sha256": _sha256(args.script_layout_checkpoint)}
        if args.script_layout_checkpoint else None
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        parser.error(f"refusing to overwrite layout report: {args.output}")
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    print(json.dumps({
        "output": str(args.output), "sha256": _sha256(args.output),
        "direct": report["direct"]["metrics"],
        "crohme_relation": report.get("crohme_noncommercial", {}).get("relation_graph"),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
