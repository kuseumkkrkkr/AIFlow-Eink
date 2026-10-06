#!/usr/bin/env python3
"""Build a frozen CROHME split candidate cache for research-only evaluation.

CROHME is CC BY-NC in this workspace.  The output must never enter a product
training directory.  This script performs HWR inference only; it does not fit
or modify the shape encoder, character head, grouping, or source InkML files.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import xml.etree.ElementTree as ET

import torch

from audit_replay_protocol_v1 import CROHME_ALIASES
from evaluate_48hz_prefix_v1 import _load_model, _ordinal_48hz, _sha256
from evaluate_homograph_context_reranker_v1 import (
    _decorate,
    _geometry,
    _metrics,
    _score_items,
)
from train_independent_formula_context_v1 import DEFAULT_HWR
from replay_evaluate_hwr_v1 import _digest, _trace_points


SCHEMA = "aiflow-crohme-standard-candidate-cache/v1"
REPORT_SCHEMA = "aiflow-crohme-standard-candidate-build-report/v1"


def _event(name: str, **values: object) -> None:
    print(json.dumps({"event": name, **values}, ensure_ascii=False), flush=True)


def _d_path(path: Path, label: str, *, must_exist: bool = True) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.drive.upper() != "D:":
        raise ValueError(f"{label} must remain on D: {resolved}")
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"missing {label}: {resolved}")
    return resolved


def _local_name(node: ET.Element) -> str:
    return node.tag.rsplit("}", 1)[-1]


def _parse_root(path: Path) -> tuple[ET.Element, dict | None]:
    try:
        return ET.parse(path).getroot(), None
    except ET.ParseError as error:
        raw = path.read_bytes()
        try:
            decoded = raw.decode("latin-1")
            root = ET.fromstring(decoded)
        except (UnicodeDecodeError, ET.ParseError) as fallback_error:
            raise ET.ParseError(
                f"unrecoverable InkML {path}: utf8={error}; "
                f"latin1={fallback_error}"
            ) from fallback_error
        return root, {
            "record_id": path.name,
            "original_error": str(error),
            "fallback_encoding": "latin-1",
            "file_sha256": _sha256(path),
        }


def _writer(path: Path) -> str:
    root, _fallback = _parse_root(path)
    values = [
        str(node.text).strip()
        for node in root.iter()
        if (
            _local_name(node) == "annotation"
            and node.attrib.get("type") == "writer"
            and node.text
            and str(node.text).strip()
        )
    ]
    if values:
        return values[0]
    match = re.match(r"^(\d+-\d+)-\d+$", path.stem)
    return (
        f"inferred-session:{match.group(1)}"
        if match
        else f"missing-writer:{path.stem}"
    )


def _sample(path: Path) -> tuple[dict | None, dict | None]:
    root, fallback = _parse_root(path)
    trace_ids: list[str] = []
    traces: dict[str, list[tuple[float, float, None]]] = {}
    for node in root.iter():
        if _local_name(node) != "trace":
            continue
        points = _trace_points(node.text)
        if points:
            trace_id = node.attrib.get("id", str(len(trace_ids))).strip()
            trace_ids.append(trace_id)
            traces[trace_id] = [(point[0], point[1], None) for point in points]
    index_by_id = {trace_id: index for index, trace_id in enumerate(trace_ids)}
    groups: list[list[int]] = []
    labels: list[str] = []
    for group in root.iter():
        if _local_name(group) != "traceGroup":
            continue
        annotation = next((
            child
            for child in group
            if (
                _local_name(child) == "annotation"
                and child.attrib.get("type") == "truth"
                and child.text
            )
        ), None)
        refs = [
            child.attrib.get("traceDataRef", "").strip()
            for child in group
            if _local_name(child) == "traceView"
        ]
        if annotation is None or not refs or any(ref not in index_by_id for ref in refs):
            continue
        raw_label = annotation.text.strip().replace("$", "")
        labels.append(CROHME_ALIASES.get(raw_label, raw_label))
        groups.append([index_by_id[ref] for ref in refs])
    if not groups:
        return None, fallback
    return {
        "strokes": [traces[trace_id] for trace_id in trace_ids],
        "groups": groups,
        "labels": labels,
        "partition_complete": (
            set().union(*(set(group) for group in groups))
            == set(range(len(trace_ids)))
        ),
    }, fallback


def _crohme_items_tolerant(
    root: Path, labels: set[str], split_name: str,
) -> tuple[list[dict], dict, set[str], set[str], list[dict], dict[str, str]]:
    duplicates = []
    parse_fallbacks: dict[str, dict] = {}
    seen = set()
    items: list[dict] = []
    unsupported = Counter()
    invalid = Counter()
    all_formula_ids: set[str] = set()
    failed_formula_ids: set[str] = set()
    incomplete_truth_partition_formulas = 0
    writers: dict[str, str] = {}
    for path in sorted(root.rglob("*.inkml")):
        parsed, fallback = _parse_root(path)
        record_id = path.relative_to(root).as_posix()
        if fallback is not None:
            fallback = dict(fallback, record_id=record_id)
            parse_fallbacks[fallback["record_id"]] = fallback
        trace_ids: list[str] = []
        traces: dict[str, list[list[float]]] = {}
        for node in parsed.iter():
            if _local_name(node) != "trace":
                continue
            points = _trace_points(node.text)
            if points:
                trace_id = node.attrib.get("id", str(len(trace_ids))).strip()
                trace_ids.append(trace_id)
                traces[trace_id] = points
        truths = [
            str(node.text or "").strip()
            for node in parsed.iter()
            if (
                _local_name(node) == "annotation"
                and node.attrib.get("type") == "truth"
            )
        ]
        if not trace_ids or not truths:
            continue
        fingerprint = _digest([truths[0], [traces[key] for key in trace_ids]])
        if fingerprint in seen:
            duplicates.append(record_id)
            continue
        seen.add(fingerprint)
        all_formula_ids.add(record_id)
        writer_values = [
            str(node.text).strip()
            for node in parsed.iter()
            if (
                _local_name(node) == "annotation"
                and node.attrib.get("type") == "writer"
                and node.text
                and str(node.text).strip()
            )
        ]
        if writer_values:
            writers[record_id] = writer_values[0]
        else:
            inferred = re.match(r"^(\d+-\d+)-\d+$", path.stem)
            writers[record_id] = (
                f"inferred-session:{inferred.group(1)}"
                if inferred
                else f"missing-writer:{path.stem}"
            )
        index_by_id = {trace_id: index for index, trace_id in enumerate(trace_ids)}
        groups: list[list[int]] = []
        group_labels: list[str] = []
        for group in parsed.iter():
            if _local_name(group) != "traceGroup":
                continue
            annotation = next((
                child
                for child in group
                if (
                    _local_name(child) == "annotation"
                    and child.attrib.get("type") == "truth"
                    and child.text
                )
            ), None)
            refs = [
                child.attrib.get("traceDataRef", "").strip()
                for child in group
                if _local_name(child) == "traceView"
            ]
            if (
                annotation is None
                or not refs
                or any(ref not in index_by_id for ref in refs)
            ):
                continue
            raw_label = annotation.text.strip().replace("$", "")
            group_labels.append(CROHME_ALIASES.get(raw_label, raw_label))
            groups.append([index_by_id[ref] for ref in refs])
        if not groups:
            failed_formula_ids.add(record_id)
            continue
        partition_complete = (
            set().union(*(set(group) for group in groups))
            == set(range(len(trace_ids)))
        )
        if not partition_complete:
            failed_formula_ids.add(record_id)
            incomplete_truth_partition_formulas += 1
        for index, (group, raw_label) in enumerate(
            zip(groups, group_labels, strict=True)
        ):
            label = CROHME_ALIASES.get(raw_label, raw_label)
            if label not in labels:
                unsupported[label] += 1
                failed_formula_ids.add(record_id)
                continue
            strokes = [
                [(point[0], point[1], None) for point in traces[trace_ids[stroke_index]]]
                for stroke_index in group
            ]
            if sum(len(stroke) for stroke in strokes) < 2:
                invalid[label] += 1
                failed_formula_ids.add(record_id)
                continue
            items.append({
                "record_id": f"{record_id}:{index}",
                "label": label,
                "source": f"crohme2019_{split_name}",
                "formula_id": record_id,
                "strokes": _ordinal_48hz(strokes),
            })
    coverage = {
        "protocol_formulas": len(all_formula_ids),
        "exact_formula_duplicates_removed": len(duplicates),
        "unsupported_truth_groups": sum(unsupported.values()),
        "unsupported_labels": dict(unsupported.most_common()),
        "invalid_single_point_truth_groups": sum(invalid.values()),
        "incomplete_truth_partition_formulas": incomplete_truth_partition_formulas,
        "fully_supported_formulas": len(all_formula_ids - failed_formula_ids),
        "parse_fallback_files": len(parse_fallbacks),
    }
    return (
        items,
        coverage,
        all_formula_ids,
        failed_formula_ids,
        [parse_fallbacks[key] for key in sorted(parse_fallbacks)],
        writers,
    )


def _directory_digest(root: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    files = sorted(root.glob("*.inkml"), key=lambda value: value.name)
    total_bytes = 0
    for path in files:
        size = path.stat().st_size
        total_bytes += size
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest(), len(files), total_bytes


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(
            filename="", fileobj=raw, mode="wb", compresslevel=6, mtime=0
        ) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8", newline="\n") as stream:
                for row in rows:
                    stream.write(
                        json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                        + "\n"
                    )


def _candidate_metrics(rows: list[dict]) -> dict:
    predictions = {
        str(row["record_id"]): str(row["final_topk"][0]) for row in rows
    }
    metrics = _metrics(rows, predictions)
    formulae: dict[str, list[dict]] = defaultdict(list)
    ranks = Counter()
    for row in rows:
        formulae[str(row["formula_id"])].append(row)
        truth = str(row["label"])
        candidates = [str(value) for value in row["final_topk"]]
        rank = candidates.index(truth) + 1 if truth in candidates else None
        ranks[str(rank) if rank is not None else "outside"] += 1
    top5_hits = len(rows) - ranks.get("outside", 0)
    return {
        "all_top1": metrics["all_top1"],
        "formula_exact": metrics["formula_exact"],
        "strict_macro_top1": metrics["strict_macro_top1"],
        "strict_micro_top1": metrics["strict_micro_top1"],
        "top5": top5_hits / len(rows),
        "top5_count": top5_hits,
        "truth_rank_counts": dict(sorted(ranks.items())),
        "formula_top5_oracle_count": sum(
            all(str(row["label"]) in row["final_topk"] for row in sequence)
            for sequence in formulae.values()
        ),
        "formula_top5_oracle": sum(
            all(str(row["label"]) in row["final_topk"] for row in sequence)
            for sequence in formulae.values()
        )
        / len(formulae),
    }


def _validate_rows(rows: list[dict]) -> dict:
    record_ids = [str(row["record_id"]) for row in rows]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("duplicate CROHME candidate record IDs")
    bad_width = bad_unique = bad_probability = bad_context = 0
    for row in rows:
        candidates = [str(value) for value in row["final_topk"]]
        probabilities = [float(value) for value in row["final_topk_probabilities"]]
        bad_width += len(candidates) != 5 or len(probabilities) != 5
        bad_unique += len(candidates) != len(set(candidates))
        bad_probability += any(value < 0.0 or value > 1.0 for value in probabilities)
        context = row["context"]
        bad_context += not (
            0 <= int(context["index"]) < int(context["length"])
        )
    if any((bad_width, bad_unique, bad_probability, bad_context)):
        raise ValueError(
            "invalid candidate cache: "
            f"width={bad_width}, unique={bad_unique}, "
            f"probability={bad_probability}, context={bad_context}"
        )
    return {
        "duplicate_record_ids": 0,
        "bad_candidate_width": 0,
        "duplicate_candidates": 0,
        "bad_probabilities": 0,
        "bad_context_indices": 0,
        "candidate_width": 5,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--split-name", choices=("train", "valid", "test"), required=True)
    parser.add_argument("--hwr-checkpoint", type=Path, default=DEFAULT_HWR)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    split_root = _d_path(args.split_root, "CROHME split root")
    hwr_path = _d_path(args.hwr_checkpoint, "frozen HWR checkpoint")
    output = _d_path(args.output, "candidate output", must_exist=False)
    report_output = _d_path(args.report_output, "candidate report", must_exist=False)
    if output.exists() or report_output.exists():
        parser.error("refusing to overwrite CROHME candidate evidence")

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")

    directory_sha, inkml_files, source_bytes = _directory_digest(split_root)
    model, labels, _payload = _load_model(hwr_path, device)
    items, coverage, formula_ids, failed_formula_ids, parse_fallbacks, writers = (
        _crohme_items_tolerant(split_root, set(labels), args.split_name)
    )
    if not items:
        raise ValueError("CROHME split produced no supported glyphs")
    inferred_writers = sum(
        value.startswith("inferred-session:") for value in writers.values()
    )
    missing_writers = sum(
        value.startswith("missing-writer:") for value in writers.values()
    )
    scores = _score_items(items, model, labels, device, batch_size=args.batch_size)
    geometry = _geometry(items)
    saved_rows = [
        {
            "record_id": str(item["record_id"]),
            "label": str(item["label"]),
            "source": f"crohme2019_{args.split_name}",
            "formula_id": str(item["formula_id"]),
            "writer_group": writers[str(item["formula_id"])],
            **scores[str(item["record_id"])],
        }
        for item in items
    ]
    rows = _decorate(saved_rows, items, scores, geometry)
    for row in rows:
        row["dataset_split"] = args.split_name
        row["data_rights"] = "CC BY-NC research-only"
        row["schema"] = SCHEMA
    validation = _validate_rows(rows)
    metrics = _candidate_metrics(rows)
    _write_rows(output, rows)

    formulas_per_writer = Counter(
        writers[formula_id] for formula_id in formula_ids
    )
    report = {
        "schema": REPORT_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "split": args.split_name,
        "rights": {
            "dataset": "CROHME2019",
            "license_in_workspace": "CC BY-NC 4.0",
            "research_only": True,
            "product_training_eligible": False,
            "commercial_checkpoint_adoption_forbidden": True,
        },
        "training_performed": False,
        "shape_hwr_frozen": True,
        "truth_grouping_supplied": True,
        "official_end_to_end_expression_rate": False,
        "counts": {
            "inkml_files": inkml_files,
            "protocol_formulas": len(formula_ids),
            "supported_records": len(rows),
            "supported_formulas": len({str(row["formula_id"]) for row in rows}),
            "fully_supported_formulas": int(coverage["fully_supported_formulas"]),
            "failed_or_partial_formula_ids": len(failed_formula_ids),
            "writers": len(set(writers.values())),
            "missing_writer_annotations": inferred_writers + missing_writers,
            "writer_inferred_from_filename_session": inferred_writers,
            "unresolved_writer_formulas": missing_writers,
            "minimum_formulas_per_writer": min(formulas_per_writer.values()),
            "maximum_formulas_per_writer": max(formulas_per_writer.values()),
        },
        "coverage": coverage,
        "parse_fallbacks": parse_fallbacks,
        "hwr_metrics": metrics,
        "validation": validation,
        "inputs": {
            "split_root": str(split_root),
            "split_directory_sha256": directory_sha,
            "split_source_bytes": source_bytes,
            "hwr_checkpoint": str(hwr_path),
            "hwr_checkpoint_sha256": _sha256(hwr_path),
        },
        "output": {
            "path": str(output),
            "sha256": _sha256(output),
        },
    }
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    _event(
        "crohme_candidate_build_complete",
        split=args.split_name,
        records=len(rows),
        formulas=len({str(row["formula_id"]) for row in rows}),
        writers=len(set(writers.values())),
        top1=metrics["all_top1"],
        top5=metrics["top5"],
        output=str(output),
        output_sha256=_sha256(output),
        report=str(report_output),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
