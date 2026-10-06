#!/usr/bin/env python3
"""Build v7 raw-writer evidence with external-synthetic fallback only."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import torch

from build_normalized_ink_v1 import SourceSample, _canonicalize
from character_tensor_v1 import tensorize
from cleanroom_trajectory_profiles_v3 import trajectory_descriptor
from cleanroom_writer_style_simulator_v4 import WRITER_STYLES
from hierarchical_writer_model_v5 import (
    DEFAULT_CHECKPOINT, DEFAULT_EXTERNAL, GlyphPlacementV5, FormulaPlanV5,
    WRITERS, _catalog, _compose_formula, _draw, _hwr_preservation,
    _label_safe_catalog, _quadrant, _direction_signature, _select,
)
from train_character_classifier_v1 import InkClassifierV1


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FROZEN = Path(
    r"D:\AIFlow-Workspace\PrivateData\math-ink-data-collector\derived"
    r"\fresh-context-acceptance-20260820-r2\frozen_acceptance"
)
DEFAULT_OUTPUT = ROOT / "artifacts/hierarchical_writer_model_v7_20260823_r1"
SCHEMA = "aiflow-hierarchical-writer-model/v7"
SAMPLE_ID = re.compile(r'"sample_id"\s*:\s*"([^"]+)"')


@dataclass(frozen=True)
class RawAllographPlanV7:
    token: str
    label_index: int
    source_domain: str
    source_row: int
    stroke_count: int
    start_quadrant: str
    end_quadrant: str
    direction_signature: tuple[int, ...]
    evidence_tier: str
    sample_id: str
    writer_id: str
    session_group: str
    group_index: int
    stroke_indices: tuple[int, ...]
    raw_stroke_sha256: str
    formulas_sha256: str
    ownership_sha256: str


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _line_sample_id(line: str) -> str:
    match = SAMPLE_ID.search(line)
    if match is None:
        raise ValueError("JSONL row has no sample_id before parsing")
    return match.group(1)


def fresh_sample_ids(path: Path) -> set[str]:
    """Read only the pre-frozen fresh sample IDs used for build exclusion."""
    identifiers = {_line_sample_id(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    if len(identifiers) != 53:
        raise ValueError(f"expected 53 frozen replay IDs, got {len(identifiers)}")
    return identifiers


def legacy_rows(frozen: Path) -> tuple[dict[str, dict], list[dict], dict]:
    """Parse legacy rows while skipping fresh row bodies before json.loads."""
    fresh_path = frozen / "ownership_fresh_acceptance.jsonl"
    formulas_path = frozen / "frozen_dataset/data/formulas_valid.jsonl"
    ownership_path = frozen / "frozen_dataset/data/ownership_train.jsonl"
    info_path = frozen / "frozen_dataset/dataset_info.json"
    fresh = fresh_sample_ids(fresh_path)
    ownership = []; skipped_ownership = 0
    with ownership_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            sample_id = _line_sample_id(line)
            if sample_id in fresh:
                skipped_ownership += 1
                continue
            ownership.append(json.loads(line))
    legacy_ids = {str(row["sample_id"]) for row in ownership}
    formulas = {}; skipped_formula = 0; unowned_formula = 0
    with formulas_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            sample_id = _line_sample_id(line)
            if sample_id in fresh:
                skipped_formula += 1
                continue
            if sample_id not in legacy_ids:
                unowned_formula += 1
                continue
            formulas[sample_id] = json.loads(line)
    if len(formulas) != 96 or len(ownership) != 96 or skipped_formula != 53 or skipped_ownership != 53 or unowned_formula != 10:
        raise ValueError("legacy/fresh frozen partition does not match 96/53 contract")
    hashes = {
        "formulas": sha256(formulas_path), "ownership": sha256(ownership_path),
        "fresh_id_exclusion": sha256(fresh_path), "dataset_info": sha256(info_path),
    }
    return formulas, ownership, {
        "hashes": hashes, "legacy_formulae": 96, "fresh_rows_body_parsed": 0,
        "fresh_formulae_skipped_before_parse": skipped_formula,
        "fresh_ownership_skipped_before_parse": skipped_ownership,
        "unowned_legacy_formulae_skipped_before_parse": unowned_formula,
        "fresh_sample_ids": len(fresh),
    }


def _raw_stroke_hash(strokes: list[dict], indices: tuple[int, ...]) -> str:
    selected = [strokes[index] for index in indices]
    value = json.dumps(selected, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_raw_catalog(
    formulas: dict[str, dict], ownership: list[dict], labels: list[str], hashes: dict,
    *, source_domain: str, evidence_tier: str,
) -> tuple[dict, np.ndarray, list[dict]]:
    label_to_index = {token: index for index, token in enumerate(labels)}
    catalog: dict[str, list[tuple[RawAllographPlanV7, np.ndarray]]] = defaultdict(list)
    features = []; metadata = []
    for annotation in ownership:
        if not annotation.get("accepted"):
            raise ValueError(f"unaccepted legacy ownership: {annotation.get('sample_id')}")
        sample_id = str(annotation["sample_id"]); source = formulas.get(sample_id)
        if source is None:
            raise ValueError(f"legacy formula missing: {sample_id}")
        if str(source["writer_id"]) != str(annotation["writer_id"]):
            raise ValueError(f"writer mismatch: {sample_id}")
        strokes = sorted(source["strokes"], key=lambda row: int(row["order"]))
        if [int(row["order"]) for row in strokes] != list(range(len(strokes))):
            raise ValueError(f"non-contiguous stroke order: {sample_id}")
        groups = [tuple(int(value) for value in group) for group in annotation["groups"]]
        tokens = [str(value) for value in annotation["labels"]]
        target = [str(cell["token"]) for cell in source.get("target_cells") or []]
        assigned = [value for group in groups for value in group]
        if tokens != target or len(groups) != len(tokens):
            raise ValueError(f"label/target contract mismatch: {sample_id}")
        if sorted(assigned) != list(range(len(strokes))) or len(assigned) != len(set(assigned)):
            raise ValueError(f"ownership is not an exact stroke partition: {sample_id}")
        for group_index, (indices, token) in enumerate(zip(groups, tokens, strict=True)):
            raw_strokes = [
                [(float(point["x"]), float(point["y"]), float(point["t_ms"])) for point in strokes[index]["points"]]
                for index in indices
            ]
            canonical = _canonicalize(SourceSample(
                "project_owned_legacy_raw", f"{sample_id}:{group_index}", token,
                "project_owned_legacy_writer_catalog", "raw_observed_writer_evidence",
                raw_strokes,
            ))
            tensor = tensorize(canonical)
            descriptor = trajectory_descriptor(tensor)
            plan = RawAllographPlanV7(
                token, label_to_index.get(token, -1), source_domain,
                len(features), descriptor.stroke_count, _quadrant(tensor[0, :2]),
                _quadrant(tensor[-1, :2]), _direction_signature(tensor),
                evidence_tier, sample_id,
                str(annotation["writer_id"]), str(source.get("session_group", "")),
                group_index, indices, _raw_stroke_hash(strokes, indices),
                hashes["formulas"], hashes["ownership"],
            )
            row = asdict(plan)
            row["canonical_record_id"] = canonical["record_id"]
            row["duration_ms"] = canonical["transform"]["duration_ms"]
            row["source_point_count"] = canonical["point_count"]
            metadata.append(row); features.append(tensor)
            catalog[token].append((plan, tensor))
    return dict(catalog), np.stack(features).astype(np.float32), metadata


def writer_profiles(metadata: list[dict], features: np.ndarray) -> dict:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(metadata):
        grouped[row["writer_id"]].append(index)
    profiles = {}
    for writer, indices in sorted(grouped.items()):
        rows = [metadata[index] for index in indices]
        tensors = features[indices]
        widths = np.ptp(tensors[:, :, 0], axis=1); heights = np.ptp(tensors[:, :, 1], axis=1)
        durations = [float(row["duration_ms"]) for row in rows if row["duration_ms"] is not None]
        profiles[writer] = {
            "evidence_tier": "raw_observed_legacy_writer_profile",
            "formulae": len({row["sample_id"] for row in rows}),
            "sessions": sorted({row["session_group"] for row in rows}),
            "glyphs": len(rows), "classes": len({row["token"] for row in rows}),
            "label_counts": dict(sorted(Counter(row["token"] for row in rows).items())),
            "stroke_count_distribution": dict(sorted(Counter(row["stroke_count"] for row in rows).items())),
            "median_aspect_ratio": float(np.median(widths / np.maximum(heights, 1e-6))),
            "median_duration_ms": float(np.median(durations)) if durations else None,
        }
    return profiles


def _formula_plans_v7() -> tuple[FormulaPlanV5, ...]:
    def baseline(formula_id: str, display: str, tokens: str) -> FormulaPlanV5:
        glyphs = tuple(GlyphPlacementV5(f"{formula_id}:g{i}", token, i * 1.05, 0.0, 1.0, "baseline") for i, token in enumerate(tokens))
        return FormulaPlanV5(formula_id, display, glyphs, (), tuple((row.glyph_id,) for row in glyphs))
    linear = baseline("linear", "1 + 1 = 2", "1+1=2")
    power_glyphs = (
        GlyphPlacementV5("power:x", "x", 0.0, 0.0, 1.0, "baseline"),
        GlyphPlacementV5("power:2a", "2", 0.70, -0.66, 0.56, "superscript", "power:x"),
        GlyphPlacementV5("power:+", "+", 1.45, 0.0, 1.0, "baseline"),
        GlyphPlacementV5("power:y", "y", 2.50, 0.0, 1.0, "baseline"),
        GlyphPlacementV5("power:2b", "2", 3.18, -0.66, 0.56, "superscript", "power:y"),
        GlyphPlacementV5("power:=", "=", 3.95, 0.0, 1.0, "baseline"),
        GlyphPlacementV5("power:2c", "2", 5.00, 0.0, 1.0, "baseline"),
        GlyphPlacementV5("power:5", "5", 6.00, 0.0, 1.0, "baseline"),
    )
    power = FormulaPlanV5(
        "power", "x^2 + y^2 = 25", power_glyphs,
        (("power:x", "power:2a", "superscript"), ("power:y", "power:2b", "superscript")),
        (("power:x", "power:+", "power:y", "power:=", "power:2c", "power:5"), ("power:2a", "power:2b")),
    )
    fraction_glyphs = (
        GlyphPlacementV5("fraction:a", "a", 0.20, -0.72, 0.72, "numerator"),
        GlyphPlacementV5("fraction:+", "+", 1.02, -0.72, 0.72, "numerator"),
        GlyphPlacementV5("fraction:b", "b", 1.84, -0.72, 0.72, "numerator"),
        GlyphPlacementV5("fraction:bar", "-", 1.02, 0.0, 2.60, "fraction_bar"),
        GlyphPlacementV5("fraction:2", "2", 1.02, 0.78, 0.72, "denominator"),
    )
    fraction = FormulaPlanV5(
        "fraction", "(a + b) / 2", fraction_glyphs,
        (("fraction:bar", "fraction:a", "numerator"), ("fraction:bar", "fraction:+", "numerator"),
         ("fraction:bar", "fraction:b", "numerator"), ("fraction:bar", "fraction:2", "denominator")),
        (("fraction:a", "fraction:+", "fraction:b"), ("fraction:bar",), ("fraction:2",)),
    )
    return linear, power, fraction


def raw_hwr_diagnostic(features: np.ndarray, metadata: list[dict], checkpoint: Path) -> dict:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = InkClassifierV1(len(payload["math_labels"]), None)
    model.load_state_dict(payload["state_dict"], strict=True); model.eval()
    supported = [index for index, row in enumerate(metadata) if row["label_index"] >= 0]
    truth = np.asarray([metadata[index]["label_index"] for index in supported], dtype=np.int64)
    with torch.no_grad():
        rank = model(torch.from_numpy(features[supported]), "math").argsort(dim=1, descending=True)[:, :5].cpu().numpy()
    return {
        "role": "diagnostic_not_selection",
        "rows": len(supported), "unsupported_evidence_rows": len(metadata) - len(supported),
        "top1": float(np.mean(rank[:, 0] == truth)),
        "top5": float(np.mean(np.any(rank == truth[:, None], axis=1))),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--external", type=Path, default=DEFAULT_EXTERNAL)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    paths = [args.frozen.resolve(), args.external.resolve(), args.checkpoint.resolve(), args.output.resolve()]
    if any(path.drive.upper() != "D:" for path in paths):
        parser.error("all inputs and outputs must remain on D:")
    if args.output.exists():
        parser.error(f"refusing to overwrite output: {args.output}")

    formulas, ownership, partition = legacy_rows(args.frozen)
    external, labels, external_audit = _catalog(args.external, None, args.checkpoint)
    raw, raw_features, raw_metadata = build_raw_catalog(
        formulas, ownership, labels, partition["hashes"],
        source_domain="project_owned_legacy_raw_observed",
        evidence_tier="raw_observed_legacy_writer_evidence",
    )
    if len(raw_metadata) != 393 or len(raw) != 28:
        raise ValueError(f"raw legacy catalog mismatch: {len(raw_metadata)} glyphs, {len(raw)} classes")
    external_safe, external_label_gate = _label_safe_catalog(external, WRITERS, args.checkpoint)
    supported_raw_by_writer: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for token, rows in raw.items():
        for option in rows:
            if option[0].label_index >= 0:
                supported_raw_by_writer[option[0].writer_id][token].append(option)
    base_writer_ids = sorted(supported_raw_by_writer)[:len(WRITERS)]
    base_writer_map = {writer.writer_id: base for writer, base in zip(WRITERS, base_writer_ids, strict=True)}
    merged = {}
    source_policy = {}
    for writer in WRITERS:
        base_writer = base_writer_map[writer.writer_id]
        writer_raw = supported_raw_by_writer[base_writer]
        rows = {}; policy = {}
        for token in labels:
            if token in writer_raw:
                rows[token] = writer_raw[token]; policy[token] = f"raw_observed_legacy_primary:{base_writer}"
            elif token in external_safe[writer.writer_id]:
                rows[token] = external_safe[writer.writer_id][token]; policy[token] = "external_synthetic_fallback"
        merged[writer.writer_id] = rows; source_policy[writer.writer_id] = policy

    formulas_v7 = _formula_plans_v7()
    generated = [_compose_formula(merged[writer.writer_id], formula, writer) for formula in formulas_v7 for writer in WRITERS]
    output = args.output.resolve(); output.mkdir(parents=True)
    np.savez_compressed(output / "raw_legacy_writer_catalog.npz", features=raw_features, tokens=np.asarray([row["token"] for row in raw_metadata]))
    with (output / "raw_legacy_writer_catalog.metadata.jsonl").open("w", encoding="utf-8") as stream:
        for row in raw_metadata:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    profiles = writer_profiles(raw_metadata, raw_features)
    (output / "raw_writer_profiles.json").write_text(json.dumps(profiles, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output / "formula_generation.jsonl").open("w", encoding="utf-8") as stream:
        for row in generated:
            stream.write(json.dumps({key: value for key, value in row.items() if key != "glyph_tensors"}, ensure_ascii=False) + "\n")
    _draw(
        output / "raw_primary_formula_four_writers.png",
        [row for row in generated if row["formula_id"] == "power"],
        "Hierarchical writer model v7: fixed raw parent writer + synthetic fallback",
    )
    alphabet = [token for token in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz" if token in labels]
    hwr = _hwr_preservation(merged, WRITERS, args.checkpoint, alphabet)
    raw_counts = Counter(row["token"] for row in raw_metadata)
    selected_domains = Counter(
        item["plan"]["source_domain"] for row in generated for item in row["allographs"]
    )
    raw_parent_writers = {
        writer.writer_id: sorted({
            item["plan"]["writer_id"]
            for row in generated if row["writer_id"] == writer.writer_id
            for item in row["allographs"] if item["plan"]["source_domain"] == "project_owned_legacy_raw_observed"
        })
        for writer in WRITERS
    }
    report = {
        "schema": SCHEMA, "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "raw_writer_evidence_with_external_synthetic_fallback_shadow",
        "partition": partition,
        "raw_catalog": {
            "formulae": len(formulas), "writers": len(profiles), "glyph_groups": len(raw_metadata),
            "classes": len(raw), "writer_ids": sorted(profiles), "label_counts": dict(sorted(raw_counts.items())),
            "t_evidence_rows": raw_counts["t"], "t_output_supported": "t" in labels,
            "evidence_tier": "raw_observed_legacy_writer_evidence",
        },
        "external_fallback": {**external_audit, "role": "unsupported_raw_class_fallback_only"},
        "external_label_safety_gate": external_label_gate,
        "formula_generation": {
            "templates": len(formulas_v7), "writer_combinations": len(generated),
            "selected_source_domains": dict(selected_domains),
            "fixed_legacy_base_writer": base_writer_map,
            "raw_parent_writers_observed": raw_parent_writers,
            "maximum_raw_parent_writers_per_generated_writer": max(map(len, raw_parent_writers.values())),
        },
        "raw_hwr_diagnostic": raw_hwr_diagnostic(raw_features, raw_metadata, args.checkpoint),
        "styled_generation_hwr_label_safety": hwr,
        "gates": {
            "legacy_raw_only_build": True, "fresh_row_bodies_parsed_during_build": False,
            "raw_primary_for_supported_classes": True, "external_synthetic_fallback_only": True,
            "coherent_fixed_raw_parent_writer": max(map(len, raw_parent_writers.values())) <= 1,
            "t_catalog_evidence_only": raw_counts["t"] == 1 and "t" not in labels,
            "t_output_promotion": False, "known_replay_used_for_selection": False,
            "promotion_ready": False, "new_untouched_writer_formula_acceptance_available": False,
            "training_performed": False, "product_runtime_changed": False,
            "checkpoint_changed": False, "crohme_rows": 0, "mathwriting_rows": 0,
        },
        "inputs": {
            "frozen_root": str(args.frozen.resolve()),
            "external": {"path": str(args.external.resolve()), "sha256": sha256(args.external)},
            "checkpoint": {"path": str(args.checkpoint.resolve()), "sha256": sha256(args.checkpoint)},
        },
    }
    (output / "source_policy.json").write_text(json.dumps(source_policy, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "raw": report["raw_catalog"], "formula_sources": dict(selected_domains), "gates": report["gates"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
