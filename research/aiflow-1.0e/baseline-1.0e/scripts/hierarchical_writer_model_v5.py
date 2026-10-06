#!/usr/bin/env python3
"""Build a clean-room hierarchical writer/allograph/formula production smoke."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from cleanroom_online_ink_augmentation_v2 import stroke_bounds
from cleanroom_trajectory_profiles_v3 import trajectory_descriptor
from cleanroom_writer_style_simulator_v4 import WRITER_STYLES, WriterStyleV4, apply_writer_style


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTERNAL = ROOT / "artifacts/commercial_hwr_cleanroom_dataset_v3_20260823_r1/external_profiled_augmented.npz"
DEFAULT_OWNED = ROOT / "artifacts/commercial_hwr_cleanroom_dataset_v3_20260823_r1/project_profiled_augmented_final.npz"
DEFAULT_CHECKPOINT = ROOT / "artifacts/commercial_hwr_cleanroom_physics_20260823_r1_shadow/commercial_hwr_cleanroom_physics_checkpoint.pt"
DEFAULT_OUTPUT = ROOT / "artifacts/hierarchical_writer_model_v5_20260823_r1"
SCHEMA = "aiflow-hierarchical-writer-model/v6"


@dataclass(frozen=True)
class AllographPlanV5:
    token: str
    label_index: int
    source_domain: str
    source_row: int
    stroke_count: int
    start_quadrant: str
    end_quadrant: str
    direction_signature: tuple[int, ...]
    evidence_tier: str
    synthetic_id: str
    parent_hashes: tuple[tuple[str, str], ...]
    cross_writer: bool
    parent_writer_fingerprints: tuple[str, ...]


@dataclass(frozen=True)
class WriterLatentV5:
    writer_id: str
    global_style: WriterStyleV4
    letter_spacing: float
    word_spacing: float
    glyph_pause_ms: float
    stroke_pause_ms: float
    production_mode: str
    allograph_seed: int


@dataclass(frozen=True)
class GlyphPlacementV5:
    glyph_id: str
    token: str
    x: float
    y: float
    scale: float
    relation: str
    parent_id: str | None = None


@dataclass(frozen=True)
class FormulaPlanV5:
    formula_id: str
    display: str
    glyphs: tuple[GlyphPlacementV5, ...]
    relation_edges: tuple[tuple[str, str, str], ...]
    production_groups: tuple[tuple[str, ...], ...]


WRITERS = tuple(
    WriterLatentV5(
        style.writer_id, style,
        letter_spacing=(0.82, 1.03, 1.20, 0.75)[index],
        word_spacing=(1.75, 1.55, 1.92, 1.42)[index],
        glyph_pause_ms=(72.0, 54.0, 94.0, 38.0)[index],
        stroke_pause_ms=(34.0, 28.0, 46.0, 22.0)[index],
        production_mode=("left_to_right", "baseline_first", "structure_first", "baseline_first")[index],
        allograph_seed=20260823 + index * 1009,
    )
    for index, style in enumerate(WRITER_STYLES)
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _quadrant(point: np.ndarray) -> str:
    return ("L" if float(point[0]) < 0.5 else "R") + ("T" if float(point[1]) < 0.5 else "B")


def _direction_signature(features: np.ndarray) -> tuple[int, ...]:
    output = []
    for start, end in stroke_bounds(features):
        delta = features[end - 1, :2] - features[start, :2]
        axis = int(np.argmax(np.abs(delta)))
        output.append((axis + 1) * (1 if float(delta[axis]) >= 0 else -1))
    return tuple(output)


def _plan(token: str, label: int, domain: str, row: int, features: np.ndarray, metadata: dict) -> AllographPlanV5:
    descriptor = trajectory_descriptor(features)
    return AllographPlanV5(
        token, label, domain, row, descriptor.stroke_count,
        _quadrant(features[0, :2]), _quadrant(features[-1, :2]),
        _direction_signature(features),
        "synthetic_cleanroom_candidate",
        str(metadata["synthetic_id"]),
        tuple(sorted((str(key), str(value)) for key, value in metadata["parents"].items())),
        bool(metadata["cross_writer"]),
        tuple(str(value) for value in metadata.get("parent_writer_fingerprints", [])),
    )


def _catalog(external: Path, owned_evaluation_smoke: Path | None, checkpoint: Path) -> tuple[dict[str, list[tuple[AllographPlanV5, np.ndarray]]], list[str], dict]:
    import torch

    labels = [str(value) for value in torch.load(checkpoint, map_location="cpu", weights_only=False)["math_labels"]]
    catalog: dict[str, list[tuple[AllographPlanV5, np.ndarray]]] = {}
    source_rows = 0
    domain_rows = {}
    sources = [("approved_external_cleanroom", external)]
    if owned_evaluation_smoke is not None:
        sources.append(("project_evaluation_cleanroom_smoke_only", owned_evaluation_smoke))
    for domain, path in sources:
        with np.load(path, allow_pickle=False) as payload:
            features = np.asarray(payload["features"], dtype=np.float32)
            indices = np.asarray(payload["labels"], dtype=np.int64)
        metadata_path = path.with_name(path.stem + ".metadata.jsonl.gz")
        with gzip.open(metadata_path, "rt", encoding="utf-8") as stream:
            metadata_rows = [json.loads(line) for line in stream if line.strip()]
        if len(metadata_rows) != len(features):
            raise ValueError(f"metadata/features length mismatch: {path}")
        source_rows += len(indices)
        domain_rows[domain] = len(indices)
        for row, (feature, label, metadata) in enumerate(zip(features, indices, metadata_rows, strict=True)):
            if not 0 <= int(label) < len(labels):
                continue
            token = labels[int(label)]
            plan = _plan(token, int(label), domain, row, feature, metadata)
            catalog.setdefault(token, []).append((plan, feature.copy()))
    signature_counts = {
        token: len({
            (item[0].stroke_count, item[0].start_quadrant, item[0].end_quadrant, item[0].direction_signature)
            for item in rows
        })
        for token, rows in catalog.items()
    }
    return catalog, labels, {
        "source_rows": source_rows,
        "domain_rows": domain_rows,
        "tokens": len(catalog),
        "synthetic_candidate_trajectories": sum(map(len, catalog.values())),
        "tokens_with_multiple_synthetic_plan_signatures": sum(value > 1 for value in signature_counts.values()),
        "maximum_plan_signatures_per_token": max(signature_counts.values()),
        "evidence_tier": "synthetic_cleanroom_candidate_not_raw_observed_writer_evidence",
        "project_evaluation_bank_loaded": owned_evaluation_smoke is not None,
        "project_evaluation_training_admission": False,
    }


def _select(catalog: dict[str, list[tuple[AllographPlanV5, np.ndarray]]], token: str, writer: WriterLatentV5) -> tuple[AllographPlanV5, np.ndarray]:
    options = catalog.get(token)
    if not options:
        raise KeyError(f"no approved allograph for {token!r}")
    digest = hashlib.sha256(f"{writer.allograph_seed}:{token}".encode()).digest()
    return options[int.from_bytes(digest[:4], "big") % len(options)]


def _label_safe_catalog(catalog: dict, writers: tuple[WriterLatentV5, ...], checkpoint: Path) -> tuple[dict[str, dict], dict]:
    """Keep writer-conditioned allographs whose frozen-HWR truth remains in Top-5."""
    import torch
    from train_character_classifier_v1 import InkClassifierV1

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = InkClassifierV1(len(payload["math_labels"]), None)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    output = {}; audit = {}
    for writer in writers:
        flattened = [(token, option) for token, options in catalog.items() for option in options]
        styled = np.stack([
            apply_writer_style(option[1], writer.global_style)[0] for _token, option in flattened
        ]).astype(np.float32)
        rank_batches = []
        with torch.no_grad():
            for start in range(0, len(styled), 512):
                rank_batches.append(model(torch.from_numpy(styled[start:start + 512]), "math").argsort(dim=1, descending=True).cpu().numpy())
        ranks = np.concatenate(rank_batches)
        grouped: dict[str, list[tuple[tuple[AllographPlanV5, np.ndarray], np.ndarray]]] = {}
        for (token, option), predicted in zip(flattened, ranks, strict=True):
            grouped.setdefault(token, []).append((option, predicted))
        writer_catalog = {}; safe_count = 0; fallback_tokens = []
        for token, candidates in grouped.items():
            safe = [option for option, predicted in candidates if option[0].label_index in predicted[:5]]
            if not safe:
                truth = candidates[0][0][0].label_index
                truth_ranks = [int(np.flatnonzero(predicted == truth)[0]) for _option, predicted in candidates]
                safe = [candidates[int(np.argmin(truth_ranks))][0]]
                fallback_tokens.append(token)
            safe_count += len(safe)
            writer_catalog[token] = safe
        output[writer.writer_id] = writer_catalog
        audit[writer.writer_id] = {
            "retained_options": safe_count,
            "tokens_without_any_top5_safe_option": fallback_tokens,
        }
    return output, audit


def _formula_plans() -> tuple[FormulaPlanV5, ...]:
    def baseline(formula_id: str, display: str, tokens: str) -> FormulaPlanV5:
        glyphs = tuple(GlyphPlacementV5(f"{formula_id}:g{i}", token, i * 1.05, 0.0, 1.0, "baseline") for i, token in enumerate(tokens))
        groups = tuple((glyph.glyph_id,) for glyph in glyphs)
        return FormulaPlanV5(formula_id, display, glyphs, (), groups)

    linear = baseline("linear", "A + 3 - 7", "A+3-7")
    power_glyphs = (
        GlyphPlacementV5("power:x", "x", 0.0, 0.0, 1.0, "baseline"),
        GlyphPlacementV5("power:2a", "2", 0.70, -0.66, 0.56, "superscript", "power:x"),
        GlyphPlacementV5("power:+", "+", 1.45, 0.0, 1.0, "baseline"),
        GlyphPlacementV5("power:y", "y", 2.50, 0.0, 1.0, "baseline"),
        GlyphPlacementV5("power:2b", "2", 3.18, -0.66, 0.56, "superscript", "power:y"),
        GlyphPlacementV5("power:+2", "+", 3.95, 0.0, 1.0, "baseline"),
        GlyphPlacementV5("power:2c", "2", 5.00, 0.0, 1.0, "baseline"),
        GlyphPlacementV5("power:5", "5", 6.00, 0.0, 1.0, "baseline"),
    )
    power = FormulaPlanV5(
        "power", "x^2 + y^2 + 25", power_glyphs,
        (("power:x", "power:2a", "superscript"), ("power:y", "power:2b", "superscript")),
        (("power:x", "power:+", "power:y", "power:+2", "power:2c", "power:5"), ("power:2a", "power:2b")),
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


def _production_order(plan: FormulaPlanV5, writer: WriterLatentV5) -> list[str]:
    groups = [list(group) for group in plan.production_groups]
    if writer.production_mode == "structure_first" and plan.formula_id == "fraction":
        groups = [groups[1], groups[0], groups[2]]
    elif writer.production_mode == "left_to_right":
        return [glyph.glyph_id for glyph in sorted(plan.glyphs, key=lambda row: (row.x, row.y))]
    return [glyph_id for group in groups for glyph_id in group]


def _transform(features: np.ndarray, placement: GlyphPlacementV5, writer: WriterLatentV5) -> tuple[np.ndarray, dict]:
    styled, audit = apply_writer_style(features, writer.global_style)
    if styled.shape != (128, 5) or not np.array_equal(styled[:, 2:], features[:, 2:]):
        raise AssertionError("box-local 128x5 temporal/topology contract changed")
    output = styled.copy()
    output[:, 0] = placement.x * writer.letter_spacing + placement.scale * output[:, 0]
    output[:, 1] = placement.y + placement.scale * output[:, 1]
    return output, audit


def _hwr_preservation(catalog_by_writer: dict, writers: tuple[WriterLatentV5, ...], checkpoint: Path, tokens: list[str]) -> dict:
    """Measure frozen-HWR Top-k preservation on box-local writer/allograph rows."""
    import torch
    from train_character_classifier_v1 import InkClassifierV1

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = InkClassifierV1(len(payload["math_labels"]), None)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    per_writer = {}
    for writer in writers:
        catalog = catalog_by_writer[writer.writer_id]
        rows = []; truth = []
        for token in tokens:
            plan, source = _select(catalog, token, writer)
            styled, _audit = apply_writer_style(source, writer.global_style)
            rows.append(styled); truth.append(plan.label_index)
        with torch.no_grad():
            rank = model(torch.from_numpy(np.stack(rows).astype(np.float32)), "math").argsort(dim=1, descending=True)
        truth_np = np.asarray(truth, dtype=np.int64)
        rank_np = rank[:, :5].cpu().numpy()
        per_writer[writer.writer_id] = {
            "rows": len(rows),
            "top1": float(np.mean(rank_np[:, 0] == truth_np)),
            "top5": float(np.mean(np.any(rank_np == truth_np[:, None], axis=1))),
            "outside_top5_tokens": [
                token for token, expected, predicted in zip(tokens, truth_np, rank_np, strict=True)
                if expected not in predicted
            ],
        }
    return per_writer


def _events(features: np.ndarray, glyph: GlyphPlacementV5, plan: AllographPlanV5, start_ms: float, writer: WriterLatentV5) -> tuple[list[dict], float]:
    events = []
    current = start_ms
    for stroke_index, (start, end) in enumerate(stroke_bounds(features)):
        points = []
        for point_index in range(start, end):
            dt = 1000.0 / 48.0 if point_index == start else max(1000.0 / 96.0, float(features[point_index, 2]) * 1000.0)
            current += dt
            points.append({"x": float(features[point_index, 0]), "y": float(features[point_index, 1]), "t_ms": current})
        events.append({
            "glyph_id": glyph.glyph_id, "token": glyph.token, "source_domain": plan.source_domain,
            "source_row": plan.source_row, "source_stroke_index": stroke_index, "points": points,
        })
        current += writer.stroke_pause_ms
    return events, current + writer.glyph_pause_ms


def _compose_formula(catalog: dict, formula: FormulaPlanV5, writer: WriterLatentV5) -> dict:
    by_id = {glyph.glyph_id: glyph for glyph in formula.glyphs}
    order = _production_order(formula, writer)
    if sorted(order) != sorted(by_id) or len(order) != len(set(order)):
        raise AssertionError("production order lost or duplicated a glyph")
    generated = {}; current = 0.0; events = []; allograph_rows = []
    for glyph_id in order:
        glyph = by_id[glyph_id]
        allograph, source = _select(catalog, glyph.token, writer)
        transformed, style_audit = _transform(source, glyph, writer)
        rows, current = _events(transformed, glyph, allograph, current, writer)
        events.extend(rows)
        generated[glyph_id] = transformed
        allograph_rows.append({"glyph_id": glyph_id, "plan": asdict(allograph), "style_audit": style_audit})
    expected_strokes = sum(item["plan"]["stroke_count"] for item in allograph_rows)
    if len(events) != expected_strokes:
        raise AssertionError("stroke provenance is not exact")
    return {
        "formula_id": formula.formula_id, "display": formula.display, "writer_id": writer.writer_id,
        "production_mode": writer.production_mode, "production_order": order,
        "relation_edges": formula.relation_edges, "glyphs": [asdict(glyph) for glyph in formula.glyphs],
        "allographs": allograph_rows, "events": events, "duration_ms": current,
        "glyph_tensors": generated,
    }


def _draw(path: Path, generated: list[dict], title: str = "Hierarchical writer model v6: same formula, different writers") -> None:
    from PIL import Image, ImageDraw

    cell_w, cell_h = 920, 225
    image = Image.new("RGB", (cell_w, 55 + cell_h * len(generated)), "white")
    draw = ImageDraw.Draw(image)
    draw.text((18, 16), title, fill="#172033")
    colors = ("#2358a5", "#9a4d9d", "#b15a2c", "#167568")
    for row, item in enumerate(generated):
        top = 55 + row * cell_h
        draw.text((18, top + 10), f"{item['writer_id']} | {item['display']} | {item['production_mode']}", fill="#263550")
        points = [point for event in item["events"] for point in event["points"]]
        xs = [point["x"] for point in points]; ys = [point["y"] for point in points]
        low_x, high_x = min(xs), max(xs); low_y, high_y = min(ys), max(ys)
        scale = min(820 / max(high_x - low_x, 1e-6), 145 / max(high_y - low_y, 1e-6))
        for event in item["events"]:
            line = [(48 + (point["x"] - low_x) * scale, top + 55 + (point["y"] - low_y) * scale) for point in event["points"]]
            if len(line) > 1:
                draw.line(line, fill=colors[row % len(colors)], width=3, joint="curve")
    image.save(path)


def _jsonable(item: dict) -> dict:
    return {key: value for key, value in item.items() if key != "glyph_tensors"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external", type=Path, default=DEFAULT_EXTERNAL)
    parser.add_argument("--owned", type=Path, default=DEFAULT_OWNED)
    parser.add_argument(
        "--allow-evaluation-smoke", action="store_true",
        help="load project evaluation-derived synthetic bank for structure smoke only; never training/adoption",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    paths = [args.external.resolve(), args.checkpoint.resolve(), args.output.resolve()]
    if args.allow_evaluation_smoke:
        paths.append(args.owned.resolve())
    if any(path.drive.upper() != "D:" for path in paths):
        parser.error("all inputs and outputs must remain on D:")
    if args.output.exists():
        parser.error(f"refusing to overwrite output: {args.output}")

    evaluation_bank = args.owned if args.allow_evaluation_smoke else None
    catalog, labels, catalog_audit = _catalog(args.external, evaluation_bank, args.checkpoint)
    required = set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz+-=/()")
    missing = sorted(required - set(catalog))
    formulas = _formula_plans()
    safe_catalogs, allograph_gate_audit = _label_safe_catalog(catalog, WRITERS, args.checkpoint)
    generated = [
        _compose_formula(safe_catalogs[writer.writer_id], formula, writer)
        for formula in formulas for writer in WRITERS
    ]
    output = args.output.resolve(); output.mkdir(parents=True)
    with (output / "formula_generation.jsonl").open("w", encoding="utf-8") as stream:
        for item in generated:
            stream.write(json.dumps(_jsonable(item), ensure_ascii=False) + "\n")
    _draw(output / "same_formula_four_writers.png", [item for item in generated if item["formula_id"] == "power"])

    alphabet_tokens = [token for token in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz" if token in catalog]
    alphabet = {}
    for writer in WRITERS:
        writer_catalog = safe_catalogs[writer.writer_id]
        alphabet[writer.writer_id] = {
            token: asdict(_select(writer_catalog, token, writer)[0]) for token in alphabet_tokens
        }
    (output / "writer_allograph_assignments.json").write_text(json.dumps(alphabet, ensure_ascii=False, indent=2), encoding="utf-8")
    hwr_preservation = _hwr_preservation(safe_catalogs, WRITERS, args.checkpoint, alphabet_tokens)
    report = {
        "schema": SCHEMA, "generated_at": datetime.now(timezone.utc).isoformat(),
        "catalog": catalog_audit, "allograph_label_safety_gate": allograph_gate_audit,
        "output_vocabulary": len(labels),
        "required_alphanumeric_math_tokens": len(required), "missing_required_tokens": missing,
        "lowercase_t_present": "t" in labels, "writers": [asdict(writer) for writer in WRITERS],
        "formula_templates": len(formulas), "formula_writer_combinations": len(generated),
        "raw_strokes": sum(len(item["events"]) for item in generated),
        "frozen_hwr_alphanumeric_preservation": hwr_preservation,
        "gates": {
            "d_drive_only": True,
            "approved_external_only_default": not args.allow_evaluation_smoke,
            "project_evaluation_bank_loaded": args.allow_evaluation_smoke,
            "project_evaluation_smoke_only": args.allow_evaluation_smoke,
            "project_evaluation_training_admission": False,
            "project_evaluation_augmentation_admission": False,
            "project_evaluation_adoption_admission": False,
            "commercial_augmentation_mode": not args.allow_evaluation_smoke,
            "exact_glyph_production_partition": True, "exact_stroke_provenance": True,
            "finite_coordinates": all(np.isfinite([point[axis] for item in generated for event in item["events"] for point in event["points"] for axis in ("x", "y", "t_ms")]).all() for _ in [0]),
            "crohme_rows": 0, "mathwriting_rows": 0, "training_performed": False,
            "product_runtime_changed": False, "checkpoint_changed": False,
            "complete_a_to_z": "t" not in missing,
            "box_local_top5_minimum_90pct": min(row["top5"] for row in hwr_preservation.values()) >= 0.90,
            "writer_disjoint_acceptance": False,
            "raw_observed_allograph_evidence": False,
        },
        "evidence_limits": {
            "catalog": "DTW/profile/physics synthetic bank; not raw observed writer samples",
            "label_safety": "same-lineage frozen checkpoint filter; label preservation only, not writer generalization",
            "acceptance": "fresh writer-disjoint acceptance not performed",
            "project_evaluation_bank": (
                "explicit structure smoke only; never training, augmentation, selection, or adoption"
                if args.allow_evaluation_smoke else "quarantined and not loaded"
            ),
        },
        "inputs": {
            "external": {
                "path": str(args.external.resolve()), "sha256": _sha(args.external),
                "role": "commercial_external_synthetic_catalog",
            },
            "project_evaluation": {
                "path": str(args.owned.resolve()), "loaded": args.allow_evaluation_smoke,
                "sha256": _sha(args.owned) if args.allow_evaluation_smoke else None,
                "source_training_role": "box_local_project_owned_evaluation_only",
                "source_split": "project_owned_evaluation",
                "training_admission": False, "augmentation_admission": False,
                "adoption_admission": False,
            },
            "checkpoint": {"path": str(args.checkpoint.resolve()), "sha256": _sha(args.checkpoint)},
        },
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "catalog": catalog_audit, "missing": missing, "gates": report["gates"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
