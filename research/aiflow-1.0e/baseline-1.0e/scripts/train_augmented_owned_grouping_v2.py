#!/usr/bin/env python3
"""Refit the runtime grouping model on all owned grouping truths.

Long, multi-line formula challenges are composed only from accepted owned ink.
Writer-disjoint folds keep every held writer out of both real and composed fit
rows.  The resulting model replaces only the grouping model inside a copied
shadow partition-ranker artifact; HWR and context checkpoints are untouched.
"""

from __future__ import annotations

from training_data_guard_v1 import (
    assert_training_entrypoint_arguments_clean,
    assert_training_path_clean,
    zero_crohme_training_manifest,
)
if __name__ == "__main__":
    assert_training_entrypoint_arguments_clean()

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

from stroke_grouping_v1 import (
    DEFAULT_LATTICE_CONFIG,
    FEATURE_NAMES,
    build_lattice,
    candidate_features,
    select_partition,
)
from train_project_owned_grouping_v1 import (
    SEED,
    Sample,
    _metrics,
    _oracle,
    _samples,
)


SCHEMA = "aiflow-augmented-owned-grouping/v2"
MODEL_VERSION = "aiflow-stroke-grouping-1.0-r6-probability-blended-shadow"
BIAS_GRID = (-1.0, -0.5, 0.0, 0.5, 1.0)
ROUTE_STROKE_GRID = (8, 10, 12, 14, 16, 20)
BASELINE_WEIGHT_GRID = (0.25, 0.50, 0.75, 0.90)
LOCAL_FEATURES = tuple(
    index for index, name in enumerate(FEATURE_NAMES)
    if name not in {"candidate_cx_formula", "candidate_cy_formula", "candidate_width_formula"}
)
FEATURE_CONFIGS = {
    "all_features": tuple(range(len(FEATURE_NAMES))),
    "formula_position_invariant": LOCAL_FEATURES,
}
COMPOSED_MASS_GRID = (0.10, 0.25, 0.50)


@dataclass(frozen=True)
class Glyph:
    writer: str
    strokes: tuple[tuple[tuple[float, float], ...], ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _point_xy(point: object) -> tuple[float, float]:
    if isinstance(point, dict):
        return float(point["x"]), float(point["y"])
    return float(point[0]), float(point[1])


def _glyphs(samples: Iterable[Sample]) -> dict[str, list[Glyph]]:
    pools: dict[str, list[Glyph]] = {}
    for sample in samples:
        ordered = sorted(sample.strokes, key=lambda row: int(row["order"]))
        for group in sample.truth:
            paths = tuple(
                tuple(_point_xy(point) for point in ordered[index]["points"])
                for index in sorted(group)
            )
            pools.setdefault(sample.writer, []).append(Glyph(sample.writer, paths))
    if any(len(pool) < 8 for pool in pools.values()):
        raise ValueError("each writer needs at least eight owned glyph groups")
    return pools


def _normalized(glyph: Glyph) -> tuple[list[np.ndarray], float, float]:
    arrays = [np.asarray(path, dtype=np.float64) for path in glyph.strokes]
    joined = np.concatenate(arrays, axis=0)
    low = joined.min(axis=0); high = joined.max(axis=0)
    extent = np.maximum(high - low, 1e-6)
    scale = max(float(extent.max()), 1e-6)
    return [(array - low) / scale for array in arrays], float(extent[0] / scale), float(extent[1] / scale)


def _placements(rng: np.random.Generator, count: int, mode: str) -> list[tuple[float, float, float]]:
    output: list[tuple[float, float, float]] = []
    if mode == "stacked":
        x = 0.0
        for index in range(count):
            row = index % 2
            output.append((x, -0.72 if row == 0 else 0.72, rng.uniform(0.62, 0.88)))
            if row == 1:
                x += rng.uniform(0.9, 1.45)
        return output
    x = 0.0; row = 0; wrap = int(rng.integers(6, 11))
    for index in range(count):
        if mode == "multiline" and index and index % wrap == 0:
            row += 1; x = rng.uniform(0.0, 0.4)
        script = mode == "scripted" and rng.random() < 0.28
        size = rng.uniform(0.72, 1.18) if not script else rng.uniform(0.42, 0.62)
        y = row * 1.7 + rng.normal(0.0, 0.08)
        if script:
            y += rng.choice((-0.72, 0.72))
        output.append((x, y, size))
        x += rng.uniform(0.75, 1.35)
    return output


def _compose(writer: str, pool: list[Glyph], index: int) -> Sample:
    seed = int.from_bytes(
        hashlib.sha256(f"{SEED}:{writer}:{index}".encode()).digest()[:8], "big"
    )
    rng = np.random.default_rng(seed)
    count = int(rng.integers(10, 23))
    mode = ("horizontal", "multiline", "scripted", "stacked")[index % 4]
    chosen = [pool[int(value)] for value in rng.integers(0, len(pool), size=count)]
    placements = _placements(rng, count, mode)
    strokes: list[dict] = []; truth: list[frozenset[int]] = []
    for glyph, (x, y, size) in zip(chosen, placements, strict=True):
        paths, width, height = _normalized(glyph)
        angle = float(rng.uniform(-0.10, 0.10))
        cosine = math.cos(angle); sine = math.sin(angle)
        origin = len(strokes)
        for path in paths:
            centered = path - np.asarray([width / 2.0, height / 2.0])
            rotated = np.column_stack([
                centered[:, 0] * cosine - centered[:, 1] * sine,
                centered[:, 0] * sine + centered[:, 1] * cosine,
            ])
            transformed = rotated * (72.0 * size) + np.asarray([72.0 * x, 72.0 * y])
            strokes.append({
                "order": len(strokes),
                "points": [
                    {"x": float(px), "y": float(py), "t_ms": float(offset) * 8.0}
                    for offset, (px, py) in enumerate(transformed)
                ],
            })
        truth.append(frozenset(range(origin, len(strokes))))
    candidates = build_lattice(strokes, **DEFAULT_LATTICE_CONFIG)
    return Sample(
        f"owned-composed:{writer}:{index:04d}", writer, strokes, tuple(truth),
        candidates, candidate_features(candidates, strokes),
    )


def _composed_by_writer(
    samples: list[Sample], per_writer: int,
) -> dict[str, list[Sample]]:
    pools = _glyphs(samples)
    output = {
        writer: [_compose(writer, pool, index) for index in range(per_writer)]
        for writer, pool in sorted(pools.items())
    }
    synthetic = [sample for rows in output.values() for sample in rows]
    oracle = _oracle(synthetic)
    if oracle["partition_recoverable"] != 1.0:
        raise ValueError(f"composed challenge is not lattice-recoverable: {oracle['missing'][:3]}")
    return output


def _fit(
    samples: list[Sample], kept: tuple[int, ...], *, composed_mass: float = 0.5,
) -> Pipeline:
    feature_rows = []; label_rows = []; domain_rows = []
    for sample in samples:
        truth = set(sample.truth)
        labels = np.asarray([
            frozenset(int(value) for value in row["source_indices"]) in truth
            for row in sample.candidates
        ], dtype=np.int64)
        domain = "composed" if sample.sample_id.startswith("owned-composed:") else "real"
        feature_rows.append(sample.features); label_rows.append(labels)
        domain_rows.append(np.full(len(labels), domain, dtype=object))
    features = np.concatenate(feature_rows); labels = np.concatenate(label_rows)
    domain_values = np.concatenate(domain_rows)
    domains = sorted(set(domain_values.tolist()))
    domain_mass = {domains[0]: 1.0} if len(domains) == 1 else {
        "real": 1.0 - composed_mass, "composed": composed_mass,
    }
    weights = np.zeros(len(labels), dtype=np.float64)
    for domain in domains:
        for label in (0, 1):
            mask = (domain_values == domain) & (labels == label)
            count = int(mask.sum())
            if count == 0:
                raise ValueError(f"missing grouping class {label} in {domain} domain")
            weights[mask] = domain_mass[domain] / (2.0 * count)
    weights *= len(labels)
    selector = ColumnTransformer(
        [("kept", "passthrough", list(kept))], remainder="drop",
        verbose_feature_names_out=False,
    )
    model = Pipeline([
        ("feature_selection", selector),
        ("classifier", HistGradientBoostingClassifier(
            learning_rate=0.07, max_iter=160, max_leaf_nodes=15,
            l2_regularization=1.0, min_samples_leaf=20, random_state=SEED,
        )),
    ])
    model.fit(features, labels, classifier__sample_weight=weights)
    return model


def _scores(model: object, samples: Iterable[Sample]) -> dict[str, np.ndarray]:
    output = {}
    for sample in samples:
        probability = model.predict_proba(sample.features)[:, 1]
        output[sample.sample_id] = np.log(
            np.clip(probability, 1e-6, 1 - 1e-6)
            / np.clip(1 - probability, 1e-6, 1)
        )
    return output


def _predict(
    samples: Iterable[Sample], scores: dict[str, np.ndarray], bias: float,
) -> dict[str, tuple[frozenset[int], ...]]:
    return {
        sample.sample_id: tuple(select_partition(
            sample.candidates, scores[sample.sample_id], len(sample.strokes),
            group_bias=bias,
        ))
        for sample in samples
    }


def _blend_scores(
    baseline: dict[str, np.ndarray], augmented: dict[str, np.ndarray],
    baseline_weight: float,
) -> dict[str, np.ndarray]:
    output = {}
    for sample_id in baseline:
        base_probability = 1.0 / (1.0 + np.exp(-baseline[sample_id]))
        augmented_probability = 1.0 / (1.0 + np.exp(-augmented[sample_id]))
        probability = (
            baseline_weight * base_probability
            + (1.0 - baseline_weight) * augmented_probability
        )
        output[sample_id] = np.log(
            np.clip(probability, 1e-6, 1 - 1e-6)
            / np.clip(1 - probability, 1e-6, 1)
        )
    return output


def _fold_audit(
    real: list[Sample], composed: dict[str, list[Sample]],
) -> tuple[dict, str, float, float, int, float]:
    writers = sorted(composed)
    all_real_predictions: dict[str, dict[str, tuple[frozenset[int], ...]]] = {
        "baseline": {}
    }
    all_composed_predictions: dict[str, dict[str, tuple[frozenset[int], ...]]] = {
        "baseline": {}
    }
    for config in FEATURE_CONFIGS:
        for mass in COMPOSED_MASS_GRID:
            for weight in BASELINE_WEIGHT_GRID:
                for bias in BIAS_GRID:
                    key = (
                        f"{config}|mass={mass:.2f}|base_weight={weight:.2f}"
                        f"|bias={bias:+.1f}"
                    )
                    all_real_predictions[key] = {}; all_composed_predictions[key] = {}
    folds = []
    for outer in writers:
        train_real = [sample for sample in real if sample.writer != outer]
        test_real = [sample for sample in real if sample.writer == outer]
        train_composed = [
            sample for writer in writers if writer != outer for sample in composed[writer]
        ]
        test_composed = composed[outer]
        baseline = _fit(train_real, FEATURE_CONFIGS["all_features"])
        baseline_real_scores = _scores(baseline, test_real)
        baseline_composed_scores = _scores(baseline, test_composed)
        all_real_predictions["baseline"].update(_predict(test_real, baseline_real_scores, 0.0))
        all_composed_predictions["baseline"].update(_predict(test_composed, baseline_composed_scores, 0.0))
        for config, kept in FEATURE_CONFIGS.items():
            for mass in COMPOSED_MASS_GRID:
                model = _fit(
                    train_real + train_composed, kept, composed_mass=mass,
                )
                real_scores = _scores(model, test_real)
                composed_scores = _scores(model, test_composed)
                for weight in BASELINE_WEIGHT_GRID:
                    blended_real_scores = _blend_scores(
                        baseline_real_scores, real_scores, weight,
                    )
                    blended_composed_scores = _blend_scores(
                        baseline_composed_scores, composed_scores, weight,
                    )
                    for bias in BIAS_GRID:
                        key = (
                            f"{config}|mass={mass:.2f}|base_weight={weight:.2f}"
                            f"|bias={bias:+.1f}"
                        )
                        all_real_predictions[key].update(
                            _predict(test_real, blended_real_scores, bias)
                        )
                        all_composed_predictions[key].update(
                            _predict(test_composed, blended_composed_scores, bias)
                        )
        folds.append({
            "held_out_writer": outer,
            "fit_real_formulas": len(train_real),
            "fit_composed_formulas": len(train_composed),
            "test_real_formulas": len(test_real),
            "test_composed_formulas": len(test_composed),
        })
    composed_all = [sample for rows in composed.values() for sample in rows]
    baseline = {
        "configuration": "baseline",
        "real": _metrics(real, all_real_predictions["baseline"]),
        "long_2d_composed": _metrics(composed_all, all_composed_predictions["baseline"]),
    }
    trials = [baseline]
    for key in all_real_predictions:
        if key == "baseline":
            continue
        for threshold in ROUTE_STROKE_GRID:
            routed_real = {
                sample.sample_id: (
                    all_real_predictions[key][sample.sample_id]
                    if len(sample.strokes) >= threshold
                    else all_real_predictions["baseline"][sample.sample_id]
                )
                for sample in real
            }
            routed_composed = {
                sample.sample_id: (
                    all_composed_predictions[key][sample.sample_id]
                    if len(sample.strokes) >= threshold
                    else all_composed_predictions["baseline"][sample.sample_id]
                )
                for sample in composed_all
            }
            trials.append({
                "configuration": f"{key}|route_strokes>={threshold}",
                "real": _metrics(real, routed_real),
                "long_2d_composed": _metrics(composed_all, routed_composed),
            })
    eligible = [
        row for row in trials if row["configuration"] != "baseline"
        and row["real"]["partition_exact"] >= baseline["real"]["partition_exact"]
        and row["real"]["pair_f1"] >= baseline["real"]["pair_f1"] - 0.002
    ]
    if not eligible:
        raise ValueError("no augmented model preserves real writer-disjoint metrics")
    winner = max(eligible, key=lambda row: (
        row["long_2d_composed"]["partition_exact"],
        row["real"]["partition_exact"], row["real"]["pair_f1"],
        -row["long_2d_composed"]["overmerge_rate"],
        -row["long_2d_composed"]["oversplit_rate"],
    ))
    config, remainder = winner["configuration"].split("|mass=", 1)
    mass_text, remainder = remainder.split("|base_weight=", 1)
    weight_text, remainder = remainder.split("|bias=", 1)
    bias_text, threshold_text = remainder.split("|route_strokes>=", 1)
    return {
        "protocol": (
            "outer writer absent from real and composed fit rows; composed formulas "
            "contain only accepted owned glyph strokes from their source writer"
        ),
        "folds": folds, "baseline": baseline, "trials": trials, "winner": winner,
    }, config, float(bias_text), float(mass_text), int(threshold_text), float(weight_text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--base-ranker", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--composed-per-writer", type=int, default=32)
    args = parser.parse_args()
    dataset_root = assert_training_path_clean(args.dataset_root, "owned grouping dataset")
    base_ranker = assert_training_path_clean(args.base_ranker, "shadow partition ranker")
    output = args.output.resolve()
    if output.exists() or output.drive.upper() != "D:":
        parser.error("output must be a new directory on D:")
    if args.composed_per_writer < 8:
        parser.error("--composed-per-writer must be at least 8")
    real, dataset = _samples(dataset_root)
    composed = _composed_by_writer(real, args.composed_per_writer)
    (
        audit, selected_config, selected_bias, selected_mass, selected_threshold,
        selected_baseline_weight,
    ) = _fold_audit(real, composed)
    all_composed = [sample for rows in composed.values() for sample in rows]
    baseline_model = _fit(real, FEATURE_CONFIGS["all_features"])
    long_formula_model = _fit(
        real + all_composed, FEATURE_CONFIGS[selected_config],
        composed_mass=selected_mass,
    )
    payload = joblib.load(base_ranker)
    required = {"schema", "model", "grouping_model", "feature_names", "lattice_config", "top_n"}
    if not required.issubset(payload):
        raise ValueError("base partition ranker contract mismatch")
    payload = dict(payload)
    payload.update({
        "model_version": MODEL_VERSION,
        "grouping_model": baseline_model,
        "group_bias": 0.0,
        "grouping_long_formula_model": long_formula_model,
        "grouping_long_formula_route": {
            "minimum_strokes": selected_threshold,
            "group_bias": selected_bias,
            "baseline_probability_weight": selected_baseline_weight,
            "target_label_or_glyph_count_input": False,
        },
        "grouping_training_scope": (
            f"all {len(real)} accepted owned formulas plus {len(all_composed)} "
            "owned-stroke composed long/2D formulas"
        ),
        "grouping_refit_contract": {
            "schema": SCHEMA,
            "feature_configuration": selected_config,
            "composed_training_mass": selected_mass,
            "minimum_strokes": selected_threshold,
            "baseline_probability_weight": selected_baseline_weight,
            "kept_features": [FEATURE_NAMES[index] for index in FEATURE_CONFIGS[selected_config]],
            "real_formulas": len(real), "composed_formulas": len(all_composed),
            "writers": len(composed), "writer_identity_feature": False,
            "target_label_or_glyph_count_input": False,
            "all_strokes_exactly_once": True,
        },
    })
    output.mkdir(parents=True)
    artifact_path = output / "partition_context_ranker.joblib"
    joblib.dump(payload, artifact_path, compress=3)
    summary = {
        "schema": SCHEMA, "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "shadow", "model_version": MODEL_VERSION,
        "dataset": dataset,
        "composed": {
            "formulas": len(all_composed), "per_writer": args.composed_per_writer,
            "writer_formulas": dict(sorted(Counter(sample.writer for sample in all_composed).items())),
            "lattice_oracle": _oracle(all_composed),
            "layouts": ["horizontal", "multiline", "scripted", "stacked"],
        },
        "writer_disjoint_development_audit": audit,
        "selection": {
            "feature_configuration": selected_config,
            "composed_training_mass": selected_mass,
            "minimum_strokes": selected_threshold,
            "baseline_probability_weight": selected_baseline_weight,
            "group_bias": selected_bias,
        },
        "artifact": {
            "file": artifact_path.name, "sha256": _sha256(artifact_path),
            "base_ranker": str(base_ranker), "base_ranker_sha256": _sha256(base_ranker),
        },
        "contracts": {
            "all_owned_grouping_truths_used_in_final_refit": len(real),
            "writer_identity_feature": False, "target_label_feature": False,
            "target_glyph_count_input": False, "all_strokes_exactly_once": True,
            "raw_strokes_mutated_at_runtime": False,
            "hwr_or_context_checkpoint_modified": False,
            "product_default_enabled": False,
            "fit_weighting": (
                "equal real/composed domain mass; balanced positive/negative candidate "
                "mass within each domain"
            ),
            "training_data_guard": zero_crohme_training_manifest(
                admitted_sources={
                    "project_owned_grouping_formulas": len(real),
                    "project_owned_stroke_compositions": len(all_composed),
                }
            ),
        },
        "limits": [
            "writer-disjoint results are development evidence, not untouched product acceptance",
            "composed formulas broaden length and 2D geometry but do not create new handwriting styles",
            "the copied context partition ranker retains its prior posthoc-shadow provenance",
        ],
    }
    summary_path = output / "grouping_refit_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n",
    )
    print(json.dumps({
        "output": str(output), "selection": summary["selection"],
        "baseline": audit["baseline"], "winner": audit["winner"],
        "artifact_sha256": summary["artifact"]["sha256"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
