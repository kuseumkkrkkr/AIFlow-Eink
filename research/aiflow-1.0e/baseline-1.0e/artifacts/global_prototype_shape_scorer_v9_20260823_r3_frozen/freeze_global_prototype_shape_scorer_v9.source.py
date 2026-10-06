"""Freeze a Top-5-preserving clean-room global prototype shape scorer.

Development is restricted to the old synthetic fit/dev writers.  The already
consumed extension r1 writers are architecture diagnostics only.  The sealed
reserve is never loaded by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from build_cleanroom_writer_style_bank_v5 import sha256
from evaluate_writer_candidate_promoter_extension_v9 import _class_sets, _evaluate_detailed, _global_map
from train_cleanroom_writer_adapter_v9 import SUPPORT_SIZES, _episode_indices
from train_writer_candidate_promoter_v9 import SEED


ROOT = Path(__file__).resolve().parents[1]
OLD_CACHE = ROOT / "artifacts/writer_adaptation_v9_frozen_cache_20260823_r1.npz"
CONSUMED_CACHE = ROOT / "artifacts/writer_adaptation_v9_extension32_frozen_cache_20260823_r1.npz"
OLD_BANK = ROOT / "artifacts/cleanroom_writer_style_v5_20260823_smoke32_r4_r2"
EXT_BANK = ROOT / "artifacts/cleanroom_writer_style_v5_20260823_extension32_r4_r1"
R1_RECEIPT = ROOT / "artifacts/writer_candidate_promoter_v9_extension_outer_20260823_r1/outer_open_receipt.json"
DEFAULT_OUTPUT = ROOT / "artifacts/global_prototype_shape_scorer_v9_20260823_r3_frozen"
ALPHAS = (0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0)
MIN_CLASS_SUPPORT = 4


def load_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]).copy() for name in ("labels", "writers", "splits", "embeddings", "logits")}


def normalize(rows: np.ndarray) -> np.ndarray:
    return rows / np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-8)


def prototype_stats(data: dict[str, np.ndarray], fit_writers: list[int]) -> dict[str, np.ndarray]:
    embeddings = normalize(data["embeddings"])
    classes = data["logits"].shape[1]
    sums = np.zeros((classes, embeddings.shape[1]), np.float64)
    counts = np.zeros(classes, np.int64)
    mask = np.isin(data["writers"], fit_writers)
    np.add.at(sums, data["labels"][mask], embeddings[mask])
    np.add.at(counts, data["labels"][mask], 1)
    prototypes = normalize((sums / np.maximum(counts[:, None], 1)).astype(np.float32))
    reliability = np.zeros(classes, np.float32)
    for label in np.flatnonzero(counts):
        rows = mask & (data["labels"] == label)
        reliability[label] = float(np.mean(embeddings[rows] @ prototypes[label]))
    admitted = (counts >= MIN_CLASS_SUPPORT) & np.isfinite(reliability) & (reliability > 0.0)
    return {"prototypes": prototypes, "counts": counts, "reliability": reliability, "admitted": admitted}


def predictions_for(
    writer_ids: list[int],
    data: dict[str, np.ndarray],
    stats: dict[str, np.ndarray],
    alpha: float,
    mapping: dict[int, str],
    use_writer_residual: bool = False,
) -> tuple[dict, list[dict]]:
    labels, writers, splits, logits = (data[name] for name in ("labels", "writers", "splits", "logits"))
    embeddings = normalize(data["embeddings"])
    probabilities: list[float] = []
    records: list[dict] = []
    decisions: list[dict] = []
    enabled: set[tuple[int, str]] = set()
    for writer in writer_ids:
        for size in SUPPORT_SIZES:
            episode = f"support_{size}"
            calibration, _ = _episode_indices(writer, size, labels, writers, splits)
            query = np.flatnonzero((writers == writer) & (splits == 1))
            residual = np.zeros(embeddings.shape[1], np.float32)
            if use_writer_residual:
                supported = calibration[stats["admitted"][labels[calibration]]]
                if len(supported):
                    residual = np.median(embeddings[supported] - stats["prototypes"][labels[supported]], axis=0).astype(np.float32)
            adjusted = normalize((stats["prototypes"] + residual[None]).astype(np.float32))
            for index in query.tolist():
                top5 = np.argsort(logits[index])[-5:][::-1].astype(int)
                baseline = int(top5[0])
                decisions.append({"writer": writer, "episode": episode, "index": index, "baseline": baseline, "truth": int(labels[index])})
                # Any incomplete Top-5 prototype coverage makes the whole row
                # strict identity.  This is label-blind at inference time and
                # prevents wrong-to-wrong motion on extension-only classes.
                if not np.all(stats["admitted"][top5]):
                    continue
                values = logits[index, top5].astype(np.float64)
                z = (values - values.mean()) / max(float(values.std()), 1e-6)
                cosines = embeddings[index] @ adjusted[top5].T
                scores = z + alpha * cosines
                allowed = stats["admitted"][top5]
                scores = np.where(allowed, scores, -np.inf)
                selected = int(top5[int(np.argmax(scores))])
                if selected != baseline:
                    records.append({"writer": writer, "episode": episode, "index": index, "candidate": selected})
                    probabilities.append(1.0)
            enabled.add((writer, episode))
    result, trace = _evaluate_detailed(
        np.asarray(probabilities, np.float32), records, decisions, 0.5, enabled,
        labels, logits, mapping,
    )
    return result, trace


def subset_result(
    writer_ids: list[int], data: dict[str, np.ndarray], stats: dict[str, np.ndarray], alpha: float,
    mapping: dict[int, str], allowed: set[int], residual: bool = False,
) -> dict:
    # Reuse the exact prediction construction, then filter the returned trace is
    # insufficient for unchanged rows; rebuild through the evaluator contract.
    labels = data["labels"]
    full, trace = predictions_for(writer_ids, data, stats, alpha, mapping, residual)
    selected = {(row["global_writer_id"], row["episode"], row["query_index"]): row["adapted"] for row in trace}
    probabilities, records, decisions = [], [], []
    enabled = set()
    for writer in writer_ids:
        for size in SUPPORT_SIZES:
            episode = f"support_{size}"
            query = np.flatnonzero((data["writers"] == writer) & (data["splits"] == 1))
            for index in query.tolist():
                if int(labels[index]) not in allowed:
                    continue
                top = int(np.argmax(data["logits"][index]))
                decisions.append({"writer": writer, "episode": episode, "index": index, "baseline": top, "truth": int(labels[index])})
                chosen = selected.get((mapping[writer], episode, index), top)
                if chosen != top:
                    records.append({"writer": writer, "episode": episode, "index": index, "candidate": chosen})
                    probabilities.append(1.0)
            enabled.add((writer, episode))
    result, _ = _evaluate_detailed(np.asarray(probabilities, np.float32), records, decisions, 0.5, enabled, labels, data["logits"], mapping)
    return result


def nonregression(result: dict) -> bool:
    return (
        not result["writer_top1_regressions_global"]
        and not result["writer_formula_exact_regressions_global"]
        and result["adapted_top1"] >= result["baseline_top1"]
        and result["adapted_formula_exact"] >= result["baseline_formula_exact"]
        and result["candidate_set_violations"] == 0
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite output: {args.output}")

    old = load_cache(OLD_CACHE)
    consumed = load_cache(CONSUMED_CACHE)
    ordered = sorted(set(old["writers"].tolist()), key=lambda writer: hashlib.sha256(f"{SEED}:writer-split:{writer}".encode()).hexdigest())
    fit, dev = ordered[:20], ordered[20:24]
    r1 = json.loads(R1_RECEIPT.read_text(encoding="utf-8"))
    consumed_ids = r1["outer_local_indices"]
    if set(consumed["writers"].tolist()) != set(consumed_ids):
        raise ValueError("consumed r1 subset cache writer mismatch")
    old_mapping = {writer: f"old_synthetic_writer_{writer:03d}" for writer in set(old["writers"].tolist())}
    extension_mapping = _global_map(EXT_BANK)
    stats = prototype_stats(old, fit)

    grid = []
    for alpha in ALPHAS:
        result, _ = predictions_for(dev, old, stats, alpha, old_mapping)
        grid.append({"alpha": alpha, "result": result, "eligible": nonregression(result) and result["adapted_top1"] > result["baseline_top1"]})
    eligible = [row for row in grid if row["eligible"]]
    if not eligible:
        args.output.mkdir(parents=True)
        (args.output / "pre_reserve_rejection.json").write_text(json.dumps({"status": "PRE_RESERVE_FAIL_CLOSED", "grid": grid}, ensure_ascii=False, indent=2), encoding="utf-8")
        return 2
    selected = max(eligible, key=lambda row: (row["result"]["adapted_top1"], row["result"]["adapted_formula_exact"], -row["alpha"]))
    alpha = float(selected["alpha"])
    consumed_result, trace = predictions_for(consumed_ids, consumed, stats, alpha, extension_mapping)
    old_classes, new_classes, class_audit = _class_sets(OLD_BANK, EXT_BANK)
    common = subset_result(consumed_ids, consumed, stats, alpha, extension_mapping, old_classes & new_classes)
    extension_only = subset_result(consumed_ids, consumed, stats, alpha, extension_mapping, new_classes - old_classes)

    # Component B is an ablation, not automatically part of the frozen scorer.
    residual_old, _ = predictions_for(dev, old, stats, alpha, old_mapping, True)
    residual_consumed, _ = predictions_for(consumed_ids, consumed, stats, alpha, extension_mapping, True)
    residual_positive = (
        nonregression(residual_old) and nonregression(residual_consumed)
        and residual_old["adapted_top1"] > selected["result"]["adapted_top1"]
        and residual_consumed["adapted_top1"] > consumed_result["adapted_top1"]
    )
    # Keep B disabled unless its independent increment clears every gate.
    residual_enabled = bool(residual_positive)

    unsupported = set(np.flatnonzero(~stats["admitted"]).tolist())
    unsupported_rows = int(np.isin(consumed["labels"], list(unsupported)).sum())
    new_only_identity = (
        extension_only["adapted_top1"] == extension_only["baseline_top1"]
        and extension_only["adapted_formula_exact"] == extension_only["baseline_formula_exact"]
        and extension_only["changed"] == 0
        and not extension_only["writer_top1_regressions_global"]
        and not extension_only["writer_formula_exact_regressions_global"]
    )
    passed = nonregression(consumed_result) and nonregression(common) and new_only_identity

    args.output.mkdir(parents=True)
    source_snapshot_path = args.output / "freeze_global_prototype_shape_scorer_v9.source.py"
    source_snapshot_path.write_bytes(Path(__file__).read_bytes())
    prototype_path = args.output / "global_class_prototypes.npz"
    config_path = args.output / "frozen_config.json"
    np.savez_compressed(prototype_path, **stats)
    config = {
        "alpha": alpha,
        "minimum_class_support": MIN_CLASS_SUPPORT,
        "writer_residual_enabled": residual_enabled,
        "fit_old_writers": fit,
        "development_old_writers": dev,
        "consumed_r1_diagnostic_writers": [extension_mapping[writer] for writer in consumed_ids],
        "reserve_opened": False,
        "unsupported_contract": "any unsupported class in baseline Top-5 => whole row identity",
        "homograph_boundary": "shape scorer only; semantic ownership remains with context layer",
    }
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    trace_path = args.output / "consumed_r1_changed_prediction_trace.jsonl"
    with trace_path.open("w", encoding="utf-8") as stream:
        for row in trace:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "schema": "aiflow-global-cleanroom-prototype-shape-scorer/v9",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "GLOBAL_PROTOTYPE_DEVELOPMENT_PASS_RESERVE_CLOSED" if passed else "PRE_RESERVE_FAIL_CLOSED",
        "boundary": "synthetic style on known external parent manifold; not commercial promotion evidence",
        "selection": {"source": "old synthetic dev4 only", "grid": grid, "selected_alpha": alpha},
        "component_ablation": {
            "A_global_prototype": {"old_dev": selected["result"], "consumed_r1_diagnostic": consumed_result},
            "B_writer_residual": {"old_dev": residual_old, "consumed_r1_diagnostic": residual_consumed, "positive_increment": residual_positive, "enabled": residual_enabled},
        },
        "coverage": {"class_sets": class_audit, "common": common, "extension_only": extension_only, "unsupported_rows": unsupported_rows},
        "gates": {
            "old_dev_nonregression_positive": selected["eligible"],
            "consumed_r1_all_writer_nonregression": nonregression(consumed_result),
            "common_all_writer_nonregression": nonregression(common),
            "extension_only_full_identity": new_only_identity,
            "candidate_set_exact": consumed_result["candidate_set_violations"] == 0,
            "reserve_opened": False,
            "crohme_opened": False,
            "mathwriting_opened": False,
            "known_replay_opened": False,
            "product_promotion": False,
        },
        "hashes": {
            "old_cache_sha256": sha256(OLD_CACHE), "consumed_r1_cache_sha256": sha256(CONSUMED_CACHE),
            "r1_receipt_sha256": sha256(R1_RECEIPT),
            "generator_script_path": str(Path(__file__).resolve()),
            "generator_script_sha256": sha256(Path(__file__)),
            "generator_source_snapshot_sha256": sha256(source_snapshot_path),
            "prototype_sha256": sha256(prototype_path), "config_sha256": sha256(config_path), "trace_sha256": sha256(trace_path),
        },
    }
    report_path = args.output / "development_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "freeze_receipt.json").write_text(json.dumps({
        "status": "GLOBAL_PROTOTYPE_SCORER_FROZEN_RESERVE_UNOPENED" if passed else "PRE_RESERVE_FAIL_CLOSED",
        "development_report_sha256": sha256(report_path), **report["hashes"],
    }, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "status": report["status"], "alpha": alpha, "residual_enabled": residual_enabled}, ensure_ascii=False))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
