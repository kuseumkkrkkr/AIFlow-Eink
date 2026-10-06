#!/usr/bin/env python3
"""Evaluate formula-disjoint few-shot writer adaptation on a frozen HWR."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from evaluate_known_writer_disjoint_replay_v7 import fresh_rows
from hierarchical_writer_model_v7 import DEFAULT_CHECKPOINT, DEFAULT_FROZEN, build_raw_catalog, sha256
from train_character_classifier_v1 import InkClassifierV1


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V7 = ROOT / "artifacts/hierarchical_writer_model_v7_20260823_r1"
DEFAULT_OUTPUT = ROOT / "artifacts/writer_adaptation_v8_20260823_r1_shadow"
SCHEMA = "aiflow-writer-adaptation/v8"
SEED = 20260823


@dataclass(frozen=True)
class AdapterConfig:
    name: str
    kind: str
    prototype_weight: float = 0.0
    prior_weight: float = 0.0
    neural_rank: int = 0
    neural_scale: float = 0.0
    neural_epochs: int = 0
    neural_lr: float = 0.0
    neural_weight_decay: float = 0.0


CONFIGS = (
    AdapterConfig("identity", "identity"),
    AdapterConfig("prototype_025", "prototype", prototype_weight=0.25),
    AdapterConfig("prototype_050", "prototype", prototype_weight=0.50),
    AdapterConfig("prototype_100", "prototype", prototype_weight=1.00),
    AdapterConfig("prototype_200", "prototype", prototype_weight=2.00),
    AdapterConfig("prior_010", "prior", prior_weight=0.10),
    AdapterConfig("prior_025", "prior", prior_weight=0.25),
    AdapterConfig("prototype_prior", "prototype_prior", prototype_weight=1.00, prior_weight=0.25),
    AdapterConfig("neural_rank2", "neural", neural_rank=2, neural_scale=1.0, neural_epochs=60, neural_lr=0.01, neural_weight_decay=0.10),
)


class LowRankCandidateAdapter(nn.Module):
    """Tiny residual adapter; it can only reorder the frozen baseline Top-5."""

    def __init__(self, classes: int, rank: int, scale: float) -> None:
        super().__init__()
        self.down = nn.Linear(128, rank, bias=False)
        self.up = nn.Linear(rank, classes, bias=True)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight); nn.init.zeros_(self.up.bias)
        self.scale = scale

    def forward(self, centered_embedding: torch.Tensor) -> torch.Tensor:
        return self.scale * torch.tanh(self.up(torch.tanh(self.down(centered_embedding))))


def _split(writer: str, sample_ids: list[str]) -> dict:
    ordered = sorted(set(sample_ids), key=lambda value: hashlib.sha256(f"{SEED}:{writer}:{value}".encode()).hexdigest())
    if len(ordered) < 4:
        raise ValueError(f"writer {writer} needs at least four formulae")
    calibration_count = min(6, max(2, len(ordered) // 4))
    calibration = ordered[:calibration_count]; evaluation = ordered[calibration_count:]
    if set(calibration) & set(evaluation):
        raise AssertionError("formula split overlap")
    return {"writer_id": writer, "calibration_formulae": calibration, "evaluation_formulae": evaluation}


def _load_v7(v7: Path) -> tuple[np.ndarray, list[dict]]:
    with np.load(v7 / "raw_legacy_writer_catalog.npz", allow_pickle=False) as payload:
        features = np.asarray(payload["features"], dtype=np.float32)
    metadata = [json.loads(line) for line in (v7 / "raw_legacy_writer_catalog.metadata.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if features.shape != (393, 128, 5) or len(metadata) != 393:
        raise ValueError("v7 raw catalog contract changed")
    return features, metadata


def _frozen_outputs(features: np.ndarray, checkpoint: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    labels = [str(value) for value in payload["math_labels"]]
    model = InkClassifierV1(len(labels), None); model.load_state_dict(payload["state_dict"], strict=True); model.eval()
    embeddings = []; logits = []
    with torch.no_grad():
        for start in range(0, len(features), 256):
            tensor = torch.from_numpy(features[start:start + 256])
            embedding = model.encode(tensor)
            embeddings.append(embedding.cpu().numpy()); logits.append(model.math_head(embedding).cpu().numpy())
    return labels, np.concatenate(embeddings), np.concatenate(logits)


def _candidate_masked(logits: np.ndarray, baseline_top5: np.ndarray) -> np.ndarray:
    output = np.full_like(logits, -1.0e9)
    rows = np.arange(len(logits))[:, None]
    output[rows, baseline_top5] = logits[rows, baseline_top5]
    return output


def _fit_apply(
    config: AdapterConfig, calibration_embeddings: np.ndarray, calibration_logits: np.ndarray,
    calibration_truth: np.ndarray, evaluation_embeddings: np.ndarray, evaluation_logits: np.ndarray,
) -> tuple[np.ndarray, dict]:
    baseline_top5 = np.argsort(evaluation_logits, axis=1)[:, -5:][:, ::-1]
    adapted = evaluation_logits.copy(); state: dict[str, Any] = {"config": asdict(config)}
    if config.kind in {"prototype", "prototype_prior"}:
        normalized_cal = calibration_embeddings / np.maximum(np.linalg.norm(calibration_embeddings, axis=1, keepdims=True), 1e-8)
        normalized_eval = evaluation_embeddings / np.maximum(np.linalg.norm(evaluation_embeddings, axis=1, keepdims=True), 1e-8)
        prototypes = {}
        for label in sorted(set(calibration_truth.tolist())):
            centroid = normalized_cal[calibration_truth == label].mean(axis=0)
            centroid /= max(float(np.linalg.norm(centroid)), 1e-8)
            prototypes[int(label)] = centroid
            adapted[:, label] += config.prototype_weight * (normalized_eval @ centroid)
        state["prototype_classes"] = sorted(prototypes)
    if config.kind in {"prior", "prototype_prior"}:
        counts = np.bincount(calibration_truth, minlength=calibration_logits.shape[1]).astype(np.float64)
        prior = np.log((counts + 0.5) / (counts.sum() + 0.5 * len(counts)))
        prior -= prior.mean()
        adapted += config.prior_weight * prior[None]
        state["calibration_class_counts"] = {str(index): int(value) for index, value in enumerate(counts) if value}
    if config.kind == "neural":
        torch.manual_seed(SEED)
        center = calibration_embeddings.mean(axis=0, keepdims=True).astype(np.float32)
        module = LowRankCandidateAdapter(calibration_logits.shape[1], config.neural_rank, config.neural_scale)
        optimizer = torch.optim.AdamW(module.parameters(), lr=config.neural_lr, weight_decay=config.neural_weight_decay)
        x = torch.from_numpy((calibration_embeddings - center).astype(np.float32))
        frozen = torch.from_numpy(calibration_logits.astype(np.float32))
        truth = torch.from_numpy(calibration_truth.astype(np.int64))
        top5 = frozen.argsort(dim=1, descending=True)[:, :5]
        eligible = (top5 == truth[:, None]).any(dim=1)
        if int(eligible.sum()) >= 2:
            for _epoch in range(config.neural_epochs):
                residual = module(x[eligible]); candidate = top5[eligible]
                values = frozen[eligible].gather(1, candidate) + residual.gather(1, candidate)
                local_truth = (candidate == truth[eligible, None]).to(torch.int64).argmax(dim=1)
                loss = F.cross_entropy(values, local_truth)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
        with torch.no_grad():
            residual = module(torch.from_numpy((evaluation_embeddings - center).astype(np.float32))).numpy()
        adapted += residual
        state.update({"fit_rows": int(eligible.sum()), "parameters": sum(value.numel() for value in module.parameters())})
    return _candidate_masked(adapted, baseline_top5), state


def _score(
    metadata: list[dict], indices: list[int], truth: np.ndarray, baseline_logits: np.ndarray,
    adapted_logits: np.ndarray,
) -> dict:
    base_rank = np.argsort(baseline_logits, axis=1)[:, -5:][:, ::-1]
    adapt_rank = np.argsort(adapted_logits, axis=1)[:, -5:][:, ::-1]
    formulae: dict[str, list[int]] = defaultdict(list)
    for local, index in enumerate(indices):
        formulae[str(metadata[index]["sample_id"])].append(local)

    def metrics(rank: np.ndarray) -> dict:
        return {
            "glyphs": len(truth), "top1": float(np.mean(rank[:, 0] == truth)),
            "top5": float(np.mean(np.any(rank == truth[:, None], axis=1))),
            "formulae": len(formulae),
            "formula_exact": float(np.mean([all(rank[i, 0] == truth[i] for i in rows) for rows in formulae.values()])),
            "formula_top5_oracle": float(np.mean([all(truth[i] in rank[i] for i in rows) for rows in formulae.values()])),
        }
    per_class = {}
    confusion = Counter()
    for label in sorted(set(truth.tolist())):
        mask = truth == label
        per_class[str(label)] = {
            "rows": int(mask.sum()), "baseline_top1": float(np.mean(base_rank[mask, 0] == label)),
            "adapted_top1": float(np.mean(adapt_rank[mask, 0] == label)),
        }
    for expected, old, new in zip(truth, base_rank[:, 0], adapt_rank[:, 0], strict=True):
        confusion[f"{expected}->{old}->{new}"] += 1
    violations = int(sum(adapt_rank[row, 0] not in base_rank[row] for row in range(len(truth))))
    return {
        "baseline": metrics(base_rank), "adapted": metrics(adapt_rank),
        "candidate_violations": violations, "per_class": per_class,
        "confusion": dict(confusion),
    }


def _writer_trial(
    config: AdapterConfig, writer: str, split: dict, features: np.ndarray, metadata: list[dict],
    embeddings: np.ndarray, logits: np.ndarray,
) -> tuple[dict, dict]:
    cal_ids = set(split["calibration_formulae"]); eval_ids = set(split["evaluation_formulae"])
    cal = [i for i, row in enumerate(metadata) if row["writer_id"] == writer and row["sample_id"] in cal_ids and row["label_index"] >= 0]
    evaluation = [i for i, row in enumerate(metadata) if row["writer_id"] == writer and row["sample_id"] in eval_ids and row["label_index"] >= 0]
    if not cal or not evaluation or cal_ids & eval_ids:
        raise ValueError(f"invalid calibration/evaluation rows for {writer}")
    cal_truth = np.asarray([metadata[i]["label_index"] for i in cal], dtype=np.int64)
    eval_truth = np.asarray([metadata[i]["label_index"] for i in evaluation], dtype=np.int64)
    adapted, state = _fit_apply(config, embeddings[cal], logits[cal], cal_truth, embeddings[evaluation], logits[evaluation])
    result = _score(metadata, evaluation, eval_truth, logits[evaluation], adapted)
    result.update({"writer_id": writer, "calibration_glyphs": len(cal), "evaluation_glyphs": len(evaluation)})
    return result, state


def _aggregate(rows: list[dict]) -> dict:
    glyphs = sum(row["baseline"]["glyphs"] for row in rows)
    formulae = sum(row["baseline"]["formulae"] for row in rows)
    output = {"writers": len(rows), "glyphs": glyphs, "formulae": formulae,
              "candidate_violations": sum(row["candidate_violations"] for row in rows)}
    for version in ("baseline", "adapted"):
        output[version] = {
            "top1": sum(row[version]["top1"] * row[version]["glyphs"] for row in rows) / glyphs,
            "top5": sum(row[version]["top5"] * row[version]["glyphs"] for row in rows) / glyphs,
            "formula_exact": sum(row[version]["formula_exact"] * row[version]["formulae"] for row in rows) / formulae,
            "formula_top5_oracle": sum(row[version]["formula_top5_oracle"] * row[version]["formulae"] for row in rows) / formulae,
        }
    return output


def _config_key(summary: dict) -> tuple:
    adapted = summary["adapted"]; baseline = summary["baseline"]
    safe = summary["candidate_violations"] == 0 and adapted["top1"] >= baseline["top1"] and adapted["formula_exact"] >= baseline["formula_exact"]
    return (int(safe), adapted["formula_exact"] - baseline["formula_exact"], adapted["top1"] - baseline["top1"], adapted["top5"])


def _writer_regressions(rows: list[dict]) -> list[dict]:
    regressions = []
    for row in rows:
        reasons = []
        if row["candidate_violations"]:
            reasons.append("candidate_violation")
        if row["adapted"]["top1"] < row["baseline"]["top1"]:
            reasons.append("top1")
        if row["adapted"]["formula_exact"] < row["baseline"]["formula_exact"]:
            reasons.append("formula_exact")
        if reasons:
            regressions.append({
                "writer_id": row["writer_id"],
                "reasons": reasons,
                "top1_delta": row["adapted"]["top1"] - row["baseline"]["top1"],
                "formula_exact_delta": row["adapted"]["formula_exact"] - row["baseline"]["formula_exact"],
            })
    return regressions


def _legacy_safe_positive(rows: list[dict]) -> bool:
    aggregate = _aggregate(rows)
    baseline = aggregate["baseline"]
    adapted = aggregate["adapted"]
    positive = adapted["top1"] > baseline["top1"] or adapted["formula_exact"] > baseline["formula_exact"]
    return not _writer_regressions(rows) and positive


def _predictions(metadata: list[dict], indices: list[int], labels: list[str], baseline: np.ndarray, adapted: np.ndarray) -> list[dict]:
    base = np.argsort(baseline, axis=1)[:, -5:][:, ::-1]; new = np.argsort(adapted, axis=1)[:, -5:][:, ::-1]
    return [{
        "sample_id": metadata[index]["sample_id"], "writer_id": metadata[index]["writer_id"],
        "group_index": metadata[index]["group_index"], "truth": metadata[index]["token"],
        "baseline_top5": [labels[value] for value in base[row]],
        "adapted_top5": [labels[value] for value in new[row]],
    } for row, index in enumerate(indices)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v7", type=Path, default=DEFAULT_V7)
    parser.add_argument("--frozen", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    paths = [args.v7.resolve(), args.frozen.resolve(), args.checkpoint.resolve(), args.output.resolve()]
    if any(path.drive.upper() != "D:" for path in paths): parser.error("all paths must remain on D:")
    if args.output.exists(): parser.error(f"refusing to overwrite output: {args.output}")
    v7_report = json.loads((args.v7 / "report.json").read_text(encoding="utf-8"))
    if sha256(args.checkpoint) != v7_report["inputs"]["checkpoint"]["sha256"]: raise ValueError("frozen checkpoint hash changed")

    legacy_features, legacy_meta = _load_v7(args.v7)
    labels, legacy_embeddings, legacy_logits = _frozen_outputs(legacy_features, args.checkpoint)
    splits = {writer: _split(writer, [row["sample_id"] for row in legacy_meta if row["writer_id"] == writer]) for writer in sorted({row["writer_id"] for row in legacy_meta})}
    grid = {}
    for config in CONFIGS:
        rows = [_writer_trial(config, writer, splits[writer], legacy_features, legacy_meta, legacy_embeddings, legacy_logits)[0] for writer in splits]
        grid[config.name] = {"config": asdict(config), "per_writer": rows, "aggregate": _aggregate(rows)}
    nested = []
    for outer in sorted(splits):
        training_rows = {config.name: [row for row in grid[config.name]["per_writer"] if row["writer_id"] != outer] for config in CONFIGS}
        eligible = [config for config in CONFIGS if config.kind != "identity" and _legacy_safe_positive(training_rows[config.name])]
        selected = max(eligible, key=lambda config: _config_key(_aggregate(training_rows[config.name]))) if eligible else CONFIGS[0]
        nested.append({"held_out_writer": outer, "selected_config": selected.name,
                       "training_fail_closed": not eligible,
                       "result": next(row for row in grid[selected.name]["per_writer"] if row["writer_id"] == outer)})
    globally_eligible = [config for config in CONFIGS if config.kind != "identity" and _legacy_safe_positive(grid[config.name]["per_writer"])]
    if not globally_eligible:
        output = args.output.resolve(); output.mkdir(parents=True)
        rejection = {
            "schema": SCHEMA,
            "status": "PRE_REPLAY_FAIL_CLOSED",
            "reason": "no adapter achieved positive aggregate gain with zero per-writer Top1/formula-exact regressions",
            "known_replay_opened": False,
            "candidate_grid": grid,
            "nested_writer_loo": nested,
        }
        (output / "pre_replay_rejection.json").write_text(json.dumps(rejection, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"output": str(output), "status": rejection["status"], "known_replay_opened": False}, ensure_ascii=False))
        return 2
    selected = max(globally_eligible, key=lambda config: _config_key(grid[config.name]["aggregate"]))
    selected_legacy = grid[selected.name]
    neural_best = max((row for row in grid.values() if row["config"]["kind"] == "neural"), key=lambda row: _config_key(row["aggregate"]))
    fallback_used = selected.kind != "neural"

    # Selection is frozen above. Only now open the known replay for one diagnostic.
    replay_formulas, replay_ownership, replay_hashes = fresh_rows(args.frozen)
    _catalog, replay_features, replay_meta = build_raw_catalog(
        replay_formulas, replay_ownership, labels, replay_hashes,
        source_domain="project_owned_known_writer_replay_raw_observed",
        evidence_tier="known_writer_disjoint_replay_diagnostic_raw_observed",
    )
    _labels2, replay_embeddings, replay_logits = _frozen_outputs(replay_features, args.checkpoint)
    replay_splits = {writer: _split(writer, [row["sample_id"] for row in replay_meta if row["writer_id"] == writer]) for writer in sorted({row["writer_id"] for row in replay_meta})}
    replay_rows = []; replay_states = {}; prediction_rows = []
    for writer in replay_splits:
        result, state = _writer_trial(selected, writer, replay_splits[writer], replay_features, replay_meta, replay_embeddings, replay_logits)
        replay_rows.append(result); replay_states[writer] = state
        cal_ids = set(replay_splits[writer]["calibration_formulae"]); eval_ids = set(replay_splits[writer]["evaluation_formulae"])
        cal = [i for i, row in enumerate(replay_meta) if row["writer_id"] == writer and row["sample_id"] in cal_ids and row["label_index"] >= 0]
        evaluation = [i for i, row in enumerate(replay_meta) if row["writer_id"] == writer and row["sample_id"] in eval_ids and row["label_index"] >= 0]
        truth = np.asarray([replay_meta[i]["label_index"] for i in cal], dtype=np.int64)
        adapted, _ = _fit_apply(selected, replay_embeddings[cal], replay_logits[cal], truth, replay_embeddings[evaluation], replay_logits[evaluation])
        prediction_rows.extend(_predictions(replay_meta, evaluation, labels, replay_logits[evaluation], adapted))
    replay_aggregate = _aggregate(replay_rows)
    legacy_writer_regressions = _writer_regressions(selected_legacy["per_writer"])
    replay_writer_regressions = _writer_regressions(replay_rows)
    automatic_reject = (
        not bool(_config_key(replay_aggregate)[0])
        or bool(legacy_writer_regressions)
        or bool(replay_writer_regressions)
    )

    output = args.output.resolve(); output.mkdir(parents=True)
    split_manifest = {
        "schema": "aiflow-writer-adaptation-split/v8", "seed": SEED,
        "policy": "sha256(seed,writer,sample_id); calibration=min(6,max(2,floor(formulae/4)))",
        "legacy": splits, "known_replay": replay_splits,
        "source_hashes": {"v7_report": sha256(args.v7 / "report.json"), **replay_hashes},
    }
    (output / "split_manifest.json").write_text(json.dumps(split_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output / "known_replay_predictions.jsonl").open("w", encoding="utf-8") as stream:
        for row in prediction_rows: stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output / "adapter_states.json").write_text(json.dumps(replay_states, ensure_ascii=False, indent=2), encoding="utf-8")
    report = {
        "schema": SCHEMA, "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "AUTO_REJECTED" if automatic_reject else "WRITER_ADAPTATION_SHADOW_DIAGNOSTIC_PASS",
        "selection": {
            "protocol": "legacy formula-disjoint nested writer-LOO; replay unopened until freeze",
            "candidate_grid": grid, "nested_writer_loo": nested,
            "selected_config": asdict(selected), "neural_candidate": neural_best,
            "fallback_from_neural": fallback_used,
            "selected_legacy_aggregate": selected_legacy["aggregate"],
            "legacy_writer_regressions": legacy_writer_regressions,
        },
        "known_replay": {
            "role": "known_writer_disjoint_replay_diagnostic_not_promotion",
            "per_writer": replay_rows, "aggregate": replay_aggregate,
            "automatic_reject": automatic_reject,
            "writer_regressions": replay_writer_regressions,
            "writer_010_warning": "only four formulae; two calibration and two evaluation; do not generalize",
        },
        "calibration_contract": {
            "inputs": ["frozen 128-d embedding", "frozen logits", "calibration labels", "calibration sample IDs"],
            "trainable_parameters": asdict(selected),
            "evaluation_labels": "scoring only; absent from fit/apply/config selection",
            "candidate_policy": "adapter may only reorder each glyph's frozen baseline Top-5",
        },
        "gates": {
            "formula_disjoint": True, "candidate_preserving": replay_aggregate["candidate_violations"] == 0,
            "replay_used_for_tuning": False, "automatic_reject_on_regression": True,
            "legacy_all_writer_nonregression": not legacy_writer_regressions,
            "replay_all_writer_nonregression": not replay_writer_regressions,
            "adapter_accepted_in_replay": not automatic_reject,
            "promotion_evidence": False, "promotion_ready": False,
            "checkpoint_changed": False, "hwr_changed": False, "runtime_changed": False,
            "crohme_rows": 0, "mathwriting_rows": 0, "v7_changed": False,
        },
        "inputs": {"v7": str(args.v7.resolve()), "checkpoint_sha256": sha256(args.checkpoint), "frozen_root": str(args.frozen.resolve())},
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "selected": selected.name, "fallback_from_neural": fallback_used,
                      "legacy": selected_legacy["aggregate"], "replay": replay_aggregate,
                      "automatic_reject": automatic_reject, "gates": report["gates"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
