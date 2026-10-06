"""Train a candidate-preserving sparse writer meta-ranker on clean-room writers."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from build_cleanroom_writer_style_bank_v5 import sha256
from evaluate_writer_adaptation_v8 import _aggregate, _frozen_outputs, _load_v7, _score, _split, _writer_regressions
from hierarchical_writer_model_v7 import DEFAULT_CHECKPOINT
from train_cleanroom_writer_adapter_v9 import DEFAULT_BANK, DEFAULT_V7, SUPPORT_SIZES, _episode_indices, _load_bank


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / "artifacts/writer_adaptation_v9_frozen_cache_20260823_r1.npz"
DEFAULT_OUTPUT = ROOT / "artifacts/writer_adaptation_v9_meta_20260823_r1_shadow"
SEED = 20260823
THRESHOLDS = (1, 2, 3)
SCALES = (0.25, 0.5, 1.0, 1.5, 2.0)
ACTION_MARGINS = (0.0, 0.02, 0.05, 0.10, 0.20)


class CandidateMetaRanker(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(11, 16), nn.Tanh(), nn.Linear(16, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def _cache_outputs(bank: Path, checkpoint: Path, cache: Path):
    arrays, report = _load_bank(bank)
    expected = {"bank": sha256(bank / "synthetic_writer_style_bank_v5_r4.npz"), "checkpoint": sha256(checkpoint)}
    if cache.exists():
        with np.load(cache, allow_pickle=False) as payload:
            if str(payload["bank_sha256"].item()) != expected["bank"] or str(payload["checkpoint_sha256"].item()) != expected["checkpoint"]:
                raise ValueError("frozen output cache provenance mismatch")
            return ({key: np.asarray(payload[key]).copy() for key in ("labels", "writers", "splits", "embeddings", "logits")}, report)
    _labels, embeddings, logits = _frozen_outputs(arrays["features"].astype(np.float32), checkpoint)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, labels=arrays["labels"], writers=arrays["writer_index"], splits=arrays["episode_split"],
                        embeddings=embeddings.astype(np.float32), logits=logits.astype(np.float32),
                        bank_sha256=np.asarray(expected["bank"]), checkpoint_sha256=np.asarray(expected["checkpoint"]))
    return ({"labels": arrays["labels"], "writers": arrays["writer_index"], "splits": arrays["episode_split"],
             "embeddings": embeddings, "logits": logits}, report)


def _candidate_features(cal_embeddings: np.ndarray, cal_truth: np.ndarray,
                        eval_embeddings: np.ndarray, eval_logits: np.ndarray, threshold: int):
    top5 = np.argsort(eval_logits, axis=1)[:, -5:][:, ::-1]
    candidate_logits = np.take_along_axis(eval_logits, top5, axis=1)
    mean = candidate_logits.mean(axis=1, keepdims=True)
    std = np.maximum(candidate_logits.std(axis=1, keepdims=True), 1.0e-6)
    baseline_z = (candidate_logits - mean) / std
    cal_norm = cal_embeddings / np.maximum(np.linalg.norm(cal_embeddings, axis=1, keepdims=True), 1.0e-8)
    eval_norm = eval_embeddings / np.maximum(np.linalg.norm(eval_embeddings, axis=1, keepdims=True), 1.0e-8)
    counts = Counter(int(value) for value in cal_truth.tolist())
    prototypes = {}
    for label, count in counts.items():
        if count >= threshold:
            centroid = cal_norm[cal_truth == label].mean(axis=0)
            prototypes[label] = centroid / max(float(np.linalg.norm(centroid)), 1.0e-8)
    support = np.zeros_like(candidate_logits, dtype=np.float32)
    cosine = np.zeros_like(candidate_logits, dtype=np.float32)
    competing_delta = np.zeros_like(candidate_logits, dtype=np.float32)
    loo_precision = np.zeros_like(candidate_logits, dtype=np.float32)
    prototype_labels = sorted(prototypes)
    prototype_matrix = np.stack([prototypes[label] for label in prototype_labels]) if prototype_labels else np.zeros((0, eval_norm.shape[1]), dtype=np.float32)
    all_similarities = eval_norm @ prototype_matrix.T if prototype_labels else np.zeros((len(eval_norm), 0), dtype=np.float32)
    class_precision = {}
    for label in prototype_labels:
        members = np.flatnonzero(cal_truth == label)
        correct = 0; scored = 0
        for target in members.tolist():
            peers = members[members != target]
            if not len(peers): continue
            own = cal_norm[peers].mean(axis=0); own /= max(float(np.linalg.norm(own)), 1.0e-8)
            competing = [float(cal_norm[target] @ prototypes[other]) for other in prototype_labels if other != label]
            correct += int(float(cal_norm[target] @ own) >= (max(competing) if competing else -1.0)); scored += 1
        class_precision[label] = correct / scored if scored else 0.0
    for column in range(5):
        for label in set(top5[:, column].tolist()):
            mask = top5[:, column] == label
            if int(label) in prototypes:
                support[mask, column] = np.log1p(counts[int(label)]) / np.log(4.0)
                cosine[mask, column] = eval_norm[mask] @ prototypes[int(label)]
                own_index = prototype_labels.index(int(label))
                if len(prototype_labels) > 1:
                    competitor = np.max(np.delete(all_similarities[mask], own_index, axis=1), axis=1)
                else:
                    competitor = np.full(int(mask.sum()), -1.0, dtype=np.float32)
                competing_delta[mask, column] = cosine[mask, column] - competitor
                loo_precision[mask, column] = class_precision[int(label)]
    supported = support > 0
    best_cosine = np.where(supported, cosine, -2.0).max(axis=1, keepdims=True)
    cosine_gap = np.where(supported, cosine - best_cosine, 0.0)
    rank = np.broadcast_to(np.arange(5, dtype=np.float32)[None] / 4.0, top5.shape)
    top1 = np.zeros_like(candidate_logits, dtype=np.float32); top1[:, 0] = 1.0
    shifted = np.concatenate((candidate_logits[:, 1:], candidate_logits[:, -1:]), axis=1)
    adjacent_margin = candidate_logits - shifted
    probabilities = np.exp(baseline_z - baseline_z.max(axis=1, keepdims=True)); probabilities /= probabilities.sum(axis=1, keepdims=True)
    entropy = -np.sum(probabilities * np.log(np.maximum(probabilities, 1.0e-8)), axis=1, keepdims=True) / np.log(5.0)
    row_margin = (candidate_logits[:, :1] - candidate_logits[:, 1:2]) / std
    features = np.stack((baseline_z, baseline_z - baseline_z[:, :1], adjacent_margin / std, rank, support,
                         cosine, competing_delta, loo_precision, top1,
                         np.broadcast_to(entropy, top5.shape), np.broadcast_to(row_margin, top5.shape)), axis=2).astype(np.float32)
    return features, baseline_z.astype(np.float32), top5.astype(np.int64), supported


def _build_fit_rows(writer_ids: list[int], labels, writers, splits, embeddings, logits, threshold):
    feature_rows = []; baseline_rows = []; mask_rows = []; targets = []
    for writer in writer_ids:
        for size in SUPPORT_SIZES:
            cal, query = _episode_indices(writer, size, labels, writers, splits)
            features, baseline, top5, supported = _candidate_features(embeddings[cal], labels[cal], embeddings[query], logits[query], threshold)
            local_truth = np.argmax(top5 == labels[query, None], axis=1)
            feature_rows.append(features); baseline_rows.append(baseline); mask_rows.append(supported); targets.append(local_truth)
    return tuple(np.concatenate(rows) for rows in (feature_rows, baseline_rows, mask_rows, targets))


def _train(writer_ids, labels, writers, splits, embeddings, logits, threshold):
    x, baseline, supported, target = _build_fit_rows(writer_ids, labels, writers, splits, embeddings, logits, threshold)
    torch.manual_seed(SEED + threshold)
    model = CandidateMetaRanker()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.05)
    x_t = torch.from_numpy(x); base_t = torch.from_numpy(baseline); support_t = torch.from_numpy(supported.astype(np.float32)); target_t = torch.from_numpy(target.astype(np.int64))
    generator = torch.Generator().manual_seed(SEED + threshold * 101)
    losses = []
    for _epoch in range(8):
        order = torch.randperm(len(x_t), generator=generator)
        epoch_loss = 0.0
        for start in range(0, len(order), 4096):
            index = order[start:start + 4096]
            scores = base_t[index] + 0.5 * torch.tanh(model(x_t[index])) * support_t[index]
            loss = F.cross_entropy(scores, target_t[index])
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            epoch_loss += float(loss) * len(index)
        losses.append(epoch_loss / len(order))
    model.eval()
    return model, losses


def _apply_raw(model, scale, cal_embeddings, cal_truth, eval_embeddings, eval_logits, threshold, action_margin=0.0):
    features, baseline, top5, supported = _candidate_features(cal_embeddings, cal_truth, eval_embeddings, eval_logits, threshold)
    if not supported.any():
        return eval_logits.copy(), {"identity_fallback": True, "supported_candidates": 0}
    with torch.no_grad():
        residual = (0.5 * torch.tanh(model(torch.from_numpy(features))).numpy()) * supported
    score = baseline + scale * residual
    proposed = np.argmax(score, axis=1)
    changed = proposed != 0
    advantage = score[np.arange(len(score)), proposed] - score[:, 0]
    revert = changed & (advantage < action_margin)
    score[revert] = baseline[revert]
    adapted = np.full_like(eval_logits, -1.0e9)
    np.put_along_axis(adapted, top5, score, axis=1)
    return adapted, {"identity_fallback": False, "supported_candidates": int(supported.sum())}


def _apply(model, scale, cal_embeddings, cal_logits, cal_truth, eval_embeddings, eval_logits, threshold,
           action_margin=0.0, use_loo_gate=False):
    if not use_loo_gate:
        return _apply_raw(model, scale, cal_embeddings, cal_truth, eval_embeddings, eval_logits, threshold, action_margin)
    by_label = defaultdict(list)
    for index, label in enumerate(cal_truth.tolist()): by_label[int(label)].append(index)
    eligible = [indices for indices in by_label.values() if len(indices) >= max(2, threshold)]
    prototype_rows = np.asarray([indices[0] for indices in eligible], dtype=np.int64)
    validation_rows = np.asarray([indices[1] for indices in eligible], dtype=np.int64)
    if not len(prototype_rows):
        return eval_logits.copy(), {"identity_fallback": True, "calibration_gate": "no_paired_support", "supported_candidates": 0}
    validation_adapted, _ = _apply_raw(model, scale, cal_embeddings[prototype_rows], cal_truth[prototype_rows],
                                        cal_embeddings[validation_rows], cal_logits[validation_rows], 1, action_margin)
    base = np.argmax(cal_logits[validation_rows], axis=1); new = np.argmax(validation_adapted, axis=1)
    truth = cal_truth[validation_rows]
    improved = int(np.sum((base != truth) & (new == truth))); regressed = int(np.sum((base == truth) & (new != truth)))
    active = improved > 0 and regressed == 0
    if not active:
        return eval_logits.copy(), {"identity_fallback": True, "calibration_gate": "no_positive_nonregressive_loo",
                                    "calibration_loo_improved": improved, "calibration_loo_regressed": regressed,
                                    "supported_candidates": 0}
    adapted, state = _apply_raw(model, scale, cal_embeddings, cal_truth, eval_embeddings, eval_logits, threshold, action_margin)
    state.update({"calibration_gate": "active", "calibration_loo_improved": improved, "calibration_loo_regressed": regressed})
    return adapted, state


def _metric(truth, baseline_logits, adapted_logits):
    base = np.argsort(baseline_logits, axis=1)[:, -5:][:, ::-1]
    new = np.argsort(adapted_logits, axis=1)[:, -5:][:, ::-1]
    old = base[:, 0] == truth; current = new[:, 0] == truth
    chunks = [np.arange(start, min(start + 8, len(truth))) for start in range(0, len(truth), 8)]
    return {"glyphs": len(truth), "formulae": len(chunks), "baseline_top1_correct": int(old.sum()),
            "adapted_top1_correct": int(current.sum()), "baseline_formula_exact": int(sum(np.all(old[c]) for c in chunks)),
            "adapted_formula_exact": int(sum(np.all(current[c]) for c in chunks)),
            "candidate_violations": int(sum(set(base[i]) != set(new[i]) for i in range(len(truth)))),
            "paired_improved": int(np.sum(~old & current)), "paired_regressed": int(np.sum(old & ~current))}


def _synthetic_eval(model, threshold, scale, action_margin, use_loo_gate, writer_ids, labels, writers, splits, embeddings, logits):
    per_writer = []
    for writer in writer_ids:
        total = Counter(); states = []
        for size in SUPPORT_SIZES:
            cal, query = _episode_indices(writer, size, labels, writers, splits)
            adapted, state = _apply(model, scale, embeddings[cal], logits[cal], labels[cal], embeddings[query], logits[query], threshold, action_margin, use_loo_gate)
            total.update(_metric(labels[query], logits[query], adapted)); states.append({"support_size": size, **state})
        per_writer.append({"synthetic_writer_index": writer, "states": states, "glyphs": total["glyphs"], "formulae": total["formulae"],
                           "baseline": {"top1": total["baseline_top1_correct"] / total["glyphs"], "formula_exact": total["baseline_formula_exact"] / total["formulae"]},
                           "adapted": {"top1": total["adapted_top1_correct"] / total["glyphs"], "formula_exact": total["adapted_formula_exact"] / total["formulae"]},
                           "candidate_violations": total["candidate_violations"], "paired_improved": total["paired_improved"], "paired_regressed": total["paired_regressed"]})
    glyphs = sum(r["glyphs"] for r in per_writer); formulae = sum(r["formulae"] for r in per_writer)
    aggregate = {"writers": len(per_writer), "glyphs": glyphs, "formulae": formulae,
                 "baseline": {"top1": sum(r["baseline"]["top1"] * r["glyphs"] for r in per_writer) / glyphs,
                              "formula_exact": sum(r["baseline"]["formula_exact"] * r["formulae"] for r in per_writer) / formulae},
                 "adapted": {"top1": sum(r["adapted"]["top1"] * r["glyphs"] for r in per_writer) / glyphs,
                             "formula_exact": sum(r["adapted"]["formula_exact"] * r["formulae"] for r in per_writer) / formulae},
                 "candidate_violations": sum(r["candidate_violations"] for r in per_writer),
                 "paired_improved": sum(r["paired_improved"] for r in per_writer), "paired_regressed": sum(r["paired_regressed"] for r in per_writer)}
    return per_writer, aggregate


def _positive(row):
    return row["candidate_violations"] == 0 and row["paired_improved"] > row["paired_regressed"] and row["adapted"]["top1"] > row["baseline"]["top1"]


def _parent_overlap(bank: Path, fit_writers: list[int], dev_writers: list[int], outer_writers: list[int]) -> dict:
    with gzip.open(bank / "synthetic_writer_style_bank_v5_r4.metadata.jsonl.gz", "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    groups = {"fit_dev": set(fit_writers + dev_writers), "outer": set(outer_writers)}
    result = {}
    collected = {}
    for name, writer_set in groups.items():
        selected = [row for row in rows if int(row["synthetic_writer_id"].rsplit("_", 1)[1]) in writer_set]
        immediate = {row["parent_synthetic_id"] for row in selected}
        roots = {value for row in selected for value in row.get("parent_fingerprints", [])}
        collected[name] = (immediate, roots)
        result[name] = {"immediate_parents": len(immediate), "root_fingerprints": len(roots)}
    immediate_overlap = collected["fit_dev"][0] & collected["outer"][0]
    root_overlap = collected["fit_dev"][1] & collected["outer"][1]
    result["overlap"] = {"immediate": len(immediate_overlap), "outer_immediate_ratio": len(immediate_overlap) / max(len(collected["outer"][0]), 1),
                         "roots": len(root_overlap), "outer_root_ratio": len(root_overlap) / max(len(collected["outer"][1]), 1)}
    result["evidence_boundary"] = "new synthetic style on known parent manifold; not parent-disjoint or promotion evidence"
    return result


def _legacy(model, threshold, scale, action_margin, use_loo_gate, v7, checkpoint):
    features, metadata = _load_v7(v7); _tokens, embeddings, logits = _frozen_outputs(features, checkpoint)
    manifests = {writer: _split(writer, [r["sample_id"] for r in metadata if r["writer_id"] == writer]) for writer in sorted({r["writer_id"] for r in metadata})}
    rows = []; states = {}
    for writer, split in manifests.items():
        cal_ids = set(split["calibration_formulae"]); eval_ids = set(split["evaluation_formulae"])
        cal = [i for i, r in enumerate(metadata) if r["writer_id"] == writer and r["sample_id"] in cal_ids and r["label_index"] >= 0]
        evaluation = [i for i, r in enumerate(metadata) if r["writer_id"] == writer and r["sample_id"] in eval_ids and r["label_index"] >= 0]
        cal_truth = np.asarray([metadata[i]["label_index"] for i in cal], dtype=np.int64)
        truth = np.asarray([metadata[i]["label_index"] for i in evaluation], dtype=np.int64)
        adapted, state = _apply(model, scale, embeddings[cal], logits[cal], cal_truth, embeddings[evaluation], logits[evaluation], threshold, action_margin, use_loo_gate)
        result = _score(metadata, evaluation, truth, logits[evaluation], adapted)
        base = np.argsort(logits[evaluation], axis=1)[:, -5:][:, ::-1]; new = np.argsort(adapted, axis=1)[:, -5:][:, ::-1]
        old = base[:, 0] == truth; current = new[:, 0] == truth
        result.update({"writer_id": writer, "calibration_glyphs": len(cal), "evaluation_glyphs": len(evaluation),
                       "paired_improved": int(np.sum(~old & current)), "paired_regressed": int(np.sum(old & ~current))})
        rows.append(result); states[writer] = state
    return manifests, rows, states


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK); parser.add_argument("--v7", type=Path, default=DEFAULT_V7)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT); parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT); args = parser.parse_args()
    for path in (args.bank.resolve(), args.v7.resolve(), args.checkpoint.resolve(), args.cache.resolve(), args.output.resolve()):
        if path.drive.upper() != "D:": parser.error("all paths must remain on D:")
    if args.output.exists(): parser.error(f"refusing to overwrite output: {args.output}")
    arrays, _bank_report = _cache_outputs(args.bank, args.checkpoint, args.cache)
    labels=arrays["labels"].astype(np.int64); writers=arrays["writers"].astype(np.int16); splits=arrays["splits"].astype(np.int8)
    embeddings=arrays["embeddings"].astype(np.float32); logits=arrays["logits"].astype(np.float32)
    ordered = sorted(set(writers.tolist()), key=lambda w: hashlib.sha256(f"{SEED}:nested:{w}".encode()).hexdigest())
    fit_writers, dev_writers, holdout_writers = ordered[:20], ordered[20:24], ordered[24:]
    models={}; losses={}; grid={}
    for threshold in THRESHOLDS:
        model, loss = _train(fit_writers, labels, writers, splits, embeddings, logits, threshold)
        models[threshold]=model; losses[threshold]=loss
        for scale in SCALES:
            for action_margin in ACTION_MARGINS:
                for use_loo_gate in (False, True):
                    dev_rows, dev = _synthetic_eval(model, threshold, scale, action_margin, use_loo_gate, dev_writers, labels, writers, splits, embeddings, logits)
                    grid[f"support{threshold}_scale{scale}_margin{action_margin}_loo{int(use_loo_gate)}"]={"threshold":threshold,"scale":scale,"action_margin":action_margin,"use_loo_gate":use_loo_gate,"dev":dev,"dev_per_writer":dev_rows}
    eligible=[row for row in grid.values() if _positive(row["dev"])]
    selected=max(eligible,key=lambda r:(r["dev"]["adapted"]["top1"]-r["dev"]["baseline"]["top1"], r["dev"]["paired_improved"]-r["dev"]["paired_regressed"])) if eligible else None
    holdout_rows=[]; holdout=None; synthetic_pass=False
    if selected:
        holdout_rows,holdout=_synthetic_eval(models[selected["threshold"]],selected["threshold"],selected["scale"],selected["action_margin"],selected["use_loo_gate"],holdout_writers,labels,writers,splits,embeddings,logits)
        synthetic_pass=_positive(holdout) and all(row["adapted"]["top1"] >= row["baseline"]["top1"] and row["candidate_violations"] == 0 for row in holdout_rows)
    legacy_opened=synthetic_pass; legacy_rows=[]; legacy_states={}; legacy_splits={}; legacy_aggregate=None; regressions=[]
    if legacy_opened:
        legacy_splits,legacy_rows,legacy_states=_legacy(models[selected["threshold"]],selected["threshold"],selected["scale"],selected["action_margin"],selected["use_loo_gate"],args.v7,args.checkpoint)
        legacy_aggregate=_aggregate(legacy_rows); regressions=_writer_regressions(legacy_rows)
    legacy_pass=bool(legacy_opened and not regressions and legacy_aggregate["candidate_violations"]==0 and (
        legacy_aggregate["adapted"]["top1"]>legacy_aggregate["baseline"]["top1"] or legacy_aggregate["adapted"]["formula_exact"]>legacy_aggregate["baseline"]["formula_exact"]))
    status="WRITER_ADAPTER_V9_LEGACY_GATE_PASS" if legacy_pass else "PRE_ACCEPTANCE_FAIL_CLOSED"
    args.output.mkdir(parents=True)
    model_path=args.output/"candidate_meta_ranker.pt"
    if selected:
        torch.save({"state_dict":models[selected["threshold"]].state_dict(),"threshold":selected["threshold"],"scale":selected["scale"],"action_margin":selected["action_margin"],"use_loo_gate":selected["use_loo_gate"],"features":11},model_path)
    split_path=args.output/"split_manifest.json"; split_path.write_text(json.dumps({"seed":SEED,"fit_writers":fit_writers,"dev_writers":dev_writers,"holdout_writers":holdout_writers,"support_sizes":SUPPORT_SIZES,"legacy":legacy_splits if legacy_opened else "unopened","known_replay":"unopened","crohme":"unopened","mathwriting":"unopened"},ensure_ascii=False,indent=2),encoding="utf-8")
    report={"schema":"aiflow-writer-adaptation-meta-ranker/v9","generated_at":datetime.now(timezone.utc).isoformat(),"status":status,
            "selection":{"fit_losses":losses,"dev_grid":grid,"selected":{k:selected[k] for k in ("threshold","scale","action_margin","use_loo_gate")} if selected else None,"holdout":holdout,"holdout_per_writer":holdout_rows,
                         "primary_metric":"glyph Top1 and paired changes; pseudo-formula exact is diagnostic only"},
            "legacy":{"opened_after_synthetic_freeze":legacy_opened,"role":"known development validation; not promotion evidence","per_writer":legacy_rows,"aggregate":legacy_aggregate,"writer_regressions":regressions,"states":legacy_states},
            "gates":{"synthetic_nested_writer_disjoint":True,"synthetic_holdout_positive":synthetic_pass,"legacy_all_writer_nonregression":legacy_opened and not regressions,"legacy_aggregate_positive":legacy_pass,"candidate_violations_zero":legacy_pass,"known_replay_opened":False,"crohme_opened":False,"mathwriting_opened":False,"acceptance_opened":False,"product_promotion":False,"checkpoint_changed":False,"hwr_changed":False,"runtime_changed":False},
            "parent_overlap_audit":_parent_overlap(args.bank,fit_writers,dev_writers,holdout_writers),
            "inputs":{"bank_sha256":sha256(args.bank/"synthetic_writer_style_bank_v5_r4.npz"),"cache_sha256":sha256(args.cache),"checkpoint_sha256":sha256(args.checkpoint),"v7_report_sha256":sha256(args.v7/"report.json") if legacy_opened else None},
            "outputs":{"model_sha256":sha256(model_path) if model_path.exists() else None,"split_manifest_sha256":sha256(split_path)},
            "contracts":{"fit":"synthetic fit writers only","config_selection":"synthetic dev writers only","synthetic_holdout":"scoring only","legacy_eval_labels":"scoring only","candidate_set":"frozen baseline Top-5 exact set","writer_id_rules":False,"promotion_evidence":False}}
    (args.output/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"output":str(args.output),"status":status,"selected":report["selection"]["selected"],"holdout":holdout,"legacy_aggregate":legacy_aggregate,"regressions":regressions},ensure_ascii=False))
    return 0 if legacy_pass else 2


if __name__=="__main__": raise SystemExit(main())
