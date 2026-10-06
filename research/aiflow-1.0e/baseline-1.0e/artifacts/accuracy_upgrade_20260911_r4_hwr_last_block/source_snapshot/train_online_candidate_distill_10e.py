#!/usr/bin/env python3
"""Raster-free online candidate ranker distilled from an external teacher.

The external OCR scores are consumed only while training the small student.
At inference the student uses the existing 5-channel ordered ink embedding,
the frozen HWR candidate probabilities, and geometry/context features.  It
cannot create tokens or modify stroke grouping.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from character_tensor_v1 import tensorize
from train_character_classifier_v1 import InkClassifierV1, apply_input_mode
from online_candidate_features_10e import _json_lines, _load_candidates, _load_formula_records, _row_numeric
from accuracy_upgrade_contract_v1 import masked_candidate_kl, writer_key


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = Path(
    r"D:\AIFlow-Workspace\PrivateData\math-ink-data-collector\derived"
    r"\hwr-head-calibration-20260819-r1\project_owned_ownership_eval_95.jsonl.gz"
)
DEFAULT_CANDIDATES = ROOT / "artifacts" / "ocr_trocr_hwr95_candidates_20260901_r1" / "candidates.jsonl.gz"
DEFAULT_TEACHER = ROOT / "artifacts" / "ocr_trocr_weight_tuned_20260901_r1" / "writer_loo_predictions.jsonl.gz"
DEFAULT_HWR = ROOT / "artifacts" / "commercial_hwr_stability_20260901_r1_shadow" / "commercial_hwr_stability_checkpoint.pt"
DEFAULT_OUTPUT = ROOT / "artifacts" / "online_candidate_distill_10e_20260902_r1"
SCHEMA = "aiflow-1.0e-online-candidate-distill/v1"


def _read_gz(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class OnlineCandidateRanker(nn.Module):
    """Scores existing candidates from ordered-ink evidence only."""

    def __init__(self, numeric_size: int, token_count: int, hidden: int = 128, token_width: int = 48) -> None:
        super().__init__()
        self.ink_projection = nn.Sequential(nn.Linear(128, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.numeric_projection = nn.Sequential(nn.Linear(numeric_size, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.token_embedding = nn.Embedding(token_count, token_width)
        self.fusion = nn.Sequential(
            nn.Linear(hidden + hidden + token_width, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1),
        )

    def forward(
        self,
        ink: torch.Tensor,
        numeric: torch.Tensor,
        token_ids: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        evidence = self.ink_projection(ink).unsqueeze(1).expand(-1, numeric.shape[1], -1)
        candidate = self.numeric_projection(numeric)
        token = self.token_embedding(token_ids)
        scores = self.fusion(torch.cat((candidate, evidence, token), dim=-1)).squeeze(-1)
        return scores.masked_fill(~mask, torch.finfo(scores.dtype).min)


def _make_samples(
    candidates: list[dict],
    raw_by_id: dict[str, dict],
    teacher_by_id: dict[str, dict],
    hwr_embeddings: dict[str, np.ndarray],
    token_to_id: dict[str, int],
) -> list[dict]:
    samples = []
    for row in candidates:
        record_id = str(row["record_id"])
        teacher = teacher_by_id.get(record_id)
        if teacher is None:
            raise ValueError(f"teacher prediction missing record: {record_id}")
        candidate_tokens = [str(value) for value in row["final_topk"]]
        if candidate_tokens != [str(value) for value in teacher["candidates"]]:
            raise ValueError(f"candidate mismatch between cache and teacher: {record_id}")
        if record_id not in raw_by_id or record_id not in hwr_embeddings:
            raise ValueError(f"online ink missing record: {record_id}")
        label = str(row["label"])
        samples.append({
            "record_id": record_id,
            "formula_id": str(row["formula_id"]),
            # 후보 cache의 writer id와 raw ink의 writer id가 다를 수 있다.
            # outer fold는 반드시 raw lineage를 사용해야 held writer leakage를 막는다.
            "writer_group": writer_key(raw_by_id[record_id]),
            "candidate_writer_group": str(row["writer_group"]),
            "label": label,
            "candidates": candidate_tokens,
            "numeric": np.asarray([_row_numeric(row, index) for index in range(len(candidate_tokens))], dtype=np.float32),
            "token_ids": np.asarray([token_to_id[token] for token in candidate_tokens], dtype=np.int64),
            "ink": np.asarray(hwr_embeddings[record_id], dtype=np.float32),
            "teacher_scores": np.asarray(teacher["adapter_scores"], dtype=np.float32),
            "teacher_provenance": teacher.get("prediction_provenance") or {},
            "target": candidate_tokens.index(label) if label in candidate_tokens else -1,
        })
    return samples


def _pack(samples: list[dict], device: torch.device) -> tuple[torch.Tensor, ...]:
    width = max(len(sample["candidates"]) for sample in samples)
    numeric_size = samples[0]["numeric"].shape[-1]
    numeric = np.zeros((len(samples), width, numeric_size), dtype=np.float32)
    token_ids = np.zeros((len(samples), width), dtype=np.int64)
    teacher_scores = np.full((len(samples), width), -1e9, dtype=np.float32)
    mask = np.zeros((len(samples), width), dtype=bool)
    ink = np.stack([sample["ink"] for sample in samples]).astype(np.float32)
    targets = np.full(len(samples), -100, dtype=np.int64)
    for row_index, sample in enumerate(samples):
        size = len(sample["candidates"])
        numeric[row_index, :size] = sample["numeric"]
        token_ids[row_index, :size] = sample["token_ids"]
        teacher_scores[row_index, :size] = sample["teacher_scores"]
        mask[row_index, :size] = True
        targets[row_index] = int(sample["target"])
    return (
        torch.from_numpy(ink).to(device), torch.from_numpy(numeric).to(device),
        torch.from_numpy(token_ids).to(device), torch.from_numpy(mask).to(device),
        torch.from_numpy(teacher_scores).to(device), torch.from_numpy(targets).to(device),
    )


def _balanced_batches(samples: list[dict], steps: int, batch_size: int, seed: int) -> list[list[dict]]:
    """수식 단위로 샘플링하되 후보 밖 정답 row도 분모에서 유지한다."""
    if not samples:
        return []
    by_formula: dict[str, list[dict]] = defaultdict(list)
    for sample in samples:
        by_formula[str(sample["formula_id"])].append(sample)
    formulas = sorted(by_formula)
    rng = random.Random(seed)
    return [[rng.choice(by_formula[rng.choice(formulas)]) for _ in range(batch_size)] for _ in range(steps)]


def _loss(scores: torch.Tensor, teacher_scores: torch.Tensor, targets: torch.Tensor, temperature: float, candidate_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, float]]:
    """legacy 비교 손실도 실제 후보만 계산하여 padding overflow를 막는다."""
    if candidate_mask is None:
        candidate_mask = teacher_scores > -1e8  # 이전 호출자 호환; 신규 경로는 명시적 mask 필수.
    if candidate_mask.shape != scores.shape or candidate_mask.dtype != torch.bool:
        raise ValueError("invalid candidate mask")
    if not torch.isfinite(scores[candidate_mask]).all() or not torch.isfinite(teacher_scores[candidate_mask]).all():
        raise ValueError("valid scores must be finite")
    valid = targets >= 0
    if not bool(valid.any()):
        return scores[candidate_mask][:0].sum(), {"ce": 0.0, "kl": 0.0, "stability": 0.0}
    row_mask = candidate_mask[valid]
    score_rows = scores[valid].masked_fill(~row_mask, torch.finfo(scores.dtype).min)
    teacher_rows = teacher_scores[valid]
    target_rows = targets[valid]
    if (target_rows >= scores.shape[-1]).any() or not row_mask.gather(1, target_rows[:, None]).all():
        raise ValueError("target must address a valid candidate")
    ce = nn.functional.cross_entropy(score_rows, target_rows)
    kl_values = []
    stability_values = []
    pair_values = []
    for index, target in enumerate(target_rows.tolist()):
        others = row_mask[index].clone()
        others[target] = False
        if not bool(others.any()):
            continue
        hardest = score_rows[index][others].max()
        if target == 0:
            stability_values.append(nn.functional.relu(0.20 - score_rows[index, 0] + hardest))
        else:
            pair_values.append(nn.functional.relu(0.35 - score_rows[index, target] + score_rows[index, 0]))
    # 후보 축을 batchmean으로 나누면 K에 따라 KL 신호가 축소된다.
    # padding을 제외한 후보 축 합산 후 유효 row 평균을 사용한다.
    kl = masked_candidate_kl(score_rows, teacher_rows, row_mask, temperature)
    zero = score_rows[row_mask][:0].sum()
    stability = torch.stack(stability_values).mean() if stability_values else zero
    pairwise = torch.stack(pair_values).mean() if pair_values else zero
    loss = 0.50 * ce + 0.30 * kl + 0.15 * stability + 0.15 * pairwise
    return loss, {"ce": float(ce.detach()), "kl": float(kl.detach()), "stability": float((stability + pairwise).detach())}


def _train(
    model: OnlineCandidateRanker,
    samples: list[dict],
    epochs: int,
    seed: int,
    device: torch.device,
    learning_rate: float,
    temperature: float,
) -> dict[str, float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    eligible = [sample for sample in samples if int(sample["target"]) >= 0]
    steps = max(1, math.ceil(len(eligible) / 32))
    totals = {"ce": 0.0, "kl": 0.0, "stability": 0.0}
    count = 0
    model.train()
    for epoch in range(epochs):
        for batch in _balanced_batches(samples, steps, 32, seed + epoch):
            ink, numeric, token_ids, mask, teacher_scores, targets = _pack(batch, device)
            scores = model(ink, numeric, token_ids, mask)
            loss, parts = _loss(scores, teacher_scores, targets, temperature, mask)
            if not bool((targets >= 0).any()):
                continue
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite legacy loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            for name in totals:
                totals[name] += parts[name]
            count += 1
    model.eval()
    return {name: value / max(1, count) for name, value in totals.items()}


@torch.inference_mode()
def _predict(
    model: OnlineCandidateRanker,
    samples: list[dict],
    device: torch.device,
    *,
    excluded_writers: list[str] | None = None,
) -> list[dict]:
    output = []
    for start in range(0, len(samples), 64):
        selected = samples[start:start + 64]
        ink, numeric, token_ids, mask, _, _ = _pack(selected, device)
        scores = model(ink, numeric, token_ids, mask).cpu().numpy()
        for sample, row_scores in zip(selected, scores):
            row_scores = row_scores[:len(sample["candidates"])]
            index = int(np.argmax(row_scores))
            output.append({
                "record_id": sample["record_id"], "formula_id": sample["formula_id"],
                "writer_group": sample["writer_group"], "label": sample["label"],
                "candidates": sample["candidates"], "baseline_token": sample["candidates"][0],
                "adapter_token": sample["candidates"][index],
                "target_in_candidates": sample["target"] >= 0,
                "adapter_scores": [float(value) for value in row_scores],
                "prediction_provenance": {
                    "schema": "aiflow-1.0e-prediction-provenance/v1",
                    "training_scope": "writer_loo",
                    "excluded_writers": sorted(str(value) for value in (excluded_writers or [])),
                },
            })
    return output


def _metrics(rows: list[dict], prediction_key: str, baseline_key: str = "baseline_token") -> dict:
    correct = [str(row[prediction_key]) == str(row["label"]) for row in rows]
    baseline = [str(row[baseline_key]) == str(row["label"]) for row in rows]
    groups: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for row, value in zip(rows, correct):
        groups[(str(row["writer_group"]), str(row["formula_id"]))].append(value)
    exact = sum(all(values) for values in groups.values())
    return {
        "rows": len(rows), "top1": sum(correct) / len(rows) if rows else 0.0,
        "top1_correct": sum(correct), "formula_exact": exact / len(groups) if groups else 0.0,
        "formula_exact_correct": exact, "formula_total": len(groups),
        "row_level_improvements": sum(not old and new for old, new in zip(baseline, correct)),
        "row_level_regressions": sum(old and not new for old, new in zip(baseline, correct)),
        "changed_rows": sum(str(row[prediction_key]) != str(row[baseline_key]) for row in rows),
    }


def _write_gz(path: Path, rows: list[dict]) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as stream:
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
                    stream.write("\n")


def _load_hwr_embeddings(
    raw_by_id: dict[str, dict],
    checkpoint_path: Path,
    device: torch.device,
    input_mode: str = "uniform-time",
) -> tuple[dict[str, np.ndarray], str, list[str]]:
    """검증된 HWR encoder를 사용해 online-ink embedding을 만든다."""
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    labels = list(payload["math_labels"])
    if payload.get("auxiliary_labels") != [] or len(labels) != 372:
        raise ValueError("unexpected HWR checkpoint contract")
    model = InkClassifierV1(len(labels), 0).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    embeddings: dict[str, np.ndarray] = {}
    ids = sorted(raw_by_id)
    for start in range(0, len(ids), 64):
        batch = np.stack([apply_input_mode(tensorize(raw_by_id[record_id]), input_mode) for record_id in ids[start:start + 64]])
        encoded = model.encode(torch.from_numpy(batch).to(device)).detach().cpu().numpy().astype(np.float32)
        embeddings.update({record_id: encoded[index] for index, record_id in enumerate(ids[start:start + 64])})
    return embeddings, _sha256(checkpoint_path), labels


def _load_hwr_embeddings_by_held_writer(
    raw_by_id: dict[str, dict],
    checkpoint_path: Path,
    device: torch.device,
    input_mode: str = "uniform-time",
) -> tuple[dict[str, dict[str, np.ndarray]], str, list[str]]:
    """각 outer held writer를 제외해 학습한 HWR state로 embedding을 만든다."""
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    labels = list(payload["math_labels"])
    states = payload.get("states_by_held_writer") or {}
    if payload.get("auxiliary_labels") not in (None, []):
        raise ValueError("unexpected auxiliary labels in HWR LOO checkpoint")
    if len(labels) != 372 or not states:
        raise ValueError("invalid HWR LOO checkpoint contract")
    ids = sorted(raw_by_id)
    embeddings_by_writer: dict[str, dict[str, np.ndarray]] = {}
    for held_writer, state in sorted(states.items()):
        model = InkClassifierV1(len(labels), 0).to(device)
        model.load_state_dict(state, strict=True)
        model.eval()
        embeddings: dict[str, np.ndarray] = {}
        for start in range(0, len(ids), 64):
            selected = ids[start:start + 64]
            batch = np.stack([apply_input_mode(tensorize(raw_by_id[record_id]), input_mode) for record_id in selected])
            encoded = model.encode(torch.from_numpy(batch).to(device)).detach().cpu().numpy().astype(np.float32)
            embeddings.update({record_id: encoded[index] for index, record_id in enumerate(selected)})
        embeddings_by_writer[str(held_writer)] = embeddings
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return embeddings_by_writer, _sha256(checkpoint_path), labels


def _assert_nested_teacher_exclusion(samples: list[dict], outer_writer: str) -> None:
    """outer writer를 포함한 teacher target 누수를 거부한다."""
    from accuracy_lineage_10e import validate_prediction
    for sample in samples:
        validate_prediction({**sample, "prediction_provenance": sample.get("teacher_provenance") or {}}, outer_writer)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--hwr-checkpoint", type=Path, default=DEFAULT_HWR)
    parser.add_argument("--hwr-loo-checkpoint", type=Path, help="writer-LOO HWR state bundle for strict outer evaluation")
    parser.add_argument("--strict-lineage", action="store_true", help="require writer-LOO HWR states instead of a final all-writer encoder")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--input-mode", choices=("preserve", "uniform-time"), default="uniform-time")
    args = parser.parse_args()
    if args.strict_lineage or args.hwr_loo_checkpoint is not None:
        parser.error("legacy training cannot verify nested ancestors or refit pairing; use run_accuracy_upgrade_10e.py")
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    _set_seed(args.seed)
    device = torch.device(args.device)
    raw_rows = _read_gz(args.raw)
    raw_by_id = {str(row["record_id"]): row for row in raw_rows}
    if len(raw_by_id) != len(raw_rows):
        raise ValueError("raw record IDs must be unique")
    candidates = _load_candidates(args.candidates, None)
    teacher_rows = _read_gz(args.teacher)
    teacher_by_id = {str(row["record_id"]): row for row in teacher_rows}
    if len(teacher_by_id) != len(teacher_rows):
        raise ValueError("teacher record IDs must be unique")
    if args.strict_lineage and args.hwr_loo_checkpoint is None:
        parser.error("--strict-lineage requires --hwr-loo-checkpoint")
    hwr_embeddings_by_writer = None
    if args.hwr_loo_checkpoint is not None:
        hwr_embeddings_by_writer, hwr_sha, hwr_labels = _load_hwr_embeddings_by_held_writer(
            raw_by_id, args.hwr_loo_checkpoint, device, args.input_mode
        )
        hwr_embeddings = next(iter(hwr_embeddings_by_writer.values()))
    else:
        hwr_embeddings, hwr_sha, hwr_labels = _load_hwr_embeddings(raw_by_id, args.hwr_checkpoint, device, args.input_mode)
    labels = list(hwr_labels)
    token_to_id = {token: index for index, token in enumerate(labels)}
    samples = _make_samples(candidates, raw_by_id, teacher_by_id, hwr_embeddings, token_to_id)
    by_writer: defaultdict[str, list[dict]] = defaultdict(list)
    for sample in samples:
        by_writer[sample["writer_group"]].append(sample)
    print(json.dumps({"event": "online_distill_start", "schema": SCHEMA, "record_count": len(samples), "formula_count": len({s['formula_id'] for s in samples}), "writer_count": len(by_writer), "device": str(device), "teacher": str(args.teacher)}, ensure_ascii=False), flush=True)

    all_predictions: list[dict] = []
    folds = []
    loss_reports = []
    numeric_size = samples[0]["numeric"].shape[-1]
    for fold_index, writer in enumerate(sorted(by_writer)):
        train = [sample for group, values in by_writer.items() if group != writer for sample in values]
        test = list(by_writer[writer])
        if hwr_embeddings_by_writer is not None:
            if writer not in hwr_embeddings_by_writer:
                raise ValueError(f"HWR LOO checkpoint has no state for held writer {writer}")
            fold_embeddings = hwr_embeddings_by_writer[writer]
            train = [{**sample, "ink": np.asarray(fold_embeddings[sample["record_id"]], dtype=np.float32)} for sample in train]
            test = [{**sample, "ink": np.asarray(fold_embeddings[sample["record_id"]], dtype=np.float32)} for sample in test]
        if args.strict_lineage:
            _assert_nested_teacher_exclusion(train, writer)
        model = OnlineCandidateRanker(numeric_size, len(labels)).to(device)
        loss_report = _train(model, train, args.epochs, args.seed + fold_index, device, args.learning_rate, args.temperature)
        loss_report["held_writer"] = writer
        loss_reports.append(loss_report)
        predictions = _predict(model, test, device, excluded_writers=[writer])
        all_predictions.extend(predictions)
        folds.append({"held_writer": writer, **_metrics(predictions, "adapter_token")})
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    refit = OnlineCandidateRanker(numeric_size, len(labels)).to(device)
    refit_loss = _train(refit, samples, args.epochs, args.seed + 1000, device, args.learning_rate, args.temperature)
    args.output.mkdir(parents=True)
    checkpoint = args.output / "online_candidate_ranker.pt"
    torch.save({
        "schema": SCHEMA, "labels": labels, "numeric_size": numeric_size,
        "hwr_checkpoint_sha256": hwr_sha, "state_dict": refit.state_dict(),
        "runtime_inputs": ["5-channel ordered online ink", "frozen HWR candidate probabilities", "formula-coordinate geometry/context", "candidate token identity"],
        "input_mode": args.input_mode,
        "hwr_embedding_scope": "writer_loo" if hwr_embeddings_by_writer is not None else "legacy_all_writer_final",
        "external_teacher_in_runtime": False,
        "strict_lineage": bool(args.strict_lineage),
        "candidate_contract": {"top_k_only": True, "token_creation": False, "row_deletion": False, "stroke_regrouping": False, "relation_mutation": False},
    }, checkpoint)
    baseline = _metrics(all_predictions, "baseline_token")
    adapter = _metrics(all_predictions, "adapter_token")
    evaluation = {
        "schema": SCHEMA, "status": "legacy_development", "training_data_boundary": "project-owned candidate cache; external teacher scores used as soft targets only",
        "external_teacher_in_runtime": False, "hwr_checkpoint_sha256": hwr_sha,
        "hwr_embedding_scope": "writer_loo" if hwr_embeddings_by_writer is not None else "legacy_all_writer_final",
        "strict_lineage": bool(args.strict_lineage),
        "data": {"raw": str(args.raw), "candidates": str(args.candidates), "teacher": str(args.teacher), "records": len(samples), "formulas": len({s['formula_id'] for s in samples}), "writers": len(by_writer)},
        "training": {"epochs": args.epochs, "seed": args.seed, "learning_rate": args.learning_rate, "temperature": args.temperature, "loss": "0.50 CE + 0.30 candidate-axis-summed teacher KL + baseline stability/pairwise margin", "numeric_size": numeric_size, "hwr_embedding_size": 128, "student_parameters": sum(parameter.numel() for parameter in refit.parameters()), "refit_loss": refit_loss},
        "baseline": baseline, "adapter": adapter, "candidate_recall": sum(bool(row["target_in_candidates"]) for row in all_predictions) / len(all_predictions),
        "writer_loo": folds, "loss_reports": loss_reports, "checkpoint": str(checkpoint),
        "adoption": {"row_level_regression_gate": adapter["row_level_regressions"] == 0, "fresh_acceptance_required": True, "product_runtime_changed": False},
    }
    (args.output / "evaluation.json").write_text(json.dumps(evaluation, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_gz(args.output / "writer_loo_predictions.jsonl.gz", all_predictions)
    print(json.dumps({"event": "complete", "output": str(args.output), "baseline": baseline, "adapter": adapter, "candidate_recall": evaluation["candidate_recall"]}, ensure_ascii=False), flush=True)
    return 0


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    raise SystemExit(main())
