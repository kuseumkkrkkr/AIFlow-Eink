#!/usr/bin/env python3
"""범용 AIFlow 1.0e student 학습·예측 runner.

이 파일은 실제 연구 데이터의 split을 만들지 않는다. caller가 고정한 fit 및
selection ID를 받아 부모 data contract의 formula_batches/pack_formulas를 통해
학습하고, writer·formula leakage를 먼저 차단한다.
"""

from __future__ import annotations

import copy
import gc
import math
import random
from collections import defaultdict
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from accuracy_upgrade_data_10e import formula_batches, pack_formulas
from formula_candidate_loss_10e import formula_candidate_loss
from formula_context_ranker_10e import FormulaContextRanker
from train_online_candidate_distill_10e import OnlineCandidateRanker


SCHEMA = "aiflow-1.0e-accuracy-student-training/v1"
EFFECTIVE_BATCH = 32
DEFAULT_FORMULA_BATCH = 8
DEFAULT_MAX_EPOCHS = 12
DEFAULT_PATIENCE = 3
DEFAULT_TEMPERATURE = 2.0
DEFAULT_TEACHER_WEIGHT = 0.3


def _seed(seed: int) -> None:
    """retry마다 동일 초기화·formula shuffle seed를 재현한다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _writer(sample: dict) -> str:
    """부모 sample의 canonical writer 필드를 읽는다."""
    value = sample.get("writer_group", sample.get("writer"))
    if value is None:
        raise ValueError("student sample requires writer_group")
    return str(value)


def _formula_key(sample: dict) -> str:
    """수식 leakage 검사용 canonical formula_key를 읽는다."""
    value = sample.get("formula_key", sample.get("formula_id"))
    if value is None:
        raise ValueError("student sample requires formula_key")
    return str(value)


def _validate_split(samples: dict[str, dict], train_ids: list[str], validation_ids: list[str] | None) -> list[str]:
    """ID 중복과 train/selection writer·formula 겹침을 학습 전에 거부한다."""
    train_ids = [str(value) for value in train_ids]
    validation_ids = [str(value) for value in (validation_ids or [])]
    if not train_ids or len(set(train_ids)) != len(train_ids):
        raise ValueError("train_ids must be non-empty and unique")
    if len(set(validation_ids)) != len(validation_ids):
        raise ValueError("validation_ids must be unique")
    unknown = (set(train_ids) | set(validation_ids)) - samples.keys()
    if unknown:
        raise ValueError(f"unknown student sample IDs: {sorted(unknown)[:3]}")
    overlap_ids = set(train_ids) & set(validation_ids)
    if overlap_ids:
        raise ValueError("train/validation record overlap")
    train_writers = {_writer(samples[key]) for key in train_ids}
    validation_writers = {_writer(samples[key]) for key in validation_ids}
    train_formulas = {_formula_key(samples[key]) for key in train_ids}
    validation_formulas = {_formula_key(samples[key]) for key in validation_ids}
    if train_writers & validation_writers:
        raise ValueError("train/validation writer overlap")
    if train_formulas & validation_formulas:
        raise ValueError("train/validation formula overlap")
    return validation_ids


def _numeric_size(samples: dict[str, dict], ids: list[str], configured: int | None) -> int:
    """fit sample에서 feature 폭을 추론하되 token vocabulary는 추론하지 않는다."""
    widths = {int(np.asarray(samples[key]["numeric"]).shape[-1]) for key in ids}
    if len(widths) != 1 or not widths:
        raise ValueError("student numeric feature width is inconsistent")
    width = widths.pop()
    if configured is not None and int(configured) != width:
        raise ValueError(f"configured numeric_size {configured} != sample feature width {width}")
    if width not in (21, 28, 40):
        raise ValueError(f"student feature width must be inferred as 21, 28 or 40, got {width}")
    return width


def _validate_vocab(samples: dict[str, dict], ids: list[str], token_count: int) -> None:
    """고정 vocabulary 범위만 검사하며 train/evaluation token union을 만들지 않는다."""
    if token_count <= 0:
        raise ValueError("token_count must be positive")
    for key in ids:
        values = np.asarray(samples[key]["token_ids"], dtype=np.int64)
        if values.size and ((values < 0).any() or (values >= token_count).any()):
            raise ValueError(f"token ID outside fixed vocabulary: {key}")


class _FormulaMLPStudent(nn.Module):
    """기존 absolute OnlineCandidateRanker를 formula [B,L,K] API로 감싼다."""

    def __init__(self, numeric_size: int, token_count: int) -> None:
        super().__init__()
        self.ranker = OnlineCandidateRanker(numeric_size, token_count, hidden=128, token_width=48)
        self.coverage_head = nn.Sequential(
            nn.LayerNorm(128), nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 1)
        )
        self.architecture = "mlp"

    def _forward_parts(
        self,
        ink: torch.Tensor,
        numeric: torch.Tensor,
        token_ids: torch.Tensor,
        candidate_mask: torch.Tensor,
        row_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """padding을 안전한 row-wise 입력으로 바꾸고 기존 MLP score를 복원한다."""
        if ink.ndim != 3 or numeric.ndim != 4 or token_ids.ndim != 3:
            raise ValueError("MLP student expects ink[B,L,128], numeric[B,L,K,N], token_ids[B,L,K]")
        if numeric.shape[:3] != token_ids.shape or candidate_mask.shape != token_ids.shape:
            raise ValueError("MLP student candidate shapes do not match")
        if row_mask is None:
            row_mask = candidate_mask.any(dim=-1)
        if row_mask.shape != ink.shape[:2] or row_mask.dtype != torch.bool:
            raise ValueError("MLP row_mask shape or dtype mismatch")
        effective = candidate_mask & row_mask.unsqueeze(-1)
        token_count = self.ranker.token_embedding.num_embeddings
        active_ids = token_ids[effective]
        if active_ids.numel() and ((active_ids < 0) | (active_ids >= token_count)).any():
            raise ValueError("active token ID outside fixed vocabulary")
        safe_tokens = token_ids.clamp(0, token_count - 1).to(torch.long)
        safe_ink = torch.where(row_mask.unsqueeze(-1), ink, torch.zeros_like(ink))
        safe_numeric = torch.where(effective.unsqueeze(-1), numeric, torch.zeros_like(numeric))
        flat_scores = self.ranker(
            safe_ink.reshape(-1, safe_ink.shape[-1]),
            safe_numeric.reshape(-1, safe_numeric.shape[-2], safe_numeric.shape[-1]),
            safe_tokens.reshape(-1, safe_tokens.shape[-1]),
            effective.reshape(-1, effective.shape[-1]),
        )
        scores = flat_scores.reshape(token_ids.shape)
        coverage = self.coverage_head(safe_ink).squeeze(-1)
        coverage = torch.where(row_mask, coverage, torch.zeros_like(coverage))
        return scores, coverage

    def forward_with_coverage(
        self,
        ink: torch.Tensor,
        numeric: torch.Tensor,
        token_ids: torch.Tensor,
        candidate_mask: torch.Tensor,
        row_mask: torch.Tensor | None = None,
        *,
        base_logits: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """legacy MLP absolute score와 coverage logit을 반환한다; base는 의도적으로 무시한다."""
        del base_logits
        return self._forward_parts(ink, numeric, token_ids, candidate_mask, row_mask)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """context와 같은 forward interface를 제공한다."""
        scores, _coverage = self.forward_with_coverage(*args, **kwargs)
        return scores


def new_student(architecture: str, numeric_size: int, token_count: int = 372) -> nn.Module:
    """고정 vocabulary로 context 또는 legacy MLP student를 생성한다."""
    if architecture == "context":
        model = FormulaContextRanker(numeric_size, token_count)
    elif architecture == "mlp":
        model = _FormulaMLPStudent(numeric_size, token_count)
    else:
        raise ValueError("architecture must be 'context' or 'mlp'")
    model.architecture = architecture
    return model


def _forward_student(model: nn.Module, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """labels/targets를 넘기지 않고 순수 inference tensors만 model에 전달한다."""
    return model.forward_with_coverage(
        batch["ink"], batch["numeric"], batch["token_ids"], batch["candidate_mask"],
        row_mask=batch["row_mask"], base_logits=batch["base_logits"],
    )


def _zero_like(value: torch.Tensor) -> torch.Tensor:
    """inactive floor의 합 없이 finite differentiable zero를 만든다."""
    return torch.tanh(torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)).sum() * 0.0


def _formula_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """row loss를 formula별 평균 후 formula 평균으로 줄인다."""
    counts = valid.sum(dim=1)
    sums = torch.where(valid, values, torch.zeros_like(values)).sum(dim=1)
    means = sums / counts.clamp_min(1).to(values.dtype)
    if not bool((counts > 0).any()):
        return _zero_like(values)
    return means[counts > 0].mean()


def _legacy_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    candidate_mask: torch.Tensor,
    row_mask: torch.Tensor,
    teacher_scores: torch.Tensor | None,
    temperature: float,
    mode: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    """old 0.5 CE + 0.3 KD + 0.15 stability + 0.15 pairwise ablation을 유지한다."""
    formula_candidate_loss(
        scores, targets, candidate_mask, row_mask, teacher_scores=teacher_scores,
        temperature=temperature, kd_weight=0.0,
    )
    active = candidate_mask & row_mask.unsqueeze(-1)
    candidate_count = scores.shape[-1]
    valid = (targets >= 0) & row_mask
    if candidate_count == 0 or not bool(valid.any()):
        zero = _zero_like(scores)
        return zero, {"ce": 0.0, "kd": 0.0, "stability": 0.0, "pairwise": 0.0}
    safe_targets = targets.clamp(0, candidate_count - 1)
    target_present = active.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
    if bool((valid & ~target_present).any()):
        raise ValueError("legacy target is outside candidate_mask")
    safe = torch.where(active, scores, torch.zeros_like(scores)).masked_fill(
        ~active, torch.finfo(scores.dtype).min
    )
    log_probs = F.log_softmax(safe, dim=-1)
    ce_rows = -log_probs.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
    ce = _formula_mean(ce_rows, valid)

    if teacher_scores is None:
        kd = _zero_like(scores)
    else:
        safe_teacher = torch.where(active, teacher_scores, torch.zeros_like(teacher_scores)).masked_fill(
            ~active, torch.finfo(teacher_scores.dtype).min
        )
        student_log = F.log_softmax(safe / temperature, dim=-1)
        teacher_probability = F.softmax(safe_teacher / temperature, dim=-1)
        kd_rows = F.kl_div(student_log, teacher_probability, reduction="none").sum(dim=-1) * temperature**2
        if mode == "legacy_kl":
            kd_rows = kd_rows / max(candidate_count, 1)
        kd = _formula_mean(kd_rows, valid)

    other = safe.clone()
    other.scatter_(2, safe_targets.unsqueeze(-1), torch.finfo(scores.dtype).min)
    hardest = other.max(dim=-1).values
    target_score = scores.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
    stability_rows = F.relu(0.20 - target_score + hardest)
    baseline = scores[..., 0]
    stability = _formula_mean(stability_rows, valid & (targets == 0))
    pairwise_rows = F.relu(0.35 - target_score + baseline)
    pairwise = _formula_mean(pairwise_rows, valid & (targets != 0) & active[..., 0])
    loss = 0.50 * ce + 0.30 * kd + 0.15 * stability + 0.15 * pairwise
    if not torch.isfinite(loss):
        raise FloatingPointError("legacy student loss is non-finite")
    return loss, {
        "ce": float(ce.detach()), "kd": float(kd.detach()),
        "stability": float(stability.detach()), "pairwise": float(pairwise.detach()),
    }


def _loss_for_batch(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    teacher_weight: float,
    temperature: float,
    loss_mode: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    """batch forward 후 modern 또는 명시적 legacy objective를 선택한다."""
    scores, coverage = _forward_student(model, batch)
    teacher = batch.get("teacher_scores") if teacher_weight > 0 else None
    if loss_mode in ("legacy", "legacy_kl"):
        loss, metrics = _legacy_loss(
            scores, batch["targets"], batch["candidate_mask"], batch["row_mask"],
            teacher, temperature, loss_mode,
        )
        metrics["coverage"] = 0.0
        metrics["loss"] = float(loss.detach())
        return loss, metrics
    if loss_mode == "standard":
        loss_mode = "modern"
    if loss_mode != "modern":
        raise ValueError("loss_mode must be 'standard', 'modern', 'legacy', or 'legacy_kl'")
    loss, metrics = formula_candidate_loss(
        scores, batch["targets"], batch["candidate_mask"], batch["row_mask"],
        teacher_scores=teacher, temperature=temperature, kd_weight=teacher_weight,
        coverage_logits=coverage,
    )
    return loss, metrics


def _check_gradients(model: nn.Module) -> float:
    """모든 존재 gradient의 finite 여부와 clip norm을 검사한다."""
    for parameter in model.parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise FloatingPointError("non-finite student gradient")
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if not torch.isfinite(norm):
        raise FloatingPointError("non-finite clipped gradient norm")
    return float(norm.detach())


def _normalize_score(value) -> tuple[float, ...]:
    """callback의 comparable tuple 또는 scalar를 deterministic tuple로 정규화한다."""
    if isinstance(value, (tuple, list)):
        result = tuple(float(item) for item in value)
    else:
        result = (float(value),)
    if not result or not all(math.isfinite(item) for item in result):
        raise ValueError("validation_score must return a finite comparable tuple")
    return result


def _diagnostic_score(model: nn.Module, samples: dict[str, dict], ids: list[str], device: torch.device):
    """callback 부재 시 명시적인 glyph/formula exact diagnostic을 만든다."""
    rows = predict_student(model, samples, ids, device)
    by_formula: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_formula[str(row["formula_id"])].append(row)
    glyph_exact = sum(row["adapter_token"] == row["label"] for row in rows) / max(len(rows), 1)
    formula_exact = sum(
        all(row["adapter_token"] == row["label"] for row in formula_rows)
        for formula_rows in by_formula.values()
    ) / max(len(by_formula), 1)
    diagnostic = {
        "glyph_exact": float(glyph_exact),
        "formula_exact": float(formula_exact),
        "formula_count": len(by_formula),
        "row_count": len(rows),
    }
    return (diagnostic["formula_exact"], diagnostic["glyph_exact"]), diagnostic


def _is_oom(error: RuntimeError) -> bool:
    """CUDA와 CPU allocator가 사용하는 대표 OOM 메시지만 retry 대상으로 삼는다."""
    text = str(error).lower()
    return "out of memory" in text or "cuda error: out of memory" in text


def _fit_attempt(
    samples: dict[str, dict],
    train_ids: list[str],
    validation_ids: list[str],
    config: dict,
    device: torch.device,
    architecture: str,
    validation_score: Callable[[nn.Module], tuple] | None,
    numeric_size: int,
    token_count: int,
    formula_batch_size: int,
    teacher_enabled: bool,
    max_epochs: int,
    patience: int,
    seed: int,
    teacher_weight: float,
    temperature: float,
    loss_mode: str,
) -> tuple[nn.Module, dict]:
    """한 batch size의 전체 학습을 수행한다; OOM이면 caller가 같은 seed로 재시작한다."""
    _seed(seed)
    model = new_student(architecture, numeric_size, token_count).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    accumulation = max(1, math.ceil(EFFECTIVE_BATCH / formula_batch_size))
    losses = []
    best_state = None
    best_score: tuple[float, ...] | None = None
    best_epoch = 0
    stale = 0
    fixed_final = validation_score is None and not validation_ids
    for epoch in range(1, max_epochs + 1):
        model.train()
        totals = defaultdict(float)
        formula_count = 0
        gradient_norms = []
        batches = list(formula_batches(samples, train_ids, formula_batch_size, seed=seed + epoch - 1))
        if not batches:
            raise ValueError("no formula batches for training")
        for window_start in range(0, len(batches), accumulation):
            window = batches[window_start:window_start + accumulation]
            window_formula_count = sum(len(formulas) for formulas in window)
            if window_formula_count <= 0:
                raise ValueError("empty formula accumulation window")
            optimizer.zero_grad(set_to_none=True)
            for formulas in window:
                batch = pack_formulas(formulas, device)
                loss, metrics = _loss_for_batch(model, batch, teacher_weight, temperature, loss_mode)
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite student loss")
                # formula loss는 batch 내부 formula mean이므로 실제 formula 수로
                # window 평균을 만들고 마지막 partial window도 같은 분모를 쓴다.
                (loss * (len(formulas) / window_formula_count)).backward()
                for key, value in metrics.items():
                    totals[key] += float(value) * len(formulas)
                formula_count += len(formulas)
            gradient_norms.append(_check_gradients(model))
            optimizer.step()
        epoch_loss = {key: value / formula_count for key, value in totals.items()}
        epoch_loss.update({"epoch": epoch, "gradient_norm": sum(gradient_norms) / max(len(gradient_norms), 1)})

        diagnostic = None
        if fixed_final:
            score = ()
            selection_source = "fixed_final"
        elif validation_score is not None:
            model.eval()
            with torch.no_grad():
                score = _normalize_score(validation_score(model))
            selection_source = "callback"
        elif validation_ids:
            model.eval()
            with torch.no_grad():
                score, diagnostic = _diagnostic_score(model, samples, validation_ids, device)
            selection_source = "glyph_formula_exact"
        else:
            score = (-float(epoch_loss["loss"]),)
            selection_source = "fit_loss"
        epoch_loss["selection_score"] = list(score)
        epoch_loss["selection_source"] = selection_source
        if diagnostic is not None:
            epoch_loss["diagnostic"] = diagnostic
        losses.append(epoch_loss)
        if fixed_final:
            best_score = score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        elif best_score is None or score > best_score:
            best_score = score
            best_epoch = epoch
            stale = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None or best_score is None:
        raise RuntimeError("student training produced no selected model")
    model.load_state_dict(best_state)
    report = {
        "best_epoch": best_epoch,
        "best_selection_score": list(best_score),
        "losses": losses,
        "actual_batch": formula_batch_size,
        "actual_batch_size": formula_batch_size,
        "accumulation_steps": accumulation,
        "effective_batch": formula_batch_size * accumulation,
        "stopped_early": len(losses) < max_epochs,
        "selection_source": losses[-1]["selection_source"],
    }
    return model, report


def fit_student(
    samples: dict[str, dict],
    train_ids: list[str],
    config: dict,
    device: torch.device,
    architecture: str = "context",
    validation_ids: list[str] | None = None,
    validation_score: Callable[[nn.Module], tuple] | None = None,
) -> tuple[nn.Module, dict]:
    """고정 split의 student를 학습하고 최적 epoch report와 함께 반환한다."""
    if not isinstance(config, dict):
        raise ValueError("config must be a dict")
    validation_ids = _validate_split(samples, train_ids, validation_ids)
    all_ids = [str(value) for value in train_ids] + validation_ids
    numeric_size = _numeric_size(samples, all_ids, config.get("numeric_size"))
    token_count = int(config.get("token_count", 372))
    _validate_vocab(samples, all_ids, token_count)
    max_epochs = int(config.get(
        "max_epochs", config.get("max_epoch", config.get("maxepoch", config.get("epochs", DEFAULT_MAX_EPOCHS)))
    ))
    patience = int(config.get("patience", DEFAULT_PATIENCE))
    seed = int(config.get("seed", 0))
    teacher_weight = float(config.get(
        "teacher_weight", config.get("teacherweight", config.get("kd_weight", DEFAULT_TEACHER_WEIGHT))
    ))
    temperature = float(config.get("temperature", DEFAULT_TEMPERATURE))
    requested_loss_mode = str(config.get("loss_mode", "standard"))
    loss_mode = "modern" if requested_loss_mode == "standard" else requested_loss_mode
    if max_epochs <= 0 or patience <= 0:
        raise ValueError("max_epochs and patience must be positive")
    if teacher_weight < 0 or not math.isfinite(teacher_weight):
        raise ValueError("teacher_weight must be finite and non-negative")
    if temperature <= 0 or not math.isfinite(temperature):
        raise ValueError("temperature must be finite and positive")
    if validation_score is not None and not callable(validation_score):
        raise ValueError("validation_score must be callable")
    teacher_flags = ["teacher_scores" in samples[key] for key in train_ids]
    if any(teacher_flags) and not all(teacher_flags):
        raise ValueError("partial teacher coverage in fit IDs")
    teacher_enabled = all(teacher_flags)
    requested_batch = int(config.get("formula_batch_size", DEFAULT_FORMULA_BATCH))
    if requested_batch not in (8, 4, 2, 1):
        raise ValueError("formula_batch_size must be one of 8, 4, 2, 1")
    retry_sizes = [size for size in (8, 4, 2, 1) if size <= requested_batch]
    retry_report = []
    for formula_batch_size in retry_sizes:
        try:
            model, attempt = _fit_attempt(
                samples, [str(value) for value in train_ids], validation_ids, config, device,
                architecture, validation_score, numeric_size, token_count, formula_batch_size,
                teacher_enabled, max_epochs, patience, seed, teacher_weight if teacher_enabled else 0.0,
                temperature, loss_mode,
            )
            retry_report.append({"batch": formula_batch_size, "status": "success"})
            best_loss = attempt["losses"][attempt["best_epoch"] - 1]
            attempt.update({
                "schema": SCHEMA, "architecture": architecture,
                "fit_ids": [str(value) for value in train_ids], "selection_ids": validation_ids,
                "global_fit_ids": [str(value) for value in train_ids],
                "global_selection_ids": validation_ids,
                "numeric_size": numeric_size, "token_count": token_count,
                "teacher_used": teacher_enabled and teacher_weight > 0,
                "key": config.get("key"), "config_key": config.get("key"),
                "max_epochs": max_epochs, "maxepoch": max_epochs, "seed": seed,
                "teacher_weight": teacher_weight, "temperature": temperature,
                "loss_mode": requested_loss_mode, "objective_mode": loss_mode,
                "retry_report": retry_report,
                "validation_diagnostic": best_loss.get("diagnostic"),
                "optimizer": {"name": "AdamW", "lr": 3e-4, "weight_decay": 1e-3, "clip_norm": 1.0},
            })
            return model, attempt
        except RuntimeError as error:
            if not _is_oom(error) or formula_batch_size == retry_sizes[-1]:
                raise
            retry_report.append({"batch": formula_batch_size, "status": "oom", "error": str(error)})
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    raise RuntimeError("all student batch-size retries failed")


def _softmax(values: np.ndarray) -> np.ndarray:
    """후보 축의 finite 확률을 계산한다."""
    shifted = values - np.max(values)
    probabilities = np.exp(shifted)
    return probabilities / max(float(probabilities.sum()), 1e-12)


@torch.inference_mode()
def predict_student(
    model: nn.Module,
    samples: dict[str, dict],
    ids: list[str],
    device: torch.device,
) -> list[dict]:
    """targets/labels를 forward에 전달하지 않고 prediction report row만 생성한다."""
    ids = [str(value) for value in ids]
    if set(ids) - samples.keys():
        raise ValueError("prediction IDs are missing from samples")
    was_training = model.training
    model.eval()
    by_record: dict[str, dict] = {}
    for formulas in formula_batches(samples, ids, DEFAULT_FORMULA_BATCH, seed=None):
        batch = pack_formulas(formulas, device)
        scores, coverage = _forward_student(model, batch)
        for formula_index, formula_rows in enumerate(formulas):
            for row_index, sample in enumerate(formula_rows):
                width = len(sample["candidates"])
                if width <= 0:
                    raise ValueError("student prediction requires at least one candidate")
                adapter_scores = scores[formula_index, row_index, :width].detach().cpu().numpy().astype(np.float64)
                baseline_logits = np.asarray(sample.get("base_logits", []), dtype=np.float64)
                if baseline_logits.shape != (width,) or not np.isfinite(baseline_logits).all():
                    raise ValueError(f"invalid base_logits for prediction row {sample['record_id']}")
                baseline_probability = _softmax(baseline_logits)
                adapter_probability = _softmax(adapter_scores)
                adapter_index = int(np.argmax(adapter_scores))
                sorted_scores = np.sort(adapter_scores)[::-1]
                margin = float(sorted_scores[0] - sorted_scores[1]) if width > 1 else 0.0
                entropy = float(-(adapter_probability * np.log(np.clip(adapter_probability, 1e-12, 1.0))).sum())
                target = int(sample.get("target", -1))
                row = {
                    "record_id": str(sample["record_id"]), "formula_id": str(sample["formula_id"]),
                    "writer_group": _writer(sample), "label": sample.get("label"),
                    "candidates": [str(value) for value in sample["candidates"]],
                    "baseline_token": str(sample["candidates"][0]),
                    "adapter_token": str(sample["candidates"][adapter_index]),
                    "adapter_scores": [float(value) for value in adapter_scores],
                    "target": target, "target_in_candidates": 0 <= target < width,
                    "coverage_probability": float(torch.sigmoid(coverage[formula_index, row_index]).cpu()),
                    "baseline_probabilities": [float(value) for value in baseline_probability],
                    "raw_margin": margin, "raw_entropy": entropy,
                    "guard": {"raw_margin": margin, "raw_entropy": entropy},
                }
                by_record[row["record_id"]] = row
    if was_training:
        model.train()
    return [by_record[str(key)] for key in ids]


__all__ = ("fit_student", "new_student", "predict_student")
