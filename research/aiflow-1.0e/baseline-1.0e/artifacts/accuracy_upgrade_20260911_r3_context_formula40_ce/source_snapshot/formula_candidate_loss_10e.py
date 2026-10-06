#!/usr/bin/env python3
"""Formula-level candidate CE/KD/coverage loss for AIFlow 1.0e."""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F


def _zero_like(value: torch.Tensor) -> torch.Tensor:
    """masked floor의 합을 사용하지 않고 항상 finite인 미분가능 zero를 만든다."""
    return torch.tanh(torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)).sum() * 0.0


def _formula_mean(values: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, int]:
    """row 값을 formula별로 먼저 평균낸 뒤 formula 평균을 계산한다."""
    counts = valid.sum(dim=1)
    row_sum = torch.where(valid, values, torch.zeros_like(values)).sum(dim=1)
    row_mean = row_sum / counts.clamp_min(1).to(values.dtype)
    formula_valid = counts > 0
    if not bool(formula_valid.any()):
        return _zero_like(values), 0
    return row_mean[formula_valid].mean(), int(formula_valid.sum().item())


def _check_shape_and_finite(
    scores: torch.Tensor,
    targets: torch.Tensor,
    candidate_mask: torch.Tensor,
    row_mask: torch.Tensor,
    teacher_scores: torch.Tensor | None,
    coverage_logits: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """loss 입력의 shape/dtype과 active 값의 finite 계약을 검사한다."""
    if scores.ndim != 3 or targets.ndim != 2 or candidate_mask.ndim != 3 or row_mask.ndim != 2:
        raise ValueError("expected scores[B,L,K], targets[B,L], candidate_mask[B,L,K], row_mask[B,L]")
    if targets.shape != scores.shape[:2] or candidate_mask.shape != scores.shape:
        raise ValueError("scores, targets, and candidate_mask shapes do not match")
    if row_mask.shape != scores.shape[:2]:
        raise ValueError("row_mask shape does not match scores")
    if candidate_mask.dtype != torch.bool or row_mask.dtype != torch.bool:
        raise ValueError("candidate_mask and row_mask must be bool")
    if targets.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError("targets must use an integer dtype")
    if teacher_scores is not None and teacher_scores.shape != scores.shape:
        raise ValueError("teacher_scores shape does not match scores")
    if coverage_logits is not None and coverage_logits.shape != scores.shape[:2]:
        raise ValueError("coverage_logits shape does not match scores")
    active_candidates = candidate_mask & row_mask.unsqueeze(-1)
    if scores[active_candidates].numel() and not torch.isfinite(scores[active_candidates]).all():
        raise ValueError("scores contains non-finite active values")
    if teacher_scores is not None:
        if teacher_scores[active_candidates].numel() and not torch.isfinite(teacher_scores[active_candidates]).all():
            raise ValueError("teacher_scores contains non-finite active values")
    if coverage_logits is not None:
        active_rows = coverage_logits[row_mask]
        if active_rows.numel() and not torch.isfinite(active_rows).all():
            raise ValueError("coverage_logits contains non-finite active values")
    return active_candidates, targets.to(torch.long)


def formula_candidate_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    candidate_mask: torch.Tensor,
    row_mask: torch.Tensor,
    teacher_scores: torch.Tensor | None = None,
    temperature: float = 2.0,
    kd_weight: float = 0.0,
    coverage_logits: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """formula 평균 CE/KD와 row coverage BCE를 계산한다.

    targets의 -1은 후보 밖 정답이며 CE/KD에서는 제외하고 coverage BCE에서는
    negative target으로 유지한다. KD는 유효 후보 축을 합산한 뒤 row 평균,
    formula 평균 순서로 줄인다.
    """
    if not scores.is_floating_point():
        raise ValueError("scores must be floating point")
    if temperature <= 0 or not torch.isfinite(torch.tensor(float(temperature))):
        raise ValueError("temperature must be finite and positive")
    if kd_weight < 0 or not torch.isfinite(torch.tensor(float(kd_weight))):
        raise ValueError("kd_weight must be finite and non-negative")
    if kd_weight > 0 and teacher_scores is None:
        raise ValueError("teacher_scores is required when kd_weight is positive")
    active_candidates, targets = _check_shape_and_finite(
        scores, targets, candidate_mask, row_mask, teacher_scores, coverage_logits
    )
    batch_size, _row_count, candidate_count = scores.shape
    active_targets = targets[row_mask]
    if bool(((active_targets < -1) | (active_targets >= candidate_count)).any()):
        raise ValueError("targets must be -1 or a candidate index")

    valid_target = (targets >= 0) & row_mask
    if bool(valid_target.any()):
        if not candidate_count:
            raise ValueError("non-missing target has no candidate axis")
        safe_targets = targets.clamp(0, candidate_count - 1)
        target_in_candidates = active_candidates.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
        if bool((valid_target & ~target_in_candidates).any()):
            raise ValueError("non-missing target is outside candidate_mask")
    elif candidate_count:
        safe_targets = targets.clamp(0, candidate_count - 1)

    if candidate_count and bool(valid_target.any()):
        safe_scores = torch.where(active_candidates, scores, torch.zeros_like(scores))
        safe_scores = safe_scores.masked_fill(~active_candidates, torch.finfo(scores.dtype).min)
        log_probs = F.log_softmax(safe_scores, dim=-1)
        ce_rows = -log_probs.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
        ce, ce_formulas = _formula_mean(ce_rows, valid_target)
    else:
        ce = _zero_like(scores)
        ce_formulas = 0

    if teacher_scores is not None and candidate_count and bool(valid_target.any()):
        safe_student = torch.where(active_candidates, scores, torch.zeros_like(scores))
        safe_teacher = torch.where(active_candidates, teacher_scores, torch.zeros_like(teacher_scores))
        safe_student = safe_student.masked_fill(~active_candidates, torch.finfo(scores.dtype).min)
        safe_teacher = safe_teacher.masked_fill(~active_candidates, torch.finfo(teacher_scores.dtype).min)
        student_log_probs = F.log_softmax(safe_student / temperature, dim=-1)
        teacher_probs = F.softmax(safe_teacher / temperature, dim=-1)
        kd_rows = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
        kd_rows = kd_rows * (temperature**2)
        kd, kd_formulas = _formula_mean(kd_rows, valid_target)
    else:
        kd = _zero_like(scores)
        kd_formulas = 0

    if coverage_logits is not None:
        coverage_valid = row_mask
        coverage_targets = (targets >= 0).to(coverage_logits.dtype)
        safe_coverage_logits = torch.where(row_mask, coverage_logits, torch.zeros_like(coverage_logits))
        coverage_rows = F.binary_cross_entropy_with_logits(
            safe_coverage_logits, coverage_targets, reduction="none"
        )
        coverage, coverage_formulas = _formula_mean(coverage_rows, coverage_valid)
    else:
        coverage = _zero_like(scores)
        coverage_formulas = 0

    loss = ce + float(kd_weight) * kd + coverage
    if not torch.isfinite(loss):
        raise FloatingPointError("formula candidate loss is non-finite")
    metrics = {
        "loss": float(loss.detach()),
        "ce": float(ce.detach()),
        "kd": float(kd.detach()),
        "coverage": float(coverage.detach()),
        "coverage_bce": float(coverage.detach()),
        "ce_formulas": float(ce_formulas),
        "kd_formulas": float(kd_formulas),
        "coverage_formulas": float(coverage_formulas),
        "valid_rows": float(valid_target.sum().item()),
        "coverage_rows": float(row_mask.sum().item()) if coverage_logits is not None else 0.0,
        "batch_size": float(batch_size),
    }
    return loss, metrics


__all__: Iterable[str] = ("formula_candidate_loss",)
