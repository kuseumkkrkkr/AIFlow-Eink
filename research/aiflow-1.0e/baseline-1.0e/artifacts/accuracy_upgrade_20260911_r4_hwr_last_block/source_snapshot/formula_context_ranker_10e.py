#!/usr/bin/env python3
"""Online-only formula context ranker for AIFlow 1.0e.

기존 online ink와 HWR 후보만 사용한다. 후보를 만들거나 순서를 정답으로
주입하지 않으며, residual head가 0으로 시작하므로 base logits를 주면
초기 순위가 그대로 보존된다.
"""

from __future__ import annotations

import json

import torch
from torch import nn


SCHEMA = "aiflow-1.0e-formula-context-ranker/v1"


def _check_finite(name: str, value: torch.Tensor, mask: torch.Tensor) -> None:
    """마스킹된 padding은 무시하고 실제 입력의 non-finite 값만 거부한다."""
    selected = value[mask]
    if selected.numel() and not torch.isfinite(selected).all():
        raise ValueError(f"{name} contains non-finite active values")


class FormulaContextRanker(nn.Module):
    """수식의 online ink row를 문맥화하고 기존 후보의 residual을 계산한다."""

    def __init__(
        self,
        numeric_size: int,
        token_count: int,
        *,
        ink_size: int = 128,
        hidden: int = 128,
        layers: int = 2,
        heads: int = 4,
        feedforward: int = 512,
        token_width: int = 48,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if numeric_size <= 0 or token_count <= 0 or ink_size <= 0:
            raise ValueError("feature and vocabulary sizes must be positive")
        if hidden <= 0 or hidden % heads:
            raise ValueError("hidden must be positive and divisible by heads")
        if layers <= 0 or feedforward <= 0 or token_width <= 0:
            raise ValueError("layers, feedforward, and token_width must be positive")
        self.config = {
            "numeric_size": numeric_size,
            "token_count": token_count,
            "ink_size": ink_size,
            "hidden": hidden,
            "layers": layers,
            "heads": heads,
            "feedforward": feedforward,
            "token_width": token_width,
            "dropout": dropout,
        }
        self.ink_projection = nn.Sequential(nn.Linear(ink_size, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.numeric_projection = nn.Sequential(nn.Linear(numeric_size, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.token_embedding = nn.Embedding(token_count, token_width)
        self.token_context_projection = nn.Sequential(
            nn.Linear(token_width, hidden), nn.LayerNorm(hidden), nn.GELU()
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context_encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.candidate_fusion = nn.Sequential(
            nn.Linear(hidden + hidden + token_width, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
        )
        self.residual_head = nn.Linear(hidden // 2, 1)
        self.coverage_head = nn.Linear(hidden, 1)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def _validate(
        self,
        ink: torch.Tensor,
        numeric: torch.Tensor,
        token_ids: torch.Tensor,
        candidate_mask: torch.Tensor,
        row_mask: torch.Tensor | None,
        base_logits: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """입력 shape, mask, 유효 token ID를 검사하고 effective mask를 만든다."""
        if ink.ndim != 3 or numeric.ndim != 4 or token_ids.ndim != 3:
            raise ValueError("expected ink[B,L,D], numeric[B,L,K,N], and token_ids[B,L,K]")
        if ink.shape[0:2] != numeric.shape[0:2] or numeric.shape[:3] != token_ids.shape:
            raise ValueError("ink, numeric, and token_ids shapes do not match")
        if candidate_mask.shape != token_ids.shape or candidate_mask.dtype != torch.bool:
            raise ValueError("candidate_mask must be bool and match token_ids")
        if ink.shape[-1] != self.config["ink_size"] or numeric.shape[-1] != self.config["numeric_size"]:
            raise ValueError("feature dimensions do not match the ranker configuration")
        if token_ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise ValueError("token_ids must use an integer dtype")
        if row_mask is None:
            row_mask = candidate_mask.any(dim=-1)
        if row_mask.shape != ink.shape[:2] or row_mask.dtype != torch.bool:
            raise ValueError("row_mask must be bool and match ink[B,L]")
        if base_logits is not None:
            if base_logits.shape != token_ids.shape or not base_logits.is_floating_point():
                raise ValueError("base_logits must be floating point with shape [B,L,K]")
        effective_mask = candidate_mask & row_mask.unsqueeze(-1)
        active_ids = token_ids[effective_mask]
        if active_ids.numel() and ((active_ids < 0) | (active_ids >= self.config["token_count"])).any():
            raise ValueError("active token_ids are outside the vocabulary")
        _check_finite("ink", ink, row_mask.unsqueeze(-1).expand_as(ink))
        _check_finite("numeric", numeric, effective_mask.unsqueeze(-1).expand_as(numeric))
        if base_logits is not None:
            _check_finite("base_logits", base_logits, effective_mask)
        return row_mask, effective_mask

    def _encode(
        self,
        ink: torch.Tensor,
        numeric: torch.Tensor,
        token_ids: torch.Tensor,
        candidate_mask: torch.Tensor,
        row_mask: torch.Tensor | None,
        base_logits: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """후보확률 가중 token 문맥을 포함한 row state와 candidate state를 계산한다."""
        row_mask, effective_mask = self._validate(
            ink, numeric, token_ids, candidate_mask, row_mask, base_logits
        )
        safe_token_ids = token_ids.clamp(0, self.config["token_count"] - 1).to(torch.long)
        token = self.token_embedding(safe_token_ids)
        token_mask = effective_mask.unsqueeze(-1).to(token.dtype)

        if base_logits is None:
            probabilities = token_mask.squeeze(-1)
        else:
            probability_logits = base_logits.to(token.dtype).masked_fill(~effective_mask, -torch.finfo(token.dtype).max)
            probabilities = torch.softmax(probability_logits, dim=-1) * token_mask.squeeze(-1)
        denominator = probabilities.sum(dim=-1, keepdim=True).clamp_min(1.0)
        probabilities = probabilities / denominator
        row_token = (token * probabilities.unsqueeze(-1)).sum(dim=2)

        safe_ink = torch.where(row_mask.unsqueeze(-1), ink, torch.zeros_like(ink))
        safe_numeric = torch.where(effective_mask.unsqueeze(-1), numeric, torch.zeros_like(numeric))
        row_count = effective_mask.sum(dim=-1, keepdim=True).clamp_min(1).to(numeric.dtype)
        row_numeric = safe_numeric.sum(dim=2) / row_count
        row_state = (
            self.ink_projection(safe_ink)
            + self.numeric_projection(row_numeric)
            + self.token_context_projection(row_token)
        )

        if row_state.shape[1] == 0:
            context = row_state
        else:
            encoder_padding = ~row_mask
            empty_formula = ~row_mask.any(dim=1)
            if empty_formula.any():
                encoder_padding = encoder_padding.clone()
                encoder_padding[empty_formula, 0] = False
            context = self.context_encoder(row_state, src_key_padding_mask=encoder_padding)
            context = torch.where(row_mask.unsqueeze(-1), context, torch.zeros_like(context))
        candidate = self.numeric_projection(safe_numeric)
        return context, candidate, token

    def _forward_parts(
        self,
        ink: torch.Tensor,
        numeric: torch.Tensor,
        token_ids: torch.Tensor,
        candidate_mask: torch.Tensor,
        row_mask: torch.Tensor | None,
        base_logits: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """residual score, coverage logit, effective mask를 공통 계산한다."""
        context, candidate, token = self._encode(
            ink, numeric, token_ids, candidate_mask, row_mask, base_logits
        )
        fused = self.candidate_fusion(
            torch.cat((context.unsqueeze(2).expand(-1, -1, numeric.shape[2], -1), candidate, token), dim=-1)
        )
        residual = self.residual_head(fused).squeeze(-1)
        effective_mask = candidate_mask & (
            candidate_mask.any(dim=-1) if row_mask is None else row_mask
        ).unsqueeze(-1)
        scores = residual
        if base_logits is not None:
            scores = scores + base_logits
        scores = scores.masked_fill(~effective_mask, torch.finfo(scores.dtype).min)
        coverage = self.coverage_head(context).squeeze(-1)
        return scores, coverage, effective_mask

    def forward(
        self,
        ink: torch.Tensor,
        numeric: torch.Tensor,
        token_ids: torch.Tensor,
        candidate_mask: torch.Tensor,
        row_mask: torch.Tensor | None = None,
        *,
        base_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """base logits와 zero-init residual을 합친 후보 score를 반환한다."""
        scores, _coverage, _mask = self._forward_parts(
            ink, numeric, token_ids, candidate_mask, row_mask, base_logits
        )
        return scores

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
        """후보 score와 각 row의 candidate coverage logit을 반환한다."""
        scores, coverage, _mask = self._forward_parts(
            ink, numeric, token_ids, candidate_mask, row_mask, base_logits
        )
        return scores, coverage


def _self_test() -> None:
    """모듈 단독 실행 시 zero-init과 padding 안전성만 빠르게 확인한다."""
    torch.manual_seed(20260910)
    model = FormulaContextRanker(24, 32).eval()
    ink = torch.randn(2, 5, 128)
    numeric = torch.randn(2, 5, 4, 24)
    token_ids = torch.randint(0, 32, (2, 5, 4))
    mask = torch.ones(2, 5, 4, dtype=torch.bool)
    mask[:, -1, 2:] = False
    row_mask = mask.any(dim=-1)
    base = torch.randn(2, 5, 4)
    scores, coverage = model.forward_with_coverage(
        ink, numeric, token_ids, mask, row_mask, base_logits=base
    )
    assert scores.shape == (2, 5, 4) and coverage.shape == (2, 5)
    assert torch.allclose(scores[mask], base[mask])
    assert torch.isfinite(scores[mask]).all() and torch.isfinite(coverage).all()
    assert torch.allclose(model.residual_head.weight, torch.zeros_like(model.residual_head.weight))


if __name__ == "__main__":
    _self_test()
    print(json.dumps({"schema": SCHEMA, "self_test": "pass"}, ensure_ascii=False))
