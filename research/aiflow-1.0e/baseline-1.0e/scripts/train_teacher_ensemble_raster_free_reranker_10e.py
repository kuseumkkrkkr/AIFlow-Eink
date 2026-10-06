#!/usr/bin/env python3
"""Teacher-heavy loss variant of the raster-free 1.0e candidate reranker.

This bounded variant tests whether the uniform three-teacher distribution can
transfer when KL is dominant.  Runtime inputs and candidate constraints are
identical to ``train_online_candidate_distill_10e.py``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch import nn

import train_online_candidate_distill_10e as base
from accuracy_upgrade_contract_v1 import masked_candidate_kl


def teacher_heavy_loss(
    scores: torch.Tensor,
    teacher_scores: torch.Tensor,
    targets: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    valid = targets >= 0
    if not bool(valid.any()):
        return scores.sum() * 0.0, {"ce": 0.0, "kl": 0.0, "stability": 0.0}
    score_rows = scores[valid]
    teacher_rows = teacher_scores[valid]
    target_rows = targets[valid]
    ce = nn.functional.cross_entropy(score_rows, target_rows)
    stability_values = []
    pair_values = []
    for index, target in enumerate(target_rows.tolist()):
        size = int((teacher_rows[index] > -1e8).sum())
        other = score_rows[index, :size].clone()
        other[target] = torch.finfo(other.dtype).min
        hardest = other.max()
        if target == 0:
            stability_values.append(nn.functional.relu(0.25 - score_rows[index, 0] + hardest))
        else:
            pair_values.append(nn.functional.relu(0.25 - score_rows[index, target] + score_rows[index, 0]))
    widths = torch.tensor(
        [(teacher_rows[index] > -1e8).sum().item() for index in range(len(teacher_rows))],
        device=score_rows.device,
    )
    candidate_mask = torch.arange(score_rows.shape[-1], device=score_rows.device).unsqueeze(0) < widths.unsqueeze(1)
    kl = masked_candidate_kl(score_rows, teacher_rows, candidate_mask, temperature)
    stability = torch.stack(stability_values).mean() if stability_values else score_rows.mean() * 0.0
    pairwise = torch.stack(pair_values).mean() if pair_values else score_rows.mean() * 0.0
    loss = 0.20 * ce + 0.70 * kl + 0.10 * stability + 0.10 * pairwise
    return loss, {
        "ce": float(ce.detach()),
        "kl": float(kl.detach()),
        "stability": float((stability + pairwise).detach()),
    }


def main() -> int:
    base._loss = teacher_heavy_loss
    code = base.main()
    output = None
    for index, value in enumerate(sys.argv):
        if value == "--output" and index + 1 < len(sys.argv):
            output = Path(sys.argv[index + 1])
        elif value.startswith("--output="):
            output = Path(value.split("=", 1)[1])
    if output is None:
        output = base.DEFAULT_OUTPUT
    evaluation_path = output / "evaluation.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluation["schema"] = "aiflow-1.0e-teacher-ensemble-raster-free-reranker/v1"
    evaluation["training"]["loss"] = "0.20 CE + 0.70 teacher KL + 0.10 baseline stability + 0.10 pairwise margin"
    evaluation["teacher_ensemble"] = "fixed uniform mean of TexTeller, UniMERNet tiny, and Microsoft TrOCR-small OOF probabilities"
    evaluation["runtime_boundary"] = "existing 5-channel ordered online-ink embedding and existing HWR Top-k only; no raster or external model"
    evaluation_path.write_text(json.dumps(evaluation, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checkpoint_path = output / "online_candidate_ranker.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["schema"] = evaluation["schema"]
    checkpoint["training_loss"] = evaluation["training"]["loss"]
    checkpoint["teacher_ensemble_in_runtime"] = False
    torch.save(checkpoint, checkpoint_path)
    print(json.dumps({
        "event": "teacher_heavy_variant_complete",
        "loss": "0.20 CE + 0.70 teacher KL + 0.10 baseline stability + 0.10 pairwise margin",
        "runtime": "raster-free existing 5-channel HWR embedding and existing Top-k only",
    }), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
