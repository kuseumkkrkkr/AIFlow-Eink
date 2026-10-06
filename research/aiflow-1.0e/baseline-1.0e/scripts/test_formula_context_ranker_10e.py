#!/usr/bin/env python3
"""CPU contract tests for the independent R3 context ranker and loss."""

from __future__ import annotations

import math
import unittest

import torch
import torch.nn.functional as F

from formula_candidate_loss_10e import formula_candidate_loss
from formula_context_ranker_10e import FormulaContextRanker


def _inputs(batch: int = 2, rows: int = 3, candidates: int = 4, numeric_size: int = 5):
    """작은 deterministic online-only batch를 만든다."""
    torch.manual_seed(10)
    ink = torch.randn(batch, rows, 128)
    numeric = torch.randn(batch, rows, candidates, numeric_size)
    token_ids = torch.randint(0, 16, (batch, rows, candidates))
    candidate_mask = torch.ones(batch, rows, candidates, dtype=torch.bool)
    row_mask = torch.ones(batch, rows, dtype=torch.bool)
    return ink, numeric, token_ids, candidate_mask, row_mask


def _small_model() -> FormulaContextRanker:
    return FormulaContextRanker(
        5, 16, hidden=16, layers=1, heads=4, feedforward=32, token_width=8, dropout=0.0
    ).eval()


class FormulaContextRankerTests(unittest.TestCase):
    def test_zero_init_preserves_base_rank_and_shuffled_candidate_order(self):
        model = _small_model()
        inputs = _inputs()
        base = torch.randn(2, 3, 4)
        scores = model(*inputs[:4], inputs[4], base_logits=base)
        self.assertTrue(torch.allclose(scores, base))
        permutation = torch.tensor([2, 0, 3, 1])
        shuffled = (
            inputs[0],
            inputs[1].index_select(-2, permutation),
            inputs[2].index_select(-1, permutation),
            inputs[3].index_select(-1, permutation),
        )
        shuffled_scores = model(
            *shuffled, inputs[4], base_logits=base.index_select(-1, permutation)
        )
        self.assertTrue(torch.allclose(shuffled_scores, base.index_select(-1, permutation)))

    def test_neighbor_token_changes_context_when_residual_head_is_nonzero(self):
        model = _small_model()
        with torch.no_grad():
            model.residual_head.weight.fill_(1.0)
        inputs = list(_inputs(batch=1, rows=3))
        first = model(*inputs[:4], inputs[4])
        inputs[2] = inputs[2].clone()
        inputs[2][0, 1, 0] = (inputs[2][0, 1, 0] + 1) % 16
        changed = model(*inputs[:4], inputs[4])
        self.assertFalse(torch.allclose(first[0, 0], changed[0, 0]))

    def test_detached_coverage_does_not_update_shared_encoder(self):
        """coverage 보조 손실만으로 공유 표현이 이동하지 않는다."""
        model = FormulaContextRanker(
            5, 16, hidden=16, layers=1, heads=4, feedforward=32,
            token_width=8, dropout=0.0, detach_coverage_features=True,
        )
        inputs = _inputs(batch=1, rows=2)
        _scores, coverage = model.forward_with_coverage(*inputs[:4], inputs[4])
        coverage.sum().backward()
        self.assertIsNotNone(model.coverage_head.weight.grad)
        self.assertIsNone(model.ink_projection[0].weight.grad)
        self.assertIsNone(model.context_encoder.layers[0].self_attn.in_proj_weight.grad)

    def test_individual_and_batch_padding_match_and_invalid_padding_is_safe(self):
        model = _small_model()
        base_inputs = _inputs(batch=1, rows=2)
        base = torch.randn(1, 2, 4)
        individual = model(*base_inputs[:4], base_inputs[4], base_logits=base)
        ink = torch.cat((base_inputs[0], torch.full((1, 2, 128), float("nan"))), dim=1)
        numeric = torch.cat((base_inputs[1], torch.full((1, 2, 4, 5), float("nan"))), dim=1)
        token_ids = torch.cat(
            (base_inputs[2], torch.full((1, 2, 4), 999999, dtype=torch.long)), dim=1
        )
        candidate_mask = torch.cat(
            (base_inputs[3], torch.zeros(1, 2, 4, dtype=torch.bool)), dim=1
        )
        row_mask = torch.tensor([[True, True, False, False]])
        batch_base = torch.cat((base, torch.zeros(1, 2, 4)), dim=1)
        padded = model(ink, numeric, token_ids, candidate_mask, row_mask, base_logits=batch_base)
        self.assertTrue(torch.allclose(individual, padded[:, :2]))
        self.assertTrue(torch.isfinite(padded).all())

    def test_shape_rejects_and_all_row_padding_is_finite(self):
        model = _small_model()
        inputs = _inputs()
        with self.assertRaises(ValueError):
            model(inputs[0], inputs[1][..., :4], inputs[2], inputs[3], inputs[4])
        ink, numeric, token_ids, candidate_mask, row_mask = _inputs(batch=1)
        row_mask.zero_()
        candidate_mask.zero_()
        token_ids.fill_(999999)
        ink.fill_(float("nan"))
        numeric.fill_(float("nan"))
        scores, coverage = model.forward_with_coverage(
            ink, numeric, token_ids, candidate_mask, row_mask
        )
        self.assertTrue(torch.isfinite(scores).all())
        self.assertTrue(torch.isfinite(coverage).all())

    def test_missing_targets_exclude_ce_kd_but_enter_coverage(self):
        scores = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]], requires_grad=True)
        targets = torch.tensor([[0, -1]])
        mask = torch.ones(1, 2, 2, dtype=torch.bool)
        rows = torch.ones(1, 2, dtype=torch.bool)
        coverage_logits = torch.tensor([[0.0, 1.0]], requires_grad=True)
        loss, metrics = formula_candidate_loss(
            scores, targets, mask, rows, coverage_logits=coverage_logits
        )
        expected_ce = F.cross_entropy(scores[:, :1].reshape(1, 2), targets[:, :1].reshape(-1))
        self.assertTrue(math.isclose(metrics["ce"], float(expected_ce), rel_tol=1e-6))
        self.assertEqual(metrics["kd"], 0.0)
        self.assertGreater(metrics["coverage"], 0.0)
        loss.backward()
        self.assertTrue(torch.isfinite(scores.grad).all())
        self.assertTrue(torch.isfinite(coverage_logits.grad).all())

    def test_exact_candidate_kl_is_temperature_squared_sum_then_formula_mean(self):
        scores = torch.tensor(
            [[[1.0, 2.0, -4.0], [0.0, 4.0, 1.0]], [[2.0, 0.0, 3.0], [7.0, 1.0, 0.0]]],
            requires_grad=True,
        )
        teacher = torch.tensor(
            [[[2.0, 1.0, -3.0], [1.0, 3.0, 0.0]], [[1.0, 0.0, 2.0], [2.0, 4.0, 0.0]]]
        )
        targets = torch.tensor([[0, -1], [2, -1]])
        mask = torch.tensor(
            [
                [[True, True, False], [True, True, True]],
                [[True, True, True], [True, False, False]],
            ]
        )
        rows = torch.tensor([[True, True], [True, False]])
        temperature = 2.0
        _, metrics = formula_candidate_loss(
            scores, targets, mask, rows, teacher_scores=teacher, temperature=temperature, kd_weight=1.0
        )
        expected_rows = []
        for batch_index in (0, 1):
            active = mask[batch_index, 0]
            student = scores[batch_index, 0][active] / temperature
            target = teacher[batch_index, 0][active] / temperature
            expected_rows.append(
                float(
                    F.kl_div(F.log_softmax(student, 0), F.softmax(target, 0), reduction="sum")
                    * temperature**2
                )
            )
        self.assertTrue(math.isclose(metrics["kd"], sum(expected_rows) / 2, rel_tol=1e-6))

    def test_padded_invalid_scores_have_no_effect_and_gradient_is_finite(self):
        scores_a = torch.tensor(
            [[[2.0, 1.0, -100.0], [1.0, 2.0, -100.0]]], requires_grad=True
        )
        scores_b = scores_a.detach().clone().requires_grad_()
        teacher_a = torch.tensor([[[1.0, 0.0, -100.0], [0.0, 1.0, -100.0]]])
        teacher_b = teacher_a.clone()
        teacher_b[:, :, 2] = 1e30
        with torch.no_grad():
            scores_b[:, :, 2] = 1e30
        targets = torch.tensor([[0, 1]])
        mask = torch.tensor([[[True, True, False], [True, True, False]]])
        rows = torch.ones(1, 2, dtype=torch.bool)
        loss_a, _ = formula_candidate_loss(
            scores_a, targets, mask, rows, teacher_scores=teacher_a, kd_weight=0.5
        )
        loss_b, _ = formula_candidate_loss(
            scores_b, targets, mask, rows, teacher_scores=teacher_b, kd_weight=0.5
        )
        self.assertTrue(torch.allclose(loss_a, loss_b))
        loss_b.backward()
        self.assertTrue(torch.isfinite(scores_b.grad).all())

    def test_all_missing_all_padding_returns_differentiable_zero(self):
        scores = torch.randn(2, 3, 4, requires_grad=True)
        targets = torch.full((2, 3), -1)
        candidate_mask = torch.zeros(2, 3, 4, dtype=torch.bool)
        row_mask = torch.zeros(2, 3, dtype=torch.bool)
        loss, metrics = formula_candidate_loss(scores, targets, candidate_mask, row_mask)
        self.assertTrue(loss.requires_grad)
        self.assertEqual(float(loss), 0.0)
        self.assertEqual(metrics["ce"], metrics["kd"], 0.0)
        self.assertEqual(metrics["kd"], metrics["coverage"], 0.0)
        loss.backward()
        self.assertTrue(torch.isfinite(scores.grad).all())

    def test_padding_targets_are_ignored_but_invalid_active_targets_are_rejected(self):
        scores = torch.zeros(1, 2, 2, requires_grad=True)
        mask = torch.ones(1, 2, 2, dtype=torch.bool)
        rows = torch.tensor([[True, False]])
        padding_targets = torch.tensor([[0, 999]])
        loss, _ = formula_candidate_loss(scores, padding_targets, mask, rows)
        self.assertTrue(torch.isfinite(loss))
        with self.assertRaises(ValueError):
            formula_candidate_loss(scores, torch.tensor([[2, -1]]), mask, rows)

    def test_missing_candidate_target_is_an_explicit_error(self):
        scores = torch.zeros(1, 1, 2, requires_grad=True)
        targets = torch.tensor([[1]])
        mask = torch.tensor([[[True, False]]])
        rows = torch.ones(1, 1, dtype=torch.bool)
        with self.assertRaises(ValueError):
            formula_candidate_loss(scores, targets, mask, rows)

    def test_inactive_nan_coverage_and_kl_values_do_not_poison_loss_or_gradients(self):
        scores = torch.tensor([[[2.0, 1.0], [float("nan"), float("nan")]]], requires_grad=True)
        teacher = torch.tensor([[[1.0, 0.0], [float("nan"), float("nan")]]])
        targets = torch.tensor([[0, -1]])
        candidate_mask = torch.ones(1, 2, 2, dtype=torch.bool)
        row_mask = torch.tensor([[True, False]])
        coverage = torch.tensor([[0.5, float("nan")]], requires_grad=True)
        loss, _ = formula_candidate_loss(
            scores, targets, candidate_mask, row_mask,
            teacher_scores=teacher, kd_weight=0.5, coverage_logits=coverage,
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(scores.grad).all())
        self.assertTrue(torch.isfinite(coverage.grad).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
