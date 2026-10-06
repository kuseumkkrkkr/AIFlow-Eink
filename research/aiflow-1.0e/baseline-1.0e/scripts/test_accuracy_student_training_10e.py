#!/usr/bin/env python3
"""CPU-only synthetic contract tests for the student runner."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from accuracy_student_training_10e import fit_student, new_student, predict_student


def _samples() -> dict[str, dict]:
    """두 row 수식 여러 개와 두 개 이상의 writer를 만든다."""
    rng = np.random.default_rng(20260911)
    result = {}
    for formula_index in range(3):
        formula_id = f"formula-{formula_index}"
        writer = f"writer-{formula_index}"
        for row_index in range(2):
            record_id = f"{formula_id}:row-{row_index}"
            candidates = ["a", "b", "c"]
            target = row_index % 2
            result[record_id] = {
                "record_id": record_id,
                "formula_id": formula_id,
                "formula_key": formula_id,
                "writer_group": writer,
                "label": candidates[target],
                "candidates": candidates,
                "target": target,
                "ink": rng.normal(size=128).astype(np.float32),
                "numeric": rng.normal(size=(3, 21)).astype(np.float32),
                "token_ids": np.asarray([0, 1, 2], dtype=np.int64),
                "base_logits": np.asarray([2.0, 1.0, 0.0], dtype=np.float32),
                "source_indices": [row_index],
            }
    return result


class AccuracyStudentTrainingTests(unittest.TestCase):
    def test_context_fit_uses_formula_batches_and_real_gradients(self):
        samples = _samples()
        train_ids = [key for key in samples if key.startswith(("formula-0", "formula-1"))]
        validation_ids = [key for key in samples if key.startswith("formula-2")]
        model, report = fit_student(
            samples,
            train_ids,
            {
                "epochs": 1, "patience": 1, "seed": 7, "key": "cpu-test",
                "token_count": 8, "kd_weight": 0.0, "loss_mode": "standard",
            },
            torch.device("cpu"),
            validation_ids=validation_ids,
        )
        self.assertEqual(report["actual_batch"], 8)
        self.assertEqual(report["accumulation_steps"], 4)
        self.assertEqual(report["effective_batch"], 32)
        self.assertEqual(report["fit_ids"], train_ids)
        self.assertEqual(report["selection_ids"], validation_ids)
        self.assertEqual(report["global_fit_ids"], train_ids)
        self.assertEqual(report["global_selection_ids"], validation_ids)
        self.assertEqual(report["loss_mode"], "standard")
        self.assertEqual(report["selection_source"], "glyph_formula_exact")
        self.assertEqual(report["best_epoch"], 1)
        self.assertTrue(report["losses"])
        self.assertGreater(report["losses"][0]["gradient_norm"], 0.0)
        predictions = predict_student(model, samples, validation_ids, torch.device("cpu"))
        self.assertEqual(len(predictions), len(validation_ids))
        self.assertTrue(all("coverage_probability" in row for row in predictions))
        self.assertTrue(all("raw_margin" in row and "raw_entropy" in row for row in predictions))
        self.assertTrue(all(row["target_in_candidates"] for row in predictions))

    def test_mlp_wrapper_has_same_interface_and_absolute_scores(self):
        model = new_student("mlp", 21, 8).eval()
        samples = _samples()
        rows = [samples[key] for key in sorted(samples)[:2]]
        ink = torch.from_numpy(np.stack([row["ink"] for row in rows])).unsqueeze(0)
        numeric = torch.from_numpy(np.stack([row["numeric"] for row in rows])).unsqueeze(0)
        token_ids = torch.from_numpy(np.stack([row["token_ids"] for row in rows])).unsqueeze(0)
        candidate_mask = torch.ones(1, 2, 3, dtype=torch.bool)
        row_mask = torch.ones(1, 2, dtype=torch.bool)
        base = torch.zeros(1, 2, 3)
        scores, coverage = model.forward_with_coverage(
            ink, numeric, token_ids, candidate_mask, row_mask=row_mask, base_logits=base
        )
        self.assertEqual(tuple(scores.shape), (1, 2, 3))
        self.assertEqual(tuple(coverage.shape), (1, 2))
        self.assertTrue(torch.isfinite(scores).all() and torch.isfinite(coverage).all())

    def test_fixed_epoch_refit_returns_final_epoch_without_validation_selection(self):
        samples = _samples()
        train_ids = [key for key in samples if key.startswith(("formula-0", "formula-1"))]
        _, report = fit_student(
            samples, train_ids,
            {"epochs": 2, "patience": 1, "seed": 9, "token_count": 8, "kd_weight": 0.0},
            torch.device("cpu"),
        )
        self.assertEqual(report["best_epoch"], 2)
        self.assertEqual(report["selection_source"], "fixed_final")
        self.assertFalse(report["stopped_early"])

    def test_writer_or_formula_overlap_is_rejected_before_fit(self):
        samples = _samples()
        train_ids = [key for key in samples if key.startswith("formula-0")]
        same_writer = [key for key in samples if key.startswith("formula-0")]
        with self.assertRaises(ValueError):
            fit_student(samples, train_ids, {"max_epochs": 1}, torch.device("cpu"), validation_ids=same_writer)
        same_formula = [key for key in samples if key.startswith("formula-0")]
        with self.assertRaises(ValueError):
            fit_student(samples, train_ids, {"max_epochs": 1}, torch.device("cpu"), validation_ids=same_formula)
        keyed = _samples()
        keyed["formula-1:row-0"]["formula_key"] = "formula-0"
        keyed["formula-1:row-1"]["formula_key"] = "formula-0"
        different_formula_same_key = [key for key in keyed if key.startswith("formula-1")]
        with self.assertRaises(ValueError):
            fit_student(keyed, train_ids, {"max_epochs": 1}, torch.device("cpu"), validation_ids=different_formula_same_key)


if __name__ == "__main__":
    unittest.main(verbosity=2)
