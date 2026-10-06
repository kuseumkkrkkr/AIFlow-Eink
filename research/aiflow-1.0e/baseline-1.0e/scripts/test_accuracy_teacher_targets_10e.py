"""Teacher 누적 학습의 수식 단위 가중 계약."""
import unittest

from accuracy_teacher_targets_10e import formula_window_weights


class TeacherTargetTests(unittest.TestCase):
    def test_partial_last_batch_uses_formula_counts(self):
        """8·8·3개 수식 batch는 세 batch를 같은 비율로 처리하지 않는다."""
        window = [[{}] * 8, [{}] * 8, [{}] * 3]
        self.assertEqual(formula_window_weights(window), [8 / 19, 8 / 19, 3 / 19])
        self.assertAlmostEqual(sum(formula_window_weights(window)), 1.0)

    def test_empty_window_rejected(self):
        with self.assertRaises(ValueError):
            formula_window_weights([])


if __name__ == '__main__':
    unittest.main()
