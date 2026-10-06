"""위험 모델과 threshold 보정의 writer 분리 계약."""
import unittest

from accuracy_evaluation_10e import fit_guard, risk_features


def _row(writer, formula, baseline, adapter, label, missing=False):
    return {
        'record_id': f'{writer}:{formula}:{baseline}:{adapter}', 'writer_group': writer,
        'formula_id': formula, 'formula_key': formula, 'baseline_token': baseline,
        'adapter_token': adapter, 'label': label, 'candidates': ['a', 'b'],
        'adapter_scores': [2.0, 1.0] if adapter == 'a' else [1.0, 2.0],
        'baseline_probabilities': [0.7, 0.3], 'coverage_probability': 0.8,
        'geometry_missing': missing,
    }


class GuardTests(unittest.TestCase):
    def test_risk_fit_and_threshold_writers_are_disjoint(self):
        fit = [_row('fit-a', 'f1', 'a', 'b', 'b'), _row('fit-b', 'f2', 'a', 'b', 'a')]
        threshold = [_row('cal', 'f3', 'a', 'b', 'b'), _row('cal', 'f4', 'a', 'b', 'a')]
        guard, report = fit_guard(fit, threshold)
        self.assertEqual(report['calibration_scope'], 'writer_disjoint_risk_fit_and_threshold_oof')
        self.assertIsNotNone(guard['model'])
        with self.assertRaisesRegex(ValueError, 'writers overlap'):
            fit_guard(fit, [dict(threshold[0], writer_group='fit-a')])

    def test_geometry_missing_is_an_explicit_feature(self):
        present = risk_features(_row('w', 'f', 'a', 'b', 'b', False))
        missing = risk_features(_row('w', 'f', 'a', 'b', 'b', True))
        self.assertEqual(present[:-1], missing[:-1])
        self.assertEqual((present[-1], missing[-1]), (0.0, 1.0))


if __name__ == '__main__':
    unittest.main()
