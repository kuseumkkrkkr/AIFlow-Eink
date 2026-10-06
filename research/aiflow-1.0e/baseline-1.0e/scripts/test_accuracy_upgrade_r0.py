"""실제로 발견한 호환성·NaN·계보 누수 회귀를 검사한다."""
import json
import tempfile
import unittest
from pathlib import Path

import torch

from accuracy_upgrade_contract_v1 import masked_candidate_kl, normalize_latex, canonical_json_sha256
from accuracy_lineage_10e import LineageRegistry, validate_prediction
from online_candidate_features_10e import _row_numeric, checkpoint_feature_version
from train_online_candidate_distill_10e import _loss


class RegressionTests(unittest.TestCase):
    """수치와 데이터 경계 실패가 조용히 성공으로 바뀌지 않는지 확인한다."""

    def test_missing_target_padding_nan(self):
        """padding floor 합산 overflow와 all-missing loss를 재현한다."""
        scores = torch.tensor([[1., torch.finfo(torch.float32).min, torch.finfo(torch.float32).min]] * 2, requires_grad=True)
        mask = torch.tensor([[True, False, False]] * 2)
        loss, _ = _loss(scores, scores.detach(), torch.tensor([-1, -1]), 2., mask)
        self.assertEqual(float(loss), 0.)
        loss.backward()
        self.assertTrue(torch.isfinite(scores.grad).all())
        zero = masked_candidate_kl(scores, scores.detach(), torch.zeros_like(mask), 2.)
        self.assertEqual(float(zero), 0.)

    def test_kl_mask_invariance(self):
        """추가 padding에 비유한 값이 있어도 유효 KL은 변하지 않는다."""
        student, teacher = torch.tensor([[1., 2.]]), torch.tensor([[2., 1.]])
        expected = masked_candidate_kl(student, teacher, torch.ones_like(student, dtype=torch.bool), 2.)
        value = masked_candidate_kl(torch.tensor([[1., 2., float('nan')]]), torch.tensor([[2., 1., float('inf')]]), torch.tensor([[True, True, False]]), 2.)
        self.assertTrue(torch.allclose(expected, value))

    def test_features_legacy_and_current(self):
        """21/28 차원은 명시적으로 생성하고 잘못된 계약은 거부한다."""
        row = dict(final_topk=['x'], final_topk_probabilities=[1.], geometry={})
        self.assertEqual(len(_row_numeric(row, 0, 'legacy21')), 21)
        self.assertEqual(len(_row_numeric(row, 0)), 28)
        self.assertEqual(checkpoint_feature_version({'numeric_size': 21}), 'legacy21')
        with self.assertRaises(ValueError):
            checkpoint_feature_version({'numeric_size': 21, 'feature_version': 'formula28'})

    def test_latex_meaning(self):
        """일반 수학 공백만 지우고 문자·텍스트·잘못된 산술식을 보존한다."""
        self.assertEqual(normalize_latex(r'\[ x + 1 \]'), 'x+1')
        self.assertNotEqual(normalize_latex(r'\text{a  b}'), normalize_latex(r'\text{a b}'))
        self.assertNotEqual(normalize_latex(r'\alpha x'), normalize_latex(r'\alphax'))
        self.assertNotEqual(normalize_latex('1+1=3'), normalize_latex('1+1=2'))
        self.assertNotEqual(normalize_latex('x'), normalize_latex(r'\times'))
        self.assertEqual(normalize_latex('x × y'), normalize_latex(r'x\times y'))
        self.assertEqual(normalize_latex('x²'), normalize_latex(r'x^{2}'))

    def test_real_ancestor_exclusion(self):
        """자신을 학습한 inner 예측은 outer 제외 선언이 있어도 실패한다."""
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / 'weights.json'
            artifact.write_text('{}', encoding='utf-8')
            records = {'a': dict(writer='outer', formula_key='x'), 'b': dict(writer='inner', formula_key='y')}
            registry = LineageRegistry(records)
            base = registry.add('external', artifact)
            bad = registry.add('student', artifact, ['b'], [base])
            row = dict(record_id='b', writer_group='inner', candidates=['y'], prediction_provenance={
                'root_id': bad, 'candidate_sha256': canonical_json_sha256(['y']), 'excluded_writers': ['outer']})
            with self.assertRaisesRegex(ValueError, 'writer leakage'):
                validate_prediction(row, 'outer', registry)
            row['prediction_provenance']['root_id'] = base
            validate_prediction(row, 'outer', registry)
            changed = registry.add('selection-leak', artifact, [], [base], ['a'])
            row['prediction_provenance']['root_id'] = changed
            with self.assertRaisesRegex(ValueError, 'writer leakage'):
                validate_prediction(row, 'outer', registry)

    def test_legacy_claim_rejected(self):
        """제외 writer 목록만 있는 옛 teacher/guard provenance는 불충분하다."""
        with self.assertRaisesRegex(ValueError, 'actual lineage'):
            validate_prediction(dict(record_id='b', writer_group='inner', candidates=['x'],
                                     prediction_provenance={'excluded_writers': ['outer', 'inner']}), 'outer')

    def test_unverified_teacher_fails_before_loading(self):
        """권리 미확인 teacher는 특징이나 모델을 읽기 전에 차단한다."""
        from accuracy_teacher_targets_10e import TeacherTargets
        with self.assertRaisesRegex(ValueError, 'blocked_data: teacher rights'):
            TeacherTargets(None, ['trocr_small'])


if __name__ == '__main__':
    unittest.main()
