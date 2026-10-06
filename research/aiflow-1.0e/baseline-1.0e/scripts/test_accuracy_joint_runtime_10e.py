"""원본 획 보존·정답 비사용·완료 멱등성·평가 분모 계약을 검사한다."""
import copy
import unittest
import numpy as np

from accuracy_joint_runtime_10e import CompletionCache, canonical_group, group_tensor, layout_output
from accuracy_evaluation_10e import formula_metrics, risk_features
from accuracy_upgrade_data_10e import display_to_latex
from accuracy_upgrade_contract_v1 import normalize_latex


class RuntimeTests(unittest.TestCase):
    """모델 점수와 무관하게 반드시 유지해야 하는 제품 입력 계약을 검증한다."""
    def test_single_point_preserved(self):
        """단일점 획은 한 관측과 마스크된 padding으로 보존한다."""
        strokes = [dict(order=0, points=[dict(x=15., y=20., t_ms=0.)])]
        row, single = canonical_group(strokes, [0], 'one')
        self.assertTrue(single)
        result = group_tensor(row, single, 'uniform-time')
        self.assertEqual(result.shape, (128, 5))
        self.assertEqual(float(result[:, 3].sum()), 1.)
        self.assertEqual(float(result[:, 4].sum()), 1.)
        self.assertEqual(len(strokes[0]['points']), 1)

    def test_complete_idempotent_and_copy_safe(self):
        """반복 완료는 재추론하지 않고 원본·반환값을 외부 변경으로부터 보호한다."""
        cache = CompletionCache(); raw = {'strokes': [{'order': 0, 'points': [{'x': 1, 'y': 2}]}]}
        called = []
        def predict():
            """테스트 호출 횟수를 기록한다."""
            called.append(1)
            return {'raw_latex': '1+1=3'}
        first = cache.complete('event', raw, {}, predict)
        first['raw_ink'].clear()
        second = cache.complete('event', raw, {}, predict)
        self.assertEqual(len(called), 1)
        self.assertEqual(second['raw_latex'], '1+1=3')
        self.assertEqual(second['raw_ink'], raw['strokes'])
        with self.assertRaises(ValueError):
            cache.complete('event', {'strokes': []}, {}, predict)

    def test_failures_stay_in_denominator(self):
        """검토·실패·미지원 표본을 제외해 정확도를 높이지 않는다."""
        metrics = formula_metrics([dict(target_latex='x', raw_latex='x'),
            dict(target_latex='y', raw_latex=None, failure_reason='unsupported', review_requested=True),
            dict(target_latex=None, raw_latex='z')])
        self.assertEqual(metrics['input_formulas'], 3)
        self.assertEqual(metrics['annotated_formulas'], 2)
        self.assertEqual(metrics['e2e_normalized_latex_exact'], .5)

    def test_risk_does_not_use_truth(self):
        """정답·writer·분할 메타데이터를 바꿔도 위험 특징은 동일하다."""
        row = dict(adapter_scores=[.2, .8], adapter_token='x', baseline_token='y')
        before = risk_features(row)
        row.update(label='nonsense', writer_group='changed', split='test')
        self.assertEqual(before, risk_features(row))

    def test_explicit_root_annotation(self):
        """Unicode prompt의 루트 범위는 괄호를 보존하여 변환한다."""
        self.assertEqual(display_to_latex('√(a)'), r'\sqrt{(a)}')
        self.assertEqual(normalize_latex(display_to_latex('x² + 1')), r'x^{2}+1')

    def test_secondary_root_cannot_change_selected_symbol(self):
        """선택되지 않은 루트 후보 때문에 실제 선택 token을 루트로 만들지 않는다."""
        def sample(key, box, candidates, probabilities):
            """serializer 입력의 최소 online 증거를 만든다."""
            return dict(record_id=key, candidates=candidates, raw_box=box,
                row=dict(record_id=key, formula_id='f', geometry=box, final_topk=candidates,
                         final_topk_probabilities=probabilities))
        large = dict(left=0., top=0., right=30., bottom=30.)
        small = dict(left=10., top=10., right=20., bottom=25.)
        latex, _, _ = layout_output([sample('a', large, ['x', r'\sqrt'], [.51, .49]), sample('b', small, ['a'], [1.])], [0, 0])
        self.assertNotIn(r'\sqrt', latex)


if __name__ == '__main__':
    unittest.main()
