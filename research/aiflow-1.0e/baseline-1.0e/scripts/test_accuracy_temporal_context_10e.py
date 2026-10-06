"""원본 시간 결측·지연 획·기하 불변성과 소유권 오류를 검사한다."""
import copy
import unittest
from accuracy_temporal_context_10e import temporal_context


class TemporalTests(unittest.TestCase):
    """평가 성능이 아닌 원본 입력 계약만 검증한다."""
    def setUp(self):
        """마지막에 찍은 점이 첫 문자에 속하는 짧은 입력을 만든다."""
        self.strokes = [dict(order=i, points=[dict(x=i, y=i, t_ms=10*i)]) for i in range(3)]
        self.groups = [[0, 2], [1]]
        self.boxes = [dict(left=1., right=4., top=1., bottom=5.), dict(left=3., right=6., top=1., bottom=5.)]

    def test_delayed_stroke_and_overlap(self):
        """뒤늦게 쓴 획을 삭제하지 않고 음수 간격으로 표현한다."""
        result = temporal_context(self.strokes, self.groups, self.boxes, 0)
        self.assertEqual(result['next_order_gap'], -.5)
        self.assertEqual(result['next_time_gap'], -.5)
        self.assertAlmostEqual(result['next_x_overlap'], 1/3)

    def test_translation_scale_and_missing_time(self):
        """공간 이동·크기 변경은 특징을 보존하고 없는 시간은 마스킹한다."""
        first = temporal_context(self.strokes, self.groups, self.boxes, 0)
        changed = [{k: v*3+10 for k, v in box.items()} for box in self.boxes]
        self.assertEqual(first, temporal_context(self.strokes, self.groups, changed, 0))
        no_time = copy.deepcopy(self.strokes)
        no_time[0]['points'][0].pop('t_ms')
        second = temporal_context(no_time, self.groups, self.boxes, 0)
        self.assertEqual(second['time_available'], 0)
        self.assertEqual(second['next_time_gap'], 0)

    def test_ownership_fail_closed(self):
        """소유권 누락과 중복은 입력 전에 거부한다."""
        for groups in ([[0], [1]], [[0, 1], [1, 2]]):
            with self.assertRaises(ValueError):
                temporal_context(self.strokes, groups, self.boxes, 0)


if __name__ == '__main__':
    unittest.main()
