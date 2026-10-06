"""원본 획 순서·시간·이웃 겹침을 결측 마스크와 함께 계산한다."""
from __future__ import annotations

import math

KEYS = ('previous_x_overlap', 'next_x_overlap', 'previous_y_overlap', 'next_y_overlap',
        'stroke_first_rel', 'stroke_last_rel', 'previous_order_gap', 'next_order_gap',
        'previous_time_gap', 'next_time_gap', 'time_available', 'source_order_available')


def temporal_context(strokes: list[dict], groups: list[list[int]], boxes: list[dict], index: int) -> dict:
    """정답 없이 선택된 grouping의 원본 계보만 읽는다; 지연 획도 제거하지 않는다."""
    if len(groups) != len(boxes) or not 0 <= index < len(groups):
        raise ValueError('temporal grouping/box alignment mismatch')
    flat = [i for group in groups for i in group]
    if len(flat) != len(set(flat)) or sorted(flat) != list(range(len(strokes))):
        raise ValueError('temporal features require exact source ownership')
    result = dict.fromkeys(KEYS, 0.)
    result['source_order_available'] = 1.
    scale = max(len(strokes) - 1, 1)
    result['stroke_first_rel'] = min(groups[index]) / scale
    result['stroke_last_rel'] = max(groups[index]) / scale
    current = boxes[index]

    def overlap(other, axis):
        """각 문자 크기에 대한 실제 겹침 비율은 이동·등방 확대에 불변이다."""
        low, high = ('left', 'right') if axis == 'x' else ('top', 'bottom')
        width = min(current[high] - current[low], other[high] - other[low])
        return max(0., min(current[high], other[high]) - max(current[low], other[low])) / max(width, 1e-9)

    times = []
    available = True
    for stroke in strokes:
        values = [point.get('t_ms') for point in stroke['points']]
        if not values or any(value is None or not math.isfinite(float(value)) for value in values):
            available = False
            times.append(None)
        else:
            times.append((min(map(float, values)), max(map(float, values))))
    if available:
        low = min(t[0] for t in times); high = max(t[1] for t in times)
        available = high > low
    result['time_available'] = float(available)
    spans = [(min(times[i][0] for i in group), max(times[i][1] for i in group)) for group in groups] if available else None
    for prefix, other_index in (('previous', index - 1), ('next', index + 1)):
        if not 0 <= other_index < len(groups):
            continue
        result[prefix + '_x_overlap'] = overlap(boxes[other_index], 'x')
        result[prefix + '_y_overlap'] = overlap(boxes[other_index], 'y')
        result[prefix + '_order_gap'] = ((min(groups[index]) - max(groups[other_index])) if prefix == 'previous'
                                         else (min(groups[other_index]) - max(groups[index]))) / scale
        if available:
            result[prefix + '_time_gap'] = ((spans[index][0] - spans[other_index][1]) if prefix == 'previous'
                                            else (spans[other_index][0] - spans[index][1])) / (high - low)
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError('nonfinite temporal context')
    return result
