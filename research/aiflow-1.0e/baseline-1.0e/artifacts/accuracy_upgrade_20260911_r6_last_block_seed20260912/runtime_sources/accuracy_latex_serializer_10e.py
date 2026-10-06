"""이미 선택한 기호·구조를 유지하면서 LaTeX 제어어 경계를 보존한다."""
from collections import defaultdict
import re

VERSION = 'selected-graph-control-word-boundaries/v1'


def join_latex(parts):
    """제어어와 뒤따르는 Latin 문자가 다른 토큰이면 구분 공백을 넣는다."""
    result = ''
    for part in parts:
        if re.search(r'(?<!\\)\\[A-Za-z]+$', result) and re.match(r'[A-Za-z]', part):
            result += ' '
        result += part
    return result


def serialize_selected_graph(rows, predictions, graph):
    """정답이나 산술 의미를 보지 않고 예측 구조 그래프만 직렬화한다."""
    by_id = {r['record_id']:r for r in rows}
    if len(by_id) != len(rows) or set(predictions) != set(by_id):
        raise ValueError('serializer input coverage mismatch')
    children = defaultdict(lambda:defaultdict(list)); has_parent=set()
    for edge in graph['edges']:
        parent,child,kind=edge['parent'],edge['child'],edge['type']
        if parent not in by_id or child not in by_id or parent==child:
            raise ValueError('invalid structure edge')
        if kind not in ('above','below','contains','superscript','subscript'):
            raise ValueError('unsupported structure relation')
        children[parent][kind].append(child);has_parent.add(child)
    emitted=set()

    def sequence(ids,active):
        """읽기 순서를 원본 x 좌표로 고정하고 token 경계를 보존한다."""
        ordered=sorted(set(ids),key=lambda key:(float(by_id[key]['geometry']['left']),key))
        return join_latex(node(key,active) for key in ordered)

    def node(key,active):
        """이미 예측된 분수·루트·첨자만 출력하며 순환을 거부한다."""
        if key in active:
            raise ValueError('structure cycle')
        if key in emitted:
            return ''
        emitted.add(key); active=active|{key}; slots=children.get(key,{})
        above,below,inside=slots.get('above',[]),slots.get('below',[]),slots.get('contains',[])
        if above and below:
            value=r'\frac{'+sequence(above,active)+'}{'+sequence(below,active)+'}'
        elif inside:
            value=r'\sqrt{'+sequence(inside,active)+'}'
        else:
            value=str(predictions[key])
        if slots.get('subscript'):
            value+='_{'+sequence(slots['subscript'],active)+'}'
        if slots.get('superscript'):
            value+='^{'+sequence(slots['superscript'],active)+'}'
        return value

    # 순환이 roots에서 고립되어 있어도 누락으로 성공시키지 않는다.
    result=sequence(set(by_id)-has_parent,frozenset())
    if emitted!=set(by_id):
        raise ValueError('unemitted or cyclic structure nodes')
    return result
