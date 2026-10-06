"""이미 선택한 기호·구조를 유지하면서 LaTeX 제어어 경계를 보존한다."""
from collections import defaultdict
import re

VERSION = 'selected-graph-validated-tree/v2'
ROOT_TOKENS = frozenset({r'\sqrt', r'\sqrt{}'})
FRACTION_BAR_TOKENS = frozenset({'-', r'\frac'})


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
    raw_edges=[]; seen_edges=set(); region_children=defaultdict(set)
    for edge in graph['edges']:
        parent,child,kind=edge['parent'],edge['child'],edge['type']
        if parent not in by_id or child not in by_id or parent==child:
            raise ValueError('invalid structure edge')
        if kind not in ('above','below','contains','superscript','subscript'):
            raise ValueError('unsupported structure relation')
        signature = (parent, child, kind)
        if signature in seen_edges:
            raise ValueError('duplicate structure edge')
        seen_edges.add(signature); raw_edges.append(signature)
        if kind == 'contains':
            region_children[parent].add(child)
    def region_contains(ancestor, node, active=frozenset()):
        """영역 포함의 전이 경로만 확인하며 순환은 뒤의 tree 검사로 넘긴다."""
        if ancestor in active:
            return False
        return node in region_children[ancestor] or any(
            region_contains(child, node, active | {ancestor}) for child in region_children[ancestor])

    parents=defaultdict(list)
    for parent, child, kind in raw_edges:
        parents[child].append((parent, kind))
    normalized_parents={}
    for child, owners in parents.items():
        # root 영역이 직접 base의 첨자까지 포괄한 간접 contains는 더 가까운
        # 출력 부모 관계가 있으면 제거한다. 서로 무관한 부모 충돌은 거부한다.
        direct=[owner for owner in owners if not (owner[1] == 'contains' and any(
            other[0] != owner[0] and region_contains(owner[0], other[0]) for other in owners))]
        if len(direct) != 1:
            raise ValueError(f'multiple structure parents: {child}')
        normalized_parents[child]=direct[0]
    children = defaultdict(lambda:defaultdict(list))
    for child, (parent, kind) in normalized_parents.items():
        children[parent][kind].append(child)
    has_parent=set(normalized_parents)
    for parent, slots in children.items():
        above, below, inside = slots.get('above', []), slots.get('below', []), slots.get('contains', [])
        if bool(above) != bool(below):
            raise ValueError('incomplete fraction structure')
        if above and predictions[parent] not in FRACTION_BAR_TOKENS:
            raise ValueError('fraction structure requires a fraction-bar token')
        if inside and predictions[parent] not in ROOT_TOKENS:
            raise ValueError('contains structure requires a root token')
        if inside and (above or below):
            raise ValueError('one token cannot be both root and fraction')
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
            raise ValueError('structure node emitted more than once')
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
