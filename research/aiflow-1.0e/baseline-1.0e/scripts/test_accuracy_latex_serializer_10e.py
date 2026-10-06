"""기호를 바꾸지 않는 출력 경계·구조 검증 테스트."""
import unittest
from accuracy_latex_serializer_10e import join_latex,serialize_selected_graph


class SerializationTests(unittest.TestCase):
    """산술 수정·기호 통합·관계 순환을 허용하지 않는다."""
    def test_control_word_boundary(self):
        """곱셈 기호 뒤 Latin 문자가 매크로 이름으로 붙지 않는다."""
        self.assertEqual(join_latex(['4',r'\times','a','+','1']),r'4\times a+1')
        self.assertEqual(join_latex([r'\alpha','x']),r'\alpha x')
        self.assertEqual(join_latex([r'\frac{1}{2}','x']),r'\frac{1}{2}x')

    def test_false_arithmetic_preserved(self):
        """실제로 쓴 오답 수식을 정답 수식으로 바꾸지 않는다."""
        tokens=list('1+1=3');rows=[dict(record_id=str(i),geometry=dict(left=i)) for i in range(len(tokens))]
        self.assertEqual(serialize_selected_graph(rows,dict(zip(map(str,range(len(tokens))),tokens)),{'edges':[]}), '1+1=3')

    def test_isolated_cycle_rejected(self):
        """root가 없는 순환도 빈 성공 출력으로 처리하지 않는다."""
        rows=[dict(record_id=k,geometry=dict(left=i)) for i,k in enumerate(['a','b'])]
        edges=[dict(parent='a',child='b',type='superscript'),dict(parent='b',child='a',type='subscript')]
        with self.assertRaises(ValueError):
            serialize_selected_graph(rows,{'a':'x','b':'2'},{'edges':edges})

    def test_nested_fraction_root_and_scripts(self):
        """분수 안 루트와 아래·위 첨자를 선택된 관계대로 보존한다."""
        labels={'bar':'-', 'root':r'\sqrt', 'x':'x', 'two':'2', 'n':'n', 'one':'1'}
        rows=[dict(record_id=k,geometry=dict(left=i)) for i,k in enumerate(labels)]
        edges=[dict(parent=a,child=b,type=t) for a,b,t in [
            ('bar','root','above'),('bar','one','below'),('root','x','contains'),
            ('x','n','subscript'),('x','two','superscript')]]
        self.assertEqual(serialize_selected_graph(rows,labels,{'edges':edges}),r'\frac{\sqrt{x_{n}^{2}}}{1}')

    def test_multiple_direct_parents_rejected(self):
        """하나의 기호가 서로 다른 출력 부모에 중복 귀속되지 않는다."""
        rows=[dict(record_id=k,geometry=dict(left=i)) for i,k in enumerate(['a','b','c'])]
        edges=[dict(parent='a',child='c',type='superscript'),dict(parent='b',child='c',type='superscript')]
        with self.assertRaisesRegex(ValueError,'multiple structure parents'):
            serialize_selected_graph(rows,{'a':'x','b':'y','c':'2'},{'edges':edges})

    def test_transitive_region_parent_is_reduced_to_direct_tree_parent(self):
        """root의 간접 영역 포함과 내부 base의 첨자 관계를 중복 출력하지 않는다."""
        labels={'root':r'\sqrt','x':'x','two':'2'}
        rows=[dict(record_id=k,geometry=dict(left=i)) for i,k in enumerate(labels)]
        edges=[dict(parent='root',child='x',type='contains'),
               dict(parent='root',child='two',type='contains'),
               dict(parent='x',child='two',type='superscript')]
        self.assertEqual(serialize_selected_graph(rows,labels,{'edges':edges}),r'\sqrt{x^{2}}')

    def test_structure_parent_token_contract(self):
        """루트·분수 구조는 해당 의미를 가진 선택 기호만 부모가 된다."""
        rows=[dict(record_id=k,geometry=dict(left=i)) for i,k in enumerate(['base','child','den'])]
        predictions={'base':'x','child':'1','den':'2'}
        with self.assertRaisesRegex(ValueError,'root token'):
            serialize_selected_graph(rows,predictions,{'edges':[dict(parent='base',child='child',type='contains')]})
        with self.assertRaisesRegex(ValueError,'fraction-bar token'):
            serialize_selected_graph(rows,predictions,{'edges':[
                dict(parent='base',child='child',type='above'),dict(parent='base',child='den',type='below')]})

    def test_incomplete_fraction_rejected(self):
        """분자 또는 분모만 있는 구조는 정상 수식으로 직렬화하지 않는다."""
        rows=[dict(record_id=k,geometry=dict(left=i)) for i,k in enumerate(['bar','num'])]
        with self.assertRaisesRegex(ValueError,'incomplete fraction'):
            serialize_selected_graph(rows,{'bar':'-','num':'1'},{'edges':[dict(parent='bar',child='num',type='above')]})


if __name__=='__main__':
    unittest.main()
