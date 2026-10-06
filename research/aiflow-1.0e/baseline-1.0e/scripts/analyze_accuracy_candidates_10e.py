"""고정 후보의 정답 포함 상한과 오수정 없는 수정 민감도를 계산한다."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from online_candidate_features_10e import _json_lines
from accuracy_upgrade_data_10e import write_json


def analyze(rows, prediction='adapter_token'):
    """학습이나 후보 변경 없이 저장된 각 수식의 누락·오답 개수를 센다."""
    groups = defaultdict(list)
    if len({r['record_id'] for r in rows}) != len(rows):
        raise ValueError('duplicate prediction IDs')
    for row in rows:
        groups[row['formula_id']].append(row)
    reachable = Counter(); unreachable = []
    for formula_id, values in groups.items():
        if any(row['label'] not in row['candidates'] for row in values):
            unreachable.append(formula_id)
        else:
            reachable[sum(row[prediction] != row['label'] for row in values)] += 1
    denominator = len(groups)
    return dict(glyph_rows=len(rows), formulas=denominator, reachable_error_histogram=dict(reachable),
        unreachable_formula_ids=sorted(unreachable), oracle_formula_correct=sum(reachable.values()),
        oracle_formula_exact=sum(reachable.values())/denominator,
        sensitivity={str(q): sum(n*q**errors for errors,n in reachable.items())/denominator for q in (0.,.25,.5,.75,1.)},
        scope='fixed truth grouping and candidates; not e2e and not a forecast',
        assumptions=['independent correction probability q for each reachable error', 'no new errors', 'unreachable errors never fixed'])


def main():
    """원본 실험을 덮어쓰지 않는 진단 JSON을 만든다."""
    p=argparse.ArgumentParser(); p.add_argument('--predictions',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    result=analyze(_json_lines(a.predictions)); write_json(a.output,result)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    main()
