"""실패를 분모에 남기는 문자·원본 획 평가와 위험 guard."""
from __future__ import annotations

from collections import defaultdict
import math
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from accuracy_upgrade_contract_v1 import normalize_latex


def glyph_metrics(rows: list[dict], key='adapter_token') -> dict:
    """정답 grouping을 제공한 문자 지표임을 명시한다."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row['writer_group'], row['formula_id'])].append(row)
    correct = sum(row.get(key) == row['label'] for row in rows)
    exact = sum(all(r.get(key) == r['label'] for r in group) for group in groups.values())
    return dict(glyph_correct=correct, glyph_total=len(rows), glyph_top1=correct / len(rows) if rows else 0.,
        glyph_formula_correct=exact, formula_total=len(groups), glyph_formula_exact=exact / len(groups) if groups else 0.,
        improvements=sum(r['baseline_token'] != r['label'] and r.get(key) == r['label'] for r in rows),
        regressions=sum(r['baseline_token'] == r['label'] and r.get(key) != r['label'] for r in rows),
        formula_regressions=sum(all(r['baseline_token'] == r['label'] for r in group) and not all(r.get(key) == r['label'] for r in group) for group in groups.values()),
        candidate_recall=sum(r['label'] in r['candidates'] for r in rows) / len(rows) if rows else 0.,
        metric_scope='truth_grouped_glyph_diagnostic')


def formula_metrics(rows: list[dict]) -> dict:
    """전체 입력 중 정답 있는 항목의 실패·검토 요청을 모두 오답에 포함한다."""
    eligible = [r for r in rows if r.get('target_latex') is not None]
    correct = sum(not r.get('failure_reason') and normalize_latex(r.get('raw_latex') or '') == normalize_latex(r['target_latex']) for r in eligible)
    raw = sum(not r.get('failure_reason') and r.get('raw_latex') == r['target_latex'] for r in eligible)
    return dict(input_formulas=len(rows), annotated_formulas=len(eligible), unannotated_formulas=len(rows) - len(eligible),
                e2e_correct=correct, e2e_normalized_latex_exact=correct / len(eligible) if eligible else None,
                raw_latex_correct=raw, raw_latex_exact=raw / len(eligible) if eligible else None,
                failures=sum(bool(r.get('failure_reason')) for r in rows), review_requests=sum(bool(r.get('review_requested')) for r in rows),
                grouping_exact=sum(bool(r.get('grouping_exact')) for r in eligible) / len(eligible) if eligible else None,
                structure_exact=sum(bool(r.get('structure_exact')) for r in eligible) / len(eligible) if eligible else None,
                structure_annotation_supported=sum(bool(r.get('structure_annotation_supported')) for r in eligible),
                structure_metric_scope='all-input conservative diagnostic; missing complex relations count as incorrect')


def risk_features(row: dict) -> list[float]:
    """정답이나 writer를 읽지 않고 online 확률·일치·coverage만 사용한다."""
    scores = np.asarray(row['adapter_scores'], np.float64)
    scores = scores - scores.max()
    probability = np.exp(scores); probability /= probability.sum()
    baseline = np.asarray(row.get('baseline_probabilities') or [1. / len(scores)] * len(scores), np.float64)
    baseline /= max(float(baseline.sum()), 1e-12)
    ordered = np.sort(probability)
    values = [float(probability.max()), float(ordered[-1] - ordered[-2]) if len(ordered) > 1 else 1.,
              float(-(probability * np.log(np.maximum(probability, 1e-12))).sum()), float(baseline.max()),
              float(-(baseline * np.log(np.maximum(baseline, 1e-12))).sum()),
              float(row['adapter_token'] == row['baseline_token']), float(row.get('coverage_probability', 0.)),
              float(row.get('geometry_missing', False))]
    if not all(math.isfinite(v) for v in values):
        raise ValueError('non-finite risk features')
    return values


def fit_guard(rows: list[dict]) -> tuple[dict, dict]:
    """outer 제외 inner OOF에서 위험 모델을 학습하고 무회귀 경계를 고른다."""
    changes = [r for r in rows if r['adapter_token'] != r['baseline_token']]
    labels = [r['adapter_token'] == r['label'] for r in changes]
    if len(set(labels)) < 2:
        return dict(model=None, threshold=None), dict(status='no_two_class_calibration', changes=len(changes))
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1., max_iter=1000, random_state=20260910))
    model.fit(np.asarray([risk_features(r) for r in changes]), labels)
    probabilities = model.predict_proba(np.asarray([risk_features(r) for r in rows]))[:, 1]
    best = None
    # None이 실제 reject-all이다. 점수 1.0도 통과시키는 가짜 reject-all을 사용하지 않는다.
    for threshold in [None] + sorted(set(map(float, probabilities))):
        predicted = [{**r, 'guarded_token': r['adapter_token'] if threshold is not None and p >= threshold else r['baseline_token']}
                     for r, p in zip(rows, probabilities)]
        metrics = glyph_metrics(predicted, 'guarded_token')
        if metrics['regressions'] or metrics['formula_regressions']:
            continue
        changed = sum(r['guarded_token'] != r['baseline_token'] for r in predicted)
        key = (metrics['glyph_formula_correct'], metrics['glyph_correct'], -changed,
               float('inf') if threshold is None else threshold)
        if best is None or key > best[0]:
            best = key, threshold, metrics
    return dict(model=model, threshold=best[1]), dict(status='fitted', calibration_scope='risk_fit_on_student_inner_oof',
        calibration_rows=len(rows), changes=len(changes), threshold=best[1], calibration=best[2])


def apply_guard(guard: dict, rows: list[dict]) -> list[dict]:
    """원본 정답을 참조하지 않고 허용된 기존 후보 변경만 적용한다."""
    probabilities = np.zeros(len(rows)) if guard['model'] is None else guard['model'].predict_proba(np.asarray([risk_features(r) for r in rows]))[:, 1]
    return [{**row, 'guarded_token': row['adapter_token'] if guard['threshold'] is not None and p >= guard['threshold'] else row['baseline_token'],
             'change_probability': float(p)} for row, p in zip(rows, probabilities)]


def writer_bootstrap(baseline: list[dict], candidate: list[dict], seed=20260910, repeats=10000) -> dict:
    """writer를 복원추출하여 동일 수식의 Exact 차이 구간을 계산한다."""
    before = {r['formula_id']: r for r in baseline}
    after = {r['formula_id']: r for r in candidate}
    if before.keys() != after.keys():
        raise ValueError('paired bootstrap requires identical input coverage')
    grouped = defaultdict(list)
    for key, row in before.items():
        other = after[key]
        if row['writer_group'] != other['writer_group'] or row['target_latex'] != other['target_latex']:
            raise ValueError('paired bootstrap target/writer mismatch')
        if row['target_latex'] is None:
            continue
        old = not row.get('failure_reason') and normalize_latex(row.get('raw_latex') or '') == normalize_latex(row['target_latex'])
        new = not other.get('failure_reason') and normalize_latex(other.get('raw_latex') or '') == normalize_latex(other['target_latex'])
        grouped[row['writer_group']].append(int(new) - int(old))
    values = list(grouped.values())
    if not values:
        raise ValueError('no annotated paired rows')
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(repeats):
        chosen = rng.integers(0, len(values), len(values))
        samples.append(sum(sum(values[i]) for i in chosen) / sum(len(values[i]) for i in chosen))
    return dict(writer_count=len(values), replicates=repeats, seed=seed,
                difference=sum(map(sum, values)) / sum(map(len, values)),
                lower95=float(np.quantile(samples, .025)), upper95=float(np.quantile(samples, .975)))
