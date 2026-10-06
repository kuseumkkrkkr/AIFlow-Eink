"""실험 종료 코드 외에 전체 예측·계보·회귀·승격 조건을 독립 검증한다."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from accuracy_lineage_10e import LineageRegistry, validate_prediction
from accuracy_upgrade_contract_v1 import normalize_latex, sha256_file
from accuracy_upgrade_data_10e import write_json
from online_candidate_features_10e import _json_lines
from accuracy_evaluation_10e import formula_metrics, glyph_metrics, writer_bootstrap


def paired_changes(before, after):
    """완전히 같은 입력 ID·정답·writer에 대해서만 개선과 회귀를 센다."""
    a, b = {r['formula_id']: r for r in before}, {r['formula_id']: r for r in after}
    if len(a) != len(before) or len(b) != len(after) or a.keys() != b.keys():
        raise ValueError('duplicate or different formula coverage')
    improvement, regression = [], []
    for key, row in a.items():
        new = b[key]
        if row['writer_group'] != new['writer_group'] or row['target_latex'] != new['target_latex']:
            raise ValueError('paired identity mismatch')
        old_ok = not row.get('failure_reason') and normalize_latex(row.get('raw_latex') or '') == normalize_latex(row['target_latex'])
        new_ok = not new.get('failure_reason') and normalize_latex(new.get('raw_latex') or '') == normalize_latex(new['target_latex'])
        if old_ok and not new_ok:
            regression.append(key)
        elif new_ok and not old_ok:
            improvement.append(key)
    return dict(improvement_ids=improvement, regression_ids=regression, improvements=len(improvement), regressions=len(regression))


def verify(directory: Path) -> dict:
    """모든 outer·inner 예측과 ancestor 실물을 재검증하고 제품 승격은 별도 차단한다."""
    ledger = json.loads((directory / 'experiment.json').read_text(encoding='utf-8'))
    if ledger['status'] != 'completed' or ledger['exit_code'] != 0:
        raise ValueError('experiment did not complete successfully')
    registry = LineageRegistry.load(directory / 'lineage_manifest.json')
    splits = json.loads((directory / 'split_manifest.json').read_text(encoding='utf-8'))['splits']
    expected_glyphs = {key for split in splits for key in split['held_ids']}
    expected_formulas = {registry.records[k]['formula_id'] for k in expected_glyphs}
    formulas = _json_lines(directory / 'formula_predictions.jsonl.gz')
    if {r['formula_id'] for r in formulas} != expected_formulas or len(formulas) != len(expected_formulas):
        raise ValueError('incomplete raw input coverage')
    glyph_path = directory / 'glyph_predictions.jsonl.gz'
    glyphs = _json_lines(glyph_path) if glyph_path.exists() else []
    if glyphs and ({r['record_id'] for r in glyphs} != expected_glyphs or len(glyphs) != len(expected_glyphs)):
        raise ValueError('incomplete glyph coverage')
    for split in splits:
        writer = split['outer_writer']
        keys = {registry.records[k]['formula_key'] for k in split['held_ids']}
        for row in glyphs:
            if row['record_id'] not in split['held_ids']:
                continue
            provenance = row.get('prediction_provenance')
            if provenance:
                validate_prediction(row, writer, registry)
                if provenance.get('guard_root_id'):
                    registry.validate(provenance['guard_root_id'], {writer}, keys)
            else:
                registry.validate(row['hwr_lineage_root_id'], {writer}, keys)
        for row in formulas:
            if row['writer_group'] != writer:
                continue
            roots = [value for key, value in row.items() if key.endswith('lineage_root_id')]
            if not roots:
                raise ValueError('raw output lacks actual lineage root')
            for root in roots:
                registry.validate(root, {writer}, keys)
    inner_path = directory / 'inner_predictions.jsonl.gz'
    inner = _json_lines(inner_path) if inner_path.exists() else []
    for row in inner:
        validate_prediction(row, row['prediction_provenance']['outer_writer'], registry)
    result = dict(status='completed', verified=True, input_formulas=len(formulas), glyph_rows=len(glyphs),
                  inner_rows=len(inner), raw=formula_metrics(formulas), source_experiment_sha256=sha256_file(str(directory / 'experiment.json')),
                  evaluation_role='consumed_development', product_promotable=False)
    baseline_path = directory / 'baseline_formula_predictions.jsonl.gz'
    if baseline_path.exists():
        baseline = _json_lines(baseline_path)
        result['paired_raw'] = paired_changes(baseline, formulas)
        result['development_writer_bootstrap'] = writer_bootstrap(baseline, formulas)
        guarded_path = directory / 'guarded_formula_predictions.jsonl.gz'
        if guarded_path.exists():
            guarded = _json_lines(guarded_path)
            if {r['formula_id'] for r in guarded} != expected_formulas or len(guarded) != len(expected_formulas):
                raise ValueError('incomplete guarded coverage')
            for row in guarded:
                for key, root in row.items():
                    if key.endswith('lineage_root_id'):
                        writer = row['writer_group']
                        excluded = {r['formula_key'] for r in registry.records.values() if r['writer'] == writer}
                        registry.validate(root, {writer}, excluded)
            result['paired_guarded'] = paired_changes(baseline, guarded)
            result['guarded'] = formula_metrics(guarded)
    result['promotion_blockers'] = ['no newly sealed writer/formula/device data', 'no real-device acceptance proof',
                                    'full commercial attribution and data-rights release review outstanding']
    if glyphs:
        result['glyph'] = glyph_metrics(glyphs)
    return result


def main():
    """학습 산출물은 변경하지 않고 별도 검증 결과만 새 경로에 기록한다."""
    p = argparse.ArgumentParser()
    p.add_argument('--experiment', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    result = verify(a.experiment)
    write_json(a.output, result)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
