"""외부 기반 HWR 미세조정의 7-writer nested 연구 진단을 실행한다."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from run_accuracy_upgrade_10e import Experiment, evaluate_raw, event, ROOT
from accuracy_hwr_finetune_10e import fit_hwr, load_replay
from accuracy_upgrade_data_10e import encode_rows, build_samples, write_json, write_rows
from accuracy_upgrade_contract_v1 import sha256_file
from accuracy_joint_runtime_10e import prepare_formula
from accuracy_evaluation_10e import formula_metrics, glyph_metrics


@torch.inference_mode()
def refreshed(exp, model, ids, scope):
    """동일 가중치의 embedding과 logits를 함께 갱신하고 정답은 입력하지 않는다."""
    result = {}
    for key in sorted(ids):
        if scope == 'head':
            item = exp.prepared[key]
            scores = model.math_head(torch.from_numpy(item['ink']).to(exp.device)).cpu().numpy()
            result[key] = {**item, 'logits': scores}
        else:
            source = exp.corpus.formulas[key]
            result[key] = prepare_formula(dict(formula_id=key, strokes=source['strokes']), model,
                                         exp.labels, exp.device, exp.config['input_mode'], exp.args.window, exp.args.neighbors)
    return result


def save_model(exp, model, report, name, fit, parents, selection=()):
    """각 refit 및 선택 가중치와 실제 감독학습 조상을 보존한다."""
    path = exp.output / 'models' / (name + '.pt')
    torch.save(dict(state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                    math_labels=exp.labels, auxiliary_labels=[], report=report), path)
    root = exp.registry.add(name, path, fit, parents, selection_ids=selection,
                            replay_report=report['external_replay_indices'])
    return root


def run(exp, scope, replay_cache):
    """inner 3-fold에서 epoch를 고르고 outer writer를 한 번씩 평가한다."""
    exp.prepare()
    _features, _labels, replay_manifest = load_replay(replay_cache, exp.labels)
    replay_root = exp.registry.add('external_training_replay', replay_cache / 'cache_manifest.json',
        source_roles='external training only', sources=replay_manifest['sets']['math_train']['sources'],
        features_sha256=sha256_file(str(replay_cache / replay_manifest['sets']['math_train']['features'])),
        labels_sha256=sha256_file(str(replay_cache / replay_manifest['sets']['math_train']['labels'])))
    all_raw, all_base, all_glyph, folds = [], [], [], []
    for outer, split in enumerate(exp.splits):
        selectors, epochs, inner_reports = [], [], []
        for inner in split['inner']:
            group, group_root = exp.fit_group(inner['fit_ids'], f'o{outer}_i{inner["index"]}')
            validation_formulas = {exp.corpus.records[k]['formula_id'] for k in inner['held_ids']}

            def score(model):
                """inner 실제 원본 획 Exact와 구조를 epoch 선택에 사용한다."""
                prepared = refreshed(exp, model, validation_formulas, scope)
                rows = evaluate_raw(exp.corpus, prepared, group, validation_formulas, exp.labels, exp.device)
                metrics = formula_metrics(rows)
                return metrics['e2e_correct'], metrics['structure_exact'] or 0.

            model, labels, report = fit_hwr(exp.corpus, inner['fit_ids'], inner['held_ids'], exp.config,
                exp.device, scope, replay_cache, validation_score=score)
            root = save_model(exp, model, report, f'o{outer}_i{inner["index"]}_hwr_select', inner['fit_ids'],
                              [exp.base_node, replay_root, group_root], inner['held_ids'])
            exp.assert_held(root, split['held_ids'])
            selectors.append(root); epochs.append(report['selected_epoch']); inner_reports.append(report)
            event('hwr_inner_complete', outer=outer, inner=inner['index'], epoch=report['selected_epoch'])
            del model
        # 사전에 고정한 중앙값 규칙이며 outer 점수로 epoch를 변경하지 않는다.
        fixed = int(np.median(epochs))
        model, labels, report = fit_hwr(exp.corpus, split['fit_ids'], [], exp.config, exp.device,
                                       scope, replay_cache, fixed_epochs=fixed)
        root = save_model(exp, model, report, f'outer{outer}_hwr_refit', split['fit_ids'],
                          [exp.base_node, replay_root] + selectors)
        exp.assert_held(root, split['held_ids'])
        group, group_root = exp.fit_group(split['fit_ids'], f'outer{outer}_raw')
        exp.assert_held(group_root, split['held_ids'])
        ids = {exp.corpus.records[k]['formula_id'] for k in split['held_ids']}
        base = evaluate_raw(exp.corpus, exp.prepared, group, ids, labels, exp.device)
        prepared = refreshed(exp, model, ids, scope)
        raw = evaluate_raw(exp.corpus, prepared, group, ids, labels, exp.device)
        vectors, logits = encode_rows(model, {k: exp.corpus.raw[k] for k in split['held_ids']}, exp.config['input_mode'], exp.device)
        # build_samples는 전체 source metadata를 순회하므로 held corpus 범위로 명시적으로 줄인다.
        from dataclasses import replace
        held_corpus = replace(exp.corpus, raw={k: exp.corpus.raw[k] for k in split['held_ids']},
                             records={k: exp.corpus.records[k] for k in split['held_ids']})
        samples = build_samples(held_corpus, vectors, logits, labels, exp.args.top_k)
        glyphs = []
        for key, sample in samples.items():
            glyphs.append(dict(record_id=key, formula_id=sample['formula_id'], writer_group=sample['writer_group'],
                label=sample['label'], candidates=sample['candidates'], baseline_token=exp.samples[key]['candidates'][0],
                adapter_token=sample['candidates'][0], hwr_lineage_root_id=root))
        for row in raw:
            row.update(hwr_lineage_root_id=root, grouping_lineage_root_id=group_root)
        all_raw.extend(raw); all_base.extend(base); all_glyph.extend(glyphs)
        folds.append(dict(writer=split['outer_writer'], selected_epoch=fixed, inner=inner_reports,
                          refit=report, raw=formula_metrics(raw), glyph=glyph_metrics(glyphs)))
        event('hwr_outer_complete', outer=outer, scope=scope, raw=formula_metrics(raw), glyph=glyph_metrics(glyphs))
        del model
    if {r['record_id'] for r in all_glyph} != exp.corpus.original_ids or len(all_glyph) != 387 or len(all_raw) != 95:
        raise ValueError('HWR prediction coverage failed')
    write_rows(exp.output / 'formula_predictions.jsonl.gz', all_raw)
    write_rows(exp.output / 'baseline_formula_predictions.jsonl.gz', all_base)
    write_rows(exp.output / 'glyph_predictions.jsonl.gz', all_glyph)
    exp.finish(dict(status='completed', scope=scope, baseline_e2e=formula_metrics(all_base), raw_e2e=formula_metrics(all_raw),
                    raw_glyph=glyph_metrics(all_glyph), folds=folds, training_performed=True,
                    missing_data={'fresh_online_mix': 'historical project-owned data only; not fresh acceptance'},
                    comparison_scope='bounded historical HWR adaptation diagnostic; not product promotion'))


def main():
    """명시적 scope와 새 출력 폴더를 요구하고 실패를 기록한다."""
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--scope', choices=['head', 'last-block'], required=True)
    p.add_argument('--replay-cache', type=Path, default=ROOT / 'artifacts/unified_head_20260813/unified_math_8ep_full/cache')
    p.add_argument('--cache', type=Path, default=ROOT / 'artifacts/accuracy_upgrade_20260911_cache')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    a = p.parse_args()
    # 공통 manifest의 인수 계약을 맞추고 모든 구성을 원장에 남긴다.
    a.stage='hwr'; a.features='formula28'; a.geometry='restored'; a.top_k=5; a.window=6; a.neighbors=4
    a.replay_cache = str(a.replay_cache)
    exp = Experiment(a, json.loads(a.config.read_text(encoding='utf-8')))
    try:
        run(exp, a.scope, Path(a.replay_cache))
    except Exception as error:
        write_json(exp.output / 'failure.json', dict(status='failed', error=str(error), type=type(error).__name__))
        ledger = json.loads((exp.output / 'experiment.json').read_text(encoding='utf-8'))
        ledger.update(status='failed', exit_code=1); write_json(exp.output / 'experiment.json', ledger)
        raise


if __name__ == '__main__':
    main()
