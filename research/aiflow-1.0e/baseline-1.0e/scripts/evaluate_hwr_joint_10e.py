"""학습 완료된 outer HWR와 grouping을 고정하고 공동 선택만 비교한다."""
import argparse
import json
import time
from pathlib import Path

import joblib
import torch

from accuracy_upgrade_data_10e import load_corpus, load_hwr, write_json, write_rows
from accuracy_upgrade_contract_v1 import sha256_file
from accuracy_lineage_10e import LineageRegistry
from accuracy_joint_runtime_10e import prepare_formula
from accuracy_evaluation_10e import formula_metrics
from run_accuracy_upgrade_10e import evaluate_raw
from verify_accuracy_experiment_10e import paired_changes, verify


def main():
    """동일 held 입력에 top-1과 joint를 새 추론하며 가중치는 변경하지 않는다."""
    p = argparse.ArgumentParser()
    p.add_argument('--experiment', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--top-k', type=int, choices=[5, 10, 20], default=5)
    p.add_argument('--structure-weight', type=float, choices=[0.5, 1., 2.], default=1.)
    a = p.parse_args()
    if a.output.exists() or a.output.resolve().drive.upper() != 'D:':
        raise ValueError('new D: output required')
    ledger = json.loads((a.experiment / 'experiment.json').read_text(encoding='utf-8'))
    if ledger['status'] != 'completed' or ledger['stage'] != 'hwr':
        raise ValueError('completed HWR experiment required')
    config = ledger['configuration']; scope = ledger['arguments']['scope']
    corpus = load_corpus(config)
    registry = LineageRegistry.load(a.experiment / 'lineage_manifest.json')
    split_payload = json.loads((a.experiment / 'split_manifest.json').read_text(encoding='utf-8'))
    manifest = json.loads((a.experiment / 'model_manifest.json').read_text(encoding='utf-8'))
    cache = json.loads((a.experiment / 'cache_manifest.json').read_text(encoding='utf-8'))
    raw_path = Path(cache['hwr_features']).with_name('raw_formula_features.pt')
    expected = json.loads(raw_path.with_suffix('.sha.json').read_text(encoding='utf-8'))
    if sha256_file(str(raw_path)) != expected['sha256']:
        raise ValueError('source feature cache SHA mismatch')
    payload = torch.load(raw_path, map_location='cpu', weights_only=False)
    if payload['cache_key'] != cache['cache_key']:
        raise ValueError('source cache key mismatch')
    prepared = payload['formulas']; device = torch.device('cpu'); torch.set_num_threads(4)
    a.output.mkdir(parents=True)
    started = time.perf_counter()
    ledger = dict(ledger, status='running', stage='hwr_joint_evaluation', parent_experiment=str(a.experiment.resolve()),
                  command_arguments={'top_k': a.top_k, 'structure_weight': a.structure_weight}, device='cpu')
    write_json(a.output / 'experiment.json', ledger)
    source_root = registry.add('frozen_raw_feature_cache', raw_path, cache_key=cache['cache_key'])
    decoder_root = registry.add('joint_decoder', Path(__file__).with_name('accuracy_joint_runtime_10e.py'),
                                top_k=a.top_k, structure_weight=a.structure_weight)
    serializer_root = registry.add('latex_serializer', Path(__file__).with_name('accuracy_latex_serializer_10e.py'))
    before, after = [], []
    try:
        for outer, split in enumerate(split_payload['splits']):
            model_path = (a.experiment / 'models' / f'outer{outer}_hwr_refit.pt').resolve()
            group_path = (a.experiment / 'models' / f'outer{outer}_raw_grouping.joblib').resolve()
            def find_root(path):
                """저장 경로와 일치하는 실제 학습 계보 노드를 찾는다."""
                matches = [k for k, node in registry.nodes.items() if node['artifact_path'] == str(path)]
                if len(matches) != 1:
                    raise ValueError('ambiguous checkpoint lineage')
                return matches[0]
            hwr_root, group_root = find_root(model_path), find_root(group_path)
            excluded = {registry.records[k]['formula_key'] for k in split['held_ids']}
            for root in (hwr_root, group_root):
                registry.validate(root, {split['outer_writer']}, excluded)
            model, labels = load_hwr(model_path, device)
            if labels != manifest['vocabulary']:
                raise ValueError('vocabulary mismatch')
            group = joblib.load(group_path)
            ids = {registry.records[k]['formula_id'] for k in split['held_ids']}
            refreshed = {}
            with torch.inference_mode():
                for key in sorted(ids):
                    if scope == 'head':
                        item = prepared[key]
                        refreshed[key] = dict(item, logits=model.math_head(torch.from_numpy(item['ink'])).numpy())
                    else:
                        refreshed[key] = prepare_formula(dict(formula_id=key, strokes=corpus.formulas[key]['strokes']),
                            model, labels, device, config['input_mode'], 6, 4)
            base = evaluate_raw(corpus, refreshed, group, ids, labels, device, k=a.top_k, joint=False)
            rows = evaluate_raw(corpus, refreshed, group, ids, labels, device, k=a.top_k,
                                joint=True, structure_weight=a.structure_weight)
            for row in base + rows:
                row.update(hwr_lineage_root_id=hwr_root, grouping_lineage_root_id=group_root,
                           feature_lineage_root_id=source_root, decoder_lineage_root_id=decoder_root,
                           serializer_lineage_root_id=serializer_root)
            before.extend(base); after.extend(rows)
            print(json.dumps(dict(outer=outer, baseline=formula_metrics(base), joint=formula_metrics(rows))), flush=True)
        if len(after) != 95 or len({r['formula_id'] for r in after}) != 95:
            raise ValueError('incomplete formula coverage')
        write_rows(a.output / 'formula_predictions.jsonl.gz', after)
        write_rows(a.output / 'baseline_formula_predictions.jsonl.gz', before)
        write_json(a.output / 'lineage_manifest.json', registry.payload())
        write_json(a.output / 'split_manifest.json', split_payload)
        manifest.update(model_id='frozen_hwr_joint_diagnostic', top_k=a.top_k, structure_weight=a.structure_weight,
                        evaluation_script_sha256=sha256_file(__file__), runtime_sha256=sha256_file(str(Path(__file__).with_name('accuracy_joint_runtime_10e.py'))),
                        serializer_sha256=sha256_file(str(Path(__file__).with_name('accuracy_latex_serializer_10e.py'))))
        write_json(a.output / 'model_manifest.json', manifest)
        report = dict(status='completed', training_performed=False, baseline_e2e=formula_metrics(before),
                      raw_e2e=formula_metrics(after), paired=paired_changes(before, after),
                      data_role='consumed_development', configuration_selection='none; fixed diagnostic setting',
                      elapsed_seconds=time.perf_counter()-started, product_activated=False, uploaded=False)
        write_json(a.output / 'evaluation.json', report)
        ledger.update(status='completed', exit_code=0); write_json(a.output / 'experiment.json', ledger)
        write_json(a.output / 'verification.json', verify(a.output))
        print(json.dumps(report), flush=True)
    except Exception as error:
        ledger.update(status='failed', exit_code=1, error=str(error))
        write_json(a.output / 'experiment.json', ledger)
        raise


if __name__ == '__main__':
    main()
