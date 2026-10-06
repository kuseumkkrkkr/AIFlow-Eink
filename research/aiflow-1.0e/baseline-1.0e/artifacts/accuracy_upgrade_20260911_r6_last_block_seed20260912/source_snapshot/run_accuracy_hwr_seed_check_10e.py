"""선택 epoch·grouping을 고정하고 HWR refit의 seed 민감도만 검사한다."""
import argparse
from dataclasses import replace
import json
import shutil
from pathlib import Path

import joblib

from run_accuracy_upgrade_10e import Experiment, ROOT, evaluate_raw, event
from run_accuracy_hwr_10e import refreshed, save_model
from accuracy_hwr_finetune_10e import fit_hwr
from accuracy_upgrade_data_10e import encode_rows, build_samples, write_json, write_rows
from accuracy_upgrade_contract_v1 import sha256_file, canonical_json_sha256
from accuracy_lineage_10e import LineageRegistry
from accuracy_evaluation_10e import formula_metrics, glyph_metrics
from verify_accuracy_experiment_10e import verify


def unique_root(registry, path):
    """부모 실험의 실제 파일과 일치하는 유일한 계보를 요구한다."""
    keys=[key for key,node in registry.nodes.items() if node['artifact_path']==str(path.resolve())]
    if len(keys)!=1:
        raise ValueError('ambiguous parent model lineage')
    return keys[0]


def main():
    """사전 고정한 두 seed 외에는 허용하지 않으며 outer 점수로 epoch를 바꾸지 않는다."""
    p=argparse.ArgumentParser()
    p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,choices=[20260911,20260912],required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--cache',type=Path,default=ROOT/'artifacts/accuracy_upgrade_20260911_cache')
    a=p.parse_args()
    parent=json.loads((a.experiment/'experiment.json').read_text(encoding='utf-8'))
    if parent['status']!='completed' or parent['stage']!='hwr' or parent['configuration']['seed']!=20260910:
        raise ValueError('completed seed-20260910 HWR selection experiment required')
    source_evaluation=json.loads((a.experiment/'evaluation.json').read_text(encoding='utf-8'))
    source_splits=json.loads((a.experiment/'split_manifest.json').read_text(encoding='utf-8'))['splits']
    registry=LineageRegistry.load(a.experiment/'lineage_manifest.json')
    config=dict(parent['configuration'],seed=a.seed)
    scope=parent['arguments']['scope']
    a.config=a.experiment/'experiment.json';a.stage='hwr_seed_check';a.features='formula28'
    a.geometry='restored';a.top_k=5;a.window=6;a.neighbors=4
    # argparse Path는 실행 원장에서 JSON으로 기록 가능하도록 명시적으로 문자열화한다.
    parent_directory=a.experiment.resolve();a.experiment=str(parent_directory)
    exp=Experiment(a,config)
    exp.model_manifest['code_sha256'][Path(__file__).name]=sha256_file(__file__)
    write_json(exp.output/'model_manifest.json',exp.model_manifest)
    try:
        if canonical_json_sha256(exp.splits)!=canonical_json_sha256(source_splits) or exp.registry.records!=registry.records:
            raise ValueError('seed repetition changed data boundaries')
        exp.registry=registry
        if exp.base_node not in registry.nodes:
            raise ValueError('base provenance changed after selection')
        exp.prepare()
        (exp.output/'models').mkdir()
        source_archive=exp.output/'runtime_sources';source_archive.mkdir()
        for filename in ('accuracy_joint_runtime_10e.py','accuracy_latex_serializer_10e.py'):
            shutil.copy2(Path(__file__).with_name(filename),source_archive/filename)
        runtime_root=registry.add('fixed_joint_runtime',source_archive/'accuracy_joint_runtime_10e.py',top_k=5,structure_weight=1.)
        serializer_root=registry.add('fixed_latex_serializer',source_archive/'accuracy_latex_serializer_10e.py')
        all_raw,all_top1,all_base,all_glyph,folds=[],[],[],[],[]
        model_paths=[];group_paths=[]
        replay=Path(parent['arguments']['replay_cache'])
        for outer,split in enumerate(exp.splits):
            selected=source_evaluation['folds'][outer]
            if selected['writer']!=split['outer_writer']:
                raise ValueError('frozen epoch writer mismatch')
            epochs=int(selected['selected_epoch'])
            original_root=unique_root(registry,parent_directory/'models'/f'outer{outer}_hwr_refit.pt')
            group_path=parent_directory/'models'/f'outer{outer}_raw_grouping.joblib'
            group_root=unique_root(registry,group_path)
            group_paths.append(registry.nodes[group_root])
            exp.assert_held(original_root,split['held_ids']);exp.assert_held(group_root,split['held_ids'])
            model,labels,report=fit_hwr(exp.corpus,split['fit_ids'],[],config,exp.device,scope,replay,fixed_epochs=epochs)
            if labels!=exp.labels:
                raise ValueError('vocabulary changed')
            parents=registry.nodes[original_root]['parents']+[group_root]
            root=save_model(exp,model,report,f'outer{outer}_hwr_refit',split['fit_ids'],parents)
            exp.assert_held(root,split['held_ids']);model_paths.append(registry.nodes[root])
            grouping=joblib.load(group_path)
            ids={exp.corpus.records[key]['formula_id'] for key in split['held_ids']}
            prepared=refreshed(exp,model,ids,scope)
            base=evaluate_raw(exp.corpus,exp.prepared,grouping,ids,labels,exp.device)
            top1=evaluate_raw(exp.corpus,prepared,grouping,ids,labels,exp.device)
            raw=evaluate_raw(exp.corpus,prepared,grouping,ids,labels,exp.device,joint=True)
            for rows,hwr_root in ((base,exp.base_node),(top1,root),(raw,root)):
                for row in rows:
                    row.update(hwr_lineage_root_id=hwr_root,grouping_lineage_root_id=group_root,
                               runtime_lineage_root_id=runtime_root,serializer_lineage_root_id=serializer_root)
            held=replace(exp.corpus,raw={key:exp.corpus.raw[key] for key in split['held_ids']},
                         records={key:exp.corpus.records[key] for key in split['held_ids']})
            vectors,logits=encode_rows(model,held.raw,config['input_mode'],exp.device)
            samples=build_samples(held,vectors,logits,labels,5)
            glyphs=[dict(record_id=key,formula_id=s['formula_id'],writer_group=s['writer_group'],label=s['label'],
                         candidates=s['candidates'],adapter_token=s['candidates'][0],baseline_token=exp.samples[key]['candidates'][0],
                         hwr_lineage_root_id=root) for key,s in samples.items()]
            all_raw.extend(raw);all_top1.extend(top1);all_base.extend(base);all_glyph.extend(glyphs)
            folds.append(dict(writer=split['outer_writer'],frozen_epoch=epochs,report=report,
                              raw=formula_metrics(raw),glyph=glyph_metrics(glyphs)))
            event('seed_refit_outer_complete',seed=a.seed,outer=outer,raw=formula_metrics(raw))
            del model
        if len(all_raw)!=95 or len({r['formula_id'] for r in all_raw})!=95 or len(all_glyph)!=387 or {r['record_id'] for r in all_glyph}!=exp.corpus.original_ids:
            raise ValueError('seed check prediction coverage failed')
        write_rows(exp.output/'formula_predictions.jsonl.gz',all_raw)
        write_rows(exp.output/'top1_formula_predictions.jsonl.gz',all_top1)
        write_rows(exp.output/'baseline_formula_predictions.jsonl.gz',all_base)
        write_rows(exp.output/'glyph_predictions.jsonl.gz',all_glyph)
        exp.model_manifest.update(model_id='fixed_hwr_joint_seed_check',scope=scope,seed=a.seed,
            parent_experiment=str(parent_directory),parent_experiment_sha256=sha256_file(str(parent_directory/'experiment.json')),
            frozen_epochs={r['writer']:r['frozen_epoch'] for r in folds},
            hwr_components=[dict(path=r['artifact_path'],sha256=r['artifact_sha256']) for r in model_paths],
            grouping_components=[dict(path=r['artifact_path'],sha256=r['artifact_sha256']) for r in group_paths],
            seed_check_script_sha256=sha256_file(__file__))
        write_json(exp.output/'model_manifest.json',exp.model_manifest)
        exp.finish(dict(status='completed',training_performed=True,scope=scope,seed=a.seed,folds=folds,
            raw_e2e=formula_metrics(all_raw),top1_e2e=formula_metrics(all_top1),baseline_e2e=formula_metrics(all_base),
            raw_glyph=glyph_metrics(all_glyph),comparison_scope='HWR refit seed sensitivity with frozen per-fold epochs and grouping; no hyperparameter reselection'))
        write_json(exp.output/'verification.json',verify(exp.output))
    except Exception as error:
        ledger=json.loads((exp.output/'experiment.json').read_text(encoding='utf-8'))
        ledger.update(status='failed',exit_code=1,error=str(error));write_json(exp.output/'experiment.json',ledger)
        raise


if __name__=='__main__':
    main()
