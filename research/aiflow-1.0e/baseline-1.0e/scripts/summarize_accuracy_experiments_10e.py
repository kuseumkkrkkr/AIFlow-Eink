"""이번 실행의 원장·SHA·검증 범위를 하나의 새 색인으로 모은다."""
import argparse
import json
from pathlib import Path
from accuracy_upgrade_data_10e import write_json
from accuracy_upgrade_contract_v1 import sha256_file


def main():
    """과거 결과를 수정하지 않고 완료·진행·자료 차단을 분리해 기록한다."""
    p=argparse.ArgumentParser();p.add_argument('--artifacts',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() or a.output.resolve().drive.upper()!='D:':
        raise ValueError('new D: index required')
    entries=[]
    for directory in sorted(a.artifacts.glob('accuracy_upgrade_20260911_*')):
        path=directory/'experiment.json'
        if not path.exists():
            continue
        ledger=json.loads(path.read_text(encoding='utf-8'));entry=dict(name=directory.name,path=str(directory.resolve()),
            status=ledger['status'],stage=ledger['stage'],seed=ledger['configuration']['seed'],
            command=ledger.get('command'),arguments=ledger.get('arguments'),parent_experiment=ledger.get('parent_experiment'),
            exit_code=ledger.get('exit_code'),elapsed_seconds=ledger.get('elapsed_seconds'),files={})
        for name in ('experiment.json','model_manifest.json','split_manifest.json','lineage_manifest.json','evaluation.json',
                     'verification.json','formula_predictions.jsonl.gz','glyph_predictions.jsonl.gz'):
            file=directory/name
            if file.exists():
                entry['files'][name]=dict(path=str(file.resolve()),sha256=sha256_file(str(file)))
        report_path=directory/'evaluation.json'
        if report_path.exists():
            report=json.loads(report_path.read_text(encoding='utf-8'))
            entry['metrics']=report.get('raw_e2e',report.get('aggregate'))
        verification=directory/'verification.json'
        if verification.exists():
            proof=json.loads(verification.read_text(encoding='utf-8'))
            entry['verification_snapshot_matches_ledger']=bool(proof.get('verified') and proof['source_experiment_sha256']==entry['files']['experiment.json']['sha256'])
        archived=directory.with_name(directory.name+'_archived')
        if (archived/'verification.json').exists():
            entry['preferred_immutable_code_lineage']=str(archived.resolve())
        entries.append(entry)
    write_json(a.output,dict(schema='aiflow-accuracy-upgrade-index/v1',experiments=entries,
        verification_scope='recorded full-coverage/ancestor verification; use archived derivatives where listed',
        data_role='consumed_development',blocked_data={
            'relation_head':'only two relation-annotated formulas',
            'fresh_online_mix_and_acceptance':'new writer/formula/device-disjoint annotated data unavailable',
            'trocr_and_three_teacher_ensemble':'checkpoint derivative distribution rights unverified',
            'product_promotion':'fresh acceptance and device evidence missing; best candidate has regressions'},
        product_activated=False,uploaded=False))
    print(json.dumps(dict(experiments=len(entries),statuses={s:sum(r['status']==s for r in entries) for s in sorted({r['status'] for r in entries})})),flush=True)


if __name__=='__main__':
    main()
