"""기존 예측은 유지하고 코드 계보의 가변 경로를 불변 사본으로 연결한다."""
import argparse
import copy
import json
from pathlib import Path
import shutil

from accuracy_lineage_10e import LineageRegistry
from accuracy_upgrade_contract_v1 import canonical_json_sha256, sha256_file
from accuracy_upgrade_data_10e import write_json, write_rows
from online_candidate_features_10e import _json_lines
from verify_accuracy_experiment_10e import verify


def main():
    """저장 예측의 root 참조만 새 manifest로 바꾸며 기존 실험은 덮어쓰지 않는다."""
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--source-snapshot',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() or a.output.resolve().drive.upper()!='D:':
        raise ValueError('new D: archive output required')
    ledger=json.loads((a.experiment/'experiment.json').read_text(encoding='utf-8'))
    if ledger['status']!='completed':
        raise ValueError('source incomplete')
    registry=LineageRegistry.load(a.experiment/'lineage_manifest.json')
    # 복사 전에 모든 필요한 코드 바이트를 실제 기록 SHA와 대조한다.
    sources={}
    for key,node in registry.nodes.items():
        path=Path(node['artifact_path'])
        if path.suffix=='.py':
            archived=a.source_snapshot/path.name
            source=archived if archived.exists() and sha256_file(str(archived))==node['artifact_sha256'] else path
            if not source.exists() or sha256_file(str(source))!=node['artifact_sha256']:
                raise ValueError('original code bytes unavailable: '+path.name)
            sources[key]=source
    a.output.mkdir(parents=True);(a.output/'runtime_sources').mkdir()
    mapped={};new_nodes={}
    def rewrite(key):
        """부모 ID를 함께 갱신하되 학습 ID와 체크포인트 내용은 바꾸지 않는다."""
        if key in mapped:
            return mapped[key]
        node=copy.deepcopy(registry.nodes[key]);node['parents']=[rewrite(k) for k in node['parents']]
        if key in sources:
            target=a.output/'runtime_sources'/(node['artifact_sha256']+'_'+sources[key].name)
            if not target.exists():
                shutil.copy2(sources[key],target)
            node['archived_from_lineage_node']=key;node['artifact_path']=str(target.resolve())
        new_key=canonical_json_sha256(node);mapped[key]=new_key;new_nodes[new_key]=node
        return new_key
    for key in registry.nodes:
        rewrite(key)
    new_registry=LineageRegistry(registry.records,new_nodes)
    write_json(a.output/'lineage_manifest.json',new_registry.payload())
    for filename in ('formula_predictions.jsonl.gz','baseline_formula_predictions.jsonl.gz'):
        rows=_json_lines(a.experiment/filename)
        for row in rows:
            for key in list(row):
                if key.endswith('lineage_root_id'):
                    row[key]=mapped[row[key]]
        write_rows(a.output/filename,rows)
    for filename in ('model_manifest.json','split_manifest.json'):
        shutil.copy2(a.experiment/filename,a.output/filename)
    report=json.loads((a.experiment/'evaluation.json').read_text(encoding='utf-8'))
    report.update(training_performed=False,neural_inference_performed=False,
                  archive_scope='identical predictions; immutable code lineage only',parent_experiment=str(a.experiment.resolve()))
    write_json(a.output/'evaluation.json',report)
    ledger.update(stage='code_lineage_archive',status='completed',exit_code=0,parent_experiment=str(a.experiment.resolve()),
                  parent_experiment_sha256=sha256_file(str(a.experiment/'experiment.json')))
    write_json(a.output/'experiment.json',ledger)
    write_json(a.output/'verification.json',verify(a.output))
    print(json.dumps(dict(status='completed',output=str(a.output),archived_code_nodes=len(sources))),flush=True)


if __name__=='__main__':
    main()
