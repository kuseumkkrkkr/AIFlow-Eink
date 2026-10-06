"""같은 outer head logits에서 K=5/10/20 후보 포함률과 상한을 진단한다."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from accuracy_upgrade_data_10e import load_corpus, load_hwr, write_json
from accuracy_upgrade_contract_v1 import sha256_file
from accuracy_lineage_10e import LineageRegistry
from analyze_accuracy_candidates_10e import analyze


def main():
    """정답은 순위 생성 후 포함 여부 계산에만 사용하고 후보는 logits로 고정한다."""
    p=argparse.ArgumentParser(); p.add_argument('--experiment',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    ledger=json.loads((a.experiment/'experiment.json').read_text(encoding='utf-8'))
    if ledger['status']!='completed' or ledger['arguments']['scope']!='head':
        raise ValueError('completed head-only experiment required for frozen embeddings')
    registry=LineageRegistry.load(a.experiment/'lineage_manifest.json')
    corpus=load_corpus(ledger['configuration'])
    splits=json.loads((a.experiment/'split_manifest.json').read_text(encoding='utf-8'))['splits']
    cache=json.loads((a.experiment/'cache_manifest.json').read_text(encoding='utf-8'))
    if sha256_file(cache['hwr_features'])!=cache['hwr_features_sha256']:
        raise ValueError('embedding cache SHA mismatch')
    payload=torch.load(cache['hwr_features'],map_location='cpu',weights_only=False)
    if payload['cache_key']!=cache['cache_key']:
        raise ValueError('embedding cache key mismatch')
    rows={k:[] for k in (5,10,20)};torch.set_num_threads(4)
    for outer,split in enumerate(splits):
        path=(a.experiment/'models'/f'outer{outer}_hwr_refit.pt').resolve()
        roots=[key for key,node in registry.nodes.items() if node['artifact_path']==str(path)]
        if len(roots)!=1:
            raise ValueError('checkpoint lineage missing')
        registry.validate(roots[0],{split['outer_writer']},{registry.records[k]['formula_key'] for k in split['held_ids']})
        model,labels=load_hwr(path,torch.device('cpu'))
        with torch.inference_mode():
            for key in split['held_ids']:
                vector=np.asarray(payload['embeddings'][key],dtype=np.float32)
                logits=model.math_head(torch.from_numpy(vector)).numpy()
                order=np.argsort(-logits,kind='stable')
                for k in rows:
                    candidates=[labels[i] for i in order[:k]]
                    record=registry.records[key]
                    rows[k].append(dict(record_id=key,formula_id=record['formula_id'],label=corpus.raw[key]['label'],
                                        candidates=candidates,adapter_token=candidates[0]))
                if not set(labels[i] for i in order[:5])<=set(labels[i] for i in order[:20]):
                    raise AssertionError('candidate inclusion failed')
    result={str(k):dict(analyze(values),candidate_recall=sum(r['label'] in r['candidates'] for r in values)/len(values)) for k,values in rows.items()}
    write_json(a.output,dict(status='completed',training_performed=False,scope='fixed head logits; truth grouping diagnostic, not e2e',ranges=result))
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
