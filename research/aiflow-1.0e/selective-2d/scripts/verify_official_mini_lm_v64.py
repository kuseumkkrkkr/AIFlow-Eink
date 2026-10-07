"""Same-input PyTorch migration verification, not generalization evaluation."""
import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np
import torch
from scipy.special import logsumexp

from official_mini_lm_v64 import load_model

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'artifacts/hwr_numpy_reconstruction_20261007_v63'
OUTPUT=ROOT/'artifacts/hwr_official_torch_migration_20261007_v64'
CHECKPOINT=ROOT/'artifacts/hwr_failure_cause_microscope_20260928/mini_lm_distill_20261003_one_layer/mini_formula_lm.pt'
CANONICAL=ROOT.parent/'augmentation-models/canonical/project_symbol_head_checkpoint.pt'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pack(requests,model):
    width=max(len(r['ids']) for r in requests)
    ids=torch.full((len(requests),width),model.pad_id,dtype=torch.long)
    attention=torch.zeros_like(ids,dtype=torch.bool)
    positions=torch.tensor([r['target'] for r in requests],dtype=torch.long)
    for i,r in enumerate(requests):
        ids[i,:len(r['ids'])]=torch.tensor(r['ids'],dtype=torch.long)
        attention[i,:len(r['ids'])]=True
    return ids,attention,positions


def main():
    if OUTPUT.exists():
        raise ValueError('refusing to overwrite migration output')
    torch.set_num_threads(1)
    expected=json.loads((OLD/'report.json').read_text())
    assert sha(CANONICAL)=='04f8608aebcf6c02d45ad6f5735229b9eaa2c4b4e1be0db4793d02273ef2d00e'
    assert sha(CHECKPOINT)==expected['provenance'][str(CHECKPOINT.relative_to(ROOT.parent.parent.parent))]
    model,payload=load_model(CHECKPOINT)
    requests=[json.loads(s) for s in (OLD/'requests.jsonl').read_text().splitlines()]
    predictions=[json.loads(s) for s in (OLD/'predictions.jsonl').read_text().splitlines()]
    reference=np.load(OLD/'migration_reference_logits.npz',allow_pickle=False)
    assert [r['record_id'] for r in requests]==reference['record_ids'].tolist()==[r['record_id'] for r in predictions]
    outputs=[]
    times=[]
    with torch.inference_mode():
        for r in requests:
            args=pack([r],model)
            for _ in range(2):
                model(*args)
            start=time.perf_counter_ns()
            logits=model(*args)
            times.append((time.perf_counter_ns()-start)/1e6)
            assert logits.dtype==torch.float32 and logits.shape==(1,372) and torch.isfinite(logits).all()
            outputs.append(logits.numpy()[0])
        batched=torch.cat([model(*pack(requests[i:i+64],model)) for i in range(0,len(requests),64)]).numpy()
    outputs=np.stack(outputs)
    max_error=float(np.max(np.abs(outputs-reference['logits'])))
    assert max_error<1e-4
    assert float(np.max(np.abs(outputs-batched)))<1e-4
    lp=outputs-logsumexp(outputs,axis=1,keepdims=True)
    ref_lp=reference['logits']-logsumexp(reference['logits'],axis=1,keepdims=True)
    assert np.max(np.abs(np.exp(lp).sum(axis=1)-1))<1e-5
    assert float(np.max(np.abs(np.exp(lp)-np.exp(ref_lp))))<1e-5
    assert np.array_equal(outputs.argmax(axis=1),reference['logits'].argmax(axis=1))
    paired=[]
    for r,scores in zip(predictions,lp,strict=True):
        topk=r['candidates']; probs=r['probabilities']
        values=scores[[model.labels.index(t) for t in topk]]
        assert np.max(np.abs(values-r['context_logp']))<1e-4
        fused=topk[int(np.argmax(np.log(np.maximum(probs,1e-12))+.5*values))]
        strict=fused if topk[0] not in {'1','|','/','0','O','o','x',r'\times'} else topk[0]
        capped=strict if probs[0]-max(probs[1:])<=.20 else topk[0]
        assert (fused,strict,capped)==(r['fused'],r['strict'],r['capped'])
        paired.append(dict(record_id=r['record_id'],formula_id=r['formula_id'],truth=r['truth'],before=r['before'],fused=fused,strict=strict,capped=capped))
    # Exercise failure contracts, using valid inputs with one invalid field each.
    ids,attention,positions=pack(requests[:1],model)
    bad=[(ids.float(),attention,positions),(ids,attention,positions.float()),
         (ids,attention[:,:-1],positions),(ids,torch.zeros_like(attention),positions),
         (ids,attention,torch.full_like(positions,ids.shape[1])),
         (torch.full_like(ids,-1),attention,positions)]
    rejected=0
    for args in bad:
        try:
            model(*args)
        except ValueError:
            rejected+=1
    assert rejected==len(bad)
    assert sha(CANONICAL)=='04f8608aebcf6c02d45ad6f5735229b9eaa2c4b4e1be0db4793d02273ef2d00e'
    result=dict(schema='aiflow-official-torch-migration/v64',torch_version=torch.__version__,torch_module=torch.__file__,
                protocol=dict(runtime='official PyTorch only',fallback_enabled=False,training_performed=False,
                              retired_final_tests_loaded=False,independent_acceptance=False,product_default_changed=False),
                verification=dict(requests=len(requests),full_logit_max_abs_error=max_error,
                                  full_probability_max_abs_error=float(np.max(np.abs(np.exp(lp)-np.exp(ref_lp)))),
                                  singleton_vs_padded_batch_max_abs_error=float(np.max(np.abs(outputs-batched))),
                                  all_final_tokens_and_formula_decisions_match=True,invalid_contracts_rejected=rejected,
                                  finite_fp32_shape_checked=True,canonical_sha256=sha(CANONICAL),
                                  tolerance='1e-4 logits/log-probabilities, 1e-5 probabilities; exact class and fused decisions required'),
                metrics=expected['metrics'],
                runtime=dict(threads=1,forward_median_ms=float(np.median(times)),warmups_per_request=2,
                             scope='official LM forward and validation only; no HWR or request packing',
                             process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                provenance={str(p.relative_to(ROOT.parent.parent.parent)):sha(p) for p in [CHECKPOINT,CANONICAL,OLD/'requests.jsonl',OLD/'predictions.jsonl',OLD/'migration_reference_logits.npz',Path(__file__),Path(__file__).with_name('official_mini_lm_v64.py')]},
                limitation='Same-input migration parity against newly generated archived NumPy reference, not new HWR generalization or literal-error acceptance.')
    OUTPUT.mkdir()
    np.savez(OUTPUT/'official_logits.npz',logits=outputs)
    (OUTPUT/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in paired))
    (OUTPUT/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
