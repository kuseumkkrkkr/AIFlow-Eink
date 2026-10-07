"""Synthetic literal transcription stress test, official torch only.

No ink, handwriting labels, training, thresholds selected, or acceptance claim.
Every synthetic visual Top-1 digit is treated as the intended literal token,
regardless of arithmetic validity. Arithmetic is never consulted by decoding.
"""
import hashlib
import json
import resource
import time
from collections import Counter
from pathlib import Path

import torch

from official_mini_lm_v64 import load_model

ROOT=Path(__file__).resolve().parents[1]
CHECKPOINT=ROOT/'artifacts/hwr_failure_cause_microscope_20260928/mini_lm_distill_20261003_one_layer/mini_formula_lm.pt'
OUTPUT=ROOT/'artifacts/hwr_literal_digit_counterfactual_20261007_v65'
STRICT={'1','|','/','0','O','o','x',r'\times'}


def main():
    if OUTPUT.exists():
        raise ValueError('refusing to overwrite results')
    torch.set_num_threads(1)
    assert hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()=='2c448199c2b4d65c4d786348ad2f3ba09322fedb5337b35d503f48d0462d01f2'
    model,payload=load_model(CHECKPOINT)
    vocabulary={label:i for i,label in enumerate(model.labels)}
    right=len(model.labels)+model.relations.index('right')
    digit_ids=torch.tensor([vocabulary[str(i)] for i in range(10)])
    counts=Counter(); changed=[]; distributions=[]; latencies=[]
    # Deliberately exhaustive simple contexts; no cohort-selected examples.
    with torch.inference_mode():
        for op in ['+','-',r'\times']:
            for a in range(10):
                for b in range(10):
                    ids=[model.cls_id,vocabulary[str(a)],right,vocabulary[op],right,
                         vocabulary[str(b)],right,vocabulary['='],right,model.mask_id,model.sep_id]
                    inputs=torch.tensor([ids],dtype=torch.long)
                    start=time.perf_counter_ns()
                    scores=model(inputs,torch.ones_like(inputs,dtype=torch.bool),torch.tensor([9])).log_softmax(-1)[0]
                    latencies.append((time.perf_counter_ns()-start)/1e6)
                    digit_scores=scores[digit_ids]
                    # This sum exists only for diagnostic tags after scoring, never as model input.
                    arithmetic_result=a+b if op=='+' else a-b if op=='-' else a*b
                    context_id=f'{a}:{op}:{b}'
                    distributions.append(dict(context_id=context_id,digit_logp=digit_scores.tolist(),
                                              most_preferred_digit=int(digit_scores.argmax()),
                                              arithmetic_result_for_posthoc_tag_only=arithmetic_result))
                    for literal in range(10):
                        for rival in range(10):
                            if rival==literal:
                                continue
                            for gap in [.10,.20,.30]:
                                # Controlled two-candidate visual distributions, not calibrated HWR.
                                visual=torch.tensor([(1+gap)/2,(1-gap)/2])
                                candidates=[str(literal),str(rival)]
                                fused=visual.log()+.5*scores[torch.tensor([vocabulary[t] for t in candidates])]
                                winner=candidates[int(fused.argmax())]
                                # Compare gap in the originally specified decimal experiment parameter.
                                # No threshold fitting; .20 is the historical fixed cap.
                                final=winner if gap<=.20 and str(literal) not in STRICT else str(literal)
                                key=f'{gap:.2f}'
                                counts[key+':total']+=1
                                counts[key+':changed']+=final!=str(literal)
                                valid=literal==arithmetic_result
                                counts[key+(':arithmetically_valid' if valid else ':arithmetically_invalid')]+=1
                                if final!=str(literal):
                                    counts[key+':changed_'+('valid' if valid else 'invalid')]+=1
                                    counts[key+':changed_to_arithmetic_result']+=int(final)==arithmetic_result
                                    changed.append(dict(context_id=context_id,literal=str(literal),rival=str(rival),
                                                        visual_gap=gap,before=str(literal),after=final,
                                                        literal_equals_arithmetic_result=valid,
                                                        after_equals_arithmetic_result=int(final)==arithmetic_result,
                                                        lm_logp_difference=float(scores[vocabulary[str(rival)]]-scores[vocabulary[str(literal)]])))
    assert counts['0.30:changed']==0
    assert sum(v for k,v in counts.items() if k.endswith(':total'))==81000
    assert len(distributions)==300
    OUTPUT.mkdir()
    for name,rows in [('changed_cases',changed),('context_scores',distributions)]:
        (OUTPUT/f'{name}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    result=dict(schema='aiflow-literal-digit-counterfactual/v65',
                protocol=dict(official_torch=True,training_performed=False,retired_final_tests_loaded=False,
                              consumed149_loaded=False,raw_ink_loaded=False,thresholds_selected=False,
                              product_changed=False,literal_target_is_visual_top1=True,
                              arithmetic_result_used_by_decoding=False,
                              assumptions='Clean symbolic neighboring tokens and synthetic two-digit visual distributions; targets stand for literal writing, including arithmetic errors.'),
                checkpoint_sha256=hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
                code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                contexts=300,synthetic_decisions=81000,counts=dict(sorted(counts.items())),
                runtime=dict(torch_version=torch.__version__,threads=1,forward_median_ms=sorted(latencies)[len(latencies)//2],
                             process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                             scope='LM forward/log-softmax only; single pass no warmup; peak includes entire synthetic audit'),
                interpretation='A changed literal is a synthetic transcription failure by construction. These are risk controls, not real handwriting error rates or evidence that the LM solves arithmetic.',
                next_step='Use explicit literal-preservation controls for future training and inference contracts; require independently transcribed writer/formula-disjoint ink before acceptance.')
    (OUTPUT/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
