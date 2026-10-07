"""Verify reconstructed LM on preserved consumed diagnostics, without raw joins."""
import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from reconstructed_mini_lm_numpy_v63 import NumpyLM, load_checkpoint

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/hwr_failure_cause_microscope_20260928'
OUTPUT = ROOT / 'artifacts/hwr_numpy_reconstruction_20261007_v63'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relation(a, b, width, height):
    dx = ((b['left'] + b['right']) - (a['left'] + a['right'])) / (2 * width)
    dy = ((b['top'] + b['bottom']) - (a['top'] + a['bottom'])) / (2 * height)
    w = max(((a['right']-a['left']) + (b['right']-b['left'])) / (2*width), 1e-6)
    h = max(((a['bottom']-a['top']) + (b['bottom']-b['top'])) / (2*height), 1e-6)
    if abs(dx) <= .25*w and abs(dy) <= .25*h:
        return 'overlap'
    if abs(dx) <= .60*w and abs(dy) >= .55*h:
        return 'above' if dy < 0 else 'below'
    if dx >= -.10*w and dy <= -.45*h:
        return 'superscript'
    if dx >= -.10*w and dy >= .45*h:
        return 'subscript'
    return 'right' if dx >= 0 else 'left'


def requests(model, symbols):
    boxes = [s['preprocessing']['raw_bbox'] for s in symbols]
    width = max(max(b['right'] for b in boxes)-min(b['left'] for b in boxes), 1e-6)
    height = max(max(b['bottom'] for b in boxes)-min(b['top'] for b in boxes), 1e-6)
    rels = [len(model.labels)+model.relations.index(relation(a,b,width,height)) for a,b in zip(boxes,boxes[1:])]
    tokens = [model.labels.index(s['prediction']['top1']) for s in symbols]
    for target in range(len(symbols)):
        ids = [model.cls]
        for i, token in enumerate(tokens):
            if i:
                ids.append(rels[i-1])
            ids.append(model.mask if i == target else token)
        yield ids + [model.sep], 1 + 2*target


def metrics(rows, key):
    before, after = {}, {}
    hits = old_hits = regressions = recoveries = 0
    for r in rows:
        old_ok, new_ok = r['before'] == r['truth'], r[key] == r['truth']
        old_hits += old_ok
        hits += new_ok
        regressions += old_ok and not new_ok
        recoveries += new_ok and not old_ok
        before.setdefault(r['formula_id'], []).append(old_ok)
        after.setdefault(r['formula_id'], []).append(new_ok)
    recovered = [f for f in before if not all(before[f]) and all(after[f])]
    regressed = [f for f in before if all(before[f]) and not all(after[f])]
    return dict(formulas=len(before), tokens=len(rows), baseline_formula_exact=sum(map(all,before.values())),
                formula_exact=sum(map(all,after.values())), baseline_token_hits=old_hits, token_hits=hits,
                recovered_formula_ids=recovered, regressed_formula_ids=regressed,
                token_recoveries=recoveries, token_regressions=regressions)


def main():
    if OUTPUT.exists():
        raise ValueError('refusing to overwrite results')
    checkpoint = BASE / 'mini_lm_distill_20261003_one_layer/mini_formula_lm.pt'
    report_path = checkpoint.with_name('mini_formula_lm_distillation_report.json')
    report = json.loads(report_path.read_text())
    assert sha(checkpoint) == report['student']['checkpoint_sha256']
    assert report['protocol']['crohme_rows_loaded'] == 0
    payload = load_checkpoint(checkpoint)
    model = NumpyLM(payload)
    trace_path = BASE / 'layer_trace_149_current_20260928/formula_traces.jsonl'
    reference_path = BASE / 'mini_lm_onnx_20261002/one_layer_trained_context_microscope_20261003.json'
    reference = json.loads(reference_path.read_text())
    assert sha(checkpoint) == reference['provenance']['checkpoint_sha256']
    comparisons, decisions, input_requests = [], [], []
    target_errors = []
    timings = {'full': [], 'target_only': []}
    # Read one archived formula at a time to avoid retaining large network traces.
    with trace_path.open() as stream:
        for line in stream:
            formula = json.loads(line)
            fid = formula['sample_id']
            oracle = formula['oracle_group_hwr']['symbols']
            for i, ((ids, pos), symbol) in enumerate(zip(requests(model,oracle),oracle,strict=True)):
                start = time.perf_counter_ns()
                full = model.forward(ids,pos)
                timings['full'].append((time.perf_counter_ns()-start)/1e6)
                start = time.perf_counter_ns()
                logits = model.forward(ids,pos,target_only=True)
                timings['target_only'].append((time.perf_counter_ns()-start)/1e6)
                target_errors.append(float(np.max(np.abs(full-logits))))
                assert np.argmax(full) == np.argmax(logits)
                logp = logits - logsumexp(logits)
                pred = symbol['prediction']
                topk, probs = pred['top5'], pred['top5_probabilities']
                scores = np.log(np.maximum(probs,1e-12)) + .5*logp[[model.labels.index(t) for t in topk]]
                full_scores = np.log(np.maximum(probs,1e-12)) + .5*(full-logsumexp(full))[[model.labels.index(t) for t in topk]]
                assert np.argmax(scores) == np.argmax(full_scores)
                fused = topk[int(np.argmax(scores))]
                locked = fused if topk[0] not in {'1','|','/','0','O','o','x',r'\times'} else topk[0]
                capped = locked if probs[0]-max(probs[1:]) <= .20 else topk[0]
                decisions.append(dict(record_id=f'{fid}:{i}',formula_id=fid,truth=symbol['target_label'],
                                      before=topk[0],fused=fused,strict=locked,capped=capped,
                                      candidates=topk, probabilities=probs, context_logp=logp[[model.labels.index(t) for t in topk]].tolist()))
                input_requests.append(dict(record_id=f'{fid}:{i}',ids=ids,target=pos))
            # Earlier layer microscope used the selected groups, not oracle groups.
            selected = []
            for s in formula['grouping']['selected_symbols']:
                selected.append(dict(preprocessing=s['preprocessing'], prediction=s['hwr_prediction']))
            cases = {c['record_id']:c for c in reference['nontrivial_token_cases'] if c['record_id'].rsplit(':',1)[0] == fid}
            for i,(ids,pos) in enumerate(requests(model,selected)):
                rid=f'{fid}:{i}'
                if rid in cases:
                    c=cases[rid]
                    assert selected[i]['prediction']['top5'] == c['hwr_top5']
                    logits=model.forward(ids,pos)
                    actual=(logits-logsumexp(logits))[[model.labels.index(t) for t in c['hwr_top5']]]
                    expected=np.asarray(c['layers']['layer_1']['context_top5_log_probs'])
                    comparisons.append(dict(record_id=rid,max_abs_error=float(np.max(np.abs(actual-expected)))))
    assert len(comparisons) == len(reference['nontrivial_token_cases']) == 87
    assert len(decisions) == 579 and len({r['formula_id'] for r in decisions}) == 149
    assert max(c['max_abs_error'] for c in comparisons) < 1e-4
    assert max(target_errors) < 1e-4
    canonical=ROOT.parent/'augmentation-models/canonical/project_symbol_head_checkpoint.pt'
    result = dict(schema='aiflow-new-numpy-reconstruction/v63',
                  protocol=dict(new_reconstruction=True, historical_v42_v62_recovered=False,
                                consumed_oracle_group_diagnostics=True, independent_acceptance=False,
                                raw_stroke_replay=False, writer_split_or_training_changes=False,
                                training_performed=False, thresholds_selected=False,crohme_rows_loaded=0,
                                product_default_changed=False,literal_mistake_preservation_proven=False),
                  provenance={str(p.relative_to(ROOT.parent.parent.parent)):sha(p) for p in [checkpoint,report_path,trace_path,reference_path,canonical,Path(__file__),Path(__file__).with_name('reconstructed_mini_lm_numpy_v63.py')]},
                  verification=dict(archived_reference_cases=87,archived_reference_class_scores=435,
                                    max_archived_score_error=max(c['max_abs_error'] for c in comparisons),
                                    target_only_max_logit_error=max(target_errors),target_only_all_579_top1_and_fused_ids_match=True,
                                    limitation='Archived reference covers 87 selected-group Top-5 score vectors only; no live PyTorch or complete-logit parity claim.'),
                  model=dict(parameters=sum(v.size for v in model.w.values()),fp32_tensor_bytes=sum(v.nbytes for v in model.w.values()),checkpoint_bytes=checkpoint.stat().st_size),
                  metrics={key:metrics(decisions,key) for key in ['fused','strict','capped']},
                  runtime=dict(host='Linux CPU', scope='LM forward only; excludes HWR, input parsing and archive loading',
                               samples=len(decisions),timing_order='full then target per request; exploratory single pass, no warm-up',
                               full_median_ms=float(np.median(timings['full'])),target_only_median_ms=float(np.median(timings['target_only'])),
                               process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                               memory_scope='entire Python audit including SciPy imports and sequential large JSON trace parsing; not decoder-only memory'),
                  decision='Reconstructed research runtime only; no product adoption. Fixed .5 fusion/.20 cap are historical policies, not selected here.')
    OUTPUT.mkdir()
    for name,rows in [('predictions',decisions),('requests',input_requests),('reference_comparisons',comparisons)]:
        (OUTPUT/f'{name}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    np.savez(OUTPUT/'reconstructed_weights_fp32.npz',**model.w)
    result['weights_npz_sha256']=sha(OUTPUT/'reconstructed_weights_fp32.npz')
    (OUTPUT/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
