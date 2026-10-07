"""Checkpointed same-input official-LM audit; no scientific round auto-repeat."""
import argparse
import json
import subprocess
from pathlib import Path

import torch

from official_mini_lm_v64 import load_model
from research_checkpoint_v1 import CheckpointStore,ResumeError,atomic_bytes,encoded

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--round-id',default='v64-migration-functional-replay')
    parser.add_argument('--requests',type=Path,default=ROOT/'artifacts/hwr_numpy_reconstruction_20261007_v63/requests.jsonl')
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'artifacts/hwr_failure_cause_microscope_20260928/mini_lm_distill_20261003_one_layer/mini_formula_lm.pt')
    parser.add_argument('--chunk-size',type=int,default=64)
    parser.add_argument('--stop-after-cursor',type=int,help='clean checkpoint-boundary return; for owned test/restart checks')
    args=parser.parse_args()
    if args.chunk_size<1:
        parser.error('chunk-size must be positive')
    torch.set_num_threads(1)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    store=CheckpointStore(args.run_dir,run_id=args.run_id,round_id=args.round_id,stage_id='official-lm-inference',kind='audit',
                          source_commit=commit,config=dict(chunk_size=args.chunk_size,threads=1,policy='logits only; same-input audit; no reranking selection'),
                          assets=dict(data=args.requests,checkpoint=args.checkpoint,runner=Path(__file__),
                                      runtime=Path(__file__).with_name('official_mini_lm_v64.py'),continuity=Path(__file__).with_name('research_checkpoint_v1.py')))
    with store.owner():
        restored=store.initialize('compute official LM outputs from validated input cursor')
        if restored and restored[0]['phase']=='stage_complete':
            destination=args.run_dir/'final_outputs.json'
            expected=encoded(restored[1]['outputs'])
            if not destination.exists() or destination.read_bytes()!=expected:
                atomic_bytes(destination,expected)
            store.recovery_summary(args.run_dir/'RECOVERY_SUMMARY.json')
            print(json.dumps(dict(status='skipped_completed_stage',cursor=restored[1]['cursor'],run_id=args.run_id)))
            return 0
        records=[json.loads(s) for s in args.requests.read_text().splitlines()]
        if not records:
            raise ResumeError('empty inference requests are unsupported; no stage was completed')
        if len({r['record_id'] for r in records})!=len(records):
            raise ResumeError('duplicate input request identity')
        outputs=restored[1]['outputs'] if restored else []
        cursor=restored[1]['cursor'] if restored else 0
        if cursor>len(records) or [r['record_id'] for r in outputs]!=[r['record_id'] for r in records[:cursor]]:
            raise ResumeError('committed prefix differs from input sequence')
        model,payload=load_model(args.checkpoint)
        with torch.inference_mode():
            while cursor<len(records):
                end=min(cursor+args.chunk_size,len(records))
                for request in records[cursor:end]:
                    ids=torch.tensor([request['ids']],dtype=torch.long)
                    logits=model(ids,torch.ones_like(ids,dtype=torch.bool),torch.tensor([request['target']],dtype=torch.long))
                    if not torch.isfinite(logits).all():
                        raise ResumeError('nonfinite official LM output')
                    outputs.append(dict(record_id=request['record_id'],logits=logits[0].tolist()))
                cursor=end
                store.save_audit(outputs,cursor=cursor,completed=cursor==len(records),
                                 next_action='verify/publish completed audit' if cursor==len(records) else 'continue remaining input prefix')
                mirror=store.recovery_summary(args.run_dir/'RECOVERY_SUMMARY.json',optional_mirror=Path('/workspace/shared/aiflow-resume')/(args.run_id+'.json'))
                if args.stop_after_cursor and cursor>=args.stop_after_cursor and cursor<len(records):
                    print(json.dumps(dict(status='checkpoint_complete',cursor=cursor,mirror=mirror)))
                    return 0
        atomic_bytes(args.run_dir/'final_outputs.json',encoded(outputs))
        print(json.dumps(dict(status='stage_complete',cursor=cursor,mirror=mirror)))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
