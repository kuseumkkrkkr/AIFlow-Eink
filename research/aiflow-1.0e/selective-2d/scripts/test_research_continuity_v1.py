"""Disposable official-PyTorch checkpoint interruption and compatibility tests."""
import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from research_checkpoint_v1 import CheckpointStore,ResumeError,atomic_bytes,encoded,file_sha

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'artifacts/hwr_research_continuity_20261008_v66_r2'
CONFIG=dict(seed=731,total_steps=12,data_items=7,optimizer='AdamW',scheduler='ExponentialLR',checkpoint_every=4,accumulation=0)


def store(path,config=None,source='deterministic-owned-test-source',data=None,fault_hook=None):
    return CheckpointStore(path,run_id='deterministic-fixture',round_id='continuity-test-v1',stage_id='toy-training',kind='training',
                source_commit=source,config=config or CONFIG,fault_hook=fault_hook,assets=dict(data=data or OUTPUT/'fixture_data.json',
                    code=Path(__file__),continuity=Path(__file__).with_name('research_checkpoint_v1.py')))


def train_child(directory,crash=None,fault=None):
    torch.set_num_threads(1)
    random.seed(CONFIG['seed']);np.random.seed(CONFIG['seed']);torch.manual_seed(CONFIG['seed'])
    model=torch.nn.Sequential(torch.nn.Linear(3,2),torch.nn.Dropout(.2))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
    scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=.97)
    x=torch.tensor(json.loads((OUTPUT/'fixture_data.json').read_text()),dtype=torch.float32)
    sampler=dict(order=torch.randperm(7).tolist(),position=0)
    step=iteration=0
    def injection(boundary,step):
        if fault==boundary and step==8:
            os._exit(76)
    s=store(directory,fault_hook=injection if fault else None)
    with s.owner():
        loaded=s.initialize('resume toy optimizer boundary')
        if loaded and loaded[0]['phase']=='stage_complete':
            print(json.dumps(dict(status='skipped_completed',step=loaded[1]['step'])))
            return
        if loaded:
            state=loaded[1];s.restore_training(state,model,optimizer,scheduler)
            sampler=state['sampler'];step=state['step'];iteration=state['iteration']
        else:
            s.save_training(model,optimizer,scheduler,step=0,iteration=0,sampler=sampler)
        while step<CONFIG['total_steps']:
            if sampler['position']==7:
                sampler=dict(order=torch.randperm(7).tolist(),position=0)
                iteration+=1
            index=sampler['order'][sampler['position']]
            # Exercise all three saved RNG streams during training.
            scale=random.random()+float(np.random.random())
            output=model(x[index:index+1])
            loss=(output-scale).square().mean()
            loss.backward();optimizer.step();optimizer.zero_grad(set_to_none=True);scheduler.step()
            sampler['position']+=1;step+=1
            if fault=='uncommitted_steps' and step==6:
                os._exit(76)
            if step%4==0:
                s.save_training(model,optimizer,scheduler,step=step,iteration=iteration,sampler=sampler,
                                completed=step==12,next_action='verify toy outputs' if step==12 else 'continue toy fixture')
                if crash==step:
                    print(json.dumps(dict(status='intentional_owned_child_exit',step=step)),flush=True)
                    os._exit(75)
        print(json.dumps(dict(status='stage_complete',step=step)))


def equivalent(a,b):
    if isinstance(a,torch.Tensor):
        return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):
        return a.keys()==b.keys() and all(equivalent(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):
        return type(a)==type(b) and len(a)==len(b) and all(equivalent(x,y) for x,y in zip(a,b))
    return a==b


def run_tests():
    if OUTPUT.exists():
        raise ValueError('refusing to overwrite existing continuity test artifacts')
    OUTPUT.mkdir()
    (OUTPUT/'fixture_data.json').write_text(json.dumps([[i/7,(i+1)/8,(i+2)/9] for i in range(7)])+'\n')
    (OUTPUT/'config.json').write_text(json.dumps(CONFIG,indent=2)+'\n')
    commands=[]

    def command(arguments,expected=0):
        cmd=[sys.executable,str(Path(__file__).resolve())]+arguments
        p=subprocess.run(cmd,capture_output=True,text=True,timeout=90)
        commands.append(dict(command=cmd,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr))
        assert p.returncode==expected,commands[-1]
        return p

    command(['--child',str(OUTPUT/'baseline')])
    command(['--child',str(OUTPUT/'interrupted'),'--crash-at','4'],expected=75)
    shutil.copytree(OUTPUT/'interrupted',OUTPUT/'crash_snapshot')
    command(['--child',str(OUTPUT/'interrupted')])
    with store(OUTPUT/'baseline').owner() as baseline:
        _,a=baseline.load()
    with store(OUTPUT/'interrupted').owner() as resumed:
        _,b=resumed.load()
    assert equivalent(a,b),'model/optimizer/scheduler/RNG/sampler final state must exactly match'
    pointer=OUTPUT/'interrupted/task_state.json'
    prior=pointer.read_bytes()
    repeated=command(['--child',str(OUTPUT/'interrupted')])
    assert 'skipped_completed' in repeated.stdout and pointer.read_bytes()==prior
    fault_results={}
    for boundary in ['payload_flushed','manifest_flushed','before_pointer_replace','after_pointer_replace','uncommitted_steps']:
        directory=OUTPUT/('fault_'+boundary)
        command(['--child',str(directory),'--fault',boundary],expected=76)
        with store(directory).owner() as interrupted:
            restored_step=interrupted.load()[1]['step']
        assert restored_step==(8 if boundary=='after_pointer_replace' else 4)
        command(['--child',str(directory)])
        with store(directory).owner() as continued:
            assert equivalent(a,continued.load()[1])
        fault_results[boundary]=dict(restored_step=restored_step,exact_final_state=True)
    with store(OUTPUT/'interrupted').owner():
        command(['--lock-probe',str(OUTPUT/'interrupted')],expected=23)
    refusals={}

    def refuse(label,instance):
        try:
            with instance.owner():
                instance.load()
        except ResumeError as error:
            refusals[label]=str(error)
        else:
            raise AssertionError(label+' should fail clearly')

    refuse('config_mismatch',store(OUTPUT/'interrupted',config={**CONFIG,'seed':732}))
    with store(OUTPUT/'interrupted',source='result-only-publication-commit').owner() as compatible:
        assert compatible.load()[1]['step']==12
        assert compatible.identity['source_commit']=='deterministic-owned-test-source'
    wrong=OUTPUT/'wrong_data.json';wrong.write_text('[0]\n')
    refuse('input_mismatch',store(OUTPUT/'interrupted',data=wrong))
    for label in ['corrupt_payload','missing_payload','corrupt_manifest','truncated_pointer']:
        target=OUTPUT/label;shutil.copytree(OUTPUT/'interrupted',target)
        task=json.loads((target/'task_state.json').read_text())
        cp=target/task['latest_checkpoint']['directory']
        if label=='corrupt_payload':
            (cp/'state.pt').write_bytes(b'bad')
        elif label=='missing_payload':
            (cp/'state.pt').unlink()  # disposable test copy only
        elif label=='corrupt_manifest':
            (cp/'manifest.json').write_text('{}')
        else:
            (target/'task_state.json').write_text('{')
        refuse(label,store(target))
    partial=OUTPUT/'initial_partial';partial.mkdir();(partial/'checkpoint-uncommitted.partial').mkdir()
    refuse('partial_initial_checkpoint',store(partial))
    # A stray new partial cannot displace the prior valid checkpoint.
    partial=OUTPUT/'partial_after_valid';shutil.copytree(OUTPUT/'crash_snapshot',partial)
    (partial/'checkpoint-orphan.partial').mkdir()
    with store(partial).owner() as valid:
        assert valid.load()[1]['step']==4
        try:
            valid.record_remote(dict(sha='a'*40,verified_head_sha='b'*40,checkpoint_manifest_sha256='c'*64))
        except ResumeError as error:
            refusals['unverified_remote']=str(error)
        else:
            raise AssertionError('mismatched remote evidence must fail')
    # Real official-LM path uses the same checkpoint layer on saved requests.
    requests=ROOT/'artifacts/hwr_numpy_reconstruction_20261007_v63/requests.jsonl'
    small=OUTPUT/'audit_requests.jsonl';small.write_text('\n'.join(requests.read_text().splitlines()[:9])+'\n')
    audit_runner=Path(__file__).with_name('run_official_mini_lm_resumable_v1.py')

    def audit(name,stop=None):
        cmd=[sys.executable,str(audit_runner),'--run-dir',str(OUTPUT/name),'--run-id',name,'--requests',str(small),'--chunk-size','3']
        if stop:
            cmd+=['--stop-after-cursor',str(stop)]
        p=subprocess.run(cmd,capture_output=True,text=True,timeout=90)
        commands.append(dict(command=cmd,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr))
        assert p.returncode==0,commands[-1]
        return p
    audit('audit_baseline');audit('audit_interrupted',3);audit('audit_interrupted')
    assert (OUTPUT/'audit_baseline/final_outputs.json').read_bytes()==(OUTPUT/'audit_interrupted/final_outputs.json').read_bytes()
    audit_pointer=OUTPUT/'audit_interrupted/task_state.json';before=audit_pointer.read_bytes()
    assert 'skipped_completed_stage' in audit('audit_interrupted').stdout
    assert audit_pointer.read_bytes()==before
    final=OUTPUT/'audit_interrupted/final_outputs.json';valid_final=final.read_bytes()
    # Disposable final derived artifacts only; inference checkpoints remain untouched.
    for label in ['missing','truncated']:
        if label=='missing':
            final.unlink()
        else:
            final.write_bytes(b'[')
        (OUTPUT/'audit_interrupted/RECOVERY_SUMMARY.json').unlink(missing_ok=True)
        assert 'skipped_completed_stage' in audit('audit_interrupted').stdout
        assert final.read_bytes()==valid_final and audit_pointer.read_bytes()==before
        assert (OUTPUT/'audit_interrupted/RECOVERY_SUMMARY.json').exists()
    # Simulate the exact post-completion/pre-final-output boundary in an owned child.
    crash_script=OUTPUT/'completed_boundary_child.py'
    crash_script.write_text('import sys,os\nfrom pathlib import Path\nsys.path.insert(0,'+repr(str(Path(__file__).parent))+')\nimport run_official_mini_lm_resumable_v1 as runner\noriginal=runner.CheckpointStore.save_audit\ndef inject(self,*a,**kw):\n result=original(self,*a,**kw)\n if kw.get("completed"): os._exit(77)\n return result\nrunner.CheckpointStore.save_audit=inject\nrunner.main()\n')
    cmd=[sys.executable,str(crash_script),'--run-dir',str(OUTPUT/'audit_boundary_crash'),'--run-id','audit_boundary_crash','--requests',str(small),'--chunk-size','3']
    result=subprocess.run(cmd,capture_output=True,text=True,timeout=90)
    commands.append(dict(command=cmd,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
    assert result.returncode==77 and not (OUTPUT/'audit_boundary_crash/final_outputs.json').exists()
    assert 'skipped_completed_stage' in audit('audit_boundary_crash').stdout
    assert (OUTPUT/'audit_boundary_crash/final_outputs.json').read_bytes()==valid_final
    # Compatibility across a real result-only Git commit in an isolated owned fixture.
    repository=OUTPUT/'publication_fixture';repository.mkdir()
    def git(*args):
        return subprocess.check_output(['git','-c','user.name=Continuity Fixture','-c','user.email=fixture@example.invalid',*args],cwd=repository,text=True).strip()
    git('init','-q');(repository/'executable.txt').write_text('frozen executable asset\n')
    git('add','executable.txt');git('commit','-qm','fixture executable snapshot');original_commit=git('rev-parse','HEAD')
    def audit_store(path,source=original_commit):
        return CheckpointStore(path,run_id='bounded-audit-fixture',round_id='continuity-tests',stage_id='audit',kind='audit',source_commit=source,
                    config={'fixed':True},assets={'code':repository/'executable.txt','data':small})
    publication=OUTPUT/'publication_resume'
    with audit_store(publication).owner() as initial:
        initial.initialize('test publication compatibility');initial.save_audit([],cursor=0)
    (repository/'result.txt').write_text('published result only\n');git('add','result.txt');git('commit','-qm','result publication');new_commit=git('rev-parse','HEAD')
    assert new_commit!=original_commit
    with audit_store(publication,new_commit).owner() as compatible:
        assert compatible.load()[1]['cursor']==0 and compatible.identity['source_commit']==original_commit
        compatible.save_audit([{'record_id':'0'}],cursor=1,completed=True)
    fresh=audit_store(publication,new_commit)
    fresh_before=(publication/'task_state.json').read_bytes()
    try:
        with fresh.owner():
            fresh.save_audit([],cursor=0)
    except ResumeError as error:
        refusals['fresh_completed_mutation']=str(error)
    else:
        raise AssertionError('fresh instance overwrote completed stage')
    assert (publication/'task_state.json').read_bytes()==fresh_before
    stale_path=OUTPUT/'stale_instance';stale=audit_store(stale_path)
    with stale.owner():
        stale.initialize('stale test');stale.save_audit([],cursor=0)
    with audit_store(stale_path).owner() as current:
        current.load();current.save_audit([{'record_id':'0'}],cursor=1)
    try:
        with stale.owner():
            stale.save_audit([{'record_id':'0'},{'record_id':'1'}],cursor=2)
    except ResumeError as error:
        refusals['stale_mutation']=str(error)
    else:
        raise AssertionError('stale instance overwrote authoritative state')
    sizes=[]
    with audit_store(OUTPUT/'bounded_metadata').owner() as bounded:
        bounded.initialize('bounded metadata');bounded.save_audit([],cursor=0)
        bounded.record_remote(dict(sha='a'*40,verified_head_sha='a'*40,checkpoint_manifest_sha256=bounded.last['latest_checkpoint']['manifest_sha256'],scope='simulated fixture metadata only'))
        for i in range(40):
            bounded.save_audit([],cursor=0)
            sizes.append(len((bounded.root/'task_state.json').read_bytes()))
            remote=bounded.last['remote_preservation']
            assert 'previous_verified' not in remote and remote['last_verified']['sha']=='a'*40
            assert 'last_verified' not in remote['last_verified']
    assert max(sizes)-min(sizes)<100
    empty=OUTPUT/'empty_requests.jsonl';empty.write_text('')
    cmd=[sys.executable,str(audit_runner),'--run-dir',str(OUTPUT/'empty_audit'),'--run-id','empty','--requests',str(empty)]
    result=subprocess.run(cmd,capture_output=True,text=True,timeout=90)
    commands.append(dict(command=cmd,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
    assert result.returncode!=0 and 'empty inference requests' in result.stderr
    (repository/'executable.txt').write_text('changed executable\n')
    try:
        with audit_store(publication,new_commit).owner() as changed:
            changed.load()
    except ResumeError as error:
        refusals['genuine_code_change']=str(error)
    else:
        raise AssertionError('changed code resumed')
    canonical=ROOT.parent/'augmentation-models/canonical/project_symbol_head_checkpoint.pt'
    assert file_sha(canonical)=='04f8608aebcf6c02d45ad6f5735229b9eaa2c4b4e1be0db4793d02273ef2d00e'
    report=dict(schema='aiflow-continuity-verification/v1',status='passed',official_torch=torch.__version__,
                training_interruption='Owned disposable child exited 75 immediately after step-4 checkpoint; OS released flock; restart restored full state',
                exact_training_final_state_match=True,exact_optimizer_scheduler_rng_sampler_match=True,
                repeated_completed_training_skipped=True,concurrent_owner_rejected=True,refusals=refusals,
                prior_valid_checkpoint_retained=True,partial_does_not_override_valid=True,
                official_audit_exact_outputs_match=True,repeated_completed_audit_skipped=True,
                result_only_git_commit_resume=True,source_provenance_retained=True,
                missing_truncated_derived_outputs_restored=True,completed_boundary_crash_restored_without_inference=True,
                fresh_stale_mutations_refused=True,empty_input_refused=True,bounded_metadata_bytes=[min(sizes),max(sizes)],fault_injections=fault_results,
                canonical_sha256=file_sha(canonical),retired_final_tests_loaded=False,
                lifecycle_limit='Local process/filesystem restart only; no cloud-machine or execution-owner restart guarantee')
    atomic_bytes(OUTPUT/'commands.json',encoded(commands));atomic_bytes(OUTPUT/'report.json',encoded(report))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--child',type=Path);p.add_argument('--crash-at',type=int);p.add_argument('--lock-probe',type=Path);p.add_argument('--fault')
    args=p.parse_args()
    if args.child:
        train_child(args.child,args.crash_at,args.fault)
    elif args.lock_probe:
        try:
            with store(args.lock_probe).owner():
                pass
        except ResumeError as error:
            print(str(error));raise SystemExit(23)
        raise SystemExit('duplicate owner unexpectedly acquired lock')
    else:
        run_tests()
