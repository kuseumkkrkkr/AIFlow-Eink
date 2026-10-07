"""Linux research checkpoint boundaries; no supervisor or remote auto-publisher."""
import contextlib
import datetime
import fcntl
import hashlib
import json
import os
import platform
import pickle
import random
import uuid
from pathlib import Path

import numpy as np
import torch

SCHEMA='aiflow-research-checkpoint/v2'


def compatible_identity(identity):
    # A result-only commit changes provenance, not executable compatibility.
    return {k:v for k,v in identity.items() if k!='source_commit'}


class ResumeError(RuntimeError):
    pass


def encoded(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def sync_dir(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_bytes(path,data):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.partial')
    with temp.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp,path)
    sync_dir(path.parent)


def runtime():
    return dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,device='cpu')


def capture_rng():
    n=np.random.get_state()
    return dict(python=random.getstate(),numpy=dict(kind=n[0],keys=n[1].tolist(),position=n[2],gaussian=n[3],cached=n[4]),torch=torch.get_rng_state())


def restore_rng(state):
    random.setstate(state['python'])
    n=state['numpy']
    np.random.set_state((n['kind'],np.asarray(n['keys'],dtype=np.uint32),n['position'],n['gaussian'],n['cached']))
    torch.set_rng_state(state['torch'])


class CheckpointStore:
    def __init__(self,directory,*,run_id,round_id,stage_id,kind,source_commit,config,assets,fault_hook=None):
        if kind not in {'training','audit'}:
            raise ValueError('unsupported stage kind')
        self.root=Path(directory)
        self.root.mkdir(parents=True,exist_ok=True)
        self.assets={k:Path(v) for k,v in assets.items()}
        try:
            asset_hashes={k:file_sha(v) for k,v in self.assets.items()}
        except OSError as error:
            raise ResumeError('missing input/code/checkpoint asset: '+str(error)) from error
        self.identity=dict(run_id=run_id,round_id=round_id,stage_id=stage_id,kind=kind,
                           source_commit=source_commit,config_sha256=digest(encoded(config)),
                           asset_sha256=asset_hashes,runtime=runtime())
        self.identity_hash=digest(encoded(compatible_identity(self.identity)))
        self.owned=False
        self.last=None
        self.fault_hook=fault_hook

    def _fault(self,boundary,step):
        if self.fault_hook:
            self.fault_hook(boundary,step)

    def _require_owner(self):
        if not self.owned:
            raise ResumeError('mutation/load requires the run lock')

    def _assets_unchanged(self):
        for name,path in self.assets.items():
            if not path.is_file() or file_sha(path)!=self.identity['asset_sha256'][name]:
                raise ResumeError('missing or changed asset: '+name)

    @contextlib.contextmanager
    def owner(self):
        if self.owned:
            raise ResumeError('owner lock is not reentrant')
        with (self.root/'owner.lock').open('a+') as lock:
            try:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ResumeError('another live process owns this run') from error
            self.owned=True
            lock.seek(0);lock.truncate();lock.write(json.dumps(dict(pid=os.getpid(),run_id=self.identity['run_id'])))
            lock.flush();os.fsync(lock.fileno())
            try:
                yield self
            finally:
                self.owned=False
                fcntl.flock(lock,fcntl.LOCK_UN)

    def _read_checkpoint(self,entry):
        name=entry['directory']
        if '/' in name or not name.startswith('checkpoint-'):
            raise ResumeError('invalid checkpoint reference')
        directory=self.root/name
        raw=(directory/'manifest.json').read_bytes()
        if digest(raw)!=entry['manifest_sha256']:
            raise ResumeError('checkpoint manifest fingerprint mismatch')
        m=json.loads(raw)
        if m['schema']!=SCHEMA or compatible_identity(m['identity'])!=compatible_identity(self.identity) or m['identity_sha256']!=self.identity_hash:
            raise ResumeError('source/config/data/code/runtime identity mismatch')
        if m['phase'] not in {'checkpoint_complete','stage_complete'} or m['step']<0 or m['iteration']<0:
            raise ResumeError('incomplete or invalid checkpoint manifest')
        if m['payload']['name'] not in {'state.pt','audit.json'}:
            raise ResumeError('invalid payload reference')
        payload=directory/m['payload']['name']
        if file_sha(payload)!=m['payload']['sha256']:
            raise ResumeError('checkpoint missing/corrupt/truncated payload')
        if self.identity['kind']=='training':
            state=torch.load(payload,map_location='cpu',weights_only=True)
            required={'model','optimizer','scheduler','rng','sampler','step','iteration','gradient_accumulation_position'}
            if set(state)!=required or state['step']!=m['step'] or state['iteration']!=m['iteration']:
                raise ResumeError('training state is incomplete/inconsistent')
            if state['gradient_accumulation_position']!=0:
                raise ResumeError('training checkpoint is not an optimizer boundary')
            sampler=state['sampler']
            order=sampler['order']
            if sorted(order)!=list(range(len(order))) or not 0<=sampler['position']<=len(order):
                raise ResumeError('invalid data order/sampler position')
            rng=state['rng']
            if set(rng)!={'python','numpy','torch'} or rng['torch'].dtype!=torch.uint8:
                raise ResumeError('incomplete RNG state')
            if any(not torch.isfinite(t).all() for t in state['model'].values()):
                raise ResumeError('nonfinite model state')
        else:
            state=json.loads(payload.read_text())
            if set(state)!={'cursor','outputs','output_sha256'} or state['cursor']!=m['step']:
                raise ResumeError('audit cursor/payload mismatch')
            if state['cursor']!=len(state['outputs']) or digest(encoded(state['outputs']))!=state['output_sha256']:
                raise ResumeError('audit output fingerprint/cursor mismatch')
            ids=[r['record_id'] for r in state['outputs']]
            if len(ids)!=len(set(ids)):
                raise ResumeError('duplicate committed audit output')
        return m,state

    def load(self):
        self._require_owner()
        self._assets_unchanged()
        pointer=self.root/'task_state.json'
        if not pointer.exists():
            # An interrupted initial save must never masquerade as a fresh run.
            if list(self.root.glob('checkpoint-*')) or list(self.root.glob('*.partial')):
                raise ResumeError('orphan/partial checkpoint exists without valid task state')
            return None
        try:
            task=json.loads(pointer.read_text())
            if task['schema']!=SCHEMA or task['identity_sha256']!=self.identity_hash or compatible_identity(task['identity'])!=compatible_identity(self.identity):
                raise ResumeError('task source/config/data/code/runtime mismatch')
            self.identity['source_commit']=task['identity']['source_commit']
            if task['phase']=='pending_local' and task['latest_checkpoint'] is None:
                if list(self.root.glob('checkpoint-*')):
                    raise ResumeError('partial initial checkpoint needs explicit recovery')
                self.last=task
                return None
            manifest,state=self._read_checkpoint(task['latest_checkpoint'])
            if task['phase']!=manifest['phase'] or task['current_step']!=manifest['step']:
                raise ResumeError('task/manifest status mismatch')
            self.last=task
            return manifest,state
        except (OSError,KeyError,ValueError,TypeError,EOFError,RuntimeError,pickle.UnpicklingError) as error:
            if isinstance(error,ResumeError):
                raise
            raise ResumeError('missing/corrupt/incompatible checkpoint: '+str(error)) from error

    def initialize(self,next_action):
        self._require_owner()
        if (self.root/'task_state.json').exists():
            return self.load()
        if list(self.root.glob('checkpoint-*')):
            raise ResumeError('orphan checkpoint exists; refusing a fabricated fresh run')
        self.last=dict(schema=SCHEMA,identity=self.identity,identity_sha256=self.identity_hash,
                       phase='pending_local',status='pending_local',checkpoint_sequence=0,
                       latest_checkpoint=None,previous_checkpoint=None,last_verified_completed_stage=None,
                       current_iteration=0,current_step=0,next_action=next_action,
                       remote_preservation=dict(status='pending'))
        atomic_bytes(self.root/'task_state.json',encoded(self.last))
        return None

    def _save(self,state,*,step,iteration,completed,next_action,last_completed_stage=None):
        self._require_owner()
        self._assets_unchanged()
        previous=self.last
        self.load()  # authoritative pointer under flock before every mutation
        if previous is None:
            raise ResumeError('initialize/load authoritative state before saving')
        if previous['checkpoint_sequence']!=self.last['checkpoint_sequence'] or previous['latest_checkpoint']!=self.last['latest_checkpoint']:
            raise ResumeError('stale checkpoint instance; reload before saving')
        if self.last and self.last['phase']=='stage_complete':
            raise ResumeError('completed stage is immutable; skip it or use a new stage identity')
        if self.last and step<self.last['current_step']:
            raise ResumeError('checkpoint step cannot move backwards')
        sequence=1 if self.last is None else self.last['checkpoint_sequence']+1
        name='checkpoint-'+str(sequence).zfill(6)+'-'+uuid.uuid4().hex
        temp=self.root/(name+'.partial')
        temp.mkdir()
        payload=temp/('state.pt' if self.identity['kind']=='training' else 'audit.json')
        with payload.open('xb') as stream:
            if self.identity['kind']=='training':
                torch.save(state,stream)
            else:
                stream.write(encoded(state))
            stream.flush();os.fsync(stream.fileno())
        self._fault('payload_flushed',step)
        phase='stage_complete' if completed else 'checkpoint_complete'
        m=dict(schema=SCHEMA,identity=self.identity,identity_sha256=self.identity_hash,
               phase=phase,step=step,iteration=iteration,next_action=next_action,
               safe_boundary='after optimizer step with no accumulated gradients' if self.identity['kind']=='training' else 'committed unique output prefix; replay uncommitted inputs only',
               payload=dict(name=payload.name,sha256=file_sha(payload),bytes=payload.stat().st_size))
        atomic_bytes(temp/'manifest.json',encoded(m));sync_dir(temp)
        self._fault('manifest_flushed',step)
        final=self.root/name
        os.rename(temp,final);sync_dir(self.root)
        entry=dict(directory=name,manifest_sha256=file_sha(final/'manifest.json'))
        # Load and validate before promoting it; the previous pointer stays intact until here.
        self._read_checkpoint(entry)
        remote=self.last.get('remote_preservation',{}) if self.last else {}
        verified=remote if remote.get('status')=='verified' else remote.get('last_verified')
        if verified:
            verified={k:v for k,v in verified.items() if k not in {'previous_verified','last_verified'}}
        task=dict(schema=SCHEMA,identity=self.identity,identity_sha256=self.identity_hash,
                  phase=phase,status='local_checkpoint_complete',checkpoint_sequence=sequence,
                  latest_checkpoint=entry,previous_checkpoint=self.last['latest_checkpoint'] if self.last else None,
                  last_verified_completed_stage=self.identity['stage_id'] if completed else last_completed_stage,
                  current_iteration=iteration,current_step=step,next_action=next_action,
                  remote_preservation=dict(status='pending',last_verified=verified),
                  updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        self._fault('before_pointer_replace',step)
        atomic_bytes(self.root/'task_state.json',encoded(task))
        self._fault('after_pointer_replace',step)
        self.last=task
        return task

    def save_training(self,model,optimizer,scheduler,*,step,iteration,sampler,gradient_accumulation_position=0,completed=False,next_action='resume training'):
        if self.identity['kind']!='training' or gradient_accumulation_position!=0:
            raise ResumeError('only training optimizer boundaries can be saved')
        if any(p.grad is not None and torch.count_nonzero(p.grad).item()!=0 for p in model.parameters()):
            raise ResumeError('clear gradients after optimizer step before checkpointing')
        state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict() if scheduler else None,
                   rng=capture_rng(),sampler=sampler,step=step,iteration=iteration,gradient_accumulation_position=0)
        return self._save(state,step=step,iteration=iteration,completed=completed,next_action=next_action)

    def restore_training(self,state,model,optimizer,scheduler):
        self._require_owner()
        if (state['scheduler'] is None)!=(scheduler is None):
            raise ResumeError('scheduler presence mismatch')
        model.load_state_dict(state['model'],strict=True)
        optimizer.load_state_dict(state['optimizer'])
        if scheduler:
            scheduler.load_state_dict(state['scheduler'])
        restore_rng(state['rng'])

    def save_audit(self,outputs,*,cursor,completed=False,next_action='resume audit'):
        if self.identity['kind']!='audit':
            raise ResumeError('audit save requested for training stage')
        state=dict(cursor=cursor,outputs=outputs,output_sha256=digest(encoded(outputs)))
        return self._save(state,step=cursor,iteration=cursor,completed=completed,next_action=next_action)

    def record_remote(self,evidence):
        """Caller supplies connector-read evidence; this method never publishes."""
        self._require_owner()
        self.load()
        if not self.last or evidence['sha']!=evidence['verified_head_sha'] or len(evidence['sha'])!=40:
            raise ResumeError('missing/mismatched verified remote commit evidence')
        if evidence['checkpoint_manifest_sha256']!=self.last['latest_checkpoint']['manifest_sha256']:
            raise ResumeError('remote preservation evidence refers to another checkpoint')
        self.last['remote_preservation']=dict(status='verified',**evidence)
        self.last['status']='verified_remote_preservation'
        atomic_bytes(self.root/'task_state.json',encoded(self.last))

    def recovery_summary(self,path,optional_mirror=None):
        self._require_owner()
        if self.last is None:
            raise ResumeError('no committed state to summarize')
        summary={k:self.last[k] for k in ['schema','identity','phase','status','latest_checkpoint','last_verified_completed_stage','current_iteration','current_step','next_action','remote_preservation']}
        atomic_bytes(path,encoded(summary))
        if optional_mirror:
            try:
                atomic_bytes(optional_mirror,encoded(summary))
            except OSError as error:
                return dict(mirrored=False,error=str(error),authoritative=str(path))
        return dict(mirrored=bool(optional_mirror),authoritative=str(path))
