"""Recover an interrupted V37 arm without changing its frozen objective or logs.

Preserve input state/logs first; replay their overlap with exact serialized record
checks, then append only previously unrecorded steps. No evaluation or tuning.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import shutil

import numpy as np
import torch
from torch.nn import functional as F

from run_hwr_closest_rival_v37 import ARMS, load, require, write
from run_hwr_teacher_correct_feature_v35 import feature_loss
from hwr_closest_rival_gap_v37 import closest_gap_loss
from run_hwr_affine_distillation_experiment_v1 import _load_teacher


def prepare(out, arm, recovery_name):
    run, plan, _, _ = load(out)
    folder, recovery = out / arm, out / arm / recovery_name
    sha = run.direct.broad.previous._sha
    require(not recovery.exists(), 'Recovery evidence already exists')
    require(not (folder / 'completed.json').exists(), 'Arm already completed')
    require(not (folder / 'failure.json').exists(), 'Inspect recorded failure before recovery')
    require(plan['python'] == platform.python_version() and plan['numpy'] == np.__version__, 'Runtime changed')
    started = json.loads((folder / 'run_started.json').read_text())
    state = torch.load(folder / 'resume_state.pt', map_location='cpu', weights_only=True)
    require(started['arm'] == arm and started['plan_sha256'] == sha(out / 'frozen_plan.json'), 'Original run mismatch')
    require(state['arm'] == arm and state['plan_sha256'] == started['plan_sha256'], 'State mismatch')
    raw = (folder / 'training_microscope.jsonl').read_bytes()
    require(raw.endswith(b'\n'), 'Partial log row needs explicit inspection')
    rows = [json.loads(line) for line in raw.splitlines()]
    require([r['step'] for r in rows] == list(range(1, len(rows) + 1)), 'Log is not contiguous')
    require(0 < state['step'] <= len(rows) < plan['steps'] and state['step'] % 128 == 0, 'Invalid recovery range')
    recovery.mkdir()
    for name in ('resume_state.pt', 'training_microscope.jsonl', 'run_started.json'):
        shutil.copyfile(folder / name, recovery / name)
    evidence = dict(schema='aiflow-v37-recovery-plan/v1', arm=arm,
        source_sha256=sha(Path(__file__)), frozen_plan_sha256=sha(out / 'frozen_plan.json'),
        original_files={n: sha(recovery / n) for n in ('resume_state.pt', 'training_microscope.jsonl', 'run_started.json')},
        resume_step=state['step'], interrupted_log_steps=len(rows), overlap_steps=len(rows) - state['step'],
        logical_target_steps=plan['steps'], objective_changed=False, optimizer_reset=False,
        owned_optimizer_rows=0, crohme_rows=0, evaluation_during_recovery=False)
    write(recovery / 'recovery_plan.json', evidence)
    print(json.dumps(evidence), flush=True)


def resume(out, arm, recovery_name):
    run, plan, arrays, schedule = load(out)
    folder, recovery = out / arm, out / arm / recovery_name
    sha = run.direct.broad.previous._sha
    evidence = json.loads((recovery / 'recovery_plan.json').read_text())
    require(evidence['arm'] == arm and evidence['source_sha256'] == sha(Path(__file__)), 'Recovery source/arm changed')
    require(evidence['frozen_plan_sha256'] == sha(out / 'frozen_plan.json'), 'Frozen plan changed')
    require(not (folder / 'completed.json').exists() and not (recovery / 'resume_started.json').exists(), 'Recovery already attempted')
    for name, expected in evidence['original_files'].items():
        require(sha(recovery / name) == expected and sha(folder / name) == expected, f'Recovery input changed: {name}')
    require(shutil.disk_usage(out).free > 512 * 1024**2, 'Insufficient disk headroom')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(plan['seed'])
    model, labels, _ = _load_teacher(run.direct.broad.previous.CHECKPOINT, torch.device('cpu'))
    teacher, tv, _ = _load_teacher(run.direct.broad.previous.CHECKPOINT, torch.device('cpu'))
    teacher.requires_grad_(False)
    require(labels == tv and len(list(model.parameters())) == 57, 'Model contract changed')
    optimizer = torch.optim.AdamW(model.parameters(), lr=plan['lr'], weight_decay=plan['weight_decay'])
    state = torch.load(recovery / 'resume_state.pt', map_location='cpu', weights_only=True)
    require(state['arm'] == arm and state['plan_sha256'] == evidence['frozen_plan_sha256'], 'State provenance changed')
    require(state['step'] == evidence['resume_step'], 'State step changed')
    expected_group = {k: v for k, v in optimizer.state_dict()['param_groups'][0].items() if k != 'params'}
    actual_group = {k: v for k, v in state['optimizer']['param_groups'][0].items() if k != 'params'}
    require(expected_group == actual_group, 'AdamW parameters changed')
    require(len(state['optimizer']['state']) == 57 and all(float(v['step']) == state['step'] for v in state['optimizer']['state'].values()), 'AdamW state scope changed')
    model.load_state_dict(state['state_dict'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    torch.set_rng_state(state['torch_rng'])
    saved_lines = (recovery / 'training_microscope.jsonl').read_text().splitlines(keepends=True)
    write(recovery / 'resume_started.json', dict(pid=os.getpid(), plan_sha256=sha(recovery / 'recovery_plan.json')))
    gap_loss = closest_gap_loss if arm == 'closest_rival' else run.near_gap_loss
    layers, handles = {}, []
    def capture(name):
        def hook(module, inputs, value):
            require(torch.isfinite(value).all(), 'Nonfinite encoder output')
            layers[name] = dict(mean=float(value.detach().mean()), std=float(value.detach().std()))
        return hook
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.TransformerEncoderLayer):
            handles.append(module.register_forward_hook(capture(name)))
    steps, replayed, appended = state['step'], 0, 0
    try:
        with (folder / 'training_microscope.jsonl').open('a') as stream:
            for step in range(state['step'], plan['steps']):
                x, y = run.direct.batch(arrays, schedule, step, 'augmented_main')
                tx, truth = torch.from_numpy(x), torch.from_numpy(y)
                optimizer.zero_grad(set_to_none=True)
                h = model.encode(tx)
                z = model.math_head(h)
                with torch.no_grad():
                    th = teacher.encode(tx)
                    tz = teacher.math_head(th)
                correct = tz.argmax(1) == truth
                mask = torch.ones_like(correct)
                ce = F.cross_entropy(z, truth)
                feature = feature_loss(h, th, mask)
                gap, gap_mask, _ = gap_loss(z, tz, truth)
                require(torch.equal(gap_mask, correct), 'Gap mask changed')
                loss = ce + plan['feature_weight'] * feature + plan['gap_weight'] * gap
                require(torch.isfinite(loss), 'Nonfinite loss')
                loss.backward()
                require(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()), 'Invalid main gradient')
                require(all(p.grad is None and not p.requires_grad for p in teacher.parameters()), 'Teacher received gradient')
                gradients = {n: float(p.grad.norm()) for n, p in model.named_parameters()}
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                require(torch.isfinite(norm), 'Invalid clipping norm')
                optimizer.step()
                steps += 1
                row = dict(step=steps, ce=float(ce.detach()), feature=float(feature.detach()), gap=float(gap.detach()),
                    loss=float(loss.detach()), feature_rows=int(mask.sum()), teacher_correct=int(correct.sum()),
                    hard_ce_weight=1., hard_target_rows=64, augmented_rows=48,
                    gradient_l2=float(norm), parameter_gradient_l2=gradients,
                    encoder_layers=dict(layers), teacher_gradients_absent=True)
                serialized = json.dumps(row) + '\n'
                if steps <= len(saved_lines):
                    require(serialized == saved_lines[step], f'Overlap replay differs at step {steps}')
                    replayed += 1
                    if steps == len(saved_lines):
                        print(json.dumps(dict(event='overlap_replay_pass', arm=arm, steps=replayed)), flush=True)
                else:
                    stream.write(serialized)
                    stream.flush()
                    appended += 1
                if steps % 128 == 0:
                    temporary = folder / 'resume_state.tmp'
                    torch.save(dict(step=steps, state_dict=model.state_dict(), optimizer=optimizer.state_dict(),
                        torch_rng=torch.get_rng_state(), plan_sha256=sha(out / 'frozen_plan.json'), arm=arm), temporary)
                    temporary.replace(folder / 'resume_state.pt')
                    print(json.dumps(dict(event='v37_recovered_training', arm=arm, step=steps, loss=row['loss'])), flush=True)
    except Exception as exc:
        write(recovery / 'failure.json', dict(error=repr(exc), logical_step=steps, replayed_steps=replayed, appended_steps=appended))
        raise
    finally:
        for handle in handles:
            handle.remove()
    require(replayed == evidence['overlap_steps'] and len(saved_lines) + appended == plan['steps'], 'Recovery coverage changed')
    checkpoint = folder / 'main_trained.pt'
    torch.save(dict(schema='aiflow-teacher-correct-feature-main/v35', state_dict=model.state_dict(),
        math_labels=labels, auxiliary_labels=[], report=dict(input_contract=dict(observed_channel_mode='uniform-time'),
        arm=arm, product_adopted=False)), checkpoint)
    require(sha(run.direct.broad.previous.CHECKPOINT) == plan['canonical_sha256'], 'Canonical changed')
    write(folder / 'completed.json', dict(steps=steps, checkpoint_sha256=sha(checkpoint),
        microscope_sha256=sha(folder / 'training_microscope.jsonl'), evaluation_forwarded_during_training=0, canonical_unchanged=True))
    write(recovery / 'recovery_result.json', dict(status='pass', schema='aiflow-v37-recovery-result/v1',
        recovery_plan_sha256=sha(recovery / 'recovery_plan.json'), source_sha256=sha(Path(__file__)),
        logical_steps=steps, executed_updates_this_process=steps - state['step'], overlap_replayed_steps=replayed,
        new_logged_steps=appended, overlap_full_serialized_records_bit_exact=True,
        optimizer_state_and_rng_restored=True, original_log_prefix_preserved=True,
        final_checkpoint_sha256=sha(checkpoint), final_microscope_sha256=sha(folder / 'training_microscope.jsonl'),
        evaluation_forward_rows=0, owned_optimizer_rows=0, crohme_rows=0, product_adopted=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'resume'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--recovery-name', default='recovery_20261007')
    args = parser.parse_args()
    require(Path(args.recovery_name).name == args.recovery_name and args.recovery_name not in ('', '.', '..'), 'Recovery name must be a directory name')
    if args.mode == 'prepare': prepare(args.output, args.arm, args.recovery_name)
    else: resume(args.output, args.arm, args.recovery_name)
