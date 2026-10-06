"""Paired cloud experiment: mask only teacher-wrong feature-retention rows.

Both arms start from canonical with identical V25 data, schedule, seed, optimizer,
near-gap weight and budget. No development/owned data enters either optimizer.
"""
import argparse
import json
import os
from pathlib import Path
import shutil

import numpy as np
import torch
from torch.nn import functional as F

from cloud_hwr_snapshot import configure
from run_hwr_affine_distillation_experiment_v1 import _load_teacher, _predict_logits
from verify_hwr_broad_head_only_v23 import hit_metrics

ARMS = ('v33_control', 'teacher_correct_feature')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def feature_loss(h, target, mask):
    """Retain original per-batch denominator; remove only selected row contributions."""
    require(not target.requires_grad, 'Teacher embedding must be detached')
    return ((h - target).square().mean(1) * mask).mean()


def selftest():
    h = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    target = torch.zeros_like(h)
    mask = torch.tensor([True, False])
    loss = feature_loss(h, target, mask)
    gradient = torch.autograd.grad(loss, h, retain_graph=True)[0]
    require(torch.equal(gradient[1], torch.zeros(2)), 'Masked row receives a feature gradient')
    require(torch.allclose(gradient[0], h[0].detach() / 2), 'Feature denominator changed')
    require(torch.allclose(feature_loss(h, target, torch.ones(2, dtype=torch.bool)), F.mse_loss(h, target)), 'Control MSE changed')
    require(float(feature_loss(h, target, torch.zeros(2, dtype=torch.bool)).detach()) == 0., 'Empty mask is not zero')
    configure().selftest()


def load(out):
    run = configure()
    parent, arrays, schedule = run.load()
    sha = run.direct.broad.previous._sha
    plan = json.loads((out / 'frozen_plan.json').read_text())
    require(plan['code_sha256'] == sha(Path(__file__)), 'Experiment code changed')
    require(plan['runtime_sha256'] == sha(Path(__file__).with_name('cloud_hwr_snapshot.py')), 'Runtime configuration changed')
    require(plan['parent_plan_sha256'] == sha(run.OUT / 'frozen_plan.json'), 'Parent changed')
    for name in ('steps', 'seed', 'lr', 'weight_decay', 'feature_weight', 'gap_weight'):
        require(plan[name] == parent[name], f'Frozen parameter changed: {name}')
    return run, plan, arrays, schedule


def prepare(out):
    require(not out.exists(), 'Refusing to overwrite an experiment')
    selftest()
    run = configure()
    parent, _, _ = run.load()
    sha = run.direct.broad.previous._sha
    plan = dict(parent)
    plan.update(schema='aiflow-teacher-correct-feature-plan/v35',
                code_sha256=sha(Path(__file__)), runtime_sha256=sha(Path(__file__).with_name('cloud_hwr_snapshot.py')),
                parent_plan_sha256=sha(run.OUT / 'frozen_plan.json'), arms=list(ARMS),
                feature_rule='Control: mean MSE over all 64 rows. Challenger: mean per-row MSE * canonical-teacher-correct mask, still divided by 64.',
                coefficient_search=False, same_platform_control=True, torch=torch.__version__,
                no_interim_evaluation=True, no_best_epoch_selection=True,
                generated_labels_human_verified=False, owned_optimizer_rows=0, crohme_rows=0,
                product_adopted=False, cloud_upload=False)
    out.mkdir(parents=True)
    write(out / 'frozen_plan.json', plan)


def train(out, arm):
    run, plan, arrays, schedule = load(out)
    folder = out / arm
    require(not folder.exists(), 'Run exists; preserve and inspect its terminal state')
    memory = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    require(int(memory['MemAvailable'].split()[0]) > 1.5 * 1024**2, 'Available memory below 1.5 GiB')
    require(shutil.disk_usage(out).free > 512 * 1024**2, 'Disk headroom below 512 MiB')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(plan['seed'])
    model, labels, _ = _load_teacher(run.direct.broad.previous.CHECKPOINT, torch.device('cpu'))
    teacher, tv, _ = _load_teacher(run.direct.broad.previous.CHECKPOINT, torch.device('cpu'))
    teacher.requires_grad_(False)
    require(labels == tv and len(list(model.parameters())) == 57, 'Model contract mismatch')
    optimizer = torch.optim.AdamW(model.parameters(), lr=plan['lr'], weight_decay=plan['weight_decay'])
    folder.mkdir()
    sha = run.direct.broad.previous._sha
    write(folder / 'run_started.json', dict(pid=os.getpid(), plan_sha256=sha(out / 'frozen_plan.json'), arm=arm))
    layers = {}
    handles = []
    def capture(name):
        def hook(module, inputs, value):
            require(torch.isfinite(value).all(), 'Nonfinite encoder output')
            layers[name] = dict(mean=float(value.detach().mean()), std=float(value.detach().std()))
        return hook
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.TransformerEncoderLayer):
            handles.append(module.register_forward_hook(capture(name)))
    steps = 0
    try:
        with (folder / 'training_microscope.jsonl').open('x') as stream:
            for step in range(plan['steps']):
                x, y = run.direct.batch(arrays, schedule, step, 'augmented_main')
                tx, truth = torch.from_numpy(x), torch.from_numpy(y)
                optimizer.zero_grad(set_to_none=True)
                h = model.encode(tx)
                z = model.math_head(h)
                with torch.no_grad():
                    th = teacher.encode(tx)
                    tz = teacher.math_head(th)
                correct = tz.argmax(1) == truth
                mask = correct if arm == 'teacher_correct_feature' else torch.ones_like(correct)
                ce = F.cross_entropy(z, truth)
                feature = feature_loss(h, th, mask)
                gap, gap_mask, _ = run.near_gap_loss(z, tz, truth)
                require(torch.equal(gap_mask, correct), 'Gap teacher mask mismatch')
                loss = ce + plan['feature_weight'] * feature + plan['gap_weight'] * gap
                require(torch.isfinite(loss), 'Nonfinite loss')
                if step == 0:
                    gz = torch.autograd.grad(ce, z, retain_graph=True)[0]
                    require(torch.allclose(gz, (z.detach().softmax(1) - F.one_hot(truth, 372)) / 64, atol=1e-7), 'CE does not reach all rows')
                    require(float(feature.detach()) < 1e-10 and float(gap.detach()) < 1e-10, 'Cold initialization mismatch')
                loss.backward()
                require(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()), 'Invalid main gradient')
                require(all(p.grad is None and not p.requires_grad for p in teacher.parameters()), 'Teacher received a gradient')
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
                stream.write(json.dumps(row) + '\n')
                stream.flush()
                if steps % 128 == 0:
                    temporary = folder / 'resume_state.tmp'
                    torch.save(dict(step=steps, state_dict=model.state_dict(), optimizer=optimizer.state_dict(),
                                    torch_rng=torch.get_rng_state(), plan_sha256=sha(out / 'frozen_plan.json'),
                                    arm=arm), temporary)
                    temporary.replace(folder / 'resume_state.pt')
                    print(json.dumps(dict(event='v35_training', arm=arm, step=steps, loss=row['loss'])), flush=True)
    except Exception as exc:
        write(folder / 'failure.json', dict(error=repr(exc), optimizer_steps=steps))
        raise
    finally:
        for handle in handles:
            handle.remove()
    checkpoint = folder / 'main_trained.pt'
    torch.save(dict(schema='aiflow-teacher-correct-feature-main/v35', state_dict=model.state_dict(),
                    math_labels=labels, auxiliary_labels=[],
                    report=dict(input_contract=dict(observed_channel_mode='uniform-time'), arm=arm,
                                product_adopted=False)), checkpoint)
    require(sha(run.direct.broad.previous.CHECKPOINT) == plan['canonical_sha256'], 'Canonical changed')
    write(folder / 'completed.json', dict(steps=steps, checkpoint_sha256=sha(checkpoint),
                                          microscope_sha256=sha(folder / 'training_microscope.jsonl'),
                                          evaluation_forwarded_during_training=0, canonical_unchanged=True))


def evaluate(out):
    run, plan, _, _ = load(out)
    require(not (out / 'comparison_result.json').exists(), 'Comparison exists')
    sha = run.direct.broad.previous._sha
    _, data, folds, _ = run.direct.broad.previous.global_load(run.direct.broad.previous.OUTPUT)
    owned = run.direct.broad.previous.ROOT / 'artifacts/hwr_owned_formula_transfer_20261005_v21'
    inputs = np.load(owned / 'inputs.npy', allow_pickle=False)
    rows = json.loads((owned / 'input_rows.json').read_text())
    metrics = {}
    for arm in ARMS:
        folder = out / arm
        done = json.loads((folder / 'completed.json').read_text())
        require(done['steps'] == 2400 and done['checkpoint_sha256'] == sha(folder / 'main_trained.pt'), 'Incomplete arm')
        model, labels, _ = _load_teacher(folder / 'main_trained.pt', torch.device('cpu'))
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        development = {}
        for fold in (1, 2):
            z = _predict_logits(model, np.array(data['features'][folds[fold]], copy=True), torch.device('cpu'), 32)
            np.save(folder / f'fold{fold}_logits.npy', z, allow_pickle=False)
            development[str(fold)] = hit_metrics(z, np.array(data['labels'][folds[fold]], copy=True), labels)
        torch.set_num_threads(1)
        z = _predict_logits(model, inputs, torch.device('cpu'), 128)
        np.save(folder / 'owned_logits.npy', z, allow_pickle=False)
        top = np.argsort(-z, axis=1, kind='stable')[:, :5]
        records = []
        for row in rows:
            tokens = [[labels[int(i)] for i in order] for order in top[row['start']:row['stop']]]
            records.append(dict(row, top1_tokens=[t[0] for t in tokens], top5_tokens=tokens))
        write(folder / 'owned_records.json', records)
        cohorts = {}
        for cohort in ('all', 'legacy_96', 'codex_reviewed_53'):
            subset = [r for r in records if cohort == 'all' or r['cohort'] == cohort]
            cohorts[cohort] = dict(formulas=len(subset), tokens=sum(len(r['truth_tokens']) for r in subset),
                                  top1_token_hits=sum(sum(a == b for a, b in zip(r['truth_tokens'], r['top1_tokens'], strict=True)) for r in subset),
                                  formula_top1_exact=sum(r['truth_tokens'] == r['top1_tokens'] for r in subset),
                                  formula_top5_complete=sum(all(t in k for t, k in zip(r['truth_tokens'], r['top5_tokens'], strict=True)) for r in subset))
        metrics[arm] = dict(development=development, owned=cohorts)
    write(out / 'comparison_result.json', dict(schema='aiflow-teacher-correct-feature-comparison/v35', metrics=metrics,
                                               same_platform_control=True, product_adopted=False,
                                               limits='Consumed oracle-group HWR diagnostic. Raw grouping/decoder and fresh writer/device acceptance not run.'))
    print(json.dumps(metrics), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'train', 'evaluate', 'selftest'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--arm', choices=ARMS)
    args = parser.parse_args()
    if args.mode == 'train':
        require(args.arm is not None, 'Select a training arm')
        train(args.output, args.arm)
    elif args.mode == 'prepare':
        prepare(args.output)
    elif args.mode == 'evaluate':
        evaluate(args.output)
    else:
        selftest()
