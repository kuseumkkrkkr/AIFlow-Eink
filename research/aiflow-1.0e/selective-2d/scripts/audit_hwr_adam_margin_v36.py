"""Replay the last 96 V35 control updates and inspect six TRAIN failures.

No new optimization budget, ownership labels, development forwards or tuning.
Gradient projections describe local directions; replay records actual AdamW steps.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from run_hwr_teacher_correct_feature_v35 import feature_loss, load, require, write
from run_hwr_affine_distillation_experiment_v1 import _load_teacher


def projection(margin_gradient, loss_gradient):
    return -float(sum((a.double() * b.double()).sum() for a, b in zip(margin_gradient, loss_gradient, strict=True)))


def logits(model, x):
    return model.math_head(model.encode(x))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Refusing to overwrite an audit')
    run, plan, arrays, schedule = load(args.parent)
    sha = run.direct.broad.previous._sha
    folder = args.parent / 'v33_control'
    done = json.loads((folder / 'completed.json').read_text())
    require(done['steps'] == 2400 and sha(folder / 'main_trained.pt') == done['checkpoint_sha256'], 'Control incomplete')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    final, labels, _ = _load_teacher(folder / 'main_trained.pt', torch.device('cpu'))
    teacher, tl, _ = _load_teacher(run.direct.broad.previous.CHECKPOINT, torch.device('cpu'))
    require(labels == tl, 'Vocabulary changed')
    teacher.requires_grad_(False)
    probes, records = [], []
    for calibration in plan['calibration_rows']:
        step = calibration['schedule_step']
        x, y = run.direct.batch(arrays, schedule, step, 'augmented_main')
        tx, truth = torch.from_numpy(x), torch.from_numpy(y)
        h = final.encode(tx)
        z = final.math_head(h)
        with torch.no_grad():
            th = teacher.encode(tx)
            tz = teacher.math_head(th)
        correct = tz.argmax(1) == truth
        wrong = correct & (z.detach().argmax(1) != truth)
        near, _ = run.near_mask(z, tz, truth)
        ce = F.cross_entropy(z, truth)
        feature = feature_loss(h, th, torch.ones_like(correct))
        gap, _, _ = run.near_gap_loss(z, tz, truth)
        params = list(final.parameters())
        terms = dict(ce=ce, feature=plan['feature_weight'] * feature, gap=plan['gap_weight'] * gap)
        gradients = {}
        for name, term in terms.items():
            raw = torch.autograd.grad(term, params, retain_graph=True, allow_unused=name == 'feature')
            for (key, _), grad in zip(final.named_parameters(), raw, strict=True):
                require(grad is not None or name == 'feature' and key.startswith('math_head.'), 'Unexpected disconnection')
            gradients[name] = tuple(torch.zeros_like(p) if g is None else g for p, g in zip(params, raw, strict=True))
        total = torch.autograd.grad(sum(terms.values()), params, retain_graph=True)
        require(all(torch.allclose(a, sum(parts), atol=2e-6, rtol=1e-5)
                    for a, parts in zip(total, zip(*gradients.values(), strict=True), strict=True)), 'Gradient sum mismatch')
        for row in wrong.nonzero().flatten().tolist():
            target, rival = int(truth[row]), int(z.detach()[row].argmax())
            margin = z[row, target] - z[row, rival]
            mg = torch.autograd.grad(margin, params, retain_graph=True)
            require(all(torch.isfinite(g).all() for g in mg), 'Nonfinite margin gradient')
            projections = {name: projection(mg, g) for name, g in gradients.items()}
            projections['total'] = projection(mg, total)
            require(np.isclose(projections['total'], sum(projections[n] for n in terms), atol=1e-5, rtol=1e-5), 'Projection sum mismatch')
            tm = float((tz[row, target] - tz[row, rival]).detach())
            require(bool(near[row, rival]) and tm > 0, 'Decision rival not penalized')
            record = dict(probe_id=len(probes), schedule_step=step, batch_row=row,
                          region='original' if row < 16 else 'augmented',
                          truth=labels[target], rival=labels[rival], final_margin=float(margin.detach()),
                          teacher_margin=tm, selected_pairs=int(near.sum()), teacher_correct_rows=int(correct.sum()),
                          decision_pair_normalization_ratio=float(near.sum()) / float(correct.sum()),
                          negative_sgd_margin_derivative=projections)
            records.append(record)
            probes.append((np.array(x[row], copy=True), target, rival))
    require(len(probes) == 6, 'V35 residual scope changed')
    px = torch.from_numpy(np.stack([p[0] for p in probes]))
    targets = torch.tensor([p[1] for p in probes])
    rivals = torch.tensor([p[2] for p in probes])
    index = torch.arange(6)
    state = torch.load(folder / 'resume_state.pt', map_location='cpu', weights_only=True)
    require(state['step'] == 2304 and state['arm'] == 'v33_control' and state['plan_sha256'] == sha(args.parent / 'frozen_plan.json'), 'Resume state mismatch')
    model, _, _ = _load_teacher(run.direct.broad.previous.CHECKPOINT, torch.device('cpu'))
    model.load_state_dict(state['state_dict'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=plan['lr'], weight_decay=plan['weight_decay'])
    optimizer.load_state_dict(state['optimizer'])
    torch.set_rng_state(state['torch_rng'])
    saved = [json.loads(line) for line in (folder / 'training_microscope.jsonl').read_text().splitlines()]
    trajectory = []
    with torch.no_grad():
        pz = logits(model, px)
        before = (pz[index, targets] - pz[index, rivals]).clone()
    initial = before.tolist()
    max_delta = 0.
    for step in range(2304, 2400):
        x, y = run.direct.batch(arrays, schedule, step, 'augmented_main')
        tx, truth = torch.from_numpy(x), torch.from_numpy(y)
        optimizer.zero_grad(set_to_none=True)
        h = model.encode(tx)
        z = model.math_head(h)
        with torch.no_grad():
            th = teacher.encode(tx)
            tz = teacher.math_head(th)
        ce = F.cross_entropy(z, truth)
        feature = feature_loss(h, th, torch.ones(64, dtype=torch.bool))
        gap, _, _ = run.near_gap_loss(z, tz, truth)
        loss = ce + plan['feature_weight'] * feature + plan['gap_weight'] * gap
        loss.backward()
        norms = {n: float(p.grad.norm()) for n, p in model.named_parameters()}
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        actual = dict(ce=float(ce.detach()), feature=float(feature.detach()), gap=float(gap.detach()),
                      loss=float(loss.detach()), gradient_l2=float(norm))
        delta = max(abs(actual[n] - saved[step][n]) for n in actual)
        delta = max(delta, max(abs(value - saved[step]['parameter_gradient_l2'][name]) for name, value in norms.items()))
        max_delta = max(max_delta, delta)
        require(delta == 0., 'Same-platform replay differs from original loss/gradient records')
        optimizer.step()
        with torch.no_grad():
            pz = logits(model, px)
            after = pz[index, targets] - pz[index, rivals]
            trajectory.append(dict(step=step + 1, margin=after.tolist(), actual_adamw_margin_delta=(after - before).tolist()))
            before = after.clone()
        if (step + 1) % 32 == 0:
            print(json.dumps(dict(event='v36_replay', step=step + 1)), flush=True)
    require(all(torch.equal(value, final.state_dict()[name]) for name, value in model.state_dict().items()), 'Final replay weights differ')
    require(sha(folder / 'main_trained.pt') == done['checkpoint_sha256'] and sha(run.direct.broad.previous.CHECKPOINT) == plan['canonical_sha256'], 'Checkpoint changed')
    summaries = []
    for i, record in enumerate(records):
        deltas = [t['actual_adamw_margin_delta'][i] for t in trajectory]
        summaries.append(dict(record, initial_margin=initial[i], replay_final_margin=trajectory[-1]['margin'][i],
                              net_adamw_margin_change=trajectory[-1]['margin'][i] - initial[i],
                              improving_updates=sum(d > 0 for d in deltas), harming_updates=sum(d < 0 for d in deltas)))
    args.output.mkdir(parents=True)
    np.save(args.output / 'probe_inputs.npy', px.numpy(), allow_pickle=False)
    write(args.output / 'audit_result.json', dict(schema='aiflow-adam-margin-replay/v36', status='pass',
        code_sha256=sha(Path(__file__)), parent_plan_sha256=sha(args.parent / 'frozen_plan.json'),
        source_checkpoint_sha256=done['checkpoint_sha256'], resume_state_sha256=sha(folder / 'resume_state.pt'),
        replayed_optimizer_steps=96, new_training_budget=0, original_losses_and_57_gradient_norms_bit_exact=True,
        max_replay_record_delta=max_delta, final_state_dict_bit_exact=True,
        records=summaries, trajectory=trajectory, owned_labels_read=0, development_forward_rows=0,
        crohme_rows=0, coefficient_selection=False, canonical_unchanged=True, product_adopted=False,
        limits='Six post-selected fitted TRAIN errors; same last-96-step replay, not all historical updates or fresh acceptance. SGD projections and actual AdamW trajectory are distinct.'))
    print(json.dumps(summaries), flush=True)


if __name__ == '__main__':
    main()
