"""Check the archived V33 run across CPU platforms without rewriting its certificate.

File hashes, class decisions and metrics must match exactly. Floating-point
scores/gradient norms are reported separately with explicit portable tolerances.
"""
import argparse
import json
import platform
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from cloud_hwr_snapshot import configure
from run_hwr_affine_distillation_experiment_v1 import _load_teacher, _predict_logits
from verify_hwr_broad_head_only_v23 import hit_metrics


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def verify():
    run = configure()
    plan, arrays, schedule = run.load()
    root = run.OUT
    sha = run.direct.broad.previous._sha
    folder = root / 'margin_retained_main'
    done = json.loads((folder / 'completed.json').read_text())
    started = json.loads((folder / 'run_started.json').read_text())
    cert = json.loads((root / 'independent_verification.json').read_text())
    result = json.loads((root / 'comparison_result.json').read_text())
    require(started['plan_sha256'] == sha(root / 'frozen_plan.json'), 'Run/plan hash mismatch')
    require(done['steps'] == 2400 and done['evaluation_forwarded_during_training'] == 0, 'Run not complete')
    require(done['checkpoint_sha256'] == cert['checkpoint_sha256'] == sha(folder / 'main_trained.pt'), 'Checkpoint hash mismatch')
    require(done['microscope_sha256'] == sha(folder / 'training_microscope.jsonl'), 'Log hash mismatch')
    require(cert['comparison_sha256'] == sha(root / 'comparison_result.json'), 'Comparison hash mismatch')
    require(cert['verifier_sha256'] == sha(Path(__file__).with_name('verify_hwr_near_rival_training_v33.py')), 'Historical verifier hash mismatch')
    source = arrays['shared_schedule'].reshape(-1)
    expected = np.concatenate((arrays['population_labels'][schedule['original_population_indices']],
                               arrays['population_labels'][source[schedule['augmented_view_indices']]]), axis=1)
    require(np.array_equal(expected, schedule['shared_target_ids']) and expected.shape == (2400, 64), 'TRAIN targets mismatch')
    require(np.unique(expected).size == 371 and np.unique(schedule['augmented_view_indices']).size == 68726, 'TRAIN coverage mismatch')
    count = 0
    max_arithmetic_delta = 0.
    with (folder / 'training_microscope.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            count += 1
            require(row['step'] == count and row['hard_ce_weight'] == 1. and row['kl_weight'] == 0., 'Step/CE mismatch')
            require(row['feature_weight'] == plan['feature_weight'] and row['gap_weight'] == plan['gap_weight'], 'Weight mismatch')
            require(row['teacher_gradients_absent'] and row['hard_target_rows'] == 64 and row['augmented_rows'] == 48, 'Gradient/data scope mismatch')
            require(len(row['encoder_layers']) == 4 and len(row['parameter_gradient_l2']) == 57, 'Layer/gradient count mismatch')
            require(all(np.isfinite(v) for v in row['parameter_gradient_l2'].values()), 'Nonfinite gradient')
            require(all(np.isfinite(v) for layer in row['encoder_layers'].values() for v in layer.values()), 'Nonfinite layer')
            require(0 <= row['active_competitor_gaps'] <= row['teacher_correct_rows'] * 9, 'Near-pair bound mismatch')
            for i in range(4):
                values = [v for n, v in row['parameter_gradient_l2'].items() if n.startswith(f'encoder.layers.{i}.')]
                require(len(values) == 12 and max(values) > 0, 'Inactive encoder layer')
            delta = max(abs(row['hard_ce'] - .25 * row['ce_real_region'] - .75 * row['ce_augmented_region']),
                        abs(row['weighted_feature'] - plan['feature_weight'] * row['feature_mse']),
                        abs(row['weighted_gap'] - plan['gap_weight'] * row['gap_loss']),
                        abs(row['loss'] - row['hard_ce'] - row['weighted_feature'] - row['weighted_gap']))
            max_arithmetic_delta = max(max_arithmetic_delta, delta)
            require(delta < 1e-6, 'Recorded loss arithmetic mismatch')
    require(count == 2400, 'Incomplete log')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    run.selftest()
    model, labels, _ = _load_teacher(folder / 'main_trained.pt', torch.device('cpu'))
    baseline, baseline_labels, _ = _load_teacher(run.BASE, torch.device('cpu'))
    teacher, teacher_labels, _ = _load_teacher(run.direct.broad.previous.CHECKPOINT, torch.device('cpu'))
    teacher.requires_grad_(False)
    require(labels == baseline_labels == teacher_labels, 'Vocabulary mismatch')
    require(sum(not torch.equal(p, teacher.state_dict()[n]) for n, p in model.named_parameters()) == 57, 'Parameter updates mismatch')
    require(all(torch.isfinite(p).all() for p in model.parameters()), 'Nonfinite checkpoint')
    gradient_deltas = []
    ratios = []
    for saved in plan['calibration_rows']:
        x, y = run.direct.batch(arrays, schedule, saved['schedule_step'], 'augmented_main')
        tx, truth = torch.from_numpy(x), torch.from_numpy(y)
        with torch.no_grad():
            tz = teacher.math_head(teacher.encode(tx))
        z = baseline.math_head(baseline.encode(tx))
        ce = F.cross_entropy(z, truth)
        gap, mask, active = run.near_gap_loss(z, tz, truth)
        params = list(baseline.parameters())
        cg = torch.autograd.grad(ce, params, retain_graph=True)
        gg = torch.autograd.grad(gap, params)
        cn = float(torch.sqrt(sum(g.square().sum() for g in cg)))
        gn = float(torch.sqrt(sum(g.square().sum() for g in gg)))
        require(int(mask.sum()) == saved['teacher_correct_rows'] and active == saved['active_near_pairs'], 'TRAIN mask decisions mismatch')
        require(np.isclose(cn, saved['ce_all57_gradient_l2'], rtol=1e-5, atol=1e-5), 'CE gradient outside portable tolerance')
        require(np.isclose(gn, saved['near_all57_gradient_l2'], rtol=1e-5, atol=1e-5), 'Gap gradient outside portable tolerance')
        gradient_deltas.append(max(abs(cn - saved['ce_all57_gradient_l2']), abs(gn - saved['near_all57_gradient_l2'])))
        ratios.append(cn / gn)
    recomputed_weight = float(np.clip(np.median(ratios), 1e-3, 100.))
    require(np.isclose(recomputed_weight, plan['gap_weight'], rtol=1e-5, atol=1e-5), 'TRAIN coefficient outside portable tolerance')
    _, data, folds, _ = run.direct.broad.previous.global_load(run.direct.broad.previous.OUTPUT)
    scored = result['metrics']['near_rival_main']
    parity = {}
    for fold in (1, 2):
        z = _predict_logits(model, np.array(data['features'][folds[fold]], copy=True), torch.device('cpu'), 32)
        saved = np.load(root / f'near_rival_main_fold{fold}_logits.npy', allow_pickle=False)
        parity[f'fold{fold}'] = check_logits(z, saved)
        require(hit_metrics(z, np.array(data['labels'][folds[fold]], copy=True), labels) == scored['development'][str(fold)], 'Development metrics mismatch')
    import audit_hwr_owned_formula_transfer_v21 as owned
    torch.set_num_threads(1)
    z = _predict_logits(model, np.load(owned.OUT / 'inputs.npy', allow_pickle=False), torch.device('cpu'), owned.BATCH)
    parity['owned'] = check_logits(z, np.load(root / 'near_rival_main_owned_logits.npy', allow_pickle=False))
    records = json.loads((root / 'near_rival_main_owned_records.json').read_text())
    top = np.argsort(-z, axis=1, kind='stable')[:, :5]
    for row in records:
        tokens = [[labels[int(i)] for i in indices] for indices in top[row['start']:row['stop']]]
        require(tokens == row['top5_tokens'] and [v[0] for v in tokens] == row['top1_tokens'], 'Owned top-5 tokens mismatch')
    for cohort in ('all', 'legacy_96', 'codex_reviewed_53'):
        require(owned.metrics(records, cohort) == scored['owned'][cohort], 'Owned metrics mismatch')
    require(sha(run.direct.broad.previous.CHECKPOINT) == plan['canonical_sha256'], 'Canonical modified')
    return dict(schema='aiflow-cloud-v33-verification/v1', status='pass',
                platform=platform.platform(), python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                historical_verifier_unchanged=True, training_records_checked=count,
                max_recorded_arithmetic_delta=max_arithmetic_delta,
                max_train_gradient_absolute_delta=max(gradient_deltas),
                gradient_tolerance=dict(atol=1e-5, rtol=1e-5),
                archived_gap_weight=plan['gap_weight'], recomputed_gap_weight=recomputed_weight,
                archived_coefficient_changed=False, logit_parity=parity,
                internal_top1=sum(scored['development'][str(f)]['all']['top1_hits'] for f in (1, 2)),
                owned=scored['owned']['all'], checkpoint_sha256=done['checkpoint_sha256'],
                optimizer_steps=0, product_adopted=False,
                limits='Portable CPU replay; historical intermediate forwards not replayed. Owned oracle diagnostics are already consumed. Bit-exact cross-platform logits are reported, not assumed.')


def check_logits(actual, saved):
    require(actual.shape == saved.shape and np.isfinite(actual).all(), 'Invalid replay logits')
    delta = float(np.abs(actual - saved).max())
    a = np.argsort(-actual, axis=1, kind='stable')[:, :5]
    b = np.argsort(-saved, axis=1, kind='stable')[:, :5]
    require(delta <= 1e-4, f'Logits outside portable absolute tolerance: {delta}')
    require(np.array_equal(a, b), 'Top-5 ordering changed')
    return dict(rows=len(actual), max_absolute_delta=delta, absolute_tolerance=1e-4,
                bit_exact=bool(np.array_equal(actual, saved)), top5_order_exact=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = verify()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
