"""Compare V29/V33 retention and gradient conflict on their eight frozen TRAIN batches.

No ownership labels, development inputs, optimizer, or coefficient search are used.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from cloud_hwr_snapshot import configure
from run_hwr_affine_distillation_experiment_v1 import _load_teacher


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def norm(values):
    return float(torch.sqrt(sum(v.double().square().sum() for v in values)))


def cosine(first, second):
    denominator = norm(first) * norm(second)
    return float(sum((a.double() * b.double()).sum() for a, b in zip(first, second, strict=True))) / denominator if denominator else 0.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    run = configure()
    plan, arrays, schedule = run.load()
    sha = run.direct.broad.previous._sha
    paths = {'v29': run.BASE, 'v33': run.OUT / 'margin_retained_main/main_trained.pt'}
    hashes = {name: sha(path) for name, path in paths.items()}
    for name, experiment in [('v29', run.algorithm.OUT), ('v33', run.OUT)]:
        certificate = json.loads((experiment / 'independent_verification.json').read_text())
        require(certificate['status'] == 'pass' and certificate['checkpoint_sha256'] == hashes[name], 'Checkpoint/certificate mismatch')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    teacher, labels, _ = _load_teacher(run.direct.broad.previous.CHECKPOINT, torch.device('cpu'))
    teacher.requires_grad_(False)
    models = {}
    for name, path in paths.items():
        model, vocabulary, _ = _load_teacher(path, torch.device('cpu'))
        require(vocabulary == labels and len(list(model.parameters())) == 57, 'Model contract mismatch')
        models[name] = model
    records = []
    for saved in plan['calibration_rows']:
        step = saved['schedule_step']
        x, y = run.direct.batch(arrays, schedule, step, 'augmented_main')
        tx, truth = torch.from_numpy(x), torch.from_numpy(y)
        with torch.no_grad():
            th = teacher.encode(tx)
            tz = teacher.math_head(th)
        teacher_correct = tz.argmax(1) == truth
        for name, model in models.items():
            h = model.encode(tx)
            z = model.math_head(h)
            ce = F.cross_entropy(z, truth)
            feature = F.mse_loss(h, th)
            near, correct, _ = run.near_gap_loss(z, tz, truth)
            actual_gap = run.algorithm.gap_loss(z, tz, truth)[0] if name == 'v29' else near
            weight = json.loads((run.algorithm.OUT / 'frozen_plan.json').read_text())['gap_weight'] if name == 'v29' else plan['gap_weight']
            params = list(model.parameters())
            terms = {'ce': ce, 'feature': plan['feature_weight'] * feature, 'gap': weight * actual_gap}
            gradients = {}
            for key, value in terms.items():
                raw = torch.autograd.grad(value, params, retain_graph=True, allow_unused=(key == 'feature'))
                # Embedding MSE has no path to the classifier head's two tensors.
                for (parameter_name, parameter), grad in zip(model.named_parameters(), raw, strict=True):
                    require(grad is not None or key == 'feature' and parameter_name.startswith('math_head.'),
                            f'Unexpected disconnected gradient: {key}/{parameter_name}')
                gradients[key] = tuple(torch.zeros_like(p) if g is None else g for p, g in zip(params, raw, strict=True))
            require(all(torch.isfinite(g).all() for values in gradients.values() for g in values), 'Nonfinite gradient')
            feature_parts = {}
            feature_rows = (h - th).square().mean(1)
            for key, mask in [('teacher_correct', teacher_correct), ('teacher_wrong', ~teacher_correct)]:
                value = plan['feature_weight'] * (feature_rows * mask).mean()
                raw = torch.autograd.grad(value, params, retain_graph=True, allow_unused=True)
                for (parameter_name, parameter), grad in zip(model.named_parameters(), raw, strict=True):
                    require(grad is not None or parameter_name.startswith('math_head.'), 'Disconnected feature partition')
                feature_parts[key] = tuple(torch.zeros_like(p) if g is None else g for p, g in zip(params, raw, strict=True))
            require(all(torch.allclose(a + b, whole, atol=2e-6, rtol=1e-5)
                        for a, b, whole in zip(*feature_parts.values(), gradients['feature'], strict=True)),
                    'Feature partition gradient composition mismatch')
            total = [sum(parts) for parts in zip(*gradients.values(), strict=True)]
            direct_total = torch.autograd.grad(sum(terms.values()), params)
            require(all(torch.allclose(a, b, atol=2e-6, rtol=1e-5) for a, b in zip(total, direct_total, strict=True)), 'Gradient composition mismatch')
            prediction = z.detach().argmax(1)
            wrong = correct & (prediction != truth)
            margins = z.detach().gather(1, truth[:, None]).squeeze(1) - z.detach().masked_fill(F.one_hot(truth, 372).bool(), -torch.inf).max(1).values
            row_data = []
            for region, first, last in [('original', 0, 16), ('augmented', 16, 64)]:
                row_data.append(dict(region=region, rows=last - first,
                                     teacher_correct=int(teacher_correct[first:last].sum()),
                                     student_correct=int((prediction[first:last] == truth[first:last]).sum()),
                                     retained_teacher_errors=int(wrong[first:last].sum())))
            records.append(dict(model=name, schedule_step=step,
                                ce=float(ce.detach()), feature=float(feature.detach()), near_loss=float(near.detach()),
                                actual_gap=float(actual_gap.detach()), gap_weight=weight,
                                gradient_norms={key: norm(value) for key, value in gradients.items()},
                                ce_feature_cosine=cosine(gradients['ce'], gradients['feature']),
                                ce_gap_cosine=cosine(gradients['ce'], gradients['gap']),
                                ce_total_cosine=cosine(gradients['ce'], total),
                                feature_partition_ce_cosine={key: cosine(gradients['ce'], value) for key, value in feature_parts.items()},
                                feature_partition_gradient_norms={key: norm(value) for key, value in feature_parts.items()},
                                teacher_correct_student_wrong=int(wrong.sum()),
                                teacher_correct_margin_mean=float(margins[correct].mean()),
                                regions=row_data))
            print(json.dumps(records[-1]), flush=True)
    summaries = {}
    for name in models:
        rows = [row for row in records if row['model'] == name]
        summaries[name] = dict(exposures=512,
                               teacher_correct_student_wrong=sum(row['teacher_correct_student_wrong'] for row in rows),
                               mean_near_loss=float(np.mean([row['near_loss'] for row in rows])),
                               mean_ce_feature_cosine=float(np.mean([row['ce_feature_cosine'] for row in rows])),
                               mean_ce_gap_cosine=float(np.mean([row['ce_gap_cosine'] for row in rows])),
                               mean_ce_total_cosine=float(np.mean([row['ce_total_cosine'] for row in rows])),
                               mean_ce_feature_correct_cosine=float(np.mean([row['feature_partition_ce_cosine']['teacher_correct'] for row in rows])),
                               mean_ce_feature_wrong_cosine=float(np.mean([row['feature_partition_ce_cosine']['teacher_wrong'] for row in rows])),
                               total_gradient_opposes_ce_batches=sum(row['ce_total_cosine'] < 0 for row in rows),
                               original_retention_errors=sum(row['regions'][0]['retained_teacher_errors'] for row in rows),
                               augmented_retention_errors=sum(row['regions'][1]['retained_teacher_errors'] for row in rows))
    require(all(sha(path) == hashes[name] for name, path in paths.items()), 'Checkpoint changed')
    result = dict(schema='aiflow-train-gradient-balance/v34', status='pass',
                  code_sha256=sha(Path(__file__)), parent_plan_sha256=sha(run.OUT / 'frozen_plan.json'),
                  checkpoint_sha256=hashes, schedule_steps=[r['schedule_step'] for r in plan['calibration_rows']],
                  records=records, summaries=summaries,
                  gradient_composition_independently_checked=True, optimizer_steps=0,
                  coefficient_selection=False, evaluation_inputs_read=0, owned_labels_read=0,
                  crohme_rows=0, product_adopted=False,
                  limits='Already fitted eight TRAIN batches. Gradient cosine is a local direction diagnostic, not a causal accuracy or generalization measurement.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(event='gradient_balance_complete', summaries=summaries)), flush=True)


if __name__ == '__main__':
    main()
