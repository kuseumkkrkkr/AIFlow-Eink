"""Post-training TRAIN-only gradient audit; no evaluator, optimizer or tuning."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from audit_hwr_gradient_balance_v34 import cosine, norm
from run_hwr_closest_rival_v38 import ARMS, load, require, write
from run_hwr_teacher_correct_feature_v35 import feature_loss
from hwr_closest_rival_gap_v37 import closest_gap_loss
from run_hwr_affine_distillation_experiment_v1 import _load_teacher


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Completed paired experiment directory')
    args = parser.parse_args()
    out = args.output
    require(not (out / 'train_gradient_diagnostic.json').exists(), 'Diagnostic exists')
    run, plan, arrays, schedule = load(out)
    sha = run.direct.broad.previous._sha
    models, hashes = {}, {}
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    teacher, labels, _ = _load_teacher(run.direct.broad.previous.CHECKPOINT, torch.device('cpu'))
    teacher.requires_grad_(False)
    for arm in ARMS:
        folder = out / arm
        completed = json.loads((folder / 'completed.json').read_text())
        require(completed['steps'] == 2400, 'Both arms must be completed')
        hashes[arm] = sha(folder / 'main_trained.pt')
        require(completed['checkpoint_sha256'] == hashes[arm], 'Checkpoint changed')
        model, vocabulary, _ = _load_teacher(folder / 'main_trained.pt', torch.device('cpu'))
        require(labels == vocabulary, 'Vocabulary changed')
        models[arm] = model
    records = []
    for calibration in plan['calibration_rows']:
        step = calibration['schedule_step']
        x, y = run.direct.batch(arrays, schedule, step, 'augmented_main')
        tx, truth = torch.from_numpy(x), torch.from_numpy(y)
        with torch.no_grad():
            th = teacher.encode(tx)
            tz = teacher.math_head(th)
        correct = tz.argmax(1) == truth
        for arm, model in models.items():
            h = model.encode(tx)
            z = model.math_head(h)
            mask = torch.ones_like(correct)
            ce = F.cross_entropy(z, truth)
            feature = feature_loss(h, th, mask)
            gap_function = closest_gap_loss if arm == 'closest_rival' else run.near_gap_loss
            gap, gap_mask, _ = gap_function(z, tz, truth)
            common_near = run.near_gap_loss(z, tz, truth)[0]
            require(torch.equal(gap_mask, correct), 'Gap mask changed')
            terms = dict(ce=ce, feature=plan['feature_weight'] * feature, gap=plan['gap_weight'] * gap)
            parameters = list(model.parameters())
            gradients = {}
            for name, value in terms.items():
                raw = torch.autograd.grad(value, parameters, retain_graph=True, allow_unused=name == 'feature')
                for (key, _), grad in zip(model.named_parameters(), raw, strict=True):
                    require(grad is not None or name == 'feature' and key.startswith('math_head.'), 'Unexpected disconnected gradient')
                gradients[name] = tuple(torch.zeros_like(p) if g is None else g for p, g in zip(parameters, raw, strict=True))
            total = tuple(sum(parts) for parts in zip(*gradients.values(), strict=True))
            direct = torch.autograd.grad(sum(terms.values()), parameters)
            require(all(torch.isfinite(v).all() for group in gradients.values() for v in group), 'Nonfinite gradient')
            require(all(torch.allclose(a, b, atol=2e-6, rtol=1e-5) for a, b in zip(total, direct, strict=True)), 'Gradient sum mismatch')
            predicted = z.detach().argmax(1)
            retained_errors = correct & (predicted != truth)
            records.append(dict(arm=arm, schedule_step=step, ce=float(ce.detach()),
                                feature=float(feature.detach()), arm_gap=float(gap.detach()), common_near_gap=float(common_near.detach()),
                                feature_rows=int(mask.sum()), teacher_correct_rows=int(correct.sum()),
                                gradient_norms={key: norm(value) for key, value in gradients.items()},
                                ce_feature_cosine=cosine(gradients['ce'], gradients['feature']),
                                ce_gap_cosine=cosine(gradients['ce'], gradients['gap']),
                                ce_total_cosine=cosine(gradients['ce'], total),
                                original_retention_errors=int(retained_errors[:16].sum()),
                                augmented_retention_errors=int(retained_errors[16:].sum())))
    summaries = {}
    for arm in ARMS:
        rows = [r for r in records if r['arm'] == arm]
        summaries[arm] = dict(exposures=512,
            teacher_correct_student_wrong=sum(r['original_retention_errors'] + r['augmented_retention_errors'] for r in rows),
            original_retention_errors=sum(r['original_retention_errors'] for r in rows),
            augmented_retention_errors=sum(r['augmented_retention_errors'] for r in rows),
            mean_arm_gap=float(np.mean([r['arm_gap'] for r in rows])),
            mean_common_near_gap=float(np.mean([r['common_near_gap'] for r in rows])),
            mean_ce_feature_cosine=float(np.mean([r['ce_feature_cosine'] for r in rows])),
            mean_ce_gap_cosine=float(np.mean([r['ce_gap_cosine'] for r in rows])),
            mean_ce_total_cosine=float(np.mean([r['ce_total_cosine'] for r in rows])))
    require(all(sha(out / arm / 'main_trained.pt') == hashes[arm] for arm in ARMS), 'Checkpoint mutated')
    require(sha(run.direct.broad.previous.CHECKPOINT) == plan['canonical_sha256'], 'Canonical mutated')
    result = dict(status='pass', schema='aiflow-closest-rival-train-audit/v38',
        code_sha256=sha(Path(__file__)), plan_sha256=sha(out / 'frozen_plan.json'),
        checkpoint_sha256=hashes, records=records, summaries=summaries,
        optimizer_steps=0, evaluation_forward_rows=0, owned_labels_read=0, crohme_rows=0,
        coefficient_selection=False, product_adopted=False,
        gradient_composition_independently_checked=True,
        limits='Post-training diagnostic on the same eight fitted TRAIN batches; not fresh acceptance or causal generalization evidence.')
    write(out / 'train_gradient_diagnostic.json', result)
    print(json.dumps(summaries), flush=True)


if __name__ == '__main__':
    main()
