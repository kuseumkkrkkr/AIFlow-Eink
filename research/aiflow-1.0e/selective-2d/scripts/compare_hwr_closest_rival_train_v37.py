"""Compare current versus closest-rival directions on fitted TRAIN only."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from audit_hwr_adam_margin_v36 import projection
from audit_hwr_gradient_balance_v34 import norm
from hwr_closest_rival_gap_v37 import closest_gap_loss, selftest
from run_hwr_teacher_correct_feature_v35 import load, require, write
from run_hwr_affine_distillation_experiment_v1 import _load_teacher


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Comparison exists')
    selftest()
    run, plan, arrays, schedule = load(args.parent)
    sha = run.direct.broad.previous._sha
    model, labels, _ = _load_teacher(args.parent / 'v33_control/main_trained.pt', torch.device('cpu'))
    teacher, tl, _ = _load_teacher(run.direct.broad.previous.CHECKPOINT, torch.device('cpu'))
    require(labels == tl, 'Vocabulary mismatch')
    teacher.requires_grad_(False)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    records, norms = [], []
    for saved in plan['calibration_rows']:
        x, y = run.direct.batch(arrays, schedule, saved['schedule_step'], 'augmented_main')
        tx, truth = torch.from_numpy(x), torch.from_numpy(y)
        z = model.math_head(model.encode(tx))
        with torch.no_grad():
            tz = teacher.math_head(teacher.encode(tx))
        params = list(model.parameters())
        old, correct, _ = run.near_gap_loss(z, tz, truth)
        new, new_mask, _ = closest_gap_loss(z, tz, truth)
        require(torch.equal(correct, new_mask), 'Teacher mask changed')
        old_g = torch.autograd.grad(plan['gap_weight'] * old, params, retain_graph=True)
        new_g = torch.autograd.grad(plan['gap_weight'] * new, params, retain_graph=True)
        require(all(torch.isfinite(v).all() for v in (*old_g, *new_g)), 'Nonfinite gradient')
        norms.append(dict(schedule_step=saved['schedule_step'], old_loss=float(old.detach()), new_loss=float(new.detach()),
                          old_weighted_gradient_l2=norm(old_g), new_weighted_gradient_l2=norm(new_g)))
        wrong = correct & (z.detach().argmax(1) != truth)
        for row in wrong.nonzero().flatten().tolist():
            rival = int(z.detach()[row].argmax())
            mg = torch.autograd.grad(z[row, truth[row]] - z[row, rival], params, retain_graph=True)
            records.append(dict(schedule_step=saved['schedule_step'], batch_row=row, truth=labels[int(truth[row])], rival=labels[rival],
                                old_gap_negative_sgd_margin_derivative=projection(mg, old_g),
                                new_gap_negative_sgd_margin_derivative=projection(mg, new_g)))
    summary = dict(residual_errors=len(records), old_harming_directions=sum(r['old_gap_negative_sgd_margin_derivative'] < 0 for r in records),
                   new_harming_directions=sum(r['new_gap_negative_sgd_margin_derivative'] < 0 for r in records))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write(args.output, dict(schema='aiflow-closest-rival-train-directions/v37', status='pass', code_sha256=sha(Path(__file__)),
                           loss_sha256=sha(Path(__file__).with_name('hwr_closest_rival_gap_v37.py')),
                           source_checkpoint_sha256=sha(args.parent / 'v33_control/main_trained.pt'),
                           records=records, norms=norms, summary=summary, coefficient_changed=False,
                           optimizer_steps=0, owned_labels_read=0, development_forward_rows=0, crohme_rows=0,
                           limits='Same fitted TRAIN batches; local negative-SGD direction, not actual AdamW motion or independent accuracy evidence.'))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
