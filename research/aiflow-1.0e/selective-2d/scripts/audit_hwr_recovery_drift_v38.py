"""Record first-step V37 recovery drift without optimization or evaluation."""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from run_hwr_closest_rival_v37 import ARMS, load, require, write
from run_hwr_closest_rival_v38 import cpu_model
from run_hwr_teacher_correct_feature_v35 import feature_loss
from hwr_closest_rival_gap_v37 import closest_gap_loss
from run_hwr_affine_distillation_experiment_v1 import _load_teacher


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Refusing diagnostic overwrite')
    run, plan, arrays, schedule = load(args.parent)
    sha = run.direct.broad.previous._sha
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    records = {}
    for arm in ARMS:
        folder = args.parent / arm
        recovery = folder / 'recovery_20261007'
        frozen = json.loads((recovery / 'recovery_plan.json').read_text())
        failure = json.loads((recovery / 'failure.json').read_text())
        for name, expected in frozen['original_files'].items():
            require(sha(recovery / name) == sha(folder / name) == expected, 'Interrupted input mutated')
        state = torch.load(recovery / 'resume_state.pt', map_location='cpu', weights_only=True)
        torch.manual_seed(plan['seed'])
        model, _, _ = _load_teacher(run.direct.broad.previous.CHECKPOINT, torch.device('cpu'))
        teacher, _, _ = _load_teacher(run.direct.broad.previous.CHECKPOINT, torch.device('cpu'))
        teacher.requires_grad_(False)
        model.load_state_dict(state['state_dict'], strict=True)
        torch.set_rng_state(state['torch_rng'])
        layers, handles = {}, []
        def capture(name):
            def hook(module, inputs, value):
                layers[name] = dict(mean=float(value.detach().mean()), std=float(value.detach().std()))
            return hook
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.TransformerEncoderLayer):
                handles.append(module.register_forward_hook(capture(name)))
        x, y = run.direct.batch(arrays, schedule, state['step'], 'augmented_main')
        tx, truth = torch.from_numpy(x), torch.from_numpy(y)
        h = model.encode(tx)
        z = model.math_head(h)
        with torch.no_grad():
            th = teacher.encode(tx)
            tz = teacher.math_head(th)
        correct = tz.argmax(1) == truth
        mask = torch.ones_like(correct)
        ce = F.cross_entropy(z, truth)
        feature = feature_loss(h, th, mask)
        gap = (closest_gap_loss if arm == 'closest_rival' else run.near_gap_loss)(z, tz, truth)[0]
        loss = ce + plan['feature_weight'] * feature + plan['gap_weight'] * gap
        loss.backward()
        gradients = {n: float(p.grad.norm()) for n, p in model.named_parameters()}
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        actual = dict(step=state['step'] + 1, ce=float(ce.detach()), feature=float(feature.detach()), gap=float(gap.detach()),
            loss=float(loss.detach()), feature_rows=64, teacher_correct=int(correct.sum()), hard_ce_weight=1.,
            hard_target_rows=64, augmented_rows=48, gradient_l2=float(norm), parameter_gradient_l2=gradients,
            encoder_layers=dict(layers), teacher_gradients_absent=True)
        expected = json.loads((recovery / 'training_microscope.jsonl').read_text().splitlines()[state['step']])
        differences = []
        def compare(a, b, name=''):
            if isinstance(a, dict):
                for key in a:
                    compare(a[key], b[key], f'{name}.{key}')
            elif a != b:
                differences.append(dict(field=name, actual=a, expected=b, delta=a - b))
        compare(actual, expected)
        require(differences and failure['logical_step'] == state['step'] + 1 and failure['appended_steps'] == 0, 'Recovery drift scope changed')
        records[arm] = dict(expected=expected, actual=actual, differences=differences,
            different_fields=len(differences), maximum_absolute_delta=max(abs(r['delta']) for r in differences),
            recovery_plan_sha256=sha(recovery / 'recovery_plan.json'), failure_sha256=sha(recovery / 'failure.json'))
        for handle in handles:
            handle.remove()
    require(sha(run.direct.broad.previous.CHECKPOINT) == plan['canonical_sha256'], 'Canonical changed')
    write(args.output, dict(schema='aiflow-recovery-drift-audit/v38', status='drift_detected',
        code_sha256=sha(Path(__file__)), parent_plan_sha256=sha(args.parent / 'frozen_plan.json'),
        cpu_model=cpu_model(), cpu_capability=torch.backends.cpu.get_cpu_capability(), torch=torch.__version__,
        records=records, optimizer_steps=0, development_forward_rows=0, owned_labels_read=0, crohme_rows=0,
        original_inputs_preserved=True, exact_recovery_rejected=True,
        limits='First overlapping fitted TRAIN step only. Numerical drift is measured; its underlying runtime cause is not isolated.'))
    print(json.dumps({arm: {k: row[k] for k in ('different_fields', 'maximum_absolute_delta')} for arm, row in records.items()}), flush=True)


if __name__ == '__main__':
    main()
