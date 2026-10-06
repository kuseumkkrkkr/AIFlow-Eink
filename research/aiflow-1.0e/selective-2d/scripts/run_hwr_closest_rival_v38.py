"""Fresh paired V38 rerun of V37 after environment recovery drift.

Historical source stays unchanged. Runtime bindings are restored after each call.
Feature retention applies to ALL rows in both arms; no failed V35 feature mask.
"""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import platform

import numpy as np
import torch

from cloud_hwr_snapshot import configure
from hwr_closest_rival_gap_v37 import closest_gap_loss, selftest as closest_selftest
import run_hwr_teacher_correct_feature_v35 as original

require, write = original.require, original.write
ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'artifacts/hwr_teacher_correct_feature_20261006_v35_cloud'
DIAGNOSTIC = ROOT / 'artifacts/hwr_adam_margin_20261006_v36/closest_rival_train_comparison.json'
ARMS = ('v33_control', 'closest_rival')
PARAMETERS = ('steps', 'batch_size', 'original_rows', 'augmented_rows', 'seed', 'lr',
              'weight_decay', 'gradient_clip', 'hard_ce_weight', 'feature_weight', 'gap_weight')


def cpu_model():
    return next(line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name'))


def selftest():
    original.selftest()
    closest_selftest()
    # Independent NumPy calculation including teacher-wrong rows and ties.
    rng = np.random.default_rng(37)
    t = rng.normal(size=(8, 372)).astype(np.float32)
    z = rng.normal(size=(8, 372)).astype(np.float32)
    y = t.argmax(1)
    y[::3] = (y[::3] + 1) % 372
    eligible = t.argmax(1) == y
    competitors = z.copy()
    competitors[np.arange(8), y] = -np.inf
    rival = competitors.argmax(1)
    deficit = np.maximum((t[np.arange(8), y] - t[np.arange(8), rival]) -
                         (z[np.arange(8), y] - z[np.arange(8), rival]), 0) * eligible
    expected = float(np.square(deficit).sum() / max(int(eligible.sum()), 1))
    sz = torch.tensor(z, requires_grad=True)
    value, mask, active = closest_gap_loss(sz, torch.tensor(t), torch.tensor(y))
    require(np.isclose(float(value.detach()), expected, atol=1e-6), 'NumPy closest loss mismatch')
    require(np.array_equal(mask.numpy(), eligible) and active == int((deficit > 0).sum()), 'NumPy mask mismatch')
    gradient = torch.autograd.grad(value, sz)[0].numpy()
    expected_gradient = np.zeros_like(z)
    scale = 2 * deficit / max(int(eligible.sum()), 1)
    expected_gradient[np.arange(8), y] = -scale
    expected_gradient[np.arange(8), rival] = scale
    require(np.allclose(gradient, expected_gradient, atol=1e-6), 'NumPy closest gradient mismatch')


def load(out):
    run = configure()
    _, arrays, schedule = run.load()
    sha = run.direct.broad.previous._sha
    plan = json.loads((out / 'frozen_plan.json').read_text())
    parent = json.loads((PARENT / 'frozen_plan.json').read_text())
    for name, expected in plan['source_sha256'].items():
        require(sha(Path(__file__).with_name(name)) == expected, f'Source changed: {name}')
    require(sha(PARENT / 'frozen_plan.json') == plan['parent_plan_sha256'], 'Parent plan changed')
    require(sha(DIAGNOSTIC) == plan['train_diagnostic_sha256'], 'Diagnostic changed')
    require(all(plan[n] == parent[n] for n in PARAMETERS), 'Training parameter changed')
    require(plan['torch'] == torch.__version__ and plan['arms'] == list(ARMS), 'Runtime/arm changed')
    require(plan['python'] == platform.python_version() and plan['numpy'] == np.__version__, 'Python/NumPy changed')
    require(plan['cpu_model'] == cpu_model() and plan['cpu_capability'] == torch.backends.cpu.get_cpu_capability(), 'CPU changed during paired experiment')
    require(plan['torch_build_config'] == torch.__config__.show(), 'Torch build changed')
    return run, plan, arrays, schedule


def prepare(out):
    require(not out.exists(), 'Refusing experiment overwrite')
    selftest()
    run = configure()
    run.load()
    sha = run.direct.broad.previous._sha
    parent = json.loads((PARENT / 'frozen_plan.json').read_text())
    certificate = json.loads((PARENT / 'independent_verification.json').read_text())
    require(certificate['status'] == 'pass', 'Parent not verified')
    diagnostic = json.loads(DIAGNOSTIC.read_text())
    require(diagnostic['status'] == 'pass' and diagnostic['coefficient_changed'] is False, 'TRAIN diagnostic invalid')
    plan = {n: parent[n] for n in PARAMETERS}
    plan.update(restart_context='V37 exact recovery rejected: first overlapping step differs after environment restart. Both arms cold-restarted together; original interrupted run preserved.',
        cpu_model=cpu_model(),
        cpu_capability=torch.backends.cpu.get_cpu_capability(),
        torch_build_config=torch.__config__.show(),
        interrupted_v37_plan_sha256=sha(ROOT / 'artifacts/hwr_closest_rival_20261006_v37/frozen_plan.json'),
        schema='aiflow-closest-rival-plan/v38', arms=list(ARMS),
        parent_plan_sha256=sha(PARENT / 'frozen_plan.json'), train_diagnostic_sha256=sha(DIAGNOSTIC),
        source_sha256={n: sha(Path(__file__).with_name(n)) for n in (
            Path(__file__).name, 'hwr_closest_rival_gap_v37.py', 'cloud_hwr_snapshot.py',
            'run_hwr_teacher_correct_feature_v35.py')},
        canonical_sha256=parent['canonical_sha256'], torch=torch.__version__, numpy=np.__version__,
        python=platform.python_version(), threads=2, calibration_rows=parent['calibration_rows'],
        cold_initialization='canonical in both arms', same_v25_data_and_schedule=True,
        objective='Hard CE over all 64 + unchanged all-row feature MSE + unchanged gap coefficient times arm-specific retention.',
        control_gap='Teacher-correct stable teacher/student Top-5 union, truth excluded; squared decreased gap mean over selected pairs.',
        challenger_gap='Teacher-correct student closest non-truth rival, tie lowest index; squared decreased teacher gap mean over teacher-correct rows.',
        gap_normalization_changes=True, coefficient_search=False,
        trainer_reuse='Unchanged V35 train/evaluate functions via temporary process-local load, arms and near-gap bindings.',
        checkpoint_container='Legacy V35 container; current experiment/objective identified by this V38 plan and report.arm.',
        generated_labels_human_verified=False, owned_optimizer_rows=0, crohme_rows=0,
        no_interim_evaluation=True, no_best_epoch_selection=True,
        all_evaluation_cases_already_consumed=True, product_adopted=False)
    out.mkdir(parents=True)
    write(out / 'frozen_plan.json', plan)


@contextmanager
def bindings(out, arm=None):
    run, _, _, _ = load(out)
    previous_load, previous_arms, previous_gap = original.load, original.ARMS, run.near_gap_loss
    try:
        original.load = load
        original.ARMS = ARMS
        if arm == 'closest_rival':
            run.near_gap_loss = closest_gap_loss
        yield
    finally:
        original.load, original.ARMS, run.near_gap_loss = previous_load, previous_arms, previous_gap


def train(out, arm):
    require(arm in ARMS, 'Unknown arm')
    with bindings(out, arm):
        original.train(out, arm)


def evaluate(out):
    require(all((out / a / 'completed.json').exists() for a in ARMS), 'Both arms must complete before evaluation')
    with bindings(out):
        original.evaluate(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'train', 'evaluate', 'selftest'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--arm', choices=ARMS)
    args = parser.parse_args()
    if args.mode == 'prepare': prepare(args.output)
    elif args.mode == 'train': train(args.output, args.arm)
    elif args.mode == 'evaluate': evaluate(args.output)
    else: selftest()
