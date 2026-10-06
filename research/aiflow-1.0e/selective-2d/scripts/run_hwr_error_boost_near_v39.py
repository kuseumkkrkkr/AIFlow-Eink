"""V39: one fresh error-boosted full-near arm with immutable V38 control reuse."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import shutil

import torch

from cloud_hwr_snapshot import configure
from hwr_error_boost_near_v39 import ERROR_BOOST, error_boost_gap, selftest as boost_selftest
from run_hwr_closest_rival_v38 import load as parent_load, PARAMETERS, ROOT
import run_hwr_teacher_correct_feature_v35 as original

require, write = original.require, original.write
PARENT = ROOT / 'artifacts/hwr_closest_rival_20261007_v38'
DIAGNOSTIC = ROOT / 'artifacts/hwr_error_boost_diagnostic_20261007_v39/train_comparison.json'
ARMS = ('v33_control', 'error_boost_near')
CONTROL_FILES = ('completed.json', 'main_trained.pt', 'training_microscope.jsonl', 'run_started.json', 'resume_state.pt')


def selftest():
    original.selftest()
    boost_selftest()


def load(out):
    run, parent, arrays, schedule = parent_load(PARENT)
    sha = run.direct.broad.previous._sha
    plan = json.loads((out / 'frozen_plan.json').read_text())
    require(plan['arms'] == list(ARMS) and plan['error_boost'] == ERROR_BOOST == 2., 'Arm/boost changed')
    for name, expected in plan['source_sha256'].items():
        require(sha(Path(__file__).with_name(name)) == expected, f'Source changed: {name}')
    require(sha(PARENT / 'frozen_plan.json') == plan['parent_plan_sha256'], 'Parent plan changed')
    require(sha(PARENT / 'independent_verification.json') == plan['control_certificate_sha256'], 'Control certificate changed')
    require(sha(DIAGNOSTIC) == plan['train_diagnostic_sha256'], 'TRAIN diagnostic changed')
    require(all(plan[n] == parent[n] for n in PARAMETERS), 'Hyperparameter changed')
    for name, expected in plan['reused_control_files'].items():
        require(sha(PARENT / 'v33_control' / name) == sha(out / 'v33_control' / name) == expected, f'Control copy changed: {name}')
    return run, plan, arrays, schedule


def prepare(out):
    require(not out.exists(), 'Refusing experiment overwrite')
    selftest()
    run, parent, _, _ = parent_load(PARENT)
    sha = run.direct.broad.previous._sha
    certificate = json.loads((PARENT / 'independent_verification.json').read_text())
    diagnostic = json.loads(DIAGNOSTIC.read_text())
    require(certificate['status'] == diagnostic['status'] == 'pass', 'Parent/diagnostic not verified')
    require(diagnostic['error_boost'] == ERROR_BOOST and not diagnostic['coefficient_changed'] and not diagnostic['boost_search'], 'Diagnostic hypothesis changed')
    require(diagnostic['source_checkpoint_sha256'] == sha(PARENT / 'v33_control/main_trained.pt'), 'TRAIN source changed')
    require(diagnostic['code_sha256'] == sha(Path(__file__).with_name('compare_hwr_error_boost_train_v39.py')) and
            diagnostic['loss_sha256'] == sha(Path(__file__).with_name('hwr_error_boost_near_v39.py')), 'Diagnostic code changed')
    out.mkdir()
    control = out / 'v33_control'
    control.mkdir()
    for name in CONTROL_FILES:
        shutil.copyfile(PARENT / 'v33_control' / name, control / name)
    plan = {n: parent[n] for n in PARAMETERS}
    plan.update(schema='aiflow-error-boost-near-plan/v39', arms=list(ARMS), error_boost=ERROR_BOOST,
        parent_plan_sha256=sha(PARENT / 'frozen_plan.json'), control_certificate_sha256=sha(PARENT / 'independent_verification.json'),
        train_diagnostic_sha256=sha(DIAGNOSTIC),
        reused_control_files={n: sha(control / n) for n in CONTROL_FILES},
        source_sha256={n: sha(Path(__file__).with_name(n)) for n in (Path(__file__).name, 'hwr_error_boost_near_v39.py', 'cloud_hwr_snapshot.py', 'run_hwr_teacher_correct_feature_v35.py')},
        canonical_sha256=parent['canonical_sha256'], torch=parent['torch'], numpy=parent['numpy'], python=parent['python'],
        cpu_model=parent['cpu_model'], cpu_capability=parent['cpu_capability'], torch_build_config=parent['torch_build_config'],
        threads=2, calibration_rows=parent['calibration_rows'], same_v25_data_and_schedule=True,
        objective='All-row CE and feature unchanged; full near-pair squared deficits, doubled on teacher-correct/student-wrong rows only.',
        pair_set_changed=False, pair_denominator_changed=False, feature_rows=64,
        weight_selection='Fixed error boost 2 chosen before TRAIN diagnostic; no grid or coefficient search. Argmax mask is detached.',
        cold_initialization='Fresh candidate cold canonical; reused V38 control originally cold canonical with identical schedule/seed/hyperparameters and same CPU.',
        reused_control=True, fresh_training_steps=2400, comparison_control_steps=2400,
        trainer_reuse='Unchanged V35 train/evaluate; process-local bindings plus separate focus telemetry.',
        checkpoint_container='Legacy V35 containers; V39 plan, arm and certificate identify this trial.',
        owned_optimizer_rows=0, crohme_rows=0, no_interim_evaluation=True, no_best_epoch_selection=True,
        coefficient_search=False, generated_labels_human_verified=False, all_evaluation_cases_already_consumed=True,
        product_adopted=False)
    write(out / 'frozen_plan.json', plan)


@contextmanager
def bindings(out, focus_stream=None):
    run, _, _, _ = load(out)
    old_load, old_arms, old_gap = original.load, original.ARMS, run.near_gap_loss
    calls = 0
    def observed_gap(z, tz, truth):
        nonlocal calls
        value, correct, active = error_boost_gap(z, tz, truth)
        baseline = old_gap(z, tz, truth)[0]
        focused = correct & (z.detach().argmax(1) != truth)
        calls += 1
        record = dict(step=calls, teacher_correct_rows=int(correct.sum()), focused_rows=int(focused.sum()),
            focused_original_rows=int(focused[:16].sum()), focused_augmented_rows=int(focused[16:].sum()),
            baseline_gap=float(baseline.detach()), actual_gap=float(value.detach()), error_boost=ERROR_BOOST,
            pair_count=int(run.near_mask(z, tz, truth)[0].sum()))
        focus_stream.write(json.dumps(record) + '\n')
        focus_stream.flush()
        return value, correct, active
    try:
        original.load, original.ARMS = load, ARMS
        if focus_stream is not None:
            run.near_gap_loss = observed_gap
        yield
    finally:
        original.load, original.ARMS, run.near_gap_loss = old_load, old_arms, old_gap


def train(out, arm):
    require(arm == 'error_boost_near', 'Control is reused; preserve it')
    load(out)
    require(not (out / arm).exists(), 'Candidate already exists')
    with (out / 'focus_microscope.jsonl').open('x') as stream:
        with bindings(out, stream):
            original.train(out, arm)


def evaluate(out):
    require(all((out / a / 'completed.json').exists() for a in ARMS), 'Both comparison models must be completed')
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
