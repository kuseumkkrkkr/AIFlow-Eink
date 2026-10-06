"""Independently check both V37 logs, checkpoint updates and replayed predictions."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from run_hwr_closest_rival_v37 import ARMS, PARENT, load, require, selftest, write
from run_hwr_affine_distillation_experiment_v1 import _load_teacher, _predict_logits
from verify_hwr_broad_head_only_v23 import hit_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    require(not (out / 'independent_verification.json').exists(), 'Certificate exists')
    run, plan, _, _ = load(out)
    selftest()
    sha = run.direct.broad.previous._sha
    result = json.loads((out / 'comparison_result.json').read_text())
    _, data, folds, _ = run.direct.broad.previous.global_load(run.direct.broad.previous.OUTPUT)
    owned = run.direct.broad.previous.ROOT / 'artifacts/hwr_owned_formula_transfer_20261005_v21'
    inputs = np.load(owned / 'inputs.npy', allow_pickle=False)
    input_rows = json.loads((owned / 'input_rows.json').read_text())
    canonical, vocabulary, _ = _load_teacher(run.direct.broad.previous.CHECKPOINT, torch.device('cpu'))
    evidence = {}
    all_records = {}
    for arm in ARMS:
        folder = out / arm
        started = json.loads((folder / 'run_started.json').read_text())
        require(started['arm'] == arm, 'Run arm mismatch')
        done = json.loads((folder / 'completed.json').read_text())
        require(started['plan_sha256'] == sha(out / 'frozen_plan.json'), 'Run plan mismatch')
        require(done['steps'] == 2400 and done['evaluation_forwarded_during_training'] == 0 and done['canonical_unchanged'], 'Run not completed correctly')
        require(done['microscope_sha256'] == sha(folder / 'training_microscope.jsonl'), 'Log changed')
        require(done['checkpoint_sha256'] == sha(folder / 'main_trained.pt'), 'Checkpoint changed')
        count, max_delta, masked_rows = 0, 0., 0
        with (folder / 'training_microscope.jsonl').open() as stream:
            for line in stream:
                row = json.loads(line)
                count += 1
                require(row['step'] == count and row['teacher_gradients_absent'], 'Step/teacher gradient mismatch')
                require(row['hard_ce_weight'] == 1. and row['hard_target_rows'] == 64 and row['augmented_rows'] == 48, 'Data/CE mismatch')
                require(row['feature_rows'] == 64, 'Feature mask mismatch')
                require(0 <= row['teacher_correct'] <= 64, 'Teacher count outside range')
                masked_rows += 64 - row['feature_rows']
                require(len(row['parameter_gradient_l2']) == 57 and len(row['encoder_layers']) == 4, 'Gradient scope mismatch')
                require(all(np.isfinite(v) for v in row['parameter_gradient_l2'].values()), 'Nonfinite gradient')
                require(all(np.isfinite(v) for layer in row['encoder_layers'].values() for v in layer.values()), 'Nonfinite encoder')
                for i in range(4):
                    values = [v for n, v in row['parameter_gradient_l2'].items() if n.startswith(f'encoder.layers.{i}.')]
                    require(len(values) == 12 and max(values) > 0, 'Inactive encoder layer')
                delta = abs(row['loss'] - row['ce'] - plan['feature_weight'] * row['feature'] - plan['gap_weight'] * row['gap'])
                require(delta < 1e-6, 'Recorded loss arithmetic mismatch')
                max_delta = max(max_delta, delta)
        require(count == 2400, 'Incomplete training log')
        model, labels, _ = _load_teacher(folder / 'main_trained.pt', torch.device('cpu'))
        require(labels == vocabulary and all(torch.isfinite(p).all() for p in model.parameters()), 'Checkpoint invalid')
        changed = sum(not torch.equal(p, canonical.state_dict()[n]) for n, p in model.named_parameters())
        require(changed == 57, 'Incomplete model updates')
        if arm == 'v33_control':
            parent_model, parent_labels, _ = _load_teacher(PARENT / arm / 'main_trained.pt', torch.device('cpu'))
            require(parent_labels == labels and all(torch.equal(p, parent_model.state_dict()[n]) for n, p in model.named_parameters()), 'Control weights differ from completed V35 baseline')
            with (folder / 'training_microscope.jsonl').open() as fresh, (PARENT / arm / 'training_microscope.jsonl').open() as previous:
                count_pairs = 0
                for now, then in zip(fresh, previous, strict=True):
                    require(now == then, 'Control serialized record differs from V35 baseline')
                    a, b = json.loads(now), json.loads(then)
                    require(a == b, 'Control full training record differs from V35 baseline')
                    count_pairs += 1
                require(count_pairs == 2400, 'Control repeat scope mismatch')
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(2)
        scored = result['metrics'][arm]
        for fold in (1, 2):
            z = _predict_logits(model, np.array(data['features'][folds[fold]], copy=True), torch.device('cpu'), 32)
            require(np.array_equal(z, np.load(folder / f'fold{fold}_logits.npy', allow_pickle=False)), 'Development replay not bit exact')
            require(hit_metrics(z, np.array(data['labels'][folds[fold]], copy=True), labels) == scored['development'][str(fold)], 'Development metrics mismatch')
        torch.set_num_threads(1)
        z = _predict_logits(model, inputs, torch.device('cpu'), 128)
        require(np.array_equal(z, np.load(folder / 'owned_logits.npy', allow_pickle=False)), 'Owned replay not bit exact')
        records = json.loads((folder / 'owned_records.json').read_text())
        require(len(records) == len(input_rows) == 149, 'Owned input coverage changed')
        top = np.argsort(-z, axis=1, kind='stable')[:, :5]
        for row, original in zip(records, input_rows, strict=True):
            require({key: row[key] for key in original} == original, 'Owned input/label provenance changed')
            tokens = [[labels[int(i)] for i in order] for order in top[row['start']:row['stop']]]
            require(tokens == row['top5_tokens'] and [v[0] for v in tokens] == row['top1_tokens'], 'Owned records mismatch')
        for cohort in ('all', 'legacy_96', 'codex_reviewed_53'):
            rows = [r for r in records if cohort == 'all' or r['cohort'] == cohort]
            truth = [t for r in rows for t in r['truth_tokens']]
            predicted = [t for r in rows for t in r['top1_tokens']]
            actual = dict(formulas=len(rows), tokens=len(truth), top1_token_hits=int(np.equal(truth, predicted).sum()),
                          formula_top1_exact=sum(r['truth_tokens'] == r['top1_tokens'] for r in rows),
                          formula_top5_complete=sum(all(t in k for t, k in zip(r['truth_tokens'], r['top5_tokens'], strict=True)) for r in rows))
            require(actual == scored['owned'][cohort], 'Owned metrics mismatch')
        all_records[arm] = records
        evidence[arm] = dict(training_steps=count, masked_feature_exposures=masked_rows,
                             max_recorded_arithmetic_delta=max_delta, changed_parameter_tensors=changed,
                             checkpoint_sha256=done['checkpoint_sha256'], final_logits_reload_bit_exact=True)
    control, candidate = (all_records[arm] for arm in ARMS)
    require([r['sample_id'] for r in control] == [r['sample_id'] for r in candidate], 'Paired sample order changed')
    before = np.array([r['truth_tokens'] == r['top1_tokens'] for r in control])
    after = np.array([r['truth_tokens'] == r['top1_tokens'] for r in candidate])
    # Replay the retained canonical on the same CPU, after both training runs.
    torch.set_num_threads(2)
    canonical_development = {}
    for fold in (1, 2):
        z = _predict_logits(canonical, np.array(data['features'][folds[fold]], copy=True), torch.device('cpu'), 32)
        canonical_development[str(fold)] = hit_metrics(z, np.array(data['labels'][folds[fold]], copy=True), vocabulary)
    torch.set_num_threads(1)
    z = _predict_logits(canonical, inputs, torch.device('cpu'), 128)
    top = np.argsort(-z, axis=1, kind='stable')[:, :5]
    canonical_records = []
    for row in input_rows:
        tokens = [[vocabulary[int(i)] for i in order] for order in top[row['start']:row['stop']]]
        canonical_records.append(dict(row, top1_tokens=[t[0] for t in tokens], top5_tokens=tokens))
    canonical_owned = dict(formulas=149, tokens=579,
        top1_token_hits=sum(sum(t == p for t, p in zip(r['truth_tokens'], r['top1_tokens'], strict=True)) for r in canonical_records),
        formula_top1_exact=sum(r['truth_tokens'] == r['top1_tokens'] for r in canonical_records),
        formula_top5_complete=sum(all(t in k for t, k in zip(r['truth_tokens'], r['top5_tokens'], strict=True)) for r in canonical_records))
    require(sha(run.direct.broad.previous.CHECKPOINT) == plan['canonical_sha256'], 'Canonical changed')
    write(out / 'independent_verification.json', dict(schema='aiflow-closest-rival-verification/v37', status='pass',
                                                     verifier_sha256=sha(Path(__file__)), comparison_sha256=sha(out / 'comparison_result.json'),
                                                     arms=evidence, paired_formula_wins=int((~before & after).sum()),
                                                     paired_formula_losses=int((before & ~after).sum()),
                                                     canonical_same_platform=dict(development=canonical_development, owned=canonical_owned),
                                                     canonical_checkpoint_sha256=plan['canonical_sha256'],
                                                     fresh_control_matches_v35_weights_and_all_2400_records_bit_exact=True,
                                                     historical_intermediate_forward_replayed=False,
                                                     owned_optimizer_rows=0, crohme_rows=0, canonical_unchanged=True,
                                                     product_adopted=False))
    print(json.dumps(evidence), flush=True)


if __name__ == '__main__':
    main()
