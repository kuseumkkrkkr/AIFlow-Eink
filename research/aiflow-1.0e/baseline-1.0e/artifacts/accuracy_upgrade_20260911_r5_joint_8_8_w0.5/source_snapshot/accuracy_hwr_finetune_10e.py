"""기존 HWR의 head 또는 마지막 block을 외부 replay 1:1로 미세조정한다."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from accuracy_upgrade_data_10e import load_hwr
from accuracy_upgrade_contract_v1 import canonical_json_sha256, sha256_file
from character_tensor_v1 import tensorize
from train_character_classifier_v1 import apply_input_mode


def load_replay(cache: Path, labels: list[str]):
    """기존 승인된 math_train 캐시만 읽고 어휘와 배열 계약을 확인한다."""
    manifest = json.loads((cache / 'cache_manifest.json').read_text(encoding='utf-8'))
    if manifest['config']['math_labels'] != labels or manifest['config']['head_mode'] != 'unified-math':
        raise ValueError('external replay vocabulary/cache mismatch')
    item = manifest['sets']['math_train']
    features = np.load(cache / item['features'], mmap_mode='r', allow_pickle=False)
    targets = np.load(cache / item['labels'], mmap_mode='r', allow_pickle=False)
    if features.shape != (len(targets), 128, 5) or targets.min() < 0 or targets.max() >= len(labels):
        raise ValueError('invalid external training replay arrays')
    return features, targets, manifest


def fit_hwr(corpus, fit_ids: list[str], validation_ids: list[str], config: dict, device: torch.device,
            scope: str, replay_cache: Path, fixed_epochs: int | None = None, validation_score=None):
    """학습과 검증 writer·수식을 분리하여 필요한 가중치만 갱신한다."""
    if scope not in ('head', 'last-block'):
        raise ValueError('HWR scope must be head or last-block')
    if not fit_ids or len(fit_ids) != len(set(fit_ids)):
        raise ValueError('fit IDs must be nonempty and unique')
    train_writers = {corpus.records[k]['writer'] for k in fit_ids}
    held_writers = {corpus.records[k]['writer'] for k in validation_ids}
    train_formulas = {corpus.records[k]['formula_key'] for k in fit_ids}
    held_formulas = {corpus.records[k]['formula_key'] for k in validation_ids}
    if train_writers & held_writers or train_formulas & held_formulas:
        raise ValueError('HWR fit/selection writer or formula leakage')
    torch.manual_seed(config['seed'])
    model, labels = load_hwr(Path(config['hwr_checkpoint']), device)
    label_to_id = {token: index for index, token in enumerate(labels)}
    if any(corpus.raw[k]['label'] not in label_to_id for k in fit_ids + validation_ids):
        raise ValueError('HWR truth outside fixed vocabulary')
    replay, replay_labels, manifest = load_replay(replay_cache, labels)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith('math_head.') or scope == 'last-block' and name.startswith(('encoder.layers.3.', 'pool_score.')))
    encoder_parameters = [p for name, p in model.named_parameters() if p.requires_grad and not name.startswith('math_head.')]
    groups = [{'params': model.math_head.parameters(), 'lr': config['learning_rate']}]
    if encoder_parameters:
        groups.append({'params': encoder_parameters, 'lr': config['encoder_learning_rate']})
    optimizer = torch.optim.AdamW(groups, weight_decay=config['weight_decay'])
    all_ids = sorted(set(fit_ids + validation_ids))
    inputs = {k: apply_input_mode(tensorize(corpus.raw[k]), config['input_mode']) for k in all_ids}
    rng = np.random.default_rng(config['seed'])
    replay_ids, history, best, bad = set(), [], None, 0
    epochs = fixed_epochs or config['epochs']
    for epoch in range(1, epochs + 1):
        ids = list(fit_ids); rng.shuffle(ids)
        losses = []
        model.eval()
        if scope == 'last-block':
            model.encoder.layers[-1].train()
        for start in range(0, len(ids), 32):
            own = ids[start:start + 32]
            ext = rng.integers(0, len(replay_labels), size=len(own))
            replay_ids.update(map(int, ext))
            x = np.concatenate([np.stack([inputs[k] for k in own]), apply_input_mode(np.asarray(replay[ext]), config['input_mode'])])
            y = np.asarray([label_to_id[corpus.raw[k]['label']] for k in own] + replay_labels[ext].tolist(), dtype=np.int64)
            optimizer.zero_grad(set_to_none=True)
            tensor = torch.from_numpy(x).to(device)
            if scope == 'head':
                with torch.no_grad():
                    embedding = model.encode(tensor)
                scores = model.math_head(embedding)
            else:
                scores = model(tensor, 'math')
            loss = nn.functional.cross_entropy(scores, torch.from_numpy(y).to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError('non-finite HWR loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step(); losses.append(float(loss.detach()))
        model.eval()
        correct = 0
        with torch.inference_mode():
            for start in range(0, len(validation_ids), 32):
                ids = validation_ids[start:start + 32]
                prediction = model(torch.from_numpy(np.stack([inputs[k] for k in ids])).to(device), 'math').argmax(-1).cpu().tolist()
                correct += sum(labels[i] == corpus.raw[k]['label'] for k, i in zip(ids, prediction))
        score = tuple(validation_score(model)) if validation_score is not None else (correct,)
        if not all(np.isfinite(score)):
            raise ValueError('nonfinite HWR selection score')
        history.append(dict(epoch=epoch, loss=float(np.mean(losses)), validation_correct=correct,
                            validation_total=len(validation_ids), selection_score=list(score)))
        if not validation_ids or best is None or score > best[0]:
            best = score, epoch, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if validation_ids and fixed_epochs is None and bad >= config['patience']:
            break
    model.load_state_dict(best[2])
    return model.eval(), labels, dict(scope=scope, selected_epoch=best[1], history=history,
        fit_ids=fit_ids, selection_ids=validation_ids, external_replay_indices=sorted(replay_ids),
        replay_manifest_sha256=sha256_file(str(replay_cache / 'cache_manifest.json')),
        external_mixing_ratio='1:1', input_mode=config['input_mode'],
        selection_metric='e2e_callback' if validation_score is not None else 'glyph_top1' if validation_ids else 'fixed_final')
