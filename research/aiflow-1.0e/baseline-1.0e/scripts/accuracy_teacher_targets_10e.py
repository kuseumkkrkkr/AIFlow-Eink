"""fold 내부에서만 점수 head를 학습하고 teacher 교차 예측을 생성한다."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from accuracy_upgrade_contract_v1 import canonical_json_sha256, sha256_file
from accuracy_lineage_10e import filtered_ids
from accuracy_upgrade_data_10e import formula_batches, pack_formulas
from formula_candidate_loss_10e import formula_candidate_loss


def formula_window_weights(window: list[list[list[dict]]]) -> list[float]:
    """누적 window의 각 batch를 실제 수식 수에 비례해 가중한다."""
    total = sum(len(formulas) for formulas in window)
    if total <= 0:
        raise ValueError('empty teacher accumulation window')
    return [len(formulas) / total for formulas in window]


class TeacherHead(nn.Module):
    """고정 OCR 특징·online 후보를 조건으로 작은 점수화 head만 학습한다."""
    def __init__(self, feature_size: int, numeric_size: int, tokens: int):
        """기존 frozen 특징의 차원을 128로 투영한다."""
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(feature_size, 128), nn.LayerNorm(128), nn.GELU())
        self.numeric = nn.Sequential(nn.Linear(numeric_size, 128), nn.LayerNorm(128), nn.GELU())
        self.tokens = nn.Embedding(tokens, 48)
        self.score = nn.Sequential(nn.Linear(304, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, features, numeric, tokens, mask):
        """이미지별 OCR 특징과 후보 token으로 점수를 계산한다."""
        evidence = self.projection(features).unsqueeze(-2).expand(*numeric.shape[:-1], 128)
        logits = self.score(torch.cat([evidence, self.numeric(numeric), self.tokens(tokens)], dim=-1)).squeeze(-1)
        return logits.masked_fill(~mask, torch.finfo(logits.dtype).min)


class TeacherTargets:
    """하나의 실행에서 정확히 같은 fit 집합의 teacher head만 재사용한다."""
    def __init__(self, experiment, names: list[str]):
        """원본 잉크·canvas와 결합된 frozen 특징 캐시만 허용한다."""
        self.experiment, self.names = experiment, names
        self.features, self.roots, self.cache = {}, {}, {}
        rights_path = Path(__file__).resolve().parents[1] / 'configs/accuracy_teacher_rights_20260911.json'
        rights = json.loads(rights_path.read_text(encoding='utf-8'))
        for name in names:
            if not rights['teachers'].get(name, {}).get('admitted', False):
                raise ValueError(f'blocked_data: teacher rights not verified: {name}')
        for name in names:
            directory = Path(experiment.config['teachers'][name])
            manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
            feature_path = directory / 'features.npz'
            if manifest.get('feature_file_sha256') != sha256_file(str(feature_path)):
                raise ValueError(f'{name}: frozen features lack verified file SHA; regenerate features')
            for key, formula in experiment.corpus.formulas.items():
                expected = canonical_json_sha256(dict(strokes=formula['strokes'], canvas=formula['canvas']))
                if manifest.get('source_formula_sha256', {}).get(key) != expected:
                    raise ValueError(f'{name}: raw formula hash mismatch; regenerate features')
            arrays = np.load(feature_path, allow_pickle=False)
            ids = arrays['formula_ids'].tolist()
            if len(set(ids)) != len(ids) or arrays['features'].shape[0] != len(ids) or not np.isfinite(arrays['features']).all():
                raise ValueError('invalid frozen feature matrix')
            self.features[name] = dict(zip(ids, arrays['features']))
            self.roots[name] = experiment.registry.add(name + '_frozen_features', feature_path,
                feature_manifest_sha256=sha256_file(str(directory / 'manifest.json')),
                rights_manifest_sha256=sha256_file(str(rights_path)),
                frozen_encoder_sha256=manifest['checkpoint_sha256'])

    def fit(self, name: str, fit_ids: list[str], samples: dict):
        """teacher 점수화 모델과 통계는 실제 fit 수식만 사용한다."""
        exp = self.experiment
        signature = canonical_json_sha256(dict(teacher=name, fit_ids=sorted(fit_ids),
            hwr=exp.hwr_node, numeric_size=next(iter(samples.values()))['numeric'].shape[-1],
            candidates={key: samples[key]['candidates'] for key in fit_ids}))
        if signature in self.cache:
            return self.cache[signature]
        if not fit_ids:
            raise ValueError('no teacher fit data after writer/formula exclusion')
        torch.manual_seed(exp.config['seed'])
        feature_map = self.features[name]
        formula_ids = sorted({samples[key]['formula_id'] for key in fit_ids})
        values = np.stack([feature_map[key] for key in formula_ids]).astype(np.float32)
        mean, std = values.mean(0), np.maximum(values.std(0), 1e-5)
        model = TeacherHead(values.shape[-1], next(iter(samples.values()))['numeric'].shape[-1], len(exp.labels)).to(exp.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=exp.config['learning_rate'], weight_decay=exp.config['weight_decay'])
        for epoch in range(exp.config['epochs']):
            model.train()
            batches = list(formula_batches(samples, fit_ids, exp.config['batch_size'], exp.config['seed'] + epoch))
            for start in range(0, len(batches), exp.config['accumulation']):
                window = batches[start:start + exp.config['accumulation']]
                weights = formula_window_weights(window)
                optimizer.zero_grad(set_to_none=True)
                for formulas, weight in zip(window, weights, strict=True):
                    batch = pack_formulas(formulas, exp.device)
                    feature = torch.from_numpy(np.stack([(feature_map[f[0]['formula_id']] - mean) / std for f in formulas])).to(exp.device)
                    feature = feature.unsqueeze(1).expand(-1, batch['ink'].shape[1], -1)
                    scores = model(feature, batch['numeric'], batch['token_ids'], batch['candidate_mask'])
                    loss, _ = formula_candidate_loss(scores, batch['targets'], batch['candidate_mask'], batch['row_mask'])
                    (loss * weight).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
        path = exp.output / 'models' / f'teacher_{name}_{signature[:16]}.pt'
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(state_dict=model.cpu().state_dict(), mean=mean, std=std, labels=exp.labels,
                        fit_ids=fit_ids, candidate_key=signature), path)
        root = exp.registry.add(name + '_scoring_head', path, fit_ids, [self.roots[name], exp.hwr_node])
        self.cache[signature] = (path, root, mean, std)
        return self.cache[signature]

    @torch.no_grad()
    def predict(self, name: str, fit_ids: list[str], held_ids: list[str], samples: dict):
        """모든 held writer와 수식이 teacher 감독학습 조상에서 빠졌는지 확인한다."""
        exp = self.experiment
        # fit 내부의 teacher 학습만 잠시 gradient를 활성화한다.
        with torch.enable_grad():
            path, root, mean, std = self.fit(name, fit_ids, samples)
        exp.assert_held(root, held_ids)
        model = TeacherHead(len(mean), next(iter(samples.values()))['numeric'].shape[-1], len(exp.labels)).to(exp.device)
        model.load_state_dict(torch.load(path, map_location='cpu', weights_only=False)['state_dict'])
        model.eval()
        output = {}
        for formulas in formula_batches(samples, held_ids, exp.config['batch_size']):
            batch = pack_formulas(formulas, exp.device)
            feature = torch.from_numpy(np.stack([(self.features[name][f[0]['formula_id']] - mean) / std for f in formulas])).to(exp.device)
            scores = model(feature.unsqueeze(1).expand(-1, batch['ink'].shape[1], -1), batch['numeric'], batch['token_ids'], batch['candidate_mask']).cpu().numpy()
            for i, formula in enumerate(formulas):
                for j, sample in enumerate(formula):
                    output[sample['record_id']] = scores[i, j, :len(sample['candidates'])]
        return output, root

    def crossfit(self, ids: list[str], samples: dict):
        """student fit 자료 안에서 writer를 하나씩 제외해 KD target을 만든다."""
        exp = self.experiment
        output, roots = dict(samples), []
        for writer in sorted({exp.corpus.records[key]['writer'] for key in ids}):
            held = [key for key in ids if exp.corpus.records[key]['writer'] == writer]
            fit = filtered_ids(exp.corpus.records, set(ids), set(held))
            predictions = []
            for name in self.names:
                values, root = self.predict(name, fit, held, samples)
                predictions.append(values); roots.append(root)
            for key in held:
                probabilities = []
                for values in predictions:
                    scores = values[key] / exp.config['temperature']
                    p = np.exp(scores - scores.max()); probabilities.append(p / p.sum())
                p = np.mean(probabilities, axis=0)
                output[key] = {**samples[key], 'teacher_scores': (np.log(np.maximum(p, 1e-12)) * exp.config['temperature']).astype(np.float32)}
        return output, sorted(set(roots))
