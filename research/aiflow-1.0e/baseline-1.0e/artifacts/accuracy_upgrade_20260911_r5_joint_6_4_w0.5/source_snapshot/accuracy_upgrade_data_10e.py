"""승인된 자체 잉크를 원본 수식·획 소유권과 연결하는 연구 입력."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import gzip
import io
import json
from pathlib import Path

import numpy as np
import torch

from accuracy_upgrade_contract_v1 import (canonical_json_sha256, formula_bounds, formula_position,
                                         normalize_latex, relative_context, sha256_file, source_bbox, writer_key)
from online_candidate_features_10e import _json_lines, _row_numeric
from character_tensor_v1 import iter_direct_ownership_examples, tensorize
from build_trocr_hwr95_candidates_v1 import _geometry
from train_character_classifier_v1 import InkClassifierV1, apply_input_mode


def write_json(path: Path, value) -> None:
    """새 실험의 JSON 산출물을 UTF-8로 기록한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')


def write_rows(path: Path, rows: list[dict]) -> None:
    """시간·파일명을 포함하지 않는 재현 가능한 gzip JSONL을 기록한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as stream, gzip.GzipFile(filename='', fileobj=stream, mode='wb', mtime=0) as zipped:
        with io.TextIOWrapper(zipped, encoding='utf-8', newline='\n') as text:
            for row in rows:
                text.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + '\n')


def display_to_latex(display: str) -> str:
    """수집 prompt의 명시적 Unicode 루트 범위를 고정 LaTeX 표기로 변환한다."""
    text = str(display)
    position = 0
    while True:
        start = text.find('√(', position)
        if start < 0:
            break
        depth, end = 1, start + 2
        while end < len(text) and depth:
            depth += (text[end] == '(') - (text[end] == ')')
            end += 1
        if depth:
            raise ValueError('unbalanced root in annotated display')
        text = text[:start] + r'\sqrt{' + text[start + 1:end] + '}' + text[end:]
        position = start + 6
    return text


@dataclass
class Corpus:
    """추론 입력과 평가 정답을 별도 필드로 관리한다."""
    raw: dict[str, dict]
    records: dict[str, dict]
    formulas: dict[str, dict]
    candidates: dict[str, dict]
    original_ids: set[str]
    input_hashes: dict[str, str]


def load_corpus(config: dict) -> Corpus:
    """원본 재생성과 SHA 대조로 writer 별칭·수식 ID 연결을 확인한다."""
    raw_rows = _json_lines(Path(config['raw']))
    candidates = _json_lines(Path(config['candidates']))
    raw = {str(row['record_id']): row for row in raw_rows}
    by_id = {str(row['record_id']): row for row in candidates}
    if len(raw) != len(raw_rows) or len(by_id) != len(candidates) or raw.keys() != by_id.keys():
        raise ValueError('raw/candidate IDs must be unique and have identical coverage')
    originals = {str(row['record_id']) for row in _json_lines(Path(config['original_candidates']))}
    if not originals <= raw.keys():
        raise ValueError('original evaluation records missing')
    hashes = {config[name]: sha256_file(config[name]) for name in ('raw', 'candidates', 'original_candidates')}
    formulas, joined = {}, {}
    for source in config['sources']:
        formula_path, ownership_path = Path(source['formulas']), Path(source['ownership'])
        hashes[str(formula_path)] = sha256_file(str(formula_path))
        hashes[str(ownership_path)] = sha256_file(str(ownership_path))
        source_formulas = {str(row['sample_id']): row for row in _json_lines(formula_path)}
        owners = {str(row['sample_id']): row for row in _json_lines(ownership_path) if row.get('accepted')}
        generated = {str(row['record_id']): row for row in iter_direct_ownership_examples(formula_path, ownership_path)}
        for record_id in sorted(raw.keys() & generated.keys()):
            row = generated[record_id]
            if row['source_fingerprint'] != raw[record_id]['source_fingerprint']:
                continue
            formula_id = str(by_id[record_id]['formula_id'])
            if formula_id not in owners or formula_id not in source_formulas:
                continue
            formula = source_formulas[formula_id]
            annotation = owners[formula_id]
            index = int(raw[record_id]['formula_group_index'])
            if annotation['labels'][index] != raw[record_id]['label'] or writer_key(row) != writer_key(raw[record_id]):
                raise ValueError('raw ownership/writer mapping mismatch')
            if formula_id in formulas and canonical_json_sha256(formulas[formula_id]['strokes']) != canonical_json_sha256(formula['strokes']):
                raise ValueError('ambiguous formula ID across raw snapshots')
            assigned = [int(i) for group in annotation['groups'] for i in group]
            if sorted(assigned) != list(range(len(formula['strokes']))):
                raise ValueError('ownership must cover every stroke exactly once')
            formulas[formula_id] = dict(formula_id=formula_id, strokes=formula['strokes'], canvas=formula.get('canvas', {}),
                target_display=str(formula['target_display']), target_latex=display_to_latex(formula['target_display']), target_relations=formula.get('target_relations', []),
                truth_groups=annotation['groups'], truth_labels=annotation['labels'], writer=writer_key(row),
                raw_sha256=canonical_json_sha256(formula['strokes']))
            joined[record_id] = dict(writer=writer_key(row), formula_id=formula_id,
                formula_key=normalize_latex(display_to_latex(formula['target_display'])), raw_sha256=canonical_json_sha256(raw[record_id]),
                role='legacy_development' if record_id in originals else 'training_only_shadow',
                source_indices=list(annotation['groups'][index]))
    if joined.keys() != raw.keys():
        raise ValueError(f'raw formula join incomplete: {len(raw.keys() - joined.keys())}')
    aliases = defaultdict(set)
    reverse_aliases = defaultdict(set)
    for record_id, row in by_id.items():
        aliases[str(row['writer_group'])].add(joined[record_id]['writer'])
        reverse_aliases[joined[record_id]['writer']].add(str(row['writer_group']))
    if any(len(values) != 1 for values in list(aliases.values()) + list(reverse_aliases.values())):
        raise ValueError('writer namespace mapping is not one-to-one')
    return Corpus(raw, joined, formulas, by_id, originals, hashes)


def load_hwr(path: Path, device: torch.device) -> tuple[InkClassifierV1, list[str]]:
    """고정 372개 어휘를 체크포인트에서 읽고 기존 신경망을 재사용한다."""
    payload = torch.load(path, map_location='cpu', weights_only=False)
    labels = list(payload['math_labels'])
    if len(labels) != 372 or len(set(labels)) != 372 or payload.get('auxiliary_labels') not in ([], None):
        raise ValueError('invalid fixed HWR vocabulary')
    model = InkClassifierV1(len(labels), 0).to(device)
    model.load_state_dict(payload['state_dict'], strict=True)
    return model.eval(), labels


@torch.inference_mode()
def encode_rows(model: InkClassifierV1, raw: dict[str, dict], input_mode: str, device: torch.device):
    """하나의 정확한 HWR 상태에서 embedding과 전체 logits를 함께 생성한다."""
    ids = sorted(raw)
    embeddings, logits = {}, {}
    for start in range(0, len(ids), 32):
        selected = ids[start:start + 32]
        batch = np.stack([apply_input_mode(tensorize(raw[key]), input_mode) for key in selected])
        ink = model.encode(torch.from_numpy(batch).to(device))
        scores = model.math_head(ink)
        for key, vector, score in zip(selected, ink.cpu().numpy(), scores.cpu().numpy()):
            embeddings[key], logits[key] = vector.astype(np.float32), score.astype(np.float32)
    return embeddings, logits


def build_samples(corpus: Corpus, embeddings: dict, logits: dict, labels: list[str], k: int = 5,
                  feature_version: str = 'formula28', geometry_mode: str = 'restored') -> dict[str, dict]:
    """정답 묶음 진단용 후보를 생성하고 수식 내 실제 좌표를 복원한다."""
    if k not in (5, 10, 20) or k > len(labels):
        raise ValueError('research K must be 5, 10 or 20')
    groups = defaultdict(list)
    for key, record in corpus.records.items():
        groups[record['formula_id']].append(key)
    result = {}
    for formula_id, keys in groups.items():
        keys.sort(key=lambda key: (min(corpus.records[key]['source_indices']), key))
        boxes = [source_bbox(corpus.raw[key]) for key in keys]
        bounds = formula_bounds(boxes)
        for index, (key, box) in enumerate(zip(keys, boxes)):
            score = np.asarray(logits[key], dtype=np.float32)
            if score.shape != (len(labels),) or not np.isfinite(score).all():
                raise ValueError('invalid full vocabulary logits')
            indices = np.argsort(-score, kind='stable')[:k]
            probability = np.exp(score - score.max()); probability /= probability.sum()
            context = relative_context(boxes[index - 1] if index else None, box,
                boxes[index + 1] if index + 1 < len(keys) else None, bounds)
            geometry = _geometry(corpus.raw[key])
            if geometry_mode == 'restored':
                geometry.update(formula_position(box, bounds))
            elif geometry_mode == 'legacy':
                context = dict.fromkeys(('previous_dx', 'previous_dy', 'next_dx', 'next_dy'), 0.)
            else:
                raise ValueError('unknown geometry mode')
            row = dict(record_id=key, formula_id=formula_id, geometry=geometry, context=context,
                       final_topk=[labels[i] for i in indices], final_topk_probabilities=probability[indices].tolist())
            target = corpus.raw[key]['label']
            result[key] = dict(record_id=key, formula_id=formula_id, writer_group=corpus.records[key]['writer'],
                formula_key=corpus.records[key]['formula_key'],
                label=target, candidates=row['final_topk'], target=row['final_topk'].index(target) if target in row['final_topk'] else -1,
                ink=embeddings[key], token_ids=indices.astype(np.int64), base_logits=score[indices],
                numeric=np.asarray([_row_numeric(row, i, feature_version) for i in range(k)], dtype=np.float32),
                row=row, source_indices=corpus.records[key]['source_indices'])
    return result


def formula_batches(samples: dict[str, dict], ids: list[str], batch_size: int, seed: int | None = None):
    """하나의 수식을 나누지 않고 모든 문자를 batch에 포함한다."""
    groups = defaultdict(list)
    for key in ids:
        groups[samples[key]['formula_id']].append(samples[key])
    formulas = [sorted(values, key=lambda item: (min(item['source_indices']), item['record_id']))
                for _, values in sorted(groups.items())]
    if seed is not None:
        np.random.default_rng(seed).shuffle(formulas)
    for start in range(0, len(formulas), batch_size):
        yield formulas[start:start + batch_size]


def pack_formulas(formulas: list[list[dict]], device: torch.device) -> dict:
    """정답 tensors를 forward tensors와 구분하고 padding 마스크를 명시한다."""
    if not formulas or any(not formula for formula in formulas):
        raise ValueError('empty training formula batch')
    b, length = len(formulas), max(map(len, formulas))
    k = max(len(sample['candidates']) for formula in formulas for sample in formula)
    n = formulas[0][0]['numeric'].shape[-1]
    values = dict(ink=np.zeros((b, length, 128), np.float32), numeric=np.zeros((b, length, k, n), np.float32),
                  token_ids=np.zeros((b, length, k), np.int64), base_logits=np.zeros((b, length, k), np.float32),
                  candidate_mask=np.zeros((b, length, k), bool), row_mask=np.zeros((b, length), bool),
                  targets=np.full((b, length), -1, np.int64))
    if any('teacher_scores' in sample for formula in formulas for sample in formula):
        if any('teacher_scores' not in sample for formula in formulas for sample in formula):
            raise ValueError('partial teacher coverage')
        values['teacher_scores'] = np.zeros((b, length, k), np.float32)
    for i, formula in enumerate(formulas):
        for j, sample in enumerate(formula):
            width = len(sample['candidates'])
            for name in ('numeric', 'token_ids', 'base_logits'):
                values[name][i, j, :width] = sample[name]
            values['ink'][i, j] = sample['ink']
            values['targets'][i, j] = sample['target']
            values['row_mask'][i, j] = True
            values['candidate_mask'][i, j, :width] = True
            if 'teacher_scores' in values:
                values['teacher_scores'][i, j, :width] = sample['teacher_scores']
    return {key: torch.from_numpy(value).to(device) for key, value in values.items()}
