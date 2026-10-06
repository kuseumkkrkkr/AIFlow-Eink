"""원본 획의 exact-cover 후보를 online HWR·문맥·기존 구조로 공동 선택한다."""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier

from accuracy_upgrade_contract_v1 import canonical_json_sha256, formula_bounds, formula_position, relative_context, source_bbox
from accuracy_upgrade_data_10e import pack_formulas
from build_normalized_ink_v1 import SourceSample, _canonicalize
from character_tensor_v1 import tensorize
from train_character_classifier_v1 import apply_input_mode
from online_candidate_features_10e import _row_numeric
from build_trocr_hwr95_candidates_v1 import _geometry
from stroke_grouping_v1 import build_lattice, candidate_features, enumerate_partitions
from formula_layout_v1 import infer_formula_layout, selected_layout_evidence_rows, serialize_formula
from accuracy_temporal_context_10e import temporal_context
from accuracy_latex_serializer_10e import serialize_selected_graph


def canonical_group(strokes: list[dict], indices: list[int], formula_id: str) -> tuple[dict, bool]:
    """예측 grouping만 받아 원본 단일점도 삭제하지 않고 정규화한다."""
    selected = [strokes[index] for index in indices]
    points = [[(float(p['x']), float(p['y']), p.get('t_ms')) for p in stroke['points']] for stroke in selected]
    if sum(map(len, points)) == 1:
        x, y, _ = points[0][0]
        row = dict(strokes=[dict(source_order=0, points=[[.5, .5, 0.]])],
                   transform={'bbox': dict(left=x, right=x, top=y, bottom=y)}, normalization={}, source='online_runtime')
        return row, True
    row = _canonicalize(SourceSample('online_runtime', formula_id + ':' + ','.join(map(str, indices)),
                                     '', 'inference', 'online_inference', points))
    return row, False


def group_tensor(row: dict, single_point: bool, input_mode: str) -> np.ndarray:
    """한 점은 첫 관측만 표시하고 127개의 padding을 새 획으로 취급하지 않는다."""
    if not single_point:
        return apply_input_mode(tensorize(row), input_mode)
    result = np.zeros((128, 5), dtype=np.float32)
    result[:, :2] = .5
    result[0, 3:] = 1.
    return result


@torch.inference_mode()
def prepare_formula(source: dict, hwr, labels: list[str], device: torch.device, input_mode='uniform-time',
                    temporal_window=6, spatial_neighbors=4) -> dict:
    """정답 필드를 읽지 않고 원본 획에서 grouping/HWR 캐시를 만든다."""
    formula_id = str(source['formula_id'])
    strokes = sorted(copy.deepcopy(source['strokes']), key=lambda row: int(row['order']))
    if [int(row['order']) for row in strokes] != list(range(len(strokes))):
        raise ValueError('raw stroke order must be unique, contiguous and zero-based')
    candidates = build_lattice(strokes, temporal_window=temporal_window, spatial_neighbors=spatial_neighbors)
    rows, tensors = [], []
    for candidate in candidates:
        row, single = canonical_group(strokes, candidate['source_indices'], formula_id)
        rows.append(row)
        tensors.append(group_tensor(row, single, input_mode))
    ink_vectors, all_logits = [], []
    for start in range(0, len(tensors), 32):
        encoded = hwr.encode(torch.from_numpy(np.stack(tensors[start:start + 32])).to(device))
        ink_vectors.extend(encoded.cpu().numpy())
        all_logits.extend(hwr.math_head(encoded).cpu().numpy())
    return dict(formula_id=formula_id, strokes=strokes, raw_sha256=canonical_json_sha256(strokes),
                candidates=candidates, features=candidate_features(candidates, strokes), canonical=rows,
                ink=np.asarray(ink_vectors, np.float32), logits=np.asarray(all_logits, np.float32))


def fit_grouping(prepared: dict[str, dict], truth_groups: dict[str, list], formula_ids: list[str], seed: int):
    """해당 fold 학습 수식의 소유권만 사용해 기존 grouping 분류기를 재학습한다."""
    features, targets, weights = [], [], []
    for formula_id in sorted(formula_ids):
        item = prepared[formula_id]
        truth = {frozenset(group) for group in truth_groups[formula_id]}
        values = np.asarray([frozenset(c['source_indices']) in truth for c in item['candidates']], dtype=np.int64)
        features.append(item['features']); targets.extend(values)
        weights.extend(np.where(values == 1, 1. / max(int(values.sum()), 1), 1. / max(int((values == 0).sum()), 1)))
    if not features or len(set(targets)) != 2:
        raise ValueError('grouping fit requires positive and negative labeled candidates')
    return HistGradientBoostingClassifier(learning_rate=.07, max_iter=160, max_leaf_nodes=15,
        l2_regularization=1., min_samples_leaf=20, random_state=seed).fit(np.concatenate(features), targets, sample_weight=weights)


def partition_samples(item: dict, groups, labels: list[str], k: int, feature_version: str, geometry_mode='restored') -> list[dict]:
    """선택 partition의 실제 좌표·획 순서로 문맥 특징을 생성한다."""
    by_group = {frozenset(c['source_indices']): i for i, c in enumerate(item['candidates'])}
    groups = sorted(groups, key=lambda g: (min(g), tuple(sorted(g))))
    indices = [by_group[frozenset(group)] for group in groups]
    boxes = [source_bbox(item['canonical'][i]) for i in indices]
    bounds = formula_bounds(boxes)
    result = []
    for j, index in enumerate(indices):
        raw = item['canonical'][index]
        scores = item['logits'][index]
        ids = np.argsort(-scores, kind='stable')[:k]
        probability = np.exp(scores - scores.max()); probability /= probability.sum()
        geometry = _geometry(raw)
        geometry.update(formula_position(boxes[j], bounds))
        context = relative_context(boxes[j - 1] if j else None, boxes[j], boxes[j + 1] if j + 1 < len(boxes) else None, bounds)
        if geometry_mode == 'legacy':
            geometry = _geometry(raw)
            context = dict.fromkeys(('previous_dx', 'previous_dy', 'next_dx', 'next_dy'), 0.)
        elif geometry_mode != 'restored':
            raise ValueError('unknown geometry mode')
        if feature_version == 'formula40':
            if geometry_mode != 'restored':
                raise ValueError('formula40 requires restored geometry')
            context.update(temporal_context(item['strokes'], groups, boxes, j))
        key = item['formula_id'] + ':' + ','.join(map(str, sorted(groups[j])))
        row = dict(record_id=key, formula_id=item['formula_id'], geometry=geometry, context=context,
                   final_topk=[labels[i] for i in ids], final_topk_probabilities=probability[ids].tolist())
        result.append(dict(record_id=key, formula_id=item['formula_id'], ink=item['ink'][index], numeric=np.asarray(
            [_row_numeric(row, i, feature_version) for i in range(k)], np.float32), token_ids=ids.astype(np.int64),
            base_logits=scores[ids], candidates=row['final_topk'], target=-1, row=row,
            source_indices=sorted(groups[j]), raw_box=boxes[j]))
    return result


def layout_output(samples: list[dict], token_indices: list[int]) -> tuple[str, dict, float]:
    """기존 serializer에 원본 좌표와 선택 token을 전달하고 순환을 검증한다."""
    rows, predictions = [], {}
    for sample, index in zip(samples, token_indices, strict=True):
        row = {**sample['row'], 'geometry': {**sample['row']['geometry'], **sample['raw_box']}}
        rows.append(row)
        predictions[sample['record_id']] = sample['candidates'][index]
    evidence, _ = selected_layout_evidence_rows(rows, predictions)
    for row in evidence:
        row['final_topk_probabilities'] = [float(token == predictions[row['record_id']]) for token in row['final_topk']]
    structure = infer_formula_layout(evidence)
    latex = serialize_selected_graph(evidence, predictions, structure)
    graph = dict(order=structure['ordered_record_ids'], edges=structure['edges'])
    score = sum(math.log(max(1e-8, min(1., float(edge['confidence'])))) for edge in structure['edges']) / max(len(structure['edges']), 1)
    return latex, graph, score


@torch.inference_mode()
def predict_formula(item: dict, grouping, labels: list[str], device: torch.device, model=None, top_k=5,
                    feature_version='formula28', structure_weight=1., joint=True, token_beam=32, guard=None, geometry_mode='restored') -> dict:
    """상위 32 partition에서 후보·구조 점수를 결합하고 모든 획을 보존한다."""
    probability = np.clip(grouping.predict_proba(item['features'])[:, 1], 1e-6, 1 - 1e-6)
    logits = np.log(probability / (1. - probability))
    partitions = enumerate_partitions(item['candidates'], logits, len(item['strokes']), top_n=32 if joint else 1, beam_width=2048)
    if not partitions:
        raise ValueError('no exact-cover partition')
    formulas = [partition_samples(item, groups, labels, top_k, feature_version, geometry_mode) for _, groups in partitions]
    batch = pack_formulas(formulas, device)
    if model is None:
        scores = batch['base_logits']
    else:
        model.eval()
        scores, coverage = model.forward_with_coverage(batch['ink'], batch['numeric'], batch['token_ids'], batch['candidate_mask'],
                       row_mask=batch['row_mask'], base_logits=batch['base_logits'])
    log_probability = torch.log_softmax(scores.masked_fill(~batch['candidate_mask'], -1e9), dim=-1).cpu().numpy()
    if guard is not None and model is not None:
        from accuracy_evaluation_10e import apply_guard
        for i, formula in enumerate(formulas):
            rows = []
            for j, sample in enumerate(formula):
                row_scores = scores[i, j, :len(sample['candidates'])].cpu().numpy()
                rows.append(dict(record_id=sample['record_id'], candidates=sample['candidates'],
                    baseline_token=sample['candidates'][0], adapter_token=sample['candidates'][int(np.argmax(row_scores))],
                    adapter_scores=row_scores.tolist(), baseline_probabilities=sample['row']['final_topk_probabilities'],
                    coverage_probability=float(torch.sigmoid(coverage[i, j]))))
            for j, row in enumerate(apply_guard(guard, rows)):
                chosen = row['candidates'].index(row['guarded_token'])
                current = log_probability[i, j, chosen]
                log_probability[i, j, :] = -1e9
                log_probability[i, j, chosen] = current
    best = None
    group_index = {frozenset(c['source_indices']): i for i, c in enumerate(item['candidates'])}
    for pindex, samples in enumerate(formulas):
        group_score = sum(math.log(float(probability[group_index[frozenset(s['source_indices'])]])) for s in samples) / len(samples)
        beams = [(0., [])]
        for j, sample in enumerate(samples):
            proposals = [(score + float(log_probability[pindex, j, token]), tokens + [token])
                         for score, tokens in beams for token in range(len(sample['candidates']))]
            beams = sorted(proposals, key=lambda row: (-row[0], row[1]))[:token_beam if joint else 1]
        for char_score, token_indices in beams:
            try:
                latex, graph, relation_score = layout_output(samples, token_indices)
            except ValueError:
                continue
            score = group_score + char_score / len(samples) + structure_weight * relation_score
            if best is None or score > best[0]:
                best = (score, dict(raw_latex=latex, groups=[s['source_indices'] for s in samples],
                    tokens=[s['candidates'][i] for s, i in zip(samples, token_indices)], structure=graph,
                    candidates=[s['candidates'] for s in samples], review_requested=False, failure_reason=None,
                    grouping_log_score=group_score, character_log_score=char_score / len(samples), relation_log_score=relation_score))
    if best is None:
        raise ValueError('all structure candidates failed')
    result = best[1]
    assigned = [i for group in result['groups'] for i in group]
    if sorted(assigned) != list(range(len(item['strokes']))):
        raise AssertionError('lost or duplicate source stroke')
    result.update(formula_id=item['formula_id'], raw_sha256=item['raw_sha256'], input_stroke_count=len(item['strokes']),
                  partition_count=len(partitions), score=best[0], model_input='online_ink_only')
    return result


class CompletionCache:
    """동일 완료 이벤트의 원본·설정을 확인하고 결과 복사본을 반환한다."""
    def __init__(self):
        """세션 범위의 멱등 캐시를 초기화한다."""
        self.values = {}

    def complete(self, event_id: str, raw: dict, configuration: dict, predict):
        """같은 ID에 다른 잉크가 들어오면 기존 결과를 오용하지 않는다."""
        key = canonical_json_sha256(dict(strokes=raw['strokes'], configuration=configuration))
        if event_id in self.values:
            previous, result = self.values[event_id]
            if previous != key:
                raise ValueError('completion event ID reused with different input')
            return copy.deepcopy(result)
        try:
            result = predict()
        except (ValueError, RuntimeError, FloatingPointError) as error:
            result = dict(raw_latex=None, review_requested=True, failure_reason=str(error))
        result['raw_ink'] = copy.deepcopy(raw['strokes'])
        self.values[event_id] = (key, copy.deepcopy(result))
        return result
