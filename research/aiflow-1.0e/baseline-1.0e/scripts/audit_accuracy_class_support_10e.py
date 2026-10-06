#!/usr/bin/env python3
"""고정 HWR 어휘의 외부 replay·자체 자료·outer fold별 지원량을 감사한다."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from accuracy_upgrade_contract_v1 import sha256_file
from accuracy_upgrade_data_10e import load_corpus, write_json


def _counts(labels: list[str], indices: np.ndarray) -> dict[str, int]:
    """고정 어휘 순서에 맞춰 0개 클래스도 빠뜨리지 않는다."""
    values = Counter(int(value) for value in np.asarray(indices).reshape(-1))
    if any(index < 0 or index >= len(labels) for index in values):
        raise ValueError('replay label index outside fixed vocabulary')
    return {label: int(values[index]) for index, label in enumerate(labels)}


def audit(config: dict, replay_cache: Path) -> dict:
    """평가 정답을 학습 입력으로 쓰지 않고 지원량과 결측만 기록한다."""
    checkpoint = Path(config['hwr_checkpoint'])
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    labels = list(payload['math_labels'])
    if len(labels) != 372 or len(set(labels)) != 372:
        raise ValueError('expected fixed 372-class vocabulary')
    manifest_path = replay_cache / 'cache_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest['config']['math_labels'] != labels:
        raise ValueError('replay vocabulary differs from checkpoint')
    train_item = manifest['sets']['math_train']
    train_path = replay_cache / train_item['labels']
    external = _counts(labels, np.load(train_path, mmap_mode='r'))

    corpus = load_corpus(config)
    own = Counter(corpus.raw[key]['label'] for key in corpus.raw)
    own_writers = defaultdict(set)
    for key, record in corpus.records.items():
        own_writers[corpus.raw[key]['label']].add(record['writer'])
    outer = {}
    writers = sorted({corpus.records[key]['writer'] for key in corpus.original_ids})
    all_ids = set(corpus.records)
    for held_writer in writers:
        held_formula_keys = {corpus.records[key]['formula_key'] for key in corpus.original_ids
                             if corpus.records[key]['writer'] == held_writer}
        fit = [key for key in all_ids if corpus.records[key]['writer'] != held_writer
               and corpus.records[key]['formula_key'] not in held_formula_keys]
        counts = Counter(corpus.raw[key]['label'] for key in fit)
        support = defaultdict(set)
        for key in fit:
            support[corpus.raw[key]['label']].add(corpus.records[key]['writer'])
        outer[held_writer] = {label: {'records': int(counts[label]), 'writers': len(support[label])}
                              for label in labels}

    rows = []
    for label in labels:
        rows.append({'label': label, 'external_replay_records': external[label],
                     'own_records_all_roles': int(own[label]), 'own_writers_all_roles': len(own_writers[label]),
                     'external_training_status': 'unobserved' if external[label] == 0 else 'observed'})
    return {
        'schema': 'aiflow-1.0e-class-support-audit/v1',
        'vocabulary_size': len(labels), 'vocabulary': labels,
        'checkpoint': str(checkpoint), 'checkpoint_sha256': sha256_file(str(checkpoint)),
        'replay_manifest': str(manifest_path), 'replay_manifest_sha256': sha256_file(str(manifest_path)),
        'data_role': 'support_audit_only', 'writer_identity_limitations': {
            'external_replay': 'writer counts unavailable in replay cache; HWRT aggregate writer identity is unverifiable',
            'own_data': 'canonical writer groups from source lineage'},
        'classes': rows, 'outer_fit_support': outer,
        'unobserved_external_labels': [label for label in labels if external[label] == 0],
        'under_supported_own_labels': [label for label in labels if len(own_writers[label]) < 3],
    }


def main() -> int:
    """명시한 새 JSON 경로에 감사 결과만 기록한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--replay-cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config = json.loads(args.config.read_text(encoding='utf-8'))
    write_json(args.output, audit(config, args.replay_cache))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
