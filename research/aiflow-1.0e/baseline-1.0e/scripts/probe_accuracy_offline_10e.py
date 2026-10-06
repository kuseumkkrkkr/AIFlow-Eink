"""teacher import와 네트워크를 차단한 별도 프로세스에서 원본 잉크 추론을 검사한다."""
from __future__ import annotations

import argparse
import copy
import importlib.abc
import json
import socket
import sys
import time
from pathlib import Path


class OfflineOnly(importlib.abc.MetaPathFinder):
    """제품 경로에 이미지 OCR 또는 teacher 모듈이 섞이면 즉시 거부한다."""
    blocked = {'transformers', 'PIL', 'unimernet', 'accuracy_teacher_targets_10e', 'train_ocr_decision_adapter_10e'}

    def find_spec(self, fullname, path=None, target=None):
        """실제 runtime import 요청에서만 금지 모듈을 판정한다."""
        if fullname.split('.')[0] in self.blocked:
            raise ImportError('offline runtime forbids ' + fullname)
        return None


def deny_network(*args, **kwargs):
    """이 검증 프로세스의 외부 연결 시도를 예외로 남긴다."""
    raise RuntimeError('network disabled for offline inference verification')


def main():
    """실제 held 수식으로 정답 비의존성과 완료 이벤트 멱등성을 검증한다."""
    p = argparse.ArgumentParser()
    p.add_argument('--experiment', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--joint', action='store_true')
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    sys.meta_path.insert(0, OfflineOnly())
    socket.socket.connect = deny_network
    socket.create_connection = deny_network
    import joblib
    import torch
    from accuracy_upgrade_data_10e import load_hwr, load_corpus, write_json
    from accuracy_student_training_10e import new_student
    from accuracy_joint_runtime_10e import prepare_formula, predict_formula, CompletionCache
    from accuracy_upgrade_contract_v1 import canonical_json_sha256, sha256_file

    torch.set_num_threads(4)
    ledger = json.loads((a.experiment / 'experiment.json').read_text(encoding='utf-8'))
    config = ledger['configuration']
    manifest = json.loads((a.experiment / 'model_manifest.json').read_text(encoding='utf-8'))
    if sha256_file(config['hwr_checkpoint']) != manifest['hwr_checkpoint_sha256']:
        raise ValueError('HWR weights mismatch')
    device = torch.device('cpu')
    hwr_only = ledger['stage'] == 'hwr'
    hwr_path = a.experiment / 'models/outer0_hwr_refit.pt' if hwr_only else Path(config['hwr_checkpoint'])
    hwr, labels = load_hwr(hwr_path, device)
    model, guard = None, None
    if not hwr_only:
        payload = torch.load(a.experiment / 'models/outer0_student.pt', map_location='cpu', weights_only=False)
        if payload['hwr_checkpoint_sha256'] != manifest['hwr_checkpoint_sha256'] or payload['labels'] != labels:
            raise ValueError('student/HWR pair mismatch')
        model = new_student(payload['architecture'], payload['numeric_size'], len(labels)).eval()
        model.load_state_dict(payload['state_dict'])
        guard = joblib.load(a.experiment / 'models/outer0_guard.joblib')
    grouping = joblib.load(a.experiment / 'models/outer0_raw_grouping.joblib')
    corpus = load_corpus(config)
    split = json.loads((a.experiment / 'split_manifest.json').read_text(encoding='utf-8'))['splits'][0]
    if hwr_only:
        from accuracy_lineage_10e import LineageRegistry
        registry = LineageRegistry.load(a.experiment / 'lineage_manifest.json')
        roots = [key for key,node in registry.nodes.items() if node['artifact_path']==str(hwr_path.resolve())]
        if len(roots)!=1 or labels!=manifest['vocabulary']:
            raise ValueError('HWR lineage or vocabulary mismatch')
        registry.validate(roots[0], {split['outer_writer']}, {registry.records[k]['formula_key'] for k in split['held_ids']})
    formula_id = corpus.records[split['held_ids'][0]]['formula_id']
    source = dict(formula_id=formula_id, strokes=copy.deepcopy(corpus.formulas[formula_id]['strokes']))
    arguments = ledger['arguments']

    def inference(raw):
        """원본 두 필드 외의 주석은 runtime이 읽을 수 없어야 한다."""
        prepared = prepare_formula(raw, hwr, labels, device, config['input_mode'], arguments['window'], arguments['neighbors'])
        return predict_formula(prepared, grouping, labels, device, model, top_k=arguments['top_k'],
            feature_version=arguments['features'], geometry_mode=arguments['geometry'], joint=a.joint, guard=guard)

    started = time.perf_counter()
    prediction = inference(source)
    mutated = {**copy.deepcopy(source), 'label': 'POISON', 'target_latex': 'WRONG', 'writer_group': 'POISON',
               'truth_groups': [[999]], 'target_relations': [{'type': 'POISON'}]}
    if prediction != inference(mutated):
        raise ValueError('ground truth altered runtime prediction')
    if source['strokes'] != corpus.formulas[formula_id]['strokes']:
        raise ValueError('runtime mutated raw ink')
    cache = CompletionCache()
    first = cache.complete('session/complete-1', source, {'version': 'research-probe'}, lambda: inference(source))
    second = cache.complete('session/complete-1', source, {'version': 'research-probe'}, lambda: inference(source))
    if first != second:
        raise ValueError('formula_complete is not idempotent')
    forbidden_loaded = sorted(set(sys.modules) & OfflineOnly.blocked)
    if forbidden_loaded:
        raise ValueError('teacher/image dependency loaded: ' + repr(forbidden_loaded))
    result = dict(status='completed', input_formula=formula_id, device='local CPU; not mobile acceptance',
                  network_disabled=True, forbidden_modules_loaded=[], annotation_invariant=True,
                  repeated_completion_identical=True, raw_ink_preserved=True, source_sha256=canonical_json_sha256(source),
                  latency_total_seconds=time.perf_counter() - started, raw_latex=prediction['raw_latex'],
                  product_activated=False, scope='research runtime probe; packaging and promotion remain gated')
    result.update(hwr_checkpoint_sha256=sha256_file(str(hwr_path)), joint=a.joint,
                  runtime_sha256=sha256_file(str(Path(__file__).with_name('accuracy_joint_runtime_10e.py'))))
    write_json(a.output, result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
