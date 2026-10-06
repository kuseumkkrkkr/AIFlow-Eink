"""AIFlow 1.0e의 명시적 구성·strict 계보·writer 분할 연구 실행기."""
from __future__ import annotations

import argparse
import copy
import json
import platform
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import torch

from accuracy_upgrade_contract_v1 import canonical_json_sha256, normalize_latex, sha256_file, LATEX_NORMALIZATION_VERSION
from accuracy_upgrade_data_10e import load_corpus, load_hwr, encode_rows, build_samples, write_json, write_rows
from accuracy_lineage_10e import LineageRegistry, filtered_ids
from accuracy_joint_runtime_10e import prepare_formula, fit_grouping, predict_formula
from accuracy_evaluation_10e import glyph_metrics, formula_metrics, fit_guard, apply_guard
from online_candidate_features_10e import _json_lines, feature_names

ROOT = Path(__file__).resolve().parents[1]


def event(name: str, **values) -> None:
    """긴 실행 중 진행 위치를 JSON 한 줄로 기록한다."""
    print(json.dumps(dict(event=name, **values), ensure_ascii=False, allow_nan=False), flush=True)


def truth_result(corpus, formula_id: str, prediction: dict) -> dict:
    """추론이 끝난 뒤에만 원본 정답과 구조 대응을 평가 결과에 합친다."""
    truth = corpus.formulas[formula_id]
    groups = {frozenset(group) for group in prediction.get('groups', [])}
    owned = {frozenset(group) for group in truth['truth_groups']}
    grouping_exact = groups == owned
    true_tokens = {frozenset(g): t for g, t in zip(truth['truth_groups'], truth['truth_labels'])}
    glyphs_exact = grouping_exact and all(true_tokens[frozenset(g)] == t for g, t in zip(prediction.get('groups', []), prediction.get('tokens', [])))
    # target relation node fNNN은 annotation group index에 대응한다.
    row_groups = {formula_id + ':' + ','.join(map(str, sorted(g))): frozenset(g) for g in prediction.get('groups', [])}
    observed = set()
    for edge in prediction.get('structure', {}).get('edges', []):
        observed.add((row_groups.get(edge['parent']), row_groups.get(edge['child']), edge['type']))
    expected = set()
    # 빈 relation 배열은 복잡한 구조의 독립 주석 완료를 뜻하지 않는다.
    relation_supported = bool(truth.get('target_relations')) or not any(
        token in truth['target_latex'] for token in ('\\sqrt', '\\frac', '^', '_', '²', '³'))
    for edge in truth.get('target_relations', []):
        try:
            parent = int(str(edge.get('from', edge.get('parent', ''))).lstrip('f'))
            child = int(str(edge.get('to', edge.get('child', ''))).lstrip('f'))
            expected.add((frozenset(truth['truth_groups'][parent]), frozenset(truth['truth_groups'][child]), str(edge.get('type', edge.get('relation'))).lower()))
        except (ValueError, IndexError):
            relation_supported = False
    result = dict(prediction, writer_group=truth['writer'], target_latex=truth['target_latex'],
        normalized_latex=normalize_latex(prediction.get('raw_latex') or ''), grouping_exact=grouping_exact,
        structure_exact=bool(glyphs_exact and relation_supported and expected == observed),
        structure_annotation_supported=relation_supported, data_role='consumed_development')
    return result


def evaluate_raw(corpus, prepared, grouping, formula_ids, labels, device, model=None, k=5,
                 feature_version='formula28', joint=False, structure_weight=1., guard=None, geometry_mode='restored'):
    """주어진 모든 원본 수식에 성공 또는 명시적 실패를 기록한다."""
    rows = []
    for formula_id in sorted(formula_ids):
        started = time.perf_counter()
        try:
            result = predict_formula(prepared[formula_id], grouping, labels, device, model, top_k=k,
                                     feature_version=feature_version, joint=joint, structure_weight=structure_weight, guard=guard,
                                     geometry_mode=geometry_mode)
        except (ValueError, RuntimeError, FloatingPointError) as error:
            result = dict(formula_id=formula_id, raw_latex=None, review_requested=True, failure_reason=str(error),
                          raw_sha256=corpus.formulas[formula_id]['raw_sha256'])
        result['latency_seconds'] = time.perf_counter() - started
        rows.append(truth_result(corpus, formula_id, result))
    return rows


class Experiment:
    """하나의 가설 실행에서 공유할 원본·가중치·계보·캐시를 보관한다."""
    def __init__(self, args, config: dict):
        """출력·입력 SHA를 고정하고 외부 기반 HWR의 학습 출처를 검사한다."""
        self.args, self.config, self.output = args, config, args.output.resolve()
        if self.output.drive.upper() != 'D:':
            raise ValueError('research artifacts must remain on D:')
        if self.output.exists():
            raise FileExistsError(f'refusing to overwrite {self.output}')
        self.corpus = load_corpus(config)
        self.device = torch.device(args.device)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA requested but unavailable')
        torch.set_num_threads(4)
        np.random.seed(config['seed']); torch.manual_seed(config['seed'])
        self.hwr_path = Path(config['hwr_checkpoint'])
        report = json.loads(Path(config['hwr_training_report']).read_text(encoding='utf-8'))
        admission = report.get('data_admission', {})
        if report.get('schema') != 'aiflow-character-classifier-external-training/v1' or admission.get('project_owned_training') is not False or admission.get('crohme_training') is not False:
            raise ValueError('base HWR is not verified external-only')
        if set(admission.get('train_sources', {})) - {'hwrt', 'uji', 'isgl', 'uci'}:
            raise ValueError('unapproved source in base HWR')
        self.hwr, self.labels = load_hwr(self.hwr_path, self.device)
        self.output.mkdir(parents=True)
        self.registry = LineageRegistry(self.corpus.records)
        self.hwr_node = self.registry.add('external_only_hwr', self.hwr_path,
            training_report_sha256=sha256_file(config['hwr_training_report']), data_rights_catalog_sha256=sha256_file(config['data_rights_catalog']))
        self.base_node = self.hwr_node
        self.samples = None
        self.prepared = None
        self.outer_writers = sorted({self.corpus.records[key]['writer'] for key in self.corpus.original_ids})
        self.started = time.perf_counter()
        self.model_manifest = dict(model_id='raw_ink_joint' if args.stage == 'joint' else 'candidate_ranker',
            hwr_checkpoint=str(self.hwr_path), hwr_checkpoint_sha256=sha256_file(str(self.hwr_path)),
            vocabulary=self.labels, vocabulary_sha256=canonical_json_sha256(self.labels),
            input_channels=['x', 'y', 'delta_t', 'stroke_start', 'observed'], normalization_version=LATEX_NORMALIZATION_VERSION,
            input_mode=config['input_mode'], feature_version=args.features, feature_names=list(feature_names(args.features)),
            external_teacher_in_runtime=False, code_sha256={p.name: sha256_file(str(p)) for p in
                [Path(__file__), ROOT / 'scripts/accuracy_joint_runtime_10e.py', ROOT / 'scripts/accuracy_upgrade_data_10e.py',
                 ROOT / 'scripts/accuracy_lineage_10e.py', ROOT / 'scripts/formula_context_ranker_10e.py',
                 ROOT / 'scripts/accuracy_student_training_10e.py', ROOT / 'scripts/formula_candidate_loss_10e.py',
                 ROOT / 'scripts/accuracy_upgrade_contract_v1.py', ROOT / 'scripts/online_candidate_features_10e.py',
                 ROOT / 'scripts/accuracy_evaluation_10e.py', ROOT / 'scripts/accuracy_teacher_targets_10e.py']})
        write_json(self.output / 'model_manifest.json', self.model_manifest)
        write_json(self.output / 'experiment.json', dict(status='running', stage=args.stage, configuration=config,
            arguments=vars(args) | {'config': str(args.config), 'output': str(args.output), 'cache': str(args.cache)},
            command=sys.argv, python=sys.version, platform=platform.platform(), torch=torch.__version__,
            cuda=torch.version.cuda, device=str(self.device), gpu=torch.cuda.get_device_name() if self.device.type == 'cuda' else None))
        self.splits = self.make_splits()
        write_json(self.output / 'split_manifest.json', dict(inputs=self.corpus.input_hashes, records=self.corpus.records,
                                                            splits=self.splits, role='consumed_development'))

    def make_splits(self) -> list[dict]:
        """7 outer writer와 각 학습 writer의 inner 3-fold를 실제 ID로 저장한다."""
        result = []
        all_ids = set(self.corpus.records)
        for outer in self.outer_writers:
            held = {key for key in self.corpus.original_ids if self.corpus.records[key]['writer'] == outer}
            train = filtered_ids(self.corpus.records, all_ids, held)
            writers = sorted({self.corpus.records[key]['writer'] for key in train})
            inner = []
            for index in range(3):
                inner_writers = set(writers[index::3])
                validation = {key for key in train if self.corpus.records[key]['writer'] in inner_writers}
                fit = filtered_ids(self.corpus.records, set(train), validation)
                if not fit or not validation:
                    raise ValueError('empty nested training/held fold')
                inner.append(dict(index=index, fit_ids=fit, held_ids=sorted(validation), held_writers=sorted(inner_writers)))
            result.append(dict(outer_writer=outer, fit_ids=train, held_ids=sorted(held), inner=inner))
        return result

    def prepare(self, raw_pipeline=True) -> None:
        """입력·코드·체크포인트 SHA가 같은 비지도 캐시만 재사용한다."""
        key = canonical_json_sha256(dict(inputs=self.corpus.input_hashes, hwr=self.model_manifest['hwr_checkpoint_sha256'],
            preprocessing={p: sha256_file(str(ROOT / 'scripts' / p)) for p in (
                'accuracy_joint_runtime_10e.py', 'accuracy_upgrade_data_10e.py', 'character_tensor_v1.py',
                'stroke_grouping_v1.py', 'train_character_classifier_v1.py')}, input_mode=self.config['input_mode'],
            window=self.args.window, neighbors=self.args.neighbors))
        cache = self.args.cache.resolve() / key
        cache.mkdir(parents=True, exist_ok=True)
        feature_path = cache / 'hwr_features.pt'
        if feature_path.exists():
            expected = json.loads(feature_path.with_suffix('.sha.json').read_text(encoding='utf-8'))
            if expected['sha256'] != sha256_file(str(feature_path)):
                raise ValueError('HWR cache file SHA mismatch')
            payload = torch.load(feature_path, map_location='cpu', weights_only=False)
            if payload['cache_key'] != key:
                raise ValueError('HWR cache mismatch')
            embeddings, logits = payload['embeddings'], payload['logits']
        else:
            embeddings, logits = encode_rows(self.hwr, self.corpus.raw, self.config['input_mode'], self.device)
            torch.save(dict(cache_key=key, embeddings=embeddings, logits=logits), feature_path)
            write_json(feature_path.with_suffix('.sha.json'), dict(sha256=sha256_file(str(feature_path))))
        self.samples = build_samples(self.corpus, embeddings, logits, self.labels, self.args.top_k,
                                     self.args.features, self.args.geometry)
        write_json(self.output / 'cache_manifest.json', dict(cache_key=key, hwr_features=str(feature_path),
                                                          hwr_features_sha256=sha256_file(str(feature_path))))
        if not raw_pipeline:
            return
        raw_path = cache / 'raw_formula_features.pt'
        if raw_path.exists():
            expected = json.loads(raw_path.with_suffix('.sha.json').read_text(encoding='utf-8'))
            if expected['sha256'] != sha256_file(str(raw_path)):
                raise ValueError('raw cache file SHA mismatch')
            payload = torch.load(raw_path, map_location='cpu', weights_only=False)
            if payload['cache_key'] != key:
                raise ValueError('raw feature cache mismatch')
            self.prepared = payload['formulas']
        else:
            self.prepared = {}
            for index, (formula_id, formula) in enumerate(sorted(self.corpus.formulas.items())):
                self.prepared[formula_id] = prepare_formula(dict(formula_id=formula_id, strokes=formula['strokes']),
                    self.hwr, self.labels, self.device, self.config['input_mode'], self.args.window, self.args.neighbors)
                if index % 20 == 0:
                    event('raw_cache', completed=index + 1, total=len(self.corpus.formulas))
            torch.save(dict(cache_key=key, formulas=self.prepared), raw_path)
            write_json(raw_path.with_suffix('.sha.json'), dict(sha256=sha256_file(str(raw_path))))

    def fit_group(self, ids: list[str], name: str):
        """동일 제외 경계 안에서 grouping을 학습하고 학습 ID를 등록한다."""
        formula_ids = sorted({self.corpus.records[key]['formula_id'] for key in ids})
        grouping = fit_grouping(self.prepared, {key: row['truth_groups'] for key, row in self.corpus.formulas.items()}, formula_ids, self.config['seed'])
        path = self.output / 'models' / (name + '_grouping.joblib')
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(grouping, path)
        root = self.registry.add('grouping', path, ids, [self.hwr_node])
        return grouping, root

    def assert_held(self, root: str, held_ids: list[str]) -> None:
        """현재 평가에 들어가기 전에 모든 감독학습 조상을 검사한다."""
        self.registry.validate(root, {self.corpus.records[k]['writer'] for k in held_ids},
                               {self.corpus.records[k]['formula_key'] for k in held_ids})

    def finish(self, report: dict) -> None:
        """실제 결과와 계보를 검증한 후 단계 완료 또는 명시적 차단을 기록한다."""
        report.update(elapsed_seconds=time.perf_counter() - self.started,
            gpu_peak_bytes=torch.cuda.max_memory_allocated() if self.device.type == 'cuda' else 0,
            data_role='consumed_development', product_activated=False, uploaded=False)
        write_json(self.output / 'lineage_manifest.json', self.registry.payload())
        write_json(self.output / 'evaluation.json', report)
        path = self.output / 'experiment.json'
        experiment = json.loads(path.read_text(encoding='utf-8'))
        experiment.update(status=report.get('status', 'completed'), exit_code=0, elapsed_seconds=report['elapsed_seconds'])
        write_json(path, experiment)
        event('completed', output=str(self.output), **report)

    def audit(self) -> None:
        """기존 예측 재현과 새 입력·분할·자료 범위를 기록한다."""
        legacy = _json_lines(ROOT / 'artifacts/auxiliary_nested_distill_guard_20260902_r1/nested_guarded_predictions.jsonl.gz')
        replay = {name: glyph_metrics(legacy, key) for name, key in
                  [('baseline', 'baseline_token'), ('student', 'adapter_token'), ('guarded', 'nested_guarded_token')]}
        expected = dict(baseline=(317, 49), student=(347, 64), guarded=(321, 51))
        for name, (glyphs, formulas) in expected.items():
            if (replay[name]['glyph_correct'], replay[name]['glyph_formula_correct']) != (glyphs, formulas):
                raise ValueError(f'legacy prediction replay mismatch: {name}')
        self.prepare(raw_pipeline=False)
        rows = [dict(record_id=k, formula_id=s['formula_id'], writer_group=s['writer_group'], label=s['label'],
                     candidates=s['candidates'], baseline_token=s['candidates'][0], adapter_token=s['candidates'][0])
                for k, s in self.samples.items() if k in self.corpus.original_ids]
        write_rows(self.output / 'baseline_glyph_predictions.jsonl.gz', rows)
        self.finish(dict(status='completed', training_performed=False, legacy_replay=replay,
            external_only_hwr=glyph_metrics(rows), counts=dict(records=len(self.corpus.raw), formulas=len(self.corpus.formulas),
            relation_formulas=sum(bool(f['target_relations']) for f in self.corpus.formulas.values())),
            blocked_stages={'relation_head': 'only two relation-annotated formulas', 'sealed_acceptance': 'no fresh writer/device dataset'}))

    def baseline(self) -> None:
        """fold별 grouping과 외부 기반 HWR의 실제 원본 획 기준선을 측정한다."""
        self.prepare()
        rows, folds = [], []
        for index, split in enumerate(self.splits):
            grouping, root = self.fit_group(split['fit_ids'], f'outer{index}')
            self.assert_held(root, split['held_ids'])
            ids = {self.corpus.records[k]['formula_id'] for k in split['held_ids']}
            predictions = evaluate_raw(self.corpus, self.prepared, grouping, ids, self.labels, self.device,
                k=self.args.top_k, feature_version=self.args.features, joint=self.args.stage == 'joint', structure_weight=self.args.structure_weight,
                geometry_mode=self.args.geometry)
            for row in predictions:
                row['lineage_root_id'] = root
            rows.extend(predictions)
            fold = dict(outer_writer=split['outer_writer'], **formula_metrics(predictions))
            folds.append(fold); event('outer_complete', index=index, **fold)
        if {r['formula_id'] for r in rows} != {self.corpus.records[k]['formula_id'] for k in self.corpus.original_ids}:
            raise ValueError('raw prediction coverage differs')
        write_rows(self.output / 'formula_predictions.jsonl.gz', rows)
        self.finish(dict(status='completed', training_performed=True, model='external_hwr_plus_fold_grouping',
                         aggregate=formula_metrics(rows), folds=folds))

    def fit_selected_student(self, ids: list[str], name: str):
        """평가 writer와 무관한 내부 writer로 epoch를 고른 뒤 허용 fit 전체를 재학습한다."""
        from accuracy_student_training_10e import fit_student
        writers = sorted({self.corpus.records[key]['writer'] for key in ids})
        if len(writers) < 3:
            raise ValueError('not enough writers for independent epoch selection')
        validation = [key for key in ids if self.corpus.records[key]['writer'] == writers[-1]]
        fit = filtered_ids(self.corpus.records, set(ids), set(validation))
        grouping, grouping_root = self.fit_group(fit, name + '_selection')
        validation_formulas = {self.corpus.records[key]['formula_id'] for key in validation}
        trial_samples, teacher_roots = (self.samples, []) if self.teacher is None else self.teacher.crossfit(fit, self.samples)

        def validation_score(model):
            """분리한 내부 writer의 원본 획 Exact만 epoch 선택에 사용한다."""
            predictions = evaluate_raw(self.corpus, self.prepared, grouping, validation_formulas, self.labels,
                self.device, model=model, k=self.args.top_k, feature_version=self.args.features, geometry_mode=self.args.geometry)
            score = formula_metrics(predictions)
            return score['e2e_correct'], score['structure_exact'] or 0.

        training_config = {**self.config, 'token_count': len(self.labels), 'kd_weight': 0. if self.teacher is None else 1.,
                           'loss_mode': self.args.loss_mode}
        trial, selection = fit_student(trial_samples, fit, training_config, self.device, self.args.architecture,
                                       validation_ids=validation, validation_score=validation_score)
        selection_path = self.output / 'models' / (name + '_selection.pt')
        torch.save(dict(state_dict={key: value.detach().cpu() for key, value in trial.state_dict().items()}, report=selection), selection_path)
        selector_root = self.registry.add('student_epoch_selection', selection_path, fit,
            [self.hwr_node, grouping_root] + teacher_roots, selection_ids=validation)
        del trial
        epochs = int(selection['best_epoch'])
        final_samples, final_teacher_roots = (self.samples, []) if self.teacher is None else self.teacher.crossfit(ids, self.samples)
        final, report = fit_student(final_samples, ids, {**training_config, 'epochs': epochs}, self.device, self.args.architecture)
        path = self.output / 'models' / (name + '_student.pt')
        torch.save(dict(state_dict={key: value.detach().cpu() for key, value in final.state_dict().items()},
            architecture=self.args.architecture, numeric_size=len(feature_names(self.args.features)),
            labels=self.labels, feature_version=self.args.features, feature_names=list(feature_names(self.args.features)),
            hwr_checkpoint_sha256=self.model_manifest['hwr_checkpoint_sha256'], input_mode=self.config['input_mode'],
            report=report, selected_epochs=epochs), path)
        root = self.registry.add('refit_student', path, ids, [self.hwr_node, selector_root] + final_teacher_roots)
        return final, root, dict(selection=selection, refit=report, checkpoint=str(path), selected_epochs=epochs)

    def student(self) -> None:
        """outer/inner 분리 모델을 실제 학습하고 guarded/raw 종단간 결과를 생성한다."""
        from accuracy_student_training_10e import predict_student
        from accuracy_teacher_targets_10e import TeacherTargets
        self.prepare()
        names = [] if self.args.teacher == 'none' else (list(self.config['teachers']) if self.args.teacher == 'ensemble' else [self.args.teacher])
        self.teacher = TeacherTargets(self, names) if names else None
        all_glyphs, all_formulas, all_baseline, fold_reports, all_guarded = [], [], [], [], []
        inner_output = []
        for outer_index, split in enumerate(self.splits):
            event('outer_start', index=outer_index, writer=split['outer_writer'], architecture=self.args.architecture, teacher=self.args.teacher)
            calibration, inner_roots = [], []
            for inner in split['inner']:
                model, root, report = self.fit_selected_student(inner['fit_ids'], f'o{outer_index}_i{inner["index"]}')
                self.assert_held(root, split['held_ids'] + inner['held_ids'])
                rows = predict_student(model, self.samples, inner['held_ids'], self.device)
                for row in rows:
                    row['prediction_provenance'] = dict(root_id=root, outer_writer=split['outer_writer'],
                        excluded_writers=[split['outer_writer']] + inner['held_writers'], candidate_sha256=canonical_json_sha256(row['candidates']))
                calibration.extend(rows); inner_output.extend(rows); inner_roots.append(root)
                event('inner_complete', outer=outer_index, inner=inner['index'], epochs=report['selected_epochs'], **glyph_metrics(rows))
                del model
            guard, guard_report = fit_guard(calibration)
            guard_path = self.output / 'models' / f'outer{outer_index}_guard.joblib'
            joblib.dump(guard, guard_path)
            guard_root = self.registry.add('guard', guard_path, split['fit_ids'], inner_roots)
            self.assert_held(guard_root, split['held_ids'])
            model, student_root, training = self.fit_selected_student(split['fit_ids'], f'outer{outer_index}')
            self.assert_held(student_root, split['held_ids'])
            glyph_rows = predict_student(model, self.samples, split['held_ids'], self.device)
            glyph_rows = apply_guard(guard, glyph_rows)
            for row in glyph_rows:
                row['prediction_provenance'] = dict(root_id=student_root, guard_root_id=guard_root,
                    excluded_writers=[split['outer_writer']], candidate_sha256=canonical_json_sha256(row['candidates']))
            all_glyphs.extend(glyph_rows)
            grouping, grouping_root = self.fit_group(split['fit_ids'], f'outer{outer_index}_raw')
            self.assert_held(grouping_root, split['held_ids'])
            formula_ids = {self.corpus.records[key]['formula_id'] for key in split['held_ids']}
            baseline = evaluate_raw(self.corpus, self.prepared, grouping, formula_ids, self.labels, self.device,
                                   k=self.args.top_k, feature_version=self.args.features, geometry_mode=self.args.geometry)
            raw = evaluate_raw(self.corpus, self.prepared, grouping, formula_ids, self.labels, self.device,
                               model=model, k=self.args.top_k, feature_version=self.args.features, joint=self.args.joint,
                               geometry_mode=self.args.geometry, structure_weight=self.args.structure_weight)
            guarded = evaluate_raw(self.corpus, self.prepared, grouping, formula_ids, self.labels, self.device,
                model=model, k=self.args.top_k, feature_version=self.args.features, joint=self.args.joint, guard=guard,
                geometry_mode=self.args.geometry, structure_weight=self.args.structure_weight)
            for row in raw:
                row['student_lineage_root_id'] = student_root; row['grouping_lineage_root_id'] = grouping_root
            for row in guarded:
                row.update(student_lineage_root_id=student_root, grouping_lineage_root_id=grouping_root, guard_lineage_root_id=guard_root)
            for row in baseline:
                row['grouping_lineage_root_id'] = grouping_root
            all_formulas.extend(raw); all_baseline.extend(baseline)
            all_guarded.extend(guarded)
            fold = dict(outer_writer=split['outer_writer'], glyph=glyph_metrics(glyph_rows),
                guarded_glyph=glyph_metrics(glyph_rows, 'guarded_token'), baseline=formula_metrics(baseline),
                raw=formula_metrics(raw), guarded=formula_metrics(guarded), training=training, guard=guard_report)
            fold_reports.append(fold)
            event('outer_complete', index=outer_index, glyph=fold['glyph'], e2e=fold['raw'])
            write_rows(self.output / 'partial_glyph_predictions.jsonl.gz', all_glyphs)
            del model
        if {row['record_id'] for row in all_glyphs} != self.corpus.original_ids or len(all_glyphs) != len(self.corpus.original_ids):
            raise ValueError('incomplete or duplicated outer prediction coverage')
        registry_path = self.output / 'lineage_manifest.json'
        write_json(registry_path, self.registry.payload())
        registry_sha = sha256_file(str(registry_path))
        for row in all_glyphs + inner_output:
            row['prediction_provenance'].update(registry_path=str(registry_path), registry_sha256=registry_sha)
        write_rows(self.output / 'glyph_predictions.jsonl.gz', all_glyphs)
        write_rows(self.output / 'inner_predictions.jsonl.gz', inner_output)
        write_rows(self.output / 'formula_predictions.jsonl.gz', all_formulas)
        write_rows(self.output / 'guarded_formula_predictions.jsonl.gz', all_guarded)
        write_rows(self.output / 'baseline_formula_predictions.jsonl.gz', all_baseline)
        self.finish(dict(status='completed', training_performed=True, architecture=self.args.architecture, teacher=self.args.teacher,
            baseline_glyph=glyph_metrics(all_glyphs, 'baseline_token'), raw_glyph=glyph_metrics(all_glyphs),
            guarded_glyph=glyph_metrics(all_glyphs, 'guarded_token'), baseline_e2e=formula_metrics(all_baseline),
            raw_e2e=formula_metrics(all_formulas), guarded_e2e=formula_metrics(all_guarded), folds=fold_reports,
            blocked_data={'relation_head': '2 relation formulas', 'product_promotion': 'no sealed writer/device acceptance'}))


def main() -> int:
    """새 출력 경로와 명시적 구성을 요구하고 실패도 실험 원장에 남긴다."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache', type=Path, default=ROOT / 'artifacts/accuracy_upgrade_20260911_cache')
    parser.add_argument('--stage', choices=['audit', 'baseline', 'student', 'joint'], required=True)
    parser.add_argument('--architecture', choices=['mlp', 'context'], default='context')
    parser.add_argument('--teacher', choices=['none', 'texteller', 'unimernet_tiny', 'trocr_small', 'ensemble'], default='none')
    parser.add_argument('--loss-mode', choices=['standard', 'legacy', 'legacy_kl'], default='standard')
    parser.add_argument('--joint', action='store_true')
    parser.add_argument('--features', choices=['legacy21', 'formula28'], default='formula28')
    parser.add_argument('--geometry', choices=['legacy', 'restored'], default='restored')
    parser.add_argument('--top-k', type=int, choices=[5, 10, 20], default=5)
    parser.add_argument('--window', type=int, choices=[6, 8], default=6)
    parser.add_argument('--neighbors', type=int, choices=[4, 8], default=4)
    parser.add_argument('--structure-weight', type=float, choices=[.5, 1., 2.], default=1.)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    experiment = None
    try:
        experiment = Experiment(args, config)
        if args.stage == 'audit':
            experiment.audit()
        elif args.stage in ('baseline', 'joint'):
            experiment.baseline()
        else:
            experiment.student()
    except Exception as error:
        if experiment is not None:
            write_json(experiment.output / 'failure.json', dict(status='failed', error_type=type(error).__name__, error=str(error)))
            path = experiment.output / 'experiment.json'
            ledger = json.loads(path.read_text(encoding='utf-8')); ledger.update(status='failed', exit_code=1)
            write_json(path, ledger)
        raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
