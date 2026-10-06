"""고정 예측의 기호·구조는 유지하고 출력 제어어 경계만 재생한다."""
import argparse
import copy
import json
from pathlib import Path
from accuracy_latex_serializer_10e import serialize_selected_graph, VERSION
from accuracy_upgrade_data_10e import load_corpus, write_json, write_rows
from accuracy_upgrade_contract_v1 import sha256_file, normalize_latex
from accuracy_evaluation_10e import formula_metrics, writer_bootstrap
from online_candidate_features_10e import _json_lines
from accuracy_lineage_10e import LineageRegistry
from verify_accuracy_experiment_10e import paired_changes


def replay(rows, corpus, serializer_root):
    """정답은 serializer에 전달하지 않고 예측 토큰·관계와 원본 x만 제공한다."""
    results=[]
    for original in rows:
        row=copy.deepcopy(original)
        if not row.get('failure_reason'):
            source=corpus.formulas[row['formula_id']]['strokes']
            evidence=[]; labels={}
            for group,token in zip(row['groups'],row['tokens'],strict=True):
                key=row['formula_id']+':'+','.join(map(str,sorted(group)))
                evidence.append(dict(record_id=key,geometry=dict(left=min(float(p['x']) for i in group for p in source[i]['points']))))
                labels[key]=token
            try:
                row['raw_latex']=serialize_selected_graph(evidence,labels,row['structure'])
            except ValueError as error:
                row.update(raw_latex=None,failure_reason=str(error),review_requested=True)
        row.update(normalized_latex=normalize_latex(row.get('raw_latex') or ''),serializer_lineage_root_id=serializer_root)
        results.append(row)
    return results


def main():
    """원본 보고서를 보존하고 saved-output 재생임을 명시한 파생 실험을 만든다."""
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists() or a.output.resolve().drive.upper()!='D:':
        raise ValueError('new D: output required')
    ledger=json.loads((a.experiment/'experiment.json').read_text(encoding='utf-8'))
    if ledger['status']!='completed':
        raise ValueError('source experiment incomplete')
    corpus=load_corpus(ledger['configuration'])
    registry=LineageRegistry.load(a.experiment/'lineage_manifest.json')
    root=registry.add('deterministic_latex_serialization',Path(__file__).with_name('accuracy_latex_serializer_10e.py'),
        version=VERSION, source_predictions_sha256=sha256_file(str(a.experiment/'formula_predictions.jsonl.gz')))
    before=_json_lines(a.experiment/'formula_predictions.jsonl.gz');after=replay(before,corpus,root)
    base_path=a.experiment/'baseline_formula_predictions.jsonl.gz'
    baseline=replay(_json_lines(base_path),corpus,root) if base_path.exists() else None
    a.output.mkdir(parents=True)
    write_rows(a.output/'formula_predictions.jsonl.gz',after)
    if baseline is not None:
        write_rows(a.output/'baseline_formula_predictions.jsonl.gz',baseline)
    if (a.experiment/'glyph_predictions.jsonl.gz').exists():
        write_rows(a.output/'glyph_predictions.jsonl.gz',_json_lines(a.experiment/'glyph_predictions.jsonl.gz'))
    write_json(a.output/'lineage_manifest.json',registry.payload())
    write_json(a.output/'split_manifest.json',json.loads((a.experiment/'split_manifest.json').read_text(encoding='utf-8')))
    model=json.loads((a.experiment/'model_manifest.json').read_text(encoding='utf-8'))
    model.update(serializer_version=VERSION,serializer_sha256=sha256_file(str(Path(__file__).with_name('accuracy_latex_serializer_10e.py'))))
    write_json(a.output/'model_manifest.json',model)
    report=dict(status='completed',source_experiment=str(a.experiment.resolve()),training_performed=False,
        inference_kind='saved predicted glyphs and structure replay; no neural inference',
        before_e2e=formula_metrics(before),raw_e2e=formula_metrics(after),serialization_changes=paired_changes(before,after),
        data_role='consumed_development',product_activated=False,uploaded=False)
    if baseline is not None:
        report.update(baseline_e2e=formula_metrics(baseline),paired=paired_changes(baseline,after),
                      development_writer_bootstrap=writer_bootstrap(baseline,after))
    write_json(a.output/'evaluation.json',report)
    ledger.update(stage='serialization_replay',status='completed',exit_code=0,parent_experiment=str(a.experiment.resolve()))
    write_json(a.output/'experiment.json',ledger)
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
