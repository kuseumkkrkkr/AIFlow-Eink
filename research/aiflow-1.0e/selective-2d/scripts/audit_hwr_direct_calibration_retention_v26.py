"""직접 CE 재학습의 실패 클래스와 기존 보정 head 지식의 소실을 학습 없이 진단한다."""
from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

import run_hwr_direct_augmented_main_v25 as direct
import audit_hwr_owned_formula_transfer_v21 as owned
from audit_hwr_encoder_head_swap_v22 import records_from_scores
from run_hwr_affine_distillation_experiment_v1 import _load_teacher
from verify_hwr_broad_head_only_v23 import hit_metrics

OUT=direct.broad.previous.ROOT / "artifacts/hwr_direct_calibration_retention_20261005_v26"


def taxonomy(before,after):
    """원본이 맞힌 문자·수식의 퇴행을 실제 GT별로 집계하며 등호 단독 원인을 구분한다."""
    lost=Counter();wins=Counter();pairs=Counter();formula_errors=[]
    for a,b in zip(before,after,strict=True):
        assert a["sample_id"]==b["sample_id"] and a["truth_tokens"]==b["truth_tokens"]
        for t,x,y in zip(a["truth_tokens"],a["top1_tokens"],b["top1_tokens"],strict=True):
            if x==t and y!=t:
                lost[t]+=1;pairs[t,y]+=1
            if x!=t and y==t:
                wins[t]+=1
        if a["top1_tokens"]==a["truth_tokens"] and b["top1_tokens"]!=b["truth_tokens"]:
            formula_errors.append([t for t,y in zip(b["truth_tokens"],b["top1_tokens"],strict=True) if t!=y])
    assert sum(lost.values())-sum(wins.values())==sum(sum(t==p for t,p in zip(r["truth_tokens"],r["top1_tokens"])) for r in before)-sum(sum(t==p for t,p in zip(r["truth_tokens"],r["top1_tokens"])) for r in after)
    return dict(lost_by_truth=dict(lost),wins_by_truth=dict(wins),lost_pairs=[dict(truth=t,predicted=p,count=n) for (t,p),n in pairs.most_common()],
        lost_formulas=len(formula_errors),lost_formulas_with_equal=sum("=" in e for e in formula_errors),
        lost_formulas_only_equal=sum(set(e)=={"="} for e in formula_errors))


def encode(model,x,batch):
    """head 교체와 무관한 실제 encoder/pooling 출력만 한 번 계산한다."""
    values=[]
    with torch.inference_mode():
        for start in range(0,len(x),batch):
            values.append(model.encode(torch.from_numpy(x[start:start+batch].copy())).numpy())
    return np.concatenate(values)


def head_scores(head,h,batch):
    """동일 encoder 출력에서 교차 head를 계산하고 독립 double affine 수치와 대조한다."""
    z=[]
    with torch.inference_mode():
        for start in range(0,len(h),batch):
            z.append(head(torch.from_numpy(h[start:start+batch].copy())).numpy())
    z=np.concatenate(z)
    expected=h.astype(np.float64) @ head.weight.detach().numpy().astype(np.float64).T + head.bias.detach().numpy().astype(np.float64)
    assert float(np.abs(z-expected).max())<1e-4 and np.isfinite(z).all()
    assert np.array_equal(np.argsort(-z,axis=1,kind="stable")[:,:5],np.argsort(-expected,axis=1,kind="stable")[:,:5])
    return z


def main():
    """실패 taxonomy와 전체/보정행 head 복원 가설을 소비된 같은 입력에서 검증한다."""
    if OUT.exists():
        raise FileExistsError("refusing calibration diagnostic overwrite")
    parent=json.loads((direct.OUT / "independent_verification.json").read_text(encoding="utf-8"));assert parent["status"]=="pass"
    calibration=direct.broad.previous.CHECKPOINT.parent / "calibration_report.json"
    report=json.loads(calibration.read_text(encoding="utf-8"))
    punct=report["calibration_train"]["punctuation"]["target_output_rows"]
    protected=sorted(set(punct+report["calibration_train"]["other_observed_symbols"]["target_output_rows"]))
    paths=dict(canonical=direct.broad.previous.CHECKPOINT,**{arm:direct.OUT / arm / "main_trained.pt" for arm in direct.ARMS})
    models={};states={};hashes={}
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    for name,path in paths.items():
        model,labels,_=_load_teacher(path,torch.device("cpu"));models[name]=model
        states[name]=copy.deepcopy(model.math_head.state_dict());hashes[name]=direct.broad.previous._sha(path)
    assert len(protected)>=len(punct) and all(0<=i<len(labels) for i in protected)
    _,samples,rows=owned.load_source();owner_x=np.load(owned.OUT / "inputs.npy",allow_pickle=False)
    OUT.mkdir()
    direct.broad.previous._write(OUT / "frozen_plan.json",dict(schema="aiflow-calibration-retention-plan/v26",
        script_sha256=direct.broad.previous._sha(Path(__file__)),parent_certificate_sha256=direct.broad.previous._sha(direct.OUT / "independent_verification.json"),
        calibration_report_sha256=direct.broad.previous._sha(calibration),checkpoint_sha256=hashes,
        historical_calibration_row_ids=protected,historical_calibration_labels=[labels[i] for i in protected],
        historical_calibration_row_count=len(protected),
        variants=["as_trained","canonical_head","restore_punctuation_rows","restore_all_calibrated_rows","canonical_encoder_trained_head"],
        optimizer_steps=0,posthoc_consumed_diagnostic=True,new_human_acceptance=False,crohme_rows=0,product_adopted=False))
    baseline=json.loads((direct.OUT / "canonical_owned_records.json").read_text(encoding="utf-8"))
    old_h=encode(models["canonical"],owner_x,owned.BATCH)
    assert np.array_equal(old_h,np.load(owned.OUT / "canonical_embeddings.npy",allow_pickle=False))
    metrics={};taxonomies={}
    for arm in direct.ARMS:
        model=models[arm];h=encode(model,owner_x,owned.BATCH);original=copy.deepcopy(model.state_dict())
        after=json.loads((direct.OUT / f"{arm}_owned_records.json").read_text(encoding="utf-8"))
        taxonomies[arm]=taxonomy(baseline,after);metrics[arm]={}
        for variant in ("as_trained","canonical_head","restore_punctuation_rows","restore_all_calibrated_rows","canonical_encoder_trained_head"):
            state=copy.deepcopy(states[arm]);current=h
            if variant=="canonical_head":
                state=states["canonical"]
            elif variant in ("restore_punctuation_rows","restore_all_calibrated_rows"):
                selected=punct if variant=="restore_punctuation_rows" else protected
                for key in ("weight","bias"):
                    state[key][selected]=states["canonical"][key][selected]
                keep=[i for i in range(372) if i not in selected]
                assert all(torch.equal(state[k][keep],states[arm][k][keep]) for k in ("weight","bias"))
            elif variant=="canonical_encoder_trained_head":
                current=old_h
            model.math_head.load_state_dict(state,strict=True)
            z=head_scores(model.math_head,current,owned.BATCH)
            if variant=="as_trained":
                assert np.array_equal(z,np.load(direct.OUT / f"{arm}_owned_logits.npy",allow_pickle=False))
            records=records_from_scores(z,labels,samples,rows)
            metrics[arm][variant]={cohort:owned.metrics(records,cohort) for cohort in ("all","legacy_96","codex_reviewed_53")}
            np.save(OUT / f"{arm}__{variant}_logits.npy",z,allow_pickle=False)
            direct.broad.previous._write(OUT / f"{arm}__{variant}_records.json",records)
            print(json.dumps(dict(event="calibration_restore_probe",arm=arm,variant=variant,metrics=metrics[arm][variant]["all"])),flush=True)
        model.load_state_dict(original,strict=True)
        assert all(torch.equal(v,model.state_dict()[k]) for k,v in original.items())
    assert all(direct.broad.previous._sha(path)==hashes[name] for name,path in paths.items())
    result=dict(schema="aiflow-calibration-retention-result/v26",status="completed_verified_counterfactual_not_adoption",
        historical_calibration_rows=len(protected),taxonomy=taxonomies,metrics=metrics,checkpoint_files_unchanged=True,model_memory_restored=True,
        all_diagonal_logits_bit_exact_with_v25=True,all_variant_top5_match_independent_double_affine=True,
        optimizer_steps=0,crohme_rows=0,new_human_acceptance=False,product_adopted=False,
        limit="Weight-only component restoration on consumed oracle groups is a posthoc diagnostic, not a deployed fix or proof of isolated causal training failure. No recovered variant is chosen as a release model.")
    direct.broad.previous._write(OUT / "retention_result.json",result)
    print(json.dumps(dict(event="retention_completed",formula_matrix={a:{v:m["all"]["formula_top1_exact"] for v,m in ms.items()} for a,ms in metrics.items()})),flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
