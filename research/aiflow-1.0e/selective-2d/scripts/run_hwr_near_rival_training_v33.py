"""직접 증강 CE를 유지하고 보조 gap 손실만 Top-5 근접 후보로 교체한다.

V29 원본 schedule과 feature 계수를 그대로 쓴다. 새 gap 계수는 TRAIN gradient로만
봉인하고, 검증된 trainer 파일은 수정하지 않는다.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import run_hwr_direct_margin_retention_v29 as algorithm

PARENT=algorithm.OUT
direct=algorithm.direct
OUT=direct.broad.previous.ROOT / "artifacts/hwr_near_rival_training_20261006_v33"
BASE=PARENT / "margin_retained_main/main_trained.pt"


def near_mask(z,teacher_z,truth):
    """동점은 class index 순서로 처리하며 정답을 뺀 두 Top-5의 합집합을 선택한다."""
    import torch
    assert z.shape==teacher_z.shape and z.shape[1]==372 and not teacher_z.requires_grad
    correct=teacher_z.argmax(1)==truth
    near=torch.zeros_like(z,dtype=torch.bool)
    near.scatter_(1,torch.argsort(teacher_z,dim=1,descending=True,stable=True)[:,:5],True)
    near.scatter_(1,torch.argsort(z.detach(),dim=1,descending=True,stable=True)[:,:5],True)
    near.scatter_(1,truth[:,None],False);near &= correct[:,None]
    return near,correct


def near_gap_loss(z,teacher_z,truth):
    """Teacher-correct 근접 pair에서만 정답 logit gap 감소를 제곱 hinge로 벌한다."""
    import torch
    near,correct=near_mask(z,teacher_z,truth)
    deficit=torch.relu((teacher_z.gather(1,truth[:,None])-teacher_z)-(z.gather(1,truth[:,None])-z))*near
    loss=deficit.square().sum()/near.sum().clamp_min(1)
    return loss,correct,int((deficit.detach()>0).sum())


def selftest():
    """정답 방향·먼 후보 무시·teacher-wrong 제외·shift·빈 mask·동점을 검사한다."""
    import torch
    teacher=torch.full((2,372),-10.)
    teacher[0,:5]=torch.tensor([5.,4.,3.,2.,1.]);teacher[1,1]=5.
    truth=torch.tensor([0,0]);student=teacher.clone();student[0,0]=3.;student.requires_grad_(True)
    loss,correct,_=near_gap_loss(student,teacher,truth);g=torch.autograd.grad(loss,student)[0]
    assert correct.tolist()==[True,False] and g[0,0]<0 and (g[0,1:]>=0).all() and (g[1]==0).all()
    shifted,_,_=near_gap_loss(student+13.,teacher+7.,truth);assert torch.allclose(loss,shifted,atol=1e-6)
    unchanged=teacher.clone();unchanged[0,300]=-9.;unchanged.requires_grad_(True)
    far,_,_=near_gap_loss(unchanged,teacher,truth);assert float(far)==0. and (torch.autograd.grad(far,unchanged)[0]==0).all()
    zero,_,_=near_gap_loss(teacher.clone().requires_grad_(True),teacher,truth);assert float(zero)==0.
    empty,mask,_=near_gap_loss(student,teacher,truth+9);assert float(empty)==0. and not mask.any()
    tied=torch.zeros((1,372));tm,_=near_mask(tied,tied,torch.tensor([0]))
    assert tm.nonzero()[:,1].tolist()==[1,2,3,4]
    print(json.dumps(dict(event="near_gap_selftest",status="pass",correct_direction=True,far_changes_ignored=True,
        teacher_wrong_excluded=True,shift_invariant=True,empty_mask_finite=True,stable_ties=True)),flush=True)


def load():
    """기존 schedule·새 손실 코드·TRAIN 봉인 계수·source 모델 SHA를 확인한다."""
    sha=direct.broad.previous._sha
    p=json.loads((OUT / "frozen_plan.json").read_text(encoding="utf-8"))
    assert sha(Path(__file__))==p["code_sha256"] and sha(Path(algorithm.__file__))==p["trainer_sha256"]
    assert sha(PARENT / "frozen_plan.json")==p["parent_v29_plan_sha256"] and sha(BASE)==p["coefficient_baseline_sha256"]
    parent=json.loads((PARENT / "frozen_plan.json").read_text(encoding="utf-8"))
    _,a,s=algorithm.previous.load()
    assert all(p[k]==parent[k] for k in ("hard_ce_weight","feature_weight","lr","weight_decay","gradient_clip","steps","seed","batch_size"))
    assert p["same_v25_data_and_schedule"] and np.unique(s["shared_target_ids"]).size==371
    return p,a,s


def prepare():
    """동일한 고정 TRAIN 8 batch에서 CE/near-gap 전체 gradient 비로 계수를 봉인한다."""
    if OUT.exists():
        raise FileExistsError("refusing near-rival experiment overwrite")
    import torch
    from torch.nn import functional as F
    from run_hwr_affine_distillation_experiment_v1 import _load_teacher
    selftest();parent,a,s=algorithm.load()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    model,_,_=_load_teacher(BASE,torch.device("cpu"));model.eval();_,teacher,_=algorithm.previous.models()
    params=list(model.parameters());rows=[]
    for step in np.linspace(0,2399,8,dtype=int):
        x,y=direct.batch(a,s,int(step),"augmented_main");tx=torch.from_numpy(x);truth=torch.from_numpy(y)
        z=model.math_head(model.encode(tx));ce=F.cross_entropy(z,truth)
        with torch.no_grad():
            tz=teacher.math_head(teacher.encode(tx))
        gap,correct,active=near_gap_loss(z,tz,truth);nm,_=near_mask(z,tz,truth)
        cg=torch.autograd.grad(ce,params,retain_graph=True);gg=torch.autograd.grad(gap,params)
        cn=float(torch.sqrt(sum(g.square().sum() for g in cg)));gn=float(torch.sqrt(sum(g.square().sum() for g in gg)))
        assert np.isfinite(cn) and np.isfinite(gn) and gn>0
        rows.append(dict(schedule_step=int(step),ce=float(ce.detach()),near_gap=float(gap.detach()),near_pairs=int(nm.sum()),
            teacher_correct_rows=int(correct.sum()),active_near_pairs=active,ce_all57_gradient_l2=cn,near_all57_gradient_l2=gn,ratio=cn/gn))
    raw=float(np.median([r["ratio"] for r in rows]));weight=float(np.clip(raw,1e-3,100.))
    OUT.mkdir();plan=copy.deepcopy(parent)
    plan.update(schema="aiflow-near-rival-training-plan/v33",code_sha256=direct.broad.previous._sha(Path(__file__)),
        trainer_sha256=direct.broad.previous._sha(Path(algorithm.__file__)),parent_v29_plan_sha256=direct.broad.previous._sha(PARENT / "frozen_plan.json"),
        coefficient_baseline_sha256=direct.broad.previous._sha(BASE),gap_weight=weight,unclamped_gap_weight=raw,calibration_rows=rows,
        gap_rule="Teacher-correct union of canonical/student stable Top-5, GT excluded; squared decreased gap averaged over selected pairs",
        coefficient_rule="Median CE/near-gap all-57-gradient norm on eight fixed TRAIN batches, clamp [0.001,100]",
        same_v25_data_and_schedule=True,failed_v31_sampler_used=False,near_topk=5,coefficient_evaluation_inputs_used=0,
        objective="Hard inherited-label CE coefficient 1 + unchanged feature MSE + sealed near-rival gap loss; not all-371 averaged gap",
        trainer_reuse="Original V29 trainer file unchanged; process-local output/load/gap-function bindings replaced and restored",
        checkpoint_container="Legacy V29 shape/container with new near-gap coefficient; experiment objective identified by this V33 frozen plan",
        cloud_upload=False,product_adopted=False)
    direct.broad.previous._write(OUT / "frozen_plan.json",plan)
    print(json.dumps(dict(event="near_gap_weight_sealed",near_gap_weight=weight,unclamped=raw,feature_weight=plan["feature_weight"],
        calibration_batches=8,evaluation_inputs_used=0)),flush=True)
    return 0


def train():
    """V29 검증 loop에서 보조 gap 함수만 교체하고 과정값·4개 층·57개 gradient를 기록한다."""
    load();old_out,old_load,old_gap=algorithm.OUT,algorithm.load,algorithm.gap_loss
    try:
        algorithm.OUT=OUT;algorithm.load=load;algorithm.gap_loss=near_gap_loss
        return algorithm.train()
    finally:
        algorithm.OUT=old_out;algorithm.load=old_load;algorithm.gap_loss=old_gap


def evaluate():
    """모든 학습을 마친 뒤 기존 모델과 같은 소비된 진단 입력에서 비교한다."""
    if (OUT / "comparison_result.json").exists():
        raise FileExistsError("comparison exists")
    plan,_,_=load();folder=OUT / "margin_retained_main"
    done=json.loads((folder / "completed.json").read_text(encoding="utf-8"))
    assert done["steps"]==2400 and direct.broad.previous._sha(folder / "main_trained.pt")==done["checkpoint_sha256"]
    baseline=json.loads((PARENT / "comparison_result.json").read_text(encoding="utf-8"))
    cert=json.loads((PARENT / "independent_verification.json").read_text(encoding="utf-8"))
    assert cert["status"]=="pass" and cert["comparison_sha256"]==direct.broad.previous._sha(PARENT / "comparison_result.json")
    import torch
    import audit_hwr_owned_formula_transfer_v21 as owned
    from audit_hwr_encoder_head_swap_v22 import records_from_scores
    from run_hwr_affine_distillation_experiment_v1 import _load_teacher,_predict_logits
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    model,labels,_=_load_teacher(folder / "main_trained.pt",torch.device("cpu"))
    _,data,folds,_=direct.broad.previous.global_load(direct.broad.previous.OUTPUT);development={}
    for fold in (1,2):
        z=_predict_logits(model,np.array(data["features"][folds[fold]],copy=True),torch.device("cpu"),32)
        np.save(OUT / f"near_rival_main_fold{fold}_logits.npy",z,allow_pickle=False)
        development[str(fold)]=direct.broad.previous.hit_metrics(z,np.array(data["labels"][folds[fold]],copy=True),labels)
    _,samples,rows=owned.load_source();torch.set_num_threads(1)
    z=_predict_logits(model,np.load(owned.OUT / "inputs.npy",allow_pickle=False),torch.device("cpu"),owned.BATCH)
    records=records_from_scores(z,labels,samples,rows)
    np.save(OUT / "near_rival_main_owned_logits.npy",z,allow_pickle=False);direct.broad.previous._write(OUT / "near_rival_main_owned_records.json",records)
    metrics=copy.deepcopy(baseline["metrics"]);metrics["near_rival_main"]=dict(development=development,
        owned={c:owned.metrics(records,c) for c in ("all","legacy_96","codex_reviewed_53")})
    direct.broad.previous._write(OUT / "comparison_result.json",dict(schema="aiflow-near-rival-comparison/v33",
        status="completed_consumed_diagnostic_not_fresh_acceptance",baseline_comparison_sha256=cert["comparison_sha256"],metrics=metrics,
        data_and_schedule_unchanged=True,hard_ce_weight=1.,feature_weight=plan["feature_weight"],near_gap_weight=plan["gap_weight"],
        all_evaluation_cases_already_consumed=True,generated_labels_human_verified=False,owned_optimizer_rows=0,crohme_rows=0,
        product_adopted=False,canonical_unchanged=direct.broad.previous._sha(direct.broad.previous.CHECKPOINT)==plan["canonical_sha256"]))
    print(json.dumps(dict(event="near_rival_model_scored",metrics=metrics["near_rival_main"])),flush=True)
    return 0


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("mode",choices=("prepare","selftest","train","evaluate"));args=parser.parse_args()
    if args.mode=="selftest":
        selftest();raise SystemExit(0)
    raise SystemExit(dict(prepare=prepare,train=train,evaluate=evaluate)[args.mode]())
