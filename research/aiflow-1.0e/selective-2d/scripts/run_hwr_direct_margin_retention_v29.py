"""직접 증강 CE·표현 보존에 TRAIN 정답/경쟁 후보 gap 보존을 추가한다.

원본 teacher가 정답인 TRAIN 입력에만 gap 손실을 적용한다. CE는 모든 행에 가중치 1이다.
기존 실험을 수정하지 않고 같은 데이터·예산으로 cold main 전체를 학습한다.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np
import run_hwr_direct_feature_retention_v27 as previous

direct=previous.direct
OUT=direct.broad.previous.ROOT / "artifacts/hwr_direct_margin_retention_20261005_v29"
BASE=previous.OUT / "feature_retained_main/main_trained.pt"


def gap_loss(z,teacher_z,truth):
    """Teacher-correct 행에서만 371개 경쟁 class 대비 정답 gap의 감소를 제곱 hinge로 벌한다."""
    import torch
    assert z.shape==teacher_z.shape and z.shape[1]==372 and not teacher_z.requires_grad
    mask=teacher_z.argmax(1)==truth
    teacher_gap=teacher_z.gather(1,truth[:,None])-teacher_z
    student_gap=z.gather(1,truth[:,None])-z
    deficit=torch.relu(teacher_gap-student_gap)*mask[:,None]
    loss=deficit.square().sum()/(mask.sum().clamp_min(1)*(z.shape[1]-1))
    return loss,mask,int((deficit.detach()>0).sum())


def selftest():
    """정답 방향 gradient·teacher-wrong 제외·상수 logit 이동 불변성을 실제 수치로 검사한다."""
    import torch
    teacher=torch.full((2,372),-2.);teacher[0,0]=4.;teacher[1,1]=4.
    truth=torch.tensor([0,0]);student=teacher.clone();student[0,0]=1.;student.requires_grad_(True)
    loss,mask,_=gap_loss(student,teacher,truth)
    grad=torch.autograd.grad(loss,student)[0]
    assert mask.tolist()==[True,False] and grad[0,0]<0 and (grad[0,1:]>=0).all() and (grad[1]==0).all()
    moved,_,_=gap_loss(student+13.,teacher+7.,truth);assert torch.allclose(loss,moved,atol=1e-6)
    zero,_,_=gap_loss(teacher.clone().requires_grad_(True),teacher,truth);assert float(zero)==0.
    wrong=truth+3;empty,empty_mask,_=gap_loss(student,teacher,wrong)
    assert float(empty)==0. and not empty_mask.any() and (torch.autograd.grad(empty,student)[0]==0).all()
    print(json.dumps(dict(event="gap_loss_selftest",status="pass",correct_class_gradient_direction=True,
        teacher_wrong_rows_zero_gradient=True,logit_shift_invariant=True,empty_mask_finite=True)),flush=True)


def load():
    """새 코드·봉인 계수와 기존 승인 TRAIN schedule을 확인한다."""
    p=json.loads((OUT / "frozen_plan.json").read_text(encoding="utf-8"))
    assert direct.broad.previous._sha(Path(__file__))==p["code_sha256"]
    assert direct.broad.previous._sha(previous.OUT / "frozen_plan.json")==p["parent_plan_sha256"]
    assert direct.broad.previous._sha(BASE)==p["feature_baseline_sha256"]
    parent,a,s=previous.load()
    assert p["feature_weight"]==parent["feature_weight"] and p["canonical_sha256"]==parent["canonical_sha256"]
    return p,a,s


def prepare():
    """기존 TRAIN 8 batch에서 전체 main gradient 비로 gap 계수를 정하며 평가 입력은 읽지 않는다."""
    if OUT.exists():
        raise FileExistsError("refusing margin experiment overwrite")
    import torch
    from torch.nn import functional as F
    from run_hwr_affine_distillation_experiment_v1 import _load_teacher
    selftest();parent,a,s=previous.load()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    model,_,_=_load_teacher(BASE,torch.device("cpu"));model.eval()
    _,teacher,_=previous.models();params=list(model.parameters());rows=[]
    for step in np.linspace(0,direct.STEPS-1,8,dtype=int):
        x,y=direct.batch(a,s,int(step),"augmented_main");tx=torch.from_numpy(x);truth=torch.from_numpy(y)
        z=model.math_head(model.encode(tx));ce=F.cross_entropy(z,truth)
        with torch.no_grad():
            tz=teacher.math_head(teacher.encode(tx))
        gap,mask,active=gap_loss(z,tz,truth)
        cg=torch.autograd.grad(ce,params,retain_graph=True);gg=torch.autograd.grad(gap,params)
        cn=float(torch.sqrt(sum(g.square().sum() for g in cg)));gn=float(torch.sqrt(sum(g.square().sum() for g in gg)))
        assert np.isfinite(cn) and np.isfinite(gn) and gn>0
        rows.append(dict(schedule_step=int(step),ce=float(ce.detach()),gap_loss=float(gap.detach()),teacher_correct_rows=int(mask.sum()),
            active_competitor_gaps=active,ce_all57_gradient_l2=cn,gap_all57_gradient_l2=gn,ratio=cn/gn))
    raw=float(np.median([r["ratio"] for r in rows]));weight=float(np.clip(raw,1e-3,100.))
    OUT.mkdir()
    plan=dict(schema="aiflow-direct-margin-retention-plan/v29",code_sha256=direct.broad.previous._sha(Path(__file__)),
        parent_plan_sha256=direct.broad.previous._sha(previous.OUT / "frozen_plan.json"),feature_baseline_sha256=direct.broad.previous._sha(BASE),
        canonical_sha256=parent["canonical_sha256"],steps=2400,batch_size=64,original_rows=16,augmented_rows=48,
        same_v25_data_and_schedule=True,all_57_main_tensors_trainable=True,hard_ce_weight=1.,kl_weight=0.,
        feature_weight=parent["feature_weight"],gap_weight=weight,unclamped_gap_weight=raw,
        gap_rule="Mean squared hinge of decreased correct-class-versus-all-371-competitor logit gaps, canonical-teacher-correct TRAIN rows only",
        coefficient_rule="Median CE/gap all-57-gradient L2 ratio on fixed eight TRAIN batches, clamp [0.001,100]",
        calibration_rows=rows,coefficient_evaluation_inputs_used=0,
        objective="Hard CE weight 1 on every real/augmented inherited GT + fixed V27 feature MSE + fixed teacher-correct gap loss",
        cold_initialization="canonical main; no warm start or frozen layer",lr=1e-5,weight_decay=1e-4,gradient_clip=1.,seed=direct.SEED,threads=2,
        development_folds=[1,2],no_interim_evaluation=True,no_best_epoch_selection=True,teacher_requires_grad=False,
        generated_labels_human_verified=False,label_inheritance_assumption=True,owned_optimizer_rows=0,crohme_rows=0,
        all_evaluation_cases_already_consumed=True,product_adopted=False,collection_or_deployment_resumed=False)
    direct.broad.previous._write(OUT / "frozen_plan.json",plan)
    print(json.dumps(dict(event="gap_weight_sealed",gap_weight=weight,unclamped=raw,feature_weight=parent["feature_weight"],
        calibration_batches=8,evaluation_inputs_used=0)),flush=True)
    return 0


def train():
    """세 손실·4개 층·57개 gradient와 teacher-correct mask를 매 step 기록한다."""
    if direct.broad.previous.aug.base._guard_commit("direct_augmented_joint_gap_retention_v29") is None:
        return 78
    plan,a,s=load();folder=OUT / "margin_retained_main"
    if folder.exists():
        raise FileExistsError("run exists; inspect live/terminal state before restart")
    folder.mkdir();direct.broad.previous._write(folder / "run_started.json",dict(process_id=os.getpid(),status="started",
        plan_sha256=direct.broad.previous._sha(OUT / "frozen_plan.json")))
    os.environ["TRACKIO_DIR"]=str(folder / "trackio")
    for key in ("TRACKIO_WEBHOOK_URL","TRACKIO_SPACE_ID","TRACKIO_SERVER_URL"):
        os.environ.pop(key,None)
    import torch,trackio
    from torch.nn import functional as F
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.manual_seed(plan["seed"])
    model,teacher,labels=previous.models();optimizer=torch.optim.AdamW(model.parameters(),lr=plan["lr"],weight_decay=plan["weight_decay"])
    layers={};handles=[]
    def capture(name):
        """실제 학습 모델 층 출력만 확인한다."""
        def hook(module,inputs,value):
            value=value.detach();assert torch.isfinite(value).all()
            layers[name]=dict(mean=float(value.mean()),std=float(value.std()))
        return hook
    for name,module in model.named_modules():
        if isinstance(module,torch.nn.TransformerEncoderLayer):
            handles.append(module.register_forward_hook(capture(name)))
    trackio.init(project="aiflow-direct-margin-retention-v29",name="margin_retained_main",space_id=None,embed=False,
        auto_log_cpu=False,auto_log_gpu=False,config=dict(steps=2400,hard_ce_weight=1.,feature_weight=plan["feature_weight"],gap_weight=plan["gap_weight"]))
    updates=0
    try:
        with (folder / "training_microscope.jsonl").open("x",encoding="utf-8") as stream:
            for step in range(direct.STEPS):
                x,y=direct.batch(a,s,step,"augmented_main");tx=torch.from_numpy(x);truth=torch.from_numpy(y)
                optimizer.zero_grad(set_to_none=True);h=model.encode(tx);z=model.math_head(h)
                with torch.no_grad():
                    target=teacher.encode(tx);tz=teacher.math_head(target)
                assert not target.requires_grad and not tz.requires_grad
                ce=F.cross_entropy(z,truth);feature=F.mse_loss(h,target);gap,mask,active=gap_loss(z,tz,truth)
                real=F.cross_entropy(z[:16],truth[:16]);aug=F.cross_entropy(z[16:],truth[16:])
                loss=ce+plan["feature_weight"]*feature+plan["gap_weight"]*gap;assert torch.isfinite(loss)
                if step==0:
                    gz=torch.autograd.grad(ce,z,retain_graph=True)[0]
                    expected=(z.detach().softmax(-1)-F.one_hot(truth,372))/64
                    assert torch.allclose(gz,expected,atol=1e-7) and float(feature.detach())<1e-10 and float(gap.detach())<1e-10
                    print(json.dumps(dict(event="joint_gap_selftest",hard_ce_full_weight=True,teacher_detached=True,
                        initial_feature_mse=float(feature.detach()),initial_gap_loss=float(gap.detach()))),flush=True)
                loss.backward()
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
                assert all(p.grad is None and not p.requires_grad for p in teacher.parameters())
                gradients={n:float(p.grad.norm()) for n,p in model.named_parameters()}
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm)
                optimizer.step();updates+=1
                numeric=dict(loss=float(loss.detach()),hard_ce=float(ce.detach()),feature_mse=float(feature.detach()),gap_loss=float(gap.detach()),
                    weighted_feature=float((plan["feature_weight"]*feature).detach()),weighted_gap=float((plan["gap_weight"]*gap).detach()),
                    feature_weight=plan["feature_weight"],gap_weight=plan["gap_weight"],hard_ce_weight=1.,kl_weight=0.,
                    ce_real_region=float(real.detach()),ce_augmented_region=float(aug.detach()),gradient_l2=float(norm),
                    teacher_correct_rows=int(mask.sum()),active_competitor_gaps=active,hard_target_rows=64,augmented_rows=48)
                stream.write(json.dumps(dict(step=step+1,**numeric,encoder_layers=copy.deepcopy(layers),parameter_gradient_l2=gradients,
                    teacher_gradients_absent=True))+"\n");stream.flush();trackio.log(numeric,step=step+1)
                if (step+1)%32==0:
                    print(json.dumps(dict(event="joint_gap_training",step=step+1,**numeric)),flush=True)
    except Exception as e:
        direct.broad.previous._write(folder / "failure.json",dict(status="terminal_failure",error=str(e),completed_optimizer_steps=updates))
        trackio.alert(title="training_failure",text=str(e),level=trackio.AlertLevel.ERROR)
        raise
    finally:
        for handle in handles:
            handle.remove()
        trackio.finish()
    checkpoint=folder / "main_trained.pt"
    torch.save(dict(schema="aiflow-direct-margin-retained-main/v29",state_dict=model.state_dict(),math_labels=labels,auxiliary_labels=[],
        report=dict(input_contract=dict(observed_channel_mode="uniform-time",math_observed_transform="uniform-time"),product_adopted=False,
            direct_augmented_ce=True,feature_retention_weight=plan["feature_weight"],gap_retention_weight=plan["gap_weight"],label_inheritance_assumption=True)),checkpoint)
    direct.broad.previous._write(folder / "completed.json",dict(status="completed",steps=updates,
        checkpoint_sha256=direct.broad.previous._sha(checkpoint),microscope_sha256=direct.broad.previous._sha(folder / "training_microscope.jsonl"),
        canonical_unchanged=direct.broad.previous._sha(direct.broad.previous.CHECKPOINT)==plan["canonical_sha256"],evaluation_forwarded_during_training=0))
    print(json.dumps(dict(event="joint_gap_completed",steps=updates)),flush=True)
    return 0


def evaluate():
    """새 모델만 동일 입력에서 평가하고 V27 검증된 비교표에 추가한다."""
    if (OUT / "comparison_result.json").exists():
        raise FileExistsError("comparison exists")
    plan,_,_=load();folder=OUT / "margin_retained_main"
    done=json.loads((folder / "completed.json").read_text(encoding="utf-8"))
    assert done["steps"]==2400 and direct.broad.previous._sha(folder / "main_trained.pt")==done["checkpoint_sha256"]
    baseline=json.loads((previous.OUT / "comparison_result.json").read_text(encoding="utf-8"))
    certificate=json.loads((previous.OUT / "independent_verification.json").read_text(encoding="utf-8"))
    assert certificate["status"]=="pass" and certificate["comparison_sha256"]==direct.broad.previous._sha(previous.OUT / "comparison_result.json")
    import torch
    import audit_hwr_owned_formula_transfer_v21 as owned
    from audit_hwr_encoder_head_swap_v22 import records_from_scores
    from run_hwr_affine_distillation_experiment_v1 import _load_teacher,_predict_logits
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    model,labels,_=_load_teacher(folder / "main_trained.pt",torch.device("cpu"))
    _,data,folds,_=direct.broad.previous.global_load(direct.broad.previous.OUTPUT);development={}
    for fold in plan["development_folds"]:
        x=np.array(data["features"][folds[fold]],copy=True);y=np.array(data["labels"][folds[fold]],copy=True)
        z=_predict_logits(model,x,torch.device("cpu"),32);np.save(OUT / f"margin_retained_main_fold{fold}_logits.npy",z,allow_pickle=False)
        development[str(fold)]=direct.broad.previous.hit_metrics(z,y,labels)
    _,samples,rows=owned.load_source();torch.set_num_threads(1)
    z=_predict_logits(model,np.load(owned.OUT / "inputs.npy",allow_pickle=False),torch.device("cpu"),owned.BATCH)
    records=records_from_scores(z,labels,samples,rows)
    np.save(OUT / "margin_retained_main_owned_logits.npy",z,allow_pickle=False);direct.broad.previous._write(OUT / "margin_retained_main_owned_records.json",records)
    metrics=copy.deepcopy(baseline["metrics"]);metrics["margin_retained_main"]=dict(development=development,
        owned={c:owned.metrics(records,c) for c in ("all","legacy_96","codex_reviewed_53")})
    result=dict(schema="aiflow-direct-margin-retention-comparison/v29",status="completed_consumed_diagnostics_not_fresh_acceptance",
        baseline_comparison_sha256=certificate["comparison_sha256"],metrics=metrics,hard_ce_weight=1.,feature_weight=plan["feature_weight"],gap_weight=plan["gap_weight"],
        all_57_main_tensors_optimized=True,all_evaluation_cases_already_consumed=True,generated_labels_human_verified=False,
        owned_optimizer_rows=0,crohme_rows=0,product_adopted=False,
        canonical_unchanged=direct.broad.previous._sha(direct.broad.previous.CHECKPOINT)==plan["canonical_sha256"])
    direct.broad.previous._write(OUT / "comparison_result.json",result)
    print(json.dumps(dict(event="joint_gap_model_scored",metrics=metrics["margin_retained_main"])),flush=True)
    return 0


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("mode",choices=("prepare","selftest","train","evaluate"));args=parser.parse_args()
    if args.mode=="selftest":
        selftest();raise SystemExit(0)
    raise SystemExit(dict(prepare=prepare,train=train,evaluate=evaluate)[args.mode]())
