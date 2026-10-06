"""광범위 증강과 예산을 유지하며 원본 TRAIN의 고정 Top-1/5 margin 보존을 추가한다.

V16 오답·검증 logits는 읽지 않고 같은 실제 입력·teacher·배치·초기 가중치를
재사용한다. 변경 변수는 +0.1 고정 teacher-correct margin hinge 하나다.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np

import run_hwr_broad_diversity_v16 as broad

OUT = broad.previous.ROOT / "artifacts/hwr_broad_margin_20261005_v17"
WEIGHT, FLOOR_FRACTION = .1, .5


def margin_loss(student, teacher, truth):
    """truth를 제외한 현재 1·5위 rival과 봉인된 teacher-correct 절반 margin을 비교한다."""
    import torch
    from torch.nn import functional as F
    if student.shape != teacher.shape or student.ndim != 2 or student.shape[1] != 372 or truth.shape != (len(student),):
        raise ValueError("margin inputs must align as [N,372] and [N]")
    if truth.dtype != torch.long or not torch.isfinite(student).all() or not torch.isfinite(teacher).all():
        raise ValueError("margin inputs need finite logits and long targets")
    rows=torch.arange(len(truth),device=student.device)
    with torch.no_grad():
        reference=teacher.detach().clone();reference[rows,truth]=-torch.inf
        other=reference.topk(5,dim=1).values
        target=teacher[rows,truth]
        order=torch.argsort(teacher,dim=1,descending=True,stable=True)
        masks=(order[:,0]==truth,(order[:,:5]==truth[:,None]).any(1))
        floors=(FLOOR_FRACTION*(target-other[:,0]),FLOOR_FRACTION*(target-other[:,4]))
    rivals=student.clone();rivals[rows,truth]=-torch.inf
    competing=rivals.topk(5,dim=1).values
    gaps=(student[rows,truth]-competing[:,0],student[rows,truth]-competing[:,4])
    terms=[];report={}
    for k,mask,floor,gap in zip((1,5),masks,floors,gaps):
        deficit=F.relu(floor-gap)
        term=deficit[mask].mean() if mask.any() else student.sum()*0
        terms.append(term)
        report[f"protected_top{k}"]=int(mask.sum())
        report[f"floor_fail_top{k}"]=int((mask & (gap.detach()<floor)).sum())
        report[f"mean_deficit_top{k}"]=float(term.detach())
    return .5*(terms[0]+terms[1]),report


def prepare() -> int:
    """데이터와 대조군 봉인 상태를 확인하고 미사용 fold 2·하나의 loss 변경을 고정한다."""
    if OUT.exists():raise FileExistsError("refusing margin experiment overwrite")
    plan,manifest,_=broad.load()
    data=json.loads((broad.OUT / "independent_data_verification.json").read_text(encoding="utf-8"))
    verified=json.loads((broad.OUT / "independent_result_verification.json").read_text(encoding="utf-8"))
    control=broad.OUT / "broad_augmented/research.pt"
    completed=json.loads((broad.OUT / "broad_augmented/completed.json").read_text(encoding="utf-8"))
    if data["plan_sha256"]!=broad.previous._sha(broad.OUT / "frozen_plan.json") or not verified["both_2400_step_arms_verified"]:
        raise ValueError("broad data/control are not independently verified")
    if broad.previous._sha(control)!=completed["checkpoint_sha256"] or completed["steps"]!=broad.STEPS:
        raise ValueError("matched control checkpoint changed")
    OUT.mkdir(parents=True)
    broad.previous._write(OUT / "frozen_plan.json",dict(schema="aiflow-broad-fixed-margin-plan/v17",
        script_sha256=broad.previous._sha(Path(__file__)),helper_sha256=broad.previous._sha(Path(broad.__file__)),
        parent_plan_sha256=broad.previous._sha(broad.OUT / "frozen_plan.json"),parent_manifest_sha256=broad.previous._sha(broad.OUT / "prepared_manifest.json"),
        data_certificate_sha256=broad.previous._sha(broad.OUT / "independent_data_verification.json"),
        control_certificate_sha256=broad.previous._sha(broad.OUT / "independent_result_verification.json"),
        control_checkpoint_sha256=completed["checkpoint_sha256"],canonical_sha256=plan["canonical_sha256"],
        steps=broad.STEPS,batch_size=broad.BATCH,learning_rate=1e-5,margin_weight=WEIGHT,reference_margin_fraction=FLOOR_FRACTION,
        unchanged_base_objective=".7 CE(real) + .2 KL(real,canonical,T=2) + .1 KL(augmented,original-canonical,T=2)",
        additional_objective="+.1 * .5 * (mean protected Top-1 deficit + mean protected Top-5 deficit)",
        protection="immutable original canonical-correct masks/floors on real optimization rows only; dynamic first/fifth OTHER rivals",
        initialization="cold canonical, identical V16 augmented initialization; NOT warm start or extra steps",
        data_reuse="exact same 76800 real exposures and corresponding 68726 generated views; no regeneration or narrowed scope",
        population_rows=manifest["population_rows"],classes=manifest["classes"],shape_temperatures=broad.TEMPERATURES,quantiles=broad.QUANTILES,
        validation_fold=2,validation_error_rows_used_for_training=0,previous_validation_logits_loaded=False,
        promotion_gate="report fresh-fold paired wins/losses and all families; no product activation; no threshold/best-epoch retry",
        independent_writer_device_acceptance=False,official_test_rows_read=0,crohme_rows=0,synthetic_hard_labels=0,
        collection_or_deployment_resumed=False,cloud_upload=False,product_adopted=False))
    print(json.dumps(dict(event="margin_plan_frozen",steps=broad.STEPS,validation_fold=2,margin_weight=WEIGHT)),flush=True)
    return 0


def load():
    """현재 코드·데이터·기존 대조군의 SHA를 확인하고 검증 logits를 읽지 않는다."""
    plan=json.loads((OUT / "frozen_plan.json").read_text(encoding="utf-8"))
    if plan["script_sha256"]!=broad.previous._sha(Path(__file__)) or plan["helper_sha256"]!=broad.previous._sha(Path(broad.__file__)):
        raise ValueError("frozen entrypoint changed")
    if plan["parent_plan_sha256"]!=broad.previous._sha(broad.OUT / "frozen_plan.json") or plan["parent_manifest_sha256"]!=broad.previous._sha(broad.OUT / "prepared_manifest.json"):
        raise ValueError("parent data plan changed")
    if plan["control_checkpoint_sha256"]!=broad.previous._sha(broad.OUT / "broad_augmented/research.pt"):
        raise ValueError("control changed")
    _,_,arrays=broad.load()
    return plan,arrays


def train() -> int:
    """전체 encoder를 동일 원본·변형에서 학습하며 각 단계의 고정 margin 위반을 기록한다."""
    if broad.previous.aug.base._guard_commit("broad_margin_train") is None:return 78
    plan,a=load()
    if (OUT / "run_started.json").exists():raise FileExistsError("already started; inspect actual live/terminal state, do not blindly restart")
    os.environ["TRACKIO_DIR"]=str(OUT / "trackio")
    for key in ("TRACKIO_WEBHOOK_URL","TRACKIO_SPACE_ID","TRACKIO_SERVER_URL"):os.environ.pop(key,None)
    import torch,trackio
    from torch.nn import functional as F
    from hwr_boundary_distillation_v1 import boundary_kl_loss
    from run_hwr_affine_distillation_experiment_v1 import _load_teacher
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    model,labels,_=_load_teacher(broad.previous.CHECKPOINT,torch.device("cpu"));model.eval()
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=1e-4)
    broad.previous._write(OUT / "run_started.json",dict(status="started",process_id=os.getpid(),plan_sha256=broad.previous._sha(OUT / "frozen_plan.json")))
    activations={};handles=[]
    def capture(name):
        """각 encoder 층의 실제 출력 유한성과 평균/표준편차를 기록한다."""
        def hook(module,inputs,output):
            value=output.detach()
            if not torch.isfinite(value).all():raise FloatingPointError("nonfinite activation")
            activations[name]=dict(mean=float(value.mean()),std=float(value.std()))
        return hook
    for name,module in model.named_modules():
        if isinstance(module,torch.nn.TransformerEncoderLayer):handles.append(module.register_forward_hook(capture(name)))
    trackio.init(project="aiflow-broad-margin-v17",name="fixed_margin",space_id=None,embed=False,auto_log_cpu=False,auto_log_gpu=False,
        config=dict(steps=broad.STEPS,margin_weight=WEIGHT,floor_fraction=FLOOR_FRACTION))
    try:
        with (OUT / "training_microscope.jsonl").open("x",encoding="utf-8") as stream:
            for step,ids in enumerate(a["shared_schedule"]):
                real=np.array(a["population_features"][ids],copy=True);view=np.array(a["scheduled_augmented_features"][step],copy=True)
                teacher=torch.from_numpy(np.array(a["population_teacher_logits"][ids],copy=True))
                truth=torch.from_numpy(np.array(a["population_labels"][ids],dtype=np.int64,copy=True))
                optimizer.zero_grad(set_to_none=True)
                embedding=model.encode(torch.from_numpy(np.concatenate((real,view))));z=model.math_head(embedding)
                ce=F.cross_entropy(z[:broad.BATCH],truth);retain=boundary_kl_loss(z[:broad.BATCH],teacher,2.,"full")
                consistency=boundary_kl_loss(z[broad.BATCH:],teacher,2.,"full")
                protected,details=margin_loss(z[:broad.BATCH],teacher,truth)
                base=.7*ce+.2*retain+.1*consistency;loss=base+WEIGHT*protected
                if not torch.isfinite(loss):raise FloatingPointError("nonfinite loss")
                loss.backward()
                gradients={name:None if p.grad is None else float(p.grad.detach().norm()) for name,p in model.named_parameters()}
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                if not torch.isfinite(norm):raise FloatingPointError("nonfinite gradient")
                optimizer.step()
                numeric=dict(loss=float(loss.detach()),base_loss=float(base.detach()),ce_real=float(ce.detach()),kl_real=float(retain.detach()),
                    kl_view=float(consistency.detach()),margin_loss=float(protected.detach()),gradient_l2=float(norm),**details)
                stream.write(json.dumps(dict(step=step+1,**numeric,encoder_layers=copy.deepcopy(activations),parameter_gradient_l2=gradients))+"\n")
                stream.flush();trackio.log(numeric,step=step+1)
                if (step+1)%32==0:print(json.dumps(dict(event="broad_margin_training",step=step+1,**numeric)),flush=True)
    except Exception as error:
        broad.previous._write(OUT / "failure.json",dict(status="terminal_failure",error=str(error),completed_steps=step))
        trackio.alert(title="training_failure",text=str(error),level=trackio.AlertLevel.ERROR);raise
    finally:
        for handle in handles:handle.remove()
        trackio.finish()
    checkpoint=OUT / "research.pt"
    torch.save(dict(schema="aiflow-broad-fixed-margin-research/v17",state_dict=model.state_dict(),math_labels=labels,auxiliary_labels=[],
        report=dict(input_contract=dict(observed_channel_mode="uniform-time",math_observed_transform="uniform-time"),product_adopted=False)),checkpoint)
    broad.previous._write(OUT / "completed.json",dict(status="completed",steps=broad.STEPS,checkpoint_sha256=broad.previous._sha(checkpoint),
        microscope_sha256=broad.previous._sha(OUT / "training_microscope.jsonl"),validation_forwarded=0,
        canonical_unchanged=broad.previous._sha(broad.previous.CHECKPOINT)==plan["canonical_sha256"]))
    print(json.dumps(dict(event="broad_margin_completed",steps=broad.STEPS)),flush=True)
    return 0


def evaluate() -> int:
    """고정 단계 종료 후에만 남겨둔 fold 2를 한 번 평가하며 이전 fold는 읽지 않는다."""
    if (OUT / "comparison_result.json").exists():raise FileExistsError("fold already consumed")
    plan,_=load();done=json.loads((OUT / "completed.json").read_text(encoding="utf-8"))
    if done["steps"]!=broad.STEPS or broad.previous._sha(OUT / "research.pt")!=done["checkpoint_sha256"]:raise ValueError("training incomplete or changed")
    _,a,folds,_=broad.previous.global_load(broad.previous.OUTPUT)
    x=np.array(a["features"][folds[2]],copy=True);y=np.array(a["labels"][folds[2]],copy=True)
    import torch
    from run_hwr_affine_distillation_experiment_v1 import _load_teacher,_predict_logits
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    paths={"canonical":broad.previous.CHECKPOINT,"broad_augmented":broad.OUT / "broad_augmented/research.pt","fixed_margin":OUT / "research.pt"}
    metrics={};scores={}
    for name,path in paths.items():
        model,labels,_=_load_teacher(path,torch.device("cpu"));scores[name]=_predict_logits(model,x,torch.device("cpu"),broad.BATCH)
        np.save(OUT / f"{name}_fold2_logits.npy",scores[name],allow_pickle=False);metrics[name]=broad.previous.hit_metrics(scores[name],y,labels)
    before=scores["broad_augmented"].argmax(1)==y;after=scores["fixed_margin"].argmax(1)==y
    result=dict(schema="aiflow-broad-fixed-margin-comparison/v17",status="completed_internal_train_diagnostic",plan_sha256=broad.previous._sha(OUT / "frozen_plan.json"),
        metrics=metrics,paired_vs_control=dict(wins=int((~before&after).sum()),losses=int((before&~after).sum()),net=int(after.sum()-before.sum())),
        validation_fold_consumed=2,prior_validation_logits_read=False,parameter_updates_in_evaluation=0,
        independent_writer_device_acceptance=False,product_adopted=False,official_test_rows_read=0,crohme_rows=0,
        canonical_unchanged=broad.previous._sha(broad.previous.CHECKPOINT)==plan["canonical_sha256"])
    broad.previous._write(OUT / "comparison_result.json",result);print(json.dumps(dict(event="margin_comparison",metrics=metrics,paired=result["paired_vs_control"])),flush=True)
    return 0


def selftest() -> int:
    """기존 logits에서 보존 손실 0, 위반 시 truth 방향 gradient, teacher 오답 mask를 검사한다."""
    import torch
    teacher=torch.zeros((2,372));truth=torch.tensor([0,7],dtype=torch.long)
    teacher[0,0]=4;teacher[1,7]=1;teacher[1,2]=2
    exact,report=margin_loss(teacher,teacher,truth)
    assert float(exact)==0 and report["protected_top1"]==1 and report["protected_top5"]==2
    student=teacher.clone();student[0,0]=.5;student[0,1]=1;student.requires_grad_(True)
    loss,report=margin_loss(student,teacher,truth);loss.backward()
    assert float(loss)>0 and student.grad[0,0]<0 and student.grad[0,1]>0
    assert report["floor_fail_top1"]==1
    print(json.dumps(dict(selftest="pass",fixed_top1_and_top5=True,synthetic_hard_labels=0)),flush=True);return 0


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("mode",choices=("selftest","prepare","train","evaluate"));args=parser.parse_args()
    raise SystemExit({"selftest":selftest,"prepare":prepare,"train":train,"evaluate":evaluate}[args.mode]())
