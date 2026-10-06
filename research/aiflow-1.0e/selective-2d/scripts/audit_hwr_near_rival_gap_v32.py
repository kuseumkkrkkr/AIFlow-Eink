"""TRAIN 입력에서 371-rival 평균 gap 손실과 실제 Top-5 근접 후보 손실을 분해한다.

라벨·계수·모델을 바꾸지 않는다. 같은 고정 TRAIN 8 batch만 사용하며 평가/소유 GT는 읽지 않는다.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

import run_hwr_direct_margin_retention_v29 as previous
import run_hwr_digit_floor_sampling_v31 as digit
from run_hwr_affine_distillation_experiment_v1 import _load_teacher

OUT=previous.direct.broad.previous.ROOT / "artifacts/hwr_near_rival_gap_audit_20261006_v32"


def main():
    """동일 source batch의 순위·손실 energy·전체 main gradient를 원본 파일 변경 없이 측정한다."""
    if OUT.exists():
        raise FileExistsError("refusing near-rival audit overwrite")
    plan,a,s=previous.load();sha=previous.direct.broad.previous._sha;write=previous.direct.broad.previous._write
    paths=dict(feature_only=previous.BASE,all_gap=previous.OUT / "margin_retained_main/main_trained.pt",
        digit_floor=digit.OUT / "margin_retained_main/main_trained.pt")
    hashes={n:sha(p) for n,p in paths.items()};canonical_sha=sha(previous.direct.broad.previous.CHECKPOINT)
    for root,key in ((previous.OUT,"all_gap"),(digit.OUT,"digit_floor")):
        certificate=json.loads((root / "independent_verification.json").read_text(encoding="utf-8"))
        assert certificate["status"]=="pass" and certificate["checkpoint_sha256"]==hashes[key]
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    teacher,labels,_=_load_teacher(previous.direct.broad.previous.CHECKPOINT,torch.device("cpu"));teacher.eval();teacher.requires_grad_(False)
    steps=np.linspace(0,2399,8,dtype=int);models={}
    for n,p in paths.items():
        model,vocab,_=_load_teacher(p,torch.device("cpu"));model.eval();assert vocab==labels;models[n]=model
    OUT.mkdir();write(OUT / "frozen_plan.json",dict(schema="aiflow-near-rival-gap-plan/v32",code_sha256=sha(Path(__file__)),
        original_v29_plan_sha256=sha(previous.OUT / "frozen_plan.json"),checkpoint_sha256=hashes,canonical_sha256=canonical_sha,
        schedule_steps=steps.tolist(),source="Same eight fixed V25 admitted TRAIN batches for all models; no V31 sampler substitution",
        near_rule="Union of canonical/student ordered Top-5, ground truth excluded; teacher-correct rows only",
        optimizer_steps=0,coefficient_selection=False,evaluation_inputs_read=0,owned_labels_read=0,crohme_rows=0,product_adopted=False))
    numeric={n:[] for n in models};rows=[];max_double_delta=0.
    for step in steps:
        x,y=previous.direct.batch(a,s,int(step),"augmented_main");tx=torch.from_numpy(x);truth=torch.from_numpy(y)
        with torch.no_grad():
            tz=teacher.math_head(teacher.encode(tx))
        correct=tz.argmax(1)==truth;tother=tz.clone();tother.scatter_(1,truth[:,None],float("-inf"))
        teacher_margin=tz.gather(1,truth[:,None]).squeeze(1)-tother.max(1).values
        for name,model in models.items():
            z=model.math_head(model.encode(tx));ce=F.cross_entropy(z,truth)
            gap,mask,_=previous.gap_loss(z,tz,truth);assert torch.equal(mask,correct)
            near=torch.zeros_like(z,dtype=torch.bool)
            near.scatter_(1,tz.topk(5,dim=1).indices,True)
            near.scatter_(1,z.detach().topk(5,dim=1).indices,True)
            near.scatter_(1,truth[:,None],False);near &= correct[:,None]
            deficit=torch.relu((tz.gather(1,truth[:,None])-tz)-(z.gather(1,truth[:,None])-z))*correct[:,None]
            squared=deficit.square();all_energy=squared.sum();near_energy=(squared*near).sum()
            near_loss=near_energy/near.sum().clamp_min(1)
            params=list(model.parameters());norms={}
            for key,value in (("ce",ce),("all_gap",gap),("near_gap",near_loss)):
                gradients=torch.autograd.grad(value,params,retain_graph=True)
                assert len(gradients)==57 and all(torch.isfinite(g).all() for g in gradients)
                norms[key]=float(torch.sqrt(sum(g.square().sum() for g in gradients)))
            zg=torch.autograd.grad(near_loss,z)[0]
            assert (zg[torch.arange(64),truth]<=1e-7).all() and (zg[~correct]==0).all()
            zz=z.detach().numpy().astype(np.float64);tt=tz.numpy().astype(np.float64)
            dg=np.maximum((tt[np.arange(64),y,None]-tt)-(zz[np.arange(64),y,None]-zz),0)*correct.numpy()[:,None]
            expected=float((dg**2*near.numpy()).sum()/max(int(near.sum()),1))
            max_double_delta=max(max_double_delta,abs(expected-float(near_loss.detach())))
            assert abs(expected-float(near_loss.detach()))<1e-5
            other=z.detach().clone();other.scatter_(1,truth[:,None],float("-inf"))
            margin=z.detach().gather(1,truth[:,None]).squeeze(1)-other.max(1).values
            record=dict(model=name,schedule_step=int(step),teacher_correct_rows=int(correct.sum()),
                student_correct_rows=int((z.detach().argmax(1)==truth).sum()),
                teacher_correct_student_wrong_rows=int((correct & (z.detach().argmax(1)!=truth)).sum()),
                teacher_correct_closest_margin_reduced_rows=int((correct & (margin<teacher_margin)).sum()),
                near_competitor_pairs=int(near.sum()),all_competitor_pairs=int(correct.sum())*371,
                all_gap_loss=float(gap.detach()),near_gap_loss=float(near_loss.detach()),
                all_squared_deficit_energy=float(all_energy.detach()),near_squared_deficit_energy=float(near_energy.detach()),
                near_energy_fraction=float((near_energy/all_energy).detach()) if float(all_energy.detach())>0 else None,
                gradient_l2=norms,teacher_correct_closest_margin_mean=float(margin[correct].mean()))
            numeric[name].append(record);rows.append(record)
    summaries={}
    for name,records in numeric.items():
        total=sum(r["all_squared_deficit_energy"] for r in records);near=sum(r["near_squared_deficit_energy"] for r in records)
        summaries[name]=dict(rows=512,teacher_correct_rows=sum(r["teacher_correct_rows"] for r in records),
            student_correct_rows=sum(r["student_correct_rows"] for r in records),
            teacher_correct_student_wrong_rows=sum(r["teacher_correct_student_wrong_rows"] for r in records),
            closest_margin_reduced_rows=sum(r["teacher_correct_closest_margin_reduced_rows"] for r in records),
            mean_all_gap_loss=float(np.mean([r["all_gap_loss"] for r in records])),
            mean_near_gap_loss=float(np.mean([r["near_gap_loss"] for r in records])),
            near_energy_fraction=near/total if total>0 else None,
            mean_all_gap_gradient_l2=float(np.mean([r["gradient_l2"]["all_gap"] for r in records])),
            mean_near_gap_gradient_l2=float(np.mean([r["gradient_l2"]["near_gap"] for r in records])))
    assert all(sha(p)==hashes[n] for n,p in paths.items()) and sha(previous.direct.broad.previous.CHECKPOINT)==canonical_sha
    result=dict(schema="aiflow-near-rival-gap-result/v32",status="verified_train_only_probe_not_accuracy_acceptance",summaries=summaries,
        records=rows,near_loss_independently_checked_in_numpy_double=True,max_numpy_delta=max_double_delta,
        near_gradient_correct_direction_and_teacher_wrong_mask_verified=True,all_57_gradient_scopes_finite=True,
        checkpoint_files_unchanged=True,optimizer_steps=0,coefficient_selection=False,evaluation_inputs_read=0,owned_labels_read=0,
        crohme_rows=0,product_adopted=False,limits="Eight fixed TRAIN batches are already fitted inputs, not generalization evidence. Near masks depend on detached predicted scores; this diagnostic does not train a near-rival objective or choose its weight.")
    write(OUT / "near_gap_result.json",result)
    print(json.dumps(dict(event="near_gap_audit_complete",summaries=summaries,max_numpy_delta=max_double_delta)),flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
