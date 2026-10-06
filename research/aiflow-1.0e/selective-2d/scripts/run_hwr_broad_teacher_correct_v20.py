"""V16의 데이터·예산을 유지하고 canonical 오답 원본에서만 두 KL 항을 끈다.

이미 소비된 내부 fold 1·2는 개발 비교만 한다. 새 acceptance가 아니며
생성물에는 hard CE를 적용하지 않는다. 제품/수집/배포 상태는 변경하지 않는다.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np
import run_hwr_broad_diversity_v16 as broad

OUT = broad.previous.ROOT / "artifacts/hwr_broad_teacher_correct_20261005_v20"


def teacher_correct_kl(student, teacher, truth, enabled=True):
    """원본의 고정 teacher 정답 mask만 쓰며 제외 행은 0, 분모는 전체 batch로 유지한다."""
    import torch
    from torch.nn import functional as F
    from hwr_boundary_distillation_v1 import boundary_kl_loss
    if student.shape != teacher.shape or student.ndim != 2 or student.shape[1] != 372 or not len(student):
        raise ValueError("nonempty matching [N,372] required")
    if truth.shape != (len(student),) or truth.dtype != torch.long or not ((truth>=0)&(truth<372)).all():
        raise ValueError("original real targets need valid long [N]")
    if not torch.isfinite(student).all() or not torch.isfinite(teacher).all():
        raise ValueError("nonfinite KL logits")
    # off 경로는 기존 함수 호출 그대로라 결과/gradient byte parity를 보존한다.
    if not enabled:
        return boundary_kl_loss(student, teacher, 2., "full")
    with torch.no_grad():
        mask = teacher.detach().argmax(1) == truth
        q = torch.softmax(teacher.detach().float()/2., 1)
    per_row = F.kl_div(F.log_softmax(student.float()/2., 1), q, reduction="none").sum(1)*4.
    return (per_row * mask).mean()


def selftest():
    """off 동등성·teacher detach·전체 batch 분모·오답 행 영 gradient를 직접 검사한다."""
    import torch
    from hwr_boundary_distillation_v1 import boundary_kl_loss
    torch.manual_seed(20)
    teacher = torch.randn(8,372,requires_grad=True)
    truth = teacher.detach().argmax(1); truth[1::2] = (truth[1::2]+1)%372
    student = torch.randn(8,372,requires_grad=True)
    off = teacher_correct_kl(student,teacher,truth,False)
    old = boundary_kl_loss(student,teacher,2.,"full")
    assert torch.equal(off,old)
    goff = torch.autograd.grad(off,student,retain_graph=True)[0]
    gold = torch.autograd.grad(old,student,retain_graph=True)[0]
    assert torch.equal(goff,gold)
    on = teacher_correct_kl(student,teacher,truth)
    gradient = torch.autograd.grad(on,student,retain_graph=True)[0]
    assert torch.count_nonzero(gradient[1::2]) == 0
    assert torch.count_nonzero(gradient[::2]) > 0
    assert torch.allclose(gradient[::2],gold[::2],atol=1e-7,rtol=1e-5)
    subset = boundary_kl_loss(student[::2],teacher[::2],2.,"full")*.5
    assert torch.allclose(on,subset,atol=1e-7,rtol=1e-5)
    assert torch.autograd.grad(on,teacher,allow_unused=True)[0] is None
    all_wrong = (teacher.detach().argmax(1)+1)%372
    zero = teacher_correct_kl(student,teacher,all_wrong)
    assert float(zero.detach()) == 0 and torch.count_nonzero(torch.autograd.grad(zero,student)[0]) == 0
    print(json.dumps(dict(selftest="pass",off_loss_and_gradient_bit_exact=True,incorrect_original_gradient_zero=True,
        full_batch_denominator=True,teacher_detached=True,synthetic_hard_labels=0)),flush=True)
    return 0


def prepare():
    """한 가지 변경과 동일 학습 예산을 고정하며 소비된 fold를 fresh로 재명명하지 않는다."""
    if OUT.exists():
        raise FileExistsError("refusing teacher-correct experiment overwrite")
    parent,manifest,a = broad.load()
    certificate = broad.previous.ROOT / "artifacts/hwr_broad_teacher_conflict_20261005_v19/independent_verification.json"
    if json.loads(certificate.read_text(encoding="utf-8"))["status"] != "pass":
        raise ValueError("preceding diagnosis not verified")
    canonical_wrong = a["population_teacher_logits"].argmax(1) != a["population_labels"]
    scheduled_wrong = int(canonical_wrong[a["shared_schedule"]].sum())
    control = broad.OUT / "broad_augmented/research.pt"
    completed = json.loads((broad.OUT / "broad_augmented/completed.json").read_text(encoding="utf-8"))
    if completed["steps"] != broad.STEPS or completed["checkpoint_sha256"] != broad.previous._sha(control):
        raise ValueError("matched-budget control changed")
    OUT.mkdir()
    broad.previous._write(OUT / "frozen_plan.json", dict(schema="aiflow-teacher-correct-plan/v20",
        script_sha256=broad.previous._sha(Path(__file__)),helper_sha256=broad.previous._sha(Path(broad.__file__)),
        parent_plan_sha256=broad.previous._sha(broad.OUT / "frozen_plan.json"),
        parent_manifest_sha256=broad.previous._sha(broad.OUT / "prepared_manifest.json"),
        diagnosis_certificate_sha256=broad.previous._sha(certificate),canonical_sha256=parent["canonical_sha256"],
        control_checkpoint_sha256=completed["checkpoint_sha256"],steps=broad.STEPS,batch_size=broad.BATCH,
        learning_rate=1e-5,weight_decay=1e-4,clip_grad_norm=1.,logit_temperature=2.,
        cold_initialization="same canonical as V16; no warm start or extra steps",classes=manifest["classes"],
        population_rows=manifest["population_rows"],scheduled_exposures=broad.STEPS*broad.BATCH,
        expected_incorrect_original_exposures=scheduled_wrong,shape_temperatures=broad.TEMPERATURES,quantiles=broad.QUANTILES,
        controlled_change="zero both KL terms only when canonical original Top1 != real GT; mean over full batch, no weight redistribution",
        objective=".7 CE(real) + .2 mean(mask * KL(real,canonical)) + .1 mean(mask * KL(view,original-canonical)), T=2",
        development_folds=[1,2],all_development_folds_already_consumed=True,new_validation_fold=False,
        previous_validation_error_rows_used_for_training=0,development_selection_bias=True,
        read_development_only_after_fixed_steps=True,interim_evaluation_or_best_epoch_selection=False,
        evaluation_gate="report paired Top1/Top5 wins/losses and every family on both consumed folds; no fresh acceptance claim",
        independent_writer_device_acceptance=False,synthetic_hard_labels=0,human_boundary_labels=0,
        official_test_rows_read=0,crohme_rows=0,product_adopted=False,collection_or_deployment_resumed=False,cloud_upload=False))
    print(json.dumps(dict(event="teacher_correct_plan_frozen",steps=broad.STEPS,incorrect_original_exposures=scheduled_wrong,
        consumed_development_folds=[1,2],fresh_validation=False)),flush=True)
    return 0


def load():
    """학습 전에 봉인 코드·원본 데이터·기준 checkpoint를 확인한다."""
    plan = json.loads((OUT / "frozen_plan.json").read_text(encoding="utf-8"))
    expected = ((Path(__file__),"script_sha256"),(Path(broad.__file__),"helper_sha256"),
        (broad.OUT / "frozen_plan.json","parent_plan_sha256"),(broad.OUT / "prepared_manifest.json","parent_manifest_sha256"),
        (broad.OUT / "broad_augmented/research.pt","control_checkpoint_sha256"))
    if any(broad.previous._sha(path) != plan[key] for path,key in expected):
        raise ValueError("frozen dependencies changed")
    _,_,a = broad.load()
    return plan,a


def train():
    """동일 64-row forward를 2,400회 실행하고 매 단계 4층·57 parameter의 수치를 기록한다."""
    if broad.previous.aug.base._guard_commit("teacher_correct_train_v20") is None:
        return 78
    plan,a = load()
    if (OUT / "run_started.json").exists():
        raise FileExistsError("already started; inspect live handle/terminal state before any restart")
    os.environ["TRACKIO_DIR"] = str(OUT / "trackio")
    for key in ("TRACKIO_WEBHOOK_URL","TRACKIO_SPACE_ID","TRACKIO_SERVER_URL"):
        os.environ.pop(key,None)
    import torch,trackio
    from torch.nn import functional as F
    from run_hwr_affine_distillation_experiment_v1 import _load_teacher
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    model,labels,_ = _load_teacher(broad.previous.CHECKPOINT,torch.device("cpu"));model.eval()
    optimizer = torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=1e-4)
    broad.previous._write(OUT / "run_started.json",dict(status="started",process_id=os.getpid(),
        plan_sha256=broad.previous._sha(OUT / "frozen_plan.json")))
    activations = {};handles = []
    def capture(name):
        """각 encoder 층의 유한성과 평균/표준편차를 기록한다."""
        def hook(module,inputs,output):
            value = output.detach()
            if not torch.isfinite(value).all():
                raise FloatingPointError("nonfinite encoder activation")
            activations[name] = dict(mean=float(value.mean()),std=float(value.std()))
        return hook
    for name,module in model.named_modules():
        if isinstance(module,torch.nn.TransformerEncoderLayer):
            handles.append(module.register_forward_hook(capture(name)))
    trackio.init(project="aiflow-teacher-correct-v20",name="teacher_correct_kl",space_id=None,embed=False,
        auto_log_cpu=False,auto_log_gpu=False,config=dict(steps=broad.STEPS,lr=1e-5,full_batch_denominator=True))
    updates = 0
    try:
        with (OUT / "training_microscope.jsonl").open("x",encoding="utf-8") as stream:
            for step,ids in enumerate(a["shared_schedule"]):
                real = np.array(a["population_features"][ids],copy=True)
                view = np.array(a["scheduled_augmented_features"][step],copy=True)
                teacher = torch.from_numpy(np.array(a["population_teacher_logits"][ids],copy=True))
                truth = torch.from_numpy(np.array(a["population_labels"][ids],dtype=np.int64,copy=True))
                optimizer.zero_grad(set_to_none=True)
                z = model.math_head(model.encode(torch.from_numpy(np.concatenate((real,view)))))
                ce = F.cross_entropy(z[:broad.BATCH],truth)
                retain = teacher_correct_kl(z[:broad.BATCH],teacher,truth)
                consistency = teacher_correct_kl(z[broad.BATCH:],teacher,truth)
                loss = .7*ce+.2*retain+.1*consistency
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite loss")
                loss.backward()
                gradients = {name:None if p.grad is None else float(p.grad.detach().norm()) for name,p in model.named_parameters()}
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                if not torch.isfinite(norm):
                    raise FloatingPointError("nonfinite gradient")
                optimizer.step();updates += 1
                numeric = dict(loss=float(loss.detach()),ce_real=float(ce.detach()),kl_real=float(retain.detach()),
                    kl_view=float(consistency.detach()),gradient_l2=float(norm),teacher_correct_rows=int((teacher.argmax(1)==truth).sum()))
                stream.write(json.dumps(dict(step=step+1,**numeric,encoder_layers=copy.deepcopy(activations),parameter_gradient_l2=gradients))+"\n")
                stream.flush();trackio.log(numeric,step=step+1)
                if (step+1)%32 == 0:
                    print(json.dumps(dict(event="teacher_correct_training",step=step+1,**numeric)),flush=True)
    except Exception as error:
        broad.previous._write(OUT / "failure.json",dict(status="terminal_failure",error=str(error),completed_optimizer_steps=updates))
        trackio.alert(title="training_failure",text=str(error),level=trackio.AlertLevel.ERROR)
        raise
    finally:
        for handle in handles:
            handle.remove()
        trackio.finish()
    checkpoint = OUT / "research.pt"
    torch.save(dict(schema="aiflow-teacher-correct-research/v20",state_dict=model.state_dict(),math_labels=labels,auxiliary_labels=[],
        report=dict(input_contract=dict(observed_channel_mode="uniform-time",math_observed_transform="uniform-time"),product_adopted=False)),checkpoint)
    broad.previous._write(OUT / "completed.json",dict(status="completed",steps=updates,checkpoint_sha256=broad.previous._sha(checkpoint),
        microscope_sha256=broad.previous._sha(OUT / "training_microscope.jsonl"),development_forwarded_during_training=0,
        canonical_unchanged=broad.previous._sha(broad.previous.CHECKPOINT)==plan["canonical_sha256"]))
    print(json.dumps(dict(event="teacher_correct_completed",steps=updates)),flush=True)
    return 0


def evaluate():
    """고정 학습 종료 뒤 이미 소비된 두 fold를 개발 비교하며 새 검증이라고 부르지 않는다."""
    if (OUT / "comparison_result.json").exists():
        raise FileExistsError("development comparison exists")
    plan,_ = load()
    done = json.loads((OUT / "completed.json").read_text(encoding="utf-8"))
    if done["steps"] != broad.STEPS or broad.previous._sha(OUT / "research.pt") != done["checkpoint_sha256"]:
        raise ValueError("fixed training incomplete or changed")
    _,a,folds,_ = broad.previous.global_load(broad.previous.OUTPUT)
    import torch
    from run_hwr_affine_distillation_experiment_v1 import _load_teacher,_predict_logits
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    model,labels,_ = _load_teacher(OUT / "research.pt",torch.device("cpu"))
    diagnostics = {}
    for fold in plan["development_folds"]:
        y = np.array(a["labels"][folds[fold]],copy=True)
        scores = _predict_logits(model,np.array(a["features"][folds[fold]],copy=True),torch.device("cpu"),broad.BATCH)
        np.save(OUT / f"teacher_correct_fold{fold}_logits.npy",scores,allow_pickle=False)
        parent_path = broad.OUT / "broad_augmented_validation_logits.npy" if fold==1 else broad.previous.ROOT / "artifacts/hwr_broad_margin_20261005_v17/broad_augmented_fold2_logits.npy"
        old = np.load(parent_path,allow_pickle=False)
        before,after = old.argmax(1)==y,scores.argmax(1)==y
        diagnostics[str(fold)] = dict(control_logits_sha256=broad.previous._sha(parent_path),
            control=broad.previous.hit_metrics(old,y,labels),teacher_correct=broad.previous.hit_metrics(scores,y,labels),
            paired_top1=dict(wins=int((~before&after).sum()),losses=int((before&~after).sum()),net=int(after.sum()-before.sum())))
    result = dict(schema="aiflow-teacher-correct-comparison/v20",status="consumed_development_comparison_not_fresh_acceptance",
        folds=diagnostics,all_folds_previously_consumed=True,development_selection_bias=True,
        plan_sha256=broad.previous._sha(OUT / "frozen_plan.json"),parameter_updates_in_evaluation=0,
        fresh_writer_device_acceptance=False,product_adopted=False,official_test_rows_read=0,crohme_rows=0,
        canonical_unchanged=broad.previous._sha(broad.previous.CHECKPOINT)==plan["canonical_sha256"])
    broad.previous._write(OUT / "comparison_result.json",result)
    print(json.dumps(result),flush=True)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode",choices=("selftest","prepare","train","evaluate"))
    args = parser.parse_args()
    raise SystemExit(dict(selftest=selftest,prepare=prepare,train=train,evaluate=evaluate)[args.mode]())
