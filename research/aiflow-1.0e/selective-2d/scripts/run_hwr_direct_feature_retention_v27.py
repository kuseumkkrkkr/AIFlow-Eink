"""직접 증강 CE를 유지하고 승인 TRAIN에서만 원본 embedding 보존을 실험한다.

보존 계수는 고정 TRAIN 8 batch의 gradient 비로 평가 전에 봉인한다.
V25 데이터·schedule·학습 예산을 재사용하며 제품 checkpoint는 교체하지 않는다.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np
import run_hwr_direct_augmented_main_v25 as direct

OUT=direct.broad.previous.ROOT / "artifacts/hwr_direct_feature_retention_20261005_v27"
BASE=direct.OUT / "augmented_main/main_trained.pt"


def load():
    """봉인한 코드·TRAIN schedule·기준 모델을 검증한다."""
    p=json.loads((OUT / "frozen_plan.json").read_text(encoding="utf-8"))
    assert direct.broad.previous._sha(Path(__file__))==p["code_sha256"]
    assert direct.broad.previous._sha(direct.OUT / "frozen_plan.json")==p["parent_plan_sha256"]
    assert direct.broad.previous._sha(BASE)==p["direct_baseline_sha256"]
    parent,a,s=direct.load()
    assert parent["canonical_sha256"]==p["canonical_sha256"]
    return p,a,s


def models():
    """학습 가능한 main과 gradient를 받지 않는 원본 teacher를 분리한다."""
    import torch
    from run_hwr_affine_distillation_experiment_v1 import _load_teacher
    model,labels,_=_load_teacher(direct.broad.previous.CHECKPOINT,torch.device("cpu"))
    teacher,teacher_labels,_=_load_teacher(direct.broad.previous.CHECKPOINT,torch.device("cpu"))
    assert labels==teacher_labels
    model.eval();teacher.eval();teacher.requires_grad_(False)
    assert len(list(model.parameters()))==57 and all(p.requires_grad for p in model.parameters())
    return model,teacher,labels


def prepare():
    """평가 입력 없이 TRAIN gradient 비로 보존 계수를 정하고 봉인한다."""
    if OUT.exists():
        raise FileExistsError("refusing retention experiment overwrite")
    import torch
    from torch.nn import functional as F
    from run_hwr_affine_distillation_experiment_v1 import _load_teacher
    parent,a,s=direct.load()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    drifted,_,_=_load_teacher(BASE,torch.device("cpu"));drifted.eval()
    _,teacher,_=models()
    encoder=[p for n,p in drifted.named_parameters() if not n.startswith("math_head.")]
    assert len(encoder)==55
    rows=[]
    for step in np.linspace(0,direct.STEPS-1,8,dtype=int):
        x,y=direct.batch(a,s,int(step),"augmented_main");tx=torch.from_numpy(x)
        h=drifted.encode(tx);ce=F.cross_entropy(drifted.math_head(h),torch.from_numpy(y))
        with torch.no_grad():
            target=teacher.encode(tx)
        feature=F.mse_loss(h,target)
        cg=torch.autograd.grad(ce,encoder,retain_graph=True)
        fg=torch.autograd.grad(feature,encoder)
        cn=float(torch.sqrt(sum(g.square().sum() for g in cg)))
        fn=float(torch.sqrt(sum(g.square().sum() for g in fg)))
        assert np.isfinite(cn) and np.isfinite(fn) and fn>0
        rows.append(dict(schedule_step=int(step),ce=float(ce.detach()),feature_mse=float(feature.detach()),
            ce_encoder_gradient_l2=cn,feature_encoder_gradient_l2=fn,ratio=cn/fn))
    raw=float(np.median([r["ratio"] for r in rows]));weight=float(np.clip(raw,.1,100.))
    OUT.mkdir()
    plan=dict(schema="aiflow-direct-feature-retention-plan/v27",code_sha256=direct.broad.previous._sha(Path(__file__)),
        parent_plan_sha256=direct.broad.previous._sha(direct.OUT / "frozen_plan.json"),
        canonical_sha256=parent["canonical_sha256"],direct_baseline_sha256=direct.broad.previous._sha(BASE),
        steps=direct.STEPS,batch_size=direct.BATCH,original_rows=direct.REAL,augmented_rows=direct.AUG,
        same_v25_data_and_schedule=True,all_57_main_tensors_trainable=True,hard_ce_weight=1.,kl_weight=0.,
        feature_weight=weight,unclamped_feature_weight=raw,coefficient_rule="median CE/MSE encoder-gradient norm on eight fixed TRAIN batches, clamp [0.1,100]",
        calibration_rows=rows,coefficient_evaluation_inputs_used=0,
        objective="mean inherited-label hard CE + frozen coefficient * MSE(student embedding, canonical embedding on SAME TRAIN input)",
        cold_initialization="canonical main, all tensors trainable",lr=1e-5,weight_decay=1e-4,gradient_clip=1.,
        seed=direct.SEED,threads=2,development_folds=[1,2],no_interim_evaluation=True,no_best_epoch_selection=True,
        teacher_requires_grad=False,generated_labels_human_verified=False,label_inheritance_assumption=True,
        owned_optimizer_rows=0,crohme_rows=0,all_evaluation_cases_already_consumed=True,
        historical_canonical_calibration_formulas_in_owned_diagnostic=95,additional_codex_reviewed_annotations=53,
        product_adopted=False,collection_or_deployment_resumed=False)
    direct.broad.previous._write(OUT / "frozen_plan.json",plan)
    print(json.dumps(dict(event="retention_weight_sealed",feature_weight=weight,unclamped=raw,
        calibration_batches=8,evaluation_inputs_used=0)),flush=True)
    return 0


def train():
    """직접 증강 CE와 표현 보존 손실을 분리 기록하며 전체 main을 학습한다."""
    if direct.broad.previous.aug.base._guard_commit("direct_augmented_feature_retention_v27") is None:
        return 78
    plan,a,s=load();folder=OUT / "feature_retained_main"
    if folder.exists():
        raise FileExistsError("run exists; inspect terminal state instead of restarting")
    folder.mkdir();direct.broad.previous._write(folder / "run_started.json",dict(process_id=os.getpid(),status="started",
        plan_sha256=direct.broad.previous._sha(OUT / "frozen_plan.json")))
    os.environ["TRACKIO_DIR"]=str(folder / "trackio")
    for key in ("TRACKIO_WEBHOOK_URL","TRACKIO_SPACE_ID","TRACKIO_SERVER_URL"):
        os.environ.pop(key,None)
    import torch,trackio
    from torch.nn import functional as F
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.manual_seed(plan["seed"])
    model,teacher,labels=models();optimizer=torch.optim.AdamW(model.parameters(),lr=plan["lr"],weight_decay=plan["weight_decay"])
    layers={};handles=[]
    def capture(name):
        """원본 teacher가 아닌 학습 main의 4개 층 출력을 검사한다."""
        def hook(module,inputs,value):
            value=value.detach();assert torch.isfinite(value).all()
            layers[name]=dict(mean=float(value.mean()),std=float(value.std()))
        return hook
    for name,module in model.named_modules():
        if isinstance(module,torch.nn.TransformerEncoderLayer):
            handles.append(module.register_forward_hook(capture(name)))
    trackio.init(project="aiflow-direct-feature-retention-v27",name="feature_retained_main",space_id=None,embed=False,
        auto_log_cpu=False,auto_log_gpu=False,config=dict(steps=direct.STEPS,hard_ce_weight=1.,feature_weight=plan["feature_weight"]))
    updates=0
    try:
        with (folder / "training_microscope.jsonl").open("x",encoding="utf-8") as stream:
            for step in range(direct.STEPS):
                x,y=direct.batch(a,s,step,"augmented_main");tx=torch.from_numpy(x);truth=torch.from_numpy(y)
                optimizer.zero_grad(set_to_none=True)
                h=model.encode(tx);z=model.math_head(h)
                with torch.no_grad():
                    target=teacher.encode(tx)
                assert not target.requires_grad and target.grad_fn is None
                ce=F.cross_entropy(z,truth);feature=F.mse_loss(h,target)
                real=F.cross_entropy(z[:direct.REAL],truth[:direct.REAL]);aug=F.cross_entropy(z[direct.REAL:],truth[direct.REAL:])
                loss=ce+plan["feature_weight"]*feature;assert torch.isfinite(loss)
                if step==0:
                    gz=torch.autograd.grad(loss,z,retain_graph=True)[0]
                    expected=(z.detach().softmax(-1)-F.one_hot(truth,372))/direct.BATCH
                    assert torch.allclose(gz,expected,atol=1e-7) and float(feature.detach())<1e-10
                    print(json.dumps(dict(event="retention_selftest",hard_ce_logit_gradient_full_weight=True,
                        teacher_detached=True,initial_feature_mse=float(feature.detach()),initial_feature_mse_tolerance=1e-10)),flush=True)
                loss.backward()
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
                assert all(p.grad is None and not p.requires_grad for p in teacher.parameters())
                gradients={n:float(p.grad.norm()) for n,p in model.named_parameters()}
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),plan["gradient_clip"]);assert torch.isfinite(norm)
                optimizer.step();updates+=1
                numeric=dict(loss=float(loss.detach()),hard_ce=float(ce.detach()),feature_mse=float(feature.detach()),
                    weighted_feature=float((plan["feature_weight"]*feature).detach()),feature_weight=plan["feature_weight"],
                    ce_real_region=float(real.detach()),ce_augmented_region=float(aug.detach()),gradient_l2=float(norm),
                    hard_ce_weight=1.,kl_weight=0.,augmented_rows=direct.AUG,hard_target_rows=direct.BATCH)
                stream.write(json.dumps(dict(step=step+1,**numeric,encoder_layers=copy.deepcopy(layers),
                    parameter_gradient_l2=gradients,teacher_gradients_absent=True))+"\n");stream.flush()
                trackio.log(numeric,step=step+1)
                if (step+1)%32==0:
                    print(json.dumps(dict(event="feature_retention_training",step=step+1,**numeric)),flush=True)
    except Exception as e:
        direct.broad.previous._write(folder / "failure.json",dict(status="terminal_failure",error=str(e),completed_optimizer_steps=updates))
        trackio.alert(title="training_failure",text=str(e),level=trackio.AlertLevel.ERROR)
        raise
    finally:
        for handle in handles:
            handle.remove()
        trackio.finish()
    checkpoint=folder / "main_trained.pt"
    torch.save(dict(schema="aiflow-direct-feature-retained-main/v27",state_dict=model.state_dict(),math_labels=labels,auxiliary_labels=[],
        report=dict(input_contract=dict(observed_channel_mode="uniform-time",math_observed_transform="uniform-time"),
            product_adopted=False,direct_augmented_ce=True,feature_retention_weight=plan["feature_weight"],label_inheritance_assumption=True)),checkpoint)
    direct.broad.previous._write(folder / "completed.json",dict(status="completed",steps=updates,
        checkpoint_sha256=direct.broad.previous._sha(checkpoint),microscope_sha256=direct.broad.previous._sha(folder / "training_microscope.jsonl"),
        canonical_unchanged=direct.broad.previous._sha(direct.broad.previous.CHECKPOINT)==plan["canonical_sha256"],evaluation_forwarded_during_training=0))
    print(json.dumps(dict(event="feature_retention_completed",steps=updates)),flush=True)
    return 0


def evaluate():
    """새 학습 모델만 재실행하고 검증된 V25 기준 수치와 같은 입력에서 비교한다."""
    if (OUT / "comparison_result.json").exists():
        raise FileExistsError("comparison exists")
    plan,_,_=load();folder=OUT / "feature_retained_main"
    done=json.loads((folder / "completed.json").read_text(encoding="utf-8"))
    assert done["steps"]==direct.STEPS and direct.broad.previous._sha(folder / "main_trained.pt")==done["checkpoint_sha256"]
    baseline=json.loads((direct.OUT / "comparison_result.json").read_text(encoding="utf-8"))
    certificate=json.loads((direct.OUT / "independent_verification.json").read_text(encoding="utf-8"))
    assert certificate["status"]=="pass" and certificate["comparison_sha256"]==direct.broad.previous._sha(direct.OUT / "comparison_result.json")
    import torch
    import audit_hwr_owned_formula_transfer_v21 as owned
    from audit_hwr_encoder_head_swap_v22 import records_from_scores
    from run_hwr_affine_distillation_experiment_v1 import _load_teacher,_predict_logits
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    model,labels,_=_load_teacher(folder / "main_trained.pt",torch.device("cpu"))
    _,data,folds,_=direct.broad.previous.global_load(direct.broad.previous.OUTPUT)
    development={}
    for fold in plan["development_folds"]:
        x=np.array(data["features"][folds[fold]],copy=True);y=np.array(data["labels"][folds[fold]],copy=True)
        z=_predict_logits(model,x,torch.device("cpu"),32)
        np.save(OUT / f"feature_retained_main_fold{fold}_logits.npy",z,allow_pickle=False)
        development[str(fold)]=direct.broad.previous.hit_metrics(z,y,labels)
    _,samples,rows=owned.load_source();torch.set_num_threads(1)
    z=_predict_logits(model,np.load(owned.OUT / "inputs.npy",allow_pickle=False),torch.device("cpu"),owned.BATCH)
    records=records_from_scores(z,labels,samples,rows)
    np.save(OUT / "feature_retained_main_owned_logits.npy",z,allow_pickle=False)
    direct.broad.previous._write(OUT / "feature_retained_main_owned_records.json",records)
    metrics=copy.deepcopy(baseline["metrics"])
    metrics["feature_retained_main"]=dict(development=development,owned={c:owned.metrics(records,c) for c in ("all","legacy_96","codex_reviewed_53")})
    result=dict(schema="aiflow-direct-feature-retention-comparison/v27",status="completed_consumed_diagnostics_not_fresh_acceptance",
        baseline_comparison_sha256=certificate["comparison_sha256"],metrics=metrics,hard_ce_weight=1.,feature_weight=plan["feature_weight"],
        all_57_main_tensors_optimized=True,all_evaluation_cases_already_consumed=True,generated_labels_human_verified=False,
        owned_optimizer_rows=0,crohme_rows=0,product_adopted=False,
        canonical_unchanged=direct.broad.previous._sha(direct.broad.previous.CHECKPOINT)==plan["canonical_sha256"])
    direct.broad.previous._write(OUT / "comparison_result.json",result)
    print(json.dumps(dict(event="feature_retained_main_scored",metrics=metrics["feature_retained_main"])),flush=True)
    return 0


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("mode",choices=("prepare","train","evaluate"));args=parser.parse_args()
    raise SystemExit(dict(prepare=prepare,train=train,evaluate=evaluate)[args.mode]())
