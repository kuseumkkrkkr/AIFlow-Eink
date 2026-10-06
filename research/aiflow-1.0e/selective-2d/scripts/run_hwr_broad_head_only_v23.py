"""넓은 V16 학습의 데이터·증강·예산을 유지하고 두 선형 head 텐서만 학습한다.

149식 acceptance-derived 입력은 optimizer에 넣지 않는다. 이미 소비된 개발
fold와 소유 입력은 고정 학습 종료 후 진단만 하며 제품/CROHME 상태는 유지한다.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np
import run_hwr_broad_diversity_v16 as broad
import run_hwr_active_trust_head_only_v11 as detach

OUT = broad.previous.ROOT / "artifacts/hwr_broad_head_only_20261005_v23"


def selftest():
    """작은 검증과 실제 TRAIN batch에서 값 불변·55개 unused·두 head gradient를 검사한다."""
    detach.self_test()
    import torch
    from run_hwr_affine_distillation_experiment_v1 import _load_teacher
    _,_,a = broad.load()
    torch.set_num_threads(2)
    model,_,_ = _load_teacher(broad.previous.CHECKPOINT,torch.device("cpu"));model.eval()
    ids = a["shared_schedule"][0,:4]
    x = torch.from_numpy(np.array(a["population_features"][ids],copy=True))
    before = model.math_head(model.encode(x)).detach().clone()
    detach.detach_encoder(model)
    after = model.math_head(model.encode(x))
    assert torch.equal(before,after.detach())
    torch.nn.functional.cross_entropy(after,torch.from_numpy(np.array(a["population_labels"][ids],dtype=np.int64,copy=True))).backward()
    named = dict(model.named_parameters())
    assert len(named)==57 and all(p.grad is None for name,p in named.items() if name not in detach.HEAD_NAMES)
    assert all(named[name].grad is not None and torch.isfinite(named[name].grad).all() for name in detach.HEAD_NAMES)
    print(json.dumps(dict(actual_model_selftest="pass",forward_bit_exact=True,frozen_tensors=55,head_gradient_tensors=2)),flush=True)
    return 0


def prepare():
    """한 가지 미분 범위 변경과 두 개발 fold·소유 입력 비회귀 비교를 학습 전에 고정한다."""
    if OUT.exists():
        raise FileExistsError("refusing head-only experiment overwrite")
    parent,manifest,_ = broad.load()
    diagnosis = broad.previous.ROOT / "artifacts/hwr_encoder_head_swap_20261005_v22/component_result.json"
    d = json.loads(diagnosis.read_text(encoding="utf-8"))
    assert d["all_nine_full_forwards_match_independent_affine_top5"] and d["checkpoint_files_unchanged"]
    control = broad.OUT / "broad_augmented/research.pt"
    complete = json.loads((broad.OUT / "broad_augmented/completed.json").read_text(encoding="utf-8"))
    assert complete["steps"]==broad.STEPS and broad.previous._sha(control)==complete["checkpoint_sha256"]
    OUT.mkdir()
    broad.previous._write(OUT / "frozen_plan.json",dict(schema="aiflow-broad-head-only-plan/v23",
        script_sha256=broad.previous._sha(Path(__file__)),helper_sha256={Path(module.__file__).name:broad.previous._sha(Path(module.__file__)) for module in (broad,detach)},
        code_sha256={n:broad.previous._sha(Path(__file__).parent/n) for n in ("hwr_boundary_distillation_v1.py",
            "run_hwr_affine_distillation_experiment_v1.py","train_character_classifier_v1.py","evaluate_48hz_prefix_v1.py",
            "audit_hwr_owned_formula_transfer_v21.py","audit_hwr_encoder_head_swap_v22.py")},
        parent_plan_sha256=broad.previous._sha(broad.OUT / "frozen_plan.json"),parent_manifest_sha256=broad.previous._sha(broad.OUT / "prepared_manifest.json"),
        diagnosis_sha256=broad.previous._sha(diagnosis),canonical_sha256=parent["canonical_sha256"],control_sha256=complete["checkpoint_sha256"],
        controlled_change="detach encode output; update only math_head.weight and math_head.bias; all 55 encoder/pooling tensors bitwise fixed",
        cold_initialization="same canonical as V16; no warm start, regeneration, extra steps, or new architecture",
        unchanged_objective=".7 CE(real) + .2 KL(real,canonical,T=2) + .1 KL(view,original-canonical,T=2)",
        steps=broad.STEPS,batch_size=broad.BATCH,forward_rows=64,threads=2,learning_rate=1e-5,weight_decay=1e-4,gradient_clip=1.,
        classes=manifest["classes"],population_rows=manifest["population_rows"],real_exposures=broad.STEPS*broad.BATCH,
        head_names=list(detach.HEAD_NAMES),nonhead_tensors=55,development_folds=[1,2],all_development_folds_previously_consumed=True,
        owned_diagnostic_formulae=149,owned_codex_reviewed_formulae=53,owned_inputs_or_labels_in_optimizer=0,
        no_interim_evaluation=True,no_best_epoch_selection=True,development_selection_bias=True,
        gate="report both internal folds and owned token/formula Top1/5; include regressions, no product adoption or fresh acceptance claim",
        synthetic_hard_labels=0,crohme_rows=0,official_test_rows_read=0,collection_or_deployment_resumed=False,product_adopted=False,cloud_upload=False))
    print(json.dumps(dict(event="head_only_plan_frozen",steps=broad.STEPS,classes=manifest["classes"],optimizer_tensors=2,owned_training_rows=0)),flush=True)
    return 0


def load():
    """봉인된 코드·원본 배열·대조군 SHA를 확인하며 진단 입력을 학습용으로 읽지 않는다."""
    p = json.loads((OUT / "frozen_plan.json").read_text(encoding="utf-8"))
    assert p["script_sha256"]==broad.previous._sha(Path(__file__))
    assert all(p["helper_sha256"][Path(m.__file__).name]==broad.previous._sha(Path(m.__file__)) for m in (broad,detach))
    assert all(broad.previous._sha(Path(__file__).parent/n)==digest for n,digest in p["code_sha256"].items())
    assert p["parent_plan_sha256"]==broad.previous._sha(broad.OUT / "frozen_plan.json")
    assert p["parent_manifest_sha256"]==broad.previous._sha(broad.OUT / "prepared_manifest.json")
    assert p["control_sha256"]==broad.previous._sha(broad.OUT / "broad_augmented/research.pt")
    _,_,a = broad.load()
    return p,a


def train():
    """동일 full forward와 loss에서 encoder 미분만 차단하고 매 단계 55개 비트를 확인한다."""
    if broad.previous.aug.base._guard_commit("broad_head_only_train_v23") is None:
        return 78
    plan,a = load()
    if (OUT / "run_started.json").exists():
        raise FileExistsError("already started; inspect actual live/terminal state before any restart")
    os.environ["TRACKIO_DIR"]=str(OUT / "trackio")
    for key in ("TRACKIO_WEBHOOK_URL","TRACKIO_SPACE_ID","TRACKIO_SERVER_URL"):
        os.environ.pop(key,None)
    import torch,trackio
    from torch.nn import functional as F
    from hwr_boundary_distillation_v1 import boundary_kl_loss
    from run_hwr_affine_distillation_experiment_v1 import _load_teacher
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    model,labels,_ = _load_teacher(broad.previous.CHECKPOINT,torch.device("cpu"));model.eval()
    named = dict(model.named_parameters());assert len(named)==57
    frozen = {n:p.detach().clone() for n,p in named.items() if n not in detach.HEAD_NAMES}
    assert len(frozen)==55
    detach.detach_encoder(model)
    optimizer = torch.optim.AdamW([named[n] for n in detach.HEAD_NAMES],lr=1e-5,weight_decay=1e-4)
    broad.previous._write(OUT / "run_started.json",dict(status="started",process_id=os.getpid(),plan_sha256=broad.previous._sha(OUT / "frozen_plan.json")))
    activation = {};handles = []
    def capture(name):
        """동결됐어도 실제로 계산된 각 층의 유한성과 분포 수치를 기록한다."""
        def hook(module,inputs,output):
            value=output.detach()
            assert torch.isfinite(value).all()
            activation[name]=dict(mean=float(value.mean()),std=float(value.std()))
        return hook
    for name,module in model.named_modules():
        if isinstance(module,torch.nn.TransformerEncoderLayer):
            handles.append(module.register_forward_hook(capture(name)))
    trackio.init(project="aiflow-broad-head-only-v23",name="head_only",space_id=None,embed=False,
        auto_log_cpu=False,auto_log_gpu=False,config=dict(steps=broad.STEPS,lr=1e-5,optimizer_tensors=2))
    updates=0
    try:
        with (OUT / "training_microscope.jsonl").open("x",encoding="utf-8") as stream:
            for step,ids in enumerate(a["shared_schedule"]):
                real=np.array(a["population_features"][ids],copy=True)
                view=np.array(a["scheduled_augmented_features"][step],copy=True)
                target=torch.from_numpy(np.array(a["population_teacher_logits"][ids],copy=True))
                truth=torch.from_numpy(np.array(a["population_labels"][ids],dtype=np.int64,copy=True))
                optimizer.zero_grad(set_to_none=True)
                embedding=model.encode(torch.from_numpy(np.concatenate((real,view))))
                assert not embedding.requires_grad
                z=model.math_head(embedding)
                ce=F.cross_entropy(z[:broad.BATCH],truth)
                retain=boundary_kl_loss(z[:broad.BATCH],target,2.,"full")
                consistent=boundary_kl_loss(z[broad.BATCH:],target,2.,"full")
                loss=.7*ce+.2*retain+.1*consistent
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite loss")
                loss.backward()
                assert all(p.grad is None for n,p in named.items() if n in frozen)
                assert all(named[n].grad is not None and torch.isfinite(named[n].grad).all() for n in detach.HEAD_NAMES)
                gradients={n:None if p.grad is None else float(p.grad.norm()) for n,p in named.items()}
                norm=torch.nn.utils.clip_grad_norm_([named[n] for n in detach.HEAD_NAMES],1.)
                assert torch.isfinite(norm)
                optimizer.step();updates += 1
                assert all(torch.equal(named[n].detach().view(torch.int32),v.view(torch.int32)) for n,v in frozen.items())
                numeric=dict(loss=float(loss.detach()),ce_real=float(ce.detach()),kl_real=float(retain.detach()),
                    kl_view=float(consistent.detach()),gradient_l2=float(norm),frozen_nonhead_tensors=55,all_nonhead_bit_equal=True)
                stream.write(json.dumps(dict(step=step+1,**numeric,encoder_layers=copy.deepcopy(activation),parameter_gradient_l2=gradients))+"\n")
                stream.flush();trackio.log(numeric,step=step+1)
                if (step+1)%32==0:
                    print(json.dumps(dict(event="broad_head_only_training",step=step+1,**numeric)),flush=True)
    except Exception as error:
        broad.previous._write(OUT / "failure.json",dict(status="terminal_failure",error=str(error),completed_optimizer_steps=updates))
        trackio.alert(title="training_failure",text=str(error),level=trackio.AlertLevel.ERROR)
        raise
    finally:
        for handle in handles:
            handle.remove()
        trackio.finish()
    checkpoint=OUT / "research.pt"
    torch.save(dict(schema="aiflow-broad-head-only-research/v23",state_dict=model.state_dict(),math_labels=labels,auxiliary_labels=[],
        report=dict(input_contract=dict(observed_channel_mode="uniform-time",math_observed_transform="uniform-time"),product_adopted=False)),checkpoint)
    broad.previous._write(OUT / "completed.json",dict(status="completed",steps=updates,
        checkpoint_sha256=broad.previous._sha(checkpoint),microscope_sha256=broad.previous._sha(OUT / "training_microscope.jsonl"),
        all_55_nonhead_tensors_bit_equal=True,development_or_owned_inputs_forwarded_during_training=0,
        canonical_unchanged=broad.previous._sha(broad.previous.CHECKPOINT)==plan["canonical_sha256"]))
    print(json.dumps(dict(event="broad_head_only_completed",steps=updates)),flush=True)
    return 0


def evaluate():
    """고정 학습 종료 뒤 소비된 개발 fold와 149식 진단을 모두 보고하여 전이를 숨기지 않는다."""
    if (OUT / "comparison_result.json").exists():
        raise FileExistsError("comparison exists")
    plan,_=load()
    done=json.loads((OUT / "completed.json").read_text(encoding="utf-8"))
    assert done["steps"]==broad.STEPS and broad.previous._sha(OUT / "research.pt")==done["checkpoint_sha256"]
    import torch
    import audit_hwr_owned_formula_transfer_v21 as owned
    from audit_hwr_encoder_head_swap_v22 import records_from_scores
    from run_hwr_affine_distillation_experiment_v1 import _load_teacher,_predict_logits
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    model,labels,_=_load_teacher(OUT / "research.pt",torch.device("cpu"))
    _,a,folds,_=broad.previous.global_load(broad.previous.OUTPUT)
    development={}
    for fold in plan["development_folds"]:
        y=np.array(a["labels"][folds[fold]],copy=True)
        z=_predict_logits(model,np.array(a["features"][folds[fold]],copy=True),torch.device("cpu"),broad.BATCH)
        np.save(OUT / f"head_only_fold{fold}_logits.npy",z,allow_pickle=False)
        old_path=broad.OUT / "broad_augmented_validation_logits.npy" if fold==1 else broad.previous.ROOT / "artifacts/hwr_broad_margin_20261005_v17/broad_augmented_fold2_logits.npy"
        old=np.load(old_path,allow_pickle=False);before=old.argmax(1)==y;after=z.argmax(1)==y
        development[str(fold)]=dict(v16=broad.previous.hit_metrics(old,y,labels),head_only=broad.previous.hit_metrics(z,y,labels),
            paired_top1=dict(wins=int((~before&after).sum()),losses=int((before&~after).sum()),net=int(after.sum()-before.sum())))
    _,samples,rows=owned.load_source()
    certificate=json.loads((owned.OUT / "independent_verification.json").read_text(encoding="utf-8"))
    assert broad.previous._sha(owned.OUT / "inputs.npy")==certificate["artifacts_sha256"]["inputs.npy"]
    torch.set_num_threads(1)
    z=_predict_logits(model,np.load(owned.OUT / "inputs.npy",allow_pickle=False),torch.device("cpu"),owned.BATCH)
    records=records_from_scores(z,labels,samples,rows)
    np.save(OUT / "head_only_owned_logits.npy",z,allow_pickle=False)
    broad.previous._write(OUT / "head_only_owned_records.json",records)
    owned_metrics={c:owned.metrics(records,c) for c in ("all","legacy_96","codex_reviewed_53")}
    before=json.loads((owned.OUT / "canonical_records.json").read_text(encoding="utf-8"))
    initial=np.array([r["top1_tokens"]==r["truth_tokens"] for r in before]);after=np.array([r["top1_tokens"]==r["truth_tokens"] for r in records])
    result=dict(schema="aiflow-broad-head-only-comparison/v23",status="consumed_development_and_owned_diagnostic_not_fresh_acceptance",
        development=development,owned=owned_metrics,owned_paired_vs_canonical=dict(wins=int((~initial&after).sum()),losses=int((initial&~after).sum())),
        frozen_plan_sha256=broad.previous._sha(OUT / "frozen_plan.json"),all_evaluation_cases_already_consumed=True,
        owned_training_rows=0,parameter_updates_in_evaluation=0,synthetic_hard_labels=0,crohme_rows=0,product_adopted=False)
    broad.previous._write(OUT / "comparison_result.json",result)
    print(json.dumps(result),flush=True)
    return 0


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=("selftest","prepare","train","evaluate"))
    args=parser.parse_args()
    raise SystemExit(dict(selftest=selftest,prepare=prepare,train=train,evaluate=evaluate)[args.mode]())
