"""사용자 지시대로 증강 라벨 직접 CE로 현재 메인 HWR 전체를 학습한다.

원본 16개+실제 변형 48개의 batch와 동일 원본-only 대조군을 비교한다.
승계 라벨은 새 사람 정답이 아니며, 의미 보존 가정을 명시한다.
진단/CROHME는 optimizer에 넣지 않고 제품 원본 checkpoint를 보존한다.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np
import run_hwr_broad_diversity_v16 as broad

OUT=broad.previous.ROOT / "artifacts/hwr_direct_augmented_main_20261005_v25"
ARMS=("real_paired_control","augmented_main")
STEPS,BATCH,REAL,AUG,SEED=2400,64,16,48,2026100525


def prepare():
    """비영 변형을 모두 활용하고 두 군의 원본 index·GT·예산을 동일하게 봉인한다."""
    if OUT.exists():
        raise FileExistsError("refusing direct-training overwrite")
    parent,manifest,a=broad.load()
    proof=json.loads((broad.OUT / "independent_data_verification.json").read_text(encoding="utf-8"))
    assert proof["status"]=="reproduced" and proof["geometry_errors"]==0 and proof["all_scheduled_endpoints_rebuilt_bit_exact"]
    provenance=json.loads((broad.OUT / "generation_provenance.json").read_text(encoding="utf-8"))
    eligible=np.array([i for i,m in enumerate(provenance) if m["changed"]],dtype=np.int64)
    assert len(eligible)==68726
    source=a["shared_schedule"].reshape(-1)
    labels=a["population_labels"]
    for i in eligible:
        m=provenance[int(i)];index=int(source[i]);donor=m["donor_population_index"]
        assert index==m["population_index"] and donor is not None and labels[index]==labels[int(donor)]
        assert broad.previous.aug.base._signature(a["population_features"][index])==broad.previous.aug.base._signature(a["population_features"][int(donor)])
    rng=np.random.default_rng(SEED)
    augmented=np.resize(rng.permutation(eligible),STEPS*AUG).reshape(STEPS,AUG)
    original=np.resize(rng.permutation(source),STEPS*REAL).reshape(STEPS,REAL)
    targets=np.concatenate((labels[original],labels[source[augmented]]),axis=1).astype(np.int64)
    assert targets.shape==(STEPS,BATCH) and np.unique(targets).size==371
    OUT.mkdir()
    arrays=dict(original_population_indices=original,augmented_view_indices=augmented,shared_target_ids=targets)
    hashes={}
    for name,value in arrays.items():
        np.save(OUT / f"{name}.npy",value,allow_pickle=False);hashes[name]=broad.previous._sha(OUT / f"{name}.npy")
    classes=parent["class_labels"];chosen=[]
    for family in ("digits","latin_letters","math_symbols"):
        seen=set()
        for i in eligible:
            c=int(labels[source[i]])
            if c not in seen and broad.previous.aug.base._family(classes[c])==family:
                chosen.append(int(i));seen.add(c)
                if len(seen)==10:
                    break
    images=[];names=[];qa_rows=[];views=a["scheduled_augmented_features"].reshape(-1,128,5)
    for i in chosen:
        original_row=a["population_features"][source[i]];view=views[i]
        assert broad.previous.aug.base._geometry(original_row,view)["valid"]
        label=classes[int(labels[source[i]])]
        images.extend((original_row,view));names.extend((f"{label} orig",f"{label} aug"))
        qa_rows.append(dict(view_index=i,class_label=label,source_class_id=int(labels[source[i]]),
            shape_temperature=provenance[i]["shape_temperature"],donor_population_index=provenance[i]["donor_population_index"]))
    broad.previous.aug.base._render(np.stack(images),names,OUT / "direct_training_qa.png")
    broad.previous._write(OUT / "qa_rows.json",qa_rows)
    modules=("run_hwr_direct_augmented_main_v25.py","run_hwr_broad_diversity_v16.py","train_character_classifier_v1.py",
        "run_hwr_affine_distillation_experiment_v1.py","audit_hwr_owned_formula_transfer_v21.py","audit_hwr_encoder_head_swap_v22.py")
    plan=dict(schema="aiflow-direct-augmented-main-plan/v25",code_sha256={n:broad.previous._sha(Path(__file__).parent/n) for n in modules},
        parent_plan_sha256=broad.previous._sha(broad.OUT / "frozen_plan.json"),parent_manifest_sha256=broad.previous._sha(broad.OUT / "prepared_manifest.json"),
        data_certificate_sha256=broad.previous._sha(broad.OUT / "independent_data_verification.json"),canonical_sha256=parent["canonical_sha256"],
        arrays_sha256=hashes,qa_image_sha256=broad.previous._sha(OUT / "direct_training_qa.png"),qa_pairs=len(chosen),
        user_authorization="User explicitly requested substantial augmented data training of the main model, followed by results; direct inherited-label CE, not KL-only.",
        architecture="same main 4-layer Transformer+pooling+372-class head; all 57 tensors optimized",cold_initialization="current canonical main; original file not replaced",
        arms=ARMS,steps=STEPS,batch_size=BATCH,original_rows_per_batch=REAL,augmented_rows_per_batch=AUG,
        total_exposures=STEPS*BATCH,changed_augmented_exposures=STEPS*AUG,unique_augmented_views=len(eligible),augmented_classes=366,
        positive_target_classes=371,missing_positive_target_labels=[classes[c] for c in range(372) if c not in np.unique(targets)],
        objective="mean hard-label CrossEntropy over 64 rows; KL/teacher terms zero; 75% augmented rows in augmented_main",
        control="same source population indices, same labels, same steps and optimizer; substitute corresponding original ink for each augmented row",
        label_source="original admitted TRAIN ground-truth class inherited through same-class/same-topology constrained deformation",
        label_invariance_assumption=True,generated_labels_human_verified=False,synthetic_hard_label_exposures=STEPS*AUG,
        learning_rate=1e-5,weight_decay=1e-4,gradient_clip=1.,threads=2,development_folds=[1,2],owned_formulas=149,
        all_evaluation_cases_already_consumed=True,no_interim_evaluation=True,no_best_epoch_selection=True,
        owned_training_rows=0,crohme_rows=0,official_test_rows_read=0,product_adopted=False,collection_or_deployment_resumed=False)
    broad.previous._write(OUT / "frozen_plan.json",plan)
    print(json.dumps(dict(event="direct_data_prepared",augmented_views=len(eligible),augmented_exposures=STEPS*AUG,
        total_exposures=STEPS*BATCH,classes=371,qa_pairs=len(chosen),visual_review_required=True)),flush=True)
    return 0


def load(require_visual=True):
    """コード・GT schedule・元データ・画像検証を確認して既存実験を上書きしない。"""
    p=json.loads((OUT / "frozen_plan.json").read_text(encoding="utf-8"))
    assert all(broad.previous._sha(Path(__file__).parent/n)==h for n,h in p["code_sha256"].items())
    assert broad.previous._sha(broad.OUT / "frozen_plan.json")==p["parent_plan_sha256"]
    assert broad.previous._sha(broad.OUT / "prepared_manifest.json")==p["parent_manifest_sha256"]
    assert broad.previous._sha(OUT / "direct_training_qa.png")==p["qa_image_sha256"]
    arrays={}
    for n,h in p["arrays_sha256"].items():
        assert broad.previous._sha(OUT / f"{n}.npy")==h
        arrays[n]=np.load(OUT / f"{n}.npy",mmap_mode="r",allow_pickle=False)
    if require_visual:
        visual=json.loads((OUT / "visual_review.json").read_text(encoding="utf-8"))
        assert visual["assistant_visual_qa_completed"] and visual["image_sha256"]==p["qa_image_sha256"]
        assert visual["human_label_approval"] is False
    _,_,a=broad.load()
    return p,a,arrays


def batch(a,schedules,step,arm):
    """두 군에서 GT와 원본 대응 index가 같고 입력 좌표만 달라지는 16+48 batch를 만든다."""
    r=schedules["original_population_indices"][step];v=schedules["augmented_view_indices"][step]
    parent=a["shared_schedule"].reshape(-1)[v]
    first=np.array(a["population_features"][r],copy=True)
    last=np.array(a["scheduled_augmented_features"].reshape(-1,128,5)[v],copy=True) if arm=="augmented_main" else np.array(a["population_features"][parent],copy=True)
    y=np.array(schedules["shared_target_ids"][step],dtype=np.int64,copy=True)
    assert np.array_equal(y,np.concatenate((a["population_labels"][r],a["population_labels"][parent])))
    return np.concatenate((first,last)),y


def selftest():
    """직접 CE가 승계 정답 방향으로 미분되고 실제 main 57개 tensor에 gradient가 전달되는지 검사한다."""
    import torch
    from torch.nn import functional as F
    from run_hwr_affine_distillation_experiment_v1 import _load_teacher
    _,a,s=load(False);x,y=batch(a,s,0,"augmented_main")
    control,cy=batch(a,s,0,"real_paired_control")
    assert np.array_equal(y,cy) and np.array_equal(x[:REAL],control[:REAL]) and (x[REAL:]!=control[REAL:]).any(axis=(1,2)).all()
    torch.set_num_threads(2)
    model,_,_=_load_teacher(broad.previous.CHECKPOINT,torch.device("cpu"));model.eval()
    z=model.math_head(model.encode(torch.from_numpy(x)));truth=torch.from_numpy(y)
    loss=F.cross_entropy(z,truth)
    assert torch.allclose(loss,.25*F.cross_entropy(z[:REAL],truth[:REAL])+.75*F.cross_entropy(z[REAL:],truth[REAL:]),atol=1e-6)
    gz=torch.autograd.grad(loss,z,retain_graph=True)[0]
    assert (gz[torch.arange(BATCH),truth]<0).all()
    loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    print(json.dumps(dict(selftest="pass",direct_ce=True,gt_pushes_correct_class=True,all_57_main_tensors_receive_gradients=True,
        paired_labels_identical=True,all_48_augmented_rows_nonzero=True)),flush=True)
    return 0


def train(arm):
    """현재 main 전체를 직접 supervised CE로 학습하고 각 step의 층·57개 gradient를 기록한다."""
    if broad.previous.aug.base._guard_commit(f"direct_supervised_main_{arm}") is None:
        return 78
    plan,a,s=load();folder=OUT / arm
    if folder.exists():
        raise FileExistsError("arm already started; inspect live/terminal state before restart")
    folder.mkdir();broad.previous._write(folder / "run_started.json",dict(process_id=os.getpid(),status="started",plan_sha256=broad.previous._sha(OUT / "frozen_plan.json")))
    os.environ["TRACKIO_DIR"]=str(folder / "trackio")
    for key in ("TRACKIO_WEBHOOK_URL","TRACKIO_SPACE_ID","TRACKIO_SERVER_URL"):
        os.environ.pop(key,None)
    import torch,trackio
    from torch.nn import functional as F
    from run_hwr_affine_distillation_experiment_v1 import _load_teacher
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    model,labels,_=_load_teacher(broad.previous.CHECKPOINT,torch.device("cpu"));model.eval()
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=1e-4)
    layers={};handles=[]
    def capture(name):
        """各層의 실제 출력 유한성과 평균·표준편차를 기록한다."""
        def hook(module,inputs,value):
            value=value.detach();assert torch.isfinite(value).all()
            layers[name]=dict(mean=float(value.mean()),std=float(value.std()))
        return hook
    for name,module in model.named_modules():
        if isinstance(module,torch.nn.TransformerEncoderLayer):
            handles.append(module.register_forward_hook(capture(name)))
    trackio.init(project="aiflow-direct-augmented-main-v25",name=arm,space_id=None,embed=False,auto_log_cpu=False,auto_log_gpu=False,
        config=dict(steps=STEPS,lr=1e-5,direct_ce=True,augmented_fraction=.75 if arm=="augmented_main" else 0.))
    updates=0
    try:
        with (folder / "training_microscope.jsonl").open("x",encoding="utf-8") as stream:
            for step in range(STEPS):
                x,y=batch(a,s,step,arm);optimizer.zero_grad(set_to_none=True)
                z=model.math_head(model.encode(torch.from_numpy(x)));truth=torch.from_numpy(y)
                real=F.cross_entropy(z[:REAL],truth[:REAL]);region=F.cross_entropy(z[REAL:],truth[REAL:])
                loss=F.cross_entropy(z,truth)
                assert torch.isfinite(loss)
                loss.backward()
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
                gradients={n:float(p.grad.norm()) for n,p in model.named_parameters()}
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm)
                optimizer.step();updates+=1
                numeric=dict(loss=float(loss.detach()),ce_real_region=float(real.detach()),ce_aug_or_paired_region=float(region.detach()),
                    gradient_l2=float(norm),augmented_rows=AUG if arm=="augmented_main" else 0,hard_target_rows=BATCH,kl_weight=0.)
                stream.write(json.dumps(dict(step=step+1,**numeric,encoder_layers=copy.deepcopy(layers),parameter_gradient_l2=gradients))+"\n")
                stream.flush();trackio.log(numeric,step=step+1)
                if (step+1)%32==0:
                    print(json.dumps(dict(event="direct_ce_training",arm=arm,step=step+1,**numeric)),flush=True)
    except Exception as e:
        broad.previous._write(folder / "failure.json",dict(status="terminal_failure",error=str(e),completed_optimizer_steps=updates))
        trackio.alert(title="training_failure",text=str(e),level=trackio.AlertLevel.ERROR)
        raise
    finally:
        for h in handles:
            h.remove()
        trackio.finish()
    checkpoint=folder / "main_trained.pt"
    torch.save(dict(schema="aiflow-direct-augmented-main/v25",state_dict=model.state_dict(),math_labels=labels,auxiliary_labels=[],
        report=dict(input_contract=dict(observed_channel_mode="uniform-time",math_observed_transform="uniform-time"),product_adopted=False,
            direct_augmented_ce=arm=="augmented_main",label_inheritance_assumption=True)),checkpoint)
    broad.previous._write(folder / "completed.json",dict(status="completed",steps=updates,checkpoint_sha256=broad.previous._sha(checkpoint),
        microscope_sha256=broad.previous._sha(folder / "training_microscope.jsonl"),evaluation_forwarded_during_training=0,
        canonical_unchanged=broad.previous._sha(broad.previous.CHECKPOINT)==plan["canonical_sha256"]))
    print(json.dumps(dict(event="direct_ce_main_completed",arm=arm,steps=updates)),flush=True)
    return 0


def evaluate():
    """두 군이 끝난 뒤 기존 main과 개발/소유 입력을 비교하고 이미 소비된 평가임을 표시한다."""
    if (OUT / "comparison_result.json").exists():
        raise FileExistsError("comparison exists")
    plan,_,_=load()
    for arm in ARMS:
        done=json.loads((OUT / arm / "completed.json").read_text(encoding="utf-8"))
        assert done["steps"]==STEPS and broad.previous._sha(OUT / arm / "main_trained.pt")==done["checkpoint_sha256"]
    import torch
    import audit_hwr_owned_formula_transfer_v21 as owned
    from audit_hwr_encoder_head_swap_v22 import records_from_scores
    from run_hwr_affine_distillation_experiment_v1 import _load_teacher,_predict_logits
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    _,a,folds,_=broad.previous.global_load(broad.previous.OUTPUT)
    _,samples,rows=owned.load_source();paths=dict(canonical=broad.previous.CHECKPOINT,**{arm:OUT / arm / "main_trained.pt" for arm in ARMS})
    metrics={}
    for name,path in paths.items():
        model,labels,_=_load_teacher(path,torch.device("cpu"));development={}
        torch.set_num_threads(2)
        for fold in plan["development_folds"]:
            y=np.array(a["labels"][folds[fold]],copy=True)
            z=_predict_logits(model,np.array(a["features"][folds[fold]],copy=True),torch.device("cpu"),32)
            np.save(OUT / f"{name}_fold{fold}_logits.npy",z,allow_pickle=False)
            development[str(fold)]=broad.previous.hit_metrics(z,y,labels)
        torch.set_num_threads(1)
        z=_predict_logits(model,np.load(owned.OUT / "inputs.npy",allow_pickle=False),torch.device("cpu"),owned.BATCH)
        records=records_from_scores(z,labels,samples,rows)
        np.save(OUT / f"{name}_owned_logits.npy",z,allow_pickle=False)
        broad.previous._write(OUT / f"{name}_owned_records.json",records)
        metrics[name]=dict(development=development,owned={c:owned.metrics(records,c) for c in ("all","legacy_96","codex_reviewed_53")})
        print(json.dumps(dict(event="direct_main_scored",model=name,metrics=metrics[name])),flush=True)
    result=dict(schema="aiflow-direct-main-comparison/v25",status="completed_consumed_development_not_fresh_acceptance",metrics=metrics,
        direct_augmented_hard_ce=True,all_57_main_tensors_optimized=True,all_evaluation_cases_already_consumed=True,
        generated_labels_human_verified=False,crohme_rows=0,owned_optimizer_rows=0,product_adopted=False,
        canonical_unchanged=broad.previous._sha(broad.previous.CHECKPOINT)==plan["canonical_sha256"])
    broad.previous._write(OUT / "comparison_result.json",result)
    return 0


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("mode",choices=("prepare","selftest","train","evaluate"));parser.add_argument("--arm",choices=ARMS)
    args=parser.parse_args()
    if args.mode=="train":
        if args.arm is None:
            parser.error("train requires --arm")
        raise SystemExit(train(args.arm))
    raise SystemExit(dict(prepare=prepare,selftest=selftest,evaluate=evaluate)[args.mode]())
