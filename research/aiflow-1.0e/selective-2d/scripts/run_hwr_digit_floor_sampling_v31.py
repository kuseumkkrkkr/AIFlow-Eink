"""V29 손실·증강을 유지하고 TRAIN 비중만큼 실제 숫자 노출 부족분을 복원한다.

검증된 V29 trainer를 파일 수정 없이 재사용한다. 원본 16행 중 배치당 최대 한 행만
교체하며, 평가 정답·사람 경계 라벨·CROHME는 샘플 선택에 쓰지 않는다.
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
OUT=direct.broad.previous.ROOT / "artifacts/hwr_digit_floor_sampling_20261006_v31"
SEED=2026100631


def prepare():
    """숫자 목표량을 TRAIN 빈도로만 계산하고 기존 증강·다른 원본 행을 고정한다."""
    if OUT.exists():
        raise FileExistsError("refusing digit-floor overwrite")
    parent,a,s=algorithm.load();y=a["population_labels"]
    labels=json.loads((direct.broad.OUT / "frozen_plan.json").read_text(encoding="utf-8"))["class_labels"]
    digit_ids=np.array([labels.index(str(i)) for i in range(10)],dtype=np.int64)
    population=np.flatnonzero(np.isin(y,digit_ids));target=(len(population)*s["shared_target_ids"].size+len(y)-1)//len(y)
    old_count=int(np.isin(s["shared_target_ids"],digit_ids).sum());extra=target-old_count
    assert 0<extra<=2400 and len(population)>=extra
    rng=np.random.default_rng(SEED);new=np.array(s["original_population_indices"],copy=True)
    batches=rng.permutation(2400)[:extra];replacements=rng.permutation(population)[:extra];changes=[]
    for step,index in zip(batches,replacements,strict=True):
        columns=np.flatnonzero(~np.isin(y[new[step]],digit_ids));assert len(columns)>0
        column=int(rng.choice(columns));old=int(new[step,column]);new[step,column]=int(index)
        changes.append(dict(step=int(step),column=column,old_population_index=old,new_population_index=int(index),label_id=int(y[index])))
    ancestor=a["shared_schedule"].reshape(-1)
    targets=np.concatenate((y[new],y[ancestor[s["augmented_view_indices"]]]),axis=1).astype(np.int64)
    difference=new!=s["original_population_indices"]
    assert int(difference.sum())==extra and difference.sum(1).max()==1
    assert int(np.isin(targets,digit_ids).sum())==target and np.unique(targets).size==371
    assert np.array_equal(targets[:,16:],s["shared_target_ids"][:,16:])
    assert np.array_equal(targets[:,:16][~difference],s["shared_target_ids"][:,:16][~difference])
    points=np.array(a["population_features"][replacements],copy=True)
    assert points.dtype==np.float32 and points.shape==(extra,128,5) and np.isfinite(points).all()
    assert ((points[:,:,:2]>=0)&(points[:,:,:2]<=1)).all() and np.isin(a["population_origin"][replacements],(0,1,3)).all()
    OUT.mkdir();hashes={}
    for name,array in dict(original_population_indices=new,shared_target_ids=targets).items():
        np.save(OUT / f"{name}.npy",array,allow_pickle=False);hashes[name]=direct.broad.previous._sha(OUT / f"{name}.npy")
    direct.broad.previous._write(OUT / "replacements.json",changes)
    selected=[]
    for digit,c in enumerate(digit_ids):
        indices=[r["new_population_index"] for r in changes if r["label_id"]==int(c)][:3]
        assert len(indices)==3
        selected.extend(dict(population_index=i,label=str(digit),source_code=int(a["population_origin"][i])) for i in indices)
    direct.broad.previous.aug.base._render(np.stack([a["population_features"][r["population_index"]] for r in selected]),
        [f"{r['label']} src{r['source_code']}" for r in selected],OUT / "new_real_digit_qa.png")
    direct.broad.previous._write(OUT / "qa_rows.json",selected)
    plan=copy.deepcopy(parent)
    plan.update(schema="aiflow-digit-floor-sampling-plan/v31",code_sha256=direct.broad.previous._sha(Path(__file__)),
        trainer_sha256=direct.broad.previous._sha(Path(algorithm.__file__)),parent_v29_plan_sha256=direct.broad.previous._sha(PARENT / "frozen_plan.json"),
        parent_v29_certificate_sha256=direct.broad.previous._sha(PARENT / "independent_verification.json"),
        sampler_seed=SEED,arrays_sha256=hashes,replacements_sha256=direct.broad.previous._sha(OUT / "replacements.json"),
        same_v25_data_and_schedule=False,all_augmented_view_slots_and_augmented_gt_unchanged=True,
        original_rows_replaced=extra,max_original_rows_replaced_per_batch=1,original_digit_exposures=old_count,new_digit_exposures=target,
        total_hard_exposures=153600,admitted_digit_population=len(population),admitted_population=len(y),
        digit_floor_rule="ceil(admitted plain-digit TRAIN fraction *153600); choose distinct admitted real digit rows with fixed RNG, no teacher or evaluation selection",
        all_371_positive_classes_preserved=True,coefficient_rule="All V29 loss coefficients inherited unchanged; no recalibration on the new schedule",
        coefficient_evaluation_inputs_used=0,qa_image_sha256=direct.broad.previous._sha(OUT / "new_real_digit_qa.png"),qa_samples=30,
        trainer_reuse="Unmodified V29 trainer; only process-local output/load bindings replaced and restored; algorithm checkpoint schema remains V29",
        trackio="Same algorithm project name V29, isolated V31 local directory; no remote Space",
        canonical_initialization_and_all_57_trainable=True,generated_labels_human_verified=False,product_adopted=False)
    direct.broad.previous._write(OUT / "frozen_plan.json",plan)
    print(json.dumps(dict(event="digit_floor_prepared",replaced_original_rows=extra,old_digit_exposures=old_count,new_digit_exposures=target,
        digit_fraction=target/153600,all_371_classes_preserved=True,augmented_inputs_unchanged=True,visual_review_required=True)),flush=True)
    return 0


def load(require_visual=True):
    """새 schedule·증강 불변·기존 손실·입력 source와 이미지 검증을 확인한다."""
    sha=direct.broad.previous._sha
    p=json.loads((OUT / "frozen_plan.json").read_text(encoding="utf-8"))
    assert sha(Path(__file__))==p["code_sha256"] and sha(Path(algorithm.__file__))==p["trainer_sha256"]
    assert sha(PARENT / "frozen_plan.json")==p["parent_v29_plan_sha256"]
    assert sha(PARENT / "independent_verification.json")==p["parent_v29_certificate_sha256"]
    parent=json.loads((PARENT / "frozen_plan.json").read_text(encoding="utf-8"))
    _,a,old=algorithm.previous.load()
    assert all(p[k]==parent[k] for k in ("hard_ce_weight","feature_weight","gap_weight","lr","weight_decay","seed","steps","batch_size"))
    s=dict(old)
    for name,digest in p["arrays_sha256"].items():
        assert sha(OUT / f"{name}.npy")==digest
        s[name]=np.load(OUT / f"{name}.npy",mmap_mode="r",allow_pickle=False)
    assert sha(OUT / "replacements.json")==p["replacements_sha256"] and sha(OUT / "new_real_digit_qa.png")==p["qa_image_sha256"]
    difference=s["original_population_indices"]!=old["original_population_indices"]
    assert int(difference.sum())==p["original_rows_replaced"] and difference.sum(1).max()==1
    assert np.array_equal(s["shared_target_ids"][:,16:],old["shared_target_ids"][:,16:]) and np.unique(s["shared_target_ids"]).size==371
    expected=np.concatenate((a["population_labels"][s["original_population_indices"]],
        a["population_labels"][a["shared_schedule"].reshape(-1)[s["augmented_view_indices"]]]),axis=1)
    assert np.array_equal(expected,s["shared_target_ids"])
    if require_visual:
        review=json.loads((OUT / "visual_review.json").read_text(encoding="utf-8"))
        assert review["assistant_visual_qa_completed"] and review["image_sha256"]==p["qa_image_sha256"]
        assert review["human_label_approval"] is False
    return p,a,s


def train():
    """검증된 V29 optimizer·손실·현미경 loop를 별도 폴더와 새 TRAIN schedule로 실행한다."""
    load();old_out,old_load=algorithm.OUT,algorithm.load
    try:
        algorithm.OUT=OUT;algorithm.load=load
        return algorithm.train()
    finally:
        algorithm.OUT=old_out;algorithm.load=old_load


def selftest():
    """모든 배치의 GT 대응과 변경 행 범위를 검사하고 기존 gap gradient 검사를 재사용한다."""
    p,a,s=load(False)
    for step in range(2400):
        x,y=direct.batch(a,s,step,"augmented_main")
        assert x.shape==(64,128,5) and y.shape==(64,) and np.isfinite(x).all()
    algorithm.selftest()
    print(json.dumps(dict(event="digit_floor_selftest",all_2400_gt_batches_match=True,changed_rows=p["original_rows_replaced"],
        untouched_augmented_rows=115200)),flush=True)
    return 0


def evaluate():
    """동일 알고리즘의 이전 V29 모델도 비교표에 포함해 새 sampler만 평가한다."""
    if (OUT / "comparison_result.json").exists():
        raise FileExistsError("comparison exists")
    plan,_,_=load();folder=OUT / "margin_retained_main"
    done=json.loads((folder / "completed.json").read_text(encoding="utf-8"))
    assert done["steps"]==2400 and direct.broad.previous._sha(folder / "main_trained.pt")==done["checkpoint_sha256"]
    cert=json.loads((PARENT / "independent_verification.json").read_text(encoding="utf-8"));assert cert["status"]=="pass"
    assert cert["comparison_sha256"]==direct.broad.previous._sha(PARENT / "comparison_result.json")
    baseline=json.loads((PARENT / "comparison_result.json").read_text(encoding="utf-8"))
    import torch
    import audit_hwr_owned_formula_transfer_v21 as owned
    from audit_hwr_encoder_head_swap_v22 import records_from_scores
    from run_hwr_affine_distillation_experiment_v1 import _load_teacher,_predict_logits
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    model,labels,_=_load_teacher(folder / "main_trained.pt",torch.device("cpu"))
    _,data,folds,_=direct.broad.previous.global_load(direct.broad.previous.OUTPUT);development={}
    for fold in (1,2):
        z=_predict_logits(model,np.array(data["features"][folds[fold]],copy=True),torch.device("cpu"),32)
        np.save(OUT / f"digit_floor_main_fold{fold}_logits.npy",z,allow_pickle=False)
        development[str(fold)]=direct.broad.previous.hit_metrics(z,np.array(data["labels"][folds[fold]],copy=True),labels)
    _,samples,rows=owned.load_source();torch.set_num_threads(1)
    z=_predict_logits(model,np.load(owned.OUT / "inputs.npy",allow_pickle=False),torch.device("cpu"),owned.BATCH)
    records=records_from_scores(z,labels,samples,rows)
    np.save(OUT / "digit_floor_main_owned_logits.npy",z,allow_pickle=False);direct.broad.previous._write(OUT / "digit_floor_main_owned_records.json",records)
    metrics=copy.deepcopy(baseline["metrics"]);metrics["digit_floor_main"]=dict(development=development,
        owned={c:owned.metrics(records,c) for c in ("all","legacy_96","codex_reviewed_53")})
    direct.broad.previous._write(OUT / "comparison_result.json",dict(schema="aiflow-digit-floor-comparison/v31",
        status="completed_consumed_diagnostic_not_fresh_acceptance",baseline_comparison_sha256=cert["comparison_sha256"],metrics=metrics,
        same_v29_algorithm_and_loss_weights=True,all_augmented_inputs_unchanged=True,all_evaluation_cases_already_consumed=True,
        generated_labels_human_verified=False,owned_optimizer_rows=0,crohme_rows=0,product_adopted=False,
        canonical_unchanged=direct.broad.previous._sha(direct.broad.previous.CHECKPOINT)==plan["canonical_sha256"]))
    print(json.dumps(dict(event="digit_floor_scored",metrics=metrics["digit_floor_main"])),flush=True)
    return 0


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("mode",choices=("prepare","selftest","train","evaluate"));args=parser.parse_args()
    raise SystemExit(dict(prepare=prepare,selftest=selftest,train=train,evaluate=evaluate)[args.mode]())
