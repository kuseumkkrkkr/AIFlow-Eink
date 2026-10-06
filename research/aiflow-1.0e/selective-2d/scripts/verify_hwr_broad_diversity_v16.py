"""광범위 학습 데이터의 실제 출처·검증 분리·타점 재현과 종료 모델 점수를 검증한다."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import run_hwr_broad_diversity_v16 as experiment


def verify_data() -> int:
    """원본 GT를 다시 대조하고 모든 노출의 donor·타점·채널·기하 제한을 확인한다."""
    out=experiment.OUT
    if (out / "independent_data_verification.json").exists():raise FileExistsError("certificate exists")
    plan,manifest,a=experiment.load()
    if a["population_teacher_logits"].dtype!=np.float32 or not np.isfinite(a["population_teacher_logits"]).all():
        raise ValueError("teacher targets are not finite FP32")
    old_plan,old,folds,excluded=experiment.previous.global_load(experiment.previous.OUTPUT)
    main=a["population_origin"]!=3;raw=a["population_raw_indices"]
    if not np.array_equal(a["population_features"][main],old["features"][raw[main]]) or not np.array_equal(a["population_labels"][main],old["labels"][raw[main]]):
        raise ValueError("original main features or GT differ")
    digit_x=np.load(experiment.SOURCE / "train_features_y_up.npy",mmap_mode="r",allow_pickle=False)
    digit_y=np.load(experiment.SOURCE / "train_digit_labels.npy",mmap_mode="r",allow_pickle=False)
    digit_ids=np.load(experiment.DIGIT_PARENT / "train_indices.npy",allow_pickle=False)
    class_ids=np.array([old_plan["class_labels"].index(str(i)) for i in range(10)])
    if not np.isin(raw[~main],digit_ids).all() or not np.array_equal(a["population_features"][~main],digit_x[raw[~main]]) or not np.array_equal(a["population_labels"][~main],class_ids[digit_y[raw[~main]]]):
        raise ValueError("digit source/GT scope differs")
    forbidden=np.concatenate((*folds,excluded))
    hashes={experiment.previous.ink_hash(old["features"][int(i)]) for i in forbidden}
    if np.isin(raw[main],forbidden).any() or any(experiment.previous.ink_hash(row) in hashes for row in a["population_features"]):
        raise ValueError("validation/prior diagnostic input leaked")
    schedule=a["shared_schedule"]
    frequency=np.unique(a["population_labels"][schedule],return_counts=True)[1]
    if schedule.shape!=(experiment.STEPS,experiment.BATCH) or len(frequency)!=371 or frequency.max()-frequency.min()>1:
        raise ValueError("balanced exposure schedule differs")
    metadata=json.loads((out / "generation_provenance.json").read_text(encoding="utf-8"))
    if len(metadata)!=experiment.STEPS*experiment.BATCH:raise ValueError("missing generation provenance")
    changed=0;classes=set();temperature_counts={};quantile_counts={};mode_counts={}
    for item in metadata:
        step,column=item["step"],item["column"];index=item["population_index"]
        if int(schedule[step,column])!=index:raise ValueError("input schedule/provenance differs")
        original=a["population_features"][index];view=a["scheduled_augmented_features"][step,column]
        fit=np.array(item["fit_population_indices"],dtype=np.int64)
        if index in fit or (a["population_ink_hashes"][fit]==a["population_ink_hashes"][index]).any() or not (a["population_labels"][fit]==a["population_labels"][index]).all():
            raise ValueError("self query or cross-class covariance used")
        if not experiment.previous.aug.base._geometry(original,view)["valid"]:
            raise ValueError("geometry contract failed")
        rebuilt=np.array(original,copy=True)
        if item["fraction"]:
            donor=item["donor_population_index"]
            if donor not in fit:raise ValueError("donor missing from fit")
            xy=experiment.previous.aug.base._aligned(a["population_features"][donor],original)
            rebuilt[:,:2]+=item["fraction"]*(xy-original[:,:2])
        if not np.array_equal(view,rebuilt):raise ValueError("generated endpoint not bit-exact")
        if item["changed"]!=bool(not np.array_equal(original,view)):raise ValueError("changed flag differs")
        if item["changed"]:
            changed+=1;classes.add(int(a["population_labels"][index]))
            for counter,key in ((temperature_counts,"shape_temperature"),(quantile_counts,"quantile"),(mode_counts,"used_mode")):
                value=str(item[key]);counter[value]=counter.get(value,0)+1
    if changed!=manifest["nonzero_augmented_views"] or len(classes)!=manifest["augmented_classes"]:
        raise ValueError("diversity counts differ")
    report=dict(schema="aiflow-broad-data-verification/v16",status="reproduced",plan_sha256=experiment.previous._sha(out / "frozen_plan.json"),
        manifest_sha256=experiment.previous._sha(out / "prepared_manifest.json"),verifier_sha256=experiment.previous._sha(Path(__file__)),
        population_real_features_and_gt_rebuilt=True,population_rows=len(a["population_labels"]),exposure_min=int(frequency.min()),exposure_max=int(frequency.max()),
        validation_and_prior_index_or_rounded_ink_overlap=0,self_query_or_alias_fit_overlap=0,geometry_errors=0,
        all_scheduled_endpoints_rebuilt_bit_exact=True,nonzero_augmented_views=changed,augmented_classes=len(classes),
        actual_temperature_counts=temperature_counts,actual_quantile_counts=quantile_counts,actual_used_direction_counts=mode_counts,
        teacher_targets_dtype="float32",
        synthetic_hard_labels=0,human_semantic_approval=False,official_test_rows_read=0,crohme_rows=0,product_adopted=False)
    experiment.previous._write(out / "independent_data_verification.json",report)
    print(json.dumps(dict(event="broad_data_verification",**report)),flush=True)
    return 0


def verify_results() -> int:
    """checkpoint를 다시 불러 372-way 순위를 별도로 계산하고 층별 실제 학습을 확인한다."""
    out=experiment.OUT
    if (out / "independent_result_verification.json").exists():raise FileExistsError("certificate exists")
    plan,_,_=experiment.load();result=json.loads((out / "comparison_result.json").read_text(encoding="utf-8"))
    data=json.loads((out / "independent_data_verification.json").read_text(encoding="utf-8"))
    if data["plan_sha256"]!=experiment.previous._sha(out / "frozen_plan.json"):raise ValueError("data plan changed")
    _,a,folds,_=experiment.previous.global_load(experiment.previous.OUTPUT)
    x=np.array(a["features"][folds[1]],copy=True);y=np.array(a["labels"][folds[1]],copy=True)
    import torch
    from run_hwr_affine_distillation_experiment_v1 import _load_teacher,_predict_logits
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    baseline,_,_=_load_teacher(experiment.previous.CHECKPOINT,torch.device("cpu"));metrics={};updated={}
    for arm in ("canonical",*experiment.ARMS):
        path=experiment.previous.CHECKPOINT if arm=="canonical" else out / arm / "research.pt"
        model,labels,_=_load_teacher(path,torch.device("cpu"));scores=_predict_logits(model,x,torch.device("cpu"),experiment.BATCH)
        saved=np.load(out / f"{arm}_validation_logits.npy",allow_pickle=False)
        if not np.array_equal(scores,saved):raise ValueError("checkpoint reload output differs")
        target=scores[np.arange(len(y)),y,None]
        rank=1+(scores>target).sum(1)+((scores==target)&(np.arange(372)[None]<y[:,None])).sum(1)
        metrics[arm]={}
        for family in ("all","digits","latin_letters","math_symbols"):
            mask=np.ones(len(y),bool) if family=="all" else np.array([experiment.previous.aug.base._family(labels[int(c)])==family for c in y])
            metrics[arm][family]=dict(rows=int(mask.sum()),top1_hits=int((rank[mask]==1).sum()),top5_hits=int((rank[mask]<=5).sum()))
        if arm!="canonical":
            done=json.loads((out / arm / "completed.json").read_text(encoding="utf-8"))
            if experiment.previous._sha(path)!=done["checkpoint_sha256"] or experiment.previous._sha(out / arm / "training_microscope.jsonl")!=done["microscope_sha256"]:raise ValueError("arm evidence changed")
            with (out / arm / "training_microscope.jsonl").open(encoding="utf-8") as stream:
                steps=0
                for line in stream:
                    row=json.loads(line);steps+=1
                    if row["step"]!=steps or len(row["encoder_layers"])!=4:raise ValueError("layer/step transfer missing")
                    if not all(np.isfinite(row[k]) for k in ("loss","ce_real","kl_real","kl_view","gradient_l2")):raise ValueError("nonfinite loss/gradient")
                    for layer in range(4):
                        prefix=f"encoder.layers.{layer}."
                        if not any(value is not None and value>0 for key,value in row["parameter_gradient_l2"].items() if key.startswith(prefix)):raise ValueError("encoder gradient missing")
                if steps!=experiment.STEPS:raise ValueError("fixed budget incomplete")
            updated[arm]={f"layer_{i}":sum(not torch.equal(baseline.state_dict()[key],value) for key,value in model.state_dict().items() if key.startswith(f"encoder.layers.{i}.")) for i in range(4)}
            if not all(updated[arm].values()):raise ValueError("encoder silently frozen")
    if metrics!=result["metrics"]:raise ValueError("independent target ranks differ")
    certificate=dict(schema="aiflow-broad-result-verification/v16",status="reproduced",verifier_sha256=experiment.previous._sha(Path(__file__)),
        result_sha256=experiment.previous._sha(out / "comparison_result.json"),both_2400_step_arms_verified=True,
        checkpoint_reload_logits_bit_exact=True,independent_stable_ranks_reproduced=True,changed_encoder_tensors_by_layer=updated,
        canonical_unchanged=experiment.previous._sha(experiment.previous.CHECKPOINT)==plan["canonical_sha256"],
        independent_writer_device_acceptance=False,product_adopted=False,crohme_rows=0,official_test_rows_read=0)
    experiment.previous._write(out / "independent_result_verification.json",certificate);print(json.dumps(certificate),flush=True)
    return 0


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("mode",choices=("data","results"));args=parser.parse_args()
    raise SystemExit(verify_data() if args.mode=="data" else verify_results())
