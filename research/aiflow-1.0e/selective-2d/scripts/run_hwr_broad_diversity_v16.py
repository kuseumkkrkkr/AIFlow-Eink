"""넓은 실제 TRAIN 풀과 8방향·4강도·다중 반경 증강을 동일 예산으로 비교한다.

새 사람 경계 정답은 만들지 않는다. 생성물은 원본 teacher soft 일관성에만
사용하고, 이전/미래 검증 및 CROHME는 학습·donor에서 제외한다.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np

import run_hwr_balanced_temperature_loop_v15 as previous
from run_hwr_pendigits_tube_probe_v1 import SOURCE, OUTPUT as DIGIT_PARENT

OUT = previous.ROOT / "artifacts/hwr_broad_diversity_20261005_v16_fp32"
STEPS, BATCH, SEED = 2400, 32, 2026100516
TEMPERATURES, QUANTILES = (.25, .5, .75, 1.), (.5, .68, .85)
ARMS = ("broad_real", "broad_augmented")


def save_array(name: str, array: np.ndarray) -> str:
    """생성 데이터 배열을 기존 파일 덮기 없이 저장하고 SHA를 반환한다."""
    path = OUT / f"{name}.npy"
    if path.exists():
        raise FileExistsError(path)
    np.save(path, array, allow_pickle=False)
    return previous._sha(path)


def selected_donors(query: np.ndarray, aligned: np.ndarray, metric: dict, radius: float) -> list[int]:
    """실제 donor의 동일 획 진행 방향을 유지하며 whitened 방향을 최대 8개 분리한다."""
    q = previous.aug.base._aligned(query).reshape(-1, 16, 2)
    dot = ((aligned[:, :, -1] - aligned[:, :, 0]) * (q[:, -1] - q[:, 0])).sum(2)
    coeff = ((q.ravel() - metric["center"]) @ metric["basis"].T) / np.sqrt(metric["eigen"])
    vectors = metric["standardized"] - coeff
    distance = np.linalg.norm(vectors, axis=1)
    compatible = np.flatnonzero((dot >= -1e-8).all(1) & (distance > 1e-8))
    if not len(compatible):
        return []
    inside = compatible[distance[compatible] <= radius]
    primary = int(inside[np.argmax(distance[inside])]) if len(inside) else int(compatible[np.argmin(distance[compatible])])
    units = vectors / np.maximum(distance[:, None], 1e-12)
    chosen = [primary]
    while len(chosen) < 8:
        remaining = [int(i) for i in compatible if i not in chosen]
        if not remaining:
            break
        maximum = units[remaining] @ units[chosen].T
        similarities = maximum.max(1)
        winner = int(np.argmin(similarities))
        if similarities[winner] > .98:
            break
        chosen.append(remaining[winner])
    return chosen


def prepare() -> int:
    """검증 봉인 후 허용 실제 풀 전체를 구성하고 76,800회 노출의 원본·변형을 고정한다."""
    if OUT.exists():
        raise FileExistsError("refusing broad experiment overwrite")
    if previous.aug.base._guard_commit("broad_prepare") is None:
        return 78
    global_plan, arrays, folds, excluded = previous.global_load(previous.OUTPUT)
    forbidden = np.concatenate((*folds, excluded))
    forbidden_hashes = {previous.ink_hash(arrays["features"][int(i)]) for i in forbidden}
    eligible = np.isin(arrays["sources"], (0, 1)); eligible[forbidden] = False
    old_ids, old_hashes = [], []
    for i in np.flatnonzero(eligible):
        digest = previous.ink_hash(arrays["features"][int(i)])
        if digest not in forbidden_hashes:
            old_ids.append(int(i)); old_hashes.append(digest)
    old_ids = np.array(old_ids, dtype=np.int64)
    digit_plan = json.loads((DIGIT_PARENT / "frozen_plan.json").read_text(encoding="utf-8"))
    item = digit_plan["artifacts"]["train_indices"]
    if previous._sha(DIGIT_PARENT / item["file"]) != item["sha256"]:
        raise ValueError("admitted digit TRAIN indices changed")
    digit_ids = np.load(DIGIT_PARENT / item["file"], allow_pickle=False)
    report = json.loads((SOURCE / "pendigits_source_audit.json").read_text(encoding="utf-8"))
    if previous._sha(SOURCE / "pendigits_source_audit.json") != digit_plan["source_report_sha256"]:
        raise ValueError("digit provenance changed")
    digit_arrays = {}
    for name in ("train_features_y_up", "train_digit_labels", "train_writer_group_hashes"):
        if previous._sha(SOURCE / f"{name}.npy") != report["artifacts"][name]["sha256"]:
            raise ValueError("digit TRAIN array changed")
        digit_arrays[name] = np.load(SOURCE / f"{name}.npy", mmap_mode="r", allow_pickle=False)
    if len(digit_ids) != 5995 or not np.isin(digit_arrays["train_writer_group_hashes"][digit_ids], digit_plan["train_groups"]).all():
        raise ValueError("digit admitted writer scope differs")
    allowed_digits = [int(i) for i in digit_ids if previous.ink_hash(digit_arrays["train_features_y_up"][int(i)]) not in forbidden_hashes]
    digit_ids = np.array(allowed_digits, dtype=np.int64)
    x = np.concatenate((arrays["features"][old_ids], digit_arrays["train_features_y_up"][digit_ids])).astype(np.float32)
    digit_class_ids = np.array([global_plan["class_labels"].index(str(i)) for i in range(10)])
    y = np.concatenate((arrays["labels"][old_ids], digit_class_ids[digit_arrays["train_digit_labels"][digit_ids]])).astype(np.int64)
    origin = np.concatenate((arrays["sources"][old_ids], np.full(len(digit_ids), 3))).astype(np.int8)
    raw_ids = np.concatenate((old_ids, digit_ids))
    hashes = np.array(old_hashes + [previous.ink_hash(row) for row in x[len(old_ids):]], dtype="S64")
    if np.unique(y).size != 371 or not np.isfinite(x).all() or not np.isin(origin, (0, 1, 3)).all():
        raise ValueError("real population contract differs")
    OUT.mkdir(parents=True)
    plan = dict(schema="aiflow-broad-diversity-plan/v16", script_sha256=previous._sha(Path(__file__)),
        helper_sha256={Path(m.__file__).name: previous._sha(Path(m.__file__)) for m in (previous, previous.aug, previous.aug.base)},
        previous_global_plan_sha256=previous._sha(previous.OUTPUT / "global_frozen_plan.json"), canonical_sha256=previous._sha(previous.CHECKPOINT),
        steps=STEPS, batch_size=BATCH, seed=SEED, learning_rate=1e-5, class_labels=global_plan["class_labels"],
        population_rows=len(x), source_counts={str(int(i)):int((origin == i).sum()) for i in np.unique(origin)},
        source_codes={"0":"HWRT real TRAIN", "1":"UJI real TRAIN", "3":"PenDigits admitted real TRAIN; not synthetic source 2"},
        digit_writer_groups=len(np.unique(digit_arrays["train_writer_group_hashes"][digit_ids])),
        shape_temperature_mixture=TEMPERATURES, empirical_radius_quantiles=QUANTILES, max_direction_modes=8,
        caps=dict(rms=.035, max_point_displacement=.105, stroke_path_ratio=[.78,1.22]),
        logit_temperature=2., arms=ARMS, objective=".7 CE(real) + .2 KL(real,canonical) + .1 KL(view,original-canonical), T=2",
        user_directed_expansion="explicit new instruction to train very broadly despite the prior automatic V15 growth gate failure; old gate/result preserved",
        validation_fold=1, all_validation_folds_excluded_from_population_and_donors=True,
        source_limit="teacher may have seen TRAIN; main validation is not writer/device-independent acceptance",
        human_boundary_labels=0, synthetic_hard_labels=0, official_test_rows_read=0, consumed_pendigits_held_examples_indexed=0,
        crohme_rows=0, product_adopted=False, collection_or_deployment_resumed=False, cloud_upload=False)
    previous._write(OUT / "frozen_plan.json", plan)
    paths = {name:save_array(name,value) for name,value in dict(population_features=x,population_labels=y,
        population_origin=origin,population_raw_indices=raw_ids,population_ink_hashes=hashes).items()}
    import torch
    from run_hwr_affine_distillation_experiment_v1 import _load_teacher, _predict_logits
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    teacher, labels, _ = _load_teacher(previous.CHECKPOINT,torch.device("cpu"))
    target_path=OUT / "population_teacher_logits.npy"
    soft=np.lib.format.open_memmap(target_path,mode="w+",dtype=np.float32,shape=(len(x),372))
    for start in range(0,len(x),8192):
        end=min(start+8192,len(x))
        soft[start:end]=_predict_logits(teacher,x[start:end],torch.device("cpu"),BATCH)
        soft.flush()
        print(json.dumps(dict(event="fresh_fp32_teacher_targets",rows=end,total=len(x))),flush=True)
    probes = np.linspace(0,len(x)-1,32,dtype=int)
    fresh = _predict_logits(teacher,x[probes],torch.device("cpu"),BATCH)
    cache_delta = float(np.abs(fresh-soft[probes]).max())
    if cache_delta > 1e-4:
        raise ValueError("fresh FP32 teacher targets fail actual model parity")
    paths["population_teacher_logits"] = previous._sha(target_path)
    del teacher,soft
    rng = np.random.default_rng(SEED)
    classes = np.unique(y)
    class_order = np.resize(rng.permutation(classes),STEPS*BATCH)
    pools = {int(c):rng.permutation(np.flatnonzero(y==c)).tolist() for c in classes}
    cursor = {int(c):0 for c in classes}; schedule=[]
    for c in class_order:
        c=int(c);schedule.append(pools[c][cursor[c]%len(pools[c])]);cursor[c]+=1
    schedule=np.array(schedule,dtype=np.int64).reshape(STEPS,BATCH)
    paths["shared_schedule"] = save_array("shared_schedule",schedule)
    buckets={}
    for i,row in enumerate(x):
        buckets.setdefault((int(y[i]),previous.aug.base._signature(row)),[]).append(i)
    banks={}
    for key,ids in buckets.items():
        unique=[];seen=set()
        for i in rng.permutation(ids):
            if hashes[i].tobytes() not in seen:
                seen.add(hashes[i].tobytes());unique.append(int(i))
            if len(unique)==64:break
        banks[key]=unique
    from threadpoolctl import threadpool_limits
    metric_cache={}; metadata=[]; changed_count=0;covered=set();rejected=0
    view_path=OUT / "scheduled_augmented_features.npy"
    views=np.lib.format.open_memmap(view_path,mode="w+",dtype=np.float32,shape=(STEPS,BATCH,128,5))
    with threadpool_limits(limits=2):
        for step,batch in enumerate(schedule):
            for column,index in enumerate(batch):
                index=int(index);original=x[index];signature=previous.aug.base._signature(original)
                key=(int(y[index]),signature)
                fit_ids=tuple(i for i in banks[key] if hashes[i]!=hashes[index])
                quantile=QUANTILES[(step+column)%len(QUANTILES)]
                temperature=TEMPERATURES[(step//3+column)%len(TEMPERATURES)]
                requested_mode=(step//12+column)%8
                changed=original.copy();donor_id=None;used_mode=None;fraction=0.
                if len(fit_ids)>=24:
                    if fit_ids not in metric_cache:
                        fit=x[list(fit_ids)]
                        try:
                            metric=previous.aug.base._fit(fit)
                            aligned=np.stack([previous.aug.base._aligned(row).reshape(-1,16,2) for row in fit])
                            distances=np.linalg.norm(metric["standardized"][:,None]-metric["standardized"][None,:],axis=2)
                            radius={q:float(np.quantile(distances[np.triu_indices(len(fit),1)],q)) for q in QUANTILES}
                            metric_cache[fit_ids]=(metric,aligned,radius)
                        except ValueError:metric_cache[fit_ids]=None
                    entry=metric_cache[fit_ids]
                    if entry is not None:
                        metric,aligned,radii=entry;radius=radii[quantile]
                        donors=selected_donors(original,aligned,metric,radius)
                        if donors:
                            used_mode=requested_mode%len(donors);j=donors[used_mode];donor_id=fit_ids[j]
                            coeff=((previous.aug.base._aligned(original).ravel()-metric["center"])@metric["basis"].T)/np.sqrt(metric["eigen"])
                            distance=float(np.linalg.norm(metric["standardized"][j]-coeff))
                            cap=min(1.,radius/max(distance,1e-8))
                            donor_xy=previous.aug.base._aligned(x[donor_id],original)
                            for scale in previous.aug.base.SCALES:
                                candidate=original.copy();candidate[:,:2]+=temperature*scale*cap*(donor_xy-original[:,:2])
                                geometry=previous.aug.base._geometry(original,candidate)
                                if geometry["valid"] and geometry["rms"]>1e-7:
                                    changed=candidate;fraction=temperature*scale*cap;break
                            else:rejected+=1
                changed_flag=not np.array_equal(original,changed)
                if changed_flag:changed_count+=1;covered.add(int(y[index]))
                views[step,column]=changed
                metadata.append(dict(step=step,column=column,population_index=index,fit_population_indices=list(fit_ids),
                    donor_population_index=donor_id,requested_mode=requested_mode,used_mode=used_mode,quantile=quantile,
                    shape_temperature=temperature,fraction=fraction,changed=changed_flag))
            if (step+1)%128==0:
                print(json.dumps(dict(event="broad_generation",steps=step+1,changed_views=changed_count,classes=len(covered))),flush=True)
    views.flush();del views
    paths["scheduled_augmented_features"]=previous._sha(view_path)
    previous._write(OUT / "generation_provenance.json",metadata)
    paths["generation_provenance.json"]=previous._sha(OUT / "generation_provenance.json")
    # 데이터 QA 표본은 숫자·문자·기호에서 각 4개 클래스, 원본/변형을 같은 축척으로 그린다.
    selected=[];seen=set()
    for family in ("digits","latin_letters","math_symbols"):
        for i,item in enumerate(metadata):
            c=int(y[item["population_index"]])
            if item["changed"] and c not in seen and previous.aug.base._family(labels[c])==family:
                seen.add(c);selected.append(i)
                if sum(previous.aug.base._family(labels[k])==family for k in seen)>=4:break
    values=np.load(view_path,mmap_mode="r",allow_pickle=False);images=[];names=[]
    for i in selected:
        item=metadata[i];images.extend((x[item["population_index"]],values[item["step"],item["column"]]));names.extend((f"P{i:05d}O",f"P{i:05d}V"))
    previous.aug.base._render(np.stack(images),names,OUT / "broad_generation_qa.png")
    paths["broad_generation_qa.png"]=previous._sha(OUT / "broad_generation_qa.png")
    unique=np.unique(schedule);frequencies=np.unique(y[schedule],return_counts=True)[1]
    previous._write(OUT / "prepared_manifest.json",dict(status="prepared_not_trained",plan_sha256=previous._sha(OUT / "frozen_plan.json"),
        artifacts=paths,classes=371,population_rows=len(x),scheduled_tokens=STEPS*BATCH,unique_scheduled_originals=len(unique),
        exposure_min=int(frequencies.min()),exposure_max=int(frequencies.max()),nonzero_augmented_views=changed_count,
        augmented_classes=len(covered),geometry_rejected_views=rejected,cached_teacher_max_delta=cache_delta,
        validation_fit_query_index_overlap=0,validation_rounded_ink_overlap=0,synthetic_hard_labels=0,
        source_counts=plan["source_counts"],digit_writer_groups=plan["digit_writer_groups"]))
    print(json.dumps(dict(event="broad_prepared",population_rows=len(x),unique_scheduled_originals=len(unique),
        scheduled_tokens=STEPS*BATCH,augmented_views=changed_count,augmented_classes=len(covered))),flush=True)
    return 0


def load() -> tuple[dict,dict,dict]:
    """봉인된 코드·도우미·데이터를 확인하며 일부 결과만 보고 재시작하지 않는다."""
    plan=json.loads((OUT / "frozen_plan.json").read_text(encoding="utf-8"));manifest=json.loads((OUT / "prepared_manifest.json").read_text(encoding="utf-8"))
    if plan["script_sha256"]!=previous._sha(Path(__file__)) or plan["canonical_sha256"]!=previous._sha(previous.CHECKPOINT):
        raise ValueError("frozen code/checkpoint differs")
    if manifest["plan_sha256"]!=previous._sha(OUT / "frozen_plan.json"):
        raise ValueError("prepared plan differs")
    for module in (previous,previous.aug,previous.aug.base):
        if previous._sha(Path(module.__file__))!=plan["helper_sha256"][Path(module.__file__).name]:raise ValueError("helper changed")
    for name,digest in manifest["artifacts"].items():
        path=OUT / name if name.endswith((".json",".png")) else OUT / f"{name}.npy"
        if previous._sha(path)!=digest:raise ValueError("prepared artifact changed")
    arrays={name:np.load(OUT / f"{name}.npy",mmap_mode="r",allow_pickle=False) for name in manifest["artifacts"] if not name.endswith((".json",".png"))}
    return plan,manifest,arrays


def train(arm: str) -> int:
    """같은 원본과 64-row forward에서 두 군을 학습하고 매 단계 층/gradient 수치를 남긴다."""
    if previous.aug.base._guard_commit(f"broad_train_{arm}") is None:return 78
    plan,_,arrays=load();folder=OUT / arm
    if folder.exists():raise FileExistsError("arm already started; inspect live PID/session before any restart")
    folder.mkdir()
    previous._write(folder / "run_started.json",dict(status="started",process_id=os.getpid(),plan_sha256=previous._sha(OUT / "frozen_plan.json")))
    os.environ["TRACKIO_DIR"]=str(folder / "trackio")
    for key in ("TRACKIO_WEBHOOK_URL","TRACKIO_SPACE_ID","TRACKIO_SERVER_URL"):os.environ.pop(key,None)
    import torch,trackio
    from torch.nn import functional as F
    from hwr_boundary_distillation_v1 import boundary_kl_loss
    from run_hwr_affine_distillation_experiment_v1 import _load_teacher
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    model,labels,_=_load_teacher(previous.CHECKPOINT,torch.device("cpu"));model.eval()
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=1e-4)
    activation={};handles=[]
    def capture(name):
        """각 층의 실제 출력 유한성과 평균/표준편차를 확인한다."""
        def hook(module,inputs,output):
            value=output.detach()
            if not torch.isfinite(value).all():raise FloatingPointError("nonfinite encoder activation")
            activation[name]=dict(mean=float(value.mean()),std=float(value.std()))
        return hook
    for name,module in model.named_modules():
        if isinstance(module,torch.nn.TransformerEncoderLayer):handles.append(module.register_forward_hook(capture(name)))
    trackio.init(project="aiflow-broad-diversity-v16",name=arm,space_id=None,embed=False,auto_log_cpu=False,auto_log_gpu=False,
        config=dict(steps=STEPS,lr=1e-5,source="real TRAIN",synthetic_hard_labels=0))
    history=folder / "training_microscope.jsonl"
    try:
        with history.open("x",encoding="utf-8") as stream:
            for step,indices in enumerate(arrays["shared_schedule"]):
                real=np.array(arrays["population_features"][indices],copy=True)
                view=np.array(arrays["scheduled_augmented_features"][step],copy=True) if arm=="broad_augmented" else real.copy()
                target=torch.from_numpy(np.array(arrays["population_teacher_logits"][indices],copy=True))
                truth=torch.from_numpy(np.array(arrays["population_labels"][indices],dtype=np.int64,copy=True))
                optimizer.zero_grad(set_to_none=True)
                embedding=model.encode(torch.from_numpy(np.concatenate((real,view))));logits=model.math_head(embedding)
                ce=F.cross_entropy(logits[:BATCH],truth)
                retain=boundary_kl_loss(logits[:BATCH],target,2.,"full");consistent=boundary_kl_loss(logits[BATCH:],target,2.,"full")
                loss=.7*ce+.2*retain+.1*consistent
                if not torch.isfinite(loss):raise FloatingPointError("nonfinite loss")
                loss.backward()
                gradients={name:None if p.grad is None else float(p.grad.detach().norm()) for name,p in model.named_parameters()}
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                if not torch.isfinite(norm):raise FloatingPointError("nonfinite gradient")
                optimizer.step()
                numeric=dict(loss=float(loss.detach()),ce_real=float(ce.detach()),kl_real=float(retain.detach()),kl_view=float(consistent.detach()),gradient_l2=float(norm))
                record=dict(step=step+1,**numeric,encoder_layers=copy.deepcopy(activation),parameter_gradient_l2=gradients)
                stream.write(json.dumps(record)+"\n");stream.flush();trackio.log(numeric,step=step+1)
                if (step+1)%32==0:print(json.dumps(dict(event="broad_training",arm=arm,step=step+1,**numeric)),flush=True)
    except Exception as error:
        previous._write(folder / "failure.json",dict(status="terminal_failure",error=str(error),completed_steps=step,parameter_updates_at_most=step+1))
        trackio.alert(title="training_failure",text=str(error),level=trackio.AlertLevel.ERROR)
        raise
    finally:
        for handle in handles:handle.remove()
        trackio.finish()
    checkpoint=folder / "research.pt"
    torch.save(dict(schema="aiflow-broad-diversity-research/v16",state_dict=model.state_dict(),math_labels=labels,auxiliary_labels=[],
        report=dict(input_contract=dict(observed_channel_mode="uniform-time",math_observed_transform="uniform-time"),product_adopted=False)),checkpoint)
    previous._write(folder / "completed.json",dict(status="completed",steps=STEPS,checkpoint_sha256=previous._sha(checkpoint),
        microscope_sha256=previous._sha(history),canonical_unchanged=previous._sha(previous.CHECKPOINT)==plan["canonical_sha256"],validation_forwarded=0))
    print(json.dumps(dict(event="broad_arm_completed",arm=arm,steps=STEPS)),flush=True)
    return 0


def evaluate() -> int:
    """두 학습이 완전히 종료된 뒤 이전에 읽지 않은 fold 1을 단회 비교한다."""
    if (OUT / "comparison_result.json").exists():raise FileExistsError("validation already consumed")
    plan,_,_=load()
    completed={arm:json.loads((OUT / arm / "completed.json").read_text(encoding="utf-8")) for arm in ARMS}
    if any(previous._sha(OUT / arm / "research.pt")!=done["checkpoint_sha256"] for arm,done in completed.items()):raise ValueError("checkpoint changed")
    _,source,folds,_=previous.global_load(previous.OUTPUT)
    x=np.array(source["features"][folds[1]],copy=True);y=np.array(source["labels"][folds[1]],copy=True)
    import torch
    from run_hwr_affine_distillation_experiment_v1 import _load_teacher,_predict_logits
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    metrics={}
    for arm in ("canonical",*ARMS):
        checkpoint=previous.CHECKPOINT if arm=="canonical" else OUT / arm / "research.pt"
        model,labels,_=_load_teacher(checkpoint,torch.device("cpu"));logits=_predict_logits(model,x,torch.device("cpu"),BATCH)
        metrics[arm]=previous.hit_metrics(logits,y,labels);save_array(f"{arm}_validation_logits",logits)
    result=dict(schema="aiflow-broad-diversity-comparison/v16",status="research_comparison_not_acceptance",metrics=metrics,
        plan_sha256=previous._sha(OUT / "frozen_plan.json"),both_arms_completed_before_validation=True,validation_fold_consumed=1,
        independent_writer_device_acceptance=False,product_adopted=False,canonical_unchanged=previous._sha(previous.CHECKPOINT)==plan["canonical_sha256"],
        official_test_rows_read=0,crohme_rows=0,synthetic_hard_labels=0)
    previous._write(OUT / "comparison_result.json",result);print(json.dumps(dict(event="broad_comparison",metrics=metrics)),flush=True)
    return 0


def main() -> int:
    """준비·각 arm 학습·종료 후 비교를 분리하며 기존 결과를 덮지 않는다."""
    parser=argparse.ArgumentParser();parser.add_argument("mode",choices=("prepare","train","evaluate","selftest"));parser.add_argument("--arm",choices=ARMS)
    args=parser.parse_args()
    if args.mode=="selftest":
        original=np.zeros((128,5),dtype=np.float32);original[:,0]=np.linspace(.2,.8,128);original[:,1]=.5;original[0,3]=1
        fit=np.stack([original.copy() for _ in range(32)])
        for i,row in enumerate(fit):row[:,:2]+=.01*np.array([np.cos(i),np.sin(i)])
        metric=previous.aug.base._fit(fit);aligned=np.stack([previous.aug.base._aligned(row).reshape(-1,16,2) for row in fit])
        assert len(selected_donors(original,aligned,metric,metric["radius"]))==8
        print(json.dumps(dict(selftest="pass",max_directions=8)));return 0
    if args.mode=="train":
        if args.arm is None:parser.error("train requires --arm")
        return train(args.arm)
    return prepare() if args.mode=="prepare" else evaluate()


if __name__=="__main__":raise SystemExit(main())
