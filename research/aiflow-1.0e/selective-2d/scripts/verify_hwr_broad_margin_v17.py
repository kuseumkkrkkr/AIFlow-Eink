"""margin 보존 실험의 실제 손실 공식·배치·층별 전달·fresh fold 출력 재현을 검증한다."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import run_hwr_broad_margin_v17 as run


def main() -> int:
    """저장 logits와 loss를 별도로 재계산하며 이전 검증 fold는 읽지 않는다."""
    if (run.OUT / "independent_verification.json").exists():raise FileExistsError("certificate exists")
    plan,a=run.load();done=json.loads((run.OUT / "completed.json").read_text(encoding="utf-8"))
    result=json.loads((run.OUT / "comparison_result.json").read_text(encoding="utf-8"))
    if done["steps"]!=run.broad.STEPS or run.broad.previous._sha(run.OUT / "research.pt")!=done["checkpoint_sha256"]:
        raise ValueError("fixed-budget checkpoint changed or incomplete")
    history_path=run.OUT / "training_microscope.jsonl"
    if run.broad.previous._sha(history_path)!=done["microscope_sha256"]:raise ValueError("microscope changed")
    protected_counts={1:[],5:[]};deficit_counts={1:0,5:0};loss_formula_max_delta=0.;steps=0
    with history_path.open(encoding="utf-8") as stream:
        for line in stream:
            r=json.loads(line);step=steps;steps+=1
            if r["step"]!=steps or len(r["encoder_layers"])!=4:raise ValueError("missing layer/step record")
            if not all(np.isfinite(r[k]) for k in ("loss","base_loss","ce_real","kl_real","kl_view","margin_loss","gradient_l2")):
                raise ValueError("nonfinite optimization record")
            ids=a["shared_schedule"][step];truth=a["population_labels"][ids];teacher=np.array(a["population_teacher_logits"][ids],copy=True)
            target=teacher[np.arange(len(truth)),truth,None]
            rank=1+(teacher>target).sum(1)+((teacher==target)&(np.arange(372)[None]<truth[:,None])).sum(1)
            for k in (1,5):
                protected=int((rank<=k).sum())
                if protected!=r[f"protected_top{k}"]:raise ValueError("immutable teacher mask differs")
                protected_counts[k].append(protected);deficit_counts[k]+=r[f"floor_fail_top{k}"]
            expected_margin=.5*(r["mean_deficit_top1"]+r["mean_deficit_top5"])
            expected_base=.7*r["ce_real"]+.2*r["kl_real"]+.1*r["kl_view"]
            expected_loss=r["base_loss"]+run.WEIGHT*r["margin_loss"]
            delta=max(abs(r["margin_loss"]-expected_margin),abs(r["base_loss"]-expected_base),abs(r["loss"]-expected_loss))
            loss_formula_max_delta=max(loss_formula_max_delta,delta)
            if delta>1e-6:raise ValueError("loss formula differs")
            for i in range(4):
                prefix=f"encoder.layers.{i}."
                if not any(v is not None and v>0 for name,v in r["parameter_gradient_l2"].items() if name.startswith(prefix)):
                    raise ValueError("encoder gradient missing")
    if steps!=run.broad.STEPS:raise ValueError("fixed budget incomplete")
    _,source,folds,_=run.broad.previous.global_load(run.broad.previous.OUTPUT)
    x=np.array(source["features"][folds[2]],copy=True);y=np.array(source["labels"][folds[2]],copy=True)
    import torch
    from run_hwr_affine_distillation_experiment_v1 import _load_teacher,_predict_logits
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    reference,_,_=_load_teacher(run.broad.previous.CHECKPOINT,torch.device("cpu"))
    paths={"canonical":run.broad.previous.CHECKPOINT,"broad_augmented":run.broad.OUT / "broad_augmented/research.pt","fixed_margin":run.OUT / "research.pt"}
    metrics={};hits={};changed={}
    for name,path in paths.items():
        model,labels,_=_load_teacher(path,torch.device("cpu"));z=_predict_logits(model,x,torch.device("cpu"),run.broad.BATCH)
        saved=np.load(run.OUT / f"{name}_fold2_logits.npy",allow_pickle=False)
        if not np.array_equal(z,saved):raise ValueError("reloaded output not bit-exact")
        target=z[np.arange(len(y)),y,None]
        rank=1+(z>target).sum(1)+((z==target)&(np.arange(372)[None]<y[:,None])).sum(1)
        hits[name]=rank==1;metrics[name]={}
        for family in ("all","digits","latin_letters","math_symbols"):
            mask=np.ones(len(y),bool) if family=="all" else np.array([run.broad.previous.aug.base._family(labels[int(c)])==family for c in y])
            metrics[name][family]=dict(rows=int(mask.sum()),top1_hits=int((rank[mask]==1).sum()),top5_hits=int((rank[mask]<=5).sum()))
        if name=="fixed_margin":
            changed={f"layer_{i}":sum(not torch.equal(reference.state_dict()[key],value) for key,value in model.state_dict().items() if key.startswith(f"encoder.layers.{i}.")) for i in range(4)}
            if not all(changed.values()):raise ValueError("encoder silently frozen")
    before,after=hits["broad_augmented"],hits["fixed_margin"]
    paired=dict(wins=int((~before&after).sum()),losses=int((before&~after).sum()),net=int(after.sum()-before.sum()))
    if metrics!=result["metrics"] or paired!=result["paired_vs_control"]:raise ValueError("independent metrics differ")
    certificate=dict(schema="aiflow-broad-fixed-margin-verification/v17",status="reproduced",
        verifier_sha256=run.broad.previous._sha(Path(__file__)),result_sha256=run.broad.previous._sha(run.OUT / "comparison_result.json"),
        all_2400_loss_formulas_rebuilt=True,loss_formula_max_absolute_delta=loss_formula_max_delta,
        immutable_teacher_masks_rebuilt_on_all_shared_batches=True,checkpoint_reload_logits_bit_exact=True,
        independent_stable_ranks_reproduced=True,all_four_encoder_layers_received_gradients=True,
        changed_encoder_tensors_by_layer=changed,total_training_floor_fail_occurrences={str(k):v for k,v in deficit_counts.items()},
        previous_validation_logits_loaded=False,validation_fold_consumed=2,independent_writer_device_acceptance=False,
        canonical_unchanged=run.broad.previous._sha(run.broad.previous.CHECKPOINT)==plan["canonical_sha256"],product_adopted=False,official_test_rows_read=0,crohme_rows=0)
    run.broad.previous._write(run.OUT / "independent_verification.json",certificate);print(json.dumps(certificate),flush=True)
    return 0


if __name__=="__main__":raise SystemExit(main())
