"""표현 보존 실험의 계수·CE 전달·층 gradient·최종 추론 수치를 재검증한다."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

import run_hwr_direct_feature_retention_v27 as run
from run_hwr_affine_distillation_experiment_v1 import _load_teacher,_predict_logits
from verify_hwr_broad_head_only_v23 import hit_metrics


def main():
    """고정 TRAIN gradient 계수를 재현하고 소비된 평가의 결과만 검증한다."""
    out=run.OUT;folder=out / "feature_retained_main"
    if (out / "independent_verification.json").exists():
        raise FileExistsError("certificate exists")
    plan,a,s=run.load()
    done=json.loads((folder / "completed.json").read_text(encoding="utf-8"))
    result=json.loads((out / "comparison_result.json").read_text(encoding="utf-8"))
    started=json.loads((folder / "run_started.json").read_text(encoding="utf-8"))
    sha=run.direct.broad.previous._sha
    assert started["plan_sha256"]==sha(out / "frozen_plan.json")
    assert done["steps"]==2400 and done["evaluation_forwarded_during_training"]==0
    assert done["checkpoint_sha256"]==sha(folder / "main_trained.pt")
    assert done["microscope_sha256"]==sha(folder / "training_microscope.jsonl")
    source=a["shared_schedule"].reshape(-1)
    expected=np.concatenate((a["population_labels"][s["original_population_indices"]],
        a["population_labels"][source[s["augmented_view_indices"]]]),axis=1)
    assert np.array_equal(expected,s["shared_target_ids"]) and expected.shape==(2400,64)
    count=0;ce_delta=0.;total_delta=0.;weighted_delta=0.
    with (folder / "training_microscope.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            r=json.loads(line);count+=1
            assert r["step"]==count and r["hard_ce_weight"]==1. and r["kl_weight"]==0.
            assert r["hard_target_rows"]==64 and r["augmented_rows"]==48 and r["teacher_gradients_absent"]
            assert r["feature_weight"]==plan["feature_weight"] and r["feature_mse"]>=0
            assert len(r["encoder_layers"])==4 and len(r["parameter_gradient_l2"])==57
            assert all(np.isfinite(v) for v in r["parameter_gradient_l2"].values())
            assert all(np.isfinite(v) for layer in r["encoder_layers"].values() for v in layer.values())
            ce_delta=max(ce_delta,abs(r["hard_ce"]-(.25*r["ce_real_region"]+.75*r["ce_augmented_region"])))
            total_delta=max(total_delta,abs(r["loss"]-r["hard_ce"]-r["weighted_feature"]))
            weighted_delta=max(weighted_delta,abs(r["weighted_feature"]-plan["feature_weight"]*r["feature_mse"]))
            for i in range(4):
                values=[v for n,v in r["parameter_gradient_l2"].items() if n.startswith(f"encoder.layers.{i}.")]
                assert len(values)==12 and max(values)>0
    assert count==2400 and max(ce_delta,total_delta,weighted_delta)<1e-6
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    canonical,labels,_=_load_teacher(run.direct.broad.previous.CHECKPOINT,torch.device("cpu"))
    canonical.eval();canonical.requires_grad_(False)
    baseline,blabels,_=_load_teacher(run.BASE,torch.device("cpu"));baseline.eval()
    model,vocab,_=_load_teacher(folder / "main_trained.pt",torch.device("cpu"));model.eval()
    assert vocab==labels==blabels
    changed=[n for n,p in model.named_parameters() if not torch.equal(p,canonical.state_dict()[n])]
    assert len(changed)==57
    calibration=[];train_mse={"direct_baseline":[],"feature_retained_main":[]}
    encoder=[p for n,p in baseline.named_parameters() if not n.startswith("math_head.")]
    for saved in plan["calibration_rows"]:
        x,y=run.direct.batch(a,s,saved["schedule_step"],"augmented_main");tx=torch.from_numpy(x)
        h=baseline.encode(tx);z=baseline.math_head(h);truth=torch.from_numpy(y)
        ce=F.cross_entropy(z,truth)
        with torch.no_grad():
            target=canonical.encode(tx)
        feature=F.mse_loss(h,target)
        cg=torch.autograd.grad(ce,encoder,retain_graph=True);fg=torch.autograd.grad(feature,encoder)
        cn=float(torch.sqrt(sum(g.square().sum() for g in cg)));fn=float(torch.sqrt(sum(g.square().sum() for g in fg)))
        assert abs(cn-saved["ce_encoder_gradient_l2"])<1e-6 and abs(fn-saved["feature_encoder_gradient_l2"])<1e-6
        calibration.append(cn/fn)
        train_mse["direct_baseline"].append(float(feature.detach()))
        # 최종 main의 CE가 teacher-target 분기와 독립적으로 전체 가중치인지 재검증한다.
        mh=model.encode(tx);mz=model.math_head(mh);mf=F.mse_loss(mh,target)
        total=F.cross_entropy(mz,truth)+plan["feature_weight"]*mf
        gz=torch.autograd.grad(total,mz)[0]
        expected_g=(mz.detach().softmax(-1)-F.one_hot(truth,372))/64
        assert torch.allclose(gz,expected_g,atol=1e-7)
        train_mse["feature_retained_main"].append(float(mf.detach()))
    assert float(np.clip(np.median(calibration),.1,100.))==plan["feature_weight"]
    _,data,folds,_=run.direct.broad.previous.global_load(run.direct.broad.previous.OUTPUT)
    scored=result["metrics"]["feature_retained_main"]
    for fold in (1,2):
        z=_predict_logits(model,np.array(data["features"][folds[fold]],copy=True),torch.device("cpu"),32)
        assert np.array_equal(z,np.load(out / f"feature_retained_main_fold{fold}_logits.npy",allow_pickle=False))
        assert hit_metrics(z,np.array(data["labels"][folds[fold]],copy=True),vocab)==scored["development"][str(fold)]
    import audit_hwr_owned_formula_transfer_v21 as owned
    torch.set_num_threads(1);owner_x=np.load(owned.OUT / "inputs.npy",allow_pickle=False)
    z=_predict_logits(model,owner_x,torch.device("cpu"),owned.BATCH)
    assert np.array_equal(z,np.load(out / "feature_retained_main_owned_logits.npy",allow_pickle=False))
    records=json.loads((out / "feature_retained_main_owned_records.json").read_text(encoding="utf-8"))
    order=np.argsort(-z,axis=1,kind="stable")[:,:5]
    for r in records:
        top=[[vocab[int(i)] for i in indices] for indices in order[r["start"]:r["stop"]]]
        assert top==r["top5_tokens"] and [v[0] for v in top]==r["top1_tokens"]
    for cohort in ("all","legacy_96","codex_reviewed_53"):
        assert owned.metrics(records,cohort)==scored["owned"][cohort]
    owner_mse={}
    with torch.no_grad():
        for name,probe in (("direct_baseline",baseline),("feature_retained_main",model)):
            chunks=[probe.encode(torch.from_numpy(np.array(owner_x[i:i+owned.BATCH],copy=True))).numpy()
                for i in range(0,len(owner_x),owned.BATCH)]
            h=np.concatenate(chunks);np.save(out / f"{name}_owned_embeddings.npy",h,allow_pickle=False)
            owner_mse[name]=float(np.mean((h.astype(np.float64)-np.load(owned.OUT / "canonical_embeddings.npy").astype(np.float64))**2))
    assert sha(run.direct.broad.previous.CHECKPOINT)==plan["canonical_sha256"]
    certificate=dict(schema="aiflow-direct-feature-retention-verification/v27",status="pass",verifier_sha256=sha(Path(__file__)),
        comparison_sha256=sha(out / "comparison_result.json"),checkpoint_sha256=done["checkpoint_sha256"],
        training_steps=count,augmented_hard_target_exposures=115200,same_v25_label_schedule_verified=True,
        all_2400_recorded_loss_formulas_verified=True,loss_formula_max_deltas=dict(ce=ce_delta,total=total_delta,weighted_feature=weighted_delta),
        all_57_gradients_finite_and_all_57_parameter_tensors_changed=True,all_four_layers_active_on_every_step=True,
        frozen_teacher_gradients_absent_in_every_record=True,train_only_coefficient_reproduced=True,
        full_weight_hard_ce_logit_gradient_verified_on_eight_final_train_batches=True,
        final_logits_checkpoint_reload_bit_exact=True,all_new_metrics_recounted=True,
        train_probe_feature_mse={n:float(np.mean(v)) for n,v in train_mse.items()},owned_diagnostic_feature_mse=owner_mse,
        counts={n:dict(internal_top1=sum(m["development"][str(f)]["all"]["top1_hits"] for f in (1,2)),
            owned_formula_top1=m["owned"]["all"]["formula_top1_exact"],owned_formula_top5=m["owned"]["all"]["formula_top5_complete"])
            for n,m in result["metrics"].items()},
        limits="Historical intermediate forwards not fully replayed. Consumed oracle groups are not fresh acceptance or official CROHME Expression Rate.",
        all_evaluation_cases_already_consumed=True,generated_labels_human_verified=False,owned_optimizer_rows=0,crohme_rows=0,
        product_adopted=False,canonical_checkpoint_unchanged=True)
    run.direct.broad.previous._write(out / "independent_verification.json",certificate)
    print(json.dumps(certificate),flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
