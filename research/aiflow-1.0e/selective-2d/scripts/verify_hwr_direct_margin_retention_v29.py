"""정답 gap 보존의 TRAIN 계수·mask·gradient·최종 성능을 재검증한다."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

import run_hwr_direct_margin_retention_v29 as run
from run_hwr_affine_distillation_experiment_v1 import _load_teacher,_predict_logits
from verify_hwr_broad_head_only_v23 import hit_metrics


def main():
    """학습 완료 후 전체 기록과 고정 TRAIN probe를 검사하며 평가 입력은 optimizer에 넣지 않는다."""
    out=run.OUT;folder=out / "margin_retained_main";sha=run.direct.broad.previous._sha
    if (out / "independent_verification.json").exists():
        raise FileExistsError("certificate exists")
    plan,a,s=run.load()
    result=json.loads((out / "comparison_result.json").read_text(encoding="utf-8"))
    done=json.loads((folder / "completed.json").read_text(encoding="utf-8"))
    start=json.loads((folder / "run_started.json").read_text(encoding="utf-8"))
    assert start["plan_sha256"]==sha(out / "frozen_plan.json")
    assert done["steps"]==2400 and done["evaluation_forwarded_during_training"]==0 and done["canonical_unchanged"]
    assert done["checkpoint_sha256"]==sha(folder / "main_trained.pt") and done["microscope_sha256"]==sha(folder / "training_microscope.jsonl")
    source=a["shared_schedule"].reshape(-1)
    expected=np.concatenate((a["population_labels"][s["original_population_indices"]],a["population_labels"][source[s["augmented_view_indices"]]]),axis=1)
    assert np.array_equal(expected,s["shared_target_ids"]) and expected.shape==(2400,64)
    count=0;max_delta=0.;correct_rows_total=0
    with (folder / "training_microscope.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            r=json.loads(line);count+=1
            assert r["step"]==count and r["hard_ce_weight"]==1. and r["kl_weight"]==0.
            assert r["hard_target_rows"]==64 and r["augmented_rows"]==48 and r["teacher_gradients_absent"]
            assert r["feature_weight"]==plan["feature_weight"] and r["gap_weight"]==plan["gap_weight"]
            assert 0<=r["teacher_correct_rows"]<=64 and 0<=r["active_competitor_gaps"]<=r["teacher_correct_rows"]*371
            assert len(r["encoder_layers"])==4 and len(r["parameter_gradient_l2"])==57
            assert all(np.isfinite(v) for v in r["parameter_gradient_l2"].values())
            assert all(np.isfinite(v) for layer in r["encoder_layers"].values() for v in layer.values())
            errors=[abs(r["hard_ce"]-(.25*r["ce_real_region"]+.75*r["ce_augmented_region"])),
                abs(r["weighted_feature"]-plan["feature_weight"]*r["feature_mse"]),abs(r["weighted_gap"]-plan["gap_weight"]*r["gap_loss"]),
                abs(r["loss"]-r["hard_ce"]-r["weighted_feature"]-r["weighted_gap"])]
            max_delta=max(max_delta,*errors);assert max(errors)<1e-6
            for i in range(4):
                values=[v for n,v in r["parameter_gradient_l2"].items() if n.startswith(f"encoder.layers.{i}.")]
                assert len(values)==12 and max(values)>0
            correct_rows_total+=r["teacher_correct_rows"]
    assert count==2400
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);run.selftest()
    model,labels,_=_load_teacher(folder / "main_trained.pt",torch.device("cpu"));model.eval()
    baseline,base_labels,_=_load_teacher(run.BASE,torch.device("cpu"));baseline.eval()
    canonical,teacher_labels,_=_load_teacher(run.direct.broad.previous.CHECKPOINT,torch.device("cpu"))
    canonical.eval();canonical.requires_grad_(False);assert labels==base_labels==teacher_labels
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert sum(not torch.equal(p,canonical.state_dict()[n]) for n,p in model.named_parameters())==57
    params=list(baseline.parameters());ratios=[];probe_gap={"feature_retained_baseline":[],"margin_retained_main":[]}
    max_numpy_gap_delta=0.
    for saved in plan["calibration_rows"]:
        x,y=run.direct.batch(a,s,saved["schedule_step"],"augmented_main");tx=torch.from_numpy(x);truth=torch.from_numpy(y)
        with torch.no_grad():
            th=canonical.encode(tx);tz=canonical.math_head(th)
        z=baseline.math_head(baseline.encode(tx));ce=F.cross_entropy(z,truth);gap,mask,active=run.gap_loss(z,tz,truth)
        cg=torch.autograd.grad(ce,params,retain_graph=True);gg=torch.autograd.grad(gap,params)
        cn=float(torch.sqrt(sum(g.square().sum() for g in cg)));gn=float(torch.sqrt(sum(g.square().sum() for g in gg)))
        assert abs(cn-saved["ce_all57_gradient_l2"])<1e-6 and abs(gn-saved["gap_all57_gradient_l2"])<1e-6
        assert int(mask.sum())==saved["teacher_correct_rows"] and active==saved["active_competitor_gaps"]
        ratios.append(cn/gn);probe_gap["feature_retained_baseline"].append(float(gap.detach()))
        mh=model.encode(tx);mz=model.math_head(mh);mc=F.cross_entropy(mz,truth);mf=F.mse_loss(mh,th);mg,mm,_=run.gap_loss(mz,tz,truth)
        # 기존 helper와 별개로 NumPy double에서 모든 371개 경쟁 gap 손실을 재계산한다.
        zz=mz.detach().numpy().astype(np.float64);tt=tz.numpy().astype(np.float64)
        eligible=tt.argmax(1)==y;tg=tt[np.arange(64),y,None]-tt;sg=zz[np.arange(64),y,None]-zz
        delta=np.maximum(tg-sg,0)*eligible[:,None];manual=float((delta**2).sum()/(max(int(eligible.sum()),1)*371))
        max_numpy_gap_delta=max(max_numpy_gap_delta,abs(manual-float(mg.detach())));assert abs(manual-float(mg.detach()))<1e-5
        assert np.array_equal(eligible,mm.numpy())
        cgrad=torch.autograd.grad(mc,mz,retain_graph=True)[0]
        ggrad=torch.autograd.grad(mg,mz,retain_graph=True)[0]
        assert torch.allclose(cgrad,(mz.detach().softmax(-1)-F.one_hot(truth,372))/64,atol=1e-7)
        assert (ggrad[torch.arange(64),truth]<=1e-7).all() and (ggrad[~mm]==0).all()
        total=mc+plan["feature_weight"]*mf+plan["gap_weight"]*mg
        assert torch.allclose(torch.autograd.grad(total,mz)[0],cgrad+plan["gap_weight"]*ggrad,atol=1e-6)
        probe_gap["margin_retained_main"].append(float(mg.detach()))
    assert float(np.clip(np.median(ratios),1e-3,100.))==plan["gap_weight"]
    _,data,folds,_=run.direct.broad.previous.global_load(run.direct.broad.previous.OUTPUT)
    scored=result["metrics"]["margin_retained_main"]
    for fold in (1,2):
        z=_predict_logits(model,np.array(data["features"][folds[fold]],copy=True),torch.device("cpu"),32)
        assert np.isfinite(z).all() and np.array_equal(z,np.load(out / f"margin_retained_main_fold{fold}_logits.npy",allow_pickle=False))
        assert hit_metrics(z,np.array(data["labels"][folds[fold]],copy=True),labels)==scored["development"][str(fold)]
    import audit_hwr_owned_formula_transfer_v21 as owned
    torch.set_num_threads(1);ox=np.load(owned.OUT / "inputs.npy",allow_pickle=False)
    z=_predict_logits(model,ox,torch.device("cpu"),owned.BATCH)
    assert np.isfinite(z).all() and np.array_equal(z,np.load(out / "margin_retained_main_owned_logits.npy",allow_pickle=False))
    records=json.loads((out / "margin_retained_main_owned_records.json").read_text(encoding="utf-8"))
    order=np.argsort(-z,axis=1,kind="stable")[:,:5]
    for r in records:
        top=[[labels[int(i)] for i in indices] for indices in order[r["start"]:r["stop"]]]
        assert top==r["top5_tokens"] and [v[0] for v in top]==r["top1_tokens"]
    for cohort in ("all","legacy_96","codex_reviewed_53"):
        assert owned.metrics(records,cohort)==scored["owned"][cohort]
    assert sha(run.direct.broad.previous.CHECKPOINT)==plan["canonical_sha256"]
    certificate=dict(schema="aiflow-direct-margin-retention-verification/v29",status="pass",verifier_sha256=sha(Path(__file__)),
        comparison_sha256=sha(out / "comparison_result.json"),checkpoint_sha256=done["checkpoint_sha256"],training_steps=count,
        same_v25_label_schedule_verified=True,augmented_hard_target_exposures=115200,
        all_2400_recorded_three_loss_formulas_verified=True,max_recorded_formula_delta=max_delta,
        all_57_gradients_finite_and_all_57_tensors_changed=True,all_four_layers_active_every_step=True,final_parameter_tensors_finite=True,
        teacher_gradients_absent_every_record=True,teacher_correct_regularizer_rows_total=correct_rows_total,
        fixed_train_coefficient_reproduced=True,gap_loss_numpy_double_reproduced=True,max_numpy_gap_delta=max_numpy_gap_delta,
        full_weight_ce_and_gap_gradient_direction_and_wrong_teacher_mask_verified=True,
        train_probe_gap_loss={n:float(np.mean(v)) for n,v in probe_gap.items()},
        final_logits_checkpoint_reload_bit_exact=True,all_new_metrics_recounted=True,
        counts={n:dict(internal_top1=sum(m["development"][str(f)]["all"]["top1_hits"] for f in (1,2)),
            owned_formula_top1=m["owned"]["all"]["formula_top1_exact"],owned_formula_top5=m["owned"]["all"]["formula_top5_complete"])
            for n,m in result["metrics"].items()},
        limits="Historical intermediate forwards not fully replayed. Consumed oracle-group diagnostic, not fresh human acceptance or official Expression Rate.",
        all_evaluation_cases_already_consumed=True,generated_labels_human_verified=False,owned_optimizer_rows=0,crohme_rows=0,
        canonical_checkpoint_unchanged=True,product_adopted=False)
    run.direct.broad.previous._write(out / "independent_verification.json",certificate)
    print(json.dumps(certificate),flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
