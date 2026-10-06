"""숫자 노출만 바꾼 실험의 schedule·기존 손실·최종 checkpoint 수치를 검증한다."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

import run_hwr_digit_floor_sampling_v31 as run
from run_hwr_affine_distillation_experiment_v1 import _load_teacher,_predict_logits
from verify_hwr_broad_head_only_v23 import hit_metrics


def main():
    """평가를 마친 뒤 새 데이터 역할을 만들지 않고 실제 변경량·loss 전달·추론을 재검증한다."""
    out=run.OUT;folder=out / "margin_retained_main";sha=run.direct.broad.previous._sha
    if (out / "independent_verification.json").exists():
        raise FileExistsError("certificate exists")
    plan,a,s=run.load()
    parent=json.loads((run.PARENT / "frozen_plan.json").read_text(encoding="utf-8"))
    parent_cert=json.loads((run.PARENT / "independent_verification.json").read_text(encoding="utf-8"))
    assert parent_cert["status"]=="pass" and parent_cert["fixed_train_coefficient_reproduced"]
    assert parent_cert["comparison_sha256"]==sha(run.PARENT / "comparison_result.json")
    assert all(plan[k]==parent[k] for k in ("hard_ce_weight","feature_weight","gap_weight","lr","weight_decay","gradient_clip","seed"))
    _,_,original=run.algorithm.previous.load()
    difference=s["original_population_indices"]!=original["original_population_indices"]
    assert difference.sum()==2383 and difference.sum(1).max()==1
    assert np.array_equal(s["augmented_view_indices"],original["augmented_view_indices"])
    assert np.array_equal(s["shared_target_ids"][:,16:],original["shared_target_ids"][:,16:])
    assert np.unique(s["shared_target_ids"]).size==371 and np.unique(s["augmented_view_indices"]).size==68726
    labels=json.loads((run.direct.broad.OUT / "frozen_plan.json").read_text(encoding="utf-8"))["class_labels"]
    digit_ids=[labels.index(str(i)) for i in range(10)]
    y=a["population_labels"];population_digits=int(np.isin(y,digit_ids).sum())
    exact_floor=(population_digits*153600+len(y)-1)//len(y)
    digit_count=int(np.isin(s["shared_target_ids"],digit_ids).sum())
    assert digit_count==exact_floor==6855
    assert np.isin(y[s["original_population_indices"]][difference],digit_ids).all()
    assert not np.isin(y[original["original_population_indices"]][difference],digit_ids).any()
    assert np.array_equal(s["shared_target_ids"][:,:16][~difference],original["shared_target_ids"][:,:16][~difference])
    done=json.loads((folder / "completed.json").read_text(encoding="utf-8"));result=json.loads((out / "comparison_result.json").read_text(encoding="utf-8"))
    started=json.loads((folder / "run_started.json").read_text(encoding="utf-8"))
    assert started["plan_sha256"]==sha(out / "frozen_plan.json") and done["steps"]==2400 and done["evaluation_forwarded_during_training"]==0
    assert done["checkpoint_sha256"]==sha(folder / "main_trained.pt") and done["microscope_sha256"]==sha(folder / "training_microscope.jsonl")
    count=0;max_loss_delta=0.
    with (folder / "training_microscope.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            r=json.loads(line);count+=1
            assert r["step"]==count and r["hard_ce_weight"]==1. and r["kl_weight"]==0.
            assert r["feature_weight"]==plan["feature_weight"] and r["gap_weight"]==plan["gap_weight"]
            assert r["teacher_gradients_absent"] and r["hard_target_rows"]==64 and r["augmented_rows"]==48
            assert 0<=r["active_competitor_gaps"]<=r["teacher_correct_rows"]*371 and 0<=r["teacher_correct_rows"]<=64
            assert len(r["parameter_gradient_l2"])==57 and len(r["encoder_layers"])==4
            assert all(np.isfinite(v) for v in r["parameter_gradient_l2"].values())
            assert all(np.isfinite(v) for layer in r["encoder_layers"].values() for v in layer.values())
            for i in range(4):
                g=[v for n,v in r["parameter_gradient_l2"].items() if n.startswith(f"encoder.layers.{i}.")]
                assert len(g)==12 and max(g)>0
            delta=max(abs(r["hard_ce"]-.25*r["ce_real_region"]-.75*r["ce_augmented_region"]),
                abs(r["weighted_feature"]-plan["feature_weight"]*r["feature_mse"]),
                abs(r["weighted_gap"]-plan["gap_weight"]*r["gap_loss"]),
                abs(r["loss"]-r["hard_ce"]-r["weighted_feature"]-r["weighted_gap"]))
            max_loss_delta=max(max_loss_delta,delta);assert delta<1e-6
    assert count==2400
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);run.algorithm.selftest()
    model,vocab,_=_load_teacher(folder / "main_trained.pt",torch.device("cpu"));model.eval()
    canonical,cv,_=_load_teacher(run.direct.broad.previous.CHECKPOINT,torch.device("cpu"));canonical.eval();canonical.requires_grad_(False)
    assert vocab==cv==labels and all(torch.isfinite(p).all() for p in model.parameters())
    assert sum(not torch.equal(p,canonical.state_dict()[n]) for n,p in model.named_parameters())==57
    max_gap_delta=0.
    for step in np.linspace(0,2399,8,dtype=int):
        x,yy=run.direct.batch(a,s,int(step),"augmented_main");tx=torch.from_numpy(x);truth=torch.from_numpy(yy)
        h=model.encode(tx);z=model.math_head(h)
        with torch.no_grad():
            th=canonical.encode(tx);tz=canonical.math_head(th)
        ce=F.cross_entropy(z,truth);feature=F.mse_loss(h,th);gap,mask,_=run.algorithm.gap_loss(z,tz,truth)
        zz=z.detach().numpy().astype(np.float64);tt=tz.numpy().astype(np.float64);correct=tt.argmax(1)==yy
        deficit=np.maximum((tt[np.arange(64),yy,None]-tt)-(zz[np.arange(64),yy,None]-zz),0)*correct[:,None]
        expected_gap=float((deficit**2).sum()/(max(int(correct.sum()),1)*371))
        max_gap_delta=max(max_gap_delta,abs(expected_gap-float(gap.detach())))
        assert abs(expected_gap-float(gap.detach()))<1e-5 and np.array_equal(correct,mask.numpy())
        cg=torch.autograd.grad(ce,z,retain_graph=True)[0];gg=torch.autograd.grad(gap,z,retain_graph=True)[0]
        assert torch.allclose(cg,(z.detach().softmax(-1)-F.one_hot(truth,372))/64,atol=1e-7)
        assert (gg[torch.arange(64),truth]<=1e-7).all() and (gg[~mask]==0).all()
        total=ce+plan["feature_weight"]*feature+plan["gap_weight"]*gap
        assert torch.allclose(torch.autograd.grad(total,z)[0],cg+plan["gap_weight"]*gg,atol=1e-6)
    _,data,folds,_=run.direct.broad.previous.global_load(run.direct.broad.previous.OUTPUT)
    scored=result["metrics"]["digit_floor_main"]
    for fold in (1,2):
        z=_predict_logits(model,np.array(data["features"][folds[fold]],copy=True),torch.device("cpu"),32)
        assert np.isfinite(z).all() and np.array_equal(z,np.load(out / f"digit_floor_main_fold{fold}_logits.npy",allow_pickle=False))
        assert hit_metrics(z,np.array(data["labels"][folds[fold]],copy=True),vocab)==scored["development"][str(fold)]
    import audit_hwr_owned_formula_transfer_v21 as owned
    torch.set_num_threads(1);z=_predict_logits(model,np.load(owned.OUT / "inputs.npy",allow_pickle=False),torch.device("cpu"),owned.BATCH)
    assert np.isfinite(z).all() and np.array_equal(z,np.load(out / "digit_floor_main_owned_logits.npy",allow_pickle=False))
    records=json.loads((out / "digit_floor_main_owned_records.json").read_text(encoding="utf-8"));top=np.argsort(-z,axis=1,kind="stable")[:,:5]
    for r in records:
        tokens=[[vocab[int(i)] for i in indices] for indices in top[r["start"]:r["stop"]]]
        assert tokens==r["top5_tokens"] and [v[0] for v in tokens]==r["top1_tokens"]
    for cohort in ("all","legacy_96","codex_reviewed_53"):
        assert owned.metrics(records,cohort)==scored["owned"][cohort]
    assert sha(run.direct.broad.previous.CHECKPOINT)==plan["canonical_sha256"]
    cert=dict(schema="aiflow-digit-floor-verification/v31",status="pass",verifier_sha256=sha(Path(__file__)),
        comparison_sha256=sha(out / "comparison_result.json"),checkpoint_sha256=done["checkpoint_sha256"],training_steps=count,
        numeric_floor_derived_from_train_only=True,digit_exposures=digit_count,replaced_original_rows=2383,
        at_most_one_original_row_replaced_per_batch=True,all_48_augmented_rows_and_gt_bit_unchanged=True,
        all_68726_changed_view_slots_preserved=True,all_371_positive_classes_preserved=True,
        v29_training_code_unchanged=True,v29_certified_loss_weights_inherited_unchanged=True,
        all_2400_recorded_loss_formulas_checked=True,max_recorded_formula_delta=max_loss_delta,
        all_57_gradients_finite_and_all_57_tensors_updated=True,all_four_encoder_layers_active_every_step=True,
        full_weight_ce_and_masked_gap_gradient_checked_on_eight_train_batches=True,gap_numpy_double_max_delta=max_gap_delta,
        final_logits_reload_bit_exact=True,all_new_metrics_recounted=True,
        counts={n:dict(internal_top1=sum(m["development"][str(f)]["all"]["top1_hits"] for f in (1,2)),
            owned_formula_top1=m["owned"]["all"]["formula_top1_exact"],owned_formula_top5=m["owned"]["all"]["formula_top5_complete"])
            for n,m in result["metrics"].items()},
        limits="Recorded intermediate loss formulas, not a full historical forward replay. Consumed oracle-group diagnostic, not fresh human acceptance.",
        generated_labels_human_verified=False,owned_optimizer_rows=0,crohme_rows=0,canonical_checkpoint_unchanged=True,product_adopted=False)
    run.direct.broad.previous._write(out / "independent_verification.json",cert)
    print(json.dumps(cert),flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
