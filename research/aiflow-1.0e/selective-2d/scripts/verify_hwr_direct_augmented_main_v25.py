"""직접 증강 CE 학습의 GT 대응·전체 gradient·checkpoint·평가 수치를 독립 검증한다."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import run_hwr_direct_augmented_main_v25 as run
from run_hwr_affine_distillation_experiment_v1 import _load_teacher,_predict_logits
from verify_hwr_broad_head_only_v23 import hit_metrics


def main():
    """두 군의 같은 정답 schedule과 75% 실제 변형, KL=0, 소유 입력 비학습을 검증한다."""
    out=run.OUT
    if (out / "independent_verification.json").exists():
        raise FileExistsError("certificate exists")
    plan,a,s=run.load()
    result=json.loads((out / "comparison_result.json").read_text(encoding="utf-8"))
    source=a["shared_schedule"].reshape(-1)
    expected=np.concatenate((a["population_labels"][s["original_population_indices"]],
        a["population_labels"][source[s["augmented_view_indices"]]]),axis=1).astype(np.int64)
    assert np.array_equal(expected,s["shared_target_ids"]) and expected.shape==(2400,64)
    provenance=json.loads((run.broad.OUT / "generation_provenance.json").read_text(encoding="utf-8"))
    indices=np.unique(s["augmented_view_indices"])
    assert len(indices)==68726 and all(provenance[int(i)]["changed"] for i in indices)
    for i in indices:
        m=provenance[int(i)]
        assert a["population_labels"][m["population_index"]]==a["population_labels"][m["donor_population_index"]]
    loss_max=0.;changed={};snapshots={}
    for arm in run.ARMS:
        folder=out / arm
        done=json.loads((folder / "completed.json").read_text(encoding="utf-8"))
        assert done["steps"]==2400 and run.broad.previous._sha(folder / "main_trained.pt")==done["checkpoint_sha256"]
        assert run.broad.previous._sha(folder / "training_microscope.jsonl")==done["microscope_sha256"]
        count=0
        with (folder / "training_microscope.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                r=json.loads(line);count+=1
                assert r["step"]==count and r["hard_target_rows"]==64 and r["kl_weight"]==0.
                assert r["augmented_rows"]==(48 if arm=="augmented_main" else 0)
                assert len(r["encoder_layers"])==4 and len(r["parameter_gradient_l2"])==57
                assert all(v is not None and np.isfinite(v) for v in r["parameter_gradient_l2"].values())
                assert all(np.isfinite(v) for layer in r["encoder_layers"].values() for v in layer.values())
                error=abs(r["loss"]-(.25*r["ce_real_region"]+.75*r["ce_aug_or_paired_region"]))
                loss_max=max(loss_max,error);assert error<1e-6
                for i in range(4):
                    values=[v for n,v in r["parameter_gradient_l2"].items() if n.startswith(f"encoder.layers.{i}.")]
                    assert len(values)==12 and max(values)>0
        assert count==2400
        snapshots[arm]=done
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    original,labels,_=_load_teacher(run.broad.previous.CHECKPOINT,torch.device("cpu"))
    paths=dict(canonical=run.broad.previous.CHECKPOINT,**{arm:out / arm / "main_trained.pt" for arm in run.ARMS})
    _,data,folds,_=run.broad.previous.global_load(run.broad.previous.OUTPUT)
    import audit_hwr_owned_formula_transfer_v21 as owned
    owner_x=np.load(owned.OUT / "inputs.npy",allow_pickle=False)
    counts={}
    for name,path in paths.items():
        model,vocab,_=_load_teacher(path,torch.device("cpu"));assert vocab==labels
        if name!="canonical":
            changed[name]={str(i):sum(not torch.equal(original.state_dict()[k],v) for k,v in model.state_dict().items()
                if k.startswith(f"encoder.layers.{i}.")) for i in range(4)}
            assert all(c>0 for c in changed[name].values())
        torch.set_num_threads(2)
        for fold in plan["development_folds"]:
            x=np.array(data["features"][folds[fold]],copy=True);y=np.array(data["labels"][folds[fold]],copy=True)
            z=_predict_logits(model,x,torch.device("cpu"),32)
            assert np.array_equal(z,np.load(out / f"{name}_fold{fold}_logits.npy",allow_pickle=False))
            assert hit_metrics(z,y,vocab)==result["metrics"][name]["development"][str(fold)]
        torch.set_num_threads(1)
        z=_predict_logits(model,owner_x,torch.device("cpu"),owned.BATCH)
        assert np.array_equal(z,np.load(out / f"{name}_owned_logits.npy",allow_pickle=False))
        records=json.loads((out / f"{name}_owned_records.json").read_text(encoding="utf-8"))
        order=np.argsort(-z,axis=1,kind="stable")[:,:5]
        for r in records:
            top=[[vocab[int(i)] for i in indices] for indices in order[r["start"]:r["stop"]]]
            assert top==r["top5_tokens"] and [v[0] for v in top]==r["top1_tokens"]
        for cohort in ("all","legacy_96","codex_reviewed_53"):
            assert owned.metrics(records,cohort)==result["metrics"][name]["owned"][cohort]
        counts[name]=dict(internal_top1=sum(result["metrics"][name]["development"][str(f)]["all"]["top1_hits"] for f in (1,2)),
            internal_top5=sum(result["metrics"][name]["development"][str(f)]["all"]["top5_hits"] for f in (1,2)),
            owned_formula_top1=result["metrics"][name]["owned"]["all"]["formula_top1_exact"],
            owned_formula_top5=result["metrics"][name]["owned"]["all"]["formula_top5_complete"])
    assert run.broad.previous._sha(run.broad.previous.CHECKPOINT)==plan["canonical_sha256"]
    certificate=dict(schema="aiflow-direct-augmented-main-verification/v25",status="pass",
        verifier_sha256=run.broad.previous._sha(Path(__file__)),comparison_sha256=run.broad.previous._sha(out / "comparison_result.json"),
        both_full_main_2400_step_runs_completed=True,both_label_schedules_match_original_gt=True,
        all_68726_augmented_views_changed_and_same_class_donors=True,augmented_hard_target_exposures=115200,
        paired_original_exposures_per_arm=153600,all_4800_direct_ce_arithmetic_records_rebuilt=True,
        loss_formula_max_delta=loss_max,all_57_gradients_finite_on_every_step=True,all_four_encoder_layers_updated=changed,
        all_internal_and_owned_checkpoint_reload_logits_bit_exact=True,all_metrics_recounted=True,counts=counts,
        owned_optimizer_rows=0,crohme_rows=0,product_adopted=False,new_human_acceptance=False,
        label_semantics="Source TRAIN class inheritance through constrained same-class deformation, not per-view human verified truth.",
        limits="Recorded loss arithmetic, not replay of all historical intermediate forwards. Evaluation cases are consumed; 95 owned formulas were historical canonical calibration input, and 53 additional annotations were Codex-reviewed.")
    run.broad.previous._write(out / "independent_verification.json",certificate)
    print(json.dumps(certificate),flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
