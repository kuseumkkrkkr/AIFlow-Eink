"""V20 고정 mask·전체 단계 수치·최종 가중치·소비된 개발 fold 결과를 독립 검증한다."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import run_hwr_broad_teacher_correct_v20 as run
from run_hwr_affine_distillation_experiment_v1 import _load_teacher, _predict_logits


def main():
    """기록된 공식은 재계산하되 과거 중간 forward까지 재현했다고 주장하지 않는다."""
    if (run.OUT / "independent_verification.json").exists():
        raise FileExistsError("certificate exists")
    plan,a = run.load()
    done = json.loads((run.OUT / "completed.json").read_text(encoding="utf-8"))
    result = json.loads((run.OUT / "comparison_result.json").read_text(encoding="utf-8"))
    assert done["steps"] == run.broad.STEPS
    assert run.broad.previous._sha(run.OUT / "research.pt") == done["checkpoint_sha256"]
    history = run.OUT / "training_microscope.jsonl"
    assert run.broad.previous._sha(history) == done["microscope_sha256"]
    total_wrong = 0;max_delta = 0.;steps = 0
    with history.open(encoding="utf-8") as stream:
        for line in stream:
            r = json.loads(line)
            ids = a["shared_schedule"][steps];steps += 1
            assert r["step"] == steps and len(r["encoder_layers"]) == 4 and len(r["parameter_gradient_l2"]) == 57
            truth = a["population_labels"][ids]
            teacher = a["population_teacher_logits"][ids]
            target = teacher[np.arange(len(truth)),truth,None]
            ranks = 1+(teacher>target).sum(1)+((teacher==target)&(np.arange(372)[None]<truth[:,None])).sum(1)
            correct = int((ranks==1).sum())
            assert r["teacher_correct_rows"] == correct
            total_wrong += len(ids)-correct
            delta = abs(r["loss"]-(.7*r["ce_real"]+.2*r["kl_real"]+.1*r["kl_view"]))
            assert delta < 1e-6
            max_delta = max(max_delta,delta)
            for i in range(4):
                assert all(np.isfinite(v) for v in r["encoder_layers"][f"encoder.layers.{i}"].values())
                gradients = [v for key,v in r["parameter_gradient_l2"].items() if key.startswith(f"encoder.layers.{i}.")]
                assert len(gradients)==12 and all(v is not None and np.isfinite(v) for v in gradients) and max(gradients)>0
            assert all(np.isfinite(r[k]) for k in ("loss","ce_real","kl_real","kl_view","gradient_l2"))
    assert steps==2400 and total_wrong==plan["expected_incorrect_original_exposures"]==14952
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    model,labels,_ = _load_teacher(run.OUT / "research.pt",torch.device("cpu"))
    canonical,_,_ = _load_teacher(run.broad.previous.CHECKPOINT,torch.device("cpu"))
    control,_,_ = _load_teacher(run.broad.OUT / "broad_augmented/research.pt",torch.device("cpu"))
    changed = {str(i):sum(not torch.equal(canonical.state_dict()[k],p) for k,p in model.state_dict().items()
        if k.startswith(f"encoder.layers.{i}.")) for i in range(4)}
    assert all(v==12 for v in changed.values())
    _,source,folds,_ = run.broad.previous.global_load(run.broad.previous.OUTPUT)
    for fold in plan["development_folds"]:
        x = np.array(source["features"][folds[fold]],copy=True)
        y = np.array(source["labels"][folds[fold]],copy=True)
        z = _predict_logits(model,x,torch.device("cpu"),run.broad.BATCH)
        saved = np.load(run.OUT / f"teacher_correct_fold{fold}_logits.npy",allow_pickle=False)
        assert np.array_equal(z,saved)
        c = _predict_logits(control,x,torch.device("cpu"),run.broad.BATCH)
        d = result["folds"][str(fold)]
        metrics = {}
        for name,scores in (("control",c),("teacher_correct",z)):
            target = scores[np.arange(len(y)),y,None]
            ranks = 1+(scores>target).sum(1)+((scores==target)&(np.arange(372)[None]<y[:,None])).sum(1)
            metrics[name] = {}
            for family in ("all","digits","latin_letters","math_symbols"):
                mask = np.ones(len(y),bool) if family=="all" else np.array([run.broad.previous.aug.base._family(labels[int(i)])==family for i in y])
                metrics[name][family] = dict(rows=int(mask.sum()),top1_hits=int((ranks[mask]==1).sum()),top5_hits=int((ranks[mask]<=5).sum()))
            assert metrics[name] == d[name]
        before,after = c.argmax(1)==y,z.argmax(1)==y
        assert d["paired_top1"] == dict(wins=int((~before&after).sum()),losses=int((before&~after).sum()),net=int(after.sum()-before.sum()))
    assert run.broad.previous._sha(run.broad.previous.CHECKPOINT) == plan["canonical_sha256"]
    assert run.broad.previous._sha(run.broad.OUT / "broad_augmented/research.pt") == plan["control_checkpoint_sha256"]
    certificate = dict(schema="aiflow-teacher-correct-verification/v20",status="pass",
        verifier_sha256=run.broad.previous._sha(Path(__file__)),result_sha256=run.broad.previous._sha(run.OUT / "comparison_result.json"),
        all_2400_immutable_masks_rebuilt=True,incorrect_original_exposures=total_wrong,loss_formula_max_delta=max_delta,
        all_four_encoder_layers_received_finite_gradients=True,changed_encoder_tensors_by_layer=changed,
        checkpoint_reload_logits_bit_exact=True,control_checkpoint_outputs_and_all_stable_ranks_reproduced=True,
        development_folds=[1,2],all_folds_previously_consumed=True,fresh_acceptance=False,product_adopted=False,
        optimizer_steps_in_verification=0,official_test_rows_read=0,crohme_rows=0,
        limit="Loss arithmetic and fixed masks reconstructed from records; historical intermediate forwards not replayed; development comparison has selection bias.")
    run.broad.previous._write(run.OUT / "independent_verification.json",certificate)
    print(json.dumps(certificate),flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
