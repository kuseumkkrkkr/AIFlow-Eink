"""V23의 단계별 미분 범위·55개 비트 보존·개발/소유 입력 전이를 독립 검증한다."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import run_hwr_broad_head_only_v23 as run
from run_hwr_affine_distillation_experiment_v1 import _load_teacher,_predict_logits


def hit_metrics(scores,truth,labels):
    """전체 372-class의 tie-stable 정답 순위를 별도 계산하여 family별로 집계한다."""
    target=scores[np.arange(len(truth)),truth,None]
    rank=1+(scores>target).sum(1)+((scores==target)&(np.arange(372)[None]<truth[:,None])).sum(1)
    result={}
    for family in ("all","digits","latin_letters","math_symbols"):
        mask=np.ones(len(truth),bool) if family=="all" else np.array([
            run.broad.previous.aug.base._family(labels[int(c)])==family for c in truth])
        result[family]=dict(rows=int(mask.sum()),top1_hits=int((rank[mask]==1).sum()),top5_hits=int((rank[mask]<=5).sum()))
    return result


def main():
    """최종 checkpoint와 실제 encoder를 재실행하고 기록된 공식을 중간 forward 재현과 구분한다."""
    if (run.OUT / "independent_verification.json").exists():
        raise FileExistsError("certificate exists")
    plan,a=run.load()
    done=json.loads((run.OUT / "completed.json").read_text(encoding="utf-8"))
    result=json.loads((run.OUT / "comparison_result.json").read_text(encoding="utf-8"))
    assert done["steps"]==2400 and run.broad.previous._sha(run.OUT / "research.pt")==done["checkpoint_sha256"]
    history=run.OUT / "training_microscope.jsonl"
    assert run.broad.previous._sha(history)==done["microscope_sha256"]
    max_delta=0.;steps=0
    with history.open(encoding="utf-8") as stream:
        for line in stream:
            r=json.loads(line);steps+=1
            assert r["step"]==steps and len(r["encoder_layers"])==4 and len(r["parameter_gradient_l2"])==57
            assert r["frozen_nonhead_tensors"]==55 and r["all_nonhead_bit_equal"] is True
            g=r["parameter_gradient_l2"]
            assert all(v is None for n,v in g.items() if n not in run.detach.HEAD_NAMES)
            assert all(g[n] is not None and np.isfinite(g[n]) and g[n]>0 for n in run.detach.HEAD_NAMES)
            assert all(np.isfinite(v) for layer in r["encoder_layers"].values() for v in layer.values())
            error=abs(r["loss"]-(.7*r["ce_real"]+.2*r["kl_real"]+.1*r["kl_view"]))
            max_delta=max(max_delta,error);assert error<1e-6
            assert all(np.isfinite(r[n]) for n in ("loss","ce_real","kl_real","kl_view","gradient_l2"))
    assert steps==2400
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    model,labels,_=_load_teacher(run.OUT / "research.pt",torch.device("cpu"))
    canonical,original_labels,_=_load_teacher(run.broad.previous.CHECKPOINT,torch.device("cpu"))
    assert labels==original_labels
    original=canonical.state_dict();new=model.state_dict()
    frozen=[name for name in original if name not in run.detach.HEAD_NAMES]
    assert len(frozen)==55 and all(torch.equal(original[n].view(torch.int32),new[n].view(torch.int32)) for n in frozen)
    assert all(not torch.equal(original[n],new[n]) for n in run.detach.HEAD_NAMES)
    _,source,folds,_=run.broad.previous.global_load(run.broad.previous.OUTPUT)
    for fold in plan["development_folds"]:
        x=np.array(source["features"][folds[fold]],copy=True);truth=np.array(source["labels"][folds[fold]],copy=True)
        scores=_predict_logits(model,x,torch.device("cpu"),run.broad.BATCH)
        assert np.array_equal(scores,np.load(run.OUT / f"head_only_fold{fold}_logits.npy",allow_pickle=False))
        assert hit_metrics(scores,truth,labels)==result["development"][str(fold)]["head_only"]
        control_path=run.broad.OUT / "broad_augmented_validation_logits.npy" if fold==1 else run.broad.previous.ROOT / "artifacts/hwr_broad_margin_20261005_v17/broad_augmented_fold2_logits.npy"
        control=np.load(control_path,allow_pickle=False)
        assert hit_metrics(control,truth,labels)==result["development"][str(fold)]["v16"]
        before,after=control.argmax(1)==truth,scores.argmax(1)==truth
        assert result["development"][str(fold)]["paired_top1"]==dict(wins=int((~before&after).sum()),losses=int((before&~after).sum()),net=int(after.sum()-before.sum()))
    import audit_hwr_owned_formula_transfer_v21 as owned
    torch.set_num_threads(1)
    x=np.load(owned.OUT / "inputs.npy",allow_pickle=False)
    outputs=[]
    with torch.inference_mode():
        for start in range(0,len(x),owned.BATCH):
            outputs.append(model.encode(torch.from_numpy(x[start:start+owned.BATCH].copy())).numpy())
    embedding=np.concatenate(outputs)
    assert np.array_equal(embedding,np.load(owned.OUT / "canonical_embeddings.npy",allow_pickle=False))
    scores=_predict_logits(model,x,torch.device("cpu"),owned.BATCH)
    assert np.array_equal(scores,np.load(run.OUT / "head_only_owned_logits.npy",allow_pickle=False))
    records=json.loads((run.OUT / "head_only_owned_records.json").read_text(encoding="utf-8"))
    order=np.argsort(-scores,axis=1,kind="stable")[:,:5]
    for r in records:
        top=[[labels[int(i)] for i in row] for row in order[r["start"]:r["stop"]]]
        assert top==r["top5_tokens"] and [k[0] for k in top]==r["top1_tokens"]
    independently_recounted={c:owned.metrics(records,c) for c in ("all","legacy_96","codex_reviewed_53")}
    assert independently_recounted==result["owned"]
    baseline=json.loads((owned.OUT / "canonical_records.json").read_text(encoding="utf-8"))
    before=np.array([r["top1_tokens"]==r["truth_tokens"] for r in baseline]);after=np.array([r["top1_tokens"]==r["truth_tokens"] for r in records])
    assert result["owned_paired_vs_canonical"]==dict(wins=int((~before&after).sum()),losses=int((before&~after).sum()))
    assert run.broad.previous._sha(run.broad.previous.CHECKPOINT)==plan["canonical_sha256"]
    certificate=dict(schema="aiflow-broad-head-only-verification/v23",status="pass",
        verifier_sha256=run.broad.previous._sha(Path(__file__)),comparison_sha256=run.broad.previous._sha(run.OUT / "comparison_result.json"),
        all_2400_loss_arithmetic_records_rebuilt=True,loss_formula_max_delta=max_delta,
        all_2400_gradient_scopes_match_two_head_55_unused=True,final_55_nonhead_int32_bits_match_canonical=True,
        only_two_head_tensors_changed=True,owned_encoder_embeddings_bit_exact_with_canonical=True,
        internal_and_owned_checkpoint_reload_logits_bit_exact=True,all_metrics_and_paired_counts_recounted=True,
        owned_training_rows=0,new_human_acceptance=False,crohme_rows=0,product_adopted=False,
        limit="Recorded loss arithmetic, not historical intermediate forward replay. Internal and owned cases were consumed; 53 owned cases have Codex rather than human visual review.")
    run.broad.previous._write(run.OUT / "independent_verification.json",certificate)
    print(json.dumps(certificate),flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
