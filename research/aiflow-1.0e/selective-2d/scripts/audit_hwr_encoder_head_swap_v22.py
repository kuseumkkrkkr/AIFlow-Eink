"""봉인된 encoder/pooling과 선형 head를 3x3 교차해 소유 입력 퇴행을 분리한다.

학습·모델 파일 변경 없이 소비된 149식을 진단한다. 53식의 Codex 검수 한계를
유지하며, 교차 연결을 새 제품 후보나 인과적 아키텍처 증명으로 해석하지 않는다.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch

import audit_hwr_owned_formula_transfer_v21 as previous
from evaluate_48hz_prefix_v1 import _load_model
from run_hwr_affine_distillation_experiment_v1 import _predict_logits

OUT = previous.broad.previous.ROOT / "artifacts/hwr_encoder_head_swap_20261005_v22"
NAMES = ("canonical","v16","v20")


def records_from_scores(scores, vocabulary, samples, rows):
    """동일 oracle 그룹과 기존 decoder로 후보만 비교하며 372-class 밖 토큰을 만들지 않는다."""
    probabilities = torch.from_numpy(scores).softmax(1).numpy()
    ordering = np.argsort(-scores,axis=1,kind="stable")[:,:5]
    records = []
    for sample,row in zip(samples,rows,strict=True):
        symbols = []
        for group,indices,p in zip(row["groups"],ordering[row["start"]:row["stop"]],
            probabilities[row["start"]:row["stop"]],strict=True):
            symbols.append(dict(stroke_indices=group,hwr_topk=[vocabulary[int(i)] for i in indices],
                hwr_topk_probabilities=[float(p[i]) for i in indices],geometry=previous.oracle._geometry(sample.strokes,group)))
        decoded = previous.oracle.decode_selective_partition(sample.sample_id,row["groups"],symbols,stroke_count=len(sample.strokes))
        assert not decoded.get("accepted") or all(t in s["hwr_topk"] for t,s in zip(decoded["tokens"],symbols,strict=True))
        records.append(dict(**row,top1_tokens=[s["hwr_topk"][0] for s in symbols],
            top5_tokens=[s["hwr_topk"] for s in symbols],decoder=decoded))
    return records


def attribution(old_embedding,new_embedding,old_weight,new_weight,old_bias,new_bias,old_scores,new_scores,truth):
    """최종 모델의 최고 비정답 rival을 고정해 선형 margin 변화의 세 항을 정확히 분해한다."""
    valid = truth>=0
    rows = np.flatnonzero(valid);target = truth[valid]
    competitor = new_scores[valid].astype(np.float64).copy()
    competitor[np.arange(len(rows)),target] = -np.inf
    rival = competitor.argmax(1)
    h,hn = old_embedding[valid].astype(np.float64),new_embedding[valid].astype(np.float64)
    w,wn = old_weight.astype(np.float64),new_weight.astype(np.float64)
    b,bn = old_bias.astype(np.float64),new_bias.astype(np.float64)
    dh,dw,db = hn-h,wn-w,bn-b
    encoder = np.sum(dh*(w[target]-w[rival]),axis=1)
    head = np.sum(h*(dw[target]-dw[rival]),axis=1)+db[target]-db[rival]
    interaction = np.sum(dh*(dw[target]-dw[rival]),axis=1)
    before = old_scores[rows,target]-old_scores[rows,rival]
    after = new_scores[rows,target]-new_scores[rows,rival]
    error = np.abs(encoder+head+interaction-(after-before))
    assert np.isfinite(error).all() and float(error.max())<1e-4
    old_correct = old_scores[valid].argmax(1)==target
    new_correct = new_scores[valid].argmax(1)==target
    lost = old_correct & ~new_correct
    assert (before[lost]>=0).all() and (after[lost]<=0).all()
    detail = dict(row_indices=rows,target=target,rival=rival,encoder=encoder,head=head,interaction=interaction,
        margin_before=before,margin_after=after,canonical_correct=old_correct,new_correct=new_correct)
    summary = dict(eligible_tokens=len(rows),unsupported_tokens=int((~valid).sum()),canonical_correct_to_new_wrong=int(lost.sum()),
        margin_reconstruction_max_delta=float(error.max()),rival_policy="final model's highest nontruth rival, frozen across decomposition",
        regressed=dict(encoder_mean=float(encoder[lost].mean()),head_mean=float(head[lost].mean()),
            interaction_mean=float(interaction[lost].mean()),encoder_negative=int((encoder[lost]<0).sum()),
            head_negative=int((head[lost]<0).sum()),interaction_negative=int((interaction[lost]<0).sum())) if lost.any() else {})
    return detail,summary


def main():
    """교차 head를 실제 전체 forward로 재계산하고 독립 double affine 연산과 대조한다."""
    if OUT.exists():
        raise FileExistsError("refusing component diagnostic overwrite")
    parent = json.loads((previous.OUT / "frozen_plan.json").read_text(encoding="utf-8"))
    certificate = json.loads((previous.OUT / "independent_verification.json").read_text(encoding="utf-8"))
    assert certificate["status"]=="pass"
    required = ["inputs.npy","input_rows.json","transfer_result.json",*[
        f"{name}_{kind}.npy" for name in NAMES for kind in ("logits","embeddings")]]
    for name in required:
        assert previous.broad.previous._sha(previous.OUT / name)==certificate["artifacts_sha256"][name]
    _,samples,rows = previous.load_source()
    x = np.load(previous.OUT / "inputs.npy",allow_pickle=False)
    assert rows==json.loads((previous.OUT / "input_rows.json").read_text(encoding="utf-8"))
    paths = dict(canonical=previous.broad.previous.CHECKPOINT,v16=previous.broad.OUT / "broad_augmented/research.pt",
        v20=previous.broad.previous.ROOT / "artifacts/hwr_broad_teacher_correct_20261005_v20/research.pt")
    models = {};heads = {};cached = {};embedding = {}
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    for name,path in paths.items():
        assert previous.broad.previous._sha(path)==parent["model_sha256"][name]
        model,labels,_ = _load_model(path,torch.device("cpu"))
        assert model.math_head.in_features==128 and model.math_head.out_features==372
        if name=="canonical":
            vocabulary=labels
        assert labels==vocabulary
        models[name]=model;heads[name]=copy.deepcopy(model.math_head.state_dict())
        cached[name]=np.load(previous.OUT / f"{name}_logits.npy",allow_pickle=False)
        embedding[name]=np.load(previous.OUT / f"{name}_embeddings.npy",allow_pickle=False)
    OUT.mkdir()
    previous.broad.previous._write(OUT / "frozen_plan.json",dict(schema="aiflow-component-swap-plan/v22",
        script_sha256=previous.broad.previous._sha(Path(__file__)),parent_certificate_sha256=previous.broad.previous._sha(previous.OUT / "independent_verification.json"),
        parent_input_sha256=certificate["artifacts_sha256"]["inputs.npy"],model_sha256=parent["model_sha256"],
        matrix="3 encoder/pooling states x 3 heads, no fitting or mixed checkpoint export",batch_size=previous.BATCH,threads=1,
        held_encoder_tensors=55,replaced_head_tensors=2,already_consumed=True,new_human_acceptance=False,
        codex_reviewed_subset=53,training_steps=0,crohme_rows=0,product_adopted=False))
    summaries = {};scores = {};max_errors = {};baseline = json.loads((previous.OUT / "transfer_result.json").read_text(encoding="utf-8"))
    for encoder_name,model in models.items():
        original = {key:value.detach().clone() for key,value in model.state_dict().items()}
        nonhead = [key for key in original if not key.startswith("math_head.")]
        assert len(nonhead)==55
        summaries[encoder_name]={}
        for head_name,state in heads.items():
            model.math_head.load_state_dict(state,strict=True)
            assert all(torch.equal(original[key],model.state_dict()[key]) for key in nonhead)
            z = _predict_logits(model,x,torch.device("cpu"),previous.BATCH)
            affine = embedding[encoder_name].astype(np.float64) @ state["weight"].numpy().astype(np.float64).T + state["bias"].numpy().astype(np.float64)
            delta = float(np.abs(z-affine).max());assert delta<1e-4
            # 두 구현의 전체 Top-5 순위도 같아야 수치 오차를 인식 차이로 착각하지 않는다.
            assert np.array_equal(np.argsort(-z,axis=1,kind="stable")[:,:5],np.argsort(-affine,axis=1,kind="stable")[:,:5])
            key=f"{encoder_name}__{head_name}";scores[key]=z;max_errors[key]=delta
            records=records_from_scores(z,vocabulary,samples,rows)
            summaries[encoder_name][head_name]={cohort:previous.metrics(records,cohort) for cohort in ("all","legacy_96","codex_reviewed_53")}
            if encoder_name==head_name:
                assert np.array_equal(z,cached[encoder_name]) and summaries[encoder_name][head_name]==baseline["metrics"][encoder_name]
            np.save(OUT / f"{key}_logits.npy",z,allow_pickle=False)
            previous.broad.previous._write(OUT / f"{key}_records.json",records)
            print(json.dumps(dict(event="component_swap",encoder=encoder_name,head=head_name,
                metrics=summaries[encoder_name][head_name]["all"])),flush=True)
        model.load_state_dict(original,strict=True)
        assert all(torch.equal(value,model.state_dict()[key]) for key,value in original.items())
    truth=np.array([vocabulary.index(t) if t in vocabulary else -1 for row in rows for t in row["truth_tokens"]])
    attribution_summaries={}
    for name in ("v16","v20"):
        detail,report=attribution(embedding["canonical"],embedding[name],heads["canonical"]["weight"].numpy(),
            heads[name]["weight"].numpy(),heads["canonical"]["bias"].numpy(),heads[name]["bias"].numpy(),
            cached["canonical"],cached[name],truth)
        np.savez_compressed(OUT / f"{name}_margin_attribution.npz",**detail)
        attribution_summaries[name]=report
    assert all(previous.broad.previous._sha(path)==parent["model_sha256"][name] for name,path in paths.items())
    result=dict(schema="aiflow-component-swap-result/v22",status="verified_component_diagnostic_not_adoption",
        metrics=summaries,margin_attribution=attribution_summaries,all_diagonal_outputs_bit_exact_with_v21=True,
        all_encoder_tensors_unchanged_during_head_swap=True,all_nine_full_forwards_match_independent_affine_top5=True,
        affine_max_absolute_delta=max_errors,model_memory_restored=True,checkpoint_files_unchanged=True,
        training_steps=0,crohme_rows=0,new_human_acceptance=False,product_adopted=False,
        limits="Swapped components can have calibration mismatch; counterfactual scores and exact linear attribution do not prove causal training failure or architecture insufficiency. Same consumed oracle groups, not fresh writer/device/formula acceptance.")
    previous.broad.previous._write(OUT / "component_result.json",result)
    print(json.dumps(dict(event="component_diagnostic_complete",matrix={e:{h:m["all"]["formula_top1_exact"]
        for h,m in values.items()} for e,values in summaries.items()},margin_attribution=attribution_summaries)),flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
