"""표현 보존 후 남은 실패를 층 출력·head 교차·정답/경쟁 margin으로 진단한다.

소비된 149식의 정답 그룹을 사용한 사후 진단이다. 학습·계수 선택·모델 승격은 없다.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch

import run_hwr_direct_feature_retention_v27 as retained
import audit_hwr_direct_calibration_retention_v26 as old
from run_hwr_affine_distillation_experiment_v1 import _load_teacher

OUT=retained.direct.broad.previous.ROOT / "artifacts/hwr_feature_margin_microscope_20261005_v28"


def traced(model,x):
    """실제 projection·4개 encoder layer·pooling을 같은 순서로 수행한다."""
    h=model.input_projection(x)+model.position;values=[h]
    for layer in model.encoder.layers:
        h=layer(h);values.append(h)
    if model.encoder.norm is not None:
        h=model.encoder.norm(h)
    w=torch.softmax(model.pool_score(h).squeeze(-1),dim=1)
    pooled=torch.sum(h*w.unsqueeze(-1),dim=1)
    return values+[pooled]


def main():
    """동일 입력의 head/층 복원과 margin 변화 분해를 검사하고 원본 모델 파일은 보존한다."""
    if OUT.exists():
        raise FileExistsError("refusing microscope overwrite")
    direct=retained.direct;sha=direct.broad.previous._sha;write=direct.broad.previous._write
    proof=json.loads((retained.OUT / "independent_verification.json").read_text(encoding="utf-8"))
    assert proof["status"]=="pass" and proof["comparison_sha256"]==sha(retained.OUT / "comparison_result.json")
    parent=json.loads((direct.OUT / "independent_verification.json").read_text(encoding="utf-8"))
    assert parent["status"]=="pass" and parent["comparison_sha256"]==sha(direct.OUT / "comparison_result.json")
    paths=dict(canonical=direct.broad.previous.CHECKPOINT,augmented=retained.BASE,
        retained=retained.OUT / "feature_retained_main/main_trained.pt")
    hashes={name:sha(p) for name,p in paths.items()}
    assert hashes["retained"]==proof["checkpoint_sha256"]
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    models={};vocab=None
    for name,p in paths.items():
        model,labels,_=_load_teacher(p,torch.device("cpu"));model.eval();models[name]=model
        if vocab is None:
            vocab=labels
        assert labels==vocab
    _,samples,rows=old.owned.load_source();x=np.load(old.owned.OUT / "inputs.npy",allow_pickle=False)
    assert x.shape==(579,128,5)
    truth=[t for row in rows for t in row["truth_tokens"]]
    assert len(truth)==len(x)
    OUT.mkdir()
    write(OUT / "frozen_plan.json",dict(schema="aiflow-feature-margin-microscope-plan/v28",script_sha256=sha(Path(__file__)),
        parent_v27_certificate_sha256=sha(retained.OUT / "independent_verification.json"),checkpoint_sha256=hashes,
        input_sha256=sha(old.owned.OUT / "inputs.npy"),formula_count=149,glyph_count=579,
        layer_names=["projection_and_position","encoder_0","encoder_1","encoder_2","encoder_3","pooling"],
        margin_path="Change encoder first with original head, then change head; rival fixed to retained model's actual wrong Top-1.",
        optimizer_steps=0,posthoc_consumed_diagnostic=True,coefficient_selection=False,product_adopted=False,crohme_rows=0))
    h={name:old.encode(m,x,old.owned.BATCH) for name,m in models.items()}
    assert np.array_equal(h["canonical"],np.load(old.owned.OUT / "canonical_embeddings.npy",allow_pickle=False))
    for name,saved in (("augmented","direct_baseline"),("retained","feature_retained_main")):
        assert np.array_equal(h[name],np.load(retained.OUT / f"{saved}_owned_embeddings.npy",allow_pickle=False))
    logits={};records={};metrics={}
    def score(name,embedding,head,expected=None):
        """교차 affine를 double 수치와 대조하고 저장된 top-k의 집계를 재확인한다."""
        z=old.head_scores(head,embedding,old.owned.BATCH)
        if expected is not None:
            assert np.array_equal(z,np.load(expected,allow_pickle=False))
        r=old.records_from_scores(z,vocab,samples,rows)
        top=np.argsort(-z,axis=1,kind="stable")[:,:5]
        for record in r:
            tokens=[[vocab[int(i)] for i in indices] for indices in top[record["start"]:record["stop"]]]
            assert tokens==record["top5_tokens"] and [v[0] for v in tokens]==record["top1_tokens"]
        logits[name]=z;records[name]=r
        metrics[name]={cohort:old.owned.metrics(r,cohort) for cohort in ("all","legacy_96","codex_reviewed_53")}
        np.save(OUT / f"{name}_logits.npy",z,allow_pickle=False);write(OUT / f"{name}_records.json",r)
        print(json.dumps(dict(event="margin_probe",variant=name,metrics=metrics[name]["all"])),flush=True)
    score("canonical",h["canonical"],models["canonical"].math_head,direct.OUT / "canonical_owned_logits.npy")
    score("augmented",h["augmented"],models["augmented"].math_head,direct.OUT / "augmented_main_owned_logits.npy")
    score("retained",h["retained"],models["retained"].math_head,retained.OUT / "feature_retained_main_owned_logits.npy")
    score("canonical_encoder_retained_head",h["canonical"],models["retained"].math_head)
    score("retained_encoder_canonical_head",h["retained"],models["canonical"].math_head)
    state=copy.deepcopy(models["retained"].state_dict())
    for i in range(4):
        layer=models["retained"].encoder.layers[i]
        original=copy.deepcopy(layer.state_dict())
        try:
            layer.load_state_dict(models["canonical"].encoder.layers[i].state_dict(),strict=True)
            score(f"restore_only_encoder_{i}",old.encode(models["retained"],x,old.owned.BATCH),models["retained"].math_head)
        finally:
            layer.load_state_dict(original,strict=True)
    assert all(torch.equal(v,models["retained"].state_dict()[k]) for k,v in state.items())
    names=["projection_and_position","encoder_0","encoder_1","encoder_2","encoder_3","pooling"]
    per_stage={name:[] for name in names};trace_delta=0.
    with torch.inference_mode():
        for start in range(0,len(x),old.owned.BATCH):
            tx=torch.from_numpy(x[start:start+old.owned.BATCH].copy())
            a=traced(models["canonical"],tx);b=traced(models["retained"],tx)
            for name,aa,bb in zip(names,a,b,strict=True):
                assert torch.isfinite(aa).all() and torch.isfinite(bb).all()
                axes=tuple(range(1,aa.ndim));per_stage[name].extend((bb-aa).square().mean(dim=axes).numpy().tolist())
            for name,values in (("canonical",a),("retained",b)):
                delta=float(np.max(np.abs(values[-1].numpy()-h[name][start:start+old.owned.BATCH])))
                trace_delta=max(trace_delta,delta);assert delta<1e-5
                zz=models[name].math_head(values[-1]).numpy()
                assert np.array_equal(np.argsort(-zz,axis=1,kind="stable")[:,:5],
                    np.argsort(-logits[name][start:start+old.owned.BATCH],axis=1,kind="stable")[:,:5])
    old_top=logits["canonical"].argmax(1);new_top=logits["retained"].argmax(1)
    label_index={t:i for i,t in enumerate(vocab)}
    margins=[];lost_indices=[];unsupported=0;max_error=0.
    wc=models["canonical"].math_head.weight.detach().numpy().astype(np.float64)
    wr=models["retained"].math_head.weight.detach().numpy().astype(np.float64)
    bc=models["canonical"].math_head.bias.detach().numpy().astype(np.float64)
    br=models["retained"].math_head.bias.detach().numpy().astype(np.float64)
    for j,t in enumerate(truth):
        if t not in label_index:
            unsupported+=1;continue
        gold=label_index[t];rival=int(new_top[j])
        if int(old_top[j])!=gold or rival==gold:
            continue
        hc=h["canonical"][j].astype(np.float64);hr=h["retained"][j].astype(np.float64)
        before=float((wc[gold]-wc[rival])@hc+bc[gold]-bc[rival])
        after=float((wr[gold]-wr[rival])@hr+br[gold]-br[rival])
        encoder=float((wc[gold]-wc[rival])@(hr-hc))
        head=float(((wr[gold]-wr[rival])-(wc[gold]-wc[rival]))@hr+(br[gold]-br[rival])-(bc[gold]-bc[rival]))
        error=abs(after-before-encoder-head);max_error=max(max_error,error);assert error<1e-10
        assert before>0 and after<0
        lost_indices.append(j)
        margins.append(dict(glyph_index=j,truth=t,rival=vocab[rival],canonical_margin=before,retained_margin=after,
            encoder_margin_delta=encoder,head_margin_delta=head,feature_mse=float(np.mean((hr-hc)**2))))
    write(OUT / "lost_correct_margin_records.json",margins)
    stage_summary={n:dict(all_glyph_mean_mse=float(np.mean(v)),lost_correct_glyph_mean_mse=float(np.mean(np.array(v)[lost_indices])))
        for n,v in per_stage.items()}
    np.save(OUT / "per_glyph_stage_mse.npy",np.array([per_stage[n] for n in names]).T,allow_pickle=False)
    tax=old.taxonomy(records["canonical"],records["retained"])
    assert sum(tax["lost_by_truth"].values())==len(margins)
    assert all(sha(p)==hashes[n] for n,p in paths.items())
    result=dict(schema="aiflow-feature-margin-microscope-result/v28",status="completed_verified_consumed_diagnostic",
        metrics=metrics,taxonomy=tax,stage_mse=stage_summary,
        lost_correct_supported_glyphs=len(margins),unsupported_truth_glyphs=unsupported,
        negative_encoder_delta_count=sum(r["encoder_margin_delta"]<0 for r in margins),
        negative_head_delta_count=sum(r["head_margin_delta"]<0 for r in margins),
        original_head_already_wrong_after_encoder_change_count=sum(r["canonical_margin"]+r["encoder_margin_delta"]<0 for r in margins),
        margin_decomposition_max_error=max_error,trace_vs_actual_pooled_max_abs_error=trace_delta,
        all_traced_ordered_top5_match_actual=True,all_diagonal_logits_bit_exact_with_frozen_results=True,
        all_variant_top5_match_independent_double_affine=True,model_memory_restored=True,checkpoint_files_unchanged=True,
        optimizer_steps=0,coefficient_selection=False,crohme_rows=0,product_adopted=False,new_human_acceptance=False,
        limits="Layer outputs include upstream changes; one-layer restoration mixes coadapted components and is not an isolated causal training experiment. Margins are posthoc diagnostics, not router threshold or training selection.")
    write(OUT / "microscope_result.json",result)
    print(json.dumps(dict(event="margin_microscope_completed",formula_hits={n:m["all"]["formula_top1_exact"] for n,m in metrics.items()},
        margin_summary={k:result[k] for k in ("lost_correct_supported_glyphs","negative_encoder_delta_count","negative_head_delta_count",
            "original_head_already_wrong_after_encoder_change_count","margin_decomposition_max_error")},stage_mse=stage_summary)),flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
