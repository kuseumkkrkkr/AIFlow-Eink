"""Freeze a class-agnostic writer-residual candidate promoter before reserve outer access."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from build_cleanroom_writer_style_bank_v5 import sha256
from train_cleanroom_writer_adapter_v9 import SUPPORT_SIZES, _episode_indices
from train_writer_candidate_promoter_v9 import CandidatePromoter, SEED, THRESHOLDS
from evaluate_writer_candidate_promoter_extension_v9 import _evaluate_detailed, _global_map


ROOT=Path(__file__).resolve().parents[1]
OLD_CACHE=ROOT/"artifacts/writer_adaptation_v9_frozen_cache_20260823_r1.npz"
CONSUMED_CACHE=ROOT/"artifacts/writer_adaptation_v9_extension32_frozen_cache_20260823_r1.npz"
OLD_BANK=ROOT/"artifacts/cleanroom_writer_style_v5_20260823_smoke32_r4_r2"
EXTENSION_BANK=ROOT/"artifacts/cleanroom_writer_style_v5_20260823_extension32_r4_r1"
R1_RECEIPT=ROOT/"artifacts/writer_candidate_promoter_v9_extension_outer_20260823_r1/outer_open_receipt.json"
DEFAULT_OUTPUT=ROOT/"artifacts/writer_residual_promoter_v9_20260823_r1_frozen"
FEATURE_NAMES=("baseline_margin","normalized_margin","candidate_rank","candidate_centered_logit","candidate_adjusted_cosine","baseline_adjusted_cosine","adjusted_cosine_advantage","candidate_global_reliability","baseline_global_reliability","writer_residual_norm","writer_residual_dispersion","top5_entropy","top1_top2_margin","top5_spread")


def load_cache(path):
    with np.load(path,allow_pickle=False) as p:return {k:np.asarray(p[k]).copy() for k in ("labels","writers","splits","embeddings","logits")}
def normalize(x):return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-8)
def top5(x):return np.argsort(x,axis=1)[:,-5:][:,::-1]


def prototype_stats(data,fit_writers):
    emb=normalize(data["embeddings"]); mask=np.isin(data["writers"],fit_writers); classes=data["logits"].shape[1]
    sums=np.zeros((classes,emb.shape[1]),np.float64);counts=np.zeros(classes,np.int64);writer_sums={};writer_counts={}
    for writer in fit_writers:
        local=data["writers"]==writer; ws=np.zeros_like(sums);wc=np.zeros_like(counts)
        np.add.at(ws,data["labels"][local],emb[local]);np.add.at(wc,data["labels"][local],1);writer_sums[writer]=ws;writer_counts[writer]=wc;sums+=ws;counts+=wc
    prototypes=sums/np.maximum(counts[:,None],1);prototypes=normalize(prototypes.astype(np.float32))
    reliability=np.zeros(classes,np.float32)
    for label in np.flatnonzero(counts):
        rows=mask&(data["labels"]==label);reliability[label]=float(np.mean(emb[rows]@prototypes[label]))
    return {"sums":sums,"counts":counts,"writer_sums":writer_sums,"writer_counts":writer_counts,"prototypes":prototypes,"reliability":reliability}


def prototypes_for(stats,exclude=None):
    sums=stats["sums"].copy();counts=stats["counts"].copy()
    if exclude in stats["writer_sums"]:sums-=stats["writer_sums"][exclude];counts-=stats["writer_counts"][exclude]
    values=sums/np.maximum(counts[:,None],1);return normalize(values.astype(np.float32)),counts


def episode_rows(writer,episode,calibration,data,stats,exclude_global_writer=None,self_check=False):
    labels,embeddings,logits,writers,splits=(data[k] for k in ("labels","embeddings","logits","writers","splits"));norm=normalize(embeddings)
    prototypes,counts=prototypes_for(stats,exclude_global_writer);query=calibration if self_check else np.flatnonzero((writers==writer)&(splits==1))
    ranks=top5(logits[query]);values=np.take_along_axis(logits[query],ranks,axis=1);spread=np.maximum(values.std(axis=1),1e-6)
    shifted=values-values.max(axis=1,keepdims=True);p=np.exp(shifted);p/=p.sum(axis=1,keepdims=True);entropy=-np.sum(p*np.log(np.maximum(p,1e-12)),axis=1)
    features=[];targets=[];records=[];decisions=[]
    for local,index in enumerate(query.tolist()):
        residual_rows=[row for row in calibration.tolist() if (not self_check or row!=index) and counts[int(labels[row])]>0]
        if residual_rows:
            residual_matrix=norm[residual_rows]-prototypes[labels[residual_rows]];residual=np.median(residual_matrix,axis=0)
            dispersion=float(np.median(np.linalg.norm(residual_matrix-residual,axis=1)))
        else:residual=np.zeros(norm.shape[1],np.float32);dispersion=0.0
        adjusted=normalize((prototypes+residual[None]).astype(np.float32));baseline=int(ranks[local,0]);base_cos=float(norm[index]@adjusted[baseline])
        decisions.append({"writer":writer,"episode":episode,"index":index,"baseline":baseline,"truth":int(labels[index])})
        for rank in range(1,5):
            candidate=int(ranks[local,rank])
            if counts[candidate]<4 or counts[baseline]<4:continue
            cosine=float(norm[index]@adjusted[candidate]);margin=float(values[local,0]-values[local,rank])
            features.append([margin,margin/spread[local],rank/4.0,float((values[local,rank]-values[local].mean())/spread[local]),cosine,base_cos,cosine-base_cos,
                             float(stats["reliability"][candidate]),float(stats["reliability"][baseline]),float(np.linalg.norm(residual)),dispersion,float(entropy[local]),float(values[local,0]-values[local,1]),float(values[local,0]-values[local,-1])])
            targets.append(float(labels[index]==candidate));records.append({"writer":writer,"episode":episode,"index":index,"candidate":candidate})
    return np.asarray(features,np.float32).reshape(-1,len(FEATURE_NAMES)),np.asarray(targets,np.float32),records,decisions


def build(writer_ids,data,stats,exclude_training=False,self_check=False):
    xs=[];ys=[];records=[];decisions=[]
    for writer in writer_ids:
        for size in SUPPORT_SIZES:
            cal,_=_episode_indices(writer,size,data["labels"],data["writers"],data["splits"])
            x,y,r,d=episode_rows(writer,f"support_{size}",cal,data,stats,writer if exclude_training else None,self_check)
            if len(x):xs.append(x);ys.append(y);records.extend(r)
            decisions.extend(d)
    return {"features":np.concatenate(xs),"targets":np.concatenate(ys),"records":records,"decisions":decisions}


def probabilities(model,x,mean,scale):
    with torch.inference_mode():return torch.sigmoid(model(torch.from_numpy(((x-mean)/scale).astype(np.float32)))).numpy()


def enabled_episodes(prob,records,decisions,threshold,labels,logits,mapping):
    by_episode=defaultdict(lambda:{"records":[],"prob":[],"decisions":[]})
    for p,row in zip(prob.tolist(),records,strict=True):by_episode[(row["writer"],row["episode"])]["records"].append(row);by_episode[(row["writer"],row["episode"])]["prob"].append(p)
    for row in decisions:by_episode[(row["writer"],row["episode"])]["decisions"].append(row)
    enabled=set();states={}
    for key,value in by_episode.items():
        result,_=_evaluate_detailed(np.asarray(value["prob"],np.float32),value["records"],value["decisions"],threshold,{key},labels,logits,mapping)
        states[f"{key[0]}:{key[1]}"]={"improved":result["improved"],"regressed":result["regressed"]}
        if result["improved"]>0 and result["regressed"]==0:enabled.add(key)
    return enabled,states


def combined_safe(results):
    return all(not r["writer_top1_regressions_global"] and not r["writer_formula_exact_regressions_global"] and r["candidate_set_violations"]==0 for r in results)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,default=DEFAULT_OUTPUT);p.add_argument("--epochs",type=int,default=60);args=p.parse_args()
    if args.output.exists():p.error(f"refusing to overwrite output: {args.output}")
    old=load_cache(OLD_CACHE);consumed=load_cache(CONSUMED_CACHE);mapping_old={i:f"old_synthetic_writer_{i:03d}" for i in set(old["writers"].tolist())};global_map=_global_map(EXTENSION_BANK)
    ordered=sorted(set(old["writers"].tolist()),key=lambda w:hashlib.sha256(f"{SEED}:writer-split:{w}".encode()).hexdigest());fit=ordered[:20];dev=ordered[20:24]
    consumed_local=json.loads(R1_RECEIPT.read_text(encoding="utf-8"))["outer_local_indices"]
    if set(consumed["writers"].tolist())!=set(consumed_local):raise ValueError("consumed r1 cache writer mismatch")
    stats=prototype_stats(old,fit);train=build(fit,old,stats,True);dev_old=build(dev,old,stats);dev_old_self=build(dev,old,stats,self_check=True);dev_ext=build(consumed_local,consumed,stats);dev_ext_self=build(consumed_local,consumed,stats,self_check=True)
    mean=train["features"].mean(axis=0).astype(np.float32);scale=np.maximum(train["features"].std(axis=0),1e-4).astype(np.float32)
    torch.manual_seed(SEED+91);model=CandidatePromoter();optimizer=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=.05)
    x=torch.from_numpy(((train["features"]-mean)/scale).astype(np.float32));y=torch.from_numpy(train["targets"]);pos=max(1,int(y.sum()));weight=min(20.0,(len(y)-pos)/pos)
    best=None;best_state=None;best_epoch=None;history=[]
    for epoch in range(1,args.epochs+1):
        model.train();order=torch.randperm(len(x));losses=[]
        for start in range(0,len(order),4096):
            batch=order[start:start+4096];out=model(x[batch]);loss=F.binary_cross_entropy_with_logits(out,y[batch],weight=torch.where(y[batch]>.5,weight,1.0));optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.0);optimizer.step();losses.append(float(loss))
        dp_old=probabilities(model,dev_old["features"],mean,scale);sp_old=probabilities(model,dev_old_self["features"],mean,scale);dp_ext=probabilities(model,dev_ext["features"],mean,scale);sp_ext=probabilities(model,dev_ext_self["features"],mean,scale)
        choices=[]
        for threshold in THRESHOLDS:
            enabled_old,_=enabled_episodes(sp_old,dev_old_self["records"],dev_old_self["decisions"],threshold,old["labels"],old["logits"],mapping_old)
            enabled_ext,_=enabled_episodes(sp_ext,dev_ext_self["records"],dev_ext_self["decisions"],threshold,consumed["labels"],consumed["logits"],global_map)
            old_result,_=_evaluate_detailed(dp_old,dev_old["records"],dev_old["decisions"],threshold,enabled_old,old["labels"],old["logits"],mapping_old)
            ext_result,_=_evaluate_detailed(dp_ext,dev_ext["records"],dev_ext["decisions"],threshold,enabled_ext,consumed["labels"],consumed["logits"],global_map)
            improved=old_result["improved"]+ext_result["improved"];regressed=old_result["regressed"]+ext_result["regressed"]
            safe=combined_safe([old_result,ext_result]);formula_nonreg=old_result["adapted_formula_exact"]>=old_result["baseline_formula_exact"] and ext_result["adapted_formula_exact"]>=ext_result["baseline_formula_exact"]
            choices.append((int(safe and formula_nonreg and improved>regressed),improved-regressed,threshold,old_result,ext_result,len(enabled_old),len(enabled_ext)))
        selected=max(choices,key=lambda z:z[:3]);history.append({"epoch":epoch,"loss":float(np.mean(losses)),"threshold":selected[2],"safe_positive":bool(selected[0]),"net":selected[1],"old_dev":selected[3],"consumed_r1_dev":selected[4],"enabled_old":selected[5],"enabled_consumed":selected[6]})
        if best is None or selected[:2]>best[:2]:best=selected;best_epoch=epoch;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    if not best or best[0]!=1:
        args.output.mkdir(parents=True);(args.output/"pre_reserve_rejection.json").write_text(json.dumps({"status":"PRE_RESERVE_FAIL_CLOSED","history":history},ensure_ascii=False,indent=2),encoding="utf-8");print(json.dumps({"status":"PRE_RESERVE_FAIL_CLOSED","best_net":best[1] if best else None}));return 2
    args.output.mkdir(parents=True);model_path=args.output/"writer_residual_promoter.pt";scaler_path=args.output/"feature_scaler.npz";proto_path=args.output/"global_class_prototypes.npz";config_path=args.output/"frozen_config.json"
    torch.save({"state_dict":best_state,"features":FEATURE_NAMES},model_path);np.savez(scaler_path,mean=mean,scale=scale);np.savez_compressed(proto_path,prototypes=stats["prototypes"],reliability=stats["reliability"],counts=stats["counts"])
    config={"epoch":best_epoch,"threshold":best[2],"fit_old_writers":fit,"development_old_writers":dev,"development_consumed_extension_writers":[global_map[i] for i in consumed_local],"reserve_opened":False,"homograph_boundary":"shape writer adapter only; semantic ownership remains with context layer"};config_path.write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding="utf-8")
    report={"schema":"aiflow-writer-residual-promoter/v9","generated_at":datetime.now(timezone.utc).isoformat(),"status":"WRITER_RESIDUAL_DEVELOPMENT_PASS_RESERVE_CLOSED","best":{"epoch":best_epoch,"threshold":best[2],"net":best[1],"old_dev":best[3],"consumed_r1_dev":best[4]},"config":config,
            "gates":{"old_dev_all_writer_top1_formula_nonregression":combined_safe([best[3]]),"consumed_r1_all_writer_top1_formula_nonregression":combined_safe([best[4]]),"aggregate_positive":best[1]>0,"candidate_set_exact":best[3]["candidate_set_violations"]==0 and best[4]["candidate_set_violations"]==0,"reserve_opened":False},
            "inputs":{"old_cache_sha256":sha256(OLD_CACHE),"consumed_r1_cache_sha256":sha256(CONSUMED_CACHE),"r1_receipt_sha256":sha256(R1_RECEIPT),"script_sha256":sha256(Path(__file__))},"outputs":{"model_sha256":sha256(model_path),"scaler_sha256":sha256(scaler_path),"prototypes_sha256":sha256(proto_path),"config_sha256":sha256(config_path)},"history":history}
    report_path=args.output/"development_report.json";report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8");(args.output/"freeze_receipt.json").write_text(json.dumps({"status":"RESIDUAL_MODEL_FROZEN_RESERVE_UNOPENED","report_sha256":sha256(report_path),**report["inputs"],**report["outputs"]},indent=2),encoding="utf-8")
    print(json.dumps({"output":str(args.output),"status":report["status"],"best":report["best"]},ensure_ascii=False));return 0


if __name__=="__main__":raise SystemExit(main())
