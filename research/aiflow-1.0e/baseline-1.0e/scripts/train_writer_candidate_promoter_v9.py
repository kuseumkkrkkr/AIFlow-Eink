"""Freeze the v9 binary candidate promoter on the old clean-room 20/4 split only."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from build_cleanroom_writer_style_bank_v5 import sha256
from train_cleanroom_writer_adapter_v9 import SUPPORT_SIZES, _episode_indices
from train_cleanroom_writer_meta_ranker_v9 import DEFAULT_BANK, DEFAULT_CACHE, _cache_outputs
from hierarchical_writer_model_v7 import DEFAULT_CHECKPOINT


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts/writer_candidate_promoter_v9_20260823_r1_frozen"
SEED = 20260823
FEATURE_NAMES = ("baseline_margin", "normalized_margin", "candidate_rank", "candidate_centered_logit",
                 "class_support", "prototype_cosine", "prototype_consistency", "calibration_baseline_accuracy",
                 "baseline_candidate_supported", "baseline_prototype_cosine", "prototype_cosine_advantage",
                 "top5_entropy", "top1_top2_margin", "top5_spread")
THRESHOLDS = tuple(float(value) for value in np.linspace(0.50, 0.99, 50))


class CandidatePromoter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(len(FEATURE_NAMES), 16), nn.GELU(), nn.Linear(16, 8), nn.GELU(), nn.Linear(8, 1))
    def forward(self, values): return self.network(values).squeeze(-1)


def _top5(logits): return np.argsort(logits, axis=1)[:, -5:][:, ::-1]
def _normalize(values): return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1.0e-8)


def _pairs(writer, episode, calibration, data, self_check=False):
    labels, embeddings, logits, writers, splits = (data[k] for k in ("labels", "embeddings", "logits", "writers", "splits"))
    query = calibration if self_check else np.where((writers == writer) & (splits == 1))[0]
    truth_cal = labels[calibration]; counts = Counter(truth_cal.tolist()); supported = sorted(k for k,v in counts.items() if v >= 2)
    cal_norm, query_norm = _normalize(embeddings[calibration]), _normalize(embeddings[query])
    cal_rank = _top5(logits[calibration]); query_rank = _top5(logits[query]); query_values = np.take_along_axis(logits[query], query_rank, axis=1)
    shifted=query_values-query_values.max(axis=1,keepdims=True); prob=np.exp(shifted); prob/=prob.sum(axis=1,keepdims=True)
    entropy=-np.sum(prob*np.log(np.maximum(prob,1e-12)),axis=1); spread=np.maximum(query_values.std(axis=1),1e-6)
    class_rows={label:np.where(truth_cal==label)[0] for label in supported}
    full={}; consistency={}; accuracy={}
    for label, rows in class_rows.items():
        proto=cal_norm[rows].mean(axis=0); proto/=max(float(np.linalg.norm(proto)),1e-8); full[label]=proto
        pair=cal_norm[rows]@cal_norm[rows].T; consistency[label]=float((pair.sum()-len(rows))/max(1,len(rows)*(len(rows)-1)))
        accuracy[label]=float(np.mean(cal_rank[rows,0]==label))
    def prototype(label, query_local):
        if label not in full: return None
        if not self_check: return full[label]
        source_local=query_local
        rows=class_rows[label]
        if truth_cal[source_local]==label: rows=rows[rows!=source_local]
        if not len(rows): return None
        value=cal_norm[rows].mean(axis=0); return value/max(float(np.linalg.norm(value)),1e-8)
    features=[]; targets=[]; records=[]; decisions=[]
    for local,index in enumerate(query.tolist()):
        baseline=int(query_rank[local,0]); decisions.append({"writer":writer,"episode":episode,"index":index,"baseline":baseline,"truth":int(labels[index])})
        base_proto=prototype(baseline,local); base_cos=float(query_norm[local]@base_proto) if base_proto is not None else 0.0
        for rank in range(1,5):
            candidate=int(query_rank[local,rank]); candidate_proto=prototype(candidate,local)
            if candidate_proto is None: continue
            cosine=float(query_norm[local]@candidate_proto); margin=float(query_values[local,0]-query_values[local,rank])
            features.append([margin,margin/spread[local],rank/4.0,float((query_values[local,rank]-query_values[local].mean())/spread[local]),
                             counts[candidate]/2.0,cosine,consistency[candidate],accuracy[candidate],float(baseline in full),base_cos,
                             cosine-base_cos,float(entropy[local]),float(query_values[local,0]-query_values[local,1]),float(query_values[local,0]-query_values[local,-1])])
            targets.append(float(labels[index]==candidate)); records.append({"writer":writer,"episode":episode,"index":index,"candidate":candidate})
    return np.asarray(features,dtype=np.float32).reshape(-1,len(FEATURE_NAMES)),np.asarray(targets,dtype=np.float32),records,decisions


def build(writer_ids,data,self_check=False):
    xs=[]; ys=[]; records=[]; decisions=[]
    for writer in writer_ids:
        for size in SUPPORT_SIZES:
            calibration,_query=_episode_indices(writer,size,data["labels"],data["writers"],data["splits"])
            x,y,r,d=_pairs(writer,f"support_{size}",calibration,data,self_check)
            if len(x): xs.append(x); ys.append(y); records.extend(r)
            decisions.extend(d)
    return {"features":np.concatenate(xs) if xs else np.empty((0,len(FEATURE_NAMES)),np.float32),
            "targets":np.concatenate(ys) if ys else np.empty(0,np.float32),"records":records,"decisions":decisions}


def probabilities(model,x,mean,scale):
    with torch.inference_mode(): return torch.sigmoid(model(torch.from_numpy(((x-mean)/scale).astype(np.float32)))).numpy()


def evaluate(probability,records,decisions,threshold,enabled=None):
    options=defaultdict(list)
    for p,row in zip(probability.tolist(),records,strict=True): options[(row["writer"],row["episode"],row["index"])].append((p,row["candidate"]))
    states=defaultdict(lambda:{"rows":0,"baseline":0,"adapted":0,"changed":0,"improved":0,"regressed":0})
    for row in decisions:
        key=(row["writer"],row["episode"],row["index"]); prediction=row["baseline"]
        if enabled is None or (row["writer"],row["episode"]) in enabled:
            choices=options.get(key,[])
            if choices:
                best=max(choices)
                if best[0]>=threshold: prediction=best[1]
        old=row["baseline"]==row["truth"]; new=prediction==row["truth"]; state=states[row["writer"]]
        state["rows"]+=1; state["baseline"]+=int(old); state["adapted"]+=int(new); state["changed"]+=int(prediction!=row["baseline"])
        state["improved"]+=int(new and not old); state["regressed"]+=int(old and not new)
    total=sum(s["rows"] for s in states.values()); old=sum(s["baseline"] for s in states.values()); new=sum(s["adapted"] for s in states.values())
    return {"writers":len(states),"rows":total,"baseline_top1":old/total,"adapted_top1":new/total,
            "improved":sum(s["improved"] for s in states.values()),"regressed":sum(s["regressed"] for s in states.values()),
            "changed":sum(s["changed"] for s in states.values()),"writer_regressions":[w for w,s in sorted(states.items()) if s["adapted"]<s["baseline"]],
            "per_writer":dict(sorted(states.items())),"candidate_set_violations":0}


def self_gate(probability,records,decisions,threshold):
    raw=evaluate(probability,records,decisions,threshold)
    enabled={(writer,episode) for writer in {d["writer"] for d in decisions} for episode in {d["episode"] for d in decisions}
             if (lambda rows: sum(r["improved"] for r in rows)>0 and sum(r["regressed"] for r in rows)==0)(
                 [s for (w,e),s in _episode_states(probability,records,decisions,threshold).items() if w==writer and e==episode])}
    return enabled,{"raw":raw,"enabled_episodes":len(enabled)}


def _episode_states(probability,records,decisions,threshold):
    options=defaultdict(list)
    for p,row in zip(probability.tolist(),records,strict=True): options[(row["writer"],row["episode"],row["index"])].append((p,row["candidate"]))
    states=defaultdict(lambda:{"improved":0,"regressed":0})
    for row in decisions:
        prediction=row["baseline"]; choices=options.get((row["writer"],row["episode"],row["index"]),[])
        if choices and max(choices)[0]>=threshold: prediction=max(choices)[1]
        old=row["baseline"]==row["truth"]; new=prediction==row["truth"]
        states[(row["writer"],row["episode"])]["improved"]+=int(new and not old); states[(row["writer"],row["episode"])]["regressed"]+=int(old and not new)
    return states


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--bank",type=Path,default=DEFAULT_BANK); parser.add_argument("--cache",type=Path,default=DEFAULT_CACHE); parser.add_argument("--checkpoint",type=Path,default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT); parser.add_argument("--epochs",type=int,default=60); args=parser.parse_args()
    if args.output.exists(): parser.error(f"refusing to overwrite output: {args.output}")
    arrays,_report=_cache_outputs(args.bank,args.checkpoint,args.cache)
    ordered=sorted(set(arrays["writers"].tolist()),key=lambda w:hashlib.sha256(f"{SEED}:writer-split:{w}".encode()).hexdigest()); train_writers=ordered[:20]; dev_writers=ordered[20:24]; consumed_outer=ordered[24:]
    train=build(train_writers,arrays); dev=build(dev_writers,arrays); dev_self=build(dev_writers,arrays,True)
    mean=train["features"].mean(axis=0).astype(np.float32); scale=np.maximum(train["features"].std(axis=0),1e-4).astype(np.float32)
    torch.manual_seed(SEED); model=CandidatePromoter(); optimizer=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=.05)
    x=torch.from_numpy(((train["features"]-mean)/scale).astype(np.float32)); y=torch.from_numpy(train["targets"]); weight=min(20.0,(len(y)-max(1,int(y.sum())))/max(1,int(y.sum())))
    best=None; best_state=None; best_epoch=None; history=[]
    for epoch in range(1,args.epochs+1):
        model.train(); order=torch.randperm(len(x)); losses=[]
        for start in range(0,len(order),4096):
            batch=order[start:start+4096]; output=model(x[batch]); loss=F.binary_cross_entropy_with_logits(output,y[batch],weight=torch.where(y[batch]>.5,weight,1.0))
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),2.0); optimizer.step(); losses.append(float(loss))
        dp=probabilities(model,dev["features"],mean,scale); sp=probabilities(model,dev_self["features"],mean,scale); choices=[]
        for threshold in THRESHOLDS:
            states=_episode_states(sp,dev_self["records"],dev_self["decisions"],threshold); enabled={k for k,v in states.items() if v["improved"]>0 and v["regressed"]==0}
            result=evaluate(dp,dev["records"],dev["decisions"],threshold,enabled); gain=result["adapted_top1"]-result["baseline_top1"]
            choices.append((int(not result["writer_regressions"] and gain>0),gain,result["improved"]-result["regressed"],threshold,result,len(enabled)))
        selected=max(choices,key=lambda v:v[:4]); history.append({"epoch":epoch,"loss":float(np.mean(losses)),"threshold":selected[3],"development":selected[4],"enabled_episodes":selected[5]})
        if best is None or selected[:3]>best[:3]: best=selected; best_epoch=epoch; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    if not best or best[0]!=1: raise RuntimeError("development gate failed")
    args.output.mkdir(parents=True); model_path=args.output/"candidate_promoter.pt"; scaler_path=args.output/"feature_scaler.npz"; config_path=args.output/"frozen_config.json"
    split_path=args.output/"development_split_manifest.json"; feature_path=args.output/"feature_schema.json"
    split_path.write_text(json.dumps({"seed":SEED,"train_writers":train_writers,"development_writers":dev_writers,"consumed_old_outer_writers":consumed_outer,"old_outer_reopened":False},indent=2),encoding="utf-8")
    feature_path.write_text(json.dumps({"feature_names":FEATURE_NAMES,"architecture":"14-16-8-1","label_id_feature":False,"writer_id_feature":False},indent=2),encoding="utf-8")
    torch.save({"state_dict":best_state,"architecture":"14-16-8-1","features":FEATURE_NAMES},model_path); np.savez(scaler_path,mean=mean,scale=scale)
    config={"threshold":best[3],"epoch":best_epoch,"train_writers":train_writers,"development_writers":dev_writers,"support_threshold":2,"self_gate":"calibration LOO imp>0/reg=0"}
    config_path.write_text(json.dumps(config,indent=2),encoding="utf-8")
    report={"schema":"aiflow-writer-candidate-promoter/v9-development-freeze","generated_at":datetime.now(timezone.utc).isoformat(),"status":"DEVELOPMENT_GATE_PASS_OUTER_CLOSED","best":{"threshold":best[3],"development":best[4],"enabled_episodes":best[5]},"config":config,"outer":None,
            "gates":{"development_nonregression_positive":True,"outer_opened":False,"legacy_opened":False,"known_replay_opened":False,"crohme_opened":False,"mathwriting_opened":False},
            "inputs":{"bank_sha256":sha256(args.bank/"synthetic_writer_style_bank_v5_r4.npz"),"bank_report_sha256":sha256(args.bank/"report.json"),"cache_sha256":sha256(args.cache),"checkpoint_sha256":sha256(args.checkpoint),"script_sha256":sha256(Path(__file__))},
            "outputs":{"model_sha256":sha256(model_path),"scaler_sha256":sha256(scaler_path),"config_sha256":sha256(config_path),"split_manifest_sha256":sha256(split_path),"feature_schema_sha256":sha256(feature_path)},"history":history}
    report_path=args.output/"development_report.json"; report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    receipt={"status":"FROZEN_OUTER_UNOPENED","development_report_sha256":sha256(report_path),**report["inputs"],**report["outputs"]}; (args.output/"freeze_receipt.json").write_text(json.dumps(receipt,indent=2),encoding="utf-8")
    print(json.dumps({"output":str(args.output),"status":report["status"],"best":report["best"],"receipt":receipt},ensure_ascii=False)); return 0


if __name__=="__main__": raise SystemExit(main())
