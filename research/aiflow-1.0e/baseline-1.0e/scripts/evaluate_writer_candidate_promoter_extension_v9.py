"""Prepare then one-shot evaluate the frozen v9 promoter on new global writers 032..063."""

from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from build_cleanroom_writer_style_bank_v5 import sha256
from hierarchical_writer_model_v7 import DEFAULT_CHECKPOINT
from evaluate_writer_adaptation_v8 import _frozen_outputs
from train_writer_candidate_promoter_v9 import CandidatePromoter, FEATURE_NAMES, SEED, _episode_states, build, evaluate, probabilities


ROOT=Path(__file__).resolve().parents[1]
DEFAULT_DEVELOPMENT=ROOT/"artifacts/writer_candidate_promoter_v9_20260823_r1_frozen"
DEFAULT_EXTENSION=ROOT/"artifacts/cleanroom_writer_style_v5_20260823_extension32_r4_r1"
DEFAULT_CACHE=ROOT/"artifacts/writer_adaptation_v9_extension32_frozen_cache_20260823_r1.npz"
DEFAULT_OUTPUT=ROOT/"artifacts/writer_candidate_promoter_v9_extension_outer_20260823_r1"


def _global_map(bank:Path):
    latents=json.loads((bank/"writer_latents.json").read_text(encoding="utf-8"))
    mapping={local:str(row["synthetic_writer_id"]) for local,row in enumerate(latents)}
    if sorted(mapping.values())!=[f"synthetic_writer_{i:03d}" for i in range(32,64)]: raise ValueError("extension global writer IDs changed")
    return mapping


def _expected_receipt(development,extension,mapping,independent_audit,use_sealed_reserve):
    ordered=sorted(mapping,key=lambda local:hashlib.sha256(f"{SEED}:extension-outer:{mapping[local]}".encode()).hexdigest())
    outer=ordered[16:] if use_sealed_reserve else ordered[:16]; reserve=[] if use_sealed_reserve else ordered[16:]
    return {"schema":"aiflow-writer-candidate-promoter/v9-outer-open-receipt","status":"PREPARED_OUTER_UNOPENED",
            "development_receipt_sha256":sha256(development/"freeze_receipt.json"),
            "development_model_sha256":sha256(development/"candidate_promoter.pt"),
            "development_scaler_sha256":sha256(development/"feature_scaler.npz"),
            "development_config_sha256":sha256(development/"frozen_config.json"),
            "development_report_sha256":sha256(development/"development_report.json"),
            "development_split_manifest_sha256":sha256(development/"development_split_manifest.json"),
            "development_feature_schema_sha256":sha256(development/"feature_schema.json"),
            "extension_bank_sha256":sha256(extension/"synthetic_writer_style_bank_v5_r4.npz"),
            "extension_metadata_sha256":sha256(extension/"synthetic_writer_style_bank_v5_r4.metadata.jsonl.gz"),
            "extension_report_sha256":sha256(extension/"report.json"),
            "extension_writer_latents_sha256":sha256(extension/"writer_latents.json"),
            "outer_opener_script_sha256":sha256(Path(__file__)),
            "independent_extension_audit_sha256":sha256(independent_audit),
            "outer_local_indices":outer,"outer_global_writer_ids":[mapping[i] for i in outer],
            "sealed_reserve_local_indices":reserve,"sealed_reserve_global_writer_ids":[mapping[i] for i in reserve],
            "previously_consumed_extension_outer_global_writer_ids":[mapping[i] for i in ordered[:16]] if use_sealed_reserve else [],
            "selection":"sha256(seed,extension-outer,global synthetic_writer_id); adapter predictions/scoring/tuning unopened; bank-construction labels and HWR admission known"}


def _outer_cache(extension:Path,checkpoint:Path,cache:Path,outer:list[int],reserve:list[int]):
    bank=extension/"synthetic_writer_style_bank_v5_r4.npz"; bank_hash=sha256(bank); checkpoint_hash=sha256(checkpoint)
    if cache.exists():raise ValueError("one-shot outer subset cache already exists")
    with np.load(bank,allow_pickle=False) as payload:
        all_writers=np.asarray(payload["writer_index"],dtype=np.int16); mask=np.isin(all_writers,np.asarray(outer,dtype=np.int16))
        features=np.asarray(payload["features"],dtype=np.float32)[mask]; labels=np.asarray(payload["labels"],dtype=np.int64)[mask]
        writers=all_writers[mask]; splits=np.asarray(payload["episode_split"],dtype=np.int8)[mask]
    _tokens,embeddings,logits=_frozen_outputs(features,checkpoint)
    cache.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(cache,labels=labels,writers=writers,splits=splits,embeddings=embeddings.astype(np.float32),logits=logits.astype(np.float32),bank_sha256=np.asarray(bank_hash),checkpoint_sha256=np.asarray(checkpoint_hash))
    arrays={"labels":labels,"writers":writers,"splits":splits,"embeddings":embeddings,"logits":logits}
    writer_set=set(arrays["writers"].tolist())
    if writer_set!=set(outer):raise ValueError("outer subset cache writer set mismatch")
    if writer_set&set(reserve):raise ValueError("sealed reserve leaked into outer subset cache")
    return arrays,{"cache_writer_set_exact":True,"reserve_rows":0,"reserve_labels":0,"reserve_logits":0,"rows":len(arrays["labels"])}


def _parent_overlap(old_bank:Path,extension:Path,outer_globals:set[str]):
    def read(path):
        with gzip.open(path,"rt",encoding="utf-8") as stream:return [json.loads(line) for line in stream if line.strip()]
    old=read(old_bank/"synthetic_writer_style_bank_v5_r4.metadata.jsonl.gz")
    new=[row for row in read(extension/"synthetic_writer_style_bank_v5_r4.metadata.jsonl.gz") if row["synthetic_writer_id"] in outer_globals]
    old_immediate={r["parent_synthetic_id"] for r in old}; new_immediate={r["parent_synthetic_id"] for r in new}
    old_roots={v for r in old for v in r.get("parent_fingerprints",[])}; new_roots={v for r in new for v in r.get("parent_fingerprints",[])}
    return {"outer_immediate":len(new_immediate),"immediate_overlap":len(old_immediate&new_immediate),"immediate_overlap_ratio":len(old_immediate&new_immediate)/max(len(new_immediate),1),
            "outer_roots":len(new_roots),"root_overlap":len(old_roots&new_roots),"root_overlap_ratio":len(old_roots&new_roots)/max(len(new_roots),1),
            "evidence_boundary":"new style latents and physics RNG on known external parent manifold; not promotion evidence"}


def _class_sets(old_bank:Path,extension:Path):
    with np.load(old_bank/"synthetic_writer_style_bank_v5_r4.npz",allow_pickle=False) as payload:old=set(np.asarray(payload["labels"],dtype=np.int64).tolist())
    with np.load(extension/"synthetic_writer_style_bank_v5_r4.npz",allow_pickle=False) as payload:new=set(np.asarray(payload["labels"],dtype=np.int64).tolist())
    digest=lambda values:hashlib.sha256(json.dumps(sorted(values),separators=(",",":")).encode()).hexdigest()
    return old,new,{"old_hash":digest(old),"new_hash":digest(new),"common_hash":digest(old&new),"old":len(old),"new":len(new),"common":len(old&new),"old_only":len(old-new),"new_only":len(new-old)}


def _evaluate_detailed(probability,records,decisions,threshold,enabled,labels,logits,mapping,allowed=None):
    options=defaultdict(list)
    for p,row in zip(probability.tolist(),records,strict=True):options[(row["writer"],row["episode"],row["index"])].append((p,row["candidate"]))
    states=defaultdict(lambda:{"rows":0,"baseline":0,"adapted":0,"changed":0,"improved":0,"regressed":0,"formulae":0,"baseline_formula_exact":0,"adapted_formula_exact":0})
    formula_rows=defaultdict(list);trace=[];positions=defaultdict(int);violations=0
    for row in decisions:
        if allowed is not None and int(labels[row["index"]]) not in allowed:continue
        writer=int(row["writer"]);episode=str(row["episode"]);position=positions[(writer,episode)];positions[(writer,episode)]+=1
        top5=np.argsort(logits[row["index"]])[-5:][::-1].astype(int).tolist();prediction=int(row["baseline"])
        choices=options.get((writer,episode,row["index"]),[])
        if (writer,episode) in enabled and choices:
            best=max(choices)
            if best[0]>=threshold:prediction=int(best[1])
        truth=int(labels[row["index"]]);old=int(row["baseline"])
        old_correct=old==truth;new_correct=prediction==truth;state=states[writer]
        state["rows"]+=1;state["baseline"]+=int(old_correct);state["adapted"]+=int(new_correct);state["changed"]+=int(prediction!=old)
        state["improved"]+=int(new_correct and not old_correct);state["regressed"]+=int(old_correct and not new_correct)
        formula_rows[(writer,episode,position//8)].append((old_correct,new_correct))
        violation=prediction not in top5;violations+=int(violation)
        if prediction!=old:trace.append({"global_writer_id":mapping[writer],"episode":episode,"query_index":int(row["index"]),"truth":truth,"baseline":old,"adapted":prediction,"baseline_top5":top5,"adapted_in_baseline_top5":not violation})
    for (writer,_episode,_chunk),rows in formula_rows.items():
        state=states[writer];state["formulae"]+=1;state["baseline_formula_exact"]+=int(all(v[0] for v in rows));state["adapted_formula_exact"]+=int(all(v[1] for v in rows))
    total=sum(s["rows"] for s in states.values());formulae=sum(s["formulae"] for s in states.values())
    result={"writers":len(states),"rows":total,"pseudo_formula_contract":"writer x sparse-calibration episode; deterministic query order chunked by 8; final short chunk retained",
            "baseline_top1":sum(s["baseline"] for s in states.values())/total,"adapted_top1":sum(s["adapted"] for s in states.values())/total,
            "formulae":formulae,"baseline_formula_exact":sum(s["baseline_formula_exact"] for s in states.values())/formulae,"adapted_formula_exact":sum(s["adapted_formula_exact"] for s in states.values())/formulae,
            "improved":sum(s["improved"] for s in states.values()),"regressed":sum(s["regressed"] for s in states.values()),"changed":sum(s["changed"] for s in states.values()),"candidate_set_violations":violations,
            "writer_top1_regressions_global":[mapping[w] for w,s in states.items() if s["adapted"]<s["baseline"]],
            "writer_formula_exact_regressions_global":[mapping[w] for w,s in states.items() if s["adapted_formula_exact"]<s["baseline_formula_exact"]],
            "per_global_writer":{mapping[w]:s for w,s in sorted(states.items())}}
    return result,trace


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--development",type=Path,default=DEFAULT_DEVELOPMENT); p.add_argument("--extension",type=Path,default=DEFAULT_EXTENSION)
    p.add_argument("--cache",type=Path,default=DEFAULT_CACHE); p.add_argument("--checkpoint",type=Path,default=DEFAULT_CHECKPOINT); p.add_argument("--output",type=Path,default=DEFAULT_OUTPUT)
    p.add_argument("--independent-audit",type=Path,required=True)
    p.add_argument("--use-sealed-reserve-as-outer",action="store_true")
    p.add_argument("--prepare",action="store_true"); p.add_argument("--open",action="store_true"); args=p.parse_args()
    if args.prepare==args.open:p.error("choose exactly one of --prepare or --open")
    extension_report=json.loads((args.extension/"report.json").read_text(encoding="utf-8"))
    if extension_report.get("status")!="STYLE_BANK_SHADOW_CANDIDATE" or not all(extension_report.get("gates",{}).values()):raise ValueError("extension style bank gates not fully admitted")
    frozen_receipt=json.loads((args.development/"freeze_receipt.json").read_text(encoding="utf-8"))
    for name,path in {"model_sha256":args.development/"candidate_promoter.pt","scaler_sha256":args.development/"feature_scaler.npz","config_sha256":args.development/"frozen_config.json","development_report_sha256":args.development/"development_report.json","split_manifest_sha256":args.development/"development_split_manifest.json","feature_schema_sha256":args.development/"feature_schema.json","checkpoint_sha256":args.checkpoint,"cache_sha256":ROOT/"artifacts/writer_adaptation_v9_frozen_cache_20260823_r1.npz"}.items():
        if frozen_receipt.get(name)!=sha256(path):raise ValueError(f"frozen development receipt mismatch: {name}")
    if frozen_receipt.get("script_sha256")!=sha256(ROOT/"scripts/train_writer_candidate_promoter_v9.py"):raise ValueError("live development training script drift")
    feature_schema=json.loads((args.development/"feature_schema.json").read_text(encoding="utf-8"))
    model_payload=torch.load(args.development/"candidate_promoter.pt",map_location="cpu",weights_only=False)
    if tuple(feature_schema["feature_names"])!=tuple(model_payload["features"]) or tuple(model_payload["features"])!=tuple(FEATURE_NAMES):raise ValueError("feature schema/model/live feature drift")
    if not args.independent_audit.exists():raise ValueError("independent extension audit missing")
    independent_audit=json.loads(args.independent_audit.read_text(encoding="utf-8"))
    if independent_audit.get("status")!="INDEPENDENT_EXTENSION_ADMISSION_PASSED" or not all(independent_audit.get("gates",{}).values()):raise ValueError("independent extension admission did not pass")
    if independent_audit.get("extension_report_sha256")!=sha256(args.extension/"report.json") or independent_audit.get("extension_bank_sha256")!=sha256(args.extension/"synthetic_writer_style_bank_v5_r4.npz"):raise ValueError("independent admission receipt does not match live extension")
    if args.use_sealed_reserve_as_outer:
        prior=ROOT/"artifacts/writer_candidate_promoter_v9_extension_outer_20260823_r1/outer_report.json"
        if not prior.exists() or json.loads(prior.read_text(encoding="utf-8")).get("status")!="SYNTHETIC_EXTENSION_OUTER_REJECTED":raise ValueError("prior extension outer rejection receipt missing")
    mapping=_global_map(args.extension); expected=_expected_receipt(args.development,args.extension,mapping,args.independent_audit,args.use_sealed_reserve_as_outer); expected["checkpoint_sha256"]=sha256(args.checkpoint); receipt_path=args.output/"outer_open_receipt.json"
    if args.prepare:
        if args.output.exists():p.error(f"refusing to overwrite output: {args.output}")
        if args.cache.exists():raise ValueError("outer subset cache must not exist before prepare")
        args.output.mkdir(parents=True); receipt_path.write_text(json.dumps(expected,ensure_ascii=False,indent=2),encoding="utf-8")
        print(json.dumps({"output":str(args.output),"status":expected["status"],"outer":expected["outer_global_writer_ids"],"reserve_count":len(expected["sealed_reserve_global_writer_ids"])},ensure_ascii=False));return 0
    if not receipt_path.exists():raise ValueError("prepare receipt missing")
    if args.cache.exists() or (args.output/"outer_report.json").exists():raise ValueError("one-shot outer already opened or interrupted; refusing rerun")
    actual=json.loads(receipt_path.read_text(encoding="utf-8"))
    if actual!=expected:raise ValueError("outer receipt or frozen input hash changed")
    arrays,cache_gate=_outer_cache(args.extension,args.checkpoint,args.cache,expected["outer_local_indices"],expected["sealed_reserve_local_indices"])
    payload=model_payload; model=CandidatePromoter();model.load_state_dict(payload["state_dict"],strict=True);model.eval()
    with np.load(args.development/"feature_scaler.npz",allow_pickle=False) as scaler:mean=np.asarray(scaler["mean"],np.float32);scale=np.asarray(scaler["scale"],np.float32)
    config=json.loads((args.development/"frozen_config.json").read_text(encoding="utf-8")); threshold=float(config["threshold"])
    outer=expected["outer_local_indices"]; examples=build(outer,arrays); self_examples=build(outer,arrays,True)
    prob=probabilities(model,examples["features"],mean,scale); self_prob=probabilities(model,self_examples["features"],mean,scale)
    states=_episode_states(self_prob,self_examples["records"],self_examples["decisions"],threshold);enabled={key for key,value in states.items() if value["improved"]>0 and value["regressed"]==0}
    result,trace=_evaluate_detailed(prob,examples["records"],examples["decisions"],threshold,enabled,arrays["labels"],arrays["logits"],mapping)
    old_classes,new_classes,class_audit=_class_sets(ROOT/"artifacts/cleanroom_writer_style_v5_20260823_smoke32_r4_r2",args.extension)
    common_result,_common_trace=_evaluate_detailed(prob,examples["records"],examples["decisions"],threshold,enabled,arrays["labels"],arrays["logits"],mapping,old_classes&new_classes)
    extension_result,_extension_trace=_evaluate_detailed(prob,examples["records"],examples["decisions"],threshold,enabled,arrays["labels"],arrays["logits"],mapping,new_classes-old_classes) if new_classes-old_classes else (None,[])
    common_pass=not common_result["writer_top1_regressions_global"] and not common_result["writer_formula_exact_regressions_global"] and common_result["adapted_top1"]>=common_result["baseline_top1"] and common_result["adapted_formula_exact"]>=common_result["baseline_formula_exact"] and common_result["candidate_set_violations"]==0
    passed=not result["writer_top1_regressions_global"] and not result["writer_formula_exact_regressions_global"] and result["adapted_top1"]>result["baseline_top1"] and result["adapted_formula_exact"]>=result["baseline_formula_exact"] and result["candidate_set_violations"]==0 and common_pass
    trace_path=args.output/"changed_prediction_trace.jsonl"
    with trace_path.open("w",encoding="utf-8") as stream:
        for row in trace:stream.write(json.dumps(row,ensure_ascii=False)+"\n")
    report={"schema":"aiflow-writer-candidate-promoter/v9-extension-outer","generated_at":datetime.now(timezone.utc).isoformat(),"status":"SYNTHETIC_EXTENSION_OUTER_PASS" if passed else "SYNTHETIC_EXTENSION_OUTER_REJECTED",
            "outer":result,"common_class_sensitivity":common_result,"extension_only_diagnostic":extension_result,"class_set_audit":class_audit,"self_gate":{"episodes":len(states),"enabled":len(enabled)},"outer_subset_cache_gate":cache_gate,"previously_consumed_extension_outer":expected["previously_consumed_extension_outer_global_writer_ids"],"sealed_reserve":{"count":len(expected["sealed_reserve_global_writer_ids"]),"global_writer_ids":expected["sealed_reserve_global_writer_ids"],"opened":False,"rows_in_cache":0,"labels_in_cache":0,"logits_in_cache":0},
            "parent_overlap_audit":_parent_overlap(ROOT/"artifacts/cleanroom_writer_style_v5_20260823_smoke32_r4_r2",args.extension,set(expected["outer_global_writer_ids"])),
            "gates":{"outer_all_global_writer_top1_nonregression":not result["writer_top1_regressions_global"],"outer_all_global_writer_formula_exact_nonregression":not result["writer_formula_exact_regressions_global"],"outer_aggregate_top1_positive":result["adapted_top1"]>result["baseline_top1"],"outer_aggregate_formula_exact_nonregression":result["adapted_formula_exact"]>=result["baseline_formula_exact"],"common_class_all_writer_and_aggregate_nonregression":common_pass,"candidate_set_exact_recomputed":result["candidate_set_violations"]==0,
                     "legacy_opened":False,"known_replay_opened":False,"crohme_opened":False,"mathwriting_opened":False,"product_promotion":False},
            "hashes":{**{k:v for k,v in expected.items() if k.endswith("sha256")},"outer_open_receipt_sha256":sha256(receipt_path),"extension_cache_sha256":sha256(args.cache),"changed_prediction_trace_sha256":sha256(trace_path)}}
    report_path=args.output/"outer_report.json";report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"output":str(args.output),"status":report["status"],"outer":result,"reserve_opened":False},ensure_ascii=False));return 0 if passed else 2


if __name__=="__main__":raise SystemExit(main())
