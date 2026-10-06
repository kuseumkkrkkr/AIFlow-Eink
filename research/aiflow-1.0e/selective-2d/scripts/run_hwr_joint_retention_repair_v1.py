"""canonical 정답·이전 회복 정답·새 source 정답을 함께 보호하는 TRAIN 교정.

새 학습 표본/사람 경계 라벨/heldout은 사용하지 않는다. 직전 실패의 보호 누락
가설만 바꾸고 같은 32회/회당 16개 방향 교정 예산을 사용한다.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from repair_hwr_fixed_rank_margins_v1 import (
    ROOT,DEFAULT_CHECKPOINT,DEFAULT_DATA_DIR,PARENT,SOURCE,RETENTION,CANDIDATE,
    OUTPUT as BASE,MAX_ROUNDS,CONSTRAINTS_PER_ROUND,CORRECTION_SLACK,
    _sha,_write,_guard_commit,rank_margins,fixed_constraints,violations,repair,
)

OUTPUT = ROOT / "artifacts/hwr_joint_retention_repair_20261005"


def union_constraints(canonical: dict, previous: dict) -> dict:
    """canonical 보호를 그대로 두고 이전 후보에서만 회복된 정답의 고정 floor를 추가한다."""
    result = {}
    for k in (1,5):
        old = canonical[f"top{k}_mask"].astype(bool)
        new = previous[f"top{k}_mask"].astype(bool)
        if old.shape != new.shape:
            raise ValueError("retention reference arrays differ")
        result[f"top{k}_mask"] = old | new
        result[f"top{k}_floor"] = np.where(old,canonical[f"top{k}_floor"],previous[f"top{k}_floor"]).copy()
    return result


def _run(out: Path) -> int:
    """2048개 봉인 TRAIN 행에서 새 정답을 보호할 때의 교정 가능성과 실제 순위를 비교한다."""
    if out.exists():
        raise FileExistsError("refusing to overwrite joint retention repair")
    if _guard_commit("before_joint_retention_repair") is None:
        return 78
    import torch
    from run_hwr_affine_distillation_experiment_v1 import _load_teacher,_predict_logits
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    base = json.loads((BASE / "frozen_plan.json").read_text(encoding="utf-8"))
    verified = json.loads((BASE / "independent_verification.json").read_text(encoding="utf-8"))
    if _sha(BASE / "repair_result.json") != verified["result_sha256"] or _sha(ROOT / "scripts/repair_hwr_fixed_rank_margins_v1.py") != base["script_sha256"]:
        raise ValueError("sealed prior repair evidence changed")
    for name,digest in base["dependencies"].items():
        if _sha(ROOT / "scripts" / name) != digest:
            raise ValueError("prior repair dependency changed")
    if _sha(DEFAULT_CHECKPOINT) != base["canonical_checkpoint_sha256"] or _sha(CANDIDATE / "directional_guard.pt") != base["candidate_checkpoint_sha256"]:
        raise ValueError("canonical or parent candidate changed")
    old = json.loads((PARENT / "frozen_plan.json").read_text(encoding="utf-8"))
    if _sha(PARENT / "frozen_plan.json") != base["source_plan_sha256"]:
        raise ValueError("source plan changed")
    arrays = {}
    for name in ("replay_features","replay_indices","train_indices"):
        item=old["artifacts"][name];path=PARENT / item["file"]
        if _sha(path)!=item["sha256"]:
            raise ValueError("source TRAIN anchor changed")
        arrays[name]=np.load(path,allow_pickle=False)
    for name,digest in old["replay_train_array_hashes"].items():
        if _sha(DEFAULT_DATA_DIR / f"train_{name}.npy")!=digest:
            raise ValueError("old real TRAIN source changed")
    old_x=arrays["replay_features"]
    old_y=np.array(np.load(DEFAULT_DATA_DIR / "train_labels.npy",mmap_mode="r",allow_pickle=False)[arrays["replay_indices"]],copy=True)
    source_report=json.loads((SOURCE / "pendigits_source_audit.json").read_text(encoding="utf-8"))
    if _sha(SOURCE / "pendigits_source_audit.json")!=old["source_report_sha256"]:
        raise ValueError("digit TRAIN source report changed")
    for name in ("train_features_y_up","train_digit_labels"):
        if _sha(SOURCE / f"{name}.npy")!=source_report["artifacts"][name]["sha256"]:
            raise ValueError("digit TRAIN source array changed")
    if _sha(RETENTION / "source_train_probe_indices.npy")!=base["source_probe_indices_sha256"]:
        raise ValueError("source TRAIN probe changed")
    indices=np.load(RETENTION / "source_train_probe_indices.npy",allow_pickle=False)
    expected=np.random.default_rng(old["seed"]+719).choice(len(arrays["train_indices"]),1024,replace=False)
    if not np.array_equal(indices,arrays["train_indices"][expected]):
        raise ValueError("source TRAIN probe selection differs")
    source_x=np.array(np.load(SOURCE / "train_features_y_up.npy",mmap_mode="r",allow_pickle=False)[indices],copy=True)
    digits=np.load(SOURCE / "train_digit_labels.npy",mmap_mode="r",allow_pickle=False)[indices]
    device=torch.device("cpu");canonical,labels,_=_load_teacher(DEFAULT_CHECKPOINT,device)
    model,candidate_labels,_=_load_teacher(CANDIDATE / "directional_guard.pt",device)
    if labels!=candidate_labels or len(np.unique(old_y))!=371:
        raise ValueError("math vocabulary/coverage differs")
    source_y=np.array([labels.index(str(d)) for d in digits],dtype=np.int64)
    old_reference=_predict_logits(canonical,old_x,device,32)
    prior_old=_predict_logits(model,old_x,device,32)
    prior_source=_predict_logits(model,source_x,device,32)
    source_reference=_predict_logits(canonical,source_x,device,32)
    if not np.array_equal(prior_old,np.load(BASE / "prior_candidate_replay_logits.npy",allow_pickle=False)) or not np.array_equal(old_reference,np.load(BASE / "canonical_replay_logits.npy",allow_pickle=False)):
        raise ValueError("prior reference logits differ")
    old_spec=union_constraints(fixed_constraints(old_reference,old_y),fixed_constraints(prior_old,old_y))
    source_spec=fixed_constraints(prior_source,source_y)
    x=np.concatenate((old_x,source_x));y=np.concatenate((old_y,source_y))
    spec={name:np.concatenate((old_spec[name],source_spec[name])) for name in old_spec}
    baseline_source_hits=int((source_reference.argmax(1)==source_y).sum())
    prior_source_hits=int((prior_source.argmax(1)==source_y).sum())
    if baseline_source_hits!=853 or prior_source_hits!=958:
        raise ValueError("source TRAIN baseline changed")
    out.mkdir(parents=True)
    plan=dict(schema="aiflow-joint-retention-repair-plan/v1",script_sha256=_sha(Path(__file__)),
              shared_repair_sha256=_sha(ROOT / "scripts/repair_hwr_fixed_rank_margins_v1.py"),prior_plan_sha256=_sha(BASE / "frozen_plan.json"),
              prior_result_sha256=_sha(BASE / "repair_result.json"),canonical_checkpoint_sha256=_sha(DEFAULT_CHECKPOINT),
              parent_checkpoint_sha256=_sha(CANDIDATE / "directional_guard.pt"),max_rounds=MAX_ROUNDS,constraints_per_round=CONSTRAINTS_PER_ROUND,
              correction_slack=CORRECTION_SLACK,norm_cap_multiple=2.,fixed_margin_fraction=.5,
              old_reference_rule="canonical floors for canonical-correct rows; parent fixed floors only for newly rescued rows",
              source_reference_rule="preserve all parent-correct source TRAIN Top-1/Top-5 rows at half parent margin",
              protected_counts={domain:{f"top{k}":int(domain_spec[f"top{k}_mask"].sum()) for k in (1,5)} for domain,domain_spec in (("old_math",old_spec),("source_digits",source_spec))},
              old_train_classes=371,source_digit_classes=10,source_gain_retention_gate=.9,
              acceptance="all immutable joint margin/rank constraints AND the unchanged 90% source TRAIN gain gate",
              failure_policy="keep failed candidate; never weaken masks/floors or retry within this frozen run",
              optimizer_created=False,parameter_optimization_performed=True,new_human_boundary_distillation_performed=False,
              held_inputs_forwarded=0,official_test_rows_read=0,crohme_rows=0,human_boundary_labels=0,product_adopted=False,
              limits="TRAIN-only finite-constraint experiment; source probe rows are now optimization anchors, NOT independent validation or product selection")
    _write(out / "frozen_plan.json",plan)
    for name,array in {"joint_features":x,"joint_labels":y,"canonical_old_logits":old_reference,"canonical_source_logits":source_reference,
                       "parent_old_logits":prior_old,"parent_source_logits":prior_source,**{f"reference_{name}":value for name,value in spec.items()}}.items():
        np.save(out / f"{name}.npy",array,allow_pickle=False)
    checks=[]
    def predict(candidate):
        """각 교정 회차에 양쪽 도메인의 실제 위반 수를 별도 반환·기록한다."""
        logits=_predict_logits(candidate,x,device,32)
        _,a=violations(logits[:len(old_x)],old_y,old_spec);_,b=violations(logits[len(old_x):],source_y,source_spec)
        checks.append(dict(check=len(checks),old_math=a,source_digits=b))
        print(json.dumps(dict(event="joint_retention_domains",**checks[-1])),flush=True)
        return logits
    state={name:value.detach().clone() for name,value in canonical.state_dict().items()}
    status=repair(model,state,x,y,spec,predict)
    logits=_predict_logits(model,x,device,32);failures,counts=violations(logits,y,spec)
    old_rank=rank_margins(logits[:len(old_x)],old_y);source_rank=rank_margins(logits[len(old_x):],source_y)
    hits=int(source_rank["top1_hit"].sum());retention=(hits-baseline_source_hits)/(prior_source_hits-baseline_source_hits)
    success=bool(status["success"] and not failures and retention>=.9)
    checkpoint=out / ("joint_repaired_research.pt" if success else "failed_joint_repair_research.pt")
    torch.save(dict(state_dict=model.state_dict(),math_labels=labels,auxiliary_labels=[],report=dict(input_contract=dict(observed_channel_mode="uniform-time"),product_adopted=False)),checkpoint)
    np.save(out / "joint_repaired_logits.npy",logits,allow_pickle=False)
    if _sha(DEFAULT_CHECKPOINT)!=plan["canonical_checkpoint_sha256"] or _sha(CANDIDATE / "directional_guard.pt")!=plan["parent_checkpoint_sha256"]:
        raise AssertionError("canonical/parent checkpoint changed")
    metrics=dict(**counts,old_train_top1_hits=int(old_rank["top1_hit"].sum()),old_train_top5_hits=int(old_rank["top5_hit"].sum()),
                 source_train_top1_hits=hits,source_train_top5_hits=int(source_rank["top5_hit"].sum()),source_training_gain_retained_fraction=retention)
    result=dict(schema="aiflow-joint-retention-repair-result/v1",status="joint_train_repair_gate_pass" if success else "joint_train_repair_gate_fail",
                frozen_plan_sha256=_sha(out / "frozen_plan.json"),repair=status,domain_checks=checks,metrics=metrics,
                checkpoint_file=checkpoint.name,checkpoint_sha256=_sha(checkpoint),canonical_checkpoint_unchanged=True,parent_checkpoint_unchanged=True,
                source_probe_now_used_for_optimization=True,eligible_for_independent_performance_claim=False,eligible_for_product_selection=False,
                parameter_optimization_performed=True,new_human_boundary_distillation_performed=False,human_boundary_labels=0,held_inputs_forwarded=0,crohme_rows=0,product_adopted=False)
    _write(out / "joint_repair_result.json",result)
    print(json.dumps(dict(event="joint_retention_complete",status=result["status"],metrics=metrics)),flush=True)
    return 0


def _self_test() -> int:
    """추가 정답 마스크가 빠지지 않고 canonical floor를 parent로 덮지 않음을 검증한다."""
    canonical={"top1_mask":np.array([True,False,False]),"top1_floor":np.array([2.,-1.,-1.]),
               "top5_mask":np.array([True,True,False]),"top5_floor":np.array([3.,1.,-1.])}
    parent={"top1_mask":np.array([True,True,False]),"top1_floor":np.array([.1,.7,-2.]),
            "top5_mask":np.array([True,False,True]),"top5_floor":np.array([.1,-2.,.9])}
    joint=union_constraints(canonical,parent)
    assert np.array_equal(joint["top1_mask"],[True,True,False]) and np.array_equal(joint["top5_mask"],[True,True,True])
    assert np.array_equal(joint["top1_floor"],[2.,.7,-2.]) and np.array_equal(joint["top5_floor"],[3.,1.,.9])
    joint["top1_floor"][0]=99;assert canonical["top1_floor"][0]==2 and parent["top1_floor"][0]==.1
    print(json.dumps(dict(self_test="pass",canonical_floor_not_weakened=True,newly_rescued_rows_protected=True,references_not_mutated=True)))
    return 0


def main() -> int:
    """기존 연구 출력을 보존하며 별도 joint 실험 또는 단위 검증을 실행한다."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=("run","self-test"),required=True)
    parser.add_argument("--output-dir",type=Path,default=OUTPUT)
    args=parser.parse_args()
    return _self_test() if args.mode=="self-test" else _run(args.output_dir)


if __name__=="__main__":
    raise SystemExit(main())
