"""완전히 거부된 head 교정을 원 anchor에서 실제 최소-step 차단 조건으로 재탐색한다."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import run_hwr_active_trust_head_only_v11 as head_scope

engine = head_scope.engine
rival_scope = head_scope.previous
OUTPUT = engine.ROOT / "artifacts/hwr_active_trust_retention_20261005_guard_replanning_v12"
GUARD_ROLE = "discovered_current_anchor_blocker"
GUARD_CAP = 16


def guard_records(active):
    """최초 실제 차단 증거를 row/rank/rival 순서로 고정해 회차간 상태를 검사한다."""
    return [dict(active[key]) for key in sorted(active)]


def update_guards(active, committed, after, trials, round_id, directions):
    """실제 승격 후 해결된 조건만 제거하고 전부 거부되면 최소-step의 새 차단 key를 기억한다."""
    discovered = []
    if committed:
        active = {key:item for key,item in active.items() if not after["safe"][key[1]][key[0]]}
    else:
        assert len(trials) == len(engine.ALPHAS) and trials[-1]["alpha"] == engine.ALPHAS[-1]
        for item in trials[-1]["new_previously_safe_failures"]:
            key = (item["row"], item["rank"], item["rival"])
            if key not in active:
                active[key] = dict(item); discovered.append(dict(item))
    replan = bool(not committed and discovered and round_id+1 < engine.ROUNDS and directions < 512)
    return active, sorted(discovered,key=lambda i:(i["row"],i["rank"],i["rival"])), replan


def make_selector(state, records):
    """v7 원 위험·정상 prefix 뒤에 실제 차단 pair를 기존 16개 대체 슬롯 안에서 보호한다."""
    prior_selector = rival_scope.previous.make_selector()
    def select(logits, y, spec, audit, bank):
        """현재 anchor 고정 pair slack 순으로 차단 조건을 넣고 나머지 슬롯은 v7 인접 후보로 채운다."""
        prior = prior_selector(logits,y,spec,audit,bank)
        primary = [i for i in prior if i["role"] in rival_scope.PRIORITY_ROLES]
        chosen=[]; seen=set()
        def add(item):
            """원 floor와 64개 상한을 유지하며 기존 prefix·차단 pair·인접 후보 중복을 제거한다."""
            key=(item["row"],item["rank"],item["rival"])
            if key not in seen and len(chosen)<engine.DIRECTION_CAP:
                seen.add(key); chosen.append(item); return True
            return False
        for item in primary: add(item)
        candidates=[]
        for item in state["active"].values():
            row,k,rv=item["row"],item["rank"],item["rival"]
            assert spec[f"top{k}_mask"][row] and rv != y[row]
            if (row,k,rv) not in seen:
                slack=float(logits[row,y[row]]-logits[row,rv])-float(spec[f"top{k}_floor"][row])
                candidates.append((slack,row,k,rv))
        candidates.sort(); admitted=0
        for _,row,k,rv in candidates:
            if admitted == GUARD_CAP or len(chosen) == engine.DIRECTION_CAP: break
            if add(dict(row=row,rank=k,rival=rv,floor=float(spec[f"top{k}_floor"][row]),role=GUARD_ROLE)): admitted+=1
        guard_admitted=admitted; order=np.argsort(-logits,axis=1,kind="stable"); alternatives=[]
        for item in primary:
            row,k=item["row"],item["rank"]; others=order[row][order[row]!=y[row]]
            rv=int(others[1 if k==1 else 5]); gap=float(logits[row,item["rival"]]-logits[row,rv]); assert gap>=0
            alternatives.append((gap,row,k,rv))
        for _,row,k,rv in sorted(alternatives):
            if admitted == rival_scope.ALTERNATIVE_CAP or len(chosen) == engine.DIRECTION_CAP: break
            if add(dict(row=row,rank=k,rival=rv,floor=float(spec[f"top{k}_floor"][row]),role="priority_near_alternative_rival")): admitted+=1
        for item in prior:
            if item["role"] not in rival_scope.PRIORITY_ROLES: add(item)
        records.append(dict(round=len(records),eligible_guard_candidates=[dict(row=row,rank=k,rival=rv,current_fixed_slack=slack)
            for slack,row,k,rv in candidates],guard_admitted=guard_admitted,combined_alternative_admitted=admitted))
        return chosen
    return select


def run_driver(out, state):
    """v11 실측·미분 기록 hook을 사용하되 모든 거부 후 실제 새 차단 조건이 있으면 같은 anchor에서 재계획한다."""
    import torch
    from run_hwr_affine_distillation_experiment_v1 import _load_teacher, _predict_logits
    if out.exists(): raise FileExistsError("refusing to replace frozen replanning experiment")
    if engine._guard_commit("before_head_guard_replanning") is None: return 78
    prior=json.loads((head_scope_base / "frozen_plan.json").read_text(encoding="utf-8"))
    verified=json.loads((head_scope_base / "independent_verification.json").read_text(encoding="utf-8"))
    assert engine._sha(head_scope_base / "active_trust_result.json")==verified["result_sha256"]
    assert engine._sha(engine.DEFAULT_CHECKPOINT)==prior["canonical_checkpoint_sha256"]
    assert engine._sha(engine.CANDIDATE / "directional_guard.pt")==prior["parent_checkpoint_sha256"]
    for name,digest in prior["reference_hashes"].items(): assert engine._sha(engine.REFERENCE / f"{name}.npy")==digest
    load=lambda name:np.load(engine.REFERENCE / f"{name}.npy",allow_pickle=False)
    x,y=load("joint_features"),load("joint_labels")
    spec={f"top{k}_{name}":load(f"reference_top{k}_{name}") for k in (1,5) for name in ("mask","floor")}
    assert x.shape==(2048,128,5) and len(np.unique(y[:1024]))==371
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True); device=torch.device("cpu")
    model,labels,_=_load_teacher(engine.CANDIDATE / "directional_guard.pt",device)
    canonical,canonical_labels,_=_load_teacher(engine.DEFAULT_CHECKPOINT,device); assert labels==canonical_labels
    named=list(model.named_parameters()); base=dict(canonical.named_parameters())
    def update_norm():
        """canonical 대비 모든 실제 FP32 tensor의 차이를 FP64로 합산한다."""
        return sum(float((p.detach().double()-base[n].detach().double()).square().sum()) for n,p in named)**.5
    z=_predict_logits(model,x,device,32)
    assert np.array_equal(z,np.concatenate((load("parent_old_logits"),load("parent_source_logits"))))
    raw_norm=update_norm(); out.mkdir(parents=True); plan=dict(prior)
    engine._write(out / "frozen_plan.json",plan)
    history=[]; bank=[]; directions=0; accepted=0; reason="bounded_head_replanning_did_not_converge"
    for round_id in range(engine.ROUNDS):
        before=engine.inspect(z,y,spec)
        if not before["failures"]: reason="all_original_joint_constraints_pass"; break
        guard_before=guard_records(state["active"]); anchor=accepted
        chosen=engine.select(z,y,spec,before,bank); directions+=len(chosen); assert directions<=512
        snapshot={n:p.detach().clone() for n,p in named}
        record=dict(round=round_id,before_merit=before["merit"],before_domains=engine.domains(z,y,spec),selected=chosen,
            total_gradient_directions=directions,trials=[],anchor_revision_before=anchor,guard_state_before=guard_before)
        gradients=[]; rhs=[]; lengths=[]; model.eval()
        for item in chosen:
            row,rv=item["row"],item["rival"]; values=model.math_head(model.encode(torch.from_numpy(x[row:row+1].copy())))
            margin=values[0,int(y[row])]-values[0,rv]
            grad=torch.autograd.grad(margin,[p for _,p in named],allow_unused=True)
            flat=torch.cat([torch.zeros_like(p,dtype=torch.float64).reshape(-1) if g is None else g.detach().double().reshape(-1)
                for (_,p),g in zip(named,grad)])
            length=float(flat.norm()); lengths.append(length); assert np.isfinite(length) and length>1e-10
            gradients.append(flat/length); rhs.append((item["floor"]+engine.SLACK-float(margin.detach()))/length)
        matrix=torch.stack(gradients); gram=(matrix@matrix.T).numpy()
        coefficients,certificate=engine.solve_polished(gram,np.array(rhs))
        record.update(solver=certificate,gradient_norms_fp64=lengths)
        np.savez(out / f"linearized_round_{round_id:03d}.npz",gram=gram,rhs=np.array(rhs),coefficients=coefficients)
        if not certificate["certified"]:
            reason="active_linearized_solver_not_certified"; record["reason"]=reason; history.append(record); break
        step=torch.from_numpy(coefficients)@matrix; record["proposal_step_l2_fp64"]=float(step.norm())
        new_bank={}; committed=False
        for trial_id,alpha in enumerate(engine.ALPHAS):
            offset=0
            with torch.no_grad():
                for name,p in named:
                    count=p.numel(); p.copy_((snapshot[name].double()+alpha*step[offset:offset+count].reshape(p.shape)).to(p.dtype)); offset+=count
            actual=_predict_logits(model,x,device,32); after=engine.inspect(actual,y,spec)
            allowed,why,newly_bad=engine.accept(before,after); norm=update_norm()
            if norm>2*raw_norm or not np.isfinite(norm): allowed=False; why="update_norm_cap"
            for item in newly_bad:
                key=(item["row"],item["rank"])
                if key not in new_bank or item["deficit"]>new_bank[key]["deficit"]: new_bank[key]=dict(item)
            trial=dict(alpha=alpha,accepted=allowed,reason=why,merit=after["merit"],update_l2_fp64=norm,
                domains=engine.domains(actual,y,spec),new_previously_safe_failures=newly_bad)
            record["trials"].append(trial); np.savez(out / f"trial_rank_{round_id:03d}_{trial_id:02d}.npz",**after["rank"])
            if round_id==0: np.save(out / f"first_trial_logits_{trial_id:02d}.npy",actual,allow_pickle=False)
            print(json.dumps(dict(event="head_guard_trial",round=round_id,anchor=anchor,alpha=alpha,accepted=allowed,
                reason=why,merit=after["merit"],new_guard_losses=len(newly_bad))),flush=True)
            if allowed:
                z=actual; accepted+=1; committed=True; record["accepted_alpha"]=alpha; break
            with torch.no_grad():
                for name,p in named: p.copy_(snapshot[name])
            assert all(torch.equal(p.detach().view(torch.int32),snapshot[n].view(torch.int32)) for n,p in named)
            trial["all_parameters_rollback_bit_exact"]=True
        bank=sorted(new_bank.values(),key=lambda i:(-i["deficit"],i["row"],i["rank"]))
        state["active"],discovered,replan=update_guards(state["active"],committed,after,record["trials"],round_id,directions)
        record.update(next_interference_bank=bank,step_accepted=committed,guard_state_after=guard_records(state["active"]),
            newly_discovered_guards=discovered,replanned_at_same_anchor=replan,anchor_revision_after=accepted)
        history.append(record)
        if not committed and not replan: reason="no_new_blocker_or_replanning_budget_exhausted"; break
    final=_predict_logits(model,x,device,32); assert np.array_equal(final,z)
    audit=engine.inspect(final,y,spec); metrics=engine.domains(final,y,spec)
    gain=(metrics["source_digits"]["top1_hits"]-853)/(958-853)
    passed=not audit["failures"] and gain>=.9 and update_norm()<=2*raw_norm
    if passed: reason="all_original_joint_constraints_pass"
    checkpoint=out / ("active_trust_repaired_research.pt" if passed else "failed_active_trust_research.pt")
    torch.save(dict(state_dict=model.state_dict(),math_labels=labels,auxiliary_labels=[],report=dict(
        input_contract=dict(observed_channel_mode="uniform-time"),product_adopted=False)),checkpoint)
    np.save(out / "active_trust_logits.npy",final,allow_pickle=False)
    assert engine._sha(engine.DEFAULT_CHECKPOINT)==plan["canonical_checkpoint_sha256"]
    assert engine._sha(engine.CANDIDATE / "directional_guard.pt")==plan["parent_checkpoint_sha256"]
    result=dict(schema="aiflow-head-guard-replanning-result/v12",status="train_gate_pass" if passed else "train_gate_fail",reason=reason,
        frozen_plan_sha256=engine._sha(out / "frozen_plan.json"),checkpoint_file=checkpoint.name,checkpoint_sha256=engine._sha(checkpoint),
        history=history,metrics=metrics,final_failures=audit["failures"],final_merit=audit["merit"],source_train_gain_retained_fraction=gain,
        accepted_steps=accepted,gradient_directions_used=directions,actual_model_checks=sum(len(h["trials"]) for h in history),
        raw_update_l2_fp64=raw_norm,final_update_l2_fp64=update_norm(),canonical_checkpoint_unchanged=True,parent_checkpoint_unchanged=True,
        parameter_optimization_performed=accepted>0,human_boundary_labels=0,held_inputs_forwarded=0,crohme_rows=0,product_adopted=False,
        eligible_for_independent_performance_claim=False,eligible_for_product_selection=False)
    engine._write(out / "active_trust_result.json",result)
    print(json.dumps(dict(event="head_guard_complete",status=result["status"],accepted_steps=accepted,directions=directions,
        failures=len(audit["failures"]),merit=audit["merit"])),flush=True)
    return 0


head_scope_base = head_scope.OUTPUT


def run() -> int:
    """봉인 v11의 head/encoder 기록과 v7의 runtime을 재사용하고 재계획 driver만 별도 적용한다."""
    prior=json.loads((head_scope_base / "frozen_plan.json").read_text(encoding="utf-8"))
    head_verified=json.loads((head_scope_base / "independent_head_verification.json").read_text(encoding="utf-8"))
    assert engine._sha(Path(head_scope.__file__))==prior["entrypoint_sha256"]
    assert engine._sha(head_scope_base / "active_trust_result.json")==head_verified["result_sha256"]
    state={"active":{}}; records=[]
    original_output,original_driver,original_factory,original_write=head_scope.OUTPUT,engine.run,rival_scope.make_selector,engine._write
    def write_variant(path,data):
        """v11 hook 뒤에서 원 예산에 재계획을 포함하는 계약 및 실제 차단 선택 기록을 봉인한다."""
        if path == OUTPUT / "frozen_plan.json":
            data.update(entrypoint_file=Path(__file__).name,entrypoint_sha256=engine._sha(Path(__file__)),
                guard_replanning_revision="v12",guard_slot_cap=GUARD_CAP,
                replanning_predecessor_result_sha256=head_verified["result_sha256"],
                changed_factors="only actual smallest-step blocker discovery and same-anchor replanning versus v11; head-only gradient, original floors/guards/alphas/solver and 64/512 caps unchanged",
                selection="unchanged v7 primary prefix; active discovered guard pairs by current fixed-pair slack share 16 slots with v7 adjacent rivals; v6 filler to 64",
                guard_memory="first exact row/rank/rival discovery from smallest step after six rejections; never remove while rolled back; after accepted step remove only rows whose original actual rank and floor pass",
                replanning_policy="continue after complete rejection only when a new exact blocker is discovered and one of original eight plans / 512 directions remains; no budget reset")
            data["dependencies"][Path(head_scope.__file__).name]=engine._sha(Path(head_scope.__file__))
        if path == OUTPUT / "active_trust_result.json":
            assert len(records)==len(data["history"]); data["guard_selection_records"]=records
        original_write(path,data)
    head_scope.OUTPUT,engine.run,rival_scope.make_selector,engine._write=OUTPUT,lambda out:run_driver(out,state),lambda:make_selector(state,records),write_variant
    try: return head_scope.run()
    finally:
        head_scope.OUTPUT,engine.run,rival_scope.make_selector,engine._write=original_output,original_driver,original_factory,original_write


def self_test() -> int:
    """특정 label 없이 차단 발견·같은 anchor 유지·승격 후 실제 해결 해제·예산 상한·빈 시작 parity를 검사한다."""
    active={}; bad=dict(row=55,rank=1,rival=3,floor=.02,margin=-.01,deficit=.03)
    trials=[dict(alpha=a,new_previously_safe_failures=[bad]) for a in engine.ALPHAS]
    after={"safe":{1:np.zeros(2048,dtype=bool),5:np.zeros(2048,dtype=bool)}}
    active,found,replan=update_guards(active,False,after,trials,0,64); assert replan and len(found)==1
    same,found,replan=update_guards(active.copy(),False,after,trials,1,128); assert same==active and not found and not replan
    kept,_,_=update_guards(active.copy(),True,after,trials,1,128); assert kept==active
    after["safe"][1][55]=True
    cleared,_,_=update_guards(active.copy(),True,after,trials,1,128); assert not cleared
    _,_,replan=update_guards({},False,after,trials,7,512); assert not replan
    y=np.zeros(2048,dtype=np.int64); z=np.full((2048,372),-100.,dtype=np.float32); z[:,1]=0.; z[:,0]=2.
    z[:4,0]=[.1,.2,.3,.4]; z[55,0]=.5
    spec={"top1_mask":np.ones(2048,dtype=bool),"top1_floor":np.ones(2048,dtype=np.float32),
        "top5_mask":np.zeros(2048,dtype=bool),"top5_floor":np.zeros(2048,dtype=np.float32)}
    audit=rival_scope.guard.inspect_guard(z,y,spec)
    assert make_selector({"active":{}},[])(z,y,spec,audit,[])==rival_scope.make_selector()(z,y,spec,audit,[])
    chosen=make_selector({"active":active},[])(z,y,spec,audit,[])
    assert any(i["row"]==55 and i["rival"]==3 and i["role"]==GUARD_ROLE for i in chosen)
    assert len(chosen)<=64 and sum(i["role"] in (GUARD_ROLE,"priority_near_alternative_rival") for i in chosen)<=16
    assert all(i["floor"]==float(spec[f'top{i["rank"]}_floor'][i["row"]]) for i in chosen)
    print(json.dumps(dict(self_test="pass",actual_guard_discovery=True,no_new_guard_stops=True,
        guards_persist_until_actual_pass=True,no_budget_reset=True,empty_guard_selection_exact_v7=True)))
    return 0


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--mode",choices=("run","self-test"),required=True)
    raise SystemExit(self_test() if parser.parse_args().mode=="self-test" else run())
