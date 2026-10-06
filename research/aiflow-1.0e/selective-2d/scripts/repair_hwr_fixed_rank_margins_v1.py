"""고정 canonical Top-1/Top-5 margin을 동시에 지키는 TRAIN 후보 교정 실험.

이전 단계의 shrinking margin을 재사용하지 않는다. 기존 학습 표본에 대한
유한 제약의 검증일 뿐 독립 정확도·인간 인지 경계·제품 승격 증거가 아니다.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from audit_hwr_probability_boundary_tube_v1 import ROOT, DEFAULT_CHECKPOINT, DEFAULT_DATA_DIR, _guard_commit
from run_hwr_pendigits_tube_probe_v1 import OUTPUT as PARENT, SOURCE, _sha, _write
from run_hwr_pendigits_retention_probe_v1 import OUTPUT as RETENTION
from run_hwr_pendigits_directional_guard_v1 import OUTPUT as CANDIDATE

OUTPUT = ROOT / "artifacts/hwr_fixed_rank_margin_repair_20261005"
MAX_ROUNDS, CONSTRAINTS_PER_ROUND = 32, 16
CORRECTION_SLACK = 1e-4


def rank_margins(logits: np.ndarray, targets: np.ndarray) -> dict:
    """truth를 제외한 최고/다섯째 rival과 실제 stable Top-1/Top-5 포함을 구한다."""
    if logits.ndim != 2 or logits.shape[1] != 372 or targets.shape != (len(logits),) or not np.isfinite(logits).all():
        raise ValueError("rank constraints require finite aligned [N,372] logits")
    if (targets < 0).any() or (targets >= 372).any():
        raise ValueError("target out of range")
    rows = np.arange(len(logits))
    other = logits.copy(); other[rows,targets] = -np.inf
    rivals = np.argsort(-other,axis=1,kind="stable")
    order = np.argsort(-logits,axis=1,kind="stable")
    return dict(top1_hit=order[:,0] == targets, top5_hit=(order[:,:5] == targets[:,None]).any(1),
                top1_rival=rivals[:,0], top5_rival=rivals[:,4],
                top1_margin=logits[rows,targets]-logits[rows,rivals[:,0]],
                top5_margin=logits[rows,targets]-logits[rows,rivals[:,4]])


def fixed_constraints(reference_logits: np.ndarray, targets: np.ndarray) -> dict:
    """초기 canonical 정답 행/절반 margin을 한 번만 봉인해 이후 줄어들지 않게 한다."""
    rank = rank_margins(reference_logits,targets)
    return {f"top{k}_{name}":rank[f"top{k}_{source}"].copy() * scale
            for k in (1,5) for name,source,scale in (("mask","hit",1),("floor","margin",.5))}


def violations(logits: np.ndarray, targets: np.ndarray, constraints: dict) -> tuple[list[dict], dict]:
    """모든 보호 행의 실제 372-way 순위와 고정 margin을 함께 검사한다."""
    rank = rank_margins(logits,targets)
    failures,counts = [],{}
    for k in (1,5):
        mask = constraints[f"top{k}_mask"].astype(bool)
        bad = mask & (~rank[f"top{k}_hit"] | (rank[f"top{k}_margin"] < constraints[f"top{k}_floor"]))
        counts[f"top{k}_constraint_violations"] = int(bad.sum())
        counts[f"top{k}_membership_regressions"] = int((mask & ~rank[f"top{k}_hit"]).sum())
        for row in np.flatnonzero(bad):
            failures.append(dict(row=int(row),rank=k,rival=int(rank[f"top{k}_rival"][row]),
                                 margin=float(rank[f"top{k}_margin"][row]),floor=float(constraints[f"top{k}_floor"][row]),
                                 deficit=float(constraints[f"top{k}_floor"][row]-rank[f"top{k}_margin"][row])))
    return sorted(failures,key=lambda item:(-item["deficit"],item["row"],item["rank"])),counts


def repair(model, canonical_state: dict, x: np.ndarray, y: np.ndarray, constraints: dict, predict,
           max_rounds: int = MAX_ROUNDS) -> dict:
    """현재 후보에서 위반 margin 미분만 교정하고 전 anchor를 반복 검증한다."""
    import torch
    named = list(model.named_parameters())
    proposal = {name:value.detach().clone() for name,value in model.state_dict().items()}
    delta = {name:proposal[name].double()-canonical_state[name].double() for name,_ in named}
    raw_norm = sum(float(value.square().sum()) for value in delta.values()) ** .5
    history,corrections = [],[]
    for round_id in range(max_rounds + 1):
        with torch.no_grad():
            for name,value in model.state_dict().items():
                value.copy_((canonical_state[name].double()+delta[name]).to(value.dtype) if name in delta else proposal[name])
        logits = predict(model)
        failures,counts = violations(logits,y,constraints)
        norm = sum(float(value.square().sum()) for value in delta.values()) ** .5
        history.append(dict(round=round_id,update_l2_fp64=norm,**counts))
        print(json.dumps(dict(event="fixed_rank_margin_check",**history[-1])),flush=True)
        if not np.isfinite(norm) or norm > 2 * max(raw_norm,1e-12):
            return dict(success=False,reason="update_norm_cap",history=history,corrections=corrections,raw_update_l2_fp64=raw_norm)
        if not failures:
            return dict(success=True,reason="all_fixed_rank_constraints_pass",history=history,corrections=corrections,raw_update_l2_fp64=raw_norm)
        if round_id == max_rounds:
            break
        model.eval()
        round_delta = {name:value.clone() for name,value in delta.items()}
        for failure in failures[:CONSTRAINTS_PER_ROUND]:
            row,rival = failure["row"],failure["rival"]
            inputs = torch.from_numpy(np.array(x[row:row+1],dtype=np.float32,copy=True))
            values = model.math_head(model.encode(inputs))
            margin = values[0,int(y[row])]-values[0,rival]
            grads = torch.autograd.grad(margin,[parameter for _,parameter in named],allow_unused=True)
            gradient = {name:g.detach().double() for (name,_),g in zip(named,grads) if g is not None}
            denominator = sum(float(g.square().sum()) for g in gradient.values())
            if not np.isfinite(denominator) or denominator <= 1e-20:
                continue
            previous_corrections = sum(float(((delta[name]-round_delta[name])*g).sum()) for name,g in gradient.items())
            estimate = failure["margin"]+previous_corrections
            increase = max(failure["floor"]+CORRECTION_SLACK-estimate,0.)
            for name,g in gradient.items():
                delta[name].add_(g,alpha=increase/denominator)
            corrections.append(dict(round=round_id,**failure,current_candidate_autograd_margin=float(margin.detach()),
                                    corrected_margin_estimate_before=estimate,required_increase=increase,gradient_l2_squared_fp64=denominator))
    return dict(success=False,reason="bounded_projection_did_not_converge",history=history,corrections=corrections,raw_update_l2_fp64=raw_norm)


def _run(out: Path) -> int:
    """봉인된 371-class TRAIN anchor와 source TRAIN probe만으로 후보 교정을 비교한다."""
    if out.exists():
        raise FileExistsError("refusing to overwrite a rank-margin repair")
    if _guard_commit("before_fixed_rank_margin_repair") is None:
        return 78
    import torch
    from run_hwr_affine_distillation_experiment_v1 import _load_teacher,_predict_logits
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    previous = json.loads((CANDIDATE / "directional_result.json").read_text(encoding="utf-8"))
    verified = json.loads((CANDIDATE / "independent_verification.json").read_text(encoding="utf-8"))
    old = json.loads((PARENT / "frozen_plan.json").read_text(encoding="utf-8"))
    if _sha(CANDIDATE / "directional_result.json") != verified["directional_result_sha256"] or previous["checkpoint_sha256"] != _sha(CANDIDATE / "directional_guard.pt"):
        raise ValueError("previous candidate evidence changed")
    if _sha(DEFAULT_CHECKPOINT) != old["checkpoint_sha256"] or _sha(CANDIDATE / "frozen_plan.json") != previous["frozen_plan_sha256"]:
        raise ValueError("canonical/previous plan changed")
    arrays = {}
    for name in ("replay_features","replay_indices","train_indices"):
        item = old["artifacts"][name];path = PARENT / item["file"]
        if _sha(path) != item["sha256"]:
            raise ValueError("sealed TRAIN anchor arrays changed")
        arrays[name] = np.load(path,allow_pickle=False)
    for name,digest in old["replay_train_array_hashes"].items():
        if _sha(DEFAULT_DATA_DIR / f"train_{name}.npy") != digest:
            raise ValueError("TRAIN replay source changed")
    x = arrays["replay_features"]
    y = np.array(np.load(DEFAULT_DATA_DIR / "train_labels.npy",mmap_mode="r",allow_pickle=False)[arrays["replay_indices"]],copy=True)
    if len(np.unique(y)) != 371:
        raise ValueError("math anchor class coverage differs")
    source_report = json.loads((SOURCE / "pendigits_source_audit.json").read_text(encoding="utf-8"))
    if _sha(SOURCE / "pendigits_source_audit.json") != old["source_report_sha256"]:
        raise ValueError("digit source report changed")
    for name in ("train_features_y_up","train_digit_labels"):
        if _sha(SOURCE / f"{name}.npy") != source_report["artifacts"][name]["sha256"]:
            raise ValueError("digit TRAIN source array changed")
    indices = np.load(RETENTION / "source_train_probe_indices.npy",allow_pickle=False)
    expected = np.random.default_rng(old["seed"]+719).choice(len(arrays["train_indices"]),1024,replace=False)
    if not np.array_equal(indices,arrays["train_indices"][expected]):
        raise ValueError("source TRAIN probe rows differ")
    probe_x = np.array(np.load(SOURCE / "train_features_y_up.npy",mmap_mode="r",allow_pickle=False)[indices],copy=True)
    digits = np.load(SOURCE / "train_digit_labels.npy",mmap_mode="r",allow_pickle=False)[indices]
    device = torch.device("cpu")
    canonical,labels,_ = _load_teacher(DEFAULT_CHECKPOINT,device)
    model,other_labels,_ = _load_teacher(CANDIDATE / "directional_guard.pt",device)
    if labels != other_labels:
        raise ValueError("candidate label order changed")
    reference = _predict_logits(canonical,x,device,32)
    before = _predict_logits(model,x,device,32)
    constraints = fixed_constraints(reference,y)
    if int(constraints["top1_mask"].sum()) != 831 or int(constraints["top5_mask"].sum()) != 1005:
        raise ValueError("canonical protected rank counts differ")
    probe_y = np.array([labels.index(str(d)) for d in digits],dtype=np.int64)
    baseline_probe = _predict_logits(canonical,probe_x,device,32)
    prior_probe = _predict_logits(model,probe_x,device,32)
    baseline_hits = int((baseline_probe.argmax(1)==probe_y).sum());prior_hits = int((prior_probe.argmax(1)==probe_y).sum())
    if baseline_hits != 853 or prior_hits != 958:
        raise ValueError("TRAIN source probe baseline changed")
    out.mkdir(parents=True)
    plan = dict(schema="aiflow-fixed-rank-margin-repair-plan/v1",script_sha256=_sha(Path(__file__)),
                canonical_checkpoint_sha256=_sha(DEFAULT_CHECKPOINT),candidate_checkpoint_sha256=_sha(CANDIDATE / "directional_guard.pt"),
                candidate_result_sha256=_sha(CANDIDATE / "directional_result.json"),source_plan_sha256=_sha(PARENT / "frozen_plan.json"),
                source_probe_indices_sha256=_sha(RETENTION / "source_train_probe_indices.npy"),
                dependencies={name:_sha(ROOT / "scripts" / name) for name in ("run_hwr_pendigits_tube_probe_v1.py","run_hwr_pendigits_directional_guard_v1.py","run_hwr_affine_distillation_experiment_v1.py","evaluate_48hz_prefix_v1.py")},
                max_rounds=MAX_ROUNDS,constraints_per_round=CONSTRAINTS_PER_ROUND,correction_slack=CORRECTION_SLACK,norm_cap_multiple=2.,
                fixed_top1_fraction=.5,fixed_top5_fraction=.5,protected_top1_rows=831,protected_top5_rows=1005,train_anchor_classes=371,
                source_train_gain_retention_gate=.9,gate_role="TRAIN-only repair viability, NOT independent model/product selection",
                local_gradient_point="current candidate, not shrinking reference; canonical floors/masks immutable",
                failure_policy="keep failed research evidence, do not fall back to canonical and claim success",
                optimizer_created=False,parameter_optimization_performed=True,new_human_boundary_distillation_performed=False,
                held_inputs_forwarded=0,official_test_rows_read=0,crohme_rows=0,human_boundary_labels=0,product_adopted=False)
    _write(out / "frozen_plan.json",plan)
    for name,array in {"canonical_replay_logits":reference,"prior_candidate_replay_logits":before,"reference_top1_mask":constraints["top1_mask"].astype(bool),
                       "reference_top5_mask":constraints["top5_mask"].astype(bool),"reference_top1_floor":constraints["top1_floor"],"reference_top5_floor":constraints["top5_floor"]}.items():
        np.save(out / f"{name}.npy",array,allow_pickle=False)
    initial_state = {name:value.detach().clone() for name,value in canonical.state_dict().items()}
    status = repair(model,initial_state,x,y,constraints,lambda candidate:_predict_logits(candidate,x,device,32))
    final = _predict_logits(model,x,device,32);probe = _predict_logits(model,probe_x,device,32)
    failures,counts = violations(final,y,constraints)
    source_hits = int((probe.argmax(1)==probe_y).sum())
    retention = (source_hits-baseline_hits)/(prior_hits-baseline_hits)
    rank = rank_margins(final,y)
    metrics = dict(**counts,source_train_probe_baseline_hits=baseline_hits,source_train_probe_before_hits=prior_hits,source_train_probe_after_hits=source_hits,
                   source_training_gain_retained_fraction=retention,old_train_top1_hits=int(rank["top1_hit"].sum()),old_train_top5_hits=int(rank["top5_hit"].sum()))
    success = bool(status["success"] and not failures and retention >= .9)
    checkpoint = out / ("repaired_research.pt" if success else "failed_repair_research.pt")
    torch.save(dict(state_dict=model.state_dict(),math_labels=labels,auxiliary_labels=[],report=dict(input_contract=dict(observed_channel_mode="uniform-time"),product_adopted=False)),checkpoint)
    np.save(out / "repaired_replay_logits.npy",final,allow_pickle=False);np.save(out / "repaired_source_train_probe_logits.npy",probe,allow_pickle=False)
    if _sha(DEFAULT_CHECKPOINT)!=plan["canonical_checkpoint_sha256"] or _sha(CANDIDATE / "directional_guard.pt")!=plan["candidate_checkpoint_sha256"]:
        raise AssertionError("canonical/prior checkpoint changed")
    result = dict(schema="aiflow-fixed-rank-margin-repair-result/v1",status="training_repair_gate_pass" if success else "training_repair_gate_fail",
                  frozen_plan_sha256=_sha(out / "frozen_plan.json"),repair=status,metrics=metrics,
                  checkpoint_file=checkpoint.name,checkpoint_sha256=_sha(checkpoint),canonical_checkpoint_unchanged=True,prior_checkpoint_unchanged=True,
                  parameter_optimization_performed=True,new_human_boundary_distillation_performed=False,eligible_for_product_selection=False,
                  held_inputs_forwarded=0,official_test_rows_read=0,crohme_rows=0,human_boundary_labels=0,product_adopted=False)
    _write(out / "repair_result.json",result)
    print(json.dumps(dict(event="fixed_rank_margin_repair_complete",status=result["status"],metrics=metrics)),flush=True)
    return 0


def _self_test() -> int:
    """다섯째 OTHER rival·canonical floor 불변·비관련 성분 보존을 선형 toy로 검사한다."""
    import torch
    class Toy(torch.nn.Module):
        """실제 모델과 같은 372-way 인터페이스를 갖는 선형 검증용 모델이다."""
        def __init__(self):
            """teacher truth margin 1.0과 구분 가능한 Top-5 순서를 만든다."""
            super().__init__();self.math_head=torch.nn.Linear(5,372)
            with torch.no_grad():
                self.math_head.weight.zero_();self.math_head.bias.zero_();self.math_head.bias[:6]=torch.tensor([1.,0.,-.1,-.2,-.3,-.4])
        def encode(self,x):
            """toy 타점의 평균을 다섯 특징으로 반환한다."""
            return x.mean(1)
    model=Toy();x=np.zeros((1,128,5),dtype=np.float32);y=np.array([0],dtype=np.int64)
    def predict(candidate):
        """toy 372-way 점수를 autograd 없이 판정용으로 반환한다."""
        with torch.no_grad():return candidate.math_head(candidate.encode(torch.from_numpy(x))).numpy()
    reference=predict(model);state={n:v.detach().clone() for n,v in model.state_dict().items()};constraints=fixed_constraints(reference,y)
    frozen=constraints["top1_floor"].copy()
    with torch.no_grad():model.math_head.bias[0]=-.5;model.math_head.weight[100,4]=.5
    result=repair(model,state,x,y,constraints,predict,max_rounds=8)
    assert result["success"] and not violations(predict(model),y,constraints)[0]
    assert np.array_equal(frozen,constraints["top1_floor"]) and model.math_head.weight[100,4].item()==.5
    fixture=np.full((1,372),-100.,dtype=np.float32);fixture[0,:7]=[9.,8.,7.,6.,5.,4.,3.]
    assert rank_margins(fixture,np.array([2]))["top5_rival"].tolist()==[5]
    print(json.dumps(dict(self_test="pass",fifth_other_rival=True,fixed_floor_does_not_shrink=True,unrelated_weight_component_preserved=True)))
    return 0


def main() -> int:
    """격리 폴더에서 post-hoc TRAIN 교정 또는 제약 단위 검증을 실행한다."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=("run","self-test"),required=True)
    parser.add_argument("--output-dir",type=Path,default=OUTPUT)
    args=parser.parse_args()
    return _self_test() if args.mode=="self-test" else _run(args.output_dir)


if __name__=="__main__":
    raise SystemExit(main())
