"""정상 경계 제약·대체 rival·실제 모델 backtracking을 합친 봉인 TRAIN 교정."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from run_hwr_joint_retention_repair_v1 import (
    ROOT, OUTPUT as REFERENCE, DEFAULT_CHECKPOINT, CANDIDATE,
    _sha, _write, _guard_commit,
)
from run_hwr_simultaneous_retention_repair_polished_v2 import solve_polished, OUTPUT as PREVIOUS
from verify_hwr_joint_retention_repair_v1 import ranks, measure

OUTPUT = ROOT / "artifacts/hwr_active_trust_retention_20261005"
ROUNDS, DIRECTION_CAP, SLACK = 8, 64, 1e-4
ALPHAS = (1., .5, .25, .125, .0625, .03125)


def inspect(logits: np.ndarray, y: np.ndarray, spec: dict) -> dict:
    """불변 실제 순위·margin으로 안전 마스크·위반 목록·제곱 deficit를 반환한다."""
    rank = ranks(logits, y)
    failures = []; safe = {}; merit = 0.
    for k in (1, 5):
        protected = spec[f"top{k}_mask"].astype(bool)
        deficit = np.maximum(spec[f"top{k}_floor"] - rank[f"top{k}_margin"], 0.)
        bad = protected & (~rank[f"top{k}_mask"] | (deficit > 0))
        safe[k] = protected & ~bad
        merit += float(np.square(deficit[protected].astype(np.float64)).sum())
        for row in np.flatnonzero(bad):
            failures.append(dict(row=int(row), rank=k, rival=int(rank[f"top{k}_rival"][row]),
                floor=float(spec[f"top{k}_floor"][row]), margin=float(rank[f"top{k}_margin"][row]), deficit=float(deficit[row])))
    return dict(rank=rank, safe=safe, merit=merit,
                failures=sorted(failures, key=lambda f: (-f["deficit"], f["row"], f["rank"])))


def accept(before: dict, after: dict) -> tuple[bool, str, list[dict]]:
    """어느 도메인이든 현재 정상인 고정 조건이 깨지면 거부하고 실제 merit 감소를 요구한다."""
    newly_failed = [item for item in after["failures"] if before["safe"][item["rank"]][item["row"]]]
    if newly_failed:
        return False, "new_previously_safe_constraint_failure", newly_failed
    if after["merit"] >= before["merit"] - 1e-12 * max(1., before["merit"]):
        return False, "actual_nonlinear_merit_not_improved", []
    return True, "no_new_failures_and_actual_merit_improved", []


def select(logits: np.ndarray, y: np.ndarray, spec: dict, audit: dict, bank: list[dict]) -> list[dict]:
    """최악 위반 32개·직전 간섭 16개·양쪽 근접 정상 8개에 대체 rival을 최대 64개까지 추가한다."""
    result = []; seen = set()
    def add(row, k, rival, role):
        """같은 row/rank/rival 중복을 제거하고 고정 floor와 선택 이유를 기록한다."""
        key = (row, k, rival)
        if key not in seen and len(result) < DIRECTION_CAP:
            seen.add(key)
            result.append(dict(row=row, rank=k, rival=rival, floor=float(spec[f"top{k}_floor"][row]), role=role))
    for item in audit["failures"][:32]:
        add(item["row"], item["rank"], item["rival"], "currently_failed")
    for item in bank[:16]:
        row, k = item["row"], item["rank"]
        add(row, k, int(audit["rank"][f"top{k}_rival"][row]), "previous_trial_interference")
    for lower, upper in ((0, 1024), (1024, len(y))):
        nearby = [(float(audit["rank"][f"top{k}_margin"][row]-spec[f"top{k}_floor"][row]), row, k)
                  for k in (1, 5) for row in range(lower, upper) if audit["safe"][k][row]]
        for _, row, k in sorted(nearby)[:4]:
            add(row, k, int(audit["rank"][f"top{k}_rival"][row]), "currently_safe_near_floor")
    order = np.argsort(-logits, axis=1, kind="stable")
    for item in result.copy():
        row, k = item["row"], item["rank"]
        others = order[row][order[row] != y[row]]
        # second/sixth OTHER는 현재 실제 Top-1/5 조건이 이미 함의하는 약한 경쟁 조건이다.
        add(row, k, int(others[1 if k == 1 else 5]), "alternative_rival")
    return result


def domains(logits: np.ndarray, y: np.ndarray, spec: dict) -> dict:
    """같은 2048개 TRAIN 행의 두 도메인 실제 순위와 고정 위반 수를 분리한다."""
    return {name: measure(logits[a:b], y[a:b], {n: v[a:b].astype(bool) if n.endswith("mask") else v[a:b] for n, v in spec.items()})
            for a, b, name in ((0, 1024, "old_math"), (1024, len(y), "source_digits"))}


def run(out: Path) -> int:
    """각 proposal을 실제 372-way 모델로 검증하고 거부 시 57개 parameter를 원 상태로 복원한다."""
    import torch
    from run_hwr_affine_distillation_experiment_v1 import _load_teacher, _predict_logits
    if out.exists():
        raise FileExistsError("refusing to replace a frozen active-trust experiment")
    if _guard_commit("before_active_trust_retention") is None:
        return 78
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    prior = json.loads((PREVIOUS / "frozen_plan.json").read_text(encoding="utf-8"))
    verification = json.loads((PREVIOUS / "independent_verification.json").read_text(encoding="utf-8"))
    assert _sha(PREVIOUS / "simultaneous_result.json") == verification["result_sha256"]
    assert _sha(DEFAULT_CHECKPOINT) == prior["canonical_checkpoint_sha256"]
    assert _sha(CANDIDATE / "directional_guard.pt") == prior["parent_checkpoint_sha256"]
    for name, digest in prior["reference_hashes"].items():
        assert _sha(REFERENCE / f"{name}.npy") == digest
    load = lambda name: np.load(REFERENCE / f"{name}.npy", allow_pickle=False)
    x, y = load("joint_features"), load("joint_labels")
    spec = {f"top{k}_{name}": load(f"reference_top{k}_{name}") for k in (1, 5) for name in ("mask", "floor")}
    assert x.shape == (2048, 128, 5) and len(np.unique(y[:1024])) == 371
    device = torch.device("cpu")
    model, labels, _ = _load_teacher(CANDIDATE / "directional_guard.pt", device)
    canonical, canonical_labels, _ = _load_teacher(DEFAULT_CHECKPOINT, device)
    assert labels == canonical_labels and len(labels) == 372
    named = list(model.named_parameters()); base = dict(canonical.named_parameters())
    def update_norm():
        """canonical에서의 실제 FP32 parameter 차이를 FP64 합으로 검사한다."""
        return sum(float((p.detach().double()-base[n].detach().double()).square().sum()) for n, p in named)**.5
    z = _predict_logits(model, x, device, 32)
    assert np.array_equal(z, np.concatenate((load("parent_old_logits"), load("parent_source_logits"))))
    raw_norm = update_norm()
    out.mkdir(parents=True)
    plan = dict(schema="aiflow-active-trust-retention-plan/v1", script_sha256=_sha(Path(__file__)),
        prior_result_sha256=verification["result_sha256"], canonical_checkpoint_sha256=_sha(DEFAULT_CHECKPOINT),
        parent_checkpoint_sha256=_sha(CANDIDATE / "directional_guard.pt"), reference_hashes=prior["reference_hashes"],
        dependencies={name: _sha(ROOT / "scripts" / name) for name in (
            "run_hwr_simultaneous_retention_repair_polished_v2.py", "run_hwr_simultaneous_retention_repair_v1.py",
            "verify_hwr_joint_retention_repair_v1.py", "run_hwr_affine_distillation_experiment_v1.py")},
        max_rounds=ROUNDS, max_gradient_directions_per_round=DIRECTION_CAP, total_direction_cap=512,
        proposal_slack=SLACK, alphas=list(ALPHAS), norm_cap_multiple=2., immutable_fixed_floor_fraction=.5,
        source_gain_retention_gate=.9, protected_counts=prior["protected_counts"],
        selection="worst failures <=32, previous interference <=16, near-safe <=4/domain, then second/sixth OTHER rivals up to 64",
        changed_factors="active safe constraints + alternative rivals + actual nonlinear monotonic-safe backtracking",
        actual_step_acceptance="no new failure of any currently safe fixed condition AND strict decrease of global squared fixed-margin deficit AND norm cap",
        final_acceptance="all original joint rank/floor constraints AND unchanged 90% TRAIN gain gate",
        failure_policy="retain last accepted TRAIN champion; no original floor changes or canonical fake success",
        optimizer_created=False, parameter_optimization_attempted=True, human_boundary_labels=0,
        held_inputs_forwarded=0, official_test_rows_read=0, crohme_rows=0, product_adopted=False,
        limits="Same optimization TRAIN anchors. Not independent accuracy, human probabilities or all-domain boundary distillation.")
    _write(out / "frozen_plan.json", plan)
    history = []; bank = []; directions = 0; accepted = 0; reason = "bounded_active_trust_did_not_converge"
    for round_id in range(ROUNDS):
        before = inspect(z, y, spec)
        if not before["failures"]:
            reason = "all_original_joint_constraints_pass"; break
        chosen = select(z, y, spec, before, bank)
        directions += len(chosen); assert directions <= 512
        snapshot = {n: p.detach().clone() for n, p in named}
        record = dict(round=round_id, before_merit=before["merit"], before_domains=domains(z, y, spec),
                      selected=chosen, total_gradient_directions=directions, trials=[])
        gradients = []; rhs = []; lengths = []
        model.eval()
        for item in chosen:
            row, rival = item["row"], item["rival"]
            values = model.math_head(model.encode(torch.from_numpy(x[row:row+1].copy())))
            margin = values[0, int(y[row])]-values[0, rival]
            grad = torch.autograd.grad(margin, [p for _, p in named], allow_unused=True)
            flat = torch.cat([torch.zeros_like(p, dtype=torch.float64).reshape(-1) if g is None else g.detach().double().reshape(-1)
                              for (_, p), g in zip(named, grad)])
            length = float(flat.norm()); lengths.append(length)
            if not np.isfinite(length) or length <= 1e-10:
                reason = "unusable_gradient"; break
            gradients.append(flat/length); rhs.append((item["floor"]+SLACK-float(margin.detach()))/length)
        if len(gradients) != len(chosen):
            record["reason"] = reason; history.append(record); break
        matrix = torch.stack(gradients); gram = (matrix @ matrix.T).numpy()
        coefficients, certificate = solve_polished(gram, np.array(rhs))
        record["solver"] = certificate; record["gradient_norms_fp64"] = lengths
        np.savez(out / f"linearized_round_{round_id:03d}.npz", gram=gram, rhs=np.array(rhs), coefficients=coefficients)
        if not certificate["certified"]:
            reason = "active_linearized_solver_not_certified"; record["reason"] = reason; history.append(record); break
        step = torch.from_numpy(coefficients) @ matrix
        record["proposal_step_l2_fp64"] = float(step.norm())
        new_bank = {}; committed = False
        for trial_id, alpha in enumerate(ALPHAS):
            offset = 0
            with torch.no_grad():
                for name, p in named:
                    count = p.numel()
                    p.copy_((snapshot[name].double()+alpha*step[offset:offset+count].reshape(p.shape)).to(p.dtype)); offset += count
            actual = _predict_logits(model, x, device, 32)
            after = inspect(actual, y, spec)
            allowed, why, newly_bad = accept(before, after)
            norm = update_norm()
            if norm > 2*raw_norm or not np.isfinite(norm):
                allowed = False; why = "update_norm_cap"
            for item in newly_bad:
                key = (item["row"], item["rank"])
                if key not in new_bank or item["deficit"] > new_bank[key]["deficit"]:
                    new_bank[key] = item
            trial = dict(alpha=alpha, accepted=allowed, reason=why, merit=after["merit"], update_l2_fp64=norm,
                         domains=domains(actual, y, spec), new_previously_safe_failures=newly_bad)
            record["trials"].append(trial)
            np.savez(out / f"trial_rank_{round_id:03d}_{trial_id:02d}.npz", **after["rank"])
            if round_id == 0:
                np.save(out / f"first_trial_logits_{trial_id:02d}.npy", actual, allow_pickle=False)
            print(json.dumps(dict(event="active_trust_trial", round=round_id, **{k:v for k,v in trial.items() if k!="new_previously_safe_failures"},
                                  new_failures=len(newly_bad))), flush=True)
            if allowed:
                z = actual; accepted += 1; committed = True; record["accepted_alpha"] = alpha; break
            with torch.no_grad():
                for name, p in named:
                    p.copy_(snapshot[name])
            assert all(torch.equal(p.detach(), snapshot[n]) for n, p in named)
            trial["all_parameters_rollback_bit_exact"] = True
        bank = sorted(new_bank.values(), key=lambda item: (-item["deficit"], item["row"], item["rank"]))
        record["next_interference_bank"] = bank; record["step_accepted"] = committed
        history.append(record)
        if not committed:
            reason = "bounded_backtracking_no_safe_improving_step"; break
    final = _predict_logits(model, x, device, 32); assert np.array_equal(final, z)
    audit = inspect(final, y, spec); metrics = domains(final, y, spec)
    gain = (metrics["source_digits"]["top1_hits"]-853)/(958-853)
    passed = not audit["failures"] and gain >= .9 and update_norm() <= 2*raw_norm
    if passed:
        reason = "all_original_joint_constraints_pass"
    checkpoint = out / ("active_trust_repaired_research.pt" if passed else "failed_active_trust_research.pt")
    torch.save(dict(state_dict=model.state_dict(), math_labels=labels, auxiliary_labels=[],
                    report=dict(input_contract=dict(observed_channel_mode="uniform-time"), product_adopted=False)), checkpoint)
    np.save(out / "active_trust_logits.npy", final, allow_pickle=False)
    assert _sha(DEFAULT_CHECKPOINT) == plan["canonical_checkpoint_sha256"]
    assert _sha(CANDIDATE / "directional_guard.pt") == plan["parent_checkpoint_sha256"]
    result = dict(schema="aiflow-active-trust-retention-result/v1", status="train_gate_pass" if passed else "train_gate_fail",
        reason=reason, frozen_plan_sha256=_sha(out / "frozen_plan.json"), checkpoint_file=checkpoint.name,
        checkpoint_sha256=_sha(checkpoint), history=history, metrics=metrics, final_failures=audit["failures"],
        final_merit=audit["merit"], source_train_gain_retained_fraction=gain, accepted_steps=accepted,
        gradient_directions_used=directions, actual_model_checks=sum(len(r["trials"]) for r in history),
        raw_update_l2_fp64=raw_norm, final_update_l2_fp64=update_norm(), canonical_checkpoint_unchanged=True,
        parent_checkpoint_unchanged=True, parameter_optimization_performed=accepted > 0,
        human_boundary_labels=0, held_inputs_forwarded=0, crohme_rows=0, product_adopted=False,
        eligible_for_independent_performance_claim=False, eligible_for_product_selection=False)
    _write(out / "active_trust_result.json", result)
    print(json.dumps(dict(event="active_trust_complete", status=result["status"], reason=reason, metrics=metrics,
                         accepted_steps=accepted, directions=directions, failures=len(audit["failures"]))), flush=True)
    return 0


def self_test() -> int:
    """실제 merit가 줄어도 정상 보호 행이 깨지면 거부하며 정상 개선만 허용하는지 검사한다."""
    y = np.zeros(3, dtype=np.int64); z = np.full((3, 372), -100., dtype=np.float32)
    z[:, 1] = 0.; z[:, 0] = [.5, 2., 2.]
    spec = {f"top{k}_{name}": value for k in (1, 5) for name, value in (
        ("mask", np.array([True, True, False])), ("floor", np.ones(3, dtype=np.float32) if k == 1 else np.zeros(3, dtype=np.float32)))}
    before = inspect(z, y, spec)
    unsafe = z.copy(); unsafe[:, 0] = [.9, .9, 2.]
    allowed, reason, broken = accept(before, inspect(unsafe, y, spec))
    assert not allowed and reason == "new_previously_safe_constraint_failure" and len(broken) == 1
    safe = z.copy(); safe[0, 0] = .75
    assert accept(before, inspect(safe, y, spec))[0]
    assert not accept(before, before)[0]
    print(json.dumps(dict(self_test="pass", unsafe_merit_improvement_rejected=True, safe_progress_accepted=True, no_progress_rejected=True)))
    return 0


def main() -> int:
    """새 출력에서만 봉인 실험 또는 작은 안전 수치 검사에 진입한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("run", "self-test"), required=True)
    args = parser.parse_args()
    return self_test() if args.mode == "self-test" else run(OUTPUT)


if __name__ == "__main__":
    raise SystemExit(main())
