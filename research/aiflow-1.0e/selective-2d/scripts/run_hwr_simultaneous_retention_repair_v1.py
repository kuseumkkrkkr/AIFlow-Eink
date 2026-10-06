"""동일 TRAIN 제약·예산에서 순차 교정 대신 Gram 기반 동시 최소노름 교정을 시험한다."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from run_hwr_joint_retention_repair_v1 import (
    OUTPUT as PREVIOUS, ROOT, DEFAULT_CHECKPOINT, CANDIDATE, PARENT, SOURCE,
    RETENTION, DEFAULT_DATA_DIR, _sha, _write, _guard_commit,
)
from repair_hwr_fixed_rank_margins_v1 import violations, rank_margins
from verify_hwr_joint_retention_repair_v1 import measure

OUTPUT = ROOT / "artifacts/hwr_simultaneous_retention_repair_20261005"
MAX_ROUNDS, MAX_CONSTRAINTS, SLACK = 32, 16, 1e-4


def solve_gram(gram: np.ndarray, rhs: np.ndarray) -> tuple[np.ndarray, dict]:
    """정규화 gradient Gram과 목표 증가량으로 비음수 dual을 풀고 실제 제약을 검사한다."""
    from scipy.optimize import minimize
    def objective(value):
        """최소노름 halfspace 문제의 convex dual 목적값·미분을 반환한다."""
        product = gram @ value
        return float(.5 * value @ product - rhs @ value), product - rhs
    answer = minimize(objective, np.zeros(len(rhs)), method="L-BFGS-B", jac=True,
                      bounds=[(0., None)] * len(rhs),
                      options=dict(maxiter=2000, maxls=50, ftol=1e-15, gtol=1e-12))
    value = answer.x
    residual = gram @ value - rhs
    stationarity = np.where(value > 1e-10, np.abs(residual), np.maximum(-residual, 0.))
    certificate = dict(solver_success=bool(answer.success), solver_iterations=int(answer.nit),
        solver_message=str(answer.message), min_primal_residual=float(residual.min()),
        max_stationarity_residual=float(stationarity.max()),
        certified=bool(np.isfinite(value).all() and (value >= 0).all()
                       and residual.min() >= -1e-9 and stationarity.max() <= 1e-8))
    return value, certificate


def run(out: Path) -> int:
    """원 후보에서 출발하며 각 회차 실제 372-way 순위·margin·동시 교정 잔차를 기록한다."""
    import torch
    from run_hwr_affine_distillation_experiment_v1 import _load_teacher, _predict_logits
    if out.exists():
        raise FileExistsError("refusing to replace a frozen simultaneous experiment")
    if _guard_commit("before_simultaneous_retention_repair") is None:
        return 78
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    previous = json.loads((PREVIOUS / "joint_repair_result.json").read_text(encoding="utf-8"))
    verified = json.loads((PREVIOUS / "independent_verification.json").read_text(encoding="utf-8"))
    prior_plan = json.loads((PREVIOUS / "frozen_plan.json").read_text(encoding="utf-8"))
    assert _sha(PREVIOUS / "joint_repair_result.json") == verified["result_sha256"]
    assert _sha(PREVIOUS / "frozen_plan.json") == previous["frozen_plan_sha256"]
    assert _sha(DEFAULT_CHECKPOINT) == prior_plan["canonical_checkpoint_sha256"]
    assert _sha(CANDIDATE / "directional_guard.pt") == prior_plan["parent_checkpoint_sha256"]
    load = lambda name: np.load(PREVIOUS / f"{name}.npy", allow_pickle=False)
    x, y = load("joint_features"), load("joint_labels")
    spec = {f"top{k}_{name}": load(f"reference_top{k}_{name}") for k in (1, 5) for name in ("mask", "floor")}
    assert x.shape == (2048, 128, 5) and len(np.unique(y[:1024])) == 371
    assert np.array_equal(x[:1024], np.load(PARENT / "replay_features.npy", allow_pickle=False))
    source_indices = np.load(RETENTION / "source_train_probe_indices.npy", allow_pickle=False)
    assert np.array_equal(x[1024:], np.load(SOURCE / "train_features_y_up.npy", mmap_mode="r", allow_pickle=False)[source_indices])
    replay_indices = np.load(PARENT / "replay_indices.npy", allow_pickle=False)
    assert np.array_equal(y[:1024], np.load(DEFAULT_DATA_DIR / "train_labels.npy", mmap_mode="r", allow_pickle=False)[replay_indices])
    device = torch.device("cpu")
    canonical, labels, _ = _load_teacher(DEFAULT_CHECKPOINT, device)
    model, other_labels, _ = _load_teacher(CANDIDATE / "directional_guard.pt", device)
    assert labels == other_labels
    digits = np.load(SOURCE / "train_digit_labels.npy", mmap_mode="r", allow_pickle=False)[source_indices]
    assert np.array_equal(y[1024:], np.array([labels.index(str(d)) for d in digits]))
    before = _predict_logits(model, x, device, 32)
    assert np.array_equal(before, np.concatenate((load("parent_old_logits"), load("parent_source_logits"))))
    assert np.array_equal(_predict_logits(canonical, x, device, 32), np.concatenate((load("canonical_old_logits"), load("canonical_source_logits"))))
    named = list(model.named_parameters())
    canonical_state = dict(canonical.named_parameters())
    raw_norm = sum(float((p.detach().double()-canonical_state[n].detach().double()).square().sum()) for n, p in named)**.5
    out.mkdir(parents=True)
    plan = dict(schema="aiflow-simultaneous-retention-plan/v1", script_sha256=_sha(Path(__file__)),
        previous_result_sha256=_sha(PREVIOUS / "joint_repair_result.json"), previous_plan_sha256=_sha(PREVIOUS / "frozen_plan.json"),
        dependency_hashes={name: _sha(ROOT / "scripts" / name) for name in (
            "run_hwr_joint_retention_repair_v1.py", "repair_hwr_fixed_rank_margins_v1.py",
            "verify_hwr_joint_retention_repair_v1.py", "run_hwr_affine_distillation_experiment_v1.py")},
        reference_hashes={name: _sha(PREVIOUS / f"{name}.npy") for name in ("joint_features", "joint_labels", "reference_top1_mask",
            "reference_top5_mask", "reference_top1_floor", "reference_top5_floor")},
        canonical_checkpoint_sha256=_sha(DEFAULT_CHECKPOINT), parent_checkpoint_sha256=_sha(CANDIDATE / "directional_guard.pt"),
        max_rounds=MAX_ROUNDS, max_constraints_per_round=MAX_CONSTRAINTS, correction_slack=SLACK, norm_cap_multiple=2.,
        selection="same globally worst 16 fixed-margin violations as sequential comparator",
        changed_factor="simultaneous minimal-norm halfspace correction instead of sequential correction",
        fixed_margin_fraction=.5, source_gain_retention_gate=.9, protected_counts=prior_plan["protected_counts"],
        solver="normalized gradient Gram convex dual, L-BFGS-B <=2000 iterations, actual primal and stationarity certificate",
        acceptance="all unchanged fixed joint rank/margin constraints AND unchanged 90% TRAIN gain gate",
        failure_policy="retain failed candidate and evidence; no floor weakening or canonical fake pass",
        optimizer_created=False, parameter_optimization_performed=True, human_boundary_labels=0,
        held_inputs_forwarded=0, official_test_rows_read=0, crohme_rows=0, product_adopted=False,
        limits="Same optimization TRAIN anchors, not independent evaluation. Digit adaptation plus 371-class retention, not all-domain human-boundary distillation.")
    _write(out / "frozen_plan.json", plan)
    history = []; reason = "bounded_simultaneous_projection_did_not_converge"; success = False
    for round_id in range(MAX_ROUNDS + 1):
        z = _predict_logits(model, x, device, 32)
        failures, counts = violations(z, y, spec)
        norm = sum(float((p.detach().double()-canonical_state[n].detach().double()).square().sum()) for n, p in named)**.5
        domains = {domain: measure(z[a:b], y[a:b], {n: v[a:b].astype(bool) if n.endswith("mask") else v[a:b] for n, v in spec.items()})
                   for a, b, domain in ((0, 1024, "old_math"), (1024, 2048, "source_digits"))}
        history.append(dict(round=round_id, update_l2_fp64=norm, domains=domains, **counts))
        print(json.dumps(dict(event="simultaneous_check", **history[-1])), flush=True)
        if not np.isfinite(norm) or norm > 2*raw_norm:
            reason = "update_norm_cap"; break
        if not failures:
            success = True; reason = "all_immutable_joint_constraints_pass"; break
        if round_id == MAX_ROUNDS:
            break
        selected = failures[:MAX_CONSTRAINTS]
        gradients = []; normalized_rhs = []; gradient_norms = []
        model.eval()
        for failure in selected:
            row, rival = failure["row"], failure["rival"]
            v = model.math_head(model.encode(torch.from_numpy(x[row:row+1].copy())))
            margin = v[0, int(y[row])] - v[0, rival]
            grads = torch.autograd.grad(margin, [p for _, p in named], allow_unused=True)
            flat = torch.cat([torch.zeros_like(p, dtype=torch.float64).reshape(-1) if g is None else g.detach().double().reshape(-1)
                              for (_, p), g in zip(named, grads)])
            length = float(flat.norm()); gradient_norms.append(length)
            if not np.isfinite(length) or length <= 1e-10:
                reason = "unusable_gradient"; break
            gradients.append(flat / length)
            normalized_rhs.append((failure["floor"] + SLACK - float(margin.detach())) / length)
        if len(gradients) != len(selected):
            break
        matrix = torch.stack(gradients)
        gram = (matrix @ matrix.T).numpy(); rhs = np.array(normalized_rhs)
        coefficients, certificate = solve_gram(gram, rhs)
        history[-1]["correction"] = dict(selected=selected, gradient_norms_fp64=gradient_norms, **certificate)
        np.savez(out / f"linearized_round_{round_id:03d}.npz", gram=gram, rhs=rhs, coefficients=coefficients)
        if not certificate["certified"]:
            reason = "local_simultaneous_solver_not_certified"; break
        change = torch.from_numpy(coefficients) @ matrix
        offset = 0
        with torch.no_grad():
            for _, p in named:
                count = p.numel()
                p.copy_((p.detach().double() + change[offset:offset+count].reshape(p.shape)).to(p.dtype)); offset += count
        history[-1]["correction"]["step_l2_fp64"] = float(change.norm())
    final = _predict_logits(model, x, device, 32)
    failures, counts = violations(final, y, spec)
    source_hits = int(rank_margins(final[1024:], y[1024:])["top1_hit"].sum())
    gain_retention = (source_hits - 853) / (958 - 853)
    success = bool(success and not failures and gain_retention >= .9)
    checkpoint = out / ("simultaneous_repaired_research.pt" if success else "failed_simultaneous_repair_research.pt")
    torch.save(dict(state_dict=model.state_dict(), math_labels=labels, auxiliary_labels=[],
                    report=dict(input_contract=dict(observed_channel_mode="uniform-time"), product_adopted=False)), checkpoint)
    np.save(out / "simultaneous_logits.npy", final, allow_pickle=False)
    assert _sha(DEFAULT_CHECKPOINT) == plan["canonical_checkpoint_sha256"]
    assert _sha(CANDIDATE / "directional_guard.pt") == plan["parent_checkpoint_sha256"]
    result = dict(schema="aiflow-simultaneous-retention-result/v1", status="train_gate_pass" if success else "train_gate_fail",
        reason=reason, frozen_plan_sha256=_sha(out / "frozen_plan.json"), checkpoint_file=checkpoint.name,
        checkpoint_sha256=_sha(checkpoint), history=history, metrics=history[-1]["domains"], final_constraints=counts,
        source_training_gain_retained_fraction=gain_retention, canonical_checkpoint_unchanged=True,
        parent_checkpoint_unchanged=True, parameter_optimization_performed=True, human_boundary_labels=0,
        held_inputs_forwarded=0, official_test_rows_read=0, crohme_rows=0, product_adopted=False,
        eligible_for_independent_performance_claim=False, eligible_for_product_selection=False)
    _write(out / "simultaneous_result.json", result)
    print(json.dumps(dict(event="simultaneous_complete", status=result["status"], reason=reason,
                         metrics=result["metrics"], constraints=counts)), flush=True)
    return 0


def self_test() -> int:
    """상충하지만 가능한 두 제약과 불가능한 두 제약의 인증 구분을 검사한다."""
    g = np.array([[1., 0.], [-.5, 1.]])
    norms = np.linalg.norm(g, axis=1); normalized = g/norms[:, None]
    coefficients, certificate = solve_gram(normalized @ normalized.T, np.ones(2)/norms)
    step = coefficients @ normalized
    assert certificate["certified"] and np.all(g @ step >= 1 - 1e-9)
    assert np.allclose(step, [1., 1.5], atol=1e-8)
    duplicate = np.array([[1., 0.], [1., 0.]])
    coefficients, certificate = solve_gram(duplicate @ duplicate.T, np.array([1., 2.]))
    assert certificate["certified"] and np.allclose(coefficients @ duplicate, [2., 0.], atol=1e-8)
    impossible = np.array([[1., 0.], [-1., 0.]])
    _, certificate = solve_gram(impossible @ impossible.T, np.ones(2))
    assert not certificate["certified"]
    print(json.dumps(dict(self_test="pass", joint_minimal_norm=True, duplicate_constraints=True, infeasible_not_certified=True)))
    return 0


def main() -> int:
    """기존 실험은 보존하며 독립 출력 또는 작은 수치 단위 검사만 실행한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("self-test", "run"), required=True)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    return self_test() if args.mode == "self-test" else run(args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
