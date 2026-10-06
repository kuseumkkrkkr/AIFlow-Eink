"""직전 회차의 고정 rival 음의 예측 오차로 조건별 제안 여유만 보완하는 TRAIN 실험."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import run_hwr_active_trust_rival_guard_v7 as previous

engine = previous.engine
guard = previous.guard
OUTPUT = engine.ROOT / "artifacts/hwr_active_trust_retention_20261005_curvature_buffer_v8"
ERROR_FACTOR, FLOOR_CAP_FRACTION, ABSOLUTE_CAP = 1.25, .25, .01


def proposal_buffers(chosen: list[dict], prior_errors: dict) -> np.ndarray:
    """원 floor를 변경하지 않고 같은 row/rank/rival의 직전 최대 음의 오차로 제안 여유를 제한한다."""
    return np.array([min(ERROR_FACTOR*prior_errors.get((i["row"], i["rank"], i["rival"]), 0.),
                         FLOOR_CAP_FRACTION*abs(i["floor"]), ABSOLUTE_CAP) for i in chosen], dtype=np.float64)


def error_records(errors: dict) -> list[dict]:
    """직전 회차만 사용할 오차 기억을 결정적인 순서의 JSON 행으로 변환한다."""
    return [dict(row=row, rank=k, rival=rival, negative_error=float(value)) for (row, k, rival), value in sorted(errors.items())]


def run() -> int:
    """봉인 원본 엔진은 수정하지 않고 동일 입력·선택·미분·backtracking을 명시적인 조건별 buffer로 실행한다."""
    import torch
    from run_hwr_affine_distillation_experiment_v1 import _load_teacher, _predict_logits
    if OUTPUT.exists():
        raise FileExistsError("refusing to replace a frozen curvature-buffer experiment")
    if engine._guard_commit("before_curvature_buffer_retention") is None:
        return 78
    prior = json.loads((previous.OUTPUT / "frozen_plan.json").read_text(encoding="utf-8"))
    verified = json.loads((previous.OUTPUT / "independent_verification.json").read_text(encoding="utf-8"))
    assert engine._sha(Path(previous.__file__)) == prior["entrypoint_sha256"]
    assert engine._sha(previous.OUTPUT / "active_trust_result.json") == verified["result_sha256"]
    assert engine._sha(engine.DEFAULT_CHECKPOINT) == prior["canonical_checkpoint_sha256"]
    assert engine._sha(engine.CANDIDATE / "directional_guard.pt") == prior["parent_checkpoint_sha256"]
    for name, digest in prior["dependencies"].items():
        assert engine._sha(engine.ROOT / "scripts" / name) == digest
    for name, digest in prior["reference_hashes"].items():
        assert engine._sha(engine.REFERENCE / f"{name}.npy") == digest
    load = lambda name: np.load(engine.REFERENCE / f"{name}.npy", allow_pickle=False)
    x, y = load("joint_features"), load("joint_labels")
    spec = {f"top{k}_{name}": load(f"reference_top{k}_{name}") for k in (1, 5) for name in ("mask", "floor")}
    assert x.shape == (2048, 128, 5) and len(np.unique(y[:1024])) == 371
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    device = torch.device("cpu")
    model, labels, _ = _load_teacher(engine.CANDIDATE / "directional_guard.pt", device)
    canonical, canonical_labels, _ = _load_teacher(engine.DEFAULT_CHECKPOINT, device)
    assert labels == canonical_labels and len(labels) == 372
    named = list(model.named_parameters()); base = dict(canonical.named_parameters())
    def update_norm():
        """canonical 대비 실제 FP32 파라미터 차이를 57개 텐서 전체에서 FP64로 합산한다."""
        return sum(float((p.detach().double()-base[n].detach().double()).square().sum()) for n, p in named)**.5
    z = _predict_logits(model, x, device, 32)
    assert np.array_equal(z, np.concatenate((load("parent_old_logits"), load("parent_source_logits"))))
    raw_norm = update_norm(); OUTPUT.mkdir(parents=True)
    plan = dict(prior)
    plan.update(schema="aiflow-curvature-buffer-plan/v8", entrypoint_file=Path(__file__).name,
        entrypoint_sha256=engine._sha(Path(__file__)), predecessor_result_sha256=verified["result_sha256"],
        driver_revision="v8_explicit_per_condition_buffers", curvature_buffer_revision="v8",
        curvature_error_factor=ERROR_FACTOR, curvature_floor_cap_fraction=FLOOR_CAP_FRACTION,
        curvature_absolute_cap=ABSOLUTE_CAP,
        curvature_memory="per row/rank/FIXED rival; maximum positive predicted-minus-actual margin over immediately prior round trials only; no initial prior-experiment errors",
        changed_factors="only per-condition proposal buffers from observed prior-round fixed-rival errors versus v7; original selection, acceptance floors, rank guards, solver, alpha and gradient caps unchanged",
        limits="Optimization TRAIN anchors only. Empirical numerical feedback, not human probabilities or all-domain semantic boundary distillation.")
    plan["dependencies"] = dict(prior["dependencies"])
    plan["dependencies"][Path(previous.__file__).name] = engine._sha(Path(previous.__file__))
    engine._write(OUTPUT / "frozen_plan.json", plan)
    trace = []; trial_trace = []
    def save_scores(actual):
        """추가 모델 호출 없이 초기·trial·최종 float32 전체 점수를 순서와 hash로 저장한다."""
        assert actual.shape == (2048, 372) and actual.dtype == np.float32 and np.isfinite(actual).all()
        assert len(trace) < plan["max_logged_forwards"]
        filename = f"full_logits_{len(trace):03d}.npy"
        np.save(OUTPUT / filename, actual, allow_pickle=False)
        item = dict(file=filename, sha256=engine._sha(OUTPUT / filename)); trace.append(item)
        return item
    save_scores(z)
    select = previous.make_selector()
    history = []; bank = []; prior_errors = {}; directions = 0; accepted = 0
    reason = "bounded_active_trust_did_not_converge"
    for round_id in range(engine.ROUNDS):
        before = guard.inspect_guard(z, y, spec)
        if not before["failures"]:
            reason = "all_original_joint_constraints_pass"; break
        chosen = select(z, y, spec, before, bank)
        buffers = proposal_buffers(chosen, prior_errors)
        directions += len(chosen); assert directions <= plan["total_direction_cap"]
        snapshot = {n: p.detach().clone() for n, p in named}
        record = dict(round=round_id, before_merit=before["merit"], before_domains=engine.domains(z, y, spec),
            selected=chosen, total_gradient_directions=directions, proposal_buffers=buffers.tolist(), trials=[],
            next_interference_bank=[], next_curvature_errors=[], step_accepted=False)
        gradients = []; rhs = []; lengths = []; model.eval()
        for index, item in enumerate(chosen):
            row, rival = item["row"], item["rival"]
            values = model.math_head(model.encode(torch.from_numpy(x[row:row+1].copy())))
            margin = values[0, int(y[row])]-values[0, rival]
            grad = torch.autograd.grad(margin, [p for _, p in named], allow_unused=True)
            flat = torch.cat([torch.zeros_like(p, dtype=torch.float64).reshape(-1) if g is None else g.detach().double().reshape(-1)
                              for (_, p), g in zip(named, grad)])
            length = float(flat.norm()); lengths.append(length)
            if not np.isfinite(length) or length <= 1e-10:
                reason = "unusable_gradient"; break
            gradients.append(flat/length)
            rhs.append((item["floor"]+engine.SLACK+buffers[index]-float(margin.detach()))/length)
        if len(gradients) != len(chosen):
            record["reason"] = reason; history.append(record); break
        matrix = torch.stack(gradients); gram = (matrix @ matrix.T).numpy()
        coefficients, certificate = engine.solve_polished(gram, np.array(rhs))
        record["solver"] = certificate; record["gradient_norms_fp64"] = lengths
        np.savez(OUTPUT / f"linearized_round_{round_id:03d}.npz", gram=gram, rhs=np.array(rhs), coefficients=coefficients)
        if not certificate["certified"]:
            reason = "active_linearized_solver_not_certified"; record["reason"] = reason; history.append(record); break
        step = torch.from_numpy(coefficients) @ matrix
        record["proposal_step_l2_fp64"] = float(step.norm())
        floors = np.array([i["floor"] for i in chosen]); lengths_array = np.array(lengths)
        fixed_before = floors+engine.SLACK+buffers-lengths_array*np.array(rhs)
        slope = lengths_array*(gram @ coefficients)
        new_bank = {}; new_errors = {}; committed = False
        for trial_id, alpha in enumerate(engine.ALPHAS):
            offset = 0
            with torch.no_grad():
                for name, p in named:
                    count = p.numel()
                    p.copy_((snapshot[name].double()+alpha*step[offset:offset+count].reshape(p.shape)).to(p.dtype)); offset += count
            actual = _predict_logits(model, x, device, 32)
            score_item = save_scores(actual); trial_trace.append(dict(round=round_id, trial=trial_id, **score_item))
            after = guard.inspect_guard(actual, y, spec)
            predicted = fixed_before+alpha*slope
            for index, item in enumerate(chosen):
                row, rival = item["row"], item["rival"]
                actual_fixed = float(actual[row, int(y[row])]-actual[row, rival])
                negative_error = max(float(predicted[index])-actual_fixed, 0.)
                key = (row, item["rank"], rival)
                if negative_error > 0:
                    new_errors[key] = max(new_errors.get(key, 0.), negative_error)
            allowed, why, newly_bad = guard.accept_guard(before, after)
            norm = update_norm()
            if norm > 2*raw_norm or not np.isfinite(norm):
                allowed = False; why = "update_norm_cap"
            for item in newly_bad:
                key = (item["row"], item["rank"])
                if key not in new_bank or item["deficit"] > new_bank[key]["deficit"]:
                    new_bank[key] = item
            trial = dict(alpha=alpha, accepted=allowed, reason=why, merit=after["merit"], update_l2_fp64=norm,
                domains=engine.domains(actual, y, spec), new_previously_safe_failures=newly_bad)
            record["trials"].append(trial)
            np.savez(OUTPUT / f"trial_rank_{round_id:03d}_{trial_id:02d}.npz", **after["rank"])
            if round_id == 0:
                np.save(OUTPUT / f"first_trial_logits_{trial_id:02d}.npy", actual, allow_pickle=False)
            print(json.dumps(dict(event="curvature_buffer_trial", round=round_id, alpha=alpha, accepted=allowed,
                reason=why, merit=after["merit"], new_failures=len(newly_bad))), flush=True)
            if allowed:
                z = actual; accepted += 1; committed = True; record["accepted_alpha"] = alpha; break
            with torch.no_grad():
                for name, p in named:
                    p.copy_(snapshot[name])
            assert all(torch.equal(p.detach(), snapshot[n]) for n, p in named)
            trial["all_parameters_rollback_bit_exact"] = True
        bank = sorted(new_bank.values(), key=lambda i: (-i["deficit"], i["row"], i["rank"]))
        prior_errors = new_errors
        record.update(next_interference_bank=bank, next_curvature_errors=error_records(new_errors), step_accepted=committed)
        history.append(record)
        if not committed:
            reason = "bounded_backtracking_no_safe_improving_step"; break
    final = _predict_logits(model, x, device, 32); assert np.array_equal(final, z); save_scores(final)
    audit = guard.inspect_guard(final, y, spec); metrics = engine.domains(final, y, spec)
    gain = (metrics["source_digits"]["top1_hits"]-853)/(958-853)
    passed = not audit["failures"] and gain >= .9 and update_norm() <= 2*raw_norm
    if passed:
        reason = "all_original_joint_constraints_pass"
    checkpoint = OUTPUT / ("active_trust_repaired_research.pt" if passed else "failed_active_trust_research.pt")
    torch.save(dict(state_dict=model.state_dict(), math_labels=labels, auxiliary_labels=[],
        report=dict(input_contract=dict(observed_channel_mode="uniform-time"), product_adopted=False)), checkpoint)
    np.save(OUTPUT / "active_trust_logits.npy", final, allow_pickle=False)
    assert engine._sha(engine.DEFAULT_CHECKPOINT) == plan["canonical_checkpoint_sha256"]
    assert engine._sha(engine.CANDIDATE / "directional_guard.pt") == plan["parent_checkpoint_sha256"]
    assert len(trace) == sum(len(h["trials"]) for h in history)+2
    result = dict(schema="aiflow-curvature-buffer-result/v8", status="train_gate_pass" if passed else "train_gate_fail",
        reason=reason, frozen_plan_sha256=engine._sha(OUTPUT / "frozen_plan.json"), checkpoint_file=checkpoint.name,
        checkpoint_sha256=engine._sha(checkpoint), history=history, metrics=metrics, final_failures=audit["failures"],
        final_merit=audit["merit"], source_train_gain_retained_fraction=gain, accepted_steps=accepted,
        gradient_directions_used=directions, actual_model_checks=sum(len(h["trials"]) for h in history),
        raw_update_l2_fp64=raw_norm, final_update_l2_fp64=update_norm(), canonical_checkpoint_unchanged=True,
        parent_checkpoint_unchanged=True, parameter_optimization_performed=accepted > 0,
        full_logits_trace=dict(initial=trace[0], trials=trial_trace, final=trace[-1], recorded_forwards=len(trace), extra_forwards=0),
        human_boundary_labels=0, held_inputs_forwarded=0, crohme_rows=0, product_adopted=False,
        eligible_for_independent_performance_claim=False, eligible_for_product_selection=False)
    engine._write(OUTPUT / "active_trust_result.json", result)
    print(json.dumps(dict(event="curvature_buffer_complete", status=result["status"], reason=reason,
        metrics=metrics, accepted_steps=accepted, directions=directions, failures=len(audit["failures"]))), flush=True)
    return 0


def self_test() -> int:
    """음의 고정 rival 오차만 반영하고 key 변경·초기 기억·절대/상대 상한을 검사한다."""
    items = [dict(row=4, rank=1, rival=3, floor=.2), dict(row=5, rank=5, rival=7, floor=.004)]
    assert np.array_equal(proposal_buffers(items, {}), np.zeros(2))
    assert np.allclose(proposal_buffers(items, {(4, 1, 3): .002, (5, 5, 7): .2}), [.0025, .001])
    assert np.allclose(proposal_buffers(items, {(4, 1, 3): .2}), [.01, 0.])
    assert np.array_equal(proposal_buffers(items, {(4, 1, 8): .2}), np.zeros(2))
    assert items[0]["floor"] == .2 and items[1]["floor"] == .004
    print(json.dumps(dict(self_test="pass", initial_memory_empty=True, fixed_rival_key_required=True,
        relative_and_absolute_caps_enforced=True, original_floors_unchanged=True)))
    return 0


def main() -> int:
    """새 봉인 실험 또는 모델을 수정하지 않는 제안 여유 단위 검증만 실행한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("run", "self-test"), required=True)
    return self_test() if parser.parse_args().mode == "self-test" else run()


if __name__ == "__main__":
    raise SystemExit(main())
