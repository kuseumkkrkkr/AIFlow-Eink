"""실제 rank 배열로 모든 trust trial의 승패·정상 제약 보존·최종 모델을 독립 검증한다."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from run_hwr_active_trust_retention_v1 import OUTPUT, REFERENCE, ROOT, DEFAULT_CHECKPOINT, CANDIDATE, _sha, _write
from verify_hwr_joint_retention_repair_v1 import ranks, measure, constraints


def summarize(rank: dict, spec: dict) -> tuple[dict, dict, float]:
    """저장 rank에서 보호 마스크·actual deficit·도메인 counts를 별도 계산한다."""
    safe = {}; metrics = {}; merit = 0.
    for k in (1, 5):
        protected = spec[f"top{k}_mask"].astype(bool)
        hit = rank[f"top{k}_mask"].astype(bool)
        deficit = np.maximum(spec[f"top{k}_floor"]-rank[f"top{k}_margin"], 0.)
        bad = protected & (~hit | (deficit > 0))
        safe[k] = protected & ~bad
        merit += float(np.square(deficit[protected].astype(np.float64)).sum())
        for a, b, domain in ((0, 1024, "old_math"), (1024, 2048, "source_digits")):
            metrics.setdefault(domain, {}).update({f"top{k}_hits": int(hit[a:b].sum()),
                f"top{k}_membership_regressions": int((protected[a:b] & ~hit[a:b]).sum()),
                f"top{k}_constraint_violations": int(bad[a:b].sum())})
    return safe, metrics, merit


def run(out: Path) -> int:
    """봉인 입력·수치 certificate·각 trial rank·57개 parameter 차이·batch 재현을 검증한다."""
    import torch
    from run_hwr_affine_distillation_experiment_v1 import _load_teacher, _predict_logits
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    destination = out / "independent_verification.json"
    if destination.exists():
        raise FileExistsError("refusing to replace verification")
    result = json.loads((out / "active_trust_result.json").read_text(encoding="utf-8"))
    plan = json.loads((out / "frozen_plan.json").read_text(encoding="utf-8"))
    assert _sha(out / "frozen_plan.json") == result["frozen_plan_sha256"]
    assert _sha(ROOT / "scripts/run_hwr_active_trust_retention_v1.py") == plan["script_sha256"]
    if "entrypoint_sha256" in plan:
        entry = plan.get("entrypoint_file", "run_hwr_active_trust_rank_guard_v2.py")
        assert entry in ("run_hwr_active_trust_rank_guard_v2.py", "run_hwr_active_trust_rank_priority_v3.py",
                         "run_hwr_active_trust_rank_persistence_v4.py", "run_hwr_active_trust_safe_priority_v5.py")
        assert _sha(ROOT / "scripts" / entry) == plan["entrypoint_sha256"]
    for name, digest in plan["dependencies"].items():
        assert _sha(ROOT / "scripts" / name) == digest
    for name, digest in plan["reference_hashes"].items():
        assert _sha(REFERENCE / f"{name}.npy") == digest
    assert _sha(DEFAULT_CHECKPOINT) == plan["canonical_checkpoint_sha256"]
    assert _sha(CANDIDATE / "directional_guard.pt") == plan["parent_checkpoint_sha256"]
    checkpoint = out / result["checkpoint_file"]
    assert _sha(checkpoint) == result["checkpoint_sha256"]
    load = lambda name: np.load(REFERENCE / f"{name}.npy", allow_pickle=False)
    x, y = load("joint_features"), load("joint_labels")
    old = constraints(load("canonical_old_logits"), load("parent_old_logits"), y[:1024])
    source = ranks(load("parent_source_logits"), y[1024:])
    source = {f"top{k}_{name}": source[f"top{k}_{original}"] * factor
              for k in (1, 5) for name, original, factor in (("mask", "mask", 1), ("floor", "margin", .5))}
    spec = {name: np.concatenate((old[name], source[name])) for name in old}
    for name, value in spec.items():
        assert np.array_equal(value, load(f"reference_{name}"))
    committed_rank = ranks(np.concatenate((load("parent_old_logits"), load("parent_source_logits"))), y)
    accepted = 0; directions = 0; trials_verified = []; certificates = []; pending_risks = set(); prior_bank = []
    for history in result["history"]:
        before_safe, before_metrics, before_merit = summarize(committed_rank, spec)
        assert before_metrics == history["before_domains"]
        assert before_merit == history["before_merit"]
        assert len(history["selected"]) <= plan["max_gradient_directions_per_round"]
        if plan.get("rank_persistence_revision") == "v4":
            pending_risks.intersection_update({(row, k) for row, k in pending_risks
                if committed_rank[f"top{k}_mask"][row] and committed_rank[f"top{k}_margin"][row] < spec[f"top{k}_floor"][row]})
            retained = [(row, k, int(committed_rank[f"top{k}_rival"][row]))
                        for _, row, k in sorted((float(committed_rank[f"top{k}_margin"][row]), row, k) for row, k in pending_risks)]
            actual_prefix = [(item["row"], item["rank"], item["rival"]) for item in history["selected"][:len(retained)]]
            assert retained == actual_prefix
            assert all(item["role"] == "retained_rank_risk_until_fixed_floor_pass" for item in history["selected"][:len(retained)])
            if plan.get("safe_interference_priority_revision") == "v5":
                prefix = []; seen = set()
                def append_prefix(row, k, role):
                    """저장 로그와 독립적으로 현재 rival·고정 floor·위험 prefix 중복을 재구성한다."""
                    key = (row, k, int(committed_rank[f"top{k}_rival"][row]))
                    if key not in seen and len(prefix) < plan["max_gradient_directions_per_round"]:
                        seen.add(key); prefix.append(dict(row=row, rank=k, rival=key[2],
                            floor=float(spec[f"top{k}_floor"][row]), role=role))
                        return True
                    return False
                for row, k, _ in retained:
                    append_prefix(row, k, "retained_rank_risk_until_fixed_floor_pass")
                for lower, upper in ((0, 1024), (1024, len(y))):
                    near = [(float(committed_rank[f"top{k}_margin"][row]), row, k)
                            for k in (1, 5) for row in range(lower, upper)
                            if spec[f"top{k}_mask"][row] and committed_rank[f"top{k}_mask"][row]]
                    for _, row, k in sorted(near)[:4]:
                        append_prefix(row, k, "currently_correct_near_rank_boundary")
                candidates = {(item["row"], item["rank"]) for item in prior_bank if before_safe[item["rank"]][item["row"]]}
                admitted = 0
                for _, row, k in sorted((float(committed_rank[f"top{k}_margin"][row]-spec[f"top{k}_floor"][row]), row, k)
                                       for row, k in candidates):
                    if append_prefix(row, k, "rejected_currently_safe_near_floor"):
                        admitted += 1
                    if admitted == plan["rejected_safe_bank_cap"] or len(prefix) == plan["max_gradient_directions_per_round"]:
                        break
                assert prefix == history["selected"][:len(prefix)]
                assert all(item["role"] not in ("retained_rank_risk_until_fixed_floor_pass",
                    "currently_correct_near_rank_boundary", "rejected_currently_safe_near_floor") for item in history["selected"][len(prefix):])
            for item in history["selected"]:
                if item["role"] in ("retained_rank_risk_until_fixed_floor_pass", "currently_correct_near_rank_boundary"):
                    row, k = item["row"], item["rank"]
                    assert committed_rank[f"top{k}_mask"][row] and spec[f"top{k}_mask"][row]
                    if committed_rank[f"top{k}_margin"][row] < spec[f"top{k}_floor"][row]:
                        pending_risks.add((row, k))
        if plan.get("rank_priority_revision") == "v3":
            prefix = []
            for lower, upper in ((0, 1024), (1024, len(y))):
                risk = [(float(committed_rank[f"top{k}_margin"][row]), row, k)
                        for k in (1, 5) for row in range(lower, upper)
                        if spec[f"top{k}_mask"][row] and committed_rank[f"top{k}_mask"][row]]
                prefix.extend((row, k, int(committed_rank[f"top{k}_rival"][row])) for _, row, k in sorted(risk)[:4])
            assert prefix == [(item["row"], item["rank"], item["rival"]) for item in history["selected"][:len(prefix)]]
        directions += len(history["selected"])
        assert directions == history["total_gradient_directions"] and directions <= plan["total_direction_cap"]
        for item in history["selected"]:
            assert item["floor"] == float(spec[f'top{item["rank"]}_floor'][item["row"]])
        if "solver" in history:
            a = np.load(out / f'linearized_round_{history["round"]:03d}.npz', allow_pickle=False)
            residual = a["gram"] @ a["coefficients"]-a["rhs"]
            stationarity = np.where(a["coefficients"] > 1e-10, np.abs(residual), np.maximum(-residual, 0.))
            certified = bool(np.isfinite(a["coefficients"]).all() and (a["coefficients"] >= 0).all()
                             and residual.min() >= -1e-9 and stationarity.max() <= 1e-8)
            assert certified == history["solver"]["certified"]
            assert np.linalg.eigvalsh(a["gram"]).min() >= -1e-10
            certificates.append(dict(round=history["round"], certified=certified, min_primal_residual=float(residual.min())))
        reconstructed_bank = {}
        for index, trial in enumerate(history["trials"]):
            with np.load(out / f'trial_rank_{history["round"]:03d}_{index:02d}.npz', allow_pickle=False) as a:
                rank = {name: a[name].copy() for name in a.files}
            safe, metrics, merit = summarize(rank, spec)
            assert metrics == trial["domains"] and merit == trial["merit"]
            newly_bad = {(int(row), k) for k in (1, 5) for row in np.flatnonzero(before_safe[k] & ~safe[k])}
            if plan.get("rank_guard_revision") == "v2":
                newly_bad |= {(int(row), k) for k in (1, 5) for row in np.flatnonzero(
                    spec[f"top{k}_mask"].astype(bool) & committed_rank[f"top{k}_mask"] & ~rank[f"top{k}_mask"])}
            assert newly_bad == {(item["row"], item["rank"]) for item in trial["new_previously_safe_failures"]}
            if plan.get("safe_interference_priority_revision") == "v5":
                for item in trial["new_previously_safe_failures"]:
                    row, k = item["row"], item["rank"]
                    floor = spec[f"top{k}_floor"][row]; margin = rank[f"top{k}_margin"][row]
                    expected_item = dict(row=row, rank=k, rival=int(rank[f"top{k}_rival"][row]),
                        floor=float(floor), margin=float(margin), deficit=float(np.maximum(floor-margin, 0.)))
                    assert item == expected_item
                    key = (row, k)
                    if key not in reconstructed_bank or item["deficit"] > reconstructed_bank[key]["deficit"]:
                        reconstructed_bank[key] = item
            expected = not newly_bad and merit < before_merit-1e-12*max(1., before_merit)
            expected = expected and np.isfinite(trial["update_l2_fp64"]) and trial["update_l2_fp64"] <= plan["norm_cap_multiple"]*result["raw_update_l2_fp64"]
            assert bool(expected) == trial["accepted"]
            assert trial["alpha"] == plan["alphas"][index]
            if history["round"] == 0:
                actual = np.load(out / f"first_trial_logits_{index:02d}.npy", allow_pickle=False)
                for name, value in ranks(actual, y).items():
                    assert np.array_equal(value, rank[name])
            if trial["accepted"]:
                assert index == len(history["trials"])-1
                committed_rank = rank; accepted += 1
            else:
                assert trial["all_parameters_rollback_bit_exact"]
            trials_verified.append(dict(round=history["round"], alpha=trial["alpha"], accepted=trial["accepted"],
                newly_failed=len(newly_bad), merit=merit))
        if plan.get("safe_interference_priority_revision") == "v5":
            prior_bank = sorted(reconstructed_bank.values(), key=lambda item: (-item["deficit"], item["row"], item["rank"]))
            assert prior_bank == history["next_interference_bank"]
    assert directions == result["gradient_directions_used"] and accepted == result["accepted_steps"]
    final = np.load(out / "active_trust_logits.npy", allow_pickle=False)
    for name, value in ranks(final, y).items():
        assert np.array_equal(value, committed_rank[name])
    safe, metrics, merit = summarize(committed_rank, spec)
    assert metrics == result["metrics"] and merit == result["final_merit"]
    device = torch.device("cpu")
    model, labels, _ = _load_teacher(checkpoint, device)
    canonical, canonical_labels, _ = _load_teacher(DEFAULT_CHECKPOINT, device)
    parent, parent_labels, _ = _load_teacher(CANDIDATE / "directional_guard.pt", device)
    assert labels == canonical_labels == parent_labels and len(np.unique(y[:1024])) == 371
    named = dict(model.named_parameters()); base = dict(canonical.named_parameters()); proposal = dict(parent.named_parameters())
    layers = [dict(name=name, canonical_delta_l2_fp64=float((p.detach().double()-base[name].detach().double()).norm()),
                   parent_delta_l2_fp64=float((p.detach().double()-proposal[name].detach().double()).norm())) for name, p in named.items()]
    norm = sum(item["canonical_delta_l2_fp64"]**2 for item in layers)**.5
    assert abs(norm-result["final_update_l2_fp64"]) < 1e-12
    checks = []
    for batch in (1, 16, 32):
        actual = _predict_logits(model, x, device, batch)
        actual_metrics = {name: measure(actual[a:b], y[a:b], {n: v[a:b].astype(bool) if n.endswith("mask") else v[a:b] for n,v in spec.items()})
                          for a, b, name in ((0, 1024, "old_math"), (1024, 2048, "source_digits"))}
        checks.append(dict(batch_size=batch, metrics=actual_metrics, metrics_exact=actual_metrics == metrics,
                           bit_exact=bool(np.array_equal(actual, final)), max_logit_delta=float(np.max(np.abs(actual-final)))))
    assert checks[-1]["bit_exact"]
    gain = (metrics["source_digits"]["top1_hits"]-853)/(958-853)
    assert gain == result["source_train_gain_retained_fraction"]
    gate = all(safe[k].sum() == spec[f"top{k}_mask"].astype(bool).sum() for k in (1, 5))
    gate = bool(gate and gain >= plan["source_gain_retention_gate"] and norm <= plan["norm_cap_multiple"]*result["raw_update_l2_fp64"])
    assert gate == (result["status"] == "train_gate_pass")
    report = dict(schema="aiflow-active-trust-retention-verification/v1", status="reproduced",
        verifier_sha256=_sha(Path(__file__)),
        verifier_snapshot="verifier_snapshot.py",
        result_sha256=_sha(out / "active_trust_result.json"), checkpoint_sha256=_sha(checkpoint),
        immutable_constraints_rebuilt_exact=True, direction_budget_verified=True, linear_certificates=certificates,
        actual_trial_acceptance_checks=trials_verified, accepted_steps=accepted, final_merit=merit,
        monotonic_safe_fixed_conditions=True, batch_checks=checks, parameter_tensors=len(layers), parameter_layers=layers,
        separate_current_rank_guard=plan.get("rank_guard_revision") == "v2",
        rank_priority_selection_verified=plan.get("rank_priority_revision") == "v3",
        rank_persistence_verified=plan.get("rank_persistence_revision") == "v4",
        rejected_safe_priority_and_bank_verified=plan.get("safe_interference_priority_revision") == "v5",
        canonical_and_parent_unchanged=True, strict_train_gate_pass=gate, human_boundary_labels=0,
        held_inputs_forwarded=0, crohme_rows=0, product_adopted=False, eligible_for_independent_performance_claim=False)
    snapshot = out / "verifier_snapshot.py"
    if snapshot.exists():
        raise FileExistsError("refusing to replace verifier source evidence")
    shutil.copyfile(Path(__file__), snapshot)
    assert _sha(snapshot) == report["verifier_sha256"]
    _write(destination, report)
    print(json.dumps(dict(status="reproduced", gate=result["status"], accepted_steps=accepted,
        trials=len(trials_verified), directions=directions, metrics=metrics, layers=len(layers))))
    return 0


def main() -> int:
    """검증할 별도 실험 폴더를 지정하고 기존 결과의 덮어쓰기를 거부한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    return run(parser.parse_args().output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
