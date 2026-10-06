"""수학 학습 리플레이의 정답 보존을 dropout 분리/업데이트 가드로 비교한다.

숫자와 기존 수학 TRAIN 데이터만 사용하며 보류 그룹/공식 test는 평가하지 않는다.
유한 학습 anchor의 보호이며 제품 비회귀나 사람 인지 경계를 증명하지 않는다.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from audit_hwr_probability_boundary_tube_v1 import DEFAULT_CHECKPOINT, DEFAULT_DATA_DIR, ROOT, _guard_commit
from run_hwr_pendigits_tube_probe_v1 import SOURCE, OUTPUT as PARENT, _sha, _write

OUTPUT = ROOT / "artifacts/hwr_pendigits_retention_probe_20261005"
FRACTIONS = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125)


def _guard_step(model, optimizer, before_weights, before_optimizer, accepts):
    """제안 업데이트를 축소하며 검사하고 전부 실패하면 가중치/Adam 상태를 복구한다."""
    import torch
    proposal = {name: value.detach().clone() for name, value in model.state_dict().items()}
    tested = []
    for fraction in FRACTIONS:
        with torch.no_grad():
            for name, value in model.state_dict().items():
                if value.is_floating_point():
                    value.copy_(proposal[name] if fraction == 1.0 else before_weights[name] + fraction * (proposal[name] - before_weights[name]))
                else:
                    value.copy_(proposal[name])
        accepted, details = accepts(model)
        tested.append({"fraction": fraction, **details})
        if accepted:
            # 부분 업데이트도 해당 gradient의 Adam moment는 유지한다. 숨겨진 재학습은 없다.
            return {"accepted_fraction": fraction, "rolled_back_optimizer": False, "checks": tested}
    model.load_state_dict(before_weights)
    optimizer.load_state_dict(before_optimizer)
    return {"accepted_fraction": 0.0, "rolled_back_optimizer": True, "checks": tested}


def _mode_loss(model, x, target_logits, deterministic: bool, temperature: float):
    """결정론 보존 forward에서만 dropout을 끄고 autograd와 모듈별 모드를 유지한다."""
    from hwr_boundary_distillation_v1 import boundary_kl_loss
    modes = [(module, module.training) for module in model.modules()]
    try:
        if deterministic:
            model.eval()
        z = model.math_head(model.encode(x))
    finally:
        for module, training in modes:
            module.training = training
    return boundary_kl_loss(z, target_logits, temperature, "full")


def _run(out: Path) -> int:
    """같은 64개 배치를 세 조건으로 학습하고 학습 풀의 보존/개선만 점검한다."""
    if out.exists():
        raise FileExistsError("refusing to overwrite/restart an existing retention probe")
    if _guard_commit("before_retention_probe") is None:
        return 78
    import torch
    from torch.nn import functional as F
    from hwr_boundary_distillation_v1 import boundary_kl_loss
    from run_hwr_affine_distillation_experiment_v1 import _load_teacher, _predict_logits
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    parent_plan_path = PARENT / "frozen_plan.json"
    old = json.loads(parent_plan_path.read_text(encoding="utf-8"))
    result = json.loads((PARENT / "probe_result.json").read_text(encoding="utf-8"))
    if result["status"] != "completed" or result["plan_sha256"] != _sha(parent_plan_path):
        raise ValueError("parent experiment is not sealed/terminal")
    if _sha(DEFAULT_CHECKPOINT) != old["checkpoint_sha256"]:
        raise ValueError("canonical checkpoint changed")
    source_report = json.loads((SOURCE / "pendigits_source_audit.json").read_text(encoding="utf-8"))
    if _sha(SOURCE / "pendigits_source_audit.json") != old["source_report_sha256"]:
        raise ValueError("parent source changed")
    arrays = {}
    # held_indices, held_logits, 공식 test는 열지 않고 학습 배열만 허용한다.
    for name in ("train_indices", "replay_indices", "replay_features"):
        info = old["artifacts"][name]
        if _sha(PARENT / info["file"]) != info["sha256"]:
            raise ValueError("parent training array changed")
        arrays[name] = np.load(PARENT / info["file"], allow_pickle=False)
    for name in ("train_features_y_up", "train_digit_labels"):
        if _sha(SOURCE / f"{name}.npy") != source_report["artifacts"][name]["sha256"]:
            raise ValueError("digit source array changed")
    for name, digest in old["replay_train_array_hashes"].items():
        if _sha(DEFAULT_DATA_DIR / f"train_{name}.npy") != digest:
            raise ValueError("old real training cache changed")
    train = arrays["train_indices"]
    x = np.array(np.load(SOURCE / "train_features_y_up.npy", mmap_mode="r", allow_pickle=False)[train], copy=True)
    digits = np.load(SOURCE / "train_digit_labels.npy", mmap_mode="r", allow_pickle=False)[train]
    replay_x = arrays["replay_features"]
    replay_y = np.array(np.load(DEFAULT_DATA_DIR / "train_labels.npy", mmap_mode="r", allow_pickle=False)[arrays["replay_indices"]], copy=True)
    device = torch.device("cpu")
    teacher, labels, _ = _load_teacher(DEFAULT_CHECKPOINT, device)
    teacher_source = _predict_logits(teacher, x, device, 32)
    teacher_replay = _predict_logits(teacher, replay_x, device, 32)
    protected = np.flatnonzero(teacher_replay.argmax(1) == replay_y)
    if len(protected) != 831:
        raise ValueError("parent protected training anchor count differs")
    protected_x = replay_x[protected]
    protected_y = replay_y[protected]
    rng = np.random.default_rng(old["seed"] + 11)
    batch_ids = rng.integers(len(train), size=(64, 32))
    replay_ids = rng.integers(len(replay_x), size=(64, 16))
    probe_ids = np.random.default_rng(old["seed"] + 719).choice(len(train), 1024, replace=False)
    source_y = np.array([labels.index(str(d)) for d in digits], dtype=np.int64)
    tx, ty = torch.from_numpy(x), torch.from_numpy(source_y)
    rx = torch.from_numpy(replay_x)
    ts, tr = torch.from_numpy(teacher_source), torch.from_numpy(teacher_replay)
    out.mkdir(parents=True)
    plan = {"schema": "aiflow-math-retention-probe-plan/v1", "steps": 64, "seed": old["seed"], "learning_rate": old["learning_rate"],
            "parent_plan_sha256": _sha(parent_plan_path), "script_sha256": _sha(Path(__file__)), "checkpoint_sha256": old["checkpoint_sha256"],
            "dependencies": {name: _sha(ROOT / "scripts" / name) for name in ("hwr_boundary_distillation_v1.py", "run_hwr_pendigits_tube_probe_v1.py", "run_hwr_affine_distillation_experiment_v1.py", "train_character_classifier_v1.py", "evaluate_48hz_prefix_v1.py")},
            "arms": {"stochastic_replay": "parent original-consistency objective, byte-reproducibility control",
                     "deterministic_replay": "same objective; old replay KL forward dropout off, autograd on",
                     "deterministic_guarded": "deterministic replay plus backtracking veto if any of 831 canonical-correct TRAIN anchors loses global Top-1"},
            "optimizer_guard_fractions": list(FRACTIONS), "full_rejection": "restore all model weights AND optimizer moments/counters",
            "fractional_acceptance": "interpolated floating weights; proposed Adam moments retained",
            "source_train_probe_rows": len(probe_ids), "old_train_replay_rows": len(replay_x), "protected_train_anchor_rows": len(protected),
            "held_inputs_forwarded": 0, "official_test_rows_read": 0, "crohme_rows": 0, "human_boundary_labels": 0,
            "model_selection": False, "product_adopted": False,
            "limitations": "TRAINING/integrity only; finite supervised anchors, not whole-domain or independent acceptance nonregression. Training guard is not a mobile inference stage."}
    _write(out / "frozen_plan.json", plan)
    np.save(out / "source_train_probe_indices.npy", train[probe_ids], allow_pickle=False)
    np.save(out / "protected_replay_indices.npy", protected, allow_pickle=False)
    baseline = {"source_train_probe_hits": int((teacher_source[probe_ids].argmax(1) == source_y[probe_ids]).sum()), "old_train_replay_hits": len(protected)}
    reports = {}
    for arm in plan["arms"]:
        if _guard_commit(f"before_retention_arm_{arm}") is None:
            return 78
        model = copy.deepcopy(teacher)
        optimizer = torch.optim.AdamW(model.parameters(), lr=old["learning_rate"], weight_decay=1.0e-4)
        history = []
        for step in range(64):
            model.train()
            torch.manual_seed(old["seed"] + 1000 + step)
            bi, ri = torch.from_numpy(batch_ids[step]), torch.from_numpy(replay_ids[step])
            optimizer.zero_grad(set_to_none=True)
            deterministic = arm != "stochastic_replay"
            # CE의 배치 모양과 dropout 난수 흐름도 세 조건에서 동일하게 유지한다.
            z = model.math_head(model.encode(torch.cat((tx[bi], rx[ri]))))
            hard = F.cross_entropy(z[:32], ty[bi])
            if deterministic:
                retention = _mode_loss(model, rx[ri], tr[ri], True, 2.0)
            else:
                retention = boundary_kl_loss(z[32:], tr[ri], 2.0, "full")
            consistency = _mode_loss(model, tx[bi], ts[bi], True, 1.0)
            loss = 0.9 * (0.7 * hard + 0.3 * retention) + 0.1 * consistency
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite loss")
            loss.backward()
            gradient_layers = {n: float(v.grad.norm()) for n, v in model.named_parameters() if v.grad is not None} if step + 1 in (1, 16, 32, 64) else None
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad):
                raise FloatingPointError("nonfinite gradient")
            guarded = arm == "deterministic_guarded"
            before_weights = {n: v.detach().clone() for n, v in model.state_dict().items()} if guarded else None
            before_optimizer = copy.deepcopy(optimizer.state_dict()) if guarded else None
            optimizer.step()
            guard = None
            if guarded:
                def accepts(candidate):
                    """831개 학습 anchor 각각의 정답을 전체 372개 출력과 경쟁시킨다."""
                    pred = _predict_logits(candidate, protected_x, device, 32).argmax(1)
                    lost = int((pred != protected_y).sum())
                    return lost == 0, {"lost_protected_top1": lost}
                guard = _guard_step(model, optimizer, before_weights, before_optimizer, accepts)
            record = {"step": step + 1, "ce_real": float(hard.detach()), "kl_replay": float(retention.detach()), "kl_consistency": float(consistency.detach()),
                      "total_loss": float(loss.detach()), "gradient_l2_before_clip": float(grad), "guard": guard}
            if gradient_layers is not None:
                record["layer_gradient_l2_before_clip"] = gradient_layers
            history.append(record)
            if (step + 1) % 8 == 0:
                print(json.dumps({"event": "retention_probe_step", "arm": arm, **{k: record[k] for k in ("step", "ce_real", "kl_replay", "total_loss")},
                                  "accepted_fraction": None if guard is None else guard["accepted_fraction"]}), flush=True)
        model.eval()
        probe = _predict_logits(model, x[probe_ids], device, 32)
        replay = _predict_logits(model, replay_x, device, 32)
        correct_before = teacher_replay.argmax(1) == replay_y
        correct_after = replay.argmax(1) == replay_y
        stats = {"source_train_probe_hits": int((probe.argmax(1) == source_y[probe_ids]).sum()), "old_train_replay_hits": int(correct_after.sum()),
                 "old_train_rescued": int((~correct_before & correct_after).sum()), "old_train_regressed": int((correct_before & ~correct_after).sum()),
                 "old_train_kl_t1": float(boundary_kl_loss(torch.from_numpy(replay), tr, 1.0, "full"))}
        if guarded and stats["old_train_regressed"] != 0:
            raise AssertionError("guarded final state lost a protected training anchor")
        checkpoint = out / f"{arm}.pt"
        torch.save({"state_dict": model.state_dict(), "math_labels": labels, "auxiliary_labels": [], "report": {"input_contract": {"observed_channel_mode": "uniform-time"}, "product_adopted": False}}, checkpoint)
        np.save(out / f"{arm}_train_probe_logits.npy", probe, allow_pickle=False)
        np.save(out / f"{arm}_train_replay_logits.npy", replay, allow_pickle=False)
        control_identical = None
        if arm == "stochastic_replay":
            parent = torch.load(PARENT / "original_consistency.pt", map_location="cpu", weights_only=False)["state_dict"]
            control_identical = all(torch.equal(model.state_dict()[n], v) for n, v in parent.items())
            if not control_identical:
                raise AssertionError("stochastic control did not reproduce frozen parent weights")
        reports[arm] = {"metrics": stats, "history": history, "checkpoint_sha256": _sha(checkpoint), "parent_control_weights_bitwise_identical": control_identical}
        _write(out / f"{arm}_completed.json", reports[arm])
        print(json.dumps({"event": "retention_arm_complete", "arm": arm, **stats}), flush=True)
    if _sha(DEFAULT_CHECKPOINT) != old["checkpoint_sha256"]:
        raise AssertionError("canonical checkpoint changed")
    final = {"schema": "aiflow-math-retention-probe-result/v1", "status": "completed", "frozen_plan_sha256": _sha(out / "frozen_plan.json"),
             "baseline_training_only": baseline, "arms": reports, "canonical_checkpoint_unchanged": True, "held_inputs_forwarded": 0,
             "official_test_rows_read": 0, "crohme_rows": 0, "product_adopted": False, "eligible_for_product_selection": False, "limitations": plan["limitations"]}
    _write(out / "retention_result.json", final)
    print(json.dumps({"event": "retention_probe_complete", "baseline": baseline, "arms": {n: v["metrics"] for n, v in reports.items()}}), flush=True)
    return 0


def _self_test() -> int:
    """완전 거부 때 Adam moment/step과 가중치가 복원되고 축소 수락이 동작하는지 검사한다."""
    import torch
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    optimizer.zero_grad(); model(torch.ones(1, 1)).sum().backward(); optimizer.step()
    before = {n: v.detach().clone() for n, v in model.state_dict().items()}
    opt = copy.deepcopy(optimizer.state_dict())
    optimizer.zero_grad(); model(torch.ones(1, 1)).sum().backward(); optimizer.step()
    result = _guard_step(model, optimizer, before, opt, lambda _: (False, {"lost_protected_top1": 1}))
    assert result["rolled_back_optimizer"] and result["accepted_fraction"] == 0.0
    assert all(torch.equal(model.state_dict()[n], v) for n, v in before.items())
    for pid, state in opt["state"].items():
        assert all(torch.equal(v, optimizer.state_dict()["state"][pid][n]) if isinstance(v, torch.Tensor) else v == optimizer.state_dict()["state"][pid][n] for n, v in state.items())
    optimizer.zero_grad(); model(torch.ones(1, 1)).sum().backward(); optimizer.step()
    proposed = model.weight.detach().clone()
    result = _guard_step(model, optimizer, before, opt, lambda m: (bool(torch.max(torch.abs(m.weight - before["weight"])) < 0.04), {}))
    assert result["accepted_fraction"] == 0.25
    assert torch.equal(model.weight, before["weight"] + 0.25 * (proposed - before["weight"]))
    print(json.dumps({"self_test": "pass", "full_rejection_restores_weights_and_adam": True, "fractional_acceptance": 0.25}))
    return 0


def main() -> int:
    """별도 비파괴 결과 폴더에서 학습 무결성 실험 또는 가드 단위 테스트를 실행한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("run", "self-test"), required=True)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    return _self_test() if args.mode == "self-test" else _run(args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
