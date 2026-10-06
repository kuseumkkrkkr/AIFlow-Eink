"""전체 업데이트 축소 대신 위반한 TRAIN 정답 경계의 방향 성분을 교정한다.

1차 margin 보정 뒤 실제 372-way 판정을 다시 검사한다. 보정 실패 시 Adam까지
복구하며, 유한 학습 anchor의 보호만 검증한다. test/보류 그룹은 평가하지 않는다.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from audit_hwr_probability_boundary_tube_v1 import DEFAULT_CHECKPOINT, DEFAULT_DATA_DIR, ROOT, _guard_commit
from run_hwr_pendigits_tube_probe_v1 import SOURCE, OUTPUT as PARENT, _sha, _write
from run_hwr_pendigits_retention_probe_v1 import OUTPUT as RETENTION, _mode_loss

OUTPUT = ROOT / "artifacts/hwr_pendigits_directional_guard_20261005"
MAX_ROUNDS = 8
MAX_CONSTRAINTS_PER_ROUND = 8


def _directional_step(model, optimizer, before, before_optimizer, anchor_x, anchor_y, predict, max_rounds=MAX_ROUNDS):
    """실제 위반 margin을 가중치 방향의 국소 미분으로 보정하고 매번 전 anchor를 확인한다."""
    import torch
    named = list(model.named_parameters())
    proposal = {n: v.detach().clone() for n, v in model.state_dict().items()}
    delta = {n: proposal[n].double() - before[n].double() for n, _ in named}
    raw_norm = sum(float(d.square().sum()) for d in delta.values()) ** 0.5
    rounds, corrections = [], []
    for round_id in range(max_rounds + 1):
        with torch.no_grad():
            for n, v in model.state_dict().items():
                v.copy_((before[n].double() + delta[n]).to(v.dtype) if n in delta else proposal[n])
        logits = predict(model)
        lost = np.flatnonzero(logits.argmax(1) != anchor_y)
        norm = sum(float(d.square().sum()) for d in delta.values()) ** 0.5
        rounds.append({"round": round_id, "lost_protected_top1": len(lost), "update_l2": norm})
        if not np.isfinite(norm) or norm > 2.0 * max(raw_norm, 1.0e-12):
            break
        if not len(lost):
            return {"accepted": True, "optimizer_rolled_back": False, "raw_update_l2": raw_norm, "final_update_l2": norm, "rounds": rounds, "corrections": corrections}
        if round_id == max_rounds:
            break
        # 동일 입력의 champion 미분으로 보정한다. autograd를 끄지 않는다.
        model.load_state_dict(before)
        model.eval()
        round_delta = {n: d.clone() for n, d in delta.items()}
        for row in lost[:MAX_CONSTRAINTS_PER_ROUND]:
            rival = int(logits[row].argmax())
            truth = int(anchor_y[row])
            x = torch.from_numpy(np.array(anchor_x[row:row + 1], dtype=np.float32, copy=True))
            z = model.math_head(model.encode(x))
            margin = z[0, truth] - z[0, rival]
            gradients = torch.autograd.grad(margin, [p for _, p in named], allow_unused=True)
            gradient = {n: g.detach().double() for (n, _), g in zip(named, gradients) if g is not None}
            denominator = sum(float(g.square().sum()) for g in gradient.values())
            current_dot = sum(float((delta[n] * g).sum()) for n, g in gradient.items())
            round_dot = sum(float((round_delta[n] * g).sum()) for n, g in gradient.items())
            # FP32 경계의 작은 여유를 두되 champion margin의 절반을 넘지 않는다.
            floor = max(0.0, min(1.0e-4, float(margin.detach()) * 0.5))
            actual_margin = float(logits[row, truth] - logits[row, rival])
            corrected_margin_estimate = actual_margin + current_dot - round_dot
            linear_margin = float(margin.detach()) + current_dot
            increase = max(floor - corrected_margin_estimate, floor - linear_margin, 0.0)
            if not np.isfinite(denominator) or denominator <= 1.0e-20:
                continue
            coefficient = increase / denominator
            for n, g in gradient.items():
                delta[n].add_(g, alpha=coefficient)
            corrections.append({"round": round_id, "protected_row": int(row), "truth_id": truth, "rival_id": rival,
                                "champion_margin": float(margin.detach()), "candidate_margin": actual_margin,
                                "linear_margin_before": linear_margin, "required_increase": increase, "gradient_l2_squared": denominator})
    model.load_state_dict(before)
    optimizer.load_state_dict(before_optimizer)
    return {"accepted": False, "optimizer_rolled_back": True, "raw_update_l2": raw_norm, "final_update_l2": 0.0, "rounds": rounds, "corrections": corrections}


def _run(out: Path) -> int:
    """이전 64-step과 같은 목표/배치로 방향 보정만 바꿔 학습 무결성을 비교한다."""
    if out.exists():
        raise FileExistsError("refusing to overwrite/restart an existing directional probe")
    if _guard_commit("before_directional_probe") is None:
        return 78
    import torch
    from torch.nn import functional as F
    from hwr_boundary_distillation_v1 import boundary_kl_loss
    from run_hwr_affine_distillation_experiment_v1 import _load_teacher, _predict_logits
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    previous = json.loads((RETENTION / "retention_result.json").read_text(encoding="utf-8"))
    previous_plan = json.loads((RETENTION / "frozen_plan.json").read_text(encoding="utf-8"))
    old = json.loads((PARENT / "frozen_plan.json").read_text(encoding="utf-8"))
    if previous["status"] != "completed" or _sha(RETENTION / "frozen_plan.json") != previous["frozen_plan_sha256"]:
        raise ValueError("previous comparison is not sealed/terminal")
    if _sha(PARENT / "frozen_plan.json") != previous_plan["parent_plan_sha256"] or _sha(DEFAULT_CHECKPOINT) != old["checkpoint_sha256"]:
        raise ValueError("source plan/canonical checkpoint changed")
    if _sha(ROOT / "scripts/run_hwr_pendigits_retention_probe_v1.py") != previous_plan["script_sha256"]:
        raise ValueError("sealed retention runner changed")
    for n, digest in previous_plan["dependencies"].items():
        if _sha(ROOT / "scripts" / n) != digest:
            raise ValueError("previous dependency changed")
    source_report = json.loads((SOURCE / "pendigits_source_audit.json").read_text(encoding="utf-8"))
    if _sha(SOURCE / "pendigits_source_audit.json") != old["source_report_sha256"]:
        raise ValueError("digit source changed")
    arrays = {}
    for n in ("train_indices", "replay_indices", "replay_features"):
        info = old["artifacts"][n]
        if _sha(PARENT / info["file"]) != info["sha256"]:
            raise ValueError("TRAIN array changed")
        arrays[n] = np.load(PARENT / info["file"], allow_pickle=False)
    for n in ("train_features_y_up", "train_digit_labels"):
        if _sha(SOURCE / f"{n}.npy") != source_report["artifacts"][n]["sha256"]:
            raise ValueError("digit source array changed")
    for n, digest in old["replay_train_array_hashes"].items():
        if _sha(DEFAULT_DATA_DIR / f"train_{n}.npy") != digest:
            raise ValueError("old real TRAIN cache changed")
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
        raise ValueError("protected TRAIN anchor count differs")
    probe_source_ids = np.load(RETENTION / "source_train_probe_indices.npy", allow_pickle=False)
    expected_probe = np.random.default_rng(old["seed"] + 719).choice(len(train), 1024, replace=False)
    if not np.array_equal(probe_source_ids, train[expected_probe]):
        raise ValueError("shared source TRAIN probe rows changed")
    lookup = {int(row): i for i, row in enumerate(train)}
    probe = np.array([lookup[int(row)] for row in probe_source_ids], dtype=np.int64)
    source_y = np.array([labels.index(str(d)) for d in digits], dtype=np.int64)
    if int((teacher_source[probe].argmax(1) == source_y[probe]).sum()) != previous["baseline_training_only"]["source_train_probe_hits"]:
        raise ValueError("source TRAIN baseline differs")
    rng = np.random.default_rng(old["seed"] + 11)
    batch_ids = rng.integers(len(train), size=(64, 32))
    replay_ids = rng.integers(len(replay_x), size=(64, 16))
    out.mkdir(parents=True)
    plan = {"schema": "aiflow-directional-guard-plan/v1", "seed": old["seed"], "steps": 64, "learning_rate": old["learning_rate"],
            "script_sha256": _sha(Path(__file__)), "retention_result_sha256": _sha(RETENTION / "retention_result.json"), "parent_plan_sha256": _sha(PARENT / "frozen_plan.json"),
            "checkpoint_sha256": old["checkpoint_sha256"], "max_correction_rounds": MAX_ROUNDS, "max_constraints_per_round": MAX_CONSTRAINTS_PER_ROUND,
            "update_norm_cap_multiple": 2.0, "margin_floor": "max(0,min(1e-4,champion_margin/2))", "protected_train_rows": len(protected),
            "objective": "same deterministic-replay objective and 64 batches as sealed retention probe",
            "correction": "cyclic first-order correction of violated logit margins; recheck all 831 TRAIN anchors after every round",
            "acceptance": "all protected anchors keep global 372-way Top-1; proposed Adam moments retained",
            "rejection": "restore model weights AND optimizer moments/counters; no weakened anchor set or scalar backtracking",
            "dependencies": {n: _sha(ROOT / "scripts" / n) for n in ("run_hwr_pendigits_retention_probe_v1.py", "run_hwr_pendigits_tube_probe_v1.py", "hwr_boundary_distillation_v1.py", "run_hwr_affine_distillation_experiment_v1.py", "train_character_classifier_v1.py", "evaluate_48hz_prefix_v1.py")},
            "source_train_probe_indices_sha256": _sha(RETENTION / "source_train_probe_indices.npy"), "held_inputs_forwarded": 0, "official_test_rows_read": 0,
            "crohme_rows": 0, "human_boundary_labels": 0, "product_adopted": False,
            "limitations": "TRAINING/integrity only; local linear approximations are not a global guarantee. Not independent recognition accuracy, human-boundary learning, or product promotion. Inference architecture unchanged."}
    _write(out / "frozen_plan.json", plan)
    tx, ty, rx = torch.from_numpy(x), torch.from_numpy(source_y), torch.from_numpy(replay_x)
    ts, tr = torch.from_numpy(teacher_source), torch.from_numpy(teacher_replay)
    model = copy.deepcopy(teacher)
    optimizer = torch.optim.AdamW(model.parameters(), lr=old["learning_rate"], weight_decay=1.0e-4)
    history = []
    for step in range(64):
        model.train(); torch.manual_seed(old["seed"] + 1000 + step)
        bi, ri = torch.from_numpy(batch_ids[step]), torch.from_numpy(replay_ids[step])
        optimizer.zero_grad(set_to_none=True)
        z = model.math_head(model.encode(torch.cat((tx[bi], rx[ri]))))
        ce = F.cross_entropy(z[:32], ty[bi])
        retention = _mode_loss(model, rx[ri], tr[ri], True, 2.0)
        consistency = _mode_loss(model, tx[bi], ts[bi], True, 1.0)
        loss = 0.9 * (0.7 * ce + 0.3 * retention) + 0.1 * consistency
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite loss")
        loss.backward()
        layers = {n: float(v.grad.norm()) for n, v in model.named_parameters() if v.grad is not None} if step + 1 in (1, 16, 32, 64) else None
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(norm):
            raise FloatingPointError("nonfinite gradient")
        before = {n: v.detach().clone() for n, v in model.state_dict().items()}
        before_opt = copy.deepcopy(optimizer.state_dict())
        optimizer.step()
        guard = _directional_step(model, optimizer, before, before_opt, replay_x[protected], replay_y[protected],
                                  lambda candidate: _predict_logits(candidate, replay_x[protected], device, 32))
        record = {"step": step + 1, "ce_real": float(ce.detach()), "kl_replay": float(retention.detach()), "kl_consistency": float(consistency.detach()),
                  "total_loss": float(loss.detach()), "gradient_l2_before_clip": float(norm), "guard": guard}
        if layers is not None:
            record["layer_gradient_l2_before_clip"] = layers
        history.append(record)
        if (step + 1) % 8 == 0:
            print(json.dumps({"event": "directional_guard_step", "step": step + 1, "accepted": guard["accepted"], "corrections": len(guard["corrections"]),
                              "checks": len(guard["rounds"]), "ce_real": record["ce_real"]}), flush=True)
    probe_logits = _predict_logits(model, x[probe], device, 32)
    replay_logits = _predict_logits(model, replay_x, device, 32)
    after = replay_logits.argmax(1) == replay_y
    before = teacher_replay.argmax(1) == replay_y
    metrics = {"source_train_probe_hits": int((probe_logits.argmax(1) == source_y[probe]).sum()), "old_train_replay_hits": int(after.sum()),
               "old_train_rescued": int((~before & after).sum()), "old_train_regressed": int((before & ~after).sum()),
               "old_train_kl_t1": float(boundary_kl_loss(torch.from_numpy(replay_logits), tr, 1.0, "full"))}
    if metrics["old_train_regressed"] != 0:
        raise AssertionError("directional guard lost protected TRAIN labels")
    checkpoint = out / "directional_guard.pt"
    torch.save({"state_dict": model.state_dict(), "math_labels": labels, "auxiliary_labels": [], "report": {"input_contract": {"observed_channel_mode": "uniform-time"}, "product_adopted": False}}, checkpoint)
    np.save(out / "train_probe_logits.npy", probe_logits, allow_pickle=False)
    np.save(out / "train_replay_logits.npy", replay_logits, allow_pickle=False)
    if _sha(DEFAULT_CHECKPOINT) != old["checkpoint_sha256"]:
        raise AssertionError("canonical checkpoint changed")
    result = {"schema": "aiflow-directional-guard-result/v1", "status": "completed", "frozen_plan_sha256": _sha(out / "frozen_plan.json"), "metrics": metrics,
              "checkpoint_sha256": _sha(checkpoint), "history": history, "accepted_steps": sum(s["guard"]["accepted"] for s in history),
              "rejected_steps": sum(not s["guard"]["accepted"] for s in history), "protected_train_rows": len(protected), "canonical_checkpoint_unchanged": True,
              "held_inputs_forwarded": 0, "official_test_rows_read": 0, "crohme_rows": 0, "human_boundary_labels": 0, "product_adopted": False,
              "eligible_for_product_selection": False, "limitations": plan["limitations"]}
    _write(out / "directional_result.json", result)
    print(json.dumps({"event": "directional_guard_complete", "metrics": metrics, "accepted_steps": result["accepted_steps"], "rejected_steps": result["rejected_steps"]}), flush=True)
    return 0


def _self_test() -> int:
    """선형 toy 모델에서 위반 성분만 보정하고, 실패 시 Adam까지 정확히 복구함을 검사한다."""
    import torch
    class Toy(torch.nn.Module):
        """테스트 전용 372-way 선형 출력으로 margin 보정을 정확히 확인한다."""
        def __init__(self):
            super().__init__(); self.math_head = torch.nn.Linear(5, 372)
            with torch.no_grad():
                self.math_head.weight.zero_(); self.math_head.bias.zero_(); self.math_head.bias[0] = 1
        def encode(self, x):
            """토이 입력의 타점 평균을 다섯 특징으로 반환한다."""
            return x.mean(1)
    x = np.zeros((1, 128, 5), dtype=np.float32); x[:, :, 0] = 1
    y = np.array([0]); model = Toy(); optimizer = torch.optim.AdamW(model.parameters(), lr=2.0)
    before = {n: v.detach().clone() for n, v in model.state_dict().items()}; opt = copy.deepcopy(optimizer.state_dict())
    (-model.math_head(model.encode(torch.from_numpy(x)))[0, 1]).backward(); optimizer.step()
    with torch.no_grad():
        model.math_head.bias[2] += 0.5
    def predict(m):
        """토이 모델의 전체 372개 점수를 판정용으로 반환한다."""
        with torch.no_grad():
            return m.math_head(m.encode(torch.from_numpy(x))).numpy()
    result = _directional_step(model, optimizer, before, opt, x, y, predict)
    assert result["accepted"] and result["corrections"] and predict(model).argmax(1)[0] == 0
    assert model.math_head.bias[2].item() == 0.5
    assert not result["optimizer_rolled_back"]
    before = {n: v.detach().clone() for n, v in model.state_dict().items()}; opt = copy.deepcopy(optimizer.state_dict())
    optimizer.zero_grad(); (-model.math_head(model.encode(torch.from_numpy(x)))[0, 1]).backward(); optimizer.step()
    result = _directional_step(model, optimizer, before, opt, x, y, predict, max_rounds=0)
    assert not result["accepted"] and result["optimizer_rolled_back"]
    assert all(torch.equal(model.state_dict()[n], v) for n, v in before.items())
    for pid, state in opt["state"].items():
        assert all(torch.equal(v, optimizer.state_dict()["state"][pid][n]) if isinstance(v, torch.Tensor) else v == optimizer.state_dict()["state"][pid][n] for n, v in state.items())
    print(json.dumps({"self_test": "pass", "unrelated_update_component_preserved": True, "rollback_weights_and_adam": True}))
    return 0


def main() -> int:
    """별도 결과 폴더에서 방향 보정 시험 또는 단위 테스트를 실행한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("run", "self-test"), required=True)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    return _self_test() if args.mode == "self-test" else _run(args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
