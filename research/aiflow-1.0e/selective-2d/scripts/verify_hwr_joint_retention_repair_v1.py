"""봉인 joint TRAIN 교정의 순위·floor·재현성과 국소 gradient 충돌을 독립 검사한다."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from run_hwr_joint_retention_repair_v1 import OUTPUT, DEFAULT_CHECKPOINT, CANDIDATE, ROOT, _sha, _write


def ranks(logits: np.ndarray, targets: np.ndarray) -> dict:
    """다른 클래스의 독립 정렬로 실제 Top-1/5와 상대 margin을 계산한다."""
    order = np.argsort(-logits, axis=1, kind="stable")
    others = np.array([row[row != target] for row, target in zip(order, targets)])
    rows = np.arange(len(targets))
    result = {}
    for k in (1, 5):
        result[f"top{k}_mask"] = (order[:, :k] == targets[:, None]).any(1)
        result[f"top{k}_rival"] = others[:, k - 1]
        result[f"top{k}_margin"] = logits[rows, targets] - logits[rows, others[:, k - 1]]
    return result


def constraints(old: np.ndarray, parent: np.ndarray, targets: np.ndarray) -> dict:
    """canonical 정답 floor를 보존하며 parent-only 정답을 합친 고정 제약을 재구성한다."""
    a, b = ranks(old, targets), ranks(parent, targets)
    return {f"top{k}_{name}": value for k in (1, 5) for name, value in (
        ("mask", a[f"top{k}_mask"] | b[f"top{k}_mask"]),
        ("floor", .5 * np.where(a[f"top{k}_mask"], a[f"top{k}_margin"], b[f"top{k}_margin"])),
    )}


def measure(logits: np.ndarray, targets: np.ndarray, spec: dict) -> dict:
    """정답 개수와 보호 행의 순위/floor 위반 개수를 반환한다."""
    rank = ranks(logits, targets)
    return {f"top{k}_{name}": int(value.sum()) for k in (1, 5) for name, value in (
        ("hits", rank[f"top{k}_mask"]),
        ("membership_regressions", spec[f"top{k}_mask"] & ~rank[f"top{k}_mask"]),
        ("constraint_violations", spec[f"top{k}_mask"] & (
            ~rank[f"top{k}_mask"] | (rank[f"top{k}_margin"] < spec[f"top{k}_floor"]))),
    )}


def main() -> int:
    """실패도 그대로 재현하고 최대 8개 TRAIN 제약의 전 층 국소 gradient를 기록한다."""
    import torch
    from run_hwr_affine_distillation_experiment_v1 import _load_teacher, _predict_logits
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    out = OUTPUT
    destination = out / "independent_verification.json"
    if destination.exists():
        raise FileExistsError("refusing to replace prior verification")
    result = json.loads((out / "joint_repair_result.json").read_text(encoding="utf-8"))
    plan = json.loads((out / "frozen_plan.json").read_text(encoding="utf-8"))
    assert _sha(out / "frozen_plan.json") == result["frozen_plan_sha256"]
    assert _sha(ROOT / "scripts/run_hwr_joint_retention_repair_v1.py") == plan["script_sha256"]
    assert _sha(ROOT / "scripts/repair_hwr_fixed_rank_margins_v1.py") == plan["shared_repair_sha256"]
    assert _sha(DEFAULT_CHECKPOINT) == plan["canonical_checkpoint_sha256"]
    assert _sha(CANDIDATE / "directional_guard.pt") == plan["parent_checkpoint_sha256"]
    checkpoint = out / result["checkpoint_file"]
    assert _sha(checkpoint) == result["checkpoint_sha256"]
    load = lambda name: np.load(out / f"{name}.npy", allow_pickle=False)
    x, y, z = load("joint_features"), load("joint_labels"), load("joint_repaired_logits")
    assert x.shape == (2048, 128, 5) and y.shape == (2048,) and z.shape == (2048, 372)
    old_x, source_x = x[:1024], x[1024:]
    old_y, source_y = y[:1024], y[1024:]
    assert len(np.unique(old_y)) == 371 and len(np.unique(source_y)) == 10
    device = torch.device("cpu")
    canonical, labels, _ = _load_teacher(DEFAULT_CHECKPOINT, device)
    parent, parent_labels, _ = _load_teacher(CANDIDATE / "directional_guard.pt", device)
    model, final_labels, _ = _load_teacher(checkpoint, device)
    assert labels == parent_labels == final_labels and len(labels) == 372
    canonical_z = _predict_logits(canonical, x, device, 32)
    parent_z = _predict_logits(parent, x, device, 32)
    assert np.array_equal(canonical_z[:1024], load("canonical_old_logits"))
    assert np.array_equal(canonical_z[1024:], load("canonical_source_logits"))
    assert np.array_equal(parent_z[:1024], load("parent_old_logits"))
    assert np.array_equal(parent_z[1024:], load("parent_source_logits"))
    old_spec = constraints(canonical_z[:1024], parent_z[:1024], old_y)
    source_rank = ranks(parent_z[1024:], source_y)
    source_spec = {f"top{k}_{name}": source_rank[f"top{k}_{source}"] * factor
                   for k in (1, 5) for name, source, factor in (("mask", "mask", 1), ("floor", "margin", .5))}
    source_spec = {name: value.astype(bool) if name.endswith("mask") else value for name, value in source_spec.items()}
    spec = {name: np.concatenate((old_spec[name], source_spec[name])) for name in old_spec}
    for name, value in spec.items():
        assert np.array_equal(value, load(f"reference_{name}"))
    metrics = {"old_math": measure(z[:1024], old_y, old_spec), "source_digits": measure(z[1024:], source_y, source_spec)}
    for domain, domain_spec in (("old_math", old_spec), ("source_digits", source_spec)):
        for k in (1, 5):
            assert int(domain_spec[f"top{k}_mask"].sum()) == plan["protected_counts"][domain][f"top{k}"]
    for k in (1, 5):
        assert metrics["old_math"][f"top{k}_hits"] == result["metrics"][f"old_train_top{k}_hits"]
        assert metrics["source_digits"][f"top{k}_hits"] == result["metrics"][f"source_train_top{k}_hits"]
        for name in ("constraint_violations", "membership_regressions"):
            assert sum(d[f"top{k}_{name}"] for d in metrics.values()) == result["metrics"][f"top{k}_{name}"]
    batch_checks = []
    for batch_size in (1, 16, 32):
        actual = _predict_logits(model, x, device, batch_size)
        batch_checks.append(dict(batch_size=batch_size, bit_exact=bool(np.array_equal(actual, z)),
            max_logit_delta=float(np.max(np.abs(actual-z))), old_math=measure(actual[:1024], old_y, old_spec),
            source_digits=measure(actual[1024:], source_y, source_spec)))
    assert batch_checks[-1]["bit_exact"]
    rank = ranks(z, y)
    selected = []
    for lower, upper, domain in ((0, 1024, "old_math"), (1024, 2048, "source_digits")):
        failures = [i for i in range(lower, upper) if spec["top1_mask"][i] and
                    rank["top1_margin"][i] < spec["top1_floor"][i]]
        selected.extend((i, domain) for i in sorted(failures, key=lambda i: float(spec["top1_floor"][i]-rank["top1_margin"][i]), reverse=True)[:4])
    named = list(model.named_parameters())
    gradients, rows = [], []
    model.eval()
    for row, domain in selected:
        logits = model.math_head(model.encode(torch.from_numpy(x[row:row+1].copy())))
        rival = int(rank["top1_rival"][row])
        gradient = torch.autograd.grad(logits[0, int(y[row])]-logits[0, rival], [p for _, p in named], allow_unused=True)
        values = [torch.zeros_like(p, dtype=torch.float64).reshape(-1) if g is None else g.detach().double().reshape(-1)
                  for (_, p), g in zip(named, gradient)]
        gradients.append(values)
        rows.append(dict(row=row, domain=domain, truth=labels[int(y[row])], rival=labels[rival],
            margin=float(rank["top1_margin"][row]), floor=float(spec["top1_floor"][row]),
            gradient_layers=[dict(name=n, l2_fp64=float(g.norm())) for (n, _), g in zip(named, values)]))
    cosine = {}
    for scope, mask in (("all_parameters", [True]*len(named)), ("math_head", [n.startswith("math_head.") for n, _ in named]),
                        ("encoder", [not n.startswith("math_head.") for n, _ in named])):
        vectors = [torch.cat([g for g, keep in zip(values, mask) if keep]) for values in gradients]
        cosine[scope] = [[float(torch.dot(a, b) / (a.norm()*b.norm()).clamp_min(1e-30)) for b in vectors] for a in vectors]
    delta_layers = [dict(name=n, l2_fp64=float((p.detach().double()-dict(canonical.named_parameters())[n].detach().double()).norm())) for n, p in named]
    assert _sha(checkpoint) == result["checkpoint_sha256"]
    verification = dict(schema="aiflow-joint-retention-independent-verification/v1", status="reproduced",
        result_sha256=_sha(out / "joint_repair_result.json"), checkpoint_sha256=_sha(checkpoint),
        immutable_constraints_rebuilt_bit_exact=True, canonical_and_parent_logits_reloaded_bit_exact=True,
        protected_counts=plan["protected_counts"], domain_metrics=metrics, batch_checks=batch_checks,
        update_l2_fp64=sum(v["l2_fp64"]**2 for v in delta_layers)**.5, update_layers=delta_layers,
        local_conflict_rows=rows, local_gradient_cosine=cosine, parameter_tensors=len(named),
        local_gradient_limit="Finite selected TRAIN constraints at one failed candidate. Negative cosine is local interference, not proof of global infeasibility or architecture insufficiency.",
        parameters_changed_during_verification=False, held_inputs_forwarded=0, crohme_rows=0,
        human_boundary_labels=0, product_adopted=False, eligible_for_independent_performance_claim=False)
    _write(destination, verification)
    print(json.dumps(dict(status=verification["status"], gate=result["status"], metrics=metrics,
        local_conflict_rows=len(rows), layers=len(named)), ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
