"""동시 TRAIN 교정의 불변 제약·수치 인증·체크포인트 재현을 별도로 검증한다."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from run_hwr_simultaneous_retention_repair_v1 import ROOT, PREVIOUS, DEFAULT_CHECKPOINT, CANDIDATE, _sha, _write
from verify_hwr_joint_retention_repair_v1 import measure, ranks, constraints


def run(out: Path) -> int:
    """저장한 실제 모델과 별도 rank 구현으로 모든 보호 행·각 선형 풀이를 검사한다."""
    import torch
    from run_hwr_affine_distillation_experiment_v1 import _load_teacher, _predict_logits
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    destination = out / "independent_verification.json"
    if destination.exists():
        raise FileExistsError("refusing to replace verification")
    plan = json.loads((out / "frozen_plan.json").read_text(encoding="utf-8"))
    result = json.loads((out / "simultaneous_result.json").read_text(encoding="utf-8"))
    assert _sha(out / "frozen_plan.json") == result["frozen_plan_sha256"]
    assert _sha(ROOT / "scripts/run_hwr_simultaneous_retention_repair_v1.py") == plan["script_sha256"]
    if "entrypoint_sha256" in plan:
        assert _sha(ROOT / "scripts/run_hwr_simultaneous_retention_repair_polished_v2.py") == plan["entrypoint_sha256"]
    for name, digest in plan["dependency_hashes"].items():
        assert _sha(ROOT / "scripts" / name) == digest
    for name, digest in plan["reference_hashes"].items():
        assert _sha(PREVIOUS / f"{name}.npy") == digest
    assert _sha(DEFAULT_CHECKPOINT) == plan["canonical_checkpoint_sha256"]
    assert _sha(CANDIDATE / "directional_guard.pt") == plan["parent_checkpoint_sha256"]
    checkpoint = out / result["checkpoint_file"]
    assert _sha(checkpoint) == result["checkpoint_sha256"]
    load = lambda name: np.load(PREVIOUS / f"{name}.npy", allow_pickle=False)
    x, y = load("joint_features"), load("joint_labels")
    z = np.load(out / "simultaneous_logits.npy", allow_pickle=False)
    old = constraints(load("canonical_old_logits"), load("parent_old_logits"), y[:1024])
    source = ranks(load("parent_source_logits"), y[1024:])
    source = {f"top{k}_{name}": source[f"top{k}_{original}"] * factor
              for k in (1, 5) for name, original, factor in (("mask", "mask", 1), ("floor", "margin", .5))}
    source = {n: v.astype(bool) if n.endswith("mask") else v for n, v in source.items()}
    spec = {n: np.concatenate((old[n], source[n])) for n in old}
    for name, value in spec.items():
        assert np.array_equal(value, load(f"reference_{name}"))
    certificates = []
    for history in result["history"]:
        if "correction" not in history:
            continue
        correction = history["correction"]
        saved = np.load(out / f'linearized_round_{history["round"]:03d}.npz', allow_pickle=False)
        g, b, coefficients = saved["gram"], saved["rhs"], saved["coefficients"]
        residual = g @ coefficients - b
        stationary = np.where(coefficients > 1e-10, np.abs(residual), np.maximum(-residual, 0.))
        certified = bool(np.isfinite(coefficients).all() and (coefficients >= 0).all()
                         and residual.min() >= -1e-9 and stationary.max() <= 1e-8)
        assert certified == correction["certified"]
        assert len(coefficients) <= plan["max_constraints_per_round"]
        assert np.allclose(g, g.T, atol=1e-12) and np.linalg.eigvalsh(g).min() >= -1e-10
        assert np.allclose(np.diag(g), 1., atol=1e-12)
        certificates.append(dict(round=history["round"], constraints=len(coefficients), certified=certified,
            min_primal_residual=float(residual.min()), max_stationarity_residual=float(stationary.max())))
    device = torch.device("cpu")
    model, labels, _ = _load_teacher(checkpoint, device)
    canonical, canonical_labels, _ = _load_teacher(DEFAULT_CHECKPOINT, device)
    parent, parent_labels, _ = _load_teacher(CANDIDATE / "directional_guard.pt", device)
    assert labels == canonical_labels == parent_labels and len(labels) == 372
    first_step = None
    if result["history"][0].get("correction", {}).get("certified"):
        # 첫 회차를 원 parent에서만 재생하여 비선형 오차와 미선택 행 간섭을 분리한다.
        first = result["history"][0]["correction"]
        selected = first["selected"]
        gradients = []; pre_margins = []; norms = []
        parent.eval(); parent_named = list(parent.named_parameters())
        for item in selected:
            row, rival = item["row"], item["rival"]
            logits = parent.math_head(parent.encode(torch.from_numpy(x[row:row+1].copy())))
            margin = logits[0, int(y[row])] - logits[0, rival]
            grad = torch.autograd.grad(margin, [p for _, p in parent_named], allow_unused=True)
            flat = torch.cat([torch.zeros_like(p, dtype=torch.float64).reshape(-1) if g is None else g.detach().double().reshape(-1)
                              for (_, p), g in zip(parent_named, grad)])
            norm = float(flat.norm()); norms.append(norm); gradients.append(flat / norm)
            pre_margins.append(float(margin.detach()))
        matrix = torch.stack(gradients)
        saved = np.load(out / "linearized_round_000.npz", allow_pickle=False)
        assert np.array_equal((matrix @ matrix.T).numpy(), saved["gram"])
        step = torch.from_numpy(saved["coefficients"]) @ matrix
        offset = 0
        with torch.no_grad():
            for _, p in parent_named:
                count = p.numel()
                p.copy_((p.detach().double()+step[offset:offset+count].reshape(p.shape)).to(p.dtype)); offset += count
        actual = _predict_logits(parent, x, device, 32)
        actual_metrics = dict(old_math=measure(actual[:1024], y[:1024], old), source_digits=measure(actual[1024:], y[1024:], source))
        assert actual_metrics == result["history"][1]["domains"]
        actual_rank = ranks(actual, y)
        initial_z = np.concatenate((load("parent_old_logits"), load("parent_source_logits")))
        initial_rank = ranks(initial_z, y)
        selected_keys = {(item["row"], item["rank"]) for item in selected}
        observations = []
        estimated = np.array(pre_margins) + np.array(norms)*(saved["gram"] @ saved["coefficients"])
        for index, item in enumerate(selected):
            row, k = item["row"], item["rank"]
            fixed_rival = float(actual[row, int(y[row])]-actual[row, item["rival"]])
            observations.append(dict(row=row, rank=k, truth=labels[int(y[row])], fixed_rival=labels[item["rival"]],
                floor=item["floor"], linearized_margin=float(estimated[index]), actual_fixed_rival_margin=fixed_rival,
                actual_rank_margin=float(actual_rank[f"top{k}_margin"][row]),
                actual_rank_constraint_pass=bool(actual_rank[f"top{k}_mask"][row] and actual_rank[f"top{k}_margin"][row] >= item["floor"])))
        newly_failed = []
        for k in (1, 5):
            protected = spec[f"top{k}_mask"].astype(bool)
            initially_ok = initial_rank[f"top{k}_mask"] & (initial_rank[f"top{k}_margin"] >= spec[f"top{k}_floor"])
            failed = ~actual_rank[f"top{k}_mask"] | (actual_rank[f"top{k}_margin"] < spec[f"top{k}_floor"])
            for row in np.flatnonzero(protected & initially_ok & failed):
                newly_failed.append(dict(row=int(row), rank=k, domain="old_math" if row < 1024 else "source_digits",
                    truth=labels[int(y[row])], rival=labels[int(actual_rank[f"top{k}_rival"][row])],
                    selected=(int(row), k) in selected_keys))
        first_step = dict(replayed_first_metrics_exact=True, gradient_gram_bit_exact=True,
            selected_constraint_observations=observations, newly_failed_previously_safe_constraints=newly_failed)
        # 위 재생에 사용한 parent 사본은 아래 최종 weight 비교에서 원 checkpoint로 다시 읽는다.
        parent, _, _ = _load_teacher(CANDIDATE / "directional_guard.pt", device)
    named = dict(model.named_parameters()); reference = dict(canonical.named_parameters()); proposal = dict(parent.named_parameters())
    layers = [dict(name=name, canonical_delta_l2_fp64=float((p.detach().double()-reference[name].detach().double()).norm()),
                   parent_delta_l2_fp64=float((p.detach().double()-proposal[name].detach().double()).norm())) for name, p in named.items()]
    checks = []
    for batch in (1, 16, 32):
        actual = _predict_logits(model, x, device, batch)
        metrics = dict(old_math=measure(actual[:1024], y[:1024], old), source_digits=measure(actual[1024:], y[1024:], source))
        assert metrics == result["metrics"]
        checks.append(dict(batch_size=batch, metrics=metrics, bit_exact=bool(np.array_equal(actual, z)),
                           max_logit_delta=float(np.max(np.abs(actual-z)))))
    assert checks[-1]["bit_exact"]
    final_counts = {f"top{k}_{name}": sum(d[f"top{k}_{name}"] for d in result["metrics"].values())
                    for k in (1, 5) for name in ("membership_regressions", "constraint_violations")}
    assert final_counts == result["final_constraints"]
    gain = (result["metrics"]["source_digits"]["top1_hits"]-853)/(958-853)
    assert gain == result["source_training_gain_retained_fraction"]
    gate = not any(final_counts.values()) and gain >= .9
    assert gate == (result["status"] == "train_gate_pass")
    verdict = dict(schema="aiflow-simultaneous-retention-independent-verification/v1", status="reproduced",
        result_sha256=_sha(out / "simultaneous_result.json"), checkpoint_sha256=_sha(checkpoint),
        constraints_rebuilt_bit_exact=True, linearized_certificates=certificates, batch_checks=checks,
        first_step_microscope=first_step,
        parameter_layers=layers, parameter_tensors=len(layers), unchanged_canonical_and_parent=True,
        weight_updates_applied=sum("step_l2_fp64" in h.get("correction", {}) for h in result["history"]),
        observed_parameter_change_from_parent=bool(any(layer["parent_delta_l2_fp64"] > 0 for layer in layers)),
        strict_train_gate_pass=gate, human_boundary_labels=0, held_inputs_forwarded=0, crohme_rows=0,
        product_adopted=False, eligible_for_independent_performance_claim=False)
    _write(destination, verdict)
    print(json.dumps(dict(status="reproduced", gate=result["status"], layers=len(layers),
        updates=verdict["weight_updates_applied"], metrics=result["metrics"])))
    return 0


def main() -> int:
    """명시된 실험 폴더를 읽어 검증 결과를 새 파일로만 남긴다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    return run(parser.parse_args().output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
