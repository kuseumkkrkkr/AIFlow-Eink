"""V19 저장 tensor를 독립 NumPy 미분식으로 다시 계산한다. 새 평가 입력은 없다."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import audit_hwr_broad_teacher_conflict_v19 as audit
from run_hwr_affine_distillation_experiment_v1 import _load_teacher


def probability(z):
    """유한 logit의 softmax를 독립 double 연산으로 계산한다."""
    e = np.exp(z.astype(np.float64) - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def alignment(ce, kl):
    """저장한 전 parameter vector에서 각 손실의 방향과 강도를 다시 계산한다."""
    a, b = float(np.linalg.norm(ce)), float(np.linalg.norm(kl))
    return dict(ce_norm=a, kl_norm=b, cosine=float(ce @ kl / (a*b)) if min(a, b) >= 1e-5 else None,
        kl_to_ce_norm_ratio=b/a if a else None)


def main():
    """원본 출처, 표본 순위, 모든 층 gradient 및 불변 SHA를 검사하고 근거를 봉인한다."""
    out = audit.OUT
    result = json.loads((out / "conflict_result.json").read_text(encoding="utf-8"))
    plan = json.loads((out / "frozen_plan.json").read_text(encoding="utf-8"))
    assert audit.broad.previous._sha(Path(audit.__file__)) == plan["script_sha256"]
    _, _, population = audit.broad.load()
    with np.load(out / "inputs.npz", allow_pickle=False) as inputs:
        ids, y, teacher, changed, steps = (inputs[k] for k in ("population_indices", "truth", "teacher", "changed", "steps"))
        assert np.array_equal(ids, population["shared_schedule"][steps].reshape(-1))
        assert np.array_equal(y, population["population_labels"][ids])
        assert np.array_equal(teacher, population["population_teacher_logits"][ids])
        assert np.array_equal(changed, np.any(population["population_features"][ids] !=
            population["scheduled_augmented_features"][steps].reshape(-1,128,5), axis=(1,2)))
    wrong = teacher.argmax(1) != y
    checks = []
    for name in ("canonical", *audit.broad.ARMS):
        path = audit.broad.previous.CHECKPOINT if name == "canonical" else audit.broad.OUT / name / "research.pt"
        assert audit.broad.previous._sha(path) == plan["checkpoint_sha256"][name]
        model, _, _ = _load_teacher(path, torch.device("cpu"))
        named = list(model.named_parameters())
        total = sum(p.numel() for _, p in named)
        logits = np.load(out / f"{name}_logits.npz", allow_pickle=False)
        real = logits["real"]
        assert real.shape == logits["view"].shape == (4096, 372)
        assert np.isfinite(real).all() and np.isfinite(logits["view"]).all()
        ce = .7 * probability(real)
        ce[np.arange(len(y)),y] -= .7
        kl = .4 * (probability(real/2.) - probability(teacher/2.))
        for category, mask in (("all", np.ones(len(y),bool)), ("teacher_wrong",wrong), ("teacher_correct",~wrong)):
            d = result["logit_gradient_diagnostics"][name][category]
            correct = real.argmax(1) == y
            assert d["rows"] == int(mask.sum()) and d["student_correct"] == int(correct[mask].sum())
            assert d["recovered"] == int((mask & wrong & correct).sum())
            assert d["regressed"] == int((mask & ~wrong & ~correct).sum())
            ca, cb = ce[mask], kl[mask]
            meaningful = (np.linalg.norm(ca,axis=1)>1e-7) & (np.linalg.norm(cb,axis=1)>1e-7)
            dot = (ca * cb).sum(1)
            assert d["nonzero_gradient_rows"] == int(meaningful.sum())
            assert d["opposed_rows"] == int((dot[meaningful]<0).sum())
            if meaningful.any():
                expected = (dot[meaningful]/(np.linalg.norm(ca[meaningful],axis=1)*np.linalg.norm(cb[meaningful],axis=1))).mean()
                assert abs(expected-d["cosine_mean"]) < 2e-5
        for case in (c for c in result["parameter_gradient_cases"] if c["model"] == name):
            values = np.load(out / case["gradients_file"],allow_pickle=False)
            assert all(values[k].shape == (total,) and np.isfinite(values[k]).all() for k in ("ce","kl_real","kl_view"))
            groups = {}
            offset = 0
            for key, p in named:
                count = p.numel()
                group = ".".join(key.split(".")[:3]) if key.startswith("encoder.layers.") else key.rsplit(".",1)[0]
                groups.setdefault(group, []).extend(range(offset, offset+count))
                offset += count
            assert len([k for k in groups if k.startswith("encoder.layers.")]) == 4
            for loss in ("kl_real", "kl_view"):
                for group, positions in groups.items():
                    a = alignment(values["ce"][positions].astype(np.float64), values[loss][positions].astype(np.float64))
                    recorded = case["layer_alignment"][loss][group]
                    assert np.isclose(a["ce_norm"], recorded["weighted_base_gradient_l2"], rtol=1e-5, atol=1e-7)
                    assert np.isclose(a["kl_norm"], recorded["weighted_boundary_gradient_l2"], rtol=1e-5, atol=1e-7)
                    if a["cosine"] is not None:
                        assert abs(a["cosine"]-recorded["cosine"]) < 1e-5
            ca = values["ce"].astype(np.float64)
            rb, vb = (values[k].astype(np.float64) for k in ("kl_real","kl_view"))
            checks.append(dict(model=name,category=case["category"],
                ce_vs_real_kl=alignment(ca, rb), ce_vs_view_kl=alignment(ca, vb),
                ce_vs_combined_kl=alignment(ca, rb+vb), ce_vs_full_objective=alignment(ca, ca+rb+vb)))
        del model
    report = dict(status="pass",schema="aiflow-independent-teacher-conflict-verification/v19",
        result_sha256=audit.broad.previous._sha(out / "conflict_result.json"),
        diagnostic_script_sha256=audit.broad.previous._sha(Path(audit.__file__)), verifier_script_sha256=audit.broad.previous._sha(Path(__file__)),
        original_train_sample_and_targets_rebuilt=True, all_rank_and_logit_conflict_counts_rebuilt=True,
        all_four_encoder_gradient_groups_rebuilt=True, global_parameter_alignment=checks,
        parameter_values=total, optimizer_steps=0, validation_folds_read=0, crohme_rows=0,
        product_adopted=False, limit="Saved final diagnostic gradients, not historical optimization trajectory or causal accuracy proof.",
        artifacts_sha256={p.name:audit.broad.previous._sha(p) for p in out.iterdir() if p.is_file()})
    audit.broad.previous._write(out / "independent_verification.json", report)
    print(json.dumps(report, ensure_ascii=False),flush=True)


if __name__ == "__main__":
    main()
