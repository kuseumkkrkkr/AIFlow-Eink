"""넓은 TRAIN 증강에서 원본 teacher 오류와 실제 정답 손실의 충돌을 측정한다.

가중치 업데이트·검증 fold 조회·생성물 hard CE 없이 봉인된 학습 입력만 분석한다.
이 결과는 학습 동역학 진단이며 정확도 검증이나 사람의 의미 승인이 아니다.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

import run_hwr_broad_diversity_v16 as broad
from audit_hwr_boundary_optimization_v1 import _layer_gradient_alignment
from hwr_boundary_distillation_v1 import boundary_kl_loss
from run_hwr_affine_distillation_experiment_v1 import _load_teacher, _predict_logits

OUT = broad.previous.ROOT / "artifacts/hwr_broad_teacher_conflict_20261005_v19"


def logit_gradients(real, teacher, truth):
    """원본 CE와 T=2 teacher KL의 샘플별 미분을 batch 축약 전 계산한다."""
    p = torch.softmax(real, 1)
    ce = .7 * (p - F.one_hot(truth, 372))
    kl = .4 * (torch.softmax(real / 2., 1) - torch.softmax(teacher / 2., 1))
    return ce, kl


def summarize(real, teacher, truth, selected):
    """동일 원본 logit 공간의 충돌 비율과 정답 회복/퇴행을 계산한다."""
    if not selected.any():
        return dict(rows=0)
    z, t, y = (torch.from_numpy(np.array(a[selected], copy=True)) for a in (real, teacher, truth))
    ce, kl = logit_gradients(z, t, y)
    dot = (ce * kl).sum(1)
    denominator = ce.norm(dim=1) * kl.norm(dim=1)
    meaningful = (ce.norm(dim=1) > 1e-7) & (kl.norm(dim=1) > 1e-7)
    cosine = dot[meaningful] / denominator[meaningful]
    correct, teacher_correct = z.argmax(1) == y, t.argmax(1) == y
    return dict(rows=len(y), teacher_correct=int(teacher_correct.sum()), student_correct=int(correct.sum()),
        recovered=int((~teacher_correct & correct).sum()), regressed=int((teacher_correct & ~correct).sum()),
        nonzero_gradient_rows=int(meaningful.sum()), opposed_rows=int((dot[meaningful] < 0).sum()),
        cosine_mean=float(cosine.mean()) if len(cosine) else None,
        weighted_kl_to_ce_norm_ratio_mean=float((kl.norm(dim=1) / ce.norm(dim=1).clamp_min(1e-12)).mean()))


def selftest():
    """해석식과 실제 KL/CE autograd를 비교하고 같은 teacher의 영 미분을 확인한다."""
    torch.manual_seed(19)
    z = torch.randn(7, 372, requires_grad=True)
    t = torch.randn(7, 372)
    y = torch.arange(7, dtype=torch.long)
    ce, kl = logit_gradients(z, t, y)
    actual_ce = torch.autograd.grad(.7 * F.cross_entropy(z, y), z, retain_graph=True)[0]
    actual_kl = torch.autograd.grad(.2 * boundary_kl_loss(z, t, 2.), z)[0]
    assert torch.allclose(actual_ce * 7, ce, atol=1e-7, rtol=1e-5)
    assert torch.allclose(actual_kl * 7, kl, atol=1e-7, rtol=1e-5)
    assert torch.count_nonzero(logit_gradients(t, t, y)[1]) == 0
    print(json.dumps(dict(selftest="pass",analytic_gradients_match_autograd=True)))


def run():
    """고정 128개 학습 batch를 분석하고 4층의 실제 parameter gradient를 따로 저장한다."""
    if OUT.exists():
        raise FileExistsError("refusing diagnostic overwrite")
    if broad.previous.aug.base._guard_commit("teacher_conflict_v19") is None:
        return 78
    plan, _, a = broad.load()
    checkpoints = {"canonical": broad.previous.CHECKPOINT,
        "broad_real": broad.OUT / "broad_real/research.pt",
        "broad_augmented": broad.OUT / "broad_augmented/research.pt"}
    sha = {k: broad.previous._sha(v) for k, v in checkpoints.items()}
    manifest = json.loads((broad.OUT / "research_candidate_manifest.json").read_text(encoding="utf-8"))
    for name in broad.ARMS:
        if sha[name] != manifest[name]["checkpoint_sha256"]:
            raise ValueError("frozen research checkpoint changed")
    steps = np.linspace(0, broad.STEPS - 1, 128, dtype=np.int64)
    ids = np.array(a["shared_schedule"][steps]).reshape(-1)
    y = np.array(a["population_labels"][ids], dtype=np.int64)
    teacher = np.array(a["population_teacher_logits"][ids], copy=True)
    x = np.array(a["population_features"][ids], copy=True)
    view = np.array(a["scheduled_augmented_features"][steps], copy=True).reshape(-1, 128, 5)
    wrong = teacher.argmax(1) != y
    changed = np.any(x != view, axis=(1, 2))
    OUT.mkdir()
    broad.previous._write(OUT / "frozen_plan.json", dict(schema="aiflow-teacher-conflict-plan/v19",
        script_sha256=broad.previous._sha(Path(__file__)), parent_plan_sha256=broad.previous._sha(broad.OUT / "frozen_plan.json"),
        checkpoint_sha256=sha, scheduled_steps=steps.tolist(), rows=len(ids), selection="128 uniformly spaced scheduled TRAIN batches; no error-based numerical selection",
        gradient_case_selection="first 32 changed rows per canonical-correct/wrong category, diagnostic strata not population estimates",
        optimizer_steps=0, validation_folds_read=0, official_test_rows_read=0, crohme_rows=0, synthetic_hard_labels=0,
        human_semantic_approval=False, product_adopted=False))
    np.savez_compressed(OUT / "inputs.npz", population_indices=ids, steps=steps, truth=y, teacher=teacher, changed=changed,
        shape_temperature=np.array([broad.TEMPERATURES[(int(s)//3+c)%4] for s in steps for c in range(broad.BATCH)]))
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    results, gradients = {}, []
    for name, checkpoint in checkpoints.items():
        model, labels, _ = _load_teacher(checkpoint, torch.device("cpu"))
        model.eval()
        initial = {k: p.detach().clone() for k, p in model.state_dict().items()}
        real_z = teacher if name == "canonical" else _predict_logits(model, x, torch.device("cpu"), broad.BATCH)
        view_z = _predict_logits(model, view, torch.device("cpu"), broad.BATCH)
        np.savez_compressed(OUT / f"{name}_logits.npz", real=real_z, view=view_z)
        results[name] = {category: summarize(real_z, teacher, y, mask) for category, mask in
            (("all", np.ones(len(ids), bool)), ("teacher_wrong", wrong), ("teacher_correct", ~wrong))}
        for category, mask in (("teacher_wrong", wrong), ("teacher_correct", ~wrong)):
            selected = np.flatnonzero(mask & changed)[:32]
            if not len(selected):
                continue
            real = torch.from_numpy(x[selected].copy())
            generated = torch.from_numpy(view[selected].copy())
            target = torch.from_numpy(teacher[selected].copy())
            truth = torch.from_numpy(y[selected].copy())
            z = model.math_head(model.encode(torch.cat((real, generated))))
            count = len(selected)
            losses = (.7 * F.cross_entropy(z[:count], truth), .2 * boundary_kl_loss(z[:count], target, 2.),
                .1 * boundary_kl_loss(z[count:], target, 2.))
            named = list(model.named_parameters())
            g = [torch.autograd.grad(loss, [p for _, p in named], retain_graph=True) for loss in losses]
            if not all(torch.isfinite(value).all() for group in g for value in group):
                raise FloatingPointError("nonfinite parameter gradient")
            filename = f"{name}_{category}_gradients.npz"
            np.savez_compressed(OUT / filename, **{k: torch.cat([v.detach().flatten() for v in values]).numpy()
                for k, values in zip(("ce", "kl_real", "kl_view"), g)})
            cases = {}
            for index, loss_name in ((1, "kl_real"), (2, "kl_view")):
                # 재사용 helper는 submodule 단위, 별도 bucket은 4개 encoder 층을 합산한다.
                bucketed = [(".".join(n.split(".")[:3]) + ".all" if n.startswith("encoder.layers.") else n, p) for n, p in named]
                aligned = _layer_gradient_alignment(bucketed, g[0], g[index])
                for item in aligned.values():
                    if min(item["weighted_base_gradient_l2"], item["weighted_boundary_gradient_l2"]) < 1e-5:
                        item["cosine"] = None
                cases[loss_name] = aligned
            gradients.append(dict(model=name, category=category, sample_rows=selected.tolist(),
                weighted_losses=[float(v.detach()) for v in losses], gradients_file=filename, layer_alignment=cases))
        assert all(torch.equal(p, model.state_dict()[k]) for k, p in initial.items())
        print(json.dumps(dict(event="teacher_conflict_model_done",model=name,diagnostics=results[name])), flush=True)
        del model
    population_y = a["population_labels"]
    population_wrong = np.asarray(a["population_teacher_logits"]).argmax(1) != population_y
    frequency = np.bincount(a["shared_schedule"].reshape(-1), minlength=len(population_y))
    assert all(broad.previous._sha(path) == sha[name] for name, path in checkpoints.items())
    result = dict(schema="aiflow-teacher-conflict-result/v19", status="completed_train_dynamics_not_accuracy",
        population_rows=len(population_y), population_teacher_wrong=int(population_wrong.sum()),
        scheduled_exposures=int(frequency.sum()), scheduled_teacher_wrong_exposures=int(frequency[population_wrong].sum()),
        sampled_rows=len(ids), sampled_unique_originals=len(np.unique(ids)), sampled_classes=len(np.unique(y)), changed_views=int(changed.sum()),
        logit_gradient_diagnostics=results, parameter_gradient_cases=gradients, model_weights_unchanged=True,
        checkpoint_sha256=sha, optimizer_steps=0, validation_folds_read=0, official_test_rows_read=0, crohme_rows=0,
        synthetic_hard_labels=0, human_semantic_approval=False, product_adopted=False,
        limits="TRAIN inputs already used in optimization; logit cosine is not parameter cosine; stratified batches are not population estimates; gradient opposition is not causal proof of accuracy harm.")
    broad.previous._write(OUT / "conflict_result.json", result)
    print(json.dumps(dict(event="teacher_conflict_completed", report=str(OUT / "conflict_result.json"))), flush=True)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("selftest", "run"))
    args = parser.parse_args()
    if args.mode == "selftest":
        selftest()
    else:
        raise SystemExit(run())
