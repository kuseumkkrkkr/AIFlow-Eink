"""전 범위 replay의 예산/후보 변화/목표별 gradient를 학습 없이 비교한다."""
from __future__ import annotations

import argparse
from pathlib import Path

import json
import numpy as np

from audit_hwr_full_domain_tube_v1 import DEFAULT_CHECKPOINT, OUTPUT as SOURCE, ROOT, _guard_commit, _signature, _render
from hwr_full_domain_replay_v1 import load_full_domain_candidates, ClassTopologyBalancedSampler, class_macro_teacher_kl
from run_hwr_pendigits_tube_probe_v1 import _sha, _write

OUTPUT = ROOT / "artifacts/hwr_full_domain_replay_microscope_20261005"
SEED = 20261006


def _hits(logits: np.ndarray, classes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """372개 클래스 전체 순위를 사용해 Top-1·Top-5·global margin을 계산한다."""
    top = np.argsort(-logits, axis=1, kind="stable")
    competing = logits.copy(); competing[np.arange(len(logits)), classes] = -np.inf
    return top[:, 0] == classes, (top[:, :5] == classes[:, None]).any(1), logits[np.arange(len(logits)), classes] - competing.max(1)


def _run(out: Path) -> int:
    """모델 가중치/heldout을 변경하지 않고 동일 class-balanced 입력에서 gradient만 확인한다."""
    if out.exists():
        raise FileExistsError("refusing to overwrite microscope artifacts")
    if _guard_commit("before_full_domain_replay_microscope") is None:
        return 78
    import torch
    from torch.nn import functional as F
    from run_hwr_affine_distillation_experiment_v1 import _load_teacher
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    sha = _sha(DEFAULT_CHECKPOINT)
    model, labels, _ = _load_teacher(DEFAULT_CHECKPOINT, torch.device("cpu"))
    report_path = SOURCE / "full_domain_tube_audit.json"
    report, x, z, classes = load_full_domain_candidates(report_path, sha, labels)
    sampler = ClassTopologyBalancedSampler(classes, x[:, 0], SEED)
    selected = sampler.sample(len(np.unique(classes)))
    assert len(np.unique(classes[selected])) == 370
    schedule = ClassTopologyBalancedSampler(classes, x[:, 0], SEED).sample(370 * 10)
    budgets = np.bincount(classes[schedule], minlength=372)
    assert (budgets[budgets > 0] == 10).all()
    rows = []
    for c in np.unique(classes):
        mask = classes == c
        first, fifth, margin = _hits(z[mask, 0], classes[mask])
        next_first, next_fifth, next_margin = _hits(z[mask, 1], classes[mask])
        rows.append(dict(class_id=int(c), label=labels[c], pairs=int(mask.sum()), baseline_top1=int(first.sum()), changed_top1=int(next_first.sum()),
                         top1_lost=int((first & ~next_first).sum()), top1_recovered=int((~first & next_first).sum()),
                         top5_lost=int((fifth & ~next_fifth).sum()), original_margin_mean=float(margin.mean()), changed_margin_mean=float(next_margin.mean()),
                         uniform_pair_sampling_probability=float(mask.sum() / len(x)), balanced_class_probability=1 / 370))
    before_first, before_fifth, margin = _hits(z[:, 0], classes)
    after_first, after_fifth, changed_margin = _hits(z[:, 1], classes)
    lost = np.flatnonzero(before_fifth & ~after_fifth)
    out.mkdir(parents=True)
    cases = ("deterministic_view_full", "stochastic_view_full", "deterministic_original_full", "deterministic_view_top5")
    plan = dict(schema="aiflow-full-domain-replay-microscope-plan/v1", source_report_sha256=_sha(report_path),
                canonical_checkpoint_sha256=sha, script_sha256=_sha(Path(__file__)),
                dependencies={n:_sha(ROOT / "scripts" / n) for n in ("hwr_full_domain_replay_v1.py", "audit_hwr_full_domain_tube_v1.py", "run_hwr_affine_distillation_experiment_v1.py")},
                seed=SEED, cases=cases, sample_classes=370, microbatch=32, temperature=2.,
                training_steps=0, optimizer_created=False, heldout_rows_read=0, crohme_rows=0,
                original_target_policy="diagnostic only: forcing original teacher probabilities onto changed inputs is an unverified semantic invariance assumption",
                view_target_policy="teacher self-replay, not new human knowledge", hard_synthetic_labels_assigned=0,
                human_boundary_labels=0, product_adopted=False)
    _write(out / "frozen_plan.json", plan)
    np.save(out / "selected_pair_indices.npy", selected, allow_pickle=False)
    initial = {n: value.detach().clone() for n, value in model.state_dict().items()}
    histories, gradients = [], {}
    for case in cases:
        model.zero_grad(set_to_none=True)
        model.train(case.startswith("stochastic")); torch.manual_seed(SEED + 100)
        losses = []
        for start in range(0, len(selected), 32):
            ids = selected[start:start + 32]
            inputs = torch.from_numpy(np.array(x[ids, 1], copy=True))
            target = torch.from_numpy(np.array(z[ids, 0 if "original" in case else 1], copy=True))
            logits = model.math_head(model.encode(inputs))
            if case.endswith("top5"):
                probabilities = torch.softmax(target / 2., 1)
                retained = torch.argsort(target, dim=1, descending=True, stable=True)[:, :5]
                keep = torch.zeros_like(target).scatter_(1, retained, 1.)
                probabilities *= keep; probabilities /= probabilities.sum(1, keepdim=True)
                loss = F.kl_div(F.log_softmax(logits / 2., 1), probabilities.detach(), reduction="batchmean") * 4.
            else:
                loss = class_macro_teacher_kl(logits, target, torch.from_numpy(classes[ids]), 2.)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite microscope loss")
            (loss * len(ids) / len(selected)).backward()
            losses.append(dict(classes=len(ids), loss=float(loss.detach())))
        layer = {n:float(p.grad.norm()) for n,p in model.named_parameters() if p.grad is not None}
        gradient = torch.cat([p.grad.detach().flatten() for p in model.parameters() if p.grad is not None])
        gradients[case] = gradient.clone()
        result = dict(case=case, weighted_loss=sum(l["loss"] * l["classes"] / len(selected) for l in losses),
                      global_gradient_l2=float(gradient.norm()), layer_gradient_l2=layer, microbatches=losses)
        histories.append(result)
        assert all(torch.equal(value, model.state_dict()[n]) for n,value in initial.items())
        print(json.dumps(dict(event="full_domain_objective_gradient", case=case, loss=result["weighted_loss"], gradient_l2=result["global_gradient_l2"])), flush=True)
    comparisons = []
    for i, a in enumerate(cases):
        for b in cases[i + 1:]:
            ga, gb = gradients[a], gradients[b]
            cosine = None if min(float(ga.norm()), float(gb.norm())) < 1e-5 else float(torch.dot(ga, gb) / (ga.norm() * gb.norm()))
            comparisons.append(dict(first=a, second=b, cosine=cosine, tiny_gradient_cosine_suppressed=cosine is None))
    mapping = json.loads((SOURCE / "blind_review_mapping.json").read_text(encoding="utf-8"))
    details = []
    for pair in lost:
        details.append(dict(pair_index=int(pair), class_id=int(classes[pair]), label=labels[classes[pair]],
                            query_training_row=mapping[2 * pair]["query_training_row"], donor_training_row=mapping[2 * pair]["donor_training_row"],
                            original_top5=[labels[i] for i in np.argsort(-z[pair, 0], kind="stable")[:5]],
                            changed_top5=[labels[i] for i in np.argsort(-z[pair, 1], kind="stable")[:5]],
                            original_margin=float(margin[pair]), changed_margin=float(changed_margin[pair])))
    if len(lost):
        _render(x[lost].reshape(-1,128,5), [mapping[2*pair + side]["candidate_id"] for pair in lost for side in (0,1)], out / "top5_candidate_losses_qa.png")
    if _sha(DEFAULT_CHECKPOINT) != sha:
        raise AssertionError("canonical checkpoint changed")
    result = dict(schema="aiflow-full-domain-replay-microscope-result/v1", status="completed_no_training", frozen_plan_sha256=_sha(out / "frozen_plan.json"),
                  class_metrics=rows, balanced_schedule=dict(draws=3700, classes=370, per_class_count=10, class_budget_ratio=1.,
                  uniform_pair_budget_ratio=float(max(row["pairs"] for row in rows) / min(row["pairs"] for row in rows))),
                  objectives=histories, gradient_comparisons=comparisons, top5_candidate_losses=details,
                  model_weights_unchanged=True, canonical_checkpoint_unchanged=True, teacher_is_not_human_target=True,
                  student_training_performed=False, eligible_for_product_selection=False, heldout_rows_read=0, crohme_rows=0, human_boundary_labels=0, product_adopted=False)
    _write(out / "replay_microscope.json", result)
    print(json.dumps(dict(event="full_domain_replay_microscope_complete", classes=370, top5_losses=len(details), optimizer_steps=0)), flush=True)
    return 0


def main() -> int:
    """별도 폴더에서 학습 전 전 범위 gradient 현미경 검증을 실행한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    return _run(args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
