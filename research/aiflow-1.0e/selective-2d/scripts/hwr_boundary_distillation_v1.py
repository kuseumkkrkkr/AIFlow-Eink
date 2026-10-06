"""Audit and consume frozen-teacher boundary pseudo-targets without hard synthetic labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "artifacts" / "hwr_probability_boundary_tube_20261004_global_top1_v2" / "probability_boundary_tube_audit.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "hwr_boundary_distillation_targets_20261004"
POLICIES = ("full", "top5")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def soft_targets(logits: np.ndarray, temperature: float, policy: str = "full") -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 372 or not np.isfinite(values).all():
        raise ValueError("pseudo-target logits must be finite [N,372] values")
    if not np.isfinite(temperature) or temperature <= 0.0 or policy not in POLICIES:
        raise ValueError("temperature must be positive and policy must be full or top5")
    scaled = values / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    probabilities = np.exp(scaled)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    if policy == "top5":
        retained = np.argsort(-values, axis=1, kind="stable")[:, :5]
        mask = np.zeros_like(probabilities)
        mask[np.arange(len(values))[:, None], retained] = 1.0
        probabilities *= mask
        probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities.astype(np.float32)


def boundary_kl_loss(student_logits, teacher_logits, temperature: float, policy: str = "full"):
    """Return T^2 KL(q_teacher || p_student); generated points receive no hard CE."""
    import torch
    import torch.nn.functional as F

    if student_logits.ndim != 2 or student_logits.shape != teacher_logits.shape or student_logits.shape[1] != 372:
        raise ValueError("student and teacher boundary logits must align as [N,372]")
    if not len(student_logits) or not torch.isfinite(student_logits).all() or not torch.isfinite(teacher_logits).all():
        raise ValueError("boundary loss needs nonempty finite student and teacher logits")
    if not np.isfinite(temperature) or temperature <= 0.0 or policy not in POLICIES:
        raise ValueError("temperature must be positive and policy must be full or top5")
    with torch.no_grad():
        target = torch.softmax(teacher_logits.detach().float() / temperature, dim=1)
        if policy == "top5":
            retained = torch.argsort(teacher_logits.detach().float(), dim=1, descending=True, stable=True)[:, :5]
            mask = torch.zeros_like(target).scatter_(1, retained, 1.0)
            target = target * mask
            target = target / target.sum(dim=1, keepdim=True)
    return F.kl_div(
        F.log_softmax(student_logits.float() / temperature, dim=1),
        target, reduction="batchmean",
    ) * temperature ** 2


def boundary_training_loss(
    base_loss, model, features, teacher_logits, generator,
    weight: float, batch_size: int, temperature: float, policy: str,
    student_mode: str = "current",
):
    """Opt-in replay; weight=0 returns the exact original loss without a forward."""
    import torch

    if not np.isfinite(weight) or not 0.0 <= weight < 1.0:
        raise ValueError("boundary weight must be finite in [0,1)")
    if weight == 0.0:
        return base_loss, base_loss.new_zeros(()), torch.empty(0, dtype=torch.long)
    if batch_size < 1 or features is None or teacher_logits is None:
        raise ValueError("enabled boundary replay needs a positive batch and verified candidates")
    if student_mode not in ("current", "deterministic"):
        raise ValueError("boundary student mode must be current or deterministic")
    if features.ndim != 3 or features.shape[1:] != (128, 5) or teacher_logits.shape != (len(features), 372) or not len(features):
        raise ValueError("boundary replay tensors must align as [N,128,5] and [N,372]")
    # A separate CPU RNG avoids changing the original sampler/augmentation streams.
    indices = torch.randint(len(features), (batch_size,), generator=generator)
    input_batch = features.index_select(0, indices.to(features.device))
    target_batch = teacher_logits.index_select(0, indices.to(teacher_logits.device))
    # eval() disables replay dropout, NOT autograd. Restore even mixed module modes.
    previous_modes = [(module, module.training) for module in model.modules()] if student_mode == "deterministic" else []
    try:
        if previous_modes:
            model.eval()
        logits = model.math_head(model.encode(input_batch))
    finally:
        for module, training in previous_modes:
            module.training = training
    boundary_loss = boundary_kl_loss(logits, target_batch, temperature, policy)
    return (1.0 - weight) * base_loss + weight * boundary_loss, boundary_loss, indices


def load_boundary_candidates(
    report_path: Path,
    checkpoint_sha256: str | None = None,
    expected_labels: list[str] | None = None,
) -> tuple[dict, np.ndarray, np.ndarray]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    provenance = report["provenance"]
    if provenance.get("heldout_rows_read") != 0 or provenance.get("crohme_rows") != 0:
        raise ValueError("boundary training candidates must come entirely from the admitted training pool")
    if provenance.get("query_source") != "UJI Pen v2 official writer-disjoint train split only" or provenance.get("fit_source") != "HWRT official train split only":
        raise ValueError("boundary candidate source/split policy differs from the audited pilot")
    if checkpoint_sha256 is not None and provenance.get("teacher_checkpoint_sha256", "").lower() != checkpoint_sha256.lower():
        raise ValueError("boundary pseudo-targets belong to a different teacher checkpoint")
    refinement = report["adaptive_teacher_boundary_refinement"]
    if refinement.get("boundary_rule") != "global-top1":
        raise ValueError("boundary training candidates must bracket the global recognition boundary")
    cache = refinement["numeric_candidate_cache"]
    if len(cache["class_labels"]) != 372 or (expected_labels is not None and cache["class_labels"] != expected_labels):
        raise ValueError("boundary pseudo-target vocabulary/order differs from the model")
    arrays = {}
    for name, item in cache["artifacts"].items():
        path = Path(item["path"])
        if _sha256(path) != item["sha256"]:
            raise ValueError(f"boundary candidate cache hash mismatch: {name}")
        array = np.load(path, allow_pickle=False)
        if list(array.shape) != item["shape"] or str(array.dtype) != item["dtype"]:
            raise ValueError(f"boundary candidate cache shape/dtype mismatch: {name}")
        arrays[name] = array
    features, logits, valid = arrays["candidate_features"], arrays["teacher_logits"], arrays["geometry_valid"]
    count = features.shape[0]
    if features.shape != (count, 2, 128, 5) or features.dtype != np.float32 or logits.shape != (count, 2, 372) or logits.dtype != np.float32 or valid.shape != (count, 2) or valid.dtype != np.bool_:
        raise ValueError("boundary features/logits/geometry masks are not aligned")
    if not count or not valid.all() or not np.isfinite(features).all() or not np.isfinite(logits).all():
        raise ValueError("boundary candidate cache contains rejected or nonfinite rows")
    if (features[..., :2] < 0.0).any() or (features[..., :2] > 1.0).any():
        raise ValueError("boundary spatial coordinates are outside the model input contract")
    for name in ("query_training_row_indices", "donor_training_row_indices"):
        if arrays[name].shape != (count,) or (arrays[name] < 0).any():
            raise ValueError("boundary cache is missing source training-row provenance")
    target_id = cache["class_labels"].index("2")
    competitors = logits.copy()
    competitors[..., target_id] = -np.inf
    margins = logits[..., target_id] - competitors.max(axis=-1)
    if not (margins[:, 0] > 0.0).all() or not (margins[:, 1] <= 0.0).all():
        raise ValueError("cached pseudo-targets do not bracket the global class-2 boundary")
    return report, features.reshape(-1, 128, 5), logits.reshape(-1, 372)


def _audit(report_path: Path, output_dir: Path) -> int:
    report, features, logits = load_boundary_candidates(report_path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty target directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    comparisons = []
    for temperature in (1.0, 2.0, 4.0):
        full = soft_targets(logits, temperature, "full").astype(np.float64)
        for policy in POLICIES:
            targets = soft_targets(logits, temperature, policy)
            q = targets.astype(np.float64)
            mask = q > 0.0
            discarded = (
                np.zeros(len(full), dtype=np.float64) if policy == "full"
                else np.clip(1.0 - np.sum(full * mask, axis=1), 0.0, 1.0)
            )
            divergence = np.sum(np.where(mask, q * (np.log(np.maximum(q, 1.0e-300)) - np.log(np.maximum(full, 1.0e-300))), 0.0), axis=1)
            path = output_dir / f"targets_{policy}_t{temperature:g}.npy"
            np.save(path, targets, allow_pickle=False)
            comparisons.append({
                "temperature": temperature, "policy": policy, "rows": len(targets),
                "probability_mass_discarded_mean": float(discarded.mean()),
                "kl_projected_target_to_full_mean": float(divergence.mean()),
                "per_sample_logit_gradient_l2_at_teacher_mean": float(np.linalg.norm(temperature * (full - q), axis=1).mean()),
                "gradient_definition": "T*(p_teacher-q_projected) before minibatch reduction for T^2 KL",
                "target_entropy_nats_mean": float(np.mean(-np.sum(q * np.log(np.maximum(q, 1.0e-300)), axis=1))),
                "target_path": str(path.resolve()), "target_sha256": _sha256(path),
            })
    result = {
        "schema": "aiflow-hwr-boundary-distillation-target-audit/v1",
        "source_report": str(report_path.resolve()), "source_report_sha256": _sha256(report_path),
        "teacher_checkpoint_sha256": report["provenance"]["teacher_checkpoint_sha256"],
        "class_labels": report["adaptive_teacher_boundary_refinement"]["numeric_candidate_cache"]["class_labels"],
        "synthetic_candidate_rows": int(len(features)), "original_training_label": "2",
        "hard_synthetic_labels_assigned": 0, "human_labels_available": False,
        "heldout_rows_read": 0, "crohme_rows": 0, "student_training_performed": False,
        "comparisons": comparisons,
    }
    result_path = output_dir / "target_policy_audit.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "boundary_target_policy_audit_complete", "report": str(result_path.resolve()), "rows": len(features), "policies": len(comparisons)}))
    return 0


def _microscope(report_path: Path, checkpoint: Path, output_dir: Path) -> int:
    from audit_hwr_probability_boundary_tube_v1 import _guard_commit
    if _guard_commit("before_boundary_training_microscope") is None:
        return 78
    import torch
    from run_hwr_affine_distillation_experiment_v1 import _load_teacher, _predict_logits

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite microscope directory: {output_dir}")
    torch.set_num_threads(2)
    checkpoint_sha = _sha256(checkpoint)
    model, labels, _ = _load_teacher(checkpoint, torch.device("cpu"))
    report, features, teacher_logits = load_boundary_candidates(report_path, checkpoint_sha, labels)
    model.eval()  # No dropout, optimizer, heldout data, or checkpoint mutation.
    actual_logits = _predict_logits(model, features, torch.device("cpu"), 16)
    x = torch.from_numpy(features.copy())
    t = torch.from_numpy(teacher_logits.copy())
    cases = []
    for policy, temperature in (("full", 1.0), ("full", 2.0), ("top5", 1.0)):
        model.zero_grad(set_to_none=True)
        base_loss = torch.zeros((), requires_grad=True)
        mixed, loss, indices = boundary_training_loss(
            base_loss, model, x, t, torch.Generator().manual_seed(20261004),
            weight=0.1, batch_size=16, temperature=temperature, policy=policy,
        )
        mixed.backward()
        parameter_gradients = [p.grad for p in model.parameters() if p.grad is not None]
        if not parameter_gradients or not all(torch.isfinite(g).all() for g in parameter_gradients):
            raise AssertionError("actual model boundary gradients are missing or nonfinite")
        layer_squared_norms = {}
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                layer = name.rsplit(".", 1)[0]
                layer_squared_norms[layer] = layer_squared_norms.get(layer, 0.0) + float(parameter.grad.float().square().sum())
        cases.append({
            "policy": policy, "temperature": temperature, "sample_rows": len(indices),
            "sample_indices": indices.tolist(), "boundary_weight": 0.1,
            "boundary_kl_loss": float(loss.detach()), "weighted_loss": float(mixed.detach()),
            "weighted_parameter_gradient_l2": float(torch.sqrt(sum(g.float().square().sum() for g in parameter_gradients))),
            "weighted_gradient_l2_by_layer": {name: float(np.sqrt(value)) for name, value in layer_squared_norms.items()},
            "base_loss_gradient": float(base_loss.grad), "finite_gradients": True,
        })
    base = torch.tensor(3.0, requires_grad=True)
    disabled, zero, indices = boundary_training_loss(base, None, None, None, None, 0.0, 16, 1.0, "full")
    if disabled is not base or float(zero) != 0.0 or len(indices):
        raise AssertionError("disabled adapter changed the original loss or ran replay")
    if not np.allclose(actual_logits, teacher_logits, atol=2.0e-5, rtol=2.0e-5):
        raise AssertionError("frozen checkpoint failed saved candidate logit parity")
    if any(abs(case["boundary_kl_loss"]) > 2.0e-5 for case in cases if case["policy"] == "full"):
        raise AssertionError("identical full teacher/student should be stationary before training")
    if any(case["weighted_parameter_gradient_l2"] > 1.0e-4 for case in cases if case["policy"] == "full"):
        raise AssertionError("full self-distillation has an unexpected nonstationary gradient")
    if _sha256(checkpoint) != checkpoint_sha:
        raise AssertionError("microscope modified the canonical checkpoint")
    result = {
        "schema": "aiflow-hwr-boundary-training-microscope/v1", "status": "pass",
        "source_report": str(report_path.resolve()), "source_report_sha256": _sha256(report_path),
        "teacher_checkpoint_sha256": checkpoint_sha, "script_sha256": _sha256(Path(__file__)),
        "candidate_rows": len(features), "max_logit_parity_error": float(np.max(np.abs(actual_logits - teacher_logits))),
        "disabled_adapter_returns_identical_loss_object": True, "cases": cases,
        "optimizer_steps": 0, "model_checkpoint_written": False,
        "hard_synthetic_labels_assigned": 0, "human_labels_available": False,
        "heldout_rows_read": 0, "crohme_rows": 0, "product_adopted": False,
        "interpretation": "Full-probability replay is stationary at the identical teacher; it preserves behavior but does not independently add correct human labels. Top5 projection changes the target and is not evidence of accuracy improvement.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "training_microscope.json"
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "boundary_training_microscope_complete", "status": "pass", "report": str(destination.resolve()), "cases": cases}))
    return 0


def _self_test() -> int:
    from audit_hwr_probability_boundary_tube_v1 import _guard_commit
    if _guard_commit("before_boundary_loss_torch_test") is None:
        return 78
    import torch

    base = torch.tensor(2.0, requires_grad=True)
    disabled, zero, indices = boundary_training_loss(base, None, None, None, None, 0.0, 16, 1.0, "full")
    if disabled is not base or float(zero) != 0.0 or len(indices):
        raise AssertionError("disabled adapter must preserve the exact original loss")
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.math_head = torch.nn.Linear(5, 372)

        def encode(self, features):
            return features.mean(dim=1)

    model = TinyModel()
    features = torch.ones(4, 128, 5)
    targets = torch.zeros(4, 372, requires_grad=True)
    outputs = []
    for _ in range(2):
        base = torch.tensor(2.0, requires_grad=True)
        mixed, loss, indices = boundary_training_loss(
            base, model, features, targets, torch.Generator().manual_seed(11), 0.25, 16, 1.0, "full",
        )
        mixed.backward()
        if not torch.isclose(base.grad, torch.tensor(0.75)) or targets.grad is not None:
            raise AssertionError("boundary loss mixing or teacher detach is incorrect")
        outputs.append((indices, loss.detach()))
    if not torch.equal(outputs[0][0], outputs[1][0]) or not torch.equal(outputs[0][1], outputs[1][1]):
        raise AssertionError("boundary replay is not deterministic for an independent fixed seed")
    model.train()
    model.math_head.eval()  # Deliberately mixed flags must survive replay.
    expected_modes = [module.training for module in model.modules()]
    model.zero_grad(set_to_none=True)
    mixed, _, _ = boundary_training_loss(
        torch.tensor(2.0, requires_grad=True), model, features, targets,
        torch.Generator().manual_seed(11), 0.25, 16, 1.0, "full", "deterministic",
    )
    mixed.backward()
    if expected_modes != [module.training for module in model.modules()] or not model.math_head.weight.grad.isfinite().all():
        raise AssertionError("deterministic replay failed mode restoration or disabled autograd")
    model.dropout = torch.nn.Dropout(0.5)
    model.encode = lambda values: model.dropout(values.mean(dim=1))
    model.eval()
    frozen = model.math_head(model.encode(features)).detach()
    model.train()
    _, deterministic_loss, _ = boundary_training_loss(
        torch.tensor(2.0, requires_grad=True), model, features, frozen,
        torch.Generator().manual_seed(11), 0.25, 16, 1.0, "full", "deterministic",
    )
    if abs(float(deterministic_loss.detach())) > 1.0e-5 or not model.dropout.training:
        raise AssertionError("replay dropout was not disabled temporarily or restored afterward")
    logits = np.linspace(-3.0, 3.0, 372, dtype=np.float32)[None, :]
    for temperature in (1.0, 2.0):
        full = soft_targets(logits, temperature)
        top5 = soft_targets(logits, temperature, "top5")
        if not np.allclose(full.sum(axis=1), 1.0) or not np.allclose(top5.sum(axis=1), 1.0) or np.count_nonzero(top5) != 5:
            raise AssertionError("soft-target support or mass is incorrect")
        student = torch.tensor(logits, requires_grad=True)
        teacher = torch.tensor(logits, requires_grad=True)
        loss = boundary_kl_loss(student, teacher, temperature)
        loss.backward()
        if abs(float(loss.detach())) > 1.0e-5 or student.grad.abs().max().item() > 1.0e-5 or teacher.grad is not None:
            raise AssertionError("full teacher self-distillation must have zero loss/gradient and detach the teacher")
        student = torch.tensor(logits, requires_grad=True)
        loss = boundary_kl_loss(student, teacher, temperature, "top5")
        loss.backward()
        if not torch.isfinite(loss) or not torch.isfinite(student.grad).all() or student.grad.norm().item() <= 0.0:
            raise AssertionError("top5 projection must expose its nonzero target-change gradient")
    print(json.dumps({"self_test": "pass", "hard_synthetic_labels": 0, "teacher_detached": True}))
    return 0


def main() -> int:
    from audit_hwr_probability_boundary_tube_v1 import DEFAULT_CHECKPOINT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("self-test", "audit", "microscope"), required=True)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()
    if args.mode == "self-test":
        return _self_test()
    if args.mode == "microscope":
        return _microscope(args.report, args.checkpoint, args.output_dir)
    return _audit(args.report, args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
