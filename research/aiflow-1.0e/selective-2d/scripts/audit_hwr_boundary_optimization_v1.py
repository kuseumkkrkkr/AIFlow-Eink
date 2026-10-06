"""Paired training-integrity smoke test; no accuracy scoring or model selection.

All real supervision comes from the previously admitted HWRT/UJI train cache.
Boundary replay has frozen teacher probabilities, not synthetic hard labels.
Fixed arms isolate base-loss scaling and stochastic boundary replay.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import hashlib
import json
from pathlib import Path
import time

from audit_hwr_probability_boundary_tube_v1 import DEFAULT_CHECKPOINT, DEFAULT_DATA_DIR, ROOT, _guard_commit

DEFAULT_BOUNDARY_REPORT = ROOT / "artifacts/hwr_probability_boundary_tube_20261004_global_top1_v2/probability_boundary_tube_audit.json"
DEFAULT_OUTPUT = ROOT / "artifacts/hwr_boundary_optimization_20261004_dropout_ablation"
SEED = 20261004


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _layer_gradient_alignment(named_parameters, base_gradients, boundary_gradients) -> dict:
    import math

    sums = defaultdict(lambda: [0.0, 0.0, 0.0])
    for (name, _), base, boundary in zip(named_parameters, base_gradients, boundary_gradients):
        layer = name.rsplit(".", 1)[0]
        if base is not None:
            sums[layer][0] += float(base.float().square().sum())
        if boundary is not None:
            sums[layer][1] += float(boundary.float().square().sum())
        if base is not None and boundary is not None:
            sums[layer][2] += float((base.float() * boundary.float()).sum())
    result = {}
    for name, (base_sq, boundary_sq, dot) in sums.items():
        denominator = math.sqrt(base_sq * boundary_sq)
        result[name] = {
            "weighted_base_gradient_l2": math.sqrt(base_sq),
            "weighted_boundary_gradient_l2": math.sqrt(boundary_sq),
            "cosine": None if denominator == 0.0 else max(-1.0, min(1.0, dot / denominator)),
        }
    return result


def _self_test() -> int:
    if _guard_commit("before_boundary_optimization_self_test") is None:
        return 78
    import torch

    named = [("layer.weight", None), ("layer.bias", None)]
    left = [torch.tensor([1.0, 0.0]), torch.tensor([0.0])]
    opposed = [torch.tensor([-2.0, 0.0]), torch.tensor([0.0])]
    result = _layer_gradient_alignment(named, left, opposed)["layer"]
    assert result["weighted_base_gradient_l2"] == 1.0
    assert result["weighted_boundary_gradient_l2"] == 2.0
    assert result["cosine"] == -1.0
    assert _layer_gradient_alignment(named, left, left)["layer"]["cosine"] == 1.0
    assert _layer_gradient_alignment(named, left, [None, None])["layer"]["cosine"] is None
    print(json.dumps({"self_test": "pass", "gradient_alignment": True, "no_accuracy_scoring": True}))
    return 0


def _package(args) -> int:
    """Attach the runtime input contract without changing any trained weights."""
    if _guard_commit("before_integrity_checkpoint_packaging") is None:
        return 78
    import torch
    from train_character_classifier_v1 import input_contract
    from run_hwr_affine_distillation_experiment_v1 import _load_teacher

    source = args.output_dir / "optimization_microscope.json"
    audit = json.loads(source.read_text(encoding="utf-8"))
    if audit.get("status") != "training_integrity_smoke_pass" or audit.get("accuracy_scoring_performed") or audit.get("crohme_rows") != 0 or audit.get("heldout_rows_read") != 0:
        raise ValueError("packaging requires a passed training-only integrity audit")
    destination_report = args.output_dir / "checkpoint_packaging.json"
    destinations = [args.output_dir / f"{arm['arm']}_smoke_compatible.pt" for arm in audit["arms"]]
    if destination_report.exists() or any(path.exists() for path in destinations):
        raise FileExistsError("refusing to overwrite packaged research checkpoints")
    packaged = []
    for arm, destination in zip(audit["arms"], destinations):
        if _sha256(Path(arm["checkpoint"])) != arm["checkpoint_sha256"]:
            raise ValueError("raw research checkpoint hash mismatch")
        payload = torch.load(arm["checkpoint"], map_location="cpu", weights_only=False)
        payload["report"] = {
            "input_contract": input_contract("uniform-time"), "not_for_model_selection": True,
            "source_optimization_report": str(source.resolve()), "source_optimization_report_sha256": _sha256(source),
            "accuracy_scoring_performed": False, "crohme_rows": 0, "heldout_rows_read": 0, "product_adopted": False,
        }
        torch.save(payload, destination)
        model, labels, _ = _load_teacher(destination, torch.device("cpu"))
        if len(labels) != 372 or not all(torch.equal(model.state_dict()[name], value) for name, value in payload["state_dict"].items()):
            raise AssertionError("runtime loading or metadata packaging changed model weights")
        packaged.append({"arm": arm["arm"], "checkpoint": str(destination.resolve()), "sha256": _sha256(destination), "runtime_contract_passed": True, "weights_bitwise_unchanged": True})
    _write_json(destination_report, {"schema": "aiflow-hwr-boundary-checkpoint-packaging/v1", "status": "pass", "checkpoints": packaged, "accuracy_scoring_performed": False, "product_adopted": False})
    print(json.dumps({"event": "integrity_checkpoint_packaging_complete", "checkpoints": len(packaged), "report": str(destination_report.resolve())}))
    return 0


def _run(args) -> int:
    if _guard_commit("before_boundary_optimization_import") is None:
        return 78
    import numpy as np
    import torch
    import torch.nn.functional as F
    from hwr_boundary_distillation_v1 import boundary_kl_loss, boundary_training_loss, load_boundary_candidates
    from run_hwr_affine_distillation_experiment_v1 import _augment_affine, _load_teacher, _predict_logits

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite smoke-test artifacts: {args.output_dir}")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cpu")
    checkpoint_sha = _sha256(args.checkpoint)
    manifest_path = args.data_dir / "prepared_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "pass" or manifest.get("current_checkpoint", {}).get("sha256") != checkpoint_sha:
        raise ValueError("admitted training-cache audit/checkpoint mismatch")
    if manifest.get("input_policy", {}).get("admitted_train") != "HWRT curated official train split + UJI Pen v2 official writer-disjoint train split":
        raise ValueError("training cache has not passed the admitted source policy")
    # Deliberately never load test_* files, official test sources, or project acceptance data.
    train_paths = {name: args.data_dir / f"train_{name}.npy" for name in ("features", "labels", "sources")}
    train = {name: np.load(path, mmap_mode="r", allow_pickle=False) for name, path in train_paths.items()}
    teacher, labels, _ = _load_teacher(args.checkpoint, device)
    boundary_report, boundary_x, boundary_logits = load_boundary_candidates(args.boundary_report, checkpoint_sha, labels)
    y, sources = train["labels"], train["sources"]
    if train["features"].shape != (len(y), 128, 5) or sources.shape != y.shape:
        raise ValueError("training cache tensors are not aligned")
    real_mask = np.isin(sources, (0, 1))
    real_classes = np.unique(y[real_mask])
    if len(real_classes) != 371 or np.any(y[real_mask] < 0) or np.any(y[real_mask] >= 372):
        raise ValueError("expected all 371 real-supported classes; synthetic-only '=' is excluded")
    rng = np.random.default_rng(SEED)
    class_order = np.tile(rng.permutation(real_classes), (args.steps * args.batch_size + 370) // 371)[:args.steps * args.batch_size]
    pools = {int(class_id): np.flatnonzero(real_mask & (y == class_id)) for class_id in real_classes}
    row_indices = np.array([rng.choice(pools[int(class_id)]) for class_id in class_order], dtype=np.int64)
    x = np.array(train["features"][row_indices], dtype=np.float32, copy=True)
    targets = np.array(y[row_indices], dtype=np.int64, copy=True)
    if not np.isfinite(x).all() or (x[..., :2] < 0.0).any() or (x[..., :2] > 1.0).any():
        raise ValueError("real training samples violate the input contract")
    teacher.eval()
    base_logits = _predict_logits(teacher, x, device, 32)
    initial_state = {name: value.detach().clone() for name, value in teacher.state_dict().items()}
    bx, bt = torch.from_numpy(boundary_x.copy()), torch.from_numpy(boundary_logits.copy())
    supervision_x, supervision_y, supervision_teacher = torch.from_numpy(x), torch.from_numpy(targets), torch.from_numpy(base_logits)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "schema": "aiflow-hwr-boundary-optimization-plan/v1",
        "fixed_before_optimizer_steps": True, "seed": SEED, "steps_per_arm": args.steps,
        "batch_size": args.batch_size, "learning_rate": args.learning_rate,
        "boundary_weight": args.boundary_weight, "boundary_temperature": 1.0, "boundary_policy": "full",
        "base_loss": "0.7 CE on real train labels + 0.3 T^2 KL at T=2 on unaugmented frozen logits",
        "arms": {"base": "L_base", "scaled_base": "(1-w)*L_base", "boundary": "(1-w)*L_base + w*L_boundary (current student mode)", "boundary_deterministic": "(1-w)*L_base + w*L_boundary (replay dropout off; autograd on)"},
        "teacher_checkpoint_sha256": checkpoint_sha, "script_sha256": _sha256(Path(__file__)),
        "source_manifest_sha256": _sha256(manifest_path), "boundary_report_sha256": _sha256(args.boundary_report),
        "training_inputs": {name: {"path": str(path.resolve()), "sha256": _sha256(path)} for name, path in train_paths.items()},
        "sampled_rows": len(row_indices), "real_classes_sampled": int(len(np.unique(targets))),
        "sampled_rows_by_source": {name: int(np.sum(sources[row_indices] == source_id)) for name, source_id in (("hwrt", 0), ("uji", 1))},
        "training_row_indices_sha256": hashlib.sha256(row_indices.tobytes()).hexdigest(),
        "boundary_rows": len(boundary_x), "boundary_scope": "class-2 teacher-boundary pilot only",
        "hard_synthetic_labels_assigned": 0, "human_boundary_labels_available": False,
        "heldout_rows_read": 0, "crohme_rows": 0, "accuracy_scoring_performed": False,
        "eligible_for_model_selection": False, "product_adopted": False,
        "limitations": "Training/integrity only; teacher-seen data. Neither accuracy improvement nor human recognition-boundary expansion can be inferred.",
    }
    _write_json(args.output_dir / "frozen_plan.json", plan)
    np.save(args.output_dir / "training_row_indices.npy", row_indices, allow_pickle=False)
    results = []
    audit_steps = {1, max(1, args.steps // 4), max(1, args.steps // 2), args.steps}
    for arm in plan["arms"]:
        if _guard_commit(f"before_optimization_arm_{arm}") is None:
            return 78
        model = copy.deepcopy(teacher)
        if not all(torch.equal(model.state_dict()[name], tensor) for name, tensor in initial_state.items()):
            raise AssertionError("paired arms did not begin at identical weights")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1.0e-4)
        replay_generator = torch.Generator().manual_seed(SEED + 74)
        history = []
        started = time.perf_counter()
        for step in range(1, args.steps + 1):
            model.train()
            # Reset the base dropout stream identically for each paired optimizer step.
            torch.manual_seed(SEED + step)
            start = (step - 1) * args.batch_size
            batch = supervision_x[start:start + args.batch_size]
            augmented, augmentation = _augment_affine(batch, torch.Generator().manual_seed(SEED + 1000 + step))
            optimizer.zero_grad(set_to_none=True)
            logits = model.math_head(model.encode(augmented))
            hard = F.cross_entropy(logits, supervision_y[start:start + args.batch_size])
            distill = boundary_kl_loss(logits, supervision_teacher[start:start + args.batch_size], 2.0, "full")
            base_loss = 0.7 * hard + 0.3 * distill
            boundary_loss = base_loss.new_zeros(())
            alignment = None
            if arm.startswith("boundary"):
                student_mode = "deterministic" if arm == "boundary_deterministic" else "current"
                loss, boundary_loss, _ = boundary_training_loss(base_loss, model, bx, bt, replay_generator, args.boundary_weight, 16, 1.0, "full", student_mode)
                if step in audit_steps:
                    named = list(model.named_parameters())
                    parameters = [parameter for _, parameter in named]
                    base_gradients = torch.autograd.grad((1.0 - args.boundary_weight) * base_loss, parameters, retain_graph=True, allow_unused=True)
                    boundary_gradients = torch.autograd.grad(args.boundary_weight * boundary_loss, parameters, retain_graph=True, allow_unused=True)
                    alignment = _layer_gradient_alignment(named, base_gradients, boundary_gradients)
            else:
                loss = base_loss if arm == "base" else (1.0 - args.boundary_weight) * base_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite {arm} loss at step {step}")
            loss.backward()
            if not all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None):
                raise FloatingPointError(f"nonfinite {arm} gradients at step {step}")
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
                raise FloatingPointError(f"nonfinite {arm} parameters at step {step}")
            trace = {
                "step": step, "hard_loss": float(hard.detach()), "base_distill_loss": float(distill.detach()),
                "boundary_loss": float(boundary_loss.detach()), "total_loss": float(loss.detach()),
                "gradient_l2_before_clip": float(gradient_norm), "augmentation": augmentation,
                "layer_gradient_alignment": alignment,
            }
            if step in audit_steps:
                # Distribution-consistency microscope, NOT correctness/accuracy scoring.
                replay_logits = _predict_logits(model, boundary_x, device, 16)
                trace["all_boundary_teacher_kl_t1"] = float(boundary_kl_loss(torch.from_numpy(replay_logits), bt, 1.0, "full"))
                print(json.dumps({"event": "boundary_optimization_progress", "arm": arm, "step": step, "steps": args.steps, "hard_loss": trace["hard_loss"], "boundary_teacher_kl": trace["all_boundary_teacher_kl_t1"]}), flush=True)
            history.append(trace)
        changed_tensors = sum(not torch.equal(model.state_dict()[name], value) for name, value in initial_state.items())
        if not changed_tensors:
            raise AssertionError("optimizer made no parameter updates")
        checkpoint = args.output_dir / f"{arm}_smoke_checkpoint.pt"
        torch.save({"schema": "aiflow-hwr-boundary-integrity-student/v1", "state_dict": model.state_dict(), "math_labels": labels, "auxiliary_labels": [], "input_mode": "uniform-time", "not_for_model_selection": True, "plan": plan}, checkpoint)
        results.append({"arm": arm, "optimizer_steps": args.steps, "changed_parameter_tensors": changed_tensors, "seconds": time.perf_counter() - started, "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": _sha256(checkpoint), "history": history})
        del model, optimizer
    if _sha256(args.checkpoint) != checkpoint_sha:
        raise AssertionError("canonical teacher checkpoint was modified")
    result = {"schema": "aiflow-hwr-boundary-optimization-audit/v1", "status": "training_integrity_smoke_pass", "plan": plan, "arms": results, "total_optimizer_steps": args.steps * len(results), "canonical_checkpoint_unchanged": True, "accuracy_scoring_performed": False, "eligible_for_model_selection": False, "heldout_rows_read": 0, "crohme_rows": 0, "product_adopted": False}
    destination = args.output_dir / "optimization_microscope.json"
    _write_json(destination, result)
    print(json.dumps({"event": "boundary_optimization_complete", "report": str(destination.resolve()), "optimizer_steps": result["total_optimizer_steps"], "accuracy_scoring_performed": False}), flush=True)
    return 0


def main() -> int:
    import math

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("self-test", "audit", "package"), required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--boundary-report", type=Path, default=DEFAULT_BOUNDARY_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--boundary-weight", type=float, default=0.1)
    args = parser.parse_args()
    if not 1 <= args.steps <= 256 or not 1 <= args.batch_size <= 64:
        parser.error("smoke test requires 1..256 steps and batch size 1..64")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0.0:
        parser.error("learning rate must be positive and finite")
    if not math.isfinite(args.boundary_weight) or not 0.0 < args.boundary_weight < 1.0:
        parser.error("boundary weight must be finite in (0,1)")
    if args.mode == "self-test":
        return _self_test()
    if args.mode == "package":
        return _package(args)
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
