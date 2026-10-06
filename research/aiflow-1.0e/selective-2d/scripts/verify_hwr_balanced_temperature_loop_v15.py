"""전 범위 A/B의 데이터 분리·균형·층 전달·checkpoint 출력과 확대 결정을 재검증한다."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import run_hwr_balanced_temperature_loop_v15 as run


def main() -> int:
    """저장 지표를 믿지 않고 현재 checkpoint와 실제 TRAIN 원본에서 재현한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=(0, 1, 2), default=0)
    args = parser.parse_args()
    if run.aug.base._guard_commit("before_balanced_loop_verification") is None:
        return 78
    out = run.OUTPUT / f"stage_{args.stage}"
    if (out / "independent_verification.json").exists():
        raise FileExistsError("refusing certificate overwrite")
    global_plan, arrays, folds, excluded = run.global_load(run.OUTPUT)
    plan = json.loads((out / "frozen_plan.json").read_text(encoding="utf-8"))
    result = json.loads((out / "stage_result.json").read_text(encoding="utf-8"))
    if result["plan_sha256"] != run._sha(out / "frozen_plan.json"):
        raise ValueError("result plan differs")
    for name, digest in plan["artifact_sha256"].items():
        suffix = ".json" if name == "fit_query_provenance" else ".npy"
        if run._sha(out / f"{name}{suffix}") != digest:
            raise ValueError("optimization data changed")
    ids = np.load(out / "train_indices.npy", allow_pickle=False)
    x = np.load(out / "train_features.npy", allow_pickle=False)
    y = np.load(out / "train_labels.npy", allow_pickle=False)
    views = np.load(out / "augmented_views.npy", allow_pickle=False)
    counts = np.load(out / "view_counts.npy", allow_pickle=False)
    forbidden = np.concatenate((*folds, excluded))
    hashes = {run.ink_hash(arrays["features"][int(i)]) for i in forbidden}
    if not np.array_equal(x, arrays["features"][ids]) or not np.array_equal(y, arrays["labels"][ids]):
        raise ValueError("original real features or labels differ")
    if np.isin(ids, forbidden).any() or any(run.ink_hash(row) in hashes for row in x):
        raise ValueError("held fold/prior diagnostic optimization overlap")
    groups = json.loads((out / "fit_query_provenance.json").read_text(encoding="utf-8"))
    for group in groups:
        fit = np.array(group["fit_training_indices"], dtype=np.int64)
        if np.isin(fit, np.concatenate((forbidden, ids))).any() or any(run.ink_hash(arrays["features"][int(i)]) in hashes for i in fit):
            raise ValueError("covariance/donor leakage")
        if not np.isin(arrays["sources"][fit], (0, 1)).all() or not (arrays["labels"][fit] == group["source_class_id"]).all():
            raise ValueError("fit source class differs")
    if any(not run.aug.base._geometry(x[i], views[i, j])["valid"] for i in range(len(x)) for j in range(int(counts[i]))):
        raise ValueError("augmented geometry invalid")
    batches = np.load(out / "shared_class_balanced_batches.npy", allow_pickle=False)
    if batches.shape != (plan["steps"], run.BATCH) or batches.min() < 0 or batches.max() >= len(x):
        raise ValueError("batch contract differs")
    classes, frequency = np.unique(y[batches], return_counts=True)
    if len(classes) != 371 or int(frequency.max() - frequency.min()) > 1:
        raise ValueError("class-balanced schedule differs")
    import torch
    from run_hwr_affine_distillation_experiment_v1 import _load_teacher, _predict_logits
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    device = torch.device("cpu")
    reference, labels, _ = _load_teacher(run.CHECKPOINT, device)
    vx = np.array(arrays["features"][folds[args.stage]], copy=True)
    vy = np.array(arrays["labels"][folds[args.stage]], copy=True)
    metrics, changed_layers = {}, {}
    for arm in ("canonical", *run.ARMS):
        path = run.CHECKPOINT if arm == "canonical" else out / f"{arm}.pt"
        if arm != "canonical" and run._sha(path) != result["checkpoint_sha256"][arm]:
            raise ValueError("checkpoint changed")
        model, model_labels, _ = _load_teacher(path, device)
        if model_labels != labels:
            raise ValueError("vocabulary differs")
        logits = _predict_logits(model, vx, device, run.BATCH)
        saved = np.load(out / f"{arm}_validation_logits.npy", allow_pickle=False)
        if not np.array_equal(logits, saved):
            raise ValueError("reloaded checkpoint logits not bit-exact")
        # stable Top-k를 target보다 큰 값과 동점의 낮은 class ID 개수로 별도 계산한다.
        target = saved[np.arange(len(vy)), vy, None]
        rank = 1 + (saved > target).sum(1) + ((saved == target) & (np.arange(372)[None] < vy[:, None])).sum(1)
        metrics[arm] = {}
        for family in ("all", "digits", "latin_letters", "math_symbols"):
            mask = np.ones(len(vy), dtype=bool) if family == "all" else np.array([run.aug.base._family(labels[int(c)]) == family for c in vy])
            metrics[arm][family] = dict(rows=int(mask.sum()), top1_hits=int((rank[mask] == 1).sum()), top5_hits=int((rank[mask] <= 5).sum()))
        if arm != "canonical":
            history = json.loads((out / f"{arm}_training_microscope.json").read_text(encoding="utf-8"))
            if len(history) != plan["steps"] or [row["step"] for row in history] != list(range(1, plan["steps"] + 1)):
                raise ValueError("missing per-step microscope records")
            layers = [f"encoder.layers.{i}" for i in range(4)]
            for row in history:
                if set(row["real_encoder_layers"]) != set(layers) or set(row["view_encoder_layers"]) != set(layers):
                    raise ValueError("missing encoder transfer records")
                if not all(np.isfinite(row[key]) for key in ("loss", "ce_real", "kl_retention", "kl_view", "gradient_l2", "embedding_std")):
                    raise ValueError("nonfinite optimization record")
                if any(not any(value is not None and value > 0 for name, value in row["parameter_gradient_l2"].items() if name.startswith(layer + ".")) for layer in layers):
                    raise ValueError("encoder did not receive gradients")
            before, after = reference.state_dict(), model.state_dict()
            changed_layers[arm] = {layer: sum(not torch.equal(before[name], after[name]) for name in before if name.startswith(layer + ".")) for layer in layers}
            if not all(changed_layers[arm].values()):
                raise ValueError("an encoder layer was silently frozen")
    if metrics != result["metrics"] or result["next_stage_allowed"] != (run.growth_gate(metrics) and args.stage < 2):
        raise ValueError("independent metrics/growth decision differ")
    certificate = dict(schema="aiflow-balanced-temperature-verification/v15", status="reproduced",
        verifier_sha256=run._sha(Path(__file__)), result_sha256=run._sha(out / "stage_result.json"),
        checkpoint_logits_reload_bit_exact=True, independent_stable_ranks_reproduced=True, geometry_errors=0,
        prior_and_all_three_fold_fit_query_index_or_rounded_ink_overlap=0, balanced_class_frequency_min=int(frequency.min()),
        balanced_class_frequency_max=int(frequency.max()), all_steps_and_four_encoder_gradient_transfers_verified=True,
        changed_encoder_tensors_by_layer=changed_layers, next_stage_allowed=result["next_stage_allowed"],
        canonical_unchanged=run._sha(run.CHECKPOINT) == global_plan["canonical_sha256"],
        independent_writer_device_acceptance=False, product_adopted=False, official_test_rows_read=0, crohme_rows=0)
    run._write(out / "independent_verification.json", certificate)
    print(json.dumps(dict(event="balanced_loop_verification", **certificate)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
