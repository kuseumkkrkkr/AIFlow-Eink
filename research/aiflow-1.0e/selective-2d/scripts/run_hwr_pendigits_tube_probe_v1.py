"""PenDigits 학습 그룹 내부에서 q68 획 변형의 증류 효과를 짝지어 검증한다.

공식 test, CROHME, 수식 acceptance는 읽지 않는다. 사람 인지 경계 실험이 아니다.
prepare에서 설정/그룹/변형을 봉인하고, run에서 두 학습 종료 후 한 번 평가한다.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

import audit_hwr_probability_boundary_tube_v1 as tube

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/hwr_pendigits_source_audit_20261005"
OUTPUT = ROOT / "artifacts/hwr_pendigits_tube_probe_20261005"
SEED = 20261005
CHECKPOINT_SHA = "04f8608aebcf6c02d45ad6f5735229b9eaa2c4b4e1be0db4793d02273ef2d00e"


def _sha(path: Path) -> str:
    """파일을 청크 단위로 읽어 봉인된 바이트의 SHA를 계산한다."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path: Path, value: dict) -> None:
    """기존 결과를 덮지 않고 UTF-8 JSON 증거를 추가한다."""
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=True, indent=2)
        stream.write("\n")


def _source() -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """이미지 QA와 모든 학습 배열 SHA가 맞는 격리 소스만 읽는다."""
    report_path = SOURCE / "pendigits_source_audit.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    qa = json.loads((SOURCE / "orientation_and_integrity_qa.json").read_text(encoding="utf-8"))
    if _sha(report_path) != qa["source_report_sha256"] or _sha(SOURCE / "train_orientation_qa.png") != qa["qa_image_sha256"]:
        raise ValueError("source/visual QA hashes differ")
    if report["source_license_displayed"] != "CC BY 4.0" or report["test_payloads_opened"] or report["crohme_rows"]:
        raise ValueError("source eligibility boundary failed")
    if qa["visual_review"]["chosen_candidate_feature_file"] != "train_features_y_up.npy":
        raise ValueError("orientation QA does not support selected candidate")
    arrays = {}
    for name in ("train_features_y_up", "train_digit_labels", "train_writer_group_hashes"):
        info = report["artifacts"][name]
        path = SOURCE / f"{name}.npy"
        if _sha(path) != info["sha256"]:
            raise ValueError(f"sealed source array changed: {name}")
        arrays[name] = np.load(path, allow_pickle=False)
    x, y, groups = (arrays[name] for name in ("train_features_y_up", "train_digit_labels", "train_writer_group_hashes"))
    if x.shape != (7494, 128, 5) or y.shape != groups.shape or y.shape != (7494,) or len(np.unique(groups)) != 30:
        raise ValueError("source row/group contract failed")
    if not np.isfinite(x).all() or x[..., :2].min() < 0 or x[..., :2].max() > 1 or set(y.tolist()) != set(range(10)):
        raise ValueError("source tensor/label contract failed")
    return report, x, y, groups


def _ink_hash(row: np.ndarray) -> str:
    """시각 좌표와 획 시작을 여섯 자리로 봉인해 입력 중복을 검사한다."""
    ink = np.round(row[:, (0, 1, 3)], 6).astype("<f4")
    return hashlib.sha256(ink.tobytes()).hexdigest()


def _render_qa(original: np.ndarray, changed: np.ndarray, labels: np.ndarray, source_ids: np.ndarray, path: Path) -> None:
    """원본/변형의 축척을 같게 맞추고 실제 학습 행 ID를 함께 표시한다."""
    from PIL import Image, ImageDraw
    image = Image.new("RGB", (1060, 1240), "white")
    draw = ImageDraw.Draw(image)
    draw.text((12, 8), "TRAIN GROUPS ONLY | original / q68 deformation | equal XY scale | no human boundary labels", fill="black")
    for digit in range(10):
        for column, index in enumerate(np.flatnonzero(labels == digit)):
            for side, features in enumerate((original, changed)):
                left, top = 12 + column * 350 + side * 170, 40 + digit * 118
                draw.text((left, top), f"{digit}: {'original' if side == 0 else 'tube'} row{source_ids[index]}", fill="black")
                values = features[index]
                starts = list(np.flatnonzero(values[:, 3] > 0.5)) + [128]
                for begin, end in zip(starts, starts[1:]):
                    points = [(left + 25 + float(p[0]) * 84, top + 20 + float(p[1]) * 84) for p in values[begin:end]]
                    if len(points) > 1:
                        draw.line(points, fill="black", width=2)
                    if points:
                        px, py = points[0]
                        draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill="red")
    image.save(path)


def _prepare(out: Path) -> int:
    """24개 학습/6개 보류 그룹을 고정하고 학습 그룹에서만 변형을 만든다."""
    if out.exists():
        raise FileExistsError("refusing to overwrite a source-local experiment")
    if tube._guard_commit("before_pendigits_probe_prepare") is None:
        return 78
    tube.np = np
    report, x, y, groups = _source()
    if _sha(tube.DEFAULT_CHECKPOINT) != CHECKPOINT_SHA:
        raise ValueError("canonical checkpoint changed")
    rng = np.random.default_rng(SEED)
    held_groups = np.sort(rng.permutation(np.unique(groups))[:6])
    held = np.flatnonzero(np.isin(groups, held_groups))
    train = np.flatnonzero(~np.isin(groups, held_groups))
    if set(groups[train]) & set(groups[held]) or len(train) + len(held) != len(x):
        raise AssertionError("inner writer-group split leaked")
    train_hashes = {_ink_hash(x[i]) for i in train}
    if any(_ink_hash(x[i]) in train_hashes for i in held):
        raise ValueError("rounded input duplicate crosses the inner split")
    replay_manifest = json.loads((tube.DEFAULT_DATA_DIR / "prepared_manifest.json").read_text(encoding="utf-8"))
    if replay_manifest.get("status") != "pass" or replay_manifest["current_checkpoint"]["sha256"] != CHECKPOINT_SHA:
        raise ValueError("replay cache/checkpoint audit differs")
    if replay_manifest["input_policy"]["admitted_train"] != "HWRT curated official train split + UJI Pen v2 official writer-disjoint train split":
        raise ValueError("replay source policy differs")
    old_x = np.load(tube.DEFAULT_DATA_DIR / "train_features.npy", mmap_mode="r", allow_pickle=False)
    old_y = np.load(tube.DEFAULT_DATA_DIR / "train_labels.npy", mmap_mode="r", allow_pickle=False)
    old_s = np.load(tube.DEFAULT_DATA_DIR / "train_sources.npy", mmap_mode="r", allow_pickle=False)
    # 공식 test 또는 별도 평가 derivative를 중복 검사 명목으로도 열지 않는다.
    held_hashes = {_ink_hash(x[i]) for i in held}
    collisions = sum(_ink_hash(row) in held_hashes for row in old_x)
    if collisions:
        raise ValueError("held inputs collide with existing train/replay inputs")
    eligible = np.isin(old_s, (0, 1))
    classes = np.unique(old_y[eligible])
    if len(classes) != 371:
        raise ValueError("expected 371 real-supported replay classes")
    class_order = np.tile(rng.permutation(classes), 3)[:1024]
    pools = {int(c): np.flatnonzero(eligible & (old_y == c)) for c in classes}
    replay_ids = np.array([rng.choice(pools[int(c)]) for c in class_order], dtype=np.int64)
    replay_x = np.array(old_x[replay_ids], copy=True)
    original = x[train].copy()
    augmented = original.copy()
    singles = tube._single_stroke_rows(original)
    geometry = []
    fit_source_ids, donor_source_ids = {}, {}
    for digit in range(10):
        ids = np.flatnonzero(singles & (y[train] == digit))
        if len(ids) < tube.MIN_FIT_ROWS:
            raise ValueError(f"digit {digit} lacks single-stroke fit support")
        fit_ids = rng.choice(ids, min(256, len(ids)), replace=False)
        metric = tube._fit_empirical_tube(original[fit_ids], 0.68)
        donors, donor_info = tube._select_empirical_donors(original[fit_ids], original[ids], metric)
        available = ~donor_info["orientation_filter_fallback"]
        accepted = np.zeros(len(ids), dtype=bool)
        scales = np.zeros(len(ids), dtype=np.float64)
        for scale in (0.25, 0.125, 0.0625, 0.03125):
            values, diag = tube._apply_tube(original[ids], donors, donor_info["max_interpolation_fraction"], scale)
            valid = available & diag["valid"] & (diag["rms"] <= 0.035) & (diag["max_point_displacement"] <= 0.105) & (diag["rms"] > 1.0e-7)
            selected = valid & ~accepted
            augmented[ids[selected]] = values[selected]
            scales[selected] = scale
            accepted |= selected
        fit_source_ids[str(digit)] = train[fit_ids].tolist()
        donor_source_ids[str(digit)] = train[fit_ids[donor_info["donor_fit_row_indices"]]].tolist()
        geometry.append({"digit": digit, "single_stroke_queries": len(ids), "fit_rows": len(fit_ids), "components": metric["component_count"],
                         "explained_variance": metric["explained_variance_ratio"], "q68_pairwise_radius": metric["pairwise_q68_radius"],
                         "accepted_changed_rows": int(accepted.sum()), "direction_fallback_rows_rejected": int((~available).sum()),
                         "scales_accepted": {str(s): int((scales == s).sum()) for s in (0.25, 0.125, 0.0625, 0.03125)}})
        print(json.dumps({"event": "pendigits_train_tube_geometry", **geometry[-1]}), flush=True)
    if not np.array_equal(original[..., 2:], augmented[..., 2:]) or not np.isfinite(augmented).all():
        raise AssertionError("augmentation modified time/topology or produced NaN")
    delta = augmented[..., :2] - original[..., :2]
    if augmented[..., :2].min() < 0 or augmented[..., :2].max() > 1:
        raise AssertionError("augmentation left canvas")
    out.mkdir(parents=True)
    paths = {}
    for name, array in {"train_indices": train, "held_indices": held, "train_augmented": augmented, "replay_indices": replay_ids, "replay_features": replay_x}.items():
        path = out / f"{name}.npy"
        np.save(path, array, allow_pickle=False)
        paths[name] = {"file": path.name, "sha256": _sha(path), "shape": list(array.shape)}
    changed = np.sqrt(np.mean(delta ** 2, axis=(1, 2))) > 1.0e-7
    qa_ids = np.concatenate([np.flatnonzero((y[train] == d) & changed)[:3] for d in range(10)])
    if len(qa_ids) != 30:
        raise ValueError("all ten digits need three changed visual samples")
    _render_qa(original[qa_ids], augmented[qa_ids], y[train][qa_ids], train[qa_ids], out / "train_tube_qa.png")
    plan = {"schema": "aiflow-pendigits-tube-probe-plan/v1", "seed": SEED, "steps": 64, "batch_size": 32, "replay_batch_size": 16, "learning_rate": 1.0e-5,
            "checkpoint_sha256": CHECKPOINT_SHA, "script_sha256": _sha(Path(__file__)), "source_report_sha256": _sha(SOURCE / "pendigits_source_audit.json"),
            "qa_sha256": _sha(out / "train_tube_qa.png"), "artifacts": paths,
            "train_rows": len(train), "held_rows": len(held), "train_groups": sorted(set(groups[train])), "held_groups": held_groups.tolist(),
            "rounded_input_collisions_inner_split": 0, "rounded_input_collisions_held_vs_replay_cache": collisions,
            "augmentation": geometry, "fit_source_indices": fit_source_ids, "donor_source_indices": donor_source_ids,
            "changed_rows": int(changed.sum()), "rms_max": float(np.sqrt(np.mean(delta ** 2, axis=(1, 2))).max()),
            "replay_cache_manifest_sha256": _sha(tube.DEFAULT_DATA_DIR / "prepared_manifest.json"),
            "replay_train_array_hashes": {n: _sha(tube.DEFAULT_DATA_DIR / f"train_{n}.npy") for n in ("features", "labels", "sources")},
            "arms": {"original_consistency": "0.9 * (0.7 CE(real PenDigits) + 0.3 KL(old real train replay,T=2)) + 0.1 KL(original PenDigits, frozen original teacher,T=1)",
                     "tube_consistency": "same base and weights; final KL uses q68-capped correlated deformation instead of original ink"},
            "initialization": "identical canonical checkpoint; all 372 logits retained", "student_consistency_dropout": "off, autograd on, module modes restored",
            "synthetic_hard_labels": 0, "human_boundary_labels": 0, "tube_metric_uses_held_rows": False,
            "source_role": "source-local digits-only research pretraining/adaptation; canonical pool unchanged",
            "writer_group_identity_limit": "comment field inferred; independence from writers of prior corpora is not verified",
            "validation_policy": "no logits/accuracy until both fixed-step arms finish; evaluate six held inner train groups once; no best epoch/threshold/model selection",
            "official_test_rows_read": 0, "crohme_rows": 0, "product_adopted": False,
            "limit": "Not official PenDigits test, human recognition probabilities, whole-math/formula acceptance or release evidence."}
    _write(out / "frozen_plan.json", plan)
    print(json.dumps({"event": "pendigits_tube_plan_frozen", "train_rows": len(train), "held_rows": len(held), "changed_rows": plan["changed_rows"]}), flush=True)
    return 0


def _metrics(logits: np.ndarray, targets: np.ndarray, digit_ids: list[int]) -> dict:
    """372개 클래스 전체 경쟁으로 Top-1/5와 숫자 2의 과잉 인식을 측정한다."""
    if logits.shape != (len(targets), 372) or not np.isfinite(logits).all():
        raise ValueError("prediction contract failed")
    prediction = logits.argmax(1)
    hit5 = (np.argsort(-logits, axis=1, kind="stable")[:, :5] == targets[:, None]).any(1)
    truth2 = targets == digit_ids[2]
    return {"rows": len(targets), "top1_hits": int((prediction == targets).sum()), "top5_hits": int(hit5.sum()),
            "top1": float((prediction == targets).mean()), "top5": float(hit5.mean()),
            "non2_rows": int((~truth2).sum()), "non2_predicted2": int(((prediction == digit_ids[2]) & ~truth2).sum()),
            "by_digit": {str(d): {"rows": int((targets == i).sum()), "top1_hits": int(((targets == i) & (prediction == targets)).sum()),
                                       "top5_hits": int(((targets == i) & hit5).sum())} for d, i in enumerate(digit_ids)}}


def _paired(before: np.ndarray, after: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict:
    """같은 보류 획의 정오 차이를 여섯 그룹 cluster-bootstrap으로 비교한다."""
    a, b = before.argmax(1) == y, after.argmax(1) == y
    unique = np.unique(groups)
    delta = np.array([int(b[groups == g].sum()) - int(a[groups == g].sum()) for g in unique])
    counts = np.array([int((groups == g).sum()) for g in unique])
    draw = np.random.default_rng(SEED + 991).integers(len(unique), size=(20000, len(unique)))
    boot = delta[draw].sum(1) / counts[draw].sum(1)
    return {"rescued": int((~a & b).sum()), "regressed": int((a & ~b).sum()), "delta_top1": float(b.mean() - a.mean()),
            "cluster_bootstrap_95_percentile": np.quantile(boot, [0.025, 0.975]).tolist(), "groups": len(unique), "draws": 20000,
            "limit": "six source-local inferred groups; not product acceptance or selection"}


def _run(out: Path) -> int:
    """봉인한 동일 배치 학습을 마친 뒤 보류 그룹의 정확도를 단 한 번 계산한다."""
    if (out / "run_started.json").exists():
        raise FileExistsError("experiment already started; inspect its live/terminal evidence, do not rerun")
    plan_path = out / "frozen_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if _sha(Path(__file__)) != plan["script_sha256"] or _sha(tube.DEFAULT_CHECKPOINT) != plan["checkpoint_sha256"]:
        raise ValueError("frozen code/checkpoint differs")
    if _sha(out / "train_tube_qa.png") != plan["qa_sha256"]:
        raise ValueError("visual QA changed")
    if tube._guard_commit("before_pendigits_probe_train") is None:
        return 78
    import torch
    from torch.nn import functional as F
    from hwr_boundary_distillation_v1 import boundary_kl_loss
    from run_hwr_affine_distillation_experiment_v1 import _load_teacher, _predict_logits
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cpu")
    _, x, digits, groups = _source()
    if _sha(SOURCE / "pendigits_source_audit.json") != plan["source_report_sha256"]:
        raise ValueError("source report changed")
    if _sha(tube.DEFAULT_DATA_DIR / "prepared_manifest.json") != plan["replay_cache_manifest_sha256"]:
        raise ValueError("replay manifest changed")
    for name, value in plan["replay_train_array_hashes"].items():
        if _sha(tube.DEFAULT_DATA_DIR / f"train_{name}.npy") != value:
            raise ValueError("replay source array changed")
    arrays = {}
    for name, info in plan["artifacts"].items():
        if _sha(out / info["file"]) != info["sha256"]:
            raise ValueError("prepared split/augmentation changed")
        arrays[name] = np.load(out / info["file"], allow_pickle=False)
    train, held = arrays["train_indices"], arrays["held_indices"]
    if set(groups[train]) & set(groups[held]) or sorted(set(groups[held])) != plan["held_groups"]:
        raise ValueError("held group split changed")
    _write(out / "run_started.json", {"plan_sha256": _sha(plan_path), "status": "started", "official_test_rows_read": 0})
    teacher, labels, _ = _load_teacher(tube.DEFAULT_CHECKPOINT, device)
    digit_ids = [labels.index(str(d)) for d in range(10)]
    targets = torch.tensor([digit_ids[int(d)] for d in digits[train]], dtype=torch.long)
    original = torch.from_numpy(x[train].copy())
    changed = torch.from_numpy(arrays["train_augmented"])
    replay = torch.from_numpy(arrays["replay_features"])
    teacher_source = torch.from_numpy(_predict_logits(teacher, x[train], device, 32))
    teacher_replay = torch.from_numpy(_predict_logits(teacher, arrays["replay_features"], device, 32))
    rng = np.random.default_rng(SEED + 11)
    batch_ids = rng.integers(len(train), size=(plan["steps"], plan["batch_size"]))
    replay_ids = rng.integers(len(replay), size=(plan["steps"], plan["replay_batch_size"]))
    trained, histories = {}, {}
    for arm in plan["arms"]:
        if tube._guard_commit(f"before_pendigits_arm_{arm}") is None:
            return 78
        model = copy.deepcopy(teacher)
        optimizer = torch.optim.AdamW(model.parameters(), lr=plan["learning_rate"], weight_decay=1.0e-4)
        history = []
        for step in range(plan["steps"]):
            model.train()
            torch.manual_seed(SEED + 1000 + step)
            bi, ri = torch.from_numpy(batch_ids[step]), torch.from_numpy(replay_ids[step])
            optimizer.zero_grad(set_to_none=True)
            joined = torch.cat((original[bi], replay[ri]))
            logits = model.math_head(model.encode(joined))
            hard = F.cross_entropy(logits[:len(bi)], targets[bi])
            retention = boundary_kl_loss(logits[len(bi):], teacher_replay[ri], 2.0, "full")
            modes = [(module, module.training) for module in model.modules()]
            try:
                model.eval()
                view = changed[bi] if arm == "tube_consistency" else original[bi]
                view_logits = model.math_head(model.encode(view))
            finally:
                for module, mode in modes:
                    module.training = mode
            consistency = boundary_kl_loss(view_logits, teacher_source[bi], 1.0, "full")
            loss = 0.9 * (0.7 * hard + 0.3 * retention) + 0.1 * consistency
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite loss")
            loss.backward()
            layer_gradients = None
            if step + 1 in (1, 16, 32, 64):
                layer_gradients = {name: float(parameter.grad.detach().norm()) for name, parameter in model.named_parameters() if parameter.grad is not None}
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(norm):
                raise FloatingPointError("nonfinite gradient")
            optimizer.step()
            record = {"step": step + 1, "ce_real": float(hard.detach()), "kl_retention": float(retention.detach()), "kl_consistency": float(consistency.detach()),
                      "total_loss": float(loss.detach()), "gradient_l2_before_clip": float(norm)}
            if layer_gradients is not None:
                record["layer_gradient_l2_before_clip"] = layer_gradients
            history.append(record)
            if (step + 1) % 16 == 0:
                print(json.dumps({"event": "pendigits_paired_train", "arm": arm, **record}), flush=True)
        checkpoint_path = out / f"{arm}.pt"
        torch.save({"schema": "aiflow-pendigits-tube-research-student/v1", "state_dict": model.state_dict(), "math_labels": labels, "auxiliary_labels": [],
                    "report": {"input_contract": {"observed_channel_mode": "uniform-time", "math_observed_transform": "uniform-time"}, "plan_sha256": _sha(plan_path), "product_adopted": False}}, checkpoint_path)
        trained[arm] = model
        histories[arm] = {"steps": history, "checkpoint_sha256": _sha(checkpoint_path)}
    # 학습/모델이 모두 확정된 이후에만 보류 그룹의 입력을 모델에 전달한다.
    _write(out / "held_evaluation_started.json", {"both_arms_finished": True, "plan_sha256": _sha(plan_path), "held_rows": len(held)})
    held_targets = np.array([digit_ids[int(d)] for d in digits[held]], dtype=np.int64)
    predictions = {"canonical": _predict_logits(teacher, x[held], device, 32)}
    predictions.update({name: _predict_logits(model, x[held], device, 32) for name, model in trained.items()})
    for name, logits in predictions.items():
        np.save(out / f"{name}_held_logits.npy", logits, allow_pickle=False)
    results = {name: _metrics(logits, held_targets, digit_ids) for name, logits in predictions.items()}
    paired = {"tube_vs_original": _paired(predictions["original_consistency"], predictions["tube_consistency"], held_targets, groups[held]),
              "original_vs_canonical": _paired(predictions["canonical"], predictions["original_consistency"], held_targets, groups[held]),
              "tube_vs_canonical": _paired(predictions["canonical"], predictions["tube_consistency"], held_targets, groups[held])}
    if _sha(tube.DEFAULT_CHECKPOINT) != CHECKPOINT_SHA:
        raise AssertionError("canonical checkpoint mutated")
    result = {"schema": "aiflow-pendigits-tube-probe-result/v1", "status": "completed", "plan_sha256": _sha(plan_path), "metrics": results, "paired": paired,
              "training_microscope": histories, "canonical_checkpoint_unchanged": True, "evaluated_inner_groups_now_consumed": plan["held_groups"],
              "synthetic_hard_labels": 0, "human_boundary_labels": 0, "official_test_rows_read": 0, "crohme_rows": 0, "product_adopted": False,
              "eligible_for_product_selection": False, "limitations": plan["limit"], "writer_group_identity_limit": plan["writer_group_identity_limit"]}
    _write(out / "probe_result.json", result)
    print(json.dumps({"event": "pendigits_tube_probe_complete", "metrics": results, "paired": paired}), flush=True)
    return 0


def _self_test() -> int:
    """전체 372-way 경쟁과 짝지은 정오/그룹 부트스트랩의 계수를 검사한다."""
    digit_ids = list(range(10))
    a = np.zeros((4, 372), dtype=np.float32)
    a[np.arange(4), [2, 2, 300, 3]] = 5
    y = np.array([2, 1, 4, 3])
    m = _metrics(a, y, digit_ids)
    assert m["top1_hits"] == 2 and m["non2_predicted2"] == 1
    b = a.copy()
    b[1, 1] = 6
    p = _paired(a, b, y, np.array(["a", "a", "b", "b"]))
    assert p["rescued"] == 1 and p["regressed"] == 0 and p["delta_top1"] == 0.25
    print(json.dumps({"self_test": "pass", "global_372_way_competition": True, "false_positive2_count": 1}))
    return 0


def main() -> int:
    """새 결과 폴더에 준비 또는 1회 실행만 허용하는 연구 CLI다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("self-test", "prepare", "run"), required=True)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    return _self_test() if args.mode == "self-test" else _prepare(args.output_dir) if args.mode == "prepare" else _run(args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
