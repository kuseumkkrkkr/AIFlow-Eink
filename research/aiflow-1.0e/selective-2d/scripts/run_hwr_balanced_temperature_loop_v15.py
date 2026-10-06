"""사용자 승인 범위의 전 클래스 실제 TRAIN + soft 증강 일관성 A/B 루프.

공식 test/CROHME/이전 보류 writer는 쓰지 않는다. 매 단계의 별도 TRAIN 내부
검증 fold를 사전에 고정하고 개선·family 비회귀를 통과할 때만 두 배 확대한다.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np

import audit_hwr_full_domain_direction_diversity_v14 as aug
from run_hwr_pendigits_tube_probe_v1 import _sha, _write

ROOT, DATA, CHECKPOINT = aug.base.ROOT, aug.base.DEFAULT_DATA_DIR, aug.base.DEFAULT_CHECKPOINT
OUTPUT = ROOT / "artifacts/hwr_balanced_temperature_loop_20261005_v15_dtype_fixed"
STAGES = ((.5, 12, 128), (.75, 24, 256), (1., 48, 512))
SEED, BATCH, VAL_PER_CLASS = 2026100515, 32, 8
ARMS = ("real_control", "soft_augmented")


def ink_hash(row: np.ndarray) -> str:
    """비좌표 채널 차이로 동일 잉크가 fold를 넘지 않도록 6자리 XY/획 시작을 해시한다."""
    from run_hwr_pendigits_tube_probe_v1 import _ink_hash
    return _ink_hash(row)


def sources() -> tuple[dict, dict]:
    """봉인된 실제 HWRT/UJI TRAIN만 읽고 테스트·teacher 캐시 logits는 읽지 않는다."""
    old = json.loads((aug.base.OUTPUT / "frozen_plan.json").read_text(encoding="utf-8"))
    manifest = json.loads((DATA / "prepared_manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "pass" or manifest["input_policy"]["admitted_train"] != "HWRT curated official train split + UJI Pen v2 official writer-disjoint train split":
        raise ValueError("TRAIN source admission differs")
    if _sha(CHECKPOINT) != old["provenance"]["teacher_checkpoint_sha256"] or _sha(DATA / "prepared_manifest.json") != old["provenance"]["prepared_manifest_sha256"]:
        raise ValueError("canonical/provenance changed")
    arrays = {}
    for name in ("features", "labels", "sources"):
        path = DATA / f"train_{name}.npy"
        if _sha(path) != old["provenance"][f"train_{name}_sha256"]:
            raise ValueError("TRAIN cache changed")
        arrays[name] = np.load(path, mmap_mode="r", allow_pickle=False)
    return old, arrays


def global_prepare(out: Path) -> None:
    """모델 실행 전에 세 개의 내부 검증 fold와 미래 온도/배수/중단 규칙을 고정한다."""
    if out.exists():
        return
    old, arrays = sources()
    x, y, s = (arrays[name] for name in ("features", "labels", "sources"))
    prior = ROOT / "artifacts/hwr_repair_complement_audit_20261005_v13"
    p = json.loads((prior / "frozen_plan.json").read_text(encoding="utf-8"))
    info = p["artifacts"]["old_indices"]
    if _sha(prior / info["file"]) != info["sha256"]:
        raise ValueError("previous consumed diagnostic IDs changed")
    blocked = set(np.load(prior / info["file"], allow_pickle=False).tolist())
    parent = ROOT / "artifacts/hwr_pendigits_tube_probe_20261005"
    parent_plan = json.loads((parent / "frozen_plan.json").read_text(encoding="utf-8"))
    item = parent_plan["artifacts"]["replay_indices"]
    if _sha(parent / item["file"]) != item["sha256"]:
        raise ValueError("previous optimization IDs changed")
    blocked.update(np.load(parent / item["file"], allow_pickle=False).tolist())
    classes = np.unique(y[np.isin(s, (0, 1))])
    if len(classes) != 371:
        raise ValueError("expected 371 real-supported classes")
    rng = np.random.default_rng(SEED)
    seen = {ink_hash(x[i]) for i in blocked}
    folds = [[] for _ in STAGES]
    for c in classes:
        pool = np.flatnonzero((y == c) & np.isin(s, (0, 1)))
        candidates = []
        for i in rng.permutation(pool):
            i = int(i)
            h = ink_hash(x[i])
            if i in blocked or h in seen:
                continue
            candidates.append(i); seen.add(h)
            if len(candidates) == len(STAGES) * VAL_PER_CLASS:
                break
        if len(candidates) != len(STAGES) * VAL_PER_CLASS:
            raise ValueError(f"class {c} lacks three nonoverlapping internal folds")
        for stage in range(len(STAGES)):
            folds[stage].extend(candidates[stage * VAL_PER_CLASS:(stage + 1) * VAL_PER_CLASS])
    out.mkdir(parents=True)
    paths = {}
    for stage, ids in enumerate(folds):
        path = out / f"validation_fold_{stage}_indices.npy"
        np.save(path, np.array(ids, dtype=np.int64), allow_pickle=False)
        paths[path.name] = _sha(path)
    np.save(out / "prior_excluded_indices.npy", np.array(sorted(blocked), dtype=np.int64), allow_pickle=False)
    paths["prior_excluded_indices.npy"] = _sha(out / "prior_excluded_indices.npy")
    _write(out / "global_frozen_plan.json", dict(schema="aiflow-balanced-temperature-global-plan/v15",
        script_sha256=_sha(Path(__file__)), canonical_sha256=_sha(CHECKPOINT), provenance=old["provenance"],
        class_labels=old["class_labels"], seed=SEED, stages=STAGES, batch_size=BATCH, validation_rows_per_class=VAL_PER_CLASS,
        artifact_sha256=paths, user_authorization="학습 승인; 개선 시 전체 기호 생성량 또는 다양성 온도 점진 확대 허용",
        training="real source IDs 0/1 CE only; unreviewed deformation uses original-teacher soft KL only; full encoder autograd",
        fixed_logit_temperature=2., shape_temperature="fraction of unchanged geometry-valid empirical q68 endpoint, NOT softmax temperature",
        next_stage_gate="augmented Top-1 > control and canonical; every family Top-1/5 >= both; no same-fold reuse; research scaling only",
        checkpoint_initialization="both arms canonical; next arms initialize their own preceding checkpoint, teacher always canonical",
        evaluation_limit="new student-excluded TRAIN internal folds, not fresh writer/device independence or product acceptance; teacher may have seen TRAIN",
        official_test_rows_read=0, consumed_pendigits_held_rows_read=0, crohme_rows=0, product_adopted=False,
        collection_or_deployment_resumed=False, cloud_upload=False))


def global_load(out: Path) -> tuple[dict, dict, list[np.ndarray], np.ndarray]:
    """현재 소스와 코드 및 이전 검사 입력 제외 목록의 봉인을 재확인한다."""
    plan = json.loads((out / "global_frozen_plan.json").read_text(encoding="utf-8"))
    if plan["script_sha256"] != _sha(Path(__file__)) or plan["canonical_sha256"] != _sha(CHECKPOINT):
        raise ValueError("frozen code/canonical changed")
    for name, digest in plan["artifact_sha256"].items():
        if _sha(out / name) != digest:
            raise ValueError("frozen fold changed")
    _, arrays = sources()
    folds = [np.load(out / f"validation_fold_{i}_indices.npy", allow_pickle=False) for i in range(3)]
    excluded = np.load(out / "prior_excluded_indices.npy", allow_pickle=False)
    return plan, arrays, folds, excluded


def prepare(out: Path, stage: int) -> int:
    """검증 fold 전체를 원본·공분산·donor에서 제외하고 클래스 균형 입력과 증강을 만든다."""
    if aug.base._guard_commit("before_balanced_temperature_prepare") is None:
        return 78
    global_prepare(out)
    plan, arrays, folds, excluded = global_load(out)
    folder = out / f"stage_{stage}"
    if folder.exists():
        raise FileExistsError("refusing prepared stage overwrite")
    if stage:
        previous = json.loads((out / f"stage_{stage-1}/stage_result.json").read_text(encoding="utf-8"))
        if not previous["next_stage_allowed"]:
            raise ValueError("previous improvement/nonregression gate failed: do not expand")
    temperature, quota, steps = STAGES[stage]
    x, y, s = (arrays[name] for name in ("features", "labels", "sources"))
    forbidden = np.concatenate((*folds, excluded))
    forbidden_hashes = {ink_hash(x[int(i)]) for i in forbidden}
    eligible = np.isin(s, (0, 1)); eligible[forbidden] = False
    rng = np.random.default_rng(SEED + 100 + stage)
    indices, seen, per_class = [], set(forbidden_hashes), {}
    for c in np.unique(y[eligible]):
        chosen = []
        for i in rng.permutation(np.flatnonzero(eligible & (y == c))):
            h = ink_hash(x[int(i)])
            if h in seen:
                continue
            chosen.append(int(i)); seen.add(h)
            if len(chosen) == quota:
                break
        if not chosen:
            raise ValueError("real class has no nonoverlapping optimization support")
        indices.extend(chosen); per_class[str(int(c))] = len(chosen)
    ids = np.array(indices, dtype=np.int64)
    tx, ty = np.array(x[ids], copy=True), np.array(y[ids], copy=True)
    query_ids = set(ids.tolist())
    views = np.repeat(tx[:, None], 4, axis=1)
    counts = np.ones(len(tx), dtype=np.int64)
    groups, actual_changed = [], np.zeros(len(tx), dtype=np.int64)
    for c in np.unique(ty):
        queries = np.flatnonzero(ty == c)
        signatures = {aug.base._signature(tx[i]) for i in queries}
        for signature in sorted(signatures):
            local = [int(i) for i in queries if aug.base._signature(tx[i]) == signature]
            fit_ids = []
            for i in rng.permutation(np.flatnonzero(eligible & (y == c))):
                i = int(i)
                if i in query_ids or ink_hash(x[i]) in forbidden_hashes or aug.base._signature(x[i]) != signature:
                    continue
                fit_ids.append(i)
                if len(fit_ids) == aug.base.MAX_FIT:
                    break
            group = dict(source_class_id=int(c), query_local_indices=local, fit_training_indices=fit_ids)
            groups.append(group)
            if len(fit_ids) < aug.base.MIN_FIT:
                group["status"] = "insufficient_fit_original_consistency_fallback"
                continue
            fit = np.array(x[fit_ids], copy=True)
            try:
                metric = aug.base._fit(fit)
            except ValueError:
                group["status"] = "degenerate_metric_original_consistency_fallback"
                continue
            group["status"] = "fitted_training_only"
            for i in local:
                _, initial = aug.base._deform(tx[i], fit, metric)
                if initial["status"] != "geometry_only_not_human_approved":
                    continue
                endpoints = []
                for donor in aug.mode_indices(tx[i], fit, metric, initial["donor_fit_index"]):
                    endpoint, _ = aug.deform_donor(tx[i], fit, metric, donor)
                    if endpoint is None:
                        continue
                    changed = tx[i].copy(); changed[:, :2] += temperature * (endpoint[:, :2] - tx[i, :, :2])
                    if aug.base._geometry(tx[i], changed)["valid"] and not any(np.array_equal(changed, other) for other in endpoints):
                        endpoints.append(changed)
                if endpoints:
                    views[i, :len(endpoints)] = endpoints
                    counts[i] = len(endpoints); actual_changed[i] = len(endpoints)
    folder.mkdir()
    artifacts = {}
    for name, value in dict(train_indices=ids, train_features=tx, train_labels=ty, augmented_views=views, view_counts=counts).items():
        path = folder / f"{name}.npy"; np.save(path, value, allow_pickle=False); artifacts[name] = _sha(path)
    _write(folder / "fit_query_provenance.json", groups)
    artifacts["fit_query_provenance"] = _sha(folder / "fit_query_provenance.json")
    _write(folder / "frozen_plan.json", dict(schema="aiflow-balanced-temperature-stage-plan/v15", global_plan_sha256=_sha(out / "global_frozen_plan.json"),
        stage=stage, shape_temperature=temperature, real_quota_per_class=quota, steps=steps, learning_rate=1e-5,
        class_real_rows=per_class, train_rows=len(tx), augmented_classes=len(np.unique(ty[actual_changed > 0])),
        nonzero_augmented_views=int(actual_changed.sum()), original_consistency_fallback_rows=int((actual_changed == 0).sum()),
        artifact_sha256=artifacts, synthetic_hard_labels=0, human_boundary_labels=0,
        validation_fit_query_index_overlap=0, validation_rounded_ink_overlap=0, product_adopted=False))
    qa = np.concatenate([np.flatnonzero((ty == c) & (actual_changed > 1))[:1] for c in np.unique(ty)])
    qa = qa[np.linspace(0, len(qa)-1, min(16, len(qa)), dtype=int)] if len(qa) else np.array([], dtype=int)
    if len(qa):
        aug.base._render(np.concatenate([views[i, :counts[i]] for i in qa]),
            [f"S{int(i):04d}V{j}" for i in qa for j in range(int(counts[i]))], folder / "training_geometry_qa.png")
    print(json.dumps(dict(event="balanced_stage_prepared", stage=stage, train_rows=len(tx), real_classes=len(per_class),
        augmented_classes=len(np.unique(ty[actual_changed > 0])), nonzero_augmented_views=int(actual_changed.sum()), shape_temperature=temperature)), flush=True)
    return 0


def hit_metrics(z: np.ndarray, y: np.ndarray, labels: list[str]) -> dict:
    """372-way Top-1/5를 숫자·문자·수학기호별로 집계한다."""
    rank = np.argsort(-z, axis=1, kind="stable")
    h1, h5 = rank[:, 0] == y, (rank[:, :5] == y[:, None]).any(1)
    result = {}
    for name in ("all", "digits", "latin_letters", "math_symbols"):
        mask = np.ones(len(y), dtype=bool) if name == "all" else np.array([aug.base._family(labels[int(c)]) == name for c in y])
        result[name] = dict(rows=int(mask.sum()), top1_hits=int(h1[mask].sum()), top5_hits=int(h5[mask].sum()))
    return result


def growth_gate(metrics: dict) -> bool:
    """전체 개선과 모든 family의 Top-1/5 비회귀가 확인된 연구 후보만 다음 규모로 간다."""
    b = metrics["soft_augmented"]
    return all(b["all"]["top1_hits"] > metrics[a]["all"]["top1_hits"] for a in ("canonical", "real_control")) and all(
        b[g][f"top{k}_hits"] >= metrics[a][g][f"top{k}_hits"]
        for a in ("canonical", "real_control") for g in b for k in (1, 5))


def run(out: Path, stage: int) -> int:
    """동일 초기 모델·클래스 배치에서 encoder까지 학습하고 종료 후 현재 fold만 단회 평가한다."""
    if aug.base._guard_commit("before_balanced_full_model_train") is None:
        return 78
    global_plan, arrays, folds, _ = global_load(out)
    folder = out / f"stage_{stage}"
    if (folder / "run_started.json").exists():
        raise FileExistsError("stage already started; inspect live process, never restart blindly")
    plan = json.loads((folder / "frozen_plan.json").read_text(encoding="utf-8"))
    if plan["global_plan_sha256"] != _sha(out / "global_frozen_plan.json"):
        raise ValueError("global plan changed")
    for name, digest in plan["artifact_sha256"].items():
        suffix = ".json" if name == "fit_query_provenance" else ".npy"
        if _sha(folder / f"{name}{suffix}") != digest:
            raise ValueError("prepared data changed")
    os.environ["TRACKIO_DIR"] = str(folder / "trackio")
    for key in ("TRACKIO_WEBHOOK_URL", "TRACKIO_SPACE_ID", "TRACKIO_SERVER_URL"):
        os.environ.pop(key, None)
    import torch
    import trackio
    from torch.nn import functional as F
    from hwr_boundary_distillation_v1 import boundary_kl_loss
    from run_hwr_affine_distillation_experiment_v1 import _load_teacher, _predict_logits
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    device = torch.device("cpu")
    teacher, labels, _ = _load_teacher(CHECKPOINT, device)
    if labels != global_plan["class_labels"]:
        raise ValueError("vocabulary differs")
    tx = np.load(folder / "train_features.npy", allow_pickle=False)
    y = np.load(folder / "train_labels.npy", allow_pickle=False)
    views = np.load(folder / "augmented_views.npy", allow_pickle=False)
    counts = np.load(folder / "view_counts.npy", allow_pickle=False)
    _write(folder / "run_started.json", dict(status="started", plan_sha256=_sha(folder / "frozen_plan.json"), process_id=os.getpid(), stage=stage))
    soft = torch.from_numpy(_predict_logits(teacher, tx, device, BATCH))
    classes = np.unique(y); rng = np.random.default_rng(SEED + stage + 991)
    order = np.resize(rng.permutation(classes), plan["steps"] * BATCH)
    pool = {int(c): rng.permutation(np.flatnonzero(y == c)).tolist() for c in classes}
    cursors = {int(c): 0 for c in classes}; batches = []
    for c in order:
        c = int(c); batches.append(pool[c][cursors[c] % len(pool[c])]); cursors[c] += 1
    batches = np.array(batches).reshape(plan["steps"], BATCH)
    np.save(folder / "shared_class_balanced_batches.npy", batches, allow_pickle=False)
    trained, checkpoints, histories = {}, {}, {}
    for arm in ARMS:
        path = CHECKPOINT if stage == 0 else out / f"stage_{stage-1}/{arm}.pt"
        model, arm_labels, _ = _load_teacher(path, device)
        if arm_labels != labels:
            raise ValueError("arm vocabulary differs")
        model.eval()  # dropout off; autograd remains on for the entire encoder.
        optimizer = torch.optim.AdamW(model.parameters(), lr=plan["learning_rate"], weight_decay=1e-4)
        activation = {}
        handles = []
        def capture(name):
            """각 encoder layer 출력의 실제 유한성·평균·표준편차를 수집한다."""
            def hook(module, inputs, output):
                value = output.detach()
                if not torch.isfinite(value).all():
                    raise FloatingPointError("nonfinite encoder activation")
                activation[name] = dict(mean=float(value.mean()), std=float(value.std()), min=float(value.min()), max=float(value.max()))
            return hook
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.TransformerEncoderLayer):
                handles.append(module.register_forward_hook(capture(name)))
        trackio.init(project="aiflow-balanced-temperature-v15", name=arm, space_id=None, embed=False, auto_log_cpu=False, auto_log_gpu=False,
            config=dict(stage=stage, shape_temperature=plan["shape_temperature"], steps=plan["steps"], lr=plan["learning_rate"]))
        history = []
        try:
            for step, bi in enumerate(batches):
                optimizer.zero_grad(set_to_none=True)
                index = torch.from_numpy(bi)
                real = torch.from_numpy(tx[bi].copy())
                embeddings = model.encode(real); logits = model.math_head(embeddings)
                original_layers = copy.deepcopy(activation)
                ce = F.cross_entropy(logits, torch.from_numpy(y[bi]).long())
                retention = boundary_kl_loss(logits, soft[index], 2., "full")
                if arm == "soft_augmented":
                    modes = (step + np.arange(BATCH)) % counts[bi]
                    changed = torch.from_numpy(views[bi, modes].copy())
                    view_logits = model.math_head(model.encode(changed))
                else:
                    view_logits = logits
                consistency = boundary_kl_loss(view_logits, soft[index], 2., "full")
                loss = .7 * ce + .2 * retention + .1 * consistency
                if not torch.isfinite(loss):
                    trackio.alert(title="nonfinite_loss", level=trackio.AlertLevel.ERROR)
                    raise FloatingPointError("nonfinite training loss")
                loss.backward()
                gradients = {name: None if p.grad is None else float(p.grad.detach().norm()) for name, p in model.named_parameters()}
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                if not torch.isfinite(norm):
                    raise FloatingPointError("nonfinite gradient")
                optimizer.step()
                numeric = dict(step=step + 1, loss=float(loss.detach()), ce_real=float(ce.detach()), kl_retention=float(retention.detach()),
                    kl_view=float(consistency.detach()), gradient_l2=float(norm), embedding_std=float(embeddings.detach().std()))
                history.append(dict(**numeric, parameter_gradient_l2=gradients, real_encoder_layers=original_layers, view_encoder_layers=copy.deepcopy(activation)))
                trackio.log(numeric, step=step + 1)
                if (step + 1) % 16 == 0:
                    print(json.dumps(dict(event="balanced_temperature_train", stage=stage, arm=arm, **numeric)), flush=True)
        finally:
            for handle in handles:
                handle.remove()
            trackio.finish()
        checkpoint = folder / f"{arm}.pt"
        torch.save(dict(schema="aiflow-balanced-temperature-research/v15", state_dict=model.state_dict(), math_labels=labels, auxiliary_labels=[],
            report=dict(input_contract=dict(observed_channel_mode="uniform-time", math_observed_transform="uniform-time"), product_adopted=False)), checkpoint)
        checkpoints[arm] = _sha(checkpoint); trained[arm] = model; histories[arm] = history
        _write(folder / f"{arm}_training_microscope.json", history)
    # 두 arm의 고정 예산이 모두 끝난 뒤, 현재 fold만 읽는다. 미래 fold는 평가하지 않는다.
    ids = folds[stage]; vx = np.array(arrays["features"][ids], copy=True); vy = np.array(arrays["labels"][ids], copy=True)
    scores = {"canonical": _predict_logits(teacher, vx, device, BATCH)}
    scores.update({name: _predict_logits(model, vx, device, BATCH) for name, model in trained.items()})
    metrics = {name: hit_metrics(z, vy, labels) for name, z in scores.items()}
    for name, z in scores.items():
        np.save(folder / f"{name}_validation_logits.npy", z, allow_pickle=False)
    if _sha(CHECKPOINT) != global_plan["canonical_sha256"]:
        raise ValueError("canonical changed")
    result = dict(schema="aiflow-balanced-temperature-stage-result/v15", status="completed_research_not_acceptance",
        plan_sha256=_sha(folder / "frozen_plan.json"), metrics=metrics, checkpoint_sha256=checkpoints,
        next_stage_allowed=growth_gate(metrics) and stage < 2, actual_optimizer_steps_per_arm=plan["steps"],
        evaluated_fold_now_consumed=stage, future_folds_forwarded=0, synthetic_hard_labels=0,
        canonical_unchanged=True, independent_accuracy_claim_allowed=False, product_adopted=False,
        official_test_rows_read=0, crohme_rows=0, collection_or_deployment_resumed=False)
    _write(folder / "stage_result.json", result)
    print(json.dumps(dict(event="balanced_temperature_result", stage=stage, metrics=metrics, next_stage_allowed=result["next_stage_allowed"])), flush=True)
    return 0


def selftest() -> int:
    """미미한 개선도 family 회귀가 있으면 확대하지 않는지 검사한다."""
    metrics = {a: {g: dict(rows=8, top1_hits=5, top5_hits=7) for g in ("all", "digits", "latin_letters", "math_symbols")}
               for a in ("canonical", *ARMS)}
    assert not growth_gate(metrics)
    metrics["soft_augmented"]["all"]["top1_hits"] = 6
    assert growth_gate(metrics)
    metrics["soft_augmented"]["latin_letters"]["top1_hits"] = 4
    assert not growth_gate(metrics)
    import torch
    from torch.nn import functional as F
    logits = torch.zeros((2, 372), requires_grad=True)
    targets = torch.from_numpy(np.array([0, 371], dtype=np.int16)).long()
    loss = F.cross_entropy(logits, targets); loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    print(json.dumps(dict(event="balanced_temperature_selftest", status="pass")), flush=True)
    return 0


def main() -> int:
    """봉인 준비와 단회 학습을 분리하며 실패 시 자동 확대·재시작하지 않는다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("selftest", "prepare", "run"))
    parser.add_argument("--stage", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    return selftest() if args.mode == "selftest" else {"prepare": prepare, "run": run}[args.mode](args.output_dir, args.stage)


if __name__ == "__main__":
    raise SystemExit(main())
