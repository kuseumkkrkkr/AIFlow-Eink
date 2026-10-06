"""봉인 head 교정 후보를 보정 행 밖의 실제 TRAIN에서 읽기 전용 비교한다.

writer/device 독립 acceptance가 아니다. 공식 test, 소비된 보류 그룹, CROHME는
추론하지 않는다. prepare로 클래스 균등 표본을 먼저 봉인하며 재학습하지 않는다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from audit_hwr_probability_boundary_tube_v1 import ROOT, DEFAULT_CHECKPOINT, DEFAULT_DATA_DIR, _guard_commit
from run_hwr_pendigits_tube_probe_v1 import OUTPUT as PARENT, SOURCE, _sha, _write
from run_hwr_pendigits_retention_probe_v1 import OUTPUT as RETENTION
from run_hwr_pendigits_directional_guard_v1 import OUTPUT as CANDIDATE

OUTPUT = ROOT / "artifacts/hwr_repair_complement_audit_20261005_v13"
REFERENCE = ROOT / "artifacts/hwr_joint_retention_repair_20261005"
FULL = ROOT / "artifacts/hwr_full_domain_tube_20261005"
V7 = ROOT / "artifacts/hwr_active_trust_retention_20261005_rival_guard_v7"
V12 = ROOT / "artifacts/hwr_active_trust_retention_20261005_guard_replanning_v12"
SEED, OLD_PER_CLASS, DIGIT_PER_CLASS = 2026100513, 8, 100
MODEL_PATHS = {
    "canonical": DEFAULT_CHECKPOINT,
    "original_parent": CANDIDATE / "directional_guard.pt",
    "v7": V7 / "failed_active_trust_research.pt",
    "v12": V12 / "active_trust_repaired_research.pt",
}
MODEL_SHA = {
    "canonical": "04f8608aebcf6c02d45ad6f5735229b9eaa2c4b4e1be0db4793d02273ef2d00e",
    "original_parent": "b0aa6411b1e299ec14773cd44eadf54341d55582af68b0dc973c846c6e1d8963",
    "v7": "4d790c635af49ded6f59841684ac0fbaf9f0f8a861ba7caf319576f44bf6349a",
    "v12": "66208dbd835b8b19dfd50a5c4dfbd6666256d780357e9569cc02f8d652ffd45c",
}
PAIR_NAMES = (("canonical", "original_parent"), ("canonical", "v7"),
              ("canonical", "v12"), ("original_parent", "v12"), ("v7", "v12"))


def family(label: str) -> str:
    """숫자/ASCII 문자/그 밖의 수학기호를 기존 전 범위 정의대로 분리한다."""
    return ("digits" if len(label) == 1 and label in "0123456789" else
            "latin_letters" if len(label) == 1 and label.isascii() and label.isalpha() else "math_symbols")


def fingerprints(row: np.ndarray) -> tuple[str, str]:
    """전체 float32 바이트와 6자리 XY/획 시작을 각각 비교해 중복을 배제한다."""
    exact = np.ascontiguousarray(row, dtype="<f4")
    ink = np.ascontiguousarray(np.round(row[:, (0, 1, 3)], 6), dtype="<f4")
    return hashlib.sha256(exact.tobytes()).hexdigest(), hashlib.sha256(ink.tobytes()).hexdigest()


def checked_array(path: Path, digest: str, mmap: bool = True) -> np.ndarray:
    """봉인 SHA가 일치하는 명시적 TRAIN 배열만 읽는다."""
    if _sha(path) != digest:
        raise ValueError(f"sealed input changed: {path}")
    return np.load(path, mmap_mode="r" if mmap else None, allow_pickle=False)


def source_inputs() -> dict:
    """원래 2048개 교정 행을 재구성하고 실제 TRAIN 출처를 확인한다."""
    parent = json.loads((PARENT / "frozen_plan.json").read_text(encoding="utf-8"))
    full = json.loads((FULL / "frozen_plan.json").read_text(encoding="utf-8"))
    manifest = json.loads((DEFAULT_DATA_DIR / "prepared_manifest.json").read_text(encoding="utf-8"))
    candidate = json.loads((V12 / "research_candidate_manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "pass" or manifest["input_policy"]["admitted_train"] != "HWRT curated official train split + UJI Pen v2 official writer-disjoint train split":
        raise ValueError("TRAIN admission policy differs")
    if _sha(DEFAULT_DATA_DIR / "prepared_manifest.json") != parent["replay_cache_manifest_sha256"]:
        raise ValueError("TRAIN manifest changed")
    if candidate["checkpoint_sha256"] != MODEL_SHA["v12"] or not candidate["strict_train_retention_gate_pass"]:
        raise ValueError("v12 has no verified frozen TRAIN gate")
    for name, path in MODEL_PATHS.items():
        if _sha(path) != MODEL_SHA[name]:
            raise ValueError(f"checkpoint changed: {name}")
    old = {name: checked_array(DEFAULT_DATA_DIR / f"train_{name}.npy", digest)
           for name, digest in parent["replay_train_array_hashes"].items()}
    ids = {name: checked_array(PARENT / parent["artifacts"][name]["file"], parent["artifacts"][name]["sha256"], False)
           for name in ("replay_indices", "train_indices")}
    digit_report = json.loads((SOURCE / "pendigits_source_audit.json").read_text(encoding="utf-8"))
    if _sha(SOURCE / "pendigits_source_audit.json") != parent["source_report_sha256"]:
        raise ValueError("source TRAIN provenance changed")
    digits = {name: checked_array(SOURCE / f"{name}.npy", digit_report["artifacts"][name]["sha256"])
              for name in ("train_features_y_up", "train_digit_labels", "train_writer_group_hashes")}
    probe = checked_array(RETENTION / "source_train_probe_indices.npy",
                          "94aa0be716f79aa75ab982ffdba6d6309e27c7800b39e6fc152605b8368b524c", False)
    joint_x = checked_array(REFERENCE / "joint_features.npy", candidate["optimization_features_sha256"])
    joint_y = checked_array(REFERENCE / "joint_labels.npy", candidate["optimization_labels_sha256"])
    labels = full["class_labels"]
    digit_ids = np.array([labels.index(str(d)) for d in range(10)], dtype=np.int64)
    if len(labels) != 372 or joint_x.shape != (2048, 128, 5) or joint_y.shape != (2048,):
        raise ValueError("frozen vocabulary/repair contract differs")
    train_ids = ids["train_indices"]
    if len(train_ids) != 5995 or len(np.unique(train_ids)) != len(train_ids) or not np.isin(probe, train_ids).all():
        raise ValueError("digit TRAIN index scope differs")
    if not np.isin(digits["train_writer_group_hashes"][train_ids], parent["train_groups"]).all():
        raise ValueError("source example outside admitted writer groups")
    if not np.array_equal(joint_x[:1024], old["features"][ids["replay_indices"]]) or not np.array_equal(joint_y[:1024], old["labels"][ids["replay_indices"]]):
        raise ValueError("old optimization anchors not reproducible")
    if not np.array_equal(joint_x[1024:], digits["train_features_y_up"][probe]) or not np.array_equal(joint_y[1024:], digit_ids[digits["train_digit_labels"][probe]]):
        raise ValueError("digit optimization anchors not reproducible")
    if old["features"].shape != (len(old["labels"]), 128, 5) or old["sources"].shape != old["labels"].shape:
        raise ValueError("old TRAIN arrays are not aligned")
    return dict(parent=parent, labels=labels, old=old, ids=ids, digits=digits, probe=probe,
                joint_x=joint_x, joint_y=joint_y, digit_ids=digit_ids)


def select_inputs(inputs: dict) -> tuple[dict[str, np.ndarray], dict]:
    """예측을 보지 않고 모든 실제 지원 클래스에서 고정 개수의 중복 없는 행을 고른다."""
    old, digits = inputs["old"], inputs["digits"]
    rng = np.random.default_rng(SEED)
    blocked = [set(), set()]
    for row in inputs["joint_x"]:
        for k, digest in enumerate(fingerprints(row)):
            blocked[k].add(digest)
    seen = [set(), set()]
    eligible = np.isin(old["sources"], (0, 1))
    eligible[inputs["ids"]["replay_indices"]] = False
    classes = np.unique(old["labels"][np.isin(old["sources"], (0, 1))])
    if len(classes) != 371:
        raise ValueError("real-supported class coverage changed")
    source_ids = inputs["ids"]["train_indices"]
    source_ids = source_ids[~np.isin(source_ids, inputs["probe"])]
    rejected = dict(anchor_exact=0, anchor_rounded_ink=0, selected_exact=0, selected_rounded_ink=0)
    selected: dict[str, list[int]] = {"old_indices": [], "digit_indices": []}
    support = {}

    def take(name: str, pool: np.ndarray, rows: np.ndarray, count: int) -> None:
        """중복이나 보류 행으로 대체하지 않고 사전 고정 클래스별 순열을 순회한다."""
        accepted = 0
        for index in rng.permutation(pool):
            a, b = fingerprints(rows[int(index)])
            reason = ("anchor_exact" if a in blocked[0] else "anchor_rounded_ink" if b in blocked[1] else
                      "selected_exact" if a in seen[0] else "selected_rounded_ink" if b in seen[1] else None)
            if reason is not None:
                rejected[reason] += 1
                continue
            selected[name].append(int(index))
            seen[0].add(a); seen[1].add(b)
            accepted += 1
            if accepted == count:
                return
        raise ValueError(f"insufficient nonoverlapping TRAIN support: {name}, {len(pool)}, {accepted}/{count}")

    for class_id in classes:
        pool = np.flatnonzero(eligible & (old["labels"] == class_id))
        support[inputs["labels"][int(class_id)]] = int(len(pool))
        take("old_indices", pool, old["features"], OLD_PER_CLASS)
    train_digits = digits["train_digit_labels"][source_ids]
    for digit in range(10):
        take("digit_indices", source_ids[train_digits == digit], digits["train_features_y_up"], DIGIT_PER_CLASS)
    a, b = (np.array(selected[name], dtype=np.int64) for name in ("old_indices", "digit_indices"))
    x = np.concatenate((old["features"][a], digits["train_features_y_up"][b]))
    y = np.concatenate((old["labels"][a], inputs["digit_ids"][digits["train_digit_labels"][b]]))
    if x.dtype != np.float32 or y.shape != (len(x),) or not np.isfinite(x).all():
        raise ValueError("selected input tensor contract differs")
    return dict(old_indices=a, digit_indices=b, features=x, targets=y), dict(
        old_rows=len(a), digit_rows=len(b), real_supported_classes=len(classes), excluded_optimization_rows=2048,
        rejected_duplicates=rejected, available_old_nonanchor_index_rows_by_class=support,
        selected_vs_optimization_index_overlap=0, selected_vs_optimization_exact_or_rounded_ink_overlap=0,
        cross_cohort_exact_or_rounded_ink_overlap=0)


def prepare(out: Path) -> int:
    """출처·표본·모델·지표를 실제 모델 실행 전에 고정한다."""
    if out.exists():
        raise FileExistsError("refusing to overwrite complementary TRAIN audit")
    if _guard_commit("before_repair_complement_prepare") is None:
        return 78
    inputs = source_inputs()
    arrays, checks = select_inputs(inputs)
    out.mkdir(parents=True)
    artifacts = {}
    for name, array in arrays.items():
        path = out / f"{name}.npy"
        np.save(path, array, allow_pickle=False)
        artifacts[name] = dict(file=path.name, sha256=_sha(path), shape=list(array.shape))
    _write(out / "frozen_plan.json", dict(
        schema="aiflow-repair-complement-train-plan/v13", script_sha256=_sha(Path(__file__)),
        seed=SEED, old_per_class=OLD_PER_CLASS, digit_per_class=DIGIT_PER_CLASS, class_labels=inputs["labels"],
        artifacts=artifacts, preparation_checks=checks, checkpoint_sha256=MODEL_SHA,
        source_plan_sha256=_sha(PARENT / "frozen_plan.json"),
        candidate_manifest_sha256=_sha(V12 / "research_candidate_manifest.json"),
        selection="label-stratified seeded permutation; exclude all 2048 repair rows and exact/rounded-ink duplicates; predictions never used",
        metrics="372-way stable Top-1/Top-5, macro class accuracy, per-family counts, paired wins/losses, immutable half-reference-margin diagnostics",
        model_pairs=PAIR_NAMES, batch_size=32, optimizer_created=False, parameter_updates=0,
        official_test_rows_read=0, held_examples_indexed_or_forwarded=0, crohme_rows=0,
        input_hash_policy="full TRAIN files fingerprinted for provenance; PenDigits example indexing only within admitted 5995 TRAIN indices",
        evaluation_role="repair-complement TRAIN diagnostic, NOT unseen-to-parent training or independent writer/device acceptance",
        promotion_policy="no automatic candidate selection or promotion; report all predeclared comparisons even on regressions",
        product_adoption_allowed=False, independent_accuracy_claim_allowed=False, human_boundary_labels=0))
    print(json.dumps(dict(event="complement_cohort_frozen", old_rows=checks["old_rows"], digit_rows=checks["digit_rows"],
                         classes=371, rejected_duplicates=checks["rejected_duplicates"])), flush=True)
    return 0


def hits_and_margins(logits: np.ndarray, y: np.ndarray, independent: bool = False) -> dict:
    """372-way 순위와 truth를 제외한 첫째/다섯째 rival margin을 반환한다."""
    if logits.shape != (len(y), 372) or not np.isfinite(logits).all():
        raise ValueError("invalid score contract")
    rows = np.arange(len(y))
    if independent:
        target = logits[rows, y, None]
        rank = 1 + (logits > target).sum(1) + ((logits == target) & (np.arange(372)[None, :] < y[:, None])).sum(1)
        top1, top5 = rank == 1, rank <= 5
    else:
        order = np.argsort(-logits, axis=1, kind="stable")
        top1, top5 = order[:, 0] == y, (order[:, :5] == y[:, None]).any(1)
    other = logits.copy(); other[rows, y] = -np.inf
    rivals = np.sort(other, axis=1)[:, (-1, -5)]
    return dict(top1=top1, top5=top5, margin1=logits[rows, y] - rivals[:, 0], margin5=logits[rows, y] - rivals[:, 1])


def summarize(scores: dict[str, np.ndarray], targets: np.ndarray, labels: list[str], old_n: int,
              independent: bool = False) -> dict:
    """모든 사전 지정 모델/도메인의 지표와 회복·손실 행을 함께 반환한다."""
    indices = np.arange(len(targets))
    groups = {"old_all": indices[:old_n], "source_digits": indices[old_n:]}
    for name in ("digits", "latin_letters", "math_symbols"):
        groups[f"old_{name}"] = indices[:old_n][np.array([family(labels[int(c)]) == name for c in targets[:old_n]])]
    ranks = {name: hits_and_margins(value, targets, independent) for name, value in scores.items()}
    models, paired = {}, {}
    for name, rank in ranks.items():
        models[name] = {}
        for group, ids in groups.items():
            classes = np.unique(targets[ids])
            models[name][group] = dict(rows=len(ids), classes=len(classes), **{
                f"top{k}_hits": int(rank[f"top{k}"][ids].sum()) for k in (1, 5)}, **{
                f"top{k}_macro": float(np.mean([rank[f"top{k}"][ids[targets[ids] == c]].mean() for c in classes])) for k in (1, 5)})
    for a, b in PAIR_NAMES:
        paired[f"{a}_to_{b}"] = {}
        for group, ids in groups.items():
            values = dict(rows=len(ids))
            for k in (1, 5):
                before, after = ranks[a][f"top{k}"][ids], ranks[b][f"top{k}"][ids]
                wins, losses = ~before & after, before & ~after
                values[f"top{k}"] = dict(wins=int(wins.sum()), losses=int(losses.sum()),
                    net=int(wins.sum() - losses.sum()), win_rows=ids[wins].tolist(), loss_rows=ids[losses].tolist(),
                    protected_half_margin_failures=int((before & (ranks[b][f"margin{k}"][ids] < .5 * ranks[a][f"margin{k}"][ids])).sum()))
            paired[f"{a}_to_{b}"][group] = values
    return dict(models=models, paired=paired)


def load_prepared(out: Path) -> tuple[dict, dict[str, np.ndarray]]:
    """예측 전 봉인한 지표·모델·표본만 허용하며 재선택을 금지한다."""
    plan = json.loads((out / "frozen_plan.json").read_text(encoding="utf-8"))
    if plan["script_sha256"] != _sha(Path(__file__)) or plan["checkpoint_sha256"] != MODEL_SHA:
        raise ValueError("entrypoint or model plan changed")
    if _sha(PARENT / "frozen_plan.json") != plan["source_plan_sha256"] or _sha(V12 / "research_candidate_manifest.json") != plan["candidate_manifest_sha256"]:
        raise ValueError("frozen source/candidate changed")
    arrays = {name: checked_array(out / info["file"], info["sha256"], False) for name, info in plan["artifacts"].items()}
    if plan["seed"] != SEED or plan["old_per_class"] != OLD_PER_CLASS or plan["digit_per_class"] != DIGIT_PER_CLASS:
        raise ValueError("selection settings changed")
    return plan, arrays


def run(out: Path) -> int:
    """네 개 봉인 모델을 동일 표본에서 한 번씩 평가하고 실패도 그대로 남긴다."""
    if (out / "complement_result.json").exists() or any(out.glob("*_logits.npy")):
        raise FileExistsError("refusing repeated scoring or incomplete-run overwrite")
    if _guard_commit("before_repair_complement_scoring") is None:
        return 78
    plan, arrays = load_prepared(out)
    import torch
    from run_hwr_affine_distillation_experiment_v1 import _load_teacher, _predict_logits
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    device = torch.device("cpu")
    scores, timings, score_files = {}, {}, {}
    for name, path in MODEL_PATHS.items():
        if _sha(path) != MODEL_SHA[name]:
            raise ValueError("checkpoint changed")
        model, labels, _ = _load_teacher(path, device)
        if labels != plan["class_labels"]:
            raise ValueError("candidate vocabulary differs")
        start = time.perf_counter()
        scores[name] = _predict_logits(model, arrays["features"], device, plan["batch_size"])
        timings[name] = time.perf_counter() - start
        score_path = out / f"{name}_logits.npy"
        np.save(score_path, scores[name], allow_pickle=False)
        score_files[name] = dict(file=score_path.name, sha256=_sha(score_path))
        print(json.dumps(dict(event="complement_model_scored", model=name, rows=len(arrays["targets"]), seconds=timings[name])), flush=True)
        del model
    if any(_sha(path) != MODEL_SHA[name] for name, path in MODEL_PATHS.items()):
        raise ValueError("read-only checkpoint guard failed")
    result = dict(schema="aiflow-repair-complement-train-result/v13", status="completed_diagnostic_not_acceptance",
        frozen_plan_sha256=_sha(out / "frozen_plan.json"), score_files=score_files, timing_seconds=timings,
        metrics=summarize(scores, arrays["targets"], plan["class_labels"], len(arrays["old_indices"])),
        parameter_updates=0, checkpoints_unchanged=True, official_test_rows_read=0, held_examples_forwarded=0,
        crohme_rows=0, independent_accuracy_claim_allowed=False, product_adoption_allowed=False)
    _write(out / "complement_result.json", result)
    print(json.dumps(dict(event="complement_metrics", models=result["metrics"]["models"])), flush=True)
    return 0


def verify(out: Path) -> int:
    """교정 입력 재구성·중복·클래스 균형 및 별도 순위 계산으로 보고 수치를 재검증한다."""
    if (out / "independent_verification.json").exists():
        raise FileExistsError("refusing verification overwrite")
    inputs = source_inputs()
    plan, arrays = load_prepared(out)
    rebuilt, _ = select_inputs(inputs)
    if any(not np.array_equal(arrays[name], rebuilt[name]) for name in arrays):
        raise ValueError("model-blind cohort is not reproducible")
    a, b = arrays["old_indices"], arrays["digit_indices"]
    if np.isin(a, inputs["ids"]["replay_indices"]).any() or np.isin(b, inputs["probe"]).any() or not np.isin(b, inputs["ids"]["train_indices"]).all():
        raise ValueError("optimization/held input indexed")
    anchors = [set(), set()]
    seen = [set(), set()]
    for row in inputs["joint_x"]:
        for k, digest in enumerate(fingerprints(row)):
            anchors[k].add(digest)
    for row in arrays["features"]:
        for k, digest in enumerate(fingerprints(row)):
            if digest in anchors[k] or digest in seen[k]:
                raise ValueError("exact or rounded-ink duplicate passed")
            seen[k].add(digest)
    counts = np.unique(arrays["targets"][:len(a)], return_counts=True)[1]
    digit_counts = np.unique(arrays["targets"][len(a):], return_counts=True)[1]
    if len(counts) != 371 or not (counts == 8).all() or len(digit_counts) != 10 or not (digit_counts == 100).all():
        raise ValueError("class-stratified quotas differ")
    result = json.loads((out / "complement_result.json").read_text(encoding="utf-8"))
    if result["frozen_plan_sha256"] != _sha(out / "frozen_plan.json"):
        raise ValueError("result plan changed")
    scores = {name: checked_array(out / info["file"], info["sha256"], False) for name, info in result["score_files"].items()}
    if set(scores) != set(MODEL_PATHS):
        raise ValueError("missing declared comparison")
    actual = summarize(scores, arrays["targets"], plan["class_labels"], len(a), independent=True)
    if actual != result["metrics"]:
        raise ValueError("independent target-rank counts differ")
    _write(out / "independent_verification.json", dict(schema="aiflow-repair-complement-verification/v13", status="reproduced",
        result_sha256=_sha(out / "complement_result.json"), script_sha256=_sha(Path(__file__)),
        input_rows=len(arrays["targets"]), real_supported_classes=371, model_blind_cohort_rebuilt=True,
        optimization_index_or_exact_or_rounded_ink_overlap=0, held_examples_indexed_or_forwarded=0,
        cross_cohort_exact_or_rounded_ink_duplicates=0, all_stable_ranks_recounted=True,
        all_predeclared_metrics_reproduced=True, parameter_updates=0, checkpoints_unchanged=True,
        independent_writer_device_acceptance=False, product_adopted=False))
    print(json.dumps(dict(event="complement_verification", status="reproduced", rows=len(arrays["targets"]))), flush=True)
    return 0


def main() -> int:
    """준비/단회 추론/읽기 전용 수치 검증을 명시적으로 분리한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "run", "verify", "selftest"))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.mode == "selftest":
        return selftest()
    return {"prepare": prepare, "run": run, "verify": verify}[args.mode](args.output_dir)


def selftest() -> int:
    """동점 순위, 다섯째 OTHER 및 입력 fingerprint의 독립 계산을 작게 검증한다."""
    rng = np.random.default_rng(13)
    logits = rng.normal(size=(9, 372)).astype(np.float32)
    logits[:3] = 0
    targets = np.array([0, 4, 5, 17, 51, 219, 371, 13, 7], dtype=np.int64)
    direct = hits_and_margins(logits, targets)
    counted = hits_and_margins(logits, targets, independent=True)
    if any(not np.array_equal(direct[key], counted[key]) for key in direct):
        raise AssertionError("stable ties differ")
    if direct["top1"][:3].tolist() != [True, False, False] or direct["top5"][:3].tolist() != [True, True, False]:
        raise AssertionError("stable tie contract differs")
    row = np.zeros((128, 5), dtype=np.float32)
    row[0, 3] = 1
    changed = row.copy(); changed[64, 0] = .01
    nonxy = row.copy(); nonxy[64, 2] = 1
    if fingerprints(row) == fingerprints(changed) or fingerprints(row)[0] == fingerprints(nonxy)[0] or fingerprints(row)[1] != fingerprints(nonxy)[1]:
        raise AssertionError("duplicate contracts differ")
    print(json.dumps(dict(event="complement_selftest", status="pass")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
