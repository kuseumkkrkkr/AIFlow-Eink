"""전 범위 증강의 한 donor 병목을 최대 네 개의 경험적 방향으로 확장한다.

기존 TRAIN 원본·fit·기하 제한은 고정한다. teacher로 방향을 선택하거나 모델을
학습하지 않으며, 생성물은 사람 검수 전 정답 라벨이 없는 geometry 후보다.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import audit_hwr_full_domain_tube_v1 as base
from run_hwr_pendigits_tube_probe_v1 import _sha, _write

OUTPUT = base.ROOT / "artifacts/hwr_full_domain_direction_diversity_20261005_v14"
BASE_SHA = "0e02fa389fffe23c160cb4be62ded3dc0937f54debe9d9adaa45a87af34817d5"
MAX_DIRECTIONS, MAX_COSINE = 4, .98


def mode_indices(original: np.ndarray, fit: np.ndarray, metric: dict, primary: int) -> list[int]:
    """기존 donor를 첫째로 유지하고 whitened 방향의 최대 코사인 유사도를 줄인다."""
    q = base._aligned(original).reshape(-1, 16, 2)
    coeff = ((q.ravel() - metric["center"]) @ metric["basis"].T) / np.sqrt(metric["eigen"])
    vectors = metric["standardized"] - coeff
    norms = np.linalg.norm(vectors, axis=1)
    units = vectors / np.maximum(norms[:, None], 1e-12)
    pool = []
    for i, row in enumerate(fit):
        donor = base._aligned(row).reshape(-1, 16, 2)
        dot = ((q[:, -1] - q[:, 0]) * (donor[:, -1] - donor[:, 0])).sum(1)
        if (dot >= -1e-8).all() and norms[i] > 1e-8:
            pool.append(i)
    chosen = [primary]
    if norms[primary] <= 1e-8:
        return chosen
    while len(chosen) < MAX_DIRECTIONS:
        remaining = [i for i in pool if i not in chosen]
        if not remaining:
            break
        similarities = {i: float((units[chosen] @ units[i]).max()) for i in remaining}
        winner = min(remaining, key=lambda i: (similarities[i], i))
        if similarities[winner] > MAX_COSINE:
            break
        chosen.append(winner)
    return chosen


def deform_donor(original: np.ndarray, fit: np.ndarray, metric: dict, donor: int) -> tuple[np.ndarray | None, dict]:
    """기존 q68 거리 캡과 동일한 scale·기하 제한으로 선택 donor의 타점을 보간한다."""
    coeff = ((base._aligned(original).ravel() - metric["center"]) @ metric["basis"].T) / np.sqrt(metric["eigen"])
    distance = float(np.linalg.norm(metric["standardized"][donor] - coeff))
    fraction = min(1., metric["radius"] / max(distance, 1e-8))
    donor_xy = base._aligned(fit[donor], original)
    trials = []
    for scale in base.SCALES:
        changed = original.copy()
        changed[:, :2] = original[:, :2] + scale * fraction * (donor_xy - original[:, :2])
        geometry = base._geometry(original, changed)
        trials.append(dict(scale=scale, **geometry))
        if geometry["valid"] and geometry["rms"] > 1e-7:
            movement = ((base._aligned(changed).ravel() - base._aligned(original).ravel()) @ metric["basis"].T) / np.sqrt(metric["eigen"])
            return changed, dict(donor_fit_index=donor, scale=scale, fraction=scale * fraction,
                donor_distance=distance, q68_radius=metric["radius"], actual_projected_movement=float(np.linalg.norm(movement)), trials=trials)
    return None, dict(donor_fit_index=donor, reason="no_nonzero_geometry_valid_endpoint", trials=trials)


def checked_inputs() -> tuple[dict, dict, dict, dict]:
    """봉인된 기존 감사·원본 TRAIN 배열만 읽고 teacher logits는 읽지 않는다."""
    result_path = base.OUTPUT / "full_domain_tube_audit.json"
    if _sha(result_path) != BASE_SHA:
        raise ValueError("sealed baseline audit changed")
    report = json.loads(result_path.read_text(encoding="utf-8"))
    plan = json.loads((base.OUTPUT / "frozen_plan.json").read_text(encoding="utf-8"))
    if _sha(base.OUTPUT / "frozen_plan.json") != report["frozen_plan_sha256"] or _sha(Path(base.__file__)) != plan["script_sha256"]:
        raise ValueError("sealed baseline algorithm changed")
    if _sha(base.DEFAULT_CHECKPOINT) != plan["provenance"]["teacher_checkpoint_sha256"]:
        raise ValueError("canonical checkpoint changed")
    if _sha(base.DEFAULT_DATA_DIR / "prepared_manifest.json") != plan["provenance"]["prepared_manifest_sha256"]:
        raise ValueError("TRAIN provenance changed")
    arrays = {}
    for name in ("train_features", "train_labels", "train_sources"):
        path = base.DEFAULT_DATA_DIR / f"{name}.npy"
        if _sha(path) != plan["provenance"][f"{name}_sha256"]:
            raise ValueError("TRAIN array changed")
        arrays[name] = np.load(path, mmap_mode="r", allow_pickle=False)
    mapping_path = base.OUTPUT / "blind_review_mapping.json"
    if _sha(mapping_path) != report["review_packet_sha256"][mapping_path.name]:
        raise ValueError("baseline mapping changed")
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    cache = report["candidate_cache"]["artifacts"]["candidate_features"]
    cache_path = base.OUTPUT / "candidate_features.npy"
    if _sha(cache_path) != cache["sha256"]:
        raise ValueError("baseline geometry cache changed")
    old = np.load(cache_path, mmap_mode="r", allow_pickle=False)
    by_query = {item["query_training_row"]: old[item["candidate_row"] + 1] for item in mapping if item["endpoint"] == "original"}
    return report, plan, arrays, by_query


def generate(out: Path) -> int:
    """전 범위 원본별 실제 방향 수·생성물·동일 donor 재현·무결성을 기록한다."""
    if out.exists():
        raise FileExistsError("refusing direction-diversity audit overwrite")
    if base._guard_commit("before_full_domain_direction_diversity") is None:
        return 78
    report, parent, arrays, by_query = checked_inputs()
    out.mkdir(parents=True)
    plan = dict(schema="aiflow-full-domain-direction-diversity-plan/v14", script_sha256=_sha(Path(__file__)),
        baseline_audit_sha256=BASE_SHA, baseline_entrypoint_sha256=parent["script_sha256"],
        max_directions=MAX_DIRECTIONS, max_pairwise_positive_cosine=MAX_COSINE, scales=base.SCALES,
        direction_selection="unchanged first donor; greedy minimum maximum whitened-direction cosine; deterministic row-index tie break",
        source_seed_limit="originals and fits inherited from sealed teacher-margin-selected TRAIN seeds; not an independent or random acceptance sample",
        class_balanced_use="per-source-class sampling required if human-approved candidates are later used; generated counts are not training weights",
        metric="empirical class/topology q68, not Gaussian SD or human recognition probability",
        actual_projected_q68_limit="diagnostic only; do not assume capped donor distance proves re-aligned endpoint radius",
        source_class_metadata_is_semantic_ground_truth=False, hard_synthetic_labels_assigned=0,
        parameter_updates=0, model_forwards=0, teacher_logits_loaded=False,
        official_test_rows_read=0, held_examples_forwarded=0, crohme_rows=0, product_adopted=False)
    _write(out / "frozen_plan.json", plan)
    features, metadata, seed_reports = [], [], []
    x, y, sources = (arrays[name] for name in ("train_features", "train_labels", "train_sources"))
    baseline_rebuilt, new_direction_failures = 0, 0
    for class_report in report["classes"]:
        class_id = class_report["class_id"]
        for group in class_report["groups"]:
            if group["status"] != "metric_fitted":
                continue
            fit_ids = np.array(group["fit_row_ids"], dtype=np.int64)
            queries = group["query_row_ids"]
            if len(set(fit_ids.tolist()) & set(queries)) or not np.isin(sources[fit_ids], (0, 1)).all() or not (y[fit_ids] == class_id).all():
                raise ValueError("fit provenance/topology isolation failed")
            fit = np.array(x[fit_ids], copy=True)
            metric = base._fit(fit)
            for query in queries:
                if query not in by_query:
                    continue
                if sources[query] not in (0, 1) or y[query] != class_id:
                    raise ValueError("query provenance differs")
                original = np.array(x[query], copy=True)
                baseline, old_info = base._deform(original, fit, metric)
                if old_info["status"] != "geometry_only_not_human_approved" or not np.array_equal(baseline, by_query[query]):
                    raise ValueError("unchanged single-direction baseline is not bit-exact")
                baseline_rebuilt += 1
                selected = mode_indices(original, fit, metric, old_info["donor_fit_index"])
                accepted, units, seen = [], [], set()
                for mode, donor in enumerate(selected):
                    changed, info = deform_donor(original, fit, metric, donor)
                    if changed is None:
                        new_direction_failures += 1
                        continue
                    if mode == 0 and not np.array_equal(changed, baseline):
                        raise ValueError("primary donor changed baseline")
                    key = changed.tobytes()
                    if key in seen:
                        continue
                    seen.add(key)
                    movement = ((base._aligned(changed).ravel() - base._aligned(original).ravel()) @ metric["basis"].T) / np.sqrt(metric["eigen"])
                    units.append(movement / max(float(np.linalg.norm(movement)), 1e-12))
                    row = len(features); features.append(changed)
                    metadata.append(dict(candidate_id=f"D{row:05d}", candidate_row=row, source_class_id=class_id,
                        query_training_row=query, fit_training_rows=fit_ids.tolist(), donor_training_row=int(fit_ids[donor]),
                        direction_slot=mode, semantic_target=None, **info))
                    accepted.append(row)
                matrix = np.stack(units) if units else np.empty((0, metric["rank"]))
                cosines = matrix @ matrix.T
                pairs = cosines[np.triu_indices(len(matrix), 1)]
                seed_reports.append(dict(query_training_row=query, source_class_id=class_id,
                    selected_directions=len(selected), generated_directions=len(accepted), candidate_rows=accepted,
                    actual_direction_positive_cosine_max=float(pairs.max()) if len(pairs) else None,
                    actual_direction_span_rank=int(np.linalg.matrix_rank(matrix, tol=1e-5))))
        if (class_id + 1) % 40 == 0:
            print(json.dumps(dict(event="direction_diversity_progress", classes=class_id + 1, candidates=len(features))), flush=True)
    features = np.stack(features).astype(np.float32)
    np.save(out / "candidate_features.npy", features, allow_pickle=False)
    _write(out / "candidate_mapping.json", metadata)
    _write(out / "seed_direction_diagnostics.json", seed_reports)
    direction_counts = np.array([seed["generated_directions"] for seed in seed_reports])
    covered = set(item["source_class_id"] for item in metadata)
    labels = parent["class_labels"]
    qa, used = [], set()
    for name in ("digits", "latin_letters", "math_symbols"):
        seeds = [seed for seed in seed_reports if base._family(labels[seed["source_class_id"]]) == name and seed["generated_directions"] >= 2]
        if name == "math_symbols":
            seeds.sort(key=lambda seed: (-len(base._spans(x[seed["query_training_row"]])), seed["source_class_id"]))
        for seed in seeds:
            if seed["source_class_id"] in used:
                continue
            used.add(seed["source_class_id"])
            qa.extend(seed["candidate_rows"][:4])
            if sum(base._family(labels[c]) == name for c in used) >= 4:
                break
    base._render(features[qa], [metadata[i]["candidate_id"] for i in qa], out / "direction_diversity_blind_qa.png")
    summary = dict(vocabulary_classes=372, generated_classes=len(covered), baseline_seed_pairs=baseline_rebuilt,
        candidate_rows=len(features), increase_vs_single_direction=len(features) / baseline_rebuilt,
        generated_directions_histogram={str(i): int((direction_counts == i).sum()) for i in range(1, 5)},
        actual_direction_span_rank_histogram={str(i): sum(seed["actual_direction_span_rank"] == i for seed in seed_reports) for i in range(1, 5)},
        selected_directions_rejected_by_geometry=new_direction_failures,
        actual_realigned_q68_exceedances=sum(item["actual_projected_movement"] > item["q68_radius"] * (1 + 1e-6) for item in metadata),
        actual_direction_cosine_exceedances=sum(seed["actual_direction_positive_cosine_max"] is not None and seed["actual_direction_positive_cosine_max"] > MAX_COSINE + 1e-6 for seed in seed_reports),
        families={name: dict(classes=sum(base._family(labels[c]) == name for c in covered),
                            candidate_rows=sum(base._family(labels[item["source_class_id"]]) == name for item in metadata)) for name in ("digits", "latin_letters", "math_symbols")},
        excluded_classes=[labels[c] for c in range(372) if c not in covered],
        multi_stroke_candidates=sum(len(base._spans(row)) > 1 for row in features),
        point_stroke_candidates=sum(any(base._signature(row)) for row in features),
        human_semantic_labels=0, parameter_updates=0, model_forwards=0)
    files = ("candidate_features.npy", "candidate_mapping.json", "seed_direction_diagnostics.json", "direction_diversity_blind_qa.png")
    _write(out / "direction_diversity_result.json", dict(schema="aiflow-full-domain-direction-diversity-result/v14",
        status="geometry_only_not_human_approved", frozen_plan_sha256=_sha(out / "frozen_plan.json"), summary=summary,
        qa_candidate_rows=qa, artifacts={name: _sha(out / name) for name in files},
        semantic_accuracy_claim_allowed=False, training_allowed_from_this_audit=False, product_adopted=False))
    print(json.dumps(dict(event="direction_diversity_summary", **summary)), flush=True)
    return 0


def verify(out: Path) -> int:
    """생성 행마다 원본·donor를 직접 재구성하고 획/채널/기하와 q68 수치를 재계산한다."""
    if (out / "independent_verification.json").exists():
        raise FileExistsError("refusing verification overwrite")
    _, _, arrays, _ = checked_inputs()
    plan = json.loads((out / "frozen_plan.json").read_text(encoding="utf-8"))
    result = json.loads((out / "direction_diversity_result.json").read_text(encoding="utf-8"))
    if plan["script_sha256"] != _sha(Path(__file__)) or result["frozen_plan_sha256"] != _sha(out / "frozen_plan.json"):
        raise ValueError("frozen entrypoint/plan changed")
    if any(_sha(out / name) != digest for name, digest in result["artifacts"].items()):
        raise ValueError("output changed")
    features = np.load(out / "candidate_features.npy", allow_pickle=False)
    mapping = json.loads((out / "candidate_mapping.json").read_text(encoding="utf-8"))
    if features.shape != (len(mapping), 128, 5) or features.dtype != np.float32:
        raise ValueError("candidate cache contract failed")
    geometry_errors, baseline_count, cache = 0, 0, {}
    for i, item in enumerate(mapping):
        query, donor = item["query_training_row"], item["donor_training_row"]
        original = np.array(arrays["train_features"][query], copy=True)
        fit_ids = item["fit_training_rows"]
        class_id = item["source_class_id"]
        if query in fit_ids or donor not in fit_ids or item["semantic_target"] is not None:
            raise ValueError("provenance or no-hard-label contract failed")
        if not np.isin(arrays["train_sources"][[query, *fit_ids]], (0, 1)).all() or not (arrays["train_labels"][[query, *fit_ids]] == class_id).all():
            raise ValueError("non-real or cross-class input used")
        key = tuple(fit_ids)
        if key not in cache:
            fit = np.array(arrays["train_features"][fit_ids], copy=True)
            cache[key] = (fit, base._fit(fit))
        fit, metric = cache[key]
        geometry_errors += not base._geometry(original, features[i])["valid"]
        if base._signature(original) != base._signature(features[i]) or not np.array_equal(original[:, 2:], features[i, :, 2:]):
            raise ValueError("topology or non-XY changed")
        coeff = ((base._aligned(original).ravel() - metric["center"]) @ metric["basis"].T) / np.sqrt(metric["eigen"])
        distance = float(np.linalg.norm(metric["standardized"][fit_ids.index(donor)] - coeff))
        fraction = min(1., metric["radius"] / max(distance, 1e-8)) * item["scale"]
        rebuilt = original.copy()
        rebuilt[:, :2] += fraction * (base._aligned(arrays["train_features"][donor], original) - original[:, :2])
        if not np.array_equal(features[i], rebuilt) or abs(fraction - item["fraction"]) > 1e-12:
            raise ValueError("donor endpoint not bit-exact")
        if item["direction_slot"] == 0:
            baseline_count += 1
    if geometry_errors or baseline_count != result["summary"]["baseline_seed_pairs"]:
        raise ValueError("geometry integrity or baseline coverage failed")
    _write(out / "independent_verification.json", dict(schema="aiflow-direction-diversity-verification/v14", status="reproduced",
        result_sha256=_sha(out / "direction_diversity_result.json"), candidates=len(mapping),
        every_endpoint_rebuilt_bit_exact=True, unchanged_primary_donors=baseline_count, geometry_errors=0,
        source_and_fit_query_isolation_verified=True, non_xy_and_point_strokes_preserved=True,
        semantic_labels_available=False, model_forwards=0, parameter_updates=0, product_adopted=False))
    print(json.dumps(dict(event="direction_diversity_verification", status="reproduced", candidates=len(mapping), geometry_errors=0)), flush=True)
    return 0


def selftest() -> int:
    """동일 방향 donor가 많아도 다른 방향을 선택하며 기존 첫째 donor는 보존하는지 검증한다."""
    original = np.zeros((128, 5), dtype=np.float32)
    original[:, 0] = np.linspace(.2, .8, 128); original[:, 1] = .5
    original[0, 3] = 1
    rows = []
    for angle in np.linspace(0, 2 * np.pi, 32, endpoint=False):
        row = original.copy()
        row[:, 0] += .01 * np.cos(angle); row[:, 1] += .01 * np.sin(angle)
        rows.append(row)
    fit = np.stack(rows); metric = base._fit(fit)
    selected = mode_indices(original, fit, metric, 0)
    assert selected[0] == 0 and len(selected) == 4 and len(set(selected)) == 4
    for donor in selected:
        changed, info = deform_donor(original, fit, metric, donor)
        assert changed is not None and base._geometry(original, changed)["valid"]
        assert np.array_equal(changed[:, 2:], original[:, 2:])
    assert selected == mode_indices(original, fit, metric, 0)
    print(json.dumps(dict(event="direction_diversity_selftest", status="pass", diverse_modes=4)), flush=True)
    return 0


def main() -> int:
    """새 학습 없이 검증·생성·원본 재현을 명시적으로 실행한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("selftest", "audit", "verify"))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    return selftest() if args.mode == "selftest" else {"audit": generate, "verify": verify}[args.mode](args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
