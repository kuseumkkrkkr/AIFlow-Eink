"""372개 숫자·문자·기호의 클래스/획 위상별 경험적 변형 공간을 검증한다.

기존 real TRAIN만 사용한다. q68은 표본 간 거리 분위수이며 사람 인지 확률이
아니다. 다획/한 점 획과 비좌표 채널을 보존하며 사람 검토 전 학습하지 않는다.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from audit_hwr_probability_boundary_tube_v1 import DEFAULT_CHECKPOINT, DEFAULT_DATA_DIR, ROOT, _guard_commit
from run_hwr_pendigits_tube_probe_v1 import _sha, _write

OUTPUT = ROOT / "artifacts/hwr_full_domain_tube_20261005"
REFERENCE = ROOT / "artifacts/hwr_probability_boundary_map_20261004_r3/probability_boundary_map.json"
MIN_FIT, MAX_FIT, QUERY_COUNT = 24, 64, 2
SCALES = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125)
SEED = 20261005


def _spans(row: np.ndarray) -> list[tuple[int, int]]:
    """128개 타점의 획 시작을 그대로 분리하고 입력 계약을 확인한다."""
    if row.shape != (128, 5) or not np.isfinite(row).all() or row[:, :2].min() < 0 or row[:, :2].max() > 1:
        raise ValueError("invalid uniform-time [128,5] input")
    starts = np.flatnonzero(row[:, 3] > 0.5)
    if not len(starts) or starts[0] != 0 or len(starts) > 43:
        raise ValueError("invalid stroke starts")
    return list(zip(starts.tolist(), np.r_[starts[1:], 128].tolist()))


def _signature(row: np.ndarray) -> tuple[bool, ...]:
    """획 수와 순서별 점 획 여부를 분리하여 다른 위상의 보간을 막는다."""
    return tuple(bool(np.linalg.norm(np.diff(row[a:b, :2], axis=0), axis=1).sum() <= 1e-8) for a, b in _spans(row))


def _aligned(row: np.ndarray, target: np.ndarray | None = None) -> np.ndarray:
    """획 순서는 바꾸지 않고 각 획을 16점 또는 query의 원래 점 수로 보간한다."""
    spans = _spans(row)
    target_spans = _spans(target) if target is not None else [(0, 16)] * len(spans)
    if target is not None and _signature(row) != _signature(target):
        raise ValueError("donor/query topology differs")
    pieces = []
    for (a, b), (c, d) in zip(spans, target_spans):
        values = row[a:b, :2]
        old = np.linspace(0, 1, len(values))
        new = np.linspace(0, 1, d - c)
        pieces.append(np.stack([np.interp(new, old, values[:, axis]) for axis in (0, 1)], axis=1))
    return np.concatenate(pieces).astype(np.float64)


def _fit(rows: np.ndarray) -> dict:
    """클래스/위상별 저차원 공분산과 실제 표본 거리의 q68 반경을 구한다."""
    if len(rows) < MIN_FIT or len({_signature(row) for row in rows}) != 1:
        raise ValueError("insufficient or mixed topology fit")
    xy = np.stack([_aligned(row).ravel() for row in rows])
    center = np.median(xy, axis=0)
    centered = xy - center
    eigen, vectors = np.linalg.eigh(centered @ centered.T / (len(rows) - 1))
    order = np.argsort(-eigen)
    eigen, vectors = eigen[order], vectors[:, order]
    rank = int((eigen > max(eigen[0] * 1e-8, 1e-12)).sum())
    if rank < 2:
        raise ValueError("insufficient shape variation")
    k = min(rank, 12, max(2, int(np.searchsorted(np.cumsum(eigen[:rank]) / eigen[:rank].sum(), .95) + 1)))
    basis = (vectors[:, :k].T @ centered) / np.sqrt(eigen[:k, None] * (len(rows) - 1))
    standardized = (centered @ basis.T) / np.sqrt(eigen[:k])
    distances = np.linalg.norm(standardized[:, None] - standardized[None, :], axis=2)
    radius = float(np.quantile(distances[np.triu_indices(len(rows), 1)], .68))
    if not np.isfinite(radius) or radius <= 1e-8:
        raise ValueError("degenerate q68 radius")
    return dict(center=center, basis=basis, eigen=eigen[:k], standardized=standardized,
                radius=radius, rank=k, explained=float(eigen[:k].sum() / eigen[:rank].sum()))


def _geometry(original: np.ndarray, changed: np.ndarray) -> dict:
    """점 획/획별 경로/좌표 이동량을 검증하며 pen-up 점프는 경로에서 제외한다."""
    delta = changed[:, :2] - original[:, :2]
    ratios, point_preserved = [], True
    for a, b in _spans(original):
        before = float(np.linalg.norm(np.diff(original[a:b, :2], axis=0), axis=1).sum())
        after = float(np.linalg.norm(np.diff(changed[a:b, :2], axis=0), axis=1).sum())
        if before <= 1e-8:
            point_preserved &= after <= 1e-8
        else:
            ratios.append(after / before)
    non_xy = bool(np.array_equal(original[:, 2:], changed[:, 2:]))
    rms, maximum = float(np.sqrt((delta ** 2).mean())), float(np.linalg.norm(delta, axis=1).max())
    valid = bool(np.isfinite(changed).all() and changed[:, :2].min() >= 0 and changed[:, :2].max() <= 1
                 and non_xy and point_preserved and rms <= .035 and maximum <= .105
                 and all(.78 <= ratio <= 1.22 for ratio in ratios))
    return dict(valid=valid, rms=rms, max_point_displacement=maximum, stroke_path_ratios=ratios,
                non_xy_unchanged=non_xy, point_strokes_preserved=bool(point_preserved))


def _deform(original: np.ndarray, fit: np.ndarray, metric: dict) -> tuple[np.ndarray, dict]:
    """동일 클래스/위상 donor를 골라 q68 안에서 유효한 최대 보간만 생성한다."""
    coeff = ((_aligned(original).ravel() - metric["center"]) @ metric["basis"].T) / np.sqrt(metric["eigen"])
    distances = np.linalg.norm(metric["standardized"] - coeff, axis=1)
    # 기록된 획 순서와 진행 방향을 유지한다. 반대 방향 donor를 자동 뒤집지 않는다.
    q = _aligned(original).reshape(-1, 16, 2)
    compatible = []
    for i, row in enumerate(fit):
        donor = _aligned(row).reshape(-1, 16, 2)
        dot = ((q[:, -1] - q[:, 0]) * (donor[:, -1] - donor[:, 0])).sum(1)
        if (dot >= -1e-8).all():
            compatible.append(i)
    if not compatible:
        return original.copy(), dict(status="no_direction_compatible_donor", trials=[])
    within = [i for i in compatible if distances[i] <= metric["radius"]]
    index = max(within, key=lambda i: distances[i]) if within else min(compatible, key=lambda i: distances[i])
    fraction = min(1., metric["radius"] / max(float(distances[index]), 1e-8))
    donor_xy = _aligned(fit[index], original)
    trials = []
    for scale in SCALES:
        changed = original.copy()
        changed[:, :2] = original[:, :2] + scale * fraction * (donor_xy - original[:, :2])
        diagnostic = _geometry(original, changed)
        trials.append(dict(scale=scale, **diagnostic))
        if diagnostic["valid"] and diagnostic["rms"] > 1e-7:
            return changed, dict(status="geometry_only_not_human_approved", donor_fit_index=index,
                                 distance=float(distances[index]), fraction=scale * fraction, trials=trials)
    return original.copy(), dict(status="no_nonzero_valid_deformation", donor_fit_index=index, trials=trials)


def _family(label: str) -> str:
    """숫자/ASCII 문자 외의 토큰은 그리스 문자 등을 포함한 수학기호로 분류한다."""
    return "digits" if len(label) == 1 and label in "0123456789" else "latin_letters" if len(label) == 1 and label.isascii() and label.isalpha() else "math_symbols"


def _render(rows: np.ndarray, ids: list[str], path: Path) -> None:
    """정답이나 teacher 후보를 숨긴 실제 타점 표본을 같은 축척으로 렌더한다."""
    from PIL import Image, ImageDraw
    columns, tile = 8, 136
    canvas = Image.new("RGB", (columns * tile, ((len(rows) + columns - 1) // columns) * tile), "white")
    draw = ImageDraw.Draw(canvas)
    for i, row in enumerate(rows):
        left, top = i % columns * tile, i // columns * tile
        draw.text((left + 4, top + 4), ids[i], fill="black")
        for a, b in _spans(row):
            points = [(left + 12 + float(x) * 108, top + 20 + float(y) * 108) for x, y in row[a:b, :2]]
            if len(points) > 1:
                draw.line(points, fill="black", width=2)
            px, py = points[0]
            draw.ellipse((px - 1.5, py - 1.5, px + 1.5, py + 1.5), fill="black")
    canvas.save(path)


def _run(out: Path) -> int:
    """봉인된 실제 TRAIN 전체 클래스의 지원량·생성물·검토 패킷을 분리 기록한다."""
    if out.exists():
        raise FileExistsError("refusing to overwrite full-domain artifacts")
    if _guard_commit("before_full_domain_audit") is None:
        return 78
    import torch
    from run_hwr_affine_distillation_experiment_v1 import _load_teacher, _predict_logits
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    provenance = reference["provenance"]
    manifest = json.loads((DEFAULT_DATA_DIR / "prepared_manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "pass" or provenance["heldout_rows_read"] or provenance["crohme_rows"]:
        raise ValueError("TRAIN reference policy failed")
    if _sha(DEFAULT_CHECKPOINT) != provenance["teacher_checkpoint_sha256"] or _sha(DEFAULT_DATA_DIR / "prepared_manifest.json") != provenance["prepared_manifest_sha256"]:
        raise ValueError("canonical checkpoint/manifest changed")
    arrays = {}
    for name in ("train_features", "train_labels", "train_sources", "teacher_train_logits"):
        path = DEFAULT_DATA_DIR / f"{name}.npy"
        if _sha(path) != provenance[f"{name}_sha256"]:
            raise ValueError(f"sealed TRAIN array changed: {name}")
        arrays[name] = np.load(path, mmap_mode="r", allow_pickle=False)
    x, y, sources, cached = (arrays[n] for n in ("train_features", "train_labels", "train_sources", "teacher_train_logits"))
    model, labels, _ = _load_teacher(DEFAULT_CHECKPOINT, torch.device("cpu"))
    if x.shape != (len(y), 128, 5) or sources.shape != y.shape or cached.shape != (len(y), 372):
        raise ValueError("TRAIN arrays not aligned")
    out.mkdir(parents=True)
    plan = dict(schema="aiflow-hwr-full-domain-tube-plan/v1", class_labels=labels, script_sha256=_sha(Path(__file__)),
                reference_sha256=_sha(REFERENCE), provenance=provenance, seed=SEED, min_fit=MIN_FIT, max_fit=MAX_FIT,
                query_rows_per_topology=QUERY_COUNT, scales=SCALES, coverage=.68,
                metric="class/topology-local low-rank covariance; empirical pairwise q68, not Gaussian SD or human recognition probability",
                query_selection="near teacher global margin zero in REAL TRAIN; fit rows exclude query IDs",
                admitted_source_ids=[0, 1], synthetic_only_classes="report unsupported; never fabricate real support",
                semantic_approval="independent blind human votes required; no generated hard labels",
                student_training_performed=False, product_adopted=False, heldout_rows_read=0, crohme_rows=0)
    _write(out / "frozen_plan.json", plan)
    rng = np.random.default_rng(SEED)
    class_reports, pairs, mapping = [], [], []
    for class_id, label in enumerate(labels):
        indices = np.flatnonzero((y == class_id) & np.isin(sources, (0, 1)))
        buckets, invalid = {}, 0
        for row_id in indices:
            try:
                signature = _signature(x[row_id])
            except ValueError:
                invalid += 1
                continue
            buckets.setdefault(signature, []).append(int(row_id))
        groups = []
        for signature, ids in sorted(buckets.items(), key=lambda item: (-len(item[1]), item[0])):
            group = dict(stroke_count=len(signature), point_stroke_mask=signature, real_rows=len(ids))
            groups.append(group)
            if len(ids) < MIN_FIT + QUERY_COUNT:
                group["status"] = "insufficient_same_topology_real_support"
                continue
            z = np.array(cached[ids], dtype=np.float32)
            truth = z[:, class_id].copy(); z[:, class_id] = -np.inf
            query_ids = np.array(ids)[np.argsort(np.abs(truth - z.max(1)), kind="stable")[:QUERY_COUNT]]
            fit_ids = rng.permutation([row for row in ids if row not in set(query_ids.tolist())])[:MAX_FIT]
            try:
                fit = np.array(x[fit_ids], copy=True)
                metric = _fit(fit)
            except ValueError as error:
                group.update(status="unsupported_shape_metric", reason=str(error))
                continue
            group.update(status="metric_fitted", fit_row_ids=fit_ids.tolist(), query_row_ids=query_ids.tolist(),
                         q68_radius=metric["radius"], covariance_rank=metric["rank"], explained_variance=metric["explained"], transforms=[])
            for query_id in query_ids:
                original = np.array(x[query_id], copy=True)
                changed, diagnostic = _deform(original, fit, metric)
                group["transforms"].append(dict(query_row_id=int(query_id), **diagnostic))
                if diagnostic["status"] != "geometry_only_not_human_approved":
                    continue
                pair_id = len(pairs)
                pairs.append(np.stack((original, changed)))
                for endpoint in (0, 1):
                    mapping.append(dict(candidate_id=f"Q{2 * pair_id + endpoint:05d}", candidate_row=2 * pair_id + endpoint,
                                        class_id=class_id, query_training_row=int(query_id),
                                        donor_training_row=int(fit_ids[diagnostic["donor_fit_index"]]), endpoint="original" if endpoint == 0 else "deformed"))
        class_reports.append(dict(class_id=class_id, label=label, family=_family(label), real_rows=len(indices),
                                  invalid_rows=invalid, groups=groups,
                                  synthetic_rows=int(((y == class_id) & (sources == 2)).sum())))
        if (class_id + 1) % 40 == 0:
            print(json.dumps(dict(event="full_domain_geometry", classes=class_id + 1, pairs=len(pairs))), flush=True)
    features = np.stack(pairs).reshape(-1, 128, 5).astype(np.float32) if pairs else np.empty((0, 128, 5), dtype=np.float32)
    logits = _predict_logits(model, features, torch.device("cpu"), 32)
    artifacts = {}
    for name, values in (("candidate_features", features), ("teacher_logits", logits)):
        path = out / f"{name}.npy"; np.save(path, values, allow_pickle=False)
        artifacts[name] = dict(path=str(path.resolve()), shape=list(values.shape), dtype=str(values.dtype), sha256=_sha(path))
    _write(out / "blind_review_mapping.json", mapping)
    review_order = rng.permutation(len(mapping))
    with (out / "blind_human_review_queue.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["candidate_id", "human_label"])
        writer.writeheader(); writer.writerows(dict(candidate_id=mapping[i]["candidate_id"], human_label="") for i in review_order)
    for page, start in enumerate(range(0, len(mapping), 80)):
        order = review_order[start:start + 80]
        _render(features[order], [mapping[i]["candidate_id"] for i in order], out / f"blind_review_{page:03d}.png")
    # 시각 QA는 family와 다획/점 획을 포함하도록 별도 균형 표본을 고른다.
    qa, seen = [], set()
    for family, multistroke in (("digits", False), ("latin_letters", False), ("math_symbols", False), ("math_symbols", True)):
        chosen = 0
        for i in range(0, len(mapping), 2):
            item = mapping[i]; row = features[i]
            if _family(labels[item["class_id"]]) == family and (len(_spans(row)) > 1) == multistroke and item["class_id"] not in seen:
                qa.extend((i, i + 1)); seen.add(item["class_id"]); chosen += 1
                if chosen == 8:
                    break
    if qa:
        _render(features[qa], [mapping[i]["candidate_id"] for i in qa], out / "diverse_geometry_qa.png")
    pair_classes = np.array([item["class_id"] for item in mapping[::2]], dtype=np.int64)
    paired_logits = logits.reshape(-1, 2, 372)
    before = paired_logits[:, 0].argmax(1) == pair_classes
    after = paired_logits[:, 1].argmax(1) == pair_classes
    generated_classes = set(pair_classes.tolist())
    summary = dict(vocabulary_classes=len(labels), classes_with_real_support=sum(c["real_rows"] > 0 for c in class_reports),
                   classes_with_generated_views=len(generated_classes), generated_pairs=len(pairs), candidate_rows=len(features),
                   multi_stroke_pairs=sum(len(_spans(pair[0])) > 1 for pair in pairs),
                   point_stroke_pairs=sum(any(_signature(pair[0])) for pair in pairs),
                   families={family: dict(vocabulary_classes=sum(c["family"] == family for c in class_reports),
                                         generated_classes=sum(c["family"] == family and c["class_id"] in generated_classes for c in class_reports))
                             for family in ("digits", "latin_letters", "math_symbols")},
                   unsupported_real_classes=[c["label"] for c in class_reports if not c["real_rows"]],
                   geometry_integrity_errors=sum(not _geometry(pair[0], pair[1])["valid"] for pair in pairs),
                   teacher_top1_retained=int((before & after).sum()), teacher_top1_lost=int((before & ~after).sum()),
                   teacher_top1_recovered=int((~before & after).sum()), human_boundary_labels=0,
                   note="Teacher TRAIN diagnostics, not recognition accuracy or human boundary estimates")
    packet_files = [out / "blind_review_mapping.json", out / "blind_human_review_queue.csv", *sorted(out.glob("blind_review_*.png"))]
    result = dict(schema="aiflow-hwr-full-domain-tube-audit/v1", status="geometry_and_teacher_probe_only",
                  frozen_plan_sha256=_sha(out / "frozen_plan.json"), provenance=dict(teacher_checkpoint_sha256=provenance["teacher_checkpoint_sha256"],
                    heldout_rows_read=0, crohme_rows=0, official_test_rows_read=0, real_training_source_ids=[0, 1]),
                  candidate_cache=dict(class_labels=labels, artifacts=artifacts), summary=summary, classes=class_reports,
                  qa_candidate_rows=qa, review_packet_sha256={p.name: _sha(p) for p in packet_files},
                  human_targets_available=False, hard_synthetic_labels_assigned=0, student_training_performed=False, product_adopted=False)
    if _sha(DEFAULT_CHECKPOINT) != provenance["teacher_checkpoint_sha256"]:
        raise AssertionError("canonical checkpoint changed")
    _write(out / "full_domain_tube_audit.json", result)
    print(json.dumps(dict(event="full_domain_audit_complete", summary=summary)), flush=True)
    return 0


def _self_test() -> int:
    """다획·점 획·pen-up 제외·채널 불변·mixed 위상 거부를 수치로 검증한다."""
    u = np.linspace(0, 1, 128, dtype=np.float32)
    row = np.zeros((128, 5), dtype=np.float32)
    row[:, 0] = .15 + .65 * u; row[:, 1] = .3 + .2 * np.sin(np.pi * u)
    row[:, 2] = u; row[:, 4] = 1; row[[0, 48, 96], 3] = 1
    row[48:96, :2] = [.4, .6]
    assert _signature(row) == (False, True, False)
    assert _aligned(row, row).shape == (128, 2)
    rows = []
    for i in range(32):
        item = row.copy()
        item[:, 0] += .003 * np.sin(i + u * 3); item[:, 1] += .003 * np.cos(i * 2 + u * 2)
        item[48:96, :2] = item[48, :2]
        rows.append(item)
    fit = np.stack(rows); metric = _fit(fit)
    changed, diagnostic = _deform(row, fit, metric)
    assert diagnostic["status"] == "geometry_only_not_human_approved" and _geometry(row, changed)["valid"]
    assert np.array_equal(changed[:, 2:], row[:, 2:]) and _signature(changed) == _signature(row)
    identical, repeat = _deform(row, fit, metric)
    assert np.array_equal(changed, identical) and diagnostic == repeat
    bad = row.copy(); bad[50, 3] = 1
    try:
        _aligned(bad, row)
    except ValueError:
        pass
    else:
        raise AssertionError("mixed topology accepted")
    point = np.tile(row[48], (128, 1)); point[:, 3] = 0; point[0, 3] = 1
    assert _signature(point) == (True,) and _geometry(point, point)["valid"]
    print(json.dumps(dict(self_test="pass", multi_stroke=True, point_stroke=True, non_xy_exact=True, deterministic=True)))
    return 0


def main() -> int:
    """제품과 격리된 전 범위 감사 또는 입력 무결성 단위 검증을 실행한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("self-test", "audit"))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    return _self_test() if args.mode == "self-test" else _run(args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
