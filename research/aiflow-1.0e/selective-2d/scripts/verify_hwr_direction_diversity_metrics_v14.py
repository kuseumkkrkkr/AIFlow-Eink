"""추가 방향·거절·실제 투영 q68·방향 span 수치를 원본 TRAIN에서 재계산한다."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import audit_hwr_full_domain_direction_diversity_v14 as audit
from run_hwr_pendigits_tube_probe_v1 import _sha, _write


def main() -> int:
    """기존 생성물을 바꾸지 않고 모든 seed의 선택과 기하 거절까지 재현한다."""
    out = audit.OUTPUT
    destination = out / "independent_direction_metric_verification.json"
    if destination.exists():
        raise FileExistsError("refusing independent metric certificate overwrite")
    if audit.base._guard_commit("before_direction_metric_verification") is None:
        return 78
    _, parent, arrays, _ = audit.checked_inputs()
    plan = json.loads((out / "frozen_plan.json").read_text(encoding="utf-8"))
    result = json.loads((out / "direction_diversity_result.json").read_text(encoding="utf-8"))
    if plan["script_sha256"] != _sha(Path(audit.__file__)) or result["frozen_plan_sha256"] != _sha(out / "frozen_plan.json"):
        raise ValueError("frozen entrypoint/plan changed")
    if any(_sha(out / name) != digest for name, digest in result["artifacts"].items()):
        raise ValueError("generated artifact changed")
    features = np.load(out / "candidate_features.npy", allow_pickle=False)
    mapping = json.loads((out / "candidate_mapping.json").read_text(encoding="utf-8"))
    seeds = json.loads((out / "seed_direction_diagnostics.json").read_text(encoding="utf-8"))
    cache, directions, ranks = {}, {}, {}
    radius_failures = cosine_failures = rejected = 0
    maximum_radius_ratio = 0.
    for seed in seeds:
        ids = seed["candidate_rows"]
        first = mapping[ids[0]]
        fit_ids = first["fit_training_rows"]
        key = tuple(fit_ids)
        if key not in cache:
            fit = np.array(arrays["train_features"][fit_ids], copy=True)
            cache[key] = (fit, audit.base._fit(fit))
        fit, metric = cache[key]
        original = np.array(arrays["train_features"][seed["query_training_row"]], copy=True)
        primary = fit_ids.index(first["donor_training_row"])
        selected = audit.mode_indices(original, fit, metric, primary)
        if len(selected) != seed["selected_directions"]:
            raise ValueError("selected direction count differs")
        observed = {mapping[i]["direction_slot"]: i for i in ids}
        for slot, donor in enumerate(selected):
            changed, _ = audit.deform_donor(original, fit, metric, donor)
            if changed is None:
                rejected += 1
                if slot in observed:
                    raise ValueError("invalid selected direction was saved")
            elif slot not in observed or not np.array_equal(changed, features[observed[slot]]):
                raise ValueError("selected endpoint missing or changed")
        vectors = []
        for i in ids:
            movement = (audit.base._aligned(features[i]).ravel() - audit.base._aligned(original).ravel()) @ metric["basis"].T
            movement /= np.sqrt(metric["eigen"])
            norm = float(np.sqrt(np.dot(movement, movement)))
            if abs(norm - mapping[i]["actual_projected_movement"]) > 1e-10:
                raise ValueError("reported projected movement differs")
            ratio = norm / metric["radius"]
            maximum_radius_ratio = max(maximum_radius_ratio, ratio)
            radius_failures += ratio > 1 + 1e-6
            vectors.append(movement / max(norm, 1e-12))
        matrix = np.stack(vectors)
        singular = np.linalg.svd(matrix, compute_uv=False)
        rank = int((singular > 1e-5).sum())
        maximum_cosine = max((float(np.dot(a, b)) for i, a in enumerate(matrix) for b in matrix[i + 1:]), default=None)
        if rank != seed["actual_direction_span_rank"] or maximum_cosine is not None and abs(maximum_cosine - seed["actual_direction_positive_cosine_max"]) > 1e-10:
            raise ValueError("actual direction geometry differs")
        cosine_failures += maximum_cosine is not None and maximum_cosine > audit.MAX_COSINE + 1e-6
        directions[str(len(ids))] = directions.get(str(len(ids)), 0) + 1
        ranks[str(rank)] = ranks.get(str(rank), 0) + 1
    summary = result["summary"]
    checks = dict(generated_directions_histogram={str(i): directions.get(str(i), 0) for i in range(1, 5)},
        actual_direction_span_rank_histogram={str(i): ranks.get(str(i), 0) for i in range(1, 5)},
        selected_directions_rejected_by_geometry=rejected, actual_realigned_q68_exceedances=int(radius_failures),
        actual_direction_cosine_exceedances=int(cosine_failures))
    if any(summary[name] != value for name, value in checks.items()):
        raise ValueError("reported selection/rejection/diversity metrics differ")
    _write(destination, dict(schema="aiflow-direction-metric-verification/v14", status="reproduced",
        verifier_sha256=_sha(Path(__file__)), result_sha256=_sha(out / "direction_diversity_result.json"),
        every_seed_selection_and_geometry_rejection_rebuilt=True, actual_projected_radius_ratio_max=maximum_radius_ratio,
        checks=checks, new_model_forwards=0, parameter_updates=0, human_semantic_approval=False))
    print(json.dumps(dict(event="independent_direction_metrics", status="reproduced", actual_projected_radius_ratio_max=maximum_radius_ratio, **checks)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
