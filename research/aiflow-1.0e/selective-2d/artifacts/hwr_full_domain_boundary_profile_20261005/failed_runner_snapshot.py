"""전 범위 변형 경로를 여러 강도로 검사해 teacher 경계와 비단조성을 찾는다.

사람 인지 경계와 다르다. 단조성을 가정하지 않고 각 유효 구간의 전체 372-way
margin 부호를 확인한다. 검토 패킷은 클래스마다 한 쌍, 정답 투표는 빈칸이다.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from audit_hwr_full_domain_tube_v1 import OUTPUT as SOURCE, ROOT, DEFAULT_CHECKPOINT, _geometry, _render, _signature, _guard_commit
from hwr_full_domain_replay_v1 import load_full_domain_candidates
from run_hwr_pendigits_tube_probe_v1 import _sha, _write

OUTPUT = ROOT / "artifacts/hwr_full_domain_boundary_profile_20261005"
PHASES = np.linspace(0., 1., 9, dtype=np.float32)
REFINE_STEPS = 10
SEED = 20261007


def _interpolate(paths: np.ndarray, phases: np.ndarray) -> np.ndarray:
    """좌표만 원본과 봉인된 최대 변형 사이에서 보간하고 다른 채널은 정확히 유지한다."""
    if paths.shape[1:] != (2, 128, 5) or phases.shape != (len(paths),) or not np.isfinite(phases).all() or (phases < 0).any() or (phases > 1).any():
        raise ValueError("invalid paths/phase alignment")
    result = paths[:, 0].copy()
    result[:, :, :2] += phases[:, None, None] * (paths[:, 1, :, :2] - paths[:, 0, :, :2])
    if not np.array_equal(result[:, :, 2:], paths[:, 0, :, 2:]):
        raise AssertionError("non-XY channels changed")
    return result


def _global_margin(logits: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """지정 클래스와 특정 rival의 차이가 아니라 모든 rival 중 최고 점수를 비교한다."""
    if logits.ndim != 2 or logits.shape[1] != 372 or classes.shape != (len(logits),):
        raise ValueError("global margin requires aligned [N,372] logits/classes")
    other = logits.copy(); target = other[np.arange(len(other)), classes].copy()
    other[np.arange(len(other)), classes] = -np.inf
    return target - other.max(1)


def _crossings(margins: np.ndarray, valid: np.ndarray) -> list[int]:
    """기하가 유효한 인접 구간만 반환한다. 끝점 부호가 같아도 내부 재진입을 찾는다."""
    if margins.shape != valid.shape or margins.ndim != 1:
        raise ValueError("margin/geometry mask misaligned")
    return [i for i in range(len(margins) - 1) if valid[i] and valid[i + 1] and (margins[i] > 0) != (margins[i + 1] > 0)]


def _refine(paths: np.ndarray, classes: np.ndarray, positive: np.ndarray, nonpositive: np.ndarray, predict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """양쪽 teacher 부호를 유지하며 실제 타점/372-way 점수를 다시 검사한다."""
    positive, nonpositive = positive.copy(), nonpositive.copy()
    active = np.ones(len(paths), dtype=bool)
    for step in range(REFINE_STEPS):
        indices = np.flatnonzero(active)
        if not len(indices):
            break
        mid = (positive[indices] + nonpositive[indices]) * .5
        features = _interpolate(paths[indices], mid)
        valid = np.array([_geometry(path[0], row)["valid"] for path, row in zip(paths[indices], features)])
        active[indices[~valid]] = False
        eligible = indices[valid]
        margins = _global_margin(predict(features[valid]), classes[eligible]) if len(eligible) else np.empty(0)
        positive[eligible[margins > 0]] = mid[valid][margins > 0]
        nonpositive[eligible[margins <= 0]] = mid[valid][margins <= 0]
        print(json.dumps(dict(event="full_domain_boundary_refinement", step=step + 1, active_pairs=int(active.sum()))), flush=True)
    return positive, nonpositive, active


def _run(out: Path) -> int:
    """1121개 TRAIN 경로를 검사하고 370클래스에 동일한 사람 검토 예산을 할당한다."""
    if out.exists():
        raise FileExistsError("refusing to overwrite boundary profile")
    if _guard_commit("before_full_domain_boundary_profile") is None:
        return 78
    import torch
    from run_hwr_affine_distillation_experiment_v1 import _load_teacher, _predict_logits
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    sha = _sha(DEFAULT_CHECKPOINT)
    model, labels, _ = _load_teacher(DEFAULT_CHECKPOINT, torch.device("cpu"))
    parent_path = SOURCE / "full_domain_tube_audit.json"
    parent, paths, cached, classes = load_full_domain_candidates(parent_path, sha, labels)
    parent_mapping = json.loads((SOURCE / "blind_review_mapping.json").read_text(encoding="utf-8"))
    out.mkdir(parents=True)
    plan = dict(schema="aiflow-full-domain-boundary-profile-plan/v1", script_sha256=_sha(Path(__file__)),
                dependencies={n:_sha(ROOT / "scripts" / n) for n in ("hwr_full_domain_replay_v1.py", "audit_hwr_full_domain_tube_v1.py", "run_hwr_affine_distillation_experiment_v1.py")},
                source_report_sha256=_sha(parent_path), teacher_checkpoint_sha256=sha, phases=PHASES.tolist(),
                refinement_steps=REFINE_STEPS, selection="one pair per available class: first geometry-valid sign-changing parent path, otherwise first original/deformed control",
                boundary_rule="target minus maximum of ALL other 371 classes; local sign bracketing, no global monotonicity assumption",
                seed=SEED, heldout_rows_read=0, crohme_rows=0, human_boundary_labels=0, training_steps=0, product_adopted=False)
    _write(out / "frozen_plan.json", plan)
    grid_features = np.stack([_interpolate(paths, np.full(len(paths), phase, dtype=np.float32)) for phase in PHASES], axis=1)
    valid = np.array([[_geometry(path[0], row)["valid"] for row in grid] for path, grid in zip(paths, grid_features)], dtype=bool)
    grid_logits = np.full((len(paths), len(PHASES), 372), np.nan, dtype=np.float32)
    flat_ids = np.flatnonzero(valid.ravel())
    flat_logits = grid_logits.reshape(-1,372)
    flat_features = grid_features.reshape(-1,128,5)
    for start in range(0,len(flat_ids),512):
        selected = flat_ids[start:start + 512]
        flat_logits[selected] = _predict_logits(model, flat_features[selected], torch.device("cpu"), 32)
        print(json.dumps(dict(event="full_domain_boundary_grid", forwarded=min(start + 512,len(flat_ids)), total=len(flat_ids))), flush=True)
    margins = np.full(valid.shape, np.nan, dtype=np.float32)
    flat_classes = np.repeat(classes, len(PHASES))
    margins.ravel()[flat_ids] = _global_margin(flat_logits[flat_ids], flat_classes[flat_ids])
    crossings = [_crossings(row, mask) for row,mask in zip(margins,valid)]
    chosen, selected_classes, kinds, positive, nonpositive = [], [], [], [], []
    for c in np.unique(classes):
        candidates = np.flatnonzero(classes == c)
        routed = [i for i in candidates if crossings[i]]
        index = int(routed[0] if routed else candidates[0])
        chosen.append(index); selected_classes.append(int(c)); kinds.append(bool(routed))
        if routed:
            edge = crossings[index][0]
            first, second = PHASES[edge:edge + 2]
            pos, neg = (first,second) if margins[index,edge] > 0 else (second,first)
        else:
            pos, neg = 0., 1.
        positive.append(pos); nonpositive.append(neg)
    chosen = np.array(chosen, dtype=np.int64); selected_classes = np.array(selected_classes, dtype=np.int64)
    kinds = np.array(kinds, dtype=bool); positive = np.array(positive, dtype=np.float32); nonpositive = np.array(nonpositive, dtype=np.float32)
    routed = np.flatnonzero(kinds)
    if len(routed):
        pos, neg, active = _refine(paths[chosen[routed]], selected_classes[routed], positive[routed], nonpositive[routed],
                                  lambda x:_predict_logits(model,x,torch.device("cpu"),32))
        positive[routed],nonpositive[routed] = pos,neg
        # 기하 중단은 숨기지 않는다. 그런 클래스는 원래의 control 쌍으로 돌린다.
        failed = routed[~active]; kinds[failed] = False; positive[failed] = 0.; nonpositive[failed] = 1.
    features = np.stack((_interpolate(paths[chosen], positive), _interpolate(paths[chosen], nonpositive)),axis=1).reshape(-1,128,5)
    logits = _predict_logits(model,features,torch.device("cpu"),32)
    pair_logits = logits.reshape(-1,2,372)
    first_margin = _global_margin(pair_logits[:,0],selected_classes)
    second_margin = _global_margin(pair_logits[:,1],selected_classes)
    bracket_valid = (first_margin > 0) & (second_margin <= 0)
    if not bracket_valid[kinds].all():
        raise AssertionError("refined candidates lost global teacher sign bracket after final forward")
    for i,row in enumerate(features):
        original = paths[chosen[i // 2],0]
        if not _geometry(original,row)["valid"] or _signature(original) != _signature(row):
            raise AssertionError("review candidate geometry/topology failed")
    cache = {}
    for name,array in (("candidate_features",features),("teacher_logits",logits),("selected_parent_pair_indices",chosen),
                       ("candidate_phase_fractions",np.stack((positive,nonpositive),1)),("teacher_bracket_mask",kinds),
                       ("grid_geometry_valid",valid),("grid_teacher_logits",grid_logits),("grid_global_margins",margins)):
        path = out / f"{name}.npy"; np.save(path,array,allow_pickle=False)
        cache[name] = dict(path=str(path.resolve()),shape=list(array.shape),dtype=str(array.dtype),sha256=_sha(path))
    mapping = []
    for i,parent_id in enumerate(chosen):
        for side in (0,1):
            mapping.append(dict(candidate_id=f"R{2*i+side:05d}",candidate_row=2*i+side,class_id=int(selected_classes[i]),
                                query_training_row=parent_mapping[2*parent_id]["query_training_row"],donor_training_row=parent_mapping[2*parent_id]["donor_training_row"],
                                parent_pair_index=int(parent_id),phase=float((positive if side==0 else nonpositive)[i]),
                                kind="teacher_boundary_not_human_boundary" if kinds[i] else "original_deformation_control",
                                endpoint=("teacher_positive" if side==0 else "teacher_nonpositive") if kinds[i] else ("original" if side==0 else "deformed")))
    _write(out / "blind_review_mapping.json",mapping)
    rng=np.random.default_rng(SEED);order=rng.permutation(len(features))
    with (out / "blind_human_review_queue.csv").open("x",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=["candidate_id","human_label"]);writer.writeheader()
        writer.writerows(dict(candidate_id=mapping[i]["candidate_id"],human_label="") for i in order)
    for page,start in enumerate(range(0,len(features),80)):
        page_ids=order[start:start+80]
        _render(features[page_ids],[mapping[i]["candidate_id"] for i in page_ids],out / f"blind_review_{page:03d}.png")
    review_files=[out / "blind_review_mapping.json",out / "blind_human_review_queue.csv",*sorted(out.glob("blind_review_*.png"))]
    per_class = []
    for c in np.unique(classes):
        mask=classes==c
        per_class.append(dict(class_id=int(c),label=labels[c],paths=int(mask.sum()),paths_with_crossings=sum(bool(crossings[i]) for i in np.flatnonzero(mask)),
                              sampled_grid_positive_counts=[int((valid[mask,j] & (margins[mask,j] > 0)).sum()) for j in range(len(PHASES))]))
    interior_only=[i for i,edges in enumerate(crossings) if edges and (margins[i,0]>0)==(margins[i,-1]>0)]
    summary=dict(classes_scanned=370,paths_scanned=len(paths),valid_grid_inputs=int(valid.sum()),invalid_grid_inputs_not_forwarded=int((~valid).sum()),
                 paths_with_teacher_crossings=sum(bool(edges) for edges in crossings),multiple_crossing_paths=sum(len(edges)>1 for edges in crossings),
                 endpoint_only_scan_missed_paths=len(interior_only),refined_review_pairs=int(kinds.sum()),control_review_pairs=int((~kinds).sum()),
                 review_classes=370,review_rows=len(features),bracket_width_max=float(np.abs(positive[kinds]-nonpositive[kinds]).max()) if kinds.any() else None,
                 refinement_is_local_not_global_monotonic=True,human_boundary_labels=0)
    report=dict(schema="aiflow-hwr-full-domain-boundary-profile/v1",status="teacher_boundary_review_candidates_only",summary=summary,
                frozen_plan_sha256=_sha(out / "frozen_plan.json"),source_report_sha256=_sha(parent_path),
                provenance=dict(teacher_checkpoint_sha256=sha,real_training_source_ids=[0,1],heldout_rows_read=0,official_test_rows_read=0,crohme_rows=0),
                candidate_cache=dict(class_labels=labels,artifacts={name:cache[name] for name in ("candidate_features","teacher_logits")}),
                profile_cache=cache,classes=per_class,endpoint_only_missed_parent_pair_indices=interior_only,
                review_packet_sha256={p.name:_sha(p) for p in review_files},human_targets_available=False,
                hard_synthetic_labels_assigned=0,student_training_performed=False,product_adopted=False,
                limitations="Nine points only sample the path; no crossing observed does not prove no boundary. Teacher-selected candidates are biased and cannot estimate population human recognition probabilities.")
    if _sha(DEFAULT_CHECKPOINT)!=sha:
        raise AssertionError("canonical checkpoint changed")
    _write(out / "boundary_profile.json",report)
    print(json.dumps(dict(event="full_domain_boundary_profile_complete",summary=summary)),flush=True)
    return 0


def _self_test() -> int:
    """제3 rival·재진입·기하 구멍·원본 채널 보존을 검증한다."""
    logits=np.zeros((2,372),dtype=np.float32);logits[:,0]=[1.,3.];logits[:,1]=.5;logits[:,2]=2.
    assert np.array_equal(_global_margin(logits,np.array([0,0])),[-1.,1.])
    margin=np.array([1.,-.2,.3,1.]);valid=np.ones(4,dtype=bool)
    assert _crossings(margin,valid)==[0,1]
    valid[1]=False;assert _crossings(margin,valid)==[]
    x=np.zeros((2,2,128,5),dtype=np.float32);x[:,:,:,0]=.5;x[:,:,:,1]=.5;x[:,:,0,3]=1.;x[:,:,:,4]=1.
    x[:,1,:,0]+=.02
    assert np.array_equal(_interpolate(x,np.zeros(2,dtype=np.float32)),x[:,0])
    out=_interpolate(x,np.ones(2,dtype=np.float32));assert np.allclose(out,x[:,1]) and np.array_equal(out[:,:,2:],x[:,0,:,2:])
    print(json.dumps(dict(self_test="pass",global_not_fixed_pair=True,nonmonotonic_reentry_detected=True,invalid_geometry_not_bridged=True)))
    return 0


def main() -> int:
    """제품과 분리된 경로 감사 또는 부호/기하 단위 검증을 실행한다."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=("self-test","audit"),required=True)
    parser.add_argument("--output-dir",type=Path,default=OUTPUT)
    args=parser.parse_args()
    return _self_test() if args.mode=="self-test" else _run(args.output_dir)


if __name__=="__main__":
    raise SystemExit(main())
