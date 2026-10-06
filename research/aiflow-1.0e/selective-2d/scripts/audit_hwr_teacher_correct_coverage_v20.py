"""V20의 teacher 정답 mask가 전 클래스·증강 강도·입력 출처를 얼마나 남기는지 검사한다."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import run_hwr_broad_teacher_correct_v20 as run


def main():
    """허용 TRAIN만 읽고 가중치 변경 없이 실제 변형과 KL 적용 범위를 전체 집계한다."""
    output = run.OUT / "kl_mask_coverage.json"
    if output.exists():
        raise FileExistsError("coverage certificate exists")
    plan,a = run.load()
    parent = json.loads((run.broad.OUT / "frozen_plan.json").read_text(encoding="utf-8"))
    labels = parent["class_labels"]
    y = a["population_labels"]
    good = a["population_teacher_logits"].argmax(1)==y
    schedule = a["shared_schedule"]
    original = np.array(y[schedule])
    keep = np.array(good[schedule])
    changed = np.zeros(schedule.shape,dtype=bool)
    rms = np.zeros(schedule.shape,dtype=np.float64)
    maximum = np.zeros(schedule.shape,dtype=np.float64)
    for start in range(0,len(schedule),64):
        stop = min(start+64,len(schedule))
        real = np.array(a["population_features"][schedule[start:stop]],copy=True)
        view = np.array(a["scheduled_augmented_features"][start:stop],copy=True)
        assert np.isfinite(view).all() and np.array_equal(real[...,2:],view[...,2:])
        assert (view[...,:2]>=0).all() and (view[...,:2]<=1).all()
        displacement = np.linalg.norm(view[...,:2].astype(np.float64)-real[...,:2],axis=-1)
        changed[start:stop] = np.any(view!=real,axis=(2,3))
        # 기존 cap은 점 이동거리 RMS가 아니라 x/y 두 좌표 성분의 RMS다.
        rms[start:stop] = np.sqrt(np.mean(displacement**2,axis=-1)/2.)
        maximum[start:stop] = displacement.max(axis=-1)
    assert int((~keep).sum())==plan["expected_incorrect_original_exposures"]==14952
    assert int(changed.sum())==68726
    counts = [np.bincount(original[mask],minlength=372) for mask in
        (np.ones(schedule.shape,bool),keep,changed,changed & keep)]
    rows = [dict(class_id=int(c),label=labels[c],family=run.broad.previous.aug.base._family(labels[c]),
        real_ce_exposures=int(counts[0][c]),original_kl_exposures=int(counts[1][c]),
        changed_views_before_mask=int(counts[2][c]),changed_views_with_kl=int(counts[3][c]),
        kl_retention_fraction=float(counts[1][c]/counts[0][c])) for c in np.flatnonzero(counts[0])]
    temperatures = np.array([[run.broad.TEMPERATURES[(step//3+column)%4]
        for column in range(run.broad.BATCH)] for step in range(run.broad.STEPS)])
    temperature_rows = []
    for t in run.broad.TEMPERATURES:
        mask = temperatures==t
        retained = mask & keep & changed
        values = rms[retained]
        temperature_rows.append(dict(shape_temperature=t,original_exposures=int(mask.sum()),
            retained_kl_exposures=int((mask & keep).sum()),retained_nonzero_views=int(retained.sum()),
            retained_nonzero_view_classes=int(np.unique(original[retained]).size),
            retained_view_rms_median=float(np.median(values)),retained_view_rms_p95=float(np.quantile(values,.95))))
    origin = a["population_origin"][schedule]
    source_rows = {parent["source_codes"][str(int(s))]:dict(real_ce=int((origin==s).sum()),
        retained_kl=int(((origin==s)&keep).sum()),retained_nonzero_views=int(((origin==s)&keep&changed).sum())) for s in np.unique(origin)}
    report = dict(schema="aiflow-teacher-correct-coverage/v20",status="pass_train_coverage_not_accuracy",
        script_sha256=run.broad.previous._sha(Path(__file__)),frozen_plan_sha256=run.broad.previous._sha(run.OUT / "frozen_plan.json"),
        real_ce_classes=int((counts[0]>0).sum()),original_kl_classes=int((counts[1]>0).sum()),
        nonzero_augmented_classes_before_mask=int((counts[2]>0).sum()),nonzero_augmented_classes_after_mask=int((counts[3]>0).sum()),
        total_real_exposures=int(counts[0].sum()),total_retained_kl_exposures=int(counts[1].sum()),
        nonzero_views_before_mask=int(counts[2].sum()),nonzero_views_after_mask=int(counts[3].sum()),
        classes_losing_all_nonzero_view_kl=[labels[c] for c in np.flatnonzero((counts[2]>0)&(counts[3]==0))],
        classes_without_original_kl=[labels[c] for c in np.flatnonzero((counts[0]>0)&(counts[1]==0))],
        low_retention_classes=[row for row in rows if row["kl_retention_fraction"]<.25],
        min_retained_exposures=int(counts[1][counts[0]>0].min()),max_retained_exposures=int(counts[1].max()),
        rms_max=float(rms.max()),rms_definition="sqrt(mean(delta_x^2,delta_y^2)); per-coordinate RMS, not Euclidean point RMS",
        euclidean_point_rms_max=float(np.sqrt(2.)*rms.max()),point_displacement_max=float(maximum.max()),non_xy_channels_preserved=True,
        audit_definition_correction="First audit attempt used Euclidean RMS against a per-coordinate cap and failed before writing a certificate; training/data unchanged.",
        temperatures=temperature_rows,sources=source_rows,classes=rows,
        all_real_ce_exposures_unchanged=True,optimizer_steps=0,development_folds_read=0,official_test_rows_read=0,
        crohme_rows=0,synthetic_hard_labels=0,human_semantic_approval=False,product_adopted=False,
        limit="Top1-correctness mask can weaken augmentation on difficult classes; this coverage audit does not prove semantic invariance or accuracy gains.")
    assert report["real_ce_classes"]==371 and report["total_real_exposures"]==76800
    assert report["rms_max"]<=.035+1e-6 and report["point_displacement_max"]<=.105+1e-6
    run.broad.previous._write(output,report)
    summary = {key:value for key,value in report.items() if key not in ("classes","sources","low_retention_classes")}
    summary["low_retention_classes"] = report["low_retention_classes"]
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
