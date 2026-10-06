"""평가 라벨을 읽지 않고 승인 TRAIN 분포와 실제 증강 학습 노출량을 비교한다."""
from __future__ import annotations

import json
import string
from pathlib import Path

import numpy as np
import run_hwr_direct_augmented_main_v25 as direct

OUT=direct.broad.previous.ROOT / "artifacts/hwr_sampling_balance_20261006_v30"
FAMILIES=("digits","latin_letters","math_symbols")


def describe(labels,vocabulary,families):
    """372개 클래스와 3개 family의 실제 행 수를 독립 집계와 대조한다."""
    labels=np.asarray(labels,dtype=np.int64).reshape(-1)
    assert len(labels)>0 and labels.min()>=0 and labels.max()<len(vocabulary)
    counts=np.bincount(labels,minlength=len(vocabulary))
    result={f:dict(rows=int(counts[families==f].sum()),fraction=float(counts[families==f].sum()/len(labels)),
        positive_classes=int((counts[families==f]>0).sum())) for f in FAMILIES}
    assert sum(r["rows"] for r in result.values())==len(labels)
    for f in FAMILIES:
        assert int(np.count_nonzero(families[labels]==f))==result[f]["rows"]
    return dict(rows=len(labels),positive_classes=int((counts>0).sum()),family=result),counts


def main():
    """원본 TRAIN·선택 원본·실제 변경 view·최종 CE schedule의 분포를 봉인한다."""
    if OUT.exists():
        raise FileExistsError("refusing balance audit overwrite")
    plan,a,s=direct.load();sha=direct.broad.previous._sha;write=direct.broad.previous._write
    population_plan=json.loads((direct.broad.OUT / "frozen_plan.json").read_text(encoding="utf-8"))
    vocabulary=population_plan["class_labels"];assert len(vocabulary)==372
    families=np.array([direct.broad.previous.aug.base._family(v) for v in vocabulary])
    independent=np.array(["digits" if v in set(string.digits) and len(v)==1 else
        "latin_letters" if len(v)==1 and v in set(string.ascii_letters) else "math_symbols" for v in vocabulary])
    assert np.array_equal(families,independent)
    y=a["population_labels"];ancestor=a["shared_schedule"].reshape(-1)
    real=y[s["original_population_indices"]]
    aug_source=ancestor[s["augmented_view_indices"]];aug=y[aug_source]
    expected=np.concatenate((real,aug),axis=1)
    assert np.array_equal(expected,s["shared_target_ids"]) and expected.shape==(2400,64)
    views=np.unique(s["augmented_view_indices"]);assert len(views)==68726
    inputs=dict(admitted_real_population=y,ancestor_balanced_schedule=y[ancestor],
        direct_real_region=real,direct_augmented_region=aug,direct_all_hard_targets=expected,
        distinct_augmented_view_slots=y[ancestor[views]])
    profiles={};counts={}
    for name,value in inputs.items():
        profiles[name],counts[name]=describe(value,vocabulary,families)
    assert profiles["admitted_real_population"]["rows"]==163982
    assert profiles["direct_real_region"]["rows"]==38400 and profiles["direct_augmented_region"]["rows"]==115200
    assert profiles["direct_all_hard_targets"]["positive_classes"]==371
    assert np.array_equal(counts["direct_real_region"]+counts["direct_augmented_region"],counts["direct_all_hard_targets"])
    comparison={}
    for f in FAMILIES:
        base=profiles["admitted_real_population"]["family"][f]["fraction"]
        actual=profiles["direct_all_hard_targets"]["family"][f]["fraction"]
        comparison[f]=dict(train_fraction=base,hard_target_fraction=actual,relative_exposure=actual/base,
            delta_percentage_points=100*(actual-base),class_count=int(np.count_nonzero(families==f)))
    OUT.mkdir()
    write(OUT / "frozen_plan.json",dict(schema="aiflow-sampling-balance-plan/v30",script_sha256=sha(Path(__file__)),
        source_plan_sha256=sha(direct.broad.OUT / "frozen_plan.json"),direct_plan_sha256=sha(direct.OUT / "frozen_plan.json"),
        direct_schedule_sha256=plan["arrays_sha256"],family_definition="ASCII digits; plain one-character ASCII letters; all remaining tokens including Greek/font variants are math_symbols",
        evaluation_inputs_read=0,owned_labels_read=0,optimizer_steps=0,crohme_rows=0,product_adopted=False))
    write(OUT / "per_class_exposure.json",[dict(class_id=i,label=vocabulary[i],family=str(families[i]),
        counts={name:int(values[i]) for name,values in counts.items()}) for i in range(372)])
    origin=a["population_origin"]
    source_counts={str(int(code)):dict(population_rows=int((origin==code).sum()),
        hard_target_exposures=int((origin[s["original_population_indices"]]==code).sum()+(origin[aug_source]==code).sum()))
        for code in np.unique(origin)}
    result=dict(schema="aiflow-sampling-balance-result/v30",status="verified_train_only_distribution_audit",profiles=profiles,
        comparison=comparison,source_counts=source_counts,missing_positive_train_labels=[vocabulary[i] for i,n in enumerate(counts["admitted_real_population"]) if n==0],
        missing_augmented_slot_labels=[vocabulary[i] for i,n in enumerate(counts["distinct_augmented_view_slots"]) if n==0],
        label_schedule_reproduced=True,family_rule_independently_reproduced=True,class_family_totals_crosschecked=True,
        unchanged_generation_and_training_artifacts=True,evaluation_inputs_read=0,owned_labels_read=0,optimizer_steps=0,crohme_rows=0,
        product_adopted=False,limits="TRAIN population frequency is not an established app-user or acceptance distribution. Family imbalance is measured, not proof of the causal source of accuracy regression.")
    write(OUT / "balance_result.json",result)
    print(json.dumps(dict(event="sampling_balance_verified",comparison=comparison,source_counts=source_counts,
        missing_positive_train_labels=result["missing_positive_train_labels"])),flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
