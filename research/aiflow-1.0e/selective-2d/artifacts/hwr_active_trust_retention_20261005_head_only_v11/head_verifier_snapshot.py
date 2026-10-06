"""원 parent embedding·선형 head 기록으로 모든 v11 점수 및 미분 Gram을 독립 재구성한다."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from run_hwr_active_trust_retention_v1 import ROOT, REFERENCE, CANDIDATE, DEFAULT_CHECKPOINT, _sha, _write

OUTPUT = ROOT / "artifacts/hwr_active_trust_retention_20261005_head_only_v11"
HEAD_NAMES = ("math_head.weight", "math_head.bias")


def run() -> int:
    """정상 runtime으로 원 encoder를 재실행하고 저장 head·정확한 gradient 식·실제 순위를 검사한다."""
    import torch
    from run_hwr_affine_distillation_experiment_v1 import _load_teacher
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    destination = OUTPUT / "independent_head_verification.json"
    snapshot = OUTPUT / "head_verifier_snapshot.py"
    if destination.exists() or snapshot.exists():
        raise FileExistsError("refusing to replace head verification")
    read = lambda name: json.loads((OUTPUT / name).read_text(encoding="utf-8"))
    result, plan, verified = read("active_trust_result.json"), read("frozen_plan.json"), read("independent_verification.json")
    assert verified["result_sha256"] == _sha(OUTPUT / "active_trust_result.json")
    assert _sha(OUTPUT / verified["verifier_snapshot"]) == verified["verifier_sha256"]
    assert plan["head_only_revision"] == "v11" and plan["head_parameter_names"] == list(HEAD_NAMES)
    assert _sha(DEFAULT_CHECKPOINT) == plan["canonical_checkpoint_sha256"]
    assert _sha(CANDIDATE / "directional_guard.pt") == plan["parent_checkpoint_sha256"]
    parent, labels, _ = _load_teacher(CANDIDATE / "directional_guard.pt", torch.device("cpu")); parent.eval()
    final, final_labels, _ = _load_teacher(OUTPUT / result["checkpoint_file"], torch.device("cpu")); final.eval()
    assert labels == final_labels and isinstance(parent.math_head, torch.nn.Linear)
    named, final_named = dict(parent.named_parameters()), dict(final.named_parameters())
    assert len(named) == len(final_named) == 57
    assert all(torch.equal(p.detach().view(torch.int32), final_named[n].detach().view(torch.int32))
               for n,p in named.items() if n not in HEAD_NAMES)
    x=np.load(REFERENCE / "joint_features.npy",allow_pickle=False); y=np.load(REFERENCE / "joint_labels.npy",allow_pickle=False)
    assert _sha(REFERENCE / "joint_features.npy") == plan["reference_hashes"]["joint_features"]
    assert _sha(REFERENCE / "joint_labels.npy") == plan["reference_hashes"]["joint_labels"]
    embedding_parts=[]
    with torch.inference_mode():
        for start in range(0,len(x),32):
            embedding_parts.append(parent.encode(torch.from_numpy(x[start:start+32].copy())).cpu().numpy())
    embedding=np.concatenate(embedding_parts); assert embedding.shape == (2048,128)
    full=result["full_logits_trace"]; full_items=[full["initial"],*full["trials"],full["final"]]
    trace=result["head_only_trace"]; heads=trace["head_parameter_trace"]
    assert len(heads)==len(full_items)==result["actual_model_checks"]+2
    assert trace["evidence_capture_extra_model_or_gradient_calls"]==0
    arrays=[]
    for index,(item,score) in enumerate(zip(heads,full_items)):
        assert item["index"]==index and item["all_nonhead_int32_bit_equal"] and item["frozen_nonhead_tensors"]==55
        loaded=[]
        for name,shape in (("weight",(372,128)),("bias",(372,))):
            assert item[f"{name}_file"]==f"head_{name}_{index:03d}.npy"
            assert _sha(OUTPUT / item[f"{name}_file"])==item[f"{name}_sha256"]
            array=np.load(OUTPUT / item[f"{name}_file"],allow_pickle=False)
            assert array.shape==shape and array.dtype==np.float32 and np.isfinite(array).all()
            loaded.append(array)
        weight,bias=loaded; arrays.append((weight,bias)); parts=[]
        with torch.inference_mode():
            for start in range(0,len(x),32):
                values=torch.nn.functional.linear(torch.from_numpy(embedding[start:start+32]),torch.from_numpy(weight),torch.from_numpy(bias))
                parts.append(values.numpy())
        assert _sha(OUTPUT / score["file"])==score["sha256"]
        assert np.array_equal(np.concatenate(parts),np.load(OUTPUT / score["file"],allow_pickle=False))
    assert all(np.array_equal(a.view(np.int32),named[n].detach().numpy().view(np.int32)) for a,n in zip(arrays[0],HEAD_NAMES))
    assert all(np.array_equal(a.view(np.int32),final_named[n].detach().numpy().view(np.int32)) for a,n in zip(arrays[-1],HEAD_NAMES))
    positions={(item["round"],item["trial"]):idx for idx,item in enumerate(full_items) if "round" in item}
    singles={}; before_head=0; cursor=0; certificates=[]
    for history in result["history"]:
        rows=np.array([i["row"] for i in history["selected"]]); rivals=np.array([i["rival"] for i in history["selected"]])
        for row in rows:
            if int(row) not in singles:
                # gradient 회차와 같은 enable-grad encoder 경로로 embedding만 재계산한다.
                singles[int(row)]=parent.encode(torch.from_numpy(x[row:row+1].copy())).detach().numpy()[0].copy()
        e=np.array([singles[int(row)] for row in rows]); e64=e.astype(np.float64)
        class_diff=np.zeros((len(rows),372),dtype=np.float64)
        class_diff[np.arange(len(rows)),y[rows]]=1.; class_diff[np.arange(len(rows)),rivals]-=1.
        lengths=np.sqrt(2*(np.square(e64).sum(axis=1)+1.))
        gram=(class_diff@class_diff.T)*(e64@e64.T+1.)/np.outer(lengths,lengths)
        stored=np.load(OUTPUT / f'linearized_round_{history["round"]:03d}.npz',allow_pickle=False)
        gram_error=float(np.max(np.abs(gram-stored["gram"]))); assert gram_error < 1e-10
        logged_lengths=np.array(history["gradient_norms_fp64"])
        assert np.allclose(lengths,logged_lengths,atol=1e-10,rtol=1e-11)
        weight,bias=arrays[before_head]; margins=[]
        for row,rv,features in zip(rows,rivals,e):
            values=torch.nn.functional.linear(torch.from_numpy(features[None]),torch.from_numpy(weight),torch.from_numpy(bias)).numpy()[0]
            margins.append(float(values[y[row]]-values[rv]))
        floors=np.array([i["floor"] for i in history["selected"]])
        expected_rhs=(floors+plan["proposal_slack"]-np.array(margins))/lengths
        rhs_error=float(np.max(np.abs(expected_rhs-stored["rhs"]))); assert rhs_error < 1e-10
        for i,features in enumerate(e64):
            record=trace["gradient_records"][cursor]; assert record["index"]==cursor and record["unused_nonhead_tensors"]==55
            norms=record["head_norms_fp64"]
            assert set(norms)==set(HEAD_NAMES)
            assert abs(norms[HEAD_NAMES[0]]-np.sqrt(2*np.square(features).sum())) < 1e-10
            assert abs(norms[HEAD_NAMES[1]]-np.sqrt(2.)) < 1e-12
            cursor+=1
        certificates.append(dict(round=history["round"],analytic_head_gram_max_delta=gram_error,
            analytic_head_rhs_max_delta=rhs_error,directions=len(rows)))
        for trial,index in enumerate(history["trials"]):
            if index["accepted"]: before_head=positions[(history["round"],trial)]
    assert cursor==len(trace["gradient_records"])==result["gradient_directions_used"]
    predecessor=json.loads((ROOT / "artifacts/hwr_active_trust_retention_20261005_rival_guard_v7/active_trust_result.json").read_text(encoding="utf-8"))
    assert _sha(ROOT / "artifacts/hwr_active_trust_retention_20261005_rival_guard_v7/active_trust_result.json")==plan["predecessor_result_sha256"]
    assert result["history"][0]["selected"]==predecessor["history"][0]["selected"]
    report=dict(schema="aiflow-head-only-verification/v11",status="reproduced",verifier_sha256=_sha(Path(__file__)),
        verifier_snapshot=snapshot.name,result_sha256=_sha(OUTPUT / "active_trust_result.json"),
        main_verification_sha256=_sha(OUTPUT / "independent_verification.json"),frozen_nonhead_final_tensors_bit_exact=55,
        every_full_score_rebuilt_bit_exact_from_original_encoder_and_archived_head=True,full_scores_rebuilt=len(heads),
        head_gradient_directions_rebuilt=cursor,analytic_head_certificates=certificates,
        initial_selected_conditions_exact_v7=True,encoder_batch32_verification_calls=len(embedding_parts),
        encoder_batch1_verification_calls=len(singles),extra_parameter_updates_in_verification=0,
        fixed_rival_prediction_error_max=max(i["fixed_rival_prediction_error_p50_p95_max"][-1] for i in verified["fixed_rival_microscope"]),
        strict_train_gate_pass=verified["strict_train_gate_pass"],human_boundary_labels=0,held_inputs_forwarded=0,
        crohme_rows=0,product_adopted=False,independent_accuracy_improvement_proven=False)
    shutil.copyfile(Path(__file__),snapshot); assert _sha(snapshot)==report["verifier_sha256"]
    _write(destination,report)
    print(json.dumps(dict(status="reproduced",frozen_tensors=55,full_scores=len(heads),directions=cursor,
        max_fixed_prediction_error=report["fixed_rival_prediction_error_max"],strict_train_gate_pass=report["strict_train_gate_pass"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
