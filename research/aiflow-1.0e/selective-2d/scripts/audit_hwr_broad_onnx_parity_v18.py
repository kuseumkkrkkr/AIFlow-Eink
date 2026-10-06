"""선택된 광범위 HWR 연구 모델의 FP32 ONNX 계약과 실제 획 batch parity를 검사한다."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import run_hwr_broad_diversity_v16 as broad
from export_mobile_hwr_onnx_v1 import MobileHwrWrapperV1

OUT=broad.previous.ROOT / "artifacts/hwr_broad_fp32_onnx_20261005_v18"
CHECKPOINT=broad.OUT / "broad_augmented/research.pt"
MODEL_SHA="5911ddef5740f8c9dab909d602591db11ca415e993e472991aa89478f239d1bf"
ONNX_SHA="8ba4792fa1d7fe44f261b2e9741c85b08aabdde5699712d44c8aa12270459cfa"
BATCHES=(1,2,3,8,16,32)
TOLERANCE=1e-4


def describe(value) -> list:
    """ONNX ValueInfo의 동적 이름과 정적 차원을 그대로 반환한다."""
    return [dimension.dim_param or dimension.dim_value for dimension in value.type.tensor_type.shape.dim]


def main() -> int:
    """실제 클래스·점 획·다획과 유효 edge 입력에서 embedding/logits/순위를 직접 비교한다."""
    if (OUT / "parity_result.json").exists() or (OUT / "frozen_parity_plan.json").exists():
        raise FileExistsError("refusing parity audit overwrite")
    if broad.previous.aug.base._guard_commit("broad_onnx_parity") is None:return 78
    if broad.previous._sha(CHECKPOINT)!=MODEL_SHA or broad.previous._sha(OUT / "hwr_fp32.onnx")!=ONNX_SHA:
        raise ValueError("selected checkpoint or export changed")
    _,manifest,a=broad.load()
    x,y=a["population_features"],a["population_labels"]
    selected=[int(np.flatnonzero(y==c)[0]) for c in np.unique(y)]
    multistroke=[];point=[]
    for i,row in enumerate(x):
        signature=broad.previous.aug.base._signature(row)
        if len(signature)>1 and len(multistroke)<16:multistroke.append(i)
        if any(signature) and len(point)<16:point.append(i)
        if len(multistroke)==16 and len(point)==16:break
    ids=np.array(list(dict.fromkeys((*selected,*multistroke,*point))),dtype=np.int64)
    real=np.array(x[ids],copy=True)
    special=[]
    for coordinate in ((.5,.5),(0.,0.),(1.,1.)):
        row=np.zeros((128,5),dtype=np.float32);row[:,:2]=coordinate
        row[1:,2]=1/127;row[0,3]=1;row[:,4]=1;special.append(row)
    # 네 점 획은 원본 객체 보존과 별개인 수치 계약용 fixture이며 정답 라벨은 없다.
    row=np.zeros((128,5),dtype=np.float32);row[1:,2]=1/127;row[:,4]=1
    for k,coordinate in enumerate(((.2,.2),(.8,.2),(.2,.8),(.8,.8))):
        row[k*32:(k+1)*32,:2]=coordinate;row[k*32,3]=1
    special.append(row)
    cases=np.concatenate((real,np.stack(special)))
    if not np.isfinite(cases).all() or any(not broad.previous.aug.base._geometry(row,row)["valid"] for row in cases):
        raise ValueError("parity ink fixtures are not valid")
    plan=dict(schema="aiflow-broad-fp32-onnx-parity-plan/v18",script_sha256=broad.previous._sha(Path(__file__)),
        checkpoint_sha256=MODEL_SHA,onnx_sha256=ONNX_SHA,parent_manifest_sha256=broad.previous._sha(broad.OUT / "prepared_manifest.json"),
        real_population_indices=ids.tolist(),real_supported_classes=len(np.unique(y[ids])),technical_fixtures=len(special),
        batch_sizes=BATCHES,threshold=TOLERANCE,metrics="absolute embedding/logit errors and stable Top-1/Top-5 ordering",
        parity_not_accuracy_evaluation=True,model_updates=0,official_test_rows_read=0,crohme_rows=0,
        independent_writer_device_acceptance=False,android_device_measured=False,product_adopted=False)
    broad.previous._write(OUT / "frozen_parity_plan.json",plan)
    np.save(OUT / "parity_inputs.npy",cases,allow_pickle=False)
    import torch,onnx,onnxruntime as ort
    from evaluate_48hz_prefix_v1 import _load_model
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    payload=onnx.load(str(OUT / "hwr_fp32.onnx"));onnx.checker.check_model(payload)
    input_info=payload.graph.input[0]
    outputs={value.name:describe(value) for value in payload.graph.output}
    if describe(input_info)[1:]!=[128,5] or not isinstance(describe(input_info)[0],str):
        raise ValueError("input static/dynamic contract differs")
    if outputs!={"embedding":["batch",128],"logits":["batch",372]} or input_info.type.tensor_type.elem_type!=onnx.TensorProto.FLOAT:
        raise ValueError("output or FP32 contract differs")
    if any(value.type.tensor_type.elem_type!=onnx.TensorProto.FLOAT for value in payload.graph.output):
        raise ValueError("output dtype differs")
    if any(value.data_type in (onnx.TensorProto.FLOAT16,onnx.TensorProto.BFLOAT16,onnx.TensorProto.DOUBLE) for value in payload.graph.initializer):
        raise ValueError("non-FP32 floating initializer")
    model,labels,_=_load_model(CHECKPOINT,torch.device("cpu"));wrapper=MobileHwrWrapperV1(model).eval()
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(OUT / "hwr_fp32.onnx"),sess_options=options,providers=["CPUExecutionProvider"])
    reports=[];scores_by_batch={}
    with torch.inference_mode():
        for batch in BATCHES:
            expected_e=[];expected_z=[];actual_e=[];actual_z=[]
            for start in range(0,len(cases),batch):
                values=np.array(cases[start:start+batch],dtype=np.float32,copy=True)
                e,z=wrapper(torch.from_numpy(values));oe,oz=session.run(["embedding","logits"],{"points":values})
                if oe.shape!=(len(values),128) or oz.shape!=(len(values),372) or oe.dtype!=np.float32 or oz.dtype!=np.float32:
                    raise ValueError("actual dynamic output shape/dtype differs")
                expected_e.append(e.numpy());expected_z.append(z.numpy());actual_e.append(oe);actual_z.append(oz)
            te,tz,oe,oz=(np.concatenate(items) for items in (expected_e,expected_z,actual_e,actual_z))
            if not np.isfinite(oe).all() or not np.isfinite(oz).all():raise ValueError("nonfinite ONNX output")
            old_order=np.argsort(-tz,axis=1,kind="stable")[:,:5];new_order=np.argsort(-oz,axis=1,kind="stable")[:,:5]
            report=dict(batch=batch,cases=len(cases),embedding_max_abs_error=float(np.abs(te-oe).max()),
                logits_max_abs_error=float(np.abs(tz-oz).max()),top1_mismatches=int((old_order[:,0]!=new_order[:,0]).sum()),
                ordered_top5_mismatches=int((old_order!=new_order).any(1).sum()))
            report["passed"]=report["embedding_max_abs_error"]<=TOLERANCE and report["logits_max_abs_error"]<=TOLERANCE and report["top1_mismatches"]==0 and report["ordered_top5_mismatches"]==0
            reports.append(report);scores_by_batch[str(batch)]=dict(torch_embedding=te,torch_logits=tz,onnx_embedding=oe,onnx_logits=oz)
            for name,array in scores_by_batch[str(batch)].items():np.save(OUT / f"batch{batch}_{name}.npy",array,allow_pickle=False)
            print(json.dumps(dict(event="actual_ink_onnx_parity",**report)),flush=True)
    rejected=[]
    for name,values in (("float64",cases[:1].astype(np.float64)),("points127",cases[:1,:127]),("channels4",cases[:1,:,:4])):
        try:session.run(None,{"points":values})
        except Exception as error:rejected.append(dict(input=name,error_type=type(error).__name__))
        else:raise ValueError("ONNX accepted malformed input contract")
    unchanged=broad.previous._sha(CHECKPOINT)==MODEL_SHA
    passed=all(report["passed"] for report in reports) and unchanged
    result=dict(schema="aiflow-broad-fp32-onnx-parity-result/v18",status="pass" if passed else "failed_parity_do_not_promote",
        plan_sha256=broad.previous._sha(OUT / "frozen_parity_plan.json"),onnx_bytes=(OUT / "hwr_fp32.onnx").stat().st_size,
        versions=dict(torch=torch.__version__,onnx=onnx.__version__,onnxruntime=ort.__version__),input=describe(input_info),outputs=outputs,
        batches=reports,malformed_inputs_rejected=rejected,checkpoint_unchanged=unchanged,
        windows_cpu_only=True,android_latency_memory_or_crash_gates_measured=False,onnx_runtime_mobile_operator_build_tested=False,
        product_adopted=False,model_updates=0,official_test_rows_read=0,crohme_rows=0)
    broad.previous._write(OUT / "parity_result.json",result)
    return 0 if passed else 1


if __name__=="__main__":raise SystemExit(main())
