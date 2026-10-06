"""봉인된 세 HWR의 소유 수식 입력 전이를 같은 oracle 그룹·전처리로 비교한다.

149식은 이미 소비됐고 추가 53식은 Codex 시각 검수다. 새 사람 acceptance가
아니며, 이 입력·라벨·오류를 학습에 사용하거나 제품 checkpoint를 교체하지 않는다.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch

import evaluate_oracle_group_decoder_ceiling_v1 as oracle
from evaluate_48hz_prefix_v1 import INPUT_MODE, _load_model
from evaluate_joint_hwr_grouping_v1 import _candidate_tensor
from train_character_classifier_v1 import apply_input_mode
import run_hwr_broad_diversity_v16 as broad

OUT = broad.previous.ROOT / "artifacts/hwr_owned_formula_transfer_20261005_v21"
DATA = Path(r"D:\AIFlow-Workspace\PrivateData\math-ink-data-collector\derived\fresh-context-acceptance-20260820-r2\frozen_acceptance\frozen_dataset")
BATCH = 128


def load_source():
    """봉인 SHA·정답 순서·exact cover·기존 149식 ID를 확인하고 96/53 집단을 나눈다."""
    info = json.loads((DATA / "dataset_info.json").read_text(encoding="utf-8"))
    assert info["schema"] == "aiflow-fresh-acceptance-frozen-dataset/v1" and info["merged_ownership_formulae"]==149
    for name,digest in info["files"].items():
        assert broad.previous._sha(DATA / "data" / name)==digest
    formulas = {str(r["sample_id"]):r for r in oracle._rows(DATA / "data/formulas_valid.jsonl")}
    annotations = [r for r in oracle._rows(DATA / "data/ownership_train.jsonl") if r.get("accepted")]
    legacy_path = DATA.parents[2] / "public-candidate-20260820-r4-replay-restored/data/ownership_train.jsonl"
    assert broad.previous._sha(legacy_path)=="5da7ac0c2c11bf7f2d98edde328dd9e8be9cdb5db398a85c5824e6f20b4bdafb"
    legacy = {str(r["sample_id"]):r for r in oracle._rows(legacy_path) if r.get("accepted")}
    old = json.loads((broad.previous.ROOT / "artifacts/oracle_group_decoder_ceiling_20260914/evaluation.json").read_text(encoding="utf-8"))
    truth = {r["sample_id"]:r["truth_tokens"] for r in old["records"]}
    assert len(legacy)==96 and len(annotations)==149 and len({r["sample_id"] for r in annotations})==149
    samples = [];rows = [];offset = 0
    for r in annotations:
        source = formulas[str(r["sample_id"])]
        assert source["quality_status"]=="valid" and "crohme" not in str(source.get("source_partition","")).lower()
        strokes = sorted(source["strokes"],key=lambda s:int(s["order"]))
        groups = [[int(i) for i in group] for group in r["groups"]]
        labels = [str(t) for t in r["labels"]]
        assert all(groups) and len(groups)==len(labels) and sorted(i for g in groups for i in g)==list(range(len(strokes)))
        assert labels==[str(c["token"]) for c in source["target_cells"]]==truth[str(r["sample_id"])]
        cohort = "legacy_96" if str(r["sample_id"]) in legacy else "codex_reviewed_53"
        if cohort=="legacy_96":
            assert r["groups"]==legacy[str(r["sample_id"])]["groups"] and r["labels"]==legacy[str(r["sample_id"])]["labels"]
        samples.append(oracle.Sample(str(r["sample_id"]),str(r["writer_id"]),strokes,tuple(),
            [{"source_indices":g} for g in groups],np.empty((0,0))))
        rows.append(dict(sample_id=str(r["sample_id"]),writer_id=str(r["writer_id"]),cohort=cohort,
            groups=groups,truth_tokens=labels,start=offset,stop=offset+len(groups),stroke_count=len(strokes)))
        offset += len(groups)
    assert offset==579 and {r["sample_id"] for r in rows}==set(truth)
    return info,samples,rows


def metrics(records, cohort="all"):
    """수식 token sequence 완전 정답과 개별 token 정답을 구분해서 집계한다."""
    rows = [r for r in records if cohort=="all" or r["cohort"]==cohort]
    return dict(formulas=len(rows),tokens=sum(len(r["truth_tokens"]) for r in rows),
        top1_token_hits=sum(sum(a==b for a,b in zip(r["top1_tokens"],r["truth_tokens"],strict=True)) for r in rows),
        top5_token_hits=sum(sum(t in k for t,k in zip(r["truth_tokens"],r["top5_tokens"],strict=True)) for r in rows),
        formula_top1_exact=sum(r["top1_tokens"]==r["truth_tokens"] for r in rows),
        formula_top5_complete=sum(all(t in k for t,k in zip(r["truth_tokens"],r["top5_tokens"],strict=True)) for r in rows),
        decoder_token_exact=sum(bool(r["decoder"].get("accepted")) and r["decoder"].get("tokens")==r["truth_tokens"] for r in rows))


def main():
    """원본·코드를 봉인하고 모델 변경 없이 동일 실제 입력의 후보·층별 수치를 저장한다."""
    if OUT.exists():
        raise FileExistsError("refusing transfer audit overwrite")
    info,samples,rows = load_source()
    models = dict(canonical=broad.previous.CHECKPOINT,v16=broad.OUT / "broad_augmented/research.pt",
        v20=broad.previous.ROOT / "artifacts/hwr_broad_teacher_correct_20261005_v20/research.pt")
    expected = dict(canonical="04f8608aebcf6c02d45ad6f5735229b9eaa2c4b4e1be0db4793d02273ef2d00e",
        v16="5911ddef5740f8c9dab909d602591db11ca415e993e472991aa89478f239d1bf",
        v20="8d5badec02f2bef7d6106a399194736fc65d246406666370656e2beceb65734f")
    assert all(broad.previous._sha(p)==expected[name] for name,p in models.items())
    inputs = apply_input_mode(np.stack([_candidate_tensor(sample,g["source_indices"])
        for sample in samples for g in sample.candidates]).astype(np.float32),INPUT_MODE)
    assert inputs.shape==(579,128,5) and np.isfinite(inputs).all()
    OUT.mkdir()
    files = ("audit_hwr_owned_formula_transfer_v21.py","evaluate_oracle_group_decoder_ceiling_v1.py",
        "evaluate_joint_hwr_grouping_v1.py","evaluate_48hz_prefix_v1.py","build_normalized_ink_v1.py",
        "character_tensor_v1.py","train_character_classifier_v1.py","selective_decoder_v1.py","formula_layout_v1.py")
    broad.previous._write(OUT / "frozen_plan.json",dict(schema="aiflow-owned-transfer-plan/v21",dataset_root=str(DATA),
        source_sha256=info["files"],dataset_info_sha256=broad.previous._sha(DATA / "dataset_info.json"),
        code_sha256={f:broad.previous._sha(Path(__file__).parent/f) for f in files},model_sha256=expected,
        cohorts=["all","legacy_96","codex_reviewed_53"],already_consumed=True,fresh_human_acceptance=False,
        groups="fixed original oracle groups; grouping accuracy not evaluated",batch_size=BATCH,threads=1,
        input_mode=INPUT_MODE,training_steps=0,threshold_changes=0,crohme_rows=0,product_adopted=False))
    np.save(OUT / "inputs.npy",inputs,allow_pickle=False)
    broad.previous._write(OUT / "input_rows.json",rows)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    summaries = {};all_records = {}
    for name,path in models.items():
        model,vocab,_ = _load_model(path,torch.device("cpu"))
        original_weights = {key:value.detach().clone() for key,value in model.state_dict().items()}
        if name=="canonical":
            original_vocab = vocab
        assert vocab==original_vocab
        outputs = [];embeddings = [];layers = [];handles = []
        def hook(stage):
            """각 인코더의 실제 batch 출력 유한성과 평균/표준편차를 기록한다."""
            def capture(module,args,value):
                assert torch.isfinite(value).all()
                layers.append(dict(stage=stage,shape=list(value.shape),mean=float(value.mean()),std=float(value.std())))
            return capture
        for stage,module in model.named_modules():
            if isinstance(module,torch.nn.TransformerEncoderLayer):
                handles.append(module.register_forward_hook(hook(stage)))
        with torch.inference_mode():
            for start in range(0,len(inputs),BATCH):
                embedding = model.encode(torch.from_numpy(inputs[start:start+BATCH].copy()))
                embeddings.append(embedding.numpy());outputs.append(model.math_head(embedding).numpy())
        for handle in handles:
            handle.remove()
        logits = np.concatenate(outputs);vectors = np.concatenate(embeddings)
        assert np.isfinite(logits).all() and logits.shape==(579,372) and vectors.shape==(579,128)
        probabilities = torch.from_numpy(logits).softmax(1).numpy()
        order = np.argsort(-probabilities,axis=1,kind="stable")[:,:5]
        records = []
        for sample,r in zip(samples,rows,strict=True):
            symbols = []
            for group,indices,prob in zip(r["groups"],order[r["start"]:r["stop"]],probabilities[r["start"]:r["stop"]],strict=True):
                symbols.append(dict(stroke_indices=group,hwr_topk=[vocab[int(i)] for i in indices],
                    hwr_topk_probabilities=[float(prob[i]) for i in indices],geometry=oracle._geometry(sample.strokes,group)))
            decoded = oracle.decode_selective_partition(sample.sample_id,r["groups"],symbols,stroke_count=len(sample.strokes))
            assert not decoded.get("accepted") or all(t in s["hwr_topk"] for t,s in zip(decoded["tokens"],symbols,strict=True))
            records.append(dict(**r,top1_tokens=[s["hwr_topk"][0] for s in symbols],
                top5_tokens=[s["hwr_topk"] for s in symbols],decoder=decoded))
        np.save(OUT / f"{name}_logits.npy",logits,allow_pickle=False)
        np.save(OUT / f"{name}_embeddings.npy",vectors,allow_pickle=False)
        broad.previous._write(OUT / f"{name}_records.json",records)
        broad.previous._write(OUT / f"{name}_layers.json",layers)
        summaries[name] = {c:metrics(records,c) for c in ("all","legacy_96","codex_reviewed_53")}
        all_records[name] = records
        assert all(torch.equal(value,model.state_dict()[key]) for key,value in original_weights.items())
        print(json.dumps(dict(event="owned_transfer_model_completed",model=name,metrics=summaries[name])),flush=True)
    paired = {}
    for name in ("v16","v20"):
        before = np.array([r["top1_tokens"]==r["truth_tokens"] for r in all_records["canonical"]])
        after = np.array([r["top1_tokens"]==r["truth_tokens"] for r in all_records[name]])
        paired[name] = dict(formula_wins=int((~before&after).sum()),formula_losses=int((before&~after).sum()))
    assert all(broad.previous._sha(p)==expected[name] for name,p in models.items())
    result = dict(schema="aiflow-owned-transfer-result/v21",status="consumed_oracle_group_diagnostic_not_product_accuracy",
        metrics=summaries,paired_formula_vs_canonical=paired,old_baseline_76_and_137_reproduced=
            summaries["canonical"]["all"]["formula_top1_exact"]==76 and summaries["canonical"]["all"]["formula_top5_complete"]==137,
        unsupported_truth_tokens=sorted({t for r in rows for t in r["truth_tokens"] if t not in vocab}),
        model_values_unchanged=True,training_steps=0,crohme_rows=0,new_human_acceptance=False,product_adopted=False,
        limit="Oracle grouping removes grouping failures. Additional 53 records have Codex, not human, visual review. All 149 cases were already consumed. Token sequence exact is not LaTeX Expression Rate.")
    broad.previous._write(OUT / "transfer_result.json",result)
    print(json.dumps(result),flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
