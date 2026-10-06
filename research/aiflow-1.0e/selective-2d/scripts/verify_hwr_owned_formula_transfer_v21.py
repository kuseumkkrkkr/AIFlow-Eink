"""V21 저장 logits와 원래 oracle 평가기를 독립 대조한다. 새 학습·승격은 없다."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch

import audit_hwr_owned_formula_transfer_v21 as audit
import evaluate_oracle_group_decoder_ceiling_v1 as legacy
from evaluate_48hz_prefix_v1 import _load_model


def main():
    """소스 SHA·579개 입력·모든 순위·decoder·4층의 유한성과 재로딩 출력을 검증한다."""
    out = audit.OUT
    if (out / "independent_verification.json").exists():
        raise FileExistsError("certificate exists")
    plan = json.loads((out / "frozen_plan.json").read_text(encoding="utf-8"))
    result = json.loads((out / "transfer_result.json").read_text(encoding="utf-8"))
    for name,digest in plan["code_sha256"].items():
        assert audit.broad.previous._sha(Path(__file__).parent/name)==digest
    for name,digest in plan["source_sha256"].items():
        assert audit.broad.previous._sha(audit.DATA / "data" / name)==digest
    x = np.load(out / "inputs.npy",allow_pickle=False)
    rows = json.loads((out / "input_rows.json").read_text(encoding="utf-8"))
    assert x.shape==(579,128,5) and len(rows)==149 and np.isfinite(x).all()
    _,samples,rebuilt_rows = audit.load_source()
    rebuilt = audit.apply_input_mode(np.stack([audit._candidate_tensor(s,g["source_indices"])
        for s in samples for g in s.candidates]).astype(np.float32),audit.INPUT_MODE)
    assert np.array_equal(x,rebuilt) and rows==rebuilt_rows
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    models = dict(canonical=audit.broad.previous.CHECKPOINT,v16=audit.broad.OUT / "broad_augmented/research.pt",
        v20=audit.broad.previous.ROOT / "artifacts/hwr_broad_teacher_correct_20261005_v20/research.pt")
    verified = {}
    for name,path in models.items():
        assert audit.broad.previous._sha(path)==plan["model_sha256"][name]
        model,vocab,_ = _load_model(path,torch.device("cpu"))
        outputs = [];embeddings = []
        with torch.inference_mode():
            for start in range(0,len(x),audit.BATCH):
                h = model.encode(torch.from_numpy(x[start:start+audit.BATCH].copy()))
                embeddings.append(h.numpy());outputs.append(model.math_head(h).numpy())
        assert np.array_equal(np.concatenate(outputs),np.load(out / f"{name}_logits.npy",allow_pickle=False))
        assert np.array_equal(np.concatenate(embeddings),np.load(out / f"{name}_embeddings.npy",allow_pickle=False))
        logits = np.load(out / f"{name}_logits.npy",allow_pickle=False)
        records = json.loads((out / f"{name}_records.json").read_text(encoding="utf-8"))
        ordering = np.argsort(-logits,axis=1,kind="stable")[:,:5]
        assert len(records)==149
        independent = {}
        for cohort in ("all","legacy_96","codex_reviewed_53"):
            selected = [r for r in records if cohort=="all" or r["cohort"]==cohort]
            token_hits = 0;five_hits = 0;formula_hits = 0;five_formulas = 0;decoded = 0
            for r in selected:
                top = [[vocab[int(i)] for i in indices] for indices in ordering[r["start"]:r["stop"]]]
                assert r["top5_tokens"]==top and r["top1_tokens"]==[k[0] for k in top]
                first = [t==k[0] for t,k in zip(r["truth_tokens"],top,strict=True)]
                fifth = [t in k for t,k in zip(r["truth_tokens"],top,strict=True)]
                token_hits += sum(first);five_hits += sum(fifth);formula_hits += all(first);five_formulas += all(fifth)
                decoded += bool(r["decoder"].get("accepted")) and r["decoder"].get("tokens")==r["truth_tokens"]
            independent[cohort] = dict(formulas=len(selected),tokens=sum(len(r["truth_tokens"]) for r in selected),
                top1_token_hits=token_hits,top5_token_hits=five_hits,formula_top1_exact=formula_hits,
                formula_top5_complete=five_formulas,decoder_token_exact=decoded)
        assert independent==result["metrics"][name]
        layers = json.loads((out / f"{name}_layers.json").read_text(encoding="utf-8"))
        assert len(layers)==20 and all(np.isfinite(r["mean"]) and np.isfinite(r["std"]) for r in layers)
        assert all(sum(r["stage"]==f"encoder.layers.{i}" for r in layers)==5 for i in range(4))
        target = out / f"legacy_evaluator_{name}.json"
        if target.exists():
            raise FileExistsError("legacy evaluator artifact exists")
        previous_args = sys.argv
        try:
            sys.argv = [legacy.__file__,"--dataset-root",str(audit.DATA),"--checkpoint",str(path),"--output",str(target)]
            legacy.main()
        finally:
            sys.argv = previous_args
        old = json.loads(target.read_text(encoding="utf-8"))
        for k,new in (("top1_token_exact","formula_top1_exact"),("top5_oracle","formula_top5_complete"),("decoder_token_exact","decoder_token_exact")):
            assert old[k]==independent["all"][new]
        mapped = {r["sample_id"]:r for r in old["records"]}
        assert all(mapped[r["sample_id"]]["top1_tokens"]==r["top1_tokens"] for r in records)
        verified[name] = independent
    certificate = dict(schema="aiflow-owned-transfer-verification/v21",status="pass",
        verifier_sha256=audit.broad.previous._sha(Path(__file__)),result_sha256=audit.broad.previous._sha(out / "transfer_result.json"),
        raw_source_files_and_all_input_tensors_rebuilt=True,checkpoint_reload_logits_and_embeddings_bit_exact=True,
        all_stable_logit_ranks_and_formula_counts_rebuilt=True,original_legacy_evaluator_matches_all_three_models=True,
        all_four_encoder_layers_finite_on_all_five_batches=True,cohort_metrics=verified,
        training_steps=0,crohme_rows=0,new_human_acceptance=False,product_adopted=False,
        artifacts_sha256={p.name:audit.broad.previous._sha(p) for p in out.iterdir() if p.is_file()},
        limit="Fixed ownership groups, consumed records, and partly Codex-reviewed annotations; no grouping/LaTeX ER or fresh acceptance claim.")
    audit.broad.previous._write(out / "independent_verification.json",certificate)
    print(json.dumps({k:v for k,v in certificate.items() if k not in ("cohort_metrics","artifacts_sha256")}),flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
