# Official PyTorch migration preparation

User direction: remove the temporary substitute from active use, preserve its evidence, use official PyTorch, and continue tiny-LM research. Canonical defaults and retired final tests remain untouched.

Completed: exact V63 source archived with SHA-256 manifest before removal from active scripts; no active Python module imports NumpyLM or the custom checkpoint reader. Active entry points are `scripts/official_mini_lm_v64.py` and `scripts/verify_official_mini_lm_v64.py`. The runtime imports official torch directly and uses weights-only checkpoint loading, the original pre-norm Transformer architecture, finite FP32 weights, and explicit input validation. There is no fallback selector.

Verification prepared: all 579 same-input requests against complete archived reconstruction logits and probabilities; exact LM Top-1 and fused/strict/capped token decisions; formula decision equality follows from identical per-formula tokens; padded batching against singleton inference; six invalid request contracts; checkpoint and canonical hashes; LM-only latency and whole verifier peak RSS. Logit/log-probability tolerance is 1e-4 and probability tolerance 1e-5, with exact decisions required. These checks concern migration equivalence, not new generalization evidence. No scientific hypothesis was retuned or rejected experiment repeated.

Verification actually completed so far: syntax compilation; archived byte equality; complete historical migration-reference regeneration equality; source search finds no substitute import in active scripts. The canonical SHA-256 was directly verified as `04f8608aebcf6c02d45ad6f5735229b9eaa2c4b4e1be0db4793d02273ef2d00e` during V63.

Runtime verification **passed** in actual Codex using `/workspace/shared/aiflow-runtime/bin/python`, torch 2.14.1+cpu, NumPy 2.3.5 and SciPy 1.17.0. CPU multiplication, autograd, NumPy bridge and pip check passed. The supplied verifier could execute its checks but could not save into the read-only runtime mount. A copy preserving every scientific/runtime assertion changes only runtime-root/output-root handling and saves `artifacts/hwr_official_torch_migration_20261007_v64/VERIFY_CODEX.json`. The runtime package tree is not committed. Its install manifest and original verification script are copied as small provenance records.

All 579 requests passed migration checks. Maximum complete-logit error against V63 was 3.0994415283203125e-6; complete probability error 8.344650268554688e-7; padded batch versus singleton maximum logit error 4.76837158203125e-6. All final token and formula decisions match. Six malformed input contracts were rejected. Canonical weights remain unchanged. Fixed-cap diagnostic accuracy remains 82/149, six formula recoveries, zero formula/token regressions, 481/579 token hits. This is same-input parity, not independent handwriting acceptance.

With one CPU thread and two warmups per request, median official LM forward plus validation was 0.215006 ms; whole verifier peak RSS 278,836 KiB. HWR/request packing is excluded. These measurements do not establish target-device performance.

Current entry point:

```sh
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 /workspace/shared/aiflow-runtime/bin/python research/aiflow-1.0e/selective-2d/scripts/verify_official_mini_lm_v64.py
```

The verifier refuses to overwrite a completed V64 output directory. The historical V63 reference can be regenerated with the explicitly archived `artifacts/hwr_numpy_reconstruction_20261007_v63/archived_source/generate_migration_reference.py`. Generated NPZ derivatives remain local; the original checkpoint is preserved in repository history. Source and text diagnostics are the durable checkpoint; they retain the hashes and instructions needed to reproduce the arrays.

After parity passes, continue synthetic literal-digit controls and provenance auditing for writer/formula-disjoint evaluation using official torch. Keep the consumed oracle-group diagnostics exploratory and the product default frozen.
