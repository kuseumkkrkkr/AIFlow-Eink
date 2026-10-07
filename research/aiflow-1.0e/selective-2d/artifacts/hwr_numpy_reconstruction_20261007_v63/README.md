# V63: new inference reconstruction

This is new work from preserved pre-V40 source and weights, not recovery of the lost V42-V62 artifacts. The canonical HWR weight hash was verified directly. No model was trained, no thresholds were selected, no retired final tests were read, and no product default changed.

The restored one-layer LM has 90,548 parameters and 362,192 FP32 tensor bytes. A narrow allowlisted reader rejects unknown pickle globals; NumPy/SciPy implement the preserved pre-norm Transformer. Full sequence and target-query-only computation agree within 1.91e-6 on all 579 oracle-group requests, with identical class Top-1 and fused decisions. Archived PyTorch reference comparisons cover only 87 selected-group Top-5 vectors (435 scores), maximum error 1.58e-5; complete live framework parity remains unavailable.

Fixed .5 fusion with the historical strict homograph lock gives 97/149 exact formulas, 23 recoveries and two regressions. The historical .20 visual-confidence cap gives 82/149, six recoveries and no formula or token regressions, 481/579 token hits versus 470/579 canonical hits. These reproduce the historical summary, but remain consumed oracle-group diagnostics; they prove neither independent acceptance nor raw-stroke accuracy nor literal handwritten-mistake preservation. The stored requests derive geometry from archived bounding boxes, without joining reused IDs to unrelated raw handwriting.

Single-pass LM-only median forward times were 0.182449 ms full and 0.126086 ms target-only on this Linux CPU. These exclude HWR and request parsing and are preliminary, with fixed ordering and no warm-up. Whole audit peak RSS was 67,772 KiB, including large trace parsing and Python/SciPy; this is not decoder-only memory or Android evidence.

Original V63 command (source now archived; this command is retired):

```sh
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python research/aiflow-1.0e/selective-2d/scripts/audit_reconstructed_mini_lm_v63.py
```

The runner refuses to overwrite its output directory. Use a new output path for reproduction by changing its OUTPUT constant. Dependencies are NumPy and SciPy; neither PyTorch nor downloaded model files are required. The original verified .pt and report are preserved in the earlier one-layer artifact directory. The generated NPZ is a lossless FP32 array export, not a trained replacement.

Next: run synthetic literal-digit counterfactual controls to quantify whether the fixed policy overwrites a visually preferred written digit with a context preference. Use this to develop explicit safety invariants, not to claim handwriting acceptance. Then audit available writer/formula-disjoint data provenance before any independent evaluation. Keep the consumed 149 diagnostic set out of selection.

## Retirement and preservation

The user requested the official PyTorch runtime on 2026-10-07. Exact substitute source bytes are preserved in `archived_source/` with an archive manifest. Both modules were removed from the active scripts directory after byte verification. No active code imports the substitute. The official replacement and verifier are `scripts/official_mini_lm_v64.py` and `scripts/verify_official_mini_lm_v64.py`; their runtime verification subsequently passed with the official shared runtime `/workspace/shared/aiflow-runtime/bin/python`. See the V64 report for complete same-input parity and actual Codex runtime checks.

Generated FP32 exports and migration reference arrays remain local; the original hash-verified checkpoint is already in repository history. The canceled NPZ upload was not retried. `migration_reference_logits.npz` was generated after archival, from all 579 saved requests using the archived V63 implementation, and is strictly same-input migration evidence. Its values are reproducible with `archived_source/generate_migration_reference.py`. No model labels or policies were selected from these outputs.
