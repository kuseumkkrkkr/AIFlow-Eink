# V65 synthetic literal-digit controls

New experiment using the verified official PyTorch runtime, after V64 migration. No real ink, consumed149, training, retired final tests or policy selection enters this experiment. Canonical assets and product defaults remain frozen.

Enumerated all 300 contexts `a op b = [MASK]`, with a,b in 0..9 and op in +, -, times. Each context has 90 ordered distinct literal/rival digit candidate pairs and three fixed visual probability gaps .10/.20/.30, totaling 81,000 decisions. Visual probabilities are `(1+gap)/2` and `(1-gap)/2`; they are controlled synthetic distributions, not measured or calibrated HWR. The visual Top-1 is the intended literal digit by construction, including mathematically incorrect writing. Neighbor symbols are assumed clean. Only the masked target receives LM output scores; no arithmetic answer is supplied to inference or decoding. Arithmetic result tags are computed solely for post-hoc characterization.

Fixed historical .5 fusion, strict lock and .20 cap overwrite 6,076/27,000 literal choices at gap .10 (22.50%) and 2,867/27,000 at gap .20 (10.62%). The above-cap .30 control changes none. Both valid and invalid arithmetic strings suffer synthetic transcription failures: at .10, 281 mathematically valid and 5,795 invalid literal cases change; at .20, 117 valid and 2,750 invalid cases change. These are exhaustive policy controls, not population error estimates or evidence of arithmetic solving. Most changed outputs do not match the arithmetic result. The conditional learned class preference itself can override synthetic correct visual choices.

Independent NumPy float64 recomputation from the saved official context log probabilities reproduces every one of the 8,943 changed cases in exact order. This recomputation checks decision arithmetic only; model inference remains official torch and no substitute model runtime is active.

With one CPU thread, the 300 single-pass model/log-softmax calls had median 0.402633 ms; no warmup was applied. Whole process peak RSS was 261,076 KiB. This excludes HWR/request construction and is Linux-host exploratory timing, not mobile acceptance.

Reproduce with:

```sh
OPENBLAS_NUM_THREADS=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 /workspace/shared/aiflow-runtime/bin/python research/aiflow-1.0e/selective-2d/scripts/audit_literal_digit_counterfactual_v65.py
```

The runner refuses existing output. All context vectors and paired changed cases are recorded. Interpret this evidence as a need for explicit literal-preservation contracts and independently transcribed ink, not a reason to tune on the reused 149. The historical V60 digit-family property is not re-presented here as a new hypothesis or gain. Next research should investigate safety for non-digit mathematical operators and training formulations that retain useful context without overwriting literal numeric/operator content.
