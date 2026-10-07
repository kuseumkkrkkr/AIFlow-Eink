# Checkpoint continuity, reviewed revision

Official runtime: `/workspace/shared/aiflow-runtime/bin/python`. Run the isolated tests with:

```sh
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 /workspace/shared/aiflow-runtime/bin/python research/aiflow-1.0e/selective-2d/scripts/test_research_continuity_v1.py
```

The test refuses existing output `artifacts/hwr_research_continuity_20261008_v66_r2`; use a new test output constant to repeat the suite. The first V66 results remain historical evidence, superseded by these expanded checks. Schema v2 deliberately rejects older incompatible run manifests.

All expanded checks passed in actual Codex. A disposable official-PyTorch child exits after step 4 and resumes to bitwise-identical model, AdamW optimizer, scheduler, Python/NumPy/Torch RNG and sampler state versus uninterrupted training. Faults after uncommitted step 6, payload flush, manifest flush, before pointer replacement and after pointer replacement recover from the last atomic pointer and reproduce the same final state. No actual research owner was terminated.

A real result-only commit in an isolated Git fixture resumes with unchanged executable/data/config/checkpoint hashes while preserving the original source commit as provenance. Genuine executable, data and config changes still fail. Fresh or stale store instances cannot mutate completed/advanced authoritative state. Metadata stays 1548–1549 bytes across 40 small checkpoints, with a flat last verified remote record rather than recursive history.

The official LM audit resumes unique committed output prefixes. The test crashes exactly after the final checkpoint and before publishing the derived output; restart restores that output without rerunning inference. Missing and truncated derived final outputs and recovery summaries are likewise rebuilt from the verified checkpoint. Empty input is rejected. Completed stages skip safely. A second live process cannot acquire the same flock. Corrupt/truncated/missing payloads, manifests and pointers fail clearly; a stray partial cannot replace the previous valid checkpoint.

Use `run_official_mini_lm_resumable_v1.py --run-dir ... --run-id ...` for official inference, with a stable run/round identity. Training clients initialize/load under `store.owner()`, restore model/optimizer/scheduler/RNG, then save only immediately after optimizer step and cleared gradients with accumulation position zero and explicit sampler order/position. Config and all executable/data/checkpoint asset hashes are frozen. The actual dedicated tiny-model training runner uses this same boundary protocol. Do not change executable inputs to repair a interrupted scientific run silently; assign a new explicit run if compatibility changes.

Task state distinguishes pending local work, validated checkpoint/stage completion and separately verified remote preservation. Publication callers must read the remote branch and pass matching commit plus checkpoint-manifest evidence; the library never invents remote verification or publishes automatically. At least the prior valid checkpoint remains present. Payloads flush/fsync before atomic rename, pointer promotion occurs only after full checkpoint validation, and partials never count as completion.

Authoritative recovery summaries are in the project. Optional `/workspace/shared/aiflow-resume` mirror errors are recorded and never stop useful work. Both that mount and this clone's `.git` are currently read-only to Codex; local HEAD remains `f4fe8cd`, while the verified research branch has published checkpoints. All unique local files remain intact. Safe local reconciliation requires a writable Git metadata view; no reset, discard, history rewrite or broad setting change was used.

Guarantee: supported Linux process restart on a surviving compatible filesystem, plus externally recoverable checkpoints only when their complete payloads and hashed inputs/code were actually preserved. A recovery summary alone is insufficient. A replacement machine needs an external execution owner and a compatible runtime. This does not provide autonomous machine replacement or 24-hour availability.
