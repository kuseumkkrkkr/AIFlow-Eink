# AIFlow Math Ink 1.0 dataset re-audit - 2026-08-10

## Outcome and adoption amendment

All working files, source archives, Hugging Face downloads, caches, and audit dependencies were kept on `D:`. The four source datasets were downloaded and opened successfully. The initial fail-closed audit recorded the limitations below; later on 2026-08-10 the project owner explicitly adopted all four for restricted roles.

| Dataset | Integrity | Retained limitation | Current decision |
|---|---|---|---|
| ISGL | Outer ZIP and online RAR extraction pass | No per-point time; inherited derivative omits 571 character rows and has no evaluation split | Approve CC BY 4.0 online English letter/digit/word training only |
| UCI Character Trajectories | Official ZIP CRC and all 2,858 arrays pass | One writer, one pen-down, transformed signals | Approve as one-writer lowercase single-stroke training only |
| HWRT | Zenodo MD5, TAR, 168,233 JSON rows, and symbol counts pass | Source IDs do not identify true writers; ODbL; isolated symbols only | Approve 168,027-row filtered derivative for math-symbol training only |
| BDSHWA | Outer and both nested ZIP CRCs pass; stratified trajectory sample passes | Metadata/biometric risks and no mathematical supervision | Approve CC BY 4.0 raw online trajectories for future general English/Bengali HWR; exclude metadata/biometrics |

Approval admits data to the 1.0 training pool; it does not retroactively change the current 0.9 checkpoint.

## HWRT filter result

- Builder: `scripts/build_hwrt_curated.py`; manifest: `datasets/10_approved_external/hwrt/derived/manifest.json`.
- Accepted 168,027 / 168,233 samples.
- Removed 193 samples containing 1,209 cross-point timestamp reversals and 13 normalized exact duplicates.
- Removed absolute timestamp origins, raw user IDs, and full browser user agents.
- Available source-group overlap across train/validation/test is zero.
- The HASYv2 paper documents that ID `16925` aggregates many Detexify contributors. Its 153,660 retained rows are training-only; true writer-disjoint evaluation is impossible to prove from HWRT metadata.
- HWRT is therefore ineligible for model selection and final evaluation even after training admission.

## Hugging Face CLI repeat search

Environment:

```powershell
$env:HF_HOME='D:\AIFlow-Workspace\Projects\Aiflow\aiflow-math-ink-1.0\research\hf_home'
hf version
hf datasets list --search 'handwriting_strokes' --limit 20 --format quiet
hf datasets list --search 'MathWriting' --limit 20 --format quiet
hf datasets list --search 'CROHME' --limit 20 --format quiet
```

- CLI version: 1.24.0.
- Exact searches for `online handwriting trajectory`, `digital ink handwriting`, `handwriting strokes coordinates`, and `InkML handwriting math` returned no repositories.
- The three downloaded coordinate repositories remain rejected for missing licence/provenance plus split leakage.
- MathWriting and CROHME remain noncommercial/research-only; mirrors cannot broaden upstream rights.

## D-only control and reproducibility

- Project cache and audit dependencies are under `research/` on D:.
- Source and derivative binaries remain Git-ignored; hashes, counts, policies, and decisions are versioned.
- `scripts/verify_datasets.ps1` verifies expected artifacts and containers. `-Full` additionally hashes large files and runs the full HWRT derivative validator.
