# Dataset normalization audit (2026-08-11)

## Conclusion

The approved datasets are **not normalized to one common stored coordinate system**. This is not automatically data corruption: AIFlow currently preserves source geometry and applies glyph-local bbox/letterbox normalization in the training path. However, a 1.0 training run must not concatenate the stored coordinates directly.

| Dataset | Stored spatial state | Time state | Verdict |
|---|---|---|---|
| Project-owned ink | Browser canvas coordinates | Formula-relative `t_ms`, starts at 0 | Partial |
| UJI Pen v2 derived | Resampled/transformed coordinate frame; baseline context is unit-scaled | Missing | Spatial derivative only |
| ISGL migrated | Original 1366x768 canvas coordinates retained | Point time missing | Schema conversion only |
| UCI Character Trajectories | Publisher differentiated/smoothed/normalized X/Y/pressure | Implicit sequence index | Pre-normalized; raw-pair verification impossible |
| HWRT curated | Source X/Y retained | Shifted to sample-relative zero; reversals filtered | Temporal only |
| BDSHWA aligned | Device raw units mapped to canvas X/Y | Epoch timestamps retained | Device-to-canvas only |
| CROHME 2019 | Raw InkML at rest | Source-dependent | Evaluation loader applies median-height 32 normalization |
| HF re-audit parquet | Not admitted to training | Unverified | Exclude until license/schema audit |

## Visual verification

Each PNG was opened and inspected after generation. The top/native rows retain the dataset coordinate axes; the following row shows the same strokes mapped to a unit bbox only as a diagnostic.

- `01_project_owned.png`: formulas occupy differing canvas locations and extents; therefore spatial normalization is not stored.
- `02_uji_derived.png`: glyphs share a roughly 0..130 derived frame, but some strokes extend below/left of zero and no time exists. It is not equivalent to unit-bbox input.
- `03_isgl_migrated.png`: axes remain in the hundreds and match the declared 1366x768 canvas. Migration did not spatially normalize.
- `04_hwrt_curated.png`: samples retain widely differing locations/scales. Timestamp curation must not be described as spatial normalization.
- `05_uci_trajectories.png`: values cluster around approximately -1..1 and are smooth, consistent with the publisher transformation. There is no raw counterpart in the package, so the transformation cannot be independently reconstructed.
- `06_bdshwa_aligned.png`: samples remain positioned in canvas space; sentence, shape, and wave tasks have very different extents. `canvas_x/y` is alignment, not glyph normalization.

## 1.0 rule

Use source-specific adapters, then one shared model-input transform: preserve ordered raw strokes, translate by glyph bbox, apply aspect-preserving scale/letterbox, derive relative time/delta features, and record the transform metadata. Never overwrite source coordinates. External corpora remain box-local representation training only; project-owned formula ink owns grouping and final-decision evaluation.

## Reproduction

```powershell
python scripts/audit_dataset_normalization.py
```

The gallery is a deterministic first-12 sample inspection per readable dataset. HWRT admission statistics and timestamp reversal checks remain governed by its existing full curation manifest; BDSHWA is archive-sampled because the nested archive is large. CROHME and rejected HF sources are verdict-only here and must not be presented as visually audited galleries.
