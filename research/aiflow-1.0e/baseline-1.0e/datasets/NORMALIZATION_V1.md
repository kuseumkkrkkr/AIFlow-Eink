# AIFlow Math Ink 1.0 canonical online-ink normalization

`scripts/build_normalized_ink_v1.py` creates local-only derivative files under
`datasets/normalized/v1/`. It never mutates raw archives or the inherited
source derivatives.

## Contract

- Preserve the source stroke sequence. Empty source strokes are explicitly
  recorded by source index; non-empty strokes retain their original index.
- Compute one sample bbox over X/Y, translate it to the origin, scale by the
  longer side, then centre-pad the shorter side in the inclusive `[0, 1]`
  square. The transform is written per record.
- Convert globally monotonic timestamps to relative duration `[0, 1]`. When a
  source clock resets only between otherwise monotonic strokes, retain each
  stroke's duration and stitch them without inventing pen-up gaps. Sources
  without per-point time use a clearly marked ordinal sequence `[0, 1]`; it is
  not a fabricated physical timestamp.
- Keep source files untouched. Canonical records contain a stable hashed
  record ID and source fingerprint, not participant paths, raw IDs, sessions,
  user agents, demographics, or BDSHWA biometric metadata.
- A source file with no pen-down points cannot be fabricated into an ink
  trajectory. It is retained in raw storage and written to a local hashed
  rejection artifact with the explicit reason instead of being silently lost.

## Run

```powershell
python scripts/build_normalized_ink_v1.py --deterministic-replay
```

After a completed build, `--deterministic-replay-only` verifies the existing
artifacts and rebuilds once into an automatically removed D: temporary folder
for a byte-identical replay check.

The BDSHWA raw nested ZIP is intentionally supplied as a local D: path. Use
`--bdshwa-nested <path>` if its temporary audit extraction has moved.

## Scope boundary

The canonical derivative is model input for the source-specific role documented
in `UTILIZED_DATASETS.md`. It does not turn isolated external trajectories into
formula grouping, ownership, relation, or final-decision labels. Project-owned
formula samples remain `unassigned_project_holdout` until a writer-holdout split
is generated.

## Character-classifier admission amendment (2026-08-12)

- A trajectory with fewer than two points is a missing character trajectory,
  not a shape to upsample. It is excluded from the canonical derivative with a
  hashed rejection record. The raw source stays unchanged.
- UCI removes only consecutive points whose X and Y are exactly equal, both
  before and after canonical coordinate rounding. The final timestamp of the
  duplicate run is retained, so the next delta-time interval preserves dwell.
  Non-consecutive return-to-point loops remain.
- The default classifier derivative contains project-owned, UJI, ISGL, UCI,
  and HWRT only. BDSHWA is excluded because it has whole-task prompts and
  stroke IDs but no character boundary or character-to-stroke ownership.
- Canonical files remain variable-length `[x, y, t]` ink. The future classifier
  obtains `128 x 5` tensors through `scripts/character_tensor_v1.py`, using
  per-stroke arc-length resampling and the channels `x`, `y`, `delta_t`,
  `stroke_start`, and `observed`.
