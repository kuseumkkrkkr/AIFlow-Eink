# Canonical online-ink normalization execution — 2026-08-11

## Completed result

- Command: `python scripts/build_normalized_ink_v1.py`
- Output: local-only `datasets/normalized/v1/` on D:, ignored by Git.
- Validator result: 191,385 records, 447,135 strokes, 34,969,570 points, six sources.
- Determinism: `python scripts/build_normalized_ink_v1.py --deterministic-replay-only` rebuilt all six sources in an isolated D: temporary directory and produced an identical manifest and gzip SHA-256 values.

## Canonical contract applied

- X/Y: sample bbox translation, max-side scale, and aspect-preserving unit-square letterbox.
- Stroke sequence: non-empty strokes retain source order. Empty source strokes are recorded by source index; raw source files are never modified.
- Time: monotonic source time becomes relative duration; missing time becomes explicitly marked ordinal sequence. BDSHWA clock resets only at some stroke boundaries, so each affected sample retains intra-stroke durations while the stroke sequence is stitched without fabricated pen-up gaps.
- Privacy: normalised records use hashed record IDs/source fingerprints and exclude BDSHWA paths, participant/session IDs, raw timestamps, demographics, and biometric metadata.

## Exceptional source records

- BDSHWA: 34 of 1,336 emitted records require `stitched_stroke_duration` because source timestamps overlap/reset at stroke boundaries but remain monotonic within each stroke.
- BDSHWA: 12 raw CSV files contain no pen-down points. They remain in raw storage and are listed in the local `bdshwa_rejections.jsonl.gz` using hashed identifiers and the explicit `empty_after_source_pen_down_filter` reason.
- HWRT: 664 zero-duration trajectories use `ordinal_index`; their source stroke and point order remains intact.

## Visual spot check

An after-normalization gallery was rendered and directly inspected for all six sources. Each sample is within `[0,1]` X/Y; multi-stroke order remains visible. BDSHWA text trajectories appear vertically compact because the rule preserves their wide aspect ratio rather than distorting characters.

## Scope boundary

This completes input canonicalization only. It does not retrain, evaluate, or promote a model. External corpora remain representation/box-local pretraining material, while project-owned formula ink retains the grouping, ownership, relation, and final-decision role.
