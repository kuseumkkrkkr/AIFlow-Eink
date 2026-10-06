# Character classifier training readiness

Status: the external-only two-epoch pilot, fixed-split tuning loop, and a
writer-disjoint project symbol-head calibration are complete; none is product
adopted. The current research candidate is documented in
`PROJECT_SYMBOL_HEAD_CALIBRATION_20260812.md`.

## Admission decisions

- Reject every one-point trajectory as missing character data. The canonical
  derivative records a hashed rejection while retaining raw source archives.
- For UCI, remove only consecutive exactly equal X/Y coordinates, including
  59 adjacent collisions created by canonical coordinate rounding. The visual
  before/after audit confirms preserved glyph geometry and preserved loops.
- Reject BDSHWA from the character-classifier corpus. Its task prompt describes
  a whole word, sentence, shape, wave, topic, or freehand task; no character
  boundary or character-to-stroke ownership is available.
- Treat HWRT labels as atomic source annotation units. Do not automatically
  split compositional LaTex labels such as `\\not\\equiv` or `\\sqrt{}`.
- The historical feasibility baseline applied both class-balanced sampling and
  capped loss weights to HWRT. The tuning loop found this was double
  correction; the current research default uses the sampler only. Do not add a
  predictive rebalancer: it cannot create handwriting variation for rare
  classes and would bias priors.

## Tuning amendment

- UJI `(` and `)` now route to the 372-class math head, not the auxiliary
  head. This preserves one target head per source row and reduces the
  auxiliary vocabulary from 97 to 95 labels.
- `=` has no real admitted **external** training row. Synthetic equals
  generated from dashes were tested and rejected; they are not present in the
  pipeline. The separate project calibration uses only real writer-disjoint
  project `=` trajectories.
- The fixed raw and normalized datasets remain unchanged. The selected cache
  is a local, D:-resident v2 derivative and is reproducible only when its
  manifest matches the pinned canonical manifest and holdout IDs.
- `calibrate_project_punctuation_v1.py` first calibrates `(`, `)`, and `=`,
  then the remaining observed project output rows. It uses external training
  rehearsal and explicitly proves that no other model row changes.

## Tensor contract

`scripts/character_tensor_v1.py` converts a canonical box-local trajectory to
an exact `128 x 5` float tensor:

1. `x`
2. `y`
3. `delta_t`
4. `stroke_start`
5. `observed`

It allocates at least one point per stroke, then uses per-stroke arc-length
resampling. `delta_t` retains the gap into each later stroke and `stroke_start`
marks that boundary. It never crops by raw point index. The generated local manifest
contains 167,363 usable HWRT rows, 211 ownership-grounded project rows from
47 formulas, and a deterministic 18,723-row holdout over all approved external
sources: HWRT 16,574, UJI 1,159, ISGL 712, and UCI 278.

The highest source stroke count is 43 (HWRT), below the fixed 128-point budget;
every admitted source therefore retains at least one resampled point per stroke.

## Evaluation protocol and pilot boundary

| Protocol | Scope | Metric | Evidence limit |
|---|---|---|---|
| CROHME2019 | Noncommercial InkML only | Formula exact | Never product training or product evidence |
| Project ownership | 47 formulas / 211 symbol groups across three hashed writers | External-only frozen score, then leave-one-writer-out symbol calibration score | Final all-writer calibration must not score its own project rows |
| Project canonical replay | Deterministic 10 of 110 formulas | Formula exact | Requires an end-to-end grouping/decision predictor |
| Current approved external data | Deterministic 10% per source and symbol, minimum one row | Symbol Top-1 and Top-5, reported by source | Technical holdout, not product evidence |

`scripts/replay_evaluate_hwr_v1.py` writes point-by-point replay pages and
scores supplied prediction JSONL. The external-only checkpoint and calibrated
candidate can score only box-local character protocols; formula-exact CROHME
and canonical replay still require the absent grouping and decision pipeline.
