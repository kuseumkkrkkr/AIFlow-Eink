# Project symbol-head calibration - 2026-08-13

Status: the five-epoch calibration remains the current local research final.
The seven-epoch and `math-observed-one` eight-epoch external-only bases both
pass writer-LOO calibration gates but are not promoted over it on a mixed
Top-1/Top-5 result. `product_adopted` remains false. This is a box-local
output-head calibration, not a formula recognizer.

## Scope and split

- Bases: the five-epoch current final and seven-epoch external-only candidate
  in `CHARACTER_CLASSIFIER_TUNING_20260812.md` (372 math outputs and 95
  auxiliary outputs).
- Project data: 211 manually ownership-grounded symbol trajectories from 47
  formulas across three hashed writer groups. It covers 25 observed math
  labels, including 6 `(`, 6 `)`, and 20 `=` samples.
- Validation: leave one project writer group out; all fixed external holdout
  rows are excluded from calibration and rehearsal.
- Finalization uses all real project rows only after writer-LOO passes. It
  deliberately marks those same rows `not_scored` after finalization.

## Method

`scripts/calibrate_project_punctuation_v1.py` keeps the encoder, auxiliary
head, and every non-project math output row byte-identical.

1. Calibrate `(`, `)`, and `=` for 250 class-balanced steps (batch 16).
2. Calibrate the other observed project math outputs for 250 class-balanced
   steps (batch 32), leaving first-stage rows fixed.
3. Each step rehearses external **training** data: 216 UJI parenthesis rows
   plus a deterministic 8,192-row non-punctuation math sample. Rehearsal loss
   weights are 8:1 then 16:1.

The selected calibration learning rate is `5e-4`. No raw archive, canonical
derivative, formula grouping, ownership grouping, or fixed holdout row is
changed or used as calibration input.

## Writer-disjoint selection

| Candidate | External base Top-1 / Top-5 | Project LOO Top-1 / Top-5 | Decision |
|---|---:|---:|---|
| 3-epoch base + `5e-4` calibration | 66.43% / 94.25% | 64.93% / 88.15% | former candidate |
| **5-epoch base + `5e-4` calibration** | **74.10% / 96.57%** | **66.35% / 91.00%** | **selected** |
| 7-epoch base + `5e-4` calibration | 76.02% / 96.96% | 64.93% / 89.57% | gates pass; do not replace selected final |
| 8-epoch `math-observed-one` + `5e-4` calibration | **77.33% / 97.29%** | **66.82% / 88.15%** | gates pass; Top-1/external challenger, Top-5 below selected final |

For the selected candidate, calibration improves its own LOO direct baseline
from 35.07% / 55.45% to 66.35% / 91.00% (Top-1 / Top-5), a 31.28-point Top-1
gain. Punctuation improves from 0/32 to 26/32 Top-1 and 31/32 Top-5. The
weakest calibration fold loses only 0.08 external-math Top-1 points, within
the one-point gate; all three writer-disjoint folds pass the direct gain gate.

The seven-epoch candidate's own LOO direct baseline improves from 45.02% /
65.88% to 64.93% / 89.57% (a 19.91-point Top-1 gain). Punctuation improves
from 0/32 to 22/32 Top-1 and 28/32 Top-5; fold gains are 14.29, 22.37, and
20.00 Top-1 points, and worst external-math regression is 0.20 points. Thus
it passes the preset research gates. It remains 1.42 / 1.43 points below the
five-epoch calibrated LOO result and has lower punctuation accuracy. The same
three writer groups must not be reused to tune learning rate, duration, or
rows, so no further correction tuning was performed.

The eight-epoch `math-observed-one` base fixes the UJI-parenthesis
source-availability leak at the math-head input. Its own LOO direct baseline
improves from 45.50% / 64.93% to 66.82% / 88.15%; punctuation improves from
10/32 to 26/32 Top-1 and 12/32 to 30/32 Top-5. Its worst calibrated
external-math result is 0.41 points above its uncalibrated base, so it passes
the external regression gate. It improves calibrated Top-1 by 0.47 points
over the selected five-epoch result, but loses 2.84 Top-5 points. The three
writer groups are already consumed by this comparison, so it is a recorded
research challenger rather than an all-writer final checkpoint.

## Final research checkpoint

`artifacts/training_length_20260812/project_symbol_5ep_final_lr5e-4/`
contains the current local-only final checkpoint. On the fixed external
holdout it reaches 74.02% Top-1 / 96.46% Top-5, versus 74.10% / 96.57% for the
uncalibrated five-epoch base. Math punctuation is 24/24 Top-1 and Top-5.

`artifacts/training_length_20260813/project_symbol_7ep_loo_lr5e-4/` is a
writer-LOO decision artifact only; no seven-epoch all-writer finalization was
run. This preserves the LOO result as an independent gate.

Only 25 math-head weight rows and their corresponding bias rows can change;
the encoder, auxiliary head, and every other math output row are asserted
byte-identical. Direct ownership is intentionally `not_scored` for this final
checkpoint because all 211 rows supplied calibration labels.

## Remaining gate

This is not commercial acceptance: three writer groups selected the calibration
setting and no independent new project-writer holdout remains. `a` and `f`
have only one writer group (four rows total). Collect new writers for a
post-finalization regression set, then separately implement and validate
stroke grouping, spatial relations, and formula decoding.

## Verification

- The five-epoch LOO report passes both research gates; calibration output
  changed only the declared 25 math-head rows.
- The calibration, classifier, tensor, and replay evaluator self-tests are
  re-run after the code change.
- Dataset normalization and archive verification are re-run in verify-only
  mode without modifying raw data.
