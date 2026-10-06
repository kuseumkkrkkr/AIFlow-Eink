# Unified 372-class output-head experiment

Status: completed research challenger; not product-adopted.

## Controlled change

- Replaced the 372-class math plus 95-class auxiliary outputs with one
  372-class math output.
- Kept the encoder, 128x5 tensor contract, optimizer, seed, sampler,
  `math-observed-one` policy, fixed holdout IDs, and eight epochs unchanged.
- Routed UJI, ISGL, and UCI rows to the math output only when their label was an
  exact member of the 372-class vocabulary. No semantic label conversion was
  used.
- Training rows: 167,088. Fixed-holdout rows in scope: 18,344.

## Accuracy

| Evaluation | Two-head baseline Top-1 / Top-5 | Unified Top-1 / Top-5 | Change |
|---|---:|---:|---:|
| Same 18,344 external rows | 77.17% / 97.26% | 79.16% / 97.66% | +1.99 / +0.40 pp |
| HWRT, 16,574 | 76.76% / 97.10% | 79.02% / 97.53% | +2.26 / +0.42 pp |
| UJI exact-overlap, 792 | 77.53% / 98.48% | 79.67% / 98.74% | +2.15 / +0.25 pp |
| ISGL exact-overlap, 700 | 79.14% / 98.57% | 75.14% / 98.71% | -4.00 / +0.14 pp |
| UCI, 278 | 95.68% / 100.00% | 96.04% / 100.00% | +0.36 / 0.00 pp |
| Project ownership, 211 glyphs | 45.50% / 64.93% | 50.24% / 80.57% | +4.74 / +15.64 pp |

The unified checkpoint contains no auxiliary-head parameters. Its 18,344-row
external score is 79.16% / 97.66%; comparison above uses those exact rows for
both models because the former two-head report also includes 379 non-overlap
general-text rows.

## Limits and decision

- Project digit `1` Top-5 improved from 8% to 84%, but Top-1 remains 0%.
- Project `=` remains 0% for Top-1 and Top-5 because no admitted external
  training row exists.
- Project `(` and `)` Top-1 fell from 83.33% each to 50.00% and 66.67%; each
  class has only six evaluation rows.
- ISGL Top-1 fell by 4.00 percentage points on the exact shared rows.

The one-head design is retained as the stronger research challenger. It must
not replace the current product checkpoint until punctuation calibration and
writer-disjoint regression gates pass.

## Reproducible outputs

- Report: `artifacts/unified_head_20260813/unified_math_8ep_full/training_report.json`
- Checkpoint: `artifacts/unified_head_20260813/unified_math_8ep_full/classifier_checkpoint.pt`
- Predictions: `artifacts/unified_head_20260813/unified_math_8ep_full/external_holdout_predictions.jsonl`
- Direct predictions: `artifacts/unified_head_20260813/unified_math_8ep_full/direct_ownership_predictions.jsonl`
