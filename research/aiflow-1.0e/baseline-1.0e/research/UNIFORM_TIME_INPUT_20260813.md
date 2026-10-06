# Uniform-time input experiment

Status: completed research ablation; rejected as the global default.

Follow-up: the uncalibrated model decision below remains valid, but the
writer-disjoint output-row calibration completed on 2026-08-14 makes uniform
time the current research candidate for project ink. It is still not product
adopted; see `HWR_48HZ_WRITER_LOO_RESOLUTION_20260814.md`.

## Controlled change

- Kept the unified 372-class head, 128x5 input, encoder, seed, sampler,
  optimizer, fixed holdout, and eight epochs unchanged.
- Replaced every tensor's `delta_t` with one normalized 128-step clock:
  `[0, 1/127, ..., 1/127]`.
- Forced `observed=1` for every source so neither timing availability nor source
  timing quality could identify the corpus.
- Preserved X/Y, point order, stroke boundaries, and learned position
  embeddings. Raw and canonical datasets were not rewritten.
- DTW was not used because an inference-time warp requires an unknown class
  template and would confound this timing-only ablation.

Before the change, all corpus-level median `delta_t` sums were one, but median
within-sequence standard deviation differed: HWRT 0.01964, UCI 0.01338, ISGL
0.00627, and UJI 0.00081. Equal range therefore did not mean equal timing
quality.

## Accuracy

| Evaluation | Relative-time baseline Top-1 / Top-5 | Uniform time Top-1 / Top-5 | Change |
|---|---:|---:|---:|
| Fixed external, 18,344 | 79.16% / 97.66% | 77.20% / 97.35% | -1.96 / -0.31 pp |
| HWRT, 16,574 | 79.02% / 97.53% | 77.37% / 97.35% | -1.65 / -0.17 pp |
| UJI, 792 | 79.67% / 98.74% | 72.47% / 96.84% | -7.20 / -1.89 pp |
| ISGL, 700 | 75.14% / 98.71% | 70.71% / 96.86% | -4.43 / -1.86 pp |
| UCI, 278 | 96.04% / 100.00% | 97.12% / 100.00% | +1.08 / 0.00 pp |
| Project ownership, 211 glyphs | 50.24% / 80.57% | 57.82% / 82.46% | +7.58 / +1.90 pp |

On paired external rows, Top-1 had 885 gains and 1,244 losses; the 95% glyph
bootstrap interval for the change was -2.45 to -1.47 percentage points. Top-5
was also negative at -0.49 to -0.13 points.

Project Top-1 had 25 gains and 9 losses. Its glyph bootstrap interval was
+2.37 to +12.80 points, but the project set contains only three writer groups;
the writer-cluster interval was -5.71 to +11.00 points. Project Top-5 intervals
also crossed zero. The apparent project gain is therefore not sufficient for
product adoption.

## Decision

Reject complete uniform timing as the global default. It improves the small
project set but significantly lowers the much larger fixed external holdout,
especially UJI and ISGL. Their non-physical ordinal timing still carries
sampling-density information after arc-length resampling; deleting all local
variation removes useful trajectory information together with timing-domain
bias.

Retain `uniform-time` only as an ablation and possible training augmentation.
A later timing policy should preserve bounded within-glyph rhythm while
removing corpus-level scale, and must be selected on writer-disjoint project
data with more than three writers.

## Reproducible outputs

- Report: `artifacts/time_normalization_20260813/uniform_time_unified_math_8ep_full/training_report.json`
- Checkpoint: `artifacts/time_normalization_20260813/uniform_time_unified_math_8ep_full/classifier_checkpoint.pt`
- External predictions: `artifacts/time_normalization_20260813/uniform_time_unified_math_8ep_full/external_holdout_predictions.jsonl`
- Project predictions: `artifacts/time_normalization_20260813/uniform_time_unified_math_8ep_full/direct_ownership_predictions.jsonl`
