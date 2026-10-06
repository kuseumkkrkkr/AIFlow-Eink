# Character classifier data audit

## UCI duplicate decision

- Raw UCI samples: 2858
- Raw exact consecutive duplicate XY points: 138636
- Canonical duplicate removals: 138695 (including 59 rounding collisions).
- Remaining canonical exact consecutive duplicates: 0
- Samples affected: 2858
- Decision: remove only adjacent exact XY duplicates; preserve return-to-point loops.

## HWRT support distribution

- Usable samples: 167363 across 369 classes.
- Mean 453.56; median 221.0; p05 58; p95 1526; range 51..3512.
- Distribution bands: `{"0.25x_to_0.5x_mean": 104, "0.5x_to_1x_mean": 75, "1x_to_2x_mean": 50, "2x_to_4x_mean": 43, "above_4x_mean": 14, "at_or_below_0.25x_mean": 83}`.
- Decision: retain class-balanced sampling only. The prior sampler-plus-loss double correction was rejected; do not use a predictive rebalancer or split atomic source labels automatically.

## BDSHWA decision

- Rejected from the character-classifier corpus: whole-task prompts and stroke IDs do not supply character boundaries or ownership.
- Raw archive remains untouched as audit evidence; it is not a tensor, training, or evaluation input.

## Classifier-input audit

- math: 151005 training rows; 371/372 output labels observed; missing `=`.
- math: support range 46..3161; 82 labels at or below 0.25x observed-class mean and 14 above 4x.
- math: invalid tensor rows 0; spatially collapsed rows 38.
- math: exact 128x5 duplicate groups 13; train/eval same-label groups 4; cross-label groups across splits 1.
- math: spatially collapsed training labels `{"\\bullet": 2, "\\cdot": 33, "\\dotsc": 2, "\\vdots": 1}`.
- math: cross-label exact-tensor evidence `[{"records": 26, "splits": {"eval": ["\\cdot"], "train": ["\\bullet", "\\cdot", "\\dotsc", "\\vdots"]}}, {"records": 2, "splits": {"train": ["\\bullet", "\\cdot"]}}, {"records": 2, "splits": {"train": ["\\backslash", "\\cdot"]}}]`.
- auxiliary: 19496 training rows; 95/95 output labels observed; missing `none`.
- auxiliary: support range 64..380; 0 labels at or below 0.25x observed-class mean and 0 above 4x.
- auxiliary: invalid tensor rows 0; spatially collapsed rows 0.
- auxiliary: exact 128x5 duplicate groups 0; train/eval same-label groups 0; cross-label groups across splits 0.
- Decision: exact-tensor collisions and fixed-holdout predictions are diagnostics, not a source of new labels, data deletion, or correction tuning.

## Fixed-holdout character confusions

- math: top diagnostic Top-1 confusions: `\sum -> \Sigma` (202), `\alpha -> \propto` (77), `\Omega -> \ohm` (71), `\bot -> \perp` (67), `\in -> \epsilon` (56).
- math: diagnostic Top-1 by training-support band: 0.25x_to_0.5x_mean: 74.62%, 0.5x_to_1x_mean: 71.78%, 1x_to_2x_mean: 70.42%, 2x_to_4x_mean: 78.10%, above_4x_mean: 79.55%, at_or_below_0.25x_mean: 75.25%.
- auxiliary: top diagnostic Top-1 confusions: `z -> Z` (22), `O -> 0` (18), `o -> 0` (18), `v -> V` (18), `C -> c` (15).
- auxiliary: diagnostic Top-1 by training-support band: 0.25x_to_0.5x_mean: 28.57%, 0.5x_to_1x_mean: 85.42%, 1x_to_2x_mean: 79.01%.
- The fixed external holdout remains excluded from epoch selection and correction-set expansion.

## Character-correction boundary

- Status: writer_disjoint_reported; decision: `retain_only_the_existing_writer_disjoint_output_row_calibration; do_not_edit_labels_or_expand_the_correction_set_from_fixed_holdout_diagnostics`.
- Existing writer-LOO output-row calibration: 0.4502 -> 0.6493 direct Top-1; punctuation 0.0000 -> 0.6875; accepted=True.
