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
- Decision: use class-balanced sampling plus a capped loss weight when training begins. Do not use a predictive rebalancer and do not split source labels automatically.

## BDSHWA decision

- Rejected from the character-classifier corpus: whole-task prompts and stroke IDs do not supply character boundaries or ownership.
- Raw archive remains untouched as audit evidence; it is not a tensor, training, or evaluation input.
