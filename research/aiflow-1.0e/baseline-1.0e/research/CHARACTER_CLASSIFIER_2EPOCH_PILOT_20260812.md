# Character classifier two-epoch feasibility pilot

Status: completed on 2026-08-12; not product adopted. This is the historical
double-balanced control, not the current tuning default. See
`CHARACTER_CLASSIFIER_TUNING_20260812.md` for the selected sampler-only
configuration and its limits.

## Historical fixed run

- Exactly two epochs, CUDA GTX 1650 (4 GB), batch size 64, AdamW `3e-4`.
- Fixed architecture: `128 x 5` input, `5 -> 128`, four 128-wide
  four-head Transformer blocks, attention pooling, a 372-class math head, and
  a 97-class auxiliary head.
- Training rows: 150,789 HWRT math rows and 19,712 UJI/ISGL/UCI auxiliary rows.
- The fixed 18,723-row external holdout, all 47 project-owned ownership
  formulas / 211 groups, CROHME, BDSHWA, and direct canonical replay were not
  training input.
- Class-balanced sampling and capped inverse-frequency loss weights were used.
  This is retained as a reproducible control only; later tuning found the two
  frequency corrections together were counterproductive.

## Optimization result

| Head | Epoch 1 loss | Epoch 2 loss |
|---|---:|---:|
| Math | 2.8179 | 0.8797 |
| Auxiliary | 2.4073 | 1.0314 |

No non-finite loss, CUDA OOM, or missing prediction occurred. Runtime was
811 seconds.

## Fixed external holdout

| Source | Rows | Top-1 | Top-5 |
|---|---:|---:|---:|
| HWRT | 16,574 | 38.72% | 80.75% |
| UJI | 1,159 | 72.22% | 95.43% |
| ISGL | 712 | 66.15% | 95.22% |
| UCI | 278 | 88.49% | 99.64% |
| All sources | 18,723 | 42.58% | 82.49% |

The result is materially above chance (math Top-1 0.27%, auxiliary Top-1
1.03%), so the tensor, vocabulary, GPU runtime, and shared-encoder objective
are technically trainable.

## Project-owned ownership evaluation

- All 211 box-local groups: Top-1 25.12%, Top-5 43.60%.
- Of the 179 groups whose 369 HWRT-backed labels were trained: Top-1 29.61%,
  Top-5 51.40%.
- The remaining 32 groups are `(`, `)`, or `=`. Those three required math-head
  outputs have no training example under the protected ownership-evaluation
  split, so they score 0/32. This is expected and must not be masked.

## Decision

**Technically feasible; not commercially ready.** The two-epoch run proves the
data loader, 128-point tensor contract, two-head Transformer, balance policy,
and checkpoint path work without leakage into the fixed external holdout.

It is not a product acceptance result because direct ownership transfer is
weak, 29 HWRT labels have no Top-1 hit in this short pilot, the three direct
math outputs above are untrained, and no formula grouping/relation/decoder
exists. Therefore CROHME's 985 formulas and the 10 direct canonical formula
replays are explicitly **not scored**.

Next admission gate: collect a writer-disjoint project-owned training split
that covers `(`, `)`, and `=`, then implement grouping and formula decoding
before using formula-exact metrics. The local checkpoint and prediction files
remain under `artifacts/character_classifier_v1_pilot_2ep/` and are not a
released model.
