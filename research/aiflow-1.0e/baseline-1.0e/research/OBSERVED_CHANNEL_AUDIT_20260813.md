# Observed-channel audit - 2026-08-13

## Decision

- Keep all approved parenthesis data.
- Reject global `observed=0`: it lowers fixed external Top-1 by 2.01 points.
- Admit `math-observed-one` as a research-only input-policy candidate: force
  `observed=1` for the math head and preserve it for the auxiliary head.
- Keep the five-epoch all-writer calibrated checkpoint as the local final:
  it has higher writer-LOO Top-5 until a product ranking criterion and fresh
  writer holdout are declared.

## Cause

`observed` is source timestamp availability, not ink geometry. Before repair:

| Math training rows | `observed=0` | `observed=1` |
|---|---:|---:|
| `(` | 108 | 0 |
| `)` | 108 | 0 |
| all other math rows | 0 | 150,789 |

All 12 direct project parentheses have `observed=1`. On the frozen
seven-epoch model, changing only that channel from one to zero turns the
direct parentheses from 0/12 to 12/12 Top-1 and Top-5. This identifies a
label-correlated source cue rather than a missing-coordinate error.

## Data support audit

| Token | HWRT | UJI | ISGL | UCI | Current conclusion |
|---|---:|---:|---:|---:|---|
| `(` | 0 | 120 | 0 | 0 | UJI supplies approved isolated examples; repair input cue |
| `)` | 0 | 120 | 0 | 0 | UJI supplies approved isolated examples; repair input cue |
| `1` | 118 | 120 | 115 | 0 | support exists, but UJI/ISGL currently route to auxiliary head |
| `=` | 0 | 0 | 0 | 0 | no approved isolated commercial source; use only writer-LOO project-head calibration |

CROHME remains noncommercial evaluation-only. BDSHWA does not provide an
approved isolated-character boundary and label derivative, so it cannot be
used as a commercial `=` source.

## Controlled results

All trials use the same external cache, 128 x 5 model contract, seed,
sampler-only balancing, and fixed external holdout. The global and math-head
policies are stored in each checkpoint input contract.

| Candidate | External Top-1 / Top-5 | Direct raw Top-1 / Top-5 | Direct parentheses Top-1 / Top-5 |
|---|---:|---:|---:|
| Seven-epoch preserved | 76.02% / 96.96% | 45.02% / 65.88% | 0/12 / 0/12 |
| Eight-epoch global zero | 74.02% / 96.52% | 42.65% / 69.19% | 7/12 / 11/12 |
| Eight-epoch math-head one | **77.33% / 97.29%** | **45.50% / 64.93%** | **10/12 / 12/12** |

The all-math `observed=1` policy makes UJI parentheses match the direct math
input contract. It does not alter the stored tensors, coordinates, timing,
stroke boundaries, labels, source splits, or output vocabulary.

## Calibration gate

`math-observed-one` writer-LOO calibration passes both predeclared gates:

- direct Top-1: 45.50% -> 66.82%;
- punctuation: 10/32 -> 26/32 Top-1 and 12/32 -> 30/32 Top-5;
- worst calibrated external-math change: +0.41 Top-1 points.

It does not replace the local five-epoch final: the latter has 66.35% /
91.00% LOO Top-1 / Top-5, versus 66.82% / 88.15% for this challenger. The
new three-group result is not used to tune another calibration pass.

## Next separate decision

Move UJI and ISGL digit `1` from the auxiliary target to the math target only
through a new fixed-split data-admission experiment. Do not duplicate rows
across heads and do not add synthetic `=` examples. A commercial isolated
`=` corpus or new writer-disjoint project data is required to improve raw
base support for equality.
