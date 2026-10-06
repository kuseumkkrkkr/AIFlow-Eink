# Final frozen grouping delta

- Training/selection/threshold tuning on validation data: **0**
- Grouping exact: **39.01% → 40.83%**
- Improved/regressed/net formulas: **24 / 10 / +14**
- Strict formula proxy: **6.50% → 6.50%**

## Stroke-count bands

| Input strokes | N | Baseline exact | Candidate exact | Improved | Regressed |
|---:|---:|---:|---:|---:|---:|
| 01-07 | 248 | 61.29% | 61.29% | 0 | 0 |
| 08-15 | 273 | 34.80% | 34.80% | 0 | 0 |
| 16-23 | 167 | 28.74% | 34.13% | 18 | 9 |
| 24-31 | 55 | 7.27% | 18.18% | 6 | 0 |
| 32+ | 26 | 3.85% | 0.00% | 0 | 1 |

## Decision

The conservative long-formula blend improves grouping modestly, but does not improve strict formula exactness. It remains shadow-only; a larger fresh, project-owned long/2D grouping corpus is required before product promotion.
