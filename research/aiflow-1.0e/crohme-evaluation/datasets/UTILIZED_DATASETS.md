# AIFlow Math Ink 1.0 utilized datasets

Last updated: 2026-08-12

## Character-classifier admission amendment (2026-08-12)

The classifier-only derivative now contains 189,334 records, 351,140 strokes,
and 28,054,726 points across five sources. One-point trajectories are rejected
as missing data: 664 HWRT, 49 UJI, and 2 ISGL records. UCI removes 138,695
consecutive exact X/Y duplicates (138,636 raw plus 59 after canonical rounding)
while preserving return-to-point loops.

BDSHWA is excluded from the character-classifier corpus. Its raw files provide
whole-task prompts and stroke IDs, but no character boundary, character label,
or character-to-stroke ownership. The raw archive remains local audit evidence
only and is not a tensor, training, or evaluation input.

## Current checkpoint evidence

| Dataset | Current use | Evidence boundary |
|---|---|---|
| AIFlow project-owned public ink | Current 0.9 calibrator training and replay evaluation | 110 valid formulas are eligible; the ownership training subset is 47 formulas / 211 symbols. Pending 4 and rejected 5 are excluded. |
| AIFlow project-owned ownership subset | 1.0 box-local symbol-head calibration after leave-one-writer-out validation | Three hashed writer groups; real rows calibrate only 25 observed math output labels. Final all-writer checkpoint does not score these training rows; no formula grouping, relation, or decoder supervision is admitted. |
| CROHME2019 valid | Noncommercial regression evaluation only | 985 parseable formulas; never product training or product-performance evidence. |

The current 0.9 checkpoint has not yet been retrained on the newly approved external pool below. The separate 1.0 research candidate first trains the external box-local base, then applies the writer-disjoint project-symbol calibration recorded in `research/PROJECT_SYMBOL_HEAD_CALIBRATION_20260812.md`.

## Canonical normalization derivative

`datasets/normalized/v1/` is a local-only, reproducible model-input derivative;
it is not committed and does not alter any raw archive. On 2026-08-11 the
builder and a separate byte-identical replay check completed with 191,385
records, 447,135 non-empty strokes, and 34,969,570 points.

| Source | Canonical records | Time representation | Explicit exclusions |
|---|---:|---|---:|
| Project-owned valid formulas | 110 | relative duration | 0 |
| UJI Pen v2 | 11,640 | ordinal point order; source has no point time | 0 |
| ISGL inherited online derivative | 7,414 | ordinal point order; source has no point time | 0 |
| UCI Character Trajectories | 2,858 | relative duration from documented 5 ms sampling | 0 |
| Curated HWRT | 168,027 | relative duration; 664 zero-duration samples use ordinal order | 0 |

The output keeps source stroke order, stores the bbox/letterbox transform, and
uses unit-square X/Y plus unit-interval time. It never implies that external
data can train formula grouping, ownership, relations, or final decisions.
See `NORMALIZATION_V1.md` for the contract and exact command.

## Approved 1.0 training pool

| Dataset | Available scope | Planned model role | Explicit limit |
|---|---|---|---|
| UJI Pen Characters v2 | 11,640 isolated characters, 60 writers, two sessions | Box-local online character trajectory pretraining | No time, pressure, formula relation, or layout labels |
| ISGL | 7,985 characters plus 3,790 words from 64 source writer IDs | English uppercase/lowercase, digits, and word-level trajectory expansion | Online data only; no per-point time; inherited derivative has 7,414 character rows and is training-only |
| UCI Character Trajectories | 2,858 samples, 20 lowercase Latin labels, one writer | Single-stroke representation pretraining | Treat as one-writer training data; never validation or model selection |
| HWRT / Detexify curated | 168,027 samples, 369 math-symbol labels, 317,996 strokes, 25,393,124 points | Box-local mathematical-symbol candidate pretraining | 193 reversing samples and 13 normalized duplicates removed; source user IDs are not reliable writer identities, so no model selection/final evaluation |
| BDSHWA | 1,348 raw trajectory CSV files across 29 participant folders | Excluded from the 1.0 character-classifier corpus | No character boundaries or character-to-stroke ownership; raw archive retained only for audit |

## Not used for product training

- Three Hugging Face `newbienewbie` trajectory datasets: rejected for missing licence/provenance and split leakage.
- CROHME and MathWriting-class corpora: noncommercial research evaluation only.
- BDSHWA: excluded from character-classifier training and evaluation because its labels are whole-task prompts, not character targets.
- BDSHWA processed biometric feature tables and participant metadata: excluded even though raw trajectories are approved for the restricted HWR role.

## Layer boundary

Within the current Math Ink model, external isolated-character or text corpora may improve only the box-local stroke encoder and character Top-k candidates. Formula grouping, stroke ownership, spatial relations, and final decisions remain supervised by project-owned formula data and validated through the existing holdout/regression gates.

ISGL word-level material may later support a separate general text-HWR branch. That future branch must keep its sequence/text objectives separate from mathematical formula layout and decision responsibilities.
