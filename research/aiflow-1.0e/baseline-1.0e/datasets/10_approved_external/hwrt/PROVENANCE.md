# HWRT / Detexify provenance and curated approval

Audited 2026-08-10 from the official Zenodo record. The project owner approved restricted box-local mathematical-symbol training after a deterministic privacy and integrity filter was produced on 2026-08-10.

## Source and integrity

- Source: HWRT / Detexify handwritten mathematical symbols, 2015-01-28 release
- Zenodo: https://zenodo.org/records/50022
- Licence: ODbL 1.0
- Local source: `raw/2015-01-28-data.tar`
- Bytes: `140790596`
- MD5: `2BF1D089CE65C0A39E57064516F1BD1C` (matches the Zenodo file record)
- SHA-256: `B96FEAFD71B01F1623997DFF3CC8AC4D18628D128CEE1B3DF1880518BBA3EA4A`
- TAR members: `symbols.csv`, `train-data.csv`, and `test-data.csv`; all members read successfully.

## Observed structure

- 369 unique mathematical-symbol IDs.
- 151,159 train rows and 17,074 test rows; every row parses as `symbol_id;user_id;strokes;user_agent`.
- Each stroke stores ordered `{x, y, time}` points. Samples contain 1-43 strokes and 1-1,475 points.
- Symbol-level train/test counts exactly match `symbols.csv`.
- Train contains 463 user IDs; test contains 118; 103 user IDs occur in both splits.
- Eight exact duplicate rows occur inside train, and two exact records occur in both train and test.
- Time bases are mixed: 1,379 train and 161 test samples use relative millisecond-like values, while the rest use epoch-millisecond-like values.
- 1,186 point-to-point time reversals were observed across the full corpus.
- `user_id` and full browser `user_agent` strings are present.

## Curated derivative

- Builder: `scripts/build_hwrt_curated.py`
- Manifest: `derived/manifest.json`
- Accepted: 168,027 of 168,233 rows; 369 labels, 317,996 strokes, and 25,393,124 points.
- Rejected: 193 samples containing 1,209 cross-point timestamp reversals and 13 normalized exact duplicates.
- The earlier audit's 1,186 count used its original transition convention; the adoption filter is stricter and checks every adjacent point across stroke boundaries, producing the authoritative 1,209 rejection-event count.
- Time: absolute origins removed; every retained sample starts at relative `t=0`; output reversal count is zero.
- Privacy: raw user IDs and full browser user agents are absent. Source IDs are replaced with dataset-local group keys.
- Split-key overlap: train/validation/test overlap is zero for the available source IDs.
- Split limitation: the [HASYv2 paper](https://arxiv.org/abs/1701.08380) states that source ID `16925` combines many Detexify contributors and that exact writer identity cannot be recovered. Its 153,660 retained samples are therefore marked `detexify_aggregate_unknown_writers` and kept in training only. True writer-disjoint evaluation is unverifiable.

## Operating rules

- Train only from the curated derivative; never load `user_agent` or raw `user_id` into a model pipeline.
- Treat all HWRT splits as training-pool/integrity partitions, not model-selection or final-evaluation evidence.
- Limit use to the box-local mathematical-symbol candidate model. HWRT supplies no formula grouping, ownership, or spatial-relation labels.
- Preserve Zenodo attribution and ODbL notices for the source and any distributed derivative database.
- Adopt a resulting checkpoint only after project-owned writer-holdout and regression gates pass.

Decision: **APPROVED EXTERNAL / CURATED - mathematical-symbol candidate training only; no HWRT-based model selection or final evaluation.**
