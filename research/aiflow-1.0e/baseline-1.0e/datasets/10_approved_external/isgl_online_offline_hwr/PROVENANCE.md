# ISGL provenance and approved use

Audited 2026-08-10 from the original Mendeley Data version 1 download. The source is CC BY 4.0 and is admitted to the restricted AIFlow Math Ink 1.0 external training pool by project-owner decision on 2026-08-10.

## Source and integrity

- Source: ISGL Online and Offline Character Recognition Dataset
- DOI/source: https://doi.org/10.17632/n7kmd7t7yx.1
- Source-page licence: CC BY 4.0
- Local source: `raw/isgl_source.zip`
- Bytes: `974053582`
- SHA-256: `A94F2473246222F9470D4B93B68CFBC756ECB4865742DB8164788359FE511693`
- Outer ZIP CRC: passed for all 12 RAR payloads.
- The four online metadata RAR files extracted successfully with `bsdtar -m`.

The old interrupted downloads and range parts were removed after the complete source passed validation.

## Observed online structure

| Subset | Records | Writers | Representation |
|---|---:|---:|---|
| Uppercase characters | 3,382 | 64 | stroke-separated X/Y text |
| Lowercase characters | 3,314 | 64 | stroke-separated X/Y text |
| Digits | 1,289 | 64 | stroke-separated X/Y text |
| Words | 3,790 | 64 | stroke-separated X/Y text |

- Character total: 7,985 records, 62 labels, 13,739 declared strokes, and 388,148 points.
- Writer IDs are 1-60 and 62-65; ID 61 is absent, yielding the claimed 64 writers.
- Each stroke has X/Y points and an aggregate `TimeTaken`; there is no per-point timestamp. The source therefore does not support true temporal resampling.
- Source-quality findings: 553 character files contain at least one empty stroke, 225 contain at least one missing stroke duration, and 38 contain no non-empty stroke.

## Comparison with inherited normalized data

`migrated/isgl_online.jsonl.gz` contains 7,414 character records. For included records, labels and non-empty-stroke coordinates match the original after the documented empty-stroke removal. It is not a lossless copy:

- 571 original character records are absent.
- Every retained row is assigned to `train`; there is no writer-disjoint validation or test split.
- Every row embeds `eligible_for_training=true`; at audit time this contradicted its pending-registry track and blocked-tier location.
- Timestamps are null and `timestamp_mode=canonical`; only aggregate stroke durations survive.

The embedded eligibility flag now agrees with the approved registry state, but the inherited derivative remains an incomplete 7,414-row training-only view rather than the canonical source.

## Known limitations and adopted scope

1. Use is based on the source-page CC BY 4.0 licence; attribution must accompany releases.
2. The inherited derivative omits 571 rows and has no exclusion manifest, so complete-corpus work must rebuild from the original source.
3. Existing rows are all `train`; ISGL cannot select a model or provide final evaluation evidence until a reproducible source rebuild creates source-writer groups.
4. Use only the online English uppercase/lowercase, digit, and word trajectories. Offline raster data is outside this HWR track.
5. ISGL cannot supervise mathematical layout, stroke ownership, spatial relations, or the final decision layer.

## Operating rules

- Register source/DOI attribution and retain the original archive hash.
- The 7,414-row inherited derivative may be used for initial training only; prefer a deterministic rebuild of all valid source rows.
- Do not report ISGL-only validation as product performance.
- Gate any resulting checkpoint on project-owned writer-holdout and the existing regression suite.

Decision: **APPROVED EXTERNAL / RESTRICTED - online English character, digit, and word trajectory training only.**
