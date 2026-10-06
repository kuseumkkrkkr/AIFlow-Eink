# BDSHWA provenance and approved use

Audited 2026-08-10 from Mendeley Data version 1. The CC BY 4.0 raw online
trajectories were initially considered for restricted non-formula HWR expansion.
The 2026-08-12 character-classifier re-audit below supersedes that scope for
AIFlow Math Ink 1.0: the archive is retained for audit only.

## Source and integrity

- Source: BDSHWA: Bengali-English Online Handwriting Dataset for Forensic Biometric Analysis
- DOI/source: https://doi.org/10.17632/99t9jhvksv.1
- Source-page licence: CC BY 4.0
- Local source: `raw/bdshwa_v1.zip`
- Bytes: `1318392171`
- SHA-256: `45FBBDCD1A9C4353F5C93BF8096FBEE9723931185F7BBFED3A86E21BA4CF20EA`
- Outer ZIP CRC: passed.

The outer archive contains two 727 MB nested ZIPs. Both nested ZIPs pass full CRC validation and contain the same 10,895 files; only `README.pdf` differs, with the newer copy replacing the draft citation with the Mendeley DOI citation. The trajectory data is otherwise CRC-identical.

## Observed structure

- 29 participant folders and metadata records.
- Raw data: 1,348 CSV files with one consistent 20-column schema: timestamp, raw/canvas X/Y, pressure, tilt, distance, pen state, stroke ID, task/script/prompt, sample/session, condition, and task index.
- Processed data: 1,338 aligned CSV files plus `master_feature_index.csv`; two aligned files reorder one derived distance feature, so processed headers are not fully uniform.
- Other raw modalities are incomplete relative to CSV: 1,344 PS, 1,346 PDF, 1,344 base PNG, 1,334 cropped PNG, and 1,334 normalized PNG files.
- Stratified audit: the smallest raw CSV for every participant and each of six categories was parsed (174 files, 1,260,201 rows, about 206 MB). All sampled rows had 20 fields, finite numeric values, monotonic timestamps, legal pen-state values, and consistent task/script context.
- The raw HWR fields are relevant to digital ink, but the prompts are Bengali/English sentences, words, freehand text, shapes, and waves rather than mathematical symbols or formula layout.

## Privacy and integrity findings

- Metadata contains age, gender, handedness, detailed collection timestamps, and forensic/biometric features.
- `Raw_Data/P012/metadata.json` incorrectly declares its internal `participant_id` as `P010`, duplicating another participant ID.
- The README states written consent, pseudonymization, CC BY 4.0 use including commercial research, and prohibitions on re-identification/surveillance. It describes the ethics status only as `Self-consented academic research`; no institutional approval identifier or commercial-product consent scope is supplied.
- The README still contains a `[City, Country]` placeholder.

## Adopted scope and exclusions

1. Historical consideration only: raw online trajectories were considered for general HWR representation and English/Bengali character, word, or sentence expansion. This is superseded for the current 1.0 character classifier.
2. Exclude participant metadata, processed forensic features, writer-identification labels, demographics, age, gender, and biometric objectives from every training manifest.
3. The `P012`/`P010` metadata collision does not enter training because all participant metadata is excluded.
4. It does not provide mathematical symbols, formula structure, stroke ownership, or relation labels and therefore cannot supervise those components.
5. Preserve source attribution, the stated no-reidentification/no-surveillance restrictions, and the CC BY 4.0 notice.

## Character-classifier re-audit (2026-08-12)

The raw CSV schema has only whole-task prompt/context fields and pen-down
stroke IDs. It has no character boundary, character label, or
character-to-stroke ownership field. Its 1,348 raw task files include
freehand, shape, wave, and topic tasks that do not have a character transcript.

Decision: **REJECTED FOR AIFlow Math Ink 1.0 CHARACTER CLASSIFIER.** The raw
archive remains retained for audit only; it must not enter tensor generation,
training, or evaluation. All identity and biometric restrictions remain in
force.
