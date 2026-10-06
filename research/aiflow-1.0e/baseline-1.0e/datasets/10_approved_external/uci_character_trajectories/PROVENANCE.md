# UCI Character Trajectories provenance and approved use

Audited 2026-08-10 from the official UCI download. Provenance, archive integrity, numeric structure, and CC BY 4.0 terms pass. The project owner explicitly accepts the corpus as one-writer training data.

## Source and integrity

- Source: UCI Machine Learning Repository, dataset 175
- URL: https://archive.ics.uci.edu/dataset/175/character+trajectories
- Licence: CC BY 4.0
- Official local source: `raw/character+trajectories.official.zip`
- Bytes: `7915950`
- SHA-256: `5D2DB017EF0D8CF0E65ED060C9E90399F78EB9F1E3CB63E22CA8C3EF4BA67D52`
- ZIP CRC: passed.
- The official download is byte-identical to the inherited `raw/character_trajectories.zip`.

## Observed structure

- `mixoutALL_shifted.mat`: 2,858 arrays, each shaped `3 x T` for X, Y, and pen-tip force.
- `T` ranges from 109 to 205; all values are finite.
- 20 lowercase Latin classes: `a b c d e g h l m n o p q r s u v w y z`.
- All samples come from one writer and retain only one pen-down segment.
- Sampling interval is documented as 0.005 seconds (200 Hz).
- The released values were numerically differentiated, Gaussian-smoothed, normalized, and time-shifted. They are not raw device coordinates.

## Known limitations and adopted scope

1. Treat all 2,858 samples as one writer and training-only; never derive validation or model-selection claims from this corpus.
2. Single pen-down samples cannot supervise multi-stroke grouping, ownership, or formula layout.
3. The released signals are differentiated, smoothed, normalized, and shifted rather than raw device coordinates.
4. Use only for lowercase Latin single-stroke representation pretraining.

## Operating rules

- Register CC BY 4.0 attribution.
- Assign the entire corpus to training under one writer group.
- Evaluate any resulting checkpoint on a held-out project-owned writer split and retain only if all character, ownership, relation, and end-to-end formula gates are non-regressive.
- Never use it to train ownership, grouping, relation, or final-decision components.

Decision: **APPROVED EXTERNAL / SINGLE WRITER - representation pretraining only.**
