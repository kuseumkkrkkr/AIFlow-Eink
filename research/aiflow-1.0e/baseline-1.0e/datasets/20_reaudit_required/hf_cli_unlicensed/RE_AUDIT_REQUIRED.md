# Hugging Face CLI trajectory candidates: rejected

Searched and downloaded with Hugging Face Hub CLI 1.24.0 on 2026-08-10. `HF_HOME` and all downloads were pinned to this D: workspace. Hub cache verification passed for all repository files, but none of these repositories supplies a licence, provenance, collection/consent statement, or writer grouping needed for writer-disjoint evaluation.

| Repository | Revision | Rows (train/val/test) | Critical finding |
|---|---|---:|---|
| `newbienewbie/handwriting_strokes` | `8f9dbef771a25b7bb11bf0385fed9a49b1f93235` | 920 / 100 / 100 | 98 of 100 validation rows also occur in test; there are 20 train/test and 22 train/validation overlaps. |
| `newbienewbie/handwriting_strokes_2strokes` | `6e2173aa7f93467850b5cffced414057bf242454` | 894 / 310 / 310 | Validation and test Parquet files are byte-identical; each has 309 unique rows and one duplicate. |
| `newbienewbie/handwriting_edge_case_in_one_stroke` | `c71311ea94b88af0d0e8c580c6bafefa13ae1640` | 858 / 22 / 157 | 16 train/validation and 45 train/test exact row overlaps. |

## Schema audit

- X, Y, and timestamp sequence lengths match in every row; no empty sequence or timestamp reversal was found.
- `traveledDistances` length disagrees with X length in all 1,120 rows of `handwriting_strokes`, 1,492 of 1,514 rows of `handwriting_strokes_2strokes`, and 970 of 1,037 rows of the edge-case set.
- Dataset cards contain generated schema/split metadata only. `card_data` is empty and no licence tag, source description, writer key, or consent statement exists.

## Integrity hashes

- `handwriting_strokes`: train `E8E11649...22CEC`, validation `A5E4AC5F...8CAED`, test `21297408...7E972`.
- `handwriting_strokes_2strokes`: train `A90988B1...F8512`; validation and test both `C83EED7E...6DE14`.
- `handwriting_edge_case_in_one_stroke`: train `A331C2A3...7AC15`, validation `EE742E79...3F1E0`, test `22086968...C85E`.

Decision: **REJECTED FOR COMMERCIAL RESEARCH AND ALL MODEL SELECTION.** Local copies remain quarantined only as audit evidence.
