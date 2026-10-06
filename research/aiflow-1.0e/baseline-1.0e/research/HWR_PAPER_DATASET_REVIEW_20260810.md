# HWR paper dataset review and stop decision - 2026-08-10

Paper metadata and markdown were inspected with Hugging Face CLI 1.24.0. The review stopped after seven representative 2022-2025 online-HWR/HMER papers because later papers reused the same public online-ink families, introduced noncommercial data, described unavailable/unlicensed private corpora, or changed representation rather than source-data rights.

## Reviewed papers

| Paper | Data used or introduced | Dataset consequence |
|---|---|---|
| [Representing Online Handwriting for Recognition in Large Vision-Language Models](https://arxiv.org/abs/2402.15307) | DeepWriting, MathWriting, VNOnDB | Useful digital-ink representation work; public sources remain noncommercial/research-limited for this project. |
| [MathWriting](https://arxiv.org/abs/2404.10690) | 230k human and 400k synthetic online expressions | Strong HMER benchmark, but its noncommercial licence prevents 1.0 product training. |
| [Character Queries](https://arxiv.org/abs/2309.03072) | IAM-OnDB and HANDS-VNOnDB plus derived character segmentations | Adds segmentation labels, not new commercial rights; both upstream corpora remain blocked. |
| [A Transformer Based Handwriting Recognition System Jointly Using Online and Offline Features](https://arxiv.org/abs/2506.20255) | IAMOn-DB, VNOn-DB, ISI-Air | Reuses established text/air-writing corpora; no new commercially clear dataset class. |
| [InkSight](https://arxiv.org/abs/2402.05804) | DeepWriting, VNOnDB, SCUT-COUCH and derived offline-to-online ink | A derivation method, not a rights-clean raw online-ink source. Official VNOnDB terms are research-only despite the paper table's MIT label. |
| [A Transformer Architecture for Online Gesture Recognition of Mathematical Expressions](https://arxiv.org/abs/2211.02643) | Author-collected corpus: 455 subjects, 21,752 glyphs, 27,477 strokes, >700k touches | Highly relevant schema, but no public download, repository, or licence was located. |
| [Online Gesture Recognition using Transformer and Natural Language Processing](https://arxiv.org/abs/2305.03407) | Expanded author corpus: >600 subjects, 69,278 glyphs, 93,330 strokes, >2M touches; about 100k generated sentences | Still paper-described without a verifiable public artefact/licence. |

## Official-rights checks

- [HANDS-VNOnDB](https://tc11.cvc.uab.es/datasets/HANDS-VNOnDB_1/) is structurally close to project ink (InkML, traces, writer IDs, transcriptions), but the official page limits ordinary access to research and directs commercial users to contact the owners.
- IAM-OnDB is noncommercial/research-oriented.
- DeepWriting uses CC BY-NC-SA plus non-distribution terms.
- MathWriting and CROHME are noncommercial research sources.
- InkSight-generated trajectories inherit source-dataset restrictions and do not cure provenance or consent.

## Saturation decision

No reviewed paper introduced a downloadable dataset that simultaneously provides ordered online strokes, writer-aware splits, labels relevant to mathematical HWR, documented participant/privacy provenance, and clear commercial rights. Further paper searching was therefore stopped. Resume only if a paper links a new primary dataset with verifiable files and commercial terms, or if the project obtains a direct commercial agreement for VNOnDB/CASIA/TUAT-class data.
