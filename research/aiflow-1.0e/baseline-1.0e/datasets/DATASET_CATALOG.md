# AIFlow Math Ink 1.0 dataset catalog

Last verified: 2026-08-20 (formula-context amendment)

| Folder | Dataset | Status | Permitted role | Local artifact |
|---|---|---|---|---|
| `00_project_owned/` | AIFlow consented public ink | Project-owned / active | Primary online HWR, ownership, formula-layout training, and project writer-holdout | Existing release data in `hf-dataset/data/` |
| `00_project_owned/generated_context/` | AIFlow generated homograph contexts | Project-owned / diagnostic only | Candidate-validity coverage research; not HWR or selected runtime training | 12,000-formula derivative and audit |
| `10_approved_external/uji_pen_characters_v2/` | UJI Pen Characters v2 | Approved external | Box-local isolated-character trajectory pretraining | Source ZIP and curated derivative |
| `10_approved_external/isgl_online_offline_hwr/` | ISGL | Approved external / restricted | Online English letters, digits, and word trajectory pretraining; training pool only | Original ZIP and inherited 7,414-row derivative |
| `10_approved_external/uci_character_trajectories/` | UCI Character Trajectories | Approved external / single writer | Lowercase Latin single-stroke representation pretraining; training pool only | Official source ZIP |
| `10_approved_external/hwrt/` | HWRT / Detexify | Approved external / curated | Box-local mathematical-symbol candidate pretraining; never model selection or final evaluation | Official TAR and filtered derivative |
| `10_approved_external/deepmind_mathematics_dataset/` | DeepMind Mathematics Dataset | Approved external / Apache-2.0 | Formula-context candidate-validity pretraining only | Pinned generator source, 80,000-formula derivative, and audit |
| `10_approved_external/bdshwa/` | BDSHWA | Rejected for 1.0 character classifier / raw retained | No classifier, tensor, or evaluation role | Original Mendeley ZIP retained as audit evidence |
| `20_reaudit_required/hf_cli_unlicensed/` | Three `newbienewbie` trajectory repos | Rejected | Audit evidence only | D-pinned Hugging Face CLI downloads |
| `30_noncommercial_evaluation/` | CROHME/MathWriting-class data | Research-only | Frozen external regression evaluation only | Never copy into product-training folders |

Approval means that a dataset may enter the stated 1.0 training pool. It does not mean that the current 0.9 checkpoint has already been retrained on it. See `UTILIZED_DATASETS.md` for that distinction.

## Integrity

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `10_approved_external/uji_pen_characters_v2/raw/uji_pen_v2.zip` | - | `0881B522911B99D9922820289441B50FD3D307F71CD7F9CC70E86872424A5F90` |
| `10_approved_external/uji_pen_characters_v2/derived/uji_math_curated.jsonl.gz` | - | `CDA2FE17C213BC8C90ED93EC8998658DDAF9D7871C75ADF521798E6E3BA3029B` |
| `10_approved_external/isgl_online_offline_hwr/raw/isgl_source.zip` | 974,053,582 | `A94F2473246222F9470D4B93B68CFBC756ECB4865742DB8164788359FE511693` |
| `10_approved_external/isgl_online_offline_hwr/migrated/isgl_online.jsonl.gz` | - | `0A4BD4FE37CD7656B2FD3586FED4D5C886FEBA1D8E5120803B60A0F04044988D` |
| `10_approved_external/uci_character_trajectories/raw/character+trajectories.official.zip` | 7,915,950 | `5D2DB017EF0D8CF0E65ED060C9E90399F78EB9F1E3CB63E22CA8C3EF4BA67D52` |
| `10_approved_external/hwrt/raw/2015-01-28-data.tar` | 140,790,596 | `B96FEAFD71B01F1623997DFF3CC8AC4D18628D128CEE1B3DF1880518BBA3EA4A` |
| `10_approved_external/hwrt/derived/train.jsonl.gz` | 153,885,264 | `4C067A06FFAB8A73FE99C10B633B1175A68E512D17CA68D587F56E6182AD2561` |
| `10_approved_external/hwrt/derived/validation.jsonl.gz` | 496,546 | `9813260509A4025E203CD36A3AE08C8A6CBC342233684C12C13B2C447D8B0A34` |
| `10_approved_external/hwrt/derived/test.jsonl.gz` | 359,995 | `902EAAA1311708D4A73F1E63A9C88AE15CEDDED945E7B915B7CE177DAD30AF8A` |
| `10_approved_external/hwrt/derived/rejections.jsonl.gz` | 1,420 | `47B0968D8387E10DED1413F74D46B504FA560F4F6CBD4CBE81ED7D859AFB6644` |
| `10_approved_external/bdshwa/raw/bdshwa_v1.zip` | 1,318,392,171 | `45FBBDCD1A9C4353F5C93BF8096FBEE9723931185F7BBFED3A86E21BA4CF20EA` |
| `10_approved_external/deepmind_mathematics_dataset/derived/formula_context_v1.jsonl.gz` | 2,756,487 | `1998779E46E693DB2452C2A677DC0D366D4252BE49CA27E3974B97DF6FCC9BC5` |
| `00_project_owned/generated_context/candidate_validity_coverage_v1.jsonl.gz` | 339,188 | `69B38D30E4A432FA2FB41DDD4EF2ECECCAB888992C378EF7904F9DEA35F4F79C` |

## Admission rules

1. Only `10_approved_external` and consented `00_project_owned` records may enter a product-training manifest.
2. In the current Math Ink path, external corpora train only the role named above. They cannot supervise formula grouping, stroke ownership, spatial relations, or the final decision layer. A future general text-HWR branch is a separate model scope.
3. UCI is explicitly one-writer training data. ISGL's inherited derivative is training-only until rebuilt from the complete source with an exclusion manifest.
4. HWRT timestamps are relative in the derivative, time-reversing samples and normalized duplicates are removed, and browser user agents/raw user IDs are absent. Its source user IDs do not prove writer identity, so HWRT cannot select or evaluate models.
5. BDSHWA is excluded from the 1.0 character-classifier corpus because it lacks character boundaries and character-to-stroke ownership. Its demographic, writer-identification, age, gender, and biometric targets/metadata remain excluded in all cases.
6. Every release must carry the source attribution and applicable CC BY 4.0 or ODbL notices. Model adoption still requires project-owned writer-holdout and non-regression gates.
7. The DeepMind formula derivative is sequence-context data only. It cannot train pen trajectories, shape HWR, grouping, ownership, or timing. Generated homograph coverage contexts remain diagnostic because their r2/r3 writer-OOF transfer gate failed.
