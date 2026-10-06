# AIFlow Math Ink v9 calibration-only writer policy 최종 판정

## 결론

- 최종 상태는 **`PRE_ACCEPTANCE_FAIL_CLOSED`**다.
- frozen 128×5·372-class HWR, checkpoint, runtime과 각 행의 baseline Top-5 집합은 변경하지 않았다.
- Legacy 7작가 formula-disjoint 평가에서 모든 작가는 calibration 증거 부족으로 `identity`를 선택했다.
- 따라서 작가별 비회귀와 candidate 보존은 통과했지만, 집계 양의 개선이 0이라 acceptance 가능 상태로 승격하지 않았다.
- known replay, fresh acceptance, CROHME, MathWriting은 열지 않았고 제품 승격도 하지 않았다.

## 단계별 결과

| 단계 | Baseline Top-1 | v9 Top-1 | 식 지표 | 판정 |
|---|---:|---:|---:|---|
| old synthetic dev 4 styles | 68.47% | 75.75% | pseudo exact 8.13%→14.38% | 개발 통과 |
| consumed synthetic diagnostic 16 styles | 65.24% | 72.69% | pseudo exact 6.48%→11.25% | 진단 통과 |
| untouched synthetic reserve 16 styles | 65.18% | 72.75% | pseudo exact 6.88%→12.11% | one-shot 통과 |
| Legacy 7 writers, 73 actual formulae | 90.33% | 90.33% | actual exact 65.75%→65.75% | **양의 개선 없음** |

Synthetic 결과는 external parent manifold가 약 98.8% 겹치는 style 변화 증거일 뿐 실제 작가 일반화나 상용 승격 증거가 아니다.

## Legacy calibration 정책

- calibration/evaluation formula: 23/73개
- calibration/evaluation glyph: 92/300개
- 작가: `writer_001`, `002`, `003`, `004`, `006`, `007`, `008`
- eval label은 prediction 생성 뒤 scoring에만 사용했다.
- predictor API는 `embeddings`, `logits`, `prototypes`, `admitted`, `alpha`만 받으며 label·metadata를 받지 않는다.
- global prototype score는 calibration에서 paired improvement가 1개 이상이고 regression이 0이며 Top-1·formula exact가 모두 비회귀일 때만 작가별로 활성화한다.

활성 작가는 0/7명이었다. calibration에서 writer 001/002/003/004/006은 각각 1/2/1/1/1개 정답을 잃었고, writer 008은 wrong-to-wrong 변경 1개, writer 007은 무변화였다. 이 때문에 전부 identity로 후퇴했다.

## 최종 gate

| Gate | 결과 |
|---|---|
| 7작가 formula-disjoint | 통과 |
| evaluation labels scoring-only | 통과 |
| 작가별 Top-1·actual formula exact 비회귀 | 통과 |
| adapted Top-5 set == frozen Top-5 set | 통과, mismatch 0 |
| candidate violation | 0 |
| 집계 최소 한 지표 양의 개선 | **실패** |
| known replay/fresh/CROHME/MathWriting unopened | 통과 |
| checkpoint/runtime/HWR unchanged | 통과 |

## 주요 증거와 SHA-256

- Legacy report: `artifacts/global_prototype_legacy_policy_v9_20260823_r2/report.json` — `bfec56507994d3593ec794f1711c2c061e4e4f54d13cc9dfbc0fcd78f9fb595a`
- decision receipt — `8df07438963bd31f53d59edf2ee9a0f1df3b50409ed2b5ec67e68022d770a88d`
- Legacy cache — `33cecbbacc1663235b9a454ab32cf50d59f49c3b9d8fd405f8e3b6920ea089d3`
- split manifest — `c41343220de612855fe0b8708f239132c33e296ff628f394e5f8a7c72d979601`
- calibration policy — `0c9cb1db91a13435ef5d5da333e675cacc4e34b668caa411fc429b847fd0f18b`
- independent rejection audit — `2c9341cff23c06d0299498e6bf42b740f0ce49a8f91cc3a8a0c85defe20e6dd1`

## 후속 경계

Legacy evaluation 300 glyph는 소비됐으므로 다음 후보 선택이나 하이퍼파라미터 튜닝에 다시 사용하지 않는다. 다음 연구는 기존 synthetic 64 styles와 Legacy calibration 92 glyph만 개발에 사용하고, label/writer ID 없는 sparse-calibration style inference를 새 독립 synthetic styles에서 먼저 검증해야 한다. 실제 상용 acceptance에는 새로 수집·사전 동결한 real writer/formula 세트가 필요하다.
