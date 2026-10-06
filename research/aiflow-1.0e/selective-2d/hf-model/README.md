---
language:
- ko
- en
library_name: pytorch
license: other
pipeline_tag: feature-extraction
base_model: cwLeeDev/aiflow-math-ink-06-intermediate
tags:
- online-handwriting
- mathematical-expression-recognition
- trajectory
- research
---

# AIFlow Math Ink 0.9

AIFlow Math Ink 0.6 seed17 base/online adapter에 결합하는 공개 데이터 재학습 additive calibrator 연구 후보입니다.

## 체크포인트 계약

- 파일: `normalized_balanced_calibrator.pt`
- SHA-256: `bf3bc80c07315b812e2419ffbddadb0ecb3bd5b73e00799fc271979a0124c2dd`
- 입력: glyph bbox crop, aspect-preserving letterbox, local character context
- 출력: 기존 378 labels + `=`
- 필압 사용: 아니오
- 필수 base: `cwLeeDev/aiflow-math-ink-06-intermediate` seed17의 `base_378.pt`, `online_adapter.pt`

이 파일만으로 추론할 수 없습니다. base encoder embedding에 additive correction을 적용해야 합니다.

## 필기 replay

| 지표 | baseline | seed31 | guarded seed31 |
|---|---:|---:|---:|
| 문자 Top-1 | 50.29% | 87.72% | 87.72% |
| 문자 Top-5 | 70.18% | 98.83% | 98.83% |
| 수식 exact | 26.53% | 67.35% | 67.35% |
| 수식 Top-5 oracle | 42.86% | 95.92% | 95.92% |
| 개선/회귀 문자 | - | 67/3 | 66/2 |

Guard는 public-train support가 3개 미만인 label에서 baseline Top-1 confidence가 0.95 이상이면 baseline을 보존합니다. 같은 replay에서 확인된 규칙이므로 독립 제품 검증 결과는 아닙니다.

## CROHME2019 비상업 검증

Kaggle `ntcuong2103/crohme2019`의 CC BY-NC 4.0 valid split에서 parse 가능한 985식을 평가했으며 학습에는 사용하지 않았습니다.

| 지표 | baseline | calibrator | guarded |
|---|---:|---:|---:|
| truth-group 문자 Top-1 | 56.96% | 74.11% | 74.18% |
| truth-group 문자 Top-5 | 79.97% | 92.45% | 92.28% |
| end-to-end group-token Top-1 recall | 48.20% | 62.17% | 62.22% |
| fully-covered 수식 exact | 7.95% | 11.98% | 12.10% |

0.6 geometry gridder의 partition exact는 40.81%, exact group recall은 82.48%, lattice group ceiling은 99.86%였습니다.

## 제한

- 연구 후보이며 `product_adopted=false`입니다.
- 필기 replay의 contributor ID가 없어 writer-disjoint를 증명하지 못했습니다.
- CROHME2019은 CC BY-NC 연구 평가이며 제품 성능 근거가 아닙니다.
- 이후 비상업 gridder artifact가 제거되어 공개 가능한 0.6 geometry gridder를 평가했습니다.
- 결정 계층의 수학적 동치 정확도는 이번 실행에서 측정하지 않았습니다.

상세 수치는 `metrics.json`과 `crohme2019_valid.json`을 참조하십시오.
