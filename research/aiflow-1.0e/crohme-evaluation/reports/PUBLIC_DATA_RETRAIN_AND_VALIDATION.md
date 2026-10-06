# 공개 데이터 재학습 및 고도화 검증 보고서

## 결론

- 공개 JSONL 재학습은 기존 seed31 결과를 정확히 재현했다.
- 희소 label 보정을 일괄 제한하는 `min support=3` 후보는 Top-1 77.78%, 수식 exact 44.90%로 하락해 폐기했다.
- 기존 0.95 고신뢰 원칙을 이용한 baseline 보존 guard는 3개 seed 모두에서 집계 정확도를 유지하면서 replay 회귀를 3건→2건으로 줄였다.
- CROHME2019에서도 guarded 후보는 Top-1과 수식 exact가 소폭 증가했지만 Top-5는 소폭 하락했다. 독립 threshold 선택이 아니므로 연구용 shadow 정책으로만 채택한다.
- 가장 큰 잔여 병목은 CROHME2019 partition exact 40.81%인 그리딩 ownership 선택이다. lattice group ceiling 99.86%와 차이가 크다.

## 데이터

- 공개 수식: 119건
- valid/pending/reject: 110/4/5
- writer/session 별칭: 8/12
- ownership 학습: 47식, 211문자, 25 label, 3 writer
- 내부 필기 replay: 49식, 171문자
- CROHME2019 valid: 985 parse 가능 식, 9,997 truth groups

## 3-seed 필기 replay

| seed | guarded Top-1 | guarded Top-5 | 수식 exact | Top-5 oracle | 개선 | 회귀 |
|---:|---:|---:|---:|---:|---:|---:|
| 17 | 85.96% | 98.25% | 67.35% | 93.88% | 63 | 2 |
| 31 | 87.72% | 98.83% | 67.35% | 95.92% | 66 | 2 |
| 47 | 85.96% | 98.25% | 65.31% | 93.88% | 63 | 2 |
| 평균 | 86.55% | 98.44% | 66.67% | 94.56% | - | - |

Baseline은 Top-1 50.29%, Top-5 70.18%, 수식 exact 26.53%, Top-5 oracle 42.86%다.

## CROHME2019 CC BY-NC valid

| 지표 | baseline | direct | guarded |
|---|---:|---:|---:|
| partition exact | 40.81% | 동일 | 동일 |
| exact group recall | 82.48% | 동일 | 동일 |
| truth-group 문자 Top-1 | 56.96% | 74.11% | 74.18% |
| truth-group 문자 Top-5 | 79.97% | 92.45% | 92.28% |
| group-token Top-1 recall | 48.20% | 62.17% | 62.22% |
| fully-covered 수식 exact | 7.95% | 11.98% | 12.10% |

## 다음 고도화 우선순위

1. public collection에 formula ownership 검수를 추가해 현재 47식을 확대한다.
2. writer별 최소 label support를 확보한 뒤 writer-disjoint split을 고정한다.
3. gridder는 새 후보 생성보다 99.86% lattice 안에서 올바른 partition을 고르는 selector를 개선한다.
4. `f`, `g`, `/`, `\sqrt{}`를 실제 online stroke로 우선 수집한다.
5. guard threshold는 새 writer holdout에서 한 번만 선택하고 기존 replay는 최종 확인에만 사용한다.

## 채택

- 공개 데이터셋: 채택
- 공개 데이터 재학습 경로: 채택
- seed31 checkpoint: 연구 채택
- 0.95 guard: shadow 채택
- `min support=3` 보정 제한: 폐기
- 제품 채택: 보류
