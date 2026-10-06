# AIFlow 1.0e 정확도 상승 연구 루프 결과

## 판정

- 이번 루프의 외부 가중치 기반 후보는 채택하지 않음.
- 제품 런타임, 기존 5-channel ordered online-HWR, Hugging Face 공개본은 변경하지 않음.
- 외부 TrOCR는 학습 soft target으로만 사용했고, online student 런타임에는 외부 신경망·래스터 입력이 없음.
- 현재 상용 후보는 기존 v4 stability shadow 상태를 유지하며, fresh writer-disjoint acceptance 전까지 production adoption 금지.

## 기준선

| 항목 | 기존 v4 stability | 평가 수 |
|---|---:|---:|
| Top-1 | 317/387 = 81.91% | 387 rows |
| 후보 recall / Top-5 | 374/387 = 96.64% | 387 rows |
| Formula exact | 49/95 = 51.58% | 95 formulas |
| Row-level regression | 0 | baseline |

## 실행 결과

### 1. LoRA candidate ranking

- 외부 TrOCR 마지막 2개 encoder block의 query/value에 rank-4 LoRA 적용.
- LoRA parameter: 12,288개.
- Raw: Top-1 323/387 = 83.46%, Formula exact 52/95 = 54.74%.
- 그러나 row-level regression 34건.
- fixed fail-closed guard: 변경 0건, 기준선과 동일.
- 판정: 기각.

### 2. Raster-free online candidate distillation

- frozen v4 HWR embedding + 후보/context feature를 사용하는 78,833-parameter online student 학습.
- 외부 teacher score는 학습 중 KL soft target으로만 사용.
- 3 epochs, learning rate 3e-4, temperature 2.0, 7-writer outer LOO.
- Raw: Top-1 331/387 = 85.53%, Formula exact 52/95 = 54.74%, regression 22건.
- fixed guard (.30 confidence / .30 margin): Top-1 318/387 = 82.17%, Formula exact 49/95 = 51.58%, regression 1건.
- nested writer guard: Top-1 316/387 = 81.65%, Formula exact 48/95 = 50.53%, regression 1건.
- 판정: 회귀 0 gate 실패로 기각.

## 해석

- 외부 teacher의 raw gain은 확인됐지만, 현재 데이터 규모와 writer 분할에서는 안전한 online-only student로 이전되지 않았다.
- nested guard가 held writer에서 회귀를 막지 못했으므로 해당 임계값 정책을 상용 runtime에 넣을 근거가 없다.
- 기존 HWR 후보 계약은 유지됐다. 후보 생성 0건, candidate contract violation 0건.
- full raster teacher의 fixed guard는 별도 shadow에서 334/387 Top-1, 59/95 Formula exact, regression 0으로 관측됐지만 post-hoc threshold 결과이며 fresh acceptance·단말 검증 전이므로 상용 채택 근거로 사용하지 않는다.

## 다음 정확도 상승 순서

1. 새 writer를 추가한 acceptance split을 먼저 확보하고, 동일한 7-writer LOO를 반복한다.
2. student가 teacher의 Top-1 하나만 모방하지 않도록 후보 전체 분포와 rank-aware margin을 distill한다.
3. formula-level exact loss를 별도 추가하되, row-level regression 0 gate를 유지한다.
4. 통과 후보만 Android/실서비스 지연·메모리·fallback 검증으로 이동한다.

이번 단계의 결론은 “정확도 상승 가능성은 있으나, 현재 online student는 상용 채택 불가”이다.

