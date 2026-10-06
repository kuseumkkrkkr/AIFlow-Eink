# AIFlow Math Ink v8 작가 적응 실험

## 판정

- 최종 adoption 상태: `AUTO_REJECTED`
- 원본 실행 report 상태: `WRITER_ADAPTATION_SHADOW_DIAGNOSTIC_PASS` (불변 실행 영수증이며, 별도 강화 판정이 이를 덮어쓴다.)
- 제품 승격: `false`
- 결론: 소량 작가 calibration의 집계 개선 가능성은 관측됐지만, legacy writer-LOO에서 작가별 회귀가 발생해 현재 adapter는 채택하지 않는다.
- 동결 유지: 128×5 입력, 372-class HWR checkpoint, runtime, CROHME, MathWriting, v7 산출물 모두 변경 0.

## 실험 계약

- legacy 7 writers에서만 설계와 하이퍼파라미터를 비교했다.
- 수식 단위 split: `SHA256(seed, writer_id, sample_id)` 순서로 calibration 2~6수식, 나머지 evaluation.
- 모든 작가에서 calibration/evaluation `sample_id` 교집합 0.
- evaluation label은 scoring에만 사용했다.
- 후보 보존: 각 glyph의 frozen baseline Top-5 안에서만 재정렬한다.
- known replay `writer_010/012`는 설정 동결 뒤 1회 진단에만 사용했으며 selection, tuning, promotion 근거가 아니다.

## Adapter 비교

- 소형 neural 우선안: 128→2→372 low-rank residual, 1,120 parameters, 60 epochs, AdamW, lr 0.01, weight decay 0.10.
- 최종 비교안: calibration 정답 클래스별 128차원 정규화 임베딩 centroid를 만들고 cosine similarity × 2.0을 frozen logit에 더하는 `prototype_200`.
- prototype은 evaluation마다 frozen Top-5 mask를 다시 적용하므로 Top-5 밖 라벨을 새로 끌어오지 않는다.
- neural은 legacy Top-1 90.33%→91.67%, 식 exact 65.75%→69.86%였지만 `writer_003`, `writer_004`가 회귀했다.
- 회귀 없는 약한 후보 `prototype_025`, `prior_010`은 정확도 개선이 0이었다.

## Legacy writer-LOO A/B

동일한 300 glyph / 73 evaluation formula 기준이다.

| 지표 | Frozen baseline | prototype_200 |
|---|---:|---:|
| Top-1 | 90.33% | 92.67% |
| Top-5 | 98.33% | 98.33% |
| Formula exact | 65.75% | 72.60% |
| Formula Top-5 oracle | 95.89% | 95.89% |
| Candidate violations | 0 | 0 |

작가별 안전 실패:

- `writer_004`: Top-1 87.10%→83.87%(-3.23%p), formula exact 50.00%→37.50%(-12.50%p).
- 따라서 집계 이득만으로 수락하지 않고 자동 거절한다.
- glyph paired 변화는 개선 8 / 회귀 1 / 총 Top-1 변경 9건이다. exact McNemar 이항 검정 p=0.0391이지만, 작가가 7명뿐이고 한 작가 안전 실패가 있어 상용 일반화 증거로 보지 않는다.

## Known replay 진단

동일한 163 glyph / 45 evaluation formula 기준이다.

| 지표 | Frozen baseline | prototype_200 |
|---|---:|---:|
| Top-1 | 84.66% | 86.50% |
| Top-5 | 94.48% | 94.48% |
| Formula exact | 57.78% | 60.00% |
| Formula Top-5 oracle | 84.44% | 84.44% |
| Candidate violations | 0 | 0 |

- `writer_010`: 4수식뿐이며 calibration 2 / evaluation 2, Top-1과 formula exact 모두 변화 없음. 일반화 해석 금지.
- `writer_012`: Top-1 85.71%→87.66%, formula exact 60.47%→62.79%.
- replay 작가별 회귀는 없지만 이미 알려진 진단셋이므로 상용 승격 근거가 아니다.
- glyph paired 변화는 개선 5 / 회귀 2 / 총 Top-1 변경 7건이며 exact McNemar p=0.4531로 통계적으로 유의하지 않다.

## 다음 연구 조건

- 현재 `prototype_200`은 배포하지 않는다.
- 다음 후보는 calibration-only 신뢰도에 따라 보정 강도를 낮추거나 baseline으로 abstain하는 작가별 안전 장치가 필요하다.
- 하이퍼파라미터를 새 fresh replay에 맞추지 않는다.
- 최종 승격은 새 작가와 새 수식을 사전 동결한 untouched acceptance 1회로만 판단한다.

## 산출물

- 재현 스크립트: `scripts/evaluate_writer_adaptation_v8.py`
- artifact: `artifacts/writer_adaptation_v8_20260823_r1_shadow`
- 원본 실행 report SHA-256: `59a49af476b43085b2d7ec060165e35b8b7d0064478eccae52f2d406469fbc47`
- 강화 판정: `artifacts/writer_adaptation_v8_20260823_r1_shadow/adoption_decision.json`
- split manifest SHA-256: `be5ff4a01bd06e2284a978126118dbb3532c1aba52bbf42978d70bce91063f1b`
