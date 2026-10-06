# AIFlow Math Ink 1.0 긴 수식 그룹핑 개선 결과

## 결론

- 원인: 후보 lattice 누락보다 **짧은 수식에서 학습한 그룹 점수가 긴 수식에서 누적되며 발생한 과병합**이 주 병목이었다.
- 조치: 소유 획만 재배치한 긴 2D 수식 224개를 만들고, 기존 모델 확률 25%와 증강 모델 확률 75%를 혼합했다.
- 안전 경계: 기존 런타임은 16획 미만에서 그대로 유지하고, 16획 이상에서만 혼합 모델을 사용한다.
- 최종 CROHME grouping exact: **300/769(39.01%) → 314/769(40.83%), +14식, +1.82%p**.
- strict 식 exact: **50/769(6.50%) → 50/769(6.50%)**, 변화 없음.
- 판정: 개선은 실재하지만 상용 채택 폭은 아니다. `r11`은 **shadow-only**이며 기본 런타임은 교체하지 않는다.

## 원인 분해

1. 소유 96식의 lattice truth-partition 복원율은 100%였다. 후보 자체가 빠지는 문제가 주원인은 아니었다.
2. 기존 geometry 모델의 writer-disjoint grouping exact는 92.71%였지만, 긴 2D 소유-획 합성식에서는 51.79%였다.
3. 긴 합성식의 과병합률은 39.73%였다. 독립 후보 확률의 작은 오류가 문자 수만큼 누적되며 전체 partition exact가 급락했다.
4. 96식 전체 모델을 짧은 식까지 전면 적용하면 미사용 writer 및 canonical 식에서 과분할이 증가했다.
5. 데이터셋 버전 사이에서 같은 `sample_id`가 다른 원시 수식을 가리키는 사례가 있었다. 이후 비교 계약은 dataset SHA와 원시 획 해시를 기준으로 해야 한다.

## 개발 데이터와 선택 프로토콜

- 실제 grouping truth: 프로젝트 소유 96식, 7 writers.
- 긴 2D 합성: 224식, writer당 32식.
- 레이아웃: horizontal, multiline, scripted, stacked.
- 외부 문자나 CROHME 획은 합성에 사용하지 않았다.
- outer held writer의 실제식과 합성식은 해당 writer를 전혀 사용하지 않은 모델로만 평가했다.
- 모델·bias·16획 경계 선택에 CROHME 및 미사용 53식은 사용하지 않았다.

| Writer-disjoint 개발 지표 | 기존 | 16획 이상 보수 혼합 |
|---|---:|---:|
| 실제 96식 partition exact | 92.71% | 92.71% |
| 실제 96식 pair-F1 | 95.32% | 95.32% |
| 긴 2D 합성 partition exact | 51.79% | 85.27% |
| 긴 2D 합성 pair-F1 | 91.67% | 98.29% |
| 긴 2D 합성 과병합 | 39.73% | 5.36% |
| 긴 2D 합성 과분할 | 20.98% | 10.27% |

## 탈락시킨 후보

- `r9` hard switch: 동결된 미사용 53식에서 38/53(71.70%) → 35/53(66.04%), 3식 순회귀. 탈락 후 이 53식은 재선택에 사용하지 않았다.
- `r10` 8획 이상 전구간 blend: 기존 110식 출력 5건 변경. canonical 식에서 글자 묶음을 쪼개는 변화가 확인되어 탈락했다.
- `r11`: 16획 미만에서 기존 grouping 모델을 그대로 보존했다. 기존 110식 mismatch 0건으로 통과했다.

## 최종 1회 CROHME 검증

검증은 formula-complete 이후에만 수행했고 autograd, 학습, selection, threshold tuning은 모두 0이다.

| 입력 획 수 | N | 기존 exact | r11 exact | 개선 | 회귀 |
|---:|---:|---:|---:|---:|---:|
| 1–7 | 248 | 61.29% | 61.29% | 0 | 0 |
| 8–15 | 273 | 34.80% | 34.80% | 0 | 0 |
| 16–23 | 167 | 28.74% | 34.13% | 18 | 9 |
| 24–31 | 55 | 7.27% | 18.18% | 6 | 0 |
| 32+ | 26 | 3.85% | 0.00% | 0 | 1 |

- 전체 grouping 개선/회귀/순증: **24 / 10 / +14식**.
- 16획 이상 과병합 proxy: **127 → 87(-40)**.
- 16획 이상 과분할 proxy: **48 → 69(+21)**.
- 16획 이상 동일 그룹 수이지만 잘못된 partition: **20 → 25(+5)**.
- layout order exact: **167/769(21.72%) → 169/769(21.98%)**.
- flat token sequence exact: **59/769(7.67%) → 59/769(7.67%)**.
- relation micro-F1: **79.97% → 79.77%**.
- strict group+layout+relation+character exact: **50/769(6.50%) → 50/769(6.50%)**.

합성식은 긴 식의 과병합을 재현하고 줄이는 데는 유효했다. 그러나 실제 2D 필기의 결합 패턴을 충분히 대표하지 못해 일부 오류가 과분할로 이동했다. 새 grouping이 맞아진 식에서도 HWR·문맥·2D layout이 동시에 맞지 않아 strict 식 정확도 증가는 없었다.

## HWR Top-5 비회귀

- HWR checkpoint SHA-256: `04f8608aebcf6c02d45ad6f5735229b9eaa2c4b4e1be0db4793d02273ef2d00e`.
- grouping 작업 전후 checkpoint가 동일하며 HWR·context 파라미터를 수정하지 않았다.
- 기존 truth-group 기준: Top-1 **72.6962%**, Top-5 **93.7536%**, 정답이 Top-5 밖 **749/11,991**.
- 749건 중 상업 이용 가능한 학습 support 0건인 행은 0이다. 원인은 미학습 vocabulary가 아니라 shape/domain 일반화 및 클래스 혼동이다.

## 무결성 및 학습 경계

- 기존 110식 projection mismatch: 0.
- 완료 전 확정: 0.
- 획 손실·중복: 0.
- raw fallback 불일치: 0.
- product auto commit: 0.
- 최종 CROHME 769식 raw fallback 보존: 769.
- training boundary audit: trainers 16, corpus builders 2, failed 0, selection finding 0.
- `r11` SHA-256: `69925b478d2170bafcf2a12576ee07f6372b5318311fa53b3d12905c5d9362bf`.

## 다음 개선에 필요한 데이터

동일 검증셋에 다시 맞추지 않는다. 다음 루프에는 서로 분리된 두 묶음이 필요하다.

1. 학습용: 프로젝트 소유 또는 상업 이용 가능한 **16획 이상 실제 2D 수식 160식 이상**, writer 8명 이상, writer당 최대 20식.
2. 미사용 acceptance: 학습 writer와 분리된 **160식 이상**. 관측 exact 90%와 Wilson 95% 하한 85%를 동시에 통과해야 한다.
3. 각 식은 exact stroke partition, stroke order/timestamp, formula-complete 경계, 2D relation label, immutable raw ink를 보유해야 한다.
4. 분포에는 지연해서 찍는 점·가로획, 위첨자/아래첨자, 분수, 중첩 괄호, 다중 행을 포함해야 한다.
5. writer 군집효과가 있으면 160식은 하한일 뿐이며 ICC에 따라 표본 수를 늘린다.

## 주요 산출물

- 동결 후보: `artifacts/project_owned_grouping_runtime_20260822_r11_conservative_long_blend_shadow/partition_context_ranker.joblib`
- 110식 비회귀: `artifacts/formula_complete_owned_nonregression_20260822_r5_conservative_long_blend/nonregression_report.json`
- 최종 CROHME: `artifacts/formula_complete_raw_crohme_20260822_r3_grouping_r11_final/validation_report.json`
- 사후 delta: `artifacts/final_grouping_delta_20260822_r1/grouping_delta_analysis.json`
- 학습 경계: `artifacts/training_boundary_audit_20260822_r5_grouping_final/training_boundary_audit.json`
