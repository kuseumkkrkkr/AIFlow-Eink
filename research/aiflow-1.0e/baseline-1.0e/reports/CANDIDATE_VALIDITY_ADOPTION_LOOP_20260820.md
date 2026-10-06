# 후보 적합성 문맥모델 채택 강화 루프

## 결론

- **상용 기본 채택 보류, shadow 유지**
- fresh 53식에서는 큰 개선을 확인했으나 `|·O·o` 정답이 0건이라 알려진 희소 실패군을 검증하지 못했다.
- 선택 후 CROHME 진단도 Top-1·식 exact·strict macro 세 기준을 모두 통과하지 못했다.
- 따라서 `adopted manifest`는 생성하지 않았다.

## 데이터와 선택 절차

- 수집기 갱신: `Listed=164, Downloaded=0, Total=164`; 새 유입 0건
- 과거 phone replay ownership 복구: 49/49건 SHA-256(session)+prompt 1:1 대응, 미매칭 0건
- 모델 선택 전에 fresh acceptance 동결
  - 미사용 writer 2명, 53식, 186문자
  - 다문자 36식, 단일문자 17식
  - broad/prompt 학습 exact token-sequence 중복 1식 제외 후 중복 0
  - 7개 확대 검수 페이지 전체 확인, correction 0, exclusion 0
  - 모델 예측 사전 열람 0
- 검수 주체는 `Codex independent visual review`이며 사람 검수로 표기하지 않았다.
- 모델 선택에는 nested writer/token-sequence OOF만 사용했다. fresh acceptance와 CROHME는 선택 고정 후 처음 열었다.

## 강화 변형 선택

| 후보 | 전체 Top-1 | 식 exact | strict macro | 새-writer Top-1 | 새-writer 식 exact | 새-writer strict | 판정 |
|---|---:|---:|---:|---:|---:|---:|---|
| 기존 r1 | 88.63% | 70.53% | 89.15% | 88.37% | 72.73% | 97.50% | **선택** |
| v1 coverage LR 1e-6 | 89.15% | 70.53% | 90.26% | 87.79% | 70.45% | 97.50% | 새-writer 회귀 |
| v2 coverage LR 3e-6 | 89.15% | 70.53% | 90.26% | 88.37% | 70.45% | 97.50% | 새-writer 식 회귀 |
| v3 LR 1e-6 + replay 5 | 89.15% | 69.47% | 89.99% | 88.95% | 72.73% | 97.50% | 전체 식 회귀 |

세 강화 변형 중 완전 비회귀 후보가 없어 기존 r1을 유지했다. 선택 체크포인트 SHA-256은 `ee0033fb42f59b09f3f130300fb5374710cfef010a71c94dbd48e4d1c93dc8b9`이다.

## Fresh acceptance 결과

| 지표 | Frozen HWR baseline | 선택 문맥모델 | 변화 |
|---|---:|---:|---:|
| 문자 Top-1 | 77.42% | **91.40%** | +13.98%p |
| 식 exact | 50.94% | **75.47%** | +24.53%p |
| strict homograph macro | 58.33% | **85.25%** | +26.92%p |
| baseline 정답의 동형문자 회귀 | - | **0건** | 통과 |

후보 밖 출력 0건, grouping mutation 0건, 동일 체크포인트 반복 예측 불일치 0건이다. 다만 이 53식의 `|·O·o` 정답은 각각 0건이므로 strict 개선을 해당 세 문자까지 일반화할 수 없다. writer별 식 수도 4/49로 불균형하다.

## 선택 후 CROHME 진단

| 지표 | 실제 | 채택 기준 | 부족 |
|---|---:|---:|---:|
| Top-1 | 77.45% | 79.49% | -2.03%p |
| 식 exact | 21.75% | 27.44% | -5.69%p |
| strict macro | 55.42% | 61.42% | -6.00%p |

CROHME는 모델 선택에는 쓰지 않았으며, 고정 후 사후 진단으로만 사용했다.

## 다음 채택 루프의 필수 수집

- 모델을 더 수정하면 현재 53식 결과는 개발 근거가 되므로 다음 모델의 untouched acceptance로 재사용하지 않는다.
- 다음 acceptance는 다시 **미사용 writer 2명 이상·50식 이상**을 모델 선택 전에 동결한다.
- 알려진 희소 문자 `|·O·o`는 각 **29개 truth instance 이상**, 합계 87개 이상을 포함한다.
  - 각 문자에서 오류 0건일 때 one-sided 95% 이항 상한을 10% 미만으로 만드는 최소 `n=29` 기준이다.
  - 동일 writer 반복만으로 채우지 말고 최소 2명의 untouched writer에 분산한다.
- 파인튜닝용 실제 `|·O·o` 수집분과 최종 acceptance 수집분은 서로 분리한다.

## 산출물

- 선택 동결: `artifacts/candidate_validity_adoption_selection_20260820_r1/selection.json`
- fresh 평가: `artifacts/candidate_validity_fresh_acceptance_20260820_r1/fresh_acceptance_report.json`
- shadow 결정: `artifacts/candidate_validity_adoption_gate_20260820_r1_shadow/shadow_decision.json`
- private acceptance 동결: `D:\AIFlow-Workspace\PrivateData\math-ink-data-collector\derived\fresh-context-acceptance-20260820-r2\frozen_acceptance\frozen_acceptance_manifest.json`
