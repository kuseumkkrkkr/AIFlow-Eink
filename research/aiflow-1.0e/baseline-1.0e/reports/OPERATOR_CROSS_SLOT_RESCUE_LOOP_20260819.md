# 연산자 십자 슬롯 배치 복원 루프

## 결론

- `1 \bot 5`를 `1 + 5`로 복원하는 후보 보존형 배치 가드를 추가했다.
- 직전 shadow 기준 직접 수집은 문자 `359/387 → 360/387`(93.02%), 식 exact `79/95 → 80/95`(84.21%)로 증가했다.
- 직접 수집 문자·식 회귀는 0건이다.
- CROHME 반복 비상업 진단은 문자 `7516/9535`, 식 exact `288/984`로 변경과 회귀가 모두 0건이다.
- 제품 기본값은 변경하지 않았으며 상태는 `shadow_runtime_only`다.

## 조건

다음 조건을 모두 만족하는 정확히 3글자 식만 검사한다.

- 현재 열이 `숫자, \bot, 숫자`다.
- 가운데 글자의 Top-20 앞 두 후보가 순서대로 `\bot`, `\perp`다.
- 같은 후보 목록의 허용 산술 연산자는 `+` 하나뿐이다.
- 가운데 궤적이 2-stroke이고 `abs(aspect_log) <= 0.3`인 십자 비율이다.
- `1.2 <= path/diagonal <= 1.8`이다.
- `+`로 바꾼 열이 숫자 구문으로 유효하다.

산술 결과는 계산하지 않는다. 문자 삽입·삭제, 순서 변경, stroke regrouping, 배치 관계 변경도 수행하지 않는다.

## 가설 분리

같은 probe에서 `I,/ → 1` 이중 피연산자 복원도 시험했지만 현재 보수 조건에서는 직접·CROHME 모두 변경 0건으로 기권했다. 근거 없이 조건을 풀지 않았다.

`\bot → +` 분기는 직접 수집 한 글자·한 식만 고쳤고 CROHME에서는 발동하지 않았다. 따라서 반복 개발 진단 gate는 통과했지만 독립 양성·음성 검증은 아직 없다.

## 누적 상태

| 범위 | 문자 Top-1 | 식 exact | 이번 단계 개선/회귀 |
|---|---:|---:|---:|
| 직접 수집 개발 95식 | 360/387 (93.02%) | 80/95 (84.21%) | 1/0식 |
| CROHME 반복 비상업 진단 | 7516/9535 (78.83%) | 288/984 (29.27%) | 0/0식 |

Top-20 식 oracle은 직접 91/95다. 현재 80/95와의 차이 11식은 후보가 있어도 의미를 보존하며 확정할 독립 근거가 없고, 추가 4식은 정답이 Top-20 밖이다.

## 재현 근거

- 가설 probe SHA-256: `7aeb6f206de12d0790323becc7fe2aae50613389ebf1548a1fbd0ed031e560cb`
- 평가 SHA-256: `0c7250ff33fc8ca6c3cbdb1b8833b91315d1fd5d11193cd6d6e3318e3aca361e`
- runtime config SHA-256: `863e2624f065accdfcf194d2c29b71a80c9713eeba639cce7ab229ce301e0977`
- 직접 shadow runtime SHA-256: `9a639ebefd22fc592a0f444187fd7e44b2680e67d1f3bbce252a77e49a9d0dfe`
- CROHME shadow runtime SHA-256: `7efec5580bce5b93494d5b21f2c900b9da4173caf24504e615d34a44a67eecbb`
- 요약 산출물: `artifacts/operator_cross_slot_rescue_20260819_r1_shadow/operator_cross_slot_rescue_summary.json`

## 승격 조건

- 새 프로젝트 소유 writer/formula의 `+`, `\bot`, `\perp` 양성·음성 표본.
- untouched 상업 acceptance에서 문자·식 비회귀.
- Top-20 비용 측정과 후보·배치·grouping 불변 재검증.
