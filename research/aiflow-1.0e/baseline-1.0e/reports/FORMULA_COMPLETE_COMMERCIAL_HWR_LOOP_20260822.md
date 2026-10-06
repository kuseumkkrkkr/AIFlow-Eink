# AIFlow Math Ink 1.0 상용화 경계 및 전체 수식 평가

## 결론

- **형태 HWR는 유지**한다. 다만 단독 Top-1 정확도를 상용 완성도로 보지 않는다.
- **전체 수식 완료 뒤 한 번만** global grouping → 고정 HWR Top-5 → 후보 보존 문맥 → 2D 구조를 실행한다.
- 현재 grouping/context 산출물은 **shadow**다. 결과는 제공하지만 제품 자동 확정은 하지 않고 `REVIEW_REQUIRED`와 원시 잉크 fallback을 반환한다.
- 상업권리 영문 보조 헤드는 writer-disjoint dev가 부족하여 **미채택**이다.
- CROHME는 모든 결과에서 **학습 0, 선택 0, gradient update 0**인 검증 전용이다.

## 1. 형태 분류기와 Top-5 밖 749건

- 진실 그룹 기준 HWR Top-1: **72.70%**
- Top-5: **93.75%**
- 정답이 Top-5 밖: **749 / 11991건**
- 상업권리/프로젝트 학습 지원 0건인 오류: **0건**
- 따라서 749건은 ‘아예 학습하지 않은 문자’ 문제가 아니다. 같은 문자의 필기 도메인 차이, 과도한 372-class 경쟁, 동형·유사형 혼동이 핵심이다.

| 문자 | Top-5 밖 | 정답 중앙 순위 | 상업권리+소유 학습 지원 |
|---|---:|---:|---:|
| `2` | 101 | 19 | 345 |
| `1` | 99 | 9 | 344 |
| `+` | 59 | 21 | 105 |
| `z` | 54 | 56.5 | 426 |
| `0` | 53 | 9 | 343 |
| `x` | 42 | 11.5 | 277 |
| `=` | 41 | 19 | 20 |
| `a` | 33 | 15 | 447 |
| `b` | 29 | 11 | 398 |
| `k` | 18 | 11.5 | 270 |
| `4` | 17 | 8 | 284 |
| `7` | 16 | 9.0 | 292 |

## 2. 상업권리 영문 필기 보강

- fit **15193건 / 96 writers**, dev **4303건 / 29 writers**, writer overlap **0**
- writer-disjoint dev Top-1/Top-5: **58.52% / 88.94%**
- 최악 writer Top-1/Top-5: **37.20% / 67.68%**
- CROHME의 기존 749건 중 보조 어휘로 평가 가능한 586건에서 Top-1 rescue **77건**, Top-5 rescue **234건**
- 이 수치는 고정 후 전이 진단일 뿐 채택 근거가 아니다. 대소문자·숫자·점 형태 충돌이 커서 별도 제품 헤드로 승격하지 않는다.

## 3. 스트리밍 대신 전체 수식 완료 기준

- 진실 grouping 제공 문자 Top-1: **72.70% → 76.51%** (+3.81%p)
- 문자열 exact proxy: **17.61% → 22.12%** (+4.51%p)
- 개선/퇴행: **680 / 223**, 순증 **457건**
- 완료 전 확정과 prefix 정확도 평가는 모두 0이다.

### 빈도 편향

- 전체 정확도는 +3.81%p, 과다예측 질량은 2574 → 2139로 감소했다.
- 그러나 진실 빈도 10~49 구간은 **-3.27%p** 퇴행했다.
- 전체 퇴행 223건 중 더 빈번한 문자로 이동한 경우는 58건(26.01%)이다.
- 즉 전체 수식 판정은 스트리밍보다 낫지만 빈도 편향을 완전히 해결하지 못했다. 이 때문에 자동 확정이 아니라 review 경계를 유지한다.

## 4. 원시 획부터 grouping·2D까지

- 대상: **769식**; truth label/writer/glyph count 입력 없음
- grouping exact: **39.01%** (300식)
- layout order exact: **21.72%**
- relation micro F1: **79.97%**
- strict grouping+layout+relation+문자 exact: **6.50%** (50식)
- 원시 획 fallback 일치: **769 / 769식**, 제품 자동 확정 **0건**
- 병목은 문자 문맥이 아니라 **상업권리 grouping/2D 정답 부족**이다. CROHME truth를 학습에 넣어 수치를 올리지는 않는다.

## 5. 보유 데이터 비회귀와 경계 감사

- 보유 110식: grouping/HWR/context/final projection mismatch **0건**
- 완료 전 확정 **0건**, 획 유실·중복 **0건**, 원시 fallback mismatch **0건**
- 훈련 진입점 **15개** + 코퍼스 준비 **2개**: 실패 **0**, CROHME 오염 선택식 **0**
- CROHME 비참조 v2 코퍼스: DeepMind **80000식**, 동형문자 coverage **12000식**
- CROHME로 48식을 제외했던 DeepMind v1과 CROHME를 읽었던 coverage v1은 격리했다.

## 6. 추가 수집량과 승격 조건

- 가정한 상용 grouping gate: 관측 exact ≥ **90.00%**, 95% Wilson 하한 ≥ **85.00%**
- 독립식 가정 최소 untouched acceptance: **159식**
- writer당 최대 20식으로 묶으면 실무 배치 **160식 / 최소 8 writers**
- 이는 통계적 acceptance 하한이지 충분한 학습량 증명이 아니다. 새 training tranche도 160식 단위로 추가하고, 이미 본 acceptance 식을 모델·epoch·임계값 선택에 재사용하지 않는다.

## 채택 상태

| 구성요소 | 상태 | 이유 |
|---|---|---|
| 기존 상업권리 372-class HWR | 유지 | Top-5 93.75%; 전체 수식 후보 공급기로 유효 |
| 상업권리 영문 auxiliary | 미채택 shadow | writer-disjoint dev Top-1 58.52% |
| formula-complete 문맥 | shadow 적용 | 후보 보존 상태에서 aggregate 개선, rare-mid 편향 잔존 |
| raw grouping/2D | 미채택 shadow | grouping exact 39.01%, strict exact 6.50% |
| 제품 자동 확정 | 비활성 | `REVIEW_REQUIRED` + 원시 잉크 fallback |
