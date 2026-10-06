# AIFlow Math Ink 1.0 clean-room 프로파일 증강 v3 결과

## 결론

- CROHME 문자 Top-1: **75.99% → 76.70%**
- CROHME Top-5: **94.72% → 94.96%**
- Top-5 밖: **633 → 604건**
- 신규 작가 Top-1: **85.48% → 85.48%**
- 신규 작가 식 exact: **60.38% → 58.49%**
- writer-LOO Top-1: **80.62% → 81.91%**
- 모델·증강 데이터의 CROHME 학습 행, 통계, 오류 라벨 사용량은 모두 0이다.
- 아래 CROHME 방향 통계는 체크포인트 동결 뒤 계산한 평가 진단이며 재학습에 사용하지 않는다.

## 데이터셋

- 외부 프로파일 합성: 4,096행
- 프로젝트 final 합성: 512행
- 프로젝트 writer-LOO 합성: fold당 512행
- 프로파일 학습 특징: stroke_count, closed_loop, loop_direction, loop_start_phase, aspect_ratio, path_length, closure, turning
- 모든 합성행에 부모 지문, DTW 거리, 클래스 프로파일 변형, 물리 시뮬레이션 적용 여부를 기록했다.

## 게이트

| 지표 | v2 | v3 | 변화 |
|---|---:|---:|---:|
| writer-LOO Top-1 | 80.62% | 81.91% | +1.29%p |
| writer-LOO Top-5 | 96.38% | 96.64% | +0.26%p |
| 외부 holdout Top-1 | 77.94% | 78.23% | +0.29%p |
| 신규 작가 Top-1 | 85.48% | 85.48% | +0.00%p |
| 신규 작가 Top-5 | 97.31% | 97.31% | +0.00%p |

- 신규 작가 Top-1 개선/회귀: **1 / 1건**
- 핵심 교환: `x`: `\kappa` → `x`; `g`: `g` → `y`

## CROHME 최종 평가

| 지표 | v2 | v3 | 변화 |
|---|---:|---:|---:|
| 문자 Top-1 | 75.99% | 76.70% | +0.71%p |
| 문자 Top-5 | 94.72% | 94.96% | +0.24%p |
| strict macro Top-1 | 53.12% | 53.72% | +0.60%p |
| 식 exact | 22.54% | 23.21% | +0.67%p |
| 식 Top-5 oracle | 67.36% | 67.86% | +0.50%p |

- Top-1 개선/회귀: **116 / 31건**
- Top-5 구조/유실: **34 / 5건**
- 지속 Top-5 밖: **599건**

## 동결 후 원형문자 방향 진단

| 문자 | 표본 | 폐곡선 검출 | Top-1 | Top-5 | 방향 분포 | 시작 사분면 |
|---|---:|---:|---:|---:|---|---|
| `0` | 263 | 228 | 59.32% | 87.83% | {"clockwise_y_up": 207, "counterclockwise_y_up": 21, "no_primary_loop": 35} | {"bottom": 170, "left": 23, "none": 35, "right": 21, "top": 14} |
| `o` | 17 | 8 | 11.76% | 76.47% | {"clockwise_y_up": 8, "no_primary_loop": 9} | {"bottom": 7, "left": 1, "none": 9} |

### `0` 방향 비교

- 승인 학습 분할: 폐곡선 290/327건, 방향 {"clockwise_y_up": 279, "counterclockwise_y_up": 11}
- CROHME 평가 분할: 폐곡선 228/263건, 방향 {"clockwise_y_up": 207, "counterclockwise_y_up": 21, "no_primary_loop": 35}
- 승인 학습 `0` 시작각 원형 평균: 4.784 rad, 집중도 0.929

- 이 표는 CROHME를 증강 파라미터로 사용했다는 뜻이 아니다. 모델 동결 후 분포 차이를 설명하기 위한 격리된 평가 결과다.
- 다음 모델 개선에 이 수치를 직접 쓰려면 CROHME benchmark를 폐기해야 하므로, 현재 상용 clean-room 계보에서는 사용하지 않는다.

## 상태

- 신규 작가 acceptance: **실패**
- 외부 기술 비회귀: **통과**
- 제품 runtime 전환: **아니오**
- v3 상태: **shadow_rejected_by_fresh_acceptance**
- 신규 작가 식 exact 회귀가 있으므로 문맥 결합 이전에도 제품 채택할 수 없다.
