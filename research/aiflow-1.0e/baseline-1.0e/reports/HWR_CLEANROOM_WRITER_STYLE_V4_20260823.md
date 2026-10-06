# AIFlow Math Ink 1.0 clean-room 합성 작가 v4

## 결론

- 같은 문자에 독립 잡음을 넣는 대신, 한 작가의 고정 latent를 영숫자 61종 전체에 공통 적용했다.
- 합성 작가 4명 × 61문자 = 244행을 생성했다.
- `delta_t`, `stroke_start`, `observed`, 획 수·대표 loop·회전방향 변화는 0건이다.
- 작가 쌍 최소 RMS 중앙값은 0.04564로, v3 미세변동보다 명확한 작가 간 분리가 생겼다.
- 상태는 **candidate_pass**다. 학습이나 제품 runtime 전환은 하지 않았다.

## 고정 writer latent

| 작가 | 특징 | slant | 폭 | 높이 | roundness | 운동속도 | 떨림 |
|---|---|---:|---:|---:|---:|---:|---:|
| compact_upright | 좁고 곧고 안정적 | -0.03 | 0.82 | 1.10 | 0.08 | 0.92 | 0.72 |
| slanted_round | 오른쪽 기울기, 둥근 획 | +0.22 | 0.98 | 1.02 | 0.28 | 0.88 | 0.62 |
| wide_relaxed | 넓고 낮으며 느슨함 | +0.04 | 1.22 | 0.88 | 0.18 | 0.82 | 0.82 |
| narrow_quick | 좁고 세로로 길며 빠름 | +0.13 | 0.76 | 1.16 | 0.04 | 1.18 | 1.22 |

각 latent는 다음 변수를 한 묶음으로 고정한다.

- slant, width/height ratio, baseline tilt
- stroke roundness, bow, terminal hook
- motor natural frequency, damping, minimum-jerk
- tremor와 low-frequency drift

## 인식 보존

| 필체 | Top-1 | Top-5 | topology 오류 | 스타일 완전 원복 |
|---|---:|---:|---:|---:|
| baseline | 75.41% | 98.36% | 0 | - |
| compact_upright | 72.13% | 95.08% | 0 | 0 |
| slanted_round | 68.85% | 91.80% | 0 | 0 |
| wide_relaxed | 73.77% | 95.08% | 0 | 1 |
| narrow_quick | 65.57% | 93.44% | 0 | 1 |

- 어려운 신규 작가 분포를 만들면서도 모든 작가의 Top-5가 90% 이상 유지됐다.
- 변형이 topology·경로길이·RMS 게이트를 넘으면 강도를 `1.00 → 0.82 → 0.64 → 0.46 → 0.28` 순으로 낮춘다.
- 끝까지 통과하지 못한 2행은 원본으로 되돌렸다.

## 경계

- 실제 특정 개인의 필체를 복제한 것이 아니다.
- 상용권리 clean-room 궤적과 사전 선언된 writer latent만 사용했다.
- CROHME·MathWriting 행, 통계, 오류 라벨 사용량은 0이다.
- HWR·문맥 체크포인트와 제품 runtime은 변경하지 않았다.
- 현재 결과는 증강 후보이며 정확도 향상은 별도 writer-LOO 학습 비교 전까지 주장하지 않는다.

## 산출물

- 구현: `scripts/cleanroom_writer_style_simulator_v4.py`
- 최종 데이터: `artifacts/cleanroom_writer_style_v4_20260823_r2/synthetic_writers_alphanumeric.npz`
- 전체 파라미터·감사: `artifacts/cleanroom_writer_style_v4_20260823_r2/writer_style_report.json`
- 시각화: `artifacts/cleanroom_writer_style_v4_20260823_r2/preview_writer_styles_digits.png`
