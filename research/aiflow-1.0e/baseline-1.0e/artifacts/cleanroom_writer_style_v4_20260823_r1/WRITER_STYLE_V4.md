# Clean-room writer-style simulator v4

## 결론

- 동일 영숫자 61종을 서로 다른 고정 writer latent 4명으로 생성했다.
- 생성 행: 244행; topology mismatch: 0행.
- writer 쌍 최소 RMS 중앙값: 0.0456.
- 글자마다 무작위 작가를 바꾸지 않고, 한 작가의 공간·운동 파라미터를 전체 문자에 고정했다.
- 학습, CROHME/MathWriting 사용, 제품 체크포인트/runtime 변경은 없다.

## 합성 작가

| writer | slant | 폭 | 높이 | roundness | 속도 | 감쇠 | 떨림 | 드리프트 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| writer_compact_upright | -0.03 | 0.82 | 1.10 | 0.08 | 0.92 | +0.07 | 0.72 | 0.68 |
| writer_slanted_round | +0.22 | 0.98 | 1.02 | 0.28 | 0.88 | +0.09 | 0.62 | 0.72 |
| writer_wide_relaxed | +0.04 | 1.22 | 0.88 | 0.18 | 0.82 | -0.04 | 0.82 | 1.12 |
| writer_narrow_quick | +0.13 | 0.76 | 1.16 | 0.04 | 1.18 | -0.06 | 1.22 | 0.88 |

## HWR label 보존

| writer | Top-1 | Top-5 | 스타일 원복 |
|---|---:|---:|---:|
| writer_compact_upright | 72.13% | 95.08% | 0 |
| writer_slanted_round | 68.85% | 91.80% | 0 |
| writer_wide_relaxed | 73.77% | 95.08% | 1 |
| writer_narrow_quick | 65.57% | 93.44% | 1 |

## 해석

- v3는 동일 필체의 미세 운동 변동, v4는 글자 집합 전체에 고정되는 작가 스타일 변동을 담당한다.
- writer latent는 slant·aspect·roundness·bow·terminal hook과 motor 응답을 묶는다.
- topology 또는 경로 길이·RMS 게이트를 넘으면 해당 문자만 스타일 강도를 낮추거나 원복한다.
- 합성 작가는 실제 개인을 복제한 것이 아니라 범위가 고정된 clean-room latent다.
