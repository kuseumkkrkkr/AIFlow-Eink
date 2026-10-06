# AIFlow Math Ink 0.9 릴리스 근거

## 선택

- seed: 31
- checkpoint SHA-256: `46eb2b283d5d61f707572e0295e4001e759fbe1693660fe92104347108dff725`
- 연구 채택: 예
- 제품 채택: 아니오

## 3-seed 재현

| seed | 문자 Top-1 | 문자 Top-5 | 수식 exact | 수식 Top-5 oracle |
|---:|---:|---:|---:|---:|
| 17 | 85.96% | 98.25% | 67.35% | 93.88% |
| 31 | 87.72% | 98.83% | 67.35% | 95.92% |
| 47 | 85.96% | 98.25% | 65.31% | 93.88% |
| 평균 | 86.55% | 98.44% | 66.67% | 94.56% |

동일 baseline은 문자 Top-1 50.29%, Top-5 70.18%, 수식 exact 26.53%, 수식 Top-5 oracle 42.86%였다.

## 실패 및 한계

- seed31 paired: improved 67, regressed 3, unchanged 101
- `f`: 6/6 → 5/6
- `g`: 4/5 → 2/5
- `/`, `\sqrt{}`는 replay에서 여전히 0/1
- 학습·replay 수식/prompt exact overlap: 0
- replay contributor ID 부재: writer-disjoint 미증명
- 0.9 CROHME/MathWriting 평가: 수행하지 않음

## 공개 데이터 판단

원본 119개 record는 모델 학습·검증 동의만 있으며 공개 재배포 동의가 없다. 따라서 원본 stroke/이미지/식별자는 제외하고 비식별 집계 카드만 공개한다.
