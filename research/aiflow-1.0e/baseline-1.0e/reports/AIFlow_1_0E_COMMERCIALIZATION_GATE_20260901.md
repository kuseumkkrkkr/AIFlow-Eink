# AIFlow 1.0e 상용화 게이트 — 2026-09-01

## 최종 판정

전체 자동 상용화는 보류한다. 현재 확정 가능한 배포 범위는 `bounded_review_required_beta`이며 `product_adopted=false`, `product_auto_commit=false`다.

제품 본체는 기존 `128×5` ordered online-ink HWR 후보를 유지한다. serial online-ink→TexTeller 경로는 외부 가중치 연구 증거로만 남기고 제품 런타임에 넣지 않는다.

## 연구 루프 결과

새 v3 clean-room 재탐색은 `profiled_last_block_fast`를 선택했다.

| 지표 | 고정 v4 후보 | 새 재탐색 | 판정 |
|---|---:|---:|---|
| writer-LOO 문자 Top-1 | 81.91% | 81.91% | 동일 |
| writer-LOO 문자 Top-5 | 96.64% | 96.38% | 회귀 |
| 선택 자료 | 7 writer / 387 glyph / 95 formula | 동일 | fresh 미사용 |
| CROHME·MathWriting 학습 | 0 | 0 | 경계 통과 |

따라서 새 후보는 승격하지 않고 v4 `alpha=0.8` shadow 후보를 보존한다. v4 고정 writer-LOO는 Top-1 81.91%, Top-5 96.64%, formula exact 51.58%, row-level Top-1 회귀 0건이다. 이 수치는 fresh acceptance가 아니다.

## 차단 gate

- 수집기 마지막 실행: `listed=164`, `downloaded=0`, `newItems=0`. 새 writer/formula acceptance가 없다.
- 수학 guard self-test는 통과했지만, 새 미접촉 제품셋에서 exact와 AST/semantic equivalence를 증명하지 못했다.
- 1.0e용 LiteRT bundle 및 실제 Android p50/p95·PSS·battery 결과가 없다.
- serial distilled 경로는 writer-LOO exact `0/47`, baseline `16/47`, train-set overfit도 실패했다.
- 외부 가중치·데이터의 배포 권리와 attribution notice를 모두 묶은 release audit이 끝나지 않았다.

## 허용되는 현재 배포 경계

- HWR Top-k 후보 보존
- 후보 밖 token 생성·삭제·stroke regrouping·관계 발명 금지
- 불확실 식은 `REVIEW_REQUIRED`로 보내고 원본 stroke를 보존
- 자동 채점/자동 확정은 켜지 않음
- 외부 OCR serial 경로는 shadow 비교만 수행

상세 machine-readable 판정은 [release_manifest.json](../artifacts/commercial_release_20260901_r1/release_manifest.json)이다.
