# AIFlow 1.0e 추가 업그레이드 및 상용화 판정 — 2026-09-01 r2

## 판정

추가 업그레이드는 완료했지만 전체 상용화는 아직 불가하다. 현재 허용 범위는 `bounded_review_required_beta`이며 `product_adopted=false`, `product_auto_commit=false`다.

제품 기본 경로는 기존 5-channel ordered online-ink HWR로 유지한다. 외부 OCR 가중치는 후보 보존형 shadow 경로에서만 비교한다.

## 추가 업그레이드

- 기존 project-owned 7 writer / 387 record / 95 formula 후보만 사용했다.
- 외부 사전학습 가중치의 encoder 마지막 block과 candidate selector만 2 epoch partial fine-tune했다.
- 새 token 생성, 후보 밖 선택, 삭제, stroke regrouping, relation 발명은 허용하지 않았다.
- confidence/margin fail-closed guard를 추가했다. 기준 미달이면 기존 HWR Top-1을 유지한다.

## 결과

| 경로 | Top-1 | 후보 Top-5 recall | 수식 exact | 행 회귀 |
|---|---:|---:|---:|---:|
| 고정 v4 HWR | 81.91% | 96.64% | 51.58% | 0 |
| 외부 adapter 원본 | 85.53% | 96.64% | 56.84% | 19 |
| 외부 adapter + fail-closed guard | 86.30% | 96.64% | 62.11% | 0 |

Guard 결과는 현재 개발 writer-LOO에서만 확인됐다. threshold `confidence=0.30`, `margin=0.30`은 known development rows를 연 뒤 정한 post-hoc 값이므로 fresh acceptance를 대체하지 않는다.

## 상용 gate

- 현재 writer-LOO와 수학/후보 보존 self-test: 통과.
- fresh untouched writer/formula acceptance: 실패. 수집기 마지막 실행은 `listed=164`, `downloaded=0`, `newItems=0`이다.
- exact/AST equivalence: 실패. guard self-test는 통과했지만 새 제품 acceptance에서 end-to-end 검증이 없다.
- Android: 실패. `adb`가 없고 1.0e LiteRT bundle 및 실제 기기 p50/p95·PSS·battery 결과가 없다.
- 권리: 실패. 외부 Azu 모델 페이지에 license 표시가 없어 상용 배포 권리와 attribution을 확정할 수 없다. TexTeller의 Apache-2.0 표시는 확인되지만 serial online-ink→TexTeller 경로는 별도 shadow 실패 경로다.

따라서 현재 후보는 성능상 유망하지만 상용 제품 기본값으로 승격하지 않는다. 사람 검토가 필요한 제한 beta/shadow만 유지한다.

재승격 조건은 fresh writer/formula-disjoint acceptance를 threshold 동결 전에 확보하고, 그 세트에서 행 비회귀·formula exact·AST/semantic equivalence, Android 실측, 외부 가중치 권리/notice 감사를 모두 통과하는 것이다.

상세 판정은 [release_manifest.json](../artifacts/commercial_release_20260901_r2/release_manifest.json), guard 증거는 [guard_report.json](../artifacts/ocr_trocr_candidate_guard_20260901_r1/guard_report.json)이다.
