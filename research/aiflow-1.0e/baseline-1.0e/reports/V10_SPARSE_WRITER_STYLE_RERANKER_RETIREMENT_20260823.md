# V10 sparse writer-style reranker 종료 판정

## 판정

- 상태: `SHADOW_RETIRED`
- 제품 채택: 불가
- writers064..095 재실행·결과 맞춤 튜닝: 금지
- frozen 128×5·372-class HWR/checkpoint/runtime: 변경 없음

## 근거

- one-shot synthetic outer: writers064..095, 32명
- 평가 query: 122,880행, sparse calibration episode 192개
- Top-1: 65.8984375% → 65.9033203125%
- 순효과: 8개선·2회귀, net +6행(+0.0048828%p)
- pseudo-8 exact: 6.484375% → 6.484375%(변화 없음)
- baseline Top-5 집합 위반: 0
- 회귀:
  - writer076/support24: `w → \varpi`
  - writer088/support24: `\otimes → \mathbb{Q}`
- strict writer×support 비회귀 gate 실패

효과가 122,880행 중 순 6행에 그쳤고 회귀가 남아 있어, 후단 reranker 복잡도를 유지할 실효가 없습니다.

## 불변 영수증

- 구현 커밋: `ae40731`
- pre-open 독립 감사 SHA-256: `b642d3c6ae79ab4bd212f223ba090fd6eaa209637f18618aed1a4602425510d5`
- post-open 독립 감사 SHA-256: `e08d8497ae91995a7fbdca9da1355049c50556f4f580d23f698b4fcb949fb55a`
- outer report SHA-256: `86791c3d68e749cfa3f8972fe0084244085ab4650ddad0732a7a1b4c7ee17e17`
- decision SHA-256: `703cc5fdb687963542c56cd52ae4534a0930154c0b605ec922d05402aa8a4410`
- cache SHA-256: `9eb60268ab46f8a61233ebdfa23494b2d037bbc5f67490a15c53a431372a7738`
- trace SHA-256: `7340dfe7ef2b254c0cb6797163669687f3d90ee1b83d95bbb2eea3256b47942a`

## 다음 연구 경계

- 000..095와 소비된 Legacy는 development-only로만 사용합니다.
- 다음 후보도 calibration-only이며 query label·writer ID·metadata를 예측 입력으로 받지 않습니다.
- baseline Top-5 집합을 100% 보존하고 불확실하면 identity로 후퇴합니다.
- Legacy 7-writer formula-disjoint nested writer-LOO에서 모든 작가의 Top-1·실제 formula exact 비회귀와 집계 양의 개선을 먼저 요구합니다.
- writers096..127은 독립 admission 후에도 untouched synthetic outer로 봉인하며, 위 Legacy gate 통과 전 scorer cache·feature·prediction을 만들지 않습니다.
- 최종 제품 승격은 완전 미사용 fresh real writer/formula의 1회 acceptance 없이는 불가합니다.
