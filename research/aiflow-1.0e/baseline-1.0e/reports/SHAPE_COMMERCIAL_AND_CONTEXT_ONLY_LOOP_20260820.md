# 형태 분류기 상용 채택 검토 및 문맥 전용 학습 루프

- 검토일: 2026-08-20
- 판정 범위: AIFlow Math Ink 1.0의 372-class 온라인 필기 형태 분류기와 Top-5 후보 보존형 문맥 분류기
- 최종 판정:
  - 형태 분류기: **상용 기본 분류기로 채택 불가**. 박스 내부 Top-5 후보를 만드는 제한적 컴포넌트로만 `shadow` 채택 가능
  - 문맥 분류기: 전용 학습은 유효했으나 비회귀와 CROHME 전이 게이트를 통과하지 못함. 기존 r6 문맥 모델을 새 형태 후보에 재결합한 `no-op`을 안전 선택으로 유지

## 1. 형태 분류기 상용 채택 판정

### 1.1 권리 게이트

프로젝트의 데이터 역할 제한과 고지 의무를 지키는 조건에서 학습 이용 자체는 통과한다.

| 데이터 | 확인된 조건 | 이 프로젝트의 허용 역할 |
|---|---|---|
| UJI Pen Characters v2 | CC BY 4.0 | 박스 내부 문자 후보 학습 |
| ISGL | CC BY 4.0 | 온라인 궤적만 사용, 박스 내부 후보 학습 |
| UCI Character Trajectories | CC BY 4.0, 단일 필기자 | 학습 보조만 사용 |
| HWRT | ODbL 1.0 | 정제된 온라인 궤적의 후보 학습만 사용 |
| 프로젝트 직접 수집 | 동의·소유권 레코드가 있는 표본 | 그룹·최종 판정과 제품 검증 |
| CROHME / MathWriting | 현재 프로젝트 계약상 비상업 평가 전용 | 학습·모델 선택 금지 |

공식 자료: [UJI](https://archive.ics.uci.edu/dataset/177/uji%2Bpen%2Bcharacters%2Bversion%2B2), [UCI Character Trajectories](https://archive.ics.uci.edu/dataset/175/character%2Btrajectories), [ISGL](https://data.mendeley.com/datasets/n7kmd7t7yx/1), [HWRT](https://zenodo.org/records/50022).

배포물에는 CC BY 출처 고지가 필요하다. HWRT로부터 파생된 데이터베이스를 배포하는 경우 ODbL의 소스 제공·동일조건 의무 적용 범위는 배포 방식에 맞춰 별도 법무 검토가 필요하다. 이 문서는 법률 의견이 아니다.

### 1.2 모델·입력 계약

- 입력: `128 points × 5 channels (x, y, Δt, stroke_start, observed)`
- 구조: `Linear 5→128` → 학습형 위치 임베딩 → 4개 Transformer block(4-head, FFN 512) → attention pooling → 단일 372-class head
- 파라미터: 858,613개
- 형태 모델 역할: 위치·그룹을 새로 만들지 않고, 상위 후보와 점수를 문맥 계층에 전달
- 제품용 전체 필기자 체크포인트 SHA-256: `04f8608aebcf6c02d45ad6f5735229b9eaa2c4b4e1be0db4793d02273ef2d00e`
- writer-LOO head 묶음 SHA-256: `80464eb50ef14acb194c944fc5c95201ce94d03e4687b721a5f097aad67706bc`
- 고정 OOF 후보 캐시 SHA-256: `d64d9a668ceed7a1d4038777a628f21cff743abb203efa0711fce735f55082b1`

### 1.3 누수 방지 성능

직접 수집 387 glyph, 95 formula, 7 writer를 writer-LOO로 평가했다. 전체 필기자에 재학습한 최종 head는 이 수치에 사용하지 않았다.

| 지표 | 결과 |
|---|---:|
| glyph Top-1 | 72.87% |
| glyph Top-5 | 95.35% |
| formula Top-1 exact | 33.68% |
| formula Top-5 oracle | 86.32% |
| writer macro Top-1 / Top-5 | 71.60% / 93.60% |
| 최저 writer Top-1 / Top-5 | 60.87% / 73.91% |
| 외부 충돌 제거 holdout Top-1 / Top-5 | 77.34% / 97.37% |

외부 holdout은 데이터 계약과 도메인이 달라 기술 진단일 뿐 제품 정확도 근거가 아니다. 핵심 동형 계열도 불안정하다.

| truth | 표본 | Top-1 | Top-5 |
|---|---:|---:|---:|
| `0` | 24 | 54.17% | 87.50% |
| `1` | 41 | 34.15% | 92.68% |
| `+` | 37 | 83.78% | 94.59% |
| `=` | 42 | 78.57% | 92.86% |
| `/` | 8 | 87.50% | 100.00% |
| `x` | 5 | 20.00% | 100.00% |
| `\times` | 18 | 61.11% | 100.00% |
| `z` | 0 | 평가 불가 | 평가 불가 |

### 1.4 판정

| 게이트 | 판정 | 근거 |
|---|---|---|
| 상업 이용 권리 | 조건부 통과 | 출처 고지·ODbL 경계·평가 전용 데이터 분리 필요 |
| 입력/출력 계약 | 통과 | 온라인 궤적, 고정 128점, 단일 372 head |
| Top-5 후보 생성 | 연구용 통과 | 전체 OOF 95.35% |
| 자동 Top-1 문자 확정 | 실패 | 전체 72.87%, 최저 writer 60.87% |
| 수식 최종 확정 | 실패 | OOF formula exact 33.68% |
| 사용자 일반화 | 실패 | 7 writer뿐이며 최저 writer Top-5 73.91% |
| 상용 기본 채택 | **실패** | 기존 산출물도 `product_adopted=false`; 새 untouched acceptance 없음 |

따라서 형태 분류기는 **원시 입력 fallback을 보전하는 Top-5 후보 생성기**로만 쓸 수 있다. Top-1 단독 응답, 수식 그룹 결정, 최종 문자열 확정 권한은 부여하지 않는다.

상용 승격 전 최소 조건은 새 필기자·새 수식의 완전 미접촉 acceptance set, `0/1/x/\times/+/=/z` 최소 표본, writer/formula 동시 분리, 모바일 실측 지연시간 p50/p95, 라이선스 고지 패키지 확인이다.

## 2. 문맥 분류기 전용 학습 루프

### 2.1 분리 계약

- 형태 분류기 학습: 0회
- 형태 분류기 gradient update: 0회
- 입력 후보: writer-LOO Top-5 캐시로 고정
- 문맥 출력: 입력 Top-5 중 하나만 선택
- 새 token 생성: 전 평가에서 0건
- grouping mutation: 전 평가에서 0건
- HWR·후보·기존 문맥·CROHME 캐시 SHA-256: 학습 전후 모두 동일
- 선택 데이터: `new_writer` 4명, 172 glyph, 44 formula에 대한 outer leave-one-writer-out
- CROHME: 학습·설정 선택에 사용하지 않고 선택 완료 뒤 전이 진단만 수행

### 2.2 학습 탐색

- r2: 문맥망 전체, learning rate `1e-5 / 3e-5 / 1e-4`, epoch `1 / 2 / 3`
- r3: `heads` 또는 `top_layer_heads`, learning rate `1e-4 / 3e-4 / 1e-3`, epoch `1 / 2 / 3`
- 모든 fold는 held writer를 학습에서 제외
- `no-op`도 정식 후보로 넣고, glyph·formula 회귀가 모두 0인 후보만 안전 선택 가능

### 2.3 새 필기자 OOF 결과

| 모델 | glyph Top-1 | formula exact | strict macro | 기존 대비 개선/회귀 | 후보 위반 |
|---|---:|---:|---:|---:|---:|
| 안전 선택: 기존 r6 재결합 | 84.88% | 59.09% | 70.83% | 0 / 0 | 0 |
| r2 전체망, lr 1e-4, 3 epoch | 87.79% | 70.45% | 90.83% | 6 / 1 | 0 |
| r3 상위층+head, lr 1e-3, 1 epoch | 87.79% | 70.45% | 90.83% | 6 / 1 | 0 |

두 학습 범위 모두 같은 기존 정답 1건을 훼손했다. `aiflow_0009`의 truth/HWR/r6는 `y`였지만 challenger가 `\oint`로 변경했다. 해당 입력의 HWR 확률은 `y=0.92547`, `\oint=0.014237`로, 단순 동형문자 교체라고 보기 어려운 과도한 문맥 개입이다. 이 1건만 막는 token 전용 규칙은 표본 과적합 위험 때문에 추가하지 않았다.

### 2.4 CROHME 사후 전이 진단

| 모델 | glyph Top-1 | formula exact | strict macro | 후보 위반 |
|---|---:|---:|---:|---:|
| 안전 선택: 기존 r6 재결합 | 78.11% | 27.44% | 56.07% | 0 |
| r2 전체망 challenger | 75.85% | 25.41% | 57.98% | 0 |
| r3 상위층+head challenger | 75.36% | 25.20% | 56.10% | 0 |

r2 challenger의 strict macro만 상승했지만 전체 glyph와 formula exact가 함께 하락했다. 따라서 추가 epoch보다 새 untouched writer/formula 데이터가 먼저 필요하다.

### 2.5 채택 아티팩트

- 안전 선택: `artifacts/context_only_incremental_20260820_r2_shadow/context_only_product.pt`
  - SHA-256: `c99a4c9f77d73cffa076d5980e1c84898caf6f6ff22d9e8907e118271b9d0c9a`
  - reload mismatch: 0
  - 상태: `shadow`; 기존 r6의 가중치를 새 형태 후보 계약에 다시 묶은 no-op
- 정확도 challenger: `artifacts/context_only_incremental_20260820_r2_shadow/context_only_challenger.pt`
  - SHA-256: `adc51c32715ff0465078a81a958fe4c011f02f10a7e2ec663c207e6aa832e891`
  - reload mismatch: 0
  - 상태: 연구용 보존, 자동 교체 금지
- 동결 범위 challenger도 동일한 회귀와 더 낮은 CROHME 전이를 보여 채택하지 않는다.

## 3. 결론

1. 형태 분류기는 상용 코드에 들어갈 수 있으나 **Top-5 후보 생성·raw fallback 보조 역할**로 한정한다. 현재 수치로 상용 기본 분류기나 최종 판정기로 홍보·채택할 수 없다.
2. 문맥 분류기만 따로 학습하는 루프는 구현·실행되었다. 형태 가중치와 후보 캐시는 변경되지 않았다.
3. 학습 challenger는 새 필기자 OOF에서 개선됐지만 1건 회귀와 CROHME 전이 하락 때문에 `shadow` 연구 산출물로 남긴다.
4. 현재 안전 선택은 r6 문맥 모델 재결합이다. 추가 학습은 새 미접촉 acceptance 데이터가 확보된 뒤 재개한다.

- 재현 코드: `scripts/train_context_only_incremental_loop_v1.py`
- 전체 결과: `artifacts/context_only_incremental_20260820_r2_shadow/context_only_loop_report.json`, `artifacts/context_only_incremental_20260820_r3_frozen_scope_shadow/context_only_loop_report.json`
