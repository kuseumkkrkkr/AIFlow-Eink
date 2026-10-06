# AIFlow Math Ink 1.0 HWR 안정화 v4 상용화 게이트

## 결론

- **개선 루프 자체는 유효**했다. v2와 v3를 가중치 공간에서 보간한 단일 체크포인트 중 `alpha=0.8`이 writer-LOO 개선을 가장 많이 유지하면서 개별 회귀 0건을 기록했다.
- **상용 런타임 전환은 보류**한다. 새 미사용 작가 데이터가 없고, 이미 관측된 186개 글자 진단에서 `g → y` 회귀와 식 exact 하락이 그대로 남았다.
- 같은 186건 또는 CROHME로 추가 튜닝하면 승인셋 과적합이 되므로 현재 계보의 반복 학습은 중지한다.
- v4는 `shadow` 체크포인트로 보관한다. HWR·문맥 체크포인트 쌍, 제품 기본값, Vercel 배포는 변경하지 않았다.

## v4 방법

- 입력·구조: `128 × 5`, hidden 128, Transformer 4블록·4헤드, 372-class 단일 출력 헤드 유지
- 추가 파라미터: 0
- 추가 gradient update: 0
- 후보: clean-room v2와 profiled v3 상태를 `alpha=0.0~1.0` 11개 지점에서 선형 보간
- 선택 자료: 프로젝트 소유 7-writer LOO 387글자·95식만 사용
- 선택 금지: CROHME, MathWriting, 이미 관측된 신규 작가 186글자
- 선택 규칙: Top-1 개별 회귀 0건, Top-5 유실 0건, 문자·식·strict macro·동형계열 비회귀 후 개선 수 최대, 동률이면 alpha 최소
- 외부 18,344글자는 선택 완료 뒤 기술 비회귀 게이트에만 사용

## writer-LOO 결과

| 지표 | v2 기준 | v4 alpha 0.8 | 변화 |
|---|---:|---:|---:|
| 문자 Top-1 | 80.62% | 81.91% | +1.29%p |
| 문자 Top-5 | 96.38% | 96.64% | +0.26%p |
| 식 exact | 47.37% | 51.58% | +4.21%p |
| 식 Top-5 oracle | 89.47% | 90.53% | +1.05%p |

- Top-1 개선/회귀: **5 / 0건**
- Top-5 구조/유실: **1 / 0건**
- Top-1 출력 변경은 6건이며, 그중 5건이 정답 구조, 1건은 오답 간 교환이다.

## 고정 외부 holdout

| 지표 | v2 기준 | v4 alpha 0.8 | 변화 |
|---|---:|---:|---:|
| 문자 Top-1 | 77.94% | 78.21% | +0.27%p |
| 문자 Top-5 | 97.38% | 97.40% | +0.02%p |
| strict macro Top-5 | 96.72% | 96.74% | +0.03%p |

- `vertical_slash`, `circle`, `cross`, `descender` 계열 Top-5는 모두 비회귀했다.
- 외부 기술 게이트: **통과**

## 이미 관측된 186글자 진단

이 분할은 v3 평가 때 이미 열렸으므로 acceptance나 모델 선택 근거가 아니다.

| 지표 | v2 기준 | v4 alpha 0.8 |
|---|---:|---:|
| 문자 Top-1 | 85.48% | 85.48% |
| 문자 Top-5 | 97.31% | 97.31% |
| 식 exact | 60.38% | 58.49% |

- 개선: truth `x`, `\kappa → x`
- 회귀: truth `g`, `g → y`
- 두 건 모두 정답은 Top-5 안에 남았다.
- 진단 게이트 중 `formula_exact_nonregression`, `top1_regressions_zero`가 실패했다.

## 새 데이터 상태

- 2026-08-22 KST 수동 폴링: `Listed=164, Downloaded=0, Total=164`
- 예약 작업 `AIFlow Math Ink Data Poller`: 활성·정상 실행
- 유효 데이터 작가 9명은 7명 학습 + 2명 이미 관측된 검증으로 모두 소진됐다.
- 추가 작가 3명의 5개 기록은 모두 1~2점뿐인 결측 궤적이라 검증·학습에 사용할 수 없다.
- 따라서 현재 로컬·원격 자료에는 상용 승격에 쓸 새 미사용 작가가 없다.

## 상용 승격 재개 조건

다음 조건을 충족하는 새 데이터가 들어온 뒤에만 루프를 재개한다.

1. 기존 9명과 겹치지 않는 작가 최소 2명
2. 최소 50식·180글자
3. `|, O, o, /, x, \times, g, q, y` 중 가능한 동형계열 포함
4. 예측을 열기 전에 ownership 전수 검수와 파일 SHA-256 동결
5. v2 대비 Top-1·Top-5·식 exact·strict macro Top-5·동형계열 Top-5 비회귀
6. 개별 Top-1 회귀 0건
7. 통과 후 v4 Top-5 캐시로 문맥 모델을 다시 결합하고 식 exact·grouping·layout 비회귀 확인
8. HWR·문맥 체크포인트 해시를 한 쌍으로 고정한 뒤에만 제품 런타임 전환

## 산출물

- v4 체크포인트: `artifacts/commercial_hwr_stability_20260823_r3_shadow/commercial_hwr_stability_checkpoint.pt`
  - SHA-256: `ba5cf14bbec6f8a4bc9192717044437331fc7aa257b85f49689d1eb2df946a66`
- v4 보고서: `artifacts/commercial_hwr_stability_20260823_r3_shadow/stability_report.json`
  - SHA-256: `9f8f3e368643d145bd0922254c23d6c71699efc50b6068d7fb3aa35ac861dd4f`
- 기존 186글자 진단: `artifacts/commercial_hwr_stability_known_diagnostic_20260823_r2/known_diagnostic_report.json`
  - SHA-256: `733dd8499e6e2f79d6ffb20a65e17790ab6e1948b5b157856cde6ab52735a081`
- 최종 학습 경계 감사: `artifacts/training_boundary_audit_20260823_r12_stability_final/training_boundary_audit.json`
  - SHA-256: `9fe818195045eb5b5eadfb456de133c58107a54cabb49337c28fbd8ff66622cd`

## 검증

- v4 체크포인트 57개 state tensor strict reload 통과
- v2/v3 `alpha=0.8` 직접 재계산과 v4 state tensor 57개 완전 일치
- Python compile 및 v4 self-test 통과
- 신규 함수 한국어 docstring 누락 0건
- 공개 데이터셋 빌더 단위 테스트 1건 통과
- 학습 경계 감사: trainer 19개, corpus builder 2개, 실패 0, selection finding 0

최종 상태: **`shadow_pending_new_untouched_acceptance` — 상용 배포 안 함**
