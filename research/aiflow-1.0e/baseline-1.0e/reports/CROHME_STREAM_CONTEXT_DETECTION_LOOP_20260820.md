# CROHME 스트리밍·Top-5 회수·필수 문맥 검출 루프

## 결론

- CROHME 합성 시간축은 실제 timestamp가 있는 프로젝트 소유 211글자·3 writer에 가장 가까운 `point_ordinal`을 선택했다.
- 직접수집 387글자·7 writer로 출력 head를 추가 학습한 확장 HWR은 CROHME test Top-1 `71.38→72.70%`, Top-5 `93.29→93.75%`로 개선했다.
- 확장 HWR 후보로 문맥망을 다시 학습했다. 고정 211글자 writer-LOO에서 HWR `88.15%`가 문맥 결합 `96.21%`로 상승했고 17글자 개선·회귀 0이었다. 식 exact는 `61.70→85.11%`였다.
- CROHME test에서는 확장 HWR `72.70%`가 문맥 결합 `75.36%`, 평가 가능 식 exact `16.96→22.61%`로 상승했다. 후보 밖 생성은 0건이다.
- 다만 이전 HWR+문맥 경로의 같은 test 결과 `76.77% / 25.56%`보다 낮다. 따라서 **확장 HWR의 candidate recall 개선은 채택 후보**, 새 문맥 checkpoint는 **상업용 shadow 후보**로 유지한다. 새 writer/formula untouched acceptance 전에는 1.0 기본값으로 승격하지 않는다.

## 고정 구조

```text
온라인 stroke exact-cover grouping
  -> box-local 372-class HWR Top-5 detection
  -> 필수 2-layer attention 문맥망
  -> 각 위치의 기존 Top-5 중 하나만 선택
  -> 2D 공식 배치·보수 guard
  -> 확신 부족 시 raw/Top-5 보존
```

- HWR와 문맥망은 gradient를 주고받지 않는다.
- 문맥 checkpoint가 없거나 HWR hash가 다르면 hard error다.
- 문맥망은 문자 삽입·삭제, stroke regrouping, 후보 밖 문자 생성, 산술 역산을 할 수 없다.
- 따라서 구조상 `Top-5 candidate detector + conditional selector`이며 일반 raster object detector는 아니다.

## 0. 실제 사용자 시간과 CROHME 합성 시계

| 합성 시계 | record 평균 MAE | writer macro MAE | 최악 writer MAE |
|---|---:|---:|---:|
| `point_ordinal` | 0.0507 | **0.0563** | 0.0753 |
| `pen_down_arc` | 0.1067 | 0.1017 | 0.1129 |
| `global_chord_arc` | 0.0869 | 0.0829 | 0.0929 |
| `stroke_equal_arc` | 0.0992 | 0.0962 | 0.1032 |
| `ordinal_chord_hybrid_50` | 0.0572 | 0.0574 | **0.0589** |

`point_ordinal`이 writer macro MAE 최저라 선택됐다. writer-LOO 3개 중 2개 fold는 이 방식을, 1개 fold는 hybrid를 골랐다. CROHME에는 timestamp가 없으므로 실제 속도 복원이 아니라 사용자 필기와 가장 가까운 합성 근사다.

확장 HWR의 한 타점 재생 결과:

| 범위 | 최종 Top-1 | 최종 Top-5 | 시간가중 Top-1 AUC | 시간가중 Top-5 AUC | 안정 정답 도달 중앙값 |
|---|---:|---:|---:|---:|---:|
| 직접 211글자, 실제 timestamp | 88.15% | 99.05% | 19.91% | 31.76% | 86.30% |
| CROHME test, `point_ordinal` | 72.70% | 93.75% | 23.61% | 35.68% | 78.79% |

AUC는 wall-clock 지연이 아니라 전체 궤적 진행률 가중 정확도다. 초반에는 bbox와 형상이 완성되지 않아 CROHME Top-1이 50% 진행에서 14.02%, 80%에서 42.77%, 100%에서 72.70%였다.

## 1. Top-5 밖 정답 순위

- 전체 11,991글자 중 Top-1 오답 3,274개.
- 오답 중 정답이 Top-5 안 2,525개, 밖 749개.
- Top-5 밖 정답 순위 평균/중앙값/p90/최대: `28.48 / 12 / 68 / 359`.
- Top-5 밖 749개 중 rank 10 이내 340개, 20 이내 509개, 50 이내 632개, 100 이내 708개다.
- Top-5 밖이어도 같은 동형 family의 다른 문자가 후보에 있는 경우는 120/749(16.02%)다. 이 경우 문맥망은 family는 볼 수 있어도 정답 자체가 후보에 없어 복원할 수 없다.

상위 누락 truth:

| truth | Top-5 밖 | 중앙 rank | 해석 |
|---|---:|---:|---|
| `2` | 101 | 19 | 폭넓은 shape/domain shift |
| `1` | 99 | 9 | 수직획 family |
| `+` | 59 | 21 | 외부 학습 81개로 부족 |
| `z` | 54 | 56.5 | 학습량보다 필기체 domain shift |
| `0` | 53 | 9 | 원·타원 family |
| `x` | 42 | 11.5 | 교차획·라틴 혼동 |
| `=` | 41 | 19 | 승인 외부 학습 0개, 프로젝트 소유 보정 의존 |

## 2. 동형문자 Top-5 회수

- 동형 오답 681개 중 정답 Top-5 포함 607개(89.13%), 밖 74개.
- 동형 누락 74개의 rank 평균/중앙값은 `9.39 / 9`; 53개가 rank 10, 72개가 rank 20 이내다.
- family별 Top-5 포함률: cross 98.53%, equality/approx 97.96%, vertical/slash 87.36%, circle 74.43%, S/5 90.91%.
- 문맥 결합 후 CROHME test 동형 truth 범위는 `57.99→64.78%`였다. circle `30.36→51.07%`, equality/approx `75.19→80.93%`로 개선했지만 cross는 `59.98→58.12%`로 회귀했다.

따라서 다음 HWR 수집 우선순위는 `1`, `0`, `2`, `+`, `=`, `z`, `x`이며, 특히 `=`은 승인 외부 단독문자 데이터가 없어 새 writer의 프로젝트 소유 양성/음성 궤적이 필요하다.

## 3. 문맥 신경망 학습·평가

- 구조: hidden 128, 2 layers, 2 attention heads, 395,769 parameters.
- 입력: 현재 formula 순서, 각 glyph의 고정 HWR Top-5와 확률, 7개 공간 관계, semantic role.
- 학습: 프로젝트 소유 47식·211글자·3 writer + 결정론적 수식 DSL 16,000 예제/fit.
- 선택: 바깥 writer-LOO, 안쪽 formula-disjoint epoch/가중치 선택. CROHME는 학습·선택에 미사용.
- 선택 epoch: 1. context/shape/grammar weight `0.25/0.5/1.0`.

| 평가 | HWR | 문맥 결합 | 식 exact HWR→문맥 | 개선/회귀 |
|---|---:|---:|---:|---:|
| 직접 writer-LOO 211글자 | 88.15% | **96.21%** | 61.70%→**85.11%** | 17/0글자 |
| CROHME valid 비상업 진단 9,535글자 | 70.61% | **73.55%** | 16.46%→**21.65%** | 465/185글자 |
| CROHME test 비상업 진단 11,991글자 | 72.70% | **75.36%** | 16.96%→**22.61%** | 474/155글자 |

CROHME test 식 exact는 849개 완전지원 식에서 `144→192`; 개선 59식, 회귀 11식이다. 전체 protocol 1,199식 분모로는 16.01%다. CROHME truth grouping을 공급한 진단으로 공식 symLG/LgEval 또는 end-to-end 검출 정확도가 아니다.

## 4. 추가 데이터·사전학습 조사

- HF x/y/t 후보 3종은 형식은 가까우나 라이선스·원출처·writer/session provenance가 없어 반입하지 않았다.
- MathWriting/CROHME/IAM/CASIA 온라인 데이터는 비상업 또는 연구 조건이라 제품 학습에 사용할 수 없다.
- AI Hub 공개 수학 필기 자료는 현재 metadata상 raster OCR 자료다.
- UEA CC BY 4.0 Handwriting은 3축 가속도라 2D stroke HWR에 부적합하다.
- MathBERT는 모델 코드 라이선스와 학습 corpus 권리를 분리 감사해야 하므로 상업 runtime에 넣지 않았다.
- Google BERT-Tiny를 프로젝트 소유 수식 584개로 파인튜닝한 기존 실험은 2 epoch 비회귀였지만 현재 159식 exact 개선이 0이었고 3~4 epoch는 2식 회귀했다.

세부 내역: `datasets/20_reaudit_required/HF_ONLINE_HWR_SEARCH_AUDIT_20260820.md`.

## 5. raw-stroke end-to-end 연결 확인

새 HWR·문맥 hash에 묶인 partition ranker를 다시 생성한 뒤 프로젝트 소유 raw 110식을 실제 런타임으로 통과시켰다.

- `grouping -> HWR Top-5 -> 필수 문맥망 -> layout/finalizer` 실행 성공 110/110식.
- 모든 stroke exact-cover 110/110식.
- 문맥망이 37식·51글자를 Top-5 안에서 재선택했다.
- 후보 밖 토큰, 문자 삭제, 문맥 단계 grouping 변경은 모두 0건.
- HWR/context hash는 각각 `04f860...2d00e`, `125dc9...671a`로 전 식에서 일치했다.
- 출력: `artifacts/context_detection_runtime_smoke_20260820_r1_shadow/raw_runtime_output.json`, SHA-256 `6371c2e94bd5b08a391b09bb6c0759334dc6bba267421e91e46db47d59f4969c`.

이 110식은 기존 학습·개발과 겹치므로 정확도 근거가 아니라 연결·무결성 smoke test다. 새 partition ranker의 뒤 49식 수치도 확장 HWR head가 해당 데이터 일부를 이미 본 조건이라 독립 정확도로 사용하지 않는다. 기존 sequence/syntax guard는 이전 checkpoint hash에 묶여 있어 이번 최소 runtime에 이식하지 않았으며, 새 pair 기준 재학습 전까지 비활성이다.

## 판정

1. 입력 시계: `point_ordinal` 유지. 시간은 분류 feature가 아니라 streaming progress/latency 표시용으로 분리한다.
2. HWR: 확장 head를 1.0 candidate-recall challenger로 선택한다.
3. 문맥: 호출 필수, Top-5 선택 전용, 분류기와 gradient 비결합으로 고정한다.
4. checkpoint: 확장 HWR와 새 문맥망 조합을 commercial shadow로 보존한다.
5. 승격 조건: 새 writer·새 formula의 프로젝트 소유 untouched 세트에서 glyph Top-1/Top-5, 식 exact, 동형 family macro, 회귀 0을 다시 통과해야 한다.

## 재현 산출물

- 스트리밍·rank·문맥 감사: `artifacts/crohme_stream_rank_context_expanded_20260820_r2_shadow/stream_rank_context_audit.json`
- 한 타점 prefix: `artifacts/prefix_48hz_expanded_hwr_crohme_test_20260820_r1_shadow/prefix_48hz_report.json`
- 문맥 학습 보고서: `artifacts/owned_formula_context_expanded_hwr_20260820_r1_shadow/owned_formula_context_report.json`
- 확장 HWR SHA-256: `04f8608aebcf6c02d45ad6f5735229b9eaa2c4b4e1be0db4793d02273ef2d00e`
- 새 문맥 checkpoint SHA-256: `125dc9cd1457a46153522f9041117d320c06071a7f71d65bf7929b5ef4c0671a`
- 최종 감사 JSON SHA-256: `d1f3c98030f16f55081a2c1bcd12becb14e6c85183918fb0c991950881c387d6`

핵심 실행:

```powershell
python -s scripts\audit_crohme_stream_rank_context_v1.py `
  --prefix-predictions artifacts\prefix_48hz_expanded_hwr_crohme_test_20260820_r1_shadow\crohme_prefix_predictions.jsonl.gz `
  --direct-prefix-predictions artifacts\prefix_48hz_expanded_hwr_crohme_test_20260820_r1_shadow\direct_prefix_predictions.jsonl.gz `
  --hwr-checkpoint <expanded-project-symbol-head.pt> `
  --context-checkpoint artifacts\owned_formula_context_expanded_hwr_20260820_r1_shadow\owned_formula_context_product.pt `
  --output <new-D-drive-output> --device cuda
```
