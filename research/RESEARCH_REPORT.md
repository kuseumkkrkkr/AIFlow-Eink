# AIFlow Math Ink 0.6 — 행동 문맥 모델 전환 보고서

## 결론

0.6의 남은 정확도는 고립 기호 분류기만 더 학습해서 해결되지 않는다. 동일한 교차획이 `x`, `X`, `\times`가 될 수 있고, 동일한 대문자도 분자·분모·첨자처럼 축소된 수식 슬롯에서는 식 전체 높이와 비교할 수 없기 때문이다.

따라서 다음 책임 분리를 채택한다.

1. trajectory/raster 신경망은 형태 후보 top-k와 확률을 만든다.
2. Tray·관계 레이어는 분자·분모·첨자·피개항 등 로컬 좌표계를 만든다.
3. 행동 문맥 레이어는 획 행동, 로컬 크기, 이웃 후보, 간격, Tray 역할을 사용해 같은 형태군 안의 의미만 재점수화한다.
4. 수식 gate와 LaTeX 문법은 그 뒤에서 작동한다.
5. 행동 증거가 부족하면 원래 top-k를 그대로 반환한다.

```text
touch / raster
→ 128×128 정규화·trajectory encoder
→ 기호 top-k
→ segmentation lattice·Tray·2D relation
→ 행동 문맥 모델
   ├─ 로컬 슬롯별 대소문자
   ├─ x / X / \times 의미 역할
   └─ O / o / 0 등 hard-family 후보 보존
→ math/non-math gate
→ Formula Expert·LaTeX
```

행동 모델은 glyph 인식기를 덮어쓰는 새 OCR이 아니다. 신경망이 보존한 후보군 안에서 사용자가 그 기호를 어떤 역할로 배치했는지 추정하는 작은 후단 모델이다.

## 지금까지 확인된 연구 결과

### Online/shared encoder

- 승인 HWRT·UJI Pen v1/v2·Pendigits 기반 128-step·19-channel 경로를 구축했다.
- seed 17·31·47 paired test 평균은 exact top-1 `83.00%`, top-5 `97.78%`, family top-1 `91.37%`다.
- logit ensemble은 exact `83.82%`, top-5 `98.04%`, casefold `90.93%`, shape-family `93.02%`다.
- exact 오류의 `43.95%`가 대소문자만 다른 오류이고 `56.86%`가 같은 shape family 안의 오류다.
- 이는 형태 후보는 상당 부분 살아 있으나 최종 의미 배정이 병목이라는 근거다.

### Raster→virtual stroke

- top-4 virtual stroke와 shared encoder 재입력 경로를 구현했다.
- 후보 다양성은 개선됐지만 최종 raster 정확도는 release gate에 크게 미달했다.
- 3-seed 차이가 작아 seed variance보다 vectorizer와 online feature 분포 정렬이 주 병목이다.
- 따라서 행동 문맥은 raster vectorizer 병목을 감추는 대체물이 아니며, 두 문제를 별도 gate로 측정해야 한다.

### 대소문자 문맥

- `k/p/y`처럼 획 자체가 다른 쌍은 exact head가 담당한다.
- `c/o/s/u/v/w/x/z`처럼 주 차이가 상대 높이인 쌍만 context head가 담당한다.
- 기존 행 전체 기준 proxy는 validation `70.31→72.14%`, test `62.19→67.34%`로 개선됐지만, 연속 수식 P 데이터의 제품 근거는 아니다.
- 행 전체 기준은 축소된 분수·첨자 안의 대문자를 소문자로 내릴 수 있으므로 이번 구현에서 로컬 슬롯 기준으로 교체했다.

### 구조·Tray

- 분수선은 완성된 분자·분모가 있을 때만 `FRACTION` Tray를 만든다.
- 근호는 radicand와 선택적 degree, 적분은 integrand와 선택적 upper/lower/differential을 분리한다.
- `=` 인식은 좌우 식의 완성 여부와 독립적으로 보존한다.
- Tray는 기호 이름을 결정하지 않고 기호가 놓인 수학적 슬롯과 로컬 기준선만 제공한다.

## `x`, `X`, `\times` 판단

`2 ? 3`이라는 이웃 정보만으로 가운데를 곱하기로 확정할 수 없다. 사용자가 변수 `x` 또는 문자 `X`를 숫자 사이에 쓸 수도 있기 때문이다. 이 모호성은 문법만으로 완전히 제거할 수 없다.

현재 행동 레이어는 다음 원칙을 적용한다.

- 양옆 숫자는 보조 feature로만 사용한다.
- 교차획 시각 확률, infix 점유, 연산자형 간격, 명시적 operator context 중 하나 이상의 구조 증거가 없으면 no-op이다.
- 조건을 통과해도 `x/X/\times` family의 총 logit 질량을 보존한 채 `\times` prior만 더한다.
- 원 신경망의 `X` 증거가 충분히 강하면 행동 prior가 뒤집지 못한다.
- 본질적으로 모호한 입력은 top-k와 낮은 confidence로 반환하며 거짓 확정을 하지 않는다.

향후 학습 feature는 다음과 같다.

- 원 신경망의 family logits와 entropy
- 앞뒤 token의 digit/identifier/operator 확률 질량
- 좌우 간격, 중심선 정렬, 로컬 높이비
- 교차획의 방향·획순서·pen-up·속도·시간 간격
- segmentation lattice의 infix 점수
- Tray ID·slot·관계 edge
- observed/canonical timestamp와 missing mask

학습 target은 원 glyph 이름 하나가 아니라 `identifier_lower`, `identifier_upper`, `multiply_operator`, `other/abstain` 의미 역할로 둔다.

## 분수·첨자 안의 대소문자

대문자 여부는 식 전체 canvas 높이와 비교하지 않는다.

- main row
- fraction numerator
- fraction denominator
- superscript/subscript
- root degree/radicand
- integral upper/lower/integrand

각 슬롯을 별도의 `local_context_id`로 묶고, 같은 슬롯의 신뢰 가능한 숫자·형태 문자만 높이 anchor로 사용한다. 슬롯에 anchor가 없으면 case logits를 보존한다. 이 원칙은 작은 분자 안의 `X`를 main row보다 작다는 이유로 `x`로 강등하는 오류를 막는다.

## 구현

`behavior_context06.py`를 추가하고 `math_context_layer.py` 앞단에 연결했다.

공개 입력 계약은 record별 다음 선택 feature를 사용한다.

```json
{
  "behavior_context": {
    "local_context_id": "fraction:f1:numerator",
    "cross_visual_probability": 0.91,
    "infix_occupancy": 0.94,
    "operator_spacing_score": 0.82,
    "explicit_operator_context": 0.0
  }
}
```

Tray가 있으면 `tray_id + tray_slot`, 일반 행이면 `baseline_group`, 아무 정보도 없으면 `main`을 사용한다. `FRACTION.numerator/denominator`와 구조 관계의 `SUPERSCRIPT/SUBSCRIPT/ABOVE/BELOW/CONTAINS`는 별도 수작업 없이 로컬 context로 자동 변환한다.

회귀 검증은 다음 네 항목을 고정했다.

1. 축소된 분자 안의 `X` 보존
2. 숫자 이웃만으로 `\times` 강제 금지
3. 획·infix·간격 증거가 겹친 애매한 교차획의 소프트 `\times` 선택과 family 질량 보존
4. 강한 원본 `X` logit 보존

관련 context 회귀 테스트는 `15/15`, 전체 Python 회귀 테스트는 `264/264` 통과했다.

## 행동 모델 파인튜닝 계획

### 단계 A — shadow feature 수집

기존 모델 weight는 고정하고 각 기호에 top-k, geometry, stroke behavior, Tray slot을 붙여 저장한다. 이미지나 원 stroke를 서버로 보내지 않는 온디바이스 원칙은 유지한다. 연구 데이터에는 원본 provenance와 writer/source/device split을 보존한다.

### 단계 B — 작은 role head

행동 head는 모바일 student에 붙일 수 있는 작은 TCN/MLP로 시작한다. pixel encoder를 다시 크게 학습하지 않고 다음 multi-task를 사용한다.

- cross semantic role CE
- local case role CE
- Tray slot consistency loss
- 원 family logit 보존 KL
- abstain/calibration loss

초기 loss weight는 validation sweep 대상이며 아직 배포 상수로 고정하지 않는다.

### 단계 C — 실제 수식 검증

합성 조합은 데이터 증강으로만 사용한다. 최종 선택은 writer/device/source-disjoint 연속 수식에서 수행한다.

- `x/X/\times` role top-1·top-k와 confusion matrix
- local slot별 case exact
- unchanged-safe rate: 행동 증거가 없는 표본의 원 top-1 보존율
- family oracle 대비 회수율
- ECE와 abstain coverage/accuracy
- main/numerator/denominator/script slice 최저 성능

비상용·용도 제한 데이터는 연구 평가에만 사용하고 배포 checkpoint 학습에는 포함하지 않는다.

## 현재 판단

- 행동 모델 방향 채택: **예**
- 기존 case layer를 로컬 슬롯 행동 layer로 교체: **예**
- `x/X/\times` hard rewrite: **아니오**
- CROHME R-track 조건부 역할 정확도 상승 확인: **예**
- 이를 제품 수치로 주장: **아니오**
- 0.6 상용 카나리: **아직 아니오**

## 첫 GPU 행동학습 결과

기존 3-seed glyph teacher를 각각 동결하고 `128×19 stroke TCN + 49차 문맥 MLP` 행동 head를 GTX 1650 CUDA에서 학습했다. CROHME 2012 `trainData`는 writer 기준 fit 1,063식/validation 275식으로 나눴고 `testDataGT` 488식은 공식 held-out으로만 한 번 평가했다.

역할 분모는 다음과 같다.

| split | lower identifier | upper identifier | multiply operator |
|---|---:|---:|---:|
| fit | 1,573 | 99 | 177 |
| validation | 401 | 22 | 44 |
| official test | 568 | 31 | 36 |

head 입력에는 다음을 사용했다.

- seed별 frozen 0.6 teacher의 lower/upper/times logits
- target 원본 stroke의 순서·방향·pen-up을 보존한 128×19 sequence
- 좌우 이웃의 정답 label이 아닌 teacher 예측 역할 확률
- bbox, 간격, 중심선, 로컬 높이·폭 비율
- 19개 cross visual feature와 stroke/point 수

| 지표 | seed 17 | seed 31 | seed 47 | 평균 ± 표준편차 |
|---|---:|---:|---:|---:|
| teacher role accuracy | 74.33% | 60.79% | 60.94% | 65.35 ± 6.35%p |
| behavior role accuracy | 93.07% | 93.86% | 93.07% | **93.33 ± 0.37%p** |
| behavior macro-F1 | 76.64% | 77.17% | 75.42% | **76.41 ± 0.73%p** |
| lower recall | 95.07% | 96.30% | 95.60% | 95.66 ± 0.50%p |
| upper recall | 61.29% | 51.61% | 54.84% | **55.91 ± 4.02%p** |
| multiply recall | 88.89% | 91.67% | 86.11% | **88.89 ± 2.27%p** |
| ECE | 4.73% | 5.09% | 4.82% | 4.88 ± 0.15%p |

평균 accuracy 이득은 `+27.98%p`이고 seed 간 편차도 작아 행동 문맥 방향은 유효하다. 특히 frozen teacher가 세 seed 모두 `\times`를 0% 회수하던 조건에서 행동 head는 86.11~91.67%를 회수했다.

그러나 다음 한계 때문에 제품 승격은 금지한다.

1. 정답 symbol grouping 이후의 조건부 역할 평가다. segmentation 오류는 분모에 없다.
2. CROHME R-track weight라 상용 checkpoint에 병합할 수 없다.
3. official test의 대문자 분모는 31개뿐이며 recall 평균도 55.91%다.
4. 분수·첨자 로컬 case의 실제 device-disjoint 제품 분모가 아니다.
5. 세 teacher별 head는 약 75.8KB지만 distillation/온디바이스 변환 단계로 넘기지 않는다.

따라서 첫 학습 결론은 `x/X/\times` 의미 역할 head 채택, 대문자 데이터 확장, predicted grouping end-to-end 재평가다. 제품용 다음 학습은 승인된 trajectory를 수학 배치로 합성한 약한 P-compatible proxy와 실제 상용 허용 연속식 데이터를 구분해 사용해야 한다.

## 후속 루프 — 합성 대문자와 predicted grouping

### 승인 trajectory 분모

HWRT·UJI Pen v1/v2의 writer-disjoint training split에는 상대크기군 각 대·소문자가 약 90~111개씩 있다. 이를 실제 stroke 그대로 사용하고 수식 행의 높이·좌우 숫자 anchor만 합성한 P-compatible proxy를 만들었다.

첫 후보는 8개 case family 1,590개를 loss weight 0.35로 추가했다. 두 번째 후보는 실제 CROHME fit 정규화 통계를 고정하고 `c/x/z` family의 대문자 284개와 소문자 77개만 weight 0.15로 추가했다.

| seed 17 official test | 원 behavior | 전체 proxy | targeted proxy |
|---|---:|---:|---:|
| accuracy | 93.07% | 93.39% | 93.39% |
| macro-F1 | 76.64% | 75.62% | 76.45% |
| upper recall | **61.29%** | 51.61% | 54.84% |
| multiply recall | 88.89% | 86.11% | 88.89% |

두 후보 모두 목표인 upper recall을 악화시켰다. 고립기호를 인위적으로 키운 데이터는 실제 연속식의 대문자 행동 분포를 대체하지 못하므로 조기 기각했고 seed 31·47로 확장하지 않았다. 작은 전체 accuracy 상승을 근거로 채택하지 않는다.

로컬 MathWriting 2024 excerpt는 train/valid/test 각 100식과 symbols/synthetic 각 100개가 있지만 formula InkML에 symbol-to-stroke group 정답이 없다. LaTeX 문자열과 전체 stroke만으로 대문자 역할 target을 강제 정렬하면 label·필순을 발명하게 되므로 supervised behavior 학습에 넣지 않는다.

### Predicted grouping 감사

공식 test 488식의 행동 대상 635개를 validation에서 고정한 lattice selector로 다시 평가했다.

| grouping 경로 | 전체 target | lower | upper | `\times` | `x` exact group |
|---|---:|---:|---:|---:|---:|
| base OCR fusion | 81.89% | 79.93% | 96.77% | 100% | 75.56% |
| Tray/infix joint | 81.42% | 79.58% | 93.55% | 100% | 75.11% |
| cross-visual + Tray/infix, gap 20% | 91.02% | 90.32% | 93.55% | 100% | 88.89% |
| full selector, gap 50% | 93.39% | 92.96% | 93.55% | 100% | **92.22%** |
| **protected family selector, gap 40%** | **92.91%** | **92.43%** | 93.55% | 100% | 91.56% |

base에서 실패 115건 중 `x`가 110건이고, 그중 96건이 두 획 oversplit이었다. 최초 cross-visual 결합은 전체 oversplit을 98→42건으로 줄이고 `x` exact grouping을 `+13.33%p` 올렸다. 반면 cross model 없는 Tray/infix 단독은 소폭 악화돼 행동 grouping 경로로 채택하지 않는다.

현재 병목은 다음처럼 분리된다.

1. `x`: 최종 결합 후 grouping recall은 91.56%로 90%대를 유지하며 잔여 38건은 oversplit 26, overmerge/혼합 12다.
2. `X/C`: upper 역할 grouping은 93.55%이므로 주 병목은 behavior role head와 실제 연속식 대문자 데이터다.
3. `\times`: grouping 100%, conditional role recall 88.89%이므로 의미 head가 병목이다.
4. grouping과 role 오류가 독립이라고 가정한 근사 end-to-end recall은 lower 88.42%, upper 52.31%, multiply 88.89%다. 이는 실측 end-to-end 수치가 아니므로 제품 지표로 사용하지 않는다.
5. 전체 488식 exact partition은 60.04%이며 fraction·superscript·단일기호 overmerge가 다음 segmentation 병목이다.

따라서 R-track grouping 기준선은 `equality + cross-visual + protected multistroke family + Tray/infix`, cross gap ratio `0.40`으로 갱신한다. 대문자는 합성 isolated proxy가 아니라 실제 symbol-group annotation이 있는 연속식만 다음 supervised 학습 후보로 허용한다.

### x prefilter 병목과 gap validation sweep

실패를 candidate 생성→cross head→exact-cover selector로 분해했다. 두 획 `x`의 lattice candidate recall은 validation/test 모두 100%이고 test cross threshold recall은 98.37%였다. 그런데 실패 후보의 cross 확률 중앙값도 0.929인데 `infix_signal=0`이었다. 원인은 cross head 앞의 bbox gap 20% prefilter가 좌우 곡선 `)(`형 x를 모델에 전달하지 않은 것이다.

전역 threshold와 merge weight를 고정하고 gap ratio `0.20~0.80`만 writer-validation에서 선택했다. Validation winner `0.50`을 equality head까지 포함한 full selector의 공식 test에 적용했다.

| 지표 | gap 20% | **gap 50%** | 변화 |
|---|---:|---:|---:|
| validation `x` grouping | 82.31% | **88.45%** | +6.14%p |
| validation exact partition | 63.27% | **65.82%** | +2.55%p |
| validation pair-F1 | 91.00% | **91.69%** | +0.69%p |
| official test `x` grouping | 88.89% | **92.22%** | +3.33%p |
| official test exact partition | 53.69% | **54.51%** | +0.82%p |
| official test pair-F1 | 88.38% | **88.72%** | +0.35%p |

gap 50%는 이 단계에서 전 지표가 함께 올라 채택했다. 이후 같은 기준에서 threshold `0.5`, weight `10`의 공격 후보는 test `x`를 94.44%로 올렸지만 exact partition 54.51→53.89%, pair-F1 88.72→88.41%, overmerge 21.11→23.57%로 악화돼 기각했다. 다획 family boost와 결합한 validation 재선택에서는 동일 test 성능을 내는 더 좁은 gap `0.40`이 선택됐다. 최종 cross 값은 gap `0.40`, threshold `0.692841...`, infix weight `8`이다.

### 다획 family must-link와 geometry 보호

`sum/pi/rightarrow/neq/pm`은 candidate recall이 validation/test 모두 100%였지만 합친 OCR alias가 기존 reliable 목록과 연결되지 않거나 3개의 singleton 점수에 밀렸다. `Sigma/pi-family/arrow-family/neq/pm` 후보에만 추가 symbol signal을 주고 boost `0~6`을 writer-validation에서 선택했다.

| 공식 test grouping | 기준 | boost 6 + 보호 |
|---|---:|---:|
| 다섯 family 전체 | 45.19% | **82.96%** |
| `\sum` | 25.81% | **70.97%** |
| `\pi` | 40.00% | **90.00%** |
| `\rightarrow` | 71.43% | **89.80%** |
| `\neq` | 23.08% | **76.92%** |
| `\pm` | 25.00% | **75.00%** |

무보호 boost 6은 전체식 exact 57.38%까지 올렸지만 `hookrightarrow`, 좁은 `Sigma`, 세로로 긴 `pm` 오인이 단일기호를 먹었다. Training writer-validation의 실제 geometry 범위에서 arrow `pair_gap_max≤0.50`, `Sigma width_ref≥2.0`, `pm height_ref≤2.0` guard를 만들고 ontology 밖 `hookrightarrow` alias를 제거했다. 보호 후 공식 test exact는 **60.04%**, pair-F1은 **91.07%**, overmerge formula rate는 21.72%다. Gap-only 대비 각각 +5.53%p, +2.35%p이며 overmerge 증가는 +0.61%p로 제한됐다.

### 최종 잔여 segmentation 병목

최종 full selector의 488식 잔여 실패는 exact 293, oversplit 89, overmerge 72, 혼합 34다. 구조별 exact는 sqrt 68.00%, plain 63.95%, subscript 57.01%, superscript 52.75%, fraction 52.10%다. 3획 기호 correct는 34→63개로 늘었고 oversplit은 57→32개로 줄었다.

| label | 오류/전체 | 주 실패 |
|---|---:|---|
| `x` | 38/450 | oversplit 26, overmerge·혼합 12 |
| `1` | 36/482 | 주변 획과 overmerge |
| `-` | 36/584 | overmerge |
| `\sin` physical part | 35/297 | overmerge 15, oversplit 18 |
| `\sum` | 9/31 | oversplit 8 |
| `i` | 9/41 | oversplit |
| `F` | 7/10 | oversplit |
| `\div` | 7/11 | oversplit |

따라서 다음 우선순위는 전역 threshold가 아니라 분수·위첨자에서 `1/-/괄호`가 이웃 Tray로 넘어가지 않게 하는 local-baseline must-not-link, 그리고 남은 `F/div/i` 다획 family다.

### Local-baseline 감사와 학습형 boundary 행동 head

최종 selector의 overmerge를 정답 기호 단위로 다시 연결했다. 공식 test에는 299개 overmerge event와 149개 고유 선택 후보가 있었다. 오류가 집중된 정답은 `1` 36건, `-` 36건, 괄호 53건이며 구조별로 superscript 150건, fraction 112건, subscript 96건이 겹쳐 있었다. 기존 fraction penalty는 299건 중 4건에만 신호를 냈다.

두 규칙형 가드는 채택하지 않았다.

1. 합친 OCR이 구성획보다 약할 때 감점하는 component competition은 validation exact를 개선하지 못하고 family grouping을 악화시켜 weight 0을 선택했다.
2. base partition에서 `SUPERSCRIPT/SUBSCRIPT/ABOVE/BELOW/CONTAINS` edge를 만든 local-baseline guard는 validation 3,293개·test 5,061개 후보에 신호를 만들었지만 실제 149개 오병합 후보에는 유효 신호가 0개였다. 이미 오병합된 partition 뒤에서 관계를 만들었기 때문에 내부 경계를 복원할 수 없었다.

이에 grouping 이전의 모든 다획 후보를 대상으로, 둘 이상의 정답 기호를 침범하는지를 예측하는 geometry boundary 행동 head를 fit writer에 학습했다. 72,223개 후보 중 66,842개가 boundary 양성이며, validation에서 threshold 0.5·penalty weight 6을 선택했다.

| 지표 | 보호 selector | boundary 행동 head | 변화 |
|---|---:|---:|---:|
| validation exact partition | 68.00% | **69.82%** | +1.82%p |
| validation pair-F1 | 92.98% | **93.64%** | +0.66%p |
| official exact partition | 60.04% | **60.25%** | +0.20%p |
| official pair-F1 | 91.07% | **91.26%** | +0.19%p |
| official overmerge formula | 21.72% | **20.49%** | -1.23%p |
| official `x` grouping | 91.56% | 91.33% | -0.22%p |
| official 다획 family grouping | 82.96% | 81.48% | -1.48%p |

사전 정의한 `x` -1%p, family -2%p, pair-F1 -0.25%p 보호 gate 안에서 exact와 pair-F1이 함께 올라 R-track head로 채택한다. 다만 CROHME 정답 경계로 학습한 비상업 연구 weight이므로 제품 checkpoint·teacher·distillation에는 사용할 수 없다. 상용 이전 경로는 동일 boundary target을 P-track 연속식으로 재학습하거나 shared trajectory encoder의 auxiliary boundary loss로 옮기는 것이다. 전체 Python 회귀는 271개가 통과했다.

### P-track boundary auxiliary smoke

> 2026-07-24 정정: 아래 최초 smoke는 adapter checkpoint의 `shared_state_dict`를 적용하지 않은 base-only 결과이므로 현재 메인 모델 성능 근거에서 철회한다. 실험 이력으로만 보존한다.

별도 sklearn guard를 최종 구조로 고정하지 않고 `MathInk06Model`의 shared 384차 trajectory embedding 위에 선택적 binary boundary head를 추가했다. 기본값은 비활성이라 기존 checkpoint key와 `forward() → exact,family` 계약은 변하지 않는다. 활성 경로만 online/raster candidate별 boundary logit을 반환하며 class imbalance용 binary auxiliary loss를 제공한다.

CROHME weight를 이전하지 않고 승인 paired training trajectory의 실제 단일기호를 음성으로, 서로 다른 두 기호를 같은 행·위첨자·아래첨자 위치에 배치한 후보를 양성으로 합성했다. Seed-17 base와 online adapter는 동결하고 head만 학습했다.

| P proxy validation | 결과 |
|---|---:|
| balanced validation samples | 2,400 |
| validation accuracy | 97.33% |
| validation F1 | 97.34% |
| validation ROC-AUC | 99.65% |
| validation single-symbol recall | 97.25% |
| validation cross-boundary recall | 97.42% |
| paired-test single/boundary recall | 96.92% / 98.00% |
| exact/family logit max abs 변화 | 0.0 / 0.0 |

Validation에서 single≥95%·boundary≥85%를 동시에 요구해 threshold 0.70을 선택했고, 선택에 사용하지 않은 paired writer/device-disjoint proxy test 2,400개에서 accuracy 97.46%, F1 97.47%, AUC 99.62%를 기록했다. 이는 P-track target 재구성이 가능하다는 smoke 증거지만 실제 연속식 writer/device-disjoint 검증은 아니다. 다음 단계는 실제 P 연속식 또는 사용자 local-only pseudo formula에서 symbol boundary annotation을 확보하고 main joint fine-tuning weight를 선택하는 것이다. 전체 Python 회귀는 275개가 통과했다.

### 5-layout P boundary joint — 3 seed

> 2026-07-24 정정: 아래 최초 joint 결과도 동일한 shared-state 로딩 누락을 상속했으므로 delta 채택을 철회한다. 공개 배포 대상에서 제거한다.

Smoke의 배치를 같은 행·상첨자·하첨자에서 분자/분모 수직 슬롯과 넓은 중위식 양변까지 다섯 종류로 확장했다. Seed별 base·online adapter·boundary head를 독립 초기화하고 다음 module만 열었다.

- `trajectory_encoder.blocks.3`
- `trajectory_encoder.attention`
- `exact_head`, `family_head`, `boundary_head`

Authentic 4,800개에는 exact CE·family CE·single-symbol boundary loss를 주고, balanced boundary 후보 2,400개에는 boundary loss를 줬다. Validation 채택 조건은 authentic exact/family 각 -0.5%p 이내, single/boundary recall 각 95% 이상이다.

| paired proxy test | seed 17 | seed 31 | seed 47 | 평균 ± 표준편차 |
|---|---:|---:|---:|---:|
| exact top-1 | 77.66% | 79.38% | 77.84% | **78.29 ± 0.77%p** |
| exact 변화 | +3.07%p | +3.20%p | +2.70%p | **+2.99 ± 0.21%p** |
| family top-1 | 86.22% | 87.70% | 86.46% | **86.80 ± 0.65%p** |
| family 변화 | +2.78%p | +3.54%p | +2.62%p | **+2.98 ± 0.40%p** |
| single-symbol recall | 96.75% | 98.25% | 97.25% | **97.42 ± 0.62%p** |
| cross-boundary recall | 98.92% | 97.92% | 98.67% | **98.50 ± 0.42%p** |
| boundary F1 | 97.86% | 98.08% | 97.97% | **97.97 ± 0.09%p** |

세 seed 모두 validation과 paired test gate를 개별 통과했고 exact/family도 모든 seed에서 개선돼 P formula-layout proxy delta로 채택한다. 이는 R-track head를 distill한 결과가 아니라 승인 P trajectory와 알려진 합성 boundary만 사용한 결과다. 다만 분리된 고립기호를 배치한 proxy이므로 실제 사용자의 연속식 stroke rhythm·co-articulation·device sampling 변동은 포함하지 않는다. 제품 검증은 false이며 다음 필수 gate는 실제 P 연속식 writer/device-disjoint boundary 평가다.

### 3-seed device contract stress

채택 delta를 별도 학습 없이 좌표 jitter, timestamp/speed 전체 결측, raw event 절반 희소화 후 6Hz 재보간, x 1.12/y 0.88 affine 조건에서 평가했다. 최초 sparse 실험은 canonical tensor를 직접 절반 삭제해 입력 계약을 위반했으므로 폐기했고, start/end·pen-up anchor를 보존한 재보간 경로로 다시 고정했다.

모든 seed·stress가 clean 대비 exact/family -3%p 이내, single/boundary recall 90% 이상 gate를 통과했다. 최악 exact 하락은 seed-47 affine `-1.53%p`, 최악 family 하락은 seed-31 affine `-1.48%p`였다. Stress 전체의 single-symbol/cross-boundary recall 최저는 각각 96.67%다. Timestamp 결측과 2배 희소 sampling은 최악 exact 하락이 각각 -0.26%p와 -0.48%p 이내였다.

이는 canonical 입력 계약의 변형 내성 증거이지 실제 Android digitizer·WDDM·battery/latency 검증이 아니다. 다음 실제 gate는 raw MotionEvent를 가진 P writer/device-disjoint 연속식이다. 전체 Python 회귀는 276개가 통과했다.

### Shared-state 로딩 결함 정정 및 3-seed 재학습

정식 배포 구성을 감사한 결과 `online_adapter.pt`는 외부 modality adapter뿐 아니라 공동 학습된 `trajectory_encoder`, `exact_head`, `family_head`의 `shared_state_dict`도 저장하지만, P boundary 로더는 외부 adapter만 읽고 shared update를 누락했다. 올바른 합성 순서는 다음과 같다.

```text
base checkpoint
→ adapter shared_state_dict
→ modality adapter
→ 선택적 boundary head
→ 검증을 통과한 경우에만 joint delta
```

로더를 수정하고 이를 보장하는 회귀 테스트를 추가한 뒤 auxiliary와 joint를 seed 17·31·47에서 다시 학습했다.

| paired proxy test | seed 17 | seed 31 | seed 47 | 평균 |
|---|---:|---:|---:|---:|
| 정정된 main baseline exact | 82.95% | 83.24% | 82.92% | **83.03%** |
| joint exact | 82.71% | 82.95% | 82.79% | 82.81% |
| exact 변화 | -0.24%p | -0.29%p | -0.13%p | **-0.22%p** |
| 정정된 main baseline family | 91.22% | 91.51% | 91.38% | **91.37%** |
| joint family | 90.77% | 90.96% | 90.90% | 90.88% |
| family 변화 | -0.45%p | -0.56%p | -0.48%p | **-0.49%p** |
| single-symbol recall | 91.58% | 94.67% | 92.25% | 92.83% |
| cross-boundary recall | 98.50% | 98.75% | 98.17% | 98.47% |

세 seed 모두 official proxy gate를 실패했다. Seed 17·47은 single-symbol 95% floor를, seed 31은 single-symbol floor와 family -0.5%p 비퇴행 조건을 실패했다. 따라서 corrected joint delta는 전부 기각하며 현재 유효 모델은 `base + shared_state_dict + modality adapter`다. 이전 78.29% 결과는 강한 main encoder를 빠뜨린 낮은 기준선과 비교한 것이므로 무효다.

Joint delta 없이 정정된 main과 새 auxiliary head만 대상으로 device stress를 다시 수행했다. Clean exact/family 평균은 각각 83.03%/91.37%다. 모든 software stress가 -3%p와 boundary recall 90% floor를 통과했고, 최악 exact/family 하락은 affine에서 -0.98%p/-1.08%p였다. 이는 auxiliary shadow의 입력 내성만 의미하며 head의 clean single recall이 91.42~94.50%라 release 채택 근거는 아니다.

### Composite torch.export

기존 export script도 base checkpoint만 읽어 실제 main adapter를 우회했다. Online은 `online adapter → shared trajectory encoder`, raster는 `virtual top-4 → raster adapter → shared trajectory encoder`가 그래프 안에 포함되도록 수정했다. Dual adapter는 입력 데이터에 따른 런타임 분기 대신 online/raster branch를 export 시점에 고정한다.

Seed 17·31·47 모두 online/raster 각각 대표 입력 76개에서 eager 대비 top-1 100% 일치, 최대 logit 절대오차 0.0으로 strict `torch.export` gate를 통과했다. 기존 출력 경로를 재사용할 때 오래된 `.pt2`가 남는 문제를 발견해 `.part → atomic replace`로 수정했고, fresh seed-17 두 graph 합계는 24,302,579 bytes로 25MiB 예산 안이다. 다만 `.pt2`는 Android LiteRT artifact가 아니다. 공식 LiteRT Torch 0.9.1은 Linux 환경을 요구하며 WSL 변환 환경 설치 중 WSL filesystem I/O 오류가 발생해 `.tflite` 변환과 runtime parity는 아직 미완료다.

Linux/Colab 변환을 위해 label이 없는 실제 HWRT-derived online/raster 대표 입력 76개를 5,721,772-byte cache로 고정했다. 최소 bundle은 composite base·adapter, 필요한 source, converter와 대표 cache만 포함하며 13개 entry·17,276,876 bytes다. Bundle 내부 SHA-256 전체 검증과 별도 디렉터리 추출 후 composite export smoke를 통과했다. Colab notebook은 공식 `litert-torch==0.9.1` 설치, 공개 Hugging Face bundle 다운로드, 해시 검증, online/raster `.tflite` 변환, 76개 전체 top-1 100%·logit 오차≤0.02·합계≤25MiB gate를 자동 실행한다.

Python 공개 런타임의 `MathInk06Engine`도 base-only였으므로 `adapter_checkpoint`를 공식 생성자 인자로 추가했다. 엔진은 base→adapter `shared_state_dict`→dual modality adapter 순서로 구성하고 online은 `adapter.online`, raster virtual trajectory는 `adapter.raster`를 거친다. 실제 seed-17 representative 입력에서 엔진과 export wrapper의 online/raster 최대 logit 오차는 모두 0.0이었다. Adapter를 생략한 호출은 호환용 base-only 경로로만 남긴다.

### Composite CPU latency·memory proxy

정정된 Python runtime을 실제 representative 76개로 Windows CPU에서 측정했다. 각 thread 조건은 같은 입력과 output checksum을 사용했다.

| Intra-op threads | Online p95 | Raster p95 |
|---:|---:|---:|
| 1 | 9.67ms | 26.63ms |
| 2 | 9.65ms | 24.99ms |
| 4 | 8.90ms | 22.10ms |

세 조건 모두 online 50ms, raster 200ms software proxy를 통과했다. 이 값은 family-fusion 0.15를 포함한 runtime 재측정이다. Model과 composite adapter의 중복 제거 tensor state는 8,947,108 bytes다. 모델 로드 후 관측한 inference RSS 증가분은 17,440,768 bytes이고 두 값을 합친 구조 proxy는 26,387,876 bytes로 100MiB 안이다. 전체 Python process RSS는 CPython·PyTorch runtime을 포함하므로 Android peak memory와 직접 비교하지 않는다. 실제 Android LiteRT 3-tier latency·memory·battery·delegate 검증은 계속 필요하다.

### 실제 P 연속식 데이터 계약

실제 데이터가 도착하기 전에 `AIFlow P Formula v1` JSON Schema와 fail-closed preflight를 추가했다. 각 formula는 다음 근거를 모두 가져야 한다.

- 고유 `formula_id`, 원본 추적용 `origin_id`
- 비어 있지 않은 `writer_id`, `device_id`, `source_id`
- `training|validation|test` split
- `rights_track=P`, `commercial_training_allowed=true`
- 양수 canvas 크기
- token과 원본 raw stroke가 포함된 정답 symbol group

Origin·writer·device·source가 둘 이상의 split에 나타나면 제품 평가를 거부한다. Timestamp와 pressure가 없는 symbol은 삭제하거나 관측값으로 위장하지 않고 missing slice로 센다. 검증된 formula의 실제 symbol group은 boundary 음성, 인접한 두 symbol group 결합은 boundary 양성으로 만든다. 이 계약은 실제 성능값을 만들지는 않지만, 향후 P 입력이 CROHME 정답 group이나 합성 고립기호 proxy와 섞이는 것을 방지한다.

### Online 3-seed 공통 오류와 역할 분리

정정된 composite 세 seed의 logit을 같은 writer-disjoint 분모에서 다시 감사했다. Validation 4,261개에서 seed별 exact top-1은 86.01~87.49%, log-probability ensemble은 87.84%, top-5는 99.53%였다. 선택에 쓰지 않은 paired-test 3,782개에서 seed별 exact는 82.92~83.24%, ensemble은 83.71%, top-5는 98.02%였다. 표본별로 정답 seed를 사후 선택하는 비배포 oracle도 88.05%라 seed 수만 늘려서는 92% exact gate에 도달하지 않는다. 세 seed가 모두 틀린 비율은 11.95%다.

반면 paired-test 예측을 고립 필기에서 구별 불가능한 visual family로 접으면 92.99%다. Exact 오류의 56.98%, 전체의 9.28%p가 같은 visual family 안의 의미 라벨 혼동이었다. 가장 큰 혼동은 `O→0`, `x→X`, `W→w`, `z→Z`, `c↔C`, `s↔S`이고, validation에는 `|→I/l`, `\setminus→\backslash`도 나타났다. 따라서 현재 신경망의 주 병목은 궤적 형태 자체보다 고립기호에 존재하지 않는 상대 크기·수식 역할 정보다.

결론적으로 0.6의 on-device 1단계는 visual family 후보를 안정적으로 반환하고, exact 대소문자·`O/0/o`·`x/X/\times` 의미는 같은 식 행의 크기와 행동 문맥 레이어가 결정해야 한다. 이를 고립기호 exact head에 더 강하게 강제하면 writer 독립 일반화를 악화시킬 가능성이 높다. 제품 92% gate는 `isolated visual-family ≥92%`와 `formula-context exact ≥92%`로 분리해 보고하되, 후자는 실제 P 연속식 없이 통과 처리하지 않는다.

Validation에서만 exact/family fusion weight를 0~0.30 범위로 고정한 결과 0.15가 선택됐다. Paired-test에는 한 번만 적용했으며 exact ensemble 83.71→83.82%(+0.11%p), macro-F1 +0.09%p였다. 개선 폭은 작지만 adapter checkpoint에 기록된 0.15가 Python 공개 runtime과 export graph에서 무시되던 결함을 수정했다. 새 seed-17 strict export는 대표 76개 top-1 100% 일치·logit 오차 0.0, online+raster 합계 24,300,116 bytes로 25MiB 안이다.

### 연속 수식 visual-family 입력 계약과 R-track 적응

고립기호 paired-test에서 92.99%였던 visual-family가 실제 연속 수식에서도 유지되는지 CROHME 정답 symbol group 조건부로 다시 측정했다. 기존 행동 head의 93.33%는 정답 group과 `c/C`, `x/X/\times`, `z/Z` 역할이 주어진 조건부 값이어서, teacher 형태군 인식과 confidence abstain을 포함한 최종 exact를 직접 계산했다.

Seed별 threshold를 trainData writer-validation에서만 고정한 행동 exact gate는 official test에서 teacher exact를 평균 33.65→47.30%(+13.65%p)로 개선했다. Rewrite precision은 81.35%였다. 그러나 teacher visual-family 자체가 평균 53.02%여서 행동 head가 보지 못하는 앞단 domain 병목이 더 컸다.

Baseline 채널 주입을 `formula`, `missing`, `local_full`, `height_only` 네 방식으로 validation 비교했다. 전체 지원기호는 `missing` visual-family 76.55%가 가장 높았지만 행동 대상은 기존 `formula`가 65.52%로 가장 높아 전역 정규화 하나로 해결되지 않았다. 네 branch 중 하나라도 정답인 비배포 validation oracle도 전체 88.06%, 행동 대상 75.16%로 92%에 못 미쳤다. 따라서 경량 selector만 추가하는 안은 기각했다.

구조적 회복 가능성을 확인하기 위해 제품 encoder와 online adapter를 동결하고, 그 뒤에 hidden 64 residual TCN formula adapter만 CROHME R-track truth group으로 학습했다. Recipe는 family CE + exact CE 0.10, baseline-context dropout 0.30이며 seed 17 probe validation에서 고정한 뒤 seed 17·31·47로 재현했다.

| 지표 | 3-seed 평균 | 최솟값 |
|---|---:|---:|
| writer-validation exact | 85.19% | 82.91% |
| writer-validation visual-family | **93.30%** | **93.03%** |
| official test exact | 82.38% | 81.13% |
| official test visual-family | 88.17% | 88.03% |

세 seed 모두 writer-validation visual-family 92%를 넘겼으므로 연속식 적응 구조 자체는 유효하다. 하지만 official test는 세 seed 모두 92%를 실패했고 visual-family 일반화 gap은 평균 5.13%p다. 즉 잔여 병목은 모델 용량이나 정규화 selector보다 unseen writer/domain coverage다. 이 adapter는 CROHME R-track weight이므로 제품 checkpoint·distillation·LiteRT bundle에는 포함하지 않는다. 다음 제품 gate는 같은 recipe를 실제 P 연속식 writer/device-disjoint 데이터에 적용하는 것이다.

### 승인 P 고립기호의 합성 수식 배치 재검증

기존 행동 학습기에 구현돼 있었지만 비활성 상태였던 `P_approved_synthetic_layout_proxy`를 seed 17에서 먼저 검증했다. HWRT/UJI의 실제 trajectory를 유지하고 같은 writer 또는 같은 source의 숫자를 좌우 anchor로 붙였으며, 대·소문자 높이 분포를 20% 겹치게 해 크기만으로 정답이 새지 않게 했다. P proxy 1,590개를 loss weight 0.35로 추가한 결과 official test accuracy는 93.07%로 같았지만 macro-F1은 76.64→73.62%, uppercase recall은 61.29→45.16%, multiplication recall은 88.89→86.11%로 하락했다. 따라서 31·47 seed 확장을 중단하고 checkpoint를 shadow 실패로 격리했다.

정확도만 보고 가중치를 조정하지 않고 실제 CROHME writer-validation 행동 feature와 합성 P proxy의 분포를 역할별로 비교했다. 절대 standardized mean difference(SMD) 0.5를 호환 기준으로 두었으나 최대 layout SMD는 2.82였다.

| feature | 실제 lowercase | P proxy lowercase | SMD | 실제 uppercase | P proxy uppercase | SMD |
|---|---:|---:|---:|---:|---:|---:|
| formula-relative bbox height | 0.347 | 0.712 | +2.58 | 0.473 | 0.937 | +2.28 |
| local height ratio | 1.139 | 0.742 | -1.18 | 1.456 | 0.976 | -1.23 |
| formula-relative bbox width | 0.069 | 0.251 | +2.82 | 0.088 | 0.211 | +2.56 |

고립기호를 숫자 사이에 배치하는 것만으로는 전체 formula bbox, Tray, 첨자와 이웃 부재 분포가 재현되지 않는다. 특히 proxy의 절대 bbox는 실제 연속식보다 두 배가량 크면서도 local height ratio는 반대 방향이라 행동 head에 모순된 신호를 줬다. 이 결과에 따라 승인 P 고립기호는 계속 shape encoder 학습에 사용하지만, 실제 formula bbox와 symbol group이 없는 한 행동 role 또는 formula adapter의 제품 근거로 사용하지 않는다. 다음 입력은 `AIFlow P Formula v1`을 통과한 실제 연속식이어야 한다.

2026-07-24 외부 source를 공식 원문으로 재검증했다. Google MathWriting의 해당 dataset README는 데이터에 CC BY-NC-SA 4.0을 명시하고, CROHME 공식 배포 페이지도 academic/research purpose only와 commercial-use 금지를 명시한다. 두 source 모두 R-track 평가에는 유용하지만 P-track checkpoint에는 사용할 수 없다. 현재 확인된 공개 외부 P 연속식 source는 없으므로 제품 학습 데이터는 AIFlow 소유 수집 또는 별도 상용 라이선스 계약으로 확보해야 한다.

### AIFlow Ink v1 → P Formula v1 사람 확정 경계

외부 P 연속식 부재가 코드 병목으로 남지 않도록 기존 drawer의 local-only `AIFlow Ink v1`을 제품 학습 계약으로 변환하는 경계를 추가했다. 자동 gridding 결과나 모델 예측 token은 정답으로 사용하지 않는다. 별도 annotation JSONL이 다음 값을 모두 제공해야 한다.

- raw record와 정확히 일치하는 `sample_id`
- device family가 아닌 실제 비식별 `device_id`
- `label_status=human_verified`
- 모든 `formula_cell_id → exact token` 전단사

변환기는 모든 raw stroke가 중복 없이 정확히 한 cell에 포함되는지 확인하고 원본 획 순서·timestamp·pressure 결측을 보존한다. 기준 0.6 checkpoint의 `exact_labels`가 중복 없는 378개인지 확인한 뒤 vocabulary 밖 token을 거부한다. 변환된 전체 corpus는 기존 P Formula preflight에서 origin·writer·device·source split 누수 0, training/validation/test 존재, P 권리, 최소 독립 source 조건을 모두 통과해야만 `.part → atomic replace` 방식으로 UTF-8 JSONL을 생성한다. 실패 시 report만 남기고 dataset은 쓰지 않는다.

```powershell
python scripts/materialize_math_ink_06_p_formula.py `
  --input research/data/P_owned/curated `
  --annotations research/data/P_owned/annotations.jsonl `
  --checkpoint research/runs/<seed>/math_ink_06_candidate.pt `
  --output research/data/P_owned/p_formula_v1.jsonl `
  --report research/data/P_owned/p_formula_preflight.json
```

이는 자동 수집 기능이 아니며 서버 전송도 하지 않는다. 사용자가 로컬에서 명시적으로 확정한 group/token만 변환한다. 실제 데이터가 없으므로 제품 성능값은 여전히 만들지 않으며, 이 단계는 다음 3-seed 학습 입력이 잘못된 pseudo label로 오염되는 것을 막는 준비다.

### P-only formula adapter 학습·3-seed gate

Materialize된 P Formula v1을 별도 가공 없이 소비하는 GPU trainer와 3-seed summarizer를 추가했다. 각 symbol은 전체 formula ink bbox를 공통 좌표계로 사용해 128×19로 canonicalize되며, 원본 필순·formula-relative `bbox_top/bottom/height/center_y`·timestamp/pressure 결측 mask를 보존한다.

학습은 product encoder와 online adapter를 동결하고 zero-init hidden-64 residual formula adapter만 연다. Loss는 family CE + exact CE 0.10, context dropout 0.30이며 source 빈도와 exact label 빈도의 역제곱근을 곱한 sampler를 사용한다. Checkpoint 선택은 validation family/visual-family만 사용하고 test는 선택된 state에 한 번 적용한다.

각 seed는 다음 조건을 모두 통과해야 한다.

- exact top-1 ≥92%
- exact top-5 ≥99%
- macro-F1 ≥90%
- writer floor ≥75%
- timestamp·pressure 결측 slice 하락 ≤3%p

Seed 17·31·47 report가 정확히 하나씩 있고 모두 개별 gate를 통과한 경우에만 single mobile student distillation을 허용한다. 세 run의 원본 P Formula JSONL byte-level SHA-256도 반드시 같아야 하며, 빈 fingerprint나 서로 다른 corpus는 요약 단계에서 거부한다. Teacher ensemble 자체의 모바일 탑재는 항상 금지하며, 세 seed가 통과해도 LiteRT parity와 Android 3-tier gate 전까지 `product_validation=false`다.

실제 seed-17 composite와 GTX 1650에서 2/2/2-symbol P fixture로 1-epoch end-to-end smoke를 실행했다. CUDA 학습, validation selection, test metric, 결측 slice, checkpoint/report 생성이 완료됐다. 극소 fixture의 seed gate는 예상대로 실패했고 checkpoint는 프로젝트 산출물·공개본에 포함하지 않았다. 이는 실행 계약 증거이며 성능 근거가 아니다.

### 3-teacher 확률 증류·단일 모바일 student

세 seed가 동일 P corpus의 개별 gate를 통과한 뒤 실행되는 fail-closed distiller를 추가했다. 각 teacher는 base → shared online adapter → P formula adapter 순서로 합성하며, exact vocabulary와 data SHA-256을 다시 대조한다. Temperature 2.0에서 seed별 softmax 확률을 평균하고 exact/family KL과 hard-label CE를 함께 사용해 hidden-64 formula adapter 하나만 학습한다.

Student는 validation visual-family/exact 순서로 한 번 선택하고 test에 한 번만 적용한다. 최종 checkpoint는 teacher weight를 포함하지 않으며 다음 두 조건을 모두 요구한다.

- student 자체 P seed gate: exact top-1 92%, top-5 99%, macro-F1 90%, writer floor 75%, 결측 slice 하락 3%p 이하
- teacher 확률 ensemble 대비 exact top-1·top-5·visual-family 하락 각각 1%p 이하

동일 2/2/2-symbol fixture를 seed 17·31·47로 각각 학습하고 요약한 뒤 GTX 1650에서 student까지 CUDA smoke를 완료했다. Data fingerprint·teacher load order·probability ensemble·student checkpoint/report 생성은 통과했다. 극소 fixture의 정식 student gate는 예상대로 실패했으며 checkpoint는 `D:\AiflowTools` 격리 경로에만 남고 프로젝트·공개 저장소에는 포함하지 않는다. `litert_exported=false`, `product_validation=false`이며 실제 P 성능 근거가 아니다.

### P Formula release 오케스트레이션·student export

정식 P corpus가 도착한 뒤 사람이 여섯 명령을 수동으로 조립하면서 서로 다른 데이터나 checkpoint를 섞지 않도록 단일 release driver를 추가했다. 입력은 동일 P Formula JSONL과 seed 17·31·47의 제품 online adapter이며 실행 순서는 다음과 같다.

```text
seed 17 train + gate
→ seed 31 train + gate
→ seed 47 train + gate
→ same-data 3-seed AND summary
→ probability-ensemble single student distillation + gate
→ student-only strict torch.export
```

각 단계의 report boolean이 `true`일 때만 다음 subprocess가 시작된다. Dry-run도 전체 preflight와 파일 존재 여부를 먼저 검사하고 UTF-8 `release_plan.json`에 정확한 argv와 data SHA-256을 남긴다.

기존 export는 base+online/raster까지만 합성해 새 formula student를 포함하지 못했다. 이를 위해 `online adapter → formula adapter → shared trajectory classifier → family fusion 0.15` 전용 wrapper를 추가했다. Export는 checkpoint schema·P track·세 teacher lineage·동일 data SHA-256·distillation gate·teacher weight 미포함을 다시 확인한다. 실제 P test tensor만 representative로 쓰고 strict eager/export top-1 100%, 최대 logit 오차 0.02 이하, graph 25MB 이하를 요구한다. Gate를 통과하지 않은 fixture student는 CLI export가 차단된다.

구조 진단에서는 실패 fixture checkpoint를 저장·배포하지 않고 메모리상 378-label composite만 합성했다. 두 대표 tensor에서 `torch.export(strict=True)` top-1 2/2, 최대 logit 오차 0.0을 확인했다. 이는 graph 호환성 근거일 뿐 fixture의 release 실패를 뒤집지 않으며 실제 LiteRT flatbuffer와 Android 검증은 계속 미완료다.

### Android ODA runtime·AIFlow Ink v2

기존 저장소에는 Python 엔진만 있고 명세의 Android `recognizeOnline`·`recognizeRaster` 구현은 없었다. `android/aiflow-math-ink-runtime` Android library를 추가해 다음 공개 경계를 실제 Kotlin 코드로 고정했다.

- `recognizeOnline(strokes, canvas): SymbolResult`
- `recognizeRaster(bitmap): SymbolResult`
- `SymbolResult.toServerPayload()`

`AIFlowInkV2`는 `rawStrokes`, `canonicalTaps`, `timestampMode`, `sourceModality`, 로컬 debug용 `virtualHypotheses`를 갖는다. 원본 touch event는 메모리에 보존하지만 공개 server payload는 `candidates`, `confidence`, `modelVersion`, `latencyMs`만 만들며 raw stroke·tap·raster·virtual hypothesis를 포함하지 않는다. Session 호출은 lock으로 직렬화해 native interpreter의 thread-safety에 의존하지 않고 tensor buffer를 재사용한다.

Android `CanonicalTapEncoder`는 관측 timestamp를 6Hz로 재표본화하고 시작·끝·pen-up anchor를 보존한다. Timestamp가 없으면 호길이 기반 canonical time과 `missing_mask=1`을 사용하며 6Hz tap과 TCN 128-step을 분리한다. 첫 구현은 128-step으로 펼칠 때 timestamp를 index로 선형 배분해 Python 대비 speed channel이 달라졌다. 이를 좌표와 동일한 호길이 보간으로 수정했고, Python 학습 전처리의 대표 4행×19채널과 Android 출력이 채널별 절대오차 `1e-4` 이내임을 고정 테스트로 확인했다. Bitmap은 실제 ink bbox를 찾은 뒤 종횡비를 보존해 8px margin의 128×128 중앙에 배치한다.

2026-07-17 갱신된 Google 공식 문서에서 Android의 현대 표준은 `CompiledModel`, 최신 LiteRT는 2.1.6으로 안내된다. 따라서 library는 `com.google.ai.edge.litert:litert:2.1.6`과 Kotlin 2.3.0을 사용하고 APK asset의 online/raster `.tflite` 두 개를 직접 로드한다. ML Kit custom model API나 서버를 거치지 않는다.

Android Gradle Plugin 8.8.2·Gradle 8.12·SDK 35·JBR 21 환경에서 Kotlin unit test 5개가 모두 통과했고 release AAR을 생성했다. AAR은 56,139 bytes, SHA-256 `2b75abe9040c537a04da6f352c139142850f3d84a2b26e2c7dc79fe23bf74bba`다. AAR에는 모델이 없으며 실제 `.tflite`가 아직 생성되지 않아 CompiledModel 실기기 inference, 3-tier latency/memory/battery는 계속 미완료다.

공식 근거:

- `https://developers.google.com/edge/litert/android`
- `https://developers.google.com/edge/litert/next/android_kotlin`

고정 recipe는 `research/configs/MATH-INK-06-P-FORMULA-v1.json`에 저장했다.

전체 회귀는 328개가 통과했다.

## 산출물

- `src/math_grid_drawer/research/behavior_context06.py`
- `src/math_grid_drawer/research/behavior_role_head06.py`
- `src/math_grid_drawer/research/math_context_layer.py`
- `scripts/train_math_ink_06_behavior_role.py`
- `scripts/summarize_math_ink_06_behavior_seeds.py`
- `scripts/audit_math_ink_06_behavior_grouping.py`
- `scripts/audit_math_ink_06_x_edge_bottleneck.py`
- `scripts/sweep_math_ink_06_x_grouping_guard.py`
- `scripts/sweep_math_ink_06_cross_gap_guard.py`
- `scripts/audit_math_ink_06_multistroke_families.py`
- `scripts/sweep_math_ink_06_multistroke_family_guard.py`
- `scripts/audit_math_ink_06_multistroke_guard_regressions.py`
- `scripts/audit_math_ink_06_local_baseline_overmerge.py`
- `scripts/sweep_math_ink_06_component_competition_guard.py`
- `scripts/sweep_math_ink_06_local_baseline_guard.py`
- `scripts/train_math_ink_06_boundary_behavior_guard.py`
- `scripts/train_math_ink_06_p_boundary_auxiliary.py`
- `scripts/train_math_ink_06_p_boundary_joint.py`
- `scripts/summarize_math_ink_06_p_boundary_joint.py`
- `scripts/evaluate_math_ink_06_p_boundary_device_stress.py`
- `scripts/export_math_ink_06_litert.py`
- `scripts/build_math_ink_06_litert_colab_bundle.py`
- `scripts/benchmark_math_ink_06_composite.py`
- `scripts/audit_math_ink_06_online_error_consensus.py`
- `scripts/calibrate_math_ink_06_online_family_fusion.py`
- `scripts/audit_math_ink_06_behavior_exact_gate.py`
- `scripts/audit_math_ink_06_formula_context_contract.py`
- `scripts/train_math_ink_06_formula_adapter.py`
- `scripts/summarize_math_ink_06_formula_adapter_seeds.py`
- `scripts/audit_math_ink_06_p_proxy_shift.py`
- `scripts/materialize_math_ink_06_p_formula.py`
- `scripts/train_math_ink_06_p_formula_adapter.py`
- `scripts/summarize_math_ink_06_p_formula_seeds.py`
- `scripts/distill_math_ink_06_p_formula_student.py`
- `scripts/export_math_ink_06_p_formula_student.py`
- `scripts/run_math_ink_06_p_formula_release.py`
- `android/aiflow-math-ink-runtime`
- `scripts/preflight_math_ink_06_p_formula.py`
- `scripts/analyze_crohme_lattice_failures.py`
- `tests/test_behavior_context06.py`
- `tests/test_behavior_role_head06.py`
- `tests/test_p_formula_gate06.py`
- `tests/test_p_formula_intake06.py`
- `tests/test_distill_math_ink_06_p_formula_student.py`
- `tests/test_export_math_ink_06_p_formula_student.py`
- `tests/test_run_math_ink_06_p_formula_release.py`
- `src/math_grid_drawer/research/p_formula_intake06.py`
- `src/math_grid_drawer/research/p_formula_dataset06.py`
- `research/contracts/aiflow_p_formula_v1.schema.json`
- `research/contracts/aiflow_p_formula_annotation_v1.schema.json`
- `research/configs/MATH-INK-06-P-FORMULA-v1.json`
- `research/colab/AIFlow_Math_Ink_06_LiteRT.ipynb`
- `research/runs/math_ink_06_litert_colab_20260724/aiflow_math_ink_06_litert_bundle.zip`
- `research/runs/math_ink_06_composite_cpu_benchmark_familyfusion_20260724/report.json`
- `research/runs/math_ink_06_online_error_consensus_validation_3seed_20260724/report.json`
- `research/runs/math_ink_06_online_error_consensus_paired_test_3seed_20260724/report.json`
- `research/runs/math_ink_06_online_family_fusion_calibration_3seed_20260724/report.json`
- `research/runs/math_ink_06_composite_export_familyfusion_seed17_20260724/export_manifest.json`
- `research/runs/math_ink_06_behavior_exact_gate_3seed_20260724/report.json`
- `research/runs/math_ink_06_formula_context_contract_3seed_20260724/report.json`
- `research/runs/math_ink_06_formula_adapter_h64_3seed_20260724/run_summary.json`
- `research/runs/math_ink_06_behavior_role_p_proxy_w035_seed17_20260724/report.json`
- `research/runs/math_ink_06_p_proxy_shift_seed17_20260724/report.json`
- `research/AIFlow-MATH-INK-0.6-BEHAVIOR-CONTEXT-REPORT-20260724.md`
- `research/runs/math_ink_06_behavior_role_3seed_20260724/run_summary.json`
- `research/runs/math_ink_06_behavior_grouping_audit_20260724/report.json`
- `research/runs/math_ink_06_behavior_grouping_cross_joint_audit_20260724/report.json`
- `research/runs/math_ink_06_p_boundary_joint_sharedfix_3seed_20260724/run_summary.json`
- `research/runs/math_ink_06_p_boundary_device_stress_sharedfix_3seed_20260724`
- `research/runs/math_ink_06_composite_export_seed17_20260724/export_manifest.json`
- `research/runs/math_ink_06_x_grouping_guard_20260724/report.json`
- `research/runs/math_ink_06_x_edge_bottleneck_20260724/report.json`
- `research/runs/math_ink_06_cross_gap_guard_full_selector_20260724/report.json`
- `research/runs/math_ink_06_behavior_grouping_full_selector_gap50_20260724/report.json`
- `research/runs/math_ink_06_x_grouping_guard_gap50_full_selector_20260724/report.json`
- `research/runs/math_ink_06_full_selector_gap50_failure_audit_20260724/report.json`
- `research/runs/math_ink_06_cross_gap_guard_family6_full_selector_20260724/report.json`
- `research/runs/math_ink_06_multistroke_family_guard_geometry_protected_20260724/report.json`
- `research/runs/math_ink_06_multistroke_guard_regression_audit_20260724/report.json`
- `research/runs/math_ink_06_full_selector_gap40_family6_protected_failure_audit_20260724/report.json`
- `research/runs/math_ink_06_behavior_grouping_full_selector_gap40_family6_protected_20260724/report.json`
- `research/runs/math_ink_06_component_competition_guard_20260724/report.json`
- `research/runs/math_ink_06_local_baseline_guard_20260724/report.json`
- `research/runs/math_ink_06_local_baseline_overmerge_audit_20260724/report.json`
- `research/runs/math_ink_06_boundary_behavior_guard_20260724/report.json`
- `research/runs/math_ink_06_p_boundary_auxiliary_smoke_20260724/report.json`
- `research/runs/math_ink_06_p_boundary_auxiliary_fullsmoke_20260724/report.json`
- `research/runs/math_ink_06_p_boundary_joint_3seed_20260724/run_summary.json`
- `research/runs/math_ink_06_p_boundary_device_stress_3seed_20260724/report.json`
