# 증류형 수학 문맥 최종 확정기

상태: **구현·writer-LOO·전이 진단·체크포인트 재로딩 검증 완료 / 연구 gate 통과 / shadow 유지**

## 결론

형상 분류기는 그대로 두고, 수식 전체 맥락으로 HWR Top-5 후보 중 최종 문자를 확정하는 후단을 추가했다. 수학 문맥을 아는 대형 교사는 학습할 때만 사용하고, 실제 실행에는 경량 BERT-Tiny 학생만 사용한다.

```text
학습: 프로젝트 수식 ──> 고정 MathBERTa ──> 372-class soft target
                         │
                         └──> BERT-Tiny 학생: KL 증류 + 프로젝트 정답 CE 0.1

실행: HWR Top-5 + 수식 순서·공간관계 ──> BERT-Tiny 문맥 점수
      ──> 역할·정확 후보 점수 ──> fence/X×/희귀문자 guard ──> 기존 Top-5 중 1개
```

이 레이어는 새 문자를 만들거나 지우지 않고 stroke grouping도 바꾸지 않는다. HWR이 후보 형상을 소유하고, 문맥 레이어는 그 후보 안에서만 의미를 확정한다.

## 모델 감사와 채택

| 모델 | 감사 결과 | 사용 |
|---|---|---|
| `witiko/mathberta` | 모델 저장소 MIT, revision `4cb18380847a27c6d0d1d3db3459a78cd1b602cd`, RoBERTa 12층·hidden 768·12 heads | 연구용 고정 교사만 허용 |
| `tbs17/MathBERT` | Hub metadata와 model card에서 상업 이용 근거가 되는 라이선스를 확인하지 못함 | 제외 |
| Google BERT-Tiny | 2층·hidden 128·2 heads | 경량 실행 학생으로 채택; 제품 승격 보류 |

MathBERTa의 원래 tokenizer에서 372개 출력 라벨 중 단일 token은 250개(67.20%)뿐이었다. 따라서 교사 라벨은 LaTeX subtoken embedding을 평균해 372개 고정 class anchor로 만들었다. 교사 전체 모델은 약 586MB이며 체크포인트에 포함하지 않는다.

모델 카드의 MIT 표시는 학습 말뭉치 권리까지 정리하지 않는다. [ArXMLiv 2020 공식 페이지](https://sigmathling.kwarc.info/resources/arxmliv-dataset-2020/)는 데이터 이용을 연구·도구 개발 목적으로 제한하고, [Stack Overflow 공식 라이선스 안내](https://stackoverflow.com/help/licensing)는 게시 시기별 CC BY-SA 의무를 명시한다. 따라서 이 교사에서 증류한 r7은 연구 결과로만 보존하고 `commercial_training_corpus_audit_passed=false`를 강제한다.

## 데이터와 누수 경계

| 구분 | 사용 |
|---|---|
| 프로젝트 ownership 47식·211문자·3 writer | 교사 soft target 증류, 프로젝트 정답 CE 0.1, writer-LOO 평가 |
| CROHME 984식·9,535 지원 문자 | CC BY-NC 전이 진단만 수행 |

각 outer writer-LOO fold에서 held writer는 학습·epoch 선택에서 제외했다. 남은 writer도 formula-disjoint fit/validation으로 나눠 교사 KL만으로 epoch를 선택한 뒤 재학습했다. product refit도 formula-disjoint 교사 KL로 epoch 40을 선택했다. CROHME 정답은 학습, epoch 선택, 설정 선택에 사용하지 않았다.

다만 역할 가중치와 exact-context scale은 같은 프로젝트 데이터의 이전 r2/r4 개발에서 고정했다. 따라서 직접 writer-LOO도 새로운 untouched acceptance가 아니라 반복 개발 진단값이다.

## 결과

### 직접 수집 writer-LOO

| 지표 | HWR | 역할 r2 | 증류 r7 | r2 대비 |
|---|---:|---:|---:|---:|
| 문자 Top-1 | 73.46% | 84.83% | **88.63%** | +3.79%p |
| strict 동형군 micro | 39.62% | 71.70% | **73.58%** | +1.89%p |
| strict 동형군 macro | 57.94% | 74.49% | **75.74%** | +1.25%p |
| 수식 exact | 34.04% | 51.06% | **68.09%** | +17.02%p |

211문자 중 36건을 변경해 33건을 개선하고 1건을 회귀했다. paired sign-test는 `p=4.07e-9`였다.

### CROHME 전이 진단

| 지표 | HWR | 역할 r2 | 증류 r7 | r2 대비 |
|---|---:|---:|---:|---:|
| 문자 Top-1 | 69.50% | 76.14% | **78.69%** | +2.55%p |
| strict 동형군 micro | 38.28% | 62.20% | **65.10%** | +2.91%p |
| strict 동형군 macro | 42.64% | 52.34% | **57.68%** | +5.34%p |
| 수식 exact | 15.85% | 24.09% | **27.44%** | +3.35%p |

9,535문자 중 1,816건을 변경해 1,042건을 개선하고 166건을 회귀했다. paired sign-test는 `p=1.72e-155`였다.

동형 문자별 정답 수는 `/` 13→12/19, `|` 38→39/62, `0` 91→150/213, `o` 1→0/11, `x` 191→371/587, `\\times` 33→57/72였다. `/·o`는 아직 상용 비회귀 조건을 충족하지 못한다.

## 후보 보존과 실행 경계

- 직접 211/211, CROHME 9,535/9,535가 원래 HWR Top-5 안에서 확정됨
- 새 token 0건, 삭제 0건, grouping 변경 0건
- `X→\\times`는 양옆이 숫자/피연산자이고 `\\times`가 이미 Top-5일 때만 허용
- 균형 fence 쌍과 프로젝트에서 신뢰 가능한 Top-1은 별도 guard로 보호
- 학생 체크포인트: 33,848,418 bytes, SHA-256 `df0bd6e80fffacd267743b64664878be9142df5ab91078008fbb6b393da34aa6`
- 결과 보고서 SHA-256 `8237299a7cd6ed130f7f7b7498cae38630ea8b285a12a3e27bf517ba1f176094`

4GB GPU에서 r7 새 프로세스 로딩은 0.34초, 배치 진단 처리량은 직접 211문자 846 glyph/s, CROHME 9,535문자 1,936 glyph/s였다. 이는 단일 GPU 배치 측정이며 동시접속 200명 서비스 보장은 아니다.

## 산출물과 재현

- 구현: `scripts/train_distilled_math_context_v1.py`
- 공통 후보·guard: `scripts/train_context_decision_layer_v1.py`
- 로컬 산출물: `artifacts/distilled_math_context_20260819_r7`
- 학생 체크포인트: `distilled_math_context_product.pt`
- 전체 fold·예측·감사 보고서: `distilled_math_context_report.json`

```powershell
$env:PYTHONNOUSERSITE='1'
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
python scripts\train_context_decision_layer_v1.py --self-test --device cuda
python scripts\train_distilled_math_context_v1.py --self-test --device cuda
python scripts\train_distilled_math_context_v1.py --device cuda --output artifacts\distilled_math_context_20260819_r7
```

## 결정

- `research_gate_passed=true`
- `candidate_preserving=true`
- `checkpoint_reload_mismatch=0` (product refit·CROHME)
- `commercial_training_corpus_audit_passed=false`
- `commercial_gate_passed=false`
- `automatic_default_replacement=false`
- `runtime_status=shadow distilled context finalizer`

r7은 현재 연구 성능 champion이지만 상용 기본 레이어로 확정하지 않는다. MathBERTa 말뭉치 권리가 상용으로 정리되지 않았고, CROHME와 직접 47식 모두 반복 관찰됐다. 다음 승격 조건은 상업 이용이 명확한 자체·합성 수식 문맥으로 교사를 교체하고, 새 프로젝트 소유 writer/formula acceptance set에서 `/·|·o·0·O·x·\\times` 비회귀, 수식 grouping, LaTeX-equivalent exact를 함께 통과하는 것이다.
