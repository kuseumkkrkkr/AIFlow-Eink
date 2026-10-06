# 자체 수식 문맥 확정 레이어

상태: **구현·writer-LOO·전이 진단·체크포인트 재로딩 검증 완료 / 상업 학습권리 gate 통과 / 정확도 gate 미통과 / shadow 유지**

## 결론

형상 분류기는 계속 HWR이 담당한다. 새 후단은 HWR Top-5, 수식 순서, 7개 공간관계를 읽고 후보 안에서만 최종 문자를 확정한다. 새 token 생성, 삭제, stroke regrouping은 불가능하다.

외부 사전학습 가중치와 외부 텍스트 말뭉치를 모두 제거했다. 2-layer·hidden 128·2-head Transformer를 무작위 초기화하고, 프로젝트 소유 47개 수식과 저장소 안의 결정적 수식 DSL만으로 학습했다. 따라서 MathBERTa 증류 r7의 말뭉치 권리 문제는 이 후보에 전이되지 않는다.

```text
HWR 372-head ──> Top-5 문자·확률
                         + 수식 순서·공간관계 7종
                         v
자체 Transformer ──> 5개 의미 역할 + 372-class 문맥 점수
                         + HWR 형상 점수 + 소유 수식 역할 문법
                         v
형상 확률 경계 ──> 기존 Top-5 중 하나만 확정
```

현재 실행기는 1차 확정열을 주변 token으로 넣어 같은 MASK 문맥 모델을 한 번 더 읽는다. 2차 pass는 새 후보를 확정할 수 없으며, 1차 문맥 보정을 HWR Top-1으로 되돌리라는 결과만 받아들인다. 이 보수적 재검토는 직접수집 결과를 바꾸지 않고 CROHME의 `\times→x` 오보정 1건을 복구했다.

## 모델과 경계

| 항목 | 확정값 |
|---|---:|
| Transformer | 무작위 초기화 BERT encoder |
| 층 / hidden / attention heads | 2 / 128 / 2 |
| 전체 파라미터 | 395,769 |
| 출력 | 5개 역할 head + 372-class head |
| 외부 사전학습 가중치 | 없음 |
| 외부 텍스트 말뭉치 | 없음 |
| 학습률 / weight decay | 3e-4 / 0.01 |
| 정확문자 보조손실 가중치 | 0.5 |
| 제품 epoch | 1 |
| 제품 설정 | 세 outer writer fold 선택값의 중앙값 |
| 후보 확률비 하한 | 0.006115167778304273 |

후보 확률비 하한은 직접수집 writer-LOO에서 실제로 맞힌 문맥 교정 중 가장 약한 `후보 HWR 확률 / HWR Top-1 확률`이다. 이보다 형상 근거가 약한 후보는 문맥 점수가 높아도 승격하지 않는다. CROHME 정답은 이 값이나 epoch·가중치 선택에 사용하지 않았다.

제품 epoch도 작은 내부 분할이 아니라 세 outer writer fold의 선택 epoch `1·3·1`의 중앙값 1로 고정했다. epoch 4 제품은 전이 과적합이 확인되어 폐기했다.

## 데이터 권리 경계

| 자료 | 용도 |
|---|---|
| 프로젝트 ownership 47식·211문자·3 writer group | 실제 수식 정답 학습, writer-LOO 개발 진단 |
| 저장소 내 결정적 수식 DSL 16,000 MASK 예제/fit | 역할·문자 문맥 학습 |
| CROHME 984식·9,535 지원 문자 | CC BY-NC 전이 개발 진단만 수행 |
| MathBERTa·ArXMLiv·Math StackExchange | 사용하지 않음 |
| Google MathWriting | CC BY-NC-SA이므로 상용 학습 후보에서 제외 |

체크포인트에는 외부 tokenizer나 교사 가중치가 없다. 입력 vocabulary는 372개 프로젝트 문자 ID, `right·left·above·below·superscript·subscript·overlap`, 네 special token으로만 구성된다.

## 결과

### 직접수집 writer-LOO

| 지표 | 고정 HWR | 역할 r2 | 비상용 증류 r7 | 자체 문맥 r6 |
|---|---:|---:|---:|---:|
| 문자 Top-1 | 73.46% | 84.83% | 88.63% | **89.10%** |
| strict 동형군 micro | 39.62% | 71.70% | 73.58% | **77.36%** |
| strict 동형군 macro | 57.94% | 74.49% | 75.74% | **79.40%** |
| 수식 exact | 34.04% | 51.06% | **68.09%** | 65.96% |

211문자 중 36건을 변경해 33건을 개선하고 기존 정답 회귀는 0건이었다. 세 held writer 정확도는 91.43%, 98.68%, 81.00%다. 이 값은 모델 개발에 반복 사용됐으므로 untouched 상용 acceptance 값은 아니다.

### CROHME 전이 개발 진단

| 지표 | 고정 HWR | 역할 r2 | 비상용 증류 r7 | 자체 문맥 r6 |
|---|---:|---:|---:|---:|
| 문자 Top-1 | 69.50% | 76.14% | **78.69%** | 77.82% |
| strict 동형군 micro | 38.28% | 62.20% | **65.10%** | 63.15% |
| strict 동형군 macro | 42.64% | 52.34% | **57.68%** | 52.01% |
| 수식 exact | 15.85% | 24.09% | 27.44% | **27.95%** |

9,535문자 중 1,979건을 변경해 1,073건을 개선하고 280건을 회귀했다. 모든 문자는 원래 HWR Top-5 안에서만 결정됐다.

동형 문자별 정답 수는 `/` 13→10/19, `|` 38→32/62, `0` 91→167/213, `o` 1→0/11, `x` 191→327/587, `\times` 33→42/72다. `/·|·o`는 비회귀 조건을 충족하지 못한다. 소유 양성식 없이 `O·o` 합성 빈도만 올린 실험은 전체·수식 exact를 낮춰 채택하지 않았다.

## 산출물과 재현

- 구현: `scripts/train_owned_formula_context_v1.py`
- 제품 최종기: `scripts/finalize_formula_context_v1.py`
- 제품 무결성 기본값: 산술 정답 강제 비활성, 연구 제안에서만 `--enable-equation-correction`
- 2-pass 복귀 경계: `scripts/context_recheck_guard_v1.py`
- 고정 체크포인트 평가기: `scripts/evaluate_owned_formula_context_checkpoint_v1.py`
- 공통 후보 결합·형상 경계: `scripts/train_context_decision_layer_v1.py`
- 최종 로컬 실행: `artifacts/owned_formula_context_20260822_r6`
- 체크포인트: `owned_formula_context_product.pt`
- 체크포인트 크기: 1,611,193 bytes
- 체크포인트 SHA-256: `7e0ff1b04ba1071867bfa6811869f242907ebe017d0996597776c6050cc2fc94`
- 보고서 SHA-256: `97670478331d187b37dd1006d715db8c39bbe6050f2b8794da35aed7318e8e27`
- 직접 제품 재적합·CROHME 재로딩 예측 불일치: 각각 0건

```powershell
$env:PYTHONNOUSERSITE='1'
python scripts\train_context_decision_layer_v1.py --self-test --device cpu
python scripts\train_owned_formula_context_v1.py --self-test --device cpu
python scripts\train_owned_formula_context_v1.py --device cuda --output artifacts\owned_formula_context_20260822_r6
```

## 결정

- `research_gate_passed=true`
- `commercial_context_training_rights_gate_passed=true`
- `commercial_accuracy_gate_passed=false`
- `candidate_preservation_rate=1.0`
- `new_tokens=0`, `grouping_mutations=0`
- `automatic_default_replacement=false`
- `runtime_status=shadow owned formula-context candidate`

r6은 현재 상업 이용 가능한 학습원만 사용한 최선의 문맥 후보지만 상용 기본 레이어는 아니다. 직접 47식과 CROHME가 모두 반복 개발에 사용됐고, 새 작성자·새 수식의 untouched acceptance 및 `/·|·o` 소유 양성식이 없다. 다음 승격은 새 프로젝트 소유 acceptance에서 후보 보존, 동형군 비회귀, 수식 exact, grouping 비회귀를 함께 통과한 뒤에만 허용한다.
