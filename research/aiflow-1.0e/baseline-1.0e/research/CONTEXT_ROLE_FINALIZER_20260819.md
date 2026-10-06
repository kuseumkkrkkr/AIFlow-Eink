# 역할 기반 MASK 문맥 최종 확정기

상태: **구현·writer-LOO·전이 진단·체크포인트 재로딩 검증 완료 / shadow 채택 / 상용 기본 승격 보류**

## 변경 이유

기존 BERT-Tiny MASK 재랭커는 전체 정확도를 높였지만, 47개 프로젝트 수식에 정답이 없는 `|·O·o`를 선택하지 못했다. 라벨별 출력 헤드가 문맥을 이해한 것이 아니라 관측된 라벨 빈도에 맞춰지는 구조였기 때문이다.

새 최종 확정기는 역할을 분리한다.

1. 고정 372-class HWR이 문자별 Top-5와 형상 확률을 만든다.
2. 고정 BERT-Tiny가 전체 수식과 7개 공간관계 token을 MASK attention으로 읽는다.
3. 프로젝트 소유 수식에서 학습한 역할 문법이 숫자·연산자·구분자·피연산자·기타의 타당성을 계산한다.
4. 문맥은 역할만 선택하고, 선택된 역할 안에서는 HWR 최상위 문자를 그대로 사용한다.
5. 절댓값 막대는 가까운 비교차 쌍만 허용하고, 쌍 내부에 등호·부등호가 있으면 막대 쌍으로 보지 않는다.
6. 프로젝트에서 5회 미만 본 HWR Top-1을 바꿀 때는 0.25의 추가 확신 마진을 요구한다.

따라서 `0·O·o`, `x·\times`, `1·|·/`의 의미 역할은 문맥이 다루되, 같은 역할 안의 대소문자·정확한 형상은 HWR이 소유한다. 새 token 생성, token 삭제, stroke regrouping은 불가능하다.

구현 당시 코드: commit `10a5f8e`의 `scripts/train_context_decision_layer_v1.py`. 현재 같은 파일은 증류형 후속 실험을 위해 schema v2로 확장됐으므로 아래 r2 산출물과 수치는 역사적 기준선으로 보존한다.

## 데이터 경계

| 구분 | 사용 |
|---|---|
| 프로젝트 ownership 47식·211문자·3 writer | 역할 문법 학습, nested formula-disjoint 설정 선택, writer-LOO 평가 |
| CROHME 984식·9,535 지원 문자 | CC BY-NC 전이 진단만 수행 |
| 직접 canonical 10식·외부 문자 10% | 검증된 수식 sequence·공간관계가 없어 문맥 평가는 수행하지 않음 |

CROHME 정답은 역할 문법, 가중치, margin 학습에 넣지 않았다. 다만 같은 CROHME를 이전 문맥 실험부터 반복 관찰했으므로 최종 상용 acceptance 근거로 간주하지 않는다.

## 누수 없는 직접 수집 writer-LOO

각 outer writer를 통째로 제외하고, 남은 writer의 수식을 다시 formula-disjoint fit/validation으로 나눠 attention·형상·문법 가중치와 margin을 선택했다.

| 지표 | HWR | 역할 문맥 최종 확정기 | 변화 |
|---|---:|---:|---:|
| 문자 Top-1 | 73.46% | **84.83%** | +11.37%p |
| strict 동형군 micro | 39.62% | **71.70%** | +32.08%p |
| strict 동형군 macro | 57.94% | **74.49%** | +16.56%p |
| 수식 exact | 34.04% | **51.06%** | +17.02%p |

- 변경 30건, 개선 26건, 회귀 2건
- paired sign test `p=3.03e-6`
- 문자별 적중: `/` 3/3 유지, `0` 9→11/16, `1` 3→17/25, `\times` 5→6/7, `x` 1/2 유지

## CROHME 전이 진단

제품 설정은 직접 수집 fold 선택값의 중앙값만 사용했다.

- attention weight 0.5
- HWR role-mass weight 0.5
- 역할 문법 weight 1.0
- 기본 변경 margin 0.25

| 지표 | HWR | 역할 문맥 최종 확정기 | 변화 |
|---|---:|---:|---:|
| 문자 Top-1 | 69.50% | **76.14%** | +6.64%p |
| strict 동형군 micro | 38.28% | **62.20%** | +23.92%p |
| strict 동형군 macro | 42.64% | **52.34%** | +9.70%p |
| 수식 exact | 15.85% | **24.09%** | +8.23%p |

- 변경 1,521건, 개선 797건, 회귀 164건
- 후보 보존 9,535/9,535, 새 token 0건, grouping 변경 0건
- 절댓값 막대 구조 보호 56건
- 희귀 Top-1 추가 margin으로 변경 보류 225건

남은 문자별 약점은 `/` 13→11/19, `|` 38→34/62, `o` 1→0/11이다. 전체·micro·macro 지표는 모두 개선됐지만 이 세 라벨의 비회귀는 아직 충족하지 못했다. 새 프로젝트 소유 양성식 없이 이 결과를 상용 기본값으로 승격하지 않는다.

## 산출물과 재현

로컬 실행 폴더: `artifacts/context_role_finalizer_20260819_r2`

- `context_role_finalizer_product.pt`: 19,196 bytes, SHA-256 `8766e1d47ef9ca33fe40ff99a7acb34c0ecc36feb6d279b5f53289954fd55642`
- `context_role_finalizer_report.json`: 전체 fold·설정·평가 기록
- `direct_writer_loo_predictions.jsonl.gz`: 직접 수집 outer writer-LOO 예측
- `direct_product_refit_predictions.jsonl.gz`: 전체 47식 문법의 training-fit 진단
- `crohme_transfer_predictions.jsonl.gz`: 프로젝트 데이터만 사용한 전이 진단

```powershell
$env:PYTHONNOUSERSITE='1'
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
python scripts\train_context_decision_layer_v1.py --self-test --device cuda
python scripts\train_context_decision_layer_v1.py --device cuda --output artifacts\context_role_finalizer_20260819_r2
```

새 프로세스에서 체크포인트를 다시 읽어 직접 수집 211건과 CROHME 9,535건을 재계산했다. 저장 예측 불일치 0건, coverage 불일치 0건, Top-5 밖 출력 0건이었다.

## 결정

- `research_gate_passed=true`
- `candidate_preserving=true`
- `checkpoint_created=true`
- `shadow_component=true`
- `automatic_default_replacement=false`

다음 승격 조건은 새 프로젝트 소유 writer/formula에서 `|·/·o` 비회귀, 수식 grouping, LaTeX-equivalent exact를 함께 통과하는 것이다.

## 후속 결과

이 r2는 연구 기준선으로 보존한다. MathBERTa 교사를 BERT-Tiny에 증류한 r7이 직접 writer-LOO와 CROHME의 Top-1·strict micro·strict macro·수식 exact를 모두 상회했다. 후속 구조와 승격 경계는 [DISTILLED_MATH_CONTEXT_20260819.md](DISTILLED_MATH_CONTEXT_20260819.md)에 기록했다.
