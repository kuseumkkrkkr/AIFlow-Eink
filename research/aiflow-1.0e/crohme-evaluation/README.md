# AIFlow Math Ink 1.0 연구 Fork

온라인 수학 필기를 `원본 stroke → 형태/그리딩 → 문자 Top-k → 보수적 결정`으로 분리하는 상용 1.0 연구 트랙입니다. 이 저장소는 0.9 릴리스의 독립 Fork이며, 0.9 작업본을 변경하지 않습니다.

## 1.0 데이터 정책

- 데이터 카탈로그: [datasets/DATASET_CATALOG.md](datasets/DATASET_CATALOG.md)
- 현재 활용·승인 데이터 목록: [datasets/UTILIZED_DATASETS.md](datasets/UTILIZED_DATASETS.md)
- 2026-08-10 원본 재감사: [research/DATASET_REAUDIT_20260810.md](research/DATASET_REAUDIT_20260810.md)
- HWR 논문·데이터셋 포화점 검토: [research/HWR_PAPER_DATASET_REVIEW_20260810.md](research/HWR_PAPER_DATASET_REVIEW_20260810.md)
- 외부 원본은 `datasets/`에 출처·상태별로 분리한다. 압축 원본은 로컬 보관 대상이며 Git에 넣지 않는다.
- `10_approved_external`만 승인된 외부 HWR 학습 풀이다. 승인과 현재 체크포인트 반영은 구분한다.
- `20_reaudit_required`는 재감사 완료 전 학습·평가·모델선택에 사용할 수 없다.
- ISGL, UCI Character Trajectories, 정제 HWRT, BDSHWA는 2026-08-10 용도 제한 조건으로 채택했다.
- 외부 데이터는 box-local 문자 후보 학습에만 쓰고, 수식 grouping·ownership·관계·최종 결정은 프로젝트 보유 수식 데이터가 담당한다.
- 비상업 데이터는 제품 학습에 사용하지 않는다.

## 공개물

- `paper/AIFlow_Math_Ink_0_9_Draft.md`: 연구 논문 초안
- `reports/public_retrain_seed31_metrics.json`: 공개 데이터 재학습·필기 replay 지표
- `reports/crohme2019_valid.json`: CROHME2019 비상업 검증
- `scripts/build_public_dataset09.py`: 비식별 공개 데이터 생성
- `scripts/validate_public_dataset09.py`: 공개 데이터 무결성 검사
- `scripts/train_normalized_balanced_touch_calibrator09.py`: 공개 데이터 재학습
- Hugging Face model: https://huggingface.co/cwLeeDev/aiflow-math-ink-0.9
- Hugging Face dataset: https://huggingface.co/datasets/cwLeeDev/aiflow-math-ink-0.9-dataset

## 공개 데이터

데이터 관리자는 참여자 공개 동의 완료를 확인했습니다. 119개 수식의 stroke sequence를 공개하되 원본 contributor/session/prompt ID, 기록 시각, 이미지 URL, IP·원본 경로는 제거했습니다. dataset-local writer/session 별칭과 formula-relative 시간만 제공합니다.

- valid: 110건
- pending: 4건, 기본 학습 제외
- reject: 5건, 기본 학습 제외
- 수동 ownership: 47개 수식·211개 문자

## 결과

공개 JSONL 재학습은 기존 seed31을 재현했습니다. 49개 수식·171개 문자 replay에서 Top-1 87.72%, Top-5 98.83%, 수식 exact 67.35%입니다. 고신뢰 baseline 보존 guard는 집계 정확도를 유지하면서 회귀를 3건에서 2건으로 줄였습니다.

CROHME2019 CC BY-NC valid 985식에서는 truth-group 문자 Top-1이 56.96%→74.11%, end-to-end group-token Top-1 recall이 48.20%→62.17%, fully-covered 수식 exact가 7.95%→11.98%였습니다. guarded 후보의 수식 exact는 12.10%입니다.

## 상태

- `research_adopted=true`
- `product_adopted=false`
- CROHME2019은 비상업 검증으로만 사용
- 자동채점 모델이 아니라 필기 인식 연구 후보
- writer-disjoint 실사용 검증과 독립 guard threshold 검증은 아직 필요

## 1.0 MASK 문맥 재랭커

Google BERT-Tiny에 372개 문자 token과 7개 공간관계 token을 추가한 후보 보존형 재랭커를 구현했다. 프로젝트 ownership 47식만 학습하고 CROHME는 전이 평가에만 사용했다. 직접 writer-LOO Top-1은 73.46%→84.36%, CROHME 지원 문자 Top-1은 69.50%→76.36%였지만, 프로젝트 학습원에 `|·O·o` 정답이 없어 CROHME strict macro가 42.64%→37.69%로 하락했다.

따라서 체크포인트는 shadow component로 생성했으며 기존 372-head의 기본 출력을 자동 교체하지 않았다. 구조·지표·동형문자별 실패와 재현 방법은 [research/MASKED_CONTEXT_BERT_TINY_20260814.md](research/MASKED_CONTEXT_BERT_TINY_20260814.md)에 기록했다.

## 1.0 역할 기반 문맥 최종 확정기

라벨별 MASK 출력 헤드는 프로젝트에 없는 `|·O·o`를 억제하므로 폐기했다. 새 후단은 HWR Top-5의 정확한 문자 형상을 그대로 두고, 고정 BERT-Tiny attention과 프로젝트 수식의 역할 문법으로 `숫자·연산자·구분자·피연산자·기타` 역할만 확정한다. 같은 역할 안에서는 HWR 순위를 바꾸지 않는다.

직접 수집 writer-LOO Top-1은 73.46%→84.83%, strict macro는 57.94%→74.49%, 수식 exact는 34.04%→51.06%였다. CROHME 진단은 Top-1 69.50%→76.14%, strict macro 42.64%→52.34%, 수식 exact 15.85%→24.09%였다. 모든 9,746건에서 원래 Top-5 밖 출력과 grouping 변경은 0건이다.

체크포인트는 재로딩 예측 불일치 0건을 확인했지만 아직 shadow 상태다. CROHME는 반복 개발에 사용된 CC BY-NC 진단셋이며, 상용 기본 승격에는 새 프로젝트 소유 writer/formula acceptance set이 필요하다. 구현·경계·재현 절차는 [research/CONTEXT_ROLE_FINALIZER_20260819.md](research/CONTEXT_ROLE_FINALIZER_20260819.md)에 기록했다.

## 1.0 증류형 수학 문맥 최종 확정기

모델 저장소가 MIT로 표시된 MathBERTa를 연구용 학습 교사로 사용해 372-class 문맥 분포를 만들고, 이를 경량 Google BERT-Tiny에 증류했다. 실행 시에는 586MB 교사가 필요하지 않으며 33.8MB 학생 체크포인트만 사용한다. 출력은 항상 HWR Top-5 안에서만 선택하고 stroke grouping은 바꾸지 않는다.

직접 수집 writer-LOO Top-1은 88.63%, strict macro는 75.74%, 수식 exact는 68.09%였다. CROHME 진단은 각각 78.69%, 57.68%, 27.44%였다. 이는 기존 역할 기반 r2의 대응 지표를 모두 상회하며, 후보 이탈과 grouping 변경은 9,746건 전체에서 0건이다.

연구 gate는 통과했지만 자동 기본 교체는 보류한다. MathBERTa 학습원인 ArXMLiv·Math StackExchange의 상용 권리 감사도 통과하지 못했고, CROHME는 반복 관찰된 CC BY-NC 진단셋이며 `/·o`는 여전히 약하다. 새 프로젝트 소유 writer/formula acceptance와 상업 이용이 명확한 교사 교체가 모두 필요하다. 구조·모델 감사·재현 절차는 [research/DISTILLED_MATH_CONTEXT_20260819.md](research/DISTILLED_MATH_CONTEXT_20260819.md)에 기록했다.

## 1.0 자체 수식 문맥 확정 레이어

외부 사전학습 가중치와 외부 텍스트 말뭉치를 제거하고, 무작위 초기화 2-layer Transformer를 프로젝트 소유 47식과 저장소 내 결정적 수식 DSL만으로 학습했다. 실행 시 HWR Top-5와 7개 공간관계만 읽고 기존 후보 중 하나를 확정하며, 새 token·삭제·stroke regrouping은 허용하지 않는다.

직접수집 writer-LOO는 Top-1 89.10%, strict macro 79.40%, 수식 exact 65.96%, 기존 정답 회귀 0건이다. CROHME 개발진단은 각각 77.82%, 52.01%, 27.95%다. 체크포인트 재로딩 불일치는 직접·CROHME 모두 0건이며 상업 학습권리 gate는 통과했다.

다만 직접 47식과 CROHME가 반복 개발에 사용됐고 `/·|·o` 전이 비회귀가 확인되지 않아 자동 기본 교체는 하지 않는다. 현재 상태는 `shadow owned formula-context candidate`이며, 구조·권리 경계·실패 문자·재현 절차는 [research/OWNED_FORMULA_CONTEXT_20260822.md](research/OWNED_FORMULA_CONTEXT_20260822.md)에 기록했다.

### 신규 수집 데이터 적용과 실제 추론 경계

2026-08-19 수집분을 포함한 ownership 96식 중 고정 372-class HWR에 없는 `t`가 든 한 식을 통째로 제외하고 95식·387글자를 감사했다. 기존 r6을 새 수집 48식에 재학습 없이 적용한 뒤, 기본·강화 exact-context 출력을 함께 계산하는 지원 기반 맥락 가드와 누락된 괄호 짝, 수평 값 사이의 유일한 중위 연산자만 복원하는 결정적 문법 가드를 연결했다. 지원 기반 가드는 기존 선택 문자의 프로젝트 support가 0이고, 강화 후보가 같은 의미 역할·프로젝트 support 5건 이상·HWR Top-1 대비 확률비 0.2 이상일 때만 Top-5 안에서 확정한다. 산술 등식을 실제로 맞게 만드는 가드는 연구 비교용 옵션으로 분리했다. 채점 제품의 HWR이 사용자가 틀리게 쓴 식을 정답으로 바꾸면 안 되기 때문이다. 신규 48식의 HWR Top-1 65.91%는 제품 기본 경로에서 84.09%, 수식 exact 31.25%는 60.42%로 올랐고 기존 HWR 정답 회귀는 0건이었다. 산술 정답 보정을 켠 연구 비교값은 각각 85.80%, 66.67%다. 반면 95식 재학습 challenger, HWR 문맥 노이즈 재학습, 제약 없는 반복 추론, 등호 없는 산술 강제와 다중 기호 숫자화는 효과 부족 또는 안전성 문제로 기각했다.

실제 추론은 [scripts/finalize_formula_context_v1.py](scripts/finalize_formula_context_v1.py)가 담당한다. 정답 `label` 없이 HWR Top-5·확률·수식 순서·공간관계만 받고, 후보 추가·글자 삭제·stroke regrouping 없이 후보 하나를 확정한다. 1차 확정열을 다시 전체 수식 MASK 문맥으로 읽는 2차 pass를 추가했으며, 2차 결과가 1차 보정을 HWR Top-1로 되돌릴 때만 복구한다. 새로운 후보 변경은 허용하지 않는다. 길이 1 입력은 호환 Top-1을 유지하되 수식 맥락이 없음을 `ambiguous_no_formula_context`로 명시한다. 사용자가 쓴 산술의 진위를 보존하는 제품 기본 경로는 전체 직접수집 Top-1 87.34%, 수식 exact 64.21%다. 산술 정답 보정을 명시적으로 켠 연구 비교값은 88.89%, 70.53%이며, 이 차이 6글자는 인식이 아니라 등식 정답 강제에서 발생했다. CROHME는 두 모드 모두 Top-1 78.11%, 수식 exact 28.56%다. 연구·데이터 생성 스크립트의 D: 저장 경계는 유지하지만 제품 최종기는 드라이브 검사를 제거해 Linux·컨테이너·UNC 경로에서 같은 고정 체크포인트를 읽을 수 있다. 이 경로는 아직 opt-in shadow 상태다. 새 수집분에는 `|`와 `o` 정답이 없어 상용 정확도 gate는 유지하며, 전체 감사는 [research/OWNED_CONTEXT_ARRIVAL_AUDIT_20260819.md](research/OWNED_CONTEXT_ARRIVAL_AUDIT_20260819.md)에 기록했다.

## 1.0 공식 배치 후처리

검증된 문자 확정 결과 뒤에 분수·루트·위첨자·아래첨자 관계 그래프와 LaTeX를 내보내는 후처리를 추가했다. 직접 수집 95식 중 관계 정답이 없는 94식의 구조 오탐은 6개에서 0개가 되었고 문자 정확도는 유지됐다. CROHME 비상업 진단의 관계 완전일치는 539/984식에서 552/984식, 관계와 문자를 모두 맞춘 엄격 식 일치는 220/984식에서 225/984식으로 증가했다.

배치 결과는 문자 모델로 되먹이지 않으며 새 문자·삭제·stroke regrouping을 허용하지 않는다. 프로젝트 소유 검토식 3개(`x²`, `y²+1`, `√(a)`)·관계 5개는 3/3 완전일치했고, 현재 직접 HWR 후보와 연결되는 `√(a)`는 관계와 문자 모두 1/1 엄격 일치했다. 다만 이 3식은 규칙 구현에 사용된 소표본이며 신규 작성자·신규 식의 untouched 검증이 아니므로 상태는 `shadow_runtime_only`다. 구현·기각 분기·승격 조건은 [reports/FORMULA_LAYOUT_GRAPH_LOOP_20260819.md](reports/FORMULA_LAYOUT_GRAPH_LOOP_20260819.md)에 기록했다.

### 문맥 기반 수식 구문 복원

공식 배치와 자체 수식 문맥 점수를 함께 읽는 마지막 후보 보존 레이어를 추가했다. 이 레이어는 현재 열이 숫자식으로 해석 가능하고 HWR Top-5 안에 구조 후보가 있을 때만 한 글자를 바꾼다. 허용 규칙은 `숫자·숫자·숫자` 3항식의 가운데 `+` 후보와, HWR 1순위와 선택 후보가 모두 등호 계열인 경우뿐이다. 산술 계산, 새 문자 생성, 글자 삭제, grouping 변경은 하지 않는다.

현재 배치 코드 기준 직접 수집 수식 완전일치는 69/95(72.63%)에서 72/95(75.79%), 문자 Top-1은 346/387(89.41%)에서 349/387(90.18%)로 올랐고 회귀는 0건이다. v2는 `/`로 확정됐지만 폭이 높이보다 충분히 길고 진행 방향이 수평인 Top-5 후보만 `-`로 복원한다. CROHME 반복 진단은 수식 완전일치 283/984→284/984, 관계·문자 동시 완전일치 225/984→226/984이며 v2 가로선도 1글자 개선·회귀 0건이다. 직접 수집 Top-5 수식 oracle이 77/95(81.05%)라서 18식은 이 후단만으로 해결할 수 없다. 반복 관찰 데이터로 규칙을 정했으므로 제품 기본값이 아니라 `shadow_runtime_only`이며, 구현·평가·SHA는 [reports/FORMULA_SYNTAX_RESCUE_LOOP_20260819.md](reports/FORMULA_SYNTAX_RESCUE_LOOP_20260819.md)에 기록했다.

중첩 관계에서 한 문자가 여러 부모를 갖는 경우 출력은 해당 노드를 감사 목록에 남기고 LaTeX를 `shadow_ambiguous_multi_parent`로 표시한다. 이 LaTeX는 관계 정답이 있는 소유 데이터로 검증되기 전까지 확정식이 아니다.

### 단일기호 보조 HWR 복원

문맥이 없는 길이 1 수식은 후단 임베딩만으로 확정할 근거가 없으므로, 기존 372-class HWR Top-5를 유지한 채 writer-LOO 보조 shape head를 제한적으로 사용한다. 직접 수집 95식에서 `/`와 `\times` 두 건만 복원해 문자 Top-1은 349/387(90.18%)에서 351/387(90.70%), 수식 완전일치는 72/95(75.79%)에서 74/95(77.89%)로 증가했고 회귀는 0건이다. 외부 collision-free 18,330글자 진단도 8건 개선·회귀 0건이었다.

이 경로는 고정 HWR Top-5 밖을 선택하지 않고 새 token·삭제·grouping 변경을 하지 않는다. 다만 임계값은 반복 관찰된 직접 수집식에서 정했고 CROHME 평가 분할에는 단일기호 수식이 없으므로 `shadow_runtime_only`다. 구현·기각 분기·SHA와 승격 조건은 [reports/SINGLETON_SHAPE_RESCUE_LOOP_20260819.md](reports/SINGLETON_SHAPE_RESCUE_LOOP_20260819.md)에 기록했다.

### 이중 HWR 숫자 operand 복원

전면 HWR fusion은 직접식 회귀 때문에 사용하지 않는다. 대신 기존 경로가 숫자 문법으로 무효이고 보조 경로의 `operand→숫자` 변경만으로 처음 유효해질 때, 한 식에서 최대 두 글자만 보조 경로에서 받는다. 직접 수집은 문자 355/387(91.73%), 수식 완전일치 76/95(80.00%)로 올랐고 CROHME 반복 진단도 286/984식으로 2식 증가했다. 두 범위 모두 문자·수식 회귀는 0건이다.

이 레이어는 기존 Top-5를 고집하지 않고 기존·보조 HWR Top-5 합집합을 권위 후보로 사용한다. 직접 3글자와 CROHME 2글자가 기존 Top-5 밖에서 회복됐지만 합집합 밖 생성, 연산자·관계·괄호·배치·grouping 변경은 0건이다. 계약 변경과 이중 추론 비용이 있으므로 `shadow_runtime_only`이며, 구현·평가·SHA는 [reports/DUAL_HWR_NUMERIC_RESCUE_LOOP_20260819.md](reports/DUAL_HWR_NUMERIC_RESCUE_LOOP_20260819.md)에 기록했다.

### Top-10 문맥·공식 배치 복원

신경망 HWR·수식 문맥·공식 배치 결과 뒤에 연구용 Top-10 회수 경로를 연결했다. 기존 열이 이미 유효하면 보존하고, 숫자가 두 개 이상인 깨진 식에서 최소 변경 복원안이 하나뿐일 때만 최대 두 글자를 확정한다. 식 순서와 배치는 기존 경로가 소유하며 Top-10의 원래 순서는 사용하지 않는다. 단일 영문 변수는 숫자로 바꾸지 않고 산술 결과도 계산하지 않는다.

직접 수집은 문자 358/387(92.51%), 수식 완전일치 78/95(82.11%)로 올랐고 CROHME 반복 진단은 문자 7516/9535(78.83%), 수식 완전일치 288/984(29.27%)로 증가했다. 두 범위 모두 문자·수식 회귀는 0건이다. 후보 상한은 직접 88/95, CROHME 773/984까지 넓어졌지만 계약·비용이 증가하고 반복 개발 자료에서 정했으므로 `shadow_runtime_only`다. 구현, 기각된 변수 숫자화, SHA와 승격 조건은 [reports/WIDE_CANDIDATE_SYNTAX_PLACEMENT_LOOP_20260819.md](reports/WIDE_CANDIDATE_SYNTAX_PLACEMENT_LOOP_20260819.md)에 기록했다.

### Top-10 문맥 신경망 재배치 감사

고정된 자체 수식 문맥 신경망으로 Top-10 후보를 전면 재정렬하면 직접 수집 식 완전일치가 78/95에서 61/95로, CROHME 반복 진단이 288/984에서 272/984로 하락했다. `X→x`만 허용한 분기도 직접 수집 1식은 고쳤지만 CROHME 문자에서 개선 1건·회귀 3건이어서 기각했다. 런타임은 변경하지 않았으며 근거는 [reports/WIDE_CONTEXT_TOP10_PROBE_REJECTION_20260819.md](reports/WIDE_CONTEXT_TOP10_PROBE_REJECTION_20260819.md)에 기록했다.

### Top-20 짝 없는 괄호 피연산자 복원

Top-20은 후보 회수 상한을 직접 수집 88/95에서 91/95로 넓혔지만 전체 문맥 재정렬과 낮은 확률 후보 허용은 회귀 때문에 기각했다. 대신 이항 연산자와 관계 기호 사이의 짝 없는 닫는 괄호가 단일 세로 stroke이고 후보의 유일한 숫자가 `1`일 때만 복원하는 배치 가드를 추가했다. 직접 수집은 문자 359/387(92.76%), 식 완전일치 79/95(83.16%)가 됐고 CROHME 반복 진단은 변경·회귀가 모두 0건이다. 반복 개발 자료에서 정한 경계이므로 `shadow_runtime_only`이며 상세 근거는 [reports/TOP20_UNMATCHED_FENCE_PLACEMENT_LOOP_20260819.md](reports/TOP20_UNMATCHED_FENCE_PLACEMENT_LOOP_20260819.md)에 기록했다.

### 연산자 십자 슬롯 복원

정확히 `숫자-\bot-숫자`인 3항식에서 Top-20 앞 두 후보가 `\bot/\perp`, 유일한 산술 후보가 `+`, 궤적이 2-stroke 십자 형상일 때만 `+`를 확정하는 배치 가드를 추가했다. 누적 직접 수집은 문자 360/387(93.02%), 식 완전일치 80/95(84.21%)이며 이번 단계의 회귀는 0건이다. CROHME 반복 진단에서는 변경이 없었다. 독립 상업 검증 전까지 `shadow_runtime_only`이며 상세 근거는 [reports/OPERATOR_CROSS_SLOT_RESCUE_LOOP_20260819.md](reports/OPERATOR_CROSS_SLOT_RESCUE_LOOP_20260819.md)에 기록했다.

### 이중 직선 1 피연산자 복원

정확히 `숫자-\div-모호문자-=-모호문자`인 5글자 식에서 두 모호문자가 모두 단일 직선 stroke이고 각 후보의 최상위 숫자가 충분한 차이로 `1`일 때만 두 위치를 함께 복원한다. 누적 직접 수집은 문자 362/387(93.54%), 식 완전일치 81/95(85.26%)이며 이번 단계의 회귀는 0건이다. CROHME 반복 진단에서는 변경이 없었다. 독립 상업 검증 전까지 `shadow_runtime_only`이며 상세 근거는 [reports/DUAL_STRAIGHT_ONE_SLOT_RESCUE_LOOP_20260819.md](reports/DUAL_STRAIGHT_ONE_SLOT_RESCUE_LOOP_20260819.md)에 기록했다.

### 통합 공식 배치 복원 레이어

세 배치 가드를 `짝 없는 괄호 → 십자 연산자 → 이중 직선 1` 순서의 단일 shadow 런타임으로 합쳤다. 기존 Top-10 구문 기준에서 직접 수집은 3식 개선·0식 회귀, CROHME는 전 단계 변경 0건이며 CLI 재생 결과도 동일하다. 남은 14식 중 10식은 의미 동형 구분용 새 소유 문맥 데이터가, 4식은 HWR 후보 recall 개선이 필요하다. 통합 계약과 승격 경계는 [reports/FORMULA_PLACEMENT_RESCUE_INTEGRATION_20260819.md](reports/FORMULA_PLACEMENT_RESCUE_INTEGRATION_20260819.md)에 기록했다.
