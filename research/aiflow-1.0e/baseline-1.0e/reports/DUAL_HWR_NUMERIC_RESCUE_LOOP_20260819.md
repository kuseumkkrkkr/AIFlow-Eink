# 이중 HWR 숫자 operand 복원 개선 루프

## 결론

- 전면 HWR fusion은 회귀 때문에 기각하고, 기존 경로가 숫자 문법으로 무효일 때만 보조 경로의 숫자 operand를 받는 gate를 추가했다.
- 직접 수집은 문자 351/387(90.70%)에서 355/387(91.73%), 수식 완전일치 74/95(77.89%)에서 76/95(80.00%)로 증가했다.
- CROHME 비상업 반복 진단은 문자 7511/9535(78.77%)에서 7513/9535(78.79%), 수식 완전일치 284/984(28.86%)에서 286/984(29.07%)로 증가했다.
- 직접·CROHME 모두 개선 2식, 문자 회귀 0, 수식 회귀 0이다.
- 현재는 `shadow_runtime_only`이며 제품 기본값은 꺼져 있다.

## 동작 계약

```text
기존 HWR Top-5 → 기존 문맥 확정·공식 배치 ───────────────┐
                                                       ├→ 숫자 operand gate → 최종 수식열
0.6 보조 HWR fusion → 보조 문맥 확정·공식 배치 ─────────┘
```

gate는 아래 조건을 모두 만족할 때만 한 식에서 최대 두 글자를 바꾼다.

- 수식 순서와 배치는 기존 경로가 소유한다. 두 경로의 순서가 다르면 식 전체를 보류한다.
- 기존 선택이 `operand`, 보조 선택이 숫자일 때만 후보로 본다.
- 기존 수식열은 단순 숫자 문법으로 무효이고, 변경 후 수식열은 처음으로 유효해야 한다.
- 숫자가 두 개 이상이고 이항 연산자 또는 관계기호가 있어야 한다.
- 연산자·관계기호·괄호·루트는 바꾸지 않는다.
- 산술 결과의 참·거짓을 계산하지 않는다.
- 선택 문자는 기존·보조 HWR Top-5 합집합 안에 있어야 한다.

마지막 항목 때문에 이전의 “기존 Top-5만 권위 후보” 계약은 이 shadow 레이어에서만 “두 HWR Top-5 합집합”으로 명시적으로 확장된다. 직접 3글자와 CROHME 2글자가 기존 Top-5 밖이지만, 두 경로 합집합 밖 생성은 0건이다.

## 결과

| 범위 | 기존 | gate 적용 | 변화 |
|---|---:|---:|---:|
| 직접 문자 Top-1 | 351/387 (90.70%) | 355/387 (91.73%) | +4, 회귀 0 |
| 직접 수식 완전일치 | 74/95 (77.89%) | 76/95 (80.00%) | +2식, 회귀 0 |
| CROHME 문자 Top-1 | 7511/9535 (78.77%) | 7513/9535 (78.79%) | +2, 회귀 0 |
| CROHME 수식 완전일치 | 284/984 (28.86%) | 286/984 (29.07%) | +2식, 회귀 0 |
| 두 HWR 합집합 밖 선택 | 0 | 0 | 유지 |
| 연산자·관계·괄호·배치·grouping 변경 | 0 | 0 | 유지 |

직접 수집 개선:

- `aiflow_0024`: `4\mathscr{C}\div\mathscr{C}=6` → `48\div8=6`
- `aiflow_0052`: `2\times8=ib` → `2\times8=16`

CROHME 반복 진단 개선:

- `26_em_99.inkml`: 숫자열 내부 `o` → `0`
- `RIT_2014_108.inkml`: 숫자식 첫 operand `w` → `2`

CROHME에서 두 경로의 배치 순서가 달랐던 5식은 전부 보류했다.

## 기각 분기

- `x/X`, `S/5`, `\chi/x`를 직접식 맥락에 맞춘 세 규칙은 각각 직접 1식에만 적용되고 CROHME 적용 사례가 0이라 과적합으로 기각했다.
- 0.6 보조 후보열을 전부 사용하면 직접식 4개 개선과 4개 회귀가 동시에 발생해 기각했다.
- 수식 단위로 보조 경로 전체를 고르는 초기 gate는 CROHME에서 문자 2개를 고치면서 2개를 망가뜨려 기각했다.
- `operand→숫자`만 허용한 현재 gate에서 이 회귀가 사라졌다.

## 재현 증거

- runtime config SHA-256: `a3ae4c6e196c7cfc219b83617f34f39fb8b6ecb5f1a5eabed7e8d5dc83c15b25`
- 평가 SHA-256: `256cfd069e2ae2436213e6a22b1682b197b12edf6401ee9f824d6019c68acc73`
- 직접 병합 runtime SHA-256: `6785a5afef6fb133f1ced28f50ae3bafddb6fe10b810dfc56a59d62ff2bde91c`
- 고정 manifest: `artifacts/dual_hwr_numeric_rescue_20260819_r1_shadow/dual_hwr_numeric_rescue_runtime_manifest.json`

실행 예시는 다음과 같다.

```powershell
python scripts/dual_hwr_numeric_rescue_v1.py `
  --baseline-finalized <기존 최종기 JSONL.gz> `
  --auxiliary-candidates <0.6 보조 HWR 후보 JSONL.gz> `
  --auxiliary-finalized <보조 최종기 JSONL.gz> `
  --config artifacts/dual_hwr_numeric_rescue_20260819_r1_shadow/dual_hwr_numeric_rescue_runtime_config.json `
  --output <병합 출력 JSONL.gz>
```

## 다음 승격 조건

- 신규 작성자의 untouched 숫자식에서 operand→숫자 개선과 비회귀를 확인한다.
- 두 HWR 경로를 한 encoder pass와 두 head로 합쳐 추론 비용을 측정한다.
- 기존 Top-5 계약을 제품에서 합집합 계약으로 바꿀지 별도 승인한다.
- 그 전에는 제품 기본 경로로 승격하지 않는다.
