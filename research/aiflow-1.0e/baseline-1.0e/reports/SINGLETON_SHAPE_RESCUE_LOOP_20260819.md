# 단일기호 보조 HWR 복원 개선 루프

## 결론

- 기존 HWR 후보열 전체를 교체하지 않고, 길이 1인 수식에만 writer-LOO 보조 shape head를 증거로 연결했다.
- 직접 수집 95식은 72/95(75.79%)에서 74/95(77.89%)로, 문자는 349/387(90.18%)에서 351/387(90.70%)로 증가했다.
- 실제 변경은 `/`와 `\times` 각 1건이며 개선 2식·회귀 0식이다.
- 기존 HWR Top-5 밖 선택, 새 token, 삭제, grouping 변경, 정답 label 유출은 모두 0건이다.
- 현재는 `shadow_runtime_only`다. 선택 임계값에 사용한 95식은 반복 관찰된 개발 자료이므로 제품 기본값으로 승격하지 않는다.

## 구조와 제한

```text
온라인 stroke
  → 기존 372-class HWR Top-5
  → 문맥 확정·공식 배치·구문 복원
  → 수식 길이가 1일 때만 보조 HWR 증거 확인
  → 기존 Top-5 안의 / 또는 \times만 제한적으로 재선택
```

보조 점수는 기존 정책과 확장 writer-LOO head 확률을 `0.4:0.6`으로 합친다. 직접 수집에서는 각 표본의 작성자를 제외한 head만 사용했다. `/`는 보조 확률 0.50 이상, `\times`는 0.45 이상일 때만 허용한다. 보조 1순위가 기존 HWR Top-5에 없으면 항상 기각한다.

이 설정은 일반 입력을 위한 새 제품 head가 아니다. 고정된 직접 수집 writer-LOO 증거 파일과 그 SHA-256을 함께 요구하는 재현 가능한 shadow 평가 경로다.

## 평가

| 범위 | 기존 | 보조 복원 | 변화 |
|---|---:|---:|---:|
| 직접 수집 문자 Top-1 | 349/387 (90.18%) | 351/387 (90.70%) | +2글자 |
| 직접 수집 수식 완전일치 | 72/95 (75.79%) | 74/95 (77.89%) | +2식 |
| 외부 collision-free 글자 holdout | 14179/18330 (77.35%) | 14187/18330 (77.40%) | +8, 회귀 0 |
| CROHME 단일기호 수식 | 0/984 | 적용 0 | 제품 검증 아님 |
| 후보 위반·새 token·삭제·grouping 변경 | 0·0·0·0 | 0·0·0·0 | 유지 |

직접 수집 변경:

- `aiflow_0049`: `1` → `/` (보조 확률 0.5130)
- `aiflow_0072`: `x` → `\times` (보조 확률 0.4784)

전체 후보열을 보조 head로 교체하는 분기는 기각했다. 확장 writer-LOO 단독은 문자 353/387이었지만 수식 완전일치가 71/95로 떨어져 5식이 회귀했다. 0.6 전면 fusion도 4식 개선과 4식 회귀가 동시에 발생했다. CROHME에서 0.4 제품 fusion은 17식 개선과 14식 회귀가 함께 발생했다.

## 재현 증거

- runtime config SHA-256: `c1287fdd1e49814abe21d9c527b04d949ff23a5d961839fb9116680c3630a3bf`
- 직접 보조 후보 SHA-256: `99c1e3a7a4f32d4dd41a1ea1c303e7d1ef49b762196f5b03c5deb30ff305e7ce`
- 단일기호 평가 SHA-256: `38137fd76e41caffc03118fa11327bc8f2dce5a13e57e24ace4f96154f55abf9`
- 해시 고정 통합 출력 SHA-256: `cf230bab5b68fa1e577903fb4a9e2944cacc379a6648aca578eade34c2fa118b`
- 레이어 OFF 호환 출력 SHA-256: `7246f95b8be26b86d29b031050b502c30990c384a8e9e3c5394a3aec4c4062ff` (기존 387글자 대비 변경 0)
- 고정 manifest: `artifacts/singleton_shape_rescue_20260819_r1_shadow/singleton_shape_rescue_runtime_manifest.json`

기존 최종기 명령에 아래 두 옵션을 함께 추가한다.

```powershell
--singleton-shape-config artifacts/singleton_shape_rescue_20260819_r1_shadow/singleton_shape_rescue_runtime_config.json
--singleton-shape-evidence <SHA-256이 config와 일치하는 writer-LOO 후보 JSONL.gz>
```

## 다음 승격 조건

- 신규 작성자의 untouched 단일기호 수식으로 `/`, `\times`, `1`, `x` 비회귀를 검증한다.
- 제품 입력에서 작성자를 제외한 head 또는 동등한 누수 방지 경로를 정의한다.
- 수식 단위 회귀 0, 후보 보존 100%, 새 token·삭제·grouping 변경 0을 동시에 유지한다.
- 이 조건 전에는 제품 기본 경로를 켜지 않는다.
