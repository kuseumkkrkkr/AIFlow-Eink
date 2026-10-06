# AIFlow 1.0e 외부 신경망 연결 실험 보고서

## 결론

- TexTeller 외부 신경망을 동결한 상태로 연결하고, 기존 HWR Top-k 후보의 선택 점수에만 외부 시각 임베딩·외부 디코딩 수식 토큰 증거를 추가했다.
- 3-writer leave-one-writer-out 평가에서 후보가 없는 행까지 포함한 전체 수식 Exact는 **16/47 (34.04%) → 35/47 (74.47%)**로 상승했다.
- 기존 후보에 정답이 들어오는 비율은 변하지 않아 **199/211 (94.31%)**이다. 따라서 나머지 12개 행은 이 선택기가 복구할 수 없다.
- 선택된 토큰이 기존 Top-k 밖으로 나간 사례는 **0건**이다. 신규 토큰 생성, 행 삭제, stroke 재그룹화, relation 변경은 하지 않았다.
- 다만 TexTeller 단독 디코딩도 이 47개 수식에서 **45/47 (95.74%)**를 직접 맞혔다. 따라서 이번 상승은 “독립적인 새 OCR 능력”이라기보다 외부 모델의 강한 증거를 기존 후보 선택에 주입한 결과로 해석해야 한다.
- 판정: **shadow-only 유지**. 운영 모델 승격 근거로 사용하지 않는다.

## 실험 경계

| 항목 | 적용 |
|---|---|
| 외부 신경망 | OleehyO/TexTeller, revision `7b96df06b9d81cdb129c3bef68b7250bc3e2b0ea` |
| 외부 가중치 | 전부 frozen, trainable parameter 0 |
| 학습 대상 | 후보 선택 adapter 159,185 parameters |
| 외부 입력 | encoder mean-pool feature, frozen decoded LaTeX token evidence |
| 기존 HWR | stroke/순서/그룹/후보 생성 경로 변경 없음 |
| 후보 계약 | 기존 HWR Top-k 중 1개만 선택 |
| 평가 | 47 formulas, 211 rows, 3 writers, writer-LOO |

이번 실행은 **외부 신경망을 추가 입력으로만 사용**했다. adapter의 정답 라벨은 기존 프로젝트 소유 라벨을 사용했으므로, “외부 데이터만으로 학습한 모델”이라고 부를 수는 없다. 완전한 external-only supervision 실험은 별도 트랙이다.

## 결과

### 전체

| 지표 | 기존 HWR 후보 1위 | 외부 증거 adapter |
|---|---:|---:|
| 후보 포함 행 기준 Top-1 | 77.89% | 98.49% |
| 전체 수식 Exact | 34.04% (16/47) | 74.47% (35/47) |
| 후보 recall | 94.31% (199/211) | 동일 |
| 외부 TexTeller 직접 수식 Exact | - | 95.74% (45/47) |

### Writer-LOO 전체 수식 Exact

| held writer | 기존 | adapter |
|---|---:|---:|
| `3e41…` | 37.50% (3/8) | 75.00% (6/8) |
| `baed…` | 31.25% (5/16) | 93.75% (15/16) |
| `ed67…` | 34.78% (8/23) | 60.87% (14/23) |

세 writer fold 모두 전체 수식 Exact가 개선됐지만, 표본이 작고 별도 formula-disjoint fresh acceptance set이 없어 일반화 증거로는 부족하다.

## 계약·재현 산출물

- 실행 코드: `scripts/train_ocr_text_evidence_adapter_10e.py`
- 계약 감사 코드: `scripts/audit_ocr_text_evidence_adapter_10e.py`
- 체크포인트: `artifacts/ocr_text_evidence_adapter_10e_20260830/ocr_text_evidence_adapter.pt`
- 평가 원본: `artifacts/ocr_text_evidence_adapter_10e_20260830/evaluation.json`
- 재감사 결과: `artifacts/ocr_text_evidence_adapter_10e_20260830/audit.json`

재감사 결과 `candidate_contract.violations = 0`이다.

## 다음 채택 게이트

1. 새 writer 및 formula-disjoint 수식으로 동일한 writer-LOO/holdout 평가를 다시 수행한다.
2. 외부 TexTeller 직접 출력, HWR-only, adapter를 분리해 비교한다.
3. 후보 recall 94.31% 한계를 넘기려면 선택기가 아니라 별도의 외부/내부 후보 생성 연구가 필요하다.
4. 위 fresh set에서 fold regression과 데이터 중복 여부를 확인하기 전까지 1.0e 운영 승격은 보류한다.
