# AIFlow 1.0e accuracy upgrade implementation

## 상태

- 구현 단계: P0~P2 기반 코드 반영
- 학습 루프 실행: 없음
- 새 checkpoint 생성: 없음
- Hugging Face 업로드: 없음
- 제품 runtime 기본값 변경: 없음
- 제품 채택: 보류

## 반영한 계약

- `scripts/accuracy_upgrade_contract_v1.py`
  - `transform.bbox` 기반 원본 좌표 복원
  - 수식 전체 기준 상대 위치·크기·중심 계산
  - writer 단위 outer LOO와 deterministic inner split
  - padding을 제외한 후보 축 합산 KL
  - 최소 LaTeX 정규화와 파일 SHA 계약
- `scripts/audit_accuracy_upgrade_v1.py`
  - raw/candidate record coverage 검사
  - raw writer lineage 검사
  - 위치 문맥이 모두 0인지 검사
  - nested split manifest 생성
- `scripts/formula_context_ranker_10e.py`
  - 2-layer, hidden 128, 4-head online-only formula context ranker
  - zero-initialized residual head
  - candidate/row padding mask

## 기존 pipeline 변경

- `build_trocr_hwr95_candidates_v1.py`
  - local glyph shape와 formula-coordinate geometry를 분리
  - normalized `(0.5, 0.5)` 중심으로 이웃 거리를 계산하지 않음
  - 기존 Top-K candidate와 raw input 보존 계약은 유지
- `train_ocr_decision_adapter_10e.py`
  - formula-relative geometry feature를 numeric input에 포함
  - checkpoint에 feature contract 기록
- `train_online_candidate_distill_10e.py`
  - raw writer lineage를 outer fold의 기준으로 사용
  - 후보 밖 정답 row를 batch에서 제거하지 않음
  - candidate-axis-summed KL 사용
  - 실제 HWR writer-LOO state bundle 지원
  - prediction provenance 기록
  - `--strict-lineage`에서 all-writer final encoder와 teacher 계보 누수를 거부
- `train_teacher_ensemble_raster_free_reranker_10e.py`
  - teacher-heavy variant도 동일한 후보 축 합산 KL 계약 사용
- `train_auxiliary_nested_distill_10e.py`
  - auxiliary/original writer 판정을 raw lineage로 변경
  - HWR LOO state와 strict teacher provenance 지원
- `evaluate_online_distill_nested_guard_v1.py`
  - `--strict-nested --inner-predictions` 경로 추가
  - outer held writer를 제외한 inner-OOF calibration row를 요구
  - legacy calibration 결과는 `strict_nested=false`로 남음

## 실제 검증 결과

새 후보 cache:

`artifacts/accuracy_upgrade_20260910_p1_candidates_r3/candidates.jsonl.gz`

감사 결과:

- raw records: 387
- candidate records: 387
- formulas: 95
- writers: 7
- source bbox 복원: 387/387
- 비영 상대 위치 문맥: 374/387
- all context zero: false
- outer writers: 7
- inner folds: 3

검증 명령:

```powershell
python scripts/accuracy_upgrade_contract_v1.py
python scripts/formula_context_ranker_10e.py
python scripts/training_data_guard_v1.py
python -m py_compile scripts/accuracy_upgrade_contract_v1.py scripts/audit_accuracy_upgrade_v1.py scripts/build_trocr_hwr95_candidates_v1.py scripts/formula_context_ranker_10e.py scripts/train_online_candidate_distill_10e.py scripts/train_auxiliary_nested_distill_10e.py scripts/train_teacher_ensemble_raster_free_reranker_10e.py scripts/evaluate_online_distill_nested_guard_v1.py
```

모든 self-test와 compile 검사는 통과했다. writer-LOO state loader도 2개 raw row smoke에서 7개 held-writer state와 372개 label을 확인했다.

## 다음 실행 전 필수 조건

- outer writer와 inner writer를 동시에 제외한 teacher 예측 파일을 먼저 생성해야 한다. 기존 teacher file에는 이 provenance가 없어 `--strict-lineage`에서 거부된다.
- formula context ranker는 아직 학습·평가·runtime 연결을 하지 않았다. 먼저 P1 위치 feature 기준선과 비교한 뒤 연결한다.
- 새 writer/device-disjoint data가 없으므로 이번 구현으로 상용 정확도를 주장하지 않는다.
- 최종 제품 후보는 strict nested 평가, fresh writer/formula/device 평가, zero observed regression을 모두 통과해야 한다.
