# CROHME 학습 경계 정정

## 정정 결론

- CROHME와 MathWriting은 **검증 전용**이다.
- `crohme_standard_context_20260820_r1_research`에서 만든 문맥 체크포인트와
  그 체크포인트로 얻은 개선 수치는 상용·기준선·모델 선택에서 제외한다.
- 원본 파일은 삭제하지 않고 `QUARANTINED.md`와 함께 감사 기록으로만 보존한다.

## 유효 기준선

- 형태 HWR: 프로젝트 보정 372-class checkpoint, SHA-256
  `04f8608aebcf6c02d45ad6f5735229b9eaa2c4b4e1be0db4793d02273ef2d00e`
- 상업권리 문맥 모델: candidate-validity checkpoint, SHA-256
  `ee0033fb42f59b09f3f130300fb5374710cfef010a71c94dbd48e4d1c93dc8b`
- 이 기준선의 CROHME test truth-group HWR는 Top-1 `72.6962%`, Top-5
  `93.7536%`, 정답이 Top-5 밖인 문자는 `749/11,991`이다.

## 재발 방지

- `training_data_guard_v1.py`가 학습 경로와 행 provenance에서
  `crohme`, `mathwriting`, `30_noncommercial_evaluation`을 차단한다.
- 새 학습 보고서는 CROHME/MathWriting row와 gradient update가 모두 0임을
  명시한다.
- CROHME 검증은 상업권리 데이터에서 모델·epoch·임계값을 모두 고정한 뒤
  별도 no-gradient 평가 명령에서만 수행한다.
