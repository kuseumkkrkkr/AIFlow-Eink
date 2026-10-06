# candidate_validity_coverage_v1 격리

- 상태: 학습 입력 금지
- 이유: 실제 제외 건수는 0건이었지만 v1 생성기가 CROHME 후보를 읽어 중복 여부를 검사했다. 평가셋이 코퍼스 생성 과정에 들어갔으므로 엄격한 검증 경계에서 재사용하지 않는다.
- 대체물: `candidate_validity_coverage_v2.jsonl.gz`
- v2 계약: 사용자 지정 동형문자군과 프로젝트 소유 dev만 사용하며 `crohme_used_for_generation_or_filtering=false`이다.

