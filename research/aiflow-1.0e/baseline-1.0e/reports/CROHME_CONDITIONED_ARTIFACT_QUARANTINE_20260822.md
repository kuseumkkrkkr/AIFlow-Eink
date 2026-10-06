# CROHME 조건화 산출물 격리

## 제품·기준선에서 제외

- `artifacts/crohme_standard_context_20260820_r1_research/`: CROHME로 직접 학습한 체크포인트. 실행 진입점도 fail-closed로 폐기했다.
- `datasets/10_approved_external/deepmind_mathematics_dataset/derived/formula_context_v1.jsonl.gz`: CROHME 중복 48식을 제외한 합성 코퍼스.
- `datasets/00_project_owned/generated_context/candidate_validity_coverage_v1.jsonl.gz`: 제외는 0건이지만 CROHME를 읽어 생성 여부를 판단한 코퍼스.
- 위 v1 코퍼스나 과거 CROHME 오류 목록으로 보강된 candidate-validity 실험 산출물: 재현 연구용 shadow만 허용하고 제품 승격 금지.
- 과거 코드에서 CROHME 점수를 research/promotion gate에 포함한 masked, prompt, role, distilled context 보고서: 보고서의 채택 판정은 무효. 필요하면 수정된 코드로 프로젝트 소유 dev에서 재학습해야 한다.

## 현재 제품 경계

- 현재 HWR 및 owned-context 가중치는 프로젝트 소유·상업권리 데이터 선택을 유지한다.
- CROHME는 체크포인트 고정 뒤 무경사 보고에만 사용한다.
- 현재 grouping/context runtime은 posthoc shadow이므로 수식 결과를 자동 확정하지 않고 `REVIEW_REQUIRED`와 원시 잉크 fallback을 반환한다.

