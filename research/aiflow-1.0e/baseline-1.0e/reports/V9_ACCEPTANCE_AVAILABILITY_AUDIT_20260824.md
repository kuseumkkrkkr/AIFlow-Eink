# v9 acceptance 가용성 메타 감사

## 판정

`fresh-context-acceptance-20260820-r2`는 v9/R4의 완전 미사용 상용 acceptance로 사용할 수 없다.

## 근거

- 동결 manifest SHA-256: `2b7dad97aa417bee25213e127c36cd1790c4c6f36f946bc798a7c2770b41e54f`
- 동결 당시: 2 writer (`writer_010`, `writer_012`), 53 formula, 학습 수행 0, 동결 전 prediction open 0.
- manifest gate: `ready_for_overall_acceptance_evaluation=true`이지만 `ready_for_adoption=false`.
- sparse truth (`|`, `O`, `o`)가 없어서 `known_sparse_truth_present=false`.
- 이 manifest와 frozen dataset은 이미 다음 계열 산출물에서 참조됐다.
  - `candidate_validity_fresh_acceptance_20260820_r1`
  - `commercial_hwr_augmentation_fresh_acceptance_20260822_r1`
  - `commercial_hwr_cleanroom_profiled_fresh_acceptance_20260823_r1`
  - `commercial_hwr_cleanroom_physics_fresh_acceptance_20260823_r1`
  - `hierarchical_writer_model_v7_20260823_r1`
  - `writer_adaptation_v8_20260823_r1_shadow`

## 영향

- 해당 세트는 historical/consumed evidence로만 보존한다.
- R4-r2 adapter 또는 이후 상용 후보의 one-shot acceptance에 다시 열지 않는다.
- 현 시점에서 확인된 완전 미사용 project-owned/commercial acceptance는 없다.

## 다음 acceptance 수집 조건

- 새 writer와 새 formula를 collection 시점부터 freeze한다.
- writer ID·session·formula hash·stroke source hash·ownership을 receipt로 고정한다.
- v9 adaptation calibration, synthetic bank 개발, 기존 fresh set과 모두 disjoint를 검증한다.
- 최소 coverage는 실험 전에 사전등록하며, 숫자·영문·연산자·괄호 및 `|/O/o` 혼동 표본을 포함한다.
- 그 전까지 제품 승격은 `false`다.
