# V12 R4 telemetry-first action-evidence design

## 판정 경계

- R3는 `INDETERMINATE_CALIBRATION_OR_QUERY`로 봉인한다. calibration 0 또는 query 0 중 하나로 추정하지 않는다.
- R3 writer `192..255`, source, output은 재사용·재실행하지 않는다.
- R4는 새 writer `256..319`만 제안한다. 이 범위도 generation-only 소스와 별도 독립 pre-generation 감사 전에는 열지 않는다.
- R3의 action 정책, morph strength, risk stratum, class/pair 선택, support 2, pair writer 16 조건은 변경하지 않는다.
- 이 단계에서는 생성·physics·frozen HWR forward·optimizer/backward·학습·후처리를 실행하지 않는다.

## 선행 telemetry 영수증

미래 R4 one-shot은 `PRE_FINAL_STAGE_TELEMETRY.json`을 final compound guard보다 먼저 원자적·배타적으로 게시해야 한다.

- 고정 임시 파일을 `O_EXCL`로 생성하고 `flush`·`fsync` 후 기존 final을 덮어쓰지 않는 rename으로 게시한다.
- final 또는 temp가 이미 있으면 fail-closed한다.
- 게시 뒤에만 empty-stage guard와 `GENERATION_FAILED.json` 작성을 허용한다.
- 영수증에는 aggregate count와 lineage hash만 저장한다.
- raw tensor, row metadata, token/label ID, writer별·pair별·class별 행은 저장하지 않는다.

## calibration stage

다음 survivor count를 순서대로 기록한다.

1. `requested_specs`
2. `specs_with_parent_candidates`
3. `specs_parent_disjoint`
4. `specs_topology_valid`
5. `specs_direct_top5`
6. `specs_homograph_clear`
7. `specs_duplicate_clear`
8. `specs_support2_complete`

보조 rejection count로 `parent_candidate_rows`, `topology_rejects`, `invalid_tensor_rejects`, `direct_top5_rejects`, `homograph_rejects`, `duplicate_rejects`를 기록한다.

## query stage

다음 survivor count를 순서대로 기록한다.

1. `planned_actions`
2. `actions_calibration_supported`
3. `actions_shared_topology`
4. `actions_truth_parent_selected`
5. `actions_relative_parent_selected`
6. `actions_with_morph_candidate`
7. `actions_direct_membership`
8. `actions_risk_exact`
9. `actions_homograph_clear`
10. `actions_accepted`

보조 count로 `morph_attempts`, `morph_topology_drops`, `auxiliary_unsupported_telemetry`, `invalid_tensor_rejects`, `duplicate_rejects`를 기록한다. auxiliary unsupported는 action drop 사유가 아니다. homograph는 context-owned identity다.

## final stage

- `calibration_rows_entering`
- `query_rows_entering`
- `rows_stored_top5_exact`
- `stored_top5_mismatch_drops`
- `rows_after_support_filter`
- `support_drops`
- `rows_after_pair16_filter`
- `pair_lt16_rows_dropped`
- `candidate_violations`

`candidate_violations`는 0이어야 하며, pre-save direct re-inference와 pair 16 재계산 계약은 R3에서 변경하지 않는다.

## first_empty_stage

- calibration, query, final survivor count를 위 순서로 순회한다.
- 첫 0 survivor의 완전한 stage 이름을 `first_empty_stage`로 기록한다.
- calibration support가 0이면 query stage보다 calibration stage를 우선한다.
- survivor가 모두 양수지만 `candidate_violations > 0`이면 `final.candidate_violations`다.
- empty stage와 candidate violation이 모두 없으면 `NONE`이다.
- 이 값은 메시지나 예외 문자열에서 추정하지 않고 aggregate count만으로 계산한다.

## 불변 경계

- external-approved parent, calibration/query parent fingerprint disjoint, topology, finite/unit-box, uniform-time, duplicate/identity, frozen Top-5 exact 계약을 유지한다.
- scorer feature에 pair/token/writer ID를 넣지 않는다.
- writers `096..127`, Legacy, real, CROHME, MathWriting을 열지 않는다.
- HWR/checkpoint/runtime을 변경하지 않는다.
- 실제 optimizer/backward 학습은 이후 별도 승인 시 CUDA 필수이며 CPU fallback은 금지한다.
