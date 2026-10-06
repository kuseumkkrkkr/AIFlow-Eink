# V12 R4-r2 telemetry-first design

R4-r1은 두 정적 계약 결함으로 보존·불채택한다. R4-r2는 정책을 바꾸지 않고 계보와 empty-stage 판별만 강화한다.

## 정책 계보

- R3 dry policy catalog 파일의 존재와 SHA-256 `66390f17...9d2b360`을 live 검증한다.
- catalog를 stdlib gzip/json으로 읽어 pair `9,867`, planned action `197,340`, pair당 writer `20`, orientation `10/10`을 재검증한다.
- support `2`, minimum pair writer `16`, risk/action 정책, threshold/class/writer 규칙을 변경하지 않는다.
- 새 writer 범위 `256..319`는 제안만 하며 generation-only 소스와 독립 pre-generation 감사 전에는 열지 않는다.

## aggregate-only telemetry

미래 `PRE_FINAL_STAGE_TELEMETRY.json`은 compound guard와 failure receipt보다 먼저 exclusive temp create, flush, fsync, non-replacing rename으로 게시한다. row tensor, row metadata, token/label ID, writer/pair/class별 행은 포함하지 않는다.

Calibration survivor chain:

1. `requested_specs`
2. `specs_with_parent_candidates`
3. `specs_parent_disjoint`
4. `specs_topology_valid`
5. `specs_direct_top5`
6. `specs_homograph_clear`
7. `specs_duplicate_clear`
8. `specs_support2_complete`

Query survivor chain:

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

Final calibration survivor chain:

1. `calibration_rows_entering`
2. `calibration_rows_stored_top5_exact`
3. `calibration_rows_after_support_retention`

Final query survivor chain:

1. `query_rows_entering`
2. `query_rows_stored_top5_exact`
3. `query_rows_after_support_filter`
4. `query_rows_after_pair16_filter`

각 chain은 내부에서만 non-increasing을 검증한다. calibration과 query 크기를 서로 비교하지 않는다. 첫 0은 calibration → query → final calibration → final query 순으로 판정한다. 이후 `candidate_violations > 0`이면 `final.candidate_violations`, 아니면 `NONE`이다.

Final rejection counters는 calibration/query stored-Top5 mismatch와 support drop을 분리하며, query pair<16 drop을 별도 기록한다. 따라서 final admission 직전 calibration 0과 query 0이 다시 합쳐지지 않는다.

## 불변 경계

- R3 failure는 `INDETERMINATE_CALIBRATION_OR_QUERY` 그대로 유지하며 재해석하지 않는다.
- R3 writer `192..255`, R4-r1, writers `096..127`, Legacy, real, CROHME, MathWriting은 열지 않는다.
- generation, physics, frozen HWR forward, optimizer/backward, training, postprocess, HWR/checkpoint/runtime 변경을 수행하지 않는다.
- generation execution은 별도 generation-only source와 독립 pre-generation audit 전까지 false다.
