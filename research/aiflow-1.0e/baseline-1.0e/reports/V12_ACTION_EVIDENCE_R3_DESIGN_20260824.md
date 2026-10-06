# v12 R3 action-evidence 설계

상태: `DESIGN_AND_STATIC_DRY_POLICY_ONLY`

## 목적

R2 raw는 독립 감사에서 query split 0행으로 REJECT됐다. R2 산출물은 불변 보존하며 재실행, 후처리, 학습에 사용하지 않는다. R3는 R2의 특정 class, threshold, writer에 맞추지 않고 action acceptance와 context fallback의 책임을 분리한다.

## R2 근거와 해석

- R2: calibration 33,806행, query 0행, retained pair 0, identity-only pair 9,867.
- 주요 telemetry: `generated_top5_unadmitted=123565`, `generated_top5_homograph_collision=10377`.
- 최종 candidate violation 1건으로 bank-wide candidate gate도 실패했다.
- 위 카운터는 시도 단계가 중첩될 수 있으므로 합산 복구율이나 예상 정확도로 사용하지 않는다.
- R3 dry 단계는 기존 catalog의 pair·risk coverage와 writer allocation 가능성만 검증한다. 실제 생성 수율은 `UNPROVEN`이다.

## action acceptance

미래 R3 query action은 저장 직전 frozen HWR direct re-inference에서 다음을 모두 만족해야 한다.

1. catalog baseline이 generated frozen Top-5에 존재한다.
2. catalog candidate가 generated frozen Top-5에 존재한다.
3. candidate rank, Top-1 margin bin, Top-5 entropy bin, stroke bucket이 catalog risk stratum과 일치한다.
4. candidate별 calibration support가 2개 이상이며 calibration/query parent fingerprint가 disjoint다.
5. source truth topology가 보존되고 baseline/candidate action에 필요한 stroke topology가 호환된다.
6. tensor가 finite/unit-box/uniform-time이고 external-approved parent만 사용한다.
7. source→generated spatial RMS가 identity threshold보다 크며 duplicate hash가 없다.

baseline 또는 candidate membership가 불일치하면 해당 row만 drop한다. pair의 unique admitted writer가 16명 미만이면 그 pair의 query를 전부 drop하고 `identity_only`로 기록한다.

## 보조 Top-5와 homograph 책임 분리

- baseline/candidate 외 보조 Top-5에 globally-unadmitted label이 있어도 action row 자체를 무효화하지 않는다.
- 보조 unsupported label은 `auxiliary_unsupported_present` telemetry로 기록한다.
- adapter runtime에서는 Top-5 option 중 unsupported candidate를 승격하지 않고 해당 option을 identity 처리한다.
- homograph collision은 context-owned whole-row identity다. query 생성 실패와 구분해 `context_owned_identity`로 기록하며 action 학습 대상에서 제외한다.
- pair/token/writer ID와 homograph 문자열은 provenance·grouping key일 뿐 scorer feature가 아니다.

## final fail-closed

save 직전 전체 admitted query를 frozen HWR로 다시 inference한다.

- 각 row의 baseline/candidate membership와 catalog risk match를 재검증한다.
- 불일치 row는 저장 후보에서 제거한다.
- 제거 후 pair별 unique writer 16명 조건을 다시 계산하고 미달 pair 전체를 제거한다.
- 남은 bank에서 candidate violation이 1건이라도 있으면 저장 자체를 거부한다.
- 최종 report의 `candidate_action_within_frozen_top5`, `pair_writer_min16_or_identity_only`, `bank_candidate_violations_zero`는 계산값이어야 하며 상수 기록을 금지한다.

## 불변 데이터 계약

- frozen 128×5·372-class HWR/checkpoint/runtime 변경 0.
- raw parent는 commercial-approved external bank만 사용한다.
- calibration/query parent fingerprint disjoint, topology 보존, delta_t uniform-time 고정.
- writer ID, pair ID, token/label ID feature 0.
- adapter는 baseline Top-5 set을 exact 보존한다.
- optimizer/backward 실행은 CUDA 필수, CPU silent fallback 0. 현재 R3 dry 단계에는 optimizer/backward가 없다.
- R2 raw NPZ/metadata, writers096..127, Legacy, real, CROHME, MathWriting 접근 0.

## 제안 writer/one-shot 경계

- R3 generation 후보 writer range: `192..255` 64명.
- 각 directed pair는 서로 다른 writer 20명, pair×writer action 최대 1개, orientation 10/10으로 계획한다.
- writers192..255는 static independent audit 전 생성하지 않는다.
- writers096..127은 R3 candidate가 writers000..095와 별도 development evidence에서 freeze된 뒤의 첫 synthetic one-shot outer로 계속 봉인한다.
- synthetic outer 통과도 fresh real writer/formula acceptance를 대체하지 않는다.

## 단계

1. R3 design + static dry policy catalog/coverage feasibility.
2. 독립 static audit.
3. 별도 generation-only source 작성과 독립 pre-generation audit.
4. 새 bank one-shot 생성.
5. independent raw/full/cross audit PASS 뒤에만 adapter development 검토.

현재 허용 범위는 1단계뿐이다.
