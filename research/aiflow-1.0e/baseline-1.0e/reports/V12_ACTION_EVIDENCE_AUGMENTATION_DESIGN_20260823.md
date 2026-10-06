# v12 action-evidence clean-room augmentation 설계

상태: `DESIGN_AND_DRY_CATALOG_ONLY`

## 목적

- frozen 128×5·372-class HWR/checkpoint/runtime는 변경하지 않는다.
- v11 후단 adapter가 실패한 원인인 candidate별 안전 action 표본 부족을 clean-room 증강으로 보강한다.
- writers096..127은 v12 frozen candidate의 첫 untouched synthetic outer로 계속 봉인한다.

## 입력 경계

- confusion catalog: 이미 소비된 synthetic writers000..095의 frozen-HWR Top5만 사용한다.
- trajectory parent: commercial-approved `external_profiled_augmented`와 clean-room pen physics만 사용한다.
- 금지: project evaluation/raw, CROHME, MathWriting, Legacy, fresh acceptance.
- r3에서 관측된 성공/실패 class ID는 선택·가중·feature에 사용하지 않는다.

## Catalog 규칙

- whole-row hard gate: Top5 전부 globally admitted, homograph collision 없음.
- candidate hard gate: truth topology와 candidate topology 호환.
- 모든 `baseline Top1 → Top5 candidate` directed pair를 catalog한다.
- label-free risk stratum은 candidate rank, Top1 margin bin, Top5 entropy bin, stroke-count bucket으로만 구성한다.
- pair/token ID는 생성 provenance와 evidence grouping에만 사용하고 adapter 숫자·범주 feature에는 넣지 않는다.

## 증강 계획

- 신규 writer ID: 128..191, 총 64명.
- writer seed: `20260824 + global_writer_id × 4099`.
- 각 directed pair마다 먼저 서로 다른 synthetic writer 20명을 고정하고, writer 한 명당 action 하나를 해당 pair의 관측 risk strata에 deterministic round-robin 배분한다.
- 따라서 pair별 unique writer는 정확히 20명이며 Wilson 최소 16-writer 증거보다 작아질 수 없다.
- pair×stratum 전체를 catalog하되 모든 unit에 20건씩 생성하지 않아 bank 규모를 제한한다.
- writer×candidate calibration anchor 2개는 여러 pair에서 공유한다.
- query action은 ambiguous morph, bounded affine, arc-length progress warp, clean-room pen dynamics를 적용한다.
- delta_t는 uniform-time 계약을 유지하고 시간 변화는 XY progression에만 반영한다.
- 실제 parent fingerprint는 calibration/query 사이 disjoint, synthetic/tensor/latent/RNG hash는 writers000..127과 disjoint여야 한다.

## Admission gate

- topology/stroke/time/finite/unit-box/duplicate/identity/provenance gate 전부 통과.
- frozen Top5 candidate set 밖 action 금지, candidate violation 0.
- cross-writer OOF candidate evidence에서 observed regression은 즉시 veto.
- regression Wilson upper ≤ .20 유지. 무회귀 action 16건 미만이면 identity.
- support/deficiency/writer×support 비회귀와 candidate set exact를 유지한다.

## 단계 경계

1. 현재: source + dry catalog만 생성. bank generation/training 0.
2. source/dry catalog 독립 static audit.
3. PREPARED-ONLY exact2 생성 후 독립 pre-generation audit.
4. bank 생성 후 full/cross-bank 독립 audit.
5. consumed000..095에서 adapter development/freeze.
6. writers096..127 one-shot outer.
7. fresh real writer/formula 1회 acceptance 전 product promotion 금지.
