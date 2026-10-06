# v12 calibration-only adapter 실행 계약

상태: `DESIGN_ONLY_GENERATION_OPEN`

이 문서는 실행 계약만 고정한다. v12 action-evidence raw bank의 성공·독립 raw/full/cross 감사 전에는 postprocess, feature cache, optimizer, backward, writers096..127 open을 수행하지 않는다.

## 불변 경계

- frozen 입력은 `128×5`, 출력은 `372-class`다.
- HWR checkpoint, runtime, baseline logits와 각 glyph의 baseline Top-5 집합은 변경하지 않는다.
- adapter는 frozen Top-5 내부 순위만 조정할 수 있다.
- unsupported class, homograph collision, support 부족, 불확실 evidence는 whole-row `identity`다.
- project evaluation/raw, Legacy evaluation 300 glyph, known replay, fresh acceptance, CROHME, MathWriting은 학습·선택에 사용하지 않는다.
- consumed writers000..095는 development-only다. writers096..127은 frozen candidate의 첫 one-shot synthetic outer로 봉인한다.
- 실제 상용 승격은 새로 수집해 사전 동결한 real writer/formula acceptance 전까지 `false`다.

## 데이터 단계

1. v12 writers128..191 raw generation 종료 영수증을 확인한다.
2. 독립 raw audit와 full/cross-bank audit가 모두 PASS일 때만 postprocess를 허용한다.
3. action evidence는 writer-OOF risk evidence로만 사용한다. pair, token, truth class, candidate class, writer ID는 grouping/provenance key이며 모델 feature가 아니다.
4. calibration/query는 parent fingerprint와 sample row가 분리되어야 한다.
5. query truth는 prediction 완료 뒤 scoring 함수에서만 접근한다.

## 예측 API 계약

prediction payload에는 `labels`, `truth`, `sample_id`, `writer_id`, `session_id`, token 문자열, pair ID가 존재하면 안 된다.

허용 입력은 다음뿐이다.

- frozen Top-5 rank와 상대 logit margin
- Top-5 entropy
- calibration support count와 dispersion
- calibration prototype cosine 또는 distance
- global/calibration embedding advantage
- topology가 일치한 궤적의 DTW geometry cost
- calibration-only reliability와 regression-risk 통계

candidate token/index는 Top-5 action lookup과 calibration prototype 조회 key로만 사용한다. 숫자·범주형 feature로 인코딩하지 않는다. option 순서를 섞어도 prediction은 exact 동일해야 한다.

## 학습 계약

- optimizer 또는 backward가 있는 실행은 CUDA 필수다.
- `torch.cuda.is_available() == false`, CUDA OOM, device mismatch면 즉시 fail-closed한다. CPU silent fallback은 금지한다.
- frozen HWR는 gradient와 optimizer parameter에서 제외한다.
- writer-OOF fold마다 held writer는 model fit, prototype, reliability, precision/FPR, normalization 통계에서 완전히 제외한다.
- 동일 writer가 train/dev/check fold에 중복되면 실행을 거부한다.
- class/pair별 observed regression은 hard veto다.
- zero-regression action evidence가 writer-disjoint 16건 미만이거나 Wilson regression upper bound가 `.20`을 초과하면 identity다.
- architecture, action cost, threshold, support, shrinkage와 homograph 목록은 outer open 전에 hash-bound freeze한다.

## 개발·평가 순서

1. writers128..191 action evidence와 consumed writers000..095만으로 architecture와 gate를 개발한다.
2. development writer-OOF에서 aggregate Top-1이 양수이고 모든 writer/support/deficiency/writer×support가 비회귀이며 candidate violation이 0이어야 한다.
3. 통과한 exact model/config/source/hash만 writers096..127 prepared-only receipt에 결합한다.
4. 독립 preopen audit PASS 뒤 writers096..127을 정확히 1회 연다.
5. outer 실패도 소비된 immutable rejection이다. 결과 맞춤 threshold/class/writer 규칙, 재실행, 동일 outer 재사용은 금지한다.
6. outer 통과도 synthetic known-parent-manifold shadow evidence일 뿐 product evidence가 아니다.

## 필수 지표와 산출물

- aggregate/per-writer/per-support/per-deficiency/writer×support Top-1
- pseudo-formula exact는 synthetic diagnostic으로만 기록
- 실제 formula exact는 향후 사전 동결 real formula acceptance에서만 승격 지표로 사용
- paired improved/regressed/unchanged와 McNemar 또는 exact paired count
- per-class confusion과 candidate action trace
- original/adapted Top-5 set exact mismatch 수
- unsupported/homograph identity 수
- writer activation coverage와 identity fallback 비율
- split/config/source/checkpoint/bank/cache/model/report SHA-256

필수 파일은 source snapshot, STARTED marker, split manifest, feature schema, frozen config, model artifact, report, decision receipt, changed-prediction trace다. 모든 종료 분기는 source와 독립 audit hash를 보존한다.

## 판정

- 개발 gate 전: `PRE_ADAPTER_FAIL_CLOSED`
- writers096..127 one-shot 전: `PRE_OUTER_FAIL_CLOSED`
- synthetic outer 통과 후: `SYNTHETIC_SHADOW_CANDIDATE`
- fresh real writer/formula acceptance 전 product promotion: `false`

현재 v12 R2 generation이 실행 중이므로 이 문서는 어떤 bank row, label, feature, prediction도 열거나 생성하지 않았다.
