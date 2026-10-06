# 계층형 작가 모델 v6 경계 수정

## 수정 이유

- r5의 project 합성 bank 부모는 `project_owned_ownership_eval_95.jsonl.gz`다.
- 부모 387행의 역할은 `box_local_project_owned_evaluation_only`, split은 `project_owned_evaluation`이다.
- 따라서 해당 계보는 학습·증강·후보 선택·상용 채택의 기본 입력이 될 수 없다.

## r6 기본 모드

- catalog 입력: `external_profiled_augmented.npz` 4,096행만 사용.
- catalog 토큰: 371종. project evaluation bank에만 있던 `=`와 출력 어휘에 없는 `t`는 기본 catalog에서 결측으로 보고한다.
- 수식 스모크는 외부 catalog만으로 가능한 `A+3-7`, `x^2+y^2+25`, `(a+b)/2`로 변경했다.
- 12개 수식 생성물과 61개 영숫자 writer assignment에서 project 계보 선택은 0건이다.
- project evaluation 파일은 기본 모드에서 로드하거나 hash하지 않는다.

## 명시적 평가 스모크

- `--allow-evaluation-smoke`를 지정한 경우에만 project evaluation 합성 bank를 읽는다.
- 이 모드는 `project_evaluation_smoke_only=true`, `commercial_augmentation_mode=false`다.
- training, augmentation, adoption admission은 모두 `false`로 고정한다.

## 기본 모드 검증

| 항목 | 결과 |
|---|---:|
| external catalog | 4,096행 / 371토큰 |
| project evaluation loaded | false |
| 생성 수식 | 3종 × 4작가 = 12개 |
| 선택된 project 계보 | 0건 |
| compact Top-1 / Top-5 | 68.85% / 100.00% |
| slanted Top-1 / Top-5 | 70.49% / 100.00% |
| wide Top-1 / Top-5 | 65.57% / 100.00% |
| quick Top-1 / Top-5 | 62.30% / 100.00% |
| exact glyph partition / stroke provenance / finite coordinates | 통과 |
| CROHME / MathWriting | 0행 |
| 학습 / runtime / checkpoint 변경 | 없음 |

Top-5는 동일 계보 동결 HWR의 label-safety 검사이며 인간 다양성·신규 작가 일반화 근거가 아니다. `writer_disjoint_acceptance=false`, `raw_observed_allograph_evidence=false`를 유지한다.

## 평가 스모크 검증

- external 4,096행 + project evaluation 합성 512행을 구조 검사에만 로드했다.
- `training_admission=false`
- `augmentation_admission=false`
- `adoption_admission=false`
- 기본 commercial 산출물과 별도 폴더에 저장했다.

## 산출물

- 구현: `scripts/hierarchical_writer_model_v5.py` (`aiflow-hierarchical-writer-model/v6`)
- 기본 commercial 경계 결과: `artifacts/hierarchical_writer_model_v6_20260823_r1`
- 명시적 평가 스모크: `artifacts/hierarchical_writer_model_v6_20260823_eval_smoke_r1`
