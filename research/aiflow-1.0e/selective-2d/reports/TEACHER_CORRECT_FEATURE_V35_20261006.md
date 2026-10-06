# V35: 교사 정답 행만의 feature 보존 — paired 학습 기각

기준일: 2026-10-06. V33 복원과 [V34 TRAIN gradient 분해](CLOUD_REPLAY_AND_GRADIENT_BALANCE_V34_20261006.md) 이후 실제 학습 루프를 재개했다. **새 조건은 소유 수식 완전일치 64/149 → 62/149로 회귀해 채택하지 않았다. Canonical을 유지한다.**

## 가설과 고정 비교

V34에서 canonical 교사가 틀린 TRAIN 행의 feature 보존 gradient가 CE와 더 음의 방향을 보였다. 이 기여만 제거하면 정답 보존이 나아지는지 비교했다. 교사 정답/오답은 상속된 TRAIN 라벨에 대한 판정이며, 생성 증강 라벨은 사람이 검증한 라벨로 간주하지 않는다.

| 조건 | Feature 보존 |
| --- | --- |
| `v33_control` | 64행 전체의 per-row embedding MSE 평균 |
| `teacher_correct_feature` | canonical 교사가 맞힌 행의 per-row MSE만 포함하되 **분모는 64 유지** |

두 조건 모두 동일 Linux CPU에서 cold canonical, seed `2026100525`, V25 고정 데이터·스케줄로 **각 2,400 step**을 완료했다. 배치 64개 중 원본 16개·증강 48개이며, 각 153,600회 노출 중 증강은 115,200회다. 동일한 68,726개 고유 증강 view를 사용했다.

Hard CE는 항상 전체 64행, 계수 1이다. Feature 계수 `0.7412244467384174`, near-gap 계수 `0.6436691228293294`, AdamW lr `1e-5`, weight decay `1e-4`, clip norm 1을 유지했다. Dropout을 끈 eval mode에서 57개 텐서를 모두 학습했다. 교사 gradient, 소유 optimizer 입력, CROHME 입력은 0이다. 새 계수 탐색·중간 평가·최적 epoch 선택은 하지 않았다.

새 조건은 전체 153,600회 중 **29,626회**에서 feature 기여만 제외했다. CE와 원래 teacher-correct near-gap 손실은 그대로다. 비교는 같은 구현·플랫폼의 두 arm 사이이며, 과거 Windows V33 전체 학습의 비트 재생을 뜻하지 않는다.

근거: [봉인 계획](../artifacts/hwr_teacher_correct_feature_20261006_v35_cloud/frozen_plan.json), [실제 trainer](../scripts/run_hwr_teacher_correct_feature_v35.py). 계획은 V33 메타데이터를 상속한다. V35 행동은 `feature_rule`, `arms`, 새 코드 SHA가 정의하며, 상속된 `objective`·`trainer_reuse`·`checkpoint_container` 설명의 범위는 [별도 해석 기록](../artifacts/hwr_teacher_correct_feature_20261006_v35_cloud/plan_interpretation.json)에 명시했다. 봉인된 계획은 수정하지 않았다.

## 최종 결과: 같은 CPU·같은 입력

| 모델 | 내부 문자 Top-1 / 5,936 | 소유 문자 Top-1 / 579 | 소유 수식 Top-1 exact / 149 | 소유 수식 Top-5 완전 포함 / 149 |
| --- | ---: | ---: | ---: | ---: |
| Canonical | 4,765 | 470 | 76 | 137 |
| V33 목적함수 대조군 | 4,863 | 443 | 64 | 125 |
| 교사 정답 feature 조건 | 4,864 | 440 | 62 | 123 |

새 조건의 내부 증가 1건은 math-symbol family였다. 숫자·Latin-letter 정답 수는 같았다. 소유 legacy 96식은 exact 41 → 39, Codex 주석 53식은 23 → 23이었다. 전체 수식 exact의 paired 결과는 **개선 0건·회귀 2건**이다.

| 회귀 수식 | 대조군이 맞힌 정답 | 새 조건의 변경 |
| --- | --- | --- |
| `aiflow_0062` | `f ( x ) + 1` | `x` → `\chi` |
| `aiflow_0008` | `b = 4` | `b` → `\flat` |

이 외 이미 틀렸던 두 수식의 후보도 바뀌었다. 전체 변경 기록은 [연구 판정](../artifacts/hwr_teacher_correct_feature_20261006_v35_cloud/research_verdict.json), cohort·family 전체 집계는 [비교 결과](../artifacts/hwr_teacher_correct_feature_20261006_v35_cloud/comparison_result.json)에 보존한다. Canonical도 검증 과정에서 같은 CPU의 전체 입력으로 다시 평가했다.

## TRAIN 후속 진단

두 학습 완료 후 기존 봉인 TRAIN 8배치, 512회 노출에서 실제 목적함수 gradient를 다시 분해했다. 각 손실 gradient의 합과 전체 손실 직접 미분의 일치, 유한 값, checkpoint 불변을 확인했다. 평가 데이터·소유 라벨·optimizer·계수 조정은 사용하지 않았다.

| TRAIN 진단 | 대조군 | 새 조건 |
| --- | ---: | ---: |
| 교사 정답·학생 오답 노출 | 6 | 6 |
| 위 오답 중 원본 / 증강 | 1 / 5 | 1 / 5 |
| 근접 후보 손실 평균 | 0.019088 | 0.019605 |
| CE ↔ 가중 feature cosine 평균 | −0.083060 | −0.033216 |
| CE ↔ 전체 gradient cosine 평균 | 0.910980 | 0.912053 |

Feature gradient의 음의 정렬은 줄었지만 잔여 정답 보존 오류는 그대로였고 소유 진단은 회귀했다. **집계 gradient cosine 개선만으로 인식 개선을 판정할 수 없다는 반례**를 기록했다. 이미 학습한 8배치의 국소 방향이며, 실제 AdamW 과거 update의 인과 추적이나 새 필기 일반화 측정은 아니다. [진단 결과](../artifacts/hwr_teacher_correct_feature_20261006_v35_cloud/train_gradient_diagnostic.json)

## 검증과 보존

[독립 검증](../artifacts/hwr_teacher_correct_feature_20261006_v35_cloud/independent_verification.json)을 통과했다.

- 총 4,800개 step 기록의 손실 산술·입력 수·feature 행 수·교사 gradient 부재를 검사했다. 최대 기록 산술 차이는 `5.96e-8`이다.
- 두 모델 각각 네 encoder 층이 활동했고, 57개 텐서의 gradient가 유한하며 57개 텐서 모두 canonical에서 변경됐다.
- 두 모델의 개발 5,936행과 소유 579행 로짓이 **같은 CPU에서 재로딩 후 비트 일치**했다. 지표와 소유 입력·라벨 provenance를 재검사했다.
- Canonical SHA `04f8608aebcf6c02d45ad6f5735229b9eaa2c4b4e1be0db4793d02273ef2d00e`는 그대로다. 원본 아카이브와 과거 인증서도 보존했다.
- 최종 모델·전체 JSONL·fold/owned 로짓·per-formula 기록·계획·인증서·진단을 함께 보존했다. `resume_state.pt`는 마지막 2,304-step 주기 상태이며 최종 optimizer 상태나 자동 resume 기능으로 부르지 않는다.

## 판정과 다음 연구 제약

교사 오답 행의 feature 보존을 제거하는 조건은 기각한다. 모든 행을 보존하는 V33 비교 기준과 기존 canonical을 유지한다. 다음 목적함수 변경 전에는 TRAIN의 잔여 교사 정답·학생 오답 6건에서 개별 경쟁 후보 margin과 실제 update를 분해해, 집계 방향이 가리는 문제를 확인해야 한다.

149식은 이미 반복 관찰한 **oracle-group HWR 진단**이다. Top-5 완전 포함은 후보 상한이며 최종 decoder 정확도가 아니다. 새 writer·device acceptance, 원본 stroke grouping·관계·decoder 통합 경로와 공식 expression rate는 검증하지 않았다. 새 조건의 제품 승격·배포는 하지 않았다.
