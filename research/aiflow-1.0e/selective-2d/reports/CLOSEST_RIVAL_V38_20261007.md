# V38: closest-rival gap — 환경 복구 후 paired 비교

기록일: **2026-10-07(한국 시간)**. **수식 exact 64/149 → 63/149, Top-5 완전 포함 125/149 → 123/149, 개선 0식·회귀 1식.** 수식 exact 개선과 cohort별 exact·Top-5 비회귀 조건을 함께 충족하지 못해 후보를 채택하지 않는다. Canonical 유지, 제품 채택·배포 없음.

## V37 중단과 복구

V37 대조군 1,165단계·후보 1,173단계의 로그와 두 1,152-step 모델·AdamW·Torch RNG가 보존됐다. 별도 엄격 복구는 step 1,153의 전체 직렬화 기록 불일치로 두 arm 모두 종료했다. 로그 추가는 0이며 실패와 원래 입력을 함께 저장했다. 독립 TRAIN 진단에서 다른 필드는 63개·62개, 최대 절대 차이는 1.668930e-6·5.245209e-6이었다. 수치 차이의 내부 원인은 격리하지 않았다.

근거: [중단 상태](../artifacts/hwr_closest_rival_20261006_v37/run_status.json), [차이의 실제 값·원래 값](../artifacts/hwr_closest_rival_20261006_v37/recovery_drift_20261007.json), [복구 코드](../scripts/resume_hwr_closest_rival_v37.py). 이전 기록·assertion·인증서를 변경하지 않았다. 기존 V37은 최종 모델이나 평가가 없는 중단 실행이다.

V38은 두 조건을 **같은 현재 CPU의 cold canonical부터** 다시 학습했다. 완료된 새 비교의 optimizer update는 4,800개다. 이전 중단 실행의 기록된 2,338개 update와 실패한 복구의 2개 update는 추가 계산이며 새 최종 모델의 학습 궤적에 합치지 않는다.

## 가설과 고정 조건

[V36](ADAM_MARGIN_REPLAY_V36_20261006.md)은 마지막 96개 AdamW update의 기록·최종 가중치를 비트 재현했다. 같은 fitted TRAIN 8배치에서 잔여 오류 6개 중 gap 국소 방향이 margin을 해치는 경우는 기존 2개·closest-rival 1개였다. 이 국소 비교는 계수 탐색이나 정확도 검증이 아니다.

두 조건 모두 64행 CE·feature MSE를 유지한다. 대조군은 교사 정답 행의 teacher/student Top-5 union에서 정답을 제외한 pair 평균 gap 손실을 쓴다. 후보는 **학생 최대 비정답 logit 클래스 하나**를 선택해 교사 대비 감소한 gap 제곱을 교사 정답 행 수로 평균한다. 동률은 작은 class index다. 선택 pair와 정규화 분모가 함께 바뀌므로 동일 계수가 동일 gradient 크기를 뜻하지 않는다.

V25 데이터·스케줄, seed 2026100525, arm당 2,400 step, 배치 64(원본 16·증강 48), 153,600회 노출과 68,726개 고유 증강 view를 유지했다. Hard CE 1, feature 0.7412244467384174, gap 0.6436691228293294, AdamW lr 1e-5·weight decay 1e-4·clip 1이다. Dropout off, 57개 텐서·4개 encoder layer를 모두 학습했다. 교사 gradient·owned optimizer 행·CROHME 행은 0이다. 계수 재탐색·중간 평가·best epoch 선택은 없다.

[봉인 계획](../artifacts/hwr_closest_rival_20261007_v38/frozen_plan.json)은 CPU 모델·capability·Torch build도 포함한다. 수정하지 않은 V35 train/evaluate 함수를 재사용해 컨테이너는 legacy V35 schema이며, V38 계획·arm·새 인증서가 현재 실험을 식별한다. [실행·재현](../../cloud/V38.md)

## 같은 CPU의 최종 결과

| 모델 | 내부 문자 Top-1 / 5,936 | 소유 문자 Top-1 / 579 | 수식 exact / 149 | Top-5 완전 포함 / 149 |
| --- | ---: | ---: | ---: | ---: |
| Canonical 재평가 | 4,765 | 470 | 76 | 137 |
| 같은 CPU V33 목적함수 대조군 | 4,863 | 443 | 64 | 125 |
| Closest-rival 후보 | 4,849 | 445 | 63 | 123 |

| Cohort | 수식 exact 대조군 → 후보 | Top-5 완전 포함 | 문자 Top-1 |
| --- | ---: | ---: | ---: |
| legacy_96 | 41 → 40 | 85 → 83 | 305 → 306 |
| codex_reviewed_53 | 23 → 23 | 40 → 40 | 138 → 139 |

| 내부 family | Top-1 대조군 → 후보 | Top-5 대조군 → 후보 |
| --- | ---: | ---: |
| digits | 132 → 130 | 157 → 157 |
| latin_letters | 613 → 616 | 812 → 812 |
| math_symbols | 4118 → 4103 | 4910 → 4910 |

Paired exact 개선 ID: 없음. 회귀 ID: aiflow_0062. 전체 변경·cohort 지표는 [판정 JSON](../artifacts/hwr_closest_rival_20261007_v38/research_verdict.json), [원래 비교 JSON](../artifacts/hwr_closest_rival_20261007_v38/comparison_result.json)에 보존했다.

회귀한 `aiflow_0062`의 정답은 `f ( x ) + 1`이다. 후보는 `x`를 `\chi`로 예측했다. 소유 문자 정답 수의 +2가 수식 완전일치 개선으로 이어지지 않았다.

## TRAIN 후속 진단

완료 후 기존 fitted TRAIN 8배치·512회 노출에서 gradient 합을 직접 미분과 비교했다. 추가 optimizer step·평가 forward·소유 라벨 읽기는 0이다. 교사 정답·학생 오답 노출은 **6 → 2**이며 원본/증강은 1/5 → 0/2다.

| 같은 TRAIN 정의 | 대조군 | 후보 |
| --- | ---: | ---: |
| 공통 near-gap 평균 | 0.019088 | 0.024362 |
| CE ↔ feature cosine | -0.083059 | -0.093628 |
| CE ↔ gap cosine | 0.165176 | 0.146310 |
| CE ↔ 전체 gradient cosine | 0.910980 | 0.838046 |

조건별 실제 gap 평균은 0.019088, 0.022776이며 정의와 분모가 달라 그 비율을 정확도 향상으로 해석하지 않는다. 표의 공통 near-gap은 두 모델 모두 원래 정의로 계산했다. 국소 gradient나 fitted TRAIN의 변화는 새로운 필기 일반화나 AdamW 인과 효과의 증거가 아니다. [전체 진단](../artifacts/hwr_closest_rival_20261007_v38/train_gradient_diagnostic.json)

## 검증과 해석 범위

[독립 검증](../artifacts/hwr_closest_rival_20261007_v38/independent_verification.json)은 총 4,800개 기록의 손실 산술·행 수·57개 유한 gradient·4개 층의 활동·전체 파라미터 갱신, 같은 현재 CPU의 최종 6,515입력 로짓 비트 재현, owned 라벨 provenance와 지표 집계를 통과했다. 모든 과거 중간 forward의 재생은 아니다.

과거 V35 대조군과 새 대조군의 2,400개 직렬화 기록 모두 달랐다. 최종 state 최대 절대 차이는 0.000237166882였으며, 최종 전체 집계 지표는 같았다. 새 같은 CPU의 두 arm이 가설 비교 기준이며 환경 간 가중치·기록의 비트 일치를 입증한 것은 아니다.

Canonical SHA `04f8608aebcf6c02d45ad6f5735229b9eaa2c4b4e1be0db4793d02273ef2d00e`는 불변이다. 원본 아카이브·과거 인증서를 보존한다. `resume_state.pt`는 마지막 2,304-step 주기 상태이며 최종 optimizer 상태가 아니다. 기존 클라우드 매니페스트의 README 해시는 040f7effa63c97507fa781d42cf77c365b224ea2 스냅샷에 적용된다.

149식은 반복 관찰한 정답 grouping 제공 HWR 진단이다. Top-5 완전 포함은 후보 상한이며 최종 decoder 정확도가 아니다. 96식 cohort와 과거 95식 calibration 분모를 섞지 않는다. 생성 증강 라벨은 사람 검증이 없고, untouched writer/device acceptance·raw-stroke grouping·관계·decoder 통합·공식 expression rate는 검증하지 않았다. Canonical을 유지한다.

다음 목적함수 비교에서는 기존 다중 근접 후보 보존 기준을 유지한다. 단일 후보의 fitted TRAIN 오류 감소만으로 보존 범위를 줄이거나 후보를 승격하지 않는다. 추가 계수 탐색의 근거로 이번 소유 평가 결과를 사용하지 않는다.
