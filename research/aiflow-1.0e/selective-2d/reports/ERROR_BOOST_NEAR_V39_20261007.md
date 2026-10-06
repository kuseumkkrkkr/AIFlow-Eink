# V39: 전체 near 후보를 유지하는 오류 행 boost

기록일: **2026-10-07(한국 시간)**. **수식 exact 64/149 → 64/149, Top-5 완전 포함 125/149 → 125/149, 개선 0식·회귀 0식.** Cohort별 exact·Top-5 비회귀는 충족했지만 수식 exact 증가가 없어 후보를 채택하지 않는다. Canonical 유지, 제품 채택·배포 없음.

## 가설과 TRAIN 근거

[V38](CLOSEST_RIVAL_V38_20261007.md)의 단일 경쟁 후보 손실은 fitted TRAIN 오류를 줄였지만 소유 수식 exact는 회귀했다. V39는 **기존 다중 near 후보와 pair 수 분모를 그대로 유지**하면서 교사가 맞고 학생이 틀린 행의 전체 near-pair 결손 제곱만 2배로 보정한다. Teacher/student stable Top-5 union에서 정답을 제외하고 teacher-correct mask를 적용한다. Student argmax는 detached다. 다른 행의 gap과 64행 CE·feature는 그대로다.

Boost 2는 TRAIN 진단 전에 고정했고 계수나 boost 검색을 하지 않았다. 기존 fitted TRAIN 8배치·512회 노출에서 잔여 오류 6개 중 gap의 국소 음의 SGD 방향이 정답 margin을 해치는 경우는 **2 → 1**이었다. 모델 forward는 TRAIN만 사용했고 optimizer·owned/development forward·CROHME 행은 0이었다. 이 국소 방향은 실제 AdamW 이동이나 새 필기 정확도의 증거가 아니다. [사전 진단](../artifacts/hwr_error_boost_diagnostic_20261007_v39/train_comparison.json), [손실 코드](../scripts/hwr_error_boost_near_v39.py)

## 고정 조건과 대조군 재사용

V25 데이터·스케줄·seed 2026100525, 2,400-step 예산, 배치 64(원본 16·증강 48), 153,600회 노출과 68,726개 고유 증강 view를 유지했다. Hard CE 1, feature 0.7412244467384174, gap 0.6436691228293294, AdamW lr 1e-5·weight decay 1e-4·clip 1이다. Dropout off, 57개 텐서·4개 encoder layer를 모두 학습했다. 중간 평가·best epoch 선택·소유 라벨 학습은 없다.

V38에서 검증된 같은 CPU 대조군의 모델·2,400개 로그·주기 optimizer 상태·시작 및 완료 기록 5개를 바이트 그대로 복사했다. 원래 V38 plan SHA와 PID를 보존했다. 출처 계획·인증서·5파일 해시를 V39 계획에 봉인하고 CPU 모델·capability·Torch build·Python·NumPy·소스·계수를 검사했다. **새 optimizer update는 후보의 2,400개뿐**이다. 독립 검증 대상 4,800개 학습 기록에는 재사용 대조군 2,400개가 포함된다.

후보만 cold canonical부터 새로 학습했다. 완료 후 두 모델 모두 최종 추론을 실행했고, 대조군 로짓·지표가 V38 출처와 비트 일치했다. Canonical 지표는 V38의 같은 CPU 인증서에서 재사용했고 V39에서 canonical forward를 새로 실행하지 않았다. [봉인 계획](../artifacts/hwr_error_boost_near_20261007_v39/frozen_plan.json), [재현 명령](../../cloud/V39.md)

수정하지 않은 V35 train/evaluate 함수를 사용해 모델·비교 JSON은 legacy V35 컨테이너다. V39 계획·arm·독립 인증서가 현재 실험을 식별한다.

## 최종 결과

| 모델 | 내부 문자 Top-1 / 5,936 | 소유 문자 Top-1 / 579 | 수식 exact / 149 | Top-5 완전 포함 / 149 |
| --- | ---: | ---: | ---: | ---: |
| Canonical（V38 인증서 지표 재사용） | 4,765 | 470 | 76 | 137 |
| V38 같은 CPU 대조군 재사용 | 4,863 | 443 | 64 | 125 |
| V39 오류 행 boost 후보 | 4,858 | 444 | 64 | 125 |

| Cohort | 수식 exact 대조군 → 후보 | Top-5 완전 포함 | 문자 Top-1 |
| --- | ---: | ---: | ---: |
| legacy_96 | 41 → 41 | 85 → 85 | 305 → 306 |
| codex_reviewed_53 | 23 → 23 | 40 → 40 | 138 → 138 |

| 내부 family | Top-1 대조군 → 후보 | Top-5 대조군 → 후보 |
| --- | ---: | ---: |
| digits | 132 → 132 | 157 → 157 |
| latin_letters | 613 → 613 | 812 → 812 |
| math_symbols | 4118 → 4113 | 4910 → 4910 |

Paired exact 개선 ID: 없음. 회귀 ID: 없음.

수식 exact 상태가 바뀐 ID는 없다.

[판정 및 전체 변경 JSON](../artifacts/hwr_error_boost_near_20261007_v39/research_verdict.json), [원래 비교 JSON](../artifacts/hwr_error_boost_near_20261007_v39/comparison_result.json)

## 실제 boost 노출과 TRAIN 후속 진단

2,400개 step의 focus 기록에서 boost 적용은 총 **1,357회**, 원본 **382회**, 증강 **975회**다. 고유 샘플 수가 아닌 반복 학습 노출 수다. 독립 검증은 행 분할·teacher count·full near-pair 범위·고정 2배 경계와 각 실제 gap이 optimizer 기록의 gap과 정확히 같음을 확인했다. [Focus 기록](../artifacts/hwr_error_boost_near_20261007_v39/focus_microscope.jsonl)

완료 후 같은 fitted TRAIN 8배치·512회 노출에서 gradient 합을 직접 미분과 비교했다. 교사 정답·학생 오답 노출은 **6 → 6**이며 원본/증강은 1/5 → 1/5다. 추가 optimizer step·평가 forward·소유 라벨 읽기는 0이다.

| 같은 TRAIN 정의 | 대조군 | 후보 |
| --- | ---: | ---: |
| 공통 near-gap 평균 | 0.019088 | 0.018448 |
| CE ↔ feature cosine | -0.083059 | -0.080755 |
| CE ↔ gap cosine | 0.165176 | 0.177789 |
| CE ↔ 전체 gradient cosine | 0.910980 | 0.905403 |

조건별 실제 gap 평균은 0.019088, 0.021132이다. 오류 행 가중치가 다르므로 그 비율은 공통 손실이나 정확도 비교가 아니다. 공통 near-gap 표는 두 모델 모두 원래 정의로 계산했다. [전체 TRAIN 진단](../artifacts/hwr_error_boost_near_20261007_v39/train_gradient_diagnostic.json)

## 검증과 해석 범위

[독립 검증](../artifacts/hwr_error_boost_near_20261007_v39/independent_verification.json)은 새 후보와 재사용 대조군의 총 4,800개 기록의 손실 산술·행 수·57개 유한 gradient·4개 층 활동·전체 파라미터 갱신을 확인했다. 완료 후 각 모델의 최종 6,515입력 로짓을 재로딩해 같은 현재 CPU에서 비트 일치시켰고, 소유 149식의 라벨 provenance와 지표를 재집계했다. 대조군은 V38 출처의 가중치·2,400개 로그·최종 로짓·지표도 그대로 일치했다. 모든 중간 forward의 재생이나 Windows/Linux 비트 일치를 뜻하지 않는다.

Canonical SHA `04f8608aebcf6c02d45ad6f5735229b9eaa2c4b4e1be0db4793d02273ef2d00e`는 불변이다. `resume_state.pt`는 마지막 2,304-step 주기 상태이며 최종 optimizer 상태가 아니다. 과거 인증서·봉인 소스·아카이브는 보존한다. V35 클라우드 매니페스트의 편집 문서 해시는 `040f7effa63c97507fa781d42cf77c365b224ea2`, V38 매니페스트의 해시는 `47bafa4cc3b9fe1d8bce4fb0bdbf2c875496050c` 스냅샷에 적용된다. 후속 문서 변경으로 과거 매니페스트를 다시 쓰지 않는다.

149식은 반복 관찰한 정답 grouping 제공 HWR 진단이다. Top-5 완전 포함은 후보 상한이며 최종 decoder 정확도가 아니다. 96식 cohort와 과거 95식 calibration 분모를 섞지 않는다. 생성 증강 라벨은 사람 검증이 없고 owned/CROHME optimizer 행은 0이다. 새 작성자·실제 E-ink acceptance·raw-stroke grouping·관계·decoder 통합·공식 expression rate를 재검증한 것은 아니다.
