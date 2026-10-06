# V36: 잔여 TRAIN 오류의 margin과 마지막 96 AdamW update 재생

기준일: 2026-10-06. V35의 교사 정답 feature 조건은 수식 진단 회귀로 기각했다. 다음 목적함수 변경 전, 유지한 V33 목적함수 대조군의 잔여 TRAIN 오류 6건을 분석했다.

## 검증한 재생 범위

V35 대조군의 2,304-step `resume_state.pt`에서 모델·AdamW 상태·Torch RNG를 복원했다. 원래 고정 스케줄의 **2,305–2,400 step, 총 96개 update**를 메모리에서 다시 실행했다. 원본 데이터·학습 코드·체크포인트는 바꾸지 않았다.

- 96개 step의 CE·feature·gap·총 손실·전체 norm 및 **57개 개별 gradient norm**이 원래 기록과 비트 단위로 일치했다. 최대 차이는 0이다.
- 재생한 최종 `state_dict`의 모든 값이 기존 V35 대조군과 비트 단위로 일치했다.
- 추가 학습 예산은 0이며, 저장된 모델을 승격·교체하지 않았다. 소유 라벨·개발 평가·CROHME 입력은 읽지 않았다.
- 전체 2,400개 과거 update를 재생한 검증은 아니다. 마지막 96개와 정해진 TRAIN 진단의 범위를 구분한다.

근거: [재생 코드](../scripts/audit_hwr_adam_margin_v36.py), [결과·96개 실제 margin trajectory](../artifacts/hwr_adam_margin_20261006_v36/audit_result.json), [진단 입력 6개](../artifacts/hwr_adam_margin_20261006_v36/probe_inputs.npy).

## 국소 gradient 방향과 실제 update는 달랐다

기존 봉인 TRAIN 8배치에서 canonical 교사가 맞히고 학생이 틀린 6회 노출을 선택했다. 정답 logit에서 최종 학생 Top-1 경쟁 후보 logit을 뺀 margin을 추적한다. 아래 실제 변화는 같은 입력과 **고정한 경쟁 후보**에 대해 2,304-step 상태부터 최종 상태까지의 변화다.

| Probe | TRAIN GT → 최종 경쟁 후보 | 원본/증강 | Gap의 국소 negative-SGD margin 미분 | 실제 마지막 96 AdamW margin 변화 |
| --- | --- | --- | ---: | ---: |
| 0 | `\triangleleft` → `\lhd` | 원본 | −0.262640 | −0.072603 |
| 1 | vertical bar → `\mid` | 증강 | +1.832720 | +0.088682 |
| 2 | `\astrosun` → `\odot` | 증강 | +0.553153 | −0.039038 |
| 3 | `\perp` → `\bot` | 증강 | −3.068080 | +0.076031 |
| 4 | `e` → `\varrho` | 증강 | +80.402280 | +0.028877 |
| 5 | `\ell` → `l` | 증강 | +41.226331 | −0.276867 |

국소 미분은 `−∇margin · ∇loss`이며 양수는 그 배치의 negative-SGD 방향에서 margin 증가를 뜻한다. 가중 CE·feature·gap의 미분 합이 전체 손실 미분과 일치함을 검사했다. **전체 손실의 국소 방향은 6건 모두 양수였지만, 실제 마지막 96개 AdamW update의 순변화는 3건에서 음수**였다. 실제 trajectory는 다른 배치들의 gradient와 AdamW 상태를 사용하므로 이 차이만으로 momentum·preconditioning 등 특정 원인의 인과 효과를 판정하지 않는다.

현재 gap 손실 자체의 국소 방향은 2건에서 음수였다. 각 배치의 212–226개 근접 pair를 평균하는 손실이며 교사 정답 행은 50–53개다. 자기 정답과 현재 경쟁 후보 pair가 포함·활성화되어도, 공유 파라미터에 대한 전체 배치 보조 gradient는 해당 margin을 줄일 수 있었다. Loss 에너지 비율을 gradient나 원인 비율로 부르지 않는다.

## V37 가설: 학생의 가장 강한 경쟁 후보 하나 보존

다음 조건은 teacher-correct 행에서 학생의 **정답을 제외한 최대 logit 후보 하나**를 선택한다. 동점은 가장 작은 class index로 처리한다. 그 후보에 대한 canonical의 정답 gap이 줄어든 양의 제곱을 teacher-correct 행 수로 평균한다. Feature는 64행 모두 보존하며 V35의 실패한 feature mask는 사용하지 않는다.

같은 8개 fitted TRAIN 배치의 국소 비교에서 gap이 margin을 해치는 방향은 **기존 2/6 → 새 조건 1/6**이었다. [비교 코드](../scripts/compare_hwr_closest_rival_train_v37.py), [전체 방향·gradient norm 결과](../artifacts/hwr_adam_margin_20261006_v36/closest_rival_train_comparison.json).

이는 정확도 개선의 증거가 아니다. 선택 pair와 정규화 분모가 함께 바뀌므로 계수가 같아도 gradient 크기는 달라진다. 이 변경을 V37 목적함수 차이로 명시하고, alpha/beta·데이터·seed·cold canonical·학습 예산은 고정한다. 새로운 대조군과 closest-rival 조건을 같은 Linux CPU에서 각각 2,400 step 학습한 뒤에만 최종 평가한다. TRAIN 진단으로 계수를 재탐색하지 않았다.

## 해석 한계

6건은 이미 학습한 512회 노출에서 최종 오류를 사후 선택한 것이다. 그중 증강 5건의 라벨 불변성은 상속 가정이며 사람이 검증한 라벨로 취급하지 않는다. Margin 추론의 배치 크기·실행 경로 차이로 원래 64행 gradient 경로와 수 µ 단위 로짓 차이가 있으므로, 원본 경로 margin과 재생 probe margin을 결과에서 별도로 보존했다. 비트 재현 주장은 원래 96개 학습 기록과 최종 모델에 대한 것이다.

새 writer/device·소유 수식 acceptance·raw-stroke grouping·decoder 검증은 이 단계에 포함하지 않았다. Canonical을 유지한다.
