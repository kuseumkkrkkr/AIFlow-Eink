# Linux CPU 복원 검증과 V34 TRAIN gradient 분해

기준일: 2026-10-06. GitHub `34a8cce`에 보존된 V33 중단 지점에서 후속 연구를 재개했다. 원본 연구 스크립트·체크포인트·인증서는 수정하지 않았다.

## 복원과 V33 재현 범위

- 분할 압축의 25개 고유 blob과 39개 원본 파일을 검증·복원했다. 이후 원본 스냅샷 매니페스트의 **8,905개 파일, 15,881,701,669바이트**를 모두 SHA-256으로 재검증했다. [원본 보존 검증](../artifacts/hwr_cloud_replay_20261006_v33/snapshot_verification.json)
- Python 3.12.14, Torch 2.5.1+cpu, NumPy 2.1.3, SciPy 1.15.3 환경이다. [설치 스크립트](../../cloud/install-linux-cpu.sh)를 실제 실행했으며 `pip check`, 문서 계약의 10개 합성 유효/무효 검사, 학습 캐시·스케줄 해시, 근접 후보 손실 기능 검사를 통과했다.
- [경로 설정 helper](../scripts/cloud_hwr_snapshot.py)는 canonical 체크포인트와 affine-distill 캐시 위치 두 곳만 프로세스 내부에서 치환한다. 기존 소스·데이터 해시 검사는 유지한다.
- 원본 V33 검증기를 복사본에서 실행했지만 TRAIN gradient norm의 절대 오차 `<1e-6` 조건에서 실패했다. 원본 검증기나 인증서를 완화·교체하지 않았다. [엄격 재실행 상태](../artifacts/hwr_cloud_replay_20261006_v33/strict_replay_status.json)
- 별도 [portable 검증기](../scripts/verify_hwr_cloud_v33.py)는 gradient norm에 `atol=1e-5, rtol=1e-5`, 로짓에 절대 오차 `1e-4`를 명시한다. 실제 최대 gradient 절대 차이는 `2.575e-5`, 로짓 차이는 `1.335e-5`였다. 기존 봉인 계수 `0.6436691228293294`는 변경하지 않았다.
- 개발 5,936행과 소유 문자 579행, **총 6,515행 모두 Top-5 순서와 최종 지표가 정확히 일치**했다. V33 내부 문자 4,863/5,936, 소유 문자 443/579, 수식 Top-1 exact 64/149, Top-5 완전 포함 125/149를 재확인했다. 운영체제 간 로짓 비트 일치는 실패로 명시한다. [별도 검증 결과](../artifacts/hwr_cloud_replay_20261006_v33/cloud_verification.json)

과거 2,400개 손실 기록의 산술, 유한 gradient, 네 encoder 층 활동, 57개 텐서 변경도 확인했다. 과거 모든 중간 forward를 재학습해 재생한 검증은 아니다. Android·실제 기기·원본 stroke grouping 경로는 실행하지 않았다.

## V34: 이미 학습한 TRAIN 8배치에서 목적함수 방향 비교

[진단 코드](../scripts/audit_hwr_gradient_balance_v34.py)는 V33의 봉인된 TRAIN 8배치, 총 512회 노출에서 V29와 V33을 비교한다. 소유 라벨·개발 평가·CROHME 입력·optimizer step·계수 재탐색은 사용하지 않았다.

CE, 가중 feature MSE, 각 모델의 실제 gap 손실을 각각 미분했다. 세 gradient의 합이 전체 손실의 직접 미분과 일치하는지 확인했다. Feature MSE는 classifier head의 두 텐서와 연결되지 않으므로 이 두 gradient만 명시적으로 0으로 처리하며, 나머지 연결과 유한 값을 검사했다.

Feature gradient를 canonical 교사가 맞힌 행과 틀린 행으로 추가 분해했다. 양쪽 모두 **원래 배치 분모 64**를 유지하며, 두 부분의 합이 원래 feature gradient와 일치하는지 검사했다. 아래 cosine은 배치별 전체 파라미터 벡터 cosine의 평균이다.

| TRAIN 진단 | V29 | V33 |
| --- | ---: | ---: |
| 교사 정답·학생 오답 노출 | 9 | 6 |
| 위 오답 중 원본 / 증강 | 2 / 7 | 1 / 5 |
| 근접 후보 gap 손실 평균 | 0.099419 | 0.019088 |
| CE ↔ 전체 feature cosine | −0.075243 | −0.083060 |
| CE ↔ 교사 정답 feature cosine | −0.018210 | −0.038578 |
| CE ↔ 교사 오답 feature cosine | −0.192526 | −0.173517 |
| CE ↔ 실제 gap cosine | 0.194893 | 0.165175 |
| CE ↔ 합산 gradient cosine | 0.969955 | 0.910980 |
| 합산 gradient가 CE를 반대하는 배치 | 0/8 | 0/8 |

근거: [전체 배치 기록과 분해 검증](../artifacts/hwr_gradient_balance_20261006_v34/gradient_balance.json).

교사가 틀린 행의 표현 보존 gradient가 CE와 더 음의 방향을 보였다. 하지만 전체 gradient는 모든 배치에서 CE와 양의 방향이며, 이는 오답 원인이나 새 필기 일반화의 인과 증거가 아니다. 이미 학습·계수 보정에 사용한 512회 노출에서 얻은 국소 진단이다.

## 다음 통제 실험

V35는 교사가 틀린 TRAIN 행의 feature 보존 기여만 제거한다. 교사가 맞힌 행은 계속 보존하고, 정규화 분모 64와 기존 feature/gap 계수는 유지한다. Linux CPU에서 기존 V33 목적함수 조건과 새 조건을 같은 데이터·seed·cold canonical·2,400-step 예산으로 각각 학습한다. 성능 비교는 두 학습이 모두 완료된 후 수행하며 중간 평가나 최적 epoch 선택은 하지 않는다.

현재 소유 149식은 반복 사용한 oracle-group HWR 진단이며 새 acceptance 데이터가 아니다. TRAIN 증강 라벨의 사람 검증, 새 writer/device, raw-stroke grouping·관계·decoder 통합 검증은 남아 있다. Canonical은 유지하며 제품 채택·배포는 하지 않는다.
