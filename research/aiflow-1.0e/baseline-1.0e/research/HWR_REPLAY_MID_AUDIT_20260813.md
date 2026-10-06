# HWR replay 중검수

- 기준 체크포인트: 단일 372종, 8 epoch, `math-observed-one`
- 체크포인트 SHA-256: `1d7313e936a2a5127dd3616ba35cd4f79b62402aa168ff6546f807da9ba63107`
- 학습 실행: 없음
- 재생 계약: slider 1단계마다 원시 점 1개를 표시하고, 마지막 점에서 예측 결과를 채점
- 화면 확인: ownership 83점, CROHME 3,445점, canonical 145점 표본이 각각 `1 / N`과 `N / N`으로 정상 재생됨
- 로컬 상세 보고서: `artifacts/replay_mid_audit_20260813/mid_audit_report.json`

## 1. CROHME valid

- 원본 986식 중 완전 동일 중복 1식을 제거한 고정 985식
- 9,535개 지원 가능 정답 문자 그룹: Top-1 60.19%, Top-5 84.90%
- 출력부 밖 문자 그룹 480개, 1타점 결측 그룹 1개, 불완전 partition 수식 1개
- 정답 그룹을 미리 주었을 때 모든 문자가 Top-1인 수식: 96/985 = 9.75%
- 출력 문자로 완전히 덮이는 751식만 보면 96/751 = 12.78%
- 이는 실제 수식 exact가 아니다. grouping·2차원 관계·LaTeX decoder가 없으므로 end-to-end 수식 정확도는 미산출이다.

## 2. 직접 수집

- ownership 47식/211문자 전부: 문자 Top-1 50.24%, Top-5 80.57%
- 정답 ownership을 미리 주었을 때 모든 문자가 Top-1인 수식: 5/47 = 10.64%
- 20식은 외부 학습 표본이 0개인 `=`를 포함한다.
- canonical 고정 10식 중 ownership이 있는 것은 6식이며, 그 부분집합의 모든 문자 Top-1은 1/6이다.
- 나머지 4식은 문자 경계 정답이 없고, 10식 모두 현행 수식 decoder가 없어 canonical formula exact는 미산출이다.

## 3. 현재 데이터 고정 10%

- 기존 프로토콜: 18,723건
- 현 단일 출력부 범위: 18,344건, 371종, 클래스당 최소 5건
- 범위 내 Top-1 79.16%, Top-5 97.66%
- 기존 18,723건 전체에서 출력부 밖 379건을 실패로 세면 Top-1 77.56%, Top-5 95.68%
- 출력부 밖: UJI 367건, ISGL 12건. 372번째 출력 `=`은 외부 데이터 표본이 없다.

## 중검수 판정

- 문자 분류 지표와 점 단위 replay는 유효하다.
- 세 평가군의 end-to-end 수식 정확도는 아직 확정할 수 없다.
- 다음 평가 구현 순서는 `stroke grouping -> box-local Top-k -> spatial relation -> formula decoder -> formula exact`이다.
- 10% 고정 분할에는 동일 128x5 tensor가 train/eval 양쪽에 나타나는 그룹이 5개(동일 label 4, 교차 label 1), eval 행으로는 `\cdot` 8건 있다. 이 8건을 빼도 Top-1 79.16%, Top-5 97.66%로 반올림값은 같지만, 최종 평가 전 group-hash 분할 재고정이 필요하다.
