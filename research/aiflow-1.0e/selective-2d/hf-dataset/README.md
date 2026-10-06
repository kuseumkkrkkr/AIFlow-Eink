---
language:
- ko
- en
license: other
task_categories:
- image-classification
tags:
- online-handwriting
- mathematical-expression-recognition
- trajectory
- stroke-sequence
pretty_name: AIFlow Math Ink 0.9 Public Stroke Dataset
size_categories:
- n<1K
---

# AIFlow Math Ink 0.9 Public Stroke Dataset

AIFlow Math Ink 0.9의 온라인 수식 필기 데이터다. 데이터 관리자는 2026-08-08 참여자들의 공개 배포 동의가 완료됐음을 확인했다. 원래 수집 동의 범위인 AIFlow 필기 인식 모델 학습·검증과 이번 공개 배포 승인을 함께 기록한다.

## 구성

| 파일 | 수식 | 용도 |
|---|---:|---|
| `data/formulas_valid.jsonl` | 110 | 검수 완료 학습·분석 후보 |
| `data/formulas_pending.jsonl` | 4 | 시각 검수 전; 기본 학습 제외 |
| `data/formulas_reject.jsonl` | 5 | 검수 탈락; 기본 학습 제외 |
| `data/ownership_train.jsonl` | 47 | 수동 검수된 stroke-to-symbol ownership |

전체 119개 수식, 8명의 dataset-local writer group, 12개 session group이다. 70건은 공개 웹 수집, 49건은 소유자 휴대폰 replay다. 문자별 수량과 파일 SHA-256은 `dataset_info.json`에 있다.

## 레코드

각 formula row는 다음을 보존한다.

- 원본 stroke 순서와 point 순서
- `x`, `y`, formula-relative `t_ms`
- pressure·tilt·contact geometry가 있는 경우 해당 센서값
- canvas 크기, 목표 token/관계, 검수 상태
- 재현 가능한 dataset-local `writer_id`, `session_group`

`ownership_train.jsonl`은 `sample_id`, 문자 label, 원본 stroke index group을 제공한다.

## 개인정보 처리

공개본에서 contributor/session/prompt 원본 ID, 기록 시각, IP, 원본 경로, PNG data URL, epoch 기반 시간 원점과 raw pressure를 제거했다. writer/session 값은 비공개 매핑과 분리된 dataset-local 별칭이다. 각 수식의 첫 point를 `t_ms=0`으로 이동했다.

## 검증

```bash
python scripts/validate_public_dataset09.py .
```

검증기는 119건·quality split·47 ownership rows·SHA-256·금지 필드 부재·stroke/point count·시간 원점·ownership 정합성을 확인한다.

## 라이선스와 제한

`DATA_USE_NOTICE.md`를 따른다. 참여자 동의 및 데이터 관리자 확인에 따라 공개 배포되지만, 제3자 상업 재사용을 포괄적으로 허가하는 표준 라이선스는 선언하지 않는다. 신원 추론·재식별·필적 기반 개인 프로파일링에 사용하지 말아야 한다.

## 알려진 한계

- 119개 수식, 8 writer로 작다.
- class 분포가 불균형하다.
- pending/reject는 학습에 자동 포함하면 안 된다.
- 47개 수식만 문자 단위 ownership이 검수됐다.
- CROHME는 이 저장소에 포함되지 않으며 별도 비상업 연구 검증에만 사용한다.
