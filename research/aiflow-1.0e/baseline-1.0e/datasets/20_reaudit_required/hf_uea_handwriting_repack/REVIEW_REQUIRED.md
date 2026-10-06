# 재감사 결과: `THUgewu/Handwriting`

- HF 저장소: https://huggingface.co/datasets/THUgewu/Handwriting
- 확인 revision: `f37e9035f9a5f509616efe9df4a1f3e1ff3dbe7f`
- 표시 라이선스: CC BY 4.0
- 규모: train 150 / test 850, 영문 26종
- 현재 판정: **AIFlow 온라인 잉크 HWR에서는 제외**

## 제외 이유

- 2D 펜 좌표가 아니라 글쓰기 동작의 3축 가속도 시계열이다.
- 길이는 152 frame으로 고정되어 있고 시간은 wall-clock timestamp가 아닌 frame index다.
- AIFlow 입력 계약인 `x, y, delta_t, stroke_start, observed`로 변환하면 x/y와 stroke 경계를 추정 생성해야 하므로 원본 신호 의미를 훼손한다.
- 26개 영문자만 있어 현재 372개 수학 기호 후보 회수 문제를 직접 보강하지 못한다.

라이선스 문제로 제외한 것이 아니라 modality 불일치로 제외했다. 별도 wearable-airwriting 연구가 생기면 독립 프로젝트에서 검토한다.
