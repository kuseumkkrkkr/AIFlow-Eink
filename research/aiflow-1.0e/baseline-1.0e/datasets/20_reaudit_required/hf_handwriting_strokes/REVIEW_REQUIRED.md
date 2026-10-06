# 재감사 필요: `newbienewbie/handwriting_strokes`

- HF 저장소: https://huggingface.co/datasets/newbienewbie/handwriting_strokes
- 확인 revision: `8f9dbef771a25b7bb11bf0385fed9a49b1f93235`
- 규모: train 920 / validation 100 / test 100
- 현재 판정: **학습 반입 금지**

## 형식 확인

- `xCoordinates`, `yCoordinates`, `timeStamps` 길이는 실제 행에서 서로 같아 온라인 궤적처럼 보인다.
- 첫 행은 62점, timestamp `0.0~757.94 ms`였다.
- 그러나 `sampleTag`는 `0`, `1`, `2` 같은 순번이고 `annotation`은 모든 행에서 `01236789COVULWZ()SNMJ`처럼 클래스 목록으로 보였다. 공개 metadata만으로 각 궤적의 정답 문자를 복원할 수 없다.

## 재감사 사유

1. dataset card에 라이선스가 없다.
2. 원 데이터 출처, 수집 동의, writer/session 분리 정보가 없다.
3. `sampleTag`와 실제 문자 label의 대응표가 없다.
4. stroke 경계 표현과 동일 writer 중복 여부를 확인할 수 없다.

원 저작권자와 클래스 매핑, 상업 이용 허가, writer/session provenance를 문서로 확보하기 전에는 다운로드 확대·정규화·학습을 금지한다.
