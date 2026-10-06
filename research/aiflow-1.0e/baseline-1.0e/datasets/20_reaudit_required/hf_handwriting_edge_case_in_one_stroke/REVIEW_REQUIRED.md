# 재감사 필요: `newbienewbie/handwriting_edge_case_in_one_stroke`

- HF 저장소: https://huggingface.co/datasets/newbienewbie/handwriting_edge_case_in_one_stroke
- 확인 revision: `c71311ea94b88af0d0e8c580c6bafefa13ae1640`
- 규모: train 858 / validation 22 / test 157
- 현재 판정: **학습 반입 금지**

## 형식 확인

- `sampleTag`가 `C`, `(`, `v`, `)`, `bs`, `7`처럼 문자 label로 보이고 x/y/t 길이도 일치한다.
- `annotation`은 `C(1)\\/7`처럼 원 수식열로 보이므로 세 후보 중 구조상 가장 가깝다.

## 재감사 사유

1. dataset card에 라이선스, 원출처, 작성자 동의가 없다.
2. `bs` 같은 label의 정규 문자 의미가 설명되지 않았다.
3. writer/session ID가 없어 train/validation/test 간 동일 작성자·중복 궤적을 검사할 수 없다.
4. 표본 수가 작고 수식/문자 분포가 제한적이다.

원출처와 상업 이용 허가를 확보하면 해시 중복, writer 분리, label mapping부터 다시 감사한다.
