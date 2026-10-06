# 재감사 필요: `newbienewbie/handwriting_strokes_2strokes`

- HF 저장소: https://huggingface.co/datasets/newbienewbie/handwriting_strokes_2strokes
- 확인 revision: `6e2173aa7f93467850b5cffced414057bf242454`
- 규모: train 894 / validation 310 / test 310
- 현재 판정: **학습 반입 금지**

## 형식 확인

- 행마다 `xCoordinates`, `yCoordinates`, `timeStamps`가 있어 궤적 포맷 자체는 호환 가능성이 있다.
- 실제 행의 `sampleTag`는 `0_0`, `1_1`처럼 표본/획 순번으로 보였고 `annotation`은 `ABDEFGHIJKPQRTXY:=`라는 공통 클래스 목록이었다.
- 따라서 행 label, 두 stroke의 glyph 소유권, 원래 문자 단위 결합 규칙을 공개 metadata만으로 확정할 수 없다.

## 재감사 사유

1. dataset card에 라이선스와 원출처가 없다.
2. 문자 label 매핑과 stroke 결합 키가 명시되지 않았다.
3. writer/session 식별자와 split 독립성이 없다.
4. 상업 이용 및 파생 모델 배포 권한을 확인할 수 없다.

권리 문서와 완전한 schema 설명을 확보하기 전에는 1.0 학습 manifest에 넣지 않는다.
