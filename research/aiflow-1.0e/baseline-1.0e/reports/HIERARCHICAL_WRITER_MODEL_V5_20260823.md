# 계층형 인간 작가 모델 v5 구현 결과

## 결과

- 전역 필기체 변형만 하던 v4를 `writer → token/allograph plan → formula relation/layout → production order → motor event` 계층으로 확장했다.
- 숫자·대문자·소문자와 수학 기호를 포함한 372개 출력 토큰의 clean-room 합성 궤적 4,608행을 catalog화했다.
- 의미 있는 수식 3종 `A+3=7`, `x^2+y^2=25`, `(a+b)/2`를 네 작가로 생성해 12개 formula event stream, 119개 provenance 보존 획을 만들었다.
- 제품 runtime, HWR/context 체크포인트, 학습 데이터는 변경하지 않았다. CROHME·MathWriting 사용량은 0행이다.

## 구현 계약

| 계층 | 구현 |
|---|---|
| Writer global | slant, 폭/높이, 기준선, roundness, 문자 간격, 획·글리프 휴지시간을 작가별 고정 |
| Allograph plan | 토큰, 획수, 시작/끝 구역, 획 방향서명, synthetic ID, 부모 hash, cross-writer 여부 보존 |
| Formula layout | baseline, superscript, numerator, denominator, fraction bar와 parent relation 분리 |
| Production order | left-to-right, baseline-first, structure-first를 공간 배치와 별개로 기록 |
| Motor events | 48 Hz 기준 시간과 작가별 휴지시간으로 raw event 생성; 글리프 인식 입력은 128×5 정규화 계약 유지 |

## 스모크 결과

| 합성 작가 | 영숫자 61종 Top-1 | Top-5 |
|---|---:|---:|
| compact upright | 70.49% | 100.00% |
| slanted round | 75.41% | 100.00% |
| wide relaxed | 73.77% | 100.00% |
| narrow quick | 67.21% | 100.00% |

- Top-5는 같은 계보의 동결 HWR로 후보를 거른 label-safety 결과다. 신규 작가 일반화 정확도가 아니다.
- 128×5 비공간 채널, 글리프 exact partition, 획 provenance, 단조 증가 이벤트 시간, 유한 좌표 검사를 통과했다.
- 시각 검수에서 작가별 기울기·비율·숫자/영문 형태와 production mode 차이를 확인했다.
- `+`는 첫 catalog보다 개선됐지만 일부 작가에서 가로획이 짧다. 새 작가 원시 수식 자료 없이는 인간 allograph로 채택하지 않는다.

## 증거 등급

- catalog 4,608행은 raw observed writer sample이 아니라 승인 원천으로 만든 DTW/profile/physics 합성 bank다.
- `366개 토큰에 복수 plan signature`는 합성 후보의 구조 다양성이지 인간에게서 관측된 allograph 분포가 아니다.
- 각 선택 plan에 `synthetic_id`, anchor/partner/style parent hash, `cross_writer`, parent writer fingerprint를 저장했다.
- fresh writer-disjoint acceptance와 raw-observed allograph evidence는 모두 `false`다.

## 미해결

1. 출력 어휘에 소문자 `t`가 없어 완전한 A-Z/a-z 지원이 아니다. 원시 `t`와 `t/f/+` 동형군을 수집한 뒤 출력부 migration 및 shadow 재학습이 필요하다.
2. 실제 인간의 획순·지연 획 빈도를 추정할 raw writer-session 데이터가 없다. 현재 production mode는 구조 스모크용 가설이다.
3. label-safety는 동일 계보 순환선택이므로 승격 근거가 아니다. 새 작가·새 수식 고정 acceptance set이 필요하다.
4. 수식 생성은 관계와 이벤트 계약을 검증했지만 formula recognizer exact 정확도는 아직 측정하지 않았다.

## 산출물

- 코드: `scripts/hierarchical_writer_model_v5.py`
- 연구: `reports/HUMAN_WRITER_MODEL_RESEARCH_20260823.md`
- 결과: `artifacts/hierarchical_writer_model_v5_20260823_r5/report.json`
- formula event: `artifacts/hierarchical_writer_model_v5_20260823_r5/formula_generation.jsonl`
- 시각화: `artifacts/hierarchical_writer_model_v5_20260823_r5/same_formula_four_writers.png`
