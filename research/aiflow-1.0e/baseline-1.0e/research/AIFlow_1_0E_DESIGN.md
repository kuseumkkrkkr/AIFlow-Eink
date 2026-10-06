# AIFlow Math Ink 1.0e 설계 메모

상태: 2026-08-29 shadow 연구 트랙

## 확인된 1.0 구조

- 온라인 입력은 `x, y, delta_t, stroke_start, observed` 5채널 128점이다.
- box-local 인코더는 4층 Transformer, hidden 128, attention head 4, attention pooling이다.
- 문자 head는 372-class logits와 HWR Top-k 후보를 제공한다.
- 수식 최종기는 2층 BERT형 Transformer와 role/grammar/geometry 보조 점수다.
- 따라서 현재 1.0은 LSTM 모델이 아니다. LSTM에 가까운 부분이 있다면 순서가 있는 trajectory를 읽고 이전·다음 문맥을 이용한다는 기능적 유사성뿐이다.

## 1.0e 가설

대형 OCR 모델의 최종 decoder를 AIFlow 결정기로 그대로 대체하지 않는다. TexTeller 같은 image-to-LaTeX 모델은 전체 수식 문자열을 생성하므로 AIFlow의 `행·stroke·box·후보` 단위와 출력 단위가 다르다. 대신 수식 raster를 동일 입력에서 병렬 생성하고, 외부 OCR의 frozen visual embedding을 AIFlow 후보 점수 어댑터에 공급한다.

```text
원본 online stroke
   ├─ 기존 1.0 경로: layout → 372-class HWR Top-k
   └─ shadow raster: frozen external formula OCR encoder → formula evidence
                                      ↓
                       1.0e candidate-only decision adapter
                                      ↓
             기존 Top-k 중 하나 / abstain·raw fallback
```

## 불변 계약

- 외부 OCR 가중치는 동결한다. 새로 학습하는 것은 작은 decision adapter뿐이다.
- adapter는 HWR Top-k 밖의 token을 만들 수 없다.
- stroke 삭제·재분할·재그룹핑·관계 생성은 하지 않는다.
- project-owned formula/candidate만 writer-LOO 학습에 사용한다.
- CROHME 및 기타 비상업 자료는 이 버전의 선택·승격 데이터로 사용하지 않는다.
- 결과는 기존 1.0을 자동 교체하지 않고 shadow 비교만 한다.

## 실제 적용 경계

2026-08-29 현재 로컬 캐시에 완전한 가중치가 확인된 외부 수식 OCR은 고정 revision `OleehyO/TexTeller`다. TrOCR Math, PP-FormulaNet, LightOnOCR, Chandra 등은 캐시 설명 또는 불완전한 snapshot만 확인되어 이번 학습에 넣지 않는다. 따라서 이번 결과는 “여러 대형 OCR의 결정기를 파인튜닝한 결과”가 아니라, 첫 번째 호환 가능한 외부 OCR teacher를 이용한 1.0e adapter 실험이다.

구현은 [train_ocr_decision_adapter_10e.py](../scripts/train_ocr_decision_adapter_10e.py)이며, 새 체크포인트는 `artifacts/ocr_decision_adapter_10e_20260829/` 아래에 생성한다.

## 해석상 주의

이 실험이 좋아져도 대형 OCR 모델 자체의 수식 인식률이 AIFlow online HWR 인식률로 전이됐다는 뜻은 아니다. 외부 embedding은 formula-level evidence이고, 최종 출력은 여전히 AIFlow의 candidate-level decision이다. writer-disjoint·formula-disjoint 새 acceptance에서 기존 1.0 비회귀를 확인하기 전에는 제품 채택을 주장하지 않는다.
