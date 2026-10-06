# AIFlow Math Ink 1.0e 외부 OCR 결정 어댑터 실험

실험일: 2026-08-29  
상태: `shadow_only`, 제품 기본 경로 미변경

## 결론

1.0은 LSTM 모델이 아니다. 현재 확인된 구조는 다음과 같다.

```text
5채널 온라인 ink(128점)
→ 4층 Transformer, hidden 128, head 4, attention pooling
→ 372-class HWR Top-k
→ 2층 BERT형 수식 문맥 결정기 + grammar/geometry guard
```

LSTM과 닮은 점은 stroke 순서와 앞뒤 문맥을 쓴다는 기능적 유사성뿐이다. 1.0e의 새 모델도 LSTM이 아니라, 외부 OCR의 formula-level 시각 증거와 AIFlow 후보별 정보를 결합하는 작은 MLP 어댑터다.

## 이번 1.0e 구현

- 외부 모델: 로컬에 완전한 가중치가 있는 `OleehyO/TexTeller`
- 고정 revision: `7b96df06b9d81cdb129c3bef68b7250bc3e2b0ea`
- 외부 OCR encoder: 동결
- 학습 대상: `ocr_decision_adapter.pt`의 158,545개 어댑터 파라미터
- 입력: 기존 HWR 후보 확률·기하·앞뒤 위치 문맥·후보 token role + OCR encoder pooled embedding
- 출력: 기존 HWR Top-k 후보 중 하나
- 금지: 후보 밖 token 생성, 행 삭제, stroke 재그룹핑, 관계 재작성
- 데이터: project-owned 47개 수식, 211개 후보 행, writer-LOO

TrOCR Math, PP-FormulaNet, LightOnOCR, Chandra도 캐시 목록에는 있었지만 이번 환경에는 완전한 호환 가중치 snapshot이 없어 실제 학습에 넣지 않았다. 그러므로 이번 결과를 “여러 대형 OCR의 결정기 파인튜닝”이라고 부를 수는 없고, “첫 번째 호환 외부 OCR teacher를 사용한 1.0e 어댑터 실험”이라고 부르는 것이 정확하다.

## 결과

| held writer | 후보 recall | 기존 HWR Top-1 | 1.0e adapter Top-1 | 기존 식 exact | 1.0e 식 exact |
|---|---:|---:|---:|---:|---:|
| `3e41…` | 94.29% | 81.82% | 72.73% | 37.50% | 12.50% |
| `baed…` | 100.00% | 72.37% | 88.16% | 31.25% | 50.00% |
| `ed678…` | 90.00% | 81.11% | 85.56% | 43.48% | 52.17% |
| aggregate | 94.31% | 77.89% | 84.42% | 38.30% | 44.68% |

aggregate만 보면 후보 내부 Top-1이 `+6.53%p`, 식 exact가 `+6.38%p` 개선됐다. 하지만 첫 writer fold에서 Top-1이 `-9.09%p`, 식 exact가 `-25.00%p` 하락했다. 따라서 외부 OCR 증거가 유효할 가능성은 보였지만, writer-disjoint 안정성은 확보하지 못했다.

또한 211행 중 12행은 정답이 HWR Top-5 안에 없었다. 이 12행은 결정기 개선만으로는 복구할 수 없는 후보 recall 병목이다.

## 독립 검증

- 예측 행 수와 입력 행 수 일치: 통과
- 모든 1.0e 선택 token이 해당 HWR 후보 안에 있음: 통과
- record ID 중복 없음: 통과
- 외부 trainable parameter: 0
- 결과 상태: `shadow_only`
- checkpoint 저장 후 재로딩·state shape·forward 계약: 통과

실험 산출물:

- [1.0e 설계 메모](../research/AIFlow_1_0E_DESIGN.md)
- [학습 스크립트](../scripts/train_ocr_decision_adapter_10e.py)
- `artifacts/ocr_decision_adapter_10e_20260829/evaluation.json`
- `artifacts/ocr_decision_adapter_10e_20260829/ocr_decision_adapter.pt`

## 적용하지 않은 기상천외한 가설

아래는 아이디어 목록일 뿐이며 이번 버전에 구현하지 않았다.

1. **OCR 합의제 국회**: TexTeller·TrOCR·FormulaNet·AIFlow가 매 글자마다 투표하고, 표결이 갈린 글자만 별도 심판 모델로 보낸다. 타개 포인트는 전체 재학습이 아니라 불일치 slice만 모으는 것이다.
2. **시간 역행 필기**: 마지막 stroke부터 거꾸로 읽어 “이 기호가 되기 직전의 의도”를 추정한다. 타개 포인트는 정방향·역방향이 같은 후보를 낼 때만 확정하는 교차 검증이다.
3. **필기체 위조 수사관**: writer style을 인식에 쓰지 않고, 오히려 “평소 이 작가가 절대 쓰지 않는 모양”을 이상치로 검출한다. 타개 포인트는 모양을 바꾸는 것이 아니라 REVIEW_REQUIRED를 더 일찍 내는 것이다.
4. **수식을 컴파일러로 취급**: OCR을 문장 번역이 아니라 `stroke program → syntax tree → executable formula` 컴파일로 보고, 문법 오류가 나는 후보만 되돌린다. 타개 포인트는 의미 보정과 사용자가 쓴 식 보존을 분리하는 것이다.
5. **반사실 우주 1만 개**: 같은 stroke를 1도씩 기울이거나 시간을 뒤집은 가상 우주를 만들고, 모든 우주에서 살아남는 후보만 채택한다. 타개 포인트는 증강 학습이 아니라 안정성 측정기로 먼저 쓰는 것이다.
6. **다음 stroke 예언자**: 현재까지의 stroke만 보고 사용자가 다음에 그을 위치와 방향을 예측해, 이미 쓴 기호의 후보를 역으로 좁힌다. 타개 포인트는 예측을 출력에 직접 넣지 않고 ambiguity 신호로만 쓰는 것이다.
7. **스스로 틀리는 전문가**: 일부러 가장 그럴듯한 오답 후보를 생성하는 모델을 별도로 두고, 주 모델이 그 오답을 설명하지 못하면 확정을 금지한다. 타개 포인트는 정확도 경쟁이 아니라 실패 설명 가능성을 gate로 삼는 것이다.
8. **잉크·픽셀 이중국적제**: 동일 수식을 online ink와 raster 두 시민권으로 동시에 심사하고, 둘의 국적이 다르면 문자를 고치지 않고 원본 stroke와 양쪽 후보를 함께 반환한다. 타개 포인트는 강제 fusion이 아니라 disagreement를 제품 신호로 활용하는 것이다.
