# BERT-Tiny MASK 문맥 재랭커 적용 결과

상태: **구현·학습·재로딩 검증 완료 / shadow component 채택 / 기본 출력 교체 보류**

## 적용 구조

- 기존 `128점 × 5채널` HWR Transformer와 단일 372-class 출력부는 고정했다.
- 후단에 Google BERT-Tiny 기반 2-layer, hidden 128, attention head 2개 문맥 모델을 붙였다.
- 372개 문자를 각각 독립 custom token으로 만들고 `right`, `left`, `above`, `below`, `superscript`, `subscript`, `overlap` 관계 token 7개를 추가했다.
- 한 위치만 `[MASK]`로 바꾸고, 나머지 위치에는 해당 문자의 정답이 아니라 고정 HWR Top-1을 넣었다.
- 최종 점수는 `log P(HWR) + λ log P(MASK)`이며 선택 범위는 원래 HWR Top-5로 제한했다. 새 문자 생성·문자 삭제·stroke regrouping은 불가능하다.
- writer-LOO 바깥 fold 안에서만 epoch와 λ를 선택했다. λ=0도 후보라 검증 이득이 없으면 원래 HWR로 자동 복귀한다.

구현: `scripts/train_masked_context_reranker_v1.py`

## 사전학습 모델 고정

- Hub ID: `google/bert_uncased_L-2_H-128_A-2`
- revision: `30b0a37ccaaa32f332884b96992754e246e48c5f`
- 라이선스: Apache-2.0
- 확장 후 parameter: 4,465,589개
- `model.safetensors` SHA-256: `7fb69ad9f6866d8983183c930e33828f326470bf6ad8bbb2ad4ed957a92e9414`
- `config.json` SHA-256: `508e1f01aae55d73355cbdd82609be2f43ba5a0d3428837adbe56cf8391f8b39`
- `vocab.txt` SHA-256: `07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3`

가중치는 D 드라이브 `research/pretrained/google-bert-tiny`에 보관하고 Git에는 넣지 않았다.

```powershell
hf download google/bert_uncased_L-2_H-128_A-2 README.md config.json model.safetensors vocab.txt --revision 30b0a37ccaaa32f332884b96992754e246e48c5f --local-dir research\pretrained\google-bert-tiny
```

## 데이터 경계

| 구분 | 처리 |
|---|---|
| 프로젝트 ownership 47식·211자·3 writer group | 유일한 문맥 학습원 |
| 직접 수집 writer-LOO | 바깥 writer 단위 평가, 안쪽 formula-disjoint 선택 |
| CROHME 984식·9,535 지원 문자 | CC BY-NC 평가 전용, 학습·epoch·λ 선택에 사용하지 않음 |
| 직접 canonical 10식 | 문자 ownership이 없어 문맥 정확도 산출 제외 |
| 외부 문자 데이터 10% | 고립 문자라 수식 문맥·관계가 없어 문맥 정확도 산출 제외 |

## 결과

| 평가 | 고정 HWR | MASK 문맥 | 변화 |
|---|---:|---:|---:|
| 직접 writer-LOO 문자 Top-1 | 73.46% | **84.36%** | +10.90%p |
| 직접 strict 동형군 micro | 39.62% | **67.92%** | +28.30%p |
| 직접 strict 동형군 macro | 57.94% | **64.97%** | +7.03%p |
| 직접 수식 exact | 34.04% | **53.19%** | +19.15%p |
| CROHME 전이 문자 Top-1 | 69.50% | **76.36%** | +6.86%p |
| CROHME strict 동형군 micro | 38.28% | **63.80%** | +25.52%p |
| CROHME strict 동형군 macro | **42.64%** | 37.69% | **-4.95%p** |
| CROHME 수식 exact | 15.85% | **25.10%** | +9.25%p |

- 직접 writer-LOO: 32건 변경, 25건 개선, 2건 회귀.
- CROHME 전이: 1,392건 변경, 821건 개선, 167건 회귀.
- 모든 9,746개 평가 문자에서 Top-5 후보 보존율 100%, 새 token 0건, grouping 변경 0건이다.
- 직접 writer-LOO의 기존 one-hot 문맥 모델은 문자 Top-1 81.04%, 수식 exact 44.68%였다. BERT-Tiny가 두 수치는 높였지만 one-hot의 회귀 0건보다 보수성은 낮다.

## 동형문자 중검수

프로젝트 문맥 학습원에는 `|`, `O`, `o` 정답이 한 건도 없다. 그 결과 전체·micro 정확도는 높아졌지만 CROHME에서 다수 클래스 쪽으로 치우쳤다.

| 정답 | HWR 적중 | MASK 적중 |
|---|---:|---:|
| `1` | 278 / 721 | **547 / 721** |
| `0` | 91 / 213 | **167 / 213** |
| `x` | 191 / 587 | **341 / 587** |
| `|` | **38 / 62** | 2 / 62 |
| `/` | **13 / 19** | 6 / 19 |
| `\times` | **33 / 72** | 12 / 72 |
| `o` | **1 / 11** | 0 / 11 |

따라서 전체 정확도만으로 기본 모델을 교체하면 안 된다. `|·O·o`의 프로젝트 소유 양성 문맥과 `/·\times` 보강 문맥을 수집한 뒤, 새 writer/formula untouched 세트에서 macro 비회귀를 확인해야 한다.

## 생성물과 재현 검증

최종 실행 폴더: `artifacts/masked_context_bert_tiny_20260814_r3`

- `masked_context_product.pt`: 33,839,202 bytes, SHA-256 `b713198ec716aa2d8e70b3801f4dee1340b9d3232d9f06792e2327a234c14856`
- `masked_context_report.json`: 전체 fold·선택·평가 기록
- `direct_writer_loo_predictions.jsonl.gz`: 누수 없는 직접 수집 평가 예측
- `direct_product_refit_predictions.jsonl.gz`: 전체 47식 재학습 모델의 training-fit 예측, 승인 근거 아님
- `crohme_transfer_predictions.jsonl.gz`: 프로젝트 데이터만 학습한 모델의 비상업 전이 진단

CUDA self-test와 세 차례 결정적 재실행을 통과했다. 최종 체크포인트를 새 프로세스에서 다시 읽어 direct product-refit 211건과 CROHME 9,535건을 재계산했으며 저장 예측 불일치 0건, Top-5 밖 선택 0건이었다.

이 PC에서는 전역 user-site의 `huggingface-hub 1.24.0`과 `transformers 4.48.0`이 충돌하므로 아래처럼 검증된 system-site 조합을 사용한다.

```powershell
$env:PYTHONNOUSERSITE='1'
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
python scripts\train_masked_context_reranker_v1.py --device cuda --output artifacts\masked_context_bert_tiny_20260814_r3
```

## 결정

- `checkpoint_created=true`
- `candidate_preserving=true`
- `shadow_component=true`
- `automatic_default_replacement=false`
- `product_promotion_gate=false`

실행 가능한 모델은 적용했지만, 희소 동형문자 macro 회귀가 있어 현재 372-head 기본 출력은 그대로 유지한다.
