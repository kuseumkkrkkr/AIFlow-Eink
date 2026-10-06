# AIFlow 1.0e 단일 외부 가중치 튜닝 실험

> 후속 hyperparameter/해동 범위 loop 결과는 `AIFlow_1_0E_WEIGHT_TUNING_LOOP_20260830.md`에 기록했다. 현재 shadow 선정 후보는 `last1 / learning-rate 5e-4 / 2epoch`다.

## 요청 반영

- Hugging Face에서 추가 HME 후보를 조사했다.
- 데이터셋으로 새 OCR을 처음부터 학습하지 않았다.
- Hugging Face에서 사전학습된 **단일** OCR 가중치 `Azu/trocr-handwritten-math`를 내려받아 사용했다.
- 기존 TexTeller는 이번 실행의 입력에서 완전히 제외했다. 두 OCR을 twin/fusion으로 결합하지 않았다.

## 선택 근거

| 후보 | 확인 결과 | 이번 역할 |
|---|---|---|
| `deepcopy/MathWriting-human` | 230k human image-LaTeX, CC BY-NC-SA | 비상업 연구 후보. 이번 튜닝 입력에는 넣지 않음 |
| `ToniDO/TeXtract_dataset` | 약 3.2M WebDataset, gated, MIT 표기 | 대용량이지만 gated/대형이라 이번 weight-tuning 범위에서 제외 |
| `deepcopy/Aida-Calculus-Math-Handwriting` | 100k 합성 calculus image-LaTeX, CDLA-Sharing-1.0 | 데이터 학습 요청이 아니므로 제외 |
| `Azu/trocr-handwritten-math` | VisionEncoderDecoder, 246.5MB `pytorch_model.bin`, CROHME 2014 HMER 가중치 | **선정한 단일 외부 신경망** |

모델 revision은 `fc8dc9829360d42b1d4bc2f2668c831a72c80379`로 고정했다. Azu 저장소에는 tokenizer가 없어, tokenizer/두 번째 OCR을 추가하지 않고 공개 TrOCR ViT 전처리만 고정했다. 따라서 실행 모델은 Azu checkpoint 하나다.

## 튜닝 경계

가져온 Azu checkpoint를 각 writer-LOO fold의 시작점으로 복원한 뒤, 외부 encoder의 마지막 DeiT block `encoder.encoder.layer[-1]`만 **1,774,464개 parameter** 업데이트했다. 여기에 **109,393-parameter candidate selector adapter**를 함께 튜닝했다. 즉, 새 OCR을 데이터셋으로 처음부터 학습한 것이 아니라, HF pretrained weight를 1.0e 후보 선택 문제에 맞게 부분 fine-tuning한 것이다. 비교용 frozen-encoder adapter 실험은 별도 ablation으로만 보관했다.

변경하지 않은 것:

- online stroke 순서·좌표·시간
- formula grouping 및 stroke ownership
- spatial relation
- HWR Top-k 후보 생성
- 후보 밖 token 생성 및 row 삭제

## 평가 결과

대상은 기존 프로젝트 소유 47 formulas, 211 candidate rows, 3 writer groups다. 학습·평가는 writer-LOO로 수행했다.

| 지표 | HWR baseline | 단일 Azu-TrOCR **weight-tuned** |
|---|---:|---:|
| 후보 포함 행 Top-1 | 77.89% | 86.43% |
| 후보 포함 행 formula Exact | 38.30% | 53.19% |
| 후보 recall | 94.31% (199/211) | 동일 |
| 전체 formula Exact | 34.04% (16/47) | 44.68% (21/47) |

Writer-LOO 전체 formula Exact는 다음과 같다.

| held writer | baseline | weight-tuned |
|---|---:|---:|
| `3e41…` | 37.50% (3/8) | 50.00% (4/8) |
| `baed…` | 31.25% (5/16) | 56.25% (9/16) |
| `ed67…` | 34.78% (8/23) | 34.78% (8/23) |

후보 계약 감사 결과는 위반 **0건**이다. 후보에 정답이 없는 12개 row는 여전히 복구되지 않았으므로, 이 모델도 후보 생성 문제를 해결하지 않는다.

## 판정

- 단일 외부 pretrained weight의 부분 fine-tuning 경로로 유효한 개선이 확인됐다.
- 현재 TexTeller shadow 후보보다 약하므로 기존 1.0e 운영/승격 모델을 교체하지 않는다.
- Azu 모델 카드의 라이선스 표기가 확인되지 않아 상용 탑재 판정은 하지 않는다.
- 상태: **shadow-only / single-model weight-tuned candidate**.

## 산출물

- 실행 코드: `scripts/train_ocr_trocr_weight_tuned_10e.py`
- 공통 계약 감사: `scripts/audit_ocr_text_evidence_adapter_10e.py`
- 선택 checkpoint: `artifacts/ocr_trocr_weight_tuned_10e_20260830_r1/ocr_trocr_weight_tuned.pt`
- evaluation: `artifacts/ocr_trocr_weight_tuned_10e_20260830_r1/evaluation.json`
- audit: `artifacts/ocr_trocr_weight_tuned_10e_20260830_r1/audit.json`
- frozen-encoder ablation: `artifacts/ocr_trocr_adapter_10e_20260830_r2/`

## 재현 명령

```powershell
@'
import sys,runpy
sys.path.insert(0,r'C:\Users\user\Documents\Codex\2026-08-30\ai\work\hfcompat')
sys.path.insert(0,r'D:\AIFlow-Workspace\Projects\Aiflow\aiflow-math-ink-1.0\scripts')
sys.argv=['scripts/train_ocr_trocr_weight_tuned_10e.py','--epochs','3','--output','artifacts/ocr_trocr_weight_tuned_10e_20260830_r1','--device','cuda','--batch-size','8','--learning-rate','3e-4']
runpy.run_path('scripts/train_ocr_trocr_weight_tuned_10e.py',run_name='__main__')
'@ | C:\Python311\python.exe -
C:\Python311\python.exe scripts\audit_ocr_text_evidence_adapter_10e.py --artifact artifacts/ocr_trocr_weight_tuned_10e_20260830_r1
```
