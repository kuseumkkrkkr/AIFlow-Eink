# AIFlow 1.0e 단일 외부 가중치 loop tuning

## 범위

- 외부 신경망은 `Azu/trocr-handwritten-math` 하나만 사용했다.
- HF 데이터셋으로 새 OCR을 학습하지 않았다. 기존 AIFlow project-owned candidate label과 writer split만 튜닝 신호로 사용했다.
- 매 trial은 동일한 pretrained revision `fc8dc9829360d42b1d4bc2f2668c831a72c80379`에서 시작했다.
- HWR Top-k 후보 생성·grouping·ownership·relation은 변경하지 않았다.

## Loop 구성

기존 `last1 / lr=3e-4 / 3epoch` 결과를 기준 trial로 재사용하고, 다음 네 조합을 새로 실행했다. 평가는 47 formulas, 211 rows, 3 writer의 writer-LOO다.

| trial | encoder 해동 | learning rate | epoch |
|---|---:|---:|---:|
| previous 기준 | last 1 block | 3e-4 | 3 |
| `last1_lr1e-4_e5` | last 1 block | 1e-4 | 5 |
| `last1_lr2e-4_e5` | last 1 block | 2e-4 | 5 |
| `last1_lr5e-4_e2` | last 1 block | 5e-4 | 2 |
| `last2_lr1e-4_e3` | last 2 blocks | 1e-4 | 3 |

선택 순서는 `전체 formula Exact → writer fold 회귀가 없는 후보 → 최소 writer fold Exact → 후보 포함 formula Exact`다.

## 결과

| trial | 전체 Formula Exact | 후보 포함 Formula Exact | writer fold 회귀 | 계약 위반 |
|---|---:|---:|---:|---:|
| `last1_lr5e-4_e2` **선정** | **44.68% (21/47)** | **55.32%** | 0 | 0 |
| previous `last1_lr3e-4_e3` | 44.68% (21/47) | 53.19% | 0 | 0 |
| `last1_lr2e-4_e5` | 44.68% (21/47) | 53.19% | 0 | 0 |
| `last1_lr1e-4_e5` | 34.04% (16/47) | 40.43% | 1 | 0 |
| `last2_lr1e-4_e3` | 10.64% (5/47) | 10.64% | 3 | 0 |

선정 trial의 held-writer Exact는 `50.00% / 56.25% / 34.78%`로 기준 trial과 같고, 후보 포함 Exact만 53.19%에서 55.32%로 올랐다. 후보 recall은 94.31%(199/211)로 변하지 않았으며, 후보 밖 12 row는 이 loop의 범위 밖이다.

## 판정

- `last1_lr5e-4_e2`를 단일 모델 shadow 후보로 기록한다.
- last 2 blocks 해동은 과적합/붕괴 징후가 명확해 폐기한다.
- 기존 TexTeller와 결합하지 않으며, 운영 승격은 하지 않는다.
- 상태: **shadow-only / single-model weight-tuned**.

## 산출물

- loop 실행기: `scripts/run_ocr_trocr_weight_tuning_loop_10e.py`
- 튜닝 실행기: `scripts/train_ocr_trocr_weight_tuned_10e.py`
- 선정 checkpoint: `artifacts/ocr_trocr_weight_tuning_loop_20260830/last1_lr5e-4_e2/ocr_trocr_weight_tuned.pt`
- 선정 audit: `artifacts/ocr_trocr_weight_tuning_loop_20260830/last1_lr5e-4_e2/audit.json`
- 전체 loop 결과: `artifacts/ocr_trocr_weight_tuning_loop_20260830/loop_report.json`
