# AIFlow 1.0e serial ink-teller training loop

## 결과

학습 오류 없이 3개 설정을 순차 실행했지만, 모든 설정의 held-writer Exact는 `0/47 (0.00%)`였다. 비교 기준인 기존 HWR candidate baseline은 `16/47 (34.04%)`다.

| 설정 | train data | epoch | online LR | bridge LR | decoder interface LR | Exact |
|---|---:|---:|---:|---:|---:|---:|
| `aux_e20_lr1e-3` | 110 | 20 | 1e-5 | 1e-3 | 2e-5 | 0/47 |
| `aux_e40_lr3e-3` | 110 | 40 | 2e-5 | 3e-3 | 1e-4 | 0/47 |
| `aux_e60_lr5e-3` | 110 | 60 | 5e-5 | 5e-3 | 2e-4 | 0/47 |

세 writer fold 모두 각 설정에서 `0`이었다.

## 적용한 loop

- 평가 대상은 기존 candidate formula 47개로 고정했다.
- 프로젝트 raw stroke/LaTeX 110개를 사용했다.
- 평가 writer를 제외한 raw만 train에 넣었다.
- AIFlow online adapter와 TexTeller decoder cross-attention interface를 튜닝했다.
- TexTeller vision/raster encoder는 모든 설정에서 실행하지 않았다.
- 외부 신규 데이터는 추가하지 않았다.

## 해석

이 결과는 단순히 epoch나 learning rate를 늘리는 문제가 아니라는 뜻이다. 현재 bridge가 매핑해야 하는 대상은 online trajectory representation과 TexTeller ViT encoder memory 사이의 분포이며, 110개 raw record만으로는 decoder가 `EOS`, delimiter, 단일 연산자/숫자 반복으로 붕괴하는 현상을 해결하지 못했다.

따라서 현재 serial path는 구조적으로는 동작하지만 성능상 기존 1.0e shadow보다 열세다. 추가적인 무작정 epoch 증가는 중단하고, 다음 단계는 paired online-ink–LaTeX 대규모 bridge pretraining 또는 TexTeller decoder와 동일한 token/memory convention을 가진 데이터 확보여야 한다.

## 판정

**serial decoder-only path: shadow/research-only, 상용 불가, 기존 raster evidence 경로 대체 불가.**

## 재현 산출물

- loop 실행기: `scripts/run_serial_ink_teller_training_loop_10e.py`
- serial 학습기: `scripts/train_serial_ink_teller_10e.py`
- loop summary: `artifacts/serial_ink_teller_training_loop_20260830/loop_summary.json`
- 각 설정의 `evaluation.json`, `serial_bridge.pt`, stdout/stderr log가 같은 폴더 아래에 있다.
