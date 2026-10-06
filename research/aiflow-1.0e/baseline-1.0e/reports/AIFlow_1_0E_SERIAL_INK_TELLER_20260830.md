# AIFlow 1.0e serial online-ink → TexTeller decoder experiment

## 판정

- **TexTeller 분해 자체는 가능**했다. `model.safetensors`에서 decoder namespace 316개만 추출해 독립 `TrOCRForCausalLM`으로 로드했다.
- **래스터 encoder는 실행하지 않았다.** 입력은 `formulas_valid.jsonl`의 ordered raw stroke/timestamp뿐이다.
- **현재 serial 품질은 기각**이다. writer-LOO 전체 Exact가 `0/47 (0.00%)`로, 기존 HWR Top-1 baseline `16/47 (34.04%)`보다 낮았다.
- 따라서 이 결과는 **구조 feasibility 증거이지 상용 후보가 아니다**. 기존 TexTeller raster twin 경로를 대체하거나 1.0e 운영 승격하지 않는다.

## 구성

```text
raw ordered strokes
  → experimental 19-channel / 128-event online encoding
  → AIFlow 0.6 public online adapter + shared trajectory prior
  → ordered 16-token bridge (128 → 768)
  → TexTeller decoder only
  → free-form LaTeX
```

- AIFlow online prior: `base_378.pt` + `online_adapter.pt`; trajectory encoder frozen, online adapter tuned.
- TexTeller: revision `7b96df06b9d81cdb129c3bef68b7250bc3e2b0ea`.
- TexTeller decoder parameters loaded: `211,681,280`.
- Tuned decoder interface: `44,113,920` cross-attention/layer-norm parameters.
- Tuned AIFlow online adapter: `20,691` parameters.
- Bridge: approximately `464k` parameters.
- 47 formulas, 3 writer groups, 20 epochs, batch size 8, CUDA GTX 1650.
- Decoder target은 TexTeller convention에 맞춰 `\[...\]`로 감쌌고, 평가는 wrapper/operator alias를 canonicalize했다.

## 결과

| 지표 | 결과 |
|---|---:|
| HWR candidate baseline formula Exact | 16/47 (34.04%) |
| serial decoder-only formula Exact | 0/47 (0.00%) |
| held writer `3e41…` | 0/8 |
| held writer `baed…` | 0/16 |
| held writer `ed678…` | 0/23 |
| raster encoder 실행 | false |
| candidate-only 계약 | false; free-form decoder |

실패 출력은 초기 EOS, `\[`, `\]`, 연산자 또는 단일 숫자의 반복으로 나타났다. 온라인 표현이 수식 형태 정보를 갖더라도, TexTeller decoder가 학습한 ViT encoder memory 분포와 단순 bridge 출력 분포 사이의 간극을 47개 수식으로 메우지 못했다.

## 해석 및 다음 gate

1. decoder와 vision encoder를 코드/가중치 수준에서 분리하는 것은 가능하다.
2. 그러나 decoder-only 이식에는 paired online-ink/formula 데이터로 bridge를 별도 사전학습하거나, 충분한 양의 paired 데이터에서 decoder cross-attention을 공동 튜닝해야 한다.
3. 현재 입력은 1.0e 최종 5-channel 계약이 아니라 공개 0.6의 19-channel 연구 계약을 실험적으로 재구성한 것이다.
4. 공개 AIFlow checkpoint metadata는 `product_validation=false`이며 연구용/noncommercial 경계가 있다.
5. 따라서 현재 산출물은 **shadow/research-only**, 상용 가능 판정은 **불가**다.

## 재현 산출물

- 실행 코드: `scripts/train_serial_ink_teller_10e.py`
- 평가 JSON: `artifacts/serial_ink_teller_10e_20260830/evaluation.json`
- writer-LOO 예측: `artifacts/serial_ink_teller_10e_20260830/writer_loo_predictions.jsonl.gz`
- refit shadow checkpoint: `artifacts/serial_ink_teller_10e_20260830/serial_bridge.pt`
- smoke 결과: `artifacts/serial_ink_teller_10e_smoke3/`

재현 명령:

```powershell
$env:PYTHONPATH='C:\Users\user\Documents\Codex\2026-08-30\ai\work\hfcompat'
& 'C:\Python311\python.exe' -u 'scripts\train_serial_ink_teller_10e.py' `
  --epochs 20 --batch-size 8 --max-length 24 `
  --output 'artifacts\serial_ink_teller_10e_20260830' --device cuda
```
