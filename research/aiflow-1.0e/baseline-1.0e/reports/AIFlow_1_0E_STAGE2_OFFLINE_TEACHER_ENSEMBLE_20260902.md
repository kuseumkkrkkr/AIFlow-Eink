# AIFlow 1.0e 2단계 offline teacher ensemble shadow 결과

## 판정

- **안전 승격 gate: PASS**
  - Top-1: `317/387` → `317/387` 비회귀
  - Top-5 / 후보 recall: `374/387` → `374/387` 비회귀
  - Formula exact: `49/95` → `49/95` 비회귀
  - nested writer-LOO held-writer row regression: `0`
  - candidate contract violation: `0`
- **효용 승격: 보류**
  - guarded 변경은 1행이지만 오답 `\longmapsto`를 다른 오답 `H`로 바꿔 정답 증가는 `0`이다.
  - 안전 gate는 통과했으나 측정된 정확도 이득이 없어 product runtime에 채택하지 않는다.
- 기존 5-channel ordered online-HWR, 후보 보존, raw fallback 계약을 유지했다.
- product runtime과 `hf-model-1.0e` 및 Hugging Face 공개본은 변경하지 않았다.

## 동일 writer-LOO 비교

평가는 project-owned 7 writers, 387 rows, 95 formulas의 동일 기존 Top-k 후보에서 수행했다. 아래 수치는 full-expression OCR ExpRate가 아니라 candidate-only OOF reranking 결과다.

| 후보 | Top-1 | Formula exact | Row regression |
|---|---:|---:|---:|
| v4 baseline | 317/387 (81.91%) | 49/95 (51.58%) | 0 |
| TexTeller frozen feature | 334/387 (86.30%) | 55/95 (57.89%) | 20 |
| UniMERNet tiny frozen feature | 329/387 (85.01%) | 51/95 (53.68%) | 20 |
| Microsoft TrOCR-small frozen feature | 338/387 (87.34%) | 60/95 (63.16%) | 22 |
| 고정 uniform probability ensemble | **342/387 (88.37%)** | **60/95 (63.16%)** | 15 |
| raster-free student raw | 330/387 (85.27%) | 53/95 (55.79%) | 24 |
| raster-free student + nested guard | 317/387 (81.91%) | 49/95 (51.58%) | **0** |

## 구현 경계

- Offline teacher:
  - ordered strokes를 project-owned raster로 렌더링한다.
  - TexTeller 768-dim, UniMERNet tiny 512-dim, Microsoft TrOCR-small 384-dim frozen encoder feature를 추출한다.
  - 각 모델을 동일 outer writer-LOO candidate adapter로 비교한다.
  - 모델별 OOF softmax 확률을 라벨 가중치 튜닝 없이 1/3씩 평균한다.
- Raster-free student:
  - 입력은 frozen v4 HWR의 128-dim ordered-ink embedding, 기존 후보 확률, geometry/context, 후보 ID뿐이다.
  - teacher ensemble은 학습 KL target으로만 사용한다.
  - loss는 `0.20 CE + 0.70 teacher KL + 0.10 baseline stability + 0.10 pairwise margin`이다.
  - 후보 생성, 행 삭제, stroke regrouping, relation mutation은 불가하다.
- Nested guard:
  - 각 held writer의 threshold는 나머지 writers에서 row regression 0인 조합 중 formula exact, Top-1, 변경 최소 순으로 선택한다.
  - runtime ground truth는 사용하지 않는다.

## 재현 명령

호환 환경은 기존 OCR 격리 venv의 `huggingface-hub==0.36.0`을 사용하며 모델 접근은 `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`로 차단한다.

```powershell
$repo = 'D:\AIFlow-Workspace\Projects\Aiflow\aiflow-math-ink-1.0'
$ocr = 'D:\AIFlow-Workspace\Projects\Aiflow\ocr-orchestration-simulation'
$py = "$ocr\.venv-texteller\Scripts\python.exe"
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'

& $py "$repo\scripts\extract_offline_teacher_features_10e.py" --kind texteller --model 'D:\AIFlow-Workspace\Caches\user\.cache\huggingface\hub\models--OleehyO--TexTeller\snapshots\7b96df06b9d81cdb129c3bef68b7250bc3e2b0ea' --output "$repo\artifacts\offline_teacher_texteller_20260902_stage2_r1"
& $py "$repo\scripts\extract_offline_teacher_features_10e.py" --kind trocr-small --model 'D:\AIFlow-Workspace\Caches\user\.cache\huggingface\hub\models--microsoft--trocr-small-handwritten\snapshots\b4648cfa171985a6745f37ddd637e98c0da958ac' --output "$repo\artifacts\offline_teacher_trocr_small_20260902_stage2_r1" --batch-size 4

Push-Location $ocr
& '.\.venv-unimernet\Scripts\python.exe' "$repo\scripts\extract_unimernet_teacher_features_10e.py" --config benchmarks\unimernet_tiny.yaml --weights models\unimernet_tiny\unimernet_tiny.pth --output "$repo\artifacts\offline_teacher_unimernet_tiny_20260902_stage2_r1" --batch-size 4
Pop-Location

Push-Location $repo
& $py scripts\train_offline_teacher_ensemble_10e.py --texteller artifacts\offline_teacher_texteller_20260902_stage2_r1\features.npz --unimernet artifacts\offline_teacher_unimernet_tiny_20260902_stage2_r1\features.npz --trocr-small artifacts\offline_teacher_trocr_small_20260902_stage2_r1\features.npz --output artifacts\offline_teacher_ensemble_20260902_stage2_r2 --epochs 3 --device cpu
& $py scripts\train_teacher_ensemble_raster_free_reranker_10e.py --teacher artifacts\offline_teacher_ensemble_20260902_stage2_r2\ensemble_probability_mean_writer_loo_predictions.jsonl.gz --output artifacts\teacher_ensemble_raster_free_reranker_20260902_stage2_r2 --epochs 3 --device cpu
& $py scripts\evaluate_online_distill_nested_guard_v1.py --predictions artifacts\teacher_ensemble_raster_free_reranker_20260902_stage2_r2\writer_loo_predictions.jsonl.gz --output artifacts\teacher_ensemble_raster_free_nested_guard_20260902_stage2_r2
Pop-Location
```

기존 출력 경로는 overwrite를 거부하므로 완전 재현 시 새 suffix를 사용해야 한다.

## 주요 SHA-256

- TexTeller feature: `8e78199f1a4ed21d8a25a66644f183dc3743a1f5b970ea9d57d185de63cd860f`
- UniMERNet tiny feature: `f16ed96fbe4fe2e909b505cd00677f6da2dfeaa58714e6cbb2ef260b0d22512`
- Microsoft TrOCR-small feature: `eb6f5fffb92d7344d6fde8c58b573d1c46ed7f638a292b1b58f98e2c3f09042e`
- Teacher ensemble evaluation: `e826e521d0c895c7278bbd2263e65599192a80537195e3442e2e8c048546a017`
- Raster-free checkpoint: `8a0b94c8009cc35397c9cba23c67e619243da0190bc6474fabbd60a77385afed`
- Nested guard report: `7ec2b51fe154d600134a7c5982164d76f84a69448330ae1b10ee01f1ccdf9366`

## 잔여 gate

- fresh unused writer/formula-disjoint acceptance가 없다.
- Android/실서비스 latency, memory, battery 검증을 수행하지 않았다.
- 따라서 `product_adopted=false`, `shadow_only`를 유지한다.
