# AIFlow Math Ink 0.6 Android runtime

이 모듈은 0.6 모델의 Android ODA 공개 경계다. ML Kit Digital Ink 모델을 교체하는 파일이 아니라, AIFlow가 변환한 두 LiteRT flatbuffer를 최신 `CompiledModel` API로 직접 실행한다.

## 공개 API

```kotlin
val session = CompiledModelLiteRtSession.create(
    context = context,
    onlineAsset = "aiflow_math_ink_06_online.tflite",
    rasterAsset = "aiflow_math_ink_06_raster.tflite",
    modelVersion = "aiflow-math-ink-0.6",
    exactLabelCount = labels.size,
)
val recognizer = AIFlowMathInk(session, labels)

val online: SymbolResult = recognizer.recognizeOnline(strokes, InkCanvas(width, height))
val raster: SymbolResult = recognizer.recognizeRaster(bitmap)
val debug: RasterDebugResult = recognizer.recognizeRasterDebug(
    RasterInput(128, 128, normalizedInk),
)
```

`SymbolResult.toServerPayload()`에는 다음 값만 들어간다.

- `candidates[{token, probability}]`
- `confidence`
- `modelVersion`
- `latencyMs`

원본 touch event, canonical tap, raster, virtual stroke는 payload에 들어가지 않는다.

`recognizeRasterDebug`는 같은 로컬 inference에서 top-4의 128×2 좌표, 128×3 pen-state logits, progress, hypothesis log-probability를 반환한다. `RasterDebugResult`에는 server payload 변환 함수가 없으며 일반 `SymbolResult.toServerPayload()`에도 이 값은 포함되지 않는다.

## 입력 계약

- 원본 touch event는 `AIFlowInkV2.rawStrokes`에 로컬 보존
- 관측 timestamp는 6Hz로 재표본화
- timestamp가 없으면 `TimestampMode.CANONICAL`, 19번 중 18번째 `missing_mask=1`
- 시작·끝·pen-up anchor 보존
- 6Hz tap과 TCN의 128-step feature를 분리
- Python 학습 전처리의 대표 4행×19채널과 절대오차 `1e-4` 이내 parity

## 빌드

Android Studio JBR과 Android SDK 35가 필요하다.

```powershell
gradle :aiflow-math-ink-runtime:testDebugUnitTest
gradle :aiflow-math-ink-runtime:assembleRelease
```

현재 LiteRT dependency는 `com.google.ai.edge.litert:litert:2.1.6`, Kotlin은 2.3.0이다. 실제 `.tflite`가 아직 생성되지 않았으므로 CompiledModel 실기기 inference·latency·memory·battery 검증은 미완료다.

## Android 3-tier benchmark

`AndroidBenchmarkRunner`는 동일한 online/raster model SHA-256 쌍과 representative tensor로 저가·중급·고급 기기를 각각 측정한다. 두 hash는 `online:<sha>\nraster:<sha>\n`의 SHA-256인 `model_bundle_sha256`으로도 결합해 모델 순서까지 고정한다.

- warm-up: online/raster 각 20회
- 측정: online/raster 각 100회 교차 실행
- gate: online p95 ≤50ms, raster p95 ≤200ms, process PSS ≤100MiB
- 배터리: charge counter의 측정 가능 여부와 실행 전후 차이를 기록
- 결과: `AndroidBenchmarkReport.toMap()`을 앱 계층에서 UTF-8 JSON으로 저장
- raster graph output: exact logits, coordinates, state logits, progress, hypothesis scores의 정확히 5개

세 기기 JSON은 다음 명령으로 합친다.

```powershell
python scripts/summarize_math_ink_06_android_benchmarks.py `
  --report low.json --report mid.json --report high.json `
  --output android-3tier-summary.json
```

요약기는 `low`·`mid`·`high`가 정확히 하나씩 존재하고 세 report의 `model_version`, online/raster 개별 SHA-256, bundle SHA-256이 같은지 확인한다. 모든 metric이 세 기기에서 통과해야 `android_release_gate_passed=true`가 된다. 이 값만으로 제품 승인을 만들 수 없도록 `product_validation=false`는 고정한다.

최종 제품 manifest는 P writer/device-disjoint release, 배포 raster 모델에 귀속된 vectorization 지표, LiteRT parity/model bundle, Android 3-tier summary를 모두 결합한다.

공개 Hugging Face의 Colab ZIP은 `R_public_conversion_only`이며 P student가 들어 있지 않다. 실제 제품용 변환은 로컬에서 `build_math_ink_06_p_mobile_colab_bundle.py`로 private ZIP을 만든 뒤 본인 Colab 세션에만 올린다. 이 private manifest는 `contains_p_data=true`, `public_upload_allowed=false`, `server_upload_permitted=false`를 강제한다.

```powershell
python scripts/build_math_ink_06_product_release_manifest.py `
  --p-release release_report.json `
  --raster-validation raster-validation.json `
  --model-bundle mobile-model-bundle.json `
  --android-summary android-3tier-summary.json `
  --output product-release.json
```

공식 참고:

- https://developers.google.com/edge/litert/android
- https://developers.google.com/edge/litert/next/android_kotlin
