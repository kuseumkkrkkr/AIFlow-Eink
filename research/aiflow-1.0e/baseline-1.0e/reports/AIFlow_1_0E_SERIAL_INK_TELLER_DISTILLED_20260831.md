# AIFlow 1.0e serial distilled gate

## 개념

- 입력은 raster가 아닌 순서가 있는 raw online stroke입니다.
- AIFlow online prior가 19-channel trajectory를 인코딩합니다.
- 128개 event를 temporal stride 8로 압축해 16개 memory token으로 만들고, 768차원 decoder cross-attention 공간으로 투영합니다.
- TexTeller ViT는 학습 중 latent target을 만드는 offline teacher로만 사용했습니다.
- runtime에는 raster encoder와 teacher가 저장되거나 호출되지 않습니다.

## 실행 증거

- Offline teacher target: `110 × 16 × 768`, 생성 완료.
- Teacher target 파일: `artifacts/serial_ink_teller_teacher_targets_20260831.npz`.
- Distill + decoder tuning: 20 + 20 epoch, CUDA, exit code 0.
- Refit distill loss: `0.16810409724712372`.
- Runtime audit: `raster_runtime=false`, `teacher_runtime=false`, `external_new_data=false`.
- Decoder-only safetensors load: 316 decoder tensors, `211681280` parameters.

## 정확도와 과적합 진단

- Writer-LOO: `0/47` exact (`0.00%`).
- Existing serial baseline: `16/47` exact (`34.04%`).
- 1 formula, 60 epoch train-set overfit diagnostic: target `\[1+1=2\]`, prediction `11111111111111111111111`, exact `false`.
- 해석: teacher latent 정렬은 학습됐지만 decoder가 train-set 한 식도 닫지 못했으므로, 이 경로의 수식 exact 출력 계약은 아직 성립하지 않습니다.

## 상용 게이트

- 판정: `FAIL / shadow-only`.
- 이유: writer-LOO exact 0/47, train-set overfit false, AIFlow 0.6 base의 product validation false.
- 현재 산출물은 online-only serial 연구 후보이며 상용 OCR 모델로 홍보하거나 제품 채택할 근거가 없습니다.

## Hub 배포

- 공개 저장소: https://huggingface.co/cwLeeDev/aiflow-math-ink-1.0e
- 초기 bundle commit: `d9a79837ba7aa5b30c670643b329adb4adb58cd9`.
- 최종 distilled checkpoint 갱신 commit: `050ac77bb19aa487cf9b9598894e4d346f7b5938`.
- 카드에는 개념·기능·입출력과 개념도 3장을 포함했습니다.
