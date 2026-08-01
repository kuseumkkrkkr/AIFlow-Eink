# AIFlow Math Ink 0.6 — Federation 정식 감사

검증일: 2026-07-24  
상태: 연구 기준선 교정 완료, 제품 출시 불가

## 결론

기존 0.6 federation은 origin ID만 보면 누수가 없었지만, 정규화 trajectory와 writer 기준으로는 두 문제가 있었다.

- UJI Pen v1의 1,364개 trajectory가 UJI Pen v2에 전부 포함돼 있었다.
- HWRT 공식 train/test 사이에 동일 writer 93명이 존재했고, 학습 코드가 train 표본을 sample hash로 다시 나눠 writer-independent 조건을 만족하지 못했다.

따라서 기존 federation checkpoint의 정확도는 연구 이력으로만 남기며 제품 근거로 사용하지 않는다. 교정한 로더로 다시 학습하기 전에는 새 기준선으로 승격하지 않는다.

## 교정 내용

1. UJI v2를 `uji-pen-family`의 canonical source로 선택했다.
2. 이동·크기를 제거한 stroke별 trajectory SHA-256으로 label-aware 중복을 제거했다.
3. 같은 trajectory에 여러 label이 붙은 표본은 자동으로 정답을 발명하지 않고 전부 격리했다.
4. HWRT 공식 test는 제품 로더에서 완전히 제외했다.
5. 승인된 HWRT train writer 275명을 안정적 hash 순서로 train/validation/test 80/10/10에 가깝게 분리했다.
6. UJI v2 공식 test는 보존하고, 공식 train writer의 10%만 validation으로 격리했다.
7. 학습·평가 스크립트는 로더의 명시적 split을 사용하며 HWRT를 sample hash로 재분할하지 않는다.

## 수정 전후

| 감사 항목 | 수정 전 | 수정 후 |
|---|---:|---:|
| 유효 supervised source | 4 | 3 |
| UJI v1↔v2 trajectory overlap | 1,364 | 0 |
| HWRT train/test writer overlap | 93 | 0 |
| 전체 origin overlap | 0 | 0 |
| 전체 trajectory signature overlap | 1,364 | 0 |
| origin/writer/device split leakage | 93 | 0 |
| UJI v2 train/validation/test | 5,971/0/2,981 | 5,372/599/2,981 |
| HWRT train/validation/test | 잘못된 sample 재분할 | 27,387/261/552 |

UJI v1은 데이터가 사라진 것이 아니라 v2의 완전한 mirror라 독립 source와 sampler mass에서 제외됐다. HWRT는 지원 어휘와 승인 train만 남겨 28,200개를 사용한다.

## 출시 gate

현재 출처 registry와 checkpoint는 다음 이유로 정식 gate를 통과하지 못한다.

- 조사 출처: 38/200
- 승인된 배포 독립 그룹: 7/30
- 실제 교정 federation supervised 독립 source: 3
- 현재 checkpoint의 `training_source_ids`: 0개
- 교정 loader로 재학습한 seed 17·31·47 checkpoint: 없음

중복·split 누수 gate는 통과했지만 데이터 수와 checkpoint provenance gate는 실패다. 따라서 `product_validation=false`가 맞다.

## 다음 실험

1. 권리 검토가 끝난 24개 미전처리 source를 raster verifier·geometry pretrain 용도로 materialize한다.
2. supervised 378-label trajectory는 mirror가 아닌 독립 source를 우선 확보한다.
3. 교정 federation으로 seed 17·31·47을 GPU 재학습한다.
4. 각 checkpoint에 실제 `training_source_ids`, 독립 group, registry SHA-256을 기록한다.
5. writer/device/source-disjoint top-1 92%, top-5 99%와 raster label 보존 90%를 모두 통과한 경우에만 모바일 student를 다시 만든다.

기계 판독 가능한 결과는
`research/runs/math_ink_06_federation_release_audit_20260724/federation_audit.json`에 저장했다.

## 교정 후 seed-17 GPU 재학습

CUDA 12.1 PyTorch를 D: 격리 환경에 설치해 GTX 1650에서 실행했다. 기존 federated trainer가 `MathInk06Engine`의 CPU 기본값을 그대로 사용하던 결함을 발견해 `--device auto|cpu|cuda`를 추가하고, 명시적 CUDA 요청이 CPU로 조용히 fallback하지 않게 했다. Raster top-4의 batch 64는 4GB WDDM peak OOM이 발생해 batch 16으로 고정했다.

Federation 이전 fusion checkpoint에서 3 epoch를 학습하고, 선택 checkpoint를 7 epoch 추가 미세조정했다. 두 단계 모두 seed 17이며 서로 다른 seed ensemble이 아니다.

| Source | Online top-1 | Online top-5 | Raster top-1 | Raster top-5 |
|---|---:|---:|---:|---:|
| Pendigits | 90.2% | 99.2% | 36.6% | 64.6% |
| UJI Pen v2 | 69.6% | 93.0% | 37.8% | 69.8% |
| HWRT 교정 test | 83.2% | 99.0% | 75.6% | 94.4% |

전체 7,031개 test를 전수 평가하면 online top-1/top-5는 82.41/96.60%, visual-family top-1은 87.01%다. Exact 오류의 26.19%는 동일 visual family 안에서 발생했다. Writer accuracy p10은 56.08%이며 최저 writer는 0%다.

상위 오류는 `1→2` 72건, `7→1` 41건, `c→C` 36건, `s→S` 27건, `x→X` 26건이다. 지원량이 많은 숫자도 `1` top-1 73.44%로 낮고, UJI 계열 소문자 `s/x/o/c`는 top-1 12.77/14.58/14.89/15.56%에 그친다. 따라서 현재 병목은 다음 두 갈래다.

1. UJI unseen writer에서 대소문자와 `O/0/o` exact 의미를 고립 shape만으로 결정하려는 문제
2. raster가 virtual stroke로 변환된 뒤 online encoder 분포를 보존하지 못하는 문제

단순 seed 31·47 반복은 모든 seed가 개별 92/99/90 gate를 통과해야 한다는 조건을 만족시킬 가능성이 낮아 보류한다. 먼저 source별 오류 감사에 따라 크기·행 문맥 exact resolver와 독립 raster vectorizer 데이터를 보강한다.

새 checkpoint SHA-256은 `9190d9b997110657a831209b5041ea0d053b9051304b87560766a2e084516589`인 3-epoch checkpoint에서 시작했으며, stage-2 checkpoint와 오류 감사는
`research/runs/math_ink_06_federation_clean_seed17_stage2_20260724/`에 저장했다. 제품 검증은 계속 false다.

## Source cap 해제와 decoder shadow 실험

UJI의 낮은 일반화가 2,000개 source cap 때문인지 확인하기 위해 Pendigits 6,000개, UJI v2 전체 5,372개, HWRT 4,457개를 사용해 총 15,829개로 다시 학습했다. GTX 1650에서는 batch 32가 OOM 없이 동작했고, batch 64만 raster top-4 peak OOM이었다.

| Source | Online top-1 | Online top-5 | Raster top-1 | Raster top-5 |
|---|---:|---:|---:|---:|
| Pendigits | 93.2% | 99.4% | 42.8% | 72.6% |
| UJI Pen v2 | 71.8% | 93.8% | 41.0% | 72.4% |
| HWRT 교정 test | 84.0% | 99.0% | 74.4% | 94.6% |

증가량은 UJI online top-1 +2.2%p, raster top-1 +3.2%p에 그쳤다. 따라서 단순 source cap 해제로 UJI writer generalization 또는 378-class exact ambiguity를 해결할 수 없다는 결론이다.

같은 checkpoint에서 raster encoder·causal virtual decoder만 2 epoch 학습한 shadow 실험도 수행했다. Pendigits raster가 validation 45.67%에서 최고 63.33%까지 오르지만 HWRT는 78.16%에서 71.65%로 떨어져 모든 interpolation이 source holdout guard에서 탈락했다. 이 변경은 checkpoint로 채택하지 않았고 3번째 epoch는 중단했다.

다음 구현 우선순위는 (1) UJI의 case/O-0/cross family를 formula-relative size·neighbor context로 푸는 exact resolver, (2) source-aware virtual decoder 또는 더 다양한 paired raster↔stroke 데이터다. 이 두 작업 전에는 seed 31·47을 반복하지 않는다. 전체 Python 회귀는 367개를 통과했다.

상대크기 resolver를 현 UJI/HWRT test의 동일-writer isolated-glyph proxy에 적용한 결과는 UJI 51.09%→45.63%(-5.47%p), HWRT 53.57%→39.29%(-14.29%p)였다. 이 proxy는 실수식의 같은 행 baseline·상대크기가 아니라 writer의 서로 무관한 glyph를 anchor로 사용하므로 product evidence가 아니다. 다만 해로운 자동 뒤집힘을 실제로 확인했으므로 resolver는 현재 고립기호 0.6 반환에 연결하지 않고, 향후 P Formula row의 validated context가 있을 때만 activation한다.

## Online affine 불변성 pilot

UJI의 writer/device 차이가 단순 기울기·크기 분포 차이인지 분리하기 위해, source 특화 규칙 없이 online 19-feature 입력의 좌표·방향 채널에만 회전 ±10°, 등방 scale ±8%를 적용하는 seed-17 대조 학습을 시작했다. 시간·pen-up·pressure·missing metadata 채널은 바꾸지 않았고, teacher와 raster branch는 원본 입력을 유지했다.

1 epoch validation pilot은 macro online top-1 **88.47%**였다. 같은 초기 checkpoint의 무증강 epoch-1은 **88.01%**, epoch-2는 **88.57%**였다. UJI online top-1은 각각 **79.67%**, **78.67%**, **80.00%**였다. 즉 초기 +1.00%p 신호는 있으나 무증강 추가 epoch의 범위 안이며, UJI 단독 개선도 +1.00%p에 그쳤다.

동일 설정의 일관된 full-source run(`stage4_affine_consistent_epoch1`)을 최종 확인한 결과, 검증 macro online top-1은 **88.07%**로 기준선 **88.57%**보다 낮았다. UJI는 80.00%로 같았지만 HWRT online top-1 90.04%→88.89%, raster top-1 78.16%→76.63%가 되어 모든 full update와 일부 interpolation이 HWRT guard에서 탈락했다. 선택 checkpoint는 **epoch 0 / alpha 0.0**, test delta는 모든 source에서 0이었다. 따라서 affine 증강은 성능 후보가 아니라 **기각된 ablation**이다.

Windows launcher의 `Path/PATH` 환경 중복으로 처음 실행이 중단되었고, 이후 중복 GPU child가 생긴 것을 즉시 정리했다. 따라서 이 pilot은 최종 test checkpoint를 만들지 않았으며 **배포 후보에 포함하지 않는다**. 다음 반복은 epoch별 checkpoint를 남기는 단일 launcher에서, affine 강도 grid와 full writer/device-disjoint test를 함께 수행해야 한다.

이 요구를 위해 federated online trainer에 epoch-state 저장/재개 계약을 추가했다. epoch-state는 student·선택 best·anchor model state, Adam state, source-balanced sampler generator, Python/NumPy/PyTorch/CUDA RNG, 선택 history를 함께 저장한다. base checkpoint SHA-256, seed, source ID, train 수, split subset 상한이 완전히 같을 때만 `--resume-state`를 허용한다. 따라서 중단된 GPU 실행은 같은 data/난수 계약으로 다음 epoch부터 복구할 수 있으며, 다른 모델이나 다른 split을 이어붙이는 오류는 fail-closed로 거부한다.

### 기하 계약 교정 후 affine+raster-loss 재검증

2026-07-25에 위 affine helper가 `canvas_x/y`·direction만 변환하고 `shape`·curvature·bbox·aspect·speed를 원본 값으로 남기는 계약 오류를 발견했다. 변환 좌표와 파생 feature가 한 sample 안에서 모순되므로, 해당 중단 run은 평가 근거가 아니다. helper는 이제 변환한 canvas coordinate에서 모든 기하 파생 채널을 재생성하며, stroke start/progress·time delta·missing/source modality만 불변으로 보존한다. 관련 단위 9건과 전체 Python 회귀 **367 passed**를 통과했다.

교정 helper로 stage-3 full-source checkpoint에 rotation ±10°, scale ±8%, raster exact/family supervised loss 0.25/0.10을 넣어 15,829개 표본에서 CUDA 1 epoch를 재실행했다. validation macro online top-1은 **88.57%→88.07%**, HWRT online/raster top-1은 **90.04/78.16%→88.89/76.63%**, UJI online top-1은 **80.00%→80.00%**였다. alpha 0.25 interpolation만 HWRT tolerance를 지켰지만 macro 88.33%라 기준선을 넘지 못했다. trainer는 `selected_epoch=0`, `alpha=0.0`으로 원 checkpoint를 보존했고, `stage4_affine_consistent_epoch1_20260725`는 배포·shadow 후보 어느 쪽에도 채택하지 않는다. 다음 loop는 단순 affine/loss 증대가 아닌 독립 online P-source와 실제 formula-row context를 우선한다.

## Cross-source same-label alignment rejection and reproducibility correction

source cap 6,000 probe의 source-label balanced sampler를 측정하면 batch 안에서 같은 exact label·다른 source positive를 갖는 anchor가 1,587/17,372, 즉 **9.14%**에 불과했다. UJI↔HWRT는 66 labels, 세 source 공통은 digit 10 labels만 겹친다. 이에 batch 절반을 동일 label의 서로 다른 source pair로 고정하고, shared trajectory embedding에 supervised contrastive loss(weight 0.05, temperature 0.20)를 추가하는 1-epoch seed-17 shadow를 만들었다. 이 sampler/loss와 epoch resume 경로의 회귀는 15건을 통과했다.

첫 실행은 global cap으로 13,371개가 되어 비교 근거에서 제외했다. source cap CSV를 추가해 재실행했지만 Pendigits에 validation split이 없어 현재 trainer가 origin partition을 먼저 적용한다는 점을 확인했다. 실제 구성은 14,286개였고, 과거 stage-3 report의 15,829개 선택 set은 현 코드와 동일하게 재현할 수 없었다. 이는 accuracy 문제가 아니라 **checkpoint training-sample provenance 부족**이다. 이후 trainer는 source별 실제 count와 `(source, sample_id, origin_id)` 전체 SHA-256을 report/checkpoint/resume contract에 기록하며, 같은 표본 수이지만 다른 origin을 사용하는 재개를 fail-closed로 막는다.

14,286개 cross-source run은 validation macro online top-1을 88.57→89.23%, UJI validation을 80.00→82.00%로 높였지만, held-out UJI/HWRT online top-1은 각각 −0.2/−0.4%p, Pendigits raster top-1은 −3.2%p였다. 따라서 validation 상승을 채택 근거로 사용하지 않고 checkpoint는 shadow로 격리한다. stage-3 과거 selection set의 재현 fingerprint가 없으므로 이 실험도 baseline 대체 근거가 아니다. 앞으로의 정식 baseline은 현재 loader가 기록한 source count와 sample-origin fingerprint에서 새로 학습한 뒤에만 ablation 비교를 허용한다.

## Provenance baseline v1

현재 loader로 stage-2 checkpoint부터 seed-17 CUDA 3 epoch를 다시 학습해 첫 재현 가능한 federation baseline을 만들었다. 선택 training set은 15,829개이며 source count는 Pendigits 4,457, UJI v2 5,372, HWRT 6,000이고, `(source, sample_id, origin_id)` SHA-256은 `fc103edba331d3682875171b04942a2028232a401a0ca72cd0b1c52da5608c91`이다. epoch 3/alpha 1.0이 validation guard를 통과해 선택됐다.

| held-out source | online top-1 / top-5 | raster top-1 / top-5 |
|---|---:|---:|
| Pendigits | 95.0 / 99.4% | 31.6 / 60.6% |
| UJI Pen v2 | 73.2 / 94.6% | 39.4 / 69.8% |
| HWRT writer-disjoint | 84.2 / 99.0% | 74.4 / 94.6% |

전수 7,031개 error audit은 online top-1/top-5 **86.06/97.50%**, visual-family top-1 90.91%, writer p10 60.0%다. Exact 오류의 34.80%는 같은 visual family 내부이며 `1→2` 38건, `7→1` 32건, `c→C` 32건, `s→S` 30건, `x→X` 27건, `o→O` 17건이 상위다. 즉 기준선·provenance는 교정됐지만 UJI unseen writer의 case/O-0와 virtual-stroke raster 분포 병목은 남아 있다. 이 checkpoint는 release 92/99/90, 30 source, 3-seed 조건을 하나도 충족하지 못하므로 `product_validation=false`다.

## 외부 data federation 발견 갱신

공식 Mendeley Data v3의 `Ink and Identity`를 새 출처로 조사했다. 이 출처는 CC BY 4.0, 서로 다른 수집 방식의 사진 기반 문자 450개라는 점에서 raster geometry/pseudo-stroke 후보가 될 수 있다. 다만 현재는 label manifest, 작성자 또는 촬영 group, 원본 content hash와 기존 raster 중복 검사가 없으므로 `rights_review`에만 추가했다. 배포 checkpoint와 verifier 학습에는 넣지 않았으며, registry 조사 수는 **39/200**으로만 변했다.

온라인 분야에서는 UCI Character Trajectories(CC BY 4.0, 한 writer)와 BRUSH(170 writer이나 non-commercial)를 다시 대조했다. 전자는 이미 geometry-only rights review로 유지하고, 후자는 상용 P-track에서 제외한다. 따라서 이번 조사로 P supervised source 수나 product gate는 증가하지 않았다.

epoch-state의 parameter·optimizer·sampler 복원과 contract mismatch 거부를 포함해 전체 Python 회귀는 **369개**를 통과했다.

후속 조사에서는 ISGL Online/Offline Character Recognition Dataset을 확인했다. CC BY 4.0이며 64 writer의 tablet `x/y`, timestamp, pen-up/down online 기록을 포함하므로 image-only 후보보다 0.6 canonical tap 및 virtual-stroke pretrain에 더 적합하다. 다만 파일 구조, label mapping, writer group, origin hash 확인 전에는 `geometry_pretrain` rights review로만 등록했다. registry 조사 수는 **40/200**이고, P supervised source·승인 독립 그룹·product gate에는 변화가 없다.
