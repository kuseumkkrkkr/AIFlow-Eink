# 48 Hz writer-LOO 개선 중검수

상태: **문자 분류 연구 후보 채택 / 상용 1.0 미채택**

## 적용한 최소 개선

- 출력부는 단일 372-class head를 유지했다.
- 사람별 속도와 출처별 timestamp 품질 차이를 제거하기 위해 제품 후보 입력은
  `uniform-time`으로 고정했다. 좌표 순서와 `stroke_start`는 보존한다.
- 외부 데이터만 학습한 8-epoch encoder는 동결하고, 직접 수집에서 실제 관측된
  출력 행만 조정했다.
- 직접 수집 평가는 3개 writer group의 leave-one-writer-out(LOO)으로만 수행했다.
  각 샘플을 평가한 head는 그 writer의 데이터를 보지 않았다.
- `=`는 합성하지 않고 직접 수집 실측 20건만 사용했다.
- 외부 rehearsal은 학습 split만 사용했고 고정 10% holdout은 보정 입력에서 제외했다.

## 보정 결과

| 평가 | 보정 전 Top-1 / Top-5 | writer-LOO 보정 후 |
|---|---:|---:|
| 직접 수집 211 glyph | 57.82% / 82.46% | **73.46% / 94.79%** |
| 괄호와 `=` 32 glyph | 37.50% / 37.50% | **81.25% / 90.63%** |
| 외부 10%, 동일 입력 충돌 제외 18,330건 | 77.23% / 97.35% | fold 최저 **77.28% / 97.36%** |

- 고정 외부 10%에는 uniform-time 변환 후 train과 바이트 단위로 같은 입력 14건이
  있었다. 위 외부 수치는 이 14건을 제외했다.
- 이 고정 10%는 앞선 반복 실험에서 이미 여러 번 확인했으므로 최종 미공개 테스트가
  아니라 기술 회귀 지표다. 이후 최종 성능 주장은 새 writer/session-disjoint 평가셋으로
  별도 확정해야 한다.
- writer-LOO head 재실행 결과 SHA-256은
  `1cefdef6b5a4e22d92ea853ac07dd699b39a1f8faf6ab47a465961da0f5f46eb`로 동일했다.
- LOO 게이트 통과 후 전체 writer로 만든 배포 후보 head는 직접 수집 정확도 산출에
  사용하지 않았다. 해당 211건은 head 보정 입력이므로 재점수하면 누수다.
- 전체-writer head의 외부 18,344건 성능은 77.33% / 97.36%다. 원래 고정 프로토콜
  18,723건에서 372-class 출력 밖 379건을 실패로 세면 75.76% / 95.39%다.

전체-writer head의 동일 입력 충돌 제외 출처별 Top-1 / Top-5:

| 출처 | 건수 | Top-1 / Top-5 |
|---|---:|---:|
| HWRT | 16,560 | 77.51% / 97.36% |
| UJI | 792 | 72.98% / 96.84% |
| ISGL | 700 | 70.86% / 97.14% |
| UCI | 278 | 96.76% / 100.00% |

## 실제 48 Hz 점진 재생

- 직접 수집: 원래 timestamp를 20.833 ms 간격으로 선형 보간했다.
- CROHME: timestamp가 없으므로 원래 타점 한 개를 48 Hz 한 tick으로 보았다.
- 모든 prefix는 그 시점까지 공개된 좌표만으로 bbox를 다시 계산했다. 완성 glyph의
  미래 bbox는 사용하지 않았다.
- 각 prefix를 128×5로 변환해 실제 모델 추론을 수행했다. HTML 재생도 48 Hz로 맞췄다.

| 평가 | 최종 Top-1 / Top-5 | 마지막 20% Top-1 / Top-5 | 전체 prefix Top-1 / Top-5 |
|---|---:|---:|---:|
| 직접 수집 211 glyph, writer-LOO | **73.46% / 94.31%** | 69.60% / 88.98% | 20.32% / 31.11% |
| CROHME 지원 9,535 truth group | **69.50% / 93.20%** | 60.33% / 85.73% | 20.08% / 33.49% |

- 직접 수집 최종 Top-5가 고정 완성점 94.79%와 48 Hz 94.31%로 1건 다른 것은
  실제 48 Hz 보간이 완성 시퀀스의 샘플 밀도를 바꾸기 때문이다.
- 직접 수집 truth grouping 수식 47건의 전 토큰 정확도는 Top-1 16/47(34.04%),
  Top-5 38/47(80.85%)다. 고정 canonical 10건 중 ownership이 있는 6건은
  1/6(16.67%), 5/6(83.33%)다.
- CROHME 전체 985건에서 미지원 토큰과 불완전 partition을 실패로 포함한
  truth-group 전 토큰 정확도는 Top-1 121/985(12.28%), Top-5 493/985(50.05%)다.
  완전 지원 751건만 보면 16.11% / 65.65%다. 이는 truth grouping을 제공한
  수치이며 end-to-end 수식 정확도가 아니다.

## 남은 문제와 결정

- 직접 수집 `=` 최종은 14/20 Top-1, 17/20 Top-5로 회복했다.
- 직접 수집 `1`은 3/25 Top-1, 22/25 Top-5다. 오답은 주로 `|` 11건과 `/` 7건이다.
  외부 학습의 `1`은 319건으로 데이터 부족이 아니며, 짧은 선형 glyph의 box-local
  시각적 동형 문제다.
- 직접 수집에서 동형군 exact/family Top-1은 세로선 12%/60%, 교차선 66.67%/100%,
  원형 56.25%/93.75%다. CROHME에서도 `sum/Sigma` exact/family Top-1이
  22.97%/78.38%다.
- 따라서 동형 클래스를 병합하거나 추가 epoch로 강제하지 않는다. 372-class Top-k를
  유지하고 수식 위치·이웃 토큰 문맥에서 최종 후보를 선택해야 한다.
- CROHME는 비상업 평가 전용이다. 제품 학습에는 사용하지 않는다.
- 상용 1.0 확정에는 project-owned grouping, spatial relation, formula decoder와 더 많은
  writer-disjoint 수집이 필요하다.

## 재현 명령

```powershell
python scripts/calibrate_project_punctuation_v1.py --canonical-root datasets/normalized/v1 --cache-dir artifacts/unified_head_20260813/unified_math_8ep_full/cache --base-checkpoint artifacts/time_normalization_20260813/uniform_time_unified_math_8ep_full/classifier_checkpoint.pt --output artifacts/unified_head_20260814/uniform_time_writer_loo_calibration --device cuda --input-mode uniform-time --leave-one-writer-out
python scripts/calibrate_project_punctuation_v1.py --canonical-root datasets/normalized/v1 --cache-dir artifacts/unified_head_20260813/unified_math_8ep_full/cache --base-checkpoint artifacts/time_normalization_20260813/uniform_time_unified_math_8ep_full/classifier_checkpoint.pt --output artifacts/unified_head_20260814/uniform_time_final_all_writers --device cuda --input-mode uniform-time --finalize-all-writers
python scripts/evaluate_48hz_prefix_v1.py --scope direct --output artifacts/prefix_48hz_20260814_direct --device cuda
python scripts/evaluate_48hz_prefix_v1.py --scope crohme --output artifacts/prefix_48hz_20260814_crohme --device cuda
```

로컬 산출물:

- `artifacts/unified_head_20260814/uniform_time_writer_loo_calibration/`
- `artifacts/unified_head_20260814/uniform_time_final_all_writers/`
- `artifacts/prefix_48hz_20260814_direct/`
- `artifacts/prefix_48hz_20260814_crohme/`
