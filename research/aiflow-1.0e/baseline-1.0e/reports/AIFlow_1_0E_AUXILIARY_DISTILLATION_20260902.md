# AIFlow 1.0e 보조 데이터 증류 결과 (2026-09-02)

## 결론

- 판정: **shadow 후보 유지**
- 원 평가셋: 7 writers / 95 formulas / 387 glyphs (변경 없음)
- 추가 학습셋: 2 writers / 46 formulas / 179 glyphs
- 최종 nested guard: Top-1 **317 → 321**, formula exact **49 → 51**, row regression **0**
- Top-5: **374/387 유지**, 후보 계약 위반 **0**
- 제품 runtime, Hugging Face 공개본: **변경 없음**

## 데이터 경계

- 출처: project-owned ordered online ink와 독립 ownership review가 있는 기존 private collector bank
- 용도: 과거에 이미 평가에 소비된 bank이므로 **fresh acceptance로 재주장하지 않고 학습 전용**으로만 사용
- 원 평가와 겹치는 동일 수식 표기 7 formulas를 완전 제외
- 최종 경계 검사: record ID, formula ID, exact formula display, writer 모두 overlap 0
- 채널: 좌표/순서/시간/압력 계열을 유지한 기존 5-channel tensor 계약

고정 manifest:

- `artifacts/auxiliary_distillation_data_20260902_r1/manifest.json`
- auxiliary candidate SHA-256: `5cfc6a5b9dc485a8957192da0b6cc43614fafc49cf126d6d1c459ae224f732e6`
- combined raw SHA-256: `fdfb7391dc8509f4c142504c08e4e0bfdb502775e306a879dc2057dbab5c29c2`

## Teacher

3개 frozen offline teacher의 특징을 141 formulas에 대해 다시 만들고, 9 writers 전체에서 writer-LOO soft target을 생성했다.

| 모델 | 결합 Top-1 | 결합 formula exact | 회귀 |
|---|---:|---:|---:|
| TexTeller | 506/566 | 96/141 | 8 |
| UniMERNet tiny | 497/566 | 90/141 | 15 |
| Microsoft TrOCR-small | 499/566 | 91/141 | 10 |
| Uniform probability ensemble | **509/566** | **98/141** | **7** |

원 7-writer 평가 부분에서 uniform teacher는 317→351, 49→67이었으나 row regression 3건이므로 runtime 후보가 아니다.

## Raster-free student

각 outer fold에서 held writer는 학습에서 제외했다. 학습에는 나머지 원 writer 6명과 disjoint auxiliary writer 2명만 넣었다. 외부 teacher는 soft target으로만 쓰였고 student runtime 입력은 다음으로 제한했다.

- 기존 5-channel ordered online-ink embedding
- frozen HWR Top-k 확률
- geometry/context
- candidate token identity

Raw student 결과:

| 지표 | 기준 | student | 변화 |
|---|---:|---:|---:|
| Top-1 | 317/387 | 347/387 | +30 |
| Formula exact | 49/95 | 64/95 | +15 |
| Row improvement / regression | 0 / 0 | 38 / 8 | 회귀 gate 실패 |

## Nested guard

임계값은 각 held writer가 아닌 다른 6 writers에서만 선택했다. 선택 규칙은 training-writer regression 0을 먼저 강제한 뒤 formula exact, Top-1, 변경 수 순으로 결정했다.

| 지표 | 기준 | guarded | 변화 |
|---|---:|---:|---:|
| Top-1 | 317/387 (81.91%) | **321/387 (82.95%)** | **+4** |
| Top-5 | 374/387 (96.64%) | **374/387 (96.64%)** | 0 |
| Formula exact | 49/95 (51.58%) | **51/95 (53.68%)** | **+2** |
| Row improvement | 0 | **4** | +4 |
| Row regression | 0 | **0** | gate 통과 |
| Candidate contract violation | 0 | **0** | gate 통과 |

Writer별 improvement/regression: `2/0, 0/0, 0/0, 1/0, 0/0, 1/0, 0/0`.

## 재현성 및 범위

- 동일 seed 재실행 raw prediction SHA-256 일치: `2f24e67f86c1381822fc29f943b955c0fb88edad77675907b429b04b43b12211`
- 동일 seed 재실행 guarded prediction SHA-256 일치: `f88a6a661d46a96a5048100b264176f628fa7e6fb33183eeb69ea0fc621e857d`
- 외부 teacher/raster의 제품 runtime 사용: 없음
- token 생성, stroke regrouping, row 삭제: 없음
- 이번 결과는 기존 95-formula bank의 shadow 검증이다. 새로운 상업 acceptance나 production 채택 근거로 확장하지 않는다.

## 산출물

- 데이터 고정: `artifacts/auxiliary_distillation_data_20260902_r1/`
- teacher 특징: `artifacts/offline_teacher_{texteller,unimernet_tiny,trocr_small}_20260902_aux_r1/`
- teacher ensemble: `artifacts/offline_teacher_ensemble_20260902_aux_r1/`
- student: `artifacts/auxiliary_nested_distill_20260902_r1/`
- 최종 guard: `artifacts/auxiliary_nested_distill_guard_20260902_r1/`
- 반복 검증: `artifacts/auxiliary_nested_distill_20260902_r2_repeat/`, `artifacts/auxiliary_nested_distill_guard_20260902_r2_repeat/`
