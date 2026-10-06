# AIFlow Math Ink 0.9: 구조 분리형 온라인 수식 필기 인식과 소규모 사용자 데이터 보정

> 연구 논문 초안 · 2026-08-08 · 제품 성능 주장 아님

저자

1. Chae Won Lee (제1저자)
2. Seok Jun Kim (제2저자)

## 초록

본 연구는 온라인 수식 필기 인식을 형태/그리딩, 문자 후보 판정, 보수적 결정의 세 단계로 분리하고, 원본 stroke 순서와 raw fallback을 보존하는 AIFlow Math Ink 구조를 제안한다. 0.9 단계에서는 glyph bbox 중심 정규화, 종횡비 보존 letterbox, class/contributor 균형 sampling을 적용한 additive calibrator를 기존 378-label trajectory encoder에 결합하고 `=` label을 추가하였다. 공개 동의가 확인된 119개 수식을 비식별 stroke 데이터로 구성하고, ownership이 검수된 47개 수식·211개 문자로 학습했다. 49개 수식·171개 문자 replay에서 seed31은 문자 Top-1을 50.29%에서 87.72%, 수식 exact를 26.53%에서 67.35%로 높였다. CROHME2019 CC BY-NC valid 985식에서는 truth-group 문자 Top-1을 56.96%에서 74.11%로 높였다. 그러나 replay의 writer-disjoint를 증명할 수 없고 guard threshold도 독립 split에서 선택되지 않았으므로 연구 후보로만 채택한다.

## 1. 연구 목적

목표는 자동채점이나 정답 저장이 아니라 사용자가 순차 입력한 수학 필기를 가능한 한 안정적으로 문자와 수식 형태로 복원하는 것이다. 완전한 LaTeX 일치만을 성공으로 보지 않고 `3X4`, `3x4`, `3*4`처럼 의미상 동형인 출력도 후속 계층이 처리할 수 있도록 후보와 원본을 보존한다.

## 2. 관련 연구

CROHME는 온라인 필기 수식 인식의 symbol segmentation, recognition, structural analysis를 함께 평가하는 대표 벤치마크다 [1]. SCAN은 stroke sequence와 spatial 구조를 결합해 온라인 수식을 인식한다 [2]. MathWriting은 대규모 온라인 수식 데이터와 공식 train/validation/test 분할을 제공한다 [3]. 본 연구는 대형 end-to-end decoder 자체보다, 실시간 입력에서 계층별 책임과 실패 복구 계약을 명확히 하는 데 초점을 둔다.

## 3. 시스템 구조

### 3.1 형태/그리딩

stroke의 시간 순서, pen-up 경계, 기하 관계를 보존해 문자 후보 group을 만든다. 이 계층은 문자를 확정하지 않고 `□ op □ = □` 같은 레이아웃 가능성과 stroke ownership 후보를 전달한다. 경계가 불확실하면 단일 partition을 강제하지 않고 후보 또는 raw stroke를 유지한다.

### 3.2 문자 판정

각 후보 group을 bbox crop과 종횡비 보존 letterbox로 정규화한 뒤 trajectory encoder에서 Top-k 문자 후보를 얻는다. 0.9 calibrator는 base encoder를 고정하고, 관측된 label의 logit만 additive correction한다. `=`는 379번째 출력으로 추가했다.

### 3.3 결정

결정기는 문자 Top-1을 무조건 덮어쓰지 않는다. 수식 위치와 인접 후보가 충분한 증거를 제공할 때만 동형 문자를 보정하고, 확신이 없으면 classifier 후보 또는 raw 결과로 abstain한다. 이는 과거 단일 문자에서 올바른 `g`, `)`를 잘못 바꾼 사례를 막기 위한 계약이다.

## 4. 데이터와 윤리

전체 수집 기록은 119개이며 valid 110, pending 4, reject 5다. 0.9 학습에는 검수된 47개 수식, 211개 문자, 25개 관측 label, 3 contributors를 사용했다. replay는 기존 수동 ownership 49개 수식, 171개 문자다. 학습과 replay 사이 exact formula 및 prompt 중복은 없지만 replay에 contributor ID가 없어 writer-disjoint split은 증명할 수 없다.

데이터 관리자는 2026-08-08 참여자들의 공개 배포 동의가 완료됐음을 확인했다. 공개본은 stroke·point 순서를 보존하되 원본 contributor/session/prompt ID, 기록 시각, 이미지 URL, IP·원본 경로를 제거하고 dataset-local writer/session 별칭과 수식 상대 시간만 제공한다. valid 110건만 기본 학습 대상으로 사용하며 pending 4건과 reject 5건은 분리한다.

## 5. 실험 설정

base는 AIFlow Math Ink 0.6 seed17의 378-label encoder와 online adapter다. 입력은 glyph-local 정규화 후 수식 상대 위치 채널을 고정해 문자 형태 경로에서 제거했다. additive head만 20 epoch 학습했고 class와 contributor 조합의 역빈도 sampling을 사용했다. seed 17, 31, 47을 비교했으며 모든 평가는 동일한 정규화 baseline과 같은 replay에서 수행했다.

## 6. 결과

| 지표 | baseline | seed17 | seed31 | seed47 | 3-seed 평균 |
|---|---:|---:|---:|---:|---:|
| 문자 Top-1 | 50.29% | 85.96% | 87.72% | 85.96% | 86.55% |
| 문자 Top-5 | 70.18% | 98.25% | 98.83% | 98.25% | 98.44% |
| 수식 exact | 26.53% | 67.35% | 67.35% | 65.31% | 66.67% |
| 수식 Top-5 oracle | 42.86% | 93.88% | 95.92% | 93.88% | 94.56% |
| `=` Top-1 | 0.00% | 100% | 100% | 100% | 100% |

seed31은 171개 문자 중 67개를 개선하고 3개를 악화시켰다. `f`는 6/6에서 5/6, `g`는 4/5에서 2/5로 하락했다. `/`와 `\sqrt{}`는 각각 단 하나의 replay 표본에서 여전히 실패했다.

학습 support가 3개 미만인 label에서 baseline Top-1 confidence가 0.95 이상이면 baseline을 보존하는 guard는 seed31의 집계 정확도를 유지하면서 개선/회귀를 67/3에서 66/2로 바꿨다. 3개 seed 모두에서 회귀는 3건에서 2건으로 줄었지만 같은 replay에서 확인했으므로 독립적인 성능 향상으로 확정하지 않는다.

### 6.1 CROHME2019 비상업 검증

Kaggle `ntcuong2103/crohme2019`의 CC BY-NC valid split에서 parse 가능한 985식을 평가했다. CROHME는 학습에 포함하지 않았고 공식 test 1,199식도 사용하지 않았다.

| 지표 | baseline | calibrator | guarded |
|---|---:|---:|---:|
| truth-group 문자 Top-1 | 56.96% | 74.11% | 74.18% |
| truth-group 문자 Top-5 | 79.97% | 92.45% | 92.28% |
| end-to-end group-token Top-1 recall | 48.20% | 62.17% | 62.22% |
| fully-covered 수식 exact | 7.95% | 11.98% | 12.10% |

공개 가능한 0.6 geometry gridder의 partition exact는 40.81%, exact group recall은 82.48%였다. lattice group ceiling은 99.86%여서 후보 생성보다 ownership 선택이 더 큰 그리딩 병목으로 남는다.

## 7. 논의

결과는 정규화와 제한적 additive calibration이 소규모 실제 필기 분포 적응에 효과가 있을 가능성을 보여준다. 하지만 표본 수가 작고 writer-disjoint가 아니므로 일반 사용자 성능이나 상용 수준을 증명하지 않는다. 높은 Top-5는 후보 보존 전략의 유용성을 뒷받침하지만, 수식 exact와 동일하지 않다. 다음 검증은 contributor 식별자가 있는 writer-disjoint split, `f/g` 및 희소 기호 보강, 그리딩 ownership 평가를 동시에 보고해야 한다.

CROHME2019 결과는 비상업 연구 평가이며 제품 성능 근거로 사용할 수 없다. 후속 비상업 gridder artifact는 상용 배포 정리 과정에서 제거되어 이번 실행에서는 공개 가능한 0.6 geometry gridder를 사용했다. MathWriting은 이번 실험에 포함하지 않았다.

## 8. 결론

AIFlow Math Ink 0.9는 단계별 책임, Top-k 보존, raw fallback이라는 구조 위에 실제 필기 분포용 경량 calibrator를 추가해 내부 replay를 개선했다. 현재 근거로는 연구 후보 채택이 타당하지만 제품 채택은 이르다. 후속 연구의 통과 조건은 writer-disjoint 재현, 희소 문자 회귀 통제, 동일 split에서 그리딩·문자·결정·수식 지표의 동시 개선이다.

## 참고문헌

1. H. Mouchère et al., “ICFHR 2014 Competition on Recognition of On-line Handwritten Mathematical Expressions (CROHME 2014),” ICFHR 2014. DOI: 10.1109/ICFHR.2014.42. https://www.cs.rit.edu/~rlaz/files/Crohme2014FinalVersion.pdf
2. J. Wang, J. Du, and J. Zhang, “Stroke Constrained Attention Network for Online Handwritten Mathematical Expression Recognition,” 2020. https://arxiv.org/abs/2002.08670
3. P. Gervais, A. Fadeeva, and A. Maksai, “MathWriting: A Dataset For Handwritten Mathematical Expression Recognition,” 2024. https://arxiv.org/abs/2404.10690
4. ntcuong2103, “CROHME2019,” Kaggle dataset mirror, CC BY-NC 4.0. https://www.kaggle.com/datasets/ntcuong2103/crohme2019

## 재현성과 공개 상태

공개 데이터와 ownership은 Hugging Face dataset에, 원시 replay 지표는 `reports/public_retrain_seed31_metrics.json`, CROHME 결과는 `reports/crohme2019_valid.json`, 학습 진입점은 `scripts/train_normalized_balanced_touch_calibrator09.py`에 있다. 공개 데이터만으로 calibrator 학습은 재현할 수 있지만 기존 필기 replay 원본은 별도 내부 검증 자료이므로 동일 replay 평가는 완전 공개 재현이 아니다. 공개 체크포인트는 base 의존성을 명시한 연구 artifact다.
