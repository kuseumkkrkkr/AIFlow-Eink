# CROHME 학습 경계 전수 감사

- 생성 시각(UTC): `2026-08-22T11:51:01.585749+00:00`
- 훈련 진입점: **19개**
- 코퍼스 준비 진입점: **2개**
- 실패: **0개**
- CROHME 오염 선택식: **0개**

## 훈련 진입점

| 파일 | 상태 | 비고 |
|---|---:|---|
| `train_augmented_owned_grouping_v2.py` | 통과 | 통과 |
| `train_candidate_validity_context_v1.py` | 통과 | 통과 |
| `train_character_classifier_v1.py` | 통과 | 통과 |
| `train_commercial_hwr_augmentation_v1.py` | 통과 | 통과 |
| `train_commercial_hwr_cleanroom_physics_v2.py` | 통과 | 통과 |
| `train_commercial_hwr_cleanroom_profiled_v3.py` | 통과 | 통과 |
| `train_commercial_latin_auxiliary_v1.py` | 통과 | 통과 |
| `train_context_decision_layer_v1.py` | 통과 | 통과 |
| `train_context_only_incremental_loop_v1.py` | 통과 | 통과 |
| `train_crohme_standard_context_loop_v1.py` | 통과 | 격리됨 |
| `train_distilled_math_context_v1.py` | 통과 | 통과 |
| `train_formula_layout_network_v1.py` | 통과 | 통과 |
| `train_independent_formula_context_v1.py` | 통과 | 통과 |
| `train_masked_context_reranker_v1.py` | 통과 | 통과 |
| `train_normalized_balanced_touch_calibrator09.py` | 통과 | 통과 |
| `train_owned_formula_context_v1.py` | 통과 | 통과 |
| `train_pairwise_shape_expert_v1.py` | 통과 | 통과 |
| `train_project_owned_grouping_v1.py` | 통과 | 통과 |
| `train_prompt_context_reranker_v1.py` | 통과 | 통과 |

## 코퍼스 준비

| 파일 | 상태 | 비고 |
|---|---:|---|
| `build_deepmind_formula_context_corpus_v1.py` | 통과 | CROHME 비참조 v2 |
| `build_candidate_validity_coverage_corpus_v1.py` | 통과 | CROHME 비참조 v2 |

## 결론

CROHME 경로는 일반 훈련 인수에서 실행 전에 거부된다. CROHME 점수나 후보 감사값은 모델, epoch, 임계값, 채택 게이트에 사용할 수 없고 별도 무경사 보고만 허용한다.
