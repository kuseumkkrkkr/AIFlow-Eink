# 계층형 작가 모델 v7: raw writer evidence

## 결론

- v6의 합성 궤적 중심 catalog에서 벗어나 legacy 프로젝트 원시 필기 96수식을 raw writer evidence로 분리했다.
- raw writer catalog는 7명, 96수식, 393 glyph group, 28클래스다. writer ID, session, formula/sample ID, exact stroke indices, 원시 획 hash와 원천 파일 hash를 보존한다.
- raw에 없는 클래스만 external clean-room synthetic 4,096행을 fallback으로 사용한다.
- 기존 fresh 53은 이미 사용된 세트이므로 `known_writer_disjoint_replay_diagnostic`일 뿐이며 상용 승격·튜닝 근거가 아니다.
- 학습, 제품 runtime, HWR/context checkpoint, CROHME/MathWriting 사용은 없다.

## frozen partition 처리

| 항목 | 결과 |
|---|---:|
| frozen formulas_valid | 159행 |
| fresh sample ID | 53개 |
| accepted legacy ownership/formula | 96개 |
| ownership 없는 legacy valid formula | 10개 제외 |
| build에서 파싱한 fresh formula/ownership 본문 | 0개 |

build는 `ownership_fresh_acceptance.jsonl`에서 sample ID만 먼저 고정한다. ownership 96개를 확정한 뒤 formula JSONL을 순회하며 fresh 53개와 ownership 없는 10개를 `json.loads` 전에 건너뛴다.

## raw writer catalog

| writer | 수식 | glyph | 클래스 | session |
|---|---:|---:|---:|---:|
| writer_001 | 10 | 32 | 19 | 1 |
| writer_002 | 23 | 100 | 24 | 2 |
| writer_003 | 20 | 88 | 23 | 1 |
| writer_004 | 10 | 35 | 21 | 1 |
| writer_006 | 20 | 80 | 22 | 1 |
| writer_007 | 8 | 35 | 15 | 1 |
| writer_008 | 5 | 23 | 13 | 1 |

- exact ownership partition: 모든 원본 획을 중복·누락 없이 한 glyph group에 배정.
- box-local 128×5 tensor: 원본 획순·stroke start·상대시간을 보존해 생성.
- 지원 raw 392글자 동결 HWR 진단: Top-1 91.33%, Top-5 98.21%.
- `t` 1건은 raw evidence로 저장했지만 372-class 출력 어휘에 없으므로 label index `-1`, 출력 승격 `false`다.

## coherent writer policy

네 생성 작가에 고정 legacy 부모 작가를 각각 하나씩 지정했다.

| 생성 작가 | 고정 raw 부모 |
|---|---|
| compact upright | writer_001 |
| slanted round | writer_002 |
| wide relaxed | writer_003 |
| narrow quick | writer_004 |

- 각 생성 작가는 고정 부모가 실제 쓴 클래스만 raw primary로 사용한다.
- 미지원 클래스만 external synthetic fallback을 사용한다.
- 생성 수식 12개, 72글리프 중 raw 61개·external fallback 11개다.
- 생성 작가별 raw 부모 writer mixing 최대치는 1이다.
- synthetic style을 raw glyph에 적용한 augmentation shadow이며 실제 부모 작가를 그대로 복제한 모델이라고 주장하지 않는다.

## known writer-disjoint replay diagnostic

- 범위: writer_010·writer_012, 53수식, 186 glyph, 23클래스.
- legacy 7작가와 writer-disjoint인 것은 확인했다.
- 문자 Top-1: 85.48%
- 문자 Top-5: 95.16%
- 수식 Top-1 exact: 60.38%
- 수식 Top-5 oracle: 86.79%

이 값은 기존 fresh-context acceptance에서 이미 사용된 세트의 재생 진단이다. 선택·튜닝·promotion evidence는 모두 `false`다. 상용 승격에는 이후 새로 수집하고 사전에 동결한 writer/formula acceptance set이 필요하다.

## 고정 gate

- `legacy_raw_only_build=true`
- `fresh_row_bodies_parsed_during_build=false`
- `raw_primary_for_supported_classes=true`
- `external_synthetic_fallback_only=true`
- `coherent_fixed_raw_parent_writer=true`
- `t_catalog_evidence_only=true`
- `t_output_promotion=false`
- `known_replay_used_for_selection=false`
- `promotion_ready=false`
- `training_performed=false`
- `product_runtime_changed=false`
- `checkpoint_changed=false`

## 산출물

- builder: `scripts/hierarchical_writer_model_v7.py`
- replay evaluator: `scripts/evaluate_known_writer_disjoint_replay_v7.py`
- build report: `artifacts/hierarchical_writer_model_v7_20260823_r1/report.json`
- raw tensors: `artifacts/hierarchical_writer_model_v7_20260823_r1/raw_legacy_writer_catalog.npz`
- raw provenance: `artifacts/hierarchical_writer_model_v7_20260823_r1/raw_legacy_writer_catalog.metadata.jsonl`
- writer profiles: `artifacts/hierarchical_writer_model_v7_20260823_r1/raw_writer_profiles.json`
- replay receipt: `artifacts/hierarchical_writer_model_v7_20260823_r1/known_writer_disjoint_replay_diagnostic/report.json`
