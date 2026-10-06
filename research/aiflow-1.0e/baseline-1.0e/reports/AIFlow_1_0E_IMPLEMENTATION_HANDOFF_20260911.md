# AIFlow 1.0e 구현 인수인계

## 실행 경계

- 기존 dirty 변경 및 과거 보고서·가중치를 보존. 새 실험은 `artifacts/accuracy_upgrade_20260911_*`에만 기록.
- 공개 모델 교체, 제품 기본 활성화, 업로드, 유료 GPU, 전역 패키지 변경 없음.
- 원본 수식 141개·문자 566개를 원본 획 소유권과 다시 연결. 기존 평가 7명·95수식·387문자와 보조 2명·46수식·179문자를 구분.
- 모든 결과는 소비된 개발 자료의 연구 결과. 신규 봉인 평가로 표시하지 않음.
- 감독학습 계보 검증 범위: 이번 project-owned fit/selection ID 및 실제 산출물·부모 SHA. 외부 사전학습 전체 원자료의 writer 목록이나 수식 중복을 새로 감사한 것은 아님. 외부 기반 HWR는 기존 학습 보고서와 승인 데이터 카탈로그를 근거로 구분.

## 구현 구성

| 기능 | 진입점 또는 모듈 |
|---|---|
| 원본 계보 연결·전체 logits·고정 어휘·수식 batch | `scripts/accuracy_upgrade_data_10e.py` |
| 위치 복원·정규화·KL 계약 | `scripts/accuracy_upgrade_contract_v1.py` |
| 실제 fit/selection 부모 그래프·SHA 검증 | `scripts/accuracy_lineage_10e.py` |
| 2층·hidden128·4head residual 문맥 모델 | `scripts/formula_context_ranker_10e.py` |
| 수식 평균 CE/KD·coverage·padding 검증 | `scripts/formula_candidate_loss_10e.py` |
| MLP/문맥 학습·early stopping·고정 epoch refit | `scripts/accuracy_student_training_10e.py` |
| fold 내부 teacher head·writer crossfit target | `scripts/accuracy_teacher_targets_10e.py` |
| 원본 exact-cover·partition32·공동 선택·멱등성 | `scripts/accuracy_joint_runtime_10e.py` |
| 전체 분모 평가·OOF 위험 guard·paired bootstrap | `scripts/accuracy_evaluation_10e.py` |
| 통합 학습/평가 | `scripts/run_accuracy_upgrade_10e.py` |
| HWR head/마지막 block·외부 replay 1:1 | `scripts/run_accuracy_hwr_10e.py` |
| 결과·계보 독립 재검증 | `scripts/verify_accuracy_experiment_10e.py` |
| 실제 teacher/network 차단 프로세스 검사 | `scripts/probe_accuracy_offline_10e.py` |
| 후보 누락 상한·수정 민감도 | `scripts/analyze_accuracy_candidates_10e.py` |
| 지연 획·시간 결측·이웃 겹침 특징 | `scripts/accuracy_temporal_context_10e.py` |
| 선택 기호·구조의 안전한 LaTeX 출력 | `scripts/accuracy_latex_serializer_10e.py` |
| 고정 HWR의 joint·K 비교 | `scripts/evaluate_hwr_joint_10e.py` |
| 동일 head logits의 K별 후보 상한 | `scripts/analyze_hwr_candidate_ranges_10e.py` |
| 승인된 시간 채널·재현 seed 설정 | `scripts/run_accuracy_fixed_variant_10e.py` |
| 고정 epoch·grouping의 HWR seed 재현 | `scripts/run_accuracy_hwr_seed_check_10e.py` |
| 과거 코드 계보의 불변 사본 연결 | `scripts/archive_accuracy_code_lineage_10e.py` |
| 전체 실행·검증 SHA 색인 | `scripts/summarize_accuracy_experiments_10e.py` |

## 재현 명령

기준 디렉터리: `D:/AIFlow-Workspace/Projects/Aiflow/aiflow-math-ink-1.0`.
Python: `C:/Python311/python.exe -s -B`. 출력은 반드시 존재하지 않는 D: 폴더를 사용.

```powershell
$env:LOKY_MAX_CPU_COUNT='4'
& 'C:\Python311\python.exe' -s -B scripts/run_accuracy_upgrade_10e.py --config configs/accuracy_upgrade_10e_20260911.json --output artifacts/NEW_EXPERIMENT --stage student --teacher none --architecture context --device cuda
& 'C:\Python311\python.exe' -s -B scripts/verify_accuracy_experiment_10e.py --experiment artifacts/NEW_EXPERIMENT --output artifacts/NEW_EXPERIMENT/verification.json
& 'C:\Python311\python.exe' -s -B scripts/probe_accuracy_offline_10e.py --experiment artifacts/NEW_EXPERIMENT --output artifacts/NEW_EXPERIMENT/offline_probe.json
```

- `--features legacy21/formula28/formula40`, `--geometry legacy/restored`, `--loss-mode legacy_kl/legacy/standard`는 별도 ablation 계약. `formula40`은 `formula28`의 앞 28차원을 유지하고 겹침·획 순서·시간 간격·결측 마스크 12개를 뒤에 붙임. restored만 허용하며 기존 체크포인트에 조용히 적용하지 않음.
- `--top-k 5/10/20`은 고정 전체 372-class logits에서 잘라냄. 평가 후보의 합집합으로 어휘를 만들지 않음.
- `--stage joint --window 6/8 --neighbors 4/8 --structure-weight 0.5/1/2`로 원본 joint 진단. 계획의 비교는 `(6,4)`와 `(8,8)` 두 설정만 사용.
- `run_accuracy_hwr_10e.py --scope head/last-block`은 동일 기본 가중치에서 inner 3-fold의 실제 E2E로 epoch를 고르고 사전 고정 중앙값으로 outer refit. 기존 자체 자료/승인 외부 replay 1:1을 사용하는 **역사 자료 진단**이며, 계획의 신규 writer 혼합 실험을 대신하지 않음.
- 기준 시드는 20260910. 학습 중 코드/입력/feature cache를 수정하지 않음.
- teacher 특징은 모델별로 한 번씩 추출. TexTeller/TrOCR CUDA, UniMERNet CPU 분리 환경 사용. GPU teacher 추출과 student GPU 학습은 겹치지 않음.

## 권리·데이터에 따른 필수 차단

- `configs/accuracy_teacher_rights_20260911.json`: TrOCR-small 및 이를 포함한 3종 앙상블은 권리 확인 전 추가 증류 보류. TexTeller·UniMERNet의 Apache-2.0 표시와 전체 배포 심사를 혼동하지 않음.
- 구조 관계 정답이 명시된 수식은 2개뿐. 9-class 관계 head의 학습·채택은 `blocked_data`.
- 신규 writer/formula/device-disjoint 봉인 자료 없음. 단일 제품 후보 승격, 통계적 독립 검증, 실제 기기 채택은 `blocked_data`.
- 복잡한 수식의 빈 relation 배열은 완전한 구조 주석으로 간주하지 않음. 구조 지표는 주석 지원 수와 함께 보수적 진단으로 보고.
- 직렬 TexTeller decoder 경로는 보류 유지. 한 수식/10수식 완전 과적합 조건을 건너뛰어 주력 경로로 채택하지 않음.
- 제품 패키지는 승격 기준을 모두 통과한 경우에만 생성. 실험 checkpoint를 제품 가중치로 조용히 복사하지 않음.

## 남은 작업의 완료 판정

현재 가능한 예정 비교와 seed 재현은 완료. 결과는 `reports/AIFlow_1_0E_IMPLEMENTATION_RESULTS_20260911.md`, 원장은 `artifacts/accuracy_upgrade_20260911_final/experiment_index.json`에 기록했다. 다음 학습 단계는 권리·주석이 확인된 새 자료가 확보된 뒤 별도 승인 범위에서 진행한다. 자동 반복 작업은 남겨 두지 않았다.

- 코드를 추가했다는 사실과 실제 7-fold 비교 완료를 분리하여 최종 실행 원장에 기록.
- 문자 학습 비교는 전체 95 raw 예측·387 glyph 예측·계보 검증을 요구. 원본 획 전용 joint 비교는 95 raw 예측을 검증하고 별도 glyph 추론이 없음을 표시. 종료 코드 0만으로 성공 처리하지 않음.
- 오수정 0 조건은 observed 개발 자료에서만 설명하며 미래 무오류 보장으로 표현하지 않음.
- 독립 자료 없는 bootstrap은 개발 자료 진단. 승격 유의성 근거로 재사용하지 않음.
- 예정 비교 외 새 seed/epoch/threshold 탐색을 점수 정체 때문에 임의 추가하지 않음.
- 시간·겹침 추가 특징은 독립 모듈 검사와 실제 feature 인터페이스 통합/학습 결과를 구분하여 원장에 남김.
