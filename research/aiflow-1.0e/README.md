# AIFlow 1.0e 누적 연구보고서

기준일: **2026-10-06** · 최근 완료 실험: **V35 teacher-correct feature paired training**

## 연구 결론과 현재 상태

- 원본 stroke의 문자 후보 생성, stroke grouping, 수식 문맥·배치 판단을 분리하고, 후보 보존과 기존 정답 회귀 여부를 함께 감사했다.
- 최근 재학습은 내부 문자 진단을 개선했으나 프로젝트 소유 수식 진단에서는 기존 canonical 체크포인트를 넘지 못했다. **Canonical을 유지하며 V33의 제품 채택·배포는 하지 않았다.**
- 클라우드에서 V33 복원을 검증하고 V34 TRAIN gradient 분해 후 V35를 두 조건 각각 2,400 step 학습했다. 교사 정답 행만의 feature 보존은 대조군 대비 수식 exact **64/149 → 62/149**, 개선 0식·회귀 2식으로 기각했다. [V35 완료 보고서](selective-2d/reports/TEACHER_CORRECT_FEATURE_V35_20261006.md)
- 최신 V33은 V29 대비 소유 수식 Top-1 완전일치가 **62/149 → 64/149**, Top-5 완전 포함이 **122/149 → 125/149**로 회복됐다. Canonical의 **76/149, 137/149**에는 미달한다.
- V30–V33 표는 기존 산출물 요약이다. V34–V35 후속 연구는 Linux CPU에서 실제 검증·학습·평가했으며, 연구 전체를 재평가한 것은 아니다.

## 1. 목적과 처리 구조

E-ink 수학 필기에서 문자 형상과 수식 구조를 함께 읽되, 사용자가 쓴 오답을 산술적으로 정답으로 바꾸지 않는 인식 경로를 연구한다.

```text
원본 stroke / point sequence
  → 입력 정규화와 문자 영역 grouping
  → 372-class HWR 후보와 확률
  → 후보·형상·의미 역할·배치 근거 감사
  → 보수적 수식 문맥 결정 / 불확실성 표시
  → 관계 그래프와 LaTeX 연구 출력
```

Canonical 입력 계약은 128×5 시퀀스(`x`, `y`, `delta_t`, `stroke_start`, `observed`)와 uniform-time 모드다. 초기 후처리는 원래 Top-k 밖 생성·삭제·재그룹화를 제한했다. 이중 HWR, Top-10/20, 후속 grouping 연구는 별도 계약 변경으로 기록했으며 모든 연구 분기에 같은 제한이 적용되는 것은 아니다. [입력·계층 감사](selective-2d/artifacts/hwr_pipeline_microscope_20260928_semantic_guard_final/summary.json)

## 2. 누적 작업 경과

| 시기 | 수행한 작업 | 결과와 해석 | 근거 |
| --- | --- | --- | --- |
| 초기 0.5–0.9 / 1.0 | stroke 전처리, 정규화, 문자 후보, 공개 데이터 재학습, Android·Flutter 런타임 자료 | 공개 모델 자료와 1.0 연구 기반 보존 | [기존 모델 자료](../../models/DOWNLOADS.md), [초기 연구 README](baseline-1.0e/README.md) |
| 8월 | 역할 기반·증류형 수식 문맥, 배치 그래프, 구문·단일기호·이중 HWR 복원 | 반복 관찰된 직접 수집 95식에서 통합 배치 복원 누적 완전일치 81/95. 독립 실사용 성능으로 해석하지 않음 | [통합 복원 보고서](selective-2d/reports/FORMULA_PLACEMENT_RESCUE_INTEGRATION_20260819.md) |
| 9월 초 | 보조·온라인 교사 증류, 가드 실험 | 교사 및 가드별 학습·평가·체크포인트를 보존. 실험 생성과 제품 승격을 구분 | [초기 실험 산출물](baseline-1.0e/artifacts) |
| 9/14–9/27 | Selective-2D: 빠른 grouping과 국소 구조 탐색 분리 | 9/27 고정 149식에서 빠른 경로와 구조 scout 모두 grouping exact 125식. HWR margin 없는 국소 2D 추가 승리 0건 | [9/27 요약](selective-2d/artifacts/selective_2d_research_loop_20260927/summary.json) |
| 9/28–10/1 | grouping·후보·문맥·semantic guard 계층별 실패 분석 | 정답 grouping을 준 HWR 진단에서도 Top-1 exact 76/149, Top-5 완전 포함 137/149. 후보 포함과 최종 결정 사이의 차이를 확인 | [계층 감사](selective-2d/artifacts/hwr_pipeline_microscope_20260928_semantic_guard_final/summary.json) |
| 10/2–10/3 | affine·곡선·진폭·확률·파형 증강 비교, 목적함수·hard-negative·구조 비교 | 증강 방식별 비교 산출물과 별도 체크포인트를 보존. 증강량 증가 자체를 성공 기준으로 삼지 않음 | [10/3 비교](selective-2d/artifacts/hwr_augmentation_microscope_20261003), [추가 모델](additional-models) |
| 10/4 | 확률 경계·경계 tube·증류 타깃·학습 gradient 감사 | 학습 손실과 실제 gradient, 입력·로짓 일치를 점검 | [학습 microscope](selective-2d/artifacts/hwr_boundary_training_microscope_20261004_v2/training_microscope.json) |
| 10/5 | active trust retention, guard, 전 도메인·dtype·ONNX·교사 충돌·encoder/head·feature/margin 유지 비교 | 내부 문자 개선과 소유 수식 회귀를 분리. V29 margin retention을 후속 목적함수 비교 기준으로 사용 | [전체 산출물](selective-2d/artifacts), [V33 비교표](selective-2d/artifacts/hwr_near_rival_training_20261006_v33/comparison_result.json) |
| 10/6 | V30 노출 비율 → V31 숫자 sampling → V32 근접 경쟁 후보 감사 → V33 목적함수 비교 | V31 수식 비회귀 실패, V33 일부 회복. Canonical 유지 | 아래 최근 연구 상세 |
| 10/6 클라우드 재개 | 원본 8,905파일 해시 검증, V33 전체 6,515입력 재현, V34 gradient 분해, V35 두 조건 학습·독립 검증 | 로컬 gradient 정렬은 개선됐지만 V35 소유 수식은 64/149 → 62/149. 조건 기각, canonical 유지 | [복원·V34](selective-2d/reports/CLOUD_REPLAY_AND_GRADIENT_BALANCE_V34_20261006.md), [V35](selective-2d/reports/TEACHER_CORRECT_FEATURE_V35_20261006.md) |

## 3. 최근 연구: V30–V33

### V30·V31 — 학습 노출 비율과 숫자 sampling

- V30에서 숫자는 실제 TRAIN 구성비 4.46% 대비 hard-target 노출 2.91%, ASCII 문자는 3.33% 대비 14.06%였다. 이는 분포 차이이며 소유 수식 회귀의 원인을 입증하지 않는다. [V30 판정](selective-2d/artifacts/hwr_sampling_balance_20261006_v30/research_verdict.json)
- V31은 배치당 실제 원본 최대 1개를 다른 승인 숫자 원본으로 교체하고 증강·학습 예산·손실 가중치는 유지했다. 내부 숫자 Top-1은 **133/160 → 139/160**, 전체는 **4,875/5,936 → 4,885/5,936**였다.
- 소유 수식 Top-1 exact는 **62/149 → 61/149**로 내려갔으며 개선 0식·회귀 1식이었다. 수식 비회귀 조건을 충족하지 못해 채택하지 않았다. [V31 판정](selective-2d/artifacts/hwr_digit_floor_sampling_20261006_v31/research_verdict.json)

### V32 — 전체 경쟁 후보 평균 손실의 한계 감사

고정 TRAIN 8배치에서 전체 경쟁 후보 gap 손실은 feature-only 대비 약 91.94% 감소했지만 근접 후보 gap 손실 감소는 약 55.29%였다. 교사가 맞힌 423행 중 학생이 틀린 9행이 남았다. 근접 후보는 교사·학생 Top-5 합집합에서 정답을 제외해 정의했다. 잔여 제곱 결손 에너지의 약 85.28%가 먼 후보에 있었으며, 이 비율은 gradient 비율이나 인과 효과를 뜻하지 않는다. 이 단계는 재학습 없는 TRAIN 진단이었다. [V32 판정](selective-2d/artifacts/hwr_near_rival_gap_audit_20261006_v32/research_verdict.json)

### V33 — 근접 경쟁 후보 gap 유지 학습

- 기존 V29 데이터·스케줄과 2,400-step 예산을 유지했다. 배치 64개 중 원본 16개·증강 48개이며 학습 가능한 57개 텐서를 모두 갱신했다.
- Hard CE 계수 1, feature 유지 계수 0.741224를 유지하고, 전체 371개 경쟁 후보 평균 gap 보조 손실을 교사가 맞힌 행의 근접 후보 gap 손실로 교체했다. 새 계수 0.643669는 고정 TRAIN 8배치 gradient로 산출했다.
- 소유 수식 optimizer 입력 0행, CROHME 입력 0행이다. 생성 증강 라벨은 사람이 검증한 라벨로 간주하지 않는다. [고정 계획](selective-2d/artifacts/hwr_near_rival_training_20261006_v33/frozen_plan.json)

| 비교 모델 | 내부 문자 Top-1 / 5,936 | 소유 문자 Top-1 / 579 | 소유 수식 Top-1 exact / 149 | 소유 수식 Top-5 완전 포함 / 149 |
| --- | ---: | ---: | ---: | ---: |
| Canonical | 4,765 | 470 | 76 | 137 |
| 실제 원본 paired control | 4,949 | 414 | 54 | 120 |
| 증강 main | 4,935 | 408 | 52 | 121 |
| Feature retention | 4,893 | 433 | 56 | 122 |
| V29 margin retention | 4,875 | 438 | 62 | 122 |
| V33 near-rival | 4,863 | 443 | 64 | 125 |

출처: [동일 입력 비교 결과](selective-2d/artifacts/hwr_near_rival_training_20261006_v33/comparison_result.json). 내부 문자 분모는 각 2,968행의 두 개발 fold를 합친 것이며, 소유 수식은 정답 grouping을 주는 진단이다. **Top-5 완전 포함은 후보 상한이며 실제 최종 수식 정확도가 아니다.**

저장된 독립 검증은 2,400개 손실식, 57개 텐서의 유한 gradient와 변경, 교사 gradient 부재, NumPy 마스크·방향·산술, 체크포인트 재로딩 로짓의 비트 일치와 최종 지표 재집계를 통과했다. 전체 과거 중간 forward를 재생한 검증은 아니다. V33 TRAIN 근접 손실은 0.099419 → 0.019088로 줄었지만 새 필기 일반화의 증거는 아니다. [검증 결과](selective-2d/artifacts/hwr_near_rival_training_20261006_v33/independent_verification.json)

## 4. V34–V35 클라우드 후속 연구

V34의 고정 TRAIN 8배치에서 교사 오답 행의 feature gradient가 CE와 더 음의 방향을 보였다. V35는 이 행의 feature 기여만 제거하고 분모 64·CE·near-gap 계수·데이터·seed를 유지했다. 같은 CPU 대조군과 새 조건 모두 2,400 step을 완료한 후 평가했다.

| 같은 CPU 비교 | 내부 문자 Top-1 / 5,936 | 소유 문자 Top-1 / 579 | 수식 Top-1 exact / 149 | Top-5 완전 포함 / 149 |
| --- | ---: | ---: | ---: | ---: |
| Canonical 재평가 | 4,765 | 470 | 76 | 137 |
| V33 목적함수 대조군 | 4,863 | 443 | 64 | 125 |
| 교사 정답 feature 조건 | 4,864 | 440 | 62 | 123 |

4,800개 학습 기록·57개 텐서 갱신·같은 CPU 로짓 비트 재현·라벨 provenance·지표 재집계 검증을 통과했다. 새 조건의 TRAIN 잔여 정답 보존 오류는 대조군과 같은 6건이었다. 집계 gradient 정렬 개선이 인식 개선으로 이어지지 않아 기각했다. [전체 결과·판정·한계](selective-2d/reports/TEACHER_CORRECT_FEATURE_V35_20261006.md), [Linux 환경 재현](cloud/README.md)

## 5. CROHME 원본 전체 범위 평가

별도 CROHME 연구 트랙은 원본 **1,199식 전체**를 분모에 포함했다. 성공식만의 점수와 혼동하지 않는다.

| 항목 | 결과 |
| --- | ---: |
| 원본 수식 평가 범위 | 1,199/1,199 |
| 추론 성공 / 오류 | 1,014 / 185 |
| Grouping exact | 353/1,199 (29.44%) |
| 평탄 문자열 exact | 59/1,199 (4.92%) |
| Group·배치·관계·문자 동시 exact | 50/1,199 (4.17%) |

185식에는 단일 point 원본 stroke가 있고 추론 오류는 ValueError로 기록됐다. 지원되지 않는 정답 token을 가진 수식도 349식이었다. 이 결과는 **로컬 proxy 평가이며 공식 CROHME Expression Rate가 아니다.** 비상업 사후 shadow 검증으로 학습·선택·임계값 조정은 수행하지 않았다. [전체 범위 평가 원본](crohme-evaluation/artifacts/crohme_raw_full_coverage_research_20260914_r3_final/validation_report.json)

## 6. 해석 한계와 남은 검증

- 95식·149식·CROHME 평가의 분모, grouping 제공 여부, 런타임과 지표가 다르므로 하나의 성능 향상 곡선으로 연결하지 않는다.
- 최근 149식은 이미 반복 관찰한 oracle-group 진단이다. V31 기록은 95식이 과거 canonical calibration 입력이며 추가 53식은 Codex 검토 주석이라고 명시한다. 별도 96식 cohort 집계와 동일한 숫자로 치환하지 않는다.
- 새 writer·새 수식·실제 E-ink 기기의 untouched acceptance 검증이 필요하다. 증강 라벨의 사람 검증, 전체 raw-stroke grouping·문자·관계 통합 평가도 남아 있다.
- 내부 문자 향상만으로 제품 승격하지 않는다. Canonical 유지, 수식 회귀와 후보 recall을 함께 평가하며 배포·실제 기기 성능은 이 보고서가 입증하지 않는다.

---

# 부록: 연구 스냅샷과 복원

- 기준일: 2026-10-06 (한국 시간)
- 출처 채팅: `AIFlow 1.0e 2단계 연구 재개` (`01a060a8-b8dd-7bd3-ae7d-3f32223b82af`)
- 기존 체크아웃과 미커밋 변경을 함께 복사했으며 원본 작업 폴더는 수정하지 않았습니다.
- 소스별 Git HEAD, 원본 경로, 파일 크기, SHA-256: [SNAPSHOT_MANIFEST.json](SNAPSHOT_MANIFEST.json)

| 폴더 | 내용 |
| --- | --- |
| [baseline-1.0e](baseline-1.0e) | 초기 1.0/1.0e 코드, 교사·증류 실험, 보고서, 누적 산출물 |
| [selective-2d](selective-2d) | Selective-2D, 후보·의미 가드, 실패 분석, 경계·유지·근접 경쟁 후보 연구와 모든 실험 산출물 |
| [crohme-evaluation](crohme-evaluation) | 원본 전체 범위 평가 코드와 저장된 평가 결과 |
| [augmentation-models](augmentation-models) | 별도 derived 폴더에 저장됐던 affine-distill 실험 체크포인트와 생성 특징 |
| [additional-models](additional-models) | 목적함수·hard-negative·아키텍처 비교 실험 체크포인트와 생성 특징 |

## 보존 범위와 검증

- 이미 커밋된 파일, 미커밋 파일, 미추적 연구 파일, Git에서 제외되던 연구 산출물을 함께 보존했습니다.
- 원본 `.gitignore`와 `.gitattributes`는 `SOURCE_GITIGNORE.txt`, `SOURCE_GITATTRIBUTES.txt`로 보존했습니다. 기존 제외 규칙·줄바꿈 변환 때문에 스냅샷 파일이 누락되거나 바이트가 바뀌지 않게 했습니다.
- Git 메타데이터, 다운로드 캐시, 실행 환경, 외부 원본 데이터, 비공개 참여자 원본 데이터는 제외했습니다. 상세 제외 범위는 매니페스트에 기록했습니다.
- 복사본과 원본의 SHA-256 및 Python 구문을 확인했습니다. 연구 전체를 재학습·재평가한 것은 아닙니다.
- shadow 결과, oracle 결과, 연구용 평가와 제품 성능·채택 상태는 각 원본 보고서의 구분을 유지합니다.

## 내려받기와 대용량 파일 복원

GitHub의 기존 LFS 예산 초과로 대용량 파일 39개를 분할 압축해 Git에 보존했습니다. 동일한 내용은 한 번만 저장하고 원래 경로와 SHA-256을 [LARGE_ARTIFACTS_MANIFEST.json](LARGE_ARTIFACTS_MANIFEST.json)에 기록합니다. [GitHub 파일·푸시 제한](https://docs.github.com/en/repositories/creating-and-managing-repositories/repository-limits)에 맞춰 압축 조각은 각 90,000,000바이트 이하로 나눴습니다.

```powershell
git clone https://github.com/kuseumkkrkkr/AIFlow-Eink.git
cd AIFlow-Eink
python research/aiflow-1.0e/restore_large_artifacts.py --verify-only
python research/aiflow-1.0e/restore_large_artifacts.py
```

복원 도구는 Python 표준 라이브러리만 사용하며, 모든 압축 조각과 원본 파일의 해시를 검사합니다. 이미 존재하는 파일이 원본과 다르면 해당 파일을 보존하고 중단합니다.

각 폴더의 `scripts/`는 해당 폴더를 작업 디렉터리로 실행합니다. 원래 로컬 데이터 경로나 체크포인트 경로를 요구하는 실험은 원본 보고서·스크립트의 입력 설정을 확인해야 합니다. 체크포인트와 데이터의 용도·라이선스 조건은 소스별 문서에 따릅니다.
