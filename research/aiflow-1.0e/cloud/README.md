# Linux CPU 연구 환경

Python 3.12에서 V33 이후 연구를 재개하기 위한 환경이다. 검증한 조합은 Torch 2.5.1+cpu, NumPy 2.1.3, SciPy 1.15.3이다. GPU나 서비스 서버는 필요하지 않다. Windows에서 생성한 원본 아카이브와 인증서는 보존한다.

관리형 클라우드의 기존 `/workspace/AIFlow-Eink` 체크아웃에서 실행한다. 이 환경 자체가 격리되어 있으므로 별도 worktree는 필요하지 않다.

```bash
cd /workspace/AIFlow-Eink
bash research/aiflow-1.0e/cloud/install-linux-cpu.sh
source /workspace/.aiflow-research-env/venv/bin/activate
export PYTHONPYCACHEPREFIX=/workspace/.aiflow-research-env/pycache
cd research/aiflow-1.0e/selective-2d
```

설치 스크립트는 체크아웃 바깥에 가상환경과 캐시를 만든다. Torch는 공식 CPU wheel index에서 별도로 설치하고, 나머지 연구 패키지는 [고정 목록](requirements-linux-cpu.txt)과 동일한 버전을 설치한다. 이 목록만 설치하면 Torch 설치 단계가 빠진다. 기존 문서/schema 검증도 별도 가상환경으로 유지한다. 원본 압축 파일·복원 파일의 SHA 검증과 TLS 검증을 끄지 않는다.

전체 아카이브 복원과 Git 객체·패키지를 포함해 약 27 GiB를 사용했다. 실험 전 디스크와 cgroup 메모리·CPU 제한을 확인한다. 관찰한 제한은 CPU 4개, 메모리 32 GiB다. V35의 두 arm은 각각 Torch thread 2개로 실행했으며, 각 2,400-step 학습에 약 40분이 걸린다.

## 이미 완료된 진단 재실행

출력 파일은 새로운 경로를 사용한다. 기존 인증서나 결과를 덮어쓰지 않는다.

```bash
python scripts/verify_hwr_cloud_v33.py --output /tmp/aiflow-v33-new-replay.json
python scripts/audit_hwr_gradient_balance_v34.py --output /tmp/aiflow-v34-new-audit.json
```

V33 strict 검증기는 Linux의 gradient norm 절대 오차 `<1e-6` 조건에서 실패한다. 별도 portable 검증기는 오차 범위와 비트 일치 여부를 공개하고, 모든 입력의 Top-5 순서와 지표 일치를 별도로 요구한다. [검증 근거](../selective-2d/reports/CLOUD_REPLAY_AND_GRADIENT_BALANCE_V34_20261006.md)

## 새 V35 paired 실행

아래 경로는 새 실험 전용 예시다. 실제 생성된 `hwr_teacher_correct_feature_20261006_v35_cloud`를 덮어쓰지 않는다. 준비 단계는 두 조건의 소스·계수·데이터 해시를 봉인한다. 실행 중 봉인된 소스 파일을 수정하지 않는다.

```bash
experiment=artifacts/hwr_teacher_correct_feature_new_run
python scripts/run_hwr_teacher_correct_feature_v35.py prepare --output "$experiment"
python scripts/run_hwr_teacher_correct_feature_v35.py train --output "$experiment" --arm v33_control &
control_pid=$!
python scripts/run_hwr_teacher_correct_feature_v35.py train --output "$experiment" --arm teacher_correct_feature &
challenger_pid=$!
wait "$control_pid"
control_status=$?
wait "$challenger_pid"
challenger_status=$?
test "$control_status" -eq 0 && test "$challenger_status" -eq 0
```

두 프로세스의 종료 코드와 두 `completed.json` 모두를 확인한 후에만 평가·검증한다. 실패했으면 해당 실행을 보존하고 원인을 조사한다.

```bash
python scripts/run_hwr_teacher_correct_feature_v35.py evaluate --output "$experiment"
python scripts/verify_hwr_teacher_correct_feature_v35.py --output "$experiment"
python scripts/audit_hwr_teacher_correct_feature_v35.py --output "$experiment"
```

검증은 총 4,800개 손실 기록·전체 모델 갱신·같은 CPU의 최종 로짓 비트 재현·소유 라벨 provenance·지표 재집계를 확인한다. 추가 TRAIN 진단은 기존 8배치에서 gradient 합성과 잔여 정답 보존 오류를 점검한다. 모든 과거 중간 forward 재생이나 untouched acceptance 검증을 뜻하지 않는다.

프로세스는 새 클라우드 작업으로 승계되지 않는다. `resume_state.pt`는 128 step마다 저장한 상태이며 자동 resume CLI는 구현되지 않았다. [새 작업 시작 지침](START.md)을 읽고 완료·실패·기록 상태부터 확인한다.

## V36 이후 진행

[V37 및 엄격 복구 CLI](V37.md)는 원본 상태·로그를 보존하고 저장 이후의 겹침을 정확 재생한다. 이번 복구에서는 부동소수 불일치로 종료됐다. [V38 재현 지침](V38.md)은 현재 환경에서 두 조건을 모두 처음부터 다시 학습하는 명령과 검증 범위를 설명한다. 원래 V35/V37 trainer에는 자동 resume 기능을 추가하지 않았다.

`CLOUD_RESULTS_MANIFEST_20261006.json`의 편집 문서 해시는 V35 결과를 저장한 `040f7effa63c97507fa781d42cf77c365b224ea2` 커밋에 적용된다. 후속 README·시작 지침의 변경으로 과거 해시를 갱신하지 않는다. 원래 연구 파일과 과거 인증서는 그대로 보존한다.

## V39 오류 행 보정

[V39 재현 지침](V39.md)은 기존 다중 near 후보와 pair 수 분모를 유지하고 교사 정답·학생 오답 행의 gap만 고정 2배로 보정한다. 검증된 같은 CPU V38 대조군의 5개 학습 파일을 그대로 재사용하며, 새 학습은 후보의 2,400단계 하나다. 준비 단계가 CPU·Torch build·출처 인증서와 복사 해시를 검사한다. 완료 후 대조군 추론도 새로 실행해 V38 출처 로짓·지표와 비트 일치를 확인한다. Canonical 지표는 V38 인증서에서 재사용한다.

`CLOUD_RESULTS_MANIFEST_20261007_V38.json`의 편집 문서 해시는 `47bafa4cc3b9fe1d8bce4fb0bdbf2c875496050c` 스냅샷에 적용된다. V39 이후의 문서 변경으로 이 과거 매니페스트를 다시 쓰지 않는다.
