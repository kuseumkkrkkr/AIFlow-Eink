# AIFlow 1.0e 누적 연구 스냅샷

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
