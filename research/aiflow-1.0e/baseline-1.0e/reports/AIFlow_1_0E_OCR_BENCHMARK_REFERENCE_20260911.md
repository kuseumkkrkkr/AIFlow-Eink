# OCR 비교 기준 — 2026-09-11 확인

- 목적: AIFlow 연구 결과와 공개 OCR 결과를 서로 다른 지표로 혼합하지 않고 제시.
- 아래 순위는 명시된 표에 포함된 모델 사이의 순위. 전체 시장 순위가 아님.
- 공개 가중치와 상업적 이용 권한은 별개. 표 등재만으로 배포 가능 판정하지 않음.
- AIFlow는 online ink 입력. 아래 모델은 이미지/PDF 입력. 동일 시험을 거치지 않은 AIFlow의 대외 순위는 **미측정**.

## 1. 필기 수식: UniMER-Test HWE

CDM 논문 Table 4의 필기 수식(HWE) 열. `ExpRate@CDM`은 렌더링 기반 전체 수식 일치율이며 LaTeX 문자열 Exact가 아님. 원문 18페이지를 PNG로 렌더링하여 표를 시각 확인함.

| 표 내 순위 | 모델 | HWE CDM | HWE ExpRate@CDM |
|---:|---|---:|---:|
| 1 | UniMERNet-base | 94.00 | 64.31% |
| 2 | UniMERNet-small | 93.70 | 63.93% |
| 3 | UniMERNet-tiny | 93.28 | 61.86% |
| 4 | Mathpix | 93.18 | 59.28% |
| 5 | Texify | 52.69 | 23.59% |
| 6 | Pix2tex | 24.53 | 0.60% |

출처: [Image Over Text: Transforming Formula Recognition Evaluation with Character Detection Matching, Table 4](https://arxiv.org/pdf/2409.03643). 논문 당시 버전 결과이며 현재 API 전체 성능을 뜻하지 않음. TexTeller는 이 표에 없으므로 순위를 부여하지 않음.

원문 확인 이미지: [cdm_table4.png](../artifacts/accuracy_upgrade_20260911_sources/cdm_table4.png).

## 2. 일반 문서 OCR: OmniDocBench v1.6_full

공식 README의 `Overall` 내림차순. 공식 화면에서 상위 행을 스크린샷으로 확인함.

| 표 내 순위 | 모델 | Overall | Formula CDM |
|---:|---|---:|---:|
| 1 | PaddleOCR-VL-1.6 | 96.34 | 97.5304 |
| 2 | MinerU2.5-Pro | 95.75 | 97.45 |
| 3 | GLM-OCR | 95.22 | 97.18 |
| 4 | PaddleOCR-VL-1.5 | 94.93 | 96.89 |
| 5 | PaddleOCR-VL | 94.18 | 95.91 |
| 13 | Gemini 3 Pro | 92.91 | 95.99 |
| 14 | Gemini 3 Flash | 92.62 | 95.16 |
| 23 | GPT-5.2 | 86.59 | 88.21 |
| 27 | Mistral OCR | 85.66 | 89.91 |

출처: [OmniDocBench 공식 평가표](https://github.com/opendatalab/OmniDocBench#end-to-end-evaluation).

- Overall은 텍스트 편집거리 변환 점수·표 TEDS·수식 CDM의 평균. 문서 전체 Exact나 필기 수식 Exact가 아님.
- 버전은 표의 제목 `v1.6_full`을 그대로 따름. 저장소 업데이트 공지와 표 버전이 다르므로 임의로 v1.7 점수로 고쳐 쓰지 않음.
- 상위 5개는 공개 모델 계열. API 모델과 같은 표에 있지만 배포 조건·입력 전처리·비용은 동일하지 않음.
- Mathpix의 이 버전 전체 문서 점수는 이 표에 없어 다른 버전 점수를 끼워 넣지 않음.

## 3. 이번 연구의 teacher 권리 경계

- TexTeller: [공식 모델 페이지](https://huggingface.co/OleehyO/TexTeller)의 Apache-2.0 표시 확인. 로컬 특징·증류 연구에 사용. 고지·데이터 권한의 배포 검토는 별도.
- UniMERNet-tiny: [공식 연결 모델 페이지](https://huggingface.co/wanderkid/unimernet_tiny)의 Apache-2.0 표시 확인.
- TrOCR-small-handwritten: [모델 페이지](https://huggingface.co/microsoft/trocr-small-handwritten)에 IAM 미세조정 이력은 있으나 가중치 라이선스 표기가 없음. [상위 소스 저장소 MIT 라이선스](https://github.com/microsoft/unilm/blob/master/LICENSE)만으로 해당 가중치·파생 배포 권리를 확정하지 않음. 엄격한 이번 경로에서는 추가 증류 및 3-teacher 앙상블을 권리 확인 전 보류.
- 이미 추출한 TrOCR 특징은 보존하지만 학습 승인 또는 배포 허가의 근거로 취급하지 않음.
- 법률 의견이나 상업적 사용의 무조건적 보증이 아님.

## 4. AIFlow 위치 해석

- AIFlow 점수는 구현 보고서의 동일 writer·동일 수식·동일 평가 규칙 실측끼리만 비교.
- 과거 정답 문자 묶음 제공 점수를 종단간 Exact로 표기하지 않음.
- 같은 HWE 이미지에서 AIFlow가 동작하려면 원본 online ink가 필요. 이미지의 합성 좌표 변환으로 공정한 online 평가를 대체하지 않음.
- 공정한 직접 순위는 동일한 새 online ink를 AIFlow에 입력하고, 그 ink를 동일 규칙으로 렌더링한 이미지를 비교 OCR에 입력하여 각각 최종 수식을 평가해야 산출 가능. 이 자료는 개발·임계값 선택에서 봉인해야 함.
- 이번 실행으로 상용 최상위 또는 특정 시장 순위를 입증할 수 없음.
