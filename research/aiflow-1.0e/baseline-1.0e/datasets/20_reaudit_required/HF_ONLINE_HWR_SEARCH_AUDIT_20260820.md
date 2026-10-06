# Hugging Face·공개 온라인 HWR 추가 데이터 감사

## 결론

- 2026-08-20 검색에서 **새로 상업 학습에 승인할 데이터는 없었다**.
- x/y/t 구조가 있는 HF 3종은 라이선스·원출처·writer provenance가 없어 재감사 폴더에만 metadata를 보존했다.
- CC BY 4.0 UEA Handwriting repack은 2D ink가 아니라 3축 가속도여서 제외했다.
- 현재 상업 학습 데이터는 기존 승인본 UJI Pen Characters v2, ISGL, UCI Character Trajectories, 정제 HWRT와 프로젝트 소유 ink를 유지한다.

## HF CLI 검색

사용한 검색어: `handwriting`, `strokes`, `mathwriting`, `crohme`, `trajectory handwriting`, `online ink`, `digital pen`, `pen trajectory`.

| 후보 | 형식 | 권리 | 판정 |
|---|---|---|---|
| `newbienewbie/handwriting_strokes` | x/y/t, label 대응 불명 | 미표기 | 재감사 |
| `newbienewbie/handwriting_strokes_2strokes` | x/y/t, stroke 결합 불명 | 미표기 | 재감사 |
| `newbienewbie/handwriting_edge_case_in_one_stroke` | x/y/t + 제한 문자 label | 미표기 | 재감사 |
| `THUgewu/Handwriting` | 3축 가속도, 26자 | CC BY 4.0 표시 | modality 불일치 |
| MathWriting mirrors | 대다수 raster/repack | 원본 CC BY-NC-SA | 상업 학습 금지 |
| CROHME mirrors | raster 또는 온라인 repack | 원본 연구·비상업 조건 | 평가 전용 |

HF README만 믿지 않고 Dataset Viewer 첫 행도 확인했다. 세 `newbienewbie` 저장소는 실제 timestamp와 좌표 배열을 갖지만, 앞의 두 저장소는 `annotation`이 행 정답이 아니라 클래스 목록으로 보였다.

## 공식 출처 교차검토

- [Google MathWriting archive README](https://github.com/google-research/google-research/blob/master/mathwriting/archive_readme.md): 온라인 InkML이지만 archive 라이선스가 CC BY-NC-SA이며 full formula에는 문자 segmentation이 없다.
- [CROHME 공식 데이터 페이지](https://www.isical.ac.in/~crohme/CROHME_data.html): 학술·연구 용도 경계로 상업 학습에 넣지 않는다.
- [IAM/On-Line 데이터베이스 안내](https://fki.tic.heia-fr.ch/databases): 비상업 연구 조건이라 제외한다.
- [CASIA OnDo](https://nlpr.ia.ac.cn/databases/CASIA-onDo/index.html): 학술 연구용이며 상업 이용은 별도 허가가 필요하다.
- [UNIPEN Zenodo](https://zenodo.org/records/1195803): 동봉 조건의 상업·재배포 범위를 현재 자료만으로 확정하지 못해 승인하지 않았다.
- AI Hub 수학 필기 [71716](https://www.aihub.or.kr/aihubdata/data/view.do?aihubDataSe=&currMenu=115&dataSetSn=71716&topMenu=100), 수식 [479](https://www.aihub.or.kr/aihubdata/data/view.do?aihubDataSe=realm&currMenu=115&dataSetSn=479&topMenu=100): 공개 설명상 PNG/JPEG·polygon·LaTeX OCR 자료로, 온라인 stroke HWR 입력이 아니다.

## 수식 문맥 사전학습 후보

- [MathBERT](https://huggingface.co/tbs17/MathBERT)는 모델/코드가 MIT로 표시되지만 학습 corpus에는 여러 교육 사이트·도서·arXiv 자료가 섞여 있어 상용 파생모델의 corpus 권리를 이번 감사로 확정하지 못했다.
- 기존 Google BERT-Tiny + 프로젝트 소유 수식 584개 파인튜닝은 2 epoch에서 현재 159식 비회귀였으나 exact 개선이 없었다. 3~4 epoch는 2식을 회귀시켜 선택하지 않았다.
- 이번 선택 문맥망은 외부 text corpus/weight 없이 프로젝트 소유 47식과 저장소 내 결정론적 수식 DSL만 사용했다.
