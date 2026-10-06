# 광범위 수학식 후보 적합성 판별 루프

검증일: 2026-08-20  
상태: **r1 shadow 유지 / 자동 승격 금지**

## 결론

- 광범위한 수학식에서 “이 위치에 이 토큰이 들어가도 되는가”를 학습하는
  별도 신경망은 유효했다.
- 최선 결과는 r1 accuracy checkpoint다. 직접수집 nested writer-OOF에서
  88.63% Top-1, 70.53% 식 exact를 기록했고, 새 writer에서는 88.37%,
  72.73%, strict macro 97.50%였다.
- 형태 HWR, 스트로크 grouping, 기존 Top-5는 고정했다. 후보 밖 출력과
  grouping 변경은 direct 387건 및 CROHME 9,535건 모두 0건이다.
- 다만 CROHME에서 `|`와 `o`가 무너졌다. 희소기호 합성 보강 r2/r3는 일부
  회복했지만 직접수집 writer-OOF가 하락했으므로 채택하지 않는다.
- r1도 fresh untouched acceptance가 없어 상용 기본값으로 승격할 수 없다.

## 데이터 감사

### 채택

- 원천: [Google DeepMind Mathematics Dataset](https://github.com/google-deepmind/mathematics_dataset)
- 라이선스: Apache-2.0
- 고정 revision: `427f45075f84b8b9774950196ad63867ca20ffb3`
- 생성 결과: 80,000개 고유 수식열, 49개 upstream module 기여
- 길이: 최소 3, 평균 11.7385, 최대 64 토큰
- frozen 372-token 계약으로 완전히 표현되는 식만 채택
- direct 및 CROHME exact token-sequence overlap: 0
- corpus SHA-256: `1998779e46e693db2452c2a677dc0d366d4252be49ca27e3974b97df6fcc9bc5`

### 감사 후 미채택

- [OpenWebMath](https://huggingface.co/datasets/open-web-math/open-web-math):
  데이터셋 표기는 ODC-By지만 원 웹페이지의 라이선스와 이용조건을 대체하지
  않으므로 직접 상용 학습 코퍼스로 채택하지 않았다.
- [mathlib4](https://github.com/leanprover-community/mathlib4): Apache-2.0이나
  Lean 소스 문법이 현재 372개 손글씨 수식 토큰과 직접 맞지 않아 사용하지 않았다.

## 모델 구조

1. 각 위치의 기존 HWR Top-5 후보를 하나씩 실제 수식열에 삽입한다.
2. Apache-2.0 Google BERT-Tiny(2 layers, hidden 128, 2 heads)에 넣는다.
3. 삽입 위치를 `token_type_id=1`로 표시하고 binary validity head가
   `P(valid | formula, position, candidate)`를 계산한다.
4. 최종 점수는 `log P(HWR) + lambda * log P(valid)`이다.
5. 최종 argmax는 기존 Top-5 안에서만 수행한다. 한 글자 식은 HWR Top-1을 잠근다.

학습 오답은 동형군 치환을 우선하고, 해당 동형군이 없을 때만 역할이 맞지 않는
토큰으로 치환했다. CROHME는 학습이나 hyperparameter 선택에 사용하지 않았다.

## r1 선택 결과

- broad pretraining: epoch 2
- direct/prompt fine-tuning: epoch 2
- fusion lambda: 3.0
- broad validation: accuracy 93.20%, loss 0.1758
- checkpoint SHA-256:
  `ee0033fb42f59b09f3f130300fb5374710cfef010a71c94dbd48e4d1c93dc8b9`
- checkpoint reload mismatch: direct 0, CROHME 0

### 직접수집 nested writer-OOF

| 모델 | 전체 Top-1 | 전체 식 exact | 전체 strict macro | 새 writer Top-1 | 새 writer 식 exact | 새 writer strict macro |
|---|---:|---:|---:|---:|---:|---:|
| HWR Top-1 | 72.87% | 33.68% | - | 68.02% | 31.82% | - |
| 독립 임베딩 r2 | - | - | - | 81.98% | 56.82% | 71.56% |
| 기존 context r6 | - | - | - | 84.88% | 59.09% | 70.83% |
| **candidate validity r1** | **88.63%** | **70.53%** | **89.15%** | **88.37%** | **72.73%** | **97.50%** |

r1은 새 writer에서 기존 r6의 세 승격 지표를 모두 넘었고 regression은 0건이었다.

### CROHME 반복 진단

| 모델 | Top-1 | 식 exact | strict macro | `|` 정답 | `o` 정답 |
|---|---:|---:|---:|---:|---:|
| HWR Top-1 | 70.61% | 16.46% | - | 38/62 | 0/11 |
| 독립 임베딩 r2 | 79.49% | 26.73% | 61.42% | - | - |
| 기존 context r6 | 78.11% | 27.44% | 56.07% | - | - |
| candidate validity r1 | 77.45% | 21.75% | 55.42% | 1/62 | 0/11 |
| coverage r2 | 77.49% | 25.00% | 56.04% | 16/62 | 0/11 |
| replay25 r3 | 74.58% | 21.04% | 58.16% | 30/62 | 2/11 |

r2/r3의 CROHME 수치는 r1 오류 분석으로 보강 토큰을 정한 뒤 얻은 값이다.
따라서 untouched 성능 근거가 아니라 반복 실험 진단으로만 사용한다.

## 희소 동형군 루프

대상 15토큰은 `0/O/o/\mathcal{O}/\circ`, `1/|/l//\mathbb{1}`,
`x/\times/X/\mathcal{X}/\chi`다. 토큰마다 800식, 총 12,000식의
project-owned 합성 문맥을 만들었다. 평가 수식열 중복은 0이다.

### r2: broad epoch 2 뒤 coverage fine-tuning

- coverage 검증 accuracy: 86.69% → 88.37% → 88.37%
- 합성 pairwise macro: r1 59.98% → coverage epoch 1 87.57%
- 합성 `|`, `o` pairwise: 각각 100%
- 그러나 새 writer Top-1은 84.30%로 내려가 r6 84.88%를 넘지 못했다.

### r3: direct fine-tuning 중 토큰당 25식 재생

- `|`는 30/62, `o`는 2/11까지 회복했다.
- 새 writer Top-1 81.98%, 식 exact 54.55%로 더 하락했다.
- 합성 문맥을 실제 사용자 문맥 배치에 섞으면 문맥 분포가 희석된다는 결론이다.

### 추가 head-only probe

r1 product의 BERT 본체를 고정하고 validity head만 5/10/25식씩 보정했으나
`|`는 2/62에 머물렀다. 이 경로는 full writer-OOF 실행 가치가 없어 중지했다.

## 판정

- **선택:** r1 accuracy를 promotion-candidate shadow로 보존
- **폐기:** r2 coverage, r3 replay25의 상용 승격
- **금지:** r1을 현재 기본 runtime으로 자동 교체
- **다음 필수 데이터:** 프로젝트 소유의 실제 `|`, `o`, `O`, `x`, `\times`
  양성 수식과 새 writer/formula untouched acceptance set
- 새 데이터 없이 합성식 비율이나 epoch만 더 늘리는 것은 writer 일반화를
  악화시켰으므로 중지한다.

## 재현 산출물

- broad corpus builder: `scripts/build_deepmind_formula_context_corpus_v1.py`
- sparse coverage builder: `scripts/build_candidate_validity_coverage_corpus_v1.py`
- trainer/runtime loader: `scripts/train_candidate_validity_context_v1.py`
- r1 report: `artifacts/candidate_validity_context_20260820_r1_shadow/candidate_validity_context_report.json`
- r2 report: `artifacts/candidate_validity_context_20260820_r2_coverage_shadow/candidate_validity_context_report.json`
- r3 report: `artifacts/candidate_validity_context_20260820_r3_replay25_shadow/candidate_validity_context_report.json`

모든 전체 실행에서 입력 SHA는 전후 동일했고, checkpoint 재로딩 결과도 저장 직전
예측과 byte-for-byte 동일한 결정(불일치 0건)을 냈다.

