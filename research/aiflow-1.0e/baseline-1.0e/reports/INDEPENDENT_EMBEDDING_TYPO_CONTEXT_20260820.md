# 독립 임베딩 수식 문맥 교정기 튜닝 결과

작성일: 2026-08-20  
상태: **shadow 채택, 상용 기본값 교체 보류**

## 결론

- 기존 r6 문맥 체크포인트를 불러오지 않는 별도 문맥모델을 구현했다.
- 고정된 372종 HWR Top-5 안에서만 재선택하며, 후보 생성·스트로크 재그룹·형태모델 학습은 하지 않는다.
- writer-LOO 전체 Top-1은 **72.87% → 87.34%**, 식 완전일치는 **33.68% → 65.26%**로 상승했다.
- 새 작성자 구간은 **68.02% → 81.98%**이나, 과거 r6의 84.88%에는 2.91%p 낮다.
- CROHME 사후 전이는 문자 Top-1 **70.61% → 79.49%**로 r6 78.11%보다 1.37%p 높다. 식 완전일치 26.73%는 r6 27.44%보다 0.71%p 낮다.
- 새 작성자·새 수식의 untouched 상용 acceptance가 없으므로 최종 승격은 하지 않는다.

## 모델 구조

```text
고정 HWR Top-5 + 확률
          │
          ├── 식의 다른 위치 Top-1 토큰 + 2D 관계 토큰
          │                │
          │       BERT-Tiny 임베딩 문맥 인코더
          │       2 layers / hidden 128 / 2 heads
          │                │
          │       MASK 위치의 372 클래스 임베딩 유사도
          │                │
          └── log P(HWR) + λ log P(context)
                           │
                  기존 Top-5 내부 argmax
```

- 기반 가중치: `google/bert_uncased_L-2_H-128_A-2`, 고정 revision, Apache-2.0.
- 수학 372 클래스 토큰 임베딩은 decoder와 묶어 masked-token retrieval로 학습한다.
- r6 또는 다른 AIFlow 문맥 체크포인트 warm-start: 없음.
- 한 글자 식은 이웃 문맥이 없으므로 HWR Top-1을 고정한다.

## 오탈자 교정 방식 반영

- NeuSpell의 핵심 원칙처럼 깨끗한 문맥과 손상 문맥을 함께 학습한다.
- 손상은 임의 문자가 아니라 각 outer training fold의 HWR Top-5 혼동 분포에서만 뽑는다.
- held writer의 혼동 정보는 noise 생성에도 쓰지 않는다.
- 생성형 언어모델은 사용하지 않는다. 따라서 후보 밖 기호 생성 가능성은 구조적으로 차단된다.

참조:

- [NeuSpell](https://github.com/neuspell/neuspell) — MIT, 실제/합성 철자 오류를 문맥 안에서 교정.
- [Misspelling Correction with Pre-trained Contextual Language Model](https://arxiv.org/abs/2101.03204) — 문맥 임베딩과 제한 후보 순위 결합.
- [Google BERT-Tiny](https://huggingface.co/google/bert_uncased_L-2_H-128_A-2) — 고정 기반 모델.

## 데이터 및 누수 차단

- 프로젝트 직접 수집: 387자, 95식, 7 writer.
- 프로젝트 소유 prompt corpus: 595식 중 580식/2,072자를 채택.
- 지원하지 않는 `t` 포함 식 11개 제외.
- CROHME와 정확히 같은 토큰열 4개를 prompt 학습에서 제외.
- outer writer-LOO에서 held 식과 동일한 토큰열을 가진 training 식 총 10개를 fold별 제외.
- inner selection도 formula ID뿐 아니라 완전히 같은 토큰열을 한 그룹으로 묶었다.
- CROHME는 epoch, λ, noise 선택에 사용하지 않고 최종 사후 전이에만 사용했다.

## 결과

| 평가 | HWR Top-1 | 독립 임베딩 accuracy | 과거 r6 |
|---|---:|---:|---:|
| 직접 수집 전체 문자 Top-1 | 72.87% | **87.34%** | 비교 부적합¹ |
| 직접 수집 전체 식 exact | 33.68% | **65.26%** | 비교 부적합¹ |
| 새 writer 문자 Top-1 | 68.02% | 81.98% | **84.88%** |
| 새 writer 식 exact | 31.82% | 56.82% | **59.09%** |
| CROHME 문자 Top-1 | 70.61% | **79.49%** | 78.11% |
| CROHME 식 exact | 16.46% | 26.73% | **27.44%** |

¹ r6는 직접 수집 legacy 일부를 학습했으므로 전체 OOF 비교에 쓰지 않았다.

선택된 accuracy 설정은 3 epoch, λ=0.5다. OOF에서 58건을 개선하고 2건을 역행했다. 두 역행은 각각 `0→O`, `\\times→x`인 동형문자 오판이다. 후보 밖 출력과 grouping 변경은 직접 수집 387건 및 CROHME 9,535건 모두 0건이다.

## 산출물

- 실행 코드: `scripts/train_independent_formula_context_v1.py`
- 상세 JSON: `artifacts/independent_embedding_typo_context_20260820_r2_shadow/independent_context_report.json`
- accuracy 체크포인트 SHA-256: `768d7ebb0db290f5fdbe87a9ace45f8b5da75308b30b53de6c535b2f9f4d8e8a`
- safe 체크포인트 SHA-256: `7ad7ccc24d2a8498ad090c8acb6bb063bf09f38f3cdf7c1fa6f8ceb70e3e4fba`
- 고정 HWR SHA-256: `04f8608aebcf6c02d45ad6f5735229b9eaa2c4b4e1be0db4793d02273ef2d00e`

재현 명령:

```powershell
$env:PYTHONNOUSERSITE='1'
python scripts\train_independent_formula_context_v1.py --device cuda
```

## 승격 조건

- 새 writer와 새 식을 함께 만족하는 untouched 수집 세트 추가.
- 형태 HWR·후보 캐시 SHA를 고정한 채 독립 acceptance 1회 수행.
- r6 대비 새 writer Top-1/식 exact 비회귀 및 동형문자 macro 비회귀.
- CROHME는 라이선스상 상용 학습·승격 근거가 아니라 학술 진단으로만 유지.
