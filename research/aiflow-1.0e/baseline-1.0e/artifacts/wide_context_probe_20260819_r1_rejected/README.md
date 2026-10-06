# Top-10 문맥 신경망 재배치 탐색

고정된 `owned_formula_context_product.pt`를 Top-10 HWR 후보에 연결해 추론만 수행한 감사 결과다. 학습, 문자 삽입·삭제, stroke regrouping, 배치 순서 변경은 수행하지 않았다.

범용 재배치는 직접 수집 식 exact를 78/95에서 61/95로 떨어뜨렸고, CROHME 반복 진단에서도 288/984에서 272/984로 떨어뜨렸다. `X`를 `x`로만 바꾸는 제한 분기는 직접 수집에서 1식을 개선했지만 CROHME 문자에서 개선 1건·퇴행 3건이어서 채택하지 않았다.

따라서 이 산출물은 `rejected`, `runtime_admitted=false`이며 현재 Top-10 구문 배치 경로를 변경하지 않는다. 상세 근거는 `reports/WIDE_CONTEXT_TOP10_PROBE_REJECTION_20260819.md`에 있다.
