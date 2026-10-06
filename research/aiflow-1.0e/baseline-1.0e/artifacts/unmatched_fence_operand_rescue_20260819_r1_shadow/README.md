# 짝 없는 닫는 괄호 피연산자 복원

Top-20 회수 후보를 제품 출력으로 전면 연결하지 않고, 수식 배치상 숫자 피연산자 자리를 점유한 짝 없는 닫는 괄호 한 종류만 검토하는 shadow 레이어다.

직접 수집 개발 자료는 문자 `358/387 → 359/387`, 식 exact `78/95 → 79/95`가 됐고 회귀는 0건이다. CROHME 반복 비상업 진단 984식에서는 변경과 회귀가 모두 0건이다.

새 writer·새 formula 상업 acceptance가 없으므로 `product_default_admitted=false`다. 상세 조건과 기각 실험은 `reports/TOP20_UNMATCHED_FENCE_PLACEMENT_LOOP_20260819.md`에 기록했다.
