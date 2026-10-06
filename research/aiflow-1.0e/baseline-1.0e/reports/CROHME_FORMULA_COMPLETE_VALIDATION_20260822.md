# CROHME 전체 수식 완료 후 검증

## 검증 경계

- CROHME 학습·epoch 선택·임계값 선택·gradient update: `0`
- 중간 prefix/스트리밍 정확도: 측정하지 않음
- 입력 종료(`formula_complete`) 후 전체 수식 문맥과 2D 배치만 채점
- truth character grouping을 사용하므로 공식 end-to-end Expression Rate는 아님

## 문자·식 결과

| 단계 | 문자 Top-1 | 식 exact proxy |
|---|---:|---:|
| 고정 HWR | 72.70% | 17.61% |
| 전체 수식 문맥 | 76.51% | 22.12% |

## 2D 배치

- traceGroup 순서 exact: `34.14%`
- 순서+문자 식 exact: `12.19%`
- relation formula exact: `56.09%`
- relation+문자 식 exact: `18.03%`

## 해석

- 실시간 prefix 수치가 낮았던 현상은 더 이상 제품 정확도 정의에 포함하지 않는다.
- 문맥층은 Top-5 밖 749건을 복구할 수 없으므로 영문 보조 HWR와 형태 데이터 보강은 별도 필요하다.
- raw-stroke 자동 grouping 성능은 이 truth-group 검증과 분리해 보고해야 한다.
