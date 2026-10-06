# v9 상용 acceptance 데이터 조건

## 현재 결론

- synthetic writer bank와 clean-room 증강은 adapter의 구조·안전 게이트를 검증하는 증거일 뿐, 제품 승격 근거가 아니다.
- CROHME·MathWriting은 학습 또는 승격 acceptance에 사용하지 않는다.
- 기존 HWR checkpoint, 128×5 입력 계약, glyph baseline Top-5 집합은 변경하지 않는다.

## acceptance 세트의 필수 경계

- project-owned 또는 상업 이용이 명시적으로 허용된 데이터만 사용한다.
- 작가 ID, 세션, 원본 수식 hash, stroke source hash를 receipt에 고정한다.
- adaptation의 calibration 작가·수식과 writer-disjoint 및 formula-disjoint여야 한다.
- evaluation prediction payload에는 정답 token·수식 label·writer ID·세션 metadata를 넣지 않는다.
- calibration은 해당 acceptance writer의 허용된 support 예시만 사용하고, evaluation formula는 calibration과 분리한다.
- homograph와 unsupported Top-5 row는 identity/raw fallback을 유지한다.

## 사전 고정해야 할 판정

- frozen HWR baseline과 candidate adapter의 Top-5 집합이 모든 glyph에서 정확히 같아야 한다.
- 각 writer의 Top-1 및 formula exact가 baseline보다 낮아지면 즉시 reject한다.
- 전체 acceptance에서 Top-1 또는 formula exact 중 하나는 양의 개선이 있어야 한다.
- candidate violation은 0이어야 한다.
- formula-level split, writer-level split, calibration source, checkpoint·adapter·evaluator hash를 one-shot 실행 전에 receipt로 고정한다.

## 필요한 신규 수집 범주

- 적응에 사용하지 않은 실제 작가의 수식 단위 online ink.
- 동일 작가 내 calibration과 evaluation을 나눌 수 있는 반복 수식 표본.
- 숫자·영문·연산자·괄호 및 혼동쌍이 포함된 수식 표본.
- timestamp·stroke boundary·원본 입력 순서·ownership 기록이 보존된 표본.

## 승격 불가 조건

- writer/formula disjointness 또는 ownership을 증명하지 못하는 경우.
- synthetic 또는 기존 Legacy 결과만 있는 경우.
- 어느 한 writer라도 회귀하거나, aggregate 개선이 없거나, candidate violation이 있는 경우.

이 문서는 데이터 수집·학습 실행 권한이 아니다. 해당 세트가 별도로 freeze되고 독립 acceptance audit을 통과하기 전까지 상태는 `SHADOW/PRE_ACCEPTANCE`다.
