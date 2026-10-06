# 인간 작가 모델 연구: 영문·숫자·수학 수기

## 결론

- 고도화 대상은 단순한 필기체 모양 변형이 아니라 `문자 내용 → allograph 선택 → 획 운동계획 → 작가 전역 스타일 → 문맥/수식 배치 → 운동 실행`의 계층이다.
- 작가 전역 특성(기울기·종횡비·속도·떨림)과 문자별 특성(시작점·획수·획순·방향·지연 획)을 같은 연속형 변형으로 처리하면 안 된다.
- 수식은 의미 관계 트리, 2차원 배치, 실제 작성 순서를 분리해야 한다. 예를 들어 위첨자는 공간상 앞 토큰에 종속되지만 작성은 본선 이후일 수 있다.
- 인식기는 한 가지 표준 획순을 정답으로 강제하지 않고 관측 순서를 증거로 보존하면서, 동일 문자에서 관측된 여러 운동계획을 허용해야 한다.
- 현재 372-class 어휘에는 소문자 `t`가 없다. 따라서 현 상태는 숫자 10종, 대문자 26종, 소문자 25종만 다루며 완전한 `a-z` 모델이라고 부를 수 없다.

## 1차 근거

| 근거 | 작가 모델에 반영할 사항 |
|---|---|
| Plamondon et al., [The lognormal handwriter](https://pubmed.ncbi.nlm.nih.gov/24391610/) | 필기는 시간상 중첩되는 신경운동 획의 합으로 보고 속도 프로파일을 별도 운동 계층에서 모델링한다. |
| [Motor equivalence in handwriting](https://pubmed.ncbi.nlm.nih.gov/10744963/) | 크기·평면·효과기가 달라도 개인 양식이 남으므로 writer-global latent와 실행 변동을 분리한다. |
| [Functional asymmetries in handwriting](https://pubmed.ncbi.nlm.nih.gov/10353378/) | 손 선호에 따라 길이·지속시간·최대속도·일관성이 달라질 수 있어 단일 속도 상수를 작가성으로 간주하지 않는다. |
| [Disentangling Writer and Character Styles](https://arxiv.org/abs/2303.14736) | 기울기·비율 같은 전역 스타일과 획 길이·곡률 같은 문자 국소 스타일을 분리한다. |
| [Allographic agraphia](https://pmc.ncbi.nlm.nih.gov/articles/PMC3095950/) | 추상 grapheme, 대소문자/형태의 allograph 선택, 구체적 운동계획을 별도 단계로 둔다. |
| [Graphic motor programs](https://pmc.ncbi.nlm.nih.gov/articles/PMC7008555/) | 문자 운동프로그램은 시작 위치, 획 순서, 진행 방향을 포함한다. |
| [Effector-independent writing representations](https://pmc.ncbi.nlm.nih.gov/articles/PMC6238209/) | 획 시퀀스 계획과 손·펜에 따른 실행을 분리하고, 시각 피드백은 방향·간격·형태를 보정한다. |
| [Motor anticipation in handwriting](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2022.807935/full) | 다음 문자가 현재 획에 영향을 주므로 영문 연결성과 간격에 이웃 문맥을 포함한다. |
| [MathWriting](https://arxiv.org/abs/2404.10690) | 온라인 수식은 획 시계열과 2차원 구조를 함께 평가해야 한다. 본 프로젝트에서는 평가 자료로만 격리한다. |
| [Stroke Constrained Attention Network](https://arxiv.org/abs/2002.08670) | 온라인 수식 인식의 기본 단위를 픽셀이 아닌 획으로 유지한다. |
| [SRT from stroke sequences](https://arxiv.org/abs/2105.06084) | 수식의 stroke order 변이를 허용하면서 관계 트리를 복원하는 별도 계층이 필요하다. |

## v4 결손 감사

- 전역 affine·곡률·운동 응답만 바꾸며, 한 문자의 획수·시작점·방향·획순 대안을 표현하지 못한다.
- 문자 인접 문맥, 단어 간격, 연결 획, 점·가로획의 지연 작성을 표현하지 못한다.
- 글리프 128×5만 생성하며 수식 관계와 수식 전체의 raw stroke production order가 없다.
- `delta_t`를 입력 정규화 계약으로 보존하는 것은 타당하지만, 실제 작가의 문자 사이 휴지시간과 수식 작성 순서는 별도 이벤트 계층에 남겨야 한다.

## v5 계약

1. `WriterLatent`: 기울기, 폭/높이, 기준선, 간격, 속도, 휴지시간 등 한 작가가 공유하는 값.
2. `AllographPlan`: 승인 원천으로 만든 clean-room 텐서에서 추출한 토큰별 획수, 시작/끝 구역, 방향, 획 순서의 후보. 원시 관측과 합성 변형을 구분해 provenance를 남기고 임의 획 역전은 금지한다.
3. `ContextPlan`: 앞뒤 문자에 따른 간격·연결 후보. 아직 학습값이 아니므로 보수적 규칙과 provenance를 남긴다.
4. `FormulaPlan`: 의미 토큰과 baseline/superscript/fraction 관계를 보존한 공간 배치.
5. `ProductionPlan`: 좌→우, 본선 우선, 분수 구조 우선 등 실제 펜 이벤트 순서. 의미·공간 순서와 독립이다.
6. `MotorExecution`: formula raw event의 시간·휴지를 생성하되 인식용 box-local 128×5의 시간품질 정규화는 유지한다.

## 상용 경계

- 현재 allograph 원형은 승인 원천을 DTW/profile/physics로 합성한 clean-room bank에서 선택한다. 이는 원시 인간에게서 관측한 다양성 증거가 아니다.
- 같은 계보의 동결 HWR로 거르는 Top-5 gate는 합성 라벨 보존 검사일 뿐, 인간 필체 다양성이나 신규 작가 일반화 성능의 증거가 아니다.
- CROHME·MathWriting은 학습, 파라미터 선택, allograph 원형 추출에 사용하지 않는다.
- 합성 수식의 생산순서 규칙은 인간 가능성 가설이지 실제 작가 통계로 주장하지 않는다. 새 작가 원시 수식으로 빈도와 타당성을 검증해야 한다.
- 제품 runtime과 HWR/context 체크포인트는 이번 단계에서 변경하지 않는다.
- 신규 작가 writer-disjoint acceptance는 아직 실시하지 않았으므로 상용 채택 gate는 닫아 둔다.

## 다음 데이터 수집 우선순위

- 누락 클래스 `t`와 `t/f/+` 동형군의 여러 작가 온라인 궤적.
- 동일 문장을 여러 번 쓴 writer-session 자료: 대소문자 allograph, 연결/비연결, 문자 간격의 작가 내 일관성 측정.
- 같은 수식을 자유 순서로 쓴 자료: 위·아래첨자, 분수선, 괄호, 점·가로획의 실제 production order 기록.
- 손잡이·펜 종류·기기 표본은 writer ID와 분리해 기록하여 작가성과 장치 변동을 혼동하지 않는다.
