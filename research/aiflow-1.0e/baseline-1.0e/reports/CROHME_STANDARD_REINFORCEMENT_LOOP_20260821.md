# CROHME 표준 강화 루프 및 병목 감사

## 결론

- CROHME train-derived writer+formula 이중 holdout에서 epoch 2, λ=3.0을 선택했다.
- 공식 test는 선택 후에만 평가했으며 후보 밖 출력·grouping 변경·reload 불일치는 모두 0건이다.
- 연구 정확도는 개선됐지만 CROHME가 CC BY-NC이므로 이 체크포인트는 상용 제품에 채택할 수 없다.
- 구조적 결론은 `문맥 재랭커 학습량 부족`과 `Top-5 밖 형태 오류`, `raw grouping`이 각각 독립 병목이라는 것이다.

## 공식 test 문자·식 결과

| 경로 | 문자 Top-1 | 식 exact (cache proxy) | strict macro |
|---|---:|---:|---:|
| 고정 HWR | 72.70% | 17.61% | 50.97% |
| 기존 상용권리 문맥모델 | 77.68% | 20.37% | 56.05% |
| CROHME 연구모델 | **86.42%** | **40.65%** | **56.89%** |

`식 exact (cache proxy)`는 지원 가능한 truth-group 행만 비교한 값이다. 공식 end-to-end Expression Rate가 아니다.

### 엄격한 protocol 분모

- fully-supported 849식 oracle-group proxy: 343/849 (40.40%)
- 전체 1,199식에서 미지원 식을 실패로 처리한 proxy: 343/1,199 (28.61%)
- 완전정답/동형문자만/실오류: 487 / 61 / 650식
- 공백만 다른 오류: 0식(공백 token 자체가 없음)

## 병목 분해

- 형태 HWR Top-1: 72.70%
- Top-5 후보 회수율: 93.75%; 정답 후보 밖 749글자
- 문맥 선택 후 Top-1: 86.42%; Top-5 안에 남은 미회수 879글자
- raw grouping exact: 306/769 (39.79%)
  - 이 값은 기존 end-to-end 구조 진단이며, 이번 고정 HWR과 체크포인트가 달라 동일 파이프라인 결합 점수로 해석하지 않는다.
- 스트리밍 전체 prefix Top-1: 21.41%; 마지막 20%: 61.34%; 안정 정답 도달 중앙값: 78.79%

따라서 문맥층은 큰 폭으로 개선 가능하지만, 후보 밖 749글자와 raw grouping 실패는 문맥층만으로 해결되지 않는다.

## 확인된 문제

- CROHME test 1,199식 중 현재 출력 계약으로 완전히 평가 가능한 식은 849식이다. 미지원 truth-group 542개: `\sin`:105, `,`:97, `t`:93, `\cos`:86, `\lim`:41, `\log`:37, `\tan`:36, `.`:31, `!`:16.
- 함수명 단위(`\sin` 등)와 단일 문자 단위가 섞여 있어 단순 클래스 추가만이 아니라 토큰·grouping 계약 정렬이 필요하다.
- 출현 92클래스 중 20클래스가 기존 문맥모델보다 후퇴했다. 큰 회귀는 `\forall` 100.00%→33.33%(n=3), `\times` 72.09%→31.40%(n=86), `h` 61.11%→25.00%(n=36), `\lambda` 75.00%→50.00%(n=4), `\mu` 37.50%→12.50%(n=8)다.
- n≥10인데 Top-1이 0%인 클래스는 `C`, `P`, `S`, `V`, `X`, `\prime`이다. 빈도 높은 소문자·숫자 쪽 문맥 prior 쏠림이다.
- 전체 Top-1은 기존 문맥보다 8.74%p 올랐지만 strict macro는 0.84%p만 올랐다. class/family-balanced 목적함수가 필요하다.
- 공식 symLG/LgEval 출력과 2D 관계 점수가 구현되지 않아 현재 식 수치를 CROHME 공식 Expression Rate로 인용할 수 없다.
- 48 Hz 평가는 prefix 복원 정확도이며 실제 장치 wall-clock p50/p95 latency 측정은 아니다.

## 누수·과적합 검수

- train에서 writer와 완전 동일 token sequence를 동시에 분리했다: fit 5,670식/272 writers, dev 370식/38 writers이며 writer·sequence·record overlap은 모두 0건이다.
- 공식 test 후보 캐시는 epoch·λ 선택을 끝낸 뒤 만들었고 선택에는 test metric을 읽지 않았다.
- train loss는 epoch 1/2/3에서 0.3614→0.2604→0.2292로 감소했지만 dev 식 exact는 43.78%→44.05%→43.51%였다. 따라서 epoch 2에서 중지한 것이 타당하며 추가 epoch만으로 개선될 근거는 없다.
- 선택 모델의 Top-1/식 exact/strict macro는 dev 88.34%/44.05%/63.49%, 반복 valid 86.29%/43.60%/58.23%, test 86.42%/40.65%/56.89%다. 전형적인 epoch 과적합 붕괴보다는 test 분포와 희소 클래스에 대한 일반화 간극이 남았다.
- train 원본에서 완전 중복 식 6개를 제거했고, writer 식별 불가 4식은 split에서 제외했다. `MfrDB0104.inkml` 1개는 잘못된 XML 바이트 때문에 원본을 바꾸지 않고 Latin-1 fallback으로 읽었다.

## strict 동형군

| 군 | n | HWR | 기존 문맥 | 연구 문맥 | Top-5 |
|---|---:|---:|---:|---:|---:|
| `vertical_slash` | 1008 | 52.98% | 73.02% | 78.87% | 89.48% |
| `circle` | 280 | 30.36% | 69.29% | 74.64% | 79.64% |
| `cross` | 824 | 61.41% | 76.58% | 85.44% | 94.42% |

## 오류가 큰 문자

| 문자 | n | 연구 Top-1 | Top-5 | 후보 밖 | 주요 혼동 |
|---|---:|---:|---:|---:|---|
| `C` | 38 | 0.00% | 94.74% | 2 | c:22, (:16 |
| `X` | 38 | 0.00% | 100.00% | 0 | x:38 |
| `S` | 23 | 0.00% | 100.00% | 0 | s:17, 5:5, \int:1 |
| `P` | 15 | 0.00% | 100.00% | 0 | p:15 |
| `V` | 15 | 0.00% | 100.00% | 0 | v:15 |
| `\prime` | 13 | 0.00% | 92.31% | 1 | 1:6, /:5, \int:1 |
| `o` | 17 | 5.88% | 76.47% | 4 | 0:13, \sigma:1, \mathcal{O}:1 |
| `Y` | 16 | 6.25% | 100.00% | 0 | y:14, x:1 |
| `\mu` | 8 | 12.50% | 100.00% | 0 | p:5, x:1, A:1 |
| `M` | 14 | 21.43% | 100.00% | 0 | m:10, N:1 |
| `h` | 36 | 25.00% | 75.00% | 9 | k:7, b:6, n:3 |
| `\times` | 86 | 31.40% | 95.35% | 4 | x:58, X:1 |
| `\forall` | 3 | 33.33% | 100.00% | 0 | v:2 |
| `l` | 21 | 38.10% | 80.95% | 4 | i:3, (:3, p:2 |
| `|` | 60 | 38.33% | 93.33% | 4 | 1:33, i:2, \int:1 |
| `I` | 10 | 40.00% | 80.00% | 2 | 1:3, =:1, \pi:1 |
| `B` | 35 | 42.86% | 100.00% | 0 | \beta:14, 3:3, b:3 |
| `z` | 153 | 43.14% | 64.71% | 54 | 2:31, y:22, 3:7 |
| `\}` | 11 | 45.45% | 90.91% | 1 | ):3, 3:2, \{:1 |
| `r` | 67 | 49.25% | 86.57% | 9 | \sqrt{}:10, \gamma:6, \ohm:3 |
| `\lambda` | 4 | 50.00% | 100.00% | 0 | d:1, x:1 |
| `g` | 44 | 52.27% | 97.73% | 1 | y:14, 8:2, a:1 |
| `q` | 45 | 53.33% | 95.56% | 2 | 9:15, 7:2, 8:1 |
| `\{` | 11 | 54.55% | 100.00% | 0 | [:2, \epsilon:1, (:1 |
| `G` | 12 | 58.33% | 91.67% | 1 | a:4, 6:1 |

## 가장 많이 틀린 식

| ID | 길이 | 오류 | 후보 밖 | truth → prediction |
|---|---:|---:|---:|---|
| `UN19wb_1120_em_1170.inkml` | 19 | 16 | 0 | `| | q q \prime | | q | | | | | | | \leq | \prime q` → `1 1 9 9 7 1 1 9 1 1 1 1 1 1 | \leq | / 9` |
| `ISICal19_1206_em_830.inkml` | 34 | 14 | 2 | `M M 0 + M 1 Y + M 2 ( I + 1 ) - 4 - M 3 S ( S + 1 ) = [ I - 1 Y 2 ]` → `m m 0 + m 1 y + m 2 ( = + 7 ) - 4 - M 3 s ( s + 1 ) = T \pi - 7 x 2 2` |
| `UN19wb_1106_em_963.inkml` | 35 | 13 | 5 | `( 1 2 5 ) - 1 3 5 ) + ( 3 5 ) - 7 2 5 ) - ( 1 2 4 6 ) - 3 ) ( 7 ( ( 7` → `( 1 2 s ) - 1 3 \int ) + 1 3 5 ) - f \sum s ) - ( / a | 6 ) - 3 ) 1 7 1 1 f` |
| `UN19_1050_em_725.inkml` | 23 | 12 | 1 | `\{ \{ A B \} C \} + \{ C A \} + \{ B C \} A \{ B \} \{ \}` → `[ \{ A 3 \{ ( \} + [ c A ) + ( 3 c \} A \{ B ) 1 \}` |
| `UN19_1006_em_82.inkml` | 44 | 11 | 4 | `\alpha ( \sum 9 I 1 v I + \sum 3 1 x 4 \{ \gamma \gamma + \alpha 2 2 4 2 \alpha N f = = \gamma I i = - i \mu i i 1 2 3 \} ) / \mu` → `\alpha ( \sum 5 1 1 v 1 + \sum 3 1 x 4 \epsilon \gamma \gamma + \alpha 2 2 4 2 \alpha 6 f = = \gamma \pm i = - \perp x i 1 1 2 3 3 ) / p` |
| `UN19_1029_em_415.inkml` | 30 | 11 | 4 | `k ( E ) 1 - V ( ) \sqrt{} E 2 - ( ) + 1 ) r l = r - r 2 r l ( V l` → `k ( e ) 1 - v ( ) \sqrt{} E a - ( ) + 1 ) \mathcal{M} p = \mathcal{M} - x 2 x p ( v 7` |
| `UN19_1038_em_554.inkml` | 42 | 9 | 2 | `2 ( 1 - 1 - 2 ( j - 1 - 2 ) j + 2 n - 3 - 2 + 1 - 4 n ( 2 n - 1 ) C j 2 \dots \pm ) = ( )` → `2 1 1 - 1 - 2 1 i - 1 - 2 ) i + 2 n - 3 - 2 4 1 - 5 n ( 2 n - 1 ) ( i 2 \dots \pm ) = 1 )` |
| `UN19_1024_em_337.inkml` | 31 | 9 | 6 | `B \int 0 d n x \int k x k + 1 d y \int l x 1 d z f ( y z ) = x x x l +` → `B \int 0 d n x \int a x h + 1 d y \int l x 1 d y f ( y y ) = k k k ( 4` |
| `UN19_1015_em_195.inkml` | 24 | 9 | 6 | `c z - - P 1 - c d z - z - P 2 + f z ( ) d z d z` → `c g - - p 7 - c d g - g - p 2 + f j ( ) d y d g` |
| `UN19_1021_em_298.inkml` | 18 | 9 | 2 | `a n d o n e g o e s d o w n f n o m` → `a m d 0 n e g 0 e B d 0 w m \theta \ohm 0 m` |
| `UN19_1047_em_680.inkml` | 40 | 8 | 8 | `2 4 + 8 + 6 1 6 + 1 6 + 6 1 6 + 8 + 8 + 6 + 1 2 + 8 + 6 1 2 + 8 + 6 + 6 6 1 2 +` → `2 4 \dashv 8 + r 1 6 f 1 6 + 6 1 8 + 8 \models 8 \models 6 7 1 2 + 8 + 6 1 2 + 8 + 6 f 6 6 1 2 +` |
| `ISICal19_1204_em_796.inkml` | 34 | 8 | 3 | `F 2 p + 2 F + \alpha p + 2 1 \alpha p + 2 p + [ ] = p 2 \alpha 1 \dots F \alpha \dots [ 2 ] [ ]` → `F 2 p + 2 F + \alpha p + 2 | \alpha b + 2 7 + F 1 = p 2 \alpha \int \dots F \alpha \dots [ 2 ] 5 1` |
| `UN19_1035_em_500.inkml` | 30 | 8 | 4 | `2 ( x ) - 4 ( \gamma + 4 ) + b + 4 B \pi 2 \sqrt{} 1 - x - + 3 x f = \sqrt{} 1` → `2 ( x ) - 4 ( x + 4 ) + \& + a \beta \pi 2 \sqrt{} 7 - x - 4 3 x 6 = \sqrt{} \&` |
| `UN19_1008_em_119.inkml` | 26 | 8 | 7 | `v - ( 0 ) - \sqrt{} ( - n + 2 n - 2 n 1 = 1 \pi - 1 1 \sqrt{} 1 )` → `v - ( \degree ) - \sqrt{} ( - n + \sum n - a n h = A \pi - A A \sqrt{} n )` |
| `UN19_1023_em_318.inkml` | 22 | 8 | 5 | `( d 2 R - 1 - 2 b / d - 2 \rightarrow a b R g a ) ( )` → `c d 2 R - \mathscr{A} - 2 \& / d - 2 \rightarrow \alpha \& R y \alpha ) c )` |
| `UN19wb_1106_em_969.inkml` | 18 | 8 | 7 | `1 - 2 \int d k p \int d l o o p \pi o o l l` → `\Delta - \sum \int d v p \int d l \mathcal{O} 0 p \pi \theta 0 l (` |
| `UN19_1017_em_226.inkml` | 33 | 7 | 1 | `3 8 c 5 + 3 7 c 4 - 7 2 9 3 - 1 7 8 2 c 2 + 1 9 3 c + 1 7 1 0 c 5` → `5 8 ( 5 + 3 n ( 4 - q 2 9 3 - 1 7 8 2 c 2 + 1 9 3 ( + 1 7 1 0 ( 5` |
| `UN19_1004_em_45.inkml` | 30 | 7 | 1 | `P x ) x 2 P 4 ( x ) x x 2 - a 2 ) ( x 2 - b 2 ) 6 ( = = 2 (` → `p x ) x e p 4 ( x ) x x e - a e ) ( x 2 - b e ) 6 ( = = e (` |
| `UN19_1042_em_600.inkml` | 28 | 7 | 3 | `y d 2 - q 2 - 1 + q 2 d q - 1 + 2 d x y x = j y x - j q` → `y d 2 - 9 2 - ) + 7 2 d 9 - 1 + z d x 7 x = j y x - j 7` |
| `UN19_1030_em_434.inkml` | 21 | 7 | 1 | `A d S ( 3 ) \times S ( 3 ) \times S ( 3 ) \times S 1 ) (` → `a d s ( 3 ) \times s ( 3 ) \times s ( 3 ) x s y ) (` |

## 채택 판정

- 연구 체크포인트: 비상용 CROHME challenger로 보관하되 **shadow 유지**.
- 상용 AIFlow 1.0 체크포인트: **미채택**. CC BY-NC 학습 가중치를 제품에 넣지 않는다.
- 상용 개선 경로: 동일한 candidate-validity 목적함수를 Apache-2.0/프로젝트 소유 수식과 새 소유 writer 데이터로 재현한다.
- grouping은 별도 ownership 모델, 후보 밖 문자는 형태 HWR 데이터 보강으로 해결한다.

## 산출물

- 선택 보고서: `D:\AIFlow-Workspace\Projects\Aiflow\aiflow-math-ink-1.0\artifacts\crohme_standard_context_20260820_r1_research\selection_report.json`
- test JSON: `D:\AIFlow-Workspace\Projects\Aiflow\aiflow-math-ink-1.0\artifacts\crohme_standard_context_20260820_r1_research\test_evaluation_report.json`
- test 예측: `D:\AIFlow-Workspace\Projects\Aiflow\aiflow-math-ink-1.0\artifacts\crohme_standard_context_20260820_r1_research\test_predictions.jsonl.gz`
