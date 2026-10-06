# CROHME 오류 등가성·과적합·레이어·시간 보전 감사

- 생성 시각(UTC): `2026-08-20T05:43:03.029825+00:00`
- 학습 수행: 없음
- 판정 경계: 동형문자는 의미상 정답이 아니라 형상 혼동 진단으로만 분리한다.

## 1. 완전정답·동형문자·실오류

공백은 CROHME symbol-group 토큰에 존재하지 않으므로 처음부터 점수에서 제외된다. 띄어쓰기만 다른 식은 완전정답으로 처리되며, 공백 때문에 오답 처리된 식은 0개다.
표의 token 열은 렌더링된 LaTeX 문자열 순서가 아니라 InkML 정답 그룹 인덱스 순서다. 각 위치는 같은 정답 그룹의 truth와 Top-1을 직접 비교한다.

| 범위 | 완전정답 | 동형문자만 | 실오류 |
|---|---:|---:|---:|
| test 전체 (1199) | 133 (11.09%) | 70 (5.84%) | 996 (83.07%) |
| 평가 가능 truth-group (849) | 133 (15.67%) | 70 (8.24%) | 646 (76.09%) |

### 문자 단위

| 완전정답 | 동형 혼동 | 실오류 |
|---:|---:|---:|
| 8559/11991 (71.38%) | 753/11991 (6.28%) | 2679/11991 (22.34%) |

### 적용한 동형군

| 군 | 토큰 | 실제 혼동 건수 |
|---|---|---:|
| `vertical_or_slash` | `/`, `1`, `I`, `\backslash`, `\mid`, `\prime`, `\setminus`, `i`, `l`, `\|` | 183 |
| `circle` | `0`, `O`, `\O`, `\circ`, `\degree`, `\fullmoon`, `\mathcal{O}`, `o` | 192 |
| `cross` | `X`, `\chi`, `\mathcal{X}`, `\mathfrak{X}`, `\times`, `x` | 145 |
| `sigma_sum` | `\Sigma`, `\sum` | 60 |
| `pi_product` | `\Pi`, `\prod` | 0 |
| `perpendicular` | `\bot`, `\perp` | 0 |
| `equality_approx` | `=`, `\approx`, `\simeq` | 97 |
| `s_five` | `5`, `S`, `s` | 31 |
| `beta_eight` | `8`, `\beta` | 0 |
| `right_arrow` | `\longrightarrow`, `\rightarrow`, `\shortrightarrow` | 45 |

이 목록 밖의 `x→\varkappa`, `c→C`, `9→g` 같은 쌍은 비슷해 보여도 기존 승인 동형군이 아니므로 실오류로 유지했다.

### 동형문자만 다른 식

| ID | truth token | prediction token | 차이 |
|---|---|---|---|
| `UN19_1009_em_131.inkml` | `1 + + 1 1 + 1 5 3 7 1 0 =` | `\| + + \| \| + \| 5 3 7 \prime \fullmoon =` | `1`→`\|` (동형:vertical_or_slash), `1`→`\|` (동형:vertical_or_slash), `1`→`\|` (동형:vertical_or_slash), `1`→`\|` (동형:vertical_or_slash), `1`→`\prime` (동형:vertical_or_slash), `0`→`\fullmoon` (동형:circle) |
| `UN19_1036_em_515.inkml` | `X 9 ( X 2 X 7 - X 3 X 6 )` | `\times 9 ( \times 2 \times 7 - \times 3 \times 6 )` | `X`→`\times` (동형:cross), `X`→`\times` (동형:cross), `X`→`\times` (동형:cross), `X`→`\times` (동형:cross), `X`→`\times` (동형:cross) |
| `ISICal19_1204_em_798.inkml` | `( 2 0 0 0 ) 5 9 2 0 - 9 3 3 5` | `( 2 O o o ) 5 9 2 O - 9 3 3 5` | `0`→`O` (동형:circle), `0`→`o` (동형:circle), `0`→`o` (동형:circle), `0`→`O` (동형:circle) |
| `UN19_1040_em_576.inkml` | `\theta ( \pm ( ( x 1 0 + + 0 1 + + y 4 0 ) ) ) \dots x 4 ) - ( y 0 \dots` | `\theta ( \pm ( ( X \mid 0 + + 0 \| + + y 4 0 ) ) ) \dots X 4 ) - ( y 0 \dots` | `x`→`X` (동형:cross), `1`→`\mid` (동형:vertical_or_slash), `1`→`\|` (동형:vertical_or_slash), `x`→`X` (동형:cross) |
| `UN19_1040_em_578.inkml` | `1 + 1 + 1 + 1` | `\| + \| + \| + \|` | `1`→`\|` (동형:vertical_or_slash), `1`→`\|` (동형:vertical_or_slash), `1`→`\|` (동형:vertical_or_slash), `1`→`\|` (동형:vertical_or_slash) |
| `UN19_1009_em_127.inkml` | `\times X X \times X` | `X \times X \times \times` | `\times`→`X` (동형:cross), `X`→`\times` (동형:cross), `X`→`\times` (동형:cross) |
| `UN19_1011_em_150.inkml` | `x b \rightarrow a x +` | `\times b \shortrightarrow a \times +` | `x`→`\times` (동형:cross), `\rightarrow`→`\shortrightarrow` (동형:right_arrow), `x`→`\times` (동형:cross) |
| `ISICal19_1202_em_771.inkml` | `C \leq 7 \times 1 0 - 7` | `C \leq 7 X I 0 - 7` | `\times`→`X` (동형:cross), `1`→`I` (동형:vertical_or_slash) |
| `ISICal19_1202_em_778.inkml` | `a + 1 0 - 2 \geq` | `a + \backslash O - 2 \geq` | `1`→`\backslash` (동형:vertical_or_slash), `0`→`O` (동형:circle) |
| `ISICal19_1209_em_859.inkml` | `3 0 2 0 \div` | `3 o 2 o \div` | `0`→`o` (동형:circle), `0`→`o` (동형:circle) |
| `UN19_1012_em_175.inkml` | `2 3 0 \div 0` | `2 3 \fullmoon \div \fullmoon` | `0`→`\fullmoon` (동형:circle), `0`→`\fullmoon` (동형:circle) |
| `UN19_1017_em_232.inkml` | `- ( x - x ) x 0 - x 0 =` | `- ( x - x ) x o - x \circ =` | `0`→`o` (동형:circle), `0`→`\circ` (동형:circle) |
| `UN19_1030_em_428.inkml` | `x 2 - y 2 - 0 x 3 =` | `x 2 - y 2 - O \chi 3 =` | `0`→`O` (동형:circle), `x`→`\chi` (동형:cross) |
| `UN19_1038_em_540.inkml` | `\sum n s n` | `\Sigma n S n` | `\sum`→`\Sigma` (동형:sigma_sum), `s`→`S` (동형:s_five) |
| `UN19_1038_em_553.inkml` | `o \in X` | `0 \in \times` | `o`→`0` (동형:circle), `X`→`\times` (동형:cross) |
| `UN19_1041_em_595.inkml` | `x 2 a 1 x a 2 = \sum 3 =` | `\chi 2 a \| x a 2 = \sum 3 =` | `x`→`\chi` (동형:cross), `1`→`\|` (동형:vertical_or_slash) |
| `UN19_1043_em_619.inkml` | `\int d x i d x j` | `\int d \chi i d \chi j` | `x`→`\chi` (동형:cross), `x`→`\chi` (동형:cross) |
| `UN19_1043_em_621.inkml` | `x y \times` | `\chi y X` | `x`→`\chi` (동형:cross), `\times`→`X` (동형:cross) |
| `UN19_1049_em_719.inkml` | `( 0 0 0 0 0 ) ( - 1 - 1 0 0 0 )` | `( 0 O 0 O 0 ) ( - 1 - 1 0 0 0 )` | `0`→`O` (동형:circle), `0`→`O` (동형:circle) |
| `UN19_1051_em_737.inkml` | `3 \times 3 \times 3 + 1 0 \times 3 + 3` | `3 X 3 \times 3 + \mid 0 \times 3 + 3` | `\times`→`X` (동형:cross), `1`→`\mid` (동형:vertical_or_slash) |
| `UN19wb_1108_em_990.inkml` | `\sum x 7 2 0 d x =` | `\Sigma x 7 2 \fullmoon d x =` | `\sum`→`\Sigma` (동형:sigma_sum), `0`→`\fullmoon` (동형:circle) |
| `UN19wb_1109_em_1008.inkml` | `s s` | `S S` | `s`→`S` (동형:s_five), `s`→`S` (동형:s_five) |
| `UN19wb_1109_em_1019.inkml` | `S 2 \times S 2 \times S 2 \times S 2` | `S 2 X S 2 \times S 2 X S 2` | `\times`→`X` (동형:cross), `\times`→`X` (동형:cross) |
| `UN19wb_1118_em_1141.inkml` | `7 7 7 - 4 0 0` | `7 7 7 - 4 O O` | `0`→`O` (동형:circle), `0`→`O` (동형:circle) |
| `UN19wb_1120_em_1172.inkml` | `4 x 3 + A x + B y 2 =` | `4 \chi 3 + A x + B y 2 \simeq` | `x`→`\chi` (동형:cross), `=`→`\simeq` (동형:equality_approx) |
| `UN19wb_1120_em_1181.inkml` | `x 3 - x 7` | `\chi 3 - \chi 7` | `x`→`\chi` (동형:cross), `x`→`\chi` (동형:cross) |
| `ISICal19_1203_em_793.inkml` | `A o o` | `A O o` | `o`→`O` (동형:circle) |
| `ISICal19_1205_em_824.inkml` | `\times [ \times [ b 1 b 2 b 3 [ ] ] ]` | `\times [ \times [ b I b 2 b 3 [ ] ] ]` | `1`→`I` (동형:vertical_or_slash) |
| `ISICal19_1206_em_835.inkml` | `4 \times 9` | `4 X 9` | `\times`→`X` (동형:cross) |
| `ISICal19_1207_em_841.inkml` | `- 3 - \sqrt{} 6 0` | `- 3 - \sqrt{} 6 o` | `0`→`o` (동형:circle) |
| `ISICal19_1210_em_877.inkml` | `1 - 2 \times 1 - 2` | `1 - 2 \times \| - 2` | `1`→`\|` (동형:vertical_or_slash) |
| `UN19_1004_em_48.inkml` | `\int x 0 d n x` | `\int x O d n x` | `0`→`O` (동형:circle) |
| `UN19_1005_em_74.inkml` | `\sum = - 1 - q i 4` | `\Sigma = - 1 - q i 4` | `\sum`→`\Sigma` (동형:sigma_sum) |
| `UN19_1007_em_102.inkml` | `- 4 - \sqrt{} 3 6 0` | `- 4 - \sqrt{} 3 6 \circ` | `0`→`\circ` (동형:circle) |
| `UN19_1008_em_111.inkml` | `p \times p` | `p X p` | `\times`→`X` (동형:cross) |
| `UN19_1009_em_129.inkml` | `x + y \leq x + \| y \| \| \| \| \|` | `x + y \leq x + \| y \| \| \prime \| \|` | `\|`→`\prime` (동형:vertical_or_slash) |
| `UN19_1010_em_147.inkml` | `\times X X` | `\times \times X` | `X`→`\times` (동형:cross) |
| `UN19_1017_em_236.inkml` | `0 r - 1 > >` | `O r - 1 > >` | `0`→`O` (동형:circle) |
| `UN19_1019_em_267.inkml` | `6 - \sqrt{} 3 6 0` | `6 - \sqrt{} 3 6 \circ` | `0`→`\circ` (동형:circle) |
| `UN19_1022_em_308.inkml` | `f x x + f y y \neq 0` | `f x x + f y y \neq \circ` | `0`→`\circ` (동형:circle) |

### 아예 틀린 식 예시

| ID | truth token | prediction token | 동형/실오류 |
|---|---|---|---|
| `UN19wb_1106_em_963.inkml` | `( 1 2 5 ) - 1 3 5 ) + ( 3 5 ) - 7 2 5 ) - ( 1 2 4 6 ) - 3 ) ( 7 ( ( 7` | `\langle \wedge 2 S ) - \lambda 3 S ) 4 \| 3 \mathcal{S} ) - \dag \Sigma \mathcal{S} ) - ( \setminus a \iota G ) - 3 ) i 7 \| i q` | `(`→`\langle` (오류), `1`→`\wedge` (오류), `5`→`S` (동형:s_five), `1`→`\lambda` (오류), `5`→`S` (동형:s_five), `+`→`4` (오류), `(`→`\|` (오류), `5`→`\mathcal{S}` (오류) |
| `ISICal19_1204_em_796.inkml` | `F 2 p + 2 F + \alpha p + 2 1 \alpha p + 2 p + [ ] = p 2 \alpha 1 \dots F \alpha \dots [ 2 ] [ ]` | `F 2 P \psi 2 F + \alpha \vDash + 2 \lceil \varpropto \vDash + 2 \vDash + \tau ] = P 2 \alpha \int \dots F \alpha \dots [ 2 \rrbracket \tau 1` | `p`→`P` (오류), `+`→`\psi` (오류), `p`→`\vDash` (오류), `1`→`\lceil` (오류), `\alpha`→`\varpropto` (오류), `p`→`\vDash` (오류), `p`→`\vDash` (오류), `[`→`\tau` (오류) |
| `UN19_1017_em_226.inkml` | `3 8 c 5 + 3 7 c 4 - 7 2 9 3 - 1 7 8 2 c 2 + 1 9 3 c + 1 7 1 0 c 5` | `\geqslant 8 C 5 + 3 \curvearrowright ( 4 - q 2 g 3 \frown 1 q 8 2 \subset 2 + 1 g 3 C + 1 7 1 \circ ( 5` | `3`→`\geqslant` (오류), `c`→`C` (오류), `7`→`\curvearrowright` (오류), `c`→`(` (오류), `7`→`q` (오류), `9`→`g` (오류), `-`→`\frown` (오류), `7`→`q` (오류) |
| `UN19_1015_em_195.inkml` | `c z - - P 1 - c d z - z - P 2 + f z ( ) d z d z` | `C g - - p 7 - C d g - g - p 2 + f J 1 ) d J d g` | `c`→`C` (오류), `z`→`g` (오류), `P`→`p` (오류), `1`→`7` (오류), `c`→`C` (오류), `z`→`g` (오류), `z`→`g` (오류), `P`→`p` (오류) |
| `UN19_1037_em_527.inkml` | `1 - ( k + ) 1 - k 2 - 1 - 2 - 1 - k + 1 - k + 1 k 2 1 = k` | `\ell - ( K \nmid ) 1 - K 2 - 1 - \ae \rightleftharpoons 1 - K \lambda 1 - K + 1 K 2 1 \Im K` | `1`→`\ell` (오류), `k`→`K` (오류), `+`→`\nmid` (오류), `k`→`K` (오류), `2`→`\ae` (오류), `-`→`\rightleftharpoons` (오류), `k`→`K` (오류), `+`→`\lambda` (오류) |
| `ISICal19_1206_em_830.inkml` | `M M 0 + M 1 Y + M 2 ( I + 1 ) - 4 - M 3 S ( S + 1 ) = [ I - 1 Y 2 ]` | `m m o + M 7 Y + M 2 ( I 4 7 ) - 4 - M 3 \mathcal{S} r S + 1 ) \simeq \tau I - 7 Y \mathcal{Z} ]` | `M`→`m` (오류), `M`→`m` (오류), `0`→`o` (동형:circle), `1`→`7` (오류), `+`→`4` (오류), `1`→`7` (오류), `S`→`\mathcal{S}` (오류), `(`→`r` (오류) |
| `UN19_1008_em_119.inkml` | `v - ( 0 ) - \sqrt{} ( - n + 2 n - 2 n 1 = 1 \pi - 1 1 \sqrt{} 1 )` | `\vee - ( \circ ) - \sqrt{} ( \setminus n + \mathcal{L} \mathcal{R} - \mathcal{L} n \Lambda = \wedge \pi - h \wedge \sqrt{} \wedge )` | `v`→`\vee` (오류), `0`→`\circ` (동형:circle), `-`→`\setminus` (오류), `2`→`\mathcal{L}` (오류), `n`→`\mathcal{R}` (오류), `2`→`\mathcal{L}` (오류), `1`→`\Lambda` (오류), `1`→`\wedge` (오류) |
| `UN19wb_1106_em_960.inkml` | `+ 4 - 1 4 - 1 2 1 - 2 - 3 - 4 E 0 = - 1 =` | `\dashv 4 - \Lambda 4 - \Lambda \mathscr{D} \wedge - a - 3 - 4 \in \circ \circlearrowright - \wedge \Im` | `+`→`\dashv` (오류), `1`→`\Lambda` (오류), `1`→`\Lambda` (오류), `2`→`\mathscr{D}` (오류), `1`→`\wedge` (오류), `2`→`a` (오류), `E`→`\in` (오류), `0`→`\circ` (동형:circle) |
| `UN19_1031_em_446.inkml` | `a [ 2 ] 1 - 2 + a 3 + a 4 + 5 - 2 = a 2` | `\varpropto [ 2 J 1 \frown 2 + \varpropto 3 + \ae \leq + \mathcal{S} \frown a = \varpropto 2` | `a`→`\varpropto` (오류), `]`→`J` (오류), `-`→`\frown` (오류), `a`→`\varpropto` (오류), `a`→`\ae` (오류), `4`→`\leq` (오류), `5`→`\mathcal{S}` (오류), `-`→`\frown` (오류) |
| `UN19wb_1108_em_992.inkml` | `y x x x + b 3 y x + c 3 x y + 3 y y = a 3 d d d d d d` | `Y x x x + \& 3 Y x \vdash C 3 x Y + 3 y Y = a 3 \pitchfork d \pitchfork d 4 d` | `y`→`Y` (오류), `b`→`\&` (오류), `y`→`Y` (오류), `+`→`\vdash` (오류), `c`→`C` (오류), `y`→`Y` (오류), `y`→`Y` (오류), `d`→`\pitchfork` (오류) |
| `UN19_1038_em_543.inkml` | `\phi x ) c ( \sqrt{} 1 a + 1 ) - ( 1 + ( 1 + 4 x - a + 1 ) - 2 a x ) ( = + 4 2 x - c x` | `\phi \varkappa ) C ( \sqrt{} \Lambda a + 1 ) - ( 1 + ( 1 + \Leftarrow x - a \shortrightarrow 1 ) - 2 a \varkappa ) 1 = + \langle 2 x - ( \chi` | `x`→`\varkappa` (오류), `c`→`C` (오류), `1`→`\Lambda` (오류), `4`→`\Leftarrow` (오류), `+`→`\shortrightarrow` (오류), `x`→`\varkappa` (오류), `(`→`1` (오류), `4`→`\langle` (오류) |
| `UN19_1042_em_600.inkml` | `y d 2 - q 2 - 1 + q 2 d q - 1 + 2 d x y x = j y x - j q` | `y d 2 - 9 2 - ) + 7 z d 9 - 1 + Z d x g \times = j g x - j 7` | `q`→`9` (오류), `1`→`)` (오류), `q`→`7` (오류), `2`→`z` (오류), `q`→`9` (오류), `2`→`Z` (오류), `y`→`g` (오류), `x`→`\times` (동형:cross) |
| `UN19_1007_em_95.inkml` | `+ 1 2 0 S R + 1 4 4 S L a a + 4 8 S L L a b + 4 8 0 S 2 L a a + 4 8 0 S 3 i j j i L b b a b` | `+ 1 2 0 \mathcal{S} R + 1 4 4 S \lfloor a a + 4 3 S \lfloor L a b + 4 8 0 \mathcal{S} 2 L \mathfrak{A} \mathfrak{A} + 4 8 0 \mathcal{S} 3 i j j i \lfloor b b a b` | `S`→`\mathcal{S}` (오류), `L`→`\lfloor` (오류), `8`→`3` (오류), `L`→`\lfloor` (오류), `S`→`\mathcal{S}` (오류), `a`→`\mathfrak{A}` (오류), `a`→`\mathfrak{A}` (오류), `S`→`\mathcal{S}` (오류) |
| `UN19_1025_em_357.inkml` | `\sqrt{} ( x 1 ) 2 + ( ) 2 + ( x 3 ) 2 ( x 2 R = x 2 + 4 )` | `- ( \times 1 ) \wr + ( ) \aa + ( \times \} ) 2 ( \nmid 2 R I \times 2 \dag \lightning )` | `\sqrt{}`→`-` (오류), `x`→`\times` (동형:cross), `2`→`\wr` (오류), `2`→`\aa` (오류), `x`→`\times` (동형:cross), `3`→`\}` (오류), `x`→`\nmid` (오류), `=`→`I` (오류) |
| `UN19wb_1106_em_969.inkml` | `1 - 2 \int d k p \int d l o o p \pi o o l l` | `\triangle - \mathfrak{A} \int d v \wp \int d \ell \mathcal{O} \sigma P \Pi \mathcal{O} \mathcal{O} l l` | `1`→`\triangle` (오류), `2`→`\mathfrak{A}` (오류), `k`→`v` (오류), `p`→`\wp` (오류), `l`→`\ell` (오류), `o`→`\mathcal{O}` (동형:circle), `o`→`\sigma` (오류), `p`→`P` (오류) |
| `ISICal19_1206_em_826.inkml` | `x d y - - 1 + 2 d + j 2 2 - 1 - 1 + 2 = j q q y x q q d x y` | `\varkappa d y - - \rceil + \sim d + j 2 2 - I - \setminus + 2 = j q \copyright y \varkappa q q \mathfrak{A} \varkappa \gamma` | `x`→`\varkappa` (오류), `1`→`\rceil` (오류), `2`→`\sim` (오류), `1`→`I` (동형:vertical_or_slash), `1`→`\setminus` (동형:vertical_or_slash), `q`→`\copyright` (오류), `x`→`\varkappa` (오류), `d`→`\mathfrak{A}` (오류) |
| `ISICal19_1210_em_873.inkml` | `3 7 c 5 + 3 6 c 4 - 2 7 5 c 3 - 7 0 2 c 2 + 7 1 1 c + 8 5 4` | `3 > \subset 5 + 3 6 \mathcal{C} 4 - 2 > 5 \subset 3 - 7 0 2 \mathcal{C} 2 f 7 \| \| \subset + 8 5 4` | `7`→`>` (오류), `c`→`\subset` (오류), `c`→`\mathcal{C}` (오류), `7`→`>` (오류), `c`→`\subset` (오류), `c`→`\mathcal{C}` (오류), `+`→`f` (오류), `1`→`\|` (동형:vertical_or_slash) |
| `UN19_1009_em_133.inkml` | `- b - c + b - 1 b a = 1 a` | `- e - C 4 \mathcal{C} - \| \varrho \alpha \leq \| \Omega` | `b`→`e` (오류), `c`→`C` (오류), `+`→`4` (오류), `b`→`\mathcal{C}` (오류), `1`→`\|` (동형:vertical_or_slash), `b`→`\varrho` (오류), `a`→`\alpha` (오류), `=`→`\leq` (오류) |
| `UN19wb_1111_em_1039.inkml` | `9 \times 9 + 1 \times - ( 3 + 3 + 1 ) 2 4 3 = 3 1 3` | `g x 9 + 1 \nmid - ( \ss + ) + 1 ) 2 h \lambda \approx \lambda 1 ]` | `9`→`g` (오류), `\times`→`x` (동형:cross), `\times`→`\nmid` (오류), `3`→`\ss` (오류), `3`→`)` (오류), `4`→`h` (오류), `3`→`\lambda` (오류), `=`→`\approx` (동형:equality_approx) |
| `ISICal19_1210_em_882.inkml` | `v ( x f 1 + x ) = + x 2 \dots` | `V ( \between \fint f \ni \chi ) = \div \between \partial \dots` | `v`→`V` (오류), `x`→`\between` (오류), `f`→`\fint` (오류), `1`→`f` (오류), `+`→`\ni` (오류), `x`→`\chi` (동형:cross), `+`→`\div` (오류), `x`→`\between` (오류) |
| `UN19_1034_em_487.inkml` | `x 1 + x 2 + x 4 2 x 3 =` | `\mathcal{U} \wedge \dashv \varkappa 2 \dag \varkappa h 2 \varkappa 3 \simeq` | `x`→`\mathcal{U}` (오류), `1`→`\wedge` (오류), `+`→`\dashv` (오류), `x`→`\varkappa` (오류), `+`→`\dag` (오류), `x`→`\varkappa` (오류), `4`→`h` (오류), `x`→`\varkappa` (오류) |
| `UN19_1018_em_242.inkml` | `a 1 + a 2 a 3 + a 4 + a 5 + = 6 a` | `a 1 + a 2 \mathfrak{A} \mathscr{S} + a \mathds{C} + \varpropto \mathcal{S} \dashv \Xi 6 \aa` | `a`→`\mathfrak{A}` (오류), `3`→`\mathscr{S}` (오류), `4`→`\mathds{C}` (오류), `a`→`\varpropto` (오류), `5`→`\mathcal{S}` (오류), `+`→`\dashv` (오류), `=`→`\Xi` (오류), `a`→`\aa` (오류) |
| `UN19_1028_em_392.inkml` | `z x x 1 3 - 1 x 3 4 x - 1 4 2 = 2 1` | `\gamma \varkappa \varkappa \wedge 3 - \Lambda \varkappa 3 4 x - \Lambda 4 z = 2 1` | `z`→`\gamma` (오류), `x`→`\varkappa` (오류), `x`→`\varkappa` (오류), `1`→`\wedge` (오류), `1`→`\Lambda` (오류), `x`→`\varkappa` (오류), `1`→`\Lambda` (오류), `2`→`z` (오류) |
| `UN19_1034_em_490.inkml` | `a x - 1 + x = 2 a i 2 a z` | `a \mathscr{H} - \wedge \rfloor \varkappa I 2 G \lambda 2 a 2` | `x`→`\mathscr{H}` (오류), `1`→`\wedge` (오류), `+`→`\rfloor` (오류), `x`→`\varkappa` (오류), `=`→`I` (오류), `a`→`G` (오류), `i`→`\lambda` (오류), `z`→`2` (오류) |
| `UN19_1035_em_502.inkml` | `5 6 c + 8 v + 5 6 v + 8 c` | `5 6 \subset \Psi 8 \sim + \int 6 \rightsquigarrow \dag \nu \subset` | `c`→`\subset` (오류), `+`→`\Psi` (오류), `v`→`\sim` (오류), `5`→`\int` (오류), `v`→`\rightsquigarrow` (오류), `+`→`\dag` (오류), `8`→`\nu` (오류), `c`→`\subset` (오류) |
| `UN19_1036_em_521.inkml` | `g ( ) ( + b - 1 ) - 2 a = a + b` | `\emptyset ( ) 1 \sphericalangle \ell - 1 ) - 2 Q = \subseteq \psi \ell` | `g`→`\emptyset` (오류), `(`→`1` (오류), `+`→`\sphericalangle` (오류), `b`→`\ell` (오류), `a`→`Q` (오류), `a`→`\subseteq` (오류), `+`→`\psi` (오류), `b`→`\ell` (오류) |
| `UN19wb_1106_em_968.inkml` | `( c ( a ) + \sqrt{} 2 ( f ) = f f b ) i` | `( C \| a ) + \sqrt{} \mathfrak{A} i \mathscr{S} ) \Im \mathscr{S} f b ) \perp` | `c`→`C` (오류), `(`→`\|` (오류), `2`→`\mathfrak{A}` (오류), `(`→`i` (오류), `f`→`\mathscr{S}` (오류), `=`→`\Im` (오류), `f`→`\mathscr{S}` (오류), `i`→`\perp` (오류) |
| `UN19_1021_em_298.inkml` | `a n d o n e g o e s d o w n f n o m` | `a m d O \mathcal{M} e g O e \Lambda d O \omega m \astrosun \Omega o m` | `n`→`m` (오류), `o`→`O` (동형:circle), `n`→`\mathcal{M}` (오류), `o`→`O` (동형:circle), `s`→`\Lambda` (오류), `o`→`O` (동형:circle), `w`→`\omega` (오류), `n`→`m` (오류) |
| `UN19_1017_em_239.inkml` | `\sum k 1 p + k = 1 = i k \sum q k j` | `\Sigma \mathcal{R} 1 8 + \mathcal{R} \asymp 1 = i R \Sigma q R \aa` | `\sum`→`\Sigma` (동형:sigma_sum), `k`→`\mathcal{R}` (오류), `p`→`8` (오류), `k`→`\mathcal{R}` (오류), `=`→`\asymp` (오류), `k`→`R` (오류), `\sum`→`\Sigma` (동형:sigma_sum), `k`→`R` (오류) |
| `UN19_1034_em_492.inkml` | `\sum = 0 g + 1 i n - 1 i x i` | `\Sigma I o g + \Lambda \lambda h - \wedge i \varkappa \perp` | `\sum`→`\Sigma` (동형:sigma_sum), `=`→`I` (오류), `0`→`o` (동형:circle), `1`→`\Lambda` (오류), `i`→`\lambda` (오류), `n`→`h` (오류), `1`→`\wedge` (오류), `x`→`\varkappa` (오류) |
| `UN19_1049_em_714.inkml` | `w ( z ) \sum n z 1 - n = \geq 0 a n` | `\omega ( Z ) \Sigma m Z 1 - m \simeq \geqslant 0 a m` | `w`→`\omega` (오류), `z`→`Z` (오류), `\sum`→`\Sigma` (동형:sigma_sum), `n`→`m` (오류), `z`→`Z` (오류), `n`→`m` (오류), `=`→`\simeq` (동형:equality_approx), `\geq`→`\geqslant` (오류) |
| `UN19wb_1106_em_965.inkml` | `1 6 \times 8 ( + 1 8 ) 2 5 6 6 \times =` | `\Lambda G \L 8 ( \psi \Lambda 8 ) 2 S 6 G X \rightleftarrows` | `1`→`\Lambda` (오류), `6`→`G` (오류), `\times`→`\L` (오류), `+`→`\psi` (오류), `1`→`\Lambda` (오류), `5`→`S` (동형:s_five), `6`→`G` (오류), `\times`→`X` (동형:cross) |
| `ISICal19_1211_em_897.inkml` | `4 ( n + 1 ) - 3 n 4 - n - =` | `4 ( \varkappa \div \int ) - 3 \varkappa \mathcal{G} \prime m - \simeq` | `n`→`\varkappa` (오류), `+`→`\div` (오류), `1`→`\int` (오류), `n`→`\varkappa` (오류), `4`→`\mathcal{G}` (오류), `-`→`\prime` (오류), `n`→`m` (오류), `=`→`\simeq` (동형:equality_approx) |
| `UN19_1036_em_516.inkml` | `A \int d x ( \sum j B ( b j ( x ) = h x ) j ) x` | `A \int d x \prime \Sigma j \mathcal{B} ( \ell j 1 x ) ] \vartriangle x ) j ) \mathscr{F}` | `(`→`\prime` (오류), `\sum`→`\Sigma` (동형:sigma_sum), `B`→`\mathcal{B}` (오류), `b`→`\ell` (오류), `(`→`1` (오류), `=`→`]` (오류), `h`→`\vartriangle` (오류), `x`→`\mathscr{F}` (오류) |
| `UN19wb_1116_em_1116.inkml` | `( a - b ) - ( - b - c ) \times ( a - b ) = ( a - k + c ) \times ( a - b ) k` | `\iota \mathfrak{A} - b ) - ( - b - c ) \times ( a - b ) = ( a - K \Psi C ) x ( \mathfrak{A} - b ) \hbar` | `(`→`\iota` (오류), `a`→`\mathfrak{A}` (오류), `k`→`K` (오류), `+`→`\Psi` (오류), `c`→`C` (오류), `\times`→`x` (동형:cross), `a`→`\mathfrak{A}` (오류), `k`→`\hbar` (오류) |
| `UN19_1015_em_206.inkml` | `x 1 ( 2 + x 2 ( ) 2 + x ( ) 2 + x 4 ( ) ) 3` | `x 1 ( \wr \dashv x \eta ( ) w + x 1 ) \wr + x 4 ( ) ) \eta` | `2`→`\wr` (오류), `+`→`\dashv` (오류), `2`→`\eta` (오류), `2`→`w` (오류), `(`→`1` (오류), `2`→`\wr` (오류), `3`→`\eta` (오류) |
| `UN19_1019_em_258.inkml` | `P 2 ( x ) x 2 - a + b = x` | `\rho 2 \lceil \between ) \between \wr - d + \vartriangle = x` | `P`→`\rho` (오류), `(`→`\lceil` (오류), `x`→`\between` (오류), `x`→`\between` (오류), `2`→`\wr` (오류), `a`→`d` (오류), `b`→`\vartriangle` (오류) |
| `UN19_1022_em_312.inkml` | `b ( x ) 1 - 2 \pi 2 ( x + \theta ) + 1 + - 2 x - \theta ) 2 + 1 = 2 ( - [` | `b ( x ) \Lambda - 2 \pi 2 ( x \dashv \ominus ) \dashv 1 + - 2 x - \ominus ) 2 \dashv 1 \succ 2 ( - [` | `1`→`\Lambda` (오류), `+`→`\dashv` (오류), `\theta`→`\ominus` (오류), `+`→`\dashv` (오류), `\theta`→`\ominus` (오류), `+`→`\dashv` (오류), `=`→`\succ` (오류) |
| `UN19_1025_em_355.inkml` | `1 - 6 ( 8 + n + n 2 ) 9` | `n - 6 ( \with + \mathcal{M} \ast \mathcal{M} \wr ) 5` | `1`→`n` (오류), `8`→`\with` (오류), `n`→`\mathcal{M}` (오류), `+`→`\ast` (오류), `n`→`\mathcal{M}` (오류), `2`→`\wr` (오류), `9`→`5` (오류) |
| `UN19_1031_em_436.inkml` | `2 + y 2 z + z k - 1 x` | `\tau \dashv g 2 g + \partial A - 1 \ae` | `2`→`\tau` (오류), `+`→`\dashv` (오류), `y`→`g` (오류), `z`→`g` (오류), `z`→`\partial` (오류), `k`→`A` (오류), `x`→`\ae` (오류) |
| `UN19_1031_em_441.inkml` | `x x a - x b a b =` | `\varkappa \ae \varpropto \frown \ae b \mathcal{C} b \tau` | `x`→`\varkappa` (오류), `x`→`\ae` (오류), `a`→`\varpropto` (오류), `-`→`\frown` (오류), `x`→`\ae` (오류), `a`→`\mathcal{C}` (오류), `=`→`\tau` (오류) |
| `UN19_1034_em_491.inkml` | `N + 1 - N - 1 ( N - 3 - 2 - 3 ) C 1 = + 1` | `N + \wedge - N - \wedge ( N - 3 - 2 - 3 ) ( \wedge \supseteq \dag \wedge` | `1`→`\wedge` (오류), `1`→`\wedge` (오류), `C`→`(` (오류), `1`→`\wedge` (오류), `=`→`\supseteq` (오류), `+`→`\dag` (오류), `1`→`\wedge` (오류) |
| `UN19_1037_em_525.inkml` | `x 2 + x 2 1 + x 2 2 + x 2 3 1 0 =` | `x 2 + x \vartheta 1 \nmid x \partial = \nmid x \partial 3 1 0 \Xi` | `2`→`\vartheta` (오류), `+`→`\nmid` (오류), `2`→`\partial` (오류), `2`→`=` (오류), `+`→`\nmid` (오류), `2`→`\partial` (오류), `=`→`\Xi` (오류) |
| `UN19_1039_em_558.inkml` | `d - d y ( d w d y ) - 2 w ( w 2 - ) 0 y - 1 =` | `d - d J ( d \omega d J ) - 2 \omega ( \omega \L - ) 0 J - 1 =` | `y`→`J` (오류), `w`→`\omega` (오류), `y`→`J` (오류), `w`→`\omega` (오류), `w`→`\omega` (오류), `2`→`\L` (오류), `y`→`J` (오류) |
| `UN19_1047_em_685.inkml` | `r c x 2 1 + 2 + x 2 3 \sqrt{} x 2 =` | `r \subset x \wr 1 \dag 2 + x \wr 3 - x \wr I` | `c`→`\subset` (오류), `2`→`\wr` (오류), `+`→`\dag` (오류), `2`→`\wr` (오류), `\sqrt{}`→`-` (오류), `2`→`\wr` (오류), `=`→`I` (오류) |
| `UN19_1016_em_220.inkml` | `4 ( x 0 - y 0 ) - 2 x 0 ( - y 0 ) ( 1 + 1 0 = )` | `4 \| x \circ - ) 0 ) - 2 x O \| - y O ) \| 1 \shortrightarrow 1 \heartsuit = )` | `(`→`\|` (오류), `0`→`\circ` (동형:circle), `y`→`)` (오류), `0`→`O` (동형:circle), `(`→`\|` (오류), `0`→`O` (동형:circle), `(`→`\|` (오류), `+`→`\shortrightarrow` (오류) |
| `UN19_1051_em_746.inkml` | `( x + y ) n \sum k 0 n C k n x y k = = n - k` | `( \times + y ) \varkappa \sum K \partial n C k h \times y k = \simeq h - \$` | `x`→`\times` (동형:cross), `n`→`\varkappa` (오류), `k`→`K` (오류), `0`→`\partial` (오류), `n`→`h` (오류), `x`→`\times` (동형:cross), `=`→`\simeq` (동형:equality_approx), `n`→`h` (오류) |
| `UN19_1033_em_471.inkml` | `a 0 z 1 2 z 3 z 4 z 5 z` | `d O Z 1 2 \mathcal{Z} 3 \mathcal{Z} 4 Z S Z` | `a`→`d` (오류), `0`→`O` (동형:circle), `z`→`Z` (오류), `z`→`\mathcal{Z}` (오류), `z`→`\mathcal{Z}` (오류), `z`→`Z` (오류), `5`→`S` (동형:s_five), `z`→`Z` (오류) |
| `UN19wb_1108_em_997.inkml` | `- p - 1 + 2 p + 1 + X = - a a a p 0` | `- \rho - 1 + 2 \rho + 1 + \times = - \mathfrak{A} M \O \rho \fullmoon` | `p`→`\rho` (오류), `p`→`\rho` (오류), `X`→`\times` (동형:cross), `a`→`\mathfrak{A}` (오류), `a`→`M` (오류), `a`→`\O` (오류), `p`→`\rho` (오류), `0`→`\fullmoon` (동형:circle) |
| `UN19wb_1114_em_1085.inkml` | `\sum n e - \theta i \sum n i = e + \theta i i = 1 1` | `\Sigma R e - \mathcal{O} i \Sigma n i = e \bot \mathcal{O} i i = \Lambda \triangleleft` | `\sum`→`\Sigma` (동형:sigma_sum), `n`→`R` (오류), `\theta`→`\mathcal{O}` (오류), `\sum`→`\Sigma` (동형:sigma_sum), `+`→`\bot` (오류), `\theta`→`\mathcal{O}` (오류), `1`→`\Lambda` (오류), `1`→`\triangleleft` (오류) |

전체 식의 모든 토큰 차이는 동명 JSON의 `formulas` 배열에 보존했다.

## 2. 과적합 여부

| 근거 | Top-1 | Top-5 | n |
|---|---:|---:|---:|
| 외부 고정 holdout (collision-free) | 77.35% | 97.36% | 18330 |
| 직접수집 writer-LOO | 73.46% | 94.79% | 211 |
| CROHME test truth-group | 71.38% | 93.29% | 11991 |

- 현재 base 학습 loss: epoch 1 `3.0912` → epoch 8 `0.5015`.
- CROHME Top-1은 외부 holdout보다 `-5.98%p`, writer-LOO보다 `-2.08%p` 낮다.
- 동일 uniform-time full-data 실행에는 epoch별 validation 곡선이 없고, all-writer 보정 자료 자체 점수도 누수 방지를 위해 계산하지 않았다.
- 관련 selection 실행은 epoch 7→8에서 math Top-1이 `74.60%`→`74.92%`로 올랐지만 Top-5는 `97.31%`→`96.97%`로 소폭 하락했다. 정확히 같은 실행이 아니므로 보조 근거다.
- 판정: **심각한 암기형 과적합 증거는 없지만 과적합을 완전히 배제할 수도 없다.** 현재 격차는 데이터 도메인·클래스 분포 차이와도 일치한다.

## 3. 문자 분류기와 후단 레이어 병목

| 단계 | 결과 | 해석 |
|---|---:|---|
| 문자 분류 Top-1 | 8559/11991 (71.38%) | 분류기 자체 오류 존재 |
| Top-1 오답 중 truth가 Top-5에 있음 | 2628/3432 (76.57%) | 후단 재랭킹 가능 영역 |
| Top-1 오답 중 truth가 Top-5 밖 | 804/3432 (23.43%) | 분류기 candidate recall 문제 |
| truth-group 식 Top-1 exact | 133/849 (15.67%) | 현재 순위 그대로 |
| truth-group 식 Top-5 oracle | 527/849 (62.07%) | 문맥층의 이론적 후보 상한 |
| raw 그룹 exact | 306/769 (39.79%) | 그룹핑 독립 병목 |
| raw flat token exact | 78/769 (10.14%) | 2D 관계는 미포함 |
| current context exact 순증가 | 0식 | 현재 후단이 후보 잠재력을 회수하지 못함 |

결론적으로 오답 문자 3432개 중 2628개는 정답이 후보 안에 있다. 오답 716식 중 394식은 모든 정답이 Top-5 안에 있고, 322식은 적어도 하나가 Top-5 밖이다. 따라서 **현재 가장 큰 회수 가능 병목은 분류기 다음의 후보 재랭킹·문맥 확정층**이다. 다만 804개 문자는 후보에도 정답이 없어 분류기 개선이 필요하고, raw 그룹핑도 별도 병목이다.

## 4. 스트리밍 속도·형상 보전

| 항목 | 결과 |
|---|---:|
| 실제 timestamp | 없음 |
| 측정 stroke/segment | 16856/556967 |
| raw point를 고정 tick으로 재생한 속도 CV 중앙값 | 0.590 |
| raw point 속도 CV p90 | 0.975 |
| 호장 길이 재표본 속도 CV 중앙값 | 0.047 |
| 고정 tick 구간속도 배수 p05/p50/p95 | 0.35 / 0.90 / 1.86 |

좌표와 획 순서는 유지되므로 최종 형상이 찌그러지는 것은 아니다. 하지만 기록점 밀도를 실제 시간으로 오인해 재생하므로 조밀한 구간은 느리고 성긴 구간은 점프해 보일 수 있다. 또한 현재 모델은 `uniform-time`에서 `delta_t`를 강제로 균등화하므로 실제 필기속도를 사용하지 않는다.

### 제안

1. CROHME 평가는 48 Hz가 아니라 **호장 길이 진행률/점 진행률 스트리밍**으로 명명하고 평가한다.
2. 시각 재생은 각 획을 누적 호장 길이로 재표본해 일정한 기하 속도로 표시한다.
3. 실제 지연시간·속도 품질은 timestamp가 있는 직접수집/장치 입력만 실제 48 Hz로 보간해 별도 측정한다.
4. 분류기는 writer 속도 편차에 강한 현재 uniform-time 입력을 유지하되, 속도 신호를 쓰려면 timestamp 품질 gate가 있는 보조 채널로 분리한다.

DTW는 기준 궤적과의 정렬·유사도 분석에는 쓸 수 있지만, timestamp가 없는 CROHME의 실제 속도를 복원하지는 못한다. 재생 시계에는 단순한 단조 호장 길이 재표본이 더 적합하다.

## 한계

- 동형문자율은 선언한 형상군에 따라 달라지는 진단값이며 공식 CROHME Expression Rate가 아니다.
- raw 식 exact는 flat token 순서이며 2D relation exact를 포함하지 않는다.
- CROHME는 비상업 연구 진단 자료이며 제품 승격 근거가 아니다.
