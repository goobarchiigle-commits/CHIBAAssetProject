# Study81 — Cluster Diversification Hypothesis

**日付**: 2026-07-04
**追加BT**: **ゼロ**（禁止指示厳守）。Study80AのResearch Assets（`trade_dataset_v2.json`／`missed_candidates_full.json`／`opportunity_cost_dataset.json`／`correlation_dataset.json`）のみ使用。
**検証物**: `src/backtest/study81_cluster_diversification_2026-07-04.py` / `backtests/cluster_dataset.json` / `backtests/cluster_statistics.json` / `backtests/portfolio_cluster_report.json` / `backtests/hidden_cases.json`

## 検証対象の仮説

> 「max_positions=3が最適」なのではなく、「4銘柄目は既存3銘柄と同じクラスターに属するため期待値が増えない」

---

## Cluster ID設計（工夫）

- **macro_cluster**: Production既存の`src/strategy/cluster.py`（`CLUSTER_MAP_DEFAULT`・`get_cluster`）をそのまま再利用。新規発明ではなく、`composite_alpha_bt.py`が実際にCLUSTER_CAP判定に使っている既存のリスクグルーピング（cyclical_macro / defensive / growth_tech / real_asset / other）。「クラスター」概念として最も直接的かつ実装と整合する定義。
- **factor_cluster**: momentum_63d_pct・atr_pct・rsrの母集団tercile（Low/Mid/High）を組み合わせた副次ラベル（例: `momH_atrL_rsrH`）。
- **alpha除外**: Study80Aで判明した通りalpha_scoreは全件ほぼ0（degenerate、`alpha_df=None`のため無効な特徴量）のため、クラスタリング次元から除外（観測は保持）。
- `cluster_id = macro_cluster + "|" + factor_cluster`

macro_cluster分布（採用309+見送り607=916件）: cyclical_macro 491 / growth_tech 377 / defensive 36 / real_asset 12。

---

## 解析2: Cluster別 勝率/PF/Expectancy/Forward Return

| Cluster | 採用n | 勝率 | PF | Expectancy | 見送りn | 見送りforward_20平均 |
|---|---|---|---|---|---|---|
| cyclical_macro | 179 | 51.4% | 2.236 | +16,615円 | 312 | +2.79% |
| growth_tech | 105 | 55.2% | 1.785 | +10,309円 | 272 | +2.88% |
| defensive | 19 | 68.4% | 3.235 | +19,206円 | 17 | +6.51% |
| real_asset | 6 | 83.3% | 8.02 | +11,700円 | 6 | +2.95% |

**所見**: defensive・real_assetは好成績だがサンプル数が極端に小さい（n=6-19）ため統計的な結論は困難。主要2クラスター（cyclical_macro/growth_tech）は同水準のPF・Expectancyで大差なし。

---

## 解析3: Portfolio内Cluster集中率（実測 vs ランダム）

| 指標 | 値 |
|---|---|
| 実測平均最大クラスター集中度（同時保有銘柄群） | **73.7%** |
| ランダム帰無仮説平均（母集団分布ベースpermutation, N=10,000） | 72.16% |
| p値（実測≥ランダム） | **0.0661** |
| 5%水準で有意か | **NO（僅差で非有意）** |

**⚠ 重要な対比**: Study80Aの同分析（raw sector・13-14分類）ではp=0.0（有意）だったが、本解析の**macro_cluster（4分類）ではp=0.0661で非有意**。粗い4分類では母集団自体の偶然一致率が高く（4分類ならランダムでも約72%集中するのが自然）、統計的検出力が下がるため。**「クラスター」の粒度定義によって結論が変わる**ことが本解析で判明した重要な留保事項。

---

## 解析4: 4銘柄目「同クラスター vs 別クラスター」Forward Return比較（核心の検定）

CAP_MISS候補（既存3銘柄で枠が埋まっていた見送り）を、見送り時点で保有していた3銘柄と**同じmacro_clusterか否か**で二分:

| 群 | n | 平均forward_20 | 中央値forward_20 |
|---|---|---|---|
| 同クラスター | 366 | **+3.46%** | — |
| 別クラスター | 79 | **+1.71%** | — |

**Mann-Whitney U検定: p=0.1443（非有意）**

**⚠ 仮説と逆方向の結果**: 「同じクラスターだから期待値が増えない」という仮説が正しければ、同クラスター群のforward_20は別クラスター群より**低い**はず。しかし実測は**同クラスター群の方がむしろ高い**（+3.46% vs +1.71%）。統計的有意差はないが、方向性は仮説と逆。

---

## 解析5: Cluster単位Opportunity Cost

| Cluster | n | 平均forward_20 |
|---|---|---|
| cyclical_macro | 310 | +2.79% |
| defensive | 17 | +6.51% |
| growth_tech | 272 | +2.88% |
| real_asset | 6 | +2.95% |

Cluster間で機会損失の大きさに極端な差はない（defensiveのみやや高いがn=17と小さい）。

---

## 解析6: Hidden Factor探索（Cluster理論で説明できないケース）

**42件抽出**。内訳:
- **35件**: 同クラスターなのに大幅プラス（forward_20>+15%）— 「同クラスターは伸びないはず」という予測に反する
- **7件**: 別クラスターなのに大幅マイナス（forward_20<-10%）— 「別クラスターなら分散効果で守られるはず」という予測に反する

**所見**: 反例42件のうち**35件（83%）が「同クラスターなのに好成績」**という、仮説と逆方向のパターンに集中している。これは解析4の「同クラスター群の方が平均forward_20が高い」という結果と整合する一貫した傾向であり、単発の外れ値ではなく**構造的な反証パターン**。多くは市場全体・セクター全体が上昇するモメンタム相場（cluster内の複数銘柄が同時に強い時期）に該当し、「同じクラスターに属する＝悪い」のではなく「強いクラスターに複数の候補が同時発生している時こそ好機」という、仮説とは逆の力学が働いている可能性を示す。

---

## 結論: **棄却（REJECT）**

**仮説として提示された「4銘柄目は既存3銘柄と同じクラスターに属するため期待値が増えない」は、本解析データにより棄却する。**

### 判定根拠

1. **解析4（核心検定）**: 同クラスター群の見送り候補は、別クラスター群よりも**平均forward_20が高い**（+3.46% vs +1.71%）。統計的有意差はない（p=0.1443）ものの、方向性は仮説の予測と正反対。「同クラスターだから期待値が増えない」という主張を支持するデータは得られなかった。
2. **解析6**: Cluster理論からの逸脱42件中83%が「同クラスターなのに好成績」という、解析4と同方向の一貫した反証パターン。単発ではなく構造的。
3. **解析3**: macro_cluster（4分類）レベルでのポートフォリオ内集中度は、ランダムと統計的に有意差なし（p=0.0661）。Study80Aのraw sector（13-14分類）レベルでは有意だったこととの対比から、「クラスター」の粒度次第で結論が変わる不安定な指標であることが判明。

### ただし: 関連する「リスク集中」仮説は別（Study80Aの知見は本Studyでは否定されていない）

本Studyが棄却したのは**「同クラスター＝期待値(リターン)が低い」という狭義の仮説**である。Study80Aで確認された**「同日競合候補群の分散縮小率が24.8%（日をまたぐ無作為抽出67.3%と比べ大幅に低い）」という別の知見（リターンの大きさではなく、リターンの"ばらつき"＝リスクの相関構造に関する知見）は、本Studyのスコープでは検証対象にしておらず、否定も肯定もしていない**。「期待値は同クラスターでも下がらないが、リスク（分散）は下がりにくい」という両立する可能性がある — この切り分けはStudy81のスコープを超えるため、ここでは判定しない（未解決のまま次の検証対象として残す）。

**改善案は提示しない（本Studyの指示通り、説明のみ）。**
