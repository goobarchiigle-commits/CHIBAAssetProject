# Study95E — Momentum Reverse Decomposition（実行結果）

**日付**: 2026-07-22
**性格**: 実行結果（Study95本体のfactor定義・universe・統計手法をそのまま再利用・新規データ取得ゼロ・新規アルファ実装なし）。
**入力**: `backtests/study95_cs_momentum_factor_level.json`（Study95本体・FAIL_ZERO_SPREAD）
**スクリプト**: `src/backtest/study95e_momentum_reverse_decomposition.py`
**出力**: `backtests/study95e_momentum_reverse_decomposition_2026-07-22.json` / `backtests/study95e_panel_enriched_2026-07-22.csv`
**QA**: panel行数=108,895——Study95本体（`reports/study95_cs_momentum_factor_level.md`記載値）と完全一致。factor計算ロジックの再利用が正しいことを確認。

---

## 0. 目的

```
Determine whether Study82 reversal generalizes to cross-sectional momentum.
```

Study82型のSize/Liquidity/Time period/Market regime分解を12-1モメンタムQ10-Q1スプレッドに適用。Holding horizonのみStudy95本体が既に1M/3M/6M/12Mで算出済みのため転用。

---

## 1. 分解結果（12M horizon中心・全horizon値は添付JSON参照）

### 1.1 Size

| Bucket | 12M spread | NW-t | hit ratio |
|---|---|---|---|
| Small | -1.22% | -0.26 | 0.537 |
| Mid | -1.07% | -0.23 | 0.516 |
| Large | -1.74% | -0.31 | 0.442 |

**spread_range=0.67pp・非単調（monotonic=false）**。全bucketで弱く負・**どこにも有意差なし**。Study82Eで見られた「Large-capで効果消失」パターンは**再現しない**——momentumの負スプレッドはSizeに依存しない。

### 1.2 Liquidity（ADV20 terciles）

| Bucket | 12M spread | NW-t | hit ratio |
|---|---|---|---|
| Low | **+1.76%** | 0.40 | 0.558 |
| Mid | -1.32% | -0.32 | 0.495 |
| High | -6.10% | -0.89 | 0.463 |

単調（monotonic=true・Low→High単調悪化）・spread_range=7.86pp（4軸中2位）。ただし**全bucket非有意**。注目点: **Low流動性bucketのみ符号が正**——Study82Eの「低流動性ほど逆転が強い」パターンとは**逆方向**。

### 1.3 Time period

| 期間 | 12M spread | NW-t | hit ratio |
|---|---|---|---|
| 2016-2019 | -0.07% | -0.01 | 0.414 |
| 2020-2022 | -8.79% | -1.04 | 0.389 |
| 2023-2026 | **+4.82%** | 0.99 | **0.733** |

**spread_range=13.61pp（4軸中最大）**・非単調。2020-2022で最悪、2023-2026で符号が正に転換（非有意だがhit ratio=73%は他期間より顕著に高い）。Study82Eの「2020-2022最大逆転」と**部分的に類似**するが、Study82では全期間を通じ符号が負のまま(2023-2026も-0.369%)だったのに対し、momentumは2023-2026で符号自体が反転——**完全な構造一致ではない**。

### 1.4 Market regime

| Regime | 12M spread | NW-t |
|---|---|---|
| Above200MA | -1.94% | -0.40 |
| Below200MA | -1.63% | -0.52 |

**spread_range=0.32pp——実質ゼロ差**。Study82Eのregime-invariance（range=0.009pp）と**同型の結果**。市場レジームはどちらの逆転/ゼロスプレッド現象も説明しない。

### 1.5 Holding horizon（Study95本体より転用・再計算なし）

| Horizon | 年率spread | NW-t |
|---|---|---|
| 1M | -4.60% | -0.70 |
| 3M | -3.90% | -0.76 |
| 6M | -5.50% | -1.10 |
| 12M | -1.83% | -0.37 |

全horizonで負符号は一貫するが**全て非有意**。Study82Eのgap分解で見られた「後半windowで符号反転」のような明確なタイミング構造は**momentumには見られない**——一貫して弱いマイナス、それだけ。

---

## 2. Root-cause ranking（spread_range_12m基準・機械算出）

```
1. time_period    13.61pp
2. liquidity       7.86pp
3. size            0.67pp
4. market_regime   0.32pp
```

**ただし全軸・全bucketで|NW-t|>2は一件も無し**（もっとも近いのはaxis_period外の1M/3M horizon一部だが、これも本分解の主軸12Mでは非有意）。Study82Eとの決定的な違い: **Study82は「強く有意な逆転を、何が説明するか」という分解だったのに対し、Study95Eは「そもそも有意な効果が存在しない（FAIL_ZERO_SPREAD）」ことの再確認**——分解すべき強い信号自体が最初から無い。

---

## 3. Decision（ユーザー指定基準の適用）

```
If Study95 shows same structure as Study82:
    establish Japanese Small-cap Mean Reversion Hypothesis
Otherwise:
    treat PEAD reversal as event-specific phenomenon
```

**判定: Otherwise（構造不一致）——PEAD逆転はevent-specific phenomenonとして扱う**

根拠（3点、いずれも「同一構造」を否定）:

1. **Size依存性が正反対**: Study82はSize単調・Large-capで効果消失(spread_t=0.34→ノイズ圏)。Study95はSize方向でほぼフラット(range=0.67pp)——小型株固有の逆転が主因なら momentum でも Small で最も負のスプレッドが出るはずだが、実際はLargeが最も負（-1.74%）でSmallが最も浅い(-1.22%)。**符号関係が逆**。
2. **Liquidity依存性も正反対**: Study82は低流動性で逆転最大。Study95は**低流動性のみ唯一の正スプレッド**——真逆の方向。
3. **統計的強度が根本的に異なる**: Study82は個別群t=5.3/7.9・多くのbucketでspread_t<-2の強い有意性。Study95は**どのbucketも有意水準に到達しない**——「逆転」と呼べる強さの現象がそもそも存在しない（原判定FAIL_ZERO_SPREADと整合）。

共通点は**market regime-invarianceのみ**（両Studyともregime分解でspread_range≈0）——これは「日本株のwinner-buying系構築が市場サイクルに関係なく弱い/逆行する」という限定的な共通性を示すが、その原因（Small-cap短期オーバーリアクション）がmomentumにも及んでいる証拠にはならない。

---

## 4. 禁止事項遵守確認

新規アルファ探索なし。最適化なし。新規データ取得なし（Study95本体と同一4キャッシュ + fins_summary ShOutFY結合のみ・新規API呼び出しゼロ）。Clenow slope×R²は本分解のスコープ外（momentum一本に限定・追加要望があれば別途）。

*作成: CLD (Fable 5)・2026-07-22。実行済み・既存キャッシュのみ再利用・新規データ取得なし。*
