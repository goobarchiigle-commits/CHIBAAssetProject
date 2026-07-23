# Study82E — PEAD Reverse Root Cause Audit（実行結果・post-mortem decomposition）

**日付**: 2026-07-22
**性格**: 実行結果（既存30,952件イベント台帳の再利用のみ・新規データ取得ゼロ・新規アルファ実装なし）。
**起案書**: `reports/study82e_proposal.md`
**入力**: `backtests/study82_phase_d_pead_events_2026-07-21.csv`（Study82 Phase D・30,952件）
**スクリプト**: `src/backtest/study82e_root_cause_audit.py`
**出力**: `backtests/study82e_root_cause_audit_2026-07-22.json` / `backtests/study82e_events_enriched_2026-07-22.csv`

**QA**: gap窓（r0_1/r2_5/r6_20/r21_40）のtelescoping積 vs Phase D既存`raw_return` — n=30,946, max_abs_diff=0.000000。完全一致（実装の正当性を確認）。

---

## 0. 背景（再掲）

```
Study82 Phase D: Positive − Negative spread = -0.538%（コスト後）
Positive群 t=5.3・Negative群 t=7.9（両群単独で強く有意）
```

Goal: **WHY did PEAD reverse?**（alpha探索ではない・successor設計ではない）

---

## 1. Tier1（正式判定基準）実測結果

### 1.1 Size（時価総額proxy: ShOutFY×Close、terciles）

| Bucket | Positive n / mean / t | Negative n / mean / t | Spread | Spread t |
|---|---|---|---|---|
| Small | 6,148 / +0.331% / 1.26 | 4,154 / +1.310% / 3.69 | **-0.979%** | -2.22 |
| Mid | 5,876 / +0.706% / 4.09 | 4,425 / +1.407% / 6.79 | **-0.701%** | -2.60 |
| Large | 5,910 / +0.847% / 5.67 | 4,391 / +0.767% / 4.31 | **+0.080%** | 0.34（非有意） |

**単調（monotonic=true, spread_rank_spearman=1.0, range=1.059pp）**。Large-capではスプレッドがほぼゼロ・非有意——逆転はSmall/Mid-cap集中現象。Large-capは「正常なPEAD方向へ回復」ではなく「効果消失（ノイズ圏）」。

### 1.2 Liquidity（ADV円ベース、entry前60営業日平均、terciles）

| Bucket | Spread | Spread t |
|---|---|---|
| Low | -0.681% | -1.88 |
| Mid | -0.617% | -1.94 |
| High | -0.316% | -1.09（非有意） |

単調（range=0.365pp）だがSizeの1/3程度の効果量。Size/Liquidityは自然に相関するため、独立要因というよりSize効果の従属的反映の可能性が高い。

### 1.3 Time period

| 期間 | Spread | Spread t |
|---|---|---|
| 2016-2019 | -0.409% | -1.33（非有意） |
| 2020-2022 | **-0.852%** | -2.40（有意・最大逆転） |
| 2023-2026 | -0.369% | -1.22（非有意） |

**非単調**（monotonic=false）。逆転は「直近のみ」ではなく2020-2022（コロナ禍〜回復期）で最大——CASE4「Only recent years reverse」は不成立。2023-2026は両群とも異例に高いリターン水準（Positive+1.954%/Negative+2.323%、t=10.51/9.76）——地合い全体の押し上げの中でも符号逆転は残存。

### 1.4 Market regime（TOPIX 200MA・トレイリング）

| Regime | Spread | Spread t |
|---|---|---|
| Above 200MA | -0.470% | -2.05（有意） |
| Below 200MA | -0.461% | -1.44（非有意、nが小さいため） |

**spread_range=0.009pp——実質ゼロ差**。逆転は強気/弱気レジームでほぼ同一マグニチュード。レジーム条件では説明できない=**regime-invariant**。

### 1.5 Gap absorption decomposition（全体）

| 窓 | Positive mean | Negative mean | Spread | Spread t |
|---|---|---|---|---|
| r0_1（発表日→翌日） | +0.032% | +0.207% | -0.175% | -2.90 |
| r2_5 | +0.145% | +0.394% | -0.249% | **-3.94（最大）** |
| r6_20 | +0.348% | +0.569% | -0.221% | -1.92 |
| r21_40 | +0.519% | +0.386% | **+0.133%** | 1.08（非有意・符号反転） |

逆転は発表直後から始まり（r0_1で既に有意）、r2_5でピーク、r6_20まで持続、**r21_40で符号が古典的PEAD方向に反転**（非有意だが方向転換）。「即時吸収で消滅」（Case5想定）ではなく「短中期(0-20d)の逆行が古典的ドリフトを圧倒し、後半(21-40d)にのみ弱く古典的方向が再浮上」という構造。

レジーム別分解（Above200MA/Below200MA）でも同一パターンが再現——regime条件下でも短期逆転→後半反転のタイミング構造自体は不変。

---

## 2. Tier2（診断専用・survival宣言には使用不可）

### 2.1 Surprise strength（|surprise_pct|パーセンタイル、方向別）

| Bucket | n | Spread | Spread t |
|---|---|---|---|
| Other（下位70%） | 21,666 | +0.050% | 0.25（ほぼゼロ） |
| Top30% | 6,190 | -1.311% | -2.90 |
| Top10% | 1,548 | -3.143% | -2.88 |
| Top5% | 1,238 | -1.910% | -1.53（n減少でノイズ増） |
| Top1% | 310 | **-7.691%** | -2.12 |

**spread_range=7.741pp——全軸中最大**。逆転は驚くほどサプライズ強度に比例して拡大。全体イベントの70%（「Other」）はスプレッド実質ゼロ=無情報。**逆転シグナルは強サプライズのごく一部に集中**——Phase Dの二値符号分類はこの70%の無情報イベントを有効シグナルと混合していた。

### 2.2 Quarter type

| Type | Spread | Spread t |
|---|---|---|
| 1Q | -0.735% | -2.14 |
| 2Q | -1.072% | -2.91 |
| 3Q | -0.736% | -1.87（**両群とも絶対リターンが負** = 異質パターン） |
| FY | **+0.512%** | 1.33（非有意だが唯一の符号反転） |

3Qは両群絶対リターンが負(-2.4%/-1.7%)という他四半期と質的に異なる挙動。FYのみ古典的方向への符号反転（情報量最大の開示ほど逆転が弱まる示唆・非有意）。

### 2.3 Pseudo quality（簡易代理指標・正式F-Score/GP/Aと非同一）

Pseudo GP/A（OP/TA代理）は単調（Low -0.987% → Mid -0.630% → High -0.062%）——Sizeと同型パターン（高収益性proxy=大型・安定企業に多い可能性が高く、Size効果と独立かは本監査だけでは分離不能）。Pseudo F-Score(5点簡易版)は非単調・n少（Low bucket n=729）で解釈保留。

---

## 3. Root-cause ranking

機械算出（spread_range_pct、Tier1のみ・恣意的重み付けなし）:

```
1. size           1.059pp
2. time_period    0.483pp
3. liquidity      0.365pp
4. market_regime  0.009pp（実質寄与なし）
```

定性統合（Tier2診断結果・gap分解を加味した解釈）:

1. **Size（Small/Mid集中）** — 最大の説明変数。Large-capで効果消失（有意差なし）。
2. **Event-timing構造（gap absorption）** — 逆転は0-20日で発生し20日超で減衰・反転。「即時吸収」ではなく「短中期の逆行→後半のみ弱い古典的方向」。
3. **サプライズ強度への比例（診断専用）** — 逆転幅は強サプライズで劇的に拡大（Top1%で-7.7pp）。全体の70%は無情報。**信号定義の粗さ（YoY EPS符号の二値分類）が、この集中効果を70%の無情報イベントで希釈**していた可能性が高い。
4. **Time period（非単調）** — 「直近のみ逆転」ではなく2020-2022最大。景気循環・ボラティリティ局面との関連は示唆されるが本監査の軸だけでは特定不能。
5. **Market regime** — 寄与ほぼゼロ。regime-invariant。
6. **Liquidity** — Sizeと同方向・同程度弱い。Sizeとの多重共線性が濃厚で独立要因と断定できない。

---

## 4. Final Decision Tree判定

```
Case1（全バケットで逆転残存）        : 不成立（Large-cap・r21_40・Otherサプライズ・FY四半期で消失/反転）
Case2（特定条件下でのみ生存）        : ★該当
Case3（Large-capのみ生存）           : 部分該当（ただし"正のPEAD復活"ではなく"効果消失=ゼロ"）
Case4（直近年のみ逆転）              : 不成立（2020-2022が最大）
Case5（短窓分解で逆転消滅=即時吸収） : 不成立（0-20dで逆転が最大化・むしろ持続）
```

**判定: CONDITIONAL SURVIVAL**

逆転はSmall/Mid-cap・強サプライズ・0-20日窓に集中する条件付き現象であり、全銘柄・全期間・全レジームに一様な「PEAD family = TERMINAL」を裏付ける証拠はない。同時に、Large-capでは効果自体が消失（有意な正のPEADへの回帰ではない）ため、単純な「大型株だけなら生きている」（Case3型の楽観）でもない。

---

## 5. Meta Question

**問い（優先度: 新規アルファ研究より上位）**: なぜwinner-buying系アノマリー（Study82 PEAD・Study83 TSMOM・Study95 CSモメンタム）が本環境で一貫して逆転するのか。

**判定: A寄り（B要素を伴う）**

- **Aを支持する証拠**: market regime分解でspread_range=0.009pp（実質ゼロ）——強気/弱気で逆転の強さが変わらない。これは「一時的な市場環境が偶然そう見せている」のではなく、構造的・持続的な現象であることを示す。またサプライズ強度への強い比例関係（Top1%で-7.7pp）は、統計ノイズでは説明しにくい経済的な規則性（強い悪材料ほど短期反発が強い=oversold-bounce/distress-related overreaction correctionという確立された市場ミクロ構造）に一致する。日本のSmall/Mid-cap株は現在、決算イベント直後0-20日において平均回帰（オーバーシュート後の戻り）が古典的PEADドリフトを恒常的に上回っている、というのが最も整合的な解釈。
- **Bの要素（無視できない）**: Phase Dの二値YoY EPS符号分類は、全イベントの70%（弱サプライズ）を実質無情報のまま強サプライズ群と混合しており、集計スプレッドの解釈力を落としていた。四半期タイプ（3Q特異・FY符号反転）を区別しない設計も、異質な現象を単一の数値に潰していた。これは「逆転という事実」自体を作り出したわけではないが（Positive/Negative両群の個別有意性はPhase D時点でも確認済み）、**Phase Dの「FAIL」という言葉が意味する解像度**を粗くしていた。

結論: **逆転は主に(A)日本Small/Mid-cap株の決算後短期オーバーリアクション訂正という実在の市場現象であり、(B)Study82実装の粗い信号定義がその解釈をさらに悪化させた、という二層構造**。どちらか一方では説明が閉じない。

---

## 6. Deliverables サマリ

1. **Root-cause ranking**: Size > Gap-timing構造 > サプライズ強度(診断) > Time period > Liquidity > Market regime(ほぼ寄与なし)
2. **逆転の説明**: Small/Mid-cap・強サプライズ・0-20日窓に集中する短期オーバーリアクション訂正が、Large-cap/弱サプライズ/21-40日窓で観測される弱い古典的PEADドリフトを圧倒している。Phase Dの粗い二値信号定義がこの構造を希釈・不明瞭化した。
3. **推奨区分**: **CONDITIONAL SURVIVAL**

---

## 7. 禁止事項遵守確認

実装提案なし。Study103再実行なし。新規momentum study提案なし。ロードマップ変更なし。新規データ取得なし（既存3キャッシュのみ結合）。successor hypothesis の具体設計は行っていない——root cause確定（本書§3-4）を受けて着手するかはユーザー判断待ち。

*作成: CLD (Fable 5)・2026-07-22。実行済み・既存キャッシュのみ再利用・新規データ取得なし。*
