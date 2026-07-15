# D_ATR_EQ Study75-Universe再ベースライン（Study76前提工程）— 結果

**日付**: 2026-07-13
**位置づけ**: `reports/study76_execution_plan.md`が定義する「Study76」（Clenow純正ベンチマーク・
D_ATR_EQ全面簡略化）とは別物。本結果はStudy76が比較対象として必要とする前提工程
「D_ATR_EQをStudy75 Universe上でfresh run再測定する」（同canon §3/§5）の実装・実行結果。

成果物: `src/backtest/study76_datr_eq_universe_c_rebaseline.py` /
`backtests/study76_datr_eq_universe_c_rebaseline_2026-07-13.json` /
`backtests/dynamic_rsr42_membership_2026-07-13.json`

**最重要指標**: **Δ_dynamic = RunB(Dynamic RSR42) − U0(静的hindsight RSR42) = IS -25.17pp / OOS +62.27pp**

---

## 設計要約

hindsight静的RSR42リストを、Study75AのUniverse C（PIT・rule-based・月次再適用）から各月T-1時点の
トレイリング・コンポジットリターン（IBD式12ヶ月加重リターン・`rsr.py::calc_composite_return`）
上位42銘柄を機械的に選ぶ「**Dynamic RSR42**」（月次固定42名ローテーション）に置換した。
選抜後の42名プール内でのみRSRパーセンタイル・`min_rsr>=75`・`dyn_rsr42_bear_rs0`のTop30/Bear20を
計算するため、本番と全く同じ解像度でゲートが動作する（**エンジンコード無改変**）。
セクターは本日稼働開始した`database/market/master/companies.parquet`の実分類を使用（E1のような
疑似セクター回避策は不要）。RunAは実行せず、既存のStudy75B U3
（`backtests/study75_survivorship_2026-07-11.json`）をNegative Control参照専用に引用する。

---

## A. IS/OOS成績

| 指標 | RunB IS (2018-2024) | RunB OOS (2025) |
|---|---|---|
| CAGR | **-16.46%** | **+61.29%** |
| Sharpe | -0.175 | 1.172 |
| Calmar | -0.194 | 1.548 |
| MaxDD | **-85.02%** | -39.59% |
| n_trades | 282 | 51 |
| avg_exposure | 42.6% | 40.5% |
| avg_simultaneous_holdings | 1.56 | 1.95 |
| win_rate | 32.6% | 35.3% |

WF5fold（単年OOS窓・2020-2024）: **2/5 PASS**（2023 +25.07%・2024 +96.22%のみ正）。
2020 -27.21% / 2021 -19.37% / 2022 -38.87% はいずれも大幅負。avg_cagr(WF)=+7.17%。

---

## B. 既存ベースラインとの比較

| ベースライン | ソース | IS CAGR | OOS CAGR |
|---|---|---|---|
| Official Production（yfinance基盤・M1適用後） | `study_m1_production_update_2026-07-04.json` | 12.22% | 11.42% |
| J-Quants U0（静的hindsight RSR42） | `study75_survivorship_2026-07-11.json` | 8.71% | -0.98% |
| **RunB Dynamic RSR42（本結果・主分析）** | 本Study | **-16.46%** | **+61.29%** |
| RunA/U3参考（Negative Control・使用非推奨） | 既存・二重汚染 | 約-30.6%（参考記録専用） | — |

**Δ_dynamic = RunB − U0**: **IS -25.17pp / OOS +62.27pp**

---

## C. アトリビューション（参考情報・主結論はΔ_dynamicのみ）

- **Δ_dynamic（主指標）**: hindsight selectionの損失（IS）またはhindsight selectionの脆さ（OOS）を
  直接示す。IS期間はhindsight-RSR42が圧倒的優位（-25.17pp）——RSR42のリスト自体が
  `selection_period: 2018-2024_backtest_universe`としてこの期間の成績を見て選ばれている以上、
  当然予想される方向。**OOS期間は逆転し、Dynamic RSR42がU0を+62.27pp上回る**——これは
  Study75C E1が既に示したOOSパーセンタイル退行（95%→70%）と整合的な追加証拠であり、
  「hindsight選定の優位性は選定窓の外では消滅または反転する」という仮説を強く支持する。
- **参考（RunA=U3との比較・Negative Control）**: U3のIS CAGR約-30.6%はパーセンタイル歪み+
  セクターキャップ崩壊バグの二重汚染を含むため定量比較には使わないが、RunB（-16.46%）が
  U3よりは大幅に良いという事実は、「fixed-count選抜による歪み補正」自体には一定の効果があった
  ことを示唆する（両者とも同じUniverse C・同じ全体傾向を共有するはずだが、RunBの方が汚染源を
  除去した分だけ改善している）。ただし定量的な寄与分解はRunAの汚染のため不可能。

---

## D. 容量診断

| 指標 | IS | OOS |
|---|---|---|
| avg_candidates | **0.45** | — |
| cap_saturation_rate_pct | 17.2% | — |
| days_at_max_positions | 294 | — |
| dyn_universe_excluded_count | 818 | — |
| avg_idle_cash_ratio_pct | 22.4% | — |
| skip_stats | sector_cap=4 / cluster_cap=18 / bear_adaptive=8 / gross_exposure=1 | — |
| missed_by_cap_count | 127 | — |
| rejected_by_lot_count | 328 | — |

**月次membership turnover**: 平均**44.57%/月**（銘柄別在籍月数の中央値=**3.0ヶ月**、平均=4.08ヶ月、
延べユニークシンボル数=1,112）。42名中平均約19銘柄が毎月入れ替わる計算——RSR42の静的性質とは
対照的に極めて回転の速いユニバースである。

**⚠️特筆すべき所見**: `avg_candidates=0.45`（1日あたり平均0.45候補しか存在しない=非常に薄い候補層）
と`max_dd=-85.02%`（IS）は、本Studyの明示的な異常判定基準（`avg_simultaneous_holdings≈1`・
`exposure≈0`・`trade_count極小`・membership件数異常・lookahead）には抵触しないため結果として
採用したが、production/U0/E1のいずれとも比較にならない極端な値である。トレイリング12ヶ月
リターンのみによる月次42名選抜は、RSR42が（選定過程は不明ながら）暗黙に持っていたであろう
質的フィルタリングを欠いており、高回転（月44.57%）と組み合わさることで個別銘柄の急落に対する
分散が薄くなっている可能性が高い。**バグではなく設計（trailing-return-only rankingの限界）による
所見の可能性が高いが、断定はしない**——後述の推奨リランでの追加検証を推奨する。

---

## E. 判定

**1. Dynamic universeは新しい正準基盤になり得るか**: **現時点ではNO**。IS期間で-25.17pp
（大幅な劣化）・MaxDD-85%は単独では採用不可能な水準。ただしOOSでは逆に大幅優位（+62.27pp）
であり、単一期間の結果で断定するには時期尚早——**WF5foldで2/5 PASSかつfold間分散が極端
（-38.87%〜+96.22%）**という事実自体が「回転の速いtrailing-momentum選抜は構造的に不安定」
という懸念を裏付ける。

**2. Study74 BLACKは維持されるか**: **維持**。今回の測定はStudy74（資本スケーリング）とは独立変数
であり、Study74の資本弾力性に関する構造的結論（lot丸め・max_positions天井）に影響を与える
新情報は含まれない。むしろ「正直なCore期待値はさらに不確実性が高い」という方向を補強する
（Study75C同様、CP1フォールバック目標の再アンカー必要性を追加で示唆）。

**3. 現行アーキテクチャの何らかの構成要素は honest universe下で生存するか**: **部分的に判定不能・
追加検証が必要**。エンジン自体（Exit・リスク・breadth・dyn_rsr42_bear_rs0）は無改変のまま
機能した（skip_stats等が正常な範囲で記録されている）ことから、**アーキテクチャ自体は
"honest universe"下でも壊れずに動作する**ことは確認できた。しかし成績の質（Sharpe -0.175 IS・
win_rate 32.6%）は、hindsight選定なしでは現行アーキテクチャの優位性の大半が消える可能性を
示す。「ranking rule自体（trailing composite return）の質」の問題なのか「アーキテクチャ
（Exit/risk/breadth）の問題」なのかは本Studyだけでは分離できない——次段階の検証課題。

---

## 推奨事項（次アクション・ユーザー決裁待ち）

1. `avg_candidates=0.45`・`max_dd=-85.02%`の原因診断（個別トレードログの精査・特定銘柄への
   集中崩れがないかの確認）を、Dynamic RSR42を破棄する前に実施する価値がある。
2. Dynamic RSR42の選抜ルール自体（trailing composite returnのみ）を、Study76canon本来の
   Clenowスコア（slope×R²・急落除外ギャップフィルター付き）に差し替えた場合の感度を確認する
   （「hindsight喪失の影響」と「選抜ルールの質」を分離するため）。
3. RunAの汚染除去版（実セクター使用）を別途fresh runし、Negative Controlをより厳密な形で
   再構築することで、C節のアトリビューション分解の精度を上げる。
4. 本結果とcanon本来のStudy76（Clenow純正ベンチマーク）を実施し、「複雑性の対価」を独立して測定する。
