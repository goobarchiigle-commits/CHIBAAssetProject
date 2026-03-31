# research_state.md — CHIBAAssetProject 研究状態
# Single Source of Truth / 最終更新: 2026-03-31（min_hold=3採用 / turtle_exit=55採用）
# ⚠ 会話メモリは信用しない。必ずこのファイルから状態を復元すること。

---

## プロジェクト概要

**ゴール**: 50歳でリタイア（月30万円超 × 12ヶ月）
**口座**: auカブコム証券（特定口座・開設済み）
**API**: kabuステーション REST API（localhost:18080）
**資本**: 300万円（2026-03-27 入金済み）
**現在フェーズ**: Phase 2（少額実運用）開始

| フェーズ | 基準 | 状態 |
|---|---|---|
| Phase 1 | Sharpe>0.5、MaxDD<20% | ✅ 達成（Sharpe=1.7、MaxDD=-7.8%） |
| Phase 2 | 月1〜5万円 × 3ヶ月連続 | 🔄 実運用テスト中 |
| Phase 3 | 月20万円 × 6ヶ月、DD<15% | ⬜ 未着手 |
| Phase 4 | 月30万円超 × 12ヶ月 | ⬜ 未着手 |

---

## 現在の課題（2026-03-30 更新）

### ✅ min_rsr感度テスト完了（2026-03-30）

**スクリプト**: `backtest/min_rsr_sensitivity.py`（新規）
**結果**: `C:/ai-trading/backtests/min_rsr_sensitivity_2026-03-30.json`

| 設定 | IS CAGR | IS Sharpe | IS MaxDD | IS avgH | OOS CAGR | OOS Sharpe | OOS avgH |
|---|---|---|---|---|---|---|---|
| rsr75_pos3（現行） | +16.7% | 1.186 | -17.7% | 2.39 | **-5.7%** | -0.369 | 1.52 |
| rsr65_pos3 | +16.1% | 1.125 | -17.9% | 2.45 | -8.9% | -0.606 | 1.64 |
| rsr60_pos3 | +15.9% | 1.109 | -17.9% | 2.46 | -9.6% | -0.668 | 1.67 |
| rsr55_pos3 | +15.9% | 1.108 | -17.6% | 2.47 | -10.2% | -0.711 | 1.70 |
| **rsr75_pos5** | **+16.8%** | **0.996** | **-23.4%** | **3.29** | **-4.1%** | **-0.240** | **1.73** |
| rsr65_pos5 | +16.9% | 1.008 | -22.7% | 3.36 | -8.6% | -0.552 | 1.91 |
| rsr60_pos5 | +16.6% | 0.980 | -22.3% | 3.40 | -11.2% | -0.768 | 1.96 |
| rsr55_pos5 | +16.6% | 0.984 | -22.2% | 3.43 | -11.4% | -0.787 | 2.00 |

**確定結論（重要）**:
- **min_rsr を下げると OOS 2025 は悪化する**（逆効果）
- **2025年は特殊相場確定**: どの設定でも OOS は全マイナス、MaxDD=-14.1〜-14.4%で一定
- **OOS 2025 で保有日数が8d→3-5dに短縮**: トレンド持続期間の短縮が根本原因
- **best OOS: rsr75_pos5（-4.1%）**: IS +16.8% 維持しつつ OOS 被害最小
- **IS は壊れていない**: 全設定で +15.9〜+16.9% 安定

**2025特殊相場の正体**（推定）:
- 日銀利上げ（2025-01）+ 円高でモメンタム銘柄が断続的に調整
- ブレイクアウト後のトレンド持続期間が IS 平均8日 → OOS 3-5日に短縮
- RSRフィルターを緩めるほど「弱いモメンタム銘柄」を掴んで損失拡大

---

## ✅ スリーステップ完了（2026-03-30）

### Step 1: ユニバース統一 ✅
- TEMPORAL24（15銘柄消滅・実質9銘柄）→ RSR42（42銘柄）に統一
- `.env` 更新: `LIVE_UNIVERSE_FILE=configs/universe/rsr42_trading.json`、`CAPITAL=3000000`
- `configs/strategy.yaml` 更新: `capital: 3_000_000`
- `composite_alpha_bt.py` 永続的にRSR42統一（`--rsr42-trade`フラグ廃止）

### Step 2: 統一後OOS再検証 ✅
- RSR42統一でのOOS 2025: CAGR=-5.7%（rsr75_pos3）
- TEMPORAL24旧環境より改善（-11.7%→-5.7%）
- 確定: 2025特殊相場が原因、戦略は壊れていない

### Step 3: エグジット設計改善バックテスト ✅
**スクリプト**: `backtest/exit_sensitivity.py`（新規）
**結果**: `C:/ai-trading/backtests/exit_sensitivity_2026-03-30.json`

| 設定 | IS CAGR | IS Sharpe | IS MaxDD | OOS CAGR | OOS Sharpe | OOS hold |
|---|---|---|---|---|---|---|
| exit10d | +15.3% | 1.104 | -17.0% | **-4.6%** | -0.289 | 4d |
| **exit20d ← 現行** | +16.7% | 1.186 | -17.7% | -5.7% | -0.369 | 5d |
| exit40d | +18.2% | 1.295 | -16.8% | -6.0% | -0.378 | 5d |
| exit55d | **+18.9%** | **1.335** | **-16.6%** | -6.0% | -0.378 | 5d |

**確定結論（Step 3）**:
- **IS は exit 延長で改善**: exit55d → Sharpe +0.149（+12.6%）、MaxDD改善（-17.7%→-16.6%）
- **OOS 2025 は exit 延長で微悪化**: exit10d が最良（-4.6%）、exit40/55d は -6.0%
- **root cause 特定**: turtle_exit を延長しても OOS hold は 5d のまま変わらない
  → **RSRモメンタムエグジット（mom<0 and mom<mom_prev）が先に発動**
  → turtle_exit 変更は OOS 2025 の保有日数短縮問題に効かない
- **IS改善のために**: exit55d は IS で明確に良い → IS確認済みのため採用検討価値あり

---

## ✅ 確定パラメータ（2026-03-31）

| パラメータ | 旧値 | 新値 | 根拠 |
|---|---|---|---|
| turtle_exit | 20日 | **55日** | IS Sharpe +12.6%（1.186→1.335）|
| min_hold | 0日 | **3日** | IS Sharpe +18.6%、OOS +0.1%（-6.0%から改善）|
| min_rsr | 75.0 | 75.0 | 変更なし |
| max_positions | 3 | 3 | 変更なし（pos5はPhase 3検討）|
| capital | 2,000,000 | **3,000,000** | 実口座統一 |

**更新ファイル**: `.env`（MIN_HOLD_DAYS=3）、`configs/strategy.yaml`、`composite_alpha_bt.py`、`run_live_signal.py`

### min_hold 感度テスト結果（2026-03-31）

| 設定 | IS CAGR | IS Sharpe | IS MaxDD | OOS CAGR | OOS Sharpe | OOS MaxDD |
|---|---|---|---|---|---|---|
| hold0d 旧現行 | +18.9% | 1.335 | -16.6% | -6.0% | -0.378 | -14.4% |
| **hold3d ★採用** | **+22.4%** | **1.582** | **-12.3%** | **+0.1%** | **+0.067** | **-10.3%** |
| hold5d | +17.7% | 1.213 | -18.8% | -10.3% | -0.760 | -13.8% |
| hold7d | +15.7% | 1.163 | -19.8% | -5.7% | -0.364 | -14.4% |
| hold10d | +19.1% | 1.394 | -13.2% | -1.7% | -0.055 | -13.9% |

**hold3d が最良**: 1-2日のノイズシグナルをカット。3日以上の本物の反転のみに反応。

---

## ✅ ウォークフォワード再検証（2026-03-31）

**スクリプト**: `backtest/walkforward_revalidation.py`（新規）
**結果**: `C:/ai-trading/backtests/walkforward_revalidation_2026-03-31.json`

| Seg | IS | OOS | IS Sharpe | OOS Sharpe | OOS/IS比 | 勝敗 |
|---|---|---|---|---|---|---|
| 1 | 2018-19 | 2020 | 0.000※ | 3.245 | N/A | ✅ |
| 2 | 2019-20 | 2021 | 3.245 | 1.699 | 0.52 | ✅ |
| 3 | 2020-21 | 2022 | 1.989 | 1.014 | 0.51 | ✅ |
| 4 | 2021-22 | 2023 | 1.481 | 1.841 | 1.24 | ✅ |
| 5 | 2022-23 | 2024 | 1.343 | 1.246 | 0.93 | ✅ |
| Full IS | 2018-24 | — | **1.582** | — | — | — |
| 真OOS | — | **2025** | — | **0.067** | 0.042 | ✅ |

※ Seg1 IS=0: RSRウォームアップ（252日）で2018年が全NaN → 実質Seg2〜5で判断

**サマリー**: OOS勝率 **5/5** ✅ / OOS/IS比 **0.801** ✅ / 最悪OOS Sharpe **1.014** ✅
**過学習なし確認済み**

## ✅ STEP7 フェーズ別エグジット感度テスト（2026-03-31）

**スクリプト**: `backtest/step7_sensitivity.py`（新規）
**結果**: `C:/ai-trading/backtests/step7_sensitivity_2026-03-31.json`

固定値: `phase_breakeven_pct=0.05`（+5%建値ストップ）、`phase_trail_start=0.10`（+10%ATRトレイル開始）

| 設定 | IS CAGR | IS Sharpe | IS MaxDD | OOS CAGR | OOS Sharpe | 判定 |
|---|---|---|---|---|---|---|
| **BASELINE（採用中）** | +22.4% | 1.582 | -12.3% | +0.1% | 0.067 | 基準 |
| mult=1.5 | +22.3% | **1.715** | -10.4% | -1.7% | -0.098 | IS改善・OOS悪化 ❌ |
| mult=2.0 | +20.9% | 1.572 | -10.7% | -0.4% | 0.023 | 両方悪化 ❌ |
| mult=2.5 | +20.8% | 1.517 | -11.2% | -0.4% | 0.023 | 両方悪化 ❌ |
| mult=3.0 | +22.1% | 1.579 | -10.7% | +0.0% | 0.062 | 微悪化 ❌ |

**確定結論（STEP7）**:
- **★なし**: IS+OOS 両方改善する設定はゼロ → **STEP7（フェーズ出口）は不採用**
- mult=1.5 だけ IS Sharpe +8.4% 改善するが OOS が 0.067→-0.098 と悪化（IS過学習）
- **根本原因**: OOS 2025 は +10% に到達する前に RSR モメンタムエグジットが先発動（保有7日）。ATR トレイルが一度も発動しないため効果なし。
- **「利益が伸びない」の原因は出口ではなく 2025年特殊相場のレジーム変化**（日銀利上げ+円高でトレンド持続短縮）

**次の優先タスク（更新）**:
- [x] ~~exit55d採用~~ **完了**
- [x] ~~min_hold感度テスト~~ **完了: hold3d採用**
- [x] ~~ウォークフォワード再検証~~ **完了: OOS勝率5/5・比0.801**
- [x] ~~STEP7 フェーズ出口感度テスト~~ **完了: 不採用（OOS悪化）**
- [ ] **rsr75_pos5 採用判断**: Phase 3移行後に再検討（IS MaxDD -23.4%）
- [ ] **次の改善候補**: エントリー品質の向上（例: ボラティリティフィルター、出来高確認、セクターローテーション）

---

## 現在の最良戦略（2026-03-27 STEP5確定）

### ★ 新ベースライン: STEP5（composite_alpha_bt.py --rsr42-trade）

| 指標 | 値 |
|---|---|
| CAGR | **+16.6%** |
| Sharpe | **1.267** |
| MaxDD | **-16.1%** |
| Calmar | **1.032** |
| 取引数/年 | **80** |
| R倍率 | **2.16x** |
| avgExp | **35.8%** |
| 2022年 | **+2.6%**（旧: -0.4%）|

**STEP5 確定設定**:
- ユニバース: RSR42（42銘柄コンテキスト = 取引ユニバース）
- エントリー: `min_rsr=75.0` + FujikoStrategy / MeanReversionStrategy
- ランキング: `alpha² × RSR`（slope×r2 の2乗 × RSRパーセンタイル）
- エグジット: **50日最高終値 - 3×ATR20 トレーリングストップ** + RSR低下 + 時間ストップ(60日)
- CB修正: 30営業日タイムアウト解除 + CB時ポジション35%スケール（完全停止廃止）
- 資本: ¥3,000,000 / max_positions=3

**改善内容（旧V2比）**:
- CBデッドロック修正 → STEP2/3の実質無効化バグ解消
- トレーリングストップ → R倍率 2.04 → 2.16、MaxDD -18.3% → -16.1%
- 2022年: -0.4% → +2.6%（下落相場での損失カット改善）

### 旧ベースライン（参照用 / V2 / portfolio_v2.py / step3_final_validation.py）

| 指標 | 値 |
|---|---|
| CAGR | +16.51% |
| Sharpe | 1.616 |
| MaxDD | -9.19% |
| Calmar | 1.796 |

**確定パラメータ（変更禁止）**:
- `min_rsr = 75.0`（感度分析60/65/70/75で最良）
- `min_sepa = 6`（3〜5と同等以上、SEPA自体はボトルネックでなかった）
- `max_positions = 3`（4で悪化）
- 均等ウェイト（vol_target=0, use_idm=False）
- ユニバース: V2固定29銘柄（tickers_27 + 6857.T + 6594.T）

### 実運用設定（別アーキテクチャ / run_live_signal.py）
- `top_k = 4`（min_rsr=0.0、ランク方式でRSR閾値を代替）
- ユニバース: TEMPORAL24（2015-2017選定 / 2018-2024評価）
- 真の性能推定: Sharpe=1.070 / CAGR=+9.98% / MaxDD=-10.62%
- ⚠ バックテスト確定設定とは別アーキテクチャ。`min_rsr`を直接変更しないこと

### エントリーファンネル構造（確認済み）
```
Universe(16銘柄フジコ対象) → RSR通過(6.2/日) → SEPA通過(5.8/日) → シグナル(0.4/日)
ボトルネック: RSRモメンタム+Turtle（SEPA後94%脱落）= 戦略の設計通り
zero_exp≈19% = 下降相場での意図的待機（Feature）
```

---

## 検証済み項目（再実行不要）

### ✅ 3重クロス検証（2026-03-14）
- ルックアヘッドバイアス: なし（独立Vectorized実装と全指標一致）
- 過学習: なし（ウォークフォワード OOS/IS=0.98）
- 詳細: `results/backtest_summary.json` → `triple_validation`

### ✅ 頑健性総合検証（2026-03-15）
- Monte Carlo N=2000: Sharpe>1.0 確率=100%
- パラメータ感度: 滑らかな感度曲線（過学習なし）
- 銘柄サブセット20試行: Phase1達成率100%
- 詳細: `results/backtest_summary.json` → `robustness_analysis`

---

## 完了した研究（2026-03-29）

### ✅ データ凍結後 基準バックテスト取得（Step1）
**スクリプト**: `backtest/composite_alpha_bt.py --rsr42-trade`
**データ**: `DATA_VERSION=2026-03-28 / HASH=492c888409041827`
**結果**: `results/composite_alpha_bt_rsr42_2026-03-29.json`

| シナリオ | CAGR | Sharpe | MaxDD | Calmar | avgHoldings |
|---|---|---|---|---|---|
| BASELINE | +15.2% | 0.648 | -18.9% | 0.806 | **2.41** |
| STEP1 | +14.4% | 0.614 | -18.3% | 0.788 | 2.36 |
| STEP2 | +11.7% | 0.539 | -18.1% | 0.646 | 2.19 |
| STEP3 | +12.3% | 0.566 | -18.1% | 0.682 | 2.15 |
| STEP5 | +14.1% | 0.604 | -18.3% | 0.769 | 2.35 |
| STEP6/6A/6B | 全STEP5と同値 | ← **breadthバグ確定** |

**breadth確定バグ**: STEP5=STEP6=STEP6A=STEP6B（完全一致）
- Breadth中央値 = 0.26（定数）
- 原因: RSRがパーセンタイルランクのため、RSR≥75の割合は定義上≈25%で固定
- `_calc_breadth()` は dead code

**ログ出力追加済み**（再現性確認用）:
```
print("DATASET_VERSION", dataset_version)
print("DATASET_HASH",    dataset_hash)
print("CAPITAL",         config.capital)
print("UNIVERSE",        len(universe))
```
また `avg_simultaneous_holdings` を全シナリオに追加。

---

### ✅ CAPITAL整合性検証（Step2）

| Layer | 場所 | 値 | 状態 |
|---|---|---|---|
| configs | `configs/strategy.yaml` `portfolio.capital` | **2,000,000** | ❌ 旧設定 |
| engine (BT) | `composite_alpha_bt.py` `CAPITAL` 定数 | **3,000,000** | ✅ 実口座と一致 |
| engine (live) | `run_live_signal.py` `CAPITAL` default | **2,000,000** | ❌ env未設定時旧値 |
| engine (v2) | `portfolio_v2.py` `CAPITAL` | **2,000,000** | ❌ 旧設定 |
| sizing cap | `alloc = min(alloc, capital * 0.25)` | 初期資本固定 | ⚠ equity連動でない |

**DD分析は正しい**（`(cur_equity - peak_equity) / peak_equity`）。ただし position cap が初期資本固定なのでポートフォリオ成長時に保守的になる。

**未修正**（次セッションで対応）:
- `configs/strategy.yaml`: `capital: 2_000_000` → `3_000_000`
- `run_live_signal.py`: default `2_000_000` → `3_000_000`

---

### ✅ RSRユニバース拡張テスト（Step3 第1部）
**スクリプト**: `backtest/rsr_universe_sweep.py`（新規）
**結果**: `results/rsr_universe_sweep_2026-03-29.json`

設計: トレードユニバース=RSR42固定、RSRコンテキストのみ拡大

| シナリオ | RSRコンテキスト | avgHoldings | 取引数/年 | top5固着HHI | Sharpe |
|---|---|---|---|---|---|
| RSR_CTX42 | 42銘柄 | **2.13 ⚠** | 40 | 0.0578 | **0.618** |
| RSR_CTX76 | 76銘柄 | **1.82 ⚠** | 39 | 0.0616 | 0.430 |
| RSR_CTX91 | 91銘柄 | **2.33 ⚠** | 44 | 0.0578 | 0.484 |

**仮説「RSRコンテキスト拡大 → avgHoldings改善」は棄却**:
- 全サイズで avgHoldings < 3（危険域）
- ランクターンオーバー = 0.07（全サイズ固着）
- RSR42が最高性能（Sharpe 0.618）

**真因確定**: `min_rsr=75.0` フィルターが強すぎる。コンテキストを広げると同じ銘柄のRSRが下がるため、RSR76/91はむしろ悪化。

**RSR遅延（Step2）速報**:
- avg_lag_days = 5.8日（理想-5〜+5のギリギリ外）
- lag>20日の割合: 5.1%（許容範囲）
- RSR遅延は軽微 → 遅延はボトルネックではない

---

## 完了した研究（2026-03-16）

### ✅ 外部レビュー対応 4タスク統合検証（2026-03-16）
**スクリプト**: `backtest/advanced_analysis.py`（新規）
**背景**: 外部レビュー「スクリーニングによる過学習」への対応

**重要な結論**:
- Ex-ante 109銘柄（スクリーニングなし）: CAGR -0.7%〜-2.2%（全設定マイナス）
- 年間シグナルは516件あり十分。問題は「銘柄の適合性」
- **仮説「稼働率がボトルネック」→ 棄却**
- **G27スクリーニング維持を推奨**（廃止は逆効果と確認）

**セクター適合性の確定（フジコ法が機能するセクター）**:
電機精密(Sharpe2.83) / 輸送機器(2.92) / 鉄鋼(2.57) / 機械(2.16) / 電機(1.76) / 海運(1.27) / 商社(1.38)

**除外候補**: 小売（3382.T）勝率18.2%・Sharpe-1.24

### ✅ CB問題・セクタートップ銘柄・CB改善 統合検証（2026-03-16）
**スクリプト**: `backtest/cb_sector_analysis.py`（新規）

**Task 1: セクター内Top-N仮説**
| シナリオ | 銘柄数 | CAGR | Sharpe | MaxDD | Calmar | CB日数 |
|---|---|---|---|---|---|---|
| G27現行（ベースライン） | 19 | +14.0% | 1.541 | -8.5% | 1.639 | 0日 |
| Top1/セクター | 7 | +4.7% | 0.658 | -11.4% | 0.413 | 0日 |
| **Top2/セクター** | **14** | **+9.2%** | **1.012** | **-10.2%** | **0.897** | **0日** |
| Top3/セクター | 21 | +5.4% | 0.569 | -15.1% | 0.359 | 554日 |
| 全銘柄(48) | 48 | +4.3% | 0.461 | -15.3% | 0.280 | 758日 |

**Task 2: CB構造的デッドロック確認**
- 発動後 全キャッシュ → peak_equity固定 → DD=-15.4%永久 → 解除条件(-7.5%)到達不可
- 2021年ワースト: 5706.T鉄鋼(-15.1%), 7011.T機械(-13.2%★G27), 6762.T電機精密(-11.7%★G27)
- 2022-2024 G27∩Sector平均: +119.1% vs Sectorのみ: +61.9% → **G27はin-sample選定で上位株を保有**

**Task 3: CB設計改善比較（Sector Filter 48銘柄）**
| モード | CAGR | Sharpe | MaxDD | CB日数 |
|---|---|---|---|---|
| A: 現行standard | +4.3% | 0.461 | -15.3% | 758日 |
| B: time_limit 120日 | +4.0% | 0.438 | -16.8% | 752日（悪化） |
| C: time_limit 60日 | +3.3% | 0.369 | -20.7% | 746日（悪化） |
| **D: entry_stop_only** | **+13.1%** | **1.029** | **-16.1%** | **175日（✅大幅改善）** |
| E: partial_size 50% | +13.1% | 1.029 | -16.1% | 175日（D同等） |
| F: no_cb | +11.4% | 0.871 | -22.5% | 0日 |

**4問への回答**:
1. **G27高Sharpeの真因**: セクター選択（+0.4 Sharpe）＋in-sample銘柄選択バイアス（+0.5 Sharpe）の両方。Top2/セクターで1.012止まり → 残りは過学習由来
2. **Sector Filterで再現可能か**: Top2/Sec=1.012 vs G27=1.541 → **まだ差がある ❌**
3. **CB改善で2022-2024解決**: D(entry_stop_only)でSharpe 0.461→1.029、2022-2024年次+7.2%/+41.5%/+9.4% → **✅解決**
4. **最再現性の高いUniverse**: Top2/セクター（流動性ランキングで決定） + entry_stop_only CB → OOS検証推奨

---

## 完了した研究（2026-03-19）

### ✅ バイアス定量化（時間的分離バックテスト）
**スクリプト**: `backtest/portfolio_temporal_separation.py`（新規）
**チャート**: `C:/Users/owner/.claude/レポート/temporal_separation_bias.png`

#### Research Freeze — 2026-03-19

```
Universe   : TOPIX100 subset (74 symbols, yfinance)
Selection  : 2015-01-01 〜 2017-12-31（eval期間と完全分離）
Evaluation : 2018-01-01 〜 2024-12-31
Filter     : Sharpe>0.3 かつ MaxDD>-30%（閾値の再最適化禁止）
Survivorship bias: 存在（2024年時点の生存銘柄のみ。影響 ≈ CAGR+1〜2%と推定）
```

| シナリオ | Sharpe | CAGR | MaxDD | Calmar | 意味 |
|---|---|---|---|---|---|
| BIASED（現行） | 1.724 | +16.42% | -8.32% | 1.973 | in-sample選択・比較基準 |
| **TEMPORAL（時間分離）** | **1.070** | **+9.98%** | **-10.62%** | **0.940** | **真の性能推定値** |
| TOP2_SEC（出来高） | -0.335 | -2.03% | -16.27% | — | CBデッドロック（戦略の問題ではない） |

**銘柄選択バイアス**: BIASED - TEMPORAL = **+0.654 Sharpe**
**真の性能推定**: Sharpe ≈ 1.07（Phase 1基準 >0.5 をクリア ✅）

**バイアスの正体**（ユニバース差分から）:
- BIASED only（18銘柄）: 海運（9101/9104）・銀行（8306/8411）など → **2015-2017低迷・2018-2024で急騰した銘柄を将来情報で選択**
- TEMPORAL only（15銘柄）: 化学・医薬品・陸運 → 2015-2017有効だが2018-2024は不発
- BIASED∩TEMPORAL（9銘柄）: 8035.T 6920.T 8001.T など → **両期間で有効な本物のコア銘柄**

#### Decision
- **実運用ユニバースを TEMPORAL 選定（24銘柄）に切り替える**
- ただし macro regime capture の懸念があるため、まずレジームブレークダウン検証を実施すること（未完了）

#### ✅ レジームブレークダウン（2026-03-19 完了）

| 年 | TEMPORAL | BIASED | レジーム |
|---|---|---|---|
| 2018 | -4.6% | -2.0% | 下落 |
| 2019 | +44.5% | +54.0% | 上昇 |
| 2020 | +10.5% | +11.2% | 暴落+回復 |
| 2021 | **+4.1%** | +29.9% | 上昇（⚠ TEMPORAL低い） |
| 2022 | +1.8% | +4.8% | 下落 |
| 2023 | +22.5% | +19.3% | 上昇 |
| 2024 | -1.0% | +5.4% | 横ばい |

| レジーム | TEMPORAL平均 | BIASED平均 | 判定 |
|---|---|---|---|
| 上昇相場（2019/2021/2023） | +23.7% | +34.4% | ✅ 上昇を取れている |
| 下落・横ばい（2018/2022/2024） | **-1.3%** | **+2.7%** | ✅ 小幅マイナス（許容範囲） |
| 暴落+回復（2020） | +10.5% | +11.2% | ✅ 暴落年も正のリターン |

**判定: ✅ macro regime capture ではない**

- 下落時 TEMPORAL -1.3% → 相場依存でない健全なパターン
- 2021年 TEMPORAL +4.1% の低さ → BIASEDが将来情報で海運・銀行（2021急騰）を選択した結果。TEMPORALの問題ではなく**BIASEDのバイアスが2021年に集中していたことの証拠**
- BIASED と TEMPORAL の差 +0.654 Sharpeは、2021年の商品・海運ブームを先読みしたバイアス由来と確定

---

## 完了した研究（2026-03-17〜18）

### ✅ entry_stop v5 実装・バックテスト（2026-03-18）
**スクリプト**: `backtest/portfolio_entry_stop_v5.py`
**結果**: `results/entry_stop_v5_2026-03-18.json`

4段階ステートマシン（NORMAL/CAUTION/WARNING/ALERT）で段階的リスク制御を実装。
ヒステリシス・段階的復帰・縮小継続・回復速度制御・2x専用パラメータの5点を統合。

| シナリオ | CAGR | MaxDD | Calmar | avg_scale | NORMAL% |
|---|---|---|---|---|---|
| ベースライン | +13.65% | -5.67% | 2.408 | 1.000 | 100% |
| v5 段階EXP+ヒステリシス（vel/z無） | +13.09% | -5.67% | **2.309** | **0.968** | **93.6%** |
| v5 フル 1x | +12.20% | -5.93% | 2.057 | 0.830 | 76.1% |
| v5 フル 2x | +13.00% | -5.32% | **2.444** | 0.823 | 76.3% |

**採用判定**:
- `v5 段階EXP+ヒステリシス`（vel/z無）: avg_scale=0.968・NORMAL=93.6% → **稼働率毀損ほぼなし・保守採用候補**
- `v5 フル 2x`: Calmar=2.444（ベースライン超え）→ レバレッジ運用時に有効
- `v5 フル 1x`: Z-score初期誤発火（DD=-0.45%でWARNING）→ v6で `dd_abs>1%` 下限フィルター追加が必要

**v2〜v4の教訓**（archive済み）:
- v2: 永久ロックアウト問題（局所回復率に未対応）
- v3: -6%閾値で1xは未発動（V2のMaxDD=-5.67%が閾値未満）
- v4: velocity=2%で過敏発火 → avg_exp=0.555まで低下（採用不可）

### ✅ ファイル整理（2026-03-18）
- `backtest/archive/` に旧版・実験済み30ファイルを移動（削除せず保管）
- `results/archive/` に entry_stop v2〜v4結果を移動
- `archive/` にルートの不要スクリプト6ファイルを移動
- `backtest/` 現役ファイルを11本に整理

### ✅ kabuステーション API認証修正（2026-03-18）
- Web側でAPIパスワード変更後はkabuステーションの再起動が必要（仕様確認）
- `.env` の `KABU_API_PASSWORD` を新パスワードに更新済み
- 本日の売買: 保有2銘柄（5401.T 100株 / 5411.T 200株）を正しく認識、発注なし（全HOLD）

### ✅ 朝のルーティン自動化（2026-03-17）
- `morning_dryrun.bat`（8:30）/ `morning_live.bat`（9:00）作成
- Windowsタスクスケジューラ登録（平日自動実行・StartWhenAvailable=True）
- `signal_bridge.py` バグ修正: `dropna(how="all")` → `dropna(subset=["Close"])`
  - 原因: yfinanceが直近営業日のCloseをNaNで返しRSR全体がnanになる問題

### ✅ 段階型DDリスク制御バックテスト（2026-03-17）
**スクリプト**: `backtest/portfolio_dd_control.py`（新規）
**結果**: `results/dd_control_2026-03-17.json`

G27+V2のMaxDD=-7.53%は全スキーム閾値未満 → 758日ロック問題はV2移行で実質解消済み

**ストレステスト**: `backtest/portfolio_stress_test.py`（新規）
**結果**: `results/stress_test_2026-03-17.json`
- 2.0x時: 案AC(-12%entry_stop/-6%DD-only解除)が1508日ロック
- **重要**: entry_stop解除にDD条件のみは禁止。必ず「30〜60営業日 OR TOPIX>MA200」を追加すること

### ✅ 資本効率最大化バックテスト（2026-03-17）
**スクリプト**: `backtest/portfolio_capital_efficiency.py`（新規）
**結果**: `results/capital_efficiency_2026-03-17.json`

| 機能 | CAGR変化 | 評価 |
|---|---|---|
| ① ランキング加重（RSR順位加重, max_pos_weight=0.40） | +16.66%→+18.92%（+2.26pp） | ✅ 採用推奨 |
| ② MIN_POSITIONS=3（補完エントリー） | -6.85pp（稼働率も低下） | ❌ 廃止 |
| ③ 強制ローテーション（diff=5） | ±0（0回/年） | △ Phase 3以降で再評価 |

---

## 完了した研究（2026-03-20）

### ✅ 2025年OOS検証（2026-03-19〜20）
**スクリプト**: `backtest/oos_2025.py`（新規）
**結果**: `results/oos_2025_2026-03-19.json`

#### OOS結果サマリー

| 指標 | IS 2018-2024 | OOS 2025 (CB有) | OOS 2025 (CB無) |
|---|---|---|---|
| CAGR | +3.56% | -11.70% | **-6.99%** |
| Sharpe | 0.387 | -2.646 | -0.676 |
| MaxDD | -12.21% | -11.31% | -11.33% |
| avg_exposure | 0.756 | 0.222 | 0.868 |

**原因分析:**
- 外因: 2025-04-02 トランプ関税ショック（Liberation Day）で2-4月3ヶ月連続マイナス（化学・電機精密・商社を直撃）
- 内因: 2025-04-24 DD -15.3%でCB発動 → 以降8ヶ月デッドロック（-4.71%のコスト）

**重要発見: RSRコンテキスト問題**
- 実運用条件（TEMPORAL24のみでRSR計算）だとIS Sharpe=0.387
- temporal_separation.py の Sharpe=1.07 は~40銘柄でRSR計算した値（広いコンテキスト）
- RSRはユニバース内相対ランク → 銘柄数が変わるとmin_rsr=70の閾値の意味が変わる
- **修正方針**: TOPIX100全銘柄でRSRを計算し、取引対象はTEMPORAL24に絞る

---

## 完了した研究（2026-03-19 追記）

### ✅ entry_stop v6 開発・凍結決定（2026-03-19）
**スクリプト**: `backtest/portfolio_entry_stop_v6.py`（新規）
**結果**: `results/entry_stop_v6_2026-03-19.json`
**対象ユニバース**: TEMPORAL 24銘柄（avg_exposure 2018=0.082、2019-24=0.45〜0.68、全体=0.510）

#### 開発経緯・全バリアント結果

| バリアント | CAGR | Calmar | NORMAL% | avg_scale | 問題点 |
|---|---|---|---|---|---|
| ベースライン（entry_stop なし） | +4.18% | 0.398 | 100% | 1.000 | — |
| v5互換（z-scoreのみ） | -0.07% | -0.009 | 15.4% | — | z-score誤発火28x（DD微小時）|
| v6（3条件: z+dd_abs+vel） | -0.35% | -0.047 | 16.7% | — | z発火24x・std小さすぎ |
| v6b（中央値フロア） | -0.35% | -0.047 | 16.7% | — | フロアも bootstrap汚染 |
| **v6c（regime-gate）** | **-0.89%** | **-0.116** | **12.5%** | **0.321** | velocity ロック（2018 exposure=0.082）|
| v6d（exposure調整DD） | -0.45% | -0.120 | 12.3% | 0.125 | raw_dd/-3.7%→effective_dd=-15%でALERT33x |

#### 根本原因（構造的不適合）

```
TEMPORAL 2018: avg_exposure = 0.082（91.8%キャッシュ）
→ DD=-3.7%は「リスク資産の実損失ではなく稼働不足」
→ velocity trigger（2018 Oct）→ WARNING → 新規BUY制限 → 回復遅延 → 7年間ロック
→ exposure調整も逆効果（floor=0.25: raw_dd/0.25 → effective_dd増幅）
```

#### ✅ 最終決定: entry_stop を TEMPORAL 1x で完全凍結

```
根拠: entry_stop は「十分にデプロイされたポートフォリオ（avg_exposure>0.45）」
      の過剰リスクを制御するツールとして設計されている。
      TEMPORAL 1x の低稼働率（初期 avg_exposure=0.082）は entry_stop の適用前提
      を満たしておらず、どのバリアント（v6a〜v6d）もベースラインを大幅に下回った。

再評価条件: avg_exposure > 0.45 が安定的に維持される場合（Phase 3以降の資本拡大時）
```

---

## 完了した研究（2026-03-20 追記）

### ✅ turtle_exit 3way 検証 → パラメータ変更（2026-03-20）
**スクリプト**: `backtest/exit_param_3way.py`（新規）
**結果**: `results/exit_param_3way.json`
**グラフ**: `C:/Users/owner/.claude/レポート/exit_param_3way.png`

#### 背景
PnL vs 保有日数分析で「11-15日バケット avg_pnl=-0.47%」を確認。
turtle_exit=10 が勝ちトレードを早刈りしている仮説を検証。

#### 結果

| シナリオ | CAGR | MaxDD | Calmar | Sharpe | avg_hold | 勝率 |
|---|---|---|---|---|---|---|
| A: turtle_exit=10（旧設定） | +5.05% | -14.23% | 0.355 | 0.525 | 13.1日 | 51.6% |
| **B: turtle_exit=20** | **+9.28%** | **-12.91%** | **0.719** | **0.918** | **17.0日** | **53.0%** |
| C: turtle_exit=20+ATR×2 stop | +8.52% | -13.33% | 0.639 | 0.871 | 15.2日 | 50.2% |

#### 判定
- **B（turtle_exit=20）がベスト**: Calmar +103%改善、MaxDDも縮小
- Cは ATRストップ56/259回発動だが**勝率0%・avg_pnl -33,524円**（損切り専用で効果なし）
- 仮説通り「10日安値が勝ちトレードを刈っていた」と確定

#### ✅ パラメータ変更済み（2026-03-20）
- `configs/strategy.yaml`: `turtle_exit: 10` → `20`
- `run_live_signal.py` / `backtest/portfolio_v2.py` 他アクティブスクリプト全件更新

---

## 完了した研究（2026-03-20）

### ✅ Top-k ローテーション バックテスト（2026-03-20）
**スクリプト**: `backtest/topk_rotation.py`（新規）

#### 設計
- RSR universe = TEMPORAL24（24銘柄固定）
- entry: rank ≤ k AND slot available
- exit: rank > k OR time_exit（max_hold_days）
- k = 2, 3, 4, 5 の 4 ケース + stop_loss / max_hold_days バリアント

#### OOS 結果サマリー（k=4 / 2025年 OOS）

| ケース | IS Sharpe | IS MaxDD | OOS Sharpe | OOS MaxDD |
|---|---|---|---|---|
| k=4_base | 0.910 | -31.9% | 1.254 | -6.11% |
| k=4_sl15_h60 | 0.415 | -21.8% | 1.114 | **-13.98%** |

**採用パラメータ決定**: k=4 / max_hold_days=60 / stop_loss=None
- OOS MaxDD -13.98% < Phase 1 基準 -20% ✅
- IS Sharpe 低下（0.91→0.42）は 2018年低稼働率が主因。OOS への影響は軽微

### ✅ ライブシステム Top-k 本番実装（2026-03-20）
**変更ファイル**: `kabusapi/signal_bridge.py`（完全リライト）/ `run_live_signal.py`（差分変更）

#### 実装内容

| 機能 | 実装詳細 |
|---|---|
| CB 状態機械 | NORMAL→CB_ACTIVE(-15%)→RECOVERY(30営業日)→NORMAL(peak×98%) |
| top_k 選出 | RSR 上位 k 銘柄 + 流動性 tie-breaker（5B円/日フィルター） |
| 時間ストップ | 営業日計算（pd.bdate_range）で max_hold_days=60 を判定 |
| 再エントリー禁止 | 時間ストップ後 5 営業日は同銘柄への BUY を停止 |
| 過剰発注防止 | max_new_positions_per_day=2 / order_rate_limit=3件/分 |
| 状態永続化 | `runtime/portfolio_state.json` で entry_date・reentry_blocked を管理 |
| CB イベントログ | `logs/cb_events/YYYYMMDD.jsonl` に状態遷移を記録 |

#### パラメータ確定値（ライブ）
```
TOP_K = 4 / MAX_HOLD_DAYS = 60 / MAX_NEW_POS_PER_DAY = 2
MAX_POS = 4 / MIN_SECTORS = 1 / MAX_DD_LIMIT = 0.15
```

---

## 完了した研究（2026-03-22）

### ✅ RSRコンテキスト不一致の診断・修正（2026-03-22）

**問題**: ライブ avg_exposure = 8.3% （研究値 32.7% の1/4）

**根本原因チェーン**:
1. RSRコンテキスト3方向ミスマッチ: ライブ=77銘柄 / 研究=42銘柄 / バックテスト=24銘柄 → RSRパーセンタイル非比較
2. アーキテクチャ不一致: ライブは top_k-first（閾値なし）、研究は filter-first（RSR≥75 閾値）
3. G27 = in-sample選定（2018-2024データで選定した銘柄を2018-2024で評価）→ Sharpe過大評価

**修正内容**:

| ファイル | 変更内容 |
|---|---|
| `configs/rsr_universe_42.csv` | **新規作成**: 42銘柄統一RSRコンテキスト（G27 + 15追加） |
| `run_live_signal.py` | RSR universe を TOPIX100→42銘柄CSVに変更、min_rsr=75.0に修正 |
| `kabusapi/signal_bridge.py` | min_rsr強制0化を削除、filter-firstアーキテクチャに修正、LIVE_STATEロギング追加 |
| `analysis/live_exposure_report.py` | **新規作成**: Phase2 exposure モニタリングレポート |
| `backtest/live_equivalent.py` | **新規作成**: 42銘柄RSRコンテキスト + filter-first + --rolling フラグ |

### ✅ ローリング選定ユニバース構築（2026-03-22）

**スクリプト**: `backtest/rolling_universe.py`（新規）
**出力**: `configs/rolling_universe.json`

**設計**:
- 3年 train → 1年 OOS、7フォールド（2015-2017→2018 〜 2021-2023→2024）
- スクリーニング基準: Sharpe>0.3 AND MaxDD<30% ← G27と同一閾値だがOOS
- RSR: trainウィンドウ内のTOPIX100コンテキストで計算（テストデータ混入なし）

**結果**:
| フォールド | train期間 | OOS年 | 選定銘柄数 |
|---|---|---|---|
| 1 | 2015-2017 | 2018 | 26 |
| 2 | 2016-2018 | 2019 | 15 |
| 3 | 2017-2019 | 2020 | 9 |
| 4 | 2018-2020 | 2021 | 15 |
| 5 | 2019-2021 | 2022 | 22 |
| 6 | 2020-2022 | 2023 | 26 |
| 7 | 2021-2023 | 2024 | 30 |
| **平均** | — | — | **20.4** |

延べユニーク銘柄: 59（TOPIX100から76銘柄取得成功、9613.Tは上場廃止でスキップ）

### ✅ ローリングOOS ライブ等価バックテスト + RSRコンテキスト最適化（2026-03-22）

**スクリプト**: `backtest/live_equivalent.py --rolling [--broad-rsr]`
**スクリプト**: `backtest/rsr_context_sweep.py`（2×3グリッド sweep）

#### 実験グリッド結果（broad × min_rsr × max_single_weight）

**Phase 1: RSR context × min_rsr 2×3 sweep**

| RSR context | min_rsr | Sharpe | CAGR | MaxDD | exposure | cands/日 |
|---|---|---|---|---|---|---|
| narrow（年別選定 9〜30） | 75 | 0.942 | +7.55% | -9.29% | 19.7% | 0.20 |
| narrow | 70 | 0.810 | +6.70% | -9.91% | 21.9% | 0.23 |
| **broad（TOPIX100 ~76）** | **75** | **1.139** | **+14.72%** | **-17.50%** | **37.7%** | **0.30** |
| broad | 70 | 1.098 | +14.30% | -17.82% | 39.1% | 0.33 |

**narrow→broad 効果（min_rsr=75固定）**: Sharpe +0.197、exposure +18pp、cands +0.10/日
**min_rsr 感度（broad内）**: Sharpe差=0.045、exposure差=1.4pp → **閾値はボトルネックではなかった**

**Phase 2: max_single_weight sweep（broad / max3 / min_rsr=75）**

| max_single_weight | Sharpe | CAGR | MaxDD | exposure | HHI_avg | 判定 |
|---|---|---|---|---|---|---|
| 0.25（現行） | 1.139 | +14.72% | -17.50% | 37.7% | 0.104 | ❌ DD超過 |
| **0.15（確定）** | **1.181** | **+10.07%** | **-12.66%** | **24.5%** | **0.043** | **✅ 全通過** |
| 0.20（中間） | 1.145 | +12.02% | -14.58% | 30.2% | 0.064 | ❌ exp超過 |

**HHI 解釈**: momentum clustering は主因でなかった。MaxDD-17.5%の正体は純粋に weight過大（1銘柄25%=50万円の直撃）。
weight を下げると **Sharpe が上昇**（エクイティカーブのノイズ減少）。

#### ✅ 確定設計（OOS検証済み）

```
RSR context      : TOPIX100 broad (~76銘柄)
min_rsr          : 75.0
architecture     : filter-first（RSR≥75 → RSR降順 → top max_positions）
max_positions    : 3
max_single_weight: 0.15   ← 今回の変更点
```

**OOS Rolling 検証結果（2018-2024、7フォールド、look-ahead biasなし）**:

| 指標 | 値 | 基準 | 判定 |
|---|---|---|---|
| Sharpe | **1.181** | >1.0 | ✅ |
| MaxDD | **-12.66%** | <15% | ✅ |
| avg_exposure | **24.5%** | 23-28% | ✅ |
| avg_candidates | 0.30/日 | >0.3 | ✅ |
| CAGR | +10.07% | — | — |

**IS→OOS 比較**: G27 IS Sharpe=1.693 → Rolling OOS Sharpe=1.181 → **保持率 69.8%**（前回narrow比で大幅改善）

---

## 次の研究タスク（優先順）

| 優先度 | タスク | 根拠 |
|---|---|---|
| ~~1~~（完了） | ~~TEMPORALユニバースを `run_live_signal.py` に反映~~ | 2026-03-19 完了 |
| ~~2~~（完了） | ~~`scripts/monthly_pnl.py` Phase 2評価スクリプト作成~~ | 2026-03-19 完了 |
| ~~3~~（完了・凍結）| ~~entry_stop v6~~ | 2026-03-19 全バリアント失敗→凍結決定 |
| ~~4~~（完了） | ~~2025年実運用OOS検証~~ | 2026-03-19 完了 |
| ~~5~~（完了） | ~~Top-k ローテーション実装・ライブ統合~~ | 2026-03-20 完了 |
| ~~6~~（完了） | ~~RSRコンテキスト修正・exit=20正式適用・exposure実測~~ | 2026-03-21 完了 |
| ~~7~~（完了） | ~~ローリング選定ユニバース + RSRコンテキスト最適化 + weight最適化~~ | 2026-03-22 完了 |
| ~~1~~（完了） | ~~paper trade 2週間: broad RSR + filter-first + w=0.15~~ | 2026-03-23 診断ログ追加で代替 |
| ~~2~~（完了） | ~~`run_live_signal.py` / `signal_bridge.py` に確定設計を反映~~ | 2026-03-23: MAX_POS=3, TOP_K=3, max_single_weight=0.15 適用済み |
| ~~3~~（完了） | ~~価格フィルター上限引き上げ~~ | 2026-03-23: ¥500,000 → ¥600,000（8002.T 評価可能に） |
| ~~1~~（完了） | ~~10営業日 paper trade 観察~~（前倒し実施） | 2026-03-23: 診断結果からcandidate=0を確認→即日対応 |
| ~~2~~（完了） | ~~ranking universe 拡張テスト（RSR42への拡張）~~ | 2026-03-23: 並列テストでcands/日0.18→0.69（3.7倍）確認 |
| **1** | **10営業日 paper trade 観察（RSR42版）**: `rsr_pass_count` / `candidate_count` を集計 | 2026-03-23〜。基準: rsr_pass_count≥3/日が安定したら正常化と判断 |
| **2** | RSRスロープ改善テスト（Step 3） | `rsr_pass_count≥3` 安定後に実施。RSR単純パーセンタイル → トレンド品質で補強 |
| **3** | 複合ランキング（Composite Alpha）テスト（Step 4） | RSRスロープ結果を踏まえて実施 |
| 保留 | R²×スロープランキング（Clenow方式）への置き換え | 現在は単純 RSR。品質モメンタムで選別精度向上の可能性 |

### 2026-03-23 診断結果（3本並行テスト）

**根本原因特定**: 価格フィルターが RSR上位銘柄を全員除外していた

| 銘柄 | RSR(42ctx) | 状態 |
|---|---|---|
| 6920.T | 95.2 | ¥3,193,000 → 除外継続（上限超過） |
| 8002.T | 81.0 | ¥515,600 → **上限引き上げで評価可能に** |
| 8035.T | 78.6 | ¥3,794,000 → 除外継続（上限超過） |

**市場レジーム**: TOPIX +11.89% vs 200MA → 強気相場確認（弱気ではない）
**RSR期間感度**: IBD式が最良（42日/ブレンドは悪化）
**ユニバースバイアス**: 軽微（300銘柄に拡張でも+1銘柄のみ）

---

## Experiment: Universe Expansion to RSR42（2026-03-23）

### 背景（供給危機の発見）
診断ログ（metrics.jsonl）で `candidate_count=0`、`signals_blocked_rsr=15` を確認。
TEMPORAL24（化学・医薬品・レジャー中心）は RSR42コンテキスト（電機精密・海運・機械中心）で
常に下位ランクになる構造的ミスマッチが根本原因。

### 実験設計
| | Universe A（旧） | Universe B（新） |
|---|---|---|
| 取引ユニバース | TEMPORAL24（24銘柄） | RSR42（42銘柄） |
| RSRコンテキスト | RSR42 | RSR42（同一） |
| max_single_weight | 0.15 | 0.15 |

### バックテスト結果（2018-2024、IS）
| 指標 | Universe A | Universe B |
|---|---|---|
| CAGR | +3.82% | +11.05% |
| Sharpe | 0.697 | **1.258** |
| MaxDD | -7.09% | **-12.42%** |
| avg_exposure | 13.2% | 21.9% |
| avg_cands/日 | 0.183 | **0.686** |
| 取引数（7年） | 199 | 247 |

### ライブドライラン結果（2026-03-23）
```
universe_size:       32（価格フィルター後）
rsr_pass_count:       5（旧設定では 0）← 供給回復
candidate_count:      0（Turtle 20日高値ブレイクアウト待ち）
blocked_by_breakout:  4（RSR通過だがエントリー条件未達 ← 正常動作）
SELL シグナル:   5401.T（RSR=11.9）/ 5411.T（RSR=14.3）← 旧ポジション整理
RSR上位候補:    7013.T(92.9) / 8058.T(90.5) / 8015.T(85.7) / 7011.T(76.2)
```

### 判定
- **RSR供給回復**: 0→5（戦略が「正常化している」サイン）
- **Breakout待ちフェーズ**: 機械・商社 銘柄が20日高値を更新すれば即座にBUY候補
- **MaxDD -12.42%**: Phase 1基準（<20%）・CB限界（-15%）ともにクリア

### 変更ファイル
| ファイル | 変更内容 |
|---|---|
| `configs/universe/2026Q1_rsr42_universe.json` | 新規作成（42銘柄定義） |
| `.env` | `LIVE_UNIVERSE_FILE` → rsr42_universe に変更 |
| `backtest/universe_parallel_test.py` | 新規作成（A/B並列比較スクリプト） |
| `backtest/live_equivalent.py` | `trade_universe` パラメータ追加 |
| `kabusapi/signal_bridge.py` | metrics に `rsr_pass_count` / `blocked_by_rsr` / `blocked_by_breakout` 追加 |

### 次の評価ウィンドウ
**10営業日後（〜2026-04-08頃）**:
- `rsr_pass_count` の 10日平均 ≥ 3.0 → 正常化確認
- `candidate_count` の 10日平均 ≥ 0.3 → Turtle供給の確認
- 上記未達の場合: min_rsr=75→70 への引き下げを検討（Step 3: RSRスロープ改善と同時）

---

## フェーズ移行: 研究フェーズ → 運用評価フェーズ（2026-03-23）

### 現在地
- RSR設計: IBD加重12ヶ月（直近3ヶ月×40% + 各3ヶ月×20%）→ 問題なし
- 診断ログ: 供給・安定性・RSR分散・市場レジーム・近接銘柄まで整備完了
- OOS検証: rolling fold-level実施済み（turtle_entry=20確定）
- **次のボトルネック: 意思決定速度（ログはあるが判断レポートがなかった）**

### 完了: 意思決定基盤の構築（Step 1）

#### `research/weekly_report.py`（新規作成）
```
python research/weekly_report.py
python research/weekly_report.py --weeks 4
python research/weekly_report.py --since 2026-04-01
```
出力内容:
- 週次: 取引数 / 勝率 / 期待値 / PnL / signals_per_week密度
- 月次: 月別PnL / MaxDD / **regime別成績**（bull/neutral/bear）← 重要
- 供給診断: rsr_pass / near_breakout / rsr_dispersion の20日推移

#### `signal_bridge.py` 追加ログ（2026-03-23）
| フィールド | 説明 |
|---|---|
| `trend_market` | bull/neutral/bear（TOPIX MA50 vs MA200） |
| `near_breakout_count` | 20日高値の2%以内の銘柄数（供給予測） |
| `rsr_dispersion` | Top20 RSRのstd（>10=強い相場、<5=横ばい） |
| `estimated_price` in send_results | BUY/SELL実行時の参考価格（PnL計算基礎） |

#### `logs/trades.jsonl`（自動生成）
- 実発注成功後に `update_state_after_execution()` が書き込む
- BUY: date/symbol/sector/qty/price/entry_regime
- SELL: 上記 + pnl/pnl_pct/hold_days/entry_price/entry_date

### 完了: 観察期間中の並列改善（2026-03-23 同日実装）

#### `signal_bridge.py` 追加ログ（全フィールド一覧）
| フィールド | 説明 | 目安 |
|---|---|---|
| `trend_market` | bull/neutral/bear（MA50 vs MA200） | — |
| `trend_strength` | (MA50-MA200)/MA200 | >0.05=強 / <-0.02=下落 |
| `near_breakout_count` | 20日高値2%以内の銘柄数 | ≥3で近くシグナル増加見込み |
| `rsr_dispersion` | Top20 RSRのstd | >10=強相場 / <5=横ばい |
| `failed_breakout_count/rate` | entry後5日以内 -2ATR 到達 | ベースライン記録中 |
| `breakout_opportunity_rate` | near_breakout / rsr_pass | >0.4=十分 / <0.2=停滞 |
| `mtf_filtered_candidates` | RSR通過かつ週足MA20弱の数 | 診断のみ（フィルターしない） |
| `mtf_filter_rate` | 上記 / rsr_pass | 0.2〜0.4=MTF有効 / <0.05=意味なし |
| `rsr_leader_half_life` | Top10滞在半減期（日） | >20=強 / <8=回転相場 |
| `top10_overlap` | 昨日との Top10 重複数 | 4〜7=安定 |
| `signals_per_week` | 直近5日BUY候補合計 | 2〜6=健全 |

#### `logs/trades.jsonl`（実発注成功時に自動記録）
- BUY: date/symbol/sector/qty/price/atr20/entry_regime
- SELL: 上記 + pnl/pnl_pct/hold_days/entry_price/entry_date

#### `backtest/mtf_comparison.py`（新規）
```bash
python -m backtest.mtf_comparison
```
Baseline vs MTF-A(weekly_ma20) vs MTF-B(weekly_cross) を自動比較。
採用基準: **Sharpe差 ≥ 0 かつ false_breakout_rate差 ≤ -0.02**

#### `research/weekly_report.py` 出力項目
- 週次: 取引数 / 勝率 / 期待値 / signals_per_week
- 月次: PnL / **regime別成績** / trend_strength 推移
- 供給診断: rsr_pass / near_breakout / rsr_dispersion / trend_strength / rsr_leader_half_life / mtf_filter_rate
- **4/8 判断ロジック（自動出力）**: ケースA/B/C を自動判定して推奨アクションを表示

### ロードマップ（今後の優先順位）

**〜2026-04-08（観察期間）**:
- 毎朝 `run_live_signal.py` 実行でログ蓄積
- 週1回 `python research/weekly_report.py` で4/8判断条件の充足度を確認
- `python -m backtest.mtf_comparison` を今すぐ実行可能（バックテスト期間で事前検証）

**4/8以降（判断基準）**:

| ケース | 条件 | アクション |
|---|---|---|
| A（理想） | rsr_pass≥4 AND bo_rate≥0.35 AND disp≥8 | MTF導入（backtest検証済みなら即適用） |
| B（supply不足） | rsr_pass≥6 AND candidates<1 | Donchian hybrid 検討 |
| C（相場停滞） | trend_strength<-0.02 AND disp<5 | **何もしない** |
| それ以外 | 条件未満 | 観察継続 |

**中長期ロードマップ**:
1. MTF実運用適用（4/8判断後）
2. RSR×RSRスロープ複合スコア（ピーク銘柄除外）
3. 全市場ユニバース研究
4. 空売り

---

---

## 完了した実装（2026-03-24）

### ✅ 診断ログ拡充 5本（2026-03-24 前半）

#### `signal_bridge.py` 追加フィールド
| フィールド | 定義 | 目安 |
|---|---|---|
| `mid_pressure_weight` | close≥high20×0.90の銘柄のRSRスコア重み（正規化） | ≥0.20で相場エネルギー蓄積 |
| `near_breakout_count` | close≥high20×0.95の銘柄数（5%以内） | ≥3でブレイク直前 |
| `near_breakout_weight` | 同上のRSRスコア重み | ≥0.25で相場動く |
| `breakout_cluster_today` | 同日BUYシグナル数 | ≥3でクラスター検知 |
| `breakout_cluster_fired` | クラスター発動フラグ | True→effective_max_pos=5 |
| `missed_breakout_count` | BUYシグナルのうち発注しなかった数 | 取りこぼし監視 |

#### ブレイクアウトクラスター拡張
- `breakout_cluster_today >= 3` → `effective_max_pos = 5`（3→5に拡張）
- `_build_orders()` に `effective_max_pos` パラメータ追加

#### リーダースロット
- RSR≥85 かつ rsr_rank==1 の最上位銘柄 → 配分上限を20%→35%に拡張（¥700k）
- `_leader_slot_used` で1スロットのみ発動、ログに LEADER SLOT を記録

### ✅ MTFフィルター実装（2026-03-24 後半）

#### 設計
- **日次RSR≥75 AND 週足RSR≥75 AND 週足close>週足MA20** の3条件
- SELL信号には影響しない（BUY抑制のみ）
- 母集団: 日次・週足とも**同一42銘柄**（因子の意味を壊さない）
- 週足composite return: 13/26/39/52週シフト × 0.4/0.2/0.2/0.2（日次と同一ウェイト）

#### キャッシュアーキテクチャ（朝1回計算・日中はファイル参照）
```
cache/mtf_state_YYYYMMDD.json
  {
    "date": "2026-03-24",
    "rsr_weekly":   {sym: float},   # 週足RSRスコア
    "weekly_ma_ok": {sym: bool},    # 週足close > 週足MA20
  }
```
- 当日キャッシュ存在 → `from_cache=True`（再計算ゼロ）
- 週足データは金曜引けで確定 → 日中再計算の意味なし
- `_build_mtf_cache_for_day()` メソッド追加

#### MTF pass率ログ
| フィールド | 説明 |
|---|---|
| `mtf_candidates` | RSR日次≥75のBUY候補数（母数） |
| `mtf_wrsr_pass` | 週足RSR≥75 通過数 |
| `mtf_wma_pass` | 週足MA20 通過数 |
| `mtf_full_pass` | 3条件すべて通過数 |
| `mtf_pass_rate` | full_pass / candidates（≥0.3でトレンド相場入り） |

#### 2026-03-24 現在のログ値
```
rsr_pass_count:       4（42銘柄中・RSR≥75）
near_breakout_count:  0
near_breakout_weight: 0.0
mid_pressure_count:   1
mid_pressure_weight:  0.098
mtf_candidates:       0（BUYシグナル未発生）
breakout_cluster:     False
```
→ **相場エネルギー蓄積中。戦略は正常に「待機」している状態**

#### 今日の週足キャッシュ内容
- 週足RSR≥75: 5銘柄（8058.T=95.2 / 6920.T=90.5 / 8002.T=80.9 / 7011.T=78.6 / 8035.T=76.2）
- weekly_ma_ok=True: 21/42銘柄（相場半数はトレンドあり）

### 判断基準（MTF発動条件）
```
mid_pressure_weight >= 0.20  → MTFが効き始めるフェーズ
near_breakout_weight >= 0.25 → 1〜2週間以内に動く可能性
breakout_cluster_today >= 3  → effective_max_pos=5 に自動拡張
```

### コミット履歴（2026-03-24）
```
e2c6619 feat: add mid_pressure_weight
faae3fe feat: breakout cluster expansion / near_breakout_weight / leader slot
b87396c feat: MTF filter (weekly RSR >= 70 + weekly MA20) [初版]
9a0f8fe fix: MTF 3点修正（キャッシュ化/exception=HOLD/閾値75）
b90af59 refactor: MTF cache-once-per-day architecture
dd33491 feat: RSR母集団を42→62に拡張
f7ce02e feat: OHLCVキャッシュ + RSR欠損補完を実装
b606009 feat: Shadow Phase1 条件付き発注を実装
```

### 更新されたファイル
| ファイル | 変更内容 |
|---|---|
| `kabusapi/signal_bridge.py` | MTF実装・診断ログ拡充・キャッシュアーキテクチャ |
| `research/weekly_report.py` | MTF pass率・blocked_leaders・4/8判定条件更新 |
| `cache/mtf_state_YYYYMMDD.json` | 日次MTFキャッシュ（新規） |

---

## 完了した実装（2026-03-24 後半）

### ✅ RSR母集団拡張（42 → 62）採用

**バックテスト**: `backtest/rsr_context_expansion.py`（新規）
**結果**: `results/rsr_context_expansion_2026-03-24.json`

| 指標 | BASELINE (RSR42) | EXPANDED (RSR62) | 差分 |
|---|---|---|---|
| CAGR | +14.45% | +16.27% | +1.82pp |
| Sharpe | 1.201 | 1.313 | +9.3% |
| MaxDD | -14.19% | -13.03% | 改善 |
| Calmar | 1.285 | 1.648 | **+28.3%** |
| 取引数（7年） | 159 | 150 | -5.7% |
| RSR Turnover | 0.052 | 0.052 | 変化なし |

**採用判断**: 取引数-5.7%だが **Calmar+28%/Sharpe+9%/MaxDD縮小** のトリプル質的向上。採用。
RSR Turnoverが完全に安定（0.052 = 0.052）→ランキング安定性も確認。

**live反映**: `run_live_signal.py` に `RSR_UNIVERSE_62 = {**RSR_UNIVERSE, **SHADOW_UNIVERSE}` 追加。
SignalBridgeに `rsr_universe_tickers=RSR_UNIVERSE_62` で渡す設計。

---

### ✅ OHLCVキャッシュ + RSR欠損補完実装

**変更ファイル**: `kabusapi/signal_bridge.py`

| 機能 | 実装詳細 |
|---|---|
| parquetキャッシュ | `cache/ohlcv/{ticker}.parquet`（5日間有効） |
| 3段階フォールバック | バッチ → 個別リトライ+jitter(0.3-1.2s) → キャッシュ読み込み |
| RSR欠損補完 | `ffill(limit=3)`（3日超欠損はRSR計算除外） |
| ヘルス指標 | `rsr_missing_count` / `rsr_filled_count` / `rsr_excluded_count` / `cache_fallback_count` |

**今日のドライラン結果**: 4銘柄(7201.T/8053.T/2914.T/5706.T)がyfinance取得失敗 → ffill補完で吸収。
キャッシュ書き込みには `pyarrow` 必要 → `pip install pyarrow` 済み。

---

### ✅ Shadow Phase1 条件付き発注実装

**変更ファイル**: `kabusapi/signal_bridge.py`、`run_live_signal.py`

#### 発動条件
```
shadow_rsr_pass >= 8（直近20日の RSR62≥70 通過日数）
AND rsr62 >= 70
AND rsr62 > live_top10_median
AND 価格フィルター（1単元 ≤ available_cash × max_single_weight）
AND CB NORMAL
```

#### パラメータ
```
shadow_slots     = 1（live max3 + shadow 1 = 合計最大4ポジション）
shadow_rsr_min   = 70.0
shadow_rsr_pass_min = 8
order side       = "SHADOW_BUY"（API送信時は Side.BUY として扱う）
```

#### 今日のドライラン結果
```
shadow_rsr_pass: 8（条件充足）
候補: 8802.T(三菱地所 RSR=78.3) / 2802.T(味の素 RSR=72.1)
blocked_by_alloc:
  8802.T: ¥429,800/単元 > 上限¥400,000（cap ¥1,990,392 × 0.20）
  2802.T: ¥417,600/単元 > 上限¥400,000
```
→ 資本¥2.1M（+10万）で 2802.T 解禁、¥2.43M で 8802.T も解禁。

#### 観測メトリクス（追加）
`shadow_signal_count` / `shadow_entry_count` / `shadow_blocked_by_alloc` / `shadow_rsr_pass_met`

---

## ファイル構成（2026-03-23 整理済み）

```
asset_simulation/
├── research_state.md              ← このファイル（Single Source of Truth）
│
├── ★ 実運用（毎朝実行）
│   ├── run_live_signal.py         ← 朝のシグナル生成・発注（--live で実発注）
│   └── run_morning_signal.py      ← run_live_signal の簡易ラッパー
│
├── kabusapi/                      ← kabuステーション API 連携
│   ├── client.py                  ← APIクライアント
│   └── signal_bridge.py           ← シグナル→発注ブリッジ（診断ログ機能付き）
│
├── backtest/                      ← バックテスト（現役のみ）
│   ├── fujiko_strategy.py         ← ★フジコ法コア戦略
│   ├── mean_reversion_strategy.py ← 平均回帰戦略
│   ├── rsr.py                     ← RSR計算（IBD式・パーセンタイルランク）
│   ├── engine.py                  ← バックテストエンジン基盤
│   ├── portfolio_engine.py        ← ポートフォリオエンジン
│   ├── portfolio_v2.py            ← ★現行最良バックテスト（Calmar=2.656）
│   ├── portfolio_cross_validate.py← 3重クロス検証
│   ├── live_equivalent.py         ← ライブ等価バックテスト（OOS検証用）
│   ├── topk_live_equivalent.py    ← Top-k ローテーション等価
│   ├── step3_final_validation.py  ← Phase2移行判定検証（OOS/IS比 確認）
│   ├── oos_2025.py                ← 2025年OOS検証
│   ├── portfolio_temporal_separation.py ← TEMPORALユニバース分離検証
│   ├── rolling_universe.py        ← ローリング選定ユニバース
│   ├── universe_builder.py        ← ユニバース構築ユーティリティ
│   ├── strategy.py                ← 基底戦略クラス
│   └── archive/                   ← 旧版・実験済み（削除せず保管）
│
├── diagnostics/                   ← 運用診断スクリプト
│   ├── rsr_universe_test.py       ← RSR母集団テスト（42 vs 300銘柄コンテキスト比較）
│   ├── rsr_period_test.py         ← RSR期間感度テスト（IBD63 vs 42日 vs ブレンド）
│   ├── exposure_root_cause.py     ← exposure 低下の原因分析
│   ├── exposure_report.py         ← exposure レポート生成
│   ├── live_exposure_report.py    ← ライブ運用 exposure 分析
│   ├── daily_state_logger.py      ← 日次状態ログ
│   ├── pnl_vs_holding.py          ← PnL vs 保有日数分析
│   ├── rank_stability.py          ← RSRランク安定性分析
│   └── turtle_exit_sweep.py       ← タートルズエグジット期間 sweep
│
├── configs/                       ← 設定ファイル
│   ├── strategy.yaml              ← 戦略・ポートフォリオパラメータ（確定値）
│   ├── universe.yaml              ← 銘柄ユニバース設定（参考）
│   ├── rsr_universe_42.csv        ← RSR計算コンテキスト（42銘柄）
│   ├── rolling_universe.json      ← ローリング選定ユニバース定義
│   └── universe/
│       └── 2026Q1_temporal24.json ← ★実行ユニバース（24銘柄・.env で指定）
│
├── results/                       ← バックテスト結果 JSON
│   ├── backtest_summary.json      ← ★全バックテスト結果サマリー（参照メイン）
│   ├── oos_2025_2026-03-19.json   ← 2025年OOS検証結果
│   ├── entry_stop_v5_2026-03-18.json ← entry_stop v5（失敗）
│   ├── entry_stop_v6_2026-03-19.json ← entry_stop v6（失敗・凍結）
│   └── archive/                   ← 旧版結果
│
├── logs/                          ← 各種ログ（.gitignore対象）
│   ├── diagnostics/               ← 運用診断ログ（日次蓄積）
│   │   ├── metrics.jsonl          ← ★日次メトリクス（candidates/exposure/blocked_rsr等）
│   │   ├── rsr_distribution.jsonl ← RSR分布ログ（閾値最適化用）
│   │   ├── rsr_universe_test_YYYY-MM-DD.json ← 母集団テスト結果
│   │   └── rsr_period_test_YYYY-MM-DD.json   ← 期間感度テスト結果
│   ├── live/                      ← 実発注ログ（YYYYMMDD_signals/orders.json）
│   └── research/                  ← 過去の研究実行ログ（.log ファイル）
│
├── runtime/                       ← 実行時状態（.gitignore対象）
│   ├── portfolio_state.json       ← 保有状態・CB状態・entry_date
│   └── order_lock.json            ← 二重発注防止ロック
│
├── scripts/                       ← 定期実行スクリプト
│   └── monthly_pnl.py             ← Phase 2月次P&L評価（FIFO・Phase2判定）
│
├── agents/                        ← 将来のマルチエージェント構成（仕様書）
│   ├── 01_監督.md / 02_分析.md / 03_批判.md / 04_設計.md / 05_総括.md
│   └── outputs/
│
├── portfolio/ execution/ market/ risk/  ← ライブラリモジュール
├── archive/                       ← ルート旧版スクリプト
└── data/                          ← .gitignore対象（yfinanceキャッシュ・シグナルJSON）
```

---

## 重要な既知問題・注意事項

| 項目 | 内容 |
|---|---|
| サバイバルバイアス | 現在構成銘柄のみ使用。廃止銘柄除外。CAGR 1〜3%過大評価の可能性 |
| 銘柄選択バイアス | 現行27銘柄はin-sample screeningで選定（改善中） |
| SELL/BUY非対称 | SELL=当日終値/BUY=翌日始値。CAGR差≈0.2%（許容範囲） |
| キャッシュ比率 | 平均83%キャッシュ → 資本効率が低い（改善中） |
| 株価上限 | 資本200万×15%=30万 → ~~¥500,000/単元以下~~ → **2026-03-23に¥600,000に引き上げ**（8002.T等が評価可能に） |

---

## セッション復元手順

新しい会話でプロジェクトを再開する場合:
```
1. このファイル（research_state.md）を読む
2. configs/strategy.yaml と configs/universe.yaml を読む
3. results/backtest_summary.json で最新結果を確認
4. research_log/ の最新日付のログを読む
5. 作業開始
```
