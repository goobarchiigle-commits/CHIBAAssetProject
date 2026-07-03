# CHIBAAssetProject 戦略仕様書

**作成日**: 2026-04-13
**生成スクリプト**: `src/generate_strategy_spec.py`（実コード直接参照・推測なし）
**対象戦略**: フジコ法 × 動的ユニバース（RSR42ベース）
**ステータス**: Phase 2 ライブ実運用中

> **注記**: 本ドキュメントは実コード・設定ファイルから自動生成されます。
> 数値はすべてソースコードまたは `src/configs/strategy.yaml` の実装値です。

---

## 目次

1. [資金・コスト設定](#1-資金コスト設定)
2. [ユニバース仕様](#2-ユニバース仕様)
3. [動的ユニバース選定](#3-動的ユニバース選定)
4. [RSR（相対強度指標）の計算方式](#4-rsr相対強度指標の計算方式)
5. [SEPA 8条件（銘柄選定フィルター）](#5-sepa-8条件銘柄選定フィルター)
6. [エントリー条件](#6-エントリー条件)
7. [エグジット条件（判定順）](#7-エグジット条件判定順)
8. [平均回帰サブ戦略](#8-平均回帰サブ戦略)
9. [ポジションサイジング](#9-ポジションサイジング)
10. [セクター・クラスター制御](#10-セクタークラスター制御)
11. [サーキットブレーカー](#11-サーキットブレーカー)
12. [マーケットショック制御](#12-マーケットショック制御)
13. [バックテスト評価指標](#13-バックテスト評価指標)
14. [ウォークフォワード検証](#14-ウォークフォワード検証)
15. [注文実行仕様（ライブ）](#15-注文実行仕様ライブ)

---

## 1. 資金・コスト設定

ソース: `src/configs/strategy.yaml` / `src/backtest/composite_alpha_bt.py`

| パラメータ | 値 | 出典 |
|---|---|---|
| 初期資本 | **3,000,000円** | `strategy.yaml: portfolio.capital` |
| スリッページ | **0.100%** | `composite_alpha_bt.py: SLIPPAGE` |
| 手数料 | **0.0550%** | `composite_alpha_bt.py: COMMISSION` |
| 片道コスト合計 | **0.1550%** | `SLIPPAGE + COMMISSION` |
| 最低手数料 | 99円 | `strategy.yaml: costs.min_commission` |
| 注文単位 | 100株 | `composite_alpha_bt.py: LOT` |

---

## 2. ユニバース仕様

ソース: `src/configs/rsr_universe_42.csv` / `src/backtest/composite_alpha_bt.py`

**固定プール**: RSR42（42銘柄）
**選定基準**: TOPIX100から取引所一部上場・流動性基準を満たす銘柄を42銘柄に絞り込んだ固定プール

### セクター別戦略割り当て

ソース: `composite_alpha_bt.py: SECTOR_STRATEGY`

| 戦略タイプ | 割り当てセクター |
|---|---|
| フジコ法（モメンタム） | ゲーム, レジャー, 商社, 機械, 海運, 電機, 電機精密, 食品 |
| 平均回帰 | ガス, 保険, 化学, 小売, 輸送機器, 鉄鋼, 銀行 |

### 銘柄一覧

| コード | 銘柄名 | セクター |
|---|---|---|
| 8035.T | — | 電機精密 |
| 6645.T | — | 電機精密 |
| 6702.T | — | 電機 |
| 6501.T | — | 電機 |
| 6762.T | — | 電機精密 |
| 6920.T | — | 電機精密 |
| 7203.T | — | 輸送機器 |
| 7201.T | — | 輸送機器 |
| 9432.T | — | 情報通信 |
| 8306.T | — | 銀行 |
| 8411.T | — | 銀行 |
| 8309.T | — | 銀行 |
| 7182.T | — | 銀行 |
| 8725.T | — | 保険 |
| 8058.T | — | 商社 |
| 8053.T | — | 商社 |
| 8002.T | — | 商社 |
| 8001.T | — | 商社 |
| 4021.T | — | 化学 |
| 2914.T | — | 食品 |
| 7011.T | — | 機械 |
| 3382.T | — | 小売 |
| 5401.T | — | 鉄鋼 |
| 5411.T | — | 鉄鋼 |
| 9531.T | — | ガス |
| 9101.T | — | 海運 |
| 9104.T | — | 海運 |
| 6857.T | — | 電機精密 |
| 6146.T | — | 電機精密 |
| 6981.T | — | 電機精密 |
| 6479.T | — | 電機精密 |
| 6506.T | — | 機械 |
| 7012.T | — | 機械 |
| 7013.T | — | 機械 |
| 9107.T | — | 海運 |
| 8015.T | — | 商社 |
| 6869.T | — | 電機精密 |
| 6594.T | — | 電機精密 |
| 8354.T | — | 銀行 |
| 5706.T | — | 鉄鋼 |
| 3197.T | — | 小売 |
| 4055.T | — | 情報通信 |

---

## 3. 動的ユニバース選定

ソース: `src/strategy/universe.py`（確定設定: `dyn_rsr42_bear_rs0`、採用 2026-04-05）

### レジーム判定

```
持続 Bear 判定:
  TOPIX < MA200 かつ 直近60営業日のうち40日以上 MA200 を下回る

  ※ 短期クラッシュ（コロナ型・2ヶ月以内）では Bear scoring を適用しない
  ※ 持続型 Bear（2022型・60営業日以上）でのみ Bear scoring に切り替え

判定コード（universe.py: is_sustained_bear）:
  topix_lb     = topix_close.iloc[-60:]
  days_below   = (topix_lb < rolling_ma200.iloc[-60:]).sum()
  is_sustained_bear = days_below >= 40
```

### Bull スコアリング（持続 Bear 以外）

```python
# universe.py: build_sym_active_df（Bull分岐）
LOSS_PENALTY_COEF = 0.10   # 直近90日損失銘柄ペナルティ係数
LOSS_PERIOD       = 90

score = (
    0.40 * zscore(mom_63d)      # 63営業日（≒3ヶ月）モメンタム
    + 0.35 * zscore(rsr)                  # RSR（ユニバース内パーセンタイル）
    + 0.25 * zscore(log_vol_20d) # 直近20日平均出来高の対数
    - 0.10 * zscore(loss_90d)             # 直近90日損失ペナルティ
)
→ Top 30 銘柄を月次で更新
```

### Bear スコアリング（持続 Bear 時）

```python
# universe.py: build_sym_active_df（Bear分岐）
# 前提フィルター: rs_topix > 0（TOPIX比プラスの銘柄のみ対象）
score = (
    0.50 * zscore(rs_topix)               # TOPIX比相対リターン（最重視）
    + 0.30 * zscore(rsr)
    + 0.20 * zscore(log_vol_20d)
    - 0.10 * zscore(loss_90d)
)
→ rs_topix > 0 フィルター後 Top 20 銘柄を月次で更新
```

### Bear 時 セクター除外

ソース: `strategy.yaml: bear_universe_filter`

```
除外セクター（Bear 持続時）:
    （設定なし）
```

### 先読み防止

```
月 T の選択は 月 T-1 末データで計算（月初1日前のデータを参照）
コード: eval_dt = close_all.index[pos - 1]  （monthly_first[key] の1日前）
```

---

## 4. RSR（相対強度指標）の計算方式

ソース: `src/backtest/rsr.py: calc_composite_return, calc_universe_rsr`

### IBD式加重12ヶ月リターン

```python
# rsr.py: calc_composite_return
r1 = prices / prices.shift(63)   - 1   # 直近 3ヶ月（63営業日）     × 40%
r2 = prices.shift(63)  / prices.shift(126) - 1   # 3〜6ヶ月前        × 20%
r3 = prices.shift(126) / prices.shift(189) - 1   # 6〜9ヶ月前        × 20%
r4 = prices.shift(189) / prices.shift(252) - 1   # 9〜12ヶ月前       × 20%
composite_return = 0.4*r1 + 0.2*r2 + 0.2*r3 + 0.2*r4
```

### ユニバース内ランク変換

```python
# rsr.py: calc_universe_rsr
rsr_df = comp_df.rank(axis=1, pct=True) * 100   # 0〜100 スケール
# 各取引日のクロスセクション・パーセンタイルランク
```

### RSRモメンタム

```python
# fujiko_strategy.py: precompute_signals
mom_period = 21   # strategy.yaml: fujiko.mom_period
mom_arr    = rsr_arr - roll(rsr_arr, 21)   # 21日前との差分
# エントリー条件: mom > 0 かつ mom > mom_prev（上昇中）
# エグジット条件: mom < 0 かつ mom < mom_prev（下降中）
```

---

## 5. SEPA 8条件（銘柄選定フィルター）

ソース: `src/backtest/fujiko_strategy.py: _calc_sepa_score_array`

```python
# 8条件それぞれ 0 or 1 のスコア（合計 0〜8 点）
score[0] = Close > MA150  AND  Close > MA200          # トレンド上位
score[1] = MA150 > MA200                              # 長期トレンド整列
score[2] = MA200 > MA200[21日前]                      # MA200 が上向き
score[3] = MA50  > MA50[1日前]                        # MA50 が上向き
score[4] = Close > MA50                               # 中期トレンド上位
score[5] = Close >= 52週安値 × 1.30                   # 52週安値から+30%以上
score[6] = Close >= 52週高値 × 0.75                   # 52週高値から-25%以内
score[7] = RSR   >= 70.0                              # RSR 70以上（SEPA独自条件）
```

| 判定 | スコア | 意味 |
|---|---|---|
| キング | 8点 | 全条件クリア・最高品質 |
| **エース（採用閾値）** | **6点以上** | `strategy.yaml: fujiko.min_sepa = 6` |
| 対象外 | 5点以下 | エントリー不可 |

---

## 6. エントリー条件

ソース: `src/backtest/fujiko_strategy.py: precompute_signals`

以下の条件を**すべて同時に**満たした場合にエントリーシグナル（+1）を発生。
実行価格: **翌営業日の始値**（寄付成行 または 寄成注文）

```python
# fujiko_strategy.py: precompute_signals
entry_mask  = sepa_score_arr >= 6           # 条件1: SEPA 6点以上
entry_mask &= rsr_arr        >= 75.0            # 条件2: RSR >= 75.0
entry_mask &= (mom_arr > 0) & (mom_arr > mom_prev)  # 条件3: RSRモメンタム 正かつ上昇
if use_turtle_entry:  # strategy.yaml: fujiko.use_turtle_entry = True
    entry_mask &= close > prev_20d_high # 条件4: 20日高値ブレイクアウト
```

### エントリー条件一覧（確定値）

| # | 条件 | 閾値 | 出典 |
|---|---|---|---|
| 1 | SEPA スコア | ≥ **6**（8点中） | `strategy.yaml: fujiko.min_sepa` |
| 2 | RSR（ユニバース内ランク） | ≥ **75.0** | `strategy.yaml: fujiko.min_rsr` |
| 3 | RSRモメンタム（21日差分） | > 0 かつ 前日比上昇 | `strategy.yaml: fujiko.mom_period = 21` |
| 4 | タートルズ S1 ブレイクアウト | 前日までの **20日高値** 超え | `strategy.yaml: fujiko.turtle_entry = 20` |
| 5 | 動的ユニバース活性 | 当月の active リストに含まれる | `strategy/universe.py: sym_active_mat` |
| 6 | 流動性フィルター（ライブのみ） | 日次売買代金 ≥ **5,000,000,000円** | `signal_bridge.py: MIN_DAILY_VALUE_YEN` |
| 7 | MTF フィルター（ライブのみ） | 週足 RSR ≥ 75.0 かつ 週足終値 > 週足 MA20 | `signal_bridge.py: MTFフィルター` |

> **エグジット vs エントリーの優先順位**: 同日にエグジットとエントリーが競合した場合、エグジットが優先される。

---

## 7. エグジット条件（判定順）

ソース: `src/kabusapi/signal_bridge.py`（ライブ実装・優先順位確定版）

以下の順番で判定し、**最初にTrueになった条件で即時エグジット**（翌営業日始値）。

```
優先順位: composite_shock > トレーリングストップ > 時間ストップ
        > RSR低下エグジット > mean_rev反発失敗 > 戦略シグナル（RSRモメンタム/タートル）
```

### 判定順詳細

| 優先 | 条件名 | 発動ロジック | 出典 |
|---|---|---|---|
| **1** | **Composite Shock Exit** | TOPIX 日次リターン ≤ **-5%** かつ 個別株 日次リターン ≤ **-8%** | `signal_bridge.py: shock_exit_mode="composite"` |
| **2** | **トレーリングストップ** | 終値 < 保有期間最高終値 − **3.0 × ATR20** | `signal_bridge.py` 参照元: `composite_alpha_bt.py: TRAIL_ATR_MULT=3.0` |
| **3** | **時間ストップ** | 保有営業日数 ≥ **60日** | `strategy.yaml: risk.max_hold_days = 60` |
| **4** | **RSR 低下エグジット** | RSR < **75.0** かつ 保有 ≥ **3日** _(緊急時は min_hold 無視)_ | `strategy.yaml: fujiko.min_rsr / risk.min_hold_days` |
| **5** | **緊急エグジット** | 含み損 ≤ **-8%**（min_hold 無視で即時発動） | `strategy.yaml: risk.emergency_exit_pct = -0.08` |
| **6** | **mean_rev 反発失敗** | 平均回帰エントリー後 **4営業日** 以内に high が +**1%** 未達 かつ 終値 < エントリー×0.995 | `signal_bridge.py: MEANREV_FAIL_DAYS / MEANREV_MIN_BOUNCE` |
| **7** | **戦略シグナル（フジコ法）** | RSRモメンタム < 0 かつ 前日比下降 _または_ 終値 < **前日までの55日安値** | `fujiko_strategy.py: exit_mask` |

### バックテストエンジン上の追加エグジット

ソース: `src/backtest/composite_alpha_bt.py: run_scenario`（STEP5 構成）

| 条件 | パラメータ | 出典 |
|---|---|---|
| トレーリング（STEP5） | 終値 < **50日最高終値** − 3.0 × ATR20 | `TRAIL_PERIOD=50, TRAIL_ATR_MULT=3.0` |
| exit_params（RSR-z） | RSR-z < **1.1**（翌日始値）| `strategy.yaml: exit_params.rsr_exit = 1.1` |
| exit_params（タイムストップ） | 保有バー数 ≥ **4**（BTエンジン用）| `strategy.yaml: exit_params.time_stop = 4` |
| exit_params（トレイル） | HWM から **-2.5%** 下落 | `strategy.yaml: exit_params.trail_stop = 0.025` |
| TIME_STOP 後クールダウン | 5営業日 再エントリー禁止 | `composite_alpha_bt.py: REENTRY_COOL = 5` |

### エグジットとmin_holdの関係

```python
# signal_bridge.py（ライブ）
is_rank_exit = (
    rsr_now < min_rsr_threshold          # RSR が閾値未満
    and (hold_td >= self.min_hold_days   # min_hold_days = 3 営業日以上
         or is_emergency_exit)           # 緊急時は hold 無視
)
```

---

## 8. 平均回帰サブ戦略

ソース: `src/backtest/mean_reversion_strategy.py`（`MR_PARAMS` は `composite_alpha_bt.py`）

### 割り当てセクター

```python
# composite_alpha_bt.py: SECTOR_STRATEGY
mean_rev セクター: ['ガス', '保険', '化学', '小売', '輸送機器', '鉄鋼', '銀行']
```

### エントリー条件

```python
# mean_reversion_strategy.py
1. RSI(5日) < 25.0       # 短期売られすぎ（Wilder EMA方式）
2. Close > MA200                  # 大局上昇トレンド内（トレンドフィルター）
3. Close > MA50 × 0.85                               # 落下するナイフ回避（-15%以内）
```

### エグジット条件

```python
# mean_reversion_strategy.py
A. RSI(5日) > 65.0     # 回復・利食い
B. Close < エントリー × (1 - 0.07)  # ストップロス -7%
C. 保有 ≥ 10営業日             # 時間切れ
```

---

## 9. ポジションサイジング

ソース: `src/backtest/composite_alpha_bt.py: run_scenario`

### サイジングモード（確定: 均等ウェイト）

```python
# composite_alpha_bt.py: sizing_mode = "existing"（キャッシュ均等割り）
# 1銘柄への投資額 = 利用可能キャッシュ / 空きスロット数
# 購入株数       = (投資額 / 翌日始値 // LOT) * LOT  # 100株単位に切り捨て
```

| パラメータ | 値 | 出典 |
|---|---|---|
| 最大同時保有数 | **3銘柄** | `strategy.yaml: portfolio.max_positions` |
| 1銘柄最大ウェイト | **25%** | `strategy.yaml: portfolio.max_single_weight` |
| symbol_cap（固定） | **40%** | `strategy.yaml: risk_controls.symbol_cap`（`dynamic_cap: False`） |
| 注文単位 | 100株 | `composite_alpha_bt.py: LOT` |

---

## 10. セクター・クラスター制御

ソース: `src/backtest/composite_alpha_bt.py: run_scenario` / `src/strategy/cluster.py`

### セクター集中制御

ソース: `strategy.yaml: risk_controls.sector_concentration`

```python
# composite_alpha_bt.py（動的ユニバース時のみ有効）
MAX_SECTOR_WEIGHT = 0.25   # sector_cap（Bear 時は bear_sector_cap）
MAX_SYMBOL_WEIGHT = 0.4   # symbol_cap

# Bear 適応 cap（TOPIX < MA200 の日）
bear_sector_cap  = 0.18   # strategy.yaml: risk_controls.bear_sector_cap
bear_cluster_cap = 0.25   # strategy.yaml: risk_controls.bear_cluster_cap
```

| 制御 | Bull 時 | Bear 時（TOPIX < MA200） | 出典 |
|---|---|---|---|
| セクター上限 | **25%** | **18%** | `risk_controls.sector_cap / bear_sector_cap` |
| クラスター上限 | **35%** | **25%** | `risk_controls.cluster_cap / bear_cluster_cap` |
| 同一セクター銘柄数 | ≤ **1銘柄** | ≤ **1銘柄** | `sector_concentration.max_names_per_sector` |
| セクター合計ウェイト | ≤ **35%** | ≤ **35%** | `sector_concentration.max_weight_per_sector` |

### クラスターマップ

ソース: `src/strategy/cluster.py: CLUSTER_MAP_DEFAULT`

| クラスター | 含まれるセクター |
|---|---|
| cyclical_macro | 卸売業, 機械, 鉄鋼, 銀行業, 保険業, 輸送用機器 |
| defensive | 情報・通信業, 医薬品, 食料品, 電気・ガス業 |
| growth_tech | 電気機器, 精密機器, サービス業 |
| real_asset | 建設業, 不動産業, 金属製品, 化学 |

### Gross Exposure 縦断制御

ソース: `strategy.yaml: risk_controls.gross_exposure_*` / `composite_alpha_bt.py`

```python
# gross_exposure_enabled = True
if   TOPIX_20d_return < -0.05: gross_cap = 0.6    # TOPIX 20日 -5% 時
elif TOPIX_60d_return < -0.08: gross_cap = 0.4    # TOPIX 60日 -8% 時
else:                          gross_cap = 1.0   # 通常時
```

| 状態 | Gross Exposure 上限 |
|---|---|
| 通常 | **100%** |
| TOPIX 20日リターン < −5% | **60%** |
| TOPIX 60日リターン < −8% | **40%** |

### クラスター相場制御（ライブのみ）

ソース: `src/kabusapi/signal_bridge.py`

```python
# signal_bridge.py
CLUSTER_LEVEL1_THRESH = 0.15   # cluster_density >= 15%: mean_rev BUY 停止
CLUSTER_LEVEL2_THRESH = 0.25   # cluster_density >= 25%: モメンタム偏重
```

---

## 11. サーキットブレーカー

ソース: `src/risk/circuit_breaker.py`

```python
# circuit_breaker.py
DD_TRIGGER  = -0.15    # ドローダウン がこれ以下 → ENTRY_STOP_ONLY（BUY 停止）
RECOVERY_DD = -0.05   # DD が回復したら NORMAL に復帰
MAX_CB_DAYS = 30   # 最大 30 営業日で強制解除

状態機械:
  NORMAL
    ↓ DD <= -15%
  ENTRY_STOP_ONLY  ← BUY 停止 / SELL は引き続き許可
    ↓ 30営業日経過 OR DD >= -5% に回復
  NORMAL
```

| パラメータ | 値 | 意味 |
|---|---|---|
| DD_TRIGGER | **-15%** | BUY 停止開始ライン |
| RECOVERY_DD | **-5%** | CB 解除ライン |
| MAX_CB_DAYS | **30営業日** | 強制解除タイムアウト |

---

## 12. マーケットショック制御

ソース: `src/backtest/composite_alpha_bt.py: run_scenario` / `src/kabusapi/signal_bridge.py`

### Composite モード（本番採用: `shock_exit_mode = "composite"`）

```python
# composite_alpha_bt.py
composite_market_thr = -0.05   # TOPIX 日次リターン -5% 以下でショック日と判定
composite_sym_thr    = -0.08   # 個別株 日次リターン -8% 以下のポジションのみ決済

# signal_bridge.py（ライブ）
_is_shock_day   = bench_ret_prev <= -0.05   # TOPIX -5%
# 個別株が -8% 以下の場合のみ翌日始値で決済
```

| 条件 | アクション |
|---|---|
| TOPIX 日次 ≤ −5% のみ | 新規 BUY 禁止 |
| TOPIX 日次 ≤ −5% **かつ** 個別株 ≤ −8% | 該当ポジション翌日始値で決済 |

---

## 13. バックテスト評価指標

ソース: `backtests/min_hold_sensitivity_2026-03-31.json`（hold3d = 確定パラメータ）

### 全体サマリー

| 指標 | IS（2020-2024） | OOS（2025） | Phase 1 基準 |
|---|---|---|---|
| CAGR | **22.4%** | **0.1%** | — |
| Sharpe | **1.582** | **0.067** | > 0.5 |
| MaxDD | **-12.32%** | **-10.29%** | < -20% |
| Calmar（CAGR / MaxDD） | **1.817** | **0.01** | > 1.0 |
| PF（勝率×平均利益 / 敗率×平均損失） | **2.68** | **1.46** | > 1.5 |
| 勝率 | **56.6%** | **54.2%** | — |
| R倍数（平均利益 / 平均損失） | **2.05** | **1.23** | — |
| 平均保有日数 | **11.8日** | **7.2日** | — |
| 年間取引数 | **51.2件/年**（219件合計） | — | ≥ 5 |
| 平均エクスポージャー | **37.0%** | **31.7%** | — |

### IS 年次リターン（2020-2024）

| 年 | リターン |
|---|---|
| 2020 | **+15.20%** |
| 2021 | **+36.81%** |
| 2022 | **+10.81%** |
| 2023 | **+19.63%** |
| 2024 | **+12.77%** |

### エグジット内訳（IS）

| 理由 | 件数 |
|---|---|
| STRATEGY_EXIT | 105件 |
| RSR_EXIT | 110件 |
| TIME_STOP | 4件 |

### 動的ユニバース採用後（2025 OOS 比較）

ソース: `backtests/step123_integration_2026-04-06.json`

| 指標 | ベース（固定） | 動的ユニバース採用後 | 改善幅 |
|---|---|---|---|
| 2025 CAGR | 0.042（×100%） | **0.123（×100%）** | — |
| 2025 MaxDD | -0.101 | **-0.037** | — |
| 2025 Sharpe | 0.453 | **1.612** | — |

---

## 14. ウォークフォワード検証

ソース: `backtests/wf_final_2026-04-04.json` / `backtests/wf_dyn_rsr42_2026-04-05.json`

### ベースライン WF（hold3d / turtle_exit=55 / min_rsr=75）

| Seg | IS 期間 | OOS 年 | IS Sharpe | OOS Sharpe | OOS/IS 比 | 合格 |
|---|---|---|---|---|---|---|
| 1 | 2018-2019 | 2020 | 0.000 | 1.225 | — | ✅ |
| 2 | 2019-2020 | 2021 | 1.225 | 0.454 | 0.371 | ✅ |
| 3 | 2020-2021 | 2022 | 0.600 | 0.375 | 0.625 | ✅ |
| 4 | 2021-2022 | 2023 | 0.596 | 1.460 | 2.450 | ✅ |
| 5 | 2022-2023 | 2024 | 0.929 | 0.791 | 0.851 | ✅ |

**総合**: 5/5 / 平均 OOS/IS 比 = **1.074**
**Full IS（2018-2024）**: CAGR=18.12% / Sharpe=0.783 / MaxDD=-18.24%
**True OOS（2025）**: CAGR=-0.98% / Sharpe=-0.075 / MaxDD=-9.94%

### 動的ユニバース WF（dyn_rsr42_bear_rs0）

| Seg | OOS 年 | OOS Sharpe | OOS MaxDD | 合格 |
|---|---|---|---|---|
| 1 | 2020 | 0.702 | -16.19% | ✅ |
| 2 | 2021 | 0.540 | -18.43% | ✅ |
| 3 | 2022 | 0.258 | -19.13% | ✅ |
| 4 | 2023 | 1.228 | -10.47% | ✅ |
| 5 | 2024 | 0.579 | -19.54% | ✅ |

**総合**: 5/5 / Full IS Sharpe = **0.812**
**True OOS 2025 Sharpe**: **0.805** / MaxDD: **-10.11%**

---

## 15. 注文実行仕様（ライブ）

ソース: `src/kabusapi/signal_bridge.py`

### 実行フロー

```
毎朝 8:30 頃:
  1. yfinance で前日終値データ取得（ローカルキャッシュ優先）
  2. RSR 計算 → 動的ユニバース活性リスト生成
  3. kabuステーション API でポジション・余力取得
  4. CB 状態評価（NORMAL / ENTRY_STOP_ONLY）
  5. シグナル生成（FujikoStrategy + MeanReversionStrategy + MTF フィルター）
  6. 注文リスト確定 → ドライラン確認
  7. --live --yes 付きの場合のみ実発注
```

### 発注制御

| パラメータ | 値 | 出典 |
|---|---|---|
| 最大新規 BUY/日 | 2件 | `signal_bridge.py: max_new_positions_per_day` |
| 発注レート制限 | **3件/分**（20秒/件） | `signal_bridge.py: ORDER_RATE_LIMIT_PER_MIN` |
| SELL 取引所 | **TSE（東証）** | `signal_bridge.py: Exchange.TSE if o.side == "SELL"` |
| BUY 取引所 | **SOR** | `signal_bridge.py: Exchange.SOR` |
| 注文種別（前場前） | 寄成（MARKET_OPEN） | `signal_bridge.py: market_hour < 9*60` |
| 注文種別（前場後） | 成行（MARKET） | `signal_bridge.py: otherwise` |
| デフォルト動作 | **ドライラン** | `--live --yes` が必要 |

### データ健全性チェック

```python
# signal_bridge.py
DATA_HEALTH_MIN_RATIO = 0.9   # RSR42 データ取得率 90% 未満でシグナル停止
```

---

## 付録: 確定パラメータ一覧

ソース: `src/configs/strategy.yaml`

| セクション | パラメータ | 値 |
|---|---|---|
| `fujiko` | `min_sepa` | 6 |
| `fujiko` | `min_rsr` | 75.0 |
| `fujiko` | `mom_period` | 21 |
| `fujiko` | `turtle_entry` | 20 |
| `fujiko` | `turtle_exit` | 55 |
| `fujiko` | `use_turtle_entry` | True |
| `portfolio` | `capital` | 3,000,000円 |
| `portfolio` | `max_positions` | 3 |
| `portfolio` | `max_single_weight` | 25% |
| `portfolio` | `max_dd_limit` | 15% |
| `risk` | `min_hold_days` | 3 |
| `risk` | `max_hold_days` | 60 |
| `risk` | `emergency_exit_pct` | -8% |
| `exit_params` | `time_stop` | 4バー |
| `exit_params` | `trail_stop` | 2.5% |
| `exit_params` | `rsr_exit` | 1.1 |
| `risk_controls` | `shock_exit_mode` | composite |
| `risk_controls` | `symbol_cap` | 40% |
| `risk_controls` | `sector_cap` | 25% |
| `risk_controls` | `cluster_cap` | 35% |
| `risk_controls` | `bear_sector_cap` | 18% |
| `risk_controls` | `bear_cluster_cap` | 25% |
| `risk_controls` | `gross_cap_normal` | 100% |
| `risk_controls` | `gross_cap_drawdown_5pct` | 60% |
| `risk_controls` | `gross_cap_drawdown_8pct` | 40% |
| `dynamic_universe` | `enabled` | True |
| `dynamic_universe` | `pool` | rsr42 |
| `dynamic_universe` | `bull_active_n` | 30 |
| `dynamic_universe` | `bear_active_n` | 20 |
| `dynamic_universe` | `bear_rs_filter` | True |

---

_生成スクリプト: `src/generate_strategy_spec.py` / 生成日時: 2026-04-13_
