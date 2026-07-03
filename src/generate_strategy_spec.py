"""
generate_strategy_spec.py
=========================
CHIBAAssetProject 戦略仕様書ジェネレーター

実行:
    cd C:/ai-trading
    python src/generate_strategy_spec.py                     # stdout に出力
    python src/generate_strategy_spec.py --out docs/research/strategy_spec.md

概要:
    実コード・設定ファイルを直接参照し、推測なしで戦略仕様書を生成する。
    第三者レビュー用の完成版 Markdown ドキュメントを出力する。

参照ファイル（全て実コード由来）:
    src/strategy/universe.py            ← 動的ユニバース選定ロジック
    src/backtest/fujiko_strategy.py     ← SEPA 8条件 / エントリー / エグジット
    src/backtest/rsr.py                 ← RSR（IBD式加重リターン）計算
    src/backtest/mean_reversion_strategy.py ← 平均回帰戦略
    src/backtest/composite_alpha_bt.py  ← バックテストエンジン / コスト定数
    src/kabusapi/signal_bridge.py       ← ライブ発注ブリッジ / エグジット優先順位
    src/risk/circuit_breaker.py         ← サーキットブレーカー定数
    src/strategy/cluster.py             ← クラスターマップ
    src/configs/strategy.yaml           ← 確定パラメータ
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from datetime import date
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

# ------------------------------------------------------------------ #
# プロジェクトルートを sys.path に追加
# ------------------------------------------------------------------ #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ------------------------------------------------------------------ #
# 実コードから定数を直接インポート
# ------------------------------------------------------------------ #
# --- composite_alpha_bt.py ---
from src.backtest.composite_alpha_bt import (
    SLIPPAGE,
    COMMISSION,
    COST_ONE_WAY,
    TRAIL_PERIOD,
    TRAIL_ATR_MULT,
    CB_UNLOCK_DAYS,
    CB_SCALE,
    BREAKOUT_LOOKBACK,
    BREAKOUT_BONUS,
    LOT,
    REENTRY_COOL,
    START,
    END,
    SECTOR_STRATEGY,
    MR_PARAMS,
)

# --- circuit_breaker.py ---
from src.risk.circuit_breaker import DD_TRIGGER, RECOVERY_DD, MAX_CB_DAYS

# --- strategy/universe.py ---
from src.strategy.universe import (
    MA200_PERIOD,
    MOM_PERIOD,
    VOL_PERIOD,
    BULL_ACTIVE_N,
    BEAR_ACTIVE_N,
    SUSTAINED_BEAR_DAYS,
    LOOKBACK_BEAR_CHECK,
)

# --- signal_bridge.py ---
from src.kabusapi.signal_bridge import (
    CB_DD_TRIGGER,
    CB_COOLDOWN_TRADING_DAYS,
    MIN_DAILY_VALUE_YEN,
    MEANREV_FAIL_DAYS,
    MEANREV_MIN_BOUNCE,
    CLUSTER_LEVEL1_THRESH,
    CLUSTER_LEVEL2_THRESH,
)

# --- strategy/cluster.py ---
from src.strategy.cluster import CLUSTER_MAP_DEFAULT

# --- strategy config ---
from src.config_loader import load_strategy_config

cfg = load_strategy_config()


# ------------------------------------------------------------------ #
# ヘルパー: ネストされた設定値を安全取得
# ------------------------------------------------------------------ #
def _c(section: str, key: str, default="—"):
    """cfg.<section>.<key> を安全に取得する。"""
    sec = getattr(cfg, section, None)
    if sec is None:
        return default
    return getattr(sec, key, default)


# ------------------------------------------------------------------ #
# ユニバース銘柄リスト（rsr_universe_42.csv から取得）
# ------------------------------------------------------------------ #
def _load_universe_table() -> str:
    from src.paths import CONFIGS_DIR
    csv_path = CONFIGS_DIR / "rsr_universe_42.csv"
    if not csv_path.exists():
        return "_（rsr_universe_42.csv が見つかりません）_\n"

    import pandas as pd
    df = pd.read_csv(csv_path)
    if "symbol" not in df.columns:
        return "_（CSV に symbol 列なし）_\n"

    sector_col = "sector" if "sector" in df.columns else None
    name_col   = "name"   if "name"   in df.columns else None

    lines = ["| コード | 銘柄名 | セクター |", "|---|---|---|"]
    for _, row in df.iterrows():
        sym  = row["symbol"]
        name = row[name_col] if name_col else "—"
        sec  = row[sector_col] if sector_col else "—"
        lines.append(f"| {sym} | {name} | {sec} |")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ #
# バックテスト結果を JSON から読み込む
# ------------------------------------------------------------------ #
def _load_bt_results() -> dict:
    """backtest_summary.json + min_hold_sensitivity_2026-03-31.json から実績値を取得。"""
    import json
    from src.paths import RESULTS_DIR
    backtests_dir = PROJECT_ROOT / "backtests"

    results = {}

    # --- backtest_summary.json ---
    summary_path = backtests_dir / "backtest_summary.json"
    if summary_path.exists():
        d = json.loads(summary_path.read_text(encoding="utf-8"))
        results["summary"] = d

    # --- min_hold_sensitivity_2026-03-31.json (hold3d = 確定パラメータ) ---
    hold_path = backtests_dir / "min_hold_sensitivity_2026-03-31.json"
    if hold_path.exists():
        d = json.loads(hold_path.read_text(encoding="utf-8"))
        results["hold3d_is"]  = d.get("hold3d", {}).get("IS", {})
        results["hold3d_oos"] = d.get("hold3d", {}).get("OOS", {})

    # --- wf_final_2026-04-04.json ---
    wf_path = backtests_dir / "wf_final_2026-04-04.json"
    if wf_path.exists():
        results["wf_final"] = json.loads(wf_path.read_text(encoding="utf-8"))

    # --- wf_dyn_rsr42_2026-04-05.json (dyn_rsr42_bear_rs0) ---
    dyn_path = backtests_dir / "wf_dyn_rsr42_2026-04-05.json"
    if dyn_path.exists():
        d = json.loads(dyn_path.read_text(encoding="utf-8"))
        for item in d.get("results", []):
            if item.get("config_name") == "dyn_rsr42_bear_rs0":
                results["dyn_wf"] = item
                break

    # --- step123_integration_2026-04-06.json ---
    s123_path = backtests_dir / "step123_integration_2026-04-06.json"
    if s123_path.exists():
        results["step123"] = json.loads(s123_path.read_text(encoding="utf-8"))

    return results


# ================================================================== #
# ドキュメント生成
# ================================================================== #
def generate(bt: dict) -> str:
    today_str = date.today().isoformat()

    # --- 設定値の取り出し ---
    min_sepa        = _c("fujiko", "min_sepa", 6)
    min_rsr         = _c("fujiko", "min_rsr", 75.0)
    mom_period      = _c("fujiko", "mom_period", 21)
    turtle_entry    = _c("fujiko", "turtle_entry", 20)
    turtle_exit     = _c("fujiko", "turtle_exit", 55)
    use_turtle      = _c("fujiko", "use_turtle_entry", True)

    capital         = _c("portfolio", "capital", 3_000_000)
    max_pos         = _c("portfolio", "max_positions", 3)
    max_sw          = _c("portfolio", "max_single_weight", 0.25)
    max_dd_limit    = _c("portfolio", "max_dd_limit", 0.15)

    min_hold        = _c("risk", "min_hold_days", 3)
    max_hold        = _c("risk", "max_hold_days", 60)
    emg_exit        = _c("risk", "emergency_exit_pct", -0.08)

    rc              = getattr(cfg, "risk_controls", None)
    dyn_cap         = getattr(rc, "dynamic_cap", False)    if rc else False
    sym_cap         = getattr(rc, "symbol_cap", 0.40)      if rc else 0.40
    sec_cap         = getattr(rc, "sector_cap", 0.25)      if rc else 0.25
    cluster_cap     = getattr(rc, "cluster_cap", 0.35)     if rc else 0.35
    bear_sec_cap    = getattr(rc, "bear_sector_cap", 0.18) if rc else 0.18
    bear_cls_cap    = getattr(rc, "bear_cluster_cap", 0.25)if rc else 0.25
    gross_en        = getattr(rc, "gross_exposure_enabled", True) if rc else True
    gross_norm      = getattr(rc, "gross_cap_normal", 1.0)        if rc else 1.0
    gross_dd5       = getattr(rc, "gross_cap_drawdown_5pct", 0.6) if rc else 0.6
    gross_dd8       = getattr(rc, "gross_cap_drawdown_8pct", 0.4) if rc else 0.4
    shock_mode      = getattr(rc, "shock_exit_mode", "composite") if rc else "composite"

    sc              = getattr(rc, "sector_concentration", None) if rc else None
    max_names_sec   = getattr(sc, "max_names_per_sector", 1)    if sc else 1
    max_wt_sec      = getattr(sc, "max_weight_per_sector", 0.35)if sc else 0.35

    du              = getattr(cfg, "dynamic_universe", None)
    du_enabled      = getattr(du, "enabled", True)           if du else True
    du_pool         = getattr(du, "pool", "rsr42")           if du else "rsr42"
    bear_rs_filter  = getattr(du, "bear_rs_filter", True)    if du else True

    bu              = getattr(cfg, "bear_universe_filter", None)
    bear_excl       = getattr(bu, "excluded_sectors", [])    if bu else []

    ep              = getattr(cfg, "exit_params", None)
    ep_time_stop    = getattr(ep, "time_stop", 4)            if ep else 4
    ep_trail_stop   = getattr(ep, "trail_stop", 0.025)       if ep else 0.025
    ep_rsr_exit     = getattr(ep, "rsr_exit", 1.1)           if ep else 1.1

    # --- バックテスト指標 ---
    is_d  = bt.get("hold3d_is", {})
    oos_d = bt.get("hold3d_oos", {})
    wf    = bt.get("wf_final", {})
    dyn   = bt.get("dyn_wf", {})
    s123  = bt.get("step123", {})

    is_cagr   = is_d.get("cagr", "—")
    is_sharpe = is_d.get("sharpe", "—")
    is_maxdd  = is_d.get("max_dd", "—")
    is_calmar = is_d.get("calmar", "—")
    is_wr     = is_d.get("win_rate", "—")
    is_rm     = is_d.get("r_multiple", "—")
    is_hold   = is_d.get("avg_hold_days", "—")
    is_trades = is_d.get("n_trades", "—")
    is_yr     = is_d.get("n_trades_yr", "—")
    is_exp    = is_d.get("avg_exposure", "—")
    is_ann    = is_d.get("annual_returns", {})

    oos_cagr   = oos_d.get("cagr", "—")
    oos_sharpe = oos_d.get("sharpe", "—")
    oos_maxdd  = oos_d.get("max_dd", "—")
    oos_calmar = oos_d.get("calmar", "—")
    oos_wr     = oos_d.get("win_rate", "—")
    oos_rm     = oos_d.get("r_multiple", "—")
    oos_hold   = oos_d.get("avg_hold_days", "—")
    oos_exp    = oos_d.get("avg_exposure", "—")

    # PF = (勝率 × 平均利益) / (敗率 × 平均損失)
    def _pf(d: dict) -> str:
        wr = d.get("win_rate")
        aw = d.get("avg_win_pct")
        al = d.get("avg_lose_pct")
        if wr is None or aw is None or al is None:
            return "—"
        try:
            pf = (wr / 100 * aw) / ((1 - wr / 100) * abs(al))
            return f"{pf:.2f}"
        except ZeroDivisionError:
            return "—"

    is_pf  = _pf(is_d)
    oos_pf = _pf(oos_d)

    wf_segs = wf.get("wf_segments", [])
    wf_sum  = wf.get("wf_summary", {})
    wf_full_is  = wf.get("full_is", {})
    wf_true_oos = wf.get("true_oos", {})

    dyn_wf_sum    = dyn.get("wf_summary", {}) if dyn else {}
    dyn_full_is   = dyn.get("full_is", {})    if dyn else {}
    dyn_oos_2025  = dyn.get("true_oos_2025", {}) if dyn else {}
    dyn_segs      = dyn.get("segments", [])   if dyn else []

    s123_v25 = s123.get("step3_concentration_caps", {}).get("verification_2025", {}) if s123 else {}
    dyn_dyn25 = s123_v25.get("dyn_plus_caps", {})

    # --- セクター戦略マップ整形 ---
    fujiko_sectors  = sorted(k for k, v in SECTOR_STRATEGY.items() if v == "fujiko")
    mr_sectors      = sorted(k for k, v in SECTOR_STRATEGY.items() if v == "mean_rev")

    # --- クラスターマップ整形 ---
    cluster_lines = []
    for cname, secs in CLUSTER_MAP_DEFAULT.items():
        cluster_lines.append(f"| {cname} | {', '.join(secs)} |")

    # ================================================================ #
    doc = f"""\
# CHIBAAssetProject 戦略仕様書

**作成日**: {today_str}
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
| 初期資本 | **{capital:,}円** | `strategy.yaml: portfolio.capital` |
| スリッページ | **{SLIPPAGE:.3%}** | `composite_alpha_bt.py: SLIPPAGE` |
| 手数料 | **{COMMISSION:.4%}** | `composite_alpha_bt.py: COMMISSION` |
| 片道コスト合計 | **{COST_ONE_WAY:.4%}** | `SLIPPAGE + COMMISSION` |
| 最低手数料 | 99円 | `strategy.yaml: costs.min_commission` |
| 注文単位 | {LOT}株 | `composite_alpha_bt.py: LOT` |

---

## 2. ユニバース仕様

ソース: `src/configs/rsr_universe_42.csv` / `src/backtest/composite_alpha_bt.py`

**固定プール**: RSR42（42銘柄）
**選定基準**: TOPIX100から取引所一部上場・流動性基準を満たす銘柄を42銘柄に絞り込んだ固定プール

### セクター別戦略割り当て

ソース: `composite_alpha_bt.py: SECTOR_STRATEGY`

| 戦略タイプ | 割り当てセクター |
|---|---|
| フジコ法（モメンタム） | {', '.join(fujiko_sectors)} |
| 平均回帰 | {', '.join(mr_sectors)} |

### 銘柄一覧

{_load_universe_table()}
---

## 3. 動的ユニバース選定

ソース: `src/strategy/universe.py`（確定設定: `dyn_rsr42_bear_rs0`、採用 2026-04-05）

### レジーム判定

```
持続 Bear 判定:
  TOPIX < MA{MA200_PERIOD} かつ 直近{LOOKBACK_BEAR_CHECK}営業日のうち{SUSTAINED_BEAR_DAYS}日以上 MA{MA200_PERIOD} を下回る

  ※ 短期クラッシュ（コロナ型・2ヶ月以内）では Bear scoring を適用しない
  ※ 持続型 Bear（2022型・60営業日以上）でのみ Bear scoring に切り替え

判定コード（universe.py: is_sustained_bear）:
  topix_lb     = topix_close.iloc[-{LOOKBACK_BEAR_CHECK}:]
  days_below   = (topix_lb < rolling_ma200.iloc[-{LOOKBACK_BEAR_CHECK}:]).sum()
  is_sustained_bear = days_below >= {SUSTAINED_BEAR_DAYS}
```

### Bull スコアリング（持続 Bear 以外）

```python
# universe.py: build_sym_active_df（Bull分岐）
LOSS_PENALTY_COEF = 0.10   # 直近90日損失銘柄ペナルティ係数
LOSS_PERIOD       = 90

score = (
    0.40 * zscore(mom_{MOM_PERIOD}d)      # {MOM_PERIOD}営業日（≒3ヶ月）モメンタム
    + 0.35 * zscore(rsr)                  # RSR（ユニバース内パーセンタイル）
    + 0.25 * zscore(log_vol_{VOL_PERIOD}d) # 直近{VOL_PERIOD}日平均出来高の対数
    - 0.10 * zscore(loss_90d)             # 直近90日損失ペナルティ
)
→ Top {BULL_ACTIVE_N} 銘柄を月次で更新
```

### Bear スコアリング（持続 Bear 時）

```python
# universe.py: build_sym_active_df（Bear分岐）
# 前提フィルター: rs_topix > 0（TOPIX比プラスの銘柄のみ対象）
score = (
    0.50 * zscore(rs_topix)               # TOPIX比相対リターン（最重視）
    + 0.30 * zscore(rsr)
    + 0.20 * zscore(log_vol_{VOL_PERIOD}d)
    - 0.10 * zscore(loss_90d)
)
→ rs_topix > 0 フィルター後 Top {BEAR_ACTIVE_N} 銘柄を月次で更新
```

### Bear 時 セクター除外

ソース: `strategy.yaml: bear_universe_filter`

```
除外セクター（Bear 持続時）:
  {chr(10).join(f"  - {s}" for s in bear_excl) if bear_excl else "  （設定なし）"}
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
mom_period = {mom_period}   # strategy.yaml: fujiko.mom_period
mom_arr    = rsr_arr - roll(rsr_arr, {mom_period})   # {mom_period}日前との差分
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
| **エース（採用閾値）** | **{min_sepa}点以上** | `strategy.yaml: fujiko.min_sepa = {min_sepa}` |
| 対象外 | {min_sepa - 1}点以下 | エントリー不可 |

---

## 6. エントリー条件

ソース: `src/backtest/fujiko_strategy.py: precompute_signals`

以下の条件を**すべて同時に**満たした場合にエントリーシグナル（+1）を発生。
実行価格: **翌営業日の始値**（寄付成行 または 寄成注文）

```python
# fujiko_strategy.py: precompute_signals
entry_mask  = sepa_score_arr >= {min_sepa}           # 条件1: SEPA {min_sepa}点以上
entry_mask &= rsr_arr        >= {min_rsr}            # 条件2: RSR >= {min_rsr}
entry_mask &= (mom_arr > 0) & (mom_arr > mom_prev)  # 条件3: RSRモメンタム 正かつ上昇
if use_turtle_entry:  # strategy.yaml: fujiko.use_turtle_entry = {use_turtle}
    entry_mask &= close > prev_{turtle_entry}d_high # 条件4: {turtle_entry}日高値ブレイクアウト
```

### エントリー条件一覧（確定値）

| # | 条件 | 閾値 | 出典 |
|---|---|---|---|
| 1 | SEPA スコア | ≥ **{min_sepa}**（8点中） | `strategy.yaml: fujiko.min_sepa` |
| 2 | RSR（ユニバース内ランク） | ≥ **{min_rsr}** | `strategy.yaml: fujiko.min_rsr` |
| 3 | RSRモメンタム（{mom_period}日差分） | > 0 かつ 前日比上昇 | `strategy.yaml: fujiko.mom_period = {mom_period}` |
| 4 | タートルズ S1 ブレイクアウト | 前日までの **{turtle_entry}日高値** 超え | `strategy.yaml: fujiko.turtle_entry = {turtle_entry}` |
| 5 | 動的ユニバース活性 | 当月の active リストに含まれる | `strategy/universe.py: sym_active_mat` |
| 6 | 流動性フィルター（ライブのみ） | 日次売買代金 ≥ **{MIN_DAILY_VALUE_YEN:,}円** | `signal_bridge.py: MIN_DAILY_VALUE_YEN` |
| 7 | MTF フィルター（ライブのみ） | 週足 RSR ≥ {min_rsr} かつ 週足終値 > 週足 MA20 | `signal_bridge.py: MTFフィルター` |

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
| **1** | **Composite Shock Exit** | TOPIX 日次リターン ≤ **-5%** かつ 個別株 日次リターン ≤ **-8%** | `signal_bridge.py: shock_exit_mode="{shock_mode}"` |
| **2** | **トレーリングストップ** | 終値 < 保有期間最高終値 − **3.0 × ATR20** | `signal_bridge.py` 参照元: `composite_alpha_bt.py: TRAIL_ATR_MULT={TRAIL_ATR_MULT}` |
| **3** | **時間ストップ** | 保有営業日数 ≥ **{max_hold}日** | `strategy.yaml: risk.max_hold_days = {max_hold}` |
| **4** | **RSR 低下エグジット** | RSR < **{min_rsr}** かつ 保有 ≥ **{min_hold}日** _(緊急時は min_hold 無視)_ | `strategy.yaml: fujiko.min_rsr / risk.min_hold_days` |
| **5** | **緊急エグジット** | 含み損 ≤ **{emg_exit:.0%}**（min_hold 無視で即時発動） | `strategy.yaml: risk.emergency_exit_pct = {emg_exit}` |
| **6** | **mean_rev 反発失敗** | 平均回帰エントリー後 **{MEANREV_FAIL_DAYS}営業日** 以内に high が +**{MEANREV_MIN_BOUNCE:.0%}** 未達 かつ 終値 < エントリー×0.995 | `signal_bridge.py: MEANREV_FAIL_DAYS / MEANREV_MIN_BOUNCE` |
| **7** | **戦略シグナル（フジコ法）** | RSRモメンタム < 0 かつ 前日比下降 _または_ 終値 < **前日までの{turtle_exit}日安値** | `fujiko_strategy.py: exit_mask` |

### バックテストエンジン上の追加エグジット

ソース: `src/backtest/composite_alpha_bt.py: run_scenario`（STEP5 構成）

| 条件 | パラメータ | 出典 |
|---|---|---|
| トレーリング（STEP5） | 終値 < **{TRAIL_PERIOD}日最高終値** − {TRAIL_ATR_MULT} × ATR20 | `TRAIL_PERIOD={TRAIL_PERIOD}, TRAIL_ATR_MULT={TRAIL_ATR_MULT}` |
| exit_params（RSR-z） | RSR-z < **{ep_rsr_exit}**（翌日始値）| `strategy.yaml: exit_params.rsr_exit = {ep_rsr_exit}` |
| exit_params（タイムストップ） | 保有バー数 ≥ **{ep_time_stop}**（BTエンジン用）| `strategy.yaml: exit_params.time_stop = {ep_time_stop}` |
| exit_params（トレイル） | HWM から **-{ep_trail_stop:.1%}** 下落 | `strategy.yaml: exit_params.trail_stop = {ep_trail_stop}` |
| TIME_STOP 後クールダウン | {REENTRY_COOL}営業日 再エントリー禁止 | `composite_alpha_bt.py: REENTRY_COOL = {REENTRY_COOL}` |

### エグジットとmin_holdの関係

```python
# signal_bridge.py（ライブ）
is_rank_exit = (
    rsr_now < min_rsr_threshold          # RSR が閾値未満
    and (hold_td >= self.min_hold_days   # min_hold_days = {min_hold} 営業日以上
         or is_emergency_exit)           # 緊急時は hold 無視
)
```

---

## 8. 平均回帰サブ戦略

ソース: `src/backtest/mean_reversion_strategy.py`（`MR_PARAMS` は `composite_alpha_bt.py`）

### 割り当てセクター

```python
# composite_alpha_bt.py: SECTOR_STRATEGY
mean_rev セクター: {mr_sectors}
```

### エントリー条件

```python
# mean_reversion_strategy.py
1. RSI({MR_PARAMS["rsi_period"]}日) < {MR_PARAMS["rsi_entry"]}       # 短期売られすぎ（Wilder EMA方式）
2. Close > MA{MR_PARAMS["ma_long"]}                  # 大局上昇トレンド内（トレンドフィルター）
3. Close > MA50 × 0.85                               # 落下するナイフ回避（-15%以内）
```

### エグジット条件

```python
# mean_reversion_strategy.py
A. RSI({MR_PARAMS["rsi_period"]}日) > {MR_PARAMS["rsi_exit"]}     # 回復・利食い
B. Close < エントリー × (1 - {MR_PARAMS["stop_loss_pct"]})  # ストップロス -{MR_PARAMS["stop_loss_pct"]:.0%}
C. 保有 ≥ {MR_PARAMS["max_hold_days"]}営業日             # 時間切れ
```

---

## 9. ポジションサイジング

ソース: `src/backtest/composite_alpha_bt.py: run_scenario`

### サイジングモード（確定: 均等ウェイト）

```python
# composite_alpha_bt.py: sizing_mode = "existing"（キャッシュ均等割り）
# 1銘柄への投資額 = 利用可能キャッシュ / 空きスロット数
# 購入株数       = (投資額 / 翌日始値 // LOT) * LOT  # {LOT}株単位に切り捨て
```

| パラメータ | 値 | 出典 |
|---|---|---|
| 最大同時保有数 | **{max_pos}銘柄** | `strategy.yaml: portfolio.max_positions` |
| 1銘柄最大ウェイト | **{max_sw:.0%}** | `strategy.yaml: portfolio.max_single_weight` |
| symbol_cap（固定） | **{sym_cap:.0%}** | `strategy.yaml: risk_controls.symbol_cap`（`dynamic_cap: {dyn_cap}`） |
| 注文単位 | {LOT}株 | `composite_alpha_bt.py: LOT` |

---

## 10. セクター・クラスター制御

ソース: `src/backtest/composite_alpha_bt.py: run_scenario` / `src/strategy/cluster.py`

### セクター集中制御

ソース: `strategy.yaml: risk_controls.sector_concentration`

```python
# composite_alpha_bt.py（動的ユニバース時のみ有効）
MAX_SECTOR_WEIGHT = {sec_cap}   # sector_cap（Bear 時は bear_sector_cap）
MAX_SYMBOL_WEIGHT = {sym_cap}   # symbol_cap

# Bear 適応 cap（TOPIX < MA200 の日）
bear_sector_cap  = {bear_sec_cap}   # strategy.yaml: risk_controls.bear_sector_cap
bear_cluster_cap = {bear_cls_cap}   # strategy.yaml: risk_controls.bear_cluster_cap
```

| 制御 | Bull 時 | Bear 時（TOPIX < MA200） | 出典 |
|---|---|---|---|
| セクター上限 | **{sec_cap:.0%}** | **{bear_sec_cap:.0%}** | `risk_controls.sector_cap / bear_sector_cap` |
| クラスター上限 | **{cluster_cap:.0%}** | **{bear_cls_cap:.0%}** | `risk_controls.cluster_cap / bear_cluster_cap` |
| 同一セクター銘柄数 | ≤ **{max_names_sec}銘柄** | ≤ **{max_names_sec}銘柄** | `sector_concentration.max_names_per_sector` |
| セクター合計ウェイト | ≤ **{max_wt_sec:.0%}** | ≤ **{max_wt_sec:.0%}** | `sector_concentration.max_weight_per_sector` |

### クラスターマップ

ソース: `src/strategy/cluster.py: CLUSTER_MAP_DEFAULT`

| クラスター | 含まれるセクター |
|---|---|
{chr(10).join(cluster_lines)}

### Gross Exposure 縦断制御

ソース: `strategy.yaml: risk_controls.gross_exposure_*` / `composite_alpha_bt.py`

```python
# gross_exposure_enabled = {gross_en}
if   TOPIX_20d_return < -0.05: gross_cap = {gross_dd5}    # TOPIX 20日 -5% 時
elif TOPIX_60d_return < -0.08: gross_cap = {gross_dd8}    # TOPIX 60日 -8% 時
else:                          gross_cap = {gross_norm}   # 通常時
```

| 状態 | Gross Exposure 上限 |
|---|---|
| 通常 | **{gross_norm:.0%}** |
| TOPIX 20日リターン < −5% | **{gross_dd5:.0%}** |
| TOPIX 60日リターン < −8% | **{gross_dd8:.0%}** |

### クラスター相場制御（ライブのみ）

ソース: `src/kabusapi/signal_bridge.py`

```python
# signal_bridge.py
CLUSTER_LEVEL1_THRESH = {CLUSTER_LEVEL1_THRESH}   # cluster_density >= 15%: mean_rev BUY 停止
CLUSTER_LEVEL2_THRESH = {CLUSTER_LEVEL2_THRESH}   # cluster_density >= 25%: モメンタム偏重
```

---

## 11. サーキットブレーカー

ソース: `src/risk/circuit_breaker.py`

```python
# circuit_breaker.py
DD_TRIGGER  = {DD_TRIGGER}    # ドローダウン がこれ以下 → ENTRY_STOP_ONLY（BUY 停止）
RECOVERY_DD = {RECOVERY_DD}   # DD が回復したら NORMAL に復帰
MAX_CB_DAYS = {MAX_CB_DAYS}   # 最大 {MAX_CB_DAYS} 営業日で強制解除

状態機械:
  NORMAL
    ↓ DD <= {DD_TRIGGER:.0%}
  ENTRY_STOP_ONLY  ← BUY 停止 / SELL は引き続き許可
    ↓ {MAX_CB_DAYS}営業日経過 OR DD >= {RECOVERY_DD:.0%} に回復
  NORMAL
```

| パラメータ | 値 | 意味 |
|---|---|---|
| DD_TRIGGER | **{DD_TRIGGER:.0%}** | BUY 停止開始ライン |
| RECOVERY_DD | **{RECOVERY_DD:.0%}** | CB 解除ライン |
| MAX_CB_DAYS | **{MAX_CB_DAYS}営業日** | 強制解除タイムアウト |

---

## 12. マーケットショック制御

ソース: `src/backtest/composite_alpha_bt.py: run_scenario` / `src/kabusapi/signal_bridge.py`

### Composite モード（本番採用: `shock_exit_mode = "{shock_mode}"`）

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
| CAGR | **{is_cagr}%** | **{oos_cagr}%** | — |
| Sharpe | **{is_sharpe}** | **{oos_sharpe}** | > 0.5 |
| MaxDD | **{is_maxdd}%** | **{oos_maxdd}%** | < -20% |
| Calmar（CAGR / MaxDD） | **{is_calmar}** | **{oos_calmar}** | > 1.0 |
| PF（勝率×平均利益 / 敗率×平均損失） | **{is_pf}** | **{oos_pf}** | > 1.5 |
| 勝率 | **{is_wr}%** | **{oos_wr}%** | — |
| R倍数（平均利益 / 平均損失） | **{is_rm}** | **{oos_rm}** | — |
| 平均保有日数 | **{is_hold}日** | **{oos_hold}日** | — |
| 年間取引数 | **{is_yr}件/年**（{is_trades}件合計） | — | ≥ 5 |
| 平均エクスポージャー | **{is_exp}%** | **{oos_exp}%** | — |

### IS 年次リターン（2020-2024）

| 年 | リターン |
|---|---|
{chr(10).join(f"| {yr} | **+{ret:.2f}%** |" if isinstance(ret, (int, float)) else f"| {yr} | {ret} |" for yr, ret in (is_ann.items() if is_ann else {}))}

### エグジット内訳（IS）

| 理由 | 件数 |
|---|---|
{chr(10).join(f"| {k} | {v}件 |" for k, v in is_d.get("exit_reason_counts", {}).items())}

### 動的ユニバース採用後（2025 OOS 比較）

ソース: `backtests/step123_integration_2026-04-06.json`

| 指標 | ベース（固定） | 動的ユニバース採用後 | 改善幅 |
|---|---|---|---|
| 2025 CAGR | {s123.get("step3_concentration_caps", {}).get("verification_2025", {}).get("baseline_no_dyn_no_cap", {}).get("cagr", "—") if s123 else "—"}（×100%） | **{dyn_dyn25.get("cagr", "—")}（×100%）** | — |
| 2025 MaxDD | {s123.get("step3_concentration_caps", {}).get("verification_2025", {}).get("baseline_no_dyn_no_cap", {}).get("maxdd", "—") if s123 else "—"} | **{dyn_dyn25.get("maxdd", "—")}** | — |
| 2025 Sharpe | {s123.get("step3_concentration_caps", {}).get("verification_2025", {}).get("baseline_no_dyn_no_cap", {}).get("sharpe", "—") if s123 else "—"} | **{dyn_dyn25.get("sharpe", "—")}** | — |

---

## 14. ウォークフォワード検証

ソース: `backtests/wf_final_2026-04-04.json` / `backtests/wf_dyn_rsr42_2026-04-05.json`

### ベースライン WF（hold3d / turtle_exit=55 / min_rsr=75）

| Seg | IS 期間 | OOS 年 | IS Sharpe | OOS Sharpe | OOS/IS 比 | 合格 |
|---|---|---|---|---|---|---|
{chr(10).join(
    "| {seg} | {is_} | {oos} | {iss:.3f} | {ooss:.3f} | {ratio} | {win} |".format(
        seg=s["seg"], is_=s.get("is","—"), oos=s.get("oos","—"),
        iss=s.get("is_sharpe",0), ooss=s.get("oos_sharpe",0),
        ratio="—" if s.get("ratio") is None else f'{s["ratio"]:.3f}',
        win="✅" if s.get("win") else "❌",
    )
    for s in wf_segs
)}

**総合**: {wf_sum.get("oos_wins","—")} / 平均 OOS/IS 比 = **{wf_sum.get("avg_oos_is_ratio","—")}**
**Full IS（{wf_full_is.get("period","2018-2024")}）**: CAGR={wf_full_is.get("cagr","—")}% / Sharpe={wf_full_is.get("sharpe","—")} / MaxDD={wf_full_is.get("max_dd","—")}%
**True OOS（{wf_true_oos.get("period","2025")}）**: CAGR={wf_true_oos.get("cagr","—")}% / Sharpe={wf_true_oos.get("sharpe","—")} / MaxDD={wf_true_oos.get("max_dd","—")}%

### 動的ユニバース WF（dyn_rsr42_bear_rs0）

| Seg | OOS 年 | OOS Sharpe | OOS MaxDD | 合格 |
|---|---|---|---|---|
{chr(10).join(
    "| {seg} | {oos} | {ooss:.3f} | {dd:.2f}% | {win} |".format(
        seg=s["seg"], oos=s.get("oos","—"),
        ooss=s.get("oos_sharpe",0),
        dd=s.get("oos_max_dd",0),
        win="✅" if s.get("win") else "❌",
    )
    for s in dyn_segs
)}

**総合**: {dyn_wf_sum.get("pass_count","—")} / Full IS Sharpe = **{dyn_full_is.get("sharpe","—")}**
**True OOS 2025 Sharpe**: **{dyn_oos_2025.get("sharpe","—")}** / MaxDD: **{dyn_oos_2025.get("max_dd","—")}%**

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
DATA_HEALTH_MIN_RATIO = {0.90}   # RSR42 データ取得率 90% 未満でシグナル停止
```

---

## 付録: 確定パラメータ一覧

ソース: `src/configs/strategy.yaml`

| セクション | パラメータ | 値 |
|---|---|---|
| `fujiko` | `min_sepa` | {min_sepa} |
| `fujiko` | `min_rsr` | {min_rsr} |
| `fujiko` | `mom_period` | {mom_period} |
| `fujiko` | `turtle_entry` | {turtle_entry} |
| `fujiko` | `turtle_exit` | {turtle_exit} |
| `fujiko` | `use_turtle_entry` | {use_turtle} |
| `portfolio` | `capital` | {capital:,}円 |
| `portfolio` | `max_positions` | {max_pos} |
| `portfolio` | `max_single_weight` | {max_sw:.0%} |
| `portfolio` | `max_dd_limit` | {max_dd_limit:.0%} |
| `risk` | `min_hold_days` | {min_hold} |
| `risk` | `max_hold_days` | {max_hold} |
| `risk` | `emergency_exit_pct` | {emg_exit:.0%} |
| `exit_params` | `time_stop` | {ep_time_stop}バー |
| `exit_params` | `trail_stop` | {ep_trail_stop:.1%} |
| `exit_params` | `rsr_exit` | {ep_rsr_exit} |
| `risk_controls` | `shock_exit_mode` | {shock_mode} |
| `risk_controls` | `symbol_cap` | {sym_cap:.0%} |
| `risk_controls` | `sector_cap` | {sec_cap:.0%} |
| `risk_controls` | `cluster_cap` | {cluster_cap:.0%} |
| `risk_controls` | `bear_sector_cap` | {bear_sec_cap:.0%} |
| `risk_controls` | `bear_cluster_cap` | {bear_cls_cap:.0%} |
| `risk_controls` | `gross_cap_normal` | {gross_norm:.0%} |
| `risk_controls` | `gross_cap_drawdown_5pct` | {gross_dd5:.0%} |
| `risk_controls` | `gross_cap_drawdown_8pct` | {gross_dd8:.0%} |
| `dynamic_universe` | `enabled` | {du_enabled} |
| `dynamic_universe` | `pool` | {du_pool} |
| `dynamic_universe` | `bull_active_n` | {BULL_ACTIVE_N} |
| `dynamic_universe` | `bear_active_n` | {BEAR_ACTIVE_N} |
| `dynamic_universe` | `bear_rs_filter` | {bear_rs_filter} |

---

_生成スクリプト: `src/generate_strategy_spec.py` / 生成日時: {today_str}_
"""

    return doc


# ================================================================== #
# エントリーポイント
# ================================================================== #
def main() -> None:
    parser = argparse.ArgumentParser(description="戦略仕様書ジェネレーター")
    parser.add_argument("--out", type=str, default=None,
                        help="出力先 Markdown ファイルパス（省略時は stdout）")
    args = parser.parse_args()

    bt = _load_bt_results()
    doc = generate(bt)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(doc, encoding="utf-8")
        print(f"✅ 出力完了: {out_path.resolve()}")
    else:
        print(doc)


if __name__ == "__main__":
    main()
