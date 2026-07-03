"""
backtest/market_shock_comparison.py
市場ショックExit方式の3モード比較（IS + OOS 2025）

比較モード:
  full_exit   : 現行（1306.T -5%以下で翌寄り全量決済）
  partial_50  : 翌寄り50%部分決済、残りは通常ルール継続
  composite   : 市場ショック + 個別銘柄 -8%以下の複合条件でのみ全量決済

検証軸:
  IS  : 2018-01-01 〜 2024-12-31
  OOS : 2025-01-01 〜 2025-12-31
  OOS April : 2025-04-01 〜 2025-04-30（関税ショック集中月）

採用条件:
  IS Sharpe 劣化 < 0.1
  OOS April損失 >= 10%削減
  MaxDD 改善

実行:
  cd C:/ai-trading
  python -m backtest.market_shock_comparison
"""

from __future__ import annotations
import gc
import os, sys, json, warnings, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import backtest.composite_alpha_bt as _bt
from backtest.rsr import calc_universe_rsr
from backtest.universe_builder import download_universe
from src.config_loader import load_strategy_config
from src.utils.memory import collect_and_log_process_memory

# OOS 2025 はスナップショット外のため yfinance を使用
os.environ.pop("DATA_VERSION", None)

# ------------------------------------------------------------------ #
# 設定
# ------------------------------------------------------------------ #
IS_START    = "2018-01-01"
IS_END      = "2024-12-31"
OOS_START   = "2025-01-01"
OOS_END     = "2025-12-31"
APRIL_START = "2025-04-01"
APRIL_END   = "2025-04-30"
FULL_END    = OOS_END

MODES = ["full_exit", "partial_50", "composite"]

OUTPUT_DIR  = "C:/ai-trading/backtests"
OUTPUT_JSON = os.path.join(OUTPUT_DIR, f"market_shock_comparison_{time.strftime('%Y-%m-%d')}.json")


# ------------------------------------------------------------------ #
# データロード（1回のみ）
# ------------------------------------------------------------------ #
def load_data():
    cfg = load_strategy_config()
    print("[1/3] ユニバース・価格データ読み込み中（RSR42, 2018-2025）...")
    rsr_syms   = _bt._load_rsr_universe()
    trade_syms = rsr_syms

    all_syms = {**rsr_syms, **trade_syms}
    universe_raw = download_universe(all_syms, start=IS_START, end=FULL_END, verbose=False)
    topix_close  = _bt._download_topix(IS_START, FULL_END)

    print("[2/3] RSR計算中...")
    rsr42_prices = {
        sym: universe_raw[sym]["df"]["Close"]
        for sym in rsr_syms if sym in universe_raw
    }
    rsr_df = calc_universe_rsr(rsr42_prices)

    print("[3/3] Composite Alpha計算中...")
    trade_prices = {
        sym: universe_raw[sym]["df"]["Close"]
        for sym in trade_syms if sym in universe_raw
    }
    alpha_df = _bt.calc_composite_alpha_matrix(trade_prices, window=_bt.COMP_ALPHA_WINDOW)
    alpha_df = alpha_df.shift(1)

    regime_df = _bt._calc_regime(topix_close)
    tech_matrices = _bt._precompute_tech_matrices(universe_raw, list(trade_syms.keys()))

    min_hold    = cfg.risk.min_hold_days
    turtle_exit = cfg.fujiko.turtle_exit
    min_rsr     = cfg.fujiko.min_rsr
    capital     = cfg.portfolio.capital
    print(f"  RSR: {rsr_df.shape[1]}銘柄 × {rsr_df.shape[0]}日")
    print(f"  min_hold={min_hold}日 / turtle_exit={turtle_exit}日 / min_rsr={min_rsr} / capital={capital:,}\n")

    return universe_raw, rsr_df, alpha_df, regime_df, trade_syms, rsr_syms, tech_matrices, topix_close, cfg


# ------------------------------------------------------------------ #
# 1モード実行
# ------------------------------------------------------------------ #
def run_one(
    universe_raw, rsr_df, alpha_df, regime_df, trade_syms, rsr_syms,
    tech_matrices, topix_close, cfg,
    mode: str, period_start: str, period_end: str
) -> dict:
    result = _bt.run_scenario(
        scenario          = "BASELINE",
        universe_raw      = universe_raw,
        rsr_df            = rsr_df,
        alpha_df          = alpha_df,
        regime_df         = regime_df,
        trade_syms        = trade_syms,
        rsr_syms          = rsr_syms,
        cfg               = cfg,
        start             = period_start,
        end               = period_end,
        capital           = float(cfg.portfolio.capital),
        verbose           = False,
        tech_matrices     = tech_matrices,
        topix_close       = topix_close,
        min_hold          = int(cfg.risk.min_hold_days),
        market_shock_mode = mode,
        enable_volume_filter     = True,
        enable_volatility_filter = False,
        enable_atr_filter        = False,
        enable_market_filter     = False,
    )
    return result


def april_pnl_from_equity(result: dict) -> float:
    """エクイティカーブから4月PnLを計算（円）"""
    eq = result.get("equity_curve")
    if eq is None or eq.empty:
        return float("nan")
    april = eq.loc[APRIL_START:APRIL_END]
    if len(april) < 2:
        return float("nan")
    return float(april.iloc[-1] - april.iloc[0])


def monthly_returns_oos(result: dict) -> dict[str, float]:
    """OOS期間の月次リターン（%）"""
    eq = result.get("equity_curve")
    if eq is None or eq.empty:
        return {}
    monthly = {}
    for (yr, mo), grp in eq.groupby([eq.index.year, eq.index.month]):
        if len(grp) < 2:
            continue
        ret = float(grp.iloc[-1] / grp.iloc[0] - 1) * 100
        monthly[f"{yr}-{mo:02d}"] = round(ret, 2)
    return monthly


# ------------------------------------------------------------------ #
# サマリー表示
# ------------------------------------------------------------------ #
def fmt_row(label: str, r: dict, april_pnl: float | None = None) -> str:
    if not r:
        return f"  {label:12s}  データなし"
    sharpe = r.get("sharpe", 0)
    cagr   = r.get("cagr", 0)
    mdd    = r.get("max_dd", 0)
    n_shock = r.get("exit_reason_counts", {}).get("MARKET_SHOCK_EXIT", 0)
    n_partial = r.get("exit_reason_counts", {}).get("MARKET_SHOCK_PARTIAL", 0)
    shock_str = f"  shock_exit={n_shock+n_partial}"
    april_str = f"  Apr={april_pnl:+,.0f}円" if april_pnl is not None and not np.isnan(april_pnl) else ""
    return (f"  {label:12s}  Sharpe={sharpe:6.3f}  CAGR={cagr:+6.1f}%  "
            f"MaxDD={mdd:+6.1f}%{shock_str}{april_str}")


# ------------------------------------------------------------------ #
# メイン
# ------------------------------------------------------------------ #
def main():
    print("=" * 76)
    print("  市場ショックExit 3モード比較（IS 2018-2024 / OOS 2025）")
    print("  モード: full_exit（現行）/ partial_50（50%部分）/ composite（複合条件）")
    print("=" * 76 + "\n")

    (universe_raw, rsr_df, alpha_df, regime_df,
     trade_syms, rsr_syms, tech_matrices, topix_close, cfg) = load_data()

    results = {}

    for mode in MODES:
        marker = " ← 現行" if mode == "full_exit" else ""
        print(f"── {mode}{marker} ──")

        r_is = run_one(universe_raw, rsr_df, alpha_df, regime_df, trade_syms, rsr_syms,
                       tech_matrices, topix_close, cfg, mode, IS_START, IS_END)
        print(fmt_row("IS (2018-24)", r_is))

        r_oos = run_one(universe_raw, rsr_df, alpha_df, regime_df, trade_syms, rsr_syms,
                        tech_matrices, topix_close, cfg, mode, OOS_START, OOS_END)
        april_pnl = april_pnl_from_equity(r_oos)
        monthly   = monthly_returns_oos(r_oos)
        print(fmt_row("OOS (2025)", r_oos, april_pnl))

        # 月次勝敗
        wins  = sum(1 for v in monthly.values() if v > 0)
        loses = sum(1 for v in monthly.values() if v <= 0)
        print(f"    月次: {wins}勝{loses}敗  {monthly}")
        print()

        results[mode] = {
            "mode":         mode,
            "IS":           {k: v for k, v in r_is.items() if k not in ("equity_curve", "_trades")},
            "OOS":          {k: v for k, v in r_oos.items() if k not in ("equity_curve", "_trades")},
            "april_pnl_jpy": april_pnl if not np.isnan(april_pnl) else None,
            "monthly_returns": monthly,
        }

        del r_is, r_oos
        collect_and_log_process_memory(f"mode {mode}")

    # ---------------------------------------------------------------- #
    # 比較サマリー
    # ---------------------------------------------------------------- #
    baseline_is_sharpe = results["full_exit"]["IS"].get("sharpe", 0)
    baseline_apr       = results["full_exit"].get("april_pnl_jpy", 0) or 0
    baseline_mdd_oos   = results["full_exit"]["OOS"].get("max_dd", 0)

    print("=" * 76)
    print("  比較サマリー")
    print(f"  {'モード':14s}  {'IS Sharpe':>10}  {'IS MaxDD':>9}  {'OOS Sharpe':>11}  "
          f"{'OOS MaxDD':>10}  {'4月PnL(円)':>12}  {'判定':>6}")
    print("  " + "-" * 74)
    for mode, v in results.items():
        ri = v["IS"]
        ro = v["OOS"]
        apr = v.get("april_pnl_jpy") or 0
        sharpe_diff = ri.get("sharpe", 0) - baseline_is_sharpe
        mdd_diff    = ro.get("max_dd", 0) - baseline_mdd_oos
        apr_improv  = apr - baseline_apr   # 正 = 改善（損失が減った）
        ok_sharpe   = sharpe_diff >= -0.1
        ok_mdd      = mdd_diff >= 0 or mode == "full_exit"
        marker = "現行" if mode == "full_exit" else ("✅" if ok_sharpe else "❌")
        print(f"  {mode:14s}  {ri.get('sharpe',0):>10.3f}  {ri.get('max_dd',0):>+9.1f}%  "
              f"{ro.get('sharpe',0):>11.3f}  {ro.get('max_dd',0):>+10.1f}%  "
              f"{apr:>+12,.0f}  {marker}")
        if mode != "full_exit":
            print(f"  {'  vs 現行':14s}  {sharpe_diff:>+10.3f}  {'':9s}  "
                  f"{'':11s}  {mdd_diff:>+10.1f}pp  {apr_improv:>+12,.0f}")
    print("=" * 76)

    # ---------------------------------------------------------------- #
    # 採用判定サマリー
    # ---------------------------------------------------------------- #
    print("\n  【採用判定】")
    print("  採用条件: IS Sharpe 劣化 < 0.1 / 4月PnL 改善 / OOS MaxDD 改善")
    for mode, v in results.items():
        if mode == "full_exit":
            continue
        ri = v["IS"]
        ro = v["OOS"]
        apr = v.get("april_pnl_jpy") or 0
        sharpe_diff = ri.get("sharpe", 0) - baseline_is_sharpe
        mdd_diff    = ro.get("max_dd", 0) - baseline_mdd_oos
        apr_improv  = apr - baseline_apr
        ok_sharpe = sharpe_diff >= -0.1
        ok_mdd    = mdd_diff >= 0
        ok_apr    = apr_improv > 0
        verdict = "✅ 採用候補" if (ok_sharpe and ok_mdd and ok_apr) else "❌ 不採用"
        print(f"  {mode:12s}: {verdict}"
              f"  (IS_Sharpe_diff={sharpe_diff:+.3f}, OOS_MaxDD_diff={mdd_diff:+.1f}pp, 4月PnL_diff={apr_improv:+,.0f}円)")

    # ---------------------------------------------------------------- #
    # JSON保存
    # ---------------------------------------------------------------- #
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  結果保存: {OUTPUT_JSON}")
    print("=" * 76)
    return 0


if __name__ == "__main__":
    sys.exit(main())
