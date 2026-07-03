"""
backtest/min_hold_sensitivity.py
最低保有日数（min_hold）感度テスト（IS + OOS 2025 同時検証）

目的:
  2025 OOS で RSRモメンタムエグジットが早期（3-5日）に発動する問題を
  「最低保有日数」で制御できるか検証する。

  min_hold=N のとき: エントリーからN日経過するまで RSRモメンタム売りを無視
  （タートルエグジット・緊急撤退は有効のまま）

検証軸:
  min_hold: 0（制約なし）, 3, 5, 7, 10

期間:
  IS  : 2018-01-01 〜 2024-12-31
  OOS : 2025-01-01 〜 2025-12-31

ユニバース: RSR42統一 / turtle_exit=55（2026-03-31確定値）
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
from src.utils.memory import collect_and_log_process_memory

# OOS 2025 はスナップショット外のため yfinance を使用
import os as _os
_os.environ.pop("DATA_VERSION", None)

# ------------------------------------------------------------------ #
# 設定
# ------------------------------------------------------------------ #
IS_START  = "2018-01-01"
IS_END    = "2024-12-31"
OOS_START = "2025-01-01"
OOS_END   = "2025-12-31"
FULL_END  = OOS_END

MIN_HOLD_VALUES = [0, 3, 5, 7, 10]
OUTPUT_DIR  = "C:/ai-trading/backtests"
OUTPUT_JSON = os.path.join(OUTPUT_DIR, f"min_hold_sensitivity_{time.strftime('%Y-%m-%d')}.json")


# ------------------------------------------------------------------ #
# データロード（1回だけ実施）
# ------------------------------------------------------------------ #
def load_data():
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

    print(f"  RSR: {rsr_df.shape[1]}銘柄 × {rsr_df.shape[0]}日")
    print(f"  取引対象: {len(trade_syms)}銘柄")
    print(f"  TURTLE_EXIT={_bt.TURTLE_EXIT}日（確定値）\n")

    return universe_raw, rsr_df, alpha_df, regime_df, trade_syms, rsr_syms


# ------------------------------------------------------------------ #
# 1シナリオ実行
# ------------------------------------------------------------------ #
def run_one(
    universe_raw, rsr_df, alpha_df, regime_df, trade_syms, rsr_syms,
    min_hold: int, period_start: str, period_end: str
) -> dict:
    result = _bt.run_scenario(
        scenario      = "BASELINE",
        universe_raw  = universe_raw,
        rsr_df        = rsr_df,
        alpha_df      = alpha_df,
        regime_df     = regime_df,
        trade_syms    = trade_syms,
        rsr_syms      = rsr_syms,
        start         = period_start,
        end           = period_end,
        capital       = _bt.CAPITAL,
        verbose       = False,
        min_hold      = int(min_hold),
    )
    return result


# ------------------------------------------------------------------ #
# サマリー表示
# ------------------------------------------------------------------ #
def fmt_row(label, r):
    if not r:
        return f"  {label:30s}  データなし"
    cagr   = r.get("cagr", 0)
    sharpe = r.get("sharpe", 0)
    mdd    = r.get("max_dd", 0)
    avgH   = r.get("avg_simultaneous_holdings", 0)
    nyr    = r.get("n_trades_yr", 0)
    avgHd  = r.get("avg_hold_days", 0)
    return (f"  {label:30s}  CAGR={cagr:+6.1f}%  Sharpe={sharpe:5.3f}"
            f"  MaxDD={mdd:6.1f}%  avgH={avgH:.2f}  tr/yr={nyr:.0f}  hold={avgHd:.0f}d")


# ------------------------------------------------------------------ #
# メイン
# ------------------------------------------------------------------ #
def main():
    print("=" * 72)
    print("  最低保有日数（min_hold）感度テスト（IS 2018-2024 / OOS 2025）")
    print("  ユニバース: RSR42統一（42銘柄）/ turtle_exit=55")
    print(f"  検証軸: min_hold = {MIN_HOLD_VALUES}")
    print("=" * 72 + "\n")

    universe_raw, rsr_df, alpha_df, regime_df, trade_syms, rsr_syms = load_data()

    results = {}

    for min_hold in MIN_HOLD_VALUES:
        key = f"hold{min_hold}d"
        marker = " ← 現行" if min_hold == 0 else ""
        print(f"── min_hold={min_hold}日{marker} ──")

        r_is = run_one(universe_raw, rsr_df, alpha_df, regime_df,
                       trade_syms, rsr_syms,
                       min_hold, IS_START, IS_END)
        print(fmt_row("IS (2018-2024)", r_is))

        r_oos = run_one(universe_raw, rsr_df, alpha_df, regime_df,
                        trade_syms, rsr_syms,
                        min_hold, OOS_START, OOS_END)

        n_oos_trades   = r_oos.get("n_trades", 0) if r_oos else 0
        trades_per_mon = round(n_oos_trades / 12, 2)
        print(fmt_row("OOS (2025)", r_oos) + f"  tpm={trades_per_mon}")
        print()

        results[key] = {
            "min_hold":  min_hold,
            "IS":        r_is,
            "OOS":       r_oos,
            "oos_trades_per_month": trades_per_mon,
        }
        del r_is, r_oos
        collect_and_log_process_memory(f"scenario {key}")

    # ---------------------------------------------------------------- #
    # サマリーテーブル
    # ---------------------------------------------------------------- #
    print("\n" + "=" * 72)
    print("  サマリー（OOS 2025）")
    print(f"  {'設定':10s}  {'CAGR':>8}  {'Sharpe':>7}  {'MaxDD':>7}  {'avgH':>5}  {'hold':>5}  {'tr/yr':>6}  {'tpm':>4}")
    print("  " + "-" * 68)
    for key, v in results.items():
        r = v["OOS"]
        marker = " ←現行" if v["min_hold"] == 0 else ""
        if not r:
            print(f"  {key:10s}  データなし")
            continue
        print(f"  {key:10s}  {r.get('cagr',0):>+7.1f}%  {r.get('sharpe',0):>7.3f}"
              f"  {r.get('max_dd',0):>+7.1f}%  {r.get('avg_simultaneous_holdings',0):>5.2f}"
              f"  {r.get('avg_hold_days',0):>5.0f}d  {r.get('n_trades_yr',0):>6.0f}"
              f"  {v['oos_trades_per_month']:>4.1f}{marker}")

    print("\n  サマリー（IS 2018-2024）")
    print(f"  {'設定':10s}  {'CAGR':>8}  {'Sharpe':>7}  {'MaxDD':>7}  {'avgH':>5}  {'hold':>5}  {'tr/yr':>6}")
    print("  " + "-" * 62)
    for key, v in results.items():
        r = v["IS"]
        marker = " ←現行" if v["min_hold"] == 0 else ""
        if not r:
            print(f"  {key:10s}  データなし")
            continue
        print(f"  {key:10s}  {r.get('cagr',0):>+7.1f}%  {r.get('sharpe',0):>7.3f}"
              f"  {r.get('max_dd',0):>+7.1f}%  {r.get('avg_simultaneous_holdings',0):>5.2f}"
              f"  {r.get('avg_hold_days',0):>5.0f}d  {r.get('n_trades_yr',0):>6.0f}{marker}")

    # ---------------------------------------------------------------- #
    # JSON保存
    # ---------------------------------------------------------------- #
    def strip_trades(d):
        if isinstance(d, dict):
            return {k: strip_trades(v) for k, v in d.items() if k not in ("_trades", "equity_curve")}
        return d

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(strip_trades(results), f, ensure_ascii=False, indent=2)
    print(f"\n  結果保存: {OUTPUT_JSON}")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    sys.exit(main())
