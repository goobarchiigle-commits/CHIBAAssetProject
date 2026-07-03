"""
backtest/step7_sensitivity.py
STEP7 フェーズ別エグジット 感度テスト

目的:
  建値ストップ（+5%で損失ゼロ化）+ ATRトレイリング（+10%以降）の
  ATR倍率（phase_trail_mult）を感度テストし、STEP7の最適値を決定する。

  IS と OOS 2025 の両方で改善するものだけ採用する。

検証軸:
  phase_trail_mult: 1.5, 2.0, 2.5, 3.0
  （固定）phase_breakeven_pct=0.05, phase_trail_start=0.10

比較ベースライン:
  BASELINE（turtle_exit=55 / min_hold=3 / フェーズ出口なし）

期間:
  IS  : 2018-01-01 〜 2024-12-31
  OOS : 2025-01-01 〜 2025-12-31

ユニバース: RSR42統一
"""

from __future__ import annotations
import os, sys, json, warnings, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import backtest.composite_alpha_bt as _bt
from backtest.rsr import calc_universe_rsr
from backtest.universe_builder import download_universe

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

BREAKEVEN_PCT  = 0.05   # +5% で建値ストップ発動（固定）
TRAIL_START    = 0.10   # +10% で ATR トレイル開始（固定）
TRAIL_MULTS    = [1.5, 2.0, 2.5, 3.0]

OUTPUT_DIR  = "C:/ai-trading/backtests"
OUTPUT_JSON = os.path.join(OUTPUT_DIR, f"step7_sensitivity_{time.strftime('%Y-%m-%d')}.json")


# ------------------------------------------------------------------ #
# データロード
# ------------------------------------------------------------------ #
def load_data():
    print("[1/3] データ読み込み中（RSR42, 2018-2025）...")
    rsr_syms     = _bt._load_rsr_universe()
    trade_syms   = rsr_syms
    all_syms     = {**rsr_syms, **trade_syms}
    universe_raw = download_universe(all_syms, start=IS_START, end=FULL_END, verbose=False)
    topix        = _bt._download_topix(IS_START, FULL_END)

    print("[2/3] RSR / Alpha計算中...")
    rsr42_prices = {s: universe_raw[s]["df"]["Close"] for s in rsr_syms if s in universe_raw}
    rsr_df       = calc_universe_rsr(rsr42_prices)
    trade_prices = {s: universe_raw[s]["df"]["Close"] for s in trade_syms if s in universe_raw}
    alpha_df     = _bt.calc_composite_alpha_matrix(trade_prices, window=_bt.COMP_ALPHA_WINDOW).shift(1)
    regime_df    = _bt._calc_regime(topix)

    print("[3/3] テクニカル指標計算中...")
    tech_matrices = _bt._precompute_tech_matrices(universe_raw, list(trade_syms.keys()))
    breadth       = _bt._calc_breadth(universe_raw)

    orig_exit = _bt.TURTLE_EXIT
    print(f"  TURTLE_EXIT={orig_exit}d / MIN_HOLD={3}d\n")

    return universe_raw, rsr_df, alpha_df, regime_df, trade_syms, rsr_syms, tech_matrices, breadth


# ------------------------------------------------------------------ #
# 1シナリオ実行
# ------------------------------------------------------------------ #
def run_one(universe_raw, rsr_df, alpha_df, regime_df, trade_syms, rsr_syms,
            tech_matrices, breadth, trail_mult, start, end) -> dict:
    return _bt.run_scenario(
        scenario          = "BASELINE",
        universe_raw      = universe_raw,
        rsr_df            = rsr_df,
        alpha_df          = alpha_df,
        regime_df         = regime_df,
        trade_syms        = trade_syms,
        rsr_syms          = rsr_syms,
        start             = start,
        end               = end,
        capital           = _bt.CAPITAL,
        verbose           = False,
        tech_matrices     = tech_matrices,
        breadth_series    = breadth,
        min_hold          = 3,
        use_phase_exit    = True if trail_mult is not None else False,
        phase_breakeven_pct = BREAKEVEN_PCT,
        phase_trail_start = TRAIL_START,
        phase_trail_mult  = trail_mult if trail_mult is not None else 2.0,
    )


# ------------------------------------------------------------------ #
# メイン
# ------------------------------------------------------------------ #
def main():
    print("=" * 72)
    print("  STEP7 フェーズ別エグジット感度テスト")
    print(f"  breakeven={BREAKEVEN_PCT:.0%}  trail_start={TRAIL_START:.0%}")
    print(f"  trail_mult sweep: {TRAIL_MULTS}")
    print("=" * 72 + "\n")

    data = load_data()
    universe_raw, rsr_df, alpha_df, regime_df, trade_syms, rsr_syms, tech_matrices, breadth = data

    results = {}

    # ── BASELINE（フェーズ出口なし）──
    print("── BASELINE（フェーズ出口なし）──")
    for period, s, e in [("IS", IS_START, IS_END), ("OOS", OOS_START, OOS_END)]:
        r = _bt.run_scenario(
            scenario="BASELINE", universe_raw=universe_raw, rsr_df=rsr_df,
            alpha_df=alpha_df, regime_df=regime_df, trade_syms=trade_syms,
            rsr_syms=rsr_syms, start=s, end=e, capital=_bt.CAPITAL,
            verbose=False, tech_matrices=tech_matrices,
            breadth_series=breadth, min_hold=3,
        )
        tag = f"tpm={round(r.get('n_trades',0)/12,1)}" if period == "OOS" else ""
        print(f"  {period:3s}  CAGR={r.get('cagr',0):+6.1f}%  Sharpe={r.get('sharpe',0):6.3f}"
              f"  MaxDD={r.get('max_dd',0):6.1f}%  hold={r.get('avg_hold_days',0):.0f}d  {tag}")
        results[f"baseline_{period}"] = r
    print()

    # ── ATR倍率スイープ ──
    for mult in TRAIL_MULTS:
        key = f"step7_mult{mult:.1f}"
        print(f"── phase_trail_mult={mult} ──")
        seg = {}
        for period, s, e in [("IS", IS_START, IS_END), ("OOS", OOS_START, OOS_END)]:
            r = run_one(universe_raw, rsr_df, alpha_df, regime_df,
                        trade_syms, rsr_syms, tech_matrices, breadth, mult, s, e)
            tag = f"tpm={round(r.get('n_trades',0)/12,1)}" if period == "OOS" else ""
            print(f"  {period:3s}  CAGR={r.get('cagr',0):+6.1f}%  Sharpe={r.get('sharpe',0):6.3f}"
                  f"  MaxDD={r.get('max_dd',0):6.1f}%  hold={r.get('avg_hold_days',0):.0f}d  {tag}")
            seg[period] = r
        results[key] = seg
        print()

    # ── サマリー ──
    print("=" * 72)
    print("  サマリー比較（BASELINE vs STEP7各設定）")
    print(f"  {'設定':20s}  {'IS CAGR':>9}  {'IS Sharpe':>10}  {'IS MaxDD':>9}  {'OOS CAGR':>9}  {'OOS Sharpe':>11}  {'IS→OOS':>7}")
    print("  " + "-" * 80)

    bl_is  = results.get("baseline_IS", {})
    bl_oos = results.get("baseline_OOS", {})
    print(f"  {'BASELINE':20s}  {bl_is.get('cagr',0):>+8.1f}%  {bl_is.get('sharpe',0):>10.3f}"
          f"  {bl_is.get('max_dd',0):>+8.1f}%  {bl_oos.get('cagr',0):>+8.1f}%  {bl_oos.get('sharpe',0):>11.3f}  {'—':>7}")

    for mult in TRAIL_MULTS:
        key = f"step7_mult{mult:.1f}"
        if key not in results:
            continue
        r_is  = results[key].get("IS", {})
        r_oos = results[key].get("OOS", {})
        is_better  = "✅" if r_is.get("sharpe", 0)  > bl_is.get("sharpe", 0)  else "  "
        oos_better = "✅" if r_oos.get("sharpe", 0) > bl_oos.get("sharpe", 0) else "  "
        both = "★" if is_better.strip() and oos_better.strip() else ""
        print(f"  {f'mult={mult}':20s}  {r_is.get('cagr',0):>+8.1f}%  {r_is.get('sharpe',0):>10.3f}{is_better}"
              f"  {r_is.get('max_dd',0):>+8.1f}%  {r_oos.get('cagr',0):>+8.1f}%  {r_oos.get('sharpe',0):>11.3f}{oos_better}  {both:>7}")

    # ── JSON保存 ──
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
