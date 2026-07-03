"""MaxDD発生時期の特定（STEP5）"""
import sys, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding='utf-8')

from backtest.composite_alpha_bt import (
    _load_rsr_universe, _download_topix, _calc_regime,
    _calc_breadth, _precompute_tech_matrices, run_scenario,
    START, END, COMP_ALPHA_WINDOW,
)
from backtest.rsr import calc_universe_rsr, calc_composite_alpha_matrix
from backtest.universe_builder import download_universe

print("データ取得中...")
rsr_syms     = _load_rsr_universe()
trade_syms   = rsr_syms
all_syms     = {**rsr_syms}
universe_raw = download_universe(all_syms, start=START, end=END, verbose=False)
topix_close  = _download_topix(START, END)

rsr42_prices = {sym: universe_raw[sym]["df"]["Close"]
                for sym in rsr_syms if sym in universe_raw}
rsr_df = calc_universe_rsr(rsr42_prices)

trade_prices = {sym: universe_raw[sym]["df"]["Close"]
                for sym in trade_syms if sym in universe_raw}
alpha_df = calc_composite_alpha_matrix(trade_prices, window=COMP_ALPHA_WINDOW)
alpha_df = alpha_df.shift(1)

regime_df      = _calc_regime(topix_close)
tech_matrices  = _precompute_tech_matrices(universe_raw, list(trade_syms.keys()))

print("STEP5 実行中...")
res = run_scenario(
    scenario     = "STEP5",
    universe_raw = universe_raw,
    rsr_df       = rsr_df,
    alpha_df     = alpha_df,
    regime_df    = regime_df,
    tech_matrices = tech_matrices,
    rsr_syms     = rsr_syms,
    trade_syms   = trade_syms,
)

eq = res["equity_curve"]  # pd.Series with DatetimeIndex

# MaxDD の発生期間を特定
peak = eq.cummax()
dd   = (eq - peak) / peak

max_dd_val  = dd.min()
max_dd_date = dd.idxmin()
# ピーク（DD開始）を特定
peak_date = eq[:max_dd_date].idxmax()

print(f"\n② MaxDD発生タイミング")
print(f"  MaxDD       : {max_dd_val*100:.1f}%")
print(f"  ピーク日    : {peak_date.date()}（{peak_date.year}年）")
print(f"  ボトム日    : {max_dd_date.date()}（{max_dd_date.year}年）")
print(f"  DD期間      : {(max_dd_date - peak_date).days}日")

# 回復日（equity がピーク水準に戻った日）
recovery = eq[max_dd_date:][eq[max_dd_date:] >= float(peak[max_dd_date])]
if len(recovery) > 0:
    recovery_date = recovery.index[0]
    print(f"  回復日      : {recovery_date.date()}（{recovery_date.year}年）")
    print(f"  回復期間    : {(recovery_date - max_dd_date).days}日")
else:
    print(f"  回復日      : 期間内に未回復")

# 年次DDサマリー
print(f"\n  年次 最大intra-year DD:")
for yr in sorted(eq.index.year.unique()):
    yr_eq = eq[eq.index.year == yr]
    yr_peak = yr_eq.cummax()
    yr_dd = ((yr_eq - yr_peak) / yr_peak).min()
    print(f"  {yr}年: {yr_dd*100:.1f}%  (年次リターン: {res['annual_returns'].get(str(yr), 'N/A')}%)")
