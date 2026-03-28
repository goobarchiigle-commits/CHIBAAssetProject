"""
最大利益トレードの保有日数分析（STEP5 / rsr42-trade モード）
"""
import sys, numpy as np
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, '.')

from backtest.composite_alpha_bt import (
    _load_rsr_universe, _download_topix, _calc_regime,
    _calc_breadth, _precompute_tech_matrices, run_scenario,
    START, END, COMP_ALPHA_WINDOW,
)
from backtest.rsr import calc_universe_rsr, calc_composite_alpha_matrix
from backtest.universe_builder import download_universe

print("データ取得中...")
rsr_syms     = _load_rsr_universe()
trade_syms   = rsr_syms   # rsr42-trade モード
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
breadth_series = _calc_breadth(rsr_df)

print("STEP5 実行中...")
res = run_scenario(
    scenario       = "STEP5",
    universe_raw   = universe_raw,
    rsr_df         = rsr_df,
    alpha_df       = alpha_df,
    regime_df      = regime_df,
    tech_matrices  = tech_matrices,
    rsr_syms       = rsr_syms,
    trade_syms     = trade_syms,
)

trades = res.get("_trades", [])
valid  = [t for t in trades if "exit_idx" in t and "entry_idx" in t]
print(f"総クローズトレード: {len(trades)}件  (hold計算可能: {len(valid)}件)")

hold_arr = np.array([t["exit_idx"] - t["entry_idx"] for t in valid])
pnl_arr  = np.array([t["pnl"] for t in valid])
win_mask = pnl_arr > 0

print()
print("=" * 55)
print("  保有日数 統計（全トレード）")
print(f"  avg     : {hold_arr.mean():.1f} 営業日")
print(f"  median  : {np.median(hold_arr):.0f} 営業日")
print(f"  p25/p75 : {np.percentile(hold_arr,25):.0f} / {np.percentile(hold_arr,75):.0f} 営業日")
print(f"  p95     : {np.percentile(hold_arr,95):.0f} 営業日")
print(f"  max     : {hold_arr.max():.0f} 営業日")
print()
print("  勝ちトレード 保有日数")
w = hold_arr[win_mask]
print(f"  avg     : {w.mean():.1f} 営業日")
print(f"  median  : {np.median(w):.0f} 営業日")
print(f"  p75/p95 : {np.percentile(w,75):.0f} / {np.percentile(w,95):.0f} 営業日")
print()
print("  負けトレード 保有日数")
lv = hold_arr[~win_mask]
print(f"  avg     : {lv.mean():.1f} 営業日")
print(f"  median  : {np.median(lv):.0f} 営業日")

print()
print("  保有日数 分布（全トレード）")
for lo, hi in [(1,3),(4,7),(8,14),(15,30),(31,60)]:
    mask = (hold_arr >= lo) & (hold_arr <= hi)
    cnt  = mask.sum()
    wr   = (mask & win_mask).sum() / cnt * 100 if cnt else 0
    pct  = cnt / len(hold_arr) * 100
    print(f"  {lo:2d}〜{hi:2d}日: {cnt:>4d}件 ({pct:5.1f}%)  勝率 {wr:4.1f}%")

print()
print("=" * 55)
print("  ★ 最大利益 Top10 トレード")
win_trades = [t for t in valid if t["pnl"] > 0]
top10 = sorted(win_trades, key=lambda x: -x["pnl"])[:10]
for i, t in enumerate(top10, 1):
    hd  = t["exit_idx"] - t["entry_idx"]
    pct = (t["exit"] / t["entry"] - 1) * 100
    print(f"  {i:2d}. {t['symbol']}  ¥{t['pnl']:>9,.0f}  {hd:>3d}日  {pct:+.1f}%  [{t.get('reason','?')}]")

print()
print("  ★ avg_hold_days（修正後）:", res.get("avg_hold_days"))
