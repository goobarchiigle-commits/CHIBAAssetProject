"""diagnostics/pnl_vs_holding.py — PnL vs 保有日数分析"""
import sys, json, warnings, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "MS Gothic"

from backtest.universe_builder        import download_universe
from backtest.rsr                     import calc_universe_rsr
from backtest.fujiko_strategy         import FujikoStrategy
from backtest.mean_reversion_strategy import MeanReversionStrategy
from backtest.portfolio_engine        import PortfolioEngine
from backtest.engine                  import TradeCost

START   = "2018-01-01"
END     = "2024-12-31"
CAPITAL = 2_000_000
OUT_DIR = "C:/ai-trading/reports"

SECTOR_STRATEGY = {
    "海運":"fujiko","機械":"fujiko","電機精密":"fujiko","商社":"fujiko",
    "電機":"fujiko","食品":"fujiko","情報通信":"dynamic","小売":"dynamic",
    "医薬品":"dynamic","不動産":"dynamic","サービス":"dynamic","レジャー":"fujiko",
    "ガス":"mean_rev","鉄鋼":"mean_rev","銀行":"mean_rev","保険":"mean_rev",
    "輸送機器":"mean_rev","化学":"mean_rev","陸運":"mean_rev","石油":"mean_rev",
}
MR = dict(rsi_period=5, rsi_entry=25.0, rsi_exit=65.0, ma_long=200,
          stop_loss_pct=0.07, max_hold_days=10, knife_threshold=0.15)
FJ = dict(min_sepa=6, min_rsr=70.0, mom_period=21,
          turtle_entry=20, turtle_exit=20, use_turtle_entry=True)


def main():
    with open("configs/universe/2026Q1_temporal24.json", encoding="utf-8") as f:
        tickers = json.load(f)["symbols"]

    print("データ取得中...")
    universe = download_universe(tickers, start=START, end=END, verbose=False)
    prices   = {s: info["df"]["Close"] for s, info in universe.items()}
    rsr      = calc_universe_rsr(prices)

    strats = {}
    for sym, info in universe.items():
        rsr_s  = rsr[sym] if sym in rsr.columns else None
        rule   = SECTOR_STRATEGY.get(info["sector"], "dynamic")
        strats[sym] = (MeanReversionStrategy(**MR) if rule == "mean_rev"
                       else FujikoStrategy(rsr_series=rsr_s, **FJ))

    res = PortfolioEngine(
        universe=universe, strategy=strats, capital=CAPITAL,
        max_dd_limit=0.15, min_sectors=1, max_positions=3,
        cost=TradeCost(), vol_target=0.0, max_single_weight=0.25,
        vol_window=20, use_idm=False, use_dd_scalar=False,
    ).run()

    # ── BUY-SELL ペアリング（FIFO） ──
    t     = res.trades
    pairs = []
    for sym, grp in t.groupby("symbol"):
        grp   = grp.sort_values("date")
        buy_q = []
        for _, row in grp.iterrows():
            if row["side"] == "BUY":
                buy_q.append(row)
            elif row["side"].startswith("SELL") and buy_q:
                b     = buy_q.pop(0)
                hold  = (row["date"] - b["date"]).days
                cost  = b["qty"] * b["price"]
                pnl_p = row["pnl"] / cost * 100 if cost > 0 else np.nan
                pairs.append({
                    "symbol":    sym,
                    "sector":    row["sector"],
                    "hold_days": hold,
                    "pnl":       row["pnl"],
                    "pnl_pct":   pnl_p,
                    "exit_type": row["side"],
                    "entry":     b["date"],
                    "exit":      row["date"],
                })

    df = pd.DataFrame(pairs)

    # ── バケット集計 ──
    edges  = [0, 5, 10, 15, 20, 30, 45, 60, 999]
    labels = ["1-5", "6-10", "11-15", "16-20", "21-30", "31-45", "46-60", "61+"]
    df["bucket"] = pd.cut(df["hold_days"], bins=edges, labels=labels, right=True)

    stats = df.groupby("bucket", observed=True).agg(
        count      = ("pnl", "count"),
        avg_pnl    = ("pnl", "mean"),
        avg_pnl_pct= ("pnl_pct", "mean"),
        win_rate   = ("pnl", lambda x: (x > 0).mean() * 100),
        median_pnl = ("pnl", "median"),
    ).reset_index()

    # ── テキスト出力 ──
    SEP = "=" * 58
    print(SEP)
    print("  PnL vs 保有日数バケット分析")
    print(SEP)
    print(f"  {'日数':>8} {'件数':>5} {'平均PnL円':>10} {'平均PnL%':>9} {'勝率':>7}")
    print("-" * 58)

    peak_idx = stats["avg_pnl_pct"].idxmax()
    for i, row in stats.iterrows():
        mark = " <- MAX" if i == peak_idx else ""
        print(
            f"  {str(row['bucket']):>8}日"
            f" {int(row['count']):>5}件"
            f" {row['avg_pnl']:>+10,.0f}円"
            f" {row['avg_pnl_pct']:>+8.2f}%"
            f" {row['win_rate']:>6.1f}%"
            f"{mark}"
        )
    print(SEP)
    print(f"  全体 avg_pnl: {df['pnl'].mean():+,.0f}円  ({df['pnl_pct'].mean():+.2f}%)")
    print(f"  全体 勝率:    {(df['pnl'] > 0).mean() * 100:.1f}%")
    print()

    peak_bucket = str(stats.loc[peak_idx, "bucket"])
    peak_pct    = stats.loc[peak_idx, "avg_pnl_pct"]
    cur_avg     = df["hold_days"].mean()

    print("  診断:")
    print(f"    PnL最大バケット : {peak_bucket}日  (avg {peak_pct:+.2f}%)")
    print(f"    現在の avg_hold : {cur_avg:.1f}日")

    peak_lo = int(peak_bucket.split("-")[0]) if "-" in peak_bucket else 61
    if peak_lo > cur_avg + 5:
        print(f"    -> exit が早すぎ。turtle_exit を延ばして {peak_lo}日前後まで保有を試みる価値あり")
    elif peak_lo < cur_avg - 5:
        print(f"    -> 現在の保有期間はやや長すぎ。早めのexit を検討")
    else:
        print("    -> 現在の保有期間は最大PnLバケットに近い。exit タイミングは適切")

    # ── グラフ ──
    BG    = "#0d1117"
    PANEL = "#161b22"
    GRID  = "#21262d"
    TEXT  = "#e6edf3"
    MUTED = "#8b949e"
    BLUE  = "#388bfd"
    GOLD  = "#f0b429"

    fig, axes = plt.subplots(3, 1, figsize=(10, 13), facecolor=BG)
    for ax in axes:
        ax.set_facecolor(PANEL)
        for sp in ax.spines.values():
            sp.set_color(GRID)

    bv    = stats.dropna(subset=["avg_pnl_pct"])
    x_lbl = bv["bucket"].astype(str).tolist()
    x_pos = np.arange(len(x_lbl))

    bar_colors = [
        GOLD if lbl == peak_bucket else ("#238636" if v >= 0 else "#da3633")
        for lbl, v in zip(x_lbl, bv["avg_pnl_pct"])
    ]

    # ax1: avg PnL%
    bars = axes[0].bar(x_pos, bv["avg_pnl_pct"],
                       color=bar_colors, edgecolor=GRID, linewidth=0.5, zorder=3)
    axes[0].axhline(0, color=MUTED, linewidth=0.8, linestyle="--")
    cur_x = [i for i, l in enumerate(x_lbl) if "11" in l or "16" in l]
    if cur_x:
        axes[0].axvline(x=cur_x[0] - 0.5, color=BLUE, linewidth=1.2, linestyle=":", alpha=0.7)
        axes[0].text(cur_x[0] - 0.45, axes[0].get_ylim()[1] * 0.7,
                     f"avg now\n{cur_avg:.0f}日", color=BLUE, fontsize=8)
    for bar, val in zip(bars, bv["avg_pnl_pct"]):
        ypos = val + 0.05 if val >= 0 else val - 0.12
        va   = "bottom" if val >= 0 else "top"
        axes[0].text(bar.get_x() + bar.get_width() / 2, ypos,
                     f"{val:+.2f}%", ha="center", va=va, fontsize=8, color=TEXT)
    axes[0].set_title("平均 PnL% vs 保有日数", color=TEXT, fontsize=12, pad=10)
    axes[0].set_ylabel("平均 PnL%", color=MUTED)
    axes[0].set_xticks(x_pos); axes[0].set_xticklabels(x_lbl, color=MUTED)
    axes[0].tick_params(colors=MUTED)
    axes[0].grid(axis="y", color=GRID, linewidth=0.5, zorder=0)

    # ax2: 勝率
    axes[1].bar(x_pos, bv["win_rate"],
                color=BLUE, alpha=0.75, edgecolor=GRID, linewidth=0.5, zorder=3)
    axes[1].axhline(50, color=MUTED, linewidth=0.8, linestyle="--")
    for i, val in enumerate(bv["win_rate"]):
        axes[1].text(i, val + 0.5, f"{val:.0f}%",
                     ha="center", va="bottom", fontsize=8, color=TEXT)
    axes[1].set_title("勝率 vs 保有日数", color=TEXT, fontsize=12, pad=10)
    axes[1].set_ylabel("勝率 %", color=MUTED)
    axes[1].set_ylim(0, 105)
    axes[1].set_xticks(x_pos); axes[1].set_xticklabels(x_lbl, color=MUTED)
    axes[1].tick_params(colors=MUTED)
    axes[1].grid(axis="y", color=GRID, linewidth=0.5, zorder=0)

    # ax3: 件数
    axes[2].bar(x_pos, bv["count"],
                color="#8957e5", alpha=0.75, edgecolor=GRID, linewidth=0.5, zorder=3)
    for i, val in enumerate(bv["count"]):
        axes[2].text(i, val + 0.3, str(int(val)),
                     ha="center", va="bottom", fontsize=8, color=TEXT)
    axes[2].set_title("取引件数 vs 保有日数", color=TEXT, fontsize=12, pad=10)
    axes[2].set_ylabel("件数", color=MUTED)
    axes[2].set_xlabel("保有日数バケット", color=MUTED)
    axes[2].set_xticks(x_pos); axes[2].set_xticklabels(x_lbl, color=MUTED)
    axes[2].tick_params(colors=MUTED)
    axes[2].grid(axis="y", color=GRID, linewidth=0.5, zorder=0)

    fig.suptitle(
        f"フジコ法  PnL vs 保有日数  2018-2024  (n={len(df)}件  avg={cur_avg:.1f}日)",
        color=TEXT, fontsize=14, y=1.01,
    )
    plt.tight_layout()

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "pnl_vs_holding_days.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"\n  グラフ保存: {out}")


if __name__ == "__main__":
    main()
