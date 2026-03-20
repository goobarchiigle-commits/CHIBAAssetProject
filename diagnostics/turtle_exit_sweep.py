"""diagnostics/turtle_exit_sweep.py — turtle_exit 感度分析"""
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
OUT_DIR = "C:/Users/owner/.claude/レポート"

SECTOR_STRATEGY = {
    "海運":"fujiko","機械":"fujiko","電機精密":"fujiko","商社":"fujiko",
    "電機":"fujiko","食品":"fujiko","情報通信":"dynamic","小売":"dynamic",
    "医薬品":"dynamic","不動産":"dynamic","サービス":"dynamic","レジャー":"fujiko",
    "ガス":"mean_rev","鉄鋼":"mean_rev","銀行":"mean_rev","保険":"mean_rev",
    "輸送機器":"mean_rev","化学":"mean_rev","陸運":"mean_rev","石油":"mean_rev",
}
MR = dict(rsi_period=5, rsi_entry=25.0, rsi_exit=65.0, ma_long=200,
          stop_loss_pct=0.07, max_hold_days=10, knife_threshold=0.15)


def build_strats(universe, rsr, turtle_exit):
    strats = {}
    for sym, info in universe.items():
        rsr_s = rsr[sym] if sym in rsr.columns else None
        rule  = SECTOR_STRATEGY.get(info["sector"], "dynamic")
        if rule == "mean_rev":
            strats[sym] = MeanReversionStrategy(**MR)
        else:
            strats[sym] = FujikoStrategy(
                rsr_series=rsr_s, min_sepa=6, min_rsr=70.0, mom_period=21,
                turtle_entry=20, turtle_exit=turtle_exit, use_turtle_entry=True,
            )
    return strats


def run(universe, strats):
    return PortfolioEngine(
        universe=universe, strategy=strats, capital=CAPITAL,
        max_dd_limit=0.15, min_sectors=1, max_positions=3,
        cost=TradeCost(), vol_target=0.0, max_single_weight=0.25,
        vol_window=20, use_idm=False, use_dd_scalar=False,
    ).run()


def avg_hold(res):
    t, pairs = res.trades, []
    for sym, grp in t.groupby("symbol"):
        buy_q = []
        for _, row in grp.sort_values("date").iterrows():
            if row["side"] == "BUY":
                buy_q.append(row["date"])
            elif row["side"].startswith("SELL") and buy_q:
                pairs.append((row["date"] - buy_q.pop(0)).days)
    return float(np.mean(pairs)) if pairs else float("nan")


def calmar(r):
    return r.cagr / abs(r.max_drawdown) if r.max_drawdown else float("inf")


def main():
    with open("configs/universe/2026Q1_temporal24.json", encoding="utf-8") as f:
        tickers = json.load(f)["symbols"]

    print("データ取得中...")
    universe = download_universe(tickers, start=START, end=END, verbose=False)
    prices   = {s: info["df"]["Close"] for s, info in universe.items()}
    rsr      = calc_universe_rsr(prices)

    # 検証する turtle_exit 値
    sweep = [5, 7, 10, 15, 20, 25, 30, 40]
    rows  = []

    SEP = "=" * 72
    print(SEP)
    print("  turtle_exit 感度分析（turtle_entry=20固定）")
    print(SEP)
    print(f"  {'exit':>5} {'CAGR%':>8} {'MaxDD%':>8} {'Calmar':>8} "
          f"{'Sharpe':>7} {'取引数':>6} {'avg_hold':>9} {'勝率%':>7}")
    print("-" * 72)

    for te in sweep:
        strats = build_strats(universe, rsr, turtle_exit=te)
        res    = run(universe, strats)
        ah     = avg_hold(res)
        mark   = " *" if te == 10 else ""   # 現行設定
        rows.append({
            "turtle_exit": te,
            "cagr":        res.cagr,
            "maxdd":       res.max_drawdown,
            "calmar":      calmar(res),
            "sharpe":      res.sharpe_ratio,
            "n_trades":    res.n_trades,
            "avg_hold":    ah,
            "win_rate":    res.win_rate,
        })
        print(
            f"  {te:>5}{mark}"
            f" {res.cagr*100:>+7.2f}%"
            f" {res.max_drawdown*100:>8.2f}%"
            f" {calmar(res):>8.3f}"
            f" {res.sharpe_ratio:>7.3f}"
            f" {res.n_trades:>6}回"
            f" {ah:>8.1f}日"
            f" {res.win_rate*100:>6.1f}%"
            f"{mark}"
        )

    print(SEP)

    df = pd.DataFrame(rows)
    best_idx = df["calmar"].idxmax()
    best     = df.loc[best_idx]
    cur      = df[df["turtle_exit"] == 10].iloc[0]

    print(f"\n  現行 (exit=10): CAGR={cur['cagr']*100:+.2f}%  Calmar={cur['calmar']:.3f}"
          f"  avg_hold={cur['avg_hold']:.1f}日")
    print(f"  最良 (exit={int(best['turtle_exit'])}): CAGR={best['cagr']*100:+.2f}%"
          f"  Calmar={best['calmar']:.3f}  avg_hold={best['avg_hold']:.1f}日")

    gain_cagr = (best["cagr"] - cur["cagr"]) * 100
    if abs(gain_cagr) >= 1.0:
        print(f"\n  -> turtle_exit={int(best['turtle_exit'])} で "
              f"CAGR {gain_cagr:+.2f}%pt {'改善' if gain_cagr > 0 else '悪化'}")
    else:
        print(f"\n  -> turtle_exit の変更による改善は軽微 ({gain_cagr:+.2f}%pt)")

    # ── グラフ ──
    BG, PANEL, GRID = "#0d1117", "#161b22", "#21262d"
    TEXT, MUTED     = "#e6edf3", "#8b949e"

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), facecolor=BG)
    axes = axes.flatten()
    for ax in axes:
        ax.set_facecolor(PANEL)
        for sp in ax.spines.values():
            sp.set_color(GRID)

    x   = df["turtle_exit"].values
    cur_x = 10

    configs = [
        (axes[0], df["cagr"]*100,       "CAGR%",       "#2ea043", True),
        (axes[1], df["calmar"],          "Calmar",      "#388bfd", True),
        (axes[2], df["maxdd"].abs()*100, "MaxDD% (abs)","#da3633", False),
        (axes[3], df["avg_hold"],        "avg_hold (日)","#8957e5", True),
    ]

    for ax, y, title, color, higher_better in configs:
        ax.plot(x, y, color=color, linewidth=2, marker="o", zorder=3)
        ax.axvline(x=cur_x, color="#f0b429", linewidth=1.2, linestyle="--", alpha=0.8)
        best_y = y.max() if higher_better else y.min()
        best_x = x[y.values.argmax() if higher_better else y.values.argmin()]
        ax.scatter([best_x], [best_y], color="#f0b429", s=80, zorder=4)
        ax.set_title(title, color=TEXT, fontsize=11)
        ax.set_xlabel("turtle_exit (日)", color=MUTED)
        ax.tick_params(colors=MUTED)
        ax.grid(color=GRID, linewidth=0.5, zorder=0)
        ax.text(cur_x + 0.3, ax.get_ylim()[0] + (ax.get_ylim()[1]-ax.get_ylim()[0])*0.05,
                "現行\n(10日)", color="#f0b429", fontsize=8)

    fig.suptitle("turtle_exit 感度分析  2018-2024  (turtle_entry=20固定)",
                 color=TEXT, fontsize=13, y=1.01)
    plt.tight_layout()

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "turtle_exit_sweep.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"\n  グラフ保存: {out}")

    # JSON保存
    df.to_json("results/turtle_exit_sweep.json",
               orient="records", force_ascii=False, indent=2)
    print("  データ保存: results/turtle_exit_sweep.json")


if __name__ == "__main__":
    main()
