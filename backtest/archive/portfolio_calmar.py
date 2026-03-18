"""
backtest/portfolio_calmar.py
Calmar比 2.0 突破テスト

【仮説】
  前回の分析から:
    - F宇宙(21銘柄/緩和A) + Vol調整 → Calmar 1.90 (ベスト)
    - max_positions=2 + 均等 → CAGR +12.06% (最高リターン)

  組み合わせ: F宇宙(21銘柄) × max_positions=3 or 2 × Vol調整
  → CAGRを上げつつDDを抑制 → Calmar 2.0 突破を狙う

【比較シナリオ】
  F_base : 21銘柄 / max6 / Vol調整  (前回ベスト Calmar=1.90)
  F_m3v  : 21銘柄 / max3 / Vol調整
  F_m2v  : 21銘柄 / max2 / Vol調整
  F_m3e  : 21銘柄 / max3 / 均等配分
  G_m3v  : 27銘柄 / max3 / Vol調整  (緩和B宇宙でも試す)

【実行方法】
  python -m backtest.portfolio_calmar
"""

from __future__ import annotations

import os, sys, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
plt.rcParams["font.family"] = "MS Gothic"

import numpy as np
import pandas as pd

from backtest.universe_builder        import download_universe
from backtest.rsr                     import calc_universe_rsr
from backtest.fujiko_strategy         import FujikoStrategy
from backtest.mean_reversion_strategy import MeanReversionStrategy
from backtest.portfolio_engine        import PortfolioEngine
from backtest.engine                  import TradeCost

START   = "2018-01-01"
END     = "2024-12-31"
CAPITAL = 2_000_000

FUJIKO_PARAMS = dict(
    min_sepa=6, min_rsr=75.0, mom_period=21,
    turtle_entry=20, turtle_exit=10, use_turtle_entry=True,
)
MR_PARAMS = dict(
    rsi_period=5, rsi_entry=25.0, rsi_exit=65.0,
    ma_long=200, stop_loss_pct=0.07, max_hold_days=10, knife_threshold=0.15,
)
VOL_TARGET = 0.02
MAX_SINGLE_WEIGHT = 0.25
VOL_WINDOW = 20

SECTOR_STRATEGY: dict[str, str] = {
    "海運":"fujiko","機械":"fujiko","電機精密":"fujiko","商社":"fujiko",
    "電機":"fujiko","ゲーム":"fujiko","レジャー":"fujiko","食品":"fujiko",
    "ガス":"mean_rev","鉄鋼":"mean_rev","銀行":"mean_rev","保険":"mean_rev",
    "輸送機器":"mean_rev","化学":"mean_rev","小売":"mean_rev",
    "サービス":"dynamic","医薬品":"dynamic","不動産":"dynamic",
    "情報通信":"dynamic","陸運":"dynamic",
}

# テストシナリオ: (label, min_sharpe, max_dd_filter, max_positions, use_vol)
SCENARIOS = [
    ("F_base: 21銘柄/max6/Vol調整", 0.4, 0.25, 6, True),
    ("F_m3v:  21銘柄/max3/Vol調整", 0.4, 0.25, 3, True),
    ("F_m3e:  21銘柄/max3/均等",    0.4, 0.25, 3, False),
    ("F_m2v:  21銘柄/max2/Vol調整", 0.4, 0.25, 2, True),
    ("G_m3v:  27銘柄/max3/Vol調整", 0.3, 0.30, 3, True),
]


def make_strategy(sym: str, sector: str, strategy_label: str, rsr_s) -> object:
    rule = SECTOR_STRATEGY.get(sector, "dynamic")
    if rule == "mean_rev" or (rule == "dynamic" and "平均回帰" in strategy_label):
        return MeanReversionStrategy(**MR_PARAMS)
    return FujikoStrategy(rsr_series=rsr_s, **FUJIKO_PARAMS)


def main() -> None:
    print("=" * 72)
    print("  Calmar比 2.0 突破テスト")
    print("  F宇宙(21銘柄) × max_positions=3/2 × Vol調整の組み合わせ")
    print(f"  期間: {START} 〜 {END}  /  初期資本: ¥{CAPITAL:,}")
    print("=" * 72)

    # ---- 1. dynamic_selection.csv 読み込み ----
    df_sel = pd.read_csv("data/dynamic_selection.csv", encoding="utf-8-sig")
    sym_to_strategy = dict(zip(df_sel["symbol"], df_sel["strategy"]))

    # ---- 2. 最広宇宙(27銘柄/緩和B)のデータ一括取得 ----
    broadest = df_sel[(df_sel["sharpe"] > 0.3) & (df_sel["maxdd"].abs() < 0.30)]
    broad_tickers = {r["symbol"]: r["sector"] for _, r in broadest.iterrows()}
    print(f"\n[1/3] データ取得中（最広ユニバース {len(broad_tickers)}銘柄）...")
    universe_all = download_universe(broad_tickers, start=START, end=END)
    print(f"  取得完了: {len(universe_all)} 銘柄")

    # ---- 3. RSR 計算 ----
    print("\n[2/3] RSR 計算中...")
    universe_prices = {sym: info["df"]["Close"] for sym, info in universe_all.items()}
    rsr_universe    = calc_universe_rsr(universe_prices)
    print(f"  RSR計算完了: {rsr_universe.shape[1]} 銘柄")

    # ---- 4. 各シナリオでバックテスト ----
    print("\n[3/3] バックテスト実行中...")
    cost = TradeCost()
    results = {}

    for label, min_shr, max_dd_filter, max_pos, use_vol in SCENARIOS:
        # 対象銘柄を抽出
        mask = (df_sel["sharpe"] > min_shr) & (df_sel["maxdd"].abs() < max_dd_filter)
        tier_syms = set(df_sel[mask]["symbol"].tolist())
        universe_t = {sym: info for sym, info in universe_all.items() if sym in tier_syms}

        if not universe_t:
            print(f"  {label}: 銘柄なし → スキップ")
            continue

        # 戦略辞書構築
        strat_dict = {}
        for sym, info in universe_t.items():
            rsr_s  = rsr_universe[sym] if sym in rsr_universe.columns else None
            lbl_   = sym_to_strategy.get(sym, "フジコ法")
            strat_dict[sym] = make_strategy(sym, info["sector"], lbl_, rsr_s)

        fj_n = sum(1 for s in strat_dict.values() if isinstance(s, FujikoStrategy))
        mr_n = sum(1 for s in strat_dict.values() if isinstance(s, MeanReversionStrategy))
        print(f"  ▶ {label} ({len(universe_t)}銘柄: FJ={fj_n} MR={mr_n}"
              f" max_pos={max_pos} {'Vol調整' if use_vol else '均等'})...")

        engine = PortfolioEngine(
            universe          = universe_t,
            strategy          = strat_dict,
            capital           = CAPITAL,
            max_dd_limit      = 0.15,
            min_sectors       = 2,          # max_pos=2/3時はmin_sectors=2に緩和
            max_positions     = max_pos,
            cost              = cost,
            vol_target        = VOL_TARGET if use_vol else 0.0,
            max_single_weight = MAX_SINGLE_WEIGHT,
            vol_window        = VOL_WINDOW,
        )
        results[label] = engine.run()

    # ---- 比較テーブル ----
    print("\n\n" + "=" * 80)
    print("  Calmar比 2.0 突破テスト — 比較結果")
    print("=" * 80)

    header = f"  {'指標':<14}"
    for lbl in results:
        header += f" {lbl:>22}"
    print(header)
    print("  " + "-" * (14 + 23 * len(results)))

    def calmar(r):
        dd = abs(r.max_drawdown)
        return r.cagr / dd if dd > 0 else float("inf")

    metrics = [
        ("年率CAGR",        lambda r: f"{r.cagr*100:>+7.2f}%"),
        ("最大DD",          lambda r: f"{r.max_drawdown*100:>+7.2f}%"),
        ("Calmar比",        lambda r: f"{calmar(r):>9.3f}"),
        ("シャープ比",      lambda r: f"{r.sharpe_ratio:>9.3f}"),
        ("総リターン",      lambda r: f"{r.total_return*100:>+7.2f}%"),
        ("決済回数",        lambda r: f"{r.n_trades:>9}回"),
        ("勝率",            lambda r: f"{r.win_rate*100:>8.1f}%"),
        ("CB発動",          lambda r: f"{r.n_circuit_breaker:>9}回"),
        ("平均保有銘柄",    lambda r: f"{r.n_positions.mean():>7.2f}銘柄"),
        ("最終資産",        lambda r: f"¥{r.equity_curve.iloc[-1]:>10,.0f}"),
        ("Phase1クリア",    lambda r: "YES ✓" if (r.sharpe_ratio>0.5 and abs(r.max_drawdown)<=0.20) else "NO  ✗"),
        ("Calmar≥2.0",     lambda r: "★ 達成 ★" if calmar(r) >= 2.0 else f"あと{2.0-calmar(r):.3f}"),
    ]

    for mlabel, fn in metrics:
        row = f"  {mlabel:<14}"
        for res in results.values():
            row += f" {fn(res):>22}"
        print(row)

    # ---- Calmarランキング ----
    print("\n  === Calmar比ランキング ===")
    ranked = sorted(results.items(), key=lambda x: calmar(x[1]), reverse=True)
    for rank, (lbl, res) in enumerate(ranked, 1):
        c = calmar(res)
        mark = " ★ 達成！" if c >= 2.0 else ""
        print(f"  {rank}. {lbl}: Calmar={c:.3f}  CAGR={res.cagr*100:+.2f}%  MaxDD={res.max_drawdown*100:.2f}%{mark}")

    # ---- 個別サマリー ----
    for lbl, res in results.items():
        print(f"\n--- {lbl} ---")
        res.summary()

    # ---- グラフ ----
    os.makedirs("data", exist_ok=True)
    png_path = "data/portfolio_calmar.png"

    fig, axes = plt.subplots(3, 1, figsize=(15, 13), sharex=True)
    fig.suptitle(
        "Calmar比 2.0 突破テスト（2018–2024）\n"
        "F宇宙21銘柄 × max_positions × Vol調整の組み合わせ",
        fontsize=12,
    )
    cmap   = plt.cm.tab10
    colors = {lbl: cmap(i) for i, lbl in enumerate(results)}

    ax = axes[0]
    for lbl, res in results.items():
        c = calmar(res)
        ax.plot(
            res.equity_curve.index,
            res.equity_curve.values / 10_000,
            color=colors[lbl], linewidth=1.8,
            label=f"{lbl}  CAGR={res.cagr*100:>+.1f}%  Calmar={c:.2f}",
        )
    ax.axhline(CAPITAL / 10_000, color="black", linestyle="--", linewidth=0.8, label="元本")
    ax.set_ylabel("資産（万円）")
    ax.set_title("資産推移")
    ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"¥{x:.0f}万"))

    ax = axes[1]
    for lbl, res in results.items():
        ax.fill_between(res.dd_series.index, res.dd_series.values*100, 0,
                        alpha=0.25, color=colors[lbl])
        ax.plot(res.dd_series.index, res.dd_series.values*100,
                color=colors[lbl], linewidth=0.9,
                label=f"{lbl} MaxDD={res.max_drawdown*100:.1f}%")
    ax.axhline(-15, color="darkred", linestyle="--", linewidth=1.2, label="CB閾値")
    ax.set_ylabel("ドローダウン（%）")
    ax.legend(fontsize=8)

    ax = axes[2]
    for lbl, res in results.items():
        ax.step(res.n_positions.index, res.n_positions.values,
                where="post", color=colors[lbl], linewidth=1.3,
                label=f"{lbl} 平均={res.n_positions.mean():.2f}")
    ax.axhline(2, color="gray", linestyle=":", linewidth=0.8)
    ax.set_ylabel("保有銘柄数")
    ax.set_ylim(0, 7)
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"\n  PNG: {png_path}")
    print("\n完了。")


if __name__ == "__main__":
    main()
