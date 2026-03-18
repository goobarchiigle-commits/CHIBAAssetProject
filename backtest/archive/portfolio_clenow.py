"""
backtest/portfolio_clenow.py
Clenowスコア（90日指数回帰スロープ×R²）によるランキング効果検証

【仮説】
  Clenow「Stocks on the Move」の知見:
  「スムーズに上昇する銘柄だけを選ぶ（高R²）とリスク調整後リターンが改善する」
  フジコ法の合格銘柄の中でClenowスコア上位を優先することで
  Calmar比が 1.825 → 2.0 を突破できる。

【比較シナリオ】
  F_base   : 21銘柄 / max2 / Vol調整 / Clenowなし  （現在ベスト Calmar=1.825）
  F_clenow : 21銘柄 / max2 / Vol調整 / Clenowスコアでソート
  G_clenow : 27銘柄 / max3 / Vol調整 / Clenowスコアでソート
  F_clenow3: 21銘柄 / max3 / Vol調整 / Clenowスコアでソート（資本効率向上狙い）

【実行方法】
  python -m backtest.portfolio_clenow
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
from backtest.rsr                     import calc_universe_rsr, calc_universe_clenow
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
VOL_TARGET        = 0.02
MAX_SINGLE_WEIGHT = 0.25
VOL_WINDOW        = 20
CLENOW_WINDOW     = 90   # 90日指数回帰

SECTOR_STRATEGY: dict[str, str] = {
    "海運":"fujiko","機械":"fujiko","電機精密":"fujiko","商社":"fujiko",
    "電機":"fujiko","ゲーム":"fujiko","レジャー":"fujiko","食品":"fujiko",
    "ガス":"mean_rev","鉄鋼":"mean_rev","銀行":"mean_rev","保険":"mean_rev",
    "輸送機器":"mean_rev","化学":"mean_rev","小売":"mean_rev",
    "サービス":"dynamic","医薬品":"dynamic","不動産":"dynamic",
    "情報通信":"dynamic","陸運":"dynamic",
}

# (label, min_sharpe, max_dd_filter, max_pos, use_clenow)
SCENARIOS = [
    ("F_base:   21銘柄/max2/Vol/Clenowなし",  0.4, 0.25, 2, False),
    ("F_clenow: 21銘柄/max2/Vol/Clenow有り",  0.4, 0.25, 2, True),
    ("F_clen3:  21銘柄/max3/Vol/Clenow有り",  0.4, 0.25, 3, True),
    ("G_clenow: 27銘柄/max3/Vol/Clenow有り",  0.3, 0.30, 3, True),
]


def make_strategy(sym: str, sector: str, strategy_label: str, rsr_s) -> object:
    rule = SECTOR_STRATEGY.get(sector, "dynamic")
    if rule == "mean_rev" or (rule == "dynamic" and "平均回帰" in strategy_label):
        return MeanReversionStrategy(**MR_PARAMS)
    return FujikoStrategy(rsr_series=rsr_s, **FUJIKO_PARAMS)


def main() -> None:
    print("=" * 72)
    print("  Clenowスコア（90日指数回帰スロープ×R²）効果検証")
    print("  Clenow「Stocks on the Move」— スムーズ上昇銘柄優先戦略")
    print(f"  期間: {START} 〜 {END}  /  初期資本: ¥{CAPITAL:,}")
    print("=" * 72)

    # ---- 1. 銘柄リスト読み込み ----
    df_sel = pd.read_csv("data/dynamic_selection.csv", encoding="utf-8-sig")
    sym_to_strategy = dict(zip(df_sel["symbol"], df_sel["strategy"]))

    # ---- 2. 最広宇宙のデータ一括取得 ----
    broadest = df_sel[(df_sel["sharpe"] > 0.3) & (df_sel["maxdd"].abs() < 0.30)]
    broad_tickers = {r["symbol"]: r["sector"] for _, r in broadest.iterrows()}
    print(f"\n[1/4] データ取得中（{len(broad_tickers)}銘柄）...")
    universe_all = download_universe(broad_tickers, start=START, end=END)
    print(f"  取得完了: {len(universe_all)} 銘柄")

    # ---- 3. RSR + Clenowスコア計算 ----
    print("\n[2/4] RSR + Clenowスコア計算中...")
    universe_prices = {sym: info["df"]["Close"] for sym, info in universe_all.items()}
    rsr_universe    = calc_universe_rsr(universe_prices)
    print(f"  RSR計算完了: {rsr_universe.shape[1]} 銘柄")

    print(f"  Clenowスコア計算中（window={CLENOW_WINDOW}日）...")
    clenow_universe = calc_universe_clenow(universe_prices, window=CLENOW_WINDOW)
    print(f"  Clenowスコア計算完了: {clenow_universe.shape[1]} 銘柄")

    # スコアのサマリー表示
    latest_scores = clenow_universe.iloc[-1].sort_values(ascending=False)
    print(f"\n  Clenowスコア上位10銘柄（{clenow_universe.index[-1].date()}時点）:")
    for sym, score in latest_scores.head(10).items():
        sector = broad_tickers.get(sym, "")
        print(f"    {sym} ({sector}): {score:>8.1f}")

    # ---- 4. 各シナリオでバックテスト ----
    print("\n[3/4] バックテスト実行中...")
    cost = TradeCost()
    results = {}

    for label, min_shr, max_dd_filter, max_pos, use_clenow in SCENARIOS:
        mask      = (df_sel["sharpe"] > min_shr) & (df_sel["maxdd"].abs() < max_dd_filter)
        tier_syms = set(df_sel[mask]["symbol"].tolist())
        universe_t = {sym: info for sym, info in universe_all.items() if sym in tier_syms}

        if not universe_t:
            continue

        strat_dict = {}
        for sym, info in universe_t.items():
            rsr_s = rsr_universe[sym] if sym in rsr_universe.columns else None
            lbl_  = sym_to_strategy.get(sym, "フジコ法")
            strat_dict[sym] = make_strategy(sym, info["sector"], lbl_, rsr_s)

        # Clenowスコア（使用シナリオのみ）
        clenow_df = clenow_universe[
            [c for c in clenow_universe.columns if c in tier_syms]
        ] if use_clenow else None

        fj_n = sum(1 for s in strat_dict.values() if isinstance(s, FujikoStrategy))
        mr_n = sum(1 for s in strat_dict.values() if isinstance(s, MeanReversionStrategy))
        clenow_str = "Clenow有" if use_clenow else "Clenowなし"
        print(f"  ▶ {label} ({len(universe_t)}銘柄: FJ={fj_n} MR={mr_n} [{clenow_str}])...")

        engine = PortfolioEngine(
            universe          = universe_t,
            strategy          = strat_dict,
            capital           = CAPITAL,
            max_dd_limit      = 0.15,
            min_sectors       = 2,
            max_positions     = max_pos,
            cost              = cost,
            vol_target        = VOL_TARGET,
            max_single_weight = MAX_SINGLE_WEIGHT,
            vol_window        = VOL_WINDOW,
            clenow_scores     = clenow_df,
        )
        results[label] = engine.run()

    # ---- 比較テーブル ----
    print("\n\n" + "=" * 80)
    print("  Clenowスコア効果検証 — 比較結果")
    print("=" * 80)

    def calmar(r):
        dd = abs(r.max_drawdown)
        return r.cagr / dd if dd > 0 else float("inf")

    header = f"  {'指標':<16}"
    for lbl in results:
        header += f" {lbl:>26}"
    print(header)
    print("  " + "-" * (16 + 27 * len(results)))

    metrics = [
        ("年率CAGR",        lambda r: f"{r.cagr*100:>+8.2f}%"),
        ("最大DD",          lambda r: f"{r.max_drawdown*100:>+8.2f}%"),
        ("Calmar比",        lambda r: f"{calmar(r):>10.3f}"),
        ("シャープ比",      lambda r: f"{r.sharpe_ratio:>10.3f}"),
        ("総リターン",      lambda r: f"{r.total_return*100:>+8.2f}%"),
        ("決済回数",        lambda r: f"{r.n_trades:>10}回"),
        ("勝率",            lambda r: f"{r.win_rate*100:>9.1f}%"),
        ("CB発動",          lambda r: f"{r.n_circuit_breaker:>10}回"),
        ("平均保有銘柄",    lambda r: f"{r.n_positions.mean():>8.2f}銘柄"),
        ("最終資産",        lambda r: f"¥{r.equity_curve.iloc[-1]:>12,.0f}"),
        ("Calmar≥2.0",     lambda r: "★ 達成 ★" if calmar(r) >= 2.0 else f"あと{2.0-calmar(r):.3f}"),
    ]

    for mlabel, fn in metrics:
        row = f"  {mlabel:<16}"
        for res in results.values():
            row += f" {fn(res):>26}"
        print(row)

    # ---- Calmarランキング ----
    print("\n  === Calmar比ランキング ===")
    ranked = sorted(results.items(), key=lambda x: calmar(x[1]), reverse=True)
    for rank, (lbl, res) in enumerate(ranked, 1):
        c = calmar(res)
        mark = " ★ Calmar 2.0 達成！" if c >= 2.0 else f" (あと{2.0-c:.3f})"
        print(f"  {rank}. {lbl}")
        print(f"     Calmar={c:.3f}  CAGR={res.cagr*100:+.2f}%  "
              f"MaxDD={res.max_drawdown*100:.2f}%  Sharpe={res.sharpe_ratio:.3f}{mark}")

    # ---- 個別サマリー ----
    for lbl, res in results.items():
        print(f"\n--- {lbl} ---")
        res.summary()

    # ---- グラフ ----
    os.makedirs("data", exist_ok=True)
    png_path = "data/portfolio_clenow.png"

    fig, axes = plt.subplots(4, 1, figsize=(15, 16), sharex=True)
    fig.suptitle(
        "Clenowスコア（90日指数回帰×R²）効果検証（2018–2024）\n"
        "Clenow「Stocks on the Move」— スムーズ上昇銘柄優先",
        fontsize=12,
    )
    cmap   = plt.cm.tab10
    colors = {lbl: cmap(i) for i, lbl in enumerate(results)}

    # 資産推移
    ax = axes[0]
    for lbl, res in results.items():
        c = calmar(res)
        ax.plot(res.equity_curve.index, res.equity_curve.values / 10_000,
                color=colors[lbl], linewidth=1.8,
                label=f"{lbl}  CAGR={res.cagr*100:>+.1f}%  Calmar={c:.2f}")
    ax.axhline(CAPITAL / 10_000, color="black", linestyle="--", linewidth=0.8, label="元本")
    ax.set_ylabel("資産（万円）")
    ax.set_title("資産推移")
    ax.legend(fontsize=7)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"¥{x:.0f}万"))

    # ドローダウン
    ax = axes[1]
    for lbl, res in results.items():
        ax.fill_between(res.dd_series.index, res.dd_series.values*100, 0,
                        alpha=0.25, color=colors[lbl])
        ax.plot(res.dd_series.index, res.dd_series.values*100,
                color=colors[lbl], linewidth=0.9,
                label=f"{lbl} MaxDD={res.max_drawdown*100:.1f}%")
    ax.axhline(-15, color="darkred", linestyle="--", linewidth=1.2, label="CB閾値")
    ax.set_ylabel("ドローダウン（%）")
    ax.legend(fontsize=7)

    # 保有銘柄数
    ax = axes[2]
    for lbl, res in results.items():
        ax.step(res.n_positions.index, res.n_positions.values,
                where="post", color=colors[lbl], linewidth=1.3,
                label=f"{lbl} 平均={res.n_positions.mean():.2f}")
    ax.set_ylabel("保有銘柄数")
    ax.set_ylim(0, 6)
    ax.legend(fontsize=7)

    # Clenowスコア上位5銘柄の推移
    ax = axes[3]
    top5 = latest_scores.head(5).index.tolist()
    for sym in top5:
        if sym in clenow_universe.columns:
            s = clenow_universe[sym]
            ax.plot(s.index, s.values, linewidth=1.0, label=sym)
    ax.set_ylabel("Clenowスコア")
    ax.set_title("上位銘柄のClenowスコア推移")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"\n  PNG: {png_path}")
    print("\n完了。")


if __name__ == "__main__":
    main()
