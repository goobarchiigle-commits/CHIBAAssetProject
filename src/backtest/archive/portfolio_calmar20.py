"""
backtest/portfolio_calmar20.py
Calmar 2.0 最終突破 — IDMベースの微調整シリーズ

【前回結果】
  IDM: 27銘柄/max3/SEPA≥6/RSR≥75  → Calmar=1.993（あと0.007）

【今回の微調整候補（過学習を避けた原則的アプローチ）】
  A. IDM_base   : 前回ベスト（再現確認）
  B. IDM_clnw   : IDM + Clenowスコアソート（原則: より安定したトレンドを優先）
  C. IDM_rsr73  : IDM + min_rsr=73（エントリー機会をわずかに拡大）
  D. IDM_sec1   : IDM + min_sectors=1（セクター制約を緩和）
  E. IDM_clnw+  : IDM + Clenow + RSR≥73（複合）

【実行方法】
  python -m backtest.portfolio_calmar20
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

MR_PARAMS = dict(
    rsi_period=5, rsi_entry=25.0, rsi_exit=65.0,
    ma_long=200, stop_loss_pct=0.07, max_hold_days=10, knife_threshold=0.15,
)
VOL_TARGET        = 0.02
MAX_SINGLE_WEIGHT = 0.25
VOL_WINDOW        = 20
CLENOW_WINDOW     = 90

SECTOR_STRATEGY: dict[str, str] = {
    "海運":"fujiko","機械":"fujiko","電機精密":"fujiko","商社":"fujiko",
    "電機":"fujiko","ゲーム":"fujiko","レジャー":"fujiko","食品":"fujiko",
    "ガス":"mean_rev","鉄鋼":"mean_rev","銀行":"mean_rev","保険":"mean_rev",
    "輸送機器":"mean_rev","化学":"mean_rev","小売":"mean_rev",
    "サービス":"dynamic","医薬品":"dynamic","不動産":"dynamic",
    "情報通信":"dynamic","陸運":"dynamic",
}

# (label, min_sharpe, max_dd_filter, max_pos, min_sepa, min_rsr, min_sectors, use_clenow)
SCENARIOS = [
    ("A_IDM_base",  0.30, 0.30, 3, 6, 75.0, 2, False),  # 前回ベスト再現
    ("B_IDM_clnw",  0.30, 0.30, 3, 6, 75.0, 2, True ),  # Clenowソート追加
    ("C_IDM_rsr73", 0.30, 0.30, 3, 6, 73.0, 2, False),  # RSR下限を73に緩和
    ("D_IDM_sec1",  0.30, 0.30, 3, 6, 75.0, 1, False),  # min_sectors=1
    ("E_IDM_clnw+", 0.30, 0.30, 3, 6, 73.0, 2, True ),  # Clenow + RSR73
]


def make_strategy(sym: str, sector: str, strategy_label: str,
                  rsr_s, min_sepa: int, min_rsr: float) -> object:
    rule = SECTOR_STRATEGY.get(sector, "dynamic")
    if rule == "mean_rev" or (rule == "dynamic" and "平均回帰" in strategy_label):
        return MeanReversionStrategy(**MR_PARAMS)
    return FujikoStrategy(
        rsr_series=rsr_s,
        min_sepa=min_sepa, min_rsr=min_rsr,
        mom_period=21, turtle_entry=20, turtle_exit=10, use_turtle_entry=True,
    )


def main() -> None:
    print("=" * 72)
    print("  Calmar 2.0 最終突破 — IDMベース微調整シリーズ")
    print("  IDM=1.993 → 残り0.007を詰める原則的アプローチ")
    print(f"  期間: {START} 〜 {END}  /  初期資本: ¥{CAPITAL:,}")
    print("=" * 72)

    # ---- 1. 銘柄リスト読み込み ----
    df_sel = pd.read_csv("data/dynamic_selection.csv", encoding="utf-8-sig")
    sym_to_strategy = dict(zip(df_sel["symbol"], df_sel["strategy"]))

    # ---- 2. データ取得（SR>0.30 / MaxDD<30%）----
    broadest = df_sel[(df_sel["sharpe"] > 0.30) & (df_sel["maxdd"].abs() < 0.30)]
    broad_tickers = {r["symbol"]: r["sector"] for _, r in broadest.iterrows()}
    print(f"\n[1/4] データ取得中（{len(broad_tickers)}銘柄）...")
    universe_all = download_universe(broad_tickers, start=START, end=END)
    print(f"  取得完了: {len(universe_all)} 銘柄")

    # ---- 3. RSR + Clenow 計算 ----
    print("\n[2/4] RSR / Clenowスコア計算中...")
    universe_prices = {sym: info["df"]["Close"] for sym, info in universe_all.items()}
    rsr_universe    = calc_universe_rsr(universe_prices)
    print(f"  RSR完了: {rsr_universe.shape[1]} 銘柄")
    clenow_universe = calc_universe_clenow(universe_prices, window=CLENOW_WINDOW)
    print(f"  Clenow完了: {clenow_universe.shape[1]} 銘柄")

    # ---- 4. バックテスト ----
    print("\n[3/4] バックテスト実行中...")
    cost = TradeCost()
    results = {}

    for label, min_shr, max_dd_f, max_pos, min_sepa, min_rsr, min_sec, use_clnw in SCENARIOS:
        mask = (df_sel["sharpe"] > min_shr) & (df_sel["maxdd"].abs() < max_dd_f)
        tier_syms = set(df_sel[mask]["symbol"].tolist())
        universe_t = {sym: info for sym, info in universe_all.items() if sym in tier_syms}

        if not universe_t:
            continue

        strat_dict = {}
        for sym, info in universe_t.items():
            rsr_s = rsr_universe[sym] if sym in rsr_universe.columns else None
            lbl_  = sym_to_strategy.get(sym, "フジコ法")
            strat_dict[sym] = make_strategy(
                sym, info["sector"], lbl_, rsr_s, min_sepa, min_rsr
            )

        # Clenowスコア（ソート用）
        cols = [c for c in clenow_universe.columns if c in tier_syms]
        clenow_df = clenow_universe[cols] if use_clnw else None

        fj_n = sum(1 for s in strat_dict.values() if isinstance(s, FujikoStrategy))
        mr_n = sum(1 for s in strat_dict.values() if isinstance(s, MeanReversionStrategy))
        print(f"  ▶ {label} ({len(universe_t)}銘柄: FJ={fj_n} MR={mr_n} "
              f"min_sec={min_sec} RSR≥{min_rsr:.0f}"
              f"{' +Clenow' if use_clnw else ''})...")

        engine = PortfolioEngine(
            universe          = universe_t,
            strategy          = strat_dict,
            capital           = CAPITAL,
            max_dd_limit      = 0.15,
            min_sectors       = min_sec,
            max_positions     = max_pos,
            cost              = cost,
            vol_target        = VOL_TARGET,
            max_single_weight = MAX_SINGLE_WEIGHT,
            vol_window        = VOL_WINDOW,
            use_idm           = True,       # 全シナリオIDM有効
            use_dd_scalar     = False,
            clenow_scores     = clenow_df,
        )
        results[label] = engine.run()

    # ---- 比較テーブル ----
    def calmar(r):
        dd = abs(r.max_drawdown)
        return r.cagr / dd if dd > 0 else float("inf")

    print("\n\n" + "=" * 85)
    print("  Calmar 2.0 最終突破 — 微調整結果")
    print("=" * 85)

    header = f"  {'指標':<14}"
    for lbl in results:
        header += f" {lbl:>18}"
    print(header)
    print("  " + "-" * (14 + 19 * len(results)))

    metrics = [
        ("年率CAGR",     lambda r: f"{r.cagr*100:>+7.2f}%"),
        ("最大DD",       lambda r: f"{r.max_drawdown*100:>+7.2f}%"),
        ("Calmar比",     lambda r: f"{calmar(r):>9.3f}"),
        ("シャープ比",   lambda r: f"{r.sharpe_ratio:>9.3f}"),
        ("総リターン",   lambda r: f"{r.total_return*100:>+7.2f}%"),
        ("決済回数",     lambda r: f"{r.n_trades:>9}回"),
        ("勝率",         lambda r: f"{r.win_rate*100:>8.1f}%"),
        ("CB発動",       lambda r: f"{r.n_circuit_breaker:>9}回"),
        ("平均保有銘柄", lambda r: f"{r.n_positions.mean():>7.2f}銘柄"),
        ("最終資産",     lambda r: f"¥{r.equity_curve.iloc[-1]:>10,.0f}"),
        ("Calmar≥2.0",  lambda r: "★ 達成 ★" if calmar(r) >= 2.0 else f"あと{2.0-calmar(r):.4f}"),
    ]

    for mlabel, fn in metrics:
        row = f"  {mlabel:<14}"
        for res in results.values():
            row += f" {fn(res):>18}"
        print(row)

    # ---- Calmarランキング ----
    print("\n  === Calmar比ランキング ===")
    ranked = sorted(results.items(), key=lambda x: calmar(x[1]), reverse=True)
    for rank, (lbl, res) in enumerate(ranked, 1):
        c = calmar(res)
        mark = " ★ Calmar 2.0 達成！" if c >= 2.0 else f" (あと{2.0-c:.4f})"
        print(f"  {rank}. {lbl}")
        print(f"     Calmar={c:.4f}  CAGR={res.cagr*100:+.2f}%  "
              f"MaxDD={res.max_drawdown*100:.2f}%  Sharpe={res.sharpe_ratio:.3f}"
              f"  平均保有={res.n_positions.mean():.2f}銘柄{mark}")

    # ---- 年次内訳（上位2シナリオ）----
    print("\n\n" + "=" * 72)
    print("  年次パフォーマンス内訳")
    print("=" * 72)

    years = list(range(2018, 2025))
    top2 = [lbl for lbl, _ in ranked[:2]]
    for lbl in top2:
        res = results[lbl]
        print(f"\n  【{lbl}  Calmar={calmar(res):.4f}】")
        print(f"  {'年':<6} {'年率リターン':>12} {'年間最大DD':>12} {'Calmar':>8}")
        print(f"  {'-'*42}")
        eq = res.equity_curve
        dd = res.dd_series
        for yr in years:
            yr_eq = eq[eq.index.year == yr]
            yr_dd = dd[dd.index.year == yr]
            if yr_eq.empty:
                continue
            ret = yr_eq.iloc[-1] / yr_eq.iloc[0] - 1
            max_dd_yr = float(yr_dd.min())
            calmar_yr = ret / abs(max_dd_yr) if max_dd_yr < 0 else float("inf")
            calmar_str = f"{calmar_yr:>8.2f}" if calmar_yr != float("inf") else "     ∞"
            print(f"  {yr:<6} {ret*100:>+11.2f}%  {max_dd_yr*100:>+11.2f}%  {calmar_str}")
        total_ret = eq.iloc[-1] / eq.iloc[0] - 1
        total_dd  = float(dd.min())
        print(f"  {'全期間':<6} {total_ret*100:>+11.2f}%  {total_dd*100:>+11.2f}%  {calmar(res):>8.2f}")

    # ---- グラフ ----
    os.makedirs("data", exist_ok=True)
    png_path = "data/portfolio_calmar20.png"

    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    fig.suptitle(
        "Calmar 2.0 最終突破 — IDMベース微調整（2018–2024）\n"
        "A=base / B=+Clenow / C=RSR73 / D=sec1 / E=Clenow+RSR73",
        fontsize=11,
    )
    cmap   = plt.cm.tab10
    colors = {lbl: cmap(i) for i, lbl in enumerate(results)}

    ax = axes[0]
    for lbl, res in results.items():
        c = calmar(res)
        lw = 2.5 if c >= 2.0 else 1.5
        ax.plot(res.equity_curve.index, res.equity_curve.values / 10_000,
                color=colors[lbl], linewidth=lw,
                label=f"{lbl}  CAGR={res.cagr*100:>+.1f}%  Calmar={c:.3f}"
                      + (" ★" if c >= 2.0 else ""))
    ax.axhline(CAPITAL / 10_000, color="black", linestyle="--", linewidth=0.8, label="元本")
    ax.set_ylabel("資産（万円）")
    ax.set_title("資産推移")
    ax.legend(fontsize=7.5)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"¥{x:.0f}万"))

    ax = axes[1]
    for lbl, res in results.items():
        ax.fill_between(res.dd_series.index, res.dd_series.values*100, 0,
                        alpha=0.20, color=colors[lbl])
        ax.plot(res.dd_series.index, res.dd_series.values*100,
                color=colors[lbl], linewidth=0.9,
                label=f"{lbl} MaxDD={res.max_drawdown*100:.2f}%")
    ax.axhline(-15, color="darkred", linestyle="--", linewidth=1.2, label="CB閾値")
    ax.set_ylabel("ドローダウン（%）")
    ax.legend(fontsize=7.5)

    ax = axes[2]
    for lbl, res in results.items():
        ax.step(res.n_positions.index, res.n_positions.values,
                where="post", color=colors[lbl], linewidth=1.3,
                label=f"{lbl} 平均={res.n_positions.mean():.2f}")
    ax.set_ylabel("保有銘柄数")
    ax.set_ylim(0, 6)
    ax.legend(fontsize=7.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"\n  PNG: {png_path}")
    print("\n完了。")


if __name__ == "__main__":
    main()
