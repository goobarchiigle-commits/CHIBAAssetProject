"""
backtest/portfolio_books.py
書籍知見（Carver/Thorp/Clenow/López de Prado）統合バックテスト

【実装した改善点】
  1. IDM（Carver）: 実保有銘柄数ベースでポジションを拡大
     - 1銘柄: ×1.00、2銘柄: ×1.20、3銘柄: ×1.48
  2. DDスケーラー（Carver）: DD > max_dd×0.5 でポジションを段階縮小
     - DD=7.5%: フルサイズ → DD=15%: エントリー禁止（線形補間）
  3. 条件緩和（Clenow/López de Prado）:
     - 宇宙拡大: SR>0.3→SR>0.2、MaxDD<30%→<35%（27→約40銘柄）
     - SEPA条件: 6→5（1段緩和）
     - RSR下限: 75→70
     - max_positions: 3→4（機会損失削減）

【比較シナリオ】
  Base   : 27銘柄/max3/SEPA≥6/RSR≥75 （ベースライン Calmar=1.713）
  IDM    : Base + IDM
  DDS    : Base + DDスケーラー
  Relax  : 宇宙拡大+条件緩和（SEPA≥5/RSR≥70/max4）
  Best   : Relax + IDM + DDスケーラー（全部入り）

【実行方法】
  python -m backtest.portfolio_books
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

# ---- 戦略パラメータ ----
MR_PARAMS = dict(
    rsi_period=5, rsi_entry=25.0, rsi_exit=65.0,
    ma_long=200, stop_loss_pct=0.07, max_hold_days=10, knife_threshold=0.15,
)
VOL_TARGET        = 0.02
MAX_SINGLE_WEIGHT = 0.25
VOL_WINDOW        = 20

SECTOR_STRATEGY: dict[str, str] = {
    "海運":"fujiko","機械":"fujiko","電機精密":"fujiko","商社":"fujiko",
    "電機":"fujiko","ゲーム":"fujiko","レジャー":"fujiko","食品":"fujiko",
    "ガス":"mean_rev","鉄鋼":"mean_rev","銀行":"mean_rev","保険":"mean_rev",
    "輸送機器":"mean_rev","化学":"mean_rev","小売":"mean_rev",
    "サービス":"dynamic","医薬品":"dynamic","不動産":"dynamic",
    "情報通信":"dynamic","陸運":"dynamic",
}

# ---- シナリオ定義 ----
# (label, min_sharpe, max_dd_filter, max_pos, min_sepa, min_rsr, use_idm, use_dd_scalar)
SCENARIOS = [
    ("Base:   27銘柄/max3/SEPA≥6/RSR≥75",         0.30, 0.30, 3, 6, 75.0, False, False),
    ("IDM:    27銘柄/max3/SEPA≥6+IDM",            0.30, 0.30, 3, 6, 75.0, True,  False),
    ("DDS:    27銘柄/max3/SEPA≥6+DDscaler",       0.30, 0.30, 3, 6, 75.0, False, True ),
    ("Relax:  ~40銘柄/max4/SEPA≥5/RSR≥70",        0.20, 0.35, 4, 5, 70.0, False, False),
    ("Best:   ~40銘柄/max4/SEPA≥5+IDM+DDS",       0.20, 0.35, 4, 5, 70.0, True,  True ),
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
    print("=" * 76)
    print("  書籍知見統合バックテスト（Carver/Thorp/Clenow/López de Prado）")
    print("  改善: IDM分散乗数 / DDスケーラー / エントリー条件緩和")
    print(f"  期間: {START} 〜 {END}  /  初期資本: ¥{CAPITAL:,}")
    print("=" * 76)

    # ---- 1. 銘柄リスト読み込み ----
    df_sel = pd.read_csv("data/dynamic_selection.csv", encoding="utf-8-sig")
    sym_to_strategy = dict(zip(df_sel["symbol"], df_sel["strategy"]))

    # ---- 2. 最広宇宙のデータ一括取得（SR>0.2 まで拡大）----
    broadest = df_sel[(df_sel["sharpe"] > 0.20) & (df_sel["maxdd"].abs() < 0.35)]
    broad_tickers = {r["symbol"]: r["sector"] for _, r in broadest.iterrows()}
    print(f"\n[1/3] データ取得中（拡大宇宙: {len(broad_tickers)}銘柄）...")
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

    for label, min_shr, max_dd_f, max_pos, min_sepa, min_rsr, use_idm, use_dds in SCENARIOS:
        mask      = (df_sel["sharpe"] > min_shr) & (df_sel["maxdd"].abs() < max_dd_f)
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

        fj_n = sum(1 for s in strat_dict.values() if isinstance(s, FujikoStrategy))
        mr_n = sum(1 for s in strat_dict.values() if isinstance(s, MeanReversionStrategy))
        opts = []
        if use_idm: opts.append("IDM")
        if use_dds: opts.append("DDscaler")
        opt_str = "+".join(opts) if opts else "なし"
        print(f"  ▶ {label} ({len(universe_t)}銘柄: FJ={fj_n} MR={mr_n} [{opt_str}])...")

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
            use_idm           = use_idm,
            use_dd_scalar     = use_dds,
        )
        results[label] = engine.run()

    # ---- 比較テーブル ----
    def calmar(r):
        dd = abs(r.max_drawdown)
        return r.cagr / dd if dd > 0 else float("inf")

    print("\n\n" + "=" * 90)
    print("  書籍知見統合バックテスト — 比較結果")
    print("=" * 90)

    header = f"  {'指標':<14}"
    for lbl in results:
        header += f" {lbl[:20]:>20}"
    print(header)
    print("  " + "-" * (14 + 21 * len(results)))

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
        ("Calmar≥2.0",  lambda r: "★ 達成 ★" if calmar(r) >= 2.0 else f"あと{2.0-calmar(r):.3f}"),
    ]

    for mlabel, fn in metrics:
        row = f"  {mlabel:<14}"
        for res in results.values():
            row += f" {fn(res):>20}"
        print(row)

    # ---- Calmarランキング ----
    print("\n  === Calmar比ランキング ===")
    ranked = sorted(results.items(), key=lambda x: calmar(x[1]), reverse=True)
    for rank, (lbl, res) in enumerate(ranked, 1):
        c = calmar(res)
        mark = " ★ Calmar 2.0 達成！" if c >= 2.0 else f" (あと{2.0-c:.3f})"
        print(f"  {rank}. {lbl}")
        print(f"     Calmar={c:.3f}  CAGR={res.cagr*100:+.2f}%  "
              f"MaxDD={res.max_drawdown*100:.2f}%  Sharpe={res.sharpe_ratio:.3f}"
              f"  平均保有={res.n_positions.mean():.2f}銘柄{mark}")

    # ---- 個別サマリー ----
    for lbl, res in results.items():
        print(f"\n--- {lbl} ---")
        res.summary()

    # ---- 年次パフォーマンス内訳 ----
    print("\n\n" + "=" * 90)
    print("  年次パフォーマンス内訳（年率リターン / 年間最大DD）")
    print("=" * 90)

    years = list(range(2018, 2025))
    for lbl, res in results.items():
        print(f"\n  【{lbl}】")
        print(f"  {'年':<6} {'年率リターン':>12} {'年間最大DD':>12} {'Calmar':>8}")
        print(f"  {'-'*42}")
        eq = res.equity_curve
        dd = res.dd_series
        for yr in years:
            yr_eq = eq[eq.index.year == yr]
            yr_dd = dd[dd.index.year == yr]
            if yr_eq.empty:
                continue
            # 年率リターン: 年初→年末（取引日ベース）
            ret = yr_eq.iloc[-1] / yr_eq.iloc[0] - 1
            # 年間最大DD（その年の中での最大ドローダウン）
            max_dd_yr = float(yr_dd.min())
            calmar_yr = ret / abs(max_dd_yr) if max_dd_yr < 0 else float("inf")
            calmar_str = f"{calmar_yr:>8.2f}" if calmar_yr != float("inf") else "     ∞"
            print(f"  {yr:<6} {ret*100:>+11.2f}%  {max_dd_yr*100:>+11.2f}%  {calmar_str}")
        # 全期間合計
        total_ret = eq.iloc[-1] / eq.iloc[0] - 1
        total_dd  = float(dd.min())
        print(f"  {'全期間':<6} {total_ret*100:>+11.2f}%  {total_dd*100:>+11.2f}%  {calmar(res):>8.2f}")

    # ---- グラフ ----
    os.makedirs("data", exist_ok=True)
    png_path = "data/portfolio_books.png"

    fig, axes = plt.subplots(3, 1, figsize=(16, 13), sharex=True)
    fig.suptitle(
        "書籍知見統合バックテスト（2018–2024）\n"
        "Carver: IDM分散乗数 + DDスケーラー / Clenow: 宇宙拡大 / López de Prado: 条件緩和",
        fontsize=11,
    )
    cmap   = plt.cm.tab10
    colors = {lbl: cmap(i) for i, lbl in enumerate(results)}

    ax = axes[0]
    for lbl, res in results.items():
        c = calmar(res)
        lw = 2.5 if "Best" in lbl else 1.5
        ax.plot(res.equity_curve.index, res.equity_curve.values / 10_000,
                color=colors[lbl], linewidth=lw,
                label=f"{lbl[:22]}  CAGR={res.cagr*100:>+.1f}%  Calmar={c:.2f}")
    ax.axhline(CAPITAL / 10_000, color="black", linestyle="--", linewidth=0.8, label="元本")
    ax.set_ylabel("資産（万円）")
    ax.set_title("資産推移")
    ax.legend(fontsize=7)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"¥{x:.0f}万"))

    ax = axes[1]
    for lbl, res in results.items():
        ax.fill_between(res.dd_series.index, res.dd_series.values*100, 0,
                        alpha=0.20, color=colors[lbl])
        ax.plot(res.dd_series.index, res.dd_series.values*100,
                color=colors[lbl], linewidth=0.9,
                label=f"{lbl[:22]} MaxDD={res.max_drawdown*100:.1f}%")
    ax.axhline(-7.5, color="orange", linestyle=":", linewidth=1.0, label="DDscaler発動(-7.5%)")
    ax.axhline(-15, color="darkred", linestyle="--", linewidth=1.2, label="CB閾値(-15%)")
    ax.set_ylabel("ドローダウン（%）")
    ax.legend(fontsize=7)

    ax = axes[2]
    for lbl, res in results.items():
        ax.step(res.n_positions.index, res.n_positions.values,
                where="post", color=colors[lbl], linewidth=1.3,
                label=f"{lbl[:22]} 平均={res.n_positions.mean():.2f}")
    ax.set_ylabel("保有銘柄数")
    ax.set_ylim(0, 7)
    ax.legend(fontsize=7)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"\n  PNG: {png_path}")
    print("\n完了。")


if __name__ == "__main__":
    main()
