"""
backtest/portfolio_full.py
フルユニバース（77銘柄）ポートフォリオバックテスト

【目的】
  Phase1銘柄14銘柄に絞っていた前回から、TOPIX100相当77銘柄全体に拡張する。
  シグナル機会を増やし資本効率を改善する。

【比較シナリオ】
  C. 14銘柄  + Vol調整 + ハイブリッド  （前回ベスト: CAGR +5.32%）
  D. 77銘柄  + 均等配分 + ハイブリッド
  E. 77銘柄  + Vol調整 + ハイブリッド  （本命）

【出力】
  - コンソール: C/D/E 比較テーブル（CAGR・平均保有銘柄数）
  - PNG: data/portfolio_full.png

【実行方法】
  python -m backtest.portfolio_full
"""

from __future__ import annotations

import os
import sys
import warnings

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

from backtest.universe_builder        import TOPIX100_TICKERS, download_universe
from backtest.rsr                     import calc_universe_rsr
from backtest.fujiko_strategy         import FujikoStrategy
from backtest.mean_reversion_strategy import MeanReversionStrategy
from backtest.portfolio_engine        import PortfolioEngine
from backtest.engine                  import TradeCost

START   = "2018-01-01"
END     = "2024-12-31"
CAPITAL = 2_000_000

FUJIKO_PARAMS = dict(
    min_sepa         = 6,
    min_rsr          = 75.0,
    mom_period       = 21,
    turtle_entry     = 20,
    turtle_exit      = 10,
    use_turtle_entry = True,
)
MR_PARAMS = dict(
    rsi_period      = 5,
    rsi_entry       = 25.0,
    rsi_exit        = 65.0,
    ma_long         = 200,
    stop_loss_pct   = 0.07,
    max_hold_days   = 10,
    knife_threshold = 0.15,
)

# ボラティリティ調整サイジング設定
VOL_TARGET        = 0.02
MAX_SINGLE_WEIGHT = 0.25
VOL_WINDOW        = 20

# セクター別採用戦略（dynamic_selection.py と同一）
SECTOR_STRATEGY: dict[str, str] = {
    "海運":    "fujiko",  "機械":    "fujiko",  "電機精密": "fujiko",
    "商社":    "fujiko",  "電機":    "fujiko",  "ゲーム":   "fujiko",
    "レジャー":"fujiko",  "食品":    "fujiko",
    "ガス":    "mean_rev","鉄鋼":    "mean_rev","銀行":     "mean_rev",
    "保険":    "mean_rev","輸送機器":"mean_rev", "化学":     "mean_rev",
    "小売":    "mean_rev",
    "サービス":"dynamic", "医薬品":  "dynamic", "不動産":   "dynamic",
    "情報通信":"dynamic", "陸運":    "dynamic",
}


def build_strategy_dict(universe: dict, rsr_universe) -> dict:
    """
    全銘柄のセクター別最適戦略辞書を構築する。
    dynamic セクターはフジコ法を基本とし、平均回帰を補完で追加。
    """
    strategies = {}
    for sym, info in universe.items():
        sector = info["sector"]
        rsr_s  = rsr_universe[sym] if sym in rsr_universe.columns else None
        rule   = SECTOR_STRATEGY.get(sector, "dynamic")

        if rule == "mean_rev":
            strategies[sym] = MeanReversionStrategy(**MR_PARAMS)
        else:
            # fujiko も dynamic もフジコ法をベースに（dynamic は両方シグナルを試みる）
            strategies[sym] = FujikoStrategy(rsr_series=rsr_s, **FUJIKO_PARAMS)

    return strategies


def main() -> None:
    print("=" * 72)
    print("  フルユニバース ポートフォリオバックテスト（77銘柄）")
    print("  C: 14銘柄/Vol調整 vs D: 77銘柄/均等 vs E: 77銘柄/Vol調整")
    print(f"  期間: {START} 〜 {END}  /  初期資本: ¥{CAPITAL:,}")
    print("=" * 72)

    # ---- 1. データ取得（77銘柄フル）----
    print("\n[1/4] ユニバース全銘柄データ取得中（77銘柄）...")
    universe_full = download_universe(TOPIX100_TICKERS, start=START, end=END)
    print(f"  取得完了: {len(universe_full)} 銘柄")

    # セクター内訳を表示
    from collections import Counter
    sector_counts = Counter(info["sector"] for info in universe_full.values())
    print("  セクター内訳:")
    for sec, cnt in sorted(sector_counts.items(), key=lambda x: -x[1]):
        rule = SECTOR_STRATEGY.get(sec, "dynamic")
        label = {"fujiko": "フジコ", "mean_rev": "平均回帰", "dynamic": "動的"}[rule]
        print(f"    {sec:<10} {cnt:>3}銘柄  [{label}]")

    # ---- 2. Phase1 14銘柄ユニバース（シナリオC用）----
    print("\n[2/4] Phase1 14銘柄データ読み込み中...")
    csv_path = "data/dynamic_selection.csv"
    df_sel    = pd.read_csv(csv_path, encoding="utf-8-sig")
    df_phase1 = df_sel[df_sel["phase1"]].copy()
    phase1_syms = set(df_phase1["symbol"].tolist())
    universe_14 = {sym: info for sym, info in universe_full.items() if sym in phase1_syms}
    print(f"  Phase1銘柄: {len(universe_14)} 銘柄")

    # ---- 3. RSR 計算 ----
    print("\n[3/4] RSR 計算中（77銘柄）...")
    universe_prices = {sym: info["df"]["Close"] for sym, info in universe_full.items()}
    rsr_universe    = calc_universe_rsr(universe_prices)
    print(f"  RSR計算完了: {rsr_universe.shape[1]} 銘柄 × {rsr_universe.shape[0]} 日")

    # ---- 4. エンジン実行 ----
    print("\n[4/4] PortfolioEngine 実行中...")
    cost = TradeCost()

    common_kwargs = dict(
        capital       = CAPITAL,
        max_dd_limit  = 0.15,
        min_sectors   = 3,
        max_positions = 6,
        cost          = cost,
    )

    # --- シナリオC: 14銘柄 + Vol調整 + ハイブリッド（前回ベスト再現）---
    print("  ▶ C: 14銘柄 + Vol調整 + ハイブリッド...")
    strat_c = {}
    sym_to_strategy = dict(zip(df_phase1["symbol"], df_phase1["strategy"]))
    for sym in universe_14:
        rsr_s = rsr_universe[sym] if sym in rsr_universe.columns else None
        label = sym_to_strategy.get(sym, "フジコ法")
        strat_c[sym] = (MeanReversionStrategy(**MR_PARAMS) if "平均回帰" in label
                        else FujikoStrategy(rsr_series=rsr_s, **FUJIKO_PARAMS))
    res_c = PortfolioEngine(
        universe=universe_14, strategy=strat_c,
        vol_target=VOL_TARGET, max_single_weight=MAX_SINGLE_WEIGHT,
        vol_window=VOL_WINDOW, **common_kwargs,
    ).run()

    # --- シナリオD: 77銘柄 + 均等配分 + ハイブリッド ---
    print("  ▶ D: 77銘柄 + 均等配分 + ハイブリッド...")
    strat_d = build_strategy_dict(universe_full, rsr_universe)
    res_d = PortfolioEngine(
        universe=universe_full, strategy=strat_d,
        **common_kwargs,
    ).run()

    # --- シナリオE: 77銘柄 + Vol調整 + ハイブリッド ---
    print("  ▶ E: 77銘柄 + Vol調整 + ハイブリッド...")
    strat_e = build_strategy_dict(universe_full, rsr_universe)
    res_e = PortfolioEngine(
        universe=universe_full, strategy=strat_e,
        vol_target=VOL_TARGET, max_single_weight=MAX_SINGLE_WEIGHT,
        vol_window=VOL_WINDOW, **common_kwargs,
    ).run()

    # ---- 比較テーブル ----
    scenarios = {
        "C: 14銘柄/Vol調整":  res_c,
        "D: 77銘柄/均等":     res_d,
        "E: 77銘柄/Vol調整":  res_e,
    }

    print("\n\n" + "=" * 72)
    print("  比較結果")
    print("=" * 72)

    header = f"  {'指標':<18}"
    for lbl in scenarios:
        header += f" {lbl:>22}"
    print(header)
    print("  " + "-" * (18 + 23 * len(scenarios)))

    metrics = [
        ("年率CAGR",        lambda r: f"{r.cagr*100:>+8.2f}%"),
        ("総リターン(7年)", lambda r: f"{r.total_return*100:>+8.2f}%"),
        ("シャープ比",      lambda r: f"{r.sharpe_ratio:>10.3f}"),
        ("最大DD",          lambda r: f"{r.max_drawdown*100:>+8.2f}%"),
        ("決済回数",        lambda r: f"{r.n_trades:>10}回"),
        ("勝率",            lambda r: f"{r.win_rate*100:>9.1f}%"),
        ("CB発動",          lambda r: f"{r.n_circuit_breaker:>10}回"),
        ("平均保有銘柄",    lambda r: f"{r.n_positions.mean():>8.2f}銘柄"),
        ("最終資産",        lambda r: f"¥{r.equity_curve.iloc[-1]:>12,.0f}"),
        ("Phase1クリア",    lambda r: "  YES ✓" if (r.sharpe_ratio>0.5 and abs(r.max_drawdown)<=0.20) else "  NO  ✗"),
    ]

    for label, fn in metrics:
        row = f"  {label:<18}"
        for res in scenarios.values():
            row += f" {fn(res):>22}"
        print(row)

    # ---- 改善幅サマリー ----
    print("\n  === 改善幅（C: 14銘柄/Vol調整 基準） ===")
    for lbl, res in list(scenarios.items())[1:]:
        cagr_diff = (res.cagr - res_c.cagr) * 100
        shr_diff  = res.sharpe_ratio - res_c.sharpe_ratio
        dd_diff   = (res.max_drawdown - res_c.max_drawdown) * 100
        pos_diff  = res.n_positions.mean() - res_c.n_positions.mean()
        print(
            f"  {lbl}: CAGR {cagr_diff:>+6.2f}%pt  "
            f"Sharpe {shr_diff:>+6.3f}  "
            f"MaxDD {dd_diff:>+5.2f}%pt  "
            f"平均保有 {pos_diff:>+5.2f}銘柄"
        )

    # ---- 個別サマリー ----
    for lbl, res in scenarios.items():
        print(f"\n--- {lbl} ---")
        res.summary()

    # ---- グラフ ----
    os.makedirs("data", exist_ok=True)
    png_path = "data/portfolio_full.png"

    fig, axes = plt.subplots(3, 1, figsize=(15, 13), sharex=True)
    fig.suptitle(
        "フルユニバース 3シナリオ比較（2018–2024）\n"
        "C: 14銘柄/Vol調整  D: 77銘柄/均等  E: 77銘柄/Vol調整",
        fontsize=12,
    )

    colors = {
        "C: 14銘柄/Vol調整": "gray",
        "D: 77銘柄/均等":    "steelblue",
        "E: 77銘柄/Vol調整": "tomato",
    }

    # 上段: 資産推移
    ax = axes[0]
    for lbl, res in scenarios.items():
        ax.plot(
            res.equity_curve.index,
            res.equity_curve.values / 10_000,
            color=colors[lbl], linewidth=1.8,
            label=f"{lbl}  CAGR={res.cagr*100:>+.1f}%  Shr={res.sharpe_ratio:.2f}",
        )
    ax.axhline(CAPITAL / 10_000, color="black", linestyle="--", linewidth=0.8, label="元本")
    ax.set_ylabel("資産（万円）")
    ax.set_title("資産推移")
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"¥{x:.0f}万"))

    # 中段: ドローダウン
    ax = axes[1]
    for lbl, res in scenarios.items():
        ax.fill_between(
            res.dd_series.index, res.dd_series.values * 100, 0,
            alpha=0.3, color=colors[lbl],
        )
        ax.plot(
            res.dd_series.index, res.dd_series.values * 100,
            color=colors[lbl], linewidth=1.0, label=f"{lbl} DD最大={res.max_drawdown*100:.1f}%",
        )
    ax.axhline(-15, color="darkred", linestyle="--", linewidth=1.2, label="CB閾値 -15%")
    ax.set_ylabel("ドローダウン（%）")
    ax.legend(fontsize=9)

    # 下段: 平均保有銘柄数
    ax = axes[2]
    for lbl, res in scenarios.items():
        ax.step(
            res.n_positions.index, res.n_positions.values,
            where="post", color=colors[lbl], linewidth=1.3,
            label=f"{lbl} 平均={res.n_positions.mean():.2f}銘柄",
        )
    ax.axhline(3, color="gray", linestyle=":", linewidth=0.8, label="目標3銘柄")
    ax.set_ylabel("保有銘柄数")
    ax.set_ylim(0, 8)
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"\n  PNG: {png_path}")
    print("\n完了。")


if __name__ == "__main__":
    main()
