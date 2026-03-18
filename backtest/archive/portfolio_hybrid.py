"""
backtest/portfolio_hybrid.py
ポートフォリオ統合テスト（3シナリオ比較）

【比較シナリオ】
  A. 均等ドル配分 + フジコ法のみ    （旧ベースライン）
  B. 均等ドル配分 + ハイブリッド戦略 （前回実装）
  C. ボラティリティ調整 + ハイブリッド（ナラン第4/6章理論実装）

【ボラティリティ調整サイジング（ナラン理論準拠）】
  target_risk_per_slot = vol_target / max_positions
  alloc = target_risk_per_slot × capital / stock_20day_vol
  上限: capital × 25%

【出力】
  - コンソール: 3シナリオ比較テーブル（CAGR・年率含む）
  - PNG: data/portfolio_hybrid.png

【実行方法】
  python -m backtest.portfolio_hybrid
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
VOL_TARGET        = 0.02    # ポートフォリオ日次変動目標 2%
MAX_SINGLE_WEIGHT = 0.25    # 1銘柄最大ウェイト 25%
VOL_WINDOW        = 20      # ボラティリティ計算期間（営業日）


def build_strategies(universe: dict, rsr_universe) -> tuple[dict, dict]:
    """
    フジコ法のみ辞書 と ハイブリッド辞書 を構築して返す。
    """
    from data.dynamic_selection_loader import load_sym_to_strategy  # noqa
    pass


def main() -> None:
    print("=" * 72)
    print("  ポートフォリオ統合テスト — 3シナリオ比較")
    print("  A: 均等配分+フジコ法  B: 均等配分+ハイブリッド  C: Vol調整+ハイブリッド")
    print(f"  期間: {START} 〜 {END}  /  初期資本: ¥{CAPITAL:,}")
    print("=" * 72)

    # ---- 1. Phase1 銘柄リストを読み込む ----
    csv_path = "data/dynamic_selection.csv"
    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} が見つかりません。dynamic_selection.py を先に実行してください。")
        sys.exit(1)

    df_sel    = pd.read_csv(csv_path, encoding="utf-8-sig")
    df_phase1 = df_sel[df_sel["phase1"]].copy()
    phase1_tickers = {r["symbol"]: r["sector"] for _, r in df_phase1.iterrows()}
    sym_to_strategy = dict(zip(df_phase1["symbol"], df_phase1["strategy"]))

    print(f"\n[1/5] Phase1 達成銘柄: {len(phase1_tickers)} 銘柄")

    # ---- 2. データ取得 ----
    print("\n[2/5] データ取得中...")
    universe = download_universe(phase1_tickers, start=START, end=END)
    print(f"  取得完了: {len(universe)} 銘柄")

    # ---- 3. RSR 計算 ----
    print("\n[3/5] RSR 計算中...")
    universe_prices = {sym: info["df"]["Close"] for sym, info in universe.items()}
    rsr_universe    = calc_universe_rsr(universe_prices)

    # ---- 4. 戦略辞書の構築 ----
    print("\n[4/5] 戦略辞書構築中...")
    cost = TradeCost()

    strat_fujiko:  dict = {}
    strat_hybrid:  dict = {}

    for sym in universe.keys():
        rsr_s = rsr_universe[sym] if sym in rsr_universe.columns else None

        # フジコ法のみ（シナリオA用）
        strat_fujiko[sym] = FujikoStrategy(rsr_series=rsr_s, **FUJIKO_PARAMS)

        # ハイブリッド（シナリオB/C用）
        label = sym_to_strategy.get(sym, "フジコ法")
        if "平均回帰" in label:
            strat_hybrid[sym] = MeanReversionStrategy(**MR_PARAMS)
        else:
            strat_hybrid[sym] = FujikoStrategy(rsr_series=rsr_s, **FUJIKO_PARAMS)

    fj_cnt = sum(1 for s in strat_hybrid.values() if isinstance(s, FujikoStrategy))
    mr_cnt = sum(1 for s in strat_hybrid.values() if isinstance(s, MeanReversionStrategy))
    print(f"  ハイブリッド構成: フジコ法={fj_cnt}銘柄 / 平均回帰={mr_cnt}銘柄")

    # ---- 5. 3シナリオ実行 ----
    print("\n[5/5] PortfolioEngine 実行中...")

    common_kwargs = dict(
        universe      = universe,
        capital       = CAPITAL,
        max_dd_limit  = 0.15,
        min_sectors   = 3,
        max_positions = 6,
        cost          = cost,
    )

    print("  ▶ シナリオA: 均等配分 + フジコ法のみ...")
    # シナリオAは戦略辞書を新規生成（状態リセット）
    strat_a = {sym: FujikoStrategy(
                   rsr_series=rsr_universe[sym] if sym in rsr_universe.columns else None,
                   **FUJIKO_PARAMS)
               for sym in universe}
    res_a = PortfolioEngine(strategy=strat_a, **common_kwargs).run()

    print("  ▶ シナリオB: 均等配分 + ハイブリッド...")
    strat_b = {}
    for sym in universe:
        rsr_s = rsr_universe[sym] if sym in rsr_universe.columns else None
        label = sym_to_strategy.get(sym, "フジコ法")
        strat_b[sym] = (MeanReversionStrategy(**MR_PARAMS) if "平均回帰" in label
                        else FujikoStrategy(rsr_series=rsr_s, **FUJIKO_PARAMS))
    res_b = PortfolioEngine(strategy=strat_b, **common_kwargs).run()

    print("  ▶ シナリオC: ボラティリティ調整 + ハイブリッド...")
    strat_c = {}
    for sym in universe:
        rsr_s = rsr_universe[sym] if sym in rsr_universe.columns else None
        label = sym_to_strategy.get(sym, "フジコ法")
        strat_c[sym] = (MeanReversionStrategy(**MR_PARAMS) if "平均回帰" in label
                        else FujikoStrategy(rsr_series=rsr_s, **FUJIKO_PARAMS))
    res_c = PortfolioEngine(
        strategy          = strat_c,
        vol_target        = VOL_TARGET,
        max_single_weight = MAX_SINGLE_WEIGHT,
        vol_window        = VOL_WINDOW,
        **common_kwargs,
    ).run()

    # ---- 比較テーブル ----
    print("\n\n" + "=" * 72)
    print("  3シナリオ比較結果")
    print("=" * 72)

    scenarios = {
        "A: 均等+フジコ":    res_a,
        "B: 均等+ハイブリッド": res_b,
        "C: Vol調整+ハイブリッド": res_c,
    }

    header = f"  {'指標':<18}"
    for label in scenarios:
        header += f" {label:>22}"
    print(header)
    print("  " + "-" * (18 + 23 * len(scenarios)))

    metrics = [
        ("総リターン",   lambda r: f"{r.total_return*100:>+8.2f}%"),
        ("年率CAGR",     lambda r: f"{r.cagr*100:>+8.2f}%"),
        ("シャープ比",   lambda r: f"{r.sharpe_ratio:>10.3f}"),
        ("最大DD",       lambda r: f"{r.max_drawdown*100:>+8.2f}%"),
        ("決済回数",     lambda r: f"{r.n_trades:>10}回"),
        ("勝率",         lambda r: f"{r.win_rate*100:>9.1f}%"),
        ("CB発動",       lambda r: f"{r.n_circuit_breaker:>10}回"),
        ("平均保有銘柄", lambda r: f"{r.n_positions.mean():>8.1f}銘柄"),
        ("最終資産",     lambda r: f"¥{r.equity_curve.iloc[-1]:>12,.0f}"),
        ("Phase1クリア", lambda r: "  YES ✓" if (r.sharpe_ratio>0.5 and abs(r.max_drawdown)<=0.20) else "  NO  ✗"),
    ]

    for label, fn in metrics:
        row = f"  {label:<18}"
        for res in scenarios.values():
            row += f" {fn(res):>22}"
        print(row)

    # ---- 改善幅サマリー ----
    print("\n  === 改善幅（A基準） ===")
    for label, res in list(scenarios.items())[1:]:
        cagr_diff  = (res.cagr - res_a.cagr) * 100
        shr_diff   = res.sharpe_ratio - res_a.sharpe_ratio
        dd_diff    = (res.max_drawdown - res_a.max_drawdown) * 100
        print(f"  {label}: CAGR {cagr_diff:>+6.2f}%pt  Sharpe {shr_diff:>+6.3f}  MaxDD {dd_diff:>+5.2f}%pt")

    # ---- 個別サマリー ----
    for label, res in scenarios.items():
        print(f"\n--- {label} ---")
        res.summary()

    # ---- グラフ ----
    os.makedirs("data", exist_ok=True)
    png_path = "data/portfolio_hybrid.png"

    fig, axes = plt.subplots(3, 1, figsize=(15, 13), sharex=True)
    fig.suptitle(
        f"ポートフォリオ統合テスト — 3シナリオ比較（2018-2024）\n"
        f"Phase1達成{len(universe)}銘柄 / 初期資本¥{CAPITAL//10000}万円 / max6銘柄",
        fontsize=12,
    )

    colors = {
        "A: 均等+フジコ":        "gray",
        "B: 均等+ハイブリッド":    "steelblue",
        "C: Vol調整+ハイブリッド": "tomato",
    }
    results = scenarios

    # 上段: 資産推移
    ax = axes[0]
    for label, res in results.items():
        curve = res.equity_curve / 10_000
        ax.plot(
            curve.index, curve.values,
            color=colors[label], linewidth=1.8,
            label=f"{label}  CAGR={res.cagr*100:>+.1f}%  Shr={res.sharpe_ratio:.2f}"
        )
    ax.axhline(CAPITAL / 10_000, color="black", linestyle="--", linewidth=0.8, label="元本")
    ax.set_ylabel("資産（万円）")
    ax.set_title("資産推移")
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"¥{x:.0f}万"))

    # 中段: ドローダウン
    ax = axes[1]
    for label, res in results.items():
        ax.fill_between(
            res.dd_series.index, res.dd_series.values * 100, 0,
            alpha=0.3, color=colors[label], label=f"{label} DD"
        )
        ax.plot(
            res.dd_series.index, res.dd_series.values * 100,
            color=colors[label], linewidth=0.8, alpha=0.7,
        )
    ax.axhline(-15, color="darkred", linestyle="--", linewidth=1.2, label="CB閾値 -15%")
    ax.set_ylabel("ドローダウン（%）")
    ax.legend(fontsize=9)

    # 下段: 保有銘柄数比較
    ax = axes[2]
    for label, res in results.items():
        ax.step(
            res.n_positions.index, res.n_positions.values,
            where="post", color=colors[label], linewidth=1.3, label=label,
        )
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
