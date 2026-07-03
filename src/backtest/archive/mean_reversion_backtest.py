"""
backtest/mean_reversion_backtest.py
短期平均回帰戦略 vs フジコ法  比較バックテスト

【実行方法】
  python -m backtest.mean_reversion_backtest

【出力】
  - コンソール: 銘柄別×戦略別の比較テーブル + Phase 1 達成率
  - PNG: data/mean_reversion_comparison.png
  - CSV: data/mean_reversion_comparison.csv

【比較条件】
  フジコ法       : SEPA>=6, RSR>=75, タートルズ20/10
  短期平均回帰   : 5日RSI<25 + MA200フィルター + 7%損切り + 10日時間制限
  期間           : 2018-01-01 〜 2024-12-31
  初期資本       : 100万円 / 銘柄
"""

from __future__ import annotations

import sys
import os
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

import numpy as np
import pandas as pd

from backtest.universe_builder        import TOPIX100_TICKERS, download_universe
from backtest.rsr                     import calc_universe_rsr
from backtest.fujiko_strategy         import FujikoStrategy
from backtest.mean_reversion_strategy import MeanReversionStrategy
from backtest.engine                  import BacktestEngine, TradeCost

START   = "2018-01-01"
END     = "2024-12-31"
CAPITAL = 1_000_000

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


def avg_hold_days(trades: list) -> float:
    buys  = [t for t in trades if t.side == "BUY"]
    sells = [t for t in trades if t.side.startswith("SELL")]
    pairs = list(zip(buys, sells[: len(buys)]))
    if not pairs:
        return 0.0
    return float(np.mean([(s.date - b.date).days for b, s in pairs]))


def main() -> None:
    print("=" * 70)
    print("  短期平均回帰戦略 vs フジコ法  比較バックテスト")
    print(f"  期間: {START} 〜 {END}  /  初期資本: ¥{CAPITAL:,} / 銘柄")
    print("=" * 70)

    # ---- 1. データ取得 ----
    print("\n[1/4] ユニバースデータ取得中...")
    universe = download_universe(TOPIX100_TICKERS, start=START, end=END)
    print(f"  取得完了: {len(universe)} 銘柄")

    # ---- 2. RSR 計算（フジコ法用）----
    print("\n[2/4] RSR 計算中...")
    universe_prices = {sym: info["df"]["Close"] for sym, info in universe.items()}
    rsr_universe    = calc_universe_rsr(universe_prices)
    print(f"  RSR 計算完了: {rsr_universe.shape[0]} 日 × {rsr_universe.shape[1]} 銘柄")

    # ---- 3. バックテスト ----
    print("\n[3/4] 銘柄別バックテスト実行中...")
    cost = TradeCost()
    rows = []

    for sym, info in universe.items():
        df     = info["df"]
        sector = info["sector"]

        # フジコ法
        rsr_s  = rsr_universe[sym] if sym in rsr_universe.columns else None
        fujiko = FujikoStrategy(rsr_series=rsr_s, **FUJIKO_PARAMS)
        res_f  = BacktestEngine(
            prices=df, strategy=fujiko, capital=CAPITAL, cost=cost, symbol=sym
        ).run()

        # 短期平均回帰
        mr    = MeanReversionStrategy(**MR_PARAMS)
        res_m = BacktestEngine(
            prices=df, strategy=mr, capital=CAPITAL, cost=cost, symbol=sym
        ).run()

        rows.append({
            "symbol":      sym,
            "sector":      sector,
            # フジコ法
            "f_return":    res_f.total_return,
            "f_sharpe":    res_f.sharpe_ratio,
            "f_maxdd":     res_f.max_drawdown,
            "f_trades":    res_f.num_trades,
            "f_winrate":   res_f.win_rate,
            "f_hold_days": avg_hold_days(res_f.trades),
            # 平均回帰
            "m_return":    res_m.total_return,
            "m_sharpe":    res_m.sharpe_ratio,
            "m_maxdd":     res_m.max_drawdown,
            "m_trades":    res_m.num_trades,
            "m_winrate":   res_m.win_rate,
            "m_hold_days": avg_hold_days(res_m.trades),
        })

        print(
            f"  {sym:<10} [{sector:<8}]  "
            f"フジコ: {res_f.total_return*100:>+6.1f}% Shr={res_f.sharpe_ratio:>5.2f}  "
            f"平均回帰: {res_m.total_return*100:>+6.1f}% Shr={res_m.sharpe_ratio:>5.2f}"
        )

    df_cmp = pd.DataFrame(rows)

    # ---- 4. 結果表示 ----
    print("\n\n[4/4] 比較結果")
    print("\n" + "=" * 70)
    print("  戦略サマリー（全銘柄平均）")
    print("=" * 70)
    print(f"  {'指標':<22} {'フジコ法':>14} {'短期平均回帰':>14}")
    print("  " + "-" * 52)

    metrics = [
        ("平均リターン (%)",    "f_return",    "m_return",    100, "+.1f"),
        ("平均シャープ比",      "f_sharpe",    "m_sharpe",    1,   ".3f"),
        ("平均最大DD (%)",      "f_maxdd",     "m_maxdd",     100, "+.1f"),
        ("総取引回数",          "f_trades",    "m_trades",    1,   ".0f"),
        ("平均勝率 (%)",        "f_winrate",   "m_winrate",   100, ".1f"),
        ("平均保有日数",        "f_hold_days", "m_hold_days", 1,   ".1f"),
    ]
    for label, fcol, mcol, mult, fmt in metrics:
        fval = df_cmp[fcol].mean() * mult
        mval = df_cmp[mcol].mean() * mult
        print(f"  {label:<22} {format(fval, fmt):>14} {format(mval, fmt):>14}")

    # Phase 1 達成率
    f_pass = ((df_cmp["f_sharpe"] > 0.5) & (df_cmp["f_maxdd"].abs() < 0.20)).sum()
    m_pass = ((df_cmp["m_sharpe"] > 0.5) & (df_cmp["m_maxdd"].abs() < 0.20)).sum()
    n = len(df_cmp)
    print(f"\n  Phase 1 達成（Sharpe>0.5 かつ MaxDD<20%）")
    print(f"    フジコ法    : {f_pass}/{n} 銘柄 ({f_pass/n*100:.0f}%)")
    print(f"    短期平均回帰: {m_pass}/{n} 銘柄 ({m_pass/n*100:.0f}%)")
    print("=" * 70)

    # 銘柄別テーブル
    print("\n=== 銘柄別 比較テーブル ===")
    hdr = (
        f"{'銘柄':<10} {'セクター':<10} "
        f"{'F-Ret':>7} {'F-Shr':>6} {'F-DD':>6} {'F-N':>4}  "
        f"{'M-Ret':>7} {'M-Shr':>6} {'M-DD':>6} {'M-N':>4}"
    )
    print(hdr)
    print("-" * len(hdr))
    for _, r in df_cmp.iterrows():
        print(
            f"{r['symbol']:<10} {r['sector']:<10} "
            f"{r['f_return']*100:>+6.1f}% {r['f_sharpe']:>6.2f} "
            f"{r['f_maxdd']*100:>+5.1f}% {int(r['f_trades']):>4}  "
            f"{r['m_return']*100:>+6.1f}% {r['m_sharpe']:>6.2f} "
            f"{r['m_maxdd']*100:>+5.1f}% {int(r['m_trades']):>4}"
        )

    # セクター別平均
    print("\n=== セクター別 平均リターン ===")
    sect = df_cmp.groupby("sector")[["f_return", "m_return"]].mean() * 100
    sect.columns = ["フジコ法(%)", "平均回帰(%)"]
    sect["差(MR-F)"] = sect["平均回帰(%)"] - sect["フジコ法(%)"]
    sect = sect.sort_values("差(MR-F)", ascending=False)
    print(sect.to_string(float_format=lambda x: f"{x:+.1f}"))

    # CSV 保存
    os.makedirs("data", exist_ok=True)
    csv_path = "data/mean_reversion_comparison.csv"
    df_cmp.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n  CSV 保存: {csv_path}")

    # ---- グラフ ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("短期平均回帰戦略 vs フジコ法\n2018-2024 TOPIX相当76銘柄", fontsize=13)

    # 1) リターン散布図
    ax = axes[0, 0]
    ax.scatter(df_cmp["f_return"] * 100, df_cmp["m_return"] * 100, alpha=0.6)
    lim = max(abs(df_cmp[["f_return", "m_return"]].values.flatten() * 100)) + 5
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.plot([-lim, lim], [-lim, lim], "r--", lw=0.8, label="等線")
    ax.set_xlabel("フジコ法 リターン (%)")
    ax.set_ylabel("短期平均回帰 リターン (%)")
    ax.set_title("リターン比較")
    ax.legend()

    # 2) シャープ比 分布
    ax = axes[0, 1]
    ax.hist(df_cmp["f_sharpe"], bins=20, alpha=0.6, label="フジコ法", color="steelblue")
    ax.hist(df_cmp["m_sharpe"], bins=20, alpha=0.6, label="平均回帰", color="tomato")
    ax.axvline(0.5, color="green", lw=1.5, linestyle="--", label="Phase1基準(0.5)")
    ax.set_xlabel("シャープ比")
    ax.set_title("シャープ比 分布")
    ax.legend()

    # 3) 勝率 比較
    ax = axes[1, 0]
    ax.scatter(df_cmp["f_winrate"] * 100, df_cmp["m_winrate"] * 100, alpha=0.6)
    ax.plot([0, 100], [0, 100], "r--", lw=0.8)
    ax.set_xlabel("フジコ法 勝率 (%)")
    ax.set_ylabel("短期平均回帰 勝率 (%)")
    ax.set_title("勝率比較")

    # 4) セクター別平均リターン
    ax = axes[1, 1]
    sect_plot = df_cmp.groupby("sector")[["f_return", "m_return"]].mean() * 100
    x = range(len(sect_plot))
    w = 0.35
    bars1 = ax.bar([i - w/2 for i in x], sect_plot["f_return"], w,
                   label="フジコ法", color="steelblue", alpha=0.8)
    bars2 = ax.bar([i + w/2 for i in x], sect_plot["m_return"], w,
                   label="平均回帰", color="tomato", alpha=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(sect_plot.index, rotation=45, ha="right", fontsize=8)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("平均リターン (%)")
    ax.set_title("セクター別 平均リターン")
    ax.legend()

    plt.tight_layout()
    png_path = "data/mean_reversion_comparison.png"
    plt.savefig(png_path, dpi=120, bbox_inches="tight")
    print(f"  PNG 保存: {png_path}")
    print("\n完了。")


if __name__ == "__main__":
    main()
