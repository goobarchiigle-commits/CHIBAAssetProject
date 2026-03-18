"""
backtest/strategy_comparison.py
フジコ法 vs REVE Lite JP 戦略比較バックテスト

【実行方法】
  python -m backtest.strategy_comparison

【出力】
  - コンソール: 銘柄別×戦略別の比較テーブル + Phase 1 達成率
  - PNG: data/strategy_comparison.png
  - CSV: data/strategy_comparison.csv

【比較条件】
  フジコ法   : SEPA>=6, RSR>=75, RSRモメンタム, タートルズ20/10
               エグジット: RSRモメンタム反転 または 10日安値割れ
  REVE Lite  : REVE スコア グレーフラグエントリー
               エグジット: TP +5% または 10バー時間制限
  期間       : 2018-01-01 〜 2024-12-31
  初期資本   : 100万円 / 銘柄（単一銘柄バックテスト）
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

import numpy as np
import pandas as pd

from backtest.universe_builder   import TOPIX100_TICKERS, download_universe
from backtest.rsr                import calc_universe_rsr
from backtest.fujiko_strategy    import FujikoStrategy
from backtest.reve_lite_strategy import ReveLiteStrategy
from backtest.engine             import BacktestEngine, TradeCost

START   = "2018-01-01"
END     = "2024-12-31"
CAPITAL = 1_000_000    # 1銘柄あたりの初期資本

FUJIKO_PARAMS = dict(
    min_sepa         = 6,
    min_rsr          = 75.0,
    mom_period       = 21,
    turtle_entry     = 20,
    turtle_exit      = 10,
    use_turtle_entry = True,
)

REVE_PARAMS = dict(
    watch_threshold  = 1.2,
    orange_threshold = 1.8,
)

REVE_ENGINE_PARAMS = dict(
    take_profit_pct = 0.05,   # TP +5%
    max_hold_days   = 10,     # 10バー時間制限
)


# ------------------------------------------------------------------ #
# ヘルパー
# ------------------------------------------------------------------ #
def avg_hold_days(trades: list) -> float:
    """BUY → SELL ペアの平均保有日数を計算する。"""
    buys  = [t for t in trades if t.side == "BUY"]
    sells = [t for t in trades if t.side.startswith("SELL")]
    pairs = list(zip(buys, sells[: len(buys)]))
    if not pairs:
        return 0.0
    days = [(s.date - b.date).days for b, s in pairs]
    return float(np.mean(days))


def tp_rate(trades: list) -> float:
    """TP で決済された割合（REVE Lite 専用）。"""
    sells = [t for t in trades if t.side.startswith("SELL")]
    if not sells:
        return 0.0
    return sum(1 for t in sells if t.side == "SELL_TP") / len(sells)


# ------------------------------------------------------------------ #
# メイン
# ------------------------------------------------------------------ #
def main() -> None:
    print("=" * 70)
    print("  フジコ法 vs REVE Lite JP  戦略比較バックテスト")
    print(f"  期間: {START} 〜 {END}  /  初期資本: ¥{CAPITAL:,} / 銘柄")
    print("=" * 70)

    # ---- 1. ユニバースデータ取得 ----
    print("\n[1/4] ユニバースデータ取得中...")
    universe = download_universe(TOPIX100_TICKERS, start=START, end=END)
    print(f"  取得完了: {len(universe)} 銘柄")

    # ---- 2. RSR 計算 ----
    print("\n[2/4] RSR 計算中...")
    universe_prices = {sym: info["df"]["Close"] for sym, info in universe.items()}
    rsr_universe    = calc_universe_rsr(universe_prices)
    print(f"  RSR 計算完了: {rsr_universe.shape[0]} 日 × {rsr_universe.shape[1]} 銘柄")

    # ---- 3. 銘柄ごとにバックテスト実行 ----
    print("\n[3/4] 銘柄別バックテスト実行中（しばらくかかります）...")
    cost = TradeCost()
    rows = []

    for sym, info in universe.items():
        df     = info["df"]
        sector = info["sector"]

        # -- フジコ法 --
        rsr_s  = rsr_universe[sym] if sym in rsr_universe.columns else None
        fujiko = FujikoStrategy(rsr_series=rsr_s, **FUJIKO_PARAMS)
        res_f  = BacktestEngine(
            prices=df, strategy=fujiko, capital=CAPITAL, cost=cost, symbol=sym
        ).run()

        # -- REVE Lite --
        reve  = ReveLiteStrategy(**REVE_PARAMS)
        res_r = BacktestEngine(
            prices=df, strategy=reve, capital=CAPITAL, cost=cost, symbol=sym,
            **REVE_ENGINE_PARAMS,
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
            # REVE Lite
            "r_return":    res_r.total_return,
            "r_sharpe":    res_r.sharpe_ratio,
            "r_maxdd":     res_r.max_drawdown,
            "r_trades":    res_r.num_trades,
            "r_winrate":   res_r.win_rate,
            "r_hold_days": avg_hold_days(res_r.trades),
            "r_tp_rate":   tp_rate(res_r.trades),
        })

        print(
            f"  {sym:<10} [{sector:<8}]  "
            f"フジコ: {res_f.total_return*100:>+6.1f}% Shr={res_f.sharpe_ratio:>5.2f}  "
            f"REVE:  {res_r.total_return*100:>+6.1f}% Shr={res_r.sharpe_ratio:>5.2f}"
        )

    df_cmp = pd.DataFrame(rows)

    # ---- 4. 結果表示 ----
    print("\n\n[4/4] 比較結果")

    # ---- 戦略サマリー ----
    print("\n" + "=" * 70)
    print("  戦略サマリー（全銘柄平均）")
    print("=" * 70)
    print(f"  {'指標':<20} {'フジコ法':>14} {'REVE Lite':>14}")
    print("  " + "-" * 50)
    metrics = [
        ("平均リターン (%)",    "f_return",    "r_return",    100, "+.1f"),
        ("平均シャープ比",      "f_sharpe",    "r_sharpe",    1,   ".3f"),
        ("平均最大DD (%)",      "f_maxdd",     "r_maxdd",     100, "+.1f"),
        ("総取引回数",          "f_trades",    "r_trades",    1,   ".0f"),
        ("平均勝率 (%)",        "f_winrate",   "r_winrate",   100, ".1f"),
        ("平均保有日数",        "f_hold_days", "r_hold_days", 1,   ".1f"),
    ]
    for label, fcol, rcol, mult, fmt in metrics:
        fval = df_cmp[fcol].mean() * mult
        rval = df_cmp[rcol].mean() * mult
        print(f"  {label:<20} {format(fval, fmt):>14} {format(rval, fmt):>14}")

    # TP 決済率（REVE Lite 専用）
    print(f"  {'TP 決済率 (REVE)':<20} {'':>14} {df_cmp['r_tp_rate'].mean()*100:>13.1f}%")

    # ---- Phase 1 達成率 ----
    f_pass = ((df_cmp["f_sharpe"] > 0.5) & (df_cmp["f_maxdd"].abs() < 0.20)).sum()
    r_pass = ((df_cmp["r_sharpe"] > 0.5) & (df_cmp["r_maxdd"].abs() < 0.20)).sum()
    n = len(df_cmp)
    print(f"\n  Phase 1 達成銘柄（Sharpe>0.5 かつ MaxDD<20%）")
    print(f"    フジコ法  : {f_pass}/{n} 銘柄 ({f_pass/n*100:.0f}%)")
    print(f"    REVE Lite : {r_pass}/{n} 銘柄 ({r_pass/n*100:.0f}%)")
    print("=" * 70)

    # ---- 銘柄別詳細テーブル ----
    print("\n=== 銘柄別 比較テーブル ===")
    hdr = (
        f"{'銘柄':<10} {'セクター':<10} "
        f"{'F-Ret':>7} {'F-Shr':>6} {'F-DD':>6} {'F-N':>4}  "
        f"{'R-Ret':>7} {'R-Shr':>6} {'R-DD':>6} {'R-N':>4} {'R-TP%':>6}"
    )
    print(hdr)
    print("-" * 80)
    for _, r in df_cmp.iterrows():
        print(
            f"{r['symbol']:<10} {r['sector']:<10} "
            f"{r['f_return']*100:>+6.1f}% {r['f_sharpe']:>6.2f} {r['f_maxdd']*100:>+5.1f}% {int(r['f_trades']):>4}  "
            f"{r['r_return']*100:>+6.1f}% {r['r_sharpe']:>6.2f} {r['r_maxdd']*100:>+5.1f}% {int(r['r_trades']):>4} "
            f"{r['r_tp_rate']*100:>5.0f}%"
        )

    # ---- CSV 保存 ----
    os.makedirs("data", exist_ok=True)
    csv_path = "data/strategy_comparison.csv"
    df_cmp.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n💾 CSV 保存: {csv_path}")

    # ---- グラフ保存 ----
    print("グラフ生成中...")
    _plot_comparison(df_cmp)
    print("グラフ保存: data/strategy_comparison.png")


# ------------------------------------------------------------------ #
# グラフ
# ------------------------------------------------------------------ #
def _plot_comparison(df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    import platform

    if platform.system() == "Windows":
        plt.rcParams["font.family"] = "MS Gothic"

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        "フジコ法 vs REVE Lite JP — 銘柄別比較（2018-2024）",
        fontsize=14,
    )

    x     = np.arange(len(df))
    width = 0.38
    lbls  = df["symbol"].tolist()

    # リターン比較
    ax = axes[0, 0]
    ax.bar(x - width / 2, df["f_return"] * 100, width, label="フジコ法",  color="royalblue", alpha=0.8)
    ax.bar(x + width / 2, df["r_return"] * 100, width, label="REVE Lite", color="tomato",    alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("総リターン (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(lbls, rotation=90, fontsize=7)
    ax.legend(fontsize=8)

    # シャープ比
    ax = axes[0, 1]
    ax.bar(x - width / 2, df["f_sharpe"], width, label="フジコ法",  color="royalblue", alpha=0.8)
    ax.bar(x + width / 2, df["r_sharpe"], width, label="REVE Lite", color="tomato",    alpha=0.8)
    ax.axhline(0.5, color="red", linestyle="--", linewidth=0.8, label="Phase1 基準(0.5)")
    ax.set_title("シャープ比")
    ax.set_xticks(x)
    ax.set_xticklabels(lbls, rotation=90, fontsize=7)
    ax.legend(fontsize=8)

    # 最大 DD
    ax = axes[1, 0]
    ax.bar(x - width / 2, df["f_maxdd"] * 100, width, label="フジコ法",  color="royalblue", alpha=0.8)
    ax.bar(x + width / 2, df["r_maxdd"] * 100, width, label="REVE Lite", color="tomato",    alpha=0.8)
    ax.axhline(-20, color="red", linestyle="--", linewidth=0.8, label="Phase1 基準(-20%)")
    ax.set_title("最大ドローダウン (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(lbls, rotation=90, fontsize=7)
    ax.legend(fontsize=8)

    # リスク・リターン散布図
    ax = axes[1, 1]
    ax.scatter(
        df["f_sharpe"], df["f_maxdd"] * 100,
        color="royalblue", alpha=0.7, s=55, label="フジコ法",  zorder=3,
    )
    ax.scatter(
        df["r_sharpe"], df["r_maxdd"] * 100,
        color="tomato", alpha=0.7, s=55, label="REVE Lite", marker="^", zorder=3,
    )
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax.axhline(-20, color="gray", linestyle="--", linewidth=0.8)
    # Phase 1 合格ゾーン（右上）
    ax.axvspan(0.5, ax.get_xlim()[1] if ax.get_xlim()[1] > 0.5 else 3.0,
               alpha=0.05, color="green")
    ax.set_xlabel("シャープ比")
    ax.set_ylabel("最大DD (%)")
    ax.set_title("リスク・リターン散布図（右上が Phase 1 合格ゾーン）")
    ax.legend(fontsize=8)

    plt.tight_layout()
    os.makedirs("data", exist_ok=True)
    plt.savefig("data/strategy_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
