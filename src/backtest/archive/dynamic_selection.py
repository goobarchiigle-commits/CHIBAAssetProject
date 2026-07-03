"""
backtest/dynamic_selection.py
動的銘柄抽出 + 年率換算（CAGR）ランキング

【設計方針】
  昨日の比較バックテスト結果を踏まえ、セクターごとに得意戦略を自動割当する。
  フジコ法が優位なセクター → フジコ法のCAGRを採用
  平均回帰が優位なセクター → 平均回帰のCAGRを採用
  両者の差が小さいセクター → 高い方を採用（動的選択）

【出力】
  - コンソール: CAGRランキング（上位20銘柄）+ セクター別サマリー
  - CSV: data/dynamic_selection.csv
  - PNG: data/dynamic_selection.png

【実行方法】
  python -m backtest.dynamic_selection

【CAGR計算式】
  CAGR = (1 + total_return) ** (365.25 / holding_days) - 1
  ※ バックテスト全期間の「在籍日数」ではなく「実際の保有日数合計」で年率換算
  ※ 保有日数ゼロ（ノートレード）の場合は total_return をそのまま表示

【先読みリーク禁止（CLAUDE.md ルール1）】
  - TimeSeriesSplit に準拠したエンジンを使用
  - 本スクリプトでは追加の学習/テスト分割は行わず全期間バックテスト
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
plt.rcParams["font.family"] = "MS Gothic"

import numpy as np
import pandas as pd

from backtest.universe_builder        import TOPIX100_TICKERS, download_universe
from backtest.rsr                     import calc_universe_rsr
from backtest.fujiko_strategy         import FujikoStrategy
from backtest.mean_reversion_strategy import MeanReversionStrategy
from backtest.engine                  import BacktestEngine, TradeCost

START   = "2018-01-01"
END     = "2024-12-31"
CAPITAL = 1_000_000   # 銘柄ごとの仮想資金

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

# セクター別 推奨戦略（昨日の比較バックテスト結果に基づく）
# フジコ法優位: 差(F-MR) が大きい順
SECTOR_STRATEGY: dict[str, str] = {
    "海運":    "fujiko",
    "機械":    "fujiko",
    "電機精密": "fujiko",
    "商社":    "fujiko",
    "電機":    "fujiko",
    "ゲーム":  "fujiko",
    "レジャー": "fujiko",
    "食品":    "fujiko",   # 差が小さい → フジコ優位
    # 平均回帰優位
    "ガス":    "mean_rev",
    "鉄鋼":    "mean_rev",
    "銀行":    "mean_rev",
    "保険":    "mean_rev",
    "輸送機器": "mean_rev",
    "化学":    "mean_rev",
    "小売":    "mean_rev",
    # 差が小さい or 両方低調 → 動的（高い方を採用）
    "サービス":  "dynamic",
    "医薬品":   "dynamic",
    "不動産":   "dynamic",
    "情報通信":  "dynamic",
    "陸運":     "dynamic",
}


# ------------------------------------------------------------------ #
# CAGR 計算
# ------------------------------------------------------------------ #
def calc_cagr(total_return: float, num_trades: int, avg_hold_days: float) -> float:
    """
    実際の保有日数合計から年率換算リターン（CAGR）を計算する。

    保有日数が 0 の場合（ノートレード）は total_return をそのまま返す。
    """
    total_hold_days = num_trades * avg_hold_days
    if total_hold_days <= 0:
        return total_return
    years = total_hold_days / 365.25
    if years <= 0:
        return total_return
    return (1 + total_return) ** (1.0 / years) - 1


def avg_hold_days(trades: list) -> float:
    buys  = [t for t in trades if t.side == "BUY"]
    sells = [t for t in trades if t.side.startswith("SELL")]
    pairs = list(zip(buys, sells[: len(buys)]))
    if not pairs:
        return 0.0
    return float(np.mean([(s.date - b.date).days for b, s in pairs]))


# ------------------------------------------------------------------ #
# メイン
# ------------------------------------------------------------------ #
def main() -> None:
    print("=" * 72)
    print("  動的銘柄抽出 + 年率換算（CAGR）ランキング")
    print(f"  期間: {START} 〜 {END}  /  初期資本: ¥{CAPITAL:,} / 銘柄")
    print("=" * 72)

    # ---- 1. データ取得 ----
    print("\n[1/4] ユニバースデータ取得中...")
    universe = download_universe(TOPIX100_TICKERS, start=START, end=END)
    print(f"  取得完了: {len(universe)} 銘柄")

    # ---- 2. RSR 計算（フジコ法用）----
    print("\n[2/4] RSR 計算中...")
    universe_prices = {sym: info["df"]["Close"] for sym, info in universe.items()}
    rsr_universe    = calc_universe_rsr(universe_prices)

    # ---- 3. 銘柄別バックテスト ----
    print("\n[3/4] 銘柄別バックテスト中...")
    cost = TradeCost()
    rows = []

    for sym, info in universe.items():
        df     = info["df"]
        sector = info["sector"]

        # フジコ法
        rsr_s  = rsr_universe[sym] if sym in rsr_universe.columns else None
        fujiko = FujikoStrategy(rsr_series=rsr_s, **FUJIKO_PARAMS)
        res_f  = BacktestEngine(prices=df, strategy=fujiko,
                                capital=CAPITAL, cost=cost, symbol=sym).run()

        # 平均回帰
        mr    = MeanReversionStrategy(**MR_PARAMS)
        res_m = BacktestEngine(prices=df, strategy=mr,
                               capital=CAPITAL, cost=cost, symbol=sym).run()

        # CAGR 計算
        f_hold = avg_hold_days(res_f.trades)
        m_hold = avg_hold_days(res_m.trades)
        f_cagr = calc_cagr(res_f.total_return, res_f.num_trades, f_hold)
        m_cagr = calc_cagr(res_m.total_return, res_m.num_trades, m_hold)

        # セクター別に採用戦略を決定
        rule = SECTOR_STRATEGY.get(sector, "dynamic")
        if rule == "fujiko":
            chosen_strategy = "フジコ法"
            chosen_cagr     = f_cagr
            chosen_sharpe   = res_f.sharpe_ratio
            chosen_maxdd    = res_f.max_drawdown
            chosen_return   = res_f.total_return
            chosen_trades   = res_f.num_trades
        elif rule == "mean_rev":
            chosen_strategy = "平均回帰"
            chosen_cagr     = m_cagr
            chosen_sharpe   = res_m.sharpe_ratio
            chosen_maxdd    = res_m.max_drawdown
            chosen_return   = res_m.total_return
            chosen_trades   = res_m.num_trades
        else:  # dynamic: 高い方
            if f_cagr >= m_cagr:
                chosen_strategy = "フジコ法(D)"
                chosen_cagr     = f_cagr
                chosen_sharpe   = res_f.sharpe_ratio
                chosen_maxdd    = res_f.max_drawdown
                chosen_return   = res_f.total_return
                chosen_trades   = res_f.num_trades
            else:
                chosen_strategy = "平均回帰(D)"
                chosen_cagr     = m_cagr
                chosen_sharpe   = res_m.sharpe_ratio
                chosen_maxdd    = res_m.max_drawdown
                chosen_return   = res_m.total_return
                chosen_trades   = res_m.num_trades

        phase1_ok = (chosen_sharpe > 0.5) and (abs(chosen_maxdd) < 0.20)

        rows.append({
            "symbol":    sym,
            "sector":    sector,
            "strategy":  chosen_strategy,
            "cagr":      chosen_cagr,
            "total_ret": chosen_return,
            "sharpe":    chosen_sharpe,
            "maxdd":     chosen_maxdd,
            "trades":    chosen_trades,
            "phase1":    phase1_ok,
            # 参考: 両戦略の生値
            "f_cagr":   f_cagr,
            "f_sharpe": res_f.sharpe_ratio,
            "f_maxdd":  res_f.max_drawdown,
            "m_cagr":   m_cagr,
            "m_sharpe": res_m.sharpe_ratio,
            "m_maxdd":  res_m.max_drawdown,
        })

        mark = "✅" if phase1_ok else "  "
        print(
            f"  {mark} {sym:<10}[{sector:<8}] {chosen_strategy:<10} "
            f"CAGR={chosen_cagr*100:>+6.1f}%  "
            f"Shr={chosen_sharpe:>5.2f}  DD={chosen_maxdd*100:>+5.1f}%"
        )

    df_result = pd.DataFrame(rows)

    # ---- 4. ランキング表示 ----
    print("\n\n[4/4] CAGR ランキング")
    print("=" * 72)

    # Phase1通過銘柄のみ抽出してランキング
    df_pass  = df_result[df_result["phase1"]].sort_values("cagr", ascending=False)
    df_all   = df_result.sort_values("cagr", ascending=False)

    print(f"\n★ Phase1達成（Sharpe>0.5 かつ MaxDD<20%）: {len(df_pass)}/{len(df_result)} 銘柄")
    print(f"\n  {'銘柄':<10} {'セクター':<10} {'戦略':<12} "
          f"{'CAGR':>8} {'総Ret':>8} {'Sharpe':>7} {'MaxDD':>7} {'取引':>5}")
    print("  " + "-" * 70)
    for _, r in df_pass.head(20).iterrows():
        print(
            f"  {r['symbol']:<10} {r['sector']:<10} {r['strategy']:<12} "
            f"{r['cagr']*100:>+7.1f}% {r['total_ret']*100:>+7.1f}% "
            f"{r['sharpe']:>7.2f} {r['maxdd']*100:>+6.1f}% {int(r['trades']):>5}"
        )

    if len(df_pass) == 0:
        print("  ※ Phase1達成銘柄なし。全銘柄のトップ10を表示:")
        for _, r in df_all.head(10).iterrows():
            print(
                f"  {r['symbol']:<10} {r['sector']:<10} {r['strategy']:<12} "
                f"CAGR={r['cagr']*100:>+7.1f}%  Shr={r['sharpe']:>6.2f}  DD={r['maxdd']*100:>+6.1f}%"
            )

    # セクター別サマリー
    print("\n=== セクター別 CAGR サマリー ===")
    sect_grp = df_result.groupby("sector")
    sect_summary = sect_grp.agg(
        銘柄数=("symbol", "count"),
        CAGR平均=("cagr", "mean"),
        CAGR最大=("cagr", "max"),
        Sharpe平均=("sharpe", "mean"),
        Phase1数=("phase1", "sum"),
    ).sort_values("CAGR平均", ascending=False)
    sect_summary["CAGR平均"] = sect_summary["CAGR平均"] * 100
    sect_summary["CAGR最大"] = sect_summary["CAGR最大"] * 100
    for idx, r in sect_summary.iterrows():
        rule = SECTOR_STRATEGY.get(idx, "dynamic")
        strat_label = {"fujiko": "フジコ", "mean_rev": "平均回帰", "dynamic": "動的"}[rule]
        print(
            f"  {idx:<10}({strat_label:<5}) "
            f"CAGR平均={r['CAGR平均']:>+6.1f}%  "
            f"最大={r['CAGR最大']:>+6.1f}%  "
            f"Phase1={int(r['Phase1数'])}/{int(r['銘柄数'])}"
        )

    # ---- CSV 保存 ----
    os.makedirs("data", exist_ok=True)
    csv_path = "data/dynamic_selection.csv"
    df_result.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n  CSV: {csv_path}")

    # ---- グラフ ----
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("動的銘柄抽出 CAGR ランキング（2018-2024）", fontsize=13)

    # 左: CAGR 上位20銘柄の棒グラフ
    ax = axes[0]
    top20 = df_all.head(20)
    colors = ["steelblue" if "フジコ" in s else "tomato" for s in top20["strategy"]]
    bars = ax.barh(
        [f"{r['symbol']}\n[{r['sector']}]" for _, r in top20.iterrows()],
        top20["cagr"] * 100,
        color=colors,
        alpha=0.8,
    )
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlabel("CAGR (%)")
    ax.set_title("CAGR 上位20銘柄")
    ax.invert_yaxis()
    # 凡例
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="steelblue", alpha=0.8, label="フジコ法"),
        Patch(color="tomato",    alpha=0.8, label="平均回帰"),
    ])

    # 右: セクター別 CAGR 平均（採用戦略色分け）
    ax = axes[1]
    sect_data = sect_summary.sort_values("CAGR平均", ascending=True)
    sect_colors = []
    for idx in sect_data.index:
        rule = SECTOR_STRATEGY.get(idx, "dynamic")
        if rule == "fujiko":
            sect_colors.append("steelblue")
        elif rule == "mean_rev":
            sect_colors.append("tomato")
        else:
            sect_colors.append("mediumpurple")
    ax.barh(sect_data.index, sect_data["CAGR平均"], color=sect_colors, alpha=0.8)
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlabel("CAGR 平均 (%)")
    ax.set_title("セクター別 CAGR 平均")
    ax.legend(handles=[
        Patch(color="steelblue",    alpha=0.8, label="フジコ法セクター"),
        Patch(color="tomato",       alpha=0.8, label="平均回帰セクター"),
        Patch(color="mediumpurple", alpha=0.8, label="動的選択"),
    ])

    plt.tight_layout()
    png_path = "data/dynamic_selection.png"
    plt.savefig(png_path, dpi=120, bbox_inches="tight")
    print(f"  PNG: {png_path}")
    print("\n完了。")


if __name__ == "__main__":
    main()
