"""
backtest/rsr_sensitivity.py
min_rsr 感度分析 + max_positions 比較

【実験設計】
  V2固定設定: 29銘柄 / 均等ウェイト / max_positions=3 / min_sepa=6
  スイープ: min_rsr = 60 / 65 / 70 / 75
  追加比較: min_rsr=70 + max_positions=4

【評価指標】
  CAGR / MaxDD / Calmar / Sharpe
  avg_exposure / efficiency(CAGR/exposure)
  turnover(年換算取引数) / return_per_trade / dd_per_exposure
  exposure p10 / p50 / p90
  zero_exposure_ratio（露光ゼロ日の割合）
"""

from __future__ import annotations
import os, sys, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from backtest.universe_builder        import download_universe
from backtest.rsr                     import calc_universe_rsr
from backtest.fujiko_strategy         import FujikoStrategy
from backtest.mean_reversion_strategy import MeanReversionStrategy
from backtest.portfolio_engine        import PortfolioEngine
from backtest.engine                  import TradeCost

# ------------------------------------------------------------------ #
# 設定
# ------------------------------------------------------------------ #
START   = "2018-01-01"
END     = "2024-12-31"
CAPITAL = 2_000_000
OUTPUT  = "C:/Users/owner/.claude/レポート"
os.makedirs(OUTPUT, exist_ok=True)

# セクター→戦略マッピング（portfolio_v2.py と同一）
SECTOR_STRATEGY: dict[str, str] = {
    "海運":"fujiko","機械":"fujiko","電機精密":"fujiko","商社":"fujiko",
    "電機":"fujiko","ゲーム":"fujiko","レジャー":"fujiko","食品":"fujiko",
    "ガス":"mean_rev","鉄鋼":"mean_rev","銀行":"mean_rev","保険":"mean_rev",
    "輸送機器":"mean_rev","化学":"mean_rev","小売":"mean_rev",
    "サービス":"dynamic","医薬品":"dynamic","不動産":"dynamic",
    "情報通信":"dynamic","陸運":"dynamic",
}
MR_PARAMS = dict(
    rsi_period=5, rsi_entry=25.0, rsi_exit=65.0,
    ma_long=200, stop_loss_pct=0.07, max_hold_days=10, knife_threshold=0.15,
)

# V2に固定追加する2銘柄
FIXED_ADD: dict[str, str] = {
    "6857.T": "電機精密",   # アドバンテスト
    "6594.T": "電機精密",   # ニデック
}


# ------------------------------------------------------------------ #
# ヘルパー
# ------------------------------------------------------------------ #
def calmar(cagr: float, maxdd: float) -> float:
    dd = abs(maxdd)
    return cagr / dd if dd > 0 else float("inf")


def compute_metrics(res, label: str) -> dict:
    cagr  = res.cagr
    maxdd = res.max_drawdown
    cal   = calmar(cagr, maxdd)
    shar  = res.sharpe_ratio
    exp   = res.exposure_series

    avg_exp  = float(exp.mean())
    eff      = cagr / avg_exp if avg_exp > 0 else float("nan")
    zero_exp = float((exp == 0).mean())

    trading_days = len(res.equity_curve)
    years        = trading_days / 252
    n_trades     = res.n_trades
    turnover     = n_trades / years if years > 0 else 0.0
    rpt          = res.total_return / n_trades if n_trades > 0 else float("nan")
    dd_per_exp   = maxdd / avg_exp if avg_exp > 0 else float("nan")

    p10, p50, p90 = float(exp.quantile(0.10)), float(exp.quantile(0.50)), float(exp.quantile(0.90))

    return {
        "シナリオ":         label,
        "CAGR%":           round(cagr * 100, 2),
        "MaxDD%":          round(maxdd * 100, 2),
        "Calmar":          round(cal, 3),
        "Sharpe":          round(shar, 3),
        "avg_exp":         round(avg_exp, 3),
        "efficiency":      round(eff, 3),    # CAGR / avg_exposure
        "turnover/yr":     round(turnover, 1),
        "ret/trade":       round(rpt * 100, 2),  # % per trade
        "dd/exposure":     round(dd_per_exp, 3),
        "exp_p10":         round(p10, 3),
        "exp_p50":         round(p50, 3),
        "exp_p90":         round(p90, 3),
        "zero_exp%":       round(zero_exp * 100, 1),
        "取引数":          n_trades,
        "平均保有銘柄":    round(float(res.n_positions.mean()), 2),
    }


def build_strats(universe: dict, rsr_uni: pd.DataFrame, sym_to_strat: dict,
                 min_rsr: float) -> dict:
    strat_dict = {}
    for sym, info in universe.items():
        rsr_s = rsr_uni[sym] if sym in rsr_uni.columns else None
        rule  = SECTOR_STRATEGY.get(info["sector"], "dynamic")
        lbl   = sym_to_strat.get(sym, "フジコ法")
        if rule == "mean_rev" or (rule == "dynamic" and "平均回帰" in lbl):
            strat_dict[sym] = MeanReversionStrategy(**MR_PARAMS)
        else:
            strat_dict[sym] = FujikoStrategy(
                rsr_series=rsr_s, min_sepa=6, min_rsr=min_rsr,
                mom_period=21, turtle_entry=20, turtle_exit=20,
                use_turtle_entry=True,
            )
    return strat_dict


def run_portfolio(universe: dict, strat_dict: dict, max_positions: int = 3) -> object:
    return PortfolioEngine(
        universe=universe, strategy=strat_dict, capital=CAPITAL,
        max_dd_limit=0.15, min_sectors=1, max_positions=max_positions,
        cost=TradeCost(), vol_target=0.0, max_single_weight=0.25,
        vol_window=20, use_idm=False, use_dd_scalar=False,
    ).run()


# ================================================================
# MAIN
# ================================================================
def main():
    print("=" * 72)
    print("  rsr_sensitivity.py — min_rsr 感度分析")
    print("  期間: 2018〜2024  /  初期資本: ¥2,000,000")
    print("  固定設定: 29銘柄 / 均等ウェイト / max_positions=3")
    print("=" * 72)

    # ---- 1. ユニバース構築 ----
    df_sel       = pd.read_csv("data/dynamic_selection.csv", encoding="utf-8-sig")
    sym_to_strat = dict(zip(df_sel["symbol"], df_sel["strategy"]))
    mask_27      = (df_sel["sharpe"] > 0.30) & (df_sel["maxdd"].abs() < 0.30)
    tickers_27   = {r["symbol"]: r["sector"] for _, r in df_sel[mask_27].iterrows()}

    # V2固定29銘柄
    all_tickers = {**tickers_27, **FIXED_ADD}

    print(f"\n[1/3] データ取得中（{len(all_tickers)}銘柄）...")
    universe = download_universe(all_tickers, start=START, end=END, verbose=False)
    print(f"  取得完了: {len(universe)} 銘柄")

    # ---- 2. RSR計算 ----
    prices_all = {sym: info["df"]["Close"] for sym, info in universe.items()}
    rsr_uni    = calc_universe_rsr(prices_all)
    print(f"  RSR計算完了: {rsr_uni.shape[1]} 銘柄")

    # ---- 3. 感度分析 ----
    print("\n[2/3] シナリオ実行中...")

    scenarios = [
        ("RSR=60 / max3", 60.0, 3),
        ("RSR=65 / max3", 65.0, 3),
        ("RSR=70 / max3", 70.0, 3),   # ← 現行ベスト
        ("RSR=75 / max3", 75.0, 3),
        ("RSR=70 / max4", 70.0, 4),   # max_positions=4 比較
    ]

    all_metrics = []
    for label, min_rsr, max_pos in scenarios:
        print(f"  [{label}]...", end="", flush=True)
        strats = build_strats(universe, rsr_uni, sym_to_strat, min_rsr)
        res    = run_portfolio(universe, strats, max_positions=max_pos)
        m      = compute_metrics(res, label)
        all_metrics.append(m)
        print(f"  CAGR={m['CAGR%']:+.2f}%  Calmar={m['Calmar']:.3f}"
              f"  avg_exp={m['avg_exp']:.3f}  eff={m['efficiency']:.3f}"
              f"  zero_exp={m['zero_exp%']:.1f}%")

    # ---- 4. 結果テーブル ----
    print("\n" + "=" * 72)
    print("  【感度分析結果】")
    print("=" * 72)

    primary_cols = ["シナリオ", "CAGR%", "MaxDD%", "Calmar", "Sharpe",
                    "avg_exp", "efficiency", "zero_exp%", "平均保有銘柄", "取引数"]
    df_main = pd.DataFrame([{c: m[c] for c in primary_cols} for m in all_metrics])
    print("\n  ■ 主要指標")
    print(df_main.to_string(index=False))

    secondary_cols = ["シナリオ", "turnover/yr", "ret/trade", "dd/exposure",
                      "exp_p10", "exp_p50", "exp_p90"]
    df_sub = pd.DataFrame([{c: m[c] for c in secondary_cols} for m in all_metrics])
    print("\n  ■ exposure詳細 / 取引効率")
    print(df_sub.to_string(index=False))

    # ---- 5. ベースライン(RSR=70/max3)との差分 ----
    base = next(m for m in all_metrics if m["シナリオ"] == "RSR=70 / max3")
    print("\n  ■ ベースライン(RSR=70/max3)との差分")
    print(f"  {'シナリオ':<20} {'ΔCAGR%':>8} {'ΔCalmar':>8} {'Δavg_exp':>9} {'Δeff':>8}")
    for m in all_metrics:
        if m["シナリオ"] == base["シナリオ"]:
            continue
        print(f"  {m['シナリオ']:<20}"
              f"  {m['CAGR%']-base['CAGR%']:>+7.2f}"
              f"  {m['Calmar']-base['Calmar']:>+7.3f}"
              f"  {m['avg_exp']-base['avg_exp']:>+8.3f}"
              f"  {m['efficiency']-base['efficiency']:>+7.3f}")

    # ---- 6. 判定サマリー ----
    print("\n  ■ 最適点の判定")
    best_calmar = max(all_metrics, key=lambda m: m["Calmar"])
    best_eff    = max(all_metrics, key=lambda m: m["efficiency"])
    print(f"  Calmar最大  : {best_calmar['シナリオ']}  ({best_calmar['Calmar']:.3f})")
    print(f"  efficiency最大: {best_eff['シナリオ']}  ({best_eff['efficiency']:.3f})")
    print(f"  現行(RSR70) : CAGR={base['CAGR%']:+.2f}%  Calmar={base['Calmar']:.3f}"
          f"  eff={base['efficiency']:.3f}  avg_exp={base['avg_exp']:.3f}")

    # ---- 7. CSV保存 ----
    df_all = pd.DataFrame(all_metrics)
    out_csv = os.path.join(OUTPUT, "rsr_sensitivity.csv")
    df_all.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n  → {out_csv}")

    print("\n完了。")
    return all_metrics


if __name__ == "__main__":
    main()
