"""
backtest/sepa_sensitivity.py
min_sepa 感度分析 + エントリーファンネル診断

【実験設計】
  RSR = 75 固定（前回感度分析で最良と確認）
  スイープ: min_sepa = 3 / 4 / 5 / 6
  ユニバース: V2固定 29銘柄 / 均等ウェイト / max_positions=3

【評価指標（主要3つ）】
  avg_exposure / Calmar / zero_exp%

【ファンネル診断】
  全取引日を週次サンプリング（約300サンプル）して以下を計測:
    RSR通過数/日   : RSR >= min_rsr を満たす銘柄数の平均
    SEPA通過数/日  : 上記のうちさらに SEPA >= min_sepa を満たす銘柄数
    シグナル通過/日: generate_signal == +1 の銘柄数の平均
"""

from __future__ import annotations
import os, sys, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from backtest.universe_builder        import download_universe
from backtest.rsr                     import calc_universe_rsr, calc_sepa, calc_rsr_momentum
from backtest.fujiko_strategy         import FujikoStrategy
from backtest.mean_reversion_strategy import MeanReversionStrategy
from backtest.portfolio_engine        import PortfolioEngine
from backtest.engine                  import TradeCost

# ------------------------------------------------------------------ #
# 設定
# ------------------------------------------------------------------ #
START    = "2018-01-01"
END      = "2024-12-31"
CAPITAL  = 2_000_000
MIN_RSR  = 75.0   # 前回分析でベストと確認
OUTPUT   = "C:/Users/owner/.claude/レポート"
os.makedirs(OUTPUT, exist_ok=True)

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
FIXED_ADD: dict[str, str] = {
    "6857.T": "電機精密",
    "6594.T": "電機精密",
}


# ------------------------------------------------------------------ #
# ヘルパー
# ------------------------------------------------------------------ #
def calmar(cagr: float, maxdd: float) -> float:
    dd = abs(maxdd)
    return cagr / dd if dd > 0 else float("inf")


def build_strats(universe: dict, rsr_uni: pd.DataFrame, sym_to_strat: dict,
                 min_sepa: int) -> dict:
    strat_dict = {}
    for sym, info in universe.items():
        rsr_s = rsr_uni[sym] if sym in rsr_uni.columns else None
        rule  = SECTOR_STRATEGY.get(info["sector"], "dynamic")
        lbl   = sym_to_strat.get(sym, "フジコ法")
        if rule == "mean_rev" or (rule == "dynamic" and "平均回帰" in lbl):
            strat_dict[sym] = MeanReversionStrategy(**MR_PARAMS)
        else:
            strat_dict[sym] = FujikoStrategy(
                rsr_series=rsr_s, min_sepa=min_sepa, min_rsr=MIN_RSR,
                mom_period=21, turtle_entry=20, turtle_exit=20,
                use_turtle_entry=True,
            )
    return strat_dict


def run_portfolio(universe: dict, strat_dict: dict) -> object:
    return PortfolioEngine(
        universe=universe, strategy=strat_dict, capital=CAPITAL,
        max_dd_limit=0.15, min_sectors=1, max_positions=3,
        cost=TradeCost(), vol_target=0.0, max_single_weight=0.25,
        vol_window=20, use_idm=False, use_dd_scalar=False,
    ).run()


def compute_metrics(res, label: str) -> dict:
    cagr  = res.cagr
    maxdd = res.max_drawdown
    cal   = calmar(cagr, maxdd)
    shar  = res.sharpe_ratio
    exp   = res.exposure_series
    avg_exp  = float(exp.mean())
    eff      = cagr / avg_exp if avg_exp > 0 else float("nan")
    zero_exp = float((exp == 0).mean())
    n_trades = res.n_trades
    return {
        "シナリオ":     label,
        "CAGR%":       round(cagr * 100, 2),
        "MaxDD%":      round(maxdd * 100, 2),
        "Calmar":      round(cal, 3),
        "Sharpe":      round(shar, 3),
        "avg_exp":     round(avg_exp, 3),
        "efficiency":  round(eff, 3),
        "zero_exp%":   round(zero_exp * 100, 1),
        "平均保有銘柄": round(float(res.n_positions.mean()), 2),
        "取引数":      n_trades,
    }


# ------------------------------------------------------------------ #
# ファンネル診断
# ------------------------------------------------------------------ #
def run_funnel_diagnostic(
    universe: dict,
    rsr_uni: pd.DataFrame,
    sym_to_strat: dict,
    min_sepa_list: list[int],
    sample_step: int = 5,   # 5営業日ごとにサンプリング（週次相当）
) -> pd.DataFrame:
    """
    各 min_sepa について取引日を週次サンプリングし、
    エントリーパイプラインの各段階を通過する銘柄数を計測する。

    計測段階:
      N_universe   : ユニバース全銘柄数（フジコ法対象のみ）
      N_rsr        : RSR >= MIN_RSR を満たす銘柄数
      N_sepa       : 上記のうち sepa_score >= min_sepa を満たす銘柄数
      N_signal     : generate_signal == +1 の銘柄数

    Returns:
        pd.DataFrame（min_sepa×診断指標）
    """
    # フジコ法対象銘柄だけ抽出
    fujiko_syms = []
    for sym, info in universe.items():
        rule = SECTOR_STRATEGY.get(info["sector"], "dynamic")
        lbl  = sym_to_strat.get(sym, "フジコ法")
        if not (rule == "mean_rev" or (rule == "dynamic" and "平均回帰" in lbl)):
            fujiko_syms.append(sym)

    n_universe = len(fujiko_syms)

    # 共通取引日を取得
    dates = None
    for sym in fujiko_syms:
        idx = universe[sym]["df"].index
        dates = idx if dates is None else dates.intersection(idx)
    dates = sorted(dates)

    # 最低限のバーが揃う日以降（252+21+2=275日目以降）
    min_bars = 278
    sample_dates = [d for i, d in enumerate(dates) if i >= min_bars and i % sample_step == 0]

    rows = []
    for min_sepa in min_sepa_list:
        rsr_counts, sepa_counts, sig_counts = [], [], []

        for date in sample_dates:
            n_rsr = n_sepa = n_sig = 0
            date_ts = pd.Timestamp(date)

            for sym in fujiko_syms:
                df_sym = universe[sym]["df"]
                if date_ts not in df_sym.index:
                    continue
                past = df_sym.loc[:date_ts]
                if len(past) < min_bars:
                    continue

                # RSR
                rsr_s = rsr_uni[sym].reindex(past.index).ffill() if sym in rsr_uni.columns else pd.Series(50.0, index=past.index)
                rsr_now = float(rsr_s.iloc[-1]) if not rsr_s.empty else 50.0
                if pd.isna(rsr_now):
                    continue

                if rsr_now < MIN_RSR:
                    continue
                n_rsr += 1

                # SEPA
                sepa_df = calc_sepa(past, rsr_s)
                sepa_score = float(sepa_df["sepa_score"].iloc[-1])
                if pd.isna(sepa_score) or sepa_score < min_sepa:
                    continue
                n_sepa += 1

                # フルシグナル（RSRモメンタム + タートル）
                strat = FujikoStrategy(
                    rsr_series=rsr_uni[sym] if sym in rsr_uni.columns else None,
                    min_sepa=min_sepa, min_rsr=MIN_RSR,
                    mom_period=21, turtle_entry=20, turtle_exit=20,
                    use_turtle_entry=True,
                )
                sig = strat.generate_signal(past)
                if sig == 1:
                    n_sig += 1

            rsr_counts.append(n_rsr)
            sepa_counts.append(n_sepa)
            sig_counts.append(n_sig)

        rows.append({
            "min_sepa":        min_sepa,
            "N_universe":      n_universe,
            "RSR通過/日(avg)":  round(np.mean(rsr_counts), 2),
            "SEPA通過/日(avg)": round(np.mean(sepa_counts), 2),
            "シグナル/日(avg)": round(np.mean(sig_counts), 2),
            "RSR→SEPA通過率%":  round(np.mean(sepa_counts) / max(np.mean(rsr_counts), 1e-9) * 100, 1),
            "SEPA→シグナル率%": round(np.mean(sig_counts) / max(np.mean(sepa_counts), 1e-9) * 100, 1),
        })
        print(f"    sepa={min_sepa}: RSR通過={rows[-1]['RSR通過/日(avg)']:.2f}  "
              f"SEPA通過={rows[-1]['SEPA通過/日(avg)']:.2f}  "
              f"シグナル={rows[-1]['シグナル/日(avg)']:.2f}")

    return pd.DataFrame(rows)


# ================================================================
# MAIN
# ================================================================
def main():
    print("=" * 72)
    print("  sepa_sensitivity.py — min_sepa 感度分析")
    print(f"  RSR固定: {MIN_RSR:.0f}  /  期間: 2018〜2024  /  資本: ¥2,000,000")
    print("  ユニバース: 29銘柄 / 均等ウェイト / max_positions=3")
    print("=" * 72)

    # ---- 1. ユニバース構築 ----
    df_sel       = pd.read_csv("data/dynamic_selection.csv", encoding="utf-8-sig")
    sym_to_strat = dict(zip(df_sel["symbol"], df_sel["strategy"]))
    mask_27      = (df_sel["sharpe"] > 0.30) & (df_sel["maxdd"].abs() < 0.30)
    tickers_27   = {r["symbol"]: r["sector"] for _, r in df_sel[mask_27].iterrows()}
    all_tickers  = {**tickers_27, **FIXED_ADD}

    print(f"\n[1/4] データ取得中（{len(all_tickers)}銘柄）...")
    universe = download_universe(all_tickers, start=START, end=END, verbose=False)
    print(f"  取得完了: {len(universe)} 銘柄")

    prices_all = {sym: info["df"]["Close"] for sym, info in universe.items()}
    rsr_uni    = calc_universe_rsr(prices_all)
    print(f"  RSR計算完了: {rsr_uni.shape[1]} 銘柄")

    # ---- 2. ファンネル診断（先に実行して構造を把握） ----
    sepa_list = [3, 4, 5, 6]
    print("\n[2/4] エントリーファンネル診断（週次サンプリング）...")
    df_funnel = run_funnel_diagnostic(universe, rsr_uni, sym_to_strat, sepa_list)

    # ---- 3. バックテスト感度分析 ----
    print("\n[3/4] バックテスト実行中...")
    all_metrics = []
    for min_sepa in sepa_list:
        label = f"sepa={min_sepa} / RSR={MIN_RSR:.0f}"
        print(f"  [{label}]...", end="", flush=True)
        strats = build_strats(universe, rsr_uni, sym_to_strat, min_sepa)
        res    = run_portfolio(universe, strats)
        m      = compute_metrics(res, label)
        all_metrics.append(m)
        print(f"  CAGR={m['CAGR%']:+.2f}%  Calmar={m['Calmar']:.3f}"
              f"  avg_exp={m['avg_exp']:.3f}  zero_exp={m['zero_exp%']:.1f}%")

    # ---- 4. 結果表示 ----
    print("\n" + "=" * 72)
    print("  【結果①: エントリーファンネル診断】")
    print("=" * 72)
    print(df_funnel.to_string(index=False))

    print("\n" + "=" * 72)
    print("  【結果②: バックテスト感度分析（主要指標）】")
    print("=" * 72)
    primary = ["シナリオ", "CAGR%", "MaxDD%", "Calmar", "Sharpe",
               "avg_exp", "efficiency", "zero_exp%", "平均保有銘柄", "取引数"]
    df_bt = pd.DataFrame([{c: m[c] for c in primary} for m in all_metrics])
    print(df_bt.to_string(index=False))

    # ベースライン(sepa=6)との差分
    base = next(m for m in all_metrics if "sepa=6" in m["シナリオ"])
    print("\n  ■ ベースライン(sepa=6)との差分")
    print(f"  {'シナリオ':<22} {'ΔCAGR%':>8} {'ΔCalmar':>8} {'Δavg_exp':>9} {'Δzero_exp%':>11}")
    for m in all_metrics:
        if "sepa=6" in m["シナリオ"]:
            continue
        print(f"  {m['シナリオ']:<22}"
              f"  {m['CAGR%']-base['CAGR%']:>+7.2f}"
              f"  {m['Calmar']-base['Calmar']:>+7.3f}"
              f"  {m['avg_exp']-base['avg_exp']:>+8.3f}"
              f"  {m['zero_exp%']-base['zero_exp%']:>+10.1f}")

    # ---- 5. 判定サマリー ----
    best_calmar = max(all_metrics, key=lambda m: m["Calmar"])
    best_eff    = max(all_metrics, key=lambda m: m["efficiency"])
    print("\n  ■ 判定")
    print(f"  Calmar最大  : {best_calmar['シナリオ']}  Calmar={best_calmar['Calmar']:.3f}")
    print(f"  efficiency最大: {best_eff['シナリオ']}  eff={best_eff['efficiency']:.3f}")
    print(f"\n  ファンネルまとめ:")
    for _, row in df_funnel.iterrows():
        print(f"    sepa={int(row['min_sepa'])}: "
              f"RSR通過={row['RSR通過/日(avg)']:.1f}銘柄 → "
              f"SEPA通過={row['SEPA通過/日(avg)']:.1f}銘柄({row['RSR→SEPA通過率%']:.0f}%) → "
              f"シグナル={row['シグナル/日(avg)']:.1f}銘柄({row['SEPA→シグナル率%']:.0f}%)")

    # ---- 6. CSV保存 ----
    df_funnel.to_csv(os.path.join(OUTPUT, "sepa_funnel.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(all_metrics).to_csv(os.path.join(OUTPUT, "sepa_sensitivity.csv"),
                                     index=False, encoding="utf-8-sig")
    print(f"\n  → {OUTPUT}/sepa_funnel.csv")
    print(f"  → {OUTPUT}/sepa_sensitivity.csv")
    print("\n完了。")
    return all_metrics, df_funnel


if __name__ == "__main__":
    main()
