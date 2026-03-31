"""
backtest/step3_final_validation.py
Step 3: 最終バックテスト + OOS検証

【確定設定（感度分析で決定済み）】
  RSR = 75 / min_sepa = 6 / max_positions = 3 / 均等ウェイト
  ユニバース: V2固定 29銘柄

【3つの検証】
  1. ウォークフォワード : 2年IS + 1年OOS × 5セグメント（2020〜2024 OOS）
  2. 年別リターン分布  : yearly CAGR / Sharpe / MaxDD / 負け年カウント
  3. 市場レジーム耐性  : COVID暴落 / 2022ベア / 低ボラ期 の各DD・リカバリ速度
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
import matplotlib.patches as mpatches
plt.rcParams["font.family"] = "MS Gothic"

import numpy as np
import pandas as pd

from backtest.universe_builder        import download_universe
from backtest.rsr                     import calc_universe_rsr
from backtest.fujiko_strategy         import FujikoStrategy
from backtest.mean_reversion_strategy import MeanReversionStrategy
from backtest.portfolio_engine        import PortfolioEngine
from backtest.engine                  import TradeCost

# ------------------------------------------------------------------ #
# 確定設定（変更禁止）
# ------------------------------------------------------------------ #
MIN_RSR     = 75.0
MIN_SEPA    = 6
MAX_POS     = 3
CAPITAL     = 2_000_000
FULL_START  = "2016-01-01"   # WF IS期間の先頭余白を含む
FULL_END    = "2024-12-31"
OUTPUT      = "C:/ai-trading/reports"
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

# ウォークフォワードセグメント: (IS開始, IS終了, OOS開始, OOS終了)
WF_SEGMENTS = [
    ("2018-01-01", "2019-12-31", "2020-01-01", "2020-12-31"),  # Fold1 OOS=2020(COVID)
    ("2019-01-01", "2020-12-31", "2021-01-01", "2021-12-31"),  # Fold2 OOS=2021(強気)
    ("2020-01-01", "2021-12-31", "2022-01-01", "2022-12-31"),  # Fold3 OOS=2022(ベア)
    ("2021-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),  # Fold4 OOS=2023
    ("2022-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),  # Fold5 OOS=2024
]

# 市場レジーム定義
REGIMES = [
    ("COVID暴落", "2020-01-01", "2020-03-31", "red"),
    ("COVID回復", "2020-04-01", "2020-12-31", "green"),
    ("強気2021", "2021-01-01", "2021-12-31", "lime"),
    ("2022ベア",  "2022-01-01", "2022-12-31", "darkred"),
    ("2023回復", "2023-01-01", "2023-12-31", "teal"),
    ("2024強気", "2024-01-01", "2024-12-31", "steelblue"),
]


# ------------------------------------------------------------------ #
# ユーティリティ
# ------------------------------------------------------------------ #
def _cagr(equity: pd.Series, initial: float) -> float:
    n = (equity.index[-1] - equity.index[0]).days / 365.25
    return float((equity.iloc[-1] / initial) ** (1 / n) - 1) if n > 0 else 0.0

def _sharpe(equity: pd.Series) -> float:
    r = equity.pct_change().dropna()
    return float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0.0

def _mdd(equity: pd.Series) -> float:
    peak = equity.cummax()
    return float(((equity - peak) / peak).min())

def _calmar(cagr: float, mdd: float) -> float:
    return cagr / abs(mdd) if mdd < 0 else float("inf")

def build_strats(universe: dict, rsr_uni: pd.DataFrame, sym_to_strat: dict) -> dict:
    strat_dict = {}
    for sym, info in universe.items():
        rsr_s = rsr_uni[sym] if sym in rsr_uni.columns else None
        rule  = SECTOR_STRATEGY.get(info["sector"], "dynamic")
        lbl   = sym_to_strat.get(sym, "フジコ法")
        if rule == "mean_rev" or (rule == "dynamic" and "平均回帰" in lbl):
            strat_dict[sym] = MeanReversionStrategy(**MR_PARAMS)
        else:
            strat_dict[sym] = FujikoStrategy(
                rsr_series=rsr_s, min_sepa=MIN_SEPA, min_rsr=MIN_RSR,
                mom_period=21, turtle_entry=20, turtle_exit=20,
                use_turtle_entry=True,
            )
    return strat_dict

def run_portfolio(universe: dict, strat_dict: dict) -> object:
    return PortfolioEngine(
        universe=universe, strategy=strat_dict, capital=CAPITAL,
        max_dd_limit=0.15, min_sectors=1, max_positions=MAX_POS,
        cost=TradeCost(), vol_target=0.0, max_single_weight=0.25,
        vol_window=20, use_idm=False, use_dd_scalar=False,
    ).run()


# ================================================================
# MAIN
# ================================================================
def main():
    print("=" * 72)
    print("  step3_final_validation.py — 最終バックテスト + OOS検証")
    print(f"  確定設定: RSR={MIN_RSR:.0f} / sepa={MIN_SEPA} / max_pos={MAX_POS} / 均等ウェイト")
    print("=" * 72)

    # ---- 0. データ取得（2016年から：WF IS期間の先行余白） ----
    df_sel       = pd.read_csv("data/dynamic_selection.csv", encoding="utf-8-sig")
    sym_to_strat = dict(zip(df_sel["symbol"], df_sel["strategy"]))
    mask_27      = (df_sel["sharpe"] > 0.30) & (df_sel["maxdd"].abs() < 0.30)
    tickers_27   = {r["symbol"]: r["sector"] for _, r in df_sel[mask_27].iterrows()}
    all_tickers  = {**tickers_27, **FIXED_ADD}

    print(f"\n[0/4] データ取得中（{len(all_tickers)}銘柄, {FULL_START}〜{FULL_END}）...")
    universe = download_universe(all_tickers, start=FULL_START, end=FULL_END, verbose=False)
    print(f"  取得完了: {len(universe)} 銘柄")
    prices_all = {sym: info["df"]["Close"] for sym, info in universe.items()}
    rsr_uni    = calc_universe_rsr(prices_all)
    print(f"  RSR計算完了: {rsr_uni.shape[1]} 銘柄")

    # ---- 1. フルperiodバックテスト（2018-2024） ----
    print("\n[1/4] フル期間バックテスト（2018-2024）...")
    univ_full = {
        sym: {"df": info["df"].loc["2018-01-01":"2024-12-31"], "sector": info["sector"]}
        for sym, info in universe.items()
        if not info["df"].loc["2018-01-01":"2024-12-31"].empty
    }
    strats_full = build_strats(univ_full, rsr_uni, sym_to_strat)
    res_full    = run_portfolio(univ_full, strats_full)

    cagr_full  = res_full.cagr
    mdd_full   = res_full.max_drawdown
    cal_full   = _calmar(cagr_full, mdd_full)
    shar_full  = res_full.sharpe_ratio
    exp_full   = float(res_full.exposure_series.mean())
    zero_full  = float((res_full.exposure_series == 0).mean())
    print(f"  CAGR={cagr_full*100:+.2f}%  MaxDD={mdd_full*100:.2f}%  "
          f"Calmar={cal_full:.3f}  Sharpe={shar_full:.3f}")
    print(f"  avg_exp={exp_full:.3f}  zero_exp={zero_full*100:.1f}%  trades={res_full.n_trades}")

    # ---- 2. 年別リターン分布 ----
    print("\n[2/4] 年別リターン分布...")
    eq   = res_full.equity_curve
    dd_s = res_full.dd_series
    exp_s = res_full.exposure_series

    yearly_rows = []
    for yr in range(2018, 2025):
        ye = eq[eq.index.year == yr]
        yd = dd_s[dd_s.index.year == yr]
        ye_exp = exp_s[exp_s.index.year == yr]
        if ye.empty:
            continue
        init = ye.iloc[0]
        ret  = float(ye.iloc[-1] / init - 1)
        mdd  = float(yd.min())
        shr  = _sharpe(ye)
        cal  = _calmar(ret, mdd) if mdd < 0 else None
        avg_exp = float(ye_exp.mean())
        yearly_rows.append({
            "年":      yr,
            "CAGR%":  round(ret * 100, 2),
            "MaxDD%": round(mdd * 100, 2),
            "Sharpe": round(shr, 3),
            "Calmar": round(cal, 2) if cal else "∞",
            "avg_exp": round(avg_exp, 3),
            "判定":   "✅" if ret > 0 else "❌",
        })
    df_yearly = pd.DataFrame(yearly_rows)

    losing_yrs = int((df_yearly["CAGR%"] < 0).sum())
    total_yrs  = len(df_yearly)
    lose_rate  = losing_yrs / total_yrs
    print(f"\n  年別サマリー:")
    print(df_yearly.to_string(index=False))
    print(f"\n  負け年: {losing_yrs}/{total_yrs} ({lose_rate*100:.0f}%)  "
          f"基準: ≤30% → {'✅ OK' if lose_rate <= 0.30 else '❌ NG'}")

    # ---- 3. ウォークフォワード（5セグメント） ----
    print("\n[3/4] ウォークフォワード検証（2yr IS + 1yr OOS × 5セグメント）...")
    wf_rows = []
    oos_equities = {}   # OOS期間の equity curve をチャート用に保存

    for fold_i, (is_s, is_e, oos_s, oos_e) in enumerate(WF_SEGMENTS, start=1):
        print(f"\n  [Fold {fold_i}] IS:{is_s[:7]}〜{is_e[:7]}  OOS:{oos_s[:7]}〜{oos_e[:7]}")

        # IS期間（パラメータ固定 = 単なる参照期間）
        univ_is = {
            sym: {"df": info["df"].loc[is_s:is_e], "sector": info["sector"]}
            for sym, info in universe.items()
            if not info["df"].loc[is_s:is_e].empty
        }
        strats_is = build_strats(univ_is, rsr_uni, sym_to_strat)
        res_is    = run_portfolio(univ_is, strats_is)
        is_cagr   = res_is.cagr
        is_shar   = res_is.sharpe_ratio
        is_mdd    = res_is.max_drawdown

        # OOS期間（真のOOS = IS期間と独立して実行）
        univ_oos = {
            sym: {"df": info["df"].loc[oos_s:oos_e], "sector": info["sector"]}
            for sym, info in universe.items()
            if not info["df"].loc[oos_s:oos_e].empty
        }
        strats_oos = build_strats(univ_oos, rsr_uni, sym_to_strat)
        res_oos    = run_portfolio(univ_oos, strats_oos)
        oos_cagr   = res_oos.cagr
        oos_shar   = res_oos.sharpe_ratio
        oos_mdd    = res_oos.max_drawdown
        oos_avg_exp = float(res_oos.exposure_series.mean())

        # IS vs OOS 比率
        ratio = oos_shar / is_shar if is_shar > 0 else float("nan")
        ok    = "✅" if ratio >= 0.70 else ("⚠️" if ratio >= 0.50 else "❌")

        oos_equities[f"Fold{fold_i}(OOS {oos_s[:4]})"] = res_oos.equity_curve

        print(f"    IS : CAGR={is_cagr*100:+.2f}%  Sharpe={is_shar:.3f}  MaxDD={is_mdd*100:.2f}%")
        print(f"    OOS: CAGR={oos_cagr*100:+.2f}%  Sharpe={oos_shar:.3f}  MaxDD={oos_mdd*100:.2f}%"
              f"  avg_exp={oos_avg_exp:.3f}")
        print(f"    OOS/IS Sharpe比: {ratio:.2f}  {ok}")

        wf_rows.append({
            "Fold":       fold_i,
            "OOS年":      oos_s[:4],
            "IS_CAGR%":   round(is_cagr * 100, 2),
            "IS_Sharpe":  round(is_shar, 3),
            "IS_MaxDD%":  round(is_mdd * 100, 2),
            "OOS_CAGR%":  round(oos_cagr * 100, 2),
            "OOS_Sharpe": round(oos_shar, 3),
            "OOS_MaxDD%": round(oos_mdd * 100, 2),
            "OOS_avgexp": round(oos_avg_exp, 3),
            "OOS/IS比":   round(ratio, 2) if not np.isnan(ratio) else "N/A",
            "判定":       ok,
        })

    df_wf = pd.DataFrame(wf_rows)
    oos_sharpes = [r["OOS_Sharpe"] for r in wf_rows]
    is_sharpes  = [r["IS_Sharpe"]  for r in wf_rows]
    avg_ratio   = np.mean([r for r in [row["OOS/IS比"] for row in wf_rows]
                           if isinstance(r, float)])
    neg_oos     = sum(1 for r in wf_rows if r["OOS_CAGR%"] < 0)

    print(f"\n  ウォークフォワード サマリー:")
    print(df_wf.to_string(index=False))
    print(f"\n  OOS平均Sharpe: {np.mean(oos_sharpes):.3f}")
    print(f"  IS平均Sharpe : {np.mean(is_sharpes):.3f}")
    print(f"  OOS/IS比 平均: {avg_ratio:.2f}  基準:≥0.70 → {'✅ OK' if avg_ratio >= 0.70 else ('⚠️ 要注意' if avg_ratio >= 0.50 else '❌ 過学習疑い')}")
    print(f"  OOS負けセグメント: {neg_oos}/5")

    # ---- 4. 市場レジーム耐性 ----
    print("\n[4/4] 市場レジーム別分析...")
    regime_rows = []
    for name, rs, re, clr in REGIMES:
        ye = eq.loc[rs:re]
        yd = dd_s.loc[rs:re]
        ye_exp = exp_s.loc[rs:re]
        if ye.empty:
            continue
        ret  = float(ye.iloc[-1] / ye.iloc[0] - 1)
        mdd  = float(yd.min())
        shr  = _sharpe(ye)
        avg_exp_r = float(ye_exp.mean())
        zero_r    = float((ye_exp == 0).mean())
        # DD回復: DDがゼロに戻るまでの日数
        dd_neg_days = int((yd < -0.01).sum())

        regime_rows.append({
            "レジーム":   name,
            "期間":       f"{rs[:7]}〜{re[:7]}",
            "リターン%":  round(ret * 100, 2),
            "MaxDD%":    round(mdd * 100, 2),
            "Sharpe":    round(shr, 3),
            "avg_exp":   round(avg_exp_r, 3),
            "zero_exp%": round(zero_r * 100, 1),
            "DD日数":    dd_neg_days,
        })
        print(f"  {name:<10} ({rs[:7]}〜{re[:7]}): "
              f"ret={ret*100:+.1f}%  MaxDD={mdd*100:.1f}%  "
              f"Sharpe={shr:.2f}  avg_exp={avg_exp_r:.2f}  zero={zero_r*100:.0f}%")

    df_regime = pd.DataFrame(regime_rows)

    # ---- 5. チャート出力 ----
    print("\n  チャート生成中...")
    fig, axes = plt.subplots(3, 1, figsize=(16, 14), sharex=False)
    fig.suptitle(
        "Step3 最終OOS検証 — RSR=75 / sepa=6 / max_pos=3 / 均等ウェイト\n"
        "V2固定29銘柄 / 2018-2024",
        fontsize=11, fontweight="bold"
    )

    # ① フル期間 資産曲線 + レジーム帯
    ax = axes[0]
    ax.plot(eq.index, eq.values / 10_000, color="steelblue", linewidth=2,
            label=f"フル期間  CAGR={cagr_full*100:+.2f}%  Calmar={cal_full:.3f}  Sharpe={shar_full:.3f}")
    ax.axhline(CAPITAL / 10_000, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    # レジーム帯を背景色で表示
    for name, rs, re, clr in REGIMES:
        try:
            ax.axvspan(pd.Timestamp(rs), pd.Timestamp(re), alpha=0.07, color=clr, label=name)
        except Exception:
            pass
    ax.set_ylabel("資産（万円）")
    ax.set_title("① フル期間 資産推移 + 市場レジーム")
    ax.legend(fontsize=7, ncol=2)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"¥{x:.0f}万"))
    ax.grid(True, alpha=0.2)

    # ② ウォークフォワード OOS エクイティ（5セグメント連結）
    ax = axes[1]
    colors_wf = ["royalblue","darkorange","green","crimson","purple"]
    for (lbl, eq_oos), clr in zip(oos_equities.items(), colors_wf):
        fold_idx = list(oos_equities.keys()).index(lbl)
        row = wf_rows[fold_idx]
        norm = eq_oos / eq_oos.iloc[0]  # 期間先頭を1に正規化して比較
        ax.plot(eq_oos.index, norm, color=clr, linewidth=1.8,
                label=f"{lbl}  OOS={row['OOS_CAGR%']:+.1f}%  Sharpe={row['OOS_Sharpe']:.2f}")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_ylabel("正規化資産（期間先頭=1）")
    ax.set_title("② ウォークフォワード OOS 各セグメント（期間先頭を1に正規化）")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # ③ OOS/IS Sharpe比 棒グラフ
    ax = axes[2]
    folds   = [f"Fold{r['Fold']}\n(OOS {r['OOS年']})" for r in wf_rows]
    oos_shr = [r["OOS_Sharpe"] for r in wf_rows]
    is_shr  = [r["IS_Sharpe"]  for r in wf_rows]
    x = np.arange(len(folds))
    w = 0.35
    bars_oos = ax.bar(x - w/2, oos_shr, w, label="OOS Sharpe", color="steelblue", alpha=0.85)
    bars_is  = ax.bar(x + w/2, is_shr,  w, label="IS Sharpe",  color="lightgray", alpha=0.85)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axhline(1.0, color="green", linestyle="--", linewidth=1.0, alpha=0.6, label="Sharpe=1.0")
    # OOS/IS比を各バーの上に表示
    for xi, row in zip(x, wf_rows):
        ratio_val = row["OOS/IS比"]
        if isinstance(ratio_val, float):
            ax.annotate(f"{ratio_val:.2f}", xy=(xi - w/2, max(row["OOS_Sharpe"], 0) + 0.05),
                        ha="center", fontsize=8,
                        color="steelblue" if ratio_val >= 0.70 else "red")
    ax.set_xticks(x)
    ax.set_xticklabels(folds, fontsize=9)
    ax.set_ylabel("Sharpe比")
    ax.set_title("③ ウォークフォワード IS vs OOS Sharpe（上の数値 = OOS/IS比）")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    out_png = os.path.join(OUTPUT, "step3_final_validation.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out_png}")

    # ---- 6. 最終判定サマリー ----
    print("\n" + "=" * 72)
    print("  【Step3 最終判定】")
    print("=" * 72)

    checks = [
        ("フル期間 Sharpe",    shar_full >= 1.3,   f"{shar_full:.3f}",     "≥1.3"),
        ("フル期間 Calmar",    cal_full  >= 1.2,   f"{cal_full:.3f}",      "≥1.2"),
        ("フル期間 MaxDD",     abs(mdd_full) < 0.15, f"{mdd_full*100:.1f}%","<15%"),
        ("負け年割合",         lose_rate <= 0.30,  f"{lose_rate*100:.0f}%", "≤30%"),
        ("WF OOS/IS Sharpe比", avg_ratio >= 0.70,  f"{avg_ratio:.2f}",      "≥0.70"),
        ("WF OOS負けセグメント", neg_oos <= 1,    f"{neg_oos}/5",          "≤1/5"),
    ]

    all_ok = True
    for name, ok, val, crit in checks:
        mark = "✅" if ok else "❌"
        if not ok:
            all_ok = False
        print(f"  {mark} {name:<25} {val:>10}  (基準: {crit})")

    print()
    if all_ok:
        print("  🎯 全チェック通過 → Step3 完了。Phase2実運用移行の準備が整った。")
    else:
        fails = [name for name, ok, _, _ in checks if not ok]
        print(f"  ⚠️  未通過: {', '.join(fails)}")
        print("  → 未通過項目を精査してからPhase2移行を判断すること。")

    # ---- 7. CSV保存 ----
    df_yearly.to_csv(os.path.join(OUTPUT, "step3_yearly.csv"),   index=False, encoding="utf-8-sig")
    df_wf.to_csv(    os.path.join(OUTPUT, "step3_walkforward.csv"), index=False, encoding="utf-8-sig")
    df_regime.to_csv(os.path.join(OUTPUT, "step3_regime.csv"),   index=False, encoding="utf-8-sig")

    print(f"\n  CSV: {OUTPUT}/step3_yearly.csv")
    print(f"  CSV: {OUTPUT}/step3_walkforward.csv")
    print(f"  CSV: {OUTPUT}/step3_regime.csv")
    print("\n完了。")

    return {
        "full": {"cagr": cagr_full, "calmar": cal_full, "sharpe": shar_full, "mdd": mdd_full},
        "yearly": df_yearly,
        "walkforward": df_wf,
        "regime": df_regime,
    }


if __name__ == "__main__":
    main()
