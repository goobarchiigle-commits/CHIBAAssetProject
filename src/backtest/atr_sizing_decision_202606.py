"""
backtest/atr_sizing_decision_202606.py
ATR Sizing 存廃判定（維持/修正/廃止）。WF実施禁止・推測禁止・実測のみ。
一回限りの監査スクリプト（恒久モジュール化しない）。

期間: 2018-01-01〜2026-06-23（既存ATR診断/感度分析と同一、一貫性確保）
risk_pct最適化・WF・新規戦略研究は対象外。サイジング方式の存廃判定のみに集中。

実施内容:
  STEP1: ATR%四分位ごとの期待値分析（trade count/win rate/avg return/median return/PF/CAGR contribution）
  STEP2: サイジング方式アブレーション A(Equal Weight)/B(Current ATR)/C(ATR_SQRT)/D(ATR_025)/E(No ATR)
         同一Universe・同一Exit（PROD_FAITHFUL: ATRトレーリング+多層RSR+MTF）、サイジングのみ変更
  STEP3: 各方式の利益上位20トレードの損益寄与率・平均ポジションサイズ・総利益寄与
  STEP4: 最終判定（KEEP/WEAKEN/REMOVE）

実行: python src/backtest/atr_sizing_decision_202606.py
"""
from __future__ import annotations
import os, sys, json, warnings, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import src.backtest.composite_alpha_bt as bt
from src.backtest.atr_sizing_diagnostic_202606 import load_all, atr_at_signal_day

FULL_START = "2018-01-01"
FULL_END   = "2026-06-23"
OUTPUT_JSON = f"C:/ai-trading/backtests/atr_sizing_decision_202606_{time.strftime('%Y-%m-%d')}.json"

METHODS = ["A_EQUAL", "B_CURRENT_ATR", "C_ATR_SQRT", "D_ATR_025", "E_NO_ATR"]


def run_method(cfg, universe_raw, rsr_df, alpha_df, regime_df, trade_syms, rsr_syms,
                sym_active_df, tech_matrices, method: str):
    """全方式: 同一Universe(動的ユニバース有効)・同一Exit(ATRトレーリング+多層RSR+MTF)。サイジングのみ変更。"""
    common_kwargs = dict(
        scenario="PROD_FAITHFUL",
        universe_raw=universe_raw, rsr_df=rsr_df, alpha_df=alpha_df, regime_df=regime_df,
        trade_syms=trade_syms, rsr_syms=rsr_syms, cfg=cfg,
        start=FULL_START, end=FULL_END, verbose=False,
        tech_matrices=tech_matrices,
        capital=cfg.portfolio.capital,
        min_hold=cfg.risk.min_hold_days,
        market_shock_mode="composite",
        rsr_exit_threshold=70.0,
        sym_active_df=sym_active_df,
        enable_atr_trailing_prod=True,
        enable_multilayer_rsr=True,
        enable_mtf_filter=True,
    )
    if method == "A_EQUAL":
        return bt.run_scenario(**common_kwargs, enable_atr_risk_sizing=False, sizing_mode="equal")
    if method == "B_CURRENT_ATR":
        return bt.run_scenario(**common_kwargs, enable_atr_risk_sizing=True,
                                risk_sizing_pct=bt.PROD_RISK_PCT, atr_sizing_exponent=1.0)
    if method == "C_ATR_SQRT":
        return bt.run_scenario(**common_kwargs, enable_atr_risk_sizing=True,
                                risk_sizing_pct=bt.PROD_RISK_PCT, atr_sizing_exponent=0.5)
    if method == "D_ATR_025":
        return bt.run_scenario(**common_kwargs, enable_atr_risk_sizing=True,
                                risk_sizing_pct=bt.PROD_RISK_PCT, atr_sizing_exponent=0.25)
    if method == "E_NO_ATR":
        return bt.run_scenario(**common_kwargs, enable_atr_risk_sizing=True, atr_sizing_no_risk=True)
    raise ValueError(method)


def main():
    cfg, universe_raw, rsr_df, alpha_df, regime_df, trade_syms, rsr_syms, sym_active_df, tech_matrices = load_all()
    capital = cfg.portfolio.capital
    years = (pd.Timestamp(FULL_END) - pd.Timestamp(FULL_START)).days / 365.25

    active_syms = [s for s in trade_syms if s in universe_raw]
    common_dates = None
    for s in active_syms:
        idx = universe_raw[s]["df"].index
        common_dates = idx if common_dates is None else common_dates.intersection(idx)
    common_dates = common_dates.sort_values()
    ts_f, te_f = pd.Timestamp(FULL_START), pd.Timestamp(FULL_END)
    common_dates = common_dates[(common_dates >= ts_f) & (common_dates <= te_f)]

    print(f"\n{'='*78}\n  [Part0] サイジング方式アブレーション 実行（期間: {FULL_START}〜{FULL_END}）\n{'='*78}")
    results = {}
    for m in METHODS:
        t0 = time.time()
        res = run_method(cfg, universe_raw, rsr_df, alpha_df, regime_df, trade_syms, rsr_syms,
                          sym_active_df, tech_matrices, m)
        results[m] = res
        print(f"  {m:15s}: CAGR={res['cagr']:+6.1f}%  Sharpe={res['sharpe']:.3f}  MaxDD={res['max_dd']:+6.1f}%  "
              f"Calmar={res['calmar']:.3f}  PF={res['profit_factor']:.3f}  avgExp={res['avg_exposure']:.1f}%  "
              f"trades={res['n_trades']}  ({time.time()-t0:.0f}s)")

    # ---------------------------------------------------------------- #
    # [STEP1] ATR%四分位ごとの期待値分析（Method A=Equal Weightのトレード集合を使用：
    #         サイジングに依存しない「素のトレード」でATR%と期待値の関係を見る）
    # ---------------------------------------------------------------- #
    print(f"\n{'='*78}\n  [STEP1] ATR%四分位 期待値分析（Method A: Equal Weightトレード集合）\n{'='*78}")
    a_trades = results["A_EQUAL"]["_trades"]
    enriched_a = []
    for t in a_trades:
        if not t.get("entry") or t.get("exit") is None or "entry_idx" not in t:
            continue
        atr_val = atr_at_signal_day(tech_matrices, common_dates, t["symbol"], t["entry_idx"])
        if np.isnan(atr_val) or t["entry"] <= 0:
            continue
        atr_pct = atr_val / t["entry"] * 100
        return_pct = (t["exit"] / t["entry"] - 1) * 100
        enriched_a.append({"symbol": t["symbol"], "atr_pct": atr_pct, "return_pct": return_pct, "pnl": t["pnl"]})

    df_a = pd.DataFrame(enriched_a)
    df_a["quartile"] = pd.qcut(df_a["atr_pct"], 4, labels=["Q1(低ATR%)", "Q2", "Q3", "Q4(高ATR%)"])

    step1_rows = []
    for q in ["Q1(低ATR%)", "Q2", "Q3", "Q4(高ATR%)"]:
        grp = df_a[df_a["quartile"] == q]
        n = len(grp)
        wins = grp[grp["pnl"] > 0]
        loses = grp[grp["pnl"] <= 0]
        win_rate = len(wins) / n * 100 if n else 0.0
        gross_profit = float(wins["pnl"].sum())
        gross_loss = float(abs(loses["pnl"].sum()))
        pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        total_pnl = float(grp["pnl"].sum())
        cagr_contrib_total_pct = total_pnl / capital * 100
        cagr_contrib_annualized_pct = cagr_contrib_total_pct / years  # 線形近似（複利でない）
        row = {
            "quartile": q, "n": n, "atr_pct_range": [round(float(grp["atr_pct"].min()), 2), round(float(grp["atr_pct"].max()), 2)],
            "win_rate": round(win_rate, 1), "avg_return_pct": round(float(grp["return_pct"].mean()), 2),
            "median_return_pct": round(float(grp["return_pct"].median()), 2),
            "pf": round(pf, 3) if pf != float("inf") else pf,
            "total_pnl_yen": round(total_pnl, 0),
            "cagr_contrib_total_pct": round(cagr_contrib_total_pct, 2),
            "cagr_contrib_annualized_pct_linear": round(cagr_contrib_annualized_pct, 2),
        }
        step1_rows.append(row)
        print(f"  {q:12s} n={n:4d}  win_rate={win_rate:5.1f}%  avg_ret={row['avg_return_pct']:+6.2f}%  "
              f"median_ret={row['median_return_pct']:+6.2f}%  PF={row['pf']!s:>6}  "
              f"total_pnl=¥{total_pnl:>12,.0f}  CAGR寄与(年率線形近似)={cagr_contrib_annualized_pct:+5.2f}%")

    # ---------------------------------------------------------------- #
    # [STEP2] サマリー表（既に上で実行・出力済み、JSON保存用に整形）
    # ---------------------------------------------------------------- #
    step2_summary = {
        m: {k: results[m][k] for k in ("cagr", "sharpe", "max_dd", "calmar", "profit_factor", "avg_exposure", "n_trades")}
        for m in METHODS
    }

    # ---------------------------------------------------------------- #
    # [STEP3] 各方式: 利益上位20トレードの損益寄与率・平均サイズ・総利益寄与
    # ---------------------------------------------------------------- #
    print(f"\n{'='*78}\n  [STEP3] 方式別 利益上位20トレード寄与分析\n{'='*78}")
    step3_rows = {}
    for m in METHODS:
        trades = [t for t in results[m]["_trades"] if t.get("pnl") is not None and t.get("qty") and t.get("entry")]
        win_trades = [t for t in trades if t["pnl"] > 0]
        gross_profit_all = float(sum(t["pnl"] for t in win_trades))
        top20 = sorted(win_trades, key=lambda t: -t["pnl"])[:20]
        top20_profit = float(sum(t["pnl"] for t in top20))
        top20_avg_size = float(np.mean([t["qty"] * t["entry"] for t in top20])) if top20 else 0.0
        contrib_pct = (top20_profit / gross_profit_all * 100) if gross_profit_all > 0 else 0.0
        step3_rows[m] = {
            "n_winners_total": len(win_trades),
            "gross_profit_all_yen": round(gross_profit_all, 0),
            "top20_profit_yen": round(top20_profit, 0),
            "top20_contribution_pct_of_gross_profit": round(contrib_pct, 1),
            "top20_avg_position_size_yen": round(top20_avg_size, 0),
        }
        print(f"  {m:15s}  top20利益=¥{top20_profit:>13,.0f}  寄与率={contrib_pct:5.1f}%  "
              f"top20平均サイズ=¥{top20_avg_size:>10,.0f}  (全勝ちトレード総利益=¥{gross_profit_all:,.0f})")

    # ---------------------------------------------------------------- #
    # [STEP4] 最終判定
    # ---------------------------------------------------------------- #
    A, B, C, D, E = (step2_summary[m] for m in METHODS)

    # 判定A: KEEP — BがAに対しSharpeまたはCalmarで明確改善（ここでは閾値+0.05を「明確」とする）
    keep_cond = (B["sharpe"] - A["sharpe"] >= 0.05) or (B["calmar"] - A["calmar"] >= 0.05)
    # 判定B: WEAKEN — CまたはDがBに対しCAGRとSharpeを同時改善
    weaken_cond_c = (C["cagr"] > B["cagr"]) and (C["sharpe"] > B["sharpe"])
    weaken_cond_d = (D["cagr"] > B["cagr"]) and (D["sharpe"] > B["sharpe"])
    weaken_cond = weaken_cond_c or weaken_cond_d
    # 判定C: REMOVE — AまたはEがBに対しCAGR・Sharpe・PFの全てで優位
    remove_cond_a = (A["cagr"] > B["cagr"]) and (A["sharpe"] > B["sharpe"]) and (A["profit_factor"] > B["profit_factor"])
    remove_cond_e = (E["cagr"] > B["cagr"]) and (E["sharpe"] > B["sharpe"]) and (E["profit_factor"] > B["profit_factor"])
    remove_cond = remove_cond_a or remove_cond_e

    if remove_cond and not keep_cond:
        decision = "REMOVE"
        winner = "A_EQUAL" if remove_cond_a and (not remove_cond_e or A["cagr"] >= E["cagr"]) else "E_NO_ATR"
        reason = (
            f"{winner}がBに対しCAGR/Sharpe/PFの全指標で優位({winner}: CAGR{step2_summary[winner]['cagr']:+.1f}%/"
            f"Sharpe{step2_summary[winner]['sharpe']:.3f}/PF{step2_summary[winner]['profit_factor']:.3f} vs "
            f"B: CAGR{B['cagr']:+.1f}%/Sharpe{B['sharpe']:.3f}/PF{B['profit_factor']:.3f})"
        )
        confidence = "HIGH" if (remove_cond_a and remove_cond_e) else "MEDIUM"
    elif weaken_cond and not keep_cond:
        decision = "WEAKEN"
        winner = "C_ATR_SQRT" if weaken_cond_c and (not weaken_cond_d or C["sharpe"] >= D["sharpe"]) else "D_ATR_025"
        reason = (
            f"{winner}がBに対しCAGR/Sharpe同時改善({winner}: CAGR{step2_summary[winner]['cagr']:+.1f}%/"
            f"Sharpe{step2_summary[winner]['sharpe']:.3f} vs B: CAGR{B['cagr']:+.1f}%/Sharpe{B['sharpe']:.3f})"
        )
        confidence = "HIGH" if (weaken_cond_c and weaken_cond_d) else "MEDIUM"
    elif keep_cond:
        decision = "KEEP"
        reason = (
            f"BがAに対しSharpe({B['sharpe']:.3f} vs {A['sharpe']:.3f})またはCalmar"
            f"({B['calmar']:.3f} vs {A['calmar']:.3f})で明確改善（閾値+0.05）"
        )
        confidence = "MEDIUM"
    else:
        decision = "WEAKEN"
        winner = "C_ATR_SQRT" if C["sharpe"] >= D["sharpe"] else "D_ATR_025"
        reason = (
            f"KEEP/REMOVE基準を満たす方式なし。{winner}がB付近で最良の折衷"
            f"(Sharpe{step2_summary[winner]['sharpe']:.3f} vs B{B['sharpe']:.3f})のため弱化を暫定推奨"
        )
        confidence = "LOW"

    print(f"\n{'='*78}\n  [STEP4] 最終判定\n{'='*78}")
    print(f"  decision = {decision}")
    print(f"  reason = {reason}")
    print(f"  confidence = {confidence}")

    out = {
        "period": [FULL_START, FULL_END],
        "step2_summary": step2_summary,
        "step1_atr_quartile": step1_rows,
        "step3_top20_contribution": step3_rows,
        "decision": decision, "reason": reason, "confidence": confidence,
    }
    os.makedirs("C:/ai-trading/backtests", exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n結果保存: {OUTPUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
