"""
backtest/study34_drawdown_attribution_202606.py
Study34: Drawdown Attribution Audit。推測禁止・コード実測のみ。
一回限りの監査スクリプト（恒久モジュール化しない）。

目的: APRIL_REPRO(-15.7%) → PROD_MINUS_ATR(-19.4%) で増加したDDの原因を定量特定。
比較対象はStudy33と同一定義（A=APRIL_REPRO sizing_mode=existing / C=PROD_MINUS_ATR sizing_mode=equal）。

実施:
  1. 最大DD期間特定（開始/終了/回復日）
  2. DD期間中の損失寄与Top20銘柄
  3. ポートフォリオ分析（平均保有数・平均相関・セクター集中度・クラスター集中度）
  4. 各機能単独寄与（Universe/ATR Trailing/MultiRSR、+MTF・+sizing切替は完全性のため追加表示）のDD変化
  5. 年別DD（2020-2026、年内ローカルmax DD）
  6. 判定: DD増加の主因をA動的ユニバース/B集中化/C Exit変更/D特定年度/Eその他へ定量帰属

実行: python src/backtest/study34_drawdown_attribution_202606.py
"""
from __future__ import annotations
import os, sys, json, warnings, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import src.backtest.composite_alpha_bt as bt
from src.backtest.atr_sizing_diagnostic_202606 import load_all
from src.strategy.cluster import CLUSTER_MAP_DEFAULT, get_cluster

FULL_START = "2018-01-01"
FULL_END   = "2026-06-23"
OUTPUT_JSON = f"C:/ai-trading/backtests/study34_drawdown_attribution_202606_{time.strftime('%Y-%m-%d')}.json"


def run(cfg, universe_raw, rsr_df, alpha_df, regime_df, trade_syms, rsr_syms, tech_matrices,
        sym_active_df, *, use_dyn, shock, rsr_exit, atr_trail, ml, sizing, mtf, sizing_mode):
    return bt.run_scenario(
        scenario="PROD_FAITHFUL" if (atr_trail or ml or sizing or mtf) else "BASELINE",
        universe_raw=universe_raw, rsr_df=rsr_df, alpha_df=alpha_df, regime_df=regime_df,
        trade_syms=trade_syms, rsr_syms=rsr_syms, cfg=cfg,
        start=FULL_START, end=FULL_END, verbose=False,
        tech_matrices=tech_matrices,
        capital=cfg.portfolio.capital,
        min_hold=cfg.risk.min_hold_days,
        market_shock_mode=shock,
        rsr_exit_threshold=rsr_exit,
        sym_active_df=(sym_active_df if use_dyn else None),
        enable_atr_trailing_prod=atr_trail,
        enable_multilayer_rsr=ml,
        enable_atr_risk_sizing=sizing,
        enable_mtf_filter=mtf,
        risk_sizing_pct=bt.PROD_RISK_PCT,
        sizing_mode=sizing_mode,
    )


def max_dd_window(eq: pd.Series, dd: pd.Series, recovery_days):
    trough_date = dd.idxmin()
    trough_pos = dd.index.get_loc(trough_date)
    peak_pos = int(np.argmax(eq.iloc[:trough_pos + 1].values))
    peak_date = eq.index[peak_pos]
    recovery_date = None
    if recovery_days is not None:
        rec_pos = trough_pos + int(recovery_days)
        if rec_pos < len(dd.index):
            recovery_date = dd.index[rec_pos]
    return peak_pos, trough_pos, peak_date, trough_date, recovery_date


def holding_sets(trades, n_days):
    """entry_idx<=i<exit_idx の日に保有していた銘柄集合を日次で構築（SELLトレードのみ・確定済み区間のみ）。"""
    held = [set() for _ in range(n_days)]
    for t in trades:
        if "entry_idx" not in t or "exit_idx" not in t:
            continue
        for i in range(max(0, t["entry_idx"]), min(n_days, t["exit_idx"])):
            held[i].add(t["symbol"])
    return held


def portfolio_concentration(held, trade_syms, corr_mat: pd.DataFrame):
    avg_holdings_list, corr_list, sector_hhi_list, cluster_hhi_list = [], [], [], []
    for syms in held:
        n = len(syms)
        avg_holdings_list.append(n)
        if n == 0:
            continue
        if n >= 2:
            syms_l = list(syms)
            pairs = [corr_mat.loc[s1, s2] for idx1, s1 in enumerate(syms_l) for s2 in syms_l[idx1 + 1:]
                     if s1 in corr_mat.index and s2 in corr_mat.columns]
            pairs = [p for p in pairs if pd.notna(p)]
            if pairs:
                corr_list.append(float(np.mean(pairs)))
        sectors = [trade_syms.get(s, "不明") for s in syms]
        sec_counts = pd.Series(sectors).value_counts()
        sector_hhi_list.append(float(((sec_counts / n) ** 2).sum()))
        clusters = [get_cluster(s, trade_syms, CLUSTER_MAP_DEFAULT) for s in syms]
        cl_counts = pd.Series(clusters).value_counts()
        cluster_hhi_list.append(float(((cl_counts / n) ** 2).sum()))
    return {
        "avg_holdings": float(np.mean(avg_holdings_list)) if avg_holdings_list else 0.0,
        "avg_correlation": float(np.mean(corr_list)) if corr_list else None,
        "avg_sector_hhi": float(np.mean(sector_hhi_list)) if sector_hhi_list else None,
        "avg_cluster_hhi": float(np.mean(cluster_hhi_list)) if cluster_hhi_list else None,
    }


def loss_contribution_top20(trades, peak_pos, trough_pos):
    window_trades = [
        t for t in trades
        if "entry_idx" in t and "exit_idx" in t and t["entry_idx"] <= trough_pos and t["exit_idx"] >= peak_pos
        and t.get("pnl") is not None
    ]
    losers = sorted(window_trades, key=lambda t: t["pnl"])[:20]
    total_window_pnl = float(sum(t["pnl"] for t in window_trades))
    top20_pnl = float(sum(t["pnl"] for t in losers))
    return losers, total_window_pnl, top20_pnl, len(window_trades)


def annual_local_max_dd(eq: pd.Series, years):
    out = {}
    for yr in years:
        grp = eq[eq.index.year == yr]
        if len(grp) < 2:
            out[str(yr)] = None
            continue
        roll_max = grp.expanding().max()
        dd = (grp - roll_max) / roll_max
        out[str(yr)] = round(float(dd.min()) * 100, 2)
    return out


def main():
    cfg, universe_raw, rsr_df, alpha_df, regime_df, trade_syms, rsr_syms, sym_active_df, tech_matrices = load_all()
    args = (cfg, universe_raw, rsr_df, alpha_df, regime_df, trade_syms, rsr_syms, tech_matrices, sym_active_df)

    print(f"\n{'='*78}\n  [Part0] A_APRIL_REPRO / C_PROD_MINUS_ATR 実行\n{'='*78}")
    res_a = run(*args, use_dyn=False, shock="full_exit", rsr_exit=75.0,
                atr_trail=False, ml=False, sizing=False, mtf=False, sizing_mode="existing")
    res_c = run(*args, use_dyn=True, shock="composite", rsr_exit=70.0,
                atr_trail=True, ml=True, sizing=False, mtf=True, sizing_mode="equal")
    print(f"  A_APRIL_REPRO    : MaxDD={res_a['max_dd']:+.1f}%  CAGR={res_a['cagr']:+.1f}%  Sharpe={res_a['sharpe']:.3f}")
    print(f"  C_PROD_MINUS_ATR : MaxDD={res_c['max_dd']:+.1f}%  CAGR={res_c['cagr']:+.1f}%  Sharpe={res_c['sharpe']:.3f}")

    # 静的相関行列（全期間・全アクティブ銘柄の日次リターン、ポートフォリオ集中度評価用の固定基準）
    active_syms = [s for s in trade_syms if s in universe_raw]
    close_df = pd.DataFrame({s: universe_raw[s]["df"]["Close"] for s in active_syms}).dropna(how="all")
    ts_f, te_f = pd.Timestamp(FULL_START), pd.Timestamp(FULL_END)
    close_df = close_df[(close_df.index >= ts_f) & (close_df.index <= te_f)]
    corr_mat = close_df.pct_change().corr()

    results_summary = {}
    for label, res in (("A_APRIL_REPRO", res_a), ("C_PROD_MINUS_ATR", res_c)):
        eq, dd = res["equity_curve"], res["drawdown_curve"]
        n_days = len(eq)
        recovery_days = res["drawdown_summary"].get("max_dd_recovery_days")
        peak_pos, trough_pos, peak_date, trough_date, recovery_date = max_dd_window(eq, dd, recovery_days)

        print(f"\n{'='*78}\n  [1] 最大DD期間特定: {label}\n{'='*78}")
        print(f"  開始(peak)={peak_date.date()}  終了(trough)={trough_date.date()}  "
              f"回復={recovery_date.date() if recovery_date is not None else '未回復'}  "
              f"DD={float(dd.loc[trough_date])*100:+.1f}%  期間={trough_pos-peak_pos}営業日")

        trades = res["_trades"]
        losers, total_window_pnl, top20_pnl, n_window_trades = loss_contribution_top20(trades, peak_pos, trough_pos)
        print(f"\n  [2] DD期間中 損失寄与Top20: window_trades={n_window_trades}  window合計pnl=¥{total_window_pnl:,.0f}  "
              f"Top20合計pnl=¥{top20_pnl:,.0f}  Top20寄与率={(top20_pnl/total_window_pnl*100) if total_window_pnl else 0:.1f}%")
        for t in losers[:20]:
            print(f"    {t['symbol']:10s} entry_idx={t['entry_idx']:5d} exit_idx={t['exit_idx']:5d} pnl=¥{t['pnl']:>12,.0f}")

        held = holding_sets(trades, n_days)
        port = portfolio_concentration(held, trade_syms, corr_mat)
        print(f"\n  [3] ポートフォリオ分析: avg_holdings={port['avg_holdings']:.2f}  "
              f"avg_correlation={port['avg_correlation']:.3f}  avg_sector_HHI={port['avg_sector_hhi']:.3f}  "
              f"avg_cluster_HHI={port['avg_cluster_hhi']:.3f}")

        annual_dd = annual_local_max_dd(eq, range(2020, 2027))
        print(f"\n  [5] 年別DD(年内ローカルmax DD): " + " / ".join(f"{y}:{v:+.1f}%" if v is not None else f"{y}:N/A" for y, v in annual_dd.items()))

        results_summary[label] = {
            "max_dd": res["max_dd"], "cagr": res["cagr"], "sharpe": res["sharpe"],
            "peak_date": str(peak_date.date()), "trough_date": str(trough_date.date()),
            "recovery_date": str(recovery_date.date()) if recovery_date is not None else None,
            "dd_window_days": trough_pos - peak_pos,
            "top20_losers": [{"symbol": t["symbol"], "entry_idx": t["entry_idx"], "exit_idx": t["exit_idx"], "pnl": t["pnl"]} for t in losers],
            "window_total_pnl": total_window_pnl, "window_top20_pnl": top20_pnl, "n_window_trades": n_window_trades,
            "portfolio": port, "annual_local_max_dd": annual_dd,
        }

    # ---------------------------------------------------------------- #
    # [4] 各機能単独寄与のDD変化（Study33waterfallと同条件、MaxDD列に注目。MTF/sizing切替も完全性のため追加）
    # ---------------------------------------------------------------- #
    print(f"\n{'='*78}\n  [4] 各機能単独寄与のDD変化（sizing_mode=existing固定で4要因+sizing切替）\n{'='*78}")
    waterfall_steps = [
        ("0_APRIL_REPRO",     dict(use_dyn=False, shock="full_exit", rsr_exit=75.0, atr_trail=False, ml=False, mtf=False, sizing_mode="existing")),
        ("1_+UNIVERSE",       dict(use_dyn=True,  shock="composite", rsr_exit=70.0, atr_trail=False, ml=False, mtf=False, sizing_mode="existing")),
        ("2_+ATR_TRAILING",   dict(use_dyn=True,  shock="composite", rsr_exit=70.0, atr_trail=True,  ml=False, mtf=False, sizing_mode="existing")),
        ("3_+MULTILAYER_RSR", dict(use_dyn=True,  shock="composite", rsr_exit=70.0, atr_trail=True,  ml=True,  mtf=False, sizing_mode="existing")),
        ("4_+MTF_FILTER(参考)", dict(use_dyn=True, shock="composite", rsr_exit=70.0, atr_trail=True,  ml=True,  mtf=True,  sizing_mode="existing")),
    ]
    wf_rows = []
    prev_dd = None
    for label, p in waterfall_steps:
        res = run(*args, use_dyn=p["use_dyn"], shock=p["shock"], rsr_exit=p["rsr_exit"],
                   atr_trail=p["atr_trail"], ml=p["ml"], sizing=False, mtf=p["mtf"], sizing_mode=p["sizing_mode"])
        d_dd = (res["max_dd"] - prev_dd) if prev_dd is not None else None
        wf_rows.append({"label": label, "max_dd": res["max_dd"], "delta_max_dd_pp": round(d_dd, 2) if d_dd is not None else None,
                         "avg_exposure": res["avg_exposure"], "n_trades": res["n_trades"]})
        prev_dd = res["max_dd"]
        dstr = f"{d_dd:+.2f}pp" if d_dd is not None else "  --  "
        print(f"  {label:22s} MaxDD={res['max_dd']:+6.1f}%({dstr})  AvgExp={res['avg_exposure']:.1f}%  trades={res['n_trades']}")
    d_sizing = res_c["max_dd"] - wf_rows[-1]["max_dd"]
    print(f"  5_sizing existing→equal(=C)  MaxDD={res_c['max_dd']:+6.1f}%({d_sizing:+.2f}pp)  AvgExp={res_c['avg_exposure']:.1f}%  trades={res_c['n_trades']}")
    wf_rows.append({"label": "5_+SIZING_EQUAL(=C, 参考)", "max_dd": res_c["max_dd"], "delta_max_dd_pp": round(d_sizing, 2),
                    "avg_exposure": res_c["avg_exposure"], "n_trades": res_c["n_trades"]})

    # ---------------------------------------------------------------- #
    # [判定] 定量帰属
    # ---------------------------------------------------------------- #
    total_dd_increase = res_a["max_dd"] - res_c["max_dd"]  # 負の値が増加分（より深いDD）
    universe_contrib = wf_rows[1]["delta_max_dd_pp"]
    atr_trail_contrib = wf_rows[2]["delta_max_dd_pp"]
    multirsr_contrib = wf_rows[3]["delta_max_dd_pp"]
    mtf_contrib = wf_rows[4]["delta_max_dd_pp"]
    sizing_contrib = wf_rows[5]["delta_max_dd_pp"]
    exit_contrib = atr_trail_contrib + multirsr_contrib + mtf_contrib

    def pct_of_total(x):
        return round(x / total_dd_increase * 100, 1) if total_dd_increase != 0 else 0.0

    annual_a = results_summary["A_APRIL_REPRO"]["annual_local_max_dd"]
    annual_c = results_summary["C_PROD_MINUS_ATR"]["annual_local_max_dd"]
    worst_year_delta = None
    worst_year = None
    for y in annual_a:
        if annual_a[y] is not None and annual_c[y] is not None:
            delta = annual_c[y] - annual_a[y]
            if worst_year_delta is None or delta < worst_year_delta:
                worst_year_delta, worst_year = delta, y

    print(f"\n{'='*78}\n  [判定] DD増加の定量帰属\n{'='*78}")
    print(f"  全体DD増加量: {total_dd_increase:+.2f}pp (A{res_a['max_dd']:+.1f}% → C{res_c['max_dd']:+.1f}%)")
    print(f"  A(動的ユニバース)寄与: {universe_contrib:+.2f}pp ({pct_of_total(universe_contrib):.1f}%)")
    print(f"  C(Exit変更=ATR Trailing+MultiRSR+MTF合計)寄与: {exit_contrib:+.2f}pp ({pct_of_total(exit_contrib):.1f}%)")
    print(f"  E(その他=sizing existing→equal切替)寄与: {sizing_contrib:+.2f}pp ({pct_of_total(sizing_contrib):.1f}%)")
    print(f"  D(特定年度): 最大悪化年={worst_year} (ΔDD={worst_year_delta:+.2f}pp)")
    print(f"  B(集中化)根拠: avg_holdings A={results_summary['A_APRIL_REPRO']['portfolio']['avg_holdings']:.2f} "
          f"vs C={results_summary['C_PROD_MINUS_ATR']['portfolio']['avg_holdings']:.2f} / "
          f"avg_correlation A={results_summary['A_APRIL_REPRO']['portfolio']['avg_correlation']:.3f} "
          f"vs C={results_summary['C_PROD_MINUS_ATR']['portfolio']['avg_correlation']:.3f}")

    out = {
        "period": [FULL_START, FULL_END],
        "results_summary": results_summary,
        "feature_isolation_max_dd": wf_rows,
        "attribution": {
            "total_dd_increase_pp": round(total_dd_increase, 2),
            "universe_contrib_pp": universe_contrib, "universe_pct_of_total": pct_of_total(universe_contrib),
            "exit_contrib_pp": round(exit_contrib, 2), "exit_pct_of_total": pct_of_total(exit_contrib),
            "sizing_contrib_pp": sizing_contrib, "sizing_pct_of_total": pct_of_total(sizing_contrib),
            "worst_year": worst_year, "worst_year_delta_pp": round(worst_year_delta, 2) if worst_year_delta is not None else None,
        },
    }
    os.makedirs("C:/ai-trading/backtests", exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n結果保存: {OUTPUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
