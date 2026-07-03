"""
backtest/study35_wf_true_oos_validation_202606.py
Study35: PROD_MINUS_ATR Walk-Forward & True OOS Validation。検証専用・本番コード変更なし。
一回限りの監査スクリプト（恒久モジュール化しない）。

WF方式論はwf_dyn_rsr42.py / wf_dynamic_universe.pyのfold構成・スコアリングを変更せず再利用:
  WF_SEGS（5fold IS/OOS）, TRUE_OOS(2025), IS_FULL(2018-2024) は src.backtest.wf_dynamic_universe から import。
  スコアリング規則（win=oos_sharpe>0, ratio=oos_sh/is_sh, pass_count, median_oos_sharpe, worst_oos_dd, avg_oos_is_ratio）
  も完全に同一の計算式をそのまま再実装（folds/date ranges/scoring/pass判定は無変更）。
  Extended OOS 2026 YTDは本タスクで新規追加された評価窓（既存foldの変更ではない、追加のみ）。

比較対象（4設定、Study32-34と同一定義）:
  A_APRIL_REPRO              : 静的ユニバース, full_exit, rsr_exit=75, ATR Trailingなし, MultiRSRなし, equal weight(sizing_mode=existing)
  B_CURRENT_PROD              : 現行本番忠実（動的ユニバース+composite+rsr_exit=70+ATR Trailing+MultiRSR+MTF+ATR Risk Sizing）
  C_PROD_MINUS_ATR             : Bと同じだしATR Risk SizingのみOFF→sizing_mode=equal（Study32 REMOVE判定後の代替）
  D_PROD_MINUS_ATR_MINUS_MTF   : Cからさらに MTF Filter を OFF

動的ユニバース(dyn_rsr42_bear_rs0)はfold毎にbuild_dyn_rsr42_active(..., end=fold終了日)で
no-lookahead再構築（wf_dyn_rsr42.pyの_build_active per-fold再構築方式を踏襲）。

実行: python src/backtest/study35_wf_true_oos_validation_202606.py
"""
from __future__ import annotations
import os, sys, json, warnings, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import src.backtest.composite_alpha_bt as bt
from src.backtest.wf_dynamic_universe import WF_SEGS, TRUE_OOS, IS_FULL
from src.backtest.atr_sizing_diagnostic_202606 import load_all
from src.strategy.universe import build_dyn_rsr42_active

FULL_START = "2018-01-01"
FULL_END   = "2026-06-23"
EXTENDED_OOS = ("2026-01-01", "2026-06-23")
OUTPUT_JSON = f"C:/ai-trading/backtests/study35_wf_true_oos_validation_202606_{time.strftime('%Y-%m-%d')}.json"

CONFIGS = {
    "A_APRIL_REPRO": dict(use_dyn=False, shock="full_exit", rsr_exit=75.0,
                          atr_trail=False, ml=False, sizing=False, mtf=False, sizing_mode="existing"),
    "B_CURRENT_PROD": dict(use_dyn=True, shock="composite", rsr_exit=70.0,
                           atr_trail=True, ml=True, sizing=True, mtf=True, sizing_mode="existing"),
    "C_PROD_MINUS_ATR": dict(use_dyn=True, shock="composite", rsr_exit=70.0,
                             atr_trail=True, ml=True, sizing=False, mtf=True, sizing_mode="equal"),
    "D_PROD_MINUS_ATR_MINUS_MTF": dict(use_dyn=True, shock="composite", rsr_exit=70.0,
                                       atr_trail=True, ml=True, sizing=False, mtf=False, sizing_mode="equal"),
}


def run_period(cfg, universe_raw, rsr_df, alpha_df, regime_df, trade_syms, rsr_syms, tech_matrices,
               conf, start, end, sym_active_df):
    return bt.run_scenario(
        scenario="PROD_FAITHFUL" if (conf["atr_trail"] or conf["ml"] or conf["sizing"] or conf["mtf"]) else "BASELINE",
        universe_raw=universe_raw, rsr_df=rsr_df, alpha_df=alpha_df, regime_df=regime_df,
        trade_syms=trade_syms, rsr_syms=rsr_syms, cfg=cfg,
        start=start, end=end, verbose=False,
        tech_matrices=tech_matrices,
        capital=cfg.portfolio.capital,
        min_hold=cfg.risk.min_hold_days,
        market_shock_mode=conf["shock"],
        rsr_exit_threshold=conf["rsr_exit"],
        sym_active_df=(sym_active_df if conf["use_dyn"] else None),
        enable_atr_trailing_prod=conf["atr_trail"],
        enable_multilayer_rsr=conf["ml"],
        enable_atr_risk_sizing=conf["sizing"],
        enable_mtf_filter=conf["mtf"],
        risk_sizing_pct=bt.PROD_RISK_PCT,
        sizing_mode=conf["sizing_mode"],
    )


def metrics4(res):
    return {"cagr": res.get("cagr"), "sharpe": res.get("sharpe"), "max_dd": res.get("max_dd"), "calmar": res.get("calmar")}


def main():
    cfg, universe_raw, rsr_df, alpha_df, regime_df, trade_syms, rsr_syms, _unused_sym_active, tech_matrices = load_all()
    topix_close = bt._download_topix(FULL_START, FULL_END)
    bear_filter_cfg = cfg.risk_controls.bear_universe_filter
    bear_exclude = list(bear_filter_cfg.excluded_sectors) if bear_filter_cfg.enabled else None

    # ── 動的ユニバース: fold境界end日付ごとにno-lookahead再構築（end日付でキャッシュ、B/C/D共有）──
    active_cache: dict[str, pd.DataFrame] = {}

    def get_active(end: str) -> pd.DataFrame:
        if end not in active_cache:
            active_cache[end] = build_dyn_rsr42_active(
                universe_raw=universe_raw, topix_close=topix_close, rsr_df=rsr_df,
                all_syms=list(trade_syms.keys()), start=FULL_START, end=end,
                bear_exclude_sectors=bear_exclude,
                sym_sector_map=dict(trade_syms) if bear_exclude else None,
            )
        return active_cache[end]

    all_config_results = {}

    for conf_name, conf in CONFIGS.items():
        print(f"\n{'='*78}\n  設定: {conf_name}\n{'='*78}")

        # ── 1. Walk-Forward（5fold、wf_dyn_rsr42.py方式論を変更なく再利用）──
        seg_results = []
        for seg_def in WF_SEGS:
            n = seg_def["seg"]
            is_s, is_e = seg_def["is"]
            oos_s, oos_e = seg_def["oos"]
            active_is = get_active(is_e) if conf["use_dyn"] else None
            active_oos = get_active(oos_e) if conf["use_dyn"] else None
            r_is = run_period(cfg, universe_raw, rsr_df, alpha_df, regime_df, trade_syms, rsr_syms, tech_matrices,
                              conf, is_s, is_e, active_is)
            r_oos = run_period(cfg, universe_raw, rsr_df, alpha_df, regime_df, trade_syms, rsr_syms, tech_matrices,
                               conf, oos_s, oos_e, active_oos)
            is_sh = r_is.get("sharpe", 0) or 0.0
            oos_sh = r_oos.get("sharpe", 0) or 0.0
            oos_dd = r_oos.get("max_dd", 0) or 0.0
            ratio = round(oos_sh / is_sh, 3) if is_sh != 0 else None
            win = oos_sh > 0
            seg_results.append({
                "seg": n, "oos_year": oos_s[:4], "is_sharpe": round(is_sh, 3), "oos_sharpe": round(oos_sh, 3),
                "oos_max_dd": round(oos_dd, 2), "oos_cagr": r_oos.get("cagr"), "oos_calmar": r_oos.get("calmar"),
                "ratio": ratio, "win": win,
            })
            mark = "OK" if win else "NG"
            print(f"  Seg{n} OOS={oos_s[:4]}  IS_Sh={is_sh:.3f}  OOS_Sh={oos_sh:.3f}  OOS_DD={oos_dd:+.1f}%  {mark}")

        oos_sharpes = [s["oos_sharpe"] for s in seg_results]
        pass_count = sum(1 for s in seg_results if s["win"])
        median_oos = float(np.median(oos_sharpes))
        worst_dd = min(s["oos_max_dd"] for s in seg_results)
        valid_ratios = [s["ratio"] for s in seg_results if s["ratio"] is not None]
        avg_ratio = float(np.mean(valid_ratios)) if valid_ratios else 0.0
        wf_summary = {"pass_count": f"{pass_count}/5", "median_oos_sharpe": round(median_oos, 3),
                      "worst_oos_dd": round(worst_dd, 2), "avg_oos_is_ratio": round(avg_ratio, 3)}
        print(f"  WF summary: {wf_summary}")

        # ── Full IS（2018-2024） ──
        active_full = get_active(IS_FULL[1]) if conf["use_dyn"] else None
        r_full_is = run_period(cfg, universe_raw, rsr_df, alpha_df, regime_df, trade_syms, rsr_syms, tech_matrices,
                               conf, *IS_FULL, active_full)
        full_is_m = metrics4(r_full_is)
        print(f"  Full IS: {full_is_m}")

        # ── 2. True OOS 2025 ──
        active_oos25 = get_active(TRUE_OOS[1]) if conf["use_dyn"] else None
        r_oos25 = run_period(cfg, universe_raw, rsr_df, alpha_df, regime_df, trade_syms, rsr_syms, tech_matrices,
                             conf, *TRUE_OOS, active_oos25)
        true_oos_m = metrics4(r_oos25)
        print(f"  True OOS 2025: {true_oos_m}")

        # ── 3. Extended OOS 2026 YTD ──
        active_oos26 = get_active(EXTENDED_OOS[1]) if conf["use_dyn"] else None
        r_oos26 = run_period(cfg, universe_raw, rsr_df, alpha_df, regime_df, trade_syms, rsr_syms, tech_matrices,
                             conf, *EXTENDED_OOS, active_oos26)
        ext_oos_m = metrics4(r_oos26)
        print(f"  Extended OOS 2026YTD: {ext_oos_m}")

        # ── 4. 年別リターン（フル期間1回run、run_scenario内のannual_returnsを再利用） ──
        active_full_period = get_active(FULL_END) if conf["use_dyn"] else None
        r_full_period = run_period(cfg, universe_raw, rsr_df, alpha_df, regime_df, trade_syms, rsr_syms, tech_matrices,
                                   conf, FULL_START, FULL_END, active_full_period)
        annual = {y: r_full_period["annual_returns"].get(y) for y in ("2020", "2021", "2022", "2023", "2024", "2025", "2026")}
        print(f"  年別リターン: {annual}")

        all_config_results[conf_name] = {
            "wf_segments": seg_results, "wf_summary": wf_summary,
            "full_is": full_is_m, "true_oos_2025": true_oos_m, "extended_oos_2026ytd": ext_oos_m,
            "annual_returns": annual,
        }

    # ---------------------------------------------------------------- #
    # 5. Attribution summary: A vs C, C vs D（Full IS基準）
    # ---------------------------------------------------------------- #
    def delta4(base, other):
        return {k: round(other[k] - base[k], 3) if (base.get(k) is not None and other.get(k) is not None) else None
                for k in ("cagr", "sharpe", "max_dd", "calmar")}

    a_full = all_config_results["A_APRIL_REPRO"]["full_is"]
    c_full = all_config_results["C_PROD_MINUS_ATR"]["full_is"]
    d_full = all_config_results["D_PROD_MINUS_ATR_MINUS_MTF"]["full_is"]
    attribution = {
        "A_vs_C_full_is_delta": delta4(a_full, c_full),
        "C_vs_D_full_is_delta": delta4(c_full, d_full),
    }
    print(f"\n{'='*78}\n  [5] Attribution Summary (Full IS基準)\n{'='*78}")
    print(f"  A vs C: {attribution['A_vs_C_full_is_delta']}")
    print(f"  C vs D: {attribution['C_vs_D_full_is_delta']}")

    # ---------------------------------------------------------------- #
    # 判定（事前定義ルール、機械的判定。閾値は本タスク指定文言をそのまま数式化）
    # ---------------------------------------------------------------- #
    a_oos = all_config_results["A_APRIL_REPRO"]["true_oos_2025"]
    a_wf = all_config_results["A_APRIL_REPRO"]["wf_summary"]
    a_ext = all_config_results["A_APRIL_REPRO"]["extended_oos_2026ytd"]

    def promote_check(name):
        r = all_config_results[name]
        oos, wf, ext = r["true_oos_2025"], r["wf_summary"], r["extended_oos_2026ytd"]
        cond_sharpe = oos["sharpe"] is not None and a_oos["sharpe"] is not None and oos["sharpe"] >= a_oos["sharpe"]
        cond_calmar = (oos["calmar"] is not None and a_oos["calmar"] is not None
                        and oos["calmar"] >= a_oos["calmar"] * 0.9)
        cond_wf = int(wf["pass_count"].split("/")[0]) >= int(a_wf["pass_count"].split("/")[0])
        # 「2026 YTDで実質的な劣化なし」= CAGR・Sharpeの両方が同時にAPRILより悪化していないこと（事前定義の機械的ルール）
        cond_2026 = not (ext["cagr"] is not None and a_ext["cagr"] is not None and ext["sharpe"] is not None and a_ext["sharpe"] is not None
                          and ext["cagr"] < a_ext["cagr"] and ext["sharpe"] < a_ext["sharpe"])
        return cond_sharpe, cond_calmar, cond_wf, cond_2026, all([cond_sharpe, cond_calmar, cond_wf, cond_2026])

    checks = {name: promote_check(name) for name in ("B_CURRENT_PROD", "C_PROD_MINUS_ATR", "D_PROD_MINUS_ATR_MINUS_MTF")}
    print(f"\n{'='*78}\n  [判定] PROMOTE_TO_BASELINE 条件チェック\n{'='*78}")
    for name, (cs, cc, cw, c26, overall) in checks.items():
        print(f"  {name:30s} Sharpe>=A:{cs}  Calmar>=A*0.9:{cc}  WF>=A:{cw}  No2026degr:{c26}  → {'PROMOTE可' if overall else 'NG'}")

    c_ok = checks["C_PROD_MINUS_ATR"][4]
    d_ok = checks["D_PROD_MINUS_ATR_MINUS_MTF"][4]

    # MTF二次目的判定: C vs D で True OOS 2025 Sharpe/Calmar の比較
    c_oos, d_oos = all_config_results["C_PROD_MINUS_ATR"]["true_oos_2025"], all_config_results["D_PROD_MINUS_ATR_MINUS_MTF"]["true_oos_2025"]
    if d_oos["sharpe"] > c_oos["sharpe"] and d_oos["calmar"] > c_oos["calmar"]:
        mtf_verdict = "NEGATIVE"  # MTF外した方(D)が良い→MTFは負の価値
    elif d_oos["sharpe"] < c_oos["sharpe"] and d_oos["calmar"] < c_oos["calmar"]:
        mtf_verdict = "POSITIVE"  # MTFありの方(C)が良い→MTFは正の価値
    else:
        mtf_verdict = "NEUTRAL"  # 方向不一致

    # 最終判定: D優位ならPROMOTE_D, CのみOKならPROMOTE_C, 両方NGならKEEP, 判定不能要素があればINCONCLUSIVE
    if d_ok and (d_oos["sharpe"] >= c_oos["sharpe"] and d_oos["calmar"] >= c_oos["calmar"]):
        final_verdict = "PROMOTE_D"
    elif c_ok:
        final_verdict = "PROMOTE_C"
    elif not c_ok and not d_ok:
        final_verdict = "KEEP_CURRENT_BASELINE"
    else:
        final_verdict = "INCONCLUSIVE"

    print(f"\n  MTF二次目的判定（C vs D, True OOS 2025基準）: mtf_value = {mtf_verdict}")
    print(f"  最終判定: final_verdict = {final_verdict}")

    out = {
        "period": [FULL_START, FULL_END],
        "configs": CONFIGS,
        "results": all_config_results,
        "attribution": attribution,
        "promote_checks": {name: {"sharpe_ok": v[0], "calmar_ok": v[1], "wf_ok": v[2], "no_2026_degradation": v[3], "overall": v[4]}
                            for name, v in checks.items()},
        "mtf_secondary_verdict": mtf_verdict,
        "final_verdict": final_verdict,
    }
    def _json_default(o):
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

    os.makedirs("C:/ai-trading/backtests", exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=_json_default)
    print(f"\n結果保存: {OUTPUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
