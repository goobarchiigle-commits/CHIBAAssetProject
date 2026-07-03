"""
study52_production_candidate_optimization.py
Production Candidate Optimization — VOL_ADJ 完全除外, ATR_EXT × EQ_SCALE 最終比較

ケース:
  A  S5 Baseline
  B  S5 + ATR Extension
  C  S5 + EQ_SCALE
  D  S5 + ATR Extension + EQ_SCALE   ← Study50 未実行; 今回の新ケース

評価:
  IS   2018-2024
  OOS  2025
  WF   5-fold (OOS 2020/2021/2022/2023/2024)
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from datetime import date

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

from src.backtest.snapshot_archaeology_202606 import build_common_dataset
import src.backtest.composite_alpha_bt as cab
from src.backtest.wf_dynamic_universe import WF_SEGS

TODAY_STR = date.today().strftime("%Y-%m-%d")

CAPITAL            = 3_000_000
MIN_HOLD           = 3
DATA_END           = "2025-12-31"
IS_START           = "2018-01-01"
IS_END             = "2024-12-31"
OOS_START          = "2025-01-01"
OOS_END            = "2025-12-31"
ADDON_SIZE_FRAC    = 0.25
ADDON_ATR_MULT     = 1.0

# ── ケース: (label, exit_policy, addon_policy) — VOL_ADJ 完全除外 ──────────
CASES = [
    ("A_BASELINE",   None,  None),
    ("B_ATR_EXT",    "A",   None),
    ("C_EQ_SCALE",   None,  "D"),
    ("D_ATR_EQ",     "A",   "D"),
]

STUDY50_PATH = ROOT / "backtests/study50_post_integrity_revalidation_2026-06-28.json"
STUDY50_KEY  = {
    "A_BASELINE": "A_BASELINE",
    "B_ATR_EXT":  "B_ATR_EXT",
    "C_EQ_SCALE": "D_EQ_SCALE",   # Study50 では D_EQ_SCALE が今回の C
}


def load_study50() -> dict:
    if not STUDY50_PATH.exists():
        return {}
    with open(STUDY50_PATH, encoding="utf-8") as f:
        return json.load(f)


def get_active(ds: dict, all_syms: list, start: str, end: str) -> pd.DataFrame:
    from src.config_loader import load_strategy_config
    cfg = load_strategy_config()
    bc  = cfg.risk_controls.bear_universe_filter
    be  = list(bc.excluded_sectors) if bc.enabled else None
    return cab.build_dyn_rsr42_active(
        universe_raw=ds["universe_raw"],
        topix_close=ds["topix_close"],
        rsr_df=ds["rsr_df"],
        all_syms=all_syms,
        start=start, end=end,
        bear_exclude_sectors=be,
        sym_sector_map=dict(ds["trade_syms"]) if be else None,
    )


def run_bt(ds: dict, sym_active_df, start: str, end: str,
           exit_policy, addon_policy) -> dict:
    return cab.run_scenario(
        scenario               = "BASELINE",
        universe_raw           = ds["universe_raw"],
        rsr_df                 = ds["rsr_df"],
        alpha_df               = None,
        regime_df              = ds["regime_df"],
        trade_syms             = ds["trade_syms"],
        rsr_syms               = ds["rsr_syms"],
        cfg                    = ds["base_cfg"],
        start                  = start, end=end, verbose=False,
        tech_matrices          = ds["tech_matrices"],
        breadth_series         = ds["breadth_series"],
        capital                = CAPITAL,
        min_hold               = MIN_HOLD,
        topix_close            = ds["topix_close"],
        market_shock_mode      = "composite",
        rsr_exit_threshold     = 70.0,
        sym_active_df          = sym_active_df,
        enable_simple_rsr_exit = True,
        enable_atr_trailing_prod = True,
        enable_multilayer_rsr  = True,
        enable_atr_risk_sizing = False,
        enable_mtf_filter      = False,
        sizing_mode            = "existing",
        exit_policy            = exit_policy,
        exit_policy_atr_mult   = ADDON_ATR_MULT,
        exit_policy_defer_days = 5,
        max_positions_ts       = None,          # VOL_ADJ 完全除外
        addon_policy           = addon_policy,
        addon_atr_mult         = ADDON_ATR_MULT,
        addon_stage2_mult      = 2.0,
        addon_max_per_pos      = 1,
        addon_size_frac        = ADDON_SIZE_FRAC,
    )


def extract(m: dict) -> dict:
    return {
        "cagr":      round(float(m.get("cagr",          0.0) or 0.0), 2),
        "sharpe":    round(float(m.get("sharpe",         0.0) or 0.0), 3),
        "max_dd":    round(float(m.get("max_dd",         0.0) or 0.0), 2),
        "calmar":    round(float(m.get("calmar",         0.0) or 0.0), 3),
        "n_trades":  int  (m.get("n_trades",  0) or 0),
        "addon_cnt": int  (m.get("addon_count", 0) or 0),
        "avg_exp":   round(float(m.get("avg_exposure",   0.0) or 0.0), 1),
    }


def extract_s50(s50: dict, key: str, period: str) -> dict | None:
    try:
        return s50["period_results"][period][key]
    except (KeyError, TypeError):
        return None


def extract_s50_wf(s50: dict, key: str) -> dict | None:
    try:
        return s50["wf_results"][key]
    except (KeyError, TypeError):
        return None


def print_period(title: str, rows: dict[str, dict]) -> None:
    base = rows.get("A_BASELINE", {})
    sep  = "─" * 80
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)
    print(f"  {'Case':<14} {'CAGR%':>8} {'Sharpe':>7} {'MaxDD%':>8} {'Calmar':>7}"
          f" {'Trades':>7} {'Exp%':>6} {'ΔCAGR':>8}")
    print(sep)
    for cn, m in rows.items():
        if not m:
            print(f"  {cn:<14}  (error)"); continue
        dc = f"{m['cagr'] - base['cagr']:+.2f}" if cn != "A_BASELINE" and base else "    —"
        print(f"  {cn:<14} {m['cagr']:>+8.2f} {m['sharpe']:>7.3f}"
              f" {m['max_dd']:>8.2f} {m['calmar']:>7.3f}"
              f" {m['n_trades']:>7} {m['avg_exp']:>6.1f} {dc:>8}")


def print_wf_summary(wf: dict[str, dict]) -> None:
    base = wf.get("A_BASELINE", {})
    sep  = "─" * 80
    print(f"\n{sep}")
    print("  Walk-Forward 5-fold サマリー")
    print(sep)
    print(f"  {'Case':<14} {'WF':>4} {'avg CAGR':>10} {'avg Sh':>7}"
          f" {'avg DD':>8} {'std CAGR':>10} {'Seg3_22':>9} {'ΔCAGR':>8}")
    print(sep)
    for cn, ep, ap in CASES:
        w = wf.get(cn, {})
        if not w:
            print(f"  {cn:<14}  (no data)"); continue
        dc   = f"{w['avg_oos_cagr'] - base['avg_oos_cagr']:+.2f}" if cn != "A_BASELINE" and base else "    —"
        s3   = f"{w['seg3_2022_cagr']:+.2f}%" if w.get("seg3_2022_cagr") is not None else "    —"
        std  = f"{w['fold_std_cagr']:.2f}" if "fold_std_cagr" in w else "    —"
        print(f"  {cn:<14} {w['wf_count']:>2}/5 {w['avg_oos_cagr']:>+10.2f}%"
              f" {w['avg_oos_sharpe']:>7.3f} {w['avg_oos_dd']:>8.2f}%"
              f" {std:>10} {s3:>9} {dc:>8}")


def print_wf_folds(wf: dict[str, dict]) -> None:
    sep = "─" * 80
    print(f"\n{sep}")
    print("  Walk-Forward 各 Fold 比較")
    print(sep)
    years = ["2020", "2021", "2022", "2023", "2024"]
    print(f"  {'Fold':<8} {'A_BASELINE':>12} {'B_ATR_EXT':>12} {'C_EQ_SCALE':>12} {'D_ATR_EQ':>12}")
    print(sep)
    for yr in years:
        row = [f"  OOS {yr}"]
        for cn, ep, ap in CASES:
            w     = wf.get(cn, {})
            segs  = w.get("segments", [])
            seg   = next((s for s in segs if s.get("oos_year") == yr), {})
            cagr  = seg.get("cagr")
            mark  = "✓" if seg.get("wf_pass") else "✗"
            val   = f"{cagr:+.2f}%{mark}" if cagr is not None else "    —"
            row.append(f"{val:>12}")
        print(" ".join(row))
    print(sep)
    print(f"  {'Seg3 2022':8} ", end="")
    for cn, ep, ap in CASES:
        w  = wf.get(cn, {})
        s3 = w.get("seg3_2022_cagr")
        mk = "✓" if s3 is not None and s3 > 0 else ("✗" if s3 is not None and s3 < -3 else "△")
        v  = f"{s3:+.2f}%{mk}" if s3 is not None else "    —"
        print(f"{v:>12}", end="")
    print()


def main() -> None:
    print("=" * 80)
    print("  Study52 — Production Candidate Optimization (VOL_ADJ 除外)")
    print(f"  Date: {TODAY_STR}   Capital: ¥{CAPITAL:,}")
    print("=" * 80)

    # ── 0. Study50 キャッシュ読み込み ─────────────────────────────────────────
    s50 = load_study50()
    if s50:
        print(f"\n[Cache] Study50 data loaded ({STUDY50_PATH.name})")
        print("  A_BASELINE, B_ATR_EXT, C_EQ_SCALE を Study50 から再利用します。")
        print("  D_ATR_EQ (ATR_EXT + EQ_SCALE, VOL_ADJ なし) のみ新規実行。")
    else:
        print(f"\n[Cache] Study50 not found — 全ケースを新規実行します。")

    # ── 1. データセット ──────────────────────────────────────────────────────
    print(f"\n[1] データセット構築 (end={DATA_END})...")
    ds       = build_common_dataset(DATA_END)
    all_syms = list(ds["trade_syms"].keys())
    print(f"  {len(all_syms)} シンボル")

    # ── 2. IS / OOS 実行 ─────────────────────────────────────────────────────
    print(f"\n[2] IS ({IS_START}~{IS_END}) / OOS ({OOS_START}~{OOS_END})")

    # キャッシュ対象ケースのマッピング
    cached_cases = {"A_BASELINE", "B_ATR_EXT", "C_EQ_SCALE"} if s50 else set()

    period_results: dict[str, dict[str, dict]] = {"FULL_IS": {}, "TRUE_OOS": {}}

    active_is  = get_active(ds, all_syms, IS_START,  IS_END)
    active_oos = get_active(ds, all_syms, OOS_START, OOS_END)

    for cn, ep, ap in CASES:
        for period_name, s, e, act in [
            ("FULL_IS",  IS_START,  IS_END,  active_is),
            ("TRUE_OOS", OOS_START, OOS_END, active_oos),
        ]:
            s50_key = STUDY50_KEY.get(cn)
            if cn in cached_cases and s50_key and s50:
                m = extract_s50(s50, s50_key, period_name)
                if m:
                    period_results[period_name][cn] = m
                    src = "cache"
                    print(f"  {cn:<14} [{period_name}]  CAGR={m['cagr']:+.2f}%  [{src}]")
                    continue

            print(f"  {cn:<14} [{period_name}]  ep={ep or 'None'}  addon={ap or 'None'}...", end=" ", flush=True)
            try:
                raw = run_bt(ds, act, s, e, ep, ap)
                m   = extract(raw)
                period_results[period_name][cn] = m
                print(f"CAGR={m['cagr']:+.2f}%  MaxDD={m['max_dd']:.2f}%  Sh={m['sharpe']:.3f}")
            except Exception as err:
                print(f"ERROR: {err}")
                period_results[period_name][cn] = {}

    print_period("FULL IS (2018-2024)", period_results["FULL_IS"])
    print_period("TRUE OOS (2025)",     period_results["TRUE_OOS"])

    # ── 3. Walk-Forward ───────────────────────────────────────────────────────
    print(f"\n[3] Walk-Forward 5-fold")

    wf_results: dict[str, dict] = {}
    for cn, ep, ap in CASES:
        s50_key = STUDY50_KEY.get(cn)
        if cn in cached_cases and s50_key and s50:
            w50 = extract_s50_wf(s50, s50_key)
            if w50:
                segs  = w50.get("segments", [])
                cagrs = [s.get("cagr", 0) for s in segs if "cagr" in s]
                w50["fold_std_cagr"] = round(float(np.std(cagrs)), 2) if cagrs else 0.0
                wf_results[cn] = w50
                print(f"\n  {cn}: [cache] WF={w50['wf_count']}/5  avg={w50['avg_oos_cagr']:+.2f}%"
                      f"  Seg3={w50.get('seg3_2022_cagr'):+.2f}%")
                continue

        print(f"\n  ── {cn}  ep={ep or 'None'}  addon={ap or 'None'} ──")
        seg_rows = []
        for seg in WF_SEGS:
            n        = seg["seg"]
            oos_s, oos_e = seg["oos"]
            act_seg  = get_active(ds, all_syms, oos_s, oos_e)
            print(f"    Seg{n} OOS {oos_s[:4]}  ", end="", flush=True)
            try:
                raw  = run_bt(ds, act_seg, oos_s, oos_e, ep, ap)
                m    = extract(raw)
                wf_p = m["sharpe"] > 0
                seg_rows.append({"seg": n, "oos_year": oos_s[:4], "wf_pass": wf_p, **m})
                mark = "✓" if wf_p else "✗"
                print(f"CAGR={m['cagr']:+.2f}%  Sh={m['sharpe']:.3f}  DD={m['max_dd']:.1f}%  {mark}")
            except Exception as err:
                print(f"ERROR: {err}")
                seg_rows.append({"seg": n, "oos_year": oos_s[:4], "wf_pass": False})

        wf_cnt    = sum(1 for r in seg_rows if r.get("wf_pass"))
        cagrlist  = [r["cagr"]   for r in seg_rows if "cagr"   in r]
        shlist    = [r["sharpe"] for r in seg_rows if "sharpe" in r]
        ddlist    = [r["max_dd"] for r in seg_rows if "max_dd" in r]
        seg3_cagr = next((r["cagr"] for r in seg_rows if r.get("oos_year") == "2022" and "cagr" in r), None)

        wf_results[cn] = {
            "wf_count":        wf_cnt,
            "avg_oos_cagr":    round(float(np.mean(cagrlist)), 2) if cagrlist else 0.0,
            "avg_oos_sharpe":  round(float(np.mean(shlist)),   3) if shlist   else 0.0,
            "avg_oos_dd":      round(float(np.mean(ddlist)),   2) if ddlist   else 0.0,
            "seg3_2022_cagr":  round(seg3_cagr, 2) if seg3_cagr is not None else None,
            "fold_std_cagr":   round(float(np.std(cagrlist)),  2) if cagrlist else 0.0,
            "segments":        seg_rows,
        }
        print(f"    WF: {wf_cnt}/5  avg={wf_results[cn]['avg_oos_cagr']:+.2f}%"
              f"  Seg3={seg3_cagr:+.2f}%" if seg3_cagr is not None else "")

    print_wf_summary(wf_results)
    print_wf_folds(wf_results)

    # ── 4. Feature Cost 評価 ──────────────────────────────────────────────────
    print(f"\n{'─'*80}")
    print("  Feature Cost (実装・運用コスト)")
    print(f"{'─'*80}")
    cost_table = {
        "A_BASELINE": {"state_files": 0, "params": 0,  "complexity": "なし",      "risk": "なし"},
        "B_ATR_EXT":  {"state_files": 1, "params": 2,  "complexity": "低",        "risk": "退出タイミングの defer"},
        "C_EQ_SCALE": {"state_files": 1, "params": 3,  "complexity": "低",        "risk": "追加発注の重複防止"},
        "D_ATR_EQ":   {"state_files": 2, "params": 5,  "complexity": "中",        "risk": "exit defer + 追加発注の組み合わせ"},
    }
    print(f"  {'Case':<14} {'State files':>12} {'Params':>8} {'Complexity':>12} {'Risk':>35}")
    print(f"  {'─'*80}")
    for cn, c in cost_table.items():
        print(f"  {cn:<14} {c['state_files']:>12} {c['params']:>8} {c['complexity']:>12}   {c['risk']}")

    # ── 5. 採用判定 ──────────────────────────────────────────────────────────
    print(f"\n{'─'*80}")
    print("  採用判定 (Study51 新基準: WF 5/5 or Seg3 > +1pp primary)")
    print(f"{'─'*80}")
    base_is  = period_results["FULL_IS"].get("A_BASELINE", {})
    base_oos = period_results["TRUE_OOS"].get("A_BASELINE", {})
    base_wf  = wf_results.get("A_BASELINE", {})

    judgments: dict[str, dict] = {}
    for cn, ep, ap in CASES:
        if cn == "A_BASELINE":
            continue
        m_is  = period_results["FULL_IS"].get(cn, {})
        m_oos = period_results["TRUE_OOS"].get(cn, {})
        wf    = wf_results.get(cn, {})
        if not m_is or not base_is:
            judgments[cn] = {"verdict": "ERROR"}; continue

        d_cagr_is   = round(m_is["cagr"]   - base_is["cagr"],   2)
        d_dd_is     = round(m_is["max_dd"] - base_is["max_dd"], 2)
        d_cagr_oos  = round(m_oos.get("cagr", 0) - base_oos.get("cagr", 0), 2) if m_oos and base_oos else None
        d_cagr_wf   = round(wf.get("avg_oos_cagr", 0) - base_wf.get("avg_oos_cagr", 0), 2) if base_wf else None
        wf_cnt      = wf.get("wf_count", 0)
        seg3        = wf.get("seg3_2022_cagr")
        seg3_base   = base_wf.get("seg3_2022_cagr")
        d_seg3      = round(seg3 - seg3_base, 2) if seg3 is not None and seg3_base is not None else None

        # 新基準 primary: WF 5/5 または Seg3 > base + 1pp
        primary_ok  = (wf_cnt >= 5) or (d_seg3 is not None and d_seg3 > 1.0)
        dd_ok       = d_dd_is > -2.0
        oos_ok      = d_cagr_oos is None or d_cagr_oos >= -1.0

        if primary_ok and dd_ok and oos_ok:
            verdict = "ADOPT"
        elif primary_ok and not dd_ok:
            verdict = "CONDITIONAL"   # primary ok だが DD コスト
        elif not primary_ok:
            verdict = "REJECT"
        else:
            verdict = "MARGINAL"

        role = []
        if d_cagr_wf is not None and d_cagr_wf > 0.5:
            role.append("Return Enhancer")
        if d_seg3 is not None and d_seg3 > 1.0:
            role.append("Robustness Enhancer")
        if not role:
            role.append("Neutral / No Evidence")

        judgments[cn] = {
            "verdict": verdict, "role": role,
            "d_cagr_is": d_cagr_is, "d_dd_is": d_dd_is,
            "wf_cnt": wf_cnt, "d_cagr_wf": d_cagr_wf,
            "d_cagr_oos": d_cagr_oos, "d_seg3": d_seg3,
        }

    print(f"\n  {'Case':<14} {'役割':<32} {'ΔCAGR_IS':>9} {'ΔMaxDD':>7} {'WF':>4}"
          f" {'ΔCAGR_WF':>9} {'ΔOOS':>7} {'ΔSeg3':>7} {'判定':>12}")
    print(f"  {'─'*110}")
    for cn, ep, ap in CASES:
        if cn == "A_BASELINE": continue
        j = judgments.get(cn, {})
        role  = " + ".join(j.get("role", ["?"]))
        dseg3 = f"{j.get('d_seg3'):+.2f}" if j.get("d_seg3") is not None else "    —"
        dois  = f"{j.get('d_cagr_oos'):+.2f}" if j.get("d_cagr_oos") is not None else "    —"
        dwf   = f"{j.get('d_cagr_wf'):+.2f}" if j.get("d_cagr_wf") is not None else "    —"
        print(f"  {cn:<14} {role:<32} {j.get('d_cagr_is', 0):>+9.2f}"
              f" {j.get('d_dd_is', 0):>+7.2f} {j.get('wf_cnt', 0):>2}/5"
              f" {dwf:>9} {dois:>7} {dseg3:>7} {j.get('verdict','?'):>12}")

    # ── 6. Production Recommendation ─────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  Production Recommendation")
    print(f"{'='*80}")

    # 最終選定: 判定結果に基づく
    adopted = [cn for cn, ep, ap in CASES if cn != "A_BASELINE"
               and judgments.get(cn, {}).get("verdict") in ("ADOPT", "CONDITIONAL")]

    fm_is  = period_results["FULL_IS"].get(adopted[-1] if adopted else "A_BASELINE", {})
    fm_oos = period_results["TRUE_OOS"].get(adopted[-1] if adopted else "A_BASELINE", {})
    fm_wf  = wf_results.get(adopted[-1] if adopted else "A_BASELINE", {})
    final  = adopted[-1] if adopted else "A_BASELINE"

    print(f"\n  採用可能ケース: {adopted if adopted else ['なし']}")
    print()
    print(f"  ┌────────────────────────────────────────────────────────────┐")
    print(f"  │  FINAL PRODUCTION CONFIGURATION: {final:<26} │")
    print(f"  ├────────────────────────────────────────────────────────────┤")
    print(f"  │  指標           │   IS 2018-2024   │   OOS 2025   │  WF   │")
    print(f"  ├─────────────────┼──────────────────┼──────────────┼───────┤")
    rows_fmt = [
        ("CAGR",    "cagr"),
        ("MaxDD",   "max_dd"),
        ("Sharpe",  "sharpe"),
        ("Calmar",  "calmar"),
    ]
    for lbl, key in rows_fmt:
        v_is  = fm_is.get(key, "?")
        v_oos = fm_oos.get(key, "?")
        wf_v  = fm_wf.get("avg_oos_cagr", "?") if key == "cagr" else (
                fm_wf.get("avg_oos_dd", "?") if key == "max_dd" else "")
        print(f"  │  {lbl:<15} │ {str(v_is):>16} │ {str(v_oos):>12} │ {str(wf_v):>5} │")
    print(f"  │  WF pass        │              — │           —  │  {fm_wf.get('wf_count','?')}/5  │")
    print(f"  │  Seg3_2022      │              — │           —  │ {str(fm_wf.get('seg3_2022_cagr','?')):>5}% │")
    print(f"  └─────────────────┴──────────────────┴──────────────┴───────┘")

    # ── 保存 ───────────────────────────────────────────────────────────────────
    out = ROOT / "backtests" / f"study52_production_optimization_{TODAY_STR}.json"
    payload = {
        "study": "Study52_ProductionCandidateOptimization",
        "date":  TODAY_STR,
        "cases": [cn for cn, *_ in CASES],
        "note":  "VOL_ADJ 完全除外。Study50キャッシュ再利用(A/B/C)、D_ATR_EQ新規実行",
        "period_results": period_results,
        "wf_results":     {k: {kk: vv for kk, vv in v.items() if kk != "segments"}
                           for k, v in wf_results.items()},
        "wf_segments":    {k: v.get("segments", []) for k, v in wf_results.items()},
        "judgments":      judgments,
        "final_case":     final,
        "final_is":       fm_is,
        "final_oos":      fm_oos,
        "final_wf":       {k: v for k, v in fm_wf.items() if k != "segments"},
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  結果保存: {out}")
    print(f"\n{'='*80}")
    print("  Study52 COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
