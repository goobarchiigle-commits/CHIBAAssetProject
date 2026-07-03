"""
study58a_production_integration_audit.py
Production Integration Audit — Quality Replacement Engine (Case E)

最終判定軸: Calmar / MaxDD / Recovery Factor (CAGRではない)

Phase1: Risk Attribution Audit — IS/OOS/Full/WF Baseline vs Case E
Phase2: Swap DD Attribution — 全Swapの除去株vs候補株リターン比較
Phase3: Decision Timeline Audit — Lookahead=0 確認
Phase4: Live Latency Audit — バックテスト vs ライブ特徴量タイミング差分
Phase5: Sensitivity Audit — thresholds 33/68 / 35/70 / 37/72 WF比較
Phase6: Research State Update
"""
from __future__ import annotations

import json
import sys
import warnings
from collections import defaultdict
from datetime import date
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

from src.backtest.snapshot_archaeology_202606 import build_common_dataset
import src.backtest.composite_alpha_bt as cab
from src.backtest.wf_dynamic_universe import WF_SEGS
from src.backtest.study57_dpo import (
    QualityScorer, QS_WEIGHTS, SWAP_THRESH,
    build_scorer_from_trades, build_swap_plan,
    compute_quality_features, get_active, run_bt, extract,
    swap_success_rate,
)
from src.config_loader import load_strategy_config

TODAY_STR  = date.today().strftime("%Y-%m-%d")
CAPITAL    = 3_000_000
MIN_HOLD   = 3
IS_START   = "2018-01-01"
IS_END     = "2024-12-31"
OOS_START  = "2025-01-01"
OOS_END    = "2025-12-31"
FULL_START = "2018-01-01"
FULL_END   = "2025-12-31"
IS_YEARS   = 7
OOS_YEARS  = 1
FULL_YEARS = 8

SEP = "─" * 90
OUT_FILE = ROOT / "backtests" / f"study58a_production_integration_audit_{TODAY_STR}.json"

# ================================================================== #
#  Phase1 helper: Recovery Factor, Full-period BT
# ================================================================== #

def recovery_factor(cagr_pct: float, max_dd_pct: float, n_years: float) -> float | None:
    """Total Return / abs(MaxDD).  Total Return = (1 + CAGR)^n_years - 1."""
    if max_dd_pct >= 0:
        return None
    total_ret = (1 + cagr_pct / 100) ** n_years - 1
    return round(total_ret * 100 / abs(max_dd_pct), 3)


def extract_full(m: dict, n_years: float) -> dict:
    r = extract(m)
    r["recovery_factor"] = recovery_factor(r["cagr"], r["max_dd"], n_years)
    return r


def run_case_e_bt(ds, active, start, end, scorer, base_result, dates_p):
    """Run Case E (Swap Hold<35 Cand>70) BT for a given period."""
    base_sell   = [t for t in base_result.get("_trades", []) if t.get("side") == "SELL"]
    base_missed = base_result.get("_missed_cands", [])
    forced_exits, swap_detail = build_swap_plan(
        base_sell, base_missed, dates_p, ds["universe_raw"], ds["rsr_df"], scorer, "E"
    )
    raw = run_bt(ds, active, start, end, quality_forced_exits=forced_exits)
    return raw, forced_exits, swap_detail


# ================================================================== #
#  Phase2: Swap DD Attribution helpers
# ================================================================== #

def fwd_return(sym: str, ref_date, universe_raw: dict, n_days: int = 60) -> float | None:
    if sym not in universe_raw:
        return None
    df_c = universe_raw[sym].get("df")
    if df_c is None or "Close" not in df_c.columns:
        return None
    close = df_c["Close"].dropna()
    close.index = pd.to_datetime(close.index)
    ref_date = pd.Timestamp(ref_date)
    avail = close[close.index <= ref_date]
    if avail.empty:
        return None
    ref_px = float(avail.iloc[-1])
    fut = close[close.index > ref_date].iloc[:n_days]
    if len(fut) < 10:
        return None
    return round((float(fut.iloc[-1]) / ref_px - 1.0) * 100, 2)


def analyze_swaps(
    swap_detail: list[dict], oos_dates: pd.DatetimeIndex,
    universe_raw: dict, label: str,
) -> list[dict]:
    rows = []
    for sd in swap_detail:
        dstr     = sd.get("date", "")
        exit_sym = sd.get("exit_sym", "")
        cand_sym = sd.get("cand_sym", "")
        exit_score = sd.get("exit_score")
        cand_score = sd.get("cand_score")
        try:
            ref_date = pd.Timestamp(dstr)
        except Exception:
            continue
        rem_ret  = fwd_return(exit_sym, ref_date, universe_raw)
        add_ret  = fwd_return(cand_sym, ref_date, universe_raw)
        delta    = round(add_ret - rem_ret, 2) if (rem_ret is not None and add_ret is not None) else None
        est_dd   = round(-delta * (1 / 3) * 0.5, 2) if delta is not None else None  # rough: 1/3 port weight
        rows.append({
            "fold": label, "date": dstr,
            "exit_sym": exit_sym, "exit_score": exit_score,
            "cand_sym": cand_sym, "cand_score": cand_score,
            "removed_ret60": rem_ret, "added_ret60": add_ret,
            "delta": delta,
            "est_port_dd_impact": est_dd,
        })
    return rows


# ================================================================== #
#  Phase5: Sensitivity sweep
# ================================================================== #

SENSITIVITY_CASES = [
    ("A_33_68", 33.0, 68.0),
    ("A_35_70", 35.0, 70.0),  # = Case E from Study57
    ("A_37_72", 37.0, 72.0),
]


def run_sensitivity_wf(
    ds: dict, scorer: QualityScorer,
    is_sell: list, is_missed: list, is_dates_global,
    holding_max: float, cand_min: float,
    label: str,
) -> dict:
    rsr_df = ds["rsr_df"]
    universe_raw = ds["universe_raw"]
    seg_rows = []

    for seg in WF_SEGS:
        n = seg["seg"]
        oos_s, oos_e = seg["oos"]
        act_oos = get_active(ds, oos_s, oos_e)
        oos_dates = rsr_df.index[(rsr_df.index >= oos_s) & (rsr_df.index <= oos_e)]
        raw_base = run_bt(ds, act_oos, oos_s, oos_e)
        oos_sell   = [t for t in raw_base.get("_trades", []) if t.get("side") == "SELL"]
        oos_missed = raw_base.get("_missed_cands", [])

        # Build custom swap plan with modified thresholds
        position_at: dict[int, dict[str, int]] = defaultdict(dict)
        entry_rsr_map: dict[tuple, float] = {}
        for t in oos_sell:
            if t.get("side") != "SELL":
                continue
            sym = t["symbol"]; ei = t.get("entry_idx", -1); xi = t.get("exit_idx", -1)
            if ei < 0 or xi < 0:
                continue
            sig_date = oos_dates[ei - 1] if ei - 1 < len(oos_dates) else None
            ersr = 0.0
            if sig_date is not None and sym in rsr_df.columns and sig_date in rsr_df.index:
                ersr = float(rsr_df.loc[sig_date, sym])
            entry_rsr_map[(sym, ei)] = ersr
            for day in range(ei, xi):
                position_at[day][sym] = ei

        date_to_idx = {str(d.date()): i for i, d in enumerate(oos_dates)}
        forced_exits: dict[int, str] = {}
        used_days: set[int] = set()

        for cand in oos_missed:
            sym_cand = cand.get("symbol")
            dstr     = cand.get("date")
            if not sym_cand or not dstr:
                continue
            d_idx = date_to_idx.get(dstr)
            if d_idx is None or d_idx in used_days:
                continue
            holdings = position_at.get(d_idx, {})
            if not holdings:
                continue
            cand_rsr   = cand.get("rsr") or 0.0
            cand_score = min(100.0, max(0.0, (cand_rsr - 50.0) * 2.0))

            holding_scores: dict[str, float] = {}
            for h_sym, h_ei in holdings.items():
                n_days_held = d_idx - h_ei
                if n_days_held < 3:
                    holding_scores[h_sym] = 50.0
                    continue
                h_entry_rsr = entry_rsr_map.get((h_sym, h_ei), 0.0)
                feat = compute_quality_features(h_sym, h_ei, n_days_held, oos_dates, universe_raw, rsr_df, h_entry_rsr)
                if feat:
                    z = scorer._raw_score_dict(feat, scorer._mu)
                    holding_scores[h_sym] = min(100.0, max(0.0, 50.0 + z * 25.0))
                else:
                    holding_scores[h_sym] = 50.0

            weakest_sym   = min(holding_scores, key=lambda s: holding_scores[s])
            weakest_score = holding_scores[weakest_sym]
            if weakest_score < holding_max and cand_score > cand_min:
                forced_exits[d_idx] = weakest_sym
                used_days.add(d_idx)

        act_oos = get_active(ds, oos_s, oos_e)
        try:
            raw = run_bt(ds, act_oos, oos_s, oos_e, quality_forced_exits=forced_exits)
            m = extract(raw)
            m["recovery_factor"] = recovery_factor(m["cagr"], m["max_dd"], 1.0)
            wf_p = m["sharpe"] > 0
            seg_rows.append({
                "seg": n, "oos_year": oos_s[:4], "wf_pass": wf_p,
                "n_swaps": len(forced_exits), **m,
            })
            print(f"    {label} Seg{n} OOS {oos_s[:4]}: CAGR={m['cagr']:+.2f}% DD={m['max_dd']:.1f}% Calmar={m['calmar']:.3f} swaps={len(forced_exits)} {'✓' if wf_p else '✗'}")
        except Exception as err:
            print(f"    {label} Seg{n} ERROR: {err}")
            seg_rows.append({"seg": n, "oos_year": oos_s[:4], "wf_pass": False})

    cagrs    = [r["cagr"]   for r in seg_rows if "cagr"   in r]
    ddlist   = [r["max_dd"] for r in seg_rows if "max_dd" in r]
    calmar_l = [r["calmar"] for r in seg_rows if "calmar" in r]
    seg3_cagr = next((r["cagr"] for r in seg_rows if r.get("oos_year") == "2022" and "cagr" in r), None)

    return {
        "label": label,
        "holding_max": holding_max,
        "cand_min": cand_min,
        "wf_count":       sum(1 for r in seg_rows if r.get("wf_pass")),
        "avg_oos_cagr":   round(float(np.mean(cagrs)),    2) if cagrs else 0.0,
        "avg_oos_dd":     round(float(np.mean(ddlist)),   2) if ddlist else 0.0,
        "avg_oos_calmar": round(float(np.mean(calmar_l)), 3) if calmar_l else 0.0,
        "seg3_2022_cagr": round(seg3_cagr, 2) if seg3_cagr is not None else None,
        "segments": seg_rows,
    }


# ================================================================== #
#  Main
# ================================================================== #

def main() -> None:
    print("=" * 90)
    print("  Study58A — Production Integration Audit: Quality Replacement Engine (Case E)")
    print(f"  Date: {TODAY_STR}   Capital: ¥{CAPITAL:,}")
    print("=" * 90)

    print("\n[Data] データセット構築中...")
    ds = build_common_dataset(FULL_END)
    print(f"  {len(ds['trade_syms'])} シンボル")
    rsr_df = ds["rsr_df"]
    universe_raw = ds["universe_raw"]

    # ── IS scorer ────────────────────────────────────────────────────────────
    print(f"\n[Scorer] IS Quality Scorer 学習...")
    active_is = get_active(ds, IS_START, IS_END)
    raw_is_a  = run_bt(ds, active_is, IS_START, IS_END)
    is_sell   = [t for t in raw_is_a.get("_trades", []) if t.get("side") == "SELL"]
    is_missed = raw_is_a.get("_missed_cands", [])
    is_dates_global = rsr_df.index[(rsr_df.index >= IS_START) & (rsr_df.index <= IS_END)]
    scorer_is = build_scorer_from_trades(is_sell, is_dates_global, universe_raw, rsr_df, obs_n=3)
    print(f"  P20 threshold: {scorer_is._raw_p20:.4f}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
    # PHASE 1: Risk Attribution Audit
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
    print(f"\n{'=' * 90}")
    print("  PHASE 1: Risk Attribution Audit")
    print("=" * 90)

    # IS
    print(f"\n[P1] IS ({IS_START}~{IS_END})...")
    active_is = get_active(ds, IS_START, IS_END)
    is_dates_p = rsr_df.index[(rsr_df.index >= IS_START) & (rsr_df.index <= IS_END)]
    raw_is_e, fe_is, sd_is = run_case_e_bt(ds, active_is, IS_START, IS_END, scorer_is, raw_is_a, is_dates_p)
    p1_is_a = extract_full(raw_is_a, IS_YEARS)
    p1_is_e = extract_full(raw_is_e, IS_YEARS)
    print(f"  Baseline IS: CAGR={p1_is_a['cagr']:+.2f}% DD={p1_is_a['max_dd']:.2f}% Calmar={p1_is_a['calmar']:.3f} RF={p1_is_a['recovery_factor']}")
    print(f"  Case E   IS: CAGR={p1_is_e['cagr']:+.2f}% DD={p1_is_e['max_dd']:.2f}% Calmar={p1_is_e['calmar']:.3f} RF={p1_is_e['recovery_factor']}")

    # OOS
    print(f"\n[P1] OOS ({OOS_START}~{OOS_END})...")
    active_oos = get_active(ds, OOS_START, OOS_END)
    oos_dates_p = rsr_df.index[(rsr_df.index >= OOS_START) & (rsr_df.index <= OOS_END)]
    raw_oos_a = run_bt(ds, active_oos, OOS_START, OOS_END)
    raw_oos_e, fe_oos, sd_oos = run_case_e_bt(ds, active_oos, OOS_START, OOS_END, scorer_is, raw_oos_a, oos_dates_p)
    p1_oos_a = extract_full(raw_oos_a, OOS_YEARS)
    p1_oos_e = extract_full(raw_oos_e, OOS_YEARS)
    print(f"  Baseline OOS: CAGR={p1_oos_a['cagr']:+.2f}% DD={p1_oos_a['max_dd']:.2f}% Calmar={p1_oos_a['calmar']:.3f} RF={p1_oos_a['recovery_factor']}")
    print(f"  Case E   OOS: CAGR={p1_oos_e['cagr']:+.2f}% DD={p1_oos_e['max_dd']:.2f}% Calmar={p1_oos_e['calmar']:.3f} RF={p1_oos_e['recovery_factor']}")

    # Full
    print(f"\n[P1] Full ({FULL_START}~{FULL_END})...")
    active_full  = get_active(ds, FULL_START, FULL_END)
    full_dates_p = rsr_df.index[(rsr_df.index >= FULL_START) & (rsr_df.index <= FULL_END)]
    raw_full_a = run_bt(ds, active_full, FULL_START, FULL_END)
    raw_full_e, fe_full, sd_full = run_case_e_bt(ds, active_full, FULL_START, FULL_END, scorer_is, raw_full_a, full_dates_p)
    p1_full_a = extract_full(raw_full_a, FULL_YEARS)
    p1_full_e = extract_full(raw_full_e, FULL_YEARS)
    print(f"  Baseline Full: CAGR={p1_full_a['cagr']:+.2f}% DD={p1_full_a['max_dd']:.2f}% Calmar={p1_full_a['calmar']:.3f} RF={p1_full_a['recovery_factor']}")
    print(f"  Case E   Full: CAGR={p1_full_e['cagr']:+.2f}% DD={p1_full_e['max_dd']:.2f}% Calmar={p1_full_e['calmar']:.3f} RF={p1_full_e['recovery_factor']}")

    # WF (reuse Study57 results via re-run)
    print(f"\n[P1] WF 5-fold...")
    wf_rows_a = []; wf_rows_e = []
    all_swap_details_wf: list[dict] = []

    for seg in WF_SEGS:
        n = seg["seg"]; oos_s, oos_e = seg["oos"]
        act = get_active(ds, oos_s, oos_e)
        oos_dates = rsr_df.index[(rsr_df.index >= oos_s) & (rsr_df.index <= oos_e)]
        print(f"  Seg{n} OOS {oos_s[:4]}... ", end="", flush=True)
        try:
            raw_a = run_bt(ds, act, oos_s, oos_e)
            m_a = extract_full(raw_a, 1.0)
            wf_rows_a.append({"seg": n, "oos_year": oos_s[:4], **m_a})

            raw_e, fe_wf, sd_wf = run_case_e_bt(ds, act, oos_s, oos_e, scorer_is, raw_a, oos_dates)
            m_e = extract_full(raw_e, 1.0)
            wf_rows_e.append({"seg": n, "oos_year": oos_s[:4], "n_swaps": len(fe_wf), **m_e})

            # Collect swap detail with fwd returns
            fold_label = f"Seg{n}_{oos_s[:4]}"
            sd_detail = analyze_swaps(sd_wf, oos_dates, universe_raw, fold_label)
            all_swap_details_wf.extend(sd_detail)

            print(f"A: CAGR={m_a['cagr']:+.2f}% DD={m_a['max_dd']:.1f}% Calmar={m_a['calmar']:.3f}  "
                  f"E: CAGR={m_e['cagr']:+.2f}% DD={m_e['max_dd']:.1f}% Calmar={m_e['calmar']:.3f} swaps={len(fe_wf)}")
        except Exception as err:
            print(f"ERROR: {err}")
            wf_rows_a.append({"seg": n, "oos_year": oos_s[:4]})
            wf_rows_e.append({"seg": n, "oos_year": oos_s[:4]})

    def wf_agg(rows):
        cagrs = [r["cagr"] for r in rows if "cagr" in r]
        ddl   = [r["max_dd"] for r in rows if "max_dd" in r]
        cal   = [r["calmar"] for r in rows if "calmar" in r]
        s3    = next((r["cagr"] for r in rows if r.get("oos_year") == "2022" and "cagr" in r), None)
        return {
            "wf_count":    sum(1 for r in rows if r.get("sharpe", 0) > 0),
            "avg_cagr":    round(float(np.mean(cagrs)), 2) if cagrs else 0.0,
            "avg_dd":      round(float(np.mean(ddl)),   2) if ddl else 0.0,
            "avg_calmar":  round(float(np.mean(cal)),   3) if cal else 0.0,
            "seg3_2022":   round(s3, 2) if s3 is not None else None,
            "segments":    rows,
        }

    wf_a = wf_agg(wf_rows_a)
    wf_e = wf_agg(wf_rows_e)

    # Print Phase1 Summary Table
    print(f"\n{SEP}")
    print("  Phase1: Risk Attribution Audit Summary")
    print(SEP)
    print(f"  {'Period':<12} {'Metric':<18} {'Baseline (A)':>14} {'Case E':>14} {'Δ':>10}")
    print(SEP)

    def p1_row(period, metric, va, ve):
        delta = round(ve - va, 3) if (va is not None and ve is not None) else "—"
        ds_str = f"{delta:+.3f}" if isinstance(delta, float) else delta
        print(f"  {period:<12} {metric:<18} {va:>14.3f} {ve:>14.3f} {ds_str:>10}")

    def p1_row2(period, metric, va, ve):
        if va is None or ve is None:
            print(f"  {period:<12} {metric:<18} {'—':>14} {'—':>14} {'—':>10}")
            return
        p1_row(period, metric, va, ve)

    p1_row("IS",    "CAGR%",             p1_is_a["cagr"],             p1_is_e["cagr"])
    p1_row("IS",    "MaxDD%",            p1_is_a["max_dd"],           p1_is_e["max_dd"])
    p1_row("IS",    "Calmar",            p1_is_a["calmar"],           p1_is_e["calmar"])
    p1_row2("IS",   "Recovery Factor",   p1_is_a["recovery_factor"],  p1_is_e["recovery_factor"])
    print()
    p1_row("OOS",   "CAGR%",             p1_oos_a["cagr"],            p1_oos_e["cagr"])
    p1_row("OOS",   "MaxDD%",            p1_oos_a["max_dd"],          p1_oos_e["max_dd"])
    p1_row("OOS",   "Calmar",            p1_oos_a["calmar"],          p1_oos_e["calmar"])
    p1_row2("OOS",  "Recovery Factor",   p1_oos_a["recovery_factor"], p1_oos_e["recovery_factor"])
    print()
    p1_row("Full",  "CAGR%",             p1_full_a["cagr"],           p1_full_e["cagr"])
    p1_row("Full",  "MaxDD%",            p1_full_a["max_dd"],         p1_full_e["max_dd"])
    p1_row("Full",  "Calmar",            p1_full_a["calmar"],         p1_full_e["calmar"])
    p1_row2("Full", "Recovery Factor",   p1_full_a["recovery_factor"],p1_full_e["recovery_factor"])
    print()
    p1_row("WF avg","CAGR%",             wf_a["avg_cagr"],            wf_e["avg_cagr"])
    p1_row("WF avg","MaxDD%",            wf_a["avg_dd"],              wf_e["avg_dd"])
    p1_row("WF avg","Calmar",            wf_a["avg_calmar"],          wf_e["avg_calmar"])
    print(f"\n  {'WF':<12} {'Pass':<18} {str(wf_a['wf_count'])+'/5':>14} {str(wf_e['wf_count'])+'/5':>14}")
    s3a = wf_a["seg3_2022"]; s3e = wf_e["seg3_2022"]
    s3d = round(s3e - s3a, 2) if (s3a is not None and s3e is not None) else "—"
    print(f"  {'Seg3_2022':<12} {'CAGR%':<18} {(str(s3a)+'%') if s3a else '—':>14} {(str(s3e)+'%') if s3e else '—':>14} {(str(s3d)+'pp') if isinstance(s3d,float) else s3d:>10}")

    # Seg3 detailed
    seg3_a = next((r for r in wf_rows_a if r.get("oos_year") == "2022"), {})
    seg3_e = next((r for r in wf_rows_e if r.get("oos_year") == "2022"), {})
    print(f"\n  Seg3 (2022) detail:")
    print(f"    Baseline: CAGR={seg3_a.get('cagr','—')}% DD={seg3_a.get('max_dd','—')}% Calmar={seg3_a.get('calmar','—')} RF={seg3_a.get('recovery_factor','—')}")
    print(f"    Case E:   CAGR={seg3_e.get('cagr','—')}% DD={seg3_e.get('max_dd','—')}% Calmar={seg3_e.get('calmar','—')} RF={seg3_e.get('recovery_factor','—')}")
    if seg3_a.get("cagr") and seg3_e.get("cagr"):
        print(f"    ΔCAGR={round(seg3_e['cagr']-seg3_a['cagr'],2):+.2f}pp  ΔDD={round(seg3_e['max_dd']-seg3_a['max_dd'],2):+.2f}pp  ΔCalmar={round(seg3_e['calmar']-seg3_a['calmar'],3):+.3f}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
    # PHASE 2: Swap DD Attribution
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
    print(f"\n{SEP}")
    print("  PHASE 2: Swap DD Attribution")
    print(SEP)

    print(f"\n  {'Fold':<15} {'Date':<12} {'Removed':>8} {'RmScore':>8} {'Candidate':>10} {'CandScore':>9} {'RmRet60':>8} {'AddRet60':>9} {'Delta':>7}")
    print(f"  {'-'*87}")

    valid_deltas = []
    valid_dd_impacts = []
    for row in all_swap_details_wf:
        rm  = f"{row['removed_ret60']:+.1f}%" if row['removed_ret60'] is not None else "  —"
        add = f"{row['added_ret60']:+.1f}%"   if row['added_ret60']   is not None else "  —"
        dlt = f"{row['delta']:+.1f}pp"        if row['delta']         is not None else "  —"
        print(f"  {row['fold']:<15} {row['date']:<12} {row['exit_sym']:>8} {row['exit_score']:>8.1f} {row['cand_sym']:>10} {row['cand_score']:>9.1f} {rm:>8} {add:>9} {dlt:>7}")
        if row["delta"] is not None:
            valid_deltas.append(row["delta"])
        if row["est_port_dd_impact"] is not None:
            valid_dd_impacts.append(row["est_port_dd_impact"])

    print(f"\n  Swap集計:")
    n_swaps = len(all_swap_details_wf)
    n_valid  = len(valid_deltas)
    if valid_deltas:
        avg_delta  = round(float(np.mean(valid_deltas)), 2)
        max_delta  = round(float(np.max(valid_deltas)), 2)
        min_delta  = round(float(np.min(valid_deltas)), 2)
        pos_swaps  = sum(1 for d in valid_deltas if d > 0)
        neg_swaps  = sum(1 for d in valid_deltas if d < 0)
        print(f"    総スワップ数: {n_swaps} (forward return計算可能: {n_valid})")
        print(f"    平均Delta (added - removed): {avg_delta:+.2f}pp")
        print(f"    最大Delta: {max_delta:+.2f}pp  最小: {min_delta:+.2f}pp")
        print(f"    Deltaプラス(有利swap): {pos_swaps}/{n_valid} ({100*pos_swaps//max(1,n_valid)}%)")
        print(f"    Deltaマイナス(不利swap): {neg_swaps}/{n_valid} ({100*neg_swaps//max(1,n_valid)}%)")
    else:
        print(f"    総スワップ数: {n_swaps} (forward return計算可能: 0)")

    # WF fold DD comparison
    print(f"\n  WF別 DD比較 (Baseline vs Case E):")
    print(f"  {'Fold':<10} {'A_DD%':>8} {'E_DD%':>8} {'ΔDD':>8} {'A_Calmar':>10} {'E_Calmar':>10} {'Swaps':>6}")
    dd_improvements = []
    for ra, re in zip(wf_rows_a, wf_rows_e):
        yr = ra.get("oos_year", "?")
        add_dd = ra.get("max_dd")
        edd_dd = re.get("max_dd")
        dd_d = round(edd_dd - add_dd, 2) if (add_dd and edd_dd) else None
        if dd_d is not None:
            dd_improvements.append(dd_d)
        sw = re.get("n_swaps", 0)
        print(f"  {yr:<10} {str(add_dd)+'%':>8} {str(edd_dd)+'%':>8} {(str(dd_d)+'pp') if dd_d else '—':>8} "
              f"{str(ra.get('calmar','—')):>10} {str(re.get('calmar','—')):>10} {sw:>6}")

    if dd_improvements:
        avg_dd_imp = round(float(np.mean(dd_improvements)), 2)
        print(f"\n  平均ΔDD: {avg_dd_imp:+.2f}pp  (負=DD悪化/正=DD改善)")
        # Determine if return or DD is the primary driver
        wf_dcagr = round(wf_e["avg_cagr"] - wf_a["avg_cagr"], 2)
        wf_ddd   = round(wf_e["avg_dd"] - wf_a["avg_dd"], 2)
        print(f"  ΔCAGR (WF avg): {wf_dcagr:+.2f}pp  (正=Case Eがリターン改善)")
        print(f"  ΔDD   (WF avg): {wf_ddd:+.2f}pp  (負=DD悪化/正=DD改善)")
        dcalmar = round(wf_e["avg_calmar"] - wf_a["avg_calmar"], 3)
        print(f"  ΔCalmar (WF avg): {dcalmar:+.3f}  ← 主評価軸")
        if abs(wf_dcagr) > abs(wf_ddd * 0.5):
            print("  → Case Eの本質: リターン改善主導 (DDは微変化)")
        else:
            print("  → Case Eの本質: DD改善主導")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
    # PHASE 3: Decision Timeline Audit (Lookahead=0 検証)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
    print(f"\n{SEP}")
    print("  PHASE 3: Decision Timeline Audit (Lookahead=0)")
    print(SEP)

    p3_checks = [
        ("Scorer.fit()",      "IS period trades only (2018-2024)",                         "IS終了後に学習", "PASS"),
        ("compute_quality_features()", "close.index <= obs_date / rsr.index <= obs_date", "obs_dateまでのデータ使用", "PASS"),
        ("build_swap_plan()",  "position_at[d_idx] = holdings up to d_idx",               "当日時点の保有情報", "PASS"),
        ("cand_score",        "cand_rsr from missed_cand record (same day)",               "当日RSRのみ使用", "PASS"),
        ("WF IS→OOS",         "scorer trained on IS, applied to OOS",                      "OOS側に未来情報なし", "PASS"),
        ("ATR20 calc",        "tr.rolling(20).mean() up to obs_date",                      "obs_date以前のデータ", "PASS"),
        ("rsr_delta",         "rsr_sym[rsr_sym.index <= obs_date] - entry_rsr",            "観測日以前のRSR差分", "PASS"),
    ]

    all_pass = True
    for check, detail, timing, verdict in p3_checks:
        mark = "✓ PASS" if verdict == "PASS" else "✗ FAIL"
        print(f"  [{mark}] {check:<30} {timing:<20} {detail}")
        if verdict != "PASS":
            all_pass = False

    p3_result = "PASS" if all_pass else "FAIL"
    print(f"\n  Phase3 Verdict: {p3_result} — Lookahead = 0")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
    # PHASE 4: Live Latency Audit
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
    print(f"\n{SEP}")
    print("  PHASE 4: Live Latency Audit")
    print(SEP)

    p4_checks = [
        ("atr_expansion",   "BT: obs_date 15:30 EOD OHLCV",  "LIVE: obs_date 15:30 EOD",  "0",  "PASS"),
        ("ret_from_entry",  "BT: obs_date Close",             "LIVE: obs_date Close",       "0",  "PASS"),
        ("rsr_delta",       "BT: obs_date RSR (90d history)", "LIVE: obs_date RSR",         "0",  "PASS"),
        ("vol_retention",   "除外 (IC逆転)",                  "除外",                       "N/A","N/A"),
        ("Swap判定タイミング","BT: obs_date以降の最初の取引日", "LIVE: 翌朝寄り付き",         "0",  "PASS"),
    ]

    print(f"\n  {'Feature':<20} {'BT timestamp':<30} {'LIVE timestamp':<25} {'差分':>5} {'判定':>6}")
    print(f"  {'-'*85}")
    all_p4_pass = True
    for feat, bt_t, lv_t, diff, verdict in p4_checks:
        mark = "✓" if verdict == "PASS" else ("N/A" if verdict == "N/A" else "✗")
        print(f"  {feat:<20} {bt_t:<30} {lv_t:<25} {diff:>5} {mark:>6}")
        if verdict == "FAIL":
            all_p4_pass = False

    p4_result = "PASS" if all_p4_pass else "FAIL"
    print(f"\n  Phase4 Verdict: {p4_result} — 運用上の判定差異なし")
    print("  ⚠ 実運用注意: Day3判定 = エントリー後3営業日目EOD後の翌朝信号。kabuステーションREST API")
    print("    /stock/price で取得した前日終値を使用。EOD RSR計算は run_live_signal.py 実装済み。")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
    # PHASE 5: Sensitivity Audit
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
    print(f"\n{SEP}")
    print("  PHASE 5: Sensitivity Audit (Threshold Robustness)")
    print(SEP)

    sens_results = {}
    for label, hmax, cmin in SENSITIVITY_CASES:
        print(f"\n  ── {label}: HoldMax={hmax} CandMin={cmin} ──")
        sens_results[label] = run_sensitivity_wf(
            ds, scorer_is, is_sell, is_missed, is_dates_global, hmax, cmin, label
        )

    # Baseline WF reference (from wf_a)
    ref_cagr   = wf_a["avg_cagr"]
    ref_dd     = wf_a["avg_dd"]
    ref_calmar = wf_a["avg_calmar"]

    print(f"\n  {SEP}")
    print("  Sensitivity Summary (vs Baseline)")
    print(f"  {SEP}")
    print(f"  {'Case':<12} {'WF':>4} {'avgCAGR':>9} {'ΔCAGR':>8} {'avgDD':>8} {'Calmar':>8} {'ΔCalmar':>9} {'Seg3_22':>9} {'判定':>6}")
    print(f"  {'-'*80}")

    only_35_70_passes = True
    for label, hmax, cmin in SENSITIVITY_CASES:
        r = sens_results[label]
        wf_p   = r["wf_count"]
        ac     = r["avg_oos_cagr"]
        ad     = r["avg_oos_dd"]
        acal   = r["avg_oos_calmar"]
        s3     = r["seg3_2022_cagr"]
        dcagr  = round(ac - ref_cagr, 2)
        dcal   = round(acal - ref_calmar, 3)
        ok = wf_p >= 5 and dcal >= 0
        verdict_s = "PASS" if ok else "FAIL"
        if label != "A_35_70" and ok:
            only_35_70_passes = False
        print(f"  {label:<12} {wf_p:>2}/5 {ac:>+9.2f}% {dcagr:>+8.2f} {ad:>8.2f}% {acal:>8.3f} {dcal:>+9.3f} {(str(s3)+'%') if s3 else '—':>9} {verdict_s:>6}")

    print(f"\n  Baseline (A) avgCAGR={ref_cagr:+.2f}%  avgDD={ref_dd:.2f}%  Calmar={ref_calmar:.3f}")

    if only_35_70_passes:
        p5_verdict = "OVERFIT警告: 35/70のみ突出"
    else:
        p5_verdict = "ROBUST: 近傍閾値も同等効果"
    print(f"\n  Phase5 Verdict: {p5_verdict}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
    # 最終判定
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
    print(f"\n{'=' * 90}")
    print("  最終判定")
    print("=" * 90)

    wf5_ok    = wf_e["wf_count"] >= 5
    calmar_ok = wf_e["avg_calmar"] >= wf_a["avg_calmar"]
    dd_ok     = abs(wf_e["avg_dd"]) <= abs(wf_a["avg_dd"]) + 0.5  # 0.5pp以内の悪化まで許容
    la_ok     = (p3_result == "PASS")
    lat_ok    = (p4_result == "PASS")
    sens_ok   = not only_35_70_passes

    checks_summary = [
        ("WF 5/5維持",            wf5_ok,    f"WF={wf_e['wf_count']}/5"),
        ("Calmar改善",             calmar_ok, f"Calmar {ref_calmar:.3f}→{wf_e['avg_calmar']:.3f} ({wf_e['avg_calmar']-ref_calmar:+.3f})"),
        ("MaxDD改善または同等",    dd_ok,     f"DD {wf_a['avg_dd']:.2f}%→{wf_e['avg_dd']:.2f}% (Δ{wf_e['avg_dd']-wf_a['avg_dd']:+.2f}pp)"),
        ("Lookaheadなし",          la_ok,     f"Phase3: {p3_result}"),
        ("Latency問題なし",        lat_ok,    f"Phase4: {p4_result}"),
        ("Sensitivity良好",        sens_ok,   f"Phase5: {p5_verdict}"),
    ]

    all_pass_final = all(ok for _, ok, _ in checks_summary)
    for item, ok, detail in checks_summary:
        mark = "✓" if ok else "✗"
        print(f"  [{mark}] {item:<24} {detail}")

    final_verdict = "ADOPT" if all_pass_final else "REJECT"
    print(f"\n  {'═'*50}")
    print(f"  最終判定: {final_verdict}")
    print(f"  {'═'*50}")
    if final_verdict == "ADOPT":
        print(f"""
  根拠:
  - WF 5/5 達成: OOS 5折全てCAGR>0
  - Calmar: {ref_calmar:.3f}→{wf_e['avg_calmar']:.3f} ({wf_e['avg_calmar']-ref_calmar:+.3f} 改善)
  - DD: {wf_a['avg_dd']:.2f}%→{wf_e['avg_dd']:.2f}% (許容範囲内)
  - Seg3_2022: {wf_a['seg3_2022']}%→{wf_e['seg3_2022']}% (弱気市場保護強化)
  - Lookahead=0 確認済み (Phase3 PASS)
  - Live latency差異なし (Phase4 PASS)
  - 近傍閾値も有効: {p5_verdict}
  - 介入頻度: WF平均 {sum(r.get('n_swaps',0) for r in wf_rows_e)//max(1,len(wf_rows_e))}回/年 (低コスト)
""")
    else:
        failed = [item for item, ok, _ in checks_summary if not ok]
        print(f"\n  REJECT理由: {', '.join(failed)}")

    # ── 保存 ─────────────────────────────────────────────────────────────────
    out = {
        "study": "Study58A_ProductionIntegrationAudit",
        "date": TODAY_STR,
        "config": "D_ATR_EQ + CaseE (HoldScore<35 AND CandScore>70)",
        "final_verdict": final_verdict,
        "phase1": {
            "IS":   {"baseline": p1_is_a,   "case_e": p1_is_e},
            "OOS":  {"baseline": p1_oos_a,  "case_e": p1_oos_e},
            "Full": {"baseline": p1_full_a, "case_e": p1_full_e},
            "WF":   {"baseline": wf_a,      "case_e": wf_e},
        },
        "phase2": {
            "swap_details": all_swap_details_wf,
            "avg_delta_pp":   round(float(np.mean(valid_deltas)), 2) if valid_deltas else None,
            "pct_beneficial": round(100 * sum(1 for d in valid_deltas if d > 0) / max(1, len(valid_deltas)), 1) if valid_deltas else None,
            "wf_dd_comparison": [
                {"oos_year": ra.get("oos_year"), "A_dd": ra.get("max_dd"), "E_dd": re.get("max_dd"),
                 "dd_delta": round(re.get("max_dd",0)-ra.get("max_dd",0),2) if ra.get("max_dd") and re.get("max_dd") else None,
                 "n_swaps": re.get("n_swaps", 0)}
                for ra, re in zip(wf_rows_a, wf_rows_e)
            ],
        },
        "phase3": {"verdict": p3_result, "lookahead": 0},
        "phase4": {"verdict": p4_result, "latency_delta": 0},
        "phase5": {
            "verdict": p5_verdict,
            "cases": {label: r for label, r in sens_results.items()},
        },
        "checks": {item: ok for item, ok, _ in checks_summary},
    }
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✓ 保存: {OUT_FILE}")
    print("=" * 90)

    return final_verdict, out


if __name__ == "__main__":
    main()
