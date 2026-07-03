"""
backtest/study46_voladj_addon_interaction_wf_202606.py

Study46 — VolAdj × Addon Interaction Walk-Forward
目的: Study41 D_VOL_ADJ と Study45 D_EQ_SCALE が加法的か相互依存かを定量測定。

2×2 factorial design (S5 baseline):
  A: S5 baseline
  B: S5 + D_VOL_ADJ only
  C: S5 + D_EQ_SCALE only
  D: S5 + D_VOL_ADJ + D_EQ_SCALE (combined)

interaction_pp = D - B - C + A
  |interaction| ≤ 0.5pp → Additive
  interaction < -0.5pp  → Cannibalization
  interaction > +0.5pp  → Synergy

Output: backtests/study46_voladj_addon_interaction_wf_202606_<date>.json
"""
from __future__ import annotations

import sys
import json
import warnings
from pathlib import Path
from datetime import date

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

from src.backtest import composite_alpha_bt as cab
from src.backtest.snapshot_archaeology_202606 import build_common_dataset
from src.backtest.wf_dynamic_universe import WF_SEGS, TRUE_OOS, IS_FULL
from src.strategy.universe import build_dyn_rsr42_active
from src.paths import RESULTS_DIR

# ── 定数 ────────────────────────────────────────────────────────────────
CAPITAL  = 3_000_000
MIN_HOLD = 3
FULL_DATA_END = "2025-12-31"
VOL_CALM_THRESHOLD = 0.008   # Study41 D_VOL_ADJ: TOPIX 20d std < 0.8%

# Study43A capital scaling estimates (from memory)
CAPITAL_20M_UPLIFT_PP = 3.32   # ¥3M → ¥20M: Full IS delta (lot制約解除)
CAPITAL_30M_UPLIFT_PP = 3.87   # ¥3M → ¥30M: Full IS delta (approximate)


def build_vol_adjusted_ts(topix_close: pd.Series, common_dates: list,
                           base_max: int = 3, calm_max: int = 4,
                           vol_threshold: float = VOL_CALM_THRESHOLD) -> pd.Series:
    topix_ret = topix_close.pct_change()
    rolling_std = topix_ret.rolling(20, min_periods=10).std()
    std_series = rolling_std.reindex(pd.Index(common_dates), method="ffill").fillna(rolling_std.median())
    max_pos_ts = pd.Series(base_max, index=std_series.index, dtype=int)
    max_pos_ts[std_series < vol_threshold] = calm_max
    return max_pos_ts


def run_case(ds: dict, sym_active_df, start: str, end: str,
             max_positions_ts, addon_policy: str = "NONE",
             addon_size_frac: float = 0.25) -> dict:
    """S5 + optional D_VOL_ADJ + optional D_EQ_SCALE addon."""
    return cab.run_scenario(
        scenario="BASELINE",
        universe_raw=ds["universe_raw"],
        rsr_df=ds["rsr_df"],
        alpha_df=None,
        regime_df=ds["regime_df"],
        trade_syms=ds["trade_syms"],
        rsr_syms=ds["rsr_syms"],
        cfg=ds["base_cfg"],
        start=start, end=end, verbose=False,
        tech_matrices=ds["tech_matrices"],
        breadth_series=ds["breadth_series"],
        capital=CAPITAL,
        min_hold=MIN_HOLD,
        topix_close=ds["topix_close"],
        market_shock_mode="composite",
        rsr_exit_threshold=70.0,
        sym_active_df=sym_active_df,
        enable_simple_rsr_exit=True,
        enable_atr_trailing_prod=True,
        enable_multilayer_rsr=True,
        enable_atr_risk_sizing=False,
        enable_mtf_filter=False,
        sizing_mode="existing",
        exit_policy="NONE",          # S5 baseline — no ATR Extension
        max_positions_ts=max_positions_ts,
        addon_policy=addon_policy,
        addon_atr_mult=1.0,
        addon_stage2_mult=2.0,
        addon_max_per_pos=1,
        addon_size_frac=addon_size_frac,
    )


def extract_metrics(res: dict) -> dict:
    return {
        "cagr":                    float(res.get("cagr", 0.0) or 0.0),
        "sharpe":                  float(res.get("sharpe", 0.0) or 0.0),
        "max_dd":                  float(res.get("max_dd", 0.0) or 0.0),
        "calmar":                  float(res.get("calmar", 0.0) or 0.0),
        "n_trades":                int(res.get("n_trades", 0) or 0),
        "addon_count":             int(res.get("addon_count", 0) or 0),
        "avg_exposure":            float(res.get("avg_exposure", 0.0) or 0.0),
        "avg_simultaneous_pos":    float(res.get("avg_simultaneous_holdings", 0.0) or 0.0),
        "idle_cash_pct":           float(res.get("avg_idle_cash_ratio_pct", 0.0) or 0.0),
        "days_at_max_pos_pct":     float(res.get("cap_saturation_rate_pct", 0.0) or 0.0),
        "q2_idle_with_winner_pct": float(res.get("q2_idle_days_with_winner_pct", 0.0) or 0.0),
    }


# ── ケース定義 ──────────────────────────────────────────────────────────
# (case_name, use_vol_adj, addon_policy)
CASES = [
    ("A_BASELINE",  False, "NONE"),
    ("B_VOL_ADJ",   True,  "NONE"),
    ("C_EQ_SCALE",  False, "D"),
    ("D_COMBINED",  True,  "D"),
]


def run_wf_all_cases(ds: dict, get_active, vol_adj_ts: pd.Series) -> dict[str, dict]:
    all_results: dict[str, dict] = {}

    for case_name, use_va, addon_pol in CASES:
        mpts = vol_adj_ts if use_va else None
        print(f"\n  ── {case_name}  vol_adj={use_va}  addon={addon_pol} ──")

        seg_results = []
        for seg in WF_SEGS:
            n = seg["seg"]
            is_s, is_e = seg["is"]
            oos_s, oos_e = seg["oos"]

            r_oos = run_case(ds, get_active(oos_e), oos_s, oos_e, mpts, addon_pol)
            m = extract_metrics(r_oos)
            wf_pass = m["sharpe"] > 0

            seg_results.append({
                "seg": n, "oos_year": oos_s[:4],
                **{k: round(v, 3) if isinstance(v, float) else v for k, v in m.items()},
                "wf_pass": wf_pass,
            })
            mark = "✓" if wf_pass else "✗"
            print(f"    Seg{n} {oos_s[:4]}  CAGR={m['cagr']:+.2f}%  Sh={m['sharpe']:.3f}  "
                  f"DD={m['max_dd']:.1f}%  Exp={m['avg_exposure']:.1f}%  "
                  f"AvgPos={m['avg_simultaneous_pos']:.2f}  Addons={m['addon_count']}  {mark}")

        # True OOS 2025
        r_true = run_case(ds, get_active(TRUE_OOS[1]), TRUE_OOS[0], TRUE_OOS[1], mpts, addon_pol)
        true_m = extract_metrics(r_true)
        print(f"    True OOS 2025: CAGR={true_m['cagr']:+.2f}%  Sh={true_m['sharpe']:.3f}  "
              f"Addons={true_m['addon_count']}  Exp={true_m['avg_exposure']:.1f}%")

        # Full IS 2018-2024
        r_full = run_case(ds, get_active(IS_FULL[1]), IS_FULL[0], IS_FULL[1], mpts, addon_pol)
        full_m = extract_metrics(r_full)
        print(f"    Full IS 2018-24: CAGR={full_m['cagr']:+.2f}%  Sh={full_m['sharpe']:.3f}  "
              f"DD={full_m['max_dd']:.1f}%  Cal={full_m['calmar']:.3f}  "
              f"Exp={full_m['avg_exposure']:.1f}%  Addons={full_m['addon_count']}  "
              f"IdleCash={full_m['idle_cash_pct']:.1f}%")

        wf_count = sum(1 for s in seg_results if s["wf_pass"])
        all_results[case_name] = {
            "case": case_name,
            "use_vol_adj": use_va,
            "addon_policy": addon_pol,
            "wf_segments": seg_results,
            "wf_count": wf_count,
            "avg_oos_cagr":   round(float(np.mean([s["cagr"] for s in seg_results])), 2),
            "avg_oos_sharpe": round(float(np.mean([s["sharpe"] for s in seg_results])), 3),
            "avg_oos_dd":     round(float(np.mean([s["max_dd"] for s in seg_results])), 2),
            "avg_oos_calmar": round(float(np.mean([s["calmar"] for s in seg_results])), 3),
            "avg_oos_exposure":       round(float(np.mean([s["avg_exposure"] for s in seg_results])), 1),
            "avg_oos_pos":            round(float(np.mean([s["avg_simultaneous_pos"] for s in seg_results])), 2),
            "avg_oos_idle_cash":      round(float(np.mean([s["idle_cash_pct"] for s in seg_results])), 1),
            "avg_oos_addon_count":    int(np.sum([s["addon_count"] for s in seg_results])),
            "seg3_2022": next((s["cagr"] for s in seg_results if s["oos_year"] == "2022"), None),
            "true_oos_2025": true_m,
            "full_is_2018_2024": full_m,
        }

    return all_results


def compute_interaction(results: dict[str, dict], metric: str = "avg_oos_cagr") -> dict:
    A = results["A_BASELINE"][metric]
    B = results["B_VOL_ADJ"][metric]
    C = results["C_EQ_SCALE"][metric]
    D = results["D_COMBINED"][metric]

    delta_b  = round(B - A, 3)
    delta_c  = round(C - A, 3)
    delta_d  = round(D - A, 3)
    inter    = round(D - B - C + A, 3)   # = ΔD - ΔB - ΔC

    if abs(inter) <= 0.5:
        classification = "ADDITIVE"
    elif inter < -0.5:
        classification = "CANNIBALIZATION"
    else:
        classification = "SYNERGY"

    return {
        "A_baseline":     round(A, 3),
        "B_vol_adj":      round(B, 3),
        "C_eq_scale":     round(C, 3),
        "D_combined":     round(D, 3),
        "delta_B_only":   delta_b,
        "delta_C_only":   delta_c,
        "delta_D":        delta_d,
        "expected_additive": round(delta_b + delta_c, 3),
        "interaction_pp": inter,
        "classification": classification,
    }


def main() -> int:
    print("=" * 78)
    print("  Study46 — VolAdj × Addon Interaction Walk-Forward (2×2 Factorial)")
    print(f"  Baseline: S5 (no ATR Extension)  capital={CAPITAL:,}")
    print("=" * 78)

    # ── 1. データセット ──────────────────────────────────────────────
    print(f"\n[1] データセット構築（end={FULL_DATA_END}）...")
    ds = build_common_dataset(FULL_DATA_END)
    all_syms = list(ds["trade_syms"].keys())
    print(f"  完了 ({len(all_syms)} syms)")

    # ── 2. common_dates ──────────────────────────────────────────────
    date_sets = [set(ds["universe_raw"][s]["df"].index)
                 for s in all_syms if s in ds["universe_raw"]]
    all_common_dates: list = sorted(set.intersection(*date_sets)) if date_sets else []
    print(f"  all_common_dates: {len(all_common_dates)} days")

    # ── 3. 動的ユニバース cache ───────────────────────────────────────
    print("\n[2] 動的ユニバース cache 構築...")
    bear_cfg = ds["base_cfg"].risk_controls.bear_universe_filter
    bear_exclude = list(bear_cfg.excluded_sectors) if bear_cfg.enabled else None
    active_cache: dict[str, pd.DataFrame] = {}

    def get_active(end_dt: str) -> pd.DataFrame:
        if end_dt not in active_cache:
            active_cache[end_dt] = build_dyn_rsr42_active(
                universe_raw=ds["universe_raw"], topix_close=ds["topix_close"],
                rsr_df=ds["rsr_df"], all_syms=all_syms,
                start=IS_FULL[0], end=end_dt,
                bear_exclude_sectors=bear_exclude,
                sym_sector_map=dict(ds["trade_syms"]) if bear_exclude else None,
            )
        return active_cache[end_dt]

    needed = set()
    for seg in WF_SEGS:
        needed.add(seg["oos"][1])
    needed.add(TRUE_OOS[1]); needed.add(IS_FULL[1])
    for ed in sorted(needed):
        print(f"  end={ed}..."); get_active(ed)
    print("  完了")

    # ── 4. Vol-Adjusted series ───────────────────────────────────────
    print("\n[3] D_VOL_ADJ max_positions series 構築...")
    vol_adj_ts = build_vol_adjusted_ts(ds["topix_close"], all_common_dates)
    calm_days = int((vol_adj_ts == 4).sum())
    print(f"  calm days (max=4): {calm_days}/{len(vol_adj_ts)} ({calm_days/max(1,len(vol_adj_ts))*100:.1f}%)")

    # ── 5. WF 4 Cases ────────────────────────────────────────────────
    print("\n[4] Walk-Forward 5-fold × 4 cases...")
    all_results = run_wf_all_cases(ds, get_active, vol_adj_ts)

    # ── 6. Interaction Analysis ──────────────────────────────────────
    print("\n[5] Interaction Analysis...")
    interaction_oos   = compute_interaction(all_results, "avg_oos_cagr")
    interaction_full  = compute_interaction(
        {k: {"avg_oos_cagr": v["full_is_2018_2024"]["cagr"]} for k, v in all_results.items()},
        "avg_oos_cagr",
    )
    interaction_true  = compute_interaction(
        {k: {"avg_oos_cagr": v["true_oos_2025"]["cagr"]} for k, v in all_results.items()},
        "avg_oos_cagr",
    )

    # ── 7. Summary Table ─────────────────────────────────────────────
    print("\n" + "=" * 88)
    print("  SUMMARY — WF OOS avg metrics（5-fold）")
    print("=" * 88)
    print(f"{'Case':<14} {'WF':>4} {'avgCAGR':>9} {'ΔCAGR':>8} {'Sharpe':>7} "
          f"{'MaxDD':>7} {'Calmar':>7} {'seg3_22':>8} {'Exp%':>6} {'AvgPos':>7} {'Addons':>7}")
    print("-" * 88)

    A_avg = all_results["A_BASELINE"]["avg_oos_cagr"]
    for case_name, _, _ in CASES:
        r = all_results[case_name]
        d_cagr = r["avg_oos_cagr"] - A_avg
        s3 = r["seg3_2022"] or 0.0
        print(f"{case_name:<14} {r['wf_count']:>4} {r['avg_oos_cagr']:>+9.2f}% "
              f"{d_cagr:>+8.2f}pp {r['avg_oos_sharpe']:>7.3f} "
              f"{r['avg_oos_dd']:>+7.2f}% {r['avg_oos_calmar']:>7.3f} "
              f"{s3:>+8.2f}% {r['avg_oos_exposure']:>6.1f}% "
              f"{r['avg_oos_pos']:>7.2f} {r['avg_oos_addon_count']:>7}")

    print("\n  True OOS 2025:")
    print(f"  {'Case':<14} {'CAGR':>9} {'Sharpe':>7} {'MaxDD':>7} {'Exp%':>6} {'Addons':>7}")
    print("  " + "-" * 55)
    for case_name, _, _ in CASES:
        t = all_results[case_name]["true_oos_2025"]
        print(f"  {case_name:<14} {t['cagr']:>+9.2f}% {t['sharpe']:>7.3f} "
              f"{t['max_dd']:>+7.2f}% {t['avg_exposure']:>6.1f}% {t['addon_count']:>7}")

    print("\n  Full IS 2018-2024:")
    print(f"  {'Case':<14} {'CAGR':>9} {'Sharpe':>7} {'MaxDD':>7} {'Calmar':>7} "
          f"{'Exp%':>6} {'IdleCash%':>10} {'AvgPos':>7} {'Addons':>7}")
    print("  " + "-" * 76)
    for case_name, _, _ in CASES:
        f = all_results[case_name]["full_is_2018_2024"]
        print(f"  {case_name:<14} {f['cagr']:>+9.2f}% {f['sharpe']:>7.3f} "
              f"{f['max_dd']:>+7.2f}% {f['calmar']:>7.3f} "
              f"{f['avg_exposure']:>6.1f}% {f['idle_cash_pct']:>10.1f}% "
              f"{f['avg_simultaneous_pos']:>7.2f} {f['addon_count']:>7}")

    # ── 8. Interaction Details ───────────────────────────────────────
    print("\n" + "=" * 78)
    print("  Interaction Analysis (CAGR pp)")
    print("=" * 78)
    for label, iact in [("WF OOS avg", interaction_oos),
                         ("Full IS 2018-24", interaction_full),
                         ("True OOS 2025", interaction_true)]:
        print(f"\n  [{label}]")
        print(f"    A (baseline):          {iact['A_baseline']:>+8.2f}%")
        print(f"    B (VOL_ADJ only):      {iact['B_vol_adj']:>+8.2f}%   ΔB={iact['delta_B_only']:>+.3f}pp")
        print(f"    C (EQ_SCALE only):     {iact['C_eq_scale']:>+8.2f}%   ΔC={iact['delta_C_only']:>+.3f}pp")
        print(f"    D (Combined):          {iact['D_combined']:>+8.2f}%   ΔD={iact['delta_D']:>+.3f}pp")
        print(f"    Expected additive D:   {iact['A_baseline']+iact['expected_additive']:>+8.2f}%  "
              f"(A + ΔB + ΔC = {iact['expected_additive']:>+.3f}pp)")
        print(f"    Interaction effect:    {iact['interaction_pp']:>+8.3f}pp  → {iact['classification']}")

    # ── 9. CAGR Ceiling Estimates ────────────────────────────────────
    d_wf_cagr   = all_results["D_COMBINED"]["avg_oos_cagr"]
    d_full_cagr = all_results["D_COMBINED"]["full_is_2018_2024"]["cagr"]
    d_true_cagr = all_results["D_COMBINED"]["true_oos_2025"]["cagr"]
    d_sharpe    = all_results["D_COMBINED"]["full_is_2018_2024"]["sharpe"]
    d_dd        = all_results["D_COMBINED"]["full_is_2018_2024"]["max_dd"]
    d_calmar    = all_results["D_COMBINED"]["full_is_2018_2024"]["calmar"]

    cap_3m_wf          = d_wf_cagr
    cap_20m_est_wf     = d_wf_cagr + CAPITAL_20M_UPLIFT_PP
    cap_30m_est_wf     = d_wf_cagr + CAPITAL_30M_UPLIFT_PP
    cap_3m_full        = d_full_cagr
    cap_20m_est_full   = d_full_cagr + CAPITAL_20M_UPLIFT_PP
    cap_30m_est_full   = d_full_cagr + CAPITAL_30M_UPLIFT_PP

    print("\n" + "=" * 78)
    print("  Production CAGR Ceiling Estimates")
    print("=" * 78)
    print(f"\n  D_COMBINED Full IS (2018-24):  CAGR={d_full_cagr:+.2f}%  "
          f"Sh={d_sharpe:.3f}  DD={d_dd:.2f}%  Cal={d_calmar:.3f}")
    print(f"  D_COMBINED True OOS 2025:      CAGR={d_true_cagr:+.2f}%")
    print(f"  D_COMBINED WF avg:             CAGR={d_wf_cagr:+.2f}%\n")
    print(f"  {'Capital':<12} {'WF-based est.':>16} {'Full IS-based est.':>20}")
    print(f"  {'-'*50}")
    print(f"  {'¥3M (live)':12} {cap_3m_wf:>+15.2f}% {cap_3m_full:>+19.2f}%")
    print(f"  {'¥20M':12} {cap_20m_est_wf:>+15.2f}% {cap_20m_est_full:>+19.2f}%"
          f"  (+{CAPITAL_20M_UPLIFT_PP:.2f}pp lot unlock)")
    print(f"  {'¥30M':12} {cap_30m_est_wf:>+15.2f}% {cap_30m_est_full:>+19.2f}%"
          f"  (+{CAPITAL_30M_UPLIFT_PP:.2f}pp lot unlock)")

    # ── 10. Gate Check ───────────────────────────────────────────────
    print("\n  Gate Check (D_COMBINED):")
    d_res = all_results["D_COMBINED"]
    a_res = all_results["A_BASELINE"]
    g1 = "OK" if d_res["wf_count"] >= 4 else "NG"
    g2 = "OK" if (d_res["seg3_2022"] or -99) >= (a_res["seg3_2022"] or -99) else "NG"
    g3 = "OK" if d_res["avg_oos_dd"] >= a_res["avg_oos_dd"] else "NG"
    iact_class = interaction_oos["classification"]
    print(f"    G1 WF≥4/5: {g1}  ({d_res['wf_count']}/5)")
    print(f"    G2 2022 non-degrad: {g2}  "
          f"(D={d_res['seg3_2022']:+.2f}%  A={a_res['seg3_2022']:+.2f}%)")
    print(f"    G3 MaxDD non-worsening: {g3}  "
          f"(D={d_res['avg_oos_dd']:+.2f}%  A={a_res['avg_oos_dd']:+.2f}%)")
    print(f"    Interaction: {iact_class}  ({interaction_oos['interaction_pp']:>+.3f}pp)")
    overall = "PASS" if all(g == "OK" for g in [g1, g2, g3]) else "PARTIAL" if g1 == "OK" else "FAIL"
    print(f"    Overall: {overall}")

    # ── 11. JSON 保存 ─────────────────────────────────────────────────
    output = {
        "study": "Study46",
        "title": "VolAdj × Addon Interaction Walk-Forward (2×2 Factorial)",
        "date":  str(date.today()),
        "params": {
            "capital": CAPITAL,
            "baseline": "S5 (exit_policy=NONE, no ATR Extension)",
            "vol_adj_threshold": VOL_CALM_THRESHOLD,
            "addon_params": {"policy": "D", "atr_mult": 1.0, "size_frac": 0.25, "max_per_pos": 1},
        },
        "cases": all_results,
        "interaction_analysis": {
            "wf_oos_avg":    interaction_oos,
            "full_is":       interaction_full,
            "true_oos_2025": interaction_true,
        },
        "production_ceiling": {
            "d_combined_wf_avg_cagr":    round(cap_3m_wf,    2),
            "d_combined_full_is_cagr":   round(cap_3m_full,  2),
            "d_combined_true_oos_cagr":  round(d_true_cagr,  2),
            "d_combined_full_is_sharpe": round(d_sharpe,     3),
            "d_combined_full_is_max_dd": round(d_dd,         2),
            "d_combined_full_is_calmar": round(d_calmar,     3),
            "ceiling_20m_wf_est":   round(cap_20m_est_wf,   2),
            "ceiling_30m_wf_est":   round(cap_30m_est_wf,   2),
            "ceiling_20m_full_est": round(cap_20m_est_full,  2),
            "ceiling_30m_full_est": round(cap_30m_est_full,  2),
        },
        "gate_check": {"G1": g1, "G2": g2, "G3": g3, "interaction": iact_class, "overall": overall},
    }

    out_path = Path(RESULTS_DIR) / f"study46_voladj_addon_interaction_wf_202606_{date.today()}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n[6] 結果保存: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
