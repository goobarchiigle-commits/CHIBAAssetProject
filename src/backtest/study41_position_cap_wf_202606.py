"""
backtest/study41_position_cap_wf_202606.py

Study41 — Position Cap Equity-Linked Walk-Forward
目的: max_positions=3（固定）が複利成長を抑制しているか定量化。
      資本稼働率・見逃しエントリー・CAGR弾力性を測定。

Cases:
  A: Control           — max_positions=3 (production baseline)
  B: Equity-Linked     — max_positions: 3→4→5 based on equity growth (10%/step)
  C_3..C_7: Stepwise   — max_positions=3,4,5,6,7 (elasticity curve)
  D: Vol-Adjusted      — max_positions: 3 normal / 4 low-TOPIX-vol (<0.8% 20d std)

Gates（全Case共通）:
  G1: WF ≥ 4/5 (OOS Sharpe > 0)
  G2: ΔCAGR > 0 vs baseline (OOS 5-fold avg)
  G3: 2022 OOS CAGR non-degradation (seg3 ≥ baseline)
  G4: MaxDD non-worsening (OOS worst_dd ≥ baseline)

Output: backtests/study41_position_cap_wf_202606_<date>.json
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

# Equity-linked: 10%成長ごとに1スロット追加、上限5
EQ_LINKED_STEP = 0.10
EQ_LINKED_MAX  = 5

# Vol-adjusted: TOPIX 20d daily return std の閾値
VOL_CALM_THRESHOLD = 0.008   # 20d std < 0.8% → max_positions=4 (calm market)

# Stepwise caps
STEPWISE_CAPS = [3, 4, 5, 6, 7]


def build_vol_adjusted_ts(topix_close: pd.Series, common_dates: list,
                           base_max: int = 3, calm_max: int = 4,
                           vol_threshold: float = VOL_CALM_THRESHOLD) -> pd.Series:
    """TOPIX 20d vol ベースの動的 max_positions 系列を生成（lookahead なし）。"""
    topix_ret = topix_close.pct_change()
    rolling_std = topix_ret.rolling(20, min_periods=10).std()
    # common_dates に reindex（ffill で穴埋め）
    std_series = rolling_std.reindex(pd.Index(common_dates), method="ffill").fillna(rolling_std.median())
    # calm: std < threshold → max=calm_max、それ以外 → max=base_max
    max_pos_ts = pd.Series(base_max, index=std_series.index, dtype=int)
    max_pos_ts[std_series < vol_threshold] = calm_max
    return max_pos_ts


def run_s5(ds: dict, sym_active_df: "pd.DataFrame | None",
           start: str, end: str,
           max_positions_override: "int | None" = None,
           capital_mode: str = "fixed",
           equity_linked_step: float = EQ_LINKED_STEP,
           equity_linked_max: int = EQ_LINKED_MAX,
           max_positions_ts: "pd.Series | None" = None) -> dict:
    """S5 production config で run_scenario を呼ぶ（Study41 cap variants対応）。"""
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
        # Study41 cap params
        max_positions_override=max_positions_override,
        max_positions_ts=max_positions_ts,
        capital_mode=capital_mode,
        equity_linked_step=equity_linked_step,
        equity_linked_max=equity_linked_max,
    )


def extract(res: dict) -> dict:
    return {k: res.get(k, 0.0) for k in ("cagr", "sharpe", "max_dd", "calmar",
                                           "n_trades", "sortino", "profit_factor")}


def extract_util(res: dict) -> dict:
    """Capital utilization メトリクスを抽出。"""
    return {
        "avg_idle_cash_ratio_pct":  res.get("avg_idle_cash_ratio_pct", None),
        "days_at_max_positions":    res.get("days_at_max_positions", None),
        "cap_saturation_rate_pct":  res.get("cap_saturation_rate_pct", None),
        "missed_by_cap_count":      res.get("missed_by_cap_count", None),
        "avg_exposure":             res.get("avg_exposure", None),
        "avg_simultaneous_holdings": res.get("avg_simultaneous_holdings", None),
    }


def compute_missed_fwd_returns(missed_cands: list[dict],
                                universe_raw: dict,
                                all_common_dates: list,
                                fwd_days: int = 20) -> dict:
    """
    見逃しエントリーの fwd_return を後処理計算。
    missed_cands: [{"date": str, "symbol": str, "rsr": float}]
    戻り値: {"avg_fwd_ret": float, "n": int, "pct_positive": float}
    """
    date_idx: dict = {str(d.date()) if hasattr(d, "date") else str(d): i
                      for i, d in enumerate(all_common_dates)}
    fwd_list: list[float] = []
    for mc in missed_cands:
        sym  = mc["symbol"]
        dstr = mc["date"]
        if sym not in universe_raw or dstr not in date_idx:
            continue
        idx = date_idx[dstr]
        close_s: pd.Series = universe_raw[sym]["df"]["Close"]
        # entry price = close on missed date (proxy)
        if idx >= len(all_common_dates):
            continue
        entry_date = all_common_dates[idx]
        if entry_date not in close_s.index:
            continue
        entry_px = float(close_s[entry_date])
        if entry_px <= 0:
            continue
        # forward close
        fwd_dates = all_common_dates[idx + 1: idx + 1 + fwd_days]
        fwd_cls = close_s.reindex(fwd_dates).dropna()
        if fwd_cls.empty:
            continue
        fwd_ret = float(fwd_cls.iloc[-1]) / entry_px - 1.0
        fwd_list.append(fwd_ret)
    if not fwd_list:
        return {"avg_fwd_ret_pct": None, "n": 0, "pct_positive": None}
    return {
        "avg_fwd_ret_pct": round(float(np.mean(fwd_list)) * 100, 2),
        "n": len(fwd_list),
        "pct_positive": round(sum(1 for r in fwd_list if r > 0) / len(fwd_list) * 100, 1),
    }


def run_wf_case(case_name: str, ds: dict, get_active,
                common_dates: list,
                max_positions_override: "int | None" = None,
                capital_mode: str = "fixed",
                max_positions_ts: "pd.Series | None" = None) -> dict:
    """1 Case の 5-fold WF + True OOS + Full IS を実行して結果辞書を返す。"""
    print(f"\n  ── {case_name} ──")
    seg_results = []
    missed_cands_all: list[dict] = []

    for seg in WF_SEGS:
        n = seg["seg"]
        is_s, is_e = seg["is"]
        oos_s, oos_e = seg["oos"]

        # IS run (WF gate用)
        r_is = run_s5(ds, get_active(is_e), is_s, is_e,
                      max_positions_override=max_positions_override,
                      capital_mode=capital_mode, max_positions_ts=max_positions_ts)
        # OOS run
        r_oos = run_s5(ds, get_active(oos_e), oos_s, oos_e,
                       max_positions_override=max_positions_override,
                       capital_mode=capital_mode, max_positions_ts=max_positions_ts)

        is_sh  = float(r_is.get("sharpe", 0.0) or 0.0)
        oos_sh = float(r_oos.get("sharpe", 0.0) or 0.0)
        oos_cagr   = float(r_oos.get("cagr",   0.0) or 0.0)
        oos_dd     = float(r_oos.get("max_dd",  0.0) or 0.0)
        oos_calmar = float(r_oos.get("calmar",  0.0) or 0.0)
        wf_pass = oos_sh > 0

        # collect missed cands from OOS
        missed_cands_all.extend(r_oos.get("_missed_cands", [])[:500])

        seg_results.append({
            "seg": n, "oos_year": oos_s[:4],
            "is_sharpe":   round(is_sh, 3),
            "oos_sharpe":  round(oos_sh, 3),
            "oos_cagr":    round(oos_cagr, 2),
            "oos_max_dd":  round(oos_dd, 2),
            "oos_calmar":  round(oos_calmar, 3),
            "oos_is_ratio": round(oos_sh / is_sh, 3) if is_sh != 0 else None,
            "wf_pass":     wf_pass,
            "cap_util":    extract_util(r_oos),
        })
        mark = "OK" if wf_pass else "NG"
        print(f"    Seg{n} OOS={oos_s[:4]}  IS_Sh={is_sh:.3f}  OOS_Sh={oos_sh:.3f}  "
              f"CAGR={oos_cagr:+.2f}%  DD={oos_dd:+.1f}%  "
              f"Idle={r_oos.get('avg_idle_cash_ratio_pct','?')}%  "
              f"Missed={r_oos.get('missed_by_cap_count','?')}  {mark}")

    # True OOS 2025
    r_true = run_s5(ds, get_active(TRUE_OOS[1]), TRUE_OOS[0], TRUE_OOS[1],
                    max_positions_override=max_positions_override,
                    capital_mode=capital_mode, max_positions_ts=max_positions_ts)
    true_m = extract(r_true)
    true_util = extract_util(r_true)
    print(f"    True OOS 2025: CAGR={true_m['cagr']:+.2f}%  Sh={true_m['sharpe']:.3f}  "
          f"DD={true_m['max_dd']:.2f}%  Idle={true_util['avg_idle_cash_ratio_pct']}%")

    # Full IS 2018-2024
    r_is_full = run_s5(ds, get_active(IS_FULL[1]), IS_FULL[0], IS_FULL[1],
                       max_positions_override=max_positions_override,
                       capital_mode=capital_mode, max_positions_ts=max_positions_ts)
    is_full_m = extract(r_is_full)
    is_full_util = extract_util(r_is_full)
    print(f"    Full IS 2018-2024: CAGR={is_full_m['cagr']:+.2f}%  Sh={is_full_m['sharpe']:.3f}  "
          f"DD={is_full_m['max_dd']:.2f}%  Idle={is_full_util['avg_idle_cash_ratio_pct']}%")

    # Missed entry forward returns (OOS missed cands pool)
    all_syms = list(ds["trade_syms"].keys())
    date_sets = [set(ds["universe_raw"][s]["df"].index)
                 for s in all_syms if s in ds["universe_raw"]]
    fwd20 = compute_missed_fwd_returns(missed_cands_all, ds["universe_raw"],
                                       common_dates, fwd_days=20)
    fwd60 = compute_missed_fwd_returns(missed_cands_all, ds["universe_raw"],
                                       common_dates, fwd_days=60)
    print(f"    Missed fwd20d: {fwd20['avg_fwd_ret_pct']}%  n={fwd20['n']}  "
          f"  fwd60d: {fwd60['avg_fwd_ret_pct']}%  n={fwd60['n']}")

    return {
        "wf_segments":       seg_results,
        "true_oos_2025":     true_m,
        "true_oos_util":     true_util,
        "full_is_2018_2024": is_full_m,
        "full_is_util":      is_full_util,
        "missed_fwd20d":     fwd20,
        "missed_fwd60d":     fwd60,
        "missed_cands_total": len(missed_cands_all),
    }


def main() -> int:
    print("=" * 78)
    print("  Study41 — Position Cap Equity-Linked Walk-Forward")
    print(f"  Baseline: S5  capital={CAPITAL:,}  min_hold={MIN_HOLD}")
    print("=" * 78)

    # ── 1. データセット ──────────────────────────────────────────────
    print(f"\n[1] データセット構築（end={FULL_DATA_END}）...")
    ds = build_common_dataset(FULL_DATA_END)
    all_syms = list(ds["trade_syms"].keys())
    print(f"  完了 ({len(all_syms)} syms)")

    # ── 2. all_common_dates ──────────────────────────────────────────
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
        needed.add(seg["is"][1]); needed.add(seg["oos"][1])
    needed.add(TRUE_OOS[1]); needed.add(IS_FULL[1])
    for ed in sorted(needed):
        print(f"  end={ed}..."); get_active(ed)
    print("  完了")

    # ── 4. Vol-Adjusted max_positions series ─────────────────────────
    print("\n[3] Vol-Adjusted max_positions series 構築...")
    vol_adj_ts = build_vol_adjusted_ts(
        ds["topix_close"], all_common_dates,
        base_max=3, calm_max=4, vol_threshold=VOL_CALM_THRESHOLD,
    )
    calm_days = int((vol_adj_ts == 4).sum())
    total_days = len(vol_adj_ts)
    print(f"  calm days (max=4): {calm_days}/{total_days} ({calm_days/max(1,total_days)*100:.1f}%)")

    # ── 5. 全ケース実行 ──────────────────────────────────────────────
    print("\n[4] Walk-Forward 実行...")
    all_results: dict[str, dict] = {}

    # Case A: Control (baseline)
    all_results["A_CONTROL"] = run_wf_case(
        "A_CONTROL (max=3)", ds, get_active, all_common_dates,
        max_positions_override=3,
    )

    # Case B: Equity-Linked
    all_results["B_EQUITY_LINKED"] = run_wf_case(
        f"B_EQUITY_LINKED (base=3, step={EQ_LINKED_STEP*100:.0f}%, max={EQ_LINKED_MAX})",
        ds, get_active, all_common_dates,
        capital_mode="equity_linked",
        max_positions_override=None,
    )

    # Case C: Stepwise expansion
    for cap in STEPWISE_CAPS:
        key = f"C_{cap}"
        all_results[key] = run_wf_case(
            f"C_{cap} (max={cap})", ds, get_active, all_common_dates,
            max_positions_override=cap,
        )

    # Case D: Vol-Adjusted
    all_results["D_VOL_ADJ"] = run_wf_case(
        f"D_VOL_ADJ (base=3, calm=4, thr={VOL_CALM_THRESHOLD:.3f})",
        ds, get_active, all_common_dates,
        max_positions_ts=vol_adj_ts,
    )

    # ── 6. Gate Evaluation ──────────────────────────────────────────
    print("\n" + "=" * 78)
    print("  [5] Gate Evaluation")
    print("=" * 78)

    baseline = all_results["A_CONTROL"]
    b_segs = baseline["wf_segments"]
    b_seg3_cagr = next((s["oos_cagr"] for s in b_segs if s["seg"] == 3), None)
    b_worst_dd  = min(s["oos_max_dd"] for s in b_segs)
    b_avg_cagr  = float(np.mean([s["oos_cagr"] for s in b_segs]))

    print(f"\n  A_CONTROL — WF:{sum(s['wf_pass'] for s in b_segs)}/5  "
          f"avg_CAGR={b_avg_cagr:+.2f}%  seg3={b_seg3_cagr:+.2f}%  "
          f"worst_DD={b_worst_dd:.2f}%")

    gate_results: dict[str, dict] = {}
    for cname, res in all_results.items():
        if cname == "A_CONTROL":
            continue
        segs = res["wf_segments"]
        wf_n      = sum(s["wf_pass"] for s in segs)
        avg_cagr  = float(np.mean([s["oos_cagr"] for s in segs]))
        seg3_cagr = next((s["oos_cagr"] for s in segs if s["seg"] == 3), None)
        worst_dd  = min(s["oos_max_dd"] for s in segs)

        g1 = wf_n >= 4
        g2 = avg_cagr > b_avg_cagr
        g3 = (seg3_cagr is not None and b_seg3_cagr is not None
              and seg3_cagr >= b_seg3_cagr)
        g4 = worst_dd >= b_worst_dd

        n_pass  = sum([g1, g2, g3, g4])
        overall = "PASS" if all([g1, g2, g3, g4]) else "FAIL"

        gate_results[cname] = {
            "wf_pass_count":            wf_n,
            "avg_oos_cagr":             round(avg_cagr, 2),
            "delta_cagr_vs_baseline":   round(avg_cagr - b_avg_cagr, 2),
            "seg3_2022_oos_cagr":       seg3_cagr,
            "worst_oos_dd":             round(worst_dd, 2),
            "delta_dd_vs_baseline":     round(worst_dd - b_worst_dd, 2),
            "G1_wf_pass": g1, "G2_cagr_pos": g2,
            "G3_2022_nondeg": g3, "G4_dd_nonworse": g4,
            "gates_passed": n_pass,
            "overall":      overall,
        }

        print(f"\n  {cname} [{overall}] ({n_pass}/4 gates)")
        print(f"    G1 WF≥4/5:       {str(g1):<5}  ({wf_n}/5)")
        print(f"    G2 ΔCAGR>0:      {str(g2):<5}  ({avg_cagr - b_avg_cagr:+.2f}pp)")
        print(f"    G3 2022 non-deg: {str(g3):<5}  ({seg3_cagr:+.2f}% vs {b_seg3_cagr:+.2f}%)")
        print(f"    G4 DD non-worse: {str(g4):<5}  ({worst_dd:.2f}% vs {b_worst_dd:.2f}%)")

    # ── 7. Elasticity Table (Case C) ─────────────────────────────────
    print("\n" + "=" * 78)
    print("  [6] Elasticity Curve (Case C Stepwise)")
    print("=" * 78)
    print(f"\n  {'Cap':>4}  {'WF':>4}  {'CAGR':>7}  {'ΔCAGR':>7}  "
          f"{'Sharpe':>7}  {'MaxDD':>7}  {'Idle%':>6}  {'Missed':>7}")
    print("  " + "-" * 64)
    for cap in STEPWISE_CAPS:
        key = f"C_{cap}"
        r = all_results[key]
        segs = r["wf_segments"]
        wf_n = sum(s["wf_pass"] for s in segs)
        avg_cagr = float(np.mean([s["oos_cagr"] for s in segs]))
        avg_sh   = float(np.mean([s["oos_sharpe"] for s in segs]))
        w_dd     = min(s["oos_max_dd"] for s in segs)
        # aggregate capital util from segments
        idle_vals = [s["cap_util"].get("avg_idle_cash_ratio_pct")
                     for s in segs if s["cap_util"].get("avg_idle_cash_ratio_pct") is not None]
        miss_vals = [s["cap_util"].get("missed_by_cap_count", 0) or 0 for s in segs]
        idle_avg = round(float(np.mean(idle_vals)), 1) if idle_vals else None
        miss_total = sum(miss_vals)
        dcagr = round(avg_cagr - b_avg_cagr, 2)
        print(f"  {cap:>4}  {wf_n}/5  {avg_cagr:>+7.2f}%  {dcagr:>+6.2f}pp  "
              f"{avg_sh:>7.3f}  {w_dd:>+7.2f}%  {str(idle_avg)+'%':>6}  {miss_total:>7}")

    # ── 8. Capital Utilization Summary ───────────────────────────────
    print("\n" + "=" * 78)
    print("  [7] Capital Utilization Summary")
    print("=" * 78)
    print(f"\n  {'Case':<20}  {'Idle%':>6}  {'Saturation%':>12}  {'MissedTotal':>12}")
    print("  " + "-" * 56)
    for cname, res in all_results.items():
        segs = res["wf_segments"]
        idle_vals = [s["cap_util"].get("avg_idle_cash_ratio_pct")
                     for s in segs if s["cap_util"].get("avg_idle_cash_ratio_pct") is not None]
        sat_vals  = [s["cap_util"].get("cap_saturation_rate_pct")
                     for s in segs if s["cap_util"].get("cap_saturation_rate_pct") is not None]
        miss_vals = [s["cap_util"].get("missed_by_cap_count", 0) or 0 for s in segs]
        idle_avg = round(float(np.mean(idle_vals)), 1) if idle_vals else "?"
        sat_avg  = round(float(np.mean(sat_vals)), 1) if sat_vals else "?"
        miss_total = sum(miss_vals)
        print(f"  {cname:<20}  {str(idle_avg)+'%':>6}  {str(sat_avg)+'%':>12}  {miss_total:>12}")

    # Missed entry forward return table
    print(f"\n  {'Case':<20}  {'Fwd20d%':>8}  {'Fwd60d%':>8}  {'n':>5}  {'Pct+':>6}")
    print("  " + "-" * 54)
    for cname in ["A_CONTROL", "B_EQUITY_LINKED", "C_3", "C_4", "C_5", "D_VOL_ADJ"]:
        if cname not in all_results:
            continue
        r = all_results[cname]
        f20 = r["missed_fwd20d"]
        f60 = r["missed_fwd60d"]
        fwd20 = f20.get("avg_fwd_ret_pct", "N/A")
        fwd60 = f60.get("avg_fwd_ret_pct", "N/A")
        n20   = f20.get("n", 0)
        pp    = f20.get("pct_positive", "N/A")
        print(f"  {cname:<20}  {str(fwd20):>8}  {str(fwd60):>8}  {n20:>5}  {str(pp):>6}")

    # ── 9. Decision ─────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("  [8] Final Decision")
    print("=" * 78)

    # Best gate-pass case (excluding A_CONTROL which is baseline)
    pass_cases = [(cn, gr) for cn, gr in gate_results.items() if gr["overall"] == "PASS"]
    if pass_cases:
        best_cn, best_gr = max(pass_cases, key=lambda x: x[1]["avg_oos_cagr"])
        decision = "MEANINGFUL_BOTTLENECK" if best_gr["delta_cagr_vs_baseline"] >= 1.0 else \
                   "MINOR_BOTTLENECK"      if best_gr["delta_cagr_vs_baseline"] >= 0.3 else \
                   "NOT_BOTTLENECK"
    else:
        best_cn, best_gr = None, None
        decision = "NOT_BOTTLENECK"

    # Elasticity: CAGR per additional slot (C_3 → C_4 → C_5)
    c3_cagr = float(np.mean([s["oos_cagr"] for s in all_results["C_3"]["wf_segments"]]))
    c4_cagr = float(np.mean([s["oos_cagr"] for s in all_results["C_4"]["wf_segments"]]))
    c5_cagr = float(np.mean([s["oos_cagr"] for s in all_results["C_5"]["wf_segments"]]))
    c6_cagr = float(np.mean([s["oos_cagr"] for s in all_results["C_6"]["wf_segments"]]))
    c7_cagr = float(np.mean([s["oos_cagr"] for s in all_results["C_7"]["wf_segments"]]))

    elasticity_pp_per_slot = {
        "3→4": round(c4_cagr - c3_cagr, 2),
        "4→5": round(c5_cagr - c4_cagr, 2),
        "5→6": round(c6_cagr - c5_cagr, 2),
        "6→7": round(c7_cagr - c6_cagr, 2),
    }
    # Diminishing return point: first slot where Δ drops below 0.5pp
    dimret_point = None
    for slot, delta in elasticity_pp_per_slot.items():
        if delta < 0.5:
            dimret_point = slot
            break

    # CAGR ceiling estimates
    # Cash-only ceiling: best C case CAGR (full IS)
    best_c_cagr = max(
        float(all_results[f"C_{cap}"]["full_is_2018_2024"]["cagr"]) for cap in STEPWISE_CAPS
    )
    s5_is_cagr = float(all_results["A_CONTROL"]["full_is_2018_2024"]["cagr"])
    addon_estimate = 1.5   # placeholder from Study39 estimate
    cash_ceiling_low  = s5_is_cagr + (best_c_cagr - s5_is_cagr) + 0.30 + addon_estimate
    cash_ceiling_high = cash_ceiling_low + 1.0  # margin

    print(f"\n  Decision: {decision}")
    if best_cn:
        print(f"  Best PASS case: {best_cn}  ΔCAGR={best_gr['delta_cagr_vs_baseline']:+.2f}pp")
    print(f"\n  Elasticity (CAGR Δ per slot):")
    for slot, delta in elasticity_pp_per_slot.items():
        print(f"    {slot}: {delta:+.2f}pp")
    if dimret_point:
        print(f"  Diminishing return point: {dimret_point}")
    print(f"\n  CAGR Ceiling Estimates:")
    print(f"    S5 baseline (IS):    {s5_is_cagr:+.2f}%")
    print(f"    Best C (IS):         {best_c_cagr:+.2f}%")
    print(f"    Cash-only est:       {cash_ceiling_low:.1f} – {cash_ceiling_high:.1f}%")
    print(f"    + Leverage 1.3x:     {cash_ceiling_low*1.3:.1f} – {cash_ceiling_high*1.3:.1f}%")

    # ── 10. JSON保存 ─────────────────────────────────────────────────
    output = {
        "study":    "Study41_Position_Cap_Equity_Linked_WF",
        "date":     date.today().isoformat(),
        "capital":  CAPITAL,
        "min_hold": MIN_HOLD,
        "params": {
            "eq_linked_step":    EQ_LINKED_STEP,
            "eq_linked_max":     EQ_LINKED_MAX,
            "vol_calm_threshold": VOL_CALM_THRESHOLD,
            "vol_calm_days_pct": round(calm_days / max(1, total_days) * 100, 1),
        },
        "baseline_summary": {
            "wf_pass_count": sum(s["wf_pass"] for s in b_segs),
            "avg_oos_cagr":  round(b_avg_cagr, 2),
            "seg3_2022_oos_cagr": b_seg3_cagr,
            "worst_oos_dd":  round(b_worst_dd, 2),
            "full_is_2018_2024": all_results["A_CONTROL"]["full_is_2018_2024"],
            "true_oos_2025":    all_results["A_CONTROL"]["true_oos_2025"],
        },
        "case_results":     {
            cn: {k: v for k, v in r.items() if not k.startswith("_")}
            for cn, r in all_results.items()
        },
        "gate_results":     gate_results,
        "elasticity_pp_per_slot": elasticity_pp_per_slot,
        "diminishing_return_point": dimret_point,
        "cagr_ceiling": {
            "s5_is_cagr_pct":     round(s5_is_cagr, 2),
            "best_cap_is_cagr_pct": round(best_c_cagr, 2),
            "cash_only_low_pct":  round(cash_ceiling_low, 1),
            "cash_only_high_pct": round(cash_ceiling_high, 1),
            "leveraged_1_3x_low_pct":  round(cash_ceiling_low * 1.3, 1),
            "leveraged_1_3x_high_pct": round(cash_ceiling_high * 1.3, 1),
        },
        "decision": decision,
        "decision_detail": (f"Best: {best_cn}  ΔCAGR={best_gr['delta_cagr_vs_baseline']:+.2f}pp"
                            if best_cn else "全Case FAIL — bottleneck非確定"),
    }

    # equity_curve等の大きなSeriesはJSON化しない
    save_path = RESULTS_DIR / f"study41_position_cap_wf_202606_{date.today().isoformat()}.json"
    save_path.write_text(
        json.dumps(output, ensure_ascii=False, default=str, indent=2), encoding="utf-8"
    )
    print(f"\n結果保存: {save_path}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
