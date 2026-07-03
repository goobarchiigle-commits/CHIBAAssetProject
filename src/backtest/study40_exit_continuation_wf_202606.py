"""
backtest/study40_exit_continuation_wf_202606.py

Study40 — Exit Continuation Alpha Recovery Walk-Forward (6-Policy)
目的: S5 tail_capture=68.2% → ≥70%、CAGR +1pp以上 を Exit policy のみで達成可能か検証。
禁止: Entry変更 / Universe変更 / Ranking変更 / Allocation変更 / Position sizing変更

Baseline: S5 production (sizing_mode="existing", composite shock, ATR trail, ml_rsr, rsr_exit=70)
Policies:
  A: ATR Extension     — RSR_EXITを1×ATR20 floor以上かつ利益圏なら5日延命
  B: RSR Persistence   — RSR_EXITをrsr_floor=65.0以上なら5日延命
  C: Donchian Re-Break — TURTLE_EXITを利益≥5%なら5日延命
  D: Partial+Runner    — exit signal時50%決済 + 残り2×ATR20 trailing runner
  E: Time Decay        — TIME_STOP時利益≥5%なら最大2回×15日延長
  F: Hybrid Persistence— A+B両条件を同時充足しないと延命しない（strictフィルター）

Gates:
  G1: WF_pass ≥ 4/5 (OOS Sharpe > 0)
  G2: ΔCAGR > 0 vs baseline (OOS 5-fold avg)
  G3: 2022 OOS CAGR non-degradation (seg3 oos_cagr ≥ baseline)
  G4: MaxDD non-worsening (OOS worst_dd ≥ baseline)
  G5: tail_capture ≥ 70% (OOS 全トレード)

Output: backtests/study40_exit_continuation_wf_202606_<date>.json
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

POLICIES: dict[str, dict] = {
    "BASELINE":   {},
    "A_ATR_EXT":  dict(exit_policy="A", exit_policy_atr_mult=1.0,   exit_policy_defer_days=5),
    "B_RSR_PERS": dict(exit_policy="B", exit_policy_rsr_floor=65.0, exit_policy_defer_days=5),
    "C_DONCHIAN": dict(exit_policy="C", exit_policy_profit_thr=0.05, exit_policy_defer_days=5),
    "D_PARTIAL":  dict(exit_policy="D", exit_policy_partial_frac=0.5),
    "E_TIME":     dict(exit_policy="E", exit_policy_profit_thr=0.05, exit_policy_time_extend=15),
    "F_HYBRID":   dict(exit_policy="F", exit_policy_rsr_floor=65.0,  exit_policy_atr_mult=1.0,
                       exit_policy_defer_days=5),
}

POLICY_RISK: dict[str, str] = {
    "BASELINE":   "NONE",
    "A_ATR_EXT":  "LOW",
    "B_RSR_PERS": "MEDIUM",  # study04 rsr65 で2022年 -8.8pp 前科
    "C_DONCHIAN": "LOW",
    "D_PARTIAL":  "HIGH",    # 部分決済でposition qty変動→addon残qty参照に干渉
    "E_TIME":     "LOW",
    "F_HYBRID":   "MEDIUM",  # B条件包含
}


def run_s5(ds: dict, sym_active_df: "pd.DataFrame | None",
           start: str, end: str, policy_kw: dict) -> dict:
    """S5 production config で run_scenario を呼ぶ。"""
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
        **policy_kw,
    )


def extract(res: dict) -> dict:
    return {k: res.get(k, 0.0) for k in ("cagr", "sharpe", "max_dd", "calmar", "n_trades")}


def compute_tail_capture(trades: list, universe_raw: dict,
                          all_syms: list, period_start: str, period_end: str,
                          all_common_dates: list) -> tuple[float, int]:
    """
    winning SELL tradeのtail_captureを計算（post-hoc、lookaheadなし）。
    tail_capture = actual_return / (actual_return + max(0, fwd_20d_max_return))
    exit_idx は period内の common_dates へのインデックス。
    """
    ts = pd.Timestamp(period_start)
    te = pd.Timestamp(period_end)
    # period内のcommon dates
    date_sets = [set(universe_raw[s]["df"].index) for s in all_syms if s in universe_raw]
    if not date_sets:
        return float("nan"), 0
    period_common = sorted(d for d in all_common_dates if ts <= d <= te)

    winning_sells = [t for t in trades
                     if t.get("side") == "SELL" and (t.get("pnl") or 0) > 0
                     and t.get("exit_idx") is not None]
    if not winning_sells:
        return float("nan"), 0

    tc_list: list[float] = []
    for t in winning_sells:
        sym = t["symbol"]
        if sym not in universe_raw:
            continue
        ei = t["exit_idx"]
        if ei >= len(period_common):
            continue
        exit_date = period_common[ei]
        exit_px  = t.get("exit", 0.0)
        entry_px = t.get("entry", 0.0)
        if exit_px <= 0 or entry_px <= 0:
            continue
        actual_ret = exit_px / entry_px - 1.0
        if actual_ret <= 0:
            continue

        close_s: pd.Series = universe_raw[sym]["df"]["Close"]
        post_dates = [d for d in all_common_dates if d > exit_date][:20]
        if not post_dates:
            tc_list.append(1.0)
            continue
        post_cls = close_s.reindex(post_dates).dropna()
        if post_cls.empty:
            tc_list.append(1.0)
            continue

        fwd_max = float(post_cls.max()) / exit_px - 1.0
        tc_list.append(actual_ret / (actual_ret + fwd_max) if fwd_max > 0 else 1.0)

    return (float(np.mean(tc_list)), len(tc_list)) if tc_list else (float("nan"), 0)


def main() -> int:
    print("=" * 78)
    print("  Study40 — Exit Continuation Alpha Recovery Walk-Forward")
    print(f"  Baseline: S5  capital={CAPITAL:,}  min_hold={MIN_HOLD}")
    print("=" * 78)

    # ── 1. データセット ──────────────────────────────────────────────
    print(f"\n[1] データセット構築（end={FULL_DATA_END}）...")
    ds = build_common_dataset(FULL_DATA_END)
    all_syms = list(ds["trade_syms"].keys())
    print(f"  完了 ({len(all_syms)} syms)")

    # ── 2. all_common_dates（tail_capture fwd lookahead用） ──────────
    date_sets = [set(ds["universe_raw"][s]["df"].index)
                 for s in all_syms if s in ds["universe_raw"]]
    all_common_dates: list = sorted(set.intersection(*date_sets)) if date_sets else []
    print(f"  all_common_dates: {len(all_common_dates)} days "
          f"({all_common_dates[0].date() if all_common_dates else '?'} .. "
          f"{all_common_dates[-1].date() if all_common_dates else '?'})")

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

    # ── 4. WF + True OOS 全ポリシー実行 ─────────────────────────────
    print("\n[3] Walk-Forward 実行...")
    all_results: dict[str, dict] = {}

    for policy_name, policy_kw in POLICIES.items():
        print(f"\n  ── {policy_name} ──")
        seg_results = []
        # fold毎にtail_capture用のtrade listを別管理
        fold_oos_info: list[dict] = []

        for seg in WF_SEGS:
            n = seg["seg"]
            is_s, is_e = seg["is"]
            oos_s, oos_e = seg["oos"]

            r_is  = run_s5(ds, get_active(is_e),  is_s, is_e,  policy_kw)
            r_oos = run_s5(ds, get_active(oos_e), oos_s, oos_e, policy_kw)

            is_sh  = float(r_is.get("sharpe", 0.0)  or 0.0)
            oos_sh = float(r_oos.get("sharpe", 0.0) or 0.0)
            oos_cagr   = float(r_oos.get("cagr", 0.0)   or 0.0)
            oos_dd     = float(r_oos.get("max_dd", 0.0)  or 0.0)
            oos_calmar = float(r_oos.get("calmar", 0.0) or 0.0)
            wf_pass = oos_sh > 0

            seg_results.append({
                "seg": n, "oos_year": oos_s[:4],
                "is_sharpe":   round(is_sh, 3),
                "oos_sharpe":  round(oos_sh, 3),
                "oos_cagr":    round(oos_cagr, 2),
                "oos_max_dd":  round(oos_dd, 2),
                "oos_calmar":  round(oos_calmar, 3),
                "oos_is_ratio": round(oos_sh / is_sh, 3) if is_sh != 0 else None,
                "wf_pass": wf_pass,
                "exit_reasons": r_oos.get("exit_reason_counts", {}),
                "n_trades_oos": r_oos.get("n_trades", 0),
            })
            # save OOS SELL trades per fold for tail_capture
            fold_oos_info.append({
                "oos_s": oos_s, "oos_e": oos_e,
                "trades": r_oos.get("_trades", []),  # _trades = SELL-side only
            })

            mark = "OK" if wf_pass else "NG"
            print(f"    Seg{n} OOS={oos_s[:4]}  IS_Sh={is_sh:.3f}  OOS_Sh={oos_sh:.3f}  "
                  f"OOS_CAGR={oos_cagr:+.2f}%  OOS_DD={oos_dd:+.1f}%  {mark}")

        # True OOS 2025
        r_true = run_s5(ds, get_active(TRUE_OOS[1]), TRUE_OOS[0], TRUE_OOS[1], policy_kw)
        true_m = extract(r_true)
        print(f"    True OOS 2025: CAGR={true_m['cagr']:+.2f}%  "
              f"Sharpe={true_m['sharpe']:.3f}  MaxDD={true_m['max_dd']:.2f}%")

        # Full IS 2018-2024
        r_is_full = run_s5(ds, get_active(IS_FULL[1]), IS_FULL[0], IS_FULL[1], policy_kw)
        is_full_m = extract(r_is_full)
        print(f"    Full IS 2018-2024: CAGR={is_full_m['cagr']:+.2f}%  "
              f"Sharpe={is_full_m['sharpe']:.3f}  MaxDD={is_full_m['max_dd']:.2f}%")

        # tail_capture: fold毎に計算して加重平均
        tc_vals_weighted: list[float] = []  # [tc] * n_wins で平均に同等
        for finfo in fold_oos_info:
            tc_val, tc_n = compute_tail_capture(
                finfo["trades"], ds["universe_raw"], all_syms,
                finfo["oos_s"], finfo["oos_e"], all_common_dates,
            )
            if not np.isnan(tc_val) and tc_n > 0:
                tc_vals_weighted.extend([tc_val] * tc_n)

        tail_capture = float(np.mean(tc_vals_weighted)) if tc_vals_weighted else float("nan")
        tc_n_total = len(tc_vals_weighted)
        print(f"    tail_capture OOS: {tail_capture*100:.1f}%  n={tc_n_total}")

        all_results[policy_name] = {
            "policy_params":             {k: str(v) for k, v in policy_kw.items()},
            "interaction_risk_with_addon": POLICY_RISK[policy_name],
            "wf_segments":               seg_results,
            "true_oos_2025":             true_m,
            "full_is_2018_2024":         is_full_m,
            "tail_capture_oos_pct":      round(tail_capture * 100, 1) if not np.isnan(tail_capture) else None,
            "tail_capture_n_winning":    tc_n_total,
        }

    # ── 5. Gate evaluation ──────────────────────────────────────────
    print("\n" + "=" * 78)
    print("  [4] Gate Evaluation")
    print("=" * 78)

    baseline = all_results["BASELINE"]
    b_segs = baseline["wf_segments"]
    b_seg3_cagr    = next((s["oos_cagr"] for s in b_segs if s["seg"] == 3), None)
    b_worst_dd     = min(s["oos_max_dd"] for s in b_segs)
    b_avg_cagr     = float(np.mean([s["oos_cagr"] for s in b_segs]))
    b_tc           = baseline.get("tail_capture_oos_pct") or 0.0

    print(f"\n  BASELINE — WF:{sum(s['wf_pass'] for s in b_segs)}/5  "
          f"avg_CAGR={b_avg_cagr:+.2f}%  seg3(2022)={b_seg3_cagr:+.2f}%  "
          f"worst_DD={b_worst_dd:.2f}%  TC={b_tc}%")

    gate_results: dict[str, dict] = {}
    for pn, res in all_results.items():
        if pn == "BASELINE":
            continue
        segs = res["wf_segments"]
        wf_n       = sum(s["wf_pass"] for s in segs)
        avg_cagr   = float(np.mean([s["oos_cagr"] for s in segs]))
        seg3_cagr  = next((s["oos_cagr"] for s in segs if s["seg"] == 3), None)
        worst_dd   = min(s["oos_max_dd"] for s in segs)
        tc_pct     = res.get("tail_capture_oos_pct") or 0.0
        avg_calmar = float(np.mean([s["oos_calmar"] for s in segs]))

        g1 = wf_n >= 4
        g2 = avg_cagr > b_avg_cagr
        g3 = (seg3_cagr is not None and b_seg3_cagr is not None and seg3_cagr >= b_seg3_cagr)
        g4 = worst_dd >= b_worst_dd
        g5 = tc_pct >= 70.0

        n_pass  = sum([g1, g2, g3, g4, g5])
        overall = "PASS" if all([g1, g2, g3, g4, g5]) else "FAIL"

        gate_results[pn] = {
            "wf_pass_count":            wf_n,
            "avg_oos_cagr":             round(avg_cagr, 2),
            "delta_cagr_vs_baseline":   round(avg_cagr - b_avg_cagr, 2),
            "seg3_2022_oos_cagr":       seg3_cagr,
            "delta_seg3_vs_baseline":   round((seg3_cagr or 0) - (b_seg3_cagr or 0), 2),
            "worst_oos_dd":             round(worst_dd, 2),
            "tail_capture_pct":         tc_pct,
            "delta_tail_capture_pp":    round(tc_pct - b_tc, 1),
            "avg_oos_calmar":           round(avg_calmar, 3),
            "G1_wf_pass": g1, "G2_cagr_pos": g2,
            "G3_2022_nondeg": g3, "G4_dd_nonworse": g4, "G5_tailcap_70": g5,
            "gates_passed": n_pass,
            "overall":      overall,
            "interaction_risk_with_addon": POLICY_RISK[pn],
        }

        print(f"\n  {pn} [{overall}] ({n_pass}/5 gates)")
        print(f"    G1 WF≥4/5:          {str(g1):<5}  ({wf_n}/5)")
        print(f"    G2 ΔCAGR>0:         {str(g2):<5}  ({avg_cagr - b_avg_cagr:+.2f}pp)")
        print(f"    G3 2022 non-deg:    {str(g3):<5}  (seg3={seg3_cagr:+.2f}% vs {b_seg3_cagr:+.2f}%)")
        print(f"    G4 DD non-worse:    {str(g4):<5}  (worst={worst_dd:.2f}% vs {b_worst_dd:.2f}%)")
        print(f"    G5 TC≥70%:          {str(g5):<5}  ({tc_pct:.1f}% Δ={tc_pct - b_tc:+.1f}pp)")
        print(f"    Risk(addon):        {POLICY_RISK[pn]}")

    # ── 6. ランキング ────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("  [5] Ranking (OOS CAGR descending, then tail_capture)")
    print("=" * 78)

    ranking = []
    for pn, gr in gate_results.items():
        ranking.append({
            "policy":           pn,
            "overall":          gr["overall"],
            "gates_passed":     gr["gates_passed"],
            "avg_oos_cagr":     gr["avg_oos_cagr"],
            "delta_cagr":       gr["delta_cagr_vs_baseline"],
            "avg_oos_calmar":   gr["avg_oos_calmar"],
            "tail_capture_pct": gr["tail_capture_pct"],
            "delta_tc_pp":      gr["delta_tail_capture_pp"],
            "interaction_risk": gr["interaction_risk_with_addon"],
        })
    ranking.sort(key=lambda x: (-x["gates_passed"], -(x["avg_oos_cagr"] or -99)))

    print(f"\n  {'Policy':<16} {'Overall':<6} {'Gates':<6} {'ΔCAGR':>7} {'TC%':>6} {'ΔTC':>5}  Risk")
    print("  " + "-" * 62)
    for r in ranking:
        print(f"  {r['policy']:<16} {r['overall']:<6} {r['gates_passed']}/5  "
              f"  {r['delta_cagr']:>+5.2f}pp  {r['tail_capture_pct']:>5.1f}%  "
              f"{r['delta_tc_pp']:>+4.1f}pp  {r['interaction_risk']}")

    # ── 7. Decision ─────────────────────────────────────────────────
    pass_list = [r for r in ranking if r["overall"] == "PASS"]
    if pass_list:
        decision = "PROMOTE_TO_CANDIDATE"
        best = pass_list[0]
        decision_detail = (f"Best: {best['policy']}  "
                           f"ΔCAGR={best['delta_cagr']:+.2f}pp  "
                           f"TC={best['tail_capture_pct']:.1f}%")
    else:
        decision = "EXHAUSTED"
        top = max(ranking, key=lambda x: (x["gates_passed"], x["avg_oos_cagr"]))
        decision_detail = (f"全Policy FAIL。最高 {top['policy']} "
                           f"({top['gates_passed']}/5 gates, "
                           f"ΔCAGR={top['delta_cagr']:+.2f}pp)")

    print(f"\n  Decision: {decision}")
    print(f"  {decision_detail}")

    # ── 8. JSON保存 ─────────────────────────────────────────────────
    output = {
        "study":      "Study40_Exit_Continuation_WF",
        "date":       date.today().isoformat(),
        "capital":    CAPITAL,
        "min_hold":   MIN_HOLD,
        "baseline_summary": {
            "wf_pass_count": sum(s["wf_pass"] for s in b_segs),
            "avg_oos_cagr":  round(b_avg_cagr, 2),
            "seg3_2022_oos_cagr": b_seg3_cagr,
            "worst_oos_dd":  round(b_worst_dd, 2),
            "tail_capture_oos_pct": b_tc,
            "full_is_2018_2024": baseline["full_is_2018_2024"],
            "true_oos_2025":    baseline["true_oos_2025"],
        },
        "policy_results": {k: {kk: vv for kk, vv in v.items() if kk != "_trades"}
                           for k, v in all_results.items()},
        "gate_results": gate_results,
        "ranking":      ranking,
        "decision":     decision,
        "decision_detail": decision_detail,
    }

    save_path = RESULTS_DIR / f"study40_exit_continuation_wf_202606_{date.today().isoformat()}.json"
    save_path.write_text(
        json.dumps(output, ensure_ascii=False, default=str, indent=2), encoding="utf-8"
    )
    print(f"\n結果保存: {save_path}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
