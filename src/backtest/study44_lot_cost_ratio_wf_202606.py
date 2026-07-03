"""
backtest/study44_lot_cost_ratio_wf_202606.py

Study44 — Lot Cost Expansion Walk-Forward
目的: Study42 で確認された lot 拒否アルファ（n=20, fwd20d=+5.59%, WR=80%）を
     max_lot_cost_ratio の緩和により実運用 CAGR 改善に転換可能か検証。

Cases:
  A: Baseline          — max_lot_cost_ratio=None（現行動作: qty=0→skip）
  B: ratio=0.30        — lot_cost ≤ capital×0.30 かつ cash≥lot_cost で 1ロット強制入場
  C: ratio=0.35
  D: ratio=0.40
  E: ratio=0.45

Gates（全Case共通）:
  G1: WF ≥ 4/5 (OOS Sharpe > 0)
  G2: ΔCAGR > +0.5pp vs Baseline (OOS 5-fold avg)
  G3: 2022 OOS CAGR non-degradation (seg3 ≥ baseline)
  G4: MaxDD non-worsening (OOS worst_dd ≥ baseline)

Interaction estimate: Study41 D_VOL_ADJ × Study44 best_case の組み合わせ効果を
Full IS で定量化（実装不要、推定のみ）。

Output: backtests/study44_lot_cost_ratio_wf_202606_<date>.json
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
FWD_DAYS = 20

# テストマトリクス
RATIO_CASES = [
    ("A_BASELINE", None),
    ("B_030",      0.30),
    ("C_035",      0.35),
    ("D_040",      0.40),
    ("E_045",      0.45),
]

# Study41 D_VOL_ADJ パラメータ（interaction estimate 用）
VOL_CALM_THRESHOLD = 0.008


def build_vol_adjusted_ts(topix_close: pd.Series, common_dates: list,
                           base_max: int = 3, calm_max: int = 4,
                           vol_threshold: float = VOL_CALM_THRESHOLD) -> pd.Series:
    topix_ret = topix_close.pct_change()
    rolling_std = topix_ret.rolling(20, min_periods=10).std()
    std_series = rolling_std.reindex(pd.Index(common_dates), method="ffill").fillna(rolling_std.median())
    max_pos_ts = pd.Series(base_max, index=std_series.index, dtype=int)
    max_pos_ts[std_series < vol_threshold] = calm_max
    return max_pos_ts


def run_s5(ds: dict, sym_active_df: "pd.DataFrame | None",
           start: str, end: str,
           max_lot_cost_ratio: "float | None" = None,
           max_positions_ts: "pd.Series | None" = None) -> dict:
    """S5 production config + Study44 lot cost ratio で run_scenario を呼ぶ。"""
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
        max_lot_cost_ratio=max_lot_cost_ratio,
        max_positions_ts=max_positions_ts,
    )


def extract(res: dict) -> dict:
    return {k: res.get(k, 0.0) for k in ("cagr", "sharpe", "max_dd", "calmar",
                                           "n_trades", "sortino", "profit_factor")}


def compute_admitted_fwd_returns(admitted_detail: list[dict],
                                  universe_raw: dict,
                                  all_common_dates: list,
                                  fwd_days: int = FWD_DAYS) -> dict:
    """Study44 ratio 救済入場の fwd_return を後処理計算。"""
    date_idx: dict = {str(d.date()) if hasattr(d, "date") else str(d): i
                      for i, d in enumerate(all_common_dates)}
    fwd_list: list[float] = []
    for rec in admitted_detail:
        sym  = rec["symbol"]
        dstr = rec["date"]
        if sym not in universe_raw or dstr not in date_idx:
            continue
        idx = date_idx[dstr]
        close_s: pd.Series = universe_raw[sym]["df"]["Close"]
        entry_date = all_common_dates[idx]
        if entry_date not in close_s.index:
            continue
        entry_px = float(close_s[entry_date])
        if entry_px <= 0:
            continue
        fwd_dates = all_common_dates[idx + 1: idx + 1 + fwd_days]
        fwd_cls = close_s.reindex(fwd_dates).dropna()
        if fwd_cls.empty:
            continue
        fwd_list.append(float(fwd_cls.iloc[-1]) / entry_px - 1.0)
    if not fwd_list:
        return {"avg_fwd_ret_pct": None, "n": 0, "pct_positive": None}
    return {
        "avg_fwd_ret_pct": round(float(np.mean(fwd_list)) * 100, 2),
        "n": len(fwd_list),
        "pct_positive": round(sum(1 for r in fwd_list if r > 0) / len(fwd_list) * 100, 1),
    }


def run_wf_case(case_name: str, ratio: "float | None", ds: dict, get_active,
                common_dates: list,
                max_positions_ts: "pd.Series | None" = None) -> dict:
    """1 Case の 5-fold WF + True OOS + Full IS を実行して結果辞書を返す。"""
    print(f"\n  ── {case_name}  ratio={ratio} ──")
    seg_results = []
    admitted_detail_all: list[dict] = []
    rejected_counts_oos = []

    for seg in WF_SEGS:
        n = seg["seg"]
        is_s, is_e = seg["is"]
        oos_s, oos_e = seg["oos"]

        # IS run（WF gate 用）
        r_is = run_s5(ds, get_active(is_e), is_s, is_e,
                      max_lot_cost_ratio=ratio, max_positions_ts=max_positions_ts)
        # OOS run
        r_oos = run_s5(ds, get_active(oos_e), oos_s, oos_e,
                       max_lot_cost_ratio=ratio, max_positions_ts=max_positions_ts)

        is_sh  = float(r_is.get("sharpe", 0.0) or 0.0)
        oos_sh = float(r_oos.get("sharpe", 0.0) or 0.0)
        oos_cagr   = float(r_oos.get("cagr",   0.0) or 0.0)
        oos_dd     = float(r_oos.get("max_dd",  0.0) or 0.0)
        oos_calmar = float(r_oos.get("calmar",  0.0) or 0.0)
        oos_trades = int(r_oos.get("n_trades", 0) or 0)
        oos_lot_rej = int(r_oos.get("rejected_by_lot_count", 0) or 0)
        oos_ratio_adm = int(r_oos.get("admitted_by_ratio_count", 0) or 0)
        wf_pass = oos_sh > 0

        admitted_detail_all.extend(r_oos.get("_admitted_by_ratio_detail", [])[:500])
        rejected_counts_oos.append(oos_lot_rej)

        seg_results.append({
            "seg": n, "oos_year": oos_s[:4],
            "is_sharpe":     round(is_sh, 3),
            "oos_sharpe":    round(oos_sh, 3),
            "oos_cagr":      round(oos_cagr, 2),
            "oos_max_dd":    round(oos_dd, 2),
            "oos_calmar":    round(oos_calmar, 3),
            "oos_n_trades":  oos_trades,
            "oos_lot_reject": oos_lot_rej,
            "oos_ratio_admitted": oos_ratio_adm,
            "oos_is_ratio":  round(oos_sh / is_sh, 3) if is_sh != 0 else None,
            "wf_pass":       wf_pass,
        })
        mark = "OK" if wf_pass else "NG"
        print(f"    Seg{n} OOS={oos_s[:4]}  IS_Sh={is_sh:.3f}  OOS_Sh={oos_sh:.3f}  "
              f"CAGR={oos_cagr:+.2f}%  DD={oos_dd:+.1f}%  "
              f"Trades={oos_trades}  LotRej={oos_lot_rej}  RatioAdm={oos_ratio_adm}  {mark}")

    # True OOS 2025
    r_true = run_s5(ds, get_active(TRUE_OOS[1]), TRUE_OOS[0], TRUE_OOS[1],
                    max_lot_cost_ratio=ratio, max_positions_ts=max_positions_ts)
    true_m = extract(r_true)
    true_adm = int(r_true.get("admitted_by_ratio_count", 0) or 0)
    true_rej = int(r_true.get("rejected_by_lot_count", 0) or 0)
    print(f"    True OOS 2025: CAGR={true_m['cagr']:+.2f}%  Sh={true_m['sharpe']:.3f}  "
          f"DD={true_m['max_dd']:.2f}%  RatioAdm={true_adm}  LotRej={true_rej}")

    # Full IS 2018-2024
    r_is_full = run_s5(ds, get_active(IS_FULL[1]), IS_FULL[0], IS_FULL[1],
                       max_lot_cost_ratio=ratio, max_positions_ts=max_positions_ts)
    is_full_m = extract(r_is_full)
    is_full_adm = int(r_is_full.get("admitted_by_ratio_count", 0) or 0)
    is_full_rej = int(r_is_full.get("rejected_by_lot_count", 0) or 0)
    print(f"    Full IS 2018-2024: CAGR={is_full_m['cagr']:+.2f}%  Sh={is_full_m['sharpe']:.3f}  "
          f"DD={is_full_m['max_dd']:.2f}%  RatioAdm={is_full_adm}  LotRej={is_full_rej}")

    # ratio 救済入場の fwd20d return（OOS 5-fold プール）
    fwd20 = compute_admitted_fwd_returns(admitted_detail_all, ds["universe_raw"],
                                          common_dates, fwd_days=FWD_DAYS)
    print(f"    Ratio-Admitted fwd{FWD_DAYS}d: {fwd20['avg_fwd_ret_pct']}%  n={fwd20['n']}  "
          f"WR={fwd20['pct_positive']}%")

    wf_count = sum(1 for s in seg_results if s["wf_pass"])
    avg_cagr = float(np.mean([s["oos_cagr"] for s in seg_results]))
    seg3 = next((s["oos_cagr"] for s in seg_results if s["oos_year"] == "2022"), None)
    worst_dd = min(s["oos_max_dd"] for s in seg_results)

    return {
        "case": case_name,
        "ratio": ratio,
        "wf_segments":       seg_results,
        "wf_count":          wf_count,
        "avg_oos_cagr":      round(avg_cagr, 2),
        "seg3_2022_cagr":    round(seg3, 2) if seg3 is not None else None,
        "worst_oos_dd":      round(worst_dd, 2),
        "true_oos_2025":     true_m,
        "true_oos_ratio_admitted": true_adm,
        "true_oos_lot_rejected": true_rej,
        "full_is_2018_2024": is_full_m,
        "full_is_ratio_admitted": is_full_adm,
        "full_is_lot_rejected": is_full_rej,
        "admitted_fwd20d":   fwd20,
        "admitted_total_oos": len(admitted_detail_all),
    }


def main() -> int:
    print("=" * 78)
    print("  Study44 — Lot Cost Ratio Walk-Forward (Revised)")
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

    # ── 4. Vol-Adjusted max_positions series（interaction estimate 用）──
    print("\n[3] Vol-Adjusted max_positions series 構築（Study41 D_VOL_ADJ interaction 用）...")
    vol_adj_ts = build_vol_adjusted_ts(
        ds["topix_close"], all_common_dates,
        base_max=3, calm_max=4, vol_threshold=VOL_CALM_THRESHOLD,
    )
    calm_days = int((vol_adj_ts == 4).sum())
    total_days = len(vol_adj_ts)
    print(f"  calm days (max=4): {calm_days}/{total_days} ({calm_days/max(1,total_days)*100:.1f}%)")

    # ── 5. WF ループ ─────────────────────────────────────────────────
    print("\n[4] WF ループ（5-fold × 5 cases）...")
    all_results: dict[str, dict] = {}

    for case_name, ratio in RATIO_CASES:
        res = run_wf_case(
            case_name=case_name, ratio=ratio, ds=ds,
            get_active=get_active, common_dates=all_common_dates,
        )
        all_results[case_name] = res

    # ── 6. Interaction estimate: Study41 D_VOL_ADJ + Study44 best case ──
    print("\n[5] Interaction estimate（Study41 D_VOL_ADJ + Study44 best ratio, Full IS のみ）...")
    # Determine best case from Full IS CAGR
    best_case = max(
        [(n, r) for n, r in RATIO_CASES if r is not None],
        key=lambda x: all_results[x[0]]["full_is_2018_2024"]["cagr"]
    )
    best_name, best_ratio = best_case
    print(f"  Best ratio case: {best_name} (ratio={best_ratio})")

    # Run: S5 + D_VOL_ADJ (no ratio)
    r_vol_only = run_s5(ds, get_active(IS_FULL[1]), IS_FULL[0], IS_FULL[1],
                        max_lot_cost_ratio=None, max_positions_ts=vol_adj_ts)
    vol_only_m = extract(r_vol_only)

    # Run: S5 + D_VOL_ADJ + best_ratio
    r_combined = run_s5(ds, get_active(IS_FULL[1]), IS_FULL[0], IS_FULL[1],
                        max_lot_cost_ratio=best_ratio, max_positions_ts=vol_adj_ts)
    combined_m = extract(r_combined)

    baseline_cagr = all_results["A_BASELINE"]["full_is_2018_2024"]["cagr"]
    vol_delta  = vol_only_m["cagr"] - baseline_cagr
    ratio_delta = all_results[best_name]["full_is_2018_2024"]["cagr"] - baseline_cagr
    combined_delta = combined_m["cagr"] - baseline_cagr
    theoretical_sum = vol_delta + ratio_delta
    interaction_effect = combined_delta - theoretical_sum

    print(f"  Baseline (A): {baseline_cagr:+.2f}%")
    print(f"  +D_VOL_ADJ only: {vol_only_m['cagr']:+.2f}%  Δ={vol_delta:+.2f}pp")
    print(f"  +{best_name} only: {all_results[best_name]['full_is_2018_2024']['cagr']:+.2f}%  Δ={ratio_delta:+.2f}pp")
    print(f"  Combined: {combined_m['cagr']:+.2f}%  Δ={combined_delta:+.2f}pp")
    print(f"  Theoretical sum: Δ={theoretical_sum:+.2f}pp  Interaction: {interaction_effect:+.2f}pp")

    interaction_result = {
        "best_ratio_case": best_name,
        "best_ratio": best_ratio,
        "baseline_full_is_cagr": round(baseline_cagr, 2),
        "vol_adj_only_cagr": round(vol_only_m["cagr"], 2),
        "ratio_only_cagr": round(all_results[best_name]["full_is_2018_2024"]["cagr"], 2),
        "combined_cagr": round(combined_m["cagr"], 2),
        "delta_vol_adj_pp": round(vol_delta, 3),
        "delta_ratio_pp":   round(ratio_delta, 3),
        "delta_combined_pp": round(combined_delta, 3),
        "theoretical_sum_pp": round(theoretical_sum, 3),
        "interaction_effect_pp": round(interaction_effect, 3),
        "combined_max_dd": round(combined_m["max_dd"], 2),
        "combined_sharpe": round(combined_m["sharpe"], 3),
        "combined_calmar": round(combined_m["calmar"], 3),
    }

    # ── 7. サマリー ──────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("  Summary — WF結果")
    print("=" * 78)

    baseline_avg = all_results["A_BASELINE"]["avg_oos_cagr"]
    baseline_seg3 = all_results["A_BASELINE"]["seg3_2022_cagr"]
    baseline_dd   = all_results["A_BASELINE"]["worst_oos_dd"]
    baseline_wf   = all_results["A_BASELINE"]["wf_count"]

    print(f"\n{'Case':<12} {'Ratio':<6} {'WF':>4} {'avgCAGR':>8} {'ΔCAGR':>8} "
          f"{'seg3_22':>8} {'worstDD':>8} {'G1':>4} {'G2':>4} {'G3':>4} {'G4':>4} {'Overall':>8}")
    print("-" * 90)

    decisions: dict[str, str] = {}
    for case_name, ratio in RATIO_CASES:
        r = all_results[case_name]
        wf  = r["wf_count"]
        avg = r["avg_oos_cagr"]
        s3  = r["seg3_2022_cagr"]
        dd  = r["worst_oos_dd"]
        d_cagr = avg - baseline_avg

        g1 = "OK" if wf >= 4 else "NG"
        g2 = "OK" if d_cagr > 0.5 else ("--" if case_name == "A_BASELINE" else "NG")
        g3 = "OK" if (s3 is None or s3 >= baseline_seg3) else "NG"
        g4 = "OK" if dd >= baseline_dd else "NG"

        if case_name == "A_BASELINE":
            overall = "BASELINE"
        elif g1 == "OK" and g2 == "OK" and g3 == "OK" and g4 == "OK":
            overall = "PASS"
        elif g1 == "OK" and g2 == "NG" and g3 == "OK" and g4 == "OK":
            overall = "WEAK"
        else:
            overall = "FAIL"

        decisions[case_name] = overall
        ratio_str = f"{ratio:.2f}" if ratio is not None else "None"
        print(f"{case_name:<12} {ratio_str:<6} {wf:>4} {avg:>+8.2f}% {d_cagr:>+8.2f}pp "
              f"{(s3 if s3 else 0):>+8.2f}% {dd:>+8.2f}% "
              f"{g1:>4} {g2:>4} {g3:>4} {g4:>4} {overall:>8}")

    # ── 8. JSON 保存 ─────────────────────────────────────────────────
    output = {
        "study": "Study44",
        "title": "Lot Cost Ratio Walk-Forward",
        "date":  str(date.today()),
        "params": {
            "capital": CAPITAL,
            "min_hold": MIN_HOLD,
            "fwd_days": FWD_DAYS,
            "baseline_s5_config": "S5 (topix_close=composite, rsr_exit=70, atr_trailing=True, no_mtf, no_atr_risk)",
            "study42_rejected_trades": 20,
            "study42_fwd20d_avg": 5.59,
            "study42_win_rate_pct": 80.0,
        },
        "cases": all_results,
        "interaction_study41_d_vol_adj": interaction_result,
        "decisions": decisions,
    }

    out_path = Path(RESULTS_DIR) / f"study44_lot_cost_ratio_wf_202606_{date.today()}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n[6] 結果保存: {out_path}")

    # ── 9. 決定サマリー ──────────────────────────────────────────────
    print("\n" + "=" * 78)
    pass_cases = [n for n, d in decisions.items() if d == "PASS"]
    weak_cases = [n for n, d in decisions.items() if d == "WEAK"]
    print(f"  PASS: {pass_cases if pass_cases else 'なし'}")
    print(f"  WEAK: {weak_cases if weak_cases else 'なし'}")
    print(f"\n  Interaction estimate (Full IS):")
    print(f"    D_VOL_ADJ: ΔCAGR={interaction_result['delta_vol_adj_pp']:+.3f}pp")
    print(f"    {best_name}: ΔCAGR={interaction_result['delta_ratio_pp']:+.3f}pp")
    print(f"    Combined:   ΔCAGR={interaction_result['delta_combined_pp']:+.3f}pp")
    print(f"    Interaction effect: {interaction_result['interaction_effect_pp']:+.3f}pp "
          f"({'additive' if abs(interaction_result['interaction_effect_pp']) < 0.2 else 'non-additive'})")
    print("=" * 78)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
