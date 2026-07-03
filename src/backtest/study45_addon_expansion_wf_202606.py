"""
backtest/study45_addon_expansion_wf_202606.py

Study45 — Addon Expansion Walk-Forward & Idle Cash Attribution
目的: 未使用資本（idle cash 58.6%）がwinner拡張で最も効率よく活用できるか検証。

Baseline: S5 + ATR Extension（Study40 A）+ D_VOL_ADJ（Study41 D）

Phase 1: Q1-Q3 Attribution（behavior変更なし、観測のみ）
  Q1: winner(>1×ATR)保有中のidle cash平均比率
  Q2: idle日のうちaddable winnerが存在する比率
  Q3: 理論的にwinner追加投資可能なidle cash量

Phase 2: Addon Policy WF（5-fold）
  A: Control（addon=NONE）
  B: Single Add（+1 ATR）
  C: Two Stage Pyramid（+1 ATR, +2 ATR）
  D: Equity Scaled Add（cash×25%サイズ, +1 ATR）
  E: Vol-Adjusted Add（B + D_VOL_ADJ calm日のみ）
  F: Hybrid（E + 高閾値 1.5×ATR）

Interaction test:
  ATR Extension only → +D_VOL_ADJ → +Addon (best)

Gates（全Case共通）:
  G1: WF ≥ 4/5
  G2: ΔCAGR > +0.5pp vs Baseline
  G3: seg3(2022) non-degradation
  G4: MaxDD non-worsening

Output: backtests/study45_addon_expansion_wf_202606_<date>.json
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

# Addon policy 定義
ADDON_CASES: list[tuple] = [
    # (case_name, addon_policy, addon_atr_mult, addon_stage2_mult, addon_max_per_pos, addon_size_frac)
    ("A_CONTROL", "NONE", 1.0, 2.0, 1, 0.25),
    ("B_SINGLE",  "B",    1.0, 2.0, 1, 0.25),   # +1×ATR, 1 lot
    ("C_PYRAMID", "C",    1.0, 2.0, 2, 0.25),   # +1×ATR then +2×ATR, pyramid
    ("D_EQ_SCALE","D",    1.0, 2.0, 1, 0.25),   # +1×ATR, size=cash×25%
    ("E_VOL_ADJ", "E",    1.0, 2.0, 1, 0.25),   # B + calm市場ゲート
    ("F_HYBRID",  "F",    1.5, 2.5, 1, 0.25),   # E + 高閾値(1.5×ATR)
]


def build_vol_adjusted_ts(topix_close: pd.Series, common_dates: list,
                           base_max: int = 3, calm_max: int = 4,
                           vol_threshold: float = VOL_CALM_THRESHOLD) -> pd.Series:
    topix_ret = topix_close.pct_change()
    rolling_std = topix_ret.rolling(20, min_periods=10).std()
    std_series = rolling_std.reindex(pd.Index(common_dates), method="ffill").fillna(rolling_std.median())
    max_pos_ts = pd.Series(base_max, index=std_series.index, dtype=int)
    max_pos_ts[std_series < vol_threshold] = calm_max
    return max_pos_ts


def run_s5_addon(ds: dict, sym_active_df: "pd.DataFrame | None",
                 start: str, end: str,
                 max_positions_ts: "pd.Series | None",
                 addon_policy: str = "NONE",
                 addon_atr_mult: float = 1.0,
                 addon_stage2_mult: float = 2.0,
                 addon_max_per_pos: int = 1,
                 addon_size_frac: float = 0.25) -> dict:
    """S5 + ATR Extension + D_VOL_ADJ + addon policy で run_scenario を呼ぶ。"""
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
        enable_atr_trailing_prod=True,   # Study40 ATR Extension
        enable_multilayer_rsr=True,
        enable_atr_risk_sizing=False,
        enable_mtf_filter=False,
        sizing_mode="existing",
        exit_policy="A",                 # Study40: ATR Extension
        max_positions_ts=max_positions_ts,  # Study41 D_VOL_ADJ
        # Study45 addon
        addon_policy=addon_policy,
        addon_atr_mult=addon_atr_mult,
        addon_stage2_mult=addon_stage2_mult,
        addon_max_per_pos=addon_max_per_pos,
        addon_size_frac=addon_size_frac,
    )


def extract(res: dict) -> dict:
    return {k: res.get(k, 0.0) for k in ("cagr", "sharpe", "max_dd", "calmar",
                                           "n_trades", "sortino", "profit_factor")}


def extract_q(res: dict) -> dict:
    return {
        "q_idle_days":          res.get("q_idle_days", 0),
        "q_idle_cash_avg_pct":  res.get("q_idle_cash_avg_pct", None),
        "q1_idle_when_winner_pct": res.get("q1_idle_when_winner_pct", None),
        "q2_idle_days_with_winner_pct": res.get("q2_idle_days_with_winner_pct", None),
        "q3_deployable_idle_cash_avg_pct": res.get("q3_deployable_idle_cash_avg_pct", None),
        "addon_count": res.get("addon_count", 0),
        "avg_exposure": res.get("avg_exposure", None),
    }


def run_wf_case(case_name: str, addon_args: tuple, ds: dict, get_active,
                vol_adj_ts: "pd.Series | None",
                common_dates: list) -> dict:
    _, addon_policy, addon_atr_mult, addon_stage2_mult, addon_max_per_pos, addon_size_frac = addon_args
    print(f"\n  ── {case_name}  policy={addon_policy}  atr_mult={addon_atr_mult} ──")

    seg_results = []

    for seg in WF_SEGS:
        n = seg["seg"]
        is_s, is_e = seg["is"]
        oos_s, oos_e = seg["oos"]

        r_is  = run_s5_addon(ds, get_active(is_e),  is_s,  is_e,  vol_adj_ts,
                              addon_policy, addon_atr_mult, addon_stage2_mult,
                              addon_max_per_pos, addon_size_frac)
        r_oos = run_s5_addon(ds, get_active(oos_e), oos_s, oos_e, vol_adj_ts,
                              addon_policy, addon_atr_mult, addon_stage2_mult,
                              addon_max_per_pos, addon_size_frac)

        is_sh   = float(r_is.get("sharpe",  0.0) or 0.0)
        oos_sh  = float(r_oos.get("sharpe", 0.0) or 0.0)
        oos_cagr   = float(r_oos.get("cagr",   0.0) or 0.0)
        oos_dd     = float(r_oos.get("max_dd",  0.0) or 0.0)
        oos_calmar = float(r_oos.get("calmar",  0.0) or 0.0)
        oos_trades = int(r_oos.get("n_trades", 0) or 0)
        oos_addons = int(r_oos.get("addon_count", 0) or 0)
        oos_exposure = r_oos.get("avg_exposure", None)
        wf_pass = oos_sh > 0

        seg_results.append({
            "seg": n, "oos_year": oos_s[:4],
            "is_sharpe":   round(is_sh, 3),
            "oos_sharpe":  round(oos_sh, 3),
            "oos_cagr":    round(oos_cagr, 2),
            "oos_max_dd":  round(oos_dd, 2),
            "oos_calmar":  round(oos_calmar, 3),
            "oos_n_trades": oos_trades,
            "oos_addon_count": oos_addons,
            "oos_avg_exposure": oos_exposure,
            "wf_pass": wf_pass,
        })
        q = extract_q(r_oos)
        mark = "OK" if wf_pass else "NG"
        print(f"    Seg{n} OOS={oos_s[:4]}  Sh={oos_sh:.3f}  CAGR={oos_cagr:+.2f}%  "
              f"DD={oos_dd:+.1f}%  Addons={oos_addons}  "
              f"Q2={q['q2_idle_days_with_winner_pct']}%  Q3={q['q3_deployable_idle_cash_avg_pct']}%  {mark}")

    # True OOS 2025
    r_true = run_s5_addon(ds, get_active(TRUE_OOS[1]), TRUE_OOS[0], TRUE_OOS[1], vol_adj_ts,
                           addon_policy, addon_atr_mult, addon_stage2_mult,
                           addon_max_per_pos, addon_size_frac)
    true_m = extract(r_true)
    true_q = extract_q(r_true)
    print(f"    True OOS 2025: CAGR={true_m['cagr']:+.2f}%  Sh={true_m['sharpe']:.3f}  "
          f"Addons={true_q['addon_count']}  Q2={true_q['q2_idle_days_with_winner_pct']}%")

    # Full IS 2018-2024
    r_full = run_s5_addon(ds, get_active(IS_FULL[1]), IS_FULL[0], IS_FULL[1], vol_adj_ts,
                           addon_policy, addon_atr_mult, addon_stage2_mult,
                           addon_max_per_pos, addon_size_frac)
    full_m = extract(r_full)
    full_q = extract_q(r_full)
    print(f"    Full IS 2018-2024: CAGR={full_m['cagr']:+.2f}%  Sh={full_m['sharpe']:.3f}  "
          f"DD={full_m['max_dd']:.2f}%  Addons={full_q['addon_count']}  "
          f"Q1={full_q['q1_idle_when_winner_pct']}%  Q2={full_q['q2_idle_days_with_winner_pct']}%  "
          f"Q3={full_q['q3_deployable_idle_cash_avg_pct']}%")

    wf_count = sum(1 for s in seg_results if s["wf_pass"])
    avg_cagr = float(np.mean([s["oos_cagr"] for s in seg_results]))
    seg3 = next((s["oos_cagr"] for s in seg_results if s["oos_year"] == "2022"), None)
    worst_dd = min(s["oos_max_dd"] for s in seg_results)

    return {
        "case": case_name,
        "addon_policy": addon_policy,
        "wf_segments":       seg_results,
        "wf_count":          wf_count,
        "avg_oos_cagr":      round(avg_cagr, 2),
        "seg3_2022_cagr":    round(seg3, 2) if seg3 is not None else None,
        "worst_oos_dd":      round(worst_dd, 2),
        "true_oos_2025":     true_m,
        "true_oos_q":        true_q,
        "full_is_2018_2024": full_m,
        "full_is_q":         full_q,
    }


def main() -> int:
    print("=" * 78)
    print("  Study45 — Addon Expansion Walk-Forward & Idle Cash Attribution")
    print(f"  Baseline: S5 + ATR Extension + D_VOL_ADJ  capital={CAPITAL:,}")
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

    # ── 4. Vol-Adjusted max_positions series（D_VOL_ADJ baseline用）──
    print("\n[3] D_VOL_ADJ max_positions series 構築...")
    vol_adj_ts = build_vol_adjusted_ts(ds["topix_close"], all_common_dates)
    calm_days = int((vol_adj_ts == 4).sum())
    print(f"  calm days (max=4): {calm_days}/{len(vol_adj_ts)} ({calm_days/max(1,len(vol_adj_ts))*100:.1f}%)")

    # ── 5. Phase 1: Q1-Q3 Attribution（A_CONTROL run）───────────────
    print("\n[4] Phase 1: Q1-Q3 Attribution (A_CONTROL = S5+ATR_EXT+D_VOL_ADJ, Full IS)...")
    r_attr = run_s5_addon(ds, get_active(IS_FULL[1]), IS_FULL[0], IS_FULL[1],
                           vol_adj_ts, "NONE")
    attr_q = extract_q(r_attr)
    attr_m = extract(r_attr)
    print(f"  Baseline Full IS: CAGR={attr_m['cagr']:+.2f}%  Sh={attr_m['sharpe']:.3f}  "
          f"DD={attr_m['max_dd']:.2f}%")
    print(f"  Q1 (idle cash % when winner exists):  {attr_q['q1_idle_when_winner_pct']}%")
    print(f"  Q2 (idle days with addable winner):   {attr_q['q2_idle_days_with_winner_pct']}%")
    print(f"  Q3 (deployable idle cash avg/day):    {attr_q['q3_deployable_idle_cash_avg_pct']}%")
    print(f"  Idle days: {attr_q['q_idle_days']}  Avg idle cash: {attr_q['q_idle_cash_avg_pct']}%")

    # ── 6. Phase 2: Addon Policy WF ─────────────────────────────────
    print("\n[5] Phase 2: Addon Policy Walk-Forward (5-fold × 6 cases)...")
    all_results: dict[str, dict] = {}

    for addon_args in ADDON_CASES:
        case_name = addon_args[0]
        res = run_wf_case(
            case_name=case_name,
            addon_args=addon_args,
            ds=ds,
            get_active=get_active,
            vol_adj_ts=vol_adj_ts,
            common_dates=all_common_dates,
        )
        all_results[case_name] = res

    # ── 7. Interaction Test（Full IS）────────────────────────────────
    print("\n[6] Interaction Test（Full IS, incrementing features）...")

    # (i) S5 only（ATR ext なし、vol_adj なし）
    r_s5_raw = cab.run_scenario(
        scenario="BASELINE",
        universe_raw=ds["universe_raw"], rsr_df=ds["rsr_df"], alpha_df=None,
        regime_df=ds["regime_df"], trade_syms=ds["trade_syms"],
        rsr_syms=ds["rsr_syms"], cfg=ds["base_cfg"],
        start=IS_FULL[0], end=IS_FULL[1], verbose=False,
        tech_matrices=ds["tech_matrices"], breadth_series=ds["breadth_series"],
        capital=CAPITAL, min_hold=MIN_HOLD,
        topix_close=ds["topix_close"], market_shock_mode="composite",
        rsr_exit_threshold=70.0, sym_active_df=get_active(IS_FULL[1]),
        enable_simple_rsr_exit=True, enable_atr_trailing_prod=True,
        enable_multilayer_rsr=True, enable_atr_risk_sizing=False,
        enable_mtf_filter=False, sizing_mode="existing",
        exit_policy="NONE",      # ATR ext なし
        max_positions_ts=None,   # vol_adj なし
    )

    # (ii) S5 + ATR Extension only
    r_atr_only = cab.run_scenario(
        scenario="BASELINE",
        universe_raw=ds["universe_raw"], rsr_df=ds["rsr_df"], alpha_df=None,
        regime_df=ds["regime_df"], trade_syms=ds["trade_syms"],
        rsr_syms=ds["rsr_syms"], cfg=ds["base_cfg"],
        start=IS_FULL[0], end=IS_FULL[1], verbose=False,
        tech_matrices=ds["tech_matrices"], breadth_series=ds["breadth_series"],
        capital=CAPITAL, min_hold=MIN_HOLD,
        topix_close=ds["topix_close"], market_shock_mode="composite",
        rsr_exit_threshold=70.0, sym_active_df=get_active(IS_FULL[1]),
        enable_simple_rsr_exit=True, enable_atr_trailing_prod=True,
        enable_multilayer_rsr=True, enable_atr_risk_sizing=False,
        enable_mtf_filter=False, sizing_mode="existing",
        exit_policy="A",         # ATR ext あり
        max_positions_ts=None,   # vol_adj なし
    )

    # (iii) ATR Extension + D_VOL_ADJ (= baseline A_CONTROL)
    r_base = extract(r_attr)

    # (iv) ATR Extension + D_VOL_ADJ + best addon
    best_case = max(
        [n for n in all_results if n != "A_CONTROL"],
        key=lambda n: all_results[n]["full_is_2018_2024"]["cagr"]
    )
    r_best_full = run_s5_addon(
        ds, get_active(IS_FULL[1]), IS_FULL[0], IS_FULL[1], vol_adj_ts,
        *ADDON_CASES[[a[0] for a in ADDON_CASES].index(best_case)][1:]
    )
    r_best_m = extract(r_best_full)

    s5_cagr   = float(extract(r_s5_raw)["cagr"])
    atr_cagr  = float(extract(r_atr_only)["cagr"])
    base_cagr = float(r_base["cagr"])
    best_cagr = float(r_best_m["cagr"])

    print(f"  S5 only (exit_policy=NONE, no vol_adj):   {s5_cagr:+.2f}%")
    print(f"  + ATR Extension (exit_policy=A):          {atr_cagr:+.2f}%  Δ={atr_cagr-s5_cagr:+.2f}pp")
    print(f"  + D_VOL_ADJ (A_CONTROL baseline):         {base_cagr:+.2f}%  Δ={base_cagr-atr_cagr:+.2f}pp")
    print(f"  + {best_case}:                            {best_cagr:+.2f}%  Δ={best_cagr-base_cagr:+.2f}pp")

    interaction_result = {
        "s5_raw_cagr":          round(s5_cagr, 2),
        "atr_ext_only_cagr":    round(atr_cagr, 2),
        "atr_plus_voladj_cagr": round(base_cagr, 2),
        "best_addon_case":      best_case,
        "best_addon_cagr":      round(best_cagr, 2),
        "delta_atr_ext_pp":     round(atr_cagr - s5_cagr, 3),
        "delta_voladj_pp":      round(base_cagr - atr_cagr, 3),
        "delta_addon_pp":       round(best_cagr - base_cagr, 3),
        "combined_delta_pp":    round(best_cagr - s5_cagr, 3),
    }

    # ── 8. Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("  Summary — WF 結果（vs A_CONTROL baseline）")
    print("=" * 78)

    baseline_avg = all_results["A_CONTROL"]["avg_oos_cagr"]
    baseline_seg3 = all_results["A_CONTROL"]["seg3_2022_cagr"]
    baseline_dd   = all_results["A_CONTROL"]["worst_oos_dd"]
    baseline_wf   = all_results["A_CONTROL"]["wf_count"]

    print(f"\n{'Case':<14} {'Pol':<5} {'WF':>4} {'avgCAGR':>9} {'ΔCAGR':>8} "
          f"{'seg3_22':>8} {'worstDD':>8} {'G1':>4} {'G2':>4} {'G3':>4} {'G4':>4} {'Overall':>8}")
    print("-" * 96)

    decisions: dict[str, str] = {}
    for addon_args in ADDON_CASES:
        case_name = addon_args[0]
        r = all_results[case_name]
        wf  = r["wf_count"]
        avg = r["avg_oos_cagr"]
        s3  = r["seg3_2022_cagr"]
        dd  = r["worst_oos_dd"]
        d_cagr = avg - baseline_avg

        g1 = "OK" if wf >= 4 else "NG"
        g2 = "OK" if d_cagr > 0.5 else ("--" if case_name == "A_CONTROL" else "NG")
        g3 = "OK" if (s3 is None or s3 >= baseline_seg3) else "NG"
        g4 = "OK" if dd >= baseline_dd else "NG"

        if case_name == "A_CONTROL":
            overall = "BASELINE"
        elif all(g == "OK" for g in [g1, g2, g3, g4]):
            overall = "PASS"
        elif g1 == "OK" and g2 == "NG" and g3 == "OK" and g4 == "OK":
            overall = "WEAK"
        else:
            overall = "FAIL"

        decisions[case_name] = overall
        print(f"{case_name:<14} {addon_args[1]:<5} {wf:>4} {avg:>+9.2f}% {d_cagr:>+8.2f}pp "
              f"{(s3 if s3 else 0):>+8.2f}% {dd:>+8.2f}% "
              f"{g1:>4} {g2:>4} {g3:>4} {g4:>4} {overall:>8}")

    # ── 9. JSON 保存 ─────────────────────────────────────────────────
    output = {
        "study": "Study45",
        "title": "Addon Expansion Walk-Forward & Idle Cash Attribution",
        "date":  str(date.today()),
        "params": {
            "capital": CAPITAL,
            "min_hold": MIN_HOLD,
            "baseline": "S5 + exit_policy=A (ATR Extension) + D_VOL_ADJ (Study41)",
            "study41_idle_cash_pct": 58.6,
        },
        "phase1_q1q2q3_attribution_full_is": attr_q,
        "baseline_full_is": attr_m,
        "cases": all_results,
        "interaction_test": interaction_result,
        "decisions": decisions,
    }

    out_path = Path(RESULTS_DIR) / f"study45_addon_expansion_wf_202606_{date.today()}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n[7] 結果保存: {out_path}")

    # ── 10. 決定サマリー ─────────────────────────────────────────────
    print("\n" + "=" * 78)
    pass_cases = [n for n, d in decisions.items() if d == "PASS"]
    weak_cases = [n for n, d in decisions.items() if d == "WEAK"]
    print(f"  PASS: {pass_cases if pass_cases else 'なし'}")
    print(f"  WEAK: {weak_cases if weak_cases else 'なし'}")
    print(f"\n  Q1-Q3 Attribution（Full IS）:")
    print(f"    Q1 idle cash % when winner present: {attr_q['q1_idle_when_winner_pct']}%")
    print(f"    Q2 idle days with addable winner:   {attr_q['q2_idle_days_with_winner_pct']}%")
    print(f"    Q3 deployable idle cash avg/day:    {attr_q['q3_deployable_idle_cash_avg_pct']}%")
    print(f"\n  Interaction test（Full IS）:")
    print(f"    S5 raw → +ATR Extension: {interaction_result['delta_atr_ext_pp']:+.3f}pp")
    print(f"    +D_VOL_ADJ:              {interaction_result['delta_voladj_pp']:+.3f}pp")
    print(f"    +{best_case}:            {interaction_result['delta_addon_pp']:+.3f}pp")
    print(f"    Combined from S5:        {interaction_result['combined_delta_pp']:+.3f}pp")
    print(f"\n  Production CAGR estimate（Full IS best）:")
    print(f"    S5 base              = {all_results['A_CONTROL']['full_is_2018_2024']['cagr']:+.2f}%")
    if pass_cases:
        best_pass = max(pass_cases, key=lambda n: all_results[n]["full_is_2018_2024"]["cagr"])
        print(f"    + {best_pass}          = {all_results[best_pass]['full_is_2018_2024']['cagr']:+.2f}%")
    print("=" * 78)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
