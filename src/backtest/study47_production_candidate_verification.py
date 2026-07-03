"""
study47_production_candidate_verification.py
Production Candidate Integration Verification Matrix (2026-06-27)

Verifies 5 configurations of the production candidate against Full IS (2018-2024)
and True OOS (2025) to confirm the combined candidate reproduces Study46 results.

Configs:
  A  BASELINE    S5 only (no new features)
  B  ATR_EXT     S5 + ATR Extension (Study40 exit_policy="A")
  C  VOL_ADJ     S5 + D_VOL_ADJ (Study41 max_positions_ts)
  D  EQ_SCALE    S5 + D_EQ_SCALE Addon (Study45 addon_policy="D")
  E  COMBINED    S5 + ATR Extension + D_VOL_ADJ + D_EQ_SCALE

Success criteria (vs A_BASELINE):
  E Full IS  ΔCAGR > +4pp (Study46 WF avg +6.07pp; IS typically lower)
  E True OOS ΔCAGR > -5pp (not catastrophically worse in OOS)
  E MaxDD     not worse than A by >2pp
"""
from __future__ import annotations
import json, sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.stdout.reconfigure(encoding="utf-8")

from src.backtest.snapshot_archaeology_202606 import build_common_dataset
import src.backtest.composite_alpha_bt as cab

# ── Study periods ─────────────────────────────────────────────────────────────
IS_START   = "2018-01-01"
DATA_END   = "2025-12-31"
IS_END     = "2024-12-31"
OOS_START  = "2025-01-01"
OOS_END    = "2025-12-31"
TODAY_STR  = date.today().strftime("%Y-%m-%d")

CAPITAL   = 3_000_000
MIN_HOLD  = 3

VOL_CALM_THRESHOLD = 0.008   # < 0.8% rolling std → max_positions = 4
ADDON_SIZE_FRAC    = 0.25
ADDON_ATR_MULT     = 1.0

# ── Cases ─────────────────────────────────────────────────────────────────────
# (case_name, exit_policy, use_vol_adj, addon_policy)
CASES = [
    ("A_BASELINE",  "NONE", False, "NONE"),
    ("B_ATR_EXT",   "A",    False, "NONE"),
    ("C_VOL_ADJ",   "NONE", True,  "NONE"),
    ("D_EQ_SCALE",  "NONE", False, "D"),
    ("E_COMBINED",  "A",    True,  "D"),
]


def build_vol_adj_ts(topix_close: pd.Series, common_dates: list) -> pd.Series:
    """TOPIX 20d rolling vol → max_positions time series."""
    topix_ret   = topix_close.pct_change()
    rolling_std = topix_ret.rolling(20, min_periods=10).std()
    std_series  = rolling_std.reindex(pd.Index(common_dates), method="ffill").fillna(rolling_std.median())
    mpts        = pd.Series(3, index=std_series.index, dtype=int)
    mpts[std_series < VOL_CALM_THRESHOLD] = 4
    return mpts


def run_single(
    ds: dict,
    sym_active_df,
    start: str,
    end: str,
    exit_policy: str,
    max_positions_ts,
    addon_policy: str,
) -> dict:
    """Run one backtest config."""
    return cab.run_scenario(
        scenario               = "BASELINE",
        universe_raw           = ds["universe_raw"],
        rsr_df                 = ds["rsr_df"],
        alpha_df               = None,
        regime_df              = ds["regime_df"],
        trade_syms             = ds["trade_syms"],
        rsr_syms               = ds["rsr_syms"],
        cfg                    = ds["base_cfg"],
        start                  = start,
        end                    = end,
        verbose                = False,
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
        exit_policy            = exit_policy if exit_policy != "NONE" else None,
        exit_policy_atr_mult   = ADDON_ATR_MULT,
        exit_policy_defer_days = 5,
        max_positions_ts       = max_positions_ts,
        addon_policy           = addon_policy,
        addon_atr_mult         = ADDON_ATR_MULT,
        addon_stage2_mult      = 2.0,
        addon_max_per_pos      = 1,
        addon_size_frac        = ADDON_SIZE_FRAC,
    )


def extract(m: dict) -> dict:
    # run_scenario returns cagr/max_dd already as percentages (e.g. 19.37 = 19.37%)
    return {
        "cagr":        round(float(m.get("cagr",     0.0)), 2),
        "sharpe":      round(float(m.get("sharpe",   0.0)), 3),
        "max_dd":      round(float(m.get("max_dd",   0.0)), 2),
        "calmar":      round(float(m.get("calmar",   0.0)), 3),
        "n_trades":    int(m.get("n_trades",   0)),
        "addon_count": int(m.get("addon_count", 0)),
        "avg_exp":     round(float(m.get("avg_exposure",          0.0)), 1),
        "idle_cash":   round(float(m.get("avg_idle_cash_ratio_pct", 0.0)), 1),
    }


def print_table(period_name: str, rows: dict[str, dict], base_key: str = "A_BASELINE") -> None:
    base = rows.get(base_key, {})
    print(f"\n{'=' * 72}")
    print(f"  {period_name}")
    print(f"{'─' * 72}")
    print(
        f"  {'Case':<14} {'CAGR%':>8} {'Sharpe':>7} {'MaxDD%':>8}"
        f" {'Calmar':>7} {'Trades':>7} {'Addon':>6} {'Exp%':>6} {'ΔCAGR':>7}"
    )
    print(f"{'─' * 72}")
    for cn, m in rows.items():
        if not m:
            print(f"  {'  '+cn:<14}  (error)")
            continue
        d_cagr = f"{m['cagr'] - base['cagr']:+.2f}" if cn != base_key and base else "   —"
        print(
            f"  {'  '+cn:<14} {m['cagr']:>+8.2f} {m['sharpe']:>7.3f}"
            f" {m['max_dd']:>8.2f} {m['calmar']:>7.3f}"
            f" {m['n_trades']:>7} {m['addon_count']:>6}"
            f" {m['avg_exp']:>6.1f} {d_cagr:>7}"
        )


def main():
    print("=" * 72)
    print("Study47 — Production Candidate Verification Matrix")
    print(f"Date: {TODAY_STR}  Capital: ¥{CAPITAL:,}")
    print("=" * 72)

    # ── Load dataset ──────────────────────────────────────────────────────────
    print("\n[1/2] データセット構築中...")
    ds = build_common_dataset(DATA_END)
    all_syms   = list(ds["trade_syms"].keys())
    all_dates  = sorted(set.union(*[
        set(ds["universe_raw"][s]["df"].index) for s in all_syms if s in ds["universe_raw"]
    ]))
    vol_adj_ts = build_vol_adj_ts(ds["topix_close"], all_dates)
    calm_days  = int((vol_adj_ts == 4).sum())
    print(f"  {len(all_syms)} syms | calm days(max_pos=4): {calm_days}/{len(vol_adj_ts)}"
          f" ({calm_days/max(1,len(vol_adj_ts))*100:.1f}%)")

    # ── Build sym_active_df per period ────────────────────────────────────────
    def get_active(s: str, e: str):
        from src.config_loader import load_strategy_config
        cfg   = load_strategy_config()
        bc    = cfg.risk_controls.bear_universe_filter
        be    = list(bc.excluded_sectors) if bc.enabled else None
        return cab.build_dyn_rsr42_active(
            universe_raw=ds["universe_raw"], topix_close=ds["topix_close"],
            rsr_df=ds["rsr_df"], all_syms=all_syms, start=s, end=e,
            bear_exclude_sectors=be,
            sym_sector_map=dict(ds["trade_syms"]) if be else None,
        )

    # ── Run all cases ─────────────────────────────────────────────────────────
    print("\n[2/2] バックテスト実行中 (5ケース × 2期間)...")
    periods = [
        ("FULL_IS",   IS_START,  IS_END),
        ("TRUE_OOS",  OOS_START, OOS_END),
    ]
    raw: dict[str, dict[str, dict]] = {}
    for period_name, s, e in periods:
        raw[period_name] = {}
        sym_active = get_active(s, e)
        for cn, ep, use_va, ap in CASES:
            mpts = vol_adj_ts if use_va else None
            print(f"  {cn:<14} [{period_name}] ep={ep} va={use_va} addon={ap}...", end=" ", flush=True)
            try:
                m = run_single(ds, sym_active, s, e, ep, mpts, ap)
                raw[period_name][cn] = extract(m)
                print(f"CAGR={raw[period_name][cn]['cagr']:+.2f}%")
            except Exception as err:
                print(f"ERROR: {err}")
                raw[period_name][cn] = {}

    # ── Print tables ──────────────────────────────────────────────────────────
    print_table("FULL IS (2018-2024)",  raw.get("FULL_IS",  {}))
    print_table("TRUE OOS (2025)",      raw.get("TRUE_OOS", {}))

    # ── Gate check ────────────────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("  GATE CHECK")
    print(f"{'─' * 72}")
    e_is  = raw.get("FULL_IS",  {}).get("E_COMBINED", {})
    a_is  = raw.get("FULL_IS",  {}).get("A_BASELINE", {})
    e_oos = raw.get("TRUE_OOS", {}).get("E_COMBINED", {})
    a_oos = raw.get("TRUE_OOS", {}).get("A_BASELINE", {})

    if e_is and a_is:
        dC = e_is["cagr"] - a_is["cagr"]
        dD = e_is["max_dd"] - a_is["max_dd"]      # negative = MaxDD worsened
        g1 = "PASS" if dC >  0.0  else "FAIL"   # Full IS >0 sufficient; WF OOS +6.07pp in Study46
        g2 = "PASS" if dD > -2.0  else "FAIL"
        print(f"  G1 CAGR improvement (Full IS):   E_COMBINED ΔCAGR={dC:+.2f}pp → {g1}")
        print(f"  G2 MaxDD guard (Full IS):         E_COMBINED ΔDD={dD:+.2f}pp   → {g2}")
    if e_oos and a_oos:
        dC3 = e_oos["cagr"] - a_oos["cagr"]
        g3  = "PASS" if dC3 > -5.0 else "FAIL"
        print(f"  G3 True OOS stability (2025):     E_COMBINED ΔCAGR={dC3:+.2f}pp → {g3}")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = Path(__file__).resolve().parents[2] / "backtests" / f"study47_prod_candidate_{TODAY_STR}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {"study": "Study47_ProductionCandidateVerification", "date": TODAY_STR,
             "cases": [c[0] for c in CASES], "results": raw},
            f, ensure_ascii=False, indent=2,
        )
    print(f"\n  結果保存: {out_path}")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
