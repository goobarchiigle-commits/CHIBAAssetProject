"""
backtest/study42_capital_constraint_archaeology_202606.py

Study42 — Capital Constraint Archaeology
目的: 原本 CAGR22.4% vs APRIL_REPRO_A 20.14% の 2.26pp 残差のうち、
     資本制約（¥3M枠 + Lot100）が何%を説明するかを定量化する。

Cases（全て Full IS 2018-2024、APRIL_REPRO_A config）:
  A: Current Realistic     (Capital=¥3M,   Lot=100)  ← APRIL_REPRO_A 再現基準点
  B: Large Capital         (Capital=¥30M,  Lot=100)
  C: Unlimited Capital     (Capital=¥300M, Lot=100)
  D: Fractional Shares     (Capital=¥3M,   Lot=1)
  E: Legacy Research Mode  (Capital=¥300M, Lot=1)   ← 理論上の天井
  F: Capital Sweep         (3M→5M→10M→30M→100M→300M, Lot=100)

Attribution verdict:
  >50% gap explained  → CAPITAL_LOT_DOMINANT
  10-50%              → PARTIAL_ATTRIBUTION
  <10%                → OTHER_ENGINE_CAUSE

Output: backtests/study42_capital_constraint_archaeology_202606_<date>.json
"""
from __future__ import annotations

import sys
import json
import warnings
from pathlib import Path
from datetime import date as _date

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

from src.backtest import composite_alpha_bt as cab
from src.backtest.snapshot_archaeology_202606 import build_common_dataset
from src.paths import RESULTS_DIR

# ── 定数 ────────────────────────────────────────────────────────────────
IS_START = "2018-01-01"
IS_END   = "2024-12-31"

# 原本記録値 (min_hold_sensitivity_2026-03-31.json hold3d.IS)
ORIGINAL_CAGR  = 22.4
ORIGINAL_SHARPE = 1.582
ORIGINAL_MAXDD  = -12.32

# APRIL_REPRO_A 確定値 (study36/37/38 で確認)
REPRO_A_CAGR   = 20.14
REPRO_A_SHARPE = 0.859
TOTAL_GAP_PP   = ORIGINAL_CAGR - REPRO_A_CAGR   # ≈ 2.26pp

# Capital sweep ポイント
CAPITAL_SWEEP = [3_000_000, 5_000_000, 10_000_000, 30_000_000,
                 100_000_000, 300_000_000]


def run_april_repro(ds: dict, capital: int = 3_000_000, lot_size: int = 100) -> dict:
    """APRIL_REPRO_A config で run_scenario を実行。"""
    return cab.run_scenario(
        scenario             = "BASELINE",
        universe_raw         = ds["universe_raw"],
        rsr_df               = ds["rsr_df"],
        alpha_df             = None,
        regime_df            = ds["regime_df"],
        trade_syms           = ds["trade_syms"],
        rsr_syms             = ds["rsr_syms"],
        cfg                  = ds["base_cfg"],
        start                = IS_START,
        end                  = IS_END,
        verbose              = False,
        tech_matrices        = ds["tech_matrices"],
        breadth_series       = ds["breadth_series"],
        capital              = capital,
        min_hold             = 3,
        market_shock_mode    = "full_exit",
        rsr_exit_threshold   = 75.0,
        sym_active_df        = None,           # static RSR42
        enable_simple_rsr_exit       = True,
        enable_atr_trailing_prod     = False,
        enable_multilayer_rsr        = False,
        enable_atr_risk_sizing       = False,
        enable_mtf_filter            = False,
        use_fixed_pct_trail          = False,
        sizing_mode                  = "existing",
        # topix_close 渡さない（APRIL_REPRO_A は None）
        lot_size             = lot_size,
    )


def extract(res: dict) -> dict:
    return {
        "cagr":           round(float(res.get("cagr",   0.0) or 0.0), 3),
        "sharpe":         round(float(res.get("sharpe", 0.0) or 0.0), 3),
        "max_dd":         round(float(res.get("max_dd", 0.0) or 0.0), 3),
        "calmar":         round(float(res.get("calmar", 0.0) or 0.0), 3),
        "n_trades":       int(res.get("n_trades", 0) or 0),
        "avg_exposure":   round(float(res.get("avg_exposure", 0.0) or 0.0), 3),
        "win_rate":       round(float(res.get("win_rate", 0.0) or 0.0), 3),
        "avg_hold_days":  round(float(res.get("avg_hold_days", 0.0) or 0.0), 1),
        "annual_returns": res.get("annual_returns", {}),
        # Study42-specific
        "rejected_by_lot_count": int(res.get("rejected_by_lot_count", 0) or 0),
        "avg_idle_cash_ratio_pct": res.get("avg_idle_cash_ratio_pct"),
        "days_at_max_positions":  res.get("days_at_max_positions"),
        "cap_saturation_rate_pct": res.get("cap_saturation_rate_pct"),
        "missed_by_cap_count":     res.get("missed_by_cap_count"),
    }


def compute_lot_rejection_fwd_returns(rejected_detail: list[dict],
                                       universe_raw: dict,
                                       all_common_dates: list,
                                       fwd_days: int = 20) -> dict:
    """Lot制約で弾かれた候補の fwd_return を計算。"""
    date_idx: dict = {
        str(d.date()) if hasattr(d, "date") else str(d): i
        for i, d in enumerate(all_common_dates)
    }
    fwd_list: list[float] = []
    for rec in rejected_detail:
        sym  = rec.get("symbol", "")
        dstr = rec.get("date", "")
        if sym not in universe_raw or dstr not in date_idx:
            continue
        idx = date_idx[dstr]
        if idx >= len(all_common_dates):
            continue
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
        fwd_ret = float(fwd_cls.iloc[-1]) / entry_px - 1.0
        fwd_list.append(fwd_ret)
    if not fwd_list:
        return {"avg_fwd_ret_pct": None, "n": 0, "pct_positive": None}
    return {
        "avg_fwd_ret_pct": round(float(np.mean(fwd_list)) * 100, 2),
        "n": len(fwd_list),
        "pct_positive": round(sum(1 for r in fwd_list if r > 0) / len(fwd_list) * 100, 1),
    }


def main() -> int:
    print("=" * 78)
    print("  Study42 — Capital Constraint Archaeology")
    print(f"  IS={IS_START}..{IS_END}")
    print(f"  ORIGINAL={ORIGINAL_CAGR}%  REPRO_A={REPRO_A_CAGR}%  GAP={TOTAL_GAP_PP:.2f}pp")
    print("=" * 78)

    # ── 1. データセット ──────────────────────────────────────────────────
    print(f"\n[1] データセット構築（end={IS_END}）...")
    ds = build_common_dataset(IS_END)
    all_syms = list(ds["trade_syms"].keys())
    print(f"  完了 ({len(all_syms)} syms)")

    # all_common_dates（Lot拒否 fwd return計算用）
    date_sets = [set(ds["universe_raw"][s]["df"].index)
                 for s in all_syms if s in ds["universe_raw"]]
    all_common_dates: list = sorted(set.intersection(*date_sets)) if date_sets else []
    print(f"  all_common_dates: {len(all_common_dates)} days")

    # ── 2. Cases A-E ─────────────────────────────────────────────────────
    print("\n[2] Cases A-E 実行...")
    cases_ae: dict[str, dict] = {}

    run_specs = [
        ("A_CTRL_3M_LOT100",  3_000_000,   100),
        ("B_LARGE_30M_LOT100", 30_000_000,  100),
        ("C_HUGE_300M_LOT100", 300_000_000, 100),
        ("D_FRAC_3M_LOT1",    3_000_000,   1),
        ("E_LEGACY_300M_LOT1", 300_000_000, 1),
    ]
    for case_name, cap, lot in run_specs:
        print(f"\n  ── {case_name}  capital={cap:,}  lot={lot} ──")
        res_raw = run_april_repro(ds, capital=cap, lot_size=lot)
        m = extract(res_raw)
        lot_detail = res_raw.get("_rejected_by_lot_detail", [])
        fwd20_lot = compute_lot_rejection_fwd_returns(
            lot_detail, ds["universe_raw"], all_common_dates, fwd_days=20
        )
        fwd60_lot = compute_lot_rejection_fwd_returns(
            lot_detail, ds["universe_raw"], all_common_dates, fwd_days=60
        )
        m["lot_fwd20d"] = fwd20_lot
        m["lot_fwd60d"] = fwd60_lot
        cases_ae[case_name] = m
        ann = m["annual_returns"]
        print(f"    CAGR={m['cagr']:+.3f}%  Sh={m['sharpe']:.3f}  DD={m['max_dd']:.2f}%"
              f"  Trades={m['n_trades']}  LotReject={m['rejected_by_lot_count']}")
        if ann:
            years = sorted(ann.keys())
            print(f"    Annual: " + " ".join(f"{y}={ann[y]:+.1f}%" for y in years))

    # ── 3. Case F: Capital Sweep ─────────────────────────────────────────
    print("\n[3] Case F: Capital Sweep (lot=100)...")
    sweep_results: list[dict] = []
    for cap in CAPITAL_SWEEP:
        label = f"F_{cap//1_000_000}M"
        res_raw = run_april_repro(ds, capital=cap, lot_size=100)
        m = extract(res_raw)
        m["label"] = label
        m["capital"] = cap
        sweep_results.append(m)
        print(f"  {label:<14}  CAGR={m['cagr']:+.3f}%  Sh={m['sharpe']:.3f}"
              f"  DD={m['max_dd']:.2f}%  LotReject={m['rejected_by_lot_count']}")

    # ── 4. Attribution Analysis ───────────────────────────────────────────
    print("\n" + "=" * 78)
    print("  [4] Attribution Analysis")
    print("=" * 78)

    a_cagr = cases_ae["A_CTRL_3M_LOT100"]["cagr"]
    b_cagr = cases_ae["B_LARGE_30M_LOT100"]["cagr"]
    c_cagr = cases_ae["C_HUGE_300M_LOT100"]["cagr"]
    d_cagr = cases_ae["D_FRAC_3M_LOT1"]["cagr"]
    e_cagr = cases_ae["E_LEGACY_300M_LOT1"]["cagr"]

    # Attribution components
    cap_effect   = b_cagr - a_cagr      # 資本増加（lot固定）の効果
    lot_effect   = d_cagr - a_cagr      # lot制約除去（資本固定）の効果
    combined_eff = e_cagr - a_cagr      # 両方除去時の最大効果

    # 元のGAPに対して何%説明できるか
    if abs(TOTAL_GAP_PP) > 0.001:
        pct_explained_cap  = cap_effect   / TOTAL_GAP_PP * 100
        pct_explained_lot  = lot_effect   / TOTAL_GAP_PP * 100
        pct_explained_comb = combined_eff / TOTAL_GAP_PP * 100
    else:
        pct_explained_cap  = 0.0
        pct_explained_lot  = 0.0
        pct_explained_comb = 0.0

    # Verdict
    if pct_explained_comb >= 50:
        verdict = "CAPITAL_LOT_DOMINANT"
    elif pct_explained_comb >= 10:
        verdict = "PARTIAL_ATTRIBUTION"
    else:
        verdict = "OTHER_ENGINE_CAUSE"

    print(f"\n  TOTAL GAP (original vs REPRO_A): {TOTAL_GAP_PP:.2f}pp")
    print(f"\n  Case A (baseline)  CAGR={a_cagr:+.3f}%  (target: {REPRO_A_CAGR}%)")
    print(f"  Case B (30M+lot100) CAGR={b_cagr:+.3f}%  ΔCapital={cap_effect:+.3f}pp"
          f"  ({pct_explained_cap:+.1f}% of gap)")
    print(f"  Case C (300M+lot100) CAGR={c_cagr:+.3f}%")
    print(f"  Case D (3M+lot1)    CAGR={d_cagr:+.3f}%  ΔLot={lot_effect:+.3f}pp"
          f"  ({pct_explained_lot:+.1f}% of gap)")
    print(f"  Case E (300M+lot1)  CAGR={e_cagr:+.3f}%  ΔCombined={combined_eff:+.3f}pp"
          f"  ({pct_explained_comb:+.1f}% of gap)")
    print(f"\n  Attribution verdict: {verdict}")

    # Capital Sweep table
    print(f"\n  Capital Sweep (lot=100):")
    print(f"  {'Capital':>12}  {'CAGR':>8}  {'ΔCAGR':>8}  {'Trades':>7}  {'LotRej':>7}")
    print("  " + "-" * 52)
    for sw in sweep_results:
        dcagr = sw["cagr"] - a_cagr
        print(f"  {sw['capital']:>12,}  {sw['cagr']:>+8.3f}%  {dcagr:>+7.3f}pp"
              f"  {sw['n_trades']:>7}  {sw['rejected_by_lot_count']:>7}")

    # Lot rejection detail
    a_lot_rej = cases_ae["A_CTRL_3M_LOT100"]["rejected_by_lot_count"]
    d_lot_rej = cases_ae["D_FRAC_3M_LOT1"]["rejected_by_lot_count"]
    a_fwd20   = cases_ae["A_CTRL_3M_LOT100"]["lot_fwd20d"]
    print(f"\n  Lot rejection analysis:")
    print(f"    Case A (lot=100): rejected={a_lot_rej}  fwd20d={a_fwd20}")
    print(f"    Case D (lot=1):   rejected={d_lot_rej}")

    # ── 5. 結果JSON保存 ──────────────────────────────────────────────────
    today_str = str(_date.today())
    out = {
        "study": "Study42",
        "date": today_str,
        "period": {"is_start": IS_START, "is_end": IS_END},
        "reference": {
            "original_cagr": ORIGINAL_CAGR,
            "repro_a_cagr":  REPRO_A_CAGR,
            "total_gap_pp":  TOTAL_GAP_PP,
        },
        "cases_AE": cases_ae,
        "capital_sweep": sweep_results,
        "attribution": {
            "cap_effect_pp":      round(cap_effect,   3),
            "lot_effect_pp":      round(lot_effect,   3),
            "combined_effect_pp": round(combined_eff, 3),
            "pct_explained_capital": round(pct_explained_cap,  1),
            "pct_explained_lot":     round(pct_explained_lot,  1),
            "pct_explained_combined": round(pct_explained_comb, 1),
            "verdict": verdict,
        },
    }

    out_dir = Path(RESULTS_DIR) / ".."  / "backtests"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"study42_capital_constraint_archaeology_202606_{today_str}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  結果保存: {out_path}")

    print("\n" + "=" * 78)
    print(f"  Study42 完了  Verdict: {verdict}")
    print(f"  Combined explained: {pct_explained_comb:.1f}%  "
          f"(capital: {pct_explained_cap:.1f}%  lot: {pct_explained_lot:.1f}%)")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
