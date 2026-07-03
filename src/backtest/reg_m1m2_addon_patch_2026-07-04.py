"""
reg_m1m2_addon_patch_2026-07-04.py
Stage1 M1/M2 PATCH後 REG（回帰確認）

目的:
  M1: addon執行価格 close→open PATCH
  M2: max_single_weight×1.5バイパス撤廃（0.25厳格化）
  の2件を composite_alpha_bt.py に適用後、D_ATR_EQ(CURRENT: rsr_exit=70, addon_policy=D)を
  fresh runし、PATCH前の基準値（Study73 2026-07-02実測: IS=12.37% OOS=13.48% WF=4/5 avg=18.37% 2022=-2.65%）
  と比較する。

判定基準（roadmap M1/M2）: |ΔCAGR|>0.5pp または 2022悪化 なら停止しユーザー報告。

禁止: パラメータ変更・新規特徴量・Production変更。観測専用（PATCH効果測定のみ）。
"""
from __future__ import annotations

import json
import sys
import warnings
from datetime import date
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

from src.backtest.snapshot_archaeology_202606 import build_common_dataset
import src.backtest.composite_alpha_bt as cab

TODAY_STR = date.today().strftime("%Y-%m-%d")
OUT_FILE  = ROOT / "backtests" / f"reg_m1m2_addon_patch_{TODAY_STR}.json"

CAPITAL   = 3_000_000
MIN_HOLD  = 3

IS_START, IS_END   = "2018-01-01", "2024-12-31"
OOS_START, OOS_END = "2025-01-01", "2025-12-31"
DATA_END = "2025-12-31"

WF_SEGS = [
    {"seg": 1, "oos_s": "2020-01-01", "oos_e": "2020-12-31", "year": "2020"},
    {"seg": 2, "oos_s": "2021-01-01", "oos_e": "2021-12-31", "year": "2021"},
    {"seg": 3, "oos_s": "2022-01-01", "oos_e": "2022-12-31", "year": "2022"},
    {"seg": 4, "oos_s": "2023-01-01", "oos_e": "2023-12-31", "year": "2023"},
    {"seg": 5, "oos_s": "2024-01-01", "oos_e": "2024-12-31", "year": "2024"},
]

# CURRENT (D_ATR_EQ) config — Study73と同一定義
CFG = {"exit_policy": "A", "addon_policy": "D", "rsr_exit": 70.0}

# PATCH前の基準値（Study73 2026-07-02, backtests/study73_production_migration_audit_2026-07-02.json）
BASELINE = {
    "is_cagr": 12.37, "oos_cagr": 13.48,
    "wf_avg_cagr": 18.37, "wf_pass": 4,
    "seg2022_cagr": -2.65,
}


def safe_float(v, default=0.0):
    try:
        f = float(v); return f if not np.isnan(f) else default
    except (TypeError, ValueError): return default


def extract_metrics(raw: dict) -> dict:
    if raw is None: return {}
    return {
        "cagr":      round(safe_float(raw.get("cagr")), 2),
        "sharpe":    round(safe_float(raw.get("sharpe")), 3),
        "max_dd":    round(safe_float(raw.get("max_dd")), 2),
        "calmar":    round(safe_float(raw.get("calmar")), 3),
        "n_trades":  int(raw.get("n_trades", 0) or 0),
        "addon_cnt": int(raw.get("addon_cnt", 0) or 0),
    }


def get_active(ds, all_syms, start, end):
    from src.config_loader import load_strategy_config
    cfg = load_strategy_config()
    bc  = cfg.risk_controls.bear_universe_filter
    be  = list(bc.excluded_sectors) if bc.enabled else None
    return cab.build_dyn_rsr42_active(
        universe_raw=ds["universe_raw"], topix_close=ds["topix_close"],
        rsr_df=ds["rsr_df"], all_syms=all_syms, start=start, end=end,
        bear_exclude_sectors=be,
        sym_sector_map=dict(ds["trade_syms"]) if be else None,
    )


def run_bt(ds, sym_active_df, start, end) -> dict:
    return cab.run_scenario(
        scenario               = "BASELINE",
        universe_raw           = ds["universe_raw"],
        rsr_df                 = ds["rsr_df"],
        alpha_df               = None,
        regime_df              = ds["regime_df"],
        trade_syms             = ds["trade_syms"],
        rsr_syms                = ds["rsr_syms"],
        cfg                    = ds["base_cfg"],
        start                  = start, end=end, verbose=False,
        tech_matrices          = ds["tech_matrices"],
        breadth_series         = ds["breadth_series"],
        capital                = CAPITAL,
        min_hold               = MIN_HOLD,
        topix_close            = ds["topix_close"],
        market_shock_mode      = "composite",
        rsr_exit_threshold     = CFG["rsr_exit"],
        sym_active_df          = sym_active_df,
        enable_simple_rsr_exit = True,
        enable_atr_trailing_prod = True,
        enable_multilayer_rsr  = True,
        enable_atr_risk_sizing = False,
        enable_mtf_filter      = False,
        sizing_mode            = "existing",
        exit_policy            = CFG["exit_policy"],
        addon_policy           = CFG["addon_policy"],
        addon_size_frac        = 0.25,
        addon_atr_mult         = 1.0,
    )


def main():
    print("=" * 80)
    print("  REG M1/M2 — Addon執行価格PATCH + max_single_weight×1.5バイパス撤廃")
    print(f"  Date: {TODAY_STR}   Capital: Y{CAPITAL:,}")
    print("=" * 80)

    ds = build_common_dataset(DATA_END)
    all_syms = list(ds["trade_syms"].keys())
    print(f"  {len(all_syms)} symbols")

    print("\n[RUN] IS 2018-2024...")
    act_is = get_active(ds, all_syms, IS_START, IS_END)
    m_is = extract_metrics(run_bt(ds, act_is, IS_START, IS_END))
    print(f"  CAGR={m_is['cagr']:+.2f}%  Sharpe={m_is['sharpe']:.3f}  MaxDD={m_is['max_dd']:.2f}%  Addon={m_is['addon_cnt']}")

    print("\n[RUN] OOS 2025...")
    act_oos = get_active(ds, all_syms, OOS_START, OOS_END)
    m_oos = extract_metrics(run_bt(ds, act_oos, OOS_START, OOS_END))
    print(f"  CAGR={m_oos['cagr']:+.2f}%  Sharpe={m_oos['sharpe']:.3f}  MaxDD={m_oos['max_dd']:.2f}%")

    print("\n[RUN] WF 5fold (2020-2024)...")
    seg_rows, cagrs, pass_cnt = [], [], 0
    for fold in WF_SEGS:
        act = get_active(ds, all_syms, fold["oos_s"], fold["oos_e"])
        m = extract_metrics(run_bt(ds, act, fold["oos_s"], fold["oos_e"]))
        wf_pass = m["cagr"] > 0
        if wf_pass: pass_cnt += 1
        cagrs.append(m["cagr"])
        seg_rows.append({**m, "year": fold["year"], "wf_pass": wf_pass})
        print(f"  {fold['year']}: CAGR={m['cagr']:+.2f}%  {'✓' if wf_pass else '✗'}")

    wf_avg = round(float(np.mean(cagrs)), 2)
    seg2022 = next(s for s in seg_rows if s["year"] == "2022")

    print("\n" + "─" * 80)
    print("  PATCH前(Study73 2026-07-02) vs PATCH後(本REG) 比較")
    print("─" * 80)
    d_is    = round(m_is["cagr"] - BASELINE["is_cagr"], 2)
    d_oos   = round(m_oos["cagr"] - BASELINE["oos_cagr"], 2)
    d_wfavg = round(wf_avg - BASELINE["wf_avg_cagr"], 2)
    d_2022  = round(seg2022["cagr"] - BASELINE["seg2022_cagr"], 2)

    print(f"  IS CAGR:    {BASELINE['is_cagr']:+.2f}% -> {m_is['cagr']:+.2f}%   Delta={d_is:+.2f}pp")
    print(f"  OOS CAGR:   {BASELINE['oos_cagr']:+.2f}% -> {m_oos['cagr']:+.2f}%   Delta={d_oos:+.2f}pp")
    print(f"  WF avg:     {BASELINE['wf_avg_cagr']:+.2f}% -> {wf_avg:+.2f}%   Delta={d_wfavg:+.2f}pp")
    print(f"  WF pass:    {BASELINE['wf_pass']}/5 -> {pass_cnt}/5")
    print(f"  2022 CAGR:  {BASELINE['seg2022_cagr']:+.2f}% -> {seg2022['cagr']:+.2f}%   Delta={d_2022:+.2f}pp")

    max_abs_delta = max(abs(d_is), abs(d_oos))
    stop_flag = (max_abs_delta > 0.5) or (d_2022 < 0)
    print(f"\n  判定: max|Delta(IS,OOS)|={max_abs_delta:.2f}pp (閾値0.5pp) / 2022悪化={d_2022 < 0}")
    print(f"  {'STOP - ユーザー報告必要' if stop_flag else 'PASS - 推定通り(<=0.3pp目安)、採用可'}")

    output = {
        "reg": "M1_M2_addon_patch", "date": TODAY_STR,
        "patch": ["M1: addon_px close->open (composite_alpha_bt.py L1666付近)",
                  "M2: max_single_weight x1.5 bypass removed (composite_alpha_bt.py L1685付近)"],
        "baseline_pre_patch": BASELINE,
        "post_patch": {
            "is": m_is, "oos": m_oos, "wf_avg_cagr": wf_avg, "wf_pass": pass_cnt,
            "wf_segments": seg_rows,
        },
        "delta": {"is_cagr": d_is, "oos_cagr": d_oos, "wf_avg": d_wfavg, "2022_cagr": d_2022},
        "stop_flag": stop_flag,
    }
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n[OUTPUT] {OUT_FILE}")


if __name__ == "__main__":
    main()
