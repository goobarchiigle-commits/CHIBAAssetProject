"""
study50_post_integrity_revalidation.py
Post-Integrity Revalidation — RSR42 + union fix ベースでの全機能再検証

前提:
  - tools/integrity_check.py が 25/25 PASS
  - composite_alpha_bt.py が union ベース (intersection BUG 修正済)

ケース:
  A_BASELINE  : S5 のみ (全機能 OFF)
  B_ATR_EXT   : S5 + ATR Extension (exit_policy="A")
  C_VOL_ADJ   : S5 + D_VOL_ADJ (max_positions_ts)
  D_EQ_SCALE  : S5 + D_EQ_SCALE addon (addon_policy="D")
  E_COMBINED  : S5 + 全機能

評価:
  - Full IS  2018-2024 (新ベースライン: 旧20.51%廃止)
  - True OOS 2025
  - WF 5-fold (OOS 2020/2021/2022/2023/2024)

採用条件:
  ΔCAGR > +1.0pp
  MaxDD 悪化 < 2pp
  WF 4/5 以上
  2025 OOS ≥ Baseline OOS
"""
from __future__ import annotations

import json
import sys
import subprocess
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

# ── 定数 ───────────────────────────────────────────────────────────────────────
CAPITAL             = 3_000_000
MIN_HOLD            = 3
DATA_END            = "2025-12-31"
IS_START            = "2018-01-01"
IS_END              = "2024-12-31"
OOS_START           = "2025-01-01"
OOS_END             = "2025-12-31"
VOL_CALM_THRESHOLD  = 0.008    # TOPIX 20d std < 0.8% → max_pos=4
ADDON_SIZE_FRAC     = 0.25
ADDON_ATR_MULT      = 1.0

ADOPT_DELTA_CAGR    = 1.0      # ΔCAGR 採用閾値 pp
ADOPT_MAX_DD_DETER  = 2.0      # MaxDD 悪化許容 pp
ADOPT_WF_MIN        = 4        # WF 最低 fold 数

# ── ケース定義 (name, exit_policy, use_vol_adj, addon_policy) ──────────────
CASES = [
    ("A_BASELINE", None,  False, None),
    ("B_ATR_EXT",  "A",   False, None),
    ("C_VOL_ADJ",  None,  True,  None),
    ("D_EQ_SCALE", None,  False, "D"),
    ("E_COMBINED", "A",   True,  "D"),
]


# ══════════════════════════════════════════════════════════════════════════════
# ヘルパー
# ══════════════════════════════════════════════════════════════════════════════

def check_integrity() -> bool:
    """IC 25/25 PASS を確認。"""
    r = subprocess.run(
        [sys.executable, str(ROOT / "tools/integrity_check.py")],
        capture_output=True, text=True, encoding="utf-8",
    )
    passed = r.returncode == 0
    for line in r.stdout.splitlines():
        if "RESULT" in line or "VERDICT" in line:
            print(f"  [IC] {line.strip()}")
    return passed


def build_vol_adj_ts(topix_close: pd.Series, union_dates: list) -> pd.Series:
    ret   = topix_close.pct_change()
    std   = ret.rolling(20, min_periods=10).std()
    idx   = pd.Index(union_dates)
    std_s = std.reindex(idx, method="ffill").fillna(std.median())
    mpts  = pd.Series(3, index=std_s.index, dtype=int)
    mpts[std_s < VOL_CALM_THRESHOLD] = 4
    return mpts


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


def run_one(
    ds: dict, sym_active_df, start: str, end: str,
    exit_policy, max_positions_ts, addon_policy,
) -> dict:
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
        max_positions_ts       = max_positions_ts,
        addon_policy           = addon_policy,
        addon_atr_mult         = ADDON_ATR_MULT,
        addon_stage2_mult      = 2.0,
        addon_max_per_pos      = 1,
        addon_size_frac        = ADDON_SIZE_FRAC,
    )


def extract(m: dict) -> dict:
    return {
        "cagr":       round(float(m.get("cagr",            0.0) or 0.0), 2),
        "sharpe":     round(float(m.get("sharpe",          0.0) or 0.0), 3),
        "max_dd":     round(float(m.get("max_dd",          0.0) or 0.0), 2),
        "calmar":     round(float(m.get("calmar",          0.0) or 0.0), 3),
        "n_trades":   int  (m.get("n_trades",    0) or 0),
        "addon_cnt":  int  (m.get("addon_count", 0) or 0),
        "avg_exp":    round(float(m.get("avg_exposure",    0.0) or 0.0), 1),
    }


def print_period_table(title: str, rows: dict[str, dict]) -> None:
    base = rows.get("A_BASELINE", {})
    sep  = "─" * 78
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)
    hdr = f"  {'Case':<14} {'CAGR%':>8} {'Sharpe':>7} {'MaxDD%':>8} {'Calmar':>7} {'Trades':>7} {'Exp%':>6} {'ΔCAGR':>8}"
    print(hdr); print(sep)
    for cn, m in rows.items():
        if not m:
            print(f"  {cn:<14}  (error)"); continue
        dc = f"{m['cagr'] - base['cagr']:+.2f}" if cn != "A_BASELINE" and base else "    —"
        print(f"  {cn:<14} {m['cagr']:>+8.2f} {m['sharpe']:>7.3f} {m['max_dd']:>8.2f}"
              f" {m['calmar']:>7.3f} {m['n_trades']:>7} {m['avg_exp']:>6.1f} {dc:>8}")


# ══════════════════════════════════════════════════════════════════════════════
# メイン
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 78)
    print("  Study50 — Post-Integrity Revalidation")
    print(f"  Date: {TODAY_STR}   Capital: ¥{CAPITAL:,}")
    print(f"  IS: {IS_START}~{IS_END}   OOS: {OOS_START}~{OOS_END}")
    print("=" * 78)

    # ── Step 0: Integrity Gate ─────────────────────────────────────────────────
    print("\n[Step 0] Integrity Check...")
    if not check_integrity():
        print("  ABORT: integrity check FAILED — 修正後に再実行してください。")
        sys.exit(1)
    print("  PASS: 25/25 — union fix 確認済み。")

    # ── Step 1: データセット ───────────────────────────────────────────────────
    print(f"\n[Step 1] データセット構築 (end={DATA_END})...")
    ds       = build_common_dataset(DATA_END)
    all_syms = list(ds["trade_syms"].keys())
    print(f"  {len(all_syms)} シンボル読み込み完了")

    # union dates for vol_adj_ts
    union_dates = sorted(set.union(*[
        set(ds["universe_raw"][s]["df"].index)
        for s in all_syms if s in ds["universe_raw"]
    ]))
    vol_adj_ts  = build_vol_adj_ts(ds["topix_close"], union_dates)
    calm_n      = int((vol_adj_ts == 4).sum())
    print(f"  union dates: {len(union_dates)} 日  calm days (max_pos=4): {calm_n} ({calm_n/len(union_dates)*100:.1f}%)")

    # ── Step 2: Full IS + True OOS ────────────────────────────────────────────
    print(f"\n[Step 2] Full IS ({IS_START}~{IS_END}) + True OOS ({OOS_START}~{OOS_END})")
    print(f"  5 ケース × 2 期間 = 10 バックテスト\n")

    active_is  = get_active(ds, all_syms, IS_START,  IS_END)
    active_oos = get_active(ds, all_syms, OOS_START, OOS_END)

    period_results: dict[str, dict[str, dict]] = {
        "FULL_IS":  {},
        "TRUE_OOS": {},
    }

    for cn, ep, use_va, ap in CASES:
        mpts = vol_adj_ts if use_va else None
        for period_name, s, e, act in [
            ("FULL_IS",  IS_START,  IS_END,  active_is),
            ("TRUE_OOS", OOS_START, OOS_END, active_oos),
        ]:
            tag = f"{cn:<14} [{period_name}]"
            print(f"  {tag}  ep={ep or 'None':<4} va={use_va}  addon={ap or 'None'}...", end=" ", flush=True)
            try:
                raw = run_one(ds, act, s, e, ep, mpts, ap)
                m   = extract(raw)
                period_results[period_name][cn] = m
                print(f"CAGR={m['cagr']:+.2f}%  MaxDD={m['max_dd']:.2f}%  Sh={m['sharpe']:.3f}")
            except Exception as err:
                print(f"ERROR: {err}")
                period_results[period_name][cn] = {}

    print_period_table("FULL IS (2018-2024) — NEW_BASELINE 固定",  period_results["FULL_IS"])
    print_period_table("TRUE OOS (2025)",                           period_results["TRUE_OOS"])

    new_baseline_is  = period_results["FULL_IS"].get("A_BASELINE",  {})
    new_baseline_oos = period_results["TRUE_OOS"].get("A_BASELINE", {})
    print(f"\n  >>> NEW_BASELINE IS  CAGR={new_baseline_is.get('cagr', '?'):+.2f}%  MaxDD={new_baseline_is.get('max_dd','?')}%")
    print(f"  >>> NEW_BASELINE OOS CAGR={new_baseline_oos.get('cagr','?'):+.2f}%")

    # ── Step 3: Walk-Forward (5-fold) ─────────────────────────────────────────
    print(f"\n[Step 3] Walk-Forward 5-fold")
    print(f"  5 ケース × 5 fold = 25 バックテスト\n")

    wf_results: dict[str, dict] = {}
    for cn, ep, use_va, ap in CASES:
        mpts = vol_adj_ts if use_va else None
        print(f"\n  ── {cn}  ep={ep or 'None'}  va={use_va}  addon={ap or 'None'} ──")
        seg_rows = []
        for seg in WF_SEGS:
            n        = seg["seg"]
            oos_s, oos_e = seg["oos"]
            act_seg  = get_active(ds, all_syms, oos_s, oos_e)
            print(f"    Seg{n} OOS {oos_s[:4]}  ", end="", flush=True)
            try:
                raw  = run_one(ds, act_seg, oos_s, oos_e, ep, mpts, ap)
                m    = extract(raw)
                wf_p = m["sharpe"] > 0
                seg_rows.append({
                    "seg": n, "oos_year": oos_s[:4],
                    "wf_pass": wf_p,
                    **m,
                })
                mark = "✓" if wf_p else "✗"
                print(f"CAGR={m['cagr']:+.2f}%  Sh={m['sharpe']:.3f}  DD={m['max_dd']:.1f}%  {mark}")
            except Exception as err:
                print(f"ERROR: {err}")
                seg_rows.append({"seg": n, "oos_year": oos_s[:4], "wf_pass": False})

        wf_cnt   = sum(1 for r in seg_rows if r.get("wf_pass"))
        cagrlist = [r["cagr"] for r in seg_rows if "cagr" in r]
        shlist   = [r["sharpe"] for r in seg_rows if "sharpe" in r]
        ddlist   = [r["max_dd"] for r in seg_rows if "max_dd" in r]
        callist  = [r["calmar"] for r in seg_rows if "calmar" in r]
        explist  = [r["avg_exp"] for r in seg_rows if "avg_exp" in r]

        seg3_cagr = next((r["cagr"] for r in seg_rows if r.get("oos_year") == "2022" and "cagr" in r), None)

        wf_results[cn] = {
            "wf_count":        wf_cnt,
            "avg_oos_cagr":    round(float(np.mean(cagrlist)), 2) if cagrlist else 0.0,
            "avg_oos_sharpe":  round(float(np.mean(shlist)),   3) if shlist   else 0.0,
            "avg_oos_dd":      round(float(np.mean(ddlist)),   2) if ddlist   else 0.0,
            "avg_oos_calmar":  round(float(np.mean(callist)),  3) if callist  else 0.0,
            "avg_oos_exp":     round(float(np.mean(explist)),  1) if explist  else 0.0,
            "seg3_2022_cagr":  round(seg3_cagr, 2) if seg3_cagr is not None else None,
            "segments":        seg_rows,
        }
        print(f"    WF: {wf_cnt}/5  avg_CAGR={wf_results[cn]['avg_oos_cagr']:+.2f}%"
              f"  avg_Sh={wf_results[cn]['avg_oos_sharpe']:.3f}"
              f"  Seg3(2022)={seg3_cagr:+.2f}%" if seg3_cagr is not None else "")

    # ── WF サマリーテーブル ────────────────────────────────────────────────────
    wf_base = wf_results.get("A_BASELINE", {})
    sep = "─" * 78
    print(f"\n{sep}")
    print("  Walk-Forward 5-fold サマリー")
    print(sep)
    print(f"  {'Case':<14} {'WF':>4} {'avgCAGR':>9} {'avgSh':>7} {'avgDD':>7} {'Seg3_22':>9} {'ΔCAGR_wf':>10}")
    print(sep)
    for cn, ep, use_va, ap in CASES:
        w = wf_results.get(cn, {})
        if not w:
            print(f"  {cn:<14}  (no data)"); continue
        dc_wf = f"{w['avg_oos_cagr'] - wf_base['avg_oos_cagr']:+.2f}" if cn != "A_BASELINE" and wf_base else "    —"
        s3    = f"{w['seg3_2022_cagr']:+.2f}%" if w.get("seg3_2022_cagr") is not None else "    —"
        print(f"  {cn:<14} {w['wf_count']:>2}/5 {w['avg_oos_cagr']:>+9.2f}%"
              f" {w['avg_oos_sharpe']:>7.3f} {w['avg_oos_dd']:>7.2f}%"
              f" {s3:>9} {dc_wf:>10}")

    # ── Step 4: 採用判定 ────────────────────────────────────────────────────────
    print(f"\n[Step 4] 採用判定  (閾値: ΔCAGR>{ADOPT_DELTA_CAGR}pp / MaxDD悪化<{ADOPT_MAX_DD_DETER}pp / WF≥{ADOPT_WF_MIN}/5 / OOS≥Baseline OOS)")
    sep = "─" * 78
    print(f"\n{sep}")
    print(f"  {'Feature':<14} {'ΔCAGR_IS':>10} {'ΔMaxDD':>8} {'WF':>5} {'OOS_CAGR':>10} {'OOS_vs_base':>12} {'判定':>6}")
    print(sep)

    adoption: dict[str, dict] = {}
    features = [cn for cn, *_ in CASES if cn != "A_BASELINE"]

    for cn in features:
        m_is  = period_results["FULL_IS"].get(cn, {})
        m_oos = period_results["TRUE_OOS"].get(cn, {})
        wf    = wf_results.get(cn, {})

        if not m_is or not new_baseline_is:
            print(f"  {cn:<14}  (data missing)")
            adoption[cn] = {"verdict": "ERROR"}
            continue

        d_cagr    = round(m_is["cagr"]    - new_baseline_is["cagr"],    2)
        d_dd      = round(m_is["max_dd"]  - new_baseline_is["max_dd"],  2)
        wf_cnt    = wf.get("wf_count", 0)
        oos_cagr  = m_oos.get("cagr", 0.0) if m_oos else 0.0
        base_oos  = new_baseline_oos.get("cagr", 0.0) if new_baseline_oos else 0.0
        d_oos     = round(oos_cagr - base_oos, 2) if new_baseline_oos else None

        ok_cagr  = d_cagr    >  ADOPT_DELTA_CAGR
        ok_dd    = d_dd      > -ADOPT_MAX_DD_DETER
        ok_wf    = wf_cnt    >= ADOPT_WF_MIN
        ok_oos   = (d_oos is None) or (d_oos >= 0.0)

        verdict  = "PASS" if (ok_cagr and ok_dd and ok_wf and ok_oos) else "FAIL"
        fail_why = []
        if not ok_cagr: fail_why.append(f"ΔCAGR={d_cagr:+.2f}<{ADOPT_DELTA_CAGR}")
        if not ok_dd:   fail_why.append(f"ΔDD={d_dd:+.2f}<-{ADOPT_MAX_DD_DETER}")
        if not ok_wf:   fail_why.append(f"WF={wf_cnt}<{ADOPT_WF_MIN}")
        if not ok_oos:  fail_why.append(f"OOS={d_oos:+.2f}<0")

        adoption[cn] = {
            "verdict": verdict, "delta_cagr_is": d_cagr, "delta_dd": d_dd,
            "wf_count": wf_cnt, "oos_cagr": oos_cagr, "delta_oos": d_oos,
            "fail_reasons": fail_why,
        }
        d_oos_str = f"{d_oos:+.2f}" if d_oos is not None else "    —"
        print(f"  {cn:<14} {d_cagr:>+10.2f} {d_dd:>+8.2f} {wf_cnt:>3}/5"
              f" {oos_cagr:>+10.2f}% {d_oos_str:>12}   {verdict}"
              + (f"  [{', '.join(fail_why)}]" if fail_why else ""))

    # ── Step 5: Production Baseline 確定 ───────────────────────────────────────
    print(f"\n[Step 5] Final Production Baseline 確定")
    adopted_features = [cn for cn in features if adoption.get(cn, {}).get("verdict") == "PASS"]

    # E_COMBINED が PASS なら E_COMBINED を採用
    if "E_COMBINED" in adopted_features:
        final_case     = "E_COMBINED"
    elif adopted_features:
        final_case     = adopted_features[-1]  # 最後に採用されたケース
    else:
        final_case     = "A_BASELINE"
        adopted_features = []

    fm_is    = period_results["FULL_IS"].get(final_case, {})
    fm_oos   = period_results["TRUE_OOS"].get(final_case, {})
    fm_wf    = wf_results.get(final_case, {})

    print(f"\n  採用機能: {adopted_features if adopted_features else ['なし — BASELINE を維持']}")
    print(f"  最終ケース: {final_case}")
    sep = "─" * 55
    print(f"\n  FINAL_PRODUCTION_BASELINE ({final_case})")
    print(f"  {sep}")
    print(f"  {'項目':<20} {'IS 2018-2024':>14} {'OOS 2025':>12}")
    print(f"  {sep}")
    for label, key in [("CAGR", "cagr"), ("MaxDD", "max_dd"), ("Sharpe", "sharpe"), ("Calmar", "calmar")]:
        v_is  = fm_is.get(key,  "?")
        v_oos = fm_oos.get(key, "?")
        print(f"  {label:<20} {str(v_is):>14} {str(v_oos):>12}")
    print(f"  {'Trades':<20} {str(fm_is.get('n_trades','?')):>14} {str(fm_oos.get('n_trades','?')):>12}")
    print(f"  {'AvgExposure':<20} {str(fm_is.get('avg_exp','?')):>14} {str(fm_oos.get('avg_exp','?')):>12}")
    print(f"  {'WF (5-fold)':<20} {str(fm_wf.get('wf_count','?'))+'/5':>14}")
    print(f"  {'WF avg CAGR':<20} {str(fm_wf.get('avg_oos_cagr','?')):>14}")
    print(f"  {'WF avg MaxDD':<20} {str(fm_wf.get('avg_oos_dd','?')):>14}")
    print(f"  {sep}")
    print(f"\n  旧 CAGR 20.51% は廃止 (4055.T 期間圧縮の産物)")
    print(f"  新 Production IS CAGR: {fm_is.get('cagr','?')}%  (2018-2024, 6.84yr, union fix)")

    # ── 保存 ───────────────────────────────────────────────────────────────────
    out_path = ROOT / "backtests" / f"study50_post_integrity_revalidation_{TODAY_STR}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "study":            "Study50_PostIntegrityRevalidation",
        "date":             TODAY_STR,
        "integrity_check":  "25/25 PASS",
        "fix":              "intersection→union (2026-06-28)",
        "period_is":        f"{IS_START}~{IS_END}",
        "period_oos":       f"{OOS_START}~{OOS_END}",
        "new_baseline_is":  new_baseline_is,
        "new_baseline_oos": new_baseline_oos,
        "period_results":   period_results,
        "wf_results":       wf_results,
        "adoption":         adoption,
        "adopted_features": adopted_features,
        "final_case":       final_case,
        "final_is":         fm_is,
        "final_oos":        fm_oos,
        "final_wf":         fm_wf,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  結果保存: {out_path}")
    print("\n" + "=" * 78)
    print("  Study50 COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()
