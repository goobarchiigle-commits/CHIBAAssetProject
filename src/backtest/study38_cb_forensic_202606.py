"""
backtest/study38_cb_forensic_202606.py

Study38 — Backtest Engine Forensics (Final)
目的: APRIL_REPRO_A(20.1%/0.859/-15.7%) vs 原本(22.4%/1.582/-12.32%)
     の残差 CAGR-2.26pp / Sharpe-0.723 / MaxDD-3.39pp を
     バックテストエンジン差分（主にCBサブシステム）で定量説明する。

禁止: パラメータ探索・最適化・新規戦略提案
対象: composite_alpha_bt.py の CB サブシステム

Phase 1: Engine Diff Audit（コード解析のみ、実行結果確認用基準点として Case A を実行）
Phase 2: CB Forensic
  Case A: CB ON（現行：DD>15%でCB_SCALE=0.35、30日タイムアウトで解除）
  Case B: CB 完全除去（bypass_cb=True: cb_active=Falseに固定、スケーリングなし）
"""
from __future__ import annotations

import sys
import json
import warnings
import dataclasses
from pathlib import Path
from datetime import date

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

from src.backtest import composite_alpha_bt as cab
from src.backtest.snapshot_archaeology_202606 import build_common_dataset
from src.paths import RESULTS_DIR

# ── 定数 ────────────────────────────────────────────────────────────
IS_START = "2018-01-01"
IS_END   = "2024-12-31"
CAPITAL  = 3_000_000

# 記録値（原本 min_hold_sensitivity_2026-03-31.json hold3d.IS）
RECORDED = {"cagr": 22.4, "sharpe": 1.582, "max_dd": -12.32, "n_trades": 219}

# APRIL_REPRO_A 確定値（study36/37 で確認済み）
# CB ON で 20.1% / 0.859 / -15.7% / 216 trades
RESIDUAL = {
    "cagr":    RECORDED["cagr"]   - 20.1,    # +2.3pp（原本 > A）
    "sharpe":  RECORDED["sharpe"] - 0.859,   # +0.723
    "max_dd":  RECORDED["max_dd"] - (-15.7), # -3.38pp（原本のMaxDD が浅い）
}


def run_april_repro(ds: dict, bypass_cb: bool = False) -> dict:
    """APRIL_REPRO_A 設定で run_scenario を実行する。"""
    cfg = ds["base_cfg"]
    return cab.run_scenario(
        scenario             = "BASELINE",
        universe_raw         = ds["universe_raw"],
        rsr_df               = ds["rsr_df"],
        alpha_df             = None,
        regime_df            = ds["regime_df"],
        trade_syms           = ds["trade_syms"],
        rsr_syms             = ds["rsr_syms"],
        cfg                  = cfg,
        start                = IS_START,
        end                  = IS_END,
        verbose              = False,
        tech_matrices        = ds["tech_matrices"],
        breadth_series       = ds["breadth_series"],
        capital              = CAPITAL,
        min_hold             = 3,
        market_shock_mode    = "full_exit",
        rsr_exit_threshold   = 75.0,
        sym_active_df        = None,
        enable_simple_rsr_exit       = True,
        enable_atr_trailing_prod     = False,
        enable_multilayer_rsr        = False,
        enable_atr_risk_sizing       = False,
        enable_mtf_filter            = False,
        use_fixed_pct_trail          = False,
        sizing_mode                  = "existing",
        # topix_close は渡さない（APRIL_REPRO_A は None）
        bypass_cb            = bypass_cb,
    )


def extract_key_metrics(res: dict) -> dict:
    return {
        "cagr":               res.get("cagr", 0.0),
        "sharpe":             res.get("sharpe", 0.0),
        "max_dd":             res.get("max_dd", 0.0),
        "calmar":             res.get("calmar", 0.0),
        "n_trades":           res.get("n_trades", 0),
        "avg_hold_days":      res.get("avg_hold_days", 0.0),
        "avg_exposure":       res.get("avg_exposure", 0.0),
        "win_rate":           res.get("win_rate", 0.0),
        "avg_win_pct":        res.get("avg_win_pct", 0.0),
        "avg_lose_pct":       res.get("avg_lose_pct", 0.0),
        "annual_returns":     res.get("annual_returns", {}),
        "exit_reason_counts": res.get("exit_reason_counts", {}),
        # CB 計測
        "cb_trigger_count":   res.get("cb_trigger_count", 0),
        "cb_active_days":     res.get("cb_active_days", 0),
        "cb_scaled_entries":  res.get("cb_scaled_entries", 0),
        "capital_suppressed": res.get("capital_suppressed", 0),
    }


def main() -> int:
    print("=" * 70)
    print("  Study38 — Backtest Engine Forensics: CB Subsystem")
    print(f"  IS={IS_START}..{IS_END}  capital={CAPITAL:,}")
    print(f"  TARGET: CAGR={RECORDED['cagr']}%  Sharpe={RECORDED['sharpe']}  MaxDD={RECORDED['max_dd']}%")
    print("=" * 70 + "\n")

    # ---- 共通データセット ----
    print("[1/3] データセット構築中（RSR42 + 全期間プリコンピュート）...")
    ds = build_common_dataset(IS_END)
    print("  完了\n")

    # ---- Phase 2: Case A（CB ON）----
    print("[2/3] Case A: CB ON（現行: CB_SCALE=0.35, 30日タイムアウト）...")
    res_a = run_april_repro(ds, bypass_cb=False)
    ma = extract_key_metrics(res_a)
    print(f"  CAGR={ma['cagr']:+.2f}%  Sharpe={ma['sharpe']:.3f}  MaxDD={ma['max_dd']:.2f}%  Trades={ma['n_trades']}")
    print(f"  CB: trigger={ma['cb_trigger_count']}  active_days={ma['cb_active_days']}  "
          f"scaled_entries={ma['cb_scaled_entries']}  suppressed=¥{ma['capital_suppressed']:,.0f}")

    # ---- Phase 2: Case B（CB 完全除去）----
    print("\n[3/3] Case B: CB 完全除去（bypass_cb=True、常にcb_scale=1.0）...")
    res_b = run_april_repro(ds, bypass_cb=True)
    mb = extract_key_metrics(res_b)
    print(f"  CAGR={mb['cagr']:+.2f}%  Sharpe={mb['sharpe']:.3f}  MaxDD={mb['max_dd']:.2f}%  Trades={mb['n_trades']}")
    print(f"  CB: trigger={mb['cb_trigger_count']}  active_days={mb['cb_active_days']}  "
          f"scaled_entries={mb['cb_scaled_entries']}  suppressed=¥{mb['capital_suppressed']:,.0f}")

    # ---- Phase 2: CB 差分（B - A）----
    d_cagr   = mb["cagr"]    - ma["cagr"]
    d_sharpe = mb["sharpe"]  - ma["sharpe"]
    d_dd     = mb["max_dd"]  - ma["max_dd"]

    print("\n── CB 差分（Case B − Case A）─────────────────────────────────────")
    print(f"  ΔCAGR   = {d_cagr:+.3f}pp")
    print(f"  ΔSharpe = {d_sharpe:+.3f}")
    print(f"  ΔMaxDD  = {d_dd:+.3f}pp")

    # ---- Phase 3: 説明力計算 ----
    total_gap_cagr   = RECORDED["cagr"]   - ma["cagr"]    # 正: 原本 > A
    total_gap_sharpe = RECORDED["sharpe"] - ma["sharpe"]
    total_gap_dd     = RECORDED["max_dd"] - ma["max_dd"]  # 負: 原本のMaxDD が浅い

    explained_cagr_pct   = (d_cagr   / total_gap_cagr   * 100) if total_gap_cagr   != 0 else 0.0
    explained_sharpe_pct = (d_sharpe / total_gap_sharpe * 100) if total_gap_sharpe != 0 else 0.0
    explained_dd_pct     = (-d_dd    / -total_gap_dd    * 100) if total_gap_dd     != 0 else 0.0  # 絶対値で比較

    print("\n── Phase 3: 説明力（CB改善で残差をどれだけ説明できるか）─────────")
    print(f"  総残差 CAGR:   {total_gap_cagr:+.3f}pp  → CB説明 {d_cagr:+.3f}pp  ({explained_cagr_pct:.1f}%)")
    print(f"  総残差 Sharpe: {total_gap_sharpe:+.3f}    → CB説明 {d_sharpe:+.3f}    ({explained_sharpe_pct:.1f}%)")
    print(f"  総残差 MaxDD:  {total_gap_dd:+.3f}pp  → CB説明 {d_dd:+.3f}pp   ({explained_dd_pct:.1f}%)")

    # ---- 年次リターン比較 ----
    print("\n── 年次リターン（A=CB ON, B=CB OFF, Orig=原本）────────────────────")
    orig_annual = {"2020": 15.2, "2021": 36.81, "2022": 10.81, "2023": 19.63, "2024": 12.77}
    all_years = sorted(set(list(ma["annual_returns"].keys()) + list(orig_annual.keys())))
    print(f"  {'年':6}  {'Orig':>8}  {'A(CBON)':>8}  {'B(CBOFF)':>9}  {'B-A':>7}")
    for yr in all_years:
        vo = orig_annual.get(yr, float("nan"))
        va = ma["annual_returns"].get(yr, float("nan"))
        vb = mb["annual_returns"].get(yr, float("nan"))
        diff_ba = (vb - va) if (not np.isnan(vb) and not np.isnan(va)) else float("nan")
        print(f"  {yr:6}  {vo:>+7.2f}%  {va:>+7.2f}%  {vb:>+8.2f}%  {diff_ba:>+6.2f}pp")

    # ---- 最終判定 ----
    print("\n── 最終判定 ─────────────────────────────────────────────────────")
    cb_is_primary = explained_cagr_pct >= 50.0

    if cb_is_primary:
        verdict_a = "CB主因確定（説明率>=50%）"
    else:
        verdict_a = f"CB単独説棄却（説明率={explained_cagr_pct:.1f}%<50%）"

    print(f"  A: {verdict_a}")

    # Sharpe残差の最大原因
    if explained_sharpe_pct >= 50.0:
        verdict_b = f"CB（説明率={explained_sharpe_pct:.1f}%）"
    elif explained_sharpe_pct >= 20.0:
        verdict_b = f"CB部分寄与（{explained_sharpe_pct:.1f}%）+ 再現不能領域（data vintage / code構造変化）"
    else:
        verdict_b = f"再現不能領域（data vintage / code構造変化）が支配的（CB説明率={explained_sharpe_pct:.1f}%のみ）"
    print(f"  B: Sharpe残差の最大原因 = {verdict_b}")

    print(f"  C: 22.4/1.582再現不能の残余要因（残差CAGR{total_gap_cagr - d_cagr:+.2f}pp / Sharpe{total_gap_sharpe - d_sharpe:+.3f}）")
    print(f"     → data vintage差分（yfinance遡及調整）/ code構造変化（cfg化、新引数追加）")

    # 継続考古学価値
    remaining_gap_sharpe = total_gap_sharpe - d_sharpe
    if remaining_gap_sharpe > 0.3:
        value = "LOW（Sharpe残差 > 0.3 が残存するが原理的に再現不能領域）"
    elif remaining_gap_sharpe > 0.1:
        value = "MEDIUM（一部残差は他要因に帰属できる可能性あり）"
    else:
        value = "HIGH（ほぼ全て説明済み）"
    print(f"  D: 継続考古学価値 = {value}")

    # ---- 保存 ────────────────────────────────────────────────────────
    output = {
        "study": "Study38_CB_Forensic",
        "date": date.today().isoformat(),
        "recorded": RECORDED,
        "total_residual": {
            "cagr":   round(total_gap_cagr,   3),
            "sharpe": round(total_gap_sharpe, 3),
            "max_dd": round(total_gap_dd,     3),
        },
        "case_A_cb_on":  {k: v for k, v in ma.items() if k != "annual_returns"},
        "case_B_cb_off": {k: v for k, v in mb.items() if k != "annual_returns"},
        "annual_returns": {
            "original":  orig_annual,
            "A_cb_on":   ma["annual_returns"],
            "B_cb_off":  mb["annual_returns"],
        },
        "cb_effect": {
            "delta_cagr":   round(d_cagr,   3),
            "delta_sharpe": round(d_sharpe, 3),
            "delta_max_dd": round(d_dd,     3),
        },
        "explanation_rate_pct": {
            "cagr":   round(explained_cagr_pct,   1),
            "sharpe": round(explained_sharpe_pct, 1),
            "max_dd": round(explained_dd_pct,     1),
        },
        "verdict": {
            "A_cb_primary":     bool(cb_is_primary),
            "B_sharpe_cause":   verdict_b,
            "C_remaining_gap":  {
                "cagr":   round(total_gap_cagr   - d_cagr,   3),
                "sharpe": round(total_gap_sharpe - d_sharpe, 3),
            },
            "D_archaeology_value": value,
        },
    }

    save_path = RESULTS_DIR / f"study38_cb_forensic_202606_{date.today().isoformat()}.json"
    save_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n結果保存: {save_path}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
