"""
src/scripts/study76d_contamination_ablation.py
Study76D — Dynamic RSR42 ffill contamination ablation.

目的: monthly rolling RSRのffill contamination（study76_dynamic_rsr42_pathology_diagnostics.md
で確定・実測99%汚染率）がRunB結果へ与えた影響を定量化する。

修正（本スクリプト内のみ・既存ファイルは無改変）: build_monthly_rolling_rsr()の非在籍月NaNを
0埋めしたバリアント build_monthly_rolling_rsr_zerofilled() を新設。FujikoStrategy側の
fill_method="ffill"はNaNが存在しなければ実質no-opになるため、この0埋めだけでffillの実害を断てる
（エンジン本体・fujiko_strategy.pyは一切変更しない）。

実施:
  1. RunB（contaminated baseline）をfresh run再現
  2. RunB_fixed（0埋め版）をfresh run
  3. Delta_bug = RunB_fixed - RunB を8指標×IS/OOSで算出
  4. 月次汚染率（非在籍銘柄がRSR>0として参照された件数）を集計

判定: |Delta_bug(IS CAGR)| が大きい→Study76結果無効化・Dynamic Universe再評価要。
      小さい→バグではなくDynamic Universe自体の特性。
"""
from __future__ import annotations

import sys
import warnings

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import json
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

from src.backtest.study75b_survivorship_bias import IS_END, IS_START, OOS_END, OOS_START
from src.backtest.study76_datr_eq_universe_c_rebaseline import capacity_diagnostics
from src.scripts.study76_dynamic_rsr42_pathology_diagnostics import rebuild_run_b_inputs, run_full_raw
from src.paths import RESULTS_DIR, REPORTS_DIR

_JST = timezone(timedelta(hours=9))


def build_monthly_rolling_rsr_zerofilled(rolling_rsr_contaminated: pd.DataFrame) -> pd.DataFrame:
    """
    既存の(NaNギャップを持つ)月次ローリングRSRを受け取り、非在籍期間を明示的に0で埋める。
    FujikoStrategy._slice_series_to_array(fill_method="ffill")はNaNが無ければ何もしないため、
    この変換だけでffill汚染を遮断できる（fujiko_strategy.py/composite_alpha_bt.pyは無改変）。
    """
    return rolling_rsr_contaminated.fillna(0.0)


def extract_metrics(raw: dict) -> dict:
    return {
        "cagr": raw.get("cagr"), "sharpe": raw.get("sharpe"), "calmar": raw.get("calmar"),
        "max_dd": raw.get("max_dd"), "n_trades": raw.get("n_trades"),
        "avg_exposure": raw.get("avg_exposure"),
        "avg_candidates": raw.get("avg_candidates"),
        "avg_simultaneous_holdings": raw.get("avg_simultaneous_holdings"),
    }


def compute_delta(fixed: dict, contaminated: dict) -> dict:
    delta = {}
    for k in fixed:
        a, b = fixed.get(k), contaminated.get(k)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            delta[k] = round(a - b, 4)
        else:
            delta[k] = None
    return delta


def compute_monthly_contamination(
    rolling_rsr_contaminated: pd.DataFrame,
    dynamic_membership: dict[str, list[str]],
) -> list[dict]:
    """
    月ごとに、その月「非在籍」の銘柄がffill()適用後に RSR>0 として参照されてしまう件数を集計する。
    （FujikoStrategyが実際に見る値を再現するため、ここで初めてffill()を適用してシミュレートする。）
    """
    ffilled = rolling_rsr_contaminated.ffill()
    months = sorted(dynamic_membership.keys())
    all_syms = set(rolling_rsr_contaminated.columns)
    rows = []
    for i, key in enumerate(months):
        month_start = pd.Timestamp(key)
        month_end = pd.Timestamp(months[i + 1]) if i + 1 < len(months) else ffilled.index.max() + pd.Timedelta(days=1)
        window = ffilled.loc[(ffilled.index >= month_start) & (ffilled.index < month_end)]
        if window.empty:
            continue
        members = set(dynamic_membership[key])
        non_members = all_syms - members
        if not non_members:
            continue
        non_member_cols = [c for c in non_members if c in window.columns]
        sub = window[non_member_cols]
        contaminated_mask = sub > 0.0  # ffillで漏れ込んだ非ゼロ値（0埋め前提なら本来ここは全てNaN/0のはず）
        n_contaminated_symbols = int(contaminated_mask.any(axis=0).sum())
        n_contaminated_symbol_days = int(contaminated_mask.sum().sum())
        rows.append({
            "month": key,
            "n_non_member_symbols": len(non_member_cols),
            "n_non_member_symbols_referenced_rsr_gt_0": n_contaminated_symbols,
            "contamination_rate_symbols_pct": round(100 * n_contaminated_symbols / max(1, len(non_member_cols)), 2),
            "n_contaminated_symbol_days": n_contaminated_symbol_days,
        })
    return rows


def judge(delta_is_cagr: float, delta_oos_cagr: float) -> dict:
    """
    しきい値: |Delta_bug CAGR| >= 5pp を「大きい」の目安とする（Study76本体のΔ_dynamicが
    25.17pp/62.27ppという規模であることに対する相対評価・恣意的な機械判定ゲートではなく
    解釈の出発点として明示する）。
    """
    threshold_pp = 5.0
    is_large = abs(delta_is_cagr) >= threshold_pp
    oos_large = abs(delta_oos_cagr) >= threshold_pp
    if is_large or oos_large:
        verdict = "A_large_delta_bug"
        note = ("Delta_bugが閾値(±5pp)を超過。Study76(RunB)結果はffill汚染の影響を無視できず、"
                "無効化・Dynamic Universeの再評価が必要。RunB_fixedを新しい基準として扱うべき。")
    else:
        verdict = "B_small_delta_bug"
        note = ("Delta_bugは閾値未満。ffill汚染はRunB結果に大きな影響を与えていない。"
                "観測された病理（低candidate数・大きいMaxDD）はDynamic Universe設計自体の特性である"
                "可能性が高い。")
    return {"threshold_pp": threshold_pp, "delta_is_cagr_pp": delta_is_cagr, "delta_oos_cagr_pp": delta_oos_cagr,
            "verdict": verdict, "note": note}


def main() -> int:
    started = datetime.now(_JST)
    print("[1/4] RunB入力を再構築中（contaminated baseline・決定論的）...")
    ds, dynamic_membership, rolling_rsr_contaminated, dynamic_active, panel = rebuild_run_b_inputs()

    print("[2/4] RunB（contaminated）をIS/OOSでfresh run再現中...")
    ds_contaminated = dict(ds)
    ds_contaminated["rsr_df"] = rolling_rsr_contaminated
    run_b_contaminated = {}
    for wname, s, e in (("IS", IS_START, IS_END), ("OOS", OOS_START, OOS_END)):
        raw = run_full_raw(ds_contaminated, dynamic_active, s, e)
        run_b_contaminated[wname] = extract_metrics(raw)
        print(f"  [RunB/{wname}] {run_b_contaminated[wname]}")

    print("[3/4] RunB_fixed（0埋め版）をIS/OOSでfresh run実行中...")
    rolling_rsr_fixed = build_monthly_rolling_rsr_zerofilled(rolling_rsr_contaminated)
    ds_fixed = dict(ds)
    ds_fixed["rsr_df"] = rolling_rsr_fixed
    run_b_fixed = {}
    for wname, s, e in (("IS", IS_START, IS_END), ("OOS", OOS_START, OOS_END)):
        raw = run_full_raw(ds_fixed, dynamic_active, s, e)
        run_b_fixed[wname] = extract_metrics(raw)
        print(f"  [RunB_fixed/{wname}] {run_b_fixed[wname]}")

    print("[4/4] Delta_bug・月次汚染率を算出中...")
    deltas = {w: compute_delta(run_b_fixed[w], run_b_contaminated[w]) for w in ("IS", "OOS")}
    monthly_contamination = compute_monthly_contamination(rolling_rsr_contaminated, dynamic_membership)
    total_non_member_symbol_days = sum(m["n_contaminated_symbol_days"] for m in monthly_contamination)
    avg_contamination_rate = float(np.mean([m["contamination_rate_symbols_pct"] for m in monthly_contamination])) \
        if monthly_contamination else 0.0

    judgment = judge(deltas["IS"]["cagr"], deltas["OOS"]["cagr"])

    out = {
        "study": "Study76D_contamination_ablation",
        "generated_at": datetime.now(_JST).isoformat(),
        "started_at": started.isoformat(),
        "run_b_contaminated_baseline": run_b_contaminated,
        "run_b_fixed_zerofilled": run_b_fixed,
        "delta_bug_fixed_minus_contaminated": deltas,
        "judgment": judgment,
        "monthly_contamination": monthly_contamination,
        "contamination_summary": {
            "avg_monthly_contamination_rate_symbols_pct": round(avg_contamination_rate, 2),
            "total_contaminated_symbol_days_all_months": total_non_member_symbol_days,
            "n_months": len(monthly_contamination),
        },
    }

    out_path = RESULTS_DIR / "study76d_results.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved: {out_path}")
    print(f"\nDelta_bug IS: {deltas['IS']}")
    print(f"Delta_bug OOS: {deltas['OOS']}")
    print(f"Judgment: {judgment['verdict']}")
    print(f"Avg monthly contamination rate: {avg_contamination_rate:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
