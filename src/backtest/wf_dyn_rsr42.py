"""
backtest/wf_dyn_rsr42.py
動的ユニバース: RSR42内動的選択版（TOPIX100拡張なし）

仮説: TOPIX100拡張はalpha_dfがRSR42ベースのため実質RSR42を制限するだけ。
     RSR42のみで Bear rs>0 フィルターを適用すれば2022改善できる？

実行:
  cd C:/ai-trading
  python src/backtest/wf_dyn_rsr42.py
"""

from __future__ import annotations
import os, sys, json, warnings, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# wf_dynamic_universe から全関数を再利用
from backtest.wf_dynamic_universe import (
    load_data, run_seg, run_wf_for_config, _build_active,
    BULL_ACTIVE_N, BEAR_ACTIVE_N, PASS_SEG3_2022_MIN, PASS_WF_WINS_MIN, PASS_2025_SH_MIN,
    WF_SEGS, TRUE_OOS, IS_FULL, OUTPUT_DIR
)
import backtest.composite_alpha_bt as _bt
from src.config_loader import load_strategy_config

os.environ.pop("DATA_VERSION", None)

OUTPUT_JSON = os.path.join(OUTPUT_DIR, f"wf_dyn_rsr42_{time.strftime('%Y-%m-%d')}.json")


def main():
    cfg = load_strategy_config()

    print("=" * 72)
    print("  動的ユニバース: RSR42内動的選択 WF検証")
    print(f"  Bull Top{BULL_ACTIVE_N}: mom(0.40)+rsr(0.35)+vol(0.25) / RSR42プール")
    print(f"  Bear Top{BEAR_ACTIVE_N}: rs_topix(0.50)+rsr(0.30)+vol(0.20) + rs>0フィルター / RSR42プール")
    print("=" * 72 + "\n")

    (universe_raw, rsr_df_rsr42, rsr_df_full, alpha_df, regime_df, rsr_syms, all_syms,
     tech_matrices, topix_close, cfg) = load_data(cfg)

    all_results = []

    # ベースライン（比較用）
    all_results.append(run_wf_for_config(
        "baseline_rsr42_fixed", None,
        universe_raw, rsr_df_rsr42, alpha_df, regime_df, rsr_syms,
        tech_matrices, topix_close, cfg,
    ))

    # 設定D: RSR42内 + Bear rs>0フィルター
    all_results.append(run_wf_for_config(
        "dyn_rsr42_bear_rs0", BULL_ACTIVE_N,
        universe_raw, rsr_df_rsr42, alpha_df, regime_df, rsr_syms,
        tech_matrices, topix_close, cfg,
        bear_n=BEAR_ACTIVE_N, bear_pool=None,
    ))

    # 設定E: RSR42内 + Bull スコアそのまま Top20（rs>0なし）
    all_results.append(run_wf_for_config(
        "dyn_rsr42_bear_score20", BULL_ACTIVE_N,
        universe_raw, rsr_df_rsr42, alpha_df, regime_df, rsr_syms,
        tech_matrices, topix_close, cfg,
        bear_n=BEAR_ACTIVE_N, bear_pool="bull_score_only",
    ))

    # 設定F: RSR42内 + Bear rs>0 + bear_n=25（やや緩め）
    all_results.append(run_wf_for_config(
        "dyn_rsr42_bear_rs0_n25", BULL_ACTIVE_N,
        universe_raw, rsr_df_rsr42, alpha_df, regime_df, rsr_syms,
        tech_matrices, topix_close, cfg,
        bear_n=25, bear_pool=None,
    ))

    print("\n" + "=" * 72)
    print("  比較サマリー（判定: Seg3_2022>0 / WF5/5 / 2025>0.80）")
    print(f"  {'設定':<28}  {'WF勝率':>6}  {'Seg3_22':>8}  {'中央値':>7}"
          f"  {'worstDD':>9}  {'IS Sh':>6}  {'2025 Sh':>7}  {'判定':>6}")
    print("  " + "-" * 80)
    for r in all_results:
        s  = r["wf_summary"]
        fi = r["full_is"]
        t  = r["true_oos_2025"]
        seg3_oos = next((x["oos_sharpe"] for x in r["segments"] if x["seg"] == 3), None)
        ok_seg3  = seg3_oos is not None and seg3_oos > PASS_SEG3_2022_MIN
        ok_wf    = s["pass_count"] == f"{PASS_WF_WINS_MIN}/5"
        ok_2025  = t["sharpe"] >= PASS_2025_SH_MIN
        verdict  = "PASS" if (ok_seg3 and ok_wf and ok_2025) else "FAIL"
        mark = "★" if verdict == "PASS" and r["config_name"] != "baseline_rsr42_fixed" else " "
        seg3_str = f"{seg3_oos:+.3f}" if seg3_oos is not None else "  N/A "
        print(f"  {mark}{r['config_name']:<27}  {s['pass_count']:>6}  {seg3_str:>8}"
              f"  {s['median_oos_sharpe']:>7.3f}  {s['worst_oos_dd']:>+9.1f}%"
              f"  {fi['sharpe']:>6.3f}  {t['sharpe']:>7.3f}  {verdict:>6}")
    print("=" * 72)

    output = {
        "date":        time.strftime("%Y-%m-%d"),
        "description": "動的ユニバース RSR42内動的選択版",
        "results":     all_results,
    }
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  結果保存: {OUTPUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
