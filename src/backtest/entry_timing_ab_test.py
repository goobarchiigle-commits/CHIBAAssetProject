"""
backtest/entry_timing_ab_test.py

Entry Timing A/B テスト — 4モード比較

Mode A (baseline):    et_enabled=False                     現行と完全一致
Mode B (boost only):  et_enabled=True, boost_weight=0.06   boost のみ（block なし）
Mode C (block only):  et_enabled=True, block_low=True       block のみ（boost なし）
Mode D (full):        et_enabled=True, boost + block_low    boost + block

比較指標: CAGR / Sharpe / MaxDD / Calmar / Exposure / Trades/yr / Missed signals

出力: reports/entry_timing_ab_result.md

Run:
    cd C:/ai-trading
    python src/backtest/entry_timing_ab_test.py
"""

from __future__ import annotations

import sys, time, json, warnings
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

from src.config_loader import load_strategy_config
from src.backtest.capital_allocation_abc import (
    load_data, run_period,
    IS_START, IS_END, OOS_START, OOS_END,
)

REPORTS_DIR = Path("reports")

# ── 4モード定義 ───────────────────────────────────────────────────────
MODES = {
    "A": dict(et_enabled=False, et_block_low=False, et_boost_weight=0.0,
              label="A Baseline (ET完全OFF)"),
    "B": dict(et_enabled=True,  et_block_low=False, et_boost_weight=0.06,
              label="B ET boost only (block=OFF, boost=0.06)"),
    "C": dict(et_enabled=True,  et_block_low=True,  et_boost_weight=0.0,
              label="C block_low_confidence (boost=0, block=ON)"),
    "D": dict(et_enabled=True,  et_block_low=True,  et_boost_weight=0.06,
              label="D Full ET (boost=0.06, block=ON)"),
}


def run_all_modes(universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
                  topix_close, rsr_syms, cfg) -> dict:
    results: dict[str, dict] = {}

    for mode_key, mode_cfg in MODES.items():
        kw = {k: v for k, v in mode_cfg.items() if k != "label"}
        print(f"\n  Mode {mode_key}: {mode_cfg['label']}")

        for period_name, start, end in [("IS", IS_START, IS_END), ("OOS", OOS_START, OOS_END)]:
            print(f"    [{period_name}] {start}〜{end} ...", end=" ", flush=True)
            t0 = time.time()
            m = run_period(
                universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
                topix_close, rsr_syms, cfg,
                start=start, end=end,
                pattern="A",  # 資本配分ロジックは常に Pattern A
                **kw,
            )
            elapsed = time.time() - t0
            print(f"CAGR={m.get('cagr',0):+.1f}%  Sharpe={m.get('sharpe',0):.3f}  "
                  f"MaxDD={m.get('max_dd',0):.1f}%  ({elapsed:.1f}s)")

            results.setdefault(mode_key, {})[period_name] = m

    return results


def write_report(results: dict, output_path: Path) -> None:
    L = []; w = L.append

    w("# Entry Timing A/B テスト結果")
    w(f"\n作成日: {time.strftime('%Y-%m-%d')}  |  IS: {IS_START}〜{IS_END}  OOS: {OOS_START}〜{OOS_END}")
    w("\n**全モード共通: Pattern A 資本配分ロジック (max_positions=3)**\n")

    w("| Mode | 設定 |")
    w("|---|---|")
    for mk, mdef in MODES.items():
        w(f"| {mk} | {mdef['label']} |")

    # ── IS 比較 ────────────────────────────────────────────────────────
    w("\n---\n## IS 2018-2024 比較\n")
    w("| Mode | CAGR | Sharpe | Sortino | MaxDD | Calmar | PF | WinRate | Exposure | Missed |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for mk in MODES:
        m = results.get(mk, {}).get("IS", {})
        w(f"| {mk} "
          f"| {m.get('cagr',0):+.1f}% "
          f"| {m.get('sharpe',0):.3f} "
          f"| {m.get('sortino',0):.3f} "
          f"| {m.get('max_dd',0):.1f}% "
          f"| {m.get('calmar',0):.3f} "
          f"| {m.get('profit_factor',0):.3f} "
          f"| {m.get('win_rate',0):.1f}% "
          f"| {m.get('avg_exposure',0):.1f}% "
          f"| {m.get('missed_signals',0)} |")

    # IS annual returns
    w("\n### IS 年次リターン (%)\n")
    yr_keys = sorted(results.get("A",{}).get("IS",{}).get("annual_returns",{}).keys())
    w("| 年 | " + " | ".join(MODES.keys()) + " |")
    w("|---|" + "---|" * len(MODES))
    for yr in yr_keys:
        row = f"| {yr} |"
        for mk in MODES:
            v = results.get(mk,{}).get("IS",{}).get("annual_returns",{}).get(yr, 0)
            row += f" {v:+.1f}% |"
        w(row)

    # Δ vs Mode A (IS)
    w("\n### Δ vs Mode A (IS)\n")
    w("| Mode | ΔCAGR | ΔSharpe | ΔMaxDD | ΔCalmar | ΔMissed |")
    w("|---|---|---|---|---|---|")
    a_is = results.get("A",{}).get("IS",{})
    for mk in ["B", "C", "D"]:
        m = results.get(mk,{}).get("IS",{})
        w(f"| {mk} "
          f"| {m.get('cagr',0)-a_is.get('cagr',0):+.2f}pp "
          f"| {m.get('sharpe',0)-a_is.get('sharpe',0):+.4f} "
          f"| {m.get('max_dd',0)-a_is.get('max_dd',0):+.2f}pp "
          f"| {m.get('calmar',0)-a_is.get('calmar',0):+.4f} "
          f"| {m.get('missed_signals',0)-a_is.get('missed_signals',0):+d} |")

    # ── OOS 比較 ────────────────────────────────────────────────────────
    w("\n---\n## OOS 2025 比較\n")
    w("| Mode | CAGR | Sharpe | MaxDD | Calmar | Missed |")
    w("|---|---|---|---|---|---|")
    for mk in MODES:
        m = results.get(mk,{}).get("OOS",{})
        w(f"| {mk} "
          f"| {m.get('cagr',0):+.1f}% "
          f"| {m.get('sharpe',0):.3f} "
          f"| {m.get('max_dd',0):.1f}% "
          f"| {m.get('calmar',0):.3f} "
          f"| {m.get('missed_signals',0)} |")

    # Δ vs Mode A (OOS)
    w("\n### Δ vs Mode A (OOS)\n")
    w("| Mode | ΔCAGR | ΔSharpe | ΔMaxDD | ΔCalmar | ΔMissed |")
    w("|---|---|---|---|---|---|")
    a_oos = results.get("A",{}).get("OOS",{})
    for mk in ["B","C","D"]:
        m = results.get(mk,{}).get("OOS",{})
        w(f"| {mk} "
          f"| {m.get('cagr',0)-a_oos.get('cagr',0):+.2f}pp "
          f"| {m.get('sharpe',0)-a_oos.get('sharpe',0):+.4f} "
          f"| {m.get('max_dd',0)-a_oos.get('max_dd',0):+.2f}pp "
          f"| {m.get('calmar',0)-a_oos.get('calmar',0):+.4f} "
          f"| {m.get('missed_signals',0)-a_oos.get('missed_signals',0):+d} |")

    # ── 分析・結論 ────────────────────────────────────────────────────
    w("\n---\n## 分析\n")

    c_is = results.get("C",{}).get("IS",{})
    d_is = results.get("D",{}).get("IS",{})
    c_oos = results.get("C",{}).get("OOS",{})
    d_oos = results.get("D",{}).get("OOS",{})

    w("### block_low_confidence 効果 (Mode C: block_only)\n")
    c_missed_delta = c_is.get("missed_signals",0) - a_is.get("missed_signals",0)
    w(f"- IS ΔCAGR: {c_is.get('cagr',0)-a_is.get('cagr',0):+.2f}pp")
    w(f"- IS ΔMissed: {c_missed_delta:+d} 件 (blockedによるmissed増加)")
    w(f"- IS ΔSharpe: {c_is.get('sharpe',0)-a_is.get('sharpe',0):+.4f}")
    w(f"- OOS ΔCAGR: {c_oos.get('cagr',0)-a_oos.get('cagr',0):+.2f}pp\n")

    w("### ET boost 効果 (Mode B - Mode A)\n")
    b_is = results.get("B",{}).get("IS",{})
    w(f"- IS ΔCAGR: {b_is.get('cagr',0)-a_is.get('cagr',0):+.2f}pp (boost=0.06 only)\n")

    # Verdict
    w("### 結論\n")
    is_c_better = c_is.get("calmar",0) > a_is.get("calmar",0)
    is_d_better = d_is.get("calmar",0) > a_is.get("calmar",0)
    oos_c_ok    = c_oos.get("cagr",0) >= a_oos.get("cagr",0) * 0.90
    oos_d_ok    = d_oos.get("cagr",0) >= a_oos.get("cagr",0) * 0.90

    if is_c_better and oos_c_ok:
        verdict = "**Mode C (block_low_confidence) IS・OOS共にAを上回る → 採用候補**"
    elif is_c_better and not oos_c_ok:
        verdict = "**Mode C IS改善あり / OOS不安定 → RESEARCH (追加検証必要)**"
    elif not is_c_better:
        verdict = "**Mode C IS改善なし → block_low_confidence の単独効果は限定的**"
    else:
        verdict = "**判断保留 — IS/OOS結果確認後に判定**"

    w(f"block_low_confidence: {verdict}\n")

    if is_d_better and oos_d_ok:
        verdict_d = "**Mode D (full ET) IS・OOS共にAを上回る → 採用候補**"
    elif is_d_better:
        verdict_d = "**Mode D IS改善 / OOS未確認 → RESEARCH**"
    else:
        verdict_d = "**Mode D IS改善なし → Full ET の複合効果も限定的**"
    w(f"full ET: {verdict_d}\n")

    w("### 次のアクション\n")
    w("| 優先度 | アクション | 条件 |")
    w("|---|---|---|")
    w("| 1 | Mode C/D の Walk-Forward (5-Fold) 検証 | IS Calmar改善が確認された場合 |")
    w("| 2 | block_low_confidence=true を strategy.yaml に反映 | WF 3/5以上でA勝利の場合 |")
    w("| 3 | ET boost_weight sweep (0.06→0.10→0.15) | Mode B/D の IS改善が >0.5pp の場合 |")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(L), encoding="utf-8")
    print(f"\n  レポート保存: {output_path}")


def main() -> int:
    cfg = load_strategy_config()

    print("=" * 68)
    print("  Entry Timing A/B Test — 4モード比較")
    print("  Mode A (baseline) / B (boost) / C (block) / D (boost+block)")
    print("=" * 68)

    print("\n[1/2] データロード中...")
    (universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
     rsr_syms, topix_close, cfg) = load_data(cfg)

    print("\n[2/2] 4モード × IS+OOS 実行中...")
    results = run_all_modes(
        universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
        topix_close, rsr_syms, cfg
    )

    out_path = REPORTS_DIR / "entry_timing_ab_result.md"
    write_report(results, out_path)

    # ── Console summary ────────────────────────────────────────────────
    print("\n" + "="*68)
    print("  ★ 結果サマリー")
    print("="*68)
    print(f"\n  {'Mode':<6} {'IS CAGR':>9} {'IS Sharpe':>10} {'IS MaxDD':>9} {'IS Calmar':>10} {'IS Missed':>10} │ {'OOS CAGR':>9} {'OOS Sharpe':>11}")
    print("  " + "-"*80)
    for mk, mdef in MODES.items():
        is_m  = results.get(mk,{}).get("IS",{})
        oos_m = results.get(mk,{}).get("OOS",{})
        print(f"  {mk:<6} {is_m.get('cagr',0):>+8.1f}% "
              f"{is_m.get('sharpe',0):>10.3f} "
              f"{is_m.get('max_dd',0):>8.1f}% "
              f"{is_m.get('calmar',0):>10.3f} "
              f"{is_m.get('missed_signals',0):>10} │ "
              f"{oos_m.get('cagr',0):>+8.1f}% "
              f"{oos_m.get('sharpe',0):>11.3f}")

    print(f"\n  レポート → {out_path}")
    print("="*68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
