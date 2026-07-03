"""
src/backtest/msw_walkforward.py

MAX_SINGLE_W (1銘柄配分上限比率) Walk-Forward Validation (5-Fold)

比較:
  Case A = 0.25 (現行 PARAMS_LOCKED)
  Case B = 0.30 (研究候補)
  Case C = 0.33 (研究候補)

固定:
  max_positions=3 / MAX_NEW=2 / min_rsr=75 / rsr_exit=70
  entry / exit / sizing — 変更なし

Fold定義:
  Fold1: Test=2021  Fold2: Test=2022 Bear  Fold3: Test=2023
  Fold4: Test=2024  Fold5: Test=2025

採用条件 (全4条件 AND):
  WF勝率         >= 4/5        (ΔCalmar > 0)
  Fold2 ΔCalmar  >= -0.05      (2022 Bear 非悪化 恒久ゲート)
  平均 ΔCAGR     >= +1.0pp
  MaxDD悪化      <= +2.0pp

出力: reports/msw_walkforward.md

実行:
    cd C:/ai-trading
    python src/backtest/msw_walkforward.py
"""

from __future__ import annotations

import sys, time, warnings
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date as Date

from src.config_loader import load_strategy_config
from src.backtest.capital_allocation_abc import load_data, run_period

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

# ─── Fold 定義 ─────────────────────────────────────────────────────────────
FOLDS = [
    ("Fold1", "2018-01-01", "2020-12-31", "2021-01-01", "2021-12-31", "2021弱年"),
    ("Fold2", "2018-01-01", "2021-12-31", "2022-01-01", "2022-12-31", "2022Bear"),
    ("Fold3", "2018-01-01", "2022-12-31", "2023-01-01", "2023-12-31", "2023強気"),
    ("Fold4", "2018-01-01", "2023-12-31", "2024-01-01", "2024-12-31", "2024"),
    ("Fold5", "2018-01-01", "2024-12-31", "2025-01-01", "2025-12-31", "2025"),
]

# ─── 研究ケース ───────────────────────────────────────────────────────────
CASES = {
    "A": 0.25,
    "B": 0.30,
    "C": 0.33,
}

# ─── 採用条件 ─────────────────────────────────────────────────────────────
WF_WIN_MIN        = 4 / 5
FOLD2_DCALMAR_MIN = -0.05
AVG_DCAGR_MIN_PP  = 1.0
MAX_DD_DEGRADE_PP = 2.0   # MaxDD悪化上限 pp (絶対値差)


# ─── 単一期間実行 ──────────────────────────────────────────────────────────
def _run_one(universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
             topix_close, rsr_syms, cfg,
             start: str, end: str, msw: float, label: str) -> dict:
    t0 = time.time()
    print(f"    {label} ...", end=" ", flush=True)
    m = run_period(
        universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
        topix_close, rsr_syms, cfg,
        start=start, end=end,
        pattern="A",
        max_new_per_day=2,
        max_single_weight_override=msw,
        return_cash_skipped=True,
        verbose=False,
    )
    n_skip  = len(m.get("_cash_skipped", []))
    n_rank1 = sum(1 for s in m.get("_cash_skipped", []) if s["rank"] == 0)
    print(
        f"CAGR={m.get('cagr', 0):+.1f}%  Cal={m.get('calmar', 0):.3f}  "
        f"DD={m.get('max_dd', 0):.1f}%  PF={m.get('profit_factor', 0):.2f}  "
        f"skip={n_skip}(r1={n_rank1})  ({time.time()-t0:.1f}s)"
    )
    m["_n_skip"]  = n_skip
    m["_n_rank1"] = n_rank1
    return m


# ─── Fold 実行 ────────────────────────────────────────────────────────────
def run_fold(fname, is_start, is_end, ts, te, feature,
             universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
             topix_close, rsr_syms, cfg) -> dict:
    print(f"\n  ── {fname} (Test={te[:4]} / {feature}) ──")
    row: dict = {"fold": fname, "test_year": te[:4], "feature": feature}
    for case, msw in CASES.items():
        row[f"{case}_IS"]   = _run_one(universe_raw, rsr_df, alpha_df, sym_active_df,
                                        regime_df, topix_close, rsr_syms, cfg,
                                        is_start, is_end, msw,
                                        f"[{fname}] {case}(msw={msw})/IS")
        row[f"{case}_Test"] = _run_one(universe_raw, rsr_df, alpha_df, sym_active_df,
                                        regime_df, topix_close, rsr_syms, cfg,
                                        ts, te, msw,
                                        f"[{fname}] {case}(msw={msw})/Test")
    return row


# ─── レポート生成 ──────────────────────────────────────────────────────────
def write_report(fold_rows: list, out: Path) -> None:
    def _m(row, case, period):
        return row.get(f"{case}_{period}", {})

    def _v(m, key, default=0.0):
        return m.get(key, default)

    def _delta(row, case, key):
        return _v(_m(row, case, "Test"), key) - _v(_m(row, "A", "Test"), key)

    lines: list = []
    A = lines.append

    A(f"# MAX_SINGLE_W Walk-Forward Validation")
    A(f"")
    A(f"作成日: {Date.today().isoformat()}")
    A(f"")
    A(f"**ケース**: A=0.25(現行) / B=0.30 / C=0.33")
    A(f"**固定**: max_positions=3 / MAX_NEW=2 / min_rsr=75 / rsr_exit=70")
    A(f"**採用条件**: WF≥4/5 ∧ Fold2 ΔCalmar≥-0.05 ∧ ΔCAGR≥+1.0pp ∧ MaxDD悪化≤+2pp")
    A(f"")
    A(f"---")

    # ── Section 1: Fold別 Test比較表 ─────────────────────────────────
    A(f"## 1. Fold別 Test期間比較")
    A(f"")
    for case in CASES:
        A(f"### Case {case} (MSW={CASES[case]})")
        A(f"")
        A(f"| Fold | Year | CAGR | Calmar | MaxDD | PF | WR | Skip | R1Skip |")
        A(f"|---|---|---|---|---|---|---|---|---|")
        for row in fold_rows:
            m = _m(row, case, "Test")
            A(f"| {row['fold']} | {row['test_year']} | {_v(m,'cagr'):+.1f}% | "
              f"{_v(m,'calmar'):.3f} | {_v(m,'max_dd'):.1f}% | "
              f"{_v(m,'profit_factor'):.2f} | {_v(m,'win_rate'):.1f}% | "
              f"{m.get('_n_skip',0)} | {m.get('_n_rank1',0)} |")
        A(f"")

    # ── Section 2: A vs B 差分表 ──────────────────────────────────────
    A(f"## 2. A vs B / A vs C 差分表 (Test期間)")
    A(f"")
    A(f"| Fold | Year | ΔCAGR(B-A) | ΔCal(B-A) | ΔDD(B-A) | ΔCAGR(C-A) | ΔCal(C-A) | ΔDD(C-A) |")
    A(f"|---|---|---|---|---|---|---|---|")
    for row in fold_rows:
        dcagr_b = _delta(row, "B", "cagr")
        dcal_b  = _delta(row, "B", "calmar")
        ddd_b   = _delta(row, "B", "max_dd")
        dcagr_c = _delta(row, "C", "cagr")
        dcal_c  = _delta(row, "C", "calmar")
        ddd_c   = _delta(row, "C", "max_dd")
        A(f"| {row['fold']} | {row['test_year']} | "
          f"{dcagr_b:+.2f}pp | {dcal_b:+.4f} | {ddd_b:+.2f}pp | "
          f"{dcagr_c:+.2f}pp | {dcal_c:+.4f} | {ddd_c:+.2f}pp |")
    A(f"")

    # ── Section 3: 2022 Bear 単独 ─────────────────────────────────────
    A(f"## 3. 2022 Bear 単独比較 (Fold2)")
    A(f"")
    fold2 = next((r for r in fold_rows if r["fold"] == "Fold2"), None)
    if fold2:
        A(f"| ケース | MSW | CAGR | Calmar | MaxDD | ΔCAGR | ΔCalmar | ΔMaxDD |")
        A(f"|---|---|---|---|---|---|---|---|")
        for case, msw in CASES.items():
            m = _m(fold2, case, "Test")
            if case == "A":
                A(f"| **{case}(baseline)** | {msw} | {_v(m,'cagr'):+.1f}% | "
                  f"{_v(m,'calmar'):.3f} | {_v(m,'max_dd'):.1f}% | — | — | — |")
            else:
                dc   = _delta(fold2, case, "cagr")
                dcal = _delta(fold2, case, "calmar")
                ddd  = _delta(fold2, case, "max_dd")
                gate_cal = "✅" if dcal >= FOLD2_DCALMAR_MIN else "❌"
                A(f"| **{case}** | {msw} | {_v(m,'cagr'):+.1f}% | "
                  f"{_v(m,'calmar'):.3f} | {_v(m,'max_dd'):.1f}% | "
                  f"{dc:+.2f}pp | {dcal:+.4f} {gate_cal} | {ddd:+.2f}pp |")
    A(f"")

    # ── Section 4: WF Summary & Verdict ──────────────────────────────
    A(f"## 4. Walk-Forward サマリー & 採用判定")
    A(f"")
    verdicts: dict = {}
    for case in ["B", "C"]:
        wins = 0
        dcals, dcagrs, ddds = [], [], []
        for row in fold_rows:
            dcal  = _delta(row, case, "calmar")
            dcagr = _delta(row, case, "cagr")
            ddd   = _delta(row, case, "max_dd")
            if dcal > 0:
                wins += 1
            dcals.append(dcal)
            dcagrs.append(dcagr)
            ddds.append(ddd)

        avg_dcal  = float(np.mean(dcals))
        avg_dcagr = float(np.mean(dcagrs))
        avg_ddd   = float(np.mean(ddds))
        fold2_dc  = dcals[1]
        worst_ddd = -min(ddds)   # ddd=Case-A (負=悪化); 最大悪化量 = -min

        crit_wf   = wins / len(fold_rows) >= WF_WIN_MIN
        crit_2022 = fold2_dc >= FOLD2_DCALMAR_MIN
        crit_cagr = avg_dcagr >= AVG_DCAGR_MIN_PP
        crit_dd   = worst_ddd <= MAX_DD_DEGRADE_PP
        all_pass  = crit_wf and crit_2022 and crit_cagr and crit_dd

        verdict = "採用" if all_pass else (
            "保留" if (crit_2022 and avg_dcagr >= 0.5) else "REJECT"
        )
        verdicts[case] = verdict

        A(f"### Case {case} (MSW={CASES[case]}) vs A")
        A(f"")
        A(f"| 指標 | 値 | 基準 | 判定 |")
        A(f"|---|---|---|---|")
        A(f"| WF勝率 | {wins}/{len(fold_rows)} | ≥4/5 | {'✅' if crit_wf else '❌'} |")
        A(f"| Fold2 ΔCalmar | {fold2_dc:+.4f} | ≥-0.05 | {'✅' if crit_2022 else '❌'} |")
        A(f"| 平均 ΔCAGR | {avg_dcagr:+.2f}pp | ≥+1.0pp | {'✅' if crit_cagr else '❌'} |")
        A(f"| MaxDD最大悪化 | {worst_ddd:+.2f}pp | ≤+2.0pp | {'✅' if crit_dd else '❌'} |")
        A(f"| 平均 ΔCalmar | {avg_dcal:+.4f} | (参考) | — |")
        A(f"| 平均 ΔMaxDD | {avg_ddd:+.2f}pp | (参考) | — |")
        A(f"")
        A(f"**★ 判定: {verdict}**")
        A(f"")

    # ── Section 5: スキップ件数比較 ──────────────────────────────────
    A(f"## 5. 資金不足スキップ件数 (全Fold合計)")
    A(f"")
    A(f"| Case | MSW | 総Skip | rank1 Skip | 減少(vs A) |")
    A(f"|---|---|---|---|---|")
    skip_a  = sum(row[f"A_Test"].get("_n_skip",  0) for row in fold_rows)
    r1a     = sum(row[f"A_Test"].get("_n_rank1", 0) for row in fold_rows)
    A(f"| A | 0.25 | {skip_a} | {r1a} | — |")
    for case, msw in [("B", 0.30), ("C", 0.33)]:
        sk = sum(row[f"{case}_Test"].get("_n_skip",  0) for row in fold_rows)
        r1 = sum(row[f"{case}_Test"].get("_n_rank1", 0) for row in fold_rows)
        red    = skip_a - sk
        red_r1 = r1a   - r1
        A(f"| {case} | {msw} | {sk} | {r1} | -{red}(-{red_r1} rank1) |")
    A(f"")

    out.write_text("\n".join(lines), encoding="utf-8")


# ─── コンソール最終サマリー ────────────────────────────────────────────────
def print_summary(fold_rows: list) -> None:
    def _m(row, case, period):
        return row.get(f"{case}_{period}", {})

    def _delta(row, case, key):
        return _m(row, case, "Test").get(key, 0.0) - _m(row, "A", "Test").get(key, 0.0)

    print("\n" + "=" * 80)
    print("  ★ Walk-Forward Summary — MAX_SINGLE_W")
    print("  Case B=0.30 / Case C=0.33 vs A=0.25")
    print("=" * 80)

    for case in ["B", "C"]:
        print(f"\n  {'Fold':<6} {'Year':<6} {'特徴':<8} "
              f"{'CAGR_A':>8} {'CAGR_'+case:>8} {'ΔCAGR':>8} "
              f"{'Cal_A':>7} {'Cal_'+case:>7} {'ΔCal':>8}  判定")
        print("  " + "-" * 85)
        wins = 0
        dcals, dcagrs, ddds = [], [], []
        for row in fold_rows:
            a = _m(row, "A", "Test")
            b = _m(row, case, "Test")
            dcal  = b.get("calmar", 0) - a.get("calmar", 0)
            dcagr = b.get("cagr",   0) - a.get("cagr",   0)
            ddd   = b.get("max_dd", 0) - a.get("max_dd", 0)
            if dcal > 0:
                wins += 1
            dcals.append(dcal); dcagrs.append(dcagr); ddds.append(ddd)
            tick = "✅" if dcal > 0 else "❌"
            print(f"  {row['fold']:<6} {row['test_year']:<6} {row['feature']:<8} "
                  f"{a.get('cagr',0):>7.1f}%  {b.get('cagr',0):>7.1f}%  "
                  f"{dcagr:>+7.2f}pp  {a.get('calmar',0):>7.3f}  "
                  f"{b.get('calmar',0):>7.3f}  {dcal:>+8.4f}  {tick}")
        avg_dcagr = float(np.mean(dcagrs))
        fold2_dc  = dcals[1]
        worst_ddd = -min(ddds)   # ddd=Case-A (負=悪化); 最大悪化量 = -min
        wf_rate   = wins / len(fold_rows)
        crit_wf   = wf_rate >= WF_WIN_MIN
        crit_2022 = fold2_dc >= FOLD2_DCALMAR_MIN
        crit_cagr = avg_dcagr >= AVG_DCAGR_MIN_PP
        crit_dd   = worst_ddd <= MAX_DD_DEGRADE_PP
        all_pass  = crit_wf and crit_2022 and crit_cagr and crit_dd
        verdict   = "採用" if all_pass else ("保留" if (crit_2022 and avg_dcagr >= 0.5) else "REJECT")
        print("  " + "-" * 85)
        print(f"  WF勝率: {wins}/5 {'✅' if crit_wf else '❌'}  "
              f"ΔCal_avg: {float(np.mean(dcals)):+.4f}  "
              f"ΔCAGR_avg: {avg_dcagr:+.2f}pp {'✅' if crit_cagr else '❌'}  "
              f"Fold2_ΔCal: {fold2_dc:+.4f} {'✅' if crit_2022 else '❌'}  "
              f"MaxDD最悪: {worst_ddd:+.2f}pp {'✅' if crit_dd else '❌'}")
        print(f"  ★ Case {case} 最終判定: {verdict}")
    print("=" * 80)


# ─── main ─────────────────────────────────────────────────────────────────
def main() -> int:
    print("=" * 80)
    print("  MAX_SINGLE_W Walk-Forward Validation")
    print("  A=0.25(現行) / B=0.30 / C=0.33  |  5-Fold × 2期間(IS+Test)")
    print("  採用条件: WF≥4/5 ∧ Fold2 ΔCal≥-0.05 ∧ ΔCAGR≥+1pp ∧ MaxDD悪化≤+2pp")
    print("=" * 80)

    cfg = load_strategy_config()
    print("\n[1/2] データロード中 (RSR42 + TOPIX)...")
    (universe_raw, rsr_df, alpha_df, sym_active_df,
     regime_df, rsr_syms, topix_close, cfg) = load_data(cfg)

    print(f"\n[2/2] 5 Fold × 3 Case × IS+Test 実行中...")
    fold_rows: list = []
    for fname, is_s, is_e, ts, te, feature in FOLDS:
        row = run_fold(fname, is_s, is_e, ts, te, feature,
                       universe_raw, rsr_df, alpha_df, sym_active_df,
                       regime_df, topix_close, rsr_syms, cfg)
        fold_rows.append(row)

    out = REPORTS_DIR / "msw_walkforward.md"
    write_report(fold_rows, out)
    print(f"\n  レポート → {out}")

    print_summary(fold_rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
