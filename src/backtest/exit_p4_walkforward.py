"""
src/backtest/exit_p4_walkforward.py

P4 含み益非対称 Exit ポリシー Walk-Forward Validation (5-Fold)

比較:
  A = Baseline (rsr_exit=70, 現行)
  B = P4       (rsr < 70 かつ close > entry + 0.5×ATR20 → eff_thr=65, otherwise 70)

Fold定義:
  Fold1: Train=2018-2020  Test=2021  (弱年)
  Fold2: Train=2018-2021  Test=2022  (Bear / 金利ショック) ← 恒久ゲート
  Fold3: Train=2018-2022  Test=2023  (強気)
  Fold4: Train=2018-2023  Test=2024
  Fold5: Train=2018-2024  Test=2025

採用条件 (全3条件 AND):
  WF勝率        >= 4/5        (ΔCalmar > 0 の数)
  Fold2 ΔCalmar >= -0.05      (2022 Bear 非悪化 恒久ゲート)
  平均ΔCAGR     >= +1.0pp

出力: reports/exit_p4_walkforward.md

実行:
    cd C:/ai-trading
    python src/backtest/exit_p4_walkforward.py
"""

from __future__ import annotations

import sys, time, warnings
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

from src.config_loader import load_strategy_config
from src.backtest.capital_allocation_abc import load_data, run_period

REPORTS_DIR = Path("reports")

FOLDS = [
    ("Fold1", "2018-01-01", "2020-12-31", "2021-01-01", "2021-12-31", "2021弱年"),
    ("Fold2", "2018-01-01", "2021-12-31", "2022-01-01", "2022-12-31", "2022Bear"),
    ("Fold3", "2018-01-01", "2022-12-31", "2023-01-01", "2023-12-31", "2023強気"),
    ("Fold4", "2018-01-01", "2023-12-31", "2024-01-01", "2024-12-31", "2024"),
    ("Fold5", "2018-01-01", "2024-12-31", "2025-01-01", "2025-12-31", "2025"),
]

WF_WIN_MIN        = 4 / 5
FOLD2_DCALMAR_MIN = -0.05
AVG_DCAGR_MIN_PP  = 1.0


# ─────────────────────────────────────────────────────────────────────────────


def _run_fold(fold_name, is_start, is_end, test_start, test_end,
              universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
              topix_close, rsr_syms, cfg) -> dict:
    row = {"fold": fold_name, "test_year": test_start[:4]}
    for mode, kwargs in [
        ("A", dict(hold_policy=None)),
        ("B", dict(hold_policy="P4")),
    ]:
        for period, start, end in [
            ("IS",   is_start,   is_end),
            ("Test", test_start, test_end),
        ]:
            label = f"{mode}_{period}"
            print(f"    [{fold_name}] {mode}/{period} ...", end=" ", flush=True)
            t0 = time.time()
            m = run_period(
                universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
                topix_close, rsr_syms, cfg,
                start=start, end=end,
                pattern="A",
                **kwargs,
            )
            print(
                f"CAGR={m.get('cagr', 0):+.1f}%  "
                f"Calmar={m.get('calmar', 0):.3f}  "
                f"DD={m.get('max_dd', 0):.1f}%  "
                f"PF={m.get('profit_factor', 0):.2f}  "
                f"RSR_EXIT={m.get('exit_reasons', {}).get('RSR_EXIT', 0)}  "
                f"AvgHold={m.get('avg_hold_days', 0):.1f}d  "
                f"({time.time() - t0:.1f}s)"
            )
            row[label] = m
    return row


def _m(row, mode, period):
    return row.get(f"{mode}_{period}", {})


def _d(b, a, key):
    return b.get(key, 0.0) - a.get(key, 0.0)


def _rsr(m):
    return m.get("exit_reasons", {}).get("RSR_EXIT", 0)


# ─────────────────────────────────────────────────────────────────────────────


def write_report(fold_rows: list, output_path: Path) -> None:
    wins = 0
    deltas_calmar: list[float] = []
    deltas_cagr:   list[float] = []

    for row in fold_rows:
        a = _m(row, "A", "Test")
        b = _m(row, "B", "Test")
        dc    = _d(b, a, "calmar")
        dcagr = _d(b, a, "cagr")
        if dc > 0:
            wins += 1
        deltas_calmar.append(dc)
        deltas_cagr.append(dcagr)

    avg_dc    = float(np.mean(deltas_calmar))
    avg_dcagr = float(np.mean(deltas_cagr))
    wf_rate   = wins / len(fold_rows)
    fold2_dc  = deltas_calmar[1]   # Fold2 = 2022 Bear

    crit_wf   = wf_rate >= WF_WIN_MIN
    crit_2022 = fold2_dc >= FOLD2_DCALMAR_MIN
    crit_cagr = avg_dcagr >= AVG_DCAGR_MIN_PP
    all_pass  = crit_wf and crit_2022 and crit_cagr

    if all_pass:
        verdict = "APPROVE ✅"
        verdict_note = (
            f"全採用条件クリア (WF={wins}/5, Fold2 ΔCal={fold2_dc:+.4f}, ΔCAGR={avg_dcagr:+.2f}pp)"
        )
    elif not crit_2022:
        verdict = "REJECT — 2022 Bear 恒久ゲート失敗 ❌"
        verdict_note = f"Fold2 ΔCalmar={fold2_dc:+.4f} < {FOLD2_DCALMAR_MIN}"
    elif not crit_wf:
        verdict = "REJECT — WF勝率不足 ❌"
        verdict_note = f"WF勝率={wins}/5 < 4/5"
    else:
        verdict = "REJECT — ΔCAGR不足 ❌"
        verdict_note = f"平均ΔCAGR={avg_dcagr:+.2f}pp < {AVG_DCAGR_MIN_PP}pp"

    L = []; w = L.append

    w("# P4 含み益非対称 Exit — Walk-Forward Validation (5-Fold)")
    w(f"\n作成日: {time.strftime('%Y-%m-%d')}  |  5-Fold Walk-Forward")
    w("\n**比較**:")
    w("- A = Baseline (rsr_exit=70, 現行 strategy.yaml)")
    w("- B = P4 (rsr < 70 かつ close > entry + 0.5×ATR20 → eff_thr=65; otherwise eff_thr=70)")
    w("\n**採用条件 (全3条件 AND)**:")
    w(f"1. WF勝率 ≥ 4/5  (ΔCalmar > 0)")
    w(f"2. 2022 Fold2 ΔCalmar ≥ {FOLD2_DCALMAR_MIN}  (Bear 非悪化 恒久ゲート)")
    w(f"3. 平均ΔCAGR ≥ +{AVG_DCAGR_MIN_PP:.1f}pp\n")

    # ── 1. IS Results ──────────────────────────────────────────────────────
    w("---\n## 1. IS Results (各Fold IS期間)\n")
    w("| Fold | IS期間 | CAGR_A | CAGR_B | ΔCAGR | Calmar_A | Calmar_B | ΔCalmar | PF_A | PF_B | RSR_A | RSR_B |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for row in fold_rows:
        fi  = [r[0] for r in FOLDS].index(row["fold"])
        ai  = _m(row, "A", "IS"); bi = _m(row, "B", "IS")
        w(f"| {row['fold']} | {FOLDS[fi][1][:4]}〜{FOLDS[fi][2][:4]} "
          f"| {ai.get('cagr',0):+.1f}% | {bi.get('cagr',0):+.1f}% "
          f"| {_d(bi,ai,'cagr'):+.2f}pp "
          f"| {ai.get('calmar',0):.3f} | {bi.get('calmar',0):.3f} "
          f"| {_d(bi,ai,'calmar'):+.4f} "
          f"| {ai.get('profit_factor',0):.2f} | {bi.get('profit_factor',0):.2f} "
          f"| {_rsr(ai)} | {_rsr(bi)} |")

    # ── 2. WF Test Results ────────────────────────────────────────────────
    w("\n---\n## 2. Walk-Forward Test Results\n")
    w("| Fold | Year | 特徴 | CAGR_A | CAGR_B | ΔCAGR | Cal_A | Cal_B | ΔCal | DD_A | DD_B | PF_A | PF_B | 判定 |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for i, row in enumerate(fold_rows):
        a = _m(row, "A", "Test"); b = _m(row, "B", "Test")
        win  = deltas_calmar[i] > 0
        icon = "✅" if win else "❌"
        w(f"| {row['fold']} | {row['test_year']} | {FOLDS[i][5]} "
          f"| {a.get('cagr',0):+.1f}% | {b.get('cagr',0):+.1f}% "
          f"| {deltas_cagr[i]:+.2f}pp "
          f"| {a.get('calmar',0):.3f} | {b.get('calmar',0):.3f} "
          f"| {deltas_calmar[i]:+.4f} "
          f"| {a.get('max_dd',0):.1f}% | {b.get('max_dd',0):.1f}% "
          f"| {a.get('profit_factor',0):.2f} | {b.get('profit_factor',0):.2f} "
          f"| {icon} |")

    # ── 3. Hold Days & Exit Reasons ───────────────────────────────────────
    w("\n---\n## 3. Hold Days & Exit Reasons (Test期間)\n")
    w("| Fold | Year | AvgHold_A | AvgHold_B | ΔHold | RSR_A | RSR_B | ΔRSR | Trades_A | Trades_B |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for i, row in enumerate(fold_rows):
        a = _m(row, "A", "Test"); b = _m(row, "B", "Test")
        ra = _rsr(a); rb = _rsr(b)
        w(f"| {row['fold']} | {row['test_year']} "
          f"| {a.get('avg_hold_days',0):.1f}d | {b.get('avg_hold_days',0):.1f}d "
          f"| {b.get('avg_hold_days',0) - a.get('avg_hold_days',0):+.1f}d "
          f"| {ra} | {rb} | {rb - ra:+d} "
          f"| {a.get('n_trades',0)} | {b.get('n_trades',0)} |")

    # ── 4. Exit Reason Distribution ───────────────────────────────────────
    w("\n---\n## 4. Exit Reason 分布 (Test全Fold合計)\n")
    total_a: dict[str, int] = defaultdict(int)
    total_b: dict[str, int] = defaultdict(int)
    for row in fold_rows:
        for r, cnt in _m(row, "A", "Test").get("exit_reasons", {}).items():
            total_a[r] += cnt
        for r, cnt in _m(row, "B", "Test").get("exit_reasons", {}).items():
            total_b[r] += cnt
    all_reasons = sorted(set(list(total_a.keys()) + list(total_b.keys())))
    n_a = sum(total_a.values()); n_b = sum(total_b.values())
    w("| Exit理由 | N_A | N_B | ΔN | %_A | %_B |")
    w("|---|---|---|---|---|---|")
    for r in all_reasons:
        na = total_a.get(r, 0); nb = total_b.get(r, 0)
        w(f"| {r} | {na} | {nb} | {nb - na:+d} "
          f"| {na / max(n_a, 1) * 100:.0f}% | {nb / max(n_b, 1) * 100:.0f}% |")
    w(f"| **合計** | **{n_a}** | **{n_b}** | {n_b - n_a:+d} | 100% | 100% |")

    # ── 5. 2022 Bear 単独 ─────────────────────────────────────────────────
    w("\n---\n## 5. 2022 Bear 単独詳細 (Fold2 / 恒久ゲート)\n")
    r22_a = _m(fold_rows[1], "A", "Test")
    r22_b = _m(fold_rows[1], "B", "Test")
    w("| 指標 | A (Baseline) | B (P4) | Δ |")
    w("|---|---|---|---|")
    for key, label in [
        ("cagr",         "CAGR"),
        ("sharpe",       "Sharpe"),
        ("max_dd",       "MaxDD"),
        ("calmar",       "Calmar"),
        ("profit_factor","PF"),
        ("avg_hold_days","AvgHold"),
    ]:
        av = r22_a.get(key, 0.0); bv = r22_b.get(key, 0.0)
        w(f"| {label} | {av:.3f} | {bv:.3f} | {bv - av:+.4f} |")
    ra22 = _rsr(r22_a); rb22 = _rsr(r22_b)
    w(f"| RSR_EXIT件数 | {ra22} | {rb22} | {rb22 - ra22:+d} |")
    w(f"| N_trades | {r22_a.get('n_trades',0)} | {r22_b.get('n_trades',0)} "
      f"| {r22_b.get('n_trades',0) - r22_a.get('n_trades',0):+d} |")
    w(f"\n**Fold2 ΔCalmar = {fold2_dc:+.4f}  基準 ≥ {FOLD2_DCALMAR_MIN} → "
      f"{'✅ PASS' if crit_2022 else '❌ FAIL（即REJECT）'}**")

    # ── 6. WF Summary ─────────────────────────────────────────────────────
    w("\n---\n## 6. Walk-Forward 統計サマリー\n")
    w("| 指標 | 値 | 採用基準 | 判定 |")
    w("|---|---|---|---|")
    w(f"| WF勝率 (ΔCalmar>0) | **{wins}/5** ({wf_rate:.0%}) | ≥4/5 | {'✅' if crit_wf else '❌'} |")
    w(f"| 2022 Fold2 ΔCalmar | **{fold2_dc:+.4f}** | ≥{FOLD2_DCALMAR_MIN} | "
      f"{'✅' if crit_2022 else '❌ 恒久ゲート'} |")
    w(f"| 平均ΔCAGR | **{avg_dcagr:+.2f}pp** | ≥+{AVG_DCAGR_MIN_PP:.1f}pp | "
      f"{'✅' if crit_cagr else '❌'} |")
    w(f"| 平均ΔCalmar | **{avg_dc:+.4f}** | >0 | {'✅' if avg_dc > 0 else '❌'} |")
    best  = max(range(len(fold_rows)), key=lambda j: deltas_calmar[j])
    worst = min(range(len(fold_rows)), key=lambda j: deltas_calmar[j])
    w(f"| 最良Fold | {fold_rows[best]['fold']} ({fold_rows[best]['test_year']}) "
      f"ΔCal={deltas_calmar[best]:+.4f} | — | — |")
    w(f"| 最悪Fold | {fold_rows[worst]['fold']} ({fold_rows[worst]['test_year']}) "
      f"ΔCal={deltas_calmar[worst]:+.4f} | — | — |")

    # ── 7. Verdict ────────────────────────────────────────────────────────
    w("\n---\n## 7. 最終判定\n")
    w(f"### **{verdict}**\n")
    w(f"{verdict_note}\n")
    if all_pass:
        w("**採用時の次手 (⚠ ASK_FIRST_ON_CHANGE 必須)**:")
        w("- `signal_bridge.py` / `run_live_signal.py` に P4 ロジックを組込み")
        w("- `capital_allocation_abc.run_period(hold_policy=\"P4\")` が backtest hooK")
    else:
        w("**次の研究候補**:")
        w("- P5 (トレイル・アンカー) または P8 (部分決済)")
        w("- 現行 rsr_exit=70 を維持してロードマップ B2 (pos cap equity連動) へ")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(L), encoding="utf-8")
    print(f"\n  レポート → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────


def main() -> int:
    cfg = load_strategy_config()

    print("=" * 72)
    print("  P4 含み益非対称 Exit — 5-Fold Walk-Forward Validation")
    print("  A=Baseline(rsr_exit=70)  /  B=P4(profit→eff_thr=65, else 70)")
    print(f"  採用条件: WF≥4/5  ∧  Fold2 ΔCal≥{FOLD2_DCALMAR_MIN}  ∧  ΔCAGR≥+{AVG_DCAGR_MIN_PP:.1f}pp")
    print("=" * 72)

    print("\n[1/2] データロード中 (RSR42 + TOPIX)...")
    (universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
     rsr_syms, topix_close, cfg) = load_data(cfg)

    print("\n[2/2] 5 Fold × A/B × IS+Test 実行中...")
    fold_rows: list[dict] = []
    for fold_name, is_start, is_end, test_start, test_end, char in FOLDS:
        print(f"\n  ── {fold_name} (Test={test_start[:4]} / {char}) ──")
        row = _run_fold(
            fold_name, is_start, is_end, test_start, test_end,
            universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
            topix_close, rsr_syms, cfg,
        )
        fold_rows.append(row)

    out_path = REPORTS_DIR / "exit_p4_walkforward.md"
    write_report(fold_rows, out_path)

    # ── Console summary ────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  ★ Walk-Forward Summary — P4 含み益非対称")
    print("=" * 72)
    hdr = f"  {'Fold':<6} {'Year':>5} {'特徴':>8} {'CAGR_A':>8} {'CAGR_B':>8} " \
          f"{'ΔCAGR':>7} {'Cal_A':>7} {'Cal_B':>7} {'ΔCal':>7} {'PF_A':>6} {'PF_B':>6} {'判定':>4}"
    print(hdr)
    print("  " + "-" * 88)
    wins2 = 0; dcals: list[float] = []; dcagrs: list[float] = []
    for i, row in enumerate(fold_rows):
        a = _m(row, "A", "Test"); b = _m(row, "B", "Test")
        dc    = _d(b, a, "calmar")
        dcagr = _d(b, a, "cagr")
        win   = dc > 0
        if win:
            wins2 += 1
        dcals.append(dc); dcagrs.append(dcagr)
        print(
            f"  {row['fold']:<6} {row['test_year']:>5} {FOLDS[i][5]:>8} "
            f"{a.get('cagr',0):>+7.1f}% {b.get('cagr',0):>+7.1f}% "
            f"{dcagr:>+6.2f}pp "
            f"{a.get('calmar',0):>7.3f} {b.get('calmar',0):>7.3f} "
            f"{dc:>+7.4f} "
            f"{a.get('profit_factor',0):>6.2f} {b.get('profit_factor',0):>6.2f} "
            f"{'✅' if win else '❌':>4}"
        )
    print("  " + "-" * 88)
    avg_dc2    = float(np.mean(dcals))
    avg_dcagr2 = float(np.mean(dcagrs))
    fold2_dc2  = dcals[1]
    crit_wf2   = wins2 >= 4
    crit_22    = fold2_dc2 >= FOLD2_DCALMAR_MIN
    crit_cg    = avg_dcagr2 >= AVG_DCAGR_MIN_PP
    all_ok     = crit_wf2 and crit_22 and crit_cg
    print(f"\n  WF勝率: {wins2}/5 ({'✅' if crit_wf2 else '❌'})  "
          f"ΔCal_avg: {avg_dc2:+.4f}  ΔCAGR_avg: {avg_dcagr2:+.2f}pp ({'✅' if crit_cg else '❌'})")
    print(f"  Fold2_2022 ΔCalmar: {fold2_dc2:+.4f}  基準≥{FOLD2_DCALMAR_MIN} ({'✅' if crit_22 else '❌ 恒久ゲート失敗'})")
    final_v = (
        "APPROVE ✅" if all_ok else
        "REJECT — 2022 Bear 恒久ゲート ❌" if not crit_22 else
        "REJECT — WF勝率不足 ❌" if not crit_wf2 else
        "REJECT — ΔCAGR不足 ❌"
    )
    print(f"\n  ★ 最終判定: {final_v}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
