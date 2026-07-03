"""
backtest/entry_timing_score30_wf.py

Entry Timing score>=30 Walk-Forward Audit (5-Fold)

Fold1 Test=2021 / Fold2 Test=2022 / Fold3 Test=2023
Fold4 Test=2024 / Fold5 Test=2025

比較:
  A = Baseline (ET完全OFF)
  B = ET score>=30 フィルター (et_enabled=True, et_min_score=30)

採用条件:
  WF勝率 >= 4/5
  平均ΔCalmar > 0
  平均ΔCAGR >= -0.5pp

出力: reports/entry_timing_score30_walkforward.md

Run:
    cd C:/ai-trading
    python src/backtest/entry_timing_score30_wf.py
"""

from __future__ import annotations

import sys, time, warnings
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

from src.config_loader import load_strategy_config
from src.backtest.capital_allocation_abc import load_data, run_period

REPORTS_DIR = Path("reports")

# 5-Fold定義: (fold_name, IS_start, IS_end, test_start, test_end)
FOLDS = [
    ("Fold1", "2018-01-01", "2020-12-31", "2021-01-01", "2021-12-31"),
    ("Fold2", "2018-01-01", "2021-12-31", "2022-01-01", "2022-12-31"),
    ("Fold3", "2018-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
    ("Fold4", "2018-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
    ("Fold5", "2018-01-01", "2024-12-31", "2025-01-01", "2025-12-31"),
]

ET_MIN_SCORE = 30.0

# 採用閾値
WF_WIN_RATE_MIN   = 4 / 5
AVG_DCALMAR_MIN   = 0.0
AVG_DCAGR_MIN_PP  = -0.5


def _run_fold(
    fold_name: str,
    is_start: str, is_end: str,
    test_start: str, test_end: str,
    universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
    topix_close, rsr_syms, cfg,
) -> dict:
    """1FoldのIS+Test, A/B両モードを実行。"""
    row: dict = {"fold": fold_name, "test_year": test_start[:4]}

    for mode, kwargs in [
        ("A", dict(et_enabled=False, et_min_score=0.0)),
        ("B", dict(et_enabled=True,  et_min_score=ET_MIN_SCORE)),
    ]:
        for period, start, end in [
            ("IS",   is_start,   is_end),
            ("Test", test_start, test_end),
        ]:
            label = f"{mode}_{period}"
            print(f"    [{fold_name}] Mode{mode} {period} {start}〜{end} ...", end=" ", flush=True)
            t0 = time.time()
            m = run_period(
                universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
                topix_close, rsr_syms, cfg,
                start=start, end=end,
                pattern="A",
                **kwargs,
            )
            elapsed = time.time() - t0
            print(
                f"CAGR={m.get('cagr', 0):+.1f}%  Sharpe={m.get('sharpe', 0):.3f}  "
                f"MaxDD={m.get('max_dd', 0):.1f}%  Calmar={m.get('calmar', 0):.3f}  "
                f"({elapsed:.1f}s)"
            )
            row[label] = m

    return row


def _fold_metrics(row: dict, mode: str, period: str) -> dict:
    return row.get(f"{mode}_{period}", {})


def _delta(b: dict, a: dict, key: str) -> float:
    return b.get(key, 0.0) - a.get(key, 0.0)


def write_report(fold_rows: list[dict], output_path: Path) -> None:
    L = []; w = L.append

    w("# Entry Timing score>=30 Walk-Forward Audit")
    w(f"\n作成日: {time.strftime('%Y-%m-%d')}  |  5-Fold Walk-Forward")
    w(f"\n**比較: A=Baseline (ET-OFF)  vs  B=ET score>={int(ET_MIN_SCORE)} フィルター**\n")
    w(f"**採用条件: WF勝率≥4/5  平均ΔCalmar>0  平均ΔCAGR≥{AVG_DCAGR_MIN_PP:.1f}pp**\n")

    # ── IS サマリー ───────────────────────────────────────────
    w("---\n## 1. IS Results (各Fold IS期間)\n")
    w("| Fold | IS期間 | CAGR_A | CAGR_B | ΔCAGR | Sharpe_A | Sharpe_B | ΔSharpe | Calmar_A | Calmar_B | ΔCalmar | MaxDD_A | MaxDD_B |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for row in fold_rows:
        a_is = _fold_metrics(row, "A", "IS")
        b_is = _fold_metrics(row, "B", "IS")
        fold_label = row["fold"]
        is_start = FOLDS[[r[0] for r in FOLDS].index(fold_label)][1]
        is_end   = FOLDS[[r[0] for r in FOLDS].index(fold_label)][2]
        w(
            f"| {fold_label} | {is_start[:4]}〜{is_end[:4]} "
            f"| {a_is.get('cagr',0):+.1f}% | {b_is.get('cagr',0):+.1f}% "
            f"| {_delta(b_is,a_is,'cagr'):+.2f}pp "
            f"| {a_is.get('sharpe',0):.3f} | {b_is.get('sharpe',0):.3f} "
            f"| {_delta(b_is,a_is,'sharpe'):+.4f} "
            f"| {a_is.get('calmar',0):.3f} | {b_is.get('calmar',0):.3f} "
            f"| {_delta(b_is,a_is,'calmar'):+.4f} "
            f"| {a_is.get('max_dd',0):.1f}% | {b_is.get('max_dd',0):.1f}% |"
        )

    # ── Test サマリー ─────────────────────────────────────────
    w("\n---\n## 2. Walk-Forward Test Results (OOS各年)\n")
    w("| Fold | Test年 | CAGR_A | CAGR_B | ΔCAGR | Sharpe_A | Sharpe_B | ΔSharpe | Calmar_A | Calmar_B | ΔCalmar | MaxDD_A | MaxDD_B | 判定 |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    wins = 0
    deltas_calmar = []
    deltas_cagr   = []
    for row in fold_rows:
        a_t = _fold_metrics(row, "A", "Test")
        b_t = _fold_metrics(row, "B", "Test")
        dc  = _delta(b_t, a_t, "calmar")
        dcagr = _delta(b_t, a_t, "cagr")
        win = dc > 0
        if win:
            wins += 1
        deltas_calmar.append(dc)
        deltas_cagr.append(dcagr)
        verdict = "✅" if win else "❌"
        w(
            f"| {row['fold']} | {row['test_year']} "
            f"| {a_t.get('cagr',0):+.1f}% | {b_t.get('cagr',0):+.1f}% "
            f"| {dcagr:+.2f}pp "
            f"| {a_t.get('sharpe',0):.3f} | {b_t.get('sharpe',0):.3f} "
            f"| {_delta(b_t,a_t,'sharpe'):+.4f} "
            f"| {a_t.get('calmar',0):.3f} | {b_t.get('calmar',0):.3f} "
            f"| {dc:+.4f} "
            f"| {a_t.get('max_dd',0):.1f}% | {b_t.get('max_dd',0):.1f}% "
            f"| {verdict} |"
        )

    avg_dc    = float(np.mean(deltas_calmar))
    avg_dcagr = float(np.mean(deltas_cagr))
    wf_rate   = wins / len(fold_rows)

    # ── Exposure / 追加メトリクス ─────────────────────────────
    w("\n---\n## 3. Blocked Signals & Exposure (Test期間)\n")
    w("| Fold | Test年 | Exposure_A | Exposure_B | ΔExp | ET_Blocked | Blocked率 | Missed_A | Missed_B | AvgHold_A | AvgHold_B |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for row in fold_rows:
        a_t = _fold_metrics(row, "A", "Test")
        b_t = _fold_metrics(row, "B", "Test")
        # ET blocked count and rate
        et_blocked = b_t.get("et_blocked_count", 0)
        n_buys_a   = len([t for t in [a_t] if t])
        # blocked率: et_blocked / (et_blocked + B's BUY trades)
        b_trades   = b_t.get("n_trades", 0) + et_blocked
        blocked_rate = (et_blocked / max(1, b_trades)) * 100
        w(
            f"| {row['fold']} | {row['test_year']} "
            f"| {a_t.get('avg_exposure',0):.1f}% | {b_t.get('avg_exposure',0):.1f}% "
            f"| {_delta(b_t,a_t,'avg_exposure'):+.1f}pp "
            f"| {et_blocked} | {blocked_rate:.1f}% "
            f"| {a_t.get('missed_signals',0)} | {b_t.get('missed_signals',0)} "
            f"| {a_t.get('avg_hold_days',0):.1f}d | {b_t.get('avg_hold_days',0):.1f}d |"
        )

    # ── 年次リターン比較 ──────────────────────────────────────
    w("\n---\n## 4. 年次リターン比較 (Test期間)\n")
    w("| 年 | CAGR_A | CAGR_B | ΔCAGR | Sharpe_A | Sharpe_B | Calmar_A | Calmar_B |")
    w("|---|---|---|---|---|---|---|---|")
    for row in fold_rows:
        a_t = _fold_metrics(row, "A", "Test")
        b_t = _fold_metrics(row, "B", "Test")
        w(
            f"| {row['test_year']} "
            f"| {a_t.get('cagr',0):+.1f}% | {b_t.get('cagr',0):+.1f}% "
            f"| {_delta(b_t,a_t,'cagr'):+.2f}pp "
            f"| {a_t.get('sharpe',0):.3f} | {b_t.get('sharpe',0):.3f} "
            f"| {a_t.get('calmar',0):.3f} | {b_t.get('calmar',0):.3f} |"
        )

    # ── 統計サマリー ──────────────────────────────────────────
    w("\n---\n## 5. Walk-Forward 統計サマリー\n")
    w(f"| 指標 | 値 | 採用基準 | 判定 |")
    w("|---|---|---|---|")
    crit1 = wf_rate >= WF_WIN_RATE_MIN
    crit2 = avg_dc > AVG_DCALMAR_MIN
    crit3 = avg_dcagr >= AVG_DCAGR_MIN_PP
    w(f"| WF勝率 (Calmar_B > Calmar_A) | **{wins}/{len(fold_rows)}** ({wf_rate:.0%}) | ≥4/5 (80%) | {'✅' if crit1 else '❌'} |")
    w(f"| 平均ΔCalmar (B-A) | **{avg_dc:+.4f}** | >0 | {'✅' if crit2 else '❌'} |")
    w(f"| 平均ΔCAGR (B-A) | **{avg_dcagr:+.2f}pp** | ≥-0.5pp | {'✅' if crit3 else '❌'} |")

    # 最悪・最良Fold
    best_fold  = max(fold_rows, key=lambda r: _delta(_fold_metrics(r,"B","Test"), _fold_metrics(r,"A","Test"), "calmar"))
    worst_fold = min(fold_rows, key=lambda r: _delta(_fold_metrics(r,"B","Test"), _fold_metrics(r,"A","Test"), "calmar"))
    w(f"| 最良Fold | {best_fold['fold']} ({best_fold['test_year']}) ΔCalmar={_delta(_fold_metrics(best_fold,'B','Test'),_fold_metrics(best_fold,'A','Test'),'calmar'):+.4f} | — | — |")
    w(f"| 最悪Fold | {worst_fold['fold']} ({worst_fold['test_year']}) ΔCalmar={_delta(_fold_metrics(worst_fold,'B','Test'),_fold_metrics(worst_fold,'A','Test'),'calmar'):+.4f} | — | — |")

    # ── 最終判定 ──────────────────────────────────────────────
    all_pass  = crit1 and crit2 and crit3
    all_fail  = not crit1 and not crit2 and not crit3
    crit_count = sum([crit1, crit2, crit3])

    if all_pass:
        verdict_label = "**APPROVE**"
        verdict_reason = (
            f"全採用条件クリア: WF勝率{wins}/{len(fold_rows)}, "
            f"平均ΔCalmar={avg_dc:+.4f}, 平均ΔCAGR={avg_dcagr:+.2f}pp"
        )
    elif crit_count >= 2:
        verdict_label = "**REVIEW**"
        verdict_reason = (
            f"条件{crit_count}/3クリア。不足条件: "
            + ([] if crit1 else [f"WF勝率{wins}/{len(fold_rows)}<4/5"])
            + ([] if crit2 else [f"ΔCalmar={avg_dc:+.4f}≤0"])
            + ([] if crit3 else [f"ΔCAGR={avg_dcagr:+.2f}pp<-0.5pp"])
        )
        verdict_reason = (
            f"条件{crit_count}/3クリア — "
            + ", ".join(
                filter(None, [
                    f"WF={wins}/{len(fold_rows)}({'OK' if crit1 else 'NG'})",
                    f"ΔCalmar={avg_dc:+.4f}({'OK' if crit2 else 'NG'})",
                    f"ΔCAGR={avg_dcagr:+.2f}pp({'OK' if crit3 else 'NG'})",
                ])
            )
        )
    else:
        verdict_label = "**REJECT**"
        verdict_reason = (
            f"採用条件不足: WF勝率{wins}/{len(fold_rows)}, "
            f"平均ΔCalmar={avg_dc:+.4f}, 平均ΔCAGR={avg_dcagr:+.2f}pp"
        )

    w(f"\n---\n## 6. 最終判定\n")
    w(f"### {verdict_label}\n")
    w(f"{verdict_reason}\n")

    if all_pass:
        w("**→ score>=30 を本番エントリーフィルターとして採用可能**")
        w(f"\n採用後の期待効果:")
        w(f"- 平均ΔCalmar: {avg_dc:+.4f}")
        w(f"- 平均ΔCAGR: {avg_dcagr:+.2f}pp")
        w(f"- WF勝率: {wins}/{len(fold_rows)} ({wf_rate:.0%})")
    elif crit_count >= 2:
        w("**→ score>=30 の本番採用には追加検証が必要**")
        if not crit1:
            w(f"- WF勝率が{wins}/{len(fold_rows)}で基準4/5未達。不安定Foldの調査が必要。")
        if not crit2:
            w(f"- 平均ΔCalmar={avg_dc:+.4f}がマイナス。フィルターが有効なリターンを阻害。")
        if not crit3:
            w(f"- 平均ΔCAGR={avg_dcagr:+.2f}ppが-0.5pp未満。CAGR損失大。")
    else:
        w("**→ score>=30 は本番エントリーフィルターとして採用不可**")
        w(f"\n根拠:")
        w(f"- WF勝率: {wins}/{len(fold_rows)} ({'基準クリア' if crit1 else '基準未達'})")
        w(f"- 平均ΔCalmar: {avg_dc:+.4f} ({'正' if crit2 else 'ゼロ以下'})")
        w(f"- 平均ΔCAGR: {avg_dcagr:+.2f}pp ({'基準内' if crit3 else '損失過大'})")

    w("\n---\n## 7. IS参照値 (確認用)\n")
    w("| 指標 | Baseline | score>=30 | Δ |")
    w("|---|---|---|---|")
    w("| CAGR | +18.1% | +17.9% | -0.2pp |")
    w("| Calmar | 1.087 | 1.169 | +0.082 |")
    w("| Sharpe | 0.779 | 0.812 | +0.033 |")
    w("| MaxDD | -16.7% | -15.3% | +1.4pp |")
    w("| Exposure | 35.0% | 34.0% | -1.0pp |")
    w("| Missed | 371 | 294 | -77 |")
    w("\n*出典: entry_timing_predictive_power_audit.md / Threshold Sweep*")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(L), encoding="utf-8")
    print(f"\n  レポート保存: {output_path}")


def main() -> int:
    cfg = load_strategy_config()

    print("=" * 72)
    print("  Entry Timing score>=30 Walk-Forward Audit — 5-Fold")
    print("  A=Baseline (ET-OFF)  /  B=ET score>=30 フィルター")
    print("=" * 72)

    print("\n[1/2] データロード中...")
    (universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
     rsr_syms, topix_close, cfg) = load_data(cfg)

    print("\n[2/2] 5 Fold × 2モード × IS+Test 実行中...")
    fold_rows: list[dict] = []
    for fold_name, is_start, is_end, test_start, test_end in FOLDS:
        print(f"\n  ── {fold_name} (Test={test_start[:4]}) ──")
        row = _run_fold(
            fold_name, is_start, is_end, test_start, test_end,
            universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
            topix_close, rsr_syms, cfg,
        )
        fold_rows.append(row)

    out_path = REPORTS_DIR / "entry_timing_score30_walkforward.md"
    write_report(fold_rows, out_path)

    # ── Console summary ────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  ★ Walk-Forward Summary")
    print("=" * 72)
    print(f"\n  {'Fold':<6} {'Year':>5} {'CAGR_A':>8} {'CAGR_B':>8} {'ΔCAGR':>7} "
          f"{'Cal_A':>7} {'Cal_B':>7} {'ΔCal':>7} {'判定':>4}")
    print("  " + "-" * 66)
    wins = 0; deltas_c = []; deltas_cagr = []
    for row in fold_rows:
        a = _fold_metrics(row, "A", "Test")
        b = _fold_metrics(row, "B", "Test")
        dc = _delta(b, a, "calmar")
        dcagr = _delta(b, a, "cagr")
        win = dc > 0
        if win: wins += 1
        deltas_c.append(dc); deltas_cagr.append(dcagr)
        print(
            f"  {row['fold']:<6} {row['test_year']:>5} "
            f"{a.get('cagr',0):>+7.1f}% {b.get('cagr',0):>+7.1f}% "
            f"{dcagr:>+6.2f}pp "
            f"{a.get('calmar',0):>7.3f} {b.get('calmar',0):>7.3f} "
            f"{dc:>+7.4f} {'✅' if win else '❌':>4}"
        )
    avg_dc    = float(np.mean(deltas_c))
    avg_dcagr = float(np.mean(deltas_cagr))
    print("  " + "-" * 66)
    print(f"  {'平均':>12} {avg_dcagr:>+14.2f}pp {avg_dc:>+22.4f}")
    wf_rate = wins / len(fold_rows)
    crit1 = wf_rate >= WF_WIN_RATE_MIN
    crit2 = avg_dc > AVG_DCALMAR_MIN
    crit3 = avg_dcagr >= AVG_DCAGR_MIN_PP
    all_pass = crit1 and crit2 and crit3
    crit_count = sum([crit1, crit2, crit3])
    verdict = "APPROVE" if all_pass else ("REVIEW" if crit_count >= 2 else "REJECT")
    print(f"\n  WF勝率: {wins}/{len(fold_rows)}  ΔCalmar_avg: {avg_dc:+.4f}  ΔCAGR_avg: {avg_dcagr:+.2f}pp")
    print(f"\n  ★ 最終判定: {verdict}")
    print(f"\n  レポート → {out_path}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
