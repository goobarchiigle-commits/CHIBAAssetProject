"""
backtest/rsr_exit_threshold_sweep.py

RSR Exit Threshold Sweep — Walk-Forward 5-Fold

比較: rsr_exit = 70 / 71 / 72 / 73 / 74 / 75 (Pattern A = 現行 75)

Fold定義:
  Fold1: Train=2018-2020  Test=2021
  Fold2: Train=2018-2021  Test=2022 (弱気)
  Fold3: Train=2018-2022  Test=2023 (強気)
  Fold4: Train=2018-2023  Test=2024
  Fold5: Train=2018-2024  Test=2025

出力: reports/rsr_exit_threshold_sweep.md

Run:
    cd C:/ai-trading
    python src/backtest/rsr_exit_threshold_sweep.py
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

THRESHOLDS = [70, 71, 72, 73, 74, 75]
BASELINE   = 75

FOLDS = [
    ("Fold1", "2018-01-01", "2020-12-31", "2021-01-01", "2021-12-31", "2021弱年"),
    ("Fold2", "2018-01-01", "2021-12-31", "2022-01-01", "2022-12-31", "2022弱気"),
    ("Fold3", "2018-01-01", "2022-12-31", "2023-01-01", "2023-12-31", "2023強気"),
    ("Fold4", "2018-01-01", "2023-12-31", "2024-01-01", "2024-12-31", "2024"),
    ("Fold5", "2018-01-01", "2024-12-31", "2025-01-01", "2025-12-31", "2025"),
]


def _g(m, key, default=0.0):
    return m.get(key, default)


def run_all_folds(universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
                  topix_close, rsr_syms, cfg) -> dict:
    """
    Returns:
        results[thr][fold_idx] = {"IS": metrics, "Test": metrics}
    """
    results: dict = {thr: [] for thr in THRESHOLDS}
    total = len(THRESHOLDS) * len(FOLDS)
    done = 0

    for fi, (fold_name, is_start, is_end, test_start, test_end, char) in enumerate(FOLDS):
        print(f"\n── {fold_name} (Test={test_start[:4]} / {char}) ──")
        for thr in THRESHOLDS:
            row: dict = {}
            for period, start, end in [("IS", is_start, is_end), ("Test", test_start, test_end)]:
                done += 1
                tag = f"[{done:2d}/{total * 2}] thr={thr} {fold_name} {period}"
                print(f"  {tag}...", end=" ", flush=True)
                t0 = time.time()
                m = run_period(
                    universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
                    topix_close, rsr_syms, cfg,
                    start=start, end=end,
                    pattern="A",
                    rsr_exit_override=float(thr),
                )
                print(f"CAGR={m.get('cagr',0):+.1f}%  "
                      f"Sharpe={m.get('sharpe',0):.3f}  "
                      f"MaxDD={m.get('max_dd',0):.1f}%  "
                      f"({time.time()-t0:.1f}s)")
                row[period] = m
            results[thr].append(row)
    return results


def _aggregate(results: dict, thr: int) -> dict:
    """5 fold Test 平均指標 + WF 勝率（vs baseline=75）"""
    rows = results[thr]
    base_rows = results[BASELINE]

    cagrs   = [_g(r["Test"], "cagr")    for r in rows]
    sharpes = [_g(r["Test"], "sharpe")  for r in rows]
    maxdds  = [_g(r["Test"], "max_dd")  for r in rows]
    calmars = [_g(r["Test"], "calmar")  for r in rows]

    wins = sum(
        1 for r, br in zip(rows, base_rows)
        if _g(r["Test"], "calmar") > _g(br["Test"], "calmar")
    ) if thr != BASELINE else None  # baseline against itself = N/A

    return {
        "avg_cagr":   float(np.mean(cagrs)),
        "avg_sharpe": float(np.mean(sharpes)),
        "avg_maxdd":  float(np.mean(maxdds)),
        "avg_calmar": float(np.mean(calmars)),
        "wf_wins":    wins,
        "cagrs":      cagrs,
        "sharpes":    sharpes,
        "maxdds":     maxdds,
        "calmars":    calmars,
    }


def write_report(results: dict, output_path: Path) -> None:
    aggs = {thr: _aggregate(results, thr) for thr in THRESHOLDS}

    # Rank by avg_calmar descending
    ranked = sorted(THRESHOLDS, key=lambda t: aggs[t]["avg_calmar"], reverse=True)

    L = []; w = L.append

    w("# RSR Exit Threshold Sweep — Walk-Forward 5-Fold")
    w(f"\n作成日: {time.strftime('%Y-%m-%d')}  |  Pattern A Baseline = rsr_exit=75")
    w("\n**比較対象**: rsr_exit = 70 / 71 / 72 / 73 / 74 / 75")
    w("**WF勝率基準**: vs Baseline (rsr_exit=75) でCalmar勝ちFold数 / 5")
    w("")

    # ─────────────────────────────────────────────
    # 1. 総合順位表
    # ─────────────────────────────────────────────
    w("---\n## 1. 総合順位表（WF 5-Fold 平均）\n")
    w("| 順位 | rsr_exit | avgCAGR | avgSharpe | avgMaxDD | avgCalmar | WF勝率(vs 75) |")
    w("|---|---|---|---|---|---|---|")
    for rank, thr in enumerate(ranked, 1):
        ag = aggs[thr]
        wf_str = f"{ag['wf_wins']}/5" if ag["wf_wins"] is not None else "N/A (Baseline)"
        marker = " **← BEST**" if rank == 1 else (" ← Baseline" if thr == BASELINE else "")
        w(f"| {rank} | **{thr}**{marker} "
          f"| {ag['avg_cagr']:+.2f}% "
          f"| {ag['avg_sharpe']:.3f} "
          f"| {ag['avg_maxdd']:.1f}% "
          f"| {ag['avg_calmar']:.3f} "
          f"| {wf_str} |")

    # ─────────────────────────────────────────────
    # 2. Fold別 Test CAGR マトリックス
    # ─────────────────────────────────────────────
    w("\n---\n## 2. Fold別 Test CAGR\n")
    fold_years = [f[3][:4] for f in FOLDS]
    fold_chars = [f[5] for f in FOLDS]
    w("| rsr_exit | " + " | ".join(fold_years) + " | avg |")
    w("|---|" + "---|" * (len(FOLDS) + 1))
    for thr in THRESHOLDS:
        cagrs = aggs[thr]["cagrs"]
        row_str = " | ".join(f"{c:+.1f}%" for c in cagrs)
        marker = " **←** " if thr == ranked[0] else ""
        w(f"| {thr}{marker} | {row_str} | {aggs[thr]['avg_cagr']:+.2f}% |")

    # ─────────────────────────────────────────────
    # 3. Fold別 Test Calmar マトリックス
    # ─────────────────────────────────────────────
    w("\n---\n## 3. Fold別 Test Calmar\n")
    w("| rsr_exit | " + " | ".join(fold_years) + " | avg |")
    w("|---|" + "---|" * (len(FOLDS) + 1))
    for thr in THRESHOLDS:
        calmars = aggs[thr]["calmars"]
        row_str = " | ".join(f"{c:.3f}" for c in calmars)
        w(f"| {thr} | {row_str} | {aggs[thr]['avg_calmar']:.3f} |")

    # ─────────────────────────────────────────────
    # 4. Fold別 Test Sharpe マトリックス
    # ─────────────────────────────────────────────
    w("\n---\n## 4. Fold別 Test Sharpe\n")
    w("| rsr_exit | " + " | ".join(fold_years) + " | avg |")
    w("|---|" + "---|" * (len(FOLDS) + 1))
    for thr in THRESHOLDS:
        sharpes = aggs[thr]["sharpes"]
        row_str = " | ".join(f"{s:.3f}" for s in sharpes)
        w(f"| {thr} | {row_str} | {aggs[thr]['avg_sharpe']:.3f} |")

    # ─────────────────────────────────────────────
    # 5. Fold別 Test MaxDD マトリックス
    # ─────────────────────────────────────────────
    w("\n---\n## 5. Fold別 Test MaxDD\n")
    w("| rsr_exit | " + " | ".join(fold_years) + " | avg |")
    w("|---|" + "---|" * (len(FOLDS) + 1))
    for thr in THRESHOLDS:
        maxdds = aggs[thr]["maxdds"]
        row_str = " | ".join(f"{d:.1f}%" for d in maxdds)
        w(f"| {thr} | {row_str} | {aggs[thr]['avg_maxdd']:.1f}% |")

    # ─────────────────────────────────────────────
    # 6. 2022年単独分析（必須）
    # ─────────────────────────────────────────────
    w("\n---\n## 6. 2022年単独分析（弱気相場 / Fold2）\n")
    w("2022年 = 金利ショック年。RSR緩和が逆効果になりやすい相場。\n")
    w("| rsr_exit | CAGR_2022 | MaxDD_2022 | Calmar_2022 | Sharpe_2022 | 判定 |")
    w("|---|---|---|---|---|---|")

    fold2_idx = 1  # Fold2 = 2022
    best_2022_calmar = max(aggs[thr]["calmars"][fold2_idx] for thr in THRESHOLDS)
    worst_2022_calmar = min(aggs[thr]["calmars"][fold2_idx] for thr in THRESHOLDS)

    for thr in THRESHOLDS:
        m22 = results[thr][fold2_idx]["Test"]
        cal = _g(m22, "calmar")
        cagr = _g(m22, "cagr")
        dd   = _g(m22, "max_dd")
        sh   = _g(m22, "sharpe")
        if cal == best_2022_calmar:
            verdict = "✅ 最良"
        elif cal == worst_2022_calmar:
            verdict = "❌ 最悪"
        else:
            verdict = "—"
        baseline_cal = aggs[BASELINE]["calmars"][fold2_idx]
        if thr != BASELINE:
            delta = cal - baseline_cal
            verdict += f" Δ{delta:+.3f}"
        w(f"| {thr} | {cagr:+.1f}% | {dd:.1f}% | {cal:.3f} | {sh:.3f} | {verdict} |")

    # ─────────────────────────────────────────────
    # 7. 全Fold × 全閾値 詳細テーブル
    # ─────────────────────────────────────────────
    w("\n---\n## 7. 全Fold × 全閾値 詳細 (Test期間)\n")
    for fi, (fold_name, is_start, is_end, test_start, test_end, char) in enumerate(FOLDS):
        w(f"\n### {fold_name} — Test={test_start[:4]} ({char})\n")
        w("| rsr_exit | CAGR | Sharpe | MaxDD | Calmar | RSR_EXIT数 |")
        w("|---|---|---|---|---|---|")
        base_calmar = _g(results[BASELINE][fi]["Test"], "calmar")
        for thr in THRESHOLDS:
            m = results[thr][fi]["Test"]
            cagr = _g(m, "cagr"); sh = _g(m, "sharpe")
            dd   = _g(m, "max_dd"); cal = _g(m, "calmar")
            rsr_cnt = m.get("exit_reasons", {}).get("RSR_EXIT", 0)
            win_mark = ""
            if thr != BASELINE:
                win_mark = " ✅" if cal > base_calmar else " ❌"
            w(f"| {thr}{win_mark} | {cagr:+.1f}% | {sh:.3f} | {dd:.1f}% | {cal:.3f} | {rsr_cnt} |")

    # ─────────────────────────────────────────────
    # 8. IS参照値 (Fold5 = Full IS 2018-2024)
    # ─────────────────────────────────────────────
    w("\n---\n## 8. IS参照値（Fold5 IS: 2018-2024）\n")
    w("| rsr_exit | CAGR | Sharpe | MaxDD | Calmar |")
    w("|---|---|---|---|---|")
    fi5 = 4  # Fold5 index
    for thr in THRESHOLDS:
        m = results[thr][fi5]["IS"]
        w(f"| {thr} | {_g(m,'cagr'):+.2f}% | {_g(m,'sharpe'):.3f} | {_g(m,'max_dd'):.1f}% | {_g(m,'calmar'):.3f} |")

    # ─────────────────────────────────────────────
    # 9. 結論
    # ─────────────────────────────────────────────
    w("\n---\n## 9. 結論\n")

    best_thr   = ranked[0]
    second_thr = ranked[1]
    best_ag    = aggs[best_thr]
    base_ag    = aggs[BASELINE]

    delta_cagr   = best_ag["avg_cagr"]   - base_ag["avg_cagr"]
    delta_sharpe = best_ag["avg_sharpe"] - base_ag["avg_sharpe"]
    delta_calmar = best_ag["avg_calmar"] - base_ag["avg_calmar"]
    delta_maxdd  = best_ag["avg_maxdd"]  - base_ag["avg_maxdd"]

    w("### 各閾値の総合評価\n")
    w("| 閾値 | 総合順位 | avgCalmar | avgCAGR | 2022Calmar | WF勝率(vs75) | 評価 |")
    w("|---|---|---|---|---|---|---|")
    for rank, thr in enumerate(ranked, 1):
        ag = aggs[thr]
        cal22 = aggs[thr]["calmars"][1]
        wf_str = f"{ag['wf_wins']}/5" if ag["wf_wins"] is not None else "N/A"
        if rank == 1 and thr != BASELINE:
            eval_str = "★ 最優秀"
        elif thr == BASELINE:
            eval_str = "現行基準"
        elif rank <= 2:
            eval_str = "○ 優秀"
        else:
            eval_str = "△ 可"
        w(f"| {thr} | {rank} | {ag['avg_calmar']:.3f} | {ag['avg_cagr']:+.2f}% | {cal22:.3f} | {wf_str} | {eval_str} |")

    w(f"\n### 最適値: **rsr_exit = {best_thr}**\n")
    if best_thr == BASELINE:
        w("**結論**: 現行 rsr_exit=75 が最良。閾値変更の効果なし。")
        w(f"\n本番採用推奨値: **75 (現行維持)**")
        w("\n採用理由:")
        w("- 全閾値比較でavgCalmar最大")
        w("- 変更コスト（WF再検証・本番変更）不要")
    else:
        w(f"**vs Baseline (rsr_exit=75)**:")
        w(f"- avg CAGR: {base_ag['avg_cagr']:+.2f}% → {best_ag['avg_cagr']:+.2f}% (Δ{delta_cagr:+.2f}pp)")
        w(f"- avg Sharpe: {base_ag['avg_sharpe']:.3f} → {best_ag['avg_sharpe']:.3f} (Δ{delta_sharpe:+.4f})")
        w(f"- avg Calmar: {base_ag['avg_calmar']:.3f} → {best_ag['avg_calmar']:.3f} (Δ{delta_calmar:+.4f})")
        w(f"- avg MaxDD: {base_ag['avg_maxdd']:.1f}% → {best_ag['avg_maxdd']:.1f}% (Δ{delta_maxdd:+.2f}pp)")

        wf_val = best_ag["wf_wins"]
        wf_ok  = wf_val is not None and wf_val >= 4

        # 2022 risk assessment
        cal22_best = aggs[best_thr]["calmars"][1]
        cal22_base = aggs[BASELINE]["calmars"][1]
        bear_risk  = cal22_best < cal22_base

        w(f"\n### 本番採用推奨値\n")
        if wf_ok and not bear_risk:
            w(f"**→ rsr_exit = {best_thr}  (無条件採用可)**\n")
            w("採用理由:")
            w(f"- WF勝率 {wf_val}/5 クリア（≥4/5）")
            w(f"- 2022年弱気相場でも Baseline 以上 (ΔCalmar={cal22_best-cal22_base:+.3f})")
            w(f"- avgCalmar改善: +{delta_calmar:.4f}")
        elif wf_ok and bear_risk:
            w(f"**→ rsr_exit = {best_thr}  (条件付き採用: Bull レジーム限定)**\n")
            w("採用理由:")
            w(f"- WF勝率 {wf_val}/5 クリア")
            w(f"- ⚠ 2022年弱気相場では Baseline より悪化 (ΔCalmar={cal22_best-cal22_base:+.3f})")
            w(f"- avgCalmar改善: +{delta_calmar:.4f}")
            w("\n推奨実装:")
            w(f"- TOPIX > MA200 (Bull) 時のみ rsr_exit={best_thr}")
            w("- Bear 時は rsr_exit=75 維持")
        else:
            w(f"**→ rsr_exit = 75  (現行維持を推奨)**\n")
            w("理由:")
            w(f"- WF勝率 {wf_val}/5 — 基準 4/5 未達" if not wf_ok else "- WF勝率OK")
            if bear_risk:
                w(f"- 2022年弱気相場で悪化 (ΔCalmar={cal22_best-cal22_base:+.3f})")
            w("- 改善が不安定で本番リスクに見合わない")

    w("\n---\n*生成: src/backtest/rsr_exit_threshold_sweep.py*")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(L), encoding="utf-8")
    print(f"\n  レポート保存: {output_path}")


def main() -> int:
    cfg = load_strategy_config()

    print("=" * 72)
    print("  RSR Exit Threshold Sweep — 5-Fold Walk-Forward")
    print(f"  閾値: {THRESHOLDS}  /  Baseline={BASELINE}")
    print(f"  総実行数: {len(THRESHOLDS)} × {len(FOLDS)} × 2 = {len(THRESHOLDS)*len(FOLDS)*2} runs")
    print("=" * 72)

    print("\n[1/2] データロード中...")
    (universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
     rsr_syms, topix_close, cfg) = load_data(cfg)

    print("\n[2/2] 全Fold × 全閾値 実行中...")
    results = run_all_folds(
        universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
        topix_close, rsr_syms, cfg,
    )

    out_path = REPORTS_DIR / "rsr_exit_threshold_sweep.md"
    write_report(results, out_path)

    # ── Console 順位サマリー ───────────────────────────────────────────
    aggs = {thr: _aggregate(results, thr) for thr in THRESHOLDS}
    ranked = sorted(THRESHOLDS, key=lambda t: aggs[t]["avg_calmar"], reverse=True)

    print("\n" + "=" * 72)
    print("  ★ RSR Exit Threshold Sweep Summary")
    print("=" * 72)
    print(f"\n  {'順位':>2}  {'rsr_exit':>8}  {'avgCAGR':>9}  {'avgSharpe':>10}  "
          f"{'avgMaxDD':>9}  {'avgCalmar':>10}  {'WF(vs75)':>9}")
    print("  " + "-" * 68)
    for rank, thr in enumerate(ranked, 1):
        ag = aggs[thr]
        wf_str = f"{ag['wf_wins']}/5" if ag["wf_wins"] is not None else "N/A"
        mark = " ★" if rank == 1 else ("  (Base)" if thr == BASELINE else "")
        print(f"  {rank:2d}  {thr:>8}{mark}  {ag['avg_cagr']:>+8.2f}%  "
              f"{ag['avg_sharpe']:>10.3f}  {ag['avg_maxdd']:>8.1f}%  "
              f"{ag['avg_calmar']:>10.3f}  {wf_str:>9}")

    best = ranked[0]
    print(f"\n  最適値: rsr_exit={best}")
    print(f"  レポート → {out_path}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
