"""
backtest/variant_d_walkforward.py

Variant D Walk-Forward Validation
5-fold WF: Train 3y / Test 1y / Rolling — 2018-2025

A: max_positions=3          (現行)
D: max_positions=4 + 4th RSR≥85

Run:
    cd C:/ai-trading
    python src/backtest/variant_d_walkforward.py
"""

from __future__ import annotations

import sys, time, warnings
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

from src.config_loader import load_strategy_config
from src.backtest.capital_allocation_abc import load_data
from src.backtest.conditional_4th_position_audit import (
    run_variant, extract_4th_trades, quality_stats,
)

REPORTS_DIR = Path("reports")

# ── 5-Fold Walk-Forward schedule ──────────────────────────────────────
# train_end = test_start - 1 day (implicit)
FOLDS = [
    ("Fold1", "2018-01-01", "2020-12-31", "2021-01-01", "2021-12-31"),
    ("Fold2", "2019-01-01", "2021-12-31", "2022-01-01", "2022-12-31"),
    ("Fold3", "2020-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
    ("Fold4", "2021-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
    ("Fold5", "2022-01-01", "2024-12-31", "2025-01-01", "2025-12-31"),
]

# primary verdict metric — Calmar balances CAGR and risk
PRIMARY_METRIC = "calmar"


# ─────────────────────────────────────────────────────────────────────
#  RUN ALL FOLDS
# ─────────────────────────────────────────────────────────────────────

def run_wf(universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
           topix_close, rsr_syms, cfg) -> dict:
    results = {}   # fold_name → {"A": metrics, "D": metrics, "D_run": full_res}

    for fold_name, train_start, train_end, test_start, test_end in FOLDS:
        print(f"\n  {fold_name}  test={test_start}〜{test_end}")

        for variant, kw in [
            ("A", dict(max_pos=3, rsr_4th=0.0,  conv_mode="none")),
            ("D", dict(max_pos=4, rsr_4th=85.0, conv_mode="none")),
        ]:
            t0  = time.time()
            res = run_variant(
                universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
                topix_close, rsr_syms, cfg,
                start=test_start, end=test_end, **kw
            )
            m = res.get("metrics", {})
            print(f"    {variant}: CAGR={m.get('cagr',0):+.1f}%  "
                  f"Sharpe={m.get('sharpe',0):.3f}  "
                  f"MaxDD={m.get('max_dd',0):.1f}%  "
                  f"Calmar={m.get('calmar',0):.3f}  "
                  f"4th_days={m.get('n4_days_pct',0):.1f}%  "
                  f"blocked={m.get('blocked_4th',0)}  ({time.time()-t0:.1f}s)")

            fold = results.setdefault(fold_name, {
                "test_start": test_start, "test_end": test_end
            })
            fold[variant]         = m
            fold[f"{variant}_run"]= res   # keep full run for 4th-slot analysis

    return results


# ─────────────────────────────────────────────────────────────────────
#  ANALYSIS
# ─────────────────────────────────────────────────────────────────────

def analyze(results: dict) -> dict:
    fold_names = [f[0] for f in FOLDS]

    # Per-fold metrics
    fold_data = []
    for fn in fold_names:
        fd = results.get(fn, {})
        ma = fd.get("A", {}); md = fd.get("D", {})
        a_pm = ma.get(PRIMARY_METRIC, 0)
        d_pm = md.get(PRIMARY_METRIC, 0)
        d_wins = d_pm > a_pm
        fold_data.append({
            "fold":     fn,
            "test":     fd.get("test_start","")[:7] + "〜" + fd.get("test_end","")[:7],
            "A_cagr":   ma.get("cagr",0),
            "D_cagr":   md.get("cagr",0),
            "A_sharpe": ma.get("sharpe",0),
            "D_sharpe": md.get("sharpe",0),
            "A_maxdd":  ma.get("max_dd",0),
            "D_maxdd":  md.get("max_dd",0),
            "A_calmar": ma.get("calmar",0),
            "D_calmar": md.get("calmar",0),
            "D_4th_pct":md.get("n4_days_pct",0),
            "D_blocked":md.get("blocked_4th",0),
            "D_missed": md.get("missed_signals",0),
            "delta_cagr":   round(md.get("cagr",0)   - ma.get("cagr",0),   2),
            "delta_sharpe": round(md.get("sharpe",0) - ma.get("sharpe",0), 3),
            "delta_maxdd":  round(md.get("max_dd",0)  - ma.get("max_dd",0), 2),
            "delta_calmar": round(md.get("calmar",0) - ma.get("calmar",0), 3),
            "d_wins":   d_wins,
        })

    # Win counts
    d_wins_total  = sum(1 for f in fold_data if f["d_wins"])
    a_wins_total  = len(fold_data) - d_wins_total

    # Average deltas
    avg_delta_cagr   = round(np.mean([f["delta_cagr"]   for f in fold_data]), 2)
    avg_delta_sharpe = round(np.mean([f["delta_sharpe"] for f in fold_data]), 3)
    avg_delta_maxdd  = round(np.mean([f["delta_maxdd"]  for f in fold_data]), 2)
    avg_delta_calmar = round(np.mean([f["delta_calmar"] for f in fold_data]), 3)

    # 4th slot contribution per fold
    slot4_per_fold = []
    for fn in fold_names:
        fd = results.get(fn, {})
        d_run = fd.get("D_run", {})
        all_trades = d_run.get("trades", [])
        fourth     = extract_4th_trades(all_trades)
        non_fourth = [t for t in all_trades if t["side"]=="SELL" and t.get("n_at_entry",0) < 3]
        qs4 = quality_stats(fourth)
        qs_other = quality_stats(non_fourth)
        total_pnl  = sum(t.get("pnl",0) for t in all_trades if t["side"]=="SELL")
        fourth_pnl = qs4.get("total_pnl", 0)
        slot4_per_fold.append({
            "fold":          fn,
            "n_4th":         qs4.get("n", 0),
            "4th_WR":        qs4.get("win_rate", 0),
            "4th_avg_ret":   qs4.get("avg_return", 0),
            "4th_PF":        qs4.get("pf", 0),
            "4th_total_pnl": fourth_pnl,
            "4th_share_pct": round(fourth_pnl / max(abs(total_pnl), 1) * 100, 1) if total_pnl != 0 else 0,
            "other_WR":      qs_other.get("win_rate", 0),
            "other_PF":      qs_other.get("pf", 0),
        })

    # Overall 4th slot stats across all folds
    all_fourth = []
    for fn in fold_names:
        d_run = results.get(fn, {}).get("D_run", {})
        all_fourth.extend(extract_4th_trades(d_run.get("trades", [])))
    qs4_all = quality_stats(all_fourth)

    # Verdict determination
    if d_wins_total >= 4:
        if avg_delta_calmar > 0:
            verdict = "1. Dは再現性あり ✅"
            detail  = f"D wins {d_wins_total}/5 folds (primary={PRIMARY_METRIC}), avg Δcalmar={avg_delta_calmar:+.3f}"
        else:
            verdict = "3. 判定不能 ⚠️"
            detail  = f"D wins {d_wins_total}/5 folds but avg Δcalmar={avg_delta_calmar:+.3f} ≤ 0"
    elif d_wins_total == 3:
        verdict = "3. 判定不能 ⚠️"
        detail  = f"D wins {d_wins_total}/5 folds (borderline)"
    else:
        verdict = "2. Dは過学習 ❌"
        detail  = f"D wins only {d_wins_total}/5 folds, avg Δcalmar={avg_delta_calmar:+.3f}"

    return {
        "fold_data":       fold_data,
        "d_wins_total":    d_wins_total,
        "a_wins_total":    a_wins_total,
        "avg_delta_cagr":  avg_delta_cagr,
        "avg_delta_sharpe":avg_delta_sharpe,
        "avg_delta_maxdd": avg_delta_maxdd,
        "avg_delta_calmar":avg_delta_calmar,
        "slot4_per_fold":  slot4_per_fold,
        "qs4_all":         qs4_all,
        "verdict":         verdict,
        "verdict_detail":  detail,
    }


# ─────────────────────────────────────────────────────────────────────
#  MARKDOWN REPORT
# ─────────────────────────────────────────────────────────────────────

def write_report(an: dict, output_path: Path) -> None:
    L = []; w = L.append

    w("# Variant D Walk-Forward Validation")
    w(f"\n作成日: {time.strftime('%Y-%m-%d')}  |  方式: 5-Fold Rolling WF (Train 3y / Test 1y)")
    w(f"\n**検証対象:** A (max_pos=3) vs D (max_pos=4, 4th slot RSR≥85)")
    w(f"**判定指標:** {PRIMARY_METRIC.upper()} (primary)\n")

    # ── Fold summary table ─────────────────────────────────────────────
    w("---\n## 5-Fold 結果サマリー\n")
    w("| Fold | Test期間 | A CAGR | D CAGR | ΔCAGR | A Sharpe | D Sharpe | A MaxDD | D MaxDD | A Calmar | D Calmar | ΔCalmar | D勝? |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for fd in an["fold_data"]:
        wins_mark = "**✅ D**" if fd["d_wins"] else "A"
        w(f"| {fd['fold']} | {fd['test']} "
          f"| {fd['A_cagr']:+.1f}% | {fd['D_cagr']:+.1f}% | {fd['delta_cagr']:+.1f}pp "
          f"| {fd['A_sharpe']:.3f} | {fd['D_sharpe']:.3f} "
          f"| {fd['A_maxdd']:.1f}% | {fd['D_maxdd']:.1f}% "
          f"| {fd['A_calmar']:.3f} | {fd['D_calmar']:.3f} | {fd['delta_calmar']:+.3f} "
          f"| {wins_mark} |")

    w(f"\n**勝敗: D {an['d_wins_total']}/5  A {an['a_wins_total']}/5**")

    # ── Average deltas ─────────────────────────────────────────────────
    w("\n### 平均差分 (D − A)\n")
    w("| 指標 | 平均Δ |")
    w("|---|---|")
    w(f"| CAGR | {an['avg_delta_cagr']:+.2f}pp |")
    w(f"| Sharpe | {an['avg_delta_sharpe']:+.3f} |")
    w(f"| MaxDD | {an['avg_delta_maxdd']:+.2f}pp |")
    w(f"| Calmar | {an['avg_delta_calmar']:+.3f} |")

    # ── 4th slot contribution per fold ────────────────────────────────
    w("\n---\n## 4th Slot Contribution 分析\n")
    w("| Fold | N_4th | 4th WR | 4th avg ret% | 4th PF | 4th total PnL | 4th PnL寄与% | 1-3枚 WR | 1-3枚 PF |")
    w("|---|---|---|---|---|---|---|---|---|")
    for s in an["slot4_per_fold"]:
        pnl_s = f"¥{s['4th_total_pnl']:+,.0f}" if s["n_4th"] > 0 else "—"
        w(f"| {s['fold']} | {s['n_4th']} "
          f"| {s['4th_WR']:.1f}% | {s['4th_avg_ret']:+.2f}% | {s['4th_PF']:.3f} "
          f"| {pnl_s} | {s['4th_share_pct']:.1f}% "
          f"| {s['other_WR']:.1f}% | {s['other_PF']:.3f} |")

    qs4 = an["qs4_all"]
    w(f"\n**全フォールド 4枚目 集計**: N={qs4.get('n',0)}  WR={qs4.get('win_rate',0):.1f}%  "
      f"avg ret={qs4.get('avg_return',0):+.2f}%  PF={qs4.get('pf',0):.3f}  "
      f"Total PnL=¥{qs4.get('total_pnl',0):+,.0f}")

    # ── 4th slot utilization detail ────────────────────────────────────
    w("\n### Variant D スロット稼働詳細\n")
    w("| Fold | 4th利用日% | blocked件数 | missed件数 |")
    w("|---|---|---|---|")
    for fd in an["fold_data"]:
        w(f"| {fd['fold']} | {fd['D_4th_pct']:.1f}% | {fd['D_blocked']} | {fd['D_missed']} |")

    # ── Verdict ────────────────────────────────────────────────────────
    w("\n---\n## 最終結論\n")
    w(f"### 判定: {an['verdict']}\n")
    w(f"根拠: {an['verdict_detail']}\n")

    # Interpretation
    d_wins = an["d_wins_total"]
    avg_dc = an["avg_delta_calmar"]
    avg_dd = an["avg_delta_maxdd"]
    avg_cagr = an["avg_delta_cagr"]
    qs4_pf = an["qs4_all"].get("pf", 0)
    qs4_wr = an["qs4_all"].get("win_rate", 0)

    w("### 詳細解釈\n")
    w(f"**WF結果 ({d_wins}/5 D勝)**")
    if d_wins >= 4:
        w("- D は5フォールド中4回以上でAを上回る → IS限定の過学習ではない")
    elif d_wins == 3:
        w("- D は5フォールド中3回でAを上回る → 優位性は限定的・相場環境依存の可能性")
    else:
        w("- D はAに対して明確な優位性を示せず → IS改善は偶発的だった可能性が高い")

    w(f"\n**4枚目スロットの実力** (全WF期間: WR={qs4_wr:.1f}%, PF={qs4_pf:.3f})")
    if qs4_pf > 1.3 and qs4_wr > 45:
        w("- 4枚目ポジション単体でも正のアルファを示す → スロット追加は品質担保できている")
    elif qs4_pf > 1.0:
        w("- 4枚目ポジションはわずかに正のPF → アルファは薄いが有害ではない")
    else:
        w("- 4枚目ポジション単体でPF<1 → スロット追加が損失源になっている")

    w(f"\n**最大DD影響** (平均ΔMaxDD={avg_dd:+.2f}pp)")
    if abs(avg_dd) <= 2.0:
        w(f"- MaxDD影響は軽微({avg_dd:+.2f}pp) → リスク増加はほぼない")
    elif avg_dd < -2.0:
        w(f"- MaxDD{avg_dd:+.2f}pp悪化 → Dは許容できるリスク増を伴う")
    else:
        w(f"- MaxDD{avg_dd:+.2f}pp悪化 → リスク増加が大きく実運用に注意")

    w("\n### 採用判断\n")
    w("| 条件 | 充足 |")
    w("|---|---|")
    w(f"| WF D勝率 ≥ 4/5 | {'✅' if d_wins >= 4 else '❌'} ({d_wins}/5) |")
    w(f"| 平均ΔCalmar ≥ 0 | {'✅' if avg_dc >= 0 else '❌'} ({avg_dc:+.3f}) |")
    w(f"| 平均ΔCalmar ≥ -0.05 | {'✅' if avg_dc >= -0.05 else '❌'} ({avg_dc:+.3f}) |")
    w(f"| 4th PF ≥ 1.0 | {'✅' if qs4_pf >= 1.0 else '❌'} ({qs4_pf:.3f}) |")
    w(f"| 平均ΔMaxDD ≥ -3pp | {'✅' if avg_dd >= -3.0 else '❌'} ({avg_dd:+.2f}pp) |")

    adopt_count = sum([d_wins >= 4, avg_dc >= 0, qs4_pf >= 1.0, avg_dd >= -3.0])
    if adopt_count >= 4:
        adopt = "**ADOPT (条件付き PARAMS_LOCKED変更申請可)** — max_positions=3→4 + 4th RSR≥85"
    elif adopt_count >= 3:
        adopt = "**RESEARCH (更なる検証後に判断)** — OOS蓄積後に再評価"
    else:
        adopt = "**REJECT** — IS改善は過学習の可能性が高い"
    w(f"\n**最終判断: {adopt}**")
    w(f"\n> PARAMS_LOCKED変更には ASK_FIRST_ON_CHANGE=true 適用 — 変更前にユーザー確認必須")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(L), encoding="utf-8")
    print(f"  レポート保存: {output_path}")


# ─────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────

def main() -> int:
    cfg = load_strategy_config()

    print("=" * 68)
    print("  Variant D Walk-Forward Validation — 5-Fold (Train3y/Test1y)")
    print("  A: max_pos=3  vs  D: max_pos=4 + 4th RSR≥85")
    print("=" * 68)

    print("\n[1/3] データロード中...")
    (universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
     rsr_syms, topix_close, cfg) = load_data(cfg)

    print("\n[2/3] 5-Fold WF実行中...")
    wf_results = run_wf(universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
                        topix_close, rsr_syms, cfg)

    print("\n[3/3] 分析 + レポート生成...")
    an = analyze(wf_results)

    out_path = REPORTS_DIR / "variant_d_walkforward.md"
    write_report(an, out_path)

    # ── Console summary ────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("  ★ WF結果サマリー")
    print("=" * 68)
    print(f"\n  {'Fold':<8} {'Test':<18} {'A CAGR':>8} {'D CAGR':>8} {'ΔCAGR':>7} "
          f"{'A Calmar':>9} {'D Calmar':>9} {'ΔCalmar':>8} {'Winner':>7}")
    print("  " + "-" * 82)
    for fd in an["fold_data"]:
        winner = "D ✅" if fd["d_wins"] else "A"
        print(f"  {fd['fold']:<8} {fd['test']:<18} "
              f"{fd['A_cagr']:>+7.1f}% {fd['D_cagr']:>+7.1f}% {fd['delta_cagr']:>+6.1f}pp "
              f"{fd['A_calmar']:>9.3f} {fd['D_calmar']:>9.3f} {fd['delta_calmar']:>+7.3f} "
              f"{winner:>7}")

    print(f"\n  D勝: {an['d_wins_total']}/5  A勝: {an['a_wins_total']}/5")
    print(f"  平均Δ: CAGR={an['avg_delta_cagr']:+.2f}pp  "
          f"Sharpe={an['avg_delta_sharpe']:+.3f}  "
          f"MaxDD={an['avg_delta_maxdd']:+.2f}pp  "
          f"Calmar={an['avg_delta_calmar']:+.3f}")

    qs4 = an["qs4_all"]
    print(f"\n  4枚目スロット WF全体: N={qs4.get('n',0)}  WR={qs4.get('win_rate',0):.1f}%  "
          f"avg={qs4.get('avg_return',0):+.2f}%  PF={qs4.get('pf',0):.3f}  "
          f"Total=¥{qs4.get('total_pnl',0):+,.0f}")

    print(f"\n  ★ 判定: {an['verdict']}")
    print(f"  根拠 : {an['verdict_detail']}")
    print(f"\n  レポート → {out_path}")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
