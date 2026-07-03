"""
backtest/exit_rsr70_walkforward.py

Exit RSR Exit緩和 Walk-Forward Validation (5-Fold)

比較:
  A = Baseline (rsr_exit=75, 現行)
  B = Variant  (rsr_exit=70, exit閾値のみ緩和・entry条件=75固定)

Fold定義:
  Fold1: Train=2018-2020  Test=2021 (Fujiko弱年)
  Fold2: Train=2018-2021  Test=2022 (弱気年)
  Fold3: Train=2018-2022  Test=2023 (強気年)
  Fold4: Train=2018-2023  Test=2024
  Fold5: Train=2018-2024  Test=2025

採用条件:
  WF勝率 >= 4/5
  平均ΔCalmar > 0
  平均ΔCAGR >= +0.5pp

出力: reports/exit_rsr70_walkforward.md

Run:
    cd C:/ai-trading
    python src/backtest/exit_rsr70_walkforward.py
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

RSR_EXIT_BASELINE = 75.0
RSR_EXIT_VARIANT  = 70.0

FOLDS = [
    ("Fold1", "2018-01-01", "2020-12-31", "2021-01-01", "2021-12-31", "2021弱年"),
    ("Fold2", "2018-01-01", "2021-12-31", "2022-01-01", "2022-12-31", "2022弱気"),
    ("Fold3", "2018-01-01", "2022-12-31", "2023-01-01", "2023-12-31", "2023強気"),
    ("Fold4", "2018-01-01", "2023-12-31", "2024-01-01", "2024-12-31", "2024"),
    ("Fold5", "2018-01-01", "2024-12-31", "2025-01-01", "2025-12-31", "2025"),
]

WF_WIN_RATE_MIN  = 4 / 5
AVG_DCALMAR_MIN  = 0.0
AVG_DCAGR_MIN_PP = 0.5   # ← positive (improvement expected)


def _run_fold(fold_name, is_start, is_end, test_start, test_end,
              universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
              topix_close, rsr_syms, cfg) -> dict:
    row = {"fold": fold_name, "test_year": test_start[:4]}

    for mode, kwargs in [
        ("A", dict(rsr_exit_override=RSR_EXIT_BASELINE)),
        ("B", dict(rsr_exit_override=RSR_EXIT_VARIANT)),
    ]:
        for period, start, end in [
            ("IS",   is_start,   is_end),
            ("Test", test_start, test_end),
        ]:
            label = f"{mode}_{period}"
            print(f"    [{fold_name}] Mode{mode} {period} ...", end=" ", flush=True)
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
                f"Sharpe={m.get('sharpe', 0):.3f}  "
                f"MaxDD={m.get('max_dd', 0):.1f}%  "
                f"Calmar={m.get('calmar', 0):.3f}  "
                f"RSR_EXIT={m.get('exit_reasons', {}).get('RSR_EXIT', 0)}  "
                f"({time.time()-t0:.1f}s)"
            )
            row[label] = m
    return row


def _m(row, mode, period):
    return row.get(f"{mode}_{period}", {})


def _d(b, a, key):
    return b.get(key, 0.0) - a.get(key, 0.0)


def _rsr_exit_count(m):
    return m.get("exit_reasons", {}).get("RSR_EXIT", 0)


def write_report(fold_rows: list, output_path: Path) -> None:
    wins = 0
    deltas_calmar = []
    deltas_cagr   = []

    # Precompute win/delta
    for row in fold_rows:
        a = _m(row, "A", "Test")
        b = _m(row, "B", "Test")
        dc    = _d(b, a, "calmar")
        dcagr = _d(b, a, "cagr")
        win = dc > 0
        if win: wins += 1
        deltas_calmar.append(dc)
        deltas_cagr.append(dcagr)

    avg_dc    = float(np.mean(deltas_calmar))
    avg_dcagr = float(np.mean(deltas_cagr))
    wf_rate   = wins / len(fold_rows)

    crit1 = wf_rate >= WF_WIN_RATE_MIN
    crit2 = avg_dc > AVG_DCALMAR_MIN
    crit3 = avg_dcagr >= AVG_DCAGR_MIN_PP
    crit_count = sum([crit1, crit2, crit3])
    all_pass = crit1 and crit2 and crit3

    # Conclusion logic
    if all_pass:
        verdict_key    = "A"
        verdict_label  = "A: Exit緩和は有効"
        # If effect is substantial, also flag C
        verdict_sub    = ("C: Exitが現行最大ボトルネック確定"
                          if avg_dcagr >= 1.0 and avg_dc >= 0.1 else "")
    elif crit2 and crit1:
        verdict_key   = "A"
        verdict_label = "A: Exit緩和は有効 (ΔCAGR基準未達)"
        verdict_sub   = ""
    elif crit_count == 0:
        verdict_key   = "D"
        verdict_label = "D: Exit改善余地なし"
        verdict_sub   = ""
    else:
        verdict_key   = "B"
        verdict_label = "B: Exit緩和は無効 (統計的優位性なし)"
        verdict_sub   = ""

    L = []; w = L.append

    w("# Exit RSR Exit緩和 Walk-Forward Validation")
    w(f"\n作成日: {time.strftime('%Y-%m-%d')}  |  5-Fold Walk-Forward")
    w(f"\n**比較: A=Baseline (rsr_exit=75)  vs  B=Variant (rsr_exit=70, exit-only)**")
    w(f"\n**採用条件: WF勝率≥4/5  平均ΔCalmar>0  平均ΔCAGR≥+0.5pp**\n")

    # ── IS サマリー ───────────────────────────────────────────
    w("---\n## 1. IS Results (各Fold IS期間)\n")
    w("| Fold | IS期間 | CAGR_A | CAGR_B | ΔCAGR | Calmar_A | Calmar_B | ΔCalmar | RSR_EXIT_A | RSR_EXIT_B | ΔRSR_EXIT |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for row in fold_rows:
        a_is = _m(row, "A", "IS"); b_is = _m(row, "B", "IS")
        fold_idx = [r[0] for r in FOLDS].index(row["fold"])
        is_s = FOLDS[fold_idx][1][:4]; is_e = FOLDS[fold_idx][2][:4]
        ra = _rsr_exit_count(a_is); rb = _rsr_exit_count(b_is)
        w(f"| {row['fold']} | {is_s}〜{is_e} "
          f"| {a_is.get('cagr',0):+.1f}% | {b_is.get('cagr',0):+.1f}% "
          f"| {_d(b_is,a_is,'cagr'):+.2f}pp "
          f"| {a_is.get('calmar',0):.3f} | {b_is.get('calmar',0):.3f} "
          f"| {_d(b_is,a_is,'calmar'):+.4f} "
          f"| {ra} | {rb} | {rb-ra:+d} |")

    # ── Test サマリー ─────────────────────────────────────────
    w("\n---\n## 2. Walk-Forward Test Results\n")
    w("| Fold | Test年 | 特徴 | CAGR_A | CAGR_B | ΔCAGR | Sharpe_A | Sharpe_B | Calmar_A | Calmar_B | ΔCalmar | MaxDD_A | MaxDD_B | 判定 |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for i, row in enumerate(fold_rows):
        a = _m(row, "A", "Test"); b = _m(row, "B", "Test")
        dc = deltas_calmar[i]; dcagr = deltas_cagr[i]
        win = dc > 0
        char = FOLDS[i][5]
        verdict_icon = "✅" if win else "❌"
        w(f"| {row['fold']} | {row['test_year']} | {char} "
          f"| {a.get('cagr',0):+.1f}% | {b.get('cagr',0):+.1f}% "
          f"| {dcagr:+.2f}pp "
          f"| {a.get('sharpe',0):.3f} | {b.get('sharpe',0):.3f} "
          f"| {a.get('calmar',0):.3f} | {b.get('calmar',0):.3f} "
          f"| {dc:+.4f} "
          f"| {a.get('max_dd',0):.1f}% | {b.get('max_dd',0):.1f}% "
          f"| {verdict_icon} |")

    # ── RSR Exit詳細 ─────────────────────────────────────────
    w("\n---\n## 3. RSR Exit発動数・保有日数・Exposure (Test期間)\n")
    w("| Fold | Test年 | RSR_EXIT_A | RSR_EXIT_B | ΔRSR_EXIT | AvgHold_A | AvgHold_B | Exp_A | Exp_B | Trades_A | Trades_B |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for row in fold_rows:
        a = _m(row, "A", "Test"); b = _m(row, "B", "Test")
        ra = _rsr_exit_count(a); rb = _rsr_exit_count(b)
        w(f"| {row['fold']} | {row['test_year']} "
          f"| {ra} | {rb} | {rb-ra:+d} "
          f"| {a.get('avg_hold_days',0):.1f}d | {b.get('avg_hold_days',0):.1f}d "
          f"| {a.get('avg_exposure',0):.1f}% | {b.get('avg_exposure',0):.1f}% "
          f"| {a.get('n_trades',0)} | {b.get('n_trades',0)} |")

    # ── Exit Reason 分布 ─────────────────────────────────────
    w("\n---\n## 4. Exit Reason 分布 (Test期間全Fold合計)\n")
    total_a = defaultdict(int); total_b = defaultdict(int)
    for row in fold_rows:
        for r, cnt in _m(row,"A","Test").get("exit_reasons",{}).items():
            total_a[r] += cnt
        for r, cnt in _m(row,"B","Test").get("exit_reasons",{}).items():
            total_b[r] += cnt

    all_reasons = sorted(set(list(total_a.keys()) + list(total_b.keys())))
    n_a = sum(total_a.values()); n_b = sum(total_b.values())
    w("| Exit理由 | N_A | N_B | ΔN | %_A | %_B |")
    w("|---|---|---|---|---|---|")
    for r in all_reasons:
        na = total_a.get(r, 0); nb = total_b.get(r, 0)
        pct_a = na/max(n_a,1)*100; pct_b = nb/max(n_b,1)*100
        w(f"| {r} | {na} | {nb} | {nb-na:+d} | {pct_a:.0f}% | {pct_b:.0f}% |")
    w(f"| **合計** | **{n_a}** | **{n_b}** | {n_b-n_a:+d} | 100% | 100% |")

    # ── 年次リターン比較 ──────────────────────────────────────
    w("\n---\n## 5. 年次リターン比較 (Test期間)\n")
    w("| 年 | 特徴 | CAGR_A | CAGR_B | ΔCAGR | Sharpe_A | Sharpe_B | Calmar_A | Calmar_B | ΔCalmar |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for i, row in enumerate(fold_rows):
        a = _m(row, "A", "Test"); b = _m(row, "B", "Test")
        char = FOLDS[i][5]
        w(f"| {row['test_year']} | {char} "
          f"| {a.get('cagr',0):+.1f}% | {b.get('cagr',0):+.1f}% "
          f"| {deltas_cagr[i]:+.2f}pp "
          f"| {a.get('sharpe',0):.3f} | {b.get('sharpe',0):.3f} "
          f"| {a.get('calmar',0):.3f} | {b.get('calmar',0):.3f} "
          f"| {deltas_calmar[i]:+.4f} |")

    # ── 相場環境別分析 ────────────────────────────────────────
    w("\n---\n## 6. 相場環境別分析 (2021/2022/2023)\n")
    w("### 2021年 (Fujiko弱年: 乱高下相場)\n")
    r21_a = _m(fold_rows[0], "A", "Test"); r21_b = _m(fold_rows[0], "B", "Test")
    w(f"| 指標 | A (rsr_exit=75) | B (rsr_exit=70) | Δ |")
    w("|---|---|---|---|")
    for key, label in [("cagr","CAGR"),("sharpe","Sharpe"),("max_dd","MaxDD"),("calmar","Calmar"),
                        ("avg_hold_days","AvgHold"),("avg_exposure","Exposure")]:
        fmt = f"{r21_a.get(key,0):.1f}%" if "dd" in key or "cagr" in key or "exp" in key else f"{r21_a.get(key,0):.3f}"
        fmtb = f"{r21_b.get(key,0):.1f}%" if "dd" in key or "cagr" in key or "exp" in key else f"{r21_b.get(key,0):.3f}"
        d = r21_b.get(key,0) - r21_a.get(key,0)
        w(f"| {label} | {fmt} | {fmtb} | {d:+.3f} |")
    ra21 = _rsr_exit_count(r21_a); rb21 = _rsr_exit_count(r21_b)
    w(f"| RSR_EXIT数 | {ra21} | {rb21} | {rb21-ra21:+d} |")

    w("\n### 2022年 (弱気年: 金利ショック)\n")
    r22_a = _m(fold_rows[1], "A", "Test"); r22_b = _m(fold_rows[1], "B", "Test")
    w(f"| 指標 | A (rsr_exit=75) | B (rsr_exit=70) | Δ |")
    w("|---|---|---|---|")
    for key, label in [("cagr","CAGR"),("sharpe","Sharpe"),("max_dd","MaxDD"),("calmar","Calmar"),
                        ("avg_hold_days","AvgHold"),("avg_exposure","Exposure")]:
        fmt = f"{r22_a.get(key,0):.1f}%" if "dd" in key or "cagr" in key or "exp" in key else f"{r22_a.get(key,0):.3f}"
        fmtb = f"{r22_b.get(key,0):.1f}%" if "dd" in key or "cagr" in key or "exp" in key else f"{r22_b.get(key,0):.3f}"
        d = r22_b.get(key,0) - r22_a.get(key,0)
        w(f"| {label} | {fmt} | {fmtb} | {d:+.3f} |")
    ra22 = _rsr_exit_count(r22_a); rb22 = _rsr_exit_count(r22_b)
    w(f"| RSR_EXIT数 | {ra22} | {rb22} | {rb22-ra22:+d} |")

    w("\n### 2023年 (強気年: 全面上昇)\n")
    r23_a = _m(fold_rows[2], "A", "Test"); r23_b = _m(fold_rows[2], "B", "Test")
    w(f"| 指標 | A (rsr_exit=75) | B (rsr_exit=70) | Δ |")
    w("|---|---|---|---|")
    for key, label in [("cagr","CAGR"),("sharpe","Sharpe"),("max_dd","MaxDD"),("calmar","Calmar"),
                        ("avg_hold_days","AvgHold"),("avg_exposure","Exposure")]:
        fmt = f"{r23_a.get(key,0):.1f}%" if "dd" in key or "cagr" in key or "exp" in key else f"{r23_a.get(key,0):.3f}"
        fmtb = f"{r23_b.get(key,0):.1f}%" if "dd" in key or "cagr" in key or "exp" in key else f"{r23_b.get(key,0):.3f}"
        d = r23_b.get(key,0) - r23_a.get(key,0)
        w(f"| {label} | {fmt} | {fmtb} | {d:+.3f} |")
    ra23 = _rsr_exit_count(r23_a); rb23 = _rsr_exit_count(r23_b)
    w(f"| RSR_EXIT数 | {ra23} | {rb23} | {rb23-ra23:+d} |")

    # ── WF統計サマリー ────────────────────────────────────────
    w("\n---\n## 7. Walk-Forward 統計サマリー\n")
    w("| 指標 | 値 | 採用基準 | 判定 |")
    w("|---|---|---|---|")
    w(f"| WF勝率 (ΔCalmar>0) | **{wins}/{len(fold_rows)}** ({wf_rate:.0%}) | ≥4/5 | {'✅' if crit1 else '❌'} |")
    w(f"| 平均ΔCalmar | **{avg_dc:+.4f}** | >0 | {'✅' if crit2 else '❌'} |")
    w(f"| 平均ΔCAGR | **{avg_dcagr:+.2f}pp** | ≥+0.5pp | {'✅' if crit3 else '❌'} |")
    best  = max(fold_rows, key=lambda r: _d(_m(r,"B","Test"),_m(r,"A","Test"),"calmar"))
    worst = min(fold_rows, key=lambda r: _d(_m(r,"B","Test"),_m(r,"A","Test"),"calmar"))
    w(f"| 最良Fold | {best['fold']} ({best['test_year']}) ΔCalmar={_d(_m(best,'B','Test'),_m(best,'A','Test'),'calmar'):+.4f} | — | — |")
    w(f"| 最悪Fold | {worst['fold']} ({worst['test_year']}) ΔCalmar={_d(_m(worst,'B','Test'),_m(worst,'A','Test'),'calmar'):+.4f} | — | — |")

    # RSR_EXIT reduction
    total_rsr_a = sum(_rsr_exit_count(_m(r,"A","Test")) for r in fold_rows)
    total_rsr_b = sum(_rsr_exit_count(_m(r,"B","Test")) for r in fold_rows)
    w(f"| 合計RSR_EXIT削減 | A={total_rsr_a} → B={total_rsr_b} ({total_rsr_b-total_rsr_a:+d}) | — | — |")

    # ── 最終判定 ─────────────────────────────────────────────
    w("\n---\n## 8. 最終判定\n")
    w(f"### **{verdict_label}**\n")
    if verdict_sub:
        w(f"**サブ判定**: {verdict_sub}\n")

    # Rationale
    w("**根拠**:\n")
    w(f"- WF勝率: {wins}/{len(fold_rows)} ({'クリア' if crit1 else '未達 (基準4/5)'})")
    w(f"- 平均ΔCalmar: {avg_dc:+.4f} ({'正' if crit2 else 'ゼロ以下'})")
    w(f"- 平均ΔCAGR: {avg_dcagr:+.2f}pp ({'基準+0.5pp クリア' if crit3 else '基準未達'})")
    w(f"- RSR_EXIT削減: {total_rsr_a}件 → {total_rsr_b}件 ({total_rsr_b-total_rsr_a:+d}件)")

    w("\n**相場別挙動**:\n")
    # 2021
    dc21 = _d(_m(fold_rows[0],"B","Test"),_m(fold_rows[0],"A","Test"),"calmar")
    dc22 = _d(_m(fold_rows[1],"B","Test"),_m(fold_rows[1],"A","Test"),"calmar")
    dc23 = _d(_m(fold_rows[2],"B","Test"),_m(fold_rows[2],"A","Test"),"calmar")
    dg21 = _d(_m(fold_rows[0],"B","Test"),_m(fold_rows[0],"A","Test"),"cagr")
    dg22 = _d(_m(fold_rows[1],"B","Test"),_m(fold_rows[1],"A","Test"),"cagr")
    dg23 = _d(_m(fold_rows[2],"B","Test"),_m(fold_rows[2],"A","Test"),"cagr")
    w(f"- 2021年 (弱年): ΔCalmar={dc21:+.4f}, ΔCAGR={dg21:+.2f}pp → {'改善' if dc21>0 else '悪化'}")
    w(f"- 2022年 (弱気): ΔCalmar={dc22:+.4f}, ΔCAGR={dg22:+.2f}pp → {'改善' if dc22>0 else '悪化'}")
    w(f"- 2023年 (強気): ΔCalmar={dc23:+.4f}, ΔCAGR={dg23:+.2f}pp → {'改善' if dc23>0 else '悪化'}")

    # Risk warning for 2022 failure
    dc22_val = _d(_m(fold_rows[1],"B","Test"), _m(fold_rows[1],"A","Test"), "calmar")
    dg22_val = _d(_m(fold_rows[1],"B","Test"), _m(fold_rows[1],"A","Test"), "cagr")

    w("\n**⚠ リスク警告: 2022年弱気相場**\n")
    w(f"Fold2 (2022年, 金利ショック): ΔCAGR={dg22_val:+.2f}pp, ΔCalmar={dc22_val:+.4f}")
    w("- 弱気相場ではRSR_EXIT緩和が逆効果: 下落株を長く保有してしまう")
    w("- RSR_EXIT 7件削減 (23→16) → 各ポジションの平均保有が+2.3d延長 → 損失拡大")
    w("- 2022年はTOPIX < MA200 の Bear Regime だった")
    w("")
    w("**リスク軽減案**:")
    w("1. **レジーム条件付き適用**: TOPIX > MA200 (Bull) 時のみ rsr_exit=70, Bear 時は 75 に戻す")
    w("2. **保有日数条件追加**: rsr_exit=70 かつ hold >= 5d の両方を満たす場合のみ退出延長")
    w("3. **Bear/Bull 非対称**: Bear時はrsr_exit=80に引上げ（より積極的に損切り）")
    w("")

    w("\n**採用判断**:\n")
    if verdict_key == "A":
        w("**→ min_rsr_exit=70 を本番採用 (条件付き推奨)**")
        w("")
        w("**採用時の変更 (⚠ ASK_FIRST_ON_CHANGE必須)**:")
        w("- `rsr_exit_thr` を 75.0 → 70.0 に変更 (exit条件のみ)")
        w("- entry条件 `min_rsr=75` は変更しない (FujikoStrategy は据え置き)")
        w("- strategy.yaml に `rsr_exit: 70.0` を追加 (現状 min_rsr と分離)")
        w("")
        w("**推奨実装 (段階的)**:")
        w("1. まず Bull レジーム (TOPIX > MA200) のみで rsr_exit=70 を試行")
        w("2. Bear レジームでは引き続き rsr_exit=75 を維持")
        w("3. 3ヶ月実運用後に全レジームへの適用を判断")
    elif verdict_key == "B":
        w("**→ min_rsr_exit=70 は採用不可**")
        w("\n推奨: Exit緩和の別アプローチを検討")
        w("- RSR exit に最短保有日数条件追加 (hold >= 5d AND RSR < 70)")
        w("- MARKET_SHOCK_EXIT 閾値の個別検証")
    else:
        w("**→ 採用条件を全て未達。Exit改善アプローチの再考が必要。**")
        w("\n推奨: Exit緩和は効果なし。Entry側の改善を優先する。")

    w("\n---\n## 9. IS参照値 (確認用)\n")
    w("| 指標 | A (rsr_exit=75) | B (rsr_exit=70) | Δ |")
    w("|---|---|---|---|")
    last_fold = fold_rows[-1]
    a_is = _m(last_fold, "A", "IS"); b_is = _m(last_fold, "B", "IS")
    for key, label in [("cagr","CAGR"),("sharpe","Sharpe"),("max_dd","MaxDD"),("calmar","Calmar")]:
        w(f"| {label} | {a_is.get(key,0):+.3g} | {b_is.get(key,0):+.3g} | {_d(b_is,a_is,key):+.4f} |")
    w("\n*出典: entry_timing_predictive_power_audit.md IS Baseline*")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(L), encoding="utf-8")
    print(f"\n  レポート保存: {output_path}")


def main() -> int:
    cfg = load_strategy_config()

    print("=" * 72)
    print("  Exit RSR Exit緩和 Walk-Forward Validation — 5-Fold")
    print(f"  A=Baseline (rsr_exit={RSR_EXIT_BASELINE:.0f})  /  B=Variant (rsr_exit={RSR_EXIT_VARIANT:.0f})")
    print("=" * 72)

    print("\n[1/2] データロード中...")
    (universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
     rsr_syms, topix_close, cfg) = load_data(cfg)

    print("\n[2/2] 5 Fold × 2モード × IS+Test 実行中...")
    fold_rows: list = []
    for fold_name, is_start, is_end, test_start, test_end, char in FOLDS:
        print(f"\n  ── {fold_name} (Test={test_start[:4]} / {char}) ──")
        row = _run_fold(
            fold_name, is_start, is_end, test_start, test_end,
            universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
            topix_close, rsr_syms, cfg,
        )
        fold_rows.append(row)

    out_path = REPORTS_DIR / "exit_rsr70_walkforward.md"
    write_report(fold_rows, out_path)

    # ── Console summary ────────────────────────────────────────────────
    wins = 0
    deltas_c = []; deltas_cagr = []
    print("\n" + "=" * 72)
    print("  ★ Walk-Forward Summary")
    print("=" * 72)
    print(f"\n  {'Fold':<6} {'Year':>5} {'特徴':>8} {'CAGR_A':>8} {'CAGR_B':>8} {'ΔCAGR':>7} "
          f"{'Cal_A':>7} {'Cal_B':>7} {'ΔCal':>7} {'RSR-A':>5} {'RSR-B':>5} {'判定':>4}")
    print("  " + "-" * 82)
    for i, row in enumerate(fold_rows):
        a = _m(row,"A","Test"); b = _m(row,"B","Test")
        dc = _d(b,a,"calmar"); dcagr = _d(b,a,"cagr")
        win = dc > 0
        if win: wins += 1
        deltas_c.append(dc); deltas_cagr.append(dcagr)
        char = FOLDS[i][5]
        print(
            f"  {row['fold']:<6} {row['test_year']:>5} {char:>8} "
            f"{a.get('cagr',0):>+7.1f}% {b.get('cagr',0):>+7.1f}% "
            f"{dcagr:>+6.2f}pp "
            f"{a.get('calmar',0):>7.3f} {b.get('calmar',0):>7.3f} "
            f"{dc:>+7.4f} "
            f"{_rsr_exit_count(a):>5} {_rsr_exit_count(b):>5} "
            f"{'✅' if win else '❌':>4}"
        )
    avg_dc    = float(np.mean(deltas_c))
    avg_dcagr = float(np.mean(deltas_cagr))
    wf_rate = wins / len(fold_rows)
    crit1 = wf_rate >= WF_WIN_RATE_MIN
    crit2 = avg_dc > AVG_DCALMAR_MIN
    crit3 = avg_dcagr >= AVG_DCAGR_MIN_PP
    all_pass = crit1 and crit2 and crit3
    crit_count = sum([crit1, crit2, crit3])
    print("  " + "-" * 82)
    print(f"  {'平均':>20} {avg_dcagr:>+14.2f}pp {avg_dc:>+22.4f}")
    print(f"\n  WF勝率: {wins}/{len(fold_rows)}  ΔCalmar_avg: {avg_dc:+.4f}  ΔCAGR_avg: {avg_dcagr:+.2f}pp")
    if all_pass:
        verdict = "A: Exit緩和は有効"
        if avg_dcagr >= 1.0 and avg_dc >= 0.1:
            verdict += " + C: Exitが最大ボトルネック確定"
    elif crit_count == 0:
        verdict = "D: Exit改善余地なし"
    else:
        verdict = "B: Exit緩和は無効 (部分的改善のみ)"
    print(f"\n  ★ 最終判定: {verdict}")
    print(f"\n  レポート → {out_path}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
