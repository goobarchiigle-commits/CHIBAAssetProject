"""
backtest/rsr_context_sweep.py

RSR コンテキスト拡大 × min_rsr 感度分析（2×3 グリッド）

実験設計:
  RSR context:
    narrow : 各年の選定銘柄のみ（9〜30銘柄 / 現行動作）
    broad  : TOPIX100全件 (~76銘柄 / ranking/trading分離）

  min_rsr: 70 / 72 / 75

目標:
  avg_exposure 25〜30%  +  Sharpe > 0.8

背景（診断から）:
  RSR context = 42 → top 25% = 10銘柄 → RSR≥75 = 5〜6銘柄 → after breakout = 0.18/day
  → RSR分布の上位尾部が薄い（母集団が小さすぎる）
  → broad context で cross-sectional alpha を強化すると候補数が安定する

実行:
  python backtest/rsr_context_sweep.py
"""

from __future__ import annotations

import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

from backtest.live_equivalent import run_backtest, START, END, CAPITAL

# ── 実験グリッド ──────────────────────────────────────────────────────── #
RSR_CONTEXTS = [
    {"use_broad_rsr": False, "label": "narrow（年別選定）"},
    {"use_broad_rsr": True,  "label": "broad（TOPIX100 ~76）"},
]
MIN_RSR_VALUES = [75, 72, 70]

# ── 合格基準 ──────────────────────────────────────────────────────────── #
TARGET_SHARPE   = 0.80
TARGET_EXPOSURE = (0.25, 0.30)


def main() -> int:
    print("=" * 76)
    print("  RSR コンテキスト × min_rsr 感度分析（2×3 グリッド）")
    print(f"  期間: {START} 〜 {END}  /  初期資本: ¥{CAPITAL:,}")
    print("=" * 76)

    rows = []
    for ctx in RSR_CONTEXTS:
        for mrsr in MIN_RSR_VALUES:
            tag = f"{ctx['label']} / min_rsr={mrsr}"
            print(f"\n  ── {tag} 実行中... ──")
            r = run_backtest(
                start=START, end=END, capital=CAPITAL,
                min_rsr=mrsr,
                use_rolling=True,
                use_broad_rsr=ctx["use_broad_rsr"],
                verbose=False,
            )
            ok_sharpe = r["sharpe"] >= TARGET_SHARPE
            ok_exp    = TARGET_EXPOSURE[0] <= r["avg_exposure"] <= TARGET_EXPOSURE[1]
            verdict   = "✅ 両方" if ok_sharpe and ok_exp else (
                        "⚠ Sharpe" if ok_sharpe else
                        "⚠ Exp" if ok_exp else "❌")
            print(
                f"  Sharpe={r['sharpe']:.3f}  CAGR={r['cagr']:+.2%}"
                f"  MaxDD={r['max_dd']:.2%}  exp={r['avg_exposure']:.1%}"
                f"  cands={r['avg_candidates']:.2f}/日  → {verdict}"
            )
            rows.append({
                "RSR_context":  ctx["label"],
                "min_rsr":      mrsr,
                "Sharpe":       f"{r['sharpe']:.3f}",
                "CAGR%":        f"{r['cagr']*100:+.2f}",
                "MaxDD%":       f"{r['max_dd']*100:.2f}",
                "Calmar":       f"{r['calmar']:.3f}",
                "exp%":         f"{r['avg_exposure']*100:.1f}",
                "cands/day":    f"{r['avg_candidates']:.2f}",
                "zero_exp%":    f"{r['zero_exp_rate']:.1f}",
                "trades":       r["trade_count"],
                "win%":         f"{r['win_rate']*100:.1f}",
                "verdict":      verdict,
            })

    # ---- 結果テーブル ----
    df = pd.DataFrame(rows)
    print("\n" + "=" * 76)
    print("  結果サマリー")
    print("=" * 76)
    print(df.to_string(index=False))

    # ---- 感度分析 ----
    print("\n  ── 感度分析 ──")
    for ctx in RSR_CONTEXTS:
        sub = [r for r in rows if r["RSR_context"] == ctx["label"]]
        sharpes  = [float(r["Sharpe"])     for r in sub]
        exps     = [float(r["exp%"])       for r in sub]
        print(f"\n  [{ctx['label']}]")
        print(f"    Sharpe 範囲: {min(sharpes):.3f} 〜 {max(sharpes):.3f}"
              f"  (差: {max(sharpes)-min(sharpes):.3f})")
        print(f"    exposure 範囲: {min(exps):.1f}% 〜 {max(exps):.1f}%"
              f"  (差: {max(exps)-min(exps):.1f}pp)")

    # ---- narrow vs broad（min_rsr=75 で比較）----
    n75 = next(r for r in rows if not r["RSR_context"].startswith("broad") and r["min_rsr"] == 75)
    b75 = next(r for r in rows if r["RSR_context"].startswith("broad") and r["min_rsr"] == 75)
    print("\n  ── narrow vs broad（min_rsr=75 固定）──")
    print(f"    Sharpe:   narrow={n75['Sharpe']}  broad={b75['Sharpe']}"
          f"  (Δ={float(b75['Sharpe'])-float(n75['Sharpe']):+.3f})")
    print(f"    exposure: narrow={n75['exp%']}%  broad={b75['exp%']}%"
          f"  (Δ={float(b75['exp%'])-float(n75['exp%']):+.1f}pp)")
    print(f"    cands:    narrow={n75['cands/day']}/日  broad={b75['cands/day']}/日"
          f"  (Δ={float(b75['cands/day'])-float(n75['cands/day']):+.2f})")

    print("\n  目標: Sharpe>0.8 / exposure 25〜30%")
    passed = [r for r in rows if r["verdict"] in ("✅ 両方",)]
    if passed:
        print(f"  ✅ 合格設定 ({len(passed)}件):")
        for r in passed:
            print(f"    {r['RSR_context']} / min_rsr={r['min_rsr']}"
                  f"  → Sharpe={r['Sharpe']}  exp={r['exp%']}%")
    else:
        print("  ⚠ 完全合格なし → 最良候補:")
        best = min(rows, key=lambda r: abs(float(r["exp%"]) - 27.5) + max(0, 0.8 - float(r["Sharpe"])) * 10)
        print(f"    {best['RSR_context']} / min_rsr={best['min_rsr']}"
              f"  → Sharpe={best['Sharpe']}  exp={best['exp%']}%")

    print("=" * 76)
    return 0


if __name__ == "__main__":
    sys.exit(main())
