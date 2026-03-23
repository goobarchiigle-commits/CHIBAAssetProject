"""
backtest/universe_parallel_test.py

ユニバース拡張の並列テスト（Step 1 of supply improvement plan）

Universe A（現行）: TEMPORAL24 取引 + RSR42 コンテキスト
Universe B（拡張）: RSR42 全取引可能 + RSR42 コンテキスト（取引機会の増加を検証）

見たい指標:
  - signals_per_day（cands/日）
  - trade_count（年間取引数）
  - avg_exposure（平均投資比率）
  - Sharpe / MaxDD / CAGR / Calmar

実行:
  python -m backtest.universe_parallel_test
  python -m backtest.universe_parallel_test --start 2018-01-01 --end 2024-12-31
  python -m backtest.universe_parallel_test --min-rsr 70
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

from backtest.live_equivalent import (
    run_backtest,
    _load_temporal24,
    _load_rsr_universe_42,
    START,
    END,
    MIN_RSR,
)


def run_comparison(start: str, end: str, min_rsr: float) -> None:
    # ---- ユニバース定義 ----
    temporal24 = _load_temporal24()
    rsr42      = _load_rsr_universe_42()

    print("=" * 72)
    print("  ユニバース拡張 並列テスト（Step 1: supply 改善）")
    print(f"  期間: {start} 〜 {end}  /  min_rsr={min_rsr}")
    print("=" * 72)
    print(f"\n  Universe A（現行）: TEMPORAL24 {len(temporal24)}銘柄")
    print(f"  Universe B（拡張）: RSR42      {len(rsr42)}銘柄")

    # A と B の差分を表示
    a_only = set(temporal24) - set(rsr42)
    b_only = set(rsr42) - set(temporal24)
    print(f"\n  A のみ（{len(a_only)}銘柄）: {', '.join(sorted(a_only)[:10])}")
    print(f"  B 追加分（{len(b_only)}銘柄）: {', '.join(sorted(b_only)[:10])}")

    # ---- Universe A バックテスト ----
    print("\n" + "─" * 72)
    print("  [Universe A] TEMPORAL24 実行中...")
    print("─" * 72)
    m_a = run_backtest(
        start=start, end=end,
        min_rsr=min_rsr,
        trade_universe=temporal24,
        verbose=True,
    )

    # ---- Universe B バックテスト ----
    print("\n" + "─" * 72)
    print("  [Universe B] RSR42 実行中...")
    print("─" * 72)
    m_b = run_backtest(
        start=start, end=end,
        min_rsr=min_rsr,
        trade_universe=rsr42,
        verbose=True,
    )

    # ---- 結果比較 ----
    _print_comparison(m_a, m_b, temporal24, rsr42)


def _print_comparison(m_a: dict, m_b: dict, uni_a: dict, uni_b: dict) -> None:
    print("\n" + "=" * 72)
    print("  比較結果")
    print("=" * 72)
    fmt_row = "  {:<22} {:>14} {:>14}  {:>8}"
    print(fmt_row.format("指標", "Universe A", "Universe B", "変化"))
    print("  " + "-" * 70)

    def diff(a, b, pct=False):
        d = b - a
        if pct:
            return f"{d:>+.2%}"
        return f"{d:>+.3f}"

    rows = [
        ("銘柄数",             f"{len(uni_a)}",       f"{len(uni_b)}",       f"{len(uni_b)-len(uni_a):+d}"),
        ("CAGR",              f"{m_a['cagr']:>+.2%}", f"{m_b['cagr']:>+.2%}", diff(m_a['cagr'],m_b['cagr'],True)),
        ("Sharpe",            f"{m_a['sharpe']:.3f}", f"{m_b['sharpe']:.3f}", diff(m_a['sharpe'],m_b['sharpe'])),
        ("MaxDD",             f"{m_a['max_dd']:.2%}", f"{m_b['max_dd']:.2%}", diff(m_a['max_dd'],m_b['max_dd'],True)),
        ("Calmar",            f"{m_a['calmar']:.3f}", f"{m_b['calmar']:.3f}", diff(m_a['calmar'],m_b['calmar'])),
        ("avg_exposure",      f"{m_a['avg_exposure']:.1%}", f"{m_b['avg_exposure']:.1%}", diff(m_a['avg_exposure'],m_b['avg_exposure'],True)),
        ("avg_cands/日",      f"{m_a['avg_candidates']:.3f}", f"{m_b['avg_candidates']:.3f}", diff(m_a['avg_candidates'],m_b['avg_candidates'])),
        ("zero_exp率",        f"{m_a['zero_exp_rate']:.1f}%", f"{m_b['zero_exp_rate']:.1f}%", f"{m_b['zero_exp_rate']-m_a['zero_exp_rate']:+.1f}pp"),
        ("取引数",             f"{m_a['trade_count']}",        f"{m_b['trade_count']}",        f"{m_b['trade_count']-m_a['trade_count']:+d}"),
        ("勝率",               f"{m_a['win_rate']:.1%}",       f"{m_b['win_rate']:.1%}",       diff(m_a['win_rate'],m_b['win_rate'],True)),
    ]
    for row in rows:
        print(fmt_row.format(*row))

    print("\n  ── 判定 ──")
    for label, m in [("A", m_a), ("B", m_b)]:
        exp = m['avg_exposure']
        cands = m['avg_candidates']
        sharpe = m['sharpe']
        print(f"\n  Universe {label}:")
        print(f"    avg_exposure  {exp:.1%}  {'✅ 正常(23~35%)' if 0.23<=exp<=0.35 else '⚠ '+('低い' if exp<0.15 else '高すぎ' if exp>0.45 else '許容範囲')}")
        print(f"    avg_cands/日  {cands:.2f}  {'✅ 十分(>0.3)' if cands>=0.3 else '⚠ 低い(<0.3) 供給不足'}")
        print(f"    Sharpe        {sharpe:.3f}  {'✅ >1.0' if sharpe>1.0 else '⚠ 目標未達'}")

    # 結論
    print("\n  ── 結論 ──")
    cand_a = m_a['avg_candidates']
    cand_b = m_b['avg_candidates']
    exp_a  = m_a['avg_exposure']
    exp_b  = m_b['avg_exposure']

    if cand_b > cand_a * 1.5:
        print(f"  ✅ Universe B で候補数 {cand_a:.2f}→{cand_b:.2f}/日（{cand_b/cand_a:.1f}倍）")
        print("     ユニバース拡張で供給改善を確認。run_live_signal.py への反映を検討。")
    else:
        print(f"  △ 候補数の改善: {cand_a:.2f}→{cand_b:.2f}/日（限定的）")
        print("     RSR閾値の引き下げ（min_rsr=75→70）も併せて検討。")

    if m_b['sharpe'] < 0.8:
        print("  ⚠ Universe B の Sharpe が低い → セクター適合性に問題がある可能性。")
        print("     RSR42の中で フジコ法適合セクター（海運・機械・電機精密）のみに絞ることを検討。")

    print("\n" + "=" * 72)


def main() -> int:
    p = argparse.ArgumentParser(description="ユニバース拡張 並列テスト")
    p.add_argument("--start",   default=START)
    p.add_argument("--end",     default=END)
    p.add_argument("--min-rsr", type=float, default=MIN_RSR)
    args = p.parse_args()

    run_comparison(start=args.start, end=args.end, min_rsr=args.min_rsr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
