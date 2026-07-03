"""
src/backtest/capital_scale_invariance.py

資本スケール不変性(invariance)検証
同一シグナル・同一 rank の下で開始資金のみ変更:
  3M / 5M / 10M / 30M

比較: selected_tickers / position_count / weight / capital_utilization
期待: ticker 一致率 > 95% / weight 差 < 1%
違反時: どの条件分岐が変化原因か出力

禁止: MSW変更 / strategy変更
"""

from __future__ import annotations
import sys, warnings
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from dataclasses import replace as dc_replace
from pathlib import Path

from src.config_loader import load_strategy_config
from src.backtest.capital_allocation_abc import load_data, run_period

CAPITAL_CASES = [
    ("3M",  3_000_000),
    ("5M",  5_000_000),
    ("10M", 10_000_000),
    ("30M", 30_000_000),
]
PERIOD_START = "2018-01-01"
PERIOD_END   = "2025-12-31"
OUT_DIR      = Path("analysis")
OUT_DIR.mkdir(exist_ok=True)

MATCH_THRESHOLD  = 95.0   # ticker 一致率 (%)
WEIGHT_THRESHOLD = 1.0    # ウェイト差上限 (pp)


# ─────────────────────────────────────────────────────────────────────

def make_cfg(cfg, capital: int):
    """frozen StrategyConfig から capital のみ変更したコピーを返す."""
    new_portfolio = dc_replace(cfg.portfolio, capital=capital)
    return dc_replace(cfg, portfolio=new_portfolio)


def reconstruct_daily(trades: list, dates_list: list[str]) -> dict[str, set]:
    """trades → {date_str: set[symbol]}  entry_idx <= idx < exit_idx の日が保有日."""
    n = len(dates_list)
    daily: dict[str, set] = {d: set() for d in dates_list}
    for t in trades:
        if t["side"] != "BUY":
            continue
        ei = t["entry_idx"]
        xi = t.get("exit_idx", n)
        for idx in range(max(0, ei), min(n, xi)):
            daily[dates_list[idx]].add(t["symbol"])
    return daily


def jaccard(a: set, b: set) -> float:
    u = a | b
    return len(a & b) / len(u) if u else 1.0


def entry_weight_pct(t: dict, capital: int) -> float:
    return t["qty"] * t["entry"] / capital * 100


def lot_rounding_error_pp(buy_px: float, capital: int, msw: float = 0.25) -> float:
    """1 lot 端数の最大ウェイト誤差 = (1 lot × buy_px) / capital × 100."""
    alloc = capital * msw
    n_lots = int(alloc / buy_px / 100)  # 整数部
    ideal_value = alloc
    actual_value = n_lots * 100 * buy_px
    return abs(ideal_value - actual_value) / capital * 100


# ─────────────────────────────────────────────────────────────────────

def main():
    cfg_base = load_strategy_config()
    msw = float(cfg_base.portfolio.max_single_weight)

    print("データ読み込み中...")
    (universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
     rsr_syms, topix_close, cfg_base) = load_data(cfg_base)

    results: dict[str, dict] = {}
    for label, cap in CAPITAL_CASES:
        cfg_c = make_cfg(cfg_base, cap)
        print(f"\n[{label}] capital=¥{cap:,}  alloc_cap=¥{cap*msw:,.0f}")
        res = run_period(
            universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
            topix_close, rsr_syms, cfg_c,
            start=PERIOD_START, end=PERIOD_END,
            pattern="A",
            return_trades=True,
            return_cash_skipped=True,
            max_new_per_day=2,
            rsr_exit_override=70.0,
            verbose=False,
        )
        trades  = res.get("_trades_raw", [])
        dates   = res.get("_common_dates", [])
        skipped = res.get("_cash_skipped", [])
        n_sells = sum(1 for t in trades if t["side"] == "SELL")
        print(f"  CAGR={res.get('cagr',0):+.1f}%  "
              f"n_trades={n_sells}  cash_skipped={len(skipped)}")

        results[label] = {
            "cap":    cap,
            "trades": trades,
            "dates":  dates,
            "daily":  reconstruct_daily(trades, dates),
            # sym -> latest skip event (lot_cost / alloc)
            "skipped_by_sym": {ev["sym"]: ev for ev in skipped},
            "metrics": {k: v for k, v in res.items() if not k.startswith("_")},
        }

    labels     = [l for l, _ in CAPITAL_CASES]
    base_label = labels[0]   # 3M
    base       = results[base_label]
    ref_dates  = base["dates"]

    print("\n\n" + "=" * 60)
    print("  資本スケール不変性検証 (Invariance Audit)")
    print(f"  期間: {PERIOD_START} ~ {PERIOD_END}  /  基準: {base_label} (¥3,000,000)")
    print(f"  MAX_SINGLE_W: {msw}  /  alloc_cap(3M)=¥{3_000_000*msw:,.0f}")
    print("=" * 60 + "\n")

    # ── 1. Ticker 一致率 ─────────────────────────────────────────────
    print("## 1. Ticker 一致率 (Jaccard 平均, 全営業日)")
    print(f"  {'比較ペア':<15} {'一致率':>8}  {'基準':>6}  判定")
    print(f"  {'-'*44}")
    match_rows = []
    for other in labels[1:]:
        other_r = results[other]
        common  = [d for d in ref_dates if d in other_r["daily"]]
        scores  = [jaccard(base["daily"][d], other_r["daily"][d]) for d in common]
        rate    = float(np.mean(scores)) * 100 if scores else 0.0
        ok      = rate >= MATCH_THRESHOLD
        print(f"  {base_label} vs {other:<10} {rate:>7.1f}%  {MATCH_THRESHOLD:>5.0f}%  "
              f"{'✅ PASS' if ok else '❌ FAIL'}")
        match_rows.append({
            "pair": f"{base_label} vs {other}",
            "match_rate_pct": round(rate, 2),
            "pass": ok,
        })

    # ── 2. ウェイト差分 (同銘柄 × 同 entry_idx) ─────────────────────
    print("\n## 2. ウェイト差分 (entry時, 同銘柄・同 entry_idx)")
    print(f"  {'比較ペア':<15} {'mean差':>7}  {'max差':>7}  {'基準':>5}  判定")
    print(f"  {'-'*52}")

    base_key_w: dict[tuple, float] = {}
    for t in base["trades"]:
        if t["side"] == "BUY":
            base_key_w[(t["entry_idx"], t["symbol"])] = entry_weight_pct(t, base["cap"])

    weight_rows = []
    for other in labels[1:]:
        diffs = []
        for t in results[other]["trades"]:
            if t["side"] != "BUY":
                continue
            key = (t["entry_idx"], t["symbol"])
            if key in base_key_w:
                diffs.append(abs(base_key_w[key] - entry_weight_pct(t, results[other]["cap"])))
        mean_d = float(np.mean(diffs)) if diffs else 0.0
        max_d  = float(np.max(diffs))  if diffs else 0.0
        ok     = max_d < WEIGHT_THRESHOLD
        print(f"  {base_label} vs {other:<10} {mean_d:>6.2f}%  {max_d:>6.2f}%  "
              f"{'<1%':>5}  {'✅ PASS' if ok else '❌ FAIL'}")
        weight_rows.append({
            "pair": f"{base_label} vs {other}",
            "mean_diff_pct": round(mean_d, 3),
            "max_diff_pct":  round(max_d, 3),
            "pass": ok,
        })

    # ── 3. ポジション数不一致 ────────────────────────────────────────
    print("\n## 3. ポジション数不一致日数")
    for other in labels[1:]:
        other_r = results[other]
        n_diff = sum(
            1 for d in ref_dates
            if len(base["daily"].get(d, set())) != len(other_r["daily"].get(d, set()))
        )
        print(f"  {base_label} vs {other:<10}: {n_diff}/{len(ref_dates)} 日 "
              f"({n_diff/len(ref_dates)*100:.1f}%)")

    # ── 4. 稼働率比較 ───────────────────────────────────────────────
    print("\n## 4. 稼働率 / CAGR / MaxDD")
    print(f"  {'ケース':<6} {'CAGR':>7}  {'MaxDD':>8}  {'稼働率':>8}  {'idle':>6}")
    print(f"  {'-'*44}")
    for label in labels:
        m = results[label]["metrics"]
        print(f"  {label:<6} {m.get('cagr',0):>+6.1f}%  "
              f"{m.get('max_dd',0):>7.1f}%  "
              f"{m.get('avg_exposure',0):>7.1f}%  "
              f"{m.get('idle_cash',0):>5.1f}%")

    # ── 5. 違反原因分類 ──────────────────────────────────────────────
    print("\n## 5. 違反原因分類")

    violations = []
    for other in labels[1:]:
        other_r = results[other]
        for d in ref_dates:
            s_base  = base["daily"].get(d, set())
            s_other = other_r["daily"].get(d, set())

            # in other not base → base が cash-skip または cascade
            for sym in s_other - s_base:
                ev = base["skipped_by_sym"].get(sym)
                if ev:
                    cause = (f"ALLOC_CAP ({base_label}: "
                             f"alloc=¥{ev['alloc']:.0f} lot=¥{ev['lot_cost']:.0f})")
                else:
                    cause = "CASCADE/SECTOR/GROSS"
                violations.append({
                    "date": d, "sym": sym,
                    "missing_in": base_label, "have_in": other,
                    "cause_type": "ALLOC_CAP" if ev else "CASCADE",
                    "cause": cause,
                })

            # in base not other → other が cascade またはまれに alloc_cap
            for sym in s_base - s_other:
                ev = other_r["skipped_by_sym"].get(sym)
                if ev:
                    cause = (f"ALLOC_CAP ({other}: "
                             f"alloc=¥{ev['alloc']:.0f} lot=¥{ev['lot_cost']:.0f})")
                else:
                    cause = "CASCADE/SECTOR/GROSS"
                violations.append({
                    "date": d, "sym": sym,
                    "missing_in": other, "have_in": base_label,
                    "cause_type": "ALLOC_CAP" if ev else "CASCADE",
                    "cause": cause,
                })

    if violations:
        vdf = pd.DataFrame(violations)
        # 重複除去: (date, sym, missing_in) 単位
        unique = vdf.drop_duplicates(["date", "sym", "missing_in"]).copy()

        type_counts = unique["cause_type"].value_counts()
        print(f"  総違反 (日×銘柄×ペア, deduped): {len(unique)}")
        for cause_t, cnt in type_counts.items():
            pct = cnt / len(unique) * 100
            print(f"    {cause_t:<20}: {cnt:4d}件 ({pct:.1f}%)")

        print(f"\n  代表的な違反 (上位20件, 日付昇順):")
        print(f"  {'日付':<12} {'銘柄':<10} {'欠落ケース':<6} {'原因'}")
        for _, row in unique.sort_values("date").head(20).iterrows():
            print(f"  {row['date']:<12} {row['sym']:<10} {row['missing_in']:<6} {row['cause']}")

        unique.to_csv(OUT_DIR / "invariance_violations.csv", index=False)
    else:
        print("  違反なし — 完全不変性 ✅")

    # ── 6. 最終判定 ──────────────────────────────────────────────────
    all_match_ok  = all(r["pass"] for r in match_rows)
    all_weight_ok = all(r["pass"] for r in weight_rows)

    print("\n## 6. 最終判定")
    print(f"  ticker 一致率 ≥ {MATCH_THRESHOLD:.0f}% : "
          f"{'PASS ✅' if all_match_ok else 'FAIL ❌'}")
    print(f"  weight 差 < {WEIGHT_THRESHOLD:.0f}pp   : "
          f"{'PASS ✅' if all_weight_ok else 'FAIL ❌'}")

    if not all_match_ok or not all_weight_ok:
        print("\n  --- 違反した条件分岐 ---")
        if not all_match_ok:
            print("  [ticker-mismatch]")
            print("    Branch: alloc = min(deployable/eff_slots, capital × MAX_SINGLE_W)")
            print(f"    3M cap: ¥{3_000_000*msw:,.0f}  →  lot_cost > ¥{3_000_000*msw:,.0f} の銘柄は購入不可")
            print(f"    30M cap: ¥{30_000_000*msw:,.0f} → 10倍の銘柄が解禁")
            print("    行番号: capital_allocation_abc.py  alloc = min(alloc, capital * MAX_SINGLE_W)")
        if not all_weight_ok:
            print("  [weight-diff]")
            print("    Branch: qty = int(alloc / buy_px / LOT) × LOT  (整数丸め)")
            print("    誤差上界: 1 lot × buy_px / capital")
            for label, cap in CAPITAL_CASES:
                # 代表的な誤差: buy_px=1000円 (¥100K/lot)
                err = 100_000 / cap * 100
                print(f"    {label}: buy_px=¥1,000 → 最大誤差 {err:.2f}pp")

    # ── CSV 保存 ─────────────────────────────────────────────────────
    pd.DataFrame(match_rows).to_csv(OUT_DIR / "invariance_ticker_match.csv", index=False)
    pd.DataFrame(weight_rows).to_csv(OUT_DIR / "invariance_weight_diff.csv", index=False)
    print(f"\n出力:")
    print(f"  analysis/invariance_ticker_match.csv")
    print(f"  analysis/invariance_weight_diff.csv")
    if violations:
        print(f"  analysis/invariance_violations.csv")


if __name__ == "__main__":
    main()
