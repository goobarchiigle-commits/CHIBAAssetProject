"""
src/backtest/alloc_wf.py

alloc 上限除去後の検証スクリプト。

計測:
  1. 不変性 (全期間 2018-2025): ticker一致率 / weight差 / CAGR差
  2. WF (年次 per-fold): 3M vs 30M CAGR差 × 5年

成功条件:
  ticker一致率 >= 97%
  weight差     <= 1pp
  CAGR差       <= 2pp (全期間 + 各年)
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
WF_YEARS     = [2021, 2022, 2023, 2024, 2025]

MATCH_THR  = 97.0
WEIGHT_THR = 1.0
CAGR_THR   = 2.0

OUT_DIR = Path("analysis")
OUT_DIR.mkdir(exist_ok=True)


def make_cfg(cfg, capital: int):
    return dc_replace(cfg, portfolio=dc_replace(cfg.portfolio, capital=capital))


def reconstruct_daily(trades: list, dates_list: list[str]) -> dict[str, set]:
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


def run_one(label: str, cap: int, cfg_base, universe_raw, rsr_df, alpha_df,
            sym_active_df, regime_df, topix_close, rsr_syms) -> dict:
    cfg_c = make_cfg(cfg_base, cap)
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
    print(f"  [{label}] CAGR={res.get('cagr',0):+.1f}%  "
          f"trades={n_sells}  cash_skipped={len(skipped)}")
    return {
        "cap":    cap,
        "trades": trades,
        "dates":  dates,
        "daily":  reconstruct_daily(trades, dates),
        "skipped_by_sym": {ev["sym"]: ev for ev in skipped},
        "metrics": {k: v for k, v in res.items() if not k.startswith("_")},
    }


# ─────────────────────────────────────────────────────────────────────

def invariance_report(results: dict, labels: list[str], ref_dates: list[str]) -> dict:
    base_label = labels[0]
    base = results[base_label]

    # weight lookup: (entry_idx, sym) → weight%
    base_key_w: dict[tuple, float] = {}
    for t in base["trades"]:
        if t["side"] == "BUY":
            base_key_w[(t["entry_idx"], t["symbol"])] = entry_weight_pct(t, base["cap"])

    rows_match, rows_weight = [], []
    for other in labels[1:]:
        other_r = results[other]

        # ticker match
        common = [d for d in ref_dates if d in other_r["daily"]]
        scores = [jaccard(base["daily"][d], other_r["daily"][d]) for d in common]
        rate   = float(np.mean(scores)) * 100 if scores else 0.0

        # weight diff
        diffs = []
        for t in other_r["trades"]:
            if t["side"] != "BUY": continue
            key = (t["entry_idx"], t["symbol"])
            if key in base_key_w:
                diffs.append(abs(base_key_w[key] - entry_weight_pct(t, other_r["cap"])))
        mean_d = float(np.mean(diffs)) if diffs else 0.0
        max_d  = float(np.max(diffs))  if diffs else 0.0

        rows_match.append({"pair": f"{base_label} vs {other}",
                           "rate": rate, "ok": rate >= MATCH_THR})
        rows_weight.append({"pair": f"{base_label} vs {other}",
                            "mean": mean_d, "max": max_d, "ok": max_d < WEIGHT_THR})
    return {"match": rows_match, "weight": rows_weight}


def wf_annual_report(results: dict, labels: list[str]) -> list[dict]:
    """Per-year CAGR for each capital level."""
    rows = []
    for year in WF_YEARS:
        row = {"year": year}
        for label in labels:
            ar = results[label]["metrics"].get("annual_returns", {})
            row[label] = ar.get(str(year), None)
        rows.append(row)
    return rows


def cagr_diff_ok(rows: list[dict], base: str, other: str) -> list[bool]:
    results = []
    for row in rows:
        a = row.get(base)
        b = row.get(other)
        if a is None or b is None:
            results.append(None)
        else:
            results.append(abs(a - b) <= CAGR_THR)
    return results


def violation_summary(results: dict, base_label: str, ref_dates: list[str]) -> None:
    labels = [l for l in results if l != base_label]
    counts = {"ALLOC_CAP": 0, "CASCADE": 0}
    for other in labels:
        other_r = results[other]
        for d in ref_dates:
            s_base  = results[base_label]["daily"].get(d, set())
            s_other = other_r["daily"].get(d, set())
            for sym in s_other - s_base:
                ev = results[base_label]["skipped_by_sym"].get(sym)
                counts["ALLOC_CAP" if ev else "CASCADE"] += 1
            for sym in s_base - s_other:
                ev = other_r["skipped_by_sym"].get(sym)
                counts["ALLOC_CAP" if ev else "CASCADE"] += 1
    total = sum(counts.values())
    if total > 0:
        print(f"  違反原因: ALLOC_CAP={counts['ALLOC_CAP']}件({counts['ALLOC_CAP']/total*100:.0f}%)  "
              f"CASCADE={counts['CASCADE']}件({counts['CASCADE']/total*100:.0f}%)")
        # Identify remaining ALLOC_CAP stocks
        alloc_syms = set()
        for other in labels:
            other_r = results[other]
            for d in ref_dates:
                s_base  = results[base_label]["daily"].get(d, set())
                s_other = other_r["daily"].get(d, set())
                for sym in s_other - s_base:
                    ev = results[base_label]["skipped_by_sym"].get(sym)
                    if ev:
                        alloc_syms.add(f"{sym}(lot=¥{ev['lot_cost']:.0f})")
        if alloc_syms:
            print(f"  残存ALLOC_CAP銘柄: {', '.join(sorted(alloc_syms))}")
    else:
        print("  違反なし")


def main():
    cfg_base = load_strategy_config()
    print("データ読み込み中...")
    (universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
     rsr_syms, topix_close, cfg_base) = load_data(cfg_base)

    print("\n=== 全期間実行 (2018-2025) ===")
    results: dict[str, dict] = {}
    for label, cap in CAPITAL_CASES:
        results[label] = run_one(
            label, cap, cfg_base, universe_raw, rsr_df, alpha_df,
            sym_active_df, regime_df, topix_close, rsr_syms,
        )

    labels     = [l for l, _ in CAPITAL_CASES]
    base_label = labels[0]   # 3M
    ref_dates  = results[base_label]["dates"]

    print("\n\n" + "=" * 65)
    print("  alloc cap除去 — 不変性 + WF 検証")
    print("=" * 65 + "\n")

    # ── Section 1: Ticker 一致率 ──────────────────────────────────────
    inv = invariance_report(results, labels, ref_dates)

    print("## 1. Ticker 一致率 (Jaccard 平均, 全営業日)")
    print(f"  {'比較ペア':<15} {'一致率':>7}  {'基準':>5}  判定")
    print(f"  {'-'*42}")
    for r in inv["match"]:
        print(f"  {r['pair']:<15} {r['rate']:>6.1f}%  {MATCH_THR:.0f}%  "
              f"{'✅ PASS' if r['ok'] else '❌ FAIL'}")

    # ── Section 2: Weight 差分 ────────────────────────────────────────
    print("\n## 2. ウェイト差分 (entry時, 同銘柄・同entry_idx)")
    print(f"  {'比較ペア':<15} {'mean差':>7}  {'max差':>7}  {'基準':>5}  判定")
    print(f"  {'-'*50}")
    for r in inv["weight"]:
        print(f"  {r['pair']:<15} {r['mean']:>6.2f}%  {r['max']:>6.2f}%  "
              f"{'<1%':>5}  {'✅ PASS' if r['ok'] else '❌ FAIL'}")

    # ── Section 3: 稼働率 / CAGR ─────────────────────────────────────
    print("\n## 3. 全期間 CAGR / MaxDD / 稼働率")
    print(f"  {'ケース':<6} {'CAGR':>7}  {'MaxDD':>7}  {'稼働率':>8}  {'idle':>6}")
    print(f"  {'-'*44}")
    for label in labels:
        m = results[label]["metrics"]
        print(f"  {label:<6} {m.get('cagr',0):>+6.1f}%  "
              f"{m.get('max_dd',0):>6.1f}%  "
              f"{m.get('avg_exposure',0):>7.1f}%  "
              f"{m.get('idle_cash',0):>5.1f}%")

    # ── Section 4: CAGR差 (全期間) ───────────────────────────────────
    print("\n## 4. 全期間 CAGR差")
    base_cagr = results[base_label]["metrics"].get("cagr", 0.0)
    cagr_ok_full = []
    for other in labels[1:]:
        other_cagr = results[other]["metrics"].get("cagr", 0.0)
        diff = abs(base_cagr - other_cagr)
        ok   = diff <= CAGR_THR
        cagr_ok_full.append(ok)
        print(f"  {base_label} vs {other}: |{base_cagr:+.1f}% - {other_cagr:+.1f}%| = {diff:.1f}pp  "
              f"{'✅' if ok else '❌'}")

    # ── Section 5: WF (年次 CAGR) ────────────────────────────────────
    print("\n## 5. WF — 年次 CAGR (per-fold比較)")
    wf_rows = wf_annual_report(results, labels)

    # Header
    hdr = f"  {'年':>4}"
    for label in labels:
        hdr += f"  {label:>7}"
    hdr += f"  {'3M-30M差':>8}  判定"
    print(hdr)
    print(f"  {'-' * (4 + len(labels)*9 + 22)}")

    wf_pass_count = 0
    wf_total = 0
    for row in wf_rows:
        yr = row["year"]
        line = f"  {yr}"
        for label in labels:
            v = row.get(label)
            line += f"  {v:>+6.1f}%" if v is not None else f"  {'N/A':>7}"
        v3 = row.get("3M")
        v30 = row.get("30M")
        if v3 is not None and v30 is not None:
            diff = abs(v3 - v30)
            ok   = diff <= CAGR_THR
            wf_total += 1
            if ok:
                wf_pass_count += 1
            line += f"  {diff:>7.1f}pp  {'✅' if ok else '❌'}"
        print(line)

    wf_note = f"({wf_pass_count}/{wf_total} years pass)"
    print(f"  WF判定: {wf_note}")

    # ── Section 6: 違反原因 ──────────────────────────────────────────
    print("\n## 6. 違反原因分類")
    violation_summary(results, base_label, ref_dates)

    # ── Section 7: 最終判定 ──────────────────────────────────────────
    all_match_ok  = all(r["ok"] for r in inv["match"])
    all_weight_ok = all(r["ok"] for r in inv["weight"])
    all_cagr_ok   = all(cagr_ok_full) and (wf_pass_count == wf_total)

    print("\n## 7. 最終判定")
    print(f"  ticker一致率 ≥ {MATCH_THR:.0f}% : {'PASS ✅' if all_match_ok  else 'FAIL ❌'}")
    print(f"  weight差 ≤ {WEIGHT_THR:.0f}pp   : {'PASS ✅' if all_weight_ok else 'FAIL ❌'}")
    print(f"  CAGR差 ≤ {CAGR_THR:.0f}pp       : {'PASS ✅' if all_cagr_ok  else 'FAIL ❌'}")

    if not (all_match_ok and all_weight_ok and all_cagr_ok):
        print("\n  --- 残存違反の条件分岐 ---")
        if not all_match_ok:
            print("  [ticker-mismatch] 残存原因:")
            print("    Branch: alloc = deployable / eff_slots")
            print("    3M: alloc ≈ 3M/3 = ¥1,000,000 per slot")
            print("    lot_cost > ¥1,000,000 の銘柄は依然 3M で購入不可")
            print("    → 修正案: 最低 1lot 保証 or 高額銘柄を universe から除外")
        if not all_weight_ok:
            print("  [weight-diff] 残存原因:")
            print("    Branch: qty = int(alloc/buy_px/LOT) × LOT")
            print("    3M: max_rounding = 1lot × buy_px / 3M  (例: ¥3000株 → 10pp)")
            print("    → 修正案: alloc = capital/max_pos (常に capital 基準、cash推移に非依存)")
        if not all_cagr_ok:
            print("  [CAGR-diff] 残存原因:")
            print("    capital規模でポートフォリオが diverge → 実現 CAGR に差")
    else:
        print("\n  → 資本スケール不変性: PASS ✅")

    # CSV 保存
    pd.DataFrame(inv["match"]).to_csv(OUT_DIR / "alloc_wf_match.csv",  index=False)
    pd.DataFrame(inv["weight"]).to_csv(OUT_DIR / "alloc_wf_weight.csv", index=False)
    pd.DataFrame(wf_rows).to_csv(OUT_DIR / "alloc_wf_annual.csv",      index=False)
    print(f"\n出力: analysis/alloc_wf_*.csv")


if __name__ == "__main__":
    main()
