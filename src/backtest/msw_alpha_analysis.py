"""
src/backtest/msw_alpha_analysis.py

MSW Alpha Quality Analysis
判定: MAX_SINGLE_W拡張が「弱い銘柄混入」vs「本来Top3救済」か

実装変更禁止。観測のみ。比較: A=現行(MSW=0.25相当), B=候補(MSW=0.33)

Tasks:
  1: intended_top3.csv   — RSR Top3 (資本/lot制約無視)
  2: executed_positions.csv — actual vs intended gap
  3: missed_top3_alpha.csv  — rank<=3 missed: fwd 20d/60d/120d
  4: msw_rescue_curve.csv   — MSW sweep 0.25→0.35 rescued_alpha
  5: positive_rescue_rate   — rescues_with_alpha>0 / all_rescues
  Report: msw_alpha_report.md

制約: Top4以下分析対象外 / RSR順位変更禁止 / selectionロジック変更禁止
"""

from __future__ import annotations
import sys, warnings
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import re
import numpy as np
import pandas as pd
from collections import defaultdict
from dataclasses import replace as dc_replace
from pathlib import Path

from src.config_loader import load_strategy_config
from src.backtest.capital_allocation_abc import load_data, run_period, LOT

CAPITAL   = 3_000_000
START     = "2018-01-01"
END       = "2025-12-31"
MSW_BASE  = 0.25
MSW_SWEEP = [0.25, 0.28, 0.30, 0.33, 0.35]
FWD_DAYS  = [20, 60, 120]
TOP_N     = 3    # Top4以下は分析対象外

OUT_DIR = Path("analysis")
OUT_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────

def _parse_rsr(reason: str) -> float:
    """Parse RSR from trade reason string 'RSR=xx.x'."""
    m = re.search(r"RSR=([\d.]+)", reason or "")
    return float(m.group(1)) if m else 0.0


def build_candidates_by_date(
    trades: list,
    cash_skipped: list,
    missed_details: list,
    dates_list: list[str],
) -> dict[str, list[dict]]:
    """
    Per-date union of all buy candidates:
      - EXECUTED: from trades BUY side
      - LOT_COST: from cash_skipped (lot_cost > alloc → qty=0)
      - CAPACITY: from missed_details (max_positions full)

    Returns: date_str → [cand, ...] sorted by RSR desc (intended rank).
    Each cand: {sym, rsr, buy_px, lot_cost, executed, reason, intended_rank}
    """
    cands: dict[str, dict[str, dict]] = defaultdict(dict)

    for t in trades:
        if t["side"] != "BUY":
            continue
        # entry_idx = next_i (= execution day), so intended signal = entry_idx - 1
        # Use entry_idx day as the "signal date" for consistency with skipped events
        date_str = dates_list[t["entry_idx"]] if t["entry_idx"] < len(dates_list) else None
        if date_str is None:
            continue
        sym = t["symbol"]
        rsr = _parse_rsr(t.get("reason", ""))
        buy_px = float(t["entry"])
        cands[date_str][sym] = {
            "sym":       sym,
            "rsr":       rsr,
            "buy_px":    buy_px,
            "lot_cost":  buy_px * LOT,
            "executed":  True,
            "reason":    "EXECUTED",
        }

    for ev in cash_skipped:
        date_str = ev["date"]
        sym = ev["sym"]
        if sym in cands[date_str]:
            continue  # already recorded as executed
        cands[date_str][sym] = {
            "sym":      sym,
            "rsr":      ev["rsr"],
            "buy_px":   ev["buy_px"],
            "lot_cost": ev["lot_cost"],
            "executed": False,
            "reason":   "LOT_COST",
        }

    for ev in missed_details:
        date_str = ev["date"]
        sym = ev["sym"]
        if sym in cands[date_str]:
            continue
        cands[date_str][sym] = {
            "sym":      sym,
            "rsr":      ev["rsr"],
            "buy_px":   ev["buy_px"],
            "lot_cost": ev["lot_cost"],
            "executed": False,
            "reason":   "CAPACITY",
        }

    result: dict[str, list[dict]] = {}
    for date_str, sym_dict in cands.items():
        sorted_cands = sorted(sym_dict.values(), key=lambda x: -x["rsr"])
        for rank, c in enumerate(sorted_cands):
            c["intended_rank"] = rank  # 0-indexed; 0=RSR#1
        result[date_str] = sorted_cands

    return result


def get_close_at(sym: str, date_idx: int, offset: int,
                 dates_list: list[str], universe_raw: dict) -> float | None:
    """Close price at dates_list[date_idx + offset]."""
    target_idx = date_idx + offset
    if target_idx >= len(dates_list) or sym not in universe_raw:
        return None
    target_date = pd.Timestamp(dates_list[target_idx])
    close = universe_raw[sym]["df"]["Close"]
    val = close.asof(target_date) if not close.empty else None
    if val is None or (isinstance(val, float) and (pd.isna(val) or val <= 0)):
        return None
    return float(val)


def fwd_return(sym: str, date_idx: int, fwd: int,
               dates_list: list[str], universe_raw: dict) -> float | None:
    """fwd-day return from day date_idx (buy day open proxy = day+1 close)."""
    p0 = get_close_at(sym, date_idx, 0, dates_list, universe_raw)
    p1 = get_close_at(sym, date_idx, fwd, dates_list, universe_raw)
    if p0 is None or p1 is None or p0 <= 0:
        return None
    return float(p1 / p0 - 1)


def date_to_idx(date_str: str, dates_list: list[str]) -> int | None:
    try:
        return dates_list.index(date_str)
    except ValueError:
        return None


# ─────────────────────────────────────────────────────────────────────
#  TASKS
# ─────────────────────────────────────────────────────────────────────

def task1_intended_top3(cands_by_date: dict) -> pd.DataFrame:
    """Task1: RSR Top3 per date (ignoring all constraints)."""
    rows = []
    for date_str, cands in sorted(cands_by_date.items()):
        for c in cands[:TOP_N]:
            rows.append({
                "date":          date_str,
                "intended_rank": c["intended_rank"] + 1,  # 1-indexed
                "sym":           c["sym"],
                "rsr":           c["rsr"],
                "buy_px":        round(c["buy_px"], 2),
                "lot_cost":      round(c["lot_cost"], 0),
                "executed":      c["executed"],
                "reason":        c["reason"],
            })
    return pd.DataFrame(rows)


def task2_execution_gap(cands_by_date: dict) -> pd.DataFrame:
    """Task2: Execution gap — intended top3 vs actual, with reject_reason."""
    rows = []
    for date_str, cands in sorted(cands_by_date.items()):
        executed_on_date = [c["sym"] for c in cands if c["executed"]]
        for c in cands[:TOP_N]:
            reject = None if c["executed"] else c["reason"]
            rows.append({
                "date":             date_str,
                "intended_rank":    c["intended_rank"] + 1,
                "sym":              c["sym"],
                "rsr":              c["rsr"],
                "lot_cost":         round(c["lot_cost"], 0),
                "executed":         c["executed"],
                "reject_reason":    reject,
                "n_executed_day":   len(executed_on_date),
            })
    return pd.DataFrame(rows)


def task3_missed_alpha(
    cands_by_date: dict,
    dates_list: list[str],
    universe_raw: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Task3: Missed alpha for rank<=3 (intended_rank 0-2) unexecuted positions.
    Returns (detail_df, summary_df).
    """
    rows = []
    for date_str, cands in sorted(cands_by_date.items()):
        d_idx = date_to_idx(date_str, dates_list)
        if d_idx is None:
            continue

        top3 = cands[:TOP_N]
        executed_top3 = [c for c in top3 if c["executed"]]
        missed_top3   = [c for c in top3 if not c["executed"]]

        if not missed_top3:
            continue

        # Counterfactual: mean fwd return of missed top3
        # Baseline: mean fwd return of executed top3
        for fwd in FWD_DAYS:
            cf_rets  = [fwd_return(c["sym"], d_idx, fwd, dates_list, universe_raw)
                        for c in missed_top3]
            bl_rets  = [fwd_return(c["sym"], d_idx, fwd, dates_list, universe_raw)
                        for c in executed_top3]
            cf_rets  = [r for r in cf_rets if r is not None]
            bl_rets  = [r for r in bl_rets if r is not None]

            for c in missed_top3:
                r = fwd_return(c["sym"], d_idx, fwd, dates_list, universe_raw)
                rows.append({
                    "date":           date_str,
                    "intended_rank":  c["intended_rank"] + 1,
                    "sym":            c["sym"],
                    "rsr":            c["rsr"],
                    "lot_cost":       round(c["lot_cost"], 0),
                    "reject_reason":  c["reason"],
                    "fwd_days":       fwd,
                    "fwd_ret":        round(r * 100, 3) if r is not None else None,
                    "executed_mean":  round(np.mean(bl_rets) * 100, 3) if bl_rets else None,
                    "alpha_loss":     round((r - np.mean(bl_rets)) * 100, 3)
                                      if (r is not None and bl_rets) else None,
                })

    detail = pd.DataFrame(rows)

    # Summary: alpha_loss by rank × fwd_days
    if detail.empty:
        return detail, pd.DataFrame()

    summary_rows = []
    for rank in range(1, TOP_N + 1):
        for fwd in FWD_DAYS:
            sub = detail[(detail["intended_rank"] == rank) & (detail["fwd_days"] == fwd)]
            sub_valid = sub.dropna(subset=["alpha_loss"])
            summary_rows.append({
                "intended_rank": rank,
                "fwd_days":      fwd,
                "n_missed":      len(sub),
                "n_valid":       len(sub_valid),
                "alpha_loss_mean": round(sub_valid["alpha_loss"].mean(), 3)
                                   if not sub_valid.empty else None,
                "missed_fwd_ret": round(sub_valid["fwd_ret"].mean(), 3)
                                  if not sub_valid.empty else None,
                "executed_mean":  round(sub_valid["executed_mean"].mean(), 3)
                                  if not sub_valid.empty else None,
            })

    summary = pd.DataFrame(summary_rows)
    return detail, summary


def task4_rescue_curve(
    cands_by_date: dict,
    dates_list: list[str],
    universe_raw: dict,
    capital: float = CAPITAL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Task4: MSW rescue value curve.
    For each MSW in sweep: find rank<=3 stocks newly accessible vs MSW_BASE.
    rescued_alpha = fwd_return(rescued) - fwd_return(displaced).
    Displaced = highest-RSR accessible stock at MSW_BASE in same slot.
    Returns (events_df, curve_df).
    """
    events = []
    for msw in MSW_SWEEP:
        threshold_base = capital * MSW_BASE   # ¥750,000 at 3M
        threshold_msw  = capital * msw

        for date_str, cands in sorted(cands_by_date.items()):
            d_idx = date_to_idx(date_str, dates_list)
            if d_idx is None:
                continue

            top3 = cands[:TOP_N]

            for c in top3:
                lot = c["lot_cost"]

                # "Newly rescued" at this MSW level (vs baseline MSW_BASE):
                # lot > threshold_base (blocked at base) AND lot <= threshold_msw (accessible)
                if not (lot > threshold_base and lot <= threshold_msw):
                    continue

                # Find "displaced" stock: highest RSR with lot_cost <= threshold_base
                # that would have filled the slot at MSW_BASE instead
                displaced_candidates = [
                    x for x in cands
                    if x["sym"] != c["sym"] and x["lot_cost"] <= threshold_base
                ]
                if not displaced_candidates:
                    continue
                displaced = displaced_candidates[0]  # already sorted by RSR desc

                for fwd in FWD_DAYS:
                    r_rescued  = fwd_return(c["sym"], d_idx, fwd, dates_list, universe_raw)
                    r_displaced = fwd_return(displaced["sym"], d_idx, fwd, dates_list, universe_raw)

                    if r_rescued is None or r_displaced is None:
                        continue

                    rescued_alpha = (r_rescued - r_displaced) * 100
                    events.append({
                        "msw":              msw,
                        "date":             date_str,
                        "rescued_sym":      c["sym"],
                        "rescued_rank":     c["intended_rank"] + 1,
                        "rescued_rsr":      c["rsr"],
                        "rescued_lot":      round(lot, 0),
                        "displaced_sym":    displaced["sym"],
                        "displaced_rank":   displaced["intended_rank"] + 1,
                        "displaced_rsr":    displaced["rsr"],
                        "fwd_days":         fwd,
                        "rescued_ret_pct":  round(r_rescued * 100, 3),
                        "displaced_ret_pct":round(r_displaced * 100, 3),
                        "rescued_alpha_pct":round(rescued_alpha, 3),
                    })

    events_df = pd.DataFrame(events)
    if events_df.empty:
        return events_df, pd.DataFrame()

    curve_rows = []
    for msw in MSW_SWEEP:
        for fwd in FWD_DAYS:
            sub = events_df[(events_df["msw"] == msw) & (events_df["fwd_days"] == fwd)]
            if sub.empty:
                curve_rows.append({
                    "msw": msw, "fwd_days": fwd,
                    "n_rescues": 0, "rescued_alpha_mean": None,
                    "pos_rescue_rate": None,
                })
                continue
            curve_rows.append({
                "msw":                 msw,
                "fwd_days":            fwd,
                "n_rescues":           len(sub),
                "rescued_alpha_mean":  round(sub["rescued_alpha_pct"].mean(), 3),
                "pos_rescue_rate":     round((sub["rescued_alpha_pct"] > 0).mean() * 100, 1),
                "rescued_ret_mean":    round(sub["rescued_ret_pct"].mean(), 3),
                "displaced_ret_mean":  round(sub["displaced_ret_pct"].mean(), 3),
            })

    return events_df, pd.DataFrame(curve_rows)


def task5_positive_rescue_rate(events_df: pd.DataFrame) -> pd.DataFrame:
    """Task5: Positive rescue rate by rank tier (rank1 / rank1-2 / rank1-3)."""
    if events_df.empty:
        return pd.DataFrame()

    rows = []
    for msw in MSW_SWEEP:
        for fwd in FWD_DAYS:
            sub = events_df[(events_df["msw"] == msw) & (events_df["fwd_days"] == fwd)]
            if sub.empty:
                continue
            for tier, max_rank in [("rank1", 1), ("rank1-2", 2), ("rank1-3", 3)]:
                tier_sub = sub[sub["rescued_rank"] <= max_rank]
                if tier_sub.empty:
                    continue
                rows.append({
                    "msw":              msw,
                    "fwd_days":         fwd,
                    "tier":             tier,
                    "n":                len(tier_sub),
                    "pos_rescue_rate":  round((tier_sub["rescued_alpha_pct"] > 0).mean() * 100, 1),
                    "mean_alpha_pct":   round(tier_sub["rescued_alpha_pct"].mean(), 3),
                })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────
#  REPORT
# ─────────────────────────────────────────────────────────────────────

def build_report(
    t2_df: pd.DataFrame,
    t3_summary: pd.DataFrame,
    t4_curve: pd.DataFrame,
    t5_df: pd.DataFrame,
    capital: float,
) -> str:
    lines = [
        "# MSW Alpha Quality Analysis",
        f"期間: {START} ~ {END}  /  capital: ¥{int(capital):,}  /  baseline MSW: {MSW_BASE}",
        "",
    ]

    # ── Section 1: Execution Gap Summary
    lines.append("## 1. Execution Gap (Top3 Intended vs Actual)")
    if not t2_df.empty:
        total = len(t2_df)
        exec_n = t2_df["executed"].sum()
        miss_n = total - exec_n
        lines.append(f"- 総観測(日×rank<=3): {total}")
        lines.append(f"- 実行: {exec_n} ({exec_n/total*100:.1f}%)")
        lines.append(f"- 未実行: {miss_n} ({miss_n/total*100:.1f}%)")
        reason_counts = t2_df[~t2_df["executed"]]["reject_reason"].value_counts()
        for reason, cnt in reason_counts.items():
            lines.append(f"  - {reason}: {cnt}件")
    lines.append("")

    # ── Section 2: Missed Alpha Summary (Task 3)
    lines.append("## 2. Missed Alpha — rank<=3 未実行ポジション")
    if not t3_summary.empty:
        lines.append("| rank | fwd_days | n_missed | alpha_loss(pp) | missed_ret(pp) | exec_ret(pp) |")
        lines.append("|------|----------|----------|----------------|----------------|--------------|")
        for _, row in t3_summary.iterrows():
            al = f"{row['alpha_loss_mean']:+.2f}" if pd.notna(row['alpha_loss_mean']) else "N/A"
            mr = f"{row['missed_fwd_ret']:+.2f}" if pd.notna(row['missed_fwd_ret']) else "N/A"
            er = f"{row['executed_mean']:+.2f}" if pd.notna(row['executed_mean']) else "N/A"
            lines.append(f"| {int(row['intended_rank'])} | {int(row['fwd_days'])}d | "
                         f"{int(row['n_missed'])} | {al} | {mr} | {er} |")
    lines.append("")

    # ── Section 3: MSW Rescue Curve (Task 4)
    lines.append("## 3. MSW Rescue Curve (fwd60d基準)")
    if not t4_curve.empty:
        sub60 = t4_curve[t4_curve["fwd_days"] == 60]
        lines.append("| MSW  | n_rescues | rescued_alpha(pp) | pos_rescue_rate | 判定 |")
        lines.append("|------|-----------|-------------------|-----------------|------|")
        for _, row in sub60.iterrows():
            if row["rescued_alpha_mean"] is None or pd.isna(row["rescued_alpha_mean"]):
                verdict = "N/A"
                alpha_s = "N/A"
                rate_s  = "N/A"
            else:
                alpha = row["rescued_alpha_mean"]
                rate  = row["pos_rescue_rate"]
                alpha_s = f"{alpha:+.2f}"
                rate_s  = f"{rate:.1f}%"
                if alpha > 2.0 and rate > 60:
                    verdict = "採用候補 ✅"
                elif alpha < 0:
                    verdict = "廃止 ❌"
                else:
                    verdict = "保留 ⚠"
            lines.append(f"| {row['msw']} | {int(row['n_rescues']) if pd.notna(row['n_rescues']) else 0} | "
                         f"{alpha_s} | {rate_s} | {verdict} |")
    lines.append("")

    # ── Section 4: Positive Rescue Rate by Rank Tier (Task 5, fwd60d)
    lines.append("## 4. Positive Rescue Rate by Rank Tier (fwd60d)")
    if not t5_df.empty:
        sub60 = t5_df[t5_df["fwd_days"] == 60]
        lines.append("| MSW  | tier    | n   | pos_rate | mean_alpha(pp) |")
        lines.append("|------|---------|-----|----------|----------------|")
        for _, row in sub60.iterrows():
            lines.append(f"| {row['msw']} | {row['tier']:<7} | {int(row['n'])} | "
                         f"{row['pos_rescue_rate']:.1f}% | {row['mean_alpha_pct']:+.2f} |")
    lines.append("")

    # ── Section 5: Adoption Verdict
    lines.append("## 5. 採用判定")
    if not t4_curve.empty:
        sub60 = t4_curve[t4_curve["fwd_days"] == 60]
        msw33 = sub60[sub60["msw"] == 0.33]
        if not msw33.empty and pd.notna(msw33.iloc[0]["rescued_alpha_mean"]):
            alpha33 = msw33.iloc[0]["rescued_alpha_mean"]
            rate33  = msw33.iloc[0]["pos_rescue_rate"]
            crit1 = alpha33 > 2.0
            crit2 = rate33 > 60.0
            lines.append(f"MSW=0.33 vs 0.25 (fwd60d):")
            lines.append(f"- rescued_alpha_mean: {alpha33:+.2f}pp  (基準>+2pp: {'✅' if crit1 else '❌'})")
            lines.append(f"- positive_rescue_rate: {rate33:.1f}%  (基準>60%: {'✅' if crit2 else '❌'})")
            if crit1 and crit2:
                verdict = "**採用候補 ✅** — Top3救済効果が優位"
            elif alpha33 < 0:
                verdict = "**廃止 ❌** — 弱い銘柄の混入が優位"
            else:
                verdict = "**保留 ⚠** — 効果不明確 / 追加検証必要"
            lines.append(f"- 判定: {verdict}")
        else:
            lines.append("- 判定: データ不足")
    lines.append("")
    lines.append(f"生成: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────

def main():
    cfg_base = load_strategy_config()
    cfg = dc_replace(cfg_base, portfolio=dc_replace(cfg_base.portfolio, capital=CAPITAL))

    print("データ読み込み中...")
    (universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
     rsr_syms, topix_close, cfg) = load_data(cfg)

    print("バックテスト実行中 (3M, 全期間)...")
    res = run_period(
        universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
        topix_close, rsr_syms, cfg,
        start=START, end=END,
        pattern="A",
        return_trades=True,
        return_cash_skipped=True,
        return_missed_details=True,
        max_new_per_day=2,
        rsr_exit_override=70.0,
        verbose=False,
    )

    trades        = res.get("_trades_raw",    [])
    dates_list    = res.get("_common_dates",  [])
    cash_skipped  = res.get("_cash_skipped",  [])
    missed_details = res.get("_missed_details", [])

    n_buys  = sum(1 for t in trades if t["side"] == "BUY")
    n_sells = sum(1 for t in trades if t["side"] == "SELL")
    print(f"  CAGR={res.get('cagr',0):+.1f}%  MaxDD={res.get('max_dd',0):.1f}%  "
          f"buys={n_buys}  sells={n_sells}  "
          f"cash_skipped={len(cash_skipped)}  capacity_blocked={len(missed_details)}")

    print("\nTask1-2: 候補収集・ギャップ分析...")
    cands_by_date = build_candidates_by_date(trades, cash_skipped, missed_details, dates_list)
    n_signal_days = len(cands_by_date)
    total_top3 = sum(min(len(v), TOP_N) for v in cands_by_date.values())
    print(f"  シグナル日数={n_signal_days}  Top3観測={total_top3}")

    t1_df = task1_intended_top3(cands_by_date)
    t2_df = task2_execution_gap(cands_by_date)
    t1_df.to_csv(OUT_DIR / "intended_top3.csv", index=False)
    t2_df.to_csv(OUT_DIR / "executed_positions.csv", index=False)
    print(f"  → analysis/intended_top3.csv ({len(t1_df)}行)")
    print(f"  → analysis/executed_positions.csv ({len(t2_df)}行)")

    # Gap summary
    if not t2_df.empty:
        miss_n = (~t2_df["executed"]).sum()
        total  = len(t2_df)
        print(f"  未実行率: {miss_n/total*100:.1f}% ({miss_n}/{total})")
        rc = t2_df[~t2_df["executed"]]["reject_reason"].value_counts()
        for r, c in rc.items():
            print(f"    {r}: {c}件")

    print("\nTask3: Missed Alpha 計算中 (20d/60d/120d)...")
    t3_detail, t3_summary = task3_missed_alpha(cands_by_date, dates_list, universe_raw)
    t3_detail.to_csv(OUT_DIR / "missed_top3_alpha.csv", index=False)
    print(f"  → analysis/missed_top3_alpha.csv ({len(t3_detail)}行)")
    if not t3_summary.empty:
        for _, row in t3_summary[t3_summary["fwd_days"] == 60].iterrows():
            al = f"{row['alpha_loss_mean']:+.2f}pp" if pd.notna(row['alpha_loss_mean']) else "N/A"
            print(f"  rank{int(row['intended_rank'])} fwd60d: alpha_loss={al}  "
                  f"n_missed={int(row['n_missed'])}")

    print("\nTask4: MSW Rescue Curve 計算中...")
    t4_events, t4_curve = task4_rescue_curve(cands_by_date, dates_list, universe_raw)
    t4_events.to_csv(OUT_DIR / "msw_rescue_events.csv", index=False)
    t4_curve.to_csv(OUT_DIR / "msw_rescue_curve.csv", index=False)
    print(f"  → analysis/msw_rescue_events.csv ({len(t4_events)}行)")
    print(f"  → analysis/msw_rescue_curve.csv ({len(t4_curve)}行)")
    if not t4_curve.empty:
        sub60 = t4_curve[t4_curve["fwd_days"] == 60]
        print("  MSW  | n_rescues | rescued_alpha(fwd60d) | pos_rate")
        for _, row in sub60.iterrows():
            if pd.notna(row["rescued_alpha_mean"]):
                print(f"  {row['msw']}  | {int(row['n_rescues']):9d} | "
                      f"{row['rescued_alpha_mean']:+8.2f}pp       | {row['pos_rescue_rate']:.1f}%")

    print("\nTask5: Positive Rescue Rate by Rank Tier...")
    t5_df = task5_positive_rescue_rate(t4_events)
    t5_df.to_csv(OUT_DIR / "msw_rescue_rate_by_tier.csv", index=False)
    print(f"  → analysis/msw_rescue_rate_by_tier.csv ({len(t5_df)}行)")
    if not t5_df.empty:
        sub = t5_df[(t5_df["fwd_days"] == 60) & (t5_df["msw"] == 0.33)]
        for _, row in sub.iterrows():
            print(f"  MSW=0.33 {row['tier']}: pos_rate={row['pos_rescue_rate']:.1f}%  "
                  f"mean_alpha={row['mean_alpha_pct']:+.2f}pp  n={int(row['n'])}")

    print("\nレポート生成中...")
    report = build_report(t2_df, t3_summary, t4_curve, t5_df, CAPITAL)
    report_path = OUT_DIR / "msw_alpha_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"  → analysis/msw_alpha_report.md")

    print("\n=== 最終判定 ===")
    if not t4_curve.empty:
        sub60 = t4_curve[t4_curve["fwd_days"] == 60]
        msw33 = sub60[sub60["msw"] == 0.33]
        if not msw33.empty and pd.notna(msw33.iloc[0].get("rescued_alpha_mean")):
            alpha33 = msw33.iloc[0]["rescued_alpha_mean"]
            rate33  = msw33.iloc[0]["pos_rescue_rate"]
            print(f"MSW=0.33 rescued_alpha(fwd60d): {alpha33:+.2f}pp  pos_rate: {rate33:.1f}%")
            if alpha33 > 2.0 and rate33 > 60:
                print("判定: 採用候補 ✅ — Top3救済効果が優位")
            elif alpha33 < 0:
                print("判定: 廃止 ❌ — 弱い銘柄の混入が優位")
            else:
                print("判定: 保留 ⚠ — 効果不明確")
        else:
            print("判定: データ不足 — 計算可能な救済イベントなし")


if __name__ == "__main__":
    main()
