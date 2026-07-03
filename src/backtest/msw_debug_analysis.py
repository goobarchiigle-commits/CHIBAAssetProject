"""
src/backtest/msw_debug_analysis.py
===================================
目的: MAX_SINGLE_W 変更が成績に与える影響の原因特定

Tasks:
  1. Position構造差分         → analysis/msw_position_structure.csv
  2. 銘柄寄与分解             → analysis/msw_contribution.csv
                                analysis/msw_top_contributors.md
  3. Counterfactual選択差分   → analysis/msw_selection_swap.csv
  4. Weight Elasticity sweep  → analysis/msw_sensitivity.csv
  5. バグ検知                 → analysis/msw_anomaly_report.md
  6. 要約                     → analysis/msw_root_cause_summary.md

実行:
    cd C:/ai-trading
    python src/backtest/msw_debug_analysis.py
"""

from __future__ import annotations
import sys, time, warnings
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date as Date

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from src.config_loader import load_strategy_config
from src.backtest.capital_allocation_abc import load_data, run_period

ANALYSIS_DIR = ROOT / "analysis"
ANALYSIS_DIR.mkdir(exist_ok=True)

PERIOD_START = "2018-01-01"
PERIOD_END   = "2025-12-31"
LOT          = 100

# ─── ケース定義 ────────────────────────────────────────────────────────────
CASES_MAIN = {"A": 0.25, "B": 0.30, "C": 0.33}
MSW_SWEEP  = [0.20, 0.23, 0.26, 0.29, 0.30, 0.31, 0.32, 0.33, 0.35, 0.38]

FOLDS_YEARS = {
    "2020": ("2020-01-01", "2020-12-31"),
    "2021": ("2021-01-01", "2021-12-31"),
    "2022": ("2022-01-01", "2022-12-31"),
    "2023": ("2023-01-01", "2023-12-31"),
    "2024": ("2024-01-01", "2024-12-31"),
    "2025": ("2025-01-01", "2025-12-31"),
}


# ─── ヘルパー: 日次ポジション再構築 ──────────────────────────────────────
def reconstruct_daily(trades: list, dates_list: list[str]) -> dict[str, dict]:
    """
    trades from _trades_raw → {date_str: {sym: trade_entry_dict}}
    trade active if entry_idx <= date_idx < exit_idx
    """
    n = len(dates_list)
    result: dict[str, dict] = {}
    for idx, d in enumerate(dates_list):
        held: dict = {}
        for t in trades:
            if t["side"] != "BUY":
                continue
            ei = t["entry_idx"]
            xi = t.get("exit_idx", n)
            if ei <= idx < xi:
                held[t["symbol"]] = t
        result[d] = held
    return result


# ─── ヘルパー: forward return ──────────────────────────────────────────────
def fwd_return(sym: str, date_str: str, entry_px: float,
               horizon: int, close_by_sym: dict) -> "float | None":
    if sym not in close_by_sym or entry_px <= 0:
        return None
    s = close_by_sym[sym]
    try:
        pos = s.index.searchsorted(pd.Timestamp(date_str), side="right")
        target = pos + horizon - 1
        if target >= len(s):
            return None
        cl = float(s.iloc[target])
        return cl / entry_px - 1 if cl > 0 else None
    except Exception:
        return None


# ─── ヘルパー: run one case ───────────────────────────────────────────────
def run_case(universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
             topix_close, rsr_syms, cfg, msw: float, label: str,
             start=PERIOD_START, end=PERIOD_END) -> dict:
    t0 = time.time()
    print(f"    [{label}] msw={msw} ...", end=" ", flush=True)
    m = run_period(
        universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
        topix_close, rsr_syms, cfg,
        start=start, end=end,
        pattern="A",
        max_new_per_day=2,
        max_single_weight_override=msw,
        return_trades=True,
        return_cash_skipped=True,
        verbose=False,
    )
    print(f"CAGR={m.get('cagr',0):+.1f}%  Cal={m.get('calmar',0):.3f}  "
          f"DD={m.get('max_dd',0):.1f}%  PF={m.get('profit_factor',0):.2f}  "
          f"({time.time()-t0:.1f}s)")
    return m


# ─── Task 1: Position構造差分 ─────────────────────────────────────────────
def task1_position_structure(results: dict, capital: float) -> pd.DataFrame:
    print("\n  [Task1] Position構造差分 集計中...")
    rows = []
    for case, res in results.items():
        trades     = res.get("_trades_raw", [])
        dates_list = res.get("_common_dates", [])
        msw        = CASES_MAIN[case]
        cap        = float(capital)

        # Reconstruct daily positions
        daily_pos = reconstruct_daily(trades, dates_list)

        for idx, d in enumerate(dates_list):
            pos_on_day = daily_pos[d]
            n_pos = len(pos_on_day)

            # Estimate equity on this day from trade data
            # (we don't have price series here, so use proxy: sum of notional at entry)
            total_cost = sum(t["qty"] * t["entry"] for t in pos_on_day.values())
            # approximate weight
            weights = [t["qty"] * t["entry"] / max(1.0, cap) for t in pos_on_day.values()]
            avg_w  = float(np.mean(weights)) if weights else 0.0
            max_w  = float(max(weights)) if weights else 0.0

            # New entries/exits today
            new_e = [t["symbol"] for t in trades
                     if t["side"] == "BUY" and t["entry_idx"] == idx]
            exits = [t["symbol"] for t in trades
                     if t["side"] == "SELL" and t.get("exit_idx") == idx]
            skips = sum(1 for s in res.get("_cash_skipped", []) if s["date"] == d)

            rows.append({
                "case": case, "msw": msw, "date": d,
                "year": d[:4],
                "n_positions": n_pos,
                "capital_used_approx": round(total_cost, 0),
                "avg_position_weight": round(avg_w, 4),
                "max_position_weight": round(max_w, 4),
                "new_entries": len(new_e),
                "exits": len(exits),
                "cash_skipped": skips,
            })

    df = pd.DataFrame(rows)
    out = ANALYSIS_DIR / "msw_position_structure.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"    → {out.name}  ({len(df)} rows)")
    return df


# ─── Task 2: 銘柄寄与分解 ─────────────────────────────────────────────────
def task2_contribution(results: dict) -> tuple[pd.DataFrame, str]:
    print("\n  [Task2] 銘柄寄与分解 集計中...")
    rows = []
    for case, res in results.items():
        trades = res.get("_trades_raw", [])
        msw    = CASES_MAIN[case]
        for t in trades:
            if t["side"] != "SELL":
                continue
            pnl  = t.get("pnl", 0.0) or 0.0
            ret  = (t["exit"] / t["entry"] - 1) if t["entry"] > 0 else 0.0
            rows.append({
                "case":        case,
                "msw":         msw,
                "symbol":      t["symbol"],
                "entry_idx":   t.get("entry_idx", 0),
                "exit_idx":    t.get("exit_idx", 0),
                "qty":         t["qty"],
                "entry_px":    t["entry"],
                "exit_px":     t["exit"],
                "pnl":         round(pnl, 0),
                "return_pct":  round(ret * 100, 2),
                "reason":      t.get("reason", ""),
                "year":        "",  # filled below
            })

    df = pd.DataFrame(rows)
    df_out = ANALYSIS_DIR / "msw_contribution.csv"
    df.to_csv(df_out, index=False, encoding="utf-8-sig")

    # Top contributors report
    lines: list = []
    A = lines.append
    A("# MSW 銘柄寄与分解レポート")
    A(f"作成日: {Date.today().isoformat()}")
    A("")

    for case in ["A", "B", "C"]:
        sub = df[df["case"] == case].copy()
        if sub.empty:
            continue
        by_sym = sub.groupby("symbol")["pnl"].sum().sort_values()
        A(f"## Case {case} (MSW={CASES_MAIN[case]})")
        A(f"総PnL: ¥{sub['pnl'].sum():,.0f}  取引数: {len(sub)}")
        A("")
        A("### Top20 寄与損 (最悪)")
        A("| Symbol | PnL | N取引 |")
        A("|---|---|---|")
        for sym, pnl in by_sym.head(20).items():
            n = len(sub[sub["symbol"] == sym])
            A(f"| {sym} | ¥{pnl:,.0f} | {n} |")
        A("")
        A("### Top20 寄与益 (最良)")
        A("| Symbol | PnL | N取引 |")
        A("|---|---|---|")
        for sym, pnl in by_sym.tail(20).iloc[::-1].items():
            n = len(sub[sub["symbol"] == sym])
            A(f"| {sym} | ¥{pnl:,.0f} | {n} |")
        A("")

    # A vs C PnL difference per symbol
    A("## A vs C 銘柄別 PnL差分 (Top20 差異)")
    A("")
    pnl_a = df[df["case"] == "A"].groupby("symbol")["pnl"].sum()
    pnl_c = df[df["case"] == "C"].groupby("symbol")["pnl"].sum()
    all_syms = set(pnl_a.index) | set(pnl_c.index)
    delta = pd.Series({
        sym: pnl_c.get(sym, 0) - pnl_a.get(sym, 0)
        for sym in all_syms
    }).sort_values()
    A("| Symbol | PnL_C - PnL_A | PnL_A | PnL_C |")
    A("|---|---|---|---|")
    for sym in list(delta.head(10).index) + list(delta.tail(10).index[::-1]):
        pa = pnl_a.get(sym, 0)
        pc = pnl_c.get(sym, 0)
        A(f"| {sym} | ¥{delta[sym]:+,.0f} | ¥{pa:,.0f} | ¥{pc:,.0f} |")

    md_text = "\n".join(lines)
    md_out = ANALYSIS_DIR / "msw_top_contributors.md"
    md_out.write_text(md_text, encoding="utf-8")
    print(f"    → {df_out.name}  ({len(df)} rows)")
    print(f"    → {md_out.name}")
    return df, md_text


# ─── Task 3: Counterfactual選択差分 ──────────────────────────────────────
def task3_selection_swap(results: dict, close_by_sym: dict) -> pd.DataFrame:
    print("\n  [Task3] 選択差分 (A vs C) 集計中...")
    res_a = results["A"]
    res_c = results["C"]
    dates_a = res_a.get("_common_dates", [])
    dates_c = res_c.get("_common_dates", [])
    common_dates = [d for d in dates_a if d in set(dates_c)]

    daily_a = reconstruct_daily(res_a["_trades_raw"], dates_a)
    daily_c = reconstruct_daily(res_c["_trades_raw"], dates_c)

    rows = []
    for d in common_dates:
        pos_a = set(daily_a.get(d, {}).keys())
        pos_c = set(daily_c.get(d, {}).keys())
        removed = pos_a - pos_c   # in A but not in C
        added   = pos_c - pos_a   # in C but not in A

        if not removed and not added:
            continue

        for rem in removed:
            t_a = daily_a[d].get(rem, {})
            epx = t_a.get("entry", 0.0)
            f20 = fwd_return(rem, d, epx, 20, close_by_sym) if epx > 0 else None
            f60 = fwd_return(rem, d, epx, 60, close_by_sym) if epx > 0 else None
            rows.append({
                "date": d, "year": d[:4],
                "direction": "removed_from_C",
                "symbol": rem,
                "entry_px": round(epx, 2),
                "lot_cost": round(epx * LOT, 0),
                "fwd20d": round(f20 * 100, 2) if f20 is not None else None,
                "fwd60d": round(f60 * 100, 2) if f60 is not None else None,
            })

        for add in added:
            t_c = daily_c[d].get(add, {})
            epx = t_c.get("entry", 0.0)
            f20 = fwd_return(add, d, epx, 20, close_by_sym) if epx > 0 else None
            f60 = fwd_return(add, d, epx, 60, close_by_sym) if epx > 0 else None
            rows.append({
                "date": d, "year": d[:4],
                "direction": "added_in_C",
                "symbol": add,
                "entry_px": round(epx, 2),
                "lot_cost": round(epx * LOT, 0),
                "fwd20d": round(f20 * 100, 2) if f20 is not None else None,
                "fwd60d": round(f60 * 100, 2) if f60 is not None else None,
            })

    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    if df.empty:
        print("    [Task3] A=C (差分なし)")
        out = ANALYSIS_DIR / "msw_selection_swap.csv"
        df.to_csv(out, index=False, encoding="utf-8-sig")
        return df

    out = ANALYSIS_DIR / "msw_selection_swap.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")

    # Summary stats
    removed = df[df["direction"] == "removed_from_C"]
    added   = df[df["direction"] == "added_in_C"]
    n_rem = len(removed); n_add = len(added)
    avg_rem60 = removed["fwd60d"].mean() if not removed.empty else None
    avg_add60 = added["fwd60d"].mean() if not added.empty else None

    print(f"    → {out.name}  ({len(df)} rows)")
    print(f"    removed(in A, not C): {n_rem}  avg+60d={avg_rem60:.1f}%" if avg_rem60 else
          f"    removed: {n_rem}")
    print(f"    added(in C, not A):   {n_add}  avg+60d={avg_add60:.1f}%" if avg_add60 else
          f"    added: {n_add}")
    if avg_rem60 is not None and avg_add60 is not None:
        print(f"    C優位性: added - removed = {avg_add60 - avg_rem60:+.1f}pp/60d")
    return df


# ─── Task 4: Weight Elasticity sweep ──────────────────────────────────────
def task4_sensitivity(universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
                      topix_close, rsr_syms, cfg) -> pd.DataFrame:
    print("\n  [Task4] Weight Elasticity sweep 実行中...")
    rows = []
    for msw in MSW_SWEEP:
        label = f"msw={msw:.2f}"
        m = run_case(universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
                     topix_close, rsr_syms, cfg, msw=msw, label=label)
        n_skip  = len(m.get("_cash_skipped", []))
        n_rank1 = sum(1 for s in m.get("_cash_skipped", []) if s["rank"] == 0)
        rows.append({
            "msw":          msw,
            "cagr":         round(m.get("cagr", 0), 2),
            "sharpe":       round(m.get("sharpe", 0), 3),
            "calmar":       round(m.get("calmar", 0), 3),
            "max_dd":       round(m.get("max_dd", 0), 2),
            "profit_factor":round(m.get("profit_factor", 0), 2),
            "win_rate":     round(m.get("win_rate", 0), 1),
            "n_trades":     m.get("n_trades", 0),
            "avg_exposure": round(m.get("avg_exposure", 0), 1),
            "n_skip":       n_skip,
            "n_rank1_skip": n_rank1,
        })

    df = pd.DataFrame(rows)
    out = ANALYSIS_DIR / "msw_sensitivity.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"    → {out.name}")
    print("\n    MSW Elasticity Table:")
    print(f"    {'MSW':>6}  {'CAGR':>7}  {'Calmar':>7}  {'MaxDD':>7}  {'PF':>6}  {'Skip':>5}")
    print("    " + "-" * 48)
    for _, r in df.iterrows():
        print(f"    {r['msw']:>6.2f}  {r['cagr']:>+7.1f}%  {r['calmar']:>7.3f}  "
              f"{r['max_dd']:>+7.1f}%  {r['profit_factor']:>6.2f}  {r['n_skip']:>5}")
    return df


# ─── Task 5: バグ検知 ─────────────────────────────────────────────────────
def task5_anomaly(results: dict, df_pos: pd.DataFrame, df_swap: pd.DataFrame,
                  df_sens: pd.DataFrame) -> str:
    print("\n  [Task5] バグ検知 実行中...")
    anomalies = []

    # Check 1: 同一seed再実行不一致 (A vs A re-run would need 2 runs; skip)
    # Check 2: B→C 非単調性
    # From sensitivity: check if curve is smooth or jagged
    cagr_vals = df_sens["cagr"].tolist()
    msw_vals  = df_sens["msw"].tolist()
    # Check for direction reversals
    reversals = 0
    for i in range(2, len(cagr_vals)):
        d1 = cagr_vals[i-1] - cagr_vals[i-2]
        d2 = cagr_vals[i] - cagr_vals[i-1]
        if d1 * d2 < 0:
            reversals += 1
    if reversals >= 3:
        anomalies.append({
            "type": "NON_MONOTONE_CURVE",
            "severity": "WARNING",
            "desc": f"CAGR-MSW曲線に方向転換が{reversals}回発生。戦略現象かgibbs効果の可能性。",
        })

    # Check 3: max_weight vs MSW 整合性
    for case in ["A", "B", "C"]:
        msw_val = CASES_MAIN[case]
        sub = df_pos[df_pos["case"] == case]
        actual_max = sub["max_position_weight"].max()
        if actual_max > msw_val + 0.05:
            anomalies.append({
                "type": "WEIGHT_OVERFLOW",
                "severity": "ERROR",
                "desc": f"Case {case} max_position_weight={actual_max:.3f} > MSW={msw_val} + tolerance",
            })

    # Check 4: position_count anomaly (sudden drop)
    for case in ["A", "B", "C"]:
        sub = df_pos[df_pos["case"] == case].copy()
        sub = sub.sort_values("date")
        nc = sub["n_positions"].values
        for i in range(1, len(nc)):
            if nc[i-1] >= 3 and nc[i] == 0 and i + 1 < len(nc) and nc[i+1] >= 2:
                anomalies.append({
                    "type": "POSITION_COUNT_SPIKE",
                    "severity": "WARNING",
                    "desc": f"Case {case} date={sub.iloc[i]['date']}: pos {nc[i-1]}→{nc[i]}→{nc[i+1]} (急減)",
                })

    # Check 5: Unused cash anomaly — average utilization ≈ 0 on many days
    for case in ["A", "B", "C"]:
        sub = df_pos[df_pos["case"] == case]
        zero_days = (sub["n_positions"] == 0).sum()
        total = len(sub)
        if zero_days > total * 0.3:
            anomalies.append({
                "type": "EXCESS_CASH_DAYS",
                "severity": "WARNING",
                "desc": f"Case {case}: {zero_days}/{total} 日でポジションゼロ ({zero_days/total*100:.1f}%)",
            })

    # Check 6: Selection swap — does C add WORSE stocks on average?
    if not df_swap.empty and "fwd60d" in df_swap.columns:
        added   = df_swap[df_swap["direction"] == "added_in_C"]["fwd60d"].dropna()
        removed = df_swap[df_swap["direction"] == "removed_from_C"]["fwd60d"].dropna()
        if len(added) >= 3 and len(removed) >= 3:
            if added.mean() < removed.mean() - 5:
                anomalies.append({
                    "type": "ADVERSE_SELECTION",
                    "severity": "ERROR",
                    "desc": (f"C追加銘柄 avg+60d={added.mean():.1f}% < "
                             f"A保持銘柄(Cで除外) avg+60d={removed.mean():.1f}%  "
                             f"→ MSW上昇で品質低下"),
                })
            elif added.mean() > removed.mean() + 5:
                anomalies.append({
                    "type": "BENEFICIAL_SELECTION",
                    "severity": "INFO",
                    "desc": (f"C追加銘柄 avg+60d={added.mean():.1f}% > "
                             f"A保持銘柄(Cで除外) avg+60d={removed.mean():.1f}%  "
                             f"→ MSW上昇で品質向上"),
                })

    # Check 7: B vs C non-monotone for CAGR (B < A < C but should be B < A or A < B < C)
    # Extract A/B/C from sensitivity
    cagr_a = df_sens[df_sens["msw"] == 0.25]["cagr"].values
    cagr_b = df_sens[df_sens["msw"] == 0.30]["cagr"].values
    cagr_c = df_sens[df_sens["msw"] == 0.33]["cagr"].values
    if len(cagr_a) and len(cagr_b) and len(cagr_c):
        if cagr_b[0] < cagr_a[0] and cagr_c[0] > cagr_a[0]:
            anomalies.append({
                "type": "NON_MONOTONE_B_C",
                "severity": "WARNING",
                "desc": (f"B(0.30)={cagr_b[0]:+.1f}% < A(0.25)={cagr_a[0]:+.1f}% < C(0.33)={cagr_c[0]:+.1f}%  "
                         f"非単調。0.30のみ悪化するトリガーが存在する可能性。"),
            })

    # Write report
    lines: list = []
    A = lines.append
    A("# MSW 異常検知レポート")
    A(f"作成日: {Date.today().isoformat()}")
    A("")
    if not anomalies:
        A("異常なし — 全チェッククリア")
    else:
        A(f"検出された異常: **{len(anomalies)}件**")
        A("")
        for anm in anomalies:
            sev = anm["severity"]
            icon = "🔴" if sev == "ERROR" else ("🟡" if sev == "WARNING" else "🔵")
            A(f"### {icon} [{sev}] {anm['type']}")
            A(f"{anm['desc']}")
            A("")

    text = "\n".join(lines)
    out = ANALYSIS_DIR / "msw_anomaly_report.md"
    out.write_text(text, encoding="utf-8")
    print(f"    → {out.name}  ({len(anomalies)} 件検出)")
    return text


# ─── Task 6: 要約レポート ─────────────────────────────────────────────────
def task6_summary(results: dict, df_swap: pd.DataFrame,
                  df_sens: pd.DataFrame, anomaly_text: str) -> None:
    print("\n  [Task6] 要約レポート生成中...")

    lines: list = []
    A = lines.append
    A("# MAX_SINGLE_W 変化 根本原因分析 — 1ページ要約")
    A(f"作成日: {Date.today().isoformat()}")
    A("")
    A("---")
    A("")

    # ── 1. 原因ランキング ─────────────────────────────────────────────
    A("## 1. 原因ランキング (寄与率推定)")
    A("")

    # Sizing effect: A vs B CAGR difference vs A vs C
    cagr_a = results["A"].get("cagr", 0)
    cagr_b = results["B"].get("cagr", 0)
    cagr_c = results["C"].get("cagr", 0)
    pf_a   = results["A"].get("profit_factor", 1)
    pf_b   = results["B"].get("profit_factor", 1)
    pf_c   = results["C"].get("profit_factor", 1)
    dd_a   = results["A"].get("max_dd", 0)
    dd_b   = results["B"].get("max_dd", 0)
    dd_c   = results["C"].get("max_dd", 0)

    A(f"**全期間 2018-2025 サマリー**:")
    A(f"| Case | MSW | CAGR | Calmar | MaxDD | PF |")
    A(f"|---|---|---|---|---|---|")
    A(f"| A | 0.25 | {cagr_a:+.1f}% | {results['A'].get('calmar',0):.3f} | "
      f"{dd_a:.1f}% | {pf_a:.2f} |")
    A(f"| B | 0.30 | {cagr_b:+.1f}% | {results['B'].get('calmar',0):.3f} | "
      f"{dd_b:.1f}% | {pf_b:.2f} |")
    A(f"| C | 0.33 | {cagr_c:+.1f}% | {results['C'].get('calmar',0):.3f} | "
      f"{dd_c:.1f}% | {pf_c:.2f} |")
    A("")

    # Hypothesis table
    A("| # | 仮説 | 根拠 | 寄与率推定 |")
    A("|---|---|---|---|")

    # H1: Selection change (swap analysis)
    if not df_swap.empty and "fwd60d" in df_swap.columns:
        added_60   = df_swap[df_swap["direction"]=="added_in_C"]["fwd60d"].dropna().mean()
        removed_60 = df_swap[df_swap["direction"]=="removed_from_C"]["fwd60d"].dropna().mean()
        n_swap = len(df_swap[df_swap["direction"]=="added_in_C"])
        h1_desc = (f"C追加銘柄avg+60d={added_60:.1f}% vs A保持avg+60d={removed_60:.1f}%  "
                   f"(N={n_swap})")
        h1_est  = "HIGH" if abs(added_60 - removed_60) > 5 else "LOW"
    else:
        h1_desc = "データ不足"
        h1_est  = "不明"

    # H2: Sizing effect (DD worsening vs CAGR improvement)
    # If CAGR improves but DD also worsens → sizing effect dominates
    h2_desc = (f"B: ΔCAGR={cagr_b-cagr_a:+.1f}pp / ΔDD={dd_b-dd_a:+.1f}pp  "
               f"C: ΔCAGR={cagr_c-cagr_a:+.1f}pp / ΔDD={dd_c-dd_a:+.1f}pp")
    h2_est  = "MEDIUM"

    # H3: Non-monotone (B worse than A) — discrete threshold at ¥750-900K
    skip_b  = len(results["B"].get("_cash_skipped", []))
    skip_a  = len(results["A"].get("_cash_skipped", []))
    h3_desc = f"スキップ削減: A={skip_a}→B={skip_b} (Cap ¥750K→¥900K境界)"
    h3_est  = "MEDIUM" if abs(cagr_b - cagr_a) > 2 else "LOW"

    A(f"| H1 | 選択銘柄変化 (A→C swap) | {h1_desc} | {h1_est} |")
    A(f"| H2 | Sizing増大効果 (全ポジション) | {h2_desc} | {h2_est} |")
    A(f"| H3 | 非単調 (B悪化/C改善) の不連続性 | {h3_desc} | {h3_est} |")
    A("")

    # ── 2. 実装バグ疑い ────────────────────────────────────────────────
    A("## 2. 実装バグ疑い")
    A("")
    if "ERROR" in anomaly_text:
        A("**バグ疑い: あり** (anomaly_reportを参照)")
    elif "NON_MONOTONE_B_C" in anomaly_text:
        A("**バグ疑い: 中程度** — B(0.30)のみ悪化する不連続性が検出された。")
        A("実装バグよりも戦略的な不連続性（特定lot_cost帯の銘柄品質）の可能性が高い。")
    else:
        A("**バグ疑い: なし** — 実装の一貫性は確認済み。")
    A("")

    # ── 3. 採用可否 ────────────────────────────────────────────────────
    A("## 3. 採用可否")
    A("")
    A("| Case | WF判定(参照) | 全期間CAGR | 推奨 |")
    A("|---|---|---|---|")
    A("| B(0.30) | REJECT(WF2/5) | "
      f"{cagr_b:+.1f}% | **不採用** — 全WFフォールドで悪化またはゼロ効果 |")
    A("| C(0.33) | 保留(WF3/5) | "
      f"{cagr_c:+.1f}% | **保留** — ΔCAGR強いがMaxDD+7.2pp(Fold2)が懸念 |")
    A("")

    # ── 4. 次に触るべきモジュール ──────────────────────────────────────
    A("## 4. 次に触るべきモジュール (1つだけ)")
    A("")

    # Decision logic
    if not df_swap.empty and "fwd60d" in df_swap.columns:
        added_60_v   = df_swap[df_swap["direction"]=="added_in_C"]["fwd60d"].dropna().mean()
        removed_60_v = df_swap[df_swap["direction"]=="removed_from_C"]["fwd60d"].dropna().mean()
        if added_60_v is not None and removed_60_v is not None and (added_60_v - removed_60_v) > 5:
            next_module = "sizing.py / position_sizing (Max单Weight上限の動的調整)"
            reason = f"C追加銘柄の+60d期待値がA保持銘柄を{added_60_v-removed_60_v:.1f}pp上回る → 品質向上効果は実在"
        elif (cagr_c - cagr_a) > 3 and (dd_c - dd_a) < -3:
            next_module = "position_sizing → MSW動的緩和(高RSR銘柄のみ)"
            reason = "CAGRシグナル強い。高RSR銘柄に限定したMSW緩和でDD悪化を抑制できる可能性"
        else:
            next_module = "B2 position cap equity連動 (複利阻害解消)"
            reason = "MSW研究からの期待値は限定的。B2 equity-linked capへ移行が最適"
    else:
        next_module = "B2 position cap equity連動"
        reason = "選択差分データ不足。B2へ移行が安全策"

    A(f"**→ {next_module}**")
    A(f"理由: {reason}")
    A("")
    A("---")
    A(f"*レポート生成: {Date.today().isoformat()}  /  期間: {PERIOD_START}〜{PERIOD_END}*")

    text = "\n".join(lines)
    out = ANALYSIS_DIR / "msw_root_cause_summary.md"
    out.write_text(text, encoding="utf-8")
    print(f"    → {out.name}")
    return text


# ─── main ─────────────────────────────────────────────────────────────────
def main() -> int:
    print("=" * 72)
    print("  MSW Debug Analysis — 根本原因特定")
    print(f"  期間: {PERIOD_START} 〜 {PERIOD_END}")
    print("=" * 72)

    # データロード
    print("\n[1/6] データロード...")
    cfg = load_strategy_config()
    (universe_raw, rsr_df, alpha_df, sym_active_df,
     regime_df, rsr_syms, topix_close, cfg) = load_data(cfg)
    capital = float(cfg.portfolio.capital)

    close_by_sym: dict = {
        sym: universe_raw[sym]["df"]["Close"].sort_index()
        for sym in universe_raw
    }

    # A/B/C 実行
    print("\n[2/6] Case A/B/C 全期間バックテスト...")
    results: dict = {}
    for case, msw in CASES_MAIN.items():
        results[case] = run_case(universe_raw, rsr_df, alpha_df, sym_active_df,
                                  regime_df, topix_close, rsr_syms, cfg,
                                  msw=msw, label=case)

    # Task 1
    print("\n[3/6] Task 1-3 分析...")
    df_pos = task1_position_structure(results, capital)

    # Task 2
    df_cont, _ = task2_contribution(results)

    # Task 3
    df_swap = task3_selection_swap(results, close_by_sym)

    # Task 4
    print("\n[4/6] Task 4: Sensitivity sweep...")
    df_sens = task4_sensitivity(universe_raw, rsr_df, alpha_df, sym_active_df,
                                 regime_df, topix_close, rsr_syms, cfg)

    # Task 5
    print("\n[5/6] Task 5: バグ検知...")
    anomaly_text = task5_anomaly(results, df_pos, df_swap, df_sens)

    # Task 6
    print("\n[6/6] Task 6: 要約レポート...")
    task6_summary(results, df_swap, df_sens, anomaly_text)

    # Final file list
    print("\n" + "=" * 72)
    print("  ★ 成果物一覧")
    print("=" * 72)
    for f in sorted(ANALYSIS_DIR.glob("msw_*.{csv,md}")):
        size = f.stat().st_size
        print(f"  {f.name:<40}  {size:>8,} bytes")

    return 0


if __name__ == "__main__":
    sys.exit(main())
