"""
src/backtest/study15b_access_elasticity.py  —  Study15B Access Elasticity Confirmation

目的: LOT制約+高単価露出による alpha 欠損が
      access 問題（≠戦略問題）であることを確定し、
      efficient_capital と plateau_region を決定する。

Cases:
  A ¥1.0M  B ¥1.2M  C ¥1.35M  D ¥1.5M  E ¥1.8M  F ¥2.2M

Output:
  reports/study15b_access_elasticity.md

Run:
    cd C:/ai-trading
    python src/backtest/study15b_access_elasticity.py
"""
from __future__ import annotations

import sys, math, time, warnings
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.config_loader import load_strategy_config
from src.backtest.capital_allocation_abc import load_data, _take, calc_metrics, LOT, COST_ONE_WAY

REPORTS_DIR = Path("reports")

# ── Fixed strategy params (Study9 Case B) ────────────────────────────
RSR_LO    = 92.0
RSR_HI    = 95.0
D90_MAX   = 5
SLOPE_MAX = 5.0
EXIT_THR  = 90.0
MIN_HOLD  = 3
SHOCK_MKT = -0.05
SHOCK_SYM = -0.08
IS_START  = "2018-01-01"

# ── Study15B capital cases ────────────────────────────────────────────
CASES = [
    {"id": "A", "capital": 1_000_000},
    {"id": "B", "capital": 1_200_000},
    {"id": "C", "capital": 1_350_000},
    {"id": "D", "capital": 1_500_000},
    {"id": "E", "capital": 1_800_000},
    {"id": "F", "capital": 2_200_000},
]

FOLDS = [
    {"id": 1, "oos_start": "2021-01-01", "oos_end": "2021-12-31"},
    {"id": 2, "oos_start": "2022-01-01", "oos_end": "2022-12-31"},
    {"id": 3, "oos_start": "2023-01-01", "oos_end": "2023-12-31"},
    {"id": 4, "oos_start": "2024-01-01", "oos_end": "2024-12-31"},
    {"id": 5, "oos_start": "2025-01-01", "oos_end": "2025-12-31"},
]

# ── Adoption criteria ──────────────────────────────────────────────────
FILL_TARGET          = 0.95
SKIP_TARGET          = 0.05
ACCESS_RECOVERY_MIN  = 0.90
CALMAR_RATIO_MIN     = 0.95   # vs virtual_frac reference

# ── Stopping condition ─────────────────────────────────────────────────
STOP_MARGINAL_ACCESS = 0.02   # Δfillable(frac) / Δcapital(¥M)
STOP_MARGINAL_ALPHA  = 0.03   # ΔCalmar / Δfillable(pp)
STOP_CONSECUTIVE     = 2


# ──────────────────────────────────────────────────────────────────────
#  MATRIX HELPERS
# ──────────────────────────────────────────────────────────────────────

def compute_cross90_mat(rsr_mat: np.ndarray) -> np.ndarray:
    n, m = rsr_mat.shape
    out = np.zeros((n, m), dtype=np.int32)
    counts = np.zeros(m, dtype=np.int32)
    for i in range(n):
        out[i] = counts
        above = rsr_mat[i] >= 90.0
        counts[above] += 1
        counts[~above] = 0
    return out

def compute_slope5(rsr_mat: np.ndarray) -> np.ndarray:
    out = np.zeros_like(rsr_mat, dtype=np.float32)
    out[5:] = rsr_mat[5:] - rsr_mat[:-5]
    return out


# ──────────────────────────────────────────────────────────────────────
#  SIMULATION
# ──────────────────────────────────────────────────────────────────────

@dataclass
class SlPos:
    symbol: str
    qty: int
    entry_price: float
    entry_idx: int
    rsr_entry: float
    d90_entry: int
    slope5_entry: float
    days_below: int = 0


def run_sim(
    open_mat, close_mat, rsr_mat, sym_active_mat,
    cross90_mat, slope5_mat, mkt_ret1,
    active_syms, sym_to_i, common_dates, n_dates: int,
    sl_cash_init: float,
    lot_size: int = 100,
    price_cap: Optional[float] = None,
) -> dict:
    sl_cash = sl_cash_init
    sl_pos: Optional[SlPos] = None
    trades: list[dict] = []
    skip_log: list[dict] = []

    eq  = np.zeros(n_dates, dtype=np.float64)
    inv = np.zeros(n_dates, dtype=np.float64)

    for i, date in enumerate(common_dates):
        ds  = str(date.date())
        s_inv = (sl_pos.qty * float(close_mat[i, sym_to_i[sl_pos.symbol]])
                 if sl_pos else 0.0)
        eq[i]  = sl_cash + s_inv
        inv[i] = s_inv

        mkt_shock = mkt_ret1 is not None and float(mkt_ret1[i]) <= SHOCK_MKT
        if i + 1 >= n_dates:
            break
        nxt = i + 1

        # EXIT
        if sl_pos:
            si  = sym_to_i[sl_pos.symbol]
            rv  = float(rsr_mat[i, si])
            ct  = float(close_mat[i, si])
            ep  = sl_pos.entry_price
            hd  = i - sl_pos.entry_idx
            do_exit = False; reason = ""

            if mkt_shock and i > 0:
                pc = float(close_mat[i-1, si])
                if pc > 0 and (ct / pc - 1) <= SHOCK_SYM:
                    do_exit = True; reason = "MKT_SHOCK"

            if not do_exit:
                if rv < EXIT_THR:
                    sl_pos.days_below += 1
                else:
                    sl_pos.days_below = 0
                if sl_pos.days_below >= 1 and hd >= MIN_HOLD:
                    do_exit = True; reason = "RSR_EXIT"

            if do_exit:
                sp  = float(open_mat[nxt, si])
                pnl = (sp - ep) * sl_pos.qty
                sl_cash += sl_pos.qty * sp * (1 - COST_ONE_WAY)
                trades.append({
                    "side": "SELL", "symbol": sl_pos.symbol,
                    "pnl": pnl, "entry": ep, "exit": sp,
                    "qty": sl_pos.qty, "hold_days": i - sl_pos.entry_idx,
                    "realized_ret": (sp / ep - 1) if ep > 0 else 0.0,
                    "reason": reason, "exit_date": ds,
                })
                sl_pos = None

        # ENTRY
        if sl_pos is None and not mkt_shock:
            cands: list[tuple] = []
            for sym in active_syms:
                si  = sym_to_i[sym]
                rv  = float(rsr_mat[i, si])
                d90 = int(cross90_mat[i, si])
                sl5 = float(slope5_mat[i, si])
                if rv < RSR_LO or rv >= RSR_HI: continue
                if d90 > D90_MAX or d90 < 1:    continue
                if sl5 > SLOPE_MAX:              continue
                if sym_active_mat is not None and float(sym_active_mat[i, si]) < 0.5: continue
                cands.append((rv, d90, sl5, sym))

            if cands:
                cands.sort(key=lambda x: (-x[0], x[1]))
                rv, d90, sl5, sym = cands[0]
                si  = sym_to_i[sym]
                bp  = float(open_mat[nxt, si])
                if bp <= 0: continue

                if price_cap is not None and bp * LOT > price_cap:
                    continue

                alloc = sl_cash * 0.95
                qty   = int(alloc / bp / lot_size) * lot_size
                cost  = qty * bp * (1 + COST_ONE_WAY)

                if qty > 0 and cost <= sl_cash:
                    sl_cash -= cost
                    sl_pos   = SlPos(
                        symbol=sym, qty=qty, entry_price=bp,
                        entry_idx=nxt, rsr_entry=rv,
                        d90_entry=d90, slope5_entry=sl5,
                    )
                    trades.append({
                        "side": "BUY", "symbol": sym,
                        "entry": bp, "qty": qty,
                        "alloc_used": cost,
                        "sl_cash_before": sl_cash + cost,
                        "min_lot_cost": LOT * bp,
                        "buy_date": ds, "rsr_entry": rv,
                    })
                else:
                    qty_virt   = int(alloc / bp)
                    price_bind = (bp > alloc)  # can't even buy 1 share
                    skip_log.append({
                        "date": ds, "symbol": sym, "price": bp,
                        "alloc": alloc, "sl_cash": sl_cash,
                        "min_lot_cost": LOT * bp,
                        "qty_virt": qty_virt,
                        "lot_binding":   not price_bind,
                        "price_binding": price_bind,
                    })

    return {"eq": eq, "inv": inv, "trades": trades, "skip_log": skip_log}


# ──────────────────────────────────────────────────────────────────────
#  METRICS
# ──────────────────────────────────────────────────────────────────────

def compute_fold_metrics(eq, inv, trades, common_dates, folds, sl_cash_init):
    date_strs = [str(d.date()) for d in common_dates]
    sells = [t for t in trades if t["side"] == "SELL"]
    fold_ms = []
    for fold in folds:
        ds_s, ds_e = fold["oos_start"], fold["oos_end"]
        idx = [i for i, s in enumerate(date_strs) if ds_s <= s <= ds_e]
        if not idx: continue
        si, ei = idx[0], idx[-1] + 1
        s_eq  = eq[si:ei]
        s_inv = inv[si:ei]
        if len(s_eq) < 5: continue
        s_tr = [t for t in sells if ds_s <= t.get("exit_date","") <= ds_e]
        exp  = [vi/max(1.0,ve) for vi,ve in zip(s_inv, s_eq)]
        m = calc_metrics(s_eq.tolist(), s_tr, exp, float(s_eq[0]),
                         list(common_dates[si:ei]))
        fold_ms.append(m)
    return fold_ms


def aggregate(fold_ms):
    if not fold_ms:
        return {"cagr": 0.0, "calmar": 0.0, "max_dd": 0.0, "sharpe": 0.0}
    return {
        "cagr":    float(np.mean([m.get("cagr",0)    for m in fold_ms])),
        "calmar":  float(np.mean([m.get("calmar",0)  for m in fold_ms])),
        "max_dd":  float(np.mean([m.get("max_dd",0)  for m in fold_ms])),
        "sharpe":  float(np.mean([m.get("sharpe",0)  for m in fold_ms])),
        "n_wf_folds": len(fold_ms),
    }


def bootstrap_dd_yen(eq, common_dates, folds, sl_cash_init, n_boot=3000, seed=42):
    rng = np.random.default_rng(seed)
    date_strs = [str(d.date()) for d in common_dates]
    fc, fd = [], []
    for fold in folds:
        ds_s, ds_e = fold["oos_start"], fold["oos_end"]
        idx = [i for i, s in enumerate(date_strs) if ds_s <= s <= ds_e]
        if not idx: continue
        si, ei = idx[0], idx[-1]+1
        s_eq = eq[si:ei]
        if len(s_eq) < 5: continue
        y    = 252/max(1,ei-si)
        cagr = (s_eq[-1]/max(1e-9, s_eq[0]))**y - 1
        pk   = np.maximum.accumulate(s_eq)
        dd   = float(np.nanmin((s_eq-pk)/pk))
        fc.append(cagr); fd.append(dd)
    if len(fc) < 2:
        return 0.0, 0.0
    fc, fd = np.array(fc), np.array(fd)
    boot = []
    for _ in range(n_boot):
        ix = rng.integers(0, len(fc), size=5)
        eq_b = sl_cash_init; mn = sl_cash_init
        for ci, di in zip(fc[ix], fd[ix]):
            mn = min(mn, eq_b*(1+di))
            eq_b = max(1e-3, eq_b*(1+ci))
        boot.append((mn-sl_cash_init)/sl_cash_init)
    a = np.array(boot)
    return abs(float(np.percentile(a,5)))*sl_cash_init, abs(float(np.percentile(a,1)))*sl_cash_init


def compute_all_metrics(result, sl_cash_init, common_dates, folds, virt_frac_calmar):
    eq, inv, trades, skips = (result["eq"], result["inv"],
                               result["trades"], result["skip_log"])
    buys  = [t for t in trades if t["side"] == "BUY"]
    sells = [t for t in trades if t["side"] == "SELL"]

    n_buy  = len(buys)
    n_skip = len(skips)
    n_att  = n_buy + n_skip
    fillable   = n_buy / max(1, n_att)
    skip_rate  = n_skip / max(1, n_att)

    fold_ms = compute_fold_metrics(eq, inv, trades, common_dates, folds, sl_cash_init)
    agg = aggregate(fold_ms)

    p95_yen, p99_yen = bootstrap_dd_yen(eq, common_dates, folds, sl_cash_init)

    # capital utilization
    cap_util   = float((inv > 0).mean())
    idle_arr   = [(eq[i]-inv[i])/max(1.0, eq[i]) for i in range(len(eq)) if eq[i]>0]
    cash_idle  = float(np.median(idle_arr)) if idle_arr else 1.0

    # fill_loss (lot rounding waste)
    fill_losses = []
    for t in buys:
        au = t.get("alloc_used", 0)
        sb = t.get("sl_cash_before", 0)
        am = sb * 0.95
        if am > 0: fill_losses.append(1 - au/am)
    fill_loss = float(np.mean(fill_losses)) if fill_losses else 0.0

    # skip analysis
    skip_prices = [s["price"] for s in skips]
    avg_cand_price = float(np.mean([t["entry"] for t in buys] + skip_prices)) if (buys or skips) else 0.0
    avg_skip_price = float(np.mean(skip_prices)) if skip_prices else 0.0
    avg_buy_price  = float(np.mean([t["entry"] for t in buys])) if buys else 0.0

    # lot_binding_rate / price_binding_rate
    n_lot_bind   = sum(1 for s in skips if s.get("lot_binding",   False))
    n_price_bind = sum(1 for s in skips if s.get("price_binding", False))
    lot_binding_rate   = n_lot_bind   / max(1, n_skip)
    price_binding_rate = n_price_bind / max(1, n_skip)

    # access_recovery = realized_calmar / virt_frac_calmar
    access_recovery = agg["calmar"] / max(0.001, virt_frac_calmar)

    # capital_efficiency = Calmar / (capital / 1_000_000)
    capital_efficiency = agg["calmar"] / (sl_cash_init / 1_000_000)

    # top5 skip symbols by count
    skip_sym_cnt = Counter(s["symbol"] for s in skips)
    top5_skip    = skip_sym_cnt.most_common(5)

    return {
        "sl_cash_init":      sl_cash_init,
        "n_buy":             n_buy,
        "n_skip":            n_skip,
        "n_attempt":         n_att,
        "fillable":          round(fillable, 4),
        "forced_skip":       round(skip_rate, 4),
        "avg_cand_price":    round(avg_cand_price, 0),
        "avg_buy_price":     round(avg_buy_price, 0),
        "avg_skip_price":    round(avg_skip_price, 0),
        "cash_idle":         round(cash_idle, 4),
        "fill_loss":         round(fill_loss, 4),
        "lot_binding_rate":  round(lot_binding_rate, 4),
        "price_binding_rate":round(price_binding_rate, 4),
        "trade_count":       len(sells),
        "avg_cagr":          round(agg["cagr"], 2),
        "avg_calmar":        round(agg["calmar"], 3),
        "avg_dd":            round(agg["max_dd"], 2),
        "avg_sharpe":        round(agg["sharpe"], 3),
        "p95_yen":           round(p95_yen, 0),
        "p99_yen":           round(p99_yen, 0),
        "access_recovery":   round(access_recovery, 4),
        "capital_efficiency":round(capital_efficiency, 4),
        "cap_util":          round(cap_util, 4),
        "n_wf_folds":        agg["n_wf_folds"],
        "top5_skip_symbols": [s for s,_ in top5_skip],
        "_eq":               eq,   # kept for CF analysis, stripped before CSV
        "_inv":              inv,
        "_trades":           trades,
        "_skips":            skips,
    }


# ──────────────────────────────────────────────────────────────────────
#  ELASTICITY METRICS
# ──────────────────────────────────────────────────────────────────────

def compute_elasticity(cases_m: list[dict]) -> list[dict]:
    """Per-step marginal metrics + adoption gate."""
    rows = []
    prev = None
    for c in cases_m:
        row = {
            "case_id":          c["case_id"],
            "capital":          c["sl_cash_init"],
            "fillable":         c["fillable"],
            "forced_skip":      c["forced_skip"],
            "avg_calmar":       c["avg_calmar"],
            "access_recovery":  c["access_recovery"],
            "capital_efficiency": c["capital_efficiency"],
            "marginal_access":  None,
            "marginal_alpha":   None,
            "stop_flag":        False,
        }
        if prev:
            dc_M  = (c["sl_cash_init"] - prev["sl_cash_init"]) / 1_000_000
            df    = c["fillable"] - prev["fillable"]
            dK    = c["avg_calmar"] - prev["avg_calmar"]
            ma    = df / dc_M if dc_M > 0 else 0.0
            ml    = dK / df   if abs(df) > 0.001 else 0.0
            row["marginal_access"] = round(ma, 4)
            row["marginal_alpha"]  = round(ml, 4)
        rows.append(row)
        prev = c
    # stop flag
    consec = 0
    for r in rows:
        ma, ml = r["marginal_access"], r["marginal_alpha"]
        if ma is not None and ml is not None:
            if ma < STOP_MARGINAL_ACCESS and abs(ml) < STOP_MARGINAL_ALPHA:
                consec += 1
                if consec >= STOP_CONSECUTIVE:
                    r["stop_flag"] = True
            else:
                consec = 0
    return rows


# ──────────────────────────────────────────────────────────────────────
#  COUNTERFACTUAL MISSED ALPHA
# ──────────────────────────────────────────────────────────────────────

def counterfactual_missed(
    skip_log: list[dict],
    open_mat, close_mat, rsr_mat,
    sym_to_i, common_dates, n_dates, idx_map,
) -> list[dict]:
    """For each skip, simulate virtual (lot=1) trade and record return."""
    rows = []
    for s in skip_log:
        sym  = s["symbol"]
        si   = sym_to_i.get(sym)
        if si is None: continue
        skip_i = idx_map.get(s["date"])
        if skip_i is None: continue
        entry_i = skip_i + 1
        if entry_i >= n_dates: continue
        ep = float(open_mat[entry_i, si])
        if ep <= 0: continue

        days_below = 0; exit_i = None; exit_p = None; reason = "OPEN_AT_END"
        for j in range(entry_i, n_dates - 1):
            rv = float(rsr_mat[j, si])
            hd = j - entry_i
            days_below = (days_below + 1) if rv < EXIT_THR else 0
            if days_below >= 1 and hd >= MIN_HOLD:
                exit_i = j; exit_p = float(open_mat[j+1, si]); reason = "RSR_EXIT"
                break

        if exit_p is None or exit_p <= 0:
            exit_p = float(close_mat[-1, si]); exit_i = n_dates-1
        hold_d  = (exit_i or entry_i) - entry_i
        vret    = (exit_p / ep - 1) if ep > 0 else 0.0
        rows.append({
            "symbol":         sym,
            "skip_date":      s["date"],
            "entry_price":    round(ep, 0),
            "exit_price":     round(exit_p, 0),
            "hold_days":      hold_d,
            "virt_ret_pct":   round(vret * 100, 2),
            "exit_reason":    reason,
            "min_lot_cost":   round(s.get("min_lot_cost", ep*LOT), 0),
            "lot_binding":    s.get("lot_binding", True),
            "price_binding":  s.get("price_binding", False),
        })
    return rows


def missed_alpha_summary(cf_rows: list[dict]) -> dict:
    if not cf_rows:
        return {"n": 0, "avg_virt_ret": 0.0, "median_virt_ret": 0.0,
                "win_rate": 0.0, "avg_hold": 0.0,
                "lot_bound_avg": 0.0, "price_bound_avg": 0.0}
    df = pd.DataFrame(cf_rows)
    ldf = df[df["lot_binding"]]
    pdf = df[df["price_binding"]]
    return {
        "n":               len(df),
        "avg_virt_ret":    round(float(df["virt_ret_pct"].mean()), 2),
        "median_virt_ret": round(float(df["virt_ret_pct"].median()), 2),
        "win_rate":        round(float((df["virt_ret_pct"] > 0).mean()), 4),
        "avg_hold":        round(float(df["hold_days"].mean()), 1),
        "lot_bound_avg":   round(float(ldf["virt_ret_pct"].mean()) if len(ldf)>0 else 0.0, 2),
        "price_bound_avg": round(float(pdf["virt_ret_pct"].mean()) if len(pdf)>0 else 0.0, 2),
    }


# ──────────────────────────────────────────────────────────────────────
#  REPORT
# ──────────────────────────────────────────────────────────────────────

def _yen(v):  return f"▲¥{int(abs(v)):,}" if v else "¥0"
def _pct(v):  return f"{v:.1%}"
def _pp(v):   return f"{v:+.1f}pp" if v is not None else "—"
def _fl(v):   return f"{v:+.4f}" if v is not None else "—"


def write_report(
    ref_m: dict,           # virtual fractional reference metrics
    cases_m: list[dict],   # per-case metrics
    elas: list[dict],      # elasticity rows
    cf_per_case: dict,     # {case_id: [cf_rows]}
    virt_frac_calmar: float,
    path: Path,
) -> None:
    today = time.strftime("%Y-%m-%d")
    L = []; w = L.append

    # ── determine efficient_capital and plateau ───────────────────────
    best_calmar   = max(m["avg_calmar"] for m in cases_m)
    calmar_floor  = best_calmar * CALMAR_RATIO_MIN

    efficient_cap = None
    for m in cases_m:
        if (m["fillable"]        >= FILL_TARGET and
            m["forced_skip"]     <= SKIP_TARGET and
            m["access_recovery"] >= ACCESS_RECOVERY_MIN and
            m["avg_calmar"]      >= calmar_floor):
            efficient_cap = m["sl_cash_init"]
            break

    # plateau = first case where marginal_access ≈ 0
    plateau_cap = None
    for i, r in enumerate(elas[1:], 1):
        ma = r.get("marginal_access", 1.0)
        if ma is not None and ma < 0.02:
            plateau_cap = r["capital"]; break

    # recommend_live_capital = efficient_cap + 20% margin (or next step)
    if efficient_cap:
        margin     = max(efficient_cap * 0.20, 300_000)
        rec_live   = efficient_cap + margin
    else:
        rec_live = CASES[-1]["capital"] * 1.2

    # capital_margin = rec_live - efficient_cap
    cap_margin = rec_live - efficient_cap if efficient_cap else 0

    # root_cause
    # — if access_recovery at best feasible > 90% and calmar near ref: ACCESS
    # — if even with full capital calmar is far below vf_calmar: ALPHA
    # — else MIXED
    best_m = cases_m[-1]
    if best_m["access_recovery"] >= 0.90:
        root_cause_final = "ACCESS"
    elif best_m["access_recovery"] < 0.70:
        root_cause_final = "ALPHA"
    else:
        root_cause_final = "MIXED"

    # decision
    if efficient_cap is not None:
        decision = "GO" if best_calmar >= virt_frac_calmar * 0.95 else "GO_LIMITED"
    else:
        decision = "NO_GO"

    next_step = (
        "Study16 Order Authority Gate"
        if decision != "NO_GO"
        else "STOP — strategy不採用確定"
    )

    # ── HEADER ────────────────────────────────────────────────────────
    w("# Study15B Access Elasticity Confirmation")
    w(f"\n作成日: {today}  |  監査のみ / 変更禁止")
    w(f"\n**Strategy**: Study9 Case B (固定)  **Reference**: Virtual Fractional Calmar={virt_frac_calmar:.3f}\n")

    # EXECUTIVE SUMMARY
    w("---\n## Executive Summary\n")
    w("| 項目 | 値 |")
    w("|---|---|")
    w(f"| **root_cause** | **{root_cause_final}** |")
    w(f"| **efficient_capital** | ¥{int(efficient_cap):,} |" if efficient_cap
      else "| efficient_capital | 未到達 |")
    w(f"| plateau_region | ¥{int(plateau_cap):,}+ |" if plateau_cap
      else "| plateau_region | データ不足 |")
    w(f"| capital_margin | ¥{int(cap_margin):,} |")
    w(f"| recommend_live_capital | ¥{int(rec_live):,} |")
    w(f"| **decision** | **{decision}** |")
    w(f"| next_phase | {next_step} |")
    w("")

    # ── REFERENCE ────────────────────────────────────────────────────
    w("---\n## 参照: Virtual Fractional (lot=1, cap=¥400k)\n")
    w("| 指標 | 値 |")
    w("|---|---|")
    w(f"| fillable | {_pct(ref_m['fillable'])} |")
    w(f"| calmar (avg fold) | {virt_frac_calmar:.3f} |")
    w(f"| CAGR | {ref_m['avg_cagr']:+.2f}% |")
    w(f"| trade_count | {ref_m['trade_count']} |")
    w("")

    # ── PER-CASE TABLE ────────────────────────────────────────────────
    w("---\n## Cases A–F: Capital Elasticity\n")

    # Adoption criteria columns
    fill_ok   = lambda m: "✅" if m["fillable"] >= FILL_TARGET else "❌"
    skip_ok   = lambda m: "✅" if m["forced_skip"] <= SKIP_TARGET else "❌"
    rec_ok    = lambda m: "✅" if m["access_recovery"] >= ACCESS_RECOVERY_MIN else "❌"
    cal_ok    = lambda m: "✅" if m["avg_calmar"] >= calmar_floor else "❌"
    all_ok    = lambda m: "🎯" if (m["fillable"] >= FILL_TARGET and
                                    m["forced_skip"] <= SKIP_TARGET and
                                    m["access_recovery"] >= ACCESS_RECOVERY_MIN and
                                    m["avg_calmar"] >= calmar_floor) else "—"

    w("| Case | cap | fillable | skip | calmar | CAGR | access_rec | cap_eff | P95_yen | 基準 |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for m in cases_m:
        cid = m["case_id"]
        w(f"| {cid}"
          f"| ¥{int(m['sl_cash_init'])//1000}k"
          f"| {m['fillable']:.1%} {fill_ok(m)}"
          f"| {m['forced_skip']:.1%} {skip_ok(m)}"
          f"| {m['avg_calmar']:.3f} {cal_ok(m)}"
          f"| {m['avg_cagr']:+.2f}%"
          f"| {m['access_recovery']:.1%} {rec_ok(m)}"
          f"| {m['capital_efficiency']:.3f}"
          f"| {_yen(m['p95_yen'])}"
          f"| {all_ok(m)} |")
    w("")

    # adoption criteria note
    w(f"> 採用基準: fillable≥{FILL_TARGET:.0%} / skip≤{SKIP_TARGET:.0%} / "
      f"access_rec≥{ACCESS_RECOVERY_MIN:.0%} / calmar≥{calmar_floor:.3f}(={CALMAR_RATIO_MIN:.0%}×best)")
    w("")

    # ── ELASTICITY TABLE ─────────────────────────────────────────────
    w("---\n## 弾力性・停止判定\n")
    w("| Case | cap | Δfillable | ΔCalmar | marginal_access | marginal_alpha | 停止 |")
    w("|---|---|---|---|---|---|---|")
    prev_fill = ref_m["fillable"]
    prev_cal  = ref_m["avg_calmar"]
    for r in elas:
        df_pp = (r["fillable"] - prev_fill) * 100
        dcal  = r["avg_calmar"] - prev_cal
        stop  = "🛑 STOP" if r.get("stop_flag") else ""
        ma_s  = f"{r['marginal_access']:.4f}" if r["marginal_access"] is not None else "—"
        ml_s  = f"{r['marginal_alpha']:.4f}"  if r["marginal_alpha"]  is not None else "—"
        w(f"| {r['case_id']}"
          f"| ¥{int(r['capital'])//1000}k"
          f"| {df_pp:+.1f}pp"
          f"| {dcal:+.3f}"
          f"| {ma_s}"
          f"| {ml_s}"
          f"| {stop} |")
        prev_fill = r["fillable"]
        prev_cal  = r["avg_calmar"]
    w("")

    # ── BINDING ANALYSIS ─────────────────────────────────────────────
    w("---\n## Lot/Price Binding Rate\n")
    w("| Case | cap | n_skip | lot_binding | price_binding | top skip 銘柄 |")
    w("|---|---|---|---|---|---|")
    for m in cases_m:
        top = ", ".join(m.get("top5_skip_symbols", [])[:3])
        w(f"| {m['case_id']}"
          f"| ¥{int(m['sl_cash_init'])//1000}k"
          f"| {m['n_skip']}"
          f"| {m['lot_binding_rate']:.0%}"
          f"| {m['price_binding_rate']:.0%}"
          f"| {top} |")
    w("")

    # ── MISSED ALPHA ─────────────────────────────────────────────────
    w("---\n## Missed Alpha (Counterfactual)\n")
    w("| Case | cap | n_missed | avg_virt_ret | win_rate | lot_bound_avg | price_bound_avg |")
    w("|---|---|---|---|---|---|---|")
    for m in cases_m:
        cid = m["case_id"]
        s = cf_per_case.get(cid, {})
        w(f"| {cid}"
          f"| ¥{int(m['sl_cash_init'])//1000}k"
          f"| {s.get('n',0)}"
          f"| {s.get('avg_virt_ret',0.0):+.2f}%"
          f"| {s.get('win_rate',0.0):.1%}"
          f"| {s.get('lot_bound_avg',0.0):+.2f}%"
          f"| {s.get('price_bound_avg',0.0):+.2f}% |")
    w("")

    # ── TOP10 MISSED TRADES (across all cases at Case A = ¥1M) ──────
    w("---\n## Top10 Missed Trades (Case A ¥1.0M)\n")
    cf_a = cf_per_case.get("A_raw", [])
    if cf_a:
        top10 = sorted(cf_a, key=lambda x: x.get("virt_ret_pct",0), reverse=True)[:10]
        df_t  = pd.DataFrame(top10)[["symbol","skip_date","entry_price",
                                      "virt_ret_pct","hold_days",
                                      "exit_reason","min_lot_cost","lot_binding"]]
        w(df_t.to_markdown(index=False))
    else:
        w("スキップなし\n")
    w("")

    # ── ACCESS LOSS BREAKDOWN ─────────────────────────────────────────
    w("---\n## Access Loss Breakdown\n")
    w("| Case | access_recovery | access_loss | lot_driven_loss | capital_gap_yen |")
    w("|---|---|---|---|---|")
    for m in cases_m:
        rec   = m["access_recovery"]
        loss  = 1 - rec
        # lot-driven: estimated as lot_binding_rate × skip fraction
        lot_d = loss * m["lot_binding_rate"]
        gap   = (efficient_cap - m["sl_cash_init"]) if efficient_cap else 0
        gap   = max(0, gap)
        w(f"| {m['case_id']}"
          f"| {rec:.1%}"
          f"| {loss:.1%}"
          f"| {lot_d:.1%}"
          f"| ¥{int(gap):,} |")
    w("")

    # ── CAPITAL FRONTIER TABLE ────────────────────────────────────────
    w("---\n## Capital Frontier Summary\n")
    w("| capital | fillable | Calmar | cap_efficiency | access_recovery | 評価 |")
    w("|---|---|---|---|---|---|")
    for m in cases_m:
        note = "✅ 採用基準達成" if all_ok(m) == "🎯" else (
               "⚠ fillable不足" if m["fillable"] < FILL_TARGET else
               "⚠ calmar不足")
        w(f"| ¥{int(m['sl_cash_init']):,}"
          f"| {m['fillable']:.1%}"
          f"| {m['avg_calmar']:.3f}"
          f"| {m['capital_efficiency']:.4f}"
          f"| {m['access_recovery']:.1%}"
          f"| {note} |")
    w("")

    # ── PLATEAU ANALYSIS ─────────────────────────────────────────────
    w("---\n## Plateau Analysis\n")
    if plateau_cap:
        w(f"**plateau_region: ¥{int(plateau_cap):,}+**\n")
        w(f"平坦化点: marginal_access < {STOP_MARGINAL_ACCESS} (Δfillable/Δ¥M)")
        w(f"この水準以上の資本追加は fillable 改善が飽和 → calmar への追加効果も限界的\n")
    else:
        w("plateau未到達 — sweep範囲内で完全飽和\n")

    # ── FINAL VERDICT ─────────────────────────────────────────────────
    w("---\n## 最終判定\n")
    w("| 項目 | 値 |")
    w("|---|---|")
    w(f"| root_cause | **{root_cause_final}** |")
    w(f"| efficient_capital | ¥{int(efficient_cap):,} |" if efficient_cap
      else "| efficient_capital | 未到達 |")
    w(f"| plateau_region | ¥{int(plateau_cap):,}+ |" if plateau_cap
      else "| plateau_region | — |")
    w(f"| capital_margin | ¥{int(cap_margin):,} (={cap_margin/efficient_cap:.0%} buffer) |"
      if efficient_cap else "| capital_margin | — |")
    w(f"| recommend_live_capital | **¥{int(rec_live):,}** |")
    w(f"| virtual_frac_calmar (ref) | {virt_frac_calmar:.3f} |")
    w(f"| best_realized_calmar | {best_calmar:.3f} |")
    w(f"| access_recovery@best | {best_m['access_recovery']:.1%} |")
    w(f"| decision | **{decision}** |")
    w(f"| next_phase | {next_step} |")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")
    print(f"  レポート: {path}")


# ──────────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────────

def main() -> int:
    cfg = load_strategy_config()
    print("=" * 68)
    print("  Study15B Access Elasticity Confirmation")
    print("  Cases: ¥1.0M / ¥1.2M / ¥1.35M / ¥1.5M / ¥1.8M / ¥2.2M")
    print("=" * 68 + "\n")

    # ── Data ──────────────────────────────────────────────────────────
    print("[1/3] データロード...")
    (universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
     rsr_syms, topix_close, cfg) = load_data(cfg)

    trade_syms  = {s: v for s, v in rsr_syms.items() if s in universe_raw}
    active_syms = list(trade_syms.keys())
    sym_to_i    = {s: i for i, s in enumerate(active_syms)}
    n_syms      = len(active_syms)

    common_dates = None
    for sym in active_syms:
        idx = universe_raw[sym]["df"].index
        common_dates = idx if common_dates is None else common_dates.intersection(idx)
    common_dates = common_dates.sort_values()
    common_dates = common_dates[
        (common_dates >= pd.Timestamp(IS_START)) &
        (common_dates <= pd.Timestamp("2025-12-31"))
    ]
    n_dates  = len(common_dates)
    idx_map  = {str(d.date()): i for i, d in enumerate(common_dates)}
    print(f"  共通日数: {n_dates}  銘柄数: {n_syms}\n")

    # ── Matrices ──────────────────────────────────────────────────────
    print("[2/3] マトリクス構築...")
    open_mat  = np.full((n_dates, n_syms), np.nan, dtype=np.float32)
    close_mat = np.full((n_dates, n_syms), np.nan, dtype=np.float32)
    for si, sym in enumerate(active_syms):
        df_src  = universe_raw[sym]["df"]
        row_idx = df_src.index.get_indexer(common_dates)
        if np.any(row_idx < 0): continue
        open_mat[:,  si] = df_src["Open"].to_numpy(dtype=np.float32)[row_idx]
        close_mat[:, si] = df_src["Close"].to_numpy(dtype=np.float32)[row_idx]

    rsr_mat = np.nan_to_num(
        _take(rsr_df, common_dates, active_syms, dtype=np.float32, fill_value=np.nan),
        nan=0.0)
    sym_active_mat = (None if sym_active_df is None else
                      _take(sym_active_df, common_dates, active_syms,
                            dtype=np.float32, fill_value=1.0))
    mkt_ret1 = None
    if topix_close is not None:
        mkt_ret1 = _take(topix_close.pct_change(), common_dates,
                         dtype=np.float32, fill_value=0.0)

    cross90_mat = compute_cross90_mat(rsr_mat)
    slope5_mat  = compute_slope5(rsr_mat)

    def sim_and_metrics(cap, lot=100, price_cap=None, label="?", vf_calmar=1.0):
        r = run_sim(open_mat, close_mat, rsr_mat, sym_active_mat,
                    cross90_mat, slope5_mat, mkt_ret1,
                    active_syms, sym_to_i, common_dates, n_dates,
                    sl_cash_init=cap, lot_size=lot, price_cap=price_cap)
        m = compute_all_metrics(r, cap, common_dates, FOLDS, vf_calmar)
        m["case_id"] = label
        m["_skip_log"] = r["skip_log"]
        return m

    # ── Reference: Virtual Fractional ────────────────────────────────
    print("[3/3] シミュレーション実行...")
    print("  [REF] Virtual Fractional (cap=400k, lot=1)...")
    ref_r  = run_sim(open_mat, close_mat, rsr_mat, sym_active_mat,
                     cross90_mat, slope5_mat, mkt_ret1,
                     active_syms, sym_to_i, common_dates, n_dates,
                     sl_cash_init=400_000, lot_size=1)
    ref_fm = compute_fold_metrics(ref_r["eq"], ref_r["inv"], ref_r["trades"],
                                  common_dates, FOLDS, 400_000)
    ref_agg         = aggregate(ref_fm)
    virt_frac_calmar = ref_agg["calmar"]
    ref_m = compute_all_metrics(ref_r, 400_000, common_dates, FOLDS, virt_frac_calmar)
    ref_m["case_id"] = "REF"
    print(f"      calmar={virt_frac_calmar:.3f}  fillable={ref_m['fillable']:.1%}")

    # ── Capital Cases A–F ─────────────────────────────────────────────
    cases_m  = []
    cf_per_case = {}
    stop_consecutive = 0

    for case in CASES:
        cid = case["id"]
        cap = case["capital"]
        print(f"  [{cid}] cap=¥{cap:,}  lot=100")
        m = sim_and_metrics(cap, lot=100, label=cid, vf_calmar=virt_frac_calmar)
        print(f"      fillable={m['fillable']:.1%}  calmar={m['avg_calmar']:.3f}"
              f"  skip={m['n_skip']}  access_rec={m['access_recovery']:.1%}")
        cases_m.append(m)

        # Counterfactual for skips in this case
        skip_log_c = m.pop("_skip_log")
        cf_rows = counterfactual_missed(
            skip_log_c, open_mat, close_mat, rsr_mat,
            sym_to_i, common_dates, n_dates, idx_map)
        cf_per_case[cid]         = missed_alpha_summary(cf_rows)
        cf_per_case[f"{cid}_raw"] = cf_rows
        if cf_rows:
            avg_v = float(np.mean([r["virt_ret_pct"] for r in cf_rows]))
            print(f"      cf_skips={len(cf_rows)}  avg_virt_ret={avg_v:+.2f}%")

    # ── Elasticity metrics ────────────────────────────────────────────
    elas = compute_elasticity(cases_m)

    # ── Report ───────────────────────────────────────────────────────
    print("\n[OUT] レポート出力...")
    write_report(
        ref_m, cases_m, elas, cf_per_case, virt_frac_calmar,
        REPORTS_DIR / "study15b_access_elasticity.md",
    )

    # Save fillability_curve CSV
    curve = [{"case": m["case_id"], "capital": m["sl_cash_init"],
              "fillable": m["fillable"], "calmar": m["avg_calmar"],
              "access_recovery": m["access_recovery"],
              "capital_efficiency": m["capital_efficiency"],
              "p95_yen": m["p95_yen"],
              "marginal_access": elas[i]["marginal_access"],
              "marginal_alpha":  elas[i]["marginal_alpha"]}
             for i, m in enumerate(cases_m)]
    pd.DataFrame(curve).to_csv(REPORTS_DIR / "study15b_frontier.csv",
                                index=False, encoding="utf-8-sig")
    print(f"  CSV: study15b_frontier.csv")
    print("\n完了.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
