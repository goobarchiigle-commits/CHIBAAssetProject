"""
src/backtest/study15_capacity_attribution.py  —  Study15 Lot-Constrained Capacity Attribution

Root-cause attribution for Study14 NO_GO (Lot Feasibility FAIL).

Case A  Baseline        cap=400k, LOT=100           (reproduce Study14)
Case B  VirtualFrac     cap=400k, LOT=1 (virtual)   (lot-constraint removed)
Case C  PriceCap        cap=400k, LOT=100, filter bp×LOT≤cap
Case D  CapSweep        cap=[400k..2.2M], LOT=100    (elasticity curve)
Case E  CapSweep+Price  cap=[400k..2.2M], LOT=100, filter bp×LOT≤cap

Output:
  reports/capacity_audit.md
  reports/capacity_cases.csv
  reports/fillability_curve.csv
  reports/counterfactual_access.csv

Run:
    cd C:/ai-trading
    python src/backtest/study15_capacity_attribution.py
"""
from __future__ import annotations

import sys, math, time, warnings
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.config_loader import load_strategy_config
from src.backtest.capital_allocation_abc import load_data, _take, calc_metrics, LOT, COST_ONE_WAY

REPORTS_DIR = Path("reports")
IS_START    = "2018-01-01"

# ── Strategy fixed (Study9 Case B) ────────────────────────────────────
RSR_LO    = 92.0
RSR_HI    = 95.0
D90_MAX   = 5
SLOPE_MAX = 5.0
EXIT_THR  = 90.0
MIN_HOLD  = 3
SHOCK_MKT = -0.05
SHOCK_SYM = -0.08

# ── Study15 case parameters ───────────────────────────────────────────
CAP_BASE = 400_000
CAP_SWEEP = [400_000, 600_000, 800_000, 1_000_000,
             1_200_000, 1_500_000, 1_800_000, 2_200_000]
SWEEP_STOP_DELTA = 5.0    # pp: stop sweep when Δfillable < 5pp

FOLDS = [
    {"id": 1, "oos_start": "2021-01-01", "oos_end": "2021-12-31"},
    {"id": 2, "oos_start": "2022-01-01", "oos_end": "2022-12-31"},
    {"id": 3, "oos_start": "2023-01-01", "oos_end": "2023-12-31"},
    {"id": 4, "oos_start": "2024-01-01", "oos_end": "2024-12-31"},
    {"id": 5, "oos_start": "2025-01-01", "oos_end": "2025-12-31"},
]

# ── matrix helpers (same as study14) ────────────────────────────────
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
#  CORE SIMULATION ENGINE
# ──────────────────────────────────────────────────────────────────────

@dataclass
class SlPos:
    symbol:       str
    qty:          int
    entry_price:  float
    entry_idx:    int
    rsr_entry:    float
    d90_entry:    int
    slope5_entry: float
    days_below:   int = 0


def run_sim(
    open_mat, close_mat, rsr_mat, sym_active_mat,
    cross90_mat, slope5_mat, mkt_ret1,
    active_syms, sym_to_i, common_dates, n_dates: int,
    sl_cash_init: float,
    lot_size: int = 100,          # virtual: 1; real: 100
    price_cap: Optional[float] = None,  # Case C/E: reject if bp*LOT > price_cap
    annual_rebalance_target: Optional[float] = None,  # governance rebalance
) -> dict:
    """
    Generic simulation for Study15 cases.

    lot_size        : minimum tradeable unit (100=real, 1=virtual fractional)
    price_cap       : if set, skip entries where bp*LOT > price_cap
    annual_rebalance_target: if set, snap sl_cash to target on each Jan-01 trading day
    """
    sl_cash = sl_cash_init
    sl_pos: SlPos | None = None
    trades:   list[dict] = []
    skip_log: list[dict] = []   # forced skips (lot constraint)
    price_rejects: list[dict] = []  # price-cap rejections

    eq  = np.zeros(n_dates, dtype=np.float64)
    inv = np.zeros(n_dates, dtype=np.float64)

    prev_year = -1

    for i, date in enumerate(common_dates):
        ds = str(date.date())
        yr = date.year

        # ── Annual rebalance governance ─────────────────────────────
        if annual_rebalance_target is not None:
            if yr != prev_year and sl_pos is None:
                sl_cash = annual_rebalance_target
            prev_year = yr

        s_inv = (sl_pos.qty * float(close_mat[i, sym_to_i[sl_pos.symbol]])
                 if sl_pos else 0.0)
        s_eq  = sl_cash + s_inv
        eq[i]  = s_eq
        inv[i] = s_inv

        mkt_shock = mkt_ret1 is not None and float(mkt_ret1[i]) <= SHOCK_MKT
        if i + 1 >= n_dates:
            break
        nxt = i + 1

        # ── EXIT ─────────────────────────────────────────────────────
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
                    "lot_size_used": lot_size,
                })
                sl_pos = None

        # ── ENTRY ────────────────────────────────────────────────────
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
                if bp <= 0:
                    continue

                min_lot_cost = bp * LOT     # real minimum (always 100 shares)

                # Case C/E: price cap filter
                if price_cap is not None and min_lot_cost > price_cap:
                    price_rejects.append({
                        "date": ds, "symbol": sym, "price": bp,
                        "min_lot_cost": min_lot_cost, "sl_cash": sl_cash,
                    })
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
                        "alloc_used": cost, "sl_cash_before": sl_cash + cost,
                        "min_lot_cost": min_lot_cost,
                        "buy_date": ds, "rsr_entry": rv,
                        "lot_size_used": lot_size,
                    })
                else:
                    # forced skip — lot constraint
                    skip_log.append({
                        "date": ds, "symbol": sym, "price": bp,
                        "min_lot_cost_real": LOT * bp,
                        "sl_cash": sl_cash, "alloc": alloc,
                        "qty_real_would": int(alloc / bp / LOT) * LOT,
                        "qty_virt_would": int(alloc / bp),  # fractional
                    })

    return {
        "eq": eq, "inv": inv,
        "trades": trades,
        "skip_log": skip_log,
        "price_rejects": price_rejects,
    }


# ──────────────────────────────────────────────────────────────────────
#  METRICS COMPUTATION
# ──────────────────────────────────────────────────────────────────────

def compute_metrics(result: dict, sl_cash_init: float, folds: list[dict],
                    common_dates, n_dates: int) -> dict:
    eq     = result["eq"]
    inv    = result["inv"]
    trades = result["trades"]
    skips  = result["skip_log"]
    price_r = result["price_rejects"]

    sells = [t for t in trades if t["side"] == "SELL"]
    buys  = [t for t in trades if t["side"] == "BUY"]

    n_buy  = len(buys)
    n_skip = len(skips)
    n_prcap = len(price_r)
    n_attempt = n_buy + n_skip   # price_rejects are filtered before attempt

    fillable = n_buy / max(1, n_attempt)
    skip_rate = n_skip / max(1, n_attempt)

    # avg buy price and fill_loss (lot-rounding waste)
    buy_prices = [t["entry"] for t in buys if t.get("entry", 0) > 0]
    avg_buy_price = float(np.mean(buy_prices)) if buy_prices else 0.0

    # fill_loss: fraction of alloc wasted due to lot rounding
    fill_losses = []
    for t in buys:
        alloc_used = t.get("alloc_used", 0)
        sl_before  = t.get("sl_cash_before", 0)
        alloc_max  = sl_before * 0.95
        if alloc_max > 0:
            fill_losses.append(1 - alloc_used / alloc_max)
    fill_loss_avg = float(np.mean(fill_losses)) if fill_losses else 0.0

    # skip distribution
    skip_prices = [s["price"]             for s in skips]
    skip_mlc    = [s["min_lot_cost_real"] for s in skips]
    avg_skip_price = float(np.mean(skip_prices)) if skip_prices else 0.0
    avg_skip_mlc   = float(np.mean(skip_mlc))    if skip_mlc   else 0.0

    # lot_concentration: top-5 symbols by skip count
    from collections import Counter
    skip_sym_count = Counter(s["symbol"] for s in skips)
    top5_skip = skip_sym_count.most_common(5)
    top5_skip_weight = sum(v for _, v in top5_skip) / max(1, n_skip)

    # portfolio-level CAGR / DD / Calmar over full period
    date_strs = [str(d.date()) for d in common_dates]
    fold_metrics = []
    for fold in folds:
        ds_s, ds_e = fold["oos_start"], fold["oos_end"]
        oos_idx = [i for i, s in enumerate(date_strs) if ds_s <= s <= ds_e]
        if not oos_idx: continue
        si, ei = oos_idx[0], oos_idx[-1] + 1
        s_eq = eq[si:ei]
        s_inv = inv[si:ei]
        if len(s_eq) < 5: continue
        s_trades = [t for t in sells
                    if ds_s <= t.get("exit_date","") <= ds_e]
        exp_list = [vi/max(1.0,ve) for vi,ve in zip(s_inv, s_eq)]
        m = calc_metrics(s_eq.tolist(), s_trades, exp_list, s_eq[0],
                         list(common_dates[si:ei]))
        fold_metrics.append(m)

    if fold_metrics:
        avg_cagr   = float(np.mean([m.get("cagr",0)        for m in fold_metrics]))
        avg_dd     = float(np.mean([m.get("max_dd",0)       for m in fold_metrics]))
        avg_calmar = float(np.mean([m.get("calmar",0)       for m in fold_metrics]))
        avg_sharpe = float(np.mean([m.get("sharpe",0)       for m in fold_metrics]))
    else:
        avg_cagr = avg_dd = avg_calmar = avg_sharpe = 0.0

    # P95 / P99 DD in yen (bootstrap)
    p95_yen, p99_yen = _bootstrap_dd_yen(eq, common_dates, folds, sl_cash_init, n_boot=3000)

    # capital utilization
    cap_util = float((inv > 0).mean())
    idle_ratios = [(eq[i] - inv[i]) / max(1.0, eq[i]) for i in range(n_dates)
                   if eq[i] > 0]
    median_idle = float(np.median(idle_ratios)) if idle_ratios else 1.0

    # avg trade quality: avg realized_ret of wins
    win_rets = [t["realized_ret"] for t in sells if t.get("realized_ret",0) > 0]
    avg_win_ret = float(np.mean(win_rets)) if win_rets else 0.0
    win_rate = len(win_rets) / max(1, len(sells))

    # effective_capacity
    eff_cap = sl_cash_init * fillable

    return {
        "sl_cash_init":      sl_cash_init,
        "n_buy":             n_buy,
        "n_skip":            n_skip,
        "n_price_reject":    n_prcap,
        "n_attempt":         n_attempt,
        "fillable_rate":     round(fillable, 4),
        "forced_skip_rate":  round(skip_rate, 4),
        "avg_buy_price":     round(avg_buy_price, 0),
        "avg_skip_price":    round(avg_skip_price, 0),
        "avg_skip_mlc":      round(avg_skip_mlc, 0),
        "fill_loss_avg":     round(fill_loss_avg, 4),
        "lot_conc_top5":     round(top5_skip_weight, 4),
        "top5_skip_symbols": [s for s,_ in top5_skip],
        "avg_cagr":          round(avg_cagr, 2),     # calc_metrics already %
        "avg_dd":            round(avg_dd, 2),       # calc_metrics already %
        "avg_calmar":        round(avg_calmar, 3),
        "avg_sharpe":        round(avg_sharpe, 3),
        "p95_dd_yen":        round(p95_yen, 0),
        "p99_dd_yen":        round(p99_yen, 0),
        "cap_util_pct":      round(cap_util * 100, 2),
        "median_idle":       round(median_idle, 4),
        "avg_win_ret_pct":   round(avg_win_ret * 100, 2),
        "win_rate":          round(win_rate, 4),
        "n_trades":          len(sells),
        "eff_capacity":      round(eff_cap, 0),
    }


def _bootstrap_dd_yen(eq, common_dates, folds, sl_cash_init: float,
                       n_boot: int = 3000, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    date_strs = [str(d.date()) for d in common_dates]
    fold_dds = []
    fold_cagrs = []
    for fold in folds:
        ds_s, ds_e = fold["oos_start"], fold["oos_end"]
        oos_idx = [i for i, s in enumerate(date_strs) if ds_s <= s <= ds_e]
        if not oos_idx: continue
        si, ei = oos_idx[0], oos_idx[-1]+1
        s_eq = eq[si:ei]
        if len(s_eq) < 5: continue
        y_fac = 252 / max(1, ei-si)
        cagr  = (s_eq[-1]/max(1e-6, s_eq[0]))**y_fac - 1
        peak  = np.maximum.accumulate(s_eq)
        dd    = float(np.nanmin((s_eq-peak)/peak))
        fold_dds.append(dd)
        fold_cagrs.append(cagr)

    if len(fold_dds) < 2:
        return abs(float(np.nanmin(eq - np.maximum.accumulate(eq))) * 0), 0.0

    fc = np.array(fold_cagrs); fd = np.array(fold_dds)
    boot_dd = []
    for _ in range(n_boot):
        idx   = rng.integers(0, len(fc), size=5)
        eq_b  = sl_cash_init
        min_eq = sl_cash_init
        for ci, di in zip(fc[idx], fd[idx]):
            trough = eq_b * (1+di)
            min_eq = min(min_eq, trough)
            eq_b   = max(0.001, eq_b*(1+ci))
        boot_dd.append((min_eq - sl_cash_init) / sl_cash_init)

    arr = np.array(boot_dd)
    p95 = abs(float(np.percentile(arr, 5))) * sl_cash_init
    p99 = abs(float(np.percentile(arr, 1))) * sl_cash_init
    return p95, p99


# ──────────────────────────────────────────────────────────────────────
#  COUNTERFACTUAL ACCESS: what if skips were executed?
# ──────────────────────────────────────────────────────────────────────

def counterfactual_access(
    skip_log_a: list[dict],
    open_mat, close_mat, rsr_mat,
    sym_to_i: dict, common_dates, n_dates: int,
    idx_map: dict,   # date_str -> row index
) -> pd.DataFrame:
    """
    For each forced skip in Case A:
      simulate: enter at open[next_day], exit at first RSR<90 with hold>=3
      record: virtual_ret, virtual_pnl_per_share, hold_days
    Represents 'what alpha was left on table'.
    """
    rows = []
    for s in skip_log_a:
        sym = s["symbol"]
        si  = sym_to_i.get(sym)
        if si is None: continue

        skip_date = s["date"]
        skip_i    = idx_map.get(skip_date)
        if skip_i is None: continue

        entry_i = skip_i + 1
        if entry_i >= n_dates: continue

        ep = float(open_mat[entry_i, si])
        if ep <= 0: continue

        # simulate exit with RSR_EXIT rule
        days_below = 0
        exit_i = None; exit_p = None; reason = "OPEN"
        for j in range(entry_i, n_dates - 1):
            rv = float(rsr_mat[j, si])
            hd = j - entry_i
            if rv < EXIT_THR:
                days_below += 1
            else:
                days_below = 0
            if days_below >= 1 and hd >= MIN_HOLD:
                exit_i = j; exit_p = float(open_mat[j+1, si]); reason = "RSR_EXIT"
                break

        if exit_p is None or exit_p <= 0:
            # still open at end of data
            exit_p = float(close_mat[-1, si])
            exit_i = n_dates - 1
            reason = "OPEN_AT_END"

        hold_d = (exit_i or entry_i) - entry_i
        virt_ret = (exit_p / ep - 1) if ep > 0 else 0.0

        rows.append({
            "skip_date":   skip_date,
            "symbol":      sym,
            "entry_price": round(ep, 0),
            "exit_price":  round(exit_p, 0),
            "hold_days":   hold_d,
            "virtual_ret_pct": round(virt_ret * 100, 2),
            "exit_reason": reason,
            "min_lot_cost": round(s.get("min_lot_cost_real", ep*LOT), 0),
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ──────────────────────────────────────────────────────────────────────
#  REPORT WRITER
# ──────────────────────────────────────────────────────────────────────

def _pct(v): return f"{v:.1%}"
def _pp(v):  return f"{v:+.2f}pp" if not (isinstance(v, float) and math.isnan(v)) else "—"
def _yen(v): return f"▲¥{int(abs(v)):,}" if v else "¥0"


def write_report(
    cA: dict, cB: dict, cC: dict,
    sweep_D: list[dict], sweep_E: list[dict],
    cf_df: pd.DataFrame,
    path: Path,
) -> None:
    today = time.strftime("%Y-%m-%d")
    L = []; w = L.append

    # ── root cause determination ─────────────────────────────────────
    fillA = cA["fillable_rate"]
    fillB = cB["fillable_rate"]   # virtual fractional
    fillC = cC["fillable_rate"]   # price cap

    # Δfillable from A to B: removing lot constraint
    delta_lot  = fillB - fillA     # lot-constraint contribution
    # Δfillable from A to C: adding price cap (restricts to affordable stocks)
    # C might have LOWER fillable due to fewer candidates
    delta_price_cap = fillC - fillA  # price-cap only effect

    # alpha_realization vs Case B
    calmar_B = cB["avg_calmar"]
    alpha_real_C = cC["avg_calmar"] / max(0.001, calmar_B)  # Calmar ratio C/B
    alpha_real_A = cA["avg_calmar"] / max(0.001, calmar_B)  # Calmar ratio A/B

    # capital_elasticity from Case D sweep
    if len(sweep_D) >= 2:
        fill_vals  = [s["fillable_rate"] for s in sweep_D]
        cap_vals   = [s["sl_cash_init"]  for s in sweep_D]
        deltas_f = [fill_vals[i+1]-fill_vals[i] for i in range(len(fill_vals)-1)]
        deltas_c = [(cap_vals[i+1]-cap_vals[i])/100_000 for i in range(len(cap_vals)-1)]
        elasticities = [df/dc if dc > 0 else 0 for df, dc in zip(deltas_f, deltas_c)]
        avg_elasticity = float(np.mean(elasticities))
        alpha_elas_arr = [
            (sweep_D[i+1]["avg_calmar"] - sweep_D[i]["avg_calmar"]) /
            max(0.001, (cap_vals[i+1]-cap_vals[i])/100_000)
            for i in range(len(sweep_D)-1)
        ]
        avg_alpha_elasticity = float(np.mean(alpha_elas_arr))
        # find minimum efficient capital (first cap where fillable >= 90%)
        min_eff_cap = next(
            (s["sl_cash_init"] for s in sweep_D if s["fillable_rate"] >= 0.90),
            sweep_D[-1]["sl_cash_init"]
        )
    else:
        avg_elasticity = avg_alpha_elasticity = 0.0
        min_eff_cap = 0

    # root cause verdict
    if delta_lot >= 0.20:   # virtual fractional gains ≥20pp fillable
        if alpha_real_C >= 0.95:
            root_cause = "LOT"   # lot constraint is primary, high-price stocks dispensable
        else:
            root_cause = "MIXED"  # lot + price both matter
    elif delta_price_cap < -0.10:
        root_cause = "PRICE"  # price cap cuts too much → high-price stocks essential
    else:
        root_cause = "CAPITAL"

    # final decision
    best_fill = max(s["fillable_rate"] for s in sweep_D) if sweep_D else fillA
    if best_fill >= 0.90 and min_eff_cap <= 2_200_000:
        decision = "GO_LIMITED"
        next_phase = f"Study16 Order Authority Gate (min_capital=¥{int(min_eff_cap):,})"
    else:
        decision = "NO_GO"
        next_phase = "市場制約として終了 (min_lot>実効資本上限)"

    # ── HEADER ───────────────────────────────────────────────────────
    w("# Study15 Lot-Constrained Capacity Attribution Audit")
    w(f"\n作成日: {today}  |  監査のみ / 変更禁止 / live注文禁止")
    w(f"\n**Strategy**: Study9 Case B (entry/exit/signal完全固定)")
    w(f"**Purpose**: Study14 NO_GO (Lot Feasibility skip=31.5%) 原因分離\n")

    # Executive summary table
    w("---\n## Executive Summary\n")
    w(f"| 項目 | 値 |")
    w("|---|---|")
    w(f"| **root_cause** | **{root_cause}** |")
    w(f"| **decision** | **{decision}** |")
    w(f"| minimum_efficient_capital | ¥{int(min_eff_cap):,} |")
    w(f"| expected_fillable | {_pct(max(s['fillable_rate'] for s in sweep_D) if sweep_D else fillA)} |")
    w(f"| expected_alpha (vs VirtualFrac) | {alpha_real_C:.1%} (Case C) |")
    w(f"| capital_elasticity | {avg_elasticity:+.4f} pp / ¥100k |")
    w(f"| alpha_elasticity | {avg_alpha_elasticity:+.4f} Calmar / ¥100k |")
    w(f"| next_phase | {next_phase} |")
    w("")

    # ── CASE A ───────────────────────────────────────────────────────
    w("---\n## Case A: Baseline (cap=¥400k, LOT=100)\n")
    _case_table(w, cA, ref=None, label="A")
    w(f"\n**skip 上位銘柄 (lot_conc_top5={cA['lot_conc_top5']:.1%})**:")
    for sym in cA.get("top5_skip_symbols", []):
        w(f"  - {sym}")
    w("")

    # ── CASE B ───────────────────────────────────────────────────────
    w("---\n## Case B: Virtual Fractional (cap=¥400k, lot=1株)\n")
    _case_table(w, cB, ref=cA, label="B")
    dlt = (cB["fillable_rate"] - cA["fillable_rate"]) * 100
    w(f"\n**Δfillable A→B (lot制約除去効果): {dlt:+.1f}pp**")
    if dlt >= 20:
        w(f"→ 単元制約が支配的 (LOT問題)")
    else:
        w(f"→ 単元制約の影響は限定的 (CAPITAL/PRICE問題)")
    w("")

    # ── CASE C ───────────────────────────────────────────────────────
    w("---\n## Case C: Price Cap (cap=¥400k, LOT=100, 最小単元≤¥400k)\n")
    _case_table(w, cC, ref=cA, label="C")
    alpha_c = cC["avg_calmar"] / max(0.001, cB["avg_calmar"])
    dlt_c   = (cC["fillable_rate"] - cA["fillable_rate"]) * 100
    w(f"\n**Calmar比 C/B: {alpha_c:.1%}  (採用基準: ≥95%)**")
    if alpha_c >= 0.95:
        w(f"→ 高単価銘柄への依存なし (高単価排除しても品質維持)")
    else:
        w(f"→ 高単価銘柄が alpha に必須 (除外でCalmar低下)")
    w(f"**Δfillable A→C: {dlt_c:+.1f}pp** (price_rejects={cC.get('n_price_reject',0)}件)")
    w("")

    # ── CASE D ───────────────────────────────────────────────────────
    w("---\n## Case D: Capital Sweep (LOT=100)\n")
    w("| cap(¥) | fillable | Δfill | calmar | ΔCAGR | P95_yen | elasticity |")
    w("|---|---|---|---|---|---|---|")
    prev_fill = cA["fillable_rate"]
    prev_cagr = cA["avg_cagr"]
    for i, s in enumerate(sweep_D):
        df = (s["fillable_rate"] - prev_fill) * 100 if i > 0 else 0.0
        dc = s["avg_cagr"] - prev_cagr          if i > 0 else 0.0
        dc_cap = (s["sl_cash_init"] - (sweep_D[i-1]["sl_cash_init"] if i>0 else s["sl_cash_init"])) / 100_000
        elas = f"{df/dc_cap:+.3f}" if i > 0 and dc_cap > 0 else "—"
        mark = "✅" if s["fillable_rate"] >= 0.90 else ""
        w(f"| ¥{int(s['sl_cash_init']):,} "
          f"| {s['fillable_rate']:.1%} {mark}"
          f"| {df:+.1f}pp "
          f"| {s['avg_calmar']:.3f} "
          f"| {dc:+.2f}pp "
          f"| {_yen(s['p95_dd_yen'])} "
          f"| {elas} |")
        prev_fill = s["fillable_rate"]
        prev_cagr = s["avg_cagr"]
    w(f"\n**avg_capital_elasticity: {avg_elasticity:+.4f} pp / ¥100k**")
    w(f"**min_efficient_capital: ¥{int(min_eff_cap):,}** (fillable≥90%を達成する最小資本)")
    w("")

    # ── CASE E ───────────────────────────────────────────────────────
    w("---\n## Case E: Capital Sweep + Price Cap\n")
    w("| cap(¥) | fillable | Δfill vs D | calmar | notes |")
    w("|---|---|---|---|---|")
    for i, (sD, sE) in enumerate(zip(sweep_D, sweep_E)):
        delta_de = (sE["fillable_rate"] - sD["fillable_rate"]) * 100
        mark = "✅" if sE["fillable_rate"] >= 0.90 else ""
        note = "price_cap isolates capital-only effect" if i == 0 else ""
        w(f"| ¥{int(sE['sl_cash_init']):,} "
          f"| {sE['fillable_rate']:.1%} {mark}"
          f"| {delta_de:+.1f}pp "
          f"| {sE['avg_calmar']:.3f} "
          f"| {note} |")

    # interpretation
    w("")
    # At same capital, does price cap raise or lower fillable vs Case D?
    deltas_DE = [(e["fillable_rate"]-d["fillable_rate"]) for d,e in zip(sweep_D, sweep_E)]
    avg_delta_DE = float(np.mean(deltas_DE)) * 100
    if avg_delta_DE > 5:
        w(f"**avg Δ(E-D)={avg_delta_DE:+.1f}pp**: 価格制約によりfillable上昇 → 高単価排除が有効")
        w("→ 高単価銘柄が単元スキップを引き起こしている (LOT + PRICE混合)")
    elif avg_delta_DE < -5:
        w(f"**avg Δ(E-D)={avg_delta_DE:+.1f}pp**: 価格制約によりfillable低下 → 高単価は必要")
        w("→ 価格制約を加えても改善しない (CAPITAL問題)")
    else:
        w(f"**avg Δ(E-D)={avg_delta_DE:+.1f}pp**: 価格制約の影響軽微")
    w("")

    # ── COUNTERFACTUAL ACCESS ─────────────────────────────────────────
    w("---\n## Counterfactual Access (スキップトレードの仮想収益)\n")
    if cf_df is not None and len(cf_df) > 0:
        avg_virt_ret  = cf_df["virtual_ret_pct"].mean()
        med_virt_ret  = cf_df["virtual_ret_pct"].median()
        pct_positive  = (cf_df["virtual_ret_pct"] > 0).mean()
        avg_hold      = cf_df["hold_days"].mean()
        w(f"n_skipped={len(cf_df)}件  avg_virtual_ret={avg_virt_ret:+.2f}%  "
          f"median={med_virt_ret:+.2f}%  win_rate={pct_positive:.1%}  "
          f"avg_hold={avg_hold:.1f}d\n")

        # lost_alpha = virt_return / actual_return
        avg_actual   = cA["avg_cagr"] if cA["avg_cagr"] != 0 else 1.0
        lost_alpha_ratio = abs(avg_virt_ret / 100) / (abs(avg_actual / 100) + 1e-9)
        w(f"**lost_alpha_from_skip** = {avg_virt_ret:+.2f}% / {avg_actual:+.2f}% = {lost_alpha_ratio:.3f}")

        if avg_virt_ret > 0:
            w(f"→ スキップされたトレードはプラス収益 — Lot制約が alpha を喪失させている")
        else:
            w(f"→ スキップされたトレードも損失傾向 — Lot制約の alpha 影響は軽微")

        w("\n**スキップ上位5件 (仮想リターン降順):**\n")
        top5 = cf_df.nlargest(5, "virtual_ret_pct")[
            ["symbol","skip_date","entry_price","virtual_ret_pct","hold_days","exit_reason"]
        ]
        w(top5.to_markdown(index=False))
    else:
        w("スキップトレードなし (または計算不可)\n")
    w("")

    # ── ATTRIBUTION SUMMARY ───────────────────────────────────────────
    w("---\n## 原因分離サマリ\n")
    w("| 要因 | Δfillable | 重要度 | 結論 |")
    w("|---|---|---|---|")

    # LOT contribution: Case B - Case A
    lot_delta = (cB["fillable_rate"] - cA["fillable_rate"]) * 100
    # PRICE contribution: among lot-constrained skips, price drives this
    price_delta = avg_delta_DE  # how much price cap changes fillable on top of capital
    # CAPITAL contribution: Case D at min_eff_cap vs 400k
    cap_delta_D = 0.0
    if sweep_D:
        at_min_eff = next((s for s in sweep_D if s["sl_cash_init"] == min_eff_cap), None)
        if at_min_eff:
            cap_delta_D = (at_min_eff["fillable_rate"] - cA["fillable_rate"]) * 100

    w(f"| LOT制約 (lot_size 100→1) | {lot_delta:+.1f}pp | {'高' if abs(lot_delta)>15 else '中' if abs(lot_delta)>5 else '低'} | 単元サイズが制限要因 |")
    w(f"| 高単価偏り (price_cap追加) | {price_delta:+.1f}pp | {'高' if abs(price_delta)>15 else '中' if abs(price_delta)>5 else '低'} | {'高単価銘柄が主因' if price_delta>5 else '高単価の影響軽微'} |")
    w(f"| 資本不足 (cap 400k→min_eff) | {cap_delta_D:+.1f}pp | {'高' if abs(cap_delta_D)>15 else '中' if abs(cap_delta_D)>5 else '低'} | 資本増加で解決可能 |")

    w(f"\n**root_cause確定: {root_cause}**\n")

    # ── FINAL VERDICT ─────────────────────────────────────────────────
    w("---\n## 最終判定\n")
    w(f"| 項目 | 値 |")
    w("|---|---|")
    w(f"| root_cause | **{root_cause}** |")
    w(f"| minimum_efficient_capital | ¥{int(min_eff_cap):,} |")
    w(f"| capital_confidence | {_assess_confidence(sweep_D, min_eff_cap)} |")
    w(f"| expected_fillable@min | "
      f"{next((s['fillable_rate'] for s in sweep_D if s['sl_cash_init']==min_eff_cap), 0):.1%} |")
    w(f"| expected_alpha@min | "
      f"{next((s['avg_calmar'] for s in sweep_D if s['sl_cash_init']==min_eff_cap), 0):.3f} Calmar |")
    w(f"| expected_DD@min | "
      f"{_yen(next((s['p95_dd_yen'] for s in sweep_D if s['sl_cash_init']==min_eff_cap), 0))} P95 |")
    w(f"| decision | **{decision}** |")
    w(f"| next_phase | {next_phase} |")
    w(f"| blocked_by | — |")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")
    print(f"  レポート: {path}")


def _case_table(w, c: dict, ref: Optional[dict], label: str):
    def delta(key, ref_c):
        if ref_c is None: return "—"
        d = c[key] - ref_c[key]
        if key in ("fillable_rate","win_rate","median_idle"): return f"{d*100:+.1f}pp"
        return f"{d:+.3f}"

    w(f"| 指標 | 値 | vs A |")
    w("|---|---|---|")
    w(f"| sl_cash_init | ¥{int(c['sl_cash_init']):,} | — |")
    w(f"| fillable_rate | {c['fillable_rate']:.1%} | {delta('fillable_rate',ref)} |")
    w(f"| forced_skip_rate | {c['forced_skip_rate']:.1%} | — |")
    w(f"| n_price_reject | {c['n_price_reject']} | — |")
    w(f"| avg_buy_price | ¥{int(c['avg_buy_price']):,} | — |")
    w(f"| avg_skip_price | ¥{int(c['avg_skip_price']):,} | — |")
    w(f"| avg_cagr (fold avg) | {c['avg_cagr']:+.2f}% | {c['avg_cagr']-(ref['avg_cagr'] if ref else 0):+.2f}pp |")
    w(f"| avg_calmar | {c['avg_calmar']:.3f} | {c['avg_calmar']-(ref['avg_calmar'] if ref else 0):+.3f} |")
    w(f"| P95_DD_yen | {_yen(c['p95_dd_yen'])} | — |")
    w(f"| fill_loss_avg | {c['fill_loss_avg']:.1%} | — |")
    w(f"| n_trades | {c['n_trades']} | — |")
    w(f"| win_rate | {c['win_rate']:.1%} | — |")


def _assess_confidence(sweep_D: list[dict], min_eff_cap: float) -> str:
    if not sweep_D: return "LOW"
    at_min = [s for s in sweep_D if s["sl_cash_init"] == min_eff_cap]
    if not at_min: return "LOW"
    m = at_min[0]
    if m["fillable_rate"] >= 0.95 and m["avg_calmar"] > 1.0:
        return "HIGH"
    if m["fillable_rate"] >= 0.90 and m["avg_calmar"] > 0.5:
        return "MEDIUM"
    return "LOW"


# ──────────────────────────────────────────────────────────────────────
#  CSV EXPORTS
# ──────────────────────────────────────────────────────────────────────

def export_csvs(cA, cB, cC, sweep_D, sweep_E, cf_df) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # capacity_cases.csv: one row per case
    cases_rows = []
    for lbl, c in [("A_baseline_400k", cA), ("B_virtualfrac_400k", cB), ("C_pricecap_400k", cC)]:
        r = {k: v for k, v in c.items() if isinstance(v, (int, float, str, bool))}
        r["case"] = lbl
        cases_rows.append(r)
    for s in sweep_D:
        r = {k: v for k, v in s.items() if isinstance(v, (int, float, str, bool))}
        r["case"] = f"D_sweep_{int(s['sl_cash_init'])//1000}k"
        cases_rows.append(r)
    for s in sweep_E:
        r = {k: v for k, v in s.items() if isinstance(v, (int, float, str, bool))}
        r["case"] = f"E_sweep_price_{int(s['sl_cash_init'])//1000}k"
        cases_rows.append(r)
    pd.DataFrame(cases_rows).to_csv(REPORTS_DIR / "capacity_cases.csv",
                                     index=False, encoding="utf-8-sig")
    print(f"  CSV: capacity_cases.csv ({len(cases_rows)} rows)")

    # fillability_curve.csv: capital vs fillable for Case D + E
    curve_rows = []
    for s in sweep_D:
        curve_rows.append({"case": "D", "capital": s["sl_cash_init"],
                            "fillable": s["fillable_rate"],
                            "calmar": s["avg_calmar"],
                            "p95_yen": s["p95_dd_yen"]})
    for s in sweep_E:
        curve_rows.append({"case": "E_pricecap", "capital": s["sl_cash_init"],
                            "fillable": s["fillable_rate"],
                            "calmar": s["avg_calmar"],
                            "p95_yen": s["p95_dd_yen"]})
    pd.DataFrame(curve_rows).to_csv(REPORTS_DIR / "fillability_curve.csv",
                                     index=False, encoding="utf-8-sig")
    print(f"  CSV: fillability_curve.csv")

    # counterfactual_access.csv
    if cf_df is not None and len(cf_df) > 0:
        cf_df.to_csv(REPORTS_DIR / "counterfactual_access.csv",
                     index=False, encoding="utf-8-sig")
        print(f"  CSV: counterfactual_access.csv ({len(cf_df)} rows)")


# ──────────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────────

def main() -> int:
    cfg = load_strategy_config()
    print("=" * 68)
    print("  Study15 Lot-Constrained Capacity Attribution Audit")
    print("  Strategy: Study9 Case B  (entry/exit/signal fixed)")
    print("=" * 68 + "\n")

    print("[1/8] データロード...")
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
    n_dates = len(common_dates)
    date_strs = [str(d.date()) for d in common_dates]
    idx_map = {s: i for i, s in enumerate(date_strs)}
    print(f"  共通日数: {n_dates}  銘柄数: {n_syms}\n")

    print("[2/8] マトリクス構築...")
    open_mat  = np.full((n_dates, n_syms), np.nan, dtype=np.float32)
    close_mat = np.full((n_dates, n_syms), np.nan, dtype=np.float32)
    for si, sym in enumerate(active_syms):
        df_src  = universe_raw[sym]["df"]
        row_idx = df_src.index.get_indexer(common_dates)
        if np.any(row_idx < 0): continue
        open_mat[:,  si] = df_src["Open"].to_numpy(dtype=np.float32)[row_idx]
        close_mat[:, si] = df_src["Close"].to_numpy(dtype=np.float32)[row_idx]

    rsr_mat = np.nan_to_num(
        _take(rsr_df, common_dates, active_syms, dtype=np.float32, fill_value=np.nan), nan=0.0)
    sym_active_mat = (None if sym_active_df is None else
                      _take(sym_active_df, common_dates, active_syms,
                            dtype=np.float32, fill_value=1.0))
    mkt_ret1 = None
    if topix_close is not None:
        mkt_ret1 = _take(topix_close.pct_change(), common_dates,
                         dtype=np.float32, fill_value=0.0)

    cross90_mat = compute_cross90_mat(rsr_mat)
    slope5_mat  = compute_slope5(rsr_mat)

    def run_and_metrics(cap: float, lot: int = LOT,
                        price_cap_val: Optional[float] = None,
                        label: str = "?") -> dict:
        print(f"  [{label}] cap=¥{int(cap):,}  lot={lot}"
              + (f"  price_cap=¥{int(price_cap_val):,}" if price_cap_val else ""))
        r = run_sim(
            open_mat, close_mat, rsr_mat, sym_active_mat,
            cross90_mat, slope5_mat, mkt_ret1,
            active_syms, sym_to_i, common_dates, n_dates,
            sl_cash_init=cap, lot_size=lot, price_cap=price_cap_val,
        )
        m = compute_metrics(r, cap, FOLDS, common_dates, n_dates)
        m["_raw"] = r
        print(f"      fillable={m['fillable_rate']:.1%}  "
              f"calmar={m['avg_calmar']:.3f}  "
              f"skip={m['n_skip']}  price_rej={m['n_price_reject']}")
        return m

    # ── Case A: Baseline ───────────────────────────────────────────
    print("[3/8] Case A: Baseline (cap=400k, LOT=100)...")
    cA = run_and_metrics(CAP_BASE, LOT, None, "A")

    # ── Case B: Virtual Fractional ─────────────────────────────────
    print("[4/8] Case B: Virtual Fractional (cap=400k, lot=1)...")
    cB = run_and_metrics(CAP_BASE, 1, None, "B")

    # ── Case C: Price Cap ─────────────────────────────────────────
    print("[5/8] Case C: Price Cap (cap=400k, filter min_lot<=400k)...")
    cC = run_and_metrics(CAP_BASE, LOT, float(CAP_BASE), "C")

    # ── Counterfactual access ──────────────────────────────────────
    print("[6/8] Counterfactual Access (skipped trades simulation)...")
    skip_log_a = cA["_raw"]["skip_log"]
    cf_df = counterfactual_access(
        skip_log_a, open_mat, close_mat, rsr_mat,
        sym_to_i, common_dates, n_dates, idx_map,
    )
    if len(cf_df) > 0:
        print(f"  n_skip_simulated={len(cf_df)}  "
              f"avg_virt_ret={cf_df['virtual_ret_pct'].mean():+.2f}%  "
              f"win_rate={(cf_df['virtual_ret_pct']>0).mean():.1%}")

    # ── Case D: Capital Sweep ──────────────────────────────────────
    print("[7/8] Case D: Capital Sweep...")
    sweep_D: list[dict] = []
    prev_fill = -1.0
    for cap in CAP_SWEEP:
        m = run_and_metrics(cap, LOT, None, f"D-{int(cap)//1000}k")
        sweep_D.append(m)
        if len(sweep_D) >= 2:
            delta = (m["fillable_rate"] - prev_fill) * 100
            if delta < SWEEP_STOP_DELTA and m["fillable_rate"] >= 0.90:
                print(f"  → Δfillable={delta:.1f}pp < {SWEEP_STOP_DELTA}pp 且つ fillable≥90%: SWEEP停止")
                break
        prev_fill = m["fillable_rate"]

    # ── Case E: Capital Sweep + Price Cap ─────────────────────────
    print("[8/8] Case E: Capital Sweep + Price Cap...")
    sweep_E: list[dict] = []
    for m_d in sweep_D:
        cap = m_d["sl_cash_init"]
        m = run_and_metrics(cap, LOT, float(CAP_BASE), f"E-{int(cap)//1000}k")
        sweep_E.append(m)

    # ── Output ────────────────────────────────────────────────────
    print("\n[OUT] レポート出力...")

    # strip _raw before CSV
    for c in [cA, cB, cC] + sweep_D + sweep_E:
        c.pop("_raw", None)

    write_report(
        cA, cB, cC, sweep_D, sweep_E, cf_df,
        REPORTS_DIR / "capacity_audit.md",
    )
    export_csvs(cA, cB, cC, sweep_D, sweep_E, cf_df)
    print("\n完了.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
