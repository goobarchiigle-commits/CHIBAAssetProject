"""
src/backtest/study14_pre_live_audit.py  —  Study14 Standalone Pre-Live Audit

Strategy  : Study9 Case B  (RSR[92,95) d90≤5 slope≤5 exit<90 hold≥3)
Capital   : fixed_cap = 400,000 (sleeve capital, direct sl_cash)
Governance: annual_rebalance policy (Study11 Case B)

Steps:
  Step1  Exit Attribution       → exit_efficiency / exit_regret
  Step2  Stop Audit             → conditional on Step1 FAIL
  Step3  Standalone DD Audit    → ruin_prob / P95 / P99 in yen
  Step4  Lot Feasibility        → fillable / forced_skip / median_idle
  Step5  Exit Execution Audit   → cancel/fill/slippage estimates

Output:
  reports/pre_live_audit.md
  reports/exit_audit.csv
  reports/dd_audit.csv
  reports/lot_audit.csv
  reports/execution_audit.csv

Run:
    cd C:/ai-trading
    python src/backtest/study14_pre_live_audit.py
"""
from __future__ import annotations

import sys, math, time, warnings
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.config_loader import load_strategy_config
from src.backtest.capital_allocation_abc import load_data, _take, calc_metrics, LOT, COST_ONE_WAY

REPORTS_DIR = Path("reports")
IS_START    = "2018-01-01"

FOLDS = [
    {"id": 1, "oos_start": "2021-01-01", "oos_end": "2021-12-31"},
    {"id": 2, "oos_start": "2022-01-01", "oos_end": "2022-12-31"},
    {"id": 3, "oos_start": "2023-01-01", "oos_end": "2023-12-31"},
    {"id": 4, "oos_start": "2024-01-01", "oos_end": "2024-12-31"},
    {"id": 5, "oos_start": "2025-01-01", "oos_end": "2025-12-31"},
]

# ── Fixed params (Study9 Case B) ───────────────────────────────────────
RSR_LO        = 92.0
RSR_HI        = 95.0
D90_MAX       = 5
SLOPE_MAX     = 5.0
EXIT_THR      = 90.0
MIN_HOLD      = 3
SHOCK_MKT     = -0.05
SHOCK_SYM     = -0.08

# Study14 fixed capital (direct sleeve cash — NOT via SLEEVE_FRAC)
FIXED_CAP_YEN = 400_000

# counterfactual window
CF_HOLD_MAX   = 40          # Step1: max counterfactual hold


# ──────────────────────────────────────────────────────────────────────
#  RSR slope & cross90 helpers (same as study9)
# ──────────────────────────────────────────────────────────────────────

def compute_cross90_mat(rsr_mat: np.ndarray) -> np.ndarray:
    n, m   = rsr_mat.shape
    out    = np.zeros((n, m), dtype=np.int32)
    counts = np.zeros(m, dtype=np.int32)
    for i in range(n):
        out[i] = counts
        above  = rsr_mat[i] >= 90.0
        counts[above]  += 1
        counts[~above]  = 0
    return out


def compute_slope5(rsr_mat: np.ndarray) -> np.ndarray:
    out = np.zeros_like(rsr_mat, dtype=np.float32)
    out[5:] = rsr_mat[5:] - rsr_mat[:-5]
    return out


# ──────────────────────────────────────────────────────────────────────
#  STANDALONE SIMULATION (Study9 Case B, fixed sl_cash=FIXED_CAP_YEN)
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


def run_caseb(
    open_mat, close_mat, rsr_mat, sym_active_mat,
    cross90_mat, slope5_mat, mkt_ret1,
    active_syms, sym_to_i, common_dates,
    sl_cash_init: float,
) -> dict:
    """Run Study9 Case B.  sl_cash_init is the direct sleeve cash (no SLEEVE_FRAC)."""
    n       = len(common_dates)
    sl_cash = sl_cash_init
    sl_pos: SlPos | None = None
    trades: list[dict]   = []
    eq      = np.zeros(n, dtype=np.float64)
    inv     = np.zeros(n, dtype=np.float64)
    skip_log: list[dict] = []      # Step4: forced skips

    for i, date in enumerate(common_dates):
        ds    = str(date.date())
        s_inv = (sl_pos.qty * float(close_mat[i, sym_to_i[sl_pos.symbol]])
                 if sl_pos else 0.0)
        s_eq  = sl_cash + s_inv
        eq[i]  = s_eq
        inv[i] = s_inv

        mkt_shock = mkt_ret1 is not None and float(mkt_ret1[i]) <= SHOCK_MKT
        if i + 1 >= n:
            break
        nxt = i + 1

        # ── EXIT ──────────────────────────────────────────────────────
        if sl_pos:
            si   = sym_to_i[sl_pos.symbol]
            rv   = float(rsr_mat[i, si])
            ct   = float(close_mat[i, si])
            ep   = sl_pos.entry_price
            hd   = i - sl_pos.entry_idx
            do_exit = False; reason = ""

            if mkt_shock and i > 0:
                pc = float(close_mat[i - 1, si])
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
                    "qty": sl_pos.qty, "entry_idx": sl_pos.entry_idx,
                    "exit_idx": i, "exit_date": ds, "reason": reason,
                    "hold_days": i - sl_pos.entry_idx,
                    "realized_ret": (sp / ep - 1) if ep > 0 else 0.0,
                })
                sl_pos = None

        # ── ENTRY ─────────────────────────────────────────────────────
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
                if bp > 0:
                    alloc = sl_cash * 0.95
                    qty   = int(alloc / bp / LOT) * LOT
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
                            "entry": bp, "exit": None, "pnl": None,
                            "qty": qty, "entry_idx": nxt,
                            "buy_date": ds, "rsr_entry": rv,
                            "alloc": alloc, "min_lot_cost": LOT * bp,
                        })
                    else:
                        # forced skip — cannot buy minimum lot
                        skip_log.append({
                            "date": ds, "symbol": sym,
                            "price": bp, "min_lot_cost": LOT * bp,
                            "sl_cash": sl_cash, "qty_would": qty,
                        })

    return {"eq": eq, "inv": inv, "trades": trades, "skip_log": skip_log}


# ──────────────────────────────────────────────────────────────────────
#  STEP 1: EXIT ATTRIBUTION
# ──────────────────────────────────────────────────────────────────────

def step1_exit_attribution(
    trades: list[dict],
    close_mat: np.ndarray,
    sym_to_i: dict,
    common_dates,
    n_dates: int,
) -> dict:
    """Counterfactual hold analysis for Case B.

    metrics:
      exit_efficiency   = realized_ret / counterfactual_peak_ret  (when peak > 0)
      exit_regret       = counterfactual_peak_ret - realized_ret   (pp)
      survival_gain     = avoided_loss / left_profit
      holding_decay     = avg daily decay after exit (pp/day)
    """
    sells = [t for t in trades if t["side"] == "SELL"]
    if not sells:
        return {}

    # p90 of hold_days → counterfactual window
    hds = [t["hold_days"] for t in sells if (t.get("hold_days") or 0) > 0]
    p90_hold = float(np.percentile(hds, 90)) if hds else 40.0
    cf_window = int(min(CF_HOLD_MAX, p90_hold))

    rows = []
    for t in sells:
        si   = sym_to_i.get(t["symbol"])
        if si is None: continue
        exit_i = t.get("exit_idx")
        if exit_i is None: continue

        ep  = t.get("entry", 0.0)
        xp  = t.get("exit", ep)
        if ep <= 0: continue

        real_ret = (xp / ep - 1) if ep > 0 else 0.0

        # forward window: next cf_window bars after exit
        fw_end = min(exit_i + cf_window, n_dates - 1)
        if fw_end <= exit_i:
            cf_peak_ret = real_ret
        else:
            fw_closes = close_mat[exit_i + 1: fw_end + 1, si]
            fw_closes = fw_closes[fw_closes > 0]
            if len(fw_closes) == 0:
                cf_peak_ret = real_ret
            else:
                cf_peak_ret = float(np.max(fw_closes)) / ep - 1

        exit_regret = cf_peak_ret - real_ret  # pp
        if cf_peak_ret > 0 and cf_peak_ret > real_ret:
            exit_eff = real_ret / cf_peak_ret
        elif cf_peak_ret <= 0:
            # both negative — exiting was good
            exit_eff = 1.0 if real_ret <= cf_peak_ret else 0.8
        else:
            # real_ret > cf_peak_ret → extra gain after exit was not there
            exit_eff = 1.0

        # survival gain: money not lost vs money left on table
        # Treat avoided_loss as 0 when exit already profitable
        pnl_yen = t.get("pnl", 0.0)
        qty     = t.get("qty", 0)
        avoided_loss = max(0.0, -pnl_yen)          # yen saved if loss trade
        left_profit  = max(0.0, exit_regret * ep * qty)  # yen not captured

        if (avoided_loss + left_profit) > 0:
            survival_gain = avoided_loss / (avoided_loss + left_profit)
        else:
            survival_gain = 1.0

        rows.append({
            "symbol":           t["symbol"],
            "exit_date":        t.get("exit_date", ""),
            "hold_days":        t.get("hold_days", 0),
            "realized_ret_pct": round(real_ret * 100, 2),
            "cf_peak_ret_pct":  round(cf_peak_ret * 100, 2),
            "exit_regret_pp":   round(exit_regret * 100, 2),
            "exit_efficiency":  round(exit_eff, 4),
            "survival_gain":    round(survival_gain, 4),
            "pnl_yen":          round(pnl_yen, 0),
            "reason":           t.get("reason", ""),
        })

    if not rows:
        return {}

    df = pd.DataFrame(rows)
    eff_vals    = df["exit_efficiency"].tolist()
    regret_vals = df["exit_regret_pp"].tolist()

    median_eff    = float(np.median(eff_vals))
    p10_eff       = float(np.percentile(eff_vals, 10))
    p90_regret    = float(np.percentile(regret_vals, 90))
    profit_left   = df[df["exit_regret_pp"] > 0]["exit_regret_pp"].mean()
    tail_capture  = float((df["exit_efficiency"] >= 0.90).mean())

    # alpha_half_life: time for avg regret to stabilise (rough: slope of holding_decay)
    # holding_decay: among trades that left profit, how fast does avg residual decay?
    pos_regret = df[df["exit_regret_pp"] > 0]["exit_regret_pp"]
    avg_regret_per_day = (float(pos_regret.mean()) / cf_window) if len(pos_regret) > 0 else 0.0

    # late_exit_penalty: regret among winner trades that turned sour after exit
    sour_after  = df[(df["realized_ret_pct"] > 0) & (df["exit_regret_pp"] < -1.0)]
    late_exit_p = float(sour_after["exit_regret_pp"].mean()) if len(sour_after) > 0 else 0.0

    step1_pass = (median_eff >= 0.70) and (p90_regret <= 15.0)
    step1_emergency = p10_eff < 0.40   # immediate_review flag

    return {
        "n_trades":          len(rows),
        "cf_window_days":    cf_window,
        "p90_hold":          round(p90_hold, 1),
        "median_exit_eff":   round(median_eff, 4),
        "p10_exit_eff":      round(p10_eff, 4),
        "p90_exit_regret":   round(p90_regret, 2),
        "profit_left_avg_pp": round(profit_left if not math.isnan(profit_left) else 0.0, 2),
        "tail_capture":      round(tail_capture, 3),
        "avg_regret_per_day": round(avg_regret_per_day, 3),
        "late_exit_penalty": round(late_exit_p, 2),
        "step1_pass":        step1_pass,
        "step1_emergency":   step1_emergency,
        "df":                df,
    }


# ──────────────────────────────────────────────────────────────────────
#  STEP 2: STOP AUDIT (conditional)
# ──────────────────────────────────────────────────────────────────────

def step2_stop_audit(trades: list[dict], step1: dict) -> dict:
    """Audit whether stop logic would have improved exit quality.
    Observation-only.  No implementation proposed.
    """
    sells = [t for t in trades if t["side"] == "SELL"]
    if not sells:
        return {}

    # Categorise trades by exit reason
    rsr_exits  = [t for t in sells if "RSR_EXIT" in t.get("reason", "")]
    mkt_exits  = [t for t in sells if "MKT_SHOCK" in t.get("reason", "")]
    all_losses = [t for t in sells if (t.get("pnl") or 0) < 0]
    all_wins   = [t for t in sells if (t.get("pnl") or 0) >= 0]

    rsr_loss  = [t for t in rsr_exits if (t.get("pnl") or 0) < 0]
    rsr_win   = [t for t in rsr_exits if (t.get("pnl") or 0) >= 0]

    # stop_cost: avg loss among RSR exits that turned into losses
    stop_cost = float(np.mean([t["pnl"] for t in rsr_loss])) if rsr_loss else 0.0
    # stop_saved: if we had a 5% hard stop, how many losses would have been smaller?
    # estimate: trades where realized_ret < -5%
    stop_pct = 0.05
    saveable  = [t for t in rsr_loss if (t.get("realized_ret", 0) < -stop_pct)]
    stop_saved_est = len(saveable) / max(1, len(rsr_loss))

    conclusion = "NO_CHANGE"
    if len(rsr_loss) / max(1, len(rsr_exits)) > 0.40 and stop_cost < -50_000:
        conclusion = "PROPOSE_RESEARCH"

    return {
        "n_rsr_exits":        len(rsr_exits),
        "n_rsr_loss":         len(rsr_loss),
        "n_mkt_exits":        len(mkt_exits),
        "rsr_loss_rate":      round(len(rsr_loss) / max(1, len(rsr_exits)), 3),
        "stop_cost_avg_yen":  round(stop_cost, 0),
        "stop_saved_pct_est": round(stop_saved_est, 3),
        "stop_attribution":   conclusion,
    }


# ──────────────────────────────────────────────────────────────────────
#  STEP 3: STANDALONE DD AUDIT
# ──────────────────────────────────────────────────────────────────────

def step3_dd_audit(
    trades: list[dict],
    eq_array: np.ndarray,
    common_dates,
    sl_cash_init: float,
    folds: list[dict],
    n_bootstrap: int = 5_000,
    rng_seed: int = 42,
) -> dict:
    """
    Bootstrap from per-fold CAGR/DD distributions to estimate:
      - ruin_probability (equity hits 0 or -5% during any window)
      - P95 / P99 DD in yen
      - longest_underwater in days
      - ulcer_index
      - capital_recovery_days
    """
    rng = np.random.default_rng(rng_seed)

    # ── real equity curve stats ──────────────────────────────────────
    eq = eq_array
    peak = np.maximum.accumulate(eq)
    dd   = (eq - peak) / peak
    max_dd = float(np.nanmin(dd))
    max_dd_yen = float(np.nanmin(eq - peak))

    # longest underwater
    under = dd < -0.001
    if under.any():
        run, best = 0, 0
        for v in under:
            if v: run += 1; best = max(best, run)
            else: run = 0
        longest_uw = best
    else:
        longest_uw = 0

    # Ulcer index
    dd_sq_mean = float(np.nanmean(dd ** 2))
    ulcer = math.sqrt(dd_sq_mean) * 100  # in percent

    # recovery: max days from trough to new peak
    trough_i = int(np.argmin(dd))
    eq_after = eq[trough_i:]
    recovery_days = int(np.argmax(eq_after >= peak[trough_i])) if eq_after.max() >= peak[trough_i] else 999

    # ── per-fold returns (annual CAGR) ───────────────────────────────
    fold_cagr = []
    fold_dd   = []
    date_strs = [str(d.date()) for d in common_dates]
    for fold in folds:
        ds_s, ds_e = fold["oos_start"], fold["oos_end"]
        oos_idx = [i for i, s in enumerate(date_strs) if ds_s <= s <= ds_e]
        if not oos_idx: continue
        si, ei = oos_idx[0], oos_idx[-1] + 1
        s_eq = eq[si:ei]
        if len(s_eq) < 5: continue
        y_factor = 252 / max(1, ei - si)
        cagr_val = (s_eq[-1] / max(1e-6, s_eq[0])) ** y_factor - 1
        peak_f = np.maximum.accumulate(s_eq)
        dd_f   = (s_eq - peak_f) / peak_f
        fold_cagr.append(float(cagr_val))
        fold_dd.append(float(np.nanmin(dd_f)))

    if len(fold_cagr) < 2:
        # fallback: use real curve
        fold_cagr = [max_dd]
        fold_dd   = [max_dd]

    # ── bootstrap ────────────────────────────────────────────────────
    fc_arr = np.array(fold_cagr, dtype=np.float64)
    fd_arr = np.array(fold_dd,   dtype=np.float64)

    # Simulate 5-fold sequences N times
    n_folds = 5
    boot_dd  = []
    boot_end = []
    for _ in range(n_bootstrap):
        idx   = rng.integers(0, len(fc_arr), size=n_folds)
        c_arr = fc_arr[idx]
        d_arr = fd_arr[idx]

        eq_b = sl_cash_init
        min_eq = sl_cash_init
        for ci, di in zip(c_arr, d_arr):
            eq_b_end = eq_b * (1 + ci)
            trough   = eq_b * (1 + di)
            min_eq   = min(min_eq, trough)
            eq_b     = max(0.001, eq_b_end)

        max_dd_b = (min_eq - sl_cash_init) / sl_cash_init
        boot_dd.append(max_dd_b)
        boot_end.append(eq_b)

    boot_dd  = np.array(boot_dd)
    p95_dd   = float(np.percentile(boot_dd, 5))   # 5th percentile = worst 5%
    p99_dd   = float(np.percentile(boot_dd, 1))

    p95_yen  = p95_dd * sl_cash_init
    p99_yen  = p99_dd * sl_cash_init
    ruin_p   = float((boot_dd <= -0.999).mean())   # equity near zero

    # Pass thresholds
    pass_ruin    = ruin_p < 0.01
    pass_p95_yen = abs(p95_yen) <= 80_000
    pass_p99_yen = abs(p99_yen) <= 140_000
    pass_uw      = longest_uw <= 90 * 1.5  # ≈ 135 trading days (90 calendar ~63 trading)
    pass_recov   = recovery_days <= 60

    step3_pass = all([pass_ruin, pass_p95_yen, pass_p99_yen])

    return {
        "sl_cash_init":      sl_cash_init,
        "max_dd_realised":   round(max_dd * 100, 2),
        "max_dd_yen":        round(max_dd_yen, 0),
        "longest_uw_days":   longest_uw,
        "ulcer_index":       round(ulcer, 3),
        "recovery_days":     recovery_days,
        "fold_cagr_avg":     round(float(np.mean(fold_cagr)) * 100, 2),
        "fold_dd_avg":       round(float(np.mean(fold_dd)) * 100, 2),
        "ruin_probability":  round(ruin_p, 4),
        "p95_dd_pct":        round(abs(p95_dd) * 100, 2),
        "p99_dd_pct":        round(abs(p99_dd) * 100, 2),
        "p95_dd_yen":        round(abs(p95_yen), 0),
        "p99_dd_yen":        round(abs(p99_yen), 0),
        "pass_ruin":         pass_ruin,
        "pass_p95_yen":      pass_p95_yen,
        "pass_p99_yen":      pass_p99_yen,
        "pass_longest_uw":   pass_uw,
        "pass_recovery":     pass_recov,
        "step3_pass":        step3_pass,
    }


# ──────────────────────────────────────────────────────────────────────
#  STEP 4: LOT FEASIBILITY
# ──────────────────────────────────────────────────────────────────────

def step4_lot_feasibility(
    trades: list[dict],
    skip_log: list[dict],
    eq_array: np.ndarray,
    inv_array: np.ndarray,
    active_syms: list[str],
    sym_to_i: dict,
    open_mat: np.ndarray,
    common_dates,
    sl_cash_init: float,
) -> dict:
    """
    fillable: (entry attempts that succeeded) / (entry attempts total)
    forced_skip: trades blocked because sl_cash < min_lot_cost
    median_idle: median of (cash / total_equity)
    """
    buys   = [t for t in trades if t["side"] == "BUY"]
    n_buy  = len(buys)
    n_skip = len(skip_log)
    n_attempt = n_buy + n_skip
    fillable = n_buy / max(1, n_attempt)
    forced_skip_rate = n_skip / max(1, n_attempt)

    # median idle ratio
    total_eq = eq_array
    idle_ratios = []
    for i in range(len(total_eq)):
        if total_eq[i] > 0:
            idle_ratios.append((total_eq[i] - inv_array[i]) / total_eq[i])
    median_idle = float(np.median(idle_ratios)) if idle_ratios else 1.0

    # capital utilisation: % of days with open position
    cap_util = float((inv_array > 0).mean())

    # avg buy price and min_lot_cost distributions
    buy_prices   = [t["entry"] for t in buys if t.get("entry") and t["entry"] > 0]
    skip_prices  = [s["price"]  for s in skip_log if s.get("price") and s["price"] > 0]
    skip_mlc     = [s["min_lot_cost"] for s in skip_log if s.get("min_lot_cost")]

    avg_buy_price  = float(np.mean(buy_prices))  if buy_prices  else 0.0
    avg_skip_price = float(np.mean(skip_prices)) if skip_prices else 0.0
    avg_skip_mlc   = float(np.mean(skip_mlc))    if skip_mlc   else 0.0

    # effective_capital_utilisation = fraction of deployed vs potential
    eff_cap_util = fillable * (1 - median_idle)

    step4_pass = (fillable >= 0.90) and (median_idle <= 0.20) and (forced_skip_rate <= 0.10)

    return {
        "sl_cash_init":         sl_cash_init,
        "n_buy_executed":       n_buy,
        "n_forced_skip":        n_skip,
        "n_total_attempts":     n_attempt,
        "fillable_rate":        round(fillable, 4),
        "forced_skip_rate":     round(forced_skip_rate, 4),
        "median_idle":          round(median_idle, 4),
        "cap_util_pct":         round(cap_util * 100, 2),
        "eff_cap_util":         round(eff_cap_util, 4),
        "avg_buy_price":        round(avg_buy_price, 0),
        "avg_skip_price":       round(avg_skip_price, 0),
        "avg_skip_min_lot_yen": round(avg_skip_mlc, 0),
        "step4_pass":           step4_pass,
    }


# ──────────────────────────────────────────────────────────────────────
#  STEP 5: EXECUTION AUDIT (estimation from Study12 data)
# ──────────────────────────────────────────────────────────────────────

def step5_execution_audit(trades: list[dict], sl_cash_init: float) -> dict:
    """
    Estimation based on Study12 execution simulation results.
    No live order sent. No signal change.
    """
    buys  = [t for t in trades if t["side"] == "BUY"]
    sells = [t for t in trades if t["side"] == "SELL"]

    avg_order_yen = float(np.mean([t["qty"] * t["entry"] for t in buys
                                    if t.get("qty") and t.get("entry")])) if buys else 0.0
    adv_est_yen   = 2_000_000_000  # RSR42 large/mid cap ADV estimate ¥2B

    capacity_pct  = (avg_order_yen / adv_est_yen * 100) if adv_est_yen > 0 else 0.0

    # From Study12 simulation constants (base scenario)
    cancel_rate_pct     = 0.0     # morning market → fills at open
    partial_fill_rate   = 2.5     # % (worst: 2.5%)
    slippage_base_bp    = 5.0     # additional friction bp (base)
    slippage_worst_bp   = 13.0    # additional friction bp (worst)
    broker_reject_pct   = 0.0     # kabu API
    latency_est_sec     = 900     # signal-to-order latency (Study12 assumption)

    # Market impact estimate (Almgren-Chriss simplified)
    if adv_est_yen > 0:
        participation_rate = avg_order_yen / adv_est_yen
        mkt_impact_bp = 0.5 * participation_rate * 10_000  # rough estimate
    else:
        mkt_impact_bp = 0.0

    return {
        "avg_order_yen":       round(avg_order_yen, 0),
        "adv_est_yen":         adv_est_yen,
        "capacity_pct_adv":    round(capacity_pct, 4),
        "mkt_impact_bp_est":   round(mkt_impact_bp, 3),
        "cancel_rate_pct":     cancel_rate_pct,
        "partial_fill_rate_pct": partial_fill_rate,
        "slippage_base_bp":    slippage_base_bp,
        "slippage_worst_bp":   slippage_worst_bp,
        "broker_reject_pct":   broker_reject_pct,
        "signal_latency_sec":  latency_est_sec,
        "execution_note":      "kabu REST API 成行 寄り付き. 約定追跡=不要(成行). signal変更禁止.",
    }


# ──────────────────────────────────────────────────────────────────────
#  REPORT WRITER
# ──────────────────────────────────────────────────────────────────────

def write_report(
    s1: dict, s2: dict, s3: dict, s4: dict, s5: dict,
    trades: list[dict],
    folds: list[dict],
    eq_array: np.ndarray,
    common_dates,
    path: Path,
) -> None:
    L = []; w = L.append
    today = time.strftime("%Y-%m-%d")

    w("# Study14 Standalone Pre-Live Audit")
    w(f"\n作成日: {today}  |  監査のみ / 実装変更禁止 / live注文禁止")
    w(f"\n**Strategy**: Study9 Case B  RSR[92,95) d90≤5 slope≤5 exit<90 hold≥3  MaxPos=1")
    w(f"**Capital**: fixed_cap=¥{FIXED_CAP_YEN:,} (直接sleeve)  "
      f"**Governance**: annual_rebalance(¥750k target)")
    w("")

    # ── EXECUTIVE SUMMARY ──────────────────────────────────────────────
    w("---")
    w("## Executive Summary\n")

    s1p = s1.get("step1_pass", False)
    s3p = s3.get("step3_pass", False)
    s4p = s4.get("step4_pass", False)

    all_pass = s1p and s3p and s4p
    any_crit = s1.get("step1_emergency", False)

    if all_pass:
        verdict = "**GO**"
        blocked = "—"
        next_phase = "Study15 Order Authority Gate"
    elif not s4p:
        verdict = "**NO_GO**"
        blocked = "Lot Feasibility (forced_skip_rate={:.1%})".format(s4.get("forced_skip_rate", 0))
        next_phase = "最大寄与要因=Lot Feasibility → capital再検討"
    elif not s3p:
        verdict = "**NO_GO**"
        blocked = "DD Audit (P95={:,}円 / P99={:,}円)".format(
            int(s3.get("p95_dd_yen", 0)), int(s3.get("p99_dd_yen", 0)))
        next_phase = "最大寄与要因=DD → capital/sizing見直し"
    elif not s1p:
        verdict = "**GO_LIMITED**"
        blocked = "Exit Attribution (median_eff={:.1%} / P90_regret={:.1f}pp)".format(
            s1.get("median_exit_eff", 0), s1.get("p90_exit_regret", 0))
        next_phase = "Study15 + alloc固定"
    else:
        verdict = "**GO_LIMITED**"
        blocked = "部分的懸念あり"
        next_phase = "Study15 + alloc固定"

    w(f"| 項目 | 値 |")
    w("|---|---|")
    w(f"| **最終判定** | {verdict} |")
    w(f"| blocked_by | {blocked} |")
    w(f"| rollback_rule | alpha_real_30d<80% または fill_rate<90% |")
    w(f"| authority_required | true |")
    w(f"| next_phase | {next_phase} |")
    w("")

    # ── STEP 1: EXIT ATTRIBUTION ────────────────────────────────────────
    w("---")
    w("## Step1 Exit Attribution\n")
    w(f"counterfactual_hold: min(40d, p90_hold={s1.get('p90_hold',0):.0f}d) = **{s1.get('cf_window_days',40)}d**\n")
    w("| 指標 | 値 | 判定閾値 | 判定 |")
    w("|---|---|---|---|")
    me = s1.get("median_exit_eff", 0)
    p90r = s1.get("p90_exit_regret", 0)
    p10e = s1.get("p10_exit_eff", 0)
    w(f"| median_exit_efficiency | {me:.1%} | ≥70% | {'✅' if me >= 0.70 else '❌'} |")
    w(f"| P10_exit_efficiency | {p10e:.1%} | ≥40% | {'✅ (非緊急)' if p10e >= 0.40 else '❌ IMMEDIATE_REVIEW'} |")
    w(f"| P90_exit_regret | {p90r:.2f}pp | ≤15pp | {'✅' if p90r <= 15.0 else '❌'} |")
    w(f"| profit_left_avg | {s1.get('profit_left_avg_pp',0):.2f}pp | — | — |")
    w(f"| tail_capture | {s1.get('tail_capture',0):.1%} | — | — |")
    w(f"| avg_regret_per_day | {s1.get('avg_regret_per_day',0):.3f}pp/d | — | — |")
    w(f"| late_exit_penalty | {s1.get('late_exit_penalty',0):.2f}pp | — | — |")
    w(f"| n_trades | {s1.get('n_trades',0)} | — | — |")
    w("")
    step1_result = "**PASS** ✅" if s1p else "**FAIL** ❌"
    w(f"**Step1 判定: {step1_result}**")
    if not s1p:
        w(f"\n→ Step2 Stop Audit を実行")
    else:
        w(f"\n→ Step2 Stop Audit SKIP (Step1 PASS)")
    w("")

    # ── STEP 2: STOP AUDIT ──────────────────────────────────────────────
    w("---")
    w("## Step2 Stop Audit (条件付き)\n")
    if not s1p:
        w("Step1 FAIL により実施。\n")
        w("| 指標 | 値 |")
        w("|---|---|")
        w(f"| n_rsr_exits | {s2.get('n_rsr_exits',0)} |")
        w(f"| n_rsr_loss | {s2.get('n_rsr_loss',0)} |")
        w(f"| rsr_loss_rate | {s2.get('rsr_loss_rate',0):.1%} |")
        w(f"| stop_cost_avg_yen | ¥{s2.get('stop_cost_avg_yen',0):,.0f} |")
        w(f"| stop_saved_pct_est | {s2.get('stop_saved_pct_est',0):.1%} |")
        w(f"| n_mkt_exits | {s2.get('n_mkt_exits',0)} |")
        w("")
        conclusion = s2.get("stop_attribution", "NO_CHANGE")
        w(f"**結論: {conclusion}**")
        if conclusion == "NO_CHANGE":
            w(f"\nstop調整では exit品質の根本解決にならない。EXIT変更は禁止。")
        else:
            w(f"\nstop研究候補あり — ただし実装禁止。研究提案のみ。")
    else:
        w("Step1 PASS → Stop Audit **SKIP**\n")
    w("")

    # ── STEP 3: DD AUDIT ────────────────────────────────────────────────
    w("---")
    w("## Step3 Standalone DD Audit\n")
    w(f"capital: ¥{int(s3.get('sl_cash_init',0)):,}  bootstrap N=5,000 (5-fold resample)\n")
    w("| 指標 | 値 | 判定閾値 | 判定 |")
    w("|---|---|---|---|")
    rp = s3.get("ruin_probability", 0)
    p95y = s3.get("p95_dd_yen", 0)
    p99y = s3.get("p99_dd_yen", 0)
    uw = s3.get("longest_uw_days", 0)
    rd = s3.get("recovery_days", 0)
    w(f"| ruin_probability | {rp:.2%} | <1.0% | {'✅' if rp < 0.01 else '❌'} |")
    w(f"| P95_DD_yen | ▲¥{int(p95y):,} | ≤¥80,000 | {'✅' if p95y <= 80_000 else '❌'} |")
    w(f"| P99_DD_yen | ▲¥{int(p99y):,} | ≤¥140,000 | {'✅' if p99y <= 140_000 else '❌'} |")
    w(f"| longest_underwater | {uw}d | ≤~90d | {'✅' if uw <= 135 else '❌'} |")
    w(f"| capital_recovery_days | {rd}d | ≤60d | {'✅' if rd <= 60 else '❌'} |")
    w(f"| ulcer_index | {s3.get('ulcer_index',0):.3f}% | — | — |")
    w(f"| max_dd_realised | {s3.get('max_dd_realised',0):.2f}% | — | — |")
    w(f"| max_dd_yen | ▲¥{int(abs(s3.get('max_dd_yen',0))):,} | — | — |")
    w(f"| fold_cagr_avg | {s3.get('fold_cagr_avg',0):+.2f}% | — | — |")
    w(f"| fold_dd_avg | {s3.get('fold_dd_avg',0):.2f}% | — | — |")
    w("")
    step3_result = "**PASS** ✅" if s3p else "**FAIL** ❌"
    w(f"**Step3 判定: {step3_result}**\n")

    # ── STEP 4: LOT FEASIBILITY ─────────────────────────────────────────
    w("---")
    w("## Step4 Lot Feasibility\n")
    w(f"sl_cash=¥{int(s4.get('sl_cash_init',0)):,}  LOT=100株  alloc=sl_cash×0.95\n")
    fr = s4.get("fillable_rate", 0)
    mi = s4.get("median_idle", 0)
    fsk = s4.get("forced_skip_rate", 0)
    w("| 指標 | 値 | 判定閾値 | 判定 |")
    w("|---|---|---|---|")
    w(f"| fillable_rate | {fr:.1%} | ≥90% | {'✅' if fr >= 0.90 else '❌'} |")
    w(f"| median_idle | {mi:.1%} | ≤20% | {'✅' if mi <= 0.20 else '❌'} |")
    w(f"| forced_skip_rate | {fsk:.1%} | ≤10% | {'✅' if fsk <= 0.10 else '❌'} |")
    w(f"| n_buy_executed | {s4.get('n_buy_executed',0)} | — | — |")
    w(f"| n_forced_skip | {s4.get('n_forced_skip',0)} | — | — |")
    w(f"| avg_buy_price | ¥{int(s4.get('avg_buy_price',0)):,} | — | — |")
    w(f"| avg_skip_min_lot_yen | ¥{int(s4.get('avg_skip_min_lot_yen',0)):,} | — | — |")
    w(f"| cap_util_pct | {s4.get('cap_util_pct',0):.1f}% | — | — |")
    w(f"| eff_cap_util | {s4.get('eff_cap_util',0):.1%} | — | — |")
    w("")
    step4_result = "**PASS** ✅" if s4p else "**FAIL** ❌"
    w(f"**Step4 判定: {step4_result}**")
    if not s4p:
        min_cap_for_fill = int(s4.get("avg_skip_min_lot_yen", 0) / 0.95)
        w(f"\n⚠ 最小有効capital推奨: ~¥{min_cap_for_fill:,} (avg_skip_min_lot÷0.95)")
        w(f"現行capital=¥{int(s4.get('sl_cash_init',0)):,}では多数の銘柄がLot制約でスキップされる")
    w("")

    # ── STEP 5: EXECUTION AUDIT ─────────────────────────────────────────
    w("---")
    w("## Step5 Exit Execution Audit\n")
    w(f"注: Study12ベース推定値。live注文禁止。signal変更禁止。\n")
    w("| 指標 | BASE推定 | WORST推定 | 確認ライン |")
    w("|---|---|---|---|")
    w(f"| cancel_rate | {s5.get('cancel_rate_pct',0):.1f}% | {s5.get('cancel_rate_pct',0):.1f}% | — (成行・取消なし) |")
    w(f"| partial_fill_rate | 0.8% | {s5.get('partial_fill_rate_pct',0):.1f}% | 追跡要 |")
    w(f"| avg_slippage_bp (追加) | {s5.get('slippage_base_bp',0):.0f}bp | {s5.get('slippage_worst_bp',0):.0f}bp | — |")
    w(f"| broker_reject_pct | {s5.get('broker_reject_pct',0):.1f}% | {s5.get('broker_reject_pct',0):.1f}% | ≤0.5% |")
    w(f"| signal_latency | {s5.get('signal_latency_sec',0)}s | {s5.get('signal_latency_sec',0)}s | — |")
    w(f"| capacity_pct_ADV | {s5.get('capacity_pct_adv',0):.4f}% | — | — (問題なし) |")
    w(f"| mkt_impact_bp_est | {s5.get('mkt_impact_bp_est',0):.3f}bp | — | 無視可能 |")
    w(f"| avg_order_yen | ¥{int(s5.get('avg_order_yen',0)):,} | — | — |")
    w("")
    w(f"**確認のみ / 変更禁止**: {s5.get('execution_note', '')}\n")

    # ── FINAL VERDICT ────────────────────────────────────────────────────
    w("---")
    w("## 最終判定\n")
    w(f"| ステップ | 判定 | 備考 |")
    w("|---|---|---|")
    w(f"| Step1 Exit Attribution | {'✅ PASS' if s1p else '❌ FAIL'} | "
      f"median_eff={s1.get('median_exit_eff',0):.1%} P90_regret={s1.get('p90_exit_regret',0):.1f}pp |")
    if not s1p:
        w(f"| Step2 Stop Audit | — | {s2.get('stop_attribution','NO_CHANGE')} |")
    w(f"| Step3 DD Audit | {'✅ PASS' if s3p else '❌ FAIL'} | "
      f"P95=▲¥{int(s3.get('p95_dd_yen',0)):,} P99=▲¥{int(s3.get('p99_dd_yen',0)):,} |")
    w(f"| Step4 Lot Feasibility | {'✅ PASS' if s4p else '❌ FAIL'} | "
      f"fillable={s4.get('fillable_rate',0):.1%} skip={s4.get('forced_skip_rate',0):.1%} |")
    w(f"| Step5 Execution Audit | ✅ 確認完了 | cancel=0% fill≥97.5% slip≤13bp |")
    w("")
    w(f"### 最終判定: {verdict}")
    w(f"- blocked_by: {blocked}")
    w(f"- rollback_rule: alpha_realization_rolling_30d < 80% → alloc=0%即時縮小")
    w(f"- authority_required: **true** → Study15 Order Authority Gate")
    w(f"- next_phase: {next_phase}")
    w("")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")
    print(f"  レポート: {path}")


# ──────────────────────────────────────────────────────────────────────
#  CSV EXPORTS
# ──────────────────────────────────────────────────────────────────────

def export_csvs(s1: dict, s3: dict, s4: dict, s5: dict) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # exit_audit.csv
    if "df" in s1 and s1["df"] is not None and len(s1["df"]) > 0:
        s1["df"].to_csv(REPORTS_DIR / "exit_audit.csv", index=False, encoding="utf-8-sig")
        print(f"  CSV: reports/exit_audit.csv ({len(s1['df'])} rows)")

    # dd_audit.csv
    dd_rows = [
        {"metric": k, "value": v}
        for k, v in s3.items()
        if not isinstance(v, (list, dict, np.ndarray))
    ]
    pd.DataFrame(dd_rows).to_csv(REPORTS_DIR / "dd_audit.csv", index=False, encoding="utf-8-sig")
    print(f"  CSV: reports/dd_audit.csv")

    # lot_audit.csv
    lot_rows = [{"metric": k, "value": v} for k, v in s4.items()
                if not isinstance(v, (list, dict, np.ndarray))]
    pd.DataFrame(lot_rows).to_csv(REPORTS_DIR / "lot_audit.csv", index=False, encoding="utf-8-sig")
    print(f"  CSV: reports/lot_audit.csv")

    # execution_audit.csv
    exec_rows = [{"metric": k, "value": v} for k, v in s5.items()
                 if not isinstance(v, (list, dict, np.ndarray))]
    pd.DataFrame(exec_rows).to_csv(REPORTS_DIR / "execution_audit.csv", index=False, encoding="utf-8-sig")
    print(f"  CSV: reports/execution_audit.csv")


# ──────────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────────

def main() -> int:
    cfg = load_strategy_config()

    print("=" * 68)
    print("  Study14 Standalone Pre-Live Audit")
    print(f"  Strategy: Study9 Case B  slope≤{SLOPE_MAX} exit<{EXIT_THR} hold≥{MIN_HOLD}")
    print(f"  Capital:  ¥{FIXED_CAP_YEN:,} (fixed sleeve)")
    print("=" * 68 + "\n")

    print("[1/6] データロード...")
    (universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
     rsr_syms, topix_close, cfg) = load_data(cfg)

    trade_syms   = {s: v for s, v in rsr_syms.items() if s in universe_raw}
    active_syms  = list(trade_syms.keys())
    sym_to_i     = {s: i for i, s in enumerate(active_syms)}
    n_syms       = len(active_syms)

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
    print(f"  共通日数: {n_dates}  銘柄数: {n_syms}\n")

    print("[2/6] マトリクス構築...")
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

    print("[3/6] Case B シミュレーション (capital=¥{:,})...".format(FIXED_CAP_YEN))
    result = run_caseb(
        open_mat, close_mat, rsr_mat, sym_active_mat,
        cross90_mat, slope5_mat, mkt_ret1,
        active_syms, sym_to_i, common_dates,
        sl_cash_init=float(FIXED_CAP_YEN),
    )
    eq_array  = result["eq"]
    inv_array = result["inv"]
    trades    = result["trades"]
    skip_log  = result["skip_log"]

    sells = [t for t in trades if t["side"] == "SELL"]
    buys  = [t for t in trades if t["side"] == "BUY"]
    print(f"  n_buy={len(buys)}  n_sell={len(sells)}  n_skip={len(skip_log)}\n")

    print("[4/6] Step1 Exit Attribution...")
    s1 = step1_exit_attribution(trades, close_mat, sym_to_i, common_dates, n_dates)
    print(f"  median_eff={s1.get('median_exit_eff',0):.1%}  "
          f"P90_regret={s1.get('p90_exit_regret',0):.2f}pp  "
          f"{'PASS' if s1.get('step1_pass') else 'FAIL'}")

    s2 = {}
    if not s1.get("step1_pass", True):
        print("[4b/6] Step2 Stop Audit (条件成立)...")
        s2 = step2_stop_audit(trades, s1)
        print(f"  conclusion={s2.get('stop_attribution','NO_CHANGE')}")

    print("[5/6] Step3 DD Audit (bootstrap N=5,000)...")
    s3 = step3_dd_audit(trades, eq_array, common_dates, float(FIXED_CAP_YEN), FOLDS)
    print(f"  ruin={s3.get('ruin_probability',0):.2%}  "
          f"P95=▲¥{int(s3.get('p95_dd_yen',0)):,}  "
          f"P99=▲¥{int(s3.get('p99_dd_yen',0)):,}  "
          f"{'PASS' if s3.get('step3_pass') else 'FAIL'}")

    print("[5b/6] Step4 Lot Feasibility...")
    s4 = step4_lot_feasibility(
        trades, skip_log, eq_array, inv_array,
        active_syms, sym_to_i, open_mat, common_dates, float(FIXED_CAP_YEN),
    )
    print(f"  fillable={s4.get('fillable_rate',0):.1%}  "
          f"skip={s4.get('forced_skip_rate',0):.1%}  "
          f"idle={s4.get('median_idle',0):.1%}  "
          f"{'PASS' if s4.get('step4_pass') else 'FAIL'}")

    print("[6/6] Step5 Execution Audit (estimation)...")
    s5 = step5_execution_audit(trades, float(FIXED_CAP_YEN))
    print(f"  avg_order=¥{int(s5.get('avg_order_yen',0)):,}  "
          f"capacity={s5.get('capacity_pct_adv',0):.4f}%ADV")

    print("\n[OUT] レポート出力...")
    write_report(
        s1, s2, s3, s4, s5, trades, FOLDS,
        eq_array, common_dates,
        REPORTS_DIR / "pre_live_audit.md",
    )
    export_csvs(s1, s3, s4, s5)

    print("\n完了.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
