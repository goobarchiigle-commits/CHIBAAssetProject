"""
src/backtest/study20_limited_live_risk_envelope.py — Study20 Limited Live Risk Envelope

Validates realized losses remain within the intended R budget during live operation.

Authority  : LIMITED_LIVE
Strategy   : Study9 Case B (FROZEN)
Allocation : 10%
Capital    : ¥1,800,000 sleeve
max_notional: ¥1,200,000 (Study19 Case C GO_10PCT config)
Duration   : 30 trading days OR 10 complete trades (whichever first)

Profit optimization / strategy modification / entry-exit-sizing optimization: PROHIBITED
RNG_SEED = 20
"""
from __future__ import annotations

import csv
import json
import sys
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

from src.backtest.capital_allocation_abc import load_data, _take, LOT
from src.config_loader import load_strategy_config

# ── Fixed constants (PARAMS_LOCKED) ────────────────────────────────────
CAPITAL        = 1_800_000.0
MAX_NOTIONAL   = 1_200_000.0   # Study19 Case C
SLIPPAGE       = 0.001
COMMISSION     = 0.00055
COST           = SLIPPAGE + COMMISSION
R_UNIT_PCT     = 0.02           # 1R = 2% of entry notional
RNG_SEED       = 20
SIM_N_DAYS     = 30
TARGET_TRADES  = 10

# Stop thresholds (spec)
STOP_LOSS_R    = 2.5            # single-trade loss
STOP_GAP_R     = 2.0            # overnight gap

# Promotion thresholds
STUDY15B_P95_DD      = 140_000.0   # ¥ (Study18 reference: 50%×280k)
VIOLATION_RATE_MAX   = 0.10
R_BUDGET_PER_TRADE   = 1.0

# Restart injection days (day indices)
RESTART_DAYS   = [7, 17]

# Latency (log-normal)
LN_RST_MU = 9.90; LN_RST_S = 0.40   # p50 ≈ 20 s

# Study9 Case B FROZEN parameters
RSR_LO    = 92.0; RSR_HI = 95.0
D90_MIN   = 1;    D90_MAX = 5
SLOPE_MAX = 5.0
EXIT_THR  = 90.0; MIN_HOLD = 3
SHOCK_MKT = -0.05; SHOCK_SYM = -0.08

REPORT_DIR    = Path("reports")
LOG_DIR       = Path("logs")
REPORT_MD     = REPORT_DIR / "study20_risk_envelope.md"
REPORT_CSV    = REPORT_DIR / "study20_daily_metrics.csv"
TRADE_CSV     = REPORT_DIR / "study20_trade_log.csv"
TELEMETRY_LOG = LOG_DIR    / "study20_telemetry.jsonl"


# ── Dataclasses ────────────────────────────────────────────────────────
@dataclass
class TradeRecord:
    trade_id:           int
    symbol:             str
    entry_date:         str
    exit_date:          str
    entry_price:        float
    exit_price:         float
    fill_qty:           int
    hold_days:          int
    exit_reason:        str
    realized_pnl:       float   # ¥
    realized_R:         float
    gap_loss_pct_sum:   float   # total neg-gap % during hold
    gap_loss_R_sum:     float
    max_gap_pct:        float   # most negative single gap %
    mfe_pct:            float   # max favorable excursion %
    mae_pct:            float   # max adverse excursion %
    exit_efficiency:    float   # % vs MFE [0-100]
    lot_constrained:    bool
    loss_source:        str     # "" if winning trade
    R_budget_violation: bool    # realized_R < -1.0


@dataclass
class DayRecord:
    day_idx:      int
    date:         str
    restart:      bool
    n_signals:    int
    n_fills:      int
    pos_qty:      int
    cash:         float
    equity:       float
    dd_pct:       float
    restart_ms:   float = 0.0


@dataclass
class Study20Result:
    sim_start:           str   = ""
    sim_end:             str   = ""
    n_trading_days:      int   = 0
    n_complete_trades:   int   = 0
    n_winning_trades:    int   = 0
    n_losing_trades:     int   = 0

    # Primary risk metrics
    max_trade_loss_R:         float = 0.0
    P50_trade_loss_R:         float = 0.0
    P95_trade_loss_R:         float = 0.0
    R_budget_violation_rate:  float = 0.0
    rolling_DD_yen:           float = 0.0
    rolling_DD_pct:           float = 0.0

    # Secondary
    capital_activation_ratio: float = 0.0
    gap_loss_R_total:          float = 0.0
    max_single_gap_R:          float = 0.0
    overnight_loss_R_total:    float = 0.0
    exit_efficiency_avg:       float = 0.0

    loss_source_breakdown: Dict[str, int] = field(default_factory=dict)

    # Infrastructure
    cash_truth_gap:          int   = 0
    position_truth_gap:      int   = 0
    reconciliation_error:    int   = 0
    restart_restore_ms:      float = 0.0
    broker_disconnect_count: int   = 0
    unexpected_position:     int   = 0

    stop_condition_fired:    str  = ""
    promo_pass:              Dict[str, bool] = field(default_factory=dict)
    risk_conformance_score:  float = 0.0
    production_decision:     str  = ""


# ── Signal generation (Study9 Case B FROZEN) ───────────────────────────
def _cross90(r: np.ndarray) -> np.ndarray:
    n, m = r.shape
    out = np.zeros((n, m), dtype=np.int32)
    cnt = np.zeros(m, dtype=np.int32)
    for i in range(n):
        out[i] = cnt
        ab = r[i] >= 90.0
        cnt[ab] += 1
        cnt[~ab] = 0
    return out


def _slope5(r: np.ndarray) -> np.ndarray:
    out = np.zeros_like(r, dtype=np.float32)
    out[5:] = r[5:] - r[:-5]
    return out


def generate_signals(
    common_dates, active_syms, sym_to_i,
    open_mat, close_mat, rsr_mat, sym_active_mat, mkt_ret1,
) -> List[dict]:
    c90  = _cross90(rsr_mat)
    sl5  = _slope5(rsr_mat)
    n_dates = len(common_dates)

    class _Pos:
        def __init__(self, sym, qty, ep, ei):
            self.symbol = sym; self.qty = qty; self.ep = ep
            self.ei = ei; self.db = 0

    pos = None
    signals: List[dict] = []
    cash = CAPITAL

    for i, date in enumerate(common_dates):
        ds = str(date.date())
        if i + 1 >= n_dates:
            break
        nxt = i + 1
        mkt_shock = (mkt_ret1 is not None and float(mkt_ret1[i]) <= SHOCK_MKT)

        if pos is not None:
            si  = sym_to_i[pos.symbol]
            rv  = float(rsr_mat[i, si])
            ct  = float(close_mat[i, si])
            hd  = i - pos.ei
            do_exit = False; reason = ""
            if mkt_shock and i > 0:
                pc = float(close_mat[i - 1, si])
                if pc > 0 and (ct / pc - 1) <= SHOCK_SYM:
                    do_exit = True; reason = "MKT_SHOCK"
            if not do_exit:
                pos.db = (pos.db + 1) if rv < EXIT_THR else 0
                if pos.db >= 1 and hd >= MIN_HOLD:
                    do_exit = True; reason = "RSR_EXIT"
            if do_exit:
                sp = float(open_mat[nxt, si])
                cash += pos.qty * sp * (1 - COST)
                signals.append({
                    "type": "EXIT", "date": ds,
                    "symbol": pos.symbol, "price": round(sp, 2),
                    "qty": pos.qty, "hold_days": hd, "reason": reason,
                })
                pos = None

        if pos is None and not mkt_shock:
            cands = []
            for sym in active_syms:
                si   = sym_to_i[sym]
                rv   = float(rsr_mat[i, si])
                d90  = int(c90[i, si])
                sl5v = float(sl5[i, si])
                if not (RSR_LO <= rv < RSR_HI): continue
                if not (D90_MIN <= d90 <= D90_MAX): continue
                if sl5v > SLOPE_MAX: continue
                if sym_active_mat is not None and float(sym_active_mat[i, si]) < 0.5:
                    continue
                cands.append((rv, d90, sl5v, sym))
            if cands:
                cands.sort(key=lambda x: (-x[0], x[1]))
                _, _, _, sym = cands[0]
                si  = sym_to_i[sym]
                bp  = float(open_mat[nxt, si])
                if bp <= 0:
                    continue
                alloc = cash * 0.95
                qty   = int(alloc / bp / LOT) * LOT
                cost  = qty * bp * (1 + COST)
                if qty > 0 and cost <= cash:
                    cash -= cost
                    pos = _Pos(sym, qty, bp, nxt)
                    signals.append({
                        "type": "ENTRY", "date": ds,
                        "symbol": sym, "price": round(bp, 2),
                        "qty": qty, "hold_days": 0,
                    })
    return signals


# ── Main simulation ────────────────────────────────────────────────────
def run_simulation(
    all_signals: List[dict],
    common_dates: pd.DatetimeIndex,
    close_mat: np.ndarray,
    open_mat: np.ndarray,
    sym_to_i: Dict[str, int],
    rng: np.random.Generator,
) -> Tuple[Study20Result, List[TradeRecord], List[DayRecord]]:

    result = Study20Result()
    result.loss_source_breakdown = {
        "signal_failure": 0, "market_gap": 0,
        "execution_delay": 0, "lot_constraint": 0, "normal_variance": 0,
    }

    # Find anchor: first fillable entry
    entries = [s for s in all_signals if s["type"] == "ENTRY"]
    fillable = [s for s in entries if s["price"] * LOT <= MAX_NOTIONAL]
    if not fillable:
        result.stop_condition_fired = "NO_FILLABLE_SIGNAL"
        result.production_decision  = "ROLLBACK"
        return result, [], []

    anchor      = fillable[0]
    anchor_date = anchor["date"]
    anchor_idx  = next(
        (k for k, dt in enumerate(common_dates) if str(dt.date()) == anchor_date),
        None,
    )
    if anchor_idx is None:
        result.stop_condition_fired = "ANCHOR_DATE_NOT_IN_DATA"
        result.production_decision  = "ROLLBACK"
        return result, [], []

    date_to_gi = {str(dt.date()): k for k, dt in enumerate(common_dates)}

    end_idx  = min(len(common_dates), anchor_idx + SIM_N_DAYS)
    sim_dates = common_dates[anchor_idx:end_idx]
    result.sim_start      = str(sim_dates[0].date())
    result.sim_end        = str(sim_dates[-1].date())
    result.n_trading_days = len(sim_dates)

    sigs_by_date: Dict[str, List[dict]] = {}
    for s in all_signals:
        sigs_by_date.setdefault(s["date"], []).append(s)

    # Simulation state
    cash = CAPITAL; broker_cash = CAPITAL
    internal_pos: Dict[str, int] = {}
    broker_pos:   Dict[str, int] = {}

    current_trade: Optional[dict] = None   # active trade context

    trades:      List[TradeRecord] = []
    day_records: List[DayRecord]   = []
    equity_curve: List[float]      = []
    running_max_equity = CAPITAL
    total_notional     = 0.0
    restart_restore_ms = 0.0
    stop_fired         = False

    for day_idx, dt in enumerate(sim_dates):
        ds = str(dt.date())
        gi = date_to_gi.get(ds)
        if gi is None:
            continue

        # ── Restart injection ─────────────────────────────────────
        is_restart   = day_idx in RESTART_DAYS
        day_rst_ms   = 0.0
        if is_restart:
            day_rst_ms = float(rng.lognormal(LN_RST_MU, LN_RST_S))
            restart_restore_ms = max(restart_restore_ms, day_rst_ms)
            for sym in set(list(internal_pos) + list(broker_pos)):
                if internal_pos.get(sym, 0) != broker_pos.get(sym, 0):
                    result.position_truth_gap   += 1
                    result.reconciliation_error += 1
            if abs(cash - broker_cash) > 1.0:
                result.cash_truth_gap       += 1
                result.reconciliation_error += 1

        # ── Gap / MFE / MAE for held position ────────────────────
        day_gap_r = 0.0
        if current_trade is not None:
            sym = current_trade["symbol"]
            si  = sym_to_i.get(sym)
            if si is not None and gi > 0:
                prev_cl = float(close_mat[gi - 1, si])
                td_op   = float(open_mat[gi, si])
                td_cl   = float(close_mat[gi, si])
                ep      = current_trade["entry_price"]

                if prev_cl > 0 and td_op > 0 and not np.isnan(prev_cl) and not np.isnan(td_op):
                    gap_pct = (td_op / prev_cl - 1) * 100.0
                    current_trade["max_gap_pct"] = min(current_trade["max_gap_pct"], gap_pct)
                    if gap_pct < 0:
                        gap_r = abs(gap_pct / 100.0) / R_UNIT_PCT
                        current_trade["gap_loss_pct_sum"] += abs(gap_pct)
                        current_trade["gap_loss_R_sum"]   += gap_r
                        day_gap_r = gap_r
                        if gap_r > STOP_GAP_R and not stop_fired and not result.stop_condition_fired:
                            result.stop_condition_fired = "unexpected_gap"
                            stop_fired = True

                if ep > 0 and not np.isnan(td_cl) and td_cl > 0:
                    hold_ret = (td_cl / ep - 1) * 100.0
                    current_trade["mfe_pct"] = max(current_trade["mfe_pct"], hold_ret)
                    current_trade["mae_pct"] = min(current_trade["mae_pct"], hold_ret)

        # ── Process signals ───────────────────────────────────────
        day_sigs  = sigs_by_date.get(ds, [])
        day_fills = 0

        if not stop_fired:
            for sig in day_sigs:
                side = "BUY" if sig["type"] == "ENTRY" else "SELL"

                if side == "BUY" and internal_pos:
                    continue
                if side == "SELL" and not internal_pos:
                    continue

                if side == "BUY":
                    if sig["price"] * LOT > MAX_NOTIONAL:
                        continue
                    max_qty      = int(MAX_NOTIONAL / sig["price"] / LOT) * LOT
                    fill_qty     = min(sig["qty"], max(max_qty, LOT))
                    lot_constrained = fill_qty < sig["qty"]
                    fp   = sig["price"] * (1.0 + SLIPPAGE)
                    cost = fill_qty * fp * (1.0 + COMMISSION)
                    if cost > cash:
                        continue
                    cash        -= cost
                    broker_cash -= cost
                    internal_pos[sig["symbol"]] = fill_qty
                    broker_pos[sig["symbol"]]   = fill_qty
                    current_trade = {
                        "trade_id":        len(trades) + 1,
                        "symbol":          sig["symbol"],
                        "entry_date":      ds,
                        "entry_day_idx":   day_idx,
                        "entry_price":     fp,
                        "fill_qty":        fill_qty,
                        "lot_constrained": lot_constrained,
                        "gap_loss_pct_sum": 0.0,
                        "gap_loss_R_sum":   0.0,
                        "max_gap_pct":      0.0,
                        "mfe_pct":          0.0,
                        "mae_pct":          0.0,
                    }
                    total_notional += fill_qty * fp
                    day_fills      += 1

                else:  # SELL
                    sym  = sig["symbol"]
                    held = internal_pos.get(sym, 0)
                    if held <= 0:
                        continue
                    fp       = sig["price"] * (1.0 - SLIPPAGE)
                    proceeds = held * fp * (1.0 - COMMISSION)
                    cash        += proceeds
                    broker_cash += proceeds
                    internal_pos.pop(sym, None)
                    broker_pos.pop(sym, None)

                    ep            = current_trade["entry_price"]
                    entry_notional = ep * held
                    R_unit        = entry_notional * R_UNIT_PCT
                    cost_basis    = ep * held * (1.0 + COMMISSION)
                    pnl           = proceeds - cost_basis
                    realized_R    = pnl / max(R_unit, 1.0)
                    hold_days     = day_idx - current_trade["entry_day_idx"]

                    # Exit efficiency vs MFE
                    mfe = current_trade["mfe_pct"]
                    actual_ret = (fp / ep - 1.0) * 100.0
                    if mfe > 0.01:
                        exit_eff = min(100.0, max(0.0, actual_ret / mfe * 100.0))
                    else:
                        exit_eff = 100.0

                    # Loss attribution
                    loss_source = ""
                    if pnl < 0:
                        gap_r_sum = current_trade["gap_loss_R_sum"]
                        if gap_r_sum > abs(realized_R) * 0.5 and current_trade["max_gap_pct"] < -2.0:
                            loss_source = "market_gap"
                        elif abs(realized_R) >= 1.0:
                            loss_source = "signal_failure"
                        elif current_trade["lot_constrained"]:
                            loss_source = "lot_constraint"
                        else:
                            loss_source = "normal_variance"

                    R_violation = realized_R < -R_BUDGET_PER_TRADE
                    total_notional += held * fp
                    day_fills      += 1

                    # Stop: trade_loss > 2.5R
                    if realized_R < -STOP_LOSS_R and not stop_fired and not result.stop_condition_fired:
                        result.stop_condition_fired = "trade_loss_2_5R"
                        stop_fired = True

                    t = TradeRecord(
                        trade_id=current_trade["trade_id"],
                        symbol=sym,
                        entry_date=current_trade["entry_date"],
                        exit_date=ds,
                        entry_price=round(ep, 2),
                        exit_price=round(fp, 2),
                        fill_qty=held,
                        hold_days=hold_days,
                        exit_reason=sig.get("reason", "RSR_EXIT"),
                        realized_pnl=round(pnl, 2),
                        realized_R=round(realized_R, 4),
                        gap_loss_pct_sum=round(current_trade["gap_loss_pct_sum"], 4),
                        gap_loss_R_sum=round(current_trade["gap_loss_R_sum"], 4),
                        max_gap_pct=round(current_trade["max_gap_pct"], 4),
                        mfe_pct=round(mfe, 4),
                        mae_pct=round(current_trade["mae_pct"], 4),
                        exit_efficiency=round(exit_eff, 2),
                        lot_constrained=current_trade["lot_constrained"],
                        loss_source=loss_source,
                        R_budget_violation=R_violation,
                    )
                    trades.append(t)
                    if loss_source:
                        result.loss_source_breakdown[loss_source] = \
                            result.loss_source_breakdown.get(loss_source, 0) + 1
                    current_trade = None

        # ── Daily reconciliation ──────────────────────────────────
        for sym in set(list(internal_pos) + list(broker_pos)):
            if internal_pos.get(sym, 0) != broker_pos.get(sym, 0):
                result.position_truth_gap   += 1
                result.reconciliation_error += 1
        if abs(cash - broker_cash) > 1.0:
            result.cash_truth_gap       += 1
            result.reconciliation_error += 1

        # Trigger integrity stops
        if result.reconciliation_error > 0 and not result.stop_condition_fired:
            result.stop_condition_fired = "reconciliation_error"
            stop_fired = True
        if result.cash_truth_gap > 0 and not result.stop_condition_fired:
            result.stop_condition_fired = "cash_truth_gap"
            stop_fired = True
        if result.position_truth_gap > 0 and not result.stop_condition_fired:
            result.stop_condition_fired = "position_truth_gap"
            stop_fired = True

        # ── Portfolio equity ──────────────────────────────────────
        pos_value = 0.0
        for sym, qty in internal_pos.items():
            si = sym_to_i.get(sym)
            if si is not None and not np.isnan(close_mat[gi, si]):
                pos_value += qty * float(close_mat[gi, si])

        equity = cash + pos_value
        equity_curve.append(equity)
        running_max_equity = max(running_max_equity, equity)
        dd_pct = (equity / running_max_equity - 1.0) * 100.0 if running_max_equity > 0 else 0.0

        day_records.append(DayRecord(
            day_idx=day_idx, date=ds, restart=is_restart,
            n_signals=len(day_sigs), n_fills=day_fills,
            pos_qty=sum(internal_pos.values()),
            cash=round(cash, 0), equity=round(equity, 0),
            dd_pct=round(dd_pct, 4), restart_ms=round(day_rst_ms, 1),
        ))

        if len(trades) >= TARGET_TRADES:
            break

    # ── Post-simulation aggregation ───────────────────────────────────
    result.n_complete_trades = len(trades)
    result.n_winning_trades  = sum(1 for t in trades if t.realized_pnl > 0)
    result.n_losing_trades   = sum(1 for t in trades if t.realized_pnl <= 0)
    result.restart_restore_ms = round(restart_restore_ms, 1)

    if equity_curve:
        eq  = np.array(equity_curve)
        rmx = np.maximum.accumulate(eq)
        dd_yen = rmx - eq
        result.rolling_DD_yen = round(float(np.max(dd_yen)), 2)
        result.rolling_DD_pct = round(
            float(np.max(dd_yen / np.maximum(rmx, 1.0))) * 100.0, 4
        )

    R_vals = [t.realized_R for t in trades]
    if R_vals:
        loss_Rs = [r for r in R_vals if r < 0]
        result.max_trade_loss_R = round(min(R_vals), 4)
        if loss_Rs:
            result.P50_trade_loss_R = round(float(np.percentile(loss_Rs, 50)), 4)
            result.P95_trade_loss_R = round(float(np.percentile(loss_Rs, 95)), 4)
        n_viol = sum(1 for t in trades if t.R_budget_violation)
        result.R_budget_violation_rate = round(
            n_viol / max(1, len(trades)), 4
        )

    budget = result.n_trading_days * MAX_NOTIONAL
    result.capital_activation_ratio = round(
        total_notional / budget * 100.0 if budget > 0 else 0.0, 4
    )

    result.gap_loss_R_total     = round(sum(t.gap_loss_R_sum for t in trades), 4)
    result.overnight_loss_R_total = result.gap_loss_R_total
    gap_rs = [t.gap_loss_R_sum for t in trades if t.gap_loss_R_sum > 0]
    result.max_single_gap_R = round(max(gap_rs) if gap_rs else 0.0, 4)

    if trades:
        result.exit_efficiency_avg = round(
            float(np.mean([t.exit_efficiency for t in trades])), 2
        )

    result.promo_pass = {
        "R_budget_violation_rate_le10pct": result.R_budget_violation_rate <= VIOLATION_RATE_MAX,
        "rolling_DD_le_study15b_p95":      result.rolling_DD_yen <= STUDY15B_P95_DD,
        "cash_truth_gap_zero":             result.cash_truth_gap == 0,
        "position_truth_gap_zero":         result.position_truth_gap == 0,
        "reconciliation_error_zero":       result.reconciliation_error == 0,
        "unexpected_position_zero":        result.unexpected_position == 0,
    }

    result.risk_conformance_score = _compute_score(result)
    result.production_decision    = _decide(result)
    return result, trades, day_records


# ── Scoring ────────────────────────────────────────────────────────────
def _compute_score(r: Study20Result) -> float:
    # violation component (40 pt): 0% → 40, 20% → 0
    if r.n_complete_trades == 0:
        vr_score = 0.0
    else:
        vr = r.R_budget_violation_rate
        vr_score = max(0.0, 40.0 * (1.0 - vr / 0.20))

    # DD component (30 pt)
    if r.rolling_DD_yen <= STUDY15B_P95_DD:
        dd_score = 30.0
    else:
        dd_score = max(0.0, 30.0 * (1.0 - (r.rolling_DD_yen - STUDY15B_P95_DD) / STUDY15B_P95_DD))

    # Integrity component (20 pt): -5 per violation type
    n_int_violations = (
        (1 if r.cash_truth_gap > 0 else 0) +
        (1 if r.position_truth_gap > 0 else 0) +
        (1 if r.reconciliation_error > 0 else 0) +
        (1 if r.unexpected_position > 0 else 0)
    )
    integrity_score = max(0.0, 20.0 - 5.0 * n_int_violations)

    # Activation component (10 pt): ≥30% → 10, <10% → 0
    car = r.capital_activation_ratio
    if car >= 30.0:
        act_score = 10.0
    elif car >= 10.0:
        act_score = 10.0 * (car - 10.0) / 20.0
    else:
        act_score = 0.0

    return round(min(100.0, vr_score + dd_score + integrity_score + act_score), 2)


def _decide(r: Study20Result) -> str:
    # Hard integrity failures → rollback
    integrity_fail = (
        r.cash_truth_gap > 0 or r.position_truth_gap > 0 or
        r.reconciliation_error > 0 or r.unexpected_position > 0
    )
    if integrity_fail:
        return "ROLLBACK"
    if r.stop_condition_fired in ("cash_truth_gap", "position_truth_gap",
                                   "reconciliation_error", "unexpected_position"):
        return "ROLLBACK"

    # Trade-level stops
    if r.stop_condition_fired in ("trade_loss_2_5R", "unexpected_gap"):
        return "ROLLBACK"

    all_promo = all(r.promo_pass.values())
    score     = r.risk_conformance_score

    if score >= 90.0 and all_promo and r.n_complete_trades >= TARGET_TRADES:
        return "GO_SCALE"
    if score >= 70.0 and all_promo:
        return "GO_10PCT"
    if score >= 50.0 or r.n_complete_trades > 0:
        return "GO_HOLD"
    return "ROLLBACK"


# ── Report writers ─────────────────────────────────────────────────────
def write_daily_csv(recs: List[DayRecord]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if not recs:
        return
    with REPORT_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0].__dict__.keys()))
        w.writeheader()
        for r in recs:
            w.writerow(r.__dict__)
    print(f"[CSV] {REPORT_CSV}")


def write_trade_csv(trades: List[TradeRecord]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if not trades:
        return
    with TRADE_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(trades[0].__dict__.keys()))
        w.writeheader()
        for t in trades:
            w.writerow(t.__dict__)
    print(f"[CSV] {TRADE_CSV}")


def _tick(v: bool) -> str:
    return "✅ PASS" if v else "❌ FAIL"


def write_md_report(
    result: Study20Result,
    trades: List[TradeRecord],
    day_recs: List[DayRecord],
) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    a = lines.append
    today = "2026-06-21"

    a("# Study20 Limited Live Risk Envelope")
    a("")
    a(f"作成日: {today}  |  リスク検証のみ / 収益最適化禁止")
    a("")
    a("**Strategy**: Study9 Case B (固定)  **Capital**: ¥1,800,000  "
      "**max_notional**: ¥1,200,000  **Allocation**: 10%  **Authority**: LIMITED_LIVE")
    a("")
    a(f"**Period**: {result.sim_start} → {result.sim_end}  "
      f"({result.n_trading_days} trading days, {result.n_complete_trades} complete trades)")
    a("")
    a("---")
    a("## Executive Summary")
    a("")
    a("| 指標 | 値 | 閾値 | 判定 |")
    a("|---|---|---|---|")
    a(f"| **risk_conformance_score** | **{result.risk_conformance_score:.1f}/100** | — | — |")
    a(f"| max_trade_loss_R | {result.max_trade_loss_R:+.4f}R | < {STOP_LOSS_R}R | "
      f"{'✅' if result.max_trade_loss_R > -STOP_LOSS_R else '❌'} |")
    a(f"| P50_trade_loss_R | {result.P50_trade_loss_R:+.4f}R | — | — |")
    a(f"| P95_trade_loss_R | {result.P95_trade_loss_R:+.4f}R | — | — |")
    a(f"| R_budget_violation_rate | {result.R_budget_violation_rate*100:.1f}% | ≤10% | "
      f"{'✅' if result.R_budget_violation_rate <= VIOLATION_RATE_MAX else '❌'} |")
    a(f"| rolling_DD | ¥{result.rolling_DD_yen:,.0f} ({result.rolling_DD_pct:.2f}%) | "
      f"≤¥{STUDY15B_P95_DD:,.0f} | "
      f"{'✅' if result.rolling_DD_yen <= STUDY15B_P95_DD else '❌'} |")
    a(f"| capital_activation_ratio | {result.capital_activation_ratio:.2f}% | ≥30% | "
      f"{'✅' if result.capital_activation_ratio >= 30.0 else '⚠️'} |")
    a(f"| exit_efficiency_avg | {result.exit_efficiency_avg:.1f}% | — | — |")
    a(f"| gap_loss_R_total | {result.gap_loss_R_total:.4f}R | — | — |")
    a(f"| overnight_loss_R_total | {result.overnight_loss_R_total:.4f}R | — | — |")
    a(f"| n_complete_trades | {result.n_complete_trades} | ≥3 (有意) | "
      f"{'✅' if result.n_complete_trades >= 3 else '⚠️'} |")
    a(f"| n_winning / n_losing | {result.n_winning_trades} / {result.n_losing_trades} | — | — |")
    a("")
    a(f"**production_decision: {result.production_decision}**")
    a("")

    # Loss attribution
    a("---")
    a("## Risk Attribution (Loss Source)")
    a("")
    a("| loss_source | 件数 | 説明 |")
    a("|---|---|---|")
    src_desc = {
        "signal_failure":  "RSRシグナルが予測失敗（モメンタム崩壊）",
        "market_gap":      "ギャップダウン優位（overnight gap > 2%）",
        "execution_delay": "約定遅延によるスリッページ超過",
        "lot_constraint":  "最小ロット制約によるサイズ縮小",
        "normal_variance": "通常分散範囲内の損失（< 1R）",
    }
    total_losses = sum(result.loss_source_breakdown.values())
    for src, cnt in result.loss_source_breakdown.items():
        pct = cnt / max(1, total_losses) * 100 if total_losses > 0 else 0
        a(f"| {src} | {cnt} ({pct:.0f}%) | {src_desc.get(src, '')} |")
    a("")

    # Trade log
    a("---")
    a("## Trade Log")
    a("")
    if trades:
        a("| # | 銘柄 | Entry | Exit | 保有日 | Entry価格 | Exit価格 | PnL | R | violation | loss_source |")
        a("|---|---|---|---|---|---|---|---|---|---|---|")
        for t in trades:
            viol = "⚠️ YES" if t.R_budget_violation else "— NO"
            a(f"| {t.trade_id} | {t.symbol} | {t.entry_date} | {t.exit_date} | "
              f"{t.hold_days}d | ¥{t.entry_price:,.0f} | ¥{t.exit_price:,.0f} | "
              f"¥{t.realized_pnl:+,.0f} | {t.realized_R:+.4f}R | {viol} | "
              f"{t.loss_source or '—'} |")
        a("")
        a("### Exit Efficiency & Gap Metrics")
        a("")
        a("| # | 銘柄 | MFE% | MAE% | exit_eff% | gap_loss_R | max_gap% |")
        a("|---|---|---|---|---|---|---|")
        for t in trades:
            a(f"| {t.trade_id} | {t.symbol} | {t.mfe_pct:+.2f}% | {t.mae_pct:+.2f}% | "
              f"{t.exit_efficiency:.1f}% | {t.gap_loss_R_sum:.4f}R | {t.max_gap_pct:.2f}% |")
    else:
        a("(トレードなし — 30日ウィンドウ内にシグナル未発火)")
    a("")

    # Infrastructure
    a("---")
    a("## Infrastructure Monitoring")
    a("")
    a("| 指標 | 値 | 判定 |")
    a("|---|---|---|")
    a(f"| cash_truth_gap | {result.cash_truth_gap} | {'✅' if result.cash_truth_gap == 0 else '❌'} |")
    a(f"| position_truth_gap | {result.position_truth_gap} | {'✅' if result.position_truth_gap == 0 else '❌'} |")
    a(f"| reconciliation_error | {result.reconciliation_error} | {'✅' if result.reconciliation_error == 0 else '❌'} |")
    a(f"| restart_restore_ms | {result.restart_restore_ms:.0f}ms | {'✅' if result.restart_restore_ms < 60_000 else '❌'} |")
    a(f"| broker_disconnect_count | {result.broker_disconnect_count} | — |")
    a(f"| unexpected_position | {result.unexpected_position} | {'✅' if result.unexpected_position == 0 else '❌'} |")
    if result.stop_condition_fired:
        a(f"| **stop_condition_fired** | **{result.stop_condition_fired}** | ⚠️ |")
    else:
        a(f"| stop_condition_fired | — なし | ✅ |")
    a("")

    # Promotion criteria
    a("---")
    a("## Promotion Criteria")
    a("")
    promo_defs = {
        "R_budget_violation_rate_le10pct": "≤ 10%",
        "rolling_DD_le_study15b_p95":      f"≤ ¥{STUDY15B_P95_DD:,.0f}",
        "cash_truth_gap_zero":             "= 0",
        "position_truth_gap_zero":         "= 0",
        "reconciliation_error_zero":       "= 0",
        "unexpected_position_zero":        "= 0",
    }
    a("| 条件 | 要件 | 判定 |")
    a("|---|---|---|")
    for k, req in promo_defs.items():
        v = result.promo_pass.get(k, False)
        a(f"| {k} | {req} | {_tick(v)} |")
    n_pass = sum(result.promo_pass.values())
    n_tot  = len(result.promo_pass)
    a("")
    a(f"**昇格基準: {n_pass}/{n_tot} PASS**")
    a("")

    # Rollback note
    if result.production_decision == "ROLLBACK":
        a("---")
        a("## Rollback Action")
        a("")
        a("- allocation: 10% → **5%** (halved)")
        a("- authority: **LIMITED_LIVE (継続)**")
        a("- 調査後に再評価")
        a("")

    # Final decision
    a("---")
    a("## 最終判定")
    a("")
    a(f"| 指標 | 値 |")
    a(f"|---|---|")
    a(f"| risk_conformance_score | **{result.risk_conformance_score:.1f}/100** |")
    a(f"| max_trade_loss_R | {result.max_trade_loss_R:+.4f}R |")
    a(f"| R_budget_violation_rate | {result.R_budget_violation_rate*100:.1f}% |")
    a(f"| rolling_DD | ¥{result.rolling_DD_yen:,.0f} |")
    a(f"| loss_source_breakdown | {result.loss_source_breakdown} |")
    a(f"| **production_decision** | **{result.production_decision}** |")
    a("")

    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[MD] {REPORT_MD}")


def _append_telemetry(result: Study20Result, trades: List[TradeRecord]) -> None:
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        record = {
            "study":                    "study20",
            "sim_start":                result.sim_start,
            "sim_end":                  result.sim_end,
            "n_trading_days":           result.n_trading_days,
            "n_complete_trades":        result.n_complete_trades,
            "max_trade_loss_R":         result.max_trade_loss_R,
            "P50_trade_loss_R":         result.P50_trade_loss_R,
            "P95_trade_loss_R":         result.P95_trade_loss_R,
            "R_budget_violation_rate":  result.R_budget_violation_rate,
            "rolling_DD_yen":           result.rolling_DD_yen,
            "rolling_DD_pct":           result.rolling_DD_pct,
            "capital_activation_ratio": result.capital_activation_ratio,
            "exit_efficiency_avg":      result.exit_efficiency_avg,
            "gap_loss_R_total":         result.gap_loss_R_total,
            "risk_conformance_score":   result.risk_conformance_score,
            "loss_source_breakdown":    result.loss_source_breakdown,
            "stop_condition_fired":     result.stop_condition_fired,
            "production_decision":      result.production_decision,
            "trade_realized_R_list":    [t.realized_R for t in trades],
        }
        with TELEMETRY_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


# ── Main ───────────────────────────────────────────────────────────────
def main() -> None:
    print("[Study20] Limited Live Risk Envelope — 30-Day Risk Validation")
    print("=" * 68)

    print("[1/4] データロード中...")
    cfg = load_strategy_config()
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
        (common_dates >= pd.Timestamp("2018-01-01")) &
        (common_dates <= pd.Timestamp("2025-12-31"))
    ]
    print(f"[1/4] 共通日数={len(common_dates)}  銘柄={n_syms}")

    print("[2/4] 価格マトリクス構築...")
    n_dates = len(common_dates)
    open_mat  = np.full((n_dates, n_syms), np.nan, dtype=np.float32)
    close_mat = np.full((n_dates, n_syms), np.nan, dtype=np.float32)
    for si, sym in enumerate(active_syms):
        df_src = universe_raw[sym]["df"]
        ri = df_src.index.get_indexer(common_dates)
        valid = ri >= 0
        if valid.any():
            open_mat[valid, si]  = df_src["Open"].to_numpy(dtype=np.float32)[ri[valid]]
            close_mat[valid, si] = df_src["Close"].to_numpy(dtype=np.float32)[ri[valid]]

    rsr_mat = np.nan_to_num(
        _take(rsr_df, common_dates, active_syms, dtype=np.float32, fill_value=np.nan),
        nan=0.0,
    )
    sym_active_mat = (
        None if sym_active_df is None
        else _take(sym_active_df, common_dates, active_syms, dtype=np.float32, fill_value=1.0)
    )
    mkt_ret1 = (
        None if topix_close is None
        else _take(topix_close.pct_change(), common_dates, dtype=np.float32, fill_value=0.0)
    )

    print("[3/4] Study9 Case B シグナル生成 (FROZEN)...")
    all_signals = generate_signals(
        common_dates, active_syms, sym_to_i,
        open_mat, close_mat, rsr_mat, sym_active_mat, mkt_ret1,
    )
    n_entry = sum(1 for s in all_signals if s["type"] == "ENTRY")
    n_exit  = sum(1 for s in all_signals if s["type"] == "EXIT")
    print(f"[3/4] total={len(all_signals)}  entries={n_entry}  exits={n_exit}")

    print("[4/4] Risk Envelope シミュレーション実行 (30d / 10 trades)...")
    rng = np.random.default_rng(RNG_SEED)
    result, trades, day_records = run_simulation(
        all_signals, common_dates, close_mat, open_mat, sym_to_i, rng
    )

    print(f"  sim_period           = {result.sim_start} → {result.sim_end}")
    print(f"  n_complete_trades    = {result.n_complete_trades}")
    print(f"  max_trade_loss_R     = {result.max_trade_loss_R:+.4f}R")
    print(f"  rolling_DD           = ¥{result.rolling_DD_yen:,.0f} ({result.rolling_DD_pct:.2f}%)")
    print(f"  violation_rate       = {result.R_budget_violation_rate*100:.1f}%")
    print(f"  exit_efficiency_avg  = {result.exit_efficiency_avg:.1f}%")
    print(f"  loss_source          = {result.loss_source_breakdown}")
    print(f"  conformance_score    = {result.risk_conformance_score:.1f}/100")
    print(f"  stop_condition       = {result.stop_condition_fired or '— なし'}")
    print(f"  production_decision  = {result.production_decision}")

    write_daily_csv(day_records)
    write_trade_csv(trades)
    write_md_report(result, trades, day_records)
    _append_telemetry(result, trades)

    print()
    print("=" * 68)
    print("Study20 Limited Live Risk Envelope — FINAL RESULTS")
    print("=" * 68)
    print(f"risk_conformance_score  = {result.risk_conformance_score}")
    print(f"max_trade_loss_R        = {result.max_trade_loss_R:+.4f}R")
    print(f"P50_trade_loss_R        = {result.P50_trade_loss_R:+.4f}R")
    print(f"P95_trade_loss_R        = {result.P95_trade_loss_R:+.4f}R")
    print(f"R_budget_violation_rate = {result.R_budget_violation_rate*100:.1f}%")
    print(f"rolling_DD              = ¥{result.rolling_DD_yen:,.0f}")
    print(f"exit_efficiency_avg     = {result.exit_efficiency_avg:.1f}%")
    for src, cnt in result.loss_source_breakdown.items():
        if cnt > 0:
            print(f"  loss_source[{src}] = {cnt}")
    print(f"production_decision     = {result.production_decision}")
    print("=" * 68)


if __name__ == "__main__":
    main()
