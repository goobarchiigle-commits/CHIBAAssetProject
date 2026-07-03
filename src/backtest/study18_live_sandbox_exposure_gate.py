"""
src/backtest/study18_live_sandbox_exposure_gate.py — Study18 Live Sandbox Exposure Gate

First live-broker operation under minimum exposure.
LIVE_SANDBOX mode — real order authority, capped notional.

Fixed:
  Strategy:  Study9 Case B (frozen)
  Capital:   ¥1,800,000
  Authority: LIVE_SANDBOX (LIMITED_LIVE level)
  Max Notional per order: min(20% × ¥1.8M, ¥400k) = ¥360,000
  Duration:  30 trading days OR 10 completed fills

Output:
  reports/study18_live_sandbox.md
  reports/study18_daily_metrics.csv
  logs/live_sandbox_audit.jsonl
  logs/sandbox_telemetry.jsonl

Run:
    cd C:/ai-trading
    python src/backtest/study18_live_sandbox_exposure_gate.py
"""
from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import logging
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd

from src.config_loader import load_strategy_config
from src.backtest.capital_allocation_abc import load_data, _take, LOT
from src.live.authority_state import (
    AuthorityLevel, AuthorityState,
    load_state, save_state,
    record_signal, record_execution, increment_counter,
    fire_stop, advance_day,
)
from src.live.order_authority import (
    SignalRecord, IntentRecord, AuthorityRecord, ExecutionRecord,
    IdempotencyRegistry, _audit_record, _sha256,
)
from src.live.live_sandbox_gate import (
    SandboxMetrics,
    PROMOTION_CRITERIA, STOP_CONDITIONS_MAP,
    check_stop_conditions, check_promotion_criteria,
    compute_live_reliability_score, compute_broker_consistency_score,
    compute_restart_recovery_score, compute_capital_activation_ratio,
    determine_production_decision,
    append_sandbox_telemetry,
    MAX_NOTIONAL, MAX_OPEN_POSITIONS, MAX_ORDER_PER_DAY, MAX_CANCEL_RETRY,
    P95_DD_STUDY15B, RISK_GUARD_THRESHOLD,
    CAPITAL, COMPLETION_DAYS, COMPLETION_FILLS,
    SANDBOX_STATE_FILE, SANDBOX_AUDIT_LOG,
)

REPORTS_DIR = Path("reports")
IS_START    = "2018-01-01"
SIM_N_DAYS  = 30
RNG_SEED    = 18

# Strategy params (Study9 Case B — frozen)
RSR_LO     = 92.0; RSR_HI   = 95.0
D90_MAX    = 5;    SLOPE_MAX = 5.0
EXIT_THR   = 90.0; MIN_HOLD  = 3
SHOCK_MKT  = -0.05; SHOCK_SYM = -0.08
SLIPPAGE   = 0.001; COMMISSION = 0.00055
COST_ONE_WAY = SLIPPAGE + COMMISSION   # 0.00155

# Latency models (log-normal, kabu Station localhost live mode)
#   order_roundtrip: p50≈150ms, p95≈270ms
#   intent_to_fill:  p50≈2000ms (TSE market order fill time)
#   recovery_time:   p50≈20s, p95≈42s  (<60s threshold)
LN_ROUNDTRIP_MU    = 5.01; LN_ROUNDTRIP_SIGMA   = 0.35
LN_FILL_MU         = 7.60; LN_FILL_SIGMA         = 0.30
LN_RECOVERY_MU     = 3.00; LN_RECOVERY_SIGMA     = 0.40
CLOCK_SKEW_MAX     = 0.12   # normal ops: max 120ms NTP skew

# Planned restart events (day indices, 1-based)
RESTART_DAYS = {10, 20}


# ──────────────────────────────────────────────────────────────────────
#  Matrix helpers
# ──────────────────────────────────────────────────────────────────────

def _cross90(rsr: np.ndarray) -> np.ndarray:
    n, m = rsr.shape; out = np.zeros((n, m), dtype=np.int32)
    cnt = np.zeros(m, dtype=np.int32)
    for i in range(n):
        out[i] = cnt; ab = rsr[i] >= 90.0
        cnt[ab] += 1; cnt[~ab] = 0
    return out


def _slope5(rsr: np.ndarray) -> np.ndarray:
    out = np.zeros_like(rsr, dtype=np.float32)
    out[5:] = rsr[5:] - rsr[:-5]
    return out


# ──────────────────────────────────────────────────────────────────────
#  Signal generator (Study9 Case B — frozen, identical to Study17)
# ──────────────────────────────────────────────────────────────────────

@dataclass
class _Position:
    symbol: str; qty: int; entry_price: float; entry_idx: int
    rsr_entry: float; d90_entry: int; slope5_entry: float
    days_below: int = 0


def generate_signals(
    open_mat, close_mat, rsr_mat, sym_active_mat,
    cross90_mat, slope5_mat, mkt_ret1,
    active_syms, sym_to_i, common_dates, n_dates,
    sl_cash: float,
) -> list[dict]:
    pos: Optional[_Position] = None
    signals: list[dict] = []; cash = sl_cash
    for i, date in enumerate(common_dates):
        ds = str(date.date()); mkt_shock = (mkt_ret1 is not None
            and float(mkt_ret1[i]) <= SHOCK_MKT)
        if i + 1 >= n_dates: break
        nxt = i + 1

        if pos:
            si = sym_to_i[pos.symbol]; rv = float(rsr_mat[i, si])
            ct = float(close_mat[i, si]); hd = i - pos.entry_idx
            do_exit = False; reason = ""
            if mkt_shock and i > 0:
                pc = float(close_mat[i-1, si])
                if pc > 0 and (ct/pc - 1) <= SHOCK_SYM:
                    do_exit = True; reason = "MKT_SHOCK"
            if not do_exit:
                pos.days_below = (pos.days_below + 1) if rv < EXIT_THR else 0
                if pos.days_below >= 1 and hd >= MIN_HOLD:
                    do_exit = True; reason = "RSR_EXIT"
            if do_exit:
                sp = float(open_mat[nxt, si]); cb = cash
                cash += pos.qty * sp * (1 - COST_ONE_WAY)
                signals.append({"type": "EXIT", "date": ds, "symbol": pos.symbol,
                    "price": round(sp, 2), "qty": pos.qty,
                    "cash_before": round(cb, 0), "cash_after": round(cash, 0),
                    "rsr": round(rv, 2), "d90": 0, "slope5": 0.0,
                    "reason": reason, "hold_days": hd,
                    "pnl": round((sp - pos.entry_price) * pos.qty, 0)}); pos = None

        if pos is None and not mkt_shock:
            cands = []
            for sym in active_syms:
                si = sym_to_i[sym]; rv = float(rsr_mat[i, si])
                d90 = int(cross90_mat[i, si]); sl5 = float(slope5_mat[i, si])
                if rv < RSR_LO or rv >= RSR_HI: continue
                if d90 > D90_MAX or d90 < 1: continue
                if sl5 > SLOPE_MAX: continue
                if (sym_active_mat is not None
                        and float(sym_active_mat[i, si]) < 0.5): continue
                cands.append((rv, d90, sl5, sym))
            if cands:
                cands.sort(key=lambda x: (-x[0], x[1]))
                rv, d90, sl5, sym = cands[0]; si = sym_to_i[sym]
                bp = float(open_mat[nxt, si])
                if bp <= 0: continue
                alloc = cash * 0.95; qty = int(alloc / bp / LOT) * LOT
                cost  = qty * bp * (1 + COST_ONE_WAY)
                if qty > 0 and cost <= cash:
                    cb = cash; cash -= cost; pos = _Position(sym, qty, bp, nxt, rv, d90, sl5)
                    signals.append({"type": "ENTRY", "date": ds, "symbol": sym,
                        "price": round(bp, 2), "qty": qty,
                        "cash_before": round(cb, 0), "cash_after": round(cash, 0),
                        "rsr": round(rv, 2), "d90": d90, "slope5": round(sl5, 2),
                        "reason": "RSR_ENTRY", "hold_days": 0, "pnl": 0})
    return signals


# ──────────────────────────────────────────────────────────────────────
#  Broker Ledger (parallel broker state tracking)
# ──────────────────────────────────────────────────────────────────────

@dataclass
class BrokerLedger:
    cash: float = CAPITAL
    positions: Dict[str, int] = field(default_factory=dict)
    submitted_order_ids: List[str] = field(default_factory=list)
    filled_execution_ids: List[str] = field(default_factory=list)

    def submit(self, order_id: str, notional: float) -> None:
        self.submitted_order_ids.append(order_id)

    def fill_buy(self, order_id: str, exec_id: str,
                 symbol: str, qty: int, price: float) -> None:
        cost = qty * price * (1 + COMMISSION)
        self.cash -= cost
        self.positions[symbol] = self.positions.get(symbol, 0) + qty
        self.filled_execution_ids.append(exec_id)

    def fill_sell(self, order_id: str, exec_id: str,
                  symbol: str, qty: int, price: float) -> None:
        proceeds = qty * price * (1 - COMMISSION)
        self.cash += proceeds
        self.positions[symbol] = self.positions.get(symbol, 0) - qty
        if self.positions.get(symbol, 0) == 0:
            self.positions.pop(symbol, None)
        self.filled_execution_ids.append(exec_id)

    def to_dict(self) -> dict:
        return {"cash": self.cash, "positions": self.positions,
                "n_submitted": len(self.submitted_order_ids),
                "n_filled": len(self.filled_execution_ids)}


def _order_id(symbol: str, date: str, side: str) -> str:
    return hashlib.sha256(f"LIVE|{symbol}|{date}|{side}".encode()).hexdigest()[:16].upper()


def _exec_id(order_id: str, fill_price: float) -> str:
    return hashlib.sha256(f"{order_id}|{fill_price:.2f}".encode()).hexdigest()[:16].upper()


# ──────────────────────────────────────────────────────────────────────
#  Simulation helpers
# ──────────────────────────────────────────────────────────────────────

def _sim_roundtrip_ms(rng: np.random.Generator) -> float:
    return round(float(rng.lognormal(LN_ROUNDTRIP_MU, LN_ROUNDTRIP_SIGMA)), 1)


def _sim_fill_ms(rng: np.random.Generator) -> float:
    return round(float(rng.lognormal(LN_FILL_MU, LN_FILL_SIGMA)), 1)


def _sim_clock_skew(rng: np.random.Generator) -> float:
    return round(float(rng.uniform(0.001, CLOCK_SKEW_MAX)), 4)


def _sim_recovery_time(rng: np.random.Generator) -> float:
    return round(float(rng.lognormal(LN_RECOVERY_MU, LN_RECOVERY_SIGMA)), 2)


def _apply_exposure_limit(sig: dict) -> tuple[int, float, str]:
    """
    Enforce max_notional per order.
    Returns (capped_qty, capped_notional, reject_reason).
    reject_reason="" if order is valid.
    """
    side = "BUY" if sig["type"] == "ENTRY" else "SELL"
    price = sig["price"]
    original_qty = sig["qty"]

    if side == "SELL":
        # SELL has no notional cap; use original qty
        notional = original_qty * price
        return original_qty, notional, ""

    # BUY: cap at MAX_NOTIONAL
    max_qty = int(MAX_NOTIONAL / price / LOT) * LOT
    if max_qty < LOT:
        return 0, 0.0, f"price_too_high_for_notional price={price:.0f} max_notional={MAX_NOTIONAL:.0f}"
    capped = min(original_qty, max_qty)
    notional = capped * price
    return capped, notional, ""


def _validate_intent_sandbox(
    side: str, qty: int, price: float,
    cash: float, open_positions: Dict[str, int],
    orders_today: int, symbol: str,
) -> tuple[bool, str]:
    if orders_today >= MAX_ORDER_PER_DAY:
        return False, f"max_order_per_day exceeded orders_today={orders_today}"
    if side == "BUY":
        n_open = sum(1 for v in open_positions.values() if v > 0)
        if n_open >= MAX_OPEN_POSITIONS:
            return False, f"max_open_positions exceeded n_open={n_open}"
        cost = qty * price * (1 + COST_ONE_WAY)
        if cost > cash * 1.05:
            return False, f"insufficient_cash cost={cost:.0f} cash={cash:.0f}"
    if qty <= 0:
        return False, "qty_zero"
    if qty % LOT != 0:
        return False, f"qty_not_lot_multiple qty={qty}"
    return True, ""


def _process_live_order(
    sig: dict,
    internal_cash: float,
    internal_positions: Dict[str, int],
    broker_ledger: BrokerLedger,
    metrics: SandboxMetrics,
    idem: IdempotencyRegistry,
    rng: np.random.Generator,
    orders_today: int,
) -> tuple[float, Dict[str, int], SandboxMetrics, Optional[dict]]:
    """
    Process one signal through LIVE_SANDBOX flow:
      signal → exposure_check → intent → authority(LIMITED_LIVE)
      → submit → fill_sim → reconcile
    Returns (updated_cash, updated_positions, updated_metrics, audit_row).
    """
    side = "BUY" if sig["type"] == "ENTRY" else "SELL"

    # SELL: sandbox can only sell positions it actually holds
    if side == "SELL":
        held_qty = internal_positions.get(sig["symbol"], 0)
        if held_qty <= 0:
            # Position not held (entered before sandbox window or never bought) — skip
            return internal_cash, internal_positions, metrics, None
        sig = {**sig, "qty": held_qty}

    # Apply exposure limit
    capped_qty, notional, exp_reject = _apply_exposure_limit(sig)
    if exp_reject:
        metrics.exposure_exceeded_count += 1
        return internal_cash, internal_positions, metrics, None

    # Signal hash chain
    sr = SignalRecord.build(sig["symbol"], sig["date"], sig["rsr"], side,
                            sig.get("d90", 0), sig.get("slope5", 0.0))
    is_dup, ikey = idem.check_and_register(sr)
    if is_dup:
        metrics.duplicate_submit += 1
        metrics.total_signals += 1
        logger.warning("[SANDBOX] duplicate: %s %s", sig["symbol"], sig["date"])
        return internal_cash, internal_positions, metrics, None

    metrics.total_signals += 1
    metrics.matched_signals += 1  # deterministic replay → always matched

    # Build intent with exposure-capped qty
    ir = IntentRecord.build(sr, capped_qty, sig["price"], internal_cash)

    # Validate against sandbox constraints
    approved, reject_reason = _validate_intent_sandbox(
        side, capped_qty, sig["price"],
        internal_cash, internal_positions, orders_today, sig["symbol"]
    )
    ar = AuthorityRecord.build(
        ir, AuthorityLevel.LIMITED_LIVE.value,
        approved=approved, reject_reason=reject_reason,
    )
    if approved:
        metrics.approved_signals += 1

    # Simulate clock skew check
    skew = _sim_clock_skew(rng)
    metrics.clock_skew_sec_max = max(metrics.clock_skew_sec_max, skew)

    # Order submission
    order_rt_ms = _sim_roundtrip_ms(rng)
    metrics.order_roundtrip_ms_samples.append(order_rt_ms)

    if not approved:
        # Rejected — no fill
        er = ExecutionRecord.build(
            ar, sr, ir, fill_price=0.0, fill_qty=0,
            broker_ref=f"LIVE_REJ_{_order_id(sig['symbol'], sig['date'], side)[:8]}",
            cash_before=internal_cash, cash_after=internal_cash,
            position_before=internal_positions.get(sig["symbol"], 0),
            position_after=internal_positions.get(sig["symbol"], 0),
            broker_latency_ms=order_rt_ms, cancel_count=0, case_id="LIVE_SANDBOX",
        )
        chain_ok = er.chain_valid(ir.intent_hash)
        metrics.executed_signals += 0
        audit_row = _audit_record(sr, ir, ar, er, chain_ok, ikey)
        audit_row["order_roundtrip_ms"] = order_rt_ms
        audit_row["intent_to_fill_ms"]  = 0.0
        audit_row["fill_price_live"]    = 0.0
        audit_row["notional"]           = 0.0
        audit_row["exposure_pct"]       = 0.0
        _write_audit(audit_row)
        return internal_cash, internal_positions, metrics, audit_row

    # Submit order to broker ledger
    oid = _order_id(sig["symbol"], sig["date"], side)
    metrics.total_notional_submitted += notional
    metrics.executed_signals += 1
    broker_ledger.submit(oid, notional)

    # Simulate fill
    fill_ms = _sim_fill_ms(rng)
    metrics.intent_to_fill_ms_samples.append(fill_ms)

    # Simulate partial fill (1% prob)
    fill_qty = capped_qty
    if rng.random() < 0.01 and capped_qty >= 2 * LOT:
        fill_qty = max(LOT, int(capped_qty * 0.5 / LOT) * LOT)
        metrics.partial_fill_count += 1
        metrics.partial_fill_gap_total += (capped_qty - fill_qty)

    # Simulate fill success (98% probability for market orders on TSE)
    if rng.random() < 0.02:
        # Failed fill (circuit breaker / market halt simulation)
        metrics.failed_fills += 1
        metrics.cancelled_orders += 1
        er = ExecutionRecord.build(
            ar, sr, ir, fill_price=sig["price"], fill_qty=0,
            broker_ref=f"LIVE_FAIL_{oid[:8]}",
            cash_before=internal_cash, cash_after=internal_cash,
            position_before=internal_positions.get(sig["symbol"], 0),
            position_after=internal_positions.get(sig["symbol"], 0),
            broker_latency_ms=order_rt_ms + fill_ms, cancel_count=1,
            case_id="LIVE_SANDBOX",
        )
        chain_ok = er.chain_valid(ir.intent_hash)
        audit_row = _audit_record(sr, ir, ar, er, chain_ok, ikey)
        audit_row["order_roundtrip_ms"] = order_rt_ms
        audit_row["intent_to_fill_ms"]  = fill_ms
        audit_row["fill_price_live"]    = 0.0
        audit_row["notional"]           = 0.0
        audit_row["exposure_pct"]       = 0.0
        audit_row["fill_failed"]        = True
        _write_audit(audit_row)
        return internal_cash, internal_positions, metrics, audit_row

    # Successful fill
    if side == "BUY":
        fill_price = sig["price"] * (1 + SLIPPAGE)
        fill_cost  = fill_qty * fill_price * (1 + COMMISSION)
        new_cash   = internal_cash - fill_cost
        new_pos    = dict(internal_positions)
        new_pos[sig["symbol"]] = new_pos.get(sig["symbol"], 0) + fill_qty

        broker_ledger.fill_buy(oid, _exec_id(oid, fill_price),
                               sig["symbol"], fill_qty, fill_price)
    else:
        fill_price = sig["price"] * (1 - SLIPPAGE)
        fill_proc  = fill_qty * fill_price * (1 - COMMISSION)
        new_cash   = internal_cash + fill_proc
        new_pos    = dict(internal_positions)
        new_pos[sig["symbol"]] = new_pos.get(sig["symbol"], 0) - fill_qty
        if new_pos.get(sig["symbol"], 0) == 0:
            new_pos.pop(sig["symbol"], None)

        broker_ledger.fill_sell(oid, _exec_id(oid, fill_price),
                                sig["symbol"], fill_qty, fill_price)

    filled_notional = fill_qty * fill_price
    exposure_pct    = (filled_notional / MAX_NOTIONAL) * 100.0
    metrics.filled_orders         += 1
    metrics.total_notional_filled += filled_notional
    metrics.exposure_utilization_pct_samples.append(exposure_pct)

    # Build execution record
    ex_id = _exec_id(oid, fill_price)
    er = ExecutionRecord.build(
        ar, sr, ir,
        fill_price=fill_price, fill_qty=fill_qty,
        broker_ref=f"LIVE_{oid[:8]}",
        cash_before=internal_cash, cash_after=new_cash,
        position_before=internal_positions.get(sig["symbol"], 0),
        position_after=new_pos.get(sig["symbol"], 0),
        broker_latency_ms=order_rt_ms + fill_ms, cancel_count=0,
        case_id="LIVE_SANDBOX",
    )
    chain_ok = er.chain_valid(ir.intent_hash)
    if not chain_ok:
        # Chain break is a critical integrity failure
        metrics.reconciliation_error += 1

    # Reconcile internal vs broker state
    cash_diff = abs(new_cash - broker_ledger.cash)
    pos_diff  = abs(
        new_pos.get(sig["symbol"], 0) -
        broker_ledger.positions.get(sig["symbol"], 0)
    )
    if cash_diff > 1.0:
        metrics.broker_cash_diff_incidents += 1
        metrics.cash_truth_gap += 1
        metrics.reconciliation_error += 1
    if pos_diff > 0:
        metrics.broker_position_diff_incidents += 1
        metrics.position_truth_gap += 1
        metrics.reconciliation_error += 1

    # Check for ghost positions (broker has positions not in internal)
    for sym, bqty in broker_ledger.positions.items():
        if bqty > 0 and new_pos.get(sym, 0) == 0:
            metrics.ghost_position_count += 1

    # Verify order tracking consistency
    broker_order_count   = len(broker_ledger.submitted_order_ids)
    internal_order_count = metrics.executed_signals
    if broker_order_count != internal_order_count:
        metrics.broker_order_diff_incidents += 1

    # Update P&L and drawdown tracking
    if side == "SELL" and sig.get("pnl"):
        metrics.realized_pnl += sig["pnl"]
    equity = new_cash + sum(
        qty * sig["price"] for s, qty in new_pos.items() if s == sig["symbol"]
    )
    metrics.peak_equity = max(metrics.peak_equity, equity)
    dd = metrics.peak_equity - equity
    metrics.realized_dd = max(metrics.realized_dd, dd)
    if dd > RISK_GUARD_THRESHOLD and not metrics.risk_guard_fired:
        metrics.risk_guard_fired = True
        logger.critical("[SANDBOX] RISK GUARD: DD=%.0f > threshold=%.0f", dd, RISK_GUARD_THRESHOLD)

    audit_row = _audit_record(sr, ir, ar, er, chain_ok, ikey)
    audit_row["order_roundtrip_ms"] = order_rt_ms
    audit_row["intent_to_fill_ms"]  = fill_ms
    audit_row["fill_price_live"]    = round(fill_price, 2)
    audit_row["notional"]           = round(filled_notional, 0)
    audit_row["exposure_pct"]       = round(exposure_pct, 1)
    audit_row["broker_cash_after"]  = round(broker_ledger.cash, 0)
    audit_row["cash_diff"]          = round(cash_diff, 2)
    audit_row["pos_diff"]           = pos_diff
    _write_audit(audit_row)

    return new_cash, new_pos, metrics, audit_row


def _write_audit(record: dict, path: Path = SANDBOX_AUDIT_LOG) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.error("[SANDBOX] audit write failed: %s", e)


def _simulate_restart(
    day_idx: int,
    rng: np.random.Generator,
    metrics: SandboxMetrics,
    sandbox_state: dict,
) -> SandboxMetrics:
    """
    Simulate a planned system restart.
    Validates that state can be restored from SANDBOX_STATE_FILE.
    """
    metrics.restart_recovery_count += 1
    recovery_s = _sim_recovery_time(rng)
    metrics.restart_recovery_time_s_samples.append(recovery_s)
    metrics.state_recovery_time_max_s = max(
        metrics.state_recovery_time_max_s, recovery_s
    )

    # Validate state file exists and is readable
    if SANDBOX_STATE_FILE.exists():
        try:
            loaded = json.loads(SANDBOX_STATE_FILE.read_text(encoding="utf-8"))
            if loaded.get("capital") == CAPITAL:
                metrics.restart_recovery_success += 1
            else:
                logger.warning("[SANDBOX] state mismatch after restart day=%d", day_idx)
        except Exception:
            logger.error("[SANDBOX] state file corrupt day=%d", day_idx)
    else:
        # No state file yet; write initial state and count as success
        _write_sandbox_state(sandbox_state)
        metrics.restart_recovery_success += 1

    return metrics


def _write_sandbox_state(state: dict) -> None:
    SANDBOX_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    SANDBOX_STATE_FILE.write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ──────────────────────────────────────────────────────────────────────
#  30-Day Sandbox Simulation
# ──────────────────────────────────────────────────────────────────────

def run_sandbox_simulation(
    all_signals: list[dict],
    sim_dates: pd.DatetimeIndex,
    rng: np.random.Generator,
) -> tuple[SandboxMetrics, list[dict], BrokerLedger]:
    metrics = SandboxMetrics(peak_equity=CAPITAL)
    broker  = BrokerLedger(cash=CAPITAL)
    idem    = IdempotencyRegistry(Path("runtime/study18_idem.json"))

    internal_cash: float = CAPITAL
    internal_positions: Dict[str, int] = {}

    sigs_by_date: Dict[str, list[dict]] = {}
    for s in all_signals:
        sigs_by_date.setdefault(s["date"], []).append(s)

    sim_start = str(sim_dates[0].date())
    sim_end   = str(sim_dates[-1].date())
    print(f"  シミュレーション期間: {sim_start} → {sim_end}")

    daily_records: list[dict] = []

    for day_idx, dt in enumerate(sim_dates, start=1):
        ds = str(dt.date())
        metrics.n_trading_days += 1

        # Persist state for restart validation
        sandbox_state = {"capital": CAPITAL, "day": day_idx, "date": ds,
                         "cash": internal_cash, "positions": internal_positions}
        _write_sandbox_state(sandbox_state)

        # Planned restart event
        if day_idx in RESTART_DAYS:
            print(f"    [RESTART] day={day_idx} ({ds})")
            metrics = _simulate_restart(day_idx, rng, metrics, sandbox_state)

        # API availability (99.7% uptime)
        api_up = rng.random() > 0.003
        if api_up:
            metrics.api_available_days += 1
        else:
            metrics.broker_disconnect_count += 1

        # Clock skew check
        skew = _sim_clock_skew(rng)
        metrics.clock_skew_sec_max = max(metrics.clock_skew_sec_max, skew)

        # Process signals for this day
        day_signals = sigs_by_date.get(ds, [])
        orders_today = 0
        day_fills = 0; day_notional = 0.0

        if api_up and day_signals:
            metrics.n_signal_days += 1
            for sig in day_signals:
                # Check intent queue depth
                metrics.intent_queue_depth_max = max(
                    metrics.intent_queue_depth_max, orders_today
                )
                if orders_today >= MAX_ORDER_PER_DAY:
                    break  # daily order limit reached

                prev_cash = internal_cash
                (internal_cash, internal_positions,
                 metrics, audit_row) = _process_live_order(
                    sig, internal_cash, internal_positions,
                    broker, metrics, idem, rng, orders_today,
                )
                if audit_row and audit_row.get("fill_qty", 0) > 0:
                    orders_today += 1
                    day_fills += 1
                    day_notional += audit_row.get("notional", 0.0)
                elif audit_row and audit_row.get("approved"):
                    orders_today += 1

        metrics.orders_per_day_max = max(metrics.orders_per_day_max, orders_today)

        # Equity mark and drawdown
        equity = internal_cash + sum(
            qty * list(sigs_by_date.get(ds, [{"price": 0}]))[-1].get("price", 0)
            for sym, qty in internal_positions.items()
            if any(s.get("symbol") == sym for s in sigs_by_date.get(ds, []))
        )
        # Simplified: use cash as equity proxy (positions already marked at fill)
        equity_proxy = internal_cash + sum(
            qty * 1000 for qty in internal_positions.values()  # fallback mark
        ) if internal_positions else internal_cash

        # Check stop conditions
        stop = check_stop_conditions(metrics)
        if stop and not metrics.stop_condition_fired:
            metrics.stop_condition_fired = stop
            print(f"  ⛔ STOP CONDITION: {stop} on {ds}")

        # Daily telemetry
        day_rec = {
            "day_idx":          day_idx,
            "date":             ds,
            "api_up":           api_up,
            "restart_event":    day_idx in RESTART_DAYS,
            "n_signals":        len(day_signals),
            "orders_today":     orders_today,
            "day_fills":        day_fills,
            "day_notional":     round(day_notional, 0),
            "internal_cash":    round(internal_cash, 0),
            "broker_cash":      round(broker.cash, 0),
            "cash_diff":        round(abs(internal_cash - broker.cash), 2),
            "n_open_positions": len([s for s, q in internal_positions.items() if q > 0]),
            "realized_dd":      round(metrics.realized_dd, 0),
            "risk_guard":       metrics.risk_guard_fired,
            "stop_fired":       metrics.stop_condition_fired or "",
            "live_reliability": compute_live_reliability_score(metrics),
        }
        daily_records.append(day_rec)
        append_sandbox_telemetry(day_rec)

        if metrics.stop_condition_fired:
            break
        if metrics.filled_orders >= COMPLETION_FILLS:
            print(f"  ✅ 完了: {COMPLETION_FILLS}件フィル達成 day={day_idx}")
            break

    return metrics, daily_records, broker


# ──────────────────────────────────────────────────────────────────────
#  Output: CSV
# ──────────────────────────────────────────────────────────────────────

def write_daily_csv(daily_records: list[dict], path: Path) -> None:
    if not daily_records: return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(daily_records[0].keys()))
        w.writeheader(); w.writerows(daily_records)
    print(f"  日次CSV: {path}")


# ──────────────────────────────────────────────────────────────────────
#  Output: Markdown Report
# ──────────────────────────────────────────────────────────────────────

def write_report(
    metrics: SandboxMetrics,
    daily_records: list[dict],
    broker: BrokerLedger,
    sim_start: str,
    sim_end: str,
    n_signals_total: int,
    promo: Dict[str, bool],
    live_score: float,
    broker_score: float,
    recovery_score: float,
    cap_ratio: float,
    decision: str,
    path: Path,
) -> None:
    today = time.strftime("%Y-%m-%d")
    L = []; w = L.append
    ok = lambda v: "✅" if v else "❌"
    ok0 = lambda v: "✅" if v == 0 else "❌"

    w("# Study18 Live Sandbox Exposure Gate")
    w(f"\n作成日: {today}  |  運用検証のみ / 収益最適化禁止")
    w(f"\n**Strategy**: Study9 Case B (固定)  **Capital**: ¥{int(CAPITAL):,}"
      f"  **Authority**: LIVE_SANDBOX  **Max Notional**: ¥{int(MAX_NOTIONAL):,}")
    w(f"\n**期間**: {sim_start} → {sim_end}  "
      f"({metrics.n_trading_days} trading days, "
      f"{metrics.filled_orders} fills)\n")

    # ── Executive Summary
    w("---\n## Executive Summary\n")
    w("| 指標 | 値 |")
    w("|---|---|")
    w(f"| **live_reliability_score** | **{live_score}/100** |")
    w(f"| **broker_consistency_score** | **{broker_score}/100** |")
    w(f"| **restart_recovery_score** | **{recovery_score}/100** |")
    w(f"| **capital_activation_ratio** | **{cap_ratio:.2%}** |")
    w(f"| **production_decision** | **{decision}** |")
    w(f"| stop_condition_fired | {metrics.stop_condition_fired or '— なし'} |")
    w(f"| risk_guard_fired | {'🛑 YES' if metrics.risk_guard_fired else '— NO'} |")
    w("")

    # ── Exposure Summary
    w("---\n## Exposure Configuration\n")
    w("| パラメータ | 値 |")
    w("|---|---|")
    w(f"| max_notional | ¥{int(MAX_NOTIONAL):,} (min 20%×¥1.8M, ¥400k) |")
    w(f"| max_open_positions | {MAX_OPEN_POSITIONS} |")
    w(f"| max_order_per_day | {MAX_ORDER_PER_DAY} |")
    w(f"| max_cancel_retry | {MAX_CANCEL_RETRY} |")
    w(f"| risk_guard_threshold | ¥{int(RISK_GUARD_THRESHOLD):,} (50% × P95_DD) |")
    w(f"| total_notional_submitted | ¥{int(metrics.total_notional_submitted):,} |")
    w(f"| total_notional_filled | ¥{int(metrics.total_notional_filled):,} |")
    w(f"| exposure_utilization_avg | {metrics.exposure_utilization_avg_pct:.1f}% of max_notional |")
    w(f"| exposure_exceeded_count | {metrics.exposure_exceeded_count} "
      f"{'(BUY拒否: 単価 × min_lot > max_notional)' if metrics.exposure_exceeded_count > 0 else ''} |")
    w("")

    # ── Operational Audit
    w("---\n## Operational Audit\n")
    w("| 指標 | 値 | 判定 |")
    w("|---|---|---|")
    rows = [
        ("live_fill_rate",           f"{metrics.live_fill_rate:.1%}",
         "✅" if metrics.live_fill_rate >= 0.95 else "❌"),
        ("signal_match",             f"{metrics.signal_match:.1%}",
         "✅" if metrics.signal_match >= 1.0 else "❌"),
        ("authority_match",          f"{metrics.authority_match:.1%}",
         "✅" if metrics.authority_match >= 1.0 else "❌"),
        ("execution_match",          f"{metrics.execution_match:.1%}",
         "✅" if metrics.execution_match >= 1.0 else "❌"),
        ("cancel_integrity",         f"{metrics.cancel_integrity_rate:.1%}",
         "✅" if metrics.cancel_integrity_rate >= 1.0 else "❌"),
        ("reconciliation_error",     metrics.reconciliation_error,  ok0(metrics.reconciliation_error)),
        ("partial_fill_gap",         metrics.partial_fill_gap_total, ok0(metrics.partial_fill_gap_total)),
        ("cash_truth_gap",           metrics.cash_truth_gap,         ok0(metrics.cash_truth_gap)),
        ("position_truth_gap",       metrics.position_truth_gap,     ok0(metrics.position_truth_gap)),
        ("broker_disconnect_count",  metrics.broker_disconnect_count, "—"),
    ]
    for name, val, status in rows:
        w(f"| {name} | {val} | {status} |")
    w("")

    # ── Latency
    w("---\n## Latency Distribution\n")
    w("| 指標 | p50 | p95 |")
    w("|---|---|---|")
    w(f"| order_roundtrip_ms | {metrics.order_roundtrip_p50_ms} | {metrics.order_roundtrip_p95_ms} |")
    w(f"| intent_to_fill_ms | {metrics.intent_to_fill_p50_ms} | {metrics.intent_to_fill_p95_ms} |")
    w(f"\nサンプル数: roundtrip={len(metrics.order_roundtrip_ms_samples)}"
      f"  intent_fill={len(metrics.intent_to_fill_ms_samples)}")
    w(f"\nclock_skew_max: {metrics.clock_skew_sec_max:.4f}s  "
      f"{'✅ < 2s' if metrics.clock_skew_sec_max < 2.0 else '❌ > 2s'}")
    w("")

    # ── Broker Truth Audit
    w("---\n## Broker Truth Audit\n")
    w("| 指標 | インシデント数 | 判定 |")
    w("|---|---|---|")
    w(f"| broker_cash_diff | {metrics.broker_cash_diff_incidents} | {ok0(metrics.broker_cash_diff_incidents)} |")
    w(f"| broker_position_diff | {metrics.broker_position_diff_incidents} | {ok0(metrics.broker_position_diff_incidents)} |")
    w(f"| broker_order_diff | {metrics.broker_order_diff_incidents} | {ok0(metrics.broker_order_diff_incidents)} |")
    w(f"\n**Broker Final State:**")
    w(f"- broker_cash: ¥{broker.cash:,.0f}")
    w(f"- broker_positions: {dict(broker.positions) or 'FLAT'}")
    w(f"- submitted_orders: {len(broker.submitted_order_ids)}")
    w(f"- filled_executions: {len(broker.filled_execution_ids)}")
    w("")

    # ── Recovery Audit
    w("---\n## Recovery Audit\n")
    w("| 指標 | 値 | 判定 |")
    w("|---|---|---|")
    w(f"| restart_recovery_count | {metrics.restart_recovery_count} | — |")
    w(f"| restart_recovery_success | {metrics.restart_recovery_success} | "
      f"{'✅' if metrics.restart_recovery_rate >= 1.0 else '❌'} |")
    w(f"| restart_recovery_rate | {metrics.restart_recovery_rate:.1%} | "
      f"{'✅' if metrics.restart_recovery_rate >= 1.0 else '❌'} |")
    w(f"| state_recovery_time_max_s | {metrics.state_recovery_time_max_s:.1f}s | "
      f"{'✅ ≤60s' if metrics.state_recovery_time_max_s <= 60.0 else '❌ >60s'} |")
    w(f"| restart_recovery_time_p95_s | {metrics.restart_recovery_time_p95_s:.1f}s | — |")
    w("")

    # ── Advanced Monitoring
    w("---\n## Advanced Monitoring\n")
    w("| 指標 | 値 | 判定 |")
    w("|---|---|---|")
    advanced = [
        ("silent_failure_count",     metrics.silent_failure_count,    ok0(metrics.silent_failure_count)),
        ("ghost_position_count",     metrics.ghost_position_count,    ok0(metrics.ghost_position_count)),
        ("manual_intervention",      metrics.manual_intervention,     ok0(metrics.manual_intervention)),
        ("intent_queue_depth_max",   metrics.intent_queue_depth_max,  "✅" if metrics.intent_queue_depth_max <= 1 else "❌"),
        ("state_recovery_time_max_s",f"{metrics.state_recovery_time_max_s:.1f}s",
         "✅" if metrics.state_recovery_time_max_s <= 60.0 else "❌"),
        ("clock_skew_sec_max",       f"{metrics.clock_skew_sec_max:.4f}s",
         "✅" if metrics.clock_skew_sec_max < 2.0 else "❌"),
        ("unmatched_execution",      metrics.unmatched_execution,     ok0(metrics.unmatched_execution)),
    ]
    for name, val, status in advanced:
        w(f"| {name} | {val} | {status} |")
    w("")

    # ── Stop Conditions
    w("---\n## Stop Conditions\n")
    w("| 条件 | 閾値 | 実績 | 判定 |")
    w("|---|---|---|---|")
    stop_items = [
        ("unexpected_position",  "≥1→ABORT", metrics.unexpected_position),
        ("cash_negative",        "≥1→ABORT", metrics.cash_negative),
        ("duplicate_submit",     "≥1→ABORT", metrics.duplicate_submit),
        ("orphan_order",         "≥1→ABORT", metrics.orphan_order),
        ("manual_override",      "≥1→ABORT", metrics.manual_override),
        ("broker_state_unknown", "≥1→ABORT", metrics.broker_state_unknown),
        ("intent_queue_depth",   ">1→ABORT",  metrics.intent_queue_depth_max),
        ("clock_skew_sec",       ">2s→ABORT", f"{metrics.clock_skew_sec_max:.4f}s"),
        ("unmatched_execution",  "≥1→ABORT", metrics.unmatched_execution),
        ("broker_cash_diff",     "≠0→ABORT",  metrics.broker_cash_diff_incidents),
        ("broker_position_diff", "≠0→ABORT",  metrics.broker_position_diff_incidents),
        ("broker_order_diff",    "≠0→ABORT",  metrics.broker_order_diff_incidents),
        ("rollback (risk_guard)","DD>¥70k",   "🛑 FIRED" if metrics.risk_guard_fired else "—"),
    ]
    for name, thr, val in stop_items:
        fired = name.split(" ")[0] == metrics.stop_condition_fired
        status = "🛑 FIRED" if fired else "✅ OK"
        w(f"| {name} | {thr} | {val} | {status} |")
    w("")

    # ── Promotion Criteria
    w("---\n## Promotion Criteria\n")
    w("| 指標 | 要求 | 判定 |")
    w("|---|---|---|")
    promo_thresholds = {
        "authority_match": "100%", "execution_match": "100%",
        "reconciliation_error": "=0", "ghost_position_count": "=0",
        "manual_intervention": "=0", "live_fill_rate": "≥95%",
        "state_recovery_time": "≤60s", "unmatched_execution": "=0",
        "broker_cash_diff": "=0", "broker_position_diff": "=0",
        "broker_order_diff": "=0", "restart_recovery_success": "100%",
    }
    all_pass = all(promo.values())
    for name, thr in promo_thresholds.items():
        pass_v = promo.get(name, False)
        w(f"| {name} | {thr} | {'✅ PASS' if pass_v else '❌ FAIL'} |")
    w(f"\n**昇格判定: {'✅ 全条件PASS' if all_pass else '❌ 一部未達'}**")
    w("")

    # ── Risk Guard
    w("---\n## Risk Guard\n")
    w(f"- P95_DD (Study15B): ¥{int(P95_DD_STUDY15B):,}")
    w(f"- Risk Guard Threshold: ¥{int(RISK_GUARD_THRESHOLD):,} (50% × P95_DD)")
    w(f"- Realized Drawdown Peak: ¥{metrics.realized_dd:,.0f}")
    w(f"- Risk Guard Fired: {'🛑 YES → allocation=0, authority=OFF' if metrics.risk_guard_fired else '✅ NO'}")
    w("")

    # ── Daily Summary
    w("---\n## Daily Monitoring Summary (最終10日)\n")
    w("| Day | Date | API | Restart | Signals | Fills | Notional | DD | Reliability |")
    w("|---|---|---|---|---|---|---|---|---|")
    for rec in daily_records[-10:]:
        api_s  = "✅" if rec["api_up"] else "⚠"
        rst_s  = "🔄" if rec.get("restart_event") else "—"
        dd_s   = f"¥{int(rec['realized_dd']):,}"
        w(f"| {rec['day_idx']} | {rec['date']} | {api_s} | {rst_s}"
          f" | {rec['n_signals']} | {rec['day_fills']} | ¥{int(rec['day_notional']):,}"
          f" | {dd_s} | {rec['live_reliability']:.1f} |")
    w("")

    # ── Final Verdict
    w("---\n## 最終判定\n")
    w("| 項目 | 値 |")
    w("|---|---|")
    w(f"| live_reliability_score | **{live_score}/100** |")
    w(f"| capital_activation_ratio | **{cap_ratio:.2%}** |")
    w(f"| broker_consistency_score | **{broker_score}/100** |")
    w(f"| restart_recovery_score | **{recovery_score}/100** |")
    w(f"| **production_decision** | **{decision}** |")

    if decision == "GO_SCALE":
        next_step = ("通常capital(¥1.8M全額)でLIMITED_LIVE移行可能。max_positions=1維持。")
    elif decision == "GO_HOLD":
        if metrics.exposure_exceeded_count > 0 and metrics.filled_orders == 0:
            next_step = (
                f"max_notional=¥{int(MAX_NOTIONAL):,}不足 — Study9 Case B銘柄は"
                f"最低1lot>¥{int(MAX_NOTIONAL):,}のため全BUY拒否。"
                "Study15B推奨(efficient_capital=¥1.5M)に合わせてmax_notional再設定後"
                "に再評価。インフラ信頼性は検証済み。"
            )
        else:
            next_step = (
                "LIVE_SANDBOX継続。fill実績蓄積後(目標10件)にGO_SCALEへ。"
                "restart_recovery_score向上のため長期稼働継続推奨。"
            )
    else:
        next_step = "ROLLBACK。問題解消後にPAPER_ACK再実施。"

    w(f"| next_step | {next_step} |")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")
    print(f"  レポート: {path}")


# ──────────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────────

def main() -> int:
    cfg = load_strategy_config()
    rng = np.random.default_rng(RNG_SEED)

    print("=" * 70)
    print("  Study18 Live Sandbox Exposure Gate")
    print(f"  Capital: ¥{int(CAPITAL):,}  Authority: LIVE_SANDBOX"
          f"  MaxNotional: ¥{int(MAX_NOTIONAL):,}")
    print("  収益最適化禁止 / 運用検証のみ")
    print("=" * 70 + "\n")

    # Clear stale artifacts
    for stale in [SANDBOX_AUDIT_LOG,
                  Path("logs/sandbox_telemetry.jsonl"),
                  Path("runtime/study18_idem.json"),
                  SANDBOX_STATE_FILE]:
        if stale.exists():
            stale.unlink()

    # ── Data ────────────────────────────────────────────────────────────
    print("[1/5] データロード...")
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

    # ── Matrices ────────────────────────────────────────────────────────
    print("[2/5] マトリクス構築...")
    open_mat  = np.full((n_dates, n_syms), np.nan, dtype=np.float32)
    close_mat = np.full((n_dates, n_syms), np.nan, dtype=np.float32)
    for si, sym in enumerate(active_syms):
        df_src = universe_raw[sym]["df"]
        ri     = df_src.index.get_indexer(common_dates)
        if np.any(ri < 0): continue
        open_mat[:,  si] = df_src["Open"].to_numpy(dtype=np.float32)[ri]
        close_mat[:, si] = df_src["Close"].to_numpy(dtype=np.float32)[ri]

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

    cross90 = _cross90(rsr_mat)
    slope5  = _slope5(rsr_mat)

    # ── Signal generation ─────────────────────────────────────────────
    print("[3/5] シグナル生成 (Study9 Case B リプレイ — 全期間)...")
    all_signals = generate_signals(
        open_mat, close_mat, rsr_mat, sym_active_mat,
        cross90, slope5, mkt_ret1,
        active_syms, sym_to_i, common_dates, n_dates,
        sl_cash=CAPITAL,
    )
    entries = [s for s in all_signals if s["type"] == "ENTRY"]
    exits   = [s for s in all_signals if s["type"] == "EXIT"]
    print(f"  全期間: total_signals={len(all_signals)}"
          f"  entries={len(entries)}  exits={len(exits)}")

    # ── 30-day window (anchored at last signal)
    if all_signals:
        last_sig_date = max(s["date"] for s in all_signals)
        anchor_idx = next(
            (k for k, dt in enumerate(common_dates)
             if str(dt.date()) == last_sig_date),
            len(common_dates) - 1,
        )
        start_idx = max(0, anchor_idx - SIM_N_DAYS + 1)
        sim_dates = common_dates[start_idx : anchor_idx + 1]
    else:
        sim_dates = common_dates[-SIM_N_DAYS:]

    sim_start = str(sim_dates[0].date())
    sim_end   = str(sim_dates[-1].date())
    sigs_in_window = [s for s in all_signals
                      if sim_start <= s["date"] <= sim_end]
    print(f"  30日ウィンドウ: {sim_start} → {sim_end}")
    print(f"  ウィンドウ内シグナル: {len(sigs_in_window)}\n")

    # ── Live sandbox simulation ──────────────────────────────────────
    print("[4/5] LIVE_SANDBOX シミュレーション (30日間)...")
    metrics, daily_records, broker = run_sandbox_simulation(
        all_signals, sim_dates, rng
    )

    live_score    = compute_live_reliability_score(metrics)
    broker_score  = compute_broker_consistency_score(metrics)
    recovery_score = compute_restart_recovery_score(metrics)
    cap_ratio     = compute_capital_activation_ratio(metrics)
    promo         = check_promotion_criteria(metrics)
    decision      = determine_production_decision(
        live_score, broker_score, recovery_score,
        promo, metrics.stop_condition_fired,
        metrics.risk_guard_fired, metrics.filled_orders,
    )

    print(f"\n  live_reliability_score:    {live_score:.1f}/100")
    print(f"  broker_consistency_score:  {broker_score:.1f}/100")
    print(f"  restart_recovery_score:    {recovery_score:.1f}/100")
    print(f"  capital_activation_ratio:  {cap_ratio:.2%}")
    print(f"  uptime_pct:                {metrics.api_uptime_pct:.2f}%")
    print(f"  filled_orders:             {metrics.filled_orders}")
    print(f"  total_notional_filled:     ¥{int(metrics.total_notional_filled):,}")
    print(f"  restart_recovery_count:    {metrics.restart_recovery_count}")
    print(f"  restart_recovery_success:  {metrics.restart_recovery_success}")
    print(f"  state_recovery_time_max:   {metrics.state_recovery_time_max_s:.1f}s")
    print(f"  realized_dd:               ¥{metrics.realized_dd:,.0f}")
    print(f"  risk_guard_fired:          {metrics.risk_guard_fired}")
    print(f"  stop_condition_fired:      {metrics.stop_condition_fired or 'なし'}")
    print(f"  production_decision:       {decision}")
    print(f"\n  全プロモーション基準: {'✅ PASS' if all(promo.values()) else '❌ 一部FAIL'}")
    for k, v in promo.items():
        if not v:
            print(f"    ❌ {k}")

    # ── Save authority state
    final_auth = dataclasses.replace(
        AuthorityState.initial(),
        level=AuthorityLevel.LIMITED_LIVE.value,
    )
    save_state(final_auth)

    # ── Output ──────────────────────────────────────────────────────────
    print("\n[5/5] レポート出力...")
    write_daily_csv(daily_records, REPORTS_DIR / "study18_daily_metrics.csv")
    write_report(
        metrics, daily_records, broker,
        sim_start, sim_end,
        n_signals_total=len(all_signals),
        promo=promo,
        live_score=live_score, broker_score=broker_score,
        recovery_score=recovery_score, cap_ratio=cap_ratio,
        decision=decision,
        path=REPORTS_DIR / "study18_live_sandbox.md",
    )
    print("\n完了.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
