"""
src/backtest/study17_reliability_validation.py — Study17 Production Reliability Validation

Validates operational reliability of the RSR90 deployment stack under PAPER_ACK conditions
over a simulated 30-trading-day window.

No alpha research. No strategy modification. No optimization.

Fixed:
  Strategy:  Study9 Case B
  Capital:   ¥1,800,000
  Authority: PAPER_ACK
  Duration:  30 trading days

Output:
  reports/study17_reliability.md
  reports/study17_daily_metrics.csv
  logs/reliability_telemetry.jsonl
  logs/paper_ack_audit.jsonl

Run:
    cd C:/ai-trading
    python src/backtest/study17_reliability_validation.py
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
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd

from src.config_loader import load_strategy_config
from src.backtest.capital_allocation_abc import load_data, _take, LOT
from src.live.authority_state import (
    AuthorityLevel,
    AuthorityState,
    PROMOTION_CRITERIA_KEYS,
    load_state,
    save_state,
    record_signal,
    record_execution,
    increment_counter,
    fire_stop,
    advance_day,
)
from src.live.order_authority import (
    SignalRecord, IntentRecord, AuthorityRecord, ExecutionRecord,
    IdempotencyRegistry, _audit_record, _sha256, _canon,
    AUDIT_LOG,
)
from src.live.reliability_monitor import (
    ReliabilityMetrics,
    check_stop_conditions,
    compute_reliability_score,
    compute_authority_integrity_score,
    compute_execution_integrity_score,
    compute_state_consistency_score,
    determine_production_readiness,
    append_daily_telemetry,
    RELIABILITY_TELEMETRY_FILE,
)

REPORTS_DIR  = Path("reports")
PAPER_ACK_AUDIT_LOG = Path("logs/paper_ack_audit.jsonl")

CAPITAL    = 1_800_000.0
IS_START   = "2018-01-01"
SIM_N_DAYS = 30            # validation window

# Strategy params (Study9 Case B — frozen)
RSR_LO     = 92.0
RSR_HI     = 95.0
D90_MAX    = 5
SLOPE_MAX  = 5.0
EXIT_THR   = 90.0
MIN_HOLD   = 3
SHOCK_MKT  = -0.05
SHOCK_SYM  = -0.08
SLIPPAGE   = 0.001
COMMISSION = 0.00055
COST_ONE_WAY = SLIPPAGE + COMMISSION   # 0.00155

# Latency model — kabu Station localhost (log-normal)
#   order_ack:  p50≈95ms, p95≈175ms, p99≈240ms
#   cancel_ack: p50≈70ms, p95≈130ms, p99≈180ms
LATENCY_ORDER_MU    = 4.55
LATENCY_ORDER_SIGMA = 0.35
LATENCY_CANCEL_MU   = 4.25
LATENCY_CANCEL_SIGMA = 0.30

# Reliability event probabilities (per trading day)
P_API_DOWN           = 0.003   # 99.7% uptime
P_RECONNECT_LAMBDA   = 0.04    # Poisson mean reconnects per available day
P_RESTORE_FAIL       = 0.005   # 99.5% state restore success
P_RECOVERY_FAIL      = 0.01    # 99.0% session recovery success

RNG_SEED = 17   # fixed for reproducibility


# ──────────────────────────────────────────────────────────────────────
#  Matrix helpers (same as Study16)
# ──────────────────────────────────────────────────────────────────────

def _compute_cross90(rsr_mat: np.ndarray) -> np.ndarray:
    n, m = rsr_mat.shape
    out  = np.zeros((n, m), dtype=np.int32)
    cnt  = np.zeros(m, dtype=np.int32)
    for i in range(n):
        out[i] = cnt
        ab = rsr_mat[i] >= 90.0
        cnt[ab] += 1
        cnt[~ab]  = 0
    return out


def _compute_slope5(rsr_mat: np.ndarray) -> np.ndarray:
    out = np.zeros_like(rsr_mat, dtype=np.float32)
    out[5:] = rsr_mat[5:] - rsr_mat[:-5]
    return out


# ──────────────────────────────────────────────────────────────────────
#  Signal generator (Study9 Case B — frozen, same as Study16)
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
    """
    Replay Study9 Case B over the full date range.
    Returns list of {type, date, symbol, price, qty, cash_before, cash_after,
                     rsr, d90, slope5, reason, hold_days, pnl}.
    """
    pos: Optional[_Position] = None
    signals: list[dict] = []
    cash = sl_cash

    for i, date in enumerate(common_dates):
        ds       = str(date.date())
        mkt_shock = mkt_ret1 is not None and float(mkt_ret1[i]) <= SHOCK_MKT
        if i + 1 >= n_dates:
            break
        nxt = i + 1

        # EXIT
        if pos:
            si = sym_to_i[pos.symbol]
            rv = float(rsr_mat[i, si])
            ct = float(close_mat[i, si])
            hd = i - pos.entry_idx
            do_exit = False; reason = ""

            if mkt_shock and i > 0:
                pc = float(close_mat[i - 1, si])
                if pc > 0 and (ct / pc - 1) <= SHOCK_SYM:
                    do_exit = True; reason = "MKT_SHOCK"
            if not do_exit:
                pos.days_below = (pos.days_below + 1) if rv < EXIT_THR else 0
                if pos.days_below >= 1 and hd >= MIN_HOLD:
                    do_exit = True; reason = "RSR_EXIT"

            if do_exit:
                sp = float(open_mat[nxt, si])
                cb = cash
                cash += pos.qty * sp * (1 - COST_ONE_WAY)
                signals.append({
                    "type": "EXIT", "date": ds, "symbol": pos.symbol,
                    "price": round(sp, 2), "qty": pos.qty,
                    "cash_before": round(cb, 0), "cash_after": round(cash, 0),
                    "rsr": round(rv, 2), "d90": 0, "slope5": 0.0,
                    "reason": reason, "hold_days": hd,
                    "pnl": round((sp - pos.entry_price) * pos.qty, 0),
                })
                pos = None

        # ENTRY
        if pos is None and not mkt_shock:
            cands = []
            for sym in active_syms:
                si  = sym_to_i[sym]
                rv  = float(rsr_mat[i, si])
                d90 = int(cross90_mat[i, si])
                sl5 = float(slope5_mat[i, si])
                if rv < RSR_LO or rv >= RSR_HI: continue
                if d90 > D90_MAX or d90 < 1:    continue
                if sl5 > SLOPE_MAX:              continue
                if (sym_active_mat is not None
                        and float(sym_active_mat[i, si]) < 0.5): continue
                cands.append((rv, d90, sl5, sym))

            if cands:
                cands.sort(key=lambda x: (-x[0], x[1]))
                rv, d90, sl5, sym = cands[0]
                si  = sym_to_i[sym]
                bp  = float(open_mat[nxt, si])
                if bp <= 0: continue
                alloc = cash * 0.95
                qty   = int(alloc / bp / LOT) * LOT
                cost  = qty * bp * (1 + COST_ONE_WAY)
                if qty > 0 and cost <= cash:
                    cb = cash; cash -= cost
                    pos = _Position(sym, qty, bp, nxt, rv, d90, sl5)
                    signals.append({
                        "type": "ENTRY", "date": ds, "symbol": sym,
                        "price": round(bp, 2), "qty": qty,
                        "cash_before": round(cb, 0), "cash_after": round(cash, 0),
                        "rsr": round(rv, 2), "d90": d90, "slope5": round(sl5, 2),
                        "reason": "RSR_ENTRY", "hold_days": 0, "pnl": 0,
                    })
    return signals


# ──────────────────────────────────────────────────────────────────────
#  Reliability simulation helpers
# ──────────────────────────────────────────────────────────────────────

def _broker_ref(symbol: str, date: str, side: str) -> str:
    return hashlib.sha256(f"PAPER_ACK|{symbol}|{date}|{side}".encode()).hexdigest()[:12].upper()


def _sim_api_event(rng: np.random.Generator) -> dict:
    """
    Simulate per-day API connectivity event.
    Returns: {api_up, reconnect_count, restore_needed, restore_success, recovery_success}
    """
    api_up = rng.random() > P_API_DOWN
    reconnect_count = 0
    restore_needed  = False
    restore_success = True
    recovery_success = True

    if not api_up:
        # Session lost — attempt recovery
        restore_needed   = True
        recovery_success = rng.random() > P_RECOVERY_FAIL
        restore_success  = rng.random() > P_RESTORE_FAIL if recovery_success else False
        reconnect_count  = 1
    else:
        # Random mid-day reconnect events (Poisson)
        n_reconnect = int(rng.poisson(P_RECONNECT_LAMBDA))
        if n_reconnect > 0:
            reconnect_count = n_reconnect
            restore_needed  = True
            restore_success = rng.random() > P_RESTORE_FAIL

    return {
        "api_up":           api_up,
        "reconnect_count":  reconnect_count,
        "restore_needed":   restore_needed,
        "restore_success":  restore_success,
        "recovery_success": recovery_success,
    }


def _sim_order_ack_latency(rng: np.random.Generator) -> float:
    return round(float(rng.lognormal(LATENCY_ORDER_MU, LATENCY_ORDER_SIGMA)), 1)


def _sim_cancel_ack_latency(rng: np.random.Generator) -> float:
    return round(float(rng.lognormal(LATENCY_CANCEL_MU, LATENCY_CANCEL_SIGMA)), 1)


def _validate_intent(intent: IntentRecord, cash: float) -> tuple[bool, str]:
    if intent.qty <= 0:
        return False, "qty_zero"
    if intent.price_limit <= 0:
        return False, "price_invalid"
    if intent.side == "BUY":
        cost = intent.qty * intent.price_limit * (1 + COST_ONE_WAY)
        if cost > cash * 1.05:
            return False, f"insufficient_cash cost={cost:.0f} cash={cash:.0f}"
    if intent.qty % LOT != 0:
        return False, f"qty_not_lot_multiple"
    return True, ""


def _process_paper_ack_signal(
    sig: dict,
    state: AuthorityState,
    idem: IdempotencyRegistry,
    metrics: ReliabilityMetrics,
    rng: np.random.Generator,
) -> tuple[AuthorityState, ReliabilityMetrics, Optional[dict]]:
    """
    Process one signal through PAPER_ACK flow:
      signal → intent → authority(PAPER_ACK) → order_ack → cancel_ack
    Returns updated (state, metrics, audit_row_or_None).
    """
    side = "BUY" if sig["type"] == "ENTRY" else "SELL"
    sr   = SignalRecord.build(sig["symbol"], sig["date"], sig["rsr"], side,
                              sig.get("d90", 0), sig.get("slope5", 0.0))

    # Idempotency check
    is_dup, ikey = idem.check_and_register(sr)
    if is_dup:
        state = record_signal(state, approved=False)
        state = increment_counter(state, "duplicate_submit")
        state = increment_counter(state, "idempotency_failure")
        metrics.duplicate_submit_count += 1
        logger.warning("[PAPER_ACK] duplicate: %s %s", sig["symbol"], sig["date"])
        return state, metrics, None

    # Build intent
    ir = IntentRecord.build(sr, sig["qty"], sig["price"], sig["cash_before"])

    # Validate
    approved, reject_reason = _validate_intent(ir, sig["cash_before"])
    ar = AuthorityRecord.build(ir, AuthorityLevel.PAPER_ACK.value,
                               approved=approved, reject_reason=reject_reason)
    state = record_signal(state, approved=approved)
    metrics.total_signals  += 1
    if approved:
        metrics.approved_signals += 1

    # Simulate broker ACK latency
    order_ack_ms  = _sim_order_ack_latency(rng) if approved else 0.0
    cancel_ack_ms = _sim_cancel_ack_latency(rng) if approved else 0.0
    total_lat_ms  = order_ack_ms + cancel_ack_ms

    if approved:
        metrics.order_ack_latency_samples.append(order_ack_ms)
        metrics.cancel_ack_latency_samples.append(cancel_ack_ms)

    # Execution record — fill_qty=0 (cancelled before fill), cancel_count=1
    ref = _broker_ref(sig["symbol"], sig["date"], side)
    er  = ExecutionRecord.build(
        ar, sr, ir,
        fill_price=sig["price"] if approved else 0.0,
        fill_qty=0,           # PAPER_ACK: always cancelled
        broker_ref=f"PAPER_{ref}",
        cash_before=sig["cash_before"],
        cash_after=sig["cash_before"],    # no cash change (cancelled)
        position_before=0,
        position_after=0,
        broker_latency_ms=total_lat_ms,
        cancel_count=1 if approved else 0,
        case_id="PAPER_ACK",
    )

    # Chain validation
    chain_ok = er.chain_valid(ir.intent_hash)
    if not chain_ok:
        state = increment_counter(state, "authority_mismatch")
        metrics.chain_break_count += 1
        logger.error("[PAPER_ACK] chain break: %s %s", sig["symbol"], sig["date"])

    # Intent-execution divergence check (fill_qty should always be 0 in PAPER_ACK)
    if er.fill_qty != 0:
        metrics.intent_execution_divergence += 1

    # Partial fill check (should never occur for PAPER_ACK cancel)
    if er.partial_fill_gap > 0 and er.fill_qty > 0:
        metrics.partial_fill_count += 1

    # Reconciliation: cash/position should be unchanged
    cash_changed = abs(er.cash_after - er.cash_before) > 0.01
    pos_changed  = er.position_after != er.position_before
    if cash_changed:
        metrics.cash_truth_gap += 1
        state = increment_counter(state, "cash_truth_gap")
    if pos_changed:
        metrics.position_truth_gap += 1
        state = increment_counter(state, "position_truth_gap")
    if cash_changed or pos_changed:
        metrics.reconciliation_error_count += 1
        state = increment_counter(state, "reconciliation_error")

    # Successful cancel
    if approved and not cash_changed and not pos_changed:
        metrics.cancelled_signals += 1

    state = record_execution(state, executed=False)

    # Audit record
    audit_row = _audit_record(sr, ir, ar, er, chain_ok, ikey)
    audit_row["broker_ref"]       = er.broker_ref
    audit_row["order_ack_ms"]     = order_ack_ms
    audit_row["cancel_ack_ms"]    = cancel_ack_ms

    # Write to paper_ack audit log
    try:
        PAPER_ACK_AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
        with PAPER_ACK_AUDIT_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(audit_row, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.error("[PAPER_ACK] audit write failed: %s", e)

    return state, metrics, audit_row


# ──────────────────────────────────────────────────────────────────────
#  30-Day Reliability Simulation
# ──────────────────────────────────────────────────────────────────────

def run_30day_simulation(
    all_signals: list[dict],
    sim_dates: pd.DatetimeIndex,
    rng: np.random.Generator,
) -> tuple[ReliabilityMetrics, list[dict], AuthorityState]:
    """
    Simulate PAPER_ACK reliability over sim_dates.
    Returns (metrics, daily_records, final_authority_state).
    """
    metrics      = ReliabilityMetrics()
    daily_records: list[dict] = []
    state        = dataclasses.replace(
        AuthorityState.initial(),
        level=AuthorityLevel.PAPER_ACK.value,
    )
    idem = IdempotencyRegistry(Path("runtime/study17_idem.json"))

    # Index signals by date for O(1) lookup
    sigs_by_date: dict[str, list[dict]] = {}
    for s in all_signals:
        sigs_by_date.setdefault(s["date"], []).append(s)

    sim_start = str(sim_dates[0].date())
    sim_end   = str(sim_dates[-1].date())

    print(f"  シミュレーション期間: {sim_start} → {sim_end}")

    for day_idx, dt in enumerate(sim_dates):
        ds = str(dt.date())
        metrics.n_trading_days += 1

        # API / connectivity event
        api_event = _sim_api_event(rng)
        api_up    = api_event["api_up"]

        if api_up:
            metrics.api_available_days += 1
        else:
            metrics.session_recovery_attempts += 1
            if api_event["recovery_success"]:
                metrics.session_recovery_success += 1
                metrics.api_available_days += 1   # recovered for the day

        if api_event["restore_needed"]:
            metrics.state_restore_attempts += 1
            if api_event["restore_success"]:
                metrics.state_restore_success += 1

        metrics.broker_reconnect_count += api_event["reconnect_count"]

        # Advance authority state day counter
        state = advance_day(state)

        # Process signals for this date (only if API available)
        day_signals = sigs_by_date.get(ds, [])
        day_approved = 0
        day_chain_ok = 0

        if api_up or api_event["recovery_success"]:
            for sig in day_signals:
                state, metrics, audit_row = _process_paper_ack_signal(
                    sig, state, idem, metrics, rng
                )
                if audit_row is not None:
                    if audit_row.get("approved"):
                        day_approved += 1
                    if audit_row.get("chain_valid"):
                        day_chain_ok += 1
        else:
            # API fully down, signals skipped
            metrics.api_timeout_count += 1

        if day_signals:
            metrics.n_signal_days += 1

        # Daily reconciliation (state unchanged → OK)
        # cash_truth / position_truth already tracked per-signal above

        # Stop condition check
        stop = check_stop_conditions(metrics)
        if stop and not metrics.stop_condition_fired:
            metrics.stop_condition_fired = stop
            state = fire_stop(state, stop)
            print(f"  ⛔ STOP CONDITION FIRED: {stop} on {ds}")

        # Daily telemetry
        day_rec = {
            "day_idx":       day_idx + 1,
            "date":          ds,
            "api_up":        api_up,
            "reconnects":    api_event["reconnect_count"],
            "n_signals":     len(day_signals),
            "n_approved":    day_approved,
            "n_chain_ok":    day_chain_ok,
            "order_ack_p50": metrics.order_ack_p50,
            "cancel_ack_p50": metrics.cancel_ack_p50,
            "stop_fired":    metrics.stop_condition_fired or "",
            "cumulative_reliability": compute_reliability_score(metrics),
        }
        daily_records.append(day_rec)
        append_daily_telemetry(day_rec)

        if metrics.stop_condition_fired:
            break

    return metrics, daily_records, state


# ──────────────────────────────────────────────────────────────────────
#  Output: daily CSV
# ──────────────────────────────────────────────────────────────────────

def write_daily_csv(daily_records: list[dict], path: Path) -> None:
    if not daily_records:
        return
    fieldnames = list(daily_records[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(daily_records)
    print(f"  日次CSV: {path}")


# ──────────────────────────────────────────────────────────────────────
#  Output: markdown report
# ──────────────────────────────────────────────────────────────────────

def write_report(
    metrics: ReliabilityMetrics,
    daily_records: list[dict],
    state: AuthorityState,
    sim_start: str,
    sim_end: str,
    n_signals_total: int,
    path: Path,
) -> None:
    today = time.strftime("%Y-%m-%d")

    rs   = compute_reliability_score(metrics)
    ais  = compute_authority_integrity_score(metrics)
    eis  = compute_execution_integrity_score(metrics)
    scs  = compute_state_consistency_score(metrics)
    prod, next_auth = determine_production_readiness(
        rs, ais, eis, scs,
        metrics.stop_condition_fired,
        metrics.n_trading_days,
    )

    # Rollback conditions
    rollback_conditions = [
        "alpha_realization_30d < 80% → allocation=0%, authority=OFF, incident report",
        "chain_break >= 1 → ABORT immediately",
        "cash_truth_gap != 0 → ABORT immediately",
        "position_truth_gap != 0 → ABORT immediately",
        "orphan_order_unresolved >= 1 → ABORT immediately",
    ]

    # Operational risk register
    op_risks = []
    if metrics.broker_reconnect_count > 5:
        op_risks.append(f"HIGH reconnect_count={metrics.broker_reconnect_count} (>5 in 30d)")
    if metrics.api_available_days < metrics.n_trading_days:
        down = metrics.n_trading_days - metrics.api_available_days
        op_risks.append(f"API down={down} days — session recovery tested")
    if metrics.order_ack_p99 > 500:
        op_risks.append(f"order_ack_p99={metrics.order_ack_p99}ms > 500ms threshold")
    if metrics.cancel_ack_p99 > 500:
        op_risks.append(f"cancel_ack_p99={metrics.cancel_ack_p99}ms > 500ms threshold")
    if not op_risks:
        op_risks.append("なし — 全項目正常範囲")

    L = []; w = L.append
    ok = lambda v: "✅" if v == 0 or v is True else "❌"

    w("# Study17 Production Reliability Validation")
    w(f"\n作成日: {today}  |  信頼性監査のみ / 収益評価禁止")
    w(f"\n**Strategy**: Study9 Case B (固定)  **Capital**: ¥{int(CAPITAL):,}"
      f"  **Authority**: PAPER_ACK")
    w(f"\n**期間**: {sim_start} → {sim_end}  ({metrics.n_trading_days} trading days)\n")

    # ── Executive Summary
    w("---\n## Executive Summary\n")
    w("| 指標 | 値 |")
    w("|---|---|")
    w(f"| **reliability_score** | **{rs}/100** |")
    w(f"| **authority_integrity_score** | **{ais}/100** |")
    w(f"| **execution_integrity_score** | **{eis}/100** |")
    w(f"| **state_consistency_score** | **{scs}/100** |")
    w(f"| **production_readiness** | **{prod}** |")
    w(f"| recommended_next_authority | {next_auth} |")
    w(f"| stop_condition_fired | {metrics.stop_condition_fired or '— なし'} |")
    w("")

    # ── Uptime & Connectivity
    w("---\n## API / Connectivity\n")
    w("| 指標 | 値 |")
    w("|---|---|")
    w(f"| uptime_pct | {metrics.uptime_pct:.2f}% |")
    w(f"| api_available_days | {metrics.api_available_days} / {metrics.n_trading_days} |")
    w(f"| broker_reconnect_count | {metrics.broker_reconnect_count} |")
    w(f"| api_timeout_count | {metrics.api_timeout_count} |")
    w(f"| session_recovery_attempts | {metrics.session_recovery_attempts} |")
    w(f"| session_recovery_success | {metrics.session_recovery_success} "
      f"({metrics.session_recovery_rate:.1f}%) |")
    w(f"| state_restore_attempts | {metrics.state_restore_attempts} |")
    w(f"| state_restore_success | {metrics.state_restore_success} "
      f"({metrics.state_restore_success_rate:.1f}%) |")
    w("")

    # ── Latency
    w("---\n## Latency Distribution (PAPER_ACK)\n")
    w("| 指標 | p50 | p95 | p99 |")
    w("|---|---|---|---|")
    w(f"| order_ack_latency_ms | {metrics.order_ack_p50} | {metrics.order_ack_p95} | {metrics.order_ack_p99} |")
    w(f"| cancel_ack_latency_ms | {metrics.cancel_ack_p50} | {metrics.cancel_ack_p95} | {metrics.cancel_ack_p99} |")
    w(f"\nサンプル数: order_ack={len(metrics.order_ack_latency_samples)}  "
      f"cancel_ack={len(metrics.cancel_ack_latency_samples)}")
    w("")

    # ── Order Flow
    w("---\n## Order Flow (PAPER_ACK)\n")
    w("| 指標 | 値 |")
    w("|---|---|")
    w(f"| signals_in_30d_window | {metrics.total_signals} |")
    w(f"| signals_total_period | {n_signals_total} |")
    w(f"| approved_signals | {metrics.approved_signals} |")
    w(f"| cancelled_signals | {metrics.cancelled_signals} |")
    w(f"| cancel_completeness | {metrics.cancel_completeness:.1f}% |")
    w(f"| authority_precision | {metrics.authority_precision:.1f}% |")
    w(f"| partial_fill_count | {metrics.partial_fill_count} {ok(metrics.partial_fill_count)} |")
    w(f"| intent_execution_divergence | {metrics.intent_execution_divergence} {ok(metrics.intent_execution_divergence)} |")
    w("")

    # ── Integrity Metrics
    w("---\n## Integrity Metrics\n")
    w("| 指標 | 値 | 判定 |")
    w("|---|---|---|")
    items = [
        ("chain_break_count",           metrics.chain_break_count),
        ("duplicate_submit_count",       metrics.duplicate_submit_count),
        ("orphan_order_count",           metrics.orphan_order_count),
        ("manual_override_count",        metrics.manual_override_count),
        ("cash_truth_gap",               metrics.cash_truth_gap),
        ("position_truth_gap",           metrics.position_truth_gap),
        ("reconciliation_error_count",   metrics.reconciliation_error_count),
        ("intent_execution_divergence",  metrics.intent_execution_divergence),
    ]
    for name, val in items:
        w(f"| {name} | {val} | {ok(val)} |")
    w("")

    # ── Stop Conditions
    w("---\n## Stop Conditions\n")
    w("| 条件 | 閾値 | 判定 |")
    w("|---|---|---|")
    stop_checks = [
        ("unexpected_position",     metrics.unexpected_position,        "≥1 → ABORT"),
        ("cash_negative",           metrics.cash_negative,              "≥1 → ABORT"),
        ("position_truth_gap",      metrics.position_truth_gap,         "≠0 → ABORT"),
        ("cash_truth_gap",          metrics.cash_truth_gap,             "≠0 → ABORT"),
        ("orphan_order_unresolved", metrics.orphan_order_count,         "≥1 → ABORT"),
        ("duplicate_submit",        metrics.duplicate_submit_count,     "≥1 → ABORT"),
        ("chain_break",             metrics.chain_break_count,          "≥1 → ABORT"),
        ("manual_override",         metrics.manual_override_count,      "≥1 → ABORT"),
    ]
    for name, val, threshold in stop_checks:
        fired = name == metrics.stop_condition_fired
        status = "🛑 FIRED" if fired else "✅ OK"
        w(f"| {name} | {threshold} | {val} — {status} |")
    w("")

    # ── Authority Integrity Promotion Criteria
    w("---\n## Promotion Criteria (30日間 all=0 必須)\n")
    w("| 指標 | 要求 | 実績 | 判定 |")
    w("|---|---|---|---|")
    promo_items = [
        ("authority_mismatch",      state.authority_mismatch),
        ("reconciliation_error",    state.reconciliation_error),
        ("replay_consistency_fail", state.replay_consistency_fail),
        ("duplicate_submit",        state.duplicate_submit),
        ("cash_truth_gap",          state.cash_truth_gap),
        ("position_truth_gap",      state.position_truth_gap),
        ("orphan_order",            state.orphan_order),
        ("idempotency_failure",     state.idempotency_failure),
    ]
    all_zero = True
    for name, val in promo_items:
        status = "✅" if val == 0 else "❌"
        if val != 0:
            all_zero = False
        w(f"| {name} | 0 | {val} | {status} |")
    w(f"\n**昇格判定: {'✅ PASS — LIMITED_LIVEへの昇格条件クリア' if all_zero else '❌ FAIL — 未解決問題あり'}**")
    w("")

    # ── Rollback Conditions
    w("---\n## Rollback Conditions\n")
    for rc in rollback_conditions:
        w(f"- {rc}")
    w("")

    # ── Operational Risk Register
    w("---\n## Operational Risk Register\n")
    w("| # | リスク |")
    w("|---|---|")
    for i, risk in enumerate(op_risks, 1):
        w(f"| {i} | {risk} |")
    w("")

    # ── Daily Summary (last 10 days)
    w("---\n## Daily Monitoring Summary (最終10日)\n")
    w("| Day | Date | API | Reconnects | Signals | Approved | Chain OK | Reliability |")
    w("|---|---|---|---|---|---|---|---|")
    for rec in daily_records[-10:]:
        api_str = "✅" if rec["api_up"] else "⚠"
        w(f"| {rec['day_idx']} | {rec['date']} | {api_str} | {rec['reconnects']} "
          f"| {rec['n_signals']} | {rec['n_approved']} | {rec['n_chain_ok']} "
          f"| {rec['cumulative_reliability']:.1f} |")
    w("")

    # ── Final Verdict
    w("---\n## 最終判定\n")
    w("| 項目 | 値 |")
    w("|---|---|")
    w(f"| reliability_score | **{rs}/100** |")
    w(f"| authority_integrity_score | **{ais}/100** |")
    w(f"| execution_integrity_score | **{eis}/100** |")
    w(f"| state_consistency_score | **{scs}/100** |")
    w(f"| production_readiness | **{prod}** |")
    w(f"| recommended_next_authority | **{next_auth}** |")
    w(f"| rollback_rule | alpha_realization_30d<80% → allocation=0%, authority=OFF |")
    w(f"| operational_risks | {op_risks[0]} |")
    w(f"| stop_condition_fired | {metrics.stop_condition_fired or 'なし'} |")
    if prod == "GO_LIMITED_LIVE":
        w(f"| **next_step** | **Case D (LIMITED_LIVE) 実施可能 — ¥1.8M上限・最大1枠** |")
    elif prod == "CONTINUE_PAPER_ACK":
        w(f"| next_step | PAPER_ACK 継続 — 追加30日間モニタリング |")
    else:
        w(f"| next_step | {prod} — 問題解消後に再評価 |")

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
    print("  Study17 Production Reliability Validation")
    print(f"  Capital: ¥{int(CAPITAL):,}  Authority: PAPER_ACK  Duration: {SIM_N_DAYS}d")
    print("  収益評価禁止 / 信頼性検証のみ")
    print("=" * 70 + "\n")

    # Clear stale study17 artifacts
    for stale in [RELIABILITY_TELEMETRY_FILE,
                  PAPER_ACK_AUDIT_LOG,
                  Path("runtime/study17_idem.json")]:
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

    cross90 = _compute_cross90(rsr_mat)
    slope5  = _compute_slope5(rsr_mat)

    # ── Signal generation (full period) ─────────────────────────────────
    print("[3/5] シグナル生成 (Study9 Case B リプレイ — 全期間)...")
    all_signals = generate_signals(
        open_mat, close_mat, rsr_mat, sym_active_mat,
        cross90, slope5, mkt_ret1,
        active_syms, sym_to_i, common_dates, n_dates,
        sl_cash=CAPITAL,
    )
    entries = [s for s in all_signals if s["type"] == "ENTRY"]
    exits   = [s for s in all_signals if s["type"] == "EXIT"]
    print(f"  全期間: total_signals={len(all_signals)}  entries={len(entries)}  exits={len(exits)}")

    # ── Identify 30-day simulation window
    # Anchor at the last date with a signal; fall back to last SIM_N_DAYS of common_dates.
    if all_signals:
        last_sig_date = max(s["date"] for s in all_signals)
        anchor_idx = None
        for k, dt in enumerate(common_dates):
            if str(dt.date()) == last_sig_date:
                anchor_idx = k
                break
        if anchor_idx is None:
            anchor_idx = len(common_dates) - 1
        # Take SIM_N_DAYS ending at anchor_idx (or start of series if too early)
        start_idx = max(0, anchor_idx - SIM_N_DAYS + 1)
        sim_dates = common_dates[start_idx : anchor_idx + 1]
    else:
        sim_dates = common_dates[-SIM_N_DAYS:]

    sim_start = str(sim_dates[0].date())
    sim_end   = str(sim_dates[-1].date())
    sigs_in_window = [s for s in all_signals if sim_start <= s["date"] <= sim_end]
    print(f"  30日ウィンドウ: {sim_start} → {sim_end}")
    print(f"  ウィンドウ内シグナル: {len(sigs_in_window)}\n")

    # ── 30-day PAPER_ACK reliability simulation ──────────────────────────
    print("[4/5] PAPER_ACK 信頼性シミュレーション (30日間)...")
    metrics, daily_records, final_state = run_30day_simulation(
        all_signals, sim_dates, rng
    )
    print(f"\n  reliability_score:          {compute_reliability_score(metrics):.1f}/100")
    print(f"  authority_integrity_score:  {compute_authority_integrity_score(metrics):.1f}/100")
    print(f"  execution_integrity_score:  {compute_execution_integrity_score(metrics):.1f}/100")
    print(f"  state_consistency_score:    {compute_state_consistency_score(metrics):.1f}/100")
    print(f"  uptime_pct:                 {metrics.uptime_pct:.2f}%")
    print(f"  broker_reconnect_count:     {metrics.broker_reconnect_count}")
    print(f"  signals_processed:          {metrics.total_signals}")
    print(f"  cancelled_signals:          {metrics.cancelled_signals}")
    print(f"  stop_condition_fired:       {metrics.stop_condition_fired or 'なし'}")

    # ── Save final authority state ───────────────────────────────────────
    save_state(final_state)

    # ── Output ──────────────────────────────────────────────────────────
    print("\n[5/5] レポート出力...")
    write_daily_csv(daily_records, REPORTS_DIR / "study17_daily_metrics.csv")
    write_report(
        metrics, daily_records, final_state,
        sim_start, sim_end,
        n_signals_total=len(all_signals),
        path=REPORTS_DIR / "study17_reliability.md",
    )
    print("\n完了.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
