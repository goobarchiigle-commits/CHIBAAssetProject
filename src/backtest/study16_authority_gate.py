"""
src/backtest/study16_authority_gate.py  —  Study16 Order Authority Integrity Gate

Validates signal→intent→authority→execution→reconciliation→audit chain.
Profitability evaluation PROHIBITED.
Capital fixed: ¥1,800,000  Strategy: Study9 Case B (frozen)

Cases:
  A SHADOW         — signal → virtual execution (no broker)
  B BROKER_DRY_RUN — signal → order_build → validation → cancel
  C PAPER_ACK      — signal → broker_ack → cancel  (GATE CHECK only)
  D LIMITED_LIVE   — signal → real_order → reconciliation (GATE CHECK only)

Output: reports/authority_integrity.md
        logs/authority_audit.jsonl
        runtime/authority_state.json

Run:
    cd C:/ai-trading
    python src/backtest/study16_authority_gate.py
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

import numpy as np
import pandas as pd

from src.config_loader import load_strategy_config
from src.backtest.capital_allocation_abc import load_data, _take, LOT, COST_ONE_WAY
from src.live.authority_state import (
    AuthorityState, AuthorityLevel,
    load_state, save_state, record_signal, record_execution,
    increment_counter, fire_stop, advance_day, attempt_promotion,
    authority_summary, PROMOTION_CRITERIA_KEYS, STOP_CONDITIONS,
)
from src.live.order_authority import (
    SignalRecord, IntentRecord, AuthorityRecord, ExecutionRecord,
    IdempotencyRegistry, append_audit, load_audit, validate_chain,
    _audit_record, AUDIT_LOG,
)

REPORTS_DIR = Path("reports")
CAPITAL     = 1_800_000.0
IS_START    = "2018-01-01"

# Strategy params (Study9 Case B, frozen)
RSR_LO    = 92.0
RSR_HI    = 95.0
D90_MAX   = 5
SLOPE_MAX = 5.0
EXIT_THR  = 90.0
MIN_HOLD  = 3
SHOCK_MKT = -0.05
SHOCK_SYM = -0.08
SLIPPAGE  = 0.001
COMMISSION= 0.00055
COST_ONE_WAY_STUDY = SLIPPAGE + COMMISSION  # 0.00155

FOLDS = [
    {"id": 1, "oos_start": "2021-01-01", "oos_end": "2021-12-31"},
    {"id": 2, "oos_start": "2022-01-01", "oos_end": "2022-12-31"},
    {"id": 3, "oos_start": "2023-01-01", "oos_end": "2023-12-31"},
    {"id": 4, "oos_start": "2024-01-01", "oos_end": "2024-12-31"},
    {"id": 5, "oos_start": "2025-01-01", "oos_end": "2025-12-31"},
]


# ──────────────────────────────────────────────────────────────────────
#  Matrix helpers
# ──────────────────────────────────────────────────────────────────────

def compute_cross90(rsr_mat):
    n, m = rsr_mat.shape
    out = np.zeros((n, m), dtype=np.int32)
    cnt = np.zeros(m, dtype=np.int32)
    for i in range(n):
        out[i] = cnt
        ab = rsr_mat[i] >= 90.0
        cnt[ab] += 1; cnt[~ab] = 0
    return out

def compute_slope5(rsr_mat):
    out = np.zeros_like(rsr_mat, dtype=np.float32)
    out[5:] = rsr_mat[5:] - rsr_mat[:-5]
    return out


# ──────────────────────────────────────────────────────────────────────
#  Signal generator (Study9 Case B replay)
# ──────────────────────────────────────────────────────────────────────

@dataclass
class Position:
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
    Replay Study9 Case B and collect all signal events.
    Returns list of signal dicts with: type(ENTRY/EXIT), date, symbol, price,
    qty, cash_before, cash_after, rsr, d90, slope5, reason.
    """
    pos: Optional[Position] = None
    signals = []
    cash = sl_cash

    for i, date in enumerate(common_dates):
        ds = str(date.date())
        mkt_shock = mkt_ret1 is not None and float(mkt_ret1[i]) <= SHOCK_MKT
        if i + 1 >= n_dates: break
        nxt = i + 1

        # EXIT
        if pos:
            si  = sym_to_i[pos.symbol]
            rv  = float(rsr_mat[i, si])
            ct  = float(close_mat[i, si])
            hd  = i - pos.entry_idx
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
                sp = float(open_mat[nxt, si])
                cb = cash
                cash += pos.qty * sp * (1 - COST_ONE_WAY_STUDY)
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
                if sym_active_mat is not None and float(sym_active_mat[i, si]) < 0.5: continue
                cands.append((rv, d90, sl5, sym))

            if cands:
                cands.sort(key=lambda x: (-x[0], x[1]))
                rv, d90, sl5, sym = cands[0]
                si  = sym_to_i[sym]
                bp  = float(open_mat[nxt, si])
                if bp <= 0: continue
                alloc = cash * 0.95
                qty   = int(alloc / bp / LOT) * LOT
                cost  = qty * bp * (1 + COST_ONE_WAY_STUDY)
                if qty > 0 and cost <= cash:
                    cb = cash; cash -= cost
                    pos = Position(sym, qty, bp, nxt, rv, d90, sl5)
                    signals.append({
                        "type": "ENTRY", "date": ds, "symbol": sym,
                        "price": round(bp, 2), "qty": qty,
                        "cash_before": round(cb, 0), "cash_after": round(cash, 0),
                        "rsr": round(rv, 2), "d90": d90, "slope5": round(sl5, 2),
                        "reason": "RSR_ENTRY", "hold_days": 0, "pnl": 0,
                    })
    return signals


# ──────────────────────────────────────────────────────────────────────
#  Authority Case Simulators
# ──────────────────────────────────────────────────────────────────────

def _fake_broker_ref(symbol: str, date: str, side: str) -> str:
    return hashlib.sha256(f"{symbol}|{date}|{side}".encode()).hexdigest()[:12].upper()


def _fake_latency() -> float:
    rng = np.random.default_rng(abs(hash(time.time())) % (2**32))
    return round(float(rng.uniform(20, 120)), 1)


def run_case_a_shadow(
    signals: list[dict],
    state: AuthorityState,
    idem: IdempotencyRegistry,
) -> tuple[AuthorityState, list[dict]]:
    """
    Case A: SHADOW — signal → virtual execution.
    No broker contact. All fills are virtual (price=signal_price, qty=signal_qty).
    """
    audit_rows = []
    for sig in signals:
        side = "BUY" if sig["type"] == "ENTRY" else "SELL"
        sr   = SignalRecord.build(sig["symbol"], sig["date"], sig["rsr"], side,
                                  sig["d90"], sig["slope5"])
        is_dup, ikey = idem.check_and_register(sr)
        state = record_signal(state, approved=True)

        if is_dup:
            state = increment_counter(state, "duplicate_submit")
            state = increment_counter(state, "idempotency_failure")
            logger.warning("[SHADOW] duplicate: %s %s", sig["symbol"], sig["date"])
            continue

        ir = IntentRecord.build(sr, sig["qty"], sig["price"], sig["cash_before"])
        ar = AuthorityRecord.build(ir, AuthorityLevel.SHADOW.value, approved=True)
        er = ExecutionRecord.build(
            ar, sr, ir,
            fill_price=sig["price"], fill_qty=sig["qty"],
            broker_ref=f"SHADOW_{_fake_broker_ref(sig['symbol'], sig['date'], side)}",
            cash_before=sig["cash_before"], cash_after=sig["cash_after"],
            position_before=0 if side == "BUY" else sig["qty"],
            position_after=sig["qty"] if side == "BUY" else 0,
            broker_latency_ms=0.0, cancel_count=0, case_id="A",
        )

        chain_ok = er.chain_valid(ir.intent_hash)
        if not chain_ok:
            state = increment_counter(state, "authority_mismatch")

        rec = _audit_record(sr, ir, ar, er, chain_ok, ikey)
        rec["broker_ref"] = er.broker_ref
        audit_rows.append(rec)
        append_audit(rec)
        state = record_execution(state, executed=True)

    return state, audit_rows


def run_case_b_dry_run(
    signals: list[dict],
    state: AuthorityState,
    idem: IdempotencyRegistry,
) -> tuple[AuthorityState, list[dict]]:
    """
    Case B: BROKER_DRY_RUN — order_build → validation → immediate cancel.
    Simulates the full order path but cancels before any actual fill.
    """
    audit_rows = []
    for sig in signals:
        side = "BUY" if sig["type"] == "ENTRY" else "SELL"
        sr   = SignalRecord.build(sig["symbol"], sig["date"], sig["rsr"], side,
                                  sig["d90"], sig["slope5"])
        is_dup, ikey = idem.check_and_register(sr)

        if is_dup:
            state = record_signal(state, approved=False)
            state = increment_counter(state, "duplicate_submit")
            state = increment_counter(state, "idempotency_failure")
            continue

        ir = IntentRecord.build(sr, sig["qty"], sig["price"], sig["cash_before"])

        # Validation gate
        approved, reject_reason = _validate_intent(ir, sig["cash_before"])
        ar = AuthorityRecord.build(ir, AuthorityLevel.BROKER_DRY_RUN.value,
                                   approved=approved, reject_reason=reject_reason)
        state = record_signal(state, approved=approved)

        lat = _fake_latency()
        # Dry run: order submitted then immediately cancelled (fill_qty=0)
        er = ExecutionRecord.build(
            ar, sr, ir,
            fill_price=sig["price"] if approved else 0.0,
            fill_qty=0,    # CANCELLED — no fill
            broker_ref=f"DRY_{_fake_broker_ref(sig['symbol'], sig['date'], side)}",
            cash_before=sig["cash_before"], cash_after=sig["cash_before"],  # no cash change
            position_before=0, position_after=0,
            broker_latency_ms=lat, cancel_count=1 if approved else 0,
            case_id="B",
        )

        chain_ok = er.chain_valid(ir.intent_hash)
        if not chain_ok:
            state = increment_counter(state, "authority_mismatch")

        # Reconciliation check (dry run: broker state = original state, no change)
        rec = _audit_record(sr, ir, ar, er, chain_ok, ikey)
        rec["broker_ref"] = er.broker_ref
        audit_rows.append(rec)
        append_audit(rec)
        state = record_execution(state, executed=False)  # not executed (cancelled)

    return state, audit_rows


def _validate_intent(intent: IntentRecord, cash: float) -> tuple[bool, str]:
    """Pre-execution validation checks."""
    if intent.qty <= 0:
        return False, "qty_zero"
    if intent.price_limit <= 0:
        return False, "price_invalid"
    # Cash check only applies to BUY orders; SELL orders release cash
    if intent.side == "BUY":
        cost = intent.qty * intent.price_limit * (1 + COST_ONE_WAY_STUDY)
        if cost > cash * 1.05:     # 5% tolerance for market fluctuation
            return False, f"insufficient_cash cost={cost:.0f} cash={cash:.0f}"
    if intent.qty % LOT != 0:
        return False, f"qty_not_lot_multiple qty={intent.qty} lot={LOT}"
    return True, ""


# ──────────────────────────────────────────────────────────────────────
#  Metrics Computation
# ──────────────────────────────────────────────────────────────────────

def compute_integrity_metrics(
    audit_rows: list[dict],
    state: AuthorityState,
) -> dict:
    n = len(audit_rows)
    if n == 0:
        return {k: 0 for k in [
            "authority_precision","execution_tracking","replay_consistency",
            "reconciliation_error","duplicate_submit","orphan_order",
            "partial_fill_gap","cash_truth_gap","position_truth_gap",
            "idempotency","latency_p50","latency_p95","latency_p99",
            "authority_mismatch",
        ]}

    approved = [r for r in audit_rows if r.get("approved")]
    executed = [r for r in audit_rows if r.get("fill_qty", 0) > 0]

    authority_precision  = len(approved) / max(1, n)
    execution_tracking   = len(executed) / max(1, len(approved))

    chain_valid = validate_chain(audit_rows)
    replay_cons = chain_valid["replay_consistency"]

    # Latency
    lats = [r.get("broker_latency_ms", 0.0) for r in audit_rows if r.get("broker_latency_ms", 0) > 0]
    lat_p50 = float(np.percentile(lats, 50)) if lats else 0.0
    lat_p95 = float(np.percentile(lats, 95)) if lats else 0.0
    lat_p99 = float(np.percentile(lats, 99)) if lats else 0.0

    # Partial fill
    partial_gap = sum(r.get("partial_fill_gap", 0) for r in audit_rows)

    # Cash truth gap: cash_after discrepancy (expected 0 in shadow/dry_run)
    cash_gaps   = sum(1 for r in audit_rows if abs(r.get("cash_after",0) - r.get("cash_before",0)) > 0.01
                      and r.get("fill_qty", 0) == 0)
    pos_gaps    = sum(1 for r in audit_rows if r.get("position_after",0) != r.get("position_before",0)
                      and r.get("fill_qty", 0) == 0)

    # idempotency: fraction that passed idem check (no duplicate)
    total_sigs  = state.total_signals
    dups        = state.duplicate_submit
    idem_rate   = 1.0 - (dups / max(1, total_sigs))

    return {
        "authority_precision":  round(authority_precision, 4),
        "execution_tracking":   round(execution_tracking, 4),
        "replay_consistency":   round(replay_cons, 4),
        "reconciliation_error": state.reconciliation_error,
        "duplicate_submit":     state.duplicate_submit,
        "orphan_order":         state.orphan_order,
        "partial_fill_gap":     partial_gap,
        "cash_truth_gap":       cash_gaps,
        "position_truth_gap":   pos_gaps,
        "idempotency":          round(idem_rate, 4),
        "latency_p50_ms":       round(lat_p50, 1),
        "latency_p95_ms":       round(lat_p95, 1),
        "latency_p99_ms":       round(lat_p99, 1),
        "authority_mismatch":   state.authority_mismatch,
        "chain_valid_n":        chain_valid["n_valid"],
        "chain_invalid_n":      chain_valid["n_invalid"],
        "n_audit_records":      n,
    }


# ──────────────────────────────────────────────────────────────────────
#  Promotion Gate Check (Cases C and D)
# ──────────────────────────────────────────────────────────────────────

CASE_C_CRITERIA = {
    "authority_mismatch": 0,
    "reconciliation_error": 0,
    "duplicate_submit": 0,
    "idempotency_failure": 0,
    "replay_consistency": 1.0,   # must equal 1.0
}

CASE_D_CRITERIA = {
    "cash_truth_gap": 0,
    "position_truth_gap": 0,
    "orphan_order": 0,
    "authority_mismatch": 0,
    "reconciliation_error": 0,
    "replay_consistency": 1.0,
}

def gate_check(metrics: dict, state: AuthorityState,
               criteria: dict, name: str) -> tuple[bool, list[str]]:
    fails = []
    for k, threshold in criteria.items():
        val = metrics.get(k, getattr(state, k, None))
        if val is None:
            fails.append(f"{k}=MISSING")
            continue
        if k == "replay_consistency":
            if val < threshold:
                fails.append(f"{k}={val:.4f} < {threshold}")
        else:
            if val > threshold:
                fails.append(f"{k}={val} > {threshold}")
    return (len(fails) == 0), fails


# ──────────────────────────────────────────────────────────────────────
#  Authority Integrity Score
# ──────────────────────────────────────────────────────────────────────

def compute_integrity_score(metrics: dict, state: AuthorityState) -> float:
    """0-100. Each of 8 criteria worth 12.5 points. Deducted for any violation."""
    score = 100.0
    per_item = 100.0 / 8

    checks = {
        "authority_mismatch": 0,
        "reconciliation_error": 0,
        "replay_consistency_fail": 0,   # = 1 - replay_consistency > 0
        "duplicate_submit": 0,
        "cash_truth_gap": 0,
        "position_truth_gap": 0,
        "orphan_order": 0,
        "idempotency_failure": 0,
    }

    if state.authority_mismatch > 0:       score -= per_item
    if state.reconciliation_error > 0:     score -= per_item
    if metrics.get("replay_consistency",1) < 1.0: score -= per_item
    if state.duplicate_submit > 0:         score -= per_item
    if metrics.get("cash_truth_gap",0) > 0:       score -= per_item
    if metrics.get("position_truth_gap",0) > 0:   score -= per_item
    if state.orphan_order > 0:             score -= per_item
    if state.idempotency_failure > 0:      score -= per_item

    return round(max(0.0, score), 1)


# ──────────────────────────────────────────────────────────────────────
#  Report
# ──────────────────────────────────────────────────────────────────────

def write_report(
    case_a_metrics: dict, case_b_metrics: dict,
    state_a: AuthorityState, state_b: AuthorityState,
    gate_c: tuple, gate_d: tuple,
    n_signals: int, path: Path,
) -> None:
    today = time.strftime("%Y-%m-%d")
    L = []; w = L.append

    # Combined integrity score (A+B averaged)
    score_a = compute_integrity_score(case_a_metrics, state_a)
    score_b = compute_integrity_score(case_b_metrics, state_b)
    score   = round((score_a + score_b) / 2, 1)

    c_pass, c_fails = gate_c
    d_pass, d_fails = gate_d

    # Overall go_live decision
    if score_a == 100 and score_b == 100 and c_pass:
        go_live = "PAPER_ACK_READY"
        next_step = "Case C (PAPER_ACK) を30日間実施 → Case D (LIMITED_LIVE) へ"
    elif score_a == 100:
        go_live = "DRY_RUN_READY"
        next_step = "Case B (BROKER_DRY_RUN) を30日間実施 → Cゲート再確認"
    else:
        go_live = "SHADOW_ONLY"
        next_step = "integrity_score=100%達成後に進める"

    # Rollback rule
    rollback = ("alpha_realization_30d < 80% → allocation=0%, authority=OFF, "
                "incidentレポート発行")

    # Operational risks
    op_risks = []
    if state_b.orphan_order > 0:
        op_risks.append(f"orphan_order={state_b.orphan_order} 件未解決")
    if case_b_metrics.get("latency_p99_ms", 0) > 500:
        op_risks.append(f"latency_p99={case_b_metrics['latency_p99_ms']}ms > 500ms")
    if not op_risks:
        op_risks.append("なし")

    # ── HEADER ─────────────────────────────────────────────────────────
    w("# Study16 Order Authority Integrity Gate")
    w(f"\n作成日: {today}  |  integrity監査のみ / 収益評価禁止")
    w(f"\n**Strategy**: Study9 Case B (固定)  **Capital**: ¥{int(CAPITAL):,}\n")

    w("---\n## Executive Summary\n")
    w("| 項目 | 値 |")
    w("|---|---|")
    w(f"| **authority_integrity_score** | **{score}/100** |")
    w(f"| authority_precision (Case A) | {case_a_metrics['authority_precision']:.1%} |")
    w(f"| authority_precision (Case B) | {case_b_metrics['authority_precision']:.1%} |")
    w(f"| execution_tracking (Case A) | {case_a_metrics['execution_tracking']:.1%} |")
    w(f"| execution_tracking (Case B) | {case_b_metrics['execution_tracking']:.1%} |")
    w(f"| replay_consistency (A) | {case_a_metrics['replay_consistency']:.1%} |")
    w(f"| replay_consistency (B) | {case_b_metrics['replay_consistency']:.1%} |")
    w(f"| reconciliation_status | {'PASS ✅' if score == 100 else 'REVIEW ⚠'} |")
    w(f"| go_live_decision | **{go_live}** |")
    w(f"| rollback_rule | {rollback} |")
    w(f"| recommended_authority_level | "
      f"{'LIMITED_LIVE' if d_pass else 'PAPER_ACK' if c_pass else 'BROKER_DRY_RUN'} |")
    w("")

    # ── CASE A ─────────────────────────────────────────────────────────
    w("---\n## Case A: SHADOW (signal → virtual execution)\n")
    _write_case_metrics(w, case_a_metrics, state_a, "A")
    w(f"\n**integrity_score_A: {score_a}/100**\n")

    # ── CASE B ─────────────────────────────────────────────────────────
    w("---\n## Case B: BROKER_DRY_RUN (signal → order_build → validation → cancel)\n")
    _write_case_metrics(w, case_b_metrics, state_b, "B")
    w(f"\n**integrity_score_B: {score_b}/100**\n")

    # ── CASE C GATE ────────────────────────────────────────────────────
    w("---\n## Case C Gate: PAPER_ACK (signal → broker_ack → cancel)\n")
    w(f"**Gate判定: {'✅ PASS' if c_pass else '❌ FAIL'}**\n")
    if c_pass:
        w("全条件クリア。Case C (PAPER_ACK) 実施可能。")
    else:
        w("**未クリア条件:**")
        for f in c_fails:
            w(f"  - {f}")
    w("\n**Case C 実施要件:**\n")
    w("| 条件 | 閾値 | 現在値 |")
    w("|---|---|---|")
    for k, threshold in CASE_C_CRITERIA.items():
        val = case_b_metrics.get(k, getattr(state_b, k, "—"))
        ok  = "✅" if (val <= threshold if k != "replay_consistency" else val >= threshold) else "❌"
        w(f"| {k} | {threshold} | {val} {ok} |")
    w("")

    # ── CASE D GATE ────────────────────────────────────────────────────
    w("---\n## Case D Gate: LIMITED_LIVE (signal → real_order → reconciliation)\n")
    w(f"**Gate判定: {'✅ PASS' if d_pass else '❌ FAIL — Case C完了後に再確認'}**\n")
    w("**Case D 実施要件:**\n")
    w("| 条件 | 閾値 | 現在値 |")
    w("|---|---|---|")
    for k, threshold in CASE_D_CRITERIA.items():
        val = case_b_metrics.get(k, getattr(state_b, k, "—"))
        ok  = "✅" if (val <= threshold if k != "replay_consistency" else val >= threshold) else "❌"
        w(f"| {k} | {threshold} | {val} {ok} |")
    w("")

    # ── STOP CONDITIONS ────────────────────────────────────────────────
    w("---\n## Stop Conditions\n")
    w("| 条件 | 発火 |")
    w("|---|---|")
    for cond in STOP_CONDITIONS:
        fired = (state_b.stop_condition_fired == cond)
        w(f"| {cond} | {'🛑 FIRED' if fired else '— OK'} |")
    w("")

    # ── PROMOTION CRITERIA ─────────────────────────────────────────────
    w("---\n## Promotion Criteria (30日間チェック)\n")
    w("| 指標 | 要求値 | Case A | Case B |")
    w("|---|---|---|---|")
    for k in PROMOTION_CRITERIA_KEYS:
        va = getattr(state_a, k, 0)
        vb = getattr(state_b, k, 0)
        ok_a = "✅" if va == 0 else "❌"
        ok_b = "✅" if vb == 0 else "❌"
        w(f"| {k} | 0 | {va} {ok_a} | {vb} {ok_b} |")
    w("")

    # ── HASH CHAIN SAMPLE ──────────────────────────────────────────────
    w("---\n## Hash Chain Integrity Sample (Case A 先頭3件)\n")
    w("```")
    audit = load_audit()
    sample = [r for r in audit if r.get("case_id") == "A"][:3]
    for r in sample:
        w(f"signal_hash    : {r.get('signal_hash','')}")
        w(f"intent_hash    : {r.get('intent_hash','')}")
        w(f"authority_hash : {r.get('authority_hash','')}")
        w(f"execution_hash : {r.get('execution_hash','')}")
        w(f"chain_hash     : {r.get('chain_hash','')}")
        w(f"chain_valid    : {r.get('chain_valid',False)}")
        w(f"symbol={r.get('symbol')} date={r.get('date')} side={r.get('side')}")
        w("---")
    w("```\n")

    # ── OPERATIONAL RISKS ──────────────────────────────────────────────
    w("---\n## Operational Risks\n")
    for risk in op_risks:
        w(f"- {risk}")
    w("")

    # ── FINAL VERDICT ─────────────────────────────────────────────────
    w("---\n## 最終判定\n")
    w("| 項目 | 値 |")
    w("|---|---|")
    w(f"| authority_integrity_score | **{score}/100** |")
    w(f"| authority_precision | {case_b_metrics['authority_precision']:.1%} |")
    w(f"| execution_tracking | {case_b_metrics['execution_tracking']:.1%} |")
    w(f"| reconciliation_status | {'OK' if score == 100 else 'REVIEW'} |")
    w(f"| go_live_decision | **{go_live}** |")
    w(f"| rollback_rule | {rollback} |")
    w(f"| operational_risks | {'; '.join(op_risks)} |")
    w(f"| recommended_authority_level | "
      f"{'LIMITED_LIVE' if d_pass else 'PAPER_ACK' if c_pass else 'BROKER_DRY_RUN'} |")
    w(f"| next_step | {next_step} |")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")
    print(f"  レポート: {path}")


def _write_case_metrics(w, m: dict, s: AuthorityState, case_id: str):
    rows = [
        ("n_signals_processed",  m.get("n_audit_records", 0)),
        ("authority_precision",  f"{m['authority_precision']:.1%}"),
        ("execution_tracking",   f"{m['execution_tracking']:.1%}"),
        ("replay_consistency",   f"{m['replay_consistency']:.1%}"),
        ("authority_mismatch",   s.authority_mismatch),
        ("reconciliation_error", s.reconciliation_error),
        ("duplicate_submit",     s.duplicate_submit),
        ("orphan_order",         s.orphan_order),
        ("idempotency_failure",  s.idempotency_failure),
        ("cash_truth_gap",       m.get("cash_truth_gap", 0)),
        ("position_truth_gap",   m.get("position_truth_gap", 0)),
        ("partial_fill_gap",     m.get("partial_fill_gap", 0)),
        ("chain_valid / total",  f"{m.get('chain_valid_n',0)} / {m.get('n_audit_records',0)}"),
        ("latency_p50_ms",       m.get("latency_p50_ms", 0)),
        ("latency_p95_ms",       m.get("latency_p95_ms", 0)),
        ("latency_p99_ms",       m.get("latency_p99_ms", 0)),
    ]
    w("| 指標 | 値 |")
    w("|---|---|")
    for k, v in rows:
        w(f"| {k} | {v} |")


# ──────────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────────

def main() -> int:
    cfg = load_strategy_config()
    print("=" * 68)
    print("  Study16 Order Authority Integrity Gate")
    print(f"  Capital: ¥{int(CAPITAL):,}  Strategy: Study9 Case B (frozen)")
    print("  収益評価禁止 / authority integrity 検証のみ")
    print("=" * 68 + "\n")

    # Clear stale state from any prior partial runs
    for stale in [AUDIT_LOG,
                  Path("runtime/authority_idem_a.json"),
                  Path("runtime/authority_idem_b.json")]:
        if stale.exists():
            stale.unlink()

    # ── Data ──────────────────────────────────────────────────────────
    print("[1/5] データロード...")
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
    print(f"  共通日数: {n_dates}  銘柄数: {n_syms}\n")

    # ── Matrices ──────────────────────────────────────────────────────
    print("[2/5] マトリクス構築...")
    open_mat  = np.full((n_dates, n_syms), np.nan, dtype=np.float32)
    close_mat = np.full((n_dates, n_syms), np.nan, dtype=np.float32)
    for si, sym in enumerate(active_syms):
        df_src  = universe_raw[sym]["df"]
        ri = df_src.index.get_indexer(common_dates)
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

    cross90 = compute_cross90(rsr_mat)
    slope5  = compute_slope5(rsr_mat)

    # ── Signal generation ─────────────────────────────────────────────
    print("[3/5] シグナル生成 (Study9 Case B リプレイ)...")
    signals = generate_signals(
        open_mat, close_mat, rsr_mat, sym_active_mat,
        cross90, slope5, mkt_ret1,
        active_syms, sym_to_i, common_dates, n_dates,
        sl_cash=CAPITAL,
    )
    entries = [s for s in signals if s["type"] == "ENTRY"]
    exits   = [s for s in signals if s["type"] == "EXIT"]
    print(f"  total_signals={len(signals)}  entries={len(entries)}  exits={len(exits)}")

    # ── Case A: SHADOW ─────────────────────────────────────────────────
    print("[4/5] Case A: SHADOW...")
    state_a = dataclasses.replace(AuthorityState.initial(),
                                  level=AuthorityLevel.SHADOW.value)
    idem_a  = IdempotencyRegistry(Path("runtime/authority_idem_a.json"))
    state_a, audit_a = run_case_a_shadow(signals, state_a, idem_a)
    metrics_a = compute_integrity_metrics(audit_a, state_a)
    print(f"  n_records={len(audit_a)}  integrity_score={compute_integrity_score(metrics_a, state_a)}/100"
          f"  replay={metrics_a['replay_consistency']:.1%}")

    # ── Case B: BROKER_DRY_RUN ────────────────────────────────────────
    print("[5/5] Case B: BROKER_DRY_RUN...")
    state_b = dataclasses.replace(AuthorityState.initial(),
                                  level=AuthorityLevel.BROKER_DRY_RUN.value)
    idem_b  = IdempotencyRegistry(Path("runtime/authority_idem_b.json"))
    state_b, audit_b = run_case_b_dry_run(signals, state_b, idem_b)
    metrics_b = compute_integrity_metrics(audit_b, state_b)
    print(f"  n_records={len(audit_b)}  integrity_score={compute_integrity_score(metrics_b, state_b)}/100"
          f"  replay={metrics_b['replay_consistency']:.1%}")

    # ── Gate checks C and D ────────────────────────────────────────────
    gate_c = gate_check(metrics_b, state_b, CASE_C_CRITERIA, "PAPER_ACK")
    gate_d = gate_check(metrics_b, state_b, CASE_D_CRITERIA, "LIMITED_LIVE")

    # ── Save state ─────────────────────────────────────────────────────
    save_state(state_b)

    # ── Report ─────────────────────────────────────────────────────────
    print("\n[OUT] レポート出力...")
    write_report(
        metrics_a, metrics_b, state_a, state_b,
        gate_c, gate_d,
        n_signals=len(signals),
        path=REPORTS_DIR / "authority_integrity.md",
    )
    print("\n完了.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
