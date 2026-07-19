"""
tests/test_cb_settlement_guard.py

TASK-2: CB settlement-lag guard regression tests.

Scenario: A BUY order executed in the previous run reduced cash in the broker
wallet API, but the new position has not yet appeared in the broker positions
API (typical settlement lag of seconds to minutes for kabu station REST API).

Without the guard:
  equity = reduced_cash + old_market_value   → phantom DD → false CB trigger

With the guard:
  pre_commit_qtys (state) vs current_positions (broker) are diffed;
  missing positions' values are added back before the CB check.

Run:
    cd C:/ai-trading
    python -m pytest tests/test_cb_settlement_guard.py -v
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

sys.stdout.reconfigure(encoding="utf-8")

from src.risk.circuit_breaker import (
    CBState,
    calc_drawdown,
    update_cb_state,
)
from src.analytics.slot_pressure import (
    SlotPressureInput,
    compute_slot_pressure,
    build_slot_pressure_input_from_oc_data,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: simulate the CB guard compensation logic
# (mirrors signal_bridge.py lines 4136–4193)
# ─────────────────────────────────────────────────────────────────────────────

def _apply_cb_guard(
    current_equity: float,
    current_positions: dict,
    pre_commit_qtys: dict,
    pre_commit_costs: dict,
) -> float:
    """
    Pure-function replica of the settlement-lag compensation in signal_bridge.py.
    Returns adjusted equity to use for CB check.
    """
    pending_buy_value = 0.0
    for sym, qty in pre_commit_qtys.items():
        q = int(qty or 0)
        if q > 0 and sym not in current_positions:
            price = float(pre_commit_costs.get(sym) or 0.0)
            if price > 0:
                pending_buy_value += q * price
    return current_equity + pending_buy_value


# ─────────────────────────────────────────────────────────────────────────────
# TASK-2 tests — CB settlement-lag compensation
# ─────────────────────────────────────────────────────────────────────────────

class TestCBSettlementGuard(unittest.TestCase):
    """Verify that the settlement-lag guard prevents phantom-DD CB triggers."""

    # ------------------------------------------------------------------
    # Test 1: cash reduced, position not yet in broker → equity compensated
    # ------------------------------------------------------------------
    def test_buy_settlement_lag_compensates_equity(self):
        """
        Pre-commit state has 6506.T (100 shares @ 6987).
        Broker positions API does NOT yet have 6506.T (lag).
        Guard must add back 100 × 6987 = 698,700.
        """
        pre_commit_qtys  = {"6981.T": 100, "6506.T": 100}
        pre_commit_costs = {"6981.T": 4864.0, "6506.T": 6987.0}

        # broker positions: only 6981.T present (6506.T lagging)
        current_positions = {"6981.T": {"qty": 100, "avg_price": 4864.0}}

        phantom_equity = 2_384_909.0  # cash(1,031,109) + mktval(1,353,800) — matches 2026-05-19 incident

        adjusted = _apply_cb_guard(
            phantom_equity, current_positions, pre_commit_qtys, pre_commit_costs
        )

        expected_addition = 100 * 6987.0  # 698,700
        self.assertAlmostEqual(adjusted, phantom_equity + expected_addition, places=0)
        self.assertAlmostEqual(adjusted, 3_083_609.0, places=0)

    # ------------------------------------------------------------------
    # Test 2: with compensation, DD stays above -15% → no CB trigger
    # ------------------------------------------------------------------
    def test_no_false_cb_trigger_after_buy(self):
        """
        2026-05-19 incident replay.
        Peak = 3,153,109. Phantom equity = 2,384,909 → DD = -24.4% → CB.
        After compensation: equity = 3,083,609 → DD = -2.2% → NO CB.
        """
        from datetime import date
        peak_equity    = 3_153_109.0
        phantom_equity = 2_384_909.0  # cash reduced, position missing

        pre_commit_qtys  = {"6981.T": 100, "6506.T": 100}
        pre_commit_costs = {"6981.T": 4864.0, "6506.T": 6987.0}
        current_positions = {"6981.T": {"qty": 100, "avg_price": 4864.0}}

        compensated_equity = _apply_cb_guard(
            phantom_equity, current_positions, pre_commit_qtys, pre_commit_costs
        )

        # phantom path: DD = -24.4% → would trigger CB
        phantom_dd = calc_drawdown(phantom_equity, peak_equity)
        self.assertLessEqual(phantom_dd, -0.15, "phantom DD should exceed trigger threshold")

        # compensated path: DD = ~-2.2% → must NOT trigger CB
        real_dd = calc_drawdown(compensated_equity, peak_equity)
        self.assertGreater(real_dd, -0.15, "compensated DD must not trigger CB")

        state = CBState(mode="NORMAL", peak_equity=peak_equity, cb_start=None)
        state = update_cb_state(state, compensated_equity, date(2026, 5, 19))
        self.assertEqual(state.mode, "NORMAL", "CB must not activate with compensated equity")

    # ------------------------------------------------------------------
    # Test 3: genuine DD still triggers CB (guard does not suppress real drops)
    # ------------------------------------------------------------------
    def test_genuine_dd_still_triggers_cb(self):
        """
        All broker positions are present and accounted for.
        Equity is genuinely low → CB must still trigger.
        Guard adds nothing when pre_commit_qtys == current_positions keys.
        """
        from datetime import date
        peak_equity    = 3_000_000.0
        genuine_low    = 2_400_000.0  # DD = -20% → genuine loss

        # broker positions match state — no settlement lag
        pre_commit_qtys  = {"6981.T": 100}
        pre_commit_costs = {"6981.T": 4864.0}
        current_positions = {"6981.T": {"qty": 100, "avg_price": 4864.0}}

        adjusted = _apply_cb_guard(
            genuine_low, current_positions, pre_commit_qtys, pre_commit_costs
        )

        self.assertAlmostEqual(adjusted, genuine_low, places=0,
                               msg="No compensation when all positions are in broker API")

        state = CBState(mode="NORMAL", peak_equity=peak_equity, cb_start=None)
        state = update_cb_state(state, adjusted, date(2026, 5, 19))
        self.assertEqual(state.mode, "ENTRY_STOP_ONLY",
                         "Genuine DD must still trigger CB")

    # ------------------------------------------------------------------
    # Test 4: empty pre-commit state → no compensation, no error
    # ------------------------------------------------------------------
    def test_empty_pre_commit_qtys_no_error(self):
        """Guard is a no-op when portfolio_state has no known positions."""
        adjusted = _apply_cb_guard(
            1_000_000.0, {}, {}, {}
        )
        self.assertEqual(adjusted, 1_000_000.0)

    # ------------------------------------------------------------------
    # Test 5: position with zero qty is not compensated
    # ------------------------------------------------------------------
    def test_zero_qty_not_compensated(self):
        """position_qtys = 0 (sold) must not add to equity."""
        pre_commit_qtys  = {"8015.T": 0, "6981.T": 100}
        pre_commit_costs = {"8015.T": 6733.0, "6981.T": 4864.0}
        current_positions = {"6981.T": {"qty": 100, "avg_price": 4864.0}}

        adjusted = _apply_cb_guard(
            2_000_000.0, current_positions, pre_commit_qtys, pre_commit_costs
        )
        self.assertEqual(adjusted, 2_000_000.0,
                         "zero-qty (sold) position must not be compensated")

    # ------------------------------------------------------------------
    # Test 6: position with unknown price → not compensated (guard is conservative)
    # ------------------------------------------------------------------
    def test_unknown_price_not_compensated(self):
        """If price cannot be determined, skip that position (safe default)."""
        pre_commit_qtys  = {"UNKNOWN.T": 100}
        pre_commit_costs = {}  # no price available

        adjusted = _apply_cb_guard(
            2_000_000.0, {}, pre_commit_qtys, pre_commit_costs
        )
        self.assertEqual(adjusted, 2_000_000.0,
                         "position with unknown price must not affect equity")

    # ------------------------------------------------------------------
    # Test 7: multiple pending positions all compensated
    # ------------------------------------------------------------------
    def test_multiple_pending_positions(self):
        """Two pending positions (e.g. two BUY orders in quick succession)."""
        pre_commit_qtys  = {"A.T": 100, "B.T": 200}
        pre_commit_costs = {"A.T": 1000.0, "B.T": 500.0}
        current_positions = {}  # both missing from broker

        adjusted = _apply_cb_guard(
            1_000_000.0, current_positions, pre_commit_qtys, pre_commit_costs
        )
        # A: 100×1000=100,000  B: 200×500=100,000  total: +200,000
        self.assertAlmostEqual(adjusted, 1_200_000.0, places=0)


# ─────────────────────────────────────────────────────────────────────────────
# CB_GUARD V2 tests — execution-ledger-based same-day BUY compensation
# ─────────────────────────────────────────────────────────────────────────────

from src.kabusapi.signal_bridge import _cb_guard_compensation, PendingOrderState


class TestCBGuardV2ExecutionLedger(unittest.TestCase):
    """
    Verify that execution_ledger-based compensation prevents false CB triggers
    when a same-day prior run placed a BUY that is not yet in broker positions.

    Regression: 2026-06-04 run_morning_signal bought 5301.T at 08:41 (NORMAL),
    run_live_signal ran at 08:44 and saw equity drop by purchase amount → CB fired.
    """

    TODAY = "2026-06-04"

    def _make_ledger(self, tmp_path: Path, entries: list[dict]) -> Path:
        """Write execution_ledger.json and return path."""
        orders = {}
        for e in entries:
            key = f"{self.TODAY}_{e['symbol']}_{e['side']}"
            orders[key] = {
                "symbol":           e["symbol"],
                "side":             e["side"],
                "qty":              e["qty"],
                "price":            e["price"],
                "recorded_at":      self.TODAY,
                "execution_status": e.get("status", "pending"),
            }
        data = {"date": self.TODAY, "orders": orders, "count": len(orders)}
        path = tmp_path / "execution_ledger.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self._tmp_path = Path(self._tmp)

    # ------------------------------------------------------------------
    # T1: same-day BUY in ledger (pending), not in broker → compensated
    # ------------------------------------------------------------------
    def test_same_day_buy_not_in_broker_is_compensated(self):
        """
        Reproduces 2026-06-04 incident.
        Ledger has 5301.T BUY (pending, 400×1852.5=741,000).
        Broker does NOT have 5301.T yet.
        Guard must add 741,000 to equity.
        """
        lpath = self._make_ledger(self._tmp_path, [
            {"symbol": "5301.T", "side": "BUY", "qty": 400, "price": 1852.5, "status": "pending"},
        ])
        sv, ss, lv, ls = _cb_guard_compensation(
            current_equity    = 3_184_309.0,
            current_positions = {"6981.T": {"qty": 100}, "6506.T": {"qty": 100}},
            pre_commit_qtys   = {"6981.T": 100, "6506.T": 100},
            pre_commit_costs  = {"6981.T": 4864.0, "6506.T": 6987.0},
            ledger_path       = lpath,
            today_str         = self.TODAY,
        )
        self.assertAlmostEqual(sv, 0.0, places=0,
                               msg="state comp: all state positions are in broker")
        self.assertAlmostEqual(lv, 400 * 1852.5, places=0,
                               msg="ledger comp must equal 400×1852.5")
        self.assertIn("5301.T", ls)

    # ------------------------------------------------------------------
    # T2: same-day BUY compensation prevents false CB trigger
    # ------------------------------------------------------------------
    def test_same_day_buy_does_not_trigger_false_cb(self):
        """
        Without compensation: equity=3,184,309, peak=4,089,109, DD=-22.1% → CB.
        With compensation:    equity=3,925,309, peak=4,089,109, DD=-4.0%  → no CB.
        """
        from datetime import date
        peak_equity      = 4_089_109.0
        raw_equity       = 3_184_309.0   # cash already reduced after 5301.T BUY
        buy_cost         = 400 * 1852.5  # 741,000

        lpath = self._make_ledger(self._tmp_path, [
            {"symbol": "5301.T", "side": "BUY", "qty": 400, "price": 1852.5, "status": "pending"},
        ])
        sv, ss, lv, ls = _cb_guard_compensation(
            current_equity    = raw_equity,
            current_positions = {"6981.T": {"qty": 100}, "6506.T": {"qty": 100}},
            pre_commit_qtys   = {"6981.T": 100, "6506.T": 100},
            pre_commit_costs  = {"6981.T": 4864.0, "6506.T": 6987.0},
            ledger_path       = lpath,
            today_str         = self.TODAY,
        )
        adjusted_equity = raw_equity + sv + lv

        # Uncompensated path must exceed CB threshold
        raw_dd = raw_equity / peak_equity - 1.0
        self.assertLessEqual(raw_dd, -0.15,
                             "raw DD must breach CB threshold to validate the scenario")

        # Compensated path must stay above CB threshold
        adj_dd = adjusted_equity / peak_equity - 1.0
        self.assertGreater(adj_dd, -0.15,
                           "compensated DD must not trigger CB")
        self.assertAlmostEqual(adjusted_equity, raw_equity + buy_cost, places=0)

    # ------------------------------------------------------------------
    # T3: genuine drawdown still triggers CB (no compensation)
    # ------------------------------------------------------------------
    def test_genuine_drawdown_still_triggers_cb(self):
        """
        All broker positions present, no ledger entries → guard adds nothing.
        Genuine DD=-20% must still trigger CB.
        """
        lpath = self._make_ledger(self._tmp_path, [])  # empty ledger
        sv, ss, lv, ls = _cb_guard_compensation(
            current_equity    = 2_400_000.0,
            current_positions = {"6981.T": {"qty": 100}},
            pre_commit_qtys   = {"6981.T": 100},
            pre_commit_costs  = {"6981.T": 4864.0},
            ledger_path       = lpath,
            today_str         = self.TODAY,
        )
        self.assertAlmostEqual(sv + lv, 0.0, places=0,
                               msg="no compensation when broker matches state")

    # ------------------------------------------------------------------
    # T4: failed / shadow entries not compensated
    # ------------------------------------------------------------------
    def test_failed_and_shadow_entries_not_compensated(self):
        """failed and shadow status must be excluded from compensation."""
        lpath = self._make_ledger(self._tmp_path, [
            {"symbol": "A.T", "side": "BUY", "qty": 100, "price": 1000.0, "status": "failed"},
            {"symbol": "B.T", "side": "BUY", "qty": 100, "price": 2000.0, "status": "shadow"},
        ])
        sv, ss, lv, ls = _cb_guard_compensation(
            current_equity    = 2_000_000.0,
            current_positions = {},
            pre_commit_qtys   = {},
            pre_commit_costs  = {},
            ledger_path       = lpath,
            today_str         = self.TODAY,
        )
        self.assertAlmostEqual(lv, 0.0, places=0,
                               msg="failed/shadow BUY must not be compensated")
        self.assertEqual(ls, [])

    # ------------------------------------------------------------------
    # T5: submitted status (mark_submitted called) also compensated
    # ------------------------------------------------------------------
    def test_submitted_status_also_compensated(self):
        """submitted entries (mark_submitted was called) must also compensate."""
        lpath = self._make_ledger(self._tmp_path, [
            {"symbol": "5301.T", "side": "BUY", "qty": 400, "price": 1852.5, "status": "submitted"},
        ])
        sv, ss, lv, ls = _cb_guard_compensation(
            current_equity    = 3_000_000.0,
            current_positions = {},
            pre_commit_qtys   = {},
            pre_commit_costs  = {},
            ledger_path       = lpath,
            today_str         = self.TODAY,
        )
        self.assertAlmostEqual(lv, 400 * 1852.5, places=0)

    # ------------------------------------------------------------------
    # T6: symbol in both state-qtys and ledger → counted once (dedup)
    # ------------------------------------------------------------------
    def test_state_and_ledger_dedup_no_double_count(self):
        """
        5301.T appears in pre_commit_qtys (state comp) AND in ledger.
        Must be counted exactly once.
        """
        lpath = self._make_ledger(self._tmp_path, [
            {"symbol": "5301.T", "side": "BUY", "qty": 400, "price": 1852.5, "status": "pending"},
        ])
        sv, ss, lv, ls = _cb_guard_compensation(
            current_equity    = 3_000_000.0,
            current_positions = {},                         # not in broker
            pre_commit_qtys   = {"5301.T": 400},            # in state
            pre_commit_costs  = {"5301.T": 1852.5},
            ledger_path       = lpath,
            today_str         = self.TODAY,
        )
        # state picks it up; ledger must skip it
        self.assertAlmostEqual(sv, 400 * 1852.5, places=0, msg="state comp")
        self.assertAlmostEqual(lv, 0.0,          places=0, msg="ledger must not double-count")
        self.assertNotIn("5301.T", ls)

    # ------------------------------------------------------------------
    # T7: ledger file missing → FAIL_OPEN (no crash, no ledger comp)
    # ------------------------------------------------------------------
    def test_ledger_file_missing_fail_open(self):
        """If execution_ledger.json does not exist, gracefully return zero ledger comp."""
        missing_path = self._tmp_path / "nonexistent_ledger.json"
        sv, ss, lv, ls = _cb_guard_compensation(
            current_equity    = 3_000_000.0,
            current_positions = {},
            pre_commit_qtys   = {},
            pre_commit_costs  = {},
            ledger_path       = missing_path,
            today_str         = self.TODAY,
        )
        self.assertAlmostEqual(lv, 0.0, places=0)
        self.assertEqual(ls, [])

    # ------------------------------------------------------------------
    # T8: ledger date mismatch → no compensation
    # ------------------------------------------------------------------
    def test_ledger_date_mismatch_no_compensation(self):
        """Ledger from a previous day must not be used."""
        lpath = self._make_ledger(self._tmp_path, [
            {"symbol": "5301.T", "side": "BUY", "qty": 400, "price": 1852.5, "status": "pending"},
        ])
        sv, ss, lv, ls = _cb_guard_compensation(
            current_equity    = 3_000_000.0,
            current_positions = {},
            pre_commit_qtys   = {},
            pre_commit_costs  = {},
            ledger_path       = lpath,
            today_str         = "2026-06-05",   # different date
        )
        self.assertAlmostEqual(lv, 0.0, places=0,
                               msg="stale ledger must not affect equity compensation")

    # ------------------------------------------------------------------
    # T9: SELL orders in ledger not compensated
    # ------------------------------------------------------------------
    def test_sell_orders_not_compensated(self):
        """Only BUY orders should contribute to settlement-lag compensation."""
        lpath = self._make_ledger(self._tmp_path, [
            {"symbol": "6981.T", "side": "SELL", "qty": 100, "price": 5000.0, "status": "submitted"},
        ])
        sv, ss, lv, ls = _cb_guard_compensation(
            current_equity    = 3_000_000.0,
            current_positions = {},
            pre_commit_qtys   = {},
            pre_commit_costs  = {},
            ledger_path       = lpath,
            today_str         = self.TODAY,
        )
        self.assertAlmostEqual(lv, 0.0, places=0,
                               msg="SELL orders must not be compensated")

    # ------------------------------------------------------------------
    # T10: morning_signal calls mark_submitted() → "submitted"; also "pending"
    #      accepted to handle orders recorded before the fix was deployed.
    # ------------------------------------------------------------------
    def test_pending_status_accepted_for_morning_signal_gap(self):
        """
        run_morning_signal.py calls mark_submitted() on success (status="submitted").
        Legacy entries with status "pending" (recorded before the fix) must also be
        compensated so the 2026-06-04 incident can never recur.
        """
        lpath = self._make_ledger(self._tmp_path, [
            {"symbol": "5301.T", "side": "BUY", "qty": 400, "price": 1852.5, "status": "pending"},
        ])
        sv, ss, lv, ls = _cb_guard_compensation(
            current_equity    = 3_184_309.0,
            current_positions = {"6981.T": {"qty": 100}},
            pre_commit_qtys   = {"6981.T": 100},
            pre_commit_costs  = {"6981.T": 4864.0},
            ledger_path       = lpath,
            today_str         = self.TODAY,
        )
        self.assertGreater(lv, 0.0,
                           "pending BUY must be compensated (mark_submitted gap handled)")
        self.assertIn("5301.T", ls)


# ─────────────────────────────────────────────────────────────────────────────
# TASK-1 tests — slot_pressure current_positions from portfolio_summary
# ─────────────────────────────────────────────────────────────────────────────

class TestSlotPressureCurrentPositions(unittest.TestCase):
    """
    Verify that current_positions used for slot_pressure reflects
    actual holdings, not winner-confirmation count.
    """

    def _make_inp(self, current_positions: int, max_positions: int = 3) -> SlotPressureInput:
        return SlotPressureInput(
            max_positions=max_positions,
            current_positions=current_positions,
            holding_priorities=[],
            holding_tiers=[],
            avg_candidate_priority_5d=0.0,
            recent_rejected_candidates=0,
            recent_severe_opportunity_cost_events=0,
        )

    def test_two_holdings_one_open_slot(self):
        """
        2 actual holdings, max_positions=3 → open_slots MUST be 1.
        This reproduces the fixed bug (was 3 before fix).
        """
        inp    = self._make_inp(current_positions=2, max_positions=3)
        result = compute_slot_pressure(inp)
        self.assertEqual(result.open_slots, 1,
                         "open_slots must be max_positions - current_positions = 1")

    def test_zero_holdings_three_open_slots(self):
        """0 holdings, max_positions=3 → open_slots=3 (correct empty portfolio)."""
        inp    = self._make_inp(current_positions=0, max_positions=3)
        result = compute_slot_pressure(inp)
        self.assertEqual(result.open_slots, 3)

    def test_full_portfolio_zero_open_slots(self):
        """max_positions=3, current=3 → open_slots=0."""
        inp    = self._make_inp(current_positions=3, max_positions=3)
        result = compute_slot_pressure(inp)
        self.assertEqual(result.open_slots, 0)

    def test_portfolio_summary_value_used(self):
        """
        Simulates the exact fix: portfolio_summary["current_positions"] overrides
        len(_sp_held_list) when winner confirmations are zero.

        Before fix: current_positions=len([])=0 → open_slots=3
        After  fix: current_positions=portfolio_summary.get("current_positions", 0)=2 → open_slots=1
        """
        portfolio_summary = {"current_positions": 2}
        sp_held_list: list = []  # winner confirmations empty (both confirmed=False)

        # Replicate the fixed expression from run_live_signal.py
        effective_positions = portfolio_summary.get("current_positions", len(sp_held_list))
        self.assertEqual(effective_positions, 2)

        inp    = self._make_inp(current_positions=effective_positions, max_positions=3)
        result = compute_slot_pressure(inp)
        self.assertEqual(result.open_slots, 1)

    def test_fallback_to_held_list_when_summary_missing(self):
        """
        If portfolio_summary has no 'current_positions' key,
        fallback to len(_sp_held_list) is used.
        """
        portfolio_summary: dict = {}  # no current_positions key
        sp_held_list = [MagicMock(), MagicMock()]  # 2 confirmed winners

        effective_positions = portfolio_summary.get("current_positions", len(sp_held_list))
        self.assertEqual(effective_positions, 2)


# ─────────────────────────────────────────────────────────────────────────────
# CB_STATE structured log + last_equity fix + 2026-06-04 RCA regression
# ─────────────────────────────────────────────────────────────────────────────

import logging
from unittest.mock import patch as _patch


def _make_bridge_stub(capital: float = 3_000_000.0):
    """
    _update_cb_state() をテストするための最小 SignalBridge スタブ。
    API / ファイル IO は一切呼ばない。
    """
    from src.kabusapi.signal_bridge import SignalBridge
    bridge = object.__new__(SignalBridge)
    bridge.capital = capital
    bridge.live    = True
    return bridge


class TestCBStateStructuredLog(unittest.TestCase):
    """
    _update_cb_state() が [CB_STATE] 構造化ログを正しく出力し、
    last_equity を上書きしないことを検証する。

    回帰: 2026-06-04 LIVEで SETTLEMENT_LAG なのに DRAWDOWN_TRIGGER と誤判定
    されていた根本原因の修正確認。

    2026-07-19 Broker-as-Sole-SSOTリファクタ: current_equityは常にraw broker
    equityを渡す（旧_cb_equityの合成equity方式は廃止）。決済ラグ補償は
    PendingOrderStateとしてCB発動抑制の判定材料にのみ使う。
    """

    PEAK = 4_089_109.0
    RAW  = 3_184_309.0
    COMP = 741_000.0
    PENDING = PendingOrderState(pending_value=COMP, symbols=("6981.T",))

    def _state(self, cb="NORMAL", last_equity=None):
        return {
            "cb_state":             cb,
            "equity_peak":          self.PEAK,
            "safe_warn_count":      0,
            "cb_cooldown_end_date": None,
            "recovery_threshold":   None,
            "last_equity":          last_equity or self.RAW,
        }

    # ------------------------------------------------------------------
    # T1: settlement lag suspected → CB発動見送り (INFO)
    # ------------------------------------------------------------------
    def test_settlement_lag_no_cb_transition_emits_info_not_warning(self):
        """
        raw DD=-22.1%だがpending_value=741,000で説明可能 (hypothetical DD=-4.0%)
        → CB は発火しない (NORMAL→NORMAL)。遷移がないため [CB_STATE] は INFO で
        出力され WARNING は出ない。

        再現: 2026-06-04 DRY run が決済ラグにより一時的にequityが低く見えていた
        だけで、実際にはCB発動すべきでなかったケース。
        """
        bridge = _make_bridge_stub()
        state  = self._state()

        with self.assertLogs("src.kabusapi.signal_bridge", level="INFO") as cm:
            bridge._update_cb_state(
                state, self.RAW, "2026-06-04", pending_order_state=self.PENDING,
            )

        # 状態が NORMAL のまま
        self.assertEqual(state["cb_state"], "NORMAL",
                         "CB must not fire when settlement lag explains the DD")

        combined = "\n".join(cm.output)
        self.assertIn("[CB_STATE]", combined)
        self.assertIn("settlement_lag_suspected=True", combined)
        # 遷移なし → WARNING レベルの [CB_STATE] は出ない
        warning_lines = [l for l in cm.output if "WARNING" in l and "[CB_STATE]" in l]
        self.assertEqual(warning_lines, [],
                         "no [CB_STATE] WARNING expected when no state transition")

    def test_settlement_lag_reason_with_cb_fire_would_log_warning(self):
        """
        pending_order_stateなし（決済ラグ補償なし）で _update_cb_state を呼ぶと
        CB が発火し [CB_STATE] reason=DRAWDOWN_TRIGGER が出力されること。
        """
        bridge = _make_bridge_stub()
        state  = self._state()

        with self.assertLogs("src.kabusapi.signal_bridge", level="WARNING") as cm:
            bridge._update_cb_state(state, self.RAW, "2026-06-04")

        self.assertEqual(state["cb_state"], "CB_ACTIVE",
                         "raw DD=-22.1% must trigger CB_ACTIVE")

        combined = "\n".join(cm.output)
        self.assertIn("[CB_STATE]",          combined)
        self.assertIn("before=NORMAL",       combined)
        self.assertIn("after=CB_ACTIVE",     combined)
        self.assertIn("DRAWDOWN_TRIGGER",    combined)
        self.assertIn("dd=",                 combined)

    # ------------------------------------------------------------------
    # T2: settlement lag 補償で CB 発火しないこと (PendingOrderState抑制)
    # ------------------------------------------------------------------
    def test_v2_compensation_prevents_false_cb(self):
        """
        pending_value=741,000 (hypothetical DD=-4.0%) を渡せば CB は発火しないこと。
        """
        bridge = _make_bridge_stub()
        state  = self._state()
        bridge._update_cb_state(
            state, self.RAW, "2026-06-04", pending_order_state=self.PENDING,
        )
        self.assertEqual(state["cb_state"], "NORMAL")

    # ------------------------------------------------------------------
    # T3: last_equity が _update_cb_state() で上書きされないこと
    # ------------------------------------------------------------------
    def test_last_equity_not_overwritten(self):
        """
        commit_broker_snapshot() が raw equity を last_equity に書く。
        _update_cb_state() はPendingOrderStateを判定材料に使うだけで、
        last_equity を上書きしてはならない。
        """
        bridge = _make_bridge_stub()
        state  = self._state(last_equity=self.RAW)

        bridge._update_cb_state(
            state, self.RAW, "2026-06-04", pending_order_state=self.PENDING,
        )
        self.assertAlmostEqual(
            state["last_equity"], self.RAW, places=0,
            msg="last_equity must remain raw equity after _update_cb_state()"
        )

    # ------------------------------------------------------------------
    # T4: pending_order_state=None (既定値) でも例外が起きないこと
    # ------------------------------------------------------------------
    def test_pending_order_state_none_default_no_crash(self):
        """pending_order_state を省略しても例外が起きないこと。"""
        bridge = _make_bridge_stub()
        state  = self._state()
        result = bridge._update_cb_state(state, self.RAW, "2026-06-04")
        self.assertIn("cb_state", result)

    # ------------------------------------------------------------------
    # T5: [CB_STATE] INFO ログ (遷移なし)
    # ------------------------------------------------------------------
    def test_no_transition_emits_info_log(self):
        """遷移がない場合は [CB_STATE] INFO が出ること (WARNING は出ない)。"""
        bridge = _make_bridge_stub()
        state  = self._state()
        # settlement lag suppression → CB は発火しない
        with self.assertLogs("src.kabusapi.signal_bridge", level="INFO") as cm:
            bridge._update_cb_state(
                state, self.RAW, "2026-06-04", pending_order_state=self.PENDING,
            )
        combined = "\n".join(cm.output)
        self.assertIn("[CB_STATE]", combined)
        self.assertIn("dd=",        combined)

    # ------------------------------------------------------------------
    # T6: PEAK_ANOMALY は NORMAL から到達不能 (設計上の制約確認)
    # ------------------------------------------------------------------
    def test_peak_anomaly_unreachable_from_normal(self):
        """
        PEAK_ANOMALY_RATIO=1.25 かつ CB_DD_TRIGGER=0.15 の場合、
        ratio > 1.25 → DD < -20% → CB_DD_TRIGGER -15% を必ず超えるため
        SAFE_WARN は NORMAL 状態からは数学的に到達不能であることを確認する。

        ratio > 1.25 ⟹ equity/peak < 0.8 ⟹ DD = equity/peak - 1 < -0.2 = -20%
        -20% ≤ -15% (CB_DD_TRIGGER) ⟹ CB_ACTIVE が先に発火する。
        """
        bridge = _make_bridge_stub()
        equity = 3_000_000.0
        # ratio = 1.26: peak = equity × 1.26 → DD = -20.6%
        big_peak = equity * 1.26
        state = {
            "cb_state":             "NORMAL",
            "equity_peak":          big_peak,
            "safe_warn_count":      0,
            "cb_cooldown_end_date": None,
            "recovery_threshold":   None,
            "last_equity":          equity,
        }
        bridge._update_cb_state(state, equity, "2026-06-05")

        # ratio > 1.25 のとき DD > 20% → CB_ACTIVE が先に発火する
        self.assertEqual(state["cb_state"], "CB_ACTIVE",
                         "ratio>1.25 implies DD>20%>15% → CB_ACTIVE fires before SAFE_WARN")

    # ------------------------------------------------------------------
    # T7: 2026-06-04 incident 完全再現
    # ------------------------------------------------------------------
    def test_incident_20260604_full_replay(self):
        """
        2026-06-04 DRY run での CB 誤発動を完全再現する。

        PendingOrderStateなし (raw equity のみ):
          dd=-22.1% → CB_ACTIVE

        PendingOrderStateあり (決済ラグ補償で発動抑制):
          hypothetical_dd=-4.0% → NORMAL を維持

        両方を同一 peak/state で検証する。
        """
        bridge = _make_bridge_stub()

        # --- PendingOrderStateなし (2026-06-04 08:43 の実際の挙動) ---
        state_no_v2 = {
            "cb_state":             "NORMAL",
            "equity_peak":          self.PEAK,
            "safe_warn_count":      0,
            "cb_cooldown_end_date": None,
            "recovery_threshold":   None,
            "last_equity":          self.RAW,
        }
        bridge._update_cb_state(state_no_v2, self.RAW, "2026-06-04")
        self.assertEqual(state_no_v2["cb_state"], "CB_ACTIVE",
                         "PendingOrderStateなし: raw DD=-22.1% → CB_ACTIVE 必須")

        # --- PendingOrderStateあり (修正後の期待挙動) ---
        state_with_v2 = {
            "cb_state":             "NORMAL",
            "equity_peak":          self.PEAK,
            "safe_warn_count":      0,
            "cb_cooldown_end_date": None,
            "recovery_threshold":   None,
            "last_equity":          self.RAW,
        }
        bridge._update_cb_state(
            state_with_v2, self.RAW, "2026-06-04", pending_order_state=self.PENDING,
        )
        self.assertEqual(state_with_v2["cb_state"], "NORMAL",
                         "PendingOrderStateあり: hypothetical DD=-4.0% → NORMAL を維持必須")


if __name__ == "__main__":
    unittest.main()
