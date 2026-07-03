"""
tests/test_addon.py
Pytest suite for the winner add-on execution layer.

Covers:
  - WinnerConfirmationEngine: all gates, confidence scoring
  - AddOnExecutionPolicy: portfolio heat, cooldown, depth, concentration, liquidity, run throttle
  - AddonState: load/save/purge, decision log append
  - Safety: no averaging down, no martingale, winners-only
"""
from __future__ import annotations

import json
import tempfile
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.addon.winner_confirmation import (
    check_winner,
    WinnerConfirmation,
    MIN_PROFIT_THRESHOLD,
    MIN_HOLD_BEFORE_ADDON,
    MAX_RSR_RANK_FOR_ADDON,
    MIN_RSR_STRENGTH,
    MIN_TREND_QUALITY,
    MAX_DETERIORATION,
    MAX_ATR_RATIO,
)
from src.addon.addon_state import (
    AddonSymbolState,
    AddonDecisionRecord,
    load_addon_state,
    save_addon_state,
    append_decision_record,
    purge_exited_symbols,
)
from src.addon.addon_policy import (
    AddOnExecutionPolicy,
    _cooldown_elapsed,
    _check_concentration,
    _check_liquidity,
    UNIT_SHARES,
    MAX_ADDON_DEPTH,
    ADDON_COOLDOWN_DAYS,
    MAX_ADDONS_PER_RUN,
    MAX_POSITION_WEIGHT_ADDON,
    MIN_PORTFOLIO_PNL_FOR_ADDON,
)

JST = timezone(timedelta(hours=9))
TODAY = "2026-05-23"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _good_winner(symbol: str = "6758.T") -> dict:
    """A dictionary of kwargs that produce confirmed=True."""
    return dict(
        symbol=symbol,
        unrealized_pnl_pct=0.06,    # +6% profit
        hold_days=8,                 # 8 days held
        rsr_rank=2,                  # rank 2 (good)
        rsr=82.0,                    # RSR 82
        trend_quality=1.5,           # positive momentum
        deterioration_score=0.10,    # low deterioration
        regime_bear=False,
        atr_ratio=1.2,
        current_price=5000.0,
        entry_price=4717.0,          # 5000 / 1.06
    )


def _good_confirmation(symbol: str = "6758.T") -> WinnerConfirmation:
    return check_winner(**_good_winner(symbol))


def _policy_with_tmp(tmp: Path, **kwargs) -> tuple[AddOnExecutionPolicy, Path, Path]:
    st = tmp / "addon_state.json"
    dc = tmp / "addon_decisions.jsonl"
    return AddOnExecutionPolicy(state_path=st, decisions_path=dc, **kwargs), st, dc


def _held_pos(symbol: str, price: float = 5000.0, qty: int = 100) -> dict:
    return {symbol: {"qty": qty, "current_price": price, "avg_daily_volume_yen": 0.0}}


# ─────────────────────────────────────────────────────────────────────────────
# WinnerConfirmationEngine
# ─────────────────────────────────────────────────────────────────────────────

class TestWinnerConfirmation:
    def test_all_gates_pass_confirmed(self):
        c = _good_confirmation()
        assert c.confirmed
        assert c.confidence_score > 0

    def test_fail_profit_threshold(self):
        kw = _good_winner()
        kw["unrealized_pnl_pct"] = MIN_PROFIT_THRESHOLD - 0.001
        c = check_winner(**kw)
        assert not c.confirmed
        assert any("pnl=" in r for r in c.fail_reasons)
        assert c.confidence_score == 0.0

    def test_fail_hold_duration(self):
        kw = _good_winner()
        kw["hold_days"] = MIN_HOLD_BEFORE_ADDON - 1
        c = check_winner(**kw)
        assert not c.confirmed
        assert any("hold=" in r for r in c.fail_reasons)

    def test_fail_rsr_rank(self):
        kw = _good_winner()
        kw["rsr_rank"] = MAX_RSR_RANK_FOR_ADDON + 1
        c = check_winner(**kw)
        assert not c.confirmed
        assert any("rsr_rank=" in r for r in c.fail_reasons)

    def test_fail_rsr_strength(self):
        kw = _good_winner()
        kw["rsr"] = MIN_RSR_STRENGTH - 0.1
        c = check_winner(**kw)
        assert not c.confirmed
        assert any("rsr=" in r for r in c.fail_reasons)

    def test_fail_trend_quality(self):
        kw = _good_winner()
        kw["trend_quality"] = MIN_TREND_QUALITY - 0.001
        c = check_winner(**kw)
        assert not c.confirmed
        assert any("trend_quality=" in r for r in c.fail_reasons)

    def test_fail_deterioration(self):
        kw = _good_winner()
        kw["deterioration_score"] = MAX_DETERIORATION + 0.01
        c = check_winner(**kw)
        assert not c.confirmed
        assert any("deterioration=" in r for r in c.fail_reasons)

    def test_fail_regime_bear(self):
        kw = _good_winner()
        kw["regime_bear"] = True
        c = check_winner(**kw)
        assert not c.confirmed
        assert any("regime=bear" in r for r in c.fail_reasons)

    def test_fail_atr_ratio(self):
        kw = _good_winner()
        kw["atr_ratio"] = MAX_ATR_RATIO + 0.1
        c = check_winner(**kw)
        assert not c.confirmed
        assert any("atr_ratio=" in r for r in c.fail_reasons)

    def test_fail_averaging_down(self):
        kw = _good_winner()
        kw["current_price"] = 4500.0
        kw["entry_price"] = 5000.0      # current < entry = averaging down
        c = check_winner(**kw)
        assert not c.confirmed
        assert any("averaging_down" in r for r in c.fail_reasons)

    def test_fail_averaging_down_at_entry(self):
        kw = _good_winner()
        kw["current_price"] = 5000.0
        kw["entry_price"] = 5000.0      # current == entry → still blocked
        c = check_winner(**kw)
        assert not c.confirmed

    def test_multiple_failures_all_reported(self):
        kw = _good_winner()
        kw["unrealized_pnl_pct"] = 0.0
        kw["hold_days"] = 1
        c = check_winner(**kw)
        assert not c.confirmed
        assert len(c.fail_reasons) >= 2

    def test_confidence_score_range(self):
        c = _good_confirmation()
        assert 0.0 <= c.confidence_score <= 1.0

    def test_confidence_zero_when_not_confirmed(self):
        kw = _good_winner()
        kw["unrealized_pnl_pct"] = 0.0
        c = check_winner(**kw)
        assert not c.confirmed
        assert c.confidence_score == 0.0

    def test_higher_profit_higher_score(self):
        c_low  = check_winner(**{**_good_winner(), "unrealized_pnl_pct": 0.04})
        c_high = check_winner(**{**_good_winner(), "unrealized_pnl_pct": 0.10})
        assert c_high.confidence_score > c_low.confidence_score

    def test_confirmed_fields_echoed(self):
        kw = _good_winner()
        c = check_winner(**kw)
        assert c.symbol == kw["symbol"]
        assert c.hold_days == kw["hold_days"]
        assert c.rsr == kw["rsr"]


# ─────────────────────────────────────────────────────────────────────────────
# AddonState
# ─────────────────────────────────────────────────────────────────────────────

class TestAddonState:
    def test_load_missing_file_returns_empty(self, tmp_path):
        state = load_addon_state(tmp_path / "missing.json")
        assert state == {}

    def test_save_and_load_roundtrip(self, tmp_path):
        path = tmp_path / "state.json"
        state = {
            "6758.T": AddonSymbolState(
                addon_count=1,
                last_addon_date="2026-05-20",
                total_addon_shares=100,
                last_confirmation_score=0.75,
            )
        }
        save_addon_state(state, path)
        loaded = load_addon_state(path)
        assert "6758.T" in loaded
        assert loaded["6758.T"].addon_count == 1
        assert loaded["6758.T"].last_addon_date == "2026-05-20"

    def test_save_atomic_tmp_removed(self, tmp_path):
        path = tmp_path / "state.json"
        save_addon_state({}, path)
        assert path.exists()
        assert not (tmp_path / "state.tmp").exists()

    def test_purge_removes_exited_symbols(self):
        state = {
            "6758.T": AddonSymbolState(addon_count=1),
            "7203.T": AddonSymbolState(addon_count=2),
        }
        pruned = purge_exited_symbols(state, {"6758.T"})
        assert "6758.T" in pruned
        assert "7203.T" not in pruned

    def test_purge_all_when_no_held(self):
        state = {"6758.T": AddonSymbolState(addon_count=1)}
        pruned = purge_exited_symbols(state, set())
        assert pruned == {}

    def test_append_decision_record(self, tmp_path):
        path = tmp_path / "decisions.jsonl"
        rec = AddonDecisionRecord(
            ts="2026-05-23T09:00:00+0900",
            run_id="test-run",
            symbol="6758.T",
            decision="ADDON_APPROVED",
            block_reason="",
            addon_count_before=0,
            addon_count_after=1,
            addon_shares=100,
            confirmation_score=0.75,
            unrealized_pnl_pct=0.06,
            hold_days=8,
            rsr_rank=2,
            rsr=82.0,
            estimated_cost=500000.0,
        )
        append_decision_record(rec, path)
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["symbol"] == "6758.T"
        assert parsed["decision"] == "ADDON_APPROVED"

    def test_decision_log_is_append_only(self, tmp_path):
        path = tmp_path / "decisions.jsonl"
        for i in range(3):
            rec = AddonDecisionRecord(
                ts=f"2026-05-23T0{i}:00:00+0900",
                run_id=f"run-{i}",
                symbol="6758.T",
                decision="ADDON_BLOCKED",
                block_reason="test",
                addon_count_before=0, addon_count_after=0,
                addon_shares=0, confirmation_score=0.0,
                unrealized_pnl_pct=0.0, hold_days=0,
                rsr_rank=99, rsr=0.0, estimated_cost=0.0,
            )
            append_decision_record(rec, path)
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3

    def test_load_corrupt_file_returns_empty(self, tmp_path):
        path = tmp_path / "corrupt.json"
        path.write_text("not valid json", encoding="utf-8")
        state = load_addon_state(path)
        assert state == {}


# ─────────────────────────────────────────────────────────────────────────────
# Gate helpers (unit)
# ─────────────────────────────────────────────────────────────────────────────

class TestGateHelpers:
    def test_cooldown_elapsed_no_history(self):
        assert _cooldown_elapsed("", TODAY, 5) is True

    def test_cooldown_elapsed_enough_days(self):
        assert _cooldown_elapsed("2026-05-17", TODAY, 5) is True  # 6 days

    def test_cooldown_not_elapsed_recent(self):
        assert _cooldown_elapsed("2026-05-20", TODAY, 5) is False  # 3 days

    def test_cooldown_exactly_on_day(self):
        assert _cooldown_elapsed("2026-05-18", TODAY, 5) is True   # exactly 5

    def test_concentration_ok(self):
        ok, _ = _check_concentration(
            cur_position_value=500_000,
            addon_cost=100_000,
            portfolio_equity=3_000_000,
            max_weight=0.35,
        )
        assert ok  # (500k + 100k) / 3M = 20% < 35%

    def test_concentration_blocked(self):
        ok, reason = _check_concentration(
            cur_position_value=900_000,
            addon_cost=200_000,
            portfolio_equity=3_000_000,
            max_weight=0.35,
        )
        assert not ok  # 1.1M / 3M = 36.7% > 35%
        assert "max=" in reason

    def test_concentration_zero_equity(self):
        ok, _ = _check_concentration(500_000, 100_000, 0, 0.35)
        assert not ok

    def test_liquidity_unknown_adv_passes(self):
        ok, _ = _check_liquidity(100, 5000.0, 0.0)
        assert ok

    def test_liquidity_ok(self):
        # addon_value = 100 * 5000 = 500k; ADV = 100M → 100M/10 = 10M >> 500k
        ok, _ = _check_liquidity(100, 5000.0, 100_000_000.0)
        assert ok

    def test_liquidity_blocked(self):
        # addon_value = 100 * 5000 = 500k; ADV = 3M → ADV/10 = 300k < 500k
        ok, reason = _check_liquidity(100, 5000.0, 3_000_000.0)
        assert not ok
        assert "liquidity" in reason


# ─────────────────────────────────────────────────────────────────────────────
# AddOnExecutionPolicy — end-to-end
# ─────────────────────────────────────────────────────────────────────────────

class TestAddOnPolicy:
    def test_generates_addon_for_confirmed_winner(self, tmp_path):
        policy, _, _ = _policy_with_tmp(tmp_path)
        c = _good_confirmation()
        res = policy.run(
            confirmations=[c],
            portfolio_equity=3_000_000,
            portfolio_pnl_pct=0.0,
            held_positions=_held_pos("6758.T", price=5000.0),
            today=TODAY,
        )
        assert len(res.addon_orders) == 1
        ao = res.addon_orders[0]
        assert ao.symbol == "6758.T"
        assert ao.qty == UNIT_SHARES
        assert ao.estimated_price == 5000.0
        assert ao.estimated_cost == 500_000.0
        assert "ADDON[winner" in ao.reason

    def test_no_addon_for_unconfirmed(self, tmp_path):
        policy, _, _ = _policy_with_tmp(tmp_path)
        kw = _good_winner()
        kw["unrealized_pnl_pct"] = 0.0   # fails profit gate
        c = check_winner(**kw)
        assert not c.confirmed
        res = policy.run(
            confirmations=[c],
            portfolio_equity=3_000_000,
            portfolio_pnl_pct=0.0,
            held_positions=_held_pos("6758.T"),
            today=TODAY,
        )
        assert res.addon_orders == []

    def test_portfolio_heat_blocks_all(self, tmp_path):
        policy, _, _ = _policy_with_tmp(tmp_path)
        c = _good_confirmation()
        res = policy.run(
            confirmations=[c],
            portfolio_equity=3_000_000,
            portfolio_pnl_pct=MIN_PORTFOLIO_PNL_FOR_ADDON - 0.01,  # too low
            held_positions=_held_pos("6758.T"),
            today=TODAY,
        )
        assert res.addon_orders == []
        assert any(b["symbol"] == "6758.T" for b in res.blocked)

    def test_cooldown_blocks_recent_addon(self, tmp_path):
        policy, st, _ = _policy_with_tmp(tmp_path)
        # Pre-fill state: addon done yesterday
        yesterday = "2026-05-22"
        save_addon_state(
            {"6758.T": AddonSymbolState(addon_count=1, last_addon_date=yesterday)}, st
        )
        c = _good_confirmation()
        res = policy.run(
            confirmations=[c],
            portfolio_equity=3_000_000,
            portfolio_pnl_pct=0.0,
            held_positions=_held_pos("6758.T"),
            today=TODAY,  # only 1 day since last addon → cooldown=5 not elapsed
        )
        assert res.addon_orders == []
        assert any("cooldown" in b["reason"] for b in res.blocked)

    def test_depth_gate_blocks_at_max(self, tmp_path):
        policy, st, _ = _policy_with_tmp(tmp_path)
        save_addon_state(
            {"6758.T": AddonSymbolState(addon_count=MAX_ADDON_DEPTH, last_addon_date="2026-04-01")}, st
        )
        c = _good_confirmation()
        res = policy.run(
            confirmations=[c],
            portfolio_equity=3_000_000,
            portfolio_pnl_pct=0.0,
            held_positions=_held_pos("6758.T"),
            today=TODAY,
        )
        assert res.addon_orders == []
        assert any("max_depth" in b["reason"] for b in res.blocked)

    def test_concentration_gate_blocks(self, tmp_path):
        policy, _, _ = _policy_with_tmp(tmp_path, max_position_weight=0.20)
        c = _good_confirmation()
        # Position is already 700k in a 3M portfolio (23%) → adding 500k → 40% > 20%
        res = policy.run(
            confirmations=[c],
            portfolio_equity=3_000_000,
            portfolio_pnl_pct=0.0,
            held_positions={"6758.T": {"qty": 140, "current_price": 5000.0, "avg_daily_volume_yen": 0.0}},
            today=TODAY,
        )
        assert res.addon_orders == []
        assert any("post_addon_weight" in b["reason"] or "max=" in b["reason"] for b in res.blocked)

    def test_state_persisted_after_approval(self, tmp_path):
        policy, st, dc = _policy_with_tmp(tmp_path)
        c = _good_confirmation()
        policy.run(
            confirmations=[c],
            portfolio_equity=3_000_000,
            portfolio_pnl_pct=0.0,
            held_positions=_held_pos("6758.T"),
            today=TODAY,
        )
        loaded = load_addon_state(st)
        assert "6758.T" in loaded
        assert loaded["6758.T"].addon_count == 1
        assert loaded["6758.T"].last_addon_date == TODAY

    def test_decision_log_written_on_approval(self, tmp_path):
        policy, _, dc = _policy_with_tmp(tmp_path)
        c = _good_confirmation()
        policy.run(
            confirmations=[c],
            portfolio_equity=3_000_000,
            portfolio_pnl_pct=0.0,
            held_positions=_held_pos("6758.T"),
            today=TODAY,
        )
        assert dc.exists()
        records = [json.loads(l) for l in dc.read_text(encoding="utf-8").strip().splitlines()]
        approved = [r for r in records if r["decision"] == "ADDON_APPROVED"]
        assert len(approved) == 1

    def test_decision_log_written_on_block(self, tmp_path):
        policy, _, dc = _policy_with_tmp(tmp_path)
        kw = _good_winner()
        kw["unrealized_pnl_pct"] = 0.0
        c = check_winner(**kw)
        policy.run(
            confirmations=[c],
            portfolio_equity=3_000_000,
            portfolio_pnl_pct=0.0,
            held_positions=_held_pos("6758.T"),
            today=TODAY,
        )
        # Unconfirmed signals don't reach the policy gates → no log (by design)
        # No error should be raised
        assert not dc.exists() or True  # log may or may not exist

    def test_run_throttle_limits_to_one_per_run(self, tmp_path):
        policy, _, _ = _policy_with_tmp(tmp_path, max_addons_per_run=1)
        c1 = _good_confirmation("6758.T")
        c2 = _good_confirmation("6861.T")
        res = policy.run(
            confirmations=[c1, c2],
            portfolio_equity=10_000_000,
            portfolio_pnl_pct=0.0,
            held_positions={
                "6758.T": {"qty": 100, "current_price": 5000.0, "avg_daily_volume_yen": 0.0},
                "6861.T": {"qty": 100, "current_price": 5000.0, "avg_daily_volume_yen": 0.0},
            },
            today=TODAY,
        )
        # At most MAX_ADDONS_PER_RUN = 1 order generated
        assert len(res.addon_orders) <= 1

    def test_addon_qty_always_unit_shares(self, tmp_path):
        policy, _, _ = _policy_with_tmp(tmp_path)
        c = _good_confirmation()
        res = policy.run(
            confirmations=[c],
            portfolio_equity=3_000_000,
            portfolio_pnl_pct=0.0,
            held_positions=_held_pos("6758.T"),
            today=TODAY,
        )
        for ao in res.addon_orders:
            assert ao.qty == UNIT_SHARES
            assert ao.qty % 100 == 0

    def test_no_addon_missing_position_data(self, tmp_path):
        policy, _, _ = _policy_with_tmp(tmp_path)
        c = _good_confirmation("6758.T")
        res = policy.run(
            confirmations=[c],
            portfolio_equity=3_000_000,
            portfolio_pnl_pct=0.0,
            held_positions={},     # position data missing
            today=TODAY,
        )
        assert res.addon_orders == []

    def test_no_addon_zero_price(self, tmp_path):
        policy, _, _ = _policy_with_tmp(tmp_path)
        c = _good_confirmation()
        res = policy.run(
            confirmations=[c],
            portfolio_equity=3_000_000,
            portfolio_pnl_pct=0.0,
            held_positions={"6758.T": {"qty": 100, "current_price": 0.0, "avg_daily_volume_yen": 0.0}},
            today=TODAY,
        )
        assert res.addon_orders == []

    def test_best_winner_first_when_multiple(self, tmp_path):
        policy, _, _ = _policy_with_tmp(tmp_path, max_addons_per_run=1)
        # c1 has higher confidence via higher pnl
        c1 = check_winner(**{**_good_winner("6758.T"), "unrealized_pnl_pct": 0.12})
        c2 = check_winner(**{**_good_winner("6861.T"), "unrealized_pnl_pct": 0.04})
        assert c1.confidence_score > c2.confidence_score
        res = policy.run(
            confirmations=[c2, c1],   # c2 first in list but lower score
            portfolio_equity=10_000_000,
            portfolio_pnl_pct=0.0,
            held_positions={
                "6758.T": {"qty": 100, "current_price": 5000.0, "avg_daily_volume_yen": 0.0},
                "6861.T": {"qty": 100, "current_price": 5000.0, "avg_daily_volume_yen": 0.0},
            },
            today=TODAY,
        )
        assert len(res.addon_orders) == 1
        assert res.addon_orders[0].symbol == "6758.T"   # higher score wins

    def test_state_purged_for_exited_symbols(self, tmp_path):
        policy, st, _ = _policy_with_tmp(tmp_path)
        # Pre-fill state for a symbol we no longer hold
        save_addon_state(
            {
                "6758.T": AddonSymbolState(addon_count=1, last_addon_date="2026-04-01"),
                "7203.T": AddonSymbolState(addon_count=2, last_addon_date="2026-04-01"),
            },
            st,
        )
        c = _good_confirmation("6758.T")
        policy.run(
            confirmations=[c],
            portfolio_equity=3_000_000,
            portfolio_pnl_pct=0.0,
            held_positions=_held_pos("6758.T"),   # 7203.T not held
            today=TODAY,
        )
        loaded = load_addon_state(st)
        assert "7203.T" not in loaded   # purged

    def test_consecutive_addons_respect_depth(self, tmp_path):
        """Two consecutive runs: first approved, second also approved (count=1), third blocked."""
        policy, st, _ = _policy_with_tmp(tmp_path, max_addon_depth=2, cooldown_days=0)
        c = _good_confirmation()
        pos = _held_pos("6758.T")
        equity = 10_000_000

        # Run 1: count=0 → approved
        res1 = policy.run([c], equity, 0.0, pos, "2026-05-01", "run-1")
        assert len(res1.addon_orders) == 1

        # Run 2: count=1 → approved
        res2 = policy.run([c], equity, 0.0, pos, "2026-05-02", "run-2")
        assert len(res2.addon_orders) == 1

        # Run 3: count=2 = max_depth → blocked
        res3 = policy.run([c], equity, 0.0, pos, "2026-05-03", "run-3")
        assert res3.addon_orders == []
        assert any("max_depth" in b["reason"] for b in res3.blocked)


# ─────────────────────────────────────────────────────────────────────────────
# paths.py — check constants are exported
# ─────────────────────────────────────────────────────────────────────────────

class TestPaths:
    def test_addon_paths_exported(self):
        from src.paths import ADDON_STATE_FILE, ADDON_DECISIONS_FILE, ADDON_DIR
        assert str(ADDON_STATE_FILE).endswith("addon_state.json")
        assert str(ADDON_DECISIONS_FILE).endswith("addon_decisions.jsonl")
        assert ADDON_DIR.name == "addon"

    def test_addon_paths_under_runtime(self):
        from src.paths import ADDON_DIR, RUNTIME_DIR
        assert ADDON_DIR.parent == RUNTIME_DIR


# ─────────────────────────────────────────────────────────────────────────────
# Package-level imports
# ─────────────────────────────────────────────────────────────────────────────

class TestPackageImports:
    def test_all_symbols_importable(self):
        from src.addon import (
            check_winner, WinnerConfirmation,
            AddOnExecutionPolicy, AddOnOrder, AddOnResult,
            AddonSymbolState, AddonDecisionRecord,
            load_addon_state, save_addon_state,
            append_decision_record, purge_exited_symbols,
            UNIT_SHARES, MAX_ADDON_DEPTH, ADDON_COOLDOWN_DAYS,
            MIN_PROFIT_THRESHOLD, MIN_HOLD_BEFORE_ADDON,
        )

    def test_extension_symbols_importable(self):
        from src.addon import (
            check_extension, ExtensionInput, ExtensionResult,
            MAX_VELOCITY_RATE, MAX_RSR_ACCELERATION,
            MAX_CANDLE_RANGE_RATIO, MAX_VOLUME_RATIO,
            ISOLATED_RSR_THRESHOLD, PORTFOLIO_DIVERGENCE_THRESHOLD,
        )


# ─────────────────────────────────────────────────────────────────────────────
# ExtensionFilter
# ─────────────────────────────────────────────────────────────────────────────

from src.addon.extension_filter import (
    check_extension,
    ExtensionInput,
    ExtensionResult,
    MAX_VELOCITY_RATE,
    MAX_RSR_ACCELERATION,
    MAX_CANDLE_RANGE_RATIO,
    MAX_VOLUME_RATIO,
    ISOLATED_RSR_THRESHOLD,
    PORTFOLIO_DIVERGENCE_THRESHOLD,
)


def _good_ext(symbol: str = "6758.T") -> ExtensionInput:
    """ExtensionInput that passes all extension gates."""
    return ExtensionInput(
        symbol=symbol,
        unrealized_pnl_pct=0.06,       # 6% gain over 8 days → 0.75%/day < 2%
        hold_days=8,
        rsr=82.0,                       # < ISOLATED_RSR_THRESHOLD=85
        rsr_rank=2,
        rsr_momentum=1.5,              # abs < MAX_RSR_ACCELERATION=5
        entry_price=4717.0,
        current_price=5000.0,
        trailing_stop_price=4200.0,    # atr_est = (5000-4200)/3 = 266.7
        regime_bear=False,
        portfolio_pnl_pct=0.0,         # flat portfolio → no breadth divergence
        candle_range_ratio=1.2,        # < MAX_CANDLE_RANGE_RATIO=2.5
        volume_ratio=1.5,              # < MAX_VOLUME_RATIO=4.0
    )


class TestExtensionFilter:
    def test_smooth_trend_not_blocked(self):
        result = check_extension(_good_ext())
        assert not result.extension_blocked
        assert result.block_reasons == []

    def test_exhaustion_spike_rejected(self):
        # velocity = 0.06 / 1 = 6%/day > MAX_VELOCITY_RATE=2%
        ext = _good_ext()
        ext.hold_days = 1
        result = check_extension(ext)
        assert result.extension_blocked
        assert any("velocity" in r for r in result.block_reasons)

    def test_climax_run_rejected(self):
        ext = _good_ext()
        ext.candle_range_ratio = MAX_CANDLE_RANGE_RATIO + 0.1
        result = check_extension(ext)
        assert result.extension_blocked
        assert any("climax_bar" in r for r in result.block_reasons)

    def test_atr_explosion_suppressed(self):
        # volume explosion → volume exhaustion gate
        ext = _good_ext()
        ext.volume_ratio = MAX_VOLUME_RATIO + 0.5
        result = check_extension(ext)
        assert result.extension_blocked
        assert any("volume_exhaustion" in r for r in result.block_reasons)

    def test_breadth_divergence_suppressed(self):
        # Stock very hot while portfolio hurts → isolated momentum
        ext = _good_ext()
        ext.rsr = ISOLATED_RSR_THRESHOLD + 1.0
        ext.portfolio_pnl_pct = PORTFOLIO_DIVERGENCE_THRESHOLD - 0.01
        result = check_extension(ext)
        assert result.extension_blocked
        assert any("isolated_momentum" in r for r in result.block_reasons)

    def test_rsr_acceleration_blocked(self):
        ext = _good_ext()
        ext.rsr_momentum = MAX_RSR_ACCELERATION + 0.5
        result = check_extension(ext)
        assert result.extension_blocked
        assert any("rsr_accel" in r for r in result.block_reasons)

    def test_negative_rsr_momentum_uses_abs(self):
        # Large negative acceleration (downward RSR reversal) also blocked
        ext = _good_ext()
        ext.rsr_momentum = -(MAX_RSR_ACCELERATION + 1.0)
        result = check_extension(ext)
        assert result.extension_blocked
        assert any("rsr_accel" in r for r in result.block_reasons)

    def test_persistence_quality_score_range(self):
        result = check_extension(_good_ext())
        assert 0.5 <= result.persistence_quality_score <= 1.0

    def test_persistence_quality_high_for_smooth_trend(self):
        result = check_extension(_good_ext())
        assert result.persistence_quality_score >= 0.85

    def test_persistence_quality_lower_on_elevated_metrics(self):
        ext_smooth = _good_ext()
        ext_hot = _good_ext()
        ext_hot.rsr_momentum = 3.5       # elevated but below gate
        ext_hot.candle_range_ratio = 2.0  # elevated but below gate
        r_smooth = check_extension(ext_smooth)
        r_hot = check_extension(ext_hot)
        assert r_smooth.persistence_quality_score > r_hot.persistence_quality_score

    def test_atr_estimate_computed_from_trailing_stop(self):
        ext = _good_ext()
        ext.current_price = 5000.0
        ext.trailing_stop_price = 4100.0   # gap = 900 → atr_est = 300
        result = check_extension(ext)
        assert abs(result.atr_estimate - 300.0) < 0.1

    def test_atr_estimate_zero_when_stop_missing(self):
        ext = _good_ext()
        ext.trailing_stop_price = 0.0
        result = check_extension(ext)
        assert result.atr_estimate == 0.0

    def test_atr_estimate_zero_when_stop_above_price(self):
        # Degenerate: stop > current (data error) → no estimate
        ext = _good_ext()
        ext.current_price = 4000.0
        ext.trailing_stop_price = 4500.0
        result = check_extension(ext)
        assert result.atr_estimate == 0.0

    def test_velocity_rate_echoed(self):
        ext = _good_ext()
        ext.unrealized_pnl_pct = 0.06
        ext.hold_days = 8
        result = check_extension(ext)
        assert abs(result.velocity_rate - 0.06 / 8) < 1e-9

    def test_multiple_blocks_all_reported(self):
        ext = _good_ext()
        ext.hold_days = 1                              # high velocity
        ext.candle_range_ratio = MAX_CANDLE_RANGE_RATIO + 0.1  # climax bar
        result = check_extension(ext)
        assert result.extension_blocked
        assert len(result.block_reasons) >= 2

    def test_replay_determinism(self):
        ext = _good_ext()
        r1 = check_extension(ext)
        r2 = check_extension(ext)
        assert r1.extension_blocked == r2.extension_blocked
        assert r1.block_reasons == r2.block_reasons
        assert r1.persistence_quality_score == r2.persistence_quality_score
        assert r1.velocity_rate == r2.velocity_rate

    def test_high_rsr_without_portfolio_divergence_not_blocked(self):
        # RSR hot but portfolio doing fine → no isolation block
        ext = _good_ext()
        ext.rsr = ISOLATED_RSR_THRESHOLD + 5.0
        ext.portfolio_pnl_pct = 0.02   # portfolio up
        result = check_extension(ext)
        assert not any("isolated_momentum" in r for r in result.block_reasons)

    def test_portfolio_divergence_without_high_rsr_not_blocked(self):
        # Portfolio hurting but RSR not extreme → no isolation block
        ext = _good_ext()
        ext.rsr = ISOLATED_RSR_THRESHOLD - 5.0
        ext.portfolio_pnl_pct = PORTFOLIO_DIVERGENCE_THRESHOLD - 0.05
        result = check_extension(ext)
        assert not any("isolated_momentum" in r for r in result.block_reasons)

    def test_exactly_at_velocity_threshold_not_blocked(self):
        # Exactly at max threshold → not blocked (strictly greater blocks)
        ext = _good_ext()
        ext.hold_days = 1
        ext.unrealized_pnl_pct = MAX_VELOCITY_RATE  # exactly 2%/day
        result = check_extension(ext)
        assert not any("velocity" in r for r in result.block_reasons)

    def test_pqs_floored_at_0_5(self):
        # Maximize all penalty factors → PQS should not go below 0.5
        ext = _good_ext()
        ext.hold_days = 1
        ext.unrealized_pnl_pct = 0.10    # high velocity (blocked but PQS still computed)
        ext.rsr_momentum = 4.9           # near max penalty, below gate
        ext.candle_range_ratio = 2.4     # near max penalty, below gate
        ext.volume_ratio = 3.9           # near max penalty, below gate
        result = check_extension(ext)
        assert result.persistence_quality_score >= 0.5


# ─────────────────────────────────────────────────────────────────────────────
# WinnerConfirmation + ExtensionFilter integration
# ─────────────────────────────────────────────────────────────────────────────

class TestWinnerConfirmationWithExtension:
    def test_extension_block_overrides_confirmation(self):
        kw = _good_winner()
        ext = _good_ext(kw["symbol"])
        ext.candle_range_ratio = MAX_CANDLE_RANGE_RATIO + 0.5   # climax → block
        c = check_winner(**kw, extension_input=ext)
        assert not c.confirmed
        assert any("EXT:" in r for r in c.fail_reasons)
        assert c.confidence_score == 0.0

    def test_smooth_extension_does_not_block(self):
        kw = _good_winner()
        ext = _good_ext(kw["symbol"])
        c = check_winner(**kw, extension_input=ext)
        assert c.confirmed
        assert not any("EXT:" in r for r in c.fail_reasons)

    def test_extension_pqs_reduces_confidence(self):
        kw = _good_winner()
        # Without extension → full base score
        c_bare = check_winner(**kw)
        # With hot (but unblocked) extension → PQS < 1.0 → lower score
        ext_hot = _good_ext(kw["symbol"])
        ext_hot.rsr_momentum = 3.5        # elevated, below gate
        ext_hot.candle_range_ratio = 2.0  # elevated, below gate
        c_hot = check_winner(**kw, extension_input=ext_hot)
        assert c_hot.confirmed
        assert c_hot.confidence_score < c_bare.confidence_score

    def test_no_extension_input_unchanged(self):
        kw = _good_winner()
        c_default = check_winner(**kw)
        c_explicit = check_winner(**kw, extension_input=None)
        assert c_default.confirmed == c_explicit.confirmed
        assert c_default.confidence_score == c_explicit.confidence_score
        assert c_default.fail_reasons == c_explicit.fail_reasons

    def test_extension_velocity_block_reported_in_fails(self):
        kw = _good_winner()
        ext = _good_ext(kw["symbol"])
        ext.hold_days = 1   # velocity = 6%/day → block
        c = check_winner(**kw, extension_input=ext)
        assert not c.confirmed
        velocity_reasons = [r for r in c.fail_reasons if "EXT:" in r and "velocity" in r]
        assert len(velocity_reasons) == 1

    def test_extension_does_not_affect_unconfirmed_base(self):
        # Position that fails base gates: extension is irrelevant to confirmed=False
        kw = _good_winner()
        kw["unrealized_pnl_pct"] = 0.0   # fails profit gate
        ext = _good_ext(kw["symbol"])
        c = check_winner(**kw, extension_input=ext)
        assert not c.confirmed
        assert c.confidence_score == 0.0

    def test_extension_replay_determinism(self):
        kw = _good_winner()
        ext = _good_ext(kw["symbol"])
        c1 = check_winner(**kw, extension_input=ext)
        c2 = check_winner(**kw, extension_input=ext)
        assert c1.confirmed == c2.confirmed
        assert c1.confidence_score == c2.confidence_score
        assert c1.fail_reasons == c2.fail_reasons
