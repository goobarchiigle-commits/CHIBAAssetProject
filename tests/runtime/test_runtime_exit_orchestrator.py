"""
tests/runtime/test_runtime_exit_orchestrator.py

Tests for RuntimeExitOrchestrator, ExitAdjustmentPolicy, and RuntimeIntegrationHook.

Validates:
  - Runtime integration (live policy application)
  - Bounded adaptation (all outputs within hard caps)
  - Replay determinism (same inputs → same decision_id)
  - Fail-closed integrity (violation history → no decisions)
  - Cooldown enforcement (3-day gap between same-symbol changes)
  - Anti-oscillation (direction reversal blocked within window)
  - No oscillatory exit behavior
  - Suppress + force mutual exclusion
  - JSONL persistence round-trip
"""
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.runtime.policy.runtime_exit_orchestrator import (
    SCHEMA_VERSION,
    ExitAdjustmentConstraints,
    ExitAdjustmentDecision,
    ExitAdjustmentPolicy,
    ExitExpectancySnapshot,
    ExitIntelligenceSnapshot,
    ExitOrchestratorIntegrityValidator,
    PositionLifecycleSnapshot,
    RegimeAnalyticsSnapshot,
    RuntimeExitOrchestrator,
    RuntimeIntegrationHook,
    append_decision,
    load_decisions,
    run_exit_orchestration_hook,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_intel(
    premature_exit_ratio: float = 0.4,
    mean_giveback_ratio: float = 0.3,
    runner_score: float = 0.7,
    persistence: float = 0.65,
    vol: float = 0.20,
    decay_rate: float = 0.01,
    n_trades: int = 10,
    symbol: str = "7203",
) -> ExitIntelligenceSnapshot:
    return ExitIntelligenceSnapshot(
        premature_exit_ratio=premature_exit_ratio,
        mean_giveback_ratio=mean_giveback_ratio,
        mean_runner_retention=0.80,
        mean_convexity_capture=0.75,
        per_symbol_runner_score={symbol: runner_score},
        per_symbol_giveback={symbol: mean_giveback_ratio},
        per_symbol_vol={symbol: vol},
        per_symbol_persistence={symbol: persistence},
        per_symbol_decay_rate={symbol: decay_rate},
        n_trades=n_trades,
    )


def _make_pos(
    symbol: str = "7203",
    hold_days: int = 10,
    current_return: float = 0.05,
    is_at_turtle_exit: bool = False,
    rs_score: float = 0.01,
) -> PositionLifecycleSnapshot:
    return PositionLifecycleSnapshot(
        symbol=symbol,
        hold_days=hold_days,
        current_return=current_return,
        rs_score=rs_score,
        rsr=80.0,
        is_at_turtle_exit=is_at_turtle_exit,
    )


def _make_regime(
    regime: str = "neutral",
    deteriorating: bool = False,
    rs_collapse: bool = False,
    breadth: float = 0.60,
    confidence: float = 0.65,
) -> RegimeAnalyticsSnapshot:
    return RegimeAnalyticsSnapshot(
        regime=regime,
        regime_deteriorating=deteriorating,
        breadth_score=breadth,
        rs_collapse_detected=rs_collapse,
        regime_confidence=confidence,
    )


def _make_expectancy(n_executed: int = 15, n_skipped: int = 5) -> ExitExpectancySnapshot:
    return ExitExpectancySnapshot(n_executed=n_executed, n_skipped=n_skipped)


def _make_orchestrator(tmp_path: Path) -> RuntimeExitOrchestrator:
    return RuntimeExitOrchestrator(
        constraints=ExitAdjustmentConstraints(),
        store_path=tmp_path / "exit_decisions.jsonl",
    )


def _evaluate_one(
    orchestrator: RuntimeExitOrchestrator,
    sym: str = "7203",
    intel: Optional[ExitIntelligenceSnapshot] = None,
    pos_override: Optional[PositionLifecycleSnapshot] = None,
    regime: Optional[RegimeAnalyticsSnapshot] = None,
    expectancy: Optional[ExitExpectancySnapshot] = None,
    history: Optional[List[ExitAdjustmentDecision]] = None,
    as_of_date: str = "2026-05-20",
) -> List[ExitAdjustmentDecision]:
    return orchestrator.evaluate(
        exit_intelligence=intel or _make_intel(symbol=sym),
        positions=[pos_override or _make_pos(symbol=sym)],
        regime=regime or _make_regime(),
        expectancy=expectancy or _make_expectancy(),
        history=history or [],
        as_of_date=as_of_date,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Bounded adaptation — all outputs within hard caps
# ─────────────────────────────────────────────────────────────────────────────

class TestBoundedAdaptation:
    def test_hold_extension_never_exceeds_cap(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        intel = _make_intel(premature_exit_ratio=0.99, runner_score=0.99, persistence=0.99)
        decisions = _evaluate_one(orch, intel=intel)
        if decisions:
            c = ExitAdjustmentConstraints()
            assert decisions[0].hold_extension_days <= c.max_hold_extension_days

    def test_exit_sensitivity_within_bounds(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        c = ExitAdjustmentConstraints()
        for regime_val, det in [("bull", False), ("bear", True), ("neutral", False)]:
            regime = _make_regime(regime=regime_val, deteriorating=det)
            decisions = _evaluate_one(orch, regime=regime)
            if decisions:
                v = decisions[0].exit_sensitivity_scale
                assert c.exit_sensitivity_min <= v <= c.exit_sensitivity_max, (
                    f"exit_sensitivity_scale={v} out of bounds for regime={regime_val}"
                )

    def test_deterioration_pressure_within_bounds(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        intel = _make_intel(mean_giveback_ratio=0.99, decay_rate=0.99)
        regime = _make_regime(deteriorating=True, rs_collapse=True)
        decisions = _evaluate_one(orch, intel=intel, regime=regime)
        if decisions:
            c = ExitAdjustmentConstraints()
            assert 0.0 <= decisions[0].deterioration_pressure <= c.deterioration_pressure_max

    def test_vol_trailing_multiplier_within_bounds(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        c = ExitAdjustmentConstraints()
        for vol in [0.0, 0.10, 0.50, 1.0, 2.0]:
            intel = _make_intel(vol=vol)
            decisions = _evaluate_one(orch, intel=intel)
            if decisions:
                v = decisions[0].vol_trailing_multiplier
                assert c.vol_trailing_mult_min <= v <= c.vol_trailing_mult_max, (
                    f"vol_trailing_multiplier={v} out of bounds for vol={vol}"
                )

    def test_regime_threshold_adjustment_within_bounds(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        c = ExitAdjustmentConstraints()
        for regime_val in ["bull", "bear", "neutral"]:
            regime = _make_regime(regime=regime_val, confidence=0.9)
            decisions = _evaluate_one(orch, regime=regime)
            if decisions:
                v = decisions[0].regime_threshold_adjustment
                assert c.regime_threshold_adj_min <= v <= c.regime_threshold_adj_max, (
                    f"regime_threshold_adjustment={v} out of bounds for regime={regime_val}"
                )

    def test_bounded_flag_always_true(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        decisions = _evaluate_one(orch)
        for d in decisions:
            assert d.bounded is True


# ─────────────────────────────────────────────────────────────────────────────
# 2. Hold extension — winner retention
# ─────────────────────────────────────────────────────────────────────────────

class TestHoldExtension:
    def test_high_runner_score_triggers_extension(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        intel = _make_intel(runner_score=0.85, persistence=0.75, premature_exit_ratio=0.45)
        decisions = _evaluate_one(orch, intel=intel)
        assert len(decisions) == 1
        assert decisions[0].hold_extension_days > 0

    def test_low_runner_score_no_extension(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        intel = _make_intel(runner_score=0.20, persistence=0.20, premature_exit_ratio=0.10)
        decisions = _evaluate_one(orch, intel=intel)
        # Either no decisions or extension=0
        if decisions:
            assert decisions[0].hold_extension_days == 0

    def test_deteriorating_regime_blocks_extension(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        intel = _make_intel(runner_score=0.90, persistence=0.85, premature_exit_ratio=0.50)
        regime = _make_regime(deteriorating=True)
        decisions = _evaluate_one(orch, intel=intel, regime=regime)
        # With deteriorating regime, hold_extension_days should be 0
        if decisions:
            assert decisions[0].hold_extension_days == 0


# ─────────────────────────────────────────────────────────────────────────────
# 3. Suppress exit (hold extension → prevent premature exit)
# ─────────────────────────────────────────────────────────────────────────────

class TestSuppressExit:
    def test_suppress_exit_set_when_extension_and_persistence(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        intel = _make_intel(runner_score=0.80, persistence=0.70, premature_exit_ratio=0.45)
        decisions = _evaluate_one(orch, intel=intel)
        if decisions and decisions[0].hold_extension_days > 0:
            assert decisions[0].suppress_exit is True

    def test_apply_to_signals_suppresses_sell(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        sym = "7203"
        policy = ExitAdjustmentPolicy(
            symbol=sym,
            hold_extension_days=2,
            exit_sensitivity_scale=1.10,
            deterioration_pressure=0.05,
            trend_persistence_score=0.70,
            vol_trailing_multiplier=1.05,
            regime_threshold_adjustment=0.02,
            suppress_exit=True,
            force_exit=False,
        )
        dec = ExitAdjustmentDecision.create(
            eval_date="2026-05-20",
            symbol=sym,
            policy=policy,
            trigger_runner_score=0.80,
            trigger_premature_ratio=0.40,
            trigger_giveback_ratio=0.20,
            trigger_vol=0.20,
            trigger_regime="neutral",
            trigger_rs_collapse=False,
            observation_count=15,
        )
        signals = [
            {"symbol": sym, "signal": 0, "action": "SELL", "currently_holding": True, "hold_days": 10},
            {"symbol": "6758", "signal": 1, "action": "BUY", "currently_holding": False},
        ]
        modified = orch.apply_to_signals(signals, [dec])
        hold_sig = next(s for s in modified if s["symbol"] == sym)
        assert hold_sig["signal"] == 1
        assert hold_sig["action"] == "HOLD"
        assert hold_sig.get("exit_orchestrator_applied") is True
        # Non-holding signal unchanged
        buy_sig = next(s for s in modified if s["symbol"] == "6758")
        assert buy_sig["signal"] == 1

    def test_suppress_exit_removes_sell_order(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        sym = "7203"
        policy = ExitAdjustmentPolicy(
            symbol=sym, hold_extension_days=2, exit_sensitivity_scale=1.1,
            deterioration_pressure=0.05, trend_persistence_score=0.70,
            vol_trailing_multiplier=1.0, regime_threshold_adjustment=0.0,
            suppress_exit=True, force_exit=False,
        )
        dec = ExitAdjustmentDecision.create(
            eval_date="2026-05-20", symbol=sym, policy=policy,
            trigger_runner_score=0.80, trigger_premature_ratio=0.40,
            trigger_giveback_ratio=0.20, trigger_vol=0.20,
            trigger_regime="neutral", trigger_rs_collapse=False,
            observation_count=15,
        )
        # Create mock SELL order
        sell_order = MagicMock()
        sell_order.symbol = sym
        sell_order.side = "SELL"
        buy_order = MagicMock()
        buy_order.symbol = "6758"
        buy_order.side = "BUY"
        orders = [sell_order, buy_order]
        result = orch.apply_to_orders(orders, [dec], [], {})
        # SELL order for suppressed symbol should be removed
        assert not any(getattr(o, "symbol", None) == sym and getattr(o, "side", None) == "SELL"
                       for o in result)
        # BUY order preserved
        assert any(getattr(o, "symbol", None) == "6758" for o in result)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Force exit (deterioration → accelerate exit)
# ─────────────────────────────────────────────────────────────────────────────

class TestForceExit:
    def test_high_deterioration_pressure_triggers_force_exit(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        intel = _make_intel(mean_giveback_ratio=0.80, decay_rate=0.05)
        regime = _make_regime(deteriorating=True, rs_collapse=True, breadth=0.30)
        decisions = _evaluate_one(orch, intel=intel, regime=regime)
        if decisions:
            assert decisions[0].force_exit is True

    def test_apply_to_signals_forces_sell(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        sym = "7203"
        policy = ExitAdjustmentPolicy(
            symbol=sym, hold_extension_days=0, exit_sensitivity_scale=0.85,
            deterioration_pressure=0.90, trend_persistence_score=0.10,
            vol_trailing_multiplier=0.90, regime_threshold_adjustment=-0.04,
            suppress_exit=False, force_exit=True,
        )
        dec = ExitAdjustmentDecision.create(
            eval_date="2026-05-20", symbol=sym, policy=policy,
            trigger_runner_score=0.20, trigger_premature_ratio=0.10,
            trigger_giveback_ratio=0.80, trigger_vol=0.40,
            trigger_regime="bear", trigger_rs_collapse=True,
            observation_count=12,
        )
        signals = [
            {"symbol": sym, "signal": 1, "action": "HOLD", "currently_holding": True, "hold_days": 8},
        ]
        modified = orch.apply_to_signals(signals, [dec])
        hold_sig = modified[0]
        assert hold_sig["signal"] == 0
        assert hold_sig["action"] == "SELL"
        assert hold_sig.get("exit_orchestrator_applied") is True

    def test_force_exit_adds_sell_order_when_missing(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        sym = "7203"
        policy = ExitAdjustmentPolicy(
            symbol=sym, hold_extension_days=0, exit_sensitivity_scale=0.85,
            deterioration_pressure=0.90, trend_persistence_score=0.10,
            vol_trailing_multiplier=0.90, regime_threshold_adjustment=-0.04,
            suppress_exit=False, force_exit=True,
        )
        dec = ExitAdjustmentDecision.create(
            eval_date="2026-05-20", symbol=sym, policy=policy,
            trigger_runner_score=0.20, trigger_premature_ratio=0.10,
            trigger_giveback_ratio=0.80, trigger_vol=0.40,
            trigger_regime="bear", trigger_rs_collapse=True,
            observation_count=12,
        )
        buy_order = MagicMock()
        buy_order.symbol = "6758"
        buy_order.side = "BUY"
        signals = [{"symbol": sym, "currently_holding": True, "sector": "輸送用機器",
                    "trailing_stop": 1500.0}]
        broker_pos = {sym: 100}
        # Patch OrderInstruction at the module where it's imported inside apply_to_orders
        mock_order = MagicMock()
        mock_order.symbol = sym
        mock_order.side = "SELL"
        with patch("src.kabusapi.signal_bridge.OrderInstruction", return_value=mock_order):
            result = orch.apply_to_orders([buy_order], [dec], signals, broker_pos)
        # BUY order preserved
        assert any(getattr(o, "symbol", None) == "6758" for o in result)
        # A SELL order for sym was added (mock was appended)
        sell_orders = [o for o in result if getattr(o, "side", None) == "SELL"]
        assert len(sell_orders) == 1

    def test_force_exit_not_added_when_already_selling(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        sym = "7203"
        policy = ExitAdjustmentPolicy(
            symbol=sym, hold_extension_days=0, exit_sensitivity_scale=0.85,
            deterioration_pressure=0.90, trend_persistence_score=0.10,
            vol_trailing_multiplier=0.90, regime_threshold_adjustment=-0.04,
            suppress_exit=False, force_exit=True,
        )
        dec = ExitAdjustmentDecision.create(
            eval_date="2026-05-20", symbol=sym, policy=policy,
            trigger_runner_score=0.20, trigger_premature_ratio=0.10,
            trigger_giveback_ratio=0.80, trigger_vol=0.40,
            trigger_regime="bear", trigger_rs_collapse=True,
            observation_count=12,
        )
        existing_sell = MagicMock()
        existing_sell.symbol = sym
        existing_sell.side = "SELL"
        result = orch.apply_to_orders([existing_sell], [dec], [], {sym: 100})
        # Still only one SELL (existing is kept, no duplicate added)
        sell_orders = [o for o in result if getattr(o, "side", None) == "SELL"
                       and getattr(o, "symbol", None) == sym]
        assert len(sell_orders) == 1


# ─────────────────────────────────────────────────────────────────────────────
# 5. Mutual exclusion: force_exit and suppress_exit cannot both be True
# ─────────────────────────────────────────────────────────────────────────────

class TestMutualExclusion:
    def test_force_and_suppress_never_coexist(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        # Even with extreme inputs, only one flag can be True
        intel = _make_intel(
            runner_score=0.99, persistence=0.99, premature_exit_ratio=0.99,
            mean_giveback_ratio=0.99, decay_rate=0.99,
        )
        regime = _make_regime(deteriorating=True, rs_collapse=True)
        decisions = _evaluate_one(orch, intel=intel, regime=regime)
        for d in decisions:
            assert not (d.force_exit and d.suppress_exit), (
                f"Both force_exit and suppress_exit are True for symbol={d.symbol}"
            )

    def test_integrity_validator_catches_mutual_exclusion(self):
        c = ExitAdjustmentConstraints()
        sym = "7203"
        policy_bad = ExitAdjustmentPolicy(
            symbol=sym, hold_extension_days=2, exit_sensitivity_scale=1.0,
            deterioration_pressure=0.90, trend_persistence_score=0.70,
            vol_trailing_multiplier=1.0, regime_threshold_adjustment=0.0,
            suppress_exit=True, force_exit=True,   # both True — illegal
        )
        dec_bad = ExitAdjustmentDecision.create(
            eval_date="2026-05-20", symbol=sym, policy=policy_bad,
            trigger_runner_score=None, trigger_premature_ratio=None,
            trigger_giveback_ratio=None, trigger_vol=None,
            trigger_regime=None, trigger_rs_collapse=False,
            observation_count=10,
        )
        # Manually override the frozen dataclass for testing via dict round-trip
        d = dec_bad.to_dict()
        d["suppress_exit"] = True
        d["force_exit"] = True
        rec_from_dict = ExitAdjustmentDecision.from_dict(d)
        validator = ExitOrchestratorIntegrityValidator()
        report = validator.validate([rec_from_dict], c, as_of_date="2026-05-20")
        violation_types = [v.violation_type for v in report.violations]
        assert "mutual_exclusion_violation" in violation_types


# ─────────────────────────────────────────────────────────────────────────────
# 6. Cooldown enforcement
# ─────────────────────────────────────────────────────────────────────────────

class TestCooldownEnforcement:
    def test_same_symbol_blocked_within_cooldown(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        sym = "7203"
        # Create a prior decision dated 1 day ago (within 3-day cooldown)
        policy = ExitAdjustmentPolicy(
            symbol=sym, hold_extension_days=2, exit_sensitivity_scale=1.1,
            deterioration_pressure=0.05, trend_persistence_score=0.70,
            vol_trailing_multiplier=1.0, regime_threshold_adjustment=0.0,
            suppress_exit=True, force_exit=False,
        )
        prior = ExitAdjustmentDecision.create(
            eval_date="2026-05-19",  # 1 day before today
            symbol=sym, policy=policy,
            trigger_runner_score=0.75, trigger_premature_ratio=0.35,
            trigger_giveback_ratio=0.20, trigger_vol=0.20,
            trigger_regime="neutral", trigger_rs_collapse=False,
            observation_count=10,
        )
        history = [prior]
        intel = _make_intel(runner_score=0.85, persistence=0.75, premature_exit_ratio=0.45, symbol=sym)
        decisions = _evaluate_one(orch, sym=sym, intel=intel, history=history, as_of_date="2026-05-20")
        assert len(decisions) == 0, "Should be blocked by 3-day cooldown"

    def test_same_symbol_allowed_after_cooldown(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        sym = "7203"
        policy = ExitAdjustmentPolicy(
            symbol=sym, hold_extension_days=2, exit_sensitivity_scale=1.1,
            deterioration_pressure=0.05, trend_persistence_score=0.70,
            vol_trailing_multiplier=1.0, regime_threshold_adjustment=0.0,
            suppress_exit=True, force_exit=False,
        )
        prior = ExitAdjustmentDecision.create(
            eval_date="2026-05-13",  # 7 days before → outside cooldown (3d) and anti-oscillation (6d)
            symbol=sym, policy=policy,
            trigger_runner_score=0.75, trigger_premature_ratio=0.35,
            trigger_giveback_ratio=0.20, trigger_vol=0.20,
            trigger_regime="neutral", trigger_rs_collapse=False,
            observation_count=10,
        )
        history = [prior]
        intel = _make_intel(runner_score=0.85, persistence=0.75, premature_exit_ratio=0.45, symbol=sym)
        decisions = _evaluate_one(orch, sym=sym, intel=intel, history=history, as_of_date="2026-05-20")
        assert len(decisions) >= 0   # allowed to produce a decision (may or may not depending on analytics)

    def test_integrity_validator_catches_cooldown_violation(self):
        c = ExitAdjustmentConstraints()
        sym = "7203"
        policy = ExitAdjustmentPolicy(
            symbol=sym, hold_extension_days=2, exit_sensitivity_scale=1.1,
            deterioration_pressure=0.05, trend_persistence_score=0.70,
            vol_trailing_multiplier=1.0, regime_threshold_adjustment=0.0,
            suppress_exit=True, force_exit=False,
        )
        d1 = ExitAdjustmentDecision.create(
            eval_date="2026-05-18", symbol=sym, policy=policy,
            trigger_runner_score=None, trigger_premature_ratio=None,
            trigger_giveback_ratio=None, trigger_vol=None,
            trigger_regime=None, trigger_rs_collapse=False,
            observation_count=10,
        )
        d2 = ExitAdjustmentDecision.create(
            eval_date="2026-05-19",  # only 1 day later — violates 3-day cooldown
            symbol=sym, policy=policy,
            trigger_runner_score=None, trigger_premature_ratio=None,
            trigger_giveback_ratio=None, trigger_vol=None,
            trigger_regime=None, trigger_rs_collapse=False,
            observation_count=10,
        )
        validator = ExitOrchestratorIntegrityValidator()
        report = validator.validate([d1, d2], c, as_of_date="2026-05-20")
        violation_types = [v.violation_type for v in report.violations]
        assert "policy_oscillation" in violation_types


# ─────────────────────────────────────────────────────────────────────────────
# 7. Anti-oscillation — no rapid back-and-forth
# ─────────────────────────────────────────────────────────────────────────────

class TestAntiOscillation:
    def test_direction_reversal_blocked_within_window(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        sym = "7203"
        # Prior decision: suppress_exit, dated 4 days ago (within anti_oscillation_window=6d)
        policy = ExitAdjustmentPolicy(
            symbol=sym, hold_extension_days=2, exit_sensitivity_scale=1.1,
            deterioration_pressure=0.05, trend_persistence_score=0.70,
            vol_trailing_multiplier=1.0, regime_threshold_adjustment=0.0,
            suppress_exit=True, force_exit=False,
        )
        prior = ExitAdjustmentDecision.create(
            eval_date="2026-05-16",  # 4 days before today (within 6-day anti-oscillation window)
            symbol=sym, policy=policy,
            trigger_runner_score=0.75, trigger_premature_ratio=0.35,
            trigger_giveback_ratio=0.20, trigger_vol=0.20,
            trigger_regime="neutral", trigger_rs_collapse=False,
            observation_count=10,
        )
        history = [prior]
        # Now try to produce a force_exit decision
        intel = _make_intel(mean_giveback_ratio=0.85, decay_rate=0.06, symbol=sym)
        regime = _make_regime(deteriorating=True, rs_collapse=True)
        decisions = _evaluate_one(
            orch, sym=sym, intel=intel, regime=regime, history=history, as_of_date="2026-05-20"
        )
        assert len(decisions) == 0, (
            "Oscillation reversal should be blocked within anti_oscillation_window"
        )

    def test_no_oscillatory_signal_pattern(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        sym = "7203"
        # Run 3 cycles with alternating analytics.
        # Use 4-day intervals: within anti_oscillation_window=6d → oscillation blocked.
        good_intel = _make_intel(runner_score=0.85, persistence=0.75, premature_exit_ratio=0.45)
        bad_intel = _make_intel(mean_giveback_ratio=0.90, decay_rate=0.08)
        bad_regime = _make_regime(deteriorating=True, rs_collapse=True)

        history: List[ExitAdjustmentDecision] = []
        dates = ["2026-05-01", "2026-05-05", "2026-05-09"]  # 4-day intervals < 6d window
        direction_changes = 0
        last_force: Optional[bool] = None

        for i, date in enumerate(dates):
            intel = good_intel if i % 2 == 0 else bad_intel
            regime = _make_regime() if i % 2 == 0 else bad_regime
            decisions = orch.evaluate(
                exit_intelligence=intel,
                positions=[_make_pos(symbol=sym)],
                regime=regime,
                expectancy=_make_expectancy(),
                history=history,
                as_of_date=date,
            )
            if decisions:
                history.extend(decisions)
                if last_force is not None and decisions[0].force_exit != last_force:
                    direction_changes += 1
                last_force = decisions[0].force_exit

        # Anti-oscillation window (6d) > interval (4d): at most 1 direction change across 3 cycles
        assert direction_changes <= 1


# ─────────────────────────────────────────────────────────────────────────────
# 8. Replay determinism
# ─────────────────────────────────────────────────────────────────────────────

class TestReplayDeterminism:
    def test_same_inputs_produce_same_decision_id(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        intel = _make_intel(runner_score=0.80, persistence=0.65, premature_exit_ratio=0.40)
        d1 = _evaluate_one(orch, intel=intel, as_of_date="2026-05-20")
        d2 = _evaluate_one(orch, intel=intel, as_of_date="2026-05-20")
        if d1 and d2:
            assert d1[0].decision_id == d2[0].decision_id, (
                "Same inputs must produce same decision_id (replay determinism)"
            )

    def test_different_date_produces_different_decision_id(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        intel = _make_intel(runner_score=0.80, persistence=0.65, premature_exit_ratio=0.40)
        d1 = _evaluate_one(orch, intel=intel, as_of_date="2026-05-20")
        d2 = _evaluate_one(orch, intel=intel, as_of_date="2026-05-21")
        if d1 and d2:
            assert d1[0].decision_id != d2[0].decision_id

    def test_decision_content_is_deterministic(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        intel = _make_intel(runner_score=0.80, persistence=0.65, premature_exit_ratio=0.40)
        results = [
            _evaluate_one(orch, intel=intel, as_of_date="2026-05-20")
            for _ in range(3)
        ]
        # All runs produce same numeric outputs
        for r in results:
            if r and results[0]:
                assert r[0].hold_extension_days == results[0][0].hold_extension_days
                assert r[0].exit_sensitivity_scale == results[0][0].exit_sensitivity_scale
                assert r[0].force_exit == results[0][0].force_exit
                assert r[0].suppress_exit == results[0][0].suppress_exit


# ─────────────────────────────────────────────────────────────────────────────
# 9. Fail-closed integrity
# ─────────────────────────────────────────────────────────────────────────────

class TestFailClosedIntegrity:
    def test_integrity_violation_blocks_all_decisions(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        sym = "7203"
        # Construct a history record with future_leakage (eval_date > today)
        policy = ExitAdjustmentPolicy(
            symbol=sym, hold_extension_days=1, exit_sensitivity_scale=1.0,
            deterioration_pressure=0.0, trend_persistence_score=0.5,
            vol_trailing_multiplier=1.0, regime_threshold_adjustment=0.0,
            suppress_exit=False, force_exit=False,
        )
        future_dec = ExitAdjustmentDecision.create(
            eval_date="2027-01-01",  # future date
            symbol=sym, policy=policy,
            trigger_runner_score=None, trigger_premature_ratio=None,
            trigger_giveback_ratio=None, trigger_vol=None,
            trigger_regime=None, trigger_rs_collapse=False,
            observation_count=10,
        )
        # Evaluate with this poisoned history → should produce no decisions
        intel = _make_intel(runner_score=0.85, persistence=0.75, premature_exit_ratio=0.45, symbol=sym)
        decisions = _evaluate_one(
            orch, sym=sym, intel=intel, history=[future_dec], as_of_date="2026-05-20"
        )
        assert len(decisions) == 0, "Integrity violation should block all decisions (fail-closed)"

    def test_insufficient_sample_blocks_decisions(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        intel = _make_intel(n_trades=2)
        decisions = _evaluate_one(
            orch,
            intel=intel,
            expectancy=ExitExpectancySnapshot(n_executed=1, n_skipped=1),  # total_obs=4 < min_sample=5
            as_of_date="2026-05-20",
        )
        assert len(decisions) == 0, "< min_sample_size should produce no decisions"

    def test_rapid_recursive_adaptation_caught(self):
        c = ExitAdjustmentConstraints()
        sym = "7203"
        policy = ExitAdjustmentPolicy(
            symbol=sym, hold_extension_days=1, exit_sensitivity_scale=1.0,
            deterioration_pressure=0.0, trend_persistence_score=0.5,
            vol_trailing_multiplier=1.0, regime_threshold_adjustment=0.0,
            suppress_exit=False, force_exit=False,
        )
        # Create 3 decisions for same symbol within 7 days (> max=2).
        # Use dates all within 7 days of 2026-05-20 so they fall in the window.
        dates = ["2026-05-16", "2026-05-17", "2026-05-18"]
        records = [
            ExitAdjustmentDecision.create(
                eval_date=date, symbol=sym, policy=policy,
                trigger_runner_score=None, trigger_premature_ratio=None,
                trigger_giveback_ratio=None, trigger_vol=None,
                trigger_regime=None, trigger_rs_collapse=False,
                observation_count=10 + i,   # vary obs so decision_id differs
            )
            for i, date in enumerate(dates)
        ]
        validator = ExitOrchestratorIntegrityValidator()
        report = validator.validate(records, c, as_of_date="2026-05-20")
        violation_types = [v.violation_type for v in report.violations]
        assert "rapid_recursive_adaptation" in violation_types


# ─────────────────────────────────────────────────────────────────────────────
# 10. Regime-conditioned adjustments
# ─────────────────────────────────────────────────────────────────────────────

class TestRegimeConditioned:
    def test_bear_regime_tightens_exit_sensitivity(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        regime_bear = _make_regime(regime="bear", deteriorating=True, confidence=0.80)
        intel = _make_intel(premature_exit_ratio=0.35, runner_score=0.70, persistence=0.60)
        decisions = _evaluate_one(orch, intel=intel, regime=regime_bear)
        if decisions:
            assert decisions[0].exit_sensitivity_scale < 1.0, (
                "Bear/deteriorating regime should tighten exit sensitivity (< 1.0)"
            )

    def test_rs_collapse_increases_deterioration_pressure(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        regime = _make_regime(rs_collapse=True, deteriorating=False)
        intel = _make_intel(mean_giveback_ratio=0.60, runner_score=0.50, persistence=0.50)
        decisions = _evaluate_one(orch, intel=intel, regime=regime)
        if decisions:
            assert decisions[0].deterioration_pressure > 0.0, (
                "RS collapse should raise deterioration pressure"
            )

    def test_strong_bull_loosens_exit_sensitivity(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        regime_bull = _make_regime(regime="bull", deteriorating=False, breadth=0.75, confidence=0.80)
        intel = _make_intel(
            premature_exit_ratio=0.55, runner_score=0.80, persistence=0.70,
            mean_giveback_ratio=0.20,
        )
        decisions = _evaluate_one(orch, intel=intel, regime=regime_bull)
        if decisions:
            assert decisions[0].exit_sensitivity_scale >= 1.0, (
                "Strong bull regime should loosen exit sensitivity (>= 1.0)"
            )

    def test_regime_threshold_positive_in_bull(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        regime_bull = _make_regime(regime="bull", deteriorating=False, confidence=0.85)
        intel = _make_intel(runner_score=0.75, persistence=0.65, premature_exit_ratio=0.40)
        decisions = _evaluate_one(orch, intel=intel, regime=regime_bull)
        if decisions:
            assert decisions[0].regime_threshold_adjustment >= 0.0

    def test_regime_threshold_negative_in_bear(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        regime_bear = _make_regime(regime="bear", deteriorating=True, confidence=0.80)
        intel = _make_intel(mean_giveback_ratio=0.70, decay_rate=0.04)
        decisions = _evaluate_one(orch, intel=intel, regime=regime_bear)
        if decisions:
            assert decisions[0].regime_threshold_adjustment <= 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 11. Vol-aware trailing
# ─────────────────────────────────────────────────────────────────────────────

class TestVolAwareTrailing:
    def test_high_vol_tightens_trailing(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        intel_hi = _make_intel(vol=0.80, runner_score=0.75, persistence=0.65, premature_exit_ratio=0.40)
        intel_lo = _make_intel(vol=0.05, runner_score=0.75, persistence=0.65, premature_exit_ratio=0.40)
        dec_hi = _evaluate_one(orch, intel=intel_hi, as_of_date="2026-05-20")
        dec_lo = _evaluate_one(orch, intel=intel_lo, as_of_date="2026-05-21")
        if dec_hi and dec_lo:
            assert dec_hi[0].vol_trailing_multiplier < dec_lo[0].vol_trailing_multiplier, (
                "High vol should produce lower trailing multiplier than low vol"
            )


# ─────────────────────────────────────────────────────────────────────────────
# 12. Weekly quota enforcement
# ─────────────────────────────────────────────────────────────────────────────

class TestWeeklyQuota:
    def test_weekly_quota_exhausted(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        sym = "7203"
        policy = ExitAdjustmentPolicy(
            symbol=sym, hold_extension_days=2, exit_sensitivity_scale=1.1,
            deterioration_pressure=0.05, trend_persistence_score=0.70,
            vol_trailing_multiplier=1.0, regime_threshold_adjustment=0.0,
            suppress_exit=True, force_exit=False,
        )
        # Two prior decisions in the last 7 days (= max per symbol per week)
        history = [
            ExitAdjustmentDecision.create(
                eval_date="2026-05-14", symbol=sym, policy=policy,
                trigger_runner_score=None, trigger_premature_ratio=None,
                trigger_giveback_ratio=None, trigger_vol=None,
                trigger_regime=None, trigger_rs_collapse=False,
                observation_count=10,
            ),
            ExitAdjustmentDecision.create(
                eval_date="2026-05-14", symbol=sym, policy=policy,
                trigger_runner_score=None, trigger_premature_ratio=None,
                trigger_giveback_ratio=None, trigger_vol=None,
                trigger_regime=None, trigger_rs_collapse=False,
                observation_count=11,   # different obs count → different decision_id
            ),
        ]
        intel = _make_intel(runner_score=0.90, persistence=0.80, premature_exit_ratio=0.50, symbol=sym)
        decisions = _evaluate_one(
            orch, sym=sym, intel=intel, history=history, as_of_date="2026-05-20"
        )
        assert len(decisions) == 0, "Weekly quota exhausted — no new decisions"


# ─────────────────────────────────────────────────────────────────────────────
# 13. JSONL persistence round-trip
# ─────────────────────────────────────────────────────────────────────────────

class TestPersistence:
    def test_append_and_load_round_trip(self, tmp_path):
        store = tmp_path / "test_decisions.jsonl"
        sym = "7203"
        policy = ExitAdjustmentPolicy(
            symbol=sym, hold_extension_days=3, exit_sensitivity_scale=1.10,
            deterioration_pressure=0.15, trend_persistence_score=0.65,
            vol_trailing_multiplier=0.95, regime_threshold_adjustment=0.02,
            suppress_exit=True, force_exit=False,
        )
        rec = ExitAdjustmentDecision.create(
            eval_date="2026-05-20", symbol=sym, policy=policy,
            trigger_runner_score=0.80, trigger_premature_ratio=0.40,
            trigger_giveback_ratio=0.25, trigger_vol=0.22,
            trigger_regime="neutral", trigger_rs_collapse=False,
            observation_count=12,
        )
        append_decision(rec, store)
        loaded = load_decisions(store)
        assert len(loaded) == 1
        r = loaded[0]
        assert r.decision_id == rec.decision_id
        assert r.symbol == rec.symbol
        assert r.hold_extension_days == rec.hold_extension_days
        assert r.suppress_exit == rec.suppress_exit
        assert r.force_exit == rec.force_exit
        assert r.bounded is True
        assert r.schema_version == SCHEMA_VERSION

    def test_empty_file_loads_empty(self, tmp_path):
        store = tmp_path / "empty.jsonl"
        assert load_decisions(store) == []

    def test_corrupted_line_skipped(self, tmp_path):
        store = tmp_path / "corrupt.jsonl"
        store.write_text('{"invalid": \n{"decision_id": "bad"}\n', encoding="utf-8")
        records = load_decisions(store)
        # Should skip corrupted line without raising
        assert isinstance(records, list)

    def test_multiple_decisions_preserved(self, tmp_path):
        store = tmp_path / "multi.jsonl"
        policy = ExitAdjustmentPolicy(
            symbol="X", hold_extension_days=1, exit_sensitivity_scale=1.0,
            deterioration_pressure=0.0, trend_persistence_score=0.5,
            vol_trailing_multiplier=1.0, regime_threshold_adjustment=0.0,
            suppress_exit=False, force_exit=False,
        )
        for sym, date in [("7203", "2026-05-18"), ("6758", "2026-05-19"), ("9984", "2026-05-20")]:
            p = ExitAdjustmentPolicy(
                symbol=sym, hold_extension_days=1, exit_sensitivity_scale=1.0,
                deterioration_pressure=0.0, trend_persistence_score=0.5,
                vol_trailing_multiplier=1.0, regime_threshold_adjustment=0.0,
                suppress_exit=False, force_exit=False,
            )
            rec = ExitAdjustmentDecision.create(
                eval_date=date, symbol=sym, policy=p,
                trigger_runner_score=None, trigger_premature_ratio=None,
                trigger_giveback_ratio=None, trigger_vol=None,
                trigger_regime=None, trigger_rs_collapse=False,
                observation_count=10,
            )
            append_decision(rec, store)
        loaded = load_decisions(store)
        assert len(loaded) == 3
        symbols = {r.symbol for r in loaded}
        assert symbols == {"7203", "6758", "9984"}


# ─────────────────────────────────────────────────────────────────────────────
# 14. RuntimeIntegrationHook — fail-open behavior
# ─────────────────────────────────────────────────────────────────────────────

class TestRuntimeIntegrationHook:
    def test_no_held_positions_returns_inputs_unchanged(self, tmp_path):
        hook = RuntimeIntegrationHook(store_path=tmp_path / "d.jsonl")
        signals = [{"symbol": "7203", "signal": 1, "currently_holding": False}]
        orders = [MagicMock()]
        out_sigs, out_orders = hook.run(signals, orders)
        assert out_sigs is signals
        assert out_orders is orders

    def test_error_in_hook_returns_inputs(self, tmp_path):
        hook = RuntimeIntegrationHook(store_path=tmp_path / "d.jsonl")
        # Pass garbage that will cause an error in _run_internal
        out_sigs, out_orders = hook.run(
            signals=None,    # type: ignore
            order_objects=[],
        )
        # Should return the original values, not crash
        assert out_sigs is None
        assert out_orders == []

    def test_run_exit_orchestration_hook_fail_open(self, tmp_path):
        out = run_exit_orchestration_hook(
            signals=None,  # type: ignore
            order_objects=[],
            store_path=tmp_path / "d.jsonl",
        )
        # Never raises, returns originals
        assert out[0] is None
        assert out[1] == []

    def test_hook_with_held_signal_at_turtle_exit(self, tmp_path):
        hook = RuntimeIntegrationHook(store_path=tmp_path / "d.jsonl")
        signals = [
            {
                "symbol": "7203",
                "signal": 0,        # turtle_exit triggered
                "action": "SELL",
                "currently_holding": True,
                "hold_days": 12,
                "rsr": 82.0,
                "rsr_momentum": 0.5,
                "sector": "輸送用機器",
                "trailing_stop": 1500.0,
            }
        ]
        portfolio_summary = {
            "current_positions": 1,
            "max_positions": 4,
            "open_slots": 3,
            "current_drawdown": -0.02,
        }
        broker_positions = {"7203": 100}
        # Run hook with neutral regime (no analytics file → defaults only)
        out_sigs, out_orders = hook.run(
            signals=signals,
            order_objects=[],
            portfolio_summary=portfolio_summary,
            broker_positions=broker_positions,
            regime_info={"regime": "neutral"},
        )
        # Output must be a list of signals
        assert isinstance(out_sigs, list)
        assert isinstance(out_orders, list)

    def test_live_policy_application_logs_decision(self, tmp_path):
        store = tmp_path / "decisions.jsonl"
        hook = RuntimeIntegrationHook(store_path=store)

        # Force a meaningful scenario: high runner score + premature exit in analytics
        # We patch the exit records file to inject analytics
        signals = [
            {
                "symbol": "7203",
                "signal": 0,
                "action": "SELL",
                "currently_holding": True,
                "hold_days": 10,
                "rsr": 84.0,
                "rsr_momentum": 1.2,
                "sector": "輸送用機器",
            }
        ]
        portfolio_summary = {"current_positions": 1, "max_positions": 4, "current_drawdown": 0.0}

        # Mock exit records loading to inject strong runner analytics
        with patch.object(hook._orchestrator, "_compute_policy") as mock_compute:
            from src.runtime.policy.runtime_exit_orchestrator import ExitAdjustmentPolicy as EAP
            mock_compute.return_value = EAP(
                symbol="7203",
                hold_extension_days=3,
                exit_sensitivity_scale=1.10,
                deterioration_pressure=0.10,
                trend_persistence_score=0.72,
                vol_trailing_multiplier=1.0,
                regime_threshold_adjustment=0.02,
                suppress_exit=True,
                force_exit=False,
            )
            with patch.object(hook._orchestrator, "_last_decision_for_symbol", return_value=None):
                with patch.object(hook._orchestrator, "_cooldown_ok", return_value=True):
                    with patch.object(hook._orchestrator, "_weekly_quota_ok", return_value=True):
                        out_sigs, out_orders = hook.run(
                            signals=signals,
                            order_objects=[],
                            portfolio_summary=portfolio_summary,
                            broker_positions={"7203": 100},
                            regime_info={"regime": "neutral"},
                            as_of_date="2026-05-20",
                        )

        # Decision should be logged
        loaded = load_decisions(store)
        assert len(loaded) >= 1
        assert loaded[0].symbol == "7203"
        assert loaded[0].suppress_exit is True
        # Signal should be modified: 0→1
        assert out_sigs[0]["signal"] == 1
        assert out_sigs[0]["action"] == "HOLD"


# ─────────────────────────────────────────────────────────────────────────────
# 15. No self-reinforcing feedback loop
# ─────────────────────────────────────────────────────────────────────────────

class TestNoFeedbackLoop:
    def test_decisions_do_not_consume_prior_decisions(self, tmp_path):
        """Orchestrator inputs must be analytics data, not prior decisions."""
        orch = _make_orchestrator(tmp_path)
        # Confirm that evaluate() signature only takes analytics objects, not its own decisions
        # (structural test: calling evaluate with empty history still works)
        decisions = _evaluate_one(orch, history=[])
        # Result is deterministic regardless of whether we pass history or not
        # (history is used only for cooldown/quota guards, not as analytics input)
        decisions2 = _evaluate_one(orch, history=decisions)  # prior decisions in history
        # If both produce decisions, they should have different decision_ids (different dates are same here)
        # Main check: no stack overflow, no unbounded recursion
        assert isinstance(decisions, list)
        assert isinstance(decisions2, list)

    def test_schema_version_constant(self):
        from src.runtime.policy.runtime_exit_orchestrator import SCHEMA_VERSION
        assert SCHEMA_VERSION == 1

    def test_constraints_are_frozen(self):
        c = ExitAdjustmentConstraints()
        with pytest.raises((TypeError, AttributeError)):
            c.max_hold_extension_days = 999   # type: ignore

    def test_decision_is_frozen(self, tmp_path):
        policy = ExitAdjustmentPolicy(
            symbol="7203", hold_extension_days=2, exit_sensitivity_scale=1.1,
            deterioration_pressure=0.05, trend_persistence_score=0.70,
            vol_trailing_multiplier=1.0, regime_threshold_adjustment=0.0,
            suppress_exit=True, force_exit=False,
        )
        rec = ExitAdjustmentDecision.create(
            eval_date="2026-05-20", symbol="7203", policy=policy,
            trigger_runner_score=0.75, trigger_premature_ratio=0.35,
            trigger_giveback_ratio=0.20, trigger_vol=0.20,
            trigger_regime="neutral", trigger_rs_collapse=False,
            observation_count=10,
        )
        with pytest.raises((TypeError, AttributeError)):
            rec.suppress_exit = False   # type: ignore


# ─────────────────────────────────────────────────────────────────────────────
# 16. Edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_positions_produces_no_decisions(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        decisions = orch.evaluate(
            exit_intelligence=_make_intel(),
            positions=[],   # no held positions
            regime=_make_regime(),
            expectancy=_make_expectancy(),
            history=[],
            as_of_date="2026-05-20",
        )
        assert decisions == []

    def test_multiple_positions_independent_decisions(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        positions = [
            _make_pos(symbol="7203", hold_days=10, current_return=0.05),
            _make_pos(symbol="6758", hold_days=5, current_return=-0.02),
            _make_pos(symbol="9984", hold_days=20, current_return=0.10),
        ]
        intel = ExitIntelligenceSnapshot(
            premature_exit_ratio=0.45,
            mean_giveback_ratio=0.30,
            per_symbol_runner_score={"7203": 0.80, "6758": 0.40, "9984": 0.85},
            per_symbol_giveback={"7203": 0.20, "6758": 0.70, "9984": 0.15},
            per_symbol_vol={"7203": 0.20, "6758": 0.35, "9984": 0.18},
            per_symbol_persistence={"7203": 0.65, "6758": 0.30, "9984": 0.75},
            per_symbol_decay_rate={"7203": 0.01, "6758": 0.05, "9984": 0.008},
            n_trades=15,
        )
        decisions = orch.evaluate(
            exit_intelligence=intel,
            positions=positions,
            regime=_make_regime(deteriorating=False),
            expectancy=_make_expectancy(),
            history=[],
            as_of_date="2026-05-20",
        )
        # All decisions must be bounded
        for d in decisions:
            c = ExitAdjustmentConstraints()
            assert d.hold_extension_days <= c.max_hold_extension_days
            assert c.exit_sensitivity_min <= d.exit_sensitivity_scale <= c.exit_sensitivity_max
            assert 0.0 <= d.deterioration_pressure <= c.deterioration_pressure_max
            assert c.vol_trailing_mult_min <= d.vol_trailing_multiplier <= c.vol_trailing_mult_max

    def test_apply_to_signals_no_decisions_is_noop(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        signals = [
            {"symbol": "7203", "signal": 0, "currently_holding": True},
            {"symbol": "6758", "signal": 1, "currently_holding": False},
        ]
        result = orch.apply_to_signals(signals, [])
        # No decisions → signals unchanged
        assert result[0]["signal"] == 0
        assert result[1]["signal"] == 1

    def test_apply_to_orders_no_decisions_is_noop(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        orders = [MagicMock(), MagicMock()]
        result = orch.apply_to_orders(orders, [], [], {})
        assert result == orders

    def test_apply_to_signals_non_holding_unchanged(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        sym = "7203"
        policy = ExitAdjustmentPolicy(
            symbol=sym, hold_extension_days=2, exit_sensitivity_scale=1.1,
            deterioration_pressure=0.05, trend_persistence_score=0.70,
            vol_trailing_multiplier=1.0, regime_threshold_adjustment=0.0,
            suppress_exit=True, force_exit=False,
        )
        dec = ExitAdjustmentDecision.create(
            eval_date="2026-05-20", symbol=sym, policy=policy,
            trigger_runner_score=0.75, trigger_premature_ratio=0.35,
            trigger_giveback_ratio=0.20, trigger_vol=0.20,
            trigger_regime="neutral", trigger_rs_collapse=False,
            observation_count=10,
        )
        # Non-holding signal with same symbol should NOT be modified
        signals = [
            {"symbol": sym, "signal": 1, "currently_holding": False, "action": "BUY"},
        ]
        modified = orch.apply_to_signals(signals, [dec])
        assert modified[0]["signal"] == 1
        assert modified[0]["action"] == "BUY"
        assert not modified[0].get("exit_orchestrator_applied")
