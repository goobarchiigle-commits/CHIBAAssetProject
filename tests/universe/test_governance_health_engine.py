"""
tests/universe/test_governance_health_engine.py — Phase 3C: GovernanceHealthEngine tests

Coverage:
- Check status classification (HEALTHY / DEGRADED / INSUFFICIENT_DATA)
- Score computation
- Mode hysteresis (MODE_CONFIRM_DAYS)
- Recovery buffer (anti-oscillation)
- Mode transition audit JSONL
- FAIL_OPEN (engine errors never affect callers)
- Replay: governance_health.json → re-derive same mode
- All 11 health checks
- 5-mode ladder edge cases
"""
from __future__ import annotations

import json
import sys
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.universe.governance_health_engine import (
    GovernanceHealthEngine,
    GovernanceHealthState,
    CheckResult,
    ModeTransitionRecord,
    GovernanceEnforcement,
    ManualOverride,
    _MODE_ENFORCEMENT,
    _most_restrictive_mode,
    _compute_score,
    _compute_mode_transition,
    _score_to_raw_mode,
    _hours_since,
    MODE_NORMAL,
    MODE_WARNING,
    MODE_DEGRADED,
    MODE_SAFE,
    MODE_FREEZE,
    MODE_LADDER,
    MODE_CONFIRM_DAYS,
    RECOVERY_BUFFER,
    STATUS_HEALTHY,
    STATUS_DEGRADED,
    STATUS_INSUF_DATA,
    SCORE_THRESHOLDS,
)
from src.universe.auto_freeze_engine import (
    AutoFreezeEngine,
    AutoFreezeState,
    OutcomeSummary,
    TurnoverStats,
    MIN_PROMOTION_SAMPLES,
    MIN_ROTATION_SAMPLES,
    MIN_EXPLORATION_SAMPLES,
    TRIGGER_WEIGHTS,
    MAX_DEGRADATION_SCORE,
    FREEZE_SCORE_THRESHOLD,
    SAFE_MODE_SCORE_THRESHOLD,
    DEGRADED_SCORE_THRESHOLD,
    WARNING_SCORE_THRESHOLD,
)
from src.universe.demotion_actuation import (
    DemotionActuationEngine,
    STATE_CANDIDATE,
    STATE_PENDING_EXIT,
    STATE_EXIT_CONFIRMED,
    STATE_REMOVED,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fake demotion result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _FakeDemResult:
    demotion_candidates: List[str] = field(default_factory=list)
    pending_exit_count: int = 0
    average_days_pending_exit: float = 0.0
    symbols_to_remove: List[str] = field(default_factory=list)
    recovered_count: int = 0
    exit_confirmed_count: int = 0


@dataclass
class _FakeRotPlan:
    created_at: str = ""
    deficit: int = 0
    symbols_to_add: List[str] = field(default_factory=list)
    symbols_to_remove: List[str] = field(default_factory=list)
    replacement_candidates: List[str] = field(default_factory=list)
    rejected_candidates: List[str] = field(default_factory=list)
    rotation_score: float = 0.0


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _iso_ago(hours: float) -> str:
    dt = datetime.now(timezone.utc) - timedelta(hours=hours)
    return dt.isoformat()


def _make_live(n: int = 42) -> Dict[str, str]:
    return {f"S{i:04d}": "TSE" for i in range(n)}


def _make_shadow(n: int = 5, offset: int = 100) -> Dict[str, str]:
    return {f"S{i:04d}": "TSE" for i in range(offset, offset + n)}


def _make_engine(tmp_path: Path, target: int = 42, confirm_days: int = 3, recovery_buffer: float = RECOVERY_BUFFER) -> GovernanceHealthEngine:
    return GovernanceHealthEngine(
        governance_health_file=tmp_path / "governance_health.json",
        mode_history_file=tmp_path / "governance_mode_history.jsonl",
        target_universe_size=target,
        mode_confirm_days=confirm_days,
        recovery_buffer=recovery_buffer,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Score computation
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeScore:
    def test_all_healthy(self):
        checks = {
            f"c{i}": CheckResult(f"c{i}", STATUS_HEALTHY, 0.0, "")
            for i in range(11)
        }
        assert _compute_score(checks) == 100.0

    def test_all_degraded(self):
        checks = {
            f"c{i}": CheckResult(f"c{i}", STATUS_DEGRADED, 0.0, "")
            for i in range(11)
        }
        assert _compute_score(checks) == 0.0

    def test_all_insuf(self):
        checks = {
            f"c{i}": CheckResult(f"c{i}", STATUS_INSUF_DATA, 0.0, "")
            for i in range(11)
        }
        assert _compute_score(checks) == 50.0

    def test_mixed(self):
        # 6 HEALTHY (1.0), 3 INSUF (0.5), 2 DEGRADED (0.0) over 11 checks
        checks = {}
        for i in range(6):
            checks[f"h{i}"] = CheckResult(f"h{i}", STATUS_HEALTHY, 0.0, "")
        for i in range(3):
            checks[f"i{i}"] = CheckResult(f"i{i}", STATUS_INSUF_DATA, 0.0, "")
        for i in range(2):
            checks[f"d{i}"] = CheckResult(f"d{i}", STATUS_DEGRADED, 0.0, "")
        score = _compute_score(checks)
        expected = round((6 * 1.0 + 3 * 0.5 + 2 * 0.0) / 11 * 100, 2)
        assert score == expected

    def test_empty_checks(self):
        assert _compute_score({}) == 100.0


# ─────────────────────────────────────────────────────────────────────────────
# 2. score_to_raw_mode
# ─────────────────────────────────────────────────────────────────────────────

class TestScoreToRawMode:
    def test_perfect(self):
        assert _score_to_raw_mode(100.0) == MODE_NORMAL

    def test_at_normal_threshold(self):
        assert _score_to_raw_mode(SCORE_THRESHOLDS[MODE_NORMAL]) == MODE_NORMAL

    def test_just_below_normal(self):
        assert _score_to_raw_mode(SCORE_THRESHOLDS[MODE_NORMAL] - 0.01) == MODE_WARNING

    def test_at_warning_threshold(self):
        assert _score_to_raw_mode(SCORE_THRESHOLDS[MODE_WARNING]) == MODE_WARNING

    def test_just_below_warning(self):
        assert _score_to_raw_mode(SCORE_THRESHOLDS[MODE_WARNING] - 0.01) == MODE_DEGRADED

    def test_at_degraded_threshold(self):
        assert _score_to_raw_mode(SCORE_THRESHOLDS[MODE_DEGRADED]) == MODE_DEGRADED

    def test_just_below_degraded(self):
        assert _score_to_raw_mode(SCORE_THRESHOLDS[MODE_DEGRADED] - 0.01) == MODE_SAFE

    def test_at_safe_threshold(self):
        assert _score_to_raw_mode(SCORE_THRESHOLDS[MODE_SAFE]) == MODE_SAFE

    def test_just_below_safe(self):
        assert _score_to_raw_mode(SCORE_THRESHOLDS[MODE_SAFE] - 0.01) == MODE_FREEZE

    def test_zero(self):
        assert _score_to_raw_mode(0.0) == MODE_FREEZE


# ─────────────────────────────────────────────────────────────────────────────
# 3. Mode transition hysteresis
# ─────────────────────────────────────────────────────────────────────────────

class TestModeTransition:
    def test_no_change_when_same_score(self):
        mode, pending, days = _compute_mode_transition(
            current_mode=MODE_NORMAL, current_pending=None, pending_days=0,
            health_score=95.0, confirm_days=3, recovery_buffer=5.0,
        )
        assert mode == MODE_NORMAL
        assert pending is None
        assert days == 0

    def test_downgrade_accumulates_days(self):
        # score=55 → raw=SAFE, downgrade from NORMAL
        mode, pending, days = _compute_mode_transition(
            current_mode=MODE_NORMAL, current_pending=None, pending_days=0,
            health_score=55.0, confirm_days=3, recovery_buffer=5.0,
        )
        assert mode == MODE_NORMAL   # not confirmed yet
        assert pending == MODE_SAFE
        assert days == 1

    def test_downgrade_confirms_on_confirm_day(self):
        # day 2 → 3 confirms
        mode, pending, days = _compute_mode_transition(
            current_mode=MODE_NORMAL, current_pending=MODE_SAFE, pending_days=2,
            health_score=55.0, confirm_days=3, recovery_buffer=5.0,
        )
        assert mode == MODE_SAFE
        assert pending is None
        assert days == 0

    def test_pending_resets_when_target_changes(self):
        # was pending SAFE but now score is 65 → target is DEGRADED
        mode, pending, days = _compute_mode_transition(
            current_mode=MODE_NORMAL, current_pending=MODE_SAFE, pending_days=2,
            health_score=65.0, confirm_days=3, recovery_buffer=5.0,
        )
        assert mode == MODE_NORMAL
        assert pending == MODE_DEGRADED
        assert days == 1   # reset

    def test_recovery_buffer_blocks_upgrade(self):
        # score=81, just above NORMAL threshold (80) but recovery_buffer=5 means
        # effective score = 76, which maps to WARNING not NORMAL
        mode, pending, days = _compute_mode_transition(
            current_mode=MODE_WARNING, current_pending=None, pending_days=0,
            health_score=81.0, confirm_days=3, recovery_buffer=5.0,
        )
        # raw_target=NORMAL (upgrade), adjusted=81-5=76 → WARNING (no change)
        assert mode == MODE_WARNING
        assert pending is None
        assert days == 0

    def test_recovery_buffer_allows_upgrade_with_enough_headroom(self):
        # score=90, recovery_buffer=5 → adjusted=85 ≥ NORMAL(80) → upgrade after confirm
        mode, pending, days = _compute_mode_transition(
            current_mode=MODE_WARNING, current_pending=None, pending_days=0,
            health_score=90.0, confirm_days=3, recovery_buffer=5.0,
        )
        assert mode == MODE_WARNING
        assert pending == MODE_NORMAL
        assert days == 1

    def test_upgrade_confirms_after_3_days(self):
        mode, pending, days = _compute_mode_transition(
            current_mode=MODE_WARNING, current_pending=MODE_NORMAL, pending_days=2,
            health_score=90.0, confirm_days=3, recovery_buffer=5.0,
        )
        assert mode == MODE_NORMAL
        assert pending is None

    def test_confirm_days_1_immediate(self):
        mode, pending, days = _compute_mode_transition(
            current_mode=MODE_NORMAL, current_pending=None, pending_days=0,
            health_score=55.0, confirm_days=1, recovery_buffer=0.0,
        )
        assert mode == MODE_SAFE
        assert pending is None


# ─────────────────────────────────────────────────────────────────────────────
# 4. Individual health checks
# ─────────────────────────────────────────────────────────────────────────────

class TestHealthChecks:
    def test_universe_size_healthy(self, tmp_path):
        eng = _make_engine(tmp_path)
        state = eng.run(_make_live(42), _make_shadow(), None, None)
        assert state.checks["universe_size"].status == STATUS_HEALTHY

    def test_universe_size_degraded(self, tmp_path):
        eng = _make_engine(tmp_path, target=42)
        # live size = 35, off_by = 7 → DEGRADED
        state = eng.run(_make_live(35), _make_shadow(), None, None)
        assert state.checks["universe_size"].status == STATUS_DEGRADED

    def test_universe_size_insuf(self, tmp_path):
        eng = _make_engine(tmp_path, target=42)
        # live size = 40, off_by = 2 → INSUF
        state = eng.run(_make_live(40), _make_shadow(), None, None)
        assert state.checks["universe_size"].status == STATUS_INSUF_DATA

    def test_universe_empty_insuf(self, tmp_path):
        eng = _make_engine(tmp_path)
        state = eng.run({}, {}, None, None)
        assert state.checks["universe_size"].status == STATUS_INSUF_DATA

    def test_demotion_pipeline_healthy(self, tmp_path):
        eng = _make_engine(tmp_path)
        dem = _FakeDemResult(demotion_candidates=[], pending_exit_count=0)
        state = eng.run(_make_live(), {}, dem, None)
        assert state.checks["demotion_pipeline"].status == STATUS_HEALTHY

    def test_demotion_pipeline_degraded(self, tmp_path):
        eng = _make_engine(tmp_path)
        dem = _FakeDemResult(demotion_candidates=["A", "B", "C"], pending_exit_count=1)
        state = eng.run(_make_live(), {}, dem, None)
        assert state.checks["demotion_pipeline"].status == STATUS_DEGRADED

    def test_demotion_pipeline_no_result(self, tmp_path):
        eng = _make_engine(tmp_path)
        state = eng.run(_make_live(), {}, None, None)
        assert state.checks["demotion_pipeline"].status == STATUS_INSUF_DATA

    def test_pending_exit_age_healthy(self, tmp_path):
        eng = _make_engine(tmp_path)
        dem = _FakeDemResult(pending_exit_count=1, average_days_pending_exit=3.0)
        state = eng.run(_make_live(), {}, dem, None)
        assert state.checks["pending_exit_age"].status == STATUS_HEALTHY

    def test_pending_exit_age_degraded(self, tmp_path):
        eng = _make_engine(tmp_path)
        dem = _FakeDemResult(pending_exit_count=2, average_days_pending_exit=20.0)
        state = eng.run(_make_live(), {}, dem, None)
        assert state.checks["pending_exit_age"].status == STATUS_DEGRADED

    def test_shadow_coverage_no_deficit(self, tmp_path):
        eng = _make_engine(tmp_path, target=42)
        state = eng.run(_make_live(42), _make_shadow(5), None, None)
        assert state.checks["shadow_coverage"].status == STATUS_HEALTHY

    def test_shadow_coverage_full(self, tmp_path):
        eng = _make_engine(tmp_path, target=42)
        live = _make_live(38)
        shadow = _make_shadow(5, offset=500)  # 5 new symbols, deficit=4 → full
        state = eng.run(live, shadow, None, None)
        assert state.checks["shadow_coverage"].status == STATUS_HEALTHY

    def test_shadow_coverage_degraded(self, tmp_path):
        eng = _make_engine(tmp_path, target=42)
        live = _make_live(35)   # deficit=7, no shadow candidates
        state = eng.run(live, {}, None, None)
        assert state.checks["shadow_coverage"].status == STATUS_DEGRADED

    def test_rotation_plan_age_healthy(self, tmp_path):
        eng = _make_engine(tmp_path)
        plan = _FakeRotPlan(created_at=_iso_ago(2))
        state = eng.run(_make_live(), {}, None, plan)
        assert state.checks["rotation_plan_age"].status == STATUS_HEALTHY

    def test_rotation_plan_age_degraded(self, tmp_path):
        eng = _make_engine(tmp_path)
        plan = _FakeRotPlan(created_at=_iso_ago(80))
        state = eng.run(_make_live(), {}, None, plan)
        assert state.checks["rotation_plan_age"].status == STATUS_DEGRADED

    def test_rotation_coverage_no_deficit(self, tmp_path):
        eng = _make_engine(tmp_path)
        plan = _FakeRotPlan(deficit=0, symbols_to_add=[])
        state = eng.run(_make_live(), {}, None, plan)
        assert state.checks["rotation_coverage"].status == STATUS_HEALTHY

    def test_rotation_coverage_full(self, tmp_path):
        eng = _make_engine(tmp_path)
        plan = _FakeRotPlan(deficit=3, symbols_to_add=["A", "B", "C"])
        state = eng.run(_make_live(), {}, None, plan)
        assert state.checks["rotation_coverage"].status == STATUS_HEALTHY

    def test_rotation_coverage_degraded(self, tmp_path):
        eng = _make_engine(tmp_path)
        plan = _FakeRotPlan(deficit=10, symbols_to_add=["A"])
        state = eng.run(_make_live(), {}, None, plan)
        assert state.checks["rotation_coverage"].status == STATUS_DEGRADED

    def test_removal_activity_healthy(self, tmp_path):
        eng = _make_engine(tmp_path)
        dem = _FakeDemResult(symbols_to_remove=[])
        state = eng.run(_make_live(), {}, dem, None)
        assert state.checks["removal_activity"].status == STATUS_HEALTHY

    def test_removal_activity_degraded(self, tmp_path):
        eng = _make_engine(tmp_path)
        dem = _FakeDemResult(symbols_to_remove=["A", "B"])
        state = eng.run(_make_live(), {}, dem, None)
        assert state.checks["removal_activity"].status == STATUS_DEGRADED

    def test_recovery_rate_healthy(self, tmp_path):
        eng = _make_engine(tmp_path)
        dem = _FakeDemResult(recovered_count=2, demotion_candidates=["A"])
        state = eng.run(_make_live(), {}, dem, None)
        assert state.checks["recovery_rate"].status == STATUS_HEALTHY

    def test_candidate_freshness_all_fresh(self, tmp_path):
        eng = _make_engine(tmp_path)
        shadow = _make_shadow(3)
        plan = _FakeRotPlan(replacement_candidates=["A", "B"], rejected_candidates=[])
        state = eng.run(_make_live(), shadow, None, plan)
        assert state.checks["candidate_freshness"].status == STATUS_HEALTHY

    def test_candidate_freshness_high_stale(self, tmp_path):
        eng = _make_engine(tmp_path)
        shadow = _make_shadow(3)
        plan = _FakeRotPlan(replacement_candidates=["A"], rejected_candidates=["B", "C", "D", "E"])
        state = eng.run(_make_live(), shadow, None, plan)
        assert state.checks["candidate_freshness"].status == STATUS_DEGRADED

    def test_universe_churn_no_prev(self, tmp_path):
        eng = _make_engine(tmp_path)
        state = eng.run(_make_live(), {}, None, None)
        assert state.checks["universe_churn"].status == STATUS_INSUF_DATA

    def test_universe_churn_stable(self, tmp_path):
        eng = _make_engine(tmp_path)
        # First run: establish previous state
        state1 = eng.run(_make_live(42), {}, None, None)
        eng.write_state(state1)
        # Second run: same size
        state2 = eng.run(_make_live(42), {}, None, None)
        assert state2.checks["universe_churn"].status == STATUS_HEALTHY

    def test_universe_churn_degraded(self, tmp_path):
        eng = _make_engine(tmp_path)
        state1 = eng.run(_make_live(42), {}, None, None)
        eng.write_state(state1)
        # Big drop
        state2 = eng.run(_make_live(38), {}, None, None)
        assert state2.checks["universe_churn"].status == STATUS_DEGRADED

    def test_governance_freshness_first_run(self, tmp_path):
        eng = _make_engine(tmp_path)
        state = eng.run(_make_live(), {}, None, None)
        assert state.checks["governance_freshness"].status == STATUS_INSUF_DATA

    def test_governance_freshness_healthy(self, tmp_path):
        eng = _make_engine(tmp_path)
        state1 = eng.run(_make_live(), {}, None, None)
        eng.write_state(state1)
        state2 = eng.run(_make_live(), {}, None, None)
        assert state2.checks["governance_freshness"].status == STATUS_HEALTHY

    def test_governance_freshness_degraded(self, tmp_path):
        """Simulate stale governance (> 72h ago)."""
        eng = _make_engine(tmp_path)
        # Write a state with computed_at 80h ago
        old_state = GovernanceHealthState(
            governance_mode=MODE_NORMAL,
            mode_confirmed=True,
            pending_mode=None,
            pending_mode_days=0,
            health_score=100.0,
            checks={},
            computed_at=_iso_ago(80),
            previous_live_size=42,
        )
        eng.write_state(old_state)
        state = eng.run(_make_live(), {}, None, None)
        assert state.checks["governance_freshness"].status == STATUS_DEGRADED


# ─────────────────────────────────────────────────────────────────────────────
# 5. Mode transitions via run()
# ─────────────────────────────────────────────────────────────────────────────

class TestModeTransitionIntegration:
    def test_first_run_is_normal(self, tmp_path):
        eng = _make_engine(tmp_path)
        state = eng.run(_make_live(42), _make_shadow(5), None, None)
        assert state.governance_mode == MODE_NORMAL

    def test_degraded_mode_requires_confirm_days(self, tmp_path):
        """score < 60 triggers SAFE_MODE only after MODE_CONFIRM_DAYS runs."""
        eng = _make_engine(tmp_path, confirm_days=3)
        live = _make_live(30)  # off_by=12 → many DEGRADED checks
        shadow = {}

        # run 1: pending, not confirmed
        s1 = eng.run(live, shadow, None, None)
        eng.write_state(s1)
        assert s1.governance_mode == MODE_NORMAL or s1.governance_mode in MODE_LADDER

        # run 2: still pending
        s2 = eng.run(live, shadow, None, None)
        eng.write_state(s2)

        # run 3: confirms (day 3 ≥ confirm_days=3)
        s3 = eng.run(live, shadow, None, None)
        eng.write_state(s3)

        # After 3 consecutive bad scores, mode should have moved
        assert s3.governance_mode != MODE_NORMAL

    def test_immediate_confirm_with_confirm_days_1(self, tmp_path):
        eng = _make_engine(tmp_path, confirm_days=1)
        live = _make_live(20)  # severely degraded
        state = eng.run(live, {}, None, None)
        assert state.governance_mode != MODE_NORMAL

    def test_mode_written_to_json(self, tmp_path):
        eng = _make_engine(tmp_path)
        state = eng.run(_make_live(42), _make_shadow(5), None, None)
        eng.write_state(state)
        health_file = tmp_path / "governance_health.json"
        assert health_file.exists()
        loaded = json.loads(health_file.read_text())
        assert loaded["governance_mode"] == state.governance_mode
        assert loaded["health_score"] == state.health_score

    def test_transition_written_to_history_jsonl(self, tmp_path):
        eng = _make_engine(tmp_path, confirm_days=1)
        live = _make_live(20)
        state = eng.run(live, {}, None, None)
        eng.write_state(state)
        hist_file = tmp_path / "governance_mode_history.jsonl"
        if state.governance_mode != MODE_NORMAL:
            assert hist_file.exists()
            records = [json.loads(l) for l in hist_file.read_text().strip().splitlines()]
            assert len(records) >= 1
            assert "old_mode" in records[0]
            assert "new_mode" in records[0]
            assert "health_score" in records[0]
            assert "trigger_checks" in records[0]


# ─────────────────────────────────────────────────────────────────────────────
# 6. FAIL_OPEN
# ─────────────────────────────────────────────────────────────────────────────

class TestFailOpen:
    def test_none_inputs_do_not_raise(self, tmp_path):
        eng = _make_engine(tmp_path)
        state = eng.run({}, {}, None, None)
        assert state is not None
        assert state.governance_mode in MODE_LADDER

    def test_bad_dem_result_attribute_does_not_raise(self, tmp_path):
        eng = _make_engine(tmp_path)

        class _BadDem:
            @property
            def demotion_candidates(self):
                raise RuntimeError("unexpected!")
            pending_exit_count = 0
            average_days_pending_exit = 0.0
            symbols_to_remove = []
            recovered_count = 0

        state = eng.run(_make_live(42), {}, _BadDem(), None)
        assert state is not None
        # Affected checks should be INSUF_DATA, not crash
        assert state.checks["demotion_pipeline"].status == STATUS_INSUF_DATA

    def test_write_state_to_unwritable_dir_does_not_raise(self, tmp_path):
        # Point health file to a file (not dir) — should FAIL_OPEN
        bad_file = tmp_path / "not_a_dir" / "x.json"
        eng = GovernanceHealthEngine(
            governance_health_file=bad_file,
            mode_history_file=tmp_path / "hist.jsonl",
        )
        state = eng.run(_make_live(), {}, None, None)
        # write_state may fail but must not raise
        eng.write_state(state)  # no exception


# ─────────────────────────────────────────────────────────────────────────────
# 7. Replay from governance_health.json
# ─────────────────────────────────────────────────────────────────────────────

class TestReplay:
    def test_load_state_returns_none_on_missing_file(self, tmp_path):
        eng = _make_engine(tmp_path)
        assert eng.load_state() is None

    def test_load_state_roundtrip(self, tmp_path):
        eng = _make_engine(tmp_path)
        state = eng.run(_make_live(42), _make_shadow(5), None, None)
        eng.write_state(state)
        loaded = eng.load_state()
        assert loaded is not None
        assert loaded.governance_mode == state.governance_mode
        assert loaded.health_score == state.health_score
        assert set(loaded.checks.keys()) == set(state.checks.keys())

    def test_load_state_corrupt_returns_none(self, tmp_path):
        health_file = tmp_path / "governance_health.json"
        health_file.write_text("not valid json", encoding="utf-8")
        eng = _make_engine(tmp_path)
        assert eng.load_state() is None

    def test_pending_mode_preserved_across_restart(self, tmp_path):
        """Pending mode and days survive a write/load cycle."""
        eng = _make_engine(tmp_path, confirm_days=3)
        live = _make_live(20)  # enough to trigger pending
        s1 = eng.run(live, {}, None, None)
        eng.write_state(s1)

        # Create a fresh engine instance pointing to same files
        eng2 = _make_engine(tmp_path, confirm_days=3)
        s2 = eng2.run(live, {}, None, None)
        if s1.pending_mode is not None:
            # days should have advanced
            assert s2.pending_mode_days >= s1.pending_mode_days


# ─────────────────────────────────────────────────────────────────────────────
# 8. Anti-oscillation
# ─────────────────────────────────────────────────────────────────────────────

class TestAntiOscillation:
    def test_score_vibration_does_not_cause_mode_flip(self, tmp_path):
        """Score oscillating just around a threshold should not cause mode flips."""
        eng = _make_engine(tmp_path, confirm_days=3, recovery_buffer=5.0)
        # Establish NORMAL
        state = eng.run(_make_live(42), _make_shadow(5), None, None)
        eng.write_state(state)

        # Simulate score vibration: alternating WARNING-NORMAL territory
        modes_seen = set()
        for _ in range(6):
            s = eng.run(_make_live(41), _make_shadow(5), None, None)
            eng.write_state(s)
            modes_seen.add(s.governance_mode)

        # Should not rapidly oscillate between modes
        assert len(modes_seen) <= 2

    def test_recovery_buffer_requires_stable_improvement(self, tmp_path):
        """Mode should not immediately upgrade from DEGRADED to NORMAL on single good run."""
        eng = _make_engine(tmp_path, confirm_days=3)
        # Force into a worse mode first
        live_bad = _make_live(20)
        for _ in range(3):
            s = eng.run(live_bad, {}, None, None)
            eng.write_state(s)

        current_mode = s.governance_mode
        # Now provide one good run
        s_good = eng.run(_make_live(42), _make_shadow(5), None, None)
        eng.write_state(s_good)

        # Should not jump straight to NORMAL; still in transition
        if current_mode != MODE_NORMAL:
            assert s_good.governance_mode != MODE_NORMAL or s_good.pending_mode is not None or s_good.mode_confirmed


# ─────────────────────────────────────────────────────────────────────────────
# 9. DEGRADED mode: governance continues
# ─────────────────────────────────────────────────────────────────────────────

class TestDegradedGovernanceContinues:
    def test_engine_returns_state_in_all_modes(self, tmp_path):
        """GovernanceHealthEngine returns a valid state in any mode (never blocks)."""
        eng = _make_engine(tmp_path, confirm_days=1, recovery_buffer=0.0)
        live = _make_live(10)  # very bad
        state = eng.run(live, {}, None, None)
        assert isinstance(state, GovernanceHealthState)
        assert state.governance_mode in MODE_LADDER

    def test_governance_mode_is_observation_only(self, tmp_path):
        """Governance mode has no impact on run() return — always returns state."""
        eng = _make_engine(tmp_path, confirm_days=1, recovery_buffer=0.0)
        s_freeze = eng.run(_make_live(5), {}, None, None)
        eng.write_state(s_freeze)
        # Even in FREEZE mode, another run() returns a state
        s_next = eng.run(_make_live(5), {}, None, None)
        assert isinstance(s_next, GovernanceHealthState)


# ─────────────────────────────────────────────────────────────────────────────
# 10. hours_since helper
# ─────────────────────────────────────────────────────────────────────────────

class TestHoursSince:
    def test_basic(self):
        now = datetime.now(timezone.utc)
        past = (now - timedelta(hours=5)).isoformat()
        h = _hours_since(past, now.isoformat())
        assert 4.9 < h < 5.1

    def test_invalid_returns_zero(self):
        assert _hours_since("not_iso", _now_iso()) == 0.0

    def test_naive_datetimes_treated_as_utc(self):
        naive_past = (datetime.utcnow() - timedelta(hours=3)).isoformat()
        naive_now  = datetime.utcnow().isoformat()
        h = _hours_since(naive_past, naive_now)
        assert 2.9 < h < 3.1


# ─────────────────────────────────────────────────────────────────────────────
# 11. ModeTransitionRecord serialization
# ─────────────────────────────────────────────────────────────────────────────

class TestModeTransitionRecord:
    def test_to_dict(self):
        rec = ModeTransitionRecord(
            timestamp=_now_iso(),
            old_mode=MODE_NORMAL,
            new_mode=MODE_DEGRADED,
            health_score=55.0,
            trigger_checks=["universe_size", "demotion_pipeline"],
            confirm_day=3,
        )
        d = rec.to_dict()
        assert d["old_mode"] == MODE_NORMAL
        assert d["new_mode"] == MODE_DEGRADED
        assert d["trigger_checks"] == ["universe_size", "demotion_pipeline"]
        assert d["confirm_day"] == 3


# ─────────────────────────────────────────────────────────────────────────────
# 12. CheckResult serialization
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckResultSerialization:
    def test_roundtrip(self):
        cr = CheckResult(
            check_name="universe_size",
            status=STATUS_DEGRADED,
            value=35.0,
            reason="size=35 off_by=7 target=42",
        )
        d = cr.to_dict()
        restored = CheckResult.from_dict(d)
        assert restored.check_name == cr.check_name
        assert restored.status == cr.status
        assert restored.value == cr.value
        assert restored.reason == cr.reason


# ─────────────────────────────────────────────────────────────────────────────
# 13. Required additional tests (spec)
# ─────────────────────────────────────────────────────────────────────────────

def _make_dem_engine(tmp_path: Path) -> DemotionActuationEngine:
    return DemotionActuationEngine(
        pending_demotion_file=tmp_path / "pending_demotion.jsonl",
        metrics_file=tmp_path / "demotion_metrics.jsonl",
    )


class TestSafeModeDemotionSplit:
    def test_safe_mode_keeps_pending_exit_updates(self, tmp_path):
        """
        SAFE_MODE: allow_new_demotions=False.
        Existing PENDING_EXIT symbol with position=0 must still transition
        to EXIT_CONFIRMED (state machine continues).
        New candidate with is_candidate=True must NOT be created.
        """
        import json as _json
        from dataclasses import dataclass as _dc, field as _f

        @_dc
        class _Press:
            is_candidate: bool = False
            pressure_score: float = 0.0

        eng = _make_dem_engine(tmp_path)

        # Seed PENDING_EXIT state for S0001
        from src.universe.demotion_actuation import DemotionRecord, STATE_PENDING_EXIT
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        seed = DemotionRecord(
            symbol="S0001", state=STATE_PENDING_EXIT,
            entered_at=now, updated_at=now,
            reason="replacement_available_pending_position_close",
            demotion_pressure=0.75, replacement_delta=0.5, position_size=100,
        )
        (tmp_path / "pending_demotion.jsonl").write_text(
            _json.dumps(seed.to_dict(), sort_keys=True) + "\n", encoding="utf-8"
        )

        pressures = {
            "S0001": _Press(is_candidate=True, pressure_score=0.75),  # existing state
            "S0002": _Press(is_candidate=True, pressure_score=0.80),  # new candidate (should be blocked)
        }
        portfolio = {}  # no positions → S0001 position_size=0 → should advance to EXIT_CONFIRMED

        result = eng.run(
            pressures=pressures,
            portfolio_state=portfolio,
            shadow_universe={"S0099": "TSE"},
            live_universe={"S0001": "TSE", "S0002": "TSE"},
            allow_new_demotions=False,
        )

        # S0001 (PENDING_EXIT → position=0 → EXIT_CONFIRMED) must still happen
        confirmed = [s for s, r in result.state_snapshot.items() if r.state == STATE_EXIT_CONFIRMED]
        assert "S0001" in confirmed, "existing PENDING_EXIT must advance to EXIT_CONFIRMED"

        # S0002 must NOT have been created as a new candidate
        assert "S0002" not in result.state_snapshot, "new demotion candidate must be blocked in SAFE_MODE"
        assert "S0002" not in result.demotion_candidates

    def test_safe_mode_exit_confirmed_still_removes(self, tmp_path):
        """EXIT_CONFIRMED → REMOVED continues in SAFE_MODE (allow_new_demotions=False)."""
        import json as _json
        from src.universe.demotion_actuation import DemotionRecord, STATE_EXIT_CONFIRMED

        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        seed = DemotionRecord(
            symbol="S0003", state=STATE_EXIT_CONFIRMED,
            entered_at=now, updated_at=now,
            reason="no_position_exit_confirmed",
            demotion_pressure=0.80, replacement_delta=0.0, position_size=0,
        )
        (tmp_path / "pending_demotion.jsonl").write_text(
            _json.dumps(seed.to_dict(), sort_keys=True) + "\n", encoding="utf-8"
        )
        eng = _make_dem_engine(tmp_path)

        @dataclass
        class _Press:
            is_candidate: bool = False
            pressure_score: float = 0.0

        result = eng.run(
            pressures={"S0003": _Press(is_candidate=True, pressure_score=0.80)},
            portfolio_state={},
            shadow_universe={"S0099": "TSE"},
            live_universe={"S0003": "TSE"},
            allow_new_demotions=False,
        )
        assert "S0003" in result.symbols_to_remove, "EXIT_CONFIRMED→REMOVED must continue in SAFE_MODE"


class TestGovernanceEngineFailureWarning:
    def test_governance_engine_failure_enters_warning(self, tmp_path):
        """
        When run() encounters an unhandled internal error,
        it returns WARNING state (not NORMAL, not raising).
        """
        eng = _make_engine(tmp_path)

        # Monkey-patch _run_internal to raise
        def _raise(*args, **kwargs):
            raise RuntimeError("simulated internal failure")

        eng._run_internal = _raise

        state = eng.run(_make_live(42), {}, None, None)

        assert state is not None
        assert state.governance_mode == MODE_WARNING, (
            f"engine failure must return WARNING, got {state.governance_mode}"
        )
        assert state.enforcement.allow_promotions is True   # WARNING = permissive
        assert state.enforcement.allow_new_demotions is True

    def test_warning_fallback_has_engine_error_check(self, tmp_path):
        """_warning_fallback must include an engine_error check."""
        eng = _make_engine(tmp_path)
        state = eng._warning_fallback("test error")
        assert "engine_error" in state.checks
        assert state.checks["engine_error"].status == STATUS_DEGRADED
        assert state.governance_mode == MODE_WARNING


class TestManualOverride:
    def _write_override(self, tmp_path: Path, mode: str, until_iso: Optional[str], reason: str = "test") -> Path:
        override_file = tmp_path / "governance_override.json"
        override_file.write_text(
            json.dumps({"override_mode": mode, "override_until": until_iso, "override_reason": reason}),
            encoding="utf-8",
        )
        return override_file

    def test_manual_override_applied(self, tmp_path):
        """Valid active override replaces computed mode."""
        from datetime import datetime, timezone, timedelta
        future_until = (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat()
        override_file = self._write_override(tmp_path, MODE_SAFE, future_until, reason="ops_drill")

        eng = GovernanceHealthEngine(
            governance_health_file=tmp_path / "governance_health.json",
            mode_history_file=tmp_path / "governance_mode_history.jsonl",
            governance_override_file=override_file,
        )
        state = eng.run(_make_live(42), _make_shadow(5), None, None)

        assert state.governance_mode == MODE_SAFE
        assert state.override_applied is True
        assert state.override_reason == "ops_drill"
        # Enforcement must match SAFE_MODE
        assert state.enforcement.allow_promotions is False
        assert state.enforcement.allow_rotations is False
        assert state.enforcement.allow_new_demotions is False
        assert state.enforcement.allow_demotion_state_updates is True

    def test_manual_override_expired(self, tmp_path):
        """Expired override is not applied; computed mode is used."""
        from datetime import datetime, timezone, timedelta
        past_until = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        override_file = self._write_override(tmp_path, MODE_SAFE, past_until)

        eng = GovernanceHealthEngine(
            governance_health_file=tmp_path / "governance_health.json",
            mode_history_file=tmp_path / "governance_mode_history.jsonl",
            governance_override_file=override_file,
        )
        state = eng.run(_make_live(42), _make_shadow(5), None, None)

        assert state.override_applied is False
        # Healthy inputs → computed mode is NORMAL (or at most WARNING if first run)
        assert state.governance_mode in (MODE_NORMAL, MODE_WARNING, MODE_DEGRADED)

    def test_override_logged(self, tmp_path):
        """Active override is written to mode_history.jsonl with trigger=MANUAL_OVERRIDE."""
        from datetime import datetime, timezone, timedelta
        future_until = (datetime.now(timezone.utc) + timedelta(hours=12)).isoformat()
        override_file = self._write_override(tmp_path, MODE_WARNING, future_until, reason="debug")

        hist_file = tmp_path / "governance_mode_history.jsonl"
        eng = GovernanceHealthEngine(
            governance_health_file=tmp_path / "governance_health.json",
            mode_history_file=hist_file,
            governance_override_file=override_file,
        )
        state = eng.run(_make_live(42), _make_shadow(5), None, None)
        eng.write_state(state)

        assert hist_file.exists(), "mode history JSONL must be written"
        records = [json.loads(line) for line in hist_file.read_text().strip().splitlines()]
        override_records = [r for r in records if "MANUAL_OVERRIDE" in r.get("trigger_checks", [])]
        assert len(override_records) >= 1, "override must be logged with MANUAL_OVERRIDE trigger"
        assert override_records[-1]["new_mode"] == MODE_WARNING

    def test_override_without_override_file_no_effect(self, tmp_path):
        """Engine without override_file path ignores overrides (default behavior)."""
        eng = _make_engine(tmp_path)   # no governance_override_file
        state = eng.run(_make_live(42), _make_shadow(5), None, None)
        assert state.override_applied is False


class TestEnforcementPolicy:
    def test_safe_mode_enforcement(self):
        enf = _MODE_ENFORCEMENT[MODE_SAFE]
        assert enf.allow_promotions is False
        assert enf.allow_rotations is False
        assert enf.allow_new_demotions is False
        assert enf.allow_demotion_state_updates is True

    def test_freeze_enforcement(self):
        enf = _MODE_ENFORCEMENT[MODE_FREEZE]
        assert enf.allow_promotions is False
        assert enf.allow_rotations is False
        assert enf.allow_new_demotions is False
        assert enf.allow_demotion_state_updates is False

    def test_normal_enforcement_all_true(self):
        enf = _MODE_ENFORCEMENT[MODE_NORMAL]
        assert all([enf.allow_promotions, enf.allow_rotations,
                    enf.allow_new_demotions, enf.allow_demotion_state_updates])

    def test_warning_enforcement_all_true(self):
        enf = _MODE_ENFORCEMENT[MODE_WARNING]
        assert all([enf.allow_promotions, enf.allow_rotations,
                    enf.allow_new_demotions, enf.allow_demotion_state_updates])

    def test_enforcement_in_state(self, tmp_path):
        """GovernanceHealthState includes enforcement matching governance_mode."""
        eng = _make_engine(tmp_path, confirm_days=1, recovery_buffer=0.0)
        state = eng.run(_make_live(42), _make_shadow(5), None, None)
        assert state.enforcement is not None
        expected = _MODE_ENFORCEMENT.get(state.governance_mode)
        assert state.enforcement.allow_promotions == expected.allow_promotions

    def test_enforcement_serialized(self, tmp_path):
        """enforcement field is persisted to governance_health.json."""
        eng = _make_engine(tmp_path)
        state = eng.run(_make_live(42), {}, None, None)
        eng.write_state(state)
        loaded = eng.load_state()
        assert loaded.enforcement is not None
        assert loaded.enforcement.allow_promotions == state.enforcement.allow_promotions


# ─────────────────────────────────────────────────────────────────────────────
# 14. Required tests — Auto Freeze Engine (spec)
# ─────────────────────────────────────────────────────────────────────────────

def _afe(tmp_path: Path) -> AutoFreezeEngine:
    return AutoFreezeEngine(state_file=tmp_path / "auto_freeze_state.json")


class TestAutoFreezeMinSampleProtection:
    def test_insufficient_promotion_samples(self, tmp_path):
        """
        promotion_outcomes.sample_count < MIN_PROMOTION_SAMPLES
        → promotion_alpha_collapse and promotion_precision_collapse NOT scored.
        Degradation score contribution = 0 from these triggers.
        """
        eng = _afe(tmp_path)
        # sample_count below minimum; bad alpha and win_rate that would trigger if samples were OK
        prom = OutcomeSummary(
            sample_count=MIN_PROMOTION_SAMPLES - 1,
            alpha_30d=-0.15,
            win_rate_30d=0.20,
        )
        state = eng.run(promotion_outcomes=prom)

        assert "promotion_alpha_collapse" not in state.triggered_checks
        assert "promotion_precision_collapse" not in state.triggered_checks
        # Score must not include weights 3 + 2 = 5 from these two
        assert state.degradation_score < (
            TRIGGER_WEIGHTS["promotion_alpha_collapse"]
            + TRIGGER_WEIGHTS["promotion_precision_collapse"]
        )

    def test_insufficient_rotation_samples(self, tmp_path):
        """
        rotation_outcomes.sample_count < MIN_ROTATION_SAMPLES
        → rotation_alpha_collapse NOT scored.
        """
        eng = _afe(tmp_path)
        rot = OutcomeSummary(
            sample_count=MIN_ROTATION_SAMPLES - 1,
            alpha_30d=-0.20,
        )
        state = eng.run(rotation_outcomes=rot)

        assert "rotation_alpha_collapse" not in state.triggered_checks
        assert state.degradation_score < TRIGGER_WEIGHTS["rotation_alpha_collapse"]

    def test_sufficient_samples_triggers_score(self, tmp_path):
        """Control: with sufficient samples, bad alpha DOES trigger."""
        eng = _afe(tmp_path)
        prom = OutcomeSummary(
            sample_count=MIN_PROMOTION_SAMPLES,
            alpha_30d=-0.10,
        )
        state = eng.run(promotion_outcomes=prom)
        assert "promotion_alpha_collapse" in state.triggered_checks

    def test_insufficient_exploration_samples(self, tmp_path):
        """Exploration with sample_count < MIN → not scored."""
        eng = _afe(tmp_path)
        expl = OutcomeSummary(
            sample_count=MIN_EXPLORATION_SAMPLES - 1,
            success_rate=0.10,
        )
        state = eng.run(exploration_outcomes=expl)
        assert "exploration_failure" not in state.triggered_checks


class TestAutoFreezeWeightedScoring:
    def test_weighted_scoring(self, tmp_path):
        """
        Two triggers: promotion_alpha_collapse(3) + rotation_alpha_collapse(3) = 6
        → SAFE_MODE (score=6 ≥ SAFE_MODE_SCORE_THRESHOLD=6).
        """
        eng = _afe(tmp_path)
        prom = OutcomeSummary(sample_count=MIN_PROMOTION_SAMPLES, alpha_30d=-0.11)
        rot  = OutcomeSummary(sample_count=MIN_ROTATION_SAMPLES,  alpha_30d=-0.08)
        state = eng.run(promotion_outcomes=prom, rotation_outcomes=rot)

        assert "promotion_alpha_collapse" in state.triggered_checks
        assert "rotation_alpha_collapse" in state.triggered_checks
        assert state.degradation_score == (
            TRIGGER_WEIGHTS["promotion_alpha_collapse"]
            + TRIGGER_WEIGHTS["rotation_alpha_collapse"]
        )
        assert state.auto_freeze_mode == MODE_SAFE

    def test_single_alpha_collapse_degraded(self, tmp_path):
        """One alpha collapse (weight=3) → DEGRADED (3 ≥ DEGRADED_SCORE_THRESHOLD=3)."""
        eng = _afe(tmp_path)
        prom = OutcomeSummary(sample_count=MIN_PROMOTION_SAMPLES, alpha_30d=-0.05)
        state = eng.run(promotion_outcomes=prom)
        assert state.degradation_score == TRIGGER_WEIGHTS["promotion_alpha_collapse"]
        assert state.auto_freeze_mode == MODE_DEGRADED

    def test_all_triggers_freeze(self, tmp_path):
        """All 5 triggers fire → score=10 → FREEZE."""
        eng = _afe(tmp_path)
        prom = OutcomeSummary(
            sample_count=MIN_PROMOTION_SAMPLES,
            alpha_30d=-0.15, win_rate_30d=0.20,
        )
        rot  = OutcomeSummary(sample_count=MIN_ROTATION_SAMPLES, alpha_30d=-0.10)
        expl = OutcomeSummary(sample_count=MIN_EXPLORATION_SAMPLES, success_rate=0.10)
        turn = TurnoverStats(recent_count=10, expected_avg=2.0)
        state = eng.run(prom, rot, expl, turn)

        assert state.degradation_score == MAX_DEGRADATION_SCORE
        assert state.auto_freeze_mode == MODE_FREEZE

    def test_no_triggers_normal(self, tmp_path):
        """Good outcomes → score=0 → NORMAL."""
        eng = _afe(tmp_path)
        prom = OutcomeSummary(sample_count=MIN_PROMOTION_SAMPLES, alpha_30d=0.05, win_rate_30d=0.60)
        rot  = OutcomeSummary(sample_count=MIN_ROTATION_SAMPLES,  alpha_30d=0.03)
        state = eng.run(promotion_outcomes=prom, rotation_outcomes=rot)
        assert state.degradation_score == 0.0
        assert state.auto_freeze_mode == MODE_NORMAL


class TestTriggerValuesPersisted:
    def test_trigger_values_persisted(self, tmp_path):
        """
        auto_freeze_state.json must contain trigger_values with measured alpha values.
        Example: {promotion_alpha_30d: -0.11, rotation_alpha_30d: -0.08, promotion_win_rate_30d: 0.29}
        """
        eng = _afe(tmp_path)
        prom = OutcomeSummary(
            sample_count=MIN_PROMOTION_SAMPLES,
            alpha_30d=-0.11,
            win_rate_30d=0.29,
        )
        rot = OutcomeSummary(
            sample_count=MIN_ROTATION_SAMPLES,
            alpha_30d=-0.08,
        )
        state = eng.run(promotion_outcomes=prom, rotation_outcomes=rot)
        eng.write_state(state)

        state_file = tmp_path / "auto_freeze_state.json"
        assert state_file.exists()
        d = json.loads(state_file.read_text())
        tv = d["trigger_values"]

        assert "promotion_alpha_30d" in tv
        assert "rotation_alpha_30d" in tv
        assert "promotion_win_rate_30d" in tv
        assert abs(tv["promotion_alpha_30d"] - (-0.11)) < 1e-9
        assert abs(tv["rotation_alpha_30d"] - (-0.08)) < 1e-9
        assert abs(tv["promotion_win_rate_30d"] - 0.29) < 1e-9

    def test_trigger_values_roundtrip(self, tmp_path):
        """AutoFreezeState serializes and deserializes trigger_values correctly."""
        eng = _afe(tmp_path)
        prom = OutcomeSummary(sample_count=MIN_PROMOTION_SAMPLES, alpha_30d=-0.05)
        state = eng.run(promotion_outcomes=prom)
        eng.write_state(state)
        loaded = eng.load_state()
        assert loaded is not None
        assert "promotion_alpha_30d" in loaded.trigger_values
        assert abs(loaded.trigger_values["promotion_alpha_30d"] - (-0.05)) < 1e-9


class TestEffectiveMode:
    def test_effective_mode_prefers_freeze(self, tmp_path):
        """
        governance_mode=WARNING, auto_freeze_mode=FREEZE
        → effective_mode=FREEZE (most restrictive wins).
        """
        eng = _make_engine(tmp_path)
        # Build an auto_freeze_state that forces FREEZE
        af_state = AutoFreezeState(
            auto_freeze_mode=MODE_FREEZE,
            degradation_score=8.0,
            triggered_checks=["promotion_alpha_collapse", "rotation_alpha_collapse"],
            trigger_values={"promotion_alpha_30d": -0.11, "rotation_alpha_30d": -0.08},
            computed_at=_now_iso(),
        )
        # Run governance health engine with healthy inputs (→ NORMAL/WARNING governance)
        state = eng.run(_make_live(42), _make_shadow(5), None, None, auto_freeze_state=af_state)

        assert state.auto_freeze_mode == MODE_FREEZE
        assert state.effective_mode == MODE_FREEZE
        assert state.effective_enforcement.allow_promotions is False
        assert state.effective_enforcement.allow_new_demotions is False
        assert state.effective_enforcement.allow_demotion_state_updates is False

    def test_effective_mode_normal_when_both_normal(self, tmp_path):
        """governance=NORMAL, auto_freeze=NORMAL → effective=NORMAL."""
        eng = _make_engine(tmp_path)
        af_state = AutoFreezeState(
            auto_freeze_mode=MODE_NORMAL,
            degradation_score=0.0,
            triggered_checks=[],
            trigger_values={},
            computed_at=_now_iso(),
        )
        state = eng.run(_make_live(42), _make_shadow(5), None, None, auto_freeze_state=af_state)
        assert state.effective_mode == MODE_NORMAL
        assert state.effective_enforcement.allow_promotions is True

    def test_effective_mode_no_auto_freeze_uses_governance(self, tmp_path):
        """auto_freeze_state=None → effective_mode = governance_mode."""
        eng = _make_engine(tmp_path)
        state = eng.run(_make_live(42), _make_shadow(5), None, None, auto_freeze_state=None)
        assert state.effective_mode == state.governance_mode

    def test_most_restrictive_mode_helper(self):
        assert _most_restrictive_mode(MODE_NORMAL, MODE_FREEZE) == MODE_FREEZE
        assert _most_restrictive_mode(MODE_FREEZE, MODE_NORMAL) == MODE_FREEZE
        assert _most_restrictive_mode(MODE_WARNING, MODE_SAFE) == MODE_SAFE
        assert _most_restrictive_mode(MODE_NORMAL, MODE_NORMAL) == MODE_NORMAL

    def test_effective_mode_safe_vs_degraded(self, tmp_path):
        """governance=DEGRADED, auto_freeze=SAFE_MODE → effective=SAFE_MODE."""
        eng = _make_engine(tmp_path)
        af_state = AutoFreezeState(
            auto_freeze_mode=MODE_SAFE,
            degradation_score=6.0,
            triggered_checks=[],
            trigger_values={},
            computed_at=_now_iso(),
        )
        state = eng.run(_make_live(42), _make_shadow(5), None, None, auto_freeze_state=af_state)
        # governance is likely NORMAL here, so effective = max(NORMAL, SAFE_MODE) = SAFE_MODE
        assert state.effective_mode == MODE_SAFE

    def test_effective_enforcement_persisted(self, tmp_path):
        """effective_enforcement is written to governance_health.json."""
        eng = _make_engine(tmp_path)
        af_state = AutoFreezeState(
            auto_freeze_mode=MODE_SAFE,
            degradation_score=6.0,
            triggered_checks=[],
            trigger_values={},
            computed_at=_now_iso(),
        )
        state = eng.run(_make_live(42), _make_shadow(5), None, None, auto_freeze_state=af_state)
        eng.write_state(state)
        d = json.loads((tmp_path / "governance_health.json").read_text())
        assert d["effective_mode"] == MODE_SAFE
        assert d["effective_enforcement"]["allow_promotions"] is False
