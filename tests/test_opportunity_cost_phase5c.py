"""
tests/test_opportunity_cost_phase5c.py
Phase 5C tests: enhanced schema, emit_opportunity_cost_from_deployment,
capacity-tracked OC events in CapitalDeploymentOS, new KPIs, fail-open.
"""
from __future__ import annotations

import json
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from src.analytics.opportunity_cost import (
    HeldPositionSummary,
    OpportunityCostRecord,
    MaterializedRecord,
    RejectedCandidate,
    build_opportunity_cost_record,
    emit_opportunity_cost_from_deployment,
    aggregate_kpis,
    format_opportunity_cost_report,
    _tier_from_continuation_score,
    _load_jsonl,
    _append_jsonl,
    SEVERITY_SEVERE,
    SEVERITY_MODERATE,
    SEVERITY_LOW,
    SCHEMA_VERSION,
)
from src.live.capital_deployment_os import (
    CapitalDeploymentOS,
    CandidateInfo,
    DeploymentContext,
    HeldPosition,
    DECISION_FULL_ENTRY,
    DECISION_REDUCED_STARTER,
    DECISION_BLOCKED,
    REGIME_NORMAL,
    REGIME_CRASH,
    BROKER_HEALTHY,
    BROKER_DEGRADED,
    make_context,
    make_candidate,
    dynamic_max_positions,
)

TODAY = "2026-05-29"


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _held_summary(
    symbol="5803.T",
    score=40.0,
    tier="exit_candidate",
    phase="weak_breakout",
    hold_days=0,
    unrealized_pnl_pct=0.0,
    rank=1,
) -> HeldPositionSummary:
    return HeldPositionSummary(
        symbol=symbol,
        priority_score=score,
        priority_tier=tier,
        phase=phase,
        hold_days=hold_days,
        unrealized_pnl_pct=unrealized_pnl_pct,
        rank=rank,
    )


def _cand_rc(
    symbol="7011.T",
    cont=0.0,
    bq=80.0,
    comp=75.0,
    surv=0.85,
    rsr=88.0,
) -> RejectedCandidate:
    return RejectedCandidate(
        symbol=symbol,
        continuation_score=cont,
        breakout_quality=bq,
        compression_score=comp,
        survivability=surv,
        rsr=rsr,
    )


def _held_pos(
    symbol="5803.T",
    qty=100,
    avg_cost=1500.0,
    unrealized_pnl_pct=0.05,
    hold_days=30,
) -> HeldPosition:
    return HeldPosition(
        symbol=symbol,
        qty=qty,
        avg_cost=avg_cost,
        unrealized_pnl_pct=unrealized_pnl_pct,
        hold_days=hold_days,
        addon_count=0,
        last_addon_date="",
        sector="電気機器",
    )


def _ctx_with_positions(positions=None, deployable=3_000_000):
    if positions is None:
        positions = {}
    return make_context(
        actual_equity=deployable * 1.1,
        effective_capital=deployable,
        deployable_capital=deployable,
        capital_utilization=0.95,
        market_regime=REGIME_NORMAL,
        broker_health=BROKER_HEALTHY,
        volatility_spike=False,
        cb_active=False,
        date=TODAY,
        positions=positions,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. _tier_from_continuation_score
# ─────────────────────────────────────────────────────────────────────────────

class TestTierFromContinuationScore:
    def test_core_continuation_at_80(self):
        assert _tier_from_continuation_score(80.0) == "core_continuation"

    def test_core_continuation_above_80(self):
        assert _tier_from_continuation_score(95.0) == "core_continuation"

    def test_high_quality_at_65(self):
        assert _tier_from_continuation_score(65.0) == "high_quality"

    def test_high_quality_between_65_and_79(self):
        assert _tier_from_continuation_score(72.5) == "high_quality"

    def test_neutral_at_45(self):
        assert _tier_from_continuation_score(45.0) == "neutral"

    def test_neutral_between_45_and_64(self):
        assert _tier_from_continuation_score(55.0) == "neutral"

    def test_exit_candidate_below_45(self):
        assert _tier_from_continuation_score(44.9) == "exit_candidate"

    def test_exit_candidate_at_zero(self):
        assert _tier_from_continuation_score(0.0) == "exit_candidate"

    def test_boundary_exactly_80(self):
        assert _tier_from_continuation_score(80.0) == "core_continuation"

    def test_boundary_just_below_80(self):
        assert _tier_from_continuation_score(79.9) == "high_quality"

    def test_negative_input_is_exit(self):
        assert _tier_from_continuation_score(-10.0) == "exit_candidate"

    def test_over_100_is_core(self):
        assert _tier_from_continuation_score(120.0) == "core_continuation"


# ─────────────────────────────────────────────────────────────────────────────
# 2. HeldPositionSummary enhanced schema
# ─────────────────────────────────────────────────────────────────────────────

class TestHeldPositionSummarySchema:
    def test_new_fields_have_defaults(self):
        h = HeldPositionSummary(
            symbol="A", priority_score=50.0, priority_tier="neutral", phase="ok"
        )
        assert h.hold_days == 0
        assert h.unrealized_pnl_pct == 0.0
        assert h.rank == 1

    def test_new_fields_settable(self):
        h = _held_summary(hold_days=45, unrealized_pnl_pct=0.12, rank=3)
        assert h.hold_days == 45
        assert h.unrealized_pnl_pct == 0.12
        assert h.rank == 3

    def test_asdict_includes_new_fields(self):
        h = _held_summary(hold_days=10, unrealized_pnl_pct=-0.05, rank=2)
        d = asdict(h)
        assert "hold_days" in d
        assert "unrealized_pnl_pct" in d
        assert "rank" in d


# ─────────────────────────────────────────────────────────────────────────────
# 3. OpportunityCostRecord enhanced schema
# ─────────────────────────────────────────────────────────────────────────────

class TestOpportunityCostRecordSchema:
    def _make_record(self, **kwargs) -> OpportunityCostRecord:
        defaults = dict(
            date=TODAY,
            rejected_symbol="7011.T",
            candidate_priority_estimate=75.0,
            candidate_breakout_quality=80.0,
            candidate_compression_score=70.0,
            blocked_by_symbol="5803.T",
            blocked_position_priority=40.0,
            blocked_position_tier="exit_candidate",
            priority_gap=35.0,
            blocked_phase="weak",
            portfolio_full=True,
            opportunity_cost_severity=SEVERITY_MODERATE,
        )
        defaults.update(kwargs)
        return OpportunityCostRecord(**defaults)

    def test_phase5c_fields_have_defaults(self):
        r = self._make_record()
        assert r.run_id == ""
        assert r.candidate_virtual_rank == 0
        assert r.blocked_position_age_days == 0
        assert r.blocked_unrealized_pnl_pct == 0.0
        assert r.blocked_rank == 1
        assert r.active_positions == 0
        assert r.max_positions == 3

    def test_phase5c_fields_settable(self):
        r = self._make_record(
            run_id="run-001",
            candidate_virtual_rank=2,
            blocked_position_age_days=47,
            blocked_unrealized_pnl_pct=0.15,
            blocked_rank=3,
            active_positions=3,
            max_positions=3,
        )
        assert r.run_id == "run-001"
        assert r.candidate_virtual_rank == 2
        assert r.blocked_position_age_days == 47
        assert r.blocked_unrealized_pnl_pct == 0.15
        assert r.blocked_rank == 3
        assert r.active_positions == 3
        assert r.max_positions == 3

    def test_asdict_serializes_all_phase5c_fields(self):
        r = self._make_record(run_id="x", blocked_position_age_days=20)
        d = asdict(r)
        assert d["run_id"] == "x"
        assert d["blocked_position_age_days"] == 20
        assert "candidate_virtual_rank" in d
        assert "active_positions" in d
        assert "max_positions" in d


# ─────────────────────────────────────────────────────────────────────────────
# 4. build_opportunity_cost_record — new params
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildOCRecordPhase5C:
    def test_override_priority_used_instead_of_estimate(self):
        cand = _cand_rc(bq=0.0, comp=0.0, surv=0.0, rsr=0.0)  # would estimate ~0
        held = _held_summary(score=30.0)
        rec = build_opportunity_cost_record(
            cand, held, TODAY, True, override_priority=85.0
        )
        assert rec.candidate_priority_estimate == 85.0

    def test_new_fields_populated_from_held(self):
        held = _held_summary(score=40.0, hold_days=30, unrealized_pnl_pct=0.10, rank=3)
        cand = _cand_rc()
        rec = build_opportunity_cost_record(
            cand, held, TODAY, True,
            run_id="test-run",
            candidate_virtual_rank=2,
            active_positions=3,
            max_positions=3,
        )
        assert rec.run_id == "test-run"
        assert rec.candidate_virtual_rank == 2
        assert rec.blocked_position_age_days == 30
        assert rec.blocked_unrealized_pnl_pct == 0.10
        assert rec.blocked_rank == 3
        assert rec.active_positions == 3
        assert rec.max_positions == 3

    def test_default_params_maintain_backward_compat(self):
        cand = _cand_rc()
        held = _held_summary()
        rec = build_opportunity_cost_record(cand, held, TODAY, True)
        assert rec.run_id == ""
        assert rec.candidate_virtual_rank == 0
        assert rec.blocked_position_age_days == 0
        assert rec.max_positions == 3

    def test_gap_computed_from_override_priority(self):
        cand = _cand_rc()
        held = _held_summary(score=60.0)
        rec = build_opportunity_cost_record(
            cand, held, TODAY, True, override_priority=90.0
        )
        assert rec.priority_gap == pytest.approx(30.0)


# ─────────────────────────────────────────────────────────────────────────────
# 5. emit_opportunity_cost_from_deployment
# ─────────────────────────────────────────────────────────────────────────────

class TestEmitOpportunityCostFromDeployment:
    def _make_cand_info(
        self,
        symbol="7011.T",
        cont=72.0,
        surv=0.82,
        pred=65.0,
    ) -> CandidateInfo:
        return make_candidate(
            symbol=symbol,
            price=2500.0,
            continuation_score=cont,
            survivability=surv,
            predictive_score=pred,
            is_held=False,
        )

    def test_writes_jsonl_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            events_file = Path(tmp) / "oc_events.jsonl"
            ctx = _ctx_with_positions({
                "5803.T": _held_pos("5803.T", hold_days=30, unrealized_pnl_pct=0.05)
            })
            emit_opportunity_cost_from_deployment(
                candidate=self._make_cand_info(),
                context=ctx,
                held_priorities={"5803.T": 40.0},
                candidate_virtual_rank=1,
                max_positions=3,
                events_file=events_file,
                run_id="test-001",
            )
            records = _load_jsonl(events_file)
            assert len(records) == 1
            r = records[0]
            assert r["rejected_symbol"] == "7011.T"
            assert r["run_id"] == "test-001"

    def test_portfolio_full_true_in_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            events_file = Path(tmp) / "oc_events.jsonl"
            ctx = _ctx_with_positions({"5803.T": _held_pos("5803.T")})
            emit_opportunity_cost_from_deployment(
                candidate=self._make_cand_info(),
                context=ctx,
                held_priorities={"5803.T": 35.0},
                candidate_virtual_rank=1,
                max_positions=3,
                events_file=events_file,
            )
            r = _load_jsonl(events_file)[0]
            assert r["portfolio_full"] is True

    def test_continuation_score_used_as_priority(self):
        with tempfile.TemporaryDirectory() as tmp:
            events_file = Path(tmp) / "oc_events.jsonl"
            ctx = _ctx_with_positions({"5803.T": _held_pos("5803.T")})
            emit_opportunity_cost_from_deployment(
                candidate=self._make_cand_info(cont=80.0),
                context=ctx,
                held_priorities={"5803.T": 40.0},
                candidate_virtual_rank=1,
                max_positions=3,
                events_file=events_file,
            )
            r = _load_jsonl(events_file)[0]
            assert r["candidate_priority_estimate"] == pytest.approx(80.0)

    def test_blocked_position_age_and_pnl_captured(self):
        with tempfile.TemporaryDirectory() as tmp:
            events_file = Path(tmp) / "oc_events.jsonl"
            ctx = _ctx_with_positions({
                "5803.T": _held_pos("5803.T", hold_days=47, unrealized_pnl_pct=0.15)
            })
            emit_opportunity_cost_from_deployment(
                candidate=self._make_cand_info(),
                context=ctx,
                held_priorities={"5803.T": 40.0},
                candidate_virtual_rank=2,
                max_positions=3,
                events_file=events_file,
            )
            r = _load_jsonl(events_file)[0]
            assert r["blocked_position_age_days"] == 47
            assert r["blocked_unrealized_pnl_pct"] == pytest.approx(0.15)
            assert r["candidate_virtual_rank"] == 2

    def test_max_positions_stored_in_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            events_file = Path(tmp) / "oc_events.jsonl"
            ctx = _ctx_with_positions({"5803.T": _held_pos("5803.T")})
            emit_opportunity_cost_from_deployment(
                candidate=self._make_cand_info(),
                context=ctx,
                held_priorities={"5803.T": 40.0},
                candidate_virtual_rank=1,
                max_positions=4,
                events_file=events_file,
            )
            r = _load_jsonl(events_file)[0]
            assert r["max_positions"] == 4

    def test_active_positions_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            events_file = Path(tmp) / "oc_events.jsonl"
            positions = {
                "A.T": _held_pos("A.T"),
                "B.T": _held_pos("B.T"),
                "C.T": _held_pos("C.T"),
            }
            ctx = _ctx_with_positions(positions)
            emit_opportunity_cost_from_deployment(
                candidate=self._make_cand_info(),
                context=ctx,
                held_priorities={"A.T": 50.0, "B.T": 40.0, "C.T": 30.0},
                candidate_virtual_rank=1,
                max_positions=3,
                events_file=events_file,
            )
            r = _load_jsonl(events_file)[0]
            assert r["active_positions"] == 3

    def test_lowest_priority_position_is_blocked_by(self):
        with tempfile.TemporaryDirectory() as tmp:
            events_file = Path(tmp) / "oc_events.jsonl"
            ctx = _ctx_with_positions({
                "HIGH.T": _held_pos("HIGH.T"),
                "LOW.T": _held_pos("LOW.T"),
            })
            emit_opportunity_cost_from_deployment(
                candidate=self._make_cand_info(),
                context=ctx,
                held_priorities={"HIGH.T": 80.0, "LOW.T": 20.0},
                candidate_virtual_rank=1,
                max_positions=3,
                events_file=events_file,
            )
            r = _load_jsonl(events_file)[0]
            assert r["blocked_by_symbol"] == "LOW.T"

    def test_fail_open_on_empty_held_priorities(self):
        with tempfile.TemporaryDirectory() as tmp:
            events_file = Path(tmp) / "oc_events.jsonl"
            ctx = _ctx_with_positions()
            # Should not raise even though held_priorities is empty
            emit_opportunity_cost_from_deployment(
                candidate=self._make_cand_info(),
                context=ctx,
                held_priorities={},
                candidate_virtual_rank=1,
                max_positions=3,
                events_file=events_file,
            )
            assert not events_file.exists()  # nothing written

    def test_fail_open_on_bad_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            events_file = Path(tmp) / "oc_events.jsonl"
            ctx = _ctx_with_positions({"A.T": _held_pos("A.T")})

            class _Bomb:
                """Object that raises on most attribute access except __class__."""
                @property
                def continuation_score(self):
                    raise RuntimeError("boom")
                @property
                def symbol(self):
                    raise RuntimeError("boom")
                @property
                def predictive_score(self):
                    raise RuntimeError("boom")

            # Should not raise — fail-open swallows the RuntimeError
            emit_opportunity_cost_from_deployment(
                candidate=_Bomb(),
                context=ctx,
                held_priorities={"A.T": 40.0},
                candidate_virtual_rank=1,
                max_positions=3,
                events_file=events_file,
            )

    def test_run_id_propagated_to_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            events_file = Path(tmp) / "oc_events.jsonl"
            ctx = _ctx_with_positions({"5803.T": _held_pos("5803.T")})
            emit_opportunity_cost_from_deployment(
                candidate=self._make_cand_info(),
                context=ctx,
                held_priorities={"5803.T": 40.0},
                candidate_virtual_rank=1,
                max_positions=3,
                events_file=events_file,
                run_id="abc-123",
            )
            r = _load_jsonl(events_file)[0]
            assert r["run_id"] == "abc-123"

    def test_ts_field_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            events_file = Path(tmp) / "oc_events.jsonl"
            ctx = _ctx_with_positions({"5803.T": _held_pos("5803.T")})
            emit_opportunity_cost_from_deployment(
                candidate=self._make_cand_info(),
                context=ctx,
                held_priorities={"5803.T": 40.0},
                candidate_virtual_rank=1,
                max_positions=3,
                events_file=events_file,
            )
            r = _load_jsonl(events_file)[0]
            assert "ts" in r

    def test_tier_assigned_from_continuation_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            events_file = Path(tmp) / "oc_events.jsonl"
            ctx = _ctx_with_positions({"5803.T": _held_pos("5803.T")})
            # Held position with score 80 → core_continuation
            emit_opportunity_cost_from_deployment(
                candidate=self._make_cand_info(cont=50.0),
                context=ctx,
                held_priorities={"5803.T": 80.0},
                candidate_virtual_rank=1,
                max_positions=3,
                events_file=events_file,
            )
            r = _load_jsonl(events_file)[0]
            assert r["blocked_position_tier"] == "core_continuation"


# ─────────────────────────────────────────────────────────────────────────────
# 6. CapitalDeploymentOS capacity tracking + OC emission
# ─────────────────────────────────────────────────────────────────────────────

class TestCapacityTracking:
    def _make_os(self, tmp_dir: Path) -> CapitalDeploymentOS:
        return CapitalDeploymentOS(
            opportunity_cost_dir=tmp_dir,
        )

    def _make_full_portfolio_ctx(self, deployable=1_000_000) -> DeploymentContext:
        # 1M capital → dynamic_max_positions=3; fill all 3 slots
        positions = {
            f"{i}.T": HeldPosition(
                symbol=f"{i}.T",
                qty=100,
                avg_cost=1000.0,
                unrealized_pnl_pct=0.05,
                hold_days=20,
                addon_count=0,
                last_addon_date="",
                sector="電気機器",
            )
            for i in range(1, 4)  # 3 positions = max for 1M capital
        }
        return DeploymentContext(
            actual_equity=deployable * 1.1,
            effective_capital=deployable,
            deployable_capital=deployable,
            current_positions=positions,
            market_regime=REGIME_NORMAL,
            broker_health=BROKER_HEALTHY,
            volatility_spike=False,
            cb_active=False,
            date=TODAY,
            capital_utilization=0.95,
        )

    def test_capacity_blocked_when_max_positions_reached(self):
        with tempfile.TemporaryDirectory() as tmp:
            os_obj = self._make_os(Path(tmp))
            ctx = self._make_full_portfolio_ctx()
            max_pos = dynamic_max_positions(ctx.deployable_capital)
            # Add held candidates filling positions
            held_cands = [
                make_candidate(
                    symbol=f"{i}.T",
                    price=1000.0,
                    continuation_score=50.0 + i,
                    is_held=True,
                    held_info=ctx.current_positions[f"{i}.T"],
                )
                for i in range(1, 4)
            ]
            # Add new candidates beyond capacity
            new_cands = [
                make_candidate(
                    symbol=f"new{j}.T",
                    price=1000.0,
                    continuation_score=70.0,
                    survivability=0.80,
                )
                for j in range(1, 3)
            ]
            decisions = os_obj.run(held_cands + new_cands, ctx, run_id="test-run")
            new_decisions = [
                d for d in decisions if d.symbol.startswith("new")
            ]
            # All new entries must be BLOCKED (capacity full)
            for d in new_decisions:
                assert d.decision_type == DECISION_BLOCKED
                assert "capacity_blocked" in d.suppression_reason

    def test_oc_events_written_when_capacity_full(self):
        with tempfile.TemporaryDirectory() as tmp:
            oc_dir = Path(tmp) / "oc"
            os_obj = CapitalDeploymentOS(opportunity_cost_dir=oc_dir)
            ctx = self._make_full_portfolio_ctx()
            held_cands = [
                make_candidate(
                    symbol=f"{i}.T",
                    price=1000.0,
                    continuation_score=40.0,
                    is_held=True,
                    held_info=ctx.current_positions[f"{i}.T"],
                )
                for i in range(1, 4)
            ]
            new_cand = make_candidate(
                symbol="7011.T",
                price=1000.0,
                continuation_score=70.0,
                survivability=0.80,
            )
            os_obj.run(held_cands + [new_cand], ctx, run_id="oc-test")
            events_file = oc_dir / "oc_events.jsonl"
            assert events_file.exists()
            records = _load_jsonl(events_file)
            assert len(records) >= 1
            assert records[0]["rejected_symbol"] == "7011.T"

    def test_no_oc_events_when_capacity_available(self):
        with tempfile.TemporaryDirectory() as tmp:
            oc_dir = Path(tmp) / "oc"
            os_obj = CapitalDeploymentOS(opportunity_cost_dir=oc_dir)
            # Only 1 position held, 3M capital allows 3 — room available
            one_pos = {"1.T": HeldPosition(
                symbol="1.T", qty=100, avg_cost=1000.0, unrealized_pnl_pct=0.05,
                hold_days=20, addon_count=0, last_addon_date="", sector="電気機器",
            )}
            ctx = DeploymentContext(
                actual_equity=3_300_000,
                effective_capital=3_000_000,
                deployable_capital=3_000_000,
                current_positions=one_pos,
                market_regime=REGIME_NORMAL,
                broker_health=BROKER_HEALTHY,
                volatility_spike=False,
                cb_active=False,
                date=TODAY,
                capital_utilization=0.33,
            )
            held_cand = make_candidate(
                symbol="1.T", price=1000.0, continuation_score=50.0,
                is_held=True, held_info=one_pos["1.T"],
            )
            new_cand = make_candidate(
                symbol="7011.T", price=1000.0, continuation_score=70.0, survivability=0.80,
            )
            os_obj.run([held_cand, new_cand], ctx)
            events_file = oc_dir / "oc_events.jsonl"
            if events_file.exists():
                records = _load_jsonl(events_file)
                assert len(records) == 0

    def test_oc_not_emitted_when_safety_gate_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            oc_dir = Path(tmp) / "oc"
            os_obj = CapitalDeploymentOS(opportunity_cost_dir=oc_dir)
            ctx = self._make_full_portfolio_ctx()
            # Override to crash regime (safety gate fails → governance_passed=False)
            from dataclasses import replace
            ctx = replace(ctx, market_regime=REGIME_CRASH)
            held_cands = [
                make_candidate(
                    symbol=f"{i}.T", price=1000.0, continuation_score=40.0,
                    is_held=True, held_info=ctx.current_positions[f"{i}.T"],
                )
                for i in range(1, 4)
            ]
            new_cand = make_candidate(
                symbol="7011.T", price=1000.0, continuation_score=70.0, survivability=0.80,
            )
            os_obj.run(held_cands + [new_cand], ctx)
            events_file = oc_dir / "oc_events.jsonl"
            if events_file.exists():
                assert len(_load_jsonl(events_file)) == 0

    def test_oc_not_emitted_when_volatility_spike(self):
        with tempfile.TemporaryDirectory() as tmp:
            oc_dir = Path(tmp) / "oc"
            os_obj = CapitalDeploymentOS(opportunity_cost_dir=oc_dir)
            ctx = self._make_full_portfolio_ctx()
            from dataclasses import replace
            ctx = replace(ctx, volatility_spike=True)
            held_cands = [
                make_candidate(
                    symbol=f"{i}.T", price=1000.0, continuation_score=40.0,
                    is_held=True, held_info=ctx.current_positions[f"{i}.T"],
                )
                for i in range(1, 4)
            ]
            new_cand = make_candidate(
                symbol="7011.T", price=1000.0, continuation_score=70.0, survivability=0.80,
            )
            os_obj.run(held_cands + [new_cand], ctx)
            events_file = oc_dir / "oc_events.jsonl"
            if events_file.exists():
                assert len(_load_jsonl(events_file)) == 0

    def test_downgrade_to_capacity_blocked_is_blocked(self):
        from src.live.capital_deployment_os import _make_blocked
        ctx = make_context(
            actual_equity=3_300_000,
            effective_capital=3_000_000,
            deployable_capital=3_000_000,
            date=TODAY,
        )
        cand = make_candidate(symbol="X.T", price=1000.0, continuation_score=70.0)
        decision = _make_blocked(cand, ctx, "original")
        from dataclasses import replace
        from src.live.capital_deployment_os import DECISION_BLOCKED
        downgraded = CapitalDeploymentOS._downgrade_to_capacity_blocked(decision)
        assert downgraded.decision_type == DECISION_BLOCKED
        assert downgraded.target_size == 0
        assert "capacity_blocked" in downgraded.suppression_reason

    def test_run_id_passed_to_oc_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            oc_dir = Path(tmp) / "oc"
            os_obj = CapitalDeploymentOS(opportunity_cost_dir=oc_dir)
            ctx = self._make_full_portfolio_ctx()
            held_cands = [
                make_candidate(
                    symbol=f"{i}.T", price=1000.0, continuation_score=40.0,
                    is_held=True, held_info=ctx.current_positions[f"{i}.T"],
                )
                for i in range(1, 4)
            ]
            new_cand = make_candidate(
                symbol="7011.T", price=1000.0, continuation_score=70.0, survivability=0.80,
            )
            os_obj.run(held_cands + [new_cand], ctx, run_id="my-run-xyz")
            events_file = oc_dir / "oc_events.jsonl"
            if events_file.exists():
                records = _load_jsonl(events_file)
                if records:
                    assert records[0]["run_id"] == "my-run-xyz"

    def test_no_oc_dir_no_error(self):
        # If opportunity_cost_dir is None, capacity blocking still works but no JSONL
        os_obj = CapitalDeploymentOS(opportunity_cost_dir=None)
        ctx = self._make_full_portfolio_ctx()
        held_cands = [
            make_candidate(
                symbol=f"{i}.T", price=1000.0, continuation_score=40.0,
                is_held=True, held_info=ctx.current_positions[f"{i}.T"],
            )
            for i in range(1, 4)
        ]
        new_cand = make_candidate(
            symbol="7011.T", price=1000.0, continuation_score=70.0, survivability=0.80,
        )
        # Should not raise
        decisions = os_obj.run(held_cands + [new_cand], ctx)
        assert decisions is not None


# ─────────────────────────────────────────────────────────────────────────────
# 7. aggregate_kpis new KPIs
# ─────────────────────────────────────────────────────────────────────────────

class TestAggregateKpisPhase5C:
    def _write_records(self, path: Path, records: list) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        for r in records:
            with path.open("a") as f:
                f.write(json.dumps(r) + "\n")

    def _base_record(self, severity=SEVERITY_SEVERE, max_positions=3) -> dict:
        return {
            "date": TODAY,
            "rejected_symbol": "7011.T",
            "candidate_priority_estimate": 80.0,
            "candidate_breakout_quality": 75.0,
            "candidate_compression_score": 70.0,
            "blocked_by_symbol": "5803.T",
            "blocked_position_priority": 35.0,
            "blocked_position_tier": "exit_candidate",
            "priority_gap": 45.0,
            "blocked_phase": "weak",
            "portfolio_full": True,
            "opportunity_cost_severity": severity,
            "schema_version": SCHEMA_VERSION,
            "rejected_candidate_5d_return": 0.05,
            "blocked_position_5d_return": 0.02,
            "rotation_counterfactual_delta": 0.03,
            "materialized_date": TODAY,
            "materialization_complete": True,
            "max_positions": max_positions,
        }

    def test_severe_event_rate_all_severe(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mat.jsonl"
            self._write_records(path, [self._base_record(SEVERITY_SEVERE)] * 4)
            kpis = aggregate_kpis(path)
            assert kpis["severe_event_rate"] == pytest.approx(1.0)

    def test_severe_event_rate_none_severe(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mat.jsonl"
            self._write_records(path, [self._base_record(SEVERITY_LOW)] * 3)
            kpis = aggregate_kpis(path)
            assert kpis["severe_event_rate"] == pytest.approx(0.0)

    def test_severe_event_rate_mixed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mat.jsonl"
            records = [
                self._base_record(SEVERITY_SEVERE),
                self._base_record(SEVERITY_SEVERE),
                self._base_record(SEVERITY_MODERATE),
            ]
            self._write_records(path, records)
            kpis = aggregate_kpis(path)
            assert kpis["severe_event_rate"] == pytest.approx(2 / 3, rel=1e-3)

    def test_opportunity_cost_per_slot_single_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mat.jsonl"
            self._write_records(path, [self._base_record(max_positions=3)])
            kpis = aggregate_kpis(path)
            assert kpis["opportunity_cost_per_slot"] == pytest.approx(1 / 3, rel=1e-3)

    def test_opportunity_cost_per_slot_multiple_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mat.jsonl"
            records = [self._base_record(max_positions=3)] * 3
            self._write_records(path, records)
            kpis = aggregate_kpis(path)
            # 3 events / (3 * 3 slots) = 1/3
            assert kpis["opportunity_cost_per_slot"] == pytest.approx(1 / 3, rel=1e-3)

    def test_empty_file_returns_zero_new_kpis(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mat.jsonl"
            kpis = aggregate_kpis(path)
            assert kpis["severe_event_rate"] == 0.0
            assert kpis["opportunity_cost_per_slot"] == 0.0

    def test_new_kpis_present_in_result_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mat.jsonl"
            kpis = aggregate_kpis(path)
            assert "severe_event_rate" in kpis
            assert "opportunity_cost_per_slot" in kpis


# ─────────────────────────────────────────────────────────────────────────────
# 8. format_opportunity_cost_report with new fields
# ─────────────────────────────────────────────────────────────────────────────

class TestFormatReportPhase5C:
    def _write_event(self, path: Path, age_days: int = 0, pnl_pct: float = 0.0) -> None:
        record = {
            "date": TODAY,
            "rejected_symbol": "7011.T",
            "candidate_priority_estimate": 80.0,
            "candidate_breakout_quality": 75.0,
            "candidate_compression_score": 70.0,
            "blocked_by_symbol": "5803.T",
            "blocked_position_priority": 35.0,
            "blocked_position_tier": "exit_candidate",
            "priority_gap": 45.0,
            "blocked_phase": "weak",
            "portfolio_full": True,
            "opportunity_cost_severity": SEVERITY_SEVERE,
            "schema_version": SCHEMA_VERSION,
            "blocked_position_age_days": age_days,
            "blocked_unrealized_pnl_pct": pnl_pct,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(json.dumps(record) + "\n")

    def test_age_shown_in_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            self._write_event(path, age_days=47, pnl_pct=0.12)
            report = format_opportunity_cost_report(path, date=TODAY)
            assert "age=47d" in report

    def test_pnl_shown_in_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            self._write_event(path, age_days=47, pnl_pct=0.12)
            report = format_opportunity_cost_report(path, date=TODAY)
            assert "pnl=" in report

    def test_no_age_section_when_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            self._write_event(path, age_days=0, pnl_pct=0.0)
            report = format_opportunity_cost_report(path, date=TODAY)
            # age=0d should not appear (suppressed)
            assert "age=0d" not in report
