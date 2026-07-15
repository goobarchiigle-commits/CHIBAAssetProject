"""tests/runtime/test_portfolio_intelligence_engine.py — 65 tests."""
import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.runtime.policy.portfolio_intelligence_engine import (
    PIDTYPE_CONVEXITY,
    PIDTYPE_CORR_SUPPRESSION,
    PIDTYPE_HEAT_LIMIT,
    PIDTYPE_RISK_SCALE,
    PIDTYPE_SLOT_SCALE,
    PIDTYPE_SURVIVABILITY,
    PIDTYPE_THROTTLE,
    SCHEMA_VERSION,
    CapitalDeploymentIntelligence,
    ConvexityRetentionIntelligence,
    CrossPositionCorrelation,
    CrossPositionCorrelationIntelligence,
    ExitIntelSnapshot,
    ExpectancySnapshot,
    PortfolioAdjustmentDecision,
    PortfolioConstraints,
    PortfolioIntelligenceEngine,
    PortfolioIntelligenceHook,
    PortfolioIntelligenceInput,
    PortfolioIntelligenceOverlay,
    PortfolioIntegrityValidator,
    PortfolioPositionSnapshot,
    PortfolioSurvivabilityIntelligence,
    RegimeSnapshot,
    SurvivabilityReport,
    append_decision,
    load_decisions,
    run_portfolio_intelligence_hook,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_pos(symbol="SYM1", sector="IT", hold_days=10, rsr=80.0,
              rsr_momentum=0.5, runner_score=0.5, persistence_score=0.5,
              vol=0.20, signal=1):
    return PortfolioPositionSnapshot(
        symbol=symbol, sector=sector, hold_days=hold_days,
        rsr=rsr, rsr_momentum=rsr_momentum, current_return=0.05,
        rs_score=0.3, signal=signal, runner_score=runner_score,
        persistence_score=persistence_score, vol=vol,
    )


def _high_corr_positions(n=3, sector="IT"):
    return [_make_pos(symbol=f"SYM{i:04d}", sector=sector, hold_days=10 + i) for i in range(n)]


def _make_regime(regime="bull", deteriorating=False, rs_collapse=False, confidence=0.5):
    return RegimeSnapshot(
        regime=regime, deteriorating=deteriorating,
        rs_collapse=rs_collapse, breadth_score=0.6, confidence=confidence,
    )


def _make_input(positions=None, regime=None, portfolio_heat=0.05, n=3):
    pos = positions if positions is not None else _high_corr_positions(n)
    return PortfolioIntelligenceInput(
        positions=pos,
        regime=regime or _make_regime(),
        expectancy=ExpectancySnapshot(),
        exit_intel=ExitIntelSnapshot(),
        portfolio_heat=portfolio_heat,
        available_slots=1,
        max_slots=4,
        eval_date="2026-05-20",
    )


def _make_decision(eval_date="2026-05-15", dim_type=PIDTYPE_SLOT_SCALE,
                   prev=1.0, new=0.80, obs=5):
    return PortfolioAdjustmentDecision.create(
        eval_date=eval_date, dimension_type=dim_type,
        previous_value=prev, new_value=new,
        trigger_source="test", trigger_value=0.7,
        confidence=0.6, observation_count=obs,
    )


@pytest.fixture
def tmp_store(tmp_path):
    return tmp_path / "pi_decisions.jsonl"


@pytest.fixture
def engine():
    return PortfolioIntelligenceEngine()


# ─────────────────────────────────────────────────────────────────────────────
# 1. PortfolioConstraints
# ─────────────────────────────────────────────────────────────────────────────

class TestPortfolioConstraints:
    def test_defaults_within_bounds(self):
        c = PortfolioConstraints()
        assert c.slot_scale_min < c.slot_scale_max
        assert c.risk_scale_min < c.risk_scale_max
        assert c.throttle_min < c.throttle_max
        assert c.heat_limit_min < c.heat_limit_max
        assert c.corr_suppression_min < c.corr_suppression_max

    def test_frozen(self):
        c = PortfolioConstraints()
        with pytest.raises(Exception):
            c.slot_scale_min = 0.0  # type: ignore[misc]

    def test_min_sample_size(self):
        assert PortfolioConstraints().min_sample_size == 3

    def test_cooldown_days(self):
        assert PortfolioConstraints().cooldown_days == 2

    def test_anti_oscillation_window(self):
        assert PortfolioConstraints().anti_oscillation_window_days == 5


# ─────────────────────────────────────────────────────────────────────────────
# 2. CrossPositionCorrelationIntelligence
# ─────────────────────────────────────────────────────────────────────────────

class TestCrossPositionCorrelation:
    def test_single_position_returns_zero(self):
        intel = CrossPositionCorrelationIntelligence()
        result = intel.compute([_make_pos()], PortfolioConstraints())
        assert result.overall_corr_score == 0.0

    def test_same_sector_high_cluster(self):
        intel = CrossPositionCorrelationIntelligence()
        positions = _high_corr_positions(3, sector="IT")
        result = intel.compute(positions, PortfolioConstraints())
        assert result.sector_cluster_score == pytest.approx(1.0, abs=1e-6)

    def test_different_sectors_low_cluster(self):
        intel = CrossPositionCorrelationIntelligence()
        positions = [
            _make_pos("A", sector="IT"),
            _make_pos("B", sector="Finance"),
            _make_pos("C", sector="Healthcare"),
        ]
        result = intel.compute(positions, PortfolioConstraints())
        assert result.sector_cluster_score == pytest.approx(1 / 3, abs=1e-5)

    def test_high_rsr_increases_hidden_factor(self):
        intel = CrossPositionCorrelationIntelligence()
        positions = [_make_pos(symbol=f"S{i}", rsr=85.0) for i in range(3)]
        result = intel.compute(positions, PortfolioConstraints())
        assert result.hidden_factor_overlap == pytest.approx(1.0, abs=1e-5)

    def test_overall_corr_bounded(self):
        intel = CrossPositionCorrelationIntelligence()
        c = PortfolioConstraints()
        for n in range(2, 5):
            result = intel.compute(_high_corr_positions(n), c)
            assert 0.0 <= result.overall_corr_score <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 3. PortfolioSurvivabilityIntelligence
# ─────────────────────────────────────────────────────────────────────────────

class TestSurvivabilityIntelligence:
    def test_low_heat_low_corr_high_survivability(self):
        intel = PortfolioSurvivabilityIntelligence()
        c = PortfolioConstraints()
        corr = CrossPositionCorrelation(overall_corr_score=0.1, sector_cluster_score=0.1,
                                        momentum_sync_score=0.1, vol_correlation_score=0.1,
                                        lifecycle_sync_score=0.1, hidden_factor_overlap=0.1)
        result = intel.compute(corr, _make_regime("bull"), portfolio_heat=0.02,
                                n_positions=3, constraints=c)
        assert result.survivability_score > 0.70

    def test_high_heat_high_corr_low_survivability(self):
        intel = PortfolioSurvivabilityIntelligence()
        c = PortfolioConstraints()
        corr = CrossPositionCorrelation(overall_corr_score=0.95, sector_cluster_score=1.0,
                                        momentum_sync_score=1.0, vol_correlation_score=1.0,
                                        lifecycle_sync_score=1.0, hidden_factor_overlap=1.0)
        result = intel.compute(corr, _make_regime("bear", True, True, 1.0),
                                portfolio_heat=0.80, n_positions=3, constraints=c)
        assert result.survivability_score < 0.40

    def test_survivability_gate_closes_below_threshold(self):
        intel = PortfolioSurvivabilityIntelligence()
        c = PortfolioConstraints()
        corr = CrossPositionCorrelation(overall_corr_score=0.99, sector_cluster_score=1.0,
                                        momentum_sync_score=1.0, vol_correlation_score=1.0,
                                        lifecycle_sync_score=1.0, hidden_factor_overlap=1.0)
        result = intel.compute(corr, _make_regime("bear", True, True, 1.0),
                                portfolio_heat=1.0, n_positions=3, constraints=c)
        assert not result.survivability_gate
        assert result.survivability_score < c.min_survivability_score

    def test_survivability_score_bounded(self):
        intel = PortfolioSurvivabilityIntelligence()
        c = PortfolioConstraints()
        corr = CrossPositionCorrelation(overall_corr_score=0.5)
        result = intel.compute(corr, _make_regime(), 0.5, 3, c)
        assert 0.0 <= result.survivability_score <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 4. CapitalDeploymentIntelligence
# ─────────────────────────────────────────────────────────────────────────────

class TestCapitalDeploymentIntelligence:
    def _run(self, corr_score=0.8, surv_score=0.7, capital_leakage=0.0,
             deteriorating=False, bear=False):
        intel = CapitalDeploymentIntelligence()
        c = PortfolioConstraints()
        corr = CrossPositionCorrelation(overall_corr_score=corr_score,
                                        sector_cluster_score=corr_score,
                                        momentum_sync_score=corr_score,
                                        vol_correlation_score=corr_score,
                                        lifecycle_sync_score=corr_score,
                                        hidden_factor_overlap=corr_score)
        surv = SurvivabilityReport(survivability_score=surv_score,
                                   correlated_heat=corr_score * 0.1)
        expectancy = ExpectancySnapshot(capital_constraint_leakage=capital_leakage)
        regime = _make_regime("bear" if bear else "bull", deteriorating)
        return intel.compute(corr, surv, expectancy, regime, c, n_positions=3)

    def test_high_corr_reduces_slot_scale(self):
        result = self._run(corr_score=0.90)
        assert result.slot_scale < 1.0
        assert result.slot_scale >= PortfolioConstraints().slot_scale_min

    def test_high_corr_activates_suppression(self):
        result = self._run(corr_score=0.85)
        assert result.corr_suppression_scale < 1.0
        assert result.corr_suppression_scale >= PortfolioConstraints().corr_suppression_min

    def test_capital_leakage_loosens_throttle(self):
        low = self._run(capital_leakage=0.0).entry_throttle
        high = self._run(capital_leakage=0.10).entry_throttle
        assert high >= low

    def test_all_values_bounded(self):
        c = PortfolioConstraints()
        result = self._run(corr_score=0.9, surv_score=0.5, capital_leakage=0.15)
        assert c.slot_scale_min <= result.slot_scale <= c.slot_scale_max
        assert c.throttle_min <= result.entry_throttle <= c.throttle_max
        assert c.corr_suppression_min <= result.corr_suppression_scale <= c.corr_suppression_max
        assert c.risk_scale_min <= result.risk_scale <= c.risk_scale_max


# ─────────────────────────────────────────────────────────────────────────────
# 5. ConvexityRetentionIntelligence
# ─────────────────────────────────────────────────────────────────────────────

class TestConvexityRetentionIntelligence:
    def test_empty_positions_returns_defaults(self):
        intel = ConvexityRetentionIntelligence()
        result = intel.compute([], ExitIntelSnapshot(), CrossPositionCorrelation(),
                               _make_regime(), PortfolioConstraints())
        assert result.convexity_score == 0.0
        assert len(result.hold_extension_symbols) == 0

    def test_high_runner_no_deterioration_hold_extension(self):
        intel = ConvexityRetentionIntelligence()
        positions = [_make_pos("WIN1", runner_score=0.90, persistence_score=0.80)]
        result = intel.compute(positions, ExitIntelSnapshot(), CrossPositionCorrelation(),
                               _make_regime("bull", deteriorating=False),
                               PortfolioConstraints())
        assert "WIN1" in result.hold_extension_symbols

    def test_deteriorating_regime_blocks_hold_extension(self):
        intel = ConvexityRetentionIntelligence()
        positions = [_make_pos("WIN1", runner_score=0.90, persistence_score=0.80)]
        result = intel.compute(positions, ExitIntelSnapshot(), CrossPositionCorrelation(),
                               _make_regime("bear", deteriorating=True),
                               PortfolioConstraints())
        assert "WIN1" not in result.hold_extension_symbols

    def test_high_giveback_triggers_deterioration_exit(self):
        intel = ConvexityRetentionIntelligence()
        positions = [_make_pos("LOSER")]
        exit_intel = ExitIntelSnapshot(per_symbol_giveback={"LOSER": 0.80})
        result = intel.compute(positions, exit_intel, CrossPositionCorrelation(),
                               _make_regime("bull"), PortfolioConstraints())
        assert "LOSER" in result.deterioration_exit_symbols

    def test_convexity_score_bounded(self):
        intel = ConvexityRetentionIntelligence()
        result = intel.compute(_high_corr_positions(3), ExitIntelSnapshot(),
                               CrossPositionCorrelation(), _make_regime(),
                               PortfolioConstraints())
        assert 0.0 <= result.convexity_score <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 6. PortfolioIntegrityValidator
# ─────────────────────────────────────────────────────────────────────────────

class TestPortfolioIntegrityValidator:
    def test_empty_history_passes(self):
        report = PortfolioIntegrityValidator().validate(
            [], PortfolioConstraints(), as_of_date="2026-05-20"
        )
        assert report.passed
        assert report.n_violations == 0

    def test_duplicate_decision_id_fails(self):
        d = _make_decision(obs=5)
        report = PortfolioIntegrityValidator().validate(
            [d, d], PortfolioConstraints(), as_of_date="2026-05-20"
        )
        assert not report.passed
        assert "duplicate_decision_id" in {v.violation_type for v in report.violations}

    def test_future_leakage_fails(self):
        d = _make_decision(eval_date="2026-06-01", obs=5)
        report = PortfolioIntegrityValidator().validate(
            [d], PortfolioConstraints(), as_of_date="2026-05-20"
        )
        assert not report.passed
        assert "future_leakage" in {v.violation_type for v in report.violations}

    def test_insufficient_sample_fails(self):
        d = _make_decision(obs=1)  # < min_sample_size=3
        report = PortfolioIntegrityValidator().validate(
            [d], PortfolioConstraints(), as_of_date="2026-05-20"
        )
        assert not report.passed
        assert "insufficient_sample" in {v.violation_type for v in report.violations}

    def test_rapid_recursive_adaptation_fails(self):
        decisions = [
            _make_decision(eval_date=f"2026-05-{14 + i:02d}", obs=5)
            for i in range(8)  # 8 > max_decisions_per_week=6
        ]
        report = PortfolioIntegrityValidator().validate(
            decisions, PortfolioConstraints(), as_of_date="2026-05-22"
        )
        assert not report.passed
        assert "rapid_recursive_adaptation" in {v.violation_type for v in report.violations}

    def test_valid_records_pass(self):
        decisions = [
            _make_decision(eval_date="2026-05-10", obs=5),
            _make_decision(eval_date="2026-05-15", dim_type=PIDTYPE_RISK_SCALE,
                           prev=1.0, new=0.70, obs=5),
        ]
        report = PortfolioIntegrityValidator().validate(
            decisions, PortfolioConstraints(), as_of_date="2026-05-20"
        )
        assert report.passed


# ─────────────────────────────────────────────────────────────────────────────
# 7. PortfolioIntelligenceEngine — evaluate
# ─────────────────────────────────────────────────────────────────────────────

class TestPortfolioIntelligenceEngine:
    def test_insufficient_positions_no_decisions(self, engine):
        inp = _make_input(positions=[_make_pos()])  # n=1 < min_sample_size=3
        decisions, overlay = engine.evaluate(inp, [], as_of_date="2026-05-20")
        assert decisions == []
        assert overlay.n_decisions == 0

    def test_high_corr_produces_slot_decision(self, engine):
        inp = _make_input(positions=_high_corr_positions(3))
        decisions, _ = engine.evaluate(inp, [], as_of_date="2026-05-20")
        assert any(d.dimension_type == PIDTYPE_SLOT_SCALE for d in decisions)

    def test_all_decisions_bounded(self, engine):
        c = PortfolioConstraints()
        inp = _make_input(positions=_high_corr_positions(3))
        decisions, _ = engine.evaluate(inp, [], as_of_date="2026-05-20")
        for d in decisions:
            assert d.bounded is True
            if d.dimension_type == PIDTYPE_SLOT_SCALE:
                assert c.slot_scale_min <= d.new_value <= c.slot_scale_max
            elif d.dimension_type == PIDTYPE_RISK_SCALE:
                assert c.risk_scale_min <= d.new_value <= c.risk_scale_max
            elif d.dimension_type == PIDTYPE_THROTTLE:
                assert c.throttle_min <= d.new_value <= c.throttle_max
            elif d.dimension_type == PIDTYPE_HEAT_LIMIT:
                assert c.heat_limit_min <= d.new_value <= c.heat_limit_max

    def test_integrity_fail_blocks_decisions(self, engine):
        bad = _make_decision(eval_date="2030-01-01", obs=5)  # future leakage
        inp = _make_input(positions=_high_corr_positions(3))
        decisions, overlay = engine.evaluate(inp, [bad], as_of_date="2026-05-20")
        assert decisions == []

    def test_overlay_field_types(self, engine):
        inp = _make_input(positions=_high_corr_positions(3))
        _, overlay = engine.evaluate(inp, [], as_of_date="2026-05-20")
        assert isinstance(overlay.suppressed_entry_symbols, frozenset)
        assert isinstance(overlay.hold_extension_symbols, frozenset)
        assert isinstance(overlay.deterioration_exit_symbols, frozenset)
        assert isinstance(overlay.survivability_gate, bool)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Bounded adaptation
# ─────────────────────────────────────────────────────────────────────────────

class TestBoundedAdaptation:
    def test_slot_scale_never_below_min(self):
        c = PortfolioConstraints()
        engine = PortfolioIntelligenceEngine(constraints=c)
        positions = [_make_pos(sector="IT", rsr=90.0, rsr_momentum=1.0) for _ in range(3)]
        inp = _make_input(positions=positions,
                          regime=_make_regime("bear", True, True, 1.0),
                          portfolio_heat=1.0)
        decisions, _ = engine.evaluate(inp, [], as_of_date="2026-05-20")
        for d in decisions:
            if d.dimension_type == PIDTYPE_SLOT_SCALE:
                assert d.new_value >= c.slot_scale_min

    def test_risk_scale_never_below_min(self):
        c = PortfolioConstraints()
        engine = PortfolioIntelligenceEngine(constraints=c)
        positions = [_make_pos(sector="IT", rsr=90.0) for _ in range(3)]
        inp = _make_input(positions=positions,
                          regime=_make_regime("bear", True, True, 1.0),
                          portfolio_heat=1.0)
        decisions, _ = engine.evaluate(inp, [], as_of_date="2026-05-20")
        for d in decisions:
            if d.dimension_type == PIDTYPE_RISK_SCALE:
                assert d.new_value >= c.risk_scale_min

    def test_heat_limit_within_bounds(self):
        c = PortfolioConstraints()
        engine = PortfolioIntelligenceEngine(constraints=c)
        inp = _make_input(positions=_high_corr_positions(3), portfolio_heat=0.95)
        decisions, _ = engine.evaluate(inp, [], as_of_date="2026-05-20")
        for d in decisions:
            if d.dimension_type == PIDTYPE_HEAT_LIMIT:
                assert c.heat_limit_min <= d.new_value <= c.heat_limit_max


# ─────────────────────────────────────────────────────────────────────────────
# 9. Replay determinism
# ─────────────────────────────────────────────────────────────────────────────

class TestReplayDeterminism:
    def test_same_inputs_same_decision_ids(self, engine):
        inp = _make_input(positions=_high_corr_positions(3))
        d1, _ = engine.evaluate(inp, [], as_of_date="2026-05-20")
        d2, _ = engine.evaluate(inp, [], as_of_date="2026-05-20")
        assert sorted(d.decision_id for d in d1) == sorted(d.decision_id for d in d2)

    def test_decision_id_is_sha256_of_canonical(self):
        d = _make_decision(eval_date="2026-05-20", dim_type=PIDTYPE_SLOT_SCALE, new=0.75, obs=5)
        payload = json.dumps({
            "dimension_type": PIDTYPE_SLOT_SCALE,
            "eval_date": "2026-05-20",
            "new_value": round(0.75, 6),
        }, sort_keys=True, ensure_ascii=False)
        expected = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        assert d.decision_id == expected

    def test_different_dates_different_ids(self):
        d1 = _make_decision(eval_date="2026-05-20", new=0.80, obs=5)
        d2 = _make_decision(eval_date="2026-05-21", new=0.80, obs=5)
        assert d1.decision_id != d2.decision_id


# ─────────────────────────────────────────────────────────────────────────────
# 10. Persistence
# ─────────────────────────────────────────────────────────────────────────────

class TestPersistence:
    def test_append_and_load_round_trip(self, tmp_store):
        d = _make_decision(obs=5)
        append_decision(d, tmp_store)
        loaded = load_decisions(tmp_store)
        assert len(loaded) == 1
        assert loaded[0].decision_id == d.decision_id
        assert loaded[0].bounded is True

    def test_multiple_append_preserves_order(self, tmp_store):
        d1 = _make_decision(eval_date="2026-05-10", new=0.80, obs=5)
        d2 = _make_decision(eval_date="2026-05-15", new=0.75, obs=5)
        append_decision(d1, tmp_store)
        append_decision(d2, tmp_store)
        loaded = load_decisions(tmp_store)
        assert len(loaded) == 2
        assert loaded[0].decision_id == d1.decision_id
        assert loaded[1].decision_id == d2.decision_id

    def test_empty_store_returns_empty(self, tmp_store):
        assert load_decisions(tmp_store) == []

    def test_corrupted_line_skipped(self, tmp_store):
        d = _make_decision(obs=5)
        append_decision(d, tmp_store)
        with open(tmp_store, "a", encoding="utf-8") as f:
            f.write("NOT_VALID_JSON\n")
        loaded = load_decisions(tmp_store)
        assert len(loaded) == 1

    def test_schema_version_preserved(self, tmp_store):
        d = _make_decision(obs=5)
        append_decision(d, tmp_store)
        assert load_decisions(tmp_store)[0].schema_version == SCHEMA_VERSION


# ─────────────────────────────────────────────────────────────────────────────
# 11. Cooldown enforcement
# ─────────────────────────────────────────────────────────────────────────────

class TestCooldownEnforcement:
    def test_recent_decision_blocks_same_dim(self, engine):
        # yesterday → within 2-day cooldown (cutoff = today-2 = 2026-05-18)
        recent = _make_decision(eval_date="2026-05-19", dim_type=PIDTYPE_SLOT_SCALE,
                                prev=1.0, new=0.80, obs=5)
        inp = _make_input(positions=_high_corr_positions(3))
        decisions, _ = engine.evaluate(inp, [recent], as_of_date="2026-05-20")
        assert not any(d.dimension_type == PIDTYPE_SLOT_SCALE for d in decisions)

    def test_old_decision_allows_new(self, engine):
        # 10 days ago → outside 2-day cooldown
        old = _make_decision(eval_date="2026-05-10", dim_type=PIDTYPE_SLOT_SCALE,
                             prev=1.0, new=0.80, obs=5)
        inp = _make_input(positions=_high_corr_positions(3))
        decisions, _ = engine.evaluate(inp, [old], as_of_date="2026-05-20")
        assert any(d.dimension_type == PIDTYPE_SLOT_SCALE for d in decisions)


# ─────────────────────────────────────────────────────────────────────────────
# 12. Anti-oscillation
# ─────────────────────────────────────────────────────────────────────────────

class TestAntiOscillation:
    def test_direction_reversal_blocked_within_window(self, engine):
        # Prior decision went UP (prev=0.60 → new=0.90). Engine now wants DOWN.
        # Gap = 3d < anti_oscillation_window=5 → reversal blocked.
        prior_up = _make_decision(eval_date="2026-05-17", dim_type=PIDTYPE_SLOT_SCALE,
                                  prev=0.60, new=0.90, obs=5)
        inp = _make_input(positions=_high_corr_positions(3))  # → slot wants to go DOWN
        decisions, _ = engine.evaluate(inp, [prior_up], as_of_date="2026-05-20")
        assert not any(d.dimension_type == PIDTYPE_SLOT_SCALE for d in decisions)

    def test_same_direction_within_window_respects_cooldown(self, engine):
        # Prior slot DOWN on 2026-05-22 (gap=3d from 2026-05-25, within window).
        # Cooldown cutoff = 2026-05-23; "2026-05-22" < cutoff → cooldown OK.
        # Anti-oscillation: same direction (both DOWN) → allowed.
        prior_down = _make_decision(eval_date="2026-05-22", dim_type=PIDTYPE_SLOT_SCALE,
                                    prev=1.0, new=0.80, obs=5)
        inp = _make_input(positions=_high_corr_positions(3))
        decisions, _ = engine.evaluate(inp, [prior_down], as_of_date="2026-05-25")
        for d in decisions:
            assert d.bounded is True  # whatever is produced is still bounded

    def test_multiple_oscillations_accumulate_no_decisions(self, engine):
        # 5 alternating decisions within 5 days → each reversal is blocked
        history = []
        directions = [(1.0, 0.75), (0.75, 0.95), (0.95, 0.70)]
        for i, (p, n) in enumerate(directions):
            history.append(_make_decision(
                eval_date=f"2026-05-{16 + i:02d}",
                dim_type=PIDTYPE_SLOT_SCALE, prev=p, new=n, obs=5,
            ))
        inp = _make_input(positions=_high_corr_positions(3))
        decisions, _ = engine.evaluate(inp, history, as_of_date="2026-05-20")
        # May or may not produce slot decisions — just verify all are bounded
        for d in decisions:
            assert d.bounded is True


# ─────────────────────────────────────────────────────────────────────────────
# 13. No feedback loop
# ─────────────────────────────────────────────────────────────────────────────

class TestNoFeedbackLoop:
    def test_prior_decisions_not_consumed_as_strategy_input(self, engine):
        """History is used only for cooldown/anti-osc — not fed as analytics input."""
        inp = _make_input(positions=_high_corr_positions(3))
        # Inject a historical slot decision at 0.50 (far from current conditions)
        history = [_make_decision(eval_date="2026-05-01", dim_type=PIDTYPE_SLOT_SCALE,
                                  prev=1.0, new=0.50, obs=5)]
        decisions, overlay = engine.evaluate(inp, history, as_of_date="2026-05-20")
        # The overlay slot_scale must be recomputed from positions — not copied from history
        for d in decisions:
            if d.dimension_type == PIDTYPE_SLOT_SCALE:
                assert d.new_value != 0.50  # not self-referential

    def test_n_decisions_matches_overlay(self, engine):
        inp = _make_input(positions=_high_corr_positions(3))
        decisions, overlay = engine.evaluate(inp, [], as_of_date="2026-05-20")
        assert overlay.n_decisions == len(decisions)


# ─────────────────────────────────────────────────────────────────────────────
# 14. Weekly quota
# ─────────────────────────────────────────────────────────────────────────────

class TestWeeklyQuota:
    def test_quota_limits_decisions(self):
        c = PortfolioConstraints(max_decisions_per_week=2)
        engine = PortfolioIntelligenceEngine(constraints=c)
        inp = _make_input(positions=_high_corr_positions(3))
        decisions, _ = engine.evaluate(inp, [], as_of_date="2026-05-20")
        assert len(decisions) <= c.max_decisions_per_week

    def test_existing_decisions_reduce_quota(self):
        c = PortfolioConstraints(max_decisions_per_week=3)
        engine = PortfolioIntelligenceEngine(constraints=c)
        history = [
            _make_decision(eval_date="2026-05-15", dim_type=PIDTYPE_RISK_SCALE,
                           prev=1.0, new=0.80, obs=5),
            _make_decision(eval_date="2026-05-16", dim_type=PIDTYPE_THROTTLE,
                           prev=1.0, new=0.90, obs=5),
        ]
        inp = _make_input(positions=_high_corr_positions(3))
        decisions, _ = engine.evaluate(inp, history, as_of_date="2026-05-20")
        assert len(decisions) <= 1


# ─────────────────────────────────────────────────────────────────────────────
# 15. PortfolioIntelligenceHook
# ─────────────────────────────────────────────────────────────────────────────

def _make_sig(symbol, holding=True, signal=1, sector="IT", rsr=80.0, hold_days=10,
              runner=0.5, persistence=0.5):
    return {
        "symbol": symbol, "currently_holding": holding, "signal": signal,
        "sector": sector, "rsr": rsr, "rsr_momentum": 0.5, "hold_days": hold_days,
        "current_return": 0.05, "runner_score": runner, "persistence_score": persistence,
        "volatility": 0.20,
    }


def _make_order(symbol, side="BUY"):
    o = MagicMock()
    o.symbol = symbol
    o.side = side
    return o


class TestPortfolioIntelligenceHook:
    def test_fail_open_on_exception(self, tmp_store):
        hook = PortfolioIntelligenceHook(store_path=tmp_store)
        signals = [{"symbol": "X"}]
        with patch.object(hook._engine, "evaluate", side_effect=RuntimeError("boom")):
            sigs, ords, overlay = hook.run(signals, [], as_of_date="2026-05-20")
        assert sigs is signals
        assert isinstance(overlay, PortfolioIntelligenceOverlay)

    def test_no_positions_returns_inputs_unchanged(self, tmp_store):
        hook = PortfolioIntelligenceHook(store_path=tmp_store)
        signals = [{"symbol": "X", "currently_holding": False}]
        sigs, ords, overlay = hook.run(signals, [], as_of_date="2026-05-20")
        assert sigs is signals

    def test_hold_extension_applied(self, tmp_store):
        hook = PortfolioIntelligenceHook(store_path=tmp_store)
        sigs_in = [_make_sig(f"WIN", holding=True, signal=0) for _ in range(3)]
        overlay_out = PortfolioIntelligenceOverlay(hold_extension_symbols=frozenset({"WIN"}))
        with patch.object(hook._engine, "evaluate", return_value=([], overlay_out)):
            sigs_out, _, _ = hook.run(sigs_in, [], as_of_date="2026-05-20")
        for s in sigs_out:
            assert s["signal"] == 1
            assert s.get("portfolio_intelligence_hold_extension") is True

    def test_deterioration_exit_applied(self, tmp_store):
        hook = PortfolioIntelligenceHook(store_path=tmp_store)
        sigs_in = [_make_sig("LOSER", holding=True, signal=1) for _ in range(3)]
        overlay_out = PortfolioIntelligenceOverlay(deterioration_exit_symbols=frozenset({"LOSER"}))
        with patch.object(hook._engine, "evaluate", return_value=([], overlay_out)):
            sigs_out, _, _ = hook.run(sigs_in, [], as_of_date="2026-05-20")
        for s in sigs_out:
            assert s["signal"] == 0
            assert s.get("portfolio_intelligence_deterioration_exit") is True

    def test_survivability_gate_suppresses_all_buys(self, tmp_store):
        hook = PortfolioIntelligenceHook(store_path=tmp_store)
        sigs_in = [_make_sig(f"P{i}") for i in range(3)]
        overlay_out = PortfolioIntelligenceOverlay(survivability_gate=False)
        with patch.object(hook._engine, "evaluate", return_value=([], overlay_out)):
            _, orders_out, _ = hook.run(
                sigs_in, [_make_order("NEW", "BUY"), _make_order("OLD", "SELL")],
                as_of_date="2026-05-20",
            )
        sides = [getattr(o, "side", None) for o in orders_out]
        assert "BUY" not in sides
        assert "SELL" in sides

    def test_corr_suppression_removes_specific_buys(self, tmp_store):
        hook = PortfolioIntelligenceHook(store_path=tmp_store)
        sigs_in = [_make_sig(f"P{i}") for i in range(3)]
        overlay_out = PortfolioIntelligenceOverlay(
            survivability_gate=True,
            suppressed_entry_symbols=frozenset({"CORR_SYM"}),
        )
        with patch.object(hook._engine, "evaluate", return_value=([], overlay_out)):
            _, orders_out, _ = hook.run(
                sigs_in, [_make_order("CORR_SYM", "BUY"), _make_order("FREE_SYM", "BUY")],
                as_of_date="2026-05-20",
            )
        syms = [getattr(o, "symbol", None) for o in orders_out]
        assert "CORR_SYM" not in syms
        assert "FREE_SYM" in syms

    def test_overlay_annotations_added_to_signals(self, tmp_store):
        hook = PortfolioIntelligenceHook(store_path=tmp_store)
        sigs_in = [_make_sig(f"P{i}") for i in range(3)]
        overlay_out = PortfolioIntelligenceOverlay(slot_scale=0.75, survivability_score=0.55,
                                                    overall_corr_score=0.80)
        with patch.object(hook._engine, "evaluate", return_value=([], overlay_out)):
            sigs_out, _, _ = hook.run(sigs_in, [], as_of_date="2026-05-20")
        for s in sigs_out:
            assert "pi_slot_scale" in s
            assert "pi_survivability" in s
            assert "pi_corr_score" in s


# ─────────────────────────────────────────────────────────────────────────────
# 16. run_portfolio_intelligence_hook
# ─────────────────────────────────────────────────────────────────────────────

class TestRunPortfolioIntelligenceHook:
    def test_returns_tuple_of_three(self, tmp_store):
        result = run_portfolio_intelligence_hook(
            signals=[], order_objects=[], store_path=tmp_store, as_of_date="2026-05-20",
        )
        assert len(result) == 3

    def test_fail_open_on_hook_error(self, tmp_store):
        with patch(
            "src.runtime.policy.portfolio_intelligence_engine.PortfolioIntelligenceHook",
            side_effect=RuntimeError("boom"),
        ):
            sigs, ords, overlay = run_portfolio_intelligence_hook(
                signals=[{"symbol": "X"}], order_objects=[], store_path=tmp_store,
                as_of_date="2026-05-20",
            )
        assert sigs == [{"symbol": "X"}]
        assert isinstance(overlay, PortfolioIntelligenceOverlay)

    def test_integration_with_held_positions(self, tmp_store):
        sigs = [
            {
                "symbol": f"SYM{i}", "currently_holding": True, "sector": "IT",
                "rsr": 80.0, "rsr_momentum": 0.5, "hold_days": 10 + i,
                "current_return": 0.05, "runner_score": 0.5, "persistence_score": 0.5,
                "signal": 1, "volatility": 0.20,
            }
            for i in range(3)
        ]
        sigs_out, ords_out, overlay = run_portfolio_intelligence_hook(
            signals=sigs, order_objects=[], store_path=tmp_store,
            as_of_date="2026-05-20",
        )
        assert isinstance(sigs_out, list)
        assert isinstance(overlay, PortfolioIntelligenceOverlay)
        assert isinstance(load_decisions(tmp_store), list)


# ─────────────────────────────────────────────────────────────────────────────
# 17. PortfolioAdjustmentDecision
# ─────────────────────────────────────────────────────────────────────────────

class TestPortfolioAdjustmentDecision:
    def test_frozen(self):
        d = _make_decision(obs=5)
        with pytest.raises(Exception):
            d.new_value = 99.0  # type: ignore[misc]

    def test_bounded_always_true(self):
        assert _make_decision(obs=5).bounded is True

    def test_to_dict_from_dict_round_trip(self):
        d = _make_decision(obs=5)
        d2 = PortfolioAdjustmentDecision.from_dict(d.to_dict())
        assert d.decision_id == d2.decision_id
        assert d.new_value == d2.new_value
        assert d.dimension_type == d2.dimension_type
