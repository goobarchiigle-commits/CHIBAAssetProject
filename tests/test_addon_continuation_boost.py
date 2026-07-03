"""
tests/test_addon_continuation_boost.py — Tests for continuation breakout boost
in winner_confirmation.py.

Covers:
  - apply_continuation_breakout_boost: all gate conditions
  - WinnerConfirmation new fields: continuation_breakout, continuation_boost_applied
  - Boost semantics: cap at 1.0, multiplier echoed, ordering only
  - Fail-safe: no exception propagation
"""
from __future__ import annotations

import pytest
from dataclasses import replace

from src.addon.winner_confirmation import (
    WinnerConfirmation,
    apply_continuation_breakout_boost,
    CONTINUATION_BREAKOUT_BOOST,
    CONTINUATION_BREAKOUT_MIN_COMPRESSION,
    CONTINUATION_BREAKOUT_MIN_VOLUME,
    CONTINUATION_BREAKOUT_MIN_RSR_MOM,
    check_winner,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_confirmed(symbol: str = "1234.T", score: float = 0.50) -> WinnerConfirmation:
    return WinnerConfirmation(
        symbol=symbol,
        confirmed=True,
        confidence_score=score,
        unrealized_pnl_pct=0.05,
        hold_days=7,
        rsr_rank=1,
        rsr=85.0,
        trend_quality=0.5,
        deterioration_score=0.1,
        regime_bear=False,
        atr_ratio=1.0,
    )


def _make_unconfirmed(symbol: str = "1234.T") -> WinnerConfirmation:
    return WinnerConfirmation(
        symbol=symbol,
        confirmed=False,
        confidence_score=0.0,
    )


# ── Default field values ───────────────────────────────────────────────────────

class TestNewFields:
    def test_continuation_breakout_default_false(self):
        conf = _make_confirmed()
        assert conf.continuation_breakout is False

    def test_continuation_boost_applied_default_one(self):
        conf = _make_confirmed()
        assert conf.continuation_boost_applied == 1.0

    def test_unconfirmed_defaults(self):
        conf = _make_unconfirmed()
        assert conf.continuation_breakout is False
        assert conf.continuation_boost_applied == 1.0


# ── Gate: conf.confirmed must be True ─────────────────────────────────────────

class TestBoostRequiresConfirmed:
    def test_no_boost_when_not_confirmed(self):
        conf = _make_unconfirmed()
        out, mult = apply_continuation_breakout_boost(
            conf,
            continuation_breakout=True,
            compression_score=80.0,
            volume_ratio_5d=1.5,
            rsr_momentum=0.1,
        )
        assert out is conf
        assert mult == 1.0
        assert out.continuation_breakout is False
        assert out.continuation_boost_applied == 1.0

    def test_no_boost_confirmed_false_with_all_conditions_met(self):
        conf = replace(_make_confirmed(), confirmed=False, confidence_score=0.0)
        out, mult = apply_continuation_breakout_boost(
            conf, True, 80.0, 1.5, 0.5
        )
        assert out is conf
        assert mult == 1.0


# ── Gate: continuation_breakout flag ──────────────────────────────────────────

class TestBoostRequiresContinuationBreakout:
    def test_no_boost_when_flag_false(self):
        conf = _make_confirmed()
        out, mult = apply_continuation_breakout_boost(
            conf,
            continuation_breakout=False,
            compression_score=80.0,
            volume_ratio_5d=1.5,
            rsr_momentum=0.1,
        )
        assert out is conf
        assert mult == 1.0


# ── Gate: compression_score ────────────────────────────────────────────────────

class TestBoostCompressionGate:
    def test_no_boost_when_compression_below_min(self):
        conf = _make_confirmed()
        out, mult = apply_continuation_breakout_boost(
            conf, True,
            compression_score=CONTINUATION_BREAKOUT_MIN_COMPRESSION - 0.1,
            volume_ratio_5d=1.5,
            rsr_momentum=0.1,
        )
        assert out is conf
        assert mult == 1.0

    def test_boost_when_compression_at_min(self):
        conf = _make_confirmed()
        out, mult = apply_continuation_breakout_boost(
            conf, True,
            compression_score=CONTINUATION_BREAKOUT_MIN_COMPRESSION,
            volume_ratio_5d=1.5,
            rsr_momentum=0.1,
        )
        assert out is not conf
        assert mult == CONTINUATION_BREAKOUT_BOOST

    def test_boost_when_compression_above_min(self):
        conf = _make_confirmed()
        out, mult = apply_continuation_breakout_boost(
            conf, True,
            compression_score=95.0,
            volume_ratio_5d=1.5,
            rsr_momentum=0.1,
        )
        assert mult == CONTINUATION_BREAKOUT_BOOST


# ── Gate: volume_ratio_5d ─────────────────────────────────────────────────────

class TestBoostVolumeGate:
    def test_no_boost_when_volume_below_min(self):
        conf = _make_confirmed()
        out, mult = apply_continuation_breakout_boost(
            conf, True,
            compression_score=80.0,
            volume_ratio_5d=CONTINUATION_BREAKOUT_MIN_VOLUME - 0.01,
            rsr_momentum=0.1,
        )
        assert out is conf
        assert mult == 1.0

    def test_boost_when_volume_at_min(self):
        conf = _make_confirmed()
        out, mult = apply_continuation_breakout_boost(
            conf, True,
            compression_score=80.0,
            volume_ratio_5d=CONTINUATION_BREAKOUT_MIN_VOLUME,
            rsr_momentum=0.1,
        )
        assert mult == CONTINUATION_BREAKOUT_BOOST

    def test_no_boost_when_volume_is_zero(self):
        conf = _make_confirmed()
        out, mult = apply_continuation_breakout_boost(
            conf, True,
            compression_score=80.0,
            volume_ratio_5d=0.0,
            rsr_momentum=0.1,
        )
        assert mult == 1.0


# ── Gate: rsr_momentum ────────────────────────────────────────────────────────

class TestBoostRSRMomentumGate:
    def test_no_boost_when_momentum_negative(self):
        conf = _make_confirmed()
        out, mult = apply_continuation_breakout_boost(
            conf, True,
            compression_score=80.0,
            volume_ratio_5d=1.5,
            rsr_momentum=-0.01,
        )
        assert out is conf
        assert mult == 1.0

    def test_boost_when_momentum_zero(self):
        conf = _make_confirmed()
        out, mult = apply_continuation_breakout_boost(
            conf, True,
            compression_score=80.0,
            volume_ratio_5d=1.5,
            rsr_momentum=CONTINUATION_BREAKOUT_MIN_RSR_MOM,
        )
        assert mult == CONTINUATION_BREAKOUT_BOOST

    def test_boost_when_momentum_positive(self):
        conf = _make_confirmed()
        out, mult = apply_continuation_breakout_boost(
            conf, True,
            compression_score=80.0,
            volume_ratio_5d=1.5,
            rsr_momentum=0.5,
        )
        assert mult == CONTINUATION_BREAKOUT_BOOST


# ── Boost semantics ───────────────────────────────────────────────────────────

class TestBoostSemantics:
    def test_score_multiplied_by_boost(self):
        conf = _make_confirmed(score=0.50)
        out, mult = apply_continuation_breakout_boost(
            conf, True, 80.0, 1.5, 0.1
        )
        expected = round(0.50 * CONTINUATION_BREAKOUT_BOOST, 4)
        assert out.confidence_score == expected
        assert mult == CONTINUATION_BREAKOUT_BOOST

    def test_score_capped_at_one(self):
        conf = _make_confirmed(score=0.99)
        out, _ = apply_continuation_breakout_boost(conf, True, 80.0, 1.5, 0.1)
        assert out.confidence_score == 1.0

    def test_score_exactly_one_stays_one(self):
        conf = _make_confirmed(score=1.0)
        out, _ = apply_continuation_breakout_boost(conf, True, 80.0, 1.5, 0.1)
        assert out.confidence_score == 1.0

    def test_continuation_breakout_flag_set_true(self):
        conf = _make_confirmed()
        out, _ = apply_continuation_breakout_boost(conf, True, 80.0, 1.5, 0.1)
        assert out.continuation_breakout is True

    def test_continuation_boost_applied_set(self):
        conf = _make_confirmed()
        out, _ = apply_continuation_breakout_boost(conf, True, 80.0, 1.5, 0.1)
        assert out.continuation_boost_applied == CONTINUATION_BREAKOUT_BOOST

    def test_original_conf_not_mutated(self):
        conf = _make_confirmed(score=0.50)
        out, _ = apply_continuation_breakout_boost(conf, True, 80.0, 1.5, 0.1)
        assert conf.confidence_score == 0.50
        assert conf.continuation_breakout is False
        assert conf.continuation_boost_applied == 1.0

    def test_returns_new_object_on_boost(self):
        conf = _make_confirmed()
        out, _ = apply_continuation_breakout_boost(conf, True, 80.0, 1.5, 0.1)
        assert out is not conf

    def test_returns_same_object_when_no_boost(self):
        conf = _make_confirmed()
        out, _ = apply_continuation_breakout_boost(conf, False, 80.0, 1.5, 0.1)
        assert out is conf

    def test_multiplier_one_when_no_boost(self):
        conf = _make_confirmed()
        _, mult = apply_continuation_breakout_boost(conf, False, 80.0, 1.5, 0.1)
        assert mult == 1.0

    def test_multiplier_equals_boost_constant_on_boost(self):
        conf = _make_confirmed()
        _, mult = apply_continuation_breakout_boost(conf, True, 80.0, 1.5, 0.1)
        assert mult == CONTINUATION_BREAKOUT_BOOST

    def test_other_fields_unchanged_on_boost(self):
        conf = _make_confirmed(symbol="6920.T", score=0.40)
        out, _ = apply_continuation_breakout_boost(conf, True, 80.0, 1.5, 0.1)
        assert out.symbol == "6920.T"
        assert out.confirmed is True
        assert out.rsr == conf.rsr
        assert out.hold_days == conf.hold_days


# ── Fail-safe ─────────────────────────────────────────────────────────────────

class TestFailSafe:
    def test_returns_original_on_exception(self):
        conf = _make_confirmed()
        # Pass non-numeric types that will cause internal float() to raise
        out, mult = apply_continuation_breakout_boost(
            conf,
            continuation_breakout=True,
            compression_score="bad",   # type: ignore
            volume_ratio_5d=1.5,
            rsr_momentum=0.1,
        )
        # Should not raise; fail-safe returns original
        assert mult == 1.0

    def test_no_raise_on_none_inputs(self):
        conf = _make_confirmed()
        try:
            out, mult = apply_continuation_breakout_boost(
                conf, True, None, None, None  # type: ignore
            )
        except Exception as exc:
            pytest.fail(f"apply_continuation_breakout_boost raised: {exc}")


# ── Integration: check_winner → boost pipeline ────────────────────────────────

class TestCheckWinnerBoostPipeline:
    def test_confirmed_winner_gets_boosted(self):
        conf = check_winner(
            symbol="1234.T",
            unrealized_pnl_pct=0.06,
            hold_days=7,
            rsr_rank=1,
            rsr=85.0,
            trend_quality=0.5,
            deterioration_score=0.1,
            regime_bear=False,
            atr_ratio=1.0,
            current_price=1100.0,
            entry_price=1000.0,
        )
        assert conf.confirmed
        boosted, mult = apply_continuation_breakout_boost(
            conf, True, 80.0, 1.5, 0.3
        )
        assert boosted.confidence_score >= conf.confidence_score
        assert mult == CONTINUATION_BREAKOUT_BOOST
        assert boosted.continuation_breakout is True

    def test_unconfirmed_winner_not_boosted(self):
        conf = check_winner(
            symbol="1234.T",
            unrealized_pnl_pct=0.01,  # below MIN_PROFIT_THRESHOLD
            hold_days=7,
            rsr_rank=1,
            rsr=85.0,
            trend_quality=0.5,
            deterioration_score=0.1,
            regime_bear=False,
        )
        assert not conf.confirmed
        out, mult = apply_continuation_breakout_boost(conf, True, 80.0, 1.5, 0.3)
        assert out is conf
        assert mult == 1.0

    def test_boost_ordering_effect(self):
        # Two confirmed winners; continuation_breakout one should rank higher after boost
        conf_a = _make_confirmed(symbol="A", score=0.55)
        conf_b = _make_confirmed(symbol="B", score=0.60)
        # Boost A
        boosted_a, _ = apply_continuation_breakout_boost(conf_a, True, 80.0, 1.5, 0.1)
        boosted_b = conf_b  # no boost
        ranking = sorted(
            [boosted_a, boosted_b],
            key=lambda c: c.confidence_score,
            reverse=True,
        )
        # boosted_a = 0.55 * 1.15 = 0.6325 > 0.60
        assert ranking[0].symbol == "A"
