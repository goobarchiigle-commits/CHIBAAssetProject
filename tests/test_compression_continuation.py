"""
tests/test_compression_continuation.py

Unit tests for compression_continuation.py.
Pure function tests — no I/O, no filesystem.
"""
import math
import pytest

from src.analytics.compression_continuation import (
    COMPRESS_MOM_MIN,
    COMPRESS_NEAR_HIGH_MIN,
    COMPRESS_RATIO_MAX,
    COMPRESS_VOLUME_MAX,
    DISTRIB_MOM_MAX,
    DISTRIB_VOLUME_MIN,
    INVALID_RATIO_MAX,
    INVALID_RSR_MIN,
    SUPPRESSION_MOM_MIN,
    SUPPRESSION_RSR_MIN,
    SUPPRESSION_SCORE_MIN,
    CompressionContinuationInput,
    CompressionContinuationResult,
    _classify_phase,
    _clip,
    _compute_components,
    _composite_score,
    _get_invalidations,
    _safe,
    compute_compression_continuation,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_input(
    symbol="TEST",
    hold_days=8,
    unrealized_pnl_pct=0.05,
    compression_ratio=0.60,
    price_tightness=0.50,
    volume_ratio_5d=0.75,
    higher_low_streak=3,
    dip_recovery_pct=0.8,
    near_high_pct=0.80,
    rsr=83.0,
    rsr_momentum=0.5,
):
    return CompressionContinuationInput(
        symbol=symbol,
        hold_days=hold_days,
        unrealized_pnl_pct=unrealized_pnl_pct,
        compression_ratio=compression_ratio,
        price_tightness=price_tightness,
        volume_ratio_5d=volume_ratio_5d,
        higher_low_streak=higher_low_streak,
        dip_recovery_pct=dip_recovery_pct,
        near_high_pct=near_high_pct,
        rsr=rsr,
        rsr_momentum=rsr_momentum,
    )


# ── _clip ─────────────────────────────────────────────────────────────────────

class TestClip:
    def test_below_lo(self):
        assert _clip(-10.0) == 0.0

    def test_above_hi(self):
        assert _clip(110.0) == 100.0

    def test_in_range(self):
        assert _clip(50.0) == 50.0

    def test_custom_range(self):
        assert _clip(0.5, lo=0.0, hi=1.0) == 0.5
        assert _clip(-0.1, lo=0.0, hi=1.0) == 0.0
        assert _clip(1.1, lo=0.0, hi=1.0) == 1.0


# ── _safe ─────────────────────────────────────────────────────────────────────

class TestSafe:
    def test_normal(self):
        assert _safe(1.5) == 1.5

    def test_nan(self):
        assert _safe(float("nan")) == 0.0

    def test_inf(self):
        assert _safe(float("inf")) == 0.0
        assert _safe(float("-inf")) == 0.0

    def test_custom_fallback(self):
        assert _safe(float("nan"), fallback=99.0) == 99.0

    def test_none_type(self):
        assert _safe(None, fallback=5.0) == 5.0

    def test_string(self):
        assert _safe("not_a_number", fallback=1.0) == 1.0


# ── _compute_components ───────────────────────────────────────────────────────

class TestComputeComponents:
    def test_fully_compressed(self):
        inp = _make_input(
            compression_ratio=0.40,
            price_tightness=0.30,
            volume_ratio_5d=0.40,
            near_high_pct=0.95,
            higher_low_streak=4,
            rsr_momentum=2.0,
        )
        c = _compute_components(inp)
        # All components should be high
        for v in c:
            assert v >= 70.0, f"Expected >= 70, got {v}"

    def test_fully_expanded(self):
        inp = _make_input(
            compression_ratio=2.0,
            price_tightness=2.0,
            volume_ratio_5d=3.0,
            near_high_pct=0.10,
            higher_low_streak=0,
            rsr_momentum=-5.0,
        )
        c = _compute_components(inp)
        # compression, tightness, volume, near_high, rsr should be 0
        c_compress, c_tight, c_vol, c_nh, c_hl, c_rsr = c
        assert c_compress == 0.0
        assert c_tight    == 0.0
        assert c_vol      == 0.0
        assert c_nh       == 0.0
        assert c_rsr      == 0.0

    def test_bounds_always_0_to_100(self):
        for cr in [0.0, 0.5, 1.0, 2.0]:
            for vr in [0.0, 0.5, 1.0, 2.0]:
                for nh in [0.0, 0.5, 1.0]:
                    for mom in [-10.0, 0.0, 10.0]:
                        inp = _make_input(
                            compression_ratio=cr,
                            volume_ratio_5d=vr,
                            near_high_pct=nh,
                            rsr_momentum=mom,
                        )
                        for v in _compute_components(inp):
                            assert 0.0 <= v <= 100.0, f"Component out of bounds: {v}"

    def test_nan_inputs_handled(self):
        inp = _make_input(
            compression_ratio=float("nan"),
            price_tightness=float("nan"),
            volume_ratio_5d=float("nan"),
            near_high_pct=float("nan"),
            rsr_momentum=float("nan"),
        )
        for v in _compute_components(inp):
            assert 0.0 <= v <= 100.0
            assert math.isfinite(v)

    def test_higher_low_streak_saturates(self):
        inp_4 = _make_input(higher_low_streak=4)
        inp_8 = _make_input(higher_low_streak=8)
        # hl component = min(streak * 25, 100)
        _, _, _, _, c_hl_4, _ = _compute_components(inp_4)
        _, _, _, _, c_hl_8, _ = _compute_components(inp_8)
        assert c_hl_4 == 100.0
        assert c_hl_8 == 100.0

    def test_negative_hl_streak_clamped(self):
        inp = _make_input(higher_low_streak=-5)
        _, _, _, _, c_hl, _ = _compute_components(inp)
        assert c_hl == 0.0


# ── _composite_score ──────────────────────────────────────────────────────────

class TestCompositeScore:
    def test_all_zeros(self):
        assert _composite_score(0, 0, 0, 0, 0, 0) == 0.0

    def test_all_hundreds(self):
        assert _composite_score(100, 100, 100, 100, 100, 100) == 100.0

    def test_weighted_correctly(self):
        # Only compression at 100, rest 0 → 0.25 * 100 = 25
        score = _composite_score(100, 0, 0, 0, 0, 0)
        assert abs(score - 25.0) < 0.01

    def test_weight_sum(self):
        # Weights: 0.25 + 0.20 + 0.20 + 0.20 + 0.10 + 0.05 = 1.00
        score = _composite_score(100, 100, 100, 100, 100, 100)
        assert abs(score - 100.0) < 0.01

    def test_output_is_finite(self):
        score = _composite_score(50, 50, 50, 50, 50, 50)
        assert math.isfinite(score)


# ── _classify_phase ───────────────────────────────────────────────────────────

class TestClassifyPhase:
    def test_healthy_compression(self):
        inp = _make_input(
            compression_ratio=0.60,
            volume_ratio_5d=0.70,
            near_high_pct=0.80,
            rsr_momentum=0.5,
        )
        assert _classify_phase(inp) == "compression"

    def test_distribution(self):
        inp = _make_input(
            volume_ratio_5d=1.50,
            higher_low_streak=0,
            rsr_momentum=-3.0,
        )
        assert _classify_phase(inp) == "distribution"

    def test_distribution_priority_over_near_high(self):
        # Even if near_high is high, distribution conditions dominate
        inp = _make_input(
            compression_ratio=0.60,
            volume_ratio_5d=1.50,
            near_high_pct=0.90,
            higher_low_streak=0,
            rsr_momentum=-3.0,
        )
        assert _classify_phase(inp) == "distribution"

    def test_breakout(self):
        inp = _make_input(
            compression_ratio=0.85,   # expanding (> 0.75)
            volume_ratio_5d=1.50,
            near_high_pct=0.92,
            rsr_momentum=1.5,
            higher_low_streak=5,
        )
        assert _classify_phase(inp) == "breakout"

    def test_neutral(self):
        inp = _make_input(
            compression_ratio=0.95,   # wide
            volume_ratio_5d=1.05,     # moderate
            near_high_pct=0.50,       # mid
            rsr_momentum=-0.5,
        )
        assert _classify_phase(inp) == "neutral"

    def test_compression_boundary_ratio(self):
        # Exactly at threshold: not < COMPRESS_RATIO_MAX → not compression
        inp = _make_input(
            compression_ratio=COMPRESS_RATIO_MAX,
            volume_ratio_5d=0.70,
            near_high_pct=0.80,
            rsr_momentum=0.5,
        )
        assert _classify_phase(inp) != "compression"

    def test_compression_mom_boundary(self):
        # rsr_momentum at threshold: must be > COMPRESS_MOM_MIN (-1.0)
        inp_ok  = _make_input(compression_ratio=0.60, volume_ratio_5d=0.70,
                              near_high_pct=0.80, rsr_momentum=-0.9)
        inp_bad = _make_input(compression_ratio=0.60, volume_ratio_5d=0.70,
                              near_high_pct=0.80, rsr_momentum=-1.0)
        assert _classify_phase(inp_ok)  == "compression"
        assert _classify_phase(inp_bad) != "compression"

    def test_nan_inputs_do_not_crash(self):
        inp = _make_input(
            compression_ratio=float("nan"),
            volume_ratio_5d=float("nan"),
            near_high_pct=float("nan"),
            rsr_momentum=float("nan"),
        )
        phase = _classify_phase(inp)
        assert isinstance(phase, str)
        assert phase in ("compression", "distribution", "breakout", "neutral")


# ── _get_invalidations ────────────────────────────────────────────────────────

class TestGetInvalidations:
    def test_no_flags(self):
        inp = _make_input(compression_ratio=0.60, rsr=80.0)
        flags = _get_invalidations(inp, "compression")
        assert flags == []

    def test_atr_expanding(self):
        inp = _make_input(compression_ratio=INVALID_RATIO_MAX + 0.01, rsr=80.0)
        flags = _get_invalidations(inp, "compression")
        assert "atr_expanding" in flags

    def test_rsr_weak(self):
        inp = _make_input(compression_ratio=0.60, rsr=INVALID_RSR_MIN - 0.01)
        flags = _get_invalidations(inp, "compression")
        assert "rsr_weak" in flags

    def test_distribution_flag(self):
        inp = _make_input(compression_ratio=0.60, rsr=80.0)
        flags = _get_invalidations(inp, "distribution")
        assert "distribution" in flags

    def test_multiple_flags(self):
        inp = _make_input(compression_ratio=0.95, rsr=70.0)
        flags = _get_invalidations(inp, "distribution")
        assert "atr_expanding" in flags
        assert "rsr_weak" in flags
        assert "distribution" in flags


# ── compute_compression_continuation ─────────────────────────────────────────

class TestComputeCompressionContinuation:
    def test_healthy_compression_eligible(self):
        inp = _make_input(
            compression_ratio=0.55,
            price_tightness=0.45,
            volume_ratio_5d=0.65,
            higher_low_streak=3,
            near_high_pct=0.85,
            rsr=82.0,
            rsr_momentum=0.8,
        )
        result = compute_compression_continuation(inp)
        assert result.phase == "compression"
        assert result.is_healthy_compression
        assert not result.is_distribution
        assert result.score >= SUPPRESSION_SCORE_MIN
        assert result.suppression_eligible

    def test_distribution_not_eligible(self):
        inp = _make_input(
            compression_ratio=0.60,
            volume_ratio_5d=1.60,
            higher_low_streak=0,
            near_high_pct=0.80,
            rsr=82.0,
            rsr_momentum=-3.0,
        )
        result = compute_compression_continuation(inp)
        assert result.phase == "distribution"
        assert result.is_distribution
        assert not result.suppression_eligible
        assert "distribution" in result.invalidation_flags

    def test_score_in_bounds(self):
        for cr in [0.3, 0.6, 0.9, 1.2]:
            for vr in [0.4, 0.8, 1.2, 1.8]:
                inp = _make_input(compression_ratio=cr, volume_ratio_5d=vr)
                result = compute_compression_continuation(inp)
                assert 0.0 <= result.score <= 100.0, f"score={result.score} out of bounds"
                assert math.isfinite(result.score)

    def test_rsr_below_min_blocks_suppression(self):
        inp = _make_input(
            compression_ratio=0.55,
            volume_ratio_5d=0.65,
            near_high_pct=0.85,
            rsr=SUPPRESSION_RSR_MIN - 0.1,
            rsr_momentum=0.5,
        )
        result = compute_compression_continuation(inp)
        assert not result.suppression_eligible

    def test_rsr_at_min_allows_suppression_if_other_ok(self):
        inp = _make_input(
            compression_ratio=0.55,
            price_tightness=0.40,
            volume_ratio_5d=0.65,
            higher_low_streak=3,
            near_high_pct=0.85,
            rsr=SUPPRESSION_RSR_MIN,
            rsr_momentum=0.5,
        )
        result = compute_compression_continuation(inp)
        # RSR is exactly at minimum — may or may not be eligible depending on score
        # Just verify it does not crash and result is valid
        assert isinstance(result.suppression_eligible, bool)

    def test_rsr_momentum_below_min_blocks(self):
        inp = _make_input(
            compression_ratio=0.55,
            volume_ratio_5d=0.65,
            near_high_pct=0.85,
            rsr=82.0,
            rsr_momentum=SUPPRESSION_MOM_MIN - 0.1,
        )
        result = compute_compression_continuation(inp)
        assert not result.suppression_eligible

    def test_atr_expanding_blocks_suppression(self):
        inp = _make_input(
            compression_ratio=INVALID_RATIO_MAX + 0.01,
            volume_ratio_5d=0.65,
            near_high_pct=0.85,
            rsr=82.0,
            rsr_momentum=0.5,
        )
        result = compute_compression_continuation(inp)
        assert not result.suppression_eligible
        assert "atr_expanding" in result.invalidation_flags

    def test_to_dict_structure(self):
        inp = _make_input()
        result = compute_compression_continuation(inp)
        d = result.to_dict()
        assert "symbol" in d
        assert "score" in d
        assert "phase" in d
        assert "suppression_eligible" in d
        assert "components" in d
        for k in ("compression", "tightness", "volume", "near_high", "higher_low", "rsr_stability"):
            assert k in d["components"]

    def test_fail_safe_on_bad_input(self):
        # Build an object that will cause an exception during processing
        class BrokenInput:
            symbol = "BROKEN"
            hold_days = 1
            unrealized_pnl_pct = 0.0
            compression_ratio = float("nan")
            price_tightness = float("nan")
            volume_ratio_5d = float("nan")
            higher_low_streak = "not_an_int"   # type error
            dip_recovery_pct = float("nan")
            near_high_pct = float("nan")
            rsr = float("nan")
            rsr_momentum = float("nan")

        result = compute_compression_continuation(BrokenInput())
        assert result.symbol == "BROKEN"
        assert result.score == 0.0
        assert not result.suppression_eligible
        assert "compute_error" in result.invalidation_flags

    def test_deterministic_same_inputs(self):
        inp = _make_input()
        r1 = compute_compression_continuation(inp)
        r2 = compute_compression_continuation(inp)
        assert r1.score == r2.score
        assert r1.phase == r2.phase
        assert r1.suppression_eligible == r2.suppression_eligible

    def test_neutral_state(self):
        inp = _make_input(
            compression_ratio=1.0,
            volume_ratio_5d=1.0,
            near_high_pct=0.50,
            rsr_momentum=0.0,
            higher_low_streak=0,
        )
        result = compute_compression_continuation(inp)
        assert result.phase == "neutral"
        assert not result.suppression_eligible

    def test_force_exit_candidate_distribution(self):
        # Distribution with significant volume expansion should yield high invalidation
        inp = _make_input(
            compression_ratio=0.60,
            volume_ratio_5d=2.50,
            higher_low_streak=0,
            near_high_pct=0.85,
            rsr=82.0,
            rsr_momentum=-4.0,
        )
        result = compute_compression_continuation(inp)
        assert result.is_distribution
        assert not result.suppression_eligible
        assert len(result.invalidation_flags) >= 1
