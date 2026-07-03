"""tests/analytics/test_exit_intelligence/test_deterioration.py"""
import pytest

from src.analytics.exit_intelligence.deterioration import (
    ConvexityRetentionAnalyzer,
    MomentumDecayAnalyzer,
    PeakDecayVelocityAnalyzer,
    PersistenceExtensionLayer,
    StructuralFailureDetector,
)
from src.analytics.exit_intelligence.observation import ExitPathSnapshot


def _make_snap(
    day, symbol="1234", entry="2026-01-01",
    close=1050.0, peak=1060.0, rs=80.0,
    momentum=0.03, momentum_prev=0.02,
    volume_ratio=1.2, vol_compression=0.5,
    upper_shadow=0.3, exit_signals=0,
    breadth=0.6, trend_pers=0.7, regime="bull",
    atr=0.02,
):
    return ExitPathSnapshot(
        snapshot_id=ExitPathSnapshot.make_snapshot_id(symbol, entry, f"2026-01-{day:02d}"),
        symbol=symbol,
        entry_date=entry,
        snapshot_date=f"2026-01-{day:02d}",
        day_number=day,
        close_price=close,
        entry_price=1000.0,
        unrealized_pnl_fraction=(close - 1000.0) / 1000.0,
        unrealized_pnl_jpy=(close - 1000.0) * 100,
        peak_price_so_far=peak,
        distance_from_peak_pct=(peak - close) / peak if peak > 0 else 0.0,
        rs_rank=rs,
        atr_pct=atr,
        breadth_score=breadth,
        regime=regime,
        trend_persistence=trend_pers,
        momentum=momentum,
        momentum_prev=momentum_prev,
        volume_ratio=volume_ratio,
        volatility_compression=vol_compression,
        upper_shadow_ratio=upper_shadow,
        exit_signal_count=exit_signals,
        deterioration_velocity=0.002,
    )


def _healthy_window(n=5):
    return [
        _make_snap(i, close=1000.0 + i * 10, peak=1000.0 + i * 10,
                   rs=82.0, momentum=0.02 + i * 0.002, volume_ratio=1.2,
                   vol_compression=0.5, upper_shadow=0.2)
        for i in range(1, n + 1)
    ]


def _deteriorating_window(n=5):
    return [
        _make_snap(i, close=1050.0 - i * 15, peak=1080.0,
                   rs=70.0 - i * 3, momentum=-0.01 * i,
                   volume_ratio=0.8, vol_compression=0.3,
                   upper_shadow=0.7, exit_signals=1)
        for i in range(1, n + 1)
    ]


class TestMomentumDecayAnalyzer:
    def setup_method(self):
        self.analyzer = MomentumDecayAnalyzer()

    def test_healthy_low_decay(self):
        snaps = _healthy_window()
        m = self.analyzer.analyze(snaps)
        assert m.momentum_velocity > 0
        assert m.volume_deterioration < 0.5

    def test_deteriorating_high_decay(self):
        snaps = _deteriorating_window()
        m = self.analyzer.analyze(snaps)
        assert m.momentum_velocity < 0

    def test_volume_decay_all_below_1(self):
        snaps = [_make_snap(i, volume_ratio=0.7) for i in range(1, 6)]
        m = self.analyzer.analyze(snaps)
        assert m.volume_deterioration == pytest.approx(1.0)

    def test_single_snapshot(self):
        snap = [_make_snap(1)]
        m = self.analyzer.analyze(snap)
        assert m is not None

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            self.analyzer.analyze([])

    def test_upper_shadow_expansion(self):
        snaps = [_make_snap(i, upper_shadow=0.9) for i in range(1, 6)]
        m = self.analyzer.analyze(snaps)
        assert m.upper_shadow_expansion > 0.8

    def test_failed_continuation_count(self):
        snaps = [_make_snap(i, exit_signals=2) for i in range(1, 4)]
        m = self.analyzer.analyze(snaps)
        assert m.failed_continuation_count == 3

    def test_atr_contraction_volatility_compression(self):
        snaps = [_make_snap(i, vol_compression=0.95) for i in range(1, 6)]
        m = self.analyzer.analyze(snaps)
        assert m.range_contraction > 0.8


class TestStructuralFailureDetector:
    def setup_method(self):
        self.detector = StructuralFailureDetector()

    def test_healthy_no_failure(self):
        snaps = _healthy_window()
        s = self.detector.analyze(snaps)
        assert not s.breakout_failure
        assert s.structural_failure_score < 0.4

    def test_rs_collapse(self):
        snaps = [_make_snap(1, rs=90.0), _make_snap(2, rs=85.0), _make_snap(3, rs=70.0)]
        s = self.detector.analyze(snaps)
        assert s.rs_collapse

    def test_no_rs_collapse_small_drop(self):
        snaps = [_make_snap(1, rs=82.0), _make_snap(2, rs=80.0), _make_snap(3, rs=79.0)]
        s = self.detector.analyze(snaps)
        assert not s.rs_collapse

    def test_breakout_failure_pattern(self):
        snaps = [
            _make_snap(1, close=1050, momentum=0.05),
            _make_snap(2, close=1030, momentum=-0.02),
            _make_snap(3, close=990, momentum=-0.05),
        ]
        # peak > close by >5%, momentum was positive
        s = self.detector.analyze(snaps)
        assert s.breakout_failure or s.support_rejection

    def test_convexity_collapse_high_distance(self):
        snaps = [_make_snap(i, close=800.0, peak=1050.0) for i in range(1, 4)]
        s = self.detector.analyze(snaps)
        assert s.convexity_collapse

    def test_breadth_deterioration(self):
        snaps = [_make_snap(i, breadth=0.25) for i in range(1, 4)]
        s = self.detector.analyze(snaps)
        assert s.breadth_deterioration

    def test_composite_score_range(self):
        for _ in range(10):
            snaps = _deteriorating_window()
            s = self.detector.analyze(snaps)
            assert 0.0 <= s.structural_failure_score <= 1.0


class TestPeakDecayVelocityAnalyzer:
    def setup_method(self):
        self.analyzer = PeakDecayVelocityAnalyzer()

    def test_slow_deterioration(self):
        # Very small distance from peak, growing slowly
        snaps = [_make_snap(i, close=1060.0 - i * 0.5, peak=1065.0) for i in range(1, 6)]
        m = self.analyzer.analyze(snaps)
        assert m.decay_classification == "slow"

    def test_fast_convex_collapse(self):
        # Large rapid decline
        snaps = [_make_snap(i, close=1060.0 - i * 30, peak=1100.0) for i in range(1, 6)]
        m = self.analyzer.analyze(snaps)
        assert m.decay_classification == "fast_convex"

    def test_orderly_consolidation(self):
        snaps = [_make_snap(i, close=1040.0 - i * 3, peak=1060.0) for i in range(1, 6)]
        m = self.analyzer.analyze(snaps)
        assert m.decay_classification == "orderly"

    def test_days_from_peak_nonnegative(self):
        snaps = _healthy_window()
        m = self.analyzer.analyze(snaps)
        assert m.days_from_peak >= 0

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            self.analyzer.analyze([])

    def test_drawdown_acceleration_sign(self):
        # Accelerating drawdown
        snaps = [_make_snap(i, close=1060.0 - i ** 2 * 3, peak=1080.0) for i in range(1, 7)]
        m = self.analyzer.analyze(snaps)
        assert m.deterioration_slope >= 0  # distance growing


class TestConvexityRetentionAnalyzer:
    def setup_method(self):
        self.analyzer = ConvexityRetentionAnalyzer()

    def test_high_tight_flag(self):
        snaps = [
            _make_snap(i, close=1058.0, peak=1060.0, rs=82.0,
                       vol_compression=0.85, momentum=0.01)
            for i in range(1, 6)
        ]
        r = self.analyzer.analyze(snaps)
        assert r.is_high_tight_flag
        assert r.is_constructive_compression

    def test_persistent_leader(self):
        snaps = [_make_snap(i, rs=85.0) for i in range(1, 6)]
        r = self.analyzer.analyze(snaps)
        assert r.is_persistent_leader

    def test_non_leader(self):
        snaps = [_make_snap(i, rs=50.0) for i in range(1, 6)]
        r = self.analyzer.analyze(snaps)
        assert not r.is_persistent_leader

    def test_convexity_retention_score_range(self):
        for snaps in [_healthy_window(), _deteriorating_window()]:
            r = self.analyzer.analyze(snaps)
            assert 0.0 <= r.convexity_retention_score <= 1.0

    def test_winner_evaporation_low_score(self):
        # Position rapidly losing ground from peak
        snaps = [_make_snap(i, close=900.0, peak=1100.0, rs=30.0,
                            momentum=-0.08) for i in range(1, 6)]
        r = self.analyzer.analyze(snaps)
        assert r.convexity_retention_score < 0.3


class TestPersistenceExtensionLayer:
    def setup_method(self):
        self.layer = PersistenceExtensionLayer()

    def test_strong_persistence(self):
        snaps = [_make_snap(i, rs=88.0, momentum=0.03, regime="bull",
                            trend_pers=0.9) for i in range(1, 11)]
        p = self.layer.analyze(snaps)
        assert p.combined_persistence_score > 0.6
        assert p.trend_persistence_durability == pytest.approx(1.0)

    def test_weak_persistence(self):
        snaps = [_make_snap(i, rs=45.0, momentum=-0.02, regime="bear",
                            trend_pers=0.2) for i in range(1, 11)]
        p = self.layer.analyze(snaps)
        assert p.combined_persistence_score < 0.5

    def test_regime_persistence(self):
        snaps = [_make_snap(i, regime="bull") for i in range(1, 11)]
        p = self.layer.analyze(snaps)
        assert p.regime_persistence_score == pytest.approx(1.0)

    def test_mixed_regime_reduces_persistence(self):
        snaps = [_make_snap(i, regime="bull" if i <= 5 else "bear") for i in range(1, 11)]
        p = self.layer.analyze(snaps)
        assert p.regime_persistence_score < 1.0

    def test_combined_score_range(self):
        for snaps in [_healthy_window(10), _deteriorating_window(10)]:
            p = self.layer.analyze(snaps)
            assert 0.0 <= p.combined_persistence_score <= 1.0
