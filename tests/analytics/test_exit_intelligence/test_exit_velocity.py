"""tests/analytics/test_exit_intelligence/test_exit_velocity.py"""
import pytest

from src.analytics.exit_intelligence.deterioration import (
    ConvexityRetentionAnalyzer,
    MomentumDecayAnalyzer,
    PeakDecayVelocityAnalyzer,
    PersistenceExtensionLayer,
    StructuralFailureDetector,
)
from src.analytics.exit_intelligence.exit_velocity import ExitVelocityScorer, ExitVelocityStore
from src.analytics.exit_intelligence.models import (
    ConvexityRetentionMetrics,
    MomentumDecayMetrics,
    PeakDecayVelocityMetrics,
    PersistenceMetrics,
    PortfolioExitPressureSnapshot,
    StructuralFailureMetrics,
)
from src.analytics.exit_intelligence.observation import ExitPathSnapshot


def _make_snap(
    day=5, close=1050.0, peak=1060.0, rs=80.0,
    momentum=0.02, momentum_prev=0.01,
    vol_compression=0.5, upper_shadow=0.3,
    breadth=0.6, exit_signals=0, volume_ratio=1.1,
):
    return ExitPathSnapshot(
        snapshot_id="test",
        symbol="1234", entry_date="2026-01-01",
        snapshot_date="2026-01-05", day_number=day,
        close_price=close, entry_price=1000.0,
        unrealized_pnl_fraction=(close - 1000) / 1000,
        unrealized_pnl_jpy=(close - 1000) * 100,
        peak_price_so_far=peak,
        distance_from_peak_pct=(peak - close) / peak,
        rs_rank=rs, atr_pct=0.02, breadth_score=breadth,
        regime="bull", trend_persistence=0.7,
        momentum=momentum, momentum_prev=momentum_prev,
        volume_ratio=volume_ratio, volatility_compression=vol_compression,
        upper_shadow_ratio=upper_shadow,
        exit_signal_count=exit_signals, deterioration_velocity=0.002,
    )


def _make_momentum_metrics(velocity=0.001, vol_det=0.1, range_cont=0.3,
                            upper_shadow=0.3, breakout_exhaustion=0.1,
                            failed_cont=0):
    return MomentumDecayMetrics(
        symbol="1234", as_of_date="2026-01-05",
        momentum_velocity=velocity, volume_deterioration=vol_det,
        range_contraction=range_cont, upper_shadow_expansion=upper_shadow,
        breakout_exhaustion_score=breakout_exhaustion,
        failed_continuation_count=failed_cont,
    )


def _make_structural_metrics(failure_score=0.1, vwap_fail=False, rs_collapse=False,
                              bf=False, breadth_det=False, conv_collapse=False):
    return StructuralFailureMetrics(
        symbol="1234", as_of_date="2026-01-05",
        breakout_failure=bf, vwap_reclaim_failure=vwap_fail,
        rs_collapse=rs_collapse, breadth_deterioration=breadth_det,
        failed_trend_continuation=False, support_rejection=False,
        convexity_collapse=conv_collapse,
        structural_failure_score=failure_score,
    )


def _make_peak_metrics(velocity=0.005, classification="slow"):
    return PeakDecayVelocityMetrics(
        symbol="1234", as_of_date="2026-01-05",
        peak_decay_velocity=velocity, drawdown_acceleration=0.0,
        days_from_peak=2, deterioration_slope=0.001,
        decay_classification=classification,
    )


def _make_conv_metrics(is_htf=False, is_leader=True, retention=0.8):
    return ConvexityRetentionMetrics(
        symbol="1234", as_of_date="2026-01-05",
        is_constructive_compression=False, is_high_tight_flag=is_htf,
        is_persistent_leader=is_leader,
        trend_continuation_quality=0.7, convexity_retention_score=retention,
    )


def _make_pers_metrics(combined=0.75):
    return PersistenceMetrics(
        symbol="1234", as_of_date="2026-01-05",
        trend_persistence_durability=0.8, leader_persistence_score=0.8,
        regime_persistence_score=0.7, sector_leadership_continuity=0.6,
        combined_persistence_score=combined,
    )


def _make_portfolio_pressure(replacement=0.0):
    return PortfolioExitPressureSnapshot(
        pressure_id="test", snapshot_date="2026-01-05",
        concurrent_positions=2, gross_exposure_jpy=2_000_000,
        sector_concentration_max=0.5, signal_queue_pressure=0.3,
        replacement_pressure=replacement, portfolio_heat=0.05,
        cash_pressure=0.3, capital_utilization=0.7, avg_holding_days=5.0,
        worst_position_deterioration=0.08,
    )


class TestExitVelocityScorer:
    def setup_method(self):
        self.scorer = ExitVelocityScorer()

    def _score(self, snap=None, mom=None, struct=None, peak=None,
               conv=None, pers=None, port=None, slippage=0.0):
        return self.scorer.score(
            symbol="1234", scored_date="2026-01-05",
            latest_snapshot=snap or _make_snap(),
            momentum_decay=mom or _make_momentum_metrics(),
            structural_failure=struct or _make_structural_metrics(),
            peak_decay=peak or _make_peak_metrics(),
            convexity_retention=conv or _make_conv_metrics(),
            persistence=pers or _make_pers_metrics(),
            portfolio_pressure=port,
            avg_exec_slippage_bps=slippage,
        )

    def test_healthy_hold_band(self):
        evs = self._score()
        # Healthy conditions → low score
        assert evs.total_score <= 4.0
        assert evs.action_band in ("hold", "tighten_risk")

    def test_high_pressure_aggressive_exit(self):
        snap = _make_snap(rs=20.0, close=800.0, peak=1100.0,
                          momentum=-0.10, upper_shadow=0.9,
                          vol_compression=0.1, breadth=0.2)
        mom = _make_momentum_metrics(velocity=-0.05, vol_det=0.9, upper_shadow=0.9,
                                     breakout_exhaustion=0.8, failed_cont=5)
        struct = _make_structural_metrics(failure_score=0.9, vwap_fail=True,
                                          rs_collapse=True, bf=True, breadth_det=True)
        conv = _make_conv_metrics(is_htf=False, is_leader=False, retention=0.1)
        evs = self._score(snap=snap, mom=mom, struct=struct, conv=conv)
        assert evs.total_score >= 5.0

    def test_total_score_range(self):
        for _ in range(20):
            evs = self._score()
            assert 0.0 <= evs.total_score <= 10.0

    def test_action_band_consistency(self):
        for score_val, expected_band in [
            (1.5, "hold"), (3.0, "tighten_risk"), (5.0, "staged_reduction"), (7.5, "aggressive_exit")
        ]:
            # Manufacture a known total score range
            evs = self._score()
            # Just validate the scoring bands mapping
            from src.analytics.exit_intelligence.exit_velocity import _action_band
            assert _action_band(score_val) == expected_band

    def test_persistent_leader_dampens_score(self):
        snap_bad = _make_snap(rs=20.0, close=850.0, peak=1100.0, momentum=-0.08)
        conv_leader = _make_conv_metrics(is_htf=False, is_leader=True, retention=0.8)
        conv_none = _make_conv_metrics(is_htf=False, is_leader=False, retention=0.1)
        evs_leader = self._score(snap=snap_bad, conv=conv_leader)
        evs_none = self._score(snap=snap_bad, conv=conv_none)
        assert evs_leader.total_score <= evs_none.total_score

    def test_htf_dampens_score(self):
        snap_bad = _make_snap(rs=20.0, close=850.0, peak=1100.0, momentum=-0.08)
        conv_htf = _make_conv_metrics(is_htf=True, is_leader=False, retention=0.9)
        conv_none = _make_conv_metrics(is_htf=False, is_leader=False, retention=0.1)
        evs_htf = self._score(snap=snap_bad, conv=conv_htf)
        evs_none = self._score(snap=snap_bad, conv=conv_none)
        assert evs_htf.total_score <= evs_none.total_score

    def test_portfolio_pressure_increases_score(self):
        evs_no_port = self._score(port=None)
        evs_with_port = self._score(port=_make_portfolio_pressure(replacement=1.0))
        assert evs_with_port.total_score >= evs_no_port.total_score

    def test_determinism(self):
        snap = _make_snap()
        evs1 = self._score(snap=snap)
        evs2 = self._score(snap=snap)
        assert evs1.total_score == evs2.total_score
        assert evs1.action_band == evs2.action_band

    def test_score_id_determinism(self):
        id1 = self.scorer.score(
            "1234", "2026-01-05", _make_snap(),
            _make_momentum_metrics(), _make_structural_metrics(),
            _make_peak_metrics(), _make_conv_metrics(), _make_pers_metrics(),
        ).score_id
        id2 = self.scorer.score(
            "1234", "2026-01-05", _make_snap(),
            _make_momentum_metrics(), _make_structural_metrics(),
            _make_peak_metrics(), _make_conv_metrics(), _make_pers_metrics(),
        ).score_id
        assert id1 == id2

    def test_all_component_scores_in_range(self):
        evs = self._score()
        for comp in [
            evs.rs_deterioration, evs.breadth_decay, evs.breakout_failure,
            evs.vwap_rejection, evs.upper_shadow, evs.volume_decay,
            evs.volatility_compression, evs.momentum_decay,
            evs.distance_from_peak, evs.execution_deterioration,
            evs.portfolio_replacement_pressure,
        ]:
            assert 0.0 <= comp <= 1.0 + 1e-9

    def test_rs_collapse_raises_score(self):
        evs_normal = self._score(struct=_make_structural_metrics(failure_score=0.1))
        evs_collapse = self._score(struct=_make_structural_metrics(failure_score=0.9, rs_collapse=True))
        assert evs_collapse.total_score > evs_normal.total_score

    def test_volatility_compression_winner(self):
        # Constructive compression should keep score low
        snap = _make_snap(vol_compression=0.9, close=1058.0, peak=1060.0, rs=85.0)
        conv = _make_conv_metrics(is_htf=True, is_leader=True)
        evs = self._score(snap=snap, conv=conv)
        assert evs.action_band in ("hold", "tighten_risk")


class TestExitVelocityStore:
    def test_append_load(self, tmp_path):
        store = ExitVelocityStore(tmp_path / "vel.jsonl")
        scorer = ExitVelocityScorer()
        evs = scorer.score(
            "1234", "2026-01-05", _make_snap(),
            _make_momentum_metrics(), _make_structural_metrics(),
            _make_peak_metrics(), _make_conv_metrics(), _make_pers_metrics(),
        )
        store.append(evs)
        loaded = store.load_all()
        assert len(loaded) == 1
        assert loaded[0].symbol == "1234"

    def test_load_latest_by_symbol(self, tmp_path):
        store = ExitVelocityStore(tmp_path / "vel.jsonl")
        scorer = ExitVelocityScorer()
        for date in ["2026-01-03", "2026-01-05", "2026-01-04"]:
            from src.analytics.exit_intelligence.models import ExitVelocityScore
            evs = ExitVelocityScore(
                score_id=ExitVelocityScore.make_score_id("1234", date),
                symbol="1234", scored_date=date,
                rs_deterioration=0.2, breadth_decay=0.2,
                breakout_failure=0.1, vwap_rejection=0.1,
                upper_shadow=0.3, volume_decay=0.2,
                volatility_compression=0.3, momentum_decay=0.2,
                distance_from_peak=0.1, execution_deterioration=0.0,
                portfolio_replacement_pressure=0.0,
                total_score=2.5, action_band="hold",
                component_weights={}, generated_at="now",
            )
            store.append(evs)
        latest = store.load_latest_by_symbol("1234")
        assert latest.scored_date == "2026-01-05"
