"""tests/analytics/test_exit_intelligence/test_validator.py"""
import pytest

from src.analytics.exit_intelligence.models import (
    ExitObservationRecord,
    ExitVelocityScore,
)
from src.analytics.exit_intelligence.observation import ExitPathSnapshot
from src.analytics.exit_intelligence.validator import IntegrityValidator


def _make_obs(symbol="1234", entry="2026-01-01", exit_date="2026-01-10",
              holding_days=9, distance=0.05, velocity_score=2.5):
    return ExitObservationRecord(
        obs_id=ExitObservationRecord.make_obs_id(symbol, entry, exit_date),
        symbol=symbol, entry_date=entry, exit_date=exit_date,
        holding_days=holding_days, entry_price=1000.0, exit_price=1100.0,
        realized_return=0.10, realized_pnl_jpy=10000.0,
        exit_reason="turtle_exit_55d",
        unrealized_pnl_at_exit=10000.0, distance_from_peak_pct=distance,
        rs_rank_at_exit=80.0, atr_pct_at_exit=0.02,
        breadth_score_at_exit=0.65, regime_at_exit="bull",
        exit_velocity_score=velocity_score, path_snapshot_count=9,
    )


def _make_snap(symbol="1234", entry="2026-01-01", day=1, snap_date="2026-01-01",
               peak=1060.0, close=1050.0):
    return ExitPathSnapshot(
        snapshot_id=ExitPathSnapshot.make_snapshot_id(symbol, entry, snap_date),
        symbol=symbol, entry_date=entry, snapshot_date=snap_date,
        day_number=day, close_price=close, entry_price=1000.0,
        unrealized_pnl_fraction=0.05, unrealized_pnl_jpy=5000.0,
        peak_price_so_far=peak, distance_from_peak_pct=(peak - close) / peak,
        rs_rank=80.0, atr_pct=0.02, breadth_score=0.6, regime="bull",
        trend_persistence=0.7, momentum=0.02, momentum_prev=0.01,
        volume_ratio=1.1, volatility_compression=0.5,
        upper_shadow_ratio=0.3, exit_signal_count=0,
        deterioration_velocity=0.002,
    )


def _make_evs(score=2.5, band="hold"):
    return ExitVelocityScore(
        score_id=ExitVelocityScore.make_score_id("1234", "2026-01-10"),
        symbol="1234", scored_date="2026-01-10",
        rs_deterioration=0.2, breadth_decay=0.15,
        breakout_failure=0.1, vwap_rejection=0.1,
        upper_shadow=0.2, volume_decay=0.15,
        volatility_compression=0.2, momentum_decay=0.15,
        distance_from_peak=0.1, execution_deterioration=0.0,
        portfolio_replacement_pressure=0.0,
        total_score=score, action_band=band,
        component_weights={}, generated_at="now",
    )


class TestIntegrityValidator:
    def setup_method(self):
        self.validator = IntegrityValidator()

    # ── Exit observations ──────────────────────────────────────────────────────

    def test_valid_observations_pass(self):
        obs = [_make_obs()]
        report = self.validator.validate_exit_observations(obs)
        assert report.passed
        assert report.checked_records == 1

    def test_exit_before_entry_fails(self):
        obs = _make_obs(entry="2026-01-10", exit_date="2026-01-01")
        obs.obs_id = ExitObservationRecord.make_obs_id(
            obs.symbol, "2026-01-10", "2026-01-01"
        )
        report = self.validator.validate_exit_observations([obs])
        errors = [v for v in report.violations if v.rule == "no_exit_before_entry"]
        assert len(errors) == 1
        assert not report.passed

    def test_negative_holding_days_fails(self):
        obs = _make_obs(holding_days=-1)
        report = self.validator.validate_exit_observations([obs])
        assert not report.passed

    def test_future_leakage_fails(self):
        obs = _make_obs(exit_date="2026-02-01")
        report = self.validator.validate_exit_observations(
            [obs], evaluation_date="2026-01-15"
        )
        errors = [v for v in report.violations if v.rule == "no_future_leakage"]
        assert len(errors) == 1

    def test_tampered_obs_id_fails(self):
        obs = _make_obs()
        obs.obs_id = "tampered_id"
        report = self.validator.validate_exit_observations([obs])
        errors = [v for v in report.violations if v.rule == "immutable_observation_integrity"]
        assert len(errors) == 1

    def test_duplicate_obs_id_fails(self):
        obs = _make_obs()
        report = self.validator.validate_exit_observations([obs, obs])
        errors = [v for v in report.violations if v.rule == "append_order_consistency"]
        assert len(errors) == 1

    def test_negative_distance_from_peak_fails(self):
        obs = _make_obs(distance=-0.05)
        report = self.validator.validate_exit_observations([obs])
        errors = [v for v in report.violations if v.rule == "monotone_peak_validation"]
        assert len(errors) == 1

    # ── Path snapshots ─────────────────────────────────────────────────────────

    def test_valid_snapshots_pass(self):
        snaps = [
            _make_snap(day=1, snap_date="2026-01-01", peak=1010.0),
            _make_snap(day=2, snap_date="2026-01-02", peak=1020.0),
            _make_snap(day=3, snap_date="2026-01-03", peak=1025.0),
        ]
        report = self.validator.validate_path_snapshots(snaps)
        assert report.passed

    def test_non_monotone_day_number_fails(self):
        snaps = [
            _make_snap(day=3, snap_date="2026-01-01"),
            _make_snap(day=1, snap_date="2026-01-02"),
        ]
        report = self.validator.validate_path_snapshots(snaps)
        errors = [v for v in report.violations if v.rule == "monotone_day_numbers"]
        assert len(errors) >= 1

    def test_decreasing_peak_fails(self):
        snaps = [
            _make_snap(day=1, snap_date="2026-01-01", peak=1060.0),
            _make_snap(day=2, snap_date="2026-01-02", peak=1050.0),  # peak decreased
        ]
        report = self.validator.validate_path_snapshots(snaps)
        errors = [v for v in report.violations if v.rule == "monotone_peak_validation"]
        assert len(errors) >= 1

    def test_future_snapshot_fails(self):
        snaps = [_make_snap(day=1, snap_date="2026-02-01")]
        report = self.validator.validate_path_snapshots(
            snaps, evaluation_date="2026-01-15"
        )
        errors = [v for v in report.violations if v.rule == "no_future_leakage"]
        assert len(errors) == 1

    # ── Velocity scores ────────────────────────────────────────────────────────

    def test_valid_velocity_score_passes(self):
        evs = _make_evs(score=1.5, band="hold")  # 1.5 <= 2.0 → hold
        report = self.validator.validate_velocity_scores([evs])
        assert report.passed

    def test_score_out_of_range_fails(self):
        evs = _make_evs(score=12.0, band="aggressive_exit")
        report = self.validator.validate_velocity_scores([evs])
        errors = [v for v in report.violations if v.rule == "score_bounds"]
        assert len(errors) == 1

    def test_action_band_inconsistency_fails(self):
        evs = _make_evs(score=8.0, band="hold")  # score=8 should be aggressive_exit
        report = self.validator.validate_velocity_scores([evs])
        errors = [v for v in report.violations if v.rule == "action_band_consistency"]
        assert len(errors) == 1

    def test_all_bands_valid(self):
        for score, band in [(1.0, "hold"), (3.5, "tighten_risk"),
                             (5.5, "staged_reduction"), (7.5, "aggressive_exit")]:
            evs = _make_evs(score=score, band=band)
            report = self.validator.validate_velocity_scores([evs])
            assert report.passed, f"Failed for score={score} band={band}: {report.violations}"

    # ── Multiple records ───────────────────────────────────────────────────────

    def test_multiple_valid_records(self):
        obs_list = [
            _make_obs(symbol=f"100{i}", exit_date=f"2026-01-{10+i}")
            for i in range(5)
        ]
        report = self.validator.validate_exit_observations(obs_list)
        assert report.passed
        assert report.checked_records == 5

    def test_report_summary_structure(self):
        obs = [_make_obs()]
        report = self.validator.validate_exit_observations(obs)
        summary = report.summary()
        assert "passed" in summary
        assert "checked_records" in summary
        assert "error_count" in summary
        assert "warning_count" in summary
        assert "violations" in summary
