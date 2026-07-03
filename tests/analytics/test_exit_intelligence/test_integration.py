"""tests/analytics/test_exit_intelligence/test_integration.py

End-to-end integration tests:
  - Replay determinism
  - Corruption detection
  - Counterfactual consistency
  - Full pipeline from snapshot → velocity score → report
  - Portfolio pressure → staged exit
  - Regime transitions
"""
import json
import pytest

from src.analytics.exit_intelligence.deterioration import (
    ConvexityRetentionAnalyzer,
    MomentumDecayAnalyzer,
    PeakDecayVelocityAnalyzer,
    PersistenceExtensionLayer,
    StructuralFailureDetector,
)
from src.analytics.exit_intelligence.exit_velocity import ExitVelocityScorer
from src.analytics.exit_intelligence.lifecycle import (
    CounterfactualExitAnalyzer,
    LoserRetentionAnalysis,
    PartialExitAllocator,
)
from src.analytics.exit_intelligence.models import ExitObservationRecord
from src.analytics.exit_intelligence.observation import ExitPathSnapshot
from src.analytics.exit_intelligence.validator import IntegrityValidator


def _make_snaps(
    n=10, symbol="1234", entry="2026-01-01",
    close_fn=None, peak_fn=None, rs=80.0,
    momentum_fn=None,
):
    close_fn = close_fn or (lambda i: 1000.0 + i * 5)
    peak_fn = peak_fn or (lambda i: max(1000.0 + j * 5 for j in range(1, i + 1)))
    momentum_fn = momentum_fn or (lambda i: 0.02 + i * 0.001)
    snaps = []
    for i in range(1, n + 1):
        close = close_fn(i)
        peak = peak_fn(i)
        snap = ExitPathSnapshot(
            snapshot_id=ExitPathSnapshot.make_snapshot_id(symbol, entry, f"2026-01-{i:02d}"),
            symbol=symbol, entry_date=entry,
            snapshot_date=f"2026-01-{i:02d}", day_number=i,
            close_price=close, entry_price=1000.0,
            unrealized_pnl_fraction=(close - 1000.0) / 1000.0,
            unrealized_pnl_jpy=(close - 1000.0) * 100,
            peak_price_so_far=peak,
            distance_from_peak_pct=(peak - close) / peak if peak > 0 else 0.0,
            rs_rank=rs, atr_pct=0.02, breadth_score=0.6,
            regime="bull", trend_persistence=0.7,
            momentum=momentum_fn(i), momentum_prev=momentum_fn(max(i - 1, 1)),
            volume_ratio=1.1, volatility_compression=0.5,
            upper_shadow_ratio=0.3, exit_signal_count=0,
            deterioration_velocity=0.002,
        )
        snaps.append(snap)
    return snaps


class TestReplayDeterminism:
    def test_same_snapshots_same_velocity_score(self):
        snaps = _make_snaps(n=10)
        scorer = ExitVelocityScorer()
        analyzers = {
            "mom": MomentumDecayAnalyzer(),
            "struct": StructuralFailureDetector(),
            "peak": PeakDecayVelocityAnalyzer(),
            "conv": ConvexityRetentionAnalyzer(),
            "pers": PersistenceExtensionLayer(),
        }

        def _compute():
            m = analyzers["mom"].analyze(snaps)
            s = analyzers["struct"].analyze(snaps)
            p = analyzers["peak"].analyze(snaps)
            c = analyzers["conv"].analyze(snaps)
            pr = analyzers["pers"].analyze(snaps)
            return scorer.score(
                "1234", "2026-01-10", snaps[-1], m, s, p, c, pr
            )

        evs1 = _compute()
        evs2 = _compute()
        assert evs1.total_score == evs2.total_score
        assert evs1.action_band == evs2.action_band
        assert evs1.score_id == evs2.score_id

    def test_different_snapshots_different_scores(self):
        snaps_good = _make_snaps(n=10, rs=85.0, close_fn=lambda i: 1000 + i * 8)
        snaps_bad = _make_snaps(
            n=10, rs=30.0,
            close_fn=lambda i: 1000 - i * 15,
            peak_fn=lambda i: 1000.0,
            momentum_fn=lambda i: -0.03 * i,
        )
        scorer = ExitVelocityScorer()
        analyzers = {
            "mom": MomentumDecayAnalyzer(),
            "struct": StructuralFailureDetector(),
            "peak": PeakDecayVelocityAnalyzer(),
            "conv": ConvexityRetentionAnalyzer(),
            "pers": PersistenceExtensionLayer(),
        }

        def _score(snaps):
            return scorer.score(
                "1234", "2026-01-10", snaps[-1],
                analyzers["mom"].analyze(snaps),
                analyzers["struct"].analyze(snaps),
                analyzers["peak"].analyze(snaps),
                analyzers["conv"].analyze(snaps),
                analyzers["pers"].analyze(snaps),
            )

        good = _score(snaps_good)
        bad = _score(snaps_bad)
        assert bad.total_score > good.total_score

    def test_observation_ids_replay_consistent(self):
        obs_id1 = ExitObservationRecord.make_obs_id("1234", "2026-01-01", "2026-01-10")
        obs_id2 = ExitObservationRecord.make_obs_id("1234", "2026-01-01", "2026-01-10")
        assert obs_id1 == obs_id2


class TestCorruptionDetection:
    def test_tampered_obs_id_detected(self):
        obs = ExitObservationRecord(
            obs_id="tampered_id_000",
            symbol="1234", entry_date="2026-01-01",
            exit_date="2026-01-10", holding_days=9,
            entry_price=1000.0, exit_price=1100.0,
            realized_return=0.10, realized_pnl_jpy=10000.0,
            exit_reason="turtle_exit_55d",
            unrealized_pnl_at_exit=10000.0, distance_from_peak_pct=0.05,
            rs_rank_at_exit=80.0, atr_pct_at_exit=0.02,
            breadth_score_at_exit=0.65, regime_at_exit="bull",
            exit_velocity_score=2.5, path_snapshot_count=9,
        )
        validator = IntegrityValidator()
        report = validator.validate_exit_observations([obs])
        errors = [v for v in report.violations if v.rule == "immutable_observation_integrity"]
        assert len(errors) == 1

    def test_dataset_integrity_hash(self, tmp_path):
        from src.analytics.exit_intelligence.dataset import (
            ExitResearchDatasetBuilder, ExitResearchDatasetStore,
        )
        store = ExitResearchDatasetStore(tmp_path / "ds.jsonl")
        builder = ExitResearchDatasetBuilder()
        obs = ExitObservationRecord(
            obs_id=ExitObservationRecord.make_obs_id("1234", "2026-01-01", "2026-01-10"),
            symbol="1234", entry_date="2026-01-01", exit_date="2026-01-10",
            holding_days=9, entry_price=1000.0, exit_price=1100.0,
            realized_return=0.10, realized_pnl_jpy=10000.0,
            exit_reason="turtle_exit_55d",
            unrealized_pnl_at_exit=10000.0, distance_from_peak_pct=0.05,
            rs_rank_at_exit=80.0, atr_pct_at_exit=0.02,
            breadth_score_at_exit=0.65, regime_at_exit="bull",
            exit_velocity_score=2.5, path_snapshot_count=9,
        )
        record = builder.build_record(obs)
        store.append(record)
        assert store.verify_integrity()

    def test_peak_decrease_in_snapshots_detected(self):
        snaps = [
            _make_snaps(n=1)[0],
        ]
        # Manually set decreasing peak (impossible in real system)
        s2 = ExitPathSnapshot(
            snapshot_id=ExitPathSnapshot.make_snapshot_id("1234", "2026-01-01", "2026-01-02"),
            symbol="1234", entry_date="2026-01-01", snapshot_date="2026-01-02",
            day_number=2, close_price=1010.0, entry_price=1000.0,
            unrealized_pnl_fraction=0.01, unrealized_pnl_jpy=1000.0,
            peak_price_so_far=1000.0,  # decreased from 1005
            distance_from_peak_pct=0.0, rs_rank=80.0, atr_pct=0.02,
            breadth_score=0.6, regime="bull", trend_persistence=0.7,
            momentum=0.01, momentum_prev=0.0, volume_ratio=1.0,
            volatility_compression=0.5, upper_shadow_ratio=0.3,
            exit_signal_count=0, deterioration_velocity=0.0,
        )
        validator = IntegrityValidator()
        report = validator.validate_path_snapshots(snaps + [s2])
        errors = [v for v in report.violations if v.rule == "monotone_peak_validation"]
        assert len(errors) >= 1


class TestCounterfactualConsistency:
    def test_premature_exit_classification_consistent(self):
        obs = ExitObservationRecord(
            obs_id=ExitObservationRecord.make_obs_id("1234", "2026-01-01", "2026-01-10"),
            symbol="1234", entry_date="2026-01-01", exit_date="2026-01-10",
            holding_days=9, entry_price=1000.0, exit_price=1050.0,
            realized_return=0.05, realized_pnl_jpy=5000.0,
            exit_reason="turtle_exit_55d",
            unrealized_pnl_at_exit=5000.0, distance_from_peak_pct=0.02,
            rs_rank_at_exit=80.0, atr_pct_at_exit=0.02,
            breadth_score_at_exit=0.65, regime_at_exit="bull",
            exit_velocity_score=2.5, path_snapshot_count=9,
        )
        analyzer = CounterfactualExitAnalyzer()
        prices_rising = [1090.0 + i * 5 for i in range(20)]  # +4% at d5
        prices_falling = [1040.0 - i * 5 for i in range(20)]

        cf_premature = analyzer.analyze(obs, prices_rising)
        cf_not_premature = analyzer.analyze(obs, prices_falling)

        assert cf_premature.exit_was_premature is True
        assert cf_not_premature.exit_was_premature is False

    def test_loser_retention_consistent_classifications(self):
        la = LoserRetentionAnalysis()

        # High velocity loser → good_cut regardless of post-exit
        obs_high_vel = ExitObservationRecord(
            obs_id=ExitObservationRecord.make_obs_id("1234", "2026-01-01", "2026-01-10"),
            symbol="1234", entry_date="2026-01-01", exit_date="2026-01-10",
            holding_days=9, entry_price=1000.0, exit_price=950.0,
            realized_return=-0.05, realized_pnl_jpy=-5000.0,
            exit_reason="stop_loss",
            unrealized_pnl_at_exit=-5000.0, distance_from_peak_pct=0.10,
            rs_rank_at_exit=35.0, atr_pct_at_exit=0.03,
            breadth_score_at_exit=0.3, regime_at_exit="bear",
            exit_velocity_score=7.5, path_snapshot_count=9,
        )
        r = la.classify(obs_high_vel, post_exit_return_5d=-0.03)
        assert r.hold_classification == "good_cut"


class TestFullPipeline:
    def test_winner_evaporation_pattern(self):
        """Winner that slowly gives back gains → tighten_risk band."""
        snaps = _make_snaps(
            n=10,
            close_fn=lambda i: 1100.0 - (i - 5) ** 2 * 2 if i > 5 else 1100.0,
            peak_fn=lambda i: 1100.0,
            rs=75.0,
        )
        scorer = ExitVelocityScorer()
        m = MomentumDecayAnalyzer().analyze(snaps)
        s = StructuralFailureDetector().analyze(snaps)
        p = PeakDecayVelocityAnalyzer().analyze(snaps)
        c = ConvexityRetentionAnalyzer().analyze(snaps)
        pr = PersistenceExtensionLayer().analyze(snaps)
        evs = scorer.score("1234", "2026-01-10", snaps[-1], m, s, p, c, pr)
        assert evs.total_score <= 6.0  # Not aggressive exit — slow evaporation

    def test_fast_reversal_aggressive_exit(self):
        """Fast reversal → high score → aggressive exit."""
        snaps = _make_snaps(
            n=8,
            close_fn=lambda i: 1100.0 if i <= 3 else 1100.0 - (i - 3) * 40,
            peak_fn=lambda i: 1100.0,
            rs=30.0,
            momentum_fn=lambda i: 0.05 if i <= 3 else -0.08 * (i - 3),
        )
        scorer = ExitVelocityScorer()
        m = MomentumDecayAnalyzer().analyze(snaps)
        s = StructuralFailureDetector().analyze(snaps)
        p = PeakDecayVelocityAnalyzer().analyze(snaps)
        c = ConvexityRetentionAnalyzer().analyze(snaps)
        pr = PersistenceExtensionLayer().analyze(snaps)
        evs = scorer.score("1234", "2026-01-08", snaps[-1], m, s, p, c, pr)
        assert evs.total_score >= 4.0

    def test_atr_contraction_recognized_constructive(self):
        """Tight ATR contraction near peak → high tight flag → score dampened."""
        snaps = _make_snaps(
            n=8,
            close_fn=lambda i: 1058.0 + (i % 2) * 1.0,  # tiny range
            peak_fn=lambda i: 1062.0,
            rs=87.0,
        )
        # Override volatility_compression to high
        for s in snaps:
            s.volatility_compression = 0.9
            s.distance_from_peak_pct = 0.003

        scorer = ExitVelocityScorer()
        m = MomentumDecayAnalyzer().analyze(snaps)
        s_met = StructuralFailureDetector().analyze(snaps)
        p = PeakDecayVelocityAnalyzer().analyze(snaps)
        c = ConvexityRetentionAnalyzer().analyze(snaps)
        pr = PersistenceExtensionLayer().analyze(snaps)
        evs = scorer.score("1234", "2026-01-08", snaps[-1], m, s_met, p, c, pr)
        assert evs.action_band in ("hold", "tighten_risk")

    def test_staged_exit_allocator_integration(self):
        """Full: deteriorating → staged_reduction → 33% first reduction."""
        snaps = _make_snaps(
            n=10,
            close_fn=lambda i: 1050.0 - i * 10,
            peak_fn=lambda i: 1080.0,
            rs=45.0,
            momentum_fn=lambda i: -0.02 * i,
        )
        scorer = ExitVelocityScorer()
        m = MomentumDecayAnalyzer().analyze(snaps)
        s = StructuralFailureDetector().analyze(snaps)
        p = PeakDecayVelocityAnalyzer().analyze(snaps)
        c = ConvexityRetentionAnalyzer().analyze(snaps)
        pr = PersistenceExtensionLayer().analyze(snaps)
        evs = scorer.score("1234", "2026-01-10", snaps[-1], m, s, p, c, pr)

        alloc = PartialExitAllocator()
        decision = alloc.allocate(evs)
        assert decision.action_band == evs.action_band
        if decision.action_band == "staged_reduction":
            assert decision.recommended_first_reduction == pytest.approx(0.33)
        elif decision.action_band == "aggressive_exit":
            assert decision.recommended_first_reduction == 1.0

    def test_regime_transition_updates_velocity(self):
        """Regime switching bull→bear increases deterioration pressure."""
        snaps_bull = [
            ExitPathSnapshot(
                snapshot_id=f"snap_bull_{i}",
                symbol="1234", entry_date="2026-01-01",
                snapshot_date=f"2026-01-{i:02d}", day_number=i,
                close_price=1000.0 + i * 5, entry_price=1000.0,
                unrealized_pnl_fraction=i * 0.005,
                unrealized_pnl_jpy=i * 500,
                peak_price_so_far=1000.0 + i * 5,
                distance_from_peak_pct=0.0,
                rs_rank=82.0, atr_pct=0.02, breadth_score=0.65,
                regime="bull", trend_persistence=0.8,
                momentum=0.02, momentum_prev=0.015,
                volume_ratio=1.2, volatility_compression=0.5,
                upper_shadow_ratio=0.2, exit_signal_count=0,
                deterioration_velocity=0.003,
            )
            for i in range(1, 6)
        ]
        snaps_bear = [
            ExitPathSnapshot(
                snapshot_id=f"snap_bear_{i}",
                symbol="1234", entry_date="2026-01-01",
                snapshot_date=f"2026-01-{5+i:02d}", day_number=5 + i,
                close_price=1025.0 - i * 15, entry_price=1000.0,
                unrealized_pnl_fraction=0.025 - i * 0.015,
                unrealized_pnl_jpy=2500 - i * 1500,
                peak_price_so_far=1025.0,
                distance_from_peak_pct=i * 15 / 1025.0,
                rs_rank=55.0 - i * 5, atr_pct=0.03, breadth_score=0.35,
                regime="bear", trend_persistence=0.3,
                momentum=-0.03 * i, momentum_prev=-0.02 * i,
                volume_ratio=0.8, volatility_compression=0.3,
                upper_shadow_ratio=0.75, exit_signal_count=2,
                deterioration_velocity=-0.01,
            )
            for i in range(1, 6)
        ]
        all_snaps = snaps_bull + snaps_bear
        scorer = ExitVelocityScorer()
        m = MomentumDecayAnalyzer().analyze(all_snaps)
        s = StructuralFailureDetector().analyze(all_snaps)
        p = PeakDecayVelocityAnalyzer().analyze(all_snaps)
        c = ConvexityRetentionAnalyzer().analyze(all_snaps)
        pr = PersistenceExtensionLayer().analyze(all_snaps)
        evs_after = scorer.score("1234", "2026-01-10", all_snaps[-1], m, s, p, c, pr)

        # Just bull
        scorer2 = ExitVelocityScorer()
        m2 = MomentumDecayAnalyzer().analyze(snaps_bull)
        s2 = StructuralFailureDetector().analyze(snaps_bull)
        p2 = PeakDecayVelocityAnalyzer().analyze(snaps_bull)
        c2 = ConvexityRetentionAnalyzer().analyze(snaps_bull)
        pr2 = PersistenceExtensionLayer().analyze(snaps_bull)
        evs_bull = scorer2.score("1234", "2026-01-05", snaps_bull[-1], m2, s2, p2, c2, pr2)

        assert evs_after.total_score >= evs_bull.total_score
