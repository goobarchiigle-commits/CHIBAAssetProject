"""tests/analytics/test_exit_intelligence/test_dataset_and_report.py"""
import json
from pathlib import Path

import pytest

from src.analytics.exit_intelligence.dataset import (
    ExitResearchDatasetBuilder,
    ExitResearchDatasetStore,
)
from src.analytics.exit_intelligence.daily_report import run_daily_report
from src.analytics.exit_intelligence.models import ExitObservationRecord


def _make_obs(symbol="1234", entry="2026-01-01", exit_date="2026-01-10",
              realized_return=0.10, holding_days=9, velocity_score=2.5):
    return ExitObservationRecord(
        obs_id=ExitObservationRecord.make_obs_id(symbol, entry, exit_date),
        symbol=symbol, entry_date=entry, exit_date=exit_date,
        holding_days=holding_days, entry_price=1000.0,
        exit_price=1000.0 * (1 + realized_return),
        realized_return=realized_return,
        realized_pnl_jpy=realized_return * 100_000,
        exit_reason="turtle_exit_55d",
        unrealized_pnl_at_exit=realized_return * 100_000,
        distance_from_peak_pct=0.05, rs_rank_at_exit=80.0,
        atr_pct_at_exit=0.02, breadth_score_at_exit=0.65,
        regime_at_exit="bull", exit_velocity_score=velocity_score,
        path_snapshot_count=holding_days,
    )


class TestExitResearchDatasetBuilder:
    def setup_method(self):
        self.builder = ExitResearchDatasetBuilder()

    def test_build_record_basic(self):
        obs = _make_obs()
        record = self.builder.build_record(obs)
        assert record.obs_id == obs.obs_id
        assert record.symbol == "1234"
        assert record.realized_return == pytest.approx(0.10)

    def test_build_record_with_velocity(self):
        from src.analytics.exit_intelligence.models import ExitVelocityScore
        obs = _make_obs()
        evs = ExitVelocityScore(
            score_id=ExitVelocityScore.make_score_id("1234", "2026-01-10"),
            symbol="1234", scored_date="2026-01-10",
            rs_deterioration=0.2, breadth_decay=0.2, breakout_failure=0.1,
            vwap_rejection=0.1, upper_shadow=0.2, volume_decay=0.2,
            volatility_compression=0.2, momentum_decay=0.2,
            distance_from_peak=0.1, execution_deterioration=0.0,
            portfolio_replacement_pressure=0.0, total_score=5.0,
            action_band="staged_reduction", component_weights={}, generated_at="now",
        )
        record = self.builder.build_record(obs, velocity_score=evs)
        assert record.exit_velocity_score == 5.0
        assert record.action_band == "staged_reduction"

    def test_research_id_deterministic(self):
        obs = _make_obs()
        r1 = self.builder.build_record(obs)
        r2 = self.builder.build_record(obs)
        assert r1.research_id == r2.research_id

    def test_integrity_hash_deterministic(self):
        obs_list = [_make_obs()]
        records = [self.builder.build_record(obs) for obs in obs_list]
        h1 = self.builder.compute_integrity_hash(records)
        h2 = self.builder.compute_integrity_hash(records)
        assert h1 == h2

    def test_integrity_hash_changes_on_modification(self):
        obs = _make_obs()
        r1 = self.builder.build_record(obs)
        r2 = self.builder.build_record(_make_obs(realized_return=0.20))
        h1 = self.builder.compute_integrity_hash([r1])
        h2 = self.builder.compute_integrity_hash([r2])
        assert h1 != h2

    def test_counterfactual_fields_preserved(self):
        obs = _make_obs()
        record = self.builder.build_record(
            obs, post_exit_return_5d=0.03, exit_was_premature=True
        )
        assert record.post_exit_return_5d == pytest.approx(0.03)
        assert record.exit_was_premature is True


class TestExitResearchDatasetStore:
    def test_append_and_load(self, tmp_path):
        store = ExitResearchDatasetStore(tmp_path / "dataset.jsonl")
        builder = ExitResearchDatasetBuilder()
        record = builder.build_record(_make_obs())
        store.append(record)
        loaded = store.load_all()
        assert len(loaded) == 1
        assert loaded[0].symbol == "1234"

    def test_multiple_appends(self, tmp_path):
        store = ExitResearchDatasetStore(tmp_path / "dataset.jsonl")
        builder = ExitResearchDatasetBuilder()
        for i in range(5):
            obs = _make_obs(symbol=f"100{i}", exit_date=f"2026-01-{10+i}")
            store.append(builder.build_record(obs))
        assert len(store.load_all()) == 5

    def test_integrity_verification_passes(self, tmp_path):
        store = ExitResearchDatasetStore(tmp_path / "dataset.jsonl")
        builder = ExitResearchDatasetBuilder()
        store.append(builder.build_record(_make_obs()))
        assert store.verify_integrity()

    def test_snapshot_stats_structure(self, tmp_path):
        store = ExitResearchDatasetStore(tmp_path / "dataset.jsonl")
        stats = store.snapshot_stats()
        assert stats["total_records"] == 0

    def test_snapshot_stats_with_data(self, tmp_path):
        store = ExitResearchDatasetStore(tmp_path / "dataset.jsonl")
        builder = ExitResearchDatasetBuilder()
        obs_winners = [_make_obs(symbol=f"100{i}", realized_return=0.05,
                                  exit_date=f"2026-01-{10+i}") for i in range(3)]
        obs_losers = [_make_obs(symbol=f"200{i}", realized_return=-0.03,
                                 exit_date=f"2026-01-{10+i}") for i in range(2)]
        for obs in obs_winners + obs_losers:
            store.append(builder.build_record(obs))
        stats = store.snapshot_stats()
        assert stats["total_records"] == 5
        assert stats["positive_return_rate"] == pytest.approx(0.6)

    def test_empty_store_integrity(self, tmp_path):
        store = ExitResearchDatasetStore(tmp_path / "dataset.jsonl")
        assert store.verify_integrity()


class TestDailyReport:
    def test_report_generates(self, tmp_path):
        """Daily report generates without error from empty stores."""
        report = run_daily_report(
            obs_path=tmp_path / "obs.jsonl",
            path_snapshots_path=tmp_path / "paths.jsonl",
            velocity_path=tmp_path / "vel.jsonl",
            portfolio_path=tmp_path / "port.jsonl",
            dataset_path=tmp_path / "ds.jsonl",
            reports_dir=tmp_path / "reports",
            report_date="2026-01-15",
        )
        assert report["report_date"] == "2026-01-15"
        assert "section_6_alerts" in report

    def test_report_writes_markdown(self, tmp_path):
        run_daily_report(
            obs_path=tmp_path / "obs.jsonl",
            path_snapshots_path=tmp_path / "paths.jsonl",
            velocity_path=tmp_path / "vel.jsonl",
            portfolio_path=tmp_path / "port.jsonl",
            dataset_path=tmp_path / "ds.jsonl",
            reports_dir=tmp_path / "reports",
            report_date="2026-01-15",
        )
        md_file = tmp_path / "reports" / "exit_intelligence_daily_2026-01-15.md"
        assert md_file.exists()
        content = md_file.read_text(encoding="utf-8")
        assert "Exit Intelligence Daily Report" in content
        assert "2026-01-15" in content

    def test_report_writes_json(self, tmp_path):
        run_daily_report(
            obs_path=tmp_path / "obs.jsonl",
            path_snapshots_path=tmp_path / "paths.jsonl",
            velocity_path=tmp_path / "vel.jsonl",
            portfolio_path=tmp_path / "port.jsonl",
            dataset_path=tmp_path / "ds.jsonl",
            reports_dir=tmp_path / "reports",
            report_date="2026-01-15",
        )
        json_file = tmp_path / "reports" / "exit_intelligence_daily_2026-01-15.json"
        assert json_file.exists()
        data = json.loads(json_file.read_text(encoding="utf-8"))
        assert data["report_date"] == "2026-01-15"

    def test_report_with_observation_data(self, tmp_path):
        from src.analytics.exit_intelligence.observation import ExitObservationStore
        store = ExitObservationStore(tmp_path / "obs.jsonl")
        store.append(_make_obs(velocity_score=3.0))
        report = run_daily_report(
            obs_path=tmp_path / "obs.jsonl",
            path_snapshots_path=tmp_path / "paths.jsonl",
            velocity_path=tmp_path / "vel.jsonl",
            portfolio_path=tmp_path / "port.jsonl",
            dataset_path=tmp_path / "ds.jsonl",
            reports_dir=tmp_path / "reports",
            report_date="2026-01-10",
        )
        assert report["section_3_today_exits"]["count"] == 1

    def test_report_deterministic(self, tmp_path):
        """Same inputs → same report."""
        kwargs = dict(
            obs_path=tmp_path / "obs.jsonl",
            path_snapshots_path=tmp_path / "paths.jsonl",
            velocity_path=tmp_path / "vel.jsonl",
            portfolio_path=tmp_path / "port.jsonl",
            dataset_path=tmp_path / "ds.jsonl",
            reports_dir=tmp_path / "reports",
            report_date="2026-01-15",
        )
        r1 = run_daily_report(**kwargs)
        r2 = run_daily_report(**kwargs)
        assert r1["section_5_research_dataset"] == r2["section_5_research_dataset"]

    def test_high_pressure_alert(self, tmp_path):
        """Velocity score ≥ 5 triggers high-pressure alert."""
        from src.analytics.exit_intelligence.models import ExitVelocityScore
        from src.analytics.exit_intelligence.exit_velocity import ExitVelocityStore
        vs = ExitVelocityScore(
            score_id=ExitVelocityScore.make_score_id("1234", "2026-01-15"),
            symbol="1234", scored_date="2026-01-15",
            rs_deterioration=0.6, breadth_decay=0.5, breakout_failure=0.7,
            vwap_rejection=0.6, upper_shadow=0.7, volume_decay=0.6,
            volatility_compression=0.5, momentum_decay=0.6,
            distance_from_peak=0.5, execution_deterioration=0.1,
            portfolio_replacement_pressure=0.4, total_score=6.5,
            action_band="staged_reduction", component_weights={}, generated_at="now",
        )
        ExitVelocityStore(tmp_path / "vel.jsonl").append(vs)
        report = run_daily_report(
            obs_path=tmp_path / "obs.jsonl",
            path_snapshots_path=tmp_path / "paths.jsonl",
            velocity_path=tmp_path / "vel.jsonl",
            portfolio_path=tmp_path / "port.jsonl",
            dataset_path=tmp_path / "ds.jsonl",
            reports_dir=tmp_path / "reports",
            report_date="2026-01-15",
        )
        assert report["section_6_alerts"]["high_pressure_count"] == 1
