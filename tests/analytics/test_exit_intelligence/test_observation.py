"""tests/analytics/test_exit_intelligence/test_observation.py"""
import json
import tempfile
from pathlib import Path

import pytest

from src.analytics.exit_intelligence.models import (
    ExitObservationRecord,
    ExitPathSnapshot,
    ExecutionExitObservation,
    PortfolioExitPressureSnapshot,
    SCHEMA_VERSION,
)
from src.analytics.exit_intelligence.observation import (
    ExitObservationStore,
    ExitPathSnapshotStore,
    ExecutionExitObservationStore,
    PortfolioExitPressureStore,
    build_exit_path_snapshot,
)


def _make_obs(symbol="1234", entry="2026-01-01", exit_date="2026-01-10") -> ExitObservationRecord:
    return ExitObservationRecord(
        obs_id=ExitObservationRecord.make_obs_id(symbol, entry, exit_date),
        symbol=symbol,
        entry_date=entry,
        exit_date=exit_date,
        holding_days=9,
        entry_price=1000.0,
        exit_price=1100.0,
        realized_return=0.10,
        realized_pnl_jpy=10000.0,
        exit_reason="turtle_exit_55d",
        unrealized_pnl_at_exit=10000.0,
        distance_from_peak_pct=0.05,
        rs_rank_at_exit=80.0,
        atr_pct_at_exit=0.02,
        breadth_score_at_exit=0.65,
        regime_at_exit="bull",
        exit_velocity_score=2.5,
        path_snapshot_count=9,
        generated_at="2026-01-10T09:00:00+09:00",
    )


def _make_path_snap(symbol="1234", entry="2026-01-01", day=1, close=1010.0):
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
        peak_price_so_far=max(close, 1010.0),
        distance_from_peak_pct=max(0, (max(close, 1010.0) - close) / max(close, 1010.0)),
        rs_rank=78.0,
        atr_pct=0.02,
        breadth_score=0.6,
        regime="bull",
        trend_persistence=0.7,
        momentum=0.02,
        momentum_prev=0.01,
        volume_ratio=1.1,
        volatility_compression=0.6,
        upper_shadow_ratio=0.3,
        exit_signal_count=0,
        deterioration_velocity=0.002,
    )


class TestExitObservationStore:
    def test_append_and_load(self, tmp_path):
        store = ExitObservationStore(tmp_path / "obs.jsonl")
        obs = _make_obs()
        store.append(obs)
        records = store.load_all()
        assert len(records) == 1
        assert records[0].obs_id == obs.obs_id
        assert records[0].symbol == "1234"

    def test_multiple_appends(self, tmp_path):
        store = ExitObservationStore(tmp_path / "obs.jsonl")
        for i in range(3):
            obs = _make_obs(symbol=f"100{i}", entry="2026-01-01",
                            exit_date=f"2026-01-{10+i}")
            store.append(obs)
        records = store.load_all()
        assert len(records) == 3

    def test_load_by_symbol(self, tmp_path):
        store = ExitObservationStore(tmp_path / "obs.jsonl")
        store.append(_make_obs("1111"))
        store.append(_make_obs("2222"))
        store.append(_make_obs("1111", exit_date="2026-02-01"))
        assert len(store.load_by_symbol("1111")) == 2
        assert len(store.load_by_symbol("2222")) == 1

    def test_empty_file_returns_empty_list(self, tmp_path):
        store = ExitObservationStore(tmp_path / "nonexistent.jsonl")
        assert store.load_all() == []

    def test_obs_id_determinism(self):
        id1 = ExitObservationRecord.make_obs_id("1234", "2026-01-01", "2026-01-10")
        id2 = ExitObservationRecord.make_obs_id("1234", "2026-01-01", "2026-01-10")
        assert id1 == id2

    def test_obs_id_unique_by_inputs(self):
        id1 = ExitObservationRecord.make_obs_id("1234", "2026-01-01", "2026-01-10")
        id2 = ExitObservationRecord.make_obs_id("1234", "2026-01-01", "2026-01-11")
        assert id1 != id2

    def test_generated_at_auto_filled(self, tmp_path):
        store = ExitObservationStore(tmp_path / "obs.jsonl")
        obs = _make_obs()
        obs.generated_at = ""
        store.append(obs)
        loaded = store.load_all()
        assert loaded[0].generated_at != ""

    def test_schema_version_preserved(self, tmp_path):
        store = ExitObservationStore(tmp_path / "obs.jsonl")
        store.append(_make_obs())
        assert store.load_all()[0].schema_version == SCHEMA_VERSION


class TestExitPathSnapshotStore:
    def test_append_and_load(self, tmp_path):
        store = ExitPathSnapshotStore(tmp_path / "paths.jsonl")
        snap = _make_path_snap(day=1)
        store.append(snap)
        records = store.load_all()
        assert len(records) == 1

    def test_load_by_symbol_sorted(self, tmp_path):
        store = ExitPathSnapshotStore(tmp_path / "paths.jsonl")
        for day in [3, 1, 2]:
            store.append(_make_path_snap(day=day))
        loaded = store.load_by_symbol("1234")
        days = [r.day_number for r in loaded]
        assert days == sorted(days)

    def test_load_open_position(self, tmp_path):
        store = ExitPathSnapshotStore(tmp_path / "paths.jsonl")
        for d in range(1, 6):
            store.append(_make_path_snap(day=d))
        # Different entry date
        s2 = _make_path_snap(symbol="1234", entry="2026-02-01", day=1)
        store.append(s2)
        open_pos = store.load_open_position("1234", "2026-01-01")
        assert len(open_pos) == 5
        assert all(s.entry_date == "2026-01-01" for s in open_pos)

    def test_snapshot_id_determinism(self):
        id1 = ExitPathSnapshot.make_snapshot_id("1234", "2026-01-01", "2026-01-05")
        id2 = ExitPathSnapshot.make_snapshot_id("1234", "2026-01-01", "2026-01-05")
        assert id1 == id2


class TestPortfolioExitPressureStore:
    def test_append_load_latest(self, tmp_path):
        store = PortfolioExitPressureStore(tmp_path / "pressure.jsonl")
        for d in range(1, 4):
            snap = PortfolioExitPressureSnapshot(
                pressure_id=PortfolioExitPressureSnapshot.make_pressure_id(f"2026-01-0{d}"),
                snapshot_date=f"2026-01-0{d}",
                concurrent_positions=2,
                gross_exposure_jpy=2_000_000.0,
                sector_concentration_max=0.5,
                signal_queue_pressure=0.3,
                replacement_pressure=0.2,
                portfolio_heat=0.05,
                cash_pressure=0.3,
                capital_utilization=0.7,
                avg_holding_days=5.0,
                worst_position_deterioration=0.08,
            )
            store.append(snap)
        latest = store.load_latest()
        assert latest is not None
        assert latest.snapshot_date == "2026-01-03"

    def test_empty_returns_none(self, tmp_path):
        store = PortfolioExitPressureStore(tmp_path / "pressure.jsonl")
        assert store.load_latest() is None


class TestBuildExitPathSnapshot:
    def test_basic(self):
        snap = build_exit_path_snapshot(
            symbol="1234", entry_date="2026-01-01",
            snapshot_date="2026-01-05", day_number=5,
            close_price=1050.0, entry_price=1000.0,
            peak_price_so_far=1060.0,
            rs_rank=80.0, atr_pct=0.02,
            breadth_score=0.6, regime="bull",
            trend_persistence=0.7,
            momentum=0.03, momentum_prev=0.02,
            volume_ratio=1.2, volatility_compression=0.5,
            upper_shadow_ratio=0.3, exit_signal_count=0,
        )
        assert snap.symbol == "1234"
        assert snap.unrealized_pnl_fraction == pytest.approx(0.05, rel=1e-5)
        assert snap.distance_from_peak_pct == pytest.approx(
            (1060.0 - 1050.0) / 1060.0, rel=1e-5
        )

    def test_negative_pnl(self):
        snap = build_exit_path_snapshot(
            symbol="1234", entry_date="2026-01-01",
            snapshot_date="2026-01-05", day_number=5,
            close_price=950.0, entry_price=1000.0,
            peak_price_so_far=1000.0,
            rs_rank=40.0, atr_pct=0.03,
            breadth_score=0.35, regime="bear",
            trend_persistence=0.3,
            momentum=-0.05, momentum_prev=-0.02,
            volume_ratio=0.8, volatility_compression=0.3,
            upper_shadow_ratio=0.7, exit_signal_count=2,
        )
        assert snap.unrealized_pnl_fraction < 0
        assert snap.distance_from_peak_pct >= 0
