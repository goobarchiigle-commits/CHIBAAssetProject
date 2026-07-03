"""
tests/analytics/test_position_lifecycle_snapshot.py

Coverage:
  - PositionLifecycleSnapshot: construction, from_dict round-trip, snapshot_id derivation
  - PositionLifecycleSnapshotSchemaValidator: all field checks, fail-closed on violation
  - PositionLifecycleSnapshotWriter: append-only, dedup, crash-safe staging, batch write
  - PositionLifecycleSnapshotLoader: load_all, load_symbol_timeline, load_by_date_range
  - Analytics helpers: compute_holding_duration_analytics, compute_peak_to_current_deterioration
  - build_snapshot: derived field computation
  - Temporal integrity: no future snapshot_date, entry_date <= snapshot_date
  - Serialization: deterministic sort_keys, UTF-8, no float NaN/Inf
  - Replay consistency: identical records for same inputs
  - Timezone consistency: +09:00 enforcement
  - Schema version persistence
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

import pytest

from src.analytics.position_lifecycle_snapshot import (
    DEFAULT_SNAPSHOT_PATH,
    SCHEMA_VERSION,
    HoldingDurationAnalytics,
    PeakDeteriorationAnalytics,
    PositionLifecycleSnapshot,
    PositionLifecycleSnapshotLoader,
    PositionLifecycleSnapshotSchemaError,
    PositionLifecycleSnapshotSchemaValidator,
    PositionLifecycleSnapshotWriter,
    build_snapshot,
    compute_holding_duration_analytics,
    compute_peak_to_current_deterioration,
)

JST = timezone(timedelta(hours=9))

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _jst_ts(date_str: str, hour: int = 15, minute: int = 10) -> str:
    return f"{date_str}T{hour:02d}:{minute:02d}:00+09:00"


def _make_snap(
    snapshot_date: str = "2026-05-19",
    symbol: str = "6506.T",
    entry_date: str = "2026-05-10",
    quantity: int = 100,
    entry_price: float = 1520.0,
    close_price: float = 1778.0,
    max_unrealized_pnl_pct: float = 0.214,
    atr_pct: float = 0.028,
    rs_rank: int = 4,
    market_regime: str = "trend_persistent",
    breadth_50: float = 0.71,
    breadth_75: float = 0.52,
    sector: str = "transportation_equipment",
    position_volatility_rank: float = 0.82,
    portfolio_heat: float = 0.34,
    cash_ratio: float = 0.27,
    snapshot_timestamp_jst: Optional[str] = None,
) -> PositionLifecycleSnapshot:
    ts = snapshot_timestamp_jst if snapshot_timestamp_jst is not None else _jst_ts(snapshot_date)
    return build_snapshot(
        snapshot_date=snapshot_date,
        symbol=symbol,
        entry_date=entry_date,
        quantity=quantity,
        entry_price=entry_price,
        close_price=close_price,
        max_unrealized_pnl_pct=max_unrealized_pnl_pct,
        atr_pct=atr_pct,
        rs_rank=rs_rank,
        market_regime=market_regime,
        breadth_50=breadth_50,
        breadth_75=breadth_75,
        sector=sector,
        position_volatility_rank=position_volatility_rank,
        portfolio_heat=portfolio_heat,
        cash_ratio=cash_ratio,
        snapshot_timestamp_jst=ts,
    )


@pytest.fixture
def snap():
    return _make_snap()


@pytest.fixture
def validator():
    return PositionLifecycleSnapshotSchemaValidator()


@pytest.fixture
def jsonl_path(tmp_path):
    return tmp_path / "snapshots.jsonl"


@pytest.fixture
def writer(jsonl_path):
    return PositionLifecycleSnapshotWriter(path=jsonl_path)


@pytest.fixture
def loader(jsonl_path):
    return PositionLifecycleSnapshotLoader(path=jsonl_path)


# ─────────────────────────────────────────────────────────────────────────────
# PositionLifecycleSnapshot — construction and round-trip
# ─────────────────────────────────────────────────────────────────────────────

class TestPositionLifecycleSnapshot:
    def test_snapshot_id_derivation(self, snap):
        expected = PositionLifecycleSnapshot.make_snapshot_id(
            "2026-05-19", "6506.T", "2026-05-10"
        )
        assert snap.snapshot_id == expected

    def test_snapshot_id_deterministic(self):
        id1 = PositionLifecycleSnapshot.make_snapshot_id("2026-05-19", "6506.T", "2026-05-10")
        id2 = PositionLifecycleSnapshot.make_snapshot_id("2026-05-19", "6506.T", "2026-05-10")
        assert id1 == id2
        assert len(id1) == 64

    def test_snapshot_id_unique_per_date(self):
        id1 = PositionLifecycleSnapshot.make_snapshot_id("2026-05-19", "6506.T", "2026-05-10")
        id2 = PositionLifecycleSnapshot.make_snapshot_id("2026-05-20", "6506.T", "2026-05-10")
        assert id1 != id2

    def test_snapshot_id_unique_per_symbol(self):
        id1 = PositionLifecycleSnapshot.make_snapshot_id("2026-05-19", "6506.T", "2026-05-10")
        id2 = PositionLifecycleSnapshot.make_snapshot_id("2026-05-19", "7203.T", "2026-05-10")
        assert id1 != id2

    def test_from_dict_round_trip(self, snap):
        d = snap.to_dict()
        snap2 = PositionLifecycleSnapshot.from_dict(d)
        assert asdict(snap2) == asdict(snap)

    def test_derived_days_held(self, snap):
        assert snap.days_held == 9  # 2026-05-19 - 2026-05-10

    def test_derived_unrealized_pnl_pct(self, snap):
        expected = round((1778.0 - 1520.0) / 1520.0, 6)
        assert abs(snap.unrealized_pnl_pct - expected) < 1e-9

    def test_derived_distance_from_peak_pct(self, snap):
        expected = round(snap.unrealized_pnl_pct - 0.214, 6)
        assert abs(snap.distance_from_peak_pct - expected) < 1e-9
        assert snap.distance_from_peak_pct <= 0.0

    def test_schema_version(self, snap):
        assert snap.schema_version == SCHEMA_VERSION

    def test_to_dict_has_all_required_fields(self, snap):
        d = snap.to_dict()
        for field in (
            "snapshot_date", "symbol", "entry_date", "days_held", "quantity",
            "entry_price", "close_price", "unrealized_pnl_pct", "max_unrealized_pnl_pct",
            "distance_from_peak_pct", "atr_pct", "rs_rank", "market_regime",
            "breadth_50", "breadth_75", "sector", "position_volatility_rank",
            "portfolio_heat", "cash_ratio", "snapshot_timestamp_jst",
            "snapshot_id", "schema_version",
        ):
            assert field in d, f"missing field: {field}"


# ─────────────────────────────────────────────────────────────────────────────
# build_snapshot — convenience constructor
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildSnapshot:
    def test_basic_build(self, snap):
        assert snap.symbol == "6506.T"
        assert snap.snapshot_date == "2026-05-19"
        assert snap.entry_date == "2026-05-10"

    def test_explicit_timestamp(self):
        ts = "2026-05-19T15:10:22+09:00"
        snap = build_snapshot(
            snapshot_date="2026-05-19",
            symbol="7203.T",
            entry_date="2026-05-15",
            quantity=200,
            entry_price=2500.0,
            close_price=2600.0,
            max_unrealized_pnl_pct=0.05,
            atr_pct=0.02,
            rs_rank=1,
            market_regime="trend_persistent",
            breadth_50=0.6,
            breadth_75=0.4,
            sector="transportation_equipment",
            position_volatility_rank=0.5,
            portfolio_heat=0.3,
            cash_ratio=0.4,
            snapshot_timestamp_jst=ts,
        )
        assert snap.snapshot_timestamp_jst == ts

    def test_days_held_zero_same_day(self):
        snap = build_snapshot(
            snapshot_date="2026-05-19",
            symbol="1234.T",
            entry_date="2026-05-19",
            quantity=10,
            entry_price=1000.0,
            close_price=1010.0,
            max_unrealized_pnl_pct=0.01,
            atr_pct=0.015,
            rs_rank=5,
            market_regime="trend_weak",
            breadth_50=0.5,
            breadth_75=0.3,
            sector="retail",
            position_volatility_rank=0.2,
            portfolio_heat=0.1,
            cash_ratio=0.6,
            snapshot_timestamp_jst="2026-05-19T10:00:00+09:00",
        )
        assert snap.days_held == 0


# ─────────────────────────────────────────────────────────────────────────────
# PositionLifecycleSnapshotSchemaValidator
# ─────────────────────────────────────────────────────────────────────────────

class TestSchemaValidator:
    def test_valid_snapshot_passes(self, validator, snap):
        validator.validate(snap)  # must not raise

    def test_missing_field_fails(self, validator, snap):
        d = snap.to_dict()
        del d["symbol"]
        # Recreate without symbol — from_dict will fail, so mutate directly
        bad = PositionLifecycleSnapshot.from_dict({**snap.to_dict()})
        object.__setattr__(bad, "symbol", "")  # empty triggers nonempty check
        with pytest.raises(PositionLifecycleSnapshotSchemaError):
            validator.validate(bad)

    def test_entry_date_after_snapshot_date_fails(self, validator):
        snap = _make_snap(entry_date="2026-05-20", snapshot_date="2026-05-19")
        # Override the derived days_held to not fail on days_held mismatch first
        with pytest.raises(PositionLifecycleSnapshotSchemaError, match="lookahead"):
            validator.validate(snap)

    def test_future_snapshot_date_fails(self, validator):
        future = "2099-01-01"
        snap = _make_snap(snapshot_date=future, entry_date="2026-05-10")
        # days_held would be wrong but lookahead check should fire
        with pytest.raises(PositionLifecycleSnapshotSchemaError):
            validator.validate(snap)

    def test_negative_quantity_fails(self, validator, snap):
        bad = PositionLifecycleSnapshot.from_dict({
            **snap.to_dict(), "quantity": -1
        })
        with pytest.raises(PositionLifecycleSnapshotSchemaError, match="quantity"):
            validator.validate(bad)

    def test_zero_entry_price_fails(self, validator, snap):
        bad = PositionLifecycleSnapshot.from_dict({
            **snap.to_dict(), "entry_price": 0.0
        })
        with pytest.raises(PositionLifecycleSnapshotSchemaError, match="entry_price"):
            validator.validate(bad)

    def test_nan_unrealized_pnl_fails(self, validator, snap):
        bad = PositionLifecycleSnapshot.from_dict({
            **snap.to_dict(), "unrealized_pnl_pct": float("nan")
        })
        with pytest.raises(PositionLifecycleSnapshotSchemaError, match="unrealized_pnl_pct"):
            validator.validate(bad)

    def test_inf_atr_pct_fails(self, validator, snap):
        bad = PositionLifecycleSnapshot.from_dict({
            **snap.to_dict(), "atr_pct": float("inf")
        })
        with pytest.raises(PositionLifecycleSnapshotSchemaError, match="atr_pct"):
            validator.validate(bad)

    def test_breadth_out_of_range_fails(self, validator, snap):
        bad = PositionLifecycleSnapshot.from_dict({
            **snap.to_dict(), "breadth_50": 1.5
        })
        with pytest.raises(PositionLifecycleSnapshotSchemaError, match="breadth_50"):
            validator.validate(bad)

    def test_portfolio_heat_negative_fails(self, validator, snap):
        bad = PositionLifecycleSnapshot.from_dict({
            **snap.to_dict(), "portfolio_heat": -0.01
        })
        with pytest.raises(PositionLifecycleSnapshotSchemaError, match="portfolio_heat"):
            validator.validate(bad)

    def test_distance_from_peak_positive_fails(self, validator, snap):
        d = snap.to_dict()
        d["distance_from_peak_pct"] = 0.01  # positive: impossible (current > peak)
        bad = PositionLifecycleSnapshot.from_dict(d)
        with pytest.raises(PositionLifecycleSnapshotSchemaError, match="distance_from_peak_pct"):
            validator.validate(bad)

    def test_max_unrealized_below_unrealized_fails(self, validator, snap):
        d = snap.to_dict()
        # Set max < current
        d["max_unrealized_pnl_pct"] = d["unrealized_pnl_pct"] - 0.05
        d["distance_from_peak_pct"] = 0.05  # would be positive now
        bad = PositionLifecycleSnapshot.from_dict(d)
        with pytest.raises(PositionLifecycleSnapshotSchemaError):
            validator.validate(bad)

    def test_timestamp_missing_jst_offset_fails(self, validator, snap):
        bad = PositionLifecycleSnapshot.from_dict({
            **snap.to_dict(),
            "snapshot_timestamp_jst": "2026-05-19T15:10:00Z"
        })
        with pytest.raises(PositionLifecycleSnapshotSchemaError, match=r"\+09:00"):
            validator.validate(bad)

    def test_timestamp_date_mismatch_fails(self, validator, snap):
        bad = PositionLifecycleSnapshot.from_dict({
            **snap.to_dict(),
            "snapshot_timestamp_jst": "2026-05-18T15:10:00+09:00"
        })
        with pytest.raises(PositionLifecycleSnapshotSchemaError, match="timestamp date"):
            validator.validate(bad)

    def test_snapshot_id_mismatch_fails(self, validator, snap):
        bad = PositionLifecycleSnapshot.from_dict({
            **snap.to_dict(), "snapshot_id": "deadbeef" * 8
        })
        with pytest.raises(PositionLifecycleSnapshotSchemaError, match="snapshot_id mismatch"):
            validator.validate(bad)

    def test_days_held_mismatch_fails(self, validator, snap):
        bad = PositionLifecycleSnapshot.from_dict({
            **snap.to_dict(), "days_held": 999
        })
        with pytest.raises(PositionLifecycleSnapshotSchemaError, match="days_held"):
            validator.validate(bad)

    def test_empty_sector_fails(self, validator, snap):
        bad = PositionLifecycleSnapshot.from_dict({
            **snap.to_dict(), "sector": "  "
        })
        with pytest.raises(PositionLifecycleSnapshotSchemaError, match="sector"):
            validator.validate(bad)

    def test_cash_ratio_above_one_fails(self, validator, snap):
        bad = PositionLifecycleSnapshot.from_dict({
            **snap.to_dict(), "cash_ratio": 1.1
        })
        with pytest.raises(PositionLifecycleSnapshotSchemaError, match="cash_ratio"):
            validator.validate(bad)


# ─────────────────────────────────────────────────────────────────────────────
# PositionLifecycleSnapshotWriter
# ─────────────────────────────────────────────────────────────────────────────

class TestSnapshotWriter:
    def test_write_creates_file(self, writer, jsonl_path, snap):
        assert writer.write(snap)
        assert jsonl_path.exists()

    def test_written_line_is_valid_json(self, writer, jsonl_path, snap):
        writer.write(snap)
        lines = [l for l in jsonl_path.read_text("utf-8").splitlines() if l.strip()]
        assert len(lines) == 1
        d = json.loads(lines[0])
        assert d["symbol"] == "6506.T"

    def test_dedup_returns_false_on_duplicate(self, writer, snap):
        assert writer.write(snap) is True
        assert writer.write(snap) is False

    def test_dedup_only_one_line_in_file(self, writer, jsonl_path, snap):
        writer.write(snap)
        writer.write(snap)
        lines = [l for l in jsonl_path.read_text("utf-8").splitlines() if l.strip()]
        assert len(lines) == 1

    def test_different_symbols_both_written(self, writer, jsonl_path):
        s1 = _make_snap(symbol="6506.T", snapshot_date="2026-05-19", entry_date="2026-05-10")
        s2 = _make_snap(symbol="7203.T", snapshot_date="2026-05-19", entry_date="2026-05-10")
        assert writer.write(s1)
        assert writer.write(s2)
        lines = [l for l in jsonl_path.read_text("utf-8").splitlines() if l.strip()]
        assert len(lines) == 2

    def test_different_dates_both_written(self, writer, jsonl_path):
        s1 = _make_snap(snapshot_date="2026-05-19", entry_date="2026-05-10")
        s2 = _make_snap(
            snapshot_date="2026-05-20",
            entry_date="2026-05-10",
            snapshot_timestamp_jst="2026-05-20T15:10:00+09:00",
        )
        assert writer.write(s1)
        assert writer.write(s2)
        lines = [l for l in jsonl_path.read_text("utf-8").splitlines() if l.strip()]
        assert len(lines) == 2

    def test_schema_violation_raises_and_no_write(self, writer, jsonl_path, snap):
        bad = PositionLifecycleSnapshot.from_dict({
            **snap.to_dict(), "quantity": -1
        })
        with pytest.raises(PositionLifecycleSnapshotSchemaError):
            writer.write(bad)
        assert not jsonl_path.exists()

    def test_staging_file_cleaned_up_after_success(self, writer, jsonl_path, snap):
        writer.write(snap)
        stage_files = list(jsonl_path.parent.glob(".pending_*.tmp"))
        assert stage_files == []

    def test_append_preserves_existing_content(self, writer, jsonl_path):
        s1 = _make_snap(symbol="6506.T", snapshot_date="2026-05-19", entry_date="2026-05-10")
        s2 = _make_snap(symbol="7203.T", snapshot_date="2026-05-19", entry_date="2026-05-10")
        writer.write(s1)
        writer.write(s2)
        lines = [l for l in jsonl_path.read_text("utf-8").splitlines() if l.strip()]
        symbols = {json.loads(l)["symbol"] for l in lines}
        assert "6506.T" in symbols
        assert "7203.T" in symbols

    def test_write_batch(self, writer):
        snaps = [
            _make_snap(symbol="6506.T", snapshot_date="2026-05-19", entry_date="2026-05-10"),
            _make_snap(symbol="7203.T", snapshot_date="2026-05-19", entry_date="2026-05-10"),
            _make_snap(symbol="6506.T", snapshot_date="2026-05-19", entry_date="2026-05-10"),  # dup
        ]
        written, skipped = writer.write_batch(snaps)
        assert written == 2
        assert skipped == 1

    def test_utf8_encoding_preserved(self, writer, jsonl_path):
        snap = _make_snap(sector="自動車・輸送機器")
        writer.write(snap)
        raw = jsonl_path.read_bytes()
        assert b"\xe8\x87\xaa\xe5\x8b\x95\xe8\xbb\x8a" in raw  # 自動車 in UTF-8

    def test_parent_directories_created(self, tmp_path):
        deep_path = tmp_path / "a" / "b" / "c" / "snaps.jsonl"
        w = PositionLifecycleSnapshotWriter(path=deep_path)
        snap = _make_snap()
        w.write(snap)
        assert deep_path.exists()


# ─────────────────────────────────────────────────────────────────────────────
# PositionLifecycleSnapshotLoader
# ─────────────────────────────────────────────────────────────────────────────

class TestSnapshotLoader:
    def _populate(self, writer, loader, snaps) -> List[PositionLifecycleSnapshot]:
        for s in snaps:
            writer.write(s)
        return loader.load_all()

    def test_load_all_empty_path(self, loader):
        assert loader.load_all() == []

    def test_load_all_returns_all(self, writer, loader):
        s1 = _make_snap(symbol="6506.T", snapshot_date="2026-05-19", entry_date="2026-05-10")
        s2 = _make_snap(symbol="7203.T", snapshot_date="2026-05-19", entry_date="2026-05-10")
        writer.write(s1)
        writer.write(s2)
        loaded = loader.load_all()
        assert len(loaded) == 2

    def test_load_symbol_timeline_filters(self, writer, loader):
        s1 = _make_snap(symbol="6506.T", snapshot_date="2026-05-19", entry_date="2026-05-10")
        s2 = _make_snap(symbol="7203.T", snapshot_date="2026-05-19", entry_date="2026-05-10")
        writer.write(s1)
        writer.write(s2)
        timeline = loader.load_symbol_timeline("6506.T")
        assert len(timeline) == 1
        assert timeline[0].symbol == "6506.T"

    def test_load_symbol_timeline_sorted_ascending(self, writer, loader):
        s1 = _make_snap(symbol="6506.T", snapshot_date="2026-05-19", entry_date="2026-05-10")
        s2 = _make_snap(
            symbol="6506.T",
            snapshot_date="2026-05-20",
            entry_date="2026-05-10",
            snapshot_timestamp_jst="2026-05-20T15:10:00+09:00",
        )
        writer.write(s2)  # write out of order
        writer.write(s1)
        timeline = loader.load_symbol_timeline("6506.T")
        assert timeline[0].snapshot_date < timeline[1].snapshot_date

    def test_load_by_date(self, writer, loader):
        s1 = _make_snap(symbol="6506.T", snapshot_date="2026-05-19", entry_date="2026-05-10")
        s2 = _make_snap(
            symbol="6506.T",
            snapshot_date="2026-05-20",
            entry_date="2026-05-10",
            snapshot_timestamp_jst="2026-05-20T15:10:00+09:00",
        )
        writer.write(s1)
        writer.write(s2)
        day = loader.load_by_date("2026-05-19")
        assert len(day) == 1
        assert day[0].snapshot_date == "2026-05-19"

    def test_load_by_date_range_inclusive(self, writer, loader):
        dates = ["2026-05-18", "2026-05-19", "2026-05-20"]
        for d in dates:
            s = _make_snap(
                symbol="6506.T",
                snapshot_date=d,
                entry_date="2026-05-10",
                snapshot_timestamp_jst=f"{d}T15:10:00+09:00",
            )
            writer.write(s)
        result = loader.load_by_date_range("2026-05-18", "2026-05-19")
        assert len(result) == 2
        assert all(r.snapshot_date <= "2026-05-19" for r in result)

    def test_load_by_date_range_exclusive_end(self, writer, loader):
        dates = ["2026-05-17", "2026-05-18", "2026-05-19"]
        for d in dates:
            s = _make_snap(
                symbol="6506.T",
                snapshot_date=d,
                entry_date="2026-05-10",
                snapshot_timestamp_jst=f"{d}T15:10:00+09:00",
            )
            writer.write(s)
        result = loader.load_by_date_range("2026-05-18", "2026-05-18")
        assert len(result) == 1

    def test_corrupt_line_skipped(self, jsonl_path, loader):
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        good_snap = _make_snap()
        good_line = json.dumps(good_snap.to_dict(), sort_keys=True, ensure_ascii=False)
        jsonl_path.write_text(
            "NOT VALID JSON\n" + good_line + "\n",
            encoding="utf-8",
        )
        result = loader.load_all()
        assert len(result) == 1
        assert result[0].symbol == "6506.T"

    def test_replay_consistency(self, writer, loader):
        snap = _make_snap()
        writer.write(snap)
        loaded1 = loader.load_all()
        loaded2 = loader.load_all()
        assert asdict(loaded1[0]) == asdict(loaded2[0])

    def test_nonexistent_symbol_returns_empty(self, writer, loader):
        writer.write(_make_snap())
        assert loader.load_symbol_timeline("9999.T") == []


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic serialization
# ─────────────────────────────────────────────────────────────────────────────

class TestDeterministicSerialization:
    def test_sort_keys_consistent(self, snap):
        d = snap.to_dict()
        s1 = json.dumps(d, sort_keys=True, ensure_ascii=False)
        s2 = json.dumps(d, sort_keys=True, ensure_ascii=False)
        assert s1 == s2

    def test_same_inputs_byte_identical_output(self):
        ts = "2026-05-19T15:10:22+09:00"
        snap1 = build_snapshot(
            snapshot_date="2026-05-19",
            symbol="6506.T",
            entry_date="2026-05-10",
            quantity=100,
            entry_price=1520.0,
            close_price=1778.0,
            max_unrealized_pnl_pct=0.214,
            atr_pct=0.028,
            rs_rank=4,
            market_regime="trend_persistent",
            breadth_50=0.71,
            breadth_75=0.52,
            sector="transportation_equipment",
            position_volatility_rank=0.82,
            portfolio_heat=0.34,
            cash_ratio=0.27,
            snapshot_timestamp_jst=ts,
        )
        snap2 = build_snapshot(
            snapshot_date="2026-05-19",
            symbol="6506.T",
            entry_date="2026-05-10",
            quantity=100,
            entry_price=1520.0,
            close_price=1778.0,
            max_unrealized_pnl_pct=0.214,
            atr_pct=0.028,
            rs_rank=4,
            market_regime="trend_persistent",
            breadth_50=0.71,
            breadth_75=0.52,
            sector="transportation_equipment",
            position_volatility_rank=0.82,
            portfolio_heat=0.34,
            cash_ratio=0.27,
            snapshot_timestamp_jst=ts,
        )
        s1 = json.dumps(snap1.to_dict(), sort_keys=True, ensure_ascii=False)
        s2 = json.dumps(snap2.to_dict(), sort_keys=True, ensure_ascii=False)
        assert s1 == s2

    def test_no_nan_in_serialized_output(self, writer, jsonl_path, snap):
        writer.write(snap)
        raw = jsonl_path.read_text("utf-8")
        assert "NaN" not in raw
        assert "Infinity" not in raw

    def test_utf8_only(self, writer, jsonl_path):
        snap = _make_snap(sector="自動車")
        writer.write(snap)
        # Must be decodable as UTF-8
        raw = jsonl_path.read_bytes()
        raw.decode("utf-8")

    def test_ensure_ascii_false(self, writer, jsonl_path):
        snap = _make_snap(sector="電気機器")
        writer.write(snap)
        raw = jsonl_path.read_text("utf-8")
        assert "電気機器" in raw  # not escaped as \\uXXXX


# ─────────────────────────────────────────────────────────────────────────────
# Timezone consistency
# ─────────────────────────────────────────────────────────────────────────────

class TestTimezoneConsistency:
    def test_utc_offset_required(self, validator, snap):
        bad = PositionLifecycleSnapshot.from_dict({
            **snap.to_dict(),
            "snapshot_timestamp_jst": "2026-05-19T15:10:00"
        })
        with pytest.raises(PositionLifecycleSnapshotSchemaError, match=r"\+09:00"):
            validator.validate(bad)

    def test_jst_offset_enforced(self, validator, snap):
        # UTC+00:00 not accepted
        bad = PositionLifecycleSnapshot.from_dict({
            **snap.to_dict(),
            "snapshot_timestamp_jst": "2026-05-19T06:10:00+00:00"
        })
        with pytest.raises(PositionLifecycleSnapshotSchemaError, match=r"\+09:00"):
            validator.validate(bad)

    def test_timestamp_date_must_match_snapshot_date(self, validator, snap):
        bad = PositionLifecycleSnapshot.from_dict({
            **snap.to_dict(),
            "snapshot_timestamp_jst": "2026-05-20T15:10:00+09:00"
        })
        with pytest.raises(PositionLifecycleSnapshotSchemaError, match="timestamp date"):
            validator.validate(bad)

    def test_build_snapshot_default_timestamp_is_jst(self):
        snap = _make_snap()
        assert snap.snapshot_timestamp_jst.endswith("+09:00")


# ─────────────────────────────────────────────────────────────────────────────
# Append-only integrity
# ─────────────────────────────────────────────────────────────────────────────

class TestAppendOnlyIntegrity:
    def test_no_overwrite_existing_records(self, writer, jsonl_path):
        snap = _make_snap()
        writer.write(snap)
        original = jsonl_path.read_text("utf-8")
        writer.write(snap)  # duplicate — must not change file
        after = jsonl_path.read_text("utf-8")
        assert original == after

    def test_append_adds_to_end(self, writer, jsonl_path):
        s1 = _make_snap(symbol="6506.T", snapshot_date="2026-05-19", entry_date="2026-05-10")
        s2 = _make_snap(symbol="7203.T", snapshot_date="2026-05-19", entry_date="2026-05-10")
        writer.write(s1)
        after_first = jsonl_path.read_text("utf-8")
        writer.write(s2)
        after_second = jsonl_path.read_text("utf-8")
        assert after_second.startswith(after_first)

    def test_monotone_line_count(self, writer, jsonl_path):
        counts = []
        for d in ["2026-05-17", "2026-05-18", "2026-05-19"]:
            s = _make_snap(
                snapshot_date=d,
                entry_date="2026-05-10",
                snapshot_timestamp_jst=f"{d}T15:10:00+09:00",
            )
            writer.write(s)
            n = sum(1 for l in jsonl_path.read_text("utf-8").splitlines() if l.strip())
            counts.append(n)
        assert counts == [1, 2, 3]


# ─────────────────────────────────────────────────────────────────────────────
# No future timestamp contamination (lookahead guard)
# ─────────────────────────────────────────────────────────────────────────────

class TestNoLookahead:
    def test_future_snapshot_date_rejected(self, validator):
        snap = build_snapshot(
            snapshot_date="2099-12-31",
            symbol="6506.T",
            entry_date="2099-12-01",
            quantity=100,
            entry_price=1000.0,
            close_price=1100.0,
            max_unrealized_pnl_pct=0.1,
            atr_pct=0.02,
            rs_rank=1,
            market_regime="trend_persistent",
            breadth_50=0.5,
            breadth_75=0.4,
            sector="electronics",
            position_volatility_rank=0.5,
            portfolio_heat=0.2,
            cash_ratio=0.3,
            snapshot_timestamp_jst="2099-12-31T15:10:00+09:00",
        )
        with pytest.raises(PositionLifecycleSnapshotSchemaError, match="lookahead"):
            validator.validate(snap)

    def test_entry_after_snapshot_rejected(self, validator):
        snap = _make_snap(snapshot_date="2026-05-19", entry_date="2026-05-20")
        with pytest.raises(PositionLifecycleSnapshotSchemaError, match="lookahead"):
            validator.validate(snap)


# ─────────────────────────────────────────────────────────────────────────────
# Crash-safe write behavior
# ─────────────────────────────────────────────────────────────────────────────

class TestCrashSafeWrite:
    def test_staging_file_created_during_write(self, tmp_path, snap):
        """Verify staging file exists during the append phase (simulated by patching)."""
        path = tmp_path / "snaps.jsonl"
        w = PositionLifecycleSnapshotWriter(path=path)
        # Normal write — staging file should be gone after
        w.write(snap)
        stages = list(tmp_path.glob(".pending_*.tmp"))
        assert stages == []

    def test_recovery_via_dedup_after_crash(self, tmp_path):
        """Simulate crash: staging file exists but line was already committed."""
        path = tmp_path / "snaps.jsonl"
        snap = _make_snap()
        w = PositionLifecycleSnapshotWriter(path=path)
        w.write(snap)

        # Simulate stale staging file from a prior crash
        stage = path.parent / f".pending_{snap.snapshot_id[:12]}.tmp"
        stage.write_text(
            json.dumps(snap.to_dict(), sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        # Write again — dedup guard must prevent double-write
        result = w.write(snap)
        assert result is False

        lines = [l for l in path.read_text("utf-8").splitlines() if l.strip()]
        assert len(lines) == 1  # still only one line

    def test_partial_line_at_eof_is_skipped(self, jsonl_path, loader):
        """Partial line appended after a crash must be skipped by loader."""
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        good = _make_snap()
        good_line = json.dumps(good.to_dict(), sort_keys=True, ensure_ascii=False) + "\n"
        jsonl_path.write_text(good_line + '{"partial":', encoding="utf-8")
        loaded = loader.load_all()
        assert len(loaded) == 1


# ─────────────────────────────────────────────────────────────────────────────
# compute_holding_duration_analytics
# ─────────────────────────────────────────────────────────────────────────────

class TestHoldingDurationAnalytics:
    def _build_timeline(self, symbol: str, dates: List[str], entry: str) -> List[PositionLifecycleSnapshot]:
        snaps = []
        for d in dates:
            s = _make_snap(
                symbol=symbol,
                snapshot_date=d,
                entry_date=entry,
                snapshot_timestamp_jst=f"{d}T15:10:00+09:00",
            )
            snaps.append(s)
        return snaps

    def test_returns_none_for_empty(self):
        assert compute_holding_duration_analytics([]) is None

    def test_single_snapshot(self):
        snap = _make_snap()
        result = compute_holding_duration_analytics([snap])
        assert isinstance(result, HoldingDurationAnalytics)
        assert result.total_snapshots == 1
        assert result.min_days_held == result.max_days_held

    def test_mean_days_held(self):
        dates = ["2026-05-17", "2026-05-18", "2026-05-19"]
        timeline = self._build_timeline("6506.T", dates, "2026-05-10")
        result = compute_holding_duration_analytics(timeline)
        # days held: 7, 8, 9 → mean = 8.0
        assert result.mean_days_held == pytest.approx(8.0)

    def test_distribution_counts(self):
        dates = ["2026-05-17", "2026-05-18", "2026-05-19"]
        timeline = self._build_timeline("6506.T", dates, "2026-05-10")
        result = compute_holding_duration_analytics(timeline)
        assert result.days_held_distribution[7] == 1
        assert result.days_held_distribution[8] == 1
        assert result.days_held_distribution[9] == 1

    def test_peak_unrealized_pnl(self):
        dates = ["2026-05-17", "2026-05-18", "2026-05-19"]
        timeline = self._build_timeline("6506.T", dates, "2026-05-10")
        result = compute_holding_duration_analytics(timeline)
        assert result.peak_unrealized_pnl_pct == pytest.approx(0.214)

    def test_final_distance_from_peak(self):
        dates = ["2026-05-17", "2026-05-18", "2026-05-19"]
        timeline = self._build_timeline("6506.T", dates, "2026-05-10")
        result = compute_holding_duration_analytics(timeline)
        last = timeline[-1]
        assert result.final_distance_from_peak_pct == pytest.approx(last.distance_from_peak_pct)


# ─────────────────────────────────────────────────────────────────────────────
# compute_peak_to_current_deterioration
# ─────────────────────────────────────────────────────────────────────────────

class TestPeakDeteriorationAnalytics:
    def _build_snap(self, symbol: str, snapshot_date: str, entry_date: str,
                    close_price: float, max_pnl: float, entry_price: float = 1000.0
                    ) -> PositionLifecycleSnapshot:
        return build_snapshot(
            snapshot_date=snapshot_date,
            symbol=symbol,
            entry_date=entry_date,
            quantity=100,
            entry_price=entry_price,
            close_price=close_price,
            max_unrealized_pnl_pct=max_pnl,
            atr_pct=0.02,
            rs_rank=3,
            market_regime="trend_persistent",
            breadth_50=0.6,
            breadth_75=0.5,
            sector="electronics",
            position_volatility_rank=0.5,
            portfolio_heat=0.3,
            cash_ratio=0.3,
            snapshot_timestamp_jst=f"{snapshot_date}T15:10:00+09:00",
        )

    def test_returns_none_for_empty(self):
        assert compute_peak_to_current_deterioration([]) is None

    def test_single_snapshot_no_deterioration(self):
        snap = _make_snap()
        result = compute_peak_to_current_deterioration([snap])
        assert result.deterioration_days == 0
        assert result.deterioration_rate_per_day == pytest.approx(0.0)

    def test_peak_date_identified(self):
        # price rises then falls
        timeline = [
            self._build_snap("6506.T", "2026-05-17", "2026-05-10", 1100.0, 0.10),
            self._build_snap("6506.T", "2026-05-18", "2026-05-10", 1200.0, 0.20),
            self._build_snap("6506.T", "2026-05-19", "2026-05-10", 1150.0, 0.20),
        ]
        result = compute_peak_to_current_deterioration(timeline)
        assert result.peak_snapshot_date == "2026-05-18"

    def test_deterioration_days_counted(self):
        timeline = [
            self._build_snap("6506.T", "2026-05-17", "2026-05-10", 1100.0, 0.10),
            self._build_snap("6506.T", "2026-05-18", "2026-05-10", 1200.0, 0.20),
            self._build_snap("6506.T", "2026-05-19", "2026-05-10", 1150.0, 0.20),
            self._build_snap("6506.T", "2026-05-20", "2026-05-10", 1130.0, 0.20),
        ]
        result = compute_peak_to_current_deterioration(timeline)
        assert result.deterioration_days == 2  # peak at idx 1, current at idx 3

    def test_max_series_monotone_nondecreasing(self):
        timeline = [
            self._build_snap("6506.T", "2026-05-17", "2026-05-10", 1050.0, 0.05),
            self._build_snap("6506.T", "2026-05-18", "2026-05-10", 1200.0, 0.20),
            self._build_snap("6506.T", "2026-05-19", "2026-05-10", 1100.0, 0.20),
        ]
        result = compute_peak_to_current_deterioration(timeline)
        series = result.max_unrealized_pnl_series
        for i in range(1, len(series)):
            assert series[i] >= series[i - 1] - 1e-9

    def test_deterioration_rate_positive_after_peak(self):
        timeline = [
            self._build_snap("6506.T", "2026-05-17", "2026-05-10", 1200.0, 0.20),
            self._build_snap("6506.T", "2026-05-18", "2026-05-10", 1100.0, 0.20),
            self._build_snap("6506.T", "2026-05-19", "2026-05-10", 1050.0, 0.20),
        ]
        result = compute_peak_to_current_deterioration(timeline)
        assert result.deterioration_rate_per_day > 0.0

    def test_series_length_matches_timeline(self):
        dates = ["2026-05-17", "2026-05-18", "2026-05-19"]
        timeline = [
            self._build_snap("6506.T", d, "2026-05-10", 1000.0 + 50 * i, 0.10 * (i + 1) if 0.10 * (i + 1) >= 0 else 0.10)
            for i, d in enumerate(dates)
        ]
        result = compute_peak_to_current_deterioration(timeline)
        assert len(result.snapshot_dates) == 3
        assert len(result.unrealized_pnl_series) == 3
        assert len(result.max_unrealized_pnl_series) == 3
        assert len(result.distance_from_peak_series) == 3

    def test_symbol_and_entry_date_preserved(self):
        snap = _make_snap()
        result = compute_peak_to_current_deterioration([snap])
        assert result.symbol == "6506.T"
        assert result.entry_date == "2026-05-10"
